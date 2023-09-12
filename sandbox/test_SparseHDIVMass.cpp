/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Kokkos_Core.hpp"

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_HDIV_HEX_In_FEM.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Utils.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"

#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_DOFManager.hpp"

#include "kokkosTools.hpp"

using namespace std;
using Teuchos::RCP;
using Teuchos::rcp;

int main(int argc, char * argv[]) {
  
  TEUCHOS_TEST_FOR_EXCEPTION(argc==1,std::runtime_error,"Error: this test requires a mesh file");
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<Teuchos::MpiComm<int>> Comm = Teuchos::rcp( new Teuchos::MpiComm<int>(MPI_COMM_WORLD) );
  
  Kokkos::initialize();
  
  typedef Teuchos::RCP<const shards::CellTopology> topo_RCP;
  typedef Teuchos::RCP<Intrepid2::Basis<PHX::Device::execution_space, double, double > > basis_RCP;

  {
    
    // ==========================================================
    // Create a mesh from the file defined by the user
    // ==========================================================

    std::string input_file_name = argv[1];
    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
    
    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory());
    pl->set("File Name",input_file_name);
    
    mesh_factory->setParameterList(pl);
    Teuchos::RCP<panzer_stk::STK_Interface> mesh = mesh_factory->buildUncommitedMesh(*(Comm->getRawMpiComm()));
    
    mesh_factory->completeMeshConstruction(*mesh,*(Comm->getRawMpiComm()));
    if (Comm->getRank() == 0) {
      mesh->printMetaData(std::cout);
    }
    
    std::vector<string> blocknames;
    mesh->getElementBlockNames(blocknames);
    
    // ==========================================================
    // Build a Panzer DOF manager
    // ==========================================================
    
    int degree = 2; // hard coding for the moment
    int dim = mesh->getDimension();

    Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
    Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
    DOF->setConnManager(conn,*(Comm->getRawMpiComm()));
    DOF->setOrientationsRequired(true);

    topo_RCP cell_topology = mesh->getCellTopology(blocknames[0]);

    // Just adding basis functions on the first block for now  
    basis_RCP basis = Teuchos::rcp(new Intrepid2::Basis_HDIV_HEX_In_FEM<PHX::Device::execution_space,double,double>(degree,Intrepid2::POINTTYPE_EQUISPACED) );
    Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
    DOF->addField(blocknames[0], "T", Pattern, panzer::FieldType::CG);
    
    DOF->buildGlobalUnknowns();
      
    if (Comm->getRank() == 0) {
      DOF->printFieldInformation(std::cout);
      std::cout << "================================================" << std::endl << std::endl;  
    }

    // ==========================================================
    // Get basis size and nominal quadrature order.
    // ==========================================================

    const Intrepid2::ordinal_type basis_size = basis->getCardinality();
    const unsigned int quad_order = 2*degree;

    Kokkos::DynRankView<double,PHX::Device> dofCoords("dofCoords", basis_size, dim);
    basis->getDofCoords(dofCoords);
    //MrHyDE::KokkosTools::print(dofCoords);

    /*
    auto dofordinal = basis->getAllDofTags();
    //MrHyDE::KokkosTools::print(dofordinal);
    for (size_type i=0; i<dofordinal.extent(0); ++i) {
      std::cout << i+1 << " ";
      for (size_type j=0; j<dofordinal.extent(1); ++j) {
        //for (size_type k=0; k<dofordinal.extent(2); ++k) {
          std::cout << dofordinal(i,j) << " ";
        //}
      }
      std::cout << std::endl;
    }
    */

    Intrepid2::DefaultCubatureFactory cubature_factory;

    vector<int> offsets = DOF->getGIDFieldOffsets(blocknames[0],0);

    double mwt = 1.0;//
    Kokkos::View<int*,PHX::Device> off("offsets in view",offsets.size());
    auto off_host = Kokkos::create_mirror_view(off);
    for (size_t i=0; i<offsets.size(); ++i) {
      off_host(i) = offsets[i];
    }
    Kokkos::deep_copy(off,off_host);

    std::vector<stk::mesh::Entity> elems;
    mesh->getMyElements(blocknames[0], elems);

    // Vector of five mass matrices, built from the following combinations
    // of integration rules:
    //
    //   mass[0] - GLEG, GLEG, GLEG
    //   mass[1] - GLEG, GLOB, GLOB
    //   mass[2] - GLOB, GLEG, GLOB
    //   mass[3] - GLOB, GLOB, GLEG
    //   mass[4] - GLOB, GLOB, GLOB
    //
    // where GLEG stands for Gauss-Legendre and GLOB stands for Gauss-Lobatto.
    std::vector<Kokkos::View<double***,PHX::Device>> mass;

    { // START SCOPE 1: Tensor product of Gauss-Legendre rules (Intrepid default for QUAD and HEX).

      Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double> > basis_cubature =
        cubature_factory.create<PHX::Device::execution_space, double, double>(*cell_topology, quad_order);
      const int cubature_dim  = basis_cubature->getDimension();
      const int num_cubature_points = basis_cubature->getNumPoints();

      Kokkos::DynRankView<double,PHX::Device> ref_ip("reference integration points", num_cubature_points, cubature_dim);
      Kokkos::DynRankView<double,PHX::Device> ref_wts("reference weights", num_cubature_points);
      basis_cubature->getCubature(ref_ip, ref_wts);

      Kokkos::DynRankView<double,PHX::Device> ref_basis_vals("reference basis values", basis_size, ref_ip.extent(0), dim);
      basis->getValues(ref_basis_vals, ref_ip, Intrepid2::OPERATOR_VALUE);

      // ==========================================================
      // Map to physical frame
      // ==========================================================

      const int num_nodes_per_elem = cell_topology->getNodeCount();
      const size_t num_elems = elems.size();
      const unsigned int num_ip = ref_ip.extent(0);

      // TODO: I'm very much butchering my devices here and this will probably break cuda builds with tests enabled
      // Setup for discretization: jacobians, physical points, weights, and values
      Kokkos::DynRankView<double,PHX::Device> nodes("nodes", num_elems, num_nodes_per_elem, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian("jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_inv("inverse of jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_det("determinant of jacobian", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> phys_ip("physical ip", num_elems, num_ip, dim);
      Kokkos::DynRankView<double,PHX::Device> wts("physical weights", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> basis_vals("physical basis vals", num_elems, basis_size, num_ip, dim);

      mesh->getElementVertices(elems, blocknames[0], nodes);

      Intrepid2::CellTools<PHX::Device::execution_space>::mapToPhysicalFrame(phys_ip, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobian(jacobian, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianDet(jacobian_det, jacobian);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianInv(jacobian_inv, jacobian);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::computeCellMeasure(wts, jacobian_det, ref_wts);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::HCURLtransformVALUE(basis_vals, jacobian_inv, ref_basis_vals);
      // Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
      // Kokkos::DynRankView<double,PHX::Device> basis_vals2("basis transformed and oriented", num_elems, basis_size, num_ip, dim);
      // OrientTools::modifyBasisByOrientation(basis_vals2, basis_vals, orientation, basis);

      // ==========================================================
      // Build the mass matrix
      // ==========================================================

      Kokkos::View<double***,PHX::Device> newmass("local mass",num_elems, basis_size, basis_size);

      Kokkos::parallel_for("testSparseMass construct mass",
                           Kokkos::RangePolicy<PHX::Device::execution_space>(0,num_elems),
                           KOKKOS_LAMBDA (const int elem ) {
        for (auto i=0; i<basis_vals.extent(1); i++ ) {
          for (auto j=0; j<basis_vals.extent(1); j++ ) {
            for (auto pt=0; pt<basis_vals.extent(2); pt++ ) {
              for (auto dim=0; dim<basis_vals.extent(3); dim++ ) {
                //newmass(elem,off(i),off(j)) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
                newmass(elem,i,j) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
              }
            }
          }
        }
      });
      mass.push_back(newmass);
    } // END SCOPE 1: Tensor product of Gauss-Legendre rules (Intrepid default for QUAD and HEX).

    { // START SCOPE 2: Tensor product of Gauss-Legendre, Gauss-Lobatto and Gauss-Lobatto rules.

      const auto line_cubature_x = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS_LOBATTO);
      const auto line_cubature_y = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS);
      const auto line_cubature_z = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS);
      Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double>> basis_cubature =
        Teuchos::rcp(new Intrepid2::CubatureTensor<PHX::Device::execution_space, double, double>(line_cubature_x, line_cubature_y, line_cubature_z));
      const int cubature_dim  = basis_cubature->getDimension();
      const int num_cubature_points = basis_cubature->getNumPoints();

      Kokkos::DynRankView<double,PHX::Device> ref_ip("reference integration points", num_cubature_points, cubature_dim);
      Kokkos::DynRankView<double,PHX::Device> ref_wts("reference weights", num_cubature_points);
      basis_cubature->getCubature(ref_ip, ref_wts);

      Kokkos::DynRankView<double,PHX::Device> ref_basis_vals("reference basis values", basis_size, ref_ip.extent(0), dim);
      basis->getValues(ref_basis_vals, ref_ip, Intrepid2::OPERATOR_VALUE);

      //MrHyDE::KokkosTools::print(ref_ip);
      //MrHyDE::KokkosTools::print(ref_basis_vals);

      // ==========================================================
      // Map to physical frame
      // ==========================================================

      const int num_nodes_per_elem = cell_topology->getNodeCount();
      const size_t num_elems = elems.size();
      const unsigned int num_ip = ref_ip.extent(0);

      // TODO: I'm very much butchering my devices here and this will probably break cuda builds with tests enabled
      // Setup for discretization: jacobians, physical points, weights, and values
      Kokkos::DynRankView<double,PHX::Device> nodes("nodes", num_elems, num_nodes_per_elem, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian("jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_inv("inverse of jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_det("determinant of jacobian", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> phys_ip("physical ip", num_elems, num_ip, dim);
      Kokkos::DynRankView<double,PHX::Device> wts("physical weights", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> basis_vals("physical basis vals", num_elems, basis_size, num_ip, dim);

      mesh->getElementVertices(elems, blocknames[0], nodes);

      Intrepid2::CellTools<PHX::Device::execution_space>::mapToPhysicalFrame(phys_ip, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobian(jacobian, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianDet(jacobian_det, jacobian);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianInv(jacobian_inv, jacobian);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::computeCellMeasure(wts, jacobian_det, ref_wts);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::HCURLtransformVALUE(basis_vals, jacobian_inv, ref_basis_vals);
      // Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
      // Kokkos::DynRankView<double,PHX::Device> basis_vals2("basis transformed and oriented", num_elems, basis_size, num_ip, dim);
      // OrientTools::modifyBasisByOrientation(basis_vals2, basis_vals, orientation, basis);

      // ==========================================================
      // Build the mass matrix
      // ==========================================================

      Kokkos::View<double***,PHX::Device> newmass("local mass",num_elems, basis_size, basis_size);
      
      Kokkos::parallel_for("testSparseMass construct mass",
                           Kokkos::RangePolicy<PHX::Device::execution_space>(0,num_elems),
                           KOKKOS_LAMBDA (const int elem ) {
        for (auto i=0; i<basis_vals.extent(1); i++ ) {
          for (auto j=0; j<basis_vals.extent(1); j++ ) {
            for (auto pt=0; pt<basis_vals.extent(2); pt++ ) {
              for (auto dim=0; dim<basis_vals.extent(3); dim++ ) {
                //newmass(elem,off(i),off(j)) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
                newmass(elem,i,j) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
              }
            }
          }
        }
      });
      mass.push_back(newmass);
    } // END SCOPE 2: Tensor product of Gauss-Legendre, Gauss-Lobatto and Gauss-Lobatto rules.


    { // START SCOPE 3: Tensor product of Gauss-Lobatto, Gauss-Legendre and Gauss-Lobatto rules.

      const auto line_cubature_x = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS);
      const auto line_cubature_y = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS_LOBATTO);
      const auto line_cubature_z = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS);
      Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double>> basis_cubature =
        Teuchos::rcp(new Intrepid2::CubatureTensor<PHX::Device::execution_space, double, double>(line_cubature_x, line_cubature_y, line_cubature_z));
      const int cubature_dim  = basis_cubature->getDimension();
      const int num_cubature_points = basis_cubature->getNumPoints();

      Kokkos::DynRankView<double,PHX::Device> ref_ip("reference integration points", num_cubature_points, cubature_dim);
      Kokkos::DynRankView<double,PHX::Device> ref_wts("reference weights", num_cubature_points);
      basis_cubature->getCubature(ref_ip, ref_wts);

      Kokkos::DynRankView<double,PHX::Device> ref_basis_vals("reference basis values", basis_size, ref_ip.extent(0), dim);
      basis->getValues(ref_basis_vals, ref_ip, Intrepid2::OPERATOR_VALUE);

      //MrHyDE::KokkosTools::print(ref_ip);

      // ==========================================================
      // Map to physical frame
      // ==========================================================

      const int num_nodes_per_elem = cell_topology->getNodeCount();
      const size_t num_elems = elems.size();
      const unsigned int num_ip = ref_ip.extent(0);

      // TODO: I'm very much butchering my devices here and this will probably break cuda builds with tests enabled
      // Setup for discretization: jacobians, physical points, weights, and values
      Kokkos::DynRankView<double,PHX::Device> nodes("nodes", num_elems, num_nodes_per_elem, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian("jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_inv("inverse of jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_det("determinant of jacobian", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> phys_ip("physical ip", num_elems, num_ip, dim);
      Kokkos::DynRankView<double,PHX::Device> wts("physical weights", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> basis_vals("physical basis vals", num_elems, basis_size, num_ip, dim);

      mesh->getElementVertices(elems, blocknames[0], nodes);

      Intrepid2::CellTools<PHX::Device::execution_space>::mapToPhysicalFrame(phys_ip, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobian(jacobian, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianDet(jacobian_det, jacobian);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianInv(jacobian_inv, jacobian);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::computeCellMeasure(wts, jacobian_det, ref_wts);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::HCURLtransformVALUE(basis_vals, jacobian_inv, ref_basis_vals);
      // Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
      // Kokkos::DynRankView<double,PHX::Device> basis_vals2("basis transformed and oriented", num_elems, basis_size, num_ip, dim);
      // OrientTools::modifyBasisByOrientation(basis_vals2, basis_vals, orientation, basis);

      // ==========================================================
      // Build the mass matrix
      // ==========================================================

      Kokkos::View<double***,PHX::Device> newmass("local mass",num_elems, basis_size, basis_size);

      Kokkos::parallel_for("testSparseMass construct mass",
                           Kokkos::RangePolicy<PHX::Device::execution_space>(0,num_elems),
                           KOKKOS_LAMBDA (const int elem ) {
        for (auto i=0; i<basis_vals.extent(1); i++ ) {
          for (auto j=0; j<basis_vals.extent(1); j++ ) {
            for (auto pt=0; pt<basis_vals.extent(2); pt++ ) {
              for (auto dim=0; dim<basis_vals.extent(3); dim++ ) {
                //newmass(elem,off(i),off(j)) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
                newmass(elem,i,j) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
              }
            }
          }
        }
      });
      mass.push_back(newmass);
    } // END SCOPE 3: Tensor product of Gauss-Lobatto, Gauss-Legendre and Gauss-Lobatto rules.


    { // START SCOPE 4: Tensor product of Gauss-Lobatto, Gauss-Lobatto and Gauss-Legendre rules.

      const auto line_cubature_x = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS);
      const auto line_cubature_y = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS);
      const auto line_cubature_z = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS_LOBATTO);
      Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double>> basis_cubature =
        Teuchos::rcp(new Intrepid2::CubatureTensor<PHX::Device::execution_space, double, double>(line_cubature_x, line_cubature_y, line_cubature_z));
      const int cubature_dim  = basis_cubature->getDimension();
      const int num_cubature_points = basis_cubature->getNumPoints();

      Kokkos::DynRankView<double,PHX::Device> ref_ip("reference integration points", num_cubature_points, cubature_dim);
      Kokkos::DynRankView<double,PHX::Device> ref_wts("reference weights", num_cubature_points);
      basis_cubature->getCubature(ref_ip, ref_wts);

      Kokkos::DynRankView<double,PHX::Device> ref_basis_vals("reference basis values", basis_size, ref_ip.extent(0), dim);
      basis->getValues(ref_basis_vals, ref_ip, Intrepid2::OPERATOR_VALUE);

      //MrHyDE::KokkosTools::print(ref_ip);

      // ==========================================================
      // Map to physical frame
      // ==========================================================

      const int num_nodes_per_elem = cell_topology->getNodeCount();
      const size_t num_elems = elems.size();
      const unsigned int num_ip = ref_ip.extent(0);

      // TODO: I'm very much butchering my devices here and this will probably break cuda builds with tests enabled
      // Setup for discretization: jacobians, physical points, weights, and values
      Kokkos::DynRankView<double,PHX::Device> nodes("nodes", num_elems, num_nodes_per_elem, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian("jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_inv("inverse of jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_det("determinant of jacobian", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> phys_ip("physical ip", num_elems, num_ip, dim);
      Kokkos::DynRankView<double,PHX::Device> wts("physical weights", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> basis_vals("physical basis vals", num_elems, basis_size, num_ip, dim);

      mesh->getElementVertices(elems, blocknames[0], nodes);

      Intrepid2::CellTools<PHX::Device::execution_space>::mapToPhysicalFrame(phys_ip, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobian(jacobian, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianDet(jacobian_det, jacobian);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianInv(jacobian_inv, jacobian);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::computeCellMeasure(wts, jacobian_det, ref_wts);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::HCURLtransformVALUE(basis_vals, jacobian_inv, ref_basis_vals);
      // Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
      // Kokkos::DynRankView<double,PHX::Device> basis_vals2("basis transformed and oriented", num_elems, basis_size, num_ip, dim);
      // OrientTools::modifyBasisByOrientation(basis_vals2, basis_vals, orientation, basis);

      // ==========================================================
      // Build the mass matrix
      // ==========================================================

      Kokkos::View<double***,PHX::Device> newmass("local mass",num_elems, basis_size, basis_size);

      Kokkos::parallel_for("testSparseMass construct mass",
                           Kokkos::RangePolicy<PHX::Device::execution_space>(0,num_elems),
                           KOKKOS_LAMBDA (const int elem ) {
        for (auto i=0; i<basis_vals.extent(1); i++ ) {
          for (auto j=0; j<basis_vals.extent(1); j++ ) {
            for (auto pt=0; pt<basis_vals.extent(2); pt++ ) {
              for (auto dim=0; dim<basis_vals.extent(3); dim++ ) {
                //newmass(elem,off(i),off(j)) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
                newmass(elem,i,j) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
              }
            }
          }
        }
      });
      mass.push_back(newmass);
    } // END SCOPE 4: Tensor product of Gauss-Lobatto, Gauss-Lobatto and Gauss-Legendre rules.


    { // START SCOPE 5: Tensor product of Gauss-Lobatto, Gauss-Lobatto and Gauss-Lobatto rules.

      const auto line_cubature_x = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS_LOBATTO);
      const auto line_cubature_y = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS_LOBATTO);
      const auto line_cubature_z = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quad_order-1, Intrepid2::POLYTYPE_GAUSS_LOBATTO);
      Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double>> basis_cubature =
        Teuchos::rcp(new Intrepid2::CubatureTensor<PHX::Device::execution_space, double, double>(line_cubature_x, line_cubature_y, line_cubature_z));
      const int cubature_dim  = basis_cubature->getDimension();
      const int num_cubature_points = basis_cubature->getNumPoints();

      Kokkos::DynRankView<double,PHX::Device> ref_ip("reference integration points", num_cubature_points, cubature_dim);
      Kokkos::DynRankView<double,PHX::Device> ref_wts("reference weights", num_cubature_points);
      basis_cubature->getCubature(ref_ip, ref_wts);

      Kokkos::DynRankView<double,PHX::Device> ref_basis_vals("reference basis values", basis_size, ref_ip.extent(0), dim);
      basis->getValues(ref_basis_vals, ref_ip, Intrepid2::OPERATOR_VALUE);

      //MrHyDE::KokkosTools::print(ref_ip);
      //MrHyDE::KokkosTools::print(ref_basis_vals);

      // ==========================================================
      // Map to physical frame
      // ==========================================================

      const int num_nodes_per_elem = cell_topology->getNodeCount();
      const size_t num_elems = elems.size();
      const unsigned int num_ip = ref_ip.extent(0);

      // TODO: I'm very much butchering my devices here and this will probably break cuda builds with tests enabled
      // Setup for discretization: jacobians, physical points, weights, and values
      Kokkos::DynRankView<double,PHX::Device> nodes("nodes", num_elems, num_nodes_per_elem, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian("jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_inv("inverse of jacobian", num_elems, num_ip, dim, dim);
      Kokkos::DynRankView<double,PHX::Device> jacobian_det("determinant of jacobian", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> phys_ip("physical ip", num_elems, num_ip, dim);
      Kokkos::DynRankView<double,PHX::Device> wts("physical weights", num_elems, num_ip);
      Kokkos::DynRankView<double,PHX::Device> basis_vals("physical basis vals", num_elems, basis_size, num_ip, dim);

      mesh->getElementVertices(elems, blocknames[0], nodes);

      Intrepid2::CellTools<PHX::Device::execution_space>::mapToPhysicalFrame(phys_ip, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobian(jacobian, ref_ip, nodes, *cell_topology);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianDet(jacobian_det, jacobian);
      Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianInv(jacobian_inv, jacobian);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::computeCellMeasure(wts, jacobian_det, ref_wts);

      Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::HCURLtransformVALUE(basis_vals, jacobian_inv, ref_basis_vals);
      // Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
      // Kokkos::DynRankView<double,PHX::Device> basis_vals2("basis transformed and oriented", num_elems, basis_size, num_ip, dim);
      // OrientTools::modifyBasisByOrientation(basis_vals2, basis_vals, orientation, basis);

      // ==========================================================
      // Build the mass matrix
      // ==========================================================

      Kokkos::View<double***,PHX::Device> newmass("local mass",num_elems, basis_size, basis_size);

      Kokkos::parallel_for("testSparseMass construct mass",
                           Kokkos::RangePolicy<PHX::Device::execution_space>(0,num_elems),
                           KOKKOS_LAMBDA (const int elem ) {
        for (auto i=0; i<basis_vals.extent(1); i++ ) {
          for (auto j=0; j<basis_vals.extent(1); j++ ) {
            for (auto pt=0; pt<basis_vals.extent(2); pt++ ) {
              for (auto dim=0; dim<basis_vals.extent(3); dim++ ) {
                //newmass(elem,off(i),off(j)) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
                newmass(elem,i,j) += basis_vals(elem,i,pt,dim)*basis_vals(elem,j,pt,dim)*wts(elem,pt)*mwt;
              }
            }
          }
        }
      });
      mass.push_back(newmass);

    } // END SCOPE 5: Tensor product of Gauss-Lobatto, Gauss-Lobatto and Gauss-Lobatto rules.
    
    MrHyDE::KokkosTools::printToFile(mass[0],"mass.HDIV.GGG.txt");
    MrHyDE::KokkosTools::printToFile(mass[1],"mass.HDIV.GBB.txt");
    MrHyDE::KokkosTools::printToFile(mass[2],"mass.HDIV.BGB.txt");
    MrHyDE::KokkosTools::printToFile(mass[3],"mass.HDIV.BBG.txt");
    MrHyDE::KokkosTools::printToFile(mass[4],"mass.HDIV.BBB.txt");

    {
      const size_t num_elems = mass[0].extent(0);
      //const size_t basis_size = mass[0].extent(1);
      
      Kokkos::View<double***,PHX::Device> newmass("local mass",num_elems, basis_size, basis_size);
      auto BBB_mass = mass[4];
      Kokkos::parallel_for("testSparseMass construct mass",
                           Kokkos::RangePolicy<PHX::Device::execution_space>(0,num_elems),
                           KOKKOS_LAMBDA (const int elem ) {
        for (auto i=0; i<newmass.extent(1); i++ ) {
          for (auto j=0; j<newmass.extent(1); j++ ) {
            newmass(elem,off(i),off(j))  = BBB_mass(elem,i,j);
          }
        }
      });

      // xx component from GBB mass
      // yy component from BGB mass
      // zz component from BBG mass

      auto GBB_mass = mass[1];
      auto BGB_mass = mass[2];
      auto BBG_mass = mass[3];
      Kokkos::parallel_for("testSparseMass construct mass",
                           Kokkos::RangePolicy<PHX::Device::execution_space>(0,num_elems),
                           KOKKOS_LAMBDA (const int elem ) {
        // xx piece
        size_type start = 0;
        size_type stop = newmass.extent(1)/dim;
        for (auto i=start; i<stop; i++ ) {
          for (auto j=start; j<stop; j++ ) {
            newmass(elem,off(i),off(j))  = GBB_mass(elem,i,j);
          }
        }
 
        if (dim > 1) {
          // yy piece
          size_type start = newmass.extent(1)/dim;
          size_type stop = 2*newmass.extent(1)/dim;
          for (auto i=start; i<stop; i++ ) {
            for (auto j=start; j<stop; j++ ) {
              newmass(elem,off(i),off(j))  = BGB_mass(elem,i,j);
            }
          }
        }

        if (dim > 2) {
          // zz piece
          size_type start = 2*newmass.extent(1)/dim;
          size_type stop = 3*newmass.extent(1)/dim;
          for (auto i=start; i<stop; i++ ) {
            for (auto j=start; j<stop; j++ ) {
              newmass(elem,off(i),off(j))  = BBG_mass(elem,i,j);
            }
          }
        }
      });

      mass.push_back(newmass);
    }
    MrHyDE::KokkosTools::printToFile(mass[5],"mass.HDIV.SPARSE.txt");

    // ==========================================================
    // Assess the sparsity
    // ==========================================================


  }

  Kokkos::finalize();


  int val = 0;
  return val;
}


