
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Kokkos_Core.hpp"

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_HCURL_HEX_In_FEM.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"

#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_DOFManager.hpp"

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
    basis_RCP basis = Teuchos::rcp(new Intrepid2::Basis_HCURL_HEX_In_FEM<PHX::Device::execution_space,double,double>(degree,Intrepid2::POINTTYPE_EQUISPACED) );
    Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
    DOF->addField(blocknames[0], "T", Pattern, panzer::FieldType::CG);
    
    DOF->buildGlobalUnknowns();
      
    if (Comm->getRank() == 0) {
      DOF->printFieldInformation(std::cout);
      std::cout << "================================================" << std::endl << std::endl;  
    }

    // ==========================================================
    // Get basis functions at qpts on reference element
    // ==========================================================
     
    const Intrepid2::ordinal_type basis_size = basis->getCardinality();
    const unsigned int quad_order = 2*degree;

    Intrepid2::DefaultCubatureFactory cubature_factory;
    Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double> > basis_cubature  = cubature_factory.create<PHX::Device::execution_space, double, double>(*cell_topology, quad_order);
    const int cubature_dim  = basis_cubature->getDimension();
    const int num_cubature_points = basis_cubature->getNumPoints();
    
    Kokkos::DynRankView<double,PHX::Device> ref_ip("reference integration points", num_cubature_points, cubature_dim);
    Kokkos::DynRankView<double,PHX::Device> ref_weights("reference weights", num_cubature_points);
    basis_cubature->getCubature(ref_ip, ref_weights);

    Kokkos::DynRankView<double,PHX::Device> ref_basis_vals("reference basis values", basis_size, ref_ip.extent(0));
    basis->getValues(ref_basis_vals, ref_ip, Intrepid2::OPERATOR_VALUE);
        
    // ==========================================================
    // Map to physical frame
    // ==========================================================
    
    std::vector<stk::mesh::Entity> elems;
    mesh->getMyElements(blocknames[0], elems);

    const int num_nodes_per_elem = cell_topology->getNodeCount();
    const size_t num_elems = elems.size();
    const unsigned int num_basis = ref_basis_vals.extent(0);
    const unsigned int num_ip = ref_ip.extent(0);

    // TODO: I'm very much butchering my devices here and this will probably break cuda builds with tests enabled
    // Setup for discretization: jacobians, physical points, weights, and values
    Kokkos::DynRankView<double,PHX::Device> nodes("nodes", num_elems, num_nodes_per_elem, dim);
    Kokkos::DynRankView<double,PHX::Device> jacobian("jacobian", num_elems, num_ip, dim, dim);
    Kokkos::DynRankView<double,PHX::Device> jacobian_inv("inverse of jacobian", num_elems, num_ip, dim, dim);
    Kokkos::DynRankView<double,PHX::Device> jacobian_det("determinant of jacobian", num_elems, num_ip);
    Kokkos::DynRankView<double,PHX::Device> phys_ip("phys ip", num_elems, num_ip, dim);
    Kokkos::DynRankView<double,PHX::Device> phys_weights("phys weights", num_elems, num_ip);
    Kokkos::DynRankView<double,PHX::Device> phys_basis_vals("phys basis vals", num_elems, num_basis, num_ip, dim);

    mesh->getElementVertices(elems, blocknames[0], nodes);

    Intrepid2::CellTools<PHX::Device::execution_space>::mapToPhysicalFrame(phys_ip, ref_ip, nodes, *cell_topology);
    //Intrepid2::CellTools<PHX::Device::execution_space>::setJacobian(database_jacobian_dynview, ref_ip, nodes, *cell_topology);

    // ==========================================================
    // Build the mass matrix
    // ==========================================================
    
    /*
    Kokkos::View<Scalar***,PHX::Device> mass("local mass",num_elems, num_basis, num_basis);
    
    vector<int> offsets = DOF->getGIDFieldOffsets(blocknames[0],0);
      
    ScalarT mwt = 1.0;//
    //View_Sc4 cbasis = tbasis[wkset[block]->set_usebasis[set][n]];
    Kokkos::View<int*,PHX::Device> off("offsets in view",offsets.size());
    auto off_host = Kokkos::create_mirror_view(off);
    for (size_t i=0; i<offsets.size(); ++i) {
      off_host(i) = offsets[i];
    }
    Kokkos::deep_copy(off_host,off);

    parallel_for("testSparseMass construct mass",
                 RangePolicy<PHX::Device::execution_space>(0,mass.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type i=0; i<cbasis.extent(1); i++ ) {
        for (size_type j=0; j<cbasis.extent(1); j++ ) {
          for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
            for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
              mass(e,off(i),off(j)) += cbasis(e,i,pt,dim)*cbasis(e,j,pt,dim)*wts(e,pt)*mwt;
            }
          }
        }
      }
    });
    */

    // ==========================================================
    // Assess the sparsity
    // ==========================================================
    

  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


