/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Time.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include "Kokkos_Core.hpp"

#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_LineMeshFactory.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_CubeHexMeshFactory.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_DOFManager.hpp"

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_OrientationTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Utils.hpp"

#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp" // unused
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp" // unused
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp" // unused

#include "exodusII.h"
#include "mpi.h"

#include "databaseView.hpp"

int main(int argc,char * argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);

  Teuchos::RCP<Teuchos::MpiComm<int>> comm = Teuchos::rcp(new typename Teuchos::MpiComm<int>(MPI_COMM_WORLD));
  const int my_rank = comm->getRank();
  const int num_procs = comm->getSize();
  {
    // -----------------------------------------
    // Setup for the problem and mesh
    // -----------------------------------------

    Teuchos::CommandLineProcessor clp(false);
    int dim = 3;
    int nx = 10;
    int ny = 10;
    int nz = 10;
    int degree = 1;
    clp.setOption("dim", &dim, "Spatial dimension to solve the projection problem in (default: 3)");
    clp.setOption("nx", &nx, "Number of elements in x-direction for inline mesh (default: 10)");
    clp.setOption("ny", &ny, "Number of elements in y-direction for inline mesh (default: 10)");
    clp.setOption("nz", &nz, "Number of elements in z-direction for inline mesh (default: 10)");
    clp.setOption("degree", &degree, "Finite element degree to solver the problem with (default: 1)");

    clp.recogniseAllOptions(true);
    switch (clp.parse(argc, argv)) {
      case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED: 
        return EXIT_SUCCESS;
      case Teuchos::CommandLineProcessor::PARSE_ERROR:
      case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION: 
        return EXIT_FAILURE;
      case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
        break;
    }

    if(my_rank == 0) {
      std::cout << "Running databaseView test on an inline mesh with"
                  << " nx=" << nx;
      if(dim>1)
        std::cout << " ny=" << ny;
      if(dim>2)
        std::cout << " nz=" << nz;
      std::cout << std::endl;
    }

    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
    {
      if(dim == 1)
        mesh_factory = Teuchos::rcp(new panzer_stk::LineMeshFactory());
      else if(dim == 2)
        mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory());
      else if(dim == 3)
        mesh_factory = Teuchos::rcp(new panzer_stk::CubeHexMeshFactory());

      Teuchos::RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
      pl->set("X Elements",nx);
      pl->set("X0",0.0);
      pl->set("Xf",1.0);
      if(dim >= 2) {
        pl->set("Y Elements",ny);
        pl->set("Y0",0.0);
        pl->set("Yf",1.0);
      }
      if(dim == 3) {
        pl->set("Z Elements",nz);
        pl->set("Z0",0.0);
        pl->set("Zf",1.0);
      }
      mesh_factory->setParameterList(pl);
    }
    Teuchos::RCP<panzer_stk::STK_Interface> mesh = mesh_factory->buildMesh(*(comm->getRawMpiComm()));
    std::vector<std::string> blocknames;
    mesh->getElementBlockNames(blocknames);


    Teuchos::RCP<panzer_stk::STKConnManager> connection_manager = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
    Teuchos::RCP<panzer::DOFManager> dof_manager = Teuchos::rcp(new panzer::DOFManager());
    dof_manager->setConnManager(connection_manager,*(comm->getRawMpiComm()));
    
    // -----------------------------------------
    // Set the discretization and grab the quadrature and weights
    // -----------------------------------------

    std::vector<std::vector<size_t>> myElements;
    std::string block_name = blocknames[0];
    Teuchos::RCP<const shards::CellTopology> cell_topology = mesh->getCellTopology(block_name);
    Teuchos::RCP<Intrepid2::Basis<PHX::Device>> basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::Device::execution_space,double,double>());
    Teuchos::RCP<const panzer::Intrepid2FieldPattern> pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
    dof_manager->addField(block_name, "p", pattern, panzer::FieldType::CG);
    dof_manager->buildGlobalUnknowns();
    
    if (comm->getRank() == 0) {
      dof_manager->printFieldInformation(std::cout);
    }

    const Intrepid2::ordinal_type basis_size = basis->getCardinality();
    const unsigned int quad_order = 2*1;

    Intrepid2::DefaultCubatureFactory cubature_factory;
    Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double> > basis_cubature  = cubature_factory.create<PHX::Device::execution_space, double, double>(*cell_topology, quad_order);
    const int cubature_dim  = basis_cubature->getDimension();
    const int num_cubature_points = basis_cubature->getNumPoints();
    
    Kokkos::DynRankView<double,PHX::Device> ref_ip("reference integration points", num_cubature_points, cubature_dim);
    Kokkos::DynRankView<double,PHX::Device> ref_weights("reference weights", num_cubature_points);
    basis_cubature->getCubature(ref_ip, ref_weights);

    Kokkos::DynRankView<double,PHX::Device> ref_basis_vals("reference basis values", basis_size, ref_ip.extent(0));
    Kokkos::DynRankView<double,PHX::Device> ref_basis_grad("basis values", basis_size, ref_ip.extent(0), dim);
    basis->getValues(ref_basis_vals, ref_ip, Intrepid2::OPERATOR_VALUE);
    basis->getValues(ref_basis_grad, ref_ip, Intrepid2::OPERATOR_GRAD);
    
    
    // ----- 
    // other preallocations
    // -----
    std::vector<stk::mesh::Entity> elems;
    mesh->getMyElements(block_name, elems);

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
    Kokkos::DynRankView<double,PHX::Device> phys_basis_vals("phys basis vals", num_elems, num_basis, num_ip);
    Kokkos::DynRankView<double,PHX::Device> phys_grad_vals("phys grad vals", num_elems, num_basis, num_ip);

    mesh->getElementVertices(elems, blocknames[0], nodes);

    // create a database to replace all of num_elems Jacobians with a single Jacobian
    Kokkos::View<LO*, AssemblyDevice> database_indices("database indices", num_elems);
    for(size_t i = 0; i < num_elems; ++i)
      database_indices(i) = (LO) 0;
    const int num_unique_elements = 1;
    Kokkos::DynRankView<double,PHX::Device> database_jacobian_dynview("database jacobian", num_unique_elements, num_ip, dim, dim);

    std::cout << "database_jacobian_dynview (" << database_jacobian_dynview.extent(0) << "," << database_jacobian_dynview.extent(1) << "," << database_jacobian_dynview.extent(2) << "," << database_jacobian_dynview.extent(3) << ")" << std::endl;

    Intrepid2::CellTools<PHX::Device::execution_space>::mapToPhysicalFrame(phys_ip, ref_ip, nodes, *cell_topology);
    Intrepid2::CellTools<PHX::Device::execution_space>::setJacobian(database_jacobian_dynview, ref_ip, nodes, *cell_topology);

    MrHyDE::DatabaseView<Kokkos::DynRankView<double,PHX::Device>> database_jacobian(database_jacobian_dynview, database_indices);

    std::cout << "DatabaseView database_jacobian (" << database_jacobian.extent(0) << "," << database_jacobian.extent(1) << "," << database_jacobian.extent(2) << "," << database_jacobian.extent(3) << ")" << std::endl;

    database_jacobian.printMemory();

    // Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianInv(jacobian_inv, database_jacobian);
    // Intrepid2::CellTools<PHX::Device::execution_space>::setJacobianDet(jacobian_det, database_jacobian);
    // Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::computeCellMeasure(phys_weights, jacobian_det, ref_weights);
    // Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::HGRADtransformVALUE(phys_basis_vals, ref_basis_vals);
    // Intrepid2::FunctionSpaceTools<PHX::Device::execution_space>::HGRADtransformGRAD(phys_grad_vals, jacobian_inv, ref_basis_grad);
  }
}