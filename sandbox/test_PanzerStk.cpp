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
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HCURL_HEX_I1_FEM.hpp"
#include "Intrepid2_HDIV_HEX_I1_FEM.hpp"
#include "Intrepid2_HVOL_C0_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"

#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_LineMeshFactory.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SquareTriMeshFactory.hpp"
#include "Panzer_STK_CubeHexMeshFactory.hpp"
#include "Panzer_STK_CubeTetMeshFactory.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_DOFManager.hpp"

using namespace std;
using Teuchos::RCP;
using Teuchos::rcp;

int main(int argc, char * argv[]) {
  
  //TEUCHOS_TEST_FOR_EXCEPTION(argc==1,std::runtime_error,"Error: this test requires a mesh file");
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<Teuchos::MpiComm<int>> Comm = Teuchos::rcp( new Teuchos::MpiComm<int>(MPI_COMM_WORLD) );

  // Pause for 10s to see how much memory is used
  if (Comm->getRank() == 0) {
    std::cout << "MPI is set up" << std::endl;
  }
    
  sleep(5);

  Kokkos::initialize();
  
  typedef Teuchos::RCP<const shards::CellTopology> topo_RCP;
  typedef Teuchos::RCP<Intrepid2::Basis<PHX::Device::execution_space, double, double > > basis_RCP;
  
  // Pause for 10s to see how much memory is used
  if (Comm->getRank() == 0) {
    std::cout << "Kokkos is set up" << std::endl;
  }
    
  sleep(5);

  {
    int dimension = 2; 
    std::string shape = "quad";
    int NX = 1024, NY = 1024, NZ = 10;
    double xmin = 0.0, ymin = 0.0, zmin = 0.0;
    double xmax = 1.0, ymax = 1.0, zmax = 1.0;

    // ==========================================================
    // Create a series of meshes from the file defined by the user
    // ==========================================================
    { 
      
      RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
      Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
      if (argc > 1) {
        std::string input_file_name = argv[1];
        pl->set("File Name",input_file_name);
        mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory());
      }
      else {
        
        pl->set("X Blocks",1);
        pl->set("X Elements",NX);
        pl->set("X0",xmin);
        pl->set("Xf",xmax);
        if (dimension > 1) {
          pl->set("X Procs", Comm->getSize());
          pl->set("Y Blocks",1);
          pl->set("Y Elements",NY);
          pl->set("Y0",ymin);
          pl->set("Yf",ymax);
          pl->set("Y Procs", 1);
        }
        if (dimension > 2) {
          pl->set("Z Blocks",1);
          pl->set("Z Elements",NZ);
          pl->set("Z0",zmin);
          pl->set("Zf",zmax);
          pl->set("Z Procs", 1);
        }
        if (dimension == 1) {
          mesh_factory = Teuchos::rcp(new panzer_stk::LineMeshFactory());
        }
        else if (dimension == 2) {
          if (shape == "quad") {
            mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory());
          }
          if (shape == "tri") {
            mesh_factory = Teuchos::rcp(new panzer_stk::SquareTriMeshFactory());
          }
        }
        else if (dimension == 3) {
          if (shape == "hex") {
            mesh_factory = Teuchos::rcp(new panzer_stk::CubeHexMeshFactory());
          }
          if (shape == "tet") {
            mesh_factory = Teuchos::rcp(new panzer_stk::CubeTetMeshFactory());
          }
        }
      }
      
      mesh_factory->setParameterList(pl);
      Teuchos::RCP<panzer_stk::STK_Interface> mesh = mesh_factory->buildUncommitedMesh(*(Comm->getRawMpiComm()));
    
      mesh_factory->completeMeshConstruction(*mesh,*(Comm->getRawMpiComm()));
      if (Comm->getRank() == 0) {
        mesh->printMetaData(std::cout);
      }
    
      // Pause for 10s to see how much memory is used
      if (Comm->getRank() == 0) {
        std::cout << "Mesh has been finalized" << std::endl;
      }
      sleep(10);

      std::vector<string> blocknames;
      mesh->getElementBlockNames(blocknames);
    
      // ==========================================================
      // Test out a DOF managers
      // ==========================================================
    
      bool addDOF = true;
      bool buildUnknowns = true;
      int degree = 1;
      if (addDOF) { // all DOF objects are scoped by this flag
        Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
        Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
        DOF->setConnManager(conn,*(Comm->getRawMpiComm()));
        DOF->setOrientationsRequired(true);
      
        for (size_t b=0; b<blocknames.size(); b++) {
          topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
          basis_RCP basis;
          if (dimension == 1) {
            basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_LINE_Cn_FEM<PHX::Device::execution_space,double,double>(degree,Intrepid2::POINTTYPE_EQUISPACED) );
          }
          else if (dimension == 2){
            if (shape == "quad") {
              basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_Cn_FEM<PHX::Device::execution_space,double,double>(degree,Intrepid2::POINTTYPE_EQUISPACED) );
            }
            else {
              basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_TRI_Cn_FEM<PHX::Device::execution_space,double,double>(degree,Intrepid2::POINTTYPE_EQUISPACED) );
            }
          }
          else {
            if (shape == "hex") {
              basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_Cn_FEM<PHX::Device::execution_space,double,double>(degree,Intrepid2::POINTTYPE_EQUISPACED) );
            }
            else {
              basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_TET_Cn_FEM<PHX::Device::execution_space,double,double>(degree,Intrepid2::POINTTYPE_EQUISPACED) );
            }
          }
          Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
          DOF->addField(blocknames[b], "T", Pattern, panzer::FieldType::CG);
        }
        if (buildUnknowns) {
          DOF->buildGlobalUnknowns();
          if (Comm->getRank() == 0) {
            DOF->printFieldInformation(std::cout);
            std::cout << "================================================" << std::endl << std::endl;
          }
        }
        if (Comm->getRank() == 0) {
          std::cout << "DOF manager has been set up" << std::endl;
        }
        sleep(10);

        //DOF = Teuchos::null;
        //conn = Teuchos::null;
        //mesh = Teuchos::null;
        //mesh_factory = Teuchos::null;
        
        if (Comm->getRank() == 0) {
          std::cout << "DOF manager has been destroyed" << std::endl;
        }
        sleep(10);
      }
    }
    if (Comm->getRank() == 0) {
      std::cout << "Mesh has been destroyed" << std::endl;
    }
    sleep(10);
    
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


