
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Kokkos_Core.hpp"

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HCURL_HEX_I1_FEM.hpp"
#include "Intrepid2_HDIV_HEX_I1_FEM.hpp"
#include "Intrepid2_HVOL_C0_FEM.hpp"

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
    
    // ==========================================================
    // Create a series of meshes from the file defined by the user
    // ==========================================================
    {
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
      if (addDOF) { // all DOF objects are scoped by this flag
        Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
        Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
        DOF->setConnManager(conn,*(Comm->getRawMpiComm()));
        DOF->setOrientationsRequired(true);
      
        for (size_t b=0; b<blocknames.size(); b++) {
          topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
          basis_RCP basis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_HEX_C1_FEM<PHX::Device::execution_space,double,double>() );
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


