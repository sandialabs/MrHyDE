
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Kokkos_Core.hpp"

#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"

using namespace std;
using Teuchos::RCP;
using Teuchos::rcp;

int main(int argc, char * argv[]) {
  
  TEUCHOS_TEST_FOR_EXCEPTION(argc==1,std::runtime_error,"Error: this test requires a mesh file");
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<Teuchos::MpiComm<int>> Comm = Teuchos::rcp( new Teuchos::MpiComm<int>(MPI_COMM_WORLD) );
  
  Kokkos::initialize();
  
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
    
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


