
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
  
  // Pause for 10s to see how much memory is used
  sleep(10);

  TEUCHOS_TEST_FOR_EXCEPTION(argc==1,std::runtime_error,"Error: this test requires a mesh file");
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<Teuchos::MpiComm<int>> Comm = Teuchos::rcp( new Teuchos::MpiComm<int>(MPI_COMM_WORLD) );

  // Pause for 10s to see how much memory is used
  sleep(10);

  Kokkos::initialize();
  
  // Pause for 10s to see how much memory is used
  sleep(10);

  {
    int numIters = 1;
    if (argc == 3) {
      numIters = atoi(argv[2]);
    }

    // ==========================================================
    // Create a series of meshes from the file defined by the user
    // ==========================================================
    
    for (int iter=0; iter<numIters; ++iter) {
    
      if (Comm->getRank() == 0) {
        std::cout << "PanzerStk Test: Processing mesh " << iter+1 << " out of " << numIters << std::endl;
      }
      
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

    // Pause for 10s to see how much memory is used
    sleep(10);

    /*
    // ==========================================================
    // Create a series of meshes from the file defined by the user
    // ==========================================================
    
    int numGrp = 10000;
    int numElem = 100;
    int numip = 8;
    int numDOF = 8;
    int dim = 3;

    size_t totalcost = sizeof(double)*numGrp*numElem*numip*numDOF*dim;

    std::cout << "Allocating " << static_cast<double>(totalcost)/1.0e6 << " MB in basis data" << std::endl;
    std::vector<Kokkos::View<double****> > basis_vals;
    for (int j=0; j<numGrp; ++j) {
      Kokkos::View<double****> A("basis vals",numElem,numDOF,numip,dim);
      basis_vals.push_back(A);
    //  usleep(10000);

    }
    sleep(10);
    */
   
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


