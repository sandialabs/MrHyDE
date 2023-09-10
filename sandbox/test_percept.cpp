/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "trilinos.hpp"
#include "preferences.hpp"
#include "subgridMeshFactory.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  //MpiComm Comm(MPI_COMM_WORLD);
  Teuchos::RCP<MpiComm> Comm = Teuchos::rcp( new MpiComm(MPI_COMM_WORLD) );
  
  Kokkos::initialize();

  {
    Kokkos::View<ScalarT**,HostDevice> nodes("nodes",4,2);
    nodes(0,0) = -1.0;
    nodes(0,1) = -1.0;
    nodes(1,0) = 1.0;
    nodes(1,1) = -1.0;
    nodes(2,0) = 1.0;
    nodes(2,1) = 1.0;
    nodes(3,0) = -1.0;
    nodes(3,1) = 1.0;
    
    vector<vector<GO> > connectivity;
    vector<GO> e0 = {0,1,2,3};
    
    string shape = "quad";
    string blockID = "eb";
    
    panzer_stk::SubGridMeshFactory meshFactory(shape, nodes, connectivity, blockID);
    
    Teuchos::RCP<panzer_stk::STK_Interface> mesh = meshFactory.buildMesh(MPI_COMM_WORLD);
    //Teuchos::RCP<panzer_stk::STK_Interface> mesh = meshFactory.buildMesh(Comm);
    
    meshFactory.completeMeshConstruction(*mesh,MPI_COMM_WORLD);
    
    int numRefine = 2;
    mesh->refineMesh(numRefine, true);
    
  }
  
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


