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
#include "simplemeshmanager.hpp"

using namespace std;
using Teuchos::RCP;
using Teuchos::rcp;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<Teuchos::MpiComm<int>> Comm = Teuchos::rcp( new Teuchos::MpiComm<int>(MPI_COMM_WORLD) );

  Kokkos::initialize();
  
  {
    
    int NX = 1024, NY = 1024;
    double xmin = 0.0, ymin = 0.0;
    double xmax = 1.0, ymax = 1.0;

    // ==========================================================
    // Create a simple mesh
    // ==========================================================
    { 
      Teuchos::ParameterList pl;
      pl.sublist("Geometry").set("X0",     xmin);
      pl.sublist("Geometry").set("Width",  xmax-xmin);
      pl.sublist("Geometry").set("NX",     NX);
      pl.sublist("Geometry").set("Y0",     ymin);
      pl.sublist("Geometry").set("Height", ymax-ymin);
      pl.sublist("Geometry").set("NY",     NY);
    
      auto simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle<ScalarT>>(new SimpleMeshManager_Rectangle<ScalarT>(pl));
    }
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


