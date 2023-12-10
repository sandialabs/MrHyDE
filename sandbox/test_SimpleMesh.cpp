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

    // ==========================================================
    // Create a simple mesh
    // ==========================================================
    {
      int NX = 512, NY = 512;
      double xmin = 0.0, ymin = 0.0;
      double xmax = 1.0, ymax = 1.0;
      Teuchos::ParameterList pl;
      pl.sublist("Geometry").set("X0",     xmin);
      pl.sublist("Geometry").set("Width",  xmax-xmin);
      pl.sublist("Geometry").set("NX",     NX);
      pl.sublist("Geometry").set("Y0",     ymin);
      pl.sublist("Geometry").set("Height", ymax-ymin);
      pl.sublist("Geometry").set("NY",     NY);

      auto simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle<ScalarT>>(new SimpleMeshManager_Rectangle<ScalarT>(pl));
    }

    // ==========================================================
    // Create the parallel version of a simple mesh
    // ==========================================================
    {
      int NX = 2, NY = 3;
      double xmin = 0.0, ymin = 0.0;
      double xmax = 1.0, ymax = 1.0;
      Teuchos::ParameterList pl;
      pl.sublist("Geometry").set("X0",     xmin);
      pl.sublist("Geometry").set("Width",  xmax-xmin);
      pl.sublist("Geometry").set("NX",     NX);
      pl.sublist("Geometry").set("Y0",     ymin);
      pl.sublist("Geometry").set("Height", ymax-ymin);
      pl.sublist("Geometry").set("NY",     NY);

      int xprocs = 3;
      int yprocs = 2;

      int procid = -1;
      long long gid = -1;

      procid = 1;
      auto simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle_Parallel<ScalarT>>(
                           new SimpleMeshManager_Rectangle_Parallel<ScalarT>(pl, procid, xprocs, yprocs));
      gid = simple_mesh->localToGlobal(11);
      std::cout << "LID 11 on proc 1 is GID " << gid << std::endl; // 25
      std::cout << "LID 11 is " << ( simple_mesh->isShared(11) ? "" : "not " ) << "shared" << std::endl; // is
      std::cout << "LID  6 is " << (  simple_mesh->isShared(6) ? "" : "not " ) << "shared" << std::endl << std::endl; // is not

      procid = 2;
      simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle_Parallel<ScalarT>>(
                      new SimpleMeshManager_Rectangle_Parallel<ScalarT>(pl, procid, xprocs, yprocs));
      gid = simple_mesh->localToGlobal(9);
      std::cout << "LID  9 on proc 2 is GID " << gid << std::endl; // 25
      std::cout << "LID 11 is " << ( simple_mesh->isShared(11) ? "" : "not " ) << "shared" << std::endl; // is not
      std::cout << "LID  6 is " << (  simple_mesh->isShared(6) ? "" : "not " ) << "shared" << std::endl << std::endl; // is not

      procid = 4;
      simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle_Parallel<ScalarT>>(
                      new SimpleMeshManager_Rectangle_Parallel<ScalarT>(pl, procid, xprocs, yprocs));
      gid = simple_mesh->localToGlobal(2);
      std::cout << "LID  2 on proc 4 is GID " << gid << std::endl; // 25
      std::cout << "LID 11 is " << ( simple_mesh->isShared(11) ? "" : "not " ) << "shared" << std::endl; // is not
      std::cout << "LID  6 is " << (  simple_mesh->isShared(6) ? "" : "not " ) << "shared" << std::endl << std::endl; // is not

      procid = 5;
      simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle_Parallel<ScalarT>>(
                      new SimpleMeshManager_Rectangle_Parallel<ScalarT>(pl, procid, xprocs, yprocs));
      gid = simple_mesh->localToGlobal(0);
      std::cout << "LID  0 on proc 5 is GID " << gid << std::endl; // 25
      std::cout << "LID 11 is " << ( simple_mesh->isShared(11) ? "" : "not " ) << "shared" << std::endl; // is not
      std::cout << "LID  6 is " << (  simple_mesh->isShared(6) ? "" : "not " ) << "shared" << std::endl << std::endl; // is not

      procid = 3;
      simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle_Parallel<ScalarT>>(
                      new SimpleMeshManager_Rectangle_Parallel<ScalarT>(pl, procid, xprocs, yprocs));
      gid = simple_mesh->localToGlobal(9);
      std::cout << "LID  9 on proc 5 is GID " << gid << std::endl; // 42
      std::cout << "LID 11 is " << ( simple_mesh->isShared(11) ? "" : "not " ) << "shared" << std::endl; // is not
      std::cout << "LID  5 is " << (  simple_mesh->isShared(5) ? "" : "not " ) << "shared" << std::endl << std::endl; // is

    }
  }

  Kokkos::finalize();


  int val = 0;
  return val;
}


