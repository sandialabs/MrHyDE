/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "trilinos.hpp"
#include "preferences.hpp"

#include "Panzer_DOFManager.hpp"

#include "meshInterface.hpp"
#include "CompadreInterface.hpp"

#include <iostream>


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// test a 3D example of the mesh data and using the old approach vs using Compadre's functionality
// this tries to make both approaches respect the framework but also be as similar as possible in order to be a fair comparison
// in short, this test computes nearest neighbors using perm_xy.dat on an nx x ny x nz grid and does two things
// 1. Compares the results of the nearest neighbors to ensure that both approaches are correct
// 2. Compares the timing results of both approaches to see if one is advantageous over the other

int main(int argc, char * argv[]) {

  
#if !defined(MrHyDE_DISABLE_COMPADRE)

  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Teuchos::RCP<MpiComm> Comm = Teuchos::rcp( new MpiComm(MPI_COMM_WORLD) );

  int myRank = Comm->getRank();
  //int numRanks = Comm->getSize();

  Kokkos::initialize();
  { // Begin Kokkos scope

  if(myRank == 0) {
    std::cout << std::endl << "Notes:" << std::endl;
    std::cout << "  1. This will segfault if it isn't run from the MrHyDE-build/src/ directory!" << std::endl;
    std::cout << "  2. There is an abuse of notation here. If a point has more than one 'exact' closest neighbor, an additional step needs to be taken to choose either of the two closest neighbors." << std::endl;
    std::cout << "  3. This test is being phased out now that the changes have been implemented in the data class." << std::endl << std::endl;
  }

  const unsigned int spaceDim = 3;
  const string mesh_data_pts_file = "perm_xy.dat";

  // INPUT - MODIFY FOR PERFORMANCE TESTING
  bool output_cnode = false; // dump the values in cnode to the screen after each example for eyeball norm debugging

  // number of cells in x,y,z directions
  unsigned int nx = 100;
  unsigned int ny = 100;
  unsigned int nz = 100;

  // coordinates of domain
  ScalarT xmin = 0., xmax = 1.;
  ScalarT ymin = 0., ymax = 1.;
  ScalarT zmin = 0., zmax = 1.;

  // DO NOT MODIFY BEYOND HERE
  unsigned int numElem = nx*ny*nz;

  // step between elements
  ScalarT dx = (xmax-xmin)/nx;
  ScalarT dy = (ymax-ymin)/ny;
  ScalarT dz = (zmax-zmin)/nz;

  // centers are offset by exactly 1/2*width
  ScalarT offset_x = dx/2.;
  ScalarT offset_y = dy/2.;
  ScalarT offset_z = dz/2.;

  // compute "centers" of cells on this hypothetical mesh
  Kokkos::View<ScalarT**, HostDevice> centers("centers",numElem,spaceDim);
  for (unsigned int ix=0; ix<nx; ix++) {
    for (unsigned int iy=0; iy<ny; iy++) {
      for (unsigned int iz=0; iz<nz; iz++) {
        centers(iz*ny*nx + iy*nx + ix,0) = offset_x + ix*dx;
        centers(iz*ny*nx + iy*nx + ix,1) = offset_y + iy*dy;
        centers(iz*ny*nx + iy*nx + ix,2) = offset_z + iz*dz;
      }
    }
  }

  Teuchos::RCP<MrHyDE::Data> mesh_data;
  mesh_data = Teuchos::rcp(new MrHyDE::Data("mesh data", spaceDim, mesh_data_pts_file));

  if(myRank == 0)
    std::cout << "Starting the mesh_data->findClosestNode approach..." << std::endl;

  // Test the mesh_data->findClosestNode approach
  Kokkos::View<int*, HostDevice> cnode_meshdata("cnode_meshdata",numElem);
  Teuchos::Time meshdataTimer("meshdata approach",true);
  {
    for (unsigned int ix=0; ix<nx; ix++) {
      for (unsigned int iy=0; iy<ny; iy++) {
        for (unsigned int iz=0; iz<nz; iz++) {
          // ScalarT distance = 0.0;

          // unsigned int j = iz*ny*nx + iy*nx + ix;
          // find the closest mesh_data point to "center"
	  //          cnode_meshdata(j) = mesh_data->findClosestNode(centers(j,0), centers(j,1), centers(j,2), distance);
        }
      }
    }
  }
  ScalarT meshdataTime = meshdataTimer.stop();
  // End the mesh_data->findClosestNode approach

  if(myRank == 0) {
    std::cout << "Finished the mesh_data->findClosestNode approach!" << std::endl;
    std::cout << "Elapsed time: " << meshdataTime << std::endl;
    std::cout << std::endl;

    if(output_cnode == true) {
      std::cout << "cnode_meshdata has " << cnode_meshdata.extent(0) << " entries" << std::endl;
      std::cout << "cnode_meshdata = {" << cnode_meshdata(0);
      for(size_type j=1; j<cnode_meshdata.extent(0); ++j)
        std::cout << ", " << cnode_meshdata(j);
      std::cout << "}" << std::endl << std::endl;
    }

    std::cout << "Starting the CompadreTools_constructNeighborLists approach..." << std::endl;
  }

  // Test the CompadreTools_constructNeighborLists approach
  Kokkos::View<int*> cnode_compadre("cnode_compadre",numElem);
  Teuchos::Time compadreTimer("compadre approach",true);
  {
    Kokkos::View<ScalarT**, AssemblyDevice> sensor_coords = mesh_data->getPoints();
    Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
    auto neighborlists = CompadreInterface_constructNeighborLists(sensor_coords, centers, distance);
    cnode_compadre = neighborlists.getNeighborLists();
  }
  ScalarT compadreTime = compadreTimer.stop();
  // End the CompadreTools_constructNeighborLists approach


  if(myRank == 0) {
    std::cout << "Finished the CompadreTools_constructNeighborLists approach!" << std::endl;
    std::cout << "Elapsed time: " << compadreTime << std::endl;
    std::cout << std::endl;

    if(output_cnode == true) {
      std::cout << "cnode_compadre has " << cnode_compadre.extent(0) << " entries" << std::endl;
      std::cout << "cnode_compadre = {" << cnode_compadre(0);
      for(size_type j=1; j<cnode_compadre.extent(0); ++j)
        std::cout << ", " << cnode_compadre(j);
      std::cout << "}" << std::endl << std::endl;
    }
  }

  } // End Kokkos scope
  Kokkos::finalize();
#endif
  
  return 0;
}

