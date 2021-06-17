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

#ifndef COMPADRE_TOOLS_H
#define COMPADRE_TOOLS_H

#include "trilinos.hpp"
#include "preferences.hpp"

#include "Compadre_Config.h"
#include "Compadre_GMLS.hpp"
#include "Compadre_Evaluator.hpp"
#include "Compadre_PointCloudSearch.hpp"
#include "Compadre_KokkosParser.hpp"
#include "Compadre_NeighborLists.hpp"

#include <iostream>
#include <iterator>

// Compadre relies on being able to access memory on the host,
// however, it also uses Kokkos::parallel_for internally, 
// which means the most performant option for GPU may be UVM.
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  typedef Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace> CompadreDevice;
#else
  typedef Kokkos::Device<AssemblyExec, AssemblyMem> CompadreDevice;
#endif

// GH Notes:
// If AssemblyDevice is Host, there's no need for mirrors of copies.
// If AssemblyDevice is OMP, there's no need for mirrors or copies.
// If AssemblyDevice is Cuda, then we need a host accessible version of that memory,
//   which we then use to create a UVM view



/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// returns a Compadre::NeighborLists object which lists which sensor is
// closest to each cell.
template<typename sensors_view_type>
inline
Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> > 
CompadreTools_constructNeighborLists(const sensors_view_type &sensor_coords_in,
                                     const Kokkos::View<ScalarT**, AssemblyDevice> &cell_coords_assembly,
                                     Kokkos::View<ScalarT*, AssemblyDevice> &epsilon_assembly) {

  Kokkos::View<ScalarT**, CompadreDevice> sensor_coords("sensor coords Compadre device", sensor_coords_in.extent(0), sensor_coords_in.extent(1));
  Kokkos::View<ScalarT**, CompadreDevice> cell_coords("cell coords", cell_coords_assembly.extent(0), cell_coords_assembly.extent(1));

  // GH: Probably don't need this since Kokkos is smart
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  // We need host accessible versions of the input views
  auto cell_coords_mirror = Kokkos::create_mirror_view(cell_coords_assembly);
  Kokkos::deep_copy(cell_coords_mirror, cell_coords_assembly);

  for(unsigned int i=0; i<sensor_coords.extent(0); ++i)
    for(unsigned int d=0; d<sensor_coords.extent(1); ++d)
      sensor_coords(i,d) = sensor_coords_in(i,d);

  for(unsigned int i=0; i<cell_coords.extent(0); ++i)
    for(unsigned int d=0; d<cell_coords.extent(1); ++d)
      cell_coords(i,d) = cell_coords_mirror(i,d);
#else 
  sensor_coords = sensor_coords_in;
  cell_coords = cell_coords_assembly;
#endif

  //Kokkos::Timer timer;
  //timer.reset();
  
  // LO number_sensor_coords = sensor_coords.extent(0);
  LO number_cell_coords = cell_coords.extent(0);
  LO dimension = sensor_coords.extent(1);
  TEUCHOS_ASSERT(cell_coords.extent(1)==sensor_coords.extent(1));
  int min_neighbors = 1; // must find at least 1 neighbor
  double epsilon_mult = 1.000000001; // if you want to search for many neighbors within a multiplied radius of the closest neighbor, increase this

  Kokkos::View<int*, CompadreDevice> neighbor_lists("neighbor lists", 0); // first column is # of neighbors
  
  // this will count the number of neighbors for each sensor
  Kokkos::View<int*, CompadreDevice> number_of_neighbors_list("number of neighbor lists", number_cell_coords); // first column is # of neighbors
  
  // each target site has a window size
  Kokkos::View<double*, CompadreDevice> epsilon("epsilon Compadre device", number_cell_coords);
  
  //double time1 = timer.seconds();
  //printf("Step 1 time:   %e \n", time1);
  //timer.reset();
  
  auto point_cloud_search(Compadre::CreatePointCloudSearch(sensor_coords, dimension));
  
  //double time2 = timer.seconds();
  //printf("Step 2 time:   %e \n", time2);
  //timer.reset();
  
  size_t storage_size = point_cloud_search.generateCRNeighborListsFromKNNSearch(true /*dry run*/, cell_coords, neighbor_lists, number_of_neighbors_list, epsilon, min_neighbors, epsilon_mult);

  Kokkos::resize(neighbor_lists, storage_size);

  // GH: fixing this next 
  TEUCHOS_ASSERT(neighbor_lists.extent(0)==cell_coords.extent(0)); // if this assert fails, some points have multiple neighbors of equal distance, which requires some updates in implementation

  point_cloud_search.generateCRNeighborListsFromKNNSearch(false /*not dry run*/, cell_coords, neighbor_lists, number_of_neighbors_list, epsilon, min_neighbors, epsilon_mult);
  
  Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> > neighbor_lists_object = Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> >(neighbor_lists, number_of_neighbors_list);
  // referred to as nla sometimes, has methods in Compadre_NeighborLists.hpp
    
  //double time3 = timer.seconds();
  //printf("Step 3 time:   %e \n", time3);
  //timer.reset();
  
  return neighbor_lists_object;
}




#endif
