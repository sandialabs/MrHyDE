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


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// Compadre interface doesn't work with GPUs yet
#if !defined(MrHyDE_DISABLE_COMPADRE)
// returns a Compadre::NeighborLists object which lists which sensor is
// closest to each cell.
KOKKOS_INLINE_FUNCTION
Compadre::NeighborLists<Kokkos::View<int*> > 
CompadreTools_constructNeighborLists(const Kokkos::View<ScalarT**, AssemblyDevice> &sensor_coords,
                                     const Kokkos::View<ScalarT**, AssemblyDevice> &cell_coords,
                                     Kokkos::View<ScalarT*, AssemblyDevice> &epsilon) {

  // LO number_sensor_coords = sensor_coords.extent(0);
  LO number_cell_coords = cell_coords.extent(0);
  LO dimension = sensor_coords.extent(1);
  TEUCHOS_ASSERT(cell_coords.extent(1)==sensor_coords.extent(1));
  int min_neighbors = 1; // must find at least 1 neighbor
  double epsilon_mult = 1.000000001; // if you want to search for many neighbors within a multiplied radius of the closest neighbor, increase this

  Kokkos::View<int*, AssemblyDevice> neighbor_lists_device("neighbor lists", 0); // first column is # of neighbors
  Kokkos::View<int*>::HostMirror neighbor_lists = Kokkos::create_mirror_view(neighbor_lists_device);
  
  // this will count the number of neighbors for each sensor
  Kokkos::View<int*, AssemblyDevice> number_of_neighbors_list_device("number of neighbor lists", number_cell_coords); // first column is # of neighbors
  Kokkos::View<int*>::HostMirror number_of_neighbors_list = Kokkos::create_mirror_view(number_of_neighbors_list_device);
  
  // each target site has a window size
  Kokkos::resize(epsilon,number_cell_coords);
  Kokkos::View<double*>::HostMirror epsilon_h = Kokkos::create_mirror_view(epsilon);
  
  auto point_cloud_search(Compadre::CreatePointCloudSearch(sensor_coords, dimension));
  
  size_t storage_size = point_cloud_search.generateCRNeighborListsFromKNNSearch(true /*dry run*/, cell_coords, neighbor_lists, number_of_neighbors_list, epsilon_h, min_neighbors, epsilon_mult);

  Kokkos::resize(neighbor_lists_device, storage_size);
  
  TEUCHOS_ASSERT(neighbor_lists_device.extent(0)==cell_coords.extent(0)); // if this assert fails, some points have multiple neighbors of equal distance, which requires some updates in implementation

  neighbor_lists = Kokkos::create_mirror_view(neighbor_lists_device);
  
  point_cloud_search.generateCRNeighborListsFromKNNSearch(false /*not dry run*/, cell_coords, neighbor_lists, number_of_neighbors_list, epsilon_h, min_neighbors, epsilon_mult);
  
  Compadre::NeighborLists<Kokkos::View<int*> > neighbor_lists_object = Compadre::NeighborLists<Kokkos::View<int*> >(neighbor_lists, number_of_neighbors_list);
  // referred to as nla sometimes, has methods in Compadre_NeighborLists.hpp
    
  return neighbor_lists_object;
}

#endif



#endif
