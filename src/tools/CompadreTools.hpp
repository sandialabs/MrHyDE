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

// returns a Compadre::NeighborLists object which lists which seed point is
// closest to each test point
template<typename view_type>
inline
Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> > 
CompadreTools_constructNeighborLists(const view_type &points_in,
                                     const Kokkos::View<ScalarT**, AssemblyDevice> &tstpts_assembly,
                                     Kokkos::View<ScalarT*, AssemblyDevice> &epsilon_assembly) {

  Kokkos::View<ScalarT**, CompadreDevice> points("points Compadre device", points_in.extent(0), points_in.extent(1));
  Kokkos::View<ScalarT**, CompadreDevice> tstpts("tst pts", tstpts_assembly.extent(0), tstpts_assembly.extent(1));

  // GH: Probably don't need this since Kokkos is smart
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  // We need host accessible versions of the input views
  //auto tstpts_mirror = Kokkos::create_mirror_view(tstpts_assembly);
  //Kokkos::deep_copy(tstpts_mirror, tstpts_assembly);

  // TMW: This should work ... still need to test though
  deep_copy(tstpts, tstpts_assembly);
  deep_copy(points, points_in);
  
  //for(unsigned int i=0; i<points.extent(0); ++i)
  //  for(unsigned int d=0; d<points.extent(1); ++d)
  //    sensor_coords(i,d) = sensor_coords_in(i,d);

  //for(unsigned int i=0; i<cell_coords.extent(0); ++i)
  //  for(unsigned int d=0; d<cell_coords.extent(1); ++d)
  //    cell_coords(i,d) = cell_coords_mirror(i,d);
#else 
  points = points_in;
  tstpts = tstpts_assembly;
#endif
  
  // LO number_sensor_coords = sensor_coords.extent(0);
  LO number_tstpts = tstpts.extent(0);
  LO dimension = points.extent(1);
  
  // Make sure each point set is the same dimension
  TEUCHOS_ASSERT(points.extent(1) == tstpts.extent(1));
  
  int min_neighbors = 1; // must find at least 1 neighbor
  
  // TMW: setting this to 1.0 still does not work ... I suspect rounding errors
  //      setting to 1.1 since we can now handle multiple neighbors
  //      closest neighbor is still the first one
  double epsilon_mult = 1.1; // if you want to search for many neighbors within a multiplied radius of the closest neighbor, increase this

  Kokkos::View<int*, CompadreDevice> neighbor_lists("neighbor lists", 0); // first column is # of neighbors
  
  // this will count the number of neighbors for each sensor
  Kokkos::View<int*, CompadreDevice> number_of_neighbors_list("number of neighbor lists", number_tstpts); // first column is # of neighbors
  
  // each target site has a window size
  Kokkos::View<double*, CompadreDevice> epsilon("epsilon Compadre device", number_tstpts);
  
  auto point_cloud_search(Compadre::CreatePointCloudSearch(points, dimension));
  
  size_t storage_size = point_cloud_search.generateCRNeighborListsFromKNNSearch(true /*dry run*/, tstpts, neighbor_lists, number_of_neighbors_list, epsilon, min_neighbors, epsilon_mult);

  Kokkos::resize(neighbor_lists, storage_size);

  // GH: fixing this next
  // TMW: this has been fixed within data.cpp
  //TEUCHOS_ASSERT(neighbor_lists.extent(0)==cell_coords.extent(0)); // if this assert fails, some points have multiple neighbors of equal distance, which requires some updates in implementation

  point_cloud_search.generateCRNeighborListsFromKNNSearch(false /*not dry run*/, tstpts, neighbor_lists, number_of_neighbors_list, epsilon, min_neighbors, epsilon_mult);
  
  Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> > neighbor_lists_object = Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> >(neighbor_lists, number_of_neighbors_list);
  // referred to as nla sometimes, has methods in Compadre_NeighborLists.hpp
    
  return neighbor_lists_object;
}




#endif
