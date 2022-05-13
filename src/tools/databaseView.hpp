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

#ifndef MRHYDE_DATABASEVIEW_H
#define MRHYDE_DATABASEVIEW_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // New data structure to wrap views and other data
  // =================================================================
  
  template <class ViewType>
  class DatabaseView {

  public:
    ViewType view_;
    Kokkos::View<LO*,AssemblyDevice> key_;
    
    DatabaseView(ViewType view, Kokkos::View<LO*,AssemblyDevice> key)
    : view_(view),
      key_(key)
    {}
      
    KOKKOS_INLINE_FUNCTION    
    DatabaseView() {};
    
    KOKKOS_INLINE_FUNCTION
    ~DatabaseView() {};
    
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0) const {
      return view_(key_(i0));
    }
        
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1) const {
      return view_(key_(i0), i1);
    }
        
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1, const size_type & i2) const {
      return view_(key_(i0), i1, i2);
    }
        
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1, const size_type & i2, const size_type & i3) const {
      return view_(key_(i0), i1, i2, i3);
    }
    
    KOKKOS_INLINE_FUNCTION
    size_type extent(const size_type & dim) const {
      if(dim==0)
        return key_.extent(dim);
      else
        return view_.extent(dim);
    }
    
    void print() {
      std::cout << "Printing DatabaseView" << std::endl;
      KokkosTools::print(view_);
      KokkosTools::print(key_);
    }

    void printMemory() {
      std::cout << "Printing memory savings for DatabaseView" << std::endl;
      std::cout << "Key length = " << key_.extent(0) << std::endl;
      std::cout << "Underlying view extent(0) = " << view_.extent(0) << std::endl;
      std::cout << "Underlying view total size = " << view_.extent(0)*view_.extent(1)*view_.extent(2)*view_.extent(3) << " entries " << std::endl;
      std::cout << "View total size without database = " << key_.extent(0)*view_.extent(1)*view_.extent(2)*view_.extent(3) << " entries " << std::endl;
      std::cout << "Total savings = " << (key_.extent(0) - view_.extent(0))*view_.extent(1)*view_.extent(2)*view_.extent(3) - key_.extent(0) << " entries " << std::endl;
      std::cout << "              = " << ((key_.extent(0) - view_.extent(0))*view_.extent(1)*view_.extent(2)*view_.extent(3) - key_.extent(0))*sizeof(typename ViewType::value_type) << " bytes " << std::endl;
    }
  };
  
}

#endif

