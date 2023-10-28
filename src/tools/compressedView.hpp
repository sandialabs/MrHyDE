/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_COMPRESSEDVIEW_H
#define MRHYDE_COMPRESSEDVIEW_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // New data structure to store compressed views (using a database)
  // =================================================================
  
  template <class ViewType>
  class CompressedView {

  private:
    ViewType view_;
    Kokkos::View<LO*,AssemblyDevice> key_;
    bool have_key_;
    size_t rank_;

  public:
    CompressedView(ViewType view, Kokkos::View<LO*,AssemblyDevice> key)
    : view_(view),
      key_(key)
    {
      have_key_ = true;
      rank_ = view_.rank();
    }

    CompressedView(ViewType view)
    : view_(view)
    {
      have_key_ = false;
      rank_ = view_.rank();
    }
      
    KOKKOS_INLINE_FUNCTION    
    CompressedView() {};
    
    KOKKOS_INLINE_FUNCTION
    ~CompressedView() {};
    
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0) const {
      if (have_key_) {
        return view_(key_(i0));
      }
      else {
        return view_(i0);
      }
    }
        
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1) const {
      if (have_key_) {
        return view_(key_(i0), i1);
      }
      else {
        return view_(i0, i1);
      }

    }
        
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1, const size_type & i2) const {
      if (have_key_) {
        return view_(key_(i0), i1, i2);
      }
      else {
        return view_(i0, i1, i2);
      }
    }
        
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1, const size_type & i2, const size_type & i3) const {
      if (have_key_) {
        return view_(key_(i0), i1, i2, i3);
      }
      else {
        return view_(i0, i1, i2, i3);
      }
    }
    
    // Decompress the data stored in view_ using key_
    // This should ONLY be used if the decompressed view is temporary

    KOKKOS_INLINE_FUNCTION
    size_type extent(const size_type & dim) const {
      if (dim==0) {
        if (have_key_) {
          return key_.extent(dim);
        }
        else {
          return view_.extent(dim);
        }
      }
      else {
        return view_.extent(dim);
      }
    }
    
    KOKKOS_INLINE_FUNCTION
    size_type size() const {
      if (have_key_) {
        return key_.size();
      }
      else {
        return view_.size();
      }
    }

    KOKKOS_INLINE_FUNCTION
    ViewType getView() const {
      return view_;
    }
    
    KOKKOS_INLINE_FUNCTION
    Kokkos::View<LO*,AssemblyDevice> getKey() const {
      return key_;
    }

    KOKKOS_INLINE_FUNCTION
    bool getHaveKey() const {
      return have_key_;
    }
    
    void print() {
      std::cout << "Printing CompressedView" << std::endl;
      KokkosTools::print(view_);
      KokkosTools::print(key_);
    }

    void printMemory() {
      std::cout << "Printing memory savings for CompressedView" << std::endl;
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

