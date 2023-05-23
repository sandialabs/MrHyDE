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

#ifndef MRHYDE_SUPERCOMPRESSEDVIEW_H
#define MRHYDE_SUPERCOMPRESSEDVIEW_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // New data structure to store compressed views (using a database)
  // =================================================================
  
  template <class ViewType>
  class SuperCompressedView {

  public:
    //! The underlying data (typically very small)
    ViewType view_;

    //! The random access key for grabbing the underlying data
    Kokkos::View<LO*,AssemblyDevice> key_;
    //! The orientations of the basis functions
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientations_;
    //! The diagonal scalings for the view
    View_Sc2 diagonal_scaling_;
    
    //! A boolean whether the data has been compressed. If not, the view behaves like a normal view.
    bool is_compressed_;

    //! Constructor for the case where the view is compressed.
    SuperCompressedView(ViewType view, Kokkos::View<LO*,AssemblyDevice> key, Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientations, View_Sc2 diagonal_scaling)
    : view_(view),
      key_(key),
      orientations_(orientations),
      diagonal_scaling_(diagonal_scaling)
    {
      is_compressed_ = true;
    }

    //! Constructor for the case where the view is not compressed.
    SuperCompressedView(ViewType view)
    : view_(view)
    {
      is_compressed_ = false;
    }
    
    //! Default constructor
    KOKKOS_INLINE_FUNCTION    
    SuperCompressedView() {};
    
    //! Default destructor
    KOKKOS_INLINE_FUNCTION
    ~SuperCompressedView() {};
    
    //! Access for a one-dimensional view
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0) const {
      if (is_compressed_) {
        return view_(key_(i0));
      }
      else {
        return view_(i0);
      }
    }

    //! Access for a two-dimensional view
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1) const {
      if (is_compressed_) {
        return view_(key_(i0), i1);
      }
      else {
        return view_(i0, i1);
      }

    }

    //! Access for a three-dimensional view
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1, const size_type & i2) const {
      if (is_compressed_) {
        return view_(key_(i0), i1, i2);
      }
      else {
        return view_(i0, i1, i2);
      }
    }

    //! Access for a four-dimensional view
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1, const size_type & i2, const size_type & i3) const {
      if (is_compressed_) {
        return view_(key_(i0), i1, i2, i3);
      }
      else {
        return view_(i0, i1, i2, i3);
      }
    }

    //! The extent of access the compressed view allows. If there is a key, it is the key's extent. Otherwise it is the view's extent.
    KOKKOS_INLINE_FUNCTION
    size_type extent(const size_type & dim) const {
      if (dim==0) {
        if (is_compressed_) {
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

    //! The size of the compressed view
    KOKKOS_INLINE_FUNCTION
    size_type size() const {
      if (is_compressed_) {
        return key_.size();
      }
      else {
        return view_.size();
      }
    }

    //! Get the underlying view
    KOKKOS_INLINE_FUNCTION
    ViewType getView() const {
      return view_;
    }
    
    //! Print the underlying view and key
    void print() {
      std::cout << "Printing SuperCompressedView" << std::endl;
      KokkosTools::print(view_);
      KokkosTools::print(key_);
    }

    //! Print the memory savings for the compressed view
    void printMemory() {
      std::cout << "Printing memory savings for SuperCompressedView" << std::endl;
      std::cout << "Key length = " << key_.extent(0) << std::endl;
      std::cout << "Diagonal scaling = " << diagonal_scaling_.extent(0)*diagonal_scaling_.extent(1) << std::endl;
      std::cout << "Orientations = " << orientations_.extent(0) << std::endl;
      std::cout << "Underlying view extent(0) = " << view_.extent(0) << std::endl;
      std::cout << "Underlying view total size = " << view_.extent(0)*view_.extent(1)*view_.extent(2)*view_.extent(3) << " entries " << std::endl;
      std::cout << "View total size without database = " << key_.extent(0)*view_.extent(1)*view_.extent(2)*view_.extent(3) << " entries " << std::endl;
      std::cout << "Total savings = " << (key_.extent(0) - view_.extent(0))*view_.extent(1)*view_.extent(2)*view_.extent(3) - key_.extent(0) << " entries " << std::endl;
      std::cout << " FIXME FIXME  = " << ((key_.extent(0) - view_.extent(0))*view_.extent(1)*view_.extent(2)*view_.extent(3) - key_.extent(0))*sizeof(typename ViewType::value_type) << " bytes " << std::endl;
    }
  };
  
}

#endif
