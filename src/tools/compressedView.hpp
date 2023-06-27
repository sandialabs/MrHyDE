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

#ifndef MRHYDE_COMPRESSEDVIEW_H
#define MRHYDE_COMPRESSEDVIEW_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // New data structure to store compressed views (using a database)
  // =================================================================
  
  /**
   * \brief A class which compresses data by storing a view and then allowing random access into it based on a key
   * Additional functionality allows for on-the-fly scalar multiplication
  */
  template <class ViewType>
  class CompressedView {

  public:
    // GH: put booleans at top of class for fastest access
    //! A boolean whether the key has been allocated. If not, the view behaves like a normal view.
    bool have_key_;
    //! A boolean whether the scales have been allocated. If not, the view compresses by index only
    bool have_scales_;

    //! The underlying data
    ViewType view_;
    //! The random access key for grabbing the underlying data (used when have_key_ is true)
    Kokkos::View<LO*,AssemblyDevice> key_;
    //! The scalar scales for the data (used when have_scales_ is true)
    View_Sc2 scales_;

    //! Constructor for the case where the view is compressed and scaled.
    CompressedView(ViewType view, Kokkos::View<LO*,AssemblyDevice> key, View_Sc2 scales)
    : view_(view),
      key_(key),
      scales_(scales)
    {
      have_key_ = true;
      have_scales_ = true;
    }

    //! Constructor for the case where the view is compressed.
    CompressedView(ViewType view, Kokkos::View<LO*,AssemblyDevice> key)
    : view_(view),
      key_(key)
    {
      have_key_ = true;
      have_scales_ = false;
      // TODO: Remove the following lines after performance testing is done
      have_scales_ = true;
      scales_ = View_Sc2("database scales", key_.extent(0), 3);
      Kokkos::deep_copy(scales_,1.0);
    }

    //! Constructor for the case where the view is not compressed.
    CompressedView(ViewType view)
    : view_(view)
    {
      have_key_ = false;
      have_scales_ = false;
    }

    //! Default constructor
    KOKKOS_INLINE_FUNCTION    
    CompressedView() {};

    //! Default destructor
    KOKKOS_INLINE_FUNCTION
    ~CompressedView() {};

    //! Indexing for 1-dimensional views
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0) const {
      if (have_key_) {
        return view_(key_(i0));
      }
      else {
        return view_(i0);
      }
    }

    //! Indexing for 2-dimensional views
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1) const {
      if (have_key_) {
        return view_(key_(i0), i1);
      }
      else {
        return view_(i0, i1);
      }

    }

    //! Indexing for 3-dimensional views
    KOKKOS_INLINE_FUNCTION
    typename ViewType::reference_type operator()(const size_type & i0, const size_type & i1, const size_type & i2) const {
      if (have_key_) {
        return view_(key_(i0), i1, i2);
      }
      else {
        return view_(i0, i1, i2);
      }
    }

    //! Indexing for 4-dimensional views, commonly of the form (elem,dof,pt,dim)
    KOKKOS_INLINE_FUNCTION
    ScalarT operator()(const size_type & i0, const size_type & i1, const size_type & i2, const size_type & i3) const {
      if (have_scales_) {
        return view_(key_(i0), i1, i2, i3)*scales_(i0,i3);
      }
      else if (have_key_) {
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

    //! The size of the compressed view
    KOKKOS_INLINE_FUNCTION
    size_type size() const {
      if (have_key_) {
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
      std::cout << "Printing CompressedView" << std::endl;
      KokkosTools::print(view_);
      KokkosTools::print(key_);
    }

    //! Print the memory savings for the compressed view
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
