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
  */
  class CompressedViewSc3 {

  public:
    //! The underlying data
    View_Sc3 view_;
    //! The random access key for grabbing the underlying data
    Kokkos::View<LO*,AssemblyDevice> key_;
    //! A boolean whether the key has been allocated. If not, the view behaves like a normal view.
    bool have_key_;

    //! Constructor for the case where the view is compressed.
    CompressedViewSc3(View_Sc3 view, Kokkos::View<LO*,AssemblyDevice> key)
    : view_(view),
      key_(key)
    {
      have_key_ = true;
    }

    //! Constructor for the case where the view is not compressed.
    CompressedViewSc3(View_Sc3 view)
    : view_(view)
    {
      have_key_ = false;
    }
    
    //! Default constructor
    KOKKOS_INLINE_FUNCTION    
    CompressedViewSc3() {};
    
    //! Default destructor
    KOKKOS_INLINE_FUNCTION
    ~CompressedViewSc3() {};

    //! Decompress the view by returning a view with all the copied data
    View_Sc3 decompress() const {
      if (have_key_) {
        View_Sc3 decompressed_view("decompressed view",key_.extent(0),view_.extent(1),view_.extent(2));
        parallel_for("decompress view",
                     RangePolicy<AssemblyExec>(0,key_.extent(0)),
                     KOKKOS_LAMBDA (const size_type i) {
          for (size_type j=0; j<view_.extent(1); ++j) {
            for (size_type k=0; k<view_.extent(2); ++k) {
              decompressed_view(i,j,k) = view_(key_(i),j,k);
            }
          }
        });
        return decompressed_view;
      }
      else {
        return view_;
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
    View_Sc3 getView() const {
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
      std::cout << "              = " << ((key_.extent(0) - view_.extent(0))*view_.extent(1)*view_.extent(2)*view_.extent(3) - key_.extent(0))*sizeof(typename View_Sc3::value_type) << " bytes " << std::endl;
    }
  };

  /**
   * \brief A class which compresses data by storing a view and then allowing random access into it based on a key
   */
  class CompressedViewSc4 {

  public:
    //! The underlying data
    View_Sc4 view_;
    //! The random access key for grabbing the underlying data
    Kokkos::View<LO*,AssemblyDevice> key_;
    //! A boolean whether the key has been allocated. If not, the view behaves like a normal view.
    bool have_key_;

    //! Constructor for the case where the view is compressed.
    CompressedViewSc4(View_Sc4 view, Kokkos::View<LO*,AssemblyDevice> key)
    : view_(view),
      key_(key)
    {
      have_key_ = true;
    }

    //! Constructor for the case where the view is not compressed.
    CompressedViewSc4(View_Sc4 view)
    : view_(view)
    {
      have_key_ = false;
    }
    
    //! Default constructor
    KOKKOS_INLINE_FUNCTION    
    CompressedViewSc4() {};
    
    //! Default destructor
    KOKKOS_INLINE_FUNCTION
    ~CompressedViewSc4() {};

    //! Decompress the view by returning a view with all the copied data
    View_Sc4 decompress() const {
      if (have_key_) {
        View_Sc4 decompressed_view("decompressed view",key_.extent(0),view_.extent(1),view_.extent(2),view_.extent(3));
        parallel_for("decompress view",
                     RangePolicy<AssemblyExec>(0,key_.extent(0)),
                     KOKKOS_LAMBDA (const size_type i) {
          for (size_type j=0; j<view_.extent(1); ++j) {
            for (size_type k=0; k<view_.extent(2); ++k) {
              for (size_type l=0; l<view_.extent(3); ++l) {
                decompressed_view(i,j,k,l) = view_(key_(i),j,k,l);
              }
            }
          }
        });
        return decompressed_view;
      }
      else {
        return view_;
      }
    }

    //! Access for a four-dimensional view
    KOKKOS_INLINE_FUNCTION
    typename View_Sc4::reference_type operator()(const size_type & i0, const size_type & i1, const size_type & i2, const size_type & i3) const {
      if (have_key_) {
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
    View_Sc4 getView() const {
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
      std::cout << "              = " << ((key_.extent(0) - view_.extent(0))*view_.extent(1)*view_.extent(2)*view_.extent(3) - key_.extent(0))*sizeof(typename View_Sc4::value_type) << " bytes " << std::endl;
    }
  };
  
}

#endif

