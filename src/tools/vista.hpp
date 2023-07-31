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

#ifndef MRHYDE_VISTA_H
#define MRHYDE_VISTA_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // New data structure to wrap views and other data
  // =================================================================
  
  class Vista {

    #ifndef MrHyDE_NO_AD
      typedef Kokkos::View<AD*,ContLayout,AssemblyDevice> View_AD1;
      typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2;
      typedef Kokkos::View<AD***,ContLayout,AssemblyDevice> View_AD3;
      typedef Kokkos::View<AD****,ContLayout,AssemblyDevice> View_AD4;
    #else
      typedef View_Sc1 View_AD1;
      typedef View_Sc2 View_AD2;
      typedef View_Sc3 View_AD3;
      typedef View_Sc4 View_AD4;
    #endif
    
  private:
    
    bool is_view_, is_AD_;
    
    // Various data storage types
    // Only one of these will get used
    View_AD2 viewdata_;
    View_Sc2 viewdata_Sc_;
    
  public:
    KOKKOS_INLINE_FUNCTION    
    Vista() {};
    
    KOKKOS_INLINE_FUNCTION
    ~Vista() {};
    
#ifndef MrHyDE_NO_AD
    Vista(View_AD2 vdata) {
      viewdata_ = vdata;
      is_AD_ = true;
      is_view_ = true;
    }
#endif
    
    Vista(View_Sc2 vdata) {
      viewdata_Sc_ = vdata;
      viewdata_ = View_AD2("2D view",vdata.extent(0),vdata.extent(1));
      is_view_ = true;
      is_AD_ = false;
    }

#ifndef MrHyDE_NO_AD
    Vista(AD & data_) {
      viewdata_ = View_AD2("2D view",1,1);
      deep_copy(viewdata_,data_);
      is_view_ = false;
      is_AD_ = true;
    }
#endif
    
    Vista(ScalarT & data_) {
      viewdata_ = View_AD2("2D view",1,1);
      deep_copy(viewdata_,data_);
      is_view_ = false;
      is_AD_ = false;
    }

#ifndef MrHyDE_NO_AD
    void update(View_AD2 vdata) {
      viewdata_ = vdata;
    }
#endif
    
    void update(View_Sc2 vdata) {
      viewdata_Sc_ = vdata;
    }
    
#ifndef MrHyDE_NO_AD
    void update(AD & data_) {
      deep_copy(viewdata_,data_);
    }
#endif
    
    void update(ScalarT & data_) {
      deep_copy(viewdata_,data_);
    }
    
    KOKKOS_INLINE_FUNCTION
    View_AD2::reference_type operator()(const size_type & i0, const size_type & i1) const {
      if (is_view_) {
        if (is_AD_) {
          return viewdata_(i0,i1);
        }
        else {
#ifndef MrHyDE_NO_AD
          viewdata_(i0,i1).val() = viewdata_Sc_(i0,i1);
          return viewdata_(i0,i1);
#else
          return viewdata_Sc_(i0,i1);
#endif
        }
      }
      else {
        return viewdata_(0,0);
      }
    }
    
    bool isView() {
      return is_view_;
    }

    bool isAD() {
      return is_AD_;
    }
    
    View_AD2 getData() {
      return viewdata_;
    }

    View_Sc2 getDataSc() {
      return viewdata_Sc_;
    }
    
    /*
    KOKKOS_INLINE_FUNCTION
    size_type extent(const size_type & dim) const {
      if (is_view_) {
        return viewdata_.extent(dim);
      }
      else {
        return 1;
      }
    }
    */
    
    void print() {
      std::cout << "Printing Vista -------" <<std::endl;
      std::cout << "  Is View: " << is_view_ << std::endl;
      std::cout << "  Is AD: " << is_AD_ << std::endl;
      
    }
  };
  
}

#endif

