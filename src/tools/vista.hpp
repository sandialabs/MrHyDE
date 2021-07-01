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
  public:
    
    bool isView, isAD;
    
    // Various data storage types
    // Only one of these will get used
    View_AD2 viewdata;
    View_Sc2 viewdata_Sc;
    
    KOKKOS_INLINE_FUNCTION    
    Vista() {};
    
    KOKKOS_INLINE_FUNCTION
    ~Vista() {};
    
    Vista(View_AD2 vdata) {
      viewdata = vdata;
      isAD = true;
      isView = true;
    }
    
    Vista(View_Sc2 vdata) {
      viewdata_Sc = vdata;
      viewdata = View_AD2("2D view",1,1);
      isView = true;
      isAD = false;
    }
    
    Vista(AD & data_) {
      viewdata = View_AD2("2D view",1,1);
      deep_copy(viewdata,data_);
      isView = false;
      isAD = true;
    }
    
    Vista(ScalarT & data_) {
      viewdata = View_AD2("2D view",1,1);
      deep_copy(viewdata,data_);
      isView = false;
      isAD = false;
    }
    
    void update(View_AD2 vdata) {
      viewdata = vdata;
    }
    
    void update(View_Sc2 vdata) {
      viewdata_Sc = vdata;
    }
    
    void update(AD & data_) {
      deep_copy(viewdata,data_);
    }
    
    void update(ScalarT & data_) {
      deep_copy(viewdata,data_);
    }
    
    KOKKOS_INLINE_FUNCTION
    View_AD2::reference_type operator()(const size_type & i0, const size_type & i1) const {
      if (isView) {
        if (isAD) {
          return viewdata(i0,i1);
        }
        else {
          viewdata(0,0).val() = viewdata_Sc(i0,i1);
          return viewdata(0,0);
        }
      }
      else {
        if (isAD) {
          return viewdata(0,0);
        }
        else {
          return viewdata(0,0);
        }
      }
    }
    
    void print() {
      std::cout << "Printing Vista -------" <<std::endl;
      std::cout << "  Is View: " << isView << std::endl;
      std::cout << "  Is AD: " << isAD << std::endl;
      
    }
  };
  
}

#endif

