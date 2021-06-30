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
    ScalarT data_Sc;
    AD data;
    
    AD outval = 0.0;

    KOKKOS_INLINE_FUNCTION    
    Vista() {};
    
    KOKKOS_INLINE_FUNCTION
    ~Vista() {};
    
    Vista(View_AD2 vdata) : isView(true) {
      viewdata = vdata;
      viewdata_Sc = View_Sc2("empty view",0,0);
      data = 0.0;
      data_Sc = 0.0;
      
      isAD = true;
      isView = true;
    }
    
    Vista(View_Sc2 vdata) {
      viewdata_Sc = vdata;
      viewdata = View_AD2("empty view",1,1);
      data = 0.0;
      data_Sc = 0.0;
      
      isView = true;
      isAD = false;
    }
    
    Vista(AD & data_) {
      viewdata = View_AD2("empty view",1,1);
      viewdata_Sc = View_Sc2("empty view",0,0);
      data = data_;
      data_Sc = 0.0;
     
      deep_copy(viewdata,data); 
      isView = false;
      isAD = true;
    }
    
    Vista(ScalarT & data_) {
      viewdata = View_AD2("empty view",1,1);
      viewdata_Sc = View_Sc2("empty view",0,0);
      data = 0.0;
      data_Sc = data_;
     
      deep_copy(viewdata,data_Sc); 
      isView = false;
      isAD = false;
    }
    
    //KOKKOS_FORCEINLINE_FUNCTION
    KOKKOS_INLINE_FUNCTION
    //AD& operator()(const size_type & i0, const size_type & i1) const {
    View_AD2::reference_type operator()(const size_type & i0, const size_type & i1) const {
      if (isView) {
        if (isAD) {
          return viewdata(i0,i1);
        }
        else {
          viewdata(0,0).val() = viewdata_Sc(i0,i1);
          return viewdata(0,0);
          //AD newval = viewdata_Sc(i0,i1);
          //return newval;
        }
      }
      else {
        if (isAD) {
          //viewdata(0,0) = data;
          return viewdata(0,0);
          //return data;
        }
        else {
          //viewdata(0,0) = data_Sc;
          return viewdata(0,0);
          //AD newval = data_Sc;
          //return newval;
        }
      }
    }
    /*
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<isView, AD>::type
    operator()(const size_type& i0, const size_type& i1) const {
      return viewdata(i0,i1);
    }
    */
    /*
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<isView && !isAD,
    View_Sc2::reference_type>::type
    operator()(const size_type& i0, const size_type& i1) const {
      return viewdata_Sc(i0,i1);
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<!isView && isAD,
    View_AD2::reference_type>::type
    operator()(const size_type& i0, const size_type& i1) const {
      return data;
    }
    
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<!isView && !isAD,
    View_Sc2::reference_type>::type
    operator()(const size_type& i0, const size_type& i1) const {
      return data_Sc;
    }
     */
  };
  
}

#endif

