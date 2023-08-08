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
  
  template<class EvalT>
  class Vista {

    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
  //private:
    
    bool is_view_, is_AD_;
    
    // Various data storage types
    // Only one of these will get used
    View_EvalT2 viewdata_;
    View_Sc2 viewdata_Sc_;
    
  public:
    KOKKOS_INLINE_FUNCTION    
    Vista() {};
    
    KOKKOS_INLINE_FUNCTION
    ~Vista() {};
    
#ifndef MrHyDE_NO_AD
    Vista(View_EvalT2 vdata);
#endif
    
    Vista(View_Sc2 vdata);

//#ifndef MrHyDE_NO_AD
//    Vista(EvalT & data_);
//#endif
    
    
    Vista(ScalarT & data_);

#ifndef MrHyDE_NO_AD
    void update(View_EvalT2 vdata);
#endif
    
    void update(View_Sc2 vdata);
    
#ifndef MrHyDE_NO_AD
    void update(EvalT & data_);
#endif
    
    void updateSc(ScalarT & data_);

    //void updateParam(AD & pdata_);
    
    KOKKOS_INLINE_FUNCTION
    typename Kokkos::View<EvalT**,ContLayout,AssemblyDevice>::reference_type operator()(const size_type & i0, const size_type & i1) const {
      if (is_view_) {
        if (is_AD_) {
          return viewdata_(i0,i1);
        }
        else {
          #ifndef MrHyDE_NO_AD
            //viewdata_(i0,i1).val() = viewdata_Sc_(i0,i1);
            viewdata_(i0,i1) = viewdata_Sc_(i0,i1);
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
    
    bool isView();

    bool isAD();
    
    View_EvalT2 getData();

    View_Sc2 getDataSc();
    
    void print();
  };
  
}

#endif

