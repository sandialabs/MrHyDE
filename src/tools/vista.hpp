/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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
    

    Vista(View_EvalT2 vdata);

#ifndef MrHyDE_NO_AD    
    Vista(View_Sc2 vdata);
#endif

//#ifndef MrHyDE_NO_AD
//    Vista(EvalT & data_);
//#endif
    
    
    Vista(ScalarT & data_);


    void update(View_EvalT2 vdata);

#ifndef MrHyDE_NO_AD    
    void update(View_Sc2 vdata);
#endif    

    void update(EvalT & data_);

//#ifndef MrHyDE_NO_AD    
    void updateSc(ScalarT & data_);
//#endif

    //void updateParam(AD & pdata_);
    
    KOKKOS_INLINE_FUNCTION
    typename Kokkos::View<EvalT**,ContLayout,AssemblyDevice>::reference_type operator()(const size_type & i0, const size_type & i1) const {
      if (is_view_) {
        if (is_AD_) {
          return viewdata_(i0,i1);
        }
        else {
          viewdata_(i0,i1) = viewdata_Sc_(i0,i1);
          return viewdata_(i0,i1);
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

