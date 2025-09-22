/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "vista.hpp"

using namespace MrHyDE;
    

template<class EvalT>
Vista<EvalT>::Vista(View_EvalT2 vdata) {
  viewdata_ = vdata;
  is_AD_ = true;
  is_view_ = true;
}


#ifndef MrHyDE_NO_AD    
template<class EvalT>
Vista<EvalT>::Vista(View_Sc2 vdata) {
  viewdata_Sc_ = vdata;
  viewdata_ = View_EvalT2("2D view",vdata.extent(0),vdata.extent(1));
  is_view_ = true;
  is_AD_ = false;
}
#endif

//#ifndef MrHyDE_NO_AD
//template<class EvalT>
//Vista<EvalT>::Vista(EvalT & data_) {
//  viewdata_ = View_EvalT2("2D view",1,1);
//  deep_copy(viewdata_,data_);
//  is_view_ = false;
//  is_AD_ = true;
//}
//#endif

template<class EvalT> 
Vista<EvalT>::Vista(ScalarT & data_) {
  viewdata_ = View_EvalT2("2D view",1,1);
  deep_copy(viewdata_,data_);
  is_view_ = false;
  is_AD_ = false;
}


template<class EvalT>
void Vista<EvalT>::update(View_EvalT2 vdata) {
  viewdata_ = vdata;
}


#ifndef MrHyDE_NO_AD    
template<class EvalT>
void Vista<EvalT>::update(View_Sc2 vdata) {
  viewdata_Sc_ = vdata;
}
#endif


template<class EvalT>
void Vista<EvalT>::update(EvalT & data_) {
  deep_copy(viewdata_,data_);
}


template<class EvalT>   
void Vista<EvalT>::updateSc(ScalarT & data_) {
  deep_copy(viewdata_,data_);
}

template<class EvalT>
bool Vista<EvalT>::isView() {
  return is_view_;
}

template<class EvalT>
bool Vista<EvalT>::isAD() {
  return is_AD_;
}
    
template<class EvalT>
Kokkos::View<EvalT**,ContLayout,AssemblyDevice> Vista<EvalT>::getData() {
  return viewdata_;
}

template<class EvalT>
View_Sc2 Vista<EvalT>::getDataSc() {
  return viewdata_Sc_;
}
    
template<class EvalT>
void Vista<EvalT>::print() {
  std::cout << "Printing Vista -------" <<std::endl;
  std::cout << "  Is View: " << is_view_ << std::endl;
  std::cout << "  Is AD: " << is_AD_ << std::endl;    
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::Vista<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::Vista<AD>;

// Standard built-in types
template class MrHyDE::Vista<AD2>;
template class MrHyDE::Vista<AD4>;
template class MrHyDE::Vista<AD8>;
template class MrHyDE::Vista<AD16>;
template class MrHyDE::Vista<AD18>;
template class MrHyDE::Vista<AD24>;
template class MrHyDE::Vista<AD32>;
#endif
