/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "vlasov_fokker_planck_1d2v.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
VFP1d2v<EvalT>::VFP1d2v(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "VFP1d2v";

  // save spaceDim here because it is needed (potentially) before workset is finalized
  // TODO is this needed?

  spaceDim = dimension_;

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP1d2v<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                            Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP1d2v<EvalT>::volumeResidual() {
  
  Vista<EvalT> source_rho, source_rhoux, source_rhouy, source_rhouz, source_rhoE;

  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP1d2v<EvalT>::boundaryResidual() {
  
  auto bcs = wkset->var_bcs;

  //int cside = wkset->currentside;

  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void VFP1d2v<EvalT>::computeFlux() {

  
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP1d2v<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;

}



//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::VFP1d2v<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::VFP1d2v<AD>;

// Standard built-in types
template class MrHyDE::VFP1d2v<AD2>;
template class MrHyDE::VFP1d2v<AD4>;
template class MrHyDE::VFP1d2v<AD8>;
template class MrHyDE::VFP1d2v<AD16>;
template class MrHyDE::VFP1d2v<AD18>;
template class MrHyDE::VFP1d2v<AD24>;
template class MrHyDE::VFP1d2v<AD32>;
#endif
