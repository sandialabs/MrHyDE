/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "vlasov_fokker_planck_0d2v.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
VFP0d2v<EvalT>::VFP0d2v(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "VFP0d2v";

  // save spaceDim here because it is needed (potentially) before workset is finalized
  // TODO is this needed?

  spaceDim = dimension_;
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                            Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  // TODO not supported right now?
  functionManager->addFunction("source rho",fs.get<string>("source rho","0.0"),"ip");
  functionManager->addFunction("source rhoux",fs.get<string>("source rhoux","0.0"),"ip");
  functionManager->addFunction("source rhoE", fs.get<string>("source rhoE", "0.0"),"ip");
  if (spaceDim > 1) {
    functionManager->addFunction("source rhouy",fs.get<string>("source rhouy","0.0"),"ip");
  }
  if (spaceDim > 2) {
    functionManager->addFunction("source rhouz",fs.get<string>("source rhouz","0.0"),"ip");
  }

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::volumeResidual() {
  
  Vista<EvalT> source_rho, source_rhoux, source_rhouy, source_rhouz, source_rhoE;

  auto wts = wkset->wts;
  auto res = wkset->res;

  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::boundaryResidual() {
  
  auto bcs = wkset->var_bcs;
  
  //int cside = wkset->currentside;
  
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::computeFlux() {

}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;

}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::VFP0d2v<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::VFP0d2v<AD>;

// Standard built-in types
template class MrHyDE::VFP0d2v<AD2>;
template class MrHyDE::VFP0d2v<AD4>;
template class MrHyDE::VFP0d2v<AD8>;
template class MrHyDE::VFP0d2v<AD16>;
template class MrHyDE::VFP0d2v<AD18>;
template class MrHyDE::VFP0d2v<AD24>;
template class MrHyDE::VFP0d2v<AD32>;
#endif
