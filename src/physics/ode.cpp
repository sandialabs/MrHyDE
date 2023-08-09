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

#include "ode.hpp"
#include "vista.hpp"

using namespace MrHyDE;

// ========================================================================================
// ========================================================================================

template<class EvalT>
ODE<EvalT>::ODE(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "ode";
  myvars.push_back("q");
  mybasistypes.push_back("HVOL");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void ODE<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                          Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  functionManager->addFunction("ODE source",fs.get<string>("ODE source","0.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void ODE<EvalT>::volumeResidual() {
  
  Vista<EvalT> source;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("ODE source","ip");
  }
    
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
  auto basis = wkset->getBasis("q");
  auto res = wkset->getResidual();
  auto off = wkset->getOffsets("q");
  auto dqdt = wkset->getSolutionField("q_t");
  auto wts = wkset->wts;
  
  // Simply solves q_dot = f(q,t)
  parallel_for("ODE volume resid",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int e ) {
    for (size_type pt=0; pt<wts.extent(1); ++pt) {
      res(e,off(0)) += (dqdt(e,pt) - source(e,pt))*wts(e,pt);
    }
  });
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

// Avoid redefining since ScalarT=AD if no AD
#ifndef MrHyDE_NO_AD
template class MrHyDE::ODE<ScalarT>;
#endif

// Custom AD type
template class MrHyDE::ODE<AD>;

// Standard built-in types
template class MrHyDE::ODE<AD2>;
template class MrHyDE::ODE<AD4>;
template class MrHyDE::ODE<AD8>;
template class MrHyDE::ODE<AD16>;
template class MrHyDE::ODE<AD18>;
template class MrHyDE::ODE<AD24>;
template class MrHyDE::ODE<AD32>;
