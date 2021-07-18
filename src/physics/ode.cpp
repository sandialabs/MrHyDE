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

ODE::ODE(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_)
  : physicsbase(settings, isaux_)
{
  
  label = "ode";
  myvars.push_back("q");
  mybasistypes.push_back("HVOL");
  
}

// ========================================================================================
// ========================================================================================

void ODE::defineFunctions(Teuchos::ParameterList & fs,
                          Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  functionManager->addFunction("ODE source",fs.get<string>("ODE source","0.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void ODE::volumeResidual() {
  
  Vista source;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("ODE source","ip");
  }
    
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
  auto basis = wkset->getBasis("q");
  auto res = wkset->getResidual();
  auto off = wkset->getOffsets("q");
  auto dqdt = wkset->getData("q_t");
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
