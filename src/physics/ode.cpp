/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "ode.hpp"

// ========================================================================================
// ========================================================================================

ODE::ODE(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  label = "ode";
  myvars.push_back("q");
  mybasistypes.push_back("HVOL");
  
}

// ========================================================================================
// ========================================================================================

void ODE::defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                          Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  Teuchos::ParameterList fs = settings->sublist("Functions");
  functionManager->addFunction("ODE source",fs.get<string>("ODE source","0.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void ODE::volumeResidual() {
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("ODE source","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  int q_basis = wkset->usebasis[qnum];
  basis = wkset->basis[q_basis];
  
  // Simply solves q_dot = f(q,t)
  auto off = Kokkos::subview(offsets,qnum,Kokkos::ALL());
  parallel_for("ODE volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
    res(e,off(0)) += sol_dot(e,qnum,0,0) - source(e,0);
  });
}

// ========================================================================================
// ========================================================================================

void ODE::boundaryResidual() {
  
}

// ========================================================================================
// ========================================================================================

void ODE::edgeResidual() {}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void ODE::computeFlux() {
}

// ========================================================================================
// ========================================================================================

void ODE::setVars(vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "q") {
      qnum = i;
    }
  }
}
