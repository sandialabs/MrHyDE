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

#include "maxwell.hpp"
using namespace MrHyDE;

maxwell::maxwell(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_)
  : physicsbase(settings, isaux_)
{
  
  label = "maxwell";
  
  myvars.push_back("E");
  myvars.push_back("B");
  mybasistypes.push_back("HCURL");
  mybasistypes.push_back("HDIV");
  
  useLeapFrog = settings->sublist("Physics").get<bool>("use leap frog",false);
}

// ========================================================================================
// ========================================================================================

void maxwell::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("current x",fs.get<string>("current x","0.0"),"ip");
  functionManager->addFunction("current y",fs.get<string>("current y","0.0"),"ip");
  functionManager->addFunction("current z",fs.get<string>("current z","0.0"),"ip");
  functionManager->addFunction("mu",fs.get<string>("permeability","1.0"),"ip");
  functionManager->addFunction("epsilon",fs.get<string>("permittivity","1.0"),"ip");
  functionManager->addFunction("sigma",fs.get<string>("conductivity","0.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void maxwell::volumeResidual() {
  
  
  int E_basis = wkset->usebasis[Enum];
  int B_basis = wkset->usebasis[Bnum];
  
  Vista mu, epsilon, sigma;
  Vista current_x, current_y, current_z;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    current_x = functionManager->evaluate("current x","ip");
    current_y = functionManager->evaluate("current y","ip");
    current_z = functionManager->evaluate("current z","ip");
    mu = functionManager->evaluate("mu","ip");
    epsilon = functionManager->evaluate("epsilon","ip");
    sigma = functionManager->evaluate("sigma","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  int stage = wkset->current_stage_KV(0);
  ScalarT Ewt = 1.0, Bwt = 1.0;
  
  if (useLeapFrog) {
    if (stage == 0) {
      Ewt = 0.0;
    }
    else if (stage == 1) {
      Bwt = 0.0;
    }
  }
  
  {
    // (dB/dt + curl E,V) = 0
    //if (stage == 0) {
    auto basis = wkset->basis[B_basis];
    auto dBx_dt = wkset->getData("B_t[x]");
    auto dBy_dt = wkset->getData("B_t[y]");
    auto dBz_dt = wkset->getData("B_t[z]");
    auto curlE_x = wkset->getData("curl(E)[x]");
    auto curlE_y = wkset->getData("curl(E)[y]");
    auto curlE_z = wkset->getData("curl(E)[z]");
    
    auto off = subview(wkset->offsets, Bnum, ALL());
    auto wts = wkset->wts;
    auto res = wkset->res;
    
    parallel_for("Maxwells B volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD f0 = (dBx_dt(elem,pt) + Bwt*curlE_x(elem,pt))*wts(elem,pt);
        AD f1 = (dBy_dt(elem,pt) + Bwt*curlE_y(elem,pt))*wts(elem,pt);
        AD f2 = (dBz_dt(elem,pt) + Bwt*curlE_z(elem,pt))*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
          res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
          res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
        }
      }
    });
    //}
  }
  
  {
    // (eps*dE/dt,V) - (1/mu B, curl V) + (sigma E,V) = -(current,V)
    // Rewritten as: (eps*dEdt + sigma E + current, V) - (1/mu B, curl V) = 0
    
    //if (stage == 1) {
    auto basis = wkset->basis[E_basis];
    auto basis_curl = wkset->basis_curl[E_basis];
    auto dEx_dt = wkset->getData("E_t[x]");
    auto dEy_dt = wkset->getData("E_t[y]");
    auto dEz_dt = wkset->getData("E_t[z]");
    auto Bx = wkset->getData("B[x]");
    auto By = wkset->getData("B[y]");
    auto Bz = wkset->getData("B[z]");
    auto Ex = wkset->getData("E[x]");
    auto Ey = wkset->getData("E[y]");
    auto Ez = wkset->getData("E[z]");
    auto off = subview(wkset->offsets, Enum, ALL());
    auto wts = wkset->wts;
    auto res = wkset->res;
    
    parallel_for("Maxwells E volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD f0 = (epsilon(elem,pt)*dEx_dt(elem,pt) + Ewt*(sigma(elem,pt)*Ex(elem,pt) + current_x(elem,pt)))*wts(elem,pt);
        AD f1 = (epsilon(elem,pt)*dEy_dt(elem,pt) + Ewt*(sigma(elem,pt)*Ey(elem,pt) + current_y(elem,pt)))*wts(elem,pt);
        AD f2 = (epsilon(elem,pt)*dEz_dt(elem,pt) + Ewt*(sigma(elem,pt)*Ez(elem,pt) + current_z(elem,pt)))*wts(elem,pt);
        AD c0 = - Ewt/mu(elem,pt)*Bx(elem,pt)*wts(elem,pt);
        AD c1 = - Ewt/mu(elem,pt)*By(elem,pt)*wts(elem,pt);
        AD c2 = - Ewt/mu(elem,pt)*Bz(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f0*basis(elem,dof,pt,0) + c0*basis_curl(elem,dof,pt,0);
          res(elem,off(dof)) += f1*basis(elem,dof,pt,1) + c1*basis_curl(elem,dof,pt,1);
          res(elem,off(dof)) += f2*basis(elem,dof,pt,2) + c2*basis_curl(elem,dof,pt,2);
        }
      }
    });
    //}
  }
  
}

// ========================================================================================
// ========================================================================================

void maxwell::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;
 
  vector<string> varlist = wkset->varlist;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "E")
      Enum = i;
    if (varlist[i] == "B")
      Bnum = i;
  }
}
