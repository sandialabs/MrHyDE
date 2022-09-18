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

#include "shallowice.hpp"
using namespace MrHyDE;

// ========================================================================================
// ========================================================================================

shallowice::shallowice(Teuchos::ParameterList & settings, const int & dimension_)
  : physicsbase(settings, dimension_)
{
  
  label = "shallowice";
  
  myvars.push_back("s");
  mybasistypes.push_back("HGRAD");
  
  //velFromNS = settings->sublist("Physics").get<bool>("Get velocity from navierstokes",false);
  //burgersflux = settings->sublist("Physics").get<bool>("Add Burgers",false);
}

// ========================================================================================
// ========================================================================================

void shallowice::defineFunctions(Teuchos::ParameterList & fs,
                                 Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  // Functions
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"ip");
  functionManager->addFunction("diffusion",fs.get<string>("diffusion","1.0"),"ip");
  functionManager->addFunction("diffusion",fs.get<string>("diffusion","1.0"),"side ip");
  
}

// ========================================================================================
// ========================================================================================

void shallowice::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int s_basis_num = wkset->usebasis[snum];
  auto basis = wkset->basis[s_basis_num];
  auto basis_grad = wkset->basis_grad[s_basis_num];
  auto wts = wkset->wts;
  
  Vista source, diff;
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    diff = functionManager->evaluate("diffusion","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  
  auto S = wkset->getSolutionField("s");
  auto dS_dt = wkset->getSolutionField("s_t");



  View_AD2 dS_dx, dS_dy, dS_dz;
  dS_dx = wkset->getSolutionField("grad(s)[x]");
  if (spaceDim > 1) {
    dS_dy = wkset->getSolutionField("grad(s)[y]");
  }
  auto off = Kokkos::subview(wkset->offsets, snum, Kokkos::ALL());
  auto res = wkset->res;
  
  if (spaceDim == 1) {
    parallel_for("shallowice volume resid 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD f = (dS_dt(elem,pt) - source(elem,pt))*wts(elem,pt);
        AD Fx = diff(elem,pt)*dS_dx(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
        }
      }
    });
  }
  else if (spaceDim == 2) {
    parallel_for("shallowice volume resid 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD f = (dS_dt(elem,pt) - source(elem,pt))*wts(elem,pt);
        AD Fx = diff(elem,pt)*dS_dx(elem,pt)*wts(elem,pt);
        AD Fy = diff(elem,pt)*dS_dy(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
        }
      }
    });
  }
  
  
}

// ========================================================================================
// ========================================================================================

void shallowice::boundaryResidual() {
  // not re-implemented yet
}

// ========================================================================================
// ========================================================================================

void shallowice::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;

  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "s") {
      snum = i;
    }
  }
}
