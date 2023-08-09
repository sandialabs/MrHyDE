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

#include "cdr.hpp"
using namespace MrHyDE;

// ========================================================================================
// ========================================================================================

template<class EvalT>
cdr<EvalT>::cdr(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "cdr";
  
  myvars.push_back("c");
  mybasistypes.push_back("HGRAD");
  
  //velFromNS = settings->sublist("Physics").get<bool>("Get velocity from navierstokes",false);
  //burgersflux = settings->sublist("Physics").get<bool>("Add Burgers",false);
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void cdr<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                          Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  // Functions
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"ip");
  functionManager->addFunction("diffusion",fs.get<string>("diffusion","1.0"),"ip");
  functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),"ip");
  functionManager->addFunction("density",fs.get<string>("density","1.0"),"ip");
  functionManager->addFunction("reaction",fs.get<string>("reaction","1.0"),"ip");
  functionManager->addFunction("xvel",fs.get<string>("xvel","1.0"),"ip");
  functionManager->addFunction("yvel",fs.get<string>("yvel","1.0"),"ip");
  functionManager->addFunction("zvel",fs.get<string>("zvel","1.0"),"ip");
  functionManager->addFunction("SUPG tau",fs.get<string>("SUPG tau","0.0"),"ip");
  
  functionManager->addFunction("diffusion",fs.get<string>("diffusion","1.0"),"side ip");
  functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),"side ip");
  
  //regParam = settings->sublist("Analysis").sublist("ROL").get<ScalarT>("regularization parameter",1.e-6);
  //moveVort = settings->sublist("Physics").get<bool>("moving vortices",true);
  //finTime = settings->sublist("Solver").get<ScalarT>("finaltime",1.0);
  //data_noise_std = settings->sublist("Analysis").get("Additive Normal Noise Standard Dev",0.0);
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void cdr<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int c_basis_num = wkset->usebasis[cnum];
  auto basis = wkset->basis[c_basis_num];
  auto basis_grad = wkset->basis_grad[c_basis_num];
  auto wts = wkset->wts;
  
  Vista<EvalT> source, diff, cp, rho, reax, xvel, yvel, zvel, tau;
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    diff = functionManager->evaluate("diffusion","ip");
    cp = functionManager->evaluate("specific heat","ip");
    rho = functionManager->evaluate("density","ip");
    reax = functionManager->evaluate("reaction","ip");
    xvel = functionManager->evaluate("xvel","ip");
    yvel = functionManager->evaluate("yvel","ip");
    zvel = functionManager->evaluate("zvel","ip");
    tau = functionManager->evaluate("SUPG tau","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto C = wkset->getSolutionField("c");
  auto dC_dt = wkset->getSolutionField("c_t");
  View_EvalT2 dC_dx, dC_dy, dC_dz;
  dC_dx = wkset->getSolutionField("grad(c)[x]");
  if (spaceDim > 1) {
    dC_dy = wkset->getSolutionField("grad(c)[y]");
  }
  if (spaceDim > 2) {
    dC_dz = wkset->getSolutionField("grad(c)[z]");
  }
  auto off = Kokkos::subview(wkset->offsets, cnum, Kokkos::ALL());
  auto res = wkset->res;
  
  if (spaceDim == 1) {
    parallel_for("cdr volume resid 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT f = (dC_dt(elem,pt) + xvel(elem,pt)*dC_dx(elem,pt) + reax(elem,pt) - source(elem,pt))*wts(elem,pt);
        EvalT Fx = 1.0/(rho(elem,pt)*cp(elem,pt))*diff(elem,pt)*dC_dx(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
        }
      }
    });
  }
  else if (spaceDim == 2) {
    parallel_for("cdr volume resid 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT f = (dC_dt(elem,pt) + xvel(elem,pt)*dC_dx(elem,pt) + yvel(elem,pt)*dC_dy(elem,pt) + reax(elem,pt) - source(elem,pt))*wts(elem,pt);
        EvalT Fx = 1.0/(rho(elem,pt)*cp(elem,pt))*diff(elem,pt)*dC_dx(elem,pt)*wts(elem,pt);
        EvalT Fy = 1.0/(rho(elem,pt)*cp(elem,pt))*diff(elem,pt)*dC_dy(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
        }
      }
    });
  }
  else if (spaceDim == 3) {
    parallel_for("cdr volume resid 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT f = (dC_dt(elem,pt) + xvel(elem,pt)*dC_dx(elem,pt) + yvel(elem,pt)*dC_dy(elem,pt) + zvel(elem,pt)*dC_dz(elem,pt) + reax(elem,pt) - source(elem,pt))*wts(elem,pt);
        EvalT Fx = 1.0/(rho(elem,pt)*cp(elem,pt))*diff(elem,pt)*dC_dx(elem,pt)*wts(elem,pt);
        EvalT Fy = 1.0/(rho(elem,pt)*cp(elem,pt))*diff(elem,pt)*dC_dy(elem,pt)*wts(elem,pt);
        EvalT Fz = 1.0/(rho(elem,pt)*cp(elem,pt))*diff(elem,pt)*dC_dz(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
        }
      }
    });
  }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void cdr<EvalT>::boundaryResidual() {
  // not re-implemented yet
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void cdr<EvalT>::edgeResidual() {}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void cdr<EvalT>::computeFlux() {
  // not re-implemented yet
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void cdr<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;

  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "c") {
      cnum = i;
    }
  }
}

// ========================================================================================
// return the value of the stabilization parameter
// ========================================================================================

template<class EvalT>
template<class T>
T cdr<EvalT>::computeTau(const T & localdiff, const T & xvl, const T & yvl, const T & zvl, const ScalarT & h) const {
  
  ScalarT C1 = 4.0;
  ScalarT C2 = 2.0;
  int spaceDim = wkset->dimension;
  
  T nvel;
  if (spaceDim == 1)
    nvel = xvl*xvl;
  else if (spaceDim == 2)
    nvel = xvl*xvl + yvl*yvl;
  else if (spaceDim == 3)
    nvel = xvl*xvl + yvl*yvl + zvl*zvl;
  
  if (nvel > 1E-12)
    nvel = sqrt(nvel);
  
  return 4.0/(C1*localdiff/h/h + C2*(nvel)/h); //msconvdiff has a 1.0 instead of a 4.0 in the numerator
  
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

// Avoid redefining since ScalarT=AD if no AD
#ifndef MrHyDE_NO_AD
template class MrHyDE::cdr<ScalarT>;
#endif

// Custom AD type
template class MrHyDE::cdr<AD>;

// Standard built-in types
template class MrHyDE::cdr<AD2>;
template class MrHyDE::cdr<AD4>;
template class MrHyDE::cdr<AD8>;
template class MrHyDE::cdr<AD16>;
template class MrHyDE::cdr<AD18>;
template class MrHyDE::cdr<AD24>;
template class MrHyDE::cdr<AD32>;