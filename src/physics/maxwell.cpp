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

template<class EvalT>
maxwell<EvalT>::maxwell(Teuchos::ParameterList & settings, const int & dimension_)
: PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "maxwell";
  
  spaceDim = dimension_;
  myvars.push_back("E");
  myvars.push_back("B");
  
  mybasistypes.push_back("HCURL");
  if (spaceDim == 2) {
    mybasistypes.push_back("HVOL");
  }
  else if (spaceDim == 3) {
    mybasistypes.push_back("HDIV");
  }
  
  useLeapFrog = settings.get<bool>("use leap frog",false);
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("current x",fs.get<string>("current x","0.0"),"ip");
  functionManager->addFunction("current y",fs.get<string>("current y","0.0"),"ip");
  functionManager->addFunction("current z",fs.get<string>("current z","0.0"),"ip");
  functionManager->addFunction("mu",fs.get<string>("permeability","1.0"),"ip");
  functionManager->addFunction("refractive index",fs.get<string>("refractive index","1.0"),"ip");
  functionManager->addFunction("epsilon",fs.get<string>("permittivity","1.0"),"ip");
  functionManager->addFunction("sigma",fs.get<string>("conductivity","0.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell<EvalT>::volumeResidual() {
  
  
  int E_basis = wkset->usebasis[Enum];
  int B_basis = wkset->usebasis[Bnum];
  
  Vista<EvalT> mu, epsilon, sigma, rindex;
  Vista<EvalT> current_x, current_y, current_z;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    current_x = functionManager->evaluate("current x","ip");
    current_y = functionManager->evaluate("current y","ip");
    if (spaceDim > 2) {
      current_z = functionManager->evaluate("current z","ip");
    }
    mu = functionManager->evaluate("mu","ip");
    epsilon = functionManager->evaluate("epsilon","ip");
    rindex = functionManager->evaluate("refractive index","ip");
    sigma = functionManager->evaluate("sigma","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  int stage = wkset->current_stage;
  
  {
    if (spaceDim == 2) {
      // (dB/dt + curl E,V) = 0
      
      auto basis = wkset->basis[B_basis];
      auto dB_dt = wkset->getSolutionField("B_t");
      
      auto off = subview(wkset->offsets, Bnum, ALL());
      auto wts = wkset->wts;
      auto res = wkset->res;
      
      if (useLeapFrog) {
        if (stage == 0) {
          auto curlE = wkset->getSolutionField("curl(E)[x]");
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT f0 = (dB_dt(elem,pt) + curlE(elem,pt))*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
              }
            }
          });
        }
        else {
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT f0 = dB_dt(elem,pt)*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
              }
            }
          });
        }
      }
      else {
        auto curlE = wkset->getSolutionField("curl(E)[x]");
        parallel_for("Maxwells B volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT f0 = (dB_dt(elem,pt) + curlE(elem,pt))*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
            }
          }
        });
      }
    }
    else if (spaceDim == 3) {
      
      // (dB/dt + curl E,V) = 0
      
      auto off = subview(wkset->offsets, Bnum, ALL());
      auto wts = wkset->wts;
      auto res = wkset->res;
      auto basis = wkset->basis[B_basis];
      auto dBx_dt = wkset->getSolutionField("B_t[x]");
      auto dBy_dt = wkset->getSolutionField("B_t[y]");
      auto dBz_dt = wkset->getSolutionField("B_t[z]");
      
      if (useLeapFrog) {
        if (stage == 0) {
          auto curlE_x = wkset->getSolutionField("curl(E)[x]");
          auto curlE_y = wkset->getSolutionField("curl(E)[y]");
          auto curlE_z = wkset->getSolutionField("curl(E)[z]");
          
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT f0 = (dBx_dt(elem,pt) + curlE_x(elem,pt))*wts(elem,pt);
              EvalT f1 = (dBy_dt(elem,pt) + curlE_y(elem,pt))*wts(elem,pt);
              EvalT f2 = (dBz_dt(elem,pt) + curlE_z(elem,pt))*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
              }
            }
          });
        }
        else {
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT f0 = dBx_dt(elem,pt)*wts(elem,pt);
              EvalT f1 = dBy_dt(elem,pt)*wts(elem,pt);
              EvalT f2 = dBz_dt(elem,pt)*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
              }
            }
          });
        }
      }
      else {
        auto curlE_x = wkset->getSolutionField("curl(E)[x]");
        auto curlE_y = wkset->getSolutionField("curl(E)[y]");
        auto curlE_z = wkset->getSolutionField("curl(E)[z]");
        
        parallel_for("Maxwells B volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT f0 = (dBx_dt(elem,pt) + curlE_x(elem,pt))*wts(elem,pt);
            EvalT f1 = (dBy_dt(elem,pt) + curlE_y(elem,pt))*wts(elem,pt);
            EvalT f2 = (dBz_dt(elem,pt) + curlE_z(elem,pt))*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
              res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
              res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
            }
          }
        });
      }
    }
  }
  
  {
    // (eps*dE/dt,V) - (1/mu B, curl V) + (sigma E,V) = -(current,V)
    // Rewritten as: (eps*dEdt + sigma E + current, V) - (1/mu B, curl V) = 0
    
    if (spaceDim == 2) {
      if (!useLeapFrog || stage == 1) {
        auto basis = wkset->basis[E_basis];
        auto basis_curl = wkset->basis_curl[E_basis];
        
        auto dEx_dt = wkset->getSolutionField("E_t[x]");
        auto dEy_dt = wkset->getSolutionField("E_t[y]");
        auto B = wkset->getSolutionField("B");
        auto Ex = wkset->getSolutionField("E[x]");
        auto Ey = wkset->getSolutionField("E[y]");
        auto off = subview(wkset->offsets, Enum, ALL());
        auto wts = wkset->wts;
        auto res = wkset->res;
        
        parallel_for("Maxwells E volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT f0 = (epsilon(elem,pt)*rindex(elem,pt)*rindex(elem,pt)*dEx_dt(elem,pt) + (sigma(elem,pt)*Ex(elem,pt) + current_x(elem,pt)))*wts(elem,pt);
            EvalT f1 = (epsilon(elem,pt)*rindex(elem,pt)*rindex(elem,pt)*dEy_dt(elem,pt) + (sigma(elem,pt)*Ey(elem,pt) + current_y(elem,pt)))*wts(elem,pt);
            EvalT c0 = -1.0/mu(elem,pt)*B(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0) + c0*basis_curl(elem,dof,pt,0) + f1*basis(elem,dof,pt,1);
            }
          }
        });
      }
    }
    else if (spaceDim == 3) {
      
      if (!useLeapFrog || stage == 1) {
        auto basis = wkset->basis[E_basis];
        auto basis_curl = wkset->basis_curl[E_basis];
        auto dEx_dt = wkset->getSolutionField("E_t[x]");
        auto dEy_dt = wkset->getSolutionField("E_t[y]");
        auto dEz_dt = wkset->getSolutionField("E_t[z]");
        auto Bx = wkset->getSolutionField("B[x]");
        auto By = wkset->getSolutionField("B[y]");
        auto Bz = wkset->getSolutionField("B[z]");
        auto Ex = wkset->getSolutionField("E[x]");
        auto Ey = wkset->getSolutionField("E[y]");
        auto Ez = wkset->getSolutionField("E[z]");
        auto off = subview(wkset->offsets, Enum, ALL());
        auto wts = wkset->wts;
        auto res = wkset->res;
      
        parallel_for("Maxwells E volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT eps = epsilon(elem,pt);
            EvalT f0 = (rindex(elem,pt)*rindex(elem,pt)*dEx_dt(elem,pt) + 1.0/eps*(sigma(elem,pt)*Ex(elem,pt) + current_x(elem,pt)))*wts(elem,pt);
            EvalT f1 = (rindex(elem,pt)*rindex(elem,pt)*dEy_dt(elem,pt) + 1.0/eps*(sigma(elem,pt)*Ey(elem,pt) + current_y(elem,pt)))*wts(elem,pt);
            EvalT f2 = (rindex(elem,pt)*rindex(elem,pt)*dEz_dt(elem,pt) + 1.0/eps*(sigma(elem,pt)*Ez(elem,pt) + current_z(elem,pt)))*wts(elem,pt);
            
            EvalT c0 = -1.0/mu(elem,pt)*1.0/eps*Bx(elem,pt)*wts(elem,pt);
            EvalT c1 = -1.0/mu(elem,pt)*1.0/eps*By(elem,pt)*wts(elem,pt);
            EvalT c2 = -1.0/mu(elem,pt)*1.0/eps*Bz(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0) + c0*basis_curl(elem,dof,pt,0);
              res(elem,off(dof)) += f1*basis(elem,dof,pt,1) + c1*basis_curl(elem,dof,pt,1);
              res(elem,off(dof)) += f2*basis(elem,dof,pt,2) + c2*basis_curl(elem,dof,pt,2);
            }
          }
        });
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell<EvalT>::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  
  
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  
  if (spaceDim == 2) {
    View_Sc2 nx, ny;
    nx = wkset->getScalarField("n[x]");
    ny = wkset->getScalarField("n[y]");
    
    //double gamma = 0.0;
    if (bcs(Bnum,cside) == "Neumann") { // Really ABC
      // Computes -nxnxE in B equation
      
      parallel_for("maxwell bndry resid ABC",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
    
      });
    }
    
  }
  else if (spaceDim == 3) {
    View_Sc2 nx, ny, nz;
    nx = wkset->getScalarField("n[x]");
    ny = wkset->getScalarField("n[y]");
    nz = wkset->getScalarField("n[z]");
    auto Ex = wkset->getSolutionField("E[x]");
    auto Ey = wkset->getSolutionField("E[y]");
    auto Ez = wkset->getSolutionField("E[z]");
    auto off = subview(wkset->offsets, Enum, ALL());
    auto basis = wkset->basis_side[wkset->usebasis[Enum]];
    
    double gamma = -0.9944;
    if (bcs(Bnum,cside) == "Neumann") { // Really ABC
      // Contributes -<nxnxE,V> along boundary in B equation
      
      parallel_for("maxwell bndry resid ABC",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
    
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT nce_x = ny(elem,pt)*Ez(elem,pt) - nz(elem,pt)*Ey(elem,pt);
          EvalT nce_y = nz(elem,pt)*Ex(elem,pt) - nx(elem,pt)*Ez(elem,pt);
          EvalT nce_z = nx(elem,pt)*Ey(elem,pt) - ny(elem,pt)*Ex(elem,pt);
          EvalT c0 = -(1.0+gamma)*(ny(elem,pt)*nce_z - nz(elem,pt)*nce_y)*wts(elem,pt);
          EvalT c1 = -(1.0+gamma)*(nz(elem,pt)*nce_x - nx(elem,pt)*nce_z)*wts(elem,pt);
          EvalT c2 = -(1.0+gamma)*(nx(elem,pt)*nce_y - ny(elem,pt)*nce_x)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += c0*basis(elem,dof,pt,0) + c1*basis(elem,dof,pt,1) + c2*basis(elem,dof,pt,2);
          }
        }
      });
      
      
      /*
      auto Bx = wkset->getSolutionField("B[x]");
      auto By = wkset->getSolutionField("B[y]");
      auto Bz = wkset->getSolutionField("B[z]");
      parallel_for("maxwell bndry resid ABC",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
    
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          
          EvalT nce_x = ny(elem,pt)*Bz(elem,pt) - nz(elem,pt)*By(elem,pt);
          EvalT nce_y = nz(elem,pt)*Bx(elem,pt) - nx(elem,pt)*Bz(elem,pt);
          EvalT nce_z = nx(elem,pt)*By(elem,pt) - ny(elem,pt)*Bx(elem,pt);
          EvalT c0 = nce_x*wts(elem,pt);
          EvalT c1 = nce_y*wts(elem,pt);
          EvalT c2 = nce_z*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += c0*basis(elem,dof,pt,0) + c1*basis(elem,dof,pt,1) + c2*basis(elem,dof,pt,2);
          }
        }
      });
       */
       
    }
    
  }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
  
  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "E")
      Enum = i;
    if (varlist[i] == "B")
      Bnum = i;
    //if (varlist[i] == "E2")
    //  E2num = i;
    //if (varlist[i] == "B2")
    //  B2num = i;
  }
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::maxwell<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::maxwell<AD>;

// Standard built-in types
template class MrHyDE::maxwell<AD2>;
template class MrHyDE::maxwell<AD4>;
template class MrHyDE::maxwell<AD8>;
template class MrHyDE::maxwell<AD16>;
template class MrHyDE::maxwell<AD18>;
template class MrHyDE::maxwell<AD24>;
template class MrHyDE::maxwell<AD32>;
#endif
