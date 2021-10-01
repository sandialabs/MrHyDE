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

#include "kuramotoSivashinsky.hpp"

using namespace MrHyDE;

// ========================================================================================
// ========================================================================================

KuramotoSivashinsky::KuramotoSivashinsky(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_)
  : physicsbase(settings, isaux_)
{
  
  label = "Kuramoto-Sivashinsky";
  myvars.push_back("u");
  myvars.push_back("w");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");

}

// ========================================================================================
// ========================================================================================

void KuramotoSivashinsky::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager> & functionManager_) {
  functionManager = functionManager_;

}

// ========================================================================================
// ========================================================================================

void KuramotoSivashinsky::volumeResidual() {
  
  int spacedim = wkset->dimension;
  
  // Get some information from the workset
  auto basis = wkset->getBasis("u");
  auto basis_grad = wkset->getBasisGrad("u");
  auto res = wkset->res;
  auto wts = wkset->wts;
  
  auto u = wkset->getData("u");
  auto dudt = wkset->getData("u_t");
  
  // Solves the first equation u_t + grad^2(w) + w + 1/2*|grad(u)|^2 = 0
  // i.e. (u_t,p) - (grad(w),grad(p)) + (w,p) + (1/2*|grad(u)|^2,p) = 0
  {
    auto off = subview(wkset->offsets,u_num,ALL());
    if (spacedim == 1) {
      auto dudx = wkset->getData("grad(u)[x]");
      auto w = wkset->getData("w");
      auto dwdx = wkset->getData("grad(w)[x]");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD gradu_sq = 0.5*dudx(elem,pt)*dudx(elem,pt);
          AD f = (dudt(elem,pt) + w(elem,pt) + gradu_sq)*wts(elem,pt);
          AD Fx = -(dwdx(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
          }
        }
      });
    }
    else if (spacedim == 2) {
      auto dudx = wkset->getData("grad(u)[x]");
      auto dudy = wkset->getData("grad(u)[y]");
      auto w = wkset->getData("w");
      auto dwdx = wkset->getData("grad(w)[x]");
      auto dwdy = wkset->getData("grad(w)[y]");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD gradu_sq = 0.5*(dudx(elem,pt)*dudx(elem,pt) + dudy(elem,pt)*dudy(elem,pt));
          AD f = (dudt(elem,pt) + w(elem,pt) + gradu_sq)*wts(elem,pt);
          AD Fx = -(dwdx(elem,pt))*wts(elem,pt);
          AD Fy = -(dwdy(elem,pt))*wts(elem,pt);
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    else if (spacedim == 3) {
      auto dudx = wkset->getData("grad(u)[x]");
      auto dudy = wkset->getData("grad(u)[y]");
      auto dudz = wkset->getData("grad(u)[z]");
      auto w = wkset->getData("w");
      auto dwdx = wkset->getData("grad(w)[x]");
      auto dwdy = wkset->getData("grad(w)[y]");
      auto dwdz = wkset->getData("grad(w)[z]");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD gradu_sq = 0.5*(dudx(elem,pt)*dudx(elem,pt) + dudy(elem,pt)*dudy(elem,pt) + dudz(elem,pt)*dudz(elem,pt));
          AD f = (dudt(elem,pt) + w(elem,pt) + gradu_sq)*wts(elem,pt);
          AD Fx = -(dwdx(elem,pt))*wts(elem,pt);
          AD Fy = -(dwdy(elem,pt))*wts(elem,pt);
          AD Fz = -(dwdz(elem,pt))*wts(elem,pt);
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
          }
        }
      });
    }
  }
  
  // Solves the second equation grad^2(u) - w = 0
  // i.e. (\grad u, \grad v) + (w, v) = 0
  {
    auto off = subview(wkset->offsets,w_num,ALL());
    if (spacedim == 1) {
      auto dudx = wkset->getData("grad(u)[x]");
      auto w = wkset->getData("w");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD f = (w(elem,pt))*wts(elem,pt);
          AD Fx = (dudx(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
          }
        }
      });
    }
    else if (spacedim == 2) {
      auto dudx = wkset->getData("grad(u)[x]");
      auto dudy = wkset->getData("grad(u)[y]");
      auto w = wkset->getData("w");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD f = (w(elem,pt))*wts(elem,pt);
          AD Fx = (dudx(elem,pt))*wts(elem,pt);
          AD Fy = (dudy(elem,pt))*wts(elem,pt);
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    else if (spacedim == 3) {
      auto dudx = wkset->getData("grad(u)[x]");
      auto dudy = wkset->getData("grad(u)[y]");
      auto dudz = wkset->getData("grad(u)[z]");
      auto w = wkset->getData("w");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD f = (w(elem,pt))*wts(elem,pt);
          AD Fx = (dudx(elem,pt))*wts(elem,pt);
          AD Fy = (dudy(elem,pt))*wts(elem,pt);
          AD Fz = (dudz(elem,pt))*wts(elem,pt);
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
          }
        }
      });
    }
  }

}

void KuramotoSivashinsky::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  u_num = -1;
  w_num = -1;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "u")
      u_num = i;
    else if (varlist[i] == "w")
      w_num = i;
  }
}