/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "kuramotoSivashinsky.hpp"

using namespace MrHyDE;

// ========================================================================================
// ========================================================================================

template<class EvalT>
KuramotoSivashinsky<EvalT>::KuramotoSivashinsky(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "Kuramoto-Sivashinsky";
  myvars.push_back("u");
  myvars.push_back("w");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void KuramotoSivashinsky<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  functionManager = functionManager_;

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void KuramotoSivashinsky<EvalT>::volumeResidual() {
  
  int spacedim = wkset->dimension;
  
  // Get some information from the workset
  auto basis = wkset->getBasis("u");
  auto basis_grad = wkset->getBasisGrad("u");
  auto res = wkset->res;
  auto wts = wkset->wts;
  
  auto u = wkset->getSolutionField("u");
  auto dudt = wkset->getSolutionField("u_t");
  
  // Solves the first equation u_t + grad^2(w) + w + 1/2*|grad(u)|^2 = 0
  // i.e. (u_t,p) - (grad(w),grad(p)) + (w,p) + (1/2*|grad(u)|^2,p) = 0
  {
    auto off = subview(wkset->offsets,u_num,ALL());
    if (spacedim == 1) {
      auto dudx = wkset->getSolutionField("grad(u)[x]");
      auto w = wkset->getSolutionField("w");
      auto dwdx = wkset->getSolutionField("grad(w)[x]");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT gradu_sq = 0.5*dudx(elem,pt)*dudx(elem,pt);
          EvalT f = (dudt(elem,pt) + w(elem,pt) + gradu_sq)*wts(elem,pt);
          EvalT Fx = -(dwdx(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
          }
        }
      });
    }
    else if (spacedim == 2) {
      auto dudx = wkset->getSolutionField("grad(u)[x]");
      auto dudy = wkset->getSolutionField("grad(u)[y]");
      auto w = wkset->getSolutionField("w");
      auto dwdx = wkset->getSolutionField("grad(w)[x]");
      auto dwdy = wkset->getSolutionField("grad(w)[y]");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT gradu_sq = 0.5*(dudx(elem,pt)*dudx(elem,pt) + dudy(elem,pt)*dudy(elem,pt));
          EvalT f = (dudt(elem,pt) + w(elem,pt) + gradu_sq)*wts(elem,pt);
          EvalT Fx = -(dwdx(elem,pt))*wts(elem,pt);
          EvalT Fy = -(dwdy(elem,pt))*wts(elem,pt);
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    else if (spacedim == 3) {
      auto dudx = wkset->getSolutionField("grad(u)[x]");
      auto dudy = wkset->getSolutionField("grad(u)[y]");
      auto dudz = wkset->getSolutionField("grad(u)[z]");
      auto w = wkset->getSolutionField("w");
      auto dwdx = wkset->getSolutionField("grad(w)[x]");
      auto dwdy = wkset->getSolutionField("grad(w)[y]");
      auto dwdz = wkset->getSolutionField("grad(w)[z]");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT gradu_sq = 0.5*(dudx(elem,pt)*dudx(elem,pt) + dudy(elem,pt)*dudy(elem,pt) + dudz(elem,pt)*dudz(elem,pt));
          EvalT f = (dudt(elem,pt) + w(elem,pt) + gradu_sq)*wts(elem,pt);
          EvalT Fx = -(dwdx(elem,pt))*wts(elem,pt);
          EvalT Fy = -(dwdy(elem,pt))*wts(elem,pt);
          EvalT Fz = -(dwdz(elem,pt))*wts(elem,pt);
          
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
      auto dudx = wkset->getSolutionField("grad(u)[x]");
      auto w = wkset->getSolutionField("w");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT f = (w(elem,pt))*wts(elem,pt);
          EvalT Fx = (dudx(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
          }
        }
      });
    }
    else if (spacedim == 2) {
      auto dudx = wkset->getSolutionField("grad(u)[x]");
      auto dudy = wkset->getSolutionField("grad(u)[y]");
      auto w = wkset->getSolutionField("w");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT f = (w(elem,pt))*wts(elem,pt);
          EvalT Fx = (dudx(elem,pt))*wts(elem,pt);
          EvalT Fy = (dudy(elem,pt))*wts(elem,pt);
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    else if (spacedim == 3) {
      auto dudx = wkset->getSolutionField("grad(u)[x]");
      auto dudy = wkset->getSolutionField("grad(u)[y]");
      auto dudz = wkset->getSolutionField("grad(u)[z]");
      auto w = wkset->getSolutionField("w");
      parallel_for("Kuramoto-Sivashinsky volume resid",
                  RangePolicy<AssemblyExec>(0,wkset->numElem),
                  KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT f = (w(elem,pt))*wts(elem,pt);
          EvalT Fx = (dudx(elem,pt))*wts(elem,pt);
          EvalT Fy = (dudy(elem,pt))*wts(elem,pt);
          EvalT Fz = (dudz(elem,pt))*wts(elem,pt);
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
          }
        }
      });
    }
  }

}

template<class EvalT>
void KuramotoSivashinsky<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

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


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::KuramotoSivashinsky<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::KuramotoSivashinsky<AD>;

// Standard built-in types
template class MrHyDE::KuramotoSivashinsky<AD2>;
template class MrHyDE::KuramotoSivashinsky<AD4>;
template class MrHyDE::KuramotoSivashinsky<AD8>;
template class MrHyDE::KuramotoSivashinsky<AD16>;
template class MrHyDE::KuramotoSivashinsky<AD18>;
template class MrHyDE::KuramotoSivashinsky<AD24>;
template class MrHyDE::KuramotoSivashinsky<AD32>;
#endif
