/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "shallowice.hpp"
using namespace MrHyDE;

// ========================================================================================
// ========================================================================================

template<class EvalT>
shallowice<EvalT>::shallowice(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "shallowice";
  
  myvars.push_back("s");
  mybasistypes.push_back("HGRAD");
  
  //velFromNS = settings->sublist("Physics").get<bool>("Get velocity from navierstokes",false);
  //burgersflux = settings->sublist("Physics").get<bool>("Add Burgers",false);
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowice<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                 Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  // Functions
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"ip");
  functionManager->addFunction("diffusion",fs.get<string>("diffusion","1.0"),"ip");
  functionManager->addFunction("diffusion",fs.get<string>("diffusion","1.0"),"side ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowice<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int s_basis_num = wkset->usebasis[snum];
  auto basis = wkset->basis[s_basis_num];
  auto basis_grad = wkset->basis_grad[s_basis_num];
  auto wts = wkset->wts;
  
  Vista<EvalT> source, diff;
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    diff = functionManager->evaluate("diffusion","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  
  auto S = wkset->getSolutionField("s");
  auto dS_dt = wkset->getSolutionField("s_t");



  View_EvalT2 dS_dx, dS_dy, dS_dz;
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
        EvalT f = (dS_dt(elem,pt) - source(elem,pt))*wts(elem,pt);
        EvalT Fx = diff(elem,pt)*dS_dx(elem,pt)*wts(elem,pt);
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
        EvalT f = (dS_dt(elem,pt) - source(elem,pt))*wts(elem,pt);
        EvalT Fx = diff(elem,pt)*dS_dx(elem,pt)*wts(elem,pt);
        EvalT Fy = diff(elem,pt)*dS_dy(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
        }
      }
    });
  }
  
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowice<EvalT>::boundaryResidual() {
  // not re-implemented yet
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowice<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;

  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "s") {
      snum = i;
    }
  }
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::shallowice<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::shallowice<AD>;

// Standard built-in types
template class MrHyDE::shallowice<AD2>;
template class MrHyDE::shallowice<AD4>;
template class MrHyDE::shallowice<AD8>;
template class MrHyDE::shallowice<AD16>;
template class MrHyDE::shallowice<AD18>; // AquiEEP_merge
template class MrHyDE::shallowice<AD24>;
template class MrHyDE::shallowice<AD32>;
#endif
