/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "rosenbluth_potential.hpp"
using namespace MrHyDE;

template<class EvalT>
rosenbluth<EvalT>::rosenbluth(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
    include_Heqn = false;
    include_Geqn = false;
      
    if (settings.isParameter("active variables")) {
      string active = settings.get<string>("active variables");
      std::size_t foundE = active.find("Hvars");
      if (foundE!=std::string::npos) {
        include_Heqn = true;
      }
      std::size_t foundB = active.find("Gvars");
      if (foundB!=std::string::npos) {
        include_Geqn = true;
      }
    }
    else { // maybe throw an error here
      include_Heqn = true;
      include_Geqn = true;
    }
    
    // Standard data
    if (include_Heqn) {
      label = "rosenbluth";
      myvars.push_back("rH_H");
      mybasistypes.push_back("HGRAD");
      
      myvars.push_back("rH_C");
      mybasistypes.push_back("HGRAD");
      
      myvars.push_back("rH_G");
      mybasistypes.push_back("HGRAD");
      
      myvars.push_back("rH_E");
      mybasistypes.push_back("HGRAD");
    }
    
    if (include_Geqn) {
      
      myvars.push_back("rG_H");
      mybasistypes.push_back("HGRAD");
      
      myvars.push_back("rG_C");
      mybasistypes.push_back("HGRAD");
      
      myvars.push_back("rG_G");
      mybasistypes.push_back("HGRAD");
      
      myvars.push_back("rG_E");
      mybasistypes.push_back("HGRAD");
    }
    
    spaceDim = 0;
    velDim = 2;
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void rosenbluth<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                             Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  
  if (include_Heqn) {
    
    functionManager->addFunction("source_rH_H",fs.get<string>("source_H_H","8.0*pi*H"),"ip");
    functionManager->addFunction("source_rH_C",fs.get<string>("source_H_H","8.0*pi*C"),"ip");
    functionManager->addFunction("source_rH_G",fs.get<string>("source_H_H","8.0*pi*G"),"ip");
    functionManager->addFunction("source_rH_E",fs.get<string>("source_H_H","8.0*pi*E"),"ip");
  }
  
  if (include_Geqn) {
    
    functionManager->addFunction("source_rG_H",fs.get<string>("source_rH_H","-1.0*rH_H"),"ip");
    functionManager->addFunction("source_rG_C",fs.get<string>("source_rH_C","-1.0*rH_C"),"ip");
    functionManager->addFunction("source_rG_G",fs.get<string>("source_rH_G","-1.0*rH_G"),"ip");
    functionManager->addFunction("source_rG_E",fs.get<string>("source_rH_E","-1.0*rH_E"),"ip");
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void rosenbluth<EvalT>::volumeResidual() {
  
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  if (include_Heqn) {
    Vista<EvalT> source_rhh, source_rhc, source_rhg, source_rhe;
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      source_rhh = functionManager->evaluate("source_rH_H","ip");
      source_rhc = functionManager->evaluate("source_rH_C","ip");
      source_rhg = functionManager->evaluate("source_rH_G","ip");
      source_rhe = functionManager->evaluate("source_rH_E","ip");
    }
    
    // Laplace w.r.t velocity for rH_H
    {
      
      Teuchos::TimeMonitor resideval(*volumeResidualFill);
      
      int basis_num = wkset->usebasis[rhhnum];
      auto basis = wkset->basis[basis_num];
      auto basis_grad = wkset->basis_grad[basis_num]; // velocity grad only
      auto off = subview(wkset->offsets, rhhnum, ALL());
      
      auto dHdx = wkset->getSolutionField("grad(rH_H)[x]");
      auto dHdy = wkset->getSolutionField("grad(rH_H)[y]");
      parallel_for("porous HGRAD volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += -1.0*source_rhh(elem,pt)*basis(elem,dof,pt,0) + dHdx(elem,pt)*basis_grad(elem,dof,pt,0) + dHdy(elem,pt)*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    
    // Laplace w.r.t velocity for rH_C
    {
      
      Teuchos::TimeMonitor resideval(*volumeResidualFill);
      
      int basis_num = wkset->usebasis[rhcnum];
      auto basis = wkset->basis[basis_num];
      auto basis_grad = wkset->basis_grad[basis_num]; // velocity grad only
      auto off = subview(wkset->offsets, rhcnum, ALL());
      
      auto dHdx = wkset->getSolutionField("grad(rH_C)[x]");
      auto dHdy = wkset->getSolutionField("grad(rH_C)[y]");
      parallel_for("porous HGRAD volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += -1.0*source_rhc(elem,pt)*basis(elem,dof,pt,0) + dHdx(elem,pt)*basis_grad(elem,dof,pt,0) + dHdy(elem,pt)*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    
    // Laplace w.r.t velocity for rH_G
    {
      
      Teuchos::TimeMonitor resideval(*volumeResidualFill);
      
      int basis_num = wkset->usebasis[rhgnum];
      auto basis = wkset->basis[basis_num];
      auto basis_grad = wkset->basis_grad[basis_num]; // velocity grad only
      auto off = subview(wkset->offsets, rhgnum, ALL());
      
      auto dHdx = wkset->getSolutionField("grad(rH_G)[x]");
      auto dHdy = wkset->getSolutionField("grad(rH_G)[y]");
      parallel_for("porous HGRAD volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += -1.0*source_rhg(elem,pt)*basis(elem,dof,pt,0) + dHdx(elem,pt)*basis_grad(elem,dof,pt,0) + dHdy(elem,pt)*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    
    // Laplace w.r.t velocity for rH_E
    {
      
      Teuchos::TimeMonitor resideval(*volumeResidualFill);
      
      int basis_num = wkset->usebasis[rhenum];
      auto basis = wkset->basis[basis_num];
      auto basis_grad = wkset->basis_grad[basis_num]; // velocity grad only
      auto off = subview(wkset->offsets, rhenum, ALL());
      
      auto dHdx = wkset->getSolutionField("grad(rH_E)[x]");
      auto dHdy = wkset->getSolutionField("grad(rH_E)[y]");
      parallel_for("porous HGRAD volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += -1.0*source_rhe(elem,pt)*basis(elem,dof,pt,0) + dHdx(elem,pt)*basis_grad(elem,dof,pt,0) + dHdy(elem,pt)*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    
  }
  
  if (include_Geqn) {
    Vista<EvalT> source_rgh, source_rgc, source_rgg, source_rge;
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      source_rgh = functionManager->evaluate("source_rG_H","ip");
      source_rgc = functionManager->evaluate("source_rG_C","ip");
      source_rgg = functionManager->evaluate("source_rG_G","ip");
      source_rge = functionManager->evaluate("source_rG_E","ip");
    }
    
    // Laplace w.r.t velocity for rG_H
    {
      
      Teuchos::TimeMonitor resideval(*volumeResidualFill);
      
      int basis_num = wkset->usebasis[rghnum];
      auto basis = wkset->basis[basis_num];
      auto basis_grad = wkset->basis_grad[basis_num]; // velocity grad only
      auto off = subview(wkset->offsets, rghnum, ALL());
      
      auto dGdx = wkset->getSolutionField("grad(rG_H)[x]");
      auto dGdy = wkset->getSolutionField("grad(rG_H)[y]");
      parallel_for("porous HGRAD volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += -1.0*source_rgh(elem,pt)*basis(elem,dof,pt,0) + dGdx(elem,pt)*basis_grad(elem,dof,pt,0) + dGdy(elem,pt)*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    
    // Laplace w.r.t velocity for rG_C
    {
      
      Teuchos::TimeMonitor resideval(*volumeResidualFill);
      
      int basis_num = wkset->usebasis[rgcnum];
      auto basis = wkset->basis[basis_num];
      auto basis_grad = wkset->basis_grad[basis_num]; // velocity grad only
      auto off = subview(wkset->offsets, rgcnum, ALL());
      
      auto dGdx = wkset->getSolutionField("grad(rG_C)[x]");
      auto dGdy = wkset->getSolutionField("grad(rG_C)[y]");
      parallel_for("porous HGRAD volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += -1.0*source_rgc(elem,pt)*basis(elem,dof,pt,0) + dGdx(elem,pt)*basis_grad(elem,dof,pt,0) + dGdy(elem,pt)*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    
    // Laplace w.r.t velocity for rG_G
    {
      
      Teuchos::TimeMonitor resideval(*volumeResidualFill);
      
      int basis_num = wkset->usebasis[rggnum];
      auto basis = wkset->basis[basis_num];
      auto basis_grad = wkset->basis_grad[basis_num]; // velocity grad only
      auto off = subview(wkset->offsets, rggnum, ALL());
      
      auto dGdx = wkset->getSolutionField("grad(rG_G)[x]");
      auto dGdy = wkset->getSolutionField("grad(rG_G)[y]");
      parallel_for("porous HGRAD volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += -1.0*source_rgg(elem,pt)*basis(elem,dof,pt,0) + dGdx(elem,pt)*basis_grad(elem,dof,pt,0) + dGdy(elem,pt)*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    
    // Laplace w.r.t velocity for rH_E
    {
      
      Teuchos::TimeMonitor resideval(*volumeResidualFill);
      
      int basis_num = wkset->usebasis[rgenum];
      auto basis = wkset->basis[basis_num];
      auto basis_grad = wkset->basis_grad[basis_num]; // velocity grad only
      auto off = subview(wkset->offsets, rgenum, ALL());
      
      auto dGdx = wkset->getSolutionField("grad(rG_E)[x]");
      auto dGdy = wkset->getSolutionField("grad(rG_E)[y]");
      parallel_for("porous HGRAD volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += -1.0*source_rge(elem,pt)*basis(elem,dof,pt,0) + dGdx(elem,pt)*basis_grad(elem,dof,pt,0) + dGdy(elem,pt)*basis_grad(elem,dof,pt,1);
          }
        }
      });
    }
    
  }
}


// ========================================================================================
// ========================================================================================

template<class EvalT>
void rosenbluth<EvalT>::boundaryResidual() {
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void rosenbluth<EvalT>::edgeResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void rosenbluth<EvalT>::computeFlux() {};

// ========================================================================================
// ========================================================================================

template<class EvalT>
void rosenbluth<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
  wkset = wkset_;
  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "rH_H") {
      rhhnum = i;
    }
    if (varlist[i] == "rH_C") {
      rhcnum = i;
    }
    if (varlist[i] == "rH_G") {
      rhgnum = i;
    }
    if (varlist[i] == "rH_E") {
      rhenum = i;
    }
    if (varlist[i] == "rG_H") {
      rghnum = i;
    }
    if (varlist[i] == "rG_C") {
      rgcnum = i;
    }
    if (varlist[i] == "rG_G") {
      rggnum = i;
    }
    if (varlist[i] == "rG_E") {
      rgenum = i;
    }
  }

}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::rosenbluth<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::rosenbluth<AD>;

// Standard built-in types
template class MrHyDE::rosenbluth<AD2>;
template class MrHyDE::rosenbluth<AD4>;
template class MrHyDE::rosenbluth<AD8>;
template class MrHyDE::rosenbluth<AD16>;
template class MrHyDE::rosenbluth<AD18>;
template class MrHyDE::rosenbluth<AD24>;
template class MrHyDE::rosenbluth<AD32>;
#endif
