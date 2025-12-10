/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "ellipticPrior.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
ellipticPrior<EvalT>::ellipticPrior(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  // Standard data
  label = "ellipticPrior";
  if (settings.isSublist("Active variables")) {
    if (settings.sublist("Active variables").isParameter("T")) {
      myvars.push_back("T");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("T","HGRAD"));
    }
  }
  else {
    myvars.push_back("T");
    mybasistypes.push_back("HGRAD");
  }

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void ellipticPrior<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  
  functionManager->addFunction("ellipticPrior source",fs.get<string>("ellipticPrior source","0.0"),"ip");
  functionManager->addFunction("ellipticPrior diffusion",fs.get<string>("ellipticPrior diffusion","1.0"),"ip");
  functionManager->addFunction("ellipticPrior reaction",fs.get<string>("ellipticPrior reaction","1.0"),"ip");
  functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),"side ip"); 
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void ellipticPrior<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  auto basis = wkset->basis[T_basis_num];
  auto basis_grad = wkset->basis_grad[T_basis_num];
  
  Vista<EvalT> source, diff, react;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("ellipticPrior source","ip");
    diff = functionManager->evaluate("ellipticPrior diffusion","ip");
    react = functionManager->evaluate("ellipticPrior reaction","ip");
  }
  
  // Contributes:
  // (f(u),v) + (DF(u),nabla v)
  // f(u) = de/dt - source
  // DF(u) = diff*grad(e)
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
 
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  auto T = wkset->getSolutionField("T"); //e_vol;
  auto dTdt = wkset->getSolutionField("T_t"); //dedt_vol;
  
  auto off = subview( wkset->offsets, T_num, ALL());
  
  auto dTdx = wkset->getSolutionField("grad(T)[x]"); //dedx_vol;
  auto dTdy = wkset->getSolutionField("grad(T)[y]"); //dedy_vol;
  auto dTdz = wkset->getSolutionField("grad(T)[z]"); //dedz_vol;
  
  size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
  
  parallel_for("ellipticPrior volume resid 3D part 1",
               TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
               MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
      for (size_type pt=0; pt<basis.extent(2); ++pt ) {
        res(elem,off(dof)) += (dTdt(elem,pt) - source(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
        res(elem,off(dof)) += diff(elem,pt)*dTdx(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,0) + react(elem,pt)*T(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
        if (spaceDim > 1) {
          res(elem,off(dof)) += diff(elem,pt)*dTdy(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,1);
        }
        if (spaceDim > 2) {
          res(elem,off(dof)) += diff(elem,pt)*dTdz(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,2);
        }
    }
    }
  });

}


// ========================================================================================
// ========================================================================================

template<class EvalT>
void ellipticPrior<EvalT>::boundaryResidual() {
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void ellipticPrior<EvalT>::computeFlux() {
  
  int spaceDim = wkset->dimension;
  // TMW: sf is still an issue for GPUs
  ScalarT sf = 1.0;
  
  Vista<EvalT> diff_side;
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    diff_side = functionManager->evaluate("ellipticPrior diffusion","side ip");
  }
  
  View_Sc2 nx, ny, nz;
  View_EvalT2 T, dTdx, dTdy, dTdz;
  nx = wkset->getScalarField("n[x]");
  T = wkset->getSolutionField("T");
  dTdx = wkset->getSolutionField("grad(T)[x]"); //dedx_side;
  if (spaceDim > 1) {
    ny = wkset->getScalarField("n[y]");
    dTdy = wkset->getSolutionField("grad(T)[y]"); //dedy_side;
  }
  if (spaceDim > 2) {
    nz = wkset->getScalarField("n[z]");
    dTdz = wkset->getSolutionField("grad(T)[z]"); //dedz_side;
  }
  
  auto h = wkset->getSideElementSize();
  int dim = wkset->dimension;
  ScalarT epen = 10.0;
  {
    auto fluxT = subview(wkset->flux, ALL(), T_num, ALL());
    auto lambda = wkset->getSolutionField("aux e");
    
    {
      Teuchos::TimeMonitor localtime(*fluxFill);
      
      parallel_for("ellipticPrior bndry resid wD",
                   TeamPolicy<AssemblyExec>(wkset->numElem, Kokkos::AUTO, VECTORSIZE),
                   MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<nx.extent(1); pt+=team.team_size() ) {
          fluxT(elem,pt) = epen/h(elem)*diff_side(elem,pt)*(lambda(elem,pt)-T(elem,pt));
          fluxT(elem,pt) += sf*diff_side(elem,pt)*dTdx(elem,pt)*nx(elem,pt);
          if (dim > 1) {
            fluxT(elem,pt) += sf*diff_side(elem,pt)*dTdy(elem,pt)*ny(elem,pt);
          }
          if (dim > 2) {
            fluxT(elem,pt) += sf*diff_side(elem,pt)*dTdz(elem,pt)*nz(elem,pt);
          }
        }
      });
    }
    
  }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void ellipticPrior<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  for (size_t i = 0; i < varlist.size(); i++)
  {
    if (varlist[i] == "T")
      T_num = i;
  }
  if (wkset->isInitialized)
  { // safeguard against proc having no elem on block
    T_basis_num = wkset->usebasis[T_num];
  }

  vector<string> auxvarlist = wkset->aux_varlist;
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "T")
      auxT_num = i;
  }
  
}

//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::ellipticPrior<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::ellipticPrior<AD>;

// Standard built-in types
template class MrHyDE::ellipticPrior<AD2>;
template class MrHyDE::ellipticPrior<AD4>;
template class MrHyDE::ellipticPrior<AD8>;
template class MrHyDE::ellipticPrior<AD16>;
template class MrHyDE::ellipticPrior<AD18>;
template class MrHyDE::ellipticPrior<AD24>;
template class MrHyDE::ellipticPrior<AD32>;
#endif
