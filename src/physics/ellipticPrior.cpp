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
  // Extra data
  formparam = settings.get<ScalarT>("form_param",1.0);
  have_nsvel = false;
  // Solely for testing purposes
  test_IQs = settings.get<bool>("test integrated quantities",false);
  have_advection = settings.get<bool>("include advection",false);


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
  functionManager->addFunction("ellipticPrior diffusion",fs.get<string>("ellipticPrior diffusion","1.0"),"side ip");
  functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),"ip");
  functionManager->addFunction("density",fs.get<string>("density","1.0"),"ip");
  functionManager->addFunction("bx",fs.get<string>("advection x","0.0"),"ip");
  functionManager->addFunction("by",fs.get<string>("advection y","0.0"),"ip");
  functionManager->addFunction("bz",fs.get<string>("advection z","0.0"),"ip");
  functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),"side ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void ellipticPrior<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  auto basis = wkset->basis[T_basis_num];
  auto basis_grad = wkset->basis_grad[T_basis_num];
  
  Vista<EvalT> source, diff, react, cp, rho, bx, by, bz;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("ellipticPrior source","ip");
    diff = functionManager->evaluate("ellipticPrior diffusion","ip");
    react = functionManager->evaluate("ellipticPrior reaction","ip");
    cp = functionManager->evaluate("specific heat","ip");
    rho = functionManager->evaluate("density","ip");
    if (have_advection) {
      bx = functionManager->evaluate("bx","ip");
      if (spaceDim > 1) {
        by = functionManager->evaluate("by","ip");
      }
      if (spaceDim > 2) {
        bz = functionManager->evaluate("bz","ip");
      }
    }
  }
  
  // Contributes:
  // (f(u),v) + (DF(u),nabla v)
  // f(u) = rho*cp*de/dt - source
  // DF(u) = diff*grad(e)
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
 
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  auto T = wkset->getSolutionField("T"); //e_vol;
  auto dTdt = wkset->getSolutionField("T_t"); //dedt_vol;
  
  auto off = subview( wkset->offsets, T_num, ALL());
  bool have_nsvel_ = have_nsvel;
  
  auto dTdx = wkset->getSolutionField("grad(T)[x]"); //dedx_vol;
  auto dTdy = wkset->getSolutionField("grad(T)[y]"); //dedy_vol;
  auto dTdz = wkset->getSolutionField("grad(T)[z]"); //dedz_vol;
  
  View_EvalT2 Ux, Uy, Uz;
  if (have_nsvel) {
    Ux = wkset->getSolutionField("ux");
    Uy = wkset->getSolutionField("uy");
    Uz = wkset->getSolutionField("uz");
  }
  size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
  
  
  parallel_for("ellipticPrior volume resid 3D part 1",
               TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
      for (size_type pt=0; pt<basis.extent(2); ++pt ) {
        res(elem,off(dof)) += (rho(elem,pt)*cp(elem,pt)*dTdt(elem,pt) - source(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
        res(elem,off(dof)) += diff(elem,pt)*dTdx(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,0) + react(elem,pt)*T(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
        if (spaceDim > 1) {
          res(elem,off(dof)) += diff(elem,pt)*dTdy(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,1);
        }
        if (spaceDim > 2) {
          res(elem,off(dof)) += diff(elem,pt)*dTdz(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,2);
        }
        if (have_nsvel_) {
          if (spaceDim == 1) {
            res(elem,off(dof)) += Ux(elem,pt)*dTdx(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
          }
          else if (spaceDim == 2) {
            res(elem,off(dof)) += (Ux(elem,pt)*dTdx(elem,pt) + Uy(elem,pt)*dTdy(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
          }
          else {
            res(elem,off(dof)) += (Ux(elem,pt)*dTdx(elem,pt) + Uy(elem,pt)*dTdy(elem,pt) + Uz(elem,pt)*dTdz(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
  
        }
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
  if (wkset->isAdjoint) {
    sf = formparam;
  }
  
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
    //Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto fluxT = subview(wkset->flux, ALL(), T_num, ALL());
    auto lambda = wkset->getSolutionField("aux e");
    
    {
      Teuchos::TimeMonitor localtime(*fluxFill);
      
      parallel_for("ellipticPrior bndry resid wD",
                   TeamPolicy<AssemblyExec>(wkset->numElem, Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<nx.extent(1); pt+=team.team_size() ) {
          fluxT(elem,pt) = epen/h(elem)*diff_side(elem,pt)*(lambda(elem,pt)-T(elem,pt));
          //fluxT(elem,pt) += epen/h(elem)*diff_side(elem,pt)*(lambda(elem,pt)-T(elem,pt));
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
  
  ux_num = -1;
  uy_num = -1;
  uz_num = -1;
  
  vector<string> varlist = wkset->varlist;
  //if (!isaux) {
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "T")
        T_num = i;
      if (varlist[i] == "ux")
        ux_num = i;
      if (varlist[i] == "uy")
        uy_num = i;
      if (varlist[i] == "uz")
        uz_num = i;
    }
  if (wkset->isInitialized) { // safeguard against proc having no elem on block
    T_basis_num = wkset->usebasis[T_num];
  }
    if (ux_num >=0)
      have_nsvel = true;
  //}
  
  vector<string> auxvarlist = wkset->aux_varlist;
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "T")
      auxT_num = i;
  }
  
  // Set these views so we don't need to search repeatedly
  
  /*
  if (mybasistypes[0] == "HGRAD") {
    wkset->get("T",e_vol);
    wkset->get("e side",e_side);
    wkset->get("e_t",dedt_vol);
    
    wkset->get("grad(e)[x]",dedx_vol);
    wkset->get("grad(e)[y]",dedy_vol);
    wkset->get("grad(e)[z]",dedz_vol);
    
    wkset->get("grad(e)[x] side",dedx_side);
    wkset->get("grad(e)[y] side",dedy_side);
    wkset->get("grad(e)[z] side",dedz_side);
    
    if (have_nsvel) {
      wkset->get("ux",ux_vol);
      wkset->get("uy",uy_vol);
      wkset->get("uz",uz_vol);
    }
  }
   */

  // testing purposes only
  if (test_IQs) IQ_start = wkset->addIntegratedQuantities(3);

}

// ========================================================================================
// return the integrands for the integrated quantities (testing only for now)
// ========================================================================================

template<class EvalT>
std::vector< std::vector<string> > ellipticPrior<EvalT>::setupIntegratedQuantities(const int & spaceDim) {

  std::vector< std::vector<string> > integrandsNamesAndTypes;

  // if not requested, be sure to return an empty vector
  if ( !(test_IQs) ) return integrandsNamesAndTypes;

  std::vector<string> IQ = {"T","ellipticPrior vol total T","volume"};
  integrandsNamesAndTypes.push_back(IQ);

  IQ = {"T","ellipticPrior bnd total e","boundary"};
  integrandsNamesAndTypes.push_back(IQ);

  // TODO -- BWR assumes the diffusion coefficient is 1.
  // I was getting all zeroes if I used "diff"
  string integrand = "(n[x]*grad(T)[x])";
  if (spaceDim == 2) integrand = "(n[x]*grad(T)[x] + n[y]*grad(T)[y])";
  if (spaceDim == 3) integrand = "(n[x]*grad(T)[x] + n[y]*grad(T)[y] + n[z]*grad(T)[z])";

  IQ = {integrand,"ellipticPrior bnd heat flux","boundary"};
  integrandsNamesAndTypes.push_back(IQ);

  return integrandsNamesAndTypes;

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
