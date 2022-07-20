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

#include "thermal.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

thermal::thermal(Teuchos::ParameterList & settings, const int & dimension_)
  : physicsbase(settings, dimension_)
{
  
  // Standard data
  label = "thermal";
  if (settings.isSublist("Active variables")) {
    if (settings.sublist("Active variables").isParameter("e")) {
      myvars.push_back("e");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("e","HGRAD"));
    }
  }
  else {
    myvars.push_back("e");
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

void thermal::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  
  functionManager->addFunction("thermal source",fs.get<string>("thermal source","0.0"),"ip");
  functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),"ip");
  functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),"ip");
  functionManager->addFunction("density",fs.get<string>("density","1.0"),"ip");
  functionManager->addFunction("bx",fs.get<string>("advection x","0.0"),"ip");
  functionManager->addFunction("by",fs.get<string>("advection y","0.0"),"ip");
  functionManager->addFunction("bz",fs.get<string>("advection z","0.0"),"ip");
  functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),"side ip");
  functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),"side ip");
  
}

// ========================================================================================
// ========================================================================================

void thermal::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  auto basis = wkset->basis[e_basis_num];
  auto basis_grad = wkset->basis_grad[e_basis_num];
  
  Vista source, diff, cp, rho, bx, by, bz;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("thermal source","ip");
    diff = functionManager->evaluate("thermal diffusion","ip");
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
  
  auto T = wkset->getSolutionField("e"); //e_vol;
  auto dTdt = wkset->getSolutionField("e_t"); //dedt_vol;
  
  auto off = subview( wkset->offsets, e_num, ALL());
  bool have_nsvel_ = have_nsvel;
  
  auto dTdx = wkset->getSolutionField("grad(e)[x]"); //dedx_vol;
  auto dTdy = wkset->getSolutionField("grad(e)[y]"); //dedy_vol;
  auto dTdz = wkset->getSolutionField("grad(e)[z]"); //dedz_vol;
  
  View_AD2 Ux, Uy, Uz;
  if (have_nsvel) {
    Ux = wkset->getSolutionField("ux");
    Uy = wkset->getSolutionField("uy");
    Uz = wkset->getSolutionField("uz");
  }
  size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
  
  auto bindex = wkset->basis_index;

  parallel_for("Thermal volume resid 3D part 1",
               TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VectorSize),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    LO bind = bindex(elem);
    for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
      for (size_type pt=0; pt<basis.extent(2); ++pt ) {
        res(elem,off(dof)) += (rho(elem,pt)*cp(elem,pt)*dTdt(elem,pt) - source(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
        res(elem,off(dof)) += diff(elem,pt)*dTdx(elem,pt)*wts(elem,pt)*basis_grad(bind,dof,pt,0);
        if (spaceDim > 1) {
          res(elem,off(dof)) += diff(elem,pt)*dTdy(elem,pt)*wts(elem,pt)*basis_grad(bind,dof,pt,1);
        }
        if (spaceDim > 2) {
          res(elem,off(dof)) += diff(elem,pt)*dTdz(elem,pt)*wts(elem,pt)*basis_grad(bind,dof,pt,2);
        }
        if (have_nsvel_) {
          if (spaceDim == 1) {
            res(elem,off(dof)) += Ux(elem,pt)*dTdx(elem,pt)*wts(elem,pt)*basis(bind,dof,pt,0);
          }
          else if (spaceDim == 2) {
            res(elem,off(dof)) += (Ux(elem,pt)*dTdx(elem,pt) + Uy(elem,pt)*dTdy(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
          }
          else {
            res(elem,off(dof)) += (Ux(elem,pt)*dTdx(elem,pt) + Uy(elem,pt)*dTdy(elem,pt) + Uz(elem,pt)*dTdz(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
          }
        }
        if (have_advection) {
          if (spaceDim == 1) {
            res(elem,off(dof)) += bx(elem,pt)*dTdx(elem,pt)*wts(elem,pt)*basis(bind,dof,pt,0);
          }
          else if (spaceDim == 2) {
            res(elem,off(dof)) += (bx(elem,pt)*dTdx(elem,pt) + by(elem,pt)*dTdy(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
          }
          else {
            res(elem,off(dof)) += (bx(elem,pt)*dTdx(elem,pt) + by(elem,pt)*dTdy(elem,pt) + bz(elem,pt)*dTdz(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
          }
        }
      }
    }
  });
  
}


// ========================================================================================
// ========================================================================================

void thermal::boundaryResidual() {
  
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  string bctype = bcs(e_num,cside);

  auto basis = wkset->basis_side[e_basis_num];
  auto basis_grad = wkset->basis_grad_side[e_basis_num];
  
  Vista nsource, diff_side, robin_alpha;
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (bctype == "weak Dirichlet" ) {
      nsource = functionManager->evaluate("Dirichlet e " + wkset->sidename,"side ip");
    }
    else if (bctype == "Neumann") {
      nsource = functionManager->evaluate("Neumann e " + wkset->sidename,"side ip");
    }
    diff_side = functionManager->evaluate("thermal diffusion","side ip");
    robin_alpha = functionManager->evaluate("robin alpha","side ip");
    
  }
  
  ScalarT sf = formparam;
  if (wkset->isAdjoint) {
    sf = 1.0;
    adjrhs = wkset->adjrhs;
  }
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  auto wts = wkset->wts_side;
  auto h = wkset->h;
  auto res = wkset->res;
  auto off = subview( wkset->offsets, e_num, ALL());
  int dim = wkset->dimension;
  
  // Contributes
  // <g(u),v> + <p(u),grad(v)\cdot n>
  
  auto bindex = wkset->basis_index;

  if (bcs(e_num,cside) == "Neumann") { // Neumann BCs
    parallel_for("Thermal bndry resid part 1",
                 TeamPolicy<AssemblyExec>(wkset->numElem, Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      LO bind = bindex(elem);
      for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
        for (size_type pt=0; pt<basis.extent(2); ++pt ) {
          res(elem,off(dof)) += -nsource(elem,pt)*wts(elem,pt)*basis(bind,dof,pt,0);
        }
      }
    });
  }
  else if (bcs(e_num,cside) == "weak Dirichlet" || bcs(e_num,cside) == "interface") {
    auto T = wkset->getSolutionField("e"); //e_side;
    auto dTdx = wkset->getSolutionField("grad(e)[x]"); //dedx_side;
    auto dTdy = wkset->getSolutionField("grad(e)[y]"); //dedy_side;
    auto dTdz = wkset->getSolutionField("grad(e)[z]"); //dedz_side;
    auto nx = wkset->getScalarField("n[x]");
    auto ny = wkset->getScalarField("n[y]");
    auto nz = wkset->getScalarField("n[z]");
    Vista bdata;
    
    if (bcs(e_num,cside) == "weak Dirichlet") {
      bdata = nsource;
    }
    else if (bcs(e_num,cside) == "interface") {
      bdata = wkset->getSolutionField("aux e");
    }
    ScalarT epen = 10.0;
    parallel_for("Thermal bndry resid wD",
                 TeamPolicy<AssemblyExec>(wkset->numElem, Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      LO bind = bindex(elem);
      if (dim == 1) {
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += epen/h(elem)*diff_side(elem,pt)*(T(elem,pt)-bdata(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
            res(elem,off(dof)) += -diff_side(elem,pt)*dTdx(elem,pt)*nx(elem,pt)*wts(elem,pt)*basis(bind,dof,pt,0);
            res(elem,off(dof)) += -sf*diff_side(elem,pt)*(T(elem,pt) - bdata(elem,pt))*wts(elem,pt)*basis_grad(bind,dof,pt,0)*nx(elem,pt);
          }
        }
      }
      else if (dim == 2) {
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += epen/h(elem)*diff_side(elem,pt)*(T(elem,pt)-bdata(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
            res(elem,off(dof)) += -diff_side(elem,pt)*(dTdx(elem,pt)*nx(elem,pt)+dTdy(elem,pt)*ny(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
            res(elem,off(dof)) += -sf*diff_side(elem,pt)*(T(elem,pt) - bdata(elem,pt))*wts(elem,pt)*(basis_grad(bind,dof,pt,0)*nx(elem,pt) + basis_grad(bind,dof,pt,1)*ny(elem,pt));
          }
        }
      }
      else {
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += epen/h(elem)*diff_side(elem,pt)*(T(elem,pt)-bdata(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
            res(elem,off(dof)) += -diff_side(elem,pt)*(dTdx(elem,pt)*nx(elem,pt)+dTdy(elem,pt)*ny(elem,pt)+dTdz(elem,pt)*nz(elem,pt))*wts(elem,pt)*basis(bind,dof,pt,0);
            res(elem,off(dof)) += -sf*diff_side(elem,pt)*(T(elem,pt) - bdata(elem,pt))*wts(elem,pt)*(basis_grad(bind,dof,pt,0)*nx(elem,pt) + basis_grad(bind,dof,pt,1)*ny(elem,pt) + + basis_grad(bind,dof,pt,2)*nz(elem,pt));
          }
        }
      }
      
    });
    
  }
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void thermal::computeFlux() {
  
  int spaceDim = wkset->dimension;
  // TMW: sf is still an issue for GPUs
  ScalarT sf = 1.0;
  if (wkset->isAdjoint) {
    sf = formparam;
  }
  
  Vista diff_side;
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    diff_side = functionManager->evaluate("thermal diffusion","side ip");
  }
  
  View_Sc2 nx, ny, nz;
  View_AD2 T, dTdx, dTdy, dTdz;
  nx = wkset->getScalarField("n[x]");
  T = wkset->getSolutionField("e");
  dTdx = wkset->getSolutionField("grad(e)[x]"); //dedx_side;
  if (spaceDim > 1) {
    ny = wkset->getScalarField("n[y]");
    dTdy = wkset->getSolutionField("grad(e)[y]"); //dedy_side;
  }
  if (spaceDim > 2) {
    nz = wkset->getScalarField("n[z]");
    dTdz = wkset->getSolutionField("grad(e)[z]"); //dedz_side;
  }
  
  auto h = wkset->h;
  int dim = wkset->dimension;
  ScalarT epen = 10.0;
  {
    //Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto fluxT = subview(wkset->flux, ALL(), e_num, ALL());
    auto lambda = wkset->getSolutionField("aux e");
    
    {
      Teuchos::TimeMonitor localtime(*fluxFill);
      
      parallel_for("Thermal bndry resid wD",
                   TeamPolicy<AssemblyExec>(wkset->numElem, Kokkos::AUTO, VectorSize),
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

void thermal::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;
  
  ux_num = -1;
  uy_num = -1;
  uz_num = -1;
  
  vector<string> varlist = wkset->varlist;
  //if (!isaux) {
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "e")
        e_num = i;
      if (varlist[i] == "ux")
        ux_num = i;
      if (varlist[i] == "uy")
        uy_num = i;
      if (varlist[i] == "uz")
        uz_num = i;
    }
  if (wkset->isInitialized) { // safeguard against proc having no elem on block
    e_basis_num = wkset->usebasis[e_num];
  }
    if (ux_num >=0)
      have_nsvel = true;
  //}
  
  vector<string> auxvarlist = wkset->aux_varlist;
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "e")
      auxe_num = i;
  }
  
  // Set these views so we don't need to search repeatedly
  
  /*
  if (mybasistypes[0] == "HGRAD") {
    wkset->get("e",e_vol);
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

std::vector< std::vector<string> > thermal::setupIntegratedQuantities(const int & spaceDim) {

  std::vector< std::vector<string> > integrandsNamesAndTypes;

  // if not requested, be sure to return an empty vector
  if ( !(test_IQs) ) return integrandsNamesAndTypes;

  std::vector<string> IQ = {"e","thermal vol total e","volume"};
  integrandsNamesAndTypes.push_back(IQ);

  IQ = {"e","thermal bnd total e","boundary"};
  integrandsNamesAndTypes.push_back(IQ);

  // TODO -- BWR assumes the diffusion coefficient is 1.
  // I was getting all zeroes if I used "diff"
  string integrand = "(n[x]*grad(e)[x])";
  if (spaceDim == 2) integrand = "(n[x]*grad(e)[x] + n[y]*grad(e)[y])";
  if (spaceDim == 3) integrand = "(n[x]*grad(e)[x] + n[y]*grad(e)[y] + n[z]*grad(e)[z])";

  IQ = {integrand,"thermal bnd heat flux","boundary"};
  integrandsNamesAndTypes.push_back(IQ);

  return integrandsNamesAndTypes;

}
