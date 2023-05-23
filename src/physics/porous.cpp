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

#include "porous.hpp"
using namespace MrHyDE;

porous::porous(Teuchos::ParameterList & settings, const int & dimension_)
  : physicsbase(settings, dimension_)
{
  
  // Standard data
  label = "porous";
  myvars.push_back("p");
  mybasistypes.push_back("HGRAD");
  formparam = settings.get<ScalarT>("form_param",1.0);
}

// ========================================================================================
// ========================================================================================

void porous::defineFunctions(Teuchos::ParameterList & fs,
                             Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  
  functionManager->addFunction("source",fs.get<string>("porous source","0.0"),"ip");
  functionManager->addFunction("permeability",fs.get<string>("permeability","1.0"),"ip");
  functionManager->addFunction("porosity",fs.get<string>("porosity","1.0"),"ip");
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","1.0"),"ip");
  functionManager->addFunction("reference density",fs.get<string>("reference density","1.0"),"ip");
  functionManager->addFunction("reference pressure",fs.get<string>("reference pressure","1.0"),"ip");
  functionManager->addFunction("compressibility",fs.get<string>("compressibility","0.0"),"ip");
  functionManager->addFunction("gravity",fs.get<string>("gravity","1.0"),"ip");

  functionManager->addFunction("source",fs.get<string>("porous source","0.0"),"side ip");
  functionManager->addFunction("permeability",fs.get<string>("permeability","1.0"),"side ip");
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","1.0"),"side ip");
  functionManager->addFunction("reference density",fs.get<string>("reference density","1.0"),"side ip");
  functionManager->addFunction("reference pressure",fs.get<string>("reference pressure","1.0"),"side ip");
  functionManager->addFunction("compressibility",fs.get<string>("compressibility","0.0"),"side ip");
  functionManager->addFunction("gravity",fs.get<string>("gravity","1.0"),"side ip");
}

// ========================================================================================
// ========================================================================================

void porous::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int p_basis_num = wkset->usebasis[pnum];
  auto basis = wkset->getDecompressedBasis(p_basis_num);
  auto basis_grad = wkset->getDecompressedBasisGrad(p_basis_num);
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  Vista perm, porosity, viscosity, densref, pref, comp, gravity, source;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    perm = functionManager->evaluate("permeability","ip");
    porosity = functionManager->evaluate("porosity","ip");
    viscosity = functionManager->evaluate("viscosity","ip");
    densref = functionManager->evaluate("reference density","ip");
    pref = functionManager->evaluate("reference pressure","ip");
    comp = functionManager->evaluate("compressibility","ip");
    gravity = functionManager->evaluate("gravity","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  auto psol = wkset->getSolutionField("p");
  auto pdot = wkset->getSolutionField("p_t");
  auto off = subview(wkset->offsets, pnum, ALL());
  
  if (spaceDim == 1) {
    auto dpdx = wkset->getSolutionField("grad(p)[x]");
    parallel_for("porous HGRAD volume resid 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<psol.extent(1); pt++ ) {
        AD Kdens = perm(elem,pt)/viscosity(elem,pt)*densref(elem,pt)*(1.0+comp(elem,pt)*(psol(elem,pt) - pref(elem,pt)));
        AD M = porosity(elem,pt)*densref(elem,pt)*comp(elem,pt)*pdot(elem,pt) - source(elem,pt);
        M *= wts(elem,pt);
        AD Kx = Kdens*dpdx(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += M*basis(elem,dof,pt,0) + Kx*basis_grad(elem,dof,pt,0);
        }
      }
    });
  }
  else if (spaceDim == 2) {
    auto dpdx = wkset->getSolutionField("grad(p)[x]");
    auto dpdy = wkset->getSolutionField("grad(p)[y]");
    parallel_for("porous HGRAD volume resid 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<psol.extent(1); pt++ ) {
        AD Kdens = perm(elem,pt)/viscosity(elem,pt)*densref(elem,pt)*(1.0+comp(elem,pt)*(psol(elem,pt) - pref(elem,pt)));
        AD M = porosity(elem,pt)*densref(elem,pt)*comp(elem,pt)*pdot(elem,pt) - source(elem,pt);
        M *= wts(elem,pt);
        AD Kx = Kdens*dpdx(elem,pt)*wts(elem,pt);
        AD Ky = Kdens*dpdy(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += M*basis(elem,dof,pt,0) + Kx*basis_grad(elem,dof,pt,0) + Ky*basis_grad(elem,dof,pt,1);
        }
      }
    });
  }
  else if (spaceDim == 3) {
    auto dpdx = wkset->getSolutionField("grad(p)[x]");
    auto dpdy = wkset->getSolutionField("grad(p)[y]");
    auto dpdz = wkset->getSolutionField("grad(p)[z]");
    parallel_for("porous HGRAD volume resid 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<psol.extent(1); pt++ ) {
        AD Kdens = perm(elem,pt)/viscosity(elem,pt)*densref(elem,pt)*(1.0+comp(elem,pt)*(psol(elem,pt) - pref(elem,pt)));
        AD M = porosity(elem,pt)*densref(elem,pt)*comp(elem,pt)*pdot(elem,pt) - source(elem,pt);
        M *= wts(elem,pt);
        AD Kx = Kdens*dpdx(elem,pt)*wts(elem,pt);
        AD Ky = Kdens*dpdy(elem,pt)*wts(elem,pt);
        AD Kz = Kdens*dpdz(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += M*basis(elem,dof,pt,0) + Kx*basis_grad(elem,dof,pt,0) + Ky*basis_grad(elem,dof,pt,1) + Kz*basis_grad(elem,dof,pt,2);
        }
      }
    });
  }
  
}


// ========================================================================================
// ========================================================================================

void porous::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  string bctype = bcs(pnum,cside);
  
  int basis_num = wkset->usebasis[pnum];
  auto basis = wkset->basis_side[basis_num];
  auto basis_grad = wkset->basis_grad_side[basis_num];
  
  Vista perm, viscosity, densref, pref, comp, gravity, source; // porosity is currently unused
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (bctype == "weak Dirichlet" ) {
      source = functionManager->evaluate("Dirichlet p " + wkset->sidename,"side ip");
    }
    else if (bctype == "Neumann") {
      source = functionManager->evaluate("Neumann p " + wkset->sidename,"side ip");
    }
    perm = functionManager->evaluate("permeability","side ip");
    viscosity = functionManager->evaluate("viscosity","side ip");
    densref = functionManager->evaluate("reference density","side ip");
    pref = functionManager->evaluate("reference pressure","side ip");
    comp = functionManager->evaluate("compressibility","side ip");
    gravity = functionManager->evaluate("gravity","side ip");
    
  }
  
  ScalarT sf = formparam;
  if (wkset->isAdjoint) {
    sf = 1.0;
    adjrhs = wkset->adjrhs;
  }
  
  auto wts = wkset->wts_side;
  auto h = wkset->h;
  auto res = wkset->res;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  View_Sc2 nx, ny, nz;
  View_AD2 dpdx, dpdy, dpdz;
  nx = wkset->getScalarField("n[x]");
  dpdx = wkset->getSolutionField("grad(p)[x]");
  if (spaceDim > 1) {
    ny = wkset->getScalarField("n[y]");
    dpdy = wkset->getSolutionField("grad(p)[y]");
  }
  if (spaceDim > 2) {
    nz = wkset->getScalarField("n[z]");
    dpdz = wkset->getSolutionField("grad(p)[z]");
  }
  
  auto psol = wkset->getSolutionField("p");
  auto off = subview(wkset->offsets, pnum, ALL());

  if (bcs(pnum,cside) == "Neumann") { //Neumann
    parallel_for("porous HGRAD bndry resid Neumann",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD s = -source(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += s*basis(elem,dof,pt,0);
        }
      }
    });
  }
  else if (bcs(pnum,cside) == "weak Dirichlet") { // weak Dirichlet
    parallel_for("porous HGRAD bndry resid weak Dirichlet",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis_grad.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD pval = psol(elem,pt);
        AD dens = densref(elem,pt)*(1.0+comp(elem,pt)*(pval - pref(elem,pt)));
        AD Kval = perm(elem,pt)/viscosity(elem,pt)*dens;
        AD weakDiriScale = 10.0*Kval/h(elem);
        AD Kgradp_dot_n = Kval*dpdx(elem,pt)*nx(elem,pt);
        if (dim > 1) {
          Kgradp_dot_n += Kval*dpdy(elem,pt)*ny(elem,pt);
        }
        if (dim > 2) {
          Kgradp_dot_n += Kval*dpdz(elem,pt)*nz(elem,pt);
        }
        Kgradp_dot_n *= wts(elem,pt);
        AD pdiff = (pval - source(elem,pt))*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT v = basis(elem,dof,pt,0);
          AD Kgradv_dot_n = Kval*basis_grad(elem,dof,pt,0)*nx(elem,pt);
          if (dim > 1) {
            Kgradv_dot_n += Kval*basis_grad(elem,dof,pt,1)*ny(elem,pt);
          }
          if (dim > 2) {
            Kgradv_dot_n += Kval*basis_grad(elem,dof,pt,2)*nz(elem,pt);
          }
          res(elem,off(dof)) += -Kgradp_dot_n*v - sf*Kgradv_dot_n*pdiff + weakDiriScale*pdiff*v;
        }
      }
    });
  }
  else if (bcs(pnum,cside) == "interface") { // multiscale weak Dirichlet
    auto lambda = wkset->getSolutionField("aux "+auxvar);
    parallel_for("porous HGRAD bndry resid MS weak Dirichlet",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis_grad.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD pval = psol(elem,pt);
        AD dens = densref(elem,pt)*(1.0+comp(elem,pt)*(pval - pref(elem,pt)));
        AD Kval = perm(elem,pt)/viscosity(elem,pt)*dens;
        AD weakDiriScale = 10.0*Kval/h(elem);
        AD Kgradp_dot_n = Kval*dpdx(elem,pt)*nx(elem,pt);
        if (dim > 1) {
          Kgradp_dot_n += Kval*dpdy(elem,pt)*ny(elem,pt);
        }
        if (dim > 2) {
          Kgradp_dot_n += Kval*dpdz(elem,pt)*nz(elem,pt);
        }
        Kgradp_dot_n *= wts(elem,pt);
        AD pdiff = (pval - lambda(elem,pt))*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT v = basis(elem,dof,pt,0);
          AD Kgradv_dot_n = Kval*basis_grad(elem,dof,pt,0)*nx(elem,pt);
          if (dim > 1) {
            Kgradv_dot_n += Kval*basis_grad(elem,dof,pt,1)*ny(elem,pt);
          }
          if (dim > 2) {
            Kgradv_dot_n += Kval*basis_grad(elem,dof,pt,2)*nz(elem,pt);
          }
          res(elem,off(dof)) += -Kgradp_dot_n*v - sf*Kgradv_dot_n*pdiff + weakDiriScale*pdiff*v;
        }
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

void porous::edgeResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void porous::computeFlux() {
  
  int spaceDim = wkset->dimension;
  ScalarT sf = 1.0; // TMW: not on device
  if (wkset->isAdjoint) {
    sf = formparam;
  }
  
  Vista perm, porosity, viscosity, densref, pref, comp, gravity, source;
  
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    perm = functionManager->evaluate("permeability","side ip");
    viscosity = functionManager->evaluate("viscosity","side ip");
    densref = functionManager->evaluate("reference density","side ip");
    pref = functionManager->evaluate("reference pressure","side ip");
    comp = functionManager->evaluate("compressibility","side ip");
    gravity = functionManager->evaluate("gravity","side ip");
  }
  
  auto h = wkset->h;
  auto basis_grad = wkset->basis_side[wkset->usebasis[pnum]];
 
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    View_Sc2 nx, ny, nz;
    View_AD2 dpdx, dpdy, dpdz;
    nx = wkset->getScalarField("n[x]");
    dpdx = wkset->getSolutionField("grad(p)[x]");
    if (spaceDim > 1) {
      ny = wkset->getScalarField("n[y]");
      dpdy = wkset->getSolutionField("grad(p)[y]");
    }
    if (spaceDim > 2) {
      nz = wkset->getScalarField("n[z]");
      dpdz = wkset->getSolutionField("grad(p)[z]");
    }
    
    auto pflux = subview(wkset->flux, ALL(), pnum, ALL());
    auto psol = wkset->getSolutionField("p");
    auto lambda = wkset->getSolutionField("aux "+auxvar);
    parallel_for("porous HGRAD flux",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis_grad.extent(3);
      for (size_type pt=0; pt<pflux.extent(1); pt++) {
        AD dens = densref(elem,pt)*(1.0+comp(elem,pt)*(psol(elem,pt) - pref(elem,pt)));
        AD Kval = perm(elem,pt)/viscosity(elem,pt)*dens;
        
        AD penalty = 10.0*Kval/h(elem);
        AD Kgradp_dot_n = Kval*dpdx(elem,pt)*nx(elem,pt);
        if (dim > 1) {
          Kgradp_dot_n += Kval*dpdy(elem,pt)*ny(elem,pt);
        }
        if (dim > 2) {
          Kgradp_dot_n += Kval*(dpdz(elem,pt) - gravity(elem,pt)*dens)*nz(elem,pt);
        }
        pflux(elem,pt) += sf*Kgradp_dot_n + penalty*(lambda(elem,pt)-psol(elem,pt));
        
      }
    });
  }
  
}

// ========================================================================================
// ========================================================================================

void porous::setWorkset(Teuchos::RCP<workset> & wkset_) {
  wkset = wkset_;
  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "p") {
      pnum = i;
    }
  }

  vector<string> auxvarlist = wkset->aux_varlist;
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "p") {
      auxpnum = i;
      auxvar = "p";
    }
    if (auxvarlist[i] == "lambda") {
      auxpnum = i;
      auxvar = "lambda";
    }
    if (auxvarlist[i] == "pbndry") {
      auxpnum = i;
      auxvar = "pbndry";
    }
  }
}

// ========================================================================================
// ========================================================================================

void porous::updatePerm(View_AD2 perm) {
  
  View_Sc2 data = wkset->extra_data;
  
  parallel_for("porous HGRAD update perm",
               RangePolicy<AssemblyExec>(0,perm.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<perm.extent(1); pt++) {
      perm(elem,pt) = data(elem,0);
    }
  });
}
