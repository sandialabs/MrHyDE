/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "porous.hpp"

porous::porous(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  // Standard data
  label = "porous";
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  myvars.push_back("p");
  mybasistypes.push_back("HGRAD");
  formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
}

// ========================================================================================
// ========================================================================================

void porous::defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                             Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("source",fs.get<string>("porous source","0.0"),"ip");
  functionManager->addFunction("permeability",fs.get<string>("permeability","1.0"),"ip");
  functionManager->addFunction("porosity",fs.get<string>("porosity","1.0"),"ip");
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","1.0"),"ip");
  functionManager->addFunction("reference density",fs.get<string>("reference density","1.0"),"ip");
  functionManager->addFunction("reference pressure",fs.get<string>("reference pressure","1.0"),"ip");
  functionManager->addFunction("compressibility",fs.get<string>("compressibility","0.0"),"ip");
  functionManager->addFunction("gravity",fs.get<string>("gravity","1.0"),"ip");
}

// ========================================================================================
// ========================================================================================

void porous::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int p_basis_num = wkset->usebasis[pnum];
  auto basis = wkset->basis[p_basis_num];
  auto basis_grad = wkset->basis_grad[p_basis_num];
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  FDATA perm, porosity, viscosity, densref, pref, comp, gravity, source;
  
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
  
  auto psol = Kokkos::subview(sol, Kokkos::ALL(), pnum, Kokkos::ALL(), 0);
  auto pdot = Kokkos::subview(sol_dot, Kokkos::ALL(), pnum, Kokkos::ALL(), 0);
  auto pgrad = Kokkos::subview(sol_grad, Kokkos::ALL(), pnum, Kokkos::ALL(), Kokkos::ALL());
  auto off = Kokkos::subview(offsets, pnum, Kokkos::ALL());
  
  if (spaceDim == 1) {
    parallel_for("porous HGRAD volume resid 1D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<psol.extent(1); pt++ ) {
        AD Kdens = perm(elem,pt)/viscosity(elem,pt)*densref(elem,pt)*(1.0+comp(elem,pt)*(psol(elem,pt) - pref(elem,pt)));
        AD M = porosity(elem,pt)*densref(elem,pt)*comp(elem,pt)*pdot(elem,pt) - source(elem,pt);
        M *= wts(elem,pt);
        AD Kx = Kdens*pgrad(elem,pt,0)*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += M*basis(elem,dof,pt) + Kx*basis_grad(elem,dof,pt,0);
        }
      }
    });
  }
  else if (spaceDim == 2) {
    parallel_for("porous HGRAD volume resid 2D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<psol.extent(1); pt++ ) {
        AD Kdens = perm(elem,pt)/viscosity(elem,pt)*densref(elem,pt)*(1.0+comp(elem,pt)*(psol(elem,pt) - pref(elem,pt)));
        AD M = porosity(elem,pt)*densref(elem,pt)*comp(elem,pt)*pdot(elem,pt) - source(elem,pt);
        M *= wts(elem,pt);
        AD Kx = Kdens*pgrad(elem,pt,0)*wts(elem,pt);
        AD Ky = Kdens*pgrad(elem,pt,1)*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += M*basis(elem,dof,pt) + Kx*basis_grad(elem,dof,pt,0) + Ky*basis_grad(elem,dof,pt,1);
        }
      }
    });
  }
  else if (spaceDim == 3) {
    parallel_for("porous HGRAD volume resid 3D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<psol.extent(1); pt++ ) {
        AD Kdens = perm(elem,pt)/viscosity(elem,pt)*densref(elem,pt)*(1.0+comp(elem,pt)*(psol(elem,pt) - pref(elem,pt)));
        AD M = porosity(elem,pt)*densref(elem,pt)*comp(elem,pt)*pdot(elem,pt) - source(elem,pt);
        M *= wts(elem,pt);
        AD Kx = Kdens*pgrad(elem,pt,0)*wts(elem,pt);
        AD Ky = Kdens*pgrad(elem,pt,1)*wts(elem,pt);
        AD Kz = Kdens*pgrad(elem,pt,2)*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += M*basis(elem,dof,pt) + Kx*basis_grad(elem,dof,pt,0) + Ky*basis_grad(elem,dof,pt,1) + Kz*basis_grad(elem,dof,pt,2);
        }
      }
    });
  }
  
}


// ========================================================================================
// ========================================================================================

void porous::boundaryResidual() {
  
  
  bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  int sidetype = bcs(pnum,cside);
  
  int basis_num = wkset->usebasis[pnum];
  int numBasis = wkset->basis_side[basis_num].extent(1);
  auto basis = wkset->basis_side[basis_num];
  auto basis_grad = wkset->basis_grad_side[basis_num];
  
  FDATA perm, porosity, viscosity, densref, pref, comp, gravity, source;
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (sidetype == 4 ) {
      source = functionManager->evaluate("Dirichlet p " + wkset->sidename,"side ip");
    }
    else if (sidetype == 2) {
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
  
  // Since normals, wts and h get re-directed often, these need to be reset
  auto normals = wkset->normals;
  auto wts = wkset->wts_side;
  auto h = wkset->h;
  auto res = wkset->res;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  auto psol = Kokkos::subview(sol, Kokkos::ALL(), pnum, Kokkos::ALL(), 0);
  auto pgrad = Kokkos::subview(sol_grad, Kokkos::ALL(), pnum, Kokkos::ALL(), Kokkos::ALL());
  auto off = Kokkos::subview(offsets, pnum, Kokkos::ALL());
  
  if (bcs(pnum,cside) == 2) { //Neumann
    parallel_for("porous HGRAD bndry resid Neumann",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<basis.extent(2); pt++ ) {
        AD s = -source(elem,pt)*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += s*basis(elem,dof,pt);
        }
      }
    });
  }
  else if (bcs(pnum,cside) == 4) { // weak Dirichlet
    parallel_for("porous HGRAD bndry resid weak Dirichlet",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<basis.extent(2); pt++ ) {
        AD pval = psol(elem,pt);
        AD dens = densref(elem,pt)*(1.0+comp(elem,pt)*(pval - pref(elem,pt)));
        AD Kval = perm(elem,pt)/viscosity(elem,pt)*dens;
        AD weakDiriScale = 10.0*Kval/h(elem);
        AD Kgradp_dot_n = 0.0;
        for (int dim=0; dim<normals.extent(2); dim++) {
          Kgradp_dot_n += Kval*pgrad(elem,pt,dim)*normals(elem,pt,dim);
        }
        Kgradp_dot_n *= wts(elem,pt);
        AD pdiff = (pval - source(elem,pt))*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT v = basis(elem,dof,pt);
          AD Kgradv_dot_n = 0.0;
          for (int dim=0; dim<normals.extent(2); dim++) {
            Kgradv_dot_n += Kval*basis_grad(elem,dof,pt,dim)*normals(elem,pt,dim);
          }
          res(elem,off(dof)) += -Kgradp_dot_n*v - sf*Kgradv_dot_n*pdiff + weakDiriScale*pdiff*v;
        }
      }
    });
  }
  else if (bcs(pnum,cside) == 5) { // multiscale weak Dirichlet
    auto lambda = Kokkos::subview(aux_side,Kokkos::ALL(), pnum, Kokkos::ALL());
    parallel_for("porous HGRAD bndry resid MS weak Dirichlet",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<basis.extent(2); pt++ ) {
        AD pval = psol(elem,pt);
        AD dens = densref(elem,pt)*(1.0+comp(elem,pt)*(pval - pref(elem,pt)));
        AD Kval = perm(elem,pt)/viscosity(elem,pt)*dens;
        AD weakDiriScale = 10.0*Kval/h(elem);
        AD Kgradp_dot_n = 0.0;
        for (int dim=0; dim<normals.extent(2); dim++) {
          Kgradp_dot_n += Kval*pgrad(elem,pt,dim)*normals(elem,pt,dim);
        }
        Kgradp_dot_n *= wts(elem,pt);
        AD pdiff = (pval - lambda(elem,pt))*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT v = basis(elem,dof,pt);
          AD Kgradv_dot_n = 0.0;
          for (int dim=0; dim<normals.extent(2); dim++) {
            Kgradv_dot_n += Kval*basis_grad(elem,dof,pt,dim)*normals(elem,pt,dim);
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
  
  ScalarT sf = 1.0; // TMW: not on device
  if (wkset->isAdjoint) {
    sf = formparam;
  }
  
  FDATA perm, porosity, viscosity, densref, pref, comp, gravity, source;
  
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    perm = functionManager->evaluate("permeability","side ip");
    viscosity = functionManager->evaluate("viscosity","side ip");
    densref = functionManager->evaluate("reference density","side ip");
    pref = functionManager->evaluate("reference pressure","side ip");
    comp = functionManager->evaluate("compressibility","side ip");
    gravity = functionManager->evaluate("gravity","side ip");
    
  }
  
  // Since normals get recomputed often, this needs to be reset
  auto normals = wkset->normals;
  auto h = wkset->h;
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    auto pflux = Kokkos::subview(flux, Kokkos::ALL(), pnum, Kokkos::ALL());
    auto psol = Kokkos::subview(sol_side, Kokkos::ALL(), pnum, Kokkos::ALL(), 0);
    auto pgrad = Kokkos::subview(sol_grad_side, Kokkos::ALL(), pnum, Kokkos::ALL(), Kokkos::ALL());
    auto lambda = Kokkos::subview(aux_side, Kokkos::ALL(), pnum, Kokkos::ALL());
    parallel_for("porous HGRAD flux",RangePolicy<AssemblyExec>(0,normals.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      
      for (size_t pt=0; pt<pflux.extent(1); pt++) {
        AD dens = densref(elem,pt)*(1.0+comp(elem,pt)*(psol(elem,pt) - pref(elem,pt)));
        AD Kval = perm(elem,pt)/viscosity(elem,pt)*dens;
        
        AD penalty = 10.0*Kval/h(elem);
        AD Kgradp_dot_n  = 0.0;
        for (int dim=0; dim<normals.extent(2); dim++) {
          Kgradp_dot_n += Kval*pgrad(elem,pt,dim)*normals(elem,pt,dim);
        }
        if (normals.extent(2)>2) {
          Kgradp_dot_n += -Kval*gravity(elem,pt)*dens*normals(elem,pt,2);
        }
        pflux(elem,pt) += sf*Kgradp_dot_n + penalty*(lambda(elem,pt)-psol(elem,pt));
        
      }
    });
  }
  
}

// ========================================================================================
// ========================================================================================

void porous::setVars(std::vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "p") {
      pnum = i;
    }
  }
}

// ========================================================================================
// ========================================================================================

void porous::updatePerm(FDATA perm) {
  
  Kokkos::View<ScalarT**,AssemblyDevice> data = wkset->extra_data;
  
  parallel_for("porous HGRAD update perm",RangePolicy<AssemblyExec>(0,perm.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (int pt=0; pt<perm.extent(1); pt++) {
      perm(elem,pt) = data(elem,0);
    }
  });
}
