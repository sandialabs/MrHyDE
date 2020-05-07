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
  basis = wkset->basis[p_basis_num];
  basis_grad = wkset->basis_grad[p_basis_num];
  
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
  
  if (spaceDim == 1) {
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<sol.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          int resindex = offsets(pnum,i); // TMW: e_num is not on the assembly device
          AD dens = densref(e,k)*(1.0+comp(e,k)*(sol(e,pnum,k,0) - pref(e,k)));
          
          res(e,resindex) += porosity(e,k)*densref(e,k)*comp(e,k)*sol_dot(e,pnum,k,0)*basis(e,i,k) + // transient term
          perm(e,k)/viscosity(e,k)*dens*(sol_grad(e,pnum,k,0)*basis_grad(e,i,k,0)) // diffusion terms
          -source(e,k)*basis(e,i,k); // source/well model
          
        }
      }
    });
  }
  else if (spaceDim == 2) {
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<sol.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          int resindex = offsets(pnum,i); // TMW: e_num is not on the assembly device
          AD dens = densref(e,k)*(1.0+comp(e,k)*(sol(e,pnum,k,0) - pref(e,k)));
          
          res(e,resindex) += porosity(e,k)*densref(e,k)*comp(e,k)*sol_dot(e,pnum,k,0)*basis(e,i,k) + // transient term
          perm(e,k)/viscosity(e,k)*dens*(sol_grad(e,pnum,k,0)*basis_grad(e,i,k,0) + sol_grad(e,pnum,k,1)*basis_grad(e,i,k,1)) // diffusion terms
          -source(e,k)*basis(e,i,k); // source/well model
        }
      }
    });
  }
  else if (spaceDim == 3) {
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<sol.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          int resindex = offsets(pnum,i); // TMW: e_num is not on the assembly device
          
          AD dens = densref(e,k)*(1.0+comp(e,k)*(sol(e,pnum,k,0) - pref(e,k)));
          
          res(e,resindex) += porosity(e,k)*densref(e,k)*comp(e,k)*sol_dot(e,pnum,k,0)*basis(e,i,k) + // transient term
          perm(e,k)/viscosity(e,k)*dens*(sol_grad(e,pnum,k,0)*basis_grad(e,i,k,0) + sol_grad(e,pnum,k,1)*basis_grad(e,i,k,1) +
                                         (sol_grad(e,pnum,k,2) - gravity(e,k)*dens*1.0)*basis_grad(e,i,k,2)) // diffusion terms
          -source(e,k)*basis(e,i,k); // source/well model
          
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
  basis = wkset->basis_side[basis_num];
  basis_grad = wkset->basis_grad_side[basis_num];
  
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
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  ScalarT v = 0.0;
  ScalarT dvdx = 0.0;
  ScalarT dvdy = 0.0;
  ScalarT dvdz = 0.0;
  
  for (int e=0; e<basis.extent(0); e++) {
    if (bcs(pnum,cside) == 2) {
      for (int k=0; k<basis.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          int resindex = offsets(pnum,i);
          res(e,resindex) += -source(e,k)*basis(e,i,k);
        }
      }
    }
    
    if (bcs(pnum,cside) == 4 || bcs(pnum,cside) == 5) {
      
      for (int k=0; k<basis.extent(2); k++ ) {
        
        AD pval = sol_side(e,pnum,k,0);
        AD dpdx = sol_grad_side(e,pnum,k,0);
        AD dpdy, dpdz;
        if (spaceDim > 1) {
          dpdy = sol_grad_side(e,pnum,k,1);
        }
        if (spaceDim > 2) {
          dpdz = sol_grad_side(e,pnum,k,2);
        }
        
        AD lambda;
        
        if (bcs(pnum,cside) == 5) {
          lambda = aux_side(e,pnum,k);
        }
        else {
          lambda = source(e,k);
        }
        
        for (int i=0; i<basis.extent(1); i++ ) {
          int resindex = offsets(pnum,i);
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          if (spaceDim > 1)
            dvdy = basis_grad(e,i,k,1);
          if (spaceDim > 2)
            dvdz = basis_grad(e,i,k,2);
          
          AD dens = densref(e,k)*(1.0+comp(e,k)*(pval - pref(e,k)));
          AD Kval = perm(e,k)/viscosity(e,k)*dens;
          AD weakDiriScale = 10.0*Kval/wkset->h(e);
          
          res(e,resindex) += -Kval*dpdx*normals(e,k,0)*v - sf*Kval*dvdx*normals(e,k,0)*(pval-lambda) + weakDiriScale*(pval-lambda)*v;
          if (spaceDim > 1) {
            res(e,resindex) += -Kval*dpdy*normals(e,k,1)*v - sf*Kval*dvdy*normals(e,k,1)*(pval-lambda);
          }
          if (spaceDim > 2) {
            res(e,resindex) += -Kval*(dpdz - gravity(e,k)*dens)*normals(e,k,2)*v - sf*Kval*dvdz*normals(e,k,2)*(pval-lambda);
          }
          
          //if (wkset->isAdjoint) {
          //  adjrhs(e,resindex) += sf*diff_side(e,k)*dvdx*normals(e,k,0)*lambda - weakDiriScale*lambda*v;
          //  if (spaceDim > 1)
          //  adjrhs(e,resindex) += sf*diff_side(e,k)*dvdy*normals(e,k,1)*lambda;
          //  if (spaceDim > 2)
          //  adjrhs(e,resindex) += sf*diff_side(e,k)*dvdz*normals(e,k,2)*lambda;
          //}
        }
        
      }
    }
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
  
  ScalarT sf = 1.0;
  if (wkset->isAdjoint) {
    sf = formparam;
  }
  
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
  normals = wkset->normals;
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    for (int e=0; e<flux.extent(0); e++) {
      
      for (size_t k=0; k<wkset->ip_side.extent(1); k++) {
        AD dens = densref(e,k)*(1.0+comp(e,k)*(sol_side(e,pnum,k,0) - pref(e,k)));
        AD Kval = perm(e,k)/viscosity(e,k)*dens;
        
        AD penalty = 10.0*Kval/wkset->h(e);
        flux(e,pnum,k) += sf*Kval*sol_grad_side(e,pnum,k,0)*normals(e,k,0) +
        penalty*(aux_side(e,pnum,k)-sol_side(e,pnum,k,0));
        if (spaceDim > 1) {
          flux(e,pnum,k) += sf*Kval*sol_grad_side(e,pnum,k,1)*normals(e,k,1);
        }
        if (spaceDim > 2) {
          flux(e,pnum,k) += sf*Kval*(sol_grad_side(e,pnum,k,2) - gravity(e,k)*dens)*normals(e,k,2);
        }
      }
    }
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
