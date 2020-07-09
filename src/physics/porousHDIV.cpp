/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "porousHDIV.hpp"

porousHDIV::porousHDIV(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  label = "porousHDIV";
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  
  if (settings->sublist("Physics").isSublist("Active variables")) {
    if (settings->sublist("Physics").sublist("Active variables").isParameter("p")) {
      myvars.push_back("p");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("p","HVOL"));
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("u")) {
      myvars.push_back("u");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("u","HDIV"));
    }
  }
  else {
    myvars.push_back("p");
    myvars.push_back("u");
    mybasistypes.push_back("HVOL");
    mybasistypes.push_back("HDIV");
  }
  dxnum = 0;
  dynum = 0;
  dznum = 0;
  
}

// ========================================================================================
// ========================================================================================

void porousHDIV::defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                                 Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"ip");
  
  
}

// ========================================================================================
// ========================================================================================

void porousHDIV::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int resindex;
  int p_basis = wkset->usebasis[pnum];
  int u_basis = wkset->usebasis[unum];
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
  }
  
  basis = wkset->basis[u_basis];
  basis_div = wkset->basis_div[u_basis];
  wts = wkset->wts;
  
  // (K^-1 u,v) - (p,div v) - src*v (src not added yet)
  parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT vx = 0.0;
    ScalarT vy = 0.0;
    ScalarT vz = 0.0;
    ScalarT divv = 0.0;
    AD uy = 0.0, uz = 0.0;
    
    for (int k=0; k<sol.extent(2); k++ ) {
      for (int i=0; i<basis.extent(1); i++ ) {
        AD p = sol(e,pnum,k,0);
        AD ux = sol(e,unum,k,0);
        
        if (spaceDim > 1) {
          uy = sol(e,unum,k,1);
        }
        if (spaceDim > 2) {
          uz = sol(e,unum,k,2);
        }
        
        vx = basis(e,i,k,0);
        
        if (spaceDim > 1) {
          vy = basis(e,i,k,1);
        }
        if (spaceDim > 2) {
          vz = basis(e,i,k,2);
        }
        divv = basis_div(e,i,k);
        int resindex = offsets(unum,i);
        
        res(e,resindex) += (1.0*(ux*vx+uy*vy+uz*vz) - p*divv)*wts(e,k);
        
      }
    }
    
  });
  
  basis = wkset->basis[p_basis];
  
  // -(div u,q) + src*q (src not added yet)
  parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    
    for (int k=0; k<sol.extent(2); k++ ) {
      for (int i=0; i<basis.extent(1); i++ ) {
        ScalarT v = basis(e,i,k,0);
        AD divu = sol_div(e,unum,k);
        int resindex = offsets(pnum,i);
        res(e,resindex) += (-divu*v + source(e,k)*v)*wts(e,k);
      }
    }
  });
  
}


// ========================================================================================
// ========================================================================================

void porousHDIV::boundaryResidual() {
  
  Kokkos::View<int**,UnifiedDevice> bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  int sidetype;
  sidetype = bcs(pnum,cside);
  
  basis = wkset->basis_side[unum];
  wts = wkset->wts_side;
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (sidetype == 1 ) {
      bsource = functionManager->evaluate("Dirichlet p " + wkset->sidename,"side ip");
    }
    
  }
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  ScalarT vx = 0.0, vy = 0.0, vz = 0.0;
  ScalarT nx = 0.0, ny = 0.0, nz = 0.0;
  for (int e=0; e<basis.extent(0); e++) {
    if (bcs(pnum,cside) == 1) {
      for (int k=0; k<basis.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          vx = basis(e,i,k,0);
          nx = normals(e,k,0);
          if (spaceDim>1) {
            vy = basis(e,i,k,1);
            ny = normals(e,k,1);
          }
          if (spaceDim>2) {
            vz = basis(e,i,k,2);
            nz = normals(e,k,2);
          }
          int resindex = offsets(unum,i);
          res(e,resindex) += (bsource(e,k)*(vx*nx+vy*ny+vz*nz))*wts(e,k);
        }
      }
    }
    else if (bcs(pnum,cside) == 5) {
      for (int k=0; k<basis.extent(2); k++ ) {
        AD lambda = aux_side(e,auxpnum,k);
        
        for (int i=0; i<basis.extent(1); i++ ) {
          vx = basis(e,i,k,0);
          nx = normals(e,k,0);
          if (spaceDim>1) {
            vy = basis(e,i,k,1);
            ny = normals(e,k,1);
          }
          if (spaceDim>2) {
            vz = basis(e,i,k,2);
            nz = normals(e,k,2);
          }
          int resindex = offsets(unum,i);
          res(e,resindex) += (lambda*(vx*nx+vy*ny+vz*nz))*wts(e,k);
        }
      }
    }
  }
}


// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void porousHDIV::computeFlux() {
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    AD ux = 0.0, uy = 0.0, uz = 0.0;
    ScalarT nx = 0.0, ny = 0.0, nz = 0.0;
    for (int e=0; e<flux.extent(0); e++) {
      
      for (size_t k=0; k<normals.extent(1); k++) {
        ux = sol_side(e,unum,k,0);
        nx = normals(e,k,0);
        
        if (spaceDim > 1) {
          uy = sol_side(e,unum,k,1);
          ny = normals(e,k,1);
        }
        if (spaceDim > 2) {
          uz = sol_side(e,unum,k,2);
          nz = normals(e,k,2);
        }
        flux(e,auxpnum,k) = (ux*nx+uy*ny+uz*nz);
      }
    }
  }
  
}

// ========================================================================================
// ========================================================================================

void porousHDIV::setVars(std::vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "p")
      pnum = i;
    if (varlist[i] == "u")
      unum = i;
    if (varlist[i] == "dx")
      dxnum = i;
    if (varlist[i] == "dy")
      dynum = i;
    if (varlist[i] == "dz")
      dznum = i;
  }
}

// ========================================================================================
// ========================================================================================

void porousHDIV::setAuxVars(std::vector<string> & auxvarlist) {
  
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "p")
      auxpnum = i;
    if (auxvarlist[i] == "lambda")
      auxpnum = i;
    if (auxvarlist[i] == "pbndry")
      auxpnum = i;
    if (auxvarlist[i] == "u")
      auxunum = i;
  }
}
