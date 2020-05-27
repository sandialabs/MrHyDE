/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "thermal_enthalpy.hpp"

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

thermal_enthalpy::thermal_enthalpy(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  label = "thermal_enthalpy";
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  
  myvars.push_back("e");
  myvars.push_back("H");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  
  formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
  
  have_nsvel = false;
}

// ========================================================================================
// ========================================================================================

void thermal_enthalpy::defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                                       Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("thermal source",fs.get<string>("thermal source","0.0"),"ip");
  functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),"ip");
  functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),"ip");
  functionManager->addFunction("density",fs.get<string>("density","1.0"),"ip");
  functionManager->addFunction("thermal Neumann source",fs.get<string>("thermal Neumann source","0.0"),"side ip");
  functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),"side ip");
  functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),"side ip");
}

// ========================================================================================
// ========================================================================================

void thermal_enthalpy::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("thermal source","ip");
    diff = functionManager->evaluate("thermal diffusion","ip");
    cp = functionManager->evaluate("specific heat","ip");
    rho = functionManager->evaluate("density","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  basis = wkset->basis[e_basis];
  basis_grad = wkset->basis_grad[e_basis];
  
  parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    ScalarT dvdz = 0.0;
    
    for (int k=0; k<sol.extent(2); k++ ) {
      AD T = sol(e,e_num,k,0);
      AD T_dot = sol_dot(e,e_num,k,0);
      AD dTdx = sol_grad(e,e_num,k,0);
      AD H = sol(e,H_num,k,0);
      AD H_dot = sol_dot(e,H_num,k,0);
      AD dHdx = sol_grad(e,H_num,k,0);
      AD dTdy, dHdy, dTdz, dHdz;
      if (spaceDim > 1) {
        dTdy = sol_grad(e,e_num,k,1);
        dHdy = sol_grad(e,H_num,k,1);
      }
      if (spaceDim > 2) {
        dTdz = sol_grad(e,e_num,k,2);
        dHdz = sol_grad(e,H_num,k,2);
      }
      AD ux, uy, uz;
      if (have_nsvel) {
        ux = sol(e,ux_num,k,0);
        if (spaceDim > 1) {
          uy = sol(e,uy_num,k,1);
        }
        if (spaceDim > 2) {
          uz = sol(e,uz_num,k,2);
        }
      }
      for (int i=0; i<basis.extent(1); i++ ) {
        
        int resindex = offsets(e_num,i);
        v = basis(e,i,k);
        dvdx = basis_grad(e,i,k,0);
        if (spaceDim > 1) {
          dvdy = basis_grad(e,i,k,1);
        }
        if (spaceDim > 2) {
          dvdz = basis_grad(e,i,k,2);
        }
        res(e,resindex) += H_dot*v + diff(e,k)*(dTdx*dvdx + dTdy*dvdy + dTdz*dvdz) - source(e,k)*v;
        if (have_nsvel) {
          res(e,resindex) += (ux*dvdx + uy*dvdy + uz*dvdz);
        }
      }
    }
  });
  
  
  basis = wkset->basis[H_basis];
  basis_grad = wkset->basis_grad[H_basis];
  auto off = Kokkos::subview( offsets, H_num, Kokkos::ALL());
  
  parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    ScalarT dvdz = 0.0;
    
    for (int k=0; k<sol.extent(2); k++ ) {
      AD T = sol(e,e_num,k,0);
      AD H = sol(e,H_num,k,0);
      
      for (int i=0; i<basis.extent(1); i++ ) {
        int resindex = off(i);
        v = basis(e,i,k);
        // make cp_integral and gfunc udfuncs
        //cp_integral = 320.3*e + 0.379/2.0*e*e;
        AD cp_integral = 438.0*T + 0.169/2.0*T*T;
        //        if (e.val() <= 1648.0) {
        AD gfunc;
        if (T.val() <= 1673.0) {
          gfunc = 0.0;
        }
        //else if (e.val() >= 1673.0) {
        else if (T.val() >= 1723.0) {
          gfunc = 1.0;
        }
        else {
          //gfunc = (e - 1648.0)/(1673.0 - 1648.0);
          gfunc = (T - 1673.0)/(1723.0 - 1673.0);
        }
        // T_ref = 293.75
        //wkset->res(resindex) += -(H - rho(k)*cp_integral - rho(k)*latent_heat*gfunc + rho(k)*(320.3*293.75 + 0.379*293.75*293.75/2.0))*v;
        res(e,resindex) += -(H - rho(e,k)*cp_integral - rho(e,k)*latent_heat*gfunc + rho(e,k)*(438.0*293.75 + 0.169*293.75*293.75/2.0))*v;
      }
    }
  });
  
}


// ========================================================================================
// ========================================================================================

void thermal_enthalpy::boundaryResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int e_basis_num = wkset->usebasis[e_num];
  numBasis = wkset->basis_side[e_basis_num].extent(1);
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    nsource = functionManager->evaluate("thermal Neumann source","side ip");
    diff_side = functionManager->evaluate("thermal diffusion","side ip");
    robin_alpha = functionManager->evaluate("robin alpha","side ip");
  }
  
  ScalarT sf = formparam;
  if (wkset->isAdjoint) {
    sf = 1.0;
  }
  
  sideinfo = wkset->sideinfo;
  basis = wkset->basis_side[e_basis_num];
  basis_grad = wkset->basis_grad_side[e_basis_num];
  DRV ip = wkset->ip_side;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  int cside = wkset->currentside;
  for (int e=0; e<sideinfo.extent(0); e++) {
    if (sideinfo(e,e_num,cside,0) == 2) { // Element e is on the side
      for (int k=0; k<basis.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          int resindex = offsets(e_num,i);
          res(e,resindex) += -nsource(e,k)*basis(e,i,k);
        }
      }
    }
    else if (sideinfo(e,e_num,cside,0) == 1){ // Weak Dirichlet
      for (int k=0; k<basis.extent(2); k++ ) {
        AD eval = sol_side(e,e_num,k,0);
        dedx = sol_grad_side(e,e_num,k,0);
        ScalarT x = ip(e,k,0);
        ScalarT y = 0.0;
        ScalarT z = 0.0;
        if (spaceDim > 1) {
          dedy = sol_grad_side(e,e_num,k,1);
          y = ip(e,k,1);
        }
        if (spaceDim > 2) {
          dedz = sol_grad_side(e,e_num,k,2);
          z = ip(e,k,2);
        }
        
        if (sideinfo(e,e_num,cside,1) == -1)
          lambda = aux_side(e,e_num,k);
        else {
          lambda = 0.0; //udfunc->boundaryDirichletValue(label,"e",x,y,z,wkset->time,wkset->sidename,wkset->isAdjoint);
          
          //  lambda = this->getDirichletValue("e", x, y, z, wkset->time,
          //                                  wkset->sidename, wkset->isAdjoint);
        }
        
        for (int i=0; i<basis.extent(1); i++ ) {
          int resindex = offsets(e_num,i);
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          if (spaceDim > 1)
            dvdy = basis_grad(e,i,k,1);
          if (spaceDim > 2)
            dvdz = basis_grad(e,i,k,2);
          
          weakDiriScale = 10.0*diff_side(e,k)/wkset->h(e);
          res(e,resindex) += -diff_side(e,k)*dedx*normals(e,k,0)*v - sf*diff_side(e,k)*dvdx*normals(e,k,0)*(eval-lambda) + weakDiriScale*(eval-lambda)*v;
          if (spaceDim > 1) {
            res(e,resindex) += -diff_side(e,k)*dedy*normals(e,k,1)*v - sf*diff_side(e,k)*dvdy*normals(e,k,1)*(eval-lambda);
          }
          if (spaceDim > 2) {
            res(e,resindex) += -diff_side(e,k)*dedz*normals(e,k,2)*v - sf*diff_side(e,k)*dvdz*normals(e,k,2)*(eval-lambda);
          }
          if (wkset->isAdjoint) {
            adjrhs(e,resindex) += sf*diff_side(e,k)*dvdx*normals(e,k,0)*lambda - weakDiriScale*lambda*v;
            if (spaceDim > 1)
              adjrhs(e,resindex) += sf*diff_side(e,k)*dvdy*normals(e,k,1)*lambda;
            if (spaceDim > 2)
              adjrhs(e,resindex) += sf*diff_side(e,k)*dvdz*normals(e,k,2)*lambda;
          }
        }
      }
    }
    
  }
  
}

// ========================================================================================
// ========================================================================================

void thermal_enthalpy::edgeResidual() {}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void thermal_enthalpy::computeFlux() {
  
  ScalarT sf = 1.0;
  if (wkset->isAdjoint) {
    sf = formparam;
  }
  
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    diff_side = functionManager->evaluate("thermal diffusion","side ip");
  }
  
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    for (int n=0; n<flux.extent(0); n++) {
      
      for (size_t i=0; i<wkset->ip_side.extent(1); i++) {
        penalty = 10.0*diff_side(n,i)/wkset->h(n);
        flux(n,e_num,i) += sf*diff_side(n,i)*sol_grad_side(n,e_num,i,0)*normals(n,i,0) + penalty*(aux_side(n,e_num,i)-sol_side(n,e_num,i,0));
        if (spaceDim > 1) {
          flux(n,e_num,i) += sf*diff_side(n,i)*sol_grad_side(n,e_num,i,1)*normals(n,i,1);
        }
        if (spaceDim > 2) {
          flux(n,e_num,i) += sf*diff_side(n,i)*sol_grad_side(n,e_num,i,2)*normals(n,i,2);
        }
      }
    }
  }
}


// ========================================================================================
// ========================================================================================

void thermal_enthalpy::setVars(std::vector<string> & varlist) {
  ux_num = -1;
  uy_num = -1;
  uz_num = -1;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "e")
      e_num = i;
    if (varlist[i] == "H")
      H_num = i;
    if (varlist[i] == "ux")
      ux_num = i;
    if (varlist[i] == "uy")
      uy_num = i;
    if (varlist[i] == "uz")
      uz_num = i;
  }
  if (ux_num >=0)
    have_nsvel = true;
}

