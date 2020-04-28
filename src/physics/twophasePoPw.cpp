/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "twophasePoPw.hpp"

twophasePoPw::twophasePoPw(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
                           const size_t & numip_side_, const int & numElem_,
                           Teuchos::RCP<FunctionManager> & functionManager_) :
numip(numip_), numip_side(numip_side_), numElem(numElem_) {
  
  // Standard data
  functionManager = functionManager_;
  label = "twophase";
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  myvars.push_back("Po");
  myvars.push_back("Pw");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
  
  
  // Functions
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("permeability",fs.get<string>("permeability","1.0"),numElem,numip,"ip");
  functionManager->addFunction("porosity",fs.get<string>("porosity","1.0"),numElem,numip,"ip");
  functionManager->addFunction("gravity",fs.get<string>("gravity","1.0"),numElem,numip,"ip");
  functionManager->addFunction("cap press inverse",fs.get<string>("capillary pressure inverse","1.0"),numElem,numip,"ip");
  functionManager->addFunction("dcap press inverse",fs.get<string>("derivative capillary pressure inverse","0.0"),numElem,numip,"ip");
  
  functionManager->addFunction("source oil",fs.get<string>("source oil","0.0"),numElem,numip,"ip");
  functionManager->addFunction("viscosity oil",fs.get<string>("viscosity oil","0.0"),numElem,numip,"ip");
  functionManager->addFunction("relative permeability oil",fs.get<string>("relative permeability oil","1.0"),numElem,numip,"ip");
  functionManager->addFunction("reference density oil",fs.get<string>("reference density oil","1.0"),numElem,numip,"ip");
  functionManager->addFunction("reference pressure oil",fs.get<string>("reference pressure oil","1.0"),numElem,numip,"ip");
  functionManager->addFunction("compressibility oil",fs.get<string>("compressibility oil","0.0"),numElem,numip,"ip");
  
  functionManager->addFunction("source water",fs.get<string>("source water","0.0"),numElem,numip,"ip");
  functionManager->addFunction("viscosity water",fs.get<string>("viscosity water","0.0"),numElem,numip,"ip");
  functionManager->addFunction("relative permeability water",fs.get<string>("relative permeability water","1.0"),numElem,numip,"ip");
  functionManager->addFunction("reference density water",fs.get<string>("reference density water","1.0"),numElem,numip,"ip");
  functionManager->addFunction("reference pressure water",fs.get<string>("reference pressure water","1.0"),numElem,numip,"ip");
  functionManager->addFunction("compressibility water",fs.get<string>("compressibility water","0.0"),numElem,numip,"ip");
  
}

// ========================================================================================
// ========================================================================================

void twophasePoPw::volumeResidual() {
  
  // NOTES:
  // sol, sol_grad, etc. are set by the physics_base class
  
  // This formulation solves for N_o and P_o
  
  // This does assume that both pw and po use the same basis ... easy to generalize
  
  
  int p_basis_num = wkset->usebasis[Ponum];
  basis = wkset->basis[p_basis_num];
  basis_grad = wkset->basis_grad[p_basis_num];
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    porosity = functionManager->evaluate("porosity","ip");
    perm = functionManager->evaluate("permeability","ip");
    
    relperm_o = functionManager->evaluate("relative permeability oil","ip");
    source_o = functionManager->evaluate("source oil","ip");
    viscosity_o = functionManager->evaluate("viscosity oil","ip");
    densref_o = functionManager->evaluate("reference density oil","ip");
    pref_o = functionManager->evaluate("reference pressure oil","ip");
    comp_o = functionManager->evaluate("compressibility oil","ip");
    
    relperm_w = functionManager->evaluate("relative permeability water","ip");
    source_w = functionManager->evaluate("source water","ip");
    viscosity_w = functionManager->evaluate("viscosity water","ip");
    densref_w = functionManager->evaluate("reference density water","ip");
    pref_w = functionManager->evaluate("reference pressure water","ip");
    comp_w = functionManager->evaluate("compressibility water","ip");
    
    gravity = functionManager->evaluate("gravity","ip");
    cpinv = functionManager->evaluate("cap press inverse","ip");
    dcpinv = functionManager->evaluate("dcap press inverse","ip");
    
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  //AD rho_o, rho_w, S_o, S_w, P_w, dP_o_dt, dS_w_dt, dP_w_dt, drho_w_dt, dN_w_dt;
  //AD dS_w_dx, dP_w_dx, dS_w_dy, dP_w_dy, dS_w_dz, dP_w_dz;
  
  if (spaceDim == 1) {
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<sol.extent(2); k++ ) { // loop over integration points
        AD Po = sol(e,Ponum,k,0);
        AD Pw = sol(e,Pwnum,k,0);
        AD dPo_dx = sol_grad(e,Ponum,k,0);
        AD dPo_dy = sol_grad(e,Ponum,k,1);
        AD dPo_dz = sol_grad(e,Ponum,k,2);
        AD dPw_dx = sol_grad(e,Pwnum,k,0);
        AD dPw_dy = sol_grad(e,Pwnum,k,1);
        AD dPw_dz = sol_grad(e,Pwnum,k,2);
        AD dPw_dt = sol_dot(e,Pwnum,k,0);
        AD dPo_dt = sol_dot(e,Ponum,k,0);
        AD Sw = cpinv(e,k);
        AD So = 1.0 - Sw;
        AD dSw_dt = dcpinv(e,k)*(dPo_dt - dPw_dt);//-1.0/(rhoo*rhoo)*(dNo_dt*rhoo - No*dPo_dt);
        AD dSo_dt = -dSw_dt;
        
        AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
        AD drhoo_dt = densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dt);
        
        AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
        AD drhow_dt = densref_w(e,k)*(0.0+comp_w(e,k)*dPw_dt);
        AD dNw_dt = Sw*drhow_dt + dSw_dt*rhow;
        AD dNo_dt = So*drhoo_dt + dSo_dt*rhoo;
        
        for (int i=0; i<basis.extent(1); i++ ) { // loop over basis functions
          
          // No equation
          resindex = offsets(Pwnum,i);
          
          res(e,resindex) += porosity(e,k)*dNo_dt*basis(e,i,k) + // transient term
          perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo*(dPo_dx*basis_grad(e,i,k,0)) // diffusion terms
          - source_o(e,k)*basis(e,i,k);
          
          // Po equation
          resindex = offsets(Ponum,i);
          
          res(e,resindex) += porosity(e,k)*dNw_dt*basis(e,i,k) + // transient term
          perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow*(dPw_dx*basis_grad(e,i,k,0)) // diffusion terms
          -source_w(e,k)*basis(e,i,k);
        }
      }
    });
  }
  else if (spaceDim == 2) {
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<sol.extent(2); k++ ) {
        AD Po = sol(e,Ponum,k,0);
        AD Pw = sol(e,Pwnum,k,0);
        AD dPo_dx = sol_grad(e,Ponum,k,0);
        AD dPo_dy = sol_grad(e,Ponum,k,1);
        AD dPo_dz = sol_grad(e,Ponum,k,2);
        AD dPw_dx = sol_grad(e,Pwnum,k,0);
        AD dPw_dy = sol_grad(e,Pwnum,k,1);
        AD dPw_dz = sol_grad(e,Pwnum,k,2);
        AD dPw_dt = sol_dot(e,Pwnum,k,0);
        AD dPo_dt = sol_dot(e,Ponum,k,0);
        AD Sw = cpinv(e,k);
        AD So = 1.0 - Sw;
        AD dSw_dt = dcpinv(e,k)*(dPo_dt - dPw_dt);//-1.0/(rhoo*rhoo)*(dNo_dt*rhoo - No*dPo_dt);
        AD dSo_dt = -dSw_dt;
        
        AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
        AD drhoo_dt = densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dt);
        
        AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
        AD drhow_dt = densref_w(e,k)*(0.0+comp_w(e,k)*dPw_dt);
        AD dNw_dt = Sw*drhow_dt + dSw_dt*rhow;
        AD dNo_dt = So*drhoo_dt + dSo_dt*rhoo;
        
        for (int i=0; i<basis.extent(1); i++ ) { // loop over basis functions
          
          // No equation
          resindex = offsets(Ponum,i);
          
          res(e,resindex) += porosity(e,k)*dNo_dt*basis(e,i,k) + // transient term
          perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo*(dPo_dx*basis_grad(e,i,k,0) +
                                                          dPo_dy*basis_grad(e,i,k,1)) // diffusion terms
          -source_o(e,k)*basis(e,i,k);
          
          // Po equation
          resindex = offsets(Pwnum,i);
          
          res(e,resindex) += porosity(e,k)*dNw_dt*basis(e,i,k) + // transient term
          perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow*(dPw_dx*basis_grad(e,i,k,0) +
                                                          dPw_dy*basis_grad(e,i,k,1)) // diffusion terms
          -source_w(e,k)*basis(e,i,k);
        }
      }
    });
  }
  else if (spaceDim == 3) {
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<sol.extent(2); k++ ) {
        AD Po = sol(e,Ponum,k,0);
        AD Pw = sol(e,Pwnum,k,0);
        AD dPo_dx = sol_grad(e,Ponum,k,0);
        AD dPo_dy = sol_grad(e,Ponum,k,1);
        AD dPo_dz = sol_grad(e,Ponum,k,2);
        AD dPw_dx = sol_grad(e,Pwnum,k,0);
        AD dPw_dy = sol_grad(e,Pwnum,k,1);
        AD dPw_dz = sol_grad(e,Pwnum,k,2);
        AD dPw_dt = sol_dot(e,Pwnum,k,0);
        AD dPo_dt = sol_dot(e,Ponum,k,0);
        AD Sw = cpinv(e,k);
        AD So = 1.0 - Sw;
        AD dSw_dt = dcpinv(e,k)*(dPo_dt - dPw_dt);//-1.0/(rhoo*rhoo)*(dNo_dt*rhoo - No*dPo_dt);
        AD dSo_dt = -dSw_dt;
        
        AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
        AD drhoo_dt = densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dt);
        
        AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
        AD drhow_dt = densref_w(e,k)*(0.0+comp_w(e,k)*dPw_dt);
        AD dNw_dt = Sw*drhow_dt + dSw_dt*rhow;
        AD dNo_dt = So*drhoo_dt + dSo_dt*rhoo;
        
        for (int i=0; i<basis.extent(1); i++ ) { // loop over basis functions
          
          // Po equation
          resindex = offsets(Pwnum,i);
          
          res(e,resindex) += porosity(e,k)*dNo_dt*basis(e,i,k) + // transient term
          perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo*(dPo_dx*basis_grad(e,i,k,0) +
                                                          dPo_dy*basis_grad(e,i,k,1) +
                                                          dPo_dz*basis_grad(e,i,k,2) -
                                                          rhoo*gravity(e,k)*1.0) // diffusion terms
          -source_o(e,k)*basis(e,i,k);
          
          // Pw equation
          resindex = offsets(Ponum,i);
          
          res(e,resindex) += porosity(e,k)*dNw_dt*basis(e,i,k) + // transient term
          perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow*(dPw_dx*basis_grad(e,i,k,0) +
                                                          dPw_dy*basis_grad(e,i,k,1) +
                                                          dPw_dz*basis_grad(e,i,k,2) -
                                                          rhow*gravity(e,k)*1.0) // diffusion terms
          -source_w(e,k)*basis(e,i,k);
          
        }
      }
    });
  }
  
}


// ========================================================================================
// ========================================================================================

void twophasePoPw::boundaryResidual() {
  
  sideinfo = wkset->sideinfo;
  Kokkos::View<int**,AssemblyDevice> bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  int sidetype = bcs(Ponum,cside); // TMW to do: allow No to use different BCs
  
  int basis_num = wkset->usebasis[Ponum];
  int numBasis = wkset->basis_side[basis_num].extent(1);
  basis = wkset->basis_side[basis_num];
  basis_grad = wkset->basis_grad_side[basis_num];
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (sidetype == 4 ) {
      source_w = functionManager->evaluate("Dirichlet Pw " + wkset->sidename,"side ip");
      source_o = functionManager->evaluate("Dirichlet No " + wkset->sidename,"side ip");
    }
    else if (sidetype == 2) {
      source_w = functionManager->evaluate("Neumann Pw " + wkset->sidename,"side ip");
      source_o = functionManager->evaluate("Neumann No " + wkset->sidename,"side ip");
    }
    
    perm = functionManager->evaluate("permeability","side ip");
    relperm_o = functionManager->evaluate("relative permeability oil","side ip");
    source_o = functionManager->evaluate("source oil","side ip");
    viscosity_o = functionManager->evaluate("viscosity oil","side ip");
    densref_o = functionManager->evaluate("reference density oil","side ip");
    pref_o = functionManager->evaluate("reference pressure oil","side ip");
    comp_o = functionManager->evaluate("compressibility oil","side ip");
    
    relperm_w = functionManager->evaluate("relative permeability water","side ip");
    source_w = functionManager->evaluate("source water","side ip");
    viscosity_w = functionManager->evaluate("viscosity water","side ip");
    densref_w = functionManager->evaluate("reference density water","side ip");
    pref_w = functionManager->evaluate("reference pressure water","side ip");
    comp_w = functionManager->evaluate("compressibility water","side ip");
    
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
    if (bcs(Ponum,cside) == 2) {
      for (int k=0; k<basis.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          resindex = offsets(Ponum,i);
          res(e,resindex) += -source_o(e,k)*basis(e,i,k);
          
          resindex = offsets(Pwnum,i); // index into Pw eqn
          res(e,resindex) += -source_w(e,k)*basis(e,i,k);
        }
      }
    }
    
    if (bcs(Ponum,cside) == 4 || bcs(Ponum,cside) == 5) {
      
      for (int k=0; k<basis.extent(2); k++ ) {
        
        AD lambda_Po, lambda_Pw;
        
        if (bcs(Ponum,cside) == 5) {
          lambda_Po = aux_side(e,Ponum,k);
          lambda_Pw = aux_side(e,Pwnum,k);
        }
        else {
          lambda_Po = source_o(e,k);
          lambda_Pw = source_w(e,k);
        }
        AD Po = sol_side(e,Ponum,k,0);
        AD Pw = sol_side(e,Pwnum,k,0);
        AD dPo_dx = sol_grad_side(e,Ponum,k,0);
        AD dPo_dy = sol_grad_side(e,Ponum,k,1);
        AD dPo_dz = sol_grad_side(e,Ponum,k,2);
        AD dPw_dx = sol_grad_side(e,Pwnum,k,0);
        AD dPw_dy = sol_grad_side(e,Pwnum,k,1);
        AD dPw_dz = sol_grad_side(e,Pwnum,k,2);
        
        AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
        AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
        
        for (int i=0; i<basis.extent(1); i++ ) {
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          if (spaceDim > 1)
            dvdy = basis_grad(e,i,k,1);
          if (spaceDim > 2)
            dvdz = basis_grad(e,i,k,2);
          
          // Po eqn
          AD Kval = perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo;
          AD weakDiriScale = 10.0*Kval/wkset->h(e);
          
          resindex = offsets(Ponum,i);
          
          res(e,resindex) += -Kval*dPo_dx*normals(e,k,0)*v - sf*Kval*dvdx*normals(e,k,0)*(Po-lambda_Po) + weakDiriScale*(Po-lambda_Po)*v;
          if (spaceDim > 1) {
            res(e,resindex) += -Kval*dPo_dy*normals(e,k,1)*v - sf*Kval*dvdy*normals(e,k,1)*(Po-lambda_Po);
          }
          if (spaceDim > 2) {
            res(e,resindex) += -Kval*(dPo_dz - rhow*gravity(e,k)*rhoo)*normals(e,k,2)*v - sf*Kval*dvdz*normals(e,k,2)*(Po-lambda_Po);
          }
          
          // Pw eqn
          Kval = perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow;
          weakDiriScale = 10.0*Kval/wkset->h(e);
          
          resindex = offsets(Pwnum,i);
          
          res(e,resindex) += -Kval*dPw_dx*normals(e,k,0)*v - sf*Kval*dvdx*normals(e,k,0)*(Pw-lambda_Pw) + weakDiriScale*(Pw-lambda_Pw)*v;
          if (spaceDim > 1) {
            res(e,resindex) += -Kval*dPw_dy*normals(e,k,1)*v - sf*Kval*dvdy*normals(e,k,1)*(Pw-lambda_Pw);
          }
          if (spaceDim > 2) {
            res(e,resindex) += -Kval*(dPw_dz - rhow*gravity(e,k)*rhow)*normals(e,k,2)*v - sf*Kval*dvdz*normals(e,k,2)*(Pw-lambda_Pw);
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

void twophasePoPw::edgeResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void twophasePoPw::computeFlux() {
  
  ScalarT sf = 1.0;
  if (wkset->isAdjoint) {
    sf = formparam;
  }
  
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    perm = functionManager->evaluate("permeability","side ip");
    relperm_o = functionManager->evaluate("relative permeability oil","side ip");
    source_o = functionManager->evaluate("source oil","side ip");
    viscosity_o = functionManager->evaluate("viscosity oil","side ip");
    densref_o = functionManager->evaluate("reference density oil","side ip");
    pref_o = functionManager->evaluate("reference pressure oil","side ip");
    comp_o = functionManager->evaluate("compressibility oil","side ip");
    
    relperm_w = functionManager->evaluate("relative permeability water","side ip");
    source_w = functionManager->evaluate("source water","side ip");
    viscosity_w = functionManager->evaluate("viscosity water","side ip");
    densref_w = functionManager->evaluate("reference density water","side ip");
    pref_w = functionManager->evaluate("reference pressure water","side ip");
    comp_w = functionManager->evaluate("compressibility water","side ip");
    
    gravity = functionManager->evaluate("gravity","side ip");
    
  }
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    for (int e=0; e<numElem; e++) {
      
      for (size_t k=0; k<wkset->ip_side.extent(1); k++) {
        AD lambda_Po, lambda_Pw;
        
        lambda_Po = aux_side(e,Ponum,k);
        lambda_Pw = aux_side(e,Pwnum,k);
        
        AD Po = sol_side(e,Ponum,k,0);
        AD Pw = sol_side(e,Pwnum,k,0);
        AD dPo_dx = sol_grad_side(e,Ponum,k,0);
        AD dPo_dy = sol_grad_side(e,Ponum,k,1);
        AD dPo_dz = sol_grad_side(e,Ponum,k,2);
        AD dPw_dx = sol_grad_side(e,Pwnum,k,0);
        AD dPw_dy = sol_grad_side(e,Pwnum,k,1);
        AD dPw_dz = sol_grad_side(e,Pwnum,k,2);
        
        AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
        AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
        
        AD Kval_o = perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo;
        AD penalty_o = 10.0*Kval_o/wkset->h(e);
        AD Kval_w = perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow;
        AD penalty_w = 10.0*Kval_w/wkset->h(e);
        
        flux(e,Ponum,k) += sf*Kval_o*dPo_dx*normals(e,k,0) + penalty_o*(lambda_Po-Po);
        flux(e,Pwnum,k) += sf*Kval_w*dPw_dx*normals(e,k,0) + penalty_w*(lambda_Pw-Pw);
        if (spaceDim > 1) {
          flux(e,Ponum,k) += sf*Kval_o*dPo_dy*normals(e,k,1);
          flux(e,Pwnum,k) += sf*Kval_w*dPw_dy*normals(e,k,1);
        }
        if (spaceDim > 2) {
          flux(e,Ponum,k) += sf*Kval_o*(dPo_dz - gravity(e,k)*rhoo)*normals(e,k,2);
          flux(e,Pwnum,k) += sf*Kval_w*(dPw_dz - gravity(e,k)*rhow)*normals(e,k,2);
        }
      }
    }
  }
  
}

// ========================================================================================
// ========================================================================================

void twophasePoPw::setVars(std::vector<string> & varlist_) {
  varlist = varlist_;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "Po") {
      Ponum = i;
    }
    if (varlist[i] == "Pw") {
      Pwnum = i;
    }
  }
}
