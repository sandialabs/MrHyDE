/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "maxwell.hpp"

maxwell::maxwell(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
                 const size_t & numip_side_, const int & numElem_,
                 Teuchos::RCP<FunctionManager> & functionManager_,
                 const size_t & blocknum_) :
numip(numip_), numip_side(numip_side_), numElem(numElem_), blocknum(blocknum_) {
  
  label = "maxwell";
  functionManager = functionManager_;
  spaceDim = settings->sublist("Mesh").get<int>("dim",3);
  
  myvars.push_back("E");
  myvars.push_back("B");
  mybasistypes.push_back("HCURL");
  mybasistypes.push_back("HDIV");
  
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("mag source x",fs.get<string>("mag source x","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("mag source y",fs.get<string>("mag source y","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("mag source z",fs.get<string>("mag source z","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("elec source x",fs.get<string>("elec source x","1.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("elec source y",fs.get<string>("elec source y","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("elec source z",fs.get<string>("elec source z","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("mu",fs.get<string>("mu","1.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("epsilon",fs.get<string>("epsilon","1.0"),numElem,numip,"ip",blocknum);
  
}

// ========================================================================================
// ========================================================================================

void maxwell::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int resindex;
  int E_basis = wkset->usebasis[Enum];
  int B_basis = wkset->usebasis[Bnum];
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    mag_source_x = functionManager->evaluate("mag source x","ip",blocknum);
    mag_source_y = functionManager->evaluate("mag source y","ip",blocknum);
    mag_source_z = functionManager->evaluate("mag source z","ip",blocknum);
    elec_source_x = functionManager->evaluate("elec source x","ip",blocknum);
    elec_source_y = functionManager->evaluate("elec source y","ip",blocknum);
    elec_source_z = functionManager->evaluate("elec source z","ip",blocknum);
    mu = functionManager->evaluate("mu","ip",blocknum);
    epsilon = functionManager->evaluate("epsilon","ip",blocknum);
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  basis = wkset->basis[B_basis];
  //basis_curl = wkset->basis_curl[B_basis];
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    
    // (dB/dt,V) + (curl E,V) = (S_mag,V)
    for (int k=0; k<sol.dimension(2); k++ ) {
      for (int i=0; i<basis.dimension(1); i++ ) {
        AD dBx_dt = sol_dot(e,Bnum,k,0);
        AD dBy_dt = sol_dot(e,Bnum,k,1);
        AD dBz_dt = sol_dot(e,Bnum,k,2);
        
        ScalarT vx = basis(e,i,k,0);
        ScalarT vy = basis(e,i,k,1);
        ScalarT vz = basis(e,i,k,2);
        
        AD cEx = sol_curl(e,Enum,k,0);
        AD cEy = sol_curl(e,Enum,k,1);
        AD cEz = sol_curl(e,Enum,k,2);
        
        int resindex = offsets(Bnum,i);
        res(e,resindex) += dBx_dt*vx + cEx*vx - mag_source_x(e,k)*vx;
        res(e,resindex) += dBy_dt*vy + cEy*vy - mag_source_y(e,k)*vy;
        res(e,resindex) += dBz_dt*vz + cEz*vz - mag_source_z(e,k)*vz;
        
        
      }
    }
    
  });
  
  basis = wkset->basis[E_basis];
  basis_curl = wkset->basis_curl[E_basis];
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    
    // (eps*dE/dt,V) - (1/mu B, curl V) = (S_elec,V)
    for (int k=0; k<sol.dimension(2); k++ ) {
      for (int i=0; i<basis.dimension(1); i++ ) {
        AD dEx_dt = sol_dot(e,Enum,k,0);
        AD dEy_dt = sol_dot(e,Enum,k,1);
        AD dEz_dt = sol_dot(e,Enum,k,2);
        
        ScalarT vx = basis(e,i,k,0);
        ScalarT vy = basis(e,i,k,1);
        ScalarT vz = basis(e,i,k,2);
        
        ScalarT cvx = basis_curl(e,i,k,0);
        ScalarT cvy = basis_curl(e,i,k,1);
        ScalarT cvz = basis_curl(e,i,k,2);
        
        AD Bx = sol(e,Bnum,k,0);
        AD By = sol(e,Bnum,k,1);
        AD Bz = sol(e,Bnum,k,2);
        
        int resindex = offsets(Enum,i);
        res(e,resindex) += epsilon(e,k)*dEx_dt*vx - Bx/mu(e,k)*cvx - elec_source_x(e,k)*vx;
        res(e,resindex) += epsilon(e,k)*dEy_dt*vy - By/mu(e,k)*cvy - elec_source_y(e,k)*vy;
        res(e,resindex) += epsilon(e,k)*dEz_dt*vz - Bz/mu(e,k)*cvz - elec_source_z(e,k)*vz;
        
        
      }
    }
    
  });
  
}


// ========================================================================================
// ========================================================================================

void maxwell::boundaryResidual() {
  
  // Nothing implemented yet
  
}


// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void maxwell::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void maxwell::setVars(std::vector<string> & varlist_) {
  varlist = varlist_;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "E")
      Enum = i;
    if (varlist[i] == "B")
      Bnum = i;
  }
}
