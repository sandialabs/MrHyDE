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
  
  functionManager->addFunction("current x",fs.get<string>("current x","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("current y",fs.get<string>("current y","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("current z",fs.get<string>("current z","0.0"),numElem,numip,"ip",blocknum);
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
    current_x = functionManager->evaluate("current x","ip",blocknum);
    current_y = functionManager->evaluate("current y","ip",blocknum);
    current_z = functionManager->evaluate("current z","ip",blocknum);
    mu = functionManager->evaluate("mu","ip",blocknum);
    epsilon = functionManager->evaluate("epsilon","ip",blocknum);
  }
  
  //KokkosTools::print(epsilon);
  //KokkosTools::print(mu);
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  basis = wkset->basis[B_basis];
  //basis_curl = wkset->basis_curl[B_basis];
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    
    // (dB/dt,V) + (curl E,V) = (S_mag,V)
    for (int k=0; k<sol.extent(2); k++ ) {
      for (int i=0; i<basis.extent(1); i++ ) {
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
        res(e,resindex) += dBx_dt*vx + cEx*vx;
        res(e,resindex) += dBy_dt*vy + cEy*vy;
        res(e,resindex) += dBz_dt*vz + cEz*vz;
        
        
      }
    }
    
  });
  
  basis = wkset->basis[E_basis];
  basis_curl = wkset->basis_curl[E_basis];
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    
    // (eps*dE/dt,V) - (1/mu B, curl V) = (S_elec,V)
    for (int k=0; k<sol.extent(2); k++ ) {
      for (int i=0; i<basis.extent(1); i++ ) {
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
        res(e,resindex) += epsilon(e,k)*dEx_dt*vx - Bx/mu(e,k)*cvx + current_x(e,k)*vx;
        res(e,resindex) += epsilon(e,k)*dEy_dt*vy - By/mu(e,k)*cvy + current_y(e,k)*vy;
        res(e,resindex) += epsilon(e,k)*dEz_dt*vz - Bz/mu(e,k)*cvz + current_z(e,k)*vz;
        
        
      }
    }
    
  });
  //KokkosTools::print(res);
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
