/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "maxwell.hpp"

maxwell::maxwell(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  label = "maxwell";
  spaceDim = settings->sublist("Mesh").get<int>("dim",3);
  
  myvars.push_back("E");
  myvars.push_back("B");
  mybasistypes.push_back("HCURL");
  mybasistypes.push_back("HDIV");
}

// ========================================================================================
// ========================================================================================

void maxwell::defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                              Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("current x",fs.get<string>("current x","0.0"),"ip");
  functionManager->addFunction("current y",fs.get<string>("current y","0.0"),"ip");
  functionManager->addFunction("current z",fs.get<string>("current z","0.0"),"ip");
  functionManager->addFunction("mu",fs.get<string>("mu","1.0"),"ip");
  functionManager->addFunction("epsilon",fs.get<string>("epsilon","1.0"),"ip");
  
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
    current_x = functionManager->evaluate("current x","ip");
    current_y = functionManager->evaluate("current y","ip");
    current_z = functionManager->evaluate("current z","ip");
    mu = functionManager->evaluate("mu","ip");
    epsilon = functionManager->evaluate("epsilon","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  {
    basis = wkset->basis[B_basis];
    
    auto dBdt = Kokkos::subview(sol_dot, Kokkos::ALL(), Bnum, Kokkos::ALL(), Kokkos::ALL());
    auto curlE = Kokkos::subview(sol_curl, Kokkos::ALL(), Enum, Kokkos::ALL(), Kokkos::ALL());
    auto off = Kokkos::subview(offsets, Bnum, Kokkos::ALL());
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      
      // (dB/dt,V) + (curl E,V) = (S_mag,V)
      for (int k=0; k<sol.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          res(e,off(i)) += (dBdt(e,k,0) + curlE(e,k,0))*basis(e,i,k,0) +
          (dBdt(e,k,1) + curlE(e,k,1))*basis(e,i,k,1) +
          (dBdt(e,k,2) + curlE(e,k,2))*basis(e,i,k,2);
        }
      }
      
    });
  }
  
  {
    basis = wkset->basis[E_basis];
    basis_curl = wkset->basis_curl[E_basis];
    auto dEdt = Kokkos::subview(sol_dot, Kokkos::ALL(), Enum, Kokkos::ALL(), Kokkos::ALL());
    auto B = Kokkos::subview(sol, Kokkos::ALL(), Bnum, Kokkos::ALL(), Kokkos::ALL());
    auto off = Kokkos::subview(offsets, Enum, Kokkos::ALL());
    
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      
      // (eps*dE/dt,V) - (1/mu B, curl V) = (S_elec,V)
      for (int k=0; k<sol.extent(2); k++ ) {
        for (int i=0; i<basis.extent(1); i++ ) {
          
          res(e,off(i)) += epsilon(e,k)*dEdt(e,k,0)*basis(e,i,k,0) - B(e,k,0)*basis_curl(e,i,k,0) + current_x(e,k)*basis(e,i,k,0);
          res(e,off(i)) += epsilon(e,k)*dEdt(e,k,1)*basis(e,i,k,1) - B(e,k,1)*basis_curl(e,i,k,1) + current_y(e,k)*basis(e,i,k,1);
          res(e,off(i)) += epsilon(e,k)*dEdt(e,k,2)*basis(e,i,k,2) - B(e,k,2)*basis_curl(e,i,k,2) + current_z(e,k)*basis(e,i,k,2);
          
        }
      }
      
    });
  }
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

void maxwell::setVars(std::vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "E")
      Enum = i;
    if (varlist[i] == "B")
      Bnum = i;
  }
}
