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

#include "maxwell.hpp"
using namespace MrHyDE;

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

void maxwell::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("current x",fs.get<string>("current x","0.0"),"ip");
  functionManager->addFunction("current y",fs.get<string>("current y","0.0"),"ip");
  functionManager->addFunction("current z",fs.get<string>("current z","0.0"),"ip");
  functionManager->addFunction("mu",fs.get<string>("permeability","1.0"),"ip");
  functionManager->addFunction("epsilon",fs.get<string>("permittivity","1.0"),"ip");
  functionManager->addFunction("sigma",fs.get<string>("conductivity","0.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void maxwell::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int resindex;
  int E_basis = wkset->usebasis[Enum];
  int B_basis = wkset->usebasis[Bnum];
  
  FDATA mu, epsilon, sigma;
  FDATA current_x, current_y, current_z;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    current_x = functionManager->evaluate("current x","ip");
    current_y = functionManager->evaluate("current y","ip");
    current_z = functionManager->evaluate("current z","ip");
    mu = functionManager->evaluate("mu","ip");
    epsilon = functionManager->evaluate("epsilon","ip");
    sigma = functionManager->evaluate("sigma","ip");
  }
  
  /*
  Kokkos::deep_copy(current_x,0.0);
  auto ip = wkset->ip;
  ScalarT zmin = -1.599e-6;
  ScalarT zmax = 1.599e-6;
  ScalarT wl_center = 3.0e-6;
  ScalarT wl_band = 1.0e-6;
  ScalarT toff = 20.0e-15;
  ScalarT amp = 1.0;
  ScalarT epsilon_ = 8.854187817e-12;
  ScalarT mu_ = 1.2566370614e-6;
  ScalarT c = std::sqrt(1.0/epsilon_/mu_);
  ScalarT fr_center = c / wl_center;
  ScalarT fr_band   = c / (std::pow(wl_center,2) - std::pow(wl_band,2)/4) * wl_band;
    
  ScalarT sigma_ = fr_band/(2.0*std::sqrt(2.0*std::log(2.0)));
  ScalarT time = wkset->time_KV(0);
  parallel_for("Maxwells B volume resid",RangePolicy<AssemblyExec>(0,sigma.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (int pt=0; pt<sigma.extent(1); pt++) {
      ScalarT z = ip(elem,pt,2);
      if (z>=zmin && z<=zmax) {
        current_x(elem,pt) = amp*std::cos(2.0*PI*fr_center*(time - toff))*std::exp(-2.0*std::pow(PI*sigma_*(time-toff),2.0));
      }
    }
  });
  */
  
  //cout << current_x(0,0) << endl;
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  {
    // (dB/dt + curl E,V) = 0
    
    auto basis = wkset->basis[B_basis];
    auto dBdt = Kokkos::subview(sol_dot, Kokkos::ALL(), Bnum, Kokkos::ALL(), Kokkos::ALL());
    auto curlE = Kokkos::subview(sol_curl, Kokkos::ALL(), Enum, Kokkos::ALL(), Kokkos::ALL());
    auto off = Kokkos::subview(offsets, Bnum, Kokkos::ALL());
    auto wts = wkset->wts;
    auto res = wkset->res;
    
    parallel_for("Maxwells B volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<basis.extent(2); pt++ ) {
        AD f0 = (dBdt(elem,pt,0) + curlE(elem,pt,0))*wts(elem,pt);
        AD f1 = (dBdt(elem,pt,1) + curlE(elem,pt,1))*wts(elem,pt);
        AD f2 = (dBdt(elem,pt,2) + curlE(elem,pt,2))*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
          res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
          res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
        }
      }
    });
  }
  
  {
    // (eps*dE/dt,V) - (1/mu B, curl V) + (sigma E,V) = -(current,V)
    // Rewritten as: (eps*dEdt + sigma E + current, V) - (1/mu B, curl V) = 0
    
    auto basis = wkset->basis[E_basis];
    auto basis_curl = wkset->basis_curl[E_basis];
    auto dEdt = Kokkos::subview(sol_dot, Kokkos::ALL(), Enum, Kokkos::ALL(), Kokkos::ALL());
    auto B = Kokkos::subview(sol, Kokkos::ALL(), Bnum, Kokkos::ALL(), Kokkos::ALL());
    auto E = Kokkos::subview(sol, Kokkos::ALL(), Enum, Kokkos::ALL(), Kokkos::ALL());
    auto off = Kokkos::subview(offsets, Enum, Kokkos::ALL());
    auto wts = wkset->wts;
    auto res = wkset->res;
    
    parallel_for("Maxwells E volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<basis.extent(2); pt++ ) {
        AD f0 = (epsilon(elem,pt)*dEdt(elem,pt,0) + sigma(elem,pt)*E(elem,pt,0) + current_x(elem,pt))*wts(elem,pt);
        AD f1 = (epsilon(elem,pt)*dEdt(elem,pt,1) + sigma(elem,pt)*E(elem,pt,1) + current_y(elem,pt))*wts(elem,pt);
        AD f2 = (epsilon(elem,pt)*dEdt(elem,pt,2) + sigma(elem,pt)*E(elem,pt,2) + current_z(elem,pt))*wts(elem,pt);
        AD c0 = - 1.0/mu(elem,pt)*B(elem,pt,0)*wts(elem,pt);
        AD c1 = - 1.0/mu(elem,pt)*B(elem,pt,1)*wts(elem,pt);
        AD c2 = - 1.0/mu(elem,pt)*B(elem,pt,2)*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f0*basis(elem,dof,pt,0) + c0*basis_curl(elem,dof,pt,0);
          res(elem,off(dof)) += f1*basis(elem,dof,pt,1) + c1*basis_curl(elem,dof,pt,1);
          res(elem,off(dof)) += f2*basis(elem,dof,pt,2) + c2*basis_curl(elem,dof,pt,2);
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
