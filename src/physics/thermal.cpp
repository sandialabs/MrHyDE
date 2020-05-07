/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "thermal.hpp"


// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

thermal::thermal(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  // Standard data
  label = "thermal";
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  if (settings->sublist("Physics").isSublist("Active variables")) {
    if (settings->sublist("Physics").sublist("Active variables").isParameter("e")) {
      myvars.push_back("e");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("e","HGRAD"));
    }
  }
  else {
    myvars.push_back("e");
    mybasistypes.push_back("HGRAD");
  }
  // Extra data
  formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
  have_nsvel = false;
  
}

// ========================================================================================
// ========================================================================================

void thermal::defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                              Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("thermal source",fs.get<string>("thermal source","0.0"),"ip");
  functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),"ip");
  functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),"ip");
  functionManager->addFunction("density",fs.get<string>("density","1.0"),"ip");
  functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),"side ip");
  functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),"side ip");
  
}

// ========================================================================================
// ========================================================================================

void thermal::volumeResidual() {
  
  int e_basis_num = wkset->usebasis[e_num];
  basis = wkset->basis[e_basis_num];
  basis_grad = wkset->basis_grad[e_basis_num];
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("thermal source","ip");
    diff = functionManager->evaluate("thermal diffusion","ip");
    cp = functionManager->evaluate("specific heat","ip");
    rho = functionManager->evaluate("density","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  // Contributes:
  // (f(u),v) + (DF(u),nabla v)
  // f(u) = rho*cp*de/dt - source
  // DF(u) = diff*grad(e)
  
  auto T = Kokkos::subview( sol, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
  auto dTdt = Kokkos::subview( sol_dot, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
  auto gradT = Kokkos::subview( sol_grad, Kokkos::ALL(), e_num, Kokkos::ALL(), Kokkos::ALL());
  auto off = Kokkos::subview( offsets, e_num, Kokkos::ALL());
  
  if (spaceDim == 1) {
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<sol.extent(2); k++ ) {
        AD f = rho(e,k)*cp(e,k)*dTdt(e,k) - source(e,k);
        AD DFx = diff(e,k)*gradT(e,k,0);
        for (int i=0; i<basis.extent(1); i++ ) {
          res(e,off(i)) += f*basis(e,i,k) + DFx*basis_grad(e,i,k,0);
        }
      }
    });
  }
  else if (spaceDim == 2) {
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<sol.extent(2); k++ ) {
        AD f = rho(e,k)*cp(e,k)*dTdt(e,k) - source(e,k);
        AD DFx = diff(e,k)*gradT(e,k,0);
        AD DFy = diff(e,k)*gradT(e,k,1);
        for (int i=0; i<basis.extent(1); i++ ) {
          res(e,off(i)) += f*basis(e,i,k) + DFx*basis_grad(e,i,k,0) + DFy*basis_grad(e,i,k,1);
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<sol.extent(2); k++ ) {
        AD f = rho(e,k)*cp(e,k)*dTdt(e,k) - source(e,k);
        AD DFx = diff(e,k)*gradT(e,k,0);
        AD DFy = diff(e,k)*gradT(e,k,1);
        AD DFz = diff(e,k)*gradT(e,k,2);
        for (int i=0; i<basis.extent(1); i++ ) {
          res(e,off(i)) += f*basis(e,i,k) + DFx*basis_grad(e,i,k,0) + DFy*basis_grad(e,i,k,1) + DFz*basis_grad(e,i,k,2);
        }
      }
    });
  }
  
  // Contributes:
  // (f(u),v)
  // f(u) = U * grad(e) (U from Navier Stokes)
  
  if (have_nsvel) {
    if (spaceDim == 1) {
      auto Ux = Kokkos::subview( sol, Kokkos::ALL(), ux_num, Kokkos::ALL(), 0);
      parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.extent(2); k++ ) {
          AD f = Ux(e,k)*gradT(e,k,0);
          for (int i=0; i<basis.extent(1); i++ ) {
            res(e,off(i)) += f*basis(e,i,k);
          }
        }
      });
    }
    else if (spaceDim == 2) {
      auto Ux = Kokkos::subview( sol, Kokkos::ALL(), ux_num, Kokkos::ALL(), 0);
      auto Uy = Kokkos::subview( sol, Kokkos::ALL(), uy_num, Kokkos::ALL(), 0);
      parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.extent(2); k++ ) {
          AD f = Ux(e,k)*gradT(e,k,0) + Uy(e,k)*gradT(e,k,1);
          for (int i=0; i<basis.extent(1); i++ ) {
            res(e,off(i)) += f*basis(e,i,k);
          }
        }
      });
    }
    else if (spaceDim == 3) {
      auto Ux = Kokkos::subview( sol, Kokkos::ALL(), ux_num, Kokkos::ALL(), 0);
      auto Uy = Kokkos::subview( sol, Kokkos::ALL(), uy_num, Kokkos::ALL(), 0);
      auto Uz = Kokkos::subview( sol, Kokkos::ALL(), uz_num, Kokkos::ALL(), 0);
      parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.extent(2); k++ ) {
          AD f = Ux(e,k)*gradT(e,k,0) + Uy(e,k)*gradT(e,k,1) + Uz(e,k)*gradT(e,k,2);
          for (int i=0; i<basis.extent(1); i++ ) {
            res(e,off(i)) += f*basis(e,i,k);
          }
        }
      });
    }
  }
  
}


// ========================================================================================
// ========================================================================================

void thermal::boundaryResidual() {
  
  bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  int sidetype = bcs(e_num,cside);
  
  int e_basis_num = wkset->usebasis[e_num];
  basis = wkset->basis_side[e_basis_num];
  basis_grad = wkset->basis_grad_side[e_basis_num];
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (sidetype == 4 ) {
      nsource = functionManager->evaluate("Dirichlet e " + wkset->sidename,"side ip");
    }
    else if (sidetype == 2) {
      nsource = functionManager->evaluate("Neumann e " + wkset->sidename,"side ip");
    }
    diff_side = functionManager->evaluate("thermal diffusion","side ip");
    robin_alpha = functionManager->evaluate("robin alpha","side ip");
    
  }
  
  ScalarT sf = formparam;
  if (wkset->isAdjoint) {
    sf = 1.0;
    adjrhs = wkset->adjrhs;
  }
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  auto T = Kokkos::subview( sol_side, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
  auto gradT = Kokkos::subview( sol_grad_side, Kokkos::ALL(), e_num, Kokkos::ALL(), Kokkos::ALL());
  auto off = Kokkos::subview( offsets, e_num, Kokkos::ALL());
  
  // Contributes
  // <g(u),v> + <p(u),grad(v)\cdot n>
  
  if (bcs(e_num,cside) == 2) { // Neumann BCs
    parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<basis.extent(2); k++ ) {
        AD g = -nsource(e,k);
        for (int i=0; i<basis.extent(1); i++ ) {
          res(e,off(i)) += g*basis(e,i,k);
        }
      }
    });
  }
  else if (bcs(e_num,cside) == 4) {
    if (spaceDim == 1) {
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<basis.extent(2); k++ ) {
          AD g = 10.0*diff_side(e,k)/h(e)*(T(e,k)-nsource(e,k)) - diff_side(e,k)*gradT(e,k,0)*normals(e,k,0);
          AD p = -sf*diff_side(e,k)*(T(e,k) - nsource(e,k));
          for (int i=0; i<basis.extent(1); i++ ) {
            ScalarT gradv_dot_n = basis_grad(e,i,k,0)*normals(e,k,0);
            res(e,off(i)) += g*basis(e,i,k) + p*gradv_dot_n;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<basis.extent(2); k++ ) {
          AD g = 10.0*diff_side(e,k)/h(e)*(T(e,k)-nsource(e,k)) - diff_side(e,k)*(gradT(e,k,0)*normals(e,k,0) + gradT(e,k,1)*normals(e,k,1));
          AD p = -sf*diff_side(e,k)*(T(e,k) - nsource(e,k));
          for (int i=0; i<basis.extent(1); i++ ) {
            ScalarT gradv_dot_n = basis_grad(e,i,k,0)*normals(e,k,0) + basis_grad(e,i,k,1)*normals(e,k,1);
            res(e,off(i)) += g*basis(e,i,k) + p*gradv_dot_n;
          }
        }
      });
    }
    else if (spaceDim == 3) {
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<basis.extent(2); k++ ) {
          AD g = 10.0*diff_side(e,k)/h(e)*(T(e,k)-nsource(e,k)) - diff_side(e,k)*(gradT(e,k,0)*normals(e,k,0) + gradT(e,k,1)*normals(e,k,1) + gradT(e,k,2)*normals(e,k,2));
          AD p = -sf*diff_side(e,k)*(T(e,k) - nsource(e,k));
          for (int i=0; i<basis.extent(1); i++ ) {
            ScalarT gradv_dot_n = basis_grad(e,i,k,0)*normals(e,k,0) + basis_grad(e,i,k,1)*normals(e,k,1) + basis_grad(e,i,k,2)*normals(e,k,2);
            res(e,off(i)) += g*basis(e,i,k) + p*gradv_dot_n;
          }
        }
      });
    }
    //if (wkset->isAdjoint) {
    //  adjrhs(e,resindex) += sf*diff_side(e,k)*gradv_dot_n*lambda - weakDiriScale*lambda*basis(e,i,k);
    //}
  }
  else if (bcs(e_num,cside) == 5) {
    auto lambda = Kokkos::subview( aux_side, Kokkos::ALL(), auxe_num, Kokkos::ALL());
    
    if (spaceDim == 1) {
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<basis.extent(2); k++ ) {
          AD g = 10.0*diff_side(e,k)/h(e)*(T(e,k)-lambda(e,k)) - diff_side(e,k)*gradT(e,k,0)*normals(e,k,0);
          AD p = -sf*diff_side(e,k)*(T(e,k) - lambda(e,k));
          for (int i=0; i<basis.extent(1); i++ ) {
            ScalarT gradv_dot_n = basis_grad(e,i,k,0)*normals(e,k,0);
            res(e,off(i)) += g*basis(e,i,k) + p*gradv_dot_n;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<basis.extent(2); k++ ) {
          AD g = 10.0*diff_side(e,k)/h(e)*(T(e,k)-lambda(e,k)) - diff_side(e,k)*(gradT(e,k,0)*normals(e,k,0) + gradT(e,k,1)*normals(e,k,1));
          AD p = -sf*diff_side(e,k)*(T(e,k) - lambda(e,k));
          for (int i=0; i<basis.extent(1); i++ ) {
            ScalarT gradv_dot_n = basis_grad(e,i,k,0)*normals(e,k,0) + basis_grad(e,i,k,1)*normals(e,k,1);
            res(e,off(i)) += g*basis(e,i,k) + p*gradv_dot_n;
          }
        }
      });
    }
    else if (spaceDim == 3) {
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<basis.extent(2); k++ ) {
          AD g = 10.0*diff_side(e,k)/h(e)*(T(e,k)-lambda(e,k)) - diff_side(e,k)*(gradT(e,k,0)*normals(e,k,0) + gradT(e,k,1)*normals(e,k,1) + gradT(e,k,2)*normals(e,k,2));
          AD p = -sf*diff_side(e,k)*(T(e,k) - lambda(e,k));
          for (int i=0; i<basis.extent(1); i++ ) {
            ScalarT gradv_dot_n = basis_grad(e,i,k,0)*normals(e,k,0) + basis_grad(e,i,k,1)*normals(e,k,1) + basis_grad(e,i,k,2)*normals(e,k,2);
            res(e,off(i)) += g*basis(e,i,k) + p*gradv_dot_n;
          }
        }
      });
    }
    //if (wkset->isAdjoint) {
    //  adjrhs(e,resindex) += sf*diff_side(e,k)*gradv_dot_n*lambda - weakDiriScale*lambda*basis(e,i,k);
    //}

  }
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void thermal::computeFlux() {
  
  // TMW: sf is still an issue for GPUs
  ScalarT sf = 1.0;
  if (wkset->isAdjoint) {
    sf = formparam;
  }
  
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    diff_side = functionManager->evaluate("thermal diffusion","side ip");
  }
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto T = Kokkos::subview( sol_side, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
    auto gradT = Kokkos::subview( sol_grad_side, Kokkos::ALL(), e_num, Kokkos::ALL(), Kokkos::ALL());
    auto fluxT = Kokkos::subview( flux, Kokkos::ALL(), e_num, Kokkos::ALL());
    auto lambda = Kokkos::subview( aux_side, Kokkos::ALL(), auxe_num, Kokkos::ALL());
    
    if (spaceDim == 1) {
      parallel_for(RangePolicy<AssemblyExec>(0,flux.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t k=0; k<normals.extent(1); k++) {
          fluxT(e,k) += sf*diff_side(e,k)*gradT(e,k,0)*normals(e,k,0) + 10.0*diff_side(e,k)/h(e)*(lambda(e,k)-T(e,k));
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for(RangePolicy<AssemblyExec>(0,flux.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t i=0; i<normals.extent(1); i++) {
          for (size_t k=0; k<normals.extent(1); k++) {
            fluxT(e,k) += sf*diff_side(e,k)*(gradT(e,k,0)*normals(e,k,0) + gradT(e,k,1)*normals(e,k,1)) + 10.0*diff_side(e,k)/h(e)*(lambda(e,k)-T(e,k));
          }
        }
      });
    }
    else if (spaceDim == 3) {
      parallel_for(RangePolicy<AssemblyExec>(0,flux.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t i=0; i<normals.extent(1); i++) {
          for (size_t k=0; k<normals.extent(1); k++) {
            fluxT(e,k) += sf*diff_side(e,k)*(gradT(e,k,0)*normals(e,k,0) + gradT(e,k,1)*normals(e,k,1) + + gradT(e,k,2)*normals(e,k,2)) + 10.0*diff_side(e,k)/h(e)*(lambda(e,k)-T(e,k));
          }
        }
      });
    }
    
  }
  
}

// ========================================================================================
// ========================================================================================

void thermal::setVars(std::vector<string> & varlist) {
  //varlist = varlist_;
  ux_num = -1;
  uy_num = -1;
  uz_num = -1;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "e")
      e_num = i;
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

// ========================================================================================
// ========================================================================================

void thermal::setAuxVars(std::vector<string> & auxvarlist) {
  
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "e")
      auxe_num = i;
  }
  
}
