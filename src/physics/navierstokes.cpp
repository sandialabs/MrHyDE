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

#include "navierstokes.hpp"
using namespace MrHyDE;


// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

navierstokes::navierstokes(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  label = "navierstokes";
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  
  verbosity = settings->sublist("Physics").get<int>("Verbosity",0);
  
  myvars.push_back("ux");
  myvars.push_back("pr");
  if (spaceDim > 1) {
    myvars.push_back("uy");
  }
  if (spaceDim > 2) {
    myvars.push_back("uz");
  }
  
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  if (spaceDim > 1) {
    mybasistypes.push_back("HGRAD");
  }
  if (spaceDim > 2) {
    mybasistypes.push_back("HGRAD");
  }
  
  
  useSUPG = settings->sublist("Physics").get<bool>("useSUPG",false);
  usePSPG = settings->sublist("Physics").get<bool>("usePSPG",false);
  T_ambient = settings->sublist("Physics").get<ScalarT>("T_ambient",0.0);
  beta = settings->sublist("Physics").get<ScalarT>("beta",1.0);
  model_params = Kokkos::View<ScalarT*,AssemblyDevice>("NS params on device",2);
  auto host_params = Kokkos::create_mirror_view(model_params);
  host_params(0) = T_ambient;
  host_params(1) = beta;
  Kokkos::deep_copy(model_params,host_params);
  
  have_energy = false;
  
}

// ========================================================================================
// ========================================================================================

void navierstokes::defineFunctions(Teuchos::ParameterList & fs,
                                   Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("source ux",fs.get<string>("source ux","0.0"),"ip");
  functionManager->addFunction("source pr",fs.get<string>("source pr","0.0"),"ip");
  functionManager->addFunction("source uy",fs.get<string>("source uy","0.0"),"ip");
  functionManager->addFunction("source uz",fs.get<string>("source uz","0.0"),"ip");
  functionManager->addFunction("density",fs.get<string>("density","1.0"),"ip");
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","1.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void navierstokes::volumeResidual() {
  
  FDATA dens, visc, source_ux, source_pr, source_uy, source_uz;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source_ux = functionManager->evaluate("source ux","ip");
    source_pr = functionManager->evaluate("source pr","ip");
    if (spaceDim > 1) {
      source_uy = functionManager->evaluate("source uy","ip");
    }
    if (spaceDim > 2) {
      source_uz = functionManager->evaluate("source uz","ip");
    }
    dens = functionManager->evaluate("density","ip");
    visc = functionManager->evaluate("viscosity","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res =wkset->res;
  
  if (spaceDim == 1) {
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto Ux = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
      auto Ux_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
      auto gradUx = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),ux_num,Kokkos::ALL(),Kokkos::ALL());
      auto Pr = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),pr_num,Kokkos::ALL(),0);
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      
      // Ux equation
      parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUx(elem,pt,0) - Pr(elem,pt);
          Fx *= wts(elem,pt);
          AD F = Ux_dot(elem,pt) + Ux(elem,pt)*gradUx(elem,pt,0) - source_ux(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + F*basis(elem,dof,pt);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto gradPr = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),pr_num,Kokkos::ALL(),Kokkos::ALL());
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*Ux_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUx(elem,pt,0)) + gradPr(elem,pt,0) - dens(elem,pt)*source_ux(elem,pt);
            AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_ux(elem,pt);
              AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
              }
            }
          });
        }
      }
    }
    
    {
      /////////////////////////////
      // pressure equation
      /////////////////////////////
      
      int pr_basis = wkset->usebasis[pr_num];
      auto basis = wkset->basis[pr_basis];
      auto basis_grad = wkset->basis_grad[pr_basis];
      auto gradUx = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),ux_num,Kokkos::ALL(),Kokkos::ALL());
      auto off = Kokkos::subview(wkset->offsets,pr_num,Kokkos::ALL());
      
      parallel_for("NS pr volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = gradUx(elem,pt,0)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt);
          }
        }
      });
          
      if (usePSPG) {
        
        auto h = wkset->h;
        auto gradPr = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),pr_num,Kokkos::ALL(),Kokkos::ALL());
        auto Ux = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
        auto Ux_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
        
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*Ux_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUx(elem,pt,0)) + gradPr(elem,pt,0) - dens(elem,pt)*source_ux(elem,pt);
            AD Sx = tau*stabres*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
            }
          }
        });
        if (have_energy) {
          auto params = model_params;
          auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt);
              AD Sx = tau*stabres*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
              }
            }
          });
          //stabres += dens(e,k)*(eval-T_ambient)*source_ux(e,k);
        }
      }
    }
  }
  else if (spaceDim == 2) {
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto Ux = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
      auto Uy = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
      auto Ux_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
      auto gradUx = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),ux_num,Kokkos::ALL(),Kokkos::ALL());
      auto Pr = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),pr_num,Kokkos::ALL(),0);
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      
      // Ux equation
      parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUx(elem,pt,0) - Pr(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUx(elem,pt,1);
          Fy *= wts(elem,pt);
          AD F = Ux_dot(elem,pt) + Ux(elem,pt)*gradUx(elem,pt,0) + Uy(elem,pt)*gradUx(elem,pt,1) - source_ux(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto gradPr = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),pr_num,Kokkos::ALL(),Kokkos::ALL());
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*Ux_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUx(elem,pt,0) + Uy(elem,pt)*gradUx(elem,pt,1)) + gradPr(elem,pt,0) - dens(elem,pt)*source_ux(elem,pt);
            AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_ux(elem,pt);
              AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
              }
            }
          });
        }
      }
    }
    
    {
      // Uy equation
      int uy_basis = wkset->usebasis[uy_num];
      auto basis = wkset->basis[uy_basis];
      auto basis_grad = wkset->basis_grad[uy_basis];
      auto Ux = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
      auto Uy = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
      auto Uy_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
      auto gradUy = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),uy_num,Kokkos::ALL(),Kokkos::ALL());
      auto Pr = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),pr_num,Kokkos::ALL(),0);
      auto off = Kokkos::subview(wkset->offsets,uy_num,Kokkos::ALL());
      
      parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUy(elem,pt,0);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUy(elem,pt,1) - Pr(elem,pt);
          Fy *= wts(elem,pt);
          AD F = Uy_dot(elem,pt) + Ux(elem,pt)*gradUy(elem,pt,0) + Uy(elem,pt)*gradUy(elem,pt,1) - source_uy(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
        parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto gradPr = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),pr_num,Kokkos::ALL(),Kokkos::ALL());
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*Uy_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUy(elem,pt,0) + Uy(elem,pt)*gradUy(elem,pt,1)) + gradPr(elem,pt,1) - dens(elem,pt)*source_uy(elem,pt);
            AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_uy(elem,pt);
              AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
              }
            }
          });
        }
      }
    }
    
    {
      /////////////////////////////
      // pressure equation
      /////////////////////////////
      
      int pr_basis = wkset->usebasis[pr_num];
      auto basis = wkset->basis[pr_basis];
      auto basis_grad = wkset->basis_grad[pr_basis];
      auto gradUx = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),ux_num,Kokkos::ALL(),Kokkos::ALL());
      auto gradUy = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),uy_num,Kokkos::ALL(),Kokkos::ALL());
      auto off = Kokkos::subview(wkset->offsets,pr_num,Kokkos::ALL());
      
      parallel_for("NS pr volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = (gradUx(elem,pt,0) + gradUy(elem,pt,1))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt);
          }
        }
      });
      
      if (usePSPG) {
        
        auto h = wkset->h;
        auto gradPr = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),pr_num,Kokkos::ALL(),Kokkos::ALL());
        auto Ux = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
        auto Uy = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
        auto Ux_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
        auto Uy_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
        
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12+Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD Sx = dens(elem,pt)*Ux_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUx(elem,pt,0) + Uy(elem,pt)*gradUx(elem,pt,1)) + gradPr(elem,pt,0) - dens(elem,pt)*source_ux(elem,pt);
            Sx *= tau*wts(elem,pt);
            AD Sy = dens(elem,pt)*Uy_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUy(elem,pt,0) + Uy(elem,pt)*gradUy(elem,pt,1)) + gradPr(elem,pt,1) - dens(elem,pt)*source_uy(elem,pt);
            Sy *= tau*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
        if (have_energy) {
          auto params = model_params;
          auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12+Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD Sx = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt);
              Sx *= tau*wts(elem,pt);
              AD Sy = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt);
              Sy *= tau*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);;
              }
            }
          });
          //stabres += dens(e,k)*(eval-T_ambient)*source_ux(e,k);
        }
      }
    }
  }
  else if (spaceDim == 3) {
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto Ux = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
      auto Uy = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
      auto Uz = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uz_num,Kokkos::ALL(),0);
      auto Ux_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
      auto gradUx = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),ux_num,Kokkos::ALL(),Kokkos::ALL());
      auto Pr = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),pr_num,Kokkos::ALL(),0);
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      
      // Ux equation
      parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUx(elem,pt,0) - Pr(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUx(elem,pt,1);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*gradUx(elem,pt,2);
          Fz *= wts(elem,pt);
          AD F = Ux_dot(elem,pt) + Ux(elem,pt)*gradUx(elem,pt,0) + Uy(elem,pt)*gradUx(elem,pt,1) + Uz(elem,pt)*gradUx(elem,pt,2) - source_ux(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto gradPr = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),pr_num,Kokkos::ALL(),Kokkos::ALL());
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt) + Uz(elem,pt)*Uz(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*Ux_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUx(elem,pt,0) + Uy(elem,pt)*gradUx(elem,pt,1) + Uz(elem,pt)*gradUx(elem,pt,2)) + gradPr(elem,pt,0) - dens(elem,pt)*source_ux(elem,pt);
            AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
            AD Sz = tau*stabres*Uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt) + Uz(elem,pt)*Uz(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_ux(elem,pt);
              AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
              AD Sz = tau*stabres*Uz(elem,pt)*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
              }
            }
          });
        }
      }
    }
    
    {
      // Uy equation
      int uy_basis = wkset->usebasis[uy_num];
      auto basis = wkset->basis[uy_basis];
      auto basis_grad = wkset->basis_grad[uy_basis];
      auto Ux = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
      auto Uy = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
      auto Uz = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uz_num,Kokkos::ALL(),0);
      auto Uy_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
      auto gradUy = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),uy_num,Kokkos::ALL(),Kokkos::ALL());
      auto Pr = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),pr_num,Kokkos::ALL(),0);
      auto off = Kokkos::subview(wkset->offsets,uy_num,Kokkos::ALL());
      
      parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUy(elem,pt,0);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUy(elem,pt,1) - Pr(elem,pt);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*gradUy(elem,pt,2);
          Fz *= wts(elem,pt);
          AD F = Uy_dot(elem,pt) + Ux(elem,pt)*gradUy(elem,pt,0) + Uy(elem,pt)*gradUy(elem,pt,1) + Uz(elem,pt)*gradUy(elem,pt,2) - source_uy(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
        parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto gradPr = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),pr_num,Kokkos::ALL(),Kokkos::ALL());
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt) + Uz(elem,pt)*Uz(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*Uy_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUy(elem,pt,0) + Uy(elem,pt)*gradUy(elem,pt,1) + Uz(elem,pt)*gradUy(elem,pt,2)) + gradPr(elem,pt,1) - dens(elem,pt)*source_uy(elem,pt);
            AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
            AD Sz = tau*stabres*Uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt) + Uz(elem,pt)*Uz(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_uy(elem,pt);
              AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
              AD Sz = tau*stabres*Uz(elem,pt)*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
              }
            }
          });
        }
      }
    }
    
    {
      // Uz equation
      int uz_basis = wkset->usebasis[uz_num];
      auto basis = wkset->basis[uz_basis];
      auto basis_grad = wkset->basis_grad[uz_basis];
      auto Ux = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
      auto Uy = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
      auto Uz = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uz_num,Kokkos::ALL(),0);
      auto Uz_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),uz_num,Kokkos::ALL(),0);
      auto gradUz = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),uz_num,Kokkos::ALL(),Kokkos::ALL());
      auto Pr = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),pr_num,Kokkos::ALL(),0);
      auto off = Kokkos::subview(wkset->offsets,uy_num,Kokkos::ALL());
      
      parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUz(elem,pt,0);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUz(elem,pt,1);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*gradUz(elem,pt,2) - Pr(elem,pt);
          Fz *= wts(elem,pt);
          AD F = Uz_dot(elem,pt) + Ux(elem,pt)*gradUz(elem,pt,0) + Uy(elem,pt)*gradUz(elem,pt,1) + Uz(elem,pt)*gradUz(elem,pt,2) - source_uz(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
        parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto gradPr = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),pr_num,Kokkos::ALL(),Kokkos::ALL());
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt) + Uz(elem,pt)*Uz(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*Uz_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUz(elem,pt,0) + Uy(elem,pt)*gradUz(elem,pt,1) + Uz(elem,pt)*gradUz(elem,pt,2)) + gradPr(elem,pt,2) - dens(elem,pt)*source_uz(elem,pt);
            AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
            AD Sz = tau*stabres*Uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt) + Uz(elem,pt)*Uz(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_uz(elem,pt);
              AD Sx = tau*stabres*Ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*Uy(elem,pt)*wts(elem,pt);
              AD Sz = tau*stabres*Uz(elem,pt)*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
              }
            }
          });
        }
      }
    }
    
    {
      /////////////////////////////
      // pressure equation
      /////////////////////////////
      
      int pr_basis = wkset->usebasis[pr_num];
      auto basis = wkset->basis[pr_basis];
      auto basis_grad = wkset->basis_grad[pr_basis];
      auto gradUx = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),ux_num,Kokkos::ALL(),Kokkos::ALL());
      auto gradUy = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),uy_num,Kokkos::ALL(),Kokkos::ALL());
      auto gradUz = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),uz_num,Kokkos::ALL(),Kokkos::ALL());
      auto off = Kokkos::subview(wkset->offsets,pr_num,Kokkos::ALL());
      
      parallel_for("NS pr volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = (gradUx(elem,pt,0) + gradUy(elem,pt,1) + gradUz(elem,pt,2))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt);
          }
        }
      });
      
      if (usePSPG) {
        
        auto h = wkset->h;
        auto gradPr = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),pr_num,Kokkos::ALL(),Kokkos::ALL());
        auto Ux = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
        auto Uy = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
        auto Uz = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),uz_num,Kokkos::ALL(),0);
        auto Ux_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),ux_num,Kokkos::ALL(),0);
        auto Uy_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),uy_num,Kokkos::ALL(),0);
        auto Uz_dot = Kokkos::subview(wkset->local_soln_dot,Kokkos::ALL(),uz_num,Kokkos::ALL(),0);
        
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12+Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt) + Uz(elem,pt)*Uz(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD Sx = dens(elem,pt)*Ux_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUx(elem,pt,0) + Uy(elem,pt)*gradUx(elem,pt,1) + Uz(elem,pt)*gradUx(elem,pt,2)) + gradPr(elem,pt,0) - dens(elem,pt)*source_ux(elem,pt);
            Sx *= tau*wts(elem,pt);
            AD Sy = dens(elem,pt)*Uy_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUy(elem,pt,0) + Uy(elem,pt)*gradUy(elem,pt,1) + Uz(elem,pt)*gradUy(elem,pt,2)) + gradPr(elem,pt,1) - dens(elem,pt)*source_uy(elem,pt);
            Sy *= tau*wts(elem,pt);
            AD Sz = dens(elem,pt)*Uz_dot(elem,pt) + dens(elem,pt)*(Ux(elem,pt)*gradUz(elem,pt,0) + Uy(elem,pt)*gradUz(elem,pt,1) + Uz(elem,pt)*gradUz(elem,pt,2)) + gradPr(elem,pt,2) - dens(elem,pt)*source_uz(elem,pt);
            Sz *= tau*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        if (have_energy) {
          auto params = model_params;
          auto E = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),e_num,Kokkos::ALL(),0);
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12+Ux(elem,pt)*Ux(elem,pt) + Uy(elem,pt)*Uy(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD Sx = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt);
              Sx *= tau*wts(elem,pt);
              AD Sy = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt);
              Sy *= tau*wts(elem,pt);
              AD Sz = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uz(elem,pt);
              Sz *= tau*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
              }
            }
          });
          //stabres += dens(e,k)*(eval-T_ambient)*source_ux(e,k);
        }
      }
    }
  }
  /*
            dvdx = basis_grad(e,i,k,0);
            
            AD tau = this->computeTau(visc(e,k), ux, uy, uz, wkset->h(e));
            
            AD stabres = dens(e,k)*ux_dot + dens(e,k)*(ux*duxdx + uy*duxdy + uz*duxdz) + dprdx - dens(e,k)*source_ux(e,k);
            
            if (have_energy) {
              stabres += dens(e,k)*(eval-T_ambient)*source_ux(e,k);
            }
            
            res(e,resindex) += (tau*(stabres)*dvdx)*wts(e,k);
            
            if (spaceDim > 1) {
              dvdy = basis_grad(e,i,k,1);
              AD dprdy = sol_grad(e,pr_num,k,1);
              AD uy_dot = sol_dot(e,uy_num,k,0);
              AD duydx = sol_grad(e,uy_num,k,0);
              AD duydy = sol_grad(e,uy_num,k,1);
              AD duydz = sol_grad(e,uy_num,k,2);
              stabres = dens(e,k)*uy_dot + dens(e,k)*(ux*duydx + uy*duydy + uz*duydz) + dprdy - dens(e,k)*source_uy(e,k);
              if (have_energy) {
                stabres += dens(e,k)*(eval-T_ambient)*source_uy(e,k);
              }
              res(e,resindex) += (tau*(stabres)*dvdy)*wts(e,k);
            }
            
            if (spaceDim > 2) {
              dvdz = basis_grad(e,i,k,2);
              AD dprdz = sol_grad(e,pr_num,k,2);
              AD uz_dot = sol_dot(e,uz_num,k,0);
              AD duzdx = sol_grad(e,uz_num,k,0);
              AD duzdy = sol_grad(e,uz_num,k,1);
              AD duzdz = sol_grad(e,uz_num,k,2);
              stabres = dens(e,k)*uz_dot + dens(e,k)*(ux*duzdx + uy*duzdy + uz*duzdz) + dprdz - dens(e,k)*source_uz(e,k);
              if (have_energy) {
                stabres += dens(e,k)*(eval-T_ambient)*source_uz(e,k);
              }
              res(e,resindex) += (tau*(stabres)*dvdz)*wts(e,k);
              
            }
          }
        }
      }
    });
    }
  }*/
  
  /////////////////////////////
  // uy equation
  /////////////////////////////
  /*
  if (spaceDim > 1) {
    
    int uy_basis = wkset->usebasis[uy_num];
    basis = wkset->basis[uy_basis];
    basis_grad = wkset->basis_grad[uy_basis];
    
    parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
      
      ScalarT v = 0.0;
      ScalarT dvdx = 0.0;
      ScalarT dvdy = 0.0;
      ScalarT dvdz = 0.0;
      
      for( int k=0; k<sol.extent(2); k++ ) {
        
        AD ux = sol(e,ux_num,k,0);
        AD uy_dot = sol_dot(e,uy_num,k,0);
        AD duydx = sol_grad(e,uy_num,k,0);
        
        AD pr = sol(e,pr_num,k,0);
        AD dprdy = sol_grad(e,pr_num,k,1);
        
        AD uy = sol(e,uy_num,k,0);
        AD duydy = sol_grad(e,uy_num,k,1);
        
        AD uz, duydz, eval;
        if (spaceDim > 2) {
          uz = sol(e,uz_num,k,0);
          duydz = sol_grad(e,uy_num,k,2);
        }
        
        if (have_energy) {
          eval = sol(e,e_num,k,0);
        }
        
        for( int i=0; i<basis.extent(1); i++ ) {
          int resindex = offsets(uy_num,i);
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          if (spaceDim > 1) {
            dvdy = basis_grad(e,i,k,1);
          }
          if (spaceDim > 2) {
            dvdz = basis_grad(e,i,k,2);
          }
          
          res(e,resindex) += (dens(e,k)*uy_dot*v + visc(e,k)*(duydx*dvdx + duydy*dvdy + duydz*dvdz) + dens(e,k)*(ux*duydx + uy*duydy + uz*duydz)*v - pr*dvdy - dens(e,k)*source_uy(e,k)*v)*wts(e,k);
          
          if (have_energy) {
            res(e,resindex) += (dens(e,k)*beta*(eval-T_ambient)*source_uy(e,k)*v)*wts(e,k);
          }
          
          if(useSUPG) {
            AD tau = this->computeTau(visc(e,k), ux, uy, uz, wkset->h(e));
            
            AD stabres = dens(e,k)*uy_dot + dens(e,k)*(ux*duydx + uy*duydy + uz*duydz) + dprdy - dens(e,k)*source_uy(e,k);
            
            if (have_energy) {
              stabres += dens(e,k)*beta*(eval-T_ambient)*source_uy(e,k);
            }
            
            res(e,resindex) += (tau*(stabres)*(ux*dvdx + uy*dvdy + uz*dvdz))*wts(e,k);
            
          }
        }
      }
    });
  }*/
  
  /////////////////////////////
  // uz equation
  /////////////////////////////
  /*
  if (spaceDim > 2) {
    int uz_basis = wkset->usebasis[uz_num];
    basis = wkset->basis[uz_basis];
    basis_grad = wkset->basis_grad[uz_basis];
    
    parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
      
      ScalarT v = 0.0;
      ScalarT dvdx = 0.0;
      ScalarT dvdy = 0.0;
      ScalarT dvdz = 0.0;
      
      for( int k=0; k<sol.extent(2); k++ ) {
        
        AD ux = sol(e,ux_num,k,0);
        AD uz_dot = sol_dot(e,uz_num,k,0);
        AD duzdx = sol_grad(e,uz_num,k,0);
        
        AD pr = sol(e,pr_num,k,0);
        AD dprdz = sol_grad(e,pr_num,k,2);
        AD uy = sol(e,uy_num,k,0);
        AD duzdy = sol_grad(e,uz_num,k,1);
        AD uz = sol(e,uz_num,k,0);
        AD duzdz = sol_grad(e,uz_num,k,2);
        
        AD eval;
        if (have_energy) {
          eval = sol(e,e_num,k,0);
        }
        
        for( int i=0; i<basis.extent(1); i++ ) {
          
          int resindex = offsets(uz_num,i);
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          dvdy = basis_grad(e,i,k,1);
          dvdz = basis_grad(e,i,k,2);
          
          res(e,resindex) += (dens(e,k)*uz_dot*v + visc(e,k)*(duzdx*dvdx + duzdy*dvdy + duzdz*dvdz) + dens(e,k)*(ux*duzdx + uy*duzdy + uz*duzdz)*v - pr*dvdz - dens(e,k)*source_uz(e,k)*v)*wts(e,k);
          
          if (have_energy) {
            res(e,resindex) += (dens(e,k)*(eval-T_ambient)*source_uz(e,k)*v)*wts(e,k);
          }
          
          if(useSUPG) {
            AD tau = this->computeTau(visc(e,k), ux, uy, uz, wkset->h(e));
            
            AD stabres = dens(e,k)*uz_dot + dens(e,k)*(ux*duzdx + uy*duzdy + uz*duzdz) + dprdz - dens(e,k)*source_uz(e,k);
            
            if (have_energy) {
              stabres += dens(e,k)*(e-T_ambient)*source_uz(e,k);
            }
            
            res(e,resindex) += (tau*(stabres)*(ux*dvdx + uy*dvdy + uz*dvdz))*wts(e,k);
            
          }
        }
      }
    });
  }*/
  
}

// ========================================================================================
// ========================================================================================

void navierstokes::boundaryResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void navierstokes::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void navierstokes::setVars(std::vector<string> & varlist) {
  e_num = -1;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "ux")
      ux_num = i;
    if (varlist[i] == "pr")
      pr_num = i;
    if (varlist[i] == "uy")
      uy_num = i;
    if (varlist[i] == "uz")
      uz_num = i;
    if (varlist[i] == "e")
      e_num = i;
  }
  if (e_num >= 0)
    have_energy = true;
}


// ========================================================================================
// return the value of the stabilization parameter
// ========================================================================================

AD navierstokes::computeTau(const AD & localdiff, const AD & xvl, const AD & yvl, const AD & zvl, const ScalarT & h) const {
  
  ScalarT C1 = 4.0;
  ScalarT C2 = 2.0;
  
  AD nvel = 0.0;
  if (spaceDim == 1)
    nvel = xvl*xvl;
  else if (spaceDim == 2)
    nvel = xvl*xvl + yvl*yvl;
  else if (spaceDim == 3)
    nvel = xvl*xvl + yvl*yvl + zvl*zvl;
  
  if (nvel > 1E-12)
    nvel = sqrt(nvel);
  
  AD tau;
  tau = 1/(C1*localdiff/h/h + C2*(nvel)/h);
  return tau;
}

