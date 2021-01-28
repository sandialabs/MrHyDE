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

navierstokes::navierstokes(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_)
  : physicsbase(settings, isaux_)
{
  
  label = "navierstokes";
  int spaceDim = settings->sublist("Mesh").get<int>("dimension",2);
  
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
  
  int spaceDim = wkset->dimension;
  View_AD2 dens, visc, source_ux, source_pr, source_uy, source_uz;
  
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
      auto ux = wkset->getData("ux");
      auto dux_dt = wkset->getData("ux_t");
      auto dux_dx = wkset->getData("grad(ux)[x]");
      auto pr = wkset->getData("pr");
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      
      // Ux equation
      parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
          Fx *= wts(elem,pt);
          AD F = dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) - source_ux(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getData("e");
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dx = wkset->getData("grad(pr)[x]");
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getData("e");
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_ux(elem,pt);
              AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
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
      auto dux_dx = wkset->getData("grad(ux)[x]");
      auto off = Kokkos::subview(wkset->offsets,pr_num,Kokkos::ALL());
      
      parallel_for("NS pr volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = dux_dx(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });
          
      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getData("grad(pr)[x]");
        auto ux = wkset->getData("ux");
        auto dux_dt = wkset->getData("ux_t");
        
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            AD Sx = tau*stabres*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
            }
          }
        });
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getData("e");
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt));
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
      auto ux = wkset->getData("ux");
      auto uy = wkset->getData("uy");
      auto dux_dt = wkset->getData("ux_t");
      auto dux_dx = wkset->getData("grad(ux)[x]");
      auto dux_dy = wkset->getData("grad(ux)[y]");
      auto pr = wkset->getData("pr");
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      
      // Ux equation
      parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*dux_dy(elem,pt);
          Fy *= wts(elem,pt);
          AD F = dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) - source_ux(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getData("e");
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dx = wkset->getData("grad(pr)[x]");
        auto dpr_dy = wkset->getData("grad(pr)[y]");
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getData("e");
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_ux(elem,pt);
              AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
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
      auto ux = wkset->getData("ux");
      auto uy = wkset->getData("uy");
      auto duy_dt = wkset->getData("uy_t");
      auto duy_dx = wkset->getData("grad(uy)[x]");
      auto duy_dy = wkset->getData("grad(uy)[y]");
      auto pr = wkset->getData("pr");
      auto off = Kokkos::subview(wkset->offsets,uy_num,Kokkos::ALL());
      
      parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*duy_dx(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*duy_dy(elem,pt) - pr(elem,pt);
          Fy *= wts(elem,pt);
          AD F = duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) - source_uy(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getData("e");
        parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dy = wkset->getData("grad(pr)[y]");
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*duy_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) + dpr_dy(elem,pt) - dens(elem,pt)*source_uy(elem,pt);
            AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getData("e");
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_uy(elem,pt);
              AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
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
      auto dux_dx = wkset->getData("grad(ux)[x]");
      auto duy_dy = wkset->getData("grad(uy)[y]");
      auto off = Kokkos::subview(wkset->offsets,pr_num,Kokkos::ALL());
      
      parallel_for("NS pr volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = (dux_dx(elem,pt) + duy_dy(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });
      
      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getData("grad(pr)[x]");
        auto dpr_dy = wkset->getData("grad(pr)[y]");
        auto ux =wkset->getData("ux");
        auto uy = wkset->getData("uy");
        auto dux_dt = wkset->getData("ux_t");
        auto duy_dt = wkset->getData("uy_t");
        auto dux_dy = wkset->getData("grad(ux)[y]");
        auto duy_dx = wkset->getData("grad(uy)[x]");
        
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12+ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD Sx = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            Sx *= tau*wts(elem,pt);
            AD Sy = dens(elem,pt)*duy_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duy_dy(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) + dpr_dy(elem,pt) - dens(elem,pt)*source_uy(elem,pt);
            Sy *= tau*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getData("e");
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12+ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt));
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
      auto ux = wkset->getData("ux");
      auto uy = wkset->getData("uy");
      auto uz = wkset->getData("uz");
      auto dux_dt = wkset->getData("ux_t");
      auto dux_dx = wkset->getData("grad(ux)[x]");
      auto dux_dy = wkset->getData("grad(ux)[y]");
      auto dux_dz = wkset->getData("grad(ux)[z]");
      auto pr = wkset->getData("pr");
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      
      // Ux equation
      parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*dux_dy(elem,pt);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*dux_dz(elem,pt);
          Fz *= wts(elem,pt);
          AD F = dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt) - source_ux(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getData("e");
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dx = wkset->getData("grad(pr)[x]");
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt) + uz(elem,pt)*uz(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            AD Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getData("e");
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt) + uz(elem,pt)*uz(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_ux(elem,pt);
              AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
              AD Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
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
      auto ux = wkset->getData("ux");
      auto uy = wkset->getData("uy");
      auto uz = wkset->getData("uz");
      auto duy_dt = wkset->getData("uy_t");
      auto duy_dx = wkset->getData("grad(uy)[x]");
      auto duy_dy = wkset->getData("grad(uy)[y]");
      auto duy_dz = wkset->getData("grad(uy)[z]");
      auto pr = wkset->getData("pr");
      auto off = Kokkos::subview(wkset->offsets,uy_num,Kokkos::ALL());
      
      parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*duy_dy(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*duy_dy(elem,pt) - pr(elem,pt);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*duy_dz(elem,pt);
          Fz *= wts(elem,pt);
          AD F = duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt) - source_uy(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getData("e");
        parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dy = wkset->getData("grad(pr)[y]");
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt) + uz(elem,pt)*uz(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*duy_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) + dpr_dy(elem,pt) - dens(elem,pt)*source_uy(elem,pt);
            AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            AD Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getData("e");
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt) + uz(elem,pt)*uz(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_uy(elem,pt);
              AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
              AD Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
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
      auto ux = wkset->getData("ux");
      auto uy = wkset->getData("uy");
      auto uz = wkset->getData("uz");
      auto duz_dt = wkset->getData("uz_t");
      auto duz_dx = wkset->getData("grad(uz)[x]");
      auto duz_dy = wkset->getData("grad(uz)[y]");
      auto duz_dz = wkset->getData("grad(uz)[z]");
      auto pr = wkset->getData("pr");
      auto off = Kokkos::subview(wkset->offsets,uy_num,Kokkos::ALL());
      
      parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*duz_dx(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*duz_dy(elem,pt);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*duz_dz(elem,pt) - pr(elem,pt);
          Fz *= wts(elem,pt);
          AD F = duz_dt(elem,pt) + ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt) - source_uz(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getData("e");
        parallel_for("NS uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dz = wkset->getData("grad(pr)[z]");
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt) + uz(elem,pt)*uz(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD stabres = dens(elem,pt)*duz_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) + dpr_dz(elem,pt) - dens(elem,pt)*source_uz(elem,pt);
            AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            AD Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getData("e");
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12 + ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt) + uz(elem,pt)*uz(elem,pt));
              AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
              AD stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_uz(elem,pt);
              AD Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              AD Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
              AD Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
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
      auto dux_dx = wkset->getData("grad(ux)[x]");
      auto duy_dy = wkset->getData("grad(uy)[y]");
      auto duz_dz = wkset->getData("grad(uz)[z]");
      auto off = Kokkos::subview(wkset->offsets,pr_num,Kokkos::ALL());
      
      parallel_for("NS pr volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = (dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });
      
      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getData("grad(pr)[x]");
        auto dpr_dy = wkset->getData("grad(pr)[y]");
        auto dpr_dz = wkset->getData("grad(pr)[z]");
        auto ux = wkset->getData("ux");
        auto uy = wkset->getData("uy");
        auto uz = wkset->getData("uz");
        auto dux_dt = wkset->getData("ux_t");
        auto duy_dt = wkset->getData("uy_t");
        auto duz_dt = wkset->getData("uz_t");
        auto dux_dy = wkset->getData("grad(ux)[y]");
        auto dux_dz = wkset->getData("grad(ux)[z]");
        auto duy_dx = wkset->getData("grad(uy)[x]");
        auto duy_dz = wkset->getData("grad(uy)[z]");
        auto duz_dx = wkset->getData("grad(uz)[x]");
        auto duz_dy = wkset->getData("grad(uz)[y]");
        
        parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          ScalarT C1 = 4.0;
          ScalarT C2 = 2.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD nvel = sqrt(1.0e-12+ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt) + uz(elem,pt)*uz(elem,pt));
            AD tau = 1.0/(C1*visc(elem,pt)/h(elem)/h(elem) + C2*(nvel)/h(elem));
            AD Sx = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            Sx *= tau*wts(elem,pt);
            AD Sy = dens(elem,pt)*duy_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) + dpr_dy(elem,pt) - dens(elem,pt)*source_uy(elem,pt);
            Sy *= tau*wts(elem,pt);
            AD Sz = dens(elem,pt)*duz_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) + dpr_dz(elem,pt) - dens(elem,pt)*source_uz(elem,pt);
            Sz *= tau*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getData("e");
          parallel_for("NS ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ScalarT C1 = 4.0;
            ScalarT C2 = 2.0;
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD nvel = sqrt(1.0e-12+ux(elem,pt)*ux(elem,pt) + uy(elem,pt)*uy(elem,pt) + uz(elem,pt)*uz(elem,pt));
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
// ========================================================================================
// ========================================================================================

void navierstokes::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;

  vector<string> varlist = wkset->varlist;
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
  
  int spaceDim = wkset->dimension;
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

