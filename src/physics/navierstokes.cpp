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

// TODO BWR -- rho is both on the convective part but we have nu and rho*source showing up too
// this is inconsistent and needs fixing! 

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
navierstokes<EvalT>::navierstokes(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "navierstokes";
  int spaceDim = dimension_;
  
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
  
  
  useSUPG = settings.get<bool>("useSUPG",false);
  usePSPG = settings.get<bool>("usePSPG",false);
  T_ambient = settings.get<ScalarT>("T_ambient",0.0);
  beta = settings.get<ScalarT>("beta",1.0);
  model_params = Kokkos::View<ScalarT*,AssemblyDevice>("NS params on device",2);
  auto host_params = create_mirror_view(model_params);
  host_params(0) = T_ambient;
  host_params(1) = beta;
  deep_copy(model_params,host_params);
  
  have_energy = false;
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void navierstokes<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                   Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
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

template<class EvalT>
void navierstokes<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  ScalarT dt = wkset->deltat;
  bool isTransient = wkset->isTransient;
  Vista<EvalT> dens, visc, source_ux, source_pr, source_uy, source_uz;
  
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
      auto ux = wkset->getSolutionField("ux");
      auto dux_dt = wkset->getSolutionField("ux_t");
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,ux_num,ALL());
      
      // Ux equation
      parallel_for("NS ux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = visc(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
          Fx *= wts(elem,pt);
          EvalT F = dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) - source_ux(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getSolutionField("e");
        parallel_for("NS ux volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        parallel_for("NS ux volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),0.0,0.0,h(elem),spaceDim,dt,isTransient);
            EvalT stabres = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0); 
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getSolutionField("e");
          parallel_for("NS ux volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),0.0,0.0,h(elem),spaceDim,dt,isTransient);
              EvalT stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_ux(elem,pt);
              EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
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
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto off = subview(wkset->offsets,pr_num,ALL());
      
      parallel_for("NS pr volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT divu = dux_dx(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });

      // TODO BWR -- viscous contribution and a divide by rho missing (added in now)?
          
      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto ux = wkset->getSolutionField("ux");
        auto dux_dt = wkset->getSolutionField("ux_t");
        
        parallel_for("NS pr volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            // BWR -- OK, I'm assuming that the viscosity defined here mean kinematic viscosity... which is not how I was interpretting it
            EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),0.0,0.0,h(elem),spaceDim,dt,isTransient);
            EvalT stabres = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            EvalT Sx = tau*stabres*wts(elem,pt)/dens(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0); 
            }
          }
        });
        if (have_energy) {
          // BWR -- TODO did not check this or any of the energy eqns
          auto params = model_params;
          auto E = wkset->getSolutionField("e");
          parallel_for("NS pr volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),0.0,0.0,h(elem),spaceDim,dt,isTransient);
              EvalT stabres = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt);
              EvalT Sx = tau*stabres*wts(elem,pt);
              for( size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
              }
            }
          });
          
        }
      }
    }
  }
  else if (spaceDim == 2) {
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto dux_dt = wkset->getSolutionField("ux_t");
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,ux_num,ALL());
      
      // Ux equation
      parallel_for("NS ux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = visc(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
          Fx *= wts(elem,pt);
          EvalT Fy = visc(elem,pt)*dux_dy(elem,pt);
          Fy *= wts(elem,pt);
          EvalT F = dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) - source_ux(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getSolutionField("e");
        parallel_for("NS ux volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]"); // TODO unnecesary?
        parallel_for("NS ux volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),0.0,h(elem),spaceDim,dt,isTransient);
            EvalT stabres = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getSolutionField("e");
          parallel_for("NS ux volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),0.0,h(elem),spaceDim,dt,isTransient);
              EvalT stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_ux(elem,pt);
              EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
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
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto duy_dt = wkset->getSolutionField("uy_t");
      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,uy_num,ALL());
      
      parallel_for("NS uy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = visc(elem,pt)*duy_dx(elem,pt);
          Fx *= wts(elem,pt);
          EvalT Fy = visc(elem,pt)*duy_dy(elem,pt) - pr(elem,pt);
          Fy *= wts(elem,pt);
          EvalT F = duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) - source_uy(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getSolutionField("e");
        parallel_for("NS uy volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        parallel_for("NS uy volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),0.0,h(elem),spaceDim,dt,isTransient);
            EvalT stabres = dens(elem,pt)*duy_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) + dpr_dy(elem,pt) - dens(elem,pt)*source_uy(elem,pt);
            EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getSolutionField("e");
          parallel_for("NS ux volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),0.0,h(elem),spaceDim,dt,isTransient);
              EvalT stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_uy(elem,pt);
              EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
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
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
      auto off = subview(wkset->offsets,pr_num,ALL());
      
      parallel_for("NS pr volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT divu = (dux_dx(elem,pt) + duy_dy(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });
      
      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        auto ux =wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto dux_dt = wkset->getSolutionField("ux_t");
        auto duy_dt = wkset->getSolutionField("uy_t");
        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        
        parallel_for("NS pr volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),0.0,h(elem),spaceDim,dt,isTransient);
            EvalT Sx = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            Sx *= tau*wts(elem,pt)/dens(elem,pt);
            EvalT Sy = dens(elem,pt)*duy_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) + dpr_dy(elem,pt) - dens(elem,pt)*source_uy(elem,pt);
            Sy *= tau*wts(elem,pt)/dens(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
        if (have_energy) {
          // TODO BWR -- again not messing with this for now
          auto params = model_params;
          auto E = wkset->getSolutionField("e");
          parallel_for("NS pr volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),0.0,h(elem),spaceDim,dt,isTransient);
              EvalT Sx = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt);
              Sx *= tau*wts(elem,pt);
              EvalT Sy = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt);
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
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto uz = wkset->getSolutionField("uz");
      auto dux_dt = wkset->getSolutionField("ux_t");
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
      auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,ux_num,ALL());
      
      // Ux equation
      parallel_for("NS ux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = visc(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
          Fx *= wts(elem,pt);
          EvalT Fy = visc(elem,pt)*dux_dy(elem,pt);
          Fy *= wts(elem,pt);
          EvalT Fz = visc(elem,pt)*dux_dz(elem,pt);
          Fz *= wts(elem,pt);
          EvalT F = dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt) - source_ux(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getSolutionField("e");
        parallel_for("NS ux volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        parallel_for("NS ux volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),h(elem),spaceDim,dt,isTransient);
            EvalT stabres = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            EvalT Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getSolutionField("e");
          parallel_for("NS ux volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),h(elem),spaceDim,dt,isTransient);
              EvalT stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_ux(elem,pt);
              EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
              EvalT Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
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
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto uz = wkset->getSolutionField("uz");
      auto duy_dt = wkset->getSolutionField("uy_t");
      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
      auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,uy_num,ALL());
      
      parallel_for("NS uy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = visc(elem,pt)*duy_dx(elem,pt);
          Fx *= wts(elem,pt);
          EvalT Fy = visc(elem,pt)*duy_dy(elem,pt) - pr(elem,pt);
          Fy *= wts(elem,pt);
          EvalT Fz = visc(elem,pt)*duy_dz(elem,pt);
          Fz *= wts(elem,pt);
          EvalT F = duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt) - source_uy(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getSolutionField("e");
        parallel_for("NS uy volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        parallel_for("NS uy volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),h(elem),spaceDim,dt,isTransient);
            EvalT stabres = dens(elem,pt)*duy_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) + dpr_dy(elem,pt) - dens(elem,pt)*source_uy(elem,pt);
            EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            EvalT Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getSolutionField("e");
          parallel_for("NS uy volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),h(elem),spaceDim,dt,isTransient);
              EvalT stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_uy(elem,pt);
              EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
              EvalT Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
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
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto uz = wkset->getSolutionField("uz");
      auto duz_dt = wkset->getSolutionField("uz_t");
      auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
      auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,uy_num,ALL());
      
      parallel_for("NS uy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = visc(elem,pt)*duz_dx(elem,pt);
          Fx *= wts(elem,pt);
          EvalT Fy = visc(elem,pt)*duz_dy(elem,pt);
          Fy *= wts(elem,pt);
          EvalT Fz = visc(elem,pt)*duz_dz(elem,pt) - pr(elem,pt);
          Fz *= wts(elem,pt);
          EvalT F = duz_dt(elem,pt) + ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt) - source_uz(elem,pt);
          F *= dens(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // Energy contribution
      if (have_energy) {
        auto params = model_params;
        auto E = wkset->getSolutionField("e");
        parallel_for("NS uy volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT F = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += F*basis(elem,dof,pt,0);
            }
          }
        });
      }
      
      // SUPG contribution
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dz = wkset->getSolutionField("grad(pr)[z]");
        parallel_for("NS uz volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),h(elem),spaceDim,dt,isTransient);
            EvalT stabres = dens(elem,pt)*duz_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) + dpr_dz(elem,pt) - dens(elem,pt)*source_uz(elem,pt);
            EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
            EvalT Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        
        if (have_energy) {
          auto params = model_params;
          auto E = wkset->getSolutionField("e");
          parallel_for("NS uz volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),h(elem),spaceDim,dt,isTransient);
              EvalT stabres = dens(elem,pt)*params(1)*(E(elem,pt) - params(0))*source_uz(elem,pt);
              EvalT Sx = tau*stabres*ux(elem,pt)*wts(elem,pt);
              EvalT Sy = tau*stabres*uy(elem,pt)*wts(elem,pt);
              EvalT Sz = tau*stabres*uz(elem,pt)*wts(elem,pt);
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
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
      auto off = subview(wkset->offsets,pr_num,ALL());
      
      parallel_for("NS pr volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT divu = (dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt))*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });
      
      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        auto dpr_dz = wkset->getSolutionField("grad(pr)[z]");
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto uz = wkset->getSolutionField("uz");
        auto dux_dt = wkset->getSolutionField("ux_t");
        auto duy_dt = wkset->getSolutionField("uy_t");
        auto duz_dt = wkset->getSolutionField("uz_t");
        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
        auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
        auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
        auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
        
        parallel_for("NS pr volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),h(elem),spaceDim,dt,isTransient);
            EvalT Sx = dens(elem,pt)*dux_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) + dpr_dx(elem,pt) - dens(elem,pt)*source_ux(elem,pt);
            Sx *= tau*wts(elem,pt)/dens(elem,pt);
            EvalT Sy = dens(elem,pt)*duy_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) + dpr_dy(elem,pt) - dens(elem,pt)*source_uy(elem,pt);
            Sy *= tau*wts(elem,pt)/dens(elem,pt);
            EvalT Sz = dens(elem,pt)*duz_dt(elem,pt) + dens(elem,pt)*(ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) + dpr_dz(elem,pt) - dens(elem,pt)*source_uz(elem,pt);
            Sz *= tau*wts(elem,pt)/dens(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
        if (have_energy) {
          // BWR TODO check and change, see above
          auto params = model_params;
          auto E = wkset->getSolutionField("e");
          parallel_for("NS pr volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              EvalT tau = this->computeTau(visc(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),h(elem),spaceDim,dt,isTransient);
              EvalT Sx = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_ux(elem,pt);
              Sx *= tau*wts(elem,pt);
              EvalT Sy = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uy(elem,pt);
              Sy *= tau*wts(elem,pt);
              EvalT Sz = dens(elem,pt)*params(1)*(E(elem,pt)-params(0))*source_uz(elem,pt);
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

template<class EvalT>
void navierstokes<EvalT>::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  
  string ux_sidetype = bcs(ux_num,cside);
  string uy_sidetype = "Dirichlet";
  string uz_sidetype = "Dirichlet";
  if (spaceDim > 1) {
    uy_sidetype = bcs(uy_num,cside);
  }
  if (spaceDim > 2) {
    uz_sidetype = bcs(uz_num,cside);
  }
  
  Vista<EvalT> source_ux, source_uy, source_uz;
  
  if (ux_sidetype != "Dirichlet" || uy_sidetype != "Dirichlet" || uz_sidetype != "Dirichlet") {
    
    {
      //Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
      if (ux_sidetype == "Neumann") {
        source_ux = functionManager->evaluate("Neumann ux " + wkset->sidename,"side ip");
      }
      if (uy_sidetype == "Neumann") {
        source_uy = functionManager->evaluate("Neumann uy " + wkset->sidename,"side ip");
      }
      if (uz_sidetype == "Neumann") {
        source_uz = functionManager->evaluate("Neumann uz " + wkset->sidename,"side ip");
      }
    }
    
    // Since normals get recomputed often, this needs to be reset
    auto wts = wkset->wts_side;
    auto h = wkset->h;
    auto res = wkset->res;
    
    //Teuchos::TimeMonitor localtime(*boundaryResidualFill);
    
    if (spaceDim == 1) {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis_side[ux_basis];
      auto off = Kokkos::subview( wkset->offsets, ux_num, Kokkos::ALL());
      if (ux_sidetype == "Neumann") { // Neumann
        parallel_for("NS ux bndry resid 1D N",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += (-source_ux(e,k)*basis(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
    }
    else if (spaceDim == 2) {
      
      // ux equation boundary residual
      {
        int ux_basis = wkset->usebasis[ux_num];
        auto basis = wkset->basis_side[ux_basis];
        auto off = Kokkos::subview( wkset->offsets, ux_num, Kokkos::ALL());
        
        if (ux_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("NS ux bndry resid 2D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_ux(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
      }
      
      // uy equation boundary residual
      {
        int uy_basis = wkset->usebasis[uy_num];
        auto basis = wkset->basis_side[uy_basis];
        auto off = Kokkos::subview( wkset->offsets, uy_num, Kokkos::ALL());
        if (uy_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("NS uy bndry resid 2D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_uy(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
      }
    }
    
    else if (spaceDim == 3) {
      
      // ux equation boundary residual
      {
        int ux_basis = wkset->usebasis[ux_num];
        auto basis = wkset->basis_side[ux_basis];
        auto off = Kokkos::subview( wkset->offsets, ux_num, Kokkos::ALL());
        if (ux_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("NS ux bndry resid 3D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_ux(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
      }
      
      // uy equation boundary residual
      {
        int uy_basis = wkset->usebasis[uy_num];
        auto basis = wkset->basis_side[uy_basis];
        auto off = Kokkos::subview( wkset->offsets, uy_num, Kokkos::ALL());
        if (uy_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("NS uy bndry resid 3D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_uy(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
      }
      
      // uz equation boundary residual
      {
        int uz_basis = wkset->usebasis[uz_num];
        auto basis = wkset->basis_side[uz_basis];
        auto off = Kokkos::subview( wkset->offsets, uz_num, Kokkos::ALL());
        if (uz_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("NS uz bndry resid 3D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_uz(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
      }
    }
  }
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void navierstokes<EvalT>::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void navierstokes<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

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

template<class EvalT>
KOKKOS_FUNCTION EvalT navierstokes<EvalT>::computeTau(const EvalT & localdiff, const EvalT & xvl, const EvalT & yvl, const EvalT & zvl, const ScalarT & h, const int & spaceDim, const ScalarT & dt, const bool & isTransient) const {
  
  ScalarT C1 = 4.0;
  ScalarT C2 = 2.0;
  ScalarT C3 = isTransient ? 2.0 : 0.0; // only if transient -- TODO not sure BWR
  
  EvalT nvel = 0.0;
  if (spaceDim == 1)
    nvel = xvl*xvl;
  else if (spaceDim == 2)
    nvel = xvl*xvl + yvl*yvl;
  else if (spaceDim == 3)
    nvel = xvl*xvl + yvl*yvl + zvl*zvl;
  
  if (nvel > 1E-12)
    nvel = sqrt(nvel);
  
  EvalT tau;
  // see, e.g. wikipedia article on SUPG/PSPG 
  // coefficients can be changed/tuned for different scenarios (including order of time scheme)
  // https://arxiv.org/pdf/1710.08898.pdf had a good, clear writeup of the final eqns
  tau = (C1*localdiff/h/h)*(C1*localdiff/h/h) + (C2*nvel/h)*(C2*nvel/h) + (C3/dt)*(C3/dt);
  tau = 1./sqrt(tau);

  return tau;
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::navierstokes<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::navierstokes<AD>;

// Standard built-in types
template class MrHyDE::navierstokes<AD2>;
template class MrHyDE::navierstokes<AD4>;
template class MrHyDE::navierstokes<AD8>;
template class MrHyDE::navierstokes<AD16>;
template class MrHyDE::navierstokes<AD18>;
template class MrHyDE::navierstokes<AD24>;
template class MrHyDE::navierstokes<AD32>;
#endif
