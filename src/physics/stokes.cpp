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

#include "stokes.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

stokes::stokes(Teuchos::ParameterList & settings, const int & dimension_)
  : physicsbase(settings, dimension_)
{
  
  label = "stokes";
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
  
  
  useLSIC = settings.get<bool>("useLSIC",false);
  usePSPG = settings.get<bool>("usePSPG",false);
  
}

// ========================================================================================
// ========================================================================================

void stokes::defineFunctions(Teuchos::ParameterList & fs,
                             Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("source ux",fs.get<string>("source ux","0.0"),"ip");
  functionManager->addFunction("source pr",fs.get<string>("source pr","0.0"),"ip");
  functionManager->addFunction("source uy",fs.get<string>("source uy","0.0"),"ip");
  functionManager->addFunction("source uz",fs.get<string>("source uz","0.0"),"ip");
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","1.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void stokes::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  Vista visc, source_ux, source_pr, source_uy, source_uz;
  
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
    visc = functionManager->evaluate("viscosity","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  /////////////////////////////
  // ux equation
  /////////////////////////////
  
  if (spaceDim == 1) {
    auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
    auto Pr = wkset->getSolutionField("pr");
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto off = subview(wkset->offsets,ux_num,ALL());
      parallel_for("Stokes ux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*dux_dx(elem,pt) - Pr(elem,pt);
          Fx *= wts(elem,pt);
          AD g = -source_ux(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + g*basis(elem,dof,pt,0);
          }
        }
      });
    }
    
    {
      int pr_basis = wkset->usebasis[pr_num];
      auto basis = wkset->basis[pr_basis];
      auto basis_grad = wkset->basis_grad[pr_basis];
      auto off = subview(wkset->offsets,pr_num,ALL());
      
      parallel_for("Stokes pr volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = dux_dx(elem,pt);
          divu *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });

      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        
        parallel_for("Stokes pr volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          ScalarT alpha = 1.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD tau = alpha*h(elem)*h(elem)/(2.*visc(elem,pt));
            AD stabres = dpr_dx(elem,pt) + source_ux(elem,pt);
            AD Sx = tau*stabres*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
            }
          }
        });
      }

      if (useLSIC) {
        
        auto h = wkset->h;
        auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
        
        parallel_for("Stokes pr volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          ScalarT alpha = 1.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD tau = alpha*h(elem)*h(elem)/(2.*visc(elem,pt));
            AD stabres = dux_dx(elem,pt);
            AD S = tau*stabres*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += S*basis_grad(elem,dof,pt,0);
            }
          }
        });
      }
    }
  }
  
  if (spaceDim == 2) {
    auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
    auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
    auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
    auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
    auto Pr = wkset->getSolutionField("pr");
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      parallel_for("Stokes ux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*dux_dx(elem,pt) - Pr(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*dux_dy(elem,pt);
          Fy *= wts(elem,pt);
          AD g = -source_ux(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + g*basis(elem,dof,pt,0);
          }
        }
      });
    }
    
    {
      int uy_basis = wkset->usebasis[uy_num];
      auto basis = wkset->basis[uy_basis];
      auto basis_grad = wkset->basis_grad[uy_basis];
      auto off = subview(wkset->offsets,uy_num,ALL());
      parallel_for("Stokes uy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*duy_dx(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*duy_dy(elem,pt) - Pr(elem,pt);
          Fy *= wts(elem,pt);
          AD g = -source_uy(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + g*basis(elem,dof,pt,0);
          }
        }
      });
    }
    
    {
      int pr_basis = wkset->usebasis[pr_num];
      auto basis = wkset->basis[pr_basis];
      auto basis_grad = wkset->basis_grad[pr_basis];
      auto off = Kokkos::subview(wkset->offsets,pr_num,Kokkos::ALL());
      
      parallel_for("Stokes pr volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = dux_dx(elem,pt) + duy_dy(elem,pt);
          divu *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });

      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        
        parallel_for("Stokes pr volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          ScalarT alpha = 1.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            //AD tau = alpha*h(elem)*h(elem)/(2.*visc(elem,pt));
            AD tau = alpha*h(elem)/(2.*visc(elem,pt));
            AD Sx = dpr_dx(elem,pt) + source_ux(elem,pt);
            Sx *= tau*wts(elem,pt);
            AD Sy = dpr_dy(elem,pt) + source_uy(elem,pt);
            Sy *= tau*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
      }

      if (useLSIC) {
        
        auto h = wkset->h;
        auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
        auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
        
        parallel_for("Stokes pr volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          ScalarT alpha = 1.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD tau = alpha*h(elem)*h(elem)/(2.*visc(elem,pt));
            AD stabres = dux_dx(elem,pt) + duy_dy(elem,pt);
            AD S = tau*stabres*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += S*(basis_grad(elem,dof,pt,0) + basis_grad(elem,dof,pt,1));
            }
          }
        });
      }
    }
  }
  
  if (spaceDim == 3) {
    auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
    auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
    auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
    auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
    auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
    auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
    auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
    auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
    auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
    auto Pr = wkset->getSolutionField("pr");
    
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto off = subview(wkset->offsets,ux_num,ALL());
      parallel_for("Stokes ux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*dux_dx(elem,pt) - Pr(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*dux_dy(elem,pt);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*dux_dz(elem,pt);
          Fz *= wts(elem,pt);
          AD g = -source_ux(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + g*basis(elem,dof,pt,0);
          }
        }
      });
    }
    
    {
      int uy_basis = wkset->usebasis[uy_num];
      auto basis = wkset->basis[uy_basis];
      auto basis_grad = wkset->basis_grad[uy_basis];
      auto off = subview(wkset->offsets,uy_num,ALL());
      parallel_for("Stokes uy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*duy_dx(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*duy_dy(elem,pt) - Pr(elem,pt);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*duy_dz(elem,pt);
          Fz *= wts(elem,pt);
          AD g = -source_uy(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + g*basis(elem,dof,pt,0);
          }
        }
      });
    }
    
    {
      int uz_basis = wkset->usebasis[uz_num];
      auto basis = wkset->basis[uz_basis];
      auto basis_grad = wkset->basis_grad[uz_basis];
      auto off = subview(wkset->offsets,uz_num,ALL());
      parallel_for("Stokes uy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*duz_dx(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*duz_dy(elem,pt);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*duz_dz(elem,pt) - Pr(elem,pt);
          Fz *= wts(elem,pt);
          AD g = -source_uz(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + g*basis(elem,dof,pt,0);
          }
        }
      });
    }

    {
      int pr_basis = wkset->usebasis[pr_num];
      auto basis = wkset->basis[pr_basis];
      auto basis_grad = wkset->basis_grad[pr_basis];
      auto off = subview(wkset->offsets,pr_num,ALL());
      
      parallel_for("Stokes pr volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt);
          divu *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });

      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        auto dpr_dz = wkset->getSolutionField("grad(pr)[z]");
        
        parallel_for("Stokes pr volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
	  ScalarT alpha = 1.0;
	  for (size_type pt=0; pt<basis.extent(2); pt++ ) {
	    AD tau = alpha*h(elem)*h(elem)/(2.*visc(elem,pt));
	    AD Sx = dpr_dx(elem,pt) + source_ux(elem,pt);
	    Sx *= tau*wts(elem,pt);
	    AD Sy = dpr_dy(elem,pt) + source_uy(elem,pt);
	    Sy *= tau*wts(elem,pt);
	    AD Sz = dpr_dz(elem,pt) + source_uz(elem,pt);
	    Sz *= tau*wts(elem,pt);
	    for( size_type dof=0; dof<basis.extent(1); dof++ ) {
	      res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
	    }
	  }
	});
      }

      if (useLSIC) {
        
        auto h = wkset->h;
        auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
        auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
        auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
        
        parallel_for("Stokes pr volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
        ScalarT alpha = 1.0;
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
	    AD tau = alpha*h(elem)*h(elem)/(2.*visc(elem,pt));
	    AD stabres = dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt);
	    AD S = tau*stabres*wts(elem,pt);
	    for( size_type dof=0; dof<basis.extent(1); dof++ ) {
	      res(elem,off(dof)) += S*(basis_grad(elem,dof,pt,0) + basis_grad(elem,dof,pt,1) + basis_grad(elem,dof,pt,2));
	    }
	  }
	});
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

void stokes::boundaryResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void stokes::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void stokes::setWorkset(Teuchos::RCP<Workset> & wkset_) {

  wkset = wkset_;
  vector<string> varlist = wkset->varlist;

  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "ux")
      ux_num = i;
    if (varlist[i] == "pr")
      pr_num = i;
    if (varlist[i] == "uy")
      uy_num = i;
    if (varlist[i] == "uz")
      uz_num = i;
  }
  
}
