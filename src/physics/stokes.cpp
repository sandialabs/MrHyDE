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

stokes::stokes(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_) {
  
  label = "stokes";
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
  
  
  //useSUPG = settings->sublist("Physics").get<bool>("useSUPG",false);
  //usePSPG = settings->sublist("Physics").get<bool>("usePSPG",false);
  T_ambient = settings->sublist("Physics").get<ScalarT>("T_ambient",0.0);
  beta = settings->sublist("Physics").get<ScalarT>("beta",1.0);
  
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
  
  View_AD2_sv visc, source_ux, source_pr, source_uy, source_uz;
  
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
    auto gradUx = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),ux_num,Kokkos::ALL(),Kokkos::ALL());
    auto Pr = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),pr_num,Kokkos::ALL(),0);
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      parallel_for("Stokes ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUx(elem,pt,0) - Pr(elem,pt);
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
      auto off = Kokkos::subview(wkset->offsets,pr_num,Kokkos::ALL());
      
      parallel_for("Stokes pr volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = gradUx(elem,pt,0);
          divu *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });
    }
  }
  
  if (spaceDim == 2) {
    auto gradUx = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),ux_num,Kokkos::ALL(),Kokkos::ALL());
    auto gradUy = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),uy_num,Kokkos::ALL(),Kokkos::ALL());
    auto Pr = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),pr_num,Kokkos::ALL(),0);
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      parallel_for("Stokes ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUx(elem,pt,0) - Pr(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUx(elem,pt,1);
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
      auto off = Kokkos::subview(wkset->offsets,uy_num,Kokkos::ALL());
      parallel_for("Stokes uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUy(elem,pt,0);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUy(elem,pt,1) - Pr(elem,pt);
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
      
      parallel_for("Stokes pr volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = gradUx(elem,pt,0) + gradUy(elem,pt,1);
          divu *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });
    }
  }
  
  if (spaceDim == 3) {
    auto gradUx = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),ux_num,Kokkos::ALL(),Kokkos::ALL());
    auto gradUy = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),uy_num,Kokkos::ALL(),Kokkos::ALL());
    auto gradUz = Kokkos::subview(wkset->local_soln_grad,Kokkos::ALL(),uz_num,Kokkos::ALL(),Kokkos::ALL());
    auto Pr = Kokkos::subview(wkset->local_soln,Kokkos::ALL(),pr_num,Kokkos::ALL(),0);
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto off = Kokkos::subview(wkset->offsets,ux_num,Kokkos::ALL());
      parallel_for("Stokes ux volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUx(elem,pt,0) - Pr(elem,pt);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUx(elem,pt,1);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*gradUx(elem,pt,2);
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
      auto off = Kokkos::subview(wkset->offsets,uy_num,Kokkos::ALL());
      parallel_for("Stokes uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUy(elem,pt,0);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUy(elem,pt,1) - Pr(elem,pt);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*gradUy(elem,pt,2);
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
      auto off = Kokkos::subview(wkset->offsets,uz_num,Kokkos::ALL());
      parallel_for("Stokes uy volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Fx = visc(elem,pt)*gradUz(elem,pt,0);
          Fx *= wts(elem,pt);
          AD Fy = visc(elem,pt)*gradUz(elem,pt,1);
          Fy *= wts(elem,pt);
          AD Fz = visc(elem,pt)*gradUz(elem,pt,2) - Pr(elem,pt);
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
      auto off = Kokkos::subview(wkset->offsets,pr_num,Kokkos::ALL());
      
      parallel_for("Stokes pr volume resid",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for( size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD divu = gradUx(elem,pt,0) + gradUy(elem,pt,1) + gradUz(elem,pt,2);
          divu *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += divu*basis(elem,dof,pt,0);
          }
        }
      });
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

void stokes::setVars(std::vector<string> & varlist) {
  //    e_num = -1;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "ux")
      ux_num = i;
    if (varlist[i] == "pr")
      pr_num = i;
    if (varlist[i] == "uy")
      uy_num = i;
    if (varlist[i] == "uz")
      uz_num = i;
    //      if (varlist[i] == "e")
    //        e_num = i;
  }
  //    if (e_num >= 0)
  //      have_energy = true;
}
