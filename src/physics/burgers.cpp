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

#include "burgers.hpp"

using namespace MrHyDE;

// ========================================================================================
// ========================================================================================

Burgers::Burgers(Teuchos::ParameterList & settings, const int & dimension_)
  : physicsbase(settings, dimension_)
{
  
  label = "Burgers";
  myvars.push_back("u");
  mybasistypes.push_back("HGRAD");
  
}

// ========================================================================================
// ========================================================================================

void Burgers::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  functionManager->addFunction("Burgers source",fs.get<string>("Burgers source","0.0"),"ip");
  functionManager->addFunction("diffusion",fs.get<string>("diffusion","0.0"),"ip");
  functionManager->addFunction("xvel",fs.get<string>("xvel","1.0"),"ip");
  functionManager->addFunction("yvel",fs.get<string>("yvel","1.0"),"ip");
  functionManager->addFunction("zvel",fs.get<string>("zvel","1.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

void Burgers::volumeResidual() {
  
  // Evaluate the functions we always need
  auto source = functionManager->evaluate("Burgers source","ip");
  auto eps = functionManager->evaluate("diffusion","ip");
  auto vx = functionManager->evaluate("xvel","ip");
  
  // Get some information from the workset
  auto basis = wkset->getBasis("u");
  auto basis_grad = wkset->getBasisGrad("u");
  auto res = wkset->res;
  auto wts = wkset->wts;
  auto off = wkset->getOffsets("u");
  auto u = wkset->getSolutionField("u");
  auto dudt = wkset->getSolutionField("u_t");
  
  // Solves dudt + div (1/2*v u^2 - eps grad u) = source(x,t)
  if (wkset->dimension == 1) {
    auto dudx = wkset->getSolutionField("grad(u)[x]");
    parallel_for("Burgers volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD usq = 0.5*u(elem,pt)*u(elem,pt);
        AD f = (dudt(elem,pt) - source(elem,pt))*wts(elem,pt);
        AD Fx = (eps(elem,pt)*dudx(elem,pt) - vx(elem,pt)*usq)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
        }
      }
    });
  }
  else if (wkset->dimension == 2) {
    auto dudx = wkset->getSolutionField("grad(u)[x]");
    auto dudy = wkset->getSolutionField("grad(u)[y]");
    auto vy = functionManager->evaluate("yvel","ip");
    parallel_for("Burgers volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD usq = 0.5*u(elem,pt)*u(elem,pt);
        AD f = (dudt(elem,pt) - source(elem,pt))*wts(elem,pt);
        AD Fx = (eps(elem,pt)*dudx(elem,pt) - vx(elem,pt)*usq)*wts(elem,pt);
        AD Fy = (eps(elem,pt)*dudy(elem,pt) - vy(elem,pt)*usq)*wts(elem,pt);
        
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
        }
      }
    });
  }
  else if (wkset->dimension == 3) {
    auto dudx = wkset->getSolutionField("grad(u)[x]");
    auto dudy = wkset->getSolutionField("grad(u)[y]");
    auto dudz = wkset->getSolutionField("grad(u)[z]");
    auto vy = functionManager->evaluate("yvel","ip");
    auto vz = functionManager->evaluate("zvel","ip");
    parallel_for("Burgers volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD usq = 0.5*u(elem,pt)*u(elem,pt);
        AD f = (dudt(elem,pt) - source(elem,pt))*wts(elem,pt);
        AD Fx = (eps(elem,pt)*dudx(elem,pt) - vx(elem,pt)*usq)*wts(elem,pt);
        AD Fy = (eps(elem,pt)*dudy(elem,pt) - vy(elem,pt)*usq)*wts(elem,pt);
        AD Fz = (eps(elem,pt)*dudz(elem,pt) - vz(elem,pt)*usq)*wts(elem,pt);
        
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += f*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
        }
      }
    });
  }
  
}

