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

#include "physics_test.hpp"
using namespace MrHyDE;

physicsTest::physicsTest(Teuchos::ParameterList & settings, const int & dimension_)
  : physicsbase(settings, dimension_)
{
  Teuchos::ParameterList test_settings = settings.sublist("test settings");

  // Standard data
  label = "physicsTest";
  myvars.push_back(test_settings.get<string>("variable name","p"));
  mybasistypes.push_back(test_settings.get<string>("discretization","HGRAD"));
  myoperators.push_back(test_settings.get<string>("operator","projection"));

  std::cout << "Using the following physicsTest settings: " << std::endl;
  std::cout << test_settings << std::endl;
}

// ========================================================================================
// ========================================================================================

void physicsTest::defineFunctions(Teuchos::ParameterList & fs,
                             Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
}

// ========================================================================================
// ========================================================================================

void physicsTest::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int p_basis_num = wkset->usebasis[pnum];
  auto wts = wkset->wts;
  auto res = wkset->res;
  auto off = subview(wkset->offsets, pnum, ALL());

  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  if(mybasistypes[0] == "HGRAD") { // If we're working with scalar-valued basis functions
    auto psol = wkset->getData("p");
  
    if(myoperators[0] == "projection") {
      auto basis = wkset->basis[p_basis_num];
      for(int elem = 0; elem < wkset->numElem; elem++) {
        for (size_type pt=0; pt<psol.extent(1); pt++) {
          AD mass = psol(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += mass*basis(elem,dof,pt,0);
            std::cout << "dof " << dof << ", point " << pt << ": " << basis(elem,dof,pt,0) << std::endl;
          }
        }
      }
    }
    else if(myoperators[0] == "Laplace") {
      auto basis_grad = wkset->basis_grad[p_basis_num];
      if (spaceDim == 1) {
        auto dpdx = wkset->getData("grad(p)[x]");
        for(int elem = 0; elem < wkset->numElem; elem++) {
          for (size_type pt=0; pt<psol.extent(1); pt++) {
            AD Kx = dpdx(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis_grad.extent(1); dof++) {
              res(elem,off(dof)) += Kx*basis_grad(elem,dof,pt,0);
              std::cout << "dof " << dof << ", point " << pt << " grad: (" << basis_grad(elem,dof,pt,0) << ")" << std::endl;
            }
          }
        }
      }
      else if (spaceDim == 2) {
        auto dpdx = wkset->getData("grad(p)[x]");
        auto dpdy = wkset->getData("grad(p)[y]");
        for(int elem = 0; elem < wkset->numElem; elem++) {
          for (size_type pt=0; pt<psol.extent(1); pt++) {
            AD Kx = dpdx(elem,pt)*wts(elem,pt);
            AD Ky = dpdy(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis_grad.extent(1); dof++) {
              res(elem,off(dof)) += Kx*basis_grad(elem,dof,pt,0) + Ky*basis_grad(elem,dof,pt,1);
              std::cout << "dof " << dof << ", point " << pt << " grad: (" << basis_grad(elem,dof,pt,0) << "," << basis_grad(elem,dof,pt,1) << ")" << std::endl;
            }
          }
        }
      }
      else if (spaceDim == 3) {
        auto dpdx = wkset->getData("grad(p)[x]");
        auto dpdy = wkset->getData("grad(p)[y]");
        auto dpdz = wkset->getData("grad(p)[z]");
        for(int elem = 0; elem < wkset->numElem; elem++) {
          for (size_type pt=0; pt<psol.extent(1); pt++) {
            AD Kx = dpdx(elem,pt)*wts(elem,pt);
            AD Ky = dpdy(elem,pt)*wts(elem,pt);
            AD Kz = dpdz(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis_grad.extent(1); dof++) {
              res(elem,off(dof)) += Kx*basis_grad(elem,dof,pt,0) + Ky*basis_grad(elem,dof,pt,1) + Kz*basis_grad(elem,dof,pt,2);
              std::cout << "dof " << dof << ", point " << pt << " grad: (" << basis_grad(elem,dof,pt,0) << "," << basis_grad(elem,dof,pt,1) << "," << basis_grad(elem,dof,pt,2) << ")" << std::endl;
            }
          }
        }
      }
    }
    else {
      std::cout << "Operator name " << myoperators[0] << " not found! No assembly was performed on volumes." << std::endl;
    }
  }
  else if(mybasistypes[0] == "HDIV" || mybasistypes[0] == "HDIV_AC" || mybasistypes[0] == "HCURL") { // If we're working with vector-valued basis functions
    auto px = wkset->getData("p[x]");
    auto py = wkset->getData("p[y]");
    //auto pz = wkset->getData("p[z]");
  
    if(myoperators[0] == "projection") {
      auto basis = wkset->basis[p_basis_num];
      for(int elem = 0; elem < wkset->numElem; elem++) {
        for (size_type pt=0; pt<px.extent(1); pt++) {
          AD mass_x = px(elem,pt)*wts(elem,pt);
          AD mass_y = py(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += mass_x*basis(elem,dof,pt,0) + mass_y*basis(elem,dof,pt,1);
            std::cout << "dof " << dof << ", point " << pt << ": (" << basis(elem,dof,pt,0) << "," << basis(elem,dof,pt,1) << ")" << std::endl;
          }
        }
      }
    }
    else {
      std::cout << "Operator name " << myoperators[0] << " not found! No assembly was performed on volumes." << std::endl;
    }
  }
}


// ========================================================================================
// ========================================================================================

void physicsTest::boundaryResidual() {
  // No boundary conditions for now
}

// ========================================================================================
// ========================================================================================

void physicsTest::edgeResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void physicsTest::computeFlux() {
  // No flux for now
}

// ========================================================================================
// ========================================================================================

void physicsTest::setWorkset(Teuchos::RCP<workset> & wkset_) {
  wkset = wkset_;
  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "p") {
      pnum = i;
    }
  }
}

// ========================================================================================
// ========================================================================================

void physicsTest::updatePerm(View_AD2 perm) {
  
  View_Sc2 data = wkset->extra_data;
  
  parallel_for("physicsTest HGRAD update perm",
               RangePolicy<AssemblyExec>(0,perm.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<perm.extent(1); pt++) {
      perm(elem,pt) = data(elem,0);
    }
  });
}
