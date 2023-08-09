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

template<class EvalT>
physicsTest<EvalT>::physicsTest(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  Teuchos::ParameterList test_settings = settings.sublist("test settings");

  // Standard data
  label = "physicsTest";
  myvars.push_back("p");
  mybasistypes.push_back(test_settings.get<string>("discretization","HGRAD"));
  myoperators.push_back(test_settings.get<string>("operator","projection"));

  std::cout << "Using the following physicsTest settings: " << std::endl;
  std::cout << test_settings;
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void physicsTest<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                             Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void physicsTest<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int p_basis_num = wkset->usebasis[pnum];
  auto wts = wkset->wts;
  auto res = wkset->res;
  auto off = subview(wkset->offsets, pnum, ALL());

  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  if(mybasistypes[0] == "HGRAD" || mybasistypes[0] == "HFACE" || mybasistypes[0] == "HVOL") { // If we're working with scalar-valued basis functions
    auto psol = wkset->getSolutionField("p");
  
    if(myoperators[0] == "projection") {
      auto basis = wkset->basis[p_basis_num];
      for(int elem = 0; elem < wkset->numElem; elem++) {
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          for (size_type pt=0; pt<psol.extent(1); pt++) {
            EvalT mass = psol(elem,pt)*wts(elem,pt);
            res(elem,off(dof)) += mass*basis(elem,dof,pt,0);
            std::cout << "dof " << dof << ", point " << pt << ": " << basis(elem,dof,pt,0) << std::endl;
          }
        }
      }
    }
    else if(myoperators[0] == "Laplace") {
      auto basis_grad = wkset->basis_grad[p_basis_num];
      if (spaceDim == 1) {
        auto dpdx = wkset->getSolutionField("grad(p)[x]");
        for(int elem = 0; elem < wkset->numElem; elem++) {
          for (size_type dof=0; dof<basis_grad.extent(1); dof++) {
            for (size_type pt=0; pt<psol.extent(1); pt++) {
              EvalT Kx = dpdx(elem,pt)*wts(elem,pt);
              res(elem,off(dof)) += Kx*basis_grad(elem,dof,pt,0);
              std::cout << "dof " << dof << ", point " << pt << " grad: (" << basis_grad(elem,dof,pt,0) << ")" << std::endl;
            }
          }
        }
      }
      else if (spaceDim == 2) {
        auto dpdx = wkset->getSolutionField("grad(p)[x]");
        auto dpdy = wkset->getSolutionField("grad(p)[y]");
        for(int elem = 0; elem < wkset->numElem; elem++) {
          for (size_type dof=0; dof<basis_grad.extent(1); dof++) {
            for (size_type pt=0; pt<psol.extent(1); pt++) {
              EvalT Kx = dpdx(elem,pt)*wts(elem,pt);
              EvalT Ky = dpdy(elem,pt)*wts(elem,pt);
              res(elem,off(dof)) += Kx*basis_grad(elem,dof,pt,0) + Ky*basis_grad(elem,dof,pt,1);
              std::cout << "dof " << dof << ", point " << pt << " grad: (" << basis_grad(elem,dof,pt,0) << "," << basis_grad(elem,dof,pt,1) << ")" << std::endl;
            }
          }
        }
      }
      else if (spaceDim == 3) {
        auto dpdx = wkset->getSolutionField("grad(p)[x]");
        auto dpdy = wkset->getSolutionField("grad(p)[y]");
        auto dpdz = wkset->getSolutionField("grad(p)[z]");
        for(int elem = 0; elem < wkset->numElem; elem++) {
          for (size_type dof=0; dof<basis_grad.extent(1); dof++) {
            for (size_type pt=0; pt<psol.extent(1); pt++) {
              EvalT Kx = dpdx(elem,pt)*wts(elem,pt);
              EvalT Ky = dpdy(elem,pt)*wts(elem,pt);
              EvalT Kz = dpdz(elem,pt)*wts(elem,pt);
              res(elem,off(dof)) += Kx*basis_grad(elem,dof,pt,0) + Ky*basis_grad(elem,dof,pt,1) + Kz*basis_grad(elem,dof,pt,2);
              std::cout << "dof " << dof << ", point " << pt << " grad: (" << basis_grad(elem,dof,pt,0) << "," << basis_grad(elem,dof,pt,1) << "," << basis_grad(elem,dof,pt,2) << ")" << std::endl;
            }
          }
        }
      }
    }
    else {
      std::cout << "Operator name " << myoperators[0] << " is not valid for the specified problem. No assembly was performed on volumes!" << std::endl;
    }
  }
  else if(mybasistypes[0] == "HDIV" || mybasistypes[0] == "HDIV_AC" || mybasistypes[0] == "HCURL") { // If we're working with vector-valued basis functions
    if(myoperators[0] == "projection") {

      if (spaceDim == 1) {
        auto px = wkset->getSolutionField("p[x]");
        auto basis = wkset->basis[p_basis_num];
        for(int elem = 0; elem < wkset->numElem; elem++) {
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            for (size_type pt=0; pt<px.extent(1); pt++) {
              EvalT mass_x = px(elem,pt)*wts(elem,pt);
              res(elem,off(dof)) += mass_x*basis(elem,dof,pt,0);
              std::cout << "dof " << dof << ", point " << pt << ": (" << basis(elem,dof,pt,0) << ")" << std::endl;
            }
          }
        }
      }
      else if (spaceDim == 2) {
        auto px = wkset->getSolutionField("p[x]");
        auto py = wkset->getSolutionField("p[y]");
        auto basis = wkset->basis[p_basis_num];
        for(int elem = 0; elem < wkset->numElem; elem++) {
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            for (size_type pt=0; pt<px.extent(1); pt++) {
              EvalT mass_x = px(elem,pt)*wts(elem,pt);
              EvalT mass_y = py(elem,pt)*wts(elem,pt);
              res(elem,off(dof)) += mass_x*basis(elem,dof,pt,0) + mass_y*basis(elem,dof,pt,1);
              std::cout << "dof " << dof << ", point " << pt << ": (" << basis(elem,dof,pt,0) << "," << basis(elem,dof,pt,1) << ")" << std::endl;
            }
          }
        }
      }
      else if (spaceDim == 3) {
        auto px = wkset->getSolutionField("p[x]");
        auto py = wkset->getSolutionField("p[y]");
        auto pz = wkset->getSolutionField("p[z]");
        auto basis = wkset->basis[p_basis_num];
        for(int elem = 0; elem < wkset->numElem; elem++) {
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            for (size_type pt=0; pt<px.extent(1); pt++) {
              EvalT mass_x = px(elem,pt)*wts(elem,pt);
              EvalT mass_y = py(elem,pt)*wts(elem,pt);
              EvalT mass_z = pz(elem,pt)*wts(elem,pt);
              res(elem,off(dof)) += mass_x*basis(elem,dof,pt,0) + mass_y*basis(elem,dof,pt,1) + mass_z*basis(elem,dof,pt,2);
              std::cout << "dof " << dof << ", point " << pt << ": (" << basis(elem,dof,pt,0) << "," << basis(elem,dof,pt,1) << "," << basis(elem,dof,pt,2) << ")" << std::endl;
            }
          }
        }
      }
    }
    else {
      std::cout << "Operator name " << myoperators[0] << " is not valid for the specified problem. No assembly was performed on volumes!" << std::endl;
    }
  }
}


// ========================================================================================
// ========================================================================================

template<class EvalT>
void physicsTest<EvalT>::boundaryResidual() {
  // No boundary conditions for now
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void physicsTest<EvalT>::edgeResidual() {
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void physicsTest<EvalT>::computeFlux() {
  // No flux for now
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void physicsTest<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
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

template<class EvalT>
void physicsTest<EvalT>::updatePerm(View_EvalT2 perm) {
  
  View_Sc2 data = wkset->extra_data;
  
  parallel_for("physicsTest HGRAD update perm",
               RangePolicy<AssemblyExec>(0,perm.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<perm.extent(1); pt++) {
      perm(elem,pt) = data(elem,0);
    }
  });
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

// Avoid redefining since ScalarT=AD if no AD
#ifndef MrHyDE_NO_AD
template class MrHyDE::physicsTest<ScalarT>;
#endif

// Custom AD type
template class MrHyDE::physicsTest<AD>;

// Standard built-in types
template class MrHyDE::physicsTest<AD2>;
template class MrHyDE::physicsTest<AD4>;
template class MrHyDE::physicsTest<AD8>;
template class MrHyDE::physicsTest<AD16>;
template class MrHyDE::physicsTest<AD18>;
template class MrHyDE::physicsTest<AD24>;
template class MrHyDE::physicsTest<AD32>;
