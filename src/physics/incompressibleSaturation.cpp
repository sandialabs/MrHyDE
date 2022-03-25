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

#include "incompressibleSaturation.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

incompressibleSaturation::incompressibleSaturation(Teuchos::ParameterList & settings, const int & dimension_)
  : physicsbase(settings, dimension_)
{
  
  label = "incompressibleSaturation";

  // save spaceDim here because it is needed (potentially) before workset is finalized

  spaceDim = dimension_;

  // The state is defined by (S,u) where S is the water saturation and u is the velocity
  // phi is the constant porosity

  myvars.push_back("S");
  myvars.push_back("ux");
  if (spaceDim > 1) {
    myvars.push_back("uy");
  }
  if (spaceDim > 2) {
    myvars.push_back("uz");
  }

  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  if (spaceDim > 1) {
    mybasistypes.push_back("HGRAD");
  }
  if (spaceDim > 2) {
    mybasistypes.push_back("HGRAD");
  }
  
  // Params from input file

  phi = settings.get<ScalarT>("porosity",.5);

}

// ========================================================================================
// ========================================================================================

void incompressibleSaturation::defineFunctions(Teuchos::ParameterList & fs,
                            Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  // TODO not supported right now?
  functionManager->addFunction("source S",fs.get<string>("source S","0.0"),"ip");
  functionManager->addFunction("source ux",fs.get<string>("source ux","0.0"),"ip");
  if (spaceDim > 1) {
    functionManager->addFunction("source uy",fs.get<string>("source uy","0.0"),"ip");
  }
  if (spaceDim > 2) {
    functionManager->addFunction("source uz",fs.get<string>("source uz","0.0"),"ip");
  }

  // Storage for the flux vectors

  fluxes_vol  = View_AD4("flux", functionManager->numElem,
                         functionManager->numip, 1, spaceDim); // neqn = 1

}

// ========================================================================================
// ========================================================================================

void incompressibleSaturation::volumeResidual() {
  
  vector<Vista> sourceterms;

  // TODO is this a good way to do this??
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    sourceterms.push_back(functionManager->evaluate("source S","ip"));
    sourceterms.push_back(functionManager->evaluate("source ux","ip"));
    if (spaceDim > 1) {
      sourceterms.push_back(functionManager->evaluate("source uy","ip"));
    }
    if (spaceDim > 2) {
      sourceterms.push_back(functionManager->evaluate("source uz","ip"));
    }

    // Update fluxes 
    this->computeFluxVector();
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;
  auto varlist = wkset->varlist;

  // The flux storage is (numElem,numip,eqn,dimension)
  
  // outer loop over equations
  for (size_t iEqn=0; iEqn<varlist.size(); ++iEqn) {
    int basis_num = wkset->usebasis[iEqn];
    auto basis = wkset->basis[basis_num];
    auto basis_grad = wkset->basis_grad[basis_num];
    auto dSi_dt = wkset->getSolutionField(varlist[iEqn]+"_t");
    auto source_i = sourceterms[iEqn];
    auto off = subview(wkset->offsets,iEqn,ALL());

    if (spaceDim == 1) {

      // All equations are of the form
      // (v_i,phi d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (v_i,source)

      parallel_for("saturation volume resid " + varlist[iEqn] + " 1D",
              RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += phi*dSi_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,iEqn,0)*basis_grad(elem,dof,pt,0) 
                + source_i(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }
    else if (spaceDim == 2) {

      // All equations are of the form
      // (v_i,phi d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (dv_i/dx_2,F_{y,i}) - (v_i,source)

      parallel_for("saturation volume resid " + varlist[iEqn] + " 2D",
              RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += phi*dSi_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,iEqn,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,iEqn,1)*basis_grad(elem,dof,pt,1)
                + source_i(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }
    else if (spaceDim == 3) {

      // All equations are of the form
      // (v_i,phi d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (dv_i/dx_2,F_{y,i}) - (dv_i/dx_3,F_{z,i}) -(v_i,source)

      parallel_for("saturation volume resid " + varlist[iEqn] + " 3D",
              RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += phi*dSi_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,iEqn,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,iEqn,1)*basis_grad(elem,dof,pt,1)
                + fluxes_vol(elem,pt,iEqn,2)*basis_grad(elem,dof,pt,2)
                + source_i(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }
  }
}

// ========================================================================================
// ========================================================================================

void incompressibleSaturation::boundaryResidual() {
  
  // Nothing for now...

}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void incompressibleSaturation::computeFlux() {

  // Nothing for now...

}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

void incompressibleSaturation::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;

  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); ++i) {
    if (varlist[i] == "S")
      S_num = i;
    else if (varlist[i] == "ux")
      ux_num = i;
    else if (varlist[i] == "uy")
      uy_num = i;
    else if (varlist[i] == "uz")
      uz_num = i;
  }

}

// ========================================================================================
// compute the fluxes
// ========================================================================================

void incompressibleSaturation::computeFluxVector() {

  Teuchos::TimeMonitor localtime(*fluxVectorFill);

  // The flux storage is (numElem,numip,eqn,dimension)
  // The volume fluxes are defined in terms of the state

  auto fluxes = fluxes_vol;
  // these are always needed
  auto ux = wkset->getSolutionField("ux");
  
  if (spaceDim == 1) {

    parallel_for("saturation fluxes 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // S equation -- F_x = ux

        fluxes(elem,pt,S_num,0) = ux(elem,pt);

      }
    });
  } 
  else if (spaceDim == 2) {
    // get the second velocity component
    auto uy = wkset->getSolutionField("uy");

    parallel_for("saturation fluxes 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // S equation -- F_x = ux F_y = uy

        fluxes(elem,pt,S_num,0) = ux(elem,pt);
        fluxes(elem,pt,S_num,1) = uy(elem,pt);

      }
    });
  } 
  else if (spaceDim == 3) {
    // get the second and third velocity component
    auto uy = wkset->getSolutionField("uy");
    auto uz = wkset->getSolutionField("uz");

    parallel_for("saturation fluxes 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // S equation -- F_x = ux F_y = uy F_z = uz

        fluxes(elem,pt,S_num,0) = ux(elem,pt);
        fluxes(elem,pt,S_num,1) = uy(elem,pt);
        fluxes(elem,pt,S_num,2) = uz(elem,pt);

      }
    });
  }
}
