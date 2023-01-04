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

#include "euler.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

euler::euler(Teuchos::ParameterList & settings, const int & dimension_)
  : physicsbase(settings, dimension_)
{
  
  label = "euler";

  // save spaceDim here because it is needed (potentially) before workset is finalized
  // TODO is this needed?

  spaceDim = dimension_;

  myvars.push_back("rho");
  myvars.push_back("rhoux");
  myvars.push_back("rhoE");
  if (spaceDim > 1) {
    myvars.push_back("rhouy");
  }
  if (spaceDim > 2) {
    myvars.push_back("rhouz");
  }

  // we take the state to be the vector of conserved quantities (\rho,\rho u_i, \rho E)
  // where E is the total energy density per unit mass
  
  // TODO appropriate types?
  // I think this is setup to do the fully hybridized
 
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

  // TODO :: Dimensionless params?
  // Following Anderson and Tanehill 
  // V_\infty, L, \rho_\infty, T_\infty are needed
  
  // Stabilization options, need to choose one
  maxEVstab = settings.get<bool>("max EV stabilization",false);
  roestab = settings.get<bool>("Roe-like stabilization",false);

  if ( ! ( maxEVstab || roestab ) ) {
    std::cout << "Error: No stabilization method chosen! Specify in input file!" << std::endl;
  }

  // We default to properties of air at 293 K, Mach .01
  // Store these as model params as they are constant, but they can also be 
  // normal ADs as with p0, T, etc.
 
  modelparams = Kokkos::View<ScalarT*,AssemblyDevice>("parameters for euler",8);
  auto modelparams_host = Kokkos::create_mirror_view(modelparams);

  // Specific heat at constant pressure  units are L^2/T^2-K (K must be Kelvin) 
  modelparams_host(cp_mp_num) = settings.get<ScalarT>("cp",1004.5);
  // Ratio of specific heats
  modelparams_host(gamma_mp_num) = settings.get<ScalarT>("gamma",1.4);
  // Specific gas constant  units are J/kg-K
  modelparams_host(RGas_mp_num) = settings.get<ScalarT>("RGas",287.0);
  // Reference velocity  units are m/s
  modelparams_host(URef_mp_num) = settings.get<ScalarT>("URef",3.431143);
  // Reference length  units are m  TODO don't think it matters?
  modelparams_host(LRef_mp_num) = settings.get<ScalarT>("LRef",1.0); 
  // Reference density  units are kg/m^3
  modelparams_host(rhoRef_mp_num) = settings.get<ScalarT>("rhoRef",1.189);
  // Reference temperature  units are K
  modelparams_host(TRef_mp_num) = settings.get<ScalarT>("TRef",293.0);
  // Reference Mach number  U/sqrt(gamma*R*T)
  modelparams_host(MRef_mp_num) = 
    modelparams_host(URef_mp_num)/sqrt(modelparams_host(gamma_mp_num)*
                                       modelparams_host(RGas_mp_num)*
                                       modelparams_host(TRef_mp_num));

  Kokkos::deep_copy(modelparams, modelparams_host);

}

// ========================================================================================
// ========================================================================================

void euler::defineFunctions(Teuchos::ParameterList & fs,
                            Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  // TODO not supported right now?
  functionManager->addFunction("source rho",fs.get<string>("source rho","0.0"),"ip");
  functionManager->addFunction("source rhoux",fs.get<string>("source rhoux","0.0"),"ip");
  functionManager->addFunction("source rhoE", fs.get<string>("source rhoE", "0.0"),"ip");
  if (spaceDim > 1) {
    functionManager->addFunction("source rhouy",fs.get<string>("source rhouy","0.0"),"ip");
  }
  if (spaceDim > 2) {
    functionManager->addFunction("source rhouz",fs.get<string>("source rhouz","0.0"),"ip");
  }

  // Storage for the inviscid flux vectors

  fluxes_vol  = View_AD4("inviscid flux", functionManager->numElem,
                         functionManager->numip, spaceDim + 2, spaceDim); // neqn = spaceDim + 2
  fluxes_side = View_AD4("inviscid flux", functionManager->numElem,
                         functionManager->numip_side, spaceDim + 2, spaceDim); // see above 

  // Storage for stabilization term/boundary flux

  // The stabilization term which completes the numerical flux along interfaces
  // \hat{F} \cdot n = F(\hat{S}) \cdot n + Stab(S,\hat{S}) \times (S - \hat{S}) 
  // We store all of Stab(S,\hat{S}) \times (S - \hat{S}) 
  
  // Additionally, this storage is used for the boundary flux B(\hat{S}).
  // This is needed by the computeFlux routine (see for more details).

  stab_bound_side = View_AD3("stab/boundary term", functionManager->numElem,
                             functionManager->numip_side, spaceDim + 2); // see above 

  // Storage for the thermodynamic properties
  // TODO I don't think T is technically needed... 
  
  props_vol  = View_AD3("thermo props", functionManager->numElem,
                       functionManager->numip, 3);
  props_side = View_AD3("thermo props", functionManager->numElem,
                       functionManager->numip_side, 3);

}

// ========================================================================================
// ========================================================================================

void euler::volumeResidual() {
  
  Vista source_rho, source_rhoux, source_rhouy, source_rhouz, source_rhoE;

  // TODO not currently using source terms
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source_rho = functionManager->evaluate("source rho","ip");
    source_rhoux = functionManager->evaluate("source rhoux","ip");
    source_rhoE  = functionManager->evaluate("source rhoE","ip");
    if (spaceDim > 1) {
      source_rhouy = functionManager->evaluate("source rhouy","ip");
    }
    if (spaceDim > 2) {
      source_rhouz = functionManager->evaluate("source rhouz","ip");
    }

    // Update thermodynamic and fluxes properties

    this->computeThermoProps(false); // not on_side
    this->computeInviscidFluxes(false); // not on_side 
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;

  // The flux storage is (numElem,numip,eqn,dimension)
  
  if (spaceDim == 1) {

    // All equations are of the form
    // (v_i,d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (v_i,source)

    // rho
    {
      int rho_basis = wkset->usebasis[rho_num];
      auto basis = wkset->basis[rho_basis];
      auto basis_grad = wkset->basis_grad[rho_basis];
      auto drho_dt = wkset->getSolutionField("rho_t");
      auto off = subview(wkset->offsets,rho_num,ALL());
      
      parallel_for("euler rho volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drho_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rho_num,0)*basis_grad(elem,dof,pt,0) 
                + source_rho(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

    // rhoux
    {
      int rhoux_basis = wkset->usebasis[rhoux_num];
      auto basis = wkset->basis[rhoux_basis];
      auto basis_grad = wkset->basis_grad[rhoux_basis];
      auto drhoux_dt = wkset->getSolutionField("rhoux_t");
      auto off = subview(wkset->offsets,rhoux_num,ALL());
      
      parallel_for("euler rhoux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drhoux_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rhoux_num,0)*basis_grad(elem,dof,pt,0) 
                + source_rhoux(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }
      
    // rhoE
    {
      int rhoE_basis = wkset->usebasis[rhoE_num];
      auto basis = wkset->basis[rhoE_basis];
      auto basis_grad = wkset->basis_grad[rhoE_basis];
      auto drhoE_dt = wkset->getSolutionField("rhoE_t");
      auto off = subview(wkset->offsets,rhoE_num,ALL());
      
      parallel_for("euler rhoE volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drhoE_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rhoE_num,0)*basis_grad(elem,dof,pt,0) 
                + source_rhoE(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

  }
  else if (spaceDim == 2) {

    // All equations are of the form
    // (v_i,d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (dv_i/dx_2,F_{y,i}) - (v_i,source)

    // rho
    {
      int rho_basis = wkset->usebasis[rho_num];
      auto basis = wkset->basis[rho_basis];
      auto basis_grad = wkset->basis_grad[rho_basis];
      auto drho_dt = wkset->getSolutionField("rho_t");
      auto off = subview(wkset->offsets,rho_num,ALL());
      
      parallel_for("euler rho volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drho_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rho_num,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,rho_num,1)*basis_grad(elem,dof,pt,1)
                + source_rho(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

    // rhoux
    {
      int rhoux_basis = wkset->usebasis[rhoux_num];
      auto basis = wkset->basis[rhoux_basis];
      auto basis_grad = wkset->basis_grad[rhoux_basis];
      auto drhoux_dt = wkset->getSolutionField("rhoux_t");
      auto off = subview(wkset->offsets,rhoux_num,ALL());
      
      parallel_for("euler rhoux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drhoux_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rhoux_num,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,rhoux_num,1)*basis_grad(elem,dof,pt,1)
                + source_rhoux(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

    // rhouy
    {
      int rhouy_basis = wkset->usebasis[rhouy_num];
      auto basis = wkset->basis[rhouy_basis];
      auto basis_grad = wkset->basis_grad[rhouy_basis];
      auto drhouy_dt = wkset->getSolutionField("rhouy_t");
      auto off = subview(wkset->offsets,rhouy_num,ALL());
      
      parallel_for("euler rhouy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drhouy_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rhouy_num,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,rhouy_num,1)*basis_grad(elem,dof,pt,1)
                + source_rhouy(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

    // rhoE
    {
      int rhoE_basis = wkset->usebasis[rhoE_num];
      auto basis = wkset->basis[rhoE_basis];
      auto basis_grad = wkset->basis_grad[rhoE_basis];
      auto drhoE_dt = wkset->getSolutionField("rhoE_t");
      auto off = subview(wkset->offsets,rhoE_num,ALL());
      
      parallel_for("euler rhoE volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drhoE_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rhoE_num,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,rhoE_num,1)*basis_grad(elem,dof,pt,1)
                + source_rhoE(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

  }
  else if (spaceDim == 3) {

    // All equations are of the form
    // (v_i,d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (dv_i/dx_2,F_{y,i}) 
    // - (dv_i/dx_3,F_{z,i}) - (v_i,source)

    // rho
    {
      int rho_basis = wkset->usebasis[rho_num];
      auto basis = wkset->basis[rho_basis];
      auto basis_grad = wkset->basis_grad[rho_basis];
      auto drho_dt = wkset->getSolutionField("rho_t");
      auto off = subview(wkset->offsets,rho_num,ALL());
      
      parallel_for("euler rho volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drho_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rho_num,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,rho_num,1)*basis_grad(elem,dof,pt,1)
                + fluxes_vol(elem,pt,rho_num,2)*basis_grad(elem,dof,pt,2)
                + source_rho(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

    // rhoux
    {
      int rhoux_basis = wkset->usebasis[rhoux_num];
      auto basis = wkset->basis[rhoux_basis];
      auto basis_grad = wkset->basis_grad[rhoux_basis];
      auto drhoux_dt = wkset->getSolutionField("rhoux_t");
      auto off = subview(wkset->offsets,rhoux_num,ALL());
      
      parallel_for("euler rhoux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drhoux_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rhoux_num,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,rhoux_num,1)*basis_grad(elem,dof,pt,1)
                + fluxes_vol(elem,pt,rhoux_num,2)*basis_grad(elem,dof,pt,2)
                + source_rhoux(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

    // rhouy
    {
      int rhouy_basis = wkset->usebasis[rhouy_num];
      auto basis = wkset->basis[rhouy_basis];
      auto basis_grad = wkset->basis_grad[rhouy_basis];
      auto drhouy_dt = wkset->getSolutionField("rhouy_t");
      auto off = subview(wkset->offsets,rhouy_num,ALL());
      
      parallel_for("euler rhouy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drhouy_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rhouy_num,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,rhouy_num,1)*basis_grad(elem,dof,pt,1)
                + fluxes_vol(elem,pt,rhouy_num,2)*basis_grad(elem,dof,pt,2)
                + source_rhouy(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

    // rhouz
    {
      int rhouz_basis = wkset->usebasis[rhouz_num];
      auto basis = wkset->basis[rhouz_basis];
      auto basis_grad = wkset->basis_grad[rhouz_basis];
      auto drhouz_dt = wkset->getSolutionField("rhouz_t");
      auto off = subview(wkset->offsets,rhouz_num,ALL());
      
      parallel_for("euler rhouz volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drhouz_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rhouz_num,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,rhouz_num,1)*basis_grad(elem,dof,pt,1)
                + fluxes_vol(elem,pt,rhouz_num,2)*basis_grad(elem,dof,pt,2)
                + source_rhouz(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

    // rhoE
    {
      int rhoE_basis = wkset->usebasis[rhoE_num];
      auto basis = wkset->basis[rhoE_basis];
      auto basis_grad = wkset->basis_grad[rhoE_basis];
      auto drhoE_dt = wkset->getSolutionField("rhoE_t");
      auto off = subview(wkset->offsets,rhoE_num,ALL());
      
      parallel_for("euler rhoE volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += drhoE_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,rhoE_num,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,rhoE_num,1)*basis_grad(elem,dof,pt,1)
                + fluxes_vol(elem,pt,rhoE_num,2)*basis_grad(elem,dof,pt,2)
                + source_rhoE(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }

  }
}

// ========================================================================================
// ========================================================================================

void euler::boundaryResidual() {
  
  auto bcs = wkset->var_bcs;

  int cside = wkset->currentside;

  Vista source_rho, source_rhoux, source_rhoE, source_rhouy, source_rhouz;

  string rho_sidetype = bcs(rho_num,cside);
  string rhoux_sidetype = bcs(rhoux_num,cside);
  string rhoE_sidetype = bcs(rhoE_num,cside);
  string rhouy_sidetype = ""; string rhouz_sidetype = "";
  if (spaceDim > 1) {
    rhouy_sidetype = bcs(rhouy_num,cside);
  }
  if (spaceDim > 2) {
    rhouz_sidetype = bcs(rhouz_num,cside);
  }

  {
    Teuchos::TimeMonitor funceval(*boundaryResidualFunc);

    // Update thermodynamic and fluxes properties

    this->computeThermoProps(true); // on_side
    this->computeInviscidFluxes(true); // on_side 
    this->computeStabilizationTerm();

  }

  auto wts = wkset->wts_side;
  auto h = wkset->h;
  auto res = wkset->res;

  Teuchos::TimeMonitor localtime(*boundaryResidualFill);

  // These are always needed
  auto nx = wkset->getScalarField("n[x]");
  auto fluxes = fluxes_side;
  auto stab = stab_bound_side;

  // all boundary contributions are of the form ( F(\hat{S}_i) \cdot n, v_i )
  // The boundary conditions are enforced weakly with the trace variables
  // so we ALWAYS compute the aforementioned inner product here

  if (spaceDim == 1) {

    for (int iEqn=0; iEqn<spaceDim+2; ++iEqn) {
  
      int basis_num = wkset->usebasis[iEqn];
      auto basis = wkset->basis_side[basis_num];
      auto off = subview(wkset->offsets,iEqn,ALL());
      
      parallel_for("euler boundary resid eqn: " + std::to_string(iEqn),
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (fluxes(elem,pt,iEqn,0)*nx(elem,pt) + 
                stab(elem,pt,iEqn))*wts(elem,pt)*basis(elem,dof,pt,0);
          }
        }
      });
    }
  }
  else if (spaceDim == 2) {

    // need ny
    auto ny = wkset->getScalarField("n[y]");

    for (int iEqn=0; iEqn<spaceDim+2; ++iEqn) {
  
      int basis_num = wkset->usebasis[iEqn];
      auto basis = wkset->basis_side[basis_num];
      auto off = subview(wkset->offsets,iEqn,ALL());
      
      parallel_for("euler boundary resid eqn: " + std::to_string(iEqn),
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += 
              (fluxes(elem,pt,iEqn,0)*nx(elem,pt) + fluxes(elem,pt,iEqn,1)*ny(elem,pt) +
                stab(elem,pt,iEqn))*wts(elem,pt)*basis(elem,dof,pt,0);
          }
        }
      });
    }
  }
  else if (spaceDim == 3) {
    // need ny, nz
    auto ny = wkset->getScalarField("n[y]");
    auto nz = wkset->getScalarField("n[z]");

    for (int iEqn=0; iEqn<spaceDim+2; ++iEqn) {
  
      int basis_num = wkset->usebasis[iEqn];
      auto basis = wkset->basis_side[basis_num];
      auto off = subview(wkset->offsets,iEqn,ALL());
      
      parallel_for("euler boundary resid eqn: " + std::to_string(iEqn),
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += 
              (fluxes(elem,pt,iEqn,0)*nx(elem,pt) + fluxes(elem,pt,iEqn,1)*ny(elem,pt) 
               + fluxes(elem,pt,iEqn,2)*nz(elem,pt) +
                stab(elem,pt,iEqn))*wts(elem,pt)*basis(elem,dof,pt,0);
          }
        }
      });
    }
  }
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void euler::computeFlux() {

  // see Peraire 2011 AIAA for the details of the implementation
  // they take fluxes on the interfaces to be F(\hat{S}) \cdot n + Stab(S,\hat{S}) ( S - \hat{S} )
  // where Stab = L \| \Lambda \| R or \lambda_{max} I
  //
  // L \Lambda R is the eigendecomposition of J = dF(\hat{S})/d\hat{S} \cdot n
  // the flux Jacobian evaluated using the trace variables.
  //
  // This routine contributes F \cdot n on the interior skeleton and 
  // B(\hat{S}) (the boundary flux) on the boundary of the domain
  //
  // TODO Currently, the BCs are more or less coupled so if one side type 
  // indicates we are at a boundary, then ALL others follow.
  // This is consistent with Peraire, but perhaps could be generalized later.
  
  auto bcs = wkset->var_bcs;

  int cside = wkset->currentside;
  string sidetype = bcs(rho_num,cside);

  {
    Teuchos::TimeMonitor localtime(*fluxFunc);

    this->computeThermoProps(true); // on_side
    this->computeInviscidFluxes(true); // on_side 
    // see note above
    if (sidetype == "Far-field" || sidetype == "Slip") {
      this->computeBoundaryTerm();
    } 
    else {
      this->computeStabilizationTerm();
    }
  }

  {
    Teuchos::TimeMonitor localtime(*fluxFill);

    if (sidetype == "Far-field" || sidetype == "Slip") {
      // just need to copy the boundary term

      auto interfaceFlux = wkset->flux;
      auto bound = stab_bound_side;

      parallel_for("euler boundary flux copy",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (int ieqn=0; ieqn<spaceDim+2; ++ieqn) {
          for (size_type pt=0; pt<bound.extent(1); ++pt) {
            interfaceFlux(elem,ieqn,pt) = bound(elem,pt,ieqn);
          }
        }
      });
    }
    else if (sidetype == "interface") {
      
      // These are always needed
      auto nx = wkset->getScalarField("n[x]");

      auto fluxes = fluxes_side;
      auto stab = stab_bound_side;

      auto interfaceFlux = wkset->flux;

      if (spaceDim == 1) {

        parallel_for("euler flux 1D",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (int ieqn=0; ieqn<spaceDim+2; ++ieqn) {
            for (size_type pt=0; pt<nx.extent(1); ++pt) {
              interfaceFlux(elem,ieqn,pt) = fluxes(elem,pt,ieqn,0)*nx(elem,pt)
                + stab(elem,pt,ieqn);
            }
          }
        });
      } 
      else if (spaceDim == 2) {
        // second normal needed
        auto ny = wkset->getScalarField("n[y]");

        parallel_for("euler flux 2D",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (int ieqn=0; ieqn<spaceDim+2; ++ieqn) {
            for (size_type pt=0; pt<nx.extent(1); ++pt) {
              interfaceFlux(elem,ieqn,pt) = fluxes(elem,pt,ieqn,0)*nx(elem,pt)
                + fluxes(elem,pt,ieqn,1)*ny(elem,pt) + stab(elem,pt,ieqn);
            }
          }
        });
      } 
      else if (spaceDim == 3) {
        // second and third normal needed
        auto ny = wkset->getScalarField("n[y]");
        auto nz = wkset->getScalarField("n[z]");

        parallel_for("euler flux 3D",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (int ieqn=0; ieqn<spaceDim+2; ++ieqn) {
            for (size_type pt=0; pt<nx.extent(1); ++pt) {
              interfaceFlux(elem,ieqn,pt) = fluxes(elem,pt,ieqn,0)*nx(elem,pt)
                + fluxes(elem,pt,ieqn,1)*ny(elem,pt) + fluxes(elem,pt,ieqn,2)*nz(elem,pt) 
                + stab(elem,pt,ieqn);
            }
          }
        });
      } 
    } 
    else {
      cout << "Something's gone wrong... Euler computeFlux()" << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

void euler::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;

  // TODO make this less hard codey?
  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); ++i) {
    if (varlist[i] == "rho")
      rho_num = i;
    else if (varlist[i] == "rhoux")
      rhoux_num = i;
    else if (varlist[i] == "rhouy")
      rhouy_num = i;
    else if (varlist[i] == "rhouz")
      rhouz_num = i;
    else if (varlist[i] == "rhoE")
      rhoE_num = i;
  }

  // TODO make this less hard codey?
  vector<string> auxvarlist = wkset->aux_varlist;
  for (size_t i=0; i<auxvarlist.size(); ++i) {
    if (auxvarlist[i] == "rho")
      auxrho_num = i;
    else if (auxvarlist[i] == "rhoux")
      auxrhoux_num = i;
    else if (auxvarlist[i] == "rhouy")
      auxrhouy_num = i;
    else if (auxvarlist[i] == "rhouz")
      auxrhouz_num = i;
    else if (auxvarlist[i] == "rhoE")
      auxrhoE_num = i;
  }

}

// ========================================================================================
// compute the inviscid fluxes
// ========================================================================================

void euler::computeInviscidFluxes(const bool & on_side) {

  Teuchos::TimeMonitor localtime(*invFluxesFill);

  // The flux storage is (numElem,numip,eqn,dimension)
  // The face fluxes are defined in terms of the trace variables
  // The volume fluxes are defined in terms of the state
  // TODO does being on a side == interface for these schemes?
  // will we require data in any other context....

  auto fluxes = on_side ? fluxes_side : fluxes_vol;
  // these are always needed
  auto rho = on_side ? wkset->getSolutionField("aux rho") : wkset->getSolutionField("rho");
  auto rhoux = on_side ? wkset->getSolutionField("aux rhoux") : wkset->getSolutionField("rhoux");
  auto rhoE = on_side ? wkset->getSolutionField("aux rhoE") : wkset->getSolutionField("rhoE");
  auto props = on_side ? props_side : props_vol;

  // TODO this is the same for face or side, can I collapse?
  // ? : operator... are they of the same type?
  // TODO WILL THAT BE BAD??

  if (spaceDim == 1) {

    parallel_for("euler inviscid fluxes 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // rho equation -- F_x = rhoux

        fluxes(elem,pt,rho_num,0) = rhoux(elem,pt);

        // rhoux equation -- F_x = rhoux**2/rho + p

        fluxes(elem,pt,rhoux_num,0) = 
          rhoux(elem,pt)*rhoux(elem,pt)/rho(elem,pt) + props(elem,pt,p0_num);

        // rhoE equation -- F_x = rhoE*rhoux/rho + p * rhoux/rho 

        fluxes(elem,pt,rhoE_num,0) = 
          rhoE(elem,pt)*rhoux(elem,pt)/rho(elem,pt) 
          + props(elem,pt,p0_num)*rhoux(elem,pt)/rho(elem,pt);

      }
    });
  } 
  else if (spaceDim == 2) {
    // get the second momentum component
    auto rhouy = on_side ? wkset->getSolutionField("aux rhouy") : wkset->getSolutionField("rhouy");

    parallel_for("euler inviscid fluxes 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // rho equation -- F_x = rhoux F_y = rhouy

        fluxes(elem,pt,rho_num,0) = rhoux(elem,pt);
        fluxes(elem,pt,rho_num,1) = rhouy(elem,pt);

        // rhoux equation -- F_x = rhoux**2/rho + p F_y = rhoux*rhouy/rho

        fluxes(elem,pt,rhoux_num,0) = 
          rhoux(elem,pt)*rhoux(elem,pt)/rho(elem,pt) + props(elem,pt,p0_num);
        fluxes(elem,pt,rhoux_num,1) = rhoux(elem,pt)*rhouy(elem,pt)/rho(elem,pt);

        // rhouy equation -- F_x = rhoux*rhouy/rho F_y = rhouy**2/rho + p

        fluxes(elem,pt,rhouy_num,0) = 
          rhoux(elem,pt)*rhouy(elem,pt)/rho(elem,pt); 
        fluxes(elem,pt,rhouy_num,1) = 
          rhouy(elem,pt)*rhouy(elem,pt)/rho(elem,pt) + props(elem,pt,p0_num);

        // rhoE equation -- F_x = rhoE*rhoux/rho + p * rhoux/rho 
        //                  F_y = rhoE*rhouy/rho + p * rhouy/rho

        fluxes(elem,pt,rhoE_num,0) = 
          rhoE(elem,pt)*rhoux(elem,pt)/rho(elem,pt) 
          + props(elem,pt,p0_num)*rhoux(elem,pt)/rho(elem,pt);
        fluxes(elem,pt,rhoE_num,1) = 
          rhoE(elem,pt)*rhouy(elem,pt)/rho(elem,pt) 
          + props(elem,pt,p0_num)*rhouy(elem,pt)/rho(elem,pt);

      }
    });

  } 
  else {
    // get the second and third momentum component
    auto rhouy = on_side ? wkset->getSolutionField("aux rhouy") : wkset->getSolutionField("rhouy");
    auto rhouz = on_side ? wkset->getSolutionField("aux rhouz") : wkset->getSolutionField("rhouz");

    parallel_for("euler inviscid fluxes 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // rho equation -- F_x = rhoux F_y = rhouy F_z = rhouz

        fluxes(elem,pt,rho_num,0) = rhoux(elem,pt);
        fluxes(elem,pt,rho_num,1) = rhouy(elem,pt);
        fluxes(elem,pt,rho_num,2) = rhouz(elem,pt);

        // rhoux equation -- F_x = rhoux**2/rho + p F_y = rhoux*rhouy/rho F_z rhoux*rhouz/rho

        fluxes(elem,pt,rhoux_num,0) = 
          rhoux(elem,pt)*rhoux(elem,pt)/rho(elem,pt) + props(elem,pt,p0_num);
        fluxes(elem,pt,rhoux_num,1) = rhoux(elem,pt)*rhouy(elem,pt)/rho(elem,pt);
        fluxes(elem,pt,rhoux_num,2) = rhoux(elem,pt)*rhouz(elem,pt)/rho(elem,pt);

        // rhouy equation -- F_x = rhoux*rhouy/rho F_y = rhouy**2/rho + p F_z = rhouy*rhouz/rho

        fluxes(elem,pt,rhouy_num,0) = 
          rhoux(elem,pt)*rhouy(elem,pt)/rho(elem,pt); 
        fluxes(elem,pt,rhouy_num,1) = 
          rhouy(elem,pt)*rhouy(elem,pt)/rho(elem,pt) + props(elem,pt,p0_num);
        fluxes(elem,pt,rhouy_num,2) = 
          rhouy(elem,pt)*rhouz(elem,pt)/rho(elem,pt); 

        // rhoE equation -- F_x = rhoE*rhoux/rho + p * rhoux/rho 
        //                  F_y = rhoE*rhouy/rho + p * rhouy/rho
        //                  F_z = rhoE*rhouz/rho + p * rhouz/rho

        fluxes(elem,pt,rhoE_num,0) = 
          rhoE(elem,pt)*rhoux(elem,pt)/rho(elem,pt) 
          + props(elem,pt,p0_num)*rhoux(elem,pt)/rho(elem,pt);
        fluxes(elem,pt,rhoE_num,1) = 
          rhoE(elem,pt)*rhouy(elem,pt)/rho(elem,pt) 
          + props(elem,pt,p0_num)*rhouy(elem,pt)/rho(elem,pt); 
        fluxes(elem,pt,rhoE_num,2) = 
          rhoE(elem,pt)*rhouz(elem,pt)/rho(elem,pt) 
          + props(elem,pt,p0_num)*rhouz(elem,pt)/rho(elem,pt); 

      }

    });
  }

}

// ========================================================================================
// compute the thermodynamic properties
// ========================================================================================

void euler::computeThermoProps(const bool & on_side)
{

  // TODO :: getSolutionField("blah",false) for testing?

  Teuchos::TimeMonitor localtime(*thermoPropFill);

  auto props = on_side ? props_side : props_vol;
  // these are always needed
  auto rho = on_side ? wkset->getSolutionField("aux rho") : 
                       wkset->getSolutionField("rho");
  auto rhoux = on_side ? wkset->getSolutionField("aux rhoux") : 
                         wkset->getSolutionField("rhoux");
  auto rhoE = on_side ? wkset->getSolutionField("aux rhoE") : 
                        wkset->getSolutionField("rhoE");

  View_AD2 rhouy, rhouz; // TODO not sure this is the best way

  if ( spaceDim > 1 ) {
    rhouy = on_side ? wkset->getSolutionField("aux rhouy") : 
                      wkset->getSolutionField("rhouy");
  } 
  if ( spaceDim > 2 ) {
    rhouz = on_side ? wkset->getSolutionField("aux rhouz") : 
                      wkset->getSolutionField("rhouz");
  }

  parallel_for("euler thermo props",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {

    ScalarT gamma = modelparams(gamma_mp_num); 
    ScalarT MachNum = modelparams(MRef_mp_num);

    for (size_type pt=0; pt<props.extent(1); ++pt) {

      // TODO CHECK ME NON DIM
      // p0 = (gamma - 1)(rhoE - KE)
      // KE = .5 \rho \|u\|^2
      // T = gamma * M_\infty^2 p0/\rho
      // a_sound = sqrt(T) / M_\infty 

      // TODO SEE ABOVE !!

      props(elem,pt,p0_num) = (gamma - 1.)*(rhoE(elem,pt) - .5*rhoux(elem,pt)*rhoux(elem,pt)/rho(elem,pt));

      if (spaceDim > 1) {
        props(elem,pt,p0_num) += (gamma - 1.)*(-.5*rhouy(elem,pt)*rhouy(elem,pt)/rho(elem,pt));
      }
      if (spaceDim > 2) {
        props(elem,pt,p0_num) += (gamma - 1.)*(-.5*rhouz(elem,pt)*rhouz(elem,pt)/rho(elem,pt));
      }

      // pressure done

      props(elem,pt,T_num) = gamma*MachNum*MachNum*props(elem,pt,p0_num)/rho(elem,pt);
      props(elem,pt,a_num) = sqrt(props(elem,pt,T_num))/MachNum;

    }
  });

}

// ========================================================================================
// compute the stabilization term
// ========================================================================================

void euler::computeStabilizationTerm() {

  // The two proposed stabilization matrices in Peraire 2011 are based off of eigendecompositions
  // of the flux Jacobians
  // Let A = dF_x(\hat{S})/d\hat{S} n_x + dF_y(\hat{S})/d\hat{S} n_y + dF_z(\hat{S})d\hat{S} n_z
  // Then A = R Lambda L.
  //
  // The two options are Stab = R \| Lambda \| L or Stab = \lambda_max I
  // where \lambda_max is the maximum of \| Lambda \| (the absolute values of Lambda).
  //
  // Here we are interested in computing the result Stab \times (S - \hat{S})
  
  Teuchos::TimeMonitor localtime(*stabCompFill);

  using namespace std;
  
  // these are always needed
  auto rho = wkset->getSolutionField("rho");
  auto rho_hat = wkset->getSolutionField("aux rho");
  auto rhoux = wkset->getSolutionField("rhoux");
  auto rhoux_hat = wkset->getSolutionField("aux rhoux");
  auto rhoE = wkset->getSolutionField("rhoE");
  auto rhoE_hat = wkset->getSolutionField("aux rhoE");
  auto props = props_side; // get the properties evaluated with trace variables

  auto stabterm = stab_bound_side;
  auto nx = wkset->getScalarField("n[x]");

  View_AD2 rhouy, rhouy_hat, rhouz, rhouz_hat; // only assign if necessary
  View_Sc2 ny, nz;

  if (spaceDim > 1) {
    rhouy = wkset->getSolutionField("rhouy");
    rhouy_hat = wkset->getSolutionField("aux rhouy");
    ny = wkset->getScalarField("n[y]");
  }
  if (spaceDim > 2) {
    rhouz = wkset->getSolutionField("rhouz");
    rhouz_hat = wkset->getSolutionField("aux rhouz");
    nz = wkset->getScalarField("n[z]");
  }

  parallel_for("euler stabilization",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {

    View_AD2 leftEV,rightEV; // Local eigendecomposition
    View_AD1 Lambda; // diagonal matrix
    View_AD1 deltaS; // S - \hat{S} vector
    View_AD1 tmp; // temporary vector

    ScalarT gamma = modelparams(gamma_mp_num); 

    deltaS = View_AD1("delta S", spaceDim + 2);

    if (roestab) {
      Lambda = View_AD1("Lambda", spaceDim + 2); 
      leftEV = View_AD2("left EV", spaceDim + 2, spaceDim + 2); 
      rightEV = View_AD2("right EV", spaceDim + 2, spaceDim + 2); 
      tmp = View_AD1("tmp", spaceDim + 2);
    }
    
    for (size_type pt=0; pt<stabterm.extent(1); ++pt) {

      // get the appropriate portion of the stability term
      auto stab_sub = Kokkos::subview( stabterm, elem, pt, Kokkos::ALL());

      // form (S - \hat{S})
      deltaS(rho_num) = rho(elem,pt) - rho_hat(elem,pt);
      deltaS(rhoux_num) = rhoux(elem,pt) - rhoux_hat(elem,pt);
      deltaS(rhoE_num) = rhoE(elem,pt) - rhoE_hat(elem,pt);

      if (spaceDim > 1) {
        deltaS(rhouy_num) = rhouy(elem,pt) - rhouy_hat(elem,pt);
      }
      if (spaceDim > 2) {
        deltaS(rhouz_num) = rhouz(elem,pt) - rhouz_hat(elem,pt);
      }

      if (roestab) {
        // fill the stabilization matrices
        if (spaceDim == 1) {
          this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              rhoux_hat(elem,pt),rho_hat(elem,pt),props(elem,pt,a_num),gamma);
        } else if (spaceDim == 2) {
          this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              rhoux_hat(elem,pt),rhouy_hat(elem,pt),rho_hat(elem,pt),
              nx(elem,pt),ny(elem,pt),props(elem,pt,a_num),gamma);
        } else if (spaceDim == 3) {
           this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              rhoux_hat(elem,pt),rhouy_hat(elem,pt),rhouz_hat(elem,pt),rho_hat(elem,pt),
              nx(elem,pt),ny(elem,pt),nz(elem,pt),props(elem,pt,a_num),gamma);
        }

        this->matVec(leftEV,deltaS,tmp); // L deltaS --> tmp
        // hit with the absolute value of the diagonal matrix
        for (int i=0; i<spaceDim + 2; ++i) {
          tmp(i) *= abs( Lambda(i) );
        }
        // R tmp = R AbsLambda L deltaS --> stab_sub 
        this->matVec(rightEV,tmp,stab_sub);

      } else {
        // the stabilization is just the max abs EV times delta S
        AD vn = nx(elem,pt)*rhoux_hat(elem,pt)/rho_hat(elem,pt);
        if (spaceDim > 1) vn += ny(elem,pt)*rhouy_hat(elem,pt)/rho_hat(elem,pt);
        if (spaceDim > 2) vn += nz(elem,pt)*rhouz_hat(elem,pt)/rho_hat(elem,pt);
        // max of | vn + a |, | vn - a |
        AD lambdaMax = max(abs(vn + props(elem,pt,a_num)),abs(vn - props(elem,pt,a_num)));

        for (int i=0; i<spaceDim+2; ++i) {
          stab_sub(i) = deltaS(i) * lambdaMax;
        }
      }
    }
  });
}

// ========================================================================================
// compute the boundary flux
// ========================================================================================

void euler::computeBoundaryTerm() {

  // The two BC types in Peraire 2011 are inflow/outflow and slip.
  //
  // The inflow/outflow BC is based off of eigendecompositions
  // of the flux Jacobians
  // Let A = dF_x(\hat{S})/d\hat{S} n_x + dF_y(\hat{S})/d\hat{S} n_y + dF_z(\hat{S})d\hat{S} n_z
  // Then A = R Lambda L.
  //
  // B = A^+ ( S - \hat{S} ) - A^- ( S_\infty - \hat{S} )
  // A^{\pm} = ( A \pm | A | ) / 2
  // S_\infty is the freestream condition
  
  using namespace std;

  Teuchos::TimeMonitor localtime(*boundCompFill);

  auto bcs = wkset->var_bcs;

  int cside = wkset->currentside;
  // TODO since our BCs come in for the whole state, just need one... TODO CHECK CHECK GENERALIZE??
  string sidetype = bcs(rho_num,cside);

  if ( (sidetype != "Far-field") && (sidetype != "Slip") 
        && (sidetype != "interface") ) {
    cout << "Error :: Euler module does not support your chosen boundary condition!" << endl;
  }

  // these are always needed
  auto rho = wkset->getSolutionField("rho");
  auto rho_hat = wkset->getSolutionField("aux rho");
  auto rhoux = wkset->getSolutionField("rhoux");
  auto rhoux_hat = wkset->getSolutionField("aux rhoux");
  auto rhoE = wkset->getSolutionField("rhoE");
  auto rhoE_hat = wkset->getSolutionField("aux rhoE");
  auto props = props_side; // get the properties evaluated with trace variables

  auto boundterm = stab_bound_side;
  auto nx = wkset->getScalarField("n[x]");

  View_AD2 rhouy, rhouy_hat, rhouz, rhouz_hat; // and only assign if necessary?
  View_Sc2 ny, nz;

  if (spaceDim > 1) {
    rhouy = wkset->getSolutionField("rhouy");
    rhouy_hat = wkset->getSolutionField("aux rhouy");
    ny = wkset->getScalarField("n[y]");
  }
  if (spaceDim > 2) {
    rhouz = wkset->getSolutionField("rhouz");
    rhouz_hat = wkset->getSolutionField("aux rhouz");
    nz = wkset->getScalarField("n[z]");
  }

  // Get the freestream info if needed
  Vista source_rho, source_rhoux, source_rhoE, source_rhouy, source_rhouz;

  if (sidetype == "Far-field") { 
    source_rho = functionManager->evaluate("Far-field rho " + wkset->sidename,"side ip");
    source_rhoux = functionManager->evaluate("Far-field rhoux " + wkset->sidename,"side ip");
    source_rhoE = functionManager->evaluate("Far-field rhoE " + wkset->sidename,"side ip");
    if (spaceDim > 1) { 
      source_rhouy = functionManager->evaluate("Far-field rhouy " + wkset->sidename,"side ip");
    }
    if (spaceDim > 2) { 
      source_rhouz = functionManager->evaluate("Far-field rhouz " + wkset->sidename,"side ip");
    }
  }

  parallel_for("euler boundary term",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {

    View_AD2 leftEV,rightEV; // Local eigendecomposition
    View_AD1 Lambda; // diagonal matrix
    View_AD1 deltaS; // S - \hat{S} vector
    View_AD1 tmp; // temporary vector

    ScalarT gamma = modelparams(gamma_mp_num); 

    if (sidetype == "Far-field") {

      // allocate storage
      deltaS = View_AD1("delta S", spaceDim + 2);
      Lambda = View_AD1("Lambda", spaceDim + 2); 
      leftEV = View_AD2("left EV", spaceDim + 2, spaceDim + 2); 
      rightEV = View_AD2("right EV", spaceDim + 2, spaceDim + 2); 
      tmp = View_AD1("tmp", spaceDim + 2);
    }
    
    for (size_type pt=0; pt<boundterm.extent(1); ++pt) {

      // get the appropriate portion of the boundary term
      // TODO Not sure about FORCING this to be view_ad1 here (and above)
      auto bound_sub = Kokkos::subview( boundterm, elem, pt, Kokkos::ALL());
      
      if (sidetype == "Far-field") {
        // get the local eigendecomposition
        if (spaceDim == 1) {
          this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              rhoux_hat(elem,pt),rho_hat(elem,pt),props(elem,pt,a_num),gamma);
        } else if (spaceDim == 2) {
          this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              rhoux_hat(elem,pt),rhouy_hat(elem,pt),rho_hat(elem,pt),
              nx(elem,pt),ny(elem,pt),props(elem,pt,a_num),gamma);
        } else if (spaceDim == 3) {
           this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              rhoux_hat(elem,pt),rhouy_hat(elem,pt),rhouz_hat(elem,pt),rho_hat(elem,pt),
              nx(elem,pt),ny(elem,pt),nz(elem,pt),props(elem,pt,a_num),gamma);
        }

        // form (S - \hat{S})
        deltaS(rho_num) = rho(elem,pt) - rho_hat(elem,pt);
        deltaS(rhoux_num) = rhoux(elem,pt) - rhoux_hat(elem,pt);
        deltaS(rhoE_num) = rhoE(elem,pt) - rhoE_hat(elem,pt);

        if (spaceDim > 1) {
          deltaS(rhouy_num) = rhouy(elem,pt) - rhouy_hat(elem,pt);
        }
        if (spaceDim > 2) {
          deltaS(rhouz_num) = rhouz(elem,pt) - rhouz_hat(elem,pt);
        }

        // A^+ = R ( Lambda + | Lambda | ) L / 2
        // First, get A^+ \times delta S

        this->matVec(leftEV,deltaS,tmp); // L deltaS --> tmp
        // hit with the diagonal matrix
        for (int i=0; i<spaceDim + 2; ++i) {
          tmp(i) *= ( Lambda(i) + abs( Lambda(i) ) ) / 2.;
        }
        // R tmp = A^+ deltaS --> bound_sub 
        this->matVec(rightEV,tmp,bound_sub);

        // now form ( S_\infty - \hat{S} )

        deltaS(rho_num) = source_rho(elem,pt) - rho_hat(elem,pt);
        deltaS(rhoux_num) = source_rhoux(elem,pt) - rhoux_hat(elem,pt);
        deltaS(rhoE_num) = source_rhoE(elem,pt) - rhoE_hat(elem,pt);

        if (spaceDim > 1) {
          deltaS(rhouy_num) = source_rhouy(elem,pt) - rhouy_hat(elem,pt);
        }
        if (spaceDim > 2) {
          deltaS(rhouz_num) = source_rhouz(elem,pt) - rhouz_hat(elem,pt);
        }

        // A^- = R ( Lambda - | Lambda | ) L / 2
        // First, get A^- \times delta S

        this->matVec(leftEV,deltaS,tmp); // L deltaS --> tmp
        // hit with the diagonal matrix
        for (int i=0; i<spaceDim + 2; ++i) {
          tmp(i) *= ( Lambda(i) - abs( Lambda(i) ) ) / 2.;
        }
        // R tmp = A^- deltaS --> deltaS
        this->matVec(rightEV,tmp,deltaS);

        // finalize
        
        for (int i=0; i<spaceDim + 2; ++i) {
          bound_sub(i) -= deltaS(i);
        }

      } else {
        // Apply the slip condition
        AD vn = nx(elem,pt)*rhoux(elem,pt)/rho(elem,pt);
        if (spaceDim > 1) vn += ny(elem,pt)*rhouy(elem,pt)/rho(elem,pt);
        if (spaceDim > 2) vn += nz(elem,pt)*rhouz(elem,pt)/rho(elem,pt);

        // density and energy are matched
        
        bound_sub(rho_num) = rho(elem,pt) - rho_hat(elem,pt);
        bound_sub(rhoE_num) = rhoE(elem,pt) - rhoE_hat(elem,pt);

        // force normal velocity to be zero
        // TODO does this make sense? are units correct? This is from their paper CHECKME!

        bound_sub(rhoux_num) = 
          ( rhoux(elem,pt)/rho(elem,pt) - vn*nx(elem,pt) ) - rhoux_hat(elem,pt)/rho_hat(elem,pt);

        if (spaceDim > 1) {
          bound_sub(rhouy_num) = 
            ( rhouy(elem,pt)/rho(elem,pt) - vn*ny(elem,pt) ) - rhouy_hat(elem,pt)/rho_hat(elem,pt);
        }
        if (spaceDim > 2) {
          bound_sub(rhouz_num) = 
            ( rhouz(elem,pt)/rho(elem,pt) - vn*nz(elem,pt) ) - rhouz_hat(elem,pt)/rho_hat(elem,pt);
        }
      }
    }
  });
}

// ========================================================================================
// Fill in the local eigendecomposition matrices
// ========================================================================================

KOKKOS_FUNCTION void euler::eigendecompFluxJacobian(View_AD2 leftEV, View_AD1 Lambda, View_AD2 rightEV, 
        const AD & rhoux, const AD & rho, const AD & a_sound, const ScalarT & gamma) {

  // In 1D, the eigenvalues are ux - a, ux, and ux + a
  // The right eigenvectors are 
  // [1, ux - a, a^2/(gamma - 1) + .5*ux**2 - ux*a]^T
  // [1, ux, 1/2 ux**2]^T
  // [1, ux + a, a^2/(gamma - 1) + .5*ux**2 + ux*a]^T
  //
  // The corresponding left eigenvectors are TODO CHECK
  // [(gamma-1)/4*ux^2/a^2 + ux/(2*a), -(gamma-1)/a^3*ux - 1/(2*a), (gamma-1)/(2*a^2)]
  // [1 - (gamma-1)/2*ux^2/a^2, (gamma-1)*ux/a^2, -(gamma-1)/a^2]
  // [(gamma-1)/4*ux^2/a^2 - ux/(2*a), -(gamma-1)/a^3*ux + 1/(2*a), (gamma-1)/(2*a^2)]

  // TODO CHECK BELOW 
  
  const ScalarT gm1 = gamma - 1.;

  rightEV(0,0) = 1.; rightEV(1,0) = rhoux/rho - a_sound; 
  rightEV(2,0) = a_sound*a_sound/gm1 + .5*rhoux*rhoux/(rho*rho) - rhoux/rho*a_sound;

  rightEV(0,1) = 1.; rightEV(1,1) = rhoux/rho; rightEV(2,1) = .5*rhoux*rhoux/(rho*rho);

  rightEV(0,2) = 1.; rightEV(1,2) = rhoux/rho + a_sound; 
  rightEV(2,2) = a_sound*a_sound/gm1 + .5*rhoux*rhoux/(rho*rho) + rhoux/rho*a_sound;

  leftEV(0,0) = gm1/4.*rhoux*rhoux/(rho*rho)/(a_sound*a_sound) + rhoux/rho/(2.*a_sound);
  leftEV(0,1) = -gm1/(a_sound*a_sound*a_sound)*rhoux/rho - 1./(2.*a_sound);
  leftEV(0,2) = gm1/(2.*a_sound*a_sound);

  leftEV(1,0) = 1. - gm1/2.*rhoux*rhoux/(rho*rho)/(a_sound*a_sound);
  leftEV(1,1) = gm1*rhoux/rho/(a_sound*a_sound);
  leftEV(1,2) = -gm1/(a_sound*a_sound);

  leftEV(2,0) = gm1/4.*rhoux*rhoux/(rho*rho)/(a_sound*a_sound) - rhoux/rho/(2.*a_sound);
  leftEV(2,1) = -gm1/(a_sound*a_sound*a_sound)*rhoux/rho + 1./(2.*a_sound);
  leftEV(2,2) = gm1/(2.*a_sound*a_sound);

  Lambda(0) = rhoux/rho - a_sound; Lambda(1) = rhoux/rho; Lambda(2) = rhoux/rho + a_sound; 
  
}

KOKKOS_FUNCTION void euler::eigendecompFluxJacobian(View_AD2 leftEV, View_AD1 Lambda, View_AD2 rightEV, 
    const AD & rhoux, const AD & rhouy, const AD & rho, const ScalarT & nx, const ScalarT & ny,
    const AD & a_sound, const ScalarT & gamma) {

  // This follows Rohde 2001 (AIAA)

  AD vn = rhoux/rho*nx + rhouy/rho*ny;
  AD ek_m = .5 * (rhoux*rhoux + rhouy*rhouy)/(rho*rho);
  ScalarT gm1 = gamma - 1.;

  // TODO CHECK BELOW 

  // Equation 11 with the z parts removed/truncated gives the right EVs as
  // [1, ux - a*nx, uy - a*ny, a^2/(gamma-1) + ek_m - vn*a]^T
  // [1, ux, uy, ek_m]^T
  // [1, ux + a*nx, uv + a*ny, a^2/(gamma-1) + ek_m + vn*a]^T
  // [0, ny, -nx, ux*ny - uy*nx]^T

  // Equation 16 with the z parts removed/truncated gives the left EVs as
  // 1/(2*a^2) [ ( (gamma-1)*ek_m + a*vn ), (1 - gamma)*ux - a*nx, (1 - gamma)*uy - a*ny, (gamma - 1) ]
  // 1/a^2 [ a^2 - (gamma-1)*ek_m, (gamma-1)*ux, (gamma-1)*uy, (1 - gamma) ]
  // 1/(2*a^2) [ ( (gamma-1)*ek_m - a*vn ), (1 - gamma)*ux + a*nx, (1 - gamma)*uy + a*ny, (gamma - 1) ]
  // [uy*nx-ux*ny, ny, -nx , 0 ]
  // 
  // where equations 23 and 24 have been used appropriately to remove the potential singularity if nx = 0

  rightEV(0,0) = 1.; rightEV(1,0) = rhoux/rho - a_sound*nx; rightEV(2,0) = rhouy/rho - a_sound*ny;
  rightEV(3,0) = a_sound*a_sound/gm1 + ek_m - vn*a_sound;

  rightEV(0,1) = 1.; rightEV(1,1) = rhoux/rho; rightEV(2,1) = rhouy/rho; rightEV(3,1) = ek_m;

  rightEV(0,2) = 1.; rightEV(1,2) = rhoux/rho + a_sound*nx; rightEV(2,2) = rhouy/rho + a_sound*ny;
  rightEV(3,2) = a_sound*a_sound/gm1 + ek_m + vn*a_sound;

  rightEV(0,3) = 0.; rightEV(1,3) = ny; rightEV(2,3) = -nx; rightEV(3,3) = rhoux/rho*ny - rhouy/rho*nx;

  leftEV(0,0) = 1./(2.*a_sound*a_sound) * (gm1*ek_m + a_sound*vn);
  leftEV(0,1) = 1./(2.*a_sound*a_sound) * (-gm1*rhoux/rho - a_sound*nx);
  leftEV(0,2) = 1./(2.*a_sound*a_sound) * (-gm1*rhouy/rho - a_sound*ny);
  leftEV(0,3) = 1./(2.*a_sound*a_sound) * gm1;

  leftEV(1,0) = 1./(a_sound*a_sound) * (a_sound*a_sound - gm1*ek_m);
  leftEV(1,1) = 1./(a_sound*a_sound) * (gm1*rhoux/rho);
  leftEV(1,2) = 1./(a_sound*a_sound) * (gm1*rhouy/rho);
  leftEV(1,3) = 1./(a_sound*a_sound) * (-gm1);

  leftEV(2,0) = 1./(2.*a_sound*a_sound) * (gm1*ek_m - a_sound*vn);
  leftEV(2,1) = 1./(2.*a_sound*a_sound) * (-gm1*rhoux/rho + a_sound*nx);
  leftEV(2,2) = 1./(2.*a_sound*a_sound) * (-gm1*rhouy/rho + a_sound*ny);
  leftEV(2,3) = 1./(2.*a_sound*a_sound) * gm1;

  leftEV(3,0) = rhouy/rho*nx - rhoux/rho*ny; leftEV(3,1) = ny; leftEV(3,2) = -nx; leftEV(3,3) = 0.;

  Lambda(0) = vn - a_sound; Lambda(1) = vn; Lambda(2) = vn + a_sound; Lambda(3) = vn;

}

KOKKOS_FUNCTION void euler::eigendecompFluxJacobian(View_AD2 leftEV, View_AD1 Lambda, View_AD2 rightEV, 
    const AD & rhoux, const AD & rhouy, const AD & rhouz, const AD & rho, 
    const ScalarT & nx, const ScalarT & ny, const ScalarT & nz,
    const AD & a_sound, const ScalarT & gamma) {

  // This follows Rohde 2001 (AIAA)

  using namespace std;

  AD vn = rhoux/rho*nx + rhouy/rho*ny + rhouz/rho*nz;
  AD ek_m = .5 * (rhoux*rhoux + rhouy*rhouy + rhouz*rhouz)/(rho*rho);
  ScalarT gm1 = gamma - 1.;

  // Rohde gives three sets of right/left EV pairs.
  // For our purposes, we choose which pair to use based on largest (magnitude) component of n.
  // This will avoid the matrices from becoming singular 
  
  int whichMat = 1;
  
  if ( ( abs(ny) > abs(nx) || abs(nz) > abs(nx) ) ) {
    
    // TODO CHECK ME

    // nx out of the running
    // need a default in case they are equal

    whichMat = 2;
    if ( abs(nz) > abs(ny) ) whichMat = 3;
  }

  // The eigenvalues are the same so let's nab those first
  
  Lambda(0) = vn - a_sound; Lambda(1) = vn; Lambda(2) = vn + a_sound;
  Lambda(3) = vn; Lambda(4) = vn;

  // TODO CHECK BELOW 

  // For nx biggest....
  // Equation 11 gives
  // [1, ux - a*nx, uy - a*ny, uz - a*nz, a^2/(gamma-1) + ek_m - vn*a]^T
  // [1, ux, uy, uz, ek_m]^T
  // [1, ux + a*nx, uv + a*ny, uz + a*nz, a^2/(gamma-1) + ek_m + vn*a]^T
  // [0, ny, -nx, 0, ux*ny - uy*nx]^T
  // [0, -nz, 0, nx, uz*nx - ux*nz]^T

  // The left EVs are (equation 16)
  // 1/(2*a^2) [ ( (gamma-1)*ek_m + a*vn ), (1 - gamma)*ux - a*nx, (1 - gamma)*uy - a*ny, (1 - gamma)*uz - a*nz, (gamma - 1) ]
  // 1/a^2 [ a^2 - (gamma-1)*ek_m, (gamma-1)*ux, (gamma-1)*uy, (gamma-1)*uz, (1 - gamma) ]
  // 1/(2*a^2) [ ( (gamma-1)*ek_m - a*vn ), (1 - gamma)*ux + a*nx, (1 - gamma)*uy + a*ny, (1 - gamma)*uz + a*nz, (gamma - 1) ]
  // [ (uy - vn*ny)/nx, ny, (ny^2-1)/nx, ny*nz/nx, 0 ]
  // [ (vn*nz - uz)/nx, -nz, -ny*nz/nx, (1-nz^2)/nx, 0 ]

  // For ny biggest....
  // Equation 13 gives
  // [1, ux - a*nx, uy - a*ny, uz - a*nz, a^2/(gamma-1) + ek_m - vn*a]^T
  // [1, ux, uy, uz, ek_m]^T
  // [1, ux + a*nx, uv + a*ny, uz + a*nz, a^2/(gamma-1) + ek_m + vn*a]^T
  // [0, ny, -nx, 0, ux*ny - uy*nx]^T
  // [0, 0, nz, -ny, uy*nz - uz*ny]^T

  // The left EVs are (equation 18)
  // 1/(2*a^2) [ ( (gamma-1)*ek_m + a*vn ), (1 - gamma)*ux - a*nx, (1 - gamma)*uy - a*ny, (1 - gamma)*uz - a*nz, (gamma - 1) ]
  // 1/a^2 [ a^2 - (gamma-1)*ek_m, (gamma-1)*ux, (gamma-1)*uy, (gamma-1)*uz, (1 - gamma) ]
  // 1/(2*a^2) [ ( (gamma-1)*ek_m - a*vn ), (1 - gamma)*ux + a*nx, (1 - gamma)*uy + a*ny, (1 - gamma)*uz + a*nz, (gamma - 1) ]
  // [ (vn*nx - ux)/ny, (1-nx^2)/ny, -nx, -nx*nz/ny, 0 ]
  // [ (uz - vn*nz)/ny, nx*nz/ny, nz, (nz^2-1)/ny, 0 ]

  // For nz biggest....
  // Equation 14 gives
  // [1, ux - a*nx, uy - a*ny, uz - a*nz, a^2/(gamma-1) + ek_m - vn*a]^T
  // [1, ux, uy, uz, ek_m]^T
  // [1, ux + a*nx, uv + a*ny, uz + a*nz, a^2/(gamma-1) + ek_m + vn*a]^T
  // [0, -nz, 0, nx, uz*nx - ux*nz]^T
  // [0, 0, nz, -ny, uy*nz - uz*ny]^T

  // The left EVs are (equation 19)
  // 1/(2*a^2) [ ( (gamma-1)*ek_m + a*vn ), (1 - gamma)*ux - a*nx, (1 - gamma)*uy - a*ny, (1 - gamma)*uz - a*nz, (gamma - 1) ]
  // 1/a^2 [ a^2 - (gamma-1)*ek_m, (gamma-1)*ux, (gamma-1)*uy, (gamma-1)*uz, (1 - gamma) ]
  // 1/(2*a^2) [ ( (gamma-1)*ek_m - a*vn ), (1 - gamma)*ux + a*nx, (1 - gamma)*uy + a*ny, (1 - gamma)*uz + a*nz, (gamma - 1) ]
  // [ (ux-vn*nx)/nz, (nx^2-1)/nz, nx*ny/nz, nx, 0 ]
  // [ (vn*ny - uy)/nz, -nx*ny/nz, (1-ny^2)/nz, -ny, 0 ]

  // We start by filling in the first three columns/rows which are the same for each pair

  rightEV(0,0) = 1.; rightEV(1,0) = rhoux/rho - a_sound*nx;
  rightEV(2,0) = rhouy/rho - a_sound*ny; rightEV(3,0) = rhouz/rho - a_sound*nz;
  rightEV(4,0) = a_sound*a_sound/gm1 + ek_m - vn*a_sound;

  rightEV(0,1) = 1.; rightEV(1,1) = rhoux/rho; rightEV(2,1) = rhouy/rho; 
  rightEV(3,1) = rhouz/rho; rightEV(4,1) = ek_m;

  rightEV(0,2) = 1.; rightEV(1,2) = rhoux/rho + a_sound*nx; 
  rightEV(2,2) = rhouy/rho + a_sound*ny; rightEV(3,2) = rhouz/rho + a_sound*nz;
  rightEV(4,2) = a_sound*a_sound/gm1 + ek_m + vn*a_sound;

  leftEV(0,0) = 1./(2.*a_sound*a_sound) * (gm1*ek_m + a_sound*vn);
  leftEV(0,1) = 1./(2.*a_sound*a_sound) * (-gm1*rhoux/rho - a_sound*nx);
  leftEV(0,2) = 1./(2.*a_sound*a_sound) * (-gm1*rhouy/rho - a_sound*ny);
  leftEV(0,3) = 1./(2.*a_sound*a_sound) * (-gm1*rhouz/rho - a_sound*nz);
  leftEV(0,4) = 1./(2.*a_sound*a_sound) * gm1;

  leftEV(1,0) = 1./(a_sound*a_sound) * (a_sound*a_sound - gm1*ek_m);
  leftEV(1,1) = 1./(a_sound*a_sound) * (gm1*rhoux/rho);
  leftEV(1,2) = 1./(a_sound*a_sound) * (gm1*rhouy/rho);
  leftEV(1,3) = 1./(a_sound*a_sound) * (gm1*rhouz/rho);
  leftEV(1,4) = 1./(a_sound*a_sound) * (-gm1);

  leftEV(2,0) = 1./(2.*a_sound*a_sound) * (gm1*ek_m - a_sound*vn);
  leftEV(2,1) = 1./(2.*a_sound*a_sound) * (-gm1*rhoux/rho + a_sound*nx);
  leftEV(2,2) = 1./(2.*a_sound*a_sound) * (-gm1*rhouy/rho + a_sound*ny);
  leftEV(2,3) = 1./(2.*a_sound*a_sound) * (-gm1*rhouz/rho + a_sound*nz);
  leftEV(2,4) = 1./(2.*a_sound*a_sound) * gm1;

  // now handle the last two columns/rows on a case-by-case basis

  if (whichMat == 1) {

    rightEV(0,3) = 0.; rightEV(1,3) = ny; rightEV(2,3) = -nx; rightEV(3,3) = 0.;
    rightEV(4,3) = rhoux/rho*ny - rhouy/rho*nx;

    rightEV(0,4) = 0; rightEV(1,4) = -nz; rightEV(2,4) = 0.; rightEV(3,4) = nx;
    rightEV(4,4) = rhouz/rho*nx - rhoux/rho*nz;

    leftEV(3,0) = (rhouy/rho - vn*ny)/nx; leftEV(3,1) = ny; leftEV(3,2) = (ny*ny-1.)/nx;
    leftEV(3,3) = ny*nz/nx; leftEV(3,4) = 0.;

    leftEV(4,0) = (vn*nz - rhouz/rho)/nx; leftEV(4,1) = -nz; leftEV(4,2) = -ny*nz/nx;
    leftEV(4,3) = (1.-nz*nz)/nx; leftEV(4,4) = 0.;

  } else if (whichMat == 2) {

    rightEV(0,3) = 0.; rightEV(1,3) = ny; rightEV(2,3) = -nx; rightEV(3,3) = 0.;
    rightEV(4,3) = rhoux/rho*ny - rhouy/rho*nx;

    rightEV(0,4) = 0; rightEV(1,4) = 0.; rightEV(2,4) = nz; rightEV(3,4) = -ny;
    rightEV(4,4) = rhouy/rho*nz - rhouz/rho*ny;

    leftEV(3,0) = (vn*nx - rhoux/rho)/ny; leftEV(3,1) = (1.-nx*nx)/ny; leftEV(3,2) = -nx;
    leftEV(3,3) = -nx*nz/ny; leftEV(3,4) = 0.;

    leftEV(4,0) = (rhouz/rho - vn*nz)/ny; leftEV(4,1) = nx*nz/ny; leftEV(4,2) = nz;
    leftEV(4,3) = (nz*nz-1.)/ny; leftEV(4,4) = 0.;

  } else if (whichMat == 3) {

    rightEV(0,3) = 0.; rightEV(1,3) = -nz; rightEV(2,3) = 0.; rightEV(3,3) = nx;
    rightEV(4,3) = rhouz/rho*nx - rhoux/rho*nz;

    rightEV(0,4) = 0; rightEV(1,4) = 0.; rightEV(2,4) = nz; rightEV(3,4) = -ny;
    rightEV(4,4) = rhouy/rho*nz - rhouz/rho*ny;

    leftEV(3,0) = (rhoux/rho - vn*nx)/nz; leftEV(3,1) = (nx*nx-1.)/nz; leftEV(3,2) = nx*ny/nz;
    leftEV(3,3) = nx; leftEV(3,4) = 0.;

    leftEV(4,0) = (vn*ny - rhouy/rho)/nz; leftEV(4,1) = -nx*ny/nz; leftEV(4,2) = (1.-ny*ny)/nz;
    leftEV(4,3) = -ny; leftEV(4,4) = 0.;

  }

}

// ========================================================================================
// Fill in the local normal flux Jacobian
// ========================================================================================

//KOKKOS_FUNCTION void updateNormalFluxJacobian(View_AD2 & dFdn, const AD & rhoux,
//    const AD & rho, const & AD a_sound, const & ScalarT gamma) {
//
//  const ScalarT gm1 = gamma - 1.;
//
//  // TODO CHECK
//
//  // In 1-D, the flux Jacobian matrix is
//  // [ 0, 1, 0 ]
//  // [ .5*gm1*ux^2, (3 - gamma)*ux, gm1 ]
//  // [ ux*( .5*gm1*ux^2 - a^2/gm1 - .5*ux^2), a^2/gm1 + .5*ux^2 - gm1*ux^2, gamma*ux ]
//
//  dFdn(0,0) = 0.; dFdn(0,1) = 1.; dFdn(0,2) = 0.;
//  
//  dFdn(1,0) = .5*gm1*rhoux*rhoux/(rho*rho); dFdn(1,1) = (3. - gamma)*rhoux/rho;
//  dFdn(1,2) = gm1;
//
//  dFdn(2,0) = rhoux/rho*( .5*(gamma-2.)*rhoux*rhoux/(rho*rho) - a_sound*a_sound/gm1 );
//  dFdn(2,1) = a_sound*a_sound/gm1 + (3./2.-gamma)*rhoux*rhoux/(rho*rho);
//  dFdn(2,2) = gamma*rhoux/rho;
//
//}
//
//KOKKOS_FUNCTION void updateNormalFluxJacobian(View_AD2 & dFdn, const AD & rhoux,
//    const & AD rhouy, const AD & rho, const AD & nx, const AD & ny, 
//    const & AD a_sound, const & ScalarT gamma) {
//
//  // This follows Rohde 2001 (AIAA)
//
//  AD vn = rhoux/rho*nx + rhouy/rho*ny;
//  AD ek_m = .5 * (rhoux*rhoux + rhouy*rhouy)/(rho*rho);
//  ScalarT gm1 = gamma - 1.;
//
//  // In 2-D, the flux Jacobian matrix is 
//  // [ 0, nx, ny, 0 ]
//  // [ gm1*ek_m*nx - ux*vn, vn - (gamma-2)*ux*nx, ux*ny - gm1*uy*nx, gm1*nx ]
//  // [ gm1*ek_m*ny - uy*vn, uy*nx - gm1*ux*ny, vn - (gamma-2)*uy*ny, gm1*ny ]
//  // [ vn*(gm1*ek_m - a^2/gm1 - ek_m), 
//  //  (a^2/gm1 + ek_m)*nx - gm1*ux*vn, (a^2/gm1 + ek_m)*ny - gm1*uy*vn, gamma*vn ]
//
//  dFdn(0,0) = 0.; dFdn(0,1) = nx; dFdn(0,2) = ny; dFdn(0,3) = 0.;
//
//  dFdn(1,0) = gm1*ek_m*nx - rhoux/rho*vn; dFdn(1,1) = vn - (gamma-2.)*rhoux/rho*nx;
//  dFdn(1,2) = rhoux/rho*ny - gm1*rhouy/rho*nx; dFdn(1,3) = gm1*nx;
//
//  dFdn(2,0) = gm1*ek_m*ny - rhouy/rho*vn; dFdn(2,1) = rhouy/rho*nx - gm1*rhoux/rho*ny;
//  dFdn(2,2) = vn - (gamma-2.)*rhouy/rho*ny; dFdn(2,3) = gm1*ny;
//
//  dFdn(3,0) = vn*(gm1*ek_m - a_sound*a_sound/gm1 - ek_m);
//  dFdn(3,1) = (a_sound*a_sound/gm1 + ek_m)*nx - gm1*rhoux/rho*vn;
//  dFdn(3,2) = (a_sound*a_sound/gm1 + ek_m)*ny - gm1*rhouy/rho*vn;
//  dFdn(3,3) = gamma*vn;
//
//}
//
//KOKKOS_FUNCTION void updateNormalFluxJacobian(View_AD2 & dFdn, const AD & rhoux,
//    const & AD rhouy, const & AD rhouz const AD & rho, 
//    const AD & nx, const AD & ny, const AD & nz,
//    const & AD a_sound, const & ScalarT gamma) {
//
//  // TODO NOT NEEDED?
//
//  // This follows Rohde 2001 (AIAA)
//
//  AD vn = rhoux/rho*nx + rhouy/rho*ny + rhouz/rho*nz;
//  AD ek_m = .5 * (rhoux*rhoux + rhouy*rhouy + rhouz*rhouz)/(rho*rho);
//  ScalarT gm1 = gamma - 1.;
//
//  // In 3-D, the flux Jacobian matrix is 
//  // [ 0, nx, ny, nz, 0 ]
//  // [ gm1*ek_m*nx - ux*vn, vn - (gamma-2)*ux*nx, ux*ny - gm1*uy*nx, ux*nz - gm1*uz*nx, gm1*nx ]
//  // [ gm1*ek_m*ny - uy*vn, uy*nx - gm1*ux*ny, vn - (gamma-2)*uy*ny, uy*nz - gm1*uz*ny, gm1*ny ]
//  // [ gm1*ek_m*nz - uz*vn, uz*nx - gm1*ux*nz, uz*ny - gm1*uy*nz, vn - (gamma-2)*uz*nz, gm1*nz ]
//  // [ vn*(gm1*ek_m - a^2/gm1 - ek_m), 
//  //  (a^2/gm1 + ek_m)*nx - gm1*ux*vn, (a^2/gm1 + ek_m)*ny - gm1*uy*vn, (a^2/gm1 + ek_m)*nz - gm1*uz*vn,
//  //  gamma*vn ]
//
//  dFdn(0,0) = 0.; dFdn(0,1) = nx; dFdn(0,2) = ny; dFdn(0,3) = nz; dFdn(0,4) = 0.;
//
//  dFdn(1,0) = gm1*ek_m*nx - rhoux/rho*vn; dFdn(1,1) = vn - (gamma-2.)*rhoux/rho*nx;
//  dFdn(1,2) = rhoux/rho*ny - gm1*rhouy/rho*nx; dFdn(1,3) = rhoux/rho*nz - gm1*rhouz/rho*nx; 
//  dFdn(1,4) = gm1*nx;
//
//  dFdn(2,0) = gm1*ek_m*ny - rhouy/rho*vn; dFdn(2,1) = rhouy/rho*nx - gm1*rhoux/rho*ny;
//  dFdn(2,2) = vn - (gamma-2.)*rhouy/rho*ny; dFdn(2,3) = rhouyz/rho*nz - gm1*rhouz/rho*ny;
//  dFdn(2,4) = gm1*ny;
//
//  dFdn(3,0) = gm1*ek_m*nz - rhouz/rho*vn; dFdn(3,1) = rhouz/rho*nx - gm1*rhoux/rho*nz; 
//  dFdn(3,2) = rhouz/rho*ny - gm1*rhouy/rho*nz; dFdn(3,3) = vn - (gamma-2.)*rhouz/rho*nz; 
//  dFdn(3,4) = gm1*nx;
//
//  dFdn(4,0) = vn*(gm1*ek_m - a_sound*a_sound/gm1 - ek_m);
//  dFdn(4,1) = (a_sound*a_sound/gm1 + ek_m)*nx - gm1*rhoux/rho*vn;
//  dFdn(4,2) = (a_sound*a_sound/gm1 + ek_m)*ny - gm1*rhouy/rho*vn;
//  dFdn(4,3) = (a_sound*a_sound/gm1 + ek_m)*nz - gm1*rhouz/rho*vn;
//  dFdn(4,4) = gamma*vn;
//
//}