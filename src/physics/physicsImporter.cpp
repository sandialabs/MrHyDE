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

#include "physicsImporter.hpp"

// Enabled physics modules:
#include "porous.hpp"
#include "porousHDIV.hpp"
#include "porousHDIV_hybridized.hpp"
#include "porousHDIV_weakGalerkin.hpp"
#include "cdr.hpp"
#include "thermal.hpp"
#include "msphasefield.hpp"
#include "stokes.hpp"
#include "navierstokes.hpp"
#include "linearelasticity.hpp"
#include "helmholtz.hpp"
#include "maxwells_fp.hpp"
#include "shallowwater.hpp"
#include "maxwell.hpp"
#include "ode.hpp"

// Disabled/out-of-date physics modules
//#include "maxwell_hybridized.hpp"
//#include "thermal_enthalpy.hpp"
//#include "twophasePoNo.hpp"
//#include "twophasePoPw.hpp"
//#include "msconvdiff.hpp"
//#include "phasesolidification.hpp"
//#include "mwhelmholtz.hpp"
//#include "peridynamics.hpp"
//#include "euler.hpp"
//#include "burgers.hpp"
//#include "phasefield.hpp"
//#include "thermal_fr.hpp"

using namespace MrHyDE;

vector<Teuchos::RCP<physicsbase> > physicsImporter::import(vector<string> & module_list,
                                                           Teuchos::RCP<Teuchos::ParameterList> & settings,
                                                           const bool & isaux,
                                                           Teuchos::RCP<MpiComm> & Commptr) {
  
  vector<Teuchos::RCP<physicsbase> > modules;
  
  for (size_t mod=0; mod<module_list.size(); mod++) {
    string modname = module_list[mod];
    
    // Porous media (single phase slightly compressible)
    if (modname == "porous") {
      modules.push_back(Teuchos::rcp(new porous(settings, isaux) ) );
    }
    
    // Porous media with HDIV basis
    if (modname == "porousHDIV") {
      modules.push_back(Teuchos::rcp(new porousHDIV(settings, isaux) ) );
    }
    
    // Hybridized porous media with HDIV basis
    if (modname == "porousHDIV_hybrid") {
      modules.push_back(Teuchos::rcp(new porousHDIV_HYBRID(settings, isaux) ) );
    }
    
    // weak Galerkin porous media with HDIV basis
    if (modname == "porousHDIV_weakGalerkin") {
      modules.push_back(Teuchos::rcp(new porousHDIV_WG(settings, isaux) ) );
    }
    
    /*
     // Two phase porous media
     if (modname == "twophase") {
     string formulation = blockPhysSettings[b].get<string>("formulation","PoNo");
     if (formulation == "PoPw"){
     //modules.push_back(Teuchos::rcp(new twophasePoPw(settings) ) );
     }
     else if (formulation == "PoNo"){
     modules.push_back(Teuchos::rcp(new twophasePoNo(settings) ) );
     }
     else if (formulation == "PoPw"){
     modules.push_back(Teuchos::rcp(new twophasePoPw(settings) ) );
     }
     
     }
     */
    
    // Convection diffusion
    if (modname == "cdr" || modname == "CDR") {
      modules.push_back(Teuchos::rcp(new cdr(settings, isaux) ) );
    }
    
    // Thermal
    if (modname == "thermal") {
      modules.push_back(Teuchos::rcp(new thermal(settings, isaux) ) );
    }
    
    /*
     // Thermal with enthalpy variable
     if (modname == "thermal enthalpy") {
     modules.push_back(Teuchos::rcp(new thermal_enthalpy(settings) ) );
     }
     */
    
    // Shallow Water
    if (modname == "shallow water") {
      modules.push_back(Teuchos::rcp(new shallowwater(settings, isaux) ) );
    }
    
    // Maxwell
    if (modname == "maxwell") {
      modules.push_back(Teuchos::rcp(new maxwell(settings, isaux) ) );
    }
    
    /*
     // Maxwell hybridized
     if (modname == "maxwell hybrid") {
     modules.push_back(Teuchos::rcp(new maxwell_HYBRID(settings, isaux) ) );
     }
     */
    
    // Multiple Species PhaseField
    if (modname == "msphasefield") {
      modules.push_back(Teuchos::rcp(new msphasefield(settings, isaux, Commptr) ) );
    }
    
    // Stokes
    if (modname == "stokes" || modname == "Stokes") {
      modules.push_back(Teuchos::rcp(new stokes(settings, isaux) ) );
    }
    
    // Navier Stokes
    if (modname == "navier stokes" || modname == "Navier Stokes") {
      modules.push_back(Teuchos::rcp(new navierstokes(settings, isaux) ) );
    }
    
    // Linear Elasticity
    if (modname == "linearelasticity" || modname == "linear elasticity") {
      modules.push_back(Teuchos::rcp(new linearelasticity(settings, isaux) ) );
    }
    
    // Helmholtz
    if (modname == "helmholtz") {
      modules.push_back(Teuchos::rcp(new helmholtz(settings, isaux) ) );
    }
    
    // Maxwell's (potential of electric field, curl-curl frequency domain (Boyse et al (1992))
    if (modname == "maxwells_freq_pot"){
      modules.push_back(Teuchos::rcp(new maxwells_fp(settings, isaux) ) );
    }
    
    // Scalar ODE for testing time integrators independent of spatial discretizations
    if (modname == "ODE"){
      modules.push_back(Teuchos::rcp(new ODE(settings, isaux) ) );
    }
    
  }
  
  return modules;
  
}
