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
#include "physics_test.hpp"
#include "porous.hpp"
#include "porousMixed.hpp"
#include "porousMixedHybridized.hpp"
#include "porousWeakGalerkin.hpp"
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
#include "mhd.hpp"
#include "ode.hpp"
#include "burgers.hpp"
#include "kuramotoSivashinsky.hpp"
#include "llamas.hpp"
#include "variableDensityNS.hpp"
#include "euler.hpp"
#include "shallowwaterHybridized.hpp"
#include "incompressibleSaturation.hpp"
#include "shallowice.hpp"

#if defined(MrHyDE_ENABLE_MIRAGE)
#include "mirage.hpp"
#endif

//#include "cns.hpp"

using namespace MrHyDE;

vector<Teuchos::RCP<physicsbase> > physicsImporter::import(vector<string> & module_list,
                                                           Teuchos::ParameterList & settings,
                                                           const int & dimension,
                                                           Teuchos::RCP<MpiComm> & Commptr) {
  
  vector<Teuchos::RCP<physicsbase> > modules;
  
  for (size_t mod=0; mod<module_list.size(); mod++) {
    string modname = module_list[mod];
    
    // Test module which procedurally assembles and dumps basis functions based on parameterlist settings
    if (modname == "physicsTest") {
      modules.push_back(Teuchos::rcp(new physicsTest(settings, dimension) ) );
    }

    // Porous media (single phase slightly compressible)
    if (modname == "porous") {
      modules.push_back(Teuchos::rcp(new porous(settings, dimension) ) );
    }
    
    // Porous media with HDIV basis
    if (modname == "porous mixed") {
      modules.push_back(Teuchos::rcp(new porousMixed(settings, dimension) ) );
    }
    
    // Hybridized porous media with HDIV basis
    if (modname == "porous mixed hybridized") {
      modules.push_back(Teuchos::rcp(new porousMixedHybrid(settings, dimension) ) );
    }
    
    // weak Galerkin porous media with HDIV basis
    if (modname == "porous weak Galerkin") {
      modules.push_back(Teuchos::rcp(new porousWeakGalerkin(settings, dimension) ) );
    }
        
    // Convection diffusion
    if (modname == "cdr" || modname == "CDR") {
      modules.push_back(Teuchos::rcp(new cdr(settings, dimension) ) );
    }
    
    // Thermal
    if (modname == "thermal") {
      modules.push_back(Teuchos::rcp(new thermal(settings, dimension) ) );
    }
    
    // Shallow Water
    if (modname == "shallow water") {
      modules.push_back(Teuchos::rcp(new shallowwater(settings, dimension) ) );
    }

    // Shallow Ice
    if (modname == "shallow ice") {
      modules.push_back(Teuchos::rcp(new shallowice(settings, dimension) ) );
    }

    // Shallow water hybridized
    if (modname == "shallow water hybridized") {
      modules.push_back(Teuchos::rcp(new shallowwaterHybridized(settings, dimension) ) );
    }
    
    // Maxwell
    if (modname == "maxwell") {
      modules.push_back(Teuchos::rcp(new maxwell(settings, dimension) ) );
    }
    
    // Multiple Species PhaseField
    if (modname == "msphasefield") {
      modules.push_back(Teuchos::rcp(new msphasefield(settings, dimension, Commptr) ) );
    }
    
    // Stokes
    if (modname == "stokes" || modname == "Stokes") {
      modules.push_back(Teuchos::rcp(new stokes(settings, dimension) ) );
    }
    
    // Navier Stokes
    if (modname == "navier stokes" || modname == "Navier Stokes") {
      modules.push_back(Teuchos::rcp(new navierstokes(settings, dimension) ) );
    }

    // MHD
    if (modname == "mhd" || modname == "MHD") {
      modules.push_back(Teuchos::rcp(new mhd(settings, dimension) ) );
    }
    
    // Linear Elasticity
    if (modname == "linearelasticity" || modname == "linear elasticity") {
      modules.push_back(Teuchos::rcp(new linearelasticity(settings, dimension) ) );
    }
    
    // Helmholtz
    if (modname == "helmholtz") {
      modules.push_back(Teuchos::rcp(new helmholtz(settings, dimension) ) );
    }
    
    // Maxwell's (potential of electric field, curl-curl frequency domain (Boyse et al (1992))
    if (modname == "maxwells_freq_pot"){
      modules.push_back(Teuchos::rcp(new maxwells_fp(settings, dimension) ) );
    }
    
    // Scalar ODE for testing time integrators independent of spatial discretizations
    if (modname == "ODE"){
      modules.push_back(Teuchos::rcp(new ODE(settings, dimension) ) );
    }
    
    // Scalar Burgers equation
    if (modname == "Burgers"){
      modules.push_back(Teuchos::rcp(new Burgers(settings, dimension) ) );
    }

    // Kuramoto-Sivashinsky equation
    if (modname == "Kuramoto-Sivashinsky"){
      modules.push_back(Teuchos::rcp(new KuramotoSivashinsky(settings, dimension) ) );
    }
    
    // Llamas equation
    if (modname == "llamas"){
      modules.push_back(Teuchos::rcp(new llamas(settings, dimension) ) );
    }

    // Variable-density Navier-Stokes 
    if (modname == "VDNS"){
      modules.push_back(Teuchos::rcp(new VDNS(settings, dimension) ) );
    }

    // Euler equations
    if (modname == "Euler" || modname == "euler"){
      modules.push_back(Teuchos::rcp(new euler(settings, dimension) ) );
    }

    // Incompressible saturation equation
    if (modname == "incompressible saturation" || modname == "inc sat" ){
      modules.push_back(Teuchos::rcp(new incompressibleSaturation(settings, dimension) ) );
    }
    
    // Compressible Navier-Stokes
    //if (modname == "CNS" || modname == "cns"){
    //  modules.push_back(Teuchos::rcp(new cns(settings, dimension) ) );
    //}

    #if defined(MrHyDE_ENABLE_MIRAGE)
    // Physics for Mirage
    if (modname == "Mirage" || modname == "mirage"){
      modules.push_back(Teuchos::rcp(new mirage(settings, dimension) ) );
    }
    #endif    
  }
  
  return modules;
  
}
