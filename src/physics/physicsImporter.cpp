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
#include "ode.hpp"
#include "burgers.hpp"
#include "kuramotoSivashinsky.hpp"
#include "llamas.hpp"
#include "variableDensityNS.hpp"
#include "euler.hpp"
#include "shallowwaterHybridized.hpp"
#include "incompressibleSaturation.hpp"
#include "shallowice.hpp"
#include "hartmann.hpp"

#if defined(MrHyDE_ENABLE_MIRAGE)
#include "mirage.hpp"
#endif

//#include "cns.hpp"

using namespace MrHyDE;

template<class EvalT>
vector<Teuchos::RCP<PhysicsBase<EvalT> > > PhysicsImporter<EvalT>::import(vector<string> & module_list,
                                                           Teuchos::ParameterList & settings,
                                                           const int & dimension,
                                                           Teuchos::RCP<MpiComm> & Commptr) {
  
  vector<Teuchos::RCP<PhysicsBase<EvalT> > > modules;
  
  for (size_t mod=0; mod<module_list.size(); mod++) {
    string modname = module_list[mod];
    
    // Test module which procedurally assembles and dumps basis functions based on parameterlist settings
    if (modname == "physicsTest") {
      modules.push_back(Teuchos::rcp(new physicsTest<EvalT>(settings, dimension) ) );
    }

    // Porous media (single phase slightly compressible)
    if (modname == "porous") {
      modules.push_back(Teuchos::rcp(new porous<EvalT>(settings, dimension) ) );
    }
    
    // Porous media with HDIV basis
    if (modname == "porous mixed") {
      modules.push_back(Teuchos::rcp(new porousMixed<EvalT>(settings, dimension) ) );
    }
    
    // Hybridized porous media with HDIV basis
    if (modname == "porous mixed hybridized") {
      modules.push_back(Teuchos::rcp(new porousMixedHybrid<EvalT>(settings, dimension) ) );
    }
    
    // weak Galerkin porous media with HDIV basis
    if (modname == "porous weak Galerkin") {
      modules.push_back(Teuchos::rcp(new porousWeakGalerkin<EvalT>(settings, dimension) ) );
    }
    
    // Convection diffusion
    if (modname == "cdr" || modname == "CDR") {
      modules.push_back(Teuchos::rcp(new cdr<EvalT>(settings, dimension) ) );
    }
    
    // Thermal
    if (modname == "thermal") {
      modules.push_back(Teuchos::rcp(new thermal<EvalT>(settings, dimension) ) );
    }
    
    // Shallow Water
    if (modname == "shallow water") {
      modules.push_back(Teuchos::rcp(new shallowwater<EvalT>(settings, dimension) ) );
    }

    // Shallow Ice
    if (modname == "shallow ice") {
      modules.push_back(Teuchos::rcp(new shallowice<EvalT>(settings, dimension) ) );
    }

    // Shallow water hybridized
    if (modname == "shallow water hybridized") {
      modules.push_back(Teuchos::rcp(new shallowwaterHybridized<EvalT>(settings, dimension) ) );
    }
    
    // Maxwell
    if (modname == "maxwell") {
      modules.push_back(Teuchos::rcp(new maxwell<EvalT>(settings, dimension) ) );
    }
    
    // Multiple Species PhaseField
    if (modname == "msphasefield") {
      modules.push_back(Teuchos::rcp(new msphasefield<EvalT>(settings, dimension, Commptr) ) );
    }
    
    // Stokes
    if (modname == "stokes" || modname == "Stokes") {
      modules.push_back(Teuchos::rcp(new stokes<EvalT>(settings, dimension) ) );
    }
    
    // Navier Stokes
    if (modname == "navier stokes" || modname == "Navier Stokes") {
      modules.push_back(Teuchos::rcp(new navierstokes<EvalT>(settings, dimension) ) );
    }
    // Hartmann
    if (modname == "hartmann") {
      modules.push_back(Teuchos::rcp(new hartmann<EvalT>(settings, dimension) ) );
    }
    
    // Linear Elasticity
    if (modname == "linearelasticity" || modname == "linear elasticity") {
      modules.push_back(Teuchos::rcp(new linearelasticity<EvalT>(settings, dimension) ) );
    }
    
    // Helmholtz
    if (modname == "helmholtz") {
      modules.push_back(Teuchos::rcp(new helmholtz<EvalT>(settings, dimension) ) );
    }
    
    // Maxwell's (potential of electric field, curl-curl frequency domain (Boyse et al (1992))
    if (modname == "maxwells_freq_pot"){
      modules.push_back(Teuchos::rcp(new maxwells_fp<EvalT>(settings, dimension) ) );
    }
    
    // Scalar ODE for testing time integrators independent of spatial discretizations
    if (modname == "ODE"){
      modules.push_back(Teuchos::rcp(new ODE<EvalT>(settings, dimension) ) );
    }
    
    // Scalar Burgers equation
    if (modname == "Burgers"){
      modules.push_back(Teuchos::rcp(new Burgers<EvalT>(settings, dimension) ) );
    }

    // Kuramoto-Sivashinsky equation
    if (modname == "Kuramoto-Sivashinsky"){
      modules.push_back(Teuchos::rcp(new KuramotoSivashinsky<EvalT>(settings, dimension) ) );
    }
    
    // Llamas equation
    if (modname == "llamas"){
      modules.push_back(Teuchos::rcp(new llamas<EvalT>(settings, dimension) ) );
    }

    // Variable-density Navier-Stokes 
    if (modname == "VDNS"){
      modules.push_back(Teuchos::rcp(new VDNS<EvalT>(settings, dimension) ) );
    }

    // Euler equations
    if (modname == "Euler" || modname == "euler"){
      modules.push_back(Teuchos::rcp(new euler<EvalT>(settings, dimension) ) );
    }

    // Incompressible saturation equation
    if (modname == "incompressible saturation" || modname == "inc sat" ){
      modules.push_back(Teuchos::rcp(new incompressibleSaturation<EvalT>(settings, dimension) ) );
    }
    
    // Compressible Navier-Stokes
    //if (modname == "CNS" || modname == "cns"){
    //  modules.push_back(Teuchos::rcp(new cns(settings, dimension) ) );
    //}

    #if defined(MrHyDE_ENABLE_MIRAGE)
    // Physics for Mirage
    if (modname == "Mirage" || modname == "mirage"){
      modules.push_back(Teuchos::rcp(new mirage<EvalT>(settings, dimension) ) );
    }
    #endif  
    
  }
  
  return modules;
  
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::PhysicsImporter<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::PhysicsImporter<AD>;

// Standard built-in types
template class MrHyDE::PhysicsImporter<AD2>;
template class MrHyDE::PhysicsImporter<AD4>;
template class MrHyDE::PhysicsImporter<AD8>;
template class MrHyDE::PhysicsImporter<AD16>;
//template class MrHyDE::PhysicsImporter<AD18>; // AquiEEP_merge
template class MrHyDE::PhysicsImporter<AD24>;
template class MrHyDE::PhysicsImporter<AD32>;
#endif
