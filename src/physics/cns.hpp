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

/** @file cns.hpp
 *
 * @brief Compressible Navier-Stokes physics module
 *
 * Solves the compressible Navier-Stokes equations for conservation
 * of mass, momentum, and energy.
 * Transport and thermodynamic properties are assumed to be functions
 * of temperature.
 * We employ an ideal gas law.
 */

#ifndef CDNS_H
#define CDNS_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /** Compressible Navier-Stokes physics module 
   *
   * Solves the compressible Navier-Stokes equations for conservation
   * of mass, momentum, and energy.
   * Transport and thermodynamic properties are assumed to be functions
   * of temperature.
   * We employ an ideal gas law.
   */

  class cns : public physicsbase {
  public:

    cns() {} ;
    
    ~cns() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    cns(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
    
    // ========================================================================================
    // ========================================================================================
    
    void boundaryResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    //void setVars(std::vector<string> & varlist_);
    
    void setWorkset(Teuchos::RCP<workset> & wkset_);

    // ========================================================================================
    // return the value of the stabilization parameter 
    // ========================================================================================
    
    /* @brief Returns the value of the stabilization parameter (SUPG/PSPG).
     *
     * @param[in] rhoDiffl  Diffusivity times density
     * @param[in] xvl  x-component of the velocity
     * @param[in] yvl  y-component of the velocity
     * @param[in] zvl  z-component of the velocity
     * @param[in] rho  Density
     * @param[in] h  Element diameter
     * @param[in] spaceDim  Number of spatial dimensions
     * @param[in] dt  Timestep
     * @param[in] isTransient  Bool indicating if the simulation is transient
     * @return SUPG/PSPG stabilization parameter (type AD)
     *
     * @details The diffusivity weighted by the density is somewhat generic
     * so this is appropriate for different conservation equations.
     *
     */

    KOKKOS_FUNCTION AD computeTau(const AD & rhoDiffl, const AD & xvl, const AD & yvl, const AD & zvl, const AD & rho, const ScalarT & h, const int & spaceDim, const ScalarT & dt, const bool & isTransient) const;

  private:
    
    int rho_num, rhoux_num, rhouy_num, rhouz_num, rhoE_num;

    bool useSUPG, usePSPG, useGRADDIV;

    string KEdef;

    Kokkos::View<ScalarT*,AssemblyDevice> model_params;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cns::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cns::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cns::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cns::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cns::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cns::computeFlux() - evaluation of flux");
  };
  
}

#endif
