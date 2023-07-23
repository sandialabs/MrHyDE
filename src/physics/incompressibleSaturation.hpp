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

/** @file incompressibleSaturation.hpp
 *
 * @brief Shallow water physics module, hybridized version
 *
 * Solves the shallow water equations with a hybridized formulation.
 * See Samii (J. Sci. Comp. 2019). 
 */

#ifndef MRHYDE_INCOMPRESSIBLESATURATION_H
#define MRHYDE_INCOMPRESSIBLESATURATION_H

#include "physicsBase.hpp"
#include "wells.hpp"

namespace MrHyDE {
  
  /** 
   * \brief Two-phase, incompressible saturation equation module. 
   *
   * Solves the two-phase, incompressible saturation equation 
   * for the water phase.
   * 
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "uy" is the uy.
   *   - "f_w" is the f_w.
   *   - "uz" is the uz.
   *   - "ux" is the ux.
   *   - "source_S" is the source_S.
   */

  class incompressibleSaturation : public physicsbase {
  public:

    incompressibleSaturation() {};
    
    ~incompressibleSaturation() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    incompressibleSaturation(Teuchos::ParameterList & settings, const int & dimension);
    
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
    
    void setWorkset(Teuchos::RCP<Workset> & wkset_);

    /* @brief Update the fluxes for the residual calculation.
     *
     */

    void computeFluxVector();

// TODO This needs to be handled in a better way, temporary!
#ifndef MrHyDE_UNITTEST_HIDE_PRIVATE_VARS
  private:
#endif

    int spaceDim;
    
    int S_num;

    View_AD4 fluxes_vol; // Storage for the fluxes

    wells myWells;
    bool useWells;

    ScalarT phi; // porosity

    Kokkos::View<ScalarT*,AssemblyDevice> modelparams;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::computeFlux() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> fluxVectorFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::computeFluxVector() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxVectorFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::computeFluxVector() - evaluation of flux");

  };
  
}

#endif
