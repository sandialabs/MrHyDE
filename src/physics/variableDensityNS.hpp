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

/** @file variableDensityNS.hpp
 *
 * @brief Variable-density Navier-Stokes physics module
 *
 * Solves the variable-density Navier-Stokes equations for conservation
 * of mass, momentum, and a scalar transport equation (\f$T\f$).
 * Transport and thermodynamic properties are assumed to be functions
 * of temperature.
 * We employ the low-Mach formulation where the thermodynamic pressure
 * and the density are decoupled.
 */

#ifndef MRHYDE_VDNS_H
#define MRHYDE_VDNS_H

#include "physicsBase.hpp"

namespace MrHyDE {
  /*
  static void navierstokesHelp() {
    cout << "********** Help and Documentation for the Variable Density Navier Stokes Physics Module **********" << endl << endl;
    cout << "Model:" << endl << endl;
    cout << "User defined functions: " << endl << endl;
  }
  */
  
  /** 
   * \brief Variable-density Navier-Stokes physics module 
   *
   * Solves the variable-density Navier-Stokes equations for conservation
   * of mass, momentum, and a scalar transport equation (\f$T\f$).
   * Transport and thermodynamic properties are assumed to be functions
   * of temperature.
   * We employ the low-Mach formulation where the thermodynamic pressure
   * and the density are decoupled.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "RGas" is the RGas.
   *   - "source ux" is the source ux.
   *   - "PrNum" is the PrNum.
   *   - "source uz" is the source uz.
   *   - "source T" is the source T.
   *   - "rho" is the rho.
   *   - "cp" is the cp.
   *   - "p0" is the p0.
   *   - "gamma" is the gamma.
   *   - "source pr" is the source pr.
   *   - "mu" is the mu.
   *   - "lambda" is the lambda.
   *   - "source uy" is the source uy.
   */

  template<class EvalT>
  class VDNS : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    VDNS() {} ;
    
    ~VDNS() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    VDNS(Teuchos::ParameterList & settings, const int & dimension_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT> > & functionManager_);
    
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
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);

    /**
     * @brief Returns the integrands and their types (boundary/volume) for integrated quantities required
     * by the VDNS module. These are needed for closed systems where the background thermodynamic
     * pressure changes over time.
     *
     * @return integrandsNamesAndTypes  Integrands, names, and type (boundary/volume) (matrix of strings).
     */
    
    std::vector< std::vector<string> > setupIntegratedQuantities(const int & spaceDim);

    /**
     * @brief Updates the background thermodynamic pressure and estimates \f$\frac{dp_0}{dt}\f$
     * which is required in the energy equation.
     */
    
    void updateIntegratedQuantitiesDependents();
    
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
     */

    KOKKOS_FUNCTION EvalT computeTau(const EvalT & rhoDiffl, const EvalT & xvl, const EvalT & yvl, const EvalT & zvl, 
                                     const EvalT & rho, const ScalarT & h, const int & spaceDim, const ScalarT & dt, 
                                     const bool & isTransient) const;

  private:
    
    int ux_num, uy_num, uz_num, pr_num, T_num, IQ_start;

    bool useSUPG, usePSPG, useGRADDIV, openSystem, inoutflow;
    
    Kokkos::View<ScalarT*,AssemblyDevice> model_params;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::computeFlux() - evaluation of flux");
  };
  
}

#endif
