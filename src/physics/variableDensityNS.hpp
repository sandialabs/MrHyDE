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

#ifndef VDNS_H
#define VDNS_H

#include "physicsBase.hpp"

namespace MrHyDE {
  /*
  static void navierstokesHelp() {
    cout << "********** Help and Documentation for the Variable Density Navier Stokes Physics Module **********" << endl << endl;
    cout << "Model:" << endl << endl;
    cout << "User defined functions: " << endl << endl;
  }
  */
  
  /** Variable-density Navier-Stokes physics module 
   *
   * Solves the variable-density Navier-Stokes equations for conservation
   * of mass, momentum, and a scalar transport equation (\f$T\f$).
   * Transport and thermodynamic properties are assumed to be functions
   * of temperature.
   * We employ the low-Mach formulation where the thermodynamic pressure
   * and the density are decoupled.
   *
   */

  class VDNS : public physicsbase {
  public:

    VDNS() {} ;
    
    ~VDNS() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    VDNS(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_);
    
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
     * @return SUPG/PSPG stabilization parameter (type AD)
     *
     * @details The diffusivity weighted by the density is somewhat generic
     * so this is appropriate for different conservation equations.
     *
     */

    AD computeTau(const AD & rhoDiffl, const AD & xvl, const AD & yvl, const AD & zvl, const AD & rho, const ScalarT & h) const;

    // TODO MOVE TO YAML 
    /* @brief Return the density as a function of \f$T\f$.
     *
     * @param[in] T  Temperature
     * @param[in] p0  Thermodynamic pressure
     * @param[in] RGas  Specific gas constant
     * @return density (type AD)
     *
     */

    //AD fRho(const AD & T, const AD & p0, const AD & RGas);

    ///* @brief Return the partial derivative of the EOS with respect to temperature.
    // *
    // * @param[in] T  Temperature
    // * @param[in] p0  Thermodynamic pressure
    // * @param[in] RGas  Specific gas constant
    // * @return \f$\partial \rho / \partial T\f$ (type AD)
    // *
    // */

    //AD dfRhodT(const AD & T, const AD & p0, const AD & RGas);

    ///* @brief Return the dynamic viscosity as a function of \f$T\f$ using Sutherland's Law.
    // *
    // * @param[in] T  Temperature
    // * @param[in] TRef  Reference temperature 
    // * @param[in] S  Sutherland temperature
    // * @param[in] muRef  Reference dynamic viscosity
    // * @return dynamic viscosity (type AD)
    // *
    // */

    //AD fMu(const AD & T, const & ScalarT TRef, const & ScalarT S, const & ScalarT muRef);

    ///* @brief Return the thermal conductivity as a function of \f$T\f$.
    // *
    // * @param[in] mu  Dynamic viscosity
    // * @param[in] cp  Specific heat at constant pressure
    // * @param[in] PrNum  Prandtl number
    // * @return thermal conductivity (type AD)
    // *
    // */

    //AD fLambda(const AD & mu, const AD & cp, const ScalarT & PrNum);
    
  private:
    
    int ux_num, uy_num, uz_num, pr_num, T_num;

    bool useSUPG, usePSPG;
    
    Kokkos::View<ScalarT*,AssemblyDevice> model_params;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::computeFlux() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> updateProps = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VDNS::updateThermAndTransProps() - function evals");
  };
  
}

#endif
