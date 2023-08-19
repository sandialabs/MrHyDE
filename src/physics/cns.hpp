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

#ifndef MRHYDE_CDNS_H
#define MRHYDE_CDNS_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /** 
   * \brief Compressible Navier-Stokes physics module 
   *
   * Solves the compressible Navier-Stokes equations for conservation
   * of mass, momentum, and energy.
   * Transport and thermodynamic properties are assumed to be functions
   * of temperature.
   * We employ an ideal gas law.
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
   *   - "source E" is the source E.
   *   - "source uz" is the source uz.
   *   - "kappa" is the kappa.
   *   - "T" is the T.
   *   - "cp" is the cp.
   *   - "p0" is the p0.
   *   - "mu" is the mu.
   *   - "source uy" is the source uy.
   */

  template<class EvalT>
  class cns : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    cns() {} ;
    
    ~cns() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    cns(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_);
    
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

    KOKKOS_FUNCTION 
    EvalT computeTau(const EvalT & rhoDiffl, const EvalT & xvl, const EvalT & yvl, const EvalT & zvl, const EvalT & rho, 
                  const ScalarT & h, const int & spaceDim, const ScalarT & dt, const bool & isTransient) const;

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
