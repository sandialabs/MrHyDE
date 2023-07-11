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

#ifndef MRHYDE_MHD2D_H
#define MRHYDE_MHD2D_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief mhd2d physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "source ux" is the source ux.
   *   - "density" is the density.
   *   - "viscosity" is the viscosity.
   *   - "source uz" is the source uz.
   *   - "source pr" is the source pr.
   *   - "source uy" is the source uy.
   */
  class mhd2d : public physicsbase {
  public:
    
    mhd2d() {} ;
    
    ~mhd2d() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    mhd2d(Teuchos::ParameterList & settings, const int & dimension_);
    
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
    
    /* @brief Returns the value of the stabilization parameter (SUPG/PSPG)
     *
     * @param[in] localdiff  Kinematic viscosity
     * @param[in] xvl  x-component of the velocity
     * @param[in] yvl  y-component of the velocity
     * @param[in] zvl  z-component of the velocity
     * @param[in] h  Element diameter
     * @param[in] spaceDim  Number of spatial dimensions
     * @param[in] dt  Timestep
     * @param[in] isTransient  Bool indicating if the simulation is transient

     * @return SUPG/PSPG stabilization parameter (type AD)
     *
     */
    // TODO: Document these functions
    KOKKOS_FUNCTION AD computeTauMomentum(const AD &dens, const AD &visc, const AD &xvl, const AD &yvl, const AD &xmag, const AD &ymag, const ScalarT &h, const ScalarT &dt) const;
    KOKKOS_FUNCTION AD computeTauTemp(const AD &dens, const AD &xvl, const AD &yvl, const AD &Cp, const ScalarT &h, const ScalarT &dt) const;
    KOKKOS_FUNCTION AD computeTauAz(const AD &eta, const AD &xvl, const AD &yvl, const ScalarT &h, const ScalarT &dt) const;
    
  private:
    
    int ux_num, uy_num, Bx_num, By_num, Az_num, pr_num, T_num;
    
    bool useSUPG, usePSPG, useTemp;

    vector<ScalarT> pik;
    Kokkos::View<ScalarT*,AssemblyDevice> model_params;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mhd2d::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mhd2d::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mhd2d::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mhd2d::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mhd2d::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mhd2d::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
