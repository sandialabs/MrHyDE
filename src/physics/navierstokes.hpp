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

#ifndef MRHYDE_NAVIERSTOKES_H
#define MRHYDE_NAVIERSTOKES_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  class navierstokes : public physicsbase {
  public:
    
    navierstokes() {} ;
    
    ~navierstokes() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    navierstokes(Teuchos::ParameterList & settings, const int & dimension_);
    
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

    KOKKOS_FUNCTION AD computeTau(const AD & localdiff, const AD & xvl, const AD & yvl, const AD & zvl, const ScalarT & h, const int & spaceDim, const ScalarT & dt, const bool & isTransient) const;
    
  private:
    
    int ux_num, uy_num, uz_num, pr_num, e_num;
    
    bool useSUPG, usePSPG;

    vector<ScalarT> pik;
    bool have_energy;
    ScalarT T_ambient, beta;
    Kokkos::View<ScalarT*,AssemblyDevice> model_params;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
