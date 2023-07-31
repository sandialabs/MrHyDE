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

#ifndef MRHYDE_POROUS_H
#define MRHYDE_POROUS_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief Porous media physics class.
   * 
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   (\partial_t (p-p_r), q) + \left(\frac{k \rho (1+c)}{\mu} \nabla (p-p_r), \nabla q \right)
   *       &=& (f,q).
   * \f}
   * Where the unknown \f$p\f$ is the fluid pressure.
   * The following functions may be specified in the input.yaml file:
   *   - "source" is the source tern, \f$f\f$.
   *   - "permeability" is the permeability scalar \f$k\f$.
   *   - "porosity" is the porosity.
   *   - "viscosity" is the kinematic viscosity \f$\mu\f$.
   *   - "reference density" is the fluid density \f$\rho\f$.
   *   - "reference pressure" is the reference pressure \f$p_r\f$ that is subtracted from pressure \f$p\f$.
   *   - "compressibility" is the compressibility \f$c\f$.
   *   - "gravity" is gravity \f$g\f$.
   */
  class porous : public physicsbase {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    porous() {} ;
    
    ~porous() {};
    
    porous(Teuchos::ParameterList & settings, const int & dimension_);
    
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
    // ========================================================================================
    
    void edgeResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<Workset<AD> > & wkset_);

    //void setVars(std::vector<string> & varlist_);
    
    // ========================================================================================
    // ========================================================================================
    
    void updatePerm(View_AD2 perm);
    
    
  private:
    
    int pnum, auxpnum;
    
    ScalarT formparam;
    string auxvar;
    
    //Kokkos::View<int****,AssemblyDevice> sideinfo;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porous::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porous::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porous::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porous::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porous::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porous::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
