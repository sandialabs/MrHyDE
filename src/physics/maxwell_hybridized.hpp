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

#ifndef MAXWELL_HYBRID_H
#define MAXWELL_HYBRID_H

#include "physics_base.hpp"

namespace MrHyDE {
  
/*
  static void maxwellhybridHelp() {
    cout << "********** Help and Documentation for the Maxwell Hybrid (HCURL-HDIV) Physics Module **********" << endl << endl;
    cout << "Model:" << endl << endl;
    cout << "User defined functions: " << endl << endl;
  }
  */

  /*
   * This class computes the solution to the physics formed by applying the HDG finite
   * element method to 3D time-domain PDEs for Maxwell's equations. It yields the
   * finite element system described in the paper from Appl. Math. Comput. (2018) by
   * Christophe, Descombes, and Lanteri.
   *
   * The system may be written as (see equation (9)):
   *   (\varepsilon \partial_t E_h, v)_{T_h} - (H_h, curl(v))_{T_h} + (lambda_h, n x v)_{\partial T_h}                 = 0,
   *   (\mu \partial_t H_h, v)_{T_h}         - (curl(E_h), v)_{T_h} - (tau n x (H_h - lambda_h), n x v)_{\partial T_h} = 0,
   *   (n x E_h, \eta)_{\partial T_h} + (\tau(H_h - lambda_h), \eta)_{\partial T_h} - (\lambda_h, \eta)_{\Gamma_a}  = (g^{inc}, \eta)_{\Gamma_a}.
   *
   * Where tau is a local stabilization parameter. One option for choosing tau is to
   * set it equal to 1/sqrt(\varepsilon/\mu) to obtain numerical traces which are the
   * same as the upwind flux DGTD method (see Remark 2).
   *
   * The spaces are defined in the paper as follows:
   *   E_h, H_h are both polynomial DG spaces
   *   lambda_h := \hat{H_h} is a mortar polynomial space
   */
  class maxwell_HYBRID : public physicsbase {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    maxwell_HYBRID() {} ;
    
    ~maxwell_HYBRID() {};
    
    maxwell_HYBRID(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_);
    
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
    
    void faceResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    void setVars(std::vector<string> & varlist_);
    
  private:
    
    int spaceDim;
    
    int Ex_num, Ey_num, Ez_num,
    Hx_num, Hy_num, Hz_num,
    lambdax_num, lambday_num, lambdaz_num;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell_HYBRID::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell_HYBRID::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell_HYBRID::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell_HYBRID::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell_HYBRID::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell_HYBRID::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
