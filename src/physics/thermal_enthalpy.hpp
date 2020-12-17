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

#ifndef THERMAL_ENTHALPY_H
#define THERMAL_ENTHALPY_H

#include "physics_base.hpp"

namespace MrHyDE {
  /*
  static void thermal_enthalpyHelp() {
    cout << "********** Help and Documentation for the Thermal Enthalpy Physics Module **********" << endl << endl;
    cout << "Model:" << endl << endl;
    cout << "User defined functions: " << endl << endl;
  }
  */
  
  class thermal_enthalpy : public physicsbase {
  public:
    
    thermal_enthalpy() {} ;
    
    ~thermal_enthalpy() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    thermal_enthalpy(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_);
    
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
    
    void setVars(std::vector<string> & varlist_);
    
  private:
    
    data grains;
    
    int spaceDim;
    int e_num, e_basis, numBasis, ux_num, uy_num, uz_num;
    int H_num, H_basis; // for melt fraction variable
    ScalarT alpha;
    
    ScalarT v, dvdx, dvdy, dvdz, x, y, z;
    AD e, e_dot, dedx, dedy, dedz, reax, weakDiriScale, lambda, penalty;
    AD H, H_dot, dHdx, dHdy, dHdz; // spatial derivatives of g are not explicity needed atm
    AD ux, uy, uz;
    ScalarT latent_heat = 2.7e5;
    
    FDATA diff, rho, cp, source, nsource, diff_side, robin_alpha;
    
    Kokkos::View<int****,AssemblyDevice> sideinfo;
    
    string analysis_type; //to know when parameter is a sample that needs to be transformed
    
    bool useScalarRespFx;
    bool multiscale, have_nsvel;
    ScalarT formparam;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
