/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef HELMHOLTZ_H
#define HELMHOLTZ_H

#include "physics_base.hpp"

static void helmholtzHelp() {
  cout << "********** Help and Documentation for the Helmholtz Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class helmholtz : public physicsbase {
public:
  
  helmholtz() {} ;
  
  ~helmholtz() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  helmholtz(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  // ========================================================================================
  // ========================================================================================
  
  void defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                       Teuchos::RCP<FunctionManager> & functionManager_);
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual();
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual();
  
  
  void edgeResidual();
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux();
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_);
  
  
private:
  
  int spaceDim;
  int ur_num, ui_num;
  
  int verbosity;
  
  //Kokkos::View<AD****,AssemblyDevice> sol, sol_dot, sol_grad;
  //Kokkos::View<AD**,AssemblyDevice> res, adjrhs;
  //Kokkos::View<int**,AssemblyDevice> offsets;
  //Kokkos::View<int****,AssemblyDevice> sideinfo;
  //DRV urbasis, uibasis, urbasis_grad, uibasis_grad;
  
  
  AD ur, durdx, durdy, durdz, durdn, c2durdn;
  AD ui, duidx, duidy, duidz, duidn, c2duidn;
  ScalarT vr, dvrdx, dvrdy, dvrdz;
  ScalarT vi, dvidx, dvidy, dvidz;
    
  bool fractional;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::helmholtz::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::helmholtz::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::helmholtz::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::helmholtz::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::helmholtz::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::helmholtz::computeFlux() - evaluation of flux");
  
};

#endif
