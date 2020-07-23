/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef POROUSHDIVHYBRID_H
#define POROUSHDIVHYBRID_H

#include "physics_base.hpp"

static void porousHDIVHYBRIDHelp() {
  cout << "********** Help and Documentation for the Porous (HDIV) Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}


class porousHDIV_HYBRID : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  porousHDIV_HYBRID() {} ;
  
  ~porousHDIV_HYBRID() {};
  
  porousHDIV_HYBRID(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
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
  
  // ========================================================================================
  // The edge (2D) and face (3D) contributions to the residual
  // ========================================================================================
  
  void faceResidual();
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux();
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_);
  
  // ========================================================================================
  // ========================================================================================
  
  void setAuxVars(std::vector<string> & auxvarlist);
  
  // ========================================================================================
  // ========================================================================================
  
  void updatePerm();
  
private:
  
  int spaceDim;
  FDATA source, bsource, Kinv_xx, Kinv_yy, Kinv_zz;
  
  int pnum=-1, unum=-1, lambdanum=-1;
  int auxpnum=-1, auxunum=-1, auxlambdanum=-1;
  int dxnum=-1, dynum=-1, dznum=-1;
  bool isTD, addBiot, usePermData;
  ScalarT biot_alpha;
    
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_HYBRID::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_HYBRID::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_HYBRID::computeFlux() - evaluation of interface flux");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_HYBRID::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_HYBRID::boundaryResidual() - evaluation of residual");
  
};

#endif
