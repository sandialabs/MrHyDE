/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MAXWELL_H
#define MAXWELL_H

#include "physics_base.hpp"

static void maxwellHelp() {
  cout << "********** Help and Documentation for the Maxwell (HCURL-HDIV) Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}


class maxwell : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  maxwell() {} ;
  
  ~maxwell() {};
  
  maxwell(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
          const size_t & numip_side_, const int & numElem_,
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
  
  void setVars(std::vector<string> & varlist_);
  
private:
  
  FDATA mu, epsilon;
  FDATA current_x, current_y, current_z;
  
  int spaceDim, numElem, numParams, numResponses, numSteps;
  size_t numip, numip_side;
  
  int Enum, Bnum;
  
  vector<string> varlist;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::maxwell::computeFlux() - evaluation of flux");
  
};

#endif
