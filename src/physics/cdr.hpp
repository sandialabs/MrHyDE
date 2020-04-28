/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef CDR_H
#define CDR_H

#include "physics_base.hpp"

class cdr : public physicsbase {
public:
  
  cdr() {} ;
  
  ~cdr() {};
  
  // ========================================================================================
  // ========================================================================================
  
  cdr(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
      const size_t & numip_side_, const int & numElem_,
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
  
  void setVars(vector<string> & varlist_);
  
  // ========================================================================================
  // return the value of the stabilization parameter 
  // ========================================================================================
  
  template<class T>  
  T computeTau(const T & localdiff, const T & xvl, const T & yvl, const T & zvl, const ScalarT & h) const;
  
private:
  
  FDATA diff, rho, cp, xvel, yvel, zvel, reax, tau, source, nsource, diff_side, robin_alpha;
  
  int spaceDim, numElem;
  size_t numip, numip_side;
  vector<string> varlist;
  int cnum, resindex;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::computeFlux() - evaluation of flux");
  
  
};

#endif
