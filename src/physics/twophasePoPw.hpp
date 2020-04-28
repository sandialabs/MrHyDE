/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef TWOPHASEPOPW_H
#define TWOPHASEPOPW_H

#include "physics_base.hpp"

class twophasePoPw : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  twophasePoPw() {} ;
  
  ~twophasePoPw() {};
  
  twophasePoPw(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
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
  
  void setVars(std::vector<string> & varlist_);
  
private:
  
  int spaceDim, numElem;
  size_t numip, numip_side;
  ScalarT formparam;
  int Ponum, Pwnum, resindex;
  
  vector<string> varlist;
  
  FDATA perm, porosity, gravity, cpinv, dcpinv;
  FDATA relperm_o, source_o, viscosity_o, densref_o, pref_o, comp_o;
  FDATA relperm_w, source_w, viscosity_w, densref_w, pref_w, comp_w;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::computeFlux() - evaluation of flux");
  
};

#endif
