/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef THERMAL_H
#define THERMAL_H

#include "physics_base.hpp"

static void thermalHelp() {
  cout << "********** Help and Documentation for the Thermal Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class thermal : public physicsbase {
public:
  
  thermal() {} ;
  
  ~thermal() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  thermal(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
          const size_t & numip_side_, const int & numElem_,
          Teuchos::RCP<FunctionInterface> & functionManager_,
          const size_t & blocknum_) ;
  
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
  
  // ========================================================================================
  // ========================================================================================
  
  void setAuxVars(std::vector<string> & auxvarlist);
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;

  data grains;
 
  size_t numip, numip_side, blocknum;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int e_num, e_basis, numBasis, ux_num, uy_num, uz_num;
  int auxe_num = -1;
  ScalarT alpha;
  bool isTD;
  //int test, simNum;
  //string simName;
  
  ScalarT v, dvdx, dvdy, dvdz, x, y, z;
  AD e, e_dot, dedx, dedy, dedz, reax, weakDiriScale, lambda, penalty;
  AD ux, uy, uz;
  
  int resindex;
  
  FDATA diff, rho, cp, source, nsource, diff_side, robin_alpha;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  string analysis_type; //to know when parameter is a sample that needs to be transformed
  
  bool useScalarRespFx;
  bool multiscale, have_nsvel;
  ScalarT formparam;
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::computeFlux() - evaluation of flux");
  
};

#endif
