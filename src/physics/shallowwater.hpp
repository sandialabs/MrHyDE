/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SHALLOWWATER_H
#define SHALLOWWATER_H

#include "physics_base.hpp"

static void shallowwaterHelp() {
  cout << "********** Help and Documentation for the Shallow Water Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class shallowwater : public physicsbase {
public:
  
  shallowwater() {} ;
  
  ~shallowwater() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  shallowwater(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
               const size_t & numip_side_, const int & numElem_,
               Teuchos::RCP<FunctionManager> & functionManager_,
               const size_t & blocknum_);
  
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
  
private:
  
  data grains;
  
  size_t numip, numip_side, blocknum;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int H_num, Hu_num, Hv_num;
  ScalarT alpha;
  ScalarT gravity;
  bool isTD;
  //int test, simNum;
  //string simName;
  
  FDATA bath, bath_x, bath_y, visc, cor, bfric, source_Hu, source_Hv, nsource, nsource_Hu, nsource_Hv, bath_side;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  DRV Hbasis, Hbasis_grad, Hubasis, Hubasis_grad, Hvbasis, Hvbasis_grad;
  
  
  string analysis_type; //to know when parameter is a sample that needs to be transformed
  
  bool useScalarRespFx;
  bool multiscale;
  ScalarT formparam;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::shallowwater::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::shallowwater::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::shallowwater::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::shallowwater::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::shallowwater::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::shallowwater::computeFlux() - evaluation of flux");
  
  //Teuchos::RCP<DRVAD> src_test;
  //Teuchos::RCP<FunctionBase> source_Hu_fct, source_Hv_fct, nsource_H_fct, nsource_Hu_fct, nsource_Hv_fct;
  
  
};

#endif
