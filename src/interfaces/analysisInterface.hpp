/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "solverInterface.hpp"
#include "postprocessManager.hpp"
#include "parameterManager.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

static void analysisHelp(const string & details) {
  cout << "********** Help and Documentation for the Analysis Interface **********" << endl;
}

class analysis {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  analysis(const Teuchos::RCP<MpiComm> & Comm_, 
           Teuchos::RCP<Teuchos::ParameterList> & settings_, Teuchos::RCP<solver> & solver_,
           Teuchos::RCP<PostprocessManager> & postproc_, Teuchos::RCP<ParameterManager> & params_);
  
  // ========================================================================================
  /* given the parameters, solve the forward  problem */
  // ========================================================================================
  
  void run();
  
protected:
  
  Teuchos::RCP<MpiComm> Comm;
  //Teuchos::RCP<MpiComm> S_Comm;
  Teuchos::RCP<Teuchos::ParameterList> settings;
  Teuchos::RCP<solver> solve;
  Teuchos::RCP<PostprocessManager> postproc;
  Teuchos::RCP<ParameterManager> params;
  
  ScalarT response;
  vector<ScalarT> gradient;
  int verbosity, milo_debug_level;
  
  bool sensIC;
};

#endif
