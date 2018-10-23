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
#include "postprocessInterface.hpp"

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
  
  analysis(const Teuchos::RCP<Epetra_MpiComm> & LA_Comm_, const Teuchos::RCP<Epetra_MpiComm> & S_Comm_,
           Teuchos::RCP<Teuchos::ParameterList> & settings_, Teuchos::RCP<solver> & solver_, Teuchos::RCP<postprocess> & postproc_);
  
  // ========================================================================================
  /* given the parameters, solve the forward  problem */
  // ========================================================================================
  
  void run();
  
protected:
  
  //Epetra_MpiComm Comm;
  Teuchos::RCP<Epetra_MpiComm> LA_Comm;
  Teuchos::RCP<Epetra_MpiComm> S_Comm;
  Teuchos::RCP<Teuchos::ParameterList> settings;
  Teuchos::RCP<solver> solve;
  Teuchos::RCP<postprocess> postproc;
  
  double response;
  vector<double> gradient;
  int verbosity;
  
  bool sensIC;
};

#endif
