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

#ifndef MRHYDE_ANALYSIS_H
#define MRHYDE_ANALYSIS_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "solverManager.hpp"
#include "postprocessManager.hpp"
#include "parameterManager.hpp"

namespace MrHyDE {
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  /*
  static void analysisHelp(const string & details) {
    cout << "********** Help and Documentation for the Analysis Interface **********" << endl;
  }
  */
  
  class analysis {
    
    typedef Tpetra::Map<LO, GO, SolverNode>               LA_Map;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector>                  vector_RCP;
    
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    analysis(const Teuchos::RCP<MpiComm> & Comm_, 
             Teuchos::RCP<Teuchos::ParameterList> & settings_, Teuchos::RCP<solver<SolverNode> > & solver_,
             Teuchos::RCP<PostprocessManager<SolverNode> > & postproc_, Teuchos::RCP<ParameterManager<SolverNode> > & params_);
    
    // ========================================================================================
    /* given the parameters, solve the forward  problem */
    // ========================================================================================
    
    void run();
    
  protected:
    
    Teuchos::RCP<MpiComm> Comm;
    //Teuchos::RCP<MpiComm> S_Comm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<solver<SolverNode> > solve;
    Teuchos::RCP<PostprocessManager<SolverNode> > postproc;
    Teuchos::RCP<ParameterManager<SolverNode> > params;
    
    ScalarT response;
    vector<ScalarT> gradient;
    int verbosity, milo_debug_level;
    
    bool sensIC;
  };
  
}

#endif
