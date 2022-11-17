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

/** \file   analysisManager.hpp
 \brief  Creates the analysis manager which performs the high-level interface to the solution strategies, e.g, standard run, dry run, UQ, ROL-optimization.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_ANALYSIS_MANAGER_H
#define MRHYDE_ANALYSIS_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "solverManager.hpp"
#include "postprocessManager.hpp"
#include "parameterManager.hpp"

namespace MrHyDE {
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /** \class  MrHyDE::AnalysisManager
   \brief  Executes the simulation based on the user-provided analysis mode.
   */
  
  class AnalysisManager {
    
    typedef Tpetra::Map<LO, GO, SolverNode>               LA_Map;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector>                  vector_RCP;
    
  public:
    
    AnalysisManager() {};
    
    ~AnalysisManager() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    AnalysisManager(const Teuchos::RCP<MpiComm> & Comm_,
                    Teuchos::RCP<Teuchos::ParameterList> & settings_,
                    Teuchos::RCP<SolverManager<SolverNode> > & solver_,
                    Teuchos::RCP<PostprocessManager<SolverNode> > & postproc_,
                    Teuchos::RCP<ParameterManager<SolverNode> > & params_);
    
    // ========================================================================================
    // ========================================================================================
    
    void run();
    
    DFAD forwardSolve();

    MrHyDE_OptVector adjointSolve();

    vector<Teuchos::Array<ScalarT> > UQSolve();

    void ROLSolve();

    void ROL2Solve();
    
    void recoverSolution(vector_RCP & solution, string & data_type, string & file_name);
 
    // ========================================================================================
    // ========================================================================================
    
    void updateRotationData(const int & newrandseed);
    
    void writeSolutionToText(string & filename, vector<vector<vector_RCP> > & soln);

  protected:
    
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<SolverManager<SolverNode> > solve;
    Teuchos::RCP<PostprocessManager<SolverNode> > postproc;
    Teuchos::RCP<ParameterManager<SolverNode> > params;
    
    ScalarT response;
    Teuchos::RCP<MrHyDE_OptVector> gradient;
    int verbosity, debug_level;
    
    bool sensIC;
  };
  
}

#endif
