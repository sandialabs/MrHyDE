/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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
#include "MrHyDE_Debugger.hpp"

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
    
    /**
     * @brief Constructor that actually sets everything up.
     *
     * @param[in]  comm        Teuchos MPI Communicator
     * @param[in]  settings  Teuchos ParameterList containing .all of the user-specified settings
     * @param[in]  solver      MrHyDE solver manager
     * @param[in]  postproc      MrHyDE postprocessing manager
     * @param[in]  param      MrHyDE parameter (including discretized parameters) manager
     */
    
    AnalysisManager(const Teuchos::RCP<MpiComm> & comm,
                    Teuchos::RCP<Teuchos::ParameterList> & settings,
                    Teuchos::RCP<SolverManager<SolverNode> > & solver,
                    Teuchos::RCP<PostprocessManager<SolverNode> > & postproc,
                    Teuchos::RCP<ParameterManager<SolverNode> > & params);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief 
     */
    
    void run();
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief
     */
    
    void run(std::string & analysis_type);
    
    // ========================================================================================
    // ========================================================================================
    
    DFAD forwardSolve();

    // ========================================================================================
    // ========================================================================================
    
    MrHyDE_OptVector adjointSolve();

    // ========================================================================================
    // ========================================================================================
    
    vector<Teuchos::Array<ScalarT> > UQSolve();

    // ========================================================================================
    // ========================================================================================
    
    void ROLSolve();

    // ========================================================================================
    // ========================================================================================
    
    void ROL2Solve();
    
    // ========================================================================================
    // ========================================================================================
    
    void DCISolve();
    
    // ========================================================================================
    // ========================================================================================
    
    void restartSolve();
    
    // ========================================================================================
    // ========================================================================================
    
    void recoverSolution(vector_RCP & solution, string & data_type,
                         string & plist_filename, string & file_name);
 
    // ========================================================================================
    // ========================================================================================
    
    void updateRotationData(const int & newrandseed);
    
    // ========================================================================================
    // ========================================================================================
    
    void writeSolutionToText(string & filename, vector<vector<vector_RCP> > & soln,
                             const bool & only_write_final = false);

  private:
    
    Teuchos::RCP<MpiComm> comm_;
    Teuchos::RCP<Teuchos::ParameterList> settings_;
    Teuchos::RCP<SolverManager<SolverNode> > solver_;
    Teuchos::RCP<PostprocessManager<SolverNode> > postproc_;
    Teuchos::RCP<ParameterManager<SolverNode> > params_;
    Teuchos::RCP<MrHyDE_Debugger> debugger_;
    
    int verbosity_;
  };
  
}

#endif
