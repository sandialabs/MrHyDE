/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/** \file   analysisManager.hpp
 *  \brief  Creates the analysis manager which performs the high-level interface to the solution strategies,
 *          e.g., standard run, dry run, UQ, and ROL-based optimization.
 *  \author Created by T. Wildey
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

/**
 * \class AnalysisManager
 * \brief Executes the simulation based on the user-selected analysis mode.
 *
 * Provides the high-level driver that invokes different solution strategies
 * such as forward solves, adjoint solves, uncertainty quantification (UQ),
 * and several ROL-based optimization workflows.
 */
class AnalysisManager {

  using LA_Map         = Tpetra::Map<LO, GO, SolverNode>;                ///< Local alias for Tpetra map
  using LA_MultiVector = Tpetra::MultiVector<ScalarT,LO,GO,SolverNode>;  ///< Local alias for Tpetra multivector
  using vector_RCP     = Teuchos::RCP<LA_MultiVector>;                   ///< RCP alias for multivector

public:

  /** @brief Default constructor */
  AnalysisManager() {};

  /** @brief Default destructor */
  ~AnalysisManager() {};

  // ========================================================================================
  // Constructor
  // ========================================================================================

  /**
   * @brief Main constructor that sets up the analysis environment.
   *
   * @param[in]  comm        MPI communicator
   * @param[in]  settings    Global settings from input file
   * @param[in]  solver      Reference to the solver manager
   * @param[in]  postproc    Reference to the postprocessing manager
   * @param[in]  params      Reference to the parameter manager
   */
  AnalysisManager(const Teuchos::RCP<MpiComm> & comm,
                  Teuchos::RCP<Teuchos::ParameterList> & settings,
                  Teuchos::RCP<SolverManager<SolverNode> > & solver,
                  Teuchos::RCP<PostprocessManager<SolverNode> > & postproc,
                  Teuchos::RCP<ParameterManager<SolverNode> > & params);

  // ========================================================================================
  // Run routines
  // ========================================================================================

  /**
   * @brief Executes the analysis specified in the settings.
   */
  void run();

  /**
   * @brief Executes analysis based on an explicitly provided type.
   * @param[in] analysis_type  String defining the requested analysis mode
   */
  void run(std::string & analysis_type);

  // ========================================================================================
  // Solve routines
  // ========================================================================================

  /** @brief Perform a standard forward solve. */
  void forwardSolve();

  /**
   * @brief Perform an adjoint solve.
   * @return A vector of adjoint variables (ROL-compatible wrapper)
   */
  MrHyDE_OptVector adjointSolve();

  /**
   * @brief Run the uncertainty quantification (UQ) solve.
   * @return A nested vector of scalar arrays representing UQ results
   */
  vector<Teuchos::Array<ScalarT> > UQSolve();

  /** @brief Perform an optimization solve using ROL. */
  void ROLSolve();

  /** @brief Perform a second ROL-based optimization strategy. */
  void ROL2Solve();

  /** @brief Perform stochastic optimization using ROL. */
  void ROLStochSolve();

#if defined(MrHyDE_ENABLE_HDSA)
  /** @brief Execute HDSA analysis. */
  void HDSASolve();

  /** @brief Read Exodus output and perform a forward solve. */
  void readExoForwardSolve();
#endif

  /** @brief Run DCI analysis. */
  void DCISolve();

  /** @brief Run scalable DCI analysis. */
  void ScalableDCISolve();

  /** @brief Bayesian scalable solve routine. */
  void ScalableBayesSolve();

  /** @brief Perform a restart solve (load previous state and continue). */
  void restartSolve();

  /**
   * @brief Recover a stored solution.
   * @param[out] solution       Returned solution vector
   * @param[in]  data_type      Type of stored data
   * @param[in]  plist_filename Parameter list filename
   * @param[in]  file_name      Data file name
   */
  void recoverSolution(vector_RCP & solution, string & data_type,
                       string & plist_filename, string & file_name);

  /**
   * @brief Update rotation-related data for cases involving randomization.
   * @param[in] newrandseed   New random seed to use
   */
  void updateRotationData(const int & newrandseed);

  /**
   * @brief Write solution vectors to text files.
   * @param[in]  filename           Output file name
   * @param[in]  soln               Nested solution vector
   * @param[in]  only_write_final   If true, write only the last time step
   */
  void writeSolutionToText(string & filename, vector<vector<vector_RCP> > & soln,
                           const bool & only_write_final = false);

private:

  Teuchos::RCP<MpiComm> comm_;                    ///< MPI communicator
  Teuchos::RCP<Teuchos::ParameterList> settings_; ///< Input settings
  Teuchos::RCP<SolverManager<SolverNode> > solver_; ///< Solver manager
  Teuchos::RCP<PostprocessManager<SolverNode> > postproc_; ///< Postprocessing manager
  Teuchos::RCP<ParameterManager<SolverNode> > params_; ///< Parameter manager
  Teuchos::RCP<MrHyDE_Debugger> debugger_; ///< Debugging utility

  int verbosity_; ///< Verbosity level for logging

  Teuchos::RCP<Teuchos::Time> roltimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AnalysisManager::ROLSolve()"); ///< Timer for ROLSolve
  Teuchos::RCP<Teuchos::Time> rol2timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AnalysisManager::ROL2Solve()"); ///< Timer for ROL2Solve
};

} // namespace MrHyDE

#endif
