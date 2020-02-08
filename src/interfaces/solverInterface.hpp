/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SOLVER_H
#define SOLVER_H

#include "trilinos.hpp"
#include "Panzer_DOFManager.hpp"

#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "multiscaleInterface.hpp"
#include "discretizationInterface.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "solutionStorage.hpp"

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosBlockGmresSolMgr.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

// Amesos includes
#include "Amesos2.hpp"

typedef Belos::LinearProblem<ScalarT, LA_MultiVector, LA_Operator> LA_LinearProblem;

void static solverHelp(const string & details) {
  cout << "********** Help and Documentation for the Solver Interface **********" << endl;
}

class solver {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  solver(const Teuchos::RCP<MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
         Teuchos::RCP<meshInterface> & mesh_,
         Teuchos::RCP<discretization> & disc_,
         Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager> & DOF_,
         Teuchos::RCP<AssemblyManager> & assembler_,
         Teuchos::RCP<ParameterManager> & params_);
  
  
  // ========================================================================================
  // Set up the Tpetra objects (maps, importers, exporters and graphs)
  // These do need to be recomputed whenever the mesh changes */
  // ========================================================================================
  
  void setupLinearAlgebra();
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void finalizeWorkset();
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void forwardModel(DFAD & obj);
  
  // ========================================================================================
  /* given the parameters, solve the fractional forward  problem */
  // ========================================================================================
  
  void forwardModel_fr(DFAD & obj, ScalarT yt, ScalarT st);
  
  // ========================================================================================
  // ========================================================================================
  
  void adjointModel(vector<ScalarT> & gradient);
    
  // ========================================================================================
  /* solve the problem */
  // ========================================================================================
  
  void transientSolver(vector_RCP & initial, DFAD & obj, vector<ScalarT> & gradient,
                       ScalarT & start_time, ScalarT & end_time);
  
  // ========================================================================================
  // ========================================================================================
  
  int nonlinearSolver(vector_RCP & u, vector_RCP & u_dot,
                      vector_RCP & phi, vector_RCP & phi_dot,
                      const ScalarT & alpha, const ScalarT & beta);
  
  // ========================================================================================
  // ========================================================================================
  
  void setButcherTableau();
  
  // ========================================================================================
  // ========================================================================================
  
  int explicitRKTimeSolver(vector_RCP & u, vector_RCP & u_dot,
                           vector_RCP & phi, vector_RCP & phi_dot,
                           matrix_RCP & mass);
  
  // ========================================================================================
  // ========================================================================================
  
  DFAD computeObjective(const vector_RCP & F_soln, const ScalarT & time, const size_t & tindex);
  
  // ========================================================================================
  // ========================================================================================
  
  void computeSensitivities(vector_RCP & u, vector_RCP & u_dot,
                            vector_RCP & a2, vector<ScalarT> & gradient,
                            const ScalarT & alpha, const ScalarT & beta);
  
  
  // ========================================================================================
  // ========================================================================================
  
  void setDirichlet(vector_RCP & initial);
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP setInitialParams();
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP setInitial();
  
  // ========================================================================================
  // Linear solver for Tpetra stack
  // ========================================================================================
  
  void linearSolver(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  void setupMassSolver(matrix_RCP & mass, vector_RCP & r, vector_RCP & soln);

  // ========================================================================================
  // Preconditioner for Tpetra stack
  // ========================================================================================
  
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > buildPreconditioner(const matrix_RCP & J);
  
  // ========================================================================================
  // ========================================================================================
  
  void setBatchID(const LO & bID);
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP blankState();
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void finalizeMultiscale() ;
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<AssemblyManager> assembler;
  Teuchos::RCP<ParameterManager> params;
  Teuchos::RCP<meshInterface>  mesh;
  
  Teuchos::RCP<const LA_Map> LA_owned_map, LA_overlapped_map;
  Teuchos::RCP<LA_CrsGraph> LA_owned_graph, LA_overlapped_graph;
  Teuchos::RCP<LA_Export> exporter;
  Teuchos::RCP<LA_Import> importer;
  
  LO numUnknowns, numUnknownsOS;
  GO globalNumUnknowns;
  int verbosity, batchID, spaceDim, numsteps, gNLiter, milo_debug_level, MaxNLiter, time_order, liniter, kspace;
  
  size_t maxEntries;
  
  vector<GO> owned, ownedAndShared, LA_owned, LA_ownedAndShared;
  
  ScalarT NLtol, final_time, lintol, dropTol, fillParam, current_time, initial_time, deltat;
  
  string solver_type, NLsolver, initial_type, response_type, multigrid_type, smoother_type;
  string TDsolver;
  
  bool line_search, useL2proj, allow_remesh, useDomDecomp, useDirect, usePrec, discretized_stochastic;
  bool isInitial, isTransient, useadjoint, is_final_time, usestrongDBCs, compute_flux, useLinearSolver, timeImplicit;
  bool compute_objective, compute_sensitivity, compute_aux_sensitivity, use_custom_initial_param_guess, store_adjPrev, use_meas_as_dbcs;
  
  Kokkos::View<ScalarT**,HostDevice> butcher_A;
  Kokkos::View<ScalarT*,HostDevice> butcher_b, butcher_c;
  Teuchos::RCP<LA_LinearProblem> massProblem;
  Teuchos::RCP<Belos::SolverManager<ScalarT, LA_MultiVector, LA_Operator> > massSolver;
  //vector<ScalarT> solvetimes;
  
  Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > Am2Solver;
  bool have_symbolic_factor;
  
  Teuchos::RCP<SolutionStorage<LA_MultiVector> > soln, adj_soln, soln_dot;
  
  //vector<vector_RCP> fwdsol;
  //vector<vector_RCP> adjsol;
  vector<string> blocknames;
  vector<vector<string> > varlist;
  
  vector<vector<LO> > numBasis, useBasis;
  vector<LO> maxBasis, numVars;
  
  Teuchos::RCP<MultiScale> multiscale_manager;
  
private:
  
  Teuchos::RCP<MpiComm> Comm;
  Teuchos::RCP<discretization> disc;
  Teuchos::RCP<physics> phys;
  Teuchos::RCP<const panzer::DOFManager> DOF;
  
  Teuchos::RCP<Teuchos::Time> assemblytimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - total assembly");
  Teuchos::RCP<Teuchos::Time> linearsolvertimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::linearSolver()");
  Teuchos::RCP<Teuchos::Time> gathertimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - gather");
  Teuchos::RCP<Teuchos::Time> phystimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - physics evaluation");
  Teuchos::RCP<Teuchos::Time> boundarytimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - boundary evaluation");
  Teuchos::RCP<Teuchos::Time> inserttimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - insert");
  Teuchos::RCP<Teuchos::Time> dbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - strong Dirichlet BCs");
  Teuchos::RCP<Teuchos::Time> completetimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - fill complete");
  Teuchos::RCP<Teuchos::Time> msprojtimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - multiscale projection");
  
};

#endif
