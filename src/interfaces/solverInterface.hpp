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
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "multiscaleInterface.hpp"
#include "discretizationInterface.hpp"
#include "discretizationTools.hpp"
#include "cell.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"

void static solverHelp(const string & details) {
  cout << "********** Help and Documentation for the Solver Interface **********" << endl;
}

class solver {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  solver(const Teuchos::RCP<LA_MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
         Teuchos::RCP<meshInterface> & mesh_,
         //Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
         Teuchos::RCP<discretization> & disc_,
         Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager<int,int> > & DOF_,
         Teuchos::RCP<AssemblyManager> & assembler_,
         Teuchos::RCP<ParameterManager> & params_);
  
  
  // ========================================================================================
  // Set up the Epetra objects (maps, importers, exporters and graphs)
  // These do need to be recomputed whenever the mesh changes */
  // ========================================================================================
  
  void setupLinearAlgebra();
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<Epetra_CrsGraph> buildEpetraOverlappedGraph(Epetra_MpiComm & EP_Comm);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<Epetra_CrsGraph> buildEpetraOwnedGraph(Epetra_MpiComm & EP_Comm);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void finalizeWorkset();
  
  /////////////////////////////////////////////////////////////////////////////
  // Read in discretized data from an exodus mesh
  /////////////////////////////////////////////////////////////////////////////

  void readMeshData(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  //void setupSensors(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  vector_RCP forwardModel(DFAD & obj);
  
  // ========================================================================================
  /* given the parameters, solve the fractional forward  problem */
  // ========================================================================================
  
  vector_RCP forwardModel_fr(DFAD & obj, ScalarT yt, ScalarT st);
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP adjointModel(vector_RCP & F_soln, vector<ScalarT> & gradient);
  
  
  // ========================================================================================
  /* solve the problem */
  // ========================================================================================
  
  void transientSolver(vector_RCP & initial, vector_RCP & L_soln,
                       vector_RCP & SolMat, DFAD & obj, vector<ScalarT> & gradient);
  
  // ========================================================================================
  // ========================================================================================
  
  void nonlinearSolver(vector_RCP & u, vector_RCP & u_dot,
                       vector_RCP & phi, vector_RCP & phi_dot,
                       const ScalarT & alpha, const ScalarT & beta);
  
  // ========================================================================================
  // ========================================================================================
  
  DFAD computeObjective(const vector_RCP & F_soln, const ScalarT & time, const size_t & tindex);
  
  // ========================================================================================
  // ========================================================================================
  
  vector<ScalarT> computeSensitivities(const vector_RCP & GF_soln,
                                      const vector_RCP & GA_soln);
  
  // ========================================================================================
  // Compute the sensitivity of the objective with respect to discretized parameters
  // ========================================================================================
  
  vector<ScalarT> computeDiscretizedSensitivities(const vector_RCP & F_soln,
                                                 const vector_RCP & A_soln);
  
  // ========================================================================================
  // ========================================================================================
  
  void computeSensitivities(vector_RCP & u, vector_RCP & u_dot,
                            vector_RCP & a2, vector<ScalarT> & gradient,
                            const ScalarT & alpha, const ScalarT & beta);
  
  // ========================================================================================
  // The following function is the adjoint-based error estimate
  // Not to be confused with the postprocess::computeError function which uses a true
  //   solution to perform verification studies
  // ========================================================================================
  
  ScalarT computeError(const vector_RCP & GF_soln, const vector_RCP & GA_soln);
  
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
  
  // ========================================================================================
  // Linear solver for Epetra stack
  // ========================================================================================
  
  void linearSolver(Teuchos::RCP<Epetra_CrsMatrix> & J,
                    Teuchos::RCP<Epetra_MultiVector> & r,
                    Teuchos::RCP<Epetra_MultiVector> & soln);
  
  // ========================================================================================
  // Preconditioner for Tpetra stack
  // ========================================================================================
  
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > buildPreconditioner(const matrix_RCP & J);
  
  // ========================================================================================
  // Preconditioner for Epetra stack
  // ========================================================================================
  
  ML_Epetra::MultiLevelPreconditioner* buildPreconditioner(const Teuchos::RCP<Epetra_CrsMatrix> & J);
  
  // ========================================================================================
  // ========================================================================================
  
  void updateMeshData(const int & newseed);
  
  // ========================================================================================
  // ========================================================================================
  
  void setBatchID(const int & bID);
  
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
  
  Teuchos::RCP<const LA_Map> LA_owned_map, LA_overlapped_map;
  Teuchos::RCP<LA_CrsGraph> LA_owned_graph, LA_overlapped_graph;
  Teuchos::RCP<LA_Export> exporter;
  Teuchos::RCP<LA_Import> importer;
  
  int numUnknowns;     					 // total number of unknowns
  int numUnknownsOS;     					 // total number of unknowns
  int globalNumUnknowns;
  vector<int> owned;					 // GIDs that live on the local processor.
  vector<int> ownedAndShared;				 // GIDs that live or are shared on the local processor.
  vector<int> LA_owned;					 // GIDs that live on the local processor.
  vector<int> LA_ownedAndShared;				 // GIDs that live or are shared on the local processor.
  
  int allow_remesh, MaxNLiter, time_order;
  ScalarT NLtol, finaltime;
  string solver_type, NLsolver, initial_type;
  bool line_search, useL2proj;
  
  ScalarT lintol, dropTol, fillParam;
  int liniter, kspace;
  bool useDomDecomp, useDirect, usePrec;
  
  int verbosity;
  string response_type, multigrid_type, smoother_type;
  bool discretized_stochastic;
  
  int batchID; //which stochastic collocation batch; to avoid multiple processors trying to stash at once to same file
  
  ScalarT current_time;
  
  vector<ScalarT> solvetimes;
  
  vector<vector_RCP> fwdsol;
  vector<vector_RCP> adjsol;

  int spaceDim;
  vector<string> blocknames;
  bool isInitial;
  vector<vector<int> > numBasis;
  vector<vector<int> > useBasis;
  vector<int> maxBasis;
  bool isTransient, useadjoint;
  bool is_final_time, usestrongDBCs;
  
  //vector<FCint> offsets;
  vector<int> numVars;        				 // Number of variables used by the application (may not be used yet)
  int numsteps;
  vector<vector<string> > varlist;
  
  Teuchos::RCP<MultiScale> multiscale_manager;
  
  bool compute_objective, compute_sensitivity;
  bool use_custom_initial_param_guess;
  bool store_adjPrev;
  int gNLiter;
  bool use_meas_as_dbcs;
  
  
private:
  
  Teuchos::RCP<LA_MpiComm> Comm;
  //Teuchos::RCP<Teuchos::ParameterList> settings;
  //Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
  Teuchos::RCP<meshInterface>  mesh;
  Teuchos::RCP<discretization> disc;
  Teuchos::RCP<physics> phys;
  Teuchos::RCP<const panzer::DOFManager<int,int> > DOF;
  //Teuchos::RCP<TimeIntegrator> timeInt;
  
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
