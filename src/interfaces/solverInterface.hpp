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
#include "physicsInterface.hpp"
#include "multiscaleInterface.hpp"
#include "discretizationInterface.hpp"
#include "discretizationTools.hpp"
#include "cell.hpp"
#include "assemblyManager.hpp"

void static solverHelp(const string & details) {
  cout << "********** Help and Documentation for the Solver Interface **********" << endl;
}

class solver {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  solver(const Teuchos::RCP<LA_MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
         Teuchos::RCP<panzer_stk::STK_Interface> & mesh_, Teuchos::RCP<discretization> & disc_,
         Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager<int,int> > & DOF_,
         Teuchos::RCP<AssemblyManager> & assembler_);
         //vector<vector<Teuchos::RCP<cell> > > & cells_);
  
  // ========================================================================================
  // Set up the Epetra objects (maps, importers, exporters and graphs)
  // These do need to be recomputed whenever the mesh changes */
  // ========================================================================================
  
  void setupLinearAlgebra();
  
  Teuchos::RCP<Epetra_CrsGraph> buildEpetraOverlappedGraph(Epetra_MpiComm & EP_Comm);
  
  Teuchos::RCP<Epetra_CrsGraph> buildEpetraOwnedGraph(Epetra_MpiComm & EP_Comm);
  
  // ========================================================================================
  // Set up the parameters (inactive, active, stochastic, discrete)
  // Communicate these parameters back to the physics interface and the enabled modules
  // ========================================================================================
  
  // move to parameter manager
  void setupParameters(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  /////////////////////////////////////////////////////////////////////////////
  // Read in discretized data from an exodus mesh
  /////////////////////////////////////////////////////////////////////////////

  // move ??
  void readMeshData(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  // move to sensor/particle manager
  void setupSensors(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  // move to parameter manager
  int getNumParams(const int & type);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  // move to parameter manager
  int getNumParams(const std::string & type);
  
  // ========================================================================================
  // return the discretized parameters as vector for use with ROL
  // ========================================================================================
  
  // move to parameter manager
  vector<ScalarT> getDiscretizedParamsVector();
  
  // ========================================================================================
  /* given the parameters, solve the forward  problem */
  // ========================================================================================
  
  // keep here
  vector_RCP forwardModel(DFAD & obj);
  
  // ========================================================================================
  /* given the parameters, solve the fractional forward  problem */
  // ========================================================================================
  
  // keep here
  vector_RCP forwardModel_fr(DFAD & obj, ScalarT yt, ScalarT st);
  
  // ========================================================================================
  // ========================================================================================
  
  // keep here
  vector_RCP adjointModel(vector_RCP & F_soln, vector<ScalarT> & gradient);
  
  
  // ========================================================================================
  /* solve the problem */
  // ========================================================================================
  
  // keep here
  void transientSolver(vector_RCP & initial, vector_RCP & L_soln,
                       vector_RCP & SolMat, DFAD & obj, vector<ScalarT> & gradient);
  
  // ========================================================================================
  // ========================================================================================
  
  // keep here
  void nonlinearSolver(vector_RCP & u, vector_RCP & u_dot,
                       vector_RCP & phi, vector_RCP & phi_dot,
                       const ScalarT & alpha, const ScalarT & beta);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to mesh interface
  void remesh(const vector_RCP & u);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to postprocess
  DFAD computeObjective(const vector_RCP & F_soln, const ScalarT & time, const size_t & tindex);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to postprocess
  vector<ScalarT> computeSensitivities(const vector_RCP & GF_soln,
                                      const vector_RCP & GA_soln);
  
  // ========================================================================================
  // Compute the sensitivity of the objective with respect to discretized parameters
  // ========================================================================================
  
  // move to postprocess
  vector<ScalarT> computeDiscretizedSensitivities(const vector_RCP & F_soln,
                                                 const vector_RCP & A_soln);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to postprocess
  void computeSensitivities(vector_RCP & u, vector_RCP & u_dot,
                            vector_RCP & a2, vector<ScalarT> & gradient,
                            const ScalarT & alpha, const ScalarT & beta);
  
  // ========================================================================================
  // The following function is the adjoint-based error estimate
  // Not to be confused with the postprocess::computeError function which uses a true
  //   solution to perform verification studies
  // ========================================================================================
  
  // move to postprocess
  ScalarT computeError(const vector_RCP & GF_soln, const vector_RCP & GA_soln);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to assembly manager
  //void updateJacDBC(matrix_RCP & J, size_t & e, size_t & block, int & fieldNum,
  //                  size_t & localSideId, const bool & compute_disc_sens);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to assembly manager
  //void updateJacDBC(matrix_RCP & J, const vector<int> & dofs, const bool & compute_disc_sens);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to assembly manager
  //void updateResDBC(vector_RCP & resid, size_t & e, size_t & block, int & fieldNum,
  //                  size_t & localSideId);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to assembly manager
  //void updateResDBC(vector_RCP & resid, const vector<int> & dofs);
  
  
  // ========================================================================================
  // ========================================================================================
  
  // move to assembly manager
  //void updateResDBCsens(vector_RCP & resid, size_t & e, size_t & block, int & fieldNum, size_t & localSideId,
  //                      const std::string & gside, const ScalarT & current_time);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to assembly manager
  void setDirichlet(vector_RCP & initial);
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP setInitialParams();
  
  // ========================================================================================
  // ========================================================================================
  
  // keep here
  vector_RCP setInitial();
  
  // ========================================================================================
  // ========================================================================================
  
  // move to assembly manager
  
   void computeJacRes(vector_RCP & u, vector_RCP & u_dot,
                     vector_RCP & phi, vector_RCP & phi_dot,
                     const ScalarT & alpha, const ScalarT & beta,
                     const bool & compute_jacobian, const bool & compute_sens,
                     const bool & compute_disc_sens,
                     vector_RCP & res, matrix_RCP & J);
  
  
  // ========================================================================================
  // Linear solver for Tpetra stack
  // ========================================================================================
  
  // keep here
  void linearSolver(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  
  // ========================================================================================
  // Linear solver for Epetra stack
  // ========================================================================================
  
  // keep here
  void linearSolver(Teuchos::RCP<Epetra_CrsMatrix> & J,
                    Teuchos::RCP<Epetra_MultiVector> & r,
                    Teuchos::RCP<Epetra_MultiVector> & soln);
  
  // ========================================================================================
  // Preconditioner for Tpetra stack
  // ========================================================================================
  
  // keep here
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > buildPreconditioner(const matrix_RCP & J);
  
  // ========================================================================================
  // Preconditioner for Epetra stack
  // ========================================================================================
  
  // keep here
  ML_Epetra::MultiLevelPreconditioner* buildPreconditioner(const Teuchos::RCP<Epetra_CrsMatrix> & J);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  void sacadoizeParams(const bool & seed_active);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  void updateParams(const vector<ScalarT> & newparams, const int & type);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  void updateParams(const vector<ScalarT> & newparams, const std::string & stype);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to mesh interface
  void updateMeshData(const int & newseed);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  vector<ScalarT> getParams(const int & type);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  vector<string> getParamsNames(const int & type);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  vector<size_t> getParamsLengths(const int & type);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  vector<ScalarT> getParams(const std::string & stype);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  vector<vector<ScalarT> > getParamBounds(const std::string & stype);
  
  // ========================================================================================
  // ========================================================================================
  
  // move???
  void setBatchID(const int & bID);
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  void stashParams();
  
  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  vector<ScalarT> getStochasticParams(const std::string & whichparam);

  // ========================================================================================
  // ========================================================================================
  
  // move to parameter manager
  vector<ScalarT> getFractionalParams(const std::string & whichparam);
  
  // ========================================================================================
  // ========================================================================================
  
  // keep here
  vector_RCP blankState();
  
  // ========================================================================================
  //
  // ========================================================================================
  
  // move to assembly manager???
  //void performGather(const size_t & block, const vector_RCP & vec, const int & type,
  //                   const size_t & index);
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  // move to mesh interface
  DRV getElemNodes(const int & block, const int & elemID) ;
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  // keep???
  void finalizeMultiscale() ;
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<AssemblyManager> assembler;
  //vector<Teuchos::RCP<workset> > wkset;
  
  Teuchos::RCP<const LA_Map> LA_owned_map;
  Teuchos::RCP<const LA_Map> LA_overlapped_map, sol_overlapped_map;
  
  Teuchos::RCP<LA_CrsGraph> LA_owned_graph;
  Teuchos::RCP<LA_CrsGraph> LA_overlapped_graph;
  
  Teuchos::RCP<LA_Export> exporter;
  Teuchos::RCP<LA_Import> importer;
  
  
  Teuchos::RCP<const LA_Map> param_owned_map;
  Teuchos::RCP<const LA_Map> param_overlapped_map;
  
  Teuchos::RCP<LA_Export> param_exporter;
  Teuchos::RCP<LA_Import> param_importer;
  
  
  vector<DRV> elemnodes;
  
  int numUnknowns;     					 // total number of unknowns
  int numUnknownsOS;     					 // total number of unknowns
  int globalNumUnknowns;
  vector<int> owned;					 // GIDs that live on the local processor.
  vector<int> ownedAndShared;				 // GIDs that live or are shared on the local processor.
  vector<int> LA_owned;					 // GIDs that live on the local processor.
  vector<int> LA_ownedAndShared;				 // GIDs that live or are shared on the local processor.
  
  int allow_remesh, MaxNLiter, meshmod_xvar, meshmod_yvar, meshmod_zvar, time_order;
  ScalarT NLtol, meshmod_TOL, meshmod_center, meshmod_layer_size, finaltime;
  string solver_type, NLsolver, initial_type;
  bool line_search, meshmod_usesmoother, useL2proj;
  
  ScalarT lintol, dropTol, fillParam;
  int liniter, kspace;
  bool useDomDecomp, useDirect, usePrec;
  
  vector<Kokkos::View<ScalarT**,HostDevice> > sensor_data;
  Kokkos::View<ScalarT**,HostDevice> sensor_points;
  //FCint sensor_locations;
  int numSensors;

  
  vector<string> paramnames;
  vector<vector<ScalarT> > paramvals;
  vector<Teuchos::RCP<vector<AD> > > paramvals_AD;
  Kokkos::View<AD**,AssemblyDevice> paramvals_KVAD;
  
  vector<vector_RCP> Psol;
  vector<vector_RCP> auxsol;
  vector<vector_RCP> dRdP;
  bool have_dRdP;
  Teuchos::RCP<const panzer::DOFManager<int,int> > discparamDOF;
  vector<vector<ScalarT> > paramLowerBounds;
  vector<vector<ScalarT> > paramUpperBounds;
  vector<string> discretized_param_basis_types;
  vector<int> discretized_param_basis_orders, discretized_param_usebasis;
  vector<string> discretized_param_names;
  vector<basis_RCP> discretized_param_basis;
  Teuchos::RCP<panzer::DOFManager<int,int> > paramDOF;
  vector<vector<int> > paramoffsets;
  vector<int> paramNumBasis;
  int numParamUnknowns;     					 // total number of unknowns
  int numParamUnknownsOS;     					 // total number of unknowns
  int globalParamUnknowns; // total number of unknowns across all processors
  vector<int> paramOwned;					 // GIDs that live on the local processor.
  vector<int> paramOwnedAndShared;				 // GIDs that live or are shared on the local processor.
  
  vector<int> paramtypes;
  vector<vector<int>> paramNodes;  // for distinguishing between parameter fields when setting initial
  vector<vector<int>> paramNodesOS;// values and bounds
  int num_inactive_params, num_active_params, num_stochastic_params, num_discrete_params, num_discretized_params;
  vector<ScalarT> initialParamValues, lowerParamBounds, upperParamBounds, discparamVariance;
  vector<ScalarT> domainRegConstants, boundaryRegConstants;
  vector<string> boundaryRegSides;
  vector<int> domainRegTypes, domainRegIndices, boundaryRegTypes, boundaryRegIndices;
  
  
  int verbosity;
  string response_type, multigrid_type, smoother_type;
  bool discretized_stochastic;
  
  
  vector<string> stochastic_distribution, discparam_distribution;
  vector<ScalarT> stochastic_mean, stochastic_variance, stochastic_min, stochastic_max;
  

  //fractional parameters
  vector<ScalarT> s_exp;
  vector<ScalarT> h_mesh;

  int batchID; //which stochastic collocation batch; to avoid multiple processors trying to stash at once to same file
  
  //vector<vector<Teuchos::RCP<cell> > > cells;
  
  ScalarT current_time;
  
  
  vector<ScalarT> solvetimes;
  
  vector<vector_RCP> fwdsol;
  vector<vector_RCP> adjsol;

  
  
  int spaceDim;
  vector<string> blocknames;
  bool isInitial;
  vector<vector<int> > numBasis;
  vector<vector<int> > useBasis;
  vector<int> maxbasis;
  bool isTransient, useadjoint, have_sensor_data, have_sensor_points;
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
  
  //vector<Teuchos::RCP<SubGridModel> > subgridModels;

  // variables read in from an exodus mesh
  vector_RCP meas;
  vector<vector<ScalarT> > nfield_vals, efield_vals;
  vector<string> nfield_names, efield_names;
  int numResponses;
  bool use_meas_as_dbcs;
  
private:
  
  Teuchos::RCP<LA_MpiComm> Comm;
  //Teuchos::RCP<Teuchos::ParameterList> settings;
  Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
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
