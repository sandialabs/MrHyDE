/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/
#ifndef MRHYDE_SOLVER_MANAGER
#define MRHYDE_SOLVER_MANAGER

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "multiscaleManager.hpp"
#include "discretizationInterface.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "postprocessManager.hpp"
#include "solutionStorage.hpp"
#include "linearAlgebraInterface.hpp"
#include "MrHyDE_Debugger.hpp"

namespace MrHyDE {

/**
 * @class SolverManager
 * @brief Manages the setup, execution, and solution of steady and transient PDE systems.
 *
 * @tparam Node The Tpetra node type used for linear algebra device execution.
 */
template<class Node>
class SolverManager {
  
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;  ///< Local alias for sparse matrix type
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector; ///< Local alias for multivector type
  typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;     ///< Shared pointer to a multivector
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;     ///< Shared pointer to a matrix
  typedef typename Node::device_type              LA_device;      ///< Linear algebra device type
  
public:
  
  /** @brief Default constructor */
  SolverManager() {};
  
  /** @brief Destructor */
  ~SolverManager() {};
  
  /**
   * @brief Fully constructing constructor.
   * @param[in] Comm_ MPI communicator
   * @param[in] settings_ Global solver settings
   * @param[in] mesh_ Mesh interface
   * @param[in] disc_ Discretization interface
   * @param[in] phys_ Physics interface
   * @param[in] assembler_ Assembly manager
   * @param[in] params_ Parameter manager
   */
  SolverManager(const Teuchos::RCP<MpiComm> & Comm_,
                Teuchos::RCP<Teuchos::ParameterList> & settings_,
                Teuchos::RCP<MeshInterface> & mesh_,
                Teuchos::RCP<DiscretizationInterface> & disc_,
                Teuchos::RCP<PhysicsInterface> & phys_,
                Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                Teuchos::RCP<ParameterManager<Node> > & params_);
  
  /** @brief Completes solver setup after object construction */
  void completeSetup();
  
  /**
   * @brief Prepare structures for explicit mass matrix operations.
   */
  void setupExplicitMass();
  
  /**
   * @brief Set up mass matrix actions for discretized parameters.
   */
  void setupDiscretizedParamMass();
  
  /**
   * @brief Assign a Butcher tableau to a physics set.
   * @param[in] tableau The selected tableau
   * @param[in] set Physics set index
   */
  void setButcherTableau(const vector<string> & tableau, const int & set);
  
  /**
   * @brief Assign BDF weights for a given order.
   * @param[in] order Time integration order for each set
   * @param[in] set Physics set index
   */
  void setBackwardDifference(const vector<int> & order, const int & set);
  
  /** @brief Configure fixed DOFs based on settings */
  void setupFixedDOFs(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  /** @brief Finalize workset allocation for assembly */
  void finalizeWorkset();
  
  /**
   * @brief Finalizes a workset for a given evaluation type.
   * @tparam EvalT Evaluation type
   * @param[in,out] wkset Workset container
   * @param[in] paramvals_KV Parameter values
   * @param[in] paramdot_KV Time derivative of parameter values
   */
  template<class EvalT>
  void finalizeWorkset(vector<Teuchos::RCP<Workset<EvalT> > > & wkset,
                       Kokkos::View<EvalT**,AssemblyDevice> paramvals_KV,
                       Kokkos::View<EvalT**,AssemblyDevice> paramdot_KV);
  
  /** @brief Executes the forward PDE solve */
  void forwardModel(ScalarT & objective);
  
  /** @brief Executes adjoint solve for gradient computation */
  void adjointModel(MrHyDE_OptVector & gradient);
  
  /** @brief Incremental forward solve (for Hessian-vector products) */
  void incrementalForwardModel(ScalarT & objective);
  
  /** @brief Incremental adjoint solve (for Hessian-vector products) */
  void incrementalAdjointModel(MrHyDE_OptVector & hessvec);
  
  /** @brief Solve steady-state PDE */
  void steadySolver(vector<vector_RCP> & u);
  
  /**
   * @brief Run a transient simulation.
   * @param[in,out] initial Initial condition vectors
   * @param[out] gradient Computed gradient
   * @param[in] start_time Simulation start time
   * @param[in] end_time End time
   */
  void transientSolver(vector<vector_RCP> & initial,
                       MrHyDE_OptVector & gradient,
                       ScalarT & start_time, ScalarT & end_time);
  
  /** @brief Nonlinear solve for a specific stage */
  int nonlinearSolver(const size_t & set, const size_t & stage,
                      vector<vector_RCP> & sol,
                      vector<vector_RCP> & sol_stage,
                      vector<vector_RCP> & sol_prev,
                      vector<vector_RCP> & phi,
                      vector<vector_RCP> & phi_stage,
                      vector<vector_RCP> & phi_prev);
  
  /** @brief Explicit time-stepping solve */
  int explicitSolver(const size_t & set, const size_t & stage,
                     vector<vector_RCP> & sol,
                     vector<vector_RCP> & sol_stage,
                     vector<vector_RCP> & sol_prev,
                     vector<vector_RCP> & phi,
                     vector<vector_RCP> & phi_stage,
                     vector<vector_RCP> & phi_prev );
  
  /** @brief Solve linearized state system */
  void stateSolve(vector<vector_RCP> &sol,const vector<vector_RCP> &forcing);
  
  /** @brief Apply Dirichlet boundary conditions */
  void setDirichlet(const size_t & set, vector_RCP & u);
  
  /** @brief Project solution into Dirichlet-constrained space */
  void projectDirichlet(const size_t & set);
  
  /** @brief Generate initial parameter vector */
  vector_RCP setInitialParams();
  
  /** @brief Generate initial solution vector(s) */
  vector<vector_RCP> setInitial();
  
  /** @brief Assign a batch ID for multi-sample solves */
  void setBatchID(const LO & bID);
  
  /** @brief Return a blank state vector */
  vector_RCP blankState();
  
  /** @brief Load solution from restart files */
  vector<vector_RCP> getRestartSolution();
  
  /** @brief Load adjoint solution from restart files */
  vector<vector_RCP> getRestartAdjointSolution();
  
  /** @brief Finalize parameter setup */
  void finalizeParams() ;
  
  /** @brief Finalize multiscale coupling */
  void finalizeMultiscale() ;
  
  /**
   * @brief Preconditioned conjugate gradient solve.
   * @param[in] set Physics set index
   * @param[in] J Jacobian matrix
   * @param[in] b RHS vector
   * @param[in,out] x Solution vector
   * @param[in] Minv Preconditioner operator
   * @param[in] tol Tolerance
   * @param[in] maxiter Maximum iterations
   */
  void PCG(const size_t & set, matrix_RCP & J, vector_RCP & b, vector_RCP & x, vector_RCP & Minv,
           const ScalarT & tol, const int & maxiter);
  
  /**
   * @brief Matrix-free PCG solve.
   * @param[in] set Physics set index
   * @param[in] b RHS vector
   * @param[in,out] x Solution vector
   * @param[in] Minv Preconditioner operator
   * @param[in] tol Tolerance
   * @param[in] maxiter Maximum iterations
   */
  void matrixFreePCG(const size_t & set, vector_RCP & b, vector_RCP & x, vector_RCP & Minv,
                     const ScalarT & tol, const int & maxiter);
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Public data members (with inline comments)
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<MpiComm> Comm;                      // MPI communicator for parallel execution
  Teuchos::RCP<Teuchos::ParameterList> settings;   // Solver and physics settings from input
  Teuchos::RCP<MeshInterface>  mesh;               // Geometric and topological mesh information
  Teuchos::RCP<DiscretizationInterface> disc;      // DOF managers and discretization data
  Teuchos::RCP<PhysicsInterface> physics;          // Governs PDE physics and residual construction
  Teuchos::RCP<LinearAlgebraInterface<Node> > linalg; // Interface for linear solvers and operators
  Teuchos::RCP<AssemblyManager<Node> > assembler;  // Assembles residuals and Jacobians
  Teuchos::RCP<ParameterManager<Node> > params;    // Manages model and control parameters
  Teuchos::RCP<PostprocessManager<Node> > postproc; // Handles output and diagnostic evaluations
  Teuchos::RCP<MultiscaleManager> multiscale_manager; // Multiscale coupling data and operations
  Teuchos::RCP<MrHyDE_Debugger> debugger;          // Debugging utilities and timed sections
  
  int verbosity;                                    // Verbosity level for logging
  int batchID;                                      // Batch ID when running batched solves
  int dimension;                                    // Spatial dimension of the problem
  int gNLiter;                                      // Global nonlinear iteration counter
  int maxNLiter;                                    // Maximum allowed nonlinear iterations
  int subcycles;                                    // Number of subcycling steps in explicit schemes
  
  // Time integration configuration per physics set
  vector<int> BDForder;         // Backward-difference order per set
  vector<int> startupBDForder;  // Startup orders until enough steps accumulated
  vector<int> startupSteps;     // Number of startup steps required
  vector<int> numsteps;         // Number of BDF steps (history terms)
  vector<int> numstages;        // Number of RK stages per set
  vector<int> maxnumsteps;      // Maximum steps across sets
  vector<int> maxnumstages;     // Maximum stages across sets
  int numEvaluations;           // Number of RHS evaluations performed
  int maxTimeStepCuts;          // Maximum reductions of time step allowed
  vector<string> ButcherTab;    // RK Butcher tableau per set
  vector<string> startupButcherTab; // Startup RK schemes
  
  ScalarT NLtol;                 // Nonlinear tolerance (relative)
  ScalarT NLabstol;              // Nonlinear tolerance (absolute)
  ScalarT final_time;            // Simulation end time
  ScalarT lintol;        // Linear solver tolerance (redeclared from input)
  ScalarT current_time;          // Current simulation time
  ScalarT initial_time;          // Initial simulation time
  ScalarT deltat;                // Active timestep size
  ScalarT amplification_factor;  // Amplification factor for adaptive time stepping
  
  string solver_type;            // Nonlinear solver type string tag
  string initial_type;           // Initialization method string
  
  bool line_search;              // Whether line search is used for Newton
  bool useL2proj;                // Use L2 projection for some operations
  bool discretized_stochastic;   // Whether stochastic parameters are discretized fields
  bool fully_explicit;           // True if scheme is fully explicit
  bool use_custom_PCG;           // Use customized PCG instead of standard
  bool isInitial;                // Flags first iteration of transient solve
  bool isTransient;              // Indicates transient or steady solve
  bool is_adjoint;               // Whether adjoint system is being solved
  bool is_final_time;            // Whether current step is at final time
  bool usestrongDBCs;            // Use strong Dirichlet enforcement
  bool use_restart = false;      // Use restart solution if available
  bool compute_objective;        // Compute objective functional
  bool use_custom_initial_param_guess; // Whether initial param guess is overridden
  bool store_adjPrev;            // Whether adjoints from previous steps are stored
  bool use_meas_as_dbcs;         // Use measurements as Dirichlet conditions
  bool compute_fwd_sens;         // Whether forward sensitivities are computed
  
  vector<bool> scalarDirichletData;   // True if scalar Dirichlet values exist
  vector<bool> staticDirichletData;   // True if static Dirichlet data provided
  vector<bool> scalarInitialData;     // True if scalar initial conditions exist
  vector<bool> have_initial_conditions;      // Per-physics-set initial condition flags
  vector<bool> have_static_Dirichlet_data;   // Per-set static Dirichlet data flags
  
  bool useRelativeTOL;           // Relative convergence check
  bool useAbsoluteTOL;           // Absolute convergence check
  bool allowBacktracking;        // Allow Newton backtracking
  bool store_vectors;            // Store iteration histories
  bool use_param_mass;           // Use parameter mass matrix
  
  vector<vector<vector<ScalarT>>> scalarDirichletValues; // Dirichlet data per set/block/var
  vector<vector<vector<ScalarT>>> scalarInitialValues;   // Initial state per set/block/var
  
  vector<Teuchos::RCP<LA_MultiVector>> fixedDOF_soln; // Stores fixed DOF solutions
  vector<Teuchos::RCP<LA_MultiVector>> invdiagMass;   // Inverse diagonal mass matrices per set
  vector<Teuchos::RCP<LA_MultiVector>> diagMass;       // Diagonal mass matrices per set
  vector<matrix_RCP> explicitMass;                     // Explicit mass matrices
  
  Teuchos::RCP<LA_MultiVector> diagParamMass; // Diagonal mass for parameters
  matrix_RCP paramMass;                      // Parameter mass matrix
  
  vector<string> blocknames;      // Element block names
  vector<string> setnames;        // Physics set names
  vector<vector<vector<string>>> varlist; // Variable names per set/block
  
  vector<vector<vector<LO>>> numBasis;   // Number of basis functions per var
  vector<vector<size_t>> maxBasis;       // Maximum basis per block
  vector<vector<size_t>> numVars;        // Number of variables per set/block
  vector<vector<vector<LO>>> useBasis;   // Whether basis is actually used
  
  vector<vector_RCP> res;                // Residual vectors
  vector<vector_RCP> res_over;           // Overlapped residuals
  vector<vector_RCP> du;                 // Newton increment
  vector<vector_RCP> du_over;            // Overlapped Newton increment
  vector<vector_RCP> restart_solution;   // Stored restart state
  vector<vector_RCP> restart_adjoint_solution; // Stored restart adjoint
  
  vector<vector_RCP> q_pcg, z_pcg, p_pcg, r_pcg; // PCG storage
  vector<vector_RCP> p_pcg_over, q_pcg_over;     // Overlapped PCG storage
  
  Kokkos::View<ScalarT**,HostDevice> butcher_A; // RK A-matrix
  Kokkos::View<ScalarT*,HostDevice> butcher_b;  // RK b-vector
  Kokkos::View<ScalarT*,HostDevice> butcher_c;  // RK c-vector
  
  vector<vector<vector_RCP>> previous_adjoints;     // Adjoint history
  vector<vector<vector_RCP>> previous_incadjoints;  // Incremental adjoint history
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Private timers (inline comments)
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<Teuchos::Time> transientsolvertimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::transientSolver()"); // Timer for transient solve
  Teuchos::RCP<Teuchos::Time> nonlinearsolvertimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::nonlinearSolver()"); // Timer for nonlinear solver
  Teuchos::RCP<Teuchos::Time> explicitsolvertimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::explicitSolver()"); // Timer for explicit solves
  
  Teuchos::RCP<Teuchos::Time> initsettimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::setInitial()"); // Timer for initial condition setup
  Teuchos::RCP<Teuchos::Time> dbcsettimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::setDirichlet()"); // Timer for Dirichlet setup
  Teuchos::RCP<Teuchos::Time> dbcprojtimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::projectDirichlet()"); // Timer for Dirichlet projection
  Teuchos::RCP<Teuchos::Time> fixeddofsetuptimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::setupFixedDOFs()"); // Timer for fixed DOF initialization
  Teuchos::RCP<Teuchos::Time> msprojtimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::projectDirichlet()"); // Timer for multiscale projection
  Teuchos::RCP<Teuchos::Time> normLAtimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::nonlinearSolver() - norm LA"); // Timer for LA norm step
  Teuchos::RCP<Teuchos::Time> updateLAtimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::nonlinearSolver() - update LA"); // Timer for LA updates
  Teuchos::RCP<Teuchos::Time> PCGtimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::PCG - total"); // Timer for total PCG operations
  Teuchos::RCP<Teuchos::Time> PCGApplyOptimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::PCG - apply Op"); // Timer for PCG operator application
  Teuchos::RCP<Teuchos::Time> PCGApplyPrectimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::PCG - apply prec"); // Timer for PCG preconditioner
  
  Teuchos::RCP<Teuchos::Time> forwardtimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::forward()"); // Timer for forward model
  Teuchos::RCP<Teuchos::Time> adjointtimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::adjoint()"); // Timer for adjoint model
  Teuchos::RCP<Teuchos::Time> transientadjointrhstimer =
  Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::transientSolver() - adjoint RHS"); // Timer for transient solve
  
};

}

#endif
