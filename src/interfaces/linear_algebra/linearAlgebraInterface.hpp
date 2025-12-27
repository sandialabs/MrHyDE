/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

/** \file   linearAlgebraInterface.hpp
 *  \brief  Contains the interface to the linear algebra tools from Trilinos.
 *  \author Created by T. Wildey
 */

#ifndef MRHYDE_LINEAR_ALGEBRA_H
#define MRHYDE_LINEAR_ALGEBRA_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "discretizationInterface.hpp"
#include "parameterManager.hpp"
#include "MrHyDE_Debugger.hpp"

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

// Amesos includes
#include "Amesos2.hpp"

// Options for various linear solvers
#include "linearSolverContext.hpp"

namespace MrHyDE {

/** \class LinearAlgebraInterface
 *  \brief Interface wrapper to Tpetra, Belos, MueLu, and Amesos2.
 *
 *  Provides helper routines for matrix assembly, linear solves, graph
 *  construction, preconditioner management, and distributed vector
 *  operations.
 *
 *  \tparam Node  Tpetra execution node type.
 */
template<class Node>
class LinearAlgebraInterface {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Tpetra::CrsGraph<LO,GO,Node>            LA_CrsGraph;
  typedef Tpetra::Export<LO,GO,Node>              LA_Export;
  typedef Tpetra::Import<LO,GO,Node>              LA_Import;
  typedef Tpetra::Map<LO,GO,Node>                 LA_Map;
  typedef Tpetra::Operator<ScalarT,LO,GO,Node>    LA_Operator;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
  typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
  typedef typename Node::device_type              LA_device;
  typedef Belos::LinearProblem<ScalarT,LA_MultiVector,LA_Operator> LA_LinearProblem;
  
public:
  /** \brief Default constructor. */
  LinearAlgebraInterface() {};
  
  /** \brief Destructor. */
  ~LinearAlgebraInterface() {};
  
  /** \brief Construct from MPI communicator, settings, discretization, and parameters.
   *  \param Comm_    MPI communicator.
   *  \param settings_  Parameter list of solver and algebra settings.
   *  \param disc_      Discretization interface.
   *  \param params_    Parameter manager.
   */
  LinearAlgebraInterface(const Teuchos::RCP<MpiComm> & Comm_,
                         Teuchos::RCP<Teuchos::ParameterList> & settings_,
                         Teuchos::RCP<DiscretizationInterface> & disc_,
                         Teuchos::RCP<ParameterManager<Node> > & params_);
  
  // ========================================================================================
  // ========================================================================================
  /**
   * @brief Set up linear algebra data structures, maps, graphs, and import/export objects.
   */
  void setupLinearAlgebra();
  
  // ========================================================================================
  // Get physics state linear algebra objects
  // ========================================================================================
  
  /**
   * @brief Create a new owned Tpetra multivector for the given physics set.
   * @param set   Index of the physics set.
   * @param numvecs Number of vector columns to allocate.
   * @return Newly allocated multivector with owned map.
   */
  vector_RCP getNewVector(const size_t & set, const int & numvecs = 1);
  
  /**
   * @brief Create a new overlapped multivector, or owned if overlap not available.
   * @param set     Physics set index.
   * @param numvecs Number of vector columns.
   * @return New multivector on the overlapped or owned map.
   */
  vector_RCP getNewOverlappedVector(const size_t & set, const int & numvecs = 1);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Allocate or reuse a Jacobian matrix for a given physics set.
   * @param set Physics set index.
   * @return Newly created or reused matrix.
   */
  matrix_RCP getNewMatrix(const size_t & set);
  
  /**
   * @brief Allocate or reuse a Jacobian matrix for a given physics set.
   * @param set Physics set index.
   * @return Newly created or reused matrix.
   */
  matrix_RCP getNewL2Matrix(const size_t & set);
  
  /**
   * @brief Allocate or reuse a Jacobian matrix for a given physics set.
   * @param set Physics set index.
   * @return Newly created or reused matrix.
   */
  matrix_RCP getNewBndryL2Matrix(const size_t & set);
  
  /**
   * @brief Create matrices for Jacobians associated with previous timesteps (adjoint solves).
   * @param set      Physics set index.
   * @param numsteps Number of previous steps to allocate.
   * @return Vector of Jacobian matrices.
   */
  vector<matrix_RCP> getNewPreviousMatrix(const size_t & set, const size_t & numsteps);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Create a new parameter-space matrix.
   * @return Newly allocated parameter matrix.
   */
  matrix_RCP getNewParamMatrix();
  
  /**
   * @brief Create a new parameter–state coupling matrix.
   * @param set Physics set index.
   * @return Newly created state-parameter matrix.
   */
  matrix_RCP getNewParamStateMatrix(const size_t & set);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Create a new matrix with variable row sizes.
   * @param set    Physics set index.
   * @param maxent Number of entries per row.
   * @return Newly created matrix.
   */
  matrix_RCP getNewMatrix(const size_t & set, vector<size_t> & maxent);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Query whether the Jacobian for this set can be reused.
   * @param set Physics set index.
   * @return True if reuse is enabled and a Jacobian exists.
   */
  bool getJacobianReuse(const size_t & set);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Query whether the parameter Jacobian can be reused.
   * @return True if reuse is enabled and a Jacobian exists.
   */
  bool getParamJacobianReuse();
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Query whether the parameter/state Jacobian for this set can be reused.
   * @param set Physics set index.
   * @return True if reuse is enabled and a Jacobian exists.
   */
  bool getParamStateJacobianReuse(const size_t & set);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Reset Jacobian-related data for all physics sets.
   */
  void resetAllJacobian();
  
  /**
   * @brief Reset Jacobian-related data for all physics sets.
   */
  void resetJacobian();
  
  /**
   * @brief Reset Jacobian-related data for all physics sets.
   */
  void resetL2Jacobian();
  
  /**
   * @brief Reset Jacobian-related data for all physics sets.
   */
  void resetBndryL2Jacobian();
  
  /**
   * @brief Reset Jacobian-related data for all physics sets.
   */
  void resetParamJacobian();
  
  /**
   * @brief Reset Jacobian-related data for all physics sets.
   */
  void resetParamStateJacobian();
  
  /**
   * @brief Reset Jacobian-related data for all physics sets.
   */
  void resetPrevJacobian();
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Create a new overlapped matrix using the stored overlapped graph.
   * @param set Physics set index.
   * @return Newly allocated overlapped matrix.
   */
  matrix_RCP getNewOverlappedMatrix(const size_t & set);
  
  // ========================================================================================
  // ========================================================================================
  /**
   * @brief Create a new overlapped rectangular matrix.
   * @param colmap  Column map describing the layout of columns.
   * @param set     Physics set index.
   * @return Newly allocated rectangular CRS matrix on the overlapped map.
   */
  matrix_RCP getNewOverlappedRectangularMatrix(Teuchos::RCP<const LA_Map> & colmap, const size_t & set);
  
  // ========================================================================================
  // ========================================================================================
  /**
   * @brief Create a new owned rectangular CRS matrix.
   * @param colmap  Column map defining the column distribution.
   * @param set     Physics set index.
   * @return Newly allocated rectangular matrix on the owned map.
   */
  matrix_RCP getNewRectangularMatrix(Teuchos::RCP<const LA_Map> & colmap, const size_t & set);
  
  // ========================================================================================
  // Get discretized parameter linear algebra objects
  // ========================================================================================
  /**
   * @brief Create a new parameter multivector using the owned parameter map.
   * @param numvecs  Number of vector components to allocate (default = 1).
   * @return Newly allocated parameter multivector.
   */
  vector_RCP getNewParamVector(const int & numvecs = 1);
  
  // ========================================================================================
  // ========================================================================================
  /**
   * @brief Create a new overlapped parameter multivector.
   * @param numvecs  Number of vector columns (default = 1).
   * @return Newly allocated overlapped parameter multivector.
   */
  vector_RCP getNewOverlappedParamVector(const int & numvecs = 1);
  
  /**
   * @brief Create a new overlapped parameter matrix.
   * @return Newly allocated overlapped parameter matrix.
   */
  matrix_RCP getNewOverlappedParamMatrix();
  
  /**
   * @brief Create a new overlapped parameter–state matrix for a given set.
   * @param set Index of the set.
   * @return Newly allocated overlapped parameter–state matrix.
   */
  matrix_RCP getNewOverlappedParamStateMatrix(const size_t & set);
  
  /**
   * @brief Export vector from overlapped to owned map using ADD combine mode.
   * @param set Block/set index.
   * @param vec Destination (owned) vector.
   * @param vec_over Source (overlapped) vector.
   */
  void exportVectorFromOverlapped(const size_t & set, vector_RCP & vec, vector_RCP & vec_over);
  
  /**
   * @brief Export vector from overlapped to owned map using REPLACE mode.
   * @param set Block/set index.
   * @param vec Destination (owned) vector.
   * @param vec_over Source (overlapped) vector.
   */
  void exportVectorFromOverlappedReplace(const size_t & set, vector_RCP & vec, vector_RCP & vec_over);
  
  /**
   * @brief Export parameter vector from overlapped to owned map using ADD mode.
   * @param vec Destination parameter vector.
   * @param vec_over Source overlapped parameter vector.
   */
  void exportParamVectorFromOverlapped(vector_RCP & vec, vector_RCP & vec_over);
  
  /**
   * @brief Export parameter vector from overlapped to owned using REPLACE mode.
   * @param vec Destination parameter vector.
   * @param vec_over Source overlapped vector.
   */
  void exportParamVectorFromOverlappedReplace(vector_RCP & vec, vector_RCP & vec_over);
  
  /**
   * @brief Export matrix from overlapped to owned using ADD mode.
   * @param set Block/set index.
   * @param mat Destination (owned) matrix.
   * @param mat_over Source (overlapped) matrix.
   */
  void exportMatrixFromOverlapped(const size_t & set, matrix_RCP & mat, matrix_RCP & mat_over);
  
  /**
   * @brief Export parameter–state matrix from overlapped to owned.
   * @param set Block/set index.
   * @param mat Destination matrix.
   * @param mat_over Source overlapped matrix.
   */
  void exportParamStateMatrixFromOverlapped(const size_t & set, matrix_RCP & mat, matrix_RCP & mat_over);
  
  /**
   * @brief Export parameter matrix from overlapped to owned.
   * @param mat Destination matrix.
   * @param mat_over Source overlapped matrix.
   */
  void exportParamMatrixFromOverlapped(matrix_RCP & mat, matrix_RCP & mat_over);
  
  /**
   * @brief Import vector from owned to overlapped using ADD mode.
   * @param set Block/set index.
   * @param vec_over Destination overlapped vector.
   * @param vec Source owned vector.
   */
  void importVectorToOverlapped(const size_t & set, vector_RCP & vec_over, const vector_RCP & vec);
  
  /**
   * @brief Finalize fill of a matrix using parameter and owned maps.
   * @param set Index of the discretization set.
   * @param mat Matrix to be completed.
   */
  void fillCompleteParamState(const size_t & set, matrix_RCP & mat);
  
  /**
   * @brief Finalize fill of a matrix using its internally stored maps.
   * @param mat Matrix to be completed.
   */
  void fillComplete(matrix_RCP & mat);
  
  /**
   * @brief Get number of locally owned or overlapped elements.
   * @param set Index of the discretization set.
   * @return Local number of elements.
   */
  size_t getLocalNumElements(const size_t & set);
  
  /**
   * @brief Get number of locally owned or overlapped parameter elements.
   * @return Local number of parameter elements.
   */
  size_t getLocalNumParamElements();
  
  /**
   * @brief Create a new (possibly overlapped) CrsGraph for a set.
   * @param set Index of the discretization set.
   * @param maxEntriesPerRow Maximum entries per row.
   * @return Newly allocated graph.
   */
  Teuchos::RCP<LA_CrsGraph> getNewOverlappedGraph(const size_t & set, vector<size_t> & maxEntriesPerRow);
  
  /**
   * @brief Create a new parameter CrsGraph (overlapped or owned).
   * @param maxEntriesPerRow Maximum entries per row.
   * @return Newly allocated parameter graph.
   */
  Teuchos::RCP<LA_CrsGraph> getNewParamOverlappedGraph(vector<size_t> & maxEntriesPerRow);
  
  /**
   * @brief Get global ID from a local ID for a given set.
   * @param set Index of the discretization set.
   * @param lid Local index.
   * @return Global index.
   */
  GO getGlobalElement(const size_t & set, const LO & lid);
  
  /**
   * @brief Get global parameter ID from local parameter ID.
   * @param lid Local parameter index.
   * @return Global parameter index.
   */
  GO getGlobalParamElement(const LO & lid);
  
  /**
   * @brief Check if overlapped maps are used.
   * @return True if overlapped maps exist.
   */
  bool getHaveOverlapped();
  
  /**
   * @brief Get local ID from a global ID using overlapped or owned map.
   * @param set Index of the discretization set.
   * @param gid Global index.
   * @return Local index.
   */
  LO getOverlappedLID(const size_t & set, const GO & gid);
  
  /**
   * @brief Get local ID in owned map only.
   * @param set Index of the discretization set.
   * @param gid Global index.
   * @return Owned local index.
   */
  LO getOwnedLID(const size_t & set, const GO & gid);
  
  // ========================================================================================
  // Write the Jacobian and/or residual to a matrix-market text file
  // ========================================================================================
  
  /**
   * @brief Write Jacobian, residual, and/or solution vectors to MatrixMarket files.
   *
   * This routine optionally writes the Jacobian matrix, the residual vector, and the
   * solution vector to disk in MatrixMarket format.
   * WARNING: Tpetra gathers full data to rank 0 during writing, so very large matrices
   * may cause memory exhaustion.
   *
   * @param J             Jacobian matrix to write.
   * @param r             Residual vector to write.
   * @param soln          Solution vector to write.
   * @param jac_filename  Filename for Jacobian (default: "jacobian.mm").
   * @param res_filename  Filename for residual (default: "residual.mm").
   * @param sol_filename  Filename for solution (default: "solution.mm").
   */
  void writeToFile(matrix_RCP &J, vector_RCP &r, vector_RCP &soln,
                   const std::string &jac_filename="jacobian.mm",
                   const std::string &res_filename="residual.mm",
                   const std::string &sol_filename="solution.mm");
  
  void writeVectorToFile(ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > & vec, string & filename);
    
  void writeStateToFile(vector<vector_RCP> & soln, const std::string & filebase, const int & stepnum);

  vector_RCP readParameterVectorFromFile(const std::string & filename);
  
  vector_RCP readStateVectorFromFile(const std::string & filename, const size_t & set);
    
  // ========================================================================================
  // Belos solver parameter list accessor
  // ========================================================================================
  
  /**
   * @brief Retrieve a Belos solver parameter sublist from the global parameter list.
   *
   * @param belosSublist  Name of the Belos sublist to retrieve.
   * @return RCP to the corresponding parameter list.
   */
  Teuchos::RCP<Teuchos::ParameterList> getBelosParameterList(Teuchos::RCP<LinearSolverContext<Node> > & cntxt);
  
  // ========================================================================================
  // Linear solver on Tpetra stack for Jacobians of states
  // ========================================================================================
  
  /**
   * @brief Solve a linear system J * soln = r for state Jacobians using user-specified options.
   *
   * @param opt   Linear solver options object (Belos/MueLu settings, etc.).
   * @param J     Jacobian matrix.
   * @param r     Right-hand side (residual vector).
   * @param soln  Solution vector to be filled.
   */
  void linearSolver(Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                    matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  /**
   * @brief Solve a linear system J * soln = r for the state Jacobian associated with a set index.
   *
   * @param set   Index for the current variable set.
   * @param J     Jacobian matrix.
   * @param r     Right-hand side vector.
   * @param soln  Solution vector to be filled.
   */
  void linearSolver(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  // ========================================================================================
  // Linear solver for parameter Jacobians
  // ========================================================================================
  
  /**
   * @brief Solve a linear system associated with discretized parameter Jacobians.
   *
   * @param J     Parameter Jacobian matrix.
   * @param r     Right-hand side vector.
   * @param soln  Solution vector.
   */
  void linearSolverParam(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  // ========================================================================================
  // Linear solver for boundary L2 projections (Dirichlet BCs)
  // ========================================================================================
  
  /**
   * @brief Solve a boundary L2 projection linear system for state variables.
   *
   * @param set   Variable set index.
   * @param J     System matrix.
   * @param r     RHS vector.
   * @param soln  Solution vector.
   */
  void linearSolverBoundaryL2(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  /**
   * @brief Solve a boundary L2 projection linear system for discretized parameters.
   *
   * @param J     System matrix.
   * @param r     RHS vector.
   * @param soln  Solution vector.
   */
  void linearSolverBoundaryL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  // ========================================================================================
  // Linear solver for L2 projection (initial conditions)
  // ========================================================================================
  
  /**
   * @brief Solve an L2 projection linear system for state variables (e.g., initial conditions).
   *
   * @param set   Variable set index.
   * @param J     System matrix.
   * @param r     RHS vector.
   * @param soln  Solution vector.
   */
  void linearSolverL2(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  /**
   * @brief Solve an L2 projection linear system for discretized parameters.
   *
   * @param J     System matrix.
   * @param r     RHS vector.
   * @param soln  Solution vector.
   */
  void linearSolverL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  // ========================================================================================
  // Preconditioner for Tpetra stack
  // ========================================================================================
  
  /**
   * @brief Build a MueLu preconditioner for a given matrix.
   *
   * @param J            System matrix to precondition.
   * @param precSublist  Name of the MueLu sublist in the parameter list.
   * @return RCP to a MueLu preconditioner operator.
   */
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT,LO,GO,Node> > buildAMGPreconditioner(const matrix_RCP & J,
                                                                                  const Teuchos::RCP<LinearSolverContext<Node> > & cntxt);
  
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  vector<Teuchos::RCP<const LA_Map> > owned_map;              //!< Owned (non-overlapped) maps for each equation set.
  vector<Teuchos::RCP<const LA_Map> > overlapped_map;          //!< Overlapped (ghosted) maps for each equation set.
  vector<Teuchos::RCP<LA_CrsGraph> > overlapped_graph;         //!< Overlapped sparsity graphs (owned graphs unused).
  vector<Teuchos::RCP<LA_Export> > exporter;                   //!< Exporters for owned → overlapped transfer.
  vector<Teuchos::RCP<LA_Import> > importer;                   //!< Importers for overlapped → owned transfer.
  
  vector<Teuchos::RCP<LinearSolverContext<Node> > > context;        //!< Solver context for standard Jacobian solves.
  vector<Teuchos::RCP<LinearSolverContext<Node> > > context_L2;     //!< Solver context for L2 projection solves.
  vector<Teuchos::RCP<LinearSolverContext<Node> > > context_BndryL2;//!< Solver context for boundary L2 projection solves.
  vector<vector<Teuchos::RCP<LinearSolverContext<Node> > > > context_prev;        //!< Solver context for previous Jacobians.
  Teuchos::RCP<LinearSolverContext<Node> > context_param;           //!< Solver context for discretized parameter solves.
  Teuchos::RCP<LinearSolverContext<Node> > context_param_L2;        //!< Solver context for L2 parameter projection solves.
  Teuchos::RCP<LinearSolverContext<Node> > context_param_BndryL2;   //!< Solver context for boundary L2 parameter solves.
  vector<Teuchos::RCP<LinearSolverContext<Node> > > context_param_state;        //!< Solver context for standard Jacobian solves.
  //!
  ///////////////////////////////////////////////////////////////////////////////////////////
  // (could be) Private data members
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<MpiComm> comm;                                 //!< MPI communicator for parallel linear algebra.
  Teuchos::RCP<Teuchos::ParameterList> settings;              //!< Global settings including solver/L.A. parameters.
  Teuchos::RCP<DiscretizationInterface> disc;                 //!< Mesh and DOF discretization interface.
  Teuchos::RCP<ParameterManager<Node> > params;               //!< Parameter manager (continuous/discretized).
  Teuchos::RCP<MrHyDE_Debugger> debugger;                     //!< Debugging and diagnostics interface.
  
  int verbosity;                                              //!< Verbosity level for logging and solver output.
  vector<string> setnames;                                    //!< Names of all equation sets handled.
  bool do_dump_jacobian, do_dump_residual, do_dump_solution;  //!< Flags controlling matrix-market output.
  bool have_overlapped;                                       //!< True if overlapped data structures are enabled.
  
  // Maps, graphs, importers and exporters
  size_t max_entries;                                         //!< Max nonzeros allocated per matrix row.
  
  Teuchos::RCP<const LA_Map> param_owned_map;                 //!< Owned map for discretized parameters.
  Teuchos::RCP<const LA_Map> param_overlapped_map;            //!< Overlapped map for discretized parameters.
  Teuchos::RCP<LA_CrsGraph> param_overlapped_graph;           //!< Overlapped sparsity graph for parameters.
  Teuchos::RCP<LA_Export> param_exporter;                     //!< Exporter for param owned → overlapped.
  Teuchos::RCP<LA_Import> param_importer;                     //!< Importer for param overlapped → owned.
  
  vector<Teuchos::RCP<const LA_Map> > paramstate_owned_map;   //!< Owned maps for parameter–state coupling.
  vector<Teuchos::RCP<const LA_Map> > paramstate_overlapped_map; //!< Overlapped maps for parameter–state coupling.
  vector<Teuchos::RCP<LA_CrsGraph> > paramstate_overlapped_graph; //!< Overlapped graphs for parameter–state systems.
  
  //vector<matrix_RCP> matrix;                                  //!< Owned matrices for each equation set.
  //vector<matrix_RCP> overlapped_matrix;                       //!< Overlapped matrices (ghosted).
  
  // Linear solvers and preconditioner settings
  int maxLinearIters;                                         //!< Maximum number of solver iterations.
  int maxKrylovVectors;                                       //!< Maximum number of Krylov vectors (restarts).
  string belos_residual_scaling;                              //!< Belos residual scaling setting.
  ScalarT linearTOL;                                          //!< Solver tolerance for linear solves.
  bool doCondEst;                                             //!< Whether to compute condition number estimates.
  
  // Timers
  Teuchos::RCP<Teuchos::Time> setupLAtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::setup");
  Teuchos::RCP<Teuchos::Time> newvectortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNewVector()");
  Teuchos::RCP<Teuchos::Time> newovervectortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNewOverlappedVector()");
  Teuchos::RCP<Teuchos::Time> newmatrixtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNew*Matrix()");
  Teuchos::RCP<Teuchos::Time> writefiletimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::write()");
  Teuchos::RCP<Teuchos::Time> readfiletimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::read()");
  Teuchos::RCP<Teuchos::Time> linearsolvertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::linearSolver*()");
  Teuchos::RCP<Teuchos::Time> fillcompletetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::fillComplete*()");
  Teuchos::RCP<Teuchos::Time> exporttimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::export*()");
  Teuchos::RCP<Teuchos::Time> importtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::import*()");
  Teuchos::RCP<Teuchos::Time> prectimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::buildPreconditioner()");
};

}

#endif
