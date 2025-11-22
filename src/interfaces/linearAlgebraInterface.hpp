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

namespace MrHyDE {
/** \class  LinearSolverOptions
 *  \brief  Stores the specifications for a given linear solver.
 *
 *  This class holds configuration options for Amesos2, Belos, and MueLu
 *  solvers and preconditioners. It also stores reusable solver components
 *  such as Jacobians, symbolic factorizations, and preconditioners.
 *
 *  \tparam Node  Tpetra execution node type.
 */
template<class Node>
class LinearSolverOptions {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
  
public:
  /** \brief Default constructor. */
  LinearSolverOptions() {};
  
  /** \brief Destructor. */
  ~LinearSolverOptions() {};
  
  /** \brief Construct options from a parameter list.
   *  \param settings  Parameter list containing all solver settings.
   */
  LinearSolverOptions(Teuchos::ParameterList & settings) {
    amesos_type = settings.get<string>("Amesos solver","KLU2");
    belos_type = settings.get<string>("Belos solver","Block GMRES");
    belos_sublist = settings.get<string>("Belos settings","Belos Settings");
    prec_sublist = settings.get<string>("Preconditioner settings","Preconditioner Settings");
    
    use_direct = settings.get<bool>("use direct solver",false);
    prec_type = settings.get<string>("preconditioner type","AMG");
    use_preconditioner = settings.get<bool>("use preconditioner",true);
    reuse_preconditioner = settings.get<bool>("reuse preconditioner",true);
    right_preconditioner = settings.get<bool>("right preconditioner",false);
    reuse_jacobian = settings.get<bool>("reuse Jacobian",false);
    
    have_preconditioner = false;
    have_symb_factor = false;
    have_jacobian = false;
  }
  
  // Public data members
  string amesos_type;   /**< Amesos2 solver type (e.g., KLU2). */
  string belos_type;    /**< Belos solver type (e.g., GMRES). */
  string prec_type;     /**< Preconditioner type (e.g., AMG). */
  string belos_sublist; /**< Sublist name for Belos settings. */
  string prec_sublist;  /**< Sublist name for preconditioner settings. */
  bool use_direct;            /**< Use direct Amesos2 solver. */
  bool use_preconditioner;      /**< Whether to apply a preconditioner. */
  bool right_preconditioner;    /**< Whether to apply right preconditioning. */
  bool reuse_preconditioner;    /**< Whether to reuse an existing preconditioner. */
  bool reuse_jacobian;          /**< Whether to reuse an existing Jacobian. */
  bool have_jacobian;          /**< Indicates whether a Jacobian has been constructed. */
  bool have_preconditioner;     /**< Indicates whether a preconditioner exists. */
  bool have_symb_factor;        /**< Indicates whether symbolic factorization exists. */
  bool have_previous_jacobian;  /**< Indicates whether previous Jacobians exist for reuse. */
  
  Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > amesos_solver; /**< Reusable Amesos2 direct solver. */
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > prec; /**< MueLu AMG preconditioner operator. */
  Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > prec_dd; /**< Ifpack2 domain decomposition preconditioner. */
  
  matrix_RCP jac; /**< Current Jacobian matrix. */
  vector<matrix_RCP> jac_prev; /**< Previously stored Jacobians for reuse. */
};

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
  // Note: These are in the header because of the templateing and the objects they return
  //       There is probably a way to have these in the .cpp file, but they are fairly simple
  // ========================================================================================
  
  /**
   * @brief Create a new owned Tpetra multivector for the given physics set.
   * @param set   Index of the physics set.
   * @param numvecs Number of vector columns to allocate.
   * @return Newly allocated multivector with owned map.
   */
  vector_RCP getNewVector(const size_t & set, const int & numvecs = 1) {
    Teuchos::TimeMonitor vectimer(*newvectortimer);
    vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(owned_map[set],numvecs));
    return newvec;
  }
  
  /**
   * @brief Create a new overlapped multivector, or owned if overlap not available.
   * @param set     Physics set index.
   * @param numvecs Number of vector columns.
   * @return New multivector on the overlapped or owned map.
   */
  vector_RCP getNewOverlappedVector(const size_t & set, const int & numvecs = 1){
    Teuchos::TimeMonitor vectimer(*newovervectortimer);
    vector_RCP newvec;
    if (have_overlapped) {
      newvec = Teuchos::rcp(new LA_MultiVector(overlapped_map[set],numvecs));
    }
    else {
      newvec = Teuchos::rcp(new LA_MultiVector(owned_map[set],numvecs));
    }
    return newvec;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Allocate or reuse a Jacobian matrix for a given physics set.
   * @param set Physics set index.
   * @return Newly created or reused matrix.
   */
  matrix_RCP getNewMatrix(const size_t & set) {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    matrix_RCP newmat;
    if (options[set]->reuse_jacobian) {
      if (options[set]->have_jacobian) {
        newmat = options[set]->jac;
      }
      else {
        newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
        options[set]->jac = newmat;
        options[set]->have_jacobian = true;
      }
    }
    else {
      newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
    }
    return newmat;
  }
  
  /**
   * @brief Create matrices for Jacobians associated with previous timesteps (adjoint solves).
   * @param set      Physics set index.
   * @param numsteps Number of previous steps to allocate.
   * @return Vector of Jacobian matrices.
   */
  vector<matrix_RCP> getNewPreviousMatrix(const size_t & set, const size_t & numsteps) {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    vector<matrix_RCP> newmat;
    if (options[set]->reuse_jacobian) {
      if (options[set]->have_previous_jacobian) {
        newmat = options[set]->jac_prev;
      }
      else {
        for (size_t k=0; k<numsteps; ++k) {
          matrix_RCP M = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
          newmat.push_back(M);
        }
        options[set]->jac_prev = newmat;
        options[set]->have_previous_jacobian = true;
      }
    }
    else {
      for (size_t k=0; k<numsteps; ++k) {
        matrix_RCP M = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
        newmat.push_back(M);
      }
    }
    return newmat;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Create a new parameter-space matrix.
   * @return Newly allocated parameter matrix.
   */
  matrix_RCP getNewParamMatrix() {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_owned_map, max_entries));
    return newmat;
  }
  
  /**
   * @brief Create a new parameter–state coupling matrix.
   * @param set Physics set index.
   * @return Newly created state-parameter matrix.
   */
  matrix_RCP getNewParamStateMatrix(const size_t & set) {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_owned_map, max_entries));
    return newmat;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Create a new matrix with variable row sizes.
   * @param set    Physics set index.
   * @param maxent Number of entries per row.
   * @return Newly created matrix.
   */
  matrix_RCP getNewMatrix(const size_t & set, vector<size_t> & maxent) {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], maxent));
    return newmat;
  }
  
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
   * @brief Reset Jacobian-related data for all physics sets.
   */
  void resetJacobian() {
    for (size_t set=0; set<options.size(); ++set) {
      this->resetJacobian(set);
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Reset Jacobian, preconditioner, and symbolic factorization flags for a specific physics set.
   * @param set Index of the physics set.
   */
  void resetJacobian(const size_t & set) {
    options[set]->have_jacobian = false;
    options[set]->have_previous_jacobian = false;
    options[set]->have_preconditioner = false;
    options[set]->jac = Teuchos::null;
    options[set]->prec = Teuchos::null;
    options[set]->prec_dd = Teuchos::null;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Create a new overlapped matrix using the stored overlapped graph.
   * @param set Physics set index.
   * @return Newly allocated overlapped matrix.
   */
  matrix_RCP getNewOverlappedMatrix(const size_t & set) {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(/** @brief Overlapped CRS graphs (owned graphs unused). */
                                                      overlapped_graph[set]));
    return newmat;
  }
  
  // ========================================================================================
  // ========================================================================================
  /**
   * @brief Create a new overlapped rectangular matrix.
   * @param colmap  Column map describing the layout of columns.
   * @param set     Physics set index.
   * @return Newly allocated rectangular CRS matrix on the overlapped map.
   */
  matrix_RCP getNewOverlappedRectangularMatrix(Teuchos::RCP<const LA_Map> & colmap, const size_t & set) {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(/** @brief Overlapped maps for each set. */
                                                      overlapped_map[set], colmap, max_entries));
    return newmat;
  }
  
  // ========================================================================================
  // ========================================================================================
  /**
   * @brief Create a new owned rectangular CRS matrix.
   * @param colmap  Column map defining the column distribution.
   * @param set     Physics set index.
   * @return Newly allocated rectangular matrix on the owned map.
   */
  matrix_RCP getNewRectangularMatrix(Teuchos::RCP<const LA_Map> & colmap, const size_t & set) {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(/** @brief Owned maps for each set. */
                                                      owned_map[set], colmap, max_entries));
    return newmat;
  }
  
  // ========================================================================================
  // Get discretized parameter linear algebra objects
  // ========================================================================================
  /**
   * @brief Create a new parameter multivector using the owned parameter map.
   * @param numvecs  Number of vector components to allocate (default = 1).
   * @return Newly allocated parameter multivector.
   */
  vector_RCP getNewParamVector(const int & numvecs = 1) {
    Teuchos::TimeMonitor vectimer(*newvectortimer);
    vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(param_owned_map, numvecs));
    return newvec;
  }
  
  // ========================================================================================
  // ========================================================================================
  /**
   * @brief Create a new overlapped parameter multivector.
   * @param numvecs  Number of vector columns (default = 1).
   * @return Newly allocated overlapped parameter multivector.
   */
  vector_RCP getNewOverlappedParamVector(const int & numvecs = 1) {
    Teuchos::TimeMonitor vectimer(*newvectortimer);
    vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(param_overlapped_map, numvecs));
    return newvec;
  }
  
  
  /**
   * @brief Create a new overlapped parameter matrix.
   * @return Newly allocated overlapped parameter matrix.
   */
  matrix_RCP getNewOverlappedParamMatrix() {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_overlapped_graph));
    return newmat;
  }
  
  /**
   * @brief Create a new overlapped parameter–state matrix for a given set.
   * @param set Index of the set.
   * @return Newly allocated overlapped parameter–state matrix.
   */
  matrix_RCP getNewOverlappedParamStateMatrix(const size_t & set) {
    Teuchos::TimeMonitor mattimer(*newmatrixtimer);
    matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(paramstate_overlapped_graph[set]));
    return newmat;
  }
  
  /**
   * @brief Export vector from overlapped to owned map using ADD combine mode.
   * @param set Block/set index.
   * @param vec Destination (owned) vector.
   * @param vec_over Source (overlapped) vector.
   */
  void exportVectorFromOverlapped(const size_t & set, vector_RCP & vec, vector_RCP & vec_over) {
    Teuchos::TimeMonitor mattimer(*exporttimer);
    if (comm->getSize() > 1) {
      vec->putScalar(0.0);
      vec->doExport(*vec_over, *(/** @brief Exporters for owned→overlapped communication. */
                                 exporter[set]), Tpetra::ADD);
    }
    else {
      vec->assign(*vec_over);
    }
  }
  
  /**
   * @brief Export vector from overlapped to owned map using REPLACE mode.
   * @param set Block/set index.
   * @param vec Destination (owned) vector.
   * @param vec_over Source (overlapped) vector.
   */
  void exportVectorFromOverlappedReplace(const size_t & set, vector_RCP & vec, vector_RCP & vec_over) {
    Teuchos::TimeMonitor mattimer(*exporttimer);
    if (comm->getSize() > 1) {
      vec->putScalar(0.0);
      vec->doExport(*vec_over, *(exporter[set]), Tpetra::REPLACE);
    }
    else {
      vec->assign(*vec_over);
    }
  }
  
  /**
   * @brief Export parameter vector from overlapped to owned map using ADD mode.
   * @param vec Destination parameter vector.
   * @param vec_over Source overlapped parameter vector.
   */
  void exportParamVectorFromOverlapped(vector_RCP & vec, vector_RCP & vec_over) {
    Teuchos::TimeMonitor mattimer(*exporttimer);
    vec->putScalar(0.0);
    vec->doExport(*vec_over, *param_exporter, Tpetra::ADD);
  }
  
  /**
   * @brief Export parameter vector from overlapped to owned using REPLACE mode.
   * @param vec Destination parameter vector.
   * @param vec_over Source overlapped vector.
   */
  void exportParamVectorFromOverlappedReplace(vector_RCP & vec, vector_RCP & vec_over) {
    Teuchos::TimeMonitor mattimer(*exporttimer);
    vec->putScalar(0.0);
    vec->doExport(*vec_over, *param_exporter, Tpetra::REPLACE);
  }
  
  /**
   * @brief Export matrix from overlapped to owned using ADD mode.
   * @param set Block/set index.
   * @param mat Destination (owned) matrix.
   * @param mat_over Source (overlapped) matrix.
   */
  void exportMatrixFromOverlapped(const size_t & set, matrix_RCP & mat, matrix_RCP & mat_over) {
    Teuchos::TimeMonitor mattimer(*exporttimer);
    mat->setAllToScalar(0.0);
    mat->doExport(*mat_over, *(exporter[set]), Tpetra::ADD);
  }
  
  /**
   * @brief Export parameter–state matrix from overlapped to owned.
   * @param set Block/set index.
   * @param mat Destination matrix.
   * @param mat_over Source overlapped matrix.
   */
  void exportParamStateMatrixFromOverlapped(const size_t & set, matrix_RCP & mat, matrix_RCP & mat_over) {
    Teuchos::TimeMonitor mattimer(*exporttimer);
    mat->setAllToScalar(0.0);
    mat->doExport(*mat_over, *(param_exporter), Tpetra::ADD);
  }
  
  /**
   * @brief Export parameter matrix from overlapped to owned.
   * @param mat Destination matrix.
   * @param mat_over Source overlapped matrix.
   */
  void exportParamMatrixFromOverlapped(matrix_RCP & mat, matrix_RCP & mat_over) {
    Teuchos::TimeMonitor mattimer(*exporttimer);
    mat->setAllToScalar(0.0);
    mat->doExport(*mat_over, *param_exporter, Tpetra::ADD);
  }
  
  /**
   * @brief Import vector from owned to overlapped using ADD mode.
   * @param set Block/set index.
   * @param vec_over Destination overlapped vector.
   * @param vec Source owned vector.
   */
  void importVectorToOverlapped(const size_t & set, vector_RCP & vec_over, const vector_RCP & vec) {
    Teuchos::TimeMonitor mattimer(*importtimer);
    vec_over->putScalar(0.0);
    vec_over->doImport(*vec, *(/** @brief Importers for overlapped→owned communication. */
                               importer[set]), Tpetra::ADD);
  }
  
  /**
   * @brief Finalize fill of a matrix using parameter and owned maps.
   * @param set Index of the discretization set.
   * @param mat Matrix to be completed.
   */
  void fillCompleteParamState(const size_t & set, matrix_RCP & mat) {
    Teuchos::TimeMonitor mattimer(*fillcompletetimer);
    mat->fillComplete(owned_map[set], param_owned_map);
  }
  
  /**
   * @brief Finalize fill of a matrix using its internally stored maps.
   * @param mat Matrix to be completed.
   */
  void fillComplete(matrix_RCP & mat) {
    Teuchos::TimeMonitor mattimer(*fillcompletetimer);
    mat->fillComplete();
  }
  
  /**
   * @brief Get number of locally owned or overlapped elements.
   * @param set Index of the discretization set.
   * @return Local number of elements.
   */
  size_t getLocalNumElements(const size_t & set) {
    size_t numElem = 0;
    if (have_overlapped) {
      numElem = overlapped_map[set]->getLocalNumElements();
    }
    else {
      numElem = owned_map[set]->getLocalNumElements();
    }
    return numElem;
  }
  
  /**
   * @brief Get number of locally owned or overlapped parameter elements.
   * @return Local number of parameter elements.
   */
  size_t getLocalNumParamElements() {
    size_t numElem = 0;
    if (have_overlapped) {
      numElem = param_overlapped_map->getLocalNumElements();
    }
    else {
      numElem = param_owned_map->getLocalNumElements();
    }
    return numElem;
  }
  
  /**
   * @brief Create a new (possibly overlapped) CrsGraph for a set.
   * @param set Index of the discretization set.
   * @param maxEntriesPerRow Maximum entries per row.
   * @return Newly allocated graph.
   */
  Teuchos::RCP<LA_CrsGraph> getNewOverlappedGraph(const size_t & set, vector<size_t> & maxEntriesPerRow) {
    Teuchos::RCP<LA_CrsGraph> newgraph;
    if (have_overlapped) {
      newgraph = Teuchos::rcp(new LA_CrsGraph(overlapped_map[set], maxEntriesPerRow));
    }
    else {
      newgraph = Teuchos::rcp(new LA_CrsGraph(owned_map[set], maxEntriesPerRow));
    }
    return newgraph;
  }
  
  /**
   * @brief Create a new parameter CrsGraph (overlapped or owned).
   * @param maxEntriesPerRow Maximum entries per row.
   * @return Newly allocated parameter graph.
   */
  Teuchos::RCP<LA_CrsGraph> getNewParamOverlappedGraph(vector<size_t> & maxEntriesPerRow) {
    Teuchos::RCP<LA_CrsGraph> newgraph;
    if (have_overlapped) {
      newgraph = Teuchos::rcp(new LA_CrsGraph(param_overlapped_map, maxEntriesPerRow));
    }
    else {
      newgraph = Teuchos::rcp(new LA_CrsGraph(param_owned_map, maxEntriesPerRow));
    }
    return newgraph;
  }
  
  /**
   * @brief Get global ID from a local ID for a given set.
   * @param set Index of the discretization set.
   * @param lid Local index.
   * @return Global index.
   */
  GO getGlobalElement(const size_t & set, const LO & lid) {
    GO gid = 0;
    if (have_overlapped) {
      gid = overlapped_map[set]->getGlobalElement(lid);
    }
    else {
      gid = owned_map[set]->getGlobalElement(lid);
    }
    return gid;
  }
  
  /**
   * @brief Get global parameter ID from local parameter ID.
   * @param lid Local parameter index.
   * @return Global parameter index.
   */
  GO getGlobalParamElement(const LO & lid) {
    GO gid = 0;
    if (have_overlapped) {
      gid = param_overlapped_map->getGlobalElement(lid);
    }
    else {
      gid = param_owned_map->getGlobalElement(lid);
    }
    return gid;
  }
  
  /**
   * @brief Check if overlapped maps are used.
   * @return True if overlapped maps exist.
   */
  bool getHaveOverlapped() {
    return have_overlapped;
  }
  
  /**
   * @brief Get local ID from a global ID using overlapped or owned map.
   * @param set Index of the discretization set.
   * @param gid Global index.
   * @return Local index.
   */
  LO getOverlappedLID(const size_t & set, const GO & gid) {
    LO lid = 0;
    if (have_overlapped) {
      lid = overlapped_map[set]->getLocalElement(gid);
    }
    else {
      lid = owned_map[set]->getLocalElement(gid);
    }
    return lid;
  }
  
  /**
   * @brief Get local ID in owned map only.
   * @param set Index of the discretization set.
   * @param gid Global index.
   * @return Owned local index.
   */
  LO getOwnedLID(const size_t & set, const GO & gid) {
    return owned_map[set]->getLocalElement(gid);
  }
  
  
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
                   const std::string &sol_filename="solution.mm") {
    Teuchos::TimeMonitor localtimer(*writefiletimer);
    
    if(do_dump_jacobian)
      Tpetra::MatrixMarket::Writer<LA_CrsMatrix>::writeSparseFile(jac_filename,*J);
    if(do_dump_residual)
      Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(res_filename,*r);
    if(do_dump_solution)
      Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(sol_filename,*soln);
  }
  
  // ========================================================================================
  // Belos solver parameter list accessor
  // ========================================================================================
  
  /**
   * @brief Retrieve a Belos solver parameter sublist from the global parameter list.
   *
   * @param belosSublist  Name of the Belos sublist to retrieve.
   * @return RCP to the corresponding parameter list.
   */
  Teuchos::RCP<Teuchos::ParameterList> getBelosParameterList(const string & belosSublist);
  
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
  void linearSolver(Teuchos::RCP<LinearSolverOptions<Node> > & opt,
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
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT,LO,GO,Node> >
  buildPreconditioner(const matrix_RCP & J, const string & precSublist);
  
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  vector<Teuchos::RCP<const LA_Map> > owned_map;              //!< Owned (non-overlapped) maps for each equation set.
  vector<Teuchos::RCP<const LA_Map> > overlapped_map;          //!< Overlapped (ghosted) maps for each equation set.
  vector<Teuchos::RCP<LA_CrsGraph> > overlapped_graph;         //!< Overlapped sparsity graphs (owned graphs unused).
  vector<Teuchos::RCP<LA_Export> > exporter;                   //!< Exporters for owned → overlapped transfer.
  vector<Teuchos::RCP<LA_Import> > importer;                   //!< Importers for overlapped → owned transfer.
  
  vector<Teuchos::RCP<LinearSolverOptions<Node> > > options;        //!< Solver options for standard Jacobian solves.
  vector<Teuchos::RCP<LinearSolverOptions<Node> > > options_L2;     //!< Solver options for L2 projection solves.
  vector<Teuchos::RCP<LinearSolverOptions<Node> > > options_BndryL2;//!< Solver options for boundary L2 projection solves.
  
  Teuchos::RCP<LinearSolverOptions<Node> > options_param;           //!< Solver options for discretized parameter solves.
  Teuchos::RCP<LinearSolverOptions<Node> > options_param_L2;        //!< Solver options for L2 parameter projection solves.
  Teuchos::RCP<LinearSolverOptions<Node> > options_param_BndryL2;   //!< Solver options for boundary L2 parameter solves.
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Private data members
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
  
  vector<matrix_RCP> matrix;                                  //!< Owned matrices for each equation set.
  vector<matrix_RCP> overlapped_matrix;                       //!< Overlapped matrices (ghosted).
  
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
  Teuchos::RCP<Teuchos::Time> linearsolvertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::linearSolver*()");
  Teuchos::RCP<Teuchos::Time> fillcompletetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::fillComplete*()");
  Teuchos::RCP<Teuchos::Time> exporttimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::export*()");
  Teuchos::RCP<Teuchos::Time> importtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::import*()");
  Teuchos::RCP<Teuchos::Time> prectimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::buildPreconditioner()");
};

}

#endif
