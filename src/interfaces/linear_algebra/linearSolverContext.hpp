/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_LINEAR_ALGEBRA_OPTS_H
#define MRHYDE_LINEAR_ALGEBRA_OPTS_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "block_prec/ParamUtils.hpp"
#include <cctype>

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>
#include <MueLu_RefMaxwell.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>

// Amesos includes
#include "Amesos2.hpp"

namespace MrHyDE {

/** \brief Schur approximation and block-triangular options (variant, damping, pivot block, strictness). */
struct SchurConfig {
  /** Canonical Schur approximation family (currently base or diag). */
  std::string approximation_type;
  /** Active Schur variant used by assembly path (kept in sync with approximation_type). */
  std::string variant;
  /**< Pivot block index p. pivot_block=0 -> Schur on block 1: S = J11 - J10*inv(J00)*J01.
   *   pivot_block=1 -> Schur on block 0: S = J00 - J01*inv(J11)*J10. */
  int pivot_block;
  /**< Gamma in diag Schur. E.g. pivot_block=0: S = J11 - gamma*J10*diag(J00)^{-1}*J01. */
  ScalarT damping;
  bool diag_use_lumped_pivot_diagonal;
  std::string pivot_block_preconditioner_type;
  bool pivot_block_diag_use_lumped_diagonal;
  std::string schur_block_preconditioner_type;
};

/** \brief RefMaxwell auxiliary matrices and vectors (D0, M1, coords, nullspace) and debug/strict flags. */
template<class Node>
struct RefMaxwellData {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
  typedef typename Teuchos::ScalarTraits<ScalarT>::coordinateType CoordScalar;
  typedef Tpetra::MultiVector<CoordScalar,LO,GO,Node> LA_CoordMultiVector;
  Teuchos::RCP<LA_CrsMatrix> D0_matrix;   /**< Discrete gradient (HGRAD -> HCURL). */
  Teuchos::RCP<LA_CrsMatrix> M1_matrix;   /**< Edge mass matrix for HCURL block. */
  Teuchos::RCP<LA_CrsMatrix> D1_matrix;
  Teuchos::RCP<LA_CrsMatrix> M2_matrix;
  Teuchos::RCP<LA_CoordMultiVector> nodal_coords;
  Teuchos::RCP<LA_MultiVector> nullspace;
  Teuchos::RCP<LA_MultiVector> ads_null11;
  Teuchos::RCP<LA_MultiVector> ads_null22;
  bool strict_refmaxwell;  /**< Enforce that pivot block preconditioner is RefMaxwell when strict mode is active. */
  std::string xml_param_file = "";  /**< Path to XML parameter file for RefMaxwell configuration. If provided, XML is used. */
};

/**
 * \struct AMGData
 * \brief Configuration data for MueLu AMG (Algebraic MultiGrid) preconditioner.
 *
 * Stores AMG-specific configuration. If xml_param_file is non-empty, parameters are loaded from XML.
 */
struct AMGData {
  std::string xml_param_file = "";  /**< Path to XML parameter file for AMG configuration. If provided, XML is used. */
};

/** \class  LinearSolverContext
 *  \brief  Stores the specifications for a given linear solver.
 *
 *  This class holds configuration options for Amesos2, Belos, and MueLu
 *  solvers and preconditioners. It also stores reusable solver components
 *  such as matrices,  symbolic factorizations, and preconditioners.
 *
 *  The linear algebra interface holds multiple contexts - one for each type of matrix that might be used.
 *
 *  \tparam Node  Tpetra execution node type.
 */
template<class Node>
class LinearSolverContext {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
  typedef typename Teuchos::ScalarTraits<ScalarT>::coordinateType CoordScalar;
  typedef Tpetra::MultiVector<CoordScalar,LO,GO,Node> LA_CoordMultiVector;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
  
public:
  /** \brief Default constructor. */
  LinearSolverContext() {};
  
  /** \brief Destructor. */
  ~LinearSolverContext() {};
  
  /** \brief Construct options from a parameter list.
   *  \param settings  Parameter list containing all solver settings.
   */
  LinearSolverContext(Teuchos::ParameterList & settings) {
    // Parse order is intentional: discover/validate sublists first, then root defaults,
    // then block-specific overrides, then strict-RefMaxwell inference.
    parseBelosAndAmesosSettings(settings);
    parseSublists(settings);
    validateSublists();
    parseGeneralSettings(settings);
    bool strictRefMaxwellSetExplicitly = parseStrictRefMaxwellFromRoot(settings);
    parsePreconditionerSublist(strictRefMaxwellSetExplicitly);
    parsePivotBlockSublist(strictRefMaxwellSetExplicitly);
    parseSchurBlockSublist(strictRefMaxwellSetExplicitly);
    inferStrictRefMaxwell(strictRefMaxwellSetExplicitly);
    initializeRuntimeState();
  }

  void reset() {
    have_matrix = false;
    have_preconditioner = false;
    matrix = Teuchos::null;
    prec = Teuchos::null;
    prec_dd = Teuchos::null;
    prec_block = Teuchos::null;
    refmaxwell_prec = Teuchos::null;
    schur_refmaxwell_prec = Teuchos::null;
    jacobian_rebuilt_this_step = true;
  }
  
  // Public data members
  string amesos_type;   /**< Amesos2 solver type (e.g., KLU2). */
  string belos_type;    /**< Belos solver type (e.g., GMRES). */
  string prec_type;     /**< Preconditioner type (e.g., AMG). */
  bool use_direct;            /**< Use direct Amesos2 solver. */
  bool use_preconditioner;      /**< Whether to apply a preconditioner. */
  bool right_preconditioner;    /**< Whether to apply right preconditioning. */
  bool reuse_preconditioner;    /**< Whether to reuse an existing preconditioner. */
  string block_prec_backend;    /**< Block-preconditioner backend (mrhyde, teko_hybrid, teko_full). */
  string preconditioner_reuse_type; /**< Reuse mode (none, update, or full). */
  bool reuse_matrix;          /**< Whether to reuse an existing Jacobian. */
  bool jacobian_rebuilt_this_step; /**< True when Jacobian values were rebuilt before this linear solve. */
  bool have_matrix;           /**< Indicates whether a Jacobian has been constructed. */
  bool have_preconditioner;     /**< Indicates whether a preconditioner exists. */
  bool have_symb_factor;        /**< Indicates whether symbolic factorization exists. */
  /**< Grouped Schur and block-tri options (variant, damping, pivot block, strictness). */
  SchurConfig schur;
  /**< AMG preconditioner configuration data including XML parameter support. */
  AMGData amg;
  /**< RefMaxwell matrices/vectors (D0, M1, coords, nullspace) and debug/strict flags. */
  RefMaxwellData<Node> refMaxwell;

  Teuchos::ParameterList prec_sublist, belos_sublist;
  Teuchos::ParameterList pivot_block_sublist, schur_block_sublist;

  Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > amesos_solver; /**< Reusable Amesos2 direct solver. */
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > prec; /**< MueLu AMG preconditioner operator. */
  Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > prec_dd; /**< Ifpack2 domain decomposition preconditioner. */
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > prec_block; /**< Block-diagonal AMG preconditioner operator. */

  matrix_RCP matrix; /**< Current Jacobian matrix. */

  // Cached RefMaxwell preconditioner for reuse.
  Teuchos::RCP<MueLu::RefMaxwell<ScalarT, LO, GO, Node> > refmaxwell_prec;
  Teuchos::RCP<MueLu::RefMaxwell<ScalarT, LO, GO, Node> > schur_refmaxwell_prec; /**< Cached ADS for Schur. */

  size_t equation_set_index; /**< Set index when linearSolver(set,...) is used; for block prec. */

private:
  void parseBelosAndAmesosSettings(Teuchos::ParameterList & settings) {
    amesos_type = settings.get<string>("Amesos solver","KLU2");
    belos_type = settings.get<string>("Belos solver","Block GMRES");
  }

  void parseSublists(Teuchos::ParameterList & settings) {
    belos_sublist = settings.isSublist("Belos Settings")
      ? settings.sublist("Belos Settings")
      : Teuchos::ParameterList("empty");
    prec_sublist = settings.isSublist("Preconditioner Settings")
      ? settings.sublist("Preconditioner Settings")
      : Teuchos::ParameterList("empty");
    pivot_block_sublist = settings.isSublist("Pivot Block Settings")
      ? settings.sublist("Pivot Block Settings")
      : Teuchos::ParameterList("empty");
    schur_block_sublist = settings.isSublist("Schur Block Settings")
      ? settings.sublist("Schur Block Settings")
      : Teuchos::ParameterList("empty");
  }

  void validateSublists() {
    if (prec_sublist.name() != "empty") {
      validatePreconditionerSettingsSection(prec_sublist, "Preconditioner Settings");
    }
    if (pivot_block_sublist.name() != "empty") {
      validatePivotBlockSettingsSection(pivot_block_sublist, "Pivot Block Settings");
    }
    if (schur_block_sublist.name() != "empty") {
      validateSchurBlockSettingsSection(schur_block_sublist, "Schur Block Settings");
    }
  }

  void parseGeneralSettings(Teuchos::ParameterList & settings) {
    use_direct = settings.get<bool>("use direct solver",false);
    prec_type = canonicalPreconditionerType(settings.get<string>("preconditioner type","AMG"));
    use_preconditioner = settings.get<bool>("use preconditioner",true);
    reuse_preconditioner = settings.get<bool>("reuse preconditioner",true);
    preconditioner_reuse_type = canonicalReuseType(settings.get<string>("preconditioner reuse type","update"));
    const std::string blockPrecDefault = (prec_type == "block triangular") ? "teko_full" : "mrhyde";
    block_prec_backend = canonicalBlockPrecBackend(settings.get<string>("block prec backend", blockPrecDefault));
    right_preconditioner = settings.get<bool>("right preconditioner",false);
    reuse_matrix = settings.get<bool>("reuse Jacobian",false);
    schur.approximation_type = canonicalSchurApproximationType(settings.get<string>("Schur approximation type","base"));
    schur.variant = schur.approximation_type;
    schur.pivot_block = settings.get<int>("Schur pivot block",0);
    schur.damping = settings.get<ScalarT>("Schur damping",Teuchos::ScalarTraits<ScalarT>::one());
    schur.diag_use_lumped_pivot_diagonal =
      settings.get<bool>("Schur diag use lumped pivot diagonal", false);
    schur.pivot_block_preconditioner_type =
      canonicalBlockPrecType(settings.get<string>("Pivot block preconditioner type","AMG"));
    schur.pivot_block_diag_use_lumped_diagonal =
      settings.get<bool>("Pivot block diag use lumped diagonal", false);
    schur.schur_block_preconditioner_type = "AMG";
    refMaxwell.strict_refmaxwell = false;
  }

  bool parseStrictRefMaxwellFromRoot(Teuchos::ParameterList & settings) {
    if (!settings.isParameter("strict RefMaxwell")) return false;
    refMaxwell.strict_refmaxwell = settings.get<bool>("strict RefMaxwell");
    return true;
  }

  void parsePreconditionerSublist(bool & strictRefMaxwellSetExplicitly) {
    if (prec_sublist.name() == "empty") return;
    if (prec_sublist.isParameter("block prec backend")) {
      block_prec_backend = canonicalBlockPrecBackend(prec_sublist.get<string>("block prec backend"));
    }
    if (prec_sublist.isParameter("strict RefMaxwell")) {
      refMaxwell.strict_refmaxwell = prec_sublist.get<bool>("strict RefMaxwell");
      strictRefMaxwellSetExplicitly = true;
    }
    if (prec_sublist.isParameter("Schur pivot block")) {
      schur.pivot_block = prec_sublist.get<int>("Schur pivot block");
    }
    if (prec_sublist.isParameter("Schur diag use lumped pivot diagonal")) {
      schur.diag_use_lumped_pivot_diagonal =
        prec_sublist.get<bool>("Schur diag use lumped pivot diagonal");
    }
  }

  void parsePivotBlockSublist(bool & strictRefMaxwellSetExplicitly) {
    if (pivot_block_sublist.name() == "empty") return;
    if (pivot_block_sublist.isParameter("preconditioner type")) {
      schur.pivot_block_preconditioner_type =
        canonicalBlockPrecType(pivot_block_sublist.get<string>("preconditioner type"));
    }
    if (pivot_block_sublist.isParameter("diag use lumped diagonal")) {
      schur.pivot_block_diag_use_lumped_diagonal =
        pivot_block_sublist.get<bool>("diag use lumped diagonal");
    }
    if (pivot_block_sublist.isParameter("strict RefMaxwell")) {
      refMaxwell.strict_refmaxwell = pivot_block_sublist.get<bool>("strict RefMaxwell");
      strictRefMaxwellSetExplicitly = true;
    }
    // Check for XML parameter file in AMG Settings
    if (pivot_block_sublist.isSublist("AMG Settings")) {
      Teuchos::ParameterList & amgSettings = pivot_block_sublist.sublist("AMG Settings");
      if (amgSettings.isParameter("xml param file")) {
        amg.xml_param_file = amgSettings.get<string>("xml param file");
      }
    }
  }

  void parseSchurBlockSublist(bool & strictRefMaxwellSetExplicitly) {
    if (schur_block_sublist.name() == "empty") return;
    if (schur_block_sublist.isParameter("preconditioner type")) {
      schur.schur_block_preconditioner_type =
        canonicalBlockPrecType(schur_block_sublist.get<string>("preconditioner type"));
    }
    else if (schur_block_sublist.isSublist("RefMaxwell Settings") &&
             schur_block_sublist.sublist("RefMaxwell Settings").isParameter("preconditioner type")) {
      schur.schur_block_preconditioner_type =
        canonicalBlockPrecType(
          schur_block_sublist.sublist("RefMaxwell Settings").template get<string>("preconditioner type"));
    }
    else if (schur_block_sublist.isSublist("ADS Settings") &&
             schur_block_sublist.sublist("ADS Settings").isParameter("preconditioner type")) {
      schur.schur_block_preconditioner_type =
        schur_block_sublist.sublist("ADS Settings").template get<string>("preconditioner type");
    }
    if (schur_block_sublist.isParameter("approximation type")) {
      schur.approximation_type =
        canonicalSchurApproximationType(schur_block_sublist.get<string>("approximation type"));
      schur.variant = schur.approximation_type;
    }
    if (schur_block_sublist.isParameter("pivot block")) {
      schur.pivot_block = schur_block_sublist.get<int>("pivot block");
    }
    if (schur_block_sublist.isParameter("diag use lumped pivot diagonal")) {
      schur.diag_use_lumped_pivot_diagonal =
        schur_block_sublist.get<bool>("diag use lumped pivot diagonal");
    }
    if (schur_block_sublist.isParameter("strict RefMaxwell")) {
      refMaxwell.strict_refmaxwell = schur_block_sublist.get<bool>("strict RefMaxwell");
      strictRefMaxwellSetExplicitly = true;
    }
    // Check for XML parameter file in RefMaxwell Settings
    if (schur_block_sublist.isSublist("RefMaxwell Settings")) {
      Teuchos::ParameterList & refmaxwellSettings = schur_block_sublist.sublist("RefMaxwell Settings");
      if (refmaxwellSettings.isParameter("xml param file")) {
        refMaxwell.xml_param_file = refmaxwellSettings.get<string>("xml param file");
      }
    }
    // Check for XML parameter file in AMG Settings
    if (schur_block_sublist.isSublist("AMG Settings")) {
      Teuchos::ParameterList & amgSettings = schur_block_sublist.sublist("AMG Settings");
      if (amgSettings.isParameter("xml param file")) {
        amg.xml_param_file = amgSettings.get<string>("xml param file");
      }
    }
  }

  void inferStrictRefMaxwell(const bool strictRefMaxwellSetExplicitly) {
    if (strictRefMaxwellSetExplicitly) return;
    std::string pivotTypeUpper = schur.pivot_block_preconditioner_type;
    for (size_t i = 0; i < pivotTypeUpper.size(); ++i) {
      pivotTypeUpper[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(pivotTypeUpper[i])));
    }
    refMaxwell.strict_refmaxwell = (pivotTypeUpper == "REFMAXWELL");
  }

  void initializeRuntimeState() {
    have_preconditioner = false;
    have_symb_factor = false;
    have_matrix = false;
    jacobian_rebuilt_this_step = true;
    equation_set_index = 0;
  }
};

} // MrHyDE

#endif
