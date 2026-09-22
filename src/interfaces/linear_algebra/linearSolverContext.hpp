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
#include "block_prec/InverseLibraryOps.hpp"
#include <cctype>

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosSolverManager.hpp>
#include <BelosTpetraAdapter.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>
#include <MueLu_RefMaxwell.hpp>
#include <MueLu_Maxwell1.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>

// Amesos includes
#include "Amesos2.hpp"

namespace MrHyDE {

/** \brief Schur approximation and block-triangular options (type, damping, pivot block, strictness). */
struct SchurConfig {
  /** Canonical Schur approximation family (currently base or diag). */
  std::string approximation_type;
  /**< Pivot block index p. pivot_block=0 -> Schur on block 1: S = J11 - J10*inv(J00)*J01.
   *   pivot_block=1 -> Schur on block 0: S = J00 - J01*inv(J11)*J10. */
  int pivot_block;
  /**< Gamma in diag Schur. E.g. pivot_block=0: S = J11 - gamma*J10*diag(J00)^{-1}*J01. */
  ScalarT damping;
  bool diag_use_lumped_pivot_diagonal;
  /**< Canonical triangle for the block Gauss-Seidel sweep (auto, upper, lower). */
  std::string triangle;
  std::string pivot_block_preconditioner_type;
  bool pivot_block_diag_use_lumped_diagonal;
  std::string schur_block_preconditioner_type;
};

/** \brief RefMaxwell auxiliary matrices and vectors (D0, M1, coords) and debug/strict flags. */
template<class Node>
struct RefMaxwellData {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
  typedef typename Teuchos::ScalarTraits<ScalarT>::coordinateType CoordScalar;
  typedef Tpetra::MultiVector<CoordScalar,LO,GO,Node> LA_CoordMultiVector;
  Teuchos::RCP<LA_CrsMatrix> D0_matrix;   /**< Discrete gradient (HGRAD -> HCURL). */
  Teuchos::RCP<LA_CrsMatrix> M1_matrix;   /**< Edge mass matrix for HCURL block. */
  Teuchos::RCP<LA_CoordMultiVector> nodal_coords;
  /** Lumped nodal mass, integral(N_n), for the RefMaxwell addon. */
  Teuchos::RCP<LA_MultiVector> nodal_lumped_mass;
  /** Curl-curl coefficient of S, alpha_u^2 * gamma / (alpha_t * mu), read off
   *  the Schur correction. The pivot hierarchy has no beta of its own. */
  ScalarT schur_addon_beta = 0.0;
  /** Memo for schur_addon_beta: the probe was taken at this stage_alpha_u. */
  bool schur_addon_beta_valid = false;
  ScalarT schur_addon_beta_alpha_u = 0.0;
  ScalarT schur_addon_beta_built = 0.0; /**< beta baked into the cached Schur hierarchy. */
  bool schur_addon_wanted = false;      /**< The Schur XML asked for the addon. */
  std::string xml_param_file_pivot = "";
  std::string xml_param_file_schur = "";
};

/** \brief Per-variable-block auxiliary data. Read by block-diagonal
 *  preconditioning and by the block-triangular Schur mass correction. */
template<class Node>
struct BlockData {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node> LA_CrsMatrix;
  typedef typename Teuchos::ScalarTraits<ScalarT>::coordinateType CoordScalar;
  typedef Tpetra::MultiVector<CoordScalar,LO,GO,Node> LA_CoordMultiVector;
  std::vector<Teuchos::RCP<LA_CrsMatrix> > mass_matrices;      /**< Mass matrix per variable block. */
  std::vector<Teuchos::RCP<LA_CoordMultiVector> > dof_coords;  /**< Coordinates per variable block for MueLu. */
};

/** \brief Maxwell1 (Reitzinger-Schoberl / energy-min) configuration. Reuses
 *  D0 and nodal_coords from RefMaxwellData; only needs its own XML.
 *  D0_normalized is a sign-preserving normalization of D0 (each nonzero
 *  becomes +-1). Built lazily on first Maxwell1 build. */
template<class Node>
struct Maxwell1Data {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node> LA_CrsMatrix;
  std::string xml_param_file_pivot = "";
  std::string xml_param_file_schur = "";
  Teuchos::RCP<LA_CrsMatrix> D0_normalized;
};

/**
 * \struct AMGData
 * \brief Configuration data for MueLu AMG (Algebraic MultiGrid) preconditioner.
 *
 * Stores the monolithic AMG path configuration. Block AMG reads its XML from the
 * pivot/Schur AMG Settings sublists instead.
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
    // Parse order is intentional: discover/validate sublists first, then root
    // defaults, then block-specific overrides.
    parseBelosAndAmesosSettings(settings);
    parseSublists(settings);
    validateSublists();
    parseGeneralSettings(settings);
    parsePreconditionerSublist();
    parsePivotBlockSublist();
    parseSchurBlockSublist();
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
    maxwell1_prec = Teuchos::null;
    schur_maxwell1_prec = Teuchos::null;
    belos_solver_mgr = Teuchos::null;
    belos_problem = Teuchos::null;
    inverse_library = Teuchos::null;
    jacobian_rebuilt_this_step = true;
  }
  
  // Public data members
  string amesos_type;   /**< Amesos2 solver type (e.g., KLU2). */
  string belos_type;    /**< Belos solver type (e.g., GMRES). */
  bool flexible_gmres = false;  /**< Outer Belos uses Flexible GMRES; gates inner-Krylov wraps. */
  string prec_type;     /**< Preconditioner type (e.g., AMG). */
  bool use_direct;            /**< Use direct Amesos2 solver. */
  bool use_preconditioner;      /**< Whether to apply a preconditioner. */
  bool right_preconditioner;    /**< Whether to apply right preconditioning. */
  string preconditioner_reuse_type; /**< Reuse mode (none, update, or full). */
  bool reuse_matrix;          /**< Whether to reuse an existing Jacobian. */
  ScalarT stage_alpha_u = 1.0; /**< DIRK spatial-term scaling a_ss/b_s. */
  bool jacobian_rebuilt_this_step; /**< True when Jacobian values were rebuilt before this linear solve. */
  bool have_matrix;           /**< Indicates whether a Jacobian has been constructed. */
  bool have_preconditioner;     /**< Indicates whether a preconditioner exists. */
  bool have_symb_factor;        /**< Indicates whether symbolic factorization exists. */
  /**< Grouped Schur and block-tri options (type, damping, pivot block, strictness). */
  SchurConfig schur;
  /**< AMG preconditioner configuration data including XML parameter support. */
  AMGData amg;
  /**< RefMaxwell matrices/vectors (D0, M1, coords) and debug/strict flags. */
  RefMaxwellData<Node> refMaxwell;

  BlockData<Node> block;
  /**< Maxwell1 (Reitzinger-Schoberl / energy-min) configuration; reuses D0 + coords from refMaxwell. */
  Maxwell1Data<Node> maxwell1;

  Teuchos::ParameterList prec_sublist, belos_sublist;
  Teuchos::ParameterList pivot_block_sublist, schur_block_sublist;

  // Cached across solves so GCRODR's recycled subspace and RCG's conjugate
  // vectors survive; the LinearProblem is cached so the SolverManager's
  // back-pointer stays valid when LHS/RHS/operator are swapped in place.
  Teuchos::RCP<Belos::SolverManager<ScalarT,
                                    Tpetra::MultiVector<ScalarT,LO,GO,Node>,
                                    Tpetra::Operator<ScalarT,LO,GO,Node> > > belos_solver_mgr;
  Teuchos::RCP<Belos::LinearProblem<ScalarT,
                                    Tpetra::MultiVector<ScalarT,LO,GO,Node>,
                                    Tpetra::Operator<ScalarT,LO,GO,Node> > > belos_problem;
  bool reuse_belos_solver_mgr;

  Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > amesos_solver; /**< Reusable Amesos2 direct solver. */
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > prec; /**< MueLu AMG preconditioner operator. */
  Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > prec_dd; /**< Ifpack2 domain decomposition preconditioner. */
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > prec_block; /**< Block-diagonal AMG preconditioner operator. */

  matrix_RCP matrix; /**< Current Jacobian matrix. */

  // Cached RefMaxwell preconditioner for reuse.
  Teuchos::RCP<MueLu::RefMaxwell<ScalarT, LO, GO, Node> > refmaxwell_prec;
  Teuchos::RCP<MueLu::RefMaxwell<ScalarT, LO, GO, Node> > schur_refmaxwell_prec; /**< Cached RefMaxwell for the Schur block. */
  // Cached Maxwell1 (Reitzinger-Schoberl / energy-min) preconditioner for reuse.
  Teuchos::RCP<MueLu::Maxwell1<ScalarT, LO, GO, Node> > maxwell1_prec;
  Teuchos::RCP<MueLu::Maxwell1<ScalarT, LO, GO, Node> > schur_maxwell1_prec;

  size_t equation_set_index; /**< Set index when linearSolver(set,...) is used; for block prec. */

  /** Built on first use; block preconditioner paths only. */
  block_prec::InverseLibraryCache<Node> & inverseLibrary(const int verbosity, const int rank) {
    if (inverse_library.is_null()) {
      inverse_library = Teuchos::rcp(new block_prec::InverseLibraryCache<Node>(
        reuseKeepsHierarchy(preconditioner_reuse_type), verbosity, rank));
    }
    return *inverse_library;
  }

private:
  Teuchos::RCP<block_prec::InverseLibraryCache<Node> > inverse_library;

  void parseBelosAndAmesosSettings(Teuchos::ParameterList & settings) {
    amesos_type = settings.get<string>("Amesos solver","KLU2");
    belos_type = settings.get<string>("Belos solver","Block GMRES");
    // Accept "Flexible Gmres" at top level or inside "Belos Settings".
    bool topFlex = settings.isType<bool>("Flexible Gmres") && settings.get<bool>("Flexible Gmres");
    bool subFlex = settings.isSublist("Belos Settings")
                 && settings.sublist("Belos Settings").isType<bool>("Flexible Gmres")
                 && settings.sublist("Belos Settings").get<bool>("Flexible Gmres");
    flexible_gmres = topFlex || subFlex;
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
    preconditioner_reuse_type = canonicalReuseType(settings.get<string>("preconditioner reuse type","update"));
    // 'reuse preconditioner' predates the string and only ever meant none-or-not.
    if (settings.isType<bool>("reuse preconditioner") &&
        !settings.get<bool>("reuse preconditioner")) {
      preconditioner_reuse_type = "none";
    }
    right_preconditioner = settings.get<bool>("right preconditioner",false);
    reuse_matrix = settings.get<bool>("reuse Jacobian",false);
    schur.approximation_type = "base";
    schur.pivot_block = 0;
    schur.damping = Teuchos::ScalarTraits<ScalarT>::one();
    schur.diag_use_lumped_pivot_diagonal = false;
    schur.triangle = "auto";
    schur.pivot_block_preconditioner_type = "AMG";
    schur.pivot_block_diag_use_lumped_diagonal = false;
    schur.schur_block_preconditioner_type = "AMG";
  }

  void parsePreconditionerSublist() {
    if (prec_sublist.name() == "empty") return;
    if (prec_sublist.isParameter("xml param file")) {
      amg.xml_param_file = prec_sublist.get<string>("xml param file");
    }
    else if (prec_sublist.isSublist("AMG Settings") &&
             prec_sublist.sublist("AMG Settings").isParameter("xml param file")) {
      amg.xml_param_file = prec_sublist.sublist("AMG Settings").template get<string>("xml param file");
    }
  }

  void parsePivotBlockSublist() {
    if (pivot_block_sublist.name() == "empty") return;
    if (pivot_block_sublist.isParameter("preconditioner type")) {
      schur.pivot_block_preconditioner_type =
        canonicalBlockPrecType(pivot_block_sublist.get<string>("preconditioner type"));
    }
    if (pivot_block_sublist.isParameter("diag use lumped diagonal")) {
      schur.pivot_block_diag_use_lumped_diagonal =
        pivot_block_sublist.get<bool>("diag use lumped diagonal");
    }
    if (pivot_block_sublist.isSublist("RefMaxwell Settings")) {
      Teuchos::ParameterList & refmaxwellSettings = pivot_block_sublist.sublist("RefMaxwell Settings");
      if (refmaxwellSettings.isParameter("xml param file")) {
        refMaxwell.xml_param_file_pivot = refmaxwellSettings.get<string>("xml param file");
      }
    }
    if (pivot_block_sublist.isSublist("Maxwell1 Settings")) {
      Teuchos::ParameterList & maxwell1Settings = pivot_block_sublist.sublist("Maxwell1 Settings");
      if (maxwell1Settings.isParameter("xml param file")) {
        maxwell1.xml_param_file_pivot = maxwell1Settings.get<string>("xml param file");
      }
    }
  }

  void parseSchurBlockSublist() {
    if (schur_block_sublist.name() == "empty") return;
    if (schur_block_sublist.isParameter("preconditioner type")) {
      schur.schur_block_preconditioner_type =
        canonicalBlockPrecType(schur_block_sublist.get<string>("preconditioner type"));
    }
    if (schur_block_sublist.isParameter("approximation type")) {
      schur.approximation_type =
        canonicalSchurApproximationType(schur_block_sublist.get<string>("approximation type"));
    }
    if (schur_block_sublist.isParameter("pivot block")) {
      schur.pivot_block = schur_block_sublist.get<int>("pivot block");
    }
    if (schur_block_sublist.isParameter("diag use lumped pivot diagonal")) {
      schur.diag_use_lumped_pivot_diagonal =
        schur_block_sublist.get<bool>("diag use lumped pivot diagonal");
    }
    if (schur_block_sublist.isParameter("triangle")) {
      schur.triangle = canonicalSchurTriangle(schur_block_sublist.get<string>("triangle"));
    }
    if (schur_block_sublist.isParameter("damping")) {
      schur.damping = schur_block_sublist.get<ScalarT>("damping");
    }
    if (schur_block_sublist.isSublist("RefMaxwell Settings")) {
      Teuchos::ParameterList & refmaxwellSettings = schur_block_sublist.sublist("RefMaxwell Settings");
      if (refmaxwellSettings.isParameter("xml param file")) {
        refMaxwell.xml_param_file_schur = refmaxwellSettings.get<string>("xml param file");
      }
    }
    if (schur_block_sublist.isSublist("Maxwell1 Settings")) {
      Teuchos::ParameterList & maxwell1Settings = schur_block_sublist.sublist("Maxwell1 Settings");
      if (maxwell1Settings.isParameter("xml param file")) {
        maxwell1.xml_param_file_schur = maxwell1Settings.get<string>("xml param file");
      }
    }
  }

  void initializeRuntimeState() {
    have_preconditioner = false;
    have_symb_factor = false;
    have_matrix = false;
    jacobian_rebuilt_this_step = true;
    equation_set_index = 0;
    const std::string bu = toUpperAsciiCopy(belos_type);
    reuse_belos_solver_mgr = (bu == "GCRODR" || bu == "RCG");
    belos_solver_mgr = Teuchos::null;
    belos_problem = Teuchos::null;
  }
};

} // MrHyDE

#endif
