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

// Schur and block-triangular options.
// Ooverview at the top of linearAlgebraInterface_blockprec.hpp.
struct SchurConfig {
  std::string approximation_type;   // base or diag
  int pivot_block;                  // variable index taking the pivot role
  ScalarT damping;                  // gamma in the diag Schur correction
  bool diag_use_lumped_pivot_diagonal;
  std::string triangle;             // auto, upper, lower
  std::string pivot_block_preconditioner_type;
  bool pivot_block_diag_use_lumped_diagonal;
  std::string schur_block_preconditioner_type;
};

// Auxiliary operators for the H(curl) preconditioners, built once per set by
// SolverManager::setupBlockTriangularAuxiliary and shared by RefMaxwell, Maxwell1 and
// the Hiptmair smoother. D0/M1/Kn are defined above the
// buildRefMaxwellPreconditioner (linearAlgebraInterface_solvers.hpp).
template<class Node>
struct RefMaxwellData {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
  typedef typename Teuchos::ScalarTraits<ScalarT>::coordinateType CoordScalar;
  typedef Tpetra::MultiVector<CoordScalar,LO,GO,Node> LA_CoordMultiVector;
  Teuchos::RCP<LA_CrsMatrix> D0_matrix;
  Teuchos::RCP<LA_CrsMatrix> M1_matrix;
  Teuchos::RCP<LA_CoordMultiVector> nodal_coords;
  Teuchos::RCP<LA_MultiVector> nodal_lumped_mass;

  ScalarT schur_addon_beta = 0.0;
  bool schur_addon_beta_valid = false;
  ScalarT schur_addon_beta_alpha_u = 0.0;
  ScalarT schur_addon_beta_built = 0.0;

  std::string xml_param_file_pivot = "";
  std::string xml_param_file_schur = "";
};

// Per-variable-block data, indexed the way block_prec::buildBlockMaps orders variables.
template<class Node>
struct BlockData {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node> LA_CrsMatrix;
  typedef typename Teuchos::ScalarTraits<ScalarT>::coordinateType CoordScalar;
  typedef Tpetra::MultiVector<CoordScalar,LO,GO,Node> LA_CoordMultiVector;
  std::vector<Teuchos::RCP<LA_CrsMatrix> > mass_matrices;
  std::vector<Teuchos::RCP<LA_CoordMultiVector> > dof_coords;   // per-DOF, for MueLu
};

// Maxwell1 (Reitzinger-Schoberl) reuses D0, M1 and the coords from RefMaxwellData.
template<class Node>
struct Maxwell1Data {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node> LA_CrsMatrix;
  std::string xml_param_file_pivot = "";
  std::string xml_param_file_schur = "";
  Teuchos::RCP<LA_CrsMatrix> D0_normalized;
};

// Monolithic AMG only.
struct AMGData {
  std::string xml_param_file = "";
};

// Settings and reusable state for one linear solver: the parsed deck, plus the
// matrices, factorizations and preconditioners that survive across solves.
template<class Node>
class LinearSolverContext {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
  typedef typename Teuchos::ScalarTraits<ScalarT>::coordinateType CoordScalar;
  typedef Tpetra::MultiVector<CoordScalar,LO,GO,Node> LA_CoordMultiVector;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
  
public:
  LinearSolverContext() {};
  ~LinearSolverContext() {};

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
  
  // Parsed deck
  string amesos_type;                 // KLU2, SuperLU, ...
  string belos_type;                  // Block GMRES, BiCGStab, GCRODR, ...
  bool flexible_gmres = false;        // gates the inner-Krylov wraps
  string prec_type;                   // AMG, Ifpack2, domain decomposition, block diagonal/triangular
  bool use_direct;
  bool use_preconditioner;
  bool right_preconditioner;
  string preconditioner_reuse_type;   // none, update, full
  bool reuse_matrix;
  ScalarT stage_alpha_u = 1.0;        // DIRK spatial-term scaling a_ss/b_s

  // Runtime state
  bool jacobian_rebuilt_this_step;
  bool have_matrix;
  bool have_preconditioner;
  bool have_symb_factor;

  SchurConfig schur;
  AMGData amg;
  RefMaxwellData<Node> refMaxwell;
  BlockData<Node> block;
  Maxwell1Data<Node> maxwell1;

  Teuchos::ParameterList prec_sublist, belos_sublist;
  Teuchos::ParameterList pivot_block_sublist, schur_block_sublist;

  // Cached across solves so GCRODR's recycled subspace and RCG's conjugate
  // vectors survive.
  Teuchos::RCP<Belos::SolverManager<ScalarT,
                                    Tpetra::MultiVector<ScalarT,LO,GO,Node>,
                                    Tpetra::Operator<ScalarT,LO,GO,Node> > > belos_solver_mgr;
  Teuchos::RCP<Belos::LinearProblem<ScalarT,
                                    Tpetra::MultiVector<ScalarT,LO,GO,Node>,
                                    Tpetra::Operator<ScalarT,LO,GO,Node> > > belos_problem;
  bool reuse_belos_solver_mgr;

  Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > amesos_solver;
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > prec;      // monolithic AMG
  Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > prec_dd; // domain decomposition
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > prec_block;        // block diagonal/triangular

  matrix_RCP matrix;

  // RefMaxwell and Maxwell1 are cached per block role, one hierarchy each, and reused
  // through resetMatrix.
  Teuchos::RCP<MueLu::RefMaxwell<ScalarT, LO, GO, Node> > refmaxwell_prec;
  Teuchos::RCP<MueLu::RefMaxwell<ScalarT, LO, GO, Node> > schur_refmaxwell_prec;
  Teuchos::RCP<MueLu::Maxwell1<ScalarT, LO, GO, Node> > maxwell1_prec;
  Teuchos::RCP<MueLu::Maxwell1<ScalarT, LO, GO, Node> > schur_maxwell1_prec;

  size_t equation_set_index;   // set index when linearSolver(set,...) is used

  // RefMaxwell XML for one block role, parsed once. Callers that modify it must copy!
  const Teuchos::ParameterList & refMaxwellParams(const bool forSchur,
                                                  const Teuchos::Comm<int> & comm) {
    Teuchos::RCP<Teuchos::ParameterList> & cached = forSchur ? refmaxwell_xml_schur
                                                             : refmaxwell_xml_pivot;
    if (cached.is_null()) {
      cached = Teuchos::rcp(new Teuchos::ParameterList());
      const string & file = forSchur ? refMaxwell.xml_param_file_schur
                                     : refMaxwell.xml_param_file_pivot;
      if (!file.empty()) block_prec::loadXmlBroadcast(file, *cached, comm, "RefMaxwell");
    }
    return *cached;
  }

  bool schurAddonWanted(const Teuchos::Comm<int> & comm) {
    if (block_prec::parseBlockPrecType(schur.schur_block_preconditioner_type) != block_prec::BlockPrecType::RefMaxwell) {
      return false;
    }
    return block_prec::refMaxwellAddonEnabled(refMaxwellParams(true, comm));
  }

  block_prec::InverseLibraryCache<Node> & inverseLibrary(const int verbosity, const int rank) {
    if (inverse_library.is_null()) {
      inverse_library = Teuchos::rcp(new block_prec::InverseLibraryCache<Node>(
        block_prec::reuseKeepsHierarchy(preconditioner_reuse_type), verbosity, rank));
    }
    return *inverse_library;
  }

private:
  Teuchos::RCP<block_prec::InverseLibraryCache<Node> > inverse_library;
  Teuchos::RCP<Teuchos::ParameterList> refmaxwell_xml_pivot, refmaxwell_xml_schur;

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
      block_prec::validatePivotBlockSettingsSection(pivot_block_sublist, "Pivot Block Settings");
    }
    if (schur_block_sublist.name() != "empty") {
      block_prec::validateSchurBlockSettingsSection(schur_block_sublist, "Schur Block Settings");
    }
  }

  void parseGeneralSettings(Teuchos::ParameterList & settings) {
    use_direct = settings.get<bool>("use direct solver",false);
    prec_type = block_prec::canonicalPreconditionerType(settings.get<string>("preconditioner type","AMG"));
    use_preconditioner = settings.get<bool>("use preconditioner",true);
    preconditioner_reuse_type = block_prec::canonicalReuseType(settings.get<string>("preconditioner reuse type","update"));
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
        block_prec::canonicalBlockPrecType(pivot_block_sublist.get<string>("preconditioner type"));
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
        block_prec::canonicalBlockPrecType(schur_block_sublist.get<string>("preconditioner type"));
    }
    if (schur_block_sublist.isParameter("approximation type")) {
      schur.approximation_type =
        block_prec::canonicalSchurApproximationType(schur_block_sublist.get<string>("approximation type"));
    }
    if (schur_block_sublist.isParameter("pivot block")) {
      schur.pivot_block = schur_block_sublist.get<int>("pivot block");
    }
    if (schur_block_sublist.isParameter("diag use lumped pivot diagonal")) {
      schur.diag_use_lumped_pivot_diagonal =
        schur_block_sublist.get<bool>("diag use lumped pivot diagonal");
    }
    if (schur_block_sublist.isParameter("triangle")) {
      schur.triangle = block_prec::canonicalSchurTriangle(schur_block_sublist.get<string>("triangle"));
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
    const std::string bu = block_prec::toUpperAsciiCopy(belos_type);
    reuse_belos_solver_mgr = (bu == "GCRODR" || bu == "RCG");
    belos_solver_mgr = Teuchos::null;
    belos_problem = Teuchos::null;
  }
};

} // MrHyDE

#endif
