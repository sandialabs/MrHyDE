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
  bool merge_pivot_variables;       // group every non-target variable into the pivot role
  int target_block;                 // variable index taking the Schur role; -1 infers it
  std::string pivot_variable;       // names the pivot instead of indexing it
  std::string target_variable;      // names the Schur target instead of indexing it
  std::string variable_groups;      // 'ux,uy; pr' spells the partition; last role is target
  ScalarT mass_scale;               // multiplies M_p in the 'mass' Schur variant
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

  const Teuchos::ParameterList & pivotSettings() const {
    return role_sublists.empty() ? pivot_block_sublist : role_sublists.front();
  }

  const Teuchos::ParameterList & targetSettings() const {
    return role_sublists.empty() ? schur_block_sublist : role_sublists.back();
  }

  const Teuchos::ParameterList & roleSettings(const size_t role) const {
    if (!role_sublists.empty()) return role_sublists[role];
    return (role == 0) ? pivot_block_sublist : schur_block_sublist;
  }

  std::string rolePrecType(const size_t role) const {
    if (!role_sublists.empty()) {
      const Teuchos::ParameterList & list = role_sublists[role];
      if (list.isParameter("preconditioner")) {
        return list.template get<std::string>("preconditioner");
      }
      return "AMG";
    }
    return (role == 0) ? schur.pivot_block_preconditioner_type
                       : schur.schur_block_preconditioner_type;
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
  // Named-role layout: one container, one sublist per group, target role last.
  Teuchos::ParameterList scheme_sublist;
  std::string scheme_sublist_name;
  std::string monolithic_sublist_name;
  bool flat_prec_sublist_used = false;
  std::string deprecated_prec_sublist_message;
  std::string schur_target_role;
  std::vector<std::string> role_names;      // role order, target last
  std::vector<std::string> role_variables;  // parallel: 'ux, uy' per role
  std::vector<Teuchos::ParameterList> role_sublists;

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
    // Monolithic settings: a per-scheme container is preferred, and the flat
    // 'Preconditioner Settings' is the deprecated flat form.
    prec_sublist = Teuchos::ParameterList("empty");
    static const char * monolithic[] = {"Ifpack2 Settings", "MueLu Settings",
                                        "Domain Decomposition Settings"};
    for (size_t k = 0; k < 3; ++k) {
      if (!settings.isSublist(monolithic[k])) continue;
      TEUCHOS_TEST_FOR_EXCEPTION(prec_sublist.name() != "empty", std::runtime_error,
        "More than one monolithic settings container is present; use one.");
      prec_sublist = settings.sublist(monolithic[k]);
      monolithic_sublist_name = monolithic[k];
    }
    if (monolithic_sublist_name.empty() && settings.isSublist("Preconditioner Settings")) {
      prec_sublist = settings.sublist("Preconditioner Settings");
      flat_prec_sublist_used = true;
    }
    pivot_block_sublist = settings.isSublist("Pivot Block Settings")
      ? settings.sublist("Pivot Block Settings")
      : Teuchos::ParameterList("empty");
    schur_block_sublist = settings.isSublist("Schur Block Settings")
      ? settings.sublist("Schur Block Settings")
      : Teuchos::ParameterList("empty");
    parseSchemeContainer(settings);
  }

  // One container per scheme, holding the grouping and one sublist per role. Named roles
  void parseSchemeContainer(Teuchos::ParameterList & settings) {
    static const char * names[] = {"Block Triangular Settings", "Block Diagonal Settings"};
    scheme_sublist = Teuchos::ParameterList("empty");
    scheme_sublist_name = "";
    for (size_t k = 0; k < 2; ++k) {
      if (!settings.isSublist(names[k])) continue;
      TEUCHOS_TEST_FOR_EXCEPTION(!scheme_sublist_name.empty(), std::runtime_error,
        "Both 'Block Triangular Settings' and 'Block Diagonal Settings' are present; use one.");
      scheme_sublist = settings.sublist(names[k]);
      scheme_sublist_name = names[k];
    }
    if (scheme_sublist_name.empty()) return;

    // 'variable groups' is an ordered name -> variable-list map; input order is role order.
    TEUCHOS_TEST_FOR_EXCEPTION(!scheme_sublist.isSublist("variable groups"), std::runtime_error,
      scheme_sublist_name << " requires a 'variable groups' sublist naming each role.");
    const Teuchos::ParameterList & groups = scheme_sublist.sublist("variable groups");
    for (Teuchos::ParameterList::ConstIterator it = groups.begin(); it != groups.end(); ++it) {
      role_names.push_back(groups.name(it));
      role_variables.push_back(groups.get<std::string>(groups.name(it)));
    }
    TEUCHOS_TEST_FOR_EXCEPTION(role_names.size() < 2, std::runtime_error,
      scheme_sublist_name << " 'variable groups' needs at least two roles, got "
      << role_names.size() << ".");

    if (scheme_sublist.isParameter("schur target")) {
      schur_target_role = scheme_sublist.get<std::string>("schur target");
    }
    orderRolesTargetLast();
    collectRoleSublists();
  }

  // Only the last role escapes being used as an elimination weight, so the target goes
  // there regardless of the order the groups were listed in.
  void orderRolesTargetLast() {
    if (schur_target_role.empty()) return;
    size_t at = role_names.size();
    for (size_t r = 0; r < role_names.size(); ++r) {
      if (role_names[r] == schur_target_role) at = r;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(at == role_names.size(), std::runtime_error,
      "'schur target: " << schur_target_role << "' names no group in 'variable groups'.");
    role_names.push_back(role_names[at]);
    role_variables.push_back(role_variables[at]);
    role_names.erase(role_names.begin() + at);
    role_variables.erase(role_variables.begin() + at);
  }

  void collectRoleSublists() {
    role_sublists.clear();
    for (size_t r = 0; r < role_names.size(); ++r) {
      TEUCHOS_TEST_FOR_EXCEPTION(!scheme_sublist.isSublist(role_names[r]), std::runtime_error,
        scheme_sublist_name << " has no '" << role_names[r]
        << "' sublist for the role of that name.");
      role_sublists.push_back(scheme_sublist.sublist(role_names[r]));
    }
    // Anything else that is a sublist is a typo, not an ignorable extra.
    for (Teuchos::ParameterList::ConstIterator it = scheme_sublist.begin();
         it != scheme_sublist.end(); ++it) {
      const std::string key = scheme_sublist.name(it);
      if (!scheme_sublist.isSublist(key) || key == "variable groups") continue;
      bool known = false;
      for (size_t r = 0; r < role_names.size(); ++r) known = known || role_names[r] == key;
      TEUCHOS_TEST_FOR_EXCEPTION(!known, std::runtime_error,
        scheme_sublist_name << " sublist '" << key
        << "' names no group in 'variable groups'.");
    }
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
    if (flat_prec_sublist_used) {
      const std::string want = (prec_type == "Ifpack2") ? "Ifpack2 Settings"
                             : (prec_type == "AMG") ? "MueLu Settings"
                             : "Domain Decomposition Settings";
      // 'AMG' maps to 'MueLu Settings' rather than 'AMG Settings', because the latter
      // already names the nested MueLu list inside a role.
      deprecated_prec_sublist_message =
        "WARNING: 'Preconditioner Settings' is deprecated and will be removed.\n"
        "         This deck has 'preconditioner type: " + prec_type + "', so rename it to '"
        + want + "'.\n"
        "         The full mapping:\n"
        "           preconditioner type: Ifpack2               ->  Ifpack2 Settings\n"
        "           preconditioner type: AMG                   ->  MueLu Settings\n"
        "           preconditioner type: domain decomposition  ->  Domain Decomposition Settings\n"
        "         Keys inside the sublist do not change.";
    }
    // A scheme container that does not match the selected scheme is a silent no-op
    if (!scheme_sublist_name.empty()) {
      const std::string want = (scheme_sublist_name == "Block Triangular Settings")
        ? "block triangular" : "block diagonal";
      TEUCHOS_TEST_FOR_EXCEPTION(prec_type != want, std::runtime_error,
        "'" << scheme_sublist_name << "' is present but 'preconditioner type' is '"
        << prec_type << "'; it must be '" << want << "'.");
    }
    use_preconditioner = settings.get<bool>("use preconditioner",true);
    preconditioner_reuse_type = block_prec::canonicalReuseType(settings.get<string>("preconditioner reuse type","update"));
    if (settings.isType<bool>("reuse preconditioner") &&
        !settings.get<bool>("reuse preconditioner")) {
      preconditioner_reuse_type = "none";
    }
    right_preconditioner = settings.get<bool>("right preconditioner",false);
    reuse_matrix = settings.get<bool>("reuse Jacobian",false);
    schur.approximation_type = "base";
    schur.merge_pivot_variables = true;
    schur.target_block = -1;
    schur.pivot_variable = "";
    schur.target_variable = "";
    schur.variable_groups = "";
    schur.mass_scale = 1.0;
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
    if (role_sublists.empty() && pivot_block_sublist.name() == "empty") return;
    if (pivotSettings().isParameter("preconditioner type")) {
      schur.pivot_block_preconditioner_type =
        block_prec::canonicalBlockPrecType(pivotSettings().template get<string>("preconditioner type"));
    }
    if (pivotSettings().isParameter("diag use lumped diagonal")) {
      schur.pivot_block_diag_use_lumped_diagonal =
        pivotSettings().template get<bool>("diag use lumped diagonal");
    }
    if (pivotSettings().isSublist("RefMaxwell Settings")) {
      Teuchos::ParameterList & refmaxwellSettings = pivotSettingsMutable().sublist("RefMaxwell Settings");
      if (refmaxwellSettings.isParameter("xml param file")) {
        refMaxwell.xml_param_file_pivot = refmaxwellSettings.get<string>("xml param file");
      }
    }
    if (pivotSettings().isSublist("Maxwell1 Settings")) {
      Teuchos::ParameterList & maxwell1Settings = pivotSettingsMutable().sublist("Maxwell1 Settings");
      if (maxwell1Settings.isParameter("xml param file")) {
        maxwell1.xml_param_file_pivot = maxwell1Settings.get<string>("xml param file");
      }
    }
  }

  const Teuchos::ParameterList & schemeSettings() const {
    return scheme_sublist_name.empty() ? schur_block_sublist : scheme_sublist;
  }

  // Nested RefMaxwell/Maxwell1 Settings live in a role sublist, not the container.
  Teuchos::ParameterList & pivotSettingsMutable() {
    return role_sublists.empty() ? pivot_block_sublist : role_sublists.front();
  }

  Teuchos::ParameterList & targetSettingsMutable() {
    return role_sublists.empty() ? schur_block_sublist : role_sublists.back();
  }

  void parseSchurBlockSublist() {
    if (scheme_sublist_name.empty() && schur_block_sublist.name() == "empty") return;
    if (schemeSettings().isParameter("preconditioner type")) {
      schur.schur_block_preconditioner_type =
        block_prec::canonicalBlockPrecType(schemeSettings().template get<string>("preconditioner type"));
    }
    if (schemeSettings().isParameter("approximation type")) {
      schur.approximation_type =
        block_prec::canonicalSchurApproximationType(schemeSettings().template get<string>("approximation type"));
    }
    if (schemeSettings().isParameter("pivot block")) {
      schur.pivot_block = schemeSettings().template get<int>("pivot block");
    }
    if (schemeSettings().isParameter("mass scale")) {
      schur.mass_scale = schemeSettings().template get<ScalarT>("mass scale");
    }
    if (schemeSettings().isParameter("target block")) {
      schur.target_block = schemeSettings().template get<int>("target block");
    }
    if (schemeSettings().isParameter("pivot variable")) {
      schur.pivot_variable = schemeSettings().template get<std::string>("pivot variable");
    }
    if (schemeSettings().isParameter("target variable")) {
      schur.target_variable = schemeSettings().template get<std::string>("target variable");
    }
    // isParameter is true for sublists too, and the named layout spells this as one.
    if (schemeSettings().isParameter("variable groups") &&
        !schemeSettings().isSublist("variable groups")) {
      schur.variable_groups = schemeSettings().template get<std::string>("variable groups");
    }
    if (schemeSettings().isParameter("merge pivot variables")) {
      schur.merge_pivot_variables = schemeSettings().template get<bool>("merge pivot variables");
    }
    if (schemeSettings().isParameter("diag use lumped pivot diagonal")) {
      schur.diag_use_lumped_pivot_diagonal =
        schemeSettings().template get<bool>("diag use lumped pivot diagonal");
    }
    if (schemeSettings().isParameter("triangle")) {
      schur.triangle = block_prec::canonicalSchurTriangle(schemeSettings().template get<string>("triangle"));
    }
    if (schemeSettings().isParameter("damping")) {
      schur.damping = schemeSettings().template get<ScalarT>("damping");
    }
    if (targetSettings().isSublist("RefMaxwell Settings")) {
      Teuchos::ParameterList & refmaxwellSettings = targetSettingsMutable().sublist("RefMaxwell Settings");
      if (refmaxwellSettings.isParameter("xml param file")) {
        refMaxwell.xml_param_file_schur = refmaxwellSettings.get<string>("xml param file");
      }
    }
    if (targetSettings().isSublist("Maxwell1 Settings")) {
      Teuchos::ParameterList & maxwell1Settings = targetSettingsMutable().sublist("Maxwell1 Settings");
      if (maxwell1Settings.isParameter("xml param file")) {
        maxwell1.xml_param_file_schur = maxwell1Settings.get<string>("xml param file");
      }
    }
    // Mirror role 0 and the target into the flat fields the two-role path reads.
    // Only the triangular container: block-diagonal roles name Ifpack2 smoothers such as
    // RELAXATION, which are not block-preconditioner types.
    if (!role_sublists.empty() && scheme_sublist_name == "Block Triangular Settings") {
      schur.pivot_block_preconditioner_type =
        block_prec::canonicalBlockPrecType(rolePrecType(0));
      schur.schur_block_preconditioner_type =
        block_prec::canonicalBlockPrecType(rolePrecType(role_sublists.size() - 1));
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
