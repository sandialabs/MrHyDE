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
// Overview at the top of linearAlgebraInterface_blockprec.hpp.
struct SchurConfig {
  std::string approximation_type;   // base, diag or mass
  ScalarT damping;                  // gamma in the diag Schur correction
  bool correction_use_lumped_weight;   // 'diag use lumped pivot diagonal'
  std::string triangle;             // auto, upper, lower
  bool diagonal_prec_use_lumped;    // 'diag use lumped diagonal', the Diagonal split inverse
  ScalarT mass_scale;               // multiplies M_p in the 'mass' Schur variant
};

// Auxiliary operators for the H(curl) preconditioners, built once per set by
// SolverManager::setupBlockTriangularAuxiliary and shared by RefMaxwell, Maxwell1 and
// the Hiptmair smoother. D0/M1/Kn are defined above the
// buildRefMaxwellPreconditioner (block_prec/MaxwellInverse.hpp).
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
  Teuchos::RCP<LA_CrsMatrix> D0_normalized;
};

// One named group of variables receiving a single approximate inverse, PETSc's PCFIELDSPLIT
// idea. Each split keeps its own preconditioner hierarchy so reuse never crosses splits.
template<class Node>
struct FieldSplit {
  std::string name;                   // key from 'variable groups'
  std::string variables;              // that key's comma-separated variable list
  Teuchos::ParameterList settings;    // the split sublist
  // AMG, RefMaxwell, Maxwell1, Direct or Diagonal; the block-diagonal path also accepts
  // an Ifpack2 smoother name here.
  std::string prec_type;
  Teuchos::RCP<Teuchos::ParameterList> refmaxwell_xml, maxwell1_xml;
  Teuchos::RCP<MueLu::RefMaxwell<ScalarT,LO,GO,Node> > refmaxwell_prec;
  Teuchos::RCP<MueLu::Maxwell1<ScalarT,LO,GO,Node> > maxwell1_prec;
  ScalarT addon_beta_built = 0.0;     // beta baked into refmaxwell_prec
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
    parseSplitSublists();
    parseSchemeSettings();
    initializeRuntimeState();
  }

  size_t numSplits() const { return splits.size(); }

  const FieldSplit<Node> & split(const size_t r) const {
    TEUCHOS_TEST_FOR_EXCEPTION(r >= splits.size(), std::runtime_error,
      "Block preconditioner split " << r << " is not defined; this deck has "
      << splits.size() << " splits.");
    return splits[r];
  }

  const Teuchos::ParameterList & splitSettings(const size_t r) const { return split(r).settings; }

  std::string splitPrecType(const size_t r) const { return split(r).prec_type; }

  // Elimination weights run inside a group only, so a split with no group-mate before it
  // is inverted as given rather than on a Schur approximation.
  size_t splitGroupOf(const size_t r) const {
    return split_group_of.empty() ? 0 : split_group_of[r];
  }

  // A block-diagonal group ignores the coupling between its own splits, so its members get
  // no Schur correction from each other either.
  bool splitGroupIsJacobi(const size_t g) const {
    return g < split_group_jacobi.size() && split_group_jacobi[g];
  }

  // Split order, for block_prec::resolveVariableGroups.
  std::vector<std::string> splitVariableSpecs() const {
    std::vector<std::string> out;
    for (size_t r = 0; r < splits.size(); ++r) out.push_back(splits[r].variables);
    return out;
  }

  void reset() {
    have_matrix = false;
    have_preconditioner = false;
    matrix = Teuchos::null;
    prec = Teuchos::null;
    prec_dd = Teuchos::null;
    prec_block = Teuchos::null;
    for (size_t r = 0; r < splits.size(); ++r) {
      splits[r].refmaxwell_prec = Teuchos::null;
      splits[r].maxwell1_prec = Teuchos::null;
    }
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
  // One container per scheme, one sublist per split, Schur target last.
  Teuchos::ParameterList scheme_sublist;
  std::string scheme_sublist_name;
  std::string monolithic_sublist_name;
  std::string schur_target_split;
  std::vector<FieldSplit<Node> > splits;      // split order, Schur target last in its group
  // Optional 'split groups': Gauss-Seidel over groups, each composed internally by 'group
  // composition'. Empty means the flat sweep BlockTriangularFactory does.
  std::vector<std::string> split_group_names;
  std::vector<std::vector<size_t> > split_groups;
  std::vector<size_t> split_group_of;        // group id per split, empty without groups
  // Per group: block diagonal over its splits, or a pivot/target chain through them.
  std::vector<bool> split_group_jacobi;
  // The Schur target is the last split of its group, so without groups it is the last
  // split overall. Only this split takes the 'mass' term and the RefMaxwell addon.
  size_t schur_target_index = 0;

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

  size_t equation_set_index;   // set index when linearSolver(set,...) is used

  // One split's XML, parsed once. Callers that modify it must copy!
  const Teuchos::ParameterList & refMaxwellParams(const size_t r,
                                                  const Teuchos::Comm<int> & comm) {
    return splitXml(splits[r].refmaxwell_xml, splits[r].settings, "RefMaxwell Settings", comm);
  }

  const Teuchos::ParameterList & maxwell1Params(const size_t r,
                                                const Teuchos::Comm<int> & comm) {
    return splitXml(splits[r].maxwell1_xml, splits[r].settings, "Maxwell1 Settings", comm);
  }

  bool schurAddonWanted(const Teuchos::Comm<int> & comm) {
    const size_t target = schur_target_index;
    if (block_prec::parseBlockPrecType(split(target).prec_type) !=
        block_prec::BlockPrecType::RefMaxwell) {
      return false;
    }
    return block_prec::refMaxwellAddonEnabled(refMaxwellParams(target, comm));
  }

  static std::string splitXmlFile(const Teuchos::ParameterList & split, const char * sublistName) {
    if (!split.isSublist(sublistName)) return "";
    const Teuchos::ParameterList & sub = split.sublist(sublistName);
    return sub.isParameter("xml param file") ? sub.template get<std::string>("xml param file")
                                             : std::string("");
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

  const Teuchos::ParameterList & splitXml(Teuchos::RCP<Teuchos::ParameterList> & cached,
                                         const Teuchos::ParameterList & settings,
                                         const char * sublistName,
                                         const Teuchos::Comm<int> & comm) {
    if (cached.is_null()) {
      cached = Teuchos::rcp(new Teuchos::ParameterList());
      const std::string file = splitXmlFile(settings, sublistName);
      if (!file.empty()) {
        block_prec::loadXmlBroadcast(file, *cached, comm, sublistName);
      }
    }
    return *cached;
  }

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
    }
    parseSchemeContainer(settings);
  }

  // One container per scheme, holding the grouping and one sublist per split. Named splits
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

    // 'variable groups' is an ordered name -> variable-list map; input order is split order.
    TEUCHOS_TEST_FOR_EXCEPTION(!scheme_sublist.isSublist("variable groups"), std::runtime_error,
      scheme_sublist_name << " requires a 'variable groups' sublist naming each split.");
    const Teuchos::ParameterList & groups = scheme_sublist.sublist("variable groups");
    for (Teuchos::ParameterList::ConstIterator it = groups.begin(); it != groups.end(); ++it) {
      FieldSplit<Node> split;
      split.name = groups.name(it);
      split.variables = groups.get<std::string>(split.name);
      splits.push_back(split);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(splits.size() < 2, std::runtime_error,
      scheme_sublist_name << " 'variable groups' needs at least two splits, got "
      << splits.size() << ".");

    if (scheme_sublist.isParameter("schur target")) {
      schur_target_split = scheme_sublist.get<std::string>("schur target");
    }
    orderSplits();
    collectSplitSublists();
  }

  // Split order is the concatenation of the declared groups, each in its listed order.
  // Without the key every split is its own group.
  void orderSplits() {
    if (!scheme_sublist.isSublist("split groups")) {
      orderSplitsTargetLast();
      schur_target_index = splits.size() - 1;
      return;
    }
    // Both paths default the target to the last split, so adding 'split groups' to a deck
    // that names no 'schur target' does not move it.
    const Teuchos::ParameterList & groups = scheme_sublist.sublist("split groups");
    std::vector<FieldSplit<Node> > ordered;
    std::vector<bool> seen(splits.size(), false);
    for (Teuchos::ParameterList::ConstIterator it = groups.begin(); it != groups.end(); ++it) {
      const std::string name = groups.name(it);
      // A group is either 'name: a, b' or a sublist with 'splits' and 'composition'.
      std::string spec;
      bool jacobi = false;
      if (groups.isSublist(name)) {
        const Teuchos::ParameterList & entry = groups.sublist(name);
        TEUCHOS_TEST_FOR_EXCEPTION(!entry.isParameter("splits"), std::runtime_error,
          "'split groups' entry '" << name << "' is a sublist, so it needs a 'splits' key "
          "naming the splits it holds.");
        spec = entry.get<std::string>("splits");
        if (entry.isParameter("composition")) {
          const std::string how =
            block_prec::toUpperAsciiCopy(entry.get<std::string>("composition"));
          TEUCHOS_TEST_FOR_EXCEPTION(how != "JACOBI" && how != "GAUSS-SEIDEL",
            std::runtime_error,
            "'split groups' entry '" << name << "' has composition '"
            << entry.get<std::string>("composition")
            << "'. Supported: gauss-seidel for a pivot/target chain through the group's "
            "splits, jacobi for a block-diagonal inverse over them.");
          jacobi = (how == "JACOBI");
        }
      }
      else {
        spec = groups.get<std::string>(name);
      }
      std::vector<size_t> members;
      for (const std::string & member : block_prec::splitCommaList(spec)) {
        const size_t at = splitIndexByName(member);
        TEUCHOS_TEST_FOR_EXCEPTION(at == splits.size(), std::runtime_error,
          "'split groups' entry '" << name << "' names '" << member
          << "', which is not a group in 'variable groups'. Declared: "
          << declaredSplitNames() << ".");
        TEUCHOS_TEST_FOR_EXCEPTION(seen[at], std::runtime_error,
          "'split groups' names '" << member << "' more than once.");
        seen[at] = true;
        members.push_back(at);
      }
      TEUCHOS_TEST_FOR_EXCEPTION(members.empty(), std::runtime_error,
        "'split groups' entry '" << name << "' names no splits.");
      // The target is only ever the last of its group: it is the one split in the group
      // that is never inverted as an elimination weight.
      for (size_t m = 0; m + 1 < members.size(); ++m) {
        if (splits[members[m]].name != schur_target_split) continue;
        members.push_back(members[m]);
        members.erase(members.begin() + m);
        break;
      }
      std::vector<size_t> indices;
      for (size_t m = 0; m < members.size(); ++m) {
        indices.push_back(ordered.size());
        ordered.push_back(splits[members[m]]);
        split_group_of.push_back(split_groups.size());
      }
      split_group_names.push_back(name);
      split_groups.push_back(indices);
      split_group_jacobi.push_back(jacobi);
    }
    for (size_t r = 0; r < splits.size(); ++r) {
      TEUCHOS_TEST_FOR_EXCEPTION(!seen[r], std::runtime_error,
        "'split groups' leaves split '" << splits[r].name
        << "' out; every split must appear exactly once.");
    }
    splits = ordered;
    schur_target_index = splits.size() - 1;
    if (!schur_target_split.empty()) {
      schur_target_index = splitIndexByName(schur_target_split);
      TEUCHOS_TEST_FOR_EXCEPTION(schur_target_index == splits.size(), std::runtime_error,
        "'schur target: " << schur_target_split << "' names no group in 'variable groups'.");
    }
  }

  // A deck may write 'damping: 2' rather than '2.0', and Teuchos stores that as an int.
  static ScalarT scalarParam(const Teuchos::ParameterList & pl, const char * key,
                             const ScalarT fallback) {
    if (!pl.isParameter(key)) return fallback;
    if (pl.getEntry(key).isType<int>()) return static_cast<ScalarT>(pl.get<int>(key));
    return pl.get<ScalarT>(key);
  }

  // splits.size() is 2 to 5, so a scan beats keeping an index in step with the reorder.
  size_t splitIndexByName(const std::string & name) const {
    for (size_t r = 0; r < splits.size(); ++r) {
      if (splits[r].name == name) return r;
    }
    return splits.size();
  }

  std::string declaredSplitNames() const {
    std::string out;
    for (size_t r = 0; r < splits.size(); ++r) {
      if (r) out += ", ";
      out += "'" + splits[r].name + "'";
    }
    return out;
  }

  // Only the last split escapes being used as an elimination weight, so the target goes
  // there regardless of the order the groups were listed in.
  void orderSplitsTargetLast() {
    if (schur_target_split.empty()) return;
    const size_t at = splitIndexByName(schur_target_split);
    TEUCHOS_TEST_FOR_EXCEPTION(at == splits.size(), std::runtime_error,
      "'schur target: " << schur_target_split << "' names no group in 'variable groups'.");
    splits.push_back(splits[at]);
    splits.erase(splits.begin() + at);
  }

  void collectSplitSublists() {
    for (size_t r = 0; r < splits.size(); ++r) {
      TEUCHOS_TEST_FOR_EXCEPTION(!scheme_sublist.isSublist(splits[r].name), std::runtime_error,
        scheme_sublist_name << " has no '" << splits[r].name
        << "' sublist for the split of that name.");
      splits[r].settings = scheme_sublist.sublist(splits[r].name);
      splits[r].prec_type = splits[r].settings.isParameter("preconditioner")
        ? splits[r].settings.template get<std::string>("preconditioner")
        : std::string("AMG");
    }
    // Anything else that is a sublist is a typo, not an ignorable extra.
    for (Teuchos::ParameterList::ConstIterator it = scheme_sublist.begin();
         it != scheme_sublist.end(); ++it) {
      const std::string key = scheme_sublist.name(it);
      if (!scheme_sublist.isSublist(key) || key == "variable groups" ||
          key == "split groups") continue;
      bool known = false;
      for (size_t r = 0; r < splits.size(); ++r) known = known || splits[r].name == key;
      TEUCHOS_TEST_FOR_EXCEPTION(!known, std::runtime_error,
        scheme_sublist_name << " sublist '" << key
        << "' names no group in 'variable groups'.");
    }
  }

  // Split sublists themselves carry arbitrary MueLu/Ifpack2 keys, so only the nested
  // RefMaxwell/Maxwell1 lists can be checked against a fixed key set.
  void validateSublists() {
    for (size_t r = 0; r < splits.size(); ++r) {
      block_prec::validateNestedBlockSublists(splits[r].settings,
                                              scheme_sublist_name + "." + splits[r].name);
    }
  }

  void parseGeneralSettings(Teuchos::ParameterList & settings) {
    use_direct = settings.get<bool>("use direct solver",false);
    prec_type = block_prec::canonicalPreconditionerType(settings.get<string>("preconditioner type","AMG"));
    // A scheme container that does not match the selected scheme is a silent no-op
    if (!scheme_sublist_name.empty()) {
      const std::string want = (scheme_sublist_name == "Block Triangular Settings")
        ? "block triangular" : "block diagonal";
      TEUCHOS_TEST_FOR_EXCEPTION(prec_type != want, std::runtime_error,
        "'" << scheme_sublist_name << "' is present but 'preconditioner type' is '"
        << prec_type << "'; it must be '" << want << "'.");
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(prec_type == "block triangular" || prec_type == "block diagonal",
        std::runtime_error,
        "'preconditioner type: " << prec_type << "' requires a '"
        << (prec_type == "block triangular" ? "Block Triangular Settings"
                                            : "Block Diagonal Settings")
        << "' sublist naming the splits in 'variable groups'.");
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
    schur.mass_scale = 1.0;
    schur.damping = Teuchos::ScalarTraits<ScalarT>::one();
    schur.correction_use_lumped_weight = false;
    schur.triangle = "auto";
    schur.diagonal_prec_use_lumped = false;
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

  void parseSplitSublists() {
    if (splits.empty()) return;
    const Teuchos::ParameterList & pivot = splits.front().settings;
    if (pivot.isParameter("diag use lumped diagonal")) {
      schur.diagonal_prec_use_lumped =
        pivot.template get<bool>("diag use lumped diagonal");
    }
  }

  void parseSchemeSettings() {
    if (scheme_sublist_name.empty()) return;
    if (scheme_sublist.isParameter("approximation type")) {
      schur.approximation_type =
        block_prec::canonicalSchurApproximationType(scheme_sublist.get<string>("approximation type"));
    }
    schur.mass_scale = scalarParam(scheme_sublist, "mass scale", schur.mass_scale);
    if (scheme_sublist.isParameter("diag use lumped pivot diagonal")) {
      schur.correction_use_lumped_weight =
        scheme_sublist.get<bool>("diag use lumped pivot diagonal");
    }
    if (scheme_sublist.isParameter("triangle")) {
      schur.triangle = block_prec::canonicalSchurTriangle(scheme_sublist.get<string>("triangle"));
    }
    schur.damping = scalarParam(scheme_sublist, "damping", schur.damping);
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
