/***********************************************************************
MrHyDE - Parameter canonicalization, validation, filtering, and RefMaxwell assembly.
Owns key normalization and safe list sanitization before dispatching settings
into MueLu/Ifpack2/RefMaxwell. Does not own block extraction or operator build.
Read with BlockTypes first, then this file, then BlockAssembly/solvers call sites.
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_PARAM_UTILS_HPP
#define MRHYDE_BLOCK_PREC_PARAM_UTILS_HPP

#include "block_prec/BlockTypes.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_TestForException.hpp>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fstream>

namespace MrHyDE {

// Not Teuchos::updateParametersFromXmlFileAndBroadcast: it throws on rank 0
// before its broadcast, so a bad path hangs the other ranks.
inline void loadXmlBroadcast(const std::string & file,
                             Teuchos::ParameterList & out,
                             const Teuchos::Comm<int> & comm,
                             const std::string & context) {
  std::string text;
  int len = -1;
  if (comm.getRank() == 0) {
    std::ifstream in(file.c_str());
    if (in) {
      std::ostringstream ss;
      ss << in.rdbuf();
      text = ss.str();
      len = static_cast<int>(text.size());
    }
  }
  Teuchos::broadcast<int,int>(comm, 0, 1, &len);
  TEUCHOS_TEST_FOR_EXCEPTION(len < 0, std::runtime_error,
    "Cannot open XML file '" << file << "' for " << context << ".");
  text.resize(len);
  if (len > 0) Teuchos::broadcast<int,char>(comm, 0, len, &text[0]);
  Teuchos::updateParametersFromXmlString(text, Teuchos::ptr(&out));
}

template<class Node>
class LinearSolverContext;

inline std::string canonicalPreconditionerType(const std::string & raw) {
  const std::string u = toUpperAsciiCopy(raw);
  if (u == "AMG" || u == "MUELU") return "AMG";
  if (u == "IFPACK2") return "Ifpack2";
  if (u == "DOMAIN DECOMPOSITION") return "domain decomposition";
  if (u == "BLOCK DIAGONAL") return "block diagonal";
  if (u == "BLOCK TRIANGULAR") return "block triangular";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Unsupported preconditioner type '" << raw
    << "'. Supported values: AMG, Ifpack2, domain decomposition, block diagonal, block triangular.");
  return "AMG";
}

inline std::string canonicalBlockPrecType(const std::string & raw) {
  const BlockPrecType t = parseBlockPrecType(raw);
  if (t == BlockPrecType::AMG) return "AMG";
  if (t == BlockPrecType::RefMaxwell) return "RefMaxwell";
  if (t == BlockPrecType::Maxwell1) return "Maxwell1";
  if (t == BlockPrecType::Direct) return "Direct";
  return "Diagonal";
}

inline std::string canonicalSchurApproximationType(const std::string & raw) {
  return schurVariantName(parseSchurVariant(raw));
}

inline std::string canonicalSchurTriangle(const std::string & raw) {
  return triangleSideName(parseTriangleSide(raw));
}

// Hierarchy: keep the preconditioner and re-setup it against the new J, which
// is where MueLu applies its own 'reuse: type'. Operator: keep it untouched.
inline bool reuseKeepsHierarchy(const std::string & t) {
  return t == "update" || t == "full";
}

inline bool reuseKeepsOperator(const std::string & t, const bool jacobianRebuilt) {
  return t == "full" || (t == "update" && !jacobianRebuilt);
}

inline std::string canonicalReuseType(const std::string & raw) {
  const std::string u = toUpperAsciiCopy(raw);
  if (u == "NONE") return "none";
  if (u == "UPDATE") return "update";
  if (u == "FULL") return "full";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Unsupported preconditioner reuse type '" << raw
    << "'. Supported values: none, update, full.");
  return "update";
}

// Load a complete MueLu parameter list from XML when configured.
inline bool loadMueLuXmlIfPresent(const Teuchos::ParameterList & amgSublist,
                                  Teuchos::ParameterList & outParams,
                                  const std::string & context,
                                  const Teuchos::RCP<const Teuchos::Comm<int> > & comm) {
  if (!amgSublist.isParameter("xml param file")) return false;
  const std::string xmlFile = amgSublist.get<std::string>("xml param file");
  if (xmlFile.empty()) return false;
  loadXmlBroadcast(xmlFile, outParams, *comm, context);
  return true;
}

// Map integer verbosity to MueLu string (none/low/medium/high).
inline void normalizeMueLuVerbosity(Teuchos::ParameterList & mueluParams, const int verbosity) {
  if (mueluParams.isParameter("verbosity") && mueluParams.getEntry("verbosity").isType<int>()) {
    const int v = mueluParams.get<int>("verbosity");
    mueluParams.set("verbosity", std::string(v <= 0 ? "none" : v <= 1 ? "low" : v <= 2 ? "medium" : "high"));
  }
  if (verbosity >= 20) {
    mueluParams.set("verbosity", "high");
  }
}

inline Teuchos::ParameterList defaultMueLuParams() {
  Teuchos::ParameterList mueluParams;
  mueluParams.set("verbosity", "none");
  mueluParams.set("coarse: max size", 500);
  mueluParams.set("coarse: type", "KLU");
  mueluParams.set("multigrid algorithm", "sa");
  mueluParams.set("aggregation: type", "uncoupled");
  mueluParams.set("aggregation: drop scheme", "classical");
  mueluParams.set("smoother: type", "CHEBYSHEV");
  mueluParams.set("repartition: enable", false);
  mueluParams.set("reuse: type", "none");
  mueluParams.setName("MueLu");
  return mueluParams;
}

inline Teuchos::ParameterList validRefMaxwellParams() {
  Teuchos::ParameterList v("RefMaxwell Settings");
  v.set("xml param file", "");
  v.set("filter SM", false);
  v.set("filter threshold", 1.0e-14);
  v.set("verify complex", false);
  return v;
}

inline Teuchos::ParameterList validMaxwell1Params() {
  Teuchos::ParameterList v = validRefMaxwellParams();
  v.setName("Maxwell1 Settings");
  v.set("verify Kn consistency", false);
  v.set("use Kn from M1", true);
  return v;
}

inline Teuchos::ParameterList validPivotBlockParams() {
  Teuchos::ParameterList v("Pivot Block Settings");
  v.set("preconditioner type", "AMG");
  v.set("diag use lumped diagonal", false);
  v.set("hgrad basis name", "");
  v.set("hcurl basis name", "");
  v.set("inner krylov solver", "");
  v.set("inner krylov max iters", 5);
  v.set("inner krylov tol", 1.0e-2);
  // MueLu and the nested MrHyDE lists are validated on their own terms.
  v.sublist("AMG Settings").disableRecursiveValidation();
  v.sublist("RefMaxwell Settings").disableRecursiveValidation();
  v.sublist("Maxwell1 Settings").disableRecursiveValidation();
  return v;
}

inline Teuchos::ParameterList validSchurBlockParams() {
  Teuchos::ParameterList v = validPivotBlockParams();
  v.setName("Schur Block Settings");
  v.set("approximation type", "base");
  v.set("pivot block", 0);
  v.set("triangle", "auto");
  v.set("damping", 1.0);
  v.set("diag use lumped pivot diagonal", false);
  v.set("smoother: type", "");
  v.sublist("smoother: params").disableRecursiveValidation();
  return v;
}

inline void validateRefMaxwellSettingsSection(const Teuchos::ParameterList & list, const std::string &) {
  list.validateParameters(validRefMaxwellParams());
}

inline void validateMaxwell1SettingsSection(const Teuchos::ParameterList & list, const std::string &) {
  list.validateParameters(validMaxwell1Params());
}

inline void validateNestedBlockSublists(const Teuchos::ParameterList & list, const std::string & sectionName) {
  if (list.isSublist("RefMaxwell Settings")) {
    validateRefMaxwellSettingsSection(list.sublist("RefMaxwell Settings"), sectionName + ".RefMaxwell Settings");
  }
  if (list.isSublist("Maxwell1 Settings")) {
    validateMaxwell1SettingsSection(list.sublist("Maxwell1 Settings"), sectionName + ".Maxwell1 Settings");
  }
}

inline void validatePivotBlockSettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  list.validateParameters(validPivotBlockParams());
  if (list.isParameter("preconditioner type")) {
    canonicalBlockPrecType(list.get<std::string>("preconditioner type"));
  }
  validateNestedBlockSublists(list, sectionName);
}

inline void validateSchurBlockSettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  list.validateParameters(validSchurBlockParams());
  if (list.isParameter("preconditioner type")) {
    canonicalBlockPrecType(list.get<std::string>("preconditioner type"));
  }
  if (list.isParameter("approximation type")) {
    canonicalSchurApproximationType(list.get<std::string>("approximation type"));
  }
  if (list.isParameter("triangle")) {
    canonicalSchurTriangle(list.get<std::string>("triangle"));
  }
  validateNestedBlockSublists(list, sectionName);
}

// Promote all params from a sublist to top level (for Ifpack2 Chebyshev: smoother: params -> top).
inline void promoteSublistToTopLevel(Teuchos::ParameterList & list, const std::string & sublistName) {
  if (!list.isSublist(sublistName)) return;
  const Teuchos::ParameterList & sub = list.sublist(sublistName);
  for (Teuchos::ParameterList::ConstIterator it = sub.begin(); it != sub.end(); ++it) {
    const std::string key = sub.name(it);
    if (sub.isSublist(key)) continue;
    if (sub.isType<int>(key))
      list.set(key, sub.get<int>(key));
    else if (sub.isType<double>(key))
      list.set(key, sub.get<double>(key));
    else if (sub.isType<std::string>(key))
      list.set(key, sub.get<std::string>(key));
    else if (sub.isType<bool>(key))
      list.set(key, sub.get<bool>(key));
  }
  list.remove(sublistName, false);
}

// Keys consumed by MrHyDE before dispatching to MueLu/Ifpack2.
inline const std::vector<std::string> & mrhydeOwnedKeys() {
  static const std::vector<std::string> keys = {
    "preconditioner type", "preconditioner variant", "use mass matrix", "xml param file",
    "hgrad basis name", "hcurl basis name",
    "hgrad basis order", "hcurl basis order",
    "inner krylov solver", "inner krylov max iters", "inner krylov tol",
    "approximation type", "pivot block", "triangle", "damping",
    "diag use lumped diagonal", "diag use lumped pivot diagonal",
    "filter SM", "filter threshold", "verify complex", "verify Kn consistency",
    "use Kn from M1"
  };
  return keys;
}

inline const std::vector<std::string> & mrhydeOwnedSublists() {
  static const std::vector<std::string> keys = {
    "AMG Settings", "RefMaxwell Settings", "Maxwell1 Settings"
  };
  return keys;
}

inline void removeMrHyDEOwnedKeys(Teuchos::ParameterList & list) {
  for (const auto & k : mrhydeOwnedKeys()) list.remove(k, false);
  for (const auto & k : mrhydeOwnedSublists()) list.remove(k, false);
}

inline void removeIfpack2OnlyKeys(Teuchos::ParameterList & list) {
  static const char * prefixes[] = {
    "relaxation: ", "chebyshev: ", "partitioner: ", "fact: ", "schwarz: "
  };
  std::vector<std::string> removeKeys;
  for (Teuchos::ParameterList::ConstIterator it = list.begin(); it != list.end(); ++it) {
    const std::string key = list.name(it);
    if (list.isSublist(key)) continue;
    for (size_t p = 0; p < sizeof(prefixes) / sizeof(prefixes[0]); ++p) {
      if (key.rfind(prefixes[p], 0) == 0) {
        removeKeys.push_back(key);
        break;
      }
    }
  }
  for (size_t i = 0; i < removeKeys.size(); ++i) {
    list.remove(removeKeys[i], false);
  }
}

inline bool isHiptmairSmoother(const std::string & type) {
  return toUpperAsciiCopy(type).find("HIPTMAIR") != std::string::npos;
}

// Check top-level and per-level smoother settings.
inline bool mueluParamsWantHiptmair(const Teuchos::ParameterList & pl) {
  if (pl.isParameter("smoother: type") && isHiptmairSmoother(pl.get<std::string>("smoother: type"))) return true;
  for (Teuchos::ParameterList::ConstIterator it = pl.begin(); it != pl.end(); ++it) {
    const std::string & key = pl.name(it);
    if (key.rfind("level ", 0) != 0 || !pl.isSublist(key)) continue;
    const auto & sub = pl.sublist(key);
    if (sub.isParameter("smoother: type") && isHiptmairSmoother(sub.get<std::string>("smoother: type"))) return true;
  }
  return false;
}

inline void sanitizeDirectCoarseParams(Teuchos::ParameterList & sublist) {
  static const std::set<std::string> directCoarse = {
    "DIRECTSOLVER", "KLU", "KLU2", "AMESOS2", "AMESOS-KLU", "AMESOS-KLU2",
    "SUPERLU", "SUPERLU_DIST"};
  if (!sublist.isParameter("coarse: type") || !sublist.isSublist("coarse: params")) return;
  if (!directCoarse.count(toUpperAsciiCopy(sublist.get<std::string>("coarse: type")))) return;
  // Dropping the whole sublist would also discard legitimate Amesos2 options.
  removeIfpack2OnlyKeys(sublist.sublist("coarse: params"));
}

// Print warning on rank 0 if RefMaxwell sublist uses Krylov smoother.
inline void warnNonStationarySmoother(const Teuchos::ParameterList & refmaxwellParams,
                                      const std::string & listName,
                                      const Teuchos::RCP<const Teuchos::Comm<int> > & comm) {
  if (!refmaxwellParams.isSublist(listName)) return;
  const auto & sub = refmaxwellParams.sublist(listName);
  if (!sub.isParameter("smoother: type")) return;
  std::string stype = sub.get<std::string>("smoother: type");
  toUpperAscii(stype);
  if (stype == "CG" || stype == "GMRES" || stype == "BICGSTAB" ||
      stype == "BLOCK CG" || stype == "BLOCK GMRES") {
    if (comm != Teuchos::null && comm->getRank() == 0) {
      std::cout << "WARNING: RefMaxwell " << listName
                << " smoother type '" << stype
                << "' is a Krylov solver. This makes the preconditioner "
                << "non-stationary and can cause outer GMRES stagnation."
                << std::endl;
    }
  }
}

} // namespace MrHyDE
#endif
