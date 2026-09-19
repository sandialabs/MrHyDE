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

inline void loadXmlBroadcast(const std::string & file,
                             Teuchos::ParameterList & out,
                             const Teuchos::Comm<int> & comm,
                             const std::string & context) {
  int ok = 1;
  if (comm.getRank() == 0) {
    std::ifstream probe(file.c_str());
    ok = probe.good() ? 1 : 0;
  }
  Teuchos::broadcast<int,int>(comm, 0, 1, &ok);
  TEUCHOS_TEST_FOR_EXCEPTION(ok == 0, std::runtime_error,
    "Cannot open XML file '" << file << "' for " << context << ".");
  Teuchos::updateParametersFromXmlFileAndBroadcast(file, Teuchos::ptr(&out), comm);
}

template<class Node>
class LinearSolverContext;

inline bool keyInSet(const std::string & key, const std::set<std::string> & allowed) {
  return allowed.find(key) != allowed.end();
}

inline void throwUnknownKey(const std::string & sectionName, const std::string & key,
                            const std::set<std::string> & allowedParams,
                            const std::set<std::string> & allowedSublists) {
  std::ostringstream msg;
  msg << "Unknown key '" << key << "' in section '" << sectionName << "'.";
  msg << "\nAllowed parameters:";
  for (std::set<std::string>::const_iterator it = allowedParams.begin(); it != allowedParams.end(); ++it) {
    msg << "\n  - " << *it;
  }
  msg << "\nAllowed sublists:";
  for (std::set<std::string>::const_iterator it = allowedSublists.begin(); it != allowedSublists.end(); ++it) {
    msg << "\n  - " << *it;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, msg.str());
}

inline void validateAllowedKeys(const Teuchos::ParameterList & list,
                                const std::string & sectionName,
                                const std::set<std::string> & allowedParams,
                                const std::set<std::string> & allowedSublists) {
  for (Teuchos::ParameterList::ConstIterator it = list.begin(); it != list.end(); ++it) {
    const std::string key = list.name(it);
    if (list.isSublist(key)) {
      if (!keyInSet(key, allowedSublists)) {
        throwUnknownKey(sectionName, key, allowedParams, allowedSublists);
      }
    }
    else if (!keyInSet(key, allowedParams)) {
      throwUnknownKey(sectionName, key, allowedParams, allowedSublists);
    }
  }
}

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
  return schurVariantName(parseSchurVariant(raw, raw));
}

inline std::string canonicalSchurTriangle(const std::string & raw) {
  return triangleSideName(parseTriangleSide(raw));
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

inline std::set<std::string> defaultRefMaxwellAllowedParams() {
  const char * keys[] = {
    "xml param file",
    "use lumped M0inv",
    "hgrad basis name", "hcurl basis name",
    "hgrad basis order", "hcurl basis order",
    "D0 file", "coordinates file",
    "filter SM", "filter threshold",
    "verify complex"
  };
  return std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0]));
}

inline std::set<std::string> defaultRefMaxwellAllowedSublists() {
  return std::set<std::string>();
}

inline std::set<std::string> defaultMaxwell1AllowedParams() {
  const char * keys[] = {
    "xml param file",
    "hgrad basis name", "hcurl basis name",
    "hgrad basis order", "hcurl basis order",
    "filter SM", "filter threshold",
    "verify complex", "verify Kn consistency"
  };
  return std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0]));
}

inline std::set<std::string> defaultMaxwell1AllowedSublists() {
  return std::set<std::string>();
}

inline void validateRefMaxwellSettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  validateAllowedKeys(list, sectionName, defaultRefMaxwellAllowedParams(), defaultRefMaxwellAllowedSublists());
}

inline void validateMaxwell1SettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  validateAllowedKeys(list, sectionName, defaultMaxwell1AllowedParams(), defaultMaxwell1AllowedSublists());
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
  const char * keys[] = {
    "preconditioner type", "diag use lumped diagonal",
    "hgrad basis name", "hcurl basis name",
    "inner krylov solver", "inner krylov max iters", "inner krylov tol"
  };
  const char * subkeys[] = {"AMG Settings", "RefMaxwell Settings", "Maxwell1 Settings"};
  validateAllowedKeys(list, sectionName,
    std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0])),
    std::set<std::string>(subkeys, subkeys + sizeof(subkeys) / sizeof(subkeys[0])));
  if (list.isParameter("preconditioner type")) {
    canonicalBlockPrecType(list.get<std::string>("preconditioner type"));
  }
  validateNestedBlockSublists(list, sectionName);
}

inline void validateSchurBlockSettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  const char * keys[] = {
    "preconditioner type", "approximation type", "pivot block", "triangle",
    "diag use lumped pivot diagonal",
    "hgrad basis name", "hcurl basis name",
    "smoother: type", "diag use lumped diagonal",
    "inner krylov solver", "inner krylov max iters", "inner krylov tol"
  };
  const char * subkeys[] = {"smoother: params", "AMG Settings", "RefMaxwell Settings", "Maxwell1 Settings"};
  validateAllowedKeys(list, sectionName,
    std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0])),
    std::set<std::string>(subkeys, subkeys + sizeof(subkeys) / sizeof(subkeys[0])));
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
    "use lumped M0inv", "refmaxwell: use lumped M0inv",
    "Schur approximation type", "Schur pivot block", "Schur triangle", "Schur damping",
    "Schur diag use lumped pivot diagonal",
    "Pivot block preconditioner type", "Pivot block diag use lumped diagonal",
    "approximation type", "pivot block", "triangle",
    "diag use lumped diagonal", "diag use lumped pivot diagonal",
    "filter SM", "filter threshold", "verify complex", "verify Kn consistency",
    "D0 file", "coordinates file"
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

// Remove coarse: params when coarse solver is direct (e.g. KLU).
inline void sanitizeDirectCoarseParams(Teuchos::ParameterList & sublist) {
  if (!sublist.isParameter("coarse: type")) return;
  const std::string coarseType = sublist.get<std::string>("coarse: type");
  const bool isDirect =
    coarseType == "DirectSolver" || coarseType == "DIRECTSOLVER" ||
    coarseType == "KLU" || coarseType == "Klu" || coarseType == "klu" ||
    coarseType == "Amesos2" || coarseType == "AMESOS2" ||
    coarseType == "Amesos-KLU" || coarseType == "Amesos-KLU2";
  if (isDirect) {
    sublist.remove("coarse: params", false);
  }
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
