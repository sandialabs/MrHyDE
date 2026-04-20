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

namespace MrHyDE {

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
  if (t == BlockPrecType::Direct) return "Direct";
  return "Diagonal";
}

inline std::string canonicalSchurApproximationType(const std::string & raw) {
  return schurVariantName(parseSchurVariant(raw, raw));
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

inline std::string canonicalBlockPrecBackend(const std::string & raw) {
  const std::string u = toUpperAsciiCopy(raw);
  if (u == "MRHYDE") return "mrhyde";
  if (u == "TEKO_HYBRID") return "teko_hybrid";
  if (u == "TEKO_FULL") return "teko_full";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Unsupported block preconditioner backend '" << raw
    << "'. Supported values: mrhyde, teko_hybrid, teko_full.");
  return "mrhyde";
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

// AMG parameters that are valid in YAML (includes both XML control and MueLu parameters)
inline std::set<std::string> defaultAmgAllowedParams() {
  const char * keys[] = {
    "xml param file",  // Optional: path to XML parameter file (takes precedence over YAML)
    "verbosity", "print initial parameters", "number of equations", "multigrid algorithm", "max levels",
    "smoother: type", "smoother: overlap", "smoother: pre or post", "coarse: type", "coarse: max size",
    "aggregation: type", "aggregation: drop tol", "aggregation: damping factor", "aggregation: min agg size",
    "aggregation: max agg size", "eigen-analysis: type", "problem: symmetric", "transpose: use implicit",
    "repartition: enable", "repartition: start level", "coarse: params", "parameterlist: syntax"
  };
  return std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0]));
}

inline std::set<std::string> defaultAmgAllowedSublists() {
  const char * keys[] = {"smoother: params", "coarse: params"};
  return std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0]));
}

inline std::set<std::string> defaultRefMaxwellAllowedParams() {
  const char * keys[] = {
    "xml param file",  // Required: path to XML parameter file
    // MrHyDE-specific parameters (not passed to MueLu):
    "use lumped M0inv",  // MrHyDE builds M0inv
    "hgrad basis name", "hcurl basis name",  // Basis function specification
    "hgrad basis order", "hcurl basis order",  // Basis order specification
    "D0 file", "coordinates file"  // Optional: load auxiliary data from files
    // Note: All MueLu RefMaxwell parameters must be specified in the XML file
  };
  return std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0]));
}

// No sublists allowed in RefMaxwell Settings - all MueLu config goes in XML
// There are too many refmaxwell parameters to list here, so we don't allow any sublists.
inline std::set<std::string> defaultRefMaxwellAllowedSublists() {
  return std::set<std::string>(); 
}

inline void validateAmgSettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  validateAllowedKeys(list, sectionName, defaultAmgAllowedParams(), defaultAmgAllowedSublists());
}

inline void validateRefMaxwellSettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  validateAllowedKeys(list, sectionName, defaultRefMaxwellAllowedParams(), defaultRefMaxwellAllowedSublists());
}

inline void validatePivotBlockSettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  const char * keys[] = {
    "preconditioner type", "diag use lumped diagonal", "strict RefMaxwell", "debug RefMaxwell maps",
    "hgrad basis name", "hcurl basis name"
  };
  const char * subkeys[] = {"AMG Settings", "RefMaxwell Settings", "ADS Settings"};
  validateAllowedKeys(list, sectionName,
    std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0])),
    std::set<std::string>(subkeys, subkeys + sizeof(subkeys) / sizeof(subkeys[0])));
  if (list.isParameter("preconditioner type")) {
    canonicalBlockPrecType(list.get<std::string>("preconditioner type"));
  }
  if (list.isSublist("AMG Settings")) {
    validateAmgSettingsSection(list.sublist("AMG Settings"), sectionName + ".AMG Settings");
  }
  if (list.isSublist("RefMaxwell Settings")) {
    validateRefMaxwellSettingsSection(list.sublist("RefMaxwell Settings"), sectionName + ".RefMaxwell Settings");
  }
}

inline void validateSchurBlockSettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  const char * keys[] = {
    "preconditioner type", "approximation type", "pivot block",
    "diag use lumped pivot diagonal", "strict RefMaxwell", "debug RefMaxwell maps",
    "hgrad basis name", "hcurl basis name",
    "smoother: type", "diag use lumped diagonal"
  };
  const char * subkeys[] = {"smoother: params", "AMG Settings", "RefMaxwell Settings", "ADS Settings"};
  validateAllowedKeys(list, sectionName,
    std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0])),
    std::set<std::string>(subkeys, subkeys + sizeof(subkeys) / sizeof(subkeys[0])));
  if (list.isParameter("preconditioner type")) {
    canonicalBlockPrecType(list.get<std::string>("preconditioner type"));
  }
  if (list.isParameter("approximation type")) {
    canonicalSchurApproximationType(list.get<std::string>("approximation type"));
  }
  if (list.isSublist("AMG Settings")) {
    validateAmgSettingsSection(list.sublist("AMG Settings"), sectionName + ".AMG Settings");
  }
  if (list.isSublist("RefMaxwell Settings")) {
    validateRefMaxwellSettingsSection(list.sublist("RefMaxwell Settings"), sectionName + ".RefMaxwell Settings");
  }
}

inline void validatePreconditionerSettingsSection(const Teuchos::ParameterList & list, const std::string & sectionName) {
  const char * keys[] = {
    "preconditioner type", "preconditioner variant", "strict RefMaxwell", "debug RefMaxwell maps",
    "Schur pivot block",
    "block prec backend",
    "Schur diag use lumped pivot diagonal", "Pivot block diag use lumped diagonal", "diag use lumped diagonal",
    "hgrad basis name", "hcurl basis name", "hgrad basis order", "hcurl basis order", "D0 file", "coordinates file",
    "smoother: type", "verbosity", "print initial parameters", "multigrid algorithm", "max levels",
    "cycle type", "sa: use filtered matrix", "sa: damping factor",
    "coarse: type", "coarse: max size", "number of equations", "aggregation: type", "aggregation: drop tol",
    "eigen-analysis: type", "relaxation: type", "relaxation: sweeps", "relaxation: damping factor",
    "relaxation: backward mode", "chebyshev: degree", "chebyshev: ratio eigenvalue",
    "chebyshev: min eigenvalue", "chebyshev: eigenvalue max iterations", "use lumped M0inv", "mode",
    "disable addon", "disable addon 22", "enable reuse", "use as preconditioner", "max coarse size"
  };
  const char * subkeys[] = {"smoother: params", "AMG Settings", "RefMaxwell Settings", "11list", "22list"};
  validateAllowedKeys(list, sectionName,
    std::set<std::string>(keys, keys + sizeof(keys) / sizeof(keys[0])),
    std::set<std::string>(subkeys, subkeys + sizeof(subkeys) / sizeof(subkeys[0])));
  if (list.isParameter("preconditioner type")) {
    canonicalPreconditionerType(list.get<std::string>("preconditioner type"));
  }
  if (list.isSublist("AMG Settings")) {
    validateAmgSettingsSection(list.sublist("AMG Settings"), sectionName + ".AMG Settings");
  }
  if (list.isSublist("RefMaxwell Settings")) {
    validateRefMaxwellSettingsSection(list.sublist("RefMaxwell Settings"), sectionName + ".RefMaxwell Settings");
  }
}

inline void keepAllowedKeys(Teuchos::ParameterList & list,
                            const std::set<std::string> & allowedParams,
                            const std::set<std::string> & allowedSublists) {
  std::vector<std::string> removeKeys;
  for (Teuchos::ParameterList::ConstIterator it = list.begin(); it != list.end(); ++it) {
    const std::string key = list.name(it);
    if (list.isSublist(key)) {
      if (!keyInSet(key, allowedSublists)) {
        removeKeys.push_back(key);
      }
    }
    else if (!keyInSet(key, allowedParams)) {
      removeKeys.push_back(key);
    }
  }
  for (size_t i = 0; i < removeKeys.size(); ++i) {
    list.remove(removeKeys[i], false);
  }
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

// Strip only context and dispatch keys for block-diagonal block list. Keeps relaxation:*
// and chebyshev:* so they reach Ifpack2; MueLu path strips those in buildAmgBlockOperator.
inline void stripBlockDiagonalBlockList(Teuchos::ParameterList & list) {
  const std::set<std::string> allowedParams = {
    "verbosity", "print initial parameters", "number of equations", "multigrid algorithm", "max levels",
    "smoother: type", "smoother: overlap", "smoother: pre or post", "coarse: type", "coarse: max size",
    "aggregation: type", "aggregation: drop tol", "aggregation: damping factor", "aggregation: min agg size",
    "aggregation: max agg size", "eigen-analysis: type", "problem: symmetric", "transpose: use implicit",
    "repartition: enable", "repartition: start level", "coarse: params", "parameterlist: syntax",
    "relaxation: type", "relaxation: sweeps", "relaxation: damping factor", "relaxation: backward mode",
    "chebyshev: degree", "chebyshev: ratio eigenvalue", "chebyshev: min eigenvalue",
    "chebyshev: eigenvalue max iterations"
  };
  const std::set<std::string> allowedSublists = {"smoother: params", "coarse: params"};
  keepAllowedKeys(list, allowedParams, allowedSublists);
}

// This function is only for block diagonal code path.
// It still uses YAML-based parameters.
inline void stripContextAndMethodKeys(Teuchos::ParameterList & list) {
  stripBlockDiagonalBlockList(list);
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
