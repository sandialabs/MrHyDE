#ifndef MRHYDE_BLOCK_PREC_INVERSE_LIBRARY_OPS_HPP
#define MRHYDE_BLOCK_PREC_INVERSE_LIBRARY_OPS_HPP

#include "block_prec/TekoAdapter.hpp"

#include <Teko_InverseFactory.hpp>
#include <Teko_InverseLibrary.hpp>
#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Stratimikos_MueLuHelpers.hpp>

#include <iostream>
#include <map>
#include <string>

namespace MrHyDE {
namespace block_prec {

// Teko's sub-solver registry plus the inverses built from it; labels must be
// unique per block. Per context: addInverse retains every list it is handed.
template<class Node>
class InverseLibraryCache {
public:
  using CrsMatrixRCP = typename BlockTypes<Node>::CrsMatrixRCP;

  InverseLibraryCache(const bool reuse, const int verbosity, const int rank)
    : reuse_(reuse), verbosity_(verbosity), rank_(rank) {
    Teuchos::RCP<Stratimikos::DefaultLinearSolverBuilder> builder =
      Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder);
    Stratimikos::enableMueLu<ScalarT,LO,GO,Node>(*builder);
    Stratimikos::enableMueLuRefMaxwell<ScalarT,LO,GO,Node>(*builder);
    Stratimikos::enableMueLuMaxwell1<ScalarT,LO,GO,Node>(*builder);
    lib_ = Teko::InverseLibrary::buildFromStratimikos(builder);
  }

  // Teko::rebuildInverse re-initializes a cached inverse in place.
  Teko::LinearOp build(const std::string & type,
                       const Teuchos::ParameterList & params,
                       const std::string & label,
                       const CrsMatrixRCP & A,
                       const Teko::LinearOp & precOp = Teuchos::null) {
    Teuchos::RCP<Teko::InverseFactory> factory = registerInverse(type, params, label);
    Entry & entry = cache_[label];
    const Teko::LinearOp source = tpetraToThyraConst<Node>(A);
    const bool haveCached = reuse_ && entry.type == type && !entry.inverse.is_null()
                            && !carriesNodeMatrix(params);
    if (haveCached && precOp.is_null()) {
      Teko::rebuildInverse(*factory, source, entry.inverse);
    }
    else if (haveCached) {
      Teko::rebuildInverse(*factory, source, precOp, entry.inverse);
    }
    else {
      entry.type = type;
      entry.inverse = precOp.is_null() ? Teko::buildInverse(*factory, source)
                                       : Teko::buildInverse(*factory, source, precOp);
    }
    logBuild(label, haveCached ? "rebuilt in place" : "built new");
    return entry.inverse;
  }

private:
  struct Entry {
    std::string type;
    Teko::InverseLinearOp inverse;
  };

  // Re-registering drops the previous list, releasing its MueLu 'user data' RCPs.
  Teuchos::RCP<Teko::InverseFactory> registerInverse(const std::string & type,
                                                     const Teuchos::ParameterList & params,
                                                     const std::string & label) {
    Teuchos::ParameterList entry(params);
    entry.set("Type", type);
    lib_->addInverse(label, entry);
    return lib_->getInverseFactory(label);
  }

  // MueLu reuse swaps only the level-0 operator, so a stale D0^T A D0 would survive.
  static bool carriesNodeMatrix(const Teuchos::ParameterList & params) {
    return params.isSublist("user data") &&
           params.sublist("user data").isParameter("NodeMatrix");
  }

  void logBuild(const std::string & label, const char * action) const {
    if (verbosity_ < 10 || rank_ != 0) return;
    std::cout << "[InverseLibrary] " << label << ": " << action << std::endl;
  }

  Teuchos::RCP<Teko::InverseLibrary> lib_;
  std::map<std::string, Entry> cache_;
  bool reuse_ = false;
  int verbosity_ = 0;
  int rank_ = 0;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
