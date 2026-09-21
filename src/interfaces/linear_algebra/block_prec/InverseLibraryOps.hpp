#ifndef MRHYDE_BLOCK_PREC_INVERSE_LIBRARY_OPS_HPP
#define MRHYDE_BLOCK_PREC_INVERSE_LIBRARY_OPS_HPP

#include "block_prec/TekoAdapter.hpp"

#include <Teko_InverseFactory.hpp>
#include <Teko_InverseLibrary.hpp>
#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Stratimikos_MueLuHelpers.hpp>

#include <string>

namespace MrHyDE {
namespace block_prec {

/** Registry for the sub-solvers Teko can build itself: Amesos2, Ifpack2,
 *  MueLu and Belos.
 */
template<class Node>
inline Teuchos::RCP<Teko::InverseLibrary> genericInverseLibrary() {
  static Teuchos::RCP<Teko::InverseLibrary> lib;
  if (lib.is_null()) {
    Teuchos::RCP<Stratimikos::DefaultLinearSolverBuilder> builder =
      Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder);
    Stratimikos::enableMueLu<ScalarT,LO,GO,Node>(*builder);
    lib = Teko::InverseLibrary::buildFromStratimikos(builder);
  }
  return lib;
}

/** Build one sub-solver inverse.
 */
template<class Node>
inline Teko::LinearOp buildLibraryInverse(const std::string & type,
                                          const Teuchos::ParameterList & params,
                                          const std::string & label,
                                          const typename BlockTypes<Node>::CrsMatrixRCP & A) {
  Teuchos::ParameterList entry(params);
  entry.set("Type", type);
  Teuchos::RCP<Teko::InverseLibrary> lib = genericInverseLibrary<Node>();
  lib->addInverse(label, entry);
  return Teko::buildInverse(*lib->getInverseFactory(label), tpetraToThyraConst<Node>(A));
}

template<class Node>
inline Teko::LinearOp buildLibraryInverse(const std::string & type,
                                          const Teuchos::ParameterList & params,
                                          const std::string & label,
                                          const typename BlockTypes<Node>::CrsMatrixRCP & A,
                                          const Teko::LinearOp & precOp) {
  Teuchos::ParameterList entry(params);
  entry.set("Type", type);
  Teuchos::RCP<Teko::InverseLibrary> lib = genericInverseLibrary<Node>();
  lib->addInverse(label, entry);
  return Teko::buildInverse(*lib->getInverseFactory(label),
                            tpetraToThyraConst<Node>(A), precOp);
}

} // namespace block_prec
} // namespace MrHyDE

#endif
