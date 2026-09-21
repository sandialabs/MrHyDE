#ifndef MRHYDE_BLOCK_PREC_OPERATORS_HPP
#define MRHYDE_BLOCK_PREC_OPERATORS_HPP

#include "block_prec/BlockTypes.hpp"


#include <algorithm>
#include <string>
#include <vector>

namespace MrHyDE {
namespace block_prec {
namespace detail {

template<class Node>
void requireNoTranspose(const Teuchos::ETransp mode, const std::string & opName) {
  TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::runtime_error,
                             opName + "::apply only supports NO_TRANS mode.");
}

// isSameAs is an all-reduce, so it runs once; local lengths are what the
// kernels index with, so those are checked every call.
template<class Node>
void checkApplyMaps(const typename BlockTypes<Node>::MultiVector & X,
                    const typename BlockTypes<Node>::MultiVector & Y,
                    const typename BlockTypes<Node>::MapRCP & map,
                    bool & checked, const std::string & opName) {
  TEUCHOS_TEST_FOR_EXCEPTION(X.getLocalLength() != map->getLocalNumElements() ||
                             Y.getLocalLength() != map->getLocalNumElements(),
    std::runtime_error, opName + ": local length mismatch.");
  if (checked) return;
  TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*map) || !Y.getMap()->isSameAs(*map),
    std::runtime_error, opName + ": map mismatch.");
  checked = true;
}


} // namespace detail

/** Applies diag(M)^{-1} as a Tpetra::Operator (point-Jacobi or lumped inverse). */
template<class Node>
class DiagonalInverseOperator : public Tpetra::Operator<ScalarT, LO, GO, Node> {
public:
  using Types = BlockTypes<Node>;
  using LA_Map = typename Types::Map;
  using LA_MultiVector = typename Types::MultiVector;
  using LA_Vector = typename Types::Vector;

  explicit DiagonalInverseOperator(const Teuchos::RCP<LA_Vector> & invDiagIn)
    : invDiag_(invDiagIn) {}

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return invDiag_->getMap(); }
  Teuchos::RCP<const LA_Map> getRangeMap() const override { return invDiag_->getMap(); }
  bool hasTransposeApply() const override { return true; }

  void apply(const LA_MultiVector & X, LA_MultiVector & Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS &&
                               mode != Teuchos::TRANS &&
                               mode != Teuchos::CONJ_TRANS,
      std::runtime_error,
      "DiagonalInverseOperator only supports NO_TRANS, TRANS, or CONJ_TRANS.");
    detail::checkApplyMaps<Node>(X, Y, invDiag_->getMap(), maps_checked_, "DiagonalInverseOperator");
    Y.elementWiseMultiply(alpha, *invDiag_, X, beta);
  }

private:
  Teuchos::RCP<LA_Vector> invDiag_;
  mutable bool maps_checked_ = false;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
