#ifndef MRHYDE_BLOCK_PREC_OPERATORS_HPP
#define MRHYDE_BLOCK_PREC_OPERATORS_HPP

#include "block_prec/BlockTypes.hpp"

#include <KokkosKernels_ArithTraits.hpp>

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

template<class Node, class RowFunctor>
void forEachLocalRow(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > & mat,
                     RowFunctor && f) {
  using Types = BlockTypes<Node>;
  using HostInds = typename Types::HostInds;
  using HostVals = typename Types::HostVals;
  const LO nrows = mat->getLocalNumRows();
  const size_t maxEnt = std::max(size_t(1), mat->getLocalMaxNumRowEntries());
  HostInds colLids("col_lids", maxEnt);
  HostVals colVals("col_vals", maxEnt);
  auto colMap = mat->getColMap();
  for (LO rowLid = 0; rowLid < nrows; ++rowLid) {
    GO rowGid = mat->getRowMap()->getGlobalElement(rowLid);
    size_t nent = mat->getNumEntriesInLocalRow(rowLid);
    if (nent == 0) continue;
    mat->getLocalRowCopy(rowLid, colLids, colVals, nent);
    f(rowGid, colLids, colVals, nent, colMap);
  }
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
    const auto xView = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
    const auto dView = invDiag_->getLocalViewDevice(Tpetra::Access::ReadOnly);
    const size_t nrows = static_cast<size_t>(X.getLocalLength());
    const size_t nvec = static_cast<size_t>(X.getNumVectors());
    const bool useConjugate = (mode == Teuchos::CONJ_TRANS);
    const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
    // 0 * NaN = NaN in IEEE 754; Belos hands us uninitialized Y, so guard beta==0.
    const bool overwrite = (beta == zero);
    // OverwriteAll skips syncing uninitialized Y from host.
    using dev_view_t = decltype(Y.getLocalViewDevice(Tpetra::Access::ReadWrite));
    dev_view_t yView = overwrite ? Y.getLocalViewDevice(Tpetra::Access::OverwriteAll)
                                 : Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
    Kokkos::parallel_for("DiagonalInverseOperator::apply",
        Kokkos::RangePolicy<typename Node::execution_space, size_t>(0, nrows),
        KOKKOS_LAMBDA(const size_t i) {
          const ScalarT dinv = useConjugate ? KokkosKernels::ArithTraits<ScalarT>::conj(dView(i, 0))
                                            : dView(i, 0);
          for (size_t j = 0; j < nvec; ++j) {
            const ScalarT ax = alpha * dinv * xView(i, j);
            yView(i, j) = overwrite ? ax : (beta * yView(i, j) + ax);
          }
        });
  }

private:
  Teuchos::RCP<LA_Vector> invDiag_;
  mutable bool maps_checked_ = false;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
