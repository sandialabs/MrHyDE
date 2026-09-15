#ifndef MRHYDE_BLOCK_PREC_OPERATORS_HPP
#define MRHYDE_BLOCK_PREC_OPERATORS_HPP

#include "block_prec/BlockTypes.hpp"

#include <Amesos2.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosSolverManager.hpp>

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

template<class Node>
void combineWithAlphaBeta(const typename BlockTypes<Node>::MultiVector & opX,
                          const ScalarT alpha,
                          const ScalarT beta,
                          typename BlockTypes<Node>::MultiVector & Y) {
  Y.update(alpha, opX, beta);
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
    TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*invDiag_->getMap()) ||
                               !Y.getMap()->isSameAs(*invDiag_->getMap()),
      std::runtime_error,
      "DiagonalInverseOperator map mismatch.");
    const auto xView = X.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto yView = Y.getLocalViewHost(Tpetra::Access::ReadWrite);
    const auto dView = invDiag_->getLocalViewHost(Tpetra::Access::ReadOnly);
    const size_t nrows = static_cast<size_t>(X.getLocalLength());
    const size_t nvec = static_cast<size_t>(X.getNumVectors());
    const bool useConjugate = (mode == Teuchos::CONJ_TRANS);
    const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
    // 0 * NaN = NaN in IEEE 754; Belos hands us uninitialized Y, so guard beta==0.
    const bool overwrite = (beta == zero);
    for (size_t i = 0; i < nrows; ++i) {
      ScalarT dinv = dView(i, 0);
      if (useConjugate) {
        dinv = Teuchos::ScalarTraits<ScalarT>::conjugate(dinv);
      }
      for (size_t j = 0; j < nvec; ++j) {
        const ScalarT ax = alpha * dinv * xView(i, j);
        yView(i, j) = overwrite ? ax : (beta * yView(i, j) + ax);
      }
    }
  }

private:
  Teuchos::RCP<LA_Vector> invDiag_;
};

/** Wraps an Amesos2 direct solver as a Tpetra::Operator for block-level inversion. */
template<class Node>
class DirectSolveOperator : public Tpetra::Operator<ScalarT, LO, GO, Node> {
public:
  using Types = BlockTypes<Node>;
  using LA_Map = typename Types::Map;
  using LA_MultiVector = typename Types::MultiVector;
  using CrsMatrix = typename Types::CrsMatrix;
  using Solver = Amesos2::Solver<CrsMatrix, LA_MultiVector>;

  DirectSolveOperator(const Teuchos::RCP<Solver> & solverIn,
                      const Teuchos::RCP<const LA_Map> & mapIn)
    : solver_(solverIn), map_(mapIn) {}

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return map_; }
  Teuchos::RCP<const LA_Map> getRangeMap() const override { return map_; }
  bool hasTransposeApply() const override { return false; }

  void apply(const LA_MultiVector & X, LA_MultiVector & Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::runtime_error,
      "DirectSolveOperator does not support transpose.");
    TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*map_) || !Y.getMap()->isSameAs(*map_),
      std::runtime_error, "DirectSolveOperator map mismatch.");
    const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
    if (beta == zero) {
      solver_->setB(Teuchos::rcpFromRef(const_cast<LA_MultiVector &>(X)));
      solver_->setX(Teuchos::rcpFromRef(Y));
      solver_->solve();
      if (alpha != Teuchos::ScalarTraits<ScalarT>::one()) Y.scale(alpha);
    }
    else {
      Teuchos::RCP<LA_MultiVector> Yold = Teuchos::rcp(new LA_MultiVector(Y, Teuchos::Copy));
      Teuchos::RCP<LA_MultiVector> Z = Teuchos::rcp(new LA_MultiVector(map_, Y.getNumVectors()));
      solver_->setB(Teuchos::rcpFromRef(const_cast<LA_MultiVector &>(X)));
      solver_->setX(Z);
      solver_->solve();
      Y.update(alpha, *Z, beta, *Yold, zero);
    }
  }

private:
  Teuchos::RCP<Solver> solver_;
  Teuchos::RCP<const LA_Map> map_;
};

// Variable inner iteration counts require a flexible outer solver such as FGMRES.
template<class Node>
class KrylovWrappedBlockOperator : public Tpetra::Operator<ScalarT, LO, GO, Node> {
public:
  using Types = BlockTypes<Node>;
  using LA_Map = typename Types::Map;
  using LA_MultiVector = typename Types::MultiVector;
  using LA_Operator = typename Types::Operator;
  using LA_LinearProblem = Belos::LinearProblem<ScalarT, LA_MultiVector, LA_Operator>;
  using LA_SolverManager = Belos::SolverManager<ScalarT, LA_MultiVector, LA_Operator>;

  KrylovWrappedBlockOperator(const typename Types::CrsMatrixRCP & blockMat,
                             const Teuchos::RCP<LA_SolverManager> & solver,
                             const Teuchos::RCP<LA_LinearProblem> & problem)
    : solver_(solver), problem_(problem),
      map_(blockMat->getRowMap()) {}

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return map_; }
  Teuchos::RCP<const LA_Map> getRangeMap()  const override { return map_; }
  bool hasTransposeApply() const override { return false; }

  void apply(const LA_MultiVector & X, LA_MultiVector & Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override {
    detail::requireNoTranspose<Node>(mode, "KrylovWrappedBlockOperator");
    TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*map_) || !Y.getMap()->isSameAs(*map_),
      std::runtime_error, "KrylovWrappedBlockOperator: map mismatch.");
    Teuchos::RCP<LA_MultiVector> rhs = Teuchos::rcp(new LA_MultiVector(X, Teuchos::Copy));
    Teuchos::RCP<LA_MultiVector> sol = Teuchos::rcp(new LA_MultiVector(map_, X.getNumVectors()));
    problem_->setProblem(sol, rhs);
    solver_->reset(Belos::Problem);
    solver_->solve();
    detail::combineWithAlphaBeta<Node>(*sol, alpha, beta, Y);
  }

private:
  Teuchos::RCP<LA_SolverManager> solver_;
  Teuchos::RCP<LA_LinearProblem> problem_;
  Teuchos::RCP<const LA_Map> map_;
};

} // namespace block_prec
} // namespace MrHyDE

#endif
