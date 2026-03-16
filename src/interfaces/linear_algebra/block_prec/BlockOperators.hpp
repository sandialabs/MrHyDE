#ifndef MRHYDE_BLOCK_PREC_OPERATORS_HPP
#define MRHYDE_BLOCK_PREC_OPERATORS_HPP

#include "block_prec/BlockTypes.hpp"

#include <Amesos2.hpp>

#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

namespace MrHyDE {
namespace block_prec {

// BlockOperators.hpp owns apply-time operators for block preconditioners.
// Includes diagonal/direct inverse wrappers plus block-diagonal and block-triangular
// composition operators and their import/export workspace helpers.
// It does not own extraction/assembly logic or parameter-list policy.
namespace detail {

template<class Node>
void requireNoTranspose(const Teuchos::ETransp mode, const std::string & opName) {
  TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::runtime_error,
                             opName + "::apply only supports NO_TRANS mode.");
}

template<class Node>
void ensureWorkspace(const typename BlockTypes<Node>::MapRCP & map, const size_t numVecs,
                     typename BlockTypes<Node>::MultiVecRCP & workspace) {
  using Types = BlockTypes<Node>;
  using LA_MultiVector = typename Types::MultiVector;
  if (workspace.is_null() || workspace->getNumVectors() != numVecs ||
      !workspace->getMap()->isSameAs(*map)) {
    workspace = Teuchos::rcp(new LA_MultiVector(map, numVecs));
  }
}

template<class Node>
void combineWithAlphaBeta(const typename BlockTypes<Node>::MultiVector & opX,
                          const ScalarT alpha,
                          const ScalarT beta,
                          typename BlockTypes<Node>::MultiVector & Y) {
  Y.update(alpha, opX, beta);
}

template<class Node>
void checkedImport(const typename BlockTypes<Node>::MultiVector & X,
                   const std::vector<typename BlockTypes<Node>::ImportRCP> & imports,
                   const size_t b,
                   typename BlockTypes<Node>::MultiVector & Xb) {
  TEUCHOS_TEST_FOR_EXCEPTION(b >= imports.size() || imports[b].is_null(), std::runtime_error,
                             "Missing import map for block index.");
  Xb.doImport(X, *imports[b], Tpetra::INSERT);
}

template<class Node>
void checkedExport(const typename BlockTypes<Node>::MultiVector & Yb,
                   const std::vector<typename BlockTypes<Node>::ExportRCP> & exports,
                   const size_t b,
                   typename BlockTypes<Node>::MultiVector & Y) {
  TEUCHOS_TEST_FOR_EXCEPTION(b >= exports.size() || exports[b].is_null(), std::runtime_error,
                             "Missing export map for block index.");
  Y.doExport(Yb, *exports[b], Tpetra::ADD);
}

template<class Node>
void validateTransfers(const std::vector<typename BlockTypes<Node>::ImportRCP> & imports,
                       const std::vector<typename BlockTypes<Node>::ExportRCP> & exports,
                       const size_t expectedBlocks) {
  TEUCHOS_TEST_FOR_EXCEPTION(imports.size() < expectedBlocks || exports.size() < expectedBlocks,
                             std::runtime_error, "Missing block import/export transfer maps.");
  for (size_t b = 0; b < expectedBlocks; ++b) {
    TEUCHOS_TEST_FOR_EXCEPTION(imports[b].is_null() || exports[b].is_null(),
                               std::runtime_error, "Null block import/export transfer map.");
  }
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

template<class Node>
struct BlockTriangularWorkspace {
  using Types = BlockTypes<Node>;
  using map_rcp = typename Types::MapRCP;
  using mv_rcp = typename Types::MultiVecRCP;

  static void ensureGroup(const map_rcp & map, const size_t numVecs,
                          const std::initializer_list<mv_rcp*> & slots) {
    for (auto * slot : slots) {
      ensureWorkspace<Node>(map, numVecs, *slot);
    }
  }

  void ensure(const std::vector<map_rcp> & blockMaps, const map_rcp & fullMap, const size_t numVecs) {
    ensureGroup(blockMaps[0], numVecs, {&x0, &y0, &r0, &upperCouplingProduct});
    ensureGroup(blockMaps[1], numVecs, {&x1, &y1, &residual, &correction, &r1, &lowerCouplingProduct});
    ensureGroup(fullMap, numVecs, {&yfull});
  }

  mv_rcp x0, x1, y0, y1;
  mv_rcp residual, correction;
  mv_rcp r0, r1, upperCouplingProduct, lowerCouplingProduct;
  mv_rcp yfull;
};

template<class Node>
struct BlockDiagonalWorkspace {
  using Types = BlockTypes<Node>;
  using map_rcp = typename Types::MapRCP;
  using mv_rcp = typename Types::MultiVecRCP;

  void ensure(const std::vector<map_rcp> & blockMaps, const size_t numVecs) {
    if (xBlocks.size() != blockMaps.size()) {
      xBlocks.resize(blockMaps.size());
      yBlocks.resize(blockMaps.size());
    }
    for (size_t b = 0; b < blockMaps.size(); ++b) {
      ensureWorkspace<Node>(blockMaps[b], numVecs, xBlocks[b]);
      ensureWorkspace<Node>(blockMaps[b], numVecs, yBlocks[b]);
    }
  }

  std::vector<mv_rcp> xBlocks;
  std::vector<mv_rcp> yBlocks;
};

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
    for (size_t i = 0; i < nrows; ++i) {
      ScalarT dinv = dView(i, 0);
      if (useConjugate) {
        dinv = Teuchos::ScalarTraits<ScalarT>::conjugate(dinv);
      }
      for (size_t j = 0; j < nvec; ++j) {
        yView(i, j) = beta * yView(i, j) + alpha * dinv * xView(i, j);
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

/** Block-diagonal preconditioner: applies diag(M_0, M_1, ...) via import/apply/export per block. */
template<class Node>
class BlockDiagonalOperator : public Tpetra::Operator<ScalarT, LO, GO, Node> {
public:
  using Types = BlockTypes<Node>;
  using LA_Map = typename Types::Map;
  using LA_MultiVector = typename Types::MultiVector;
  using LA_Operator = typename Types::Operator;
  using map_rcp = typename Types::MapRCP;
  using mv_rcp = typename Types::MultiVecRCP;
  using import_rcp = typename Types::ImportRCP;
  using export_rcp = typename Types::ExportRCP;
  using workspace_type = detail::BlockDiagonalWorkspace<Node>;

  BlockDiagonalOperator(const map_rcp & fullMap,
                        const std::vector<map_rcp> & blockMaps,
                        const std::vector<Teuchos::RCP<LA_Operator> > & blockPrecs)
    : fullMap_(fullMap), blockMaps_(blockMaps), blockPrecs_(blockPrecs) {}

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return fullMap_; }
  Teuchos::RCP<const LA_Map> getRangeMap() const override { return fullMap_; }

  void apply(const LA_MultiVector& X, LA_MultiVector& Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override {
    detail::requireNoTranspose<Node>(mode, "BlockDiagonalOperator");
    const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
    const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
    detail::ensureWorkspace<Node>(fullMap_, X.getNumVectors(), wsYfull_);
    ws_.ensure(blockMaps_, X.getNumVectors());
    detail::validateTransfers<Node>(imports_, exports_, blockMaps_.size());
    wsYfull_->putScalar(0.0);
    for (size_t b = 0; b < blockMaps_.size(); ++b) {
      if (blockPrecs_[b].is_null()) continue;
      mv_rcp X_b = ws_.xBlocks[b];
      mv_rcp Y_b = ws_.yBlocks[b];
      detail::checkedImport<Node>(X, imports_, b, *X_b);
      Y_b->putScalar(0.0);
      blockPrecs_[b]->apply(*X_b, *Y_b, Teuchos::NO_TRANS, one, zero);
      detail::checkedExport<Node>(*Y_b, exports_, b, *wsYfull_);
    }
    detail::combineWithAlphaBeta<Node>(*wsYfull_, alpha, beta, Y);
  }

  bool hasTransposeApply() const override { return false; }

  void setImportExport(const std::vector<Teuchos::RCP<Tpetra::Import<LO,GO,Node> > > & imports,
                       const std::vector<Teuchos::RCP<Tpetra::Export<LO,GO,Node> > > & exports) {
    imports_ = imports;
    exports_ = exports;
  }

private:
  map_rcp fullMap_;
  std::vector<map_rcp> blockMaps_;
  std::vector<Teuchos::RCP<LA_Operator> > blockPrecs_;
  std::vector<import_rcp> imports_;
  std::vector<export_rcp> exports_;
  mutable workspace_type ws_;
  mutable mv_rcp wsYfull_;
};

/** Block-triangular preconditioner: lower or upper triangular Schur-complement solve.
 *
 *  Lower:  y_0 = M_0^{-1} b_0,  y_1 = S^{-1}(b_1 - J_{10} y_0)
 *  Upper:  y_1 = S^{-1} b_1,    y_0 = M_0^{-1}(b_0 - J_{01} y_1)
 */
template<class Node>
class BlockTriangularOperator : public Tpetra::Operator<ScalarT, LO, GO, Node> {
public:
  using Types = BlockTypes<Node>;
  using LA_Map = typename Types::Map;
  using LA_MultiVector = typename Types::MultiVector;
  using LA_Operator = typename Types::Operator;
  using map_rcp = typename Types::MapRCP;
  using mv_rcp = typename Types::MultiVecRCP;
  using import_rcp = typename Types::ImportRCP;
  using export_rcp = typename Types::ExportRCP;
  using workspace_type = detail::BlockTriangularWorkspace<Node>;

  BlockTriangularOperator(
      const map_rcp & fullMap,
      const std::vector<map_rcp> & blockMaps,
      const Teuchos::RCP<LA_Operator> & pivotPrec,
      const Teuchos::RCP<LA_Operator> & SchurPrec,
      const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & J10,
      const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & J01,
      const bool useUpperTriangular)
    : fullMap_(fullMap), blockMaps_(blockMaps), pivotPrec_(pivotPrec), SchurPrec_(SchurPrec),
      J10_(J10), J01_(J01), useUpperTriangular_(useUpperTriangular) {
    TEUCHOS_TEST_FOR_EXCEPTION(blockMaps_.size() < 2, std::runtime_error,
      "BlockTriangularOperator requires exactly two block maps.");
    imports_.resize(2);
    exports_.resize(2);
    imports_[0] = Teuchos::rcp(new typename Types::Import(fullMap_, blockMaps_[0]));
    exports_[0] = Teuchos::rcp(new typename Types::Export(blockMaps_[0], fullMap_));
    imports_[1] = Teuchos::rcp(new typename Types::Import(fullMap_, blockMaps_[1]));
    exports_[1] = Teuchos::rcp(new typename Types::Export(blockMaps_[1], fullMap_));
  }

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return fullMap_; }
  Teuchos::RCP<const LA_Map> getRangeMap() const override { return fullMap_; }
  bool hasTransposeApply() const override { return false; }

  void apply(const LA_MultiVector& X, LA_MultiVector& Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override {
    detail::requireNoTranspose<Node>(mode, "BlockTriangularOperator");
    if (pivotPrec_.is_null() || SchurPrec_.is_null()) {
      if (beta != Teuchos::ScalarTraits<ScalarT>::zero()) Y.scale(beta);
      else Y.putScalar(0.0);
      return;
    }
    ws_.ensure(blockMaps_, fullMap_, X.getNumVectors());
    mv_rcp X0 = ws_.x0;
    mv_rcp X1 = ws_.x1;
    mv_rcp Y0 = ws_.y0;
    mv_rcp Y1 = ws_.y1;
    detail::checkedImport<Node>(X, imports_, 0, *X0);
    detail::checkedImport<Node>(X, imports_, 1, *X1);
    Y0->putScalar(0.0);
    Y1->putScalar(0.0);
    if (useUpperTriangular_) applyUpperTriangular(X0, X1, Y0, Y1);
    else applyLowerTriangular(X0, X1, Y0, Y1);
    ws_.yfull->putScalar(0.0);
    detail::checkedExport<Node>(*Y0, exports_, 0, *ws_.yfull);
    detail::checkedExport<Node>(*Y1, exports_, 1, *ws_.yfull);
    detail::combineWithAlphaBeta<Node>(*ws_.yfull, alpha, beta, Y);
  }

private:
  void applyUpperTriangular(const mv_rcp & X0, const mv_rcp & X1,
                            const mv_rcp & Y0, const mv_rcp & Y1) const {
    const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
    const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
    Y1->putScalar(0.0);
    mv_rcp residual = ws_.residual;
    residual->update(one, *X1, zero);
    mv_rcp correction = ws_.correction;
    correction->putScalar(0.0);
    SchurPrec_->apply(*residual, *correction);
    Y1->update(one, *correction, one);
    mv_rcp R0 = ws_.r0;
    R0->update(one, *X0, zero);
    if (!J01_.is_null()) {
      mv_rcp J01Y1 = ws_.upperCouplingProduct;
      J01Y1->putScalar(0.0);
      J01_->apply(*Y1, *J01Y1);
      R0->update(-one, *J01Y1, one);
    }
    pivotPrec_->apply(*R0, *Y0);
  }

  void applyLowerTriangular(const mv_rcp & X0, const mv_rcp & X1,
                            const mv_rcp & Y0, const mv_rcp & Y1) const {
    const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
    const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
    pivotPrec_->apply(*X0, *Y0);
    mv_rcp R1 = ws_.r1;
    R1->update(one, *X1, zero);
    if (!J10_.is_null()) {
      mv_rcp J10Y0 = ws_.lowerCouplingProduct;
      J10Y0->putScalar(0.0);
      J10_->apply(*Y0, *J10Y0);
      R1->update(-one, *J10Y0, one);
    }
    Y1->putScalar(0.0);
    mv_rcp residual = ws_.residual;
    residual->update(one, *R1, zero);
    mv_rcp correction = ws_.correction;
    correction->putScalar(0.0);
    SchurPrec_->apply(*residual, *correction);
    Y1->update(one, *correction, one);
  }

  map_rcp fullMap_;
  std::vector<map_rcp> blockMaps_;
  Teuchos::RCP<LA_Operator> pivotPrec_, SchurPrec_;
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > J10_, J01_;
  bool useUpperTriangular_;
  std::vector<import_rcp> imports_;
  std::vector<export_rcp> exports_;
  mutable workspace_type ws_;
};

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildBlockTriangularOperator(
    const typename BlockTypes<Node>::MapRCP & fullMap,
    const std::vector<typename BlockTypes<Node>::MapRCP> & triMaps,
    const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & pivotPrec,
    const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & SchurPrec,
    const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & J10,
    const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & J01,
    const bool useUpperTriangular) {
  return Teuchos::rcp(new BlockTriangularOperator<Node>(
    fullMap, triMaps,
    Teuchos::rcp_implicit_cast<typename BlockTypes<Node>::Operator>(pivotPrec),
    Teuchos::rcp_implicit_cast<typename BlockTypes<Node>::Operator>(SchurPrec),
    J10, J01, useUpperTriangular));
}

} // namespace block_prec
} // namespace MrHyDE

#endif
