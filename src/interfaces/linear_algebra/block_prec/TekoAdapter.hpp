/***********************************************************************
 MrHyDE - Adapter wrapping a Teko block preconditioner as a Tpetra::Operator
 on the full (multi-block) map.
 ************************************************************************/

#ifndef MRHYDE_LINEAR_ALGEBRA_TEKO_ADAPTER_H
#define MRHYDE_LINEAR_ALGEBRA_TEKO_ADAPTER_H

#include "block_prec/BlockTypes.hpp"

#include <Teko_Utilities.hpp>
#include <Teko_JacobiPreconditionerFactory.hpp>
#include <Teko_GaussSeidelPreconditionerFactory.hpp>
#include <Teko_BlockInvDiagonalStrategy.hpp>

#include <Thyra_DefaultBlockedLinearOp.hpp>
#include <Thyra_DefaultProductMultiVector.hpp>
#include <Thyra_DefaultProductVector.hpp>
#include <Thyra_DefaultProductVectorSpace.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>

namespace MrHyDE {
namespace block_prec {

template<class Node>
inline Teuchos::RCP<const Thyra::LinearOpBase<ScalarT> >
tpetraToThyraConst(const typename BlockTypes<Node>::CrsMatrixRCP & A) {
  using MatOp = Tpetra::Operator<ScalarT,LO,GO,Node>;
  Teuchos::RCP<const MatOp> op = A;
  auto range  = Thyra::createVectorSpace<ScalarT,LO,GO,Node>(A->getRangeMap());
  auto domain = Thyra::createVectorSpace<ScalarT,LO,GO,Node>(A->getDomainMap());
  return Thyra::createConstLinearOp<ScalarT,LO,GO,Node>(op, range, domain);
}

// Teko requires non-const handles for inverse operators.
template<class Node>
inline Teuchos::RCP<Thyra::LinearOpBase<ScalarT> >
tpetraToThyra(const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & op,
              const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & rangeMap,
              const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & domainMap) {
  auto range  = Thyra::createVectorSpace<ScalarT,LO,GO,Node>(rangeMap);
  auto domain = Thyra::createVectorSpace<ScalarT,LO,GO,Node>(domainMap);
  return Thyra::createLinearOp<ScalarT,LO,GO,Node>(op, range, domain);
}

template<class Node>
inline Teko::BlockedLinearOp
buildThyraBlocked2x2(const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                     const typename BlockTypes<Node>::CrsMatrixRCP & J01,
                     const typename BlockTypes<Node>::CrsMatrixRCP & J10,
                     const typename BlockTypes<Node>::CrsMatrixRCP & J11) {
  auto A00 = tpetraToThyraConst<Node>(J00);
  auto A11 = tpetraToThyraConst<Node>(J11);
  auto A01 = J01.is_null() ? Teuchos::null : tpetraToThyraConst<Node>(J01);
  auto A10 = J10.is_null() ? Teuchos::null : tpetraToThyraConst<Node>(J10);
  Teko::LinearOp lo;
  if (A01.is_null() && A10.is_null()) {
    auto blo = Thyra::defaultBlockedLinearOp<ScalarT>();
    blo->beginBlockFill(2, 2);
    blo->setBlock(0, 0, A00);
    blo->setBlock(1, 1, A11);
    blo->endBlockFill();
    lo = blo;
  } else {
    lo = Thyra::block2x2<ScalarT>(A00, A01, A10, A11);
  }
  return Teko::toBlockedLinearOp(lo);
}

// Adapt a blocked Thyra preconditioner to a monolithic Tpetra operator.
template<class Node>
class TekoTpetraAdapter : public Tpetra::Operator<ScalarT,LO,GO,Node> {
public:
  using LA_MultiVector = Tpetra::MultiVector<ScalarT,LO,GO,Node>;
  using LA_Map = Tpetra::Map<LO,GO,Node>;
  using LA_Import = Tpetra::Import<LO,GO,Node>;
  using LA_Export = Tpetra::Export<LO,GO,Node>;
  using ThyraVecSpace = Thyra::VectorSpaceBase<ScalarT>;

  TekoTpetraAdapter(const Teuchos::RCP<const LA_Map> & fullMap,
                    const std::vector<Teuchos::RCP<const LA_Map> > & blockMaps,
                    const Teko::LinearOp & tekoPrec)
    : fullMap_(fullMap), tekoPrec_(tekoPrec) {
    const size_t nb = blockMaps.size();
    imports_.resize(nb);
    exports_.resize(nb);
    thyraSpaces_.resize(nb);
    xBlock_.resize(nb);
    yBlock_.resize(nb);
    xThyra_.resize(nb);
    yThyra_.resize(nb);
    Teuchos::Array<Teuchos::RCP<const ThyraVecSpace> > spacesArr(nb);
    for (size_t b = 0; b < nb; ++b) {
      imports_[b] = Teuchos::rcp(new LA_Import(fullMap_, blockMaps[b]));
      exports_[b] = Teuchos::rcp(new LA_Export(blockMaps[b], fullMap_));
      thyraSpaces_[b] = Thyra::createVectorSpace<ScalarT,LO,GO,Node>(blockMaps[b]);
      spacesArr[b] = thyraSpaces_[b];
    }
    productSpace_ = Thyra::productVectorSpace<ScalarT>(spacesArr());
  }

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return fullMap_; }
  Teuchos::RCP<const LA_Map> getRangeMap()  const override { return fullMap_; }
  bool hasTransposeApply() const override { return false; }

  void apply(const LA_MultiVector & X,
             LA_MultiVector & Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta  = Teuchos::ScalarTraits<ScalarT>::zero()) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::runtime_error,
      "TekoTpetraAdapter::apply: transpose not supported.");
    const size_t nb = imports_.size();
    const size_t nvec = X.getNumVectors();
    ensureWorkspace(nvec);
    for (size_t b = 0; b < nb; ++b) {
      xBlock_[b]->doImport(X, *imports_[b], Tpetra::REPLACE);
    }

    Teuchos::RCP<const Thyra::MultiVectorBase<ScalarT> > xProdConst = xProd_;
    Thyra::apply(*tekoPrec_, Thyra::NOTRANS, *xProdConst, yProd_.ptr());

    // Export directly for Y = P*X; otherwise stage the sum before scaling Y.
    const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
    const ScalarT one  = Teuchos::ScalarTraits<ScalarT>::one();
    if (beta == zero && alpha == one) {
      for (size_t b = 0; b < nb; ++b) {
        Y.doExport(*yBlock_[b], *exports_[b], Tpetra::REPLACE);
      }
    } else {
      for (size_t b = 0; b < nb; ++b) {
        yFull_->doExport(*yBlock_[b], *exports_[b], Tpetra::REPLACE);
      }
      if (beta == zero) Y.putScalar(zero);
      else if (beta != one) Y.scale(beta);
      Y.update(alpha, *yFull_, one);
    }
  }

private:
  void ensureWorkspace(const size_t nvec) const {
    const size_t nb = imports_.size();
    if (cached_nvec_ == nvec && !xProd_.is_null()) return;
    Teuchos::Array<Teuchos::RCP<Thyra::MultiVectorBase<ScalarT> > > xArr(nb), yArr(nb);
    for (size_t b = 0; b < nb; ++b) {
      const auto & blockMap = imports_[b]->getTargetMap();
      xBlock_[b] = Teuchos::rcp(new LA_MultiVector(blockMap, nvec, false));
      yBlock_[b] = Teuchos::rcp(new LA_MultiVector(blockMap, nvec, false));
      xThyra_[b] = Thyra::createMultiVector<ScalarT,LO,GO,Node>(xBlock_[b], thyraSpaces_[b]);
      yThyra_[b] = Thyra::createMultiVector<ScalarT,LO,GO,Node>(yBlock_[b], thyraSpaces_[b]);
      xArr[b] = xThyra_[b];
      yArr[b] = yThyra_[b];
    }
    xProd_ = Thyra::defaultProductMultiVector<ScalarT>(productSpace_, xArr());
    yProd_ = Thyra::defaultProductMultiVector<ScalarT>(productSpace_, yArr());
    yFull_ = Teuchos::rcp(new LA_MultiVector(fullMap_, nvec, false));
    cached_nvec_ = nvec;
  }

  Teuchos::RCP<const LA_Map> fullMap_;
  std::vector<Teuchos::RCP<LA_Import> > imports_;
  std::vector<Teuchos::RCP<LA_Export> > exports_;
  std::vector<Teuchos::RCP<const ThyraVecSpace> > thyraSpaces_;
  Teuchos::RCP<const Thyra::DefaultProductVectorSpace<ScalarT> > productSpace_;
  Teko::LinearOp tekoPrec_;

  mutable std::vector<Teuchos::RCP<LA_MultiVector> > xBlock_, yBlock_;
  mutable std::vector<Teuchos::RCP<Thyra::MultiVectorBase<ScalarT> > > xThyra_, yThyra_;
  mutable Teuchos::RCP<Thyra::DefaultProductMultiVector<ScalarT> > xProd_, yProd_;
  mutable Teuchos::RCP<LA_MultiVector> yFull_;
  mutable size_t cached_nvec_ = 0;
};

namespace detail {

template<class Node, class BuildFactoryFn>
inline Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
finalizeTekoNativeBlockOp(const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & fullMap,
                          const std::vector<Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > > & blockMaps,
                          Teko::BlockedLinearOp & blocked,
                          const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & inv0,
                          const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & inv1,
                          const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                          const typename BlockTypes<Node>::CrsMatrixRCP & J11,
                          BuildFactoryFn && factoryBuild) {
  Teko::LinearOp thyraInv0 = tpetraToThyra<Node>(inv0, J00->getRangeMap(), J00->getDomainMap());
  Teko::LinearOp thyraInv1 = tpetraToThyra<Node>(inv1, J11->getRangeMap(), J11->getDomainMap());
  Teuchos::RCP<Teko::BlockInvDiagonalStrategy> strategy =
    Teuchos::rcp(new Teko::StaticInvDiagStrategy(thyraInv0, thyraInv1));
  Teko::BlockPreconditionerState state;
  Teko::LinearOp tekoPrec = factoryBuild(strategy)->buildPreconditionerOperator(blocked, state);
  return Teuchos::rcp(new TekoTpetraAdapter<Node>(fullMap, blockMaps, tekoPrec));
}

} // namespace detail

// TODO: generalize to N-block via Teko's variadic block factories.
template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildTekoNativeBlockDiagonal(const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & fullMap,
                             const std::vector<Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > > & blockMaps,
                             const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                             const typename BlockTypes<Node>::CrsMatrixRCP & J11,
                             const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & inv0,
                             const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & inv1) {
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.size() != 2, std::runtime_error,
    "buildTekoNativeBlockDiagonal currently supports only 2x2 systems (got "
    << blockMaps.size() << " block maps).");
  Teko::BlockedLinearOp blocked =
    buildThyraBlocked2x2<Node>(J00, Teuchos::null, Teuchos::null, J11);
  return detail::finalizeTekoNativeBlockOp<Node>(
    fullMap, blockMaps, blocked, inv0, inv1, J00, J11,
    [](const Teuchos::RCP<Teko::BlockInvDiagonalStrategy> & s) {
      return Teuchos::rcp(new Teko::JacobiPreconditionerFactory(s));
    });
}

// TODO: generalize to N-block via Teko's variadic block factories.
template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildTekoNativeBlockTriangular(const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & fullMap,
                               const std::vector<Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > > & blockMaps,
                               const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                               const typename BlockTypes<Node>::CrsMatrixRCP & J01,
                               const typename BlockTypes<Node>::CrsMatrixRCP & J10,
                               const typename BlockTypes<Node>::CrsMatrixRCP & J11,
                               const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & inv0,
                               const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & inv1,
                               const bool useUpperTriangular) {
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.size() != 2, std::runtime_error,
    "buildTekoNativeBlockTriangular currently supports only 2x2 systems.");
  Teko::BlockedLinearOp blocked = buildThyraBlocked2x2<Node>(J00, J01, J10, J11);
  const Teko::TriSolveType triType = useUpperTriangular ? Teko::GS_UseUpperTriangle
                                                        : Teko::GS_UseLowerTriangle;
  return detail::finalizeTekoNativeBlockOp<Node>(
    fullMap, blockMaps, blocked, inv0, inv1, J00, J11,
    [triType](const Teuchos::RCP<Teko::BlockInvDiagonalStrategy> & s) {
      return Teuchos::rcp(new Teko::GaussSeidelPreconditionerFactory(triType, s));
    });
}

} // namespace block_prec
} // namespace MrHyDE

#endif
