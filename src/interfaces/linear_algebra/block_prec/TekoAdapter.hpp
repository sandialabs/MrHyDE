/***********************************************************************
 MrHyDE - Teko/Thyra composition for the block preconditioners.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_LINEAR_ALGEBRA_TEKO_ADAPTER_H
#define MRHYDE_LINEAR_ALGEBRA_TEKO_ADAPTER_H

#include "block_prec/BlockTypes.hpp"

#include <Teko_Utilities.hpp>
#include <Teko_InverseFactory.hpp>
#include <Teko_PreconditionerInverseFactory.hpp>
#include <Teko_JacobiPreconditionerFactory.hpp>
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

// Inverse of tpetraToThyraConst. Teko's own TpetraHelpers hard-fix the node type in
// its config, so they cannot serve MrHyDE's templated Node.
template<class Node>
inline Teuchos::RCP<const typename BlockTypes<Node>::CrsMatrix>
thyraToTpetraCrs(const Teko::LinearOp & op) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  if (op.is_null()) return Teuchos::null;
  auto tpOp = Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ScalarT,LO,GO,Node> >(op);
  TEUCHOS_TEST_FOR_EXCEPTION(tpOp.is_null(), std::runtime_error,
    "thyraToTpetraCrs: operator is not a Thyra::TpetraLinearOp over this node type.");
  auto crs = Teuchos::rcp_dynamic_cast<const LA_CrsMatrix>(tpOp->getConstTpetraOperator());
  TEUCHOS_TEST_FOR_EXCEPTION(crs.is_null(), std::runtime_error,
    "thyraToTpetraCrs: Tpetra operator is not a CrsMatrix.");
  return crs;
}

template<class Node>
inline Teuchos::RCP<Thyra::LinearOpBase<ScalarT> >
tpetraToThyra(const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & op,
              const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & rangeMap,
              const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & domainMap) {
  auto range  = Thyra::createVectorSpace<ScalarT,LO,GO,Node>(rangeMap);
  auto domain = Thyra::createVectorSpace<ScalarT,LO,GO,Node>(domainMap);
  return Thyra::createLinearOp<ScalarT,LO,GO,Node>(op, range, domain);
}

// Split-ordered NxN assembly; unset blocks are zero to Thyra.
template<class Node>
Teko::BlockedLinearOp
buildThyraBlockedFromSplits(
    const std::vector<std::vector<typename BlockTypes<Node>::CrsMatrixRCP> > & blocks) {
  const int nb = static_cast<int>(blocks.size());
  auto blo = Thyra::defaultBlockedLinearOp<ScalarT>();
  blo->beginBlockFill(nb, nb);
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < nb; ++j) {
      if (blocks[i][j].is_null()) continue;
      blo->setBlock(i, j, tpetraToThyraConst<Node>(blocks[i][j]));
    }
  }
  blo->endBlockFill();
  return Teko::toBlockedLinearOp(Teko::LinearOp(blo));
}

// Adapt a blocked Thyra preconditioner to a monolithic Tpetra operator:
//
//     X on fullMap --import--> [x0; x1] --tekoPrec--> [y0; y1] --export--> Y on fullMap
//
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
    GO blockSum = 0;
    for (size_t b = 0; b < nb; ++b) blockSum += static_cast<GO>(blockMaps[b]->getGlobalNumElements());
    TEUCHOS_TEST_FOR_EXCEPTION(blockSum != static_cast<GO>(fullMap_->getGlobalNumElements()),
      std::runtime_error,
      "TekoTpetraAdapter: block maps hold " << blockSum << " rows but the full map has "
      << fullMap_->getGlobalNumElements() << "; they must partition it.");
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

// Teko drives the factory: buildInverse -> initializePrec -> buildPreconditionerOperator.
inline Teko::LinearOp
tekoBuildInverse(const Teuchos::RCP<Teko::PreconditionerFactory> & factory,
                 Teko::BlockedLinearOp & blocked) {
  Teuchos::RCP<Teko::InverseFactory> inverse =
    Teuchos::rcp(new Teko::PreconditionerInverseFactory(factory, Teuchos::null));
  return Teko::buildInverse(*inverse, blocked);
}

template<class BuildFactoryFn>
inline Teko::LinearOp
composeTekoBlockOp(Teko::BlockedLinearOp & blocked,
                   const std::vector<Teko::LinearOp> & invs,
                   BuildFactoryFn && factoryBuild) {
  Teuchos::RCP<Teko::BlockInvDiagonalStrategy> strategy =
    Teuchos::rcp(new Teko::StaticInvDiagStrategy(invs));
  return tekoBuildInverse(factoryBuild(strategy), blocked);
}

inline Teuchos::RCP<Teko::BlockPreconditionerFactory>
makeJacobiFactory(const Teuchos::RCP<Teko::BlockInvDiagonalStrategy> & strategy) {
  return Teuchos::rcp(new Teko::JacobiPreconditionerFactory(strategy));
}

template<class Node, class BuildFactoryFn>
inline Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
finalizeTekoNativeBlockOp(const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & fullMap,
                          const std::vector<Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > > & blockMaps,
                          Teko::BlockedLinearOp & blocked,
                          const std::vector<Teko::LinearOp> & invs,
                          BuildFactoryFn && factoryBuild) {
  TEUCHOS_TEST_FOR_EXCEPTION(invs.size() != blockMaps.size(), std::runtime_error,
    "finalizeTekoNativeBlockOp: " << invs.size() << " inverses for " << blockMaps.size()
    << " block maps.");
  Teko::LinearOp tekoPrec = composeTekoBlockOp(blocked, invs, factoryBuild);
  return Teuchos::rcp(new TekoTpetraAdapter<Node>(fullMap, blockMaps, tekoPrec));
}

} // namespace detail

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildTekoNativeBlockDiagonal(const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & fullMap,
                             const std::vector<Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > > & blockMaps,
                             const std::vector<typename BlockTypes<Node>::CrsMatrixRCP> & diagBlocks,
                             const std::vector<Teko::LinearOp> & invs) {
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.size() != diagBlocks.size(), std::runtime_error,
    "buildTekoNativeBlockDiagonal: " << blockMaps.size() << " block maps for "
    << diagBlocks.size() << " diagonal blocks.");
  std::vector<std::vector<typename BlockTypes<Node>::CrsMatrixRCP> > rows(
    diagBlocks.size(),
    std::vector<typename BlockTypes<Node>::CrsMatrixRCP>(diagBlocks.size(), Teuchos::null));
  for (size_t b = 0; b < diagBlocks.size(); ++b) rows[b][b] = diagBlocks[b];
  Teko::BlockedLinearOp blocked = buildThyraBlockedFromSplits<Node>(rows);
  return detail::finalizeTekoNativeBlockOp<Node>(fullMap, blockMaps, blocked, invs,
                                                detail::makeJacobiFactory);
}

} // namespace block_prec
} // namespace MrHyDE

#endif
