#ifndef MRHYDE_BLOCK_PREC_VERIFY_HPP
#define MRHYDE_BLOCK_PREC_VERIFY_HPP

#include "block_prec/BlockAssembly.hpp"
#include <iomanip>

namespace MrHyDE {
namespace block_prec {


template<class Node>
void verifyBlockSystem(const BlockSystem<Node> & blocks,
                       const typename BlockTypes<Node>::CrsMatrixRCP & J,
                       const typename BlockTypes<Node>::CrsMatrixRCP & SchurApprox,
                       const typename BlockTypes<Node>::CrsMatrixRCP & D0,
                       const typename BlockTypes<Node>::CrsMatrixRCP & M1,
                       const Teuchos::RCP<const Tpetra::MultiVector<
                         typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node> > & coords,
                       const Teuchos::RCP<typename BlockTypes<Node>::MultiVector> & lumpedMass,
                       const ScalarT damping,
                       const bool useLumpedWeightDiagonal,
                       const bool schurIsDiag,
                       const int verbosity) {
  if (verbosity < 5) return;
  using Types = BlockTypes<Node>;
  using LA_Vector = typename Types::Vector;
  using LA_CrsMatrix = typename Types::CrsMatrix;
  using Import = Tpetra::Import<LO,GO,Node>;
  using Export = Tpetra::Export<LO,GO,Node>;
  const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
  const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
  const int rank = J->getRowMap()->getComm()->getRank();

  Teuchos::RCP<const typename Types::Map> fullMap = J->getRowMap();
  LA_Vector x(fullMap), Jx(fullMap), y(fullMap);
  detail::fillProbe<Node>(x);
  J->apply(x, Jx);

  Import impP(fullMap, blocks.pivotMap), impT(fullMap, blocks.targetMap);
  Export expP(blocks.pivotMap, fullMap), expT(blocks.targetMap, fullMap);
  LA_Vector x0(blocks.pivotMap), x1(blocks.targetMap);
  LA_Vector y0(blocks.pivotMap), y1(blocks.targetMap);
  x0.doImport(x, impP, Tpetra::REPLACE);
  x1.doImport(x, impT, Tpetra::REPLACE);

  blocks.J00->apply(x0, y0);
  blocks.J01->apply(x1, y0, Teuchos::NO_TRANS, one, one);
  blocks.J11->apply(x1, y1);
  blocks.J10->apply(x0, y1, Teuchos::NO_TRANS, one, one);
  y.putScalar(zero);
  y.doExport(y0, expP, Tpetra::REPLACE);
  y.doExport(y1, expT, Tpetra::REPLACE);
  y.update(-one, Jx, one);
  const auto nJx = Jx.norm2();
  const auto ndiff = y.norm2();
  if (rank == 0) {
    std::cout << "[BLOCK-VERIFY] round-trip rel = "
              << (nJx > 0 ? ndiff / nJx : ndiff) << std::endl;
  }

  // curl(grad) = 0, so the off-diagonal coupling annihilates range(D0).
  typename Types::CrsMatrixRCP curlBlock;
  if (!D0.is_null()) {
    if (D0->getRangeMap()->isSameAs(*blocks.J10->getDomainMap()))      curlBlock = blocks.J10;
    else if (D0->getRangeMap()->isSameAs(*blocks.J01->getDomainMap())) curlBlock = blocks.J01;
  }
  if (!curlBlock.is_null()) {
    LA_Vector v(D0->getDomainMap()), D0v(D0->getRangeMap()), c(curlBlock->getRangeMap());
    detail::fillProbe<Node>(v);
    D0->apply(v, D0v);
    curlBlock->apply(D0v, c);
    // divide by |J| as well otherwise this tracks element-size spread
    const auto nD0v = D0v.norm2() * curlBlock->getFrobeniusNorm();
    const auto nc = c.norm2();
    if (rank == 0) {
      std::cout << "[BLOCK-VERIFY] J10*D0 rel = "
                << (nD0v > 0 ? nc / nD0v : nc) << std::endl;
    }
  }

  // g'*M1*g matches sum(m_n) for Panzer's D0; a +-1 D0 gives 4x that.
  if (!D0.is_null() && !M1.is_null() && !coords.is_null() && !lumpedMass.is_null() &&
      M1->getRowMap()->isSameAs(*D0->getRangeMap()) &&
      coords->getMap()->isSameAs(*D0->getDomainMap())) {
    LA_Vector xd(D0->getDomainMap()), g(D0->getRangeMap()), M1g(D0->getRangeMap());
    for (size_t d = 0; d < coords->getNumVectors(); ++d) {
      auto cv = coords->getVector(d)->getLocalViewHost(Tpetra::Access::ReadOnly);
      {
        auto xv = xd.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t i = 0; i < xv.extent(0); ++i) xv(i, 0) = static_cast<ScalarT>(cv(i, 0));
      }
      D0->apply(xd, g);
      M1->apply(g, M1g);
      const auto q = g.dot(M1g);
      const auto vol = lumpedMass->getVector(0)->norm1();
      const auto rel = (vol > 0) ? std::abs(q - vol) / vol : std::abs(q - vol);
      if (rank == 0) {
        std::cout << "[BLOCK-VERIFY] D0-scale rel = "
                  << std::setprecision(14) << rel << std::setprecision(6) << std::endl;
      }
    }
  }

  if (schurIsDiag && !SchurApprox.is_null()) {
    const detail::InverseDiagonalResult<Node> w =
      detail::buildInverseDiagonal<Node>(
        Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(blocks.J00), useLumpedWeightDiagonal);
    Teuchos::RCP<LA_Vector> dinv =
      detail::inverseDiagonalVector<Node>(blocks.J00->getRowMap(), w);
    LA_Vector t(blocks.pivotMap), Sx(blocks.targetMap), mf(blocks.targetMap);
    LA_Vector dt(blocks.pivotMap);
    blocks.J01->apply(x1, t);
    dt.elementWiseMultiply(one, *dinv, t, zero);
    blocks.J10->apply(dt, mf);
    blocks.J11->apply(x1, Sx);
    mf.update(one, Sx, -damping);
    SchurApprox->apply(x1, Sx);
    Sx.update(-one, mf, one);
    const auto nmf = mf.norm2();
    const auto nSx = Sx.norm2();
    if (rank == 0) {
      std::cout << "[BLOCK-VERIFY] schur rel = "
                << (nmf > 0 ? nSx / nmf : nSx) << std::endl;
    }
  }
}

} // namespace block_prec
} // namespace MrHyDE

#endif
