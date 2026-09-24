#ifndef MRHYDE_BLOCK_PREC_VERIFY_HPP
#define MRHYDE_BLOCK_PREC_VERIFY_HPP

#include "block_prec/BlockAssembly.hpp"

#include <MueLu_Maxwell_Utils.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>

#include <iomanip>
#include <utility>
#include <vector>

namespace MrHyDE {
namespace block_prec {


// On interior nodes curl*D0 = 0 forces Kn_SM = s * Kn_M1, so fit s and measure
// the rest. Boundary nodes differ only by Dirichlet treatment, so they are cut.
template<class Node>
void verifyKnConsistency(
    const Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > & Kn_M1,
    const Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > & SM_wrap,
    const Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > & D0_wrap,
    const Kokkos::View<bool*, typename Node::device_type::memory_space> & BCdomainNodal,
    const Teuchos::Comm<int> & comm,
    const int verbosity) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using HostInds = typename BlockTypes<Node>::HostInds;
  using HostVals = typename BlockTypes<Node>::HostVals;

  Teuchos::ParameterList rapList;
  rapList.set("rap: fix zero diagonals", false);
  Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > Kn_SM =
    MueLu::Maxwell_Utils<ScalarT,LO,GO,Node>::PtAPWrapper(SM_wrap, D0_wrap, rapList, "Kn_from_SM");
  Teuchos::rcp_const_cast<Xpetra::CrsGraph<LO,GO,Node> >(Kn_SM->getCrsGraph())->computeGlobalConstants();

  // Hard casts: a silent skip makes the check report nothing while looking enabled.
  auto asTpetra = [](const Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > & K) {
    auto wrap = Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrixWrap<ScalarT,LO,GO,Node> >(K);
    TEUCHOS_TEST_FOR_EXCEPTION(wrap.is_null(), std::runtime_error,
      "verify Kn consistency: Kn is not an Xpetra::CrsMatrixWrap.");
    auto op = Teuchos::rcp_dynamic_cast<Xpetra::TpetraCrsMatrix<ScalarT,LO,GO,Node> >(wrap->getCrsMatrix());
    TEUCHOS_TEST_FOR_EXCEPTION(op.is_null(), std::runtime_error,
      "verify Kn consistency: Kn is not backed by an Xpetra::TpetraCrsMatrix.");
    return op->getTpetra_CrsMatrix();
  };
  Teuchos::RCP<const LA_CrsMatrix> knM1 = asTpetra(Kn_M1), knSM = asTpetra(Kn_SM);

  Teuchos::RCP<typename BlockTypes<Node>::Vector> diag =
    Teuchos::rcp(new typename BlockTypes<Node>::Vector(knM1->getRowMap(), true));
  knM1->getLocalDiagCopy(*diag);
  const double diagMax = diag->normInf();

  auto bcRow = Kokkos::create_mirror_view(BCdomainNodal);
  Kokkos::deep_copy(bcRow, BCdomainNodal);
  auto bcColDev = detail::knColumnMask<Node>(Kn_M1, BCdomainNodal);
  auto bcCol = Kokkos::create_mirror_view(bcColDev);
  Kokkos::deep_copy(bcCol, bcColDev);

  // Column maps need not match, and local order says nothing about global order.
  auto colMapM1 = knM1->getColMap();
  auto colMapSM = knSM->getColMap();
  const LO nrows = static_cast<LO>(knM1->getRowMap()->getLocalNumElements());
  const size_t maxEnt = std::max<size_t>(1, std::max(knM1->getLocalMaxNumRowEntries(),
                                                     knSM->getLocalMaxNumRowEntries()));
  HostInds cM("kn_cM", maxEnt), cS("kn_cS", maxEnt);
  HostVals vM("kn_vM", maxEnt), vS("kn_vS", maxEnt);
  std::vector<std::pair<double,double> > pairs;
  pairs.reserve(static_cast<size_t>(nrows) * maxEnt);
  for (LO r = 0; r < nrows; ++r) {
    if (bcRow(r)) continue;
    size_t nM = knM1->getNumEntriesInLocalRow(r), nS = knSM->getNumEntriesInLocalRow(r);
    if (nM == 0) continue;
    knM1->getLocalRowCopy(r, cM, vM, nM);
    if (nS > 0) knSM->getLocalRowCopy(r, cS, vS, nS);
    for (size_t k = 0; k < nM; ++k) {
      if (bcCol(cM(k))) continue;
      const LO lidSM = colMapSM->getLocalElement(colMapM1->getGlobalElement(cM(k)));
      double sVal = 0.0;
      for (size_t q = 0; q < nS; ++q) if (cS(q) == lidSM) { sVal = vS(q); break; }
      pairs.push_back(std::make_pair(static_cast<double>(vM(k)), sVal));
    }
  }

  double acc[3] = {0.0, 0.0, static_cast<double>(pairs.size())};
  for (size_t i = 0; i < pairs.size(); ++i) {
    acc[0] += pairs[i].first * pairs[i].second;
    acc[1] += pairs[i].first * pairs[i].first;
  }
  double accG[3];
  Teuchos::reduceAll<int,double>(comm, Teuchos::REDUCE_SUM, 3, acc, accG);
  const double sFit = (accG[1] > 0.0) ? (accG[0] / accG[1]) : 1.0;

  double local[2] = {0.0, diagMax};
  for (size_t i = 0; i < pairs.size(); ++i) {
    local[0] = std::max(local[0], std::abs(sFit * pairs[i].first - pairs[i].second));
  }
  double globalMax[2];
  Teuchos::reduceAll<int,double>(comm, Teuchos::REDUCE_MAX, 2, local, globalMax);
  const double scale = std::abs(sFit) * globalMax[1];
  if (comm.getRank() != 0) return;
  if (verbosity >= 6) {
    std::cout << "[Maxwell1 verify Kn] interior entries compared=" << static_cast<size_t>(accG[2])
              << " fitted Kn_SM/Kn_M1=" << sFit
              << " max|s*Kn_M1 - Kn_SM|=" << globalMax[0]
              << " scale=" << scale << std::endl;
  }
  if (scale > 0.0 && globalMax[0] > 1e-10 * scale) {
    std::cout << "[Maxwell1 verify Kn] WARN: residual " << globalMax[0]
              << " exceeds 1e-10 * " << scale << "; Kn_from_M1 is not a scalar "
              << "multiple of Kn_from_SM on the interior block." << std::endl;
  }
}

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
    detail::InverseDiagonalCounts w;
    Teuchos::RCP<LA_Vector> dinv =
      detail::buildInverseDiagonal<Node>(
        Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(blocks.J00), useLumpedWeightDiagonal, w);
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
