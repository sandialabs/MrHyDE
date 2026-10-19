/***********************************************************************
 MrHyDE - Schur approximation pieces for block preconditioners.
 SchurInvDiagStrategy::getInvD composes these into S_k for each split k.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_SCHUR_APPROXIMATION_HPP
#define MRHYDE_BLOCK_PREC_SCHUR_APPROXIMATION_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/BlockTypes.hpp"

#include <TpetraExt_MatrixMatrix.hpp>
#include <iomanip>

namespace MrHyDE {
namespace block_prec {


// S = base + scale * left * inv(diag(weight)) * right, filled from the BlockSystem by
// SchurInvDiagStrategy::getInvD.
template<class Node>
struct SchurAssemblyInputs {
  using matrix_rcp = typename block_prec::BlockTypes<Node>::CrsMatrixRCP;
  matrix_rcp base;
  matrix_rcp left;
  matrix_rcp weight;
  matrix_rcp right;
  ScalarT scale = Teuchos::ScalarTraits<ScalarT>::zero();
};



// diag: beta = alpha_u^2 (v'corr v)/(w' diag(M_pivot)^-1 w), w = J01 v.
// exact-curl: corr = beta C^T M_pivot C, so beta = (v'corr v)/(u' M_pivot u), u = C v.
template<class Node>
ScalarT addonBeta(const BlockSystem<Node> & blocks,
                  const typename BlockTypes<Node>::CrsMatrixRCP & corr,
                  const typename BlockTypes<Node>::CrsMatrixRCP & massPivot,
                  const typename BlockTypes<Node>::CrsMatrixRCP & curl,
                  const bool useLumpedWeightDiagonal,
                  const bool useAbsoluteWeightDiagonal,
                  const ScalarT alphaU,
                  const int verbosity) {
  using Types = BlockTypes<Node>;
  using LA_Vector = typename Types::Vector;
  using LA_CrsMatrix = typename Types::CrsMatrix;
  const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
  const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();

  LA_Vector z(blocks.maps[0]), v(blocks.maps[1]), cv(blocks.maps[1]);
  LA_Vector w(blocks.maps[0]), dw(blocks.maps[0]);
  detail::fillProbe<Node>(z);
  blocks.blocks[1][0]->apply(z, v);
  corr->apply(v, cv);
  if (curl.is_null()) {
    blocks.blocks[0][1]->apply(v, w);
    detail::InverseDiagonalCounts wgt;
    Teuchos::RCP<LA_Vector> dB =
      detail::buildInverseDiagonal<Node>(
        Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(massPivot), useLumpedWeightDiagonal, wgt,
        useAbsoluteWeightDiagonal);
    dw.elementWiseMultiply(one, *dB, w, zero);
  }
  else {
    curl->apply(v, w);
    massPivot->apply(w, dw);
  }

  const ScalarT num = v.dot(cv);
  const ScalarT den = w.dot(dw);
  // An unassembled J gives an empty correction, so beta is undefined here.
  if (den == zero || num <= zero) {
    if (verbosity >= 5 && blocks.maps[1]->getComm()->getRank() == 0) {
      std::cout << "[ADDON] beta undefined (num=" << num << ", den=" << den
                << "); running without the addon." << std::endl;
    }
    return zero;
  }
  const ScalarT beta = (curl.is_null() ? alphaU * alphaU : one) * (num / den);

  if (verbosity >= 5 && blocks.maps[1]->getComm()->getRank() == 0) {
    std::cout << "[ADDON] beta = " << std::setprecision(14) << beta
              << std::setprecision(6) << std::endl;
  }
  return beta;
}

// Least-squares s in J01*x = s*J00*C*x over two probes, and the relative misfit.
template<class Node>
std::pair<ScalarT, typename Teuchos::ScalarTraits<ScalarT>::magnitudeType>
fitCurlScale(const BlockSystem<Node> & blocks,
             const typename BlockTypes<Node>::CrsMatrixRCP & curl) {
  using MultiVector = typename BlockTypes<Node>::MultiVector;
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  MultiVector x(blocks.maps[1], 2), Cx(curl->getRangeMap(), 2);
  MultiVector y(blocks.maps[0], 2), z(blocks.maps[0], 2);
  detail::fillProbe<Node>(x);
  blocks.blocks[0][1]->apply(x, y);
  curl->apply(x, Cx);
  blocks.blocks[0][0]->apply(Cx, z);

  Teuchos::Array<ScalarT> yz(2), zz(2), yy(2);
  y.dot(z, yz());
  z.dot(z, zz());
  y.dot(y, yy());
  const ScalarT den = zz[0] + zz[1];
  const ScalarT ny = yy[0] + yy[1];
  if (den == ScalarT(0) || ny == ScalarT(0)) {
    return std::make_pair(ScalarT(0), std::numeric_limits<MagT>::infinity());
  }
  const ScalarT s = (yz[0] + yz[1]) / den;
  y.update(-s, z, ScalarT(1));
  y.dot(y, yy());
  return std::make_pair(s, std::sqrt(static_cast<MagT>((yy[0] + yy[1]) / ny)));
}

// s in J01 = s J00 C under 'exact-curl' when the fit holds, otherwise 0 (diag fallback).
// The fit fails when dB/dt and curl E carry different weights, as with an absorbing boundary on B.
template<class Node>
ScalarT exactCurlScale(const BlockSystem<Node> & blocks,
                       const typename BlockTypes<Node>::CrsMatrixRCP & curl,
                       const int verbosity) {
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
  const std::pair<ScalarT, MagT> fit = curl.is_null()
    ? std::make_pair(zero, std::numeric_limits<MagT>::infinity())
    : fitCurlScale<Node>(blocks, curl);
  const bool holds = (fit.second <= 1.0e-8);
  if (blocks.maps[0]->getComm()->getRank() == 0 && (!holds || verbosity >= 5)) {
    if (holds) {
      std::cout << "[BlockTri] exact-curl: s = " << fit.first << " rel = " << fit.second
                << std::endl;
    }
    else {
      std::cout << "[BlockTri] exact-curl: rel = " << fit.second << " > 1e-8, using diag"
                << std::endl;
    }
  }
  return holds ? fit.first : zero;
}

template<class Node>
void validateSchurAssemblyInputs(const SchurAssemblyInputs<Node> & inputs,
                                 const std::string & label) {
  TEUCHOS_TEST_FOR_EXCEPTION(inputs.base.is_null(), std::runtime_error,
    label << " requires non-null base matrix for target maps.");
  TEUCHOS_TEST_FOR_EXCEPTION(inputs.left.is_null() || inputs.weight.is_null() || inputs.right.is_null(),
    std::runtime_error,
    label << " requires non-null left/weight/right matrices.");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputs.base->getRowMap()->isSameAs(*inputs.left->getRowMap()),
    std::runtime_error,
    label << " map mismatch: base row map must match left row map.");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputs.base->getDomainMap()->isSameAs(*inputs.right->getDomainMap()),
    std::runtime_error,
    label << " map mismatch: base domain map must match right domain map.");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputs.left->getDomainMap()->isSameAs(*inputs.right->getRowMap()),
    std::runtime_error,
    label << " map mismatch: left domain map must match right row map.");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputs.weight->getRowMap()->isSameAs(*inputs.right->getRowMap()) ||
                             !inputs.weight->getDomainMap()->isSameAs(*inputs.right->getRowMap()),
    std::runtime_error,
    label << " map mismatch: weight row/domain maps must match right row map.");
}

// C = scale * left * inv(diag(weight)) * right.
//
// Must be a distributed product: a per-rank table of inv(diag(weight)) has no
// entry for pivot DOFs owned elsewhere, so those couplings would be dropped.
template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP buildCorrectionMatrix(
    const SchurAssemblyInputs<Node> & inputs,
    const Teuchos::RCP<typename block_prec::BlockTypes<Node>::Vector> & weightInverse) {
  using Types = block_prec::BlockTypes<Node>;
  using LA_CrsMatrix = typename Types::CrsMatrix;

  // weight and right share a row map, so this row scaling needs no communication.
  typename Types::Vector scaled(weightInverse->getMap(), false);
  scaled.scale(inputs.scale, *weightInverse);
  LA_CrsMatrix scaledRight(*inputs.right, Teuchos::Copy);
  scaledRight.leftScale(scaled);

  typename Types::CrsMatrixRCP corr =
    Teuchos::rcp(new LA_CrsMatrix(inputs.left->getRowMap(), 0));
  Tpetra::MatrixMatrix::Multiply(*inputs.left, false, scaledRight, false, *corr);
  return corr;
}

// S = base + scale * corr; corr has fill-in outside base's graph, so this returns a new matrix.
template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP addCorrection(
    const typename block_prec::BlockTypes<Node>::CrsMatrixRCP & base,
    const typename block_prec::BlockTypes<Node>::CrsMatrixRCP & corr,
    const ScalarT scale = Teuchos::ScalarTraits<ScalarT>::one()) {
  const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
  return Tpetra::MatrixMatrix::add(scale, false, *corr, one, false, *base,
                                   base->getDomainMap(), base->getRowMap());
}

} // namespace block_prec
} // namespace MrHyDE

#endif
