/***********************************************************************
 MrHyDE - Schur approximation pieces for block preconditioners.
 SchurInvDiagStrategy::getInvD composes these into S_k for each split k.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
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



// beta = alpha_u^2 * (v'*corr*v) / (w'*diag(M_pivot)^-1*w), w = J01*v.
template<class Node>
ScalarT addonBeta(const BlockSystem<Node> & blocks,
                  const typename BlockTypes<Node>::CrsMatrixRCP & corr,
                  const typename BlockTypes<Node>::CrsMatrixRCP & massPivot,
                  const bool useLumpedWeightDiagonal,
                  const ScalarT alphaU,
                  const int verbosity) {
  using Types = BlockTypes<Node>;
  using LA_Vector = typename Types::Vector;
  using LA_CrsMatrix = typename Types::CrsMatrix;
  const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
  const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();

  // J10 has its Dirichlet rows zeroed, so v = J10*z vanishes there; J01 does not.
  LA_Vector z(blocks.maps[0]), v(blocks.maps[1]), cv(blocks.maps[1]);
  LA_Vector w(blocks.maps[0]), dw(blocks.maps[0]);
  detail::fillProbe<Node>(z);
  blocks.blocks[1][0]->apply(z, v);
  corr->apply(v, cv);
  blocks.blocks[0][1]->apply(v, w);

  detail::InverseDiagonalCounts wgt;
  Teuchos::RCP<LA_Vector> dB =
    detail::buildInverseDiagonal<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(massPivot), useLumpedWeightDiagonal, wgt);
  dw.elementWiseMultiply(one, *dB, w, zero);

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
  // Both off-diagonal blocks carry the DIRK spatial scaling, which cancels in r.
  const ScalarT beta = alphaU * alphaU * (num / den);

  if (verbosity >= 5 && blocks.maps[1]->getComm()->getRank() == 0) {
    std::cout << "[ADDON] beta = " << std::setprecision(14) << beta
              << std::setprecision(6) << std::endl;
  }
  return beta;
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
