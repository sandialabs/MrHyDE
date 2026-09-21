/***********************************************************************
 MrHyDE - Schur approximation builders for block preconditioners.
 Block arguments (J00, J11, J10, J01) follow the pivot-relative convention:
 J00 = pivot diagonal, J11 = target diagonal (see BlockSystem / linearAlgebraInterface_blockprec overview).
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_SCHUR_APPROXIMATION_HPP
#define MRHYDE_BLOCK_PREC_SCHUR_APPROXIMATION_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/BlockTypes.hpp"

#include <TpetraExt_MatrixMatrix.hpp>
#include <iomanip>

namespace MrHyDE {
namespace block_prec {

// =============================================================================
// Helpers
// =============================================================================

// S = base + scale * left * inv(weight) * right.
template<class Node>
struct SchurAssemblyInputs {
  using matrix_rcp = typename block_prec::BlockTypes<Node>::CrsMatrixRCP;
  matrix_rcp base;
  matrix_rcp left;
  matrix_rcp weight;
  matrix_rcp right;
  ScalarT scale = Teuchos::ScalarTraits<ScalarT>::zero();
  bool useLumpedWeightDiagonal = false;
};


// =============================================================================
// Schur builders
// =============================================================================

// S = J11
template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP schurBase(const typename block_prec::BlockTypes<Node>::CrsMatrixRCP & J11) {
  using LA_CrsMatrix = typename block_prec::BlockTypes<Node>::CrsMatrix;
  return Teuchos::rcp(new LA_CrsMatrix(*J11, Teuchos::Copy));
}

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
  LA_Vector z(blocks.pivotMap), v(blocks.targetMap), cv(blocks.targetMap);
  LA_Vector w(blocks.pivotMap), dw(blocks.pivotMap);
  detail::fillProbe<Node>(z);
  blocks.J10->apply(z, v);
  corr->apply(v, cv);
  blocks.J01->apply(v, w);

  detail::InverseDiagonalCounts wgt;
  Teuchos::RCP<LA_Vector> dB =
    detail::buildInverseDiagonal<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(massPivot), useLumpedWeightDiagonal, wgt);
  dw.elementWiseMultiply(one, *dB, w, zero);

  const ScalarT num = v.dot(cv);
  const ScalarT den = w.dot(dw);
  // An unassembled J gives an empty correction, so beta is undefined here.
  if (den == zero || num <= zero) return zero;
  // Both off-diagonal blocks carry the DIRK spatial scaling, which cancels in r.
  const ScalarT beta = alphaU * alphaU * (num / den);

  if (verbosity >= 5 && blocks.targetMap->getComm()->getRank() == 0) {
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
    const int verbosity) {
  using Types = block_prec::BlockTypes<Node>;
  using LA_CrsMatrix = typename Types::CrsMatrix;

  block_prec::detail::InverseDiagonalCounts weight;
  // weight and right share a row map, so this row scaling needs no communication.
  Teuchos::RCP<typename Types::Vector> dinv =
    block_prec::detail::buildInverseDiagonal<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(inputs.weight),
      inputs.useLumpedWeightDiagonal, weight);
  block_prec::detail::reportInverseDiagonal<Node>(
    weight, "Schur weight diag inverse", inputs.weight->getRowMap()->getComm(), verbosity);
  dinv->scale(inputs.scale);
  LA_CrsMatrix scaledRight(*inputs.right, Teuchos::Copy);
  scaledRight.leftScale(*dinv);

  typename Types::CrsMatrixRCP corr =
    Teuchos::rcp(new LA_CrsMatrix(inputs.left->getRowMap(), 0));
  Tpetra::MatrixMatrix::Multiply(*inputs.left, false, scaledRight, false, *corr);
  return corr;
}

// S = base + corr. corr has fill-in outside base's graph, so this returns a new
// matrix rather than summing into a copy of base.
template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP addCorrection(
    const typename block_prec::BlockTypes<Node>::CrsMatrixRCP & base,
    const typename block_prec::BlockTypes<Node>::CrsMatrixRCP & corr) {
  const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
  return Tpetra::MatrixMatrix::add(one, false, *corr, one, false, *base,
                                   base->getDomainMap(), base->getRowMap());
}

template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP buildSchurApproximation(const block_prec::BlockSystem<Node> & blocks,
                                                             const LinearSolverContext<Node> & cntxt,
                                                             typename block_prec::BlockTypes<Node>::CrsMatrixRCP * diagTermOut = nullptr,
                                                             const int verbosity = 0) {
  using matrix_rcp = typename block_prec::BlockTypes<Node>::CrsMatrixRCP;
  if (diagTermOut != nullptr) *diagTermOut = Teuchos::null;
  const SchurVariant variant = parseSchurVariant(cntxt.schur.approximation_type);
  if (variant == SchurVariant::Base) {
    return schurBase<Node>(blocks.J11);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(variant != SchurVariant::Diag, std::runtime_error,
    "buildSchurApproximation: unsupported Schur variant.");

  SchurAssemblyInputs<Node> inputs;
  inputs.base = blocks.J11;
  inputs.left = blocks.J10;
  inputs.weight = blocks.J00;
  inputs.right = blocks.J01;
  inputs.scale = -cntxt.schur.damping;
  inputs.useLumpedWeightDiagonal = cntxt.schur.diag_use_lumped_pivot_diagonal;
  validateSchurAssemblyInputs<Node>(inputs, "Schur assembly 'diag'");

  const matrix_rcp corr = buildCorrectionMatrix<Node>(inputs, verbosity);
  if (diagTermOut != nullptr) *diagTermOut = corr;
  return addCorrection<Node>(inputs.base, corr);
}

} // namespace block_prec
} // namespace MrHyDE

#endif
