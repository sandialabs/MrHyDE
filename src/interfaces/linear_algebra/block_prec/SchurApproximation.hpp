/***********************************************************************
 MrHyDE - Schur approximation builders for block preconditioners.
 Block arguments (J00, J11, J10, J01) follow the pivot-relative convention:
 J00 = pivot diagonal, J11 = target diagonal (see BlockSystem / linearAlgebraInterface_blockprec overview).
 Call flow:
- buildSchurApproximation(...) is the entry point.
  - Base variant returns S = J11 using schurBase(J11).
  - Diag variant builds full Schur:
      S = J11 + correction,
      correction = -damping * J10 * inv(diag_or_lumped(J00)) * J01.
    This path uses assembleSchurFromInputs(...), which preallocates union sparsity
    from base + correction before insertion and fillComplete.
    - Optional (Diag only): if diagTermOut is requested, correction-only output is built:
        C = -damping * J10 * inv(diag_or_lumped(J00)) * J01.
      This uses assembleDiagCorrectionOnlyFromInputs(...) (no subtraction path).
 Shared internals:
 - assembleSchurFromInputs(...) and assembleDiagCorrectionOnlyFromInputs(...)
   delegate to assembleSchurCoreFromInputs(...) with includeBase=true/false.
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_SCHUR_APPROXIMATION_HPP
#define MRHYDE_BLOCK_PREC_SCHUR_APPROXIMATION_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/BlockTypes.hpp"

namespace MrHyDE {
namespace block_prec {

// =============================================================================
// Helpers
// =============================================================================

// Inputs for generic Schur assembly.
// Generic form: S = base + scale * left * inv(weight) * right.
// Base-only case: hasCorrection = false => S = base.
template<class Node>
struct SchurAssemblyInputs {
  using matrix_rcp = typename block_prec::BlockTypes<Node>::CrsMatrixRCP;
  matrix_rcp base;
  matrix_rcp left;
  matrix_rcp weight;
  matrix_rcp right;
  ScalarT scale = Teuchos::ScalarTraits<ScalarT>::zero();
  bool useLumpedWeightDiagonal = false;
  bool hasCorrection = false;
  std::string tag = "schur";
};


// =============================================================================
// Schur builders
// =============================================================================

// S = J11
template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP schurBase(const typename block_prec::BlockTypes<Node>::CrsMatrixRCP & J11) {
  using Types = block_prec::BlockTypes<Node>;
  using LA_CrsMatrix = typename Types::CrsMatrix;
  using matrix_rcp = typename Types::CrsMatrixRCP;
  using map_rcp = typename Types::MapRCP;
  using host_inds_type = typename Types::HostInds;
  using host_vals_type = typename Types::HostVals;

  // Column map must be J11's col map so inserted column GIDs (from J11's rows) are valid.
  matrix_rcp schur = Teuchos::rcp(new LA_CrsMatrix(
    J11->getRowMap(), J11->getColMap(),
    std::max(size_t(1), static_cast<size_t>(J11->getLocalMaxNumRowEntries()))));
  block_prec::detail::forEachLocalRow<Node>(J11, [&](GO rowGid, const host_inds_type & colLids,
      const host_vals_type & colVals, size_t nent, const map_rcp & colMap) {
    std::vector<GO> gids(nent);
    std::vector<ScalarT> vals(nent);
    for (size_t k = 0; k < nent; ++k) {
      gids[k] = colMap->getGlobalElement(colLids(k));
      vals[k] = colVals(k);
    }
    schur->insertGlobalValues(rowGid, gids, vals);
  });
  schur->fillComplete(J11->getDomainMap(), J11->getRowMap());
  return schur;
}

// Build map GID -> inv(diag or lumped) for weight matrix.
template<class Node>
std::unordered_map<GO, ScalarT> buildInverseDiagonalWeight(
    const typename block_prec::BlockTypes<Node>::CrsMatrixRCP & weight,
    const bool useLumped) {
  const block_prec::detail::InverseDiagonalResult<Node> result =
    block_prec::detail::buildInverseDiagonal<Node>(
      Teuchos::rcp_implicit_cast<const typename block_prec::BlockTypes<Node>::CrsMatrix>(weight),
      useLumped);
  return result.invByRow;
}

// S = base + scale * left * inv(weight) * right with union sparsity.
// Two-pass assembly:
// 1) Build row allocation from the union of base and correction reachability.
// 2) Accumulate values in a row map and insert once per row.
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

template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP assembleSchurCoreFromInputs(
    const SchurAssemblyInputs<Node> & inputs,
    const std::unordered_map<GO, ScalarT> & invDiagWeight,
    const bool includeBase) {
  using Types = block_prec::BlockTypes<Node>;
  using LA_CrsMatrix = typename Types::CrsMatrix;
  using map_rcp = typename Types::MapRCP;
  using host_inds_type = typename Types::HostInds;
  using host_vals_type = typename Types::HostVals;
  const LO invalid = Teuchos::OrdinalTraits<LO>::invalid();
  // Pass 1: compute per-row allocation from union(base row graph, correction reachability).
  std::vector<size_t> rowAlloc(static_cast<size_t>(inputs.base->getRowMap()->getLocalNumElements()), size_t(1));
  const map_rcp leftColMap = inputs.left->getColMap();
  const LO baseLocalRows = inputs.base->getRowMap()->getLocalNumElements();
  for (LO baseRowLid = 0; baseRowLid < baseLocalRows; ++baseRowLid) {
    const GO rowGid = inputs.base->getRowMap()->getGlobalElement(baseRowLid);
    std::set<GO> rowCols;
    if (includeBase) {
      size_t nbase = inputs.base->getNumEntriesInLocalRow(baseRowLid);
      if (nbase > 0) {
        host_inds_type baseColLids("base_col_lids_alloc", std::max(size_t(1), nbase));
        host_vals_type baseVals("base_vals_alloc", std::max(size_t(1), nbase));
        inputs.base->getLocalRowCopy(baseRowLid, baseColLids, baseVals, nbase);
        for (size_t j = 0; j < nbase; ++j) {
          rowCols.insert(inputs.base->getColMap()->getGlobalElement(baseColLids(j)));
        }
      }
    }
    const LO leftRowLid = inputs.left->getRowMap()->getLocalElement(rowGid);
    if (leftRowLid != invalid) {
      size_t nleft = inputs.left->getNumEntriesInLocalRow(leftRowLid);
      if (nleft > 0) {
        host_inds_type leftColLids("left_col_lids_alloc", std::max(size_t(1), nleft));
        host_vals_type leftVals("left_vals_alloc", std::max(size_t(1), nleft));
        inputs.left->getLocalRowCopy(leftRowLid, leftColLids, leftVals, nleft);
        for (size_t k = 0; k < nleft; ++k) {
          const GO midGid = leftColMap->getGlobalElement(leftColLids(k));
          typename std::unordered_map<GO, ScalarT>::const_iterator invIt = invDiagWeight.find(midGid);
          if (invIt == invDiagWeight.end()) continue;
          const LO rightLid = inputs.right->getRowMap()->getLocalElement(midGid);
          if (rightLid == invalid) continue;
          size_t nright = inputs.right->getNumEntriesInLocalRow(rightLid);
          if (nright == 0) continue;
          host_inds_type rightColLids("right_col_lids_alloc", std::max(size_t(1), nright));
          host_vals_type rightVals("right_vals_alloc", std::max(size_t(1), nright));
          inputs.right->getLocalRowCopy(rightLid, rightColLids, rightVals, nright);
          for (size_t j = 0; j < nright; ++j) {
            rowCols.insert(inputs.right->getColMap()->getGlobalElement(rightColLids(j)));
          }
        }
      }
    }
    rowAlloc[static_cast<size_t>(baseRowLid)] = std::max(size_t(1), rowCols.size());
  }
  typename block_prec::BlockTypes<Node>::CrsMatrixRCP schur =
    Teuchos::rcp(new LA_CrsMatrix(inputs.base->getRowMap(), rowAlloc));
  // Pass 2: accumulate row values in a temporary map and insert once per row.
  for (LO rowLid = 0; rowLid < baseLocalRows; ++rowLid) {
    const GO rowGid = schur->getRowMap()->getGlobalElement(rowLid);
    std::map<GO, ScalarT> accum;

    if (includeBase) {
      size_t nbase = inputs.base->getNumEntriesInLocalRow(rowLid);
      if (nbase > 0) {
        host_inds_type baseColLids("base_col_lids", std::max(size_t(1), nbase));
        host_vals_type baseVals("base_vals", std::max(size_t(1), nbase));
        inputs.base->getLocalRowCopy(rowLid, baseColLids, baseVals, nbase);
        for (size_t j = 0; j < nbase; ++j) {
          const GO colGid = inputs.base->getColMap()->getGlobalElement(baseColLids(j));
          accum[colGid] += baseVals(j);
        }
      }
    }

    const LO leftRowLid = inputs.left->getRowMap()->getLocalElement(rowGid);
    if (leftRowLid != invalid) {
      size_t nleft = inputs.left->getNumEntriesInLocalRow(leftRowLid);
      if (nleft > 0) {
        host_inds_type leftColLids("left_col_lids", std::max(size_t(1), nleft));
        host_vals_type leftVals("left_vals", std::max(size_t(1), nleft));
        inputs.left->getLocalRowCopy(leftRowLid, leftColLids, leftVals, nleft);
        for (size_t k = 0; k < nleft; ++k) {
          const GO midGid = leftColMap->getGlobalElement(leftColLids(k));
          typename std::unordered_map<GO, ScalarT>::const_iterator invIt = invDiagWeight.find(midGid);
          if (invIt == invDiagWeight.end()) continue;
          const LO rightLid = inputs.right->getRowMap()->getLocalElement(midGid);
          if (rightLid == invalid) continue;
          size_t nright = inputs.right->getNumEntriesInLocalRow(rightLid);
          if (nright == 0) continue;
          host_inds_type rightColLids("right_col_lids", std::max(size_t(1), nright));
          host_vals_type rightVals("right_vals", std::max(size_t(1), nright));
          inputs.right->getLocalRowCopy(rightLid, rightColLids, rightVals, nright);
          const ScalarT factor = inputs.scale * leftVals(k) * invIt->second;
          for (size_t j = 0; j < nright; ++j) {
            const GO colGid = inputs.right->getColMap()->getGlobalElement(rightColLids(j));
            accum[colGid] += factor * rightVals(j);
          }
        }
      }
    }

    if (accum.empty()) continue;
    std::vector<GO> gids;
    std::vector<ScalarT> vals;
    gids.reserve(accum.size());
    vals.reserve(accum.size());
    for (typename std::map<GO, ScalarT>::const_iterator it = accum.begin(); it != accum.end(); ++it) {
      gids.push_back(it->first);
      vals.push_back(it->second);
    }
    schur->insertGlobalValues(rowGid, gids, vals);
  }
  schur->fillComplete(inputs.base->getDomainMap(), inputs.base->getRowMap());
  return schur;
}

template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP assembleSchurFromInputs(const SchurAssemblyInputs<Node> & inputs) {
  validateSchurAssemblyInputs<Node>(inputs, "Schur assembly '" + inputs.tag + "'");
  if (!inputs.hasCorrection) {
    return schurBase<Node>(inputs.base);
  }
  const std::unordered_map<GO, ScalarT> invDiagWeight = buildInverseDiagonalWeight<Node>(
      inputs.weight, inputs.useLumpedWeightDiagonal);
  return assembleSchurCoreFromInputs<Node>(inputs, invDiagWeight, true);
}

// Build correction-only matrix directly:
//   C = scale * left * inv(weight) * right
//
// Motivation:
// Correction assembly should not depend on subtracting a base matrix graph from a full Schur
// graph. Subtraction-based recovery is graph-sensitive and can report dropped entries in
// strict modes even when correction math is valid.
//
// This direct builder avoids the fragile subtraction phase entirely. We still preallocate a
// union graph for correction contributions and insert rows once before fillComplete.
template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP assembleDiagCorrectionOnlyFromInputs(
    const SchurAssemblyInputs<Node> & inputs) {
  validateSchurAssemblyInputs<Node>(inputs, "Correction-only assembly");
  const std::unordered_map<GO, ScalarT> invDiagWeight = buildInverseDiagonalWeight<Node>(
      inputs.weight, inputs.useLumpedWeightDiagonal);
  return assembleSchurCoreFromInputs<Node>(inputs, invDiagWeight, false);
}

// Entry point for Schur build. Select variant and optionally output the diagonal correction term.
template<class Node>
typename block_prec::BlockTypes<Node>::CrsMatrixRCP buildSchurApproximation(const block_prec::BlockSystem<Node> & blocks,
                                                             const LinearSolverContext<Node> & cntxt,
                                                             typename block_prec::BlockTypes<Node>::CrsMatrixRCP * diagTermOut = nullptr) {
  using matrix_rcp = typename block_prec::BlockTypes<Node>::CrsMatrixRCP;
  if (diagTermOut != nullptr) *diagTermOut = Teuchos::null;
  const SchurVariant variant = parseSchurVariant(cntxt.schur.variant, cntxt.schur.approximation_type);
  matrix_rcp schur;
  if (variant == SchurVariant::Base) {
    schur = schurBase<Node>(blocks.J11);
  }
  else if (variant == SchurVariant::Diag) {
    // Must use assembleSchurFromInputs (not a base-only graph) so that the
    // union sparsity pattern of J11 and the correction J10*diag(J00)^{-1}*J01 is allocated
    // before fillComplete. The correction product creates fill-in (edges connected through
    // shared face DOFs but not sharing an element) that lies outside J11's graph. Inserting
    // into a fill-completed J11 copy via sumIntoGlobalValues silently drops these entries.
    SchurAssemblyInputs<Node> inputs;
    inputs.base = blocks.J11;
    inputs.left = blocks.J10;
    inputs.weight = blocks.J00;
    inputs.right = blocks.J01;
    inputs.scale = -cntxt.schur.damping;
    inputs.useLumpedWeightDiagonal = cntxt.schur.diag_use_lumped_pivot_diagonal;
    inputs.hasCorrection = true;
    inputs.tag = "diag";
    schur = assembleSchurFromInputs<Node>(inputs);
    if (diagTermOut != nullptr) {
      *diagTermOut = assembleDiagCorrectionOnlyFromInputs<Node>(inputs);
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "buildSchurApproximation: unsupported Schur variant.");
  }
  return schur;
}

} // namespace block_prec
} // namespace MrHyDE

#endif
