/***********************************************************************
 MrHyDE - Block preconditioners for 2x2 block systems (block 0 / block 1).
 ************************************************************************/

#ifndef MRHYDE_LINEAR_ALGEBRA_BLOCK_PREC_H
#define MRHYDE_LINEAR_ALGEBRA_BLOCK_PREC_H

#include "linearAlgebraInterface.hpp"
#include "block_prec/ParamUtils.hpp"
#include "block_prec/BlockTypes.hpp"
#include "block_prec/BlockOperators.hpp"
#include "block_prec/BlockAssembly.hpp"
#include "block_prec/SchurApproximation.hpp"
#include "block_prec/TekoAdapter.hpp"
#include "block_prec/BlockVerify.hpp"
#include "block_prec/BlockTriangularFactory.hpp"

#include <Ifpack2_Factory.hpp>
#include <Xpetra_TripleMatrixMultiply.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>

namespace MrHyDE {

// ========================================================================================
// Mathematical overview
// ========================================================================================
//
// Block preconditioners for 2x2 mixed systems:
//
//   [ J00  J01 ] [ x0 ] = [ b0 ]
//   [ J10  J11 ] [ x1 ]   [ b1 ]
//
// Block diagonal:
//   M = diag(M0, M1), with Mb approximating Jbb^{-1}.
//
// Lower block triangular:
//   y0 = J00^{-1} b0
//   y1 = S^{-1} (b1 - J10 y0)
//
// Upper block triangular:
//   y1 = S^{-1} b1
//   y0 = J00^{-1} (b0 - J01 y1)
//
// Exact Schur complement (system indices):
//   Pivot 0: S = J11 - J10 * J00^{-1} * J01
//   Pivot 1: S = J00 - J01 * J11^{-1} * J10
//
// Block extraction convention: pivot block index (0 or 1) is set by Schur pivot block.
// After extraction, blocks are always named by role.
// In code:
//    - J00 = pivot diagonal,
//    - J11 = target diagonal,
//    - J10 = target-from-pivot,
//    - J01 = pivot-from-target.
// So when
//    - pivot is 0, J00 is system (0,0) and J11 is (1,1);
//    - when pivot is 1, J00 is (1,1) and J11 is (0,0).
//
// Schur variants:
//   (all variants approximate the exact Schur complement above)
//   base:  S = J11
//   diag:  S = J11 - gamma * J10 * diag(J00)^{-1} * J01
//
// RefMaxwell addon (off by default, MueLu's default too):
//   addon11 = M1 * D0 * M0(1/beta)^-1 * D0^T * M1, a Hodge-Laplacian term.
//   beta = alpha_u^2 * gamma / (alpha_t * mu), the curl-curl coefficient of S.
//   m_n = integral(N_n), the lumped nodal mass, built in the auxiliary setup.
//   gamma/(alpha_t*mu) is read off the assembled Schur correction; alpha_u
//   cancels there and is reapplied from the integrator.
//   'refmaxwell: disable addon' in the XML is the only switch.

template<class Node>
using LATypes = block_prec::BlockTypes<Node>;

namespace block_prec {
namespace detail {

template<class Node>
Teuchos::ParameterList mergeBlockSettings(LinearAlgebraInterface<Node> & interface,
                                          const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                                          const size_t blockIndex) {
  Teuchos::ParameterList list;
  list.set("relaxation: type", "Jacobi");
  if (cntxt != Teuchos::null && cntxt->prec_sublist.name() != "empty") {
    list.setParameters(cntxt->prec_sublist);
  }
  if (interface.settings != Teuchos::null) {
    Teuchos::ParameterList & solverList = interface.settings->sublist("Solver");
    const std::string blockKey = "Block " + std::to_string(blockIndex) + " Settings";
    if (solverList.isSublist(blockKey)) {
      list.setParameters(solverList.sublist(blockKey));
    }
  }
  return list;
}

inline void ensureRelaxationDampingDouble(Teuchos::ParameterList & list) {
  if (!list.isParameter("relaxation: damping factor")) return;
  const Teuchos::ParameterEntry & e = list.getEntry("relaxation: damping factor");
  if (e.isType<double>()) return;
  const double val = e.isType<int>() ? static_cast<double>(list.get<int>("relaxation: damping factor")) : 1.0;
  list.remove("relaxation: damping factor", false);
  list.set("relaxation: damping factor", val);
}

inline std::string resolveBlockMethod(Teuchos::ParameterList & blockList) {
  std::string method = blockList.get<std::string>("preconditioner variant", "RELAXATION");
  if (toUpperAsciiCopy(method) != "AMG" &&
      blockList.isParameter("smoother: type") &&
      toUpperAsciiCopy(blockList.get<std::string>("smoother: type")) == "CHEBYSHEV") {
    method = "Chebyshev";
  }
  const std::string methodUpper = toUpperAsciiCopy(method);
  if (methodUpper == "CHEBYSHEV" && !blockList.isParameter("chebyshev: degree") &&
      !(blockList.isSublist("smoother: params") &&
        blockList.sublist("smoother: params").isParameter("chebyshev: degree"))) {
    blockList.set("chebyshev: degree", 2);
  }
  return method;
}

inline bool isHiptmairSmoother(const std::string & type) {
  return toUpperAsciiCopy(type).find("HIPTMAIR") != std::string::npos;
}

inline bool mueluParamsWantCoordinates(const Teuchos::ParameterList & pl) {
  const auto contains = [&](const std::string & key) {
    return pl.isParameter(key) &&
           pl.get<std::string>(key).find("distance laplacian") != std::string::npos;
  };
  return contains("aggregation: drop scheme") ||
         contains("aggregation: strength-of-connection: matrix");
}

// Check top-level and per-level smoother settings.
inline bool mueluParamsWantHiptmair(const Teuchos::ParameterList & pl) {
  if (pl.isParameter("smoother: type") && isHiptmairSmoother(pl.get<std::string>("smoother: type"))) return true;
  for (Teuchos::ParameterList::ConstIterator it = pl.begin(); it != pl.end(); ++it) {
    const std::string & key = pl.name(it);
    if (key.rfind("level ", 0) != 0 || !pl.isSublist(key)) continue;
    const auto & sub = pl.sublist(key);
    if (sub.isParameter("smoother: type") && isHiptmairSmoother(sub.get<std::string>("smoother: type"))) return true;
  }
  return false;
}

// A_n = D0^T * A_edge * D0, wrapped as Xpetra for MueLu 'user data.NodeMatrix'.
template<class Node>
Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> >
buildAuxNodalMatrix(const typename LATypes<Node>::CrsMatrixRCP & A_edge,
                    const typename LATypes<Node>::CrsMatrixRCP & D0) {
  using XpetraMatrix = Xpetra::Matrix<ScalarT,LO,GO,Node>;
  Teuchos::RCP<XpetraMatrix> A_x  = wrapAsXpetraMatrix<Node>(A_edge);
  Teuchos::RCP<XpetraMatrix> D0_x = wrapAsXpetraMatrix<Node>(D0);
  Teuchos::RCP<XpetraMatrix> A_n = Xpetra::MatrixFactory<ScalarT,LO,GO,Node>::Build(D0_x->getDomainMap());
  Xpetra::TripleMatrixMultiply<ScalarT,LO,GO,Node>::MultiplyRAP(
    *D0_x, /*transposeR=*/true,
    *A_x,  /*transposeA=*/false,
    *D0_x, /*transposeP=*/false,
    *A_n,
    /*call_fillComplete=*/true, /*doOptimizeStorage=*/true);
  return A_n;
}

template<class Node>
Teko::LinearOp
buildAmgBlockOperator(const typename LATypes<Node>::CrsMatrixRCP & blockMat,
                      const Teuchos::ParameterList & blockList,
                      const Teuchos::RCP<Tpetra::MultiVector<
                          typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node> > & dofCoords,
                      const typename LATypes<Node>::CrsMatrixRCP & D0_matrix = Teuchos::null) {
  Teuchos::ParameterList mueluList;

  if (!loadMueLuXmlIfPresent(blockList, mueluList, "block-diag AMG", blockMat->getComm())) {
    mueluList = defaultMueLuParams();
    Teuchos::ParameterList filteredBlockList(blockList);
    removeMrHyDEOwnedKeys(filteredBlockList);
    // MueLu rejects the top-level relaxation parameters added by mergeBlockSettings.
    removeIfpack2OnlyKeys(filteredBlockList);
    mueluList.setParameters(filteredBlockList);
  }

  if (mueluParamsWantCoordinates(mueluList)) {
    TEUCHOS_TEST_FOR_EXCEPTION(dofCoords.is_null(), std::runtime_error,
      "MueLu params request distance-laplacian aggregation but no per-DOF coordinates were "
      "supplied for this block. Set 'hgrad basis name' and 'hcurl basis name' in the "
      "corresponding Block N Settings so setupBlockTriangularAuxiliary can build the coords.");
    TEUCHOS_TEST_FOR_EXCEPTION(!dofCoords->getMap()->isSameAs(*blockMat->getRowMap()), std::runtime_error,
      "Per-DOF coordinate MultiVector map does not match block matrix row map "
      "(coords length=" << dofCoords->getGlobalLength() << ", block rows=" << blockMat->getGlobalNumRows() << ").");
    mueluList.sublist("user data").set("Coordinates", dofCoords);
  }

  // A_n = D0^T A D0 must be rebuilt whenever blockMat changes (mass swap, Newton rebuild).
  if (mueluParamsWantHiptmair(mueluList)) {
    TEUCHOS_TEST_FOR_EXCEPTION(D0_matrix.is_null(), std::runtime_error,
      "MueLu params request HIPTMAIR smoothing but no D0 (discrete gradient) matrix was "
      "supplied for this block. Set 'hgrad basis name' and 'hcurl basis name' in the "
      "corresponding Block N Settings so setupBlockTriangularAuxiliary can build D0.");
    TEUCHOS_TEST_FOR_EXCEPTION(!D0_matrix->getRangeMap()->isSameAs(*blockMat->getRowMap()), std::runtime_error,
      "HIPTMAIR setup: D0 range map does not match block matrix row map "
      "(D0 range=" << D0_matrix->getRangeMap()->getGlobalNumElements()
      << ", block rows=" << blockMat->getGlobalNumRows() << ").");
    mueluList.sublist("user data").set("NodeMatrix", buildAuxNodalMatrix<Node>(blockMat, D0_matrix));
    mueluList.sublist("user data").set("D0", wrapAsXpetraMatrix<Node>(D0_matrix));
  }

  return block_prec::buildLibraryInverse<Node>("MueLu", mueluList, "BlockDiag AMG", blockMat);
}

template<class Node>
Teko::LinearOp
buildIfpack2BlockOperator(LinearAlgebraInterface<Node> & interface,
                          const typename LATypes<Node>::CrsMatrixRCP & blockMat,
                          const Teuchos::ParameterList & blockListIn,
                          const std::string & method,
                          const size_t blockIndex) {
  Teuchos::ParameterList blockList(blockListIn);
  if (interface.verbosity >= 15 && interface.comm->getRank() == 0) {
    std::cout << "Preconditioner parameters (block diagonal, block " << blockIndex
              << ", method " << method << "):" << std::endl;
    blockList.print(std::cout);
  }

  const std::string methodUpper = toUpperAsciiCopy(method);
  Teuchos::ParameterList ifpackList(blockList);
  removeMrHyDEOwnedKeys(ifpackList);
  if (methodUpper == "CHEBYSHEV") {
    ifpackList.remove("smoother: type", false);
    promoteSublistToTopLevel(ifpackList, "smoother: params");
  }
  else {
    ifpackList.remove("smoother: type", false);
    ifpackList.remove("smoother: params", false);
    ensureRelaxationDampingDouble(ifpackList);
  }

  Teuchos::ParameterList entry;
  entry.set("Prec Type", method);
  entry.sublist("Ifpack2 Settings").setParameters(ifpackList);
  return block_prec::buildLibraryInverse<Node>("Ifpack2", entry,
    "BlockDiag block " + std::to_string(blockIndex) + " Ifpack2", blockMat);
}

template<class Node>
Teko::LinearOp
buildSingleBlockPreconditioner(LinearAlgebraInterface<Node> & interface,
                               const typename LATypes<Node>::CrsMatrixRCP & blockMat,
                               const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                               const size_t blockIndex,
                               const bool useRefMaxwellOnBlock0) {
  const std::string label = "BlockDiag block " + std::to_string(blockIndex);
  if (blockIndex == 0 && useRefMaxwellOnBlock0) {
    return block_prec::buildBlockOperator<Node>(interface, blockMat, cntxt, cntxt->pivot_block_sublist,
      BlockPrecType::RefMaxwell, false, label,
      [] { return Teko::LinearOp(); });
  }

  Teuchos::ParameterList blockList = mergeBlockSettings<Node>(interface, cntxt, blockIndex);

  // 'use mass matrix' swaps the Jacobian block for its assembled mass (M1 HCURL / M2 HDIV).
  typename LATypes<Node>::CrsMatrixRCP preconditioner_matrix = blockMat;
  const bool useMassMatrix = blockList.isParameter("use mass matrix") &&
                             blockList.get<bool>("use mass matrix");
  if (useMassMatrix) {
    TEUCHOS_TEST_FOR_EXCEPTION(cntxt.is_null() ||
      blockIndex >= cntxt->block.mass_matrices.size() ||
      cntxt->block.mass_matrices[blockIndex].is_null(),
      std::runtime_error,
      "'use mass matrix: true' on Block " << blockIndex << " Settings but no block mass matrix "
      "was assembled. Check that block-diagonal preconditioning is active and that "
      "setupBlockTriangularAuxiliary ran (needs 'use mass matrix' on at least one Block N Settings).");
    preconditioner_matrix = cntxt->block.mass_matrices[blockIndex];
    if (interface.verbosity >= 10 && interface.comm->getRank() == 0) {
      std::cout << "[BlockDiag] Block " << blockIndex
                << ": substituting mass matrix for extracted Jacobian block" << std::endl;
    }
  }

  // Block-diagonal blocks pick AMG or an Ifpack2 smoother by name, so they all
  // take buildBlockOperator's generic branch.
  const std::string method = resolveBlockMethod(blockList);
  return block_prec::buildBlockOperator<Node>(interface, preconditioner_matrix, cntxt, blockList,
    BlockPrecType::AMG, false, label,
    [&] () -> Teko::LinearOp {
      if (toUpperAsciiCopy(method) != "AMG") {
        return buildIfpack2BlockOperator<Node>(interface, preconditioner_matrix, blockList, method, blockIndex);
      }
      if (interface.verbosity >= 15 && interface.comm->getRank() == 0) {
        std::cout << "Preconditioner parameters (block diagonal, block " << blockIndex
                  << ", method AMG):" << std::endl;
        blockList.print(std::cout);
      }
      typedef typename Teuchos::ScalarTraits<ScalarT>::coordinateType CoordScalar;
      Teuchos::RCP<Tpetra::MultiVector<CoordScalar,LO,GO,Node> > dofCoords;
      if (!cntxt.is_null() && blockIndex < cntxt->block.dof_coords.size()) {
        dofCoords = cntxt->block.dof_coords[blockIndex];
      }
      // Associate D0 with the HCURL block by matching its range map.
      typename LATypes<Node>::CrsMatrixRCP D0_for_block;
      if (!cntxt.is_null() && !cntxt->refMaxwell.D0_matrix.is_null() &&
          cntxt->refMaxwell.D0_matrix->getRangeMap()->isSameAs(*preconditioner_matrix->getRowMap())) {
        D0_for_block = cntxt->refMaxwell.D0_matrix;
      }
      return buildAmgBlockOperator<Node>(preconditioner_matrix, blockList, dofCoords, D0_for_block);
    });
}

} // namespace detail
} // namespace block_prec

// ========================================================================================
// Algorithm and operators
// ========================================================================================

// ========================================================================================
// Block extraction and maps
// ========================================================================================

// Build one Tpetra map per variable block from discretization (owned GIDs per variable).
template<class Node>
vector<Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > >
LinearAlgebraInterface<Node>::buildBlockMaps(const size_t & set) {
  using Types = LATypes<Node>;
  using LA_Map = typename Types::Map;
  if (set < block_maps_built.size() && block_maps_built[set]) {
    return block_maps_cache[set];
  }
  vector<std::set<GO> > var_gids;
  const vector<string> & blocknames = disc->block_names;
  const size_t numblocks = blocknames.size();
  if (numblocks == 0) return vector<Teuchos::RCP<const LA_Map> >();

  vector<vector<int> > voff0 = disc->getOffsets(static_cast<int>(set), 0);
  const size_t numvars = voff0.size();
  var_gids.resize(numvars);

  // Gather owned GIDs per variable across all element blocks.
  for (size_t b = 0; b < numblocks; ++b) {
    auto EIDs = disc->my_elements[b];
    vector<vector<int> > voff = disc->getOffsets(static_cast<int>(set), static_cast<int>(b));
    TEUCHOS_TEST_FOR_EXCEPTION(voff.size() != numvars, std::runtime_error,
      "buildBlockMaps: element block " << b << " has " << voff.size()
      << " variables, block 0 has " << numvars << ".");
    for (size_t e = 0; e < EIDs.extent(0); ++e) {
      size_t elemID = EIDs(e);
      vector<GO> gids = disc->getGIDs(set, b, elemID);
      for (size_t v = 0; v < voff.size() && v < var_gids.size(); ++v) {
        for (size_t k = 0; k < voff[v].size(); ++k) {
          int off = voff[v][k];
          if (off >= 0 && (size_t)off < gids.size()) {
            GO gid = gids[off];
            if (owned_map[set]->isNodeGlobalElement(gid))
              var_gids[v].insert(gid);
          }
        }
      }
    }
  }

  vector<Teuchos::RCP<const LA_Map> > blockMaps(numvars);
  for (size_t v = 0; v < numvars; ++v) {
    std::vector<GO> gid_vec(var_gids[v].begin(), var_gids[v].end());
    std::sort(gid_vec.begin(), gid_vec.end());
    blockMaps[v] = Teuchos::rcp(new LA_Map(Teuchos::OrdinalTraits<GO>::invalid(), gid_vec, 0, comm));
  }
  if (block_maps_built.size() <= set) {
    block_maps_cache.resize(set + 1);
    block_maps_built.resize(set + 1, false);
  }
  block_maps_cache[set] = blockMaps;
  block_maps_built[set] = true;
  return blockMaps;
}

// Extract diagonal block by remapping J to blockMap x blockMap.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> >
LinearAlgebraInterface<Node>::extractDiagonalBlock(
    const matrix_RCP & J,
    const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & blockMap) {
  using LA_CrsMatrix = typename LATypes<Node>::CrsMatrix;
  const Teuchos::RCP<const LA_CrsMatrix> Jconst =
    Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(J);
  return block_prec::detail::remapBlockToMaps<Node>(Jconst, blockMap, blockMap);
}

// ========================================================================================
// Block diagonal preconditioner
// ========================================================================================
// Build block-diagonal prec: one sub-preconditioner per variable block (AMG/RefMaxwell/Ifpack2 per block).
template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
LinearAlgebraInterface<Node>::buildBlockDiagonalPreconditioner(const matrix_RCP & J,
                                                               const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                                                               const size_t & set) {
  Teuchos::TimeMonitor localtimer(*prectimer);
  using Types = LATypes<Node>;
  using LA_Map = typename Types::Map;

  BlockPrecType pivotType = parseBlockPrecType((cntxt != Teuchos::null) ? cntxt->schur.pivot_block_preconditioner_type : "AMG");
  const bool useRefMaxwellOnBlock0 = (pivotType == BlockPrecType::RefMaxwell);

  // Build one local map per variable block.
  vector<Teuchos::RCP<const LA_Map> > blockMaps = this->buildBlockMaps(set);
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.size() < 2, std::runtime_error,
    "Block-diagonal preconditioner needs at least two variable blocks, but set "
    << set << " has " << blockMaps.size() << ".");
  if (this->verbosity >= 10 && this->comm->getRank() == 0) {
    std::cout << "[BlockDiag] " << blockMaps.size() << " variable blocks" << std::endl;
  }

  const std::vector<std::vector<matrix_RCP> > remappedBlocks =
    block_prec::detail::extractAndRemapBlocks<Node>(J, blockMaps, true);

  // Build one diagonal-block preconditioner per block map.
  vector<Teko::LinearOp> blockPrecs(blockMaps.size());
  vector<matrix_RCP> diagBlocks(blockMaps.size());
  for (size_t b = 0; b < blockMaps.size(); ++b) {
    diagBlocks[b] = remappedBlocks[b][b];
    blockPrecs[b] = block_prec::detail::buildSingleBlockPreconditioner<Node>(
      *this, diagBlocks[b], cntxt, b, useRefMaxwellOnBlock0);
  }

  Teuchos::RCP<const LA_Map> fullMap = J->getRowMap();
  return block_prec::buildTekoNativeBlockDiagonal<Node>(
    fullMap, blockMaps, diagBlocks, blockPrecs);
}

// ========================================================================================
// Block triangular: MueLu and RefMaxwell
// ========================================================================================
// Default MueLu parameter list for block-triangular pivot/Schur AMG.
template<class Node>
Teuchos::ParameterList
LinearAlgebraInterface<Node>::getBlockTriangularMueLuParams(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {
  Teuchos::ParameterList mueluParams;
  const bool hasNestedSchurAmg =
    (cntxt->schur_block_sublist.name() != "empty") &&
    cntxt->schur_block_sublist.isSublist("AMG Settings");
  if (hasNestedSchurAmg &&
      loadMueLuXmlIfPresent(cntxt->schur_block_sublist.sublist("AMG Settings"), mueluParams, "Schur block", comm)) {
    normalizeMueLuVerbosity(mueluParams, verbosity);
    return mueluParams;
  }
  mueluParams = defaultMueLuParams();
  if (hasNestedSchurAmg || cntxt->prec_sublist.name() != "empty") {
    Teuchos::ParameterList filteredParams = hasNestedSchurAmg
      ? Teuchos::ParameterList(cntxt->schur_block_sublist.sublist("AMG Settings"))
      : Teuchos::ParameterList(cntxt->prec_sublist);
    removeMrHyDEOwnedKeys(filteredParams);
    removeIfpack2OnlyKeys(filteredParams);
    mueluParams.setParameters(filteredParams);
  } else {
    mueluParams.sublist("smoother: params").set("chebyshev: degree", 2);
    mueluParams.sublist("smoother: params").set("chebyshev: ratio eigenvalue", 1.2);
    mueluParams.sublist("smoother: params").set("chebyshev: min eigenvalue", 0.1);
    mueluParams.sublist("smoother: params").set("chebyshev: zero starting solution", true);
  }
  normalizeMueLuVerbosity(mueluParams, verbosity);
  return mueluParams;
}

// ========================================================================================
// Block triangular: Schur and setup
// ========================================================================================
// Build Schur approximation matrix (variant from context); optional output of diagonal correction term.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> >
LinearAlgebraInterface<Node>::buildBlockTriangularSchurApproximation(
    const block_prec::BlockSystem<Node> & blocks,
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
    matrix_RCP * diagTermOut) {
  return block_prec::buildSchurApproximation<Node>(blocks, *cntxt, diagTermOut, this->verbosity);
}

// Check D0, M1, nodal_coords and map compatibility for RefMaxwell pivot block. Assumes J00 comes
// from the same discretization pipeline as the Jacobian (block extraction only; no separate assembly).
template<class Node>
void LinearAlgebraInterface<Node>::validateRefMaxwellBlockInputs(
    const matrix_RCP & J00,
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) const {
  // RefMaxwell pivot block requires D0, M1, and nodal coords on maps compatible with J00.
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.D0_matrix.is_null(), std::runtime_error,
    "RefMaxwell pivot-block setup missing D0_matrix in solver context.");
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.M1_matrix.is_null(), std::runtime_error,
    "RefMaxwell pivot-block setup missing M1_matrix in solver context.");
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.nodal_coords.is_null(), std::runtime_error,
    "RefMaxwell pivot-block setup missing nodal_coords in solver context.");
  TEUCHOS_TEST_FOR_EXCEPTION(!J00->getRowMap()->isSameAs(*cntxt->refMaxwell.D0_matrix->getRangeMap()) ||
                             !J00->getDomainMap()->isSameAs(*cntxt->refMaxwell.D0_matrix->getRangeMap()),
    std::runtime_error,
    "RefMaxwell pivot-block setup map check failed: J00 row/domain maps must match D0 range map.");
  TEUCHOS_TEST_FOR_EXCEPTION(!cntxt->refMaxwell.M1_matrix->getRowMap()->isSameAs(*J00->getRowMap()) ||
                             !cntxt->refMaxwell.M1_matrix->getDomainMap()->isSameAs(*J00->getDomainMap()),
    std::runtime_error,
    "RefMaxwell pivot-block setup map check failed: M1 row/domain maps must match J00 map.");
  TEUCHOS_TEST_FOR_EXCEPTION(
    !cntxt->refMaxwell.nodal_coords->getMap()->isSameAs(*cntxt->refMaxwell.D0_matrix->getDomainMap()),
    std::runtime_error,
    "RefMaxwell pivot-block: nodal_coords map must match D0 domain map.");
  const GO d0Range = static_cast<GO>(cntxt->refMaxwell.D0_matrix->getRangeMap()->getGlobalNumElements());
  const GO d0Domain = static_cast<GO>(cntxt->refMaxwell.D0_matrix->getDomainMap()->getGlobalNumElements());
  TEUCHOS_TEST_FOR_EXCEPTION(
    static_cast<GO>(J00->getGlobalNumRows()) != d0Range || static_cast<GO>(J00->getGlobalNumCols()) != d0Range,
    std::runtime_error,
    "RefMaxwell pivot-block: J00 size " << J00->getGlobalNumRows() << "x" << J00->getGlobalNumCols()
    << " must match D0 range size " << d0Range << " (square edge block).");
  TEUCHOS_TEST_FOR_EXCEPTION(
    static_cast<GO>(cntxt->refMaxwell.nodal_coords->getGlobalLength()) != d0Domain,
    std::runtime_error,
    "RefMaxwell pivot-block: nodal_coords length " << cntxt->refMaxwell.nodal_coords->getGlobalLength()
    << " must match D0 domain size " << d0Domain << ".");
  const size_t localDomain = cntxt->refMaxwell.D0_matrix->getDomainMap()->getLocalNumElements();
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.nodal_coords->getLocalLength() != localDomain,
    std::runtime_error,
    "RefMaxwell pivot-block: nodal_coords local length " << cntxt->refMaxwell.nodal_coords->getLocalLength()
    << " must match D0 domain local size " << localDomain << ".");
  if (verbosity >= 10 && comm->getRank() == 0) {
    std::cout << "[RefMaxwell validation] J00 rows="
              << J00->getGlobalNumRows()
              << " D0 range=" << cntxt->refMaxwell.D0_matrix->getRangeMap()->getGlobalNumElements()
              << " D0 domain=" << cntxt->refMaxwell.D0_matrix->getDomainMap()->getGlobalNumElements()
              << " coords length=" << cntxt->refMaxwell.nodal_coords->getGlobalLength()
              << std::endl;
  }
}

// Full setup: extract blocks, build Schur approx, build/reuse pivot and Schur precs, assemble triangular operator.
template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
LinearAlgebraInterface<Node>::setupBlockTriangularPreconditioner(
    const matrix_RCP & J,
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
    const size_t & set) {
  Teuchos::TimeMonitor localtimer(*prectimer);
  using Types = LATypes<Node>;
  using LA_Map = typename Types::Map;

  // --- Phase 1: Reuse short-circuit and mode validation ---

  // full keeps the operator, update keeps it while J is unchanged, none rebuilds.
  if (!cntxt->prec_block.is_null() &&
      reuseKeepsOperator(cntxt->preconditioner_reuse_type,
                                     cntxt->jacobian_rebuilt_this_step)) {
    return cntxt->prec_block;
  }

  // --- Phases 2-5: extract, Schur approximation, block inverses, assemble ---
  block_prec::BlockTriangularFactory<Node> factory(*this, J, cntxt, set);
  return factory.build();
}

} // namespace MrHyDE

#endif
