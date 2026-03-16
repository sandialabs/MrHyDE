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

#include <Ifpack2_Factory.hpp>
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
  stripBlockDiagonalBlockList(list);
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
  if (blockList.isParameter("smoother: type") &&
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

template<class Node>
Teuchos::RCP<typename LATypes<Node>::Operator>
buildAmgBlockOperator(const typename LATypes<Node>::CrsMatrixRCP & blockMat,
                      const Teuchos::ParameterList & blockList) {
  Teuchos::ParameterList mueluList = defaultMueLuParams();
  Teuchos::ParameterList filteredBlockList(blockList);
  stripContextAndMethodKeys(filteredBlockList);
  mueluList.setParameters(filteredBlockList);
  return MueLu::CreateTpetraPreconditioner(
    Teuchos::rcp_implicit_cast<typename LATypes<Node>::Operator>(blockMat), mueluList);
}

template<class Node>
Teuchos::RCP<typename LATypes<Node>::Operator>
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
  ifpackList.remove("AMG Settings", false);
  ifpackList.remove("preconditioner variant", false);
  if (methodUpper == "CHEBYSHEV") {
    ifpackList.remove("smoother: type", false);
    promoteSublistToTopLevel(ifpackList, "smoother: params");
  }
  else {
    ifpackList.remove("smoother: type", false);
    ifpackList.remove("smoother: params", false);
    ensureRelaxationDampingDouble(ifpackList);
  }

  Teuchos::RCP<Ifpack2::Preconditioner<ScalarT,LO,GO,Node> > prec =
    Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> >(method, blockMat);
  prec->setParameters(ifpackList);
  prec->initialize();
  prec->compute();
  return Teuchos::rcp_implicit_cast<typename LATypes<Node>::Operator>(prec);
}

template<class Node>
Teuchos::RCP<typename LATypes<Node>::Operator>
buildSingleBlockPreconditioner(LinearAlgebraInterface<Node> & interface,
                               const typename LATypes<Node>::CrsMatrixRCP & blockMat,
                               const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                               const size_t blockIndex,
                               const bool useRefMaxwellOnBlock0) {
  if (blockIndex == 0 && useRefMaxwellOnBlock0) {
    interface.validateRefMaxwellBlockInputs(blockMat, cntxt);
    return interface.buildRefMaxwellPreconditioner(blockMat, cntxt, cntxt->pivot_block_sublist);
  }

  Teuchos::ParameterList blockList = mergeBlockSettings<Node>(interface, cntxt, blockIndex);
  const std::string method = resolveBlockMethod(blockList);
  if (toUpperAsciiCopy(method) == "AMG") {
    if (interface.verbosity >= 15 && interface.comm->getRank() == 0) {
      std::cout << "Preconditioner parameters (block diagonal, block " << blockIndex
                << ", method AMG):" << std::endl;
      blockList.print(std::cout);
    }
    return buildAmgBlockOperator<Node>(blockMat, blockList);
  }
  return buildIfpack2BlockOperator<Node>(interface, blockMat, blockList, method, blockIndex);
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

// Extract off-diagonal block by remapping J to rowMap x colMap.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> >
LinearAlgebraInterface<Node>::extractOffDiagonalBlock(
    const matrix_RCP & J,
    const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & rowMap,
    const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & colMap) {
  using LA_CrsMatrix = typename LATypes<Node>::CrsMatrix;
  const Teuchos::RCP<const LA_CrsMatrix> Jconst =
    Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(J);
  return block_prec::detail::remapBlockToMaps<Node>(Jconst, rowMap, colMap);
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
  using LA_Import = typename Types::Import;
  using LA_Export = typename Types::Export;

  BlockPrecType pivotType = parseBlockPrecType((cntxt != Teuchos::null) ? cntxt->schur.pivot_block_preconditioner_type : "AMG");
  const bool useRefMaxwellOnBlock0 = (pivotType == BlockPrecType::RefMaxwell);
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt != Teuchos::null && cntxt->refMaxwell.strict_refmaxwell && !useRefMaxwellOnBlock0,
    std::runtime_error,
    "Strict RefMaxwell mode requires 'Pivot block preconditioner type = RefMaxwell' when using block diagonal preconditioning.");

  // Build one local map per variable block.
  vector<Teuchos::RCP<const LA_Map> > blockMaps = this->buildBlockMaps(set);
  if (blockMaps.empty()) return Teuchos::null;
  const std::vector<std::vector<matrix_RCP> > remappedBlocks =
    block_prec::detail::extractAndRemapBlocks<Node>(J, blockMaps);
  if (blockMaps.size() == 1) {
    return block_prec::detail::buildSingleBlockPreconditioner<Node>(
      *this, remappedBlocks[0][0], cntxt, 0, useRefMaxwellOnBlock0);
  }

  // Build one diagonal-block preconditioner per block map.
  vector<Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > > blockPrecs(blockMaps.size());
  for (size_t b = 0; b < blockMaps.size(); ++b) {
    blockPrecs[b] = block_prec::detail::buildSingleBlockPreconditioner<Node>(
      *this, remappedBlocks[b][b], cntxt, b, useRefMaxwellOnBlock0);
  }

  Teuchos::RCP<const LA_Map> fullMap = J->getRowMap();
  // Compose block solves into one full-map operator via import/export.
  Teuchos::RCP<block_prec::BlockDiagonalOperator<Node> > blockOp =
    Teuchos::rcp(new block_prec::BlockDiagonalOperator<Node>(fullMap, blockMaps, blockPrecs));
  vector<Teuchos::RCP<LA_Import> > imports(blockMaps.size());
  vector<Teuchos::RCP<LA_Export> > exports(blockMaps.size());
  for (size_t b = 0; b < blockMaps.size(); ++b) {
    imports[b] = Teuchos::rcp(new LA_Import(fullMap, blockMaps[b]));
    exports[b] = Teuchos::rcp(new LA_Export(blockMaps[b], fullMap));
  }
  blockOp->setImportExport(imports, exports);
  return blockOp;
}

// ========================================================================================
// Block triangular: MueLu and RefMaxwell
// ========================================================================================
// Map integer verbosity to MueLu string (none/low/medium/high).
inline void normalizeMueLuVerbosity(Teuchos::ParameterList & mueluParams, const int verbosity) {
  if (mueluParams.isParameter("verbosity") && mueluParams.getEntry("verbosity").isType<int>()) {
    const int v = mueluParams.get<int>("verbosity");
    mueluParams.set("verbosity", std::string(v <= 0 ? "none" : v <= 1 ? "low" : v <= 2 ? "medium" : "high"));
  }
  if (verbosity >= 20) {
    mueluParams.set("verbosity", "high");
  }
}

// Default MueLu parameter list for block-triangular pivot/Schur AMG.
template<class Node>
Teuchos::ParameterList
LinearAlgebraInterface<Node>::getBlockTriangularMueLuParams(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {
  Teuchos::ParameterList mueluParams = defaultMueLuParams();
  const bool hasNestedSchurAmg =
    (cntxt->schur_block_sublist.name() != "empty") &&
    cntxt->schur_block_sublist.isSublist("AMG Settings");
  if (hasNestedSchurAmg || cntxt->prec_sublist.name() != "empty") {
    Teuchos::ParameterList filteredParams = hasNestedSchurAmg
      ? Teuchos::ParameterList(cntxt->schur_block_sublist.sublist("AMG Settings"))
      : Teuchos::ParameterList(cntxt->prec_sublist);
    stripContextAndMethodKeys(filteredParams);
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
  return block_prec::buildSchurApproximation<Node>(blocks, *cntxt, diagTermOut);
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
  TEUCHOS_TEST_FOR_EXCEPTION(J00->getGlobalNumRows() != d0Range || J00->getGlobalNumCols() != d0Range,
    std::runtime_error,
    "RefMaxwell pivot-block: J00 size " << J00->getGlobalNumRows() << "x" << J00->getGlobalNumCols()
    << " must match D0 range size " << d0Range << " (square edge block).");
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.nodal_coords->getGlobalLength() != d0Domain,
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

// Full setup: extract blocks, build Schur approx, build/reuse pivot and Schur precs, assemble teko_full triangular operator.
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
  block_prec::validateBackendSupport<Node>(cntxt);
  BlockPrecType pivotType = parseBlockPrecType(cntxt->schur.pivot_block_preconditioner_type);
  const bool strictRefMaxwell = cntxt->refMaxwell.strict_refmaxwell;
  TEUCHOS_TEST_FOR_EXCEPTION(strictRefMaxwell && pivotType != BlockPrecType::RefMaxwell, std::runtime_error,
    "Strict RefMaxwell mode requires 'Pivot block preconditioner type = RefMaxwell'.");

  // Reuse policy at operator level:
  //  - FULL: keep current operator
  //  - UPDATE: keep operator only if Jacobian has not changed
  //  - NONE: always rebuild
  std::string reuseType = cntxt->preconditioner_reuse_type;
  toUpperAscii(reuseType);
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > existing = cntxt->prec_block;
  if (!existing.is_null()) {
    if (reuseType == "FULL") return existing;
    if (reuseType == "UPDATE" && !cntxt->jacobian_rebuilt_this_step) {
      return existing;
    }
  }
  if (reuseType == "NONE") {
    existing = Teuchos::null;
  }

  // --- Phase 2: Extract block system ---
  block_prec::BlockSystem<Node> blocks = block_prec::buildBlockSystemForSet<Node>(*this, J, cntxt, set);

  // --- Phase 3: Build Schur approximation ---
  matrix_RCP SchurApprox = this->buildBlockTriangularSchurApproximation(
    blocks, cntxt, nullptr);

  // --- Phase 4: Build/reuse AMG for pivot and Schur blocks ---
  Teuchos::ParameterList schurMueLuParams = this->getBlockTriangularMueLuParams(cntxt);
  Teuchos::ParameterList pivotMueLuParams(schurMueLuParams);
  if (cntxt->pivot_block_sublist.name() != "empty" && cntxt->pivot_block_sublist.isSublist("AMG Settings")) {
    Teuchos::ParameterList filteredPivotParams(cntxt->pivot_block_sublist.sublist("AMG Settings"));
    stripContextAndMethodKeys(filteredPivotParams);
    pivotMueLuParams.setParameters(filteredPivotParams);
    if (cntxt->pivot_block_sublist.sublist("AMG Settings").isSublist("smoother: params")) {
      pivotMueLuParams.remove("smoother: params", false);
      pivotMueLuParams.sublist("smoother: params").setParameters(
        cntxt->pivot_block_sublist.sublist("AMG Settings").sublist("smoother: params"));
    }
  }

  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > pivotPrec =
    block_prec::buildPivotBlockPrec<Node>(
      *this,
      blocks.J00,
      cntxt,
      pivotMueLuParams);
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > SchurPrec =
    block_prec::buildSchurBlockPrec<Node>(
      *this,
      SchurApprox,
      cntxt,
      schurMueLuParams);

  // --- Phase 5: Assemble operator ---
  Teuchos::RCP<const LA_Map> fullMap = J->getRowMap();
  const bool useUpperTriangular = cntxt->right_preconditioner;
  if (this->verbosity >= 5 && this->comm->getRank() == 0) {
    const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
    if (cntxt->schur.damping == zero) {
      std::cout << "Schur damping is 0; diagonal correction disabled." << std::endl;
    }
  }
  vector<typename Types::MapRCP> triMaps = {blocks.pivotMap, blocks.targetMap};
  return block_prec::buildBlockTriangularOperator<Node>(
    fullMap, triMaps, pivotPrec, SchurPrec,
    blocks.J10, blocks.J01, useUpperTriangular);
}

} // namespace MrHyDE

#endif
