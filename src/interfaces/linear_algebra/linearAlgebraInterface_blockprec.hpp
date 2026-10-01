/***********************************************************************
 MrHyDE - The two block preconditioners, assembled from block_prec/.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_LINEAR_ALGEBRA_BLOCK_PREC_H
#define MRHYDE_LINEAR_ALGEBRA_BLOCK_PREC_H

#include "linearAlgebraInterface.hpp"
#include "block_prec/BlockAssembly.hpp"
#include "block_prec/BlockTriangularFactory.hpp"
#include "block_prec/BlockTypes.hpp"
#include "block_prec/HierarchicalSplitFactory.hpp"
#include "block_prec/SplitInverse.hpp"
#include "block_prec/TekoAdapter.hpp"

#include <algorithm>
#include <iostream>
#include <set>
#include <string>

namespace MrHyDE {

// Block preconditioners for mixed systems, shown here for two splits:
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
// Blocks are indexed by split, not by variable; see block_prec::BlockSystem.
//
// Schur variants:
//   (all variants approximate the exact Schur complement above)
//   base:  S = J11
//   diag:  S = J11 - damping * J10 * diag(J00)^{-1} * J01
//   mass:  S = J11 + mass scale * M11, on the Schur target only
//
// RefMaxwell addon (off by default, MueLu's default too):
//   addon11 = M1 * D0 * (beta * M0^-1) * D0^T * M1, a Hodge-Laplacian term.
//   beta = alpha_u^2 * gamma / (alpha_t * mu), the curl-curl coefficient of S.
//   m_n = integral(N_n), the lumped nodal mass, built in the auxiliary setup.
//   gamma/(alpha_t*mu) is read off the assembled Schur correction; alpha_u
//   cancels there and is reapplied from the integrator.
//   'refmaxwell: disable addon' in the XML is the only switch.

template<class Node>
using LATypes = block_prec::BlockTypes<Node>;

// Variable names in declaration order for element block 0, the order 'variable groups'
// resolves against.
template<class Node>
std::vector<std::string> variableNamesForSet(LinearAlgebraInterface<Node> & interface,
                                             const size_t set) {
  const std::vector<std::vector<std::vector<std::string> > > & vars =
    interface.disc->physics->getVarList();
  if (set < vars.size() && !vars[set].empty()) return vars[set][0];
  return std::vector<std::string>();
}

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

  // Splits may name an Ifpack2 smoother such as RELAXATION, which is not a block
  // preconditioner type, so this is a plain name test rather than parseBlockPrecType.
  const bool useRefMaxwellOnBlock0 = (cntxt != Teuchos::null) &&
    block_prec::toUpperAsciiCopy(cntxt->splitPrecType(0)) == "REFMAXWELL";

  // One map per named split.
  const std::vector<std::string> varNames = variableNamesForSet<Node>(*this, set);
  vector<Teuchos::RCP<const LA_Map> > blockMaps = block_prec::fuseSplitMaps<Node>(
    block_prec::resolveVariableGroups(cntxt->splitVariableSpecs(), varNames),
    this->buildBlockMaps(set));
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.size() < 2, std::runtime_error,
    "Block-diagonal preconditioner needs at least two variable blocks, but set "
    << set << " has " << blockMaps.size() << ".");
  if (this->verbosity >= 10 && this->comm->getRank() == 0) {
    std::cout << "[BlockDiag] " << blockMaps.size() << " variable blocks" << std::endl;
  }

  const std::vector<std::vector<matrix_RCP> > remappedBlocks =
    block_prec::detail::extractBlocks<Node>(J, blockMaps, true);

  vector<Teko::LinearOp> blockPrecs(blockMaps.size());
  vector<matrix_RCP> diagBlocks(blockMaps.size());
  for (size_t b = 0; b < blockMaps.size(); ++b) {
    diagBlocks[b] = remappedBlocks[b][b];
    blockPrecs[b] = block_prec::detail::buildSingleBlockPreconditioner<Node>(
      diagBlocks[b], cntxt, b, useRefMaxwellOnBlock0, this->verbosity);
  }

  Teuchos::RCP<const LA_Map> fullMap = J->getRowMap();
  return block_prec::buildTekoNativeBlockDiagonal<Node>(
    fullMap, blockMaps, diagBlocks, blockPrecs);
}

// ========================================================================================
// Block triangular preconditioner
// ========================================================================================
// Extract the split blocks, hand them to Teko, and let BlockTriangularFactory build the
// split inverses when Teko asks for them.
template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
LinearAlgebraInterface<Node>::setupBlockTriangularPreconditioner(
    const matrix_RCP & J,
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
    const size_t & set) {
  Teuchos::TimeMonitor localtimer(*prectimer);

  if (!this->preconditionerNeedsRebuild(cntxt, !cntxt->prec_block.is_null())) {
    return cntxt->prec_block;
  }
  block_prec::BlockSystem<Node> blocks = block_prec::buildBlockSystemForSet<Node>(
    J, this->buildBlockMaps(set), variableNamesForSet<Node>(*this, set), cntxt, set,
    this->verbosity);
  Teko::BlockedLinearOp blocked = block_prec::buildThyraBlockedFromSplits<Node>(blocks.blocks);
  Teuchos::RCP<Teko::PreconditionerFactory> factory;
  if (cntxt->split_groups.empty()) {
    factory = Teuchos::rcp(new block_prec::BlockTriangularFactory<Node>(J, cntxt, this->verbosity));
  }
  else {
    factory = Teuchos::rcp(
      new block_prec::HierarchicalSplitFactory<Node>(J, cntxt, this->verbosity));
  }
  Teko::LinearOp prec = block_prec::detail::tekoBuildInverse(factory, blocked);
  return Teuchos::rcp(new block_prec::TekoTpetraAdapter<Node>(J->getRowMap(), blocks.maps, prec));
}

} // namespace MrHyDE

#endif
