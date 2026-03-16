#ifndef MRHYDE_BLOCK_PREC_ASSEMBLY_HPP
#define MRHYDE_BLOCK_PREC_ASSEMBLY_HPP

#include "block_prec/BlockOperators.hpp"
#include "block_prec/ParamUtils.hpp"
#include "linearAlgebraInterface.hpp"
#include "linearSolverContext.hpp"

#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Teko_BlockedTpetraOperator.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace MrHyDE {
namespace block_prec {

// BlockAssembly.hpp owns Jacobian block extraction/remap and preconditioner assembly helpers.
// It builds the pivot/target BlockSystem, map-safe remapped Teko blocks, and local
// block operators consumed by the block-preconditioner orchestration layer.
// It also owns the shared diagonal/lumped inverse utility used by Schur and pivot paths.
// It does not own apply-time operator classes (BlockOperators) or Schur policy parsing.

template<class Node>
struct BlockSystem {
  using Types = BlockTypes<Node>;
  using matrix_rcp = typename Types::CrsMatrixRCP;
  using map_rcp = typename Types::MapRCP;

  map_rcp pivotMap;    // Owned map for the pivot block (block index = pivotBlock)
  map_rcp targetMap;   // Owned map for the target (Schur complement) block.
  matrix_rcp J00;      // Pivot diagonal block.
  matrix_rcp J11;      // Target diagonal block.
  matrix_rcp J10;      // Target-from-pivot off-diagonal.
  matrix_rcp J01;      // Pivot-from-target off-diagonal.
  size_t targetBlock = 0;  // Block index for the target (Schur) block.
  int pivotBlock = 0;     // Block index for the pivot block.
};

namespace detail {

template<class Node>
using MapRCP = typename BlockTypes<Node>::MapRCP;

template<class Node>
using MatrixRCP = typename BlockTypes<Node>::CrsMatrixRCP;

template<class Node>
using ConstMatrixRCP = Teuchos::RCP<const typename BlockTypes<Node>::CrsMatrix>;

template<class Node>
void validateTekoTypeCompatibility() {
  static_assert(std::is_same<ScalarT, Teko::ST>::value,
                "Teko extraction requires ScalarT to match Teko::ST.");
  static_assert(std::is_same<LO, Teko::LO>::value,
                "Teko extraction requires LO to match Teko::LO.");
  static_assert(std::is_same<GO, Teko::GO>::value,
                "Teko extraction requires GO to match Teko::GO.");
  static_assert(std::is_same<Node, Teko::NT>::value,
                "Teko extraction requires Node to match Teko::NT.");
}

template<class Node>
std::vector<std::vector<GO> >
buildBlockGidListsFromMaps(const std::vector<MapRCP<Node> > & blockMaps) {
  std::vector<std::vector<GO> > gids(blockMaps.size());
  for (size_t b = 0; b < blockMaps.size(); ++b) {
    TEUCHOS_TEST_FOR_EXCEPTION(blockMaps[b].is_null(), std::runtime_error,
      "buildBlockGidListsFromMaps: null block map at index " << b << ".");
    const size_t nLocal = blockMaps[b]->getLocalNumElements();
    gids[b].reserve(nLocal);
    for (size_t lid = 0; lid < nLocal; ++lid) {
      gids[b].push_back(blockMaps[b]->getGlobalElement(Teuchos::as<LO>(lid)));
    }
  }
  return gids;
}

template<class Node>
std::vector<std::vector<ConstMatrixRCP<Node> > >
extractRawTekoBlocks(const MatrixRCP<Node> & J,
                     const std::vector<MapRCP<Node> > & blockMaps) {
  validateTekoTypeCompatibility<Node>();
  using Types = BlockTypes<Node>;
  using Operator = typename Types::Operator;
  using CrsMatrix = typename Types::CrsMatrix;

  TEUCHOS_TEST_FOR_EXCEPTION(J.is_null(), std::runtime_error,
    "extractRawTekoBlocks: Jacobian is null.");
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.empty(), std::runtime_error,
    "extractRawTekoBlocks: block map list is empty.");

  const std::vector<std::vector<GO> > blockGids = buildBlockGidListsFromMaps<Node>(blockMaps);
  const Teuchos::RCP<const Operator> op = Teuchos::rcp_implicit_cast<const Operator>(J);
  Teko::TpetraHelpers::BlockedTpetraOperator blockedOp(blockGids, op, "MrHyDE_TekoBlockExtraction");

  const size_t nBlocks = blockMaps.size();
  std::vector<std::vector<ConstMatrixRCP<Node> > > rawBlocks(
    nBlocks, std::vector<ConstMatrixRCP<Node> >(nBlocks, Teuchos::null));

  for (size_t i = 0; i < nBlocks; ++i) {
    for (size_t j = 0; j < nBlocks; ++j) {
      const Teuchos::RCP<const Operator> blockOp = blockedOp.GetBlock(Teuchos::as<int>(i), Teuchos::as<int>(j));
      TEUCHOS_TEST_FOR_EXCEPTION(blockOp.is_null(), std::runtime_error,
        "extractRawTekoBlocks: Teko returned null block (" << i << "," << j << ").");
      const Teuchos::RCP<const CrsMatrix> blockMat = Teuchos::rcp_dynamic_cast<const CrsMatrix>(blockOp);
      TEUCHOS_TEST_FOR_EXCEPTION(blockMat.is_null(), std::runtime_error,
        "extractRawTekoBlocks: block (" << i << "," << j << ") is not a Tpetra::CrsMatrix.");
      rawBlocks[i][j] = blockMat;
    }
  }
  return rawBlocks;
}

template<class Node>
MatrixRCP<Node>
remapBlockToMaps(const ConstMatrixRCP<Node> & src,
                 const MapRCP<Node> & rowMap,
                 const MapRCP<Node> & domainMap) {
  using Types = BlockTypes<Node>;
  using CrsMatrix = typename Types::CrsMatrix;
  using HostInds = typename Types::HostInds;
  using HostVals = typename Types::HostVals;
  using IntVector = typename Types::IntVector;
  using Import = typename Types::Import;

  TEUCHOS_TEST_FOR_EXCEPTION(src.is_null(), std::runtime_error, "remapBlockToMaps: source block is null.");
  TEUCHOS_TEST_FOR_EXCEPTION(rowMap.is_null() || domainMap.is_null(), std::runtime_error,
    "remapBlockToMaps: target row/domain map is null.");

  const size_t maxEnt = std::max(size_t(1), src->getLocalMaxNumRowEntries());
  HostInds colLids("teko_remap_col_lids", maxEnt);
  HostVals colVals("teko_remap_col_vals", maxEnt);
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > srcRowMap = src->getRowMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > srcColMap = src->getColMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > srcDomainMap = src->getDomainMap();

  const bool equalRowSizes =
    srcRowMap->getGlobalNumElements() == rowMap->getGlobalNumElements() &&
    srcRowMap->getLocalNumElements() == rowMap->getLocalNumElements();
  const bool equalDomainSizes =
    srcDomainMap->getGlobalNumElements() == domainMap->getGlobalNumElements() &&
    srcDomainMap->getLocalNumElements() == domainMap->getLocalNumElements();
  const bool useLidRemap = equalRowSizes && equalDomainSizes;

  const MatrixRCP<Node> dst = Teuchos::rcp(new CrsMatrix(rowMap, maxEnt));

  if (!useLidRemap) {
    // General path for mismatched row/domain partitions: filter source columns by target
    // domain ownership and keep original GIDs that are valid in the destination maps.
    Teuchos::RCP<IntVector> blockMarker = Teuchos::rcp(new IntVector(domainMap));
    blockMarker->putScalar(1);
    Import colImport(domainMap, srcColMap);
    Teuchos::RCP<IntVector> colMarker = Teuchos::rcp(new IntVector(srcColMap));
    colMarker->putScalar(0);
    colMarker->doImport(*blockMarker, colImport, Tpetra::INSERT);
    auto markerData = colMarker->getData(0);

    const LO nRows = rowMap->getLocalNumElements();
    for (LO rowLid = 0; rowLid < nRows; ++rowLid) {
      const GO rowGid = rowMap->getGlobalElement(rowLid);
      const LO srcRowLid = srcRowMap->getLocalElement(rowGid);
      if (srcRowLid == Teuchos::OrdinalTraits<LO>::invalid()) continue;

      size_t nent = src->getNumEntriesInLocalRow(srcRowLid);
      if (nent == 0) continue;
      src->getLocalRowCopy(srcRowLid, colLids, colVals, nent);

      std::vector<GO> keepCols;
      std::vector<ScalarT> keepVals;
      keepCols.reserve(nent);
      keepVals.reserve(nent);
      for (size_t k = 0; k < nent; ++k) {
        if (markerData[colLids(k)] == 0) continue;
        const GO colGid = srcColMap->getGlobalElement(colLids(k));
        keepCols.push_back(colGid);
        keepVals.push_back(colVals(k));
      }
      if (!keepCols.empty()) {
        dst->insertGlobalValues(rowGid, keepCols, keepVals);
      }
    }
  }
  else {
    using GoVector = Tpetra::Vector<GO,LO,GO,Node>;

    // Fast path for equal local row/domain sizes: remap source column LIDs onto target
    // domain GIDs via a temporary GO vector imported from source domain to source columns.
    Teuchos::RCP<GoVector> tgtDomainOnSrcDomain = Teuchos::rcp(new GoVector(srcDomainMap));
    auto tgtDomainOnSrcDomainData = tgtDomainOnSrcDomain->getLocalViewHost(Tpetra::Access::ReadWrite);
    for (LO lid = 0; lid < srcDomainMap->getLocalNumElements(); ++lid) {
      tgtDomainOnSrcDomainData(lid, 0) = domainMap->getGlobalElement(lid);
    }

    Import srcDomainToColImport(srcDomainMap, srcColMap);
    Teuchos::RCP<GoVector> tgtDomainOnSrcCol = Teuchos::rcp(new GoVector(srcColMap));
    tgtDomainOnSrcCol->putScalar(Teuchos::OrdinalTraits<GO>::invalid());
    tgtDomainOnSrcCol->doImport(*tgtDomainOnSrcDomain, srcDomainToColImport, Tpetra::INSERT);
    auto tgtDomainOnSrcColData = tgtDomainOnSrcCol->getData(0);

    Teuchos::RCP<IntVector> srcDomainMarker = Teuchos::rcp(new IntVector(srcDomainMap));
    srcDomainMarker->putScalar(1);
    Teuchos::RCP<IntVector> srcColMarker = Teuchos::rcp(new IntVector(srcColMap));
    srcColMarker->putScalar(0);
    srcColMarker->doImport(*srcDomainMarker, srcDomainToColImport, Tpetra::INSERT);
    auto srcColMarkerData = srcColMarker->getData(0);

    const LO nRows = srcRowMap->getLocalNumElements();
    for (LO srcRowLid = 0; srcRowLid < nRows; ++srcRowLid) {
      const GO rowGid = rowMap->getGlobalElement(srcRowLid);
      size_t nent = src->getNumEntriesInLocalRow(srcRowLid);
      if (nent == 0) continue;
      src->getLocalRowCopy(srcRowLid, colLids, colVals, nent);

      std::vector<GO> keepCols;
      std::vector<ScalarT> keepVals;
      keepCols.reserve(nent);
      keepVals.reserve(nent);
      for (size_t k = 0; k < nent; ++k) {
        const LO colLid = colLids(k);
        if (srcColMarkerData[colLid] == 0) continue;
        const GO mappedColGid = tgtDomainOnSrcColData[colLid];
        if (mappedColGid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
        keepCols.push_back(mappedColGid);
        keepVals.push_back(colVals(k));
      }
      if (!keepCols.empty()) {
        dst->insertGlobalValues(rowGid, keepCols, keepVals);
      }
    }
  }

  dst->fillComplete(domainMap, rowMap);
  return dst;
}

template<class Node>
std::vector<std::vector<MatrixRCP<Node> > >
extractAndRemapBlocks(const MatrixRCP<Node> & J,
                      const std::vector<MapRCP<Node> > & blockMaps) {
  const std::vector<std::vector<ConstMatrixRCP<Node> > > rawBlocks =
    extractRawTekoBlocks<Node>(J, blockMaps);
  const size_t nBlocks = blockMaps.size();
  std::vector<std::vector<MatrixRCP<Node> > > remapped(
    nBlocks, std::vector<MatrixRCP<Node> >(nBlocks, Teuchos::null));

  for (size_t i = 0; i < nBlocks; ++i) {
    for (size_t j = 0; j < nBlocks; ++j) {
      remapped[i][j] = remapBlockToMaps<Node>(rawBlocks[i][j], blockMaps[i], blockMaps[j]);
      TEUCHOS_TEST_FOR_EXCEPTION(
        !remapped[i][j]->getRowMap()->isSameAs(*blockMaps[i]) ||
        !remapped[i][j]->getDomainMap()->isSameAs(*blockMaps[j]),
        std::runtime_error,
        "extractAndRemapBlocks: map contract check failed for block (" << i << "," << j << ").");
    }
  }
  return remapped;
}

template<class Node>
struct InverseDiagonalResult {
  std::unordered_map<GO, ScalarT> invByRow;
  GO missing = 0;
  GO usedLumped = 0;
  GO usedDiag = 0;
};

template<class Node>
InverseDiagonalResult<Node>
buildInverseDiagonal(const ConstMatrixRCP<Node> & mat,
                     const bool useLumpedDiagonal) {
  using Types = BlockTypes<Node>;
  using map_rcp = typename Types::MapRCP;
  using host_inds_type = typename Types::HostInds;
  using host_vals_type = typename Types::HostVals;

  const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
  const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
  const auto zeroMag = Teuchos::ScalarTraits<ScalarT>::magnitude(zero);
  InverseDiagonalResult<Node> result;
  forEachLocalRow<Node>(mat, [useLumpedDiagonal, &result, one, zero, zeroMag](GO rowGid, const host_inds_type & colLids,
      const host_vals_type & colVals, size_t numEntries, const map_rcp & colMap) {
    ScalarT d = zero;
    ScalarT lumped = zero;
    bool foundDiag = false;
    for (size_t k = 0; k < numEntries; ++k) {
      lumped += colVals(k);
      if (colMap->getGlobalElement(colLids(k)) == rowGid) {
        d = colVals(k);
        foundDiag = true;
      }
    }
    const bool haveDiag = foundDiag && Teuchos::ScalarTraits<ScalarT>::magnitude(d) > zeroMag;
    const bool haveLumped = Teuchos::ScalarTraits<ScalarT>::magnitude(lumped) > zeroMag;
    // Keep lumped fallback sign-consistent with the true diagonal when both exist.
    const bool useLumped = useLumpedDiagonal && haveLumped && (!haveDiag || (d * lumped) > zero);
    const ScalarT pivot = useLumped ? lumped : d;
    if ((useLumped || haveDiag) && Teuchos::ScalarTraits<ScalarT>::magnitude(pivot) > zeroMag) {
      result.invByRow[rowGid] = one / pivot;
      if (useLumped) ++result.usedLumped;
      else ++result.usedDiag;
    }
    else {
      ++result.missing;
    }
  });
  return result;
}

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>>
buildLumpedM0inv(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > & D0,
                 const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > & M1,
                 const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & nodal_map,
                 const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & edge_map) {
  using Types = BlockTypes<Node>;
  using LA_CrsMatrix = typename Types::CrsMatrix;
  using LA_MultiVector = typename Types::MultiVector;
  using HostInds = typename Types::HostInds;
  using HostVals = typename Types::HostVals;

  TEUCHOS_TEST_FOR_EXCEPTION(M1.is_null(), std::runtime_error, "buildLumpedM0inv: M1 is null.");
  TEUCHOS_TEST_FOR_EXCEPTION(M1->getGlobalNumRows() != edge_map->getGlobalNumElements() ||
                             M1->getGlobalNumCols() != edge_map->getGlobalNumElements() ||
                             !M1->getRowMap()->isSameAs(*edge_map) ||
                             !M1->getDomainMap()->isSameAs(*edge_map),
    std::runtime_error, "buildLumpedM0inv: M1 must match edge_map.");

  Teuchos::RCP<Tpetra::Vector<ScalarT,LO,GO,Node> > m1diag =
    Teuchos::rcp(new Tpetra::Vector<ScalarT,LO,GO,Node>(edge_map));
  M1->getLocalDiagCopy(*m1diag);
  auto m1diag_2d = m1diag->getLocalViewHost(Tpetra::Access::ReadOnly);

  Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > d0_col_map = D0->getColMap();
  Teuchos::RCP<LA_MultiVector> nodalMassCol = Teuchos::rcp(new LA_MultiVector(d0_col_map, 1));
  nodalMassCol->putScalar(0.0);
  auto nodal_mass_col_2d = nodalMassCol->getLocalViewHost(Tpetra::Access::ReadWrite);

  forEachLocalRow<Node>(D0, [&](GO rowGid, const HostInds & col_lids, const HostVals & row_vals, size_t nent,
                               const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > &) {
    const LO row_lid = edge_map->getLocalElement(rowGid);
    const ScalarT edgeWeight = m1diag_2d(row_lid, 0);
    for (size_t k = 0; k < nent; ++k) {
      const ScalarT d = row_vals(k);
      nodal_mass_col_2d(col_lids(k), 0) += d * d * edgeWeight;
    }
  });

  Teuchos::RCP<LA_MultiVector> nodalMass = Teuchos::rcp(new LA_MultiVector(nodal_map, 1));
  nodalMass->putScalar(0.0);
  Teuchos::RCP<Tpetra::Export<LO,GO,Node> > col_to_domain =
    Teuchos::rcp(new Tpetra::Export<LO,GO,Node>(d0_col_map, nodal_map));
  nodalMass->doExport(*nodalMassCol, *col_to_domain, Tpetra::ADD);

  Teuchos::RCP<LA_CrsMatrix> M0inv = Teuchos::rcp(new LA_CrsMatrix(nodal_map, 1));
  auto nodal_mass_2d = nodalMass->getLocalViewHost(Tpetra::Access::ReadOnly);
  const typename Teuchos::ScalarTraits<ScalarT>::magnitudeType tiny =
    Teuchos::ScalarTraits<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType>::eps();
  const size_t numLocal = nodal_map->getLocalNumElements();
  for (size_t i = 0; i < numLocal; ++i) {
    const GO gid = nodal_map->getGlobalElement(Teuchos::as<LO>(i));
    const ScalarT m = nodal_mass_2d(Teuchos::as<LO>(i), 0);
    const auto amag = Teuchos::ScalarTraits<ScalarT>::magnitude(m);
    const ScalarT invm = (amag > tiny) ? (Teuchos::ScalarTraits<ScalarT>::one() / m)
                                       : Teuchos::ScalarTraits<ScalarT>::one();
    M0inv->insertGlobalValues(gid, Teuchos::tuple<GO>(gid), Teuchos::tuple<ScalarT>(invm));
  }
  M0inv->fillComplete(nodal_map, nodal_map);
  return M0inv;
}

template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>>
buildM0invIdentity(const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & nodal_map) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  Teuchos::RCP<LA_CrsMatrix> M0inv = Teuchos::rcp(new LA_CrsMatrix(nodal_map, 1));
  const size_t numLocal = nodal_map->getLocalNumElements();
  for (size_t i = 0; i < numLocal; ++i) {
    const GO gid = nodal_map->getGlobalElement(Teuchos::as<LO>(i));
    M0inv->insertGlobalValues(gid, Teuchos::tuple<GO>(gid), Teuchos::tuple<ScalarT>(1.0));
  }
  M0inv->fillComplete(nodal_map, nodal_map);
  return M0inv;
}

} // namespace detail

template<class Node>
BlockSystem<Node> buildBlockSystemForSet(LinearAlgebraInterface<Node> & interface,
                                         const typename BlockTypes<Node>::CrsMatrixRCP & J,
                                         const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                                         const size_t set) {
  using Types = BlockTypes<Node>;
  using LA_Map = typename Types::Map;
  std::vector<Teuchos::RCP<const LA_Map> > blockMaps = interface.buildBlockMaps(set);
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.size() < 2, std::runtime_error,
    "Block-triangular preconditioner requires at least two blocks.");

  const int pivotBlock = cntxt->schur.pivot_block;
  TEUCHOS_TEST_FOR_EXCEPTION(pivotBlock < 0 || static_cast<size_t>(pivotBlock) >= blockMaps.size(),
    std::runtime_error, "Schur pivot block index is out of range.");

  size_t targetBlock = 0;
  for (size_t b = 0; b < blockMaps.size(); ++b) {
    if (b != static_cast<size_t>(pivotBlock)) {
      targetBlock = b;
      break;
    }
  }

  Teuchos::RCP<const LA_Map> pivotMap = blockMaps[static_cast<size_t>(pivotBlock)];
  Teuchos::RCP<const LA_Map> targetMap = blockMaps[targetBlock];
  std::vector<Teuchos::RCP<const LA_Map> > pairMaps(2);
  pairMaps[0] = pivotMap;
  pairMaps[1] = targetMap;
  const std::vector<std::vector<typename Types::CrsMatrixRCP> > remappedBlocks =
    detail::extractAndRemapBlocks<Node>(J, pairMaps);

  BlockSystem<Node> blocks;
  blocks.pivotMap = pivotMap;
  blocks.targetMap = targetMap;
  blocks.J00 = remappedBlocks[0][0];
  blocks.J11 = remappedBlocks[1][1];
  blocks.J10 = remappedBlocks[1][0];
  blocks.J01 = remappedBlocks[0][1];
  blocks.pivotBlock = pivotBlock;
  blocks.targetBlock = targetBlock;
  return blocks;
}

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildDiagonalBlockInverse(const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                          const bool useLumpedDiagonal,
                          const Teuchos::RCP<const Teuchos::Comm<int> > & comm,
                          const int verbosity) {
  using Types = BlockTypes<Node>;
  using LA_Vector = typename Types::Vector;

  Teuchos::RCP<LA_Vector> invDiag = Teuchos::rcp(new LA_Vector(J00->getRowMap()));
  invDiag->putScalar(Teuchos::ScalarTraits<ScalarT>::zero());
  const detail::InverseDiagonalResult<Node> invData =
    detail::buildInverseDiagonal<Node>(Teuchos::rcp_implicit_cast<const typename Types::CrsMatrix>(J00), useLumpedDiagonal);
  for (typename std::unordered_map<GO, ScalarT>::const_iterator it = invData.invByRow.begin();
       it != invData.invByRow.end(); ++it) {
    invDiag->replaceGlobalValue(it->first, it->second);
  }

  GO globalMissing = 0;
  GO globalUsedLumped = 0;
  GO globalUsedDiag = 0;
  GO localMissing = invData.missing;
  GO localUsedLumped = invData.usedLumped;
  GO localUsedDiag = invData.usedDiag;
  Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_SUM, 1, &localMissing, &globalMissing);
  Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_SUM, 1, &localUsedLumped, &globalUsedLumped);
  Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_SUM, 1, &localUsedDiag, &globalUsedDiag);
  if (verbosity >= 5 && comm->getRank() == 0) {
    std::cout << "Pivot-block diag inverse: used_diag=" << globalUsedDiag
              << " used_lumped=" << globalUsedLumped
              << " missing=" << globalMissing << std::endl;
  }
  return Teuchos::rcp(new DiagonalInverseOperator<Node>(invDiag));
}

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildDirectBlockInverse(const typename BlockTypes<Node>::CrsMatrixRCP & A) {
  using Types = BlockTypes<Node>;
  using LA_MultiVector = typename Types::MultiVector;
  using CrsMatrix = typename Types::CrsMatrix;
  using Solver = Amesos2::Solver<CrsMatrix, LA_MultiVector>;
  Teuchos::RCP<Solver> solver = Amesos2::create<CrsMatrix, LA_MultiVector>("KLU2", A);
  solver->symbolicFactorization();
  solver->numericFactorization();
  return Teuchos::rcp(new DirectSolveOperator<Node>(solver, A->getRowMap()));
}

template<class Node>
void validateBackendSupport(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {
  const std::string backend = canonicalBlockPrecBackend(cntxt->block_prec_backend);
  TEUCHOS_TEST_FOR_EXCEPTION(backend != "teko_full", std::runtime_error,
    "Block-triangular preconditioner requires 'block prec backend = teko_full'.");
}

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildOrReusePivotBlock(LinearAlgebraInterface<Node> & interface,
                       const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                       const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                       Teuchos::ParameterList & mueluParams,
                       BlockPrecType pivotType) {
  using Types = BlockTypes<Node>;
  if (pivotType == BlockPrecType::RefMaxwell) {
    interface.validateRefMaxwellBlockInputs(J00, cntxt);
    return interface.buildRefMaxwellPreconditioner(J00, cntxt, cntxt->pivot_block_sublist);
  }
  if (pivotType == BlockPrecType::Direct) {
    return buildDirectBlockInverse<Node>(J00);
  }
  if (pivotType == BlockPrecType::Diagonal) {
    return buildDiagonalBlockInverse<Node>(J00, cntxt->schur.pivot_block_diag_use_lumped_diagonal,
                                           interface.comm, interface.verbosity);
  }
  return MueLu::CreateTpetraPreconditioner(
    Teuchos::rcp_implicit_cast<typename Types::Operator>(J00), mueluParams);
}

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildOrReuseSchurBlock(LinearAlgebraInterface<Node> & interface,
                       const typename BlockTypes<Node>::CrsMatrixRCP & schurApprox,
                       const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                       Teuchos::ParameterList & mueluParams,
                       BlockPrecType schurType) {
  using Types = BlockTypes<Node>;
  TEUCHOS_TEST_FOR_EXCEPTION(schurType == BlockPrecType::Diagonal, std::runtime_error,
    "Schur block does not support Diagonal.");
  if (schurType == BlockPrecType::RefMaxwell) {
    interface.validateRefMaxwellBlockInputs(schurApprox, cntxt);
    return interface.buildRefMaxwellPreconditioner(schurApprox, cntxt, cntxt->schur_block_sublist, true);
  }
  if (schurType == BlockPrecType::Direct) {
    return buildDirectBlockInverse<Node>(schurApprox);
  }
  return MueLu::CreateTpetraPreconditioner(
    Teuchos::rcp_implicit_cast<typename Types::Operator>(schurApprox), mueluParams);
}

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildPivotBlockPrec(LinearAlgebraInterface<Node> & interface,
                    const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                    Teuchos::ParameterList & pivotMueLuParams) {
  const BlockPrecType pivotType = parseBlockPrecType(cntxt->schur.pivot_block_preconditioner_type);
  return buildOrReusePivotBlock<Node>(interface, J00, cntxt, pivotMueLuParams, pivotType);
}

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildSchurBlockPrec(LinearAlgebraInterface<Node> & interface,
                    const typename BlockTypes<Node>::CrsMatrixRCP & SchurApprox,
                    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                    Teuchos::ParameterList & schurMueLuParams) {
  const BlockPrecType schurType = parseBlockPrecType(cntxt->schur.schur_block_preconditioner_type);
  return buildOrReuseSchurBlock<Node>(interface, SchurApprox, cntxt, schurMueLuParams, schurType);
}

} // namespace block_prec
} // namespace MrHyDE

#endif
