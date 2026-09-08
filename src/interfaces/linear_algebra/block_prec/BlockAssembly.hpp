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
#include <sstream>
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
    const LO nLocalDomain = static_cast<LO>(srcDomainMap->getLocalNumElements());
    for (LO lid = 0; lid < nLocalDomain; ++lid) {
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

template<class Node>
struct FilterResult {
  Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> matrix;
  std::vector<std::pair<GO,GO>> dropped;
};

struct FilterOpts {
  bool   filterSM           = false;
  bool   verifyComplex      = false;
  bool   verifyKnConsistency = false;
  double tol                = 1.0e-14;
};

inline FilterOpts readFilterOpts(const Teuchos::ParameterList & pl) {
  FilterOpts o;
  if (pl.isParameter("filter SM"))              o.filterSM            = pl.get<bool>("filter SM");
  if (pl.isParameter("filter threshold"))       o.tol                 = pl.get<double>("filter threshold");
  if (pl.isParameter("verify complex"))         o.verifyComplex       = pl.get<bool>("verify complex");
  if (pl.isParameter("verify Kn consistency"))  o.verifyKnConsistency = pl.get<bool>("verify Kn consistency");
  return o;
}

// Drop a_ij with |a_ij| < tol * sqrt(|a_ii|*|a_jj|).
template<class Node>
FilterResult<Node>
filterExplicitZeros(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src,
                    const typename Teuchos::ScalarTraits<ScalarT>::magnitudeType tol,
                    const bool captureDropped = false) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  using LA_Vector = Tpetra::Vector<ScalarT,LO,GO,Node>;
  using LA_Import = Tpetra::Import<LO,GO,Node>;

  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node>> rowMap = src->getRowMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node>> colMap = src->getColMap();

  Teuchos::RCP<LA_Vector> rowDiag = Teuchos::rcp(new LA_Vector(rowMap, true));
  src->getLocalDiagCopy(*rowDiag);
  Teuchos::RCP<LA_Vector> colDiag;
  if (rowMap->isSameAs(*colMap)) {
    colDiag = rowDiag;
  } else {
    colDiag = Teuchos::rcp(new LA_Vector(colMap, true));
    LA_Import importer(rowMap, colMap);
    colDiag->doImport(*rowDiag, importer, Tpetra::INSERT);
  }
  auto rowDiagView = rowDiag->getLocalViewHost(Tpetra::Access::ReadOnly);
  auto colDiagView = colDiag->getLocalViewHost(Tpetra::Access::ReadOnly);

  const size_t maxEnt = std::max<size_t>(1, src->getLocalMaxNumRowEntries());
  Teuchos::RCP<LA_CrsMatrix> out = Teuchos::rcp(new LA_CrsMatrix(rowMap, maxEnt));

  FilterResult<Node> result;
  const LO n_rows = static_cast<LO>(rowMap->getLocalNumElements());
  for (LO lid = 0; lid < n_rows; ++lid) {
    const GO rowGid = rowMap->getGlobalElement(lid);
    size_t nent = src->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    host_inds_t cols("flt_cols", nent);
    host_vals_t vals("flt_vals", nent);
    src->getLocalRowCopy(lid, cols, vals, nent);

    const MagT aii = Teuchos::ScalarTraits<ScalarT>::magnitude(rowDiagView(lid, 0));

    std::vector<GO> keepGids;
    std::vector<ScalarT> keepVals;
    keepGids.reserve(nent);
    keepVals.reserve(nent);
    for (size_t k = 0; k < nent; ++k) {
      const LO colLid = cols(k);
      const GO colGid = colMap->getGlobalElement(colLid);
      if (colGid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
      const MagT a  = Teuchos::ScalarTraits<ScalarT>::magnitude(vals(k));
      const bool isDiag = (colGid == rowGid);
      if (isDiag) {
        keepGids.push_back(colGid);
        keepVals.push_back(vals(k));
        continue;
      }
      const MagT ajj = Teuchos::ScalarTraits<ScalarT>::magnitude(colDiagView(colLid, 0));
      const MagT scale = std::sqrt(aii * ajj);
      if (a >= tol * scale) {
        keepGids.push_back(colGid);
        keepVals.push_back(vals(k));
      } else if (captureDropped) {
        result.dropped.emplace_back(rowGid, colGid);
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(keepGids.empty(), std::runtime_error,
      "filterExplicitZeros: row " << rowGid << " (local " << lid << ") emptied out"
      << " (input nnz=" << nent << ", |a_ii|=" << aii << ", tol=" << tol << ")."
      << " Filter would erase this row's physics. Lower 'filter threshold' or"
      << " exclude this row category.");
    out->insertGlobalValues(rowGid, keepGids, keepVals);
  }
  out->fillComplete(src->getDomainMap(), src->getRangeMap());
  result.matrix = out;
  return result;
}

// O(nnz log nnz) per row; adequate for lowest-order bases.
template<class Node>
void assertStructuralSymmetry(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & A,
                              const std::string & label) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;
  Tpetra::RowMatrixTransposer<ScalarT,LO,GO,Node> transposer(Teuchos::rcp_const_cast<LA_CrsMatrix>(A));
  Teuchos::RCP<LA_CrsMatrix> At = transposer.createTranspose();
  const LO n = static_cast<LO>(A->getRowMap()->getLocalNumElements());
  for (LO lid = 0; lid < n; ++lid) {
    size_t nA = A->getNumEntriesInLocalRow(lid);
    size_t nT = At->getNumEntriesInLocalRow(lid);
    TEUCHOS_TEST_FOR_EXCEPTION(nA != nT, std::runtime_error,
      label << ": structural symmetry broken at row " << A->getRowMap()->getGlobalElement(lid)
      << " (nnz " << nA << " vs transpose nnz " << nT << ").");
    if (nA == 0) continue;
    host_inds_t colsA("sym_colsA", nA), colsT("sym_colsT", nT);
    host_vals_t valsA("sym_valsA", nA), valsT("sym_valsT", nT);
    A->getLocalRowCopy(lid, colsA, valsA, nA);
    At->getLocalRowCopy(lid, colsT, valsT, nT);
    std::set<GO> gA, gT;
    for (size_t k = 0; k < nA; ++k) gA.insert(A->getColMap()->getGlobalElement(colsA(k)));
    for (size_t k = 0; k < nT; ++k) gT.insert(At->getColMap()->getGlobalElement(colsT(k)));
    TEUCHOS_TEST_FOR_EXCEPTION(gA != gT, std::runtime_error,
      label << ": structural symmetry broken at row " << A->getRowMap()->getGlobalElement(lid)
      << " (column-GID sets differ).");
  }
}

// Bound the filter perturbation only; SM*D0 is nonzero.
template<class Node>
void assertKernelBound(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & SM_filtered,
                       const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & SM_orig,
                       const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & D0,
                       const typename Teuchos::ScalarTraits<ScalarT>::magnitudeType tol,
                       const std::string & label) {
  using LA_MultiVector = Tpetra::MultiVector<ScalarT,LO,GO,Node>;
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  Teuchos::RCP<LA_MultiVector> x = Teuchos::rcp(new LA_MultiVector(D0->getDomainMap(), 1));
  x->putScalar(Teuchos::ScalarTraits<ScalarT>::one());
  Teuchos::RCP<LA_MultiVector> Dx = Teuchos::rcp(new LA_MultiVector(D0->getRangeMap(), 1));
  D0->apply(*x, *Dx);
  Teuchos::Array<MagT> dx_nrm(1);
  Dx->normInf(dx_nrm());
  Teuchos::RCP<LA_MultiVector> SDx_orig = Teuchos::rcp(new LA_MultiVector(SM_orig->getRangeMap(), 1));
  Teuchos::RCP<LA_MultiVector> SDx_filt = Teuchos::rcp(new LA_MultiVector(SM_filtered->getRangeMap(), 1));
  SM_orig->apply(*Dx, *SDx_orig);
  SM_filtered->apply(*Dx, *SDx_filt);
  SDx_orig->update(-Teuchos::ScalarTraits<ScalarT>::one(), *SDx_filt, Teuchos::ScalarTraits<ScalarT>::one());
  Teuchos::Array<MagT> pert_nrm(1);
  SDx_orig->normInf(pert_nrm());
  const MagT sm_norm = SM_orig->getFrobeniusNorm();
  const MagT bound = tol * sm_norm * dx_nrm[0];
  TEUCHOS_TEST_FOR_EXCEPTION(pert_nrm[0] > bound, std::runtime_error,
    label << ": filter perturbed SM*D0 beyond tol: |(SM - SM_filt) D0 x|_inf = " << pert_nrm[0]
    << " > tol * |SM|_F * |D0 x|_inf = " << bound << " (tol=" << tol << ").");
}

// D0 and nullspace mismatches throw; symmetry and Rayleigh checks warn.
template<class Node>
bool verifyMaxwellComplex(
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & D0,
    const Teuchos::RCP<const Tpetra::MultiVector<
      typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node>> & coords,
    const Teuchos::RCP<const Tpetra::MultiVector<ScalarT,LO,GO,Node>> & nullspace,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & SM,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & M1,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & SM_f,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & M1_f,
    const std::vector<std::pair<GO,GO>> & dropped_SM,
    const std::vector<std::pair<GO,GO>> & dropped_M1,
    const typename Teuchos::ScalarTraits<ScalarT>::magnitudeType tol,
    const int verbosity,
    const int rank,
    const std::string & label) {
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using LA_MultiVector = Tpetra::MultiVector<ScalarT,LO,GO,Node>;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;

  auto log = [&](const std::string & line) {
    if (verbosity >= 6 && rank == 0) std::cout << "[" << label << " verify] " << line << std::endl;
  };
  auto logs = [&](auto&&... args) {
    if (verbosity < 6 || rank != 0) return;
    std::ostringstream os;
    (os << ... << args);
    log(os.str());
  };

  bool ok_assertions = true;
  const auto comm = D0->getRowMap()->getComm();

  // Row structure: allow entries in {+-1, +-0.5}.
  {
    const auto rowMap = D0->getRowMap();
    const LO nrows = static_cast<LO>(rowMap->getLocalNumElements());
    LO local_bad_nnz = 0, local_bad_val = 0, local_bad_sum = 0;
    LO local_empty = 0, local_single = 0, local_pair = 0;
    LO local_half = 0, local_unit = 0;
    for (LO lid = 0; lid < nrows; ++lid) {
      size_t nent = D0->getNumEntriesInLocalRow(lid);
      if (nent == 0) { local_empty++; continue; }
      if (nent > 2) local_bad_nnz++;
      host_inds_t cols("c1_cols", nent);
      host_vals_t vals("c1_vals", nent);
      D0->getLocalRowCopy(lid, cols, vals, nent);
      ScalarT sum = Teuchos::ScalarTraits<ScalarT>::zero();
      for (size_t k = 0; k < nent; ++k) {
        const ScalarT v = vals(k);
        const bool is_unit = (v == ScalarT(1.0) || v == ScalarT(-1.0));
        const bool is_half = (v == ScalarT(0.5) || v == ScalarT(-0.5));
        if (is_unit) local_unit++;
        else if (is_half) local_half++;
        else local_bad_val++;
        sum += v;
      }
      if (nent == 1) local_single++;
      else if (nent == 2) local_pair++;
      if (nent >= 2 && sum != Teuchos::ScalarTraits<ScalarT>::zero()) local_bad_sum++;
    }
    LO g[9] = {local_bad_nnz, local_bad_val, local_bad_sum,
               local_empty, local_single, local_pair, nrows,
               local_unit, local_half};
    LO gout[9];
    Teuchos::reduceAll<int,LO>(*comm, Teuchos::REDUCE_SUM, 9, g, gout);
    logs("D0 rows: total=", gout[6],
         " empty=", gout[3], " one_ep=", gout[4], " two_ep=", gout[5],
         " unit_vals=", gout[7], " half_vals=", gout[8],
         " bad_nnz=", gout[0], " bad_val=", gout[1], " bad_sum=", gout[2]);
    if (gout[0] || gout[1] || gout[2]) {
      ok_assertions = false;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "[" << label << "] D0 structure violated: bad_nnz="
        << gout[0] << " bad_val=" << gout[1] << " bad_sum=" << gout[2]);
    }
  }

  // Exact only for lowest-order Whitney elements.
  if (!coords.is_null() && !nullspace.is_null()) {
    const int dim = std::min<int>(coords->getNumVectors(), nullspace->getNumVectors());
    LA_MultiVector coords_S(D0->getDomainMap(), dim);
    {
      auto c_h = coords->getLocalViewHost(Tpetra::Access::ReadOnly);
      auto s_h = coords_S.getLocalViewHost(Tpetra::Access::OverwriteAll);
      const LO n_local = static_cast<LO>(coords->getLocalLength());
      for (int d = 0; d < dim; ++d)
        for (LO i = 0; i < n_local; ++i) s_h(i, d) = static_cast<ScalarT>(c_h(i, d));
    }
    LA_MultiVector D0coords(D0->getRangeMap(), dim);
    D0->apply(coords_S, D0coords);
    LA_MultiVector diff(D0->getRangeMap(), dim);
    diff.update(Teuchos::ScalarTraits<ScalarT>::one(), *nullspace, -Teuchos::ScalarTraits<ScalarT>::one(),
                D0coords, Teuchos::ScalarTraits<ScalarT>::zero());
    Teuchos::Array<MagT> maxAbs(dim);
    diff.normInf(maxAbs());
    MagT worst = MagT(0);
    for (int d = 0; d < dim; ++d) if (maxAbs[d] > worst) worst = maxAbs[d];
    logs("|nullspace - D0*coords|_inf = ", worst);
    const MagT tol_null = MagT(16) * Teuchos::ScalarTraits<MagT>::eps();
    if (worst > tol_null) {
      ok_assertions = false;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "[" << label << "] |nullspace - D0*coords|_inf = " << worst
        << " > " << tol_null << " (expected machine zero).");
    }
  } else {
    log("nullspace check skipped (coords or nullspace null)");
  }

  // Symmetry: |xTAy - yTAx| / (|x||y||A|_inf) with random x, y.
  auto sym_test = [&](const Teuchos::RCP<const LA_CrsMatrix> & A, const std::string & name) {
    const auto rowMap = A->getRowMap();
    MagT Ainf = MagT(0);
    {
      const LO nr = static_cast<LO>(rowMap->getLocalNumElements());
      for (LO i = 0; i < nr; ++i) {
        size_t nent = A->getNumEntriesInLocalRow(i);
        if (nent == 0) continue;
        host_inds_t cols("sym_cols", nent);
        host_vals_t vals("sym_vals", nent);
        A->getLocalRowCopy(i, cols, vals, nent);
        MagT rs = MagT(0);
        for (size_t k = 0; k < nent; ++k) rs += Teuchos::ScalarTraits<ScalarT>::magnitude(vals(k));
        if (rs > Ainf) Ainf = rs;
      }
      MagT Ainf_g = Ainf;
      Teuchos::reduceAll<int,MagT>(*comm, Teuchos::REDUCE_MAX, 1, &Ainf, &Ainf_g);
      Ainf = Ainf_g;
    }
    LA_MultiVector x(rowMap, 1), y(rowMap, 1), Ax(rowMap, 1), Ay(rowMap, 1);
    for (int seed = 0; seed < 3; ++seed) {
      x.randomize(); y.randomize();
      A->apply(x, Ax);
      A->apply(y, Ay);
      Teuchos::Array<ScalarT> xtAy(1), ytAx(1);
      Teuchos::Array<MagT> nx(1), ny(1);
      x.dot(Ay, xtAy()); y.dot(Ax, ytAx());
      x.norm2(nx());     y.norm2(ny());
      const MagT diff = Teuchos::ScalarTraits<ScalarT>::magnitude(xtAy[0] - ytAx[0]);
      const MagT denom = std::max(nx[0] * ny[0] * Ainf, MagT(1e-30));
      const MagT rel = diff / denom;
      logs("symmetry ", name, " seed=", seed, ": |xTAy - yTAx| rel = ", rel);
      if (rel > MagT(1e-13)) {
        logs("symmetry ", name, " seed=", seed, ": rel ", rel, " > 1e-13");
      }
    }
  };
  sym_test(SM_f, "SM_f");
  sym_test(M1_f, "M1_f");

  // Filter must preserve the (SM - M1)*D0*v residual.
  if (!SM.is_null() && !M1.is_null()) {
    const auto nodalMap = D0->getDomainMap();
    LA_MultiVector v(nodalMap, 1);
    v.randomize();
    LA_MultiVector D0v(D0->getRangeMap(), 1);
    D0->apply(v, D0v);
    LA_MultiVector r_unf(SM->getRangeMap(), 1);
    LA_MultiVector t1(SM->getRangeMap(), 1), t2(M1->getRangeMap(), 1);
    SM->apply(D0v, t1);
    M1->apply(D0v, t2);
    r_unf.update(Teuchos::ScalarTraits<ScalarT>::one(), t1, -Teuchos::ScalarTraits<ScalarT>::one(),
                 t2, Teuchos::ScalarTraits<ScalarT>::zero());
    LA_MultiVector r_flt(SM_f->getRangeMap(), 1);
    LA_MultiVector t1f(SM_f->getRangeMap(), 1), t2f(M1_f->getRangeMap(), 1);
    SM_f->apply(D0v, t1f);
    M1_f->apply(D0v, t2f);
    r_flt.update(Teuchos::ScalarTraits<ScalarT>::one(), t1f, -Teuchos::ScalarTraits<ScalarT>::one(),
                 t2f, Teuchos::ScalarTraits<ScalarT>::zero());
    Teuchos::Array<MagT> nr_u(1), nr_f(1), nDv(1);
    r_unf.norm2(nr_u()); r_flt.norm2(nr_f()); D0v.norm2(nDv());
    const MagT rat = (nr_u[0] > MagT(1e-30)) ? (nr_f[0] / nr_u[0]) : MagT(1);
    logs("(SM-M1)*D0v: unf=", nr_u[0], " flt=", nr_f[0], " |D0v|=", nDv[0], " ratio flt/unf=", rat);
    if (nr_u[0] > MagT(1e-30) && (rat > MagT(2.0) || rat < MagT(0.5))) {
      logs("filter shifted (SM-M1)*D0v by more than 2x (ratio ", rat, ")");
    }
  }

  // Rayleigh ratio must stay within |dropped| * tol of 1.
  auto rayleigh = [&](const Teuchos::RCP<const LA_CrsMatrix> & A,
                      const Teuchos::RCP<const LA_CrsMatrix> & A_f,
                      const size_t nDrop,
                      const std::string & name) {
    const MagT delta = std::max(MagT(1e-12), static_cast<MagT>(nDrop) * tol);
    const auto rowMap = A->getRowMap();
    LA_MultiVector x(rowMap, 1), Ax(rowMap, 1), Afx(A_f->getRangeMap(), 1);
    auto one_test = [&](const std::string & tag) {
      A->apply(x, Ax);
      A_f->apply(x, Afx);
      Teuchos::Array<ScalarT> num(1), den(1);
      x.dot(Afx, num()); x.dot(Ax, den());
      const MagT ratio = (Teuchos::ScalarTraits<ScalarT>::magnitude(den[0]) > MagT(1e-30))
        ? Teuchos::ScalarTraits<ScalarT>::magnitude(num[0] / den[0])
        : MagT(0);
      logs("Rayleigh ", name, " ", tag, ": xTA_f x / xTAx = ", ratio,
           " (window 1 +/- ", delta, ")");
      if (Teuchos::ScalarTraits<ScalarT>::magnitude(den[0]) > MagT(1e-30) &&
          (ratio > MagT(1) + delta || ratio < MagT(1) - delta)) {
        logs("Rayleigh ", name, " ", tag, ": outside window");
      }
    };
    for (int seed = 0; seed < 3; ++seed) {
      x.randomize();
      one_test("rand seed=" + std::to_string(seed));
    }
    if (!nullspace.is_null() && nullspace->getMap()->isSameAs(*rowMap)) {
      for (int d = 0; d < static_cast<int>(nullspace->getNumVectors()); ++d) {
        auto s_h = x.getLocalViewHost(Tpetra::Access::OverwriteAll);
        auto n_h = nullspace->getLocalViewHost(Tpetra::Access::ReadOnly);
        for (LO i = 0; i < static_cast<LO>(rowMap->getLocalNumElements()); ++i)
          s_h(i, 0) = n_h(i, d);
        one_test("null col=" + std::to_string(d));
      }
    }
  };
  rayleigh(SM, SM_f, dropped_SM.size(), "SM");
  rayleigh(M1, M1_f, dropped_M1.size(), "M1");

  return ok_assertions;
}

template<class Node>
struct MaxwellMatrices {
  typename BlockTypes<Node>::CrsMatrixRCP SM;
  typename BlockTypes<Node>::CrsMatrixRCP M1;
  std::vector<std::pair<GO,GO>> droppedSM;
  std::vector<std::pair<GO,GO>> droppedM1;
};

template<class Node>
MaxwellMatrices<Node>
prepareMaxwellMatrices(
    const typename BlockTypes<Node>::CrsMatrixRCP & SM,
    const typename BlockTypes<Node>::CrsMatrixRCP & M1,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & D0,
    const Teuchos::RCP<const Tpetra::MultiVector<
      typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node>> & coords,
    const Teuchos::RCP<const Tpetra::MultiVector<ScalarT,LO,GO,Node>> & nullspace,
    const FilterOpts & opts,
    const int verbosity,
    const std::string & label) {
  MaxwellMatrices<Node> out{SM, M1, {}, {}};
  auto pct_dropped = [](double in, double out_) {
    return 100.0 * (in - out_) / std::max(1.0, in);
  };
  if (opts.filterSM) {
    FilterResult<Node> smResult = filterExplicitZeros<Node>(SM, opts.tol, opts.verifyComplex);
    FilterResult<Node> m1Result = filterExplicitZeros<Node>(M1, opts.tol, opts.verifyComplex);
    out.SM = smResult.matrix;
    out.M1 = m1Result.matrix;
    out.droppedSM = smResult.dropped;
    out.droppedM1 = m1Result.dropped;
    assertStructuralSymmetry<Node>(out.M1, label + " M1 filter");
    assertKernelBound<Node>(out.SM, SM, D0, opts.tol, label + " SM filter");
    if (verbosity >= 6 && SM->getComm()->getRank() == 0) {
      const double smIn  = SM->getGlobalNumEntries();
      const double smOut = out.SM->getGlobalNumEntries();
      const double m1In  = M1->getGlobalNumEntries();
      const double m1Out = out.M1->getGlobalNumEntries();
      std::cout << "[" << label << "] filter SM tol=" << opts.tol
                << ": SM " << smIn << " -> " << smOut
                << " (dropped " << pct_dropped(smIn, smOut) << "%)"
                << ", M1 " << m1In << " -> " << m1Out
                << " (dropped " << pct_dropped(m1In, m1Out) << "%)"
                << std::endl;
    }
  }
  if (opts.verifyComplex) {
    verifyMaxwellComplex<Node>(
      D0, coords, nullspace,
      SM, M1, out.SM, out.M1,
      out.droppedSM, out.droppedM1, opts.tol, verbosity,
      SM->getComm()->getRank(), label);
  }
  return out;
}

// Drop stored zeros and snap remaining values to +-1.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>>
snapCrsMatrixSignsInPlace(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  // Tuned to O(1) D0 entries; entries in [10*eps, 100*eps] are treated as noise.
  const MagT zero_tol = Teuchos::ScalarTraits<MagT>::eps() * 1e2;
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node>> rowMap = src->getRowMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node>> colMap = src->getColMap();
  const size_t maxEnt = std::max<size_t>(1, src->getLocalMaxNumRowEntries());
  Teuchos::RCP<LA_CrsMatrix> out = Teuchos::rcp(new LA_CrsMatrix(rowMap, maxEnt));
  const LO n_rows = static_cast<LO>(rowMap->getLocalNumElements());
  for (LO lid = 0; lid < n_rows; ++lid) {
    size_t nent = src->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    host_inds_t cols("snap_cols", nent);
    host_vals_t vals("snap_vals", nent);
    src->getLocalRowCopy(lid, cols, vals, nent);
    const GO rowGid = rowMap->getGlobalElement(lid);
    std::vector<GO> keepGids;
    std::vector<ScalarT> keepVals;
    keepGids.reserve(nent);
    keepVals.reserve(nent);
    for (size_t k = 0; k < nent; ++k) {
      const ScalarT v = vals(k);
      const MagT m = Teuchos::ScalarTraits<ScalarT>::magnitude(v);
      if (m <= zero_tol) continue;
      const GO colGid = colMap->getGlobalElement(cols(k));
      if (colGid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
      keepGids.push_back(colGid);
      keepVals.push_back(v > ScalarT(0) ? ScalarT(1) : ScalarT(-1));
    }
    if (!keepGids.empty()) {
      out->insertGlobalValues(rowGid, keepGids, keepVals);
    }
  }
  out->fillComplete(src->getDomainMap(), src->getRangeMap());
  return out;
}

// Drop BC rows and columns from D0. bcColsDomain is indexed over the domain map.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>>
dropBCRowsAndCols(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src,
                  const Kokkos::View<const bool*, typename Node::device_type::memory_space> & bcRows,
                  const Kokkos::View<const bool*, typename Node::device_type::memory_space> & bcColsDomain) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using LA_Vector = Tpetra::Vector<ScalarT,LO,GO,Node>;
  using LA_Import = Tpetra::Import<LO,GO,Node>;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;
  auto bcRowsHost = Kokkos::create_mirror_view(bcRows);
  Kokkos::deep_copy(bcRowsHost, bcRows);

  const auto rowMap    = src->getRowMap();
  const auto colMap    = src->getColMap();
  const auto domainMap = src->getDomainMap();
  const LO n_rows   = static_cast<LO>(rowMap->getLocalNumElements());
  const LO n_cols   = static_cast<LO>(colMap->getLocalNumElements());
  const LO n_domain = static_cast<LO>(domainMap->getLocalNumElements());
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<size_t>(bcRowsHost.extent(0)) != static_cast<size_t>(n_rows),
    std::runtime_error, "dropBCRowsAndCols: bcRows length != row map local size.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<size_t>(bcColsDomain.extent(0)) != static_cast<size_t>(n_domain),
    std::runtime_error, "dropBCRowsAndCols: bcColsDomain length != domain map local size.");

  // Promote domain-map BC flags to the ghosted column map.
  Teuchos::RCP<LA_Vector> bcColDomainScalar = Teuchos::rcp(new LA_Vector(domainMap, false));
  {
    auto v = bcColDomainScalar->getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto bcDomHost = Kokkos::create_mirror_view(bcColsDomain);
    Kokkos::deep_copy(bcDomHost, bcColsDomain);
    for (LO i = 0; i < n_domain; ++i) v(i,0) = bcDomHost(i) ? ScalarT(1) : ScalarT(0);
  }
  Teuchos::RCP<LA_Vector> bcColColMapScalar = Teuchos::rcp(new LA_Vector(colMap, true));
  if (domainMap->isSameAs(*colMap)) {
    Tpetra::deep_copy(*bcColColMapScalar, *bcColDomainScalar);
  } else {
    LA_Import importer(domainMap, colMap);
    bcColColMapScalar->doImport(*bcColDomainScalar, importer, Tpetra::INSERT);
  }
  std::vector<bool> bcColColMap(n_cols, false);
  {
    auto v = bcColColMapScalar->getLocalViewHost(Tpetra::Access::ReadOnly);
    for (LO i = 0; i < n_cols; ++i) bcColColMap[i] = v(i,0) != ScalarT(0);
  }

  const size_t maxEnt = std::max<size_t>(1, src->getLocalMaxNumRowEntries());
  Teuchos::RCP<LA_CrsMatrix> out = Teuchos::rcp(new LA_CrsMatrix(rowMap, maxEnt));
  for (LO lid = 0; lid < n_rows; ++lid) {
    if (bcRowsHost(lid)) continue;
    size_t nent = src->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    host_inds_t cols("bcdrop_cols", nent);
    host_vals_t vals("bcdrop_vals", nent);
    src->getLocalRowCopy(lid, cols, vals, nent);
    const GO rowGid = rowMap->getGlobalElement(lid);
    std::vector<GO> keepGids;
    std::vector<ScalarT> keepVals;
    keepGids.reserve(nent);
    keepVals.reserve(nent);
    for (size_t k = 0; k < nent; ++k) {
      const LO cLid = cols(k);
      if (bcColColMap[cLid]) continue;
      const GO colGid = colMap->getGlobalElement(cLid);
      if (colGid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
      keepGids.push_back(colGid);
      keepVals.push_back(vals(k));
    }
    if (!keepGids.empty()) {
      out->insertGlobalValues(rowGid, keepGids, keepVals);
    }
  }
  out->fillComplete(src->getDomainMap(), src->getRangeMap());
  return out;
}

// Drop BC rows (retain columns).
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>>
dropBCRows(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src,
           const Kokkos::View<const bool*, typename Node::device_type::memory_space> & bcRows) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;
  auto bcHost = Kokkos::create_mirror_view(bcRows);
  Kokkos::deep_copy(bcHost, bcRows);
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node>> rowMap = src->getRowMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node>> colMap = src->getColMap();
  const size_t maxEnt = std::max<size_t>(1, src->getLocalMaxNumRowEntries());
  Teuchos::RCP<LA_CrsMatrix> out = Teuchos::rcp(new LA_CrsMatrix(rowMap, maxEnt));
  const LO n_rows = static_cast<LO>(rowMap->getLocalNumElements());
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<size_t>(bcHost.extent(0)) != static_cast<size_t>(n_rows),
    std::runtime_error, "dropBCRows: bcRows length does not match row map local size.");
  for (LO lid = 0; lid < n_rows; ++lid) {
    if (bcHost(lid)) continue;
    size_t nent = src->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    host_inds_t cols("bcdrop_cols", nent);
    host_vals_t vals("bcdrop_vals", nent);
    src->getLocalRowCopy(lid, cols, vals, nent);
    const GO rowGid = rowMap->getGlobalElement(lid);
    std::vector<GO> keepGids;
    std::vector<ScalarT> keepVals;
    keepGids.reserve(nent);
    keepVals.reserve(nent);
    for (size_t k = 0; k < nent; ++k) {
      const GO colGid = colMap->getGlobalElement(cols(k));
      if (colGid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
      keepGids.push_back(colGid);
      keepVals.push_back(vals(k));
    }
    if (!keepGids.empty()) {
      out->insertGlobalValues(rowGid, keepGids, keepVals);
    }
  }
  out->fillComplete(src->getDomainMap(), src->getRangeMap());
  return out;
}

// Symmetric rescale rows with |diag| > contrastRatio * min |diag| by
// s = 1/sqrt(|diag|).
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>>
rescalePecRows(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src,
               const typename Teuchos::ScalarTraits<ScalarT>::magnitudeType contrastRatio = 1.0e6) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using LA_Vector = Tpetra::Vector<ScalarT,LO,GO,Node>;
  using LA_Import = Tpetra::Import<LO,GO,Node>;
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;

  const auto rowMap = src->getRowMap();
  const auto colMap = src->getColMap();
  Teuchos::RCP<LA_Vector> rowDiag = Teuchos::rcp(new LA_Vector(rowMap, true));
  src->getLocalDiagCopy(*rowDiag);

  // Reference scale: global min of nonzero |diag|.
  MagT localMin = std::numeric_limits<MagT>::max();
  {
    auto dv = rowDiag->getLocalViewHost(Tpetra::Access::ReadOnly);
    const LO nlocal = static_cast<LO>(dv.extent(0));
    for (LO i = 0; i < nlocal; ++i) {
      const MagT a = Teuchos::ScalarTraits<ScalarT>::magnitude(dv(i,0));
      if (a > MagT(0) && a < localMin) localMin = a;
    }
  }
  MagT globalMin = localMin;
  Teuchos::reduceAll<int,MagT>(*rowMap->getComm(), Teuchos::REDUCE_MIN, 1, &localMin, &globalMin);
  if (!(globalMin > MagT(0))) globalMin = MagT(1);
  const MagT threshold = contrastRatio * globalMin;

  Teuchos::RCP<LA_Vector> rowScale = Teuchos::rcp(new LA_Vector(rowMap, false));
  {
    auto dv = rowDiag->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto sv = rowScale->getLocalViewHost(Tpetra::Access::OverwriteAll);
    const LO nlocal = static_cast<LO>(dv.extent(0));
    for (LO i = 0; i < nlocal; ++i) {
      const MagT a = Teuchos::ScalarTraits<ScalarT>::magnitude(dv(i,0));
      sv(i,0) = (a > threshold) ? ScalarT(MagT(1) / std::sqrt(a)) : ScalarT(1);
    }
  }

  Teuchos::RCP<LA_Vector> colScale;
  if (rowMap->isSameAs(*colMap)) {
    colScale = rowScale;
  } else {
    colScale = Teuchos::rcp(new LA_Vector(colMap, true));
    LA_Import importer(rowMap, colMap);
    colScale->doImport(*rowScale, importer, Tpetra::INSERT);
  }

  const size_t maxEnt = std::max<size_t>(1, src->getLocalMaxNumRowEntries());
  Teuchos::RCP<LA_CrsMatrix> out = Teuchos::rcp(new LA_CrsMatrix(rowMap, maxEnt));
  auto rowScaleView = rowScale->getLocalViewHost(Tpetra::Access::ReadOnly);
  auto colScaleView = colScale->getLocalViewHost(Tpetra::Access::ReadOnly);
  const LO n_rows = static_cast<LO>(rowMap->getLocalNumElements());
  for (LO lid = 0; lid < n_rows; ++lid) {
    size_t nent = src->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    host_inds_t cols("resc_cols", nent);
    host_vals_t vals("resc_vals", nent);
    src->getLocalRowCopy(lid, cols, vals, nent);
    const GO rowGid = rowMap->getGlobalElement(lid);
    const ScalarT sr = rowScaleView(lid, 0);
    std::vector<GO> outGids;
    std::vector<ScalarT> outVals;
    outGids.reserve(nent);
    outVals.reserve(nent);
    for (size_t k = 0; k < nent; ++k) {
      const GO colGid = colMap->getGlobalElement(cols(k));
      if (colGid == Teuchos::OrdinalTraits<GO>::invalid()) continue;
      const ScalarT sc = colScaleView(cols(k), 0);
      outGids.push_back(colGid);
      outVals.push_back(sr * vals(k) * sc);
    }
    if (!outGids.empty()) {
      out->insertGlobalValues(rowGid, outGids, outVals);
    }
  }
  out->fillComplete(src->getDomainMap(), src->getRangeMap());
  return out;
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
    detail::buildInverseDiagonal<Node>(J00, useLumpedDiagonal);
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
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
maybeWrapInInnerKrylov(LinearAlgebraInterface<Node> & interface,
                       const typename BlockTypes<Node>::CrsMatrixRCP & blockMat,
                       const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & innerPrec,
                       const Teuchos::ParameterList & blockList,
                       const std::string & label,
                       const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {
  if (!blockList.isParameter("inner krylov solver")) return innerPrec;
  using Types = BlockTypes<Node>;
  using LA_MultiVector = typename Types::MultiVector;
  using LA_Operator = typename Types::Operator;
  using LA_LinearProblem = Belos::LinearProblem<ScalarT, LA_MultiVector, LA_Operator>;
  // Variable inner solves need a flexible outer Krylov.
  if (!cntxt.is_null()) {
    const std::string outerType = toUpperAsciiCopy(cntxt->belos_type);
    const bool outerIsGmres = (outerType == "BLOCK GMRES" || outerType == "PSEUDO BLOCK GMRES");
    const bool flexibleFlag = cntxt->flexible_gmres;
    TEUCHOS_TEST_FOR_EXCEPTION(!outerIsGmres || !flexibleFlag, std::runtime_error,
      "[" << label << "] 'inner krylov solver' is set on a block preconditioner, but the "
      "outer Belos solver is '" << cntxt->belos_type << "'"
      << (outerIsGmres ? "" : " (not Block/Pseudo Block GMRES)")
      << (flexibleFlag ? "" : " and 'Flexible Gmres: true' is not set (top level or in Belos Settings)")
      << ". Inner-Krylov wrapping produces a variable-precision preconditioner that "
         "requires a flexible outer Krylov (FGMRES). Either remove 'inner krylov solver' "
         "from this block, or configure the outer solver as: Belos solver: Block GMRES with "
         "'Flexible Gmres: true' at top level or under 'Belos Settings'.");
  }
  const std::string innerSolver = blockList.get<std::string>("inner krylov solver");
  const int innerMaxIters = blockList.isParameter("inner krylov max iters")
    ? blockList.get<int>("inner krylov max iters") : 5;
  const double innerTol = blockList.isParameter("inner krylov tol")
    ? blockList.get<double>("inner krylov tol") : 1.0e-2;
  Teuchos::RCP<Teuchos::ParameterList> belosList = Teuchos::rcp(new Teuchos::ParameterList);
  belosList->set("Maximum Iterations", innerMaxIters);
  belosList->set("Num Blocks", innerMaxIters);
  belosList->set("Convergence Tolerance", innerTol);
  belosList->set("Verbosity", static_cast<int>(Belos::Errors));
  belosList->set("Output Frequency", 0);
  belosList->set("Output Style", static_cast<int>(Belos::Brief));
  belosList->set("Implicit Residual Scaling", std::string("Norm of Initial Residual"));
  Teuchos::RCP<LA_LinearProblem> problem = Teuchos::rcp(new LA_LinearProblem(
    Teuchos::rcp_implicit_cast<LA_Operator>(blockMat), Teuchos::null, Teuchos::null));
  problem->setRightPrec(innerPrec);
  auto solver = interface.createBelosSolverManager(problem, belosList, innerSolver);
  if (interface.verbosity >= 10 && interface.comm->getRank() == 0) {
    std::cout << "[" << label << "] wrapping block preconditioner in inner Belos '"
              << innerSolver << "' (max iters=" << innerMaxIters
              << ", tol=" << innerTol << ")" << std::endl;
  }
  return Teuchos::rcp_implicit_cast<LA_Operator>(
    Teuchos::rcp(new KrylovWrappedBlockOperator<Node>(blockMat, solver, problem)));
}

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
buildOrReusePivotBlock(LinearAlgebraInterface<Node> & interface,
                       const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                       const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                       Teuchos::ParameterList & mueluParams,
                       BlockPrecType pivotType) {
  using Types = BlockTypes<Node>;
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > innerPrec;
  if (pivotType == BlockPrecType::RefMaxwell) {
    interface.validateRefMaxwellBlockInputs(J00, cntxt);
    innerPrec = interface.buildRefMaxwellPreconditioner(J00, cntxt, cntxt->pivot_block_sublist);
  } else if (pivotType == BlockPrecType::Maxwell1) {
    interface.validateRefMaxwellBlockInputs(J00, cntxt);
    innerPrec = interface.buildMaxwell1Preconditioner(J00, cntxt, cntxt->pivot_block_sublist);
  } else if (pivotType == BlockPrecType::Direct) {
    innerPrec = buildDirectBlockInverse<Node>(J00);
  } else if (pivotType == BlockPrecType::Diagonal) {
    innerPrec = buildDiagonalBlockInverse<Node>(J00, cntxt->schur.pivot_block_diag_use_lumped_diagonal,
                                                interface.comm, interface.verbosity);
  } else {
    innerPrec = MueLu::CreateTpetraPreconditioner(
      Teuchos::rcp_implicit_cast<typename Types::Operator>(J00), mueluParams);
  }
  return maybeWrapInInnerKrylov<Node>(interface, J00, innerPrec, cntxt->pivot_block_sublist, "BlockTri pivot", cntxt);
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
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > innerPrec;
  if (schurType == BlockPrecType::RefMaxwell) {
    interface.validateRefMaxwellBlockInputs(schurApprox, cntxt);
    innerPrec = interface.buildRefMaxwellPreconditioner(schurApprox, cntxt, cntxt->schur_block_sublist, true);
  } else if (schurType == BlockPrecType::Maxwell1) {
    interface.validateRefMaxwellBlockInputs(schurApprox, cntxt);
    innerPrec = interface.buildMaxwell1Preconditioner(schurApprox, cntxt, cntxt->schur_block_sublist, true);
  } else if (schurType == BlockPrecType::Direct) {
    innerPrec = buildDirectBlockInverse<Node>(schurApprox);
  } else {
    innerPrec = MueLu::CreateTpetraPreconditioner(
      Teuchos::rcp_implicit_cast<typename Types::Operator>(schurApprox), mueluParams);
  }
  return maybeWrapInInnerKrylov<Node>(interface, schurApprox, innerPrec, cntxt->schur_block_sublist, "BlockTri Schur", cntxt);
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
