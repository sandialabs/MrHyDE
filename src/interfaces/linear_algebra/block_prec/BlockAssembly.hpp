#ifndef MRHYDE_BLOCK_PREC_ASSEMBLY_HPP
#define MRHYDE_BLOCK_PREC_ASSEMBLY_HPP

#include "block_prec/BlockOperators.hpp"
#include "block_prec/InverseLibraryOps.hpp"
#include "block_prec/ParamUtils.hpp"
#include "linearAlgebraInterface.hpp"
#include "linearSolverContext.hpp"

#include <BelosLinearProblem.hpp>
#include <BelosTpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_VectorFactory.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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
MatrixRCP<Node>
buildScaledInverseDiagonalMatrix(const Teuchos::RCP<typename BlockTypes<Node>::MultiVector> & v,
                                 const ScalarT scale) {
  using CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  auto map = v->getMap();
  MatrixRCP<Node> out = Teuchos::rcp(new CrsMatrix(map, 1));
  auto vv = v->getLocalViewHost(Tpetra::Access::ReadOnly);
  Teuchos::Array<GO> col(1);
  Teuchos::Array<ScalarT> val(1);
  for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i) {
    col[0] = map->getGlobalElement(i);
    val[0] = scale / vv(i, 0);
    out->insertGlobalValues(col[0], col(), val());
  }
  out->fillComplete(map, map);
  return out;
}

// Deterministic: randomize() would perturb MueLu's Chebyshev eigenvalue estimates.
// Seeds must differ where two probes have to be linearly independent.
template<class Node>
void fillProbe(typename BlockTypes<Node>::MultiVector & v, const int seed = 0) {
  auto vv = v.getLocalViewHost(Tpetra::Access::OverwriteAll);
  auto map = v.getMap();
  const GO period = 7 + 2 * static_cast<GO>(seed);
  const size_t nloc = map->getLocalNumElements();
  for (size_t i = 0; i < nloc; ++i) {
    const GO gi = map->getGlobalElement(static_cast<LO>(i));
    for (size_t j = 0; j < v.getNumVectors(); ++j) {
      const GO g = gi + static_cast<GO>(3 * j);
      vv(i, j) = static_cast<ScalarT>(1.0 + (g % period)) * (((g + seed) % 2) ? 1.0 : -1.0);
    }
  }
}

// Xpetra view of a Tpetra matrix, not a copy.
template<class Node>
Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> >
wrapAsXpetraMatrix(const ConstMatrixRCP<Node> & A) {
  using TpetraCrs = Tpetra::CrsMatrix<ScalarT,LO,GO,Node>;
  using XpetraCrs = Xpetra::TpetraCrsMatrix<ScalarT,LO,GO,Node>;
  using XpetraCrsMatrix = Xpetra::CrsMatrix<ScalarT,LO,GO,Node>;
  using XpetraCrsWrap = Xpetra::CrsMatrixWrap<ScalarT,LO,GO,Node>;
  if (A.is_null()) return Teuchos::null;
  return Teuchos::rcp(new XpetraCrsWrap(
    Teuchos::rcp_implicit_cast<XpetraCrsMatrix>(
      Teuchos::rcp(new XpetraCrs(Teuchos::rcp_const_cast<TpetraCrs>(A))))));
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

  const MatrixRCP<Node> dst = Teuchos::rcp(new CrsMatrix(rowMap, maxEnt));

  Teuchos::RCP<IntVector> domainMarker = Teuchos::rcp(new IntVector(srcDomainMap));
  {
    auto markerView = domainMarker->getLocalViewHost(Tpetra::Access::OverwriteAll);
    const LO nDomain = static_cast<LO>(srcDomainMap->getLocalNumElements());
    for (LO lid = 0; lid < nDomain; ++lid) {
      markerView(lid, 0) = domainMap->isNodeGlobalElement(srcDomainMap->getGlobalElement(lid)) ? 1 : 0;
    }
  }
  Teuchos::RCP<IntVector> colMarker;
  Teuchos::RCP<const Import> srcImporter = src->getGraph()->getImporter();
  if (srcImporter.is_null()) {
    colMarker = domainMarker;
  }
  else {
    colMarker = Teuchos::rcp(new IntVector(srcColMap));
    colMarker->putScalar(0);
    colMarker->doImport(*domainMarker, *srcImporter, Tpetra::INSERT);
  }
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

  dst->fillComplete(domainMap, rowMap);
  return dst;
}

template<class Node>
std::vector<std::vector<MatrixRCP<Node> > >
extractAndRemapBlocks(const MatrixRCP<Node> & J,
                      const std::vector<MapRCP<Node> > & blockMaps,
                      const bool diagonalOnly = false) {
  const size_t nBlocks = blockMaps.size();
  std::vector<std::vector<MatrixRCP<Node> > > remapped(
    nBlocks, std::vector<MatrixRCP<Node> >(nBlocks, Teuchos::null));

  for (size_t i = 0; i < nBlocks; ++i) {
    for (size_t j = 0; j < nBlocks; ++j) {
      if (diagonalOnly && i != j) continue;
      remapped[i][j] = remapBlockToMaps<Node>(J, blockMaps[i], blockMaps[j]);
      TEUCHOS_TEST_FOR_EXCEPTION(
        !remapped[i][j]->getRowMap()->isSameAs(*blockMaps[i]) ||
        !remapped[i][j]->getDomainMap()->isSameAs(*blockMaps[j]),
        std::runtime_error,
        "extractAndRemapBlocks: map contract check failed for block (" << i << "," << j << ").");
    }
  }
  return remapped;
}

struct InverseDiagonalCounts {
  GO missing = 0;
  GO usedLumped = 0;
  GO usedDiag = 0;
};

template<class Node>
Teuchos::RCP<typename BlockTypes<Node>::Vector>
buildInverseDiagonal(const ConstMatrixRCP<Node> & mat,
                     const bool useLumpedDiagonal,
                     InverseDiagonalCounts & counts) {
  using Types = BlockTypes<Node>;
  using LA_Vector = typename Types::Vector;
  const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
  const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();

  typename Types::MapRCP rowMap = mat->getRowMap();
  Teuchos::RCP<LA_Vector> inv = Teuchos::rcp(new LA_Vector(rowMap, false));
  LA_Vector diag(rowMap, false);
  mat->getLocalDiagCopy(diag);

  LA_Vector lumped(mat->getRangeMap(), false);
  if (useLumpedDiagonal) {
    LA_Vector ones(mat->getDomainMap(), false);
    ones.putScalar(one);
    mat->apply(ones, lumped);   // row sums
  } else {
    lumped.putScalar(zero);
  }

  auto dView = diag.getLocalViewDevice(Tpetra::Access::ReadOnly);
  auto lView = lumped.getLocalViewDevice(Tpetra::Access::ReadOnly);
  auto iView = inv->getLocalViewDevice(Tpetra::Access::OverwriteAll);
  const size_t nrows = static_cast<size_t>(rowMap->getLocalNumElements());
  const bool wantLumped = useLumpedDiagonal;
  GO nDiag = 0, nLumped = 0, nMissing = 0;
  Kokkos::parallel_reduce("buildInverseDiagonal",
    Kokkos::RangePolicy<typename Node::execution_space, size_t>(0, nrows),
    KOKKOS_LAMBDA(const size_t i, GO & ad, GO & al, GO & am) {
      const ScalarT d = dView(i, 0);
      const ScalarT sum = lView(i, 0);
      // Keep the lumped fallback sign-consistent with the true diagonal.
      const bool useLumped = wantLumped && sum != zero && (d == zero || (d * sum) > zero);
      const ScalarT pivot = useLumped ? sum : d;
      if (pivot != zero) {
        iView(i, 0) = one / pivot;
        if (useLumped) ++al; else ++ad;
      } else {
        iView(i, 0) = zero;
        ++am;
      }
    }, nDiag, nLumped, nMissing);
  counts.usedDiag = nDiag;
  counts.usedLumped = nLumped;
  counts.missing = nMissing;
  return inv;
}

template<class Node>
void reportInverseDiagonal(const InverseDiagonalCounts & result,
                           const std::string & label,
                           const Teuchos::RCP<const Teuchos::Comm<int> > & comm,
                           const int verbosity) {
  GO local[3] = {result.usedDiag, result.usedLumped, result.missing};
  GO global[3] = {0, 0, 0};
  Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_SUM, 3, local, global);
  if (global[2] > 0 && comm->getRank() == 0) {
    std::cout << label << ": WARNING " << global[2]
              << " rows have no usable diagonal; their inverse is zero." << std::endl;
  }
  if (verbosity < 5) return;
  if (comm->getRank() == 0) {
    std::cout << label << ": used_diag=" << global[0]
              << " used_lumped=" << global[1]
              << " missing=" << global[2] << std::endl;
  }
}


template<class Node>
struct FilterResult {
  Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> matrix;
  std::vector<std::pair<GO,GO>> dropped;  // populated only when captureDropped=true
  size_t nnzIn = 0;   // global, before filtering
  size_t nnzOut = 0;  // global, after
};

// Disabled by default to preserve the unfiltered setup.
struct FilterOpts {
  bool   filterSM           = false;
  bool   verifyComplex      = false;
  bool   verifyKnConsistency = false;  // Maxwell1 only
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

// Drop small off-diagonals relative to their row and column diagonals.
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
  GO emptiedRow = -1;
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
    if (keepGids.empty()) {
      if (emptiedRow < 0) emptiedRow = rowGid;
      continue;
    }
    out->insertGlobalValues(rowGid, keepGids, keepVals);
  }
  GO worstEmptied = -1;
  Teuchos::reduceAll<int, GO>(*src->getComm(), Teuchos::REDUCE_MAX, 1, &emptiedRow, &worstEmptied);
  TEUCHOS_TEST_FOR_EXCEPTION(worstEmptied >= 0, std::runtime_error,
    "filterExplicitZeros: 'filter threshold' emptied row " << worstEmptied << ".");
  out->fillComplete(src->getDomainMap(), src->getRangeMap());
  result.matrix = out;
  result.nnzIn = src->getGlobalNumEntries();
  result.nnzOut = out->getGlobalNumEntries();
  return result;
}

// Require every nonzero (i,j) to have a matching (j,i).
// TODO: might be expensive for high-order stencils.
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
      label << ": row " << A->getRowMap()->getGlobalElement(lid)
      << " has " << nA << " entries; its transpose has " << nT << ".");
    if (nA == 0) continue;
    host_inds_t colsA("sym_colsA", nA), colsT("sym_colsT", nT);
    host_vals_t valsA("sym_valsA", nA), valsT("sym_valsT", nT);
    A->getLocalRowCopy(lid, colsA, valsA, nA);
    At->getLocalRowCopy(lid, colsT, valsT, nT);
    std::set<GO> gA, gT;
    for (size_t k = 0; k < nA; ++k) gA.insert(A->getColMap()->getGlobalElement(colsA(k)));
    for (size_t k = 0; k < nT; ++k) gT.insert(At->getColMap()->getGlobalElement(colsT(k)));
    TEUCHOS_TEST_FOR_EXCEPTION(gA != gT, std::runtime_error,
      label << ": row " << A->getRowMap()->getGlobalElement(lid)
      << " has different columns in A and A^T.");
  }
}

// Check only the filter's change to SM*D0. The DIRK mass term makes SM*D0 nonzero.
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
    label << ": SM filter changed SM*D0 above tolerance"
    << " (measured=" << pert_nrm[0] << ", limit=" << bound << ", tol=" << tol << ").");
}

// The kernel bound rides with the SM filter in finishMaxwellInputs instead.
template<class Node>
FilterResult<Node> filterM1Checked(
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & M1,
    const FilterOpts & opts,
    const std::string & label) {
  FilterResult<Node> out = filterExplicitZeros<Node>(M1, opts.tol, opts.verifyComplex);
  assertStructuralSymmetry<Node>(out.matrix, label + " M1 filter");
  return out;
}

template<class Node>
void logFilterCounts(const FilterResult<Node> & sm, const FilterResult<Node> & m1,
                     const FilterOpts & opts, const std::string & label,
                     const int verbosity, const int rank) {
  if (verbosity < 6 || rank != 0) return;
  auto pct = [](size_t in, size_t out) {
    return 100.0 * static_cast<double>(in - out) / static_cast<double>(std::max<size_t>(in, 1));
  };
  std::cout << "[" << label << "] filter SM tol=" << opts.tol
            << ": SM " << sm.nnzIn << " -> " << sm.nnzOut
            << " (dropped " << pct(sm.nnzIn, sm.nnzOut) << "%)"
            << ", M1 " << m1.nnzIn << " -> " << m1.nnzOut
            << " (dropped " << pct(m1.nnzIn, m1.nnzOut) << "%)" << std::endl;
}

// Import a nodal domain-map mask onto Kn's column map.
template<class Node>
Kokkos::View<bool*, typename Node::device_type::memory_space>
knColumnMask(const Teuchos::RCP<Xpetra::Matrix<ScalarT, LO, GO, Node> > & Kn,
             const Kokkos::View<bool*, typename Node::device_type::memory_space> & BCdomainNodal) {
  using dev_mem_space = typename Node::device_type::memory_space;
  using XpetraVector = Xpetra::Vector<ScalarT, LO, GO, Node>;
  auto knColMap = Kn->getColMap();
  auto knRowMap = Kn->getRowMap();
  auto knDomMap = Kn->getDomainMap();
  TEUCHOS_TEST_FOR_EXCEPTION(!knRowMap->isSameAs(*knDomMap), std::runtime_error,
    "knColumnMask: Kn row and domain maps must agree.");
  const LO nColLocal = static_cast<LO>(knColMap->getLocalNumElements());
  Kokkos::View<bool*, dev_mem_space> BCcols("BCcols_kn", nColLocal);
  const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
  const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();

  Teuchos::RCP<XpetraVector> bcDom =
      Xpetra::VectorFactory<ScalarT, LO, GO, Node>::Build(knDomMap, true);
  {
    auto domView = bcDom->getLocalViewDevice(Tpetra::Access::OverwriteAll);
    Kokkos::parallel_for("BCcols_mark_domain",
        Kokkos::RangePolicy<typename Node::execution_space>(0, domView.extent(0)),
        KOKKOS_LAMBDA(const LO i) {
          domView(i, 0) = BCdomainNodal(i) ? one : zero;
        });
  }

  Teuchos::RCP<XpetraVector> bcCol = bcDom;
  auto knImporter = Kn->getCrsGraph()->getImporter();
  if (!knImporter.is_null()) {
    bcCol = Xpetra::VectorFactory<ScalarT, LO, GO, Node>::Build(knColMap, true);
    bcCol->doImport(*bcDom, *knImporter, Xpetra::INSERT);
  }

  auto colView = bcCol->getLocalViewDevice(Tpetra::Access::ReadOnly);
  Kokkos::parallel_for("BCcols_fill",
      Kokkos::RangePolicy<typename Node::execution_space>(0, nColLocal),
      KOKKOS_LAMBDA(const LO cLid) {
        BCcols(cLid) = (colView(cLid, 0) != zero);
      });
  return BCcols;
}

// resetMatrix() recomputes with reuse hints; ComputePrec defaults to true.
template<class Node, class PrecT>
Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> >
resetAndWrap(const Teuchos::RCP<PrecT> & prec,
             const Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node> > & SM,
             const char * label, const bool forSchur, const int verbosity, const int rank) {
  prec->resetMatrix(wrapAsXpetraMatrix<Node>(SM));
  if (verbosity >= 10 && rank == 0) {
    std::cout << "[" << label << "] Reusing existing hierarchy with resetMatrix()"
              << (forSchur ? " (Schur)" : "") << std::endl;
  }
  return Teuchos::rcp(new MueLu::TpetraOperator<ScalarT, LO, GO, Node>(
      Teuchos::rcp_static_cast<Xpetra::Operator<ScalarT, LO, GO, Node> >(prec)));
}

// De Rham sanity checks. D0 row structure throws; symmetry, curl(grad)=0 and
// the Rayleigh quotient only warn.
template<class Node>
void verifyMaxwellComplex(
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & D0,
    const Teuchos::RCP<const Tpetra::MultiVector<
      typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node>> & coords,
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

  const auto comm = D0->getRowMap()->getComm();

  // D0 row structure. Panzer OPERATOR_GRAD uses +-0.5; Reitzinger +-1. Both are valid.
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
    TEUCHOS_TEST_FOR_EXCEPTION(gout[0] || gout[1] || gout[2], std::runtime_error,
      "[" << label << "] invalid D0 rows: too_many_entries=" << gout[0]
      << ", invalid_values=" << gout[1] << ", nonzero_sums=" << gout[2] << ".");
  }

  // Symmetry: |xTAy - yTAx| / (|x||y||A|_inf) on two independent probes.
  auto sym_test = [&](const Teuchos::RCP<const LA_CrsMatrix> & A, const std::string & name) {
    const auto rowMap = A->getRowMap();
    // Estimate |A|_inf with the maximum absolute row sum.
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
      fillProbe<Node>(x, 2 * seed); fillProbe<Node>(y, 2 * seed + 1);
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

  // Filtering should not change the small curl-curl residual on gradients.
  if (!SM.is_null() && !M1.is_null()) {
    const auto nodalMap = D0->getDomainMap();
    LA_MultiVector v(nodalMap, 1);
    fillProbe<Node>(v);
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

  // The filtered and original Rayleigh quotients should agree within the drop bound.
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
      fillProbe<Node>(x, seed);
      one_test("probe seed=" + std::to_string(seed));
    }
  };
  rayleigh(SM, SM_f, dropped_SM.size(), "SM");
  rayleigh(M1, M1_f, dropped_M1.size(), "M1");
}

// SM is filtered even on reuse: resetMatrix() must get the matrix the
// hierarchy was built from.
template<class Node>
struct MaxwellInputs {
  ConstMatrixRCP<Node> SM_orig, M1_orig, D0;
  MatrixRCP<Node> SM, M1;
  FilterResult<Node> smFilter, m1Filter;
};

template<class Node>
MaxwellInputs<Node> filterSMOnly(const ConstMatrixRCP<Node> & SM,
                                 const ConstMatrixRCP<Node> & M1,
                                 const ConstMatrixRCP<Node> & D0,
                                 const FilterOpts & opts, const bool forReuse) {
  MaxwellInputs<Node> in;
  in.SM_orig = SM;
  in.M1_orig = M1;
  in.D0 = D0;
  in.SM = Teuchos::rcp_const_cast<typename BlockTypes<Node>::CrsMatrix>(SM);
  in.M1 = Teuchos::rcp_const_cast<typename BlockTypes<Node>::CrsMatrix>(M1);
  if (!opts.filterSM) return in;
  // Dropped entries only feed verifyMaxwellComplex, which reuse skips.
  in.smFilter = filterExplicitZeros<Node>(SM, opts.tol, opts.verifyComplex && !forReuse);
  in.SM = in.smFilter.matrix;
  return in;
}

// Build-only: the kernel bound, the M1 filter, the complex checks.
template<class Node>
void finishMaxwellInputs(MaxwellInputs<Node> & in,
                         const Teuchos::RCP<const Tpetra::MultiVector<
                           typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node>> & coords,
                         const FilterOpts & opts, const std::string & label,
                         const int verbosity, const int rank) {
  if (opts.filterSM) {
    assertKernelBound<Node>(in.SM, in.SM_orig, in.D0, opts.tol, label + " SM filter");
    in.m1Filter = filterM1Checked<Node>(in.M1_orig, opts, label);
    in.M1 = in.m1Filter.matrix;
    logFilterCounts<Node>(in.smFilter, in.m1Filter, opts, label, verbosity, rank);
  }
  if (opts.verifyComplex) {
    verifyMaxwellComplex<Node>(in.D0, coords, in.SM_orig, in.M1_orig, in.SM, in.M1,
                               in.smFilter.dropped, in.m1Filter.dropped,
                               opts.tol, verbosity, rank, label);
  }
}

// Normalize D0 to {-1, +1} and remove stored zeros. ReitzingerPFactory
// rejects all other values; dropBCRows prevents MueLu from adding zeros back.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>>
snapCrsMatrixSigns(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;

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

// Remove boundary rows so Maxwell1 cannot add stored zeros to D0.
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
    std::runtime_error, "dropBCRows: bcRows has " << bcHost.extent(0)
    << " entries; expected " << n_rows << ".");
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

} // namespace detail

template<class Node>
BlockSystem<Node> buildBlockSystemForSet(LinearAlgebraInterface<Node> & interface,
                                         const typename BlockTypes<Node>::CrsMatrixRCP & J,
                                         const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                                         const size_t set) {
  using Types = BlockTypes<Node>;
  using LA_Map = typename Types::Map;
  std::vector<Teuchos::RCP<const LA_Map> > blockMaps = interface.buildBlockMaps(set);
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.size() != 2, std::runtime_error,
    "Block-triangular preconditioner supports exactly two variable blocks, but set "
    << set << " has " << blockMaps.size()
    << ". N-block support through Teko is not implemented.");

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
  detail::InverseDiagonalCounts counts;
  Teuchos::RCP<typename Types::Vector> invDiag =
    detail::buildInverseDiagonal<Node>(
      Teuchos::rcp_implicit_cast<const typename Types::CrsMatrix>(J00), useLumpedDiagonal, counts);
  detail::reportInverseDiagonal<Node>(counts, "Pivot-block diag inverse", comm, verbosity);
  return Teuchos::rcp(new DiagonalInverseOperator<Node>(invDiag));
}

template<class Node>
Teko::LinearOp
buildDirectBlockInverse(const typename BlockTypes<Node>::CrsMatrixRCP & A,
                        const std::string & label,
                        const std::string & solverName) {
  Teuchos::ParameterList entry;
  entry.set("Solver Type", solverName);
  // Block row maps are extracted from the monolithic map, so they are not contiguous.
  entry.sublist("Amesos2 Settings").sublist(solverName).set("IsContiguous", false);
  return buildLibraryInverse<Node>("Amesos2", entry, label + " Amesos2", A);
}


template<class Node>
Teko::LinearOp
maybeWrapInInnerKrylov(LinearAlgebraInterface<Node> & interface,
                       const typename BlockTypes<Node>::CrsMatrixRCP & blockMat,
                       const Teko::LinearOp & innerPrec,
                       const Teuchos::ParameterList & blockList,
                       const std::string & label,
                       const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {
  if (!blockList.isParameter("inner krylov solver")) return innerPrec;
  // Inner Krylov gives a different operator on every apply, so the outer solver
  // has to be right-preconditioned flexible GMRES.
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt.is_null() ||
                             toUpperAsciiCopy(cntxt->belos_type) != "BLOCK GMRES" ||
                             !cntxt->flexible_gmres || !cntxt->right_preconditioner,
    std::runtime_error,
    "[" << label << "] 'inner krylov solver' requires Block GMRES with "
    "'Flexible Gmres: true' and 'right preconditioner: true'.");
  const std::string innerSolver = blockList.get<std::string>("inner krylov solver");
  const int innerMaxIters = blockList.isParameter("inner krylov max iters")
    ? blockList.get<int>("inner krylov max iters") : 5;
  const double innerTol = blockList.isParameter("inner krylov tol")
    ? blockList.get<double>("inner krylov tol") : 1.0e-2;
  Teuchos::ParameterList belosList;
  belosList.set("Solver Type", innerSolver);
  Teuchos::ParameterList & solverList =
    belosList.sublist("Solver Types").sublist(innerSolver);
  solverList.set("Maximum Iterations", innerMaxIters);
  solverList.set("Num Blocks", innerMaxIters);
  solverList.set("Convergence Tolerance", innerTol);
  solverList.set("Verbosity", static_cast<int>(Belos::Errors));
  solverList.set("Output Frequency", 0);
  solverList.set("Output Style", static_cast<int>(Belos::Brief));
  solverList.set("Implicit Residual Scaling", std::string("Norm of Initial Residual"));
  if (interface.verbosity >= 10 && interface.comm->getRank() == 0) {
    std::cout << "[" << label << "] wrapping block preconditioner in inner Belos '"
              << innerSolver << "' (max iters=" << innerMaxIters
              << ", tol=" << innerTol << ")" << std::endl;
  }
  return buildLibraryInverse<Node>("Belos", belosList, label + " inner", blockMat, innerPrec);
}

template<class Node, class GenericFn>
Teko::LinearOp
buildBlockOperator(LinearAlgebraInterface<Node> & interface,
                   const typename BlockTypes<Node>::CrsMatrixRCP & mat,
                   const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                   const Teuchos::ParameterList & blockList,
                   const BlockPrecType type,
                   const bool forSchur,
                   const std::string & label,
                   GenericFn && buildGeneric) {
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > tpetraPrec;
  Teko::LinearOp innerPrec;
  switch (type) {
    case BlockPrecType::RefMaxwell:
      interface.validateRefMaxwellBlockInputs(mat, cntxt);
      tpetraPrec = interface.buildRefMaxwellPreconditioner(mat, cntxt, blockList, forSchur);
      break;
    case BlockPrecType::Maxwell1:
      interface.validateRefMaxwellBlockInputs(mat, cntxt);
      tpetraPrec = interface.buildMaxwell1Preconditioner(mat, cntxt, blockList, forSchur);
      break;
    case BlockPrecType::Direct:
      innerPrec = buildDirectBlockInverse<Node>(mat, label,
        cntxt.is_null() ? std::string("KLU2") : cntxt->amesos_type);
      break;
    case BlockPrecType::Diagonal:
      // S is formed from J00, so inverting its diagonal is not an approximation of it.
      TEUCHOS_TEST_FOR_EXCEPTION(forSchur, std::runtime_error,
        "Schur block does not support Diagonal.");
      tpetraPrec = buildDiagonalBlockInverse<Node>(mat, cntxt->schur.pivot_block_diag_use_lumped_diagonal,
                                                   interface.comm, interface.verbosity);
      break;
    default:
      innerPrec = buildGeneric();
      break;
  }
  if (innerPrec.is_null()) {
    innerPrec = tpetraToThyra<Node>(tpetraPrec, mat->getRangeMap(), mat->getDomainMap());
  }
  return maybeWrapInInnerKrylov<Node>(interface, mat, innerPrec, blockList, label, cntxt);
}

template<class Node>
Teko::LinearOp
buildPivotBlockPrec(LinearAlgebraInterface<Node> & interface,
                    const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                    Teuchos::ParameterList & pivotMueLuParams) {
  return buildBlockOperator<Node>(interface, J00, cntxt, cntxt->pivot_block_sublist,
    parseBlockPrecType(cntxt->schur.pivot_block_preconditioner_type), false, "BlockTri pivot",
    [&] {
      return buildLibraryInverse<Node>("MueLu", pivotMueLuParams, "BlockTri pivot MueLu", J00);
    });
}

template<class Node>
Teko::LinearOp
buildSchurBlockPrec(LinearAlgebraInterface<Node> & interface,
                    const typename BlockTypes<Node>::CrsMatrixRCP & SchurApprox,
                    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                    Teuchos::ParameterList & schurMueLuParams) {
  return buildBlockOperator<Node>(interface, SchurApprox, cntxt, cntxt->schur_block_sublist,
    parseBlockPrecType(cntxt->schur.schur_block_preconditioner_type), true, "BlockTri Schur",
    [&] {
      return buildLibraryInverse<Node>("MueLu", schurMueLuParams, "BlockTri Schur MueLu", SchurApprox);
    });
}

} // namespace block_prec
} // namespace MrHyDE

#endif
