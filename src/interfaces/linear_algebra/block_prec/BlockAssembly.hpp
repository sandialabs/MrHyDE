/***********************************************************************
 MrHyDE - Jacobian block extraction and per-block preconditioner assembly.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_ASSEMBLY_HPP
#define MRHYDE_BLOCK_PREC_ASSEMBLY_HPP

#include "block_prec/CrsFilter.hpp"
#include "block_prec/InverseLibraryOps.hpp"
#include "block_prec/ParamUtils.hpp"
#include "block_prec/TekoAdapter.hpp"
#include "linearSolverContext.hpp"

#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MatrixUtils.hpp>
#include <Xpetra_TpetraVector.hpp>
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

// blocks[i][j] couples split i to split j; maps[i] is split i's owned row map. Every block
// keeps the GIDs it has in the monolithic Jacobian.
template<class Node>
struct BlockSystem {
  using Types = BlockTypes<Node>;
  using matrix_rcp = typename Types::CrsMatrixRCP;
  using map_rcp = typename Types::MapRCP;

  std::vector<map_rcp> maps;
  std::vector<std::vector<matrix_rcp> > blocks;

  size_t numBlocks() const { return maps.size(); }
};


namespace detail {

template<class Node>
using MapRCP = typename BlockTypes<Node>::MapRCP;

template<class Node>
using MatrixRCP = typename BlockTypes<Node>::CrsMatrixRCP;

template<class Node>
using ConstMatrixRCP = Teuchos::RCP<const typename BlockTypes<Node>::CrsMatrix>;

template<class Node>
Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> >
buildRefMaxwellM0inv(const Teuchos::RCP<typename BlockTypes<Node>::MultiVector> & nodalLumpedMass,
                     const ScalarT beta) {
  using LA_Vector = typename BlockTypes<Node>::Vector;
  Teuchos::RCP<LA_Vector> inv = Teuchos::rcp(new LA_Vector(nodalLumpedMass->getMap(), false));
  inv->reciprocal(*nodalLumpedMass->getVector(0));
  inv->scale(beta);
  Teuchos::RCP<const Xpetra::Vector<ScalarT,LO,GO,Node> > xinv =
    Teuchos::rcp(new Xpetra::TpetraVector<ScalarT,LO,GO,Node>(inv));
  return Xpetra::MatrixFactory<ScalarT,LO,GO,Node>::Build(xinv);
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

// Rows with no diagonal get one
template<class Node>
void repairNodalDiagonal(Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > & Kn,
                         const int verbosity) {
  using STS = Teuchos::ScalarTraits<ScalarT>;
  static const Teuchos::RCP<Teuchos::FancyOStream> quiet =
    Teuchos::fancyOStream(Teuchos::rcp(new Teuchos::oblackholestream()));
  const ScalarT maxDiag =
    MueLu::UtilitiesBase<ScalarT,LO,GO,Node>::GetMatrixDiagonal(*Kn)->normInf();
  Xpetra::MatrixUtils<ScalarT,LO,GO,Node>::CheckRepairMainDiagonal(
    Kn, true,
    (verbosity >= 10) ? *Teuchos::VerboseObjectBase::getDefaultOStream() : *quiet,
    STS::zero(), maxDiag > STS::zero() ? maxDiag : STS::one());
}

// Column map is the domain map first, then the surviving remote columns: MueLu needs a
// column map whose local part equals the row map.
template<class Node>
MatrixRCP<Node>
remapBlockToMaps(const ConstMatrixRCP<Node> & src,
                 const MapRCP<Node> & rowMap,
                 const MapRCP<Node> & domainMap) {
  using Types = BlockTypes<Node>;
  using LA_Map = typename Types::Map;
  using IntVector = typename Types::IntVector;
  using Import = typename Types::Import;
  using device_type = typename Node::device_type;

  TEUCHOS_TEST_FOR_EXCEPTION(src.is_null(), std::runtime_error, "remapBlockToMaps: source block is null.");
  TEUCHOS_TEST_FOR_EXCEPTION(rowMap.is_null() || domainMap.is_null(), std::runtime_error,
    "remapBlockToMaps: target row/domain map is null.");

  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > srcRowMap = src->getRowMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > srcColMap = src->getColMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > srcDomainMap = src->getDomainMap();

  // A column survives when the target domain map owns its GID somewhere.
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

  const LO nSrcCols = static_cast<LO>(srcColMap->getLocalNumElements());
  Kokkos::View<LO*, device_type> outColLid("remap_out_col_lid", nSrcCols);
  auto outColLidHost = Kokkos::create_mirror_view(outColLid);
  std::vector<GO> colGids;
  colGids.reserve(static_cast<size_t>(nSrcCols) + domainMap->getLocalNumElements());
  {
    auto owned = domainMap->getLocalElementList();
    for (size_t i = 0; i < static_cast<size_t>(owned.size()); ++i) colGids.push_back(owned[i]);
  }
  for (LO c = 0; c < nSrcCols; ++c) {
    if (markerData[c] == 0) {
      outColLidHost(c) = -1;
      continue;
    }
    const GO colGid = srcColMap->getGlobalElement(c);
    const LO ownedLid = domainMap->getLocalElement(colGid);
    if (ownedLid != Teuchos::OrdinalTraits<LO>::invalid()) {
      outColLidHost(c) = ownedLid;
      continue;
    }
    outColLidHost(c) = static_cast<LO>(colGids.size());
    colGids.push_back(colGid);
  }
  Kokkos::deep_copy(outColLid, outColLidHost);
  const MapRCP<Node> colMap = Teuchos::rcp(new LA_Map(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), colGids, 0, rowMap->getComm()));

  const LO nRows = static_cast<LO>(rowMap->getLocalNumElements());
  Kokkos::View<LO*, device_type> srcRowLid("remap_src_row_lid", nRows);
  auto srcRowLidHost = Kokkos::create_mirror_view(srcRowLid);
  for (LO i = 0; i < nRows; ++i) {
    const LO s = srcRowMap->getLocalElement(rowMap->getGlobalElement(i));
    srcRowLidHost(i) = (s == Teuchos::OrdinalTraits<LO>::invalid()) ? LO(-1) : s;
  }
  Kokkos::deep_copy(srcRowLid, srcRowLidHost);

  RowsFromTable<Node> rows;
  rows.srcRow = srcRowLid;
  ColumnsFromTable<Node> cols;
  cols.outCol = outColLid;
  return filterCopyCrs<Node>(*src, rowMap, colMap, domainMap, rowMap, rows, cols);
}

template<class Node>
std::vector<std::vector<MatrixRCP<Node> > >
extractBlocks(const MatrixRCP<Node> & J,
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
        "extractBlocks: map contract check failed for block (" << i << "," << j << ").");
    }
  }
  return remapped;
}

// A row that started non-empty and came back empty means the threshold was too aggressive.
template<class Node>
void requireNoEmptiedRow(const typename BlockTypes<Node>::CrsMatrix & src,
                         const typename BlockTypes<Node>::CrsMatrix & out,
                         const std::string & label) {
  using exec_space = typename Node::execution_space;
  const auto srcRows = src.getLocalMatrixDevice().graph.row_map;
  const auto outRows = out.getLocalMatrixDevice().graph.row_map;
  const LO nrows = static_cast<LO>(src.getRowMap()->getLocalNumElements());
  LO firstEmptied = nrows;
  Kokkos::parallel_reduce("requireNoEmptiedRow",
    Kokkos::RangePolicy<exec_space, LO>(0, nrows),
    KOKKOS_LAMBDA(const LO i, LO & lmin) {
      const bool hadEntries = (srcRows(i + 1) > srcRows(i));
      const bool isEmpty = (outRows(i + 1) == outRows(i));
      if (hadEntries && isEmpty && i < lmin) lmin = i;
    }, Kokkos::Min<LO>(firstEmptied));
  GO emptiedRow = (firstEmptied < nrows) ? src.getRowMap()->getGlobalElement(firstEmptied) : -1;
  GO worstEmptied = -1;
  Teuchos::reduceAll<int, GO>(*src.getComm(), Teuchos::REDUCE_MAX, 1, &emptiedRow, &worstEmptied);
  TEUCHOS_TEST_FOR_EXCEPTION(worstEmptied >= 0, std::runtime_error,
    label << " emptied row " << worstEmptied << ".");
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

  Teuchos::RCP<LA_Vector> lumped;
  if (useLumpedDiagonal) {
    lumped = Teuchos::rcp(new LA_Vector(mat->getRangeMap(), false));
    LA_Vector ones(mat->getDomainMap(), false);
    ones.putScalar(one);
    mat->apply(ones, *lumped);   // row sums
  }

  auto dView = diag.getLocalViewDevice(Tpetra::Access::ReadOnly);
  auto lView = useLumpedDiagonal ? lumped->getLocalViewDevice(Tpetra::Access::ReadOnly)
                                 : dView;
  auto iView = inv->getLocalViewDevice(Tpetra::Access::OverwriteAll);
  const size_t nrows = static_cast<size_t>(rowMap->getLocalNumElements());
  const bool wantLumped = useLumpedDiagonal;
  GO nDiag = 0, nLumped = 0, nMissing = 0;
  Kokkos::parallel_reduce("buildInverseDiagonal",
    Kokkos::RangePolicy<typename Node::execution_space, size_t>(0, nrows),
    KOKKOS_LAMBDA(const size_t i, GO & ad, GO & al, GO & am) {
      const ScalarT d = dView(i, 0);
      const ScalarT sum = wantLumped ? lView(i, 0) : zero;
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
  bool   verifyBlocks       = false;   // the block-system identities
  double tol                = 1.0e-14;
};

// 'verify' turns on every check, but individual keys can be used to over-ride it
inline FilterOpts readFilterOpts(const Teuchos::ParameterList & pl) {
  FilterOpts o;
  const bool all = pl.isParameter("verify") && pl.get<bool>("verify");
  o.verifyComplex = o.verifyKnConsistency = o.verifyBlocks = all;
  if (pl.isParameter("filter SM"))              o.filterSM            = pl.get<bool>("filter SM");
  if (pl.isParameter("filter threshold"))       o.tol                 = pl.get<double>("filter threshold");
  if (pl.isParameter("verify complex"))         o.verifyComplex       = pl.get<bool>("verify complex");
  if (pl.isParameter("verify Kn consistency"))  o.verifyKnConsistency = pl.get<bool>("verify Kn consistency");
  return o;
}

// The (rowGid, colGid) pairs the filter would drop. Only 'verify complex' wants them, so
// this host pass runs on request instead of riding in the kernel.
template<class Node>
std::vector<std::pair<GO,GO> >
droppedOffDiagonals(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src,
                    const Teuchos::RCP<const Tpetra::Vector<ScalarT,LO,GO,Node>> & rowDiag,
                    const Teuchos::RCP<const Tpetra::Vector<ScalarT,LO,GO,Node>> & colDiag,
                    const typename Teuchos::ScalarTraits<ScalarT>::magnitudeType tol) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  std::vector<std::pair<GO,GO> > dropped;
  auto rowDiagView = rowDiag->getLocalViewHost(Tpetra::Access::ReadOnly);
  auto colDiagView = colDiag->getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto rowMap = src->getRowMap();
  const auto colMap = src->getColMap();
  const size_t maxEnt = std::max<size_t>(1, src->getLocalMaxNumRowEntries());
  host_inds_t cols("flt_cols", maxEnt);
  host_vals_t vals("flt_vals", maxEnt);
  const LO n_rows = static_cast<LO>(rowMap->getLocalNumElements());
  for (LO lid = 0; lid < n_rows; ++lid) {
    size_t nent = src->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    src->getLocalRowCopy(lid, cols, vals, nent);
    const GO rowGid = rowMap->getGlobalElement(lid);
    const MagT aii = Teuchos::ScalarTraits<ScalarT>::magnitude(rowDiagView(lid, 0));
    for (size_t k = 0; k < nent; ++k) {
      const GO colGid = colMap->getGlobalElement(cols(k));
      if (colGid == rowGid) continue;
      const MagT ajj = Teuchos::ScalarTraits<ScalarT>::magnitude(colDiagView(cols(k), 0));
      if (Teuchos::ScalarTraits<ScalarT>::magnitude(vals(k)) <
          tol * std::sqrt(aii * ajj)) {
        dropped.emplace_back(rowGid, colGid);
      }
    }
  }
  return dropped;
}

template<class Node>
FilterResult<Node>
filterExplicitZeros(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src,
                    const typename Teuchos::ScalarTraits<ScalarT>::magnitudeType tol,
                    const bool captureDropped = false) {
  using LA_Vector = Tpetra::Vector<ScalarT,LO,GO,Node>;
  using LA_Import = Tpetra::Import<LO,GO,Node>;
  using device_type = typename Node::device_type;

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
  // Plain rank-1 copies so the functor's view types do not depend on Tpetra's access tags.
  Kokkos::View<ScalarT*, device_type> rowDiagVals("flt_row_diag", rowMap->getLocalNumElements());
  Kokkos::View<ScalarT*, device_type> colDiagVals("flt_col_diag", colMap->getLocalNumElements());
  Kokkos::deep_copy(rowDiagVals,
    Kokkos::subview(rowDiag->getLocalViewDevice(Tpetra::Access::ReadOnly), Kokkos::ALL(), 0));
  Kokkos::deep_copy(colDiagVals,
    Kokkos::subview(colDiag->getLocalViewDevice(Tpetra::Access::ReadOnly), Kokkos::ALL(), 0));

  DropSmallOffDiagonals<Node> drop;
  drop.rowDiag = rowDiagVals;
  drop.colDiag = colDiagVals;
  drop.rowMap = rowMap->getLocalMap();
  drop.colMap = colMap->getLocalMap();
  drop.tol = tol;

  FilterResult<Node> result;
  result.matrix = filterCopyCrs<Node>(*src, rowMap, colMap, src->getDomainMap(),
                                      src->getRangeMap(), KeepAllRows(), drop);
  requireNoEmptiedRow<Node>(*src, *result.matrix, "filterExplicitZeros: 'filter threshold'");
  result.nnzIn = src->getGlobalNumEntries();
  result.nnzOut = result.matrix->getGlobalNumEntries();
  if (captureDropped) {
    result.dropped = droppedOffDiagonals<Node>(src, rowDiag, colDiag, tol);
  }
  return result;
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

// detectBoundaryConditionsSM returns BCcols on D0's column map, not Kn's.
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
             const char * label, const size_t split, const int verbosity, const int rank) {
  prec->resetMatrix(wrapAsXpetraMatrix<Node>(SM));
  if (verbosity >= 10 && rank == 0) {
    std::cout << "[" << label << "] Reusing existing hierarchy with resetMatrix()"
              << " (split " << split << ")" << std::endl;
  }
  return Teuchos::rcp(new MueLu::TpetraOperator<ScalarT, LO, GO, Node>(
      Teuchos::rcp_static_cast<Xpetra::Operator<ScalarT, LO, GO, Node> >(prec)));
}
// RefMaxwell/Maxwell1 inputs, before and after the optional filter. SM is filtered
// even on reuse, because resetMatrix() must get the matrix the hierarchy was built
// from.
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
  
  in.smFilter = filterExplicitZeros<Node>(SM, opts.tol, opts.verifyComplex && !forReuse);
  in.SM = in.smFilter.matrix;
  return in;
}
// ReitzingerPFactory's sign kernel aborts on any D0 entry that is not exactly
// +1, -1 or 0, and Panzer emits +-0.5.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>>
snapCrsMatrixSigns(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src) {
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  SnapEntrySigns snap;
  snap.tol = Teuchos::ScalarTraits<MagT>::eps() * 1e2;
  return filterCopyCrs<Node>(*src, src->getRowMap(), src->getColMap(),
                             src->getDomainMap(), src->getRangeMap(),
                             KeepAllRows(), snap);
}

// Drop whole rows, so the result keeps D0's maps but loses those edges entirely.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>>
dropBCRows(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & src,
           const Kokkos::View<const bool*, typename Node::device_type::memory_space> & bcRows) {
  const LO n_rows = static_cast<LO>(src->getRowMap()->getLocalNumElements());
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<size_t>(bcRows.extent(0)) != static_cast<size_t>(n_rows),
    std::runtime_error, "dropBCRows: bcRows has " << bcRows.extent(0)
    << " entries; expected " << n_rows << ".");
  KeepUnflaggedRows<Node> keep;
  keep.flagged = bcRows;
  return filterCopyCrs<Node>(*src, src->getRowMap(), src->getColMap(),
                             src->getDomainMap(), src->getRangeMap(),
                             keep, KeepAllEntries());
}

} // namespace detail

inline std::string joinNames(const std::vector<std::string> & names) {
  std::string out;
  for (size_t v = 0; v < names.size(); ++v) {
    if (v) out += ", ";
    out += std::to_string(v) + "=" + names[v];
  }
  return out;
}

// Variable name to index. Throws listing declared names, since a misspelling and an
// absent variable are otherwise indistinguishable.
inline int variableIndexByName(const std::vector<std::string> & names,
                               const std::string & want, const std::string & key) {
  for (size_t v = 0; v < names.size(); ++v) {
    if (names[v] == want) return static_cast<int>(v);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Schur '" << key << ": " << want << "' names no variable in this set. Declared: "
    << joinNames(names) << ".");
  return -1;
}

// One split's comma-separated variable list to indices; 'seen' rejects repeats across splits.
inline std::vector<size_t>
splitVariableIndices(const std::string & split, const std::vector<std::string> & names,
                    std::vector<bool> & seen) {
  std::vector<size_t> group;
  for (const std::string & name : splitCommaList(split)) {
    const size_t v = static_cast<size_t>(variableIndexByName(names, name, "variable groups"));
    TEUCHOS_TEST_FOR_EXCEPTION(seen[v], std::runtime_error,
      "Schur 'variable groups' names '" << name << "' more than once.");
    seen[v] = true;
    group.push_back(v);
  }
  return group;
}

inline void requireEveryVariableGrouped(const std::vector<std::string> & names,
                                        const std::vector<bool> & seen) {
  for (size_t v = 0; v < names.size(); ++v) {
    TEUCHOS_TEST_FOR_EXCEPTION(!seen[v], std::runtime_error,
      "Schur 'variable groups' leaves '" << names[v]
      << "' out; every variable must appear exactly once. Declared: "
      << joinNames(names) << ".");
  }
}

// One entry per split, in the order 'variable groups' declares them.
inline std::vector<std::vector<size_t> >
resolveVariableGroups(const std::vector<std::string> & splits,
                      const std::vector<std::string> & names) {
  std::vector<std::vector<size_t> > groups;
  if (splits.empty()) return groups;
  std::vector<bool> seen(names.size(), false);
  for (size_t r = 0; r < splits.size(); ++r) {
    std::vector<size_t> group = splitVariableIndices(splits[r], names, seen);
    TEUCHOS_TEST_FOR_EXCEPTION(group.empty(), std::runtime_error,
      "Schur 'variable groups' split " << r << " names no variables.");
    groups.push_back(group);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(groups.size() < 2, std::runtime_error,
    "Schur 'variable groups' needs at least two splits, got " << groups.size() << ".");
  requireEveryVariableGrouped(names, seen);
  return groups;
}

// Split maps from named groups; a multi-variable group gets one fused, sorted map.
template<class Node>
std::vector<typename BlockTypes<Node>::MapRCP>
fuseSplitMaps(const std::vector<std::vector<size_t> > & groups,
             const std::vector<typename BlockTypes<Node>::MapRCP> & blockMaps) {
  using LA_Map = typename BlockTypes<Node>::Map;
  std::vector<typename BlockTypes<Node>::MapRCP> splitMaps;
  for (size_t r = 0; r < groups.size(); ++r) {
    if (groups[r].size() == 1) {
      splitMaps.push_back(blockMaps[groups[r][0]]);
      continue;
    }
    std::vector<GO> fused;
    for (size_t k = 0; k < groups[r].size(); ++k) {
      auto gids = blockMaps[groups[r][k]]->getLocalElementList();
      for (size_t g = 0; g < static_cast<size_t>(gids.size()); ++g) fused.push_back(gids[g]);
    }
    std::sort(fused.begin(), fused.end());
    splitMaps.push_back(Teuchos::rcp(new LA_Map(Teuchos::OrdinalTraits<GO>::invalid(), fused, 0,
                                               blockMaps[0]->getComm())));
  }
  return splitMaps;
}

// A fused split has no single variable index, so its mass matrix is found by map.
template<class Node>
typename BlockTypes<Node>::CrsMatrixRCP
massMatrixOnMap(const std::vector<typename BlockTypes<Node>::CrsMatrixRCP> & masses,
                const typename BlockTypes<Node>::MapRCP & map) {
  for (size_t m = 0; m < masses.size(); ++m) {
    if (!masses[m].is_null() && masses[m]->getRowMap()->isSameAs(*map)) return masses[m];
  }
  return Teuchos::null;
}

// Names resolve against declaration order, so naming a variable survives a dimension change
// that renumbers the indices.
template<class Node>
BlockSystem<Node> buildBlockSystemForSet(const typename BlockTypes<Node>::CrsMatrixRCP & J,
                                         const std::vector<typename BlockTypes<Node>::MapRCP> & blockMaps,
                                         const std::vector<std::string> & varNames,
                                         const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                                         const size_t set,
                                         const int verbosity) {
  using Types = BlockTypes<Node>;
  TEUCHOS_TEST_FOR_EXCEPTION(blockMaps.size() < 2, std::runtime_error,
    "Block-triangular preconditioner needs at least two variable blocks, but set "
    << set << " has " << blockMaps.size() << ".");

  const std::vector<std::vector<size_t> > groups =
    resolveVariableGroups(cntxt->splitVariableSpecs(), varNames);
  const size_t targetVariable = groups[cntxt->schur_target_index][0];
  const std::vector<typename Types::MapRCP> splitMaps = fuseSplitMaps<Node>(groups, blockMaps);

  const bool wantMass =
    parseSchurVariant(cntxt->schur.approximation_type) == SchurVariant::Mass;
  // getInvD only implements the diag correction, so mass would silently give S_k = J_kk.
  TEUCHOS_TEST_FOR_EXCEPTION(splitMaps.size() > 2 && wantMass, std::runtime_error,
    "Schur 'approximation type: mass' supports two splits, but this set has " << splitMaps.size()
    << "; use 'approximation type: diag'.");
  TEUCHOS_TEST_FOR_EXCEPTION(groups.back().size() != 1 && wantMass, std::runtime_error,
    "Schur 'approximation type: mass' needs one variable in the target split, but the last "
    "group in 'variable groups' holds " << groups.back().size() << ".");
  if (verbosity >= 10 && J->getComm()->getRank() == 0) {
    std::cout << "[BlockTri] " << blockMaps.size() << " variable blocks, "
              << splitMaps.size() << " splits (from variable groups); schur target '"
              << cntxt->splits[cntxt->schur_target_index].name << "'" << std::endl;
    if (!varNames.empty()) {
      std::cout << "[BlockTri] variables: " << joinNames(varNames) << "; target = "
                << (targetVariable < varNames.size() ? varNames[targetVariable] : "?") << std::endl;
    }
  }
  std::vector<std::vector<typename Types::CrsMatrixRCP> > remappedBlocks =
    detail::extractBlocks<Node>(J, splitMaps);

  // A split fusing several variables carries cross-variable explicit zeros that MueLu would
  // aggregate across.
  for (size_t r = 0; r < splitMaps.size(); ++r) {
    if (groups[r].size() < 2) continue;
    const detail::FilterResult<Node> filtered = detail::filterExplicitZeros<Node>(
      Teuchos::rcp_implicit_cast<const typename Types::CrsMatrix>(remappedBlocks[r][r]),
      1.0e-14);
    if (verbosity >= 10 && J->getComm()->getRank() == 0) {
      std::cout << "[BlockTri] fused split " << r << ": dropped "
                << (filtered.nnzIn - filtered.nnzOut) << " explicit zeros of "
                << filtered.nnzIn << std::endl;
    }
    remappedBlocks[r][r] = filtered.matrix;
  }

  BlockSystem<Node> blocks;
  blocks.maps = splitMaps;
  blocks.blocks = remappedBlocks;
  return blocks;
}

// Rebuild the split view from a blocked Thyra operator; split order and fusing are already
// baked in.
template<class Node>
BlockSystem<Node> blockSystemFromBlockedOp(const Teko::BlockedLinearOp & blo) {
  using Types = BlockTypes<Node>;
  const int nb = Teko::blockRowCount(blo);
  BlockSystem<Node> blocks;
  blocks.blocks.assign(static_cast<size_t>(nb),
                       std::vector<typename Types::CrsMatrixRCP>(static_cast<size_t>(nb),
                                                                 Teuchos::null));
  blocks.maps.resize(static_cast<size_t>(nb));
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < nb; ++j) {
      Teko::LinearOp op = Teko::getBlock(i, j, blo);
      if (op.is_null()) continue;
      typename Types::CrsMatrixRCP crs = Teuchos::rcp_const_cast<typename Types::CrsMatrix>(
        thyraToTpetraCrs<Node>(op));
      blocks.blocks[i][j] = crs;
      if (i == j) blocks.maps[i] = crs->getRowMap();
    }
    TEUCHOS_TEST_FOR_EXCEPTION(blocks.maps[i].is_null(), std::runtime_error,
      "blockSystemFromBlockedOp: diagonal block " << i << " is missing, so its map is unknown.");
  }
  return blocks;
}


} // namespace block_prec
} // namespace MrHyDE

#endif
