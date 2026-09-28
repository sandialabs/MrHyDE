/***********************************************************************
 MrHyDE - Jacobian block extraction and per-block preconditioner assembly.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_ASSEMBLY_HPP
#define MRHYDE_BLOCK_PREC_ASSEMBLY_HPP

#include "block_prec/InverseLibraryOps.hpp"
#include "block_prec/ParamUtils.hpp"
#include "linearAlgebraInterface.hpp"
#include "linearSolverContext.hpp"

#include <BelosLinearProblem.hpp>
#include <BelosTpetraOperator.hpp>
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

// Tools for preconditioner setup:
//   - block extraction and remap (BlockSystem, remapBlockToMaps)
//   - the shared diagonal/lumped inverse used by both the Schur and pivot paths
//   - the Maxwell auxiliary pipeline (SM/M1/D0 filtering, Kn, Dirichlet handling)
//   - per-block preconditioner dispatch (buildBlockOperator and friends)

// The extracted 2x2 system, labelled by role (not variable index as that may change):
//
//                 pivotMap  targetMap
//     pivotMap  [   J00       J01    ]     J00 is inverted directly,
//     targetMap [   J10       J11    ]     J11 is where the Schur complement forms.
//
//
//     pivot 0:  J00 = A(0,0)   J01 = A(0,1)      pivot 1:  J00 = A(1,1)   J01 = A(1,0)
//               J10 = A(1,0)   J11 = A(1,1)                J10 = A(0,1)   J11 = A(0,0)
//
// All blocks keep the GIDs they have in the monolithic Jacobian.
template<class Node>
struct BlockSystem {
  using Types = BlockTypes<Node>;
  using matrix_rcp = typename Types::CrsMatrixRCP;
  using map_rcp = typename Types::MapRCP;

  map_rcp pivotMap, targetMap;          // Owned row maps
  matrix_rcp J00, J01, J10, J11;
  int pivotBlock = 0;                   // Variable index for the pivot role.
  size_t targetBlock = 0;               // Variable index for the target role.
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
  std::vector<GO> keepCols;
  std::vector<ScalarT> keepVals;
  keepCols.reserve(maxEnt);
  keepVals.reserve(maxEnt);
  for (LO rowLid = 0; rowLid < nRows; ++rowLid) {
    const GO rowGid = rowMap->getGlobalElement(rowLid);
    const LO srcRowLid = srcRowMap->getLocalElement(rowGid);
    if (srcRowLid == Teuchos::OrdinalTraits<LO>::invalid()) continue;

    size_t nent = src->getNumEntriesInLocalRow(srcRowLid);
    if (nent == 0) continue;
    src->getLocalRowCopy(srcRowLid, colLids, colVals, nent);

    keepCols.clear();
    keepVals.clear();
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
  double tol                = 1.0e-14;
};

// 'verify' turns on every check, but individual keys can be used to over-ride it
inline FilterOpts readFilterOpts(const Teuchos::ParameterList & pl) {
  FilterOpts o;
  const bool all = pl.isParameter("verify") && pl.get<bool>("verify");
  o.verifyComplex = o.verifyKnConsistency = all;
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
  host_inds_t cols("flt_cols", maxEnt);
  host_vals_t vals("flt_vals", maxEnt);
  std::vector<GO> keepGids;
  std::vector<ScalarT> keepVals;
  keepGids.reserve(maxEnt);
  keepVals.reserve(maxEnt);
  const LO n_rows = static_cast<LO>(rowMap->getLocalNumElements());
  for (LO lid = 0; lid < n_rows; ++lid) {
    const GO rowGid = rowMap->getGlobalElement(lid);
    size_t nent = src->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    src->getLocalRowCopy(lid, cols, vals, nent);

    const MagT aii = Teuchos::ScalarTraits<ScalarT>::magnitude(rowDiagView(lid, 0));

    keepGids.clear();
    keepVals.clear();
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
             const char * label, const bool forSchur, const int verbosity, const int rank) {
  prec->resetMatrix(wrapAsXpetraMatrix<Node>(SM));
  if (verbosity >= 10 && rank == 0) {
    std::cout << "[" << label << "] Reusing existing hierarchy with resetMatrix()"
              << (forSchur ? " (Schur)" : "") << std::endl;
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
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;

  const MagT zero_tol = Teuchos::ScalarTraits<MagT>::eps() * 1e2;
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node>> rowMap = src->getRowMap();
  const Teuchos::RCP<const Tpetra::Map<LO,GO,Node>> colMap = src->getColMap();
  const size_t maxEnt = std::max<size_t>(1, src->getLocalMaxNumRowEntries());
  Teuchos::RCP<LA_CrsMatrix> out = Teuchos::rcp(new LA_CrsMatrix(rowMap, maxEnt));
  host_inds_t cols("snap_cols", maxEnt);
  host_vals_t vals("snap_vals", maxEnt);
  std::vector<GO> keepGids;
  std::vector<ScalarT> keepVals;
  keepGids.reserve(maxEnt);
  keepVals.reserve(maxEnt);
  const LO n_rows = static_cast<LO>(rowMap->getLocalNumElements());
  for (LO lid = 0; lid < n_rows; ++lid) {
    size_t nent = src->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    src->getLocalRowCopy(lid, cols, vals, nent);
    const GO rowGid = rowMap->getGlobalElement(lid);
    keepGids.clear();
    keepVals.clear();
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

// Drop whole rows, so the result keeps D0's maps but loses those edges entirely.
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
  host_inds_t cols("bcdrop_cols", maxEnt);
  host_vals_t vals("bcdrop_vals", maxEnt);
  std::vector<GO> keepGids;
  std::vector<ScalarT> keepVals;
  keepGids.reserve(maxEnt);
  keepVals.reserve(maxEnt);
  for (LO lid = 0; lid < n_rows; ++lid) {
    if (bcHost(lid)) continue;
    size_t nent = src->getNumEntriesInLocalRow(lid);
    if (nent == 0) continue;
    src->getLocalRowCopy(lid, cols, vals, nent);
    const GO rowGid = rowMap->getGlobalElement(lid);
    keepGids.clear();
    keepVals.clear();
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
    detail::extractBlocks<Node>(J, pairMaps);

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

// Applies diag(M)^-1. Thyra::diagonal would do the same through RTOps, which zero-fill
// Y and drop to a serial loop; elementWiseMultiply is one fused kernel.
template<class Node>
class DiagonalInverseOperator : public Tpetra::Operator<ScalarT,LO,GO,Node> {
public:
  using LA_Map = typename BlockTypes<Node>::Map;
  using LA_MultiVector = typename BlockTypes<Node>::MultiVector;
  using LA_Vector = typename BlockTypes<Node>::Vector;

  explicit DiagonalInverseOperator(const Teuchos::RCP<LA_Vector> & invDiag) : invDiag_(invDiag) {}

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return invDiag_->getMap(); }
  Teuchos::RCP<const LA_Map> getRangeMap() const override { return invDiag_->getMap(); }
  bool hasTransposeApply() const override { return true; }

  void apply(const LA_MultiVector & X, LA_MultiVector & Y,
             Teuchos::ETransp = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override {
    // Symmetric, so the transpose mode needs no special case.
    Y.elementWiseMultiply(alpha, *invDiag_, X, beta);
  }

private:
  Teuchos::RCP<LA_Vector> invDiag_;
};

template<class Node>
Teko::LinearOp
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
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > op =
    Teuchos::rcp(new DiagonalInverseOperator<Node>(invDiag));
  return tpetraToThyra<Node>(op, J00->getRangeMap(), J00->getDomainMap());
}

template<class Node>
Teko::LinearOp
buildDirectBlockInverse(LinearAlgebraInterface<Node> & interface,
                        const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                        const typename BlockTypes<Node>::CrsMatrixRCP & A,
                        const std::string & label) {
  const std::string & solverName = cntxt->amesos_type;
  Teuchos::ParameterList entry;
  entry.set("Solver Type", solverName);
  // Block row maps are extracted from the monolithic map, so they are not contiguous.
  entry.sublist("Amesos2 Settings").sublist(solverName).set("IsContiguous", false);
  return interface.inverseLibrary(cntxt).build("Amesos2", entry, label + " Amesos2", A);
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
    "'Flexible Gmres: true' and 'right preconditioner: true'; deck has '"
    << (cntxt.is_null() ? std::string("none") : cntxt->belos_type) << "'.");
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
  return interface.inverseLibrary(cntxt).build("Belos", belosList, label + " inner",
                                              blockMat, innerPrec);
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
      innerPrec = buildDirectBlockInverse<Node>(interface, cntxt, mat, label);
      break;
    case BlockPrecType::Diagonal:
      // S is formed from J00, so inverting its diagonal is not an approximation of it.
      TEUCHOS_TEST_FOR_EXCEPTION(forSchur, std::runtime_error,
        "Schur block does not support Diagonal.");
      innerPrec = buildDiagonalBlockInverse<Node>(mat, cntxt->schur.pivot_block_diag_use_lumped_diagonal,
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
      return interface.inverseLibrary(cntxt).build("MueLu", pivotMueLuParams,
                                                   "BlockTri pivot MueLu", J00);
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
      return interface.inverseLibrary(cntxt).build("MueLu", schurMueLuParams,
                                                   "BlockTri Schur MueLu", SchurApprox);
    });
}

} // namespace block_prec
} // namespace MrHyDE

#endif
