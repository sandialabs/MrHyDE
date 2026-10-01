/***********************************************************************
 MrHyDE - Checks on the block system and the Maxwell auxiliary operators.
 Everything here is off unless a 'verify' flag or verbosity >= 5 turns
 it on.

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_VERIFY_HPP
#define MRHYDE_BLOCK_PREC_VERIFY_HPP

#include "block_prec/BlockAssembly.hpp"

#include <MueLu_Maxwell_Utils.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>

#include <iomanip>
#include <utility>
#include <vector>

namespace MrHyDE {
namespace block_prec {


namespace detail {

// Require every nonzero (i,j) to have a matching (j,i).
template<class Node>
void assertStructuralSymmetry(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & A,
                              const std::string & label) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using host_inds_t = typename LA_CrsMatrix::nonconst_local_inds_host_view_type;
  using host_vals_t = typename LA_CrsMatrix::nonconst_values_host_view_type;
  Tpetra::RowMatrixTransposer<ScalarT,LO,GO,Node> transposer(Teuchos::rcp_const_cast<LA_CrsMatrix>(A));
  Teuchos::RCP<LA_CrsMatrix> At = transposer.createTranspose();
  const LO n = static_cast<LO>(A->getRowMap()->getLocalNumElements());
  const size_t maxEnt = std::max<size_t>(1, std::max(A->getLocalMaxNumRowEntries(),
                                                     At->getLocalMaxNumRowEntries()));
  host_inds_t colsA("sym_colsA", maxEnt), colsT("sym_colsT", maxEnt);
  host_vals_t valsA("sym_valsA", maxEnt), valsT("sym_valsT", maxEnt);
  const auto colMapA = A->getColMap();
  const auto colMapT = At->getColMap();
  for (LO lid = 0; lid < n; ++lid) {
    size_t nA = A->getNumEntriesInLocalRow(lid);
    size_t nT = At->getNumEntriesInLocalRow(lid);
    TEUCHOS_TEST_FOR_EXCEPTION(nA != nT, std::runtime_error,
      label << ": row " << A->getRowMap()->getGlobalElement(lid)
      << " has " << nA << " entries; its transpose has " << nT << ".");
    if (nA == 0) continue;
    A->getLocalRowCopy(lid, colsA, valsA, nA);
    At->getLocalRowCopy(lid, colsT, valsT, nT);
    std::set<GO> gA, gT;
    for (size_t k = 0; k < nA; ++k) gA.insert(colMapA->getGlobalElement(colsA(k)));
    for (size_t k = 0; k < nT; ++k) gT.insert(colMapT->getGlobalElement(colsT(k)));
    TEUCHOS_TEST_FOR_EXCEPTION(gA != gT, std::runtime_error,
      label << ": row " << A->getRowMap()->getGlobalElement(lid)
      << " has different columns in A and A^T.");
  }
}

// |(SM-SM_f)*D0*1|_inf <= tol*|SM|_F*|D0*1|_inf; DIRK mass makes SM*D0 nonzero.
template<class Node>
void assertKernelBound(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & SM_filtered,
                       const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & SM_orig,
                       const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & D0,
                       const typename Teuchos::ScalarTraits<ScalarT>::magnitudeType tol,
                       const std::string & label) {
  using LA_MultiVector = Tpetra::MultiVector<ScalarT,LO,GO,Node>;
  using MagT = typename Teuchos::ScalarTraits<ScalarT>::magnitudeType;
  // Probes with D0*1, which is zero on any edge whose two node columns both survive.
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

template<class Node>
FilterResult<Node> filterM1Checked(
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & M1,
    const FilterOpts & opts,
    const std::string & label) {
  FilterResult<Node> out = filterExplicitZeros<Node>(M1, opts.tol, opts.verifyComplex);
  assertStructuralSymmetry<Node>(out.matrix, label + " M1 filter");
  return out;
}

// De Rham sanity checks. D0 row structure throws; symmetry, the filtered (SM-M1)*D0v
// ratio and the Rayleigh quotient only warn.
template<class Node>
void verifyMaxwellComplex(
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & D0,
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

  // |(SM_f-M1_f)*D0v| / |(SM-M1)*D0v| in [0.5, 2] under filtering.
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

  // xTA_f x / xTAx in 1 +/- max(1e-12, nDrop*tol).
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


// Build-only: the kernel bound, the M1 filter, the complex checks.
template<class Node>
void finishMaxwellInputs(MaxwellInputs<Node> & in,
                         const FilterOpts & opts, const std::string & label,
                         const int verbosity, const int rank) {
  if (opts.filterSM) {
    assertKernelBound<Node>(in.SM, in.SM_orig, in.D0, opts.tol, label + " SM filter");
    in.m1Filter = filterM1Checked<Node>(in.M1_orig, opts, label);
    in.M1 = in.m1Filter.matrix;
    logFilterCounts<Node>(in.smFilter, in.m1Filter, opts, label, verbosity, rank);
  }
  if (opts.verifyComplex) {
    verifyMaxwellComplex<Node>(in.D0, in.SM_orig, in.M1_orig, in.SM, in.M1,
                               in.smFilter.dropped, in.m1Filter.dropped,
                               opts.tol, verbosity, rank, label);
  }
}

} // namespace detail

// On interior nodes curl*D0 = 0 forces Kn_SM = s * Kn_M1, so fit s and measure
// the rest. Boundary nodes differ only by Dirichlet treatment, so they are cut.
template<class Node>
void verifyKnConsistency(
    const Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > & Kn_M1,
    const Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > & SM_wrap,
    const Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > & D0_wrap,
    const Kokkos::View<bool*, typename Node::device_type::memory_space> & BCdomainNodal,
    const Teuchos::Comm<int> & comm,
    const int verbosity) {
  using LA_CrsMatrix = typename BlockTypes<Node>::CrsMatrix;
  using HostInds = typename BlockTypes<Node>::HostInds;
  using HostVals = typename BlockTypes<Node>::HostVals;

  Teuchos::ParameterList rapList;
  rapList.set("rap: fix zero diagonals", false);
  Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > Kn_SM =
    MueLu::Maxwell_Utils<ScalarT,LO,GO,Node>::PtAPWrapper(SM_wrap, D0_wrap, rapList, "Kn_from_SM");
  Teuchos::rcp_const_cast<Xpetra::CrsGraph<LO,GO,Node> >(Kn_SM->getCrsGraph())->computeGlobalConstants();

  // Hard casts: a silent skip makes the check report nothing while looking enabled.
  auto asTpetra = [](const Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > & K) {
    auto wrap = Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrixWrap<ScalarT,LO,GO,Node> >(K);
    TEUCHOS_TEST_FOR_EXCEPTION(wrap.is_null(), std::runtime_error,
      "verify Kn consistency: Kn is not an Xpetra::CrsMatrixWrap.");
    auto op = Teuchos::rcp_dynamic_cast<Xpetra::TpetraCrsMatrix<ScalarT,LO,GO,Node> >(wrap->getCrsMatrix());
    TEUCHOS_TEST_FOR_EXCEPTION(op.is_null(), std::runtime_error,
      "verify Kn consistency: Kn is not backed by an Xpetra::TpetraCrsMatrix.");
    return op->getTpetra_CrsMatrix();
  };
  Teuchos::RCP<const LA_CrsMatrix> knM1 = asTpetra(Kn_M1), knSM = asTpetra(Kn_SM);

  Teuchos::RCP<typename BlockTypes<Node>::Vector> diag =
    Teuchos::rcp(new typename BlockTypes<Node>::Vector(knM1->getRowMap(), true));
  knM1->getLocalDiagCopy(*diag);
  const double diagMax = diag->normInf();

  auto bcRow = Kokkos::create_mirror_view(BCdomainNodal);
  Kokkos::deep_copy(bcRow, BCdomainNodal);
  auto bcColDev = detail::knColumnMask<Node>(Kn_M1, BCdomainNodal);
  auto bcCol = Kokkos::create_mirror_view(bcColDev);
  Kokkos::deep_copy(bcCol, bcColDev);

  // Column maps need not match, and local order says nothing about global order.
  auto colMapM1 = knM1->getColMap();
  auto colMapSM = knSM->getColMap();
  const LO nrows = static_cast<LO>(knM1->getRowMap()->getLocalNumElements());
  const size_t maxEnt = std::max<size_t>(1, std::max(knM1->getLocalMaxNumRowEntries(),
                                                     knSM->getLocalMaxNumRowEntries()));
  HostInds cM("kn_cM", maxEnt), cS("kn_cS", maxEnt);
  HostVals vM("kn_vM", maxEnt), vS("kn_vS", maxEnt);
  std::vector<std::pair<double,double> > pairs;
  pairs.reserve(static_cast<size_t>(nrows) * maxEnt);
  for (LO r = 0; r < nrows; ++r) {
    if (bcRow(r)) continue;
    size_t nM = knM1->getNumEntriesInLocalRow(r), nS = knSM->getNumEntriesInLocalRow(r);
    if (nM == 0) continue;
    knM1->getLocalRowCopy(r, cM, vM, nM);
    if (nS > 0) knSM->getLocalRowCopy(r, cS, vS, nS);
    for (size_t k = 0; k < nM; ++k) {
      if (bcCol(cM(k))) continue;
      const LO lidSM = colMapSM->getLocalElement(colMapM1->getGlobalElement(cM(k)));
      double sVal = 0.0;
      for (size_t q = 0; q < nS; ++q) if (cS(q) == lidSM) { sVal = vS(q); break; }
      pairs.push_back(std::make_pair(static_cast<double>(vM(k)), sVal));
    }
  }

  double acc[3] = {0.0, 0.0, static_cast<double>(pairs.size())};
  for (size_t i = 0; i < pairs.size(); ++i) {
    acc[0] += pairs[i].first * pairs[i].second;
    acc[1] += pairs[i].first * pairs[i].first;
  }
  double accG[3];
  Teuchos::reduceAll<int,double>(comm, Teuchos::REDUCE_SUM, 3, acc, accG);
  const double sFit = (accG[1] > 0.0) ? (accG[0] / accG[1]) : 1.0;

  double local[2] = {0.0, diagMax};
  for (size_t i = 0; i < pairs.size(); ++i) {
    local[0] = std::max(local[0], std::abs(sFit * pairs[i].first - pairs[i].second));
  }
  double globalMax[2];
  Teuchos::reduceAll<int,double>(comm, Teuchos::REDUCE_MAX, 2, local, globalMax);
  const double scale = std::abs(sFit) * globalMax[1];
  if (comm.getRank() != 0) return;
  if (verbosity >= 6) {
    std::cout << "[Maxwell1 verify Kn] interior entries compared=" << static_cast<size_t>(accG[2])
              << " fitted Kn_SM/Kn_M1=" << sFit
              << " max|s*Kn_M1 - Kn_SM|=" << globalMax[0]
              << " scale=" << scale << std::endl;
  }
  if (scale > 0.0 && globalMax[0] > 1e-10 * scale) {
    std::cout << "[Maxwell1 verify Kn] WARN: residual " << globalMax[0]
              << " exceeds 1e-10 * " << scale << "; Kn_from_M1 is not a scalar "
              << "multiple of Kn_from_SM on the interior block." << std::endl;
  }
}

template<class Node>
void verifyBlockSystem(const BlockSystem<Node> & blocks,
                       const typename BlockTypes<Node>::CrsMatrixRCP & J,
                       const typename BlockTypes<Node>::CrsMatrixRCP & SchurApprox,
                       const typename BlockTypes<Node>::CrsMatrixRCP & D0,
                       const typename BlockTypes<Node>::CrsMatrixRCP & M1,
                       const Teuchos::RCP<const Tpetra::MultiVector<
                         typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node> > & coords,
                       const Teuchos::RCP<typename BlockTypes<Node>::MultiVector> & lumpedMass,
                       const ScalarT damping,
                       const bool useLumpedWeightDiagonal,
                       const bool schurIsDiag,
                       const bool requested,
                       const int verbosity) {
  if (!requested && verbosity < 5) return;
  // Every check below assumes the 2x2 split view.
  if (blocks.numBlocks() != 2) {
    return;
  }
  using Types = BlockTypes<Node>;
  using LA_Vector = typename Types::Vector;
  using LA_CrsMatrix = typename Types::CrsMatrix;
  using Import = Tpetra::Import<LO,GO,Node>;
  using Export = Tpetra::Export<LO,GO,Node>;
  const ScalarT one = Teuchos::ScalarTraits<ScalarT>::one();
  const ScalarT zero = Teuchos::ScalarTraits<ScalarT>::zero();
  const int rank = J->getRowMap()->getComm()->getRank();

  Teuchos::RCP<const typename Types::Map> fullMap = J->getRowMap();
  LA_Vector x(fullMap), Jx(fullMap), y(fullMap);
  detail::fillProbe<Node>(x);
  J->apply(x, Jx);

  Import impP(fullMap, blocks.maps[0]), impT(fullMap, blocks.maps[1]);
  Export expP(blocks.maps[0], fullMap), expT(blocks.maps[1], fullMap);
  LA_Vector x0(blocks.maps[0]), x1(blocks.maps[1]);
  LA_Vector y0(blocks.maps[0]), y1(blocks.maps[1]);
  x0.doImport(x, impP, Tpetra::REPLACE);
  x1.doImport(x, impT, Tpetra::REPLACE);

  blocks.blocks[0][0]->apply(x0, y0);
  blocks.blocks[0][1]->apply(x1, y0, Teuchos::NO_TRANS, one, one);
  blocks.blocks[1][1]->apply(x1, y1);
  blocks.blocks[1][0]->apply(x0, y1, Teuchos::NO_TRANS, one, one);
  y.putScalar(zero);
  y.doExport(y0, expP, Tpetra::REPLACE);
  y.doExport(y1, expT, Tpetra::REPLACE);
  y.update(-one, Jx, one);
  const auto nJx = Jx.norm2();
  const auto ndiff = y.norm2();
  if (rank == 0) {
    std::cout << "[BLOCK-VERIFY] round-trip rel = "
              << (nJx > 0 ? ndiff / nJx : ndiff) << std::endl;
  }

  // curl(grad) = 0, so the off-diagonal coupling annihilates range(D0).
  typename Types::CrsMatrixRCP curlBlock;
  if (!D0.is_null()) {
    if (D0->getRangeMap()->isSameAs(*blocks.blocks[1][0]->getDomainMap())) {
      curlBlock = blocks.blocks[1][0];
    }
    else if (D0->getRangeMap()->isSameAs(*blocks.blocks[0][1]->getDomainMap())) {
      curlBlock = blocks.blocks[0][1];
    }
  }
  if (!curlBlock.is_null()) {
    LA_Vector v(D0->getDomainMap()), D0v(D0->getRangeMap()), c(curlBlock->getRangeMap());
    detail::fillProbe<Node>(v);
    D0->apply(v, D0v);
    curlBlock->apply(D0v, c);
    // Scale by |J10|_F too, otherwise this tracks element-size spread.
    const auto nD0v = D0v.norm2() * curlBlock->getFrobeniusNorm();
    const auto nc = c.norm2();
    if (rank == 0) {
      std::cout << "[BLOCK-VERIFY] J10*D0 rel = "
                << (nD0v > 0 ? nc / nD0v : nc) << std::endl;
    }
  }

  // g'*M1*g matches sum(m_n) for Panzer's D0; a +-1 D0 gives 4x that.
  if (!D0.is_null() && !M1.is_null() && !coords.is_null() && !lumpedMass.is_null() &&
      M1->getRowMap()->isSameAs(*D0->getRangeMap()) &&
      coords->getMap()->isSameAs(*D0->getDomainMap())) {
    LA_Vector xd(D0->getDomainMap()), g(D0->getRangeMap()), M1g(D0->getRangeMap());
    for (size_t d = 0; d < coords->getNumVectors(); ++d) {
      auto cv = coords->getVector(d)->getLocalViewHost(Tpetra::Access::ReadOnly);
      {
        auto xv = xd.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t i = 0; i < xv.extent(0); ++i) xv(i, 0) = static_cast<ScalarT>(cv(i, 0));
      }
      D0->apply(xd, g);
      M1->apply(g, M1g);
      const auto q = g.dot(M1g);
      const auto vol = lumpedMass->getVector(0)->norm1();
      const auto rel = (vol > 0) ? std::abs(q - vol) / vol : std::abs(q - vol);
      if (rank == 0) {
        std::cout << "[BLOCK-VERIFY] D0-scale rel = "
                  << std::setprecision(14) << rel << std::setprecision(6) << std::endl;
      }
    }
  }

  if (schurIsDiag && !SchurApprox.is_null()) {
    detail::InverseDiagonalCounts w;
    Teuchos::RCP<LA_Vector> dinv =
      detail::buildInverseDiagonal<Node>(
        Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(blocks.blocks[0][0]),
        useLumpedWeightDiagonal, w);
    LA_Vector t(blocks.maps[0]), Sx(blocks.maps[1]), mf(blocks.maps[1]);
    LA_Vector dt(blocks.maps[0]);
    blocks.blocks[0][1]->apply(x1, t);
    dt.elementWiseMultiply(one, *dinv, t, zero);
    blocks.blocks[1][0]->apply(dt, mf);
    blocks.blocks[1][1]->apply(x1, Sx);
    mf.update(one, Sx, -damping);
    SchurApprox->apply(x1, Sx);
    Sx.update(-one, mf, one);
    const auto nmf = mf.norm2();
    const auto nSx = Sx.norm2();
    if (rank == 0) {
      std::cout << "[BLOCK-VERIFY] schur rel = "
                << (nmf > 0 ? nSx / nmf : nSx) << std::endl;
    }
  }
}

} // namespace block_prec
} // namespace MrHyDE

#endif
