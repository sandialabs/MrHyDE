/***********************************************************************
 MrHyDE - RefMaxwell, Maxwell1 and the Hiptmair auxiliary data.

 Nomenclature, with n nodes and e edges:
   SM  = HCURL system block, M/dt + curl(1/mu) curl        e x e
   M1  = HCURL edge mass matrix                            e x e
   D0  = nodal-to-edge gradient, as Panzer emits it        e x n, +-0.5
   Kn  = nodal auxiliary matrix D0^T M1 D0                 n x n
   m_n = lumped nodal mass, integral(N_n)                  n

 Questions? Contact Alexey Voronin (abvoron@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_MAXWELL_INVERSE_HPP
#define MRHYDE_BLOCK_PREC_MAXWELL_INVERSE_HPP

#include "block_prec/BlockAssembly.hpp"
#include "block_prec/BlockVerify.hpp"
#include "block_prec/ParamUtils.hpp"
#include "linearSolverContext.hpp"

#include <MueLu_Maxwell_Utils.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MatrixUtils.hpp>
#include <Xpetra_TripleMatrixMultiply.hpp>

namespace MrHyDE {
namespace block_prec {
namespace detail {

template<class Node>
struct RefMaxwellXpetraInputs {
  using CoordScalarT = typename Teuchos::ScalarTraits<ScalarT>::coordinateType;
  using XpetraMatrix = Xpetra::Matrix<ScalarT, LO, GO, Node>;
  using XpetraCoordMV = Xpetra::TpetraMultiVector<CoordScalarT, LO, GO, Node>;
  Teuchos::RCP<XpetraMatrix> SM_wrap;
  Teuchos::RCP<XpetraMatrix> D0_wrap;
  Teuchos::RCP<XpetraMatrix> M1_wrap;
  Teuchos::RCP<XpetraCoordMV> coords_xpetra;
};

template<class Node>
RefMaxwellXpetraInputs<Node> buildRefMaxwellXpetraInputs(
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT, LO, GO, Node> > & J,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT, LO, GO, Node> > & D0,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT, LO, GO, Node> > & M1,
    const Teuchos::RCP<const Tpetra::MultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType, LO, GO, Node> > & nodal_coords) {
  using CoordScalarT = typename Teuchos::ScalarTraits<ScalarT>::coordinateType;
  using TpetraCoordMV = Tpetra::MultiVector<CoordScalarT, LO, GO, Node>;
  RefMaxwellXpetraInputs<Node> out;
  out.SM_wrap = wrapAsXpetraMatrix<Node>(J);
  out.D0_wrap = wrapAsXpetraMatrix<Node>(D0);
  out.M1_wrap = wrapAsXpetraMatrix<Node>(M1);
  out.coords_xpetra = Teuchos::rcp(new Xpetra::TpetraMultiVector<CoordScalarT, LO, GO, Node>(
      Teuchos::rcp_const_cast<TpetraCoordMV>(nodal_coords)));
  return out;
}

// BC diagonals keep their original value: the nodal diagonal is O(h), so 1.0 is an outlier.
template<class Node>
void applyDirichletBCsToKn(
    Teuchos::RCP<Xpetra::Matrix<ScalarT, LO, GO, Node> > & Kn,
    const Kokkos::View<bool*, typename Node::device_type::memory_space> & BCdomainNodal,
    const int verbosity) {
  using dev_mem_space = typename Node::device_type::memory_space;
  using XpetraVector = Xpetra::Vector<ScalarT, LO, GO, Node>;
  Teuchos::RCP<XpetraVector> saved = Xpetra::VectorFactory<ScalarT, LO, GO, Node>::Build(
      Kn->getRowMap(), true);
  Kn->getLocalDiagCopy(*saved);
  auto savedView = saved->getLocalViewDevice(Tpetra::Access::ReadOnly);

  auto knColMap = Kn->getColMap();
  auto knRowMap = Kn->getRowMap();
  Kokkos::View<bool*, dev_mem_space> BCcols = knColumnMask<Node>(Kn, BCdomainNodal);
  Kokkos::View<const bool*, dev_mem_space> BCdomain_c = BCdomainNodal;
  Kokkos::View<const bool*, dev_mem_space> BCcols_c = BCcols;
  MueLu::UtilitiesBase<ScalarT, LO, GO, Node>::ZeroDirichletRows(Kn, BCdomain_c);
  MueLu::UtilitiesBase<ScalarT, LO, GO, Node>::ZeroDirichletCols(Kn, BCcols_c);

  // Values-only edit, so no resumeFill/fillComplete cycle.
  auto lclKn = Kn->getLocalMatrixDevice();
  auto lclColMap = knColMap->getLocalMap();
  auto lclRowMap = knRowMap->getLocalMap();
  Kokkos::parallel_for("restore_kn_bc_diag",
      Kokkos::RangePolicy<typename Node::execution_space>(0, lclKn.numRows()),
      KOKKOS_LAMBDA(const LO r) {
        if (!BCdomain_c(r)) return;
        const GO rowGid = lclRowMap.getGlobalElement(r);
        const LO rowLidInColMap = lclColMap.getLocalElement(rowGid);
        auto row = lclKn.row(r);
        for (LO j = 0; j < row.length; ++j) {
          if (row.colidx(j) == rowLidInColMap) {
            row.value(j) = savedView(r, 0);
            break;
          }
        }
      });
  repairNodalDiagonal<Node>(Kn, verbosity);
}

// A_n = D0^T * A_edge * D0, wrapped as Xpetra for MueLu 'user data.NodeMatrix'.
template<class Node>
Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> >
buildAuxNodalMatrix(const typename BlockTypes<Node>::CrsMatrixRCP & A_edge,
                    const typename BlockTypes<Node>::CrsMatrixRCP & D0) {
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
void addHiptmairUserData(Teuchos::ParameterList & mueluList,
                         const typename BlockTypes<Node>::CrsMatrixRCP & mat,
                         const typename BlockTypes<Node>::CrsMatrixRCP & D0_matrix,
                         const std::string & settingsName,
                         const int verbosity) {
  if (!mueluParamsWantHiptmair(mueluList)) return;
  TEUCHOS_TEST_FOR_EXCEPTION(D0_matrix.is_null(), std::runtime_error,
    "MueLu params request HIPTMAIR smoothing but no D0 (discrete gradient) matrix was "
    "supplied. Set 'hgrad basis name' and 'hcurl basis name' in the corresponding "
    << settingsName << " so setupBlockTriangularAuxiliary can build D0.");
  TEUCHOS_TEST_FOR_EXCEPTION(!D0_matrix->getRangeMap()->isSameAs(*mat->getRowMap()), std::runtime_error,
    "HIPTMAIR setup: D0 range map does not match the matrix row map "
    "(D0 range=" << D0_matrix->getRangeMap()->getGlobalNumElements()
    << ", rows=" << mat->getGlobalNumRows() << ").");
  // A Dirichlet row of mat is an identity row. Left in D0, those edges let the
  // nodal correction D0*dphi write into constrained DOFs.
  using dev_mem_space = typename Node::device_type::memory_space;
  Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node>> mat_x = wrapAsXpetraMatrix<Node>(mat);
  Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node>> D0_raw = wrapAsXpetraMatrix<Node>(D0_matrix);
  Kokkos::View<bool*, dev_mem_space> BCrows, BCcols, BCdomain;
  bool allEdgesBnd = false, allNodesBnd = false;
  int BCedges = 0, BCnodes = 0;
  MueLu::Maxwell_Utils<ScalarT, LO, GO, Node>::detectBoundaryConditionsSM(
      mat_x, D0_raw, /*rowSumTol=*/ -1.0,
      BCrows, BCcols, BCdomain, BCedges, BCnodes, allEdgesBnd, allNodesBnd);
  Kokkos::View<const bool*, dev_mem_space> BCrows_const = BCrows;
  const typename BlockTypes<Node>::CrsMatrixRCP D0 = dropBCRows<Node>(D0_matrix, BCrows_const);

  Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> > Kn = buildAuxNodalMatrix<Node>(mat, D0);
  repairNodalDiagonal<Node>(Kn, verbosity);
  mueluList.sublist("user data").set("NodeMatrix", Kn);
  mueluList.sublist("user data").set("D0", wrapAsXpetraMatrix<Node>(D0));
}

} // namespace detail

// Check D0, M1, nodal_coords and map compatibility for a RefMaxwell/Maxwell1 split.
template<class Node>
void validateRefMaxwellBlockInputs(const typename BlockTypes<Node>::CrsMatrixRCP & J00,
                                   const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                                   const int verbosity) {
  // Needs range(D0) = row/domain(J00) = row/domain(M1), nodal coords on domain(D0).
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
  if (verbosity >= 10 && J00->getComm()->getRank() == 0) {
    std::cout << "[RefMaxwell validation] J00 rows="
              << J00->getGlobalNumRows()
              << " D0 range=" << cntxt->refMaxwell.D0_matrix->getRangeMap()->getGlobalNumElements()
              << " D0 domain=" << cntxt->refMaxwell.D0_matrix->getDomainMap()->getGlobalNumElements()
              << " coords length=" << cntxt->refMaxwell.nodal_coords->getGlobalLength()
              << std::endl;
  }
}

template<class Node>
Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> >
buildRefMaxwellPreconditioner(const typename BlockTypes<Node>::CrsMatrixRCP & J,
                              const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                              const size_t split,
                              const int verbosity) {
  using matrix_RCP = typename BlockTypes<Node>::CrsMatrixRCP;
  using RefMaxwellType = MueLu::RefMaxwell<ScalarT, LO, GO, Node>;
  FieldSplit<Node> & splitState = cntxt->splits[split];
  Teuchos::RCP<RefMaxwellType> & precCache = splitState.refmaxwell_prec;
  // Only the Schur target forms a Schur complement, so only it can carry the addon.
  const bool isTarget = (split == cntxt->schur_target_index);

  using XpetraMatrix = Xpetra::Matrix<ScalarT, LO, GO, Node>;
  using XpetraOperator = Xpetra::Operator<ScalarT, LO, GO, Node>;

  // D0/M1/coords and their maps are checked by validateRefMaxwellBlockInputs.
  const int rank = J->getComm()->getRank();
  const GO D0_global_rows = cntxt->refMaxwell.D0_matrix->getGlobalNumRows();
  const GO D0_global_cols = cntxt->refMaxwell.D0_matrix->getGlobalNumCols();
  matrix_RCP M1_use = cntxt->refMaxwell.M1_matrix;
  const Teuchos::ParameterList rmSettings = splitState.settings.isSublist("RefMaxwell Settings")
    ? splitState.settings.sublist("RefMaxwell Settings") : Teuchos::ParameterList();

  // Filtering and operator checks are disabled by default.
  const block_prec::detail::FilterOpts filterOpts = block_prec::detail::readFilterOpts(rmSettings);

  // MueLu bakes beta into the hierarchy, and only the Schur one has an addon.
  const ScalarT betaTarget = (isTarget && cntxt->schurAddonWanted(*J->getComm()))
    ? cntxt->refMaxwell.schur_addon_beta : 0.0;
  const ScalarT betaWas = splitState.addon_beta_built;
  // Relative, so round-off in the beta probe does not discard the hierarchy.
  if (std::abs(betaTarget - betaWas) >
      1.0e-6 * std::max(std::abs(betaTarget), std::abs(betaWas))) {
    precCache = Teuchos::null;
  }
  const bool canReuse = !precCache.is_null() &&
    block_prec::reuseKeepsHierarchy(cntxt->preconditioner_reuse_type);

  block_prec::detail::MaxwellInputs<Node> in = block_prec::detail::filterSMOnly<Node>(
    J, M1_use, cntxt->refMaxwell.D0_matrix, filterOpts, canReuse);

  // Everything past here feeds the hierarchy build, which reuse skips.
  if (canReuse) {
    return block_prec::detail::resetAndWrap<Node>(precCache, in.SM, "RefMaxwell",
                                                  split, verbosity, rank);
  }
  block_prec::detail::finishMaxwellInputs<Node>(
    in, filterOpts, "RefMaxwell", verbosity, rank);
  matrix_RCP SM_for_setup = in.SM, M1_for_setup = in.M1;

  block_prec::detail::RefMaxwellXpetraInputs<Node> xpetraInputs = block_prec::detail::buildRefMaxwellXpetraInputs<Node>(
    SM_for_setup, cntxt->refMaxwell.D0_matrix, M1_for_setup, cntxt->refMaxwell.nodal_coords);
  Teuchos::RCP<XpetraMatrix> SM_wrap = xpetraInputs.SM_wrap;
  Teuchos::RCP<XpetraMatrix> D0_wrap = xpetraInputs.D0_wrap;
  Teuchos::RCP<XpetraMatrix> M1_wrap = xpetraInputs.M1_wrap;
  auto coords_xpetra = xpetraInputs.coords_xpetra;

  if (verbosity >= 10 && rank == 0) {
    std::cout << "[RefMaxwell preflight] A rows=" << J->getGlobalNumRows()
              << " cols=" << J->getGlobalNumCols()
              << " localRows=" << J->getLocalNumRows() << std::endl;
    std::cout << "[RefMaxwell preflight] D0 rows=" << D0_global_rows
              << " cols=" << D0_global_cols
              << " localRows=" << cntxt->refMaxwell.D0_matrix->getLocalNumRows()
              << " localMaxRowNnz=" << cntxt->refMaxwell.D0_matrix->getLocalMaxNumRowEntries() << std::endl;
    std::cout << "[RefMaxwell preflight] M1 rows=" << M1_use->getGlobalNumRows()
              << " cols=" << M1_use->getGlobalNumCols()
              << " localRows=" << M1_use->getLocalNumRows() << std::endl;
    std::cout << "[RefMaxwell preflight] coords globalLength=" << cntxt->refMaxwell.nodal_coords->getGlobalLength()
              << " localLength=" << cntxt->refMaxwell.nodal_coords->getLocalLength()
              << " numVecs=" << cntxt->refMaxwell.nodal_coords->getNumVectors() << std::endl;
  }

  const std::string refmaxwellXmlFile =
    LinearSolverContext<Node>::splitXmlFile(splitState.settings, "RefMaxwell Settings");
  TEUCHOS_TEST_FOR_EXCEPTION(refmaxwellXmlFile.empty(), std::runtime_error,
    "RefMaxwell requires 'xml param file' in the '" << splitState.name
    << "' split's 'RefMaxwell Settings'.");

  // Copy: the list is cached on the context and everything below mutates it.
  Teuchos::ParameterList refmaxwellParams = cntxt->refMaxwellParams(split, *J->getComm());
  if (verbosity >= 6 && J->getComm()->getRank() == 0) {
    std::cout << "[RefMaxwell] Loaded parameters from XML file: " << refmaxwellXmlFile << std::endl;
  }

  block_prec::warnNonStationarySmoother(refmaxwellParams, "refmaxwell: 11list", J->getComm());
  block_prec::warnNonStationarySmoother(refmaxwellParams, "refmaxwell: 22list", J->getComm());

  TEUCHOS_TEST_FOR_EXCEPTION(refmaxwellParams.get<int>("refmaxwell: space number", 1) != 1,
    std::runtime_error, "RefMaxwell XML '" << refmaxwellXmlFile
    << "' sets 'refmaxwell: space number' to something other than 1, but this path "
    "supplies an edge D0 and nodal coordinates.");
  const bool wantAddon = block_prec::refMaxwellAddonEnabled(refmaxwellParams);
  TEUCHOS_TEST_FOR_EXCEPTION(wantAddon && !isTarget, std::runtime_error,
    "RefMaxwell XML '" << refmaxwellXmlFile << "' enables the addon on split '"
    << splitState.name << "', which is not the Schur target. "
    "beta is read off the Schur correction and has no pivot-block counterpart.");
  const bool haveAddon = wantAddon && cntxt->refMaxwell.schur_addon_beta > 0.0 &&
                         !cntxt->refMaxwell.nodal_lumped_mass.is_null();
  TEUCHOS_TEST_FOR_EXCEPTION(wantAddon && cntxt->refMaxwell.nodal_lumped_mass.is_null(),
    std::runtime_error,
    "RefMaxwell XML '" << refmaxwellXmlFile << "' enables the addon, but the lumped "
    "nodal mass was not built.");
  if (!haveAddon) {
    if (wantAddon && J->getComm()->getRank() == 0) {
      std::cout << "[RefMaxwell] addon requested but beta is unavailable; running without it."
                << std::endl;
    }
    refmaxwellParams.set("refmaxwell: disable addon", true);
  }
  // resetMatrix is a no-op unless the hierarchy was built with reuse enabled.
  // MueLu then defaults the sublists to "full", which freezes their smoothers.
  if (cntxt->preconditioner_reuse_type != "none") {
    refmaxwellParams.set("refmaxwell: enable reuse", true);
    for (const char * sub : {"refmaxwell: 11list", "refmaxwell: 22list"}) {
      Teuchos::ParameterList & pl = refmaxwellParams.sublist(sub);
      if (!pl.isParameter("reuse: type")) pl.set("reuse: type", "RP");
    }
  }

  if (verbosity >= 10 && J->getComm()->getRank() == 0) {
    std::cout << "[RefMaxwell] Final parameter list:" << std::endl;
    refmaxwellParams.print(std::cout, 2, true);
  }

  // The no-addon overload is this same call with Ms = M1 and a null M0inv.
  Teuchos::RCP<XpetraMatrix> M0inv_wrap;
  if (haveAddon) {
    M0inv_wrap = block_prec::detail::buildRefMaxwellM0inv<Node>(
      cntxt->refMaxwell.nodal_lumped_mass, cntxt->refMaxwell.schur_addon_beta);
  }
  // Null nullspace: MueLu forms D0*coords itself, which is what we would pass.
  precCache = Teuchos::rcp(new RefMaxwellType(
      SM_wrap, D0_wrap, M1_wrap, M0inv_wrap, M1_wrap,
      Teuchos::null, coords_xpetra,
      refmaxwellParams, true));
  splitState.addon_beta_built = haveAddon ? cntxt->refMaxwell.schur_addon_beta : 0.0;
  if (verbosity >= 10 && J->getComm()->getRank() == 0) {
    std::cout << "[RefMaxwell] Built new preconditioner hierarchy (split " << split << ")"
              << std::endl;
  }

  return Teuchos::rcp(new MueLu::TpetraOperator<ScalarT, LO, GO, Node>(
      Teuchos::rcp_static_cast<XpetraOperator>(precCache)));
}

template<class Node>
Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> >
buildMaxwell1Preconditioner(const typename BlockTypes<Node>::CrsMatrixRCP & J,
                            const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                            const size_t split,
                            const int verbosity) {
  using matrix_RCP = typename BlockTypes<Node>::CrsMatrixRCP;
  using Maxwell1Type = MueLu::Maxwell1<ScalarT, LO, GO, Node>;
  FieldSplit<Node> & splitState = cntxt->splits[split];
  Teuchos::RCP<Maxwell1Type> & precCache = splitState.maxwell1_prec;

  using XpetraMatrix = Xpetra::Matrix<ScalarT, LO, GO, Node>;
  using XpetraOperator = Xpetra::Operator<ScalarT, LO, GO, Node>;

  // D0/M1/coords are checked by validateRefMaxwellBlockInputs.
  matrix_RCP M1_use = cntxt->refMaxwell.M1_matrix;

  const int rank = J->getComm()->getRank();
  // ReitzingerP requires +-1; Panzer's D0 is +-0.5.
  if (cntxt->maxwell1.D0_normalized.is_null()) {
    cntxt->maxwell1.D0_normalized = block_prec::detail::snapCrsMatrixSigns<Node>(
      cntxt->refMaxwell.D0_matrix);
  }

  // Filtering and operator checks are disabled by default.
  const Teuchos::ParameterList m1Settings = splitState.settings.isSublist("Maxwell1 Settings")
    ? splitState.settings.sublist("Maxwell1 Settings") : Teuchos::ParameterList();
  const block_prec::detail::FilterOpts filterOpts = block_prec::detail::readFilterOpts(m1Settings);
  const bool canReuse = !precCache.is_null() &&
    block_prec::reuseKeepsHierarchy(cntxt->preconditioner_reuse_type);

  block_prec::detail::MaxwellInputs<Node> in = block_prec::detail::filterSMOnly<Node>(
    J, M1_use, cntxt->maxwell1.D0_normalized, filterOpts, canReuse);

  // Everything past here feeds the hierarchy build, which reuse skips.
  if (canReuse) {
    return block_prec::detail::resetAndWrap<Node>(precCache, in.SM, "Maxwell1",
                                                  split, verbosity, rank);
  }
  block_prec::detail::finishMaxwellInputs<Node>(
    in, filterOpts, "Maxwell1", verbosity, rank);
  matrix_RCP SM_for_setup = in.SM, M1_for_setup = in.M1;

  block_prec::detail::RefMaxwellXpetraInputs<Node> xpetraInputs = block_prec::detail::buildRefMaxwellXpetraInputs<Node>(
    SM_for_setup, cntxt->maxwell1.D0_normalized, M1_for_setup, cntxt->refMaxwell.nodal_coords);
  Teuchos::RCP<XpetraMatrix> SM_wrap = xpetraInputs.SM_wrap;
  Teuchos::RCP<XpetraMatrix> D0_wrap = xpetraInputs.D0_wrap;
  auto coords_xpetra = xpetraInputs.coords_xpetra;

  const std::string maxwell1XmlFile =
    LinearSolverContext<Node>::splitXmlFile(splitState.settings, "Maxwell1 Settings");
  TEUCHOS_TEST_FOR_EXCEPTION(maxwell1XmlFile.empty(), std::runtime_error,
    "Maxwell1 requires 'xml param file' in the '" << splitState.name
    << "' split's 'Maxwell1 Settings'.");
  // Copy: the list is cached on the context and everything below mutates it.
  Teuchos::ParameterList maxwell1Params = cntxt->maxwell1Params(split, *J->getComm());
  if (verbosity >= 6 && rank == 0) {
    std::cout << "[Maxwell1] Loaded parameters from XML file: " << maxwell1XmlFile << std::endl;
  }

  // Use M1 because PEC identity rows in SM create O(1)/O(h) diagonal contrast
  // that breaks aggregation in D0^T SM D0.
  Teuchos::RCP<XpetraMatrix> Kn_from_M1;
  const bool knXmlSet = maxwell1Params.isParameter("maxwell1: use Kn from M1");
  const bool knFromXml = knXmlSet && maxwell1Params.get<bool>("maxwell1: use Kn from M1");
  maxwell1Params.remove("maxwell1: use Kn from M1", false);
  const bool knInYaml = m1Settings.isParameter("use Kn from M1");
  
  bool useKnFromM1 = true;
  if (knInYaml)      useKnFromM1 = m1Settings.get<bool>("use Kn from M1");
  else if (knXmlSet) useKnFromM1 = knFromXml;
  if (rank == 0) {
    if (knXmlSet && knInYaml && knFromXml != useKnFromM1) {
      std::cout << "WARNING: 'use Kn from M1' is " << (useKnFromM1 ? "true" : "false")
                << " in Maxwell1 Settings and " << (knFromXml ? "true" : "false")
                << " in " << maxwell1XmlFile << ". The YAML wins." << std::endl;
    }
    else if (knXmlSet && !knInYaml && verbosity >= 1) {
      std::cout << "WARNING: 'maxwell1: use Kn from M1' is a MrHyDE key, not a MueLu one. "
                << "Set 'use Kn from M1' in Maxwell1 Settings instead." << std::endl;
    }
  }
  if (useKnFromM1) {
    Teuchos::ParameterList rapList;
    rapList.set("rap: fix zero diagonals", false);
    Kn_from_M1 = MueLu::Maxwell_Utils<ScalarT, LO, GO, Node>::PtAPWrapper(
        xpetraInputs.M1_wrap, D0_wrap, rapList, "Kn_from_M1");

    using dev_mem_space = typename Node::device_type::memory_space;
    Kokkos::View<bool*, dev_mem_space> BCrowsK, BCcolsK_d0, BCdomainK;
    bool allEdgesBnd = false, allNodesBnd = false;
    int BCedges = 0, BCnodes = 0;
    MueLu::Maxwell_Utils<ScalarT, LO, GO, Node>::detectBoundaryConditionsSM(
        SM_wrap, D0_wrap, /*rowSumTol=*/ -1.0,
        BCrowsK, BCcolsK_d0, BCdomainK,
        BCedges, BCnodes, allEdgesBnd, allNodesBnd);
    if (verbosity >= 10 && rank == 0) {
      std::cout << "[Maxwell1] Kn from M1: detected " << BCedges << " BC edges, "
                << BCnodes << " BC nodes" << std::endl;
    }

    if (BCnodes > 0) {
      block_prec::detail::applyDirichletBCsToKn<Node>(Kn_from_M1, BCdomainK, verbosity);
    }
    // MueLu hierarchy statistics require the global graph constants.
    Teuchos::rcp_const_cast<Xpetra::CrsGraph<LO, GO, Node> >(
        Kn_from_M1->getCrsGraph())->computeGlobalConstants();

    // Already pinned above. MueLu would overwrite it with a fixed 1.0.
    maxwell1Params.set("rap: fix zero diagonals", false);

    if (filterOpts.verifyKnConsistency) {
      block_prec::verifyKnConsistency<Node>(Kn_from_M1, SM_wrap, D0_wrap, BCdomainK,
                                            *J->getComm(), verbosity);
    }
  }

  precCache = Teuchos::rcp(new Maxwell1Type(
      SM_wrap, D0_wrap, Kn_from_M1, Teuchos::null, coords_xpetra,
      maxwell1Params, true));
  if (verbosity >= 10 && rank == 0) {
    std::cout << "[Maxwell1] Built new preconditioner hierarchy"
              << (useKnFromM1 ? " with Kn from M1" : "")
              << " (split " << split << ")" << std::endl;
  }

  return Teuchos::rcp(new MueLu::TpetraOperator<ScalarT, LO, GO, Node>(
      Teuchos::rcp_static_cast<XpetraOperator>(precCache)));
}

// Return D0 if this split holds the HCURL block, null otherwise.
template<class Node>
typename BlockTypes<Node>::CrsMatrixRCP
splitD0(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
       const typename BlockTypes<Node>::CrsMatrixRCP & mat) {
  if (cntxt.is_null() || cntxt->refMaxwell.D0_matrix.is_null() || mat.is_null()) {
    return Teuchos::null;
  }
  if (!cntxt->refMaxwell.D0_matrix->getRangeMap()->isSameAs(*mat->getRowMap())) {
    return Teuchos::null;
  }
  return cntxt->refMaxwell.D0_matrix;
}

// MueLu parameters for one block-triangular split: its own 'AMG Settings' if it has any,
// otherwise the defaults. Also attaches D0 and NodeMatrix when the list asks for Hiptmair.
template<class Node>
Teuchos::ParameterList splitMueLuParams(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                                       const size_t split,
                                       const typename BlockTypes<Node>::CrsMatrixRCP & mat,
                                       const int verbosity) {
  const bool isTarget = (split == cntxt->schur_target_index);
  const Teuchos::ParameterList & splitList = cntxt->splitSettings(split);
  const bool hasAmgSublist = splitList.isSublist("AMG Settings");
  const std::string label = "'" + cntxt->split(split).name + "' split settings";

  Teuchos::ParameterList mueluParams;
  if (!hasAmgSublist ||
      !loadMueLuXmlIfPresent(splitList.sublist("AMG Settings"), mueluParams,
                                         label, mat->getComm())) {
    mueluParams = defaultMueLuParams();
    // A deck tuning the monolithic list means the Schur target, the block the outer
    // solver actually sees.
    const bool useMonolithic = !hasAmgSublist && isTarget &&
                               cntxt->prec_sublist.name() != "empty";
    if (hasAmgSublist || useMonolithic) {
      const Teuchos::ParameterList & deckList = hasAmgSublist
        ? splitList.sublist("AMG Settings") : cntxt->prec_sublist;
      applyDeckMueLuOverrides(mueluParams, deckList);
    }
    else if (isTarget) {
      // Tuned for the Schur complement, so they stay off the weight splits.
      setDefaultChebyshevSmoother(mueluParams, true);
    }
  }
  normalizeMueLuVerbosity(mueluParams, verbosity);
  detail::addHiptmairUserData<Node>(mueluParams, mat, splitD0<Node>(cntxt, mat),
                                               label, verbosity);
  return mueluParams;
}

} // namespace block_prec
} // namespace MrHyDE

#endif
