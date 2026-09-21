/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

#include "block_prec/ParamUtils.hpp"
#include "block_prec/BlockAssembly.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <MueLu_Maxwell_Utils.hpp>
#include <cctype>

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
  using XpetraCrs = Xpetra::TpetraCrsMatrix<ScalarT, LO, GO, Node>;
  using XpetraCrsMatrix = Xpetra::CrsMatrix<ScalarT, LO, GO, Node>;
  using XpetraCrsWrap = Xpetra::CrsMatrixWrap<ScalarT, LO, GO, Node>;
  using TpetraCoordMV = Tpetra::MultiVector<CoordScalarT, LO, GO, Node>;
  RefMaxwellXpetraInputs<Node> out;
  out.SM_wrap = wrapAsXpetraMatrix<Node>(J);
  out.D0_wrap = wrapAsXpetraMatrix<Node>(D0);
  out.M1_wrap = wrapAsXpetraMatrix<Node>(M1);
  out.coords_xpetra = Teuchos::rcp(new Xpetra::TpetraMultiVector<CoordScalarT, LO, GO, Node>(
      Teuchos::rcp_const_cast<TpetraCoordMV>(nodal_coords)));
  return out;
}

template<class Node>
void applyDirichletBCsToKn(
    Teuchos::RCP<Xpetra::Matrix<ScalarT, LO, GO, Node> > & Kn,
    const Kokkos::View<bool*, typename Node::device_type::memory_space> & BCdomainNodal) {
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

  Kn->resumeFill();
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
  Kn->fillComplete(Kn->getDomainMap(), Kn->getRangeMap());
}

} // namespace detail
} // namespace block_prec
} // namespace MrHyDE

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolver(Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                                                matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {

  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (cntxt->use_direct) {
    if (!cntxt->have_symb_factor) {
      cntxt->amesos_solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>(cntxt->amesos_type, J, r, soln);
      cntxt->amesos_solver->symbolicFactorization();
      cntxt->have_symb_factor = true;
    }
    cntxt->amesos_solver->setA(J, Amesos2::SYMBFACT);
    cntxt->amesos_solver->setX(soln);
    cntxt->amesos_solver->setB(r);
    cntxt->amesos_solver->numericFactorization().solve();
  }
  else {
    // GCRODR and RCG carry state (recycled subspace, conjugate vectors) that
    // must survive across solves; cache the SolverManager + LinearProblem.
    Teuchos::RCP<LA_LinearProblem> Problem;
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver;
    const bool reuse = cntxt->reuse_belos_solver_mgr
                       && !cntxt->belos_solver_mgr.is_null()
                       && !cntxt->belos_problem.is_null();
    if (reuse) {
      Problem = cntxt->belos_problem;
      Problem->setOperator(J);
      Problem->setLHS(soln);
      Problem->setRHS(r);
      solver = cntxt->belos_solver_mgr;
    }
    else {
      Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    }
    if (cntxt->use_preconditioner) {
      Teuchos::RCP<LA_Operator> preconditioner = this->buildOrUpdatePreconditioner(cntxt, J);
      this->attachPreconditionerToProblem(cntxt, Problem, preconditioner);
    }
    Problem->setProblem();
    if (reuse) {
      // Explicit re-bind; some SolverManagers cache op/prec handles at setProblem
      // and only refresh on that call.
      solver->setProblem(Problem);
    }
    else {
      Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList(cntxt);
      solver = this->createBelosSolverManager(Problem, belosList, cntxt->belos_type);
      if (cntxt->reuse_belos_solver_mgr) {
        cntxt->belos_problem = Problem;
        cntxt->belos_solver_mgr = solver;
      }
    }
    this->runBelosSolveAndHandleStatus(solver, cntxt);
    this->maybeReportConditionEstimate(solver, cntxt);
  }
  
}

template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
LinearAlgebraInterface<Node>::buildOrUpdatePreconditioner(
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
    const matrix_RCP & J) {
  if (cntxt->prec_type == "domain decomposition") {
    if (!cntxt->reuse_preconditioner || !cntxt->have_preconditioner) {
      Teuchos::ParameterList & ifpackList = cntxt->prec_sublist;
      ifpackList.set("schwarz: subdomain solver","garbage");
      cntxt->prec_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
      cntxt->prec_dd->setParameters(ifpackList);
      cntxt->prec_dd->initialize();
      cntxt->prec_dd->compute();
      cntxt->have_preconditioner = true;
    }
    return Teuchos::rcp_implicit_cast<LA_Operator>(cntxt->prec_dd);
  }

  if (cntxt->prec_type == "Ifpack2") {
    if (!cntxt->reuse_preconditioner || !cntxt->have_preconditioner) {
      Teuchos::ParameterList & ifpackList = cntxt->prec_sublist;
      if (verbosity >= 15 && comm->getRank() == 0) {
        std::cout << "Preconditioner parameters (monolithic Ifpack2 RELAXATION):" << std::endl;
        ifpackList.print(std::cout);
      }
      string method = settings->sublist("Solver").get("preconditioner variant","RELAXATION");
      cntxt->prec_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> >(method, J);
      cntxt->prec_dd->setParameters(ifpackList);
      cntxt->prec_dd->initialize();
      cntxt->prec_dd->compute();
      cntxt->have_preconditioner = true;
    }
    return Teuchos::rcp_implicit_cast<LA_Operator>(cntxt->prec_dd);
  }

  if (cntxt->prec_type == "block diagonal") {
    const size_t set = cntxt->equation_set_index;
    const bool keepExisting = cntxt->have_preconditioner && !cntxt->prec_block.is_null() &&
      reuseKeepsOperator(cntxt->preconditioner_reuse_type,
                                     cntxt->jacobian_rebuilt_this_step);
    if (!keepExisting) {
      cntxt->prec_block = this->buildBlockDiagonalPreconditioner(J, cntxt, set);
      cntxt->have_preconditioner = true;
    }
    return Teuchos::rcp_implicit_cast<LA_Operator>(cntxt->prec_block);
  }

  if (cntxt->prec_type == "block triangular") {
    const size_t set = cntxt->equation_set_index;
    cntxt->prec_block = this->setupBlockTriangularPreconditioner(J, cntxt, set);
    cntxt->have_preconditioner = true;
    return Teuchos::rcp_implicit_cast<LA_Operator>(cntxt->prec_block);
  }

  if (cntxt->prec_type == "AMG") {
    if (!cntxt->reuse_preconditioner || !cntxt->have_preconditioner) {
      cntxt->prec = this->buildAMGPreconditioner(J, cntxt);
      cntxt->have_preconditioner = true;
    }
    else {
      MueLu::ReuseTpetraPreconditioner(J, *(cntxt->prec));
    }
    return Teuchos::rcp_implicit_cast<LA_Operator>(cntxt->prec);
  }

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "Unsupported preconditioner type '" << cntxt->prec_type
    << "'. Supported values: AMG, Ifpack2, domain decomposition, block diagonal, block triangular.");
  return Teuchos::null;
}

template<class Node>
void LinearAlgebraInterface<Node>::attachPreconditionerToProblem(
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
    const Teuchos::RCP<LA_LinearProblem> & problem,
    const Teuchos::RCP<LA_Operator> & preconditioner) const {
  if (preconditioner.is_null()) return;
  if (cntxt->right_preconditioner) {
    problem->setRightPrec(preconditioner);
  }
  else {
    problem->setLeftPrec(preconditioner);
  }
}

template<class Node>
Teuchos::RCP<Belos::SolverManager<ScalarT,
                                  Tpetra::MultiVector<ScalarT,LO,GO,Node>,
                                  Tpetra::Operator<ScalarT,LO,GO,Node> > >
LinearAlgebraInterface<Node>::createBelosSolverManager(
    const Teuchos::RCP<LA_LinearProblem> & problem,
    const Teuchos::RCP<Teuchos::ParameterList> & belosList,
    const std::string & belosType) const {
  using BelosMV = Tpetra::MultiVector<ScalarT,LO,GO,Node>;
  const std::string belosUpper = toUpperAsciiCopy(belosType);
  if (belosUpper == "MINRES") {
    return Teuchos::rcp(new Belos::MinresSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "BLOCK GMRES") {
    return Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "BLOCK CG") {
    return Teuchos::rcp(new Belos::BlockCGSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "BICGSTAB") {
    return Teuchos::rcp(new Belos::BiCGStabSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "GCRODR") {
    return Teuchos::rcp(new Belos::GCRODRSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "PCPG") {
    return Teuchos::rcp(new Belos::PCPGSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "PSEUDO BLOCK CG") {
    return Teuchos::rcp(new Belos::PseudoBlockCGSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "PSEUDO BLOCK GMRES") {
    return Teuchos::rcp(new Belos::PseudoBlockGmresSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "PSEUDO BLOCK STOCHASTIC CG") {
    return Teuchos::rcp(new Belos::PseudoBlockStochasticCGSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "PSEUDO BLOCK TFQMR") {
    return Teuchos::rcp(new Belos::PseudoBlockTFQMRSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "RCG") {
    return Teuchos::rcp(new Belos::RCGSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "TFQMR") {
    return Teuchos::rcp(new Belos::TFQMRSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: unrecognized Belos solver: " + belosType);
  return Teuchos::null;
}

template<class Node>
void LinearAlgebraInterface<Node>::runBelosSolveAndHandleStatus(
    const Teuchos::RCP<Belos::SolverManager<ScalarT,
                                            Tpetra::MultiVector<ScalarT,LO,GO,Node>,
                                            Tpetra::Operator<ScalarT,LO,GO,Node> > > & solver,
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) const {
  const Belos::ReturnType belosStatus = solver->solve();
  if (belosStatus == Belos::Converged) return;

  const bool strictLinearSolve = settings->sublist("Solver").template get<bool>("strict linear solve", false);
  if (verbosity >= 1 && comm->getRank() == 0) {
    std::cout << "WARNING: Belos linear solve did not converge. "
              << "solver=" << cntxt->belos_type
              << ", iters=" << solver->getNumIters()
              << ", max linear iters=" << maxLinearIters
              << ", linear TOL=" << linearTOL
              << std::endl;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(strictLinearSolve, std::runtime_error,
    "Belos linear solve failed to converge and 'strict linear solve' is enabled.");
}

template<class Node>
void LinearAlgebraInterface<Node>::maybeReportConditionEstimate(
    const Teuchos::RCP<Belos::SolverManager<ScalarT,
                                            Tpetra::MultiVector<ScalarT,LO,GO,Node>,
                                            Tpetra::Operator<ScalarT,LO,GO,Node> > > & solver,
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) const {
  if (!doCondEst) return;
  if (toUpperAsciiCopy(cntxt->belos_type) != "PSEUDO BLOCK CG") return;
  using BelosMV = Tpetra::MultiVector<ScalarT,LO,GO,Node>;
  Teuchos::RCP<Belos::PseudoBlockCGSolMgr<ScalarT,BelosMV,LA_Operator> > solverCg =
    Teuchos::rcp_dynamic_cast<Belos::PseudoBlockCGSolMgr<ScalarT,BelosMV,LA_Operator> >(solver);
  if (!solverCg.is_null() && comm->getRank() == 0) {
    std::cout << "Belos condition number estimate = " << solverCg->getConditionEstimate() << std::endl;
  }
}
// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolver(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  context[set]->equation_set_index = set;
  this->linearSolver(context[set],J,r,soln);
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverL2(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  context_L2[set]->equation_set_index = set;
  this->linearSolver(context_L2[set],J,r,soln);
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverBoundaryL2(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  context_BndryL2[set]->equation_set_index = set;
  this->linearSolver(context_BndryL2[set],J,r,soln);
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverParam(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  this->linearSolver(context_param,J,r,soln);
}

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
//  this->linearSolver(context_param_L2,J,r,soln);
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverBoundaryL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
//  this->linearSolver(context_param_BndryL2,J,r,soln);
}

// ========================================================================================
// Preconditioner for Tpetra stack
// ========================================================================================

template<class Node>
Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > LinearAlgebraInterface<Node>::buildAMGPreconditioner(const matrix_RCP & J,
                                                                                                                 const Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {

  Teuchos::TimeMonitor localtimer(*prectimer);

  Teuchos::ParameterList mueluParams;

  // Check if XML parameter file is specified (optional)
  if (!cntxt->amg.xml_param_file.empty()) {
    // Load parameters from XML file
    loadXmlBroadcast(cntxt->amg.xml_param_file, mueluParams, *J->getComm(), "AMG");
    if (verbosity >= 6 && J->getComm()->getRank() == 0) {
      std::cout << "[AMG] Loaded parameters from XML file: "
                << cntxt->amg.xml_param_file << std::endl;
    }
  } else {
    // Use YAML-based parameters with defaults
    mueluParams = defaultMueLuParams();

    if (cntxt->prec_sublist.name() != "empty" ) {
      Teuchos::ParameterList filteredParams(cntxt->prec_sublist);
      removeMrHyDEOwnedKeys(filteredParams);
      removeIfpack2OnlyKeys(filteredParams);
      mueluParams.setParameters(filteredParams);
    }
    if (cntxt->prec_sublist.name() == "empty" ) {
      mueluParams.sublist("smoother: params").set("chebyshev: degree",2);
      mueluParams.sublist("smoother: params").set("chebyshev: ratio eigenvalue",7.0);
      mueluParams.sublist("smoother: params").set("chebyshev: min eigenvalue",1.0);
      mueluParams.sublist("smoother: params").set("chebyshev: zero starting solution",true);
    }
  }

  // Convert verbosity from int to string if needed
  if (mueluParams.isParameter("verbosity") && mueluParams.getEntry("verbosity").isType<int>()) {
    int v = mueluParams.get<int>("verbosity");
    mueluParams.set("verbosity", std::string(v <= 0 ? "none" : v <= 1 ? "low" : v <= 2 ? "medium" : "high"));
  }

  if (verbosity >= 20){
    mueluParams.set("verbosity","high");
  }

  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > Mnew = MueLu::CreateTpetraPreconditioner((Teuchos::RCP<LA_Operator>)J, mueluParams);

  return Mnew;
}

// Maxwell nomenclature used below:
//   SM  = HCURL system block: M/dt + curl(1/mu) curl.
//   D0  = nodal-to-edge gradient, as Panzer emits it (+-0.5).
//   M1  = HCURL edge mass matrix.
//   Kn  = nodal auxiliary matrix D0^T M1 D0.
template<class Node>
Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> >
LinearAlgebraInterface<Node>::buildRefMaxwellPreconditioner(
    const matrix_RCP & J,
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
    const Teuchos::ParameterList & blockSublist,
    const bool forSchur) {

  Teuchos::TimeMonitor localtimer(*prectimer);
  using RefMaxwellType = MueLu::RefMaxwell<ScalarT, LO, GO, Node>;
  Teuchos::RCP<RefMaxwellType> & precCache = forSchur ? cntxt->schur_refmaxwell_prec : cntxt->refmaxwell_prec;

  using XpetraMatrix = Xpetra::Matrix<ScalarT, LO, GO, Node>;

  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.D0_matrix.is_null(), std::runtime_error,
    "RefMaxwell requires D0_matrix in context.");
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.nodal_coords.is_null(), std::runtime_error,
    "RefMaxwell requires nodal_coords in context.");
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.D0_matrix->getDomainMap().is_null(), std::runtime_error,
    "RefMaxwell requires D0 domain map to be non-null.");
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.D0_matrix->getRangeMap().is_null(), std::runtime_error,
    "RefMaxwell requires D0 range map to be non-null.");
  const int rank = J->getComm()->getRank();
  const GO J_global_rows = J->getGlobalNumRows();
  const GO D0_global_rows = cntxt->refMaxwell.D0_matrix->getGlobalNumRows();
  const GO D0_global_cols = cntxt->refMaxwell.D0_matrix->getGlobalNumCols();
  TEUCHOS_TEST_FOR_EXCEPTION(J_global_rows != D0_global_rows, std::runtime_error,
    "RefMaxwell map mismatch: system matrix has " << J_global_rows
    << " rows but D0 has " << D0_global_rows << " rows.");
  const Teuchos::RCP<const LA_Map> d0_edge_map = cntxt->refMaxwell.D0_matrix->getRangeMap();
  TEUCHOS_TEST_FOR_EXCEPTION(!J->getRowMap()->isSameAs(*d0_edge_map), std::runtime_error,
    "RefMaxwell requires A-block row map to match D0 range map.");
  TEUCHOS_TEST_FOR_EXCEPTION(!J->getDomainMap()->isSameAs(*d0_edge_map), std::runtime_error,
    "RefMaxwell requires A-block domain map to match D0 range map.");

  const Teuchos::RCP<const LA_Map> edge_map = d0_edge_map;
  matrix_RCP M1_use = cntxt->refMaxwell.M1_matrix;
  bool m1_ok = !M1_use.is_null();
  if (m1_ok) {
    m1_ok = M1_use->getGlobalNumRows() == edge_map->getGlobalNumElements() &&
            M1_use->getGlobalNumCols() == edge_map->getGlobalNumElements() &&
            M1_use->getRowMap()->isSameAs(*edge_map) &&
            M1_use->getDomainMap()->isSameAs(*edge_map);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(!m1_ok, std::runtime_error,
    "RefMaxwell requires M1_matrix with row/domain maps equal to D0 range map.");

  const Teuchos::RCP<const LA_Map> nodal_map = cntxt->refMaxwell.D0_matrix->getDomainMap();
  TEUCHOS_TEST_FOR_EXCEPTION(!cntxt->refMaxwell.nodal_coords->getMap()->isSameAs(*nodal_map), std::runtime_error,
    "RefMaxwell requires nodal coordinates map to match D0 domain map.");
  TEUCHOS_TEST_FOR_EXCEPTION(
    cntxt->refMaxwell.nodal_coords->getGlobalLength() != static_cast<Tpetra::global_size_t>(D0_global_cols),
    std::runtime_error,
    "RefMaxwell requires nodal coordinates length to match D0 column count.");
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.nodal_coords->getLocalLength() != nodal_map->getLocalNumElements(), std::runtime_error,
    "RefMaxwell requires nodal coordinates local length to match local D0 domain size.");
  using XpetraOperator = Xpetra::Operator<ScalarT, LO, GO, Node>;
  const Teuchos::ParameterList rmSettings = blockSublist.isSublist("RefMaxwell Settings")
    ? blockSublist.sublist("RefMaxwell Settings") : Teuchos::ParameterList();

  // Filtering and operator checks are disabled by default.
  const block_prec::detail::FilterOpts filterOpts = block_prec::detail::readFilterOpts(rmSettings);

  // MueLu bakes beta into the hierarchy, and only the Schur one has an addon.
  const ScalarT betaTarget = (forSchur && cntxt->refMaxwell.schur_addon_wanted)
    ? cntxt->refMaxwell.schur_addon_beta : 0.0;
  const ScalarT betaWas = forSchur ? cntxt->refMaxwell.schur_addon_beta_built : 0.0;
  if (std::abs(betaTarget - betaWas) >
      1.0e-12 * std::max(std::abs(betaTarget), std::abs(betaWas))) {
    precCache = Teuchos::null;
  }
  const bool canReuse = !precCache.is_null() &&
    reuseKeepsHierarchy(cntxt->preconditioner_reuse_type);

  block_prec::detail::MaxwellInputs<Node> in = block_prec::detail::filterSMOnly<Node>(
    J, M1_use, cntxt->refMaxwell.D0_matrix, filterOpts, canReuse);

  // Everything past here feeds the hierarchy build, which reuse skips.
  if (canReuse) {
    return block_prec::detail::resetAndWrap<Node>(precCache, in.SM, "RefMaxwell",
                                                  forSchur, verbosity, rank);
  }
  block_prec::detail::finishMaxwellInputs<Node>(
    in, cntxt->refMaxwell.nodal_coords, filterOpts, "RefMaxwell", verbosity, rank);
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

  const std::string & refmaxwellXmlFile = forSchur ? cntxt->refMaxwell.xml_param_file_schur
                                                   : cntxt->refMaxwell.xml_param_file_pivot;
  TEUCHOS_TEST_FOR_EXCEPTION(refmaxwellXmlFile.empty(), std::runtime_error,
    "RefMaxwell requires 'xml param file' in "
    << (forSchur ? "Schur" : "Pivot") << " Block Settings -> RefMaxwell Settings.");

  Teuchos::ParameterList refmaxwellParams;

  loadXmlBroadcast(refmaxwellXmlFile, refmaxwellParams, *J->getComm(), "RefMaxwell");
  if (verbosity >= 6 && J->getComm()->getRank() == 0) {
    std::cout << "[RefMaxwell] Loaded parameters from XML file: " << refmaxwellXmlFile << std::endl;
  }

  sanitizeDirectCoarseParams(refmaxwellParams.sublist("refmaxwell: 11list"));
  sanitizeDirectCoarseParams(refmaxwellParams.sublist("refmaxwell: 22list"));
  warnNonStationarySmoother(refmaxwellParams, "refmaxwell: 11list", J->getComm());
  warnNonStationarySmoother(refmaxwellParams, "refmaxwell: 22list", J->getComm());

  // AMS only for this path.
  refmaxwellParams.set("refmaxwell: space number", 1);
  // M0(1/beta)^-1 needs beta, so fall back to no addon until it is known.
  const bool wantAddon = !refmaxwellParams.get<bool>("refmaxwell: disable addon", true);
  TEUCHOS_TEST_FOR_EXCEPTION(wantAddon && !forSchur, std::runtime_error,
    "RefMaxwell XML '" << refmaxwellXmlFile << "' enables the addon on the pivot block. "
    "beta is read off the Schur correction and has no pivot-block counterpart.");
  if (forSchur) cntxt->refMaxwell.schur_addon_wanted = wantAddon;
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
    M0inv_wrap = block_prec::detail::wrapAsXpetraMatrix<Node>(
      block_prec::detail::buildScaledInverseDiagonalMatrix<Node>(
        cntxt->refMaxwell.nodal_lumped_mass, cntxt->refMaxwell.schur_addon_beta));
  }
  // Null nullspace: MueLu forms D0*coords itself, which is what we would pass.
  precCache = Teuchos::rcp(new RefMaxwellType(
      SM_wrap, D0_wrap, M1_wrap, M0inv_wrap, M1_wrap,
      Teuchos::null, coords_xpetra,
      refmaxwellParams, true));
  if (forSchur) {
    cntxt->refMaxwell.schur_addon_beta_built = haveAddon ? cntxt->refMaxwell.schur_addon_beta : 0.0;
  }
  if (verbosity >= 10 && J->getComm()->getRank() == 0) {
    std::cout << "[RefMaxwell] Built new preconditioner hierarchy" << (forSchur ? " (Schur)" : "") << std::endl;
  }

  return Teuchos::rcp(new MueLu::TpetraOperator<ScalarT, LO, GO, Node>(
      Teuchos::rcp_static_cast<XpetraOperator>(precCache)));
}

template<class Node>
Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> >
LinearAlgebraInterface<Node>::buildMaxwell1Preconditioner(
    const matrix_RCP & J,
    const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
    const Teuchos::ParameterList & blockSublist,
    const bool forSchur) {

  Teuchos::TimeMonitor localtimer(*prectimer);
  using Maxwell1Type = MueLu::Maxwell1<ScalarT, LO, GO, Node>;
  Teuchos::RCP<Maxwell1Type> & precCache = forSchur ? cntxt->schur_maxwell1_prec : cntxt->maxwell1_prec;

  using XpetraMatrix = Xpetra::Matrix<ScalarT, LO, GO, Node>;
  using XpetraOperator = Xpetra::Operator<ScalarT, LO, GO, Node>;

  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.D0_matrix.is_null(), std::runtime_error,
    "Maxwell1 requires D0_matrix in context (shared with RefMaxwell setup).");
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.nodal_coords.is_null(), std::runtime_error,
    "Maxwell1 requires nodal_coords in context.");

  matrix_RCP M1_use = cntxt->refMaxwell.M1_matrix;
  TEUCHOS_TEST_FOR_EXCEPTION(M1_use.is_null(), std::runtime_error,
    "Maxwell1 setup path reuses the RefMaxwell auxiliary M1 matrix. It is null.");

  const int rank = J->getComm()->getRank();
  // Panzer OPERATOR_GRAD emits +-0.5; MueLu's ReitzingerP requires +-1.
  if (cntxt->maxwell1.D0_normalized.is_null()) {
    cntxt->maxwell1.D0_normalized = block_prec::detail::snapCrsMatrixSigns<Node>(
      cntxt->refMaxwell.D0_matrix);
  }

  // Filtering and operator checks are disabled by default.
  const Teuchos::ParameterList m1Settings = blockSublist.isSublist("Maxwell1 Settings")
    ? blockSublist.sublist("Maxwell1 Settings") : Teuchos::ParameterList();
  const block_prec::detail::FilterOpts filterOpts = block_prec::detail::readFilterOpts(m1Settings);
  const bool canReuse = !precCache.is_null() &&
    reuseKeepsHierarchy(cntxt->preconditioner_reuse_type);

  block_prec::detail::MaxwellInputs<Node> in = block_prec::detail::filterSMOnly<Node>(
    J, M1_use, cntxt->maxwell1.D0_normalized, filterOpts, canReuse);

  // Everything past here feeds the hierarchy build, which reuse skips.
  if (canReuse) {
    return block_prec::detail::resetAndWrap<Node>(precCache, in.SM, "Maxwell1",
                                                  forSchur, verbosity, rank);
  }
  block_prec::detail::finishMaxwellInputs<Node>(
    in, cntxt->refMaxwell.nodal_coords, filterOpts, "Maxwell1", verbosity, rank);
  matrix_RCP SM_for_setup = in.SM, M1_for_setup = in.M1;

  block_prec::detail::RefMaxwellXpetraInputs<Node> xpetraInputs = block_prec::detail::buildRefMaxwellXpetraInputs<Node>(
    SM_for_setup, cntxt->maxwell1.D0_normalized, M1_for_setup, cntxt->refMaxwell.nodal_coords);
  Teuchos::RCP<XpetraMatrix> SM_wrap = xpetraInputs.SM_wrap;
  Teuchos::RCP<XpetraMatrix> D0_wrap = xpetraInputs.D0_wrap;
  auto coords_xpetra = xpetraInputs.coords_xpetra;

  const std::string & maxwell1XmlFile = forSchur ? cntxt->maxwell1.xml_param_file_schur
                                                 : cntxt->maxwell1.xml_param_file_pivot;
  TEUCHOS_TEST_FOR_EXCEPTION(maxwell1XmlFile.empty(), std::runtime_error,
    "Maxwell1 preconditioner requires 'xml param file' in the "
    << (forSchur ? "'Schur Block Settings'" : "'Pivot Block Settings'")
    << " 'Maxwell1 Settings' sublist.");
  Teuchos::ParameterList maxwell1Params;
  loadXmlBroadcast(maxwell1XmlFile, maxwell1Params, *J->getComm(), "Maxwell1");
  if (verbosity >= 6 && rank == 0) {
    std::cout << "[Maxwell1] Loaded parameters from XML file: " << maxwell1XmlFile << std::endl;
  }

  sanitizeDirectCoarseParams(maxwell1Params.sublist("maxwell1: 11list"));
  sanitizeDirectCoarseParams(maxwell1Params.sublist("maxwell1: 22list"));

  // Use M1 because PEC identity rows in SM create O(1)/O(h) diagonal contrast
  // that breaks aggregation in D0^T SM D0.
  Teuchos::RCP<XpetraMatrix> Kn_from_M1;
  const bool knXmlSet = maxwell1Params.isParameter("maxwell1: use Kn from M1");
  const bool knFromXml = knXmlSet && maxwell1Params.get<bool>("maxwell1: use Kn from M1");
  maxwell1Params.remove("maxwell1: use Kn from M1", false);
  const bool knInYaml = m1Settings.isParameter("use Kn from M1");
  const bool useKnFromM1 = knInYaml ? m1Settings.get<bool>("use Kn from M1") : knFromXml;
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

    auto computeKnGlobalConstants = [&]() {
      // MueLu hierarchy statistics require the global graph constants.
      Teuchos::rcp_const_cast<Xpetra::CrsGraph<LO, GO, Node> >(
          Kn_from_M1->getCrsGraph())->computeGlobalConstants();
    };

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

    // Remove BC rows before Maxwell1 adds zeros that ReitzingerPFactory rejects.
    // Kn already uses the full D0.
    if (BCedges > 0) {
      Kokkos::View<const bool*, dev_mem_space> BCrowsK_c = BCrowsK;
      Teuchos::RCP<LA_CrsMatrix> D0_bc_pruned = block_prec::detail::dropBCRows<Node>(
          cntxt->maxwell1.D0_normalized, BCrowsK_c);
      using XpetraCrs = Xpetra::TpetraCrsMatrix<ScalarT, LO, GO, Node>;
      using XpetraCrsWrap = Xpetra::CrsMatrixWrap<ScalarT, LO, GO, Node>;
      using XpetraCrsMatrix = Xpetra::CrsMatrix<ScalarT, LO, GO, Node>;
      D0_wrap = Teuchos::rcp(new XpetraCrsWrap(
          Teuchos::rcp_implicit_cast<XpetraCrsMatrix>(
              Teuchos::rcp(new XpetraCrs(D0_bc_pruned)))));
    }

    // Zero Dirichlet rows and columns, then restore the original diagonal.

    if (BCnodes > 0) {
      block_prec::detail::applyDirichletBCsToKn<Node>(Kn_from_M1, BCdomainK);
    }
    computeKnGlobalConstants();

    // Warn if Kn_from_M1 differs from D0^T SM D0 beyond roundoff on non-BC rows.
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
              << (forSchur ? " (Schur)" : "") << std::endl;
  }

  return Teuchos::rcp(new MueLu::TpetraOperator<ScalarT, LO, GO, Node>(
      Teuchos::rcp_static_cast<XpetraOperator>(precCache)));
}

