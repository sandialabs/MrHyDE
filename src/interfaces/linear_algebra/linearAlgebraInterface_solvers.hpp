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
  using XpetraMultiVector = Xpetra::MultiVector<ScalarT, LO, GO, Node>;
  using XpetraCoordMV = Xpetra::TpetraMultiVector<CoordScalarT, LO, GO, Node>;
  Teuchos::RCP<XpetraMatrix> SM_wrap;
  Teuchos::RCP<XpetraMatrix> D0_wrap;
  Teuchos::RCP<XpetraMatrix> M1_wrap;
  Teuchos::RCP<XpetraMatrix> M0inv_wrap;
  Teuchos::RCP<XpetraCoordMV> coords_xpetra;
  Teuchos::RCP<XpetraMultiVector> nullspace_xpetra;
};

template<class Node>
RefMaxwellXpetraInputs<Node> buildRefMaxwellXpetraInputs(
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT, LO, GO, Node> > & J,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT, LO, GO, Node> > & D0,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT, LO, GO, Node> > & M1,
    const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT, LO, GO, Node> > & M0inv,
    const Teuchos::RCP<const Tpetra::MultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType, LO, GO, Node> > & nodal_coords,
    const Teuchos::RCP<const Tpetra::MultiVector<ScalarT, LO, GO, Node> > & nullspace) {
  using CoordScalarT = typename Teuchos::ScalarTraits<ScalarT>::coordinateType;
  using TpetraCrs = Tpetra::CrsMatrix<ScalarT, LO, GO, Node>;
  using XpetraCrs = Xpetra::TpetraCrsMatrix<ScalarT, LO, GO, Node>;
  using XpetraCrsMatrix = Xpetra::CrsMatrix<ScalarT, LO, GO, Node>;
  using XpetraCrsWrap = Xpetra::CrsMatrixWrap<ScalarT, LO, GO, Node>;
  using XpetraMV = Xpetra::TpetraMultiVector<ScalarT, LO, GO, Node>;
  using TpetraCoordMV = Tpetra::MultiVector<CoordScalarT, LO, GO, Node>;
  RefMaxwellXpetraInputs<Node> out;
  out.SM_wrap = Teuchos::rcp(new XpetraCrsWrap(Teuchos::rcp_implicit_cast<XpetraCrsMatrix>(Teuchos::rcp(new XpetraCrs(Teuchos::rcp_const_cast<TpetraCrs>(J))))));
  out.D0_wrap = Teuchos::rcp(new XpetraCrsWrap(Teuchos::rcp_implicit_cast<XpetraCrsMatrix>(Teuchos::rcp(new XpetraCrs(Teuchos::rcp_const_cast<TpetraCrs>(D0))))));
  out.M1_wrap = Teuchos::rcp(new XpetraCrsWrap(Teuchos::rcp_implicit_cast<XpetraCrsMatrix>(Teuchos::rcp(new XpetraCrs(Teuchos::rcp_const_cast<TpetraCrs>(M1))))));
  out.M0inv_wrap = Teuchos::rcp(new XpetraCrsWrap(Teuchos::rcp_implicit_cast<XpetraCrsMatrix>(Teuchos::rcp(new XpetraCrs(Teuchos::rcp_const_cast<TpetraCrs>(M0inv))))));
  out.coords_xpetra = Teuchos::rcp(new Xpetra::TpetraMultiVector<CoordScalarT, LO, GO, Node>(
      Teuchos::rcp_const_cast<TpetraCoordMV>(nodal_coords)));
  out.nullspace_xpetra = nullspace.is_null()
    ? Teuchos::null
    : Teuchos::rcp(new XpetraMV(Teuchos::rcp_const_cast<Tpetra::MultiVector<ScalarT, LO, GO, Node>>(nullspace)));
  return out;
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
    if (!cntxt->reuse_preconditioner || !cntxt->have_preconditioner) {
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
    try {
      mueluParams = *Teuchos::getParametersFromXmlFile(cntxt->amg.xml_param_file);
      if (verbosity >= 6 && J->getComm()->getRank() == 0) {
        std::cout << "[AMG] Loaded parameters from XML file: "
                  << cntxt->amg.xml_param_file << std::endl;
      }
    } catch (const std::exception& e) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "Failed to load AMG parameters from XML file '"
        << cntxt->amg.xml_param_file << "': " << e.what());
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
//   D0  = nodal-to-edge gradient, normalized to {-1, +1}.
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
  const bool hasNestedRefMaxwellSettings =
    (blockSublist.name() != "empty") &&
    blockSublist.isSublist("RefMaxwell Settings");

  // Handle use lumped M0inv parameter - this is MrHyDE-specific, not passed to MueLu
  bool useLumpedM0inv = true; // default
  if (hasNestedRefMaxwellSettings) {
    const Teuchos::ParameterList & refmaxwellSettings = blockSublist.sublist("RefMaxwell Settings");
    if (refmaxwellSettings.isParameter("use lumped M0inv")) {
      useLumpedM0inv = refmaxwellSettings.template get<bool>("use lumped M0inv");
    }
  } else if (cntxt->prec_sublist.name() != "empty") {
    if (cntxt->prec_sublist.isParameter("use lumped M0inv")) {
      useLumpedM0inv = cntxt->prec_sublist.template get<bool>("use lumped M0inv");
    } else if (cntxt->prec_sublist.isParameter("refmaxwell: use lumped M0inv")) {
      useLumpedM0inv = cntxt->prec_sublist.template get<bool>("refmaxwell: use lumped M0inv");
    }
  }

  matrix_RCP M0inv = useLumpedM0inv
    ? block_prec::detail::buildLumpedM0inv<Node>(cntxt->refMaxwell.D0_matrix, M1_use, nodal_map, edge_map, verbosity)
    : block_prec::detail::buildM0invIdentity<Node>(nodal_map);

  // Filtering and operator checks are disabled by default.
  matrix_RCP SM_for_setup = J;
  matrix_RCP M1_for_setup = M1_use;
  block_prec::detail::FilterResult<Node> smFilterResult;
  block_prec::detail::FilterResult<Node> m1FilterResult;
  const block_prec::detail::FilterOpts filterOpts =
    hasNestedRefMaxwellSettings
      ? block_prec::detail::readFilterOpts(blockSublist.sublist("RefMaxwell Settings"))
      : block_prec::detail::FilterOpts{};
  if (filterOpts.filterSM) {
    smFilterResult = block_prec::detail::filterExplicitZeros<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(J), filterOpts.tol, filterOpts.verifyComplex);
    m1FilterResult = block_prec::detail::filterExplicitZeros<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(M1_use), filterOpts.tol, filterOpts.verifyComplex);
    SM_for_setup = smFilterResult.matrix;
    M1_for_setup = m1FilterResult.matrix;
    block_prec::detail::assertStructuralSymmetry<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(M1_for_setup), "RefMaxwell M1 filter");
    block_prec::detail::assertKernelBound<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(SM_for_setup),
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(J),
      cntxt->refMaxwell.D0_matrix, filterOpts.tol, "RefMaxwell SM filter");
    if (verbosity >= 6 && rank == 0) {
      const auto sm_in = J->getGlobalNumEntries(),   sm_out = SM_for_setup->getGlobalNumEntries();
      const auto m1_in = M1_use->getGlobalNumEntries(), m1_out = M1_for_setup->getGlobalNumEntries();
      std::cout << "[RefMaxwell] filter SM tol=" << filterOpts.tol
                << ": SM " << sm_in << " -> " << sm_out
                << " (dropped " << (100.0 * (sm_in - sm_out) / std::max<decltype(sm_in)>(sm_in, 1)) << "%)"
                << ", M1 " << m1_in << " -> " << m1_out
                << " (dropped " << (100.0 * (m1_in - m1_out) / std::max<decltype(m1_in)>(m1_in, 1)) << "%)"
                << std::endl;
    }
  }
  if (filterOpts.verifyComplex) {
    block_prec::detail::verifyMaxwellComplex<Node>(
      cntxt->refMaxwell.D0_matrix, cntxt->refMaxwell.nodal_coords,
      cntxt->refMaxwell.nullspace,
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(J),
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(M1_use),
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(SM_for_setup),
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(M1_for_setup),
      smFilterResult.dropped, m1FilterResult.dropped,
      filterOpts.tol, verbosity, rank, "RefMaxwell");
  }

  block_prec::detail::RefMaxwellXpetraInputs<Node> xpetraInputs = block_prec::detail::buildRefMaxwellXpetraInputs<Node>(
    SM_for_setup, cntxt->refMaxwell.D0_matrix, M1_for_setup, M0inv, cntxt->refMaxwell.nodal_coords, cntxt->refMaxwell.nullspace);
  Teuchos::RCP<XpetraMatrix> SM_wrap = xpetraInputs.SM_wrap;
  Teuchos::RCP<XpetraMatrix> D0_wrap = xpetraInputs.D0_wrap;
  Teuchos::RCP<XpetraMatrix> M1_wrap = xpetraInputs.M1_wrap;
  Teuchos::RCP<XpetraMatrix> M0inv_wrap = xpetraInputs.M0inv_wrap;
  auto coords_xpetra = xpetraInputs.coords_xpetra;
  auto nullspace_xpetra = xpetraInputs.nullspace_xpetra;

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

  try {
    refmaxwellParams = *Teuchos::getParametersFromXmlFile(refmaxwellXmlFile);
    if (verbosity >= 6 && J->getComm()->getRank() == 0) {
      std::cout << "[RefMaxwell] Loaded parameters from XML file: " << refmaxwellXmlFile << std::endl;
    }
  } catch (const std::exception& e) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "Failed to load RefMaxwell parameters from XML file '" << refmaxwellXmlFile
      << "': " << e.what());
  }

  sanitizeDirectCoarseParams(refmaxwellParams.sublist("refmaxwell: 11list"));
  sanitizeDirectCoarseParams(refmaxwellParams.sublist("refmaxwell: 22list"));
  warnNonStationarySmoother(refmaxwellParams, "refmaxwell: 11list", J->getComm());
  warnNonStationarySmoother(refmaxwellParams, "refmaxwell: 22list", J->getComm());

  // AMS only for this path.
  refmaxwellParams.set("refmaxwell: space number", 1);

  if (verbosity >= 10 && J->getComm()->getRank() == 0) {
    std::cout << "[RefMaxwell] Final parameter list:" << std::endl;
    refmaxwellParams.print(std::cout, 2, true);
  }

  using XpetraOperator = Xpetra::Operator<ScalarT, LO, GO, Node>;
  const std::string reuseType = toUpperAsciiCopy(cntxt->preconditioner_reuse_type);
  const bool canReuse = !precCache.is_null() && (reuseType == "FULL" || reuseType == "UPDATE");

  if (canReuse) {
    precCache->resetMatrix(SM_wrap);
    if (verbosity >= 10 && J->getComm()->getRank() == 0) {
      std::cout << "[RefMaxwell] Reusing existing hierarchy with resetMatrix()" << (forSchur ? " (Schur)" : "") << std::endl;
    }
  }
  else {
    precCache = Teuchos::rcp(new RefMaxwellType(
        SM_wrap, D0_wrap, M1_wrap, M0inv_wrap, M1_wrap,
        nullspace_xpetra, coords_xpetra,
        refmaxwellParams, true));
    if (verbosity >= 10 && J->getComm()->getRank() == 0) {
      std::cout << "[RefMaxwell] Built new preconditioner hierarchy" << (forSchur ? " (Schur)" : "") << std::endl;
    }
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
  const Teuchos::RCP<const LA_Map> nodal_map = cntxt->refMaxwell.D0_matrix->getDomainMap();
  matrix_RCP M0inv = block_prec::detail::buildM0invIdentity<Node>(nodal_map);

  // Panzer OPERATOR_GRAD emits +-0.5; MueLu's ReitzingerP requires +-1.
  if (cntxt->maxwell1.D0_normalized.is_null()) {
    cntxt->maxwell1.D0_normalized = block_prec::detail::snapCrsMatrixSigns<Node>(
      cntxt->refMaxwell.D0_matrix);
  }

  // Filtering and operator checks are disabled by default.
  matrix_RCP SM_for_setup = J;
  matrix_RCP M1_for_setup = M1_use;
  block_prec::detail::FilterResult<Node> smFilterResult;
  block_prec::detail::FilterResult<Node> m1FilterResult;
  const block_prec::detail::FilterOpts filterOpts =
    blockSublist.isSublist("Maxwell1 Settings")
      ? block_prec::detail::readFilterOpts(blockSublist.sublist("Maxwell1 Settings"))
      : block_prec::detail::FilterOpts{};
  if (filterOpts.filterSM) {
    smFilterResult = block_prec::detail::filterExplicitZeros<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(J), filterOpts.tol, filterOpts.verifyComplex);
    m1FilterResult = block_prec::detail::filterExplicitZeros<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(M1_use), filterOpts.tol, filterOpts.verifyComplex);
    SM_for_setup = smFilterResult.matrix;
    M1_for_setup = m1FilterResult.matrix;
    block_prec::detail::assertStructuralSymmetry<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(M1_for_setup), "Maxwell1 M1 filter");
    block_prec::detail::assertKernelBound<Node>(
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(SM_for_setup),
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(J),
      cntxt->maxwell1.D0_normalized, filterOpts.tol, "Maxwell1 SM filter");
    if (verbosity >= 6 && J->getComm()->getRank() == 0) {
      const auto sm_in = J->getGlobalNumEntries(),   sm_out = SM_for_setup->getGlobalNumEntries();
      const auto m1_in = M1_use->getGlobalNumEntries(), m1_out = M1_for_setup->getGlobalNumEntries();
      std::cout << "[Maxwell1] filter SM tol=" << filterOpts.tol
                << ": SM " << sm_in << " -> " << sm_out
                << " (dropped " << (100.0 * (sm_in - sm_out) / std::max<decltype(sm_in)>(sm_in, 1)) << "%)"
                << ", M1 " << m1_in << " -> " << m1_out
                << " (dropped " << (100.0 * (m1_in - m1_out) / std::max<decltype(m1_in)>(m1_in, 1)) << "%)"
                << std::endl;
    }
  }
  if (filterOpts.verifyComplex) {
    block_prec::detail::verifyMaxwellComplex<Node>(
      cntxt->maxwell1.D0_normalized, cntxt->refMaxwell.nodal_coords,
      cntxt->refMaxwell.nullspace,
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(J),
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(M1_use),
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(SM_for_setup),
      Teuchos::rcp_implicit_cast<const LA_CrsMatrix>(M1_for_setup),
      smFilterResult.dropped, m1FilterResult.dropped,
      filterOpts.tol, verbosity, J->getComm()->getRank(), "Maxwell1");
  }

  block_prec::detail::RefMaxwellXpetraInputs<Node> xpetraInputs = block_prec::detail::buildRefMaxwellXpetraInputs<Node>(
    SM_for_setup, cntxt->maxwell1.D0_normalized, M1_for_setup, M0inv, cntxt->refMaxwell.nodal_coords, cntxt->refMaxwell.nullspace);
  Teuchos::RCP<XpetraMatrix> SM_wrap = xpetraInputs.SM_wrap;
  Teuchos::RCP<XpetraMatrix> D0_wrap = xpetraInputs.D0_wrap;
  auto coords_xpetra = xpetraInputs.coords_xpetra;
  auto nullspace_xpetra = xpetraInputs.nullspace_xpetra;

  const std::string & maxwell1XmlFile = forSchur ? cntxt->maxwell1.xml_param_file_schur
                                                 : cntxt->maxwell1.xml_param_file_pivot;
  TEUCHOS_TEST_FOR_EXCEPTION(maxwell1XmlFile.empty(), std::runtime_error,
    "Maxwell1 preconditioner requires 'xml param file' in the "
    << (forSchur ? "'Schur Block Settings'" : "'Pivot Block Settings'")
    << " 'Maxwell1 Settings' sublist.");
  Teuchos::ParameterList maxwell1Params;
  try {
    maxwell1Params = *Teuchos::getParametersFromXmlFile(maxwell1XmlFile);
    if (verbosity >= 6 && J->getComm()->getRank() == 0) {
      std::cout << "[Maxwell1] Loaded parameters from XML file: " << maxwell1XmlFile << std::endl;
    }
  } catch (const std::exception& e) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "Failed to load Maxwell1 parameters from XML file '" << maxwell1XmlFile
      << "': " << e.what());
  }

  sanitizeDirectCoarseParams(maxwell1Params.sublist("maxwell1: 11list"));
  sanitizeDirectCoarseParams(maxwell1Params.sublist("maxwell1: 22list"));

  const std::string reuseType = toUpperAsciiCopy(cntxt->preconditioner_reuse_type);
  const bool canReuse = !precCache.is_null() && (reuseType == "FULL" || reuseType == "UPDATE");

  // Use M1 because PEC identity rows in SM create O(1)/O(h) diagonal contrast
  // that breaks aggregation in D0^T SM D0.
  Teuchos::RCP<XpetraMatrix> Kn_from_M1;
  const bool useKnFromM1 = maxwell1Params.isParameter("maxwell1: use Kn from M1") &&
                           maxwell1Params.get<bool>("maxwell1: use Kn from M1");
  maxwell1Params.remove("maxwell1: use Kn from M1", false);
  if (useKnFromM1 && !canReuse) {
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
    if (verbosity >= 10 && J->getComm()->getRank() == 0) {
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
    auto applyDirichletBCsToKn = [](
        Teuchos::RCP<XpetraMatrix> & Kn,
        const Kokkos::View<bool*, dev_mem_space> & BCdomainNodal) {
      using XpetraVector = Xpetra::Vector<ScalarT, LO, GO, Node>;
      Teuchos::RCP<XpetraVector> saved = Xpetra::VectorFactory<ScalarT, LO, GO, Node>::Build(
          Kn->getRowMap(), true);
      Kn->getLocalDiagCopy(*saved);
      auto savedView = saved->getLocalViewDevice(Tpetra::Access::ReadOnly);

      auto knColMap = Kn->getColMap();
      auto knRowMap = Kn->getRowMap();
      const LO nColLocal = static_cast<LO>(knColMap->getLocalNumElements());
      Kokkos::View<bool*, dev_mem_space> BCcols("BCcols_kn", nColLocal);
      {
        auto lclColMap = knColMap->getLocalMap();
        auto lclRowMap = knRowMap->getLocalMap();
        const LO invalidLO = Teuchos::OrdinalTraits<LO>::invalid();
        Kokkos::parallel_for("BCcols_fill",
            Kokkos::RangePolicy<typename Node::execution_space>(0, nColLocal),
            KOKKOS_LAMBDA(const LO cLid) {
              const GO cGid = lclColMap.getGlobalElement(cLid);
              const LO rLid = lclRowMap.getLocalElement(cGid);
              BCcols(cLid) = (rLid != invalidLO) ? BCdomainNodal(rLid) : false;
            });
      }
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
    };

    if (BCnodes > 0) {
      applyDirichletBCsToKn(Kn_from_M1, BCdomainK);
    }
    computeKnGlobalConstants();

    // Warn if Kn_from_M1 differs from D0^T SM D0 beyond roundoff on non-BC rows.
    if (filterOpts.verifyKnConsistency) {
      Teuchos::ParameterList rapList2;
      rapList2.set("rap: fix zero diagonals", false);
      Teuchos::RCP<XpetraMatrix> Kn_SM = MueLu::Maxwell_Utils<ScalarT, LO, GO, Node>::PtAPWrapper(
          SM_wrap, D0_wrap, rapList2, "Kn_from_SM");
      Teuchos::rcp_const_cast<Xpetra::CrsGraph<LO, GO, Node>>(Kn_SM->getCrsGraph())->computeGlobalConstants();
      Kokkos::View<bool*, dev_mem_space> BCr_sm, BCc_sm, BCd_sm;
      bool aE = false, aN = false;
      int nE = 0, nN = 0;
      MueLu::Maxwell_Utils<ScalarT, LO, GO, Node>::detectBoundaryConditionsSM(
          SM_wrap, D0_wrap, -1.0, BCr_sm, BCc_sm, BCd_sm, nE, nN, aE, aN);
      if (nN > 0) applyDirichletBCsToKn(Kn_SM, BCd_sm);
      auto knM1_wrap = Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrixWrap<ScalarT, LO, GO, Node>>(Kn_from_M1);
      auto knSM_wrap = Teuchos::rcp_dynamic_cast<Xpetra::CrsMatrixWrap<ScalarT, LO, GO, Node>>(Kn_SM);
      if (!knM1_wrap.is_null() && !knSM_wrap.is_null()) {
        using LA_Vector2 = Tpetra::Vector<ScalarT,LO,GO,Node>;
        auto knM1_op = Teuchos::rcp_dynamic_cast<Xpetra::TpetraCrsMatrix<ScalarT,LO,GO,Node>>(knM1_wrap->getCrsMatrix());
        auto knSM_op = Teuchos::rcp_dynamic_cast<Xpetra::TpetraCrsMatrix<ScalarT,LO,GO,Node>>(knSM_wrap->getCrsMatrix());
        if (!knM1_op.is_null() && !knSM_op.is_null()) {
          auto knM1_tp = knM1_op->getTpetra_CrsMatrix();
          auto knSM_tp = knSM_op->getTpetra_CrsMatrix();
          Teuchos::RCP<LA_Vector2> diagM1 = Teuchos::rcp(new LA_Vector2(knM1_tp->getRowMap(), true));
          knM1_tp->getLocalDiagCopy(*diagM1);
          auto dM1_h = diagM1->getLocalViewHost(Tpetra::Access::ReadOnly);
          double diagMaxInf = 0.0;
          const LO nrLocal = static_cast<LO>(knM1_tp->getRowMap()->getLocalNumElements());
          for (LO r = 0; r < nrLocal; ++r) {
            const double a = std::abs(dM1_h(r, 0));
            if (a > diagMaxInf) diagMaxInf = a;
          }
          double diagMaxG = 0.0;
          Teuchos::reduceAll<int,double>(*(J->getComm()), Teuchos::REDUCE_MAX, 1, &diagMaxInf, &diagMaxG);
          auto BCd_host = Kokkos::create_mirror_view(BCd_sm);
          Kokkos::deep_copy(BCd_host, BCd_sm);
          using host_inds2 = typename Tpetra::CrsMatrix<ScalarT,LO,GO,Node>::nonconst_local_inds_host_view_type;
          using host_vals2 = typename Tpetra::CrsMatrix<ScalarT,LO,GO,Node>::nonconst_values_host_view_type;
          double maxDiff = 0.0;
          size_t nCompared = 0;
          // Compare each row by merging its sorted column indices.
          for (LO r = 0; r < nrLocal; ++r) {
            if (BCd_host(r)) continue;
            size_t nentM1 = knM1_tp->getNumEntriesInLocalRow(r);
            size_t nentSM = knSM_tp->getNumEntriesInLocalRow(r);
            if (nentM1 == 0 || nentSM == 0) continue;
            host_inds2 cM("kncmp_cM", nentM1), cS("kncmp_cS", nentSM);
            host_vals2 vM("kncmp_vM", nentM1), vS("kncmp_vS", nentSM);
            knM1_tp->getLocalRowCopy(r, cM, vM, nentM1);
            knSM_tp->getLocalRowCopy(r, cS, vS, nentSM);
            size_t iM = 0, iS = 0;
            while (iM < nentM1) {
              const double vMabs = std::abs(vM(iM));
              double vSabs = 0.0;
              while (iS < nentSM && cS(iS) < cM(iM)) ++iS;
              if (iS < nentSM && cS(iS) == cM(iM)) { vSabs = std::abs(vS(iS)); ++iS; }
              const double d = std::abs(vMabs - vSabs);
              if (d > maxDiff) maxDiff = d;
              ++nCompared;
              ++iM;
            }
          }
          double maxDiffG = 0.0;
          Teuchos::reduceAll<int,double>(*(J->getComm()), Teuchos::REDUCE_MAX, 1, &maxDiff, &maxDiffG);
          size_t nCompG = 0;
          Teuchos::reduceAll<int,size_t>(*(J->getComm()), Teuchos::REDUCE_SUM, 1, &nCompared, &nCompG);
          if (verbosity >= 6 && J->getComm()->getRank() == 0) {
            std::cout << "[Maxwell1 verify Kn] non-BC entries compared=" << nCompG
                      << " max|Kn_M1 - Kn_SM|=" << maxDiffG
                      << " max|diag Kn_M1|=" << diagMaxG << std::endl;
            if (diagMaxG > 0.0 && maxDiffG > 1e-10 * diagMaxG) {
              std::cout << "[Maxwell1 verify Kn] WARN: diff exceeds 1e-10 * max|diag|; "
                        << "Kn_from_M1 and Kn_from_SM disagree on curl-curl inclusion." << std::endl;
            }
          }
        }
      }
    }
  }

  if (canReuse) {
    precCache->resetMatrix(SM_wrap);
    if (verbosity >= 10 && J->getComm()->getRank() == 0) {
      std::cout << "[Maxwell1] Reusing existing hierarchy with resetMatrix()" << (forSchur ? " (Schur)" : "") << std::endl;
    }
  }
  else {
    precCache = Teuchos::rcp(new Maxwell1Type(
        SM_wrap, D0_wrap, Kn_from_M1, nullspace_xpetra, coords_xpetra,
        maxwell1Params, true));
    if (verbosity >= 10 && J->getComm()->getRank() == 0) {
      std::cout << "[Maxwell1] Built new preconditioner hierarchy"
                << (useKnFromM1 ? " with Kn from M1" : "")
                << (forSchur ? " (Schur)" : "") << std::endl;
    }
  }

  return Teuchos::rcp(new MueLu::TpetraOperator<ScalarT, LO, GO, Node>(
      Teuchos::rcp_static_cast<XpetraOperator>(precCache)));
}

