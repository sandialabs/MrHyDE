/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

#include "block_prec/ParamUtils.hpp"
#include "block_prec/BlockAssembly.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>
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
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (cntxt->use_preconditioner) {
      Teuchos::RCP<LA_Operator> preconditioner = this->buildOrUpdatePreconditioner(cntxt, J);
      this->attachPreconditionerToProblem(cntxt, Problem, preconditioner);
    }
    
    Problem->setProblem();
    
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList(cntxt);
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver =
      this->createBelosSolverManager(Problem, belosList, cntxt->belos_type);
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
  this->linearSolver(context_L2[set],J,r,soln);
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverBoundaryL2(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
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
      stripContextAndMethodKeys(filteredParams);
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
    ? block_prec::detail::buildLumpedM0inv<Node>(cntxt->refMaxwell.D0_matrix, M1_use, nodal_map, edge_map)
    : block_prec::detail::buildM0invIdentity<Node>(nodal_map);

  block_prec::detail::RefMaxwellXpetraInputs<Node> xpetraInputs = block_prec::detail::buildRefMaxwellXpetraInputs<Node>(
    J, cntxt->refMaxwell.D0_matrix, M1_use, M0inv, cntxt->refMaxwell.nodal_coords, cntxt->refMaxwell.nullspace);
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

  // Require XML parameter file for RefMaxwell
  TEUCHOS_TEST_FOR_EXCEPTION(cntxt->refMaxwell.xml_param_file.empty(), std::runtime_error,
    "RefMaxwell preconditioner requires 'xml param file' to be specified in 'RefMaxwell Settings'. "
    << "XML files provide complete MueLu RefMaxwell configuration. See regression tests for examples.");

  Teuchos::ParameterList refmaxwellParams;

  // Load parameters from XML file
  try {
    refmaxwellParams = *Teuchos::getParametersFromXmlFile(cntxt->refMaxwell.xml_param_file);
    if (verbosity >= 6 && J->getComm()->getRank() == 0) {
      std::cout << "[RefMaxwell] Loaded parameters from XML file: " << cntxt->refMaxwell.xml_param_file << std::endl;
    }
  } catch (const std::exception& e) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "Failed to load RefMaxwell parameters from XML file '" << cntxt->refMaxwell.xml_param_file
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

  // Check if we can reuse an existing RefMaxwell hierarchy (pivot or Schur cache per forSchur).
  using XpetraOperator = Xpetra::Operator<ScalarT, LO, GO, Node>;
  std::string reuseType = cntxt->preconditioner_reuse_type;
  for (size_t i = 0; i < reuseType.size(); ++i) {
    reuseType[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(reuseType[i])));
  }
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

