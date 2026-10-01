/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

#include "block_prec/ParamUtils.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

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
  const bool isSchwarz = (cntxt->prec_type == "domain decomposition");
  if (isSchwarz || cntxt->prec_type == "Ifpack2") {
    // Ifpack2 keeps no hierarchy, so rebuilding against the new J is all 'update' can do.
    if (this->preconditionerNeedsRebuild(cntxt, !cntxt->prec_dd.is_null())) {
      Teuchos::ParameterList & ifpackList = cntxt->prec_sublist;
      const string method = isSchwarz ? string("SCHWARZ")
        : settings->sublist("Solver").get("preconditioner variant","RELAXATION");
      if (isSchwarz) {
        ifpackList.set("schwarz: subdomain solver","garbage");
      }
      else if (verbosity >= 15 && comm->getRank() == 0) {
        std::cout << "Preconditioner parameters (monolithic Ifpack2 " << method << "):" << std::endl;
        ifpackList.print(std::cout);
      }
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
    if (this->preconditionerNeedsRebuild(cntxt, !cntxt->prec_block.is_null())) {
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
    if (cntxt->preconditioner_reuse_type == "none" || !cntxt->have_preconditioner ||
        cntxt->prec.is_null()) {
      cntxt->prec = this->buildAMGPreconditioner(J, cntxt);
      cntxt->have_preconditioner = true;
    }
    else if (!block_prec::reuseKeepsOperator(cntxt->preconditioner_reuse_type,
                                 cntxt->jacobian_rebuilt_this_step)) {
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
  const std::string belosUpper = block_prec::toUpperAsciiCopy(belosType);
  // The Belos solver factory registers neither of these, so construct them directly.
  if (belosUpper == "PSEUDO BLOCK STOCHASTIC CG") {
    return Teuchos::rcp(new Belos::PseudoBlockStochasticCGSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  if (belosUpper == "RCG") {
    return Teuchos::rcp(new Belos::RCGSolMgr<ScalarT,BelosMV,LA_Operator>(problem, belosList));
  }
  Belos::SolverFactory<ScalarT,BelosMV,LA_Operator> factory;
  Teuchos::RCP<Belos::SolverManager<ScalarT,BelosMV,LA_Operator> > solver =
    factory.create(belosType, belosList);
  solver->setProblem(problem);
  return solver;
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
  if (block_prec::toUpperAsciiCopy(cntxt->belos_type) != "PSEUDO BLOCK CG") return;
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

  if (!cntxt->amg.xml_param_file.empty()) {
    block_prec::loadXmlBroadcast(cntxt->amg.xml_param_file, mueluParams, *J->getComm(), "AMG");
    if (verbosity >= 6 && J->getComm()->getRank() == 0) {
      std::cout << "[AMG] Loaded parameters from XML file: "
                << cntxt->amg.xml_param_file << std::endl;
    }
  } else {
    mueluParams = block_prec::defaultMueLuParams();

    if (cntxt->prec_sublist.name() != "empty" ) {
      Teuchos::ParameterList filteredParams(cntxt->prec_sublist);
      block_prec::removeMrHyDEOwnedKeys(filteredParams);
      block_prec::removeIfpack2OnlyKeys(filteredParams);
      mueluParams.setParameters(filteredParams);
    }
    if (cntxt->prec_sublist.name() == "empty" ) {
      block_prec::setDefaultChebyshevSmoother(mueluParams, false);
    }
  }

  block_prec::normalizeMueLuVerbosity(mueluParams, verbosity);

  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > Mnew = MueLu::CreateTpetraPreconditioner((Teuchos::RCP<LA_Operator>)J, mueluParams);

  return Mnew;
}
