/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/


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
      if (cntxt->prec_type == "domain decomposition") {
        if (!cntxt->reuse_preconditioner || !cntxt->have_preconditioner) {
          Teuchos::ParameterList & ifpackList = cntxt->prec_sublist;//settings->sublist("Solver").sublist("Ifpack2");
          ifpackList.set("schwarz: subdomain solver","garbage");
          cntxt->prec_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
          cntxt->prec_dd->setParameters(ifpackList);
          cntxt->prec_dd->initialize();
          cntxt->prec_dd->compute();
          cntxt->have_preconditioner = true;
        }
        if (cntxt->right_preconditioner) {
          Problem->setRightPrec(cntxt->prec_dd);
        }
        else {
          Problem->setLeftPrec(cntxt->prec_dd);
        }
      }
      else if (cntxt->prec_type == "Ifpack2") {
        if (!cntxt->reuse_preconditioner || !cntxt->have_preconditioner) {
          Teuchos::ParameterList & ifpackList = cntxt->prec_sublist;//settings->sublist("Solver").sublist(cntxt->prec_sublist);
          string method = settings->sublist("Solver").get("preconditioner variant","RELAXATION");
          // TMW: keeping these here for reference, but these can be set from input file
          //ifpackList.set("relaxation: type","Symmetric Gauss-Seidel");
          //ifpackList.set("relaxation: sweeps",1);
          //ifpackList.set("chebyshev: degree",2);
          //cntxt->M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("RELAXATION", J);
          //cntxt->M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("CHEBYSHEV", J);
          //ifpackList.set("fact: iluk level-of-fill",0);
          //ifpackList.set("fact: ilut level-of-fill",1.0);
          //ifpackList.set("fact: absolute threshold",0.0);
          //ifpackList.set("fact: relative threshold",1.0);
          //ifpackList.set("fact: relax value",0.0);
          //cntxt->M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("RILUK", J);
          cntxt->prec_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > (method, J);
          cntxt->prec_dd->setParameters(ifpackList);
          cntxt->prec_dd->initialize();
          cntxt->prec_dd->compute();
          cntxt->have_preconditioner = true;
        }
        
        if (cntxt->right_preconditioner) {
          Problem->setRightPrec(cntxt->prec_dd);
        }
        else {
          Problem->setLeftPrec(cntxt->prec_dd);
        }
      }
      else { // default - AMG preconditioner
        if (!cntxt->reuse_preconditioner || !cntxt->have_preconditioner) {
          cntxt->prec = this->buildAMGPreconditioner(J,cntxt);
          cntxt->have_preconditioner = true;
        }
        else {
          MueLu::ReuseTpetraPreconditioner(J,*(cntxt->prec));
        }
        if (cntxt->right_preconditioner) {
          Problem->setRightPrec(cntxt->prec);
        }
        else {
          Problem->setLeftPrec(cntxt->prec);
        }
      }
    }
    
    Problem->setProblem();
    
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList(cntxt);
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver;
    if (cntxt->belos_type == "Block GMRES" || cntxt->belos_type == "Block Gmres") {
      solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "Block CG") {
      solver = Teuchos::rcp(new Belos::BlockCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "BiCGStab") {
      // Requires right preconditioning
      solver = Teuchos::rcp(new Belos::BiCGStabSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "GCRODR") {
      solver = Teuchos::rcp(new Belos::GCRODRSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "PCPG") {
      solver = Teuchos::rcp(new Belos::PCPGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "Pseudo Block CG") {
      solver = Teuchos::rcp(new Belos::PseudoBlockCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "Pseudo Block Gmres" || cntxt->belos_type == "Pseudo Block GMRES") {
      solver = Teuchos::rcp(new Belos::PseudoBlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "Pseudo Block Stochastic CG") {
      solver = Teuchos::rcp(new Belos::PseudoBlockStochasticCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "Pseudo Block TFQMR") {
      solver = Teuchos::rcp(new Belos::PseudoBlockTFQMRSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "RCG") {
      solver = Teuchos::rcp(new Belos::RCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (cntxt->belos_type == "TFQMR") {
      solver = Teuchos::rcp(new Belos::TFQMRSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Belos solver: " + cntxt->belos_type);
    }
    // Minres and LSQR fail a simple test
    
    solver->solve();

    if(doCondEst && cntxt->belos_type == "Pseudo Block CG") {
      Teuchos::RCP<Belos::PseudoBlockCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>> solver_cg = Teuchos::rcp_dynamic_cast<Belos::PseudoBlockCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>>(solver);
      if(comm->getRank() == 0) {
        std::cout << "Belos condition number estimate = " << solver_cg->getConditionEstimate() << std::endl;
      }
    }
    
  }
  
}
// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolver(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
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

  // MrHyDE default settings
  mueluParams.set("verbosity","none");
  mueluParams.set("coarse: max size",500);
  mueluParams.set("multigrid algorithm", "sa");
  
  // Aggregation
  mueluParams.set("aggregation: type","uncoupled");
  mueluParams.set("aggregation: drop scheme","classical");
  
  //Smoothing
  mueluParams.set("smoother: type","CHEBYSHEV");
  
  // Repartitioning
  mueluParams.set("repartition: enable",false);
  
  // Reuse
  mueluParams.set("reuse: type","none");
  
  // if the user provides a "Preconditioner Settings" sublist, use it for MueLu
  // otherwise, set things with the simple approach
  if (cntxt->prec_sublist.name() != "empty" ) {
    mueluParams.setParameters(cntxt->prec_sublist);
  }
  else { // safe to define defaults for chebyshev smoother
    mueluParams.sublist("smoother: params").set("chebyshev: degree",2);
    mueluParams.sublist("smoother: params").set("chebyshev: ratio eigenvalue",7.0);
    mueluParams.sublist("smoother: params").set("chebyshev: min eigenvalue",1.0);
    mueluParams.sublist("smoother: params").set("chebyshev: zero starting solution",true);
  }
  
  if (verbosity >= 20){
    mueluParams.set("verbosity","high");
  }
  mueluParams.setName("MueLu");
  
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > Mnew = MueLu::CreateTpetraPreconditioner((Teuchos::RCP<LA_Operator>)J, mueluParams);
  
  return Mnew;
}

