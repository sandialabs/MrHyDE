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
void LinearAlgebraInterface<Node>::linearSolver(Teuchos::RCP<LinearSolverOptions<Node> > & opt,
                                                matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {

  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (opt->use_direct) {
    if (!opt->have_symb_factor) {
      opt->amesos_solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>(opt->amesos_type, J, r, soln);
      opt->amesos_solver->symbolicFactorization();
      opt->have_symb_factor = true;
    }
    opt->amesos_solver->setA(J, Amesos2::SYMBFACT);
    opt->amesos_solver->setX(soln);
    opt->amesos_solver->setB(r);
    opt->amesos_solver->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (opt->use_preconditioner) {
      if (opt->prec_type == "domain decomposition") {
        if (!opt->reuse_preconditioner || !opt->have_preconditioner) {
          Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
          ifpackList.set("schwarz: subdomain solver","garbage");
          opt->prec_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
          opt->prec_dd->setParameters(ifpackList);
          opt->prec_dd->initialize();
          opt->prec_dd->compute();
          opt->have_preconditioner = true;
        }
        if (opt->right_preconditioner) {
          Problem->setRightPrec(opt->prec_dd);
        }
        else {
          Problem->setLeftPrec(opt->prec_dd);
        }
      }
      else if (opt->prec_type == "Ifpack2") {
        if (!opt->reuse_preconditioner || !opt->have_preconditioner) {
          Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist(opt->prec_sublist);
          string method = settings->sublist("Solver").get("preconditioner variant","RELAXATION");
          // TMW: keeping these here for reference, but these can be set from input file
          //ifpackList.set("relaxation: type","Symmetric Gauss-Seidel");
          //ifpackList.set("relaxation: sweeps",1);
          //ifpackList.set("chebyshev: degree",2);
          //opt->M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("RELAXATION", J);
          //opt->M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("CHEBYSHEV", J);
          //ifpackList.set("fact: iluk level-of-fill",0);
          //ifpackList.set("fact: ilut level-of-fill",1.0);
          //ifpackList.set("fact: absolute threshold",0.0);
          //ifpackList.set("fact: relative threshold",1.0);
          //ifpackList.set("fact: relax value",0.0);
          //opt->M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("RILUK", J);
          opt->prec_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > (method, J);
          opt->prec_dd->setParameters(ifpackList);
          opt->prec_dd->initialize();
          opt->prec_dd->compute();
          opt->have_preconditioner = true;
        }
        
        if (opt->right_preconditioner) {
          Problem->setRightPrec(opt->prec_dd);
        }
        else {
          Problem->setLeftPrec(opt->prec_dd);
        }
      }
      else { // default - AMG preconditioner
        if (!opt->reuse_preconditioner || !opt->have_preconditioner) {
          opt->prec = this->buildPreconditioner(J,opt->prec_sublist);
          opt->have_preconditioner = true;
        }
        else {
          MueLu::ReuseTpetraPreconditioner(J,*(opt->prec));
        }
        if (opt->right_preconditioner) {
          Problem->setRightPrec(opt->prec);
        }
        else {
          Problem->setLeftPrec(opt->prec);
        }
      }
    }
    
    Problem->setProblem();
    
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList(opt->belos_sublist);
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver;
    if (opt->belos_type == "Block GMRES" || opt->belos_type == "Block Gmres") {
      solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "Block CG") {
      solver = Teuchos::rcp(new Belos::BlockCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "BiCGStab") {
      // Requires right preconditioning
      solver = Teuchos::rcp(new Belos::BiCGStabSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "GCRODR") {
      solver = Teuchos::rcp(new Belos::GCRODRSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "PCPG") {
      solver = Teuchos::rcp(new Belos::PCPGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "Pseudo Block CG") {
      solver = Teuchos::rcp(new Belos::PseudoBlockCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "Pseudo Block Gmres" || opt->belos_type == "Pseudo Block GMRES") {
      solver = Teuchos::rcp(new Belos::PseudoBlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "Pseudo Block Stochastic CG") {
      solver = Teuchos::rcp(new Belos::PseudoBlockStochasticCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "Pseudo Block TFQMR") {
      solver = Teuchos::rcp(new Belos::PseudoBlockTFQMRSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "RCG") {
      solver = Teuchos::rcp(new Belos::RCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belos_type == "TFQMR") {
      solver = Teuchos::rcp(new Belos::TFQMRSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Belos solver: " + opt->belos_type);
    }
    // Minres and LSQR fail a simple test
    
    solver->solve();

    if(doCondEst && opt->belos_type == "Pseudo Block CG") {
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
  this->linearSolver(options[set],J,r,soln);
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverL2(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  this->linearSolver(options_L2[set],J,r,soln);
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverBoundaryL2(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  this->linearSolver(options_BndryL2[set],J,r,soln);
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverParam(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  this->linearSolver(options_param,J,r,soln);
}

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  this->linearSolver(options_param_L2,J,r,soln);
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverBoundaryL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  this->linearSolver(options_param_BndryL2,J,r,soln);
}

// ========================================================================================
// Preconditioner for Tpetra stack
// ========================================================================================

template<class Node>
Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > LinearAlgebraInterface<Node>::buildPreconditioner(const matrix_RCP & J, const string & precSublist) {
  
  Teuchos::TimeMonitor localtimer(*prectimer);

  Teuchos::ParameterList mueluParams;

  string xmlFileName = settings->sublist("Solver").get<string>("Preconditioner xml","");

  // If there's no xml file, then we'll set the defaults and allow for shortlist additions in the input file
  if(xmlFileName == "") {
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
    if(settings->sublist("Solver").isSublist(precSublist)) {
      Teuchos::ParameterList inputPrecParams = settings->sublist("Solver").sublist(precSublist);
      mueluParams.setParameters(inputPrecParams);
    }
    else { // safe to define defaults for chebyshev smoother
      mueluParams.sublist("smoother: params").set("chebyshev: degree",2);
      mueluParams.sublist("smoother: params").set("chebyshev: ratio eigenvalue",7.0);
      mueluParams.sublist("smoother: params").set("chebyshev: min eigenvalue",1.0);
      mueluParams.sublist("smoother: params").set("chebyshev: zero starting solution",true);
    }
  }
  else { // If the "Preconditioner xml" option is specified, don't set any defaults and only use the xml
    Teuchos::updateParametersFromXmlFile(xmlFileName, Teuchos::Ptr<Teuchos::ParameterList>(&mueluParams));
  }

  if (verbosity >= 20){
    mueluParams.set("verbosity","high");
  }
  mueluParams.setName("MueLu");
  
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > Mnew = MueLu::CreateTpetraPreconditioner((Teuchos::RCP<LA_Operator>)J, mueluParams);
  
  return Mnew;
}

