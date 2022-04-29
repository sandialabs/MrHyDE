/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "linearAlgebraInterface.hpp"
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBiCGStabSolMgr.hpp>
#include <BelosGCRODRSolMgr.hpp>
#include <BelosPCPGSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosPseudoBlockStochasticCGSolMgr.hpp>
#include <BelosPseudoBlockTFQMRSolMgr.hpp>
#include <BelosRCGSolMgr.hpp>
#include <BelosTFQMRSolMgr.hpp>

using namespace MrHyDE;

// ========================================================================================
// Constructor  
// ========================================================================================

template<class Node>
LinearAlgebraInterface<Node>::LinearAlgebraInterface(const Teuchos::RCP<MpiComm> & Comm_,
                                                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                                     Teuchos::RCP<DiscretizationInterface> & disc_,
                                                     Teuchos::RCP<ParameterManager<Node> > & params_) :
Comm(Comm_), settings(settings_), disc(disc_), params(params_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting linear algebra interface constructor ..." << endl;
    }
  }
  
  verbosity = settings->get<int>("verbosity",0);
  
  setnames = disc->phys->setnames;
  
  // Generic Belos Settings - can be overridden by defining Belos sublists
  linearTOL = settings->sublist("Solver").get<double>("linear TOL",1.0E-7);
  maxLinearIters = settings->sublist("Solver").get<int>("max linear iters",100);
  maxKrylovVectors = settings->sublist("Solver").get<int>("krylov vectors",100);
  belos_residual_scaling = settings->sublist("Solver").get<string>("Belos implicit residual scaling","None");
  // Also: "Norm of Preconditioned Initial Residual" or "Norm of Initial Residual"
  
  
  // Create the solver options for the state Jacobians
  for (size_t set=0; set<setnames.size(); ++set) {
    Teuchos::ParameterList solvesettings;
    if (settings->sublist("Solver").isSublist("State linear solver")) { // for detailed control
      solvesettings = settings->sublist("Solver").sublist("State linear solver");
    }
    else { // use generic options
      solvesettings = settings->sublist("Solver");
    }
    options.push_back(Teuchos::rcp( new SolverOptions<Node>(solvesettings) ));
  }
  
  // Create the solver options for the state L2-projections
  for (size_t set=0; set<setnames.size(); ++set) {
    Teuchos::ParameterList solvesettings;
    if (settings->sublist("Solver").isSublist("State L2 linear solver")) { // for detailed control
      solvesettings = settings->sublist("Solver").sublist("State L2 linear solver");
    }
    else { // use generic options
      solvesettings = settings->sublist("Solver");
    }
    options_L2.push_back(Teuchos::rcp( new SolverOptions<Node>(solvesettings) ));
  }
  
  // Create the solver options for the state boundary L2-projections
  for (size_t set=0; set<setnames.size(); ++set) {
    Teuchos::ParameterList solvesettings;
    if (settings->sublist("Solver").isSublist("State boundary L2 linear solver")) { // for detailed control
      solvesettings = settings->sublist("Solver").sublist("State boundary L2 linear solver");
    }
    else { // use generic options
      solvesettings = settings->sublist("Solver");
    }
    solvesettings.set("use preconditioner",false);
    options_BndryL2.push_back(Teuchos::rcp( new SolverOptions<Node>(solvesettings) ));
  }
  
  // Create the solver options for the discretized parameter Jacobians
  {
    Teuchos::ParameterList solvesettings;
    if (settings->sublist("Solver").isSublist("Parameter linear solver")) { // for detailed control
      solvesettings = settings->sublist("Solver").sublist("Parameter linear solver");
    }
    else { // use generic options
      solvesettings = settings->sublist("Solver");
    }
    options_param = Teuchos::rcp( new SolverOptions<Node>(solvesettings) );
  }
  
  // Create the solver options for the discretized parameter L2-projections
  {
    Teuchos::ParameterList solvesettings;
    if (settings->sublist("Solver").isSublist("Parameter L2 linear solver")) { // for detailed control
      solvesettings = settings->sublist("Solver").sublist("Parameter L2 linear solver");
    }
    else { // use generic options
      solvesettings = settings->sublist("Solver");
    }
    options_param_L2 = Teuchos::rcp( new SolverOptions<Node>(solvesettings) );
  }
  
  // Create the solver options for the discretized parameter boundary L2-projections
  {
    Teuchos::ParameterList solvesettings;
    if (settings->sublist("Solver").isSublist("Parameter boundary L2 linear solver")) { // for detailed control
      solvesettings = settings->sublist("Solver").sublist("Parameter boundary L2 linear solver");
    }
    else { // use generic options
      solvesettings = settings->sublist("Solver");
    }
    options_param_BndryL2 = Teuchos::rcp( new SolverOptions<Node>(solvesettings) );
  }
  
  this->setupLinearAlgebra();
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished linear algebra interface constructor" << endl;
    }
  }
  
}

// ========================================================================================
// Set up the Tpetra objects (maps, importers, exporters and graphs)
// This is a separate function call in case it needs to be recomputed
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::setupLinearAlgebra() {
  
  Teuchos::TimeMonitor localtimer(*setupLAtimer);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::setupLinearAlgebraInterface..." << endl;
    }
  }
  
  std::vector<string> blocknames = disc->blocknames;
  
  // --------------------------------------------------
  // primary variable LA objects
  // --------------------------------------------------
  maxEntries = 0;
  
  for (size_t set=0; set<setnames.size(); ++set) {
    vector<GO> owned, ownedAndShared;
    owned = disc->DOF_owned[set];
    ownedAndShared = disc->DOF_ownedAndShared[set];
    //disc->DOF[set]->getOwnedIndices(owned);
    LO numUnknowns = (LO)owned.size();
    //disc->DOF[set]->getOwnedAndGhostedIndices(ownedAndShared);
    GO localNumUnknowns = numUnknowns;
    
    GO globalNumUnknowns = 0;
    Teuchos::reduceAll<LO,GO>(*Comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
    
    owned_map.push_back(Teuchos::rcp(new LA_Map(globalNumUnknowns, owned, 0, Comm)));
    overlapped_map.push_back(Teuchos::rcp(new LA_Map(globalNumUnknowns, ownedAndShared, 0, Comm)));
    
    exporter.push_back(Teuchos::rcp(new LA_Export(overlapped_map[set], owned_map[set])));
    importer.push_back(Teuchos::rcp(new LA_Import(owned_map[set], overlapped_map[set])));
    
    bool allocate_matrices = true;
    if (settings->sublist("Solver").get<bool>("fully explicit",false) ) {
      allocate_matrices = false;
    }
    
    if (allocate_matrices) {
      vector<size_t> maxEntriesPerRow(overlapped_map[set]->getLocalNumElements(), 0);
      for (size_t b=0; b<blocknames.size(); b++) {
        vector<size_t> EIDs = disc->myElements[b];
        for (size_t e=0; e<EIDs.size(); e++) {
          size_t elemID = EIDs[e];
          vector<GO> gids = disc->getGIDs(set,b,elemID); //
          //disc->DOF[set]->getElementGIDs(elemID, gids, blocknames[b]);
          for (size_t i=0; i<gids.size(); i++) {
            LO ind1 = overlapped_map[set]->getLocalElement(gids[i]);
            maxEntriesPerRow[ind1] += gids.size();
          }
        }
      }
      
      size_t curr_maxEntries = 0;
      for (size_t m=0; m<maxEntriesPerRow.size(); ++m) {
        curr_maxEntries = std::max(curr_maxEntries, maxEntriesPerRow[m]);
      }
      
      //curr_maxEntries = static_cast<size_t>(settings->sublist("Solver").get<int>("max entries per row",
      //                                                                      static_cast<int>(curr_maxEntries)));
      maxEntries = std::max(maxEntries,curr_maxEntries);
      
      overlapped_graph.push_back(Teuchos::rcp(new LA_CrsGraph(overlapped_map[set],
                                                              curr_maxEntries)));
    
      for (size_t b=0; b<blocknames.size(); b++) {
        vector<size_t> EIDs = disc->myElements[b];
        for (size_t e=0; e<EIDs.size(); e++) {
          size_t elemID = EIDs[e];
          vector<GO> gids = disc->getGIDs(set,b,elemID);
          //disc->DOF[set]->getElementGIDs(elemID, gids, blocknames[b]);
          for (size_t i=0; i<gids.size(); i++) {
            GO ind1 = gids[i];
            overlapped_graph[set]->insertGlobalIndices(ind1,gids);
          }
        }
      }
      
      overlapped_graph[set]->fillComplete();
      
      matrix.push_back(Teuchos::rcp(new LA_CrsMatrix(owned_map[set], curr_maxEntries)));
      
      overlapped_matrix.push_back(Teuchos::rcp(new LA_CrsMatrix(overlapped_graph[set])));
      
      this->fillComplete(matrix[set]);
      this->fillComplete(overlapped_matrix[set]);
    }
  }
  
  // --------------------------------------------------
  // discretized parameter LA objects
  // --------------------------------------------------
  
  {
    if (params->num_discretized_params > 0) {
      
      vector<GO> param_owned, param_ownedAndShared;
      
      params->paramDOF->getOwnedIndices(param_owned);
      LO numUnknowns = (LO)param_owned.size();
      params->paramDOF->getOwnedAndGhostedIndices(param_ownedAndShared);
      GO localNumUnknowns = numUnknowns;
      
      GO globalNumUnknowns = 0;
      Teuchos::reduceAll<LO,GO>(*Comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
      
      param_owned_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, param_owned, 0, Comm));
      param_overlapped_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, param_ownedAndShared, 0, Comm));
      
      param_exporter = Teuchos::rcp(new LA_Export(param_overlapped_map, param_owned_map));
      param_importer = Teuchos::rcp(new LA_Import(param_overlapped_map, param_owned_map));
      
      param_overlapped_graph = Teuchos::rcp( new LA_CrsGraph(param_overlapped_map,maxEntries));
      for (size_t b=0; b<blocknames.size(); b++) {
        vector<size_t> EIDs = disc->myElements[b];
        for (size_t e=0; e<EIDs.size(); e++) {
          vector<GO> gids;
          size_t elemID = EIDs[e];
          params->paramDOF->getElementGIDs(elemID, gids, blocknames[b]);
          vector<GO> stategids = disc->getGIDs(0,b,elemID);
          // TMW: warning - this is hard coded to one physics set
          //disc->DOF[0]->getElementGIDs(elemID, stategids, blocknames[b]);
          for (size_t i=0; i<gids.size(); i++) {
            GO ind1 = gids[i];
            param_overlapped_graph->insertGlobalIndices(ind1,stategids);
          }
        }
      }
      
      param_overlapped_graph->fillComplete(owned_map[0], param_owned_map); // hard coded
      
    }
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::setupLinearAlgebraInterface" << endl;
    }
  }
  
}

// ========================================================================================
// All iterative solvers use the same Belos list.  This would be easy to specialize.
// ========================================================================================

template<class Node>
Teuchos::RCP<Teuchos::ParameterList> LinearAlgebraInterface<Node>::getBelosParameterList(const string & belosSublist) {
  Teuchos::RCP<Teuchos::ParameterList> belosList = Teuchos::rcp(new Teuchos::ParameterList());
  belosList->set("Maximum Iterations",    maxLinearIters); // Maximum number of iterations allowed
  //belosList->set("Num Blocks",    1); //maxLinearIters);
  belosList->set("Convergence Tolerance", linearTOL);    // Relative convergence tolerance requested
  if (verbosity > 9) {
    belosList->set("Verbosity", Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
  }
  else {
    belosList->set("Verbosity", Belos::Errors);
  }
  if (verbosity > 8) {
    belosList->set("Output Frequency",10);
  }
  else {
    belosList->set("Output Frequency",0);
  }
  int numEqns = 1;
  if (disc->blocknames.size() == 1) {
    numEqns = disc->phys->numVars[0][0];
  }
  belosList->set("number of equations", numEqns);
  
  belosList->set("Output Style", Belos::Brief);
  belosList->set("Implicit Residual Scaling", belos_residual_scaling);
  
  if (settings->sublist("Solver").isSublist(belosSublist)) {
    Teuchos::ParameterList inputParams = settings->sublist("Solver").sublist(belosSublist);
    belosList->setParameters(inputParams);
  }
  
  return belosList;
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolver(Teuchos::RCP<SolverOptions<Node> > & opt,
                                                matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {

  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (opt->useDirect) {
    if (!opt->haveSymbFactor) {
      opt->AmesosSolver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>(opt->amesosType, J, r, soln);
      opt->AmesosSolver->symbolicFactorization();
      opt->haveSymbFactor = true;
    }
    opt->AmesosSolver->setA(J, Amesos2::SYMBFACT);
    opt->AmesosSolver->setX(soln);
    opt->AmesosSolver->setB(r);
    opt->AmesosSolver->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (opt->usePreconditioner) {
      if (opt->precType == "domain decomposition") {
        if (!opt->reusePreconditioner || !opt->havePreconditioner) {
          Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
          ifpackList.set("schwarz: subdomain solver","garbage");
          opt->M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
          opt->M_dd->setParameters(ifpackList);
          opt->M_dd->initialize();
          opt->M_dd->compute();
          opt->havePreconditioner = true;
        }
        if (opt->rightPreconditioner) {
          Problem->setRightPrec(opt->M_dd);
        }
        else {
          Problem->setLeftPrec(opt->M_dd);
        }
      }
      else if (opt->precType == "Ifpack2") {
        if (!opt->reusePreconditioner || !opt->havePreconditioner) {
          Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist(opt->precSublist);
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
          opt->M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > (method, J);
          opt->M_dd->setParameters(ifpackList);
          opt->M_dd->initialize();
          opt->M_dd->compute();
          opt->havePreconditioner = true;
        }
        
        if (opt->rightPreconditioner) {
          Problem->setRightPrec(opt->M_dd);
        }
        else {
          Problem->setLeftPrec(opt->M_dd);
        }
      }
      else { // default - AMG preconditioner
        if (!opt->reusePreconditioner || !opt->havePreconditioner) {
          opt->M = this->buildPreconditioner(J,opt->precSublist);
          opt->havePreconditioner = true;
        }
        else {
          MueLu::ReuseTpetraPreconditioner(J,*(opt->M));
        }
        if (opt->rightPreconditioner) {
          Problem->setRightPrec(opt->M);
        }
        else {
          Problem->setLeftPrec(opt->M);
        }
      }
    }
    
    Problem->setProblem();
    
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList(opt->belosSublist);
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver;
    if (opt->belosType == "Block GMRES" || opt->belosType == "Block Gmres") {
      solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "Block CG") {
      solver = Teuchos::rcp(new Belos::BlockCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "BiCGStab") {
      // Requires right preconditioning
      solver = Teuchos::rcp(new Belos::BiCGStabSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "GCRODR") {
      solver = Teuchos::rcp(new Belos::GCRODRSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "PCPG") {
      solver = Teuchos::rcp(new Belos::PCPGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "Pseudo Block CG") {
      solver = Teuchos::rcp(new Belos::PseudoBlockCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "Pseudo Block Gmres" || opt->belosType == "Pseudo Block GMRES") {
      solver = Teuchos::rcp(new Belos::PseudoBlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "Pseudo Block Stochastic CG") {
      solver = Teuchos::rcp(new Belos::PseudoBlockStochasticCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "Pseudo Block TFQMR") {
      solver = Teuchos::rcp(new Belos::PseudoBlockTFQMRSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "RCG") {
      solver = Teuchos::rcp(new Belos::RCGSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else if (opt->belosType == "TFQMR") {
      solver = Teuchos::rcp(new Belos::TFQMRSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Belos solver: " + opt->belosType);
    }
    // Minres and LSQR fail a simple test
    
    solver->solve();
    
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


// ========================================================================================
// ========================================================================================

// Explicit template instantiations
template class MrHyDE::LinearAlgebraInterface<SolverNode>;
template class MrHyDE::SolverOptions<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
template class MrHyDE::LinearAlgebraInterface<SubgridSolverNode>;
template class MrHyDE::SolverOptions<SubgridSolverNode>;
#endif
