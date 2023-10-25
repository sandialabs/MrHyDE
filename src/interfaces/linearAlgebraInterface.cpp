/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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
LinearAlgebraInterface<Node>::LinearAlgebraInterface(const Teuchos::RCP<MpiComm> & comm_,
                                                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                                     Teuchos::RCP<DiscretizationInterface> & disc_,
                                                     Teuchos::RCP<ParameterManager<Node> > & params_) :
comm(comm_), settings(settings_), disc(disc_), params(params_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting linear algebra interface constructor ..." << endl;
    }
  }
  
  verbosity = settings->get<int>("verbosity",0);
  
  setnames = disc->physics->set_names;
  
  // Generic Belos Settings - can be overridden by defining Belos sublists
  linearTOL = settings->sublist("Solver").get<double>("linear TOL",1.0E-7);
  maxLinearIters = settings->sublist("Solver").get<int>("max linear iters",100);
  maxKrylovVectors = settings->sublist("Solver").get<int>("krylov vectors",100);
  belos_residual_scaling = settings->sublist("Solver").get<string>("Belos implicit residual scaling","None");
  // Also: "Norm of Preconditioned Initial Residual" or "Norm of Initial Residual"

  // Dump to file settings (false by default)
  do_dump_jacobian = settings->sublist("Solver").get<bool>("dump jacobian",false);
  do_dump_residual = settings->sublist("Solver").get<bool>("dump residual",false);
  do_dump_solution = settings->sublist("Solver").get<bool>("dump solution",false);
  
  
  // Create the solver options for the state Jacobians
  for (size_t set=0; set<setnames.size(); ++set) {
    Teuchos::ParameterList solvesettings;
    if (settings->sublist("Solver").isSublist("State linear solver")) { // for detailed control
      solvesettings = settings->sublist("Solver").sublist("State linear solver");
    }
    else { // use generic options
      solvesettings = settings->sublist("Solver");
    }
    options.push_back(Teuchos::rcp( new LinearSolverOptions<Node>(solvesettings) ));
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
    options_L2.push_back(Teuchos::rcp( new LinearSolverOptions<Node>(solvesettings) ));
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
    options_BndryL2.push_back(Teuchos::rcp( new LinearSolverOptions<Node>(solvesettings) ));
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
    options_param = Teuchos::rcp( new LinearSolverOptions<Node>(solvesettings) );
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
    options_param_L2 = Teuchos::rcp( new LinearSolverOptions<Node>(solvesettings) );
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
    options_param_BndryL2 = Teuchos::rcp( new LinearSolverOptions<Node>(solvesettings) );
  }
  
  this->setupLinearAlgebra();
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
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
    if (comm->getRank() == 0) {
      cout << "**** Starting solver::setupLinearAlgebraInterface..." << endl;
    }
  }
  
  std::vector<string> blocknames = disc->block_names;
  
  // --------------------------------------------------
  // primary variable LA objects
  // --------------------------------------------------
  max_entries = 0;
  
  for (size_t set=0; set<setnames.size(); ++set) {
    vector<GO> owned, ownedAndShared;
    owned = disc->dof_owned[set];
    ownedAndShared = disc->dof_owned_and_shared[set];
    //disc->DOF[set]->getOwnedIndices(owned);
    LO numUnknowns = (LO)owned.size();
    //disc->DOF[set]->getOwnedAndGhostedIndices(ownedAndShared);
    GO localNumUnknowns = numUnknowns;
    GO globalNumUnknowns = 0;
    Teuchos::reduceAll<LO,GO>(*comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
    
    owned_map.push_back(Teuchos::rcp(new LA_Map(globalNumUnknowns, owned, 0, comm)));
    overlapped_map.push_back(Teuchos::rcp(new LA_Map(globalNumUnknowns, ownedAndShared, 0, comm)));
    
    exporter.push_back(Teuchos::rcp(new LA_Export(overlapped_map[set], owned_map[set])));
    importer.push_back(Teuchos::rcp(new LA_Import(owned_map[set], overlapped_map[set])));
    
    bool allocate_matrices = true;
    if (settings->sublist("Solver").get<bool>("fully explicit",false) ) {
      allocate_matrices = false;
    }
    
    if (allocate_matrices) {
      vector<size_t> max_entriesPerRow(overlapped_map[set]->getLocalNumElements(), 0);
      for (size_t b=0; b<blocknames.size(); b++) {
        vector<size_t> EIDs = disc->my_elements[b];
        for (size_t e=0; e<EIDs.size(); e++) {
          size_t elemID = EIDs[e];
          vector<GO> gids = disc->getGIDs(set,b,elemID); //
          //disc->DOF[set]->getElementGIDs(elemID, gids, blocknames[b]);
          for (size_t i=0; i<gids.size(); i++) {
            LO ind1 = overlapped_map[set]->getLocalElement(gids[i]);
            max_entriesPerRow[ind1] += gids.size();
          }
        }
      }
      
      size_t curr_max_entries = 0;
      for (size_t m=0; m<max_entriesPerRow.size(); ++m) {
        curr_max_entries = std::max(curr_max_entries, max_entriesPerRow[m]);
      }
      
      //curr_max_entries = static_cast<size_t>(settings->sublist("Solver").get<int>("max entries per row",
      //                                                                      static_cast<int>(curr_max_entries)));
      max_entries = std::max(max_entries,curr_max_entries);
      
      overlapped_graph.push_back(Teuchos::rcp(new LA_CrsGraph(overlapped_map[set],
                                                              curr_max_entries)));
    
      for (size_t b=0; b<blocknames.size(); b++) {
        vector<size_t> EIDs = disc->my_elements[b];
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
      
      matrix.push_back(Teuchos::rcp(new LA_CrsMatrix(owned_map[set], curr_max_entries)));
      
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
      Teuchos::reduceAll<LO,GO>(*comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
      
      param_owned_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, param_owned, 0, comm));
      param_overlapped_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, param_ownedAndShared, 0, comm));
      
      param_exporter = Teuchos::rcp(new LA_Export(param_overlapped_map, param_owned_map));
      param_importer = Teuchos::rcp(new LA_Import(param_owned_map, param_overlapped_map));
      
      vector<size_t> max_entriesPerRow(param_overlapped_map->getLocalNumElements(), 0);
      for (size_t b=0; b<blocknames.size(); b++) {
        vector<size_t> EIDs = disc->my_elements[b];
        for (size_t e=0; e<EIDs.size(); e++) {
          size_t elemID = EIDs[e];
          vector<GO> gids;
          params->paramDOF->getElementGIDs(elemID, gids, blocknames[b]);
          vector<GO> stategids = disc->getGIDs(0,b,elemID);
          //disc->DOF[set]->getElementGIDs(elemID, gids, blocknames[b]);
          for (size_t i=0; i<gids.size(); i++) {
            LO ind1 = param_overlapped_map->getLocalElement(gids[i]);
            max_entriesPerRow[ind1] += stategids.size();
          }
        }
      }
      
      for (size_t m=0; m<max_entriesPerRow.size(); ++m) {
        max_entries = std::max(max_entries, max_entriesPerRow[m]);
      }

      param_overlapped_graph = Teuchos::rcp( new LA_CrsGraph(param_overlapped_map, overlapped_map[0], max_entries));
      for (size_t b=0; b<blocknames.size(); b++) {
        vector<size_t> EIDs = disc->my_elements[b];
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
      //param_overlapped_graph->fillComplete(overlapped_map[0], param_overlapped_map); // hard coded
      
    }
  }

  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
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
  belosList->set("Num Blocks", maxLinearIters);
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
  if (disc->block_names.size() == 1) {
    numEqns = disc->physics->num_vars[0][0];
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


// ========================================================================================
// ========================================================================================

// Explicit template instantiations
template class MrHyDE::LinearAlgebraInterface<SolverNode>;
template class MrHyDE::LinearSolverOptions<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
template class MrHyDE::LinearAlgebraInterface<SubgridSolverNode>;
template class MrHyDE::LinearSolverOptions<SubgridSolverNode>; 
#endif
