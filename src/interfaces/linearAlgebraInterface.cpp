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

using namespace MrHyDE;

// Explicit template instantiations
template class MrHyDE::LinearAlgebraInterface<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
template class MrHyDE::LinearAlgebraInterface<SubgridSolverNode>;
#endif

// ========================================================================================
// Constructor  
// ========================================================================================

template<class Node>
LinearAlgebraInterface<Node>::LinearAlgebraInterface(const Teuchos::RCP<MpiComm> & Comm_,
                                                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                                     Teuchos::RCP<DiscretizationInterface> & disc_,
                                                     Teuchos::RCP<ParameterManager<Node> > & params_) :
Comm(Comm_), settings(settings_), disc(disc_), params(params_) {
  
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting linear algebra interface constructor ..." << endl;
    }
  }
  
  verbosity = settings->get<int>("verbosity",0);
  
  linearTOL = settings->sublist("Solver").get<ScalarT>("linear TOL",1.0E-7);
  maxLinearIters = settings->sublist("Solver").get<int>("max linear iters",100);
  maxKrylovVectors = settings->sublist("Solver").get<int>("krylov vectors",100);
  belos_residual_scaling = settings->sublist("Solver").get<string>("Belos implicit residual scaling","None");
  // Also: "Norm of Preconditioned Initial Residual" or "Norm of Initial Residual"
  
  useDirect = settings->sublist("Solver").get<bool>("use direct solver",false);
  useDirectBL2 = settings->sublist("Solver").get<bool>("use direct solver",false);
  useDirectL2 = settings->sublist("Solver").get<bool>("use direct solver",false);
  useDirectAux = settings->sublist("Solver").get<bool>("use direct solver for aux",false);
  useDirectBL2Aux = settings->sublist("Solver").get<bool>("use direct solver for aux",false);
  useDirectL2Aux = settings->sublist("Solver").get<bool>("use direct solver for aux",false);
  useDirectBL2Param = settings->sublist("Solver").get<bool>("use direct solver for param",false);
  useDirectL2Param = settings->sublist("Solver").get<bool>("use direct solver for param",false);
  
  useDomDecomp = settings->sublist("Solver").get<bool>("use domain decomposition",false);
  useDomDecompL2 = settings->sublist("Solver").get<bool>("use domain decomposition for L2 projections",false);
  useDomDecompBL2 = settings->sublist("Solver").get<bool>("use domain decomposition for DBCs",false);
  useDomDecompAux = settings->sublist("Solver").get<bool>("use domain decomposition for aux",false);
  useDomDecompL2Aux = settings->sublist("Solver").get<bool>("use domain decomposition for aux L2 projections",false);
  useDomDecompBL2Aux = settings->sublist("Solver").get<bool>("use domain decomposition for aux DBCs",false);
  useDomDecompL2Param = settings->sublist("Solver").get<bool>("use domain decomposition for param L2 projections",false);
  useDomDecompBL2Param = settings->sublist("Solver").get<bool>("use domain decomposition for param DBCs",false);
  
  usePrec = settings->sublist("Solver").get<bool>("use preconditioner",true);
  usePrecBL2 = settings->sublist("Solver").get<bool>("use preconditioner for DBCs",true);
  usePrecL2 = settings->sublist("Solver").get<bool>("use preconditioner for L2 projections",true);
  usePrecAux = settings->sublist("Solver").get<bool>("use preconditioner for aux",true);
  usePrecBL2Aux = settings->sublist("Solver").get<bool>("use preconditioner for aux DBCs",true);
  usePrecL2Aux = settings->sublist("Solver").get<bool>("use preconditioner for aux L2 projections",true);
  usePrecBL2Param = settings->sublist("Solver").get<bool>("use preconditioner for param DBCs",true);
  usePrecL2Param = settings->sublist("Solver").get<bool>("use preconditioner for param L2 projections",true);
  
  reuse_preconditioner = settings->sublist("Solver").get<bool>("reuse preconditioner",false);
  reuse_aux_preconditioner = settings->sublist("Solver").get<bool>("reuse aux preconditioner",false);
  
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
  
  {
    vector<GO> owned, ownedAndShared;
    
    disc->DOF->getOwnedIndices(owned);
    LO numUnknowns = (LO)owned.size();
    disc->DOF->getOwnedAndGhostedIndices(ownedAndShared);
    GO localNumUnknowns = numUnknowns;
    
    GO globalNumUnknowns = 0;
    Teuchos::reduceAll<LO,GO>(*Comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
    
    owned_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, owned, 0, Comm));
    overlapped_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, ownedAndShared, 0, Comm));
    
    vector<size_t> maxEntriesPerRow(overlapped_map->getNodeNumElements(), 0);
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<size_t> EIDs = disc->myElements[b];
      for (size_t e=0; e<EIDs.size(); e++) {
        vector<GO> gids;
        size_t elemID = EIDs[e];
        disc->DOF->getElementGIDs(elemID, gids, blocknames[b]);
        for (size_t i=0; i<gids.size(); i++) {
          LO ind1 = overlapped_map->getLocalElement(gids[i]);
          maxEntriesPerRow[ind1] += gids.size();
        }
      }
    }
    
    maxEntries = 0;
    for (size_t m=0; m<maxEntriesPerRow.size(); ++m) {
      maxEntries = std::max(maxEntries, maxEntriesPerRow[m]);
    }
    
    //cout << "maxEntries = " << maxEntries << endl;
    
    maxEntries = static_cast<size_t>(settings->sublist("Solver").get<int>("max entries per row",
                                                                          static_cast<int>(maxEntries)));
    
    overlapped_graph = Teuchos::rcp(new LA_CrsGraph(overlapped_map,
                                                    maxEntries,
                                                    Tpetra::StaticProfile));
    
    exporter = Teuchos::rcp(new LA_Export(overlapped_map, owned_map));
    importer = Teuchos::rcp(new LA_Import(owned_map, overlapped_map));
    
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<size_t> EIDs = disc->myElements[b];
      for (size_t e=0; e<EIDs.size(); e++) {
        vector<GO> gids;
        size_t elemID = EIDs[e];
        disc->DOF->getElementGIDs(elemID, gids, blocknames[b]);
        for (size_t i=0; i<gids.size(); i++) {
          GO ind1 = gids[i];
          overlapped_graph->insertGlobalIndices(ind1,gids);
        }
      }
    }
    
    overlapped_graph->fillComplete();
    
    matrix = Teuchos::rcp(new LA_CrsMatrix(owned_map, maxEntries, Tpetra::StaticProfile));
    
    overlapped_matrix = Teuchos::rcp(new LA_CrsMatrix(overlapped_graph));
    
    this->fillComplete(matrix);
    this->fillComplete(overlapped_matrix);
  }
  
  // --------------------------------------------------
  // aux variable LA objects
  // --------------------------------------------------
  
  if (disc->phys->have_aux) {
    vector<GO> aux_owned, aux_ownedAndShared;
    
    disc->auxDOF->getOwnedIndices(aux_owned);
    LO numUnknowns = (LO)aux_owned.size();
    disc->auxDOF->getOwnedAndGhostedIndices(aux_ownedAndShared);
    GO localNumUnknowns = numUnknowns;
    
    GO globalNumUnknowns = 0;
    Teuchos::reduceAll<LO,GO>(*Comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
    
    aux_owned_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, aux_owned, 0, Comm));
    aux_overlapped_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, aux_ownedAndShared, 0, Comm));
    aux_overlapped_graph = Teuchos::rcp( new LA_CrsGraph(aux_overlapped_map,maxEntries));
    
    aux_exporter = Teuchos::rcp(new LA_Export(aux_overlapped_map, aux_owned_map));
    aux_importer = Teuchos::rcp(new LA_Import(aux_owned_map, aux_overlapped_map));
    
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<size_t> EIDs = disc->myElements[b];
      for (size_t e=0; e<EIDs.size(); e++) {
        vector<GO> gids;
        size_t elemID = EIDs[e];
        disc->auxDOF->getElementGIDs(elemID, gids, blocknames[b]);
        for (size_t i=0; i<gids.size(); i++) {
          GO ind1 = gids[i];
          aux_overlapped_graph->insertGlobalIndices(ind1,gids);
        }
      }
    }
    
    aux_overlapped_graph->fillComplete();
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
          vector<GO> stategids;
          disc->DOF->getElementGIDs(elemID, stategids, blocknames[b]);
          for (size_t i=0; i<gids.size(); i++) {
            GO ind1 = gids[i];
            param_overlapped_graph->insertGlobalIndices(ind1,stategids);
          }
        }
      }
      
      param_overlapped_graph->fillComplete(owned_map, param_owned_map);
      
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
Teuchos::RCP<Teuchos::ParameterList> LinearAlgebraInterface<Node>::getBelosParameterList() {
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
    numEqns = disc->phys->numVars[0];
  }
  belosList->set("number of equations", numEqns);
  
  belosList->set("Output Style", Belos::Brief);
  belosList->set("Implicit Residual Scaling", belos_residual_scaling);
  return belosList;
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolver(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (useDirect) {
    if (!have_symbolic_factor) {
      Am2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, r, soln);
      Am2Solver->symbolicFactorization();
      have_symbolic_factor = true;
    }
    Am2Solver->setA(J, Amesos2::SYMBFACT);
    Am2Solver->setX(soln);
    Am2Solver->setB(r);
    Am2Solver->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (usePrec) {
      if (useDomDecomp) {
        if (!reuse_preconditioner || !have_preconditioner) {
          Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
          ifpackList.set("schwarz: subdomain solver","garbage");
          M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
          M_dd->setParameters(ifpackList);
          M_dd->initialize();
          M_dd->compute();
          have_preconditioner = true;
        }
        Problem->setLeftPrec(M_dd);
      }
      else { // default - AMG preconditioner
        if (!have_preconditioner) {
          M = this->buildPreconditioner(J,"Preconditioner Settings");
          have_preconditioner = true;
        }
        else {
          MueLu::ReuseTpetraPreconditioner(J,*M);
        }
        Problem->setLeftPrec(M);
      }
    }
    Problem->setProblem();
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList();
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    
    solver->solve();
    
  }
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverL2(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (useDirectL2) {
    Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > L2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, r, soln);
    L2Solver->symbolicFactorization();
    L2Solver->setA(J, Amesos2::SYMBFACT);
    L2Solver->setX(soln);
    L2Solver->setB(r);
    L2Solver->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (usePrecL2) {
      if (useDomDecompL2) {
        Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
        ifpackList.set("schwarz: subdomain solver","garbage");
        Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > M_L2 = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
        M_L2->setParameters(ifpackList);
        M_L2->initialize();
        M_L2->compute();
        Problem->setLeftPrec(M_L2);
      }
      else { // default - AMG preconditioner
        Teuchos::RCP<MueLu::TpetraOperator<ScalarT,LO,GO,Node> > M_L2 = this->buildPreconditioner(J,"L2 Projection Preconditioner Settings");
        Problem->setLeftPrec(M_L2);
      }
    }
    Problem->setProblem();
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList();
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    
    solver->solve();
  }
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverBoundaryL2(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (useDirectBL2) {
    Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > L2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, r, soln);
    L2Solver->symbolicFactorization();
    L2Solver->setA(J, Amesos2::SYMBFACT);
    L2Solver->setX(soln);
    L2Solver->setB(r);
    L2Solver->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (usePrecBL2) {
      if (useDomDecompBL2) {
        Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
        ifpackList.set("schwarz: subdomain solver","garbage");
        Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > M_dd_BL2 = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
        M_dd_BL2->setParameters(ifpackList);
        M_dd_BL2->initialize();
        M_dd_BL2->compute();
        Problem->setLeftPrec(M_dd_BL2);
      }
      else { // default - AMG preconditioner
        if (!have_preconditioner_BL2) {
          M_BL2 = this->buildPreconditioner(J,"Boundary L2 Projection Preconditioner Settings");
          have_preconditioner_BL2 = true;
        }
        else {
          MueLu::ReuseTpetraPreconditioner(J,*M_BL2);
        }
        Problem->setLeftPrec(M_BL2);
      }
    }
    Problem->setProblem();
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList();
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    
    solver->solve();
  }
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverAux(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (useDirectAux) {
    if (!have_aux_symbolic_factor) {
      Am2Solver_aux = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, r, soln);
      Am2Solver_aux->symbolicFactorization();
      have_aux_symbolic_factor = true;
    }
    Am2Solver_aux->setA(J, Amesos2::SYMBFACT);
    Am2Solver_aux->setX(soln);
    Am2Solver_aux->setB(r);
    Am2Solver_aux->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (usePrecAux) {
      if (useDomDecompAux) {
        if (!reuse_aux_preconditioner || !have_aux_preconditioner) {
          Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
          ifpackList.set("schwarz: subdomain solver","garbage");
          M_dd_aux = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
          M_dd_aux->setParameters(ifpackList);
          M_dd_aux->initialize();
          M_dd_aux->compute();
          have_aux_preconditioner = true;
        }
        Problem->setLeftPrec(M_dd_aux);
      }
      else { // default - AMG preconditioner
        if (!have_aux_preconditioner) {
          M_aux = this->buildPreconditioner(J,"Aux Preconditioner Settings");
          have_aux_preconditioner = true;
        }
        else {
          MueLu::ReuseTpetraPreconditioner(J,*M_aux);
        }
        Problem->setLeftPrec(M_aux);
      }
    }
    Problem->setProblem();
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList();
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    
    solver->solve();
  }
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverL2Aux(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (useDirectL2Aux) {
    Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > L2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, r, soln);
    L2Solver->symbolicFactorization();
    L2Solver->setA(J, Amesos2::SYMBFACT);
    L2Solver->setX(soln);
    L2Solver->setB(r);
    L2Solver->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (usePrecL2Aux) {
      if (useDomDecompL2Aux) {
        Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
        ifpackList.set("schwarz: subdomain solver","garbage");
        Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > M_L2 = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
        M_L2->setParameters(ifpackList);
        M_L2->initialize();
        M_L2->compute();
        Problem->setLeftPrec(M_L2);
      }
      else { // default - AMG preconditioner
        Teuchos::RCP<MueLu::TpetraOperator<ScalarT,LO,GO,Node> > M_L2 = this->buildPreconditioner(J,"Aux L2 Projection Preconditioner Settings");
        Problem->setLeftPrec(M_L2);
      }
    }
    Problem->setProblem();
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList();
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    
    solver->solve();
  }
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverBoundaryL2Aux(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (useDirectBL2Aux) {
    Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > L2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, r, soln);
    L2Solver->symbolicFactorization();
    L2Solver->setA(J, Amesos2::SYMBFACT);
    L2Solver->setX(soln);
    L2Solver->setB(r);
    L2Solver->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (usePrecBL2Aux) {
      if (useDomDecompBL2Aux) {
        Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
        ifpackList.set("schwarz: subdomain solver","garbage");
        Teuchos::RCP<Ifpack2::Preconditioner<ScalarT,LO,GO,Node>> M_BL2 = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
        M_BL2->setParameters(ifpackList);
        M_BL2->initialize();
        M_BL2->compute();
        Problem->setLeftPrec(M_BL2);
      }
      else { // default - AMG preconditioner
        Teuchos::RCP<MueLu::TpetraOperator<ScalarT,LO,GO,Node> > M_BL2 = this->buildPreconditioner(J,"Aux Boundary L2 Projection Preconditioner Settings");
        Problem->setLeftPrec(M_BL2);
      }
    }
    Problem->setProblem();
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList();
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    
    solver->solve();
  }
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (useDirectL2Param) {
    Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > L2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, r, soln);
    L2Solver->symbolicFactorization();
    L2Solver->setA(J, Amesos2::SYMBFACT);
    L2Solver->setX(soln);
    L2Solver->setB(r);
    L2Solver->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (usePrecL2Param) {
      if (useDomDecompL2Param) {
        Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
        ifpackList.set("schwarz: subdomain solver","garbage");
        Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > M_L2 = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
        M_L2->setParameters(ifpackList);
        M_L2->initialize();
        M_L2->compute();
        Problem->setLeftPrec(M_L2);
      }
      else { // default - AMG preconditioner
        Teuchos::RCP<MueLu::TpetraOperator<ScalarT,LO,GO,Node> > M_L2 = this->buildPreconditioner(J,"Param L2 Projection Preconditioner Settings");
        Problem->setLeftPrec(M_L2);
      }
    }
    Problem->setProblem();
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList();
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    
    solver->solve();
  }
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::linearSolverBoundaryL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  if (useDirectBL2Param) {
    Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > L2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, r, soln);
    L2Solver->symbolicFactorization();
    L2Solver->setA(J, Amesos2::SYMBFACT);
    L2Solver->setX(soln);
    L2Solver->setB(r);
    L2Solver->numericFactorization().solve();
  }
  else {
    Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
    if (usePrecBL2Param) {
      if (useDomDecompBL2Param) {
        Teuchos::ParameterList & ifpackList = settings->sublist("Solver").sublist("Ifpack2");
        ifpackList.set("schwarz: subdomain solver","garbage");
        Teuchos::RCP<Ifpack2::Preconditioner<ScalarT,LO,GO,Node>> M_BL2 = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
        M_BL2->setParameters(ifpackList);
        M_BL2->initialize();
        M_BL2->compute();
        Problem->setLeftPrec(M_BL2);
      }
      else { // default - AMG preconditioner
        Teuchos::RCP<MueLu::TpetraOperator<ScalarT,LO,GO,Node> > M_BL2 = this->buildPreconditioner(J,"Param Boundary L2 Projection Preconditioner Settings");
        Problem->setLeftPrec(M_BL2);
      }
    }
    Problem->setProblem();
    Teuchos::RCP<Teuchos::ParameterList> belosList = this->getBelosParameterList();
    Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT,LA_MultiVector,LA_Operator>(Problem, belosList));
    
    solver->solve();
  }
}

// ========================================================================================
// Preconditioner for Tpetra stack
// ========================================================================================

template<class Node>
Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > LinearAlgebraInterface<Node>::buildPreconditioner(const matrix_RCP & J, const string & precSublist) {
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
  if(settings->sublist("Solver").isSublist("Preconditioner Settings")) {
    Teuchos::ParameterList inputPrecParams = settings->sublist("Solver").sublist(precSublist);
    mueluParams.setParameters(inputPrecParams);
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

// ========================================================================================
// ========================================================================================
