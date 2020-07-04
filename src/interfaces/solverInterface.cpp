/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "solverInterface.hpp"

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

solver::solver(const Teuchos::RCP<MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
               Teuchos::RCP<meshInterface> & mesh_,
               Teuchos::RCP<discretization> & disc_,
               Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager> & DOF_,
               Teuchos::RCP<AssemblyManager> & assembler_,
               Teuchos::RCP<ParameterManager> & params_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_), DOF(DOF_), assembler(assembler_), params(params_) {
  
  milo_debug_level = settings->get<int>("debug level",0);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver constructor ..." << endl;
    }
  }
  
  soln = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  string analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  if (analysis_type == "forward+adjoint" || analysis_type == "ROL" || analysis_type == "ROL_SIMOPT") {
    save_solution = true; // default is false
  }
  //adj_soln = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  
  // Get the required information from the settings
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  isInitial = false;
  initial_time = settings->sublist("Solver").get<ScalarT>("initial time",0.0);
  current_time = initial_time;
  final_time = settings->sublist("Solver").get<ScalarT>("final time",1.0);
  if (settings->sublist("Solver").isParameter("delta t")) {
    deltat = settings->sublist("Solver").get<ScalarT>("delta t");
  //  numsteps = std::ceil((final_time - initial_time)/deltat);
  }
  else {
    int numTimesteps = settings->sublist("Solver").get<int>("number of steps",1);
    deltat = (final_time - initial_time)/numTimesteps;
  }
  verbosity = settings->get<int>("verbosity",0);
  usestrongDBCs = settings->sublist("Solver").get<bool>("use strong DBCs",true);
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("use measurements as DBCs", false);
  solver_type = settings->sublist("Solver").get<string>("solver","none"); // or "transient"
  allow_remesh = settings->sublist("Solver").get<bool>("remesh",false);
  time_order = settings->sublist("Solver").get<int>("time order",1);
  NLtol = settings->sublist("Solver").get<ScalarT>("nonlinear TOL",1.0E-6);
  MaxNLiter = settings->sublist("Solver").get<int>("max nonlinear iters",10);
  NLsolver = settings->sublist("Solver").get<string>("nonlinear solver","Newton");
  
  ButcherTab = settings->sublist("Solver").get<string>("transient Butcher tableau","BWE");
  BDForder = settings->sublist("Solver").get<int>("transient BDF order",1);
  
  // Additional parameters for higher-order BDF methods that require some startup procedure
  startupButcherTab = settings->sublist("Solver").get<string>("transient startup Butcher tableau",ButcherTab);
  startupBDForder = settings->sublist("Solver").get<int>("transient startup BDF order",BDForder);
  startupSteps = settings->sublist("Solver").get<int>("transient startup steps",BDForder);
  
  line_search = false;//settings->sublist("Solver").get<bool>("Use Line Search","false");
  store_adjPrev = false;
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  if (!isTransient) {
    deltat = 1.0;
  }
  
  response_type = settings->sublist("Postprocess").get("response type", "pointwise"); // or "global"
  compute_objective = settings->sublist("Postprocess").get("compute objective",false);
  compute_sensitivity = settings->sublist("Postprocess").get("compute sensitivities",false);
  compute_aux_sensitivity = settings->sublist("Solver").get("compute aux sensitivities",false);
  compute_flux = settings->sublist("Solver").get("compute flux",false);
  
  initial_type = settings->sublist("Solver").get<string>("initial type","L2-projection");
  multigrid_type = settings->sublist("Solver").get<string>("multigrid type","sa");
  smoother_type = settings->sublist("Solver").get<string>("smoother type","CHEBYSHEV"); // or RELAXATION
  useLinearSolver = settings->sublist("Solver").get<bool>("use linear solver",true);
  lintol = settings->sublist("Solver").get<ScalarT>("linear TOL",1.0E-7);
  liniter = settings->sublist("Solver").get<int>("max linear iters",100);
  kspace = settings->sublist("Solver").get<int>("krylov vectors",100);
  useDomDecomp = settings->sublist("Solver").get<bool>("use dom decomp",false);
  useDirect = settings->sublist("Solver").get<bool>("use direct solver",false);
  usePrec = settings->sublist("Solver").get<bool>("use preconditioner",true);
  dropTol = settings->sublist("Solver").get<ScalarT>("ILU drop tol",0.0); //defaults to AztecOO default
  fillParam = settings->sublist("Solver").get<ScalarT>("ILU fill param",3.0); //defaults to AztecOO default
  
  have_symbolic_factor = false;
  
  // needed information from the mesh
  mesh->mesh->getElementBlockNames(blocknames);
  
  // needed information from the physics interface
  numVars = phys->numVars; //
  vector<vector<string> > phys_varlist = phys->varlist;
  //offsets = phys->offsets;
  
  // needed information from the DOF manager
  DOF->getOwnedIndices(LA_owned);
  numUnknowns = (LO)LA_owned.size();
  DOF->getOwnedAndGhostedIndices(LA_ownedAndShared);
  numUnknownsOS = (LO)LA_ownedAndShared.size();
  GO localNumUnknowns = numUnknowns;
  
  DOF->getOwnedIndices(owned);
  DOF->getOwnedAndGhostedIndices(ownedAndShared);
  
  globalNumUnknowns = 0;
  Teuchos::reduceAll<LO,GO>(*Comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
  //Comm->SumAll(&localNumUnknowns, &globalNumUnknowns, 1);
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "Number of global unknowns: " << globalNumUnknowns << endl;
    }
  }
  // needed information from the disc interface
  vector<vector<int> > cards = disc->cards;
  
  for (size_t b=0; b<blocknames.size(); b++) {
    
    vector<int> curruseBasis(numVars[b]);
    vector<int> currnumBasis(numVars[b]);
    vector<string> currvarlist(numVars[b]);
    
    int currmaxBasis = 0;
    for (int j=0; j<numVars[b]; j++) {
      string var = phys_varlist[b][j];
      int vnum = DOF->getFieldNum(var);
      int vub = phys->getUniqueIndex(b,var);
      currvarlist[j] = var;
      curruseBasis[j] = vub;
      currnumBasis[j] = cards[b][vub];
      //currvarlist[vnum] = var;
      //curruseBasis[vnum] = vub;
      //currnumBasis[vnum] = cards[b][vub];
      currmaxBasis = std::max(currmaxBasis,cards[b][vub]);
    }
    
    phys->setVars(b,currvarlist);
    
    varlist.push_back(currvarlist);
    useBasis.push_back(curruseBasis);
    numBasis.push_back(currnumBasis);
    maxBasis.push_back(currmaxBasis);
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Tpetra maps
  /////////////////////////////////////////////////////////////////////////////
  
  maxEntries = 256;
  
  if (spaceDim == 1) {
    maxEntries = 2*maxDerivs;
  }
  else if (spaceDim == 2) {
    maxEntries = 4*maxDerivs;
  }
  else if (spaceDim == 3) {
    maxEntries = 8*maxDerivs;
  }
  //maxEntries = 512;
  
  this->setupLinearAlgebra();
  
  /////////////////////////////////////////////////////////////////////////////
  // Worksets
  /////////////////////////////////////////////////////////////////////////////
  
  assembler->createWorkset();
  numsteps = 0;
  numstages = 0;
  this->setBackwardDifference(BDForder);
  this->setButcherTableau(ButcherTab);
  if (BDForder > 1) {
    this->setBackwardDifference(startupBDForder);
    this->setButcherTableau(startupButcherTab);
  }
  
  this->finalizeWorkset();
  phys->setWorkset(assembler->wkset);
  params->wkset = assembler->wkset;
  
  if (settings->sublist("Mesh").get<bool>("have element data", false) ||
      settings->sublist("Mesh").get<bool>("have nodal data", false)) {
    mesh->readMeshData(LA_overlapped_map, assembler->cells);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  
  this->setBatchID(Comm->getRank());
  
  /////////////////////////////////////////////////////////////////////////////
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver constructor" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

void solver::setButcherTableau(const string & tableau) {
  Kokkos::View<ScalarT**,UnifiedDevice> butcher_A;
  Kokkos::View<ScalarT*,UnifiedDevice> butcher_b, butcher_c;
  
  //only filling in the non-zero entries
  if (tableau == "BWE" || tableau == "DIRK-1,1") {
    butcher_A = Kokkos::View<ScalarT**,UnifiedDevice>("butcher_A",1,1);
    butcher_A(0,0) = 1.0;
    butcher_b = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_b",1);
    butcher_b(0) = 1.0;
    butcher_c = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_c",1);
    butcher_c(0) = 1.0;
  }
  else if (tableau == "FWE") {
    butcher_A = Kokkos::View<ScalarT**,UnifiedDevice>("butcher_A",1,1);
    butcher_b = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_b",1);
    butcher_b(0) = 1.0;
    butcher_c = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_c",1);
  }
  else if (tableau == "CN") {
    butcher_A = Kokkos::View<ScalarT**,UnifiedDevice>("butcher_A",2,2);
    butcher_A(1,0) = 0.5;
    butcher_A(1,1) = 0.5;
    butcher_b = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_b",2);
    butcher_b(0) = 0.5;
    butcher_b(1) = 0.5;
    butcher_c = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_c",2);
    butcher_c(1) = 1.0;
  }
  else if (tableau == "SSPRK-3,3") {
    butcher_A = Kokkos::View<ScalarT**,UnifiedDevice>("butcher_A",3,3);
    butcher_A(1,0) = 1.0;
    butcher_A(2,0) = 0.25;
    butcher_A(2,1) = 0.25;
    butcher_b = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_b",4);
    butcher_b(0) = 1.0/6.0;
    butcher_b(1) = 1.0/6.0;
    butcher_b(2) = 2.0/3.0;
    butcher_c = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_c",4);
    butcher_c(1) = 1.0;
    butcher_c(2) = 1.0/2.0;
  }
  else if (tableau == "RK-4,4") { // Classical RK4
    butcher_A = Kokkos::View<ScalarT**,UnifiedDevice>("butcher_A",4,4);
    butcher_A(1,0) = 0.5;
    butcher_A(2,1) = 0.5;
    butcher_A(3,2) = 1.0;
    butcher_b = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_b",4);
    butcher_b(0) = 1.0/6.0;
    butcher_b(1) = 1.0/3.0;
    butcher_b(2) = 1.0/3.0;
    butcher_b(3) = 1.0/6.0;
    butcher_c = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_c",4);
    butcher_c(1) = 1.0/2.0;
    butcher_c(2) = 1.0/2.0;
    butcher_c(3) = 1.0;
  }
  else if (tableau == "DIRK-1,2") {
    butcher_A = Kokkos::View<ScalarT**,UnifiedDevice>("butcher_A",1,1);
    butcher_A(0,0) = 0.5;
    butcher_b = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_b",1);
    butcher_b(0) = 1.0;
    butcher_c = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_c",1);
    butcher_c(0) = 0.5;
  }
  else if (tableau == "DIRK-2,2") { // 2-stage, 2nd order
    butcher_A = Kokkos::View<ScalarT**,UnifiedDevice>("butcher_A",2,2);
    butcher_A(0,0) = 1.0/4.0;
    butcher_A(1,0) = 1.0/2.0;
    butcher_A(1,1) = 1.0/4.0;
    butcher_b = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_b",2);
    butcher_b(0) = 1.0/2.0;
    butcher_b(1) = 1.0/2.0;
    butcher_c = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_c",2);
    butcher_c(0) = 1.0/4.0;
    butcher_c(1) = 3.0/4.0;
  }
  else if (tableau == "DIRK-2,3") { // 2-stage, 3rd order
    butcher_A = Kokkos::View<ScalarT**,UnifiedDevice>("butcher_A",2,2);
    butcher_A(0,0) = 1.0/2.0 + std::sqrt(3)/6.0;
    butcher_A(1,0) = -std::sqrt(3)/3.0;
    butcher_A(1,1) = 1.0/2.0  + std::sqrt(3)/6.0;
    butcher_b = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_b",2);
    butcher_b(0) = 1.0/2.0;
    butcher_b(1) = 1.0/2.0;
    butcher_c = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_c",2);
    butcher_c(0) = 1.0/2.0 + std::sqrt(3)/6.0;;
    butcher_c(1) = 1.0/2.0 - std::sqrt(3)/6.0;;
  }
  else if (tableau == "DIRK-3,3") { // 3-stage, 3rd order
    ScalarT p = 0.4358665215;
    butcher_A = Kokkos::View<ScalarT**,UnifiedDevice>("butcher_A",3,3);
    butcher_A(0,0) = p;
    butcher_A(1,0) = (1.0-p)/2.0;
    butcher_A(1,1) = p;
    butcher_A(2,0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
    butcher_A(2,1) = 3.0*p*p/2.0 - 5.0*p + 5.0/4.0;
    butcher_A(2,2) = p;
    butcher_b = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_b",3);
    butcher_b(0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
    butcher_b(1) = 3.0*p*p/2.0-5.0*p+5.0/4.0;
    butcher_b(2) = p;
    butcher_c = Kokkos::View<ScalarT*,UnifiedDevice>("butcher_c",3);
    butcher_c(0) = p;
    butcher_c(1) = (1.0+p)/2.0;
    butcher_c(2) = 1.0;
  }
  for (size_t b=0; b<assembler->cells.size(); b++) {
    assembler->wkset[b]->butcher_A = butcher_A;
    assembler->wkset[b]->butcher_b = butcher_b;
    assembler->wkset[b]->butcher_c = butcher_c;
  }
  int newnumstages = butcher_A.extent(0);
  numstages = std::max(numstages,newnumstages);
  
  //KokkosTools::print(butcher_A);
  //KokkosTools::print(butcher_b);
  //KokkosTools::print(butcher_c);
  // GH: commenting this out for now; it can't tell DRV from DRVint and has an overload problem
  /*
   if (milo_debug_level > 1) {
   if (Comm->getRank() == 0) {
   cout << "******** Butcher tableau: " << endl;
   KokkosTools::print(butcher_A, "solver::setButcherTableau() - Butcher-A");
   KokkosTools::print(butcher_b, "solver::setButcherTableau() - Butcher-b");
   KokkosTools::print(butcher_c, "solver::setButcherTableau() - Butcher-c");
   }
   }*/
}

// ========================================================================================
// ========================================================================================

void solver::setBackwardDifference(const int & order) { // using order as an input to allow for dynamic changes
  
  Kokkos::View<ScalarT*,UnifiedDevice> BDF_wts;
  // Note that these do not include 1/deltat (added in wkset)
  // Not going to work properly for adaptive time stepping if BDForder>1
  if (isTransient) {
    
    if (order == 1) {
      BDF_wts = Kokkos::View<ScalarT*,UnifiedDevice>("BDF weights to compute u_dot",2);
      BDF_wts(0) = 1.0;
      BDF_wts(1) = -1.0;
    }
    else if (order == 2) {
      BDF_wts = Kokkos::View<ScalarT*,UnifiedDevice>("BDF weights to compute u_dot",3);
      BDF_wts(0) = 1.5;
      BDF_wts(1) = -2.0;
      BDF_wts(2) = 0.5;
    }
    else if (order == 3) {
      BDF_wts = Kokkos::View<ScalarT*,UnifiedDevice>("BDF weights to compute u_dot",4);
      BDF_wts(0) = 11.0/6.0;
      BDF_wts(1) = -3.0;
      BDF_wts(2) = 1.5;
      BDF_wts(3) = -1.0/3.0;
    }
    else if (order == 4) {
      BDF_wts = Kokkos::View<ScalarT*,UnifiedDevice>("BDF weights to compute u_dot",5);
      BDF_wts(0) = 25.0/12.0;
      BDF_wts(1) = -4.0;
      BDF_wts(2) = 3.0;
      BDF_wts(3) = -4.0/3.0;
      BDF_wts(4) = 1.0/4.0;
    }
    else if (order == 5) {
      BDF_wts = Kokkos::View<ScalarT*,UnifiedDevice>("BDF weights to compute u_dot",6);
      BDF_wts(0) = 137.0/60.0;
      BDF_wts(1) = -5.0;
      BDF_wts(2) = 5.0;
      BDF_wts(3) = -10.0/3.0;
      BDF_wts(4) = 75.0/60.0;
      BDF_wts(5) = -1.0/5.0;
    }
    else if (order == 6) {
      BDF_wts = Kokkos::View<ScalarT*,UnifiedDevice>("BDF weights to compute u_dot",7);
      BDF_wts(0) = 147.0/60.0;
      BDF_wts(1) = -6.0;
      BDF_wts(2) = 15.0/2.0;
      BDF_wts(3) = -20.0/3.0;
      BDF_wts(4) = 225.0/60.0;
      BDF_wts(5) = -72.0/60.0;
      BDF_wts(6) = 1.0/6.0;
    }
    
    int newnumsteps = BDF_wts.extent(0)-1;
    numsteps = std::max(numsteps,newnumsteps);
    
  }
  else { // for steady state solves, u_dot = 0.0*u
    
    BDF_wts = Kokkos::View<ScalarT*,UnifiedDevice>("BDF weights to compute u_dot",1);
    BDF_wts(0) = 1.0;
    //BDF_wts(1) = 0.0;
    
    numsteps = 1;
  }
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    assembler->wkset[b]->BDF_wts = BDF_wts;
  }
  
  //KokkosTools::print(BDF_wts);
  
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

void solver::finalizeWorkset() {
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::finalizeWorkset ..." << endl;
    }
  }
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    
    vector<vector<int> > voffsets = phys->offsets[b];
    size_t maxoff = 0;
    for (size_t i=0; i<voffsets.size(); i++) {
      if (voffsets[i].size() > maxoff) {
        maxoff = voffsets[i].size();
      }
    }
    // GH: this will not work if AssemblyMem is Cuda
    Kokkos::View<int**,AssemblyDevice> offsets_view("offsets on assembly device",voffsets.size(),maxoff);
    auto host_offsets = Kokkos::create_mirror_view(offsets_view);
    for (size_t i=0; i<voffsets.size(); i++) {
      for (size_t j=0; j<voffsets[i].size(); j++) {
        host_offsets(i,j) = voffsets[i][j];
      }
    }
    Kokkos::deep_copy(offsets_view,host_offsets);
    assembler->wkset[b]->offsets = offsets_view;//phys->voffsets[b];
    
    size_t maxpoff = 0;
    for (size_t i=0; i<params->paramoffsets.size(); i++) {
      if (params->paramoffsets[i].size() > maxpoff) {
        maxpoff = params->paramoffsets[i].size();
      }
      //maxpoff = max(maxpoff,paramoffsets[i].size());
    }
    // GH: this will not work if AssemblyMem is Cuda
    Kokkos::View<int**,AssemblyDevice> poffsets_view("param offsets on assembly device",params->paramoffsets.size(),maxpoff);
    auto host_poffsets = Kokkos::create_mirror_view(poffsets_view);
    for (size_t i=0; i<params->paramoffsets.size(); i++) {
      for (size_t j=0; j<params->paramoffsets[i].size(); j++) {
        host_poffsets(i,j) = params->paramoffsets[i][j];
      }
    }
    Kokkos::deep_copy(poffsets_view,host_poffsets);
    assembler->wkset[b]->usebasis = useBasis[b];
    assembler->wkset[b]->paramusebasis = params->discretized_param_usebasis;
    assembler->wkset[b]->paramoffsets = poffsets_view;//paramoffsets;
    assembler->wkset[b]->varlist = varlist[b];
    assembler->wkset[b]->createSolns();
    int numDOF = assembler->cells[b][0]->GIDs.extent(1);
    for (size_t e=0; e<assembler->cells[b].size(); e++) {
      assembler->cells[b][e]->wkset = assembler->wkset[b];
      assembler->cells[b][e]->setUseBasis(useBasis[b], numsteps, numstages);
      assembler->cells[b][e]->setUpAdjointPrev(numDOF);
      assembler->cells[b][e]->setUpSubGradient(params->num_active_params);
    }
    assembler->wkset[b]->params = params->paramvals_AD;
    assembler->wkset[b]->params_AD = params->paramvals_KVAD;
    assembler->wkset[b]->paramnames = params->paramnames;
    //assembler->wkset[b]->setupParamBasis(discretized_param_basis);
    
    if (assembler->boundaryCells.size() > b) { // avoid seg faults
      for (size_t e=0; e<assembler->boundaryCells[b].size(); e++) {
        if (assembler->boundaryCells[b][e]->numElem > 0) {
          assembler->boundaryCells[b][e]->wkset = assembler->wkset[b];
          assembler->boundaryCells[b][e]->setUseBasis(useBasis[b], numsteps, numstages);
          
          /*
          assembler->boundaryCells[b][e]->wksetBID = assembler->wkset[b]->addSide(assembler->boundaryCells[b][e]->nodes,
                                                                                  assembler->boundaryCells[b][e]->sidenum,
                                                                                  assembler->boundaryCells[b][e]->localSideID,
                                                                                  assembler->boundaryCells[b][e]->orientation);
           */
        }
      }
    }
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::finalizeWorkset" << endl;
    }
  }
  
}

// ========================================================================================
// Set up the Tpetra objects (maps, importers, exporters and graphs)
// These do need to be recomputed whenever the mesh changes */
// ========================================================================================

void solver::setupLinearAlgebra() {
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::setupLinearAlgebra..." << endl;
    }
  }
  
  const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();
  
  LA_owned_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, LA_owned, 0, Comm));
  LA_overlapped_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, LA_ownedAndShared, 0, Comm));
  //LA_owned_graph = createCrsGraph(LA_owned_map, maxEntries);//Teuchos::rcp(new LA_CrsGraph(Copy, *LA_owned_map, 0));
  LA_overlapped_graph = Teuchos::rcp( new LA_CrsGraph(LA_overlapped_map,maxEntries));
  
  exporter = Teuchos::rcp(new LA_Export(LA_overlapped_map, LA_owned_map));
  importer = Teuchos::rcp(new LA_Import(LA_owned_map, LA_overlapped_map));
  
  
  Kokkos::View<GO**,HostDevice> gids;
 
  for (size_t b=0; b<assembler->cells.size(); b++) {
    vector<vector<int> > curroffsets = phys->offsets[b];
    Kokkos::View<LO*,UnifiedDevice> numDOF_KV("number of DOF per variable",numVars[b]);
    for (int k=0; k<numVars[b]; k++) {
      numDOF_KV(k) = numBasis[b][k];
    }
    
    for(size_t e=0; e<assembler->cells[b].size(); e++) {
      assembler->cells[b][0]->cellData->numDOF = numDOF_KV;
      gids = assembler->cells[b][e]->GIDs;
      
      int numElem = assembler->cells[b][e]->numElem;
      if (timeImplicit) {
        // this should fail on the first iteration through if maxDerivs is not large enough
        TEUCHOS_TEST_FOR_EXCEPTION(gids.extent(1) > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of degrees of freedom per element times the number of time stages.");
      }
      //vector<vector<vector<int> > > cellindices;
      Kokkos::View<LO***,AssemblyDevice> cellindices("Local DOF indices", numElem, numVars[b], maxBasis[b]);
      auto host_cellindices = Kokkos::create_mirror_view(cellindices);
      
      for (int p=0; p<numElem; p++) {
        for (int n=0; n<numVars[b]; n++) {
          for( int i=0; i<numBasis[b][n]; i++ ) {
            GO cgid = gids(p,curroffsets[n][i]);
            host_cellindices(p,n,i) = LA_overlapped_map->getLocalElement(cgid);
          }
        }
        Teuchos::Array<GO> ind2(gids.extent(1));
        for (size_t i=0; i<gids.extent(1); i++) {
          ind2[i] = gids(p,i);
        }
        for (size_t i=0; i<gids.extent(1); i++) {
          GO ind1 = gids(p,i);
          LA_overlapped_graph->insertGlobalIndices(ind1,ind2);
        }
      }
      Kokkos::deep_copy(cellindices,host_cellindices);
      assembler->cells[b][e]->setIndex(cellindices);
    }
    
    if (assembler->boundaryCells.size() > b) {
      // boundary cells share meta-data, so no need to reset numDOF
      for(size_t e=0; e<assembler->boundaryCells[b].size(); e++) {
        gids = assembler->boundaryCells[b][e]->GIDs;
        
        int numElem = assembler->boundaryCells[b][e]->numElem;
        Kokkos::View<LO***,AssemblyDevice> cellindices("Local DOF indices", numElem, numVars[b], maxBasis[b]);
        auto host_cellindices = Kokkos::create_mirror_view(cellindices);
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<numVars[b]; n++) {
            for( int i=0; i<numBasis[b][n]; i++ ) {
              GO cgid = gids(p,curroffsets[n][i]);
              host_cellindices(p,n,i) = LA_overlapped_map->getLocalElement(cgid);
            }
          }
          Teuchos::Array<GO> ind2(gids.extent(1));
          for (size_t i=0; i<gids.extent(1); i++) {
            ind2[i] = gids(p,i);
          }
          for (size_t i=0; i<gids.extent(1); i++) {
            GO ind1 = gids(p,i);
            LA_overlapped_graph->insertGlobalIndices(ind1,ind2);
          }
        }
        Kokkos::deep_copy(cellindices,host_cellindices);
        assembler->boundaryCells[b][e]->setIndex(cellindices);
      }
    }
  }
  
  LA_overlapped_graph->fillComplete();
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::setupLinearAlgebra" << endl;
    }
  }
  
}

// ========================================================================================
/* given the parameters, solve the forward  problem */
// ========================================================================================

void solver::forwardModel(DFAD & obj) {
  
  current_time = initial_time;
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::forwardModel ..." << endl;
    }
  }
  
  useadjoint = false;
  params->sacadoizeParams(false);
  
  vector_RCP u = this->setInitial();
  
  if (solver_type == "transient") {
    soln->store(u, current_time, 0); // copies the data
  }
  
  vector_RCP zero_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
  if (solver_type == "steady-state") {
    
    this->nonlinearSolver(u, zero_soln);
    if (compute_objective) {
      obj = this->computeObjective(u, 0.0, 0);
    }
    postproc->record(current_time);
    if (save_solution) {
      soln->store(u, current_time, 0);
    }
  }
  else if (solver_type == "transient") {
    vector<ScalarT> gradient; // not really used here
    this->transientSolver(u, obj, gradient, initial_time, final_time);
  }
  else {
    // print out an error message
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::forwardModel" << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================

void solver::adjointModel(vector<ScalarT> & gradient) {
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::adjointModel ..." << endl;
    }
  }
  
  useadjoint = true;
  
  params->sacadoizeParams(false);
  
  vector_RCP phi = setInitial();
  
  vector_RCP zero_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
  
  if (solver_type == "steady-state") {
    vector_RCP u;
    bool fnd = soln->extract(u, current_time);
    
    this->nonlinearSolver(u, phi);
    
    this->computeSensitivities(u, phi, gradient);
    
  }
  else if (solver_type == "transient") {
    DFAD obj = 0.0;
    this->transientSolver(phi, obj, gradient, initial_time, final_time);
  }
  else {
    // print out an error message
  }
  
  useadjoint = false;
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::adjointModel" << endl;
    }
  }
  
}


// ========================================================================================
/* solve the problem */
// ========================================================================================

void solver::transientSolver(vector_RCP & initial, DFAD & obj, vector<ScalarT> & gradient,
                             ScalarT & start_time, ScalarT & end_time) {
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting solver::transientSolver ..." << endl;
      cout << "******** Start time = " << start_time << endl;
      cout << "******** End time = " << end_time << endl;
      cout << "******** Time step size = " << deltat << endl;
    }
  }
  
  current_time = start_time;
  if (!useadjoint) { // forward solve - adaptive time stepping
    is_final_time = false;
    vector_RCP u = initial;
    vector_RCP zero_vec = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    zero_vec->putScalar(0.0);
    
    if (usestrongDBCs) {
      this->setDirichlet(u);
    }
    for (size_t b=0; b<blocknames.size(); b++) {
      assembler->performGather(b,u,0,0);
    }
    postproc->record(current_time);
        
    for (int s=0; s<numsteps; s++) {
      assembler->resetPrevSoln();
    }
    
    int stepProg = 0;
    obj = 0.0;
    int numCuts = 0;
    int maxCuts = 5; // TMW: make this a user-defined input
    double timetol = end_time*1.0e-6; // just need to get close enough to final time
    while (current_time < (end_time-timetol) && numCuts<=maxCuts) {
      
      if (BDForder > 1 && stepProg == startupSteps) {
        this->setBackwardDifference(BDForder);
        this->setButcherTableau(ButcherTab);
      }
      numstages = assembler->wkset[0]->butcher_A.extent(0);
      
      // Increment the previous step solutions (shift history and moves u into first spot)
      assembler->resetPrevSoln(); //
      
      // Reset the stage solutions (sets all to zero)
      assembler->resetStageSoln();
      
      ////////////////////////////////////////////////////////////////////////
      // Allow the cells to change subgrid model
      ////////////////////////////////////////////////////////////////////////
      
      if (multiscale_manager->subgridModels.size() > 0) {
        Teuchos::TimeMonitor localtimer(*msprojtimer);
        ScalarT my_cost = multiscale_manager->update();
        ScalarT gmin = 0.0;
        Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
        ScalarT gmax = 0.0;
        Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);
        if (Comm->getRank() == 0 && verbosity>0) {
          cout << "***** Multiscale Load Balancing Factor " << gmax/gmin <<  endl;
        }
      }
      
      if (Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Time Step " << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      
      vector_RCP u_prev = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
      u_prev->update(1.0,*u,0.0);
      
      int status = 1;
      for (int stage = 0; stage<numstages; stage++) {
        // Need a stage solution
        vector_RCP u_stage = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
        // Set the initial guess for stage solution
        u_stage->update(1.0,*u,0.0);
        
        assembler->updateStageNumber(stage); // could probably just += 1 in wksets
        
        status = this->nonlinearSolver(u_stage, zero_vec);
        
        u->update(1.0, *u_stage, 1.0);
        u->update(-1.0, *u_prev, 1.0);
        assembler->updateStageSoln(); // moves the stage solution into u_stage (avoids mem transfer)
      }
      
      if (status == 0) { // NL solver converged
        current_time += deltat;
        
        //soln->store(u, current_time, 0);
        
        // Make sure last step solution is gathered
        // Last set of values is from a stage solution, which is potentially different
        for (size_t b=0; b<blocknames.size(); b++) {
          assembler->performGather(b,u,0,0);
        }
        
        if (compute_objective) { // fill in the objective function
          DFAD cobj = this->computeObjective(u, current_time, soln->times[0].size()-1);
          obj += cobj;
        }
        postproc->record(current_time);
        if (save_solution) {
          soln->store(u, current_time, 0);
        }
        stepProg += 1;
      }
      else { // something went wrong, cut time step and try again
        deltat *= 0.5;
        numCuts += 1;
        
        bool fnd = soln->extract(u, current_time);
        if(Comm->getRank() == 0 && verbosity > 0) {
          cout << endl << endl << "*******************************************************" << endl;
          cout << endl << "**** Cutting Time Step " << endl;
          cout << "**** Current time is " << current_time << endl << endl;
          cout << "*******************************************************" << endl << endl << endl;
        }
        
      }
    }
  }
  else { // adjoint solve - fixed time stepping based on forward solve
    current_time = final_time;
    is_final_time = true;
    
    vector_RCP u = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP u_prev = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP phi = initial;
    
    size_t numFwdSteps = soln->times[0].size()-1;
    
    //bool fndu = soln->extract(u, numsteps);
    //assembler->performGather(0,u,0,0);
    
    for (size_t timeiter = 0; timeiter<numFwdSteps; timeiter++) {
      size_t cindex = numFwdSteps-timeiter;
      
      if(Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Adjoint Time Step " << timeiter << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      
      // TMW: this is specific to implicit Euler
      // Needs to be generalized
      // Also, need to implement checkpoint/recovery
      bool fndu = soln->extract(u, cindex);
      bool fndup = soln->extract(u_prev, cindex-1);
      for (size_t b=0; b<blocknames.size(); b++) {
        assembler->performGather(b,u_prev,0,0);
        assembler->resetPrevSoln();
      }
      
      current_time = soln->times[0][cindex-1];
      
      int status = this->nonlinearSolver(u, phi);
      
      this->computeSensitivities(u,phi,gradient);
      
      is_final_time = false;
      
    }
  }
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Finished solver::transientSolver" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================


int solver::nonlinearSolver(vector_RCP & u, vector_RCP & phi) {
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting solver::nonlinearSolver ..." << endl;
    }
  }
  
  int status = 0;
  int NLiter = 0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> NLerr_first(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> NLerr_scaled(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> NLerr(1);
  NLerr_first[0] = 10*NLtol;
  NLerr_scaled[0] = NLerr_first[0];
  NLerr[0] = NLerr_first[0];
  
  if (usestrongDBCs) {
    this->setDirichlet(u);
  }
  
  //this->setConstantPin(u); //pinning attempt
  int maxiter = MaxNLiter;
  if (useadjoint) {
    maxiter = 2;
  }
  
  while( NLerr_scaled[0]>NLtol && NLiter<maxiter ) { // while not converged
    
    multiscale_manager->reset();
    
    gNLiter = NLiter;
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    matrix_RCP J = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_owned_map, maxEntries));
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    matrix_RCP J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_overlapped_graph));
    vector_RCP du_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP du = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    
    // *********************** COMPUTE THE JACOBIAN AND THE RESIDUAL **************************
    
    bool build_jacobian = true;
    
    res_over->putScalar(0.0);
    J_over->setAllToScalar(0.0);
    J_over->fillComplete();
    J_over->resumeFill();
    if ( useadjoint && (NLiter == 1))
      store_adjPrev = true;
    else
      store_adjPrev = false;
    
    assembler->assembleJacRes(u, phi, build_jacobian, false, false,
                              res_over, J_over, isTransient, current_time, useadjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time, deltat);
    
    J_over->fillComplete();
    
    //J->setAllToScalar(0.0);
    J->doExport(*J_over, *exporter, Tpetra::ADD);
    J->fillComplete();
    
    res->putScalar(0.0);
    res->doExport(*res_over, *exporter, Tpetra::ADD);
    
    if (milo_debug_level>2) {
      KokkosTools::print(J);
      KokkosTools::print(res);
    }
    // *********************** CHECK THE NORM OF THE RESIDUAL **************************
    if (NLiter == 0) {
      res->normInf(NLerr_first);
      if (NLerr_first[0] > 1.0e-14)
        NLerr_scaled[0] = 1.0;
      else
        NLerr_scaled[0] = 0.0;
    }
    else {
      res->normInf(NLerr);
      NLerr_scaled[0] = NLerr[0]/NLerr_first[0];
    }
    
    if(Comm->getRank() == 0 && verbosity > 1) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Iteration: " << NLiter << endl;
      cout << "***** Norm of nonlinear residual: " << NLerr[0] << endl;
      cout << "***** Scaled Norm of nonlinear residual: " << NLerr_scaled[0] << endl;
      cout << "*********************************************************" << endl;
    }
    
    // *********************** SOLVE THE LINEAR SYSTEM **************************
    
    if (NLerr_scaled[0] > NLtol && useLinearSolver) {
      
      this->linearSolver(J, res, du);
      
      du_over->doImport(*du, *importer, Tpetra::ADD);
      
      if (useadjoint) {
        phi->update(1.0, *du_over, 1.0);
      }
      else {
        u->update(1.0, *du_over, 1.0);
      }
    }
    NLiter++; // increment number of iterations
  } // while loop
  if (milo_debug_level>2) {
    KokkosTools::print(u);
  }
  
  if(Comm->getRank() == 0) {
    if (!useadjoint) {
      //if( (NLiter>MaxNLiter || NLerr_scaled[0]>NLtol) && verbosity > 1) {
      if( (NLiter>MaxNLiter) && verbosity > 1) {
        status = 1;
        cout << endl << endl << "********************" << endl;
        cout << endl << "SOLVER FAILED TO CONVERGE CONVERGED in " << NLiter
        << " iterations with residual norm " << NLerr[0] << endl;
        cout << "********************" << endl;
      }
    }
  }
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Finished solver::nonlinearSolver" << endl;
    }
  }
  return status;
}

// ========================================================================================
// ========================================================================================

DFAD solver::computeObjective(const vector_RCP & F_soln, const ScalarT & time, const size_t & tindex) {
  
  DFAD totaldiff = 0.0;
  AD regDomain = 0.0;
  AD regBoundary = 0.0;
  int numDomainParams = params->domainRegIndices.size();
  int numBoundaryParams = params->boundaryRegIndices.size();
  
  params->sacadoizeParams(true);
  
  int numParams = params->num_active_params + params->globalParamUnknowns;
  vector<ScalarT> regGradient(numParams);
  vector<ScalarT> dmGradient(numParams);
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    
    assembler->performGather(b, F_soln, 0, 0);
    assembler->performGather(b, params->Psol[0], 4, 0);
    
    assembler->performBoundaryGather(b, F_soln, 0, 0);
    assembler->performBoundaryGather(b, params->Psol[0], 4, 0);
    
    for (size_t e=0; e<assembler->cells[b].size(); e++) {
      
      Kokkos::View<AD**,AssemblyDevice> obj = assembler->cells[b][e]->computeObjective(time, tindex, 0);
      Kokkos::View<GO**,HostDevice> paramGIDs = assembler->cells[b][e]->paramGIDs;
      int numElem = assembler->cells[b][e]->numElem;
      
      if (obj.extent(1) > 0) {
        for (int c=0; c<numElem; c++) {
          for (size_t i=0; i<obj.extent(1); i++) {
            totaldiff += obj(c,i);
            if (params->num_active_params > 0) {
              if (obj(c,i).size() > 0) {
                ScalarT val;
                val = obj(c,i).fastAccessDx(0);
                dmGradient[0] += val;
              }
            }
            
            if (params->globalParamUnknowns > 0) {
              for (int row=0; row<params->paramoffsets[0].size(); row++) {
                GO rowIndex = paramGIDs(c,params->paramoffsets[0][row]);
                int poffset = params->paramoffsets[0][row];
                ScalarT val;
                if (obj(c,i).size() > params->num_active_params) {
                  val = obj(c,i).fastAccessDx(poffset+params->num_active_params);
                  dmGradient[rowIndex+params->num_active_params] += val;
                }
              }
            }
          }
        }
      }
      
      if ((numDomainParams > 0)){// || (numBoundaryParams > 0)) {
        
        Kokkos::View<GO**,HostDevice> paramGIDs = assembler->cells[b][e]->paramGIDs;
        
        if (numDomainParams > 0) {
          int paramIndex, rowIndex, poffset;
          ScalarT val;
          regDomain = assembler->cells[b][e]->computeDomainRegularization(params->domainRegConstants,
                                                                          params->domainRegTypes,
                                                                          params->domainRegIndices);
          
          for (int c=0; c<numElem; c++) {
            for (size_t p = 0; p < numDomainParams; p++) {
              paramIndex = params->domainRegIndices[p];
              for( size_t row=0; row<params->paramoffsets[paramIndex].size(); row++ ) {
                if (regDomain.size() > 0) {
                  rowIndex = paramGIDs(c,params->paramoffsets[paramIndex][row]);
                  poffset = params->paramoffsets[paramIndex][row];
                  val = regDomain.fastAccessDx(poffset);
                  regGradient[rowIndex+params->num_active_params] += val;
                }
              }
            }
          }
        }
      }
    }
    bool usenewbcs = true;
    if (usenewbcs) {
      for (size_t e=0; e<assembler->boundaryCells[b].size(); e++) {
        if (numBoundaryParams > 0) {
          
          
          Kokkos::View<GO**,HostDevice> paramGIDs = assembler->boundaryCells[b][e]->paramGIDs;
          
          int paramIndex, rowIndex, poffset;
          ScalarT val;
          
          regBoundary = assembler->boundaryCells[b][e]->computeBoundaryRegularization(params->boundaryRegConstants,
                                                                                      params->boundaryRegTypes,
                                                                                      params->boundaryRegIndices,
                                                                                      params->boundaryRegSides);
          
          for (int c=0; c<assembler->boundaryCells[b][e]->numElem; c++) {
            for (size_t p = 0; p < numBoundaryParams; p++) {
              paramIndex = params->boundaryRegIndices[p];
              for( size_t row=0; row<params->paramoffsets[paramIndex].size(); row++ ) {
                if (regBoundary.size() > 0) {
                  rowIndex = paramGIDs(c,params->paramoffsets[paramIndex][row]);
                  poffset = params->paramoffsets[paramIndex][row];
                  val = regBoundary.fastAccessDx(poffset);
                  regGradient[rowIndex+params->num_active_params] += val;
                }
              }
            }
          }
        }
      }
    }
    else {
      /*
      for (size_t e=0; e<assembler->cells[b].size(); e++) {
        if (numBoundaryParams > 0) {
          
          Kokkos::View<GO**,HostDevice> paramGIDs = assembler->cells[b][e]->paramGIDs;
          
          int paramIndex, rowIndex, poffset;
          ScalarT val;
          
          regBoundary = assembler->cells[b][e]->computeBoundaryRegularization(params->boundaryRegConstants,
                                                                                      params->boundaryRegTypes,
                                                                                      params->boundaryRegIndices,
                                                                                      params->boundaryRegSides);
          
          for (int c=0; c<assembler->cells[b][e]->numElem; c++) {
            for (size_t p = 0; p < numBoundaryParams; p++) {
              paramIndex = params->boundaryRegIndices[p];
              for( size_t row=0; row<params->paramoffsets[paramIndex].size(); row++ ) {
                if (regBoundary.size() > 0) {
                  rowIndex = paramGIDs(c,params->paramoffsets[paramIndex][row]);
                  poffset = params->paramoffsets[paramIndex][row];
                  val = regBoundary.fastAccessDx(poffset);
                  regGradient[rowIndex+params->num_active_params] += val;
                }
              }
            }
          }
        }
      }*/
    }
    
    totaldiff += (regDomain + regBoundary);
    //totaldiff += phys->computeTopoResp(b);
  }
  
  //to gather contributions across processors
  ScalarT meep = 0.0;
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&totaldiff.val(),&meep);
  //Comm->SumAll(&totaldiff.val(), &meep, 1);
  totaldiff.val() = meep;
  
  DFAD fullobj(numParams,meep);
  
  for (size_t j=0; j< numParams; j++) {
    ScalarT dval;
    ScalarT ldval = dmGradient[j] + regGradient[j];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&ldval,&dval);
    //Comm->SumAll(&ldval,&dval,1);
    fullobj.fastAccessDx(j) = dval;
  }
  
  params->sacadoizeParams(false);
  
  return fullobj;
  
}

// ========================================================================================
// ========================================================================================

void solver::computeSensitivities(vector_RCP & u,
                          vector_RCP & a2, vector<ScalarT> & gradient) {
  
  DFAD obj_sens = this->computeObjective(u, current_time, 0);
  
  auto u_kv = u->getLocalView<HostDevice>();
  auto a2_kv = a2->getLocalView<HostDevice>();
  
  if (params->num_active_params > 0) {
  
    params->sacadoizeParams(true);
    
    vector<ScalarT> localsens(params->num_active_params);
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(LA_owned_map,params->num_active_params)); // reset residual
    matrix_RCP J = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_owned_map, maxEntries));//Tpetra::createCrsMatrix<ScalarT>(LA_owned_map); // reset Jacobian
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,params->num_active_params)); // reset residual
    matrix_RCP J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_overlapped_graph));;//Tpetra::createCrsMatrix<ScalarT>(LA_overlapped_map); // reset Jacobian
    
    auto res_kv = res->getLocalView<HostDevice>();
    
    res_over->putScalar(0.0);
    
    bool curradjstatus = useadjoint;
    useadjoint = false;
    
    assembler->assembleJacRes(u, u, false, true, false,
                              res_over, J_over, isTransient, current_time, useadjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time, deltat);
    useadjoint = curradjstatus;
    
    res->putScalar(0.0);
    res->doExport(*res_over, *exporter, Tpetra::ADD);
    
    for (size_t paramiter=0; paramiter < params->num_active_params; paramiter++) {
      // fine-scale
      if (assembler->cells[0][0]->cellData->multiscale) {
        ScalarT subsens = 0.0;
        for (size_t b=0; b<assembler->cells.size(); b++) {
          for (size_t e=0; e<assembler->cells[b].size(); e++) {
            subsens = -assembler->cells[b][e]->subgradient(0,paramiter);
            localsens[paramiter] += subsens;
          }
        }
      }
      else { // coarse-scale
      
        ScalarT currsens = 0.0;
        for( size_t i=0; i<LA_owned.size(); i++ ) {
          currsens += a2_kv(i,0) * res_kv(i,paramiter);
        }
        localsens[paramiter] = -currsens;
      }
      
    }
    
    
    ScalarT localval = 0.0;
    ScalarT globalval = 0.0;
    for (size_t paramiter=0; paramiter < params->num_active_params; paramiter++) {
      localval = localsens[paramiter];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      ScalarT cobj = 0.0;
      if (paramiter<obj_sens.size()) {
        cobj = obj_sens.fastAccessDx(paramiter);
      }
      globalval += cobj;
      if (gradient.size()<=paramiter) {
        gradient.push_back(globalval);
      }
      else {
        gradient[paramiter] += globalval;
      }
    }
    params->sacadoizeParams(false);
  }
  
  int numDiscParams = params->getNumParams(4);
  
  if (numDiscParams > 0) {
    //params->sacadoizeParams(false);
    vector_RCP a_owned = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1)); // adjoint solution
    auto ao_kv = a_owned->getLocalView<HostDevice>();
    
    for( size_t i=0; i<LA_owned.size(); i++ ) {
      ao_kv(i,0) = a2_kv(i,0);
    }
    
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // reset residual
    matrix_RCP J = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(params->param_owned_map, maxEntries));
    matrix_RCP J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(params->param_overlapped_map, maxEntries));
    
    res_over->putScalar(0.0);
    J->setAllToScalar(0.0);
    J_over->setAllToScalar(0.0);
    
    //this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, true, false, true, res_over, J_over);
    assembler->assembleJacRes(u, u, true, false, true,
                              res_over, J_over, isTransient, current_time, useadjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time, deltat);
    J_over->fillComplete(LA_owned_map, params->param_owned_map);
    
    vector_RCP sens_over = Teuchos::rcp(new LA_MultiVector(params->param_overlapped_map,1)); // reset residual
    vector_RCP sens = Teuchos::rcp(new LA_MultiVector(params->param_owned_map,1)); // reset residual
    auto sens_kv = sens->getLocalView<HostDevice>();
    
    J->setAllToScalar(0.0);
    J->doExport(*J_over, *(params->param_exporter), Tpetra::ADD);
    J->fillComplete(LA_owned_map, params->param_owned_map);
    
    J->apply(*a_owned,*sens);
    
    vector<ScalarT> discLocalGradient(numDiscParams);
    vector<ScalarT> discGradient(numDiscParams);
    for (size_t i = 0; i < params->paramOwned.size(); i++) {
      LO gid = params->paramOwned[i];
      discLocalGradient[gid] = sens_kv(i,0);
    }
    for (size_t i = 0; i < numDiscParams; i++) {
      ScalarT globalval = 0.0;
      ScalarT localval = discLocalGradient[i];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      ScalarT cobj = 0.0;
      if ((i+params->num_active_params)<obj_sens.size()) {
        cobj = obj_sens.fastAccessDx(i+params->num_active_params);
      }
      globalval += cobj;
      if (gradient.size()<=params->num_active_params+i) {
        gradient.push_back(globalval);
      }
      else {
        gradient[params->num_active_params+i] += globalval;
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

void solver::setDirichlet(vector_RCP & initial) {
  
  Teuchos::TimeMonitor localtimer(*initdbctimer);
  
  auto init_kv = initial->getLocalView<HostDevice>();
  //auto meas_kv = meas->getLocalView<HostDevice>();
  
  // TMW: this function needs to be fixed
  vector<vector<GO> > fixedDOFs = phys->dbc_dofs;
  
  for (size_t b=0; b<blocknames.size(); b++) {
    string blockID = blocknames[b];
    Kokkos::View<int**,HostDevice> side_info;
    
    for (int n=0; n<numVars[b]; n++) {
      
      vector<size_t> localDirichletSideIDs = phys->localDirichletSideIDs[b][n];
      vector<size_t> boundDirichletElemIDs = phys->boundDirichletElemIDs[b][n];
      int fnum = DOF->getFieldNum(varlist[b][n]);
      for( size_t e=0; e<disc->myElements[b].size(); e++ ) { // loop through all the elements
        side_info = phys->getSideInfo(b,n,e);
        int numSides = side_info.extent(0);
        DRV I_elemNodes;
        vector<size_t> elist(1);
        elist[0] = e;
        mesh->mesh->getElementVertices(elist,I_elemNodes);
        
        // enforce the boundary conditions if the element is on the given boundary
        
        for( int i=0; i<numSides; i++ ) {
          if( side_info(i,0)==1 ) {
            vector<GO> elemGIDs;
            int gside_index = side_info(i,1);
            string gside = phys->sideSets[gside_index];
            size_t elemID = disc->myElements[b][e];
            DOF->getElementGIDs(elemID, elemGIDs, blockID); // global index of each node
            // get the side index and the node->global mapping for the side that is on the boundary
            const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fnum, spaceDim-1, i);
            const vector<int> elmtOffset = SideIndex.first;
            const vector<int> basisIdMap = SideIndex.second;
            // for each node that is on the boundary side
            for( size_t j=0; j<elmtOffset.size(); j++ ) {
              // get the global row and coordinate
              LO row =  LA_overlapped_map->getLocalElement(elemGIDs[elmtOffset[j]]);
              ScalarT x = I_elemNodes(0,basisIdMap[j],0);
              ScalarT y = 0.0;
              if (spaceDim > 1) {
                y = I_elemNodes(0,basisIdMap[j],1);
              }
              ScalarT z = 0.0;
              if (spaceDim > 2) {
                z = I_elemNodes(0,basisIdMap[j],2);
              }
              
              if (use_meas_as_dbcs) {
                //init_kv(row,0) = meas_kv(row,0);
              }
              else {
                // put the value into the soln vector
                AD diri_FAD_tmp;
                diri_FAD_tmp = phys->getDirichletValue(b, x, y, z, current_time, varlist[b][n], gside, useadjoint, assembler->wkset[b]);
                
                init_kv(row,0) = diri_FAD_tmp.val();
              }
            }
          }
        }
      }
    }
    // set point dbcs
    vector<GO> dbc_dofs = fixedDOFs[b];
    
    for (int i = 0; i < dbc_dofs.size(); i++) {
      LO row = LA_overlapped_map->getLocalElement(dbc_dofs[i]);
      init_kv(row,0) = 0.0; // fix to zero for now
    }
    
  }
  
}

// ========================================================================================
// ========================================================================================

vector_RCP solver::setInitialParams() {
  vector_RCP initial = Teuchos::rcp(new LA_MultiVector(params->param_overlapped_map,1));
  ScalarT value = 2.0;
  initial->putScalar(value);
  return initial;
}

// ========================================================================================
// ========================================================================================

vector_RCP solver::setInitial() {
 
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::setInitial ..." << endl;
    }
  }
  
  vector_RCP initial = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
  vector_RCP glinitial = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
  initial->putScalar(0.0);
   
  if (initial_type == "L2-projection") {
    
    // Compute the L2 projection of the initial data into the discrete space
    vector_RCP rhs = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // reset residual
    matrix_RCP mass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_overlapped_graph));//Tpetra::createCrsMatrix<ScalarT>(LA_overlapped_map); // reset Jacobian
    vector_RCP glrhs = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1)); // reset residual
    matrix_RCP glmass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_owned_map, maxEntries));//Tpetra::createCrsMatrix<ScalarT>(LA_owned_map); // reset Jacobian
    assembler->setInitial(rhs, mass, useadjoint);
    
    glmass->setAllToScalar(0.0);
    glmass->doExport(*mass, *exporter, Tpetra::ADD);
    
    glrhs->putScalar(0.0);
    glrhs->doExport(*rhs, *exporter, Tpetra::ADD);
    
    glmass->fillComplete();
    
    this->linearSolver(glmass, glrhs, glinitial);
    have_preconditioner = false; // resetting this because mass matrix may not have connectivity as Jacobians
    initial->doImport(*glinitial, *importer, Tpetra::ADD);
    
  }
  else if (initial_type == "interpolation") {
    
    assembler->setInitial(initial, useadjoint);
    
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::setInitial ..." << endl;
    }
  }
  
  return initial;
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

void solver::linearSolver(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
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
      if (!have_preconditioner) {
        M = this->buildPreconditioner(J);
        have_preconditioner = true;
      }
      else {
        MueLu::ReuseTpetraPreconditioner(J,*M);
      }
      Problem->setLeftPrec(M);
    }
    Problem->setProblem();
    
    Teuchos::RCP<Teuchos::ParameterList> belosList = Teuchos::rcp(new Teuchos::ParameterList());
    belosList->set("Maximum Iterations",    liniter); // Maximum number of iterations allowed
    belosList->set("Convergence Tolerance", lintol);    // Relative convergence tolerance requested
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
    if (assembler->cells.size() == 1) {
      numEqns = numVars[0];
    }
    belosList->set("number of equations",numEqns);
    
    belosList->set("Output Style",          Belos::Brief);
    belosList->set("Implicit Residual Scaling", "Norm of Preconditioned Initial Residual");//None");
    
    Teuchos::RCP<Belos::SolverManager<ScalarT, LA_MultiVector, LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT, LA_MultiVector, LA_Operator>(Problem, belosList));
    
    solver->solve();
  }
}

// ========================================================================================
// Preconditioner for Tpetra stack
// ========================================================================================

Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > solver::buildPreconditioner(const matrix_RCP & J) {
  Teuchos::ParameterList mueluParams;
  
  mueluParams.setName("MueLu");
  
  // Main settings
  if (verbosity >= 20){
    mueluParams.set("verbosity","high");
  }
  else {
    mueluParams.set("verbosity","none");
  }
  //int numEqns = 1;
  //if (assembler->cells.size() == 1) {
  //  numEqns = numVars[0];
  //}
  //mueluParams.set("number of equations",numEqns);
  
  mueluParams.set("coarse: max size",500);
  mueluParams.set("multigrid algorithm", multigrid_type);
  
  // Aggregation
  mueluParams.set("aggregation: type","uncoupled");
  mueluParams.set("aggregation: drop scheme","classical");
  
  //Smoothing
  Teuchos::ParameterList smootherParams = mueluParams.sublist("smoother: params");
  mueluParams.set("smoother: type",smoother_type);
  if (smoother_type == "CHEBYSHEV") {
    mueluParams.sublist("smoother: params").set("chebyshev: degree",2);
    mueluParams.sublist("smoother: params").set("chebyshev: ratio eigenvalue",7.0);
    mueluParams.sublist("smoother: params").set("chebyshev: min eigenvalue",1.0);
    mueluParams.sublist("smoother: params").set("chebyshev: zero starting solution",true);
  }
  else if (smoother_type == "RELAXATION") {
    mueluParams.sublist("smoother: params").set("relaxation: type","Jacobi");
  }
  
  // Repartitioning
  
  mueluParams.set("repartition: enable",false);
  mueluParams.set("repartition: partitioner","zoltan");
  mueluParams.set("repartition: start level",2);
  mueluParams.set("repartition: min rows per proc",800);
  mueluParams.set("repartition: max imbalance", 1.1);
  mueluParams.set("repartition: remap parts",false);
  
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > Mnew = MueLu::CreateTpetraPreconditioner((Teuchos::RCP<LA_Operator>)J, mueluParams);

  //have_preconditioner = true;
  return Mnew;
}

// ========================================================================================
// ========================================================================================

void solver::setBatchID(const int & bID){
  batchID = bID;
  params->batchID = bID;
}

// ========================================================================================
// ========================================================================================

vector_RCP solver::blankState(){
  vector_RCP F_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
  return F_soln;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void solver::finalizeMultiscale() {
  if (multiscale_manager->subgridModels.size() > 0 ) {
    for (size_t k=0; k<multiscale_manager->subgridModels.size(); k++) {
      multiscale_manager->subgridModels[k]->paramvals_KVAD = params->paramvals_KVAD;
    //  multiscale_manager->subgridModels[k]->wkset[0]->paramnames = paramnames;
    }
    
    multiscale_manager->macro_wkset = assembler->wkset;
    
    multiscale_manager->setMacroInfo(disc->basis_pointers, disc->basis_types,
                                     phys->varlist, useBasis, phys->offsets,
                                     params->paramnames, params->discretized_param_names);
    
    ScalarT my_cost = multiscale_manager->initialize();
    ScalarT gmin = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
    //Comm->MinAll(&my_cost, &gmin, 1);
    ScalarT gmax = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);
    //Comm->MaxAll(&my_cost, &gmax, 1);
    
    if(Comm->getRank() == 0 && verbosity>0) {
      cout << "***** Load Balancing Factor " << gmax/gmin <<  endl;
    }
    
    
  }

}
