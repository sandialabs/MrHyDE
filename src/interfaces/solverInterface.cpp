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

#include "solverInterface.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class Node>
solver<Node>::solver(const Teuchos::RCP<MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings_,
               Teuchos::RCP<meshInterface> & mesh_,
               Teuchos::RCP<discretization> & disc_,
               Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager> & DOF_,
               Teuchos::RCP<AssemblyManager<Node> > & assembler_,
               Teuchos::RCP<ParameterManager<Node> > & params_) :
Comm(Comm_), settings(settings_), mesh(mesh_), disc(disc_), phys(phys_), DOF(DOF_), assembler(assembler_), params(params_) {
  
  milo_debug_level = settings->get<int>("debug level",0);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver constructor ..." << endl;
    }
  }
  
  soln = Teuchos::rcp(new SolutionStorage<Node>(settings));
  string analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  if (analysis_type == "forward+adjoint" || analysis_type == "ROL" || analysis_type == "ROL_SIMOPT") {
    save_solution = true; // default is false
    if (settings->sublist("Analysis").sublist("ROL").sublist("General").get<bool>("Generate data",false)) {
      datagen_soln = Teuchos::rcp(new SolutionStorage<Node>(settings));
    }
  }
  
  numEvaluations = 0;
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
  discrete_objective_scale_factor = settings->sublist("Postprocess").get("scale factor for discrete objective",1.0);
  compute_sensitivity = settings->sublist("Postprocess").get("compute sensitivities",false);
  compute_aux_sensitivity = settings->sublist("Solver").get("compute aux sensitivities",false);
  compute_flux = settings->sublist("Solver").get("compute flux",false);
  
  initial_type = settings->sublist("Solver").get<string>("initial type","L2-projection");
  multigrid_type = settings->sublist("Solver").get<string>("multigrid type","sa");
  smoother_type = settings->sublist("Solver").get<string>("smoother type","CHEBYSHEV"); // or RELAXATION
  preconditioner_reuse_type = settings->sublist("Solver").get<string>("preconditioner reuse type","none"); // or RELAXATION
  useLinearSolver = settings->sublist("Solver").get<bool>("use linear solver",true);
  lintol = settings->sublist("Solver").get<ScalarT>("linear TOL",1.0E-7);
  liniter = settings->sublist("Solver").get<int>("max linear iters",100);
  kspace = settings->sublist("Solver").get<int>("krylov vectors",100);
  useDomDecomp = settings->sublist("Solver").get<bool>("use dom decomp",false);
  useDirect = settings->sublist("Solver").get<bool>("use direct solver",false);
  usePrec = settings->sublist("Solver").get<bool>("use preconditioner",true);
  usePrecDBC = settings->sublist("Solver").get<bool>("use preconditioner for DBCs",true);
  dropTol = settings->sublist("Solver").get<ScalarT>("ILU drop tol",0.0); //defaults to AztecOO default
  fillParam = settings->sublist("Solver").get<ScalarT>("ILU fill param",3.0); //defaults to AztecOO default
  reuse_preconditioner = settings->sublist("Solver").get<bool>("reuse preconditioner",false); //defaults to AztecOO default
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
      //int vnum = DOF->getFieldNum(var);
      int vub = phys->getUniqueIndex(b,var);
      currvarlist[j] = var;
      curruseBasis[j] = vub;
      currnumBasis[j] = cards[b][vub];
      currmaxBasis = std::max(currmaxBasis,cards[b][vub]);
    }
    
    phys->setVars(b,currvarlist);
    
    varlist.push_back(currvarlist);
    useBasis.push_back(curruseBasis);
    numBasis.push_back(currnumBasis);
    maxBasis.push_back(currmaxBasis);
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Tpetra maps maxEntries per row (not quite precise)
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
    mesh->readMeshData();//LA_overlapped_map, assembler->cells);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  
  this->setBatchID(Comm->getRank());
  
  /////////////////////////////////////////////////////////////////////////////
  
  this->setupFixedDOFs(settings);
  
  /////////////////////////////////////////////////////////////////////////////
  scalarInitialData = settings->sublist("Physics").sublist("Initial conditions").get<bool>("scalar data", false);
  
  if (scalarInitialData) {
    for (size_t b=0; b<blocknames.size(); b++) {
      
      std::string blockID = blocknames[b];
      Teuchos::ParameterList init_settings;
      if (settings->sublist("Physics").isSublist(blockID)) {
        init_settings = settings->sublist("Physics").sublist(blockID).sublist("Initial conditions");
      }
      else {
        init_settings = settings->sublist("Physics").sublist("Initial conditions");
      }
      vector<ScalarT> blockInitialValues;
      
      for (size_t var=0; var<varlist[b].size(); var++ ) {
        ScalarT value = 0.0;
        if (init_settings.isSublist(varlist[b][var])) {
          Teuchos::ParameterList currinit = init_settings.sublist(varlist[b][var]);
          Teuchos::ParameterList::ConstIterator i_itr = currinit.begin();
          while (i_itr != currinit.end()) {
            value = currinit.get<ScalarT>(i_itr->first);
            i_itr++;
          }
        }
        blockInitialValues.push_back(value);
      }
      scalarInitialValues.push_back(blockInitialValues);
    }
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver constructor" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void solver<Node>::setButcherTableau(const string & tableau) {
  Kokkos::View<ScalarT**,AssemblyDevice> dev_butcher_A;
  Kokkos::View<ScalarT*,AssemblyDevice> dev_butcher_b, dev_butcher_c;
  //auto butcher_A = Kokkos::create_mirror_view(dev_butcher_A);
  //auto butcher_b = Kokkos::create_mirror_view(dev_butcher_b);
  //auto butcher_c = Kokkos::create_mirror_view(dev_butcher_c);
  
  Kokkos::View<ScalarT**,HostDevice> butcher_A;
  Kokkos::View<ScalarT*,HostDevice> butcher_b, butcher_c;
  
  //only filling in the non-zero entries
  if (tableau == "BWE" || tableau == "DIRK-1,1") {
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
    butcher_A(0,0) = 1.0;
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
    butcher_b(0) = 1.0;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
    butcher_c(0) = 1.0;
  }
  else if (tableau == "FWE") {
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
    butcher_b(0) = 1.0;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
  }
  else if (tableau == "CN") {
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
    butcher_A(1,0) = 0.5;
    butcher_A(1,1) = 0.5;
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
    butcher_b(0) = 0.5;
    butcher_b(1) = 0.5;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
    butcher_c(1) = 1.0;
  }
  else if (tableau == "SSPRK-3,3") {
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",3,3);
    butcher_A(1,0) = 1.0;
    butcher_A(2,0) = 0.25;
    butcher_A(2,1) = 0.25;
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",4);
    butcher_b(0) = 1.0/6.0;
    butcher_b(1) = 1.0/6.0;
    butcher_b(2) = 2.0/3.0;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",4);
    butcher_c(1) = 1.0;
    butcher_c(2) = 1.0/2.0;
  }
  else if (tableau == "RK-4,4") { // Classical RK4
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",4,4);
    butcher_A(1,0) = 0.5;
    butcher_A(2,1) = 0.5;
    butcher_A(3,2) = 1.0;
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",4);
    butcher_b(0) = 1.0/6.0;
    butcher_b(1) = 1.0/3.0;
    butcher_b(2) = 1.0/3.0;
    butcher_b(3) = 1.0/6.0;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",4);
    butcher_c(1) = 1.0/2.0;
    butcher_c(2) = 1.0/2.0;
    butcher_c(3) = 1.0;
  }
  else if (tableau == "DIRK-1,2") {
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
    butcher_A(0,0) = 0.5;
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
    butcher_b(0) = 1.0;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
    butcher_c(0) = 0.5;
  }
  else if (tableau == "DIRK-2,2") { // 2-stage, 2nd order
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
    butcher_A(0,0) = 1.0/4.0;
    butcher_A(1,0) = 1.0/2.0;
    butcher_A(1,1) = 1.0/4.0;
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
    butcher_b(0) = 1.0/2.0;
    butcher_b(1) = 1.0/2.0;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
    butcher_c(0) = 1.0/4.0;
    butcher_c(1) = 3.0/4.0;
  }
  else if (tableau == "DIRK-2,3") { // 2-stage, 3rd order
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
    butcher_A(0,0) = 1.0/2.0 + std::sqrt(3)/6.0;
    butcher_A(1,0) = -std::sqrt(3)/3.0;
    butcher_A(1,1) = 1.0/2.0  + std::sqrt(3)/6.0;
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
    butcher_b(0) = 1.0/2.0;
    butcher_b(1) = 1.0/2.0;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
    butcher_c(0) = 1.0/2.0 + std::sqrt(3)/6.0;;
    butcher_c(1) = 1.0/2.0 - std::sqrt(3)/6.0;;
  }
  else if (tableau == "DIRK-3,3") { // 3-stage, 3rd order
    ScalarT p = 0.4358665215;
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",3,3);
    butcher_A(0,0) = p;
    butcher_A(1,0) = (1.0-p)/2.0;
    butcher_A(1,1) = p;
    butcher_A(2,0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
    butcher_A(2,1) = 3.0*p*p/2.0 - 5.0*p + 5.0/4.0;
    butcher_A(2,2) = p;
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",3);
    butcher_b(0) = -3.0*p*p/2.0+4.0*p-1.0/4.0;
    butcher_b(1) = 3.0*p*p/2.0-5.0*p+5.0/4.0;
    butcher_b(2) = p;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",3);
    butcher_c(0) = p;
    butcher_c(1) = (1.0+p)/2.0;
    butcher_c(2) = 1.0;
  }
  else if (tableau == "custom") {
    
    string delimiter = ", ";
    string line_delimiter = "; ";
    size_t pos = 0;
    string b_A = settings->sublist("Solver").get<string>("transient Butcher A","1.0");
    string b_b = settings->sublist("Solver").get<string>("transient Butcher b","1.0");
    string b_c = settings->sublist("Solver").get<string>("transient Butcher c","1.0");
    vector<vector<double>> A_vals;
    if (b_A.find(delimiter) == string::npos) {
      vector<double> row;
      row.push_back(std::stod(b_A));
      A_vals.push_back(row);
    }
    else {
      string token;
      size_t linepos = 0;
      vector<string> lines;
      while ((linepos = b_A.find(line_delimiter)) != string::npos) {
        string line = b_A.substr(0,linepos);
        lines.push_back(line);
        b_A.erase(0, linepos + line_delimiter.length());
      }
      lines.push_back(b_A);
      for (size_t k=0; k<lines.size(); k++) {
        string line = lines[k];
        vector<double> row;
        while ((pos = line.find(delimiter)) != string::npos) {
          token = line.substr(0, pos);
          row.push_back(std::stod(token));
          line.erase(0, pos + delimiter.length());
        }
        row.push_back(std::stod(line));
        A_vals.push_back(row);
      }
    }
    // Make sure A is square
    size_t A_nrows = A_vals.size();
    for (size_t i=0; i<A_nrows; i++) {
      if (A_vals[i].size() != A_nrows) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: custom Butcher A is not a square matrix");
      }
    }
    
    vector<double> b_vals;
    if (b_b.find(delimiter) == string::npos) {
      b_vals.push_back(std::stod(b_b));
    }
    else {
      string token;
      while ((pos = b_b.find(delimiter)) != string::npos) {
        token = b_b.substr(0, pos);
        b_vals.push_back(std::stod(token));
        b_b.erase(0, pos + delimiter.length());
      }
      b_vals.push_back(std::stod(b_b));
    }
    
    // Make sure size of b matches A
    if (b_vals.size() != A_nrows) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: custom Butcher b does not match size of A");
    }
    
    vector<double> c_vals;
    if (b_c.find(delimiter) == string::npos) {
      c_vals.push_back(std::stod(b_c));
    }
    else {
      string token;
      while ((pos = b_c.find(delimiter)) != string::npos) {
        token = b_c.substr(0, pos);
        c_vals.push_back(std::stod(token));
        b_c.erase(0, pos + delimiter.length());
      }
      c_vals.push_back(std::stod(b_c));
    }
    
    // Make sure size of c matches A
    if (c_vals.size() != A_nrows) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: custom Butcher c does not match size of A");
    }
    
    // Create the views
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",A_nrows,A_nrows);
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",A_nrows);
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",A_nrows);
    for (size_t i=0; i<A_nrows; i++) {
      for (size_t j=0; j<A_nrows; j++) {
        butcher_A(i,j) = A_vals[i][j];
      }
      butcher_b(i) = b_vals[i];
      butcher_c(i) = c_vals[i]; 
    }
    
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Butcher tableau:" + tableau);
  }
  dev_butcher_A = Kokkos::View<ScalarT**,AssemblyDevice>("butcher_A on device",butcher_A.extent(0),butcher_A.extent(1));
  dev_butcher_b = Kokkos::View<ScalarT*,AssemblyDevice>("butcher_b on device",butcher_b.extent(0));
  dev_butcher_c = Kokkos::View<ScalarT*,AssemblyDevice>("butcher_c on device",butcher_c.extent(0));
  
  auto tmp_butcher_A = Kokkos::create_mirror_view(dev_butcher_A);
  auto tmp_butcher_b = Kokkos::create_mirror_view(dev_butcher_b);
  auto tmp_butcher_c = Kokkos::create_mirror_view(dev_butcher_c);
  
  Kokkos::deep_copy(tmp_butcher_A, butcher_A);
  Kokkos::deep_copy(tmp_butcher_b, butcher_b);
  Kokkos::deep_copy(tmp_butcher_c, butcher_c);
  
  Kokkos::deep_copy(dev_butcher_A, tmp_butcher_A);
  Kokkos::deep_copy(dev_butcher_b, tmp_butcher_b);
  Kokkos::deep_copy(dev_butcher_c, tmp_butcher_c);
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    assembler->wkset[b]->butcher_A = dev_butcher_A;
    assembler->wkset[b]->butcher_b = dev_butcher_b;
    assembler->wkset[b]->butcher_c = dev_butcher_c;
  }
  int newnumstages = butcher_A.extent(0);
  numstages = std::max(numstages,newnumstages);
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void solver<Node>::setBackwardDifference(const int & order) { // using order as an input to allow for dynamic changes
  
  Kokkos::View<ScalarT*,AssemblyDevice> dev_BDF_wts;
  //Kokkos::View<ScalarT*,HostDevice> BDF_wts; // not using a mirror view on purpose
  //auto BDF_wts = Kokkos::create_mirror_view(dev_BDF_wts);
  Kokkos::View<ScalarT*,HostDevice> BDF_wts;
  // Note that these do not include 1/deltat (added in wkset)
  // Not going to work properly for adaptive time stepping if BDForder>1
  if (isTransient) {
    
    if (order == 1) {
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",2);
      BDF_wts(0) = 1.0;
      BDF_wts(1) = -1.0;
    }
    else if (order == 2) {
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",3);
      BDF_wts(0) = 1.5;
      BDF_wts(1) = -2.0;
      BDF_wts(2) = 0.5;
    }
    else if (order == 3) {
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",4);
      BDF_wts(0) = 11.0/6.0;
      BDF_wts(1) = -3.0;
      BDF_wts(2) = 1.5;
      BDF_wts(3) = -1.0/3.0;
    }
    else if (order == 4) {
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",5);
      BDF_wts(0) = 25.0/12.0;
      BDF_wts(1) = -4.0;
      BDF_wts(2) = 3.0;
      BDF_wts(3) = -4.0/3.0;
      BDF_wts(4) = 1.0/4.0;
    }
    else if (order == 5) {
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",6);
      BDF_wts(0) = 137.0/60.0;
      BDF_wts(1) = -5.0;
      BDF_wts(2) = 5.0;
      BDF_wts(3) = -10.0/3.0;
      BDF_wts(4) = 75.0/60.0;
      BDF_wts(5) = -1.0/5.0;
    }
    else if (order == 6) {
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",7);
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
    BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",1);
    BDF_wts(0) = 1.0;
    numsteps = 1;
  }
  
  dev_BDF_wts = Kokkos::View<ScalarT*,AssemblyDevice>("BDF weights on device",BDF_wts.extent(0));
  Kokkos::deep_copy(dev_BDF_wts, BDF_wts);
  for (size_t b=0; b<assembler->cells.size(); b++) {
    assembler->wkset[b]->BDF_wts = dev_BDF_wts;
  }
  
  //KokkosTools::print(BDF_wts);
  
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void solver<Node>::finalizeWorkset() {
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::finalizeWorkset ..." << endl;
    }
  }
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    
    if (assembler->cells[b].size() > 0) {
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
      int numDOF = assembler->cells[b][0]->LIDs.extent(1);
      for (size_t e=0; e<assembler->cells[b].size(); e++) {
        assembler->cells[b][e]->setWorkset(assembler->wkset[b]);
        assembler->cells[b][e]->setUseBasis(useBasis[b], numsteps, numstages);
        assembler->cells[b][e]->setUpAdjointPrev(numDOF, numsteps, numstages);
        assembler->cells[b][e]->setUpSubGradient(params->num_active_params);
      }
      assembler->wkset[b]->params = params->paramvals_AD;
      assembler->wkset[b]->params_AD = params->paramvals_KVAD;
      assembler->wkset[b]->paramnames = params->paramnames;
      //assembler->wkset[b]->setupParamBasis(discretized_param_basis);
      assembler->wkset[b]->setTime(current_time);
      if (assembler->boundaryCells.size() > b) { // avoid seg faults
        for (size_t e=0; e<assembler->boundaryCells[b].size(); e++) {
          if (assembler->boundaryCells[b][e]->numElem > 0) {
            assembler->boundaryCells[b][e]->setWorkset(assembler->wkset[b]);
            assembler->boundaryCells[b][e]->setUseBasis(useBasis[b], numsteps, numstages);
          }
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

template<class Node>
void solver<Node>::setupLinearAlgebra() {
  
  Teuchos::TimeMonitor localtimer(*setupLAtimer);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::setupLinearAlgebra..." << endl;
    }
  }
  
  LA_owned_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, LA_owned, 0, Comm));
  LA_overlapped_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, LA_ownedAndShared, 0, Comm));
  LA_overlapped_graph = Teuchos::rcp( new LA_CrsGraph(LA_overlapped_map,maxEntries));
  
  exporter = Teuchos::rcp(new LA_Export(LA_overlapped_map, LA_owned_map));
  importer = Teuchos::rcp(new LA_Import(LA_owned_map, LA_overlapped_map));
  
  // Fill in the graph
  for (size_t b=0; b<blocknames.size(); b++) {
    vector<size_t> EIDs = disc->myElements[b];
    for (size_t e=0; e<EIDs.size(); e++) {
      vector<GO> gids;
      size_t elemID = EIDs[e];
      DOF->getElementGIDs(elemID, gids, blocknames[b]);
      for (size_t i=0; i<gids.size(); i++) {
        GO ind1 = gids[i];
        LA_overlapped_graph->insertGlobalIndices(ind1,gids);
      }
    }
  }
  
  LA_overlapped_graph->fillComplete();
  
  
  if (params->num_discretized_params > 0) {
    // Fill in the graph
    params->param_overlapped_graph = Teuchos::rcp( new LA_CrsGraph(params->param_overlapped_map,maxEntries));
    
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<size_t> EIDs = disc->myElements[b];
      for (size_t e=0; e<EIDs.size(); e++) {
        vector<GO> gids;
        size_t elemID = EIDs[e];
        params->paramDOF->getElementGIDs(elemID, gids, blocknames[b]);
        vector<GO> stategids;
        DOF->getElementGIDs(elemID, stategids, blocknames[b]);
        for (size_t i=0; i<gids.size(); i++) {
          GO ind1 = gids[i];
          params->param_overlapped_graph->insertGlobalIndices(ind1,stategids);
        }
      }
    }
    
    params->param_overlapped_graph->fillComplete(LA_owned_map, params->param_owned_map);
    
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::setupLinearAlgebra" << endl;
    }
  }
  
}

// ========================================================================================
// Set up the logicals and data structures for the fixed DOF (Dirichlet and point constraints)
// ========================================================================================

template<class Node>
void solver<Node>::setupFixedDOFs(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  Teuchos::TimeMonitor localtimer(*fixeddofsetuptimer);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::setupFixedDOFs()" << endl;
    }
  }
  
  if (!phys->haveDirichlet) {
    usestrongDBCs = false;
  }
  
  if (usestrongDBCs) {
    fixedDOF_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    
    scalarDirichletData = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("scalar data", false);
    transientDirichletData = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("transient data", false);
    
    if (scalarDirichletData && transientDirichletData) {
      if (Comm->getRank() == 0) {
        cout << "Warning: Both scalar data and transient data were set to true.  This should not happen." << endl;
      }
    }
    
    if (scalarDirichletData) {
      for (size_t b=0; b<blocknames.size(); b++) {
        
        std::string blockID = blocknames[b];
        Teuchos::ParameterList dbc_settings;
        if (settings->sublist("Physics").isSublist(blockID)) {
          dbc_settings = settings->sublist("Physics").sublist(blockID).sublist("Dirichlet conditions");
        }
        else {
          dbc_settings = settings->sublist("Physics").sublist("Dirichlet conditions");
        }
        vector<ScalarT> blockDirichletValues;
        
        for (size_t var=0; var<varlist[b].size(); var++ ) {
          ScalarT value = 0.0;
          if (dbc_settings.isSublist(varlist[b][var])) {
            if (dbc_settings.sublist(varlist[b][var]).isParameter("all boundaries")) {
              value = dbc_settings.sublist(varlist[b][var]).get<ScalarT>("all boundaries");
            }
            else {
              Teuchos::ParameterList currdbcs = dbc_settings.sublist(varlist[b][var]);
              Teuchos::ParameterList::ConstIterator d_itr = currdbcs.begin();
              while (d_itr != currdbcs.end()) {
                value = currdbcs.get<ScalarT>(d_itr->first);
                d_itr++;
              }
            }
          }
          blockDirichletValues.push_back(value);
        }
        scalarDirichletValues.push_back(blockDirichletValues);
      }
    }
  }
    
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::setupFixedDOFs()" << endl;
    }
  }
  
}

// ========================================================================================
// Set up the logicals and data structures for the fixed DOF (Dirichlet and point constraints)
// ========================================================================================

template<class Node>
void solver<Node>::projectDirichlet() {

  Teuchos::TimeMonitor localtimer(*dbcprojtimer);
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::projectDirichlet()" << endl;
    }
  }
  if (usestrongDBCs) {
    fixedDOF_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP glfixedDOF_soln = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    
    vector_RCP rhs = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    matrix_RCP mass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(LA_overlapped_graph));
    vector_RCP glrhs = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    matrix_RCP glmass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(LA_owned_map, maxEntries));
    
    assembler->setDirichlet(rhs, mass, useadjoint, current_time);
    
    glmass->setAllToScalar(0.0);
    glmass->doExport(*mass, *exporter, Tpetra::ADD);
    glrhs->putScalar(0.0);
    glrhs->doExport(*rhs, *exporter, Tpetra::ADD);
    
    glmass->fillComplete();
    
    if (milo_debug_level>2) {
      //KokkosTools::print(glmass,"L2-projection matrix for DBCs");
      //KokkosTools::print(glrhs,"L2-projections RHS for DBCs");
    }
    
    if (useDirect) {
      Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > L2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", glmass, glrhs, glfixedDOF_soln);
      L2Solver->symbolicFactorization();
      L2Solver->setA(glmass, Amesos2::SYMBFACT);
      L2Solver->setX(glfixedDOF_soln);
      L2Solver->setB(glrhs);
      L2Solver->numericFactorization().solve();
    }
    else {
      Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(glmass, glfixedDOF_soln, glrhs));
      if (usePrecDBC) {
        Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > P = this->buildPreconditioner(glmass);
        Problem->setLeftPrec(P);
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
    fixedDOF_soln->doImport(*glfixedDOF_soln, *importer, Tpetra::ADD);
  }
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::projectDirichlet()" << endl;
    }
  }
  
}

// ========================================================================================
/* given the parameters, solve the forward  problem */
// ========================================================================================

template<class Node>
void solver<Node>::forwardModel(DFAD & obj) {
  
  current_time = initial_time;
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::forwardModel ..." << endl;
    }
  }
  
  useadjoint = false;
  params->sacadoizeParams(false);
  
  if (!scalarDirichletData && !transientDirichletData) {
    this->projectDirichlet();
  }
  vector_RCP u = this->setInitial();
 
  Kokkos::fence();
 
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
  
  if (postproc->write_optimization_solution) {
    postproc->writeOptimizationSolution(numEvaluations);
  }
  
  numEvaluations++;
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::forwardModel" << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void solver<Node>::adjointModel(vector<ScalarT> & gradient) {
  
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
    if (!fnd) {
      //throw error
    }
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

template<class Node>
void solver<Node>::transientSolver(vector_RCP & initial, DFAD & obj, vector<ScalarT> & gradient,
                             ScalarT & start_time, ScalarT & end_time) {
  
  Teuchos::TimeMonitor localtimer(*transientsolvertimer);
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting solver::transientSolver ..." << endl;
      cout << "******** Start time = " << start_time << endl;
      cout << "******** End time = " << end_time << endl;
      cout << "******** Time step size = " << deltat << endl;
    }
  }
  
  vector_RCP zero_vec = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
  zero_vec->putScalar(0.0);
  
  current_time = start_time;
  if (!useadjoint) { // forward solve - adaptive time stepping
    is_final_time = false;
    vector_RCP u = initial;
    
    if (usestrongDBCs) {
      this->setDirichlet(u);
    }
    {
      assembler->performGather(u,0,0);
      postproc->record(current_time);
    }
    Kokkos::fence();
 
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
        
        // Make sure last step solution is gathered
        // Last set of values is from a stage solution, which is potentially different
        assembler->performGather(u,0,0);
        
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
        if (!fnd) {
          // throw error
        }
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
    vector_RCP phi = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    
    size_t numFwdSteps = soln->times[0].size()-1;
    
    //bool fndu = soln->extract(u, numsteps);
    //assembler->performGather(0,u,0,0);
    
    for (size_t timeiter = 0; timeiter<numFwdSteps; timeiter++) {
      size_t cindex = numFwdSteps-timeiter;
      vector_RCP phi_prev = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
      phi_prev->update(1.0,*phi,0.0);
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
      if (!fndu) {
        // throw error
      }
      bool fndup = soln->extract(u_prev, cindex-1);
      if (!fndup) {
        // throw error
      }
      assembler->performGather(u_prev,0,0);
      assembler->resetPrevSoln();
      
      current_time = soln->times[0][cindex-1];
      
      // if multistage, recover forward solution at each stage
      if (numstages == 1) { // No need to re-solve in this case
        int status = this->nonlinearSolver(u, phi);
        if (status>0) {
          // throw error
        }
        this->computeSensitivities(u,phi,gradient);
      }
      else {
        useadjoint = false;
        std::vector<vector_RCP> stage_solns;
        for (int stage = 0; stage<numstages; stage++) {
          // Need a stage solution
          vector_RCP u_stage = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
          // Set the initial guess for stage solution
          u_stage->update(1.0,*u,0.0);
          
          assembler->updateStageNumber(stage); // could probably just += 1 in wksets
          
          int status = this->nonlinearSolver(u_stage, zero_vec);
          if (status>0) {
            // throw error
          }
          stage_solns.push_back(u_stage);
          //u->update(1.0, *u_stage, 1.0);
          //u->update(-1.0, *u_prev, 1.0);
          assembler->updateStageSoln(); // moves the stage solution into u_stage (avoids mem transfer)
        }
        useadjoint = true;
        //assembler->setAlternateSolution(u);
        /*
        auto vec_kv = u_prev->getLocalView<HostDevice>();
        
        Kokkos::View<LO*,AssemblyDevice> numDOF;
        Kokkos::View<ScalarT***,AssemblyDevice> data;
        Kokkos::View<int**,AssemblyDevice> offsets;
        LIDView LIDs;
        
        for (size_t b=0; b<assembler->cells.size(); b++) {
          for (size_t c=0; c<assembler->cells[b].size(); c++) {
            LIDs = assembler->cells[b][c]->LIDs;
            numDOF = assembler->cells[b][c]->cellData->numDOF;
            auto cellu = assembler->cells[b][c]->u;
            assembler->cells[b][c]->u_alt = Kokkos::View<ScalarT***,AssemblyDevice>("alternative solution",cellu.extent(0),cellu.extent(1),cellu.extent(2));
            data = assembler->cells[b][c]->u_alt;
            offsets = assembler->wkset[b]->offsets;
            
            parallel_for("assembly gather",RangePolicy<AssemblyExec>(0,data.extent(0)), KOKKOS_LAMBDA (const int elem ) {
              for (size_t var=0; var<offsets.extent(0); var++) {
                for(size_t dof=0; dof<numDOF(var); dof++ ) {
                  data(elem,var,dof) = vec_kv(LIDs(elem,offsets(var,dof)),0);
                }
              }
            });
            
            assembler->cells[b][c]->usealtsol = false;
          }
        }
          */
        vector<double> stage_grad(gradient.size(),0.0);
        
        for (int stage = numstages-1; stage>=0; stage--) {
          // Need a stage solution
          vector_RCP phi_stage = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
          // Set the initial guess for stage solution
          phi_stage->update(1.0,*phi,0.0);
          
          assembler->updateStageNumber(stage); // could probably just += 1 in wksets
          
          int status = this->nonlinearSolver(stage_solns[stage], phi_stage);
          if (status>0) {
            // throw error
          }
          phi->update(1.0, *phi_stage, 1.0);
          phi->update(-1.0, *phi_prev, 1.0);
          //assembler->updateStageSoln(); // moves the stage solution into u_stage (avoids mem transfer)
          //this->computeSensitivities(stage_solns[stage],phi_stage,stage_grad);
        }
        this->computeSensitivities(u,phi,gradient);
        //for (int k=0; k<gradient.size(); k++) {
        //  gradient[k] += stage_grad[k] - (numstages-1)*gradient[k];
        //}
        /*for (size_t b=0; b<assembler->cells.size(); b++) {
          for (size_t c=0; c<assembler->cells[b].size(); c++) {
            assembler->cells[b][c]->usealtsol = false;
          }
        }*/
      }
      //KokkosTools::print(phi);
      
      //this->computeSensitivities(u,phi,gradient);
      
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

template<class Node>
int solver<Node>::nonlinearSolver(vector_RCP & u, vector_RCP & phi) {
  
  Teuchos::TimeMonitor localtimer(*nonlinearsolvertimer);
  
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
  
  while (NLerr_scaled[0]>NLtol && NLiter<maxiter) { // while not converged and not reached max iterations
    
    multiscale_manager->reset();
    
    gNLiter = NLiter;
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    matrix_RCP J = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(LA_owned_map, maxEntries));
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    matrix_RCP J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(LA_overlapped_graph));
    vector_RCP du_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP du = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    
    // *********************** COMPUTE THE JACOBIAN AND THE RESIDUAL **************************
    
    bool build_jacobian = true;
    {
      Teuchos::TimeMonitor localtimer(*resetLAtimer);
      res_over->putScalar(0.0);
      J_over->setAllToScalar(0.0);
    }
    {
      Teuchos::TimeMonitor localtimer(*fillcompleteLAtimer);
      J_over->fillComplete();
    }
    
    J_over->resumeFill();
    if ( useadjoint && (NLiter == 1)) {
      store_adjPrev = true;
    }
    else {
      store_adjPrev = false;
    }
    
    assembler->assembleJacRes(u, phi, build_jacobian, false, false,
                              res_over, J_over, isTransient, current_time, useadjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time, deltat);
    {
      Teuchos::TimeMonitor localtimer(*fillcompleteLAtimer);
      J_over->fillComplete();
    }
    {
      Teuchos::TimeMonitor localtimer(*exportLAtimer);
      J->doExport(*J_over, *exporter, Tpetra::ADD);
    }
    {
      Teuchos::TimeMonitor localtimer(*fillcompleteLAtimer);
      J->fillComplete();
    }
    
    if (useadjoint && response_type == "discrete") {
      vector_RCP D_soln;
      bool fnd = datagen_soln->extract(D_soln, 0, current_time+deltat);
      if (fnd) {
        vector_RCP diff = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
        diff->update(1.0, *u, 0.0);
        diff->update(-1.0, *D_soln, 1.0);
        res_over->update(-1.0*discrete_objective_scale_factor,*diff,1.0);
      }
      else {
        std::cout << "Error: did not find a data-generating solution" << std::endl;
      }
    }
    
    {
      Teuchos::TimeMonitor localtimer(*resetLAtimer);
      res->putScalar(0.0);
    }
    {
      Teuchos::TimeMonitor localtimer(*exportLAtimer);
      res->doExport(*res_over, *exporter, Tpetra::ADD);
    }
    
      
    if (milo_debug_level>2) {
      //KokkosTools::print(J,"Jacobian from solver interface");
      //KokkosTools::print(res,"residual from solver interface");
    }
    // *********************** CHECK THE NORM OF THE RESIDUAL **************************
    if (NLiter == 0) {
      {
        Teuchos::TimeMonitor localtimer(*normLAtimer);
        res->normInf(NLerr_first);
      }
      if (NLerr_first[0] > 1.0e-14)
        NLerr_scaled[0] = 1.0;
      else
        NLerr_scaled[0] = 0.0;
    }
    else {
      {
        Teuchos::TimeMonitor localtimer(*normLAtimer);
        res->normInf(NLerr);
      }
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
      {
        Teuchos::TimeMonitor localtimer(*exportLAtimer);
        du_over->doImport(*du, *importer, Tpetra::ADD);
      }
        
      if (useadjoint) {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        phi->update(1.0, *du_over, 1.0);
      }
      else {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        u->update(1.0, *du_over, 1.0);
      }
    }
    NLiter++; // increment number of iterations
  } // while loop
  if (milo_debug_level>1) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> normu(1);
    u->norm2(normu);
    if (Comm->getRank() == 0) {
      cout << "Norm of solution: " << normu[0] << "    (overlapped vector so results may differ on multiple procs)" << endl;
    }
  }
  
  if (milo_debug_level>2) {
    //KokkosTools::print(u);
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

template<class Node>
DFAD solver<Node>::computeObjective(const vector_RCP & F_soln, const ScalarT & time, const size_t & tindex) {
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting solver::computeObjective ..." << std::endl;
    }
  }
  
  DFAD totaldiff = 0.0;
  AD regDomain = 0.0;
  AD regBoundary = 0.0;
  int numDomainParams = params->domainRegIndices.size();
  int numBoundaryParams = params->boundaryRegIndices.size();
  
  params->sacadoizeParams(true);
  
  int numParams = params->num_active_params + params->globalParamUnknowns;
  vector<ScalarT> regGradient(numParams);
  vector<ScalarT> dmGradient(numParams);
  
  if (response_type == "discrete") {
    vector_RCP D_soln;
    bool fnd = datagen_soln->extract(D_soln, 0, time);
    if (fnd) {
      vector_RCP diff = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
      diff->update(1.0, *F_soln, 0.0);
      diff->update(-1.0, *D_soln, 1.0);
      Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> obj(1);
      diff->norm2(obj);
      totaldiff = 0.5*discrete_objective_scale_factor*obj[0]*obj[0];
      //cout << "objfun = " << totaldiff << endl;
    }
    else {
      std::cout << "Error: did not find a data-generating solution" << std::endl;
    }
  }
  else {
    for (size_t b=0; b<assembler->cells.size(); b++) {
      
      assembler->performGather(F_soln, 0, 0);
      assembler->performGather(params->Psol[0], 4, 0);
      
      for (size_t e=0; e<assembler->cells[b].size(); e++) {
        
        Kokkos::View<AD**,AssemblyDevice> obj_dev = assembler->cells[b][e]->computeObjective(time, tindex, 0);
        Kokkos::View<ScalarT***,AssemblyDevice> obj_sc_dev("obj func as scalar on device",obj_dev.extent(0),
                                                           obj_dev.extent(1),numParams+1);
        
        parallel_for("cell objective",RangePolicy<AssemblyExec>(0,obj_dev.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type i=0; i<obj_dev.extent(1); i++) {
            obj_sc_dev(elem,i,0) = obj_dev(elem,i).val();
            for (size_type j=1; j<obj_sc_dev.extent(2); j++) {
              obj_sc_dev(elem,i,j) = obj_dev(elem,i).fastAccessDx(j-1);
            }
          }
        });
        auto obj = Kokkos::create_mirror_view(obj_sc_dev);
        Kokkos::deep_copy(obj,obj_sc_dev);
        
        size_t numElem = assembler->cells[b][e]->numElem;
        
        if (obj.extent(1) > 0) { // may be zero if using sensors
          for (size_t c=0; c<numElem; c++) {
            vector<GO> paramGIDs;
            if (params->globalParamUnknowns > 0) {
              params->paramDOF->getElementGIDs(assembler->cells[b][e]->localElemID[c],
                                               paramGIDs, blocknames[b]);
            }
            for (size_type i=0; i<obj.extent(1); i++) {
              totaldiff.val() += obj(c,i,0);
              for (int p=0; p<params->num_active_params; p++) {
                ScalarT val;
                val = obj(c,i,p+1);
                dmGradient[p] += val;
              }
              if (params->globalParamUnknowns > 0) {
                
                for (size_t row=0; row<params->paramoffsets[0].size(); row++) {
                  GO rowIndex = paramGIDs[params->paramoffsets[0][row]];
                  
                  int poffset = 1+params->paramoffsets[0][row];
                  ScalarT val;
                  val = obj(c,i,poffset+params->num_active_params);
                  dmGradient[rowIndex+params->num_active_params] += val;
                }
              }
            }
          }
        }
        
        if (numDomainParams > 0){
          int paramIndex, rowIndex, poffset;
          ScalarT val;
          regDomain = assembler->cells[b][e]->computeDomainRegularization(params->domainRegConstants,
                                                                          params->domainRegTypes,
                                                                          params->domainRegIndices);
          
          for (size_t c=0; c<numElem; c++) {
            vector<GO> paramGIDs;
            params->paramDOF->getElementGIDs(assembler->cells[b][e]->localElemID[c],
                                             paramGIDs, blocknames[b]);
            
            for (int p = 0; p < numDomainParams; p++) {
              paramIndex = params->domainRegIndices[p];
              for( size_t row=0; row<params->paramoffsets[paramIndex].size(); row++ ) {
                if (regDomain.size() > 0) {
                  rowIndex = paramGIDs[params->paramoffsets[paramIndex][row]];
                  poffset = params->paramoffsets[paramIndex][row];
                  val = regDomain.fastAccessDx(poffset);
                  regGradient[rowIndex+params->num_active_params] += val;
                }
              }
            }
            
          }
        }
      }
      if (numBoundaryParams > 0) {
        for (size_t e=0; e<assembler->boundaryCells[b].size(); e++) {
          
          int paramIndex, rowIndex, poffset;
          ScalarT val;
          
          regBoundary = assembler->boundaryCells[b][e]->computeBoundaryRegularization(params->boundaryRegConstants,
                                                                                      params->boundaryRegTypes,
                                                                                      params->boundaryRegIndices,
                                                                                      params->boundaryRegSides);
          
          for (size_t c=0; c<assembler->boundaryCells[b][e]->numElem; c++) {
            vector<GO> paramGIDs;
            params->paramDOF->getElementGIDs(assembler->boundaryCells[b][e]->localElemID[c],
                                             paramGIDs, blocknames[b]);
            
            for (int p = 0; p < numBoundaryParams; p++) {
              paramIndex = params->boundaryRegIndices[p];
              for( size_t row=0; row<params->paramoffsets[paramIndex].size(); row++ ) {
                if (regBoundary.size() > 0) {
                  
                  //rowIndex = paramGIDs(c,params->paramoffsets[paramIndex][row]);
                  rowIndex = paramGIDs[params->paramoffsets[paramIndex][row]];
                  poffset = params->paramoffsets[paramIndex][row];
                  val = regBoundary.fastAccessDx(poffset);
                  regGradient[rowIndex+params->num_active_params] += val;
                  
                }
              }
            }
          }
        }
      }
      
      totaldiff += (regDomain + regBoundary);
    }
  }
  
  //to gather contributions across processors
  ScalarT meep = 0.0;
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&totaldiff.val(),&meep);
  totaldiff.val() = meep;
  
  DFAD fullobj(numParams,meep);
  
  for (int j=0; j<numParams; j++) {
    ScalarT dval;
    ScalarT ldval = dmGradient[j] + regGradient[j];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&ldval,&dval);
    fullobj.fastAccessDx(j) = dval;
  }
  
  params->sacadoizeParams(false);
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Finished solver::computeObjective ..." << std::endl;
    }
  }
  
  return fullobj;
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void solver<Node>::computeSensitivities(vector_RCP & u,
                          vector_RCP & a2, vector<ScalarT> & gradient) {
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting solver::computeSensitivities ..." << std::endl;
    }
  }
  
  DFAD obj_sens = 0.0;
  if (response_type != "discrete") {
    obj_sens = this->computeObjective(u, current_time, 0);
  }
  
  auto u_kv = u->template getLocalView<LA_device>();
  auto a2_kv = a2->template getLocalView<LA_device>();
  
  if (params->num_active_params > 0) {
  
    params->sacadoizeParams(true);
    
    vector<ScalarT> localsens(params->num_active_params);
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(LA_owned_map,params->num_active_params));
    matrix_RCP J = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(LA_owned_map, maxEntries));
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,params->num_active_params));
    matrix_RCP J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(LA_overlapped_graph));
    
    auto res_kv = res->template getLocalView<LA_device>();
    
    res_over->putScalar(0.0);
    
    bool curradjstatus = useadjoint;
    useadjoint = false;
    
    assembler->assembleJacRes(u, u, false, true, false,
                              res_over, J_over, isTransient, current_time, useadjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time, deltat);
    useadjoint = curradjstatus;
    
    res->putScalar(0.0);
    res->doExport(*res_over, *exporter, Tpetra::ADD);
    
    for (int paramiter=0; paramiter < params->num_active_params; paramiter++) {
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
    int numderivs = (int)obj_sens.size();
    for (int paramiter=0; paramiter < params->num_active_params; paramiter++) {
      localval = localsens[paramiter];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      ScalarT cobj = 0.0;
      
      if (paramiter<numderivs) {
        cobj = obj_sens.fastAccessDx(paramiter);
      }
      globalval += cobj;
      if ((int)gradient.size()<=paramiter) {
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
    auto ao_kv = a_owned->template getLocalView<LA_device>();
    
    for( size_t i=0; i<LA_owned.size(); i++ ) {
      ao_kv(i,0) = a2_kv(i,0);
    }
    
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    matrix_RCP J = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(params->param_owned_map,
                                                                              maxEntries));
    matrix_RCP J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(params->param_overlapped_graph));
    res_over->putScalar(0.0);
    J->setAllToScalar(0.0);
    
    J_over->setAllToScalar(0.0);
    
    bool curradjstatus = useadjoint;
    useadjoint = false;
    
    assembler->assembleJacRes(u, u, true, false, true,
                              res_over, J_over, isTransient, current_time, useadjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time, deltat);
    useadjoint = curradjstatus;
    
    J_over->fillComplete(LA_owned_map, params->param_owned_map);
    
    vector_RCP sens_over = Teuchos::rcp(new LA_MultiVector(params->param_overlapped_map,1));
    vector_RCP sens = Teuchos::rcp(new LA_MultiVector(params->param_owned_map,1));
    auto sens_kv = sens->template getLocalView<LA_device>();
    
    J->setAllToScalar(0.0);
    J->doExport(*J_over, *(params->param_exporter), Tpetra::ADD);
    J->fillComplete(LA_owned_map, params->param_owned_map);
    
    J->apply(*a_owned,*sens);
    
    vector<ScalarT> discLocalGradient(numDiscParams);
    vector<ScalarT> discGradient(numDiscParams);
    for (size_t i = 0; i < params->paramOwned.size(); i++) {
      GO gid = params->paramOwned[i];
      discLocalGradient[gid] = sens_kv(i,0);
    }
    for (int i = 0; i < numDiscParams; i++) {
      ScalarT globalval = 0.0;
      ScalarT localval = discLocalGradient[i];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      ScalarT cobj = 0.0;
      if ((i+params->num_active_params)<(int)obj_sens.size()) {
        cobj = obj_sens.fastAccessDx(i+params->num_active_params);
      }
      globalval += cobj;
      if ((int)gradient.size()<=params->num_active_params+i) {
        gradient.push_back(globalval);
      }
      else {
        gradient[params->num_active_params+i] += globalval;
      }
    }
  }
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Finished solver::computeSensitivities ..." << std::endl;
    }
  }
  
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void solver<Node>::setDirichlet(vector_RCP & u) {
  
  Teuchos::TimeMonitor localtimer(*dbcsettimer);
  
  typedef typename Node::execution_space LA_exec;
  
  if (usestrongDBCs) {
    auto u_kv = u->template getLocalView<LA_device>();
    //auto meas_kv = meas->getLocalView<HostDevice>();
    
    if (!scalarDirichletData && transientDirichletData) {
      this->projectDirichlet();
    }
    
    vector<vector<Kokkos::View<LO*,LA_device> > > dbcDOFs = assembler->fixedDOF;
    if (scalarDirichletData) {
      
      for (size_t b=0; b<dbcDOFs.size(); b++) {
        for (size_t v=0; v<dbcDOFs[b].size(); v++) {
          if (dbcDOFs[b][v].extent(0)>0) {
            ScalarT value = scalarDirichletValues[b][v];
            Kokkos::View<ScalarT[1],LA_device> scalarval("scalar value");
            auto scval_host = Kokkos::create_mirror_view(scalarval);
            scval_host(0) = value;
            Kokkos::deep_copy(scalarval,scval_host);
            
            auto cdofs = dbcDOFs[b][v];
            parallel_for("solver initial scalar",RangePolicy<LA_exec>(0,cdofs.extent(0)), KOKKOS_LAMBDA (const int i ) {
              ScalarT val = scalarval(0);
              u_kv(cdofs(i),0) = val;
            });
          }
        }
      }
      /*
      for (size_t b=0; b<dbcDOFs.size(); b++) {
        for (size_t v=0; v<dbcDOFs[b].size(); v++) {
          ScalarT value = scalarDirichletValues[b][v];
          Kokkos::View<ScalarT[1],LA_device> scalarval("scalar value");
          auto scval_host = Kokkos::create_mirror_view(scalarval);
          scval_host(0) = value;
          Kokkos::deep_copy(scalarval,scval_host);
          auto dbcs = dbcDOFs[b][v];
          bool copy_views = false;
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA) && !defined(MrHyDE_SOLVERSPACE_CUDA) // Assembly=Cuda, Solver=CPU
          auto dbcs_dev = Kokkos::create_mirror(LA_device, dbcs);
          copy_views = true;
#elif !defined(MrHyDE_ASSEMBLYSPACE_CUDA) && defined(MrHyDE_SOLVERSPACE_CUDA) // Assembly=CPU, Solver=GPU
          auto dbcs_dev = Kokkos::create_mirror_view(dbcs);
          copy_views = true;
#else
          auto dbcs_dev = dbcs;
#endif
          if (copy_views) {
            Kokkos::deep_copy(dbcs_dev,dbcs);
          }
          parallel_for("solver initial scalar",RangePolicy<LA_exec>(0,dbcs.extent(0)), KOKKOS_LAMBDA (const int i ) {
          //for (size_t i=0; i<dbcDOFs[b][v].size(); i++) {
            LO ind = dbcs_dev(i);//dbcDOFs[b][v][i];
            ScalarT val = scalarval(0);
            u_kv(ind,0) = val;
          });
        }
      }*/
    }
    else {
      auto dbc_kv = fixedDOF_soln->template getLocalView<LA_device>();
      for (size_t b=0; b<dbcDOFs.size(); b++) {
        for (size_t v=0; v<dbcDOFs[b].size(); v++) {
          if (dbcDOFs[b][v].extent(0)>0) {
            auto cdofs = dbcDOFs[b][v];
            parallel_for("solver initial scalar",RangePolicy<LA_exec>(0,cdofs.extent(0)), KOKKOS_LAMBDA (const int i ) {
              u_kv(cdofs(i),0) = dbc_kv(cdofs(i),0);
            });
          }
        }
      }
      /*
      for (size_t b=0; b<dbcDOFs.size(); b++) {
        for (size_t v=0; v<dbcDOFs[b].size(); v++) {
          auto dbcs = dbcDOFs[b][v];
          parallel_for("solver initial scalar",RangePolicy<LA_exec>(0,dbcs.extent(0)), KOKKOS_LAMBDA (const int i ) {
            //for (size_t i=0; i<dbcDOFs[b][v].size(); i++) {
            LO ind = dbcs(i);//dbcDOFs[b][v][i];
            u_kv(ind,0) = dbc_kv(ind,0);
          });
        }
      }*/
    }
    
    // set point dbcs
    vector<vector<GO> > pointDOFs = phys->point_dofs;
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<GO> pt_dofs = pointDOFs[b];
      Kokkos::View<LO*,LA_device> ptdofs("pointwise dofs", pointDOFs[b].size());
      auto ptdofs_host = Kokkos::create_mirror_view(ptdofs);
      for (size_t i = 0; i < pt_dofs.size(); i++) {
        LO row = LA_overlapped_map->getLocalElement(pt_dofs[i]);
        ptdofs_host(i) = row;
      }
      Kokkos::deep_copy(ptdofs,ptdofs_host);
      parallel_for("solver initial scalar",RangePolicy<LA_exec>(0,ptdofs.extent(0)), KOKKOS_LAMBDA (const int i ) {
        LO row = ptdofs(i);
        u_kv(row,0) = 0.0; // fix to zero for now
      });
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > solver<Node>::setInitialParams() {
  vector_RCP initial = Teuchos::rcp(new LA_MultiVector(params->param_overlapped_map,1));
  ScalarT value = 2.0;
  initial->putScalar(value);
  return initial;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > solver<Node>::setInitial() {
 
  Teuchos::TimeMonitor localtimer(*initsettimer);
  typedef typename Node::execution_space LA_exec;
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting solver::setInitial ..." << endl;
    }
  }
  
  vector_RCP initial = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
  
  bool samedevice = true;
  bool usehost = false;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyMem>::accessible) {
    samedevice = false;
    if (!Kokkos::SpaceAccessibility<LA_exec, HostMem>::accessible) {
      usehost = true;
    }
    else {
      // output an error along the lines of "what the hell happened"
    }
  }
  
  if (scalarInitialData) {
    
    
    auto initial_kv = initial->template getLocalView<LA_device>();
    
    for (size_t block=0; block<assembler->cells.size(); block++) {
      
      Kokkos::View<ScalarT*,LA_device> idata("scalar initial data",scalarInitialValues[block].size());
      auto idata_host = Kokkos::create_mirror_view(idata);
      for (size_t i=0; i<scalarInitialValues[block].size(); i++) {
        idata_host(i) = scalarInitialValues[block][i];
      }
      Kokkos::deep_copy(idata,idata_host);
      
      
      if (samedevice) {
        auto offsets = assembler->wkset[block]->offsets;
        auto numDOF = assembler->cellData[block]->numDOF;
        for (size_t cell=0; cell<assembler->cells[block].size(); cell++) {
          auto LIDs = assembler->cells[block][cell]->LIDs;
          parallel_for("solver initial scalar",RangePolicy<LA_exec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for (int i=0; i<numDOF(n); i++ ) {
                initial_kv(LIDs(e,offsets(n,i)),0) = idata(n);
              }
            }
          });
        }
      }
      else if (usehost) {
        auto offsets = assembler->wkset[block]->offsets;
        auto host_offsets = Kokkos::create_mirror_view(offsets);
        Kokkos::deep_copy(host_offsets,offsets);
        auto numDOF = assembler->cellData[block]->numDOF_host;
        for (size_t cell=0; cell<assembler->cells[block].size(); cell++) {
          auto LIDs = assembler->cells[block][cell]->LIDs_host;
          parallel_for("solver initial scalar",RangePolicy<LA_exec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for (int i=0; i<numDOF(n); i++ ) {
                initial_kv(LIDs(e,host_offsets(n,i)),0) = idata(n);
              }
            }
          });
        }
      }
      
    }
    
  }
  else {
  
    initial->putScalar(0.0);
    
    vector_RCP glinitial = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    
    if (initial_type == "L2-projection") {
      
      // Compute the L2 projection of the initial data into the discrete space
      vector_RCP rhs = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
      matrix_RCP mass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(LA_overlapped_graph));
      vector_RCP glrhs = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
      matrix_RCP glmass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(LA_owned_map, maxEntries));
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

template<class Node>
void solver<Node>::linearSolver(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
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
          M_dd = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> > ("SCHWARZ", J);
          M_dd->setParameters(ifpackList);
          M_dd->initialize();
          M_dd->compute();
          have_preconditioner = true;
        }
        Problem->setLeftPrec(M_dd);
      }
      else {
        if (!have_preconditioner) {
          M = this->buildPreconditioner(J);
          have_preconditioner = true;
        }
        else {
          MueLu::ReuseTpetraPreconditioner(J,*M);
        }
        Problem->setLeftPrec(M);
      }
      
      //Problem->setRightPrec(M);
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

template<class Node>
Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > solver<Node>::buildPreconditioner(const matrix_RCP & J) {
  Teuchos::ParameterList mueluParams;
  
  // if the user provides a "Preconditioner Settings" sublist, use it for MueLu
  // otherwise, set things with the simple approach
  if(settings->sublist("Solver").isSublist("Preconditioner Settings"))
  {
    mueluParams = settings->sublist("Solver").sublist("Preconditioner Settings");
  }
  else
  {
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
      //mueluParams.sublist("smoother: params").set("relaxation: type","Jacobi");
      //mueluParams.sublist("smoother: params").set("relaxation: type","Symmetric Gauss-Seidel");
      mueluParams.sublist("smoother: params").set("relaxation: type","MT Gauss-Seidel");
      //mueluParams.sublist("smoother: params").set("relaxation: sweeps",2);
    }
  
    // Repartitioning
  
    mueluParams.set("repartition: enable",false);
    mueluParams.set("repartition: partitioner","zoltan");
    mueluParams.set("repartition: start level",2);
    mueluParams.set("repartition: min rows per proc",800);
    mueluParams.set("repartition: max imbalance", 1.1);
    mueluParams.set("repartition: remap parts",false);
  
    mueluParams.set("reuse: type",preconditioner_reuse_type);
  }
  mueluParams.setName("MueLu");
  
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > Mnew = MueLu::CreateTpetraPreconditioner((Teuchos::RCP<LA_Operator>)J, mueluParams);

  //have_preconditioner = true;
  return Mnew;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void solver<Node>::setBatchID(const int & bID){
  batchID = bID;
  params->batchID = bID;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > solver<Node>::blankState(){
  vector_RCP F_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
  return F_soln;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void solver<Node>::finalizeParams() {

  //for (size_t b=0; b<blocknames.size(); b++) {
  //  assembler->wkset[b]->paramusebasis = params->discretized_param_usebasis;
  //  assembler->wkset[b]->paramoffsets = params->paramoffsets[0];
 // }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void solver<Node>::finalizeMultiscale() {
  if (multiscale_manager->subgridModels.size() > 0 ) {
    for (size_t k=0; k<multiscale_manager->subgridModels.size(); k++) {
      multiscale_manager->subgridModels[k]->paramvals_KVAD = params->paramvals_KVAD;
    //  multiscale_manager->subgridModels[k]->wkset[0]->paramnames = paramnames;
    }
    
    multiscale_manager->macro_wkset = assembler->wkset;
    
    multiscale_manager->setMacroInfo(disc->basis_pointers, disc->basis_types,
                                     phys->varlist, useBasis, phys->offsets,
                                     assembler->cells[0][0]->cellData->numDOF,
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

