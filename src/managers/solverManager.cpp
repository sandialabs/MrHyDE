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

#include "solverManager.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class Node>
SolverManager<Node>::SolverManager(const Teuchos::RCP<MpiComm> & Comm_,
                                   Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                   Teuchos::RCP<MeshInterface> & mesh_,
                                   Teuchos::RCP<DiscretizationInterface> & disc_,
                                   Teuchos::RCP<PhysicsInterface> & phys_,
                                   Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                                   Teuchos::RCP<ParameterManager<Node> > & params_) :
Comm(Comm_), settings(settings_), mesh(mesh_), disc(disc_), phys(phys_), assembler(assembler_), params(params_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager constructor ..." << endl;
    }
  }
  
  numEvaluations = 0;
  
  // Get the required information from the settings
  spaceDim = mesh->stk_mesh->getDimension();
  isInitial = false;
  initial_time = settings->sublist("Solver").get<double>("initial time",0.0);
  current_time = initial_time;
  final_time = settings->sublist("Solver").get<double>("final time",1.0);
  if (settings->sublist("Solver").isParameter("delta t")) {
    deltat = settings->sublist("Solver").get<double>("delta t",1.0);
  }
  else {
    int numTimesteps = settings->sublist("Solver").get<int>("number of steps",1);
    deltat = (final_time - initial_time)/numTimesteps;
  }
  verbosity = settings->get<int>("verbosity",0);
  usestrongDBCs = settings->sublist("Solver").get<bool>("use strong DBCs",true);
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("use measurements as DBCs", false);
  
  solver_type = settings->sublist("Solver").get<string>("solver","none"); // or "transient"
  
  string solve_order_string = settings->sublist("Solver").get<string>("solve order","state"); // comma separated list
  // Script to break delimited list into pieces
  {
    string delimiter = ", ";
    size_t pos = 0;
    if (solve_order_string.find(delimiter) == string::npos) {
      solve_order.push_back(solve_order_string);
    }
    else {
      string token;
      while ((pos = solve_order_string.find(delimiter)) != string::npos) {
        token = solve_order_string.substr(0, pos);
        solve_order.push_back(token);
        solve_order_string.erase(0, pos + delimiter.length());
      }
      solve_order.push_back(solve_order_string);
    }
  }
  
  use_custom_PCG = settings->sublist("Solver").get<bool>("use custom PCG",false);
  
  time_order = settings->sublist("Solver").get<int>("time order",1);
  NLtol = settings->sublist("Solver").get<double>("nonlinear TOL",1.0E-6);
  NLabstol = settings->sublist("Solver").get<double>("absolute nonlinear TOL",std::min((double)NLtol,(double)1.0E-6));
  maxNLiter = settings->sublist("Solver").get<int>("max nonlinear iters",10);
  useRelativeTOL = settings->sublist("Solver").get<bool>("use relative TOL",true);
  useAbsoluteTOL = settings->sublist("Solver").get<bool>("use absolute TOL",false);
  allowBacktracking = settings->sublist("Solver").get<bool>("allow backtracking",true);
  
  ButcherTab = settings->sublist("Solver").get<string>("transient Butcher tableau","BWE");
  BDForder = settings->sublist("Solver").get<int>("transient BDF order",1);
  if (BDForder>1) {
    if (ButcherTab == "custom") {
      cout << "Warning: running a higher order BDF method with anything other than BWE/DIRK-1,1 is risky." << endl;
      cout << "The code will run, but the results may be nonsense" << endl;
    }
    else {
      if (ButcherTab != "BWE" && ButcherTab != "DIRK-1,1") {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: need to use BWE or DIRK-1,1 with higher order BDF");
      }
    }
  }
  
  // Additional parameters for higher-order BDF methods that require some startup procedure
  startupButcherTab = settings->sublist("Solver").get<string>("transient startup Butcher tableau",ButcherTab);
  startupBDForder = settings->sublist("Solver").get<int>("transient startup BDF order",BDForder);
  startupSteps = settings->sublist("Solver").get<int>("transient startup steps",BDForder);
  if (startupBDForder>1) {
    if (startupButcherTab == "custom") {
      cout << "Warning: running a higher order BDF method with anything other than BWE/DIRK-1,1 is risky." << endl;
      cout << "The code will run, but the results may be nonsense" << endl;
    }
    else {
      if (startupButcherTab != "BWE" && startupButcherTab != "DIRK-1,1") {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: need to use BWE or DIRK-1,1 with higher order BDF");
      }
    }
  }
  
  line_search = false;//settings->sublist("Solver").get<bool>("Use Line Search","false");
  store_adjPrev = false;
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  if (!isTransient) {
    deltat = 1.0;
  }
  fully_explicit = settings->sublist("Solver").get<bool>("fully explicit",false);
  
  if (fully_explicit && Comm->getRank() == 0) {
    cout << "WARNING: the fully explicit method is requested.  This is an experimental capability and may not work with all time integration methods" << endl;
  }
  
  initial_type = settings->sublist("Solver").get<string>("initial type","L2-projection");
  
  // needed information from the mesh
  mesh->stk_mesh->getElementBlockNames(blocknames);
  
  // needed information from the physics interface
  numVars = phys->numVars; //
  vector<vector<string> > phys_varlist = phys->varlist;
  
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
    
    varlist.push_back(currvarlist);
    useBasis.push_back(curruseBasis);
    numBasis.push_back(currnumBasis);
    maxBasis.push_back(currmaxBasis);
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Create linear algebra interface
  /////////////////////////////////////////////////////////////////////////////
  
  linalg = Teuchos::rcp( new LinearAlgebraInterface<Node>(Comm, settings, disc, params) );
  
  res = linalg->getNewVector();
  res_over = linalg->getNewOverlappedVector();
  du_over = linalg->getNewOverlappedVector();
  du = linalg->getNewVector();
  
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
    
  /////////////////////////////////////////////////////////////////////////////
  
  this->setBatchID(Comm->getRank());
  
  /////////////////////////////////////////////////////////////////////////////
  
  this->setupFixedDOFs(settings);
  
  /////////////////////////////////////////////////////////////////////////////
  
  have_initial_conditions = false;
  if (settings->sublist("Physics").isSublist("Initial conditions")) {
    have_initial_conditions = true;
  }
  else {
    for (size_t b=0; b<blocknames.size(); b++) {
      if (settings->sublist("Physics").isSublist(blocknames[b])) {
        if (settings->sublist("Physics").sublist(blocknames[b]).isSublist("Initial conditions")) {
          have_initial_conditions  = true;
        }
      }
    }
  }
  
  scalarInitialData = settings->sublist("Physics").sublist("Initial conditions").get<bool>("scalar data", false);
  have_static_Dirichlet_data = false;
  
  if (have_initial_conditions && scalarInitialData) {
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
  
  //---------------------------------------------------
  // Mass matrix (lumped and maybe full) for explicit
  //---------------------------------------------------
  
  if (fully_explicit) {
    matrix_RCP mass;
    
    if (!assembler->lump_mass) {
      
      typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
      typedef Tpetra::CrsGraph<LO,GO,Node>            LA_CrsGraph;
      
      vector<size_t> maxEntriesPerRow(linalg->overlapped_map->getNodeNumElements(), 0);
      for (size_t b=0; b<assembler->cells.size(); b++) {
        auto offsets = assembler->wkset[b]->offsets;
        auto numDOF = assembler->cellData[b]->numDOF;
        for (size_t e=0; e<assembler->cells[b].size(); e++) {
          auto LIDs = assembler->cells[b][e]->LIDs_host;
          
          for (size_type elem=0; elem<LIDs.extent(0); ++elem) {
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                int row = offsets(n,j);
                LO rowIndex = LIDs(elem,row);
                maxEntriesPerRow[rowIndex] += static_cast<size_t>(numDOF(n));
              }
            }
          }
        }
      }
      
      size_t maxEntries = 0;
      for (size_t m=0; m<maxEntriesPerRow.size(); ++m) {
        maxEntries = std::max(maxEntries, maxEntriesPerRow[m]);
      }
      
      maxEntries = static_cast<size_t>(settings->sublist("Solver").get<int>("max entries per row",
                                                                            static_cast<int>(maxEntries)));
      
      Teuchos::RCP<LA_CrsGraph> overlapped_graph = Teuchos::rcp(new LA_CrsGraph(linalg->overlapped_map,
                                                                                maxEntriesPerRow,
                                                                                Tpetra::StaticProfile));
    
      for (size_t b=0; b<assembler->cells.size(); b++) {
        auto offsets = assembler->wkset[b]->offsets;
        auto numDOF = assembler->cellData[b]->numDOF;
        for (size_t e=0; e<assembler->cells[b].size(); e++) {
          auto LIDs = assembler->cells[b][e]->LIDs_host;
          
          parallel_for("assembly insert Jac",
                       RangePolicy<HostExec>(0,LIDs.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                vector<GO> cols;
                int row = offsets(n,j);
                GO rowIndex = linalg->overlapped_map->getGlobalElement(LIDs(elem,row));
                for (int k=0; k<numDOF(n); k++) {
                  int col = offsets(n,k);
                  GO gcol = linalg->overlapped_map->getGlobalElement(LIDs(elem,col));
                  cols.push_back(gcol);
                }
                overlapped_graph->insertGlobalIndices(rowIndex,cols);
              }
            }
          });
        }
      }
      overlapped_graph->fillComplete();
      
      vector<GO> owned;
      disc->DOF->getOwnedIndices(owned);
      vector<size_t> maxOwnedEntriesPerRow(linalg->owned_map->getNodeNumElements(), 0);
      for (size_t i=0; i<owned.size(); ++i) {
        LO ind1 = linalg->overlapped_map->getLocalElement(owned[i]);
        LO ind2 = linalg->owned_map->getLocalElement(owned[i]);
        maxOwnedEntriesPerRow[ind2] = maxEntriesPerRow[ind1];
      }
      
      explicitMass = Teuchos::rcp(new LA_CrsMatrix(linalg->owned_map,
                                                   maxOwnedEntriesPerRow,
                                                   Tpetra::StaticProfile));
      
      mass = Teuchos::rcp(new LA_CrsMatrix(overlapped_graph));
    }
    
    diagMass = linalg->getNewVector();
    vector_RCP diagMass_over = linalg->getNewOverlappedVector();
    assembler->getWeightedMass(mass,diagMass_over);
    linalg->exportVectorFromOverlapped(diagMass, diagMass_over);
    if (!assembler->lump_mass) {
      linalg->exportMatrixFromOverlapped(explicitMass, mass);
      mass.reset();
      linalg->fillComplete(explicitMass);
    }
    
    //KokkosTools::print(diagMass);
    //linalg->resetJacobian(); // doesn't actually erase the mass matrix ... just sets a recompute flag
    
    linalg->q_pcg = linalg->getNewVector();
    linalg->z_pcg = linalg->getNewVector();
    linalg->p_pcg = linalg->getNewVector();
    linalg->r_pcg = linalg->getNewVector();
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager constructor" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setButcherTableau(const string & tableau) {
  
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
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",3);
    butcher_b(0) = 1.0/6.0;
    butcher_b(1) = 1.0/6.0;
    butcher_b(2) = 2.0/3.0;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",3);
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
  else if (tableau == "leap-frog") { // Leap-frog for Maxwells
    butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
    butcher_A(1,0) = 1.0;
    butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
    butcher_b(0) = 1.0;
    butcher_b(1) = 1.0;
    butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
    butcher_c(0) = 0.0;
    butcher_c(1) = 0.0;
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
  Kokkos::View<ScalarT**,AssemblyDevice> dev_butcher_A("butcher_A on device",butcher_A.extent(0),butcher_A.extent(1));
  Kokkos::View<ScalarT*,AssemblyDevice> dev_butcher_b("butcher_b on device",butcher_b.extent(0));
  Kokkos::View<ScalarT*,AssemblyDevice> dev_butcher_c("butcher_c on device",butcher_c.extent(0));
  
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
void SolverManager<Node>::setBackwardDifference(const int & order) { // using order as an input to allow for dynamic changes
  
  Kokkos::View<ScalarT*,AssemblyDevice> dev_BDF_wts;
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
  
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeWorkset() {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::finalizeWorkset ..." << endl;
    }
  }
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    
    if (assembler->cells[b].size() > 0) {
      vector<vector<int> > voffsets = disc->offsets[b];
      size_t maxoff = 0;
      for (size_t i=0; i<voffsets.size(); i++) {
        if (voffsets[i].size() > maxoff) {
          maxoff = voffsets[i].size();
        }
      }
      
      Kokkos::View<int**,AssemblyDevice> offsets_view("offsets on assembly device",voffsets.size(),maxoff);
      auto host_offsets = Kokkos::create_mirror_view(offsets_view);
      for (size_t i=0; i<voffsets.size(); i++) {
        for (size_t j=0; j<voffsets[i].size(); j++) {
          host_offsets(i,j) = voffsets[i][j];
        }
      }
      Kokkos::deep_copy(offsets_view,host_offsets);
      assembler->wkset[b]->offsets = offsets_view;
      
      size_t maxpoff = 0;
      for (size_t i=0; i<params->paramoffsets.size(); i++) {
        if (params->paramoffsets[i].size() > maxpoff) {
          maxpoff = params->paramoffsets[i].size();
        }
      }
      
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
      assembler->wkset[b]->paramoffsets = poffsets_view;
      assembler->wkset[b]->varlist = varlist[b];
      assembler->wkset[b]->aux_varlist = phys->aux_varlist[b];
      assembler->wkset[b]->param_varlist = params->discretized_param_names;
      
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
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager::finalizeWorkset" << endl;
    }
  }
  
}

// ========================================================================================
// Set up the logicals and data structures for the fixed DOF (Dirichlet and point constraints)
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupFixedDOFs(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  Teuchos::TimeMonitor localtimer(*fixeddofsetuptimer);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::setupFixedDOFs()" << endl;
    }
  }
  
  if (!disc->haveDirichlet) {
    usestrongDBCs = false;
  }
  
  if (usestrongDBCs) {
    fixedDOF_soln = linalg->getNewOverlappedVector();
    
    scalarDirichletData = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("scalar data", false);
    staticDirichletData = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("static data", true);
    
    if (scalarDirichletData && !staticDirichletData) {
      if (Comm->getRank() == 0) {
        cout << "Warning: The Dirichlet data was set to scalar and non-static.  This should not happen." << endl;
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
              value = dbc_settings.sublist(varlist[b][var]).template get<ScalarT>("all boundaries");
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
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager::setupFixedDOFs()" << endl;
    }
  }
  
}

// ========================================================================================
// Set up the logicals and data structures for the fixed DOF (Dirichlet and point constraints)
// ========================================================================================

template<class Node>
void SolverManager<Node>::projectDirichlet() {
  
  Teuchos::TimeMonitor localtimer(*dbcprojtimer);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::projectDirichlet()" << endl;
    }
  }
  if (usestrongDBCs) {
    fixedDOF_soln = linalg->getNewOverlappedVector();
    vector_RCP glfixedDOF_soln = linalg->getNewVector();
    
    vector_RCP rhs = linalg->getNewOverlappedVector();
    matrix_RCP mass = linalg->getNewOverlappedMatrix();
    vector_RCP glrhs = linalg->getNewVector();
    matrix_RCP glmass = linalg->getNewMatrix();
    
    assembler->setDirichlet(rhs, mass, is_adjoint, current_time);
    
    linalg->exportMatrixFromOverlapped(glmass, mass);
    linalg->exportVectorFromOverlapped(glrhs, rhs);
    linalg->fillComplete(glmass);
    
    if (debug_level>2) {
      //KokkosTools::print(glmass,"L2-projection matrix for DBCs");
      //KokkosTools::print(glrhs,"L2-projections RHS for DBCs");
    }
    
    // TODO BWR -- couldn't think of a good way to protect against
    // the preconditioner failing for HFACE, will need to be handled
    // explicitly in the input file for now (State boundary L2 linear solver)
    linalg->linearSolverBoundaryL2(glmass, glrhs, glfixedDOF_soln);
    linalg->importVectorToOverlapped(fixedDOF_soln, glfixedDOF_soln);
    
  }
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager::projectDirichlet()" << endl;
    }
  }
  
}

// ========================================================================================
/* given the parameters, solve the forward problem */
// ========================================================================================

template<class Node>
void SolverManager<Node>::forwardModel(DFAD & objective) {
  
  current_time = initial_time;
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::forwardModel ..." << endl;
    }
  }
  
  is_adjoint = false;
  params->sacadoizeParams(false);
  
  if (!scalarDirichletData) {
    if (!staticDirichletData) {
      this->projectDirichlet();
    }
    else if (!have_static_Dirichlet_data) {
      this->projectDirichlet();
      have_static_Dirichlet_data = true;
    }
  }

  vector_RCP u = this->setInitial();
    
  if (solver_type == "steady-state") {
    this->steadySolver(objective, u);
  }
  else if (solver_type == "transient") {
    vector<ScalarT> gradient; // not really used here
    this->transientSolver(u, objective, gradient, initial_time, final_time);
  }
  else {
    // print out an error message
  }
    
  if (postproc->write_optimization_solution) {
    postproc->writeOptimizationSolution(numEvaluations);
  }
  
  numEvaluations++;
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager::forwardModel" << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::steadySolver(DFAD & objective, vector_RCP & u) {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::steadySolver ..." << endl;
    }
  }
  
  vector_RCP zero_soln = linalg->getNewOverlappedVector();
  
  this->nonlinearSolver(u, zero_soln);
  postproc->record(u,current_time,true,objective);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::steadySolver" << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::adjointModel(vector<ScalarT> & gradient) {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::adjointModel ..." << endl;
    }
  }
  
  is_adjoint = true;
  
  params->sacadoizeParams(false);
  
  vector_RCP phi = setInitial();
  
  vector_RCP zero_soln = linalg->getNewOverlappedVector();
  
  if (solver_type == "steady-state") {
    vector_RCP u;
    bool fnd = postproc->soln->extract(u, current_time);
    if (!fnd) {
      //throw error
    }
    this->nonlinearSolver(u, phi);
    
    postproc->computeSensitivities(u, phi, current_time, deltat, gradient);
    
  }
  else if (solver_type == "transient") {
    DFAD obj = 0.0;
    this->transientSolver(phi, obj, gradient, initial_time, final_time);
  }
  else {
    // print out an error message
  }
  
  is_adjoint = false;
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager::adjointModel" << endl;
    }
  }
  
}


// ========================================================================================
/* solve the problem */
// ========================================================================================

template<class Node>
void SolverManager<Node>::transientSolver(vector_RCP & initial, DFAD & obj, vector<ScalarT> & gradient,
                                          ScalarT & start_time, ScalarT & end_time) {
  
  Teuchos::TimeMonitor localtimer(*transientsolvertimer);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting SolverManager::transientSolver ..." << endl;
      cout << "******** Start time = " << start_time << endl;
      cout << "******** End time = " << end_time << endl;
      cout << "******** Time step size = " << deltat << endl;
    }
  }
  
  vector_RCP zero_vec = linalg->getNewOverlappedVector();
  zero_vec->putScalar(0.0);
  
  current_time = start_time;
  if (!is_adjoint) { // forward solve - adaptive time stepping
    is_final_time = false;
    vector_RCP u = initial;
    
    if (usestrongDBCs) {
      this->setDirichlet(u);
    }
    
    {
      assembler->performGather(u,0,0);
      postproc->record(u,current_time,true,obj);
    }
    //Kokkos::fence();
    
    for (int s=0; s<numsteps; s++) {
      assembler->resetPrevSoln();
    }
    
    int stepProg = 0;
    obj = 0.0;
    int numCuts = 0;
    int maxCuts = 5; // TMW: make this a user-defined input
    double timetol = end_time*1.0e-6; // just need to get close enough to final time
    bool write_this_step = false;
    
    vector_RCP u_prev = linalg->getNewOverlappedVector();
    vector_RCP u_stage = linalg->getNewOverlappedVector();
    
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
      
      u_prev->assign(*u);
      auto BDF_wts = assembler->wkset[0]->BDF_wts;
      int status = 1;
      for (int stage = 0; stage<numstages; stage++) {
        // Need a stage solution
        // Set the initial guess for stage solution
        //u_stage->update(1.0,*u,0.0);
        u_stage->assign(*u_prev);
        
        assembler->updateStageNumber(stage); // could probably just += 1 in wksets
        
        if (fully_explicit) {
          status = this->explicitSolver(u_stage, zero_vec, stage);
        }
        else {
          status = this->nonlinearSolver(u_stage, zero_vec);
        }
        
        u->update(1.0, *u_stage, 1.0);
        u->update(-1.0, *u_prev, 1.0);
        assembler->updateStageSoln(); // moves the stage solution into u_stage (avoids mem transfer)
      }
      
      if (status == 0) { // NL solver converged
        current_time += deltat;
        stepProg += 1;
        
        // Make sure last step solution is gathered
        // Last set of values is from a stage solution, which is potentially different
        assembler->performGather(u,0,0);
        // TODO :: BWR make this more flexible (may want to save based on simulation time as well)
        if (stepProg % postproc->write_frequency == 0) write_this_step = true;
        postproc->record(u,current_time,write_this_step,obj);
        write_this_step = false;
      }
      else { // something went wrong, cut time step and try again
        deltat *= 0.5;
        numCuts += 1;
        
        bool fnd = postproc->soln->extract(u, current_time);
        if (!fnd) {
          // throw error
        }
        if (Comm->getRank() == 0 && verbosity > 0) {
          cout << endl << endl << "*******************************************************" << endl;
          cout << endl << "**** Cutting Time Step " << endl;
          cout << "**** Current time is " << current_time << endl << endl;
          cout << "*******************************************************" << endl << endl << endl;
        }
        
      }
    }
    // If the final step doesn't fall when a write is requested, catch that here  
    if (stepProg % postproc->write_frequency != 0 && postproc->write_solution) {
      postproc->writeSolution(current_time);
    }
  }
  else { // adjoint solve - fixed time stepping based on forward solve
    current_time = final_time;
    is_final_time = true;
    
    vector_RCP u = linalg->getNewOverlappedVector();
    vector_RCP u_prev = linalg->getNewOverlappedVector();
    vector_RCP phi = linalg->getNewOverlappedVector();
    
    size_t numFwdSteps = postproc->soln->times[0].size()-1;
    
    //bool fndu = soln->extract(u, numsteps);
    //assembler->performGather(0,u,0,0);
    
    for (size_t timeiter = 0; timeiter<numFwdSteps; timeiter++) {
      size_t cindex = numFwdSteps-timeiter;
      vector_RCP phi_prev = linalg->getNewOverlappedVector();
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
      bool fndu = postproc->soln->extract(u, cindex);
      if (!fndu) {
        // throw error
      }
      bool fndup = postproc->soln->extract(u_prev, cindex-1);
      if (!fndup) {
        // throw error
      }
      assembler->performGather(u_prev,0,0);
      assembler->resetPrevSoln();
      
      current_time = postproc->soln->times[0][cindex-1];
      
      // if multistage, recover forward solution at each stage
      if (numstages == 1) { // No need to re-solve in this case
        int status = this->nonlinearSolver(u, phi);
        if (status>0) {
          // throw error
        }
        postproc->computeSensitivities(u, phi, current_time, deltat, gradient);
      }
      else {
        is_adjoint = false;
        std::vector<vector_RCP> stage_solns;
        for (int stage = 0; stage<numstages; stage++) {
          // Need a stage solution
          vector_RCP u_stage = linalg->getNewOverlappedVector();
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
        is_adjoint = true;
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
          vector_RCP phi_stage = linalg->getNewOverlappedVector();
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
        postproc->computeSensitivities(u, phi, current_time, deltat, gradient);
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
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Finished SolverManager::transientSolver" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
int SolverManager<Node>::nonlinearSolver(vector_RCP & u, vector_RCP & phi) {
  
  Teuchos::TimeMonitor localtimer(*nonlinearsolvertimer);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting SolverManager::nonlinearSolver ..." << endl;
    }
  }
  
  int status = 0;
  int NLiter = 0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_first(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_scaled(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm(1);
  resnorm_first[0] = 10*NLtol;
  resnorm_scaled[0] = resnorm_first[0];
  resnorm[0] = resnorm_first[0];
  
  if (usestrongDBCs) {
    this->setDirichlet(u);
  }
  
  int maxiter = maxNLiter;
  if (is_adjoint) {
    maxiter = 2;
  }
  
  bool proceed = true;
  ScalarT alpha = 1.0;
  
  while (proceed) {
    
    multiscale_manager->reset();
    
    gNLiter = NLiter;
  
    bool build_jacobian = !linalg->getJacobianReuse();//true;
    matrix_RCP J = linalg->getNewMatrix();
    
    matrix_RCP J_over = linalg->getNewOverlappedMatrix();
    if (build_jacobian) {
      linalg->fillComplete(J_over);
    }
    
    // *********************** COMPUTE THE JACOBIAN AND THE RESIDUAL **************************
    
    res_over->putScalar(0.0);
    
    if (build_jacobian) {
      J_over->resumeFill();
      J_over->setAllToScalar(0.0);
    }
    
    store_adjPrev = false;
    if ( is_adjoint && (NLiter == 1)) {
      store_adjPrev = true;
    }
    
    assembler->assembleJacRes(u, phi, build_jacobian, false, false,
                              res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time, deltat);
    
    linalg->exportVectorFromOverlapped(res, res_over);
    
    if (is_adjoint) {
      ScalarT cdt = 0.0;
      if (solver_type == "transient") {
        cdt = deltat;
      }
      postproc->computeObjectiveGradState(u,current_time+cdt,deltat,res);
    }
    
    if (debug_level>2) {
      KokkosTools::print(res,"residual from solver interface");
    }
    // *********************** CHECK THE NORM OF THE RESIDUAL **************************
    
    {
      Teuchos::TimeMonitor localtimer(*normLAtimer);
      res->normInf(resnorm);
    }
    
    bool solve = true;
    if (NLiter == 0) {
      resnorm_first[0] = resnorm[0];
      resnorm_scaled[0] = 1.0;
    }
    else {
      resnorm_scaled[0] = resnorm[0]/resnorm_first[0];
    }
    
    if (Comm->getRank() == 0 && verbosity > 1) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Iteration: " << NLiter << endl;
      cout << "***** Norm of nonlinear residual: " << resnorm[0] << endl;
      cout << "***** Scaled Norm of nonlinear residual: " << resnorm_scaled[0] << endl;
      cout << "*********************************************************" << endl;
    }
    
    if (allowBacktracking && resnorm_scaled[0] > 1.1) {
      solve = false;
      alpha *= 0.5;
      if (is_adjoint) {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        phi->update(-1.0*alpha, *du_over, 1.0);
      }
      else {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        u->update(-1.0*alpha, *du_over, 1.0);
      }
      if (Comm->getRank() == 0 && verbosity > 1) {
        cout << "***** Backtracking: new learning rate = " << alpha << endl;
      }
      
    }
    else {
      if (useRelativeTOL && resnorm_scaled[0]<NLtol) {
        solve = false;
      }
      else if (useAbsoluteTOL && resnorm[0]<NLabstol) {
        solve = false;
      }
    }
    
    
    // *********************** SOLVE THE LINEAR SYSTEM **************************
    
    if (solve) {
      
      if (build_jacobian) {
        linalg->fillComplete(J_over);
        J->resumeFill();
        linalg->exportMatrixFromOverlapped(J, J_over);
        linalg->fillComplete(J);
      }
      
      if (debug_level>2) {
        KokkosTools::print(J,"Jacobian from solver interface");
      }
      du->putScalar(0.0);
      du_over->putScalar(0.0);
      linalg->linearSolver(J, res, du);
      linalg->importVectorToOverlapped(du_over, du);
      
      alpha = 1.0;
      if (is_adjoint) {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        phi->update(alpha, *du_over, 1.0);
      }
      else {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        u->update(alpha, *du_over, 1.0);
      }
    }
    NLiter++; // increment number of iterations
    
    if (NLiter >= maxiter) {
      proceed = false;
      // Need to perform another gather for cases where the number of iterations is tight
      assembler->performGather(u,0,0);
    }
    else if (useRelativeTOL) {
      if (resnorm_scaled[0]<NLtol) {
        proceed = false;
      }
    }
    else if (useAbsoluteTOL) {
      if (resnorm[0]<NLabstol) {
        proceed = false;
      }
    }
  } // while loop
  if (debug_level>1) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> normu(1);
    u->norm2(normu);
    if (Comm->getRank() == 0) {
      cout << "Norm of solution: " << normu[0] << "    (overlapped vector so results may differ on multiple procs)" << endl;
    }
  }
  
  if (debug_level>2) {
    KokkosTools::print(u);
  }
  
  if (Comm->getRank() == 0) {
    if (!is_adjoint) {
      if ( (NLiter>maxNLiter) && verbosity > 1) {
        status = 1;
        cout << endl << endl << "********************" << endl;
        cout << endl << "SOLVER FAILED TO CONVERGE CONVERGED in " << NLiter
        << " iterations with residual norm " << resnorm[0] << endl;
        cout << "********************" << endl;
      }
    }
  }
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Finished SolverManager::nonlinearSolver" << endl;
    }
  }
  return status;
}

// ========================================================================================
// ========================================================================================

template<class Node>
int SolverManager<Node>::explicitSolver(vector_RCP & u, vector_RCP & phi, const int & stage) {
  
  
  Teuchos::TimeMonitor localtimer(*nonlinearsolvertimer);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting SolverManager::explicitSolver ..." << endl;
    }
  }
  
  int status = 0;
  
  if (usestrongDBCs) {
    this->setDirichlet(u);
  }
  
  bool build_jacobian = false;
  
  // *********************** COMPUTE THE RESIDUAL **************************
    
  res_over->putScalar(0.0);
  matrix_RCP J_over;
  
  assembler->assembleJacRes(u, phi, build_jacobian, false, false,
                            res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                            params->num_active_params, params->Psol[0], is_final_time, deltat);
  
  
  linalg->exportVectorFromOverlapped(res, res_over);
  
  // *********************** SOLVE THE LINEAR SYSTEM **************************
  
  // Given m = diag(M^-1)
  // Given timewt = b(stage)*deltat
  // Compute du = timewt*m*res
  // Compute u += du
  
  ScalarT wt = deltat*butcher_b(stage);
  
  du_over->putScalar(0.0);
  
  if (!assembler->lump_mass) {
    res->scale(wt);
    if (use_custom_PCG) {
      linalg->PCG(explicitMass, res, du, diagMass,
                  settings->sublist("Solver").get("linear TOL",1.0e-2),
                  settings->sublist("Solver").get("max linear iters",100));
    }
    else {
      linalg->linearSolverL2(explicitMass, res, du);
    }
    
  }
  else {
    typedef typename Node::execution_space LA_exec;
    
    auto du_view = du->template getLocalView<LA_device>();
    auto res_view = res->template getLocalView<LA_device>();
    auto dm_view = diagMass->template getLocalView<LA_device>();
    
    parallel_for("explicit solver apply invdiag",
                 RangePolicy<LA_exec>(0,du_view.extent(0)),
                 KOKKOS_LAMBDA (const int k ) {
      du_view(k,0) = wt*res_view(k,0)/dm_view(k,0);
    });
  }
  linalg->importVectorToOverlapped(du_over, du);
  
  u->update(1.0, *du_over, 1.0);
  
  if (verbosity>=10) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> unorm(1);
    u->norm2(unorm);
    if (Comm->getRank() == 0) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Explicit integrator: L2 norm of solution: " << unorm[0] << endl;
      cout << "*********************************************************" << endl;
    }
  }
  
  
  assembler->performGather(u,0,0);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Finished SolverManager::explicitSolver" << endl;
    }
  }
  
  return status;
}


// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setDirichlet(vector_RCP & u) {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::setDirichlet ..." << endl;
    }
  }
  
  Teuchos::TimeMonitor localtimer(*dbcsettimer);
  
  typedef typename Node::execution_space LA_exec;
  
  if (usestrongDBCs) {
    auto u_kv = u->template getLocalView<LA_device>();
    //auto meas_kv = meas->getLocalView<HostDevice>();
    
    if (!scalarDirichletData) {
      if (!staticDirichletData) {
        this->projectDirichlet();
      }
      else if (!have_static_Dirichlet_data) {
        this->projectDirichlet();
        have_static_Dirichlet_data = true;
      }
    }
    
    //if (!scalarDirichletData && transientDirichletData) {
    //  this->projectDirichlet();
    //}
    
    vector<vector<Kokkos::View<LO*,LA_device> > > dbcDOFs = assembler->fixedDOF;
    if (scalarDirichletData) {
      
      for (size_t b=0; b<dbcDOFs.size(); b++) {
        for (size_t v=0; v<dbcDOFs[b].size(); v++) {
          if (dbcDOFs[b][v].extent(0)>0) {
            ScalarT value = scalarDirichletValues[b][v];
            auto cdofs = dbcDOFs[b][v];
            parallel_for("solver initial scalar",
                         RangePolicy<LA_exec>(0,cdofs.extent(0)),
                         KOKKOS_LAMBDA (const int i ) {
              u_kv(cdofs(i),0) = value;
            });
          }
        }
      }
    }
    else {
      auto dbc_kv = fixedDOF_soln->template getLocalView<LA_device>();
      for (size_t b=0; b<dbcDOFs.size(); b++) {
        for (size_t v=0; v<dbcDOFs[b].size(); v++) {
          if (dbcDOFs[b][v].extent(0)>0) {
            auto cdofs = dbcDOFs[b][v];
            parallel_for("solver initial scalar",
                         RangePolicy<LA_exec>(0,cdofs.extent(0)),
                         KOKKOS_LAMBDA (const int i ) {
              u_kv(cdofs(i),0) = dbc_kv(cdofs(i),0);
            });
          }
        }
      }
    }
    
    // set point dbcs
    vector<vector<GO> > pointDOFs = disc->point_dofs;
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<GO> pt_dofs = pointDOFs[b];
      Kokkos::View<LO*,LA_device> ptdofs("pointwise dofs", pointDOFs[b].size());
      auto ptdofs_host = Kokkos::create_mirror_view(ptdofs);
      for (size_t i = 0; i < pt_dofs.size(); i++) {
        LO row = linalg->overlapped_map->getLocalElement(pt_dofs[i]); // TMW: this is a temporary fix
        ptdofs_host(i) = row;
      }
      Kokkos::deep_copy(ptdofs,ptdofs_host);
      parallel_for("solver initial scalar",
                   RangePolicy<LA_exec>(0,ptdofs.extent(0)),
                   KOKKOS_LAMBDA (const int i ) {
        LO row = ptdofs(i);
        u_kv(row,0) = 0.0; // fix to zero for now
      });
    }
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager::setDirichlet" << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > SolverManager<Node>::setInitialParams() {
  vector_RCP initial = linalg->getNewParamOverlappedVector();
  ScalarT value = 2.0;
  initial->putScalar(value);
  return initial;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > SolverManager<Node>::setInitial() {
  
  Teuchos::TimeMonitor localtimer(*initsettimer);
  typedef typename Node::execution_space LA_exec;
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::setInitial ..." << endl;
    }
  }
  
  vector_RCP initial = linalg->getNewOverlappedVector();
  initial->putScalar(0.0);
  
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
  
  if (have_initial_conditions) {
    if (scalarInitialData) {
      
      auto initial_kv = initial->template getLocalView<LA_device>();
      
      for (size_t block=0; block<assembler->cellData.size(); block++) {
      
        if (assembler->cellData[block]->numElem > 0) {
          
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
              parallel_for("solver initial scalar",
                           RangePolicy<LA_exec>(0,LIDs.extent(0)),
                           KOKKOS_LAMBDA (const int e ) {
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
              parallel_for("solver initial scalar",
                           RangePolicy<LA_exec>(0,LIDs.extent(0)),
                           KOKKOS_LAMBDA (const int e ) {
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
      
    }
    else {
      
      vector_RCP glinitial = linalg->getNewVector();
      
      if (initial_type == "L2-projection") {
        // Compute the L2 projection of the initial data into the discrete space
        vector_RCP rhs = linalg->getNewOverlappedVector();
        matrix_RCP mass = linalg->getNewOverlappedMatrix();
        vector_RCP glrhs = linalg->getNewVector();
        matrix_RCP glmass = linalg->getNewMatrix();
        
        assembler->setInitial(rhs, mass, is_adjoint);
        
        linalg->exportMatrixFromOverlapped(glmass, mass);
        linalg->exportVectorFromOverlapped(glrhs, rhs);
        
        linalg->fillComplete(glmass);
        linalg->linearSolverL2(glmass, glrhs, glinitial);
        linalg->importVectorToOverlapped(initial, glinitial);
        linalg->resetJacobian();
      }
      else if (initial_type == "L2-projection-HFACE") {
        // Similar to above, but the basis support only exists on the mesh skeleton
        // The use case is setting the IC at the coarse-scale
        vector_RCP rhs = linalg->getNewOverlappedVector();
        matrix_RCP mass = linalg->getNewOverlappedMatrix();
        vector_RCP glrhs = linalg->getNewVector();
        matrix_RCP glmass = linalg->getNewMatrix();

        assembler->setInitialFace(rhs, mass, is_adjoint);

        linalg->exportMatrixFromOverlapped(glmass, mass);
        linalg->exportVectorFromOverlapped(glrhs, rhs);
        linalg->fillComplete(glmass);

        // With HFACE we ensure the preconditioner is not 
        // used for this projection (mass matrix is nearly the identity
        // and can cause issues)
        auto origPreconFlag = linalg->options_L2->usePreconditioner;
        linalg->options_L2->usePreconditioner = false;
        // do the solve
        linalg->linearSolverL2(glmass, glrhs, glinitial);
        // set back to original
        linalg->options_L2->usePreconditioner = origPreconFlag;

        linalg->importVectorToOverlapped(initial, glinitial);
        linalg->resetJacobian(); // TODO not sure of this

      } 
      else if (initial_type == "interpolation") {
        
        assembler->setInitial(initial, is_adjoint);
        
      }
    }
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager::setInitial ..." << endl;
    }
  }
  
  return initial;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setBatchID(const int & bID){
  batchID = bID;
  params->batchID = bID;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > SolverManager<Node>::blankState(){
  vector_RCP F_soln = linalg->getNewOverlappedVector();
  return F_soln;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeParams() {
  
  //for (size_t b=0; b<blocknames.size(); b++) {
  //  assembler->wkset[b]->paramusebasis = params->discretized_param_usebasis;
  //  assembler->wkset[b]->paramoffsets = params->paramoffsets[0];
  // }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeMultiscale() {
  if (multiscale_manager->subgridModels.size() > 0 ) {
    for (size_t k=0; k<multiscale_manager->subgridModels.size(); k++) {
      multiscale_manager->subgridModels[k]->paramvals_KVAD = params->paramvals_KVAD;
      //  multiscale_manager->subgridModels[k]->wkset[0]->paramnames = paramnames;
    }
    
    multiscale_manager->macro_wkset = assembler->wkset;
    vector<Kokkos::View<int*,AssemblyDevice>> macro_numDOF;
    for (size_t b=0; b<assembler->cellData.size(); ++b) {
      macro_numDOF.push_back(assembler->cellData[b]->numDOF);
    }
    multiscale_manager->setMacroInfo(disc->basis_pointers, disc->basis_types,
                                     phys->varlist, useBasis, disc->offsets,
                                     macro_numDOF,
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

// Explicit template instantiations
template class MrHyDE::SolverManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
template class MrHyDE::SolverManager<SubgridSolverNode>;
#endif
