/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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
                                   Teuchos::RCP<PhysicsInterface> & physics_,
                                   Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                                   Teuchos::RCP<ParameterManager<Node> > & params_) :
Comm(Comm_), settings(settings_), mesh(mesh_), disc(disc_), physics(physics_), assembler(assembler_), params(params_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), Comm));
  
  debugger->print("**** Starting SolverManager constructor ...");
  
  numEvaluations = 0;
  setnames = physics->set_names;
  store_vectors = false;
  if (setnames.size() > 1) {
    store_vectors = false;
  }
  // Get the required information from the settings
  dimension = physics->dimension;
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
  
  use_custom_PCG = settings->sublist("Solver").get<bool>("use custom PCG",false);
  
  NLtol = settings->sublist("Solver").get<double>("nonlinear TOL",1.0E-6);
  NLabstol = settings->sublist("Solver").get<double>("absolute nonlinear TOL",std::min((double)NLtol,(double)1.0E-6));
  maxNLiter = settings->sublist("Solver").get<int>("max nonlinear iters",10);
  subcycles = settings->sublist("Solver").get<int>("max subcycles",1);
  useRelativeTOL = settings->sublist("Solver").get<bool>("use relative TOL",true);
  useAbsoluteTOL = settings->sublist("Solver").get<bool>("use absolute TOL",false);
  allowBacktracking = settings->sublist("Solver").get<bool>("allow backtracking",true);
  compute_fwd_sens = settings->sublist("Solver").get<bool>("compute forward sensitivities",false);
  
  maxTimeStepCuts = settings->sublist("Solver").get<int>("maximum time step cuts",5);
  amplification_factor = settings->sublist("Solver").get<double>("explicit amplification factor",10.0);
  
  use_param_mass = settings->sublist("Solver").get<bool>("use parameter mass",false);
  
  line_search = false;//settings->sublist("Solver").get<bool>("Use Line Search","false");
  store_adjPrev = false;
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  if (!isTransient) {
    deltat = 1.0;
  }
  
  // Explicit integration mode - may disable some features
  fully_explicit = settings->sublist("Solver").get<bool>("fully explicit",false);
  
  initial_type = settings->sublist("Solver").get<string>("initial type","L2-projection");
  
  // needed information from the mesh
  blocknames = physics->block_names;
  
  // needed information from the physics interface
  numVars = physics->num_vars; //
  vector<vector<vector<string> > > phys_varlist = physics->var_list;
  size_t numSets = setnames.size();
  
  // needed information from the disc interface
  vector<vector<int> > cards = disc->cards;
  
  for (size_t set=0; set<numSets; ++set) {
    vector<vector<int> > set_useBasis;
    vector<vector<int> > set_numBasis;
    vector<vector<string> > set_varlist;
    
    vector<size_t> set_maxBasis;
    
    for (size_t block=0; block<blocknames.size(); ++block) {
      
      vector<int> block_useBasis(numVars[set][block]);
      vector<int> block_numBasis(numVars[set][block]);
      vector<string> block_varlist(numVars[set][block]);
      
      int block_maxBasis = 0;
      for (size_t j=0; j<numVars[set][block]; j++) {
        string var = phys_varlist[set][block][j];
        int vub = physics->getUniqueIndex(set,block,var);
        block_varlist[j] = var;
        block_useBasis[j] = vub;
        block_numBasis[j] = cards[block][vub];
        block_maxBasis = std::max(block_maxBasis,cards[block][vub]);
      }
      
      set_varlist.push_back(block_varlist);
      set_useBasis.push_back(block_useBasis);
      set_numBasis.push_back(block_numBasis);
      set_maxBasis.push_back((size_t)block_maxBasis);
      
    }
    varlist.push_back(set_varlist);
    useBasis.push_back(set_useBasis);
    numBasis.push_back(set_numBasis);
    maxBasis.push_back(set_maxBasis);
    
  }

  // TODO Keeping a separate loop for now for clarity
  // we can combine if we stick with this

  for (size_t set=0; set<numSets; ++set) {

    // The Butcher tableau and BDF coefficients can vary by physics set.
    // TODO NOT YET BY BLOCK
    // If they are universal, we get the values here.
    // If set-specific values are supplied later, they are overwritten

    string myButcherTab = settings->sublist("Solver").get<string>("transient Butcher tableau","BWE");
    int myBDForder = settings->sublist("Solver").get<int>("transient BDF order",1);

    // Additional parameters for higher-order BDF methods that require some startup procedure
    string myStartupButcherTab = settings->sublist("Solver").get<string>("transient startup Butcher tableau",myButcherTab);
    int myStartupBDForder = settings->sublist("Solver").get<int>("transient startup BDF order",myBDForder);
    int myStartupSteps = settings->sublist("Solver").get<int>("transient startup steps",myBDForder);

    // TODO allow to vary by block...
    // Check if there are settings unique to each set
    auto setSolverSettings = physics->solver_settings[set][0]; // [set][block]

    myButcherTab = 
      setSolverSettings.get<string>("transient Butcher tableau",myButcherTab);
    myBDForder = 
      setSolverSettings.get<int>("transient BDF order",1);
    myStartupButcherTab = 
      setSolverSettings.get<string>("transient startup Butcher tableau",myStartupButcherTab);
    myStartupBDForder = 
      setSolverSettings.get<int>("transient startup BDF order",myStartupBDForder);
    myStartupSteps = 
      setSolverSettings.get<int>("transient startup steps",myStartupSteps);

    if (myBDForder>1) {
      if (myButcherTab == "custom") {
        cout << "Warning: running a higher order BDF method with anything other than BWE/DIRK-1,1 is risky." << endl;
        cout << "The code will run, but the results may be nonsense" << endl;
      }
      else {
        if (myButcherTab != "BWE" && myButcherTab != "DIRK-1,1") {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: need to use BWE or DIRK-1,1 with higher order BDF");
        }
      }
    }
    if (myStartupBDForder>1) {
      if (myStartupButcherTab == "custom") {
        cout << "Warning: running a higher order BDF method with anything other than BWE/DIRK-1,1 is risky." << endl;
        cout << "The code will run, but the results may be nonsense" << endl;
      }
      else {
        if (myStartupButcherTab != "BWE" && myStartupButcherTab != "DIRK-1,1") {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: need to use BWE or DIRK-1,1 with higher order BDF");
        }
      }
    }

    ButcherTab.push_back(myButcherTab);
    BDForder.push_back(myBDForder);
    startupButcherTab.push_back(myStartupButcherTab); 
    startupBDForder.push_back(myStartupBDForder);
    startupSteps.push_back(myStartupSteps);

  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Create linear algebra interface
  /////////////////////////////////////////////////////////////////////////////
  
  //linalg = Teuchos::rcp( new LinearAlgebraInterface<Node>(Comm, settings, disc, params) );
    
  /////////////////////////////////////////////////////////////////////////////
  // Worksets
  /////////////////////////////////////////////////////////////////////////////
  
  assembler->createWorkset();
  
  // initialize vector which holds the number BDF steps and RK stages for each set 
  numsteps.resize(numSets,0);
  numstages.resize(numSets,0);
  maxnumsteps.resize(numSets,0);
  maxnumstages.resize(numSets,0);

  // set for all physics sets
  for (size_t set=0; set<numSets; ++set) {
    this->setBackwardDifference(BDForder,set);
    this->setButcherTableau(ButcherTab,set);
    if (BDForder[set] > 1) {
      this->setBackwardDifference(startupBDForder,set);
      this->setButcherTableau(startupButcherTab,set);
    }
  }
  this->finalizeWorkset();
  
  physics->setWorkset(assembler->wkset);
#ifndef MrHyDE_NO_AD
  physics->setWorkset(assembler->wkset_AD);
  physics->setWorkset(assembler->wkset_AD2);
  physics->setWorkset(assembler->wkset_AD4);
  physics->setWorkset(assembler->wkset_AD8);
  physics->setWorkset(assembler->wkset_AD16);
  physics->setWorkset(assembler->wkset_AD18);
  physics->setWorkset(assembler->wkset_AD24);
  physics->setWorkset(assembler->wkset_AD32);
#endif
  
  //if (store_vectors) {
  //  for (size_t set=0; set<numSets; ++set) {
  //    res.push_back(linalg->getNewVector(set));
  //    res_over.push_back(linalg->getNewOverlappedVector(set));
  //    du_over.push_back(linalg->getNewOverlappedVector(set));
  //    du.push_back(linalg->getNewVector(set));
  //  }
  //}

  /////////////////////////////////////////////////////////////////////////////
  
  this->setBatchID(Comm->getRank());
  
  /////////////////////////////////////////////////////////////////////////////
  
  //this->setupFixedDOFs(settings);
  
  /////////////////////////////////////////////////////////////////////////////
  
  have_initial_conditions = vector<bool>(numSets,false);
  scalarInitialData = vector<bool>(numSets,false);
  have_static_Dirichlet_data = vector<bool>(numSets,false);
  
  for (size_t set=0; set<numSets; ++set) {
    have_initial_conditions[set] = false;
    for (size_t block=0; block<blocknames.size(); ++block) {
      if (physics->physics_settings[set][block].isSublist("Initial conditions")) {
        have_initial_conditions[set] = true;
        scalarInitialData[set] = physics->physics_settings[set][block].sublist("Initial conditions").get<bool>("scalar data", false);
      }
    }
  
    if (have_initial_conditions[set] && scalarInitialData[set]) {
      vector<vector<ScalarT> > setInitialValues;
      for (size_t block=0; block<blocknames.size(); ++block) {
        
        std::string blockID = blocknames[block];
        Teuchos::ParameterList init_settings = physics->physics_settings[set][block].sublist("Initial conditions");
        
        vector<ScalarT> blockInitialValues;
        
        for (size_t var=0; var<varlist[set][block].size(); var++ ) {
          ScalarT value = 0.0;
          if (init_settings.isParameter(varlist[set][block][var])) {
            value = init_settings.get<ScalarT>(varlist[set][block][var]);
          }
          blockInitialValues.push_back(value);
        }
        setInitialValues.push_back(blockInitialValues);
      }
      scalarInitialValues.push_back(setInitialValues);
    }
    
  }
  
  debugger->print("**** Finished SolverManager constructor");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::completeSetup() {

  debugger->print("**** Starting SolverManager::completeSetup()");
  
  /////////////////////////////////////////////////////////////////////////////
  // Create linear algebra interface
  /////////////////////////////////////////////////////////////////////////////
  
  linalg = Teuchos::rcp( new LinearAlgebraInterface<Node>(Comm, settings, disc, params) );
  
  if (store_vectors) {
    for (size_t set=0; set<setnames.size(); ++set) {
      res.push_back(linalg->getNewVector(set));
      res_over.push_back(linalg->getNewOverlappedVector(set));
      du_over.push_back(linalg->getNewOverlappedVector(set));
      du.push_back(linalg->getNewVector(set));
    }
  }
  this->setupFixedDOFs(settings);

  //---------------------------------------------------
  // Mass matrix (lumped and maybe full) for explicit
  //---------------------------------------------------
  
  if (fully_explicit) {
    this->setupExplicitMass();
  }
  
  if (use_param_mass && params->num_discretized_params > 0) {
    this->setupDiscretizedParamMass();
  }
  
  debugger->print("**** Finished SolverManager::completeSetup()");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupExplicitMass() {

  debugger->print("**** Starting SolverManager::setupExplicitMass()");
  
  bool compute_matrix = true;
  if (assembler->lump_mass || assembler->matrix_free) {
    compute_matrix = false;
  }
  
  for (size_t set=0; set<useBasis.size(); ++set) {
    matrix_RCP mass;
    
    assembler->updatePhysicsSet(set);
    if (compute_matrix) {
      explicitMass.push_back(linalg->getNewMatrix(set));
      if (linalg->getHaveOverlapped()) {
        mass = linalg->getNewOverlappedMatrix(set);
      }
      else {
        mass = explicitMass[set];
      }
    }
    
    diagMass.push_back(linalg->getNewVector(set));
    vector_RCP diagMass_over;
    if (linalg->getHaveOverlapped()) {
      diagMass_over = linalg->getNewOverlappedVector(set);
    } 
    else {
      diagMass_over = diagMass[set];
    }
    
    assembler->getWeightedMass(set,mass,diagMass_over);
    
    if (linalg->getHaveOverlapped()) {
      linalg->exportVectorFromOverlapped(set,diagMass[set], diagMass_over);
      if (compute_matrix) {
        linalg->exportMatrixFromOverlapped(set,explicitMass[set], mass);
      }
    }
    
  }

  debugger->print("**** Starting SolverManager::setupExplicitMass() - fillComplete");
  
  for (size_t set=0; set<useBasis.size(); ++set) {
    
    if (compute_matrix) {
      linalg->fillComplete(explicitMass[set]);
    }
    
    if (store_vectors) {
      q_pcg.push_back(linalg->getNewVector(set));
      z_pcg.push_back(linalg->getNewVector(set));
      p_pcg.push_back(linalg->getNewVector(set));
      r_pcg.push_back(linalg->getNewVector(set));
      if (linalg->getHaveOverlapped() && assembler->matrix_free) {
        q_pcg_over.push_back(linalg->getNewOverlappedVector(set));
        p_pcg_over.push_back(linalg->getNewOverlappedVector(set));
      }
    }
  }
  
  debugger->print("**** Finished SolverManager::setupExplicitMass()");
  
}


// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupDiscretizedParamMass() {

  debugger->print("**** Starting SolverManager::setupDiscretizedParamMass()");
  
  // Hard coding this to always assemble the matrix
  // Can relax this and allow matrix-free later
  bool compute_matrix = true;
  
  matrix_RCP pmass;
  
  if (compute_matrix) {
    
    paramMass = linalg->getNewParamMatrix();
    
    if (linalg->getHaveOverlapped()) {
      pmass = linalg->getNewOverlappedParamMatrix();
    }
    else {
      pmass = paramMass;
    }
    
  }
  
  diagParamMass = linalg->getNewParamVector();
  vector_RCP diagParamMass_over;
  if (linalg->getHaveOverlapped()) {
    diagParamMass_over = linalg->getNewOverlappedParamVector();
  }
  else { // squeeze out memory for single rank demos
    diagParamMass_over = diagParamMass;
  }
  
  assembler->getParamMass(pmass,diagParamMass_over);
  
  if (linalg->getHaveOverlapped()) {
    linalg->exportParamVectorFromOverlapped(diagParamMass, diagParamMass_over);
    if (compute_matrix) {
      linalg->exportParamMatrixFromOverlapped(paramMass, pmass);
    }
  }
  

  if (compute_matrix) {
    linalg->fillComplete(paramMass);
  }
  
  params->setParamMass(diagParamMass, paramMass);
  
  debugger->print("**** Finished SolverManager::setupDiscretizedParamMass()");
  
}

//========================================================================
//========================================================================

template<class Node>
void SolverManager<Node>::setButcherTableau(const vector<string> & tableau, const int & set) {

  for (size_t block=0; block<assembler->groups.size(); ++block) {

    // TODO the RK scheme cannot be specified block by block

    auto myTableau = tableau[set];

    // only filling in the non-zero entries

    if (myTableau == "BWE" || myTableau == "DIRK-1,1") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
      butcher_A(0,0) = 1.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
      butcher_b(0) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
      butcher_c(0) = 1.0;
    }
    else if (myTableau == "FWE") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
      butcher_b(0) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
    }
    else if (myTableau == "CN") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(1,0) = 0.5;
      butcher_A(1,1) = 0.5;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 0.5;
      butcher_b(1) = 0.5;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(1) = 1.0;
    }
    else if (myTableau == "SSPRK-3,3") {
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
    else if (myTableau == "RK-4,4") { // Classical RK4
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
    else if (myTableau == "DIRK-1,2") {
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",1,1);
      butcher_A(0,0) = 0.5;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",1);
      butcher_b(0) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",1);
      butcher_c(0) = 0.5;
    }
    else if (myTableau == "DIRK-2,2") { // 2-stage, 2nd order
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
    else if (myTableau == "DIRK-2,3") { // 2-stage, 3rd order
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
    else if (myTableau == "DIRK-3,3") { // 3-stage, 3rd order
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
    else if (myTableau == "leap-frog") { // Leap-frog for Maxwells
      butcher_A = Kokkos::View<ScalarT**,HostDevice>("butcher_A",2,2);
      butcher_A(1,0) = 1.0;
      butcher_b = Kokkos::View<ScalarT*,HostDevice>("butcher_b",2);
      butcher_b(0) = 1.0;
      butcher_b(1) = 1.0;
      butcher_c = Kokkos::View<ScalarT*,HostDevice>("butcher_c",2);
      butcher_c(0) = 0.0;
      butcher_c(1) = 0.0;
    }
    else if (myTableau == "custom") {

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
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Butcher tableau:" + tableau[set]);
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

    //block_butcher_A.push_back(dev_butcher_A);
    //block_butcher_b.push_back(dev_butcher_b);
    //block_butcher_c.push_back(dev_butcher_c);
  
    int newnumstages = butcher_A.extent(0);

    maxnumstages[set] = std::max(numstages[set],newnumstages);
    numstages[set] = newnumstages;
  
    assembler->setWorksetButcher(set, block, dev_butcher_A, dev_butcher_b, dev_butcher_c);

  } // end for blocks
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setBackwardDifference(const vector<int> & order, const int & set) { // using order as an input to allow for dynamic changes

  // TODO rearrange this? and setButcher...

  for (size_t block=0; block<assembler->groups.size(); ++block) {

    // TODO currently, the BDF wts cannot be specified block by block

    Kokkos::View<ScalarT*,AssemblyDevice> dev_BDF_wts;
    Kokkos::View<ScalarT*,HostDevice> BDF_wts;

    // Note that these do not include 1/deltat (added in wkset)
    // Not going to work properly for adaptive time stepping if BDForder>1

    auto myOrder = order[set];

    if (isTransient) {

      if (myOrder == 1) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",2);
        BDF_wts(0) = 1.0;
        BDF_wts(1) = -1.0;
      }
      else if (myOrder == 2) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",3);
        BDF_wts(0) = 1.5;
        BDF_wts(1) = -2.0;
        BDF_wts(2) = 0.5;
      }
      else if (myOrder == 3) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",4);
        BDF_wts(0) = 11.0/6.0;
        BDF_wts(1) = -3.0;
        BDF_wts(2) = 1.5;
        BDF_wts(3) = -1.0/3.0;
      }
      else if (myOrder == 4) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",5);
        BDF_wts(0) = 25.0/12.0;
        BDF_wts(1) = -4.0;
        BDF_wts(2) = 3.0;
        BDF_wts(3) = -4.0/3.0;
        BDF_wts(4) = 1.0/4.0;
      }
      else if (myOrder == 5) {
        BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",6);
        BDF_wts(0) = 137.0/60.0;
        BDF_wts(1) = -5.0;
        BDF_wts(2) = 5.0;
        BDF_wts(3) = -10.0/3.0;
        BDF_wts(4) = 75.0/60.0;
        BDF_wts(5) = -1.0/5.0;
      }
      else if (myOrder == 6) {
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

      maxnumsteps[set] = std::max(maxnumsteps[set],newnumsteps);
      numsteps[set] = newnumsteps;

    }
    else { // for steady state solves, u_dot = 0.0*u
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",1);
      BDF_wts(0) = 1.0;
      numsteps[set] = 1;
      maxnumsteps[set] = 1;
    }

    dev_BDF_wts = Kokkos::View<ScalarT*,AssemblyDevice>("BDF weights on device",BDF_wts.extent(0));
    Kokkos::deep_copy(dev_BDF_wts, BDF_wts);
    
    assembler->setWorksetBDF(set, block, dev_BDF_wts);
  } // end loop blocks
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeWorkset() {
  
  debugger->print("**** Starting SolverManager::finalizeWorkset ...");
  
  this->finalizeWorkset(assembler->wkset, params->paramvals_KV, params->paramdot_KV);
#ifndef MrHyDE_NO_AD
  this->finalizeWorkset(assembler->wkset_AD, params->paramvals_KVAD, params->paramdot_KVAD);
  this->finalizeWorkset(assembler->wkset_AD2, params->paramvals_KVAD2, params->paramdot_KVAD2);
  this->finalizeWorkset(assembler->wkset_AD4, params->paramvals_KVAD4, params->paramdot_KVAD4);
  this->finalizeWorkset(assembler->wkset_AD8, params->paramvals_KVAD8, params->paramdot_KVAD8);
  this->finalizeWorkset(assembler->wkset_AD16, params->paramvals_KVAD16, params->paramdot_KVAD16);
  this->finalizeWorkset(assembler->wkset_AD18, params->paramvals_KVAD18, params->paramdot_KVAD18);
  this->finalizeWorkset(assembler->wkset_AD24, params->paramvals_KVAD24, params->paramdot_KVAD24);
  this->finalizeWorkset(assembler->wkset_AD32, params->paramvals_KVAD32, params->paramdot_KVAD32);
#endif
  
  debugger->print("**** Finished SolverManager::finalizeWorkset");
  
  
}

template<class Node>
template<class EvalT>
void SolverManager<Node>::finalizeWorkset(vector<Teuchos::RCP<Workset<EvalT> > > & wkset,
                                          Kokkos::View<EvalT**,AssemblyDevice> paramvals_KV,
                                          Kokkos::View<EvalT**,AssemblyDevice> paramdot_KV) {

  // Determine the offsets for each set as a Kokkos View
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<physics->set_names.size(); set++) {
        vector<vector<int> > voffsets = disc->offsets[set][block];
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
        wkset[block]->set_offsets.push_back(offsets_view);
        if (set == 0) {
          wkset[block]->offsets = offsets_view;
        }

      }
    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      
      vector<vector<int> > block_useBasis;
      vector<vector<string> > block_varlist;
      
      for (size_t set=0; set<useBasis.size(); ++set) {
        block_useBasis.push_back(useBasis[set][block]);
        block_varlist.push_back(varlist[set][block]);
      }
      wkset[block]->set_usebasis = block_useBasis;
      wkset[block]->set_varlist = block_varlist;
      wkset[block]->usebasis = block_useBasis[0];
      wkset[block]->varlist = block_varlist[0];
      
    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      // set defaults for time integration params since these
      // won't get set if the total number of sets is 1
      wkset[block]->butcher_A = wkset[block]->set_butcher_A[0];
      wkset[block]->butcher_b = wkset[block]->set_butcher_b[0];
      wkset[block]->butcher_c = wkset[block]->set_butcher_c[0];
      wkset[block]->BDF_wts = wkset[block]->set_BDF_wts[0];
      // update workset for first physics set
      wkset[block]->updatePhysicsSet(0);

    }
  }
  
  // Parameters do not depend on physics sets
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
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
      wkset[block]->paramusebasis = params->discretized_param_usebasis;
      wkset[block]->paramoffsets = poffsets_view;
      wkset[block]->param_varlist = params->discretized_param_names;

    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      wkset[block]->createSolutionFields();
    }
  }
  
  for (size_t block=0; block<wkset.size(); ++block) {
    if (wkset[block]->isInitialized) {
      vector<vector<int> > block_useBasis;
      for (size_t set=0; set<useBasis.size(); ++set) {
        block_useBasis.push_back(useBasis[set][block]);
      }
      assembler->groupData[block]->setSolutionFields(maxnumsteps, maxnumstages);
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        assembler->groups[block][grp]->setUseBasis(block_useBasis, maxnumsteps, maxnumstages, false);
        assembler->groups[block][grp]->setUpSubGradient(params->num_active_params);
      }
      
      wkset[block]->params_AD = paramvals_KV;
      wkset[block]->params_dot_AD = paramdot_KV;
      wkset[block]->paramnames = params->paramnames;
      wkset[block]->setTime(current_time);

      if (assembler->boundary_groups.size() > block) { // avoid seg faults
        for (size_t grp=0; grp<assembler->boundary_groups[block].size(); ++grp) {
          if (assembler->boundary_groups[block][grp]->numElem > 0) {
            assembler->boundary_groups[block][grp]->setUseBasis(block_useBasis, maxnumsteps, maxnumstages, false);
          }
        }
      }
    }
  }
  
  
}

// ========================================================================================
// Set up the logicals and data structures for the fixed DOF (Dirichlet and point constraints)
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupFixedDOFs(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  Teuchos::TimeMonitor localtimer(*fixeddofsetuptimer);
  
  debugger->print("**** Starting SolverManager::setupFixedDOFs()");
  
  if (!disc->have_dirichlet) {
    usestrongDBCs = false;
  }
  
  size_t numSets = physics->set_names.size();
  
  scalarDirichletData = vector<bool>(numSets,false);
  staticDirichletData = vector<bool>(numSets,true);
  
  if (usestrongDBCs) {
    for (size_t set=0; set<numSets; ++set) {
      fixedDOF_soln.push_back(linalg->getNewOverlappedVector(set));
    }
    for (size_t set=0; set<numSets; ++set) {
    
      scalarDirichletData[set] = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("scalar data", false);
      staticDirichletData[set] = settings->sublist("Physics").sublist("Dirichlet conditions").get<bool>("static data", true);
      
      if (scalarDirichletData[set] && !staticDirichletData[set]) {
        if (Comm->getRank() == 0) {
          cout << "Warning: The Dirichlet data was set to scalar and non-static.  This should not happen." << endl;
        }
      }
      
      if (scalarDirichletData[set]) {
        vector<vector<ScalarT> > setDirichletValues;
        for (size_t block=0; block<blocknames.size(); ++block) {
          
          std::string blockID = blocknames[block];
          Teuchos::ParameterList dbc_settings = physics->physics_settings[set][block].sublist("Dirichlet conditions");
          vector<ScalarT> blockDirichletValues;
          
          for (size_t var=0; var<varlist[set][block].size(); var++ ) {
            ScalarT value = 0.0;
            if (dbc_settings.isSublist(varlist[set][block][var])) {
              if (dbc_settings.sublist(varlist[set][block][var]).isParameter("all boundaries")) {
                value = dbc_settings.sublist(varlist[set][block][var]).template get<ScalarT>("all boundaries");
              }
              else {
                Teuchos::ParameterList currdbcs = dbc_settings.sublist(varlist[set][block][var]);
                Teuchos::ParameterList::ConstIterator d_itr = currdbcs.begin();
                while (d_itr != currdbcs.end()) {
                  value = currdbcs.get<ScalarT>(d_itr->first);
                  d_itr++;
                }
              }
            }
            blockDirichletValues.push_back(value);
          }
          setDirichletValues.push_back(blockDirichletValues);
        }
        scalarDirichletValues.push_back(setDirichletValues);
      }
    }
  }
  
  debugger->print("**** Finished SolverManager::setupFixedDOFs()");
  
}

// ========================================================================================
// Set up the logicals and data structures for the fixed DOF (Dirichlet and point constraints)
// ========================================================================================

template<class Node>
void SolverManager<Node>::projectDirichlet(const size_t & set) {
  
  Teuchos::TimeMonitor localtimer(*dbcprojtimer);
  
  debugger->print(1, "**** Starting SolverManager::projectDirichlet()");
  
  assembler->updatePhysicsSet(set);
  
  if (usestrongDBCs) {
    
    if (fixedDOF_soln.size() > set) {
      fixedDOF_soln[set] = linalg->getNewOverlappedVector(set);
    }
    else {
      fixedDOF_soln.push_back(linalg->getNewOverlappedVector(set));
    }
    
    vector_RCP glfixedDOF_soln = linalg->getNewVector(set);
    
    vector_RCP rhs = linalg->getNewOverlappedVector(set);
    matrix_RCP mass = linalg->getNewOverlappedMatrix(set);
    vector_RCP glrhs = linalg->getNewVector(set);
    matrix_RCP glmass = linalg->getNewMatrix(set);
    
    assembler->setDirichlet(set, rhs, mass, is_adjoint, current_time);
    
    linalg->exportMatrixFromOverlapped(set, glmass, mass);
    linalg->exportVectorFromOverlapped(set, glrhs, rhs);
    linalg->fillComplete(glmass);
    
    // TODO BWR -- couldn't think of a good way to protect against
    // the preconditioner failing for HFACE, will need to be handled
    // explicitly in the input file for now (State boundary L2 linear solver)
    linalg->linearSolverBoundaryL2(set, glmass, glrhs, glfixedDOF_soln);
    linalg->importVectorToOverlapped(set, fixedDOF_soln[set], glfixedDOF_soln);
    
  }
  
  debugger->print(1, "**** Finished SolverManager::projectDirichlet()");
  
}

// ========================================================================================
/* given the parameters, solve the forward problem */
// ========================================================================================

template<class Node>
void SolverManager<Node>::forwardModel(ScalarT & objective) {
  
  Teuchos::TimeMonitor localtimer(*forwardtimer);
  
  current_time = initial_time;
  
  debugger->print("**** Starting SolverManager::forwardModel ...");
  
  is_adjoint = false;
  params->sacadoizeParams(false);
  postproc->resetObjectives();
  postproc->resetSolutions();
  linalg->resetJacobian();
  
  for (size_t set=0; set<setnames.size(); ++set) {
    if (!scalarDirichletData[set]) {
      if (!staticDirichletData[set]) {
        this->projectDirichlet(set);
      }
      else if (!have_static_Dirichlet_data[set]) {
        this->projectDirichlet(set);
        have_static_Dirichlet_data[set] = true;
      }
    }
  }
  
  vector<vector_RCP> sol = this->setInitial();
    
  if (solver_type == "steady-state") {
    this->steadySolver(sol);
  }
  else if (solver_type == "transient") {
    MrHyDE_OptVector gradient; // not really used here
    this->transientSolver(sol, gradient, initial_time, final_time);
  }
  else {
    // print out an error message
  }
    
  if (postproc->write_optimization_solution) {
    postproc->writeOptimizationSolution(numEvaluations);
  }
  
  postproc->reportObjective(objective);
  
  numEvaluations++;
  
  debugger->print("**** Finished SolverManager::forwardModel");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::adjointModel(MrHyDE_OptVector & gradient) {
  
  debugger->print("**** Starting SolverManager::adjointModel ...");
  
  Teuchos::TimeMonitor localtimer(*adjointtimer);
  
  if (setnames.size()>1 && Comm->getRank() == 0) {
    cout << "MrHyDE WARNING: Adjoints are not yet implemented for multiple physics sets." << endl;
  }
  else {
    
    is_adjoint = true;
    
    params->sacadoizeParams(false);
    linalg->resetJacobian();
    
    vector<vector_RCP> phi = setInitial();
    
    if (solver_type == "steady-state") {
      // Since this is the adjoint solve, we loop over the physics sets in reverse order
      for (size_t oset=0; oset<phi.size(); ++oset) {
        size_t set = phi.size()-1-oset;
        vector<vector_RCP> sol, zero_vec;
        for (size_t iset=0; iset<phi.size(); ++iset) { // just collecting states - order doesn't matter
          sol.push_back(linalg->getNewVector(iset));
          bool fnd = postproc->soln[set]->extract(sol[iset], current_time);
          if (!fnd) {
            cout << "UNABLE TO FIND FORWARD SOLUTION" << endl;
          }
        }
        params->updateDynamicParams(0);
        this->nonlinearSolver(set, 0, sol, sol, zero_vec, phi, phi, zero_vec);
        
        postproc->computeSensitivities(sol, zero_vec, zero_vec, phi, 0, current_time, deltat, gradient);
      }
    }
    else if (solver_type == "transient") {
      DFAD obj = 0.0;
      this->transientSolver(phi, gradient, initial_time, final_time);
    }
    else {
      // print out an error message
    }
    
    is_adjoint = false;
  }
  
  debugger->print("**** Finished SolverManager::adjointModel");
  
}

// ========================================================================================
// solve an incremental forward problem for the incremental adjoint
// ========================================================================================

template<class Node>
void SolverManager<Node>::incrementalForwardModel(ScalarT & objective) {
  
}

// ========================================================================================
// solve an incremental adjoint for the hessian-vector product
// This should only be called after a forward solve, an adjoint solve, and an incremental forward solve
// ========================================================================================

template<class Node>
void SolverManager<Node>::incrementalAdjointModel(MrHyDE_OptVector & hessvec) {
  
  debugger->print("**** Starting SolverManager::incrementalAdjointModel ...");
  
  Teuchos::TimeMonitor localtimer(*adjointtimer);
  
  if (setnames.size()>1 && Comm->getRank() == 0) {
    cout << "MrHyDE WARNING: Adjoints are not yet implemented for multiple physics sets." << endl;
  }
  else {
    
    is_adjoint = true;
    
    params->sacadoizeParams(false);
    linalg->resetJacobian();
    
    vector<vector_RCP> phi = setInitial();
    
    if (solver_type == "steady-state") {
      // Since this is the adjoint solve, we loop over the physics sets in reverse order
      for (size_t oset=0; oset<phi.size(); ++oset) {
        size_t set = phi.size()-1-oset;
        vector<vector_RCP> sol, zero_vec;
        for (size_t iset=0; iset<phi.size(); ++iset) { // just collecting states - order doesn't matter
          sol.push_back(linalg->getNewVector(iset));
          bool fnd = postproc->soln[set]->extract(sol[iset], current_time);
          if (!fnd) {
            cout << "UNABLE TO FIND FORWARD SOLUTION" << endl;
          }
        }
        params->updateDynamicParams(0);
        this->nonlinearSolver(set, 0, sol, sol, zero_vec, phi, phi, zero_vec);
        
        //postproc->computeSensitivities(sol, zero_vec, zero_vec, phi, 0, current_time, deltat, gradient);
      }
    }
    else if (solver_type == "transient") {
      DFAD obj = 0.0;
      //this->transientSolver(phi, gradient, initial_time, final_time);
    }
    else {
      // print out an error message
    }
    
    is_adjoint = false;
  }
  
  debugger->print("**** Finished SolverManager::adjointModel");
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::steadySolver(vector<vector_RCP> & sol) {
  
  debugger->print("**** Starting SolverManager::steadySolver ...");
  
  for (int ss=0; ss<subcycles; ++ss) {
    for (size_t set=0; set<setnames.size(); ++set) {
      assembler->updatePhysicsSet(set);
      vector<vector_RCP> zero_soln;
      if (usestrongDBCs) {
        this->setDirichlet(set, sol[set]);
      }
      params->updateDynamicParams(0);
      this->nonlinearSolver(set, 0, sol, zero_soln, zero_soln,
                            zero_soln, zero_soln, zero_soln);
    }
  }
  postproc->record(sol, current_time, 1);
  
  debugger->print("**** Finished SolverManager::steadySolver");
  
}

// ========================================================================================
/* solve a transient problem */
// ========================================================================================

template<class Node>
void SolverManager<Node>::transientSolver(vector<vector_RCP> & initial, 
                                          MrHyDE_OptVector & gradient,
                                          ScalarT & start_time, ScalarT & end_time) {
  
  Teuchos::TimeMonitor localtimer(*transientsolvertimer);
  
  debugger->print(1, "******** Starting SolverManager::transientSolver ...");
  
  vector<vector_RCP> zero_vec(initial.size());
  
  current_time = start_time;
  if (!is_adjoint) { // forward solve - adaptive time stepping
    is_final_time = false;
    vector<vector_RCP> sol = initial;
    
    if (usestrongDBCs) {
      for (size_t set=0; set<initial.size(); ++set) {
        assembler->updatePhysicsSet(set);
        this->setDirichlet(set,sol[set]);
      }
    }
    
    postproc->record(sol,current_time,0);
    
    vector<vector<vector_RCP> > sol_prev;
    for (size_t set=0; set<sol.size(); ++set) {
      vector<vector_RCP> c_prev;
      for (int step=0; step<maxnumsteps[set]; ++step) {
        c_prev.push_back(linalg->getNewOverlappedVector(set));
      }
      sol_prev.push_back(c_prev);
    }
           
    int stepProg = 0;
    int numCuts = 0;
    int maxCuts = maxTimeStepCuts; // TMW: make this a user-defined input
    double timetol = end_time*1.0e-6; // just need to get close enough to final time
    
    while (current_time < (end_time-timetol) && numCuts<=maxCuts) {
      int status = 0;
      if (Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Time Step " << stepProg+1 << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      params->updateDynamicParams(stepProg);
      assembler->updateTimeStep(stepProg);
      
      //for (int ss=0; ss<subcycles; ++ss) {
        for (size_t set=0; set<sol.size(); ++set) {
          // this needs to come first now, so that updatePhysicsSet can pick out the
          // time integration info
          if (BDForder[set] > 1 && stepProg == startupSteps[set]) {
            // Only overwrite the current set
            this->setBackwardDifference(BDForder,set);
            this->setButcherTableau(ButcherTab,set);
          }

          assembler->updatePhysicsSet(set);
    
          // if num_stages = 1, the sol_stage = sol
          //
          vector<vector_RCP> sol_stage;
          if (maxnumstages[set] == 1) {
            sol_stage.push_back(sol[set]);
          }
          else {
            for (int stage=0; stage<maxnumstages[set]; ++stage) {
              sol_stage.push_back(linalg->getNewOverlappedVector(set));
              sol_stage[stage]->assign(*(sol[set]));
            }
          }
    
          // Increment the previous step solutions (shift history and moves u into first spot)
          for (size_t step=1; step<sol_prev[set].size(); ++step) {
            size_t ind = sol_prev[set].size()-step;
            sol_prev[set][ind]->assign(*(sol_prev[set][ind-1]));
          }
          sol_prev[set][0]->assign(*(sol[set]));
      
          ////////////////////////////////////////////////////////////////////////
          // Allow the groups to change subgrid model
          ////////////////////////////////////////////////////////////////////////
          
          vector<vector<int> > sgmodels = assembler->identifySubgridModels();
          multiscale_manager->update(sgmodels);
          
          for (int stage=0; stage<numstages[set]; stage++) {
            // Need a stage solution
            // Set the initial guess for stage solution
            // sol_stage[stage]->assign(*(sol_prev[0]));
            // Updates the current time and sets the stage number in wksets
            assembler->updateStage(stage, current_time, deltat); 

            if (usestrongDBCs) {
              this->setDirichlet(set, sol_stage[stage]);
            }
  
            if (fully_explicit) {
              status += this->explicitSolver(set, stage, sol, sol_stage, sol_prev[set], 
                                             zero_vec, zero_vec, zero_vec);
            }
            else {
              status += this->nonlinearSolver(set, stage, sol, sol_stage, sol_prev[set], zero_vec, zero_vec, zero_vec);
            }

            // u_{n+1} = u_n + \sum_stage ( u_stage - u_n )
            
            // if num_stages = 1, then we might be able to skip this 
            if (maxnumstages[set] > 1) {
              sol[set]->update(1.0, *(sol_stage[stage]), 1.0);
              sol[set]->update(-1.0, *(sol_prev[set][0]), 1.0);
            }
            multiscale_manager->completeStage();
          }
        }
      //}
      
      if (status == 0) { // NL solver converged
        current_time += deltat;
        stepProg += 1;
        
        // Make sure last step solution is gathered
        // Last set of values is from a stage solution, which is potentially different
        //assembler->performGather(sol, 0, 0);
        //for (size_t set=0; set<u.size(); ++set) {
        //  assembler->updatePhysicsSet(set);
        //  assembler->performGather(set,u[set],0,0);
        //}
        multiscale_manager->completeTimeStep();
        postproc->record(sol,current_time,stepProg);
        
      }
      else { // something went wrong, cut time step and try again
        deltat *= 0.5;
        numCuts += 1;
        for (size_t set=0; set<sol.size(); ++set) {
          assembler->revertSoln(set);
          sol[set]->assign(*(sol_prev[set][0]));
        }
        if (Comm->getRank() == 0 && verbosity > 0) {
          cout << endl << endl << "*******************************************************" << endl;
          cout << endl << "**** Cutting time step to " << deltat << endl;
          cout << "**** Current time is " << current_time << endl << endl;
          cout << "*******************************************************" << endl << endl << endl;
        }
        
      }
    }
    // If the final step doesn't fall when a write is requested, catch that here  
    if (stepProg % postproc->write_frequency != 0 && postproc->write_solution) {
      postproc->writeSolution(sol, current_time);
    }
  }
  else { // adjoint solve - fixed time stepping based on forward solve
  
    current_time = final_time;
    is_final_time = true;
    
    vector<vector_RCP> sol, sol_prev, phi, phi_prev, sol_stage, phi_stage;
    for (size_t set=0; set<1; ++set) { // hard coded for now
      sol.push_back(linalg->getNewOverlappedVector(set));
      sol_prev.push_back(linalg->getNewOverlappedVector(set));
      phi.push_back(linalg->getNewOverlappedVector(set));
      phi_prev.push_back(linalg->getNewOverlappedVector(set));
      sol_stage.push_back(linalg->getNewOverlappedVector(set));
      phi_stage.push_back(linalg->getNewOverlappedVector(set));
    }
    // Transient adjoints require derivatives of Jacobians w.r.t. previous states
    // We store the Jacobian-vector products in a (Nstep x Nstep) matrix
    if (previous_adjoints.size() == 0) {
      for (size_t i=0; i<numsteps[0]; ++i) { // hard-coded for now
        vector<vector_RCP> ivecs;
        for (size_t set=0; set<setnames.size(); ++set) { // hard-coded for now
          vector_RCP tempvec = linalg->getNewVector(set);
          tempvec->putScalar(0.0);
          ivecs.push_back(tempvec);
        }
        previous_adjoints.push_back(ivecs);
      }
    }
    else {
      for (size_t i=0; i<numsteps[0]; ++i) { // hard-coded for now
        for (size_t set=0; set<setnames.size(); ++set) { // hard-coded for now
          previous_adjoints[i][set]->putScalar(0.0);
        }
      }
    }
    
    size_t set = 0, stage = 0;
    // Just getting the number of times from first physics set should be fine
    // TODO will this be affected by having physics sets with different timesteppers?
    int store_index = 0;
    size_t numFwdSteps = postproc->soln[set]->getTotalTimes(store_index)-1; 
    
    for (size_t timeiter = 0; timeiter<numFwdSteps; timeiter++) {
      size_t cindex = numFwdSteps-timeiter;
      phi_prev[set] = linalg->getNewOverlappedVector(set);
      phi_prev[set]->update(1.0,*(phi[0]),0.0);
      if (Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Adjoint Time Step " << timeiter << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      
      // TMW: this is specific to implicit Euler
      // Needs to be generalized
      // Also, need to implement checkpoint/recovery
      bool fndu = postproc->soln[set]->extract(sol[set], cindex);
      if (!fndu) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE was not able to find forward solution");
      }
      bool fndup = postproc->soln[set]->extract(sol_prev[set], cindex-1);
      if (!fndup) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE was not able to find previous forward solution");
      }
      //params->updateDynamicParams(cindex-1);
      params->updateDynamicParams(cindex-1);
      //assembler->performGather(set,u_prev[set],0,0);
      //assembler->resetPrevSoln(set);
      
      int stime_index = cindex-1;
      
      current_time = postproc->soln[set]->getSpecificTime(store_index, stime_index);
      postproc->setTimeIndex(cindex);
      assembler->updateStage(stage, current_time, deltat);
      
      sol_stage[set]->assign(*sol[set]);
      // if multistage, recover forward solution at each stage
      if (numstages[set] == 1) { // No need to re-solve in this case
        int status = this->nonlinearSolver(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev);
        if (status>0) {
          // throw error
        }
        phi[set]->update(1.0,*(phi_stage[0]),0.0);
        postproc->computeSensitivities(sol, sol_stage, sol_prev, phi, current_time, cindex, deltat, gradient);
      }
      else {
        // NEEDS TO BE REWRITTEN
      }
      
      is_final_time = false;
      
    }
    
  }
  
  debugger->print(1, "******** Finished SolverManager::transientSolver");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
int SolverManager<Node>::nonlinearSolver(const size_t & set, const size_t & stage,
                                         vector<vector_RCP> & sol, // [set]
                                         vector<vector_RCP> & sol_stage, // [stage]
                                         vector<vector_RCP> & sol_prev, // [step]
                                         vector<vector_RCP> & phi, // [set]
                                         vector<vector_RCP> & phi_stage, // [stage]
                                         vector<vector_RCP> & phi_prev) { // [step]
   // Goal is to update sol_stage[stage]
   // Assembler will need to gather sol for other physics sets and other step/stage solutions for current set

  Teuchos::TimeMonitor localtimer(*nonlinearsolvertimer);

  debugger->print(1, "******** Starting SolverManager::nonlinearSolver ...");

  int status = 0;
  int NLiter = 0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_first(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_scaled(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm(1);
  resnorm_first[0] = 10*NLtol;
  resnorm_scaled[0] = resnorm_first[0];
  resnorm[0] = resnorm_first[0];
  
  int maxiter = maxNLiter;
  if (is_adjoint) {
    maxiter = 1;//2;
  }
    
  bool proceed = true;
  ScalarT alpha = 1.0;
  
  vector_RCP current_res, current_res_over, current_du, current_du_over;
  if (store_vectors) {
    current_res = res[set];
    current_res_over = res_over[set];
    current_du = du[set];
    current_du_over = du_over[set];
  }
  else {
    current_res = linalg->getNewVector(set);
    current_res_over = linalg->getNewOverlappedVector(set);
    current_du = linalg->getNewVector(set);
    current_du_over = linalg->getNewOverlappedVector(set);
  }
  
  while (proceed) {
    
    multiscale_manager->reset();
    multiscale_manager->macro_nl_iter = NLiter;

    gNLiter = NLiter;
  
    bool build_jacobian = !linalg->getJacobianReuse(set);
    matrix_RCP J, J_over;
    
    J = linalg->getNewMatrix(set);
    if (build_jacobian) {
      J_over = linalg->getNewOverlappedMatrix(set);
      linalg->fillComplete(J_over);
      J_over->resumeFill();
      J_over->setAllToScalar(0.0);
    }
    
    // *********************** COMPUTE THE JACOBIAN AND THE RESIDUAL **************************
    
    current_res_over->putScalar(0.0);

    store_adjPrev = false; //false;
    if ( is_adjoint && (NLiter == 1)) {
      store_adjPrev = true;
    }

    bool use_autotune = true;
    if (assembler->groupData[0]->multiscale) {
      use_autotune = false;
    }

    auto paramvec = params->getDiscretizedParamsOver();
    auto paramdot = params->getDiscretizedParamsDotOver();
    
    // This is where the residual is computed for the forward problem
    // Jacobian is computed only if the residual is large enough to merit a linear solve
    // Adjoint residual is computed below
    if (!is_adjoint) {
      if (!use_autotune) {
        assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, build_jacobian, false, false, false, 0,
                                  current_res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                                  params->num_active_params, paramvec, paramdot, is_final_time, deltat);
      }
      else {
        assembler->assembleRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev,
                               paramvec, paramdot, current_res_over, J_over, isTransient, current_time, deltat);
      }
      linalg->exportVectorFromOverlapped(set, current_res, current_res_over);
    }
    
    // *********************** CHECK THE NORM OF THE RESIDUAL **************************
    
    {
      Teuchos::TimeMonitor localtimer(*normLAtimer);
      current_res->normInf(resnorm);
    }
    
    bool solve = true;
    if (NLiter == 0) {
      resnorm_first[0] = resnorm[0];
      resnorm_scaled[0] = 1.0;
    }
    else {
      resnorm_scaled[0] = resnorm[0]/resnorm_first[0];
    }
    
    // hard code these for adjoint solves since residual is computed below and only one iteration is needed
    if (is_adjoint) {
      resnorm[0] = 1.0;
      resnorm_scaled[0] = 1.0;
    }
    
    if (Comm->getRank() == 0 && verbosity > 1) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Iteration: " << NLiter << endl;
      cout << "***** Norm of nonlinear residual: " << resnorm[0] << endl;
      cout << "***** Scaled Norm of nonlinear residual: " << resnorm_scaled[0] << endl;
      cout << "*********************************************************" << endl;
    }
    
    if (!is_adjoint && allowBacktracking && resnorm_scaled[0] > 1.1) {
      solve = false;
      alpha *= 0.5;
      Teuchos::TimeMonitor localtimer(*updateLAtimer);
      if (sol_stage.size() > 0) {
        sol_stage[stage]->update(-1.0*alpha, *(current_du_over), 1.0);
      }
      else {
        sol[set]->update(-1.0*alpha, *(current_du_over), 1.0);
      }
      if (Comm->getRank() == 0 && verbosity > 1) {
        cout << "***** Backtracking: new learning rate = " << alpha << endl;
      }
    }
    else {
      if (useRelativeTOL) {
        if (resnorm_scaled[0]<NLtol) {
          solve = false;
          proceed = false;
        }
        else if (resnorm[0]<1.0e-100) { // Not sure why this is hard coded
          solve = false;
          proceed = false;
        }
      }
      else if (useAbsoluteTOL && resnorm[0]<NLabstol) {
        solve = false;
        proceed = false;
      }
    }
    if (is_adjoint) { // Always perform one linear solve
      solve = true; // force a solve
      proceed = false; // but only one
    }
    
    // *********************** SOLVE THE LINEAR SYSTEM FOR THE UPDATE **************************
    
    if (solve) {
      
      if (build_jacobian) {
        if (use_autotune) { // If false, J was already computed when the residual was computed - just saves an extra assembly
          auto paramvec = params->getDiscretizedParamsOver();
          auto paramdot = params->getDiscretizedParamsDotOver();
          assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, build_jacobian, false, false, false, 0,
                                    current_res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                                    params->num_active_params, paramvec, paramdot, is_final_time, deltat);
        }
        linalg->fillComplete(J_over);
        J->resumeFill();
        linalg->exportMatrixFromOverlapped(set, J, J_over);
        linalg->fillComplete(J);
      }
            
      // This is where the adjoint residual is computed
      if (is_adjoint) {
        // First, the derivative of the objective w.r.t. the state
        ScalarT cdt = 0.0;
        if (isTransient) {
          cdt = deltat;
        }
        postproc->computeObjectiveGradState(set, sol[set], current_time+cdt, deltat, current_res);
        
        // We use a true adjoint residual, so we need to Jacobian^T times the current approximation
        
        auto mvprod = linalg->getNewVector(set);
        auto phi_owned = linalg->getNewVector(set);
        linalg->exportVectorFromOverlappedReplace(set, phi_owned, phi_stage[set]);
        J->apply(*phi_owned,*mvprod);
        current_res->update(-1.0, *mvprod, 1.0);
        
        
        // For transient problems, need to update adjoint residual with previous adjoint solutions (multi-step)
        // This is actually a little complicated
        // The jacobians need to be evaluated at the right time and using the right forward solution
        // The Jacobian-vector products we need on this time step should already be computed
        if (isTransient) {
          
          // use the stored Jacobian vector (Jv) products from previous time steps
          for (size_t istep=0; istep<previous_adjoints.size(); ++istep) {
            current_res->update(-1.0, *(previous_adjoints[istep][istep]), 1.0);
          }
          
          // Increment the Jv products
          size_t numSteps = sol_prev.size();
          for (size_t istep=0; istep<numSteps-1; ++istep) {
            for (size_t kstep=0; kstep<numSteps; ++kstep) {
              previous_adjoints[kstep][numSteps-istep] = previous_adjoints[kstep][numSteps-istep-1];
            }
          }
          
          // The next set of Jacobian vector products are calculated below once phi is updated
          
        }
        {
          Teuchos::TimeMonitor localtimer(*normLAtimer);
          current_res->normInf(resnorm);
        }
        
      }
      
      //******************************************************
      // Actual linear solve
      //******************************************************
      
      current_du->putScalar(0.0);
      current_du_over->putScalar(0.0);
      linalg->linearSolver(set, J, current_res, current_du);
      
      // doesn't always write to file - only if requested
      if (is_adjoint) {
        linalg->writeToFile(J, current_res, current_du, "adjoint_jacobian.mm",
                            "adjoint_residual.mm","adjoint_solution.mm");
      }
      else {
        linalg->writeToFile(J, current_res, current_du);
      }
      linalg->importVectorToOverlapped(set, current_du_over, current_du);
      
      alpha = 1.0; // what is the point of alpha?
      if (is_adjoint) {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        if (phi_stage.size() > 0) {
          phi_stage[stage]->update(alpha, *(current_du_over), 1.0);
        }
        else {
          phi[set]->update(alpha, *(current_du_over), 1.0);
        }
        
        if (isTransient) {
          // Fill in prev_mass_adjoints[:][0] - need to create new vectors since these are RCPs
          vector<matrix_RCP> Jprev = linalg->getNewPreviousMatrix(set, phi_prev.size());
          for (size_t step=0; step<phi_prev.size(); ++step) {
            
            if (build_jacobian) {
              auto paramvec = params->getDiscretizedParamsOver();
              auto paramdot = params->getDiscretizedParamsDotOver();
              matrix_RCP currJ = Jprev[step];
              matrix_RCP currJ_over = linalg->getNewOverlappedMatrix(set);
              linalg->fillComplete(currJ_over);
              currJ_over->resumeFill();
              currJ_over->setAllToScalar(0.0);
              auto dummy_res_over = linalg->getNewOverlappedVector(set);
              assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, build_jacobian, false, false, true, step,
                                        dummy_res_over, currJ_over, isTransient, current_time, is_adjoint, store_adjPrev,
                                        params->num_active_params, paramvec, paramdot, is_final_time, deltat);
              linalg->fillComplete(currJ_over);
              currJ->resumeFill();
              linalg->exportMatrixFromOverlapped(set, currJ, currJ_over);
              linalg->fillComplete(currJ);
            }
            
            auto mvprod = linalg->getNewVector(set);
            auto phip_owned = linalg->getNewVector(set);
            
            if (step == 0) {
              if (phi_stage.size() > 0) {
                linalg->exportVectorFromOverlappedReplace(set, phip_owned, phi_stage[stage]);
              }
              else {
                linalg->exportVectorFromOverlappedReplace(set, phip_owned, phi[set]);
              }
            }
            else {
              linalg->exportVectorFromOverlappedReplace(set, phip_owned, phi_prev[step-1]);
            }
            
            Jprev[step]->apply(*phip_owned,*mvprod);
            previous_adjoints[0][0]->update(1.0, *mvprod, 0.0);
            
          }
        }
      }
      else {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        if (sol_stage.size() > 0) {
          sol_stage[stage]->update(alpha, *(current_du_over), 1.0);
        }
        else {
          sol[set]->update(alpha, *(current_du_over), 1.0);
        }
      }
    }
    NLiter++; // increment number of nonlinear iterations
    
    if (is_adjoint) {
      //proceed = false;
    }
    else if (NLiter >= maxiter) {
      proceed = false;
    }
  } // while loop
  
  if (verbosity>1) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> normu(1);
    if (sol_stage.size() > 0) {
      sol_stage[stage]->norm2(normu);
    }
    else {
      sol[set]->norm2(normu);
    }
    if (Comm->getRank() == 0) {
      cout << "Norm of solution: " << normu[0] << "    (overlapped vector so results may differ on multiple procs)" << endl;
    }
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
  
  debugger->print(1, "******** Finished SolverManager::nonlinearSolver");
  
  return status;
}

// ========================================================================================
// ========================================================================================

template<class Node>
int SolverManager<Node>::explicitSolver(const size_t & set, const size_t & stage,
                                        vector<vector_RCP> & sol, // [set]
                                        vector<vector_RCP> & sol_stage, // [stage]
                                        vector<vector_RCP> & sol_prev, // [step]
                                        vector<vector_RCP> & phi, // [set]
                                        vector<vector_RCP> & phi_stage, // [stage]
                                        vector<vector_RCP> & phi_prev) { // [step]
  
  
  // Goal is just to update sol_stage[stage]
  // Other solutions are just used in assembler for gather operations

  Teuchos::TimeMonitor localtimer(*explicitsolvertimer);
  
  debugger->print(1, "******** Starting SolverManager::explicitSolver ...");
  
  int status = 0;
  assembler->updatePhysicsSet(set);
  
  if (usestrongDBCs) {
    this->setDirichlet(set,sol_stage[stage]);
  }
  
  vector_RCP current_res, current_res_over;
  if (store_vectors) {
    current_res = res[set];
    current_res_over = res_over[set];
  }
  else {
    current_res = linalg->getNewVector(set);
    if (linalg->getHaveOverlapped()) {
      current_res_over = linalg->getNewOverlappedVector(set);
    }
    else {
      current_res_over = current_res;
    }
  }

  // *********************** COMPUTE THE RESIDUAL **************************
    
  current_res_over->putScalar(0.0);
  matrix_RCP J_over;
  
  auto paramvec = params->getDiscretizedParamsOver();
  auto paramdot = params->getDiscretizedParamsDotOver();
  assembler->assembleRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev,
                         paramvec, paramdot, current_res_over, J_over, isTransient, current_time, deltat);
  
  if (linalg->getHaveOverlapped()) {
    linalg->exportVectorFromOverlapped(set, current_res, current_res_over);
  }

  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  current_res->norm2(rnorm);
  //KokkosTools::print(current_res);
  
  // *********************** SOLVE THE LINEAR SYSTEM **************************
  
  if (rnorm[0]>1.0e-100) {
    // Given m = diag(M^-1)
    // Given timewt = b(stage)*deltat
    // Compute du = timewt*m*res
    // Compute u += du
    
    ScalarT wt = deltat*butcher_b(stage);
    
    
    if (!assembler->lump_mass) {
      vector_RCP current_du, current_du_over;
      if (store_vectors) {
        current_du = du[set];
        current_du_over = du_over[set];
      }
      else {
        current_du = linalg->getNewVector(set);
        if (linalg->getHaveOverlapped()) {
          current_du_over = linalg->getNewOverlappedVector(set);
        }
        else {
          current_du_over = current_du;
        }
      }

      current_du_over->putScalar(0.0);
      if (linalg->getHaveOverlapped()) {
        current_du->putScalar(0.0);
      }

      current_res->scale(wt);
      if (assembler->matrix_free) {
        this->matrixFreePCG(set, current_res, current_du, diagMass[set],
                            settings->sublist("Solver").get("linear TOL",1.0e-2),
                            settings->sublist("Solver").get("max linear iters",100));
      }
      else {
        if (use_custom_PCG) {
          this->PCG(set, explicitMass[set], current_res, current_du, diagMass[set],
                    settings->sublist("Solver").get("linear TOL",1.0e-2),
                    settings->sublist("Solver").get("max linear iters",100));
        }
        else {
          linalg->linearSolverL2(set, explicitMass[set], current_res, current_du);
        }
      }
      if (linalg->getHaveOverlapped()) {
        linalg->importVectorToOverlapped(set, current_du_over, current_du);
      }
      
      sol_stage[stage]->update(1.0, *(current_du_over), 1.0);
      
    }
    else {
      typedef typename Node::execution_space LA_exec;
      
      // can probably avoid du in this case
      // sol += sol + wt*res/dm
      vector_RCP current_sol;
      if (linalg->getHaveOverlapped()) {
        current_sol = linalg->getNewVector(set);
        linalg->exportVectorFromOverlappedReplace(set, current_sol, sol_stage[stage]);
      }
      else {
        current_sol = sol_stage[stage];
      }

      //auto du_view = current_du->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto sol_view = current_sol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto res_view = current_res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto dm_view = diagMass[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      
      parallel_for("explicit solver apply invdiag",
                   RangePolicy<LA_exec>(0,sol_view.extent(0)),
                   KOKKOS_LAMBDA (const int k ) {
        sol_view(k,0) += wt*res_view(k,0)/dm_view(k,0);
      });

      if (linalg->getHaveOverlapped()) {
        linalg->importVectorToOverlapped(set, sol_stage[stage], current_sol);
      }
    }
    
  }
  
  if (verbosity>=10) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> unorm(1);
    sol_stage[stage]->norm2(unorm);
    if (Comm->getRank() == 0) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Explicit integrator: L2 norm of (overlapped/ghosted) solution: " << unorm[0] << endl;
      cout << "*********************************************************" << endl;
    }
  }
  
  debugger->print(1, "******** Finished SolverManager::explicitSolver");
  
  return status;
}


// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setDirichlet(const size_t & set, vector_RCP & u) {
  
  debugger->print("**** Starting SolverManager::setDirichlet ...");
  
  Teuchos::TimeMonitor localtimer(*dbcsettimer);
  
  typedef typename Node::execution_space LA_exec;
  
  if (usestrongDBCs) {
    auto u_kv = u->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    //auto meas_kv = meas->getLocalView<HostDevice>();
    
    if (!scalarDirichletData[set]) {
      if (!staticDirichletData[set]) {
        this->projectDirichlet(set);
      }
      else if (!have_static_Dirichlet_data[set]) {
        this->projectDirichlet(set);
        have_static_Dirichlet_data[set] = true;
      }
    }
    
    //if (!scalarDirichletData && transientDirichletData) {
    //  this->projectDirichlet();
    //}
    
    vector<vector<Kokkos::View<LO*,LA_device> > > dbcDOFs = assembler->fixedDOF[set];
    if (scalarDirichletData[set]) {
      
      for (size_t block=0; block<dbcDOFs.size(); ++block) {
        for (size_t v=0; v<dbcDOFs[block].size(); v++) {
          if (dbcDOFs[block][v].extent(0)>0) {
            ScalarT value = scalarDirichletValues[set][block][v];
            auto cdofs = dbcDOFs[block][v];
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
      auto dbc_kv = fixedDOF_soln[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      for (size_t block=0; block<dbcDOFs.size(); ++block) {
        for (size_t v=0; v<dbcDOFs[block].size(); v++) {
          if (dbcDOFs[block][v].extent(0)>0) {
            auto cdofs = dbcDOFs[block][v];
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
    vector<vector<GO> > pointDOFs = disc->point_dofs[set];
    for (size_t block=0; block<blocknames.size(); ++block) {
      vector<GO> pt_dofs = pointDOFs[block];
      Kokkos::View<LO*,LA_device> ptdofs("pointwise dofs", pointDOFs[block].size());
      auto ptdofs_host = Kokkos::create_mirror_view(ptdofs);
      for (size_t i = 0; i < pt_dofs.size(); i++) {
        LO row = linalg->overlapped_map[set]->getLocalElement(pt_dofs[i]); // TMW: this is a temporary fix
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
  
  debugger->print("**** Finished SolverManager::setDirichlet");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > SolverManager<Node>::setInitialParams() {
  vector_RCP initial = linalg->getNewOverlappedParamVector();
  ScalarT value = 2.0;
  initial->putScalar(value);
  return initial;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > SolverManager<Node>::setInitial() {
  
  Teuchos::TimeMonitor localtimer(*initsettimer);
  typedef typename Node::execution_space LA_exec;
  
  debugger->print("**** Starting SolverManager::setInitial ...");
  
  vector<vector_RCP> initial_solns;
  
  if (use_restart) {
    for (size_t set=0; set<restart_solution.size(); ++set) {
      initial_solns.push_back(restart_solution[set]);
    }
  }
  else {

    for (size_t set=0; set<setnames.size(); ++set) {
      assembler->updatePhysicsSet(set);
      
      vector_RCP initial = linalg->getNewOverlappedVector(set);
      initial->putScalar(0.0);
      
      bool samedevice = true;
      bool usehost = false;
      if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyMem>::accessible) {
        samedevice = false;
        if (!Kokkos::SpaceAccessibility<LA_exec, HostMem>::accessible) {
          usehost = true;
        }
        else {
          // output an error
        }
      }
      
      if (have_initial_conditions[set]) {
        if (scalarInitialData[set]) {
          
          auto initial_kv = initial->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
          
          for (size_t block=0; block<assembler->groupData.size(); block++) {
            
            assembler->updatePhysicsSet(set);
            
            if (assembler->groupData[block]->num_elem > 0) {
              
              Kokkos::View<ScalarT*,LA_device> idata("scalar initial data",scalarInitialValues[set][block].size());
              auto idata_host = Kokkos::create_mirror_view(idata);
              for (size_t i=0; i<scalarInitialValues[set][block].size(); i++) {
                idata_host(i) = scalarInitialValues[set][block][i];
              }
              Kokkos::deep_copy(idata,idata_host);
              
              if (samedevice) {
                auto offsets = assembler->wkset[block]->offsets;
                auto numDOF = assembler->groupData[block]->num_dof;
                for (size_t cell=0; cell<assembler->groups[block].size(); cell++) {
                  auto LIDs = assembler->groups[block][cell]->LIDs[set];
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
                auto numDOF = assembler->groupData[block]->num_dof_host;
                for (size_t cell=0; cell<assembler->groups[block].size(); cell++) {
                  auto LIDs = assembler->groups[block][cell]->LIDs_host[set];
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
          
          vector_RCP glinitial = linalg->getNewVector(set);
          
          if (initial_type == "L2-projection") {
            // Compute the L2 projection of the initial data into the discrete space
            vector_RCP rhs = linalg->getNewOverlappedVector(set);
            matrix_RCP mass = linalg->getNewOverlappedMatrix(set);
            vector_RCP glrhs = linalg->getNewVector(set);
            matrix_RCP glmass = linalg->getNewMatrix(set);
            
            assembler->setInitial(set, rhs, mass, is_adjoint);
            
            linalg->exportMatrixFromOverlapped(set, glmass, mass);
            linalg->exportVectorFromOverlapped(set, glrhs, rhs);
            
            linalg->fillComplete(glmass);
            linalg->linearSolverL2(set, glmass, glrhs, glinitial);
            linalg->importVectorToOverlapped(set, initial, glinitial);
            linalg->resetJacobian(set);
          }
          else if (initial_type == "L2-projection-HFACE") {
            // Similar to above, but the basis support only exists on the mesh skeleton
            // The use case is setting the IC at the coarse-scale
            vector_RCP rhs = linalg->getNewOverlappedVector(set);
            matrix_RCP mass = linalg->getNewOverlappedMatrix(set);
            vector_RCP glrhs = linalg->getNewVector(set);
            matrix_RCP glmass = linalg->getNewMatrix(set);
            
            assembler->setInitialFace(set, rhs, mass, is_adjoint);
            
            linalg->exportMatrixFromOverlapped(set, glmass, mass);
            linalg->exportVectorFromOverlapped(set, glrhs, rhs);
            linalg->fillComplete(glmass);
            
            // With HFACE we ensure the preconditioner is not
            // used for this projection (mass matrix is nearly the identity
            // and can cause issues)
            auto origPreconFlag = linalg->options_L2[set]->use_preconditioner;
            linalg->options_L2[set]->use_preconditioner = false;
            // do the solve
            linalg->linearSolverL2(set, glmass, glrhs, glinitial);
            // set back to original
            linalg->options_L2[set]->use_preconditioner = origPreconFlag;
            
            linalg->importVectorToOverlapped(set, initial, glinitial);
            linalg->resetJacobian(set); // TODO not sure of this
            
          }
          else if (initial_type == "interpolation") {
            
            assembler->setInitial(set, initial, is_adjoint);
            
          }
        }
      }
      
      initial_solns.push_back(initial);
    }
  }
  
  debugger->print("**** Finished SolverManager::setInitial ...");
  
  
  return initial_solns;
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
  size_t set = 0; // hard coded since somebody uses this
  vector_RCP F_soln = linalg->getNewOverlappedVector(set);
  return F_soln;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > SolverManager<Node>::getRestartSolution() {
  
  if (restart_solution.size() == 0) {
    for (size_t set=0; set<setnames.size(); ++set) {
      vector_RCP F_soln = linalg->getNewOverlappedVector(set);
      restart_solution.push_back(F_soln);
    }
  }
  return restart_solution;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > SolverManager<Node>::getRestartAdjointSolution() {
  
  if (restart_adjoint_solution.size() == 0) {
    for (size_t set=0; set<setnames.size(); ++set) {
      vector_RCP F_soln = linalg->getNewOverlappedVector(set);
      restart_adjoint_solution.push_back(F_soln);
    }
  }
  return restart_adjoint_solution;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeParams() {
  
  //for (size_t block=0; block<blocknames.size(); ++block) {
  //  assembler->wkset[block]->paramusebasis = params->discretized_param_usebasis;
  //  assembler->wkset[block]->paramoffsets = params->paramoffsets[0];
  // }
  
}

////////////////////////////////////////////////////////////////////////////////
// The following function is not updated for multi-set
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void SolverManager<Node>::finalizeMultiscale() {
#ifndef MrHyDE_NO_AD
  if (multiscale_manager->subgridModels.size() > 0 ) {
    for (size_t k=0; k<multiscale_manager->subgridModels.size(); k++) {
      multiscale_manager->subgridModels[k]->paramvals_KVAD = params->paramvals_KVAD;
    }
    
    multiscale_manager->macro_wkset = assembler->wkset_AD;
    vector<Kokkos::View<int*,AssemblyDevice>> macro_numDOF;
    for (size_t block=0; block<assembler->groupData.size(); ++block) {
      macro_numDOF.push_back(assembler->groupData[block]->set_num_dof[0]);
    }
    multiscale_manager->setMacroInfo(disc->basis_pointers, disc->basis_types,
                                     physics->var_list[0], useBasis[0], disc->offsets[0],
                                     macro_numDOF, params->paramnames, params->discretized_param_names);
    
    vector<vector<int> > sgmodels = assembler->identifySubgridModels();
    ScalarT my_cost = multiscale_manager->initialize(sgmodels);
    ScalarT gmin = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
    ScalarT gmax = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);
    
    assembler->multiscale_manager = multiscale_manager;
    if (Comm->getRank() == 0 && verbosity>0) {
      cout << "***** Load Balancing Factor " << gmax/gmin <<  endl;
    }
    
  }
#endif  
}


// ========================================================================================
// Specialized PCG
// ========================================================================================

template<class Node>
void SolverManager<Node>::PCG(const size_t & set, matrix_RCP & J, vector_RCP & b, vector_RCP & x,
                              vector_RCP & M, const ScalarT & tol, const int & maxiter) {
  
  Teuchos::TimeMonitor localtimer(*PCGtimer);
  
  typedef typename Node::execution_space LA_exec;
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);
  
  ScalarT rho = 1.0, rho1 = 0.0, alpha = 0.0, beta = 1.0, pq = 0.0;
  ScalarT one = 1.0, zero = 0.0;
  
  vector_RCP p, q, r, z;
  if (store_vectors) {
    p = p_pcg[set];
    q = q_pcg[set];
    r = r_pcg[set];
    z = z_pcg[set];
  }
  else {
    p = linalg->getNewVector(set);
    q = linalg->getNewVector(set);
    r = linalg->getNewVector(set);
    z = linalg->getNewVector(set);
  }
  
  p->putScalar(zero);
  q->putScalar(zero);
  r->putScalar(zero);
  z->putScalar(zero);
  
  int iter=0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  {
    Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
    J->apply(*x,*q);
  }
  
  r->assign(*b);
  r->update(-one,*q,one);
  
  r->norm2(rnorm);
  ScalarT r0 = rnorm[0];
  
  auto M_view = M->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto r_view = r->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto z_view = z->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  while (iter<maxiter && rnorm[0]/r0>tol) {
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyPrectimer);
      parallel_for("PCG apply prec",
                   RangePolicy<LA_exec>(0,z_view.extent(0)),
                   KOKKOS_LAMBDA (const int k ) {
        z_view(k,0) = r_view(k,0)/M_view(k,0);
      });
    }
    
    rho1 = rho;
    r->dot(*z, dotprod);
    rho = dotprod[0];
    if (iter == 0) {
      p->assign(*z);
    }
    else {
      beta = rho/rho1;
      p->update(one,*z,beta);
    }
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
      J->apply(*p,*q);
    }
    
    p->dot(*q,dotprod);
    pq = dotprod[0];
    alpha = rho/pq;
    
    x->update(alpha,*p,one);
    r->update(-one*alpha,*q,one);
    r->norm2(rnorm);
    
    iter++;
  }
  if (verbosity >= 10 && Comm->getRank() == 0) {
    cout << " ******* PCG Convergence Information: " << endl;
    cout << " *******     Iter: " << iter << "   " << "rnorm = " << rnorm[0]/r0 << endl;
  }
}

// ========================================================================================
// Specialized matrix-free PCG
// ========================================================================================

template<class Node>
void SolverManager<Node>::matrixFreePCG(const size_t & set, vector_RCP & b, vector_RCP & x,
                                        vector_RCP & M, const ScalarT & tol, const int & maxiter) {
  
  Teuchos::TimeMonitor localtimer(*PCGtimer);
  
  typedef typename Node::execution_space LA_exec;
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);
  
  ScalarT rho = 1.0, rho1 = 0.0, alpha = 0.0, beta = 1.0, pq = 0.0;
  ScalarT one = 1.0, zero = 0.0;
  
  vector_RCP p, q, r, z, p_over, q_over;
  if (store_vectors) {
    p = p_pcg[set];
    q = q_pcg[set];
    r = r_pcg[set];
    z = z_pcg[set];
    p_over = p_pcg_over[set];
    q_over = q_pcg_over[set];
  }
  else {
    p = linalg->getNewVector(set);
    q = linalg->getNewVector(set);
    r = linalg->getNewVector(set);
    z = linalg->getNewVector(set);
    if (linalg->getHaveOverlapped()) {
      p_over = linalg->getNewOverlappedVector(set);
      q_over = linalg->getNewOverlappedVector(set);
    }
    else {
      p_over = p;
      q_over = q;
    }
  }
   
  p->putScalar(zero);
  q->putScalar(zero);
  r->putScalar(zero);
  z->putScalar(zero);
  
  if (linalg->getHaveOverlapped()) {
    p_over->putScalar(zero);
    q_over->putScalar(zero);
  }

  int iter=0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  
  {
    Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
    if (linalg->getHaveOverlapped()) {
      linalg->importVectorToOverlapped(set, p_over, x);
    }
    else {
      p_over->assign(*x);
    }
    assembler->applyMassMatrixFree(set, p_over, q_over);
    if (linalg->getHaveOverlapped()) {
      linalg->exportVectorFromOverlapped(set, q, q_over);
    }
  }
  
  r->assign(*b);
  r->update(-one,*q,one);
  
  r->norm2(rnorm);
  ScalarT r0 = rnorm[0];
  
  auto M_view = M->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto r_view = r->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto z_view = z->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  while (iter<maxiter && rnorm[0]/r0>tol) {
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyPrectimer);
      parallel_for("PCG apply prec",
                   RangePolicy<LA_exec>(0,z_view.extent(0)),
                   KOKKOS_LAMBDA (const int k ) {
        z_view(k,0) = r_view(k,0)/M_view(k,0);
      });
    }
    
    rho1 = rho;
    r->dot(*z, dotprod);
    rho = dotprod[0];
    if (iter == 0) {
      p->assign(*z);
    }
    else {
      beta = rho/rho1;
      p->update(one,*z,beta);
    }
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
      if (linalg->getHaveOverlapped()) {
        linalg->importVectorToOverlapped(set, p_over, p);
      }
      q_over->putScalar(zero);
      assembler->applyMassMatrixFree(set, p_over, q_over);
      if (linalg->getHaveOverlapped()) {
        linalg->exportVectorFromOverlapped(set, q, q_over);
      }
    }
    
    p->dot(*q,dotprod);
    pq = dotprod[0];
    alpha = rho/pq;
    
    x->update(alpha,*p,one);
    r->update(-one*alpha,*q,one);
    r->norm2(rnorm);
    
    iter++;
  }
  if (verbosity >= 10 && Comm->getRank() == 0) {
    cout << " ******* PCG Convergence Information: " << endl;
    cout << " *******     Iter: " << iter << "   " << "rnorm = " << rnorm[0]/r0 << endl;
  }
}

// Explicit template instantiations
template class MrHyDE::SolverManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
template class MrHyDE::SolverManager<SubgridSolverNode>;
#endif
