/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

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
