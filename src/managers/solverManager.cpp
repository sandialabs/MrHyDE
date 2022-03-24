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
  setnames = phys->setnames;
  store_vectors = true;
  if (setnames.size() > 1) {
    store_vectors = false;
  }
  // Get the required information from the settings
  spaceDim = phys->spaceDim;//mesh->stk_mesh->getDimension();
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
  
  maxTimeStepCuts = settings->sublist("Solver").get<int>("maximum time step cuts",5);
  amplification_factor = settings->sublist("Solver").get<double>("explicit amplification factor",10.0);
  
  // TODO REMOVE ME
//  // The Butcher tableau and BDF coefficients can vary by physics set
//  // If they are universal, we get the values here.
//  // If set-specific values are supplied later, they are overwritten
//  ButcherTab = settings->sublist("Solver").get<string>("transient Butcher tableau","BWE");
//  BDForder = settings->sublist("Solver").get<int>("transient BDF order",1);
//  if (BDForder>1) {
//    if (ButcherTab == "custom") {
//      cout << "Warning: running a higher order BDF method with anything other than BWE/DIRK-1,1 is risky." << endl;
//      cout << "The code will run, but the results may be nonsense" << endl;
//    }
//    else {
//      if (ButcherTab != "BWE" && ButcherTab != "DIRK-1,1") {
//        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: need to use BWE or DIRK-1,1 with higher order BDF");
//      }
//    }
//  }
//  
//  // Additional parameters for higher-order BDF methods that require some startup procedure
//  startupButcherTab = settings->sublist("Solver").get<string>("transient startup Butcher tableau",ButcherTab);
//  startupBDForder = settings->sublist("Solver").get<int>("transient startup BDF order",BDForder);
//  startupSteps = settings->sublist("Solver").get<int>("transient startup steps",BDForder);
//  if (startupBDForder>1) {
//    if (startupButcherTab == "custom") {
//      cout << "Warning: running a higher order BDF method with anything other than BWE/DIRK-1,1 is risky." << endl;
//      cout << "The code will run, but the results may be nonsense" << endl;
//    }
//    else {
//      if (startupButcherTab != "BWE" && startupButcherTab != "DIRK-1,1") {
//        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: need to use BWE or DIRK-1,1 with higher order BDF");
//      }
//    }
//  }
  
  line_search = false;//settings->sublist("Solver").get<bool>("Use Line Search","false");
  store_adjPrev = false;
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  if (!isTransient) {
    deltat = 1.0;
  }
  // TODO should this also be an option?
  fully_explicit = settings->sublist("Solver").get<bool>("fully explicit",false);
  
  if (fully_explicit && Comm->getRank() == 0) {
    cout << "WARNING: the fully explicit method is requested.  This is an experimental capability and may not work with all time integration methods" << endl;
  }
  
  initial_type = settings->sublist("Solver").get<string>("initial type","L2-projection");
  
  // needed information from the mesh
  blocknames = phys->blocknames;
  //mesh->stk_mesh->getElementBlockNames(blocknames);
  
  // needed information from the physics interface
  numVars = phys->numVars; //
  vector<vector<vector<string> > > phys_varlist = phys->varlist;
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
        int vub = phys->getUniqueIndex(set,block,var);
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
    auto setSolverSettings = phys->setSolverSettings[set][0]; // [set][block]

    if (setSolverSettings.isSublist(setnames[set])) {
      myButcherTab = 
        setSolverSettings.sublist(setnames[set]).get<string>("transient Butcher tableau",myButcherTab);
      myBDForder = 
        setSolverSettings.sublist(setnames[set]).get<int>("transient BDF order",1);
      myStartupButcherTab = 
        setSolverSettings.sublist(setnames[set]).get<string>("transient startup Butcher tableau",myStartupButcherTab);
      myStartupBDForder = 
        setSolverSettings.sublist(setnames[set]).get<int>("transient startup BDF order",myStartupBDForder);
      myStartupSteps = 
        setSolverSettings.sublist(setnames[set]).get<int>("transient startup steps",myStartupSteps);
    }

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
  
  linalg = Teuchos::rcp( new LinearAlgebraInterface<Node>(Comm, settings, disc, params) );
    
  /////////////////////////////////////////////////////////////////////////////
  // Worksets
  /////////////////////////////////////////////////////////////////////////////
  
  assembler->createWorkset();
  
  // initialize vector which holds the total BDF steps and RK stages for each set 
  numsteps.resize(numSets,0);
  numstages.resize(numSets,0);

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
  
  phys->setWorkset(assembler->wkset);
  params->wkset = assembler->wkset;
  
  if (store_vectors) {
    for (size_t set=0; set<numSets; ++set) {
      res.push_back(linalg->getNewVector(set));
      res_over.push_back(linalg->getNewOverlappedVector(set));
      du_over.push_back(linalg->getNewOverlappedVector(set));
      du.push_back(linalg->getNewVector(set));
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  
  this->setBatchID(Comm->getRank());
  
  /////////////////////////////////////////////////////////////////////////////
  
  this->setupFixedDOFs(settings);
  
  /////////////////////////////////////////////////////////////////////////////
  
  have_initial_conditions = vector<bool>(numSets,false);
  scalarInitialData = vector<bool>(numSets,false);
  have_static_Dirichlet_data = vector<bool>(numSets,false);
  
  for (size_t set=0; set<numSets; ++set) {
    have_initial_conditions[set] = false;
    for (size_t block=0; block<blocknames.size(); ++block) {
      if (phys->setPhysSettings[set][block].isSublist("Initial conditions")) {
        have_initial_conditions[set] = true;
        scalarInitialData[set] = phys->setPhysSettings[set][block].sublist("Initial conditions").get<bool>("scalar data", false);
      }
    }
  
    if (have_initial_conditions[set] && scalarInitialData[set]) {
      vector<vector<ScalarT> > setInitialValues;
      for (size_t block=0; block<blocknames.size(); ++block) {
        
        std::string blockID = blocknames[block];
        Teuchos::ParameterList init_settings = phys->setPhysSettings[set][block].sublist("Initial conditions");
        
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
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager constructor" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::completeSetup() {

  //---------------------------------------------------
  // Mass matrix (lumped and maybe full) for explicit
  //---------------------------------------------------
  
  if (fully_explicit) {
    this->setupExplicitMass();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setupExplicitMass() {

  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::setupExplicitMass()" << endl;
    }
  }
  
  bool compute_matrix = true;
  if (assembler->lump_mass || assembler->matrix_free) {
    compute_matrix = false;
  }
  
  for (size_t set=0; set<useBasis.size(); ++set) {
    matrix_RCP mass;
    
    assembler->updatePhysicsSet(set);
    if (compute_matrix) {
      
      typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
      typedef Tpetra::CrsGraph<LO,GO,Node>            LA_CrsGraph;
      
      vector<size_t> maxEntriesPerRow(linalg->overlapped_map[set]->getNodeNumElements(), 0);
      for (size_t block=0; block<assembler->groups.size(); ++block) {
        auto offsets = assembler->wkset[block]->offsets;
        auto numDOF = assembler->groupData[block]->numDOF_host;
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto LIDs = assembler->groups[block][grp]->LIDs_host[set];
          
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
      
      Teuchos::RCP<LA_CrsGraph> overlapped_graph = Teuchos::rcp(new LA_CrsGraph(linalg->overlapped_map[set],
                                                                                maxEntriesPerRow));
      
      for (size_t block=0; block<assembler->groups.size(); ++block) {
        auto offsets = assembler->wkset[block]->offsets;
        auto numDOF = assembler->groupData[block]->numDOF;
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto LIDs = assembler->groups[block][grp]->LIDs_host[set];
          
          parallel_for("assembly insert Jac",
                       RangePolicy<HostExec>(0,LIDs.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                vector<GO> cols;
                int row = offsets(n,j);
                GO rowIndex = linalg->overlapped_map[set]->getGlobalElement(LIDs(elem,row));
                for (int k=0; k<numDOF(n); k++) {
                  int col = offsets(n,k);
                  GO gcol = linalg->overlapped_map[set]->getGlobalElement(LIDs(elem,col));
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
      //disc->DOF[set]->getOwnedIndices(owned);
      disc->DOF_owned[set];//->getOwnedIndices(owned);
      vector<size_t> maxOwnedEntriesPerRow(linalg->owned_map[set]->getNodeNumElements(), 0);
      for (size_t i=0; i<owned.size(); ++i) {
        LO ind1 = linalg->overlapped_map[set]->getLocalElement(owned[i]);
        LO ind2 = linalg->owned_map[set]->getLocalElement(owned[i]);
        maxOwnedEntriesPerRow[ind2] = maxEntriesPerRow[ind1];
      }
      
      explicitMass.push_back(Teuchos::rcp(new LA_CrsMatrix(linalg->owned_map[set],
                                                           maxOwnedEntriesPerRow)));
      
      mass = Teuchos::rcp(new LA_CrsMatrix(overlapped_graph));
    }
    
    diagMass.push_back(linalg->getNewVector(set));
    vector_RCP diagMass_over = linalg->getNewOverlappedVector(set);
    
    assembler->getWeightedMass(set,mass,diagMass_over);
    
    linalg->exportVectorFromOverlapped(set,diagMass[set], diagMass_over);
    if (compute_matrix) {
      linalg->exportMatrixFromOverlapped(set,explicitMass[set], mass);
    }
    //if (debug_level > 2) {
    //  KokkosTools::print(diagMass[set]);
    //}
  }

  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::setupExplicitMass() - fillComplete" << endl;
    }
  }
  
  for (size_t set=0; set<useBasis.size(); ++set) {
    
    if (compute_matrix) {
      linalg->fillComplete(explicitMass[set]);
    }
    
    if (store_vectors) {
      q_pcg.push_back(linalg->getNewVector(set));
      z_pcg.push_back(linalg->getNewVector(set));
      p_pcg.push_back(linalg->getNewVector(set));
      r_pcg.push_back(linalg->getNewVector(set));
      if (assembler->matrix_free) {
        q_pcg_over.push_back(linalg->getNewOverlappedVector(set));
        p_pcg_over.push_back(linalg->getNewOverlappedVector(set));
      }
    }
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager::setupExplicitMass()" << endl;
    }
  }
  
}

//========================================================================
//========================================================================

template<class Node>
void SolverManager<Node>::setButcherTableau(const vector<string> & tableau, const int & set) {

  for (size_t block=0; block<assembler->groups.size(); ++block) {

    // TODO removing this for now
    // If the workset on this block is not used (due to not owning any groups)
    // don't do anything
    //if ( !(assembler->wkset[block]->isInitialized) ) continue; 

    // Gather RK weights for this block before assigning them to the workset
    //Kokkos::View<ScalarT**,AssemblyDevice> block_butcher_A; 
    //Kokkos::View<ScalarT*,AssemblyDevice>  block_butcher_b, block_butcher_c;
    // TODO the RK scheme cannot be specified block by block
    // currently handled like before with member vars

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

    // TODO HERE need to figure out how to modify and what is going on
  
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
    // TODO same here?? why?
    numstages[set] = std::max(numstages[set],newnumstages);
  
    assembler->wkset[block]->set_butcher_A[set] = dev_butcher_A;//block_butcher_A;
    assembler->wkset[block]->set_butcher_b[set] = dev_butcher_b;//block_butcher_b;
    assembler->wkset[block]->set_butcher_c[set] = dev_butcher_c;//block_butcher_c;

    // TODO dont like this... but should protect against 1 set errors
    assembler->wkset[block]->butcher_A = dev_butcher_A;
    assembler->wkset[block]->butcher_b = dev_butcher_b;
    assembler->wkset[block]->butcher_c = dev_butcher_c;

  } // end for blocks
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setBackwardDifference(const vector<int> & order, const int & set) { // using order as an input to allow for dynamic changes

  // TODO rearrange this? and setButcher...

  for (size_t block=0; block<assembler->groups.size(); ++block) {

    // TODO removing this for now...
    // If the workset on this block is not used (due to not owning any groups)
    // don't do anything
    //if ( !(assembler->wkset[block]->isInitialized) ) continue; 

    // Gather BDF weights for this block before assigning them to the workset
    //Kokkos::View<ScalarT*,AssemblyDevice> block_BDF_wts;
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
      // TODO why this logic? Is the so the storage get set up correctly?
      // Not so it actually reflects the total number of steps? 
      numsteps[set] = std::max(numsteps[set],newnumsteps);

    }
    else { // for steady state solves, u_dot = 0.0*u
      BDF_wts = Kokkos::View<ScalarT*,HostDevice>("BDF weights to compute u_dot",1);
      BDF_wts(0) = 1.0;
      numsteps[set] = 1;
    }

    dev_BDF_wts = Kokkos::View<ScalarT*,AssemblyDevice>("BDF weights on device",BDF_wts.extent(0));
    Kokkos::deep_copy(dev_BDF_wts, BDF_wts);
    //block_BDF_wts.push_back(dev_BDF_wts);
    assembler->wkset[block]->set_BDF_wts[set] = dev_BDF_wts; //block_BDF_wts;
    // TODO don't like this... but should protect against one set errors ASK 
    assembler->wkset[block]->BDF_wts = dev_BDF_wts;

  } // end loop blocks
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
  
  // Determine the offsets for each set as a Kokkos View
  for (size_t block=0; block<assembler->groups.size(); ++block) {
    if (assembler->wkset[block]->isInitialized) {
      for (size_t set=0; set<phys->setnames.size(); set++) {
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
        assembler->wkset[block]->set_offsets.push_back(offsets_view);
        if (set == 0) {
          assembler->wkset[block]->offsets = offsets_view;
        }
      }
    }
  }
  
  for (size_t block=0; block<assembler->groups.size(); ++block) {
    if (assembler->wkset[block]->isInitialized) {
      
      vector<vector<int> > block_useBasis;
      vector<vector<string> > block_varlist;
      
      for (size_t set=0; set<useBasis.size(); ++set) {
        block_useBasis.push_back(useBasis[set][block]);
        block_varlist.push_back(varlist[set][block]);
      }
      assembler->wkset[block]->set_usebasis = block_useBasis;
      assembler->wkset[block]->set_varlist = block_varlist;
      assembler->wkset[block]->usebasis = block_useBasis[0];
      assembler->wkset[block]->varlist = block_varlist[0];
    }
  }
  
  for (size_t block=0; block<assembler->groups.size(); ++block) {
    if (assembler->wkset[block]->isInitialized) {
      // set defaults for time integration params since these
      // won't get set if the total number of sets is 1
      assembler->wkset[block]->butcher_A = assembler->wkset[block]->set_butcher_A[0];
      assembler->wkset[block]->butcher_b = assembler->wkset[block]->set_butcher_b[0];
      assembler->wkset[block]->butcher_c = assembler->wkset[block]->set_butcher_c[0];
      assembler->wkset[block]->BDF_wts = assembler->wkset[block]->set_BDF_wts[0];
      // update workset for first physics set
      assembler->wkset[block]->updatePhysicsSet(0);
    }
  }
  
  // Parameters do not depend on physics sets
  for (size_t block=0; block<assembler->groups.size(); ++block) {
    if (assembler->wkset[block]->isInitialized) {
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
      assembler->wkset[block]->paramusebasis = params->discretized_param_usebasis;
      assembler->wkset[block]->paramoffsets = poffsets_view;
      assembler->wkset[block]->param_varlist = params->discretized_param_names;
    }
  }
  
  for (size_t block=0; block<assembler->groups.size(); ++block) {
    if (assembler->wkset[block]->isInitialized) {
      assembler->wkset[block]->createSolutionFields();
    }
  }
  
  for (size_t block=0; block<assembler->groups.size(); ++block) {
    if (assembler->wkset[block]->isInitialized) {
      vector<vector<int> > block_useBasis;
      for (size_t set=0; set<useBasis.size(); ++set) {
        block_useBasis.push_back(useBasis[set][block]);
      }
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        assembler->groups[block][grp]->setWorkset(assembler->wkset[block]);
        assembler->groups[block][grp]->setUseBasis(block_useBasis, numsteps, numstages);
        assembler->groups[block][grp]->setUpAdjointPrev(numsteps, numstages);
        assembler->groups[block][grp]->setUpSubGradient(params->num_active_params);
      }
      
      assembler->wkset[block]->params = params->paramvals_AD;
      assembler->wkset[block]->params_AD = params->paramvals_KVAD;
      assembler->wkset[block]->paramnames = params->paramnames;
      assembler->wkset[block]->setTime(current_time);
      if (assembler->boundary_groups.size() > block) { // avoid seg faults
        for (size_t grp=0; grp<assembler->boundary_groups[block].size(); ++grp) {
          if (assembler->boundary_groups[block][grp]->numElem > 0) {
            assembler->boundary_groups[block][grp]->setWorkset(assembler->wkset[block]);
            assembler->boundary_groups[block][grp]->setUseBasis(block_useBasis, numsteps, numstages);
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
  
  size_t numSets = phys->setnames.size();
  
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
          Teuchos::ParameterList dbc_settings = phys->setPhysSettings[set][block].sublist("Dirichlet conditions");
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
void SolverManager<Node>::projectDirichlet(const size_t & set) {
  
  Teuchos::TimeMonitor localtimer(*dbcprojtimer);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::projectDirichlet()" << endl;
    }
  }
  
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
    
    if (debug_level>2) {
      //KokkosTools::print(glmass,"L2-projection matrix for DBCs");
      //KokkosTools::print(glrhs,"L2-projections RHS for DBCs");
      //KokkosTools::print(glfixedDOF_soln,"L2-projections sol for DBCs");
    }
    
    // TODO BWR -- couldn't think of a good way to protect against
    // the preconditioner failing for HFACE, will need to be handled
    // explicitly in the input file for now (State boundary L2 linear solver)
    linalg->linearSolverBoundaryL2(set, glmass, glrhs, glfixedDOF_soln);
    linalg->importVectorToOverlapped(set, fixedDOF_soln[set], glfixedDOF_soln);
    
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
  
  vector<vector_RCP> u = this->setInitial();
    
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
void SolverManager<Node>::steadySolver(DFAD & objective, vector<vector_RCP> & u) {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::steadySolver ..." << endl;
    }
  }
  
  for (int ss=0; ss<subcycles; ++ss) {
    for (size_t set=0; set<setnames.size(); ++set) {
      assembler->updatePhysicsSet(set);
      vector_RCP zero_soln;
      if (usestrongDBCs) {
        this->setDirichlet(set, u[set]);
      }
      this->nonlinearSolver(set, u[set], zero_soln);
    }
  }
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
  
  if (setnames.size()>1) {
    if (Comm->getRank() == 0) {
      cout << "MrHyDE WARNING: Adjoints are not yet implemented for multiple physics sets." << endl;
    }
  }
  else {
    
    is_adjoint = true;
    
    params->sacadoizeParams(false);
    
    vector<vector_RCP> phi = setInitial();
    
    if (solver_type == "steady-state") {
      vector<vector_RCP> u;
      u.push_back(linalg->getNewVector(0));
      bool fnd = postproc->soln[0]->extract(u[0], current_time);
      if (!fnd) {
        cout << "UNABLE TO FIND FORWARD SOLUTION" << endl;
      }
      
      this->nonlinearSolver(0, u[0], phi[0]);
      
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
  }
  
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
void SolverManager<Node>::transientSolver(vector<vector_RCP> & initial, DFAD & obj, vector<ScalarT> & gradient,
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
  
  vector<vector_RCP> zero_vec(initial.size());
  
  current_time = start_time;
  if (!is_adjoint) { // forward solve - adaptive time stepping
    is_final_time = false;
    vector<vector_RCP> u = initial;
    
    if (usestrongDBCs) {
      for (size_t set=0; set<initial.size(); ++set) {
        assembler->updatePhysicsSet(set);
        this->setDirichlet(set,u[set]);
      }
    }
    
    for (size_t set=0; set<initial.size(); ++set) {
      assembler->updatePhysicsSet(set);
      assembler->performGather(set,u[set],0,0);
    }
    
    postproc->record(u,current_time,true,obj);
    
    for (size_t set=0; set<initial.size(); ++set) {
      assembler->updatePhysicsSet(set);
      for (int s=0; s<numsteps[set]; s++) {
        assembler->resetPrevSoln(set);
      }
    }
    
    int stepProg = 0;
    obj = 0.0;
    int numCuts = 0;
    int maxCuts = maxTimeStepCuts; // TMW: make this a user-defined input
    double timetol = end_time*1.0e-6; // just need to get close enough to final time
    bool write_this_step = false;
    
    vector<vector_RCP> u_prev;
    for (size_t set=0; set<initial.size(); ++set) {
      u_prev.push_back(linalg->getNewOverlappedVector(set));
      //u_stage.push_back(linalg->getNewOverlappedVector(set));
    }
    
    while (current_time < (end_time-timetol) && numCuts<=maxCuts) {
      
      int status = 0;
      if (Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Time Step " << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      
      for (int ss=0; ss<subcycles; ++ss) {
        for (size_t set=0; set<u.size(); ++set) {
          
          // TODO this needs to come first now, so that updatePhysicsSet can pick out the
          // time integration info
          if (BDForder[set] > 1 && stepProg == startupSteps[set]) {
            // Only overwrite the current set
            this->setBackwardDifference(BDForder,set);
            this->setButcherTableau(ButcherTab,set);
          }

          assembler->updatePhysicsSet(set);
          // TODO this is potentially now handled in the setBack.. and setButch.. routines
          // Make sure we have the correct number of RK stages (which can change after the transient startup)
          // TODO move this line into the if statement above for more clarity?
          // TODO if butcher_A is NOT SET by any block on an MPI process, this 
          // is zero! and causes issues...
          // currently, all sets need to share RK/BDF data and all MPI processes
          // need this data (even if no blocks on that process have elements in that set)
          // TODO discuss... but hacked to work for now by ensuring that
          // all RK/BDF data is shared regardless
          numstages[set] =  assembler->wkset[0]->butcher_A.extent(0);
      
          // Increment the previous step solutions (shift history and moves u into first spot)
          assembler->resetPrevSoln(set); 
          
          // Reset the stage solutions (sets all to zero)
          assembler->resetStageSoln(set);
          
          ////////////////////////////////////////////////////////////////////////
          // Allow the groups to change subgrid model
          ////////////////////////////////////////////////////////////////////////
          
          multiscale_manager->update();
          vector_RCP u_stage = linalg->getNewOverlappedVector(set);

          u_prev[set]->assign(*(u[set]));

          for (int stage=0; stage<numstages[set]; stage++) {
            // Need a stage solution
            // Set the initial guess for stage solution
            u_stage->assign(*(u_prev[set]));
   
            // TODO even with hack above updatestage hangs
            // Updates the current time and sets the stage number in wksets
            assembler->updateStage(stage, current_time, deltat); 

            if (usestrongDBCs) {
              this->setDirichlet(set, u_stage);
            }
  
            if (fully_explicit) {
              status += this->explicitSolver(set, u_stage, zero_vec[set], stage);
            }
            else {
              status += this->nonlinearSolver(set, u_stage, zero_vec[set]);
            }

            // u_{n+1} = u_n + \sum_stage ( u_stage - u_n )
            
            u[set]->update(1.0, *u_stage, 1.0);
            u[set]->update(-1.0, *(u_prev[set]), 1.0);
            
            assembler->updateStageSoln(set); // moves the stage solution into u_stage
            
          }
          
        }
      }
      
      if (status == 0) { // NL solver converged
        current_time += deltat;
        stepProg += 1;
        
        // Make sure last step solution is gathered
        // Last set of values is from a stage solution, which is potentially different
        for (size_t set=0; set<u.size(); ++set) {
          assembler->updatePhysicsSet(set);
          assembler->performGather(set,u[set],0,0);
        }
        // TODO :: BWR make this more flexible (may want to save based on simulation time as well)
        if (stepProg % postproc->write_frequency == 0) write_this_step = true;
        postproc->record(u,current_time,write_this_step,obj);
        write_this_step = false;
      }
      else { // something went wrong, cut time step and try again
        deltat *= 0.5;
        numCuts += 1;
        for (size_t set=0; set<u.size(); ++set) {
          assembler->revertSoln(set);
          u[set]->assign(*(u_prev[set]));
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
      postproc->writeSolution(current_time);
    }
  }
  else { // adjoint solve - fixed time stepping based on forward solve
    current_time = final_time;
    is_final_time = true;
    
    vector<vector_RCP> u, u_prev, phi, phi_prev;
    for (size_t set=0; set<1; ++set) { // hard coded for now
      u.push_back(linalg->getNewOverlappedVector(set));
      u_prev.push_back(linalg->getNewOverlappedVector(set));
      phi.push_back(linalg->getNewOverlappedVector(set));
      phi_prev.push_back(linalg->getNewOverlappedVector(set));
    }
    
    size_t set = 0;
    // Just getting the number of times from first physics set should be fine
    // TODO will this be affected by having physics sets with different timesteppers?
    size_t numFwdSteps = postproc->soln[set]->times[0].size()-1;
    
    for (size_t timeiter = 0; timeiter<numFwdSteps; timeiter++) {
      size_t cindex = numFwdSteps-timeiter;
      phi_prev[set] = linalg->getNewOverlappedVector(set);
      phi_prev[set]->update(1.0,*(phi[set]),0.0);
      if(Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Adjoint Time Step " << timeiter << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      
      // TMW: this is specific to implicit Euler
      // Needs to be generalized
      // Also, need to implement checkpoint/recovery
      bool fndu = postproc->soln[set]->extract(u[set], cindex);
      if (!fndu) {
        // throw error
      }
      bool fndup = postproc->soln[set]->extract(u_prev[set], cindex-1);
      if (!fndup) {
        // throw error
      }
      assembler->performGather(set,u_prev[set],0,0);
      assembler->resetPrevSoln(set);
      
      current_time = postproc->soln[set]->times[0][cindex-1];
      
      // if multistage, recover forward solution at each stage
      if (numstages[set] == 1) { // No need to re-solve in this case
        int status = this->nonlinearSolver(set, u[set], phi[set]);
        if (status>0) {
          // throw error
        }
        postproc->computeSensitivities(u, phi, current_time, deltat, gradient);
      }
      else {
        /*
        is_adjoint = false;
        vector<vector_RCP> stage_solns;
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
          assembler->updateStageSoln(); // moves the stage solution into u_stage (avoids mem transfer)
        }
        is_adjoint = true;
        
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
        }
        postproc->computeSensitivities(u, phi, current_time, deltat, gradient);
        */
      }
      
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
int SolverManager<Node>::nonlinearSolver(const size_t & set, vector_RCP & u, vector_RCP & phi) {
  
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
  
  int maxiter = maxNLiter;
  if (is_adjoint) {
    maxiter = 2;
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

    gNLiter = NLiter;
  
    bool build_jacobian = !linalg->getJacobianReuse(set);//true;
    matrix_RCP J = linalg->getNewMatrix(set);

    matrix_RCP J_over = linalg->getNewOverlappedMatrix(set);
    if (build_jacobian) {
      linalg->fillComplete(J_over);
    }
    
    // *********************** COMPUTE THE JACOBIAN AND THE RESIDUAL **************************
    
    current_res_over->putScalar(0.0);
    
    if (build_jacobian) {
      J_over->resumeFill();
      J_over->setAllToScalar(0.0);
    }
    
    store_adjPrev = false;
    if ( is_adjoint && (NLiter == 1)) {
      store_adjPrev = true;
    }

    assembler->assembleJacRes(set, u, phi, build_jacobian, false, false,
                              current_res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time, deltat);
    
    linalg->exportVectorFromOverlapped(set, current_res, current_res_over);
    
    if (is_adjoint) {
      ScalarT cdt = 0.0;
      if (solver_type == "transient") {
        cdt = deltat;
      }
      postproc->computeObjectiveGradState(set, u, current_time+cdt, deltat, current_res);
    }
    
    if (debug_level>2) {
      KokkosTools::print(current_res,"residual from solver interface");
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
        phi->update(-1.0*alpha, *(current_du_over), 1.0);
      }
      else {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        u->update(-1.0*alpha, *(current_du_over), 1.0);
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
        else if (resnorm[0]<1.0e-100) {
          solve = false;
          proceed = false;
        }
      }
      else if (useAbsoluteTOL && resnorm[0]<NLabstol) {
        solve = false;
        proceed = false;
      }
    }
    
    
    // *********************** SOLVE THE LINEAR SYSTEM **************************
    
    if (solve) {
      
      if (build_jacobian) {
        linalg->fillComplete(J_over);
        J->resumeFill();
        linalg->exportMatrixFromOverlapped(set, J, J_over);
        linalg->fillComplete(J);
      }
      
      if (debug_level>2) {
        KokkosTools::print(J,"Jacobian from solver interface");
      }
      current_du->putScalar(0.0);
      current_du_over->putScalar(0.0);
      linalg->linearSolver(set, J, current_res, current_du);
      linalg->importVectorToOverlapped(set, current_du_over, current_du);
      
      alpha = 1.0;
      if (is_adjoint) {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        phi->update(alpha, *(current_du_over), 1.0);
      }
      else {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        u->update(alpha, *(current_du_over), 1.0);
      }
    }
    NLiter++; // increment number of iterations
    
    if (NLiter >= maxiter) {
      proceed = false;
      // Need to perform another gather for cases where the number of iterations is tight
      assembler->performGather(set,u,0,0);
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
int SolverManager<Node>::explicitSolver(const size_t & set, vector_RCP & u, vector_RCP & phi, const int & stage) {
  
  
  Teuchos::TimeMonitor localtimer(*explicitsolvertimer);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting SolverManager::explicitSolver ..." << endl;
    }
  }
  
  int status = 0;
  assembler->updatePhysicsSet(set);
  
  if (usestrongDBCs) {
    this->setDirichlet(set,u);
  }
  
  bool build_jacobian = false;
  
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

  // *********************** COMPUTE THE RESIDUAL **************************
    
  current_res_over->putScalar(0.0);
  matrix_RCP J_over;
  
  assembler->assembleJacRes(set, u, phi, build_jacobian, false, false,
                            current_res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                            params->num_active_params, params->Psol[0], is_final_time, deltat);
  
  
  linalg->exportVectorFromOverlapped(set, current_res, current_res_over);
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  current_res->norm2(rnorm);
  
  // *********************** SOLVE THE LINEAR SYSTEM **************************
  
  if (rnorm[0]>1.0e-100) {
    // Given m = diag(M^-1)
    // Given timewt = b(stage)*deltat
    // Compute du = timewt*m*res
    // Compute u += du
    
    ScalarT wt = deltat*butcher_b(stage);
    
    current_du_over->putScalar(0.0);
    current_du->putScalar(0.0);
    
    if (!assembler->lump_mass) {
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
    }
    else {
      typedef typename Node::execution_space LA_exec;
      
      auto du_view = current_du->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto res_view = current_res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto dm_view = diagMass[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      
      parallel_for("explicit solver apply invdiag",
                   RangePolicy<LA_exec>(0,du_view.extent(0)),
                   KOKKOS_LAMBDA (const int k ) {
        du_view(k,0) = wt*res_view(k,0)/dm_view(k,0);
      });
    }
    linalg->importVectorToOverlapped(set, current_du_over, current_du);
    
    u->update(1.0, *(current_du_over), 1.0);
    
  }
  
  if (verbosity>=10) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> unorm(1);
    u->norm2(unorm);
    if (Comm->getRank() == 0) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Explicit integrator: L2 norm of solution: " << unorm[0] << endl;
      cout << "*********************************************************" << endl;
    }
  }
  
  
  assembler->performGather(set,u,0,0);
  
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
void SolverManager<Node>::setDirichlet(const size_t & set, vector_RCP & u) {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::setDirichlet ..." << endl;
    }
  }
  
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
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > SolverManager<Node>::setInitial() {
  
  Teuchos::TimeMonitor localtimer(*initsettimer);
  typedef typename Node::execution_space LA_exec;
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting SolverManager::setInitial ..." << endl;
    }
  }
  vector<vector_RCP> initial_solns;
  
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
        // output an error along the lines of "what the hell happened"
      }
    }
    
    if (have_initial_conditions[set]) {
      if (scalarInitialData[set]) {
        
        auto initial_kv = initial->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
        
        for (size_t block=0; block<assembler->groupData.size(); block++) {
          
          assembler->updatePhysicsSet(set);
          
          if (assembler->groupData[block]->numElem > 0) {
            
            Kokkos::View<ScalarT*,LA_device> idata("scalar initial data",scalarInitialValues[set][block].size());
            auto idata_host = Kokkos::create_mirror_view(idata);
            for (size_t i=0; i<scalarInitialValues[set][block].size(); i++) {
              idata_host(i) = scalarInitialValues[set][block][i];
            }
            Kokkos::deep_copy(idata,idata_host);
            
            if (samedevice) {
              auto offsets = assembler->wkset[block]->offsets;
              auto numDOF = assembler->groupData[block]->numDOF;
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
              auto numDOF = assembler->groupData[block]->numDOF_host;
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
          auto origPreconFlag = linalg->options_L2[set]->usePreconditioner;
          linalg->options_L2[set]->usePreconditioner = false;
          // do the solve
          linalg->linearSolverL2(set, glmass, glrhs, glinitial);
          // set back to original
          linalg->options_L2[set]->usePreconditioner = origPreconFlag;
          
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
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SolverManager::setInitial ..." << endl;
    }
  }
  
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
  if (multiscale_manager->subgridModels.size() > 0 ) {
    for (size_t k=0; k<multiscale_manager->subgridModels.size(); k++) {
      multiscale_manager->subgridModels[k]->paramvals_KVAD = params->paramvals_KVAD;
    }
    
    multiscale_manager->macro_wkset = assembler->wkset;
    vector<Kokkos::View<int*,AssemblyDevice>> macro_numDOF;
    for (size_t block=0; block<assembler->groupData.size(); ++block) {
      macro_numDOF.push_back(assembler->groupData[block]->set_numDOF[0]);
    }
    multiscale_manager->setMacroInfo(disc->basis_pointers, disc->basis_types,
                                     phys->varlist[0], useBasis[0], disc->offsets[0],
                                     macro_numDOF,
                                     params->paramnames, params->discretized_param_names);
    
    ScalarT my_cost = multiscale_manager->initialize();
    ScalarT gmin = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
    ScalarT gmax = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);
    
    if (Comm->getRank() == 0 && verbosity>0) {
      cout << "***** Load Balancing Factor " << gmax/gmin <<  endl;
    }
    
  }
  
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
    p_over = linalg->getNewOverlappedVector(set);
    q_over = linalg->getNewOverlappedVector(set);
  }
   
  p->putScalar(zero);
  q->putScalar(zero);
  r->putScalar(zero);
  z->putScalar(zero);
  
  p_over->putScalar(zero);
  q_over->putScalar(zero);
  
  int iter=0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  
  {
    Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
    linalg->importVectorToOverlapped(set, p_over, x);
    assembler->applyMassMatrixFree(set, p_over, q_over);
    linalg->exportVectorFromOverlapped(set, q, q_over);
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
      linalg->importVectorToOverlapped(set, p_over, p);
      q_over->putScalar(zero);
      assembler->applyMassMatrixFree(set, p_over, q_over);
      linalg->exportVectorFromOverlapped(set, q, q_over);
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
