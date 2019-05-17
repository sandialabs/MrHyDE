/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "multiscaleInterface.hpp"
#include "solverInterface.hpp"
#include "discretizationTools.hpp"
#include "workset.hpp"


// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

solver::solver(const Teuchos::RCP<LA_MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
               Teuchos::RCP<meshInterface> & mesh_,
               Teuchos::RCP<discretization> & disc_,
               Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager<LO,GO> > & DOF_,
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
  adj_soln = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  soln_dot = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  
  // Get the required information from the settings
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  numsteps = settings->sublist("Solver").get("numSteps",1);
  verbosity = settings->get<int>("verbosity",0);
  usestrongDBCs = settings->sublist("Solver").get<bool>("use strong DBCs",true);
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("Use Measurements as DBCs", false);
  solver_type = settings->sublist("Solver").get<string>("solver","none"); // or "transient"
  allow_remesh = settings->sublist("Solver").get<bool>("Remesh",false);
  finaltime = settings->sublist("Solver").get<ScalarT>("finaltime",1.0);
  time_order = settings->sublist("Solver").get<int>("time order",1);
  NLtol = settings->sublist("Solver").get<ScalarT>("NLtol",1.0E-6);
  MaxNLiter = settings->sublist("Solver").get<int>("MaxNLiter",10);
  NLsolver = settings->sublist("Solver").get<string>("Nonlinear Solver","Newton");
  line_search = false;//settings->sublist("Solver").get<bool>("Use Line Search","false");
  store_adjPrev = false;
  
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  else {
    numsteps = 1;
  }
  
  isInitial = false;
  initial_time = settings->sublist("Solver").get<ScalarT>("Initial Time",0.0);
  current_time = initial_time;
  
  /*
  solvetimes.push_back(current_time);
  
  if (isTransient) {
    ScalarT deltat = finaltime / numsteps;
    ScalarT ctime = current_time; // local current time
    for (int timeiter = 0; timeiter < numsteps; timeiter++) {
      ctime += deltat;
      solvetimes.push_back(ctime);
    }
  }
  */
  
  response_type = settings->sublist("Postprocess").get("response type", "pointwise"); // or "global"
  compute_objective = settings->sublist("Postprocess").get("compute objective",false);
  compute_sensitivity = settings->sublist("Postprocess").get("compute sensitivities",false);
  
  initial_type = settings->sublist("Solver").get<string>("Initial type","L2-projection");
  multigrid_type = settings->sublist("Solver").get<string>("Multigrid type","sa");
  smoother_type = settings->sublist("Solver").get<string>("Smoother type","CHEBYSHEV"); // or RELAXATION
  lintol = settings->sublist("Solver").get<ScalarT>("lintol",1.0E-7);
  liniter = settings->sublist("Solver").get<int>("liniter",100);
  kspace = settings->sublist("Solver").get<int>("krylov vectors",100);
  useDomDecomp = settings->sublist("Solver").get<bool>("use dom decomp",false);
  useDirect = settings->sublist("Solver").get<bool>("use direct solver",false);
  usePrec = settings->sublist("Solver").get<bool>("use preconditioner",true);
  dropTol = settings->sublist("Solver").get<ScalarT>("ILU drop tol",0.0); //defaults to AztecOO default
  fillParam = settings->sublist("Solver").get<ScalarT>("ILU fill param",3.0); //defaults to AztecOO default
  
  // needed information from the mesh
  mesh->mesh->getElementBlockNames(blocknames);
  
  // needed information from the physics interface
  numVars = phys->numVars; //
  vector<vector<string> > phys_varlist = phys->varlist;
  //offsets = phys->offsets;
  
  // Set up the time integrator
  string timeinttype = settings->sublist("Solver").get<string>("Time integrator","RK");
  string timeintmethod = settings->sublist("Solver").get<string>("Time method","Implicit");
  int timeintorder = settings->sublist("Solver").get<int>("Time order",1);
  bool timeintstagger = settings->sublist("Solver").get<bool>("Stagger solutions",true);

  //if (timeinttype == "RK") {
  //  timeInt = Teuchos::rcp(new RungeKutta(timeintmethod,timeintorder,timeintstagger));
  //}
  
  // needed information from the DOF manager
  DOF->getOwnedIndices(LA_owned);
  numUnknowns = (LO)LA_owned.size();
  DOF->getOwnedAndGhostedIndices(LA_ownedAndShared);
  numUnknownsOS = (LO)LA_ownedAndShared.size();
  LO localNumUnknowns = numUnknowns;
  
  DOF->getOwnedIndices(owned);
  DOF->getOwnedAndGhostedIndices(ownedAndShared);
  
  int nstages = 1;//timeInt->num_stages;
  bool sol_staggered = true;//timeInt->sol_staggered;
  /*
  LA_owned = vector(numUnknowns);
  LA_ownedAndShared = vector(numUnknownsOS);
  for (size_t i=0; i<numUnknowns; i++) {
    LA_owned[i] = owned[i];
  }
  for (size_t i=0; i<numUnknownsOS; i++) {
    LA_ownedAndShared[i] = ownedAndShared[i];
  }
   */
  //for (size_t i=0; i<numUnknowns; i++) {
    //for (int s=0; s<nstages; s++) {
    //  LA_owned[i*(nstages)+s] = owned[i]*nstages+s;
    //}
  //}
  //for (size_t i=0; i<numUnknownsOS; i++) {
  //  for (int s=0; s<nstages; s++) {
  //    LA_ownedAndShared[i*(nstages)+s] = ownedAndShared[i]*nstages+s;
  //  }
  //}
  
  globalNumUnknowns = 0;
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
  //Comm->SumAll(&localNumUnknowns, &globalNumUnknowns, 1);
  
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
  // Epetra maps
  /////////////////////////////////////////////////////////////////////////////
  
  this->setupLinearAlgebra();
  
  /////////////////////////////////////////////////////////////////////////////
  // Worksets
  /////////////////////////////////////////////////////////////////////////////
  
  assembler->createWorkset();
  this->finalizeWorkset();
  phys->setWorkset(assembler->wkset);
  params->wkset = assembler->wkset;
  
  if (settings->sublist("Mesh").get<bool>("Have Element Data", false) ||
      settings->sublist("Mesh").get<bool>("Have Nodal Data", false)) {
    mesh->readMeshData(LA_overlapped_map, assembler->cells);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver constructor" << endl;
    }
  }
  
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
  
  int nstages = 1;//timeInt->num_stages;
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    vector<vector<int> > voffsets = phys->offsets[b];
    size_t maxoff = 0;
    for (size_t i=0; i<voffsets.size(); i++) {
      if (voffsets[i].size() > maxoff) {
        maxoff = voffsets[i].size();
      }
    }
    Kokkos::View<int**,HostDevice> offsets_host("offsets on host device",voffsets.size(),maxoff);
    for (size_t i=0; i<voffsets.size(); i++) {
      for (size_t j=0; j<voffsets[i].size(); j++) {
        offsets_host(i,j) = voffsets[i][j];
      }
    }
    Kokkos::View<int**,AssemblyDevice>::HostMirror offsets_device = Kokkos::create_mirror_view(offsets_host);
    Kokkos::deep_copy(offsets_host, offsets_device);
    assembler->wkset[b]->offsets = offsets_device;//phys->voffsets[b];
    
    size_t maxpoff = 0;
    for (size_t i=0; i<params->paramoffsets.size(); i++) {
      if (params->paramoffsets[i].size() > maxpoff) {
        maxpoff = params->paramoffsets[i].size();
      }
      //maxpoff = max(maxpoff,paramoffsets[i].size());
    }
    Kokkos::View<int**,HostDevice> poffsets_host("param offsets on host device",params->paramoffsets.size(),maxpoff);
    for (size_t i=0; i<params->paramoffsets.size(); i++) {
      for (size_t j=0; j<params->paramoffsets[i].size(); j++) {
        poffsets_host(i,j) = params->paramoffsets[i][j];
      }
    }
    Kokkos::View<int**,AssemblyDevice>::HostMirror poffsets_device = Kokkos::create_mirror_view(poffsets_host);
    Kokkos::deep_copy(poffsets_host, poffsets_device);
    
    assembler->wkset[b]->usebasis = useBasis[b];
    assembler->wkset[b]->paramusebasis = params->discretized_param_usebasis;
    assembler->wkset[b]->paramoffsets = poffsets_device;//paramoffsets;
    assembler->wkset[b]->varlist = varlist[b];
    int numDOF = assembler->cells[b][0]->GIDs.dimension(1);
    for (size_t e=0; e<assembler->cells[b].size(); e++) {
      assembler->cells[b][e]->wkset = assembler->wkset[b];
      assembler->cells[b][e]->setUseBasis(useBasis[b],nstages);
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
          assembler->boundaryCells[b][e]->setUseBasis(useBasis[b],nstages);
          
          assembler->wkset[b]->addSide(assembler->boundaryCells[b][e]->nodes,
                                       assembler->boundaryCells[b][e]->sidenum,
                                       assembler->boundaryCells[b][e]->localSideID,e);
        }
      }
    }
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::finalizeWorsket" << endl;
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
  
  LA_owned_map = Teuchos::rcp(new LA_Map(INVALID, LA_owned, 0, Comm));
  LA_overlapped_map = Teuchos::rcp(new LA_Map(INVALID, LA_ownedAndShared, 0, Comm));
  LA_owned_graph = createCrsGraph(LA_owned_map);//Teuchos::rcp(new LA_CrsGraph(Copy, *LA_owned_map, 0));
  LA_overlapped_graph = createCrsGraph(LA_overlapped_map);//Teuchos::rcp(new LA_CrsGraph(Copy, *LA_overlapped_map, 0));
  
  exporter = Teuchos::rcp(new LA_Export(LA_overlapped_map, LA_owned_map));
  importer = Teuchos::rcp(new LA_Import(LA_owned_map, LA_overlapped_map));
  //importer = Teuchos::rcp(new LA_Import(LA_overlapped_map, LA_owned_map));
  
  
  Kokkos::View<GO**,HostDevice> gids;
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    vector<vector<int> > curroffsets = phys->offsets[b];
    Kokkos::View<LO*,HostDevice> numDOF_KVhost("number of DOF per variable",numVars[b]);
    for (int k=0; k<numVars[b]; k++) {
      numDOF_KVhost(k) = numBasis[b][k];
    }
    Kokkos::View<LO*,AssemblyDevice> numDOF_KV = Kokkos::create_mirror_view(numDOF_KVhost);
    Kokkos::deep_copy(numDOF_KVhost, numDOF_KV);
    
    for(size_t e=0; e<assembler->cells[b].size(); e++) {
      gids = assembler->cells[b][e]->GIDs;
      
      int numElem = assembler->cells[b][e]->numElem;
      
      // this should fail on the first iteration through if maxDerivs is not large enough
      TEUCHOS_TEST_FOR_EXCEPTION(gids.dimension(1) > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of degrees of freedom per element times the number of time stages.");
      //vector<vector<vector<int> > > cellindices;
      Kokkos::View<LO***,AssemblyDevice> cellindices("Local DOF indices", numElem, numVars[b], maxBasis[b]);
      for (int p=0; p<numElem; p++) {
        //vector<vector<int> > indices;
        for (int n=0; n<numVars[b]; n++) {
          //vector<int> cindex;
          for( int i=0; i<numBasis[b][n]; i++ ) {
            GO cgid = gids(p,curroffsets[n][i]);
            cellindices(p,n,i) = LA_overlapped_map->getLocalElement(cgid);
            //cindex.push_back(LA_overlapped_map->getLocalElement(cgid));
          }
          //indices.push_back(cindex);
        }
        Teuchos::Array<GO> ind2(gids.dimension(1));
        for (size_t i=0; i<gids.dimension(1); i++) {
          ind2[i] = gids(p,i);
        }
        for (size_t i=0; i<gids.dimension(1); i++) {
          GO ind1 = gids(p,i);
          LA_overlapped_graph->insertGlobalIndices(ind1,ind2);
        }
        //cellindices.push_back(indices);
        
      }
      assembler->cells[b][e]->setIndex(cellindices, numDOF_KV);
    }
    
    if (assembler->boundaryCells.size() > b) {
      for(size_t e=0; e<assembler->boundaryCells[b].size(); e++) {
        gids = assembler->boundaryCells[b][e]->GIDs;
        
        int numElem = assembler->boundaryCells[b][e]->numElem;
        
        // this should fail on the first iteration through if maxDerivs is not large enough
        TEUCHOS_TEST_FOR_EXCEPTION(gids.dimension(1) > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of degrees of freedom per element times the number of time stages.");
        //vector<vector<vector<int> > > cellindices;
        Kokkos::View<LO***,AssemblyDevice> cellindices("Local DOF indices", numElem, numVars[b], maxBasis[b]);
        for (int p=0; p<numElem; p++) {
          //vector<vector<int> > indices;
          for (int n=0; n<numVars[b]; n++) {
            //vector<int> cindex;
            for( int i=0; i<numBasis[b][n]; i++ ) {
              GO cgid = gids(p,curroffsets[n][i]);
              cellindices(p,n,i) = LA_overlapped_map->getLocalElement(cgid);
              //cindex.push_back(LA_overlapped_map->getLocalElement(cgid));
            }
            //indices.push_back(cindex);
          }
          Teuchos::Array<GO> ind2(gids.dimension(1));
          for (size_t i=0; i<gids.dimension(1); i++) {
            ind2[i] = gids(p,i);
          }
          for (size_t i=0; i<gids.dimension(1); i++) {
            GO ind1 = gids(p,i);
            LA_overlapped_graph->insertGlobalIndices(ind1,ind2);
          }
          //cellindices.push_back(indices);
          
        }
        assembler->boundaryCells[b][e]->setIndex(cellindices, numDOF_KV);
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
// Set up the Epetra overlapped CrsGraph (for bwds compat.)
// ========================================================================================

Teuchos::RCP<Epetra_CrsGraph> solver::buildEpetraOverlappedGraph(Epetra_MpiComm & EP_Comm) {
  
  //Epetra_MpiComm EP_Comm(*(Comm->getRawMpiComm()));
  Teuchos::RCP<Epetra_Map> Ep_map = Teuchos::rcp(new Epetra_Map(-1, (LO)LA_ownedAndShared.size(), &LA_ownedAndShared[0], 0, EP_Comm));
   
  Teuchos::RCP<Epetra_CrsGraph> Ep_graph = Teuchos::rcp(new Epetra_CrsGraph(Copy, *Ep_map, 0));
  
  Kokkos::View<GO**,HostDevice> gids;
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    vector<vector<int> > curroffsets = phys->offsets[b];
    for(size_t e=0; e<assembler->cells[b].size(); e++) {
      gids = assembler->cells[b][e]->GIDs;
      for (int p=0; p<gids.dimension(0); p++) {
        for (size_t i=0; i<gids.dimension(1); i++) {
          GO ind1 = gids(p,i);
          for (size_t j=0; j<gids.dimension(1); j++) {
            GO ind2 = gids(p,j);
            int err = Ep_graph->InsertGlobalIndices(ind1,1,&ind2);
          }
        }
      }
    }
  }
  Ep_graph->FillComplete();
  return Ep_graph;
}

// ========================================================================================
// Set up the Epetra owned CrsGraph (for bwds compat.)
// ========================================================================================

Teuchos::RCP<Epetra_CrsGraph> solver::buildEpetraOwnedGraph(Epetra_MpiComm & EP_Comm) {
  
  //Epetra_MpiComm EP_Comm(*(Comm->getRawMpiComm()));
  Teuchos::RCP<Epetra_Map> Ep_map = Teuchos::rcp(new Epetra_Map(-1, (LO)LA_owned.size(), &LA_owned[0], 0, EP_Comm));
  
  Teuchos::RCP<Epetra_CrsGraph> Ep_graph = Teuchos::rcp(new Epetra_CrsGraph(Copy, *Ep_map, 0));
  
  Kokkos::View<GO**,HostDevice> gids;
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    vector<vector<int> > curroffsets = phys->offsets[b];
    for(size_t e=0; e<assembler->cells[b].size(); e++) {
      gids = assembler->cells[b][e]->GIDs;
      for (int p=0; p<gids.dimension(0); p++) {
        for (size_t i=0; i<gids.dimension(1); i++) {
          GO ind1 = gids(p,i);
          for (size_t j=0; j<gids.dimension(1); j++) {
            GO ind2 = gids(p,j);
            int err = Ep_graph->InsertGlobalIndices(ind1,1,&ind2);
          }
        }
      }
    }
  }
  Ep_graph->FillComplete();
  return Ep_graph;
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
  
  vector_RCP u = this->setInitial(); // TMW: this will be deprecated soon
  if (solver_type == "transient") {
    soln->store(u, current_time, 0); // copies the data
  }
  
  vector_RCP zero_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
  if (solver_type == "steady-state") {
    
    this->nonlinearSolver(u, zero_soln, zero_soln, zero_soln, 0.0, 1.0);
    if (compute_objective) {
      obj = this->computeObjective(u, 0.0, 0);
    }
    soln->store(u, current_time, 0);
  }
  else if (solver_type == "transient") {
    vector<ScalarT> gradient; // not really used here
    this->transientSolver(u, obj, gradient);
  }
  else {
    // print out an error message
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished solver::forwardModel" << endl;
    }
  }
  
  //return F_soln;
}

// ========================================================================================
/* given the parameters, solve the fractional forward  problem */
// ========================================================================================

void solver::forwardModel_fr(DFAD & obj, ScalarT yt, ScalarT st) {
  
  current_time = initial_time;
  
  useadjoint = false;
  assembler->wkset[0]->y = yt;
  assembler->wkset[0]->s = st;
  params->sacadoizeParams(false);
  
  // Set the initial condition
  //isInitial = true;
  
  vector_RCP u = this->setInitial(); // TMW: this will be deprecated soon
  if (solver_type == "transient") {
    soln->store(u, current_time, 0);
  }
  vector_RCP I_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
  int numsols = 1;
  if (solver_type == "transient") {
    numsols = numsteps+1;
  }
  
  //vector_RCP F_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,numsols)); // empty solution
  //vector_RCP F_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
  vector_RCP zero_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
  
  if (solver_type == "steady-state") {
    this->nonlinearSolver(u, zero_soln, zero_soln, zero_soln, 0.0, 1.0);
    soln->store(u, current_time, 0);
    if (compute_objective) {
      obj = this->computeObjective(u, 0.0, 0);
    }
    
  }
  else if (solver_type == "transient") {
    vector<ScalarT> gradient; // not really used here
    this->transientSolver(u, obj, gradient);
  }
  else {
    // print out an error message
  }
  
  //return F_soln;
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
  
  //isInitial = true;
  vector_RCP phi = setInitial(); // does this need
  // to be updated for adjoint model?
  
  // Solve the forward problem
  //int numsols = 1;
  //if (solver_type == "transient") {
  //  numsols = numsteps+1;
  //}
  
  vector_RCP zero_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
  //vector_RCP A_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,numsols)); // empty solution
  
  //auto initial_2d = initial->getLocalView<HostDevice>();
  //auto asol_2d = A_soln->getLocalView<HostDevice>();
  //auto fsol_2d = F_soln->getLocalView<HostDevice>();
  //
  //for( size_t i=0; i<ownedAndShared.size(); i++ ) {
  //  asol_2d(i,0) = initial_2d(i,0);
  //}
  
  if (solver_type == "steady-state") {
    //vector_RCP L_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
    //vector_RCP SS_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // empty solution
    //auto lsol_2d = L_soln->getLocalView<HostDevice>();
    //auto SS_2d = SS_soln->getLocalView<HostDevice>();
    
    
    //for( size_t i=0; i<ownedAndShared.size(); i++ ) {
    //  lsol_2d(i,0) = fsol_2d(i,0);
    //}
    vector_RCP u;
    bool fnd = soln->extract(u, current_time);
    this->nonlinearSolver(u, zero_soln, phi, zero_soln, 0.0, 1.0);
    
    this->computeSensitivities(u, zero_soln, phi, gradient, 0.0, 1.0);
    
  }
  else if (solver_type == "transient") {
    DFAD obj = 0.0;
    this->transientSolver(phi, obj, gradient);
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

void solver::transientSolver(vector_RCP & initial, DFAD & obj, vector<ScalarT> & gradient) {
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting solver::transientSolver ..." << endl;
    }
  }
  
  ScalarT deltat = 0.0;
  ScalarT alpha = 0.0;
  ScalarT beta = 1.0;
  deltat = finaltime / numsteps;
  if (time_order == 1){
    alpha = 1./deltat;
  }
  else if (time_order == 2) {
    alpha = 3.0/2.0/deltat;
  }
  else {
    alpha = 0.0; // would be better to print out an error message
  }
  
  
  if (!useadjoint) { // forward solve - adaptive time stepping
    is_final_time = false;
    vector_RCP u = initial;
    vector_RCP u_dot = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP zero_vec = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    u_dot->putScalar(0.0);
    zero_vec->putScalar(0.0);
    
    obj = 0.0;
    int numCuts = 0;
    int maxCuts = 5; // TMW: make this a user-defined input
    while (abs(current_time - finaltime)>1.0e-12 && numCuts<=maxCuts) {
      
      current_time += deltat;
      
      u_dot->putScalar(0.0);
      
      ////////////////////////////////////////////////////////////////////////
      // Allow the cells to change subgrid model
      ////////////////////////////////////////////////////////////////////////
      {
        Teuchos::TimeMonitor localtimer(*msprojtimer);
        ScalarT my_cost = multiscale_manager->update();
        ScalarT gmin = 0.0;
        Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
        ScalarT gmax = 0.0;
        Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);
        if(Comm->getRank() == 0 && verbosity>0) {
          cout << "***** Load Balancing Factor " << gmax/gmin <<  endl;
        }
      }
      
      if(Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Time Step " << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      
      int status = this->nonlinearSolver(u, u_dot, zero_vec, zero_vec, alpha, beta);
      
      if (status == 0) { // NL solver converged
        soln->store(u, current_time, 0);
        soln_dot->store(u_dot, current_time, 0);
        
        if (allow_remesh) {
          mesh->remesh(u, assembler->cells);
        }
        
        if (compute_objective) { // fill in the objective function
          DFAD cobj = this->computeObjective(u, current_time, soln->times[0].size()-1);
          obj += cobj;
        }
      }
      else { // something went wrong, cut time step and try again
        current_time -= deltat;
        deltat *= 0.5;
        current_time += deltat;
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
    current_time = finaltime;
    is_final_time = true;
    
    vector_RCP u = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP u_prev = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP u_dot = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP phi = initial;
    vector_RCP phi_dot = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    phi_dot->putScalar(0.0);
    
    size_t numsteps = soln->times[0].size()-1;
    
    for (size_t timeiter = 0; timeiter<numsteps; timeiter++) {
      size_t cindex = numsteps-timeiter;
      phi_dot->putScalar(0.0);
      current_time = soln->times[0][cindex];
      
      if(Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Adjoint Time Step " << timeiter << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      
      bool fndu = soln->extract(u, cindex);
      bool fndup = soln->extract(u_prev, cindex-1);
      //bool fndudot = soln->extract(u_dot, cindex);
      auto u_kv = u->getLocalView<HostDevice>();
      auto u_prev_kv = u_prev->getLocalView<HostDevice>();
      auto u_dot_kv = u_dot->getLocalView<HostDevice>();
      for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
        u_dot_kv(i,0) = alpha*u_kv(i,0) - alpha*u_prev_kv(i,0);
      }
      int status = this->nonlinearSolver(u, u_dot, phi, phi_dot, alpha, beta);
      
      adj_soln->store(phi,current_time,0);
      
      this->computeSensitivities(u,u_dot,phi,gradient,alpha,beta);
      
      //current_time -= deltat;
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


int solver::nonlinearSolver(vector_RCP & u, vector_RCP & u_dot,
                           vector_RCP & phi, vector_RCP & phi_dot,
                           const ScalarT & alpha, const ScalarT & beta) {
  
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
    
    gNLiter = NLiter;
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    matrix_RCP J = Tpetra::createCrsMatrix<ScalarT>(LA_owned_map);
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    matrix_RCP J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_overlapped_graph));
    vector_RCP du = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
    vector_RCP du_over = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    
    // *********************** COMPUTE THE JACOBIAN AND THE RESIDUAL **************************
    
    bool build_jacobian = true;
    if (NLsolver == "AA")
    build_jacobian = false;
    
    res_over->putScalar(0.0);
    J_over->setAllToScalar(0.0);
    if ( useadjoint && (NLiter == 1))
      store_adjPrev = true;
    else
      store_adjPrev = false;
    
    assembler->assembleJacRes(u, u_dot, phi, phi_dot, alpha, beta, build_jacobian, false, false,
                              res_over, J_over, isTransient, current_time, useadjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time);
    J_over->fillComplete();
    
    J->setAllToScalar(0.0);
    J->doExport(*J_over, *exporter, Tpetra::ADD);
    J->fillComplete();
    
    res->putScalar(0.0);
    res->doExport(*res_over, *exporter, Tpetra::ADD);
    
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
    
    if (NLerr_scaled[0] > NLtol) {
      
      this->linearSolver(J, res, du_over);
      
      du->doImport(*du_over, *importer, Tpetra::ADD);
      
      if (useadjoint) {
        phi->update(1.0, *du, 1.0);
        phi_dot->update(alpha, *du, 1.0);
      }
      else {
        u->update(1.0, *du, 1.0);
        u_dot->update(alpha, *du, 1.0);
      }
    }
    
    NLiter++; // increment number of iterations
  } // while loop
  
  if(Comm->getRank() == 0) {
    if (!useadjoint) {
      if( (NLiter>MaxNLiter || NLerr_scaled[0]>NLtol) && verbosity > 1) {
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
      
      if (obj.dimension(1) > 0) {
        for (int c=0; c<numElem; c++) {
          for (size_t i=0; i<obj.dimension(1); i++) {
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
      }
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

void solver::computeSensitivities(vector_RCP & u, vector_RCP & u_dot,
                          vector_RCP & a2, vector<ScalarT> & gradient,
                          const ScalarT & alpha, const ScalarT & beta) {
  
  DFAD obj_sens = this->computeObjective(u, current_time, 0);
  
  auto u_kv = u->getLocalView<HostDevice>();
  auto u_dot_kv = u_dot->getLocalView<HostDevice>();
  auto a2_kv = a2->getLocalView<HostDevice>();
  
  if (params->num_active_params > 0) {
  
    params->sacadoizeParams(true);
    
    vector<ScalarT> localsens(params->num_active_params);
    ScalarT globalsens = 0.0;
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(LA_owned_map,params->num_active_params)); // reset residual
    matrix_RCP J = Tpetra::createCrsMatrix<ScalarT>(LA_owned_map); // reset Jacobian
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,params->num_active_params)); // reset residual
    matrix_RCP J_over = Tpetra::createCrsMatrix<ScalarT>(LA_overlapped_map); // reset Jacobian
    
    auto res_kv = res->getLocalView<HostDevice>();
    
    res_over->putScalar(0.0);
    
    bool curradjstatus = useadjoint;
    useadjoint = false;
    
    assembler->assembleJacRes(u, u_dot, u, u_dot, alpha, beta, false, true, false,
                              res_over, J_over, isTransient, current_time, useadjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time);
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
    matrix_RCP J_over = Tpetra::createCrsMatrix<ScalarT>(params->param_overlapped_map); // reset Jacobian
    matrix_RCP J = Tpetra::createCrsMatrix<ScalarT>(params->param_owned_map); // reset Jacobian
    
    res_over->putScalar(0.0);
    J->setAllToScalar(0.0);
    J_over->setAllToScalar(0.0);
    
    //this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, true, false, true, res_over, J_over);
    assembler->assembleJacRes(u, u_dot, u, u_dot, alpha, beta, true, false, true,
                              res_over, J_over, isTransient, current_time, useadjoint, store_adjPrev,
                              params->num_active_params, params->Psol[0], is_final_time);
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
  
  auto init_kv = initial->getLocalView<HostDevice>();
  //auto meas_kv = meas->getLocalView<HostDevice>();
  
  // TMW: this function needs to be fixed
  vector<vector<int> > fixedDOFs = phys->dbc_dofs;
  
  for (size_t b=0; b<blocknames.size(); b++) {
    string blockID = blocknames[b];
    Kokkos::View<int**,HostDevice> side_info;
    
    for (int n=0; n<numVars[b]; n++) {
      
      vector<size_t> localDirichletSideIDs = phys->localDirichletSideIDs[b][n];
      vector<size_t> boundDirichletElemIDs = phys->boundDirichletElemIDs[b][n];
      int fnum = DOF->getFieldNum(varlist[b][n]);
      for( size_t e=0; e<disc->myElements[b].size(); e++ ) { // loop through all the elements
        side_info = phys->getSideInfo(b,n,e);
        int numSides = side_info.dimension(0);
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
    vector<LO> dbc_dofs = fixedDOFs[b];
    
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
  
  vector_RCP initial = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1));
  vector_RCP glinitial = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
  initial->putScalar(0.0);
  
  if (initial_type == "L2-projection") {
    
    // Compute the L2 projection of the initial data into the discrete space
    vector_RCP rhs = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // reset residual
    matrix_RCP mass = Tpetra::createCrsMatrix<ScalarT>(LA_overlapped_map); // reset Jacobian
    vector_RCP glrhs = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1)); // reset residual
    matrix_RCP glmass = Tpetra::createCrsMatrix<ScalarT>(LA_owned_map); // reset Jacobian
    
    assembler->setInitial(rhs, mass, useadjoint);
    
    glmass->setAllToScalar(0.0);
    glmass->doExport(*mass, *exporter, Tpetra::ADD);
    
    glrhs->putScalar(0.0);
    glrhs->doExport(*rhs, *exporter, Tpetra::ADD);
    
    glmass->fillComplete();
    
    this->linearSolver(glmass, glrhs, glinitial);
    
    initial->doImport(*glinitial, *importer, Tpetra::ADD);
    
  }
  else if (initial_type == "interpolation") {
    
    assembler->setInitial(initial, useadjoint);
    
  }
  
  return initial;
}

// ========================================================================================
// Linear Solver for Tpetra stack
// ========================================================================================

void solver::linearSolver(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  //KokkosTools::print(r);
  //LA_LinearProblem LinSys(J.get(), soln.get(), r.get());
  Teuchos::RCP<LA_LinearProblem> Problem = Teuchos::rcp(new LA_LinearProblem(J, soln, r));
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > M = buildPreconditioner(J);
  
  Problem->setLeftPrec(M);
  Problem->setProblem();
  
  Teuchos::RCP<Teuchos::ParameterList> belosList = Teuchos::rcp(new Teuchos::ParameterList());
  belosList->set("Maximum Iterations",    kspace); // Maximum number of iterations allowed
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
  belosList->set("Implicit Residual Scaling", "None");
  
  Teuchos::RCP<Belos::SolverManager<ScalarT, LA_MultiVector, LA_Operator> > solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT, LA_MultiVector, LA_Operator>(Problem, belosList));
  
  solver->solve();
}

// ========================================================================================
// Linear solver for Epetra stack (mostly deprecated)
// ========================================================================================

void solver::linearSolver(Teuchos::RCP<Epetra_CrsMatrix> & J,
                          Teuchos::RCP<Epetra_MultiVector> & r,
                          Teuchos::RCP<Epetra_MultiVector> & soln)  {
  
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
    
  Epetra_LinearProblem LinSys(J.get(), soln.get(), r.get());

  
  // SOLVE ....
  if (useDirect) {
    Amesos AmFactory;
    char* SolverType = "Amesos_Klu";
    Amesos_BaseSolver * AmSolver = AmFactory.Create(SolverType, LinSys);
    AmSolver->SymbolicFactorization();
    AmSolver->NumericFactorization();
    AmSolver->Solve();
    delete AmSolver;
  }
  else {
    AztecOO linsolver(LinSys);
    
    // Set up the preconditioner
    ML_Epetra::MultiLevelPreconditioner* MLPrec;
    
    linsolver.SetAztecOption(AZ_solver,AZ_gmres);
    if(useDomDecomp){ //domain decomposition preconditioner, specific to Helmholtz at high frequencies
      linsolver.SetAztecOption(AZ_precond,AZ_dom_decomp);
      linsolver.SetAztecOption(AZ_subdomain_solve,AZ_ilut);
      linsolver.SetAztecParam(AZ_drop,dropTol);
      linsolver.SetAztecParam(AZ_ilut_fill,fillParam);
      
      if(verbosity == 0)
      linsolver.SetAztecOption(AZ_diagnostics,AZ_none);
      
      ScalarT condest = 0.0;
      linsolver.ConstructPreconditioner(condest);
      if(condest > 1.e13 || condest < 1.0){
        linsolver.DestroyPreconditioner();
        linsolver.SetAztecParam(AZ_athresh,1.e-5);
        linsolver.SetAztecParam(AZ_rthresh,0.0);
        linsolver.ConstructPreconditioner(condest);
        if(condest > 1.e13 || condest < 1.0){
          linsolver.DestroyPreconditioner();
          linsolver.SetAztecParam(AZ_athresh,1.e-5);
          linsolver.SetAztecParam(AZ_rthresh,0.01);
          linsolver.ConstructPreconditioner(condest);
          if(condest > 1.e13 || condest < 1.0){
            linsolver.DestroyPreconditioner();
            linsolver.SetAztecParam(AZ_athresh,1.e-2);
            linsolver.SetAztecParam(AZ_rthresh,0.0);
            linsolver.ConstructPreconditioner(condest);
            if(condest > 1.e13 || condest < 1.0){
              linsolver.DestroyPreconditioner();
              linsolver.SetAztecParam(AZ_athresh,1.e-2);
              linsolver.SetAztecParam(AZ_rthresh,0.01);
              linsolver.ConstructPreconditioner(condest);
              if(condest > 1.e13){
                cout << "SAD PRECONDITIONER: condition number " << condest << endl;
              }
            }
          }
        }
      }
    }
    else if (usePrec) { //multi-level preconditioner
      MLPrec = buildPreconditioner(J);
      linsolver.SetPrecOperator(MLPrec);
    }
    else {
      linsolver.SetAztecOption(AZ_precond, AZ_none);
    }
    linsolver.SetAztecOption(AZ_kspace,kspace);
   
    if (verbosity > 8)
    linsolver.SetAztecOption(AZ_output,10);
    else
    linsolver.SetAztecOption(AZ_output,0);
    
    linsolver.Iterate(liniter,lintol);
    
    if(!useDomDecomp && usePrec)
    delete MLPrec;
   
  }
  
  //return soln;
}


// ========================================================================================
// Preconditioner for Tpetra stack
// ========================================================================================

Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > solver::buildPreconditioner(const matrix_RCP & J) {
  Teuchos::ParameterList mueluParams;
  
  mueluParams.setName("MueLu");
  
  // Main settings
  if (verbosity >= 10){
    mueluParams.set("verbosity","high");
  }
  else {
    mueluParams.set("verbosity","none");
  }
  int numEqns = 1;
  if (assembler->cells.size() == 1) {
    numEqns = numVars[0];
  }
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
  
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > M = MueLu::CreateTpetraPreconditioner((Teuchos::RCP<LA_Operator>)J, mueluParams);

  return M;
}

// ========================================================================================
// Preconditioner for Epetra stack
// ========================================================================================

ML_Epetra::MultiLevelPreconditioner* solver::buildPreconditioner(const Teuchos::RCP<Epetra_CrsMatrix> & J) {
  Teuchos::ParameterList MLList;
  ML_Epetra::SetDefaults("SA",MLList);
  MLList.set("ML output", 0);
  MLList.set("max levels",5);
  MLList.set("increasing or decreasing","increasing");
  int numEqns;
  if (assembler->cells.size() == 1)
  numEqns = numVars[0];
  else
  numEqns = 1;
  
  MLList.set("PDE equations",numEqns);
  MLList.set("aggregation: type", "Uncoupled");
  MLList.set("smoother: type","IFPACK");
  MLList.set("smoother: sweeps",1);
  MLList.set("smoother: ifpack type","ILU");
  MLList.set("smoother: ifpack overlap",1);
  MLList.set("smoother: pre or post", "both");
  MLList.set("coarse: type","Amesos-KLU");
  ML_Epetra::MultiLevelPreconditioner* MLPrec =
  new ML_Epetra::MultiLevelPreconditioner(*J, MLList);
  
  return MLPrec;
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
  vector_RCP F_soln = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,numsteps+1)); // empty solution
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
    
    multiscale_manager->setMacroInfo(disc->basis_pointers, disc->basis_types,
                                     phys->varlist, useBasis, phys->offsets,
                                     params->paramnames, params->discretized_param_names);
    
    multiscale_manager->macro_wkset = assembler->wkset;
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
