/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

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
  
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), comm));
  
  debugger->print("**** Starting linear algebra interface constructor ...");
  
  verbosity = settings->get<int>("verbosity",0);
  
  setnames = disc->physics->set_names;
  
  // Generic Belos Settings - can be overridden by defining Belos sublists
  linearTOL = settings->sublist("Solver").get<double>("linear TOL",1.0E-7);
  doCondEst = settings->sublist("Solver").get<bool>("Estimate Condition Number",false);
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
  
  debugger->print("**** Finished linear algebra interface constructor");
  
}

// ========================================================================================
// Set up the Tpetra objects (maps, importers, exporters and graphs)
// This is a separate function call in case it needs to be recomputed
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::setupLinearAlgebra() {
  
  Teuchos::TimeMonitor localtimer(*setupLAtimer);
  
  debugger->print("**** Starting solver::setupLinearAlgebraInterface...");
  
  std::vector<string> blocknames = disc->block_names;
  
  // --------------------------------------------------
  // primary variable LA objects
  // --------------------------------------------------
  max_entries = 0;
  
  for (size_t set=0; set<setnames.size(); ++set) {
    
    //auto owned = disc->dof_owned[set];
    //auto ownedAndShared = disc->dof_owned_and_shared[set];
    
    LO numUnknowns = (LO)disc->dof_owned[set].extent(0);
    //LO numUnknowns = (LO)owned.size();
    GO localNumUnknowns = numUnknowns;
    GO globalNumUnknowns = 0;

    Teuchos::reduceAll<LO,GO>(*comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
    
    owned_map.push_back(Teuchos::rcp(new LA_Map(globalNumUnknowns, disc->dof_owned[set], 0, comm)));
    
    bool allocate_matrices = true;
    if (settings->sublist("Solver").get<bool>("fully explicit",false) && settings->sublist("Solver").get<bool>("matrix free",false) ) {
      allocate_matrices = false;
    }
    have_overlapped = true;
    if (!allocate_matrices && comm->getSize() == 1) {
      have_overlapped = false;
    }
    if (have_overlapped) {
      overlapped_map.push_back(Teuchos::rcp(new LA_Map(globalNumUnknowns, disc->dof_owned_and_shared[set], 0, comm)));
      if (!allocate_matrices) {
        disc->dof_owned_and_shared[set] = Kokkos::View<GO*>("empty dof",1);
      }
      exporter.push_back(Teuchos::rcp(new LA_Export(overlapped_map[set], owned_map[set])));
      importer.push_back(Teuchos::rcp(new LA_Import(owned_map[set], overlapped_map[set])));
    }
    if (!allocate_matrices) {
      disc->dof_owned[set] = Kokkos::View<GO*>("empty dof",1);
    }
    
    if (allocate_matrices) {
      vector<size_t> max_entriesPerRow(overlapped_map[set]->getLocalNumElements(), 0);
      for (size_t b=0; b<blocknames.size(); b++) {
        auto EIDs = disc->my_elements[b];
        for (size_t e=0; e<EIDs.extent(0); e++) {
          size_t elemID = EIDs(e);
          vector<GO> gids = disc->getGIDs(set,b,elemID); //
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
        auto EIDs = disc->my_elements[b];
        for (size_t e=0; e<EIDs.extent(0); e++) {
          size_t elemID = EIDs(e);
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
  // discretized parameter/state LA objects
  // --------------------------------------------------
  
  if (params->num_discretized_params > 0) {
    
    // Maps, importers, and exporters
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
    
    // Param/state LA objects
    // TMW: warning - this is hard coded to one physics set
    for (size_t set=0; set<setnames.size(); ++set) {
      vector<size_t> max_entriesPerRow(param_overlapped_map->getLocalNumElements(), 0);
      for (size_t b=0; b<blocknames.size(); b++) {
        auto EIDs = disc->my_elements[b];
        for (size_t e=0; e<EIDs.extent(0); e++) {
          size_t elemID = EIDs(e);
          vector<GO> gids;
          params->paramDOF->getElementGIDs(elemID, gids, blocknames[b]);
          vector<GO> stategids = disc->getGIDs(set,b,elemID);
          for (size_t i=0; i<gids.size(); i++) {
            LO ind1 = param_overlapped_map->getLocalElement(gids[i]);
            max_entriesPerRow[ind1] += stategids.size();
          }
        }
      }
      
      for (size_t m=0; m<max_entriesPerRow.size(); ++m) {
        max_entries = std::max(max_entries, max_entriesPerRow[m]);
      }
      
      paramstate_overlapped_graph.push_back(Teuchos::rcp( new LA_CrsGraph(param_overlapped_map, overlapped_map[set], max_entries)));
      for (size_t b=0; b<blocknames.size(); b++) {
        auto EIDs = disc->my_elements[b];
        for (size_t e=0; e<EIDs.extent(0); e++) {
          vector<GO> gids;
          size_t elemID = EIDs(e);
          params->paramDOF->getElementGIDs(elemID, gids, blocknames[b]);
          vector<GO> stategids = disc->getGIDs(set,b,elemID);
          for (size_t i=0; i<gids.size(); i++) {
            GO ind1 = gids[i];
            paramstate_overlapped_graph[set]->insertGlobalIndices(ind1,stategids);
          }
        }
      }
      
      paramstate_overlapped_graph[set]->fillComplete(owned_map[set], param_owned_map); // hard coded
    }

    // --------------------------------------------------
    // discretized parameter LA objects
    // --------------------------------------------------
    
    vector<size_t> max_entriesPerRow(param_overlapped_map->getLocalNumElements(), 0);
    for (size_t b=0; b<blocknames.size(); b++) {
      auto EIDs = disc->my_elements[b];
      for (size_t e=0; e<EIDs.extent(0); e++) {
        size_t elemID = EIDs(e);
        vector<GO> gids;
        params->paramDOF->getElementGIDs(elemID, gids, blocknames[b]);
        for (size_t i=0; i<gids.size(); i++) {
          LO ind1 = param_overlapped_map->getLocalElement(gids[i]);
          max_entriesPerRow[ind1] += gids.size();
        }
      }
    }
      
    for (size_t m=0; m<max_entriesPerRow.size(); ++m) {
      max_entries = std::max(max_entries, max_entriesPerRow[m]);
    }
    
    param_overlapped_graph = Teuchos::rcp( new LA_CrsGraph(param_overlapped_map, max_entries));
    for (size_t b=0; b<blocknames.size(); b++) {
      auto EIDs = disc->my_elements[b];
      for (size_t e=0; e<EIDs.extent(0); e++) {
        vector<GO> gids;
        size_t elemID = EIDs(e);
        params->paramDOF->getElementGIDs(elemID, gids, blocknames[b]);
        for (size_t i=0; i<gids.size(); i++) {
          GO ind1 = gids[i];
          param_overlapped_graph->insertGlobalIndices(ind1,gids);
        }
      }
    }
    
    param_overlapped_graph->fillComplete(); // hard coded
    
  }

  debugger->print("**** Finished solver::setupLinearAlgebraInterface");
  
}

