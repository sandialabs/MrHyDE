/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
/* Minimal constructor to set up the problem */
// ========================================================================================

template <class Node>
PostprocessManager<Node>::PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                                             Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                             Teuchos::RCP<MeshInterface> & mesh_,
                                             Teuchos::RCP<DiscretizationInterface> & disc_,
                                             Teuchos::RCP<PhysicsInterface> & phys_,
                                             Teuchos::RCP<AssemblyManager<Node>> & assembler_) : Comm(Comm_), mesh(mesh_), disc(disc_), physics(phys_),
                                                                                                assembler(assembler_), settings(settings_)
{
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PostprocessManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);

  this->setup();
}

// ========================================================================================
/* Full constructor to set up the problem */
// ========================================================================================

template <class Node>
PostprocessManager<Node>::PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                                             Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                             Teuchos::RCP<MeshInterface> & mesh_,
                                             Teuchos::RCP<DiscretizationInterface> & disc_,
                                             Teuchos::RCP<PhysicsInterface> & phys_,
                                             Teuchos::RCP<MultiscaleManager> & multiscale_manager_,
                                             Teuchos::RCP<AssemblyManager<Node>> & assembler_,
                                             Teuchos::RCP<ParameterManager<Node>> & params_) : Comm(Comm_), mesh(mesh_), disc(disc_), physics(phys_),
                                                                                              assembler(assembler_), params(params_), multiscale_manager(multiscale_manager_), settings(settings_)
{
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PostprocessManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);

  this->setup();
#if defined(MrHyDE_ENABLE_HDSA)
  hdsa_solop = false;
#endif
}

// ========================================================================================
// Setup function used by different constructors
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::setup() {
  
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level", 0), Comm));
  
  debugger->print("**** Starting PostprocessManager::setup()");
  
  ////////////////////////////////////////////
  // Grab flags from settings parameter list
  
  verbosity = settings->get<int>("verbosity", 1);
  
  compute_response = settings->sublist("Postprocess").get<bool>("compute responses", false);
  write_response = settings->sublist("Postprocess").get<bool>("write responses", compute_response);
  compute_error = settings->sublist("Postprocess").get<bool>("compute errors", false);
  write_solution = settings->sublist("Postprocess").get("write solution", false);
  write_frequency = settings->sublist("Postprocess").get("write frequency", 1);
  exodus_write_frequency = settings->sublist("Postprocess").get("exodus write frequency", 1);
  write_group_number = settings->sublist("Postprocess").get("write group number", false);
  
  write_subgrid_solution = settings->sublist("Postprocess").get("write subgrid solution", false);
  write_subgrid_model = false;
  if (settings->isSublist("Subgrid")) {
    write_subgrid_model = true;
  }
  write_HFACE_variables = settings->sublist("Postprocess").get("write HFACE variables", false);
  exodus_filename = settings->sublist("Postprocess").get<string>("output file", "output") + ".exo";
  write_optimization_solution = settings->sublist("Postprocess").get("create optimization movie", false);
  compute_objective = settings->sublist("Postprocess").get("compute objective", false);
  compute_objective_grad_param = settings->sublist("Postprocess").get("compute objective grad param", true); // only turn this off if you know what you are doing
  objective_file = settings->sublist("Postprocess").get("objective output file", "");
  objective_grad_file = settings->sublist("Postprocess").get("objective gradient output file", "");
  discrete_objective_scale_factor = settings->sublist("Postprocess").get("scale factor for discrete objective", 1.0);
  cellfield_reduction = settings->sublist("Postprocess").get<string>("extra grp field reduction", "mean");
  write_database_id = settings->sublist("Solver").get<bool>("use basis database", false);
  compute_flux_response = settings->sublist("Postprocess").get("compute flux response", false);
  store_sensor_solution = settings->sublist("Postprocess").get("store sensor solution", false);
  fileoutput = settings->sublist("Postprocess").get("file output format", "text");
  write_qdata = settings->sublist("Postprocess").get("write quadrature data", false);
  write_bqdata = settings->sublist("Postprocess").get("write boundary quadrature data", false);
  
  exodus_record_start = settings->sublist("Postprocess").get("exodus record start time", -DBL_MAX);
  exodus_record_stop = settings->sublist("Postprocess").get("exodus record stop time", DBL_MAX);
  
  record_start = settings->sublist("Postprocess").get("record start time", -DBL_MAX);
  record_stop = settings->sublist("Postprocess").get("record stop time", DBL_MAX);
  
  compute_integrated_quantities = settings->sublist("Postprocess").get("compute integrated quantities", false);
  
  compute_weighted_norm = settings->sublist("Postprocess").get<bool>("compute weighted norm", false);
  
  numNodesPerElem = settings->sublist("Mesh").get<int>("numNodesPerElem", 4); // actually set by mesh interface
  dimension = physics->dimension;
  
  response_type = settings->sublist("Postprocess").get("response type", "pointwise"); // or "global"
  have_sensor_data = settings->sublist("Analysis").get("have sensor data", false);    // or "global"
  save_sensor_data = settings->sublist("Analysis").get("save sensor data", false);
  sname = settings->sublist("Analysis").get("sensor prefix", "sensor");
  
  stddev = settings->sublist("Analysis").get("additive normal noise standard dev", 0.0);
  write_dakota_output = settings->sublist("Postprocess").get("write Dakota output", false);
  
  is_hdsa_analysis = (settings->sublist("Analysis").get("analysis type", "forward") == "HDSA");
  
  // Get a few lists of strings for easy references
  varlist = physics->var_list;
  blocknames = physics->block_names;
  sideSets = physics->side_names;
  setnames = physics->set_names;
  
  ///////////////////////////////////////////////////
  // Create solution storage objects (if required)
  
  for (size_t set = 0; set < setnames.size(); ++set) {
    soln.push_back(Teuchos::rcp(new SolutionStorage<Node>(settings)));
    adj_soln.push_back(Teuchos::rcp(new SolutionStorage<Node>(settings)));
  }
  string analysis_type = settings->sublist("Analysis").get<string>("analysis type", "forward");
  save_solution = false;
  save_adjoint_solution = false; // very rarely is this true
  
  if (analysis_type == "forward+adjoint" || analysis_type == "ROL" || analysis_type == "ROL2" || analysis_type == "ROLStoch" || analysis_type == "ROL_SIMOPT" || analysis_type == "HDSA" || analysis_type == "HDSAStoch") {
    save_solution = true; // default is false
    string rolVersion = "ROL";
    if (analysis_type == "ROL2") {
      rolVersion = analysis_type;
    }
    if (settings->sublist("Analysis").sublist(rolVersion).sublist("General").get<bool>("Generate data", false)) {
      for (size_t set = 0; set < setnames.size(); ++set) {
        datagen_soln.push_back(Teuchos::rcp(new SolutionStorage<Node>(settings)));
      }
    }
  }
  
  ///////////////////////////////////////////////////
  // Default append string for output
  append = "";
  
  ///////////////////////////////////////////////////
  // Writing HFACE variables to exodus requires a fair bit of overhead
  // This is due to the fact that they have no volumetric support, just face/edge
  // So projections or interpolations are required
  // This just warns the user if exodus is enabled and HFACE variables are used,
  // then they need to turn on an extra flag to visualize these
  if (verbosity > 0 && Comm->getRank() == 0) {
    if (write_solution && !write_HFACE_variables) {
      bool have_HFACE_vars = false;
      vector<vector<vector<string>>> types = physics->types;
      for (size_t set = 0; set < types.size(); set++) {
        for (size_t block = 0; block < types[set].size(); ++block) {
          for (size_t var = 0; var < types[set][block].size(); var++) {
            if (types[set][block][var] == "HFACE") {
              have_HFACE_vars = true;
            }
          }
        }
      }
      if (have_HFACE_vars) {
        cout << "**** MrHyDE Warning: Visualization is enabled and at least one HFACE variable was found, but Postprocess-> write_HFACE_variables is set to false." << endl;
      }
    }
  }
  ///////////////////////////////////////////////////
  
  ///////////////////////////////////////////////////
  // Set up the exodus file and informuser what file name is
  
  if (write_solution && Comm->getRank() == 0 && !is_hdsa_analysis) {
    cout << endl
    << "*********************************************************" << endl;
    cout << "***** Writing the solution to " << exodus_filename << endl;
    cout << "*********************************************************" << endl;
  }
  
  isTD = false;
  if (settings->sublist("Solver").get<string>("solver", "steady-state") == "transient") {
    isTD = true;
  }
  
  if (isTD && write_solution && !is_hdsa_analysis) {
    mesh->setupExodusFile(exodus_filename);
  }
  if (write_optimization_solution && !is_hdsa_analysis) {
    mesh->setupOptimizationExodusFile("optimization_" + exodus_filename);
  }
  
  ///////////////////////////////////////////////////
  // Create the lists of postprocessing features that are enabled
  
  for (size_t block = 0; block < blocknames.size(); ++block) {
    
    if (settings->sublist("Postprocess").isSublist("Responses")) {
      Teuchos::ParameterList resps = settings->sublist("Postprocess").sublist("Responses");
      Teuchos::ParameterList::ConstIterator rsp_itr = resps.begin();
      while (rsp_itr != resps.end()) {
        string entry = resps.get<string>(rsp_itr->first);
        assembler->addFunction(block, rsp_itr->first, entry, "ip");
        rsp_itr++;
      }
    }
    if (settings->sublist("Postprocess").isSublist("Weights")) {
      Teuchos::ParameterList wts = settings->sublist("Postprocess").sublist("Weights");
      Teuchos::ParameterList::ConstIterator wts_itr = wts.begin();
      while (wts_itr != wts.end()) {
        string entry = wts.get<string>(wts_itr->first);
        assembler->addFunction(block, wts_itr->first, entry, "ip");
        wts_itr++;
      }
    }
    if (settings->sublist("Postprocess").isSublist("Targets")) {
      Teuchos::ParameterList tgts = settings->sublist("Postprocess").sublist("Targets");
      Teuchos::ParameterList::ConstIterator tgt_itr = tgts.begin();
      while (tgt_itr != tgts.end()) {
        string entry = tgts.get<string>(tgt_itr->first);
        assembler->addFunction(block, tgt_itr->first, entry, "ip");
        tgt_itr++;
      }
    }
    
    Teuchos::ParameterList blockPpSettings;
    if (settings->sublist("Postprocess").isSublist(blocknames[block])) { // adding block overwrites the default
      blockPpSettings = settings->sublist("Postprocess").sublist(blocknames[block]);
    }
    else { // default
      blockPpSettings = settings->sublist("Postprocess");
    }
    vector<vector<vector<string>>> types = physics->types;
    
    // Add true solutions to the function manager for verification studies
    Teuchos::ParameterList true_solns = blockPpSettings.sublist("True solutions");
    
    vector<std::pair<string, string>> block_error_list = this->addTrueSolutions(true_solns, types, block);
    error_list.push_back(block_error_list);
    
    // Add extra fields
    vector<string> block_ef;
    Teuchos::ParameterList efields = blockPpSettings.sublist("Extra fields");
    Teuchos::ParameterList::ConstIterator ef_itr = efields.begin();
    while (ef_itr != efields.end()) {
      string entry = efields.get<string>(ef_itr->first);
      block_ef.push_back(ef_itr->first);
      assembler->addFunction(block, ef_itr->first, entry, "ip");
      assembler->addFunction(block, ef_itr->first, entry, "point");
      ef_itr++;
    }
    extrafields_list.push_back(block_ef);
    
    // Add extra grp fields
    vector<string> block_ecf;
    Teuchos::ParameterList ecfields = blockPpSettings.sublist("Extra cell fields");
    Teuchos::ParameterList::ConstIterator ecf_itr = ecfields.begin();
    while (ecf_itr != ecfields.end()) {
      string entry = ecfields.get<string>(ecf_itr->first);
      block_ecf.push_back(ecf_itr->first);
      assembler->addFunction(block, ecf_itr->first, entry, "ip");
      ecf_itr++;
    }
    extracellfields_list.push_back(block_ecf);
    
    // Add derived quantities from physics modules
    vector<string> block_dq;
    for (size_t set = 0; set < physics->modules.size(); ++set) {
      for (size_t m = 0; m < physics->modules[set][block].size(); ++m) {
        vector<string> dqnames = physics->modules[set][block][m]->getDerivedNames();
        for (size_t k = 0; k < dqnames.size(); ++k) {
          block_dq.push_back(dqnames[k]);
        }
      }
    }
    derivedquantities_list.push_back(block_dq);
    
    // Add objective functions
    Teuchos::ParameterList obj_funs = blockPpSettings.sublist("Objective functions");
    this->addObjectiveFunctions(obj_funs, block);
    
    // Add flux responses (special case of responses)
    Teuchos::ParameterList flux_resp = blockPpSettings.sublist("Flux responses");
    this->addFluxResponses(flux_resp, block);
    
    // Add integrated quantities required by physics module
    // This needs to happen before we read in integrated quantities from the input file
    // to ensure proper ordering of the QoI
    vector<integratedQuantity> block_IQs;
    for (size_t set = 0; set < physics->modules.size(); ++set) {
      for (size_t m = 0; m < physics->modules[set][block].size(); ++m) {
        vector<vector<string>> integrandsNamesAndTypes =
        physics->modules[set][block][m]->setupIntegratedQuantities(dimension);
        vector<integratedQuantity> phys_IQs =
        this->addIntegratedQuantities(integrandsNamesAndTypes, block);
        // add the IQs from this physics to the "running total"
        block_IQs.insert(end(block_IQs), begin(phys_IQs), end(phys_IQs));
      }
    }
    
    // if the physics module requested IQs, make sure the compute flag is set to true
    // workset storage occurs in the physics module (setWorkset)
    if (block_IQs.size() > 0) {
      // BWR -- there is an edge case where the user will say false in the input file
      // but ALSO have IQs defined. This could potentially end up grabbing some of those anyway...
      compute_integrated_quantities = true;
    }
    
    // Add integrated quantities from input file
    Teuchos::ParameterList iqs = blockPpSettings.sublist("Integrated quantities");
    vector<integratedQuantity> user_IQs = this->addIntegratedQuantities(iqs, block);
    
    // add the IQs from the input file to the "running total"
    block_IQs.insert(end(block_IQs), begin(user_IQs), end(user_IQs));
    
    // finalize IQ bookkeeping, IQs are stored block by block
    // only want to do this if we actually are computing something
    if (block_IQs.size() > 0)
      integratedQuantities.push_back(block_IQs);
    
  } // end block loop
  ///////////////////////////////////////////////////
  
  // Add sensor data to objectives
  this->addSensors();
  
#if defined(MrHyDE_ENABLE_FFTW)
  fft = Teuchos::rcp(new fftInterface());
#endif
  
  debugger->print("**** Finished PostprocessManager::setup()");
  
}
  // ========================================================================================
  // ========================================================================================

template <class Node>
void PostprocessManager<Node>::completeSetup() {
  debugger->print("**** Starting PostprocessManager::completeSetup()");
  // Meeds to happen here because ip are not defined when constructor is called
  
  // Write quadrature points and weights to file if requested
  // This is useful if one want to use these as sensors
  if (write_qdata) {
    this->writeQuadratureData();
  }
  
  if (write_bqdata) {
    this->writeBoundaryQuadratureData();
  }
  
  debugger->print("**** Finished PostprocessManager::completeSetup()");
}

