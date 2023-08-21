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

#include "postprocessManager.hpp"
#include "hdf5.h"

using namespace MrHyDE;

// ========================================================================================
/* Minimal constructor to set up the problem */
// ========================================================================================

template<class Node>
PostprocessManager<Node>::PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                                             Teuchos::RCP<Teuchos::ParameterList> & settings,
                                             Teuchos::RCP<MeshInterface> & mesh_,
                                             Teuchos::RCP<DiscretizationInterface> & disc_,
                                             Teuchos::RCP<PhysicsInterface> & phys_,
                                             Teuchos::RCP<AssemblyManager<Node> > & assembler_) :
Comm(Comm_), mesh(mesh_), disc(disc_), physics(phys_),
assembler(assembler_) { 
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PostprocessManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  this->setup(settings);
}

// ========================================================================================
/* Full constructor to set up the problem */
// ========================================================================================

template<class Node>
PostprocessManager<Node>::PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                                             Teuchos::RCP<Teuchos::ParameterList> & settings,
                                             Teuchos::RCP<MeshInterface> & mesh_,
                                             Teuchos::RCP<DiscretizationInterface> & disc_,
                                             Teuchos::RCP<PhysicsInterface> & phys_,
                                             Teuchos::RCP<MultiscaleManager> & multiscale_manager_,
                                             Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                                             Teuchos::RCP<ParameterManager<Node> > & params_) :
Comm(Comm_), mesh(mesh_), disc(disc_), physics(phys_),
assembler(assembler_), params(params_), multiscale_manager(multiscale_manager_) { 
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PostprocessManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  this->setup(settings);
}

// ========================================================================================
// Setup function used by different constructors
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::setup(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0 && Comm->getRank() == 0) {
    cout << "**** Starting PostprocessManager::setup()" << endl;
  }
  
  //objectiveval = 0.0;
  
  verbosity = settings->get<int>("verbosity",1);
  
  compute_response = settings->sublist("Postprocess").get<bool>("compute responses",false);
  compute_error = settings->sublist("Postprocess").get<bool>("compute errors",false);
  write_solution = settings->sublist("Postprocess").get("write solution",false);
  write_frequency = settings->sublist("Postprocess").get("write frequency",1);
  exodus_write_frequency = settings->sublist("Postprocess").get("exodus write frequency",1);
  write_group_number = settings->sublist("Postprocess").get("write group number",false);

  write_subgrid_solution = settings->sublist("Postprocess").get("write subgrid solution",false);
  write_subgrid_model = false;
  if (settings->isSublist("Subgrid")) {
    write_subgrid_model = true;
  }
  write_HFACE_variables = settings->sublist("Postprocess").get("write HFACE variables",false);
  exodus_filename = settings->sublist("Postprocess").get<string>("output file","output")+".exo";
  write_optimization_solution = settings->sublist("Postprocess").get("create optimization movie",false);
  compute_objective = settings->sublist("Postprocess").get("compute objective",false);
  objective_file = settings->sublist("Postprocess").get("objective output file", "");
  objective_grad_file = settings->sublist("Postprocess").get("objective gradient output file", "");
  discrete_objective_scale_factor = settings->sublist("Postprocess").get("scale factor for discrete objective",1.0);
  cellfield_reduction = settings->sublist("Postprocess").get<string>("extra grp field reduction","mean");
  write_database_id = settings->sublist("Solver").get<bool>("use basis database",false);
  compute_flux_response = settings->sublist("Postprocess").get("compute flux response",false);
  store_sensor_solution = settings->sublist("Postprocess").get("store sensor solution",false);
  fileoutput = settings->sublist("Postprocess").get("file output format","text");

  exodus_record_start = settings->sublist("Postprocess").get("exodus record start time",-DBL_MAX);
  exodus_record_stop = settings->sublist("Postprocess").get("exodus record stop time",DBL_MAX);

  record_start = settings->sublist("Postprocess").get("record start time",-DBL_MAX);
  record_stop = settings->sublist("Postprocess").get("record stop time",DBL_MAX);

  compute_integrated_quantities = settings->sublist("Postprocess").get("compute integrated quantities",false);
 
  compute_weighted_norm = settings->sublist("Postprocess").get<bool>("compute weighted norm",false);
  
  setnames = physics->set_names;
  
  for (size_t set=0; set<setnames.size(); ++set) {
    soln.push_back(Teuchos::rcp(new SolutionStorage<Node>(settings)));
    adj_soln.push_back(Teuchos::rcp(new SolutionStorage<Node>(settings)));
  }
  string analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  save_solution = false;
  save_adjoint_solution = false; // very rarely is this true

  if (analysis_type == "forward+adjoint" || analysis_type == "ROL" || analysis_type == "ROL2" || analysis_type == "ROL_SIMOPT") {
    save_solution = true; // default is false
    string rolVersion = "ROL";
    if (analysis_type == "ROL2") {
      rolVersion = analysis_type;
    }
    if (settings->sublist("Analysis").sublist(rolVersion).sublist("General").get<bool>("Generate data",false)) {
      for (size_t set=0; set<setnames.size(); ++set) {
        datagen_soln.push_back(Teuchos::rcp(new SolutionStorage<Node>(settings)));
      }
    }
  }
  
  append = "";
  
  if (verbosity > 0 && Comm->getRank() == 0) {
    if (write_solution && !write_HFACE_variables) {
      bool have_HFACE_vars = false;
      vector<vector<vector<string> > > types = physics->types;
      for (size_t set=0; set<types.size(); set++) {
        for (size_t block=0; block<types[set].size(); ++block) {
          for (size_t var=0; var<types[set][block].size(); var++) {
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
  
  if (write_solution && Comm->getRank() == 0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Writing the solution to " << exodus_filename << endl;
    cout << "*********************************************************" << endl;
  }
  
  isTD = false;
  if (settings->sublist("Solver").get<string>("solver","steady-state") == "transient") {
    isTD = true;
  }
  
  if (isTD && write_solution) {
    mesh->stk_mesh->setupExodusFile(exodus_filename);
  }
  if (write_optimization_solution) {
    mesh->stk_optimization_mesh->setupExodusFile("optimization_"+exodus_filename);
  }
  
  //mesh->stk_mesh->getElementBlockNames(blocknames);
  //mesh->stk_mesh->getSidesetNames(sideSets);
  blocknames = physics->block_names;
  sideSets = physics->side_names;
  
  numNodesPerElem = settings->sublist("Mesh").get<int>("numNodesPerElem",4); // actually set by mesh interface
  dimension = physics->dimension;//mesh->stk_mesh->getDimension();
    
  response_type = settings->sublist("Postprocess").get("response type", "pointwise"); // or "global"
  have_sensor_data = settings->sublist("Analysis").get("have sensor data", false); // or "global"
  save_sensor_data = settings->sublist("Analysis").get("save sensor data",false);
  sname = settings->sublist("Analysis").get("sensor prefix","sensor");
    
  stddev = settings->sublist("Analysis").get("additive normal noise standard dev",0.0);
  write_dakota_output = settings->sublist("Postprocess").get("write Dakota output",false);
  
  varlist = physics->var_list;
  
  for (size_t block=0; block<blocknames.size(); ++block) {
        
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
    vector<vector<vector<string> > > types = physics->types;
    
    // Add true solutions to the function manager for verification studies
    Teuchos::ParameterList true_solns = blockPpSettings.sublist("True solutions");
    
    vector<std::pair<string,string> > block_error_list = this->addTrueSolutions(true_solns, types, block);
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
    for (size_t set=0; set<physics->modules.size(); ++set) {
      for (size_t m=0; m<physics->modules[set][block].size(); ++m) {
        vector<string> dqnames = physics->modules[set][block][m]->getDerivedNames();
        for (size_t k=0; k<dqnames.size(); ++k) {
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
    for (size_t set=0; set<physics->modules.size(); ++set) {
      for (size_t m=0; m<physics->modules[set][block].size(); ++m) {
        vector< vector<string> > integrandsNamesAndTypes =
        physics->modules[set][block][m]->setupIntegratedQuantities(dimension);
        vector<integratedQuantity> phys_IQs =
        this->addIntegratedQuantities(integrandsNamesAndTypes, block);
        // add the IQs from this physics to the "running total"
        block_IQs.insert(end(block_IQs),begin(phys_IQs),end(phys_IQs));
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
    block_IQs.insert(end(block_IQs),begin(user_IQs),end(user_IQs));

    // finalize IQ bookkeeping, IQs are stored block by block
    // only want to do this if we actually are computing something
    if (block_IQs.size() > 0) integratedQuantities.push_back(block_IQs);

  } // end block loop

  // Add sensor data to objectives
  this->addSensors();
  

#if defined(MrHyDE_ENABLE_FFTW)
  fft = Teuchos::rcp(new fftInterface());
#endif


  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished PostprocessManager::setup()" << endl;
    }
  }
    
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<std::pair<string,string> > PostprocessManager<Node>::addTrueSolutions(Teuchos::ParameterList & true_solns,
                                                                             vector<vector<vector<string> > > & types,
                                                                             const int & block) {
  vector<std::pair<string,string> > block_error_list;
  for (size_t set=0; set<varlist.size(); ++set) {
    vector<string> vars = varlist[set][block];
    vector<string> ctypes = types[set][block];
    for (size_t j=0; j<vars.size(); j++) {
      if (true_solns.isParameter(vars[j])) { // solution at volumetric ip
        if (ctypes[j] == "HGRAD" || ctypes[j] == "HVOL") {
          std::pair<string,string> newerr(vars[j],"L2");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>(vars[j],"0.0");
          assembler->addFunction(block, "true "+vars[j], expression, "ip");
        }
      }
      if (true_solns.isParameter("grad("+vars[j]+")[x]") || true_solns.isParameter("grad("+vars[j]+")[y]") || true_solns.isParameter("grad("+vars[j]+")[z]")) { // GRAD of the solution at volumetric ip
        if (ctypes[j] == "HGRAD") {
          std::pair<string,string> newerr(vars[j],"GRAD");
          block_error_list.push_back(newerr);
          
          string expression = true_solns.get<string>("grad("+vars[j]+")[x]","0.0");
          assembler->addFunction(block, "true grad("+vars[j]+")[x]", expression, "ip");
          if (dimension>1) {
            expression = true_solns.get<string>("grad("+vars[j]+")[y]","0.0");
            assembler->addFunction(block, "true grad("+vars[j]+")[y]", expression, "ip");
          }
          if (dimension>2) {
            expression = true_solns.get<string>("grad("+vars[j]+")[z]","0.0");
            assembler->addFunction(block, "true grad("+vars[j]+")[z]", expression, "ip");
          }
        }
      }
      if (true_solns.isParameter(vars[j]+" face")) { // solution at face/side ip
        if (ctypes[j] == "HGRAD" || ctypes[j] == "HFACE") {
          std::pair<string,string> newerr(vars[j],"L2 FACE");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>(vars[j]+" face","0.0");
          assembler->addFunction(block, "true "+vars[j], expression, "side ip");
          
        }
      }
      if (true_solns.isParameter(vars[j]+"[x]") || true_solns.isParameter(vars[j]+"[y]") || true_solns.isParameter(vars[j]+"[z]")) { // vector solution at volumetric ip
        if (ctypes[j] == "HDIV" || ctypes[j] == "HCURL") {
          std::pair<string,string> newerr(vars[j],"L2 VECTOR");
          block_error_list.push_back(newerr);
          
          string expression = true_solns.get<string>(vars[j]+"[x]","0.0");
          assembler->addFunction(block, "true "+vars[j]+"[x]", expression, "ip");
          
          if (dimension>1) {
            expression = true_solns.get<string>(vars[j]+"[y]","0.0");
            assembler->addFunction(block, "true "+vars[j]+"[y]", expression, "ip");
          }
          if (dimension>2) {
            expression = true_solns.get<string>(vars[j]+"[z]","0.0");
            assembler->addFunction(block, "true "+vars[j]+"[z]", expression, "ip");
          }
        }
      }
      if (true_solns.isParameter("div("+vars[j]+")")) { // div of solution at volumetric ip
        if (ctypes[j] == "HDIV") {
          std::pair<string,string> newerr(vars[j],"DIV");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>("div("+vars[j]+")","0.0");
          assembler->addFunction(block, "true div("+vars[j]+")", expression, "ip");
          
        }
      }
      if (true_solns.isParameter("curl("+vars[j]+")[x]") || true_solns.isParameter("curl("+vars[j]+")[y]") || true_solns.isParameter("curl("+vars[j]+")[z]")) { // vector solution at volumetric ip
        if (ctypes[j] == "HCURL") {
          std::pair<string,string> newerr(vars[j],"CURL");
          block_error_list.push_back(newerr);
          
          string expression = true_solns.get<string>("curl("+vars[j]+")[x]","0.0");
          assembler->addFunction(block, "true curl("+vars[j]+")[x]", expression, "ip");
          
          if (dimension>1) {
            expression = true_solns.get<string>("curl("+vars[j]+")[y]","0.0");
            assembler->addFunction(block, "true curl("+vars[j]+")[y]", expression, "ip");
          }
          if (dimension>2) {
            expression = true_solns.get<string>("curl("+vars[j]+")[z]","0.0");
            assembler->addFunction(block, "true curl("+vars[j]+")[z]", expression, "ip");
          }
        }
      }
    }
  }
  return block_error_list;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::addObjectiveFunctions(Teuchos::ParameterList & obj_funs,
                                                     const size_t & block) {
  Teuchos::ParameterList::ConstIterator obj_itr = obj_funs.begin();
  while (obj_itr != obj_funs.end()) {
    Teuchos::ParameterList objsettings = obj_funs.sublist(obj_itr->first);
    bool addobj = true;
    if (objsettings.isParameter("blocks")) {
      string blocklist = objsettings.get<string>("blocks");
      std::size_t found = blocklist.find(blocknames[block]);
      if (found == std::string::npos) {
        addobj = false;
      }
    }
    if (addobj) {
      objective newobj(objsettings,obj_itr->first,block);
      objectives.push_back(newobj);

      if (newobj.type == "sensors") {
        assembler->addFunction(block, newobj.name+" response", newobj.response,"point");
      }
      else if (newobj.type == "integrated response") {
        assembler->addFunction(block, newobj.name+" response", newobj.response,"ip");
      }
      else if (newobj.type == "integrated control") {
        assembler->addFunction(block, newobj.name, newobj.function, "ip");
      }

      // regularizations
      for (size_t r=0; r<newobj.regularizations.size(); ++r) {
        if (newobj.regularizations[r].type == "integrated") {
          if (newobj.regularizations[r].location == "volume") {
            assembler->addFunction(block, newobj.regularizations[r].name, newobj.regularizations[r].function, "ip");
          }
          else if (newobj.regularizations[r].location == "boundary") {
            assembler->addFunction(block, newobj.regularizations[r].name, newobj.regularizations[r].function, "side ip");
          }
        }
      }
    }
    obj_itr++;
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::addFluxResponses(Teuchos::ParameterList & flux_resp,
                                                const size_t & block) {
  Teuchos::ParameterList::ConstIterator fluxr_itr = flux_resp.begin();
  while (fluxr_itr != flux_resp.end()) {
    Teuchos::ParameterList frsettings = flux_resp.sublist(fluxr_itr->first);
    fluxResponse newflux(frsettings,fluxr_itr->first,block);
    fluxes.push_back(newflux);
    assembler->addFunction(block, "flux weight "+newflux.name, newflux.weight, "side ip");
    fluxr_itr++;
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<integratedQuantity> PostprocessManager<Node>::addIntegratedQuantities(Teuchos::ParameterList & iqs,
                                                                             const size_t & block) {
  vector<integratedQuantity> IQs;
  Teuchos::ParameterList::ConstIterator iqs_itr = iqs.begin();
  while (iqs_itr != iqs.end()) {
    Teuchos::ParameterList iqsettings = iqs.sublist(iqs_itr->first);
    integratedQuantity newIQ(iqsettings,iqs_itr->first,block);
    IQs.push_back(newIQ);
    if (newIQ.location == "volume") {
      assembler->addFunction(block, newIQ.name+" integrand", newIQ.integrand, "ip"); 
    } 
    else if (newIQ.location == "boundary") {
      assembler->addFunction(block, newIQ.name+" integrand", newIQ.integrand, "side ip");
    }

    iqs_itr++;
  }
  
  return IQs;

}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<integratedQuantity> 
PostprocessManager<Node>::addIntegratedQuantities(vector< vector<string> > & integrandsNamesAndTypes, 
                                                  const size_t & block) {

  vector<integratedQuantity> IQs; 

  // first index is QoI, second index is 0 for integrand, 1 for name, 2 for type
  for (size_t iIQ=0; iIQ<integrandsNamesAndTypes.size(); ++iIQ) {
    integratedQuantity newIQ(integrandsNamesAndTypes[iIQ][0],
                             integrandsNamesAndTypes[iIQ][1],
                             integrandsNamesAndTypes[iIQ][2],
                             block);
    IQs.push_back(newIQ);
    if (newIQ.location == "volume") {
      assembler->addFunction(block, newIQ.name+" integrand", newIQ.integrand, "ip"); 
    } 
    else if (newIQ.location == "boundary") {
      assembler->addFunction(block, newIQ.name+" integrand", newIQ.integrand, "side ip");
    }

  }
  
  return IQs;

}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::record(vector<vector_RCP> & current_soln, const ScalarT & current_time,
                                      const int & stepnum, DFAD & objectiveval) {
  
  bool write_this_step = false;
  if (stepnum % write_frequency == 0) {
    write_this_step = true;
  }

  bool write_exodus_this_step = false;
  if (stepnum % exodus_write_frequency == 0) {
    write_exodus_this_step = true;
  }
  
  if (write_exodus_this_step && current_time+1.0e-100 >= exodus_record_start && current_time-1.0e-100 <= exodus_record_stop) {
    if (write_solution) {
      this->writeSolution(current_time);
    }
  }
  if (write_this_step && current_time+1.0e-100 >= record_start && current_time-1.0e-100 <= record_stop) {
    if (compute_error) {
      this->computeError(current_time);
    }
    if (compute_response || compute_objective) {
      this->computeObjective(current_soln, current_time, objectiveval);
    }
    if (compute_flux_response) {
      this->computeFluxResponse(current_time);
    }
    if (compute_integrated_quantities) {
      this->computeIntegratedQuantities(current_time);
    }
    if (compute_weighted_norm) {
      this->computeWeightedNorm(current_soln);
    }
    if (store_sensor_solution) {
      this->computeSensorSolution(current_soln, current_time);
    }
  }
  if (save_solution) {
    for (size_t set=0; set<soln.size(); ++set) {
      soln[set]->store(current_soln[set], current_time, 0);
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::report() {
  
  Teuchos::TimeMonitor localtimer(*reportTimer);

  ////////////////////////////////////////////////////////////////////////////
  // Report the responses
  ////////////////////////////////////////////////////////////////////////////
  
  if (compute_response) {
    
    if(Comm->getRank() == 0 ) {
      if (verbosity > 0) {
        cout << endl << "*********************************************************" << endl;
        cout << "***** Computing Responses ******" << endl;
        cout << "*********************************************************" << endl;
      }
    }
    //Kokkos::Timer timer;
    //timer.reset();

    for (size_t obj=0; obj<objectives.size(); ++obj) {
      if (objectives[obj].type == "sensors") {
        if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln) {
        
          Kokkos::View<ScalarT***,HostDevice> sensor_data;
          Kokkos::View<int*,HostDevice> sensorIDs;
          size_t numtimes = 0;
          int numfields = 0;

          //timer.reset();
          int numsensors = objectives[obj].numSensors;
           
          if (numsensors>0) {
            sensorIDs = Kokkos::View<int*,HostDevice>("sensor IDs owned by proc", numsensors);
            size_t sprog=0;
            auto sensor_found = objectives[obj].sensor_found;
            for (size_type s=0; s<sensor_found.extent(0); ++s) {
              if (sensor_found(s)) {
                sensorIDs(sprog) = s;
                ++sprog;
              }
            }
            if (objectives[obj].output_type == "dft") {
              auto dft_data = objectives[obj].sensor_solution_dft;
              size_type numfreq = dft_data.extent(3);
              int numsols = objectives[obj].sensor_solution_data[0].extent_int(1); // does assume this does not change in time, which it shouldn't
              int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
              numfields = numsols*numdims;
              sensor_data = Kokkos::View<ScalarT***,HostDevice>("sensor data",numsensors, numfields, numfreq);
              for (size_t t=0; t<numfreq; ++t) {
                for (int sens=0; sens<numsensors; ++sens) {
                  size_t solprog = 0;
                  for (int sol=0; sol<numsols; ++sol) {
                    for (int d=0; d<numdims; ++d) {
                      sensor_data(sens,solprog,t) = dft_data(sens,sol,d,t).real();
                      solprog++;
                    }
                  }
                }
              }
            }
            else {
              numtimes = objectives[obj].sensor_solution_data.size(); //vector of Kokkos::Views
              int numsols = objectives[obj].sensor_solution_data[0].extent_int(1); // does assume this does not change in time, which it shouldn't
              int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
              numfields = numsols*numdims;
              sensor_data = Kokkos::View<ScalarT***,HostDevice>("sensor data",numsensors, numfields, numtimes);
              for (size_t t=0; t<numtimes; ++t) {
                auto sdat = objectives[obj].sensor_solution_data[t];
                for (int sens=0; sens<numsensors; ++sens) {
                  size_t solprog = 0;
                  for (int sol=0; sol<numsols; ++sol) {
                    for (int d=0; d<numdims; ++d) {
                      sensor_data(sens,solprog,t) = sdat(sens,sol,d);
                      solprog++;
                    }
                  }
                }
              }
              if (objectives[obj].output_type == "fft") {
#if defined(MrHyDE_ENABLE_FFTW)
                fft->compute(sensor_data, sensorIDs, global_num_sensors);
#endif
              } 
            }     
          }
      
          size_t max_numtimes = 0;
          Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&numtimes,&max_numtimes);
          
          int max_numfields = 0;
          Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&numfields,&max_numfields);
          
          if (fileoutput == "text") {
            
            for (int field = 0; field<max_numfields; ++field) {
              std::stringstream ss;
              size_t blocknum = objectives[obj].block;
              ss << field;
              string respfile = "sensor_solution_field." + ss.str() + "." + blocknames[blocknum] + ".out";
              std::ofstream respOUT;
              if (Comm->getRank() == 0) {
                bool is_open = false;
                int attempts = 0;
                int max_attempts = 100;
                while (!is_open && attempts < max_attempts) {
                  respOUT.open(respfile);
                  is_open = respOUT.is_open();
                  attempts++;
                }
                respOUT.precision(8);
                Teuchos::Array<ScalarT> time_data(max_numtimes+dimension,0.0);
                for (size_t dim=0; dim<dimension; ++dim) {
                  time_data[dim] = 0.0;
                }

                for (size_t tt=0; tt<max_numtimes; ++tt) {
                  time_data[tt+dimension] = objectives[obj].response_times[tt];
                }
              
                for (size_t tt=0; tt<max_numtimes+dimension; ++tt) {
                  respOUT << time_data[tt] << "  ";
                }
                respOUT << endl;
              }
              
              auto spts = objectives[obj].sensor_points;
              for (size_t ss=0; ss<objectives[obj].sensor_found.size(); ++ss) {
                Teuchos::Array<ScalarT> series_data(max_numtimes+dimension,0.0);
                Teuchos::Array<ScalarT> gseries_data(max_numtimes+dimension,0.0);
                if (objectives[obj].sensor_found[ss]) {
                  size_t sindex = 0;
                  for (size_t j=0; j<ss; ++j) {
                    if (objectives[obj].sensor_found(j)) {
                      sindex++;
                    }
                  }
                  for (size_t dim=0; dim<dimension; ++dim) {
                    series_data[dim] = spts(sindex,dim);  
                  }
                  
                  for (size_t tt=0; tt<max_numtimes; ++tt) {
                    series_data[tt+dimension] = sensor_data(sindex,field,tt);
                  }
                }
                
                const int numentries = max_numtimes+dimension;
                Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,numentries,&series_data[0],&gseries_data[0]);
                
                if (Comm->getRank() == 0) {
                  //respOUT << gseries_data[0] << "  " << gseries_data[1] << "  " << gseries_data[2] << "  ";
                  for (size_t tt=0; tt<max_numtimes+dimension; ++tt) {
                    respOUT << gseries_data[tt] << "  ";
                  }
                  respOUT << endl;
                }
              }
              if (Comm->getRank() == 0) {
                respOUT.close();
              }
            }
          }
#ifdef MrHyDE_USE_HDF5
          else if (fileoutput == "hdf5") {
            // PHDF5 creation
            size_t num_snaps = max_numtimes;
            const size_t alength = num_snaps;
            ScalarT *myData = new ScalarT[alength];

            for (int field = 0; field<max_numfields; ++field) {
            
              herr_t err; // HDF5 return value
              hid_t f_id; // HDF5 file ID

              // file access property list
              hid_t fapl_id;
              fapl_id = H5Pcreate(H5P_FILE_ACCESS);

              err = H5Pset_fapl_mpio(fapl_id, *(Comm->getRawMpiComm()), MPI_INFO_NULL);

              // create the file
              std::stringstream ss;
              ss << field;
              string respfile = "sensor_solution_field." + ss.str() + ".h5";
            
              f_id = H5Fcreate(respfile.c_str(),H5F_ACC_TRUNC, // overwrites file if it exists
                              H5P_DEFAULT,fapl_id);

              // free the file access template
              err = H5Pclose(fapl_id);

              // create the dataspace

              hid_t ds_id;
              hsize_t dims[2] = {objectives[obj].sensor_found.size(),num_snaps};
              ds_id = H5Screate_simple(2, dims, // [sensor_id,snap]
                                      NULL);

              // need to create a new hdf5 datatype which matches fftw_complex
              // TODO not sure about this...
              // TODO change ??
              hsize_t comp_dims[1] = {1};
              hid_t complex_id = H5Tarray_create2(H5T_NATIVE_DOUBLE,1,comp_dims);
            
              // create the storage
              hid_t field_id;
              field_id = H5Dcreate2(f_id,"soln",complex_id,ds_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
            
            
              // set up the portion of the files this process will access
              for (size_type sens=0; sens<sensorIDs.extent(0); ++sens) {
                hsize_t myID = sensorIDs(sens);
                hsize_t start[2] = {myID,0};
                hsize_t count[2] = {1,num_snaps};

                err = H5Sselect_hyperslab(ds_id,H5S_SELECT_SET,start,NULL, // contiguous
                                          count,NULL); // contiguous 
          
                for (size_t s=0; s<num_snaps; ++s) {
                  myData[s] = sensor_data(sensorIDs(sens),field,s);
                }
                hsize_t flattened[] = {num_snaps};
                hid_t ms_id = H5Screate_simple(1,flattened,NULL);
                err = H5Dwrite(field_id,complex_id,ms_id,ds_id,H5P_DEFAULT,myData);
              }
        
              err = H5Dclose(field_id);
        
              if (err > 0) {
                // say something
              }
              H5Sclose(ds_id);
              H5Fclose(f_id);
            }
            delete[] myData;
          }
#endif // MrHyDE_USE_HDF5
        }
        else {
          string respfile = objectives[obj].response_file+".out";
          std::ofstream respOUT;
          if (Comm->getRank() == 0) {
            bool is_open = false;
            int attempts = 0;
            int max_attempts = 100;
            while (!is_open && attempts < max_attempts) {
              respOUT.open(respfile);
              is_open = respOUT.is_open();
              attempts++;
            }
            respOUT.precision(16);
          }

          if (Comm->getRank() == 0) {
            for (size_t tt=0; tt<objectives[obj].response_times.size(); ++tt) {
              respOUT << objectives[obj].response_times[tt] << "  ";
            }
            respOUT << endl;
          }
          for (size_t ss=0; ss<objectives[obj].sensor_found.size(); ++ss) {
            for (size_t tt=0; tt<objectives[obj].response_times.size(); ++tt) {
              ScalarT sslval = 0.0, ssgval = 0.0;
              if (objectives[obj].sensor_found[ss]) {
                size_t sindex = 0;
                for (size_t j=0; j<ss; ++j) {
                  if (objectives[obj].sensor_found(j)) {
                    sindex++;
                  }
                }
                
                sslval = objectives[obj].response_data[tt](sindex);
              }
              Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&sslval,&ssgval);
              if (Comm->getRank() == 0) {
                respOUT << ssgval << "  ";
              }
            }
            if (Comm->getRank() == 0) {
              respOUT << endl;
            }
          }
          if (Comm->getRank() == 0) {
            respOUT.close();
          }
        }
      }
      else if (objectives[obj].type == "integrated response") {
        if (objectives[obj].save_data) {
          string respfile = objectives[obj].response_file+"."+blocknames[objectives[obj].block]+append+".out";
          std::ofstream respOUT;
          if (Comm->getRank() == 0) {
            bool is_open = false;
            int attempts = 0;
            int max_attempts = 100;
            while (!is_open && attempts < max_attempts) {
              respOUT.open(respfile);
              is_open = respOUT.is_open();
              attempts++;
            }
            respOUT.precision(16);
          }
          for (size_t tt=0; tt<objectives[obj].response_times.size(); ++tt) {
          
            if (Comm->getRank() == 0) {
              respOUT << objectives[obj].response_times[tt] << "  ";
            }
            double localval = objectives[obj].scalar_response_data[tt];
            double globalval = 0.0;
            Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
            if (Comm->getRank() == 0) {
              respOUT << globalval;
              respOUT << endl;
            }
          }
          if (Comm->getRank() == 0) {
            respOUT.close();
          }
        }
      }
      
    }
  }
  
  if (compute_flux_response) {
    if(Comm->getRank() == 0 ) {
      if (verbosity > 0) {
        cout << endl << "*********************************************************" << endl;
        cout << "***** Computing Flux Responses ******" << endl;
        cout << "*********************************************************" << endl;
      }
    }
    
    vector<ScalarT> gvals;
    
    for (size_t f=0; f<fluxes.size(); ++f) {
      for (size_t tt=0; tt<fluxes[f].vals.extent(0); ++tt) {
        ScalarT lval = fluxes[f].vals(tt);
        ScalarT gval = 0.0;
        Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&lval,&gval);
        gvals.push_back(gval);
      }
      Kokkos::deep_copy(fluxes[f].vals,0.0);
    }
    if (Comm->getRank() == 0) {
      string respfile = "flux_response.out";
      std::ofstream respOUT;
      if (Comm->getRank() == 0) {
        bool is_open = false;
        int attempts = 0;
        int max_attempts = 100;
        while (!is_open && attempts < max_attempts) {
          respOUT.open(respfile, std::ios_base::app);
          is_open = respOUT.is_open();
          attempts++;
        }
        respOUT.precision(16);
      }
      
      for (size_t g=0; g<gvals.size(); ++g) {
        cout << gvals[g] << endl;
        
        respOUT << " " << gvals[g] << "  ";
      }
      respOUT << endl;
      respOUT.close();
    }
    
  }

  if (compute_integrated_quantities) {
    if(Comm->getRank() == 0 ) {
      if (verbosity > 0) {
        cout << endl << "*********************************************************" << endl;
        cout << "****** Storing Integrated Quantities ******" << endl;
        cout << "*********************************************************" << endl;
      }
    }
    
    for (size_t iLocal=0; iLocal<integratedQuantities.size(); iLocal++) {

      // iLocal indexes over the number of blocks where IQs are defined and
      // does not necessarily match the global block ID

      size_t globalBlock = integratedQuantities[iLocal][0].block; // all IQs with same first index share a block

      if (Comm->getRank() == 0) { 
        cout << endl << "*********************************************************" << endl;
        cout << "****** Integrated Quantities on block : " << blocknames[globalBlock] <<  " ******" << endl;
        cout << "*********************************************************" << endl;
        for (size_t k=0; k<integratedQuantities[iLocal].size(); ++k) { 
          std::cout << integratedQuantities[iLocal][k].name  << " : " 
                    << integratedQuantities[iLocal][k].val(0) << std::endl; 
        }
      }
    
    } // end loop over blocks with IQs requested
    // TODO output something? Make the first print statement true!
    // BWR -- this only happens at end of sim.
  } // end if compute_integrated_quantities

  ////////////////////////////////////////////////////////////////////////////
  // Report the errors for verification tests
  ////////////////////////////////////////////////////////////////////////////
  
  if (compute_error) {
    if(Comm->getRank() == 0) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Computing errors ******" << endl << endl;
    }
    
    for (size_t block=0; block<assembler->groups.size(); block++) {// loop over blocks
      for (size_t etype=0; etype<error_list[block].size(); etype++){
        
        //for (size_t et=0; et<error_types.size(); et++){
        for (size_t time=0; time<error_times.size(); time++) {
          //for (int n=0; n<numVars[block]; n++) {
          
          ScalarT lerr = errors[time][block](etype);
          ScalarT gerr = 0.0;
          Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&lerr,&gerr);
          if(Comm->getRank() == 0) {
            string varname = error_list[block][etype].first;
            if (error_list[block][etype].second == "L2" || error_list[block][etype].second == "L2 VECTOR") {
              cout << "***** L2 norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
            }
            else if (error_list[block][etype].second == "L2 FACE") {
              cout << "***** L2-face norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
            }
            else if (error_list[block][etype].second == "GRAD") {
              cout << "***** L2 norm of the error for grad(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
            }
            else if (error_list[block][etype].second == "DIV") {
              cout << "***** L2 norm of the error for div(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
            }
            else if (error_list[block][etype].second == "CURL") {
              cout << "***** L2 norm of the error for curl(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
            }
          }
          //}
        }
      }
    }
    
    // Error in subgrid models
    if (!(Teuchos::is_null(multiscale_manager))) {
      if (multiscale_manager->getNumberSubgridModels() > 0) {
        
        for (size_t m=0; m<multiscale_manager->getNumberSubgridModels(); m++) {
          vector<string> sgvars = multiscale_manager->subgridModels[m]->varlist;
          vector<std::pair<string,string> > sg_error_list;
          // A given processor may not have any elements that use this subgrid model
          // In this case, nothing gets initialized so sgvars.size() == 0
          // Find the global max number of sgvars over all processors
          size_t nvars = sgvars.size();
          if (nvars>0) {
            sg_error_list = multiscale_manager->subgridModels[m]->getErrorList();
          }
          // really only works on one block
          size_t nerrs = sg_error_list.size();
          size_t gnerrs = 0;
          Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&nerrs,&gnerrs);
          
          for (size_t etype=0; etype<gnerrs; etype++) {
            for (size_t time=0; time<error_times.size(); time++) {
              //for (int n=0; n<gnvars; n++) {
              // Get the local contribution (if processor uses subgrid model)
              ScalarT lerr = 0.0;
              if (subgrid_errors[time][0][m].extent(0)>0) {
                lerr = subgrid_errors[time][0][m](etype); // block is not relevant
              }
              ScalarT gerr = 0.0;
              Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&lerr,&gerr);
              
              // Figure out who can print the information (lowest rank amongst procs using subgrid model)
              int myID = Comm->getRank();
              if (nvars == 0) {
                myID = 100000000;
              }
              int gID = 0;
              Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&myID,&gID);
              
              if(Comm->getRank() == gID) {
                //cout << "***** Subgrid" << m << ": " << subgrid_error_types[etype] << " norm of the error for " << sgvars[n] << " = " << sqrt(gerr) << "  (time = " << error_times[t] << ")" <<  endl;
                
                string varname = sg_error_list[etype].first;
                if (sg_error_list[etype].second == "L2" || sg_error_list[etype].second == "L2 VECTOR") {
                  cout << "***** Subgrid " << m << ": L2 norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
                }
                else if (sg_error_list[etype].second == "L2 FACE") {
                  cout << "***** Subgrid " << m << ": L2-face norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
                }
                else if (sg_error_list[etype].second == "GRAD") {
                  cout << "***** Subgrid " << m << ": L2 norm of the error for grad(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
                }
                else if (sg_error_list[etype].second == "DIV") {
                  cout << "***** Subgrid " << m << ": L2 norm of the error for div(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
                }
                else if (sg_error_list[etype].second == "CURL") {
                  cout << "***** Subgrid " << m << ": L2 norm of the error for curl(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" <<  endl;
                }
              }
              //}
            }
          }
        }
      }
    }
    
  }
  
  if (compute_weighted_norm) {
    if (Comm->getRank() == 0) {
      string respfile = "weighted_norms.out";
      std::ofstream respOUT;
      respOUT.open(respfile);
      for (size_t k=0; k<weighted_norms.size(); ++k) {
        respOUT << weighted_norms[k] << endl;
      }
      respOUT << endl;
      respOUT.close();
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeError(const ScalarT & currenttime) {
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting PostprocessManager::computeError(time)" << endl;
    }
  }
  
  Teuchos::TimeMonitor localtimer(*computeErrorTimer);
  
  error_times.push_back(currenttime);
  
  vector<Kokkos::View<ScalarT*,HostDevice> > currerror;
  int seedwhat = 0;
  
  for (size_t block=0; block<assembler->groups.size(); block++) {// loop over blocks
    
    int altblock; // Needed for subgrid error calculations
    if (assembler->wkset.size()>block && error_list.size()>block) {
      altblock = block;
    }
    else {
      altblock = 0;
    }
    // groups can use block, but everything else should be altblock
    // This is due to how the subgrid models store the groups
    
    Kokkos::View<ScalarT*,HostDevice> blockerrors("error",error_list[altblock].size());
    
    if (assembler->groups[block].size() > 0) {
      
      assembler->wkset[altblock]->setTime(currenttime);
      
      // Need to use time step solution instead of stage solution
      bool isTransient = assembler->wkset[altblock]->isTransient;
      assembler->wkset[altblock]->isTransient = false;
      assembler->groupData[altblock]->requires_transient = false;
      
      // Determine what needs to be updated in the workset
      bool have_vol_errs = false, have_face_errs = false;
      for (size_t etype=0; etype<error_list[altblock].size(); etype++){
        if (error_list[altblock][etype].second == "L2" || error_list[altblock][etype].second == "GRAD"
            || error_list[altblock][etype].second == "DIV" || error_list[altblock][etype].second == "CURL"
            || error_list[altblock][etype].second == "L2 VECTOR") {
          have_vol_errs = true;
        }
        if (error_list[altblock][etype].second == "L2 FACE") {
          have_face_errs = true;
        }
      }
      for (size_t grp=0; grp<assembler->groups[block].size(); grp++) {
        if (assembler->groups[block][grp]->active) {
        if (have_vol_errs) {
          assembler->updateWorkset(assembler->wkset[altblock],block, grp, seedwhat,0, true);
          //assembler->updateWorkset(altblock, grp, seedwhat,true);
        }
        //auto wts = assembler->wkset[block]->wts;
        auto wts = assembler->wkset[altblock]->wts;
        
        for (size_t etype=0; etype<error_list[altblock].size(); etype++) {
          string varname = error_list[altblock][etype].first;
          
          if (error_list[altblock][etype].second == "L2") {
            // compute the true solution
            string name = varname;
            View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
            auto sol = assembler->wkset[altblock]->getSolutionField(name);
            
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol(elem,pt) - tsol(elem,pt);
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
          }
          else if (error_list[altblock][etype].second == "GRAD") {
            // compute the true x-component of grad
            string name = "grad(" + varname + ")[x]";
            View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
            auto sol_x = assembler->wkset[altblock]->getSolutionField(name);
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol_x(elem,pt) - tsol(elem,pt);
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
            
            if (dimension > 1) {
              // compute the true y-component of grad
              string name = "grad(" + varname + ")[y]";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
              auto sol_y = assembler->wkset[altblock]->getSolutionField(name);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_y(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
            
            if (dimension > 2) {
              // compute the true z-component of grad
              string name = "grad(" + varname + ")[z]";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
              auto sol_z = assembler->wkset[altblock]->getSolutionField(name);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_z(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
          }
          else if (error_list[altblock][etype].second == "DIV") {
            // compute the true divergence
            string name = "div(" + varname + ")";
            View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
            auto sol_div = assembler->wkset[altblock]->getSolutionField(name);
            
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol_div(elem,pt) - tsol(elem,pt);
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
          }
          else if (error_list[altblock][etype].second == "CURL") {
            // compute the true x-component of grad
            string name = "curl(" + varname + ")[x]";
            View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
            auto sol_curl_x = assembler->wkset[altblock]->getSolutionField(name);
            
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol_curl_x(elem,pt) - tsol(elem,pt);
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
            
            if (dimension > 1) {
              // compute the true y-component of grad
              string name = "curl(" + varname + ")[y]";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
              auto sol_curl_y = assembler->wkset[altblock]->getSolutionField(name);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_curl_y(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
            
            if (dimension >2) {
              // compute the true z-component of grad
              string name = "curl(" + varname + ")[z]";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
              auto sol_curl_z = assembler->wkset[altblock]->getSolutionField(name);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_curl_z(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
          }
          else if (error_list[altblock][etype].second == "L2 VECTOR") {
            // compute the true x-component of grad
            string name = varname + "[x]";
            View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
            auto sol_x = assembler->wkset[altblock]->getSolutionField(name);
            
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol_x(elem,pt) - tsol(elem,pt);
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
            
            if (dimension > 1) {
              // compute the true y-component of grad
              string name = varname + "[y]";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
              auto sol_y = assembler->wkset[altblock]->getSolutionField(name);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_y(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
            
            if (dimension > 2) {
              // compute the true z-component of grad
              string name = varname + "[z]";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "ip");
              auto sol_z = assembler->wkset[altblock]->getSolutionField(name);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_z(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
          }
        }
        }
        if (have_face_errs) {
          assembler->wkset[altblock]->isOnSide = true;
          for (size_t face=0; face<assembler->groups[block][grp]->group_data->num_sides; face++) {
            // TMW - hard coded for now
            for (size_t set=0; set<assembler->wkset[altblock]->numSets; ++set) {
              assembler->wkset[altblock]->computeSolnSteadySeeded(set, assembler->groups[block][grp]->sol[set], seedwhat);
            }
            assembler->updateWorksetFace(block, grp, face);
            assembler->wkset[altblock]->resetSolutionFields();
            for (size_t etype=0; etype<error_list[altblock].size(); etype++) {
              string varname = error_list[altblock][etype].first;
              if (error_list[altblock][etype].second == "L2 FACE") {
                // compute the true z-component of grad
                string name = varname;
                View_Sc2 tsol = assembler->evaluateFunction(altblock, "true "+name, "side ip");
                
                auto sol = assembler->wkset[altblock]->getSolutionField(name);
                auto wts = assembler->wkset[block]->wts_side;
                
                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)),
                                KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                  double facemeasure = 0.0;
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    facemeasure += wts(elem,pt);
                  }
                  
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    ScalarT diff = sol(elem,pt) - tsol(elem,pt);
                    update += 0.5/facemeasure*diff*diff*wts(elem,pt);  // TODO - BWR what is this? why .5?
                  }
                }, error);
                blockerrors(etype) += error;
              }
            }
          }
          assembler->wkset[altblock]->isOnSide = false;
        }
      }
      assembler->wkset[altblock]->isTransient = isTransient;
      assembler->groupData[altblock]->requires_transient = isTransient;
    }
    currerror.push_back(blockerrors);
  } // end block loop
  
  // Need to move currerrors to Host
  vector<Kokkos::View<ScalarT*,HostDevice> > host_error;
  for (size_t k=0; k<currerror.size(); k++) {
    Kokkos::View<ScalarT*,HostDevice> host_cerr("error on host",currerror[k].extent(0));
    Kokkos::deep_copy(host_cerr,currerror[k]);
    host_error.push_back(host_cerr);
  }
  
  errors.push_back(host_error);
  
  if (!(Teuchos::is_null(multiscale_manager))) {
    if (multiscale_manager->getNumberSubgridModels() > 0) {
      // Collect all of the errors for each subgrid model
      vector<vector<Kokkos::View<ScalarT*,HostDevice> > > blocksgerrs;
      
      for (size_t block=0; block<assembler->groups.size(); block++) {// loop over blocks
        
        vector<Kokkos::View<ScalarT*,HostDevice> > sgerrs;
        for (size_t m=0; m<multiscale_manager->getNumberSubgridModels(); m++) {
          Kokkos::View<ScalarT*,HostDevice> err = multiscale_manager->subgridModels[m]->computeError(currenttime);
          sgerrs.push_back(err);
        }
        blocksgerrs.push_back(sgerrs);
      }
      /*
      vector<vector<Kokkos::View<ScalarT*,HostDevice> > > host_blocksgerrs;
      for (size_t k=0; k<blocksgerrs.size(); k++) {
        vector<Kokkos::View<ScalarT*,HostDevice> > host_sgerrs;
        for (size_t j=0; j<blocksgerrs[k].size(); j++) {
          Kokkos::View<ScalarT*,HostDevice> host_err("subgrid error on host",blocksgerrs[k][j].extent(0));
          Kokkos::deep_copy(host_err,blocksgerrs[k][j]);
        }
        host_blocksgerrs.push_back(host_sgerrs);
      }*/
      subgrid_errors.push_back(blocksgerrs);
    }
  }
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished PostprocessManager::computeError(time)" << endl;
    }
  }
  
}

// ========================================================================================
// TMW: I don't know if the following function is actually used for anything
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeResponse(const ScalarT & currenttime) {
  
  /*
  response_times.push_back(currenttime);
  params->sacadoizeParams(false);
  
  // TMW: may not work for multi-block
  int numresponses = phys->getNumResponses(0);
  int numSensors = 1;
  if (response_type == "pointwise" ) {
    numSensors = sensors->numSensors;
  }
  
  Kokkos::View<ScalarT**,HostDevice> curr_response("current response",
                                                   numSensors, numresponses);
  for (size_t b=0; b<assembler->groups.size(); b++) {
    for (size_t e=0; e<assembler->groups[block].size(); e++) {
  
      auto responsevals = assembler->groups[block][grp]->computeResponse(0);
      
      //auto host_response = Kokkos::create_mirror_view(responsevals);
      Kokkos::View<AD***,HostDevice> host_response("response on host",responsevals.extent(0),
                                                   responsevals.extent(1), responsevals.extent(2));
      Kokkos::deep_copy(host_response,responsevals);
      
      for (int r=0; r<numresponses; r++) {
        if (response_type == "global" ) {
          auto wts = assembler->groups[block][grp]->wts;
          auto host_wts = Kokkos::create_mirror_view(wts);
          Kokkos::deep_copy(host_wts,wts);
          
          for (size_type p=0; p<host_response.extent(0); p++) {
            for (size_t j=0; j<host_wts.extent(1); j++) {
              curr_response(0,r) += host_response(p,r,j).val() * host_wts(p,j);
            }
          }
        }
        else if (response_type == "pointwise" ) {
          if (host_response.extent(1) > 0) {
            vector<int> sensIDs = assembler->groups[block][grp]->mySensorIDs;
            for (size_type p=0; p<host_response.extent(0); p++) {
              for (size_t j=0; j<host_response.extent(2); j++) {
                curr_response(sensIDs[j],r) += host_response(p,r,j).val();
              }
            }
          }
        }
      }
    }
  }
  responses.push_back(curr_response);
  */
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeFluxResponse(const ScalarT & currenttime) {
  
  for (size_t block=0; block<assembler->groupData.size(); ++block) {
    for (size_t grp=0; grp<assembler->boundary_groups[block].size(); ++grp) {
      // setup workset for this bgrp
      
      assembler->updateWorksetBoundary(block, grp, 0, 0, true);
      
      // compute the flux
      assembler->wkset[block]->flux = View_Sc3("flux",assembler->wkset[block]->maxElem,
                                               assembler->wkset[block]->numVars[0], // hard coded
                                               assembler->wkset[block]->numsideip);
      
      physics->computeFlux<ScalarT>(0,block); // hard coded
      auto cflux = assembler->wkset[block]->flux; // View_AD3
      
      for (size_t f=0; f<fluxes.size(); ++f) {
        
        if (fluxes[f].block == block) {
          string sidename = assembler->boundary_groups[block][grp]->sidename;
          size_t found = fluxes[f].sidesets.find(sidename);
          
          if (found!=std::string::npos) {
            
            View_Sc2 wts = assembler->evaluateFunction(block, "flux weight "+fluxes[f].name,"side ip");
            auto iwts = assembler->wkset[block]->wts_side;
            
            for (size_type v=0; v<fluxes[f].vals.extent(0); ++v) {
              ScalarT value = 0.0;
              auto vflux = subview(cflux,ALL(),v,ALL());
              parallel_reduce(RangePolicy<AssemblyExec>(0,iwts.extent(0)),
                              KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<iwts.extent(1); pt++ ) {
                  ScalarT up = vflux(elem,pt)*wts(elem,pt)*iwts(elem,pt);
                  update += up;
                }
              }, value);
              fluxes[f].vals(v) += value;
            }
          }
        }
      }
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeIntegratedQuantities(const ScalarT & currenttime) {

  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting PostprocessManager::computeIntegratedQuantities ..." << std::endl;
    }
  }

  // TODO :: BWR -- currently, I am proceeding like quantities are requested over 
  // a subvolume (or subboundary, etc.) which is defined by the block
  // Hence, if a user wanted an integral over the ENTIRE volume, they would need to 
  // sum up the individual contributions (in a multiblock case)
  
  for (size_t iLocal=0; iLocal<integratedQuantities.size(); iLocal++) {

    // iLocal indexes over the number of blocks where IQs are defined and
    // does not necessarily match the global block ID
  
    size_t globalBlock = integratedQuantities[iLocal][0].block; // all IQs with same first index share a block

    vector<ScalarT> allsums; // For the final results after summing over MPI processes

    // the first n IQs are needed by the workset for residual calculations
    size_t nIQsForResidual = assembler->wkset[globalBlock]->integrated_quantities.extent(0);

    // MPI sums happen on the host and later we pass to the device (where residual is formed)
    auto hostsums = Kokkos::View<ScalarT*,HostDevice>("host IQs",nIQsForResidual);

    for (size_t iIQ=0; iIQ<integratedQuantities[iLocal].size(); ++iIQ) {

      ScalarT integral = 0.;
      ScalarT localContribution;
      
      if (integratedQuantities[iLocal][iIQ].location == "volume") {

        for (size_t grp=0; grp<assembler->groups[globalBlock].size(); ++grp) {

          localContribution = 0.; // zero out this grp's contribution JIC here but needed below

          // setup the workset for this grp
          assembler->updateWorkset(globalBlock, grp, 0,0,true);
          // get integration weights
          auto wts = assembler->wkset[globalBlock]->wts;
          // evaluate the integrand at integration points
          //auto integrand = functionManagers[globalBlock]->evaluate(integratedQuantities[iLocal][iIQ].name+" integrand","ip");
          View_Sc2 integrand = assembler->evaluateFunction(globalBlock, integratedQuantities[iLocal][iIQ].name+" integrand","ip");
          // expand this for integral integrands, etc.?
          
          parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)),
                          KOKKOS_LAMBDA (const int elem, ScalarT & update) {
                            for (size_t pt=0; pt<wts.extent(1); pt++) {
                              ScalarT Idx = wts(elem,pt)*integrand(elem,pt);
                              update += Idx;
                            }
                          },localContribution); //// TODO :: may be illegal

          // add this grp's contribution to running total

          integral += localContribution;

        } // end loop over groups

      } else if (integratedQuantities[iLocal][iIQ].location == "boundary") {

        assembler->wkset[globalBlock]->isOnSide = true;

        for (size_t grp=0; grp<assembler->boundary_groups[globalBlock].size(); ++grp) {

          localContribution = 0.; // zero out this grp's contribution

          // check if we are on one of the requested sides
          string sidename = assembler->boundary_groups[globalBlock][grp]->sidename;
          size_t found = integratedQuantities[iLocal][iIQ].boundarynames.find(sidename);

          if ( (found!=std::string::npos) || 
               (integratedQuantities[iLocal][iIQ].boundarynames == "all") ) {

            // setup the workset for this grp
            assembler->updateWorksetBoundary(globalBlock, grp, 0, 0, true); 
            // get integration weights
            auto wts = assembler->wkset[globalBlock]->wts_side;
            // evaluate the integrand at integration points
            //auto integrand = functionManagers[globalBlock]->evaluate(integratedQuantities[iLocal][iIQ].name+" integrand","side ip");
            View_Sc2 integrand = assembler->evaluateFunction(globalBlock, integratedQuantities[iLocal][iIQ].name+" integrand","side ip");
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)),
                            KOKKOS_LAMBDA (const int elem, ScalarT & update) {
                              for (size_t pt=0; pt<wts.extent(1); pt++) {
                                ScalarT Idx = wts(elem,pt)*integrand(elem,pt);
                                update += Idx;
                              }
                            },localContribution); //// TODO :: may be illegal, problematic ABOVE TOO

          } // end if requested side
          // add in this grp's contribution to running total
          integral += localContribution;
        } // end loop over boundary groups
        
        assembler->wkset[globalBlock]->isOnSide = false;
        
      } // end if volume or boundary
      // finalize the integral
      integratedQuantities[iLocal][iIQ].val(0) = integral;
      // reduce
      ScalarT gval = 0.0;
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&integral,&gval);
      if (iIQ<nIQsForResidual) {
        hostsums(iIQ) = gval;
      }
      allsums.push_back(gval);
      // save global result back in IQ storage
      integratedQuantities[iLocal][iIQ].val(0) = allsums[iIQ];
    } // end loop over integrated quantities

    // need to put in the right place now (accessible to the residual) and 
    // update any parameters which depend on the IQs
    // TODO :: BWR this ultimately is an "explicit" idea but doing things implicitly
    // would be super costly in general.
    
    // TODO CHECK THIS WITH TIM... am I dev/loc correctly?
    if (nIQsForResidual > 0) {
      Kokkos::deep_copy(assembler->wkset[globalBlock]->integrated_quantities,hostsums);
      for (size_t set=0; set<physics->modules.size(); ++set) {
        for (size_t m=0; m<physics->modules[set][globalBlock].size(); ++m) {
          // BWR -- called for all physics defined on the block regards of if they need IQs
          physics->modules[set][globalBlock][m]->updateIntegratedQuantitiesDependents();
        }
      }
    } // end if physics module needs IQs

  } // end loop over blocks (with IQs requested)

}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeWeightedNorm(vector<vector_RCP> & current_soln) {
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting PostprocessManager::computeWeightedNorm()" << endl;
    }
  }
  
  Teuchos::TimeMonitor localtimer(*computeWeightedNormTimer);
  
  typedef typename Node::execution_space LA_exec;
  typedef typename Node::device_type     LA_device;
  
  if (!have_norm_weights) {
    for (size_t set=0; set<current_soln.size(); ++set) {
      vector_RCP wts_over = linalg->getNewOverlappedVector(set);
      assembler->getWeightVector(set, wts_over);
      vector_RCP set_norm_wts = linalg->getNewVector(set);
      set_norm_wts->putScalar(0.0);
      set_norm_wts->doExport(*wts_over, *(linalg->exporter[set]), Tpetra::REPLACE);
      norm_wts.push_back(set_norm_wts);
    }
    have_norm_weights = true;
  }
  
  ScalarT totalnorm = 0.0;
  for (size_t set=0; set<current_soln.size(); ++set) {
    // current_soln is an overlapped vector ... we want
    vector_RCP soln = linalg->getNewVector(set);
    soln->putScalar(0.0);
    soln->doExport(*(current_soln[set]), *(linalg->exporter[set]), Tpetra::REPLACE);
    
    
    vector_RCP prod = linalg->getNewVector(set);
    
    auto wts_view = norm_wts[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto prod_view = prod->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto soln_view = soln->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    parallel_for("assembly insert Jac",
                 RangePolicy<LA_exec>(0,prod_view.extent(0)),
                 KOKKOS_LAMBDA (const int k ) {
      prod_view(k,0) = wts_view(k,0)*soln_view(k,0)*soln_view(k,0);
    });
    
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> l2norm(1);
    prod->norm2(l2norm);
    totalnorm += l2norm[0];
  }
  
  weighted_norms.push_back(totalnorm);
  
  if (verbosity >= 10 && Comm->getRank() == 0) {
    cout << "Weighted norm of solution: " << totalnorm << endl;
  }
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished PostprocessManager::computeWeightedNorm()" << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================

// Helper function to save data
template<class Node>
void PostprocessManager<Node>::saveObjectiveData(const DFAD& obj) {
  if(Comm->getRank() != 0) return;
  if(objective_file.length() > 0) {
    std::ofstream obj_out {objective_file};
    TEUCHOS_TEST_FOR_EXCEPTION(!obj_out.is_open(), std::runtime_error, "Could not open file to print objective value");
    obj_out << obj.val();
  }
}

// Helper function to save data
template<class Node>
void PostprocessManager<Node>::saveObjectiveGradientData(const MrHyDE_OptVector& gradient) {
  if(Comm->getRank() != 0) return;
  if(objective_grad_file.length() > 0) {
    std::ofstream obj_grad_out {objective_grad_file};
    TEUCHOS_TEST_FOR_EXCEPTION(!obj_grad_out.is_open(), std::runtime_error, "Could not open file to print objective gradient value");
    gradient.print(obj_grad_out);
  }
}

template<class Node>
void PostprocessManager<Node>::computeObjective(vector<vector_RCP> & current_soln,
                                                const ScalarT & current_time,
                                                DFAD & objectiveval) {
  
  Teuchos::TimeMonitor localtimer(*objectiveTimer);
  
  if (debug_level > 1 && Comm->getRank() == 0) {
    std::cout << "******** Starting PostprocessManager::computeObjective ..." << std::endl;
  }
  
  int numParams = params->num_active_params + params->globalParamUnknowns;
  
  // Objective function values
  vector<ScalarT> totaldiff(objectives.size(), 0.0);
  
  
  for (size_t r=0; r<objectives.size(); ++r) {
    if (objectives[r].type == "integrated control"){
      
      size_t block = objectives[r].block;
      
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
        View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
        
        auto wts = assembler->groups[block][grp]->wts;
        
        assembler->updateWorkset(block, grp, 0,0,true);
        
        auto obj_dev = assembler->function_managers[block]->evaluate(objectives[r].name,"ip");
        
        Kokkos::View<ScalarT[1],AssemblyDevice> objsum("sum of objective");
        parallel_for("grp objective",
                     RangePolicy<AssemblyExec>(0,wts.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          ScalarT tmpval = 0.0;
          for (size_type pt=0; pt<wts.extent(1); pt++) {
            tmpval += obj_dev(elem,pt)*wts(elem,pt);
          }
          Kokkos::atomic_add(&(objsum(0)),tmpval);
        });
        
        parallel_for("grp objective",
                     RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                     KOKKOS_LAMBDA (const size_type p ) {
          if (p==0) {
            objsum_dev(p) = objsum(0);
          }
        });
        
        auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
        Kokkos::deep_copy(objsum_host,objsum_dev);
        
        // Update the objective function value
        totaldiff[r] += objectives[r].weight*objsum_host(0);
        
      }
      
      
    }
    else if (objectives[r].type == "discrete control") {
      for (size_t set=0; set<current_soln.size(); ++set) {
        vector_RCP D_soln;
        bool fnd = datagen_soln[set]->extract(D_soln, 0, current_time);
        if (fnd) {
          vector_RCP diff = linalg->getNewVector(set);
          vector_RCP F_no = linalg->getNewVector(set);
          vector_RCP D_no = linalg->getNewVector(set);
          F_no->doExport(*(current_soln[set]), *(linalg->exporter[set]), Tpetra::REPLACE);
          D_no->doExport(*D_soln, *(linalg->exporter[set]), Tpetra::REPLACE);
          
          diff->update(1.0, *F_no, 0.0);
          diff->update(-1.0, *D_no, 1.0);
          Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> obj(1);
          diff->norm2(obj);
          if (Comm->getRank() == 0) {
            totaldiff[r] += objectives[r].weight*obj[0]*obj[0];
          }
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: did not find a data-generating solution");
        }
      }
    }
    else if (objectives[r].type == "integrated response") {
      
      size_t block = objectives[r].block;
      
      
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
          auto wts = assembler->groups[block][grp]->wts;
            
          assembler->updateWorkset(block, grp, 0,0,true);
        
          auto obj_dev = assembler->function_managers[block]->evaluate(objectives[r].name+" response","ip");
        
          Kokkos::View<ScalarT[1],AssemblyDevice> objsum("sum of objective");
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            ScalarT tmpval = 0.0;
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              tmpval += obj_dev(elem,pt)*wts(elem,pt);
            }
            Kokkos::atomic_add(&(objsum(0)),tmpval);
          });
        
          View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
        
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                       KOKKOS_LAMBDA (const size_type p ) {
            if (p==0) {
              objsum_dev(p) = objsum(0);
            }
          });
        
          auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
          Kokkos::deep_copy(objsum_host,objsum_dev);
        
          // Update the objective function value
          totaldiff[r] += objsum_host(0);
        
        }
      
        if (compute_response) {
          if (objectives[r].save_data) {
            objectives[r].response_times.push_back(current_time);
            objectives[r].scalar_response_data.push_back(totaldiff[r]);
            if (verbosity >= 10) {
              double localval = totaldiff[r];
              double globalval = 0.0;
              Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
              if (Comm->getRank() == 0) {
                cout << objectives[r].name << " on block " << blocknames[objectives[r].block] << ": " << globalval << endl;
              }
            }
          }
        }
      //}

      // Right now, totaldiff = response
      //             gradient = dresponse / dp
      // We want    totaldiff = wt*(response-target)^2
      //             gradient = 2*wt*(response-target)*dresponse/dp
      
      ScalarT diff = totaldiff[r] - objectives[r].target;
      totaldiff[r] = objectives[r].weight*diff*diff;
      
    }
    else if (objectives[r].type == "sensors" || objectives[r].type == "sensor response" || objectives[r].type == "pointwise response") {
      if (objectives[r].compute_sensor_soln || objectives[r].compute_sensor_average_soln) {
        // don't do anything for this use case
      }
      else {
        Kokkos::View<ScalarT*,HostDevice> sensordat;
        if (compute_response) {
          sensordat = Kokkos::View<ScalarT*,HostDevice>("sensor data to save",objectives[r].numSensors);
          objectives[r].response_times.push_back(current_time);
        }
      
        for (size_t pt=0; pt<objectives[r].numSensors; ++pt) {
          size_t tindex = 0;
          bool foundtime = false;
          for (size_type t=0; t<objectives[r].sensor_times.extent(0); ++t) {
            if (std::abs(current_time - objectives[r].sensor_times(t)) < 1.0e-12) {
              foundtime = true;
              tindex = t;
            }
          }
          
          if (compute_response || foundtime) {
          
            size_t block = objectives[r].block;
            size_t grp = objectives[r].sensor_owners(pt,0);
            size_t elem = objectives[r].sensor_owners(pt,1);
            assembler->wkset[block]->isOnPoint = true;
            auto x = assembler->wkset[block]->getScalarField("x");
            x(0,0) = objectives[r].sensor_points(pt,0);
            if (dimension > 1) {
              auto y = assembler->wkset[block]->getScalarField("y");
              y(0,0) = objectives[r].sensor_points(pt,1);
            }
            if (dimension > 2) {
              auto z = assembler->wkset[block]->getScalarField("z");
              z(0,0) = objectives[r].sensor_points(pt,2);
            }
            
            auto numDOF = assembler->groupData[block]->num_dof;
            View_Sc2 u_dof("u_dof",numDOF.extent(0),assembler->groups[block][grp]->LIDs[0].extent(1)); // hard coded
            auto cu = subview(assembler->groups[block][grp]->sol[0],elem,ALL(),ALL()); // hard coded
            parallel_for("grp response get u",
                        RangePolicy<AssemblyExec>(0,u_dof.extent(0)),
                        KOKKOS_LAMBDA (const size_type n ) {
              for (size_type n=0; n<numDOF.extent(0); n++) {
                for( int i=0; i<numDOF(n); i++ ) {
                  u_dof(n,i) = cu(n,i);
                }
              }
            });
            
            // Map the local solution to the solution and gradient at ip
            View_Sc2 u_ip("u_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
            View_Sc2 ugrad_ip("ugrad_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
            
            for (size_type var=0; var<numDOF.extent(0); var++) {
              auto cbasis = objectives[r].sensor_basis[assembler->wkset[block]->usebasis[var]];
              auto cbasis_grad = objectives[r].sensor_basis_grad[assembler->wkset[block]->usebasis[var]];
              auto u_sv = subview(u_ip, var, ALL());
              auto u_dof_sv = subview(u_dof, var, ALL());
              auto ugrad_sv = subview(ugrad_ip, var, ALL());
              
              parallel_for("grp response sensor uip",
                          RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                          KOKKOS_LAMBDA (const int dof ) {
                u_sv(0) += u_dof_sv(dof)*cbasis(pt,dof,0,0);
                for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                  ugrad_sv(dim) += u_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                }
              });
            }
            
            assembler->wkset[block]->setSolutionPoint(u_ip);
            assembler->wkset[block]->setSolutionGradPoint(ugrad_ip);
            
            // Map the local discretized params to param and grad at ip
            if (params->globalParamUnknowns > 0) {
              auto numParamDOF = assembler->groupData[block]->num_param_dof;
              
              View_Sc2 p_dof("p_dof",numParamDOF.extent(0),assembler->groups[block][grp]->paramLIDs.extent(1));
              auto cp = subview(assembler->groups[block][grp]->param,elem,ALL(),ALL());
              parallel_for("grp response get u",
                          RangePolicy<AssemblyExec>(0,p_dof.extent(0)),
                          KOKKOS_LAMBDA (const size_type n ) {
                for (size_type n=0; n<numParamDOF.extent(0); n++) {
                  for( int i=0; i<numParamDOF(n); i++ ) {
                    p_dof(n,i) = cp(n,i);
                  }
                }
              });
              
              View_Sc2 p_ip("p_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
              View_Sc2 pgrad_ip("pgrad_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
              
              for (size_type var=0; var<numParamDOF.extent(0); var++) {
                int bnum = assembler->wkset[block]->paramusebasis[var];
                auto cbasis = objectives[r].sensor_basis[bnum];
                auto cbasis_grad = objectives[r].sensor_basis_grad[bnum];
                auto p_sv = subview(p_ip, var, ALL());
                auto p_dof_sv = subview(p_dof, var, ALL());
                auto pgrad_sv = subview(pgrad_ip, var, ALL());
                
                parallel_for("grp response sensor uip",
                            RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                            KOKKOS_LAMBDA (const int dof ) {
                  p_sv(0) += p_dof_sv(dof)*cbasis(pt,dof,0,0);
                  for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                    pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                  }
                });
              }
              
              assembler->wkset[block]->setParamPoint(p_ip);
              
              assembler->wkset[block]->setParamGradPoint(pgrad_ip);
            }
            
            // Evaluate the response
            auto rdata = assembler->function_managers[block]->evaluate(objectives[r].name+" response","point");
            
            if (compute_response) {
              sensordat(pt) = rdata(0,0);
            }
            
            if (compute_objective) {
              
              // Update the value of the objective
              ScalarT diff = rdata(0,0) - objectives[r].sensor_data(pt,tindex);
              ScalarT sdiff = objectives[r].weight*diff*diff;
              totaldiff[r] += sdiff;
              
            }
            assembler->wkset[block]->isOnPoint = false;

          } // found time
        } // sensor points
        
        if (compute_response) {
          objectives[r].response_data.push_back(sensordat);
        }
      } // objectives
    }
    // ========================================================================================
    // Add regularizations (reg funcs are tied to objectives and objectives can have more than one reg)
    // ========================================================================================

    for (size_t reg=0; reg<objectives[r].regularizations.size(); ++reg) {
      if (objectives[r].regularizations[reg].type == "integrated") {
        if (objectives[r].regularizations[reg].location == "volume") {
          ScalarT regwt = objectives[r].regularizations[reg].weight;
          size_t block = objectives[r].block;
          for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
            
            auto wts = assembler->groups[block][grp]->wts;
            
            assembler->updateWorkset(block, grp, 3,0,true);
            
            auto regvals_tmp = assembler->function_managers[block]->evaluate(objectives[r].regularizations[reg].name,"ip");
            View_Sc2 regvals("regvals",wts.extent(0),wts.extent(1));
            
            parallel_for("grp objective",
                         RangePolicy<AssemblyExec>(0,wts.extent(0)),
                         KOKKOS_LAMBDA (const size_type elem ) {
              for (size_type pt=0; pt<wts.extent(1); ++pt) {
                regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
              }
            });
            
            
            auto regvals_sc_host = create_mirror_view(regvals);
            deep_copy(regvals_sc_host,regvals);
            
            auto poffs = params->paramoffsets;
            for (size_t elem=0; elem<assembler->groups[block][grp]->numElem; ++elem) {
                            
              vector<GO> paramGIDs;
              params->paramDOF->getElementGIDs(assembler->groups[block][grp]->localElemID(elem),
                                               paramGIDs, blocknames[block]);
              
              for (size_type pt=0; pt<regvals_sc_host.extent(1); ++pt) {
                totaldiff[r] += regwt*regvals_sc_host(elem,pt);
              }
            }
          }
          
        }
        else if (objectives[r].regularizations[reg].location == "boundary") {
          string bname = objectives[r].regularizations[reg].boundary_name;
          ScalarT regwt = objectives[r].regularizations[reg].weight;
          size_t block = objectives[r].block;
          assembler->wkset[block]->isOnSide = true;
          for (size_t grp=0; grp<assembler->boundary_groups[block].size(); ++grp) {
            if (assembler->boundary_groups[block][grp]->sidename == bname) {
              
              auto wts = assembler->boundary_groups[block][grp]->wts;
              
              assembler->updateWorksetBoundary(block, grp, 3, 0, true);
              
              auto regvals_tmp = assembler->function_managers[block]->evaluate(objectives[r].regularizations[reg].name,"side ip");
              View_Sc2 regvals("regvals",wts.extent(0),wts.extent(1));
              
              parallel_for("grp objective",
                           RangePolicy<AssemblyExec>(0,wts.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                for (size_type pt=0; pt<wts.extent(1); ++pt) {
                  regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
                }
              });
              
              
              auto regvals_sc_host = create_mirror_view(regvals);
              deep_copy(regvals_sc_host,regvals);
              
              auto poffs = params->paramoffsets;
              for (size_t elem=0; elem<assembler->boundary_groups[block][grp]->numElem; ++elem) {
                              
                vector<GO> paramGIDs;
                params->paramDOF->getElementGIDs(assembler->boundary_groups[block][grp]->localElemID(elem),
                                                 paramGIDs, blocknames[block]);
                
                for (size_type pt=0; pt<regvals_sc_host.extent(1); ++pt) {
                  totaldiff[r] += regwt*regvals_sc_host(elem,pt);
                }
              }
  
            }
          }
          
          assembler->wkset[block]->isOnSide = false;
        }
        
      }
    }
  }
  
  
  // For now, we scalarize the objective functions by summing them
  ScalarT totalobj = 0.0;
  for (size_t r=0; r<totaldiff.size(); ++r) {
    totalobj += totaldiff[r];
  }
  
  //to gather contributions across processors
  ScalarT meep = 0.0;
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&totalobj,&meep);
  
  DFAD fullobj(numParams,meep);
  
  if (debug_level > 1 && Comm->getRank() == 0) {
    std::cout << "******** Finished PostprocessManager::computeObjective ..." << std::endl;
  }
  
  objectiveval += fullobj;
  

}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeObjectiveGradParam(vector<vector_RCP> & current_soln,
                                                         const ScalarT & current_time,
                                                         DFAD & objectiveval) {
  
  if (debug_level > 1 && Comm->getRank() == 0) {
    std::cout << "******** Starting PostprocessManager::computeObjectiveGradParam ..." << std::endl;
  }

#ifndef MrHyDE_NO_AD
  for (size_t r=0; r<objectives.size(); ++r) {
    DFAD newobj = 0.0;
    size_t block = objectives[r].block;
    if (assembler->wkset_AD[block]->isInitialized) {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time,
                                               assembler->wkset_AD[block],
                                               assembler->function_managers_AD[block]);
    }
    else if (assembler->wkset_AD2[block]->isInitialized) {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time,
                                               assembler->wkset_AD2[block],
                                               assembler->function_managers_AD2[block]);
    }
    else if (assembler->wkset_AD4[block]->isInitialized) {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time,
                                               assembler->wkset_AD4[block],
                                               assembler->function_managers_AD4[block]);
    }
    else if (assembler->wkset_AD8[block]->isInitialized) {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time,
                                               assembler->wkset_AD8[block],
                                               assembler->function_managers_AD8[block]);
    }
    else if (assembler->wkset_AD16[block]->isInitialized) {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time,
                                               assembler->wkset_AD16[block],
                                               assembler->function_managers_AD16[block]);
    }
    else if (assembler->wkset_AD18[block]->isInitialized) {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time,
                                               assembler->wkset_AD18[block],
                                               assembler->function_managers_AD18[block]);
    }
    else if (assembler->wkset_AD24[block]->isInitialized) {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time,
                                               assembler->wkset_AD24[block],
                                               assembler->function_managers_AD24[block]);
    }
    else if (assembler->wkset_AD32[block]->isInitialized) {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time,
                                               assembler->wkset_AD32[block],
                                               assembler->function_managers_AD32[block]);
    }
    
    objectiveval += newobj;
  }

  saveObjectiveData(objectiveval);
#endif

  if (debug_level > 1 && Comm->getRank() == 0) {
    std::cout << "******** Finished PostprocessManager::computeObjectiveGradParam ..." << std::endl;
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
template<class EvalT>
DFAD PostprocessManager<Node>::computeObjectiveGradParam(const size_t & obj, vector<vector_RCP> & current_soln,
                                                         const ScalarT & current_time,
                                                         Teuchos::RCP<Workset<EvalT> > & wset,
                                                         Teuchos::RCP<FunctionManager<EvalT> > & fman) {

  Teuchos::TimeMonitor localtimer(*objectiveTimer);
  
  if (debug_level > 1 && Comm->getRank() == 0) {
    std::cout << "******** Starting PostprocessManager::computeObjectiveGradParam<EvalT> ..." << std::endl;
  }
  
  DFAD fullobj = 0.0;
  
#ifndef MrHyDE_NO_AD

  typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
  
  // Objective function values
  ScalarT objval = 0.0;
  
  int numParams = params->num_active_params + params->globalParamUnknowns;
  size_t block = objectives[obj].block;

  // Objective function gradients w.r.t params
  vector<ScalarT> gradient(numParams, 0.0);
  
  //for (size_t r=0; r<objectives.size(); ++r) {
    if (objectives[obj].type == "integrated control"){
      
      // First, compute objective value and deriv. w.r.t scalar params
      params->sacadoizeParams(true);
      
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
        View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
        //assembler->computeObjectiveGrad(block, grp, objectives[r].name, objsum_dev);
        
        auto wts = assembler->groups[block][grp]->wts;
        
        assembler->updateWorksetAD(block, grp, 0, 0, true);
        
        auto obj_dev = fman->evaluate(objectives[obj].name,"ip");
        
        Kokkos::View<EvalT[1],AssemblyDevice> objsum("sum of objective");
        parallel_for("grp objective",
                     RangePolicy<AssemblyExec>(0,wts.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          EvalT tmpval = 0.0;
          for (size_type pt=0; pt<wts.extent(1); pt++) {
            tmpval += obj_dev(elem,pt)*wts(elem,pt);
          }
          Kokkos::atomic_add(&(objsum(0)),tmpval);
        });
        
        parallel_for("grp objective",
                     RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                     KOKKOS_LAMBDA (const size_type p ) {
          size_t numder = static_cast<size_t>(objsum(0).size());
          if (p==0) {
            objsum_dev(p) = objsum(0).val();
          }
          else if (p <= numder) {
            objsum_dev(p) = objsum(0).fastAccessDx(p-1);
          }
        });
        
        auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
        Kokkos::deep_copy(objsum_host,objsum_dev);
        
        // Update the objective function value
        objval += objectives[obj].weight*objsum_host(0);
        
        // Update the gradients w.r.t scalar active parameters
        for (size_t p=0; p<params->num_active_params; p++) {
          gradient[p] += objectives[obj].weight*objsum_host(p+1);
        }
      }
      
      
      // Next, deriv w.r.t discretized params
      if (params->globalParamUnknowns > 0) {
        
        params->sacadoizeParams(false);
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          
          auto wts = assembler->groups[block][grp]->wts;
          
          assembler->updateWorksetAD(block, grp, 3, 0, true);
          
          auto obj_dev = fman->evaluate(objectives[obj].name,"ip");
          
          Kokkos::View<EvalT[1],AssemblyDevice> objsum("sum of objective");
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            EvalT tmpval = 0.0;
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              tmpval += obj_dev(elem,pt)*wts(elem,pt);
            }
            Kokkos::atomic_add(&(objsum(0)),tmpval);
          });
          
          View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
          
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                       KOKKOS_LAMBDA (const size_type p ) {
            size_t numder = static_cast<size_t>(objsum(0).size());
            if (p==0) {
              objsum_dev(p) = objsum(0).val();
            }
            else if (p <= numder) {
              objsum_dev(p) = objsum(0).fastAccessDx(p-1);
            }
          });
          
          auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
          Kokkos::deep_copy(objsum_host,objsum_dev);
          auto poffs = params->paramoffsets;
          
          for (size_t c=0; c<assembler->groups[block][grp]->numElem; c++) {
            vector<GO> paramGIDs;
            params->paramDOF->getElementGIDs(assembler->groups[block][grp]->localElemID[c],
                                             paramGIDs, blocknames[block]);
            
            for (size_t pp=0; pp<poffs.size(); ++pp) {
              for (size_t row=0; row<poffs[pp].size(); row++) {
                GO rowIndex = paramGIDs[poffs[pp][row]];
                int poffset = 1+poffs[pp][row];
                gradient[rowIndex+params->num_active_params] += objectives[obj].weight*objsum_host(poffset);
              }
            }
          }
        }
        
      }
    }
    else if (objectives[obj].type == "discrete control") {
      for (size_t set=0; set<current_soln.size(); ++set) {
        vector_RCP D_soln;
        bool fnd = datagen_soln[set]->extract(D_soln, 0, current_time);
        if (fnd) {
          vector_RCP diff = linalg->getNewVector(set);
          vector_RCP F_no = linalg->getNewVector(set);
          vector_RCP D_no = linalg->getNewVector(set);
          F_no->doExport(*(current_soln[set]), *(linalg->exporter[set]), Tpetra::REPLACE);
          D_no->doExport(*D_soln, *(linalg->exporter[set]), Tpetra::REPLACE);
          
          diff->update(1.0, *F_no, 0.0);
          diff->update(-1.0, *D_no, 1.0);
          Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> objn(1);
          diff->norm2(objn);
          if (Comm->getRank() == 0) {
            objval += objectives[obj].weight*objn[0]*objn[0];
          }
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: did not find a data-generating solution");
        }
      }
    }
    else if (objectives[obj].type == "integrated response") {
      
      // First, compute objective value and deriv. w.r.t scalar params
      //if (params->num_active_params > 0) {
        params->sacadoizeParams(true); // seed active
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
          auto wts = assembler->groups[block][grp]->wts;
            
          assembler->updateWorkset(block, grp, 0, 0, true);
        
          auto obj_dev = fman->evaluate(objectives[obj].name+" response","ip");
        
          Kokkos::View<EvalT[1],AssemblyDevice> objsum("sum of objective");
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            EvalT tmpval = 0.0;
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              tmpval += obj_dev(elem,pt)*wts(elem,pt);
            }
            Kokkos::atomic_add(&(objsum(0)),tmpval);
          });
        
          View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
        
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                       KOKKOS_LAMBDA (const size_type p ) {
            size_t numder = static_cast<size_t>(objsum(0).size());
            if (p==0) {
              objsum_dev(p) = objsum(0).val();
            }
            else if (p <= numder) {
              objsum_dev(p) = objsum(0).fastAccessDx(p-1);
            }
          });
        
          auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
          Kokkos::deep_copy(objsum_host,objsum_dev);
        
          // Update the objective function value
          objval += objsum_host(0);
        
          // Update the gradients w.r.t scalar active parameters
          for (size_t p=0; p<params->num_active_params; p++) {
            gradient[p] += objsum_host(p+1);
          }
        }
      
        if (compute_response) {
          if (objectives[obj].save_data) {
            objectives[obj].response_times.push_back(current_time);
            objectives[obj].scalar_response_data.push_back(objval);
            if (verbosity >= 10) {
              double localval = objval;
              double globalval = 0.0;
              Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
              if (Comm->getRank() == 0) {
                cout << objectives[obj].name << " on block " << blocknames[block] << ": " << globalval << endl;
              }
            }
          }
        }
      //}

      // Next, deriv w.r.t discretized params
      if (params->globalParamUnknowns > 0) {
        
        params->sacadoizeParams(false);
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          
          auto wts = assembler->groups[block][grp]->wts;
          
          assembler->updateWorksetAD(block, grp, 3,0,true);
        
          auto obj_dev = fman->evaluate(objectives[obj].name+" response","ip");
          
          Kokkos::View<EvalT[1],AssemblyDevice> objsum("sum of objective");
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            EvalT tmpval = 0.0;
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              tmpval += obj_dev(elem,pt)*wts(elem,pt);
            }
            Kokkos::atomic_add(&(objsum(0)),tmpval);
          });
          
          View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
          
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                       KOKKOS_LAMBDA (const size_type p ) {
            size_t numder = static_cast<size_t>(objsum(0).size());
            if (p==0) {
              objsum_dev(p) = objsum(0).val();
            }
            else if (p <= numder) {
              objsum_dev(p) = objsum(0).fastAccessDx(p-1);
            }
          });
          
          auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
          Kokkos::deep_copy(objsum_host,objsum_dev);
          auto poffs = params->paramoffsets;
          
          for (size_t c=0; c<assembler->groups[block][grp]->numElem; c++) {
            vector<GO> paramGIDs;
            params->paramDOF->getElementGIDs(assembler->groups[block][grp]->localElemID[c],
                                             paramGIDs, blocknames[block]);
            
            for (size_t pp=0; pp<poffs.size(); ++pp) {
              for (size_t row=0; row<poffs[pp].size(); row++) {
                GO rowIndex = paramGIDs[poffs[pp][row]];
                int poffset = 1+poffs[pp][row];
                //gradients[r][rowIndex+params->num_active_params] += objectives[r].weight*objsum_host(poffset);
                gradient[rowIndex+params->num_active_params] += objsum_host(poffset);
              }
            }
          }
        }
        
      }
      
      // Right now, totaldiff = response
      //             gradient = dresponse / dp
      // We want    totaldiff = wt*(response-target)^2
      //             gradient = 2*wt*(response-target)*dresponse/dp
      
      ScalarT diff = objval - objectives[obj].target;
      objval = objectives[obj].weight*diff*diff;
      for (size_t g=0; g<gradient.size(); ++g) {
        gradient[g] = 2.0*objectives[obj].weight*diff*gradient[g];
      }
      
      
    }
    else if (objectives[obj].type == "sensors" || objectives[obj].type == "sensor response" || objectives[obj].type == "pointwise response") {
      if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln) {
        // don't do anything for this use case
      }
      else {
        Kokkos::View<ScalarT*,HostDevice> sensordat;
        if (compute_response) {
          sensordat = Kokkos::View<ScalarT*,HostDevice>("sensor data to save",objectives[obj].numSensors);
          objectives[obj].response_times.push_back(current_time);
        }
      
        for (size_t pt=0; pt<objectives[obj].numSensors; ++pt) {
          size_t tindex = 0;
          bool foundtime = false;
          for (size_type t=0; t<objectives[obj].sensor_times.extent(0); ++t) {
            if (std::abs(current_time - objectives[obj].sensor_times(t)) < 1.0e-12) {
              foundtime = true;
              tindex = t;
            }
          }
          
          if (compute_response || foundtime) {
          
            // First compute objective and derivative w.r.t scalar params
            params->sacadoizeParams(true);
            
            size_t grp = objectives[obj].sensor_owners(pt,0);
            size_t elem = objectives[obj].sensor_owners(pt,1);
            wset->isOnPoint = true;
            auto x = wset->getScalarField("x");
            x(0,0) = objectives[obj].sensor_points(pt,0);
            if (dimension > 1) {
              auto y = wset->getScalarField("y");
              y(0,0) = objectives[obj].sensor_points(pt,1);
            }
            if (dimension > 2) {
              auto z = wset->getScalarField("z");
              z(0,0) = objectives[obj].sensor_points(pt,2);
            }
            
            auto numDOF = assembler->groupData[block]->num_dof;
            View_EvalT2 u_dof("u_dof",numDOF.extent(0),assembler->groups[block][grp]->LIDs[0].extent(1)); // hard coded
            auto cu = subview(assembler->groups[block][grp]->sol[0],elem,ALL(),ALL()); // hard coded
            parallel_for("grp response get u",
                        RangePolicy<AssemblyExec>(0,u_dof.extent(0)),
                        KOKKOS_LAMBDA (const size_type n ) {
              for (size_type n=0; n<numDOF.extent(0); n++) {
                for( int i=0; i<numDOF(n); i++ ) {
                  u_dof(n,i) = cu(n,i);
                }
              }
            });
            
            // Map the local solution to the solution and gradient at ip
            View_EvalT2 u_ip("u_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
            View_EvalT2 ugrad_ip("ugrad_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
            
            for (size_type var=0; var<numDOF.extent(0); var++) {
              auto cbasis = objectives[obj].sensor_basis[wset->usebasis[var]];
              auto cbasis_grad = objectives[obj].sensor_basis_grad[wset->usebasis[var]];
              auto u_sv = subview(u_ip, var, ALL());
              auto u_dof_sv = subview(u_dof, var, ALL());
              auto ugrad_sv = subview(ugrad_ip, var, ALL());
              
              parallel_for("grp response sensor uip",
                          RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                          KOKKOS_LAMBDA (const int dof ) {
                u_sv(0) += u_dof_sv(dof)*cbasis(pt,dof,0,0);
                for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                  ugrad_sv(dim) += u_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                }
              });
            }
            
            wset->setSolutionPoint(u_ip);
            wset->setSolutionGradPoint(ugrad_ip);
            
            // Map the local discretized params to param and grad at ip
            if (params->globalParamUnknowns > 0) {
              auto numParamDOF = assembler->groupData[block]->num_param_dof;
              
              View_EvalT2 p_dof("p_dof",numParamDOF.extent(0),assembler->groups[block][grp]->paramLIDs.extent(1));
              auto cp = subview(assembler->groups[block][grp]->param,elem,ALL(),ALL());
              parallel_for("grp response get u",
                          RangePolicy<AssemblyExec>(0,p_dof.extent(0)),
                          KOKKOS_LAMBDA (const size_type n ) {
                for (size_type n=0; n<numParamDOF.extent(0); n++) {
                  for( int i=0; i<numParamDOF(n); i++ ) {
                    p_dof(n,i) = cp(n,i);
                  }
                }
              });
              
              View_EvalT2 p_ip("p_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
              View_EvalT2 pgrad_ip("pgrad_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
              
              for (size_type var=0; var<numParamDOF.extent(0); var++) {
                int bnum = wset->paramusebasis[var];
                auto cbasis = objectives[obj].sensor_basis[bnum];
                auto cbasis_grad = objectives[obj].sensor_basis_grad[bnum];
                auto p_sv = subview(p_ip, var, ALL());
                auto p_dof_sv = subview(p_dof, var, ALL());
                auto pgrad_sv = subview(pgrad_ip, var, ALL());
                
                parallel_for("grp response sensor uip",
                            RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                            KOKKOS_LAMBDA (const int dof ) {
                  p_sv(0) += p_dof_sv(dof)*cbasis(pt,dof,0,0);
                  for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                    pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                  }
                });
              }
              
              wset->setParamPoint(p_ip);
              
              wset->setParamGradPoint(pgrad_ip);
            }
            
            // Evaluate the response
            auto rdata = fman->evaluate(objectives[obj].name+" response","point");
            
            if (compute_response) {
              sensordat(pt) = rdata(0,0).val();
            }
            
            if (compute_objective) {
              
              // Update the value of the objective
              AD diff = rdata(0,0) - objectives[obj].sensor_data(pt,tindex);
              AD sdiff = objectives[obj].weight*diff*diff;
              objval += sdiff.val();
              
              // Update the gradient w.r.t scalar active parameters
              for (size_t p=0; p<params->num_active_params; p++) {
                gradient[p] += sdiff.fastAccessDx(p);
              }
              
              // Discretized parameters
              if (params->globalParamUnknowns > 0) {
                
                // Need to compute derivative w.r.t discretized params
                params->sacadoizeParams(false);
                
                auto numParamDOF = assembler->groupData[block]->num_param_dof;
                auto poff = wset->paramoffsets;
                View_EvalT2 p_dof("p_dof",numParamDOF.extent(0),assembler->groups[block][grp]->paramLIDs.extent(1));
                auto cp = subview(assembler->groups[block][grp]->param,elem,ALL(),ALL());
                parallel_for("grp response get u",
                            RangePolicy<AssemblyExec>(0,p_dof.extent(0)),
                            KOKKOS_LAMBDA (const size_type n ) {
                  for (size_type n=0; n<numParamDOF.extent(0); n++) {
                    for( int i=0; i<numParamDOF(n); i++ ) {
                      p_dof(n,i) = AD(MAXDERIVS,poff(n,i),cp(n,i));
                    }
                  }
                });
                
                View_EvalT2 p_ip("p_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
                View_EvalT2 pgrad_ip("pgrad_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
                
                for (size_type var=0; var<numParamDOF.extent(0); var++) {
                  int bnum = wset->paramusebasis[var];
                  auto cbasis = objectives[obj].sensor_basis[bnum];
                  auto cbasis_grad = objectives[obj].sensor_basis_grad[bnum];
                  auto p_sv = subview(p_ip, var, ALL());
                  auto p_dof_sv = subview(p_dof, var, ALL());
                  auto pgrad_sv = subview(pgrad_ip, var, ALL());
                  
                  parallel_for("grp response sensor uip",
                              RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                              KOKKOS_LAMBDA (const int dof ) {
                    p_sv(0) += p_dof_sv(dof)*cbasis(pt,dof,0,0);
                    for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                      pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                    }
                  });
                }
                
                wset->setParamPoint(p_ip);
                wset->setParamGradPoint(pgrad_ip);
                
                // Evaluate the response
                auto rdata = fman->evaluate(objectives[obj].name+" response","point");
                EvalT diff = rdata(0,0) - objectives[obj].sensor_data(pt,tindex);
                EvalT sdiff = objectives[obj].weight*diff*diff;
                
                auto poffs = params->paramoffsets;
                vector<GO> paramGIDs;
                params->paramDOF->getElementGIDs(assembler->groups[block][grp]->localElemID[elem],
                                                paramGIDs, blocknames[block]);
                
                for (size_t pp=0; pp<poffs.size(); ++pp) {
                  for (size_t row=0; row<poffs[pp].size(); row++) {
                    GO rowIndex = paramGIDs[poffs[pp][row]] + params->num_active_params;
                    int poffset = poffs[pp][row];
                    gradient[rowIndex] += sdiff.fastAccessDx(poffset);
                  }
                }

              }
              
            }
            wset->isOnPoint = false;

          } // found time
        } // sensor points
        
        if (compute_response) {
          objectives[obj].response_data.push_back(sensordat);
        }
      } // objectives
    }
    // ========================================================================================
    // Add regularizations (reg funcs are tied to objectives and objectives can have more than one reg)
    // ========================================================================================

    for (size_t reg=0; reg<objectives[obj].regularizations.size(); ++reg) {
      if (objectives[obj].regularizations[reg].type == "integrated") {
        if (objectives[obj].regularizations[reg].location == "volume") {
          params->sacadoizeParams(false);
          ScalarT regwt = objectives[obj].regularizations[reg].weight;
          for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
            
            auto wts = assembler->groups[block][grp]->wts;
            
            assembler->updateWorksetAD(block, grp, 3, 0, true);
            
            auto regvals_tmp = fman->evaluate(objectives[obj].regularizations[reg].name,"ip");
            View_EvalT2 regvals("regvals",wts.extent(0),wts.extent(1));
            
            parallel_for("grp objective",
                         RangePolicy<AssemblyExec>(0,wts.extent(0)),
                         KOKKOS_LAMBDA (const size_type elem ) {
              for (size_type pt=0; pt<wts.extent(1); ++pt) {
                regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
              }
            });
            
            
            View_Sc3 regvals_sc("scalar version of AD view",wts.extent(0),wts.extent(1),MAXDERIVS+1);
            parallel_for("grp objective",
                         RangePolicy<AssemblyExec>(0,wts.extent(0)),
                         KOKKOS_LAMBDA (const size_type elem ) {
              for (size_type pt=0; pt<wts.extent(1); ++pt) {
                regvals_sc(elem,pt,0) = regvals(elem,pt).val();
                for (size_type d=0; d<regvals_sc.extent(2)-1; ++d) {
                  regvals_sc(elem,pt,d+1) = regvals(elem,pt).fastAccessDx(d);
                }
              }
            });
            
            auto regvals_sc_host = create_mirror_view(regvals_sc);
            deep_copy(regvals_sc_host,regvals_sc);
            
            auto poffs = params->paramoffsets;
            for (size_t elem=0; elem<assembler->groups[block][grp]->numElem; ++elem) {
                            
              vector<GO> paramGIDs;
              params->paramDOF->getElementGIDs(assembler->groups[block][grp]->localElemID(elem),
                                               paramGIDs, blocknames[block]);
              
              for (size_type pt=0; pt<regvals_sc_host.extent(1); ++pt) {
                objval += regwt*regvals_sc_host(elem,pt,0);
                for (size_t pp=0; pp<poffs.size(); ++pp) {
                  for (size_t row=0; row<poffs[pp].size(); row++) {
                    GO rowIndex = paramGIDs[poffs[pp][row]] + params->num_active_params;
                    int poffset = poffs[pp][row];
                    gradient[rowIndex] += regwt*regvals_sc_host(elem,pt,poffset+1);
                  }
                }
              }
            }
          }
          
        }
        else if (objectives[obj].regularizations[reg].location == "boundary") {
          string bname = objectives[obj].regularizations[reg].boundary_name;
          params->sacadoizeParams(false);
          ScalarT regwt = objectives[obj].regularizations[reg].weight;
          wset->isOnSide = true;
          for (size_t grp=0; grp<assembler->boundary_groups[block].size(); ++grp) {
            if (assembler->boundary_groups[block][grp]->sidename == bname) {
              
              auto wts = assembler->boundary_groups[block][grp]->wts;
              
              assembler->updateWorksetBoundaryAD(block, grp, 3, 0, true);
              
              auto regvals_tmp = fman->evaluate(objectives[obj].regularizations[reg].name,"side ip");
              View_EvalT2 regvals("regvals",wts.extent(0),wts.extent(1));
              
              parallel_for("grp objective",
                           RangePolicy<AssemblyExec>(0,wts.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                for (size_type pt=0; pt<wts.extent(1); ++pt) {
                  regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
                }
              });
              
              View_Sc3 regvals_sc("scalar version of AD view",wts.extent(0),wts.extent(1),MAXDERIVS+1);
              parallel_for("grp objective",
                           RangePolicy<AssemblyExec>(0,wts.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                for (size_type pt=0; pt<wts.extent(1); ++pt) {
                  regvals_sc(elem,pt,0) = regvals(elem,pt).val();
                  for (size_type d=0; d<regvals_sc.extent(2)-1; ++d) {
                    regvals_sc(elem,pt,d+1) = regvals(elem,pt).fastAccessDx(d);
                  }
                }
                
              });
              
              auto regvals_sc_host = create_mirror_view(regvals_sc);
              deep_copy(regvals_sc_host,regvals_sc);
              
              auto poffs = params->paramoffsets;
              for (size_t elem=0; elem<assembler->boundary_groups[block][grp]->numElem; ++elem) {
                              
                vector<GO> paramGIDs;
                params->paramDOF->getElementGIDs(assembler->boundary_groups[block][grp]->localElemID(elem),
                                                 paramGIDs, blocknames[block]);
                
                for (size_type pt=0; pt<regvals_sc_host.extent(1); ++pt) {
                  objval += regwt*regvals_sc_host(elem,pt,0);
                  for (size_t pp=0; pp<poffs.size(); ++pp) {
                    for (size_t row=0; row<poffs[pp].size(); row++) {
                      GO rowIndex = paramGIDs[poffs[pp][row]] + params->num_active_params;
                      int poffset = poffs[pp][row];
                      gradient[rowIndex] += regwt*regvals_sc_host(elem,pt,poffset+1);
                    }
                  }
                }
              }
  
              /*
              auto obj_dev = functionManagers[block]->evaluate(objectives[r].regularizations[reg].name,"side ip");
              
              Kokkos::View<AD[1],AssemblyDevice> objsum("sum of objective");
              parallel_for("grp objective",
                           RangePolicy<AssemblyExec>(0,obj_dev.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                AD tmpval = 0.0;
                for (size_type pt=0; pt<obj_dev.extent(1); pt++) {
                  tmpval += obj_dev(elem,pt)*wts(elem,pt);
                }
                Kokkos::atomic_add(&(objsum(0)),tmpval);
              });
              
              View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
              
              parallel_for("grp objective",
                           RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                           KOKKOS_LAMBDA (const size_type p ) {
                size_t numder = static_cast<size_t>(objsum(0).size());
                if (p==0) {
                  objsum_dev(p) = objsum(0).val();
                }
                else if (p <= numder) {
                  objsum_dev(p) = objsum(0).fastAccessDx(p-1);
                }
              });
              
              auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
              Kokkos::deep_copy(objsum_host,objsum_dev);
              
              totaldiff[r] += regwt*objsum_host(0);
              auto poffs = params->paramoffsets;
              for (size_t c=0; c<assembler->boundary_groups[block][grp]->numElem; c++) {
                vector<GO> paramGIDs;
                params->paramDOF->getElementGIDs(assembler->boundary_groups[block][grp]->localElemID(c),
                                                 paramGIDs, blocknames[block]);
                
                for (size_t pp=0; pp<poffs.size(); ++pp) {
                  for (size_t row=0; row<poffs[pp].size(); row++) {
                    GO rowIndex = paramGIDs[poffs[pp][row]];
                    int poffset = 1+poffs[pp][row];
                    gradients[r][rowIndex+params->num_active_params] += regwt*objsum_host(poffset);
                  }
                }
              }*/
  
            }
          }
          
          wset->isOnSide = false;
        }
        
      }
    }
  //}
  
  //to gather contributions across processors
  ScalarT meep = 0.0;
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&objval,&meep);
  
  fullobj = DFAD(numParams,meep);
  
  for (int j=0; j<numParams; j++) {
    ScalarT dval = 0.0;
    ScalarT ldval = gradient[j];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&ldval,&dval);
    fullobj.fastAccessDx(j) = dval;
  }
  
  params->sacadoizeParams(false);
  
  //objectiveval += fullobj;
  
  

#endif

  if (debug_level > 1 && Comm->getRank() == 0) {
    std::cout << "******** Finished PostprocessManager::computeObjectiveGradParam<EvalT> ..." << std::endl;
  }
  return fullobj;

}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeSensorSolution(vector<vector_RCP> & current_soln,
                                                     const ScalarT & current_time) {
  
  Teuchos::TimeMonitor localtimer(*sensorSolutionTimer);
  
  if (debug_level > 1 && Comm->getRank() == 0) {
    std::cout << "******** Starting PostprocessManager::computeSensorSolution ..." << std::endl;
  }
  
  for (size_t r=0; r<objectives.size(); ++r) {
    
    if (objectives[r].type == "sensors" || objectives[r].type == "sensor response" || objectives[r].type == "pointwise response") {
      if (objectives[r].compute_sensor_soln || objectives[r].compute_sensor_average_soln) {
        
        size_t block = objectives[r].block;
            
        int numSols = 0;
        for (size_t set=0; set<varlist.size(); ++set) {
          numSols += varlist[set][block].size();
        }
        Kokkos::View<ScalarT***,HostDevice> sensordat("sensor solution", objectives[r].numSensors, numSols, dimension);
        objectives[r].response_times.push_back(current_time); // might store this somewhere else
        
        for (size_t pt=0; pt<objectives[r].numSensors; ++pt) {
          
          size_t solprog = 0;
          int grp_owner = objectives[r].sensor_owners(pt,0);
          int elem_owner = objectives[r].sensor_owners(pt,1);
          for (size_t set=0; set<varlist.size(); ++set) {  
            auto numDOF = assembler->groupData[block]->set_num_dof_host[set];
            auto cu = subview(assembler->groups[block][grp_owner]->sol[set],elem_owner,ALL(),ALL());
            auto cu_host = create_mirror_view(cu);
            //KokkosTools::print(assembler->groups[block][grp_owner]->u[set]);
            deep_copy(cu_host,cu);
            for (size_type var=0; var<numDOF.extent(0); var++) {
              auto cbasis = objectives[r].sensor_basis[assembler->wkset[block]->set_usebasis[set][var]];
              for (size_type dof=0; dof<cbasis.extent(1); ++dof) {
                for (size_type dim=0; dim<cbasis.extent(3); ++dim) {
                  //sensordat(pt,solprog,dim) += cu_host(solprog,dof)*cbasis(pt,dof,0,dim);
                  sensordat(pt,solprog,dim) += cu_host(var,dof)*cbasis(pt,dof,0,dim);
                }
              }  
              solprog++;
              
            }
            //KokkosTools::print(sensordat);
            
          }
            
        } // sensor points
        //KokkosTools::print(sensordat);

        if (objectives[r].output_type == "dft") {
          std::complex<double> imagi(0.0,1.0);
          int N = objectives[r].dft_num_freqs;
          Kokkos::View<std::complex<double>****,HostDevice> newdft;
          if (objectives[r].sensor_solution_dft.extent(0) == 0) {
            newdft = Kokkos::View<std::complex<double>****,HostDevice>("KV of complex DFT",sensordat.extent(0),
                                                                       sensordat.extent(1), sensordat.extent(2),N);
            objectives[r].sensor_solution_dft = newdft;
          }
          else {
            newdft = objectives[r].sensor_solution_dft;
          }
          for (int j=0; j<N; ++j) {
            for (int k=0; k<N; ++k) {
              double freq = static_cast<double>(k*j/N);
              freq *= -2.0*PI;
              for (size_type n=0; n<newdft.extent(0); ++n) {
                for (size_type m=0; m<newdft.extent(1); ++m) {
                  for (size_type p=0; p<newdft.extent(2); ++p) {                    
                    newdft(n,m,p,k) += sensordat(n,m,p)*std::exp(imagi*freq);
                  }
                }
              }
            }
          }

        }
        else {
          objectives[r].sensor_solution_data.push_back(sensordat);
        }
        
      } // objectives
    }
    
  }  
  
  if (debug_level > 1 && Comm->getRank() == 0) {
    std::cout << "******** Finished PostprocessManager::computeSensorSolutions ..." << std::endl;
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeObjectiveGradState(const size_t & set,
                                                         vector_RCP & current_soln,
                                                         const ScalarT & current_time,
                                                         const ScalarT & deltat,
                                                         vector_RCP & grad) {
  
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting PostprocessManager::computeObjectiveGradState ..." << std::endl;
    }
  }

#ifndef MrHyDE_NO_AD
  for (size_t r=0; r<objectives.size(); ++r) {
    size_t block = objectives[r].block;
    if (assembler->wkset_AD[block]->isInitialized) {
      this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                      assembler->wkset_AD[block],
                                      assembler->function_managers_AD[block]);
    }
    else if (assembler->wkset_AD2[block]->isInitialized) {
      this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                      assembler->wkset_AD2[block],
                                      assembler->function_managers_AD2[block]);
    }
    else if (assembler->wkset_AD4[block]->isInitialized) {
      this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                      assembler->wkset_AD4[block],
                                      assembler->function_managers_AD4[block]);
    }
    else if (assembler->wkset_AD8[block]->isInitialized) {
      this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                      assembler->wkset_AD8[block],
                                      assembler->function_managers_AD8[block]);
    }
    else if (assembler->wkset_AD16[block]->isInitialized) {
      this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                      assembler->wkset_AD16[block],
                                      assembler->function_managers_AD16[block]);
    }
    else if (assembler->wkset_AD18[block]->isInitialized) {
      this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                      assembler->wkset_AD18[block],
                                      assembler->function_managers_AD18[block]);
    }
    else if (assembler->wkset_AD24[block]->isInitialized) {
      this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                      assembler->wkset_AD24[block],
                                      assembler->function_managers_AD24[block]);
    }
    else if (assembler->wkset_AD32[block]->isInitialized) {
      this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                      assembler->wkset_AD32[block],
                                      assembler->function_managers_AD32[block]);
    }
  }
#endif

  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Finished PostprocessManager::computeObjectiveGradState ..." << std::endl;
    }
  }

}

// ========================================================================================
// ========================================================================================

template<class Node>
template<class EvalT>
void PostprocessManager<Node>::computeObjectiveGradState(const size_t & set,
                                                         const size_t & obj,
                                                         vector_RCP & current_soln,
                                                         const ScalarT & current_time,
                                                         const ScalarT & deltat,
                                                         vector_RCP & grad,
                                                         Teuchos::RCP<Workset<EvalT> > & wset,
                                                         Teuchos::RCP<FunctionManager<EvalT> > & fman) {
  
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting PostprocessManager::computeObjectiveGradState<EvalT> ..." << std::endl;
    }
  }
  
#ifndef MrHyDE_NO_AD

  typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
  typedef Kokkos::View<EvalT***,ContLayout,AssemblyDevice> View_EvalT3;
  typedef Kokkos::View<EvalT****,ContLayout,AssemblyDevice> View_EvalT4;
  
  DFAD totaldiff = 0.0;
  //AD regDomain = 0.0;
  //AD regBoundary = 0.0;
  
  params->sacadoizeParams(false);
  
  int numParams = params->num_active_params + params->globalParamUnknowns;
  size_t block = objectives[obj].block;
  
  vector<ScalarT> regGradient(numParams);
  vector<ScalarT> dmGradient(numParams);
  
  typedef typename Node::device_type LA_device;
  typedef typename Node::execution_space LA_exec;
  
  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }
  // LIDs are on AssemblyDevice.  If the AssemblyDevice memory is accessible, then these are fine.
  // Copy of LIDs is stored on HostDevice.
  bool use_host_LIDs = false;
  if (!data_avail) {
    if (Kokkos::SpaceAccessibility<LA_exec, HostDevice::memory_space>::accessible) {
      use_host_LIDs = true;
    }
  }
    
  
    if (objectives[obj].type == "integrated control"){
      auto grad_over = linalg->getNewOverlappedVector(set);
      auto grad_tmp = linalg->getNewVector(set);
      auto grad_view = grad_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      
      auto offsets = wset->offsets;
      auto numDOF = assembler->groupData[block]->num_dof;
      
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
        size_t numElem = assembler->groups[block][grp]->numElem;
        size_t numip = wset->numip;
        
        View_Sc3 local_grad("local contrib to dobj/dstate",
                            assembler->groups[block][grp]->numElem,
                            assembler->groups[block][grp]->LIDs[set].extent(1),1);
        
        auto local_grad_ladev = create_mirror(LA_exec(),local_grad);
        
        //for (int w=0; w<dimension+1; ++w) {
        for (int w=0; w<1; ++w) {
          
          // Seed the state and compute the solution at the ip
          if (w==0) {
            assembler->updateWorksetAD(block, grp, 1,0, true);
          }
          else {
            View_EvalT3 u_dof("u_dof",numElem,numDOF.extent(0),
                           assembler->groups[block][grp]->LIDs[set].extent(1)); //(numElem, numVars, numDOF)
            auto u = assembler->groups[block][grp]->sol[set];
            parallel_for("grp response get u",
                         RangePolicy<AssemblyExec>(0,u_dof.extent(0)),
                         KOKKOS_LAMBDA (const size_type e ) {
              EvalT dummyval = 0.0;
              for (size_type n=0; n<numDOF.extent(0); n++) { // numDOF is on device
                for( int i=0; i<numDOF(n); i++ ) {
                  u_dof(e,n,i) = EvalT(dummyval.size(),offsets(n,i),u(e,n,i)); // offsets is on device
                }
              }
            });
            
            View_EvalT4 u_ip("u_ip",numElem,numDOF.extent(0),numip,dimension);
            View_EvalT4 ugrad_ip("ugrad_ip",numElem,numDOF.extent(0),numip,dimension);
            
            for (size_type var=0; var<numDOF.extent(0); var++) {
              int bnum = wset->usebasis[var];
              std::string btype = wset->basis_types[bnum];
              if (btype == "HCURL" || btype == "HDIV") {
                // TMW: this does not work yet
                auto cbasis = assembler->wkset_AD[block]->basis[bnum];
                auto u_sv = subview(u_ip, ALL(), var, ALL(), ALL());
                auto u_dof_sv = subview(u_dof, ALL(), var, ALL());
                parallel_for("grp response uip",
                             RangePolicy<AssemblyExec>(0,u_ip.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  for (size_type i=0; i<cbasis.extent(1); i++ ) {
                    for (size_type j=0; j<cbasis.extent(2); j++ ) {
                      for (size_type d=0; d<cbasis.extent(3); d++ ) {
                        u_sv(e,j,d) += u_dof_sv(e,i)*cbasis(e,i,j,d);
                      }
                    }
                  }
                });
              }
              else {
                auto cbasis = wset->basis[bnum];
                auto u_sv = subview(u_ip, ALL(), var, ALL(), 0);
                auto u_dof_sv = subview(u_dof, ALL(), var, ALL());
                parallel_for("grp response uip",
                             RangePolicy<AssemblyExec>(0,u_ip.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  for (size_type i=0; i<cbasis.extent(1); i++ ) {
                    for (size_type j=0; j<cbasis.extent(2); j++ ) {
                      u_sv(e,j) += u_dof_sv(e,i)*cbasis(e,i,j,0);
                    }
                  }
                });
              }
              
              if (btype == "HGRAD") {
                auto cbasis_grad = wset->basis_grad[bnum];
                auto u_dof_sv = subview(u_dof, ALL(), var, ALL());
                auto ugrad_sv = subview(ugrad_ip, ALL(), var, ALL(), ALL());
                parallel_for("grp response HGRAD",
                             RangePolicy<AssemblyExec>(0,u_ip.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  for (size_type i=0; i<cbasis_grad.extent(1); i++ ) {
                    for (size_type j=0; j<cbasis_grad.extent(2); j++ ) {
                      for (size_type s=0; s<cbasis_grad.extent(3); s++) {
                        ugrad_sv(e,j,s) += u_dof_sv(e,i)*cbasis_grad(e,i,j,s);
                      }
                    }
                  }
                });
              }
            }
            
            for (int s=0; s<dimension; s++) {
              auto ugrad_sv = subview(ugrad_ip, ALL(), ALL(), ALL(), s);
              if ((w-1) == s) {
                parallel_for("grp response seed grad 0",
                             RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  for (size_type n=0; n<numDOF.extent(0); n++) {
                    for(size_type j=0; j<ugrad_sv.extent(2); j++ ) {
                      ScalarT tmp = ugrad_sv(e,n,j).val();
                      ugrad_sv(e,n,j) = u_ip(e,n,j,0);
                      ugrad_sv(e,n,j) += -u_ip(e,n,j,0).val() + tmp;
                    }
                  }
                });
              }
              else {
                parallel_for("grp response seed grad 1",
                             RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  for (size_type n=0; n<numDOF.extent(0); n++) {
                    for(size_type j=0; j<ugrad_sv.extent(2); j++ ) {
                      ugrad_sv(e,n,j) = ugrad_sv(e,n,j).val();
                    }
                  }
                });
              }
              
            }
            parallel_for("grp response seed grad 2",
                         RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)),
                         KOKKOS_LAMBDA (const size_type e ) {
              for (size_type n=0; n<numDOF.extent(0); n++) {
                for(size_type j=0; j<u_ip.extent(2); j++ ) {
                  for(size_type s=0; s<u_ip.extent(3); s++ ) {
                    u_ip(e,n,j,s) = u_ip(e,n,j,s).val();
                  }
                }
              }
            });
            wset->setSolution(u_ip);
            wset->setSolutionGrad(ugrad_ip);
            
          }
          
          // Evaluate the objective
          auto obj_dev = fman->evaluate(objectives[obj].name,"ip");
          
          // Weight using volumetric integration weights
          auto wts = assembler->groups[block][grp]->wts;
          
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              obj_dev(elem,pt) *= objectives[obj].weight*wts(elem,pt);
            }
          });
          
          for (size_type n=0; n<numDOF.extent(0); n++) {
            int bnum = wset->usebasis[n];
            std::string btype = wset->basis_types[bnum];
            
            if (w == 0) {
              auto cbasis = wset->basis[bnum];
              
              if (btype == "HDIV" || btype == "HCURL") {

                auto tmpbasis = DRV("tmp basis",cbasis.extent(0),cbasis.extent(1),cbasis.extent(2),cbasis.extent(3));
                auto refbasis = assembler->groupData[block]->ref_basis[bnum];
                for (size_type e=0; e<tmpbasis.extent(0); ++e) {
                  for (size_type i=0; i<tmpbasis.extent(1); ++i) {
                    for (size_type j=0; j<tmpbasis.extent(2); ++j) {
                      for (size_type k=0; k<tmpbasis.extent(3); ++k) {
                        tmpbasis(e,i,j,k) = refbasis(i,j,k);
                      }
                    }
                  }
                }
                //auto tmpbasis2 = disc->applyOrientation(tmpbasis, assembler->groups[block][grp]->orientation,
                //                                        assembler->groupData[block]->basis_pointers[bnum] );
                parallel_for("grp adjust adjoint res",
                             RangePolicy<AssemblyExec>(0,local_grad.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  int nn = n; // TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        for (size_type d=0; d<cbasis.extent(3); d++) {
                          local_grad(e,offsets(nn,j),0) += -obj_dev(e,s).fastAccessDx(offsets(nn,i))*tmpbasis(e,j,s,d);
                        }
                      }
                    }
                  }
                });
              }
              else {
                parallel_for("grp adjust adjoint res",
                             RangePolicy<AssemblyExec>(0,local_grad.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(e,offsets(nn,j),0) += -obj_dev(e,s).fastAccessDx(offsets(nn,i))*cbasis(e,j,s,0);
                      }
                    }
                  }
                });
              }
            }
            else {
              
              if (btype == "HGRAD") {
                auto cbasis = wset->basis_grad[bnum];
                //auto cbasis_sv = subview(cbasis, ALL(), ALL(), ALL(), w-1);
                parallel_for("grp adjust adjoint res",
                             RangePolicy<AssemblyExec>(0,local_grad.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(e,offsets(nn,j),0) += -obj_dev(e,s).fastAccessDx(offsets(nn,i))*cbasis(e,j,s,w-1);
                      }
                    }
                  }
                });
                
                
              }
              
            }
            
          }
        }
        
        if (data_avail) {
          assembler->scatterRes(grad_view, local_grad, assembler->groups[block][grp]->LIDs[set]);
        }
        else {
          Kokkos::deep_copy(local_grad_ladev,local_grad);
          
          if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
            assembler->scatterRes(grad_view, local_grad_ladev, assembler->groups[block][grp]->LIDs_host[set]);
          }
          else { // LA_device = CUDA, AssemblyDevice = Host
            // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
            auto LIDs_dev = Kokkos::create_mirror(LA_exec(), assembler->groups[block][grp]->LIDs[set]);
            Kokkos::deep_copy(LIDs_dev,assembler->groups[block][grp]->LIDs[set]);
            assembler->scatterRes(grad_view, local_grad_ladev, LIDs_dev);
          }
          
        }
        
      }
      
      linalg->exportVectorFromOverlapped(set, grad_tmp, grad_over);
      grad->update(1.0, *grad_tmp, 1.0);
        
    }
    else if (objectives[obj].type == "integrated response") {
      auto grad_over = linalg->getNewOverlappedVector(set);
      auto grad_tmp = linalg->getNewVector(set);
      auto grad_view = grad_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      
      auto offsets = wset->offsets;
      auto numDOF = assembler->groupData[block]->num_dof;
      
      ScalarT intresp = 0.0;
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
        size_t numElem = assembler->groups[block][grp]->numElem;
        size_t numip = wset->numip;
        
        View_Sc3 local_grad("local contrib to dobj/dstate",
                            assembler->groups[block][grp]->numElem,
                            assembler->groups[block][grp]->LIDs[set].extent(1),1);
        
        auto local_grad_ladev = create_mirror(LA_exec(),local_grad);
                
        for (int w=0; w<dimension+1; ++w) {
        //for (int w=0; w<1; ++w) {
          
          // Seed the state and compute the solution at the ip
          if (w==0) {
            assembler->updateWorksetAD(block, grp, 1 ,0, true);
          }
          else {
            View_EvalT3 u_dof("u_dof",numElem,numDOF.extent(0),
                           assembler->groups[block][grp]->LIDs[set].extent(1)); //(numElem, numVars, numDOF)
            auto u = assembler->groups[block][grp]->sol[set];
            parallel_for("grp response get u",
                         RangePolicy<AssemblyExec>(0,u_dof.extent(0)),
                         KOKKOS_LAMBDA (const size_type e ) {
              EvalT dummyval = 0.0;
              for (size_type n=0; n<numDOF.extent(0); n++) { // numDOF is on device
                for( int i=0; i<numDOF(n); i++ ) {
                  u_dof(e,n,i) = EvalT(dummyval.size(),offsets(n,i),u(e,n,i)); // offsets is on device
                }
              }
            });
            
            View_EvalT4 u_ip("u_ip",numElem,numDOF.extent(0),numip,dimension);
            View_EvalT4 ugrad_ip("ugrad_ip",numElem,numDOF.extent(0),numip,dimension);
            
            for (size_type var=0; var<numDOF.extent(0); var++) {
              int bnum = wset->usebasis[var];
              std::string btype = wset->basis_types[bnum];
              if (btype == "HCURL" || btype == "HDIV") {
                // TMW: this does not work yet
              }
              else {
                auto cbasis = wset->basis[bnum];
                
                auto u_sv = subview(u_ip, ALL(), var, ALL(), 0);
                auto u_dof_sv = subview(u_dof, ALL(), var, ALL());
                parallel_for("grp response uip",
                             RangePolicy<AssemblyExec>(0,u_ip.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  for (size_type i=0; i<cbasis.extent(1); i++ ) {
                    for (size_type j=0; j<cbasis.extent(2); j++ ) {
                      u_sv(e,j) += u_dof_sv(e,i)*cbasis(e,i,j,0);
                    }
                  }
                });
              }
              
              if (btype == "HGRAD") {
                auto cbasis_grad = wset->basis_grad[bnum];
                auto u_dof_sv = subview(u_dof, ALL(), var, ALL());
                auto ugrad_sv = subview(ugrad_ip, ALL(), var, ALL(), ALL());
                parallel_for("grp response HGRAD",
                             RangePolicy<AssemblyExec>(0,u_ip.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  for (size_type i=0; i<cbasis_grad.extent(1); i++ ) {
                    for (size_type j=0; j<cbasis_grad.extent(2); j++ ) {
                      for (size_type s=0; s<cbasis_grad.extent(3); s++) {
                        ugrad_sv(e,j,s) += u_dof_sv(e,i)*cbasis_grad(e,i,j,s);
                      }
                    }
                  }
                });
              }
            }
            
            for (int s=0; s<dimension; s++) {
              auto ugrad_sv = subview(ugrad_ip, ALL(), ALL(), ALL(), s);
              if ((w-1) == s) {
                parallel_for("grp response seed grad 0",
                             RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  for (size_type n=0; n<numDOF.extent(0); n++) {
                    for(size_type j=0; j<ugrad_sv.extent(2); j++ ) {
                      ScalarT tmp = ugrad_sv(e,n,j).val();
                      ugrad_sv(e,n,j) = u_ip(e,n,j,0);
                      ugrad_sv(e,n,j) += -u_ip(e,n,j,0).val() + tmp;
                    }
                  }
                });
              }
              else {
                parallel_for("grp response seed grad 1",
                             RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  for (size_type n=0; n<numDOF.extent(0); n++) {
                    for(size_type j=0; j<ugrad_sv.extent(2); j++ ) {
                      ugrad_sv(e,n,j) = ugrad_sv(e,n,j).val();
                    }
                  }
                });
              }
              
            }
            parallel_for("grp response seed grad 2",
                         RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)),
                         KOKKOS_LAMBDA (const size_type e ) {
              for (size_type n=0; n<numDOF.extent(0); n++) {
                for(size_type j=0; j<u_ip.extent(2); j++ ) {
                  for(size_type s=0; s<u_ip.extent(3); s++ ) {
                    u_ip(e,n,j,s) = u_ip(e,n,j,s).val();
                  }
                }
              }
            });
            wset->setSolution(u_ip);
            wset->setSolutionGrad(ugrad_ip);
            
          }
          
          // Evaluate the objective
          auto obj_dev = fman->evaluate(objectives[obj].name+" response","ip");
          
          // Weight using volumetric integration weights
          auto wts = assembler->groups[block][grp]->wts;
          
          parallel_for("grp objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              //obj_dev(elem,pt) *= objectives[r].weight*wts(elem,pt);
              obj_dev(elem,pt) *= wts(elem,pt);
            }
          });
          
          if (w==0) {
            Kokkos::View<ScalarT[1],AssemblyDevice> ir("integral of response");
            parallel_for("grp objective",
                         RangePolicy<AssemblyExec>(0,wts.extent(0)),
                         KOKKOS_LAMBDA (const size_type elem ) {
              for (size_type pt=0; pt<wts.extent(1); pt++) {
                //obj_dev(elem,pt) *= objectives[r].weight*wts(elem,pt);
                ir(0) += obj_dev(elem,pt).val();
              }
            });
          
            auto ir_host = create_mirror_view(ir);
            deep_copy(ir_host,ir);
            intresp += ir_host(0);
          }

          for (size_type n=0; n<numDOF.extent(0); n++) {
            int bnum = wset->usebasis[n];
            std::string btype = wset->basis_types[bnum];
            
            if (w == 0) {
              auto cbasis = wset->basis[bnum];
              
              if (btype == "HDIV" || btype == "HCURL") {
                parallel_for("grp adjust adjoint res",
                             RangePolicy<AssemblyExec>(0,local_grad.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  int nn = n; // TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        for (size_type d=0; d<cbasis.extent(3); d++) {
                          local_grad(e,offsets(nn,j),0) += -obj_dev(e,s).fastAccessDx(offsets(nn,i))*cbasis(e,j,s,d);
                        }
                      }
                    }
                  }
                });
              }
              else {
                parallel_for("grp adjust adjoint res",
                             RangePolicy<AssemblyExec>(0,local_grad.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(e,offsets(nn,j),0) += -obj_dev(e,s).fastAccessDx(offsets(nn,i))*cbasis(e,j,s,0);
                      }
                    }
                  }
                });
              }
            }
            else {
              
              if (btype == "HGRAD") {
                auto cbasis = wset->basis_grad[bnum];
                //auto cbasis_sv = subview(cbasis, ALL(), ALL(), ALL(), w-1);
                parallel_for("grp adjust adjoint res",
                             RangePolicy<AssemblyExec>(0,local_grad.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(e,offsets(nn,j),0) += -obj_dev(e,s).fastAccessDx(offsets(nn,i))*cbasis(e,j,s,w-1);
                      }
                    }
                  }
                });
                
                
              }
              
            }
            
          }
        }
        
        if (data_avail) {
          assembler->scatterRes(grad_view, local_grad, assembler->groups[block][grp]->LIDs[set]);
        }
        else {
          Kokkos::deep_copy(local_grad_ladev,local_grad);
          
          if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
            assembler->scatterRes(grad_view, local_grad_ladev, assembler->groups[block][grp]->LIDs_host[set]);
          }
          else { // LA_device = CUDA, AssemblyDevice = Host
            // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
            auto LIDs_dev = Kokkos::create_mirror(LA_exec(), assembler->groups[block][grp]->LIDs[set]);
            Kokkos::deep_copy(LIDs_dev,assembler->groups[block][grp]->LIDs[set]);
            assembler->scatterRes(grad_view, local_grad_ladev, LIDs_dev);
          }
          
        }
      }
      
      // Right now grad_over = dresponse/du
      // We want   grad_over = 2.0*wt*(response - target)*dresponse/du
      grad_over->scale(2.0*objectives[obj].weight*(intresp - objectives[obj].target));
      
      linalg->exportVectorFromOverlapped(set, grad_tmp, grad_over);
      grad->update(1.0, *grad_tmp, 1.0);
      
    }
    else if (objectives[obj].type == "discrete control") {
      //for (size_t set=0; set<current_soln.size(); ++set) {
        vector_RCP D_soln;
        bool fnd = datagen_soln[set]->extract(D_soln, 0, current_time);
        if (fnd) {
          // TMW: this is unecessarily complicated because we store the overlapped soln
          vector_RCP diff = linalg->getNewVector(set);
          vector_RCP u_no = linalg->getNewVector(set);
          vector_RCP D_no = linalg->getNewVector(set);
          u_no->doExport(*(current_soln), *(linalg->exporter[set]), Tpetra::REPLACE);
          D_no->doExport(*D_soln, *(linalg->exporter[set]), Tpetra::REPLACE);
          diff->update(1.0, *u_no, 0.0);
          diff->update(-1.0, *D_no, 1.0);
          //grad->update(-2.0*objectives[r].weight,*diff,1.0);
          grad->update(-2.0*objectives[obj].weight,*diff,1.0);
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: did not find a data-generating solution");
        }
      //}
    }
    else if (objectives[obj].type == "sensors") {
      
      auto grad_over = linalg->getNewOverlappedVector(set);
      auto grad_tmp = linalg->getNewVector(set);
      auto grad_view = grad_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      
      for (size_t pt=0; pt<objectives[obj].numSensors; ++pt) {
        size_t tindex = 0;
        bool foundtime = false;
        for (size_type t=0; t<objectives[obj].sensor_times.extent(0); ++t) {
          if (std::abs(current_time - objectives[obj].sensor_times(t)) < 1.0e-12) {
            foundtime = true;
            tindex = t;
          }
        }
        
        if (foundtime) {
        
          size_t grp = objectives[obj].sensor_owners(pt,0);
          size_t elem = objectives[obj].sensor_owners(pt,1);
          
          wset->isOnSide = true;

          auto x = wset->getScalarField("x");
          x(0,0) = objectives[obj].sensor_points(pt,0);
          if (dimension > 1) {
            auto y = wset->getScalarField("y");
            y(0,0) = objectives[obj].sensor_points(pt,1);
          }
          if (dimension > 2) {
            auto z = wset->getScalarField("z");
            z(0,0) = objectives[obj].sensor_points(pt,2);
          }
          
          auto numDOF = assembler->groupData[block]->num_dof;
          auto offsets = wset->offsets;
          
          
          View_EvalT2 u_dof("u_dof",numDOF.extent(0),assembler->groups[block][grp]->LIDs[set].extent(1));
          auto cu = subview(assembler->groups[block][grp]->sol[set],elem,ALL(),ALL());
          parallel_for("grp response get u",
                       RangePolicy<AssemblyExec>(0,u_dof.extent(0)),
                       KOKKOS_LAMBDA (const size_type n ) {
            EvalT dummyval = 0.0;
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for( int i=0; i<numDOF(n); i++ ) {
                u_dof(n,i) = EvalT(dummyval.size(),offsets(n,i),cu(n,i));
              }
            }
          });
          
          // Map the local solution to the solution and gradient at ip
          View_EvalT2 u_ip("u_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
          View_EvalT2 ugrad_ip("ugrad_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
          
          for (size_type var=0; var<numDOF.extent(0); var++) {
            auto cbasis = objectives[obj].sensor_basis[wset->usebasis[var]];
            auto cbasis_grad = objectives[obj].sensor_basis_grad[wset->usebasis[var]];
            auto u_sv = subview(u_ip, var, ALL());
            auto u_dof_sv = subview(u_dof, var, ALL());
            auto ugrad_sv = subview(ugrad_ip, var, ALL());
            
            parallel_for("grp response sensor uip",
                         RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                         KOKKOS_LAMBDA (const int dof ) {
              u_sv(0) += u_dof_sv(dof)*cbasis(pt,dof,0,0);
              for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                ugrad_sv(dim) += u_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
              }
            });
          }
          
          // Map the local discretized params to param and grad at ip
          if (params->globalParamUnknowns > 0) {
            auto numParamDOF = assembler->groupData[block]->num_param_dof;
            
            View_EvalT2 p_dof("p_dof",numParamDOF.extent(0),assembler->groups[block][grp]->paramLIDs.extent(1));
            auto cp = subview(assembler->groups[block][grp]->param,elem,ALL(),ALL());
            parallel_for("grp response get u",
                         RangePolicy<AssemblyExec>(0,p_dof.extent(0)),
                         KOKKOS_LAMBDA (const size_type n ) {
              for (size_type n=0; n<numParamDOF.extent(0); n++) {
                for( int i=0; i<numParamDOF(n); i++ ) {
                  p_dof(n,i) = cp(n,i);
                }
              }
            });
            
            View_EvalT2 p_ip("p_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
            View_EvalT2 pgrad_ip("pgrad_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
            
            for (size_type var=0; var<numParamDOF.extent(0); var++) {
              int bnum = wset->paramusebasis[var];
              auto cbasis = objectives[obj].sensor_basis[bnum];
              auto cbasis_grad = objectives[obj].sensor_basis_grad[bnum];
              auto p_sv = subview(p_ip, var, ALL());
              auto p_dof_sv = subview(p_dof, var, ALL());
              auto pgrad_sv = subview(pgrad_ip, var, ALL());
              
              parallel_for("grp response sensor uip",
                           RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                           KOKKOS_LAMBDA (const int dof ) {
                p_sv(0) += p_dof_sv(dof)*cbasis(pt,dof,0,0);
                for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                  pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                }
              });
            }
            
            wset->setParamPoint(p_ip);
            wset->setParamGradPoint(pgrad_ip);
          }
          
          
          View_Sc3 local_grad("local contrib to dobj/dstate",
                              assembler->groups[block][grp]->numElem,
                              assembler->groups[block][grp]->LIDs[set].extent(1),1);
          
          for (int w=0; w<dimension+1; ++w) {
            if (w==0) {
              wset->setSolutionPoint(u_ip);
              wset->setSolutionGradPoint(ugrad_ip);
            }
            else {
              View_EvalT2 u_tmp("u_tmp",numDOF.extent(0),assembler->groupData[block]->dimension);
              View_EvalT2 ugrad_tmp("ugrad_tmp",numDOF.extent(0),assembler->groupData[block]->dimension);
              deep_copy(u_tmp,u_ip);
              deep_copy(ugrad_tmp,ugrad_ip);
              
              for (int s=0; s<dimension; s++) {
                auto ugrad_sv = subview(ugrad_tmp, ALL(), s);
                if ((w-1) == s) {
                  parallel_for("grp response seed grad 0",
                               RangePolicy<AssemblyExec>(0,ugrad_tmp.extent(0)),
                               KOKKOS_LAMBDA (const size_type var ) {
                    ScalarT tmp = ugrad_sv(var).val();
                    ugrad_sv(var) = u_tmp(var,0);
                    ugrad_sv(var) += -u_tmp(var,0).val() + tmp;
                  });
                }
                else {
                  parallel_for("grp response seed grad 1",
                               RangePolicy<AssemblyExec>(0,ugrad_tmp.extent(0)),
                               KOKKOS_LAMBDA (const size_type var ) {
                    ugrad_sv(var) = ugrad_sv(var).val();
                  });
                }
                
              }
              parallel_for("grp response seed grad 2",
                           RangePolicy<AssemblyExec>(0,u_tmp.extent(0)),
                           KOKKOS_LAMBDA (const size_type var ) {
                for(size_type s=0; s<u_tmp.extent(1); s++ ) {
                  u_tmp(var,s) = u_tmp(var,s).val();
                }
              });
              
              wset->setSolutionPoint(u_tmp);
              wset->setSolutionGradPoint(ugrad_tmp);
              
            }

            auto rdata = fman->evaluate(objectives[obj].name+" response","point");
            EvalT diff = rdata(0,0) - objectives[obj].sensor_data(pt,tindex);
            EvalT totaldiff = objectives[obj].weight*diff*diff;
            

            for (size_type n=0; n<numDOF.extent(0); n++) {
              int bnum = wset->usebasis[n];
              
              std::string btype = wset->basis_types[bnum];
              if (btype == "HDIV" || btype == "HCURL") {
                if (w==0) {
                  auto cbasis = objectives[obj].sensor_basis[bnum];
                  int nn = n; // TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        for (size_type d=0; d<cbasis.extent(3); d++) {
                          local_grad(elem,offsets(nn,j),0) += -totaldiff.fastAccessDx(offsets(nn,i))*cbasis(pt,j,s,d);
                        }
                      }
                    }
                  }
                }
              }
              else {
                if (w==0) {
                  auto cbasis = objectives[obj].sensor_basis[bnum];
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(elem,offsets(nn,j),0) += -totaldiff.fastAccessDx(offsets(nn,i))*cbasis(pt,j,s,0);
                      }
                    }
                  }
                }
                else {
                  auto cbasis = objectives[obj].sensor_basis_grad[bnum];
                  auto cbasis_sv = subview(cbasis,ALL(),ALL(),ALL(),w-1);
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(elem,offsets(nn,j),0) += -totaldiff.fastAccessDx(offsets(nn,i))*cbasis_sv(pt,j,s);
                      }
                    }
                  }
                }
              }
            }
          }
          
          assembler->scatterRes(grad_view, local_grad, assembler->groups[block][grp]->LIDs[set]);
          
          wset->isOnSide = false;
        }
      }
      
      linalg->exportVectorFromOverlapped(set, grad_tmp, grad_over);
      grad->update(1.0, *grad_tmp, 1.0);
      
    }
  
  
#endif
  
}


// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeSensitivities(vector<vector_RCP> & u,
                                                    vector<vector_RCP> & adjoint,
                                                    const ScalarT & current_time,
                                                    const int & tindex,
                                                    const ScalarT & deltat,
                                                    MrHyDE_OptVector & gradient) {
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting PostprocessManager::computeSensitivities ..." << std::endl;
    }
  }
  
  typedef typename Node::device_type LA_device;
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
    
  if (save_adjoint_solution) {
    for (size_t set=0; set<soln.size(); ++set) {
      adj_soln[set]->store(adjoint[set], current_time, 0);
    }
  }
  DFAD obj_sens = 0.0;
  if (response_type != "discrete") {
    this->computeObjectiveGradParam(u, current_time, obj_sens);
  }
  
  size_t set = 0; // hard coded for now
  
  auto u_kv = u[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto adjoint_kv = adjoint[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  if (params->num_active_params > 0) {
  
    params->sacadoizeParams(true);
    
    //vector<ScalarT> localsens(params->num_active_params);
    auto scalar_grad = gradient.getParameter()->getVector();

    vector<ScalarT> local_grad(scalar_grad->size(),0.0);
    vector_RCP res = linalg->getNewVector(set,params->num_active_params);
    matrix_RCP J = linalg->getNewMatrix(set);
    vector_RCP res_over = linalg->getNewOverlappedVector(set,params->num_active_params);
    matrix_RCP J_over = linalg->getNewOverlappedMatrix(set);
    
    auto res_kv = res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    
    res_over->putScalar(0.0);
    
    assembler->assembleJacRes(set, u[set], u[set], false, true, false,
                              res_over, J_over, isTD, current_time, false, false, //store_adjPrev,
                              params->num_active_params, params->Psol_over, false, deltat); //is_final_time, deltat);
    
    linalg->exportVectorFromOverlapped(set, res, res_over);
    
    for (size_t paramiter=0; paramiter<params->num_active_params; paramiter++) {
      // fine-scale
      if (assembler->groups[0][0]->group_data->multiscale) {
        ScalarT subsens = 0.0;
        for (size_t block=0; block<assembler->groups.size(); ++block) {
          for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
            subsens = -assembler->groups[block][grp]->subgradient(0,paramiter);
            local_grad[paramiter] += subsens;
          }
        }
      }
      else { // coarse-scale
      
        ScalarT currsens = 0.0;
        for( size_t i=0; i<res_kv.extent(0); i++ ) {
          currsens += adjoint_kv(i,0) * res_kv(i,paramiter);
        }
        local_grad[paramiter] = -currsens;
      }
      
    }
    
    
    ScalarT localval = 0.0;
    ScalarT globalval = 0.0;
    int numderivs = (int)obj_sens.size();
    for (size_t paramiter=0; paramiter < params->num_active_params; paramiter++) {
      localval = local_grad[paramiter];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      ScalarT cobj = 0.0;
      
      if ((int)paramiter<numderivs) {
        cobj = obj_sens.fastAccessDx(paramiter);
      }
      globalval += cobj;
      (*scalar_grad)[paramiter] += globalval;
    }
    params->sacadoizeParams(false);
  }
  
  int numDiscParams = params->getNumParams(4);
  
  if (numDiscParams > 0) {
    
    auto disc_grad = gradient.getField();
    vector_RCP curr_grad;
    if (gradient.isDynamic()) {
      curr_grad = disc_grad[tindex-1]->getVector();
    }
    else {
      curr_grad = disc_grad[0]->getVector();
    }
    
    
    auto sens = this->computeDiscreteSensitivities(u, adjoint, current_time, deltat);
    
    auto sens_kv = sens->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);

    for (size_t i = 0; i < params->paramOwned.size(); i++) {
      ScalarT cobj = 0.0;
      if ((int)(i+params->num_active_params)<obj_sens.size()) {
        cobj = obj_sens.fastAccessDx(i+params->num_active_params);
      }
      sens_kv(i,0) += cobj;
    }
    
    curr_grad->update(1.0, *sens, 1.0);
    
  }
  saveObjectiveGradientData(gradient);
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Finished PostprocessManager::computeSensitivities ..." << std::endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
ScalarT PostprocessManager<Node>::computeDualWeightedResidual(vector<vector_RCP> & u,
                                                              vector<vector_RCP> & adjoint,
                                                              const ScalarT & current_time,
                                                              const int & tindex,
                                                              const ScalarT & deltat) {
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting PostprocessManager::computeDualWeightedResidual ..." << std::endl;
    }
  }
  
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
  
  size_t set = 0; // hard coded for now
  
  // adjoint solution is overlapped
  vector_RCP adj = linalg->getNewVector(set);
  linalg->exportVectorFromOverlapped(set, adj, adjoint[set]);

  //auto adjoint_kv = adj->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);
  
    vector_RCP res = linalg->getNewVector(set,params->num_active_params);
  //  matrix_RCP J = linalg->getNewMatrix(set);
    vector_RCP res_over = linalg->getNewOverlappedVector(set,params->num_active_params);
    matrix_RCP J_over;// = linalg->getNewOverlappedMatrix(set);
    
    res_over->putScalar(0.0);
    
    assembler->assembleJacRes(set, u[set], u[set], false, false, false,
                              res_over, J_over, isTD, current_time, false, false, //store_adjPrev,
                              params->num_active_params, params->Psol_over, false, deltat); //is_final_time, deltat);
    
    linalg->exportVectorFromOverlapped(set, res, res_over);
    
  res->dot(*adj, dotprod);

  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Finished PostprocessManager::computeDualWeightedResidual ..." << std::endl;
    }
  }
  return dotprod[0];

}



// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > 
PostprocessManager<Node>::computeDiscreteSensitivities(vector<vector_RCP> & u,
                                                       vector<vector_RCP> & adjoint,
                                                       const ScalarT & current_time,
                                                       const ScalarT & deltat) {

  int set = 0; // hard-coded for now
  
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;

  vector_RCP res_over = linalg->getNewOverlappedVector(set);
  matrix_RCP J = linalg->getNewParamMatrix();
  matrix_RCP J_over = linalg->getNewParamOverlappedMatrix();
  res_over->putScalar(0.0);
  J->setAllToScalar(0.0);
  J_over->setAllToScalar(0.0);
  
  assembler->assembleJacRes(set, u[set], u[set], true, false, true,
                            res_over, J_over, isTD, current_time, false, false, //store_adjPrev,
                            params->num_active_params, params->Psol_over, false, deltat); //is_final_time, deltat);
    
  linalg->fillCompleteParam(set, J_over);
  
  vector_RCP gradient = linalg->getNewParamVector();
    
  linalg->exportParamMatrixFromOverlapped(J, J_over);

  linalg->fillCompleteParam(set,J);
  
  vector_RCP adj = linalg->getNewVector(set);
  adj->doExport(*(adjoint[set]), *(linalg->exporter[set]), Tpetra::REPLACE);
  J->apply(*adj,*gradient);
  

  return gradient;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::writeSolution(const ScalarT & currenttime) {
  
  Teuchos::TimeMonitor localtimer(*writeSolutionTimer);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting PostprocessManager::writeSolution() ..." << std::endl;
    }
  }

  plot_times.push_back(currenttime);
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    std::string blockID = blocknames[block];
    vector<size_t> myElements = disc->my_elements[block];
    
    if (myElements.size() > 0) {
        
      for (size_t set=0; set<setnames.size(); ++set) {
    
        assembler->updatePhysicsSet(set);
      
        vector<string> vartypes = physics->types[set][block];
        vector<int> varorders = physics->orders[set][block];
        int numVars = physics->num_vars[set][block]; // probably redundant
        
        for (int n = 0; n<numVars; n++) {
          
          if (vartypes[n] == "HGRAD") {
            if(assembler->groups[block][0]->group_data->require_basis_at_nodes) {
              std::string var = varlist[set][block][n];
              Kokkos::View<ScalarT**,AssemblyDevice> soln_dev = Kokkos::View<ScalarT**,AssemblyDevice>("solution",myElements.size(),
                                                                                                       numNodesPerElem);
              auto soln_computed = Kokkos::create_mirror_view(soln_dev);
              for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
                auto eID = assembler->groups[block][grp]->localElemID;
                auto sol = Kokkos::subview(assembler->getSolutionAtNodes(block, grp, n), Kokkos::ALL(), Kokkos::ALL(), 0); // last component is dimension, which is 0 for HGRAD
                parallel_for("postproc plot param HGRAD",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
                  for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                    soln_dev(eID(elem),i) = sol(elem,i);
                  }
                });
              }
              Kokkos::deep_copy(soln_computed, soln_dev);
              mesh->stk_mesh->setSolutionFieldData(var+append, blockID, myElements, soln_computed);
            } else {
              Kokkos::View<ScalarT**,AssemblyDevice> soln_dev = Kokkos::View<ScalarT**,AssemblyDevice>("solution",myElements.size(), numNodesPerElem);
              auto soln_computed = Kokkos::create_mirror_view(soln_dev);
              std::string var = varlist[set][block][n];
              for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
                auto eID = assembler->groups[block][grp]->localElemID;
                auto sol = Kokkos::subview(assembler->groups[block][grp]->sol[set], Kokkos::ALL(), n, Kokkos::ALL());
                parallel_for("postproc plot HGRAD",
                            RangePolicy<AssemblyExec>(0,eID.extent(0)),
                            KOKKOS_LAMBDA (const int elem ) {
                  for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                    soln_dev(eID(elem),i) = sol(elem,i);
                  }
                });
              }
              Kokkos::deep_copy(soln_computed, soln_dev);
              
              /*
              if (var == "dx") {
              mesh->stk_mesh->setSolutionFieldData("disp"+append+"x", blockID, myElements, soln_computed);
              }
              if (var == "dy") {
              mesh->stk_mesh->setSolutionFieldData("disp"+append+"y", blockID, myElements, soln_computed);
              }
              if (var == "dz" || var == "H") {
              mesh->stk_mesh->setSolutionFieldData("disp"+append+"z", blockID, myElements, soln_computed);
              }
              */
              
              mesh->stk_mesh->setSolutionFieldData(var+append, blockID, myElements, soln_computed);
            }
          }
          else if (vartypes[n] == "HVOL") {
            Kokkos::View<ScalarT*,AssemblyDevice> soln_dev("solution",myElements.size());
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            std::string var = varlist[set][block][n];
            for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              auto sol = Kokkos::subview(assembler->groups[block][grp]->sol[set], Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot HVOL",
                           RangePolicy<AssemblyExec>(0,eID.extent(0)),
                           KOKKOS_LAMBDA (const int elem ) {
                soln_dev(eID(elem)) = sol(elem,0);//u_kv(pindex,0);
              });
            }
            Kokkos::deep_copy(soln_computed,soln_dev);
            mesh->stk_mesh->setCellFieldData(var+append, blockID, myElements, soln_computed);
          }
          else if (vartypes[n] == "HDIV" || vartypes[n] == "HCURL") { // need to project each component onto PW-linear basis and PW constant basis
            Kokkos::View<ScalarT*,AssemblyDevice> soln_x_dev("solution",myElements.size());
            Kokkos::View<ScalarT*,AssemblyDevice> soln_y_dev("solution",myElements.size());
            Kokkos::View<ScalarT*,AssemblyDevice> soln_z_dev("solution",myElements.size());
            auto soln_x = Kokkos::create_mirror_view(soln_x_dev);
            auto soln_y = Kokkos::create_mirror_view(soln_y_dev);
            auto soln_z = Kokkos::create_mirror_view(soln_z_dev);
            std::string var = varlist[set][block][n];
            View_Sc2 sol("average solution",assembler->groupData[block]->num_elem,dimension);
            
            for (size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              
              assembler->computeSolutionAverage(block, grp, var,sol);
              parallel_for("postproc plot HDIV/HCURL",
                           RangePolicy<AssemblyExec>(0,eID.extent(0)),
                           KOKKOS_LAMBDA (const int elem ) {
                soln_x_dev(eID(elem)) = sol(elem,0);
                if (sol.extent(1) > 1) {
                  soln_y_dev(eID(elem)) = sol(elem,1);
                }
                if (sol.extent(1) > 2) {
                  soln_z_dev(eID(elem)) = sol(elem,2);
                }
              });
            }
            Kokkos::deep_copy(soln_x, soln_x_dev);
            Kokkos::deep_copy(soln_y, soln_y_dev);
            Kokkos::deep_copy(soln_z, soln_z_dev);
            mesh->stk_mesh->setCellFieldData(var+append+"x", blockID, myElements, soln_x);
            if (dimension > 1) {
              mesh->stk_mesh->setCellFieldData(var+append+"y", blockID, myElements, soln_y);
            }
            if (dimension > 2) {
              mesh->stk_mesh->setCellFieldData(var+append+"z", blockID, myElements, soln_z);
            }
            
          }
          else if (vartypes[n] == "HFACE" && write_HFACE_variables) {
            
            Kokkos::View<ScalarT*,AssemblyDevice> soln_faceavg_dev("solution",myElements.size());
            auto soln_faceavg = Kokkos::create_mirror_view(soln_faceavg_dev);
            
            Kokkos::View<ScalarT*,AssemblyDevice> face_measure_dev("face measure",myElements.size());
            assembler->wkset[block]->isOnSide = true;
            for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              for (size_t face=0; face<assembler->groupData[block]->num_sides; face++) {
                int seedwhat = 0;
                for (size_t iset=0; iset<assembler->wkset[block]->numSets; ++iset) {
                  assembler->wkset[block]->computeSolnSteadySeeded(iset, assembler->groups[block][grp]->sol[iset], seedwhat);
                }
                //assembler->groups[block][grp]->computeSolnFaceIP(face);
                assembler->updateWorksetFace(block, grp, face);
                auto wts = assembler->wkset[block]->wts_side;
                auto sol = assembler->wkset[block]->getSolutionField(varlist[set][block][n]);
                parallel_for("postproc plot HFACE",
                             RangePolicy<AssemblyExec>(0,eID.extent(0)),
                             KOKKOS_LAMBDA (const int elem ) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    face_measure_dev(eID(elem)) += wts(elem,pt);
//#ifndef MrHyDE_NO_AD
//                    soln_faceavg_dev(eID(elem)) += sol(elem,pt).val()*wts(elem,pt);
//#else
                    soln_faceavg_dev(eID(elem)) += sol(elem,pt)*wts(elem,pt);
//#endif
                  }
                });
              }
            }
            
            assembler->wkset[block]->isOnSide = false;
            parallel_for("postproc plot HFACE 2",
                         RangePolicy<AssemblyExec>(0,soln_faceavg_dev.extent(0)),
                         KOKKOS_LAMBDA (const int elem ) {
              soln_faceavg_dev(elem) *= 1.0/face_measure_dev(elem);
            });
            Kokkos::deep_copy(soln_faceavg, soln_faceavg_dev);
            mesh->stk_mesh->setCellFieldData(varlist[set][block][n]+append, blockID, myElements, soln_faceavg);
          }
        }
      }
      
      ////////////////////////////////////////////////////////////////
      // Discretized Parameters
      ////////////////////////////////////////////////////////////////
      
      vector<string> dpnames = params->discretized_param_names;
      vector<int> numParamBasis = params->paramNumBasis;
      vector<int> dp_usebasis = params->discretized_param_usebasis;
      vector<string> discParamTypes = params->discretized_param_basis_types;
      if (dpnames.size() > 0) {
        for (size_t n=0; n<dpnames.size(); n++) {
          int bnum = dp_usebasis[n];
          if (discParamTypes[bnum] == "HGRAD") {
            Kokkos::View<ScalarT**,AssemblyDevice> soln_dev = Kokkos::View<ScalarT**,AssemblyDevice>("solution",myElements.size(),
                                                                                                     numNodesPerElem);
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              auto sol = Kokkos::subview(assembler->groups[block][grp]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HGRAD",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
                for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                  soln_dev(eID(elem),i) = sol(elem,i);
                }
              });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->stk_mesh->setSolutionFieldData(dpnames[n]+append, blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HVOL") {
            Kokkos::View<ScalarT*,AssemblyDevice> soln_dev("solution",myElements.size());
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            //std::string var = varlist[block][n];
            for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              auto sol = Kokkos::subview(assembler->groups[block][grp]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HVOL",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
                soln_dev(eID(elem)) = sol(elem,0);
              });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->stk_mesh->setCellFieldData(dpnames[n]+append, blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HDIV" || discParamTypes[n] == "HCURL") {
            Kokkos::View<ScalarT*,AssemblyDevice> soln_x_dev("solution",myElements.size());
            Kokkos::View<ScalarT*,AssemblyDevice> soln_y_dev("solution",myElements.size());
            Kokkos::View<ScalarT*,AssemblyDevice> soln_z_dev("solution",myElements.size());
            auto soln_x = Kokkos::create_mirror_view(soln_x_dev);
            auto soln_y = Kokkos::create_mirror_view(soln_y_dev);
            auto soln_z = Kokkos::create_mirror_view(soln_z_dev);
            View_Sc2 sol("average solution",assembler->groupData[block]->num_elem,dimension);
            
            for (size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              
              assembler->computeParameterAverage(block, grp, dpnames[n],sol);
              //auto sol = Kokkos::subview(assembler->groups[block][grp]->u_avg, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot HDIV/HCURL",
                           RangePolicy<AssemblyExec>(0,eID.extent(0)),
                           KOKKOS_LAMBDA (const int elem ) {
                soln_x_dev(eID(elem)) = sol(elem,0);
                if (sol.extent(1) > 1) {
                  soln_y_dev(eID(elem)) = sol(elem,1);
                }
                if (sol.extent(1) > 2) {
                  soln_z_dev(eID(elem)) = sol(elem,2);
                }
              });
            }
            Kokkos::deep_copy(soln_x, soln_x_dev);
            Kokkos::deep_copy(soln_y, soln_y_dev);
            Kokkos::deep_copy(soln_z, soln_z_dev);
            mesh->stk_mesh->setCellFieldData(dpnames[n]+append+"x", blockID, myElements, soln_x);
            if (dimension > 1) {
              mesh->stk_mesh->setCellFieldData(dpnames[n]+append+"y", blockID, myElements, soln_y);
            }
            if (dimension > 2) {
              mesh->stk_mesh->setCellFieldData(dpnames[n]+append+"z", blockID, myElements, soln_z);
            }
            // TMW: this is not actually implemented yet ... not hard to do though
            /*
             Kokkos::View<ScalarT*,HostDevice> soln_x("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_y("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_z("solution",myElements.size());
             std::string var = varlist[block][n];
             size_t eprog = 0;
             for( size_t e=0; e<assembler->groups[block].size(); e++ ) {
             Kokkos::View<ScalarT**,AssemblyDevice> sol = assembler->groups[block][grp]->param_avg;
             auto host_sol = Kokkos::create_mirror_view(sol);
             Kokkos::deep_copy(host_sol,sol);
             for (int p=0; p<assembler->groups[block][grp]->numElem; p++) {
             soln_x(eprog) = host_sol(p,n,0);
             soln_y(eprog) = host_sol(p,n,1);
             soln_z(eprog) = host_sol(p,n,2);
             eprog++;
             }
             }
             
             mesh->stk_mesh->setcellFieldData(var+"x", blockID, myElements, soln_x);
             mesh->stk_mesh->setcellFieldData(var+"y", blockID, myElements, soln_y);
             mesh->stk_mesh->setcellFieldData(var+"z", blockID, myElements, soln_z);
             */
          }
        }
        
      }
      
      ////////////////////////////////////////////////////////////////
      // Extra nodal fields
      ////////////////////////////////////////////////////////////////
      // TMW: This needs to be rewritten to actually use integration points
      
      vector<string> extrafieldnames = extrafields_list[block];
      for (size_t j=0; j<extrafieldnames.size(); j++) {
        
        Kokkos::View<ScalarT**,HostDevice> efd("field data",myElements.size(), numNodesPerElem);
        /*
        for (size_t k=0; k<assembler->groups[block].size(); k++) {
          auto nodes = assembler->groups[block][k]->nodes;
          auto eID = assembler->groups[block][k]->localElemID;
          auto host_eID = Kokkos::create_mirror_view(eID);
          Kokkos::deep_copy(host_eID,eID);
          
          auto cfields = phys->getExtraFields(b, 0, nodes, currenttime, assembler->wkset_AD[block]);
          auto host_cfields = Kokkos::create_mirror_view(cfields);
          Kokkos::deep_copy(host_cfields,cfields);
          for (size_type p=0; p<host_eID.extent(0); p++) {
            for (size_t i=0; i<host_cfields.extent(1); i++) {
              efd(host_eID(p),i) = host_cfields(p,i);
            }
          }
        }
         */
        mesh->stk_mesh->setSolutionFieldData(extrafieldnames[j], blockID, myElements, efd);
      }
      
      ////////////////////////////////////////////////////////////////
      // Extra grp fields
      ////////////////////////////////////////////////////////////////
      
      if (extracellfields_list[block].size() > 0) {
        Kokkos::View<ScalarT**,AssemblyDevice> ecd_dev("grp data",myElements.size(),
                                                       extracellfields_list[block].size());
        auto ecd = Kokkos::create_mirror_view(ecd_dev);
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto eID = assembler->groups[block][grp]->localElemID;
          
          assembler->updateWorkset(block, grp, 0,0,true);
          assembler->wkset[block]->setTime(currenttime);
          
          auto cfields = this->getExtraCellFields(block, assembler->groups[block][grp]->wts);
          
          parallel_for("postproc plot param HVOL",
                       RangePolicy<AssemblyExec>(0,eID.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type r=0; r<cfields.extent(1); ++r) {
              ecd_dev(eID(elem),r) = cfields(elem,r);
            }
          });
        }
        Kokkos::deep_copy(ecd, ecd_dev);
        
        for (size_t j=0; j<extracellfields_list[block].size(); j++) {
          auto ccd = subview(ecd,ALL(),j);
          mesh->stk_mesh->setCellFieldData(extracellfields_list[block][j]+append, blockID, myElements, ccd);
        }
      }
      
      ////////////////////////////////////////////////////////////////
      // Derived quantities from physics modules
      ////////////////////////////////////////////////////////////////
      
      if (derivedquantities_list[block].size() > 0) {
        Kokkos::View<ScalarT**,AssemblyDevice> dq_dev("grp data",myElements.size(),
                                                      derivedquantities_list[block].size());
        auto dq = Kokkos::create_mirror_view(dq_dev);
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto eID = assembler->groups[block][grp]->localElemID;
          
          assembler->updateWorkset(block, grp, 0,0,true);
          assembler->wkset[block]->setTime(currenttime);
          
          auto cfields = this->getDerivedQuantities(block, assembler->groups[block][grp]->wts);
          
          parallel_for("postproc plot param HVOL",
                       RangePolicy<AssemblyExec>(0,eID.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type r=0; r<cfields.extent(1); ++r) {
              dq_dev(eID(elem),r) = cfields(elem,r);
            }
          });
        }
        Kokkos::deep_copy(dq, dq_dev);
        
        for (size_t j=0; j<derivedquantities_list[block].size(); j++) {
          auto cdq = subview(dq,ALL(),j);
          mesh->stk_mesh->setCellFieldData(derivedquantities_list[block][j]+append, blockID, myElements, cdq);
        }
      }
      
      ////////////////////////////////////////////////////////////////
      // Mesh data
      ////////////////////////////////////////////////////////////////
      // TMW This is slightly inefficient, but leaving until grp_data_seed is stored differently
      
      if (assembler->groups[block][0]->group_data->have_phi ||
          assembler->groups[block][0]->group_data->have_rotation ||
          assembler->groups[block][0]->group_data->have_extra_data) {
        
        Kokkos::View<ScalarT*,HostDevice> cdata("data",myElements.size());
        Kokkos::View<ScalarT*,HostDevice> cseed("data seed",myElements.size());
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          vector<size_t> data_seed = assembler->groups[block][grp]->data_seed;
          vector<size_t> data_seedindex = assembler->groups[block][grp]->data_seedindex;
          Kokkos::View<ScalarT**,AssemblyDevice> data = assembler->groups[block][grp]->data;
          Kokkos::View<LO*,AssemblyDevice> eID = assembler->groups[block][grp]->localElemID;
          auto host_eID = Kokkos::create_mirror_view(eID);
          Kokkos::deep_copy(host_eID,eID);
          
          for (size_type p=0; p<host_eID.extent(0); p++) {
            if (data.extent(1) == 1) {
              cdata(host_eID(p)) = data(p,0);
            }
            cseed(host_eID(p)) = data_seedindex[p];
          }
        }
        mesh->stk_mesh->setCellFieldData("mesh_data_seed", blockID, myElements, cseed);
        mesh->stk_mesh->setCellFieldData("mesh_data", blockID, myElements, cdata);
      }
      
      ////////////////////////////////////////////////////////////////
      // grp number
      ////////////////////////////////////////////////////////////////
      
      if (write_group_number) {
        Kokkos::View<ScalarT*,AssemblyDevice> grpnum_dev("grp number",myElements.size());
        auto grpnum = Kokkos::create_mirror_view(grpnum_dev);
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto eID = assembler->groups[block][grp]->localElemID;
          parallel_for("postproc plot param HVOL",
                       RangePolicy<AssemblyExec>(0,eID.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            grpnum_dev(eID(elem)) = grp; // TMW: is this what we want?
          });
        }
        Kokkos::deep_copy(grpnum, grpnum_dev);
        mesh->stk_mesh->setCellFieldData("group number", blockID, myElements, grpnum);
      }
      
      if (write_database_id) {
        Kokkos::View<ScalarT*,AssemblyDevice> jacnum_dev("unique jac ID",myElements.size());
        auto jacnum = Kokkos::create_mirror_view(jacnum_dev);
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto index = assembler->groups[block][grp]->basis_index;
          auto eID = assembler->groups[block][grp]->localElemID;
          parallel_for("postproc plot param HVOL",
                       RangePolicy<AssemblyExec>(0,eID.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            jacnum_dev(eID(elem)) = index(elem); // TMW: is this what we want?
          });
        }
        Kokkos::deep_copy(jacnum, jacnum_dev);
        mesh->stk_mesh->setCellFieldData("unique Jacobian ID", blockID, myElements, jacnum);
      }

      if (write_subgrid_model) {
        Kokkos::View<ScalarT*,AssemblyDevice> sgmodel_dev("subgrid model",myElements.size());
        auto sgmodel = Kokkos::create_mirror_view(sgmodel_dev);
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          //auto index = assembler->groups[block][grp]->basis_index;
          int sgindex = assembler->groups[block][grp]->subgrid_model_index;
          auto eID = assembler->groups[block][grp]->localElemID;
          parallel_for("postproc plot param HVOL",
                       RangePolicy<AssemblyExec>(0,eID.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            sgmodel_dev(eID(elem)) = sgindex; 
          });
        }
        Kokkos::deep_copy(sgmodel,sgmodel_dev);
        mesh->stk_mesh->setCellFieldData("subgrid model", blockID, myElements, sgmodel);
      }

    }
  }
  
  ////////////////////////////////////////////////////////////////
  // Write to Exodus
  ////////////////////////////////////////////////////////////////
  
  if (isTD) {
    mesh->stk_mesh->writeToExodus(currenttime);
  }
  else {
    mesh->stk_mesh->writeToExodus(exodus_filename);
  }
  
  if (write_subgrid_solution && multiscale_manager->getNumberSubgridModels() > 0) {
    multiscale_manager->writeSolution(currenttime, append);
  }

  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Finished PostprocessManager::writeSolution() ..." << std::endl;
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
View_Sc2 PostprocessManager<Node>::getExtraCellFields(const int & block, View_Sc2 wts) {
  
  int numElem = wts.extent(0);
  View_Sc2 fields("grp field data",numElem, extracellfields_list[block].size());
  
  for (size_t fnum=0; fnum<extracellfields_list[block].size(); ++fnum) {
    
    auto cfield = subview(fields, ALL(), fnum);
    View_Sc2 ecf = assembler->evaluateFunction(block, extracellfields_list[block][fnum], "ip");
    
    if (cellfield_reduction == "mean") { // default
      parallel_for("physics get extra grp fields",
                   RangePolicy<AssemblyExec>(0,wts.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        ScalarT grpmeas = 0.0;
        for (size_t pt=0; pt<wts.extent(1); pt++) {
          grpmeas += wts(e,pt);
        }
        for (size_t j=0; j<wts.extent(1); j++) {
          ScalarT val = ecf(e,j);
          cfield(e) += val*wts(e,j)/grpmeas;
        }
      });
    }
    else if (cellfield_reduction == "max") {
      parallel_for("physics get extra grp fields",
                   RangePolicy<AssemblyExec>(0,wts.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_t j=0; j<wts.extent(1); j++) {
          ScalarT val = ecf(e,j);
          if (val>cfield(e)) {
            cfield(e) = val;
          }
        }
      });
    }
    if (cellfield_reduction == "min") {
      parallel_for("physics get extra grp fields",
                   RangePolicy<AssemblyExec>(0,wts.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_t j=0; j<wts.extent(1); j++) {
          ScalarT val = ecf(e,j);
          if (val<cfield(e)) {
            cfield(e) = val;
          }
        }
      });
    }
  }
  
  return fields;
}

// ========================================================================================
// ========================================================================================

template<class Node>
View_Sc2 PostprocessManager<Node>::getDerivedQuantities(const int & block, View_Sc2 wts) {
  
  int numElem = wts.extent(0);
  View_Sc2 fields("grp field data",numElem, derivedquantities_list[block].size());
  
  int prog = 0;
  
  for (size_t set=0; set<physics->modules.size(); ++set) {
    for (size_t m=0; m<physics->modules[set][block].size(); ++m) {
      
      //vector<View_AD2> dqvals = physics->modules[set][block][m]->getDerivedValues();
      auto dqvals = physics->modules[set][block][m]->getDerivedValues();
      for (size_t k=0; k<dqvals.size(); k++) {
        auto cfield = subview(fields, ALL(), prog);
        auto cdq = dqvals[k];
        
        if (cellfield_reduction == "mean") { // default
          parallel_for("physics get extra grp fields",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            ScalarT grpmeas = 0.0;
            for (size_t pt=0; pt<wts.extent(1); pt++) {
              grpmeas += wts(e,pt);
            }
            for (size_t j=0; j<wts.extent(1); j++) {
              ScalarT val = cdq(e,j);
              cfield(e) += val*wts(e,j)/grpmeas;
            }
          });
        }
        else if (cellfield_reduction == "max") {
          parallel_for("physics get extra grp fields",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_t j=0; j<wts.extent(1); j++) {
              ScalarT val = cdq(e,j);
              if (val>cfield(e)) {
                cfield(e) = val;
              }
            }
          });
        }
        else if (cellfield_reduction == "min") {
          parallel_for("physics get extra grp fields",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_t j=0; j<wts.extent(1); j++) {
              ScalarT val = cdq(e,j);
              if (val<cfield(e)) {
                cfield(e) = val;
              }
            }
          });
        }
      
      prog++;
      }
    }
  }
  return fields;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::writeOptimizationSolution(const int & numEvaluations) {
  
  Teuchos::TimeMonitor localtimer(*writeSolutionTimer);
  
  for (size_t block=0; block<assembler->groups.size(); ++block) {
    std::string blockID = blocknames[block];
    //vector<vector<int> > curroffsets = disc->offsets[block];
    vector<size_t> myElements = disc->my_elements[block];
    //vector<string> vartypes = physics->types[block];
    //vector<int> varorders = physics->orders[block];
    
    if (myElements.size() > 0) {
      
      ////////////////////////////////////////////////////////////////
      // Discretized Parameters
      ////////////////////////////////////////////////////////////////
      
      vector<string> dpnames = params->discretized_param_names;
      vector<int> numParamBasis = params->paramNumBasis;
      vector<int> dp_usebasis = params->discretized_param_usebasis;
      vector<string> discParamTypes = params->discretized_param_basis_types;
      if (dpnames.size() > 0) {
        for (size_t n=0; n<dpnames.size(); n++) {
          int bnum = dp_usebasis[n];
          if (discParamTypes[bnum] == "HGRAD") {
            Kokkos::View<ScalarT**,AssemblyDevice> soln_dev = Kokkos::View<ScalarT**,AssemblyDevice>("solution",myElements.size(),numNodesPerElem);
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              auto sol = Kokkos::subview(assembler->groups[block][grp]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HGRAD",
                           RangePolicy<AssemblyExec>(0,eID.extent(0)),
                           KOKKOS_LAMBDA (const int elem ) {
                for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                  soln_dev(eID(elem),i) = sol(elem,i);
                }
              });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->stk_optimization_mesh->setSolutionFieldData(dpnames[n], blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HVOL") {
            Kokkos::View<ScalarT*,AssemblyDevice> soln_dev("solution",myElements.size());
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            //std::string var = varlist[block][n];
            for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              auto sol = Kokkos::subview(assembler->groups[block][grp]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HVOL",
                           RangePolicy<AssemblyExec>(0,eID.extent(0)),
                           KOKKOS_LAMBDA (const int elem ) {
                soln_dev(eID(elem)) = sol(elem,0);
              });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->stk_optimization_mesh->setCellFieldData(dpnames[n], blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HDIV" || discParamTypes[n] == "HCURL") {
            // TMW: this is not actually implemented yet ... not hard to do though
            /*
             Kokkos::View<ScalarT*,HostDevice> soln_x("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_y("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_z("solution",myElements.size());
             std::string var = varlist[block][n];
             size_t eprog = 0;
             for( size_t e=0; e<assembler->groups[block].size(); e++ ) {
             Kokkos::View<ScalarT**,AssemblyDevice> sol = assembler->groups[block][grp]->param_avg;
             auto host_sol = Kokkos::create_mirror_view(sol);
             Kokkos::deep_copy(host_sol,sol);
             for (int p=0; p<assembler->groups[block][grp]->numElem; p++) {
             soln_x(eprog) = host_sol(p,n,0);
             soln_y(eprog) = host_sol(p,n,1);
             soln_z(eprog) = host_sol(p,n,2);
             eprog++;
             }
             }
             
             mesh->setcellFieldData(var+"x", blockID, myElements, soln_x);
             mesh->setcellFieldData(var+"y", blockID, myElements, soln_y);
             mesh->setcellFieldData(var+"z", blockID, myElements, soln_z);
             */
          }
        }
        
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////
  // Write to Exodus
  ////////////////////////////////////////////////////////////////
  
  double timestamp = static_cast<double>(numEvaluations);
  mesh->stk_optimization_mesh->writeToExodus(timestamp);

}


// ========================================================================================
// ========================================================================================

template<class Node>
ScalarT PostprocessManager<Node>::makeSomeNoise(ScalarT stdev) {
  //generate sample from 0-centered normal with stdev
  //Box-Muller method
  //srand(time(0)); //doing this more frequently than once-per-second results in getting the same numbers...
  ScalarT U1 = rand()/ScalarT(RAND_MAX);
  ScalarT U2 = rand()/ScalarT(RAND_MAX);
  
  return stdev*sqrt(-2*log(U1))*cos(2*PI*U2);
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::addSensors() {
  if (debug_level > 0 && Comm->getRank() == 0) {
    cout << "**** Starting PostprocessManager::addSensors ..." << endl;
  }
    
  // Reading in sensors from a mesh file only works on a single element block (for now)
  // There isn't any problem with multiple blocks, it just hasn't been generalized for sensors yet
  for (size_t r=0; r<objectives.size(); ++r) {
    if (objectives[r].type == "sensors") {
  
      if (objectives[r].sensor_points_file == "mesh") {
        //Teuchos::TimeMonitor localtimer(*importexodustimer);
        this->importSensorsFromExodus(r);
      }
      else if (objectives[r].use_sensor_grid) {
        //Teuchos::TimeMonitor localtimer(*importexodustimer);
        this->importSensorsOnGrid(r);
      }
      else {
        //Teuchos::TimeMonitor localtimer(*importfiletimer);
        this->importSensorsFromFiles(r);
      }
    }
  }
  
  if (debug_level > 0 && Comm->getRank() == 0) {
    cout << "**** Finished PostprocessManager::addSensors ..." << endl;
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::importSensorsFromExodus(const int & objID) {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting PostprocessManager::importSensorsFromExodus() ..." << endl;
    }
  }
  
  //vector<string> mesh_response_names = mesh->efield_names;
  string cresp = objectives[objID].sensor_data_file;
  
  size_t block = objectives[objID].block;
  
  int numFound = 0;
  for (size_t i=0; i<assembler->groups[block].size(); i++) {
    int numSensorsIngrp = mesh->efield_vals[block][i];
    numFound += numSensorsIngrp;
  }
  
  objectives[objID].numSensors = numFound;
  
  if (numFound > 0) {
    
    Kokkos::View<ScalarT**,HostDevice> spts_host("exodus sensors on host",numFound,dimension);
    Kokkos::View<int*[2],HostDevice> spts_owners("exodus sensor owners",numFound);
    
    // TMW: as far as I can tell, this is limited to steady-state data
    Kokkos::View<ScalarT*,HostDevice> stime_host("sensor times", 1);
    stime_host(0) = 0.0;
    Kokkos::View<ScalarT**,HostDevice> sdat_host("sensor data", numFound, 1);
    
    size_t sprog = 0;
    for (size_t i=0; i<assembler->groups[block].size(); i++) {
      int numSensorsIngrp = mesh->efield_vals[block][i];
      
      if (numSensorsIngrp > 0) {
        for (int j=0; j<numSensorsIngrp; j++) {
          // sensorLocation
          std::stringstream ssSensorNum;
          ssSensorNum << j+1;
          string sensorNum = ssSensorNum.str();
          string fieldLocx = "sensor_" + sensorNum + "_Loc_x";
          ptrdiff_t ind_Locx = std::distance(mesh->efield_names.begin(),
                                             std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocx));
          spts_host(sprog,0) = mesh->efield_vals[ind_Locx][i];
          
          if (dimension > 1) {
            string fieldLocy = "sensor_" + sensorNum + "_Loc_y";
            ptrdiff_t ind_Locy = std::distance(mesh->efield_names.begin(),
                                               std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocy));
            spts_host(sprog,1) = mesh->efield_vals[ind_Locy][i];
          }
          if (dimension > 2) {
            string fieldLocz = "sensor_" + sensorNum + "_Loc_z";
            ptrdiff_t ind_Locz = std::distance(mesh->efield_names.begin(),
                                               std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocz));
            spts_host(sprog,2) = mesh->efield_vals[ind_Locz][i];
          }
          // sensorData
          ptrdiff_t ind_Resp = std::distance(mesh->efield_names.begin(),
                                             std::find(mesh->efield_names.begin(), mesh->efield_names.end(), cresp));
          sdat_host(sprog,0) = mesh->efield_vals[ind_Resp][i];
          spts_owners(sprog,0) = i;
          spts_owners(sprog,1) = 0;
          
          sprog++;
          
        }
      }
    }
    
    // ========================================
    // Create and store more compact Views based on number of sensors on this proc
    // ========================================
    
    Kokkos::View<ScalarT**,AssemblyDevice> spts("sensor point", numFound, dimension);
    Kokkos::View<ScalarT*,AssemblyDevice> stime("sensor times", stime_host.extent(0));
    Kokkos::View<ScalarT**,AssemblyDevice> sdat("sensor data", numFound, stime_host.extent(0));
    Kokkos::View<int*[2],HostDevice> sowners("sensor owners", numFound);
    
    auto stime_tmp = create_mirror_view(stime);
    deep_copy(stime_tmp,stime_host);
    deep_copy(stime,stime_tmp);
    
    auto spts_tmp = create_mirror_view(spts);
    auto sdat_tmp = create_mirror_view(sdat);
    
    size_t prog=0;
    
    for (size_type pt=0; pt<spts_host.extent(0); ++pt) {
      for (size_type j=0; j<sowners.extent(1); ++j) {
        sowners(prog,j) = spts_owners(pt,j);
    }
      
      for (size_type j=0; j<spts.extent(1); ++j) {
        spts_tmp(prog,j) = spts_host(pt,j);
      }
      
      for (size_type j=0; j<sdat.extent(1); ++j) {
        sdat_tmp(prog,j) = sdat_host(pt,j);
      }
      prog++;
    }
    deep_copy(spts,spts_tmp);
    deep_copy(sdat,sdat_tmp);
    
    objectives[objID].sensor_points = spts;
    objectives[objID].sensor_times = stime;
    objectives[objID].sensor_data = sdat;
    objectives[objID].sensor_owners = sowners;
    
    // ========================================
    // Evaluate the basis functions and grads for each sensor point
    // ========================================
    
    vector<Kokkos::View<ScalarT****,AssemblyDevice> > csensorBasis;
    vector<Kokkos::View<ScalarT****,AssemblyDevice> > csensorBasisGrad;
    for (size_t k=0; k<assembler->disc->basis_pointers[block].size(); k++) {
      auto basis_ptr = assembler->disc->basis_pointers[block][k];
      string basis_type = assembler->disc->basis_types[block][k];
      int bnum = basis_ptr->getCardinality();
      
      if (basis_type == "HGRAD") {
        Kokkos::View<ScalarT****,AssemblyDevice> cbasis("sensor basis",spts.extent(0), bnum, 1, 1);
        csensorBasis.push_back(cbasis);
        Kokkos::View<ScalarT****,AssemblyDevice> cbasisgrad("sensor basis grad",spts.extent(0), bnum, 1, dimension);
        csensorBasisGrad.push_back(cbasisgrad);
      }
      else if (basis_type == "HVOL") {
        Kokkos::View<ScalarT****,AssemblyDevice> cbasis("sensor basis",spts.extent(0), bnum, 1, 1);
        csensorBasis.push_back(cbasis);
      }
      else if (basis_type == "HDIV") {
        Kokkos::View<ScalarT****,AssemblyDevice> cbasis("sensor basis",spts.extent(0), bnum, 1, dimension);
        csensorBasis.push_back(cbasis);
      }
      else if (basis_type == "HCURL") {
        Kokkos::View<ScalarT****,AssemblyDevice> cbasis("sensor basis",spts.extent(0), bnum, 1, dimension);
        csensorBasis.push_back(cbasis);
      }
    }

    for (size_type pt=0; pt<spts.extent(0); ++pt) {
      
      DRV cpt("point",1,1,dimension);
      auto cpt_sub = subview(cpt,0,0,ALL());
      auto pp_sub = subview(spts,pt,ALL());
      Kokkos::deep_copy(cpt_sub,pp_sub);
      
      auto nodes = assembler->groups[block][sowners(pt,0)]->nodes;
      auto nodes_sv = subview(nodes,sowners(pt,1),ALL(),ALL());
      DRV cnodes("subnodes",1,nodes.extent(1),nodes.extent(2));
      auto cnodes_sv = subview(cnodes,0,ALL(),ALL());
      deep_copy(cnodes_sv,nodes_sv);

      DRV refpt("refsenspts",1,dimension);
      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> corientation("curr orient",1);
      
      DRV refpt_tmp = assembler->disc->mapPointsToReference(cpt, cnodes, assembler->groupData[block]->cell_topo);
      
      for (size_type d=0; d<refpt_tmp.extent(2); ++d) {
        refpt(0,d) = refpt_tmp(0,0,d);
      }
 
      auto orient = assembler->groups[block][sowners(pt,0)]->orientation;
      corientation(0) = orient(sowners(pt,1));

      for (size_t k=0; k<assembler->disc->basis_pointers[block].size(); k++) {
        auto basis_ptr = assembler->disc->basis_pointers[block][k];
        string basis_type = assembler->disc->basis_types[block][k];
        auto cellTopo = assembler->groupData[block]->cell_topo;
        
        Kokkos::View<ScalarT****,AssemblyDevice> bvals2, bgradvals2;
        DRV bvals = disc->evaluateBasis(block, k, cnodes, refpt, cellTopo, corientation);

        if (basis_type == "HGRAD") {

          auto bvals_sv = subview(bvals,0,ALL(),ALL());
          auto bvals2_sv = subview(csensorBasis[k],pt,ALL(),ALL(),0);
          deep_copy(bvals2_sv,bvals_sv);

          DRV bgradvals = assembler->disc->evaluateBasisGrads(basis_ptr, cnodes, refpt, cellTopo, corientation);
          auto bgradvals_sv = subview(bgradvals,0,ALL(),ALL(),ALL());
          auto bgrad_sv = subview(csensorBasisGrad[k],pt,ALL(),ALL(),ALL());
          deep_copy(bgrad_sv,bgradvals_sv);

        }
        else if (basis_type == "HVOL") {
          auto bvals_sv = subview(bvals,0,ALL(),ALL());
          auto bvals2_sv = subview(csensorBasis[k],pt,ALL(),ALL(),0);
          deep_copy(bvals2_sv,bvals_sv);
        }
        else if (basis_type == "HDIV") {
          auto bvals_sv = subview(bvals,0,ALL(),ALL(),ALL());
          auto bvals2_sv = subview(csensorBasis[k],pt,ALL(),ALL(),ALL());
          deep_copy(bvals2_sv,bvals_sv);
        }
        else if (basis_type == "HCURL") {
          auto bvals_sv = subview(bvals,0,ALL(),ALL(),ALL());
          auto bvals2_sv = subview(csensorBasis[k],pt,ALL(),ALL(),ALL());
          deep_copy(bvals2_sv,bvals_sv);
        }
      }
    }
    objectives[objID].sensor_basis = csensorBasis;
    objectives[objID].sensor_basis_grad = csensorBasisGrad;
      
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SensorManager::importSensorsFromExodus() ..." << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::importSensorsFromFiles(const int & objID) {
    
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting PostprocessManager::importSensorsFromFiles() ..." << endl;
    }
  }
  
  size_t block = objectives[objID].block;
    
  // ========================================
  // Import the data from the files
  // ========================================
  
  Data sdata;
  bool have_data = false;
  
  if (objectives[objID].sensor_data_file == "") {
    sdata = Data("Sensor Measurements", dimension,
                 objectives[objID].sensor_points_file);
  }
  else {
    sdata = Data("Sensor Measurements", dimension,
                 objectives[objID].sensor_points_file,
                 objectives[objID].sensor_data_file, false);
    have_data = true;
  }
  
  // ========================================
  // Save the locations in the appropriate view
  // ========================================
  
  Kokkos::View<ScalarT**,HostDevice> spts_host = sdata.getPoints();
  std::vector<Kokkos::View<ScalarT**,HostDevice> >  sensor_data_host;
  if (have_data) {
    sensor_data_host = sdata.getData();
  }
  
  // Check that the data matches the expected format
  if (spts_host.extent(1) != static_cast<size_type>(dimension)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
                               "Error: sensor points dimension does not match simulation dimension");
  }
  if (have_data) {
    if (spts_host.extent(0)+1 != sensor_data_host.size()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
                                 "Error: number of sensors does not match data");
    }
  }
  
  // ========================================
  // Import the data from the files
  // ========================================
  
  Kokkos::View<ScalarT*,HostDevice> stime_host;
  
  Kokkos::View<ScalarT**,HostDevice> sdat_host;
  
  if (have_data) {
    stime_host = Kokkos::View<ScalarT*,HostDevice>("sensor times", sensor_data_host[0].extent(1));
  
    for (size_type d=0; d<sensor_data_host[0].extent(1); ++d) {
      stime_host(d) = sensor_data_host[0](0,d);
    }
    
    sdat_host = Kokkos::View<ScalarT**,HostDevice>("sensor data", sensor_data_host.size()-1,
                                                   sensor_data_host[0].extent(1));
    
    for (size_type pt=1; pt<sensor_data_host.size(); ++pt) {
      for (size_type d=0; d<sensor_data_host[pt].extent(1); ++d) {
        sdat_host(pt-1,d) = sensor_data_host[pt](0,d);
      }
    }
  }
  
  // ========================================
  // Determine which element contains each sensor point
  // Note: a given processor might not find any
  // ========================================
  
  Kokkos::View<int*[2],HostDevice> spts_owners("sensor owners",spts_host.extent(0));
  Kokkos::View<bool*,HostDevice> spts_found("sensors found",spts_host.extent(0));
  
  this->locateSensorPoints(block, spts_host, spts_owners, spts_found);
  
  // ========================================
  // Determine the number of sensors on this proc
  // ========================================
  
  size_t numFound = 0;
  for (size_type pt=0; pt<spts_found.extent(0); ++pt) {
    if (spts_found(pt)) {
      numFound++;
    }
  }
  
  objectives[objID].numSensors = numFound;
  objectives[objID].sensor_found = spts_found;
  
  if (numFound > 0) {
    
    // ========================================
    // Create and store more compact Views based on number of sensors on this proc
    // ========================================
    
    Kokkos::View<ScalarT**,AssemblyDevice> spts("sensor point", numFound, dimension);
    Kokkos::View<ScalarT*,AssemblyDevice> stime;
    Kokkos::View<ScalarT**,AssemblyDevice> sdat;
    Kokkos::View<int*[2],HostDevice> sowners("sensor owners", numFound);
        
    auto spts_tmp = create_mirror_view(spts);
    
    if (have_data) {
      stime = Kokkos::View<ScalarT*,AssemblyDevice>("sensor times", stime_host.extent(0));
      auto stime_tmp = create_mirror_view(stime);
      deep_copy(stime_tmp,stime_host);
      deep_copy(stime,stime_tmp);
      
      sdat = Kokkos::View<ScalarT**,AssemblyDevice>("sensor data", numFound, stime_host.extent(0));
      auto sdat_tmp = create_mirror_view(sdat);
      size_t prog=0;
      
      for (size_type pt=0; pt<spts_host.extent(0); ++pt) {
        if (spts_found(pt)) {
          if (have_data) {
            for (size_type j=0; j<sdat.extent(1); ++j) {
              sdat_tmp(prog,j) = sdat_host(pt,j);
            }
          }
          prog++;
        }
      }
      deep_copy(sdat,sdat_tmp);
    }
    
    size_t prog=0;
    
    for (size_type pt=0; pt<spts_host.extent(0); ++pt) {
      if (spts_found(pt)) {
        for (size_type j=0; j<sowners.extent(1); ++j) {
          sowners(prog,j) = spts_owners(pt,j);
        }
        for (size_type j=0; j<spts.extent(1); ++j) {
          spts_tmp(prog,j) = spts_host(pt,j);
        }
        prog++;
      }
    }
    deep_copy(spts,spts_tmp);
    
    objectives[objID].sensor_points = spts;
    objectives[objID].sensor_times = stime;
    objectives[objID].sensor_data = sdat;
    objectives[objID].sensor_owners = sowners;
    
    // ========================================
    // Evaluate the basis functions and grads for each sensor point
    // ========================================
    
    this->computeSensorBasis(objID);
    
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SensorManager::importSensorsFromFiles() ..." << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::importSensorsOnGrid(const int & objID) {
    
  if (debug_level > 0 && Comm->getRank() == 0) {
    cout << "**** Starting PostprocessManager::importSensorsOnGrid() ..." << endl;
  }
  
  // Check that the data matches the expected format
  if (dimension != 3) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
                               "Error: defining a grid of sensor points is only implemented in 3-dimensions");
  }
  
  size_t block = objectives[objID].block;
  
  // ========================================
  // Save the locations in the appropriate view
  // ========================================
  
  double xmin = objectives[objID].sensor_grid_xmin;
  double xmax = objectives[objID].sensor_grid_xmax;
  double ymin = objectives[objID].sensor_grid_ymin;
  double ymax = objectives[objID].sensor_grid_ymax;
  double zmin = objectives[objID].sensor_grid_zmin;
  double zmax = objectives[objID].sensor_grid_zmax;
            
  int Nx = objectives[objID].sensor_grid_Nx;
  int Ny = objectives[objID].sensor_grid_Ny;
  int Nz = objectives[objID].sensor_grid_Nz;
            
  double dx = (Nx > 1) ? (xmax-xmin)/(Nx-1) : 0.0;
  double dy = (Ny > 1) ? (ymax-ymin)/(Ny-1) : 0.0;
  double dz = (Nz > 1) ? (zmax-zmin)/(Nz-1) : 0.0;
            
  std::vector<double> xgrid(Nx);
  std::vector<double> ygrid(Ny);
  std::vector<double> zgrid(Nz);
            
  double xval = xmin;
  for (int i=0; i<Nx; ++i) {
    xgrid[i] = xval;
    xval += dx;
  }
            
  double yval = ymin;
  for (int i=0; i<Ny; ++i) {
    ygrid[i] = yval;
    yval += dy;
  }
            
  double zval = zmin;
  for (int i=0; i<Nz; ++i) {
    zgrid[i] = zval;
    zval += dz;
  }
  
  Kokkos::View<ScalarT**,HostDevice> spts_host("sensor locations",Nx*Ny*Nz,dimension);
  
  size_t prog = 0;
  for (int i=0; i<Nx; i++) {
    for (int j=0; j<Ny; j++) {
      for (int k=0; k<Nz; k++) {
        spts_host(prog,0) = xgrid[i];
        spts_host(prog,1) = ygrid[j];
        spts_host(prog,2) = zgrid[k];
        prog++;
      }
    }
  }
          
  // ========================================
  // Determine which element contains each sensor point
  // Note: a given processor might not find any
  // ========================================
  
  Kokkos::View<int*[2],HostDevice> spts_owners("sensor owners",spts_host.extent(0));
  Kokkos::View<bool*,HostDevice> spts_found("sensors found",spts_host.extent(0));
  
  this->locateSensorPoints(block, spts_host, spts_owners, spts_found);

  // ========================================
  // Determine the number of sensors on this proc
  // ========================================
  
  size_t numFound = 0;
  for (size_type pt=0; pt<spts_found.extent(0); ++pt) {
    if (spts_found(pt)) {
      numFound++;
    }
  }
    
  objectives[objID].numSensors = numFound;
  objectives[objID].sensor_found = spts_found;
  
  if (numFound > 0) {
    
    // ========================================
    // Create and store more compact Views based on number of sensors on this proc
    // ========================================
    
    Kokkos::View<ScalarT**,AssemblyDevice> spts("sensor point", numFound, dimension);
    Kokkos::View<ScalarT*,AssemblyDevice> stime;
    Kokkos::View<ScalarT**,AssemblyDevice> sdat;
    Kokkos::View<int*[2],HostDevice> sowners("sensor owners", numFound);
        
    auto spts_tmp = create_mirror_view(spts);
    
    size_t prog=0;
    
    for (size_type pt=0; pt<spts_host.extent(0); ++pt) {
      if (spts_found(pt)) {
        for (size_type j=0; j<sowners.extent(1); ++j) {
          sowners(prog,j) = spts_owners(pt,j);
        }
        for (size_type j=0; j<spts.extent(1); ++j) {
          spts_tmp(prog,j) = spts_host(pt,j);
        }
        prog++;
      }
    }
    deep_copy(spts,spts_tmp);
    
    objectives[objID].sensor_points = spts;
    objectives[objID].sensor_times = stime;
    objectives[objID].sensor_data = sdat;
    objectives[objID].sensor_owners = sowners;
    
    // ========================================
    // Evaluate the basis functions and grads for each sensor point
    // ========================================
    
    this->computeSensorBasis(objID);
  }
  if (debug_level > 0 && Comm->getRank() == 0) {
    cout << "**** Finished SensorManager::importSensorsOnGrid() ..." << endl;
  }

}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeSensorBasis(const int & objID) {
    
  size_t block = objectives[objID].block;
  auto spts = objectives[objID].sensor_points;
  auto sowners = objectives[objID].sensor_owners;

  vector<Kokkos::View<ScalarT****,AssemblyDevice> > csensorBasis;
  vector<Kokkos::View<ScalarT****,AssemblyDevice> > csensorBasisGrad;
  for (size_t k=0; k<assembler->disc->basis_pointers[block].size(); k++) {
    auto basis_ptr = assembler->disc->basis_pointers[block][k];
    string basis_type = assembler->disc->basis_types[block][k];
    int bnum = basis_ptr->getCardinality();
      
    if (basis_type == "HGRAD") {
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis("sensor basis",spts.extent(0), bnum, 1, 1);
      csensorBasis.push_back(cbasis);
      Kokkos::View<ScalarT****,AssemblyDevice> cbasisgrad("sensor basis grad",spts.extent(0), bnum, 1, dimension);
      csensorBasisGrad.push_back(cbasisgrad);
    }
    else if (basis_type == "HVOL") {
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis("sensor basis",spts.extent(0), bnum, 1, 1);
      csensorBasis.push_back(cbasis);
    }
    else if (basis_type == "HDIV") {
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis("sensor basis",spts.extent(0), bnum, 1, dimension);
      csensorBasis.push_back(cbasis);
    }
    else if (basis_type == "HCURL") {
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis("sensor basis",spts.extent(0), bnum, 1, dimension);
      csensorBasis.push_back(cbasis);
    }
  }

     
  for (size_type pt=0; pt<spts.extent(0); ++pt) {
      
    DRV cpt("point",1,1,dimension);
    auto cpt_sub = subview(cpt,0,0,ALL());
    auto pp_sub = subview(spts,pt,ALL());
    Kokkos::deep_copy(cpt_sub,pp_sub);
      
    auto nodes = assembler->groups[block][sowners(pt,0)]->nodes;
    auto nodes_sv = subview(nodes,sowners(pt,1),ALL(),ALL());
    DRV cnodes("subnodes",1,nodes.extent(1),nodes.extent(2));
    auto cnodes_sv = subview(cnodes,0,ALL(),ALL());
    deep_copy(cnodes_sv,nodes_sv);

    DRV refpt("refsenspts",1,dimension);
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> corientation("curr orient",1);
      
    DRV refpt_tmp = assembler->disc->mapPointsToReference(cpt, cnodes, assembler->groupData[block]->cell_topo);

    for (size_type d=0; d<refpt_tmp.extent(2); ++d) {
      refpt(0,d) = refpt_tmp(0,0,d);
    }
      
    auto orient = assembler->groups[block][sowners(pt,0)]->orientation;
    corientation(0) = orient(sowners(pt,1));
    

    for (size_t k=0; k<assembler->disc->basis_pointers[block].size(); k++) {
      auto basis_ptr = assembler->disc->basis_pointers[block][k];
      string basis_type = assembler->disc->basis_types[block][k];
      auto cellTopo = assembler->groupData[block]->cell_topo;
        
      Kokkos::View<ScalarT****,AssemblyDevice> bvals2, bgradvals2;
      DRV bvals = disc->evaluateBasis(block, k, cnodes, refpt, cellTopo, corientation);

      if (basis_type == "HGRAD") {
        auto bvals_sv = subview(bvals,0,ALL(),ALL());
        auto bvals2_sv = subview(csensorBasis[k],pt,ALL(),ALL(),0);
        deep_copy(bvals2_sv,bvals_sv);
          
        DRV bgradvals = assembler->disc->evaluateBasisGrads(basis_ptr, cnodes, refpt, cellTopo, corientation);
        auto bgradvals_sv = subview(bgradvals,0,ALL(),ALL(),ALL());
        auto bgrad_sv = subview(csensorBasisGrad[k],pt,ALL(),ALL(),ALL());
        deep_copy(bgrad_sv,bgradvals_sv);
      }
      else if (basis_type == "HVOL") {
        auto bvals_sv = subview(bvals,0,ALL(),ALL());
        auto bvals2_sv = subview(csensorBasis[k],pt,ALL(),ALL(),0);
        deep_copy(bvals2_sv,bvals_sv);
      }
      else if (basis_type == "HDIV") {
        auto bvals_sv = subview(bvals,0,ALL(),ALL(),ALL());
        auto bvals2_sv = subview(csensorBasis[k],pt,ALL(),ALL(),ALL());
        deep_copy(bvals2_sv,bvals_sv);
      }
      else if (basis_type == "HCURL") {
        auto bvals_sv = subview(bvals,0,ALL(),ALL(),ALL());
        auto bvals2_sv = subview(csensorBasis[k],pt,ALL(),ALL(),ALL());
        deep_copy(bvals2_sv,bvals_sv);
      }
    }
  }
  objectives[objID].sensor_basis = csensorBasis;
  objectives[objID].sensor_basis_grad = csensorBasisGrad;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::locateSensorPoints(const int & block, 
                                                  Kokkos::View<ScalarT**,HostDevice> spts_host,
                                                  Kokkos::View<int*[2],HostDevice> spts_owners, 
                                                  Kokkos::View<bool*,HostDevice> spts_found) {
  
  global_num_sensors = spts_host.extent(0);
  size_t checksPerformed = 0;
    
  for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
    
    auto nodes = assembler->groups[block][grp]->nodes;
    auto nodes_host = create_mirror_view(nodes);
    deep_copy(nodes_host,nodes);
    
    // Create a bounding box for the element
    // This serves as a preprocessing check to avoid unnecessary inclusion checks
    // If a sensor point is not in the box, then it is not in the element
    Kokkos::View<double**[2],HostDevice> nodebox("bounding box",nodes_host.extent(0),dimension);
    for (size_type p=0; p<nodes_host.extent(0); ++p) {
      for (size_type dim=0; dim<nodes_host.extent(2); ++dim) {
        double dmin = 1.0e300;
        double dmax = -1.0e300;
        for (size_type k=0; k<nodes_host.extent(1); ++k) {
          dmin = std::min(dmin,nodes_host(p,k,dim));
          dmax = std::max(dmax,nodes_host(p,k,dim));
        }
        nodebox(p,dim,0) = dmin;
        nodebox(p,dim,1) = dmax;
      }
    }
    
    for (size_type pt=0; pt<spts_host.extent(0); ++pt) {
      if (!spts_found(pt)) {
        for (size_type p=0; p<nodebox.extent(0); ++p) {
          double xbuff = 0.1*(nodebox(p,0,1)-nodebox(p,0,0));
          double ybuff = 0.1*(nodebox(p,1,1)-nodebox(p,1,0));
          double zbuff = 0.1*(nodebox(p,2,1)-nodebox(p,2,0));
          bool proceed = true;
          if (spts_host(pt,0)<nodebox(p,0,0)-xbuff || spts_host(pt,0)>nodebox(p,0,1)+xbuff) {
            proceed = false;
          }
          if (proceed && dimension > 1) {
            if (spts_host(pt,1)<nodebox(p,1,0)-ybuff || spts_host(pt,1)>nodebox(p,1,1)+ybuff) {
              proceed = false;
            }
          }
          if (proceed && dimension > 2) {
            if (spts_host(pt,2)<nodebox(p,2,0)-zbuff || spts_host(pt,2)>nodebox(p,2,1)+zbuff) {
              proceed = false;
            }
          }
          
          if (proceed) {
            checksPerformed++;
            // Need to use DRV, which are on AssemblyDevice
            // We have less control here
            DRV phys_pt("phys_pt",1,1,dimension);
            auto phys_pt_host = create_mirror_view(phys_pt);
            for (size_type d=0; d<spts_host.extent(1); ++d) {
              phys_pt_host(0,0,d) = spts_host(pt,d);
            }
            deep_copy(phys_pt,phys_pt_host);
            DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
            auto n_sub = subview(nodes,p,ALL(),ALL());
            auto cn_sub = subview(cnodes,0,ALL(),ALL());
            Kokkos::deep_copy(cn_sub,n_sub);
            
            auto inRefgrp = assembler->disc->checkInclusionPhysicalData(phys_pt,cnodes,
                                                                         assembler->groupData[block]->cell_topo,
                                                                         1.0e-14);
            auto inRef_host = create_mirror_view(inRefgrp);
            deep_copy(inRef_host,inRefgrp);
            if (inRef_host(0,0)) {
              spts_found(pt) = true;
              spts_owners(pt,0) = grp;
              spts_owners(pt,1) = p;
            }
            else {
              //cout << "Sensor was in bounding box, but not in element: " << endl;
              //KokkosTools::print(phys_pt);
              //KokkosTools::print(cnodes);
            }
          }
        }
      }// found
    } // pt
  } // elem

  bool check_found = false;
  if (check_found) {
    for (size_type pt=0; pt<spts_found.extent(0); ++pt) {
      size_t fnd_flag = 0;
      if (spts_found(pt)) {
        fnd_flag = 1;
      }
      size_t globalFound = 0;
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&fnd_flag,&globalFound);
      if (Comm->getRank() == 0) {
        if (globalFound == 0) {
           cout << " - Sensor " << pt << " was not found" << endl;
        }
        else if (globalFound > 1) {
           cout << " - Sensor " << pt << " was found " << globalFound << " times" << endl;
        }
      }
    }
  }
  if (verbosity >= 10) {
    size_t numFound = 0;
    for (size_type pt=0; pt<spts_found.extent(0); ++pt) {
      if (spts_found(pt)) {
        numFound++;
      }
    }
    cout << "Total number of Intrepid inclusion checks performed on processor " << Comm->getRank() << ": " << checksPerformed << endl;
    cout << " - Processor " << Comm->getRank() << " has " << numFound << " sensors" << endl;

    size_t globalFound = 0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&numFound,&globalFound);
    if (Comm->getRank() == 0) {
      cout << " - Total Number of Sensors: " << spts_found.extent(0) << endl;
      cout << " - Total Number of Sensors Located: " << globalFound << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::setNewExodusFile(string & newfile) { 
  if (isTD && write_solution) {
    mesh->stk_mesh->setupExodusFile(newfile);
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::Array<ScalarT> PostprocessManager<Node>::collectResponses() {
  
  //
  // May be multiple objectives, which store the responses
  // Each objective can store different types of responses:
  //   1.  Integrated response: scalar over time (each proc stores own contribution)
  //   2.  Sensor response: scalar over time at each sensor (each proc stores only own sensor)
  //   3.  Sensor solution: state variable over time at each sensor (each proc stores only own sensor)
  //

  ////////////////////////////////
  // First, determne how many responses have been computed
  ////////////////////////////////
  
  int totalresp = 0;
  vector<int> response_sizes;
  for (size_t obj=0; obj<objectives.size(); ++obj) {
    if (objectives[obj].type == "sensors") {
      int totalsens = objectives[obj].sensor_found.extent_int(0); // this is actually the global number of sensors
      if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln) {
        size_t numtimes = objectives[obj].sensor_solution_data.size(); 
        int numsols = objectives[obj].sensor_solution_data[0].extent_int(1); 
        int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
        totalsens *= numtimes*numsols*numdims;
      }
      else {
        size_t numtimes = objectives[obj].response_times.size();
        totalsens *= numtimes;
      }
      response_sizes.push_back(totalsens);
      //totalresp += totalsens;
    }
    else if (objectives[obj].type == "integrated response") {
      response_sizes.push_back(objectives[obj].response_times.size());
      //totalresp += objectives[obj].response_times.size();
    }
  }

  for (size_t i=0; i<response_sizes.size(); ++i) {
    totalresp += response_sizes[i];
  }
  
  int glbresp = 0;
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&totalresp,&glbresp);
  Kokkos::View<ScalarT*,HostDevice> newresp("response",glbresp);
  Teuchos::Array<ScalarT> localarraydata(glbresp,0.0);
      
  ////////////////////////////////
  // Next, we fill in the responses
  ////////////////////////////////
  
  size_t overallprog = 0;
  for (size_t obj=0; obj<objectives.size(); ++obj) {
    if (objectives[obj].type == "sensors") {
      int numsensors = objectives[obj].numSensors;
      if (numsensors>0) {
        Kokkos::View<int*,HostDevice> sensorIDs("sensor IDs owned by proc", numsensors);
        size_t sprog=0;
        auto sensor_found = objectives[obj].sensor_found;
        for (size_type s=0; s<sensor_found.extent(0); ++s) {
          if (sensor_found(s)) {
            sensorIDs(sprog) = s;
            ++sprog;
          }
        }

        if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln) {
          for (size_type sens=0; sens<numsensors; ++sens) {
            size_t numtimes = objectives[obj].sensor_solution_data.size(); 
            int numsols = objectives[obj].sensor_solution_data[0].extent_int(1);
            int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
            int sensnum = sensorIDs(sens);
            size_t startind = overallprog + sensnum*(numsols*numdims*numtimes);
            size_t cprog = 0;
            for (int tt=0; tt<numtimes; ++tt) {
              for (int ss=0; ss<numsols; ++ss) {
                for (int dd=0; dd<numdims; ++dd) {
                  localarraydata[startind+cprog] = objectives[obj].sensor_solution_data[tt](sens,ss,dd);
                  cprog++;
                }
              }
            }
          }
        }
        else {
          for (size_type sens=0; sens<numsensors; ++sens) {
            size_t numtimes = objectives[obj].response_data.size(); 
            int sensnum = sensorIDs(sens);
            size_t startind = overallprog + sensnum*(numtimes);
            size_t cprog = 0;
            for (int tt=0; tt<numtimes; ++tt) {
              localarraydata[startind+cprog] = objectives[obj].response_data[tt](sens);
              cprog++;
            }
          }
              
        }
      }
    }
    else if (objectives[obj].type == "integrated response") {
      for (size_t tt=0; tt<objectives[obj].response_times.size(); ++tt) {
        localarraydata[overallprog+tt] = objectives[obj].scalar_response_data[tt];
      }
    }
    overallprog += response_sizes[obj];
  }

  Teuchos::Array<ScalarT> globalarraydata(glbresp,0.0);

  const int numentries = totalresp;
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,numentries,&localarraydata[0],&globalarraydata[0]);
      
  return globalarraydata;    
}

// ========================================================================================
// ========================================================================================

// Explicit template instantiations
template class MrHyDE::PostprocessManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
  template class MrHyDE::PostprocessManager<SubgridSolverNode>;
#endif
