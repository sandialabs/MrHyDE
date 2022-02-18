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
                                             vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                                             Teuchos::RCP<AssemblyManager<Node> > & assembler_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_),
assembler(assembler_), functionManagers(functionManagers_) {
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
                                             vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                                             Teuchos::RCP<MultiscaleManager> & multiscale_manager_,
                                             Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                                             Teuchos::RCP<ParameterManager<Node> > & params_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_),
assembler(assembler_), params(params_), functionManagers(functionManagers_), multiscale_manager(multiscale_manager_) {
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
  write_subgrid_solution = settings->sublist("Postprocess").get("write subgrid solution",false);
  write_HFACE_variables = settings->sublist("Postprocess").get("write HFACE variables",false);
  exodus_filename = settings->sublist("Postprocess").get<string>("output file","output")+".exo";
  write_optimization_solution = settings->sublist("Postprocess").get("create optimization movie",false);
  write_cell_number = settings->sublist("Postprocess").get("write cell number",false);
  compute_objective = settings->sublist("Postprocess").get("compute objective",false);
  discrete_objective_scale_factor = settings->sublist("Postprocess").get("scale factor for discrete objective",1.0);
  cellfield_reduction = settings->sublist("Postprocess").get<string>("extra cell field reduction","mean");
  write_database_id = settings->sublist("Solver").get<bool>("use basis database",false);
  compute_flux_response = settings->sublist("Postprocess").get("compute flux response",false);

  compute_integrated_quantities = settings->sublist("Postprocess").get("compute integrated quantities",false);
 
  compute_weighted_norm = settings->sublist("Postprocess").get<bool>("compute weighted norm",false);
  
  setnames = phys->setnames;
  
  for (size_t set=0; set<setnames.size(); ++set) {
    soln.push_back(Teuchos::rcp(new SolutionStorage<Node>(settings)));
  }
  string analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  save_solution = false;
  
  if (analysis_type == "forward+adjoint" || analysis_type == "ROL" || analysis_type == "ROL_SIMOPT") {
    save_solution = true; // default is false
    if (settings->sublist("Analysis").sublist("ROL").sublist("General").get<bool>("Generate data",false)) {
      for (size_t set=0; set<setnames.size(); ++set) {
        datagen_soln.push_back(Teuchos::rcp(new SolutionStorage<Node>(settings)));
      }
    }
  }
  
  append = "";
  
  if (verbosity > 0 && Comm->getRank() == 0) {
    if (write_solution && !write_HFACE_variables) {
      bool have_HFACE_vars = false;
      vector<vector<vector<string> > > types = phys->types;
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
  blocknames = phys->blocknames;
  sideSets = phys->sidenames;
  
  numNodesPerElem = settings->sublist("Mesh").get<int>("numNodesPerElem",4); // actually set by mesh interface
  spaceDim = phys->spaceDim;//mesh->stk_mesh->getDimension();
    
  response_type = settings->sublist("Postprocess").get("response type", "pointwise"); // or "global"
  have_sensor_data = settings->sublist("Analysis").get("have sensor data", false); // or "global"
  save_sensor_data = settings->sublist("Analysis").get("save sensor data",false);
  sname = settings->sublist("Analysis").get("sensor prefix","sensor");
    
  stddev = settings->sublist("Analysis").get("additive normal noise standard dev",0.0);
  write_dakota_output = settings->sublist("Postprocess").get("write Dakota output",false);
  
  varlist = phys->varlist;
  
  for (size_t block=0; block<blocknames.size(); ++block) {
        
    if (settings->sublist("Postprocess").isSublist("Responses")) {
      Teuchos::ParameterList resps = settings->sublist("Postprocess").sublist("Responses");
      Teuchos::ParameterList::ConstIterator rsp_itr = resps.begin();
      while (rsp_itr != resps.end()) {
        string entry = resps.get<string>(rsp_itr->first);
        functionManagers[block]->addFunction(rsp_itr->first,entry,"ip");
        rsp_itr++;
      }
    }
    if (settings->sublist("Postprocess").isSublist("Weights")) {
      Teuchos::ParameterList wts = settings->sublist("Postprocess").sublist("Weights");
      Teuchos::ParameterList::ConstIterator wts_itr = wts.begin();
      while (wts_itr != wts.end()) {
        string entry = wts.get<string>(wts_itr->first);
        functionManagers[block]->addFunction(wts_itr->first,entry,"ip");
        wts_itr++;
      }
    }
    if (settings->sublist("Postprocess").isSublist("Targets")) {
      Teuchos::ParameterList tgts = settings->sublist("Postprocess").sublist("Targets");
      Teuchos::ParameterList::ConstIterator tgt_itr = tgts.begin();
      while (tgt_itr != tgts.end()) {
        string entry = tgts.get<string>(tgt_itr->first);
        functionManagers[block]->addFunction(tgt_itr->first,entry,"ip");
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
    vector<vector<vector<string> > > types = phys->types;
    
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
      functionManagers[block]->addFunction(ef_itr->first,entry,"ip");
      functionManagers[block]->addFunction(ef_itr->first,entry,"point");
      ef_itr++;
    }
    extrafields_list.push_back(block_ef);
    
    // Add extra cell fields
    vector<string> block_ecf;
    Teuchos::ParameterList ecfields = blockPpSettings.sublist("Extra cell fields");
    Teuchos::ParameterList::ConstIterator ecf_itr = ecfields.begin();
    while (ecf_itr != ecfields.end()) {
      string entry = ecfields.get<string>(ecf_itr->first);
      block_ecf.push_back(ecf_itr->first);
      functionManagers[block]->addFunction(ecf_itr->first,entry,"ip");
      ecf_itr++;
    }
    extracellfields_list.push_back(block_ecf);
    
    // Add derived quantities from physics modules
    vector<string> block_dq;
    for (size_t set=0; set<phys->modules.size(); ++set) {
      for (size_t m=0; m<phys->modules[set][block].size(); ++m) {
        vector<string> dqnames = phys->modules[set][block][m]->getDerivedNames();
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
    for (size_t set=0; set<phys->modules.size(); ++set) {
      for (size_t m=0; m<phys->modules[set][block].size(); ++m) {
        vector< vector<string> > integrandsNamesAndTypes =
        phys->modules[set][block][m]->setupIntegratedQuantities(spaceDim);
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
          functionManagers[block]->addFunction("true "+vars[j],expression,"ip");
        }
      }
      if (true_solns.isParameter("grad("+vars[j]+")[x]") || true_solns.isParameter("grad("+vars[j]+")[y]") || true_solns.isParameter("grad("+vars[j]+")[z]")) { // GRAD of the solution at volumetric ip
        if (ctypes[j] == "HGRAD") {
          std::pair<string,string> newerr(vars[j],"GRAD");
          block_error_list.push_back(newerr);
          
          string expression = true_solns.get<string>("grad("+vars[j]+")[x]","0.0");
          functionManagers[block]->addFunction("true grad("+vars[j]+")[x]",expression,"ip");
          if (spaceDim>1) {
            expression = true_solns.get<string>("grad("+vars[j]+")[y]","0.0");
            functionManagers[block]->addFunction("true grad("+vars[j]+")[y]",expression,"ip");
          }
          if (spaceDim>2) {
            expression = true_solns.get<string>("grad("+vars[j]+")[z]","0.0");
            functionManagers[block]->addFunction("true grad("+vars[j]+")[z]",expression,"ip");
          }
        }
      }
      if (true_solns.isParameter(vars[j]+" face")) { // solution at face/side ip
        if (ctypes[j] == "HGRAD" || ctypes[j] == "HFACE") {
          std::pair<string,string> newerr(vars[j],"L2 FACE");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>(vars[j]+" face","0.0");
          functionManagers[block]->addFunction("true "+vars[j],expression,"side ip");
          
        }
      }
      if (true_solns.isParameter(vars[j]+"[x]") || true_solns.isParameter(vars[j]+"[y]") || true_solns.isParameter(vars[j]+"[z]")) { // vector solution at volumetric ip
        if (ctypes[j] == "HDIV" || ctypes[j] == "HCURL") {
          std::pair<string,string> newerr(vars[j],"L2 VECTOR");
          block_error_list.push_back(newerr);
          
          string expression = true_solns.get<string>(vars[j]+"[x]","0.0");
          functionManagers[block]->addFunction("true "+vars[j]+"[x]",expression,"ip");
          
          if (spaceDim>1) {
            expression = true_solns.get<string>(vars[j]+"[y]","0.0");
            functionManagers[block]->addFunction("true "+vars[j]+"[y]",expression,"ip");
          }
          if (spaceDim>2) {
            expression = true_solns.get<string>(vars[j]+"[z]","0.0");
            functionManagers[block]->addFunction("true "+vars[j]+"[z]",expression,"ip");
          }
        }
      }
      if (true_solns.isParameter("div("+vars[j]+")")) { // div of solution at volumetric ip
        if (ctypes[j] == "HDIV") {
          std::pair<string,string> newerr(vars[j],"DIV");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>("div("+vars[j]+")","0.0");
          functionManagers[block]->addFunction("true div("+vars[j]+")",expression,"ip");
          
        }
      }
      if (true_solns.isParameter("curl("+vars[j]+")[x]") || true_solns.isParameter("curl("+vars[j]+")[y]") || true_solns.isParameter("curl("+vars[j]+")[z]")) { // vector solution at volumetric ip
        if (ctypes[j] == "HCURL") {
          std::pair<string,string> newerr(vars[j],"CURL");
          block_error_list.push_back(newerr);
          
          string expression = true_solns.get<string>("curl("+vars[j]+")[x]","0.0");
          functionManagers[block]->addFunction("true curl("+vars[j]+")[x]",expression,"ip");
          
          if (spaceDim>1) {
            expression = true_solns.get<string>("curl("+vars[j]+")[y]","0.0");
            functionManagers[block]->addFunction("true curl("+vars[j]+")[y]",expression,"ip");
          }
          if (spaceDim>2) {
            expression = true_solns.get<string>("curl("+vars[j]+")[z]","0.0");
            functionManagers[block]->addFunction("true curl("+vars[j]+")[z]",expression,"ip");
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
    objective newobj(objsettings,obj_itr->first,block,functionManagers[block]);
    objectives.push_back(newobj);
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
    fluxResponse newflux(frsettings,fluxr_itr->first,block,functionManagers[block]);
    fluxes.push_back(newflux);
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
    integratedQuantity newIQ(iqsettings,iqs_itr->first,block,functionManagers[block]);
    IQs.push_back(newIQ);
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
                             block,functionManagers[block]);
    IQs.push_back(newIQ);
  }
  
  return IQs;

}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::record(vector<vector_RCP> & current_soln, const ScalarT & current_time,
                                      const bool & write_this_step, DFAD & objectiveval) {
  if (compute_error) {
    this->computeError(current_time);
  }
  if ((compute_response || compute_objective) && write_this_step) {
    this->computeObjective(current_soln, current_time, objectiveval);
  }
  if (write_solution && write_this_step) {
    this->writeSolution(current_time);
  }
  if (save_solution) {
    for (size_t set=0; set<soln.size(); ++set) {
      soln[set]->store(current_soln[set], current_time, 0);
    }
  }
  if (compute_flux_response) {
    this->computeFluxResponse(current_time);
  }
  if (compute_integrated_quantities) {
    this->computeIntegratedQuantities(current_time);
  }
  if (compute_weighted_norm && write_this_step) {
    this->computeWeightedNorm(current_soln);
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::report() {
  
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
    
    for (size_t obj=0; obj<objectives.size(); ++obj) {
      if (objectives[obj].type == "sensors") {
        string respfile = objectives[obj].response_file+".out";
        std::ofstream respOUT(respfile.c_str());
        respOUT.precision(16);
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
        respOUT.close();
      }
      else if (objectives[obj].type == "integrated response") {
        if (objectives[obj].save_data) {
          string respfile = objectives[obj].response_file+"."+blocknames[objectives[obj].block]+".out";
          std::ofstream respOUT(respfile.c_str());
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
          respOUT.close();
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
      respOUT.open(respfile, std::ios_base::app);
      respOUT.precision(16);
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
                    << integratedQuantities[iLocal][k].val() << std::endl; 
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
      if (multiscale_manager->subgridModels.size() > 0) {
        
        for (size_t m=0; m<multiscale_manager->subgridModels.size(); m++) {
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
    if (assembler->wkset.size()>block) {
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
      assembler->groupData[altblock]->requiresTransient = false;
      
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
      for (size_t cell=0; cell<assembler->groups[block].size(); cell++) {
        if (have_vol_errs) {
          assembler->groups[block][cell]->updateWorkset(seedwhat,true);
        }
        auto wts = assembler->groups[block][cell]->wkset->wts;
        
        for (size_t etype=0; etype<error_list[altblock].size(); etype++) {
          string varname = error_list[altblock][etype].first;
          
          if (error_list[altblock][etype].second == "L2") {
            // compute the true solution
            string expression = varname;
            auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
            auto sol = assembler->wkset[altblock]->getSolutionField(expression);
            
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                ScalarT diff = sol(elem,pt).val() - tsol(elem,pt).val();
#else
                ScalarT diff = sol(elem,pt) - tsol(elem,pt);
#endif
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
          }
          else if (error_list[altblock][etype].second == "GRAD") {
            // compute the true x-component of grad
            string expression = "grad(" + varname + ")[x]";
            auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
            auto sol_x = assembler->wkset[altblock]->getSolutionField(expression);
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                ScalarT diff = sol_x(elem,pt).val() - tsol(elem,pt).val();
#else
                ScalarT diff = sol_x(elem,pt) - tsol(elem,pt);
#endif
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
            
            if (spaceDim > 1) {
              // compute the true y-component of grad
              string expression = "grad(" + varname + ")[y]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_y = assembler->wkset[altblock]->getSolutionField(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                  ScalarT diff = sol_y(elem,pt).val() - tsol(elem,pt).val();
#else
                  ScalarT diff = sol_y(elem,pt) - tsol(elem,pt);
#endif
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
            
            if (spaceDim > 2) {
              // compute the true z-component of grad
              string expression = "grad(" + varname + ")[z]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_z = assembler->wkset[altblock]->getSolutionField(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                  ScalarT diff = sol_z(elem,pt).val() - tsol(elem,pt).val();
#else
                  ScalarT diff = sol_z(elem,pt) - tsol(elem,pt);
#endif
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
          }
          else if (error_list[altblock][etype].second == "DIV") {
            // compute the true divergence
            string expression = "div(" + varname + ")";
            auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
            auto sol_div = assembler->wkset[altblock]->getSolutionField(expression);
            
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                ScalarT diff = sol_div(elem,pt).val() - tsol(elem,pt).val();
#else
                ScalarT diff = sol_div(elem,pt) - tsol(elem,pt);
#endif
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
          }
          else if (error_list[altblock][etype].second == "CURL") {
            // compute the true x-component of grad
            string expression = "curl(" + varname + ")[x]";
            auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
            auto sol_curl_x = assembler->wkset[altblock]->getSolutionField(expression);
            
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                ScalarT diff = sol_curl_x(elem,pt).val() - tsol(elem,pt).val();
#else
                ScalarT diff = sol_curl_x(elem,pt) - tsol(elem,pt);
#endif
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
            
            if (spaceDim > 1) {
              // compute the true y-component of grad
              string expression = "curl(" + varname + ")[y]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_curl_y = assembler->wkset[altblock]->getSolutionField(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                  ScalarT diff = sol_curl_y(elem,pt).val() - tsol(elem,pt).val();
#else
                  ScalarT diff = sol_curl_y(elem,pt) - tsol(elem,pt);
#endif
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
            
            if (spaceDim >2) {
              // compute the true z-component of grad
              string expression = "curl(" + varname + ")[z]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_curl_z = assembler->wkset[altblock]->getSolutionField(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                  ScalarT diff = sol_curl_z(elem,pt).val() - tsol(elem,pt).val();
#else
                  ScalarT diff = sol_curl_z(elem,pt) - tsol(elem,pt);
#endif
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
          }
          else if (error_list[altblock][etype].second == "L2 VECTOR") {
            // compute the true x-component of grad
            string expression = varname + "[x]";
            auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
            auto sol_x = assembler->wkset[altblock]->getSolutionField(expression);
            
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                ScalarT diff = sol_x(elem,pt).val() - tsol(elem,pt).val();
#else
                ScalarT diff = sol_x(elem,pt) - tsol(elem,pt);
#endif
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
            
            if (spaceDim > 1) {
              // compute the true y-component of grad
              string expression = varname + "[y]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_y = assembler->wkset[altblock]->getSolutionField(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                  ScalarT diff = sol_y(elem,pt).val() - tsol(elem,pt).val();
#else
                  ScalarT diff = sol_y(elem,pt) - tsol(elem,pt);
#endif
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
            
            if (spaceDim > 2) {
              // compute the true z-component of grad
              string expression = varname + "[z]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_z = assembler->wkset[altblock]->getSolutionField(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                  ScalarT diff = sol_z(elem,pt).val() - tsol(elem,pt).val();
#else
                  ScalarT diff = sol_z(elem,pt) - tsol(elem,pt);
#endif
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
          }
        }
        if (have_face_errs) {
          for (size_t face=0; face<assembler->groups[block][cell]->groupData->numSides; face++) {
            // TMW - hard coded for now
            for (size_t set=0; set<assembler->wkset[altblock]->numSets; ++set) {
              assembler->wkset[altblock]->computeSolnSteadySeeded(set, assembler->groups[block][cell]->u[set], seedwhat);
            }
            //assembler->groups[block][cell]->computeSolnFaceIP(face);
            assembler->groups[block][cell]->updateWorksetFace(face);
            assembler->wkset[altblock]->resetSolutionFields();
            //assembler->groups[block][cell]->computeSolnFaceIP(face, seedwhat);
            for (size_t etype=0; etype<error_list[altblock].size(); etype++) {
              string varname = error_list[altblock][etype].first;
              if (error_list[altblock][etype].second == "L2 FACE") {
                // compute the true z-component of grad
                string expression = varname;
                auto tsol = functionManagers[altblock]->evaluate("true "+expression,"side ip");
                auto sol = assembler->wkset[altblock]->getSolutionField(expression+" side");
                auto wts = assembler->groups[block][cell]->wkset->wts_side;
                
                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)),
                                KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                  double facemeasure = 0.0;
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    facemeasure += wts(elem,pt);
                  }
                  
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                    ScalarT diff = sol(elem,pt).val() - tsol(elem,pt).val();
#else
                    ScalarT diff = sol(elem,pt) - tsol(elem,pt);
#endif
                    update += 0.5/facemeasure*diff*diff*wts(elem,pt);  // TODO - BWR what is this? why .5?
                  }
                }, error);
                blockerrors(etype) += error;
              }
            }
          }
        }
      }
      assembler->wkset[altblock]->isTransient = isTransient;
      assembler->groupData[altblock]->requiresTransient = isTransient;
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
    if (multiscale_manager->subgridModels.size() > 0) {
      // Collect all of the errors for each subgrid model
      vector<vector<Kokkos::View<ScalarT*,HostDevice> > > blocksgerrs;
      
      for (size_t block=0; block<assembler->groups.size(); block++) {// loop over blocks
        
        vector<Kokkos::View<ScalarT*,HostDevice> > sgerrs;
        for (size_t m=0; m<multiscale_manager->subgridModels.size(); m++) {
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
    for (size_t cell=0; cell<assembler->boundary_groups[block].size(); ++cell) {
      // setup workset for this bcell
      
      assembler->boundary_groups[block][cell]->updateWorkset(0,true);
      
      // compute the flux
      assembler->wkset[block]->flux = View_AD3("flux",assembler->wkset[block]->maxElem,
                                               assembler->wkset[block]->numVars[0], // hard coded
                                               assembler->wkset[block]->numsideip);
      
      assembler->groupData[block]->physics_RCP->computeFlux(0,block); // hard coded
      auto cflux = assembler->wkset[block]->flux; // View_AD3
      
      for (size_t f=0; f<fluxes.size(); ++f) {
        
        if (fluxes[f].block == block) {
          string sidename = assembler->boundary_groups[block][cell]->sidename;
          size_t found = fluxes[f].sidesets.find(sidename);
          
          if (found!=std::string::npos) {
            
            auto wts = functionManagers[block]->evaluate("flux weight "+fluxes[f].name,"side ip");
            auto iwts = assembler->wkset[block]->wts_side;
            
            for (size_type v=0; v<fluxes[f].vals.extent(0); ++v) {
              ScalarT value = 0.0;
              auto vflux = subview(cflux,ALL(),v,ALL());
              parallel_reduce(RangePolicy<AssemblyExec>(0,iwts.extent(0)),
                              KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<iwts.extent(1); pt++ ) {
#ifndef MrHyDE_NO_AD
                  ScalarT up = vflux(elem,pt).val()*wts(elem,pt).val()*iwts(elem,pt);
#else
                  ScalarT up = vflux(elem,pt)*wts(elem,pt)*iwts(elem,pt);
#endif
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

        for (size_t cell=0; cell<assembler->groups[globalBlock].size(); ++cell) {

          localContribution = 0.; // zero out this cell's contribution JIC here but needed below

          // setup the workset for this cell
          assembler->groups[globalBlock][cell]->updateWorkset(0,true);
          // get integration weights
          auto wts = assembler->wkset[globalBlock]->wts;
          // evaluate the integrand at integration points
          auto integrand = functionManagers[globalBlock]->evaluate(
                integratedQuantities[iLocal][iIQ].name+" integrand","ip");

          // expand this for integral integrands, etc.?
          
          parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)),
                          KOKKOS_LAMBDA (const int elem, ScalarT & update) {
                            for (size_t pt=0; pt<wts.extent(1); pt++) {
#ifndef MrHyDE_NO_AD
                              ScalarT Idx = wts(elem,pt)*integrand(elem,pt).val();
#else
                              ScalarT Idx = wts(elem,pt)*integrand(elem,pt);
#endif
                              update += Idx;
                            }
                          },localContribution); //// TODO :: may be illegal

          // add this cell's contribution to running total

          integral += localContribution;

        } // end loop over groups

      } else if (integratedQuantities[iLocal][iIQ].location == "boundary") {

        for (size_t cell=0; cell<assembler->boundary_groups[globalBlock].size(); ++cell) {

          localContribution = 0.; // zero out this cell's contribution

          // check if we are on one of the requested sides
          string sidename = assembler->boundary_groups[globalBlock][cell]->sidename;
          size_t found = integratedQuantities[iLocal][iIQ].boundarynames.find(sidename);

          if ( (found!=std::string::npos) || 
               (integratedQuantities[iLocal][iIQ].boundarynames == "all") ) {

            // setup the workset for this cell
            assembler->boundary_groups[globalBlock][cell]->updateWorkset(0,true); 
            // get integration weights
            auto wts = assembler->wkset[globalBlock]->wts_side;
            // evaluate the integrand at integration points
            auto integrand = functionManagers[globalBlock]->evaluate(
                  integratedQuantities[iLocal][iIQ].name+" integrand","side ip");

            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)),
                            KOKKOS_LAMBDA (const int elem, ScalarT & update) {
                              for (size_t pt=0; pt<wts.extent(1); pt++) {
#ifndef MrHyDE_NO_AD
                                ScalarT Idx = wts(elem,pt)*integrand(elem,pt).val();
#else
                                ScalarT Idx = wts(elem,pt)*integrand(elem,pt);
#endif
                                update += Idx;
                              }
                            },localContribution); //// TODO :: may be illegal, problematic ABOVE TOO

          } // end if requested side
          // add in this cell's contribution to running total
          integral += localContribution;
        } // end loop over boundary groups
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
      for (size_t set=0; set<phys->modules.size(); ++set) {
        for (size_t m=0; m<phys->modules[set][globalBlock].size(); ++m) {
          // BWR -- called for all physics defined on the block regards of if they need IQs
          phys->modules[set][globalBlock][m]->updateIntegratedQuantitiesDependents();
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

template<class Node>
void PostprocessManager<Node>::computeObjective(vector<vector_RCP> & current_soln,
                                                const ScalarT & current_time,
                                                DFAD & objectiveval) {
  
  Teuchos::TimeMonitor localtimer(*objectiveTimer);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting PostprocessManager::computeObjective ..." << std::endl;
    }
  }
  
  // Objective function values
  vector<ScalarT> totaldiff(objectives.size(), 0.0);
  
  int numParams = params->num_active_params + params->globalParamUnknowns;
  
  // Objective function gradients w.r.t params
  vector<vector<ScalarT> > gradients;
  for (size_t r=0; r<objectives.size(); ++r) {
    vector<ScalarT> rgrad(numParams,0.0);
    gradients.push_back(rgrad);
  }
  
  for (size_t r=0; r<objectives.size(); ++r) {
    if (objectives[r].type == "integrated control"){
      
      // First, compute objective value and deriv. w.r.t scalar params
      params->sacadoizeParams(true);
      size_t block = objectives[r].block;
      
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
        auto wts = assembler->groups[block][grp]->wts;
        
        assembler->groups[block][grp]->updateWorkset(0,true);
        
        auto obj_dev = functionManagers[block]->evaluate(objectives[r].name,"ip");
        
        Kokkos::View<AD[1],AssemblyDevice> objsum("sum of objective");
        parallel_for("cell objective",
                     RangePolicy<AssemblyExec>(0,wts.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          AD tmpval = 0.0;
          for (size_type pt=0; pt<wts.extent(1); pt++) {
            tmpval += obj_dev(elem,pt)*wts(elem,pt);
          }
          Kokkos::atomic_add(&(objsum(0)),tmpval);
        });
        
        View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
        
        parallel_for("cell objective",
                     RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                     KOKKOS_LAMBDA (const size_type p ) {
#ifndef MrHyDE_NO_AD
          size_t numder = static_cast<size_t>(objsum(0).size());
#else
          size_t numder = 0;
#endif
          if (p==0) {
#ifndef MrHyDE_NO_AD
            objsum_dev(p) = objsum(0).val();
#else
            objsum_dev(p) = objsum(0);
#endif
          }
          else if (p <= numder) {
#ifndef MrHyDE_NO_AD
            objsum_dev(p) = objsum(0).fastAccessDx(p-1);
#endif
          }
        });
        
        auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
        Kokkos::deep_copy(objsum_host,objsum_dev);
        
        // Update the objective function value
        totaldiff[r] += objectives[r].weight*objsum_host(0);
        
        // Update the gradients w.r.t scalar active parameters
        for (int p=0; p<params->num_active_params; p++) {
          gradients[r][p] += objectives[r].weight*objsum_host(p+1);
        }
      }
      
      
      // Next, deriv w.r.t discretized params
      if (params->globalParamUnknowns > 0) {
        
        params->sacadoizeParams(false);
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          
          auto wts = assembler->groups[block][grp]->wts;
          
          assembler->groups[block][grp]->updateWorkset(3,true);
          
          auto obj_dev = functionManagers[block]->evaluate(objectives[r].name,"ip");
          
          Kokkos::View<AD[1],AssemblyDevice> objsum("sum of objective");
          parallel_for("cell objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            AD tmpval = 0.0;
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              tmpval += obj_dev(elem,pt)*wts(elem,pt);
            }
            Kokkos::atomic_add(&(objsum(0)),tmpval);
          });
          
          View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
          
          parallel_for("cell objective",
                       RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                       KOKKOS_LAMBDA (const size_type p ) {
#ifndef MrHyDE_NO_AD
            size_t numder = static_cast<size_t>(objsum(0).size());
#else
            size_t numder = 0;
#endif
            if (p==0) {
#ifndef MrHyDE_NO_AD
              objsum_dev(p) = objsum(0).val();
#else
              objsum_dev(p) = objsum(0);
#endif
            }
            else if (p <= numder) {
#ifndef MrHyDE_NO_AD
              objsum_dev(p) = objsum(0).fastAccessDx(p-1);
#endif
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
                gradients[r][rowIndex+params->num_active_params] += objectives[r].weight*objsum_host(poffset);
              }
            }
          }
        }
        
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
      
      // First, compute objective value and deriv. w.r.t scalar params
      params->sacadoizeParams(true);
      size_t block = objectives[r].block;
      
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
        auto wts = assembler->groups[block][grp]->wts;
            
        assembler->groups[block][grp]->updateWorkset(0,true);
        
        auto obj_dev = functionManagers[block]->evaluate(objectives[r].name+" response","ip");
        
        Kokkos::View<AD[1],AssemblyDevice> objsum("sum of objective");
        parallel_for("cell objective",
                     RangePolicy<AssemblyExec>(0,wts.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          AD tmpval = 0.0;
          for (size_type pt=0; pt<wts.extent(1); pt++) {
            tmpval += obj_dev(elem,pt)*wts(elem,pt);
          }
          Kokkos::atomic_add(&(objsum(0)),tmpval);
        });
        
        View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
        
        parallel_for("cell objective",
                     RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                     KOKKOS_LAMBDA (const size_type p ) {
#ifndef MrHyDE_NO_AD
          size_t numder = static_cast<size_t>(objsum(0).size());
          if (p==0) {
            objsum_dev(p) = objsum(0).val();
          }
          else if (p <= numder) {
            objsum_dev(p) = objsum(0).fastAccessDx(p-1);
          }
#else
          if (p==0) {
            objsum_dev(p) = objsum(0);
          }
#endif
        });
        
        auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
        Kokkos::deep_copy(objsum_host,objsum_dev);
        
        
        
        // Update the objective function value
        totaldiff[r] += objsum_host(0);
        
        // Update the gradients w.r.t scalar active parameters
        for (int p=0; p<params->num_active_params; p++) {
          gradients[r][p] += objsum_host(p+1);
        }
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
      
      // Next, deriv w.r.t discretized params
      if (params->globalParamUnknowns > 0) {
        
        params->sacadoizeParams(false);
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          
          auto wts = assembler->groups[block][grp]->wts;
          
          assembler->groups[block][grp]->updateWorkset(3,true);
          
          auto obj_dev = functionManagers[block]->evaluate(objectives[r].name,"ip");
          
          Kokkos::View<AD[1],AssemblyDevice> objsum("sum of objective");
          parallel_for("cell objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            AD tmpval = 0.0;
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              tmpval += obj_dev(elem,pt)*wts(elem,pt);
            }
            Kokkos::atomic_add(&(objsum(0)),tmpval);
          });
          
          View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
          
          parallel_for("cell objective",
                       RangePolicy<AssemblyExec>(0,objsum_dev.extent(0)),
                       KOKKOS_LAMBDA (const size_type p ) {
#ifndef MrHyDE_NO_AD
            size_t numder = static_cast<size_t>(objsum(0).size());
            if (p==0) {
              objsum_dev(p) = objsum(0).val();
            }
            else if (p <= numder) {
              objsum_dev(p) = objsum(0).fastAccessDx(p-1);
            }
#else
            if (p==0) {
              objsum_dev(p) = objsum(0);
            }
#endif
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
                gradients[r][rowIndex+params->num_active_params] += objsum_host(poffset);
              }
            }
          }
        }
        
      }
      
      // Right now, totaldiff = response
      //             gradient = dresponse / dp
      // We want    totaldiff = wt*(response-target)^2
      //             gradient = 2*wt*(response-target)*dresponse/dp
      
      ScalarT diff = totaldiff[r] - objectives[r].target;
      totaldiff[r] = objectives[r].weight*diff*diff;
      for (size_t g=0; g<gradients[r].size(); ++g) {
        gradients[r][g] = 2.0*objectives[r].weight*diff*gradients[r][g];
      }
      
      
    }
    else if (objectives[r].type == "sensors" || objectives[r].type == "sensor response" || objectives[r].type == "pointwise response") {
      
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
        
          // First compute objective and derivative w.r.t scalar params
          params->sacadoizeParams(true);
          
          size_t block = objectives[r].block;
          size_t cell = objectives[r].sensor_owners(pt,0);
          size_t elem = objectives[r].sensor_owners(pt,1);
          
          auto x = assembler->wkset[block]->getScalarField("x point");
          x(0,0) = objectives[r].sensor_points(pt,0);
          if (spaceDim > 1) {
            auto y = assembler->wkset[block]->getScalarField("y point");
            y(0,0) = objectives[r].sensor_points(pt,1);
          }
          if (spaceDim > 2) {
            auto z = assembler->wkset[block]->getScalarField("z point");
            z(0,0) = objectives[r].sensor_points(pt,2);
          }
          
          auto numDOF = assembler->groupData[block]->numDOF;
          View_AD2 u_dof("u_dof",numDOF.extent(0),assembler->groups[block][cell]->LIDs[0].extent(1)); // hard coded
          auto cu = subview(assembler->groups[block][cell]->u[0],elem,ALL(),ALL()); // hard coded
          parallel_for("cell response get u",
                       RangePolicy<AssemblyExec>(0,u_dof.extent(0)),
                       KOKKOS_LAMBDA (const size_type n ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for( int i=0; i<numDOF(n); i++ ) {
                u_dof(n,i) = cu(n,i);
              }
            }
          });
          
          // Map the local solution to the solution and gradient at ip
          View_AD2 u_ip("u_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
          View_AD2 ugrad_ip("ugrad_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
          
          for (size_type var=0; var<numDOF.extent(0); var++) {
            auto cbasis = objectives[r].sensor_basis[pt][assembler->wkset[block]->usebasis[var]];
            auto cbasis_grad = objectives[r].sensor_basis_grad[pt][assembler->wkset[block]->usebasis[var]];
            auto u_sv = subview(u_ip, var, ALL());
            auto u_dof_sv = subview(u_dof, var, ALL());
            auto ugrad_sv = subview(ugrad_ip, var, ALL());
            
            parallel_for("cell response sensor uip",
                         RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                         KOKKOS_LAMBDA (const int dof ) {
              u_sv(0) += u_dof_sv(dof)*cbasis(0,dof,0,0);
              for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                ugrad_sv(dim) += u_dof_sv(dof)*cbasis_grad(0,dof,0,dim);
              }
            });
          }
          
          assembler->wkset[block]->setSolutionPoint(u_ip);
          assembler->wkset[block]->setSolutionGradPoint(ugrad_ip);
          
          // Map the local discretized params to param and grad at ip
          if (params->globalParamUnknowns > 0) {
            auto numParamDOF = assembler->groupData[block]->numParamDOF;
            
            View_AD2 p_dof("p_dof",numParamDOF.extent(0),assembler->groups[block][cell]->paramLIDs.extent(1));
            auto cp = subview(assembler->groups[block][cell]->param,elem,ALL(),ALL());
            parallel_for("cell response get u",
                         RangePolicy<AssemblyExec>(0,p_dof.extent(0)),
                         KOKKOS_LAMBDA (const size_type n ) {
              for (size_type n=0; n<numParamDOF.extent(0); n++) {
                for( int i=0; i<numParamDOF(n); i++ ) {
                  p_dof(n,i) = cp(n,i);
                }
              }
            });
            
            View_AD2 p_ip("p_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
            View_AD2 pgrad_ip("pgrad_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
            
            for (size_type var=0; var<numParamDOF.extent(0); var++) {
              int bnum = assembler->wkset[block]->paramusebasis[var];
              auto cbasis = objectives[r].sensor_basis[pt][bnum];
              auto cbasis_grad = objectives[r].sensor_basis_grad[pt][bnum];
              auto p_sv = subview(p_ip, var, ALL());
              auto p_dof_sv = subview(p_dof, var, ALL());
              auto pgrad_sv = subview(pgrad_ip, var, ALL());
              
              parallel_for("cell response sensor uip",
                           RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                           KOKKOS_LAMBDA (const int dof ) {
                p_sv(0) += p_dof_sv(dof)*cbasis(0,dof,0,0);
                for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                  pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(0,dof,0,dim);
                }
              });
            }
            
            assembler->wkset[block]->setParamPoint(p_ip);
            
            assembler->wkset[block]->setParamGradPoint(pgrad_ip);
          }
          
          // Evaluate the response
          auto rdata = functionManagers[block]->evaluate(objectives[r].name+" response","point");
          
          if (compute_response) {
#ifndef MrHyDE_NO_AD
            sensordat(pt) = rdata(0,0).val();
#else
            sensordat(pt) = rdata(0,0);
#endif
          }
          
          if (compute_objective) {
            
            // Update the value of the objective
            AD diff = rdata(0,0) - objectives[r].sensor_data(pt,tindex);
            AD sdiff = objectives[r].weight*diff*diff;
#ifndef MrHyDE_NO_AD
            totaldiff[r] += sdiff.val();
#else
            totaldiff[r] += sdiff;
#endif
            
            // Update the gradient w.r.t scalar active parameters
#ifndef MrHyDE_NO_AD
            for (int p=0; p<params->num_active_params; p++) {
              gradients[r][p] += sdiff.fastAccessDx(p);
            }
#endif
            
            // Discretized parameters
            if (params->globalParamUnknowns > 0) {
              
              // Need to compute derivative w.r.t discretized params
              params->sacadoizeParams(false);
              
              auto numParamDOF = assembler->groupData[block]->numParamDOF;
              auto poff = assembler->wkset[block]->paramoffsets;
              View_AD2 p_dof("p_dof",numParamDOF.extent(0),assembler->groups[block][cell]->paramLIDs.extent(1));
              auto cp = subview(assembler->groups[block][cell]->param,elem,ALL(),ALL());
              parallel_for("cell response get u",
                           RangePolicy<AssemblyExec>(0,p_dof.extent(0)),
                           KOKKOS_LAMBDA (const size_type n ) {
                for (size_type n=0; n<numParamDOF.extent(0); n++) {
                  for( int i=0; i<numParamDOF(n); i++ ) {
#ifndef MrHyDE_NO_AD
                    p_dof(n,i) = AD(maxDerivs,poff(n,i),cp(n,i));
#else
                    p_dof(n,i) = cp(n,i);
#endif
                  }
                }
              });
              
#ifndef MrHyDE_NO_AD
              View_AD2 p_ip("p_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
              View_AD2 pgrad_ip("pgrad_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
              
              for (size_type var=0; var<numParamDOF.extent(0); var++) {
                int bnum = assembler->wkset[block]->paramusebasis[var];
                auto cbasis = objectives[r].sensor_basis[pt][bnum];
                auto cbasis_grad = objectives[r].sensor_basis_grad[pt][bnum];
                auto p_sv = subview(p_ip, var, ALL());
                auto p_dof_sv = subview(p_dof, var, ALL());
                auto pgrad_sv = subview(pgrad_ip, var, ALL());
                
                parallel_for("cell response sensor uip",
                             RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                             KOKKOS_LAMBDA (const int dof ) {
                  p_sv(0) += p_dof_sv(dof)*cbasis(0,dof,0,0);
                  for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                    pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(0,dof,0,dim);
                  }
                });
              }
              
              assembler->wkset[block]->setParamPoint(p_ip);
              assembler->wkset[block]->setParamGradPoint(pgrad_ip);
              
              // Evaluate the response
              auto rdata = functionManagers[block]->evaluate(objectives[r].name+" response","point");
              AD diff = rdata(0,0) - objectives[r].sensor_data(pt,tindex);
              AD sdiff = objectives[r].weight*diff*diff;
              
              auto poffs = params->paramoffsets;
              vector<GO> paramGIDs;
              params->paramDOF->getElementGIDs(assembler->groups[block][cell]->localElemID[elem],
                                               paramGIDs, blocknames[block]);
              
              for (size_t pp=0; pp<poffs.size(); ++pp) {
                for (size_t row=0; row<poffs[pp].size(); row++) {
                  GO rowIndex = paramGIDs[poffs[pp][row]] + params->num_active_params;
                  int poffset = poffs[pp][row];
                  gradients[r][rowIndex] += sdiff.fastAccessDx(poffset);
                }
              }
#endif
            }
            
          }
        } // found time
      } // sensor points
      
      if (compute_response) {
        objectives[r].response_data.push_back(sensordat);
      }
    } // objectives
    
    // ========================================================================================
    // Add regularizations (reg funcs are tied to objectives and objectives can have more than one reg)
    // ========================================================================================

    for (size_t reg=0; reg<objectives[r].regularizations.size(); ++reg) {
      if (objectives[r].regularizations[reg].type == "integrated") {
        if (objectives[r].regularizations[reg].location == "volume") {
          params->sacadoizeParams(false);
          ScalarT regwt = objectives[r].regularizations[reg].weight;
          size_t block = objectives[r].block;
          for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
            
            auto wts = assembler->groups[block][grp]->wts;
            
            assembler->groups[block][grp]->updateWorkset(3,true);
            
            auto regvals_tmp = functionManagers[block]->evaluate(objectives[r].regularizations[reg].name,"ip");
            View_AD2 regvals("regvals",wts.extent(0),wts.extent(1));
            
            parallel_for("cell objective",
                         RangePolicy<AssemblyExec>(0,wts.extent(0)),
                         KOKKOS_LAMBDA (const size_type elem ) {
              for (size_type pt=0; pt<wts.extent(1); ++pt) {
                regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
              }
            });
            
            
            View_Sc3 regvals_sc("scalar version of AD view",wts.extent(0),wts.extent(1),maxDerivs+1);
            parallel_for("cell objective",
                         RangePolicy<AssemblyExec>(0,wts.extent(0)),
                         KOKKOS_LAMBDA (const size_type elem ) {
              for (size_type pt=0; pt<wts.extent(1); ++pt) {
#ifndef MrHyDE_NO_AD
                regvals_sc(elem,pt,0) = regvals(elem,pt).val();
                for (size_type d=0; d<regvals_sc.extent(2)-1; ++d) {
                  regvals_sc(elem,pt,d+1) = regvals(elem,pt).fastAccessDx(d);
                }
#else
                regvals_sc(elem,pt,0) = regvals(elem,pt);
#endif
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
                totaldiff[r] += regwt*regvals_sc_host(elem,pt,0);
                for (size_t pp=0; pp<poffs.size(); ++pp) {
                  for (size_t row=0; row<poffs[pp].size(); row++) {
                    GO rowIndex = paramGIDs[poffs[pp][row]] + params->num_active_params;
                    int poffset = poffs[pp][row];
                    gradients[r][rowIndex] += regwt*regvals_sc_host(elem,pt,poffset+1);
                  }
                }
              }
            }
          }
          
        }
        else if (objectives[r].regularizations[reg].location == "boundary") {
          string bname = objectives[r].regularizations[reg].boundary_name;
          params->sacadoizeParams(false);
          ScalarT regwt = objectives[r].regularizations[reg].weight;
          size_t block = objectives[r].block;
          for (size_t grp=0; grp<assembler->boundary_groups[block].size(); ++grp) {
            if (assembler->boundary_groups[block][grp]->sidename == bname) {
              
              auto wts = assembler->boundary_groups[block][grp]->wts;
              
              assembler->boundary_groups[block][grp]->updateWorkset(3,true);
              
              auto regvals_tmp = functionManagers[block]->evaluate(objectives[r].regularizations[reg].name,"side ip");
              View_AD2 regvals("regvals",wts.extent(0),wts.extent(1));
              
              parallel_for("cell objective",
                           RangePolicy<AssemblyExec>(0,wts.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                for (size_type pt=0; pt<wts.extent(1); ++pt) {
                  regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
                }
              });
              
              View_Sc3 regvals_sc("scalar version of AD view",wts.extent(0),wts.extent(1),maxDerivs+1);
              parallel_for("cell objective",
                           RangePolicy<AssemblyExec>(0,wts.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                for (size_type pt=0; pt<wts.extent(1); ++pt) {
#ifndef MrHyDE_NO_AD
                  regvals_sc(elem,pt,0) = regvals(elem,pt).val();
                  for (size_type d=0; d<regvals_sc.extent(2)-1; ++d) {
                    regvals_sc(elem,pt,d+1) = regvals(elem,pt).fastAccessDx(d);
                  }
#else
                  regvals_sc(elem,pt,0) = regvals(elem,pt);
#endif
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
                  totaldiff[r] += regwt*regvals_sc_host(elem,pt,0);
                  for (size_t pp=0; pp<poffs.size(); ++pp) {
                    for (size_t row=0; row<poffs[pp].size(); row++) {
                      GO rowIndex = paramGIDs[poffs[pp][row]] + params->num_active_params;
                      int poffset = poffs[pp][row];
                      gradients[r][rowIndex] += regwt*regvals_sc_host(elem,pt,poffset+1);
                    }
                  }
                }
              }
              
              /*
              auto obj_dev = functionManagers[block]->evaluate(objectives[r].regularizations[reg].name,"side ip");
              
              Kokkos::View<AD[1],AssemblyDevice> objsum("sum of objective");
              parallel_for("cell objective",
                           RangePolicy<AssemblyExec>(0,obj_dev.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                AD tmpval = 0.0;
                for (size_type pt=0; pt<obj_dev.extent(1); pt++) {
                  tmpval += obj_dev(elem,pt)*wts(elem,pt);
                }
                Kokkos::atomic_add(&(objsum(0)),tmpval);
              });
              
              View_Sc1 objsum_dev("obj func sum as scalar on device",numParams+1);
              
              parallel_for("cell objective",
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
  
  for (int j=0; j<numParams; j++) {
    ScalarT dval = 0.0;
    ScalarT ldval = 0.0;
    for (size_t r=0; r<gradients.size(); ++r) {
      ldval += gradients[r][j];
    }
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&ldval,&dval);
    fullobj.fastAccessDx(j) = dval;
  }
  
  params->sacadoizeParams(false);
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Finished PostprocessManager::computeObjective ..." << std::endl;
    }
  }
  
  objectiveval += fullobj;
  
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
  DFAD totaldiff = 0.0;
  //AD regDomain = 0.0;
  //AD regBoundary = 0.0;
  
  params->sacadoizeParams(false);
  
  int numParams = params->num_active_params + params->globalParamUnknowns;
  
  
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
    
  for (size_t r=0; r<objectives.size(); ++r) {
    if (objectives[r].type == "integrated control"){
      auto grad_over = linalg->getNewOverlappedVector(set);
      auto grad_tmp = linalg->getNewVector(set);
      auto grad_view = grad_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      size_t block = objectives[r].block;
      
      auto offsets = assembler->wkset[block]->offsets;
      auto numDOF = assembler->groupData[block]->numDOF;
      
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
        size_t numElem = assembler->groups[block][grp]->numElem;
        size_t numip = assembler->wkset[block]->numip;
        
        View_Sc3 local_grad("local contrib to dobj/dstate",
                            assembler->groups[block][grp]->numElem,
                            assembler->groups[block][grp]->LIDs[set].extent(1),1);
        
        auto local_grad_ladev = create_mirror(LA_exec(),local_grad);
        
        for (int w=0; w<spaceDim+1; ++w) {
          
          // Seed the state and compute the solution at the ip
          if (w==0) {
            assembler->groups[block][grp]->updateWorkset(1,true);
          }
          else {
            View_AD3 u_dof("u_dof",numElem,numDOF.extent(0),
                           assembler->groups[block][grp]->LIDs[set].extent(1)); //(numElem, numVars, numDOF)
            auto u = assembler->groups[block][grp]->u[set];
            parallel_for("cell response get u",
                         RangePolicy<AssemblyExec>(0,u_dof.extent(0)),
                         KOKKOS_LAMBDA (const size_type e ) {
              for (size_type n=0; n<numDOF.extent(0); n++) { // numDOF is on device
                for( int i=0; i<numDOF(n); i++ ) {
                  u_dof(e,n,i) = AD(maxDerivs,offsets(n,i),u(e,n,i)); // offsets is on device
                }
              }
            });
            
            View_AD4 u_ip("u_ip",numElem,numDOF.extent(0),numip,spaceDim);
            View_AD4 ugrad_ip("ugrad_ip",numElem,numDOF.extent(0),numip,spaceDim);
            
            for (size_type var=0; var<numDOF.extent(0); var++) {
              int bnum = assembler->wkset[block]->usebasis[var];
              std::string btype = assembler->wkset[block]->basis_types[bnum];
              if (btype == "HCURL" || btype == "HDIV") {
                // TMW: this does not work yet
              }
              else {
                auto cbasis = assembler->wkset[block]->basis[bnum];
                
                auto u_sv = subview(u_ip, ALL(), var, ALL(), 0);
                auto u_dof_sv = subview(u_dof, ALL(), var, ALL());
                parallel_for("cell response uip",
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
                auto cbasis_grad = assembler->wkset[block]->basis_grad[bnum];
                auto u_dof_sv = subview(u_dof, ALL(), var, ALL());
                auto ugrad_sv = subview(ugrad_ip, ALL(), var, ALL(), ALL());
                parallel_for("cell response HGRAD",
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
            
            for (int s=0; s<spaceDim; s++) {
              auto ugrad_sv = subview(ugrad_ip, ALL(), ALL(), ALL(), s);
              if ((w-1) == s) {
                parallel_for("cell response seed grad 0",
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
                parallel_for("cell response seed grad 1",
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
            parallel_for("cell response seed grad 2",
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
            assembler->wkset[block]->setSolution(u_ip);
            assembler->wkset[block]->setSolutionGrad(ugrad_ip);
            
          }
          
          // Evaluate the objective
          auto obj_dev = functionManagers[block]->evaluate(objectives[r].name,"ip");
          
          // Weight using volumetric integration weights
          auto wts = assembler->groups[block][grp]->wts;
          
          parallel_for("cell objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              obj_dev(elem,pt) *= objectives[r].weight*wts(elem,pt);
            }
          });
          
          for (size_type n=0; n<numDOF.extent(0); n++) {
            int bnum = assembler->wkset[block]->usebasis[n];
            std::string btype = assembler->wkset[block]->basis_types[bnum];
            
            if (w == 0) {
              auto cbasis = assembler->wkset[block]->basis[bnum];
              
              if (btype == "HDIV" || btype == "HCURL") {
                parallel_for("cell adjust adjoint res",
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
                parallel_for("cell adjust adjoint res",
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
                auto cbasis = assembler->wkset[block]->basis_grad[bnum];
                auto cbasis_sv = subview(cbasis, ALL(), ALL(), ALL(), w-1);
                parallel_for("cell adjust adjoint res",
                             RangePolicy<AssemblyExec>(0,local_grad.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(e,offsets(nn,j),0) += -obj_dev(e,s).fastAccessDx(offsets(nn,i))*cbasis_sv(e,j,s);
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
    else if (objectives[r].type == "integrated response") {
      auto grad_over = linalg->getNewOverlappedVector(set);
      auto grad_tmp = linalg->getNewVector(set);
      auto grad_view = grad_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      size_t block = objectives[r].block;
      
      auto offsets = assembler->wkset[block]->offsets;
      auto numDOF = assembler->groupData[block]->numDOF;
      
      ScalarT intresp = 0.0;
      for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
        
        size_t numElem = assembler->groups[block][grp]->numElem;
        size_t numip = assembler->wkset[block]->numip;
        
        View_Sc3 local_grad("local contrib to dobj/dstate",
                            assembler->groups[block][grp]->numElem,
                            assembler->groups[block][grp]->LIDs[set].extent(1),1);
        
        auto local_grad_ladev = create_mirror(LA_exec(),local_grad);
                
        for (int w=0; w<spaceDim+1; ++w) {
          
          // Seed the state and compute the solution at the ip
          if (w==0) {
            assembler->groups[block][grp]->updateWorkset(1,true);
          }
          else {
            View_AD3 u_dof("u_dof",numElem,numDOF.extent(0),
                           assembler->groups[block][grp]->LIDs[set].extent(1)); //(numElem, numVars, numDOF)
            auto u = assembler->groups[block][grp]->u[set];
            parallel_for("cell response get u",
                         RangePolicy<AssemblyExec>(0,u_dof.extent(0)),
                         KOKKOS_LAMBDA (const size_type e ) {
              for (size_type n=0; n<numDOF.extent(0); n++) { // numDOF is on device
                for( int i=0; i<numDOF(n); i++ ) {
                  u_dof(e,n,i) = AD(maxDerivs,offsets(n,i),u(e,n,i)); // offsets is on device
                }
              }
            });
            
            View_AD4 u_ip("u_ip",numElem,numDOF.extent(0),numip,spaceDim);
            View_AD4 ugrad_ip("ugrad_ip",numElem,numDOF.extent(0),numip,spaceDim);
            
            for (size_type var=0; var<numDOF.extent(0); var++) {
              int bnum = assembler->wkset[block]->usebasis[var];
              std::string btype = assembler->wkset[block]->basis_types[bnum];
              if (btype == "HCURL" || btype == "HDIV") {
                // TMW: this does not work yet
              }
              else {
                auto cbasis = assembler->wkset[block]->basis[bnum];
                
                auto u_sv = subview(u_ip, ALL(), var, ALL(), 0);
                auto u_dof_sv = subview(u_dof, ALL(), var, ALL());
                parallel_for("cell response uip",
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
                auto cbasis_grad = assembler->wkset[block]->basis_grad[bnum];
                auto u_dof_sv = subview(u_dof, ALL(), var, ALL());
                auto ugrad_sv = subview(ugrad_ip, ALL(), var, ALL(), ALL());
                parallel_for("cell response HGRAD",
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
            
            for (int s=0; s<spaceDim; s++) {
              auto ugrad_sv = subview(ugrad_ip, ALL(), ALL(), ALL(), s);
              if ((w-1) == s) {
                parallel_for("cell response seed grad 0",
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
                parallel_for("cell response seed grad 1",
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
            parallel_for("cell response seed grad 2",
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
            assembler->wkset[block]->setSolution(u_ip);
            assembler->wkset[block]->setSolutionGrad(ugrad_ip);
            
          }
          
          // Evaluate the objective
          auto obj_dev = functionManagers[block]->evaluate(objectives[r].name+" response","ip");
          
          // Weight using volumetric integration weights
          auto wts = assembler->groups[block][grp]->wts;
          
          parallel_for("cell objective",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              //obj_dev(elem,pt) *= objectives[r].weight*wts(elem,pt);
              obj_dev(elem,pt) *= wts(elem,pt);
            }
          });
          
          Kokkos::View<ScalarT[1],AssemblyDevice> ir("integral of response");
          parallel_for("cell objective",
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
          
          for (size_type n=0; n<numDOF.extent(0); n++) {
            int bnum = assembler->wkset[block]->usebasis[n];
            std::string btype = assembler->wkset[block]->basis_types[bnum];
            
            if (w == 0) {
              auto cbasis = assembler->wkset[block]->basis[bnum];
              
              if (btype == "HDIV" || btype == "HCURL") {
                parallel_for("cell adjust adjoint res",
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
                parallel_for("cell adjust adjoint res",
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
                auto cbasis = assembler->wkset[block]->basis_grad[bnum];
                auto cbasis_sv = subview(cbasis, ALL(), ALL(), ALL(), w-1);
                parallel_for("cell adjust adjoint res",
                             RangePolicy<AssemblyExec>(0,local_grad.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(e,offsets(nn,j),0) += -obj_dev(e,s).fastAccessDx(offsets(nn,i))*cbasis_sv(e,j,s);
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
      grad_over->scale(2.0*objectives[r].weight*(intresp - objectives[r].target));
      
      linalg->exportVectorFromOverlapped(set, grad_tmp, grad_over);
      grad->update(1.0, *grad_tmp, 1.0);
      
    }
    else if (objectives[r].type == "discrete control") {
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
          grad->update(-2.0*objectives[r].weight,*diff,1.0);
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: did not find a data-generating solution");
        }
      //}
    }
    else if (objectives[r].type == "sensors") {
      
      auto grad_over = linalg->getNewOverlappedVector(set);
      auto grad_tmp = linalg->getNewVector(set);
      auto grad_view = grad_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      
      for (size_t pt=0; pt<objectives[r].numSensors; ++pt) {
        size_t tindex = 0;
        bool foundtime = false;
        for (size_type t=0; t<objectives[r].sensor_times.extent(0); ++t) {
          if (std::abs(current_time - objectives[r].sensor_times(t)) < 1.0e-12) {
            foundtime = true;
            tindex = t;
          }
        }
        
        if (foundtime) {
        
          size_t block = objectives[r].block;
          size_t cell = objectives[r].sensor_owners(pt,0);
          size_t elem = objectives[r].sensor_owners(pt,1);
          
          auto x = assembler->wkset[block]->getScalarField("x point");
          x(0,0) = objectives[r].sensor_points(pt,0);
          if (spaceDim > 1) {
            auto y = assembler->wkset[block]->getScalarField("y point");
            y(0,0) = objectives[r].sensor_points(pt,1);
          }
          if (spaceDim > 2) {
            auto z = assembler->wkset[block]->getScalarField("z point");
            z(0,0) = objectives[r].sensor_points(pt,2);
          }
          
          auto numDOF = assembler->groupData[block]->numDOF;
          auto offsets = assembler->wkset[block]->offsets;
          
          
          View_AD2 u_dof("u_dof",numDOF.extent(0),assembler->groups[block][cell]->LIDs[set].extent(1));
          auto cu = subview(assembler->groups[block][cell]->u[set],elem,ALL(),ALL());
          parallel_for("cell response get u",
                       RangePolicy<AssemblyExec>(0,u_dof.extent(0)),
                       KOKKOS_LAMBDA (const size_type n ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for( int i=0; i<numDOF(n); i++ ) {
                u_dof(n,i) = AD(maxDerivs,offsets(n,i),cu(n,i));
              }
            }
          });
          
          // Map the local solution to the solution and gradient at ip
          View_AD2 u_ip("u_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
          View_AD2 ugrad_ip("ugrad_ip",numDOF.extent(0),assembler->groupData[block]->dimension);
          
          for (size_type var=0; var<numDOF.extent(0); var++) {
            auto cbasis = objectives[r].sensor_basis[pt][assembler->wkset[block]->usebasis[var]];
            auto cbasis_grad = objectives[r].sensor_basis_grad[pt][assembler->wkset[block]->usebasis[var]];
            auto u_sv = subview(u_ip, var, ALL());
            auto u_dof_sv = subview(u_dof, var, ALL());
            auto ugrad_sv = subview(ugrad_ip, var, ALL());
            
            parallel_for("cell response sensor uip",
                         RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                         KOKKOS_LAMBDA (const int dof ) {
              u_sv(0) += u_dof_sv(dof)*cbasis(0,dof,0,0);
              for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                ugrad_sv(dim) += u_dof_sv(dof)*cbasis_grad(0,dof,0,dim);
              }
            });
          }
          
          // Map the local discretized params to param and grad at ip
          if (params->globalParamUnknowns > 0) {
            auto numParamDOF = assembler->groupData[block]->numParamDOF;
            
            View_AD2 p_dof("p_dof",numParamDOF.extent(0),assembler->groups[block][cell]->paramLIDs.extent(1));
            auto cp = subview(assembler->groups[block][cell]->param,elem,ALL(),ALL());
            parallel_for("cell response get u",
                         RangePolicy<AssemblyExec>(0,p_dof.extent(0)),
                         KOKKOS_LAMBDA (const size_type n ) {
              for (size_type n=0; n<numParamDOF.extent(0); n++) {
                for( int i=0; i<numParamDOF(n); i++ ) {
                  p_dof(n,i) = cp(n,i);
                }
              }
            });
            
            View_AD2 p_ip("p_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
            View_AD2 pgrad_ip("pgrad_ip",numParamDOF.extent(0),assembler->groupData[block]->dimension);
            
            for (size_type var=0; var<numParamDOF.extent(0); var++) {
              int bnum = assembler->wkset[block]->paramusebasis[var];
              auto cbasis = objectives[r].sensor_basis[pt][bnum];
              auto cbasis_grad = objectives[r].sensor_basis_grad[pt][bnum];
              auto p_sv = subview(p_ip, var, ALL());
              auto p_dof_sv = subview(p_dof, var, ALL());
              auto pgrad_sv = subview(pgrad_ip, var, ALL());
              
              parallel_for("cell response sensor uip",
                           RangePolicy<AssemblyExec>(0,cbasis.extent(1)),
                           KOKKOS_LAMBDA (const int dof ) {
                p_sv(0) += p_dof_sv(dof)*cbasis(0,dof,0,0);
                for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                  pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(0,dof,0,dim);
                }
              });
            }
            
            assembler->wkset[block]->setParamPoint(p_ip);
            assembler->wkset[block]->setParamGradPoint(pgrad_ip);
          }
          
          
          View_Sc3 local_grad("local contrib to dobj/dstate",
                              assembler->groups[block][cell]->numElem,
                              assembler->groups[block][cell]->LIDs[set].extent(1),1);
          
          for (int w=0; w<spaceDim+1; ++w) {
            if (w==0) {
              assembler->wkset[block]->setSolutionPoint(u_ip);
              assembler->wkset[block]->setSolutionGradPoint(ugrad_ip);
            }
            else {
              View_AD2 u_tmp("u_tmp",numDOF.extent(0),assembler->groupData[block]->dimension);
              View_AD2 ugrad_tmp("ugrad_tmp",numDOF.extent(0),assembler->groupData[block]->dimension);
              deep_copy(u_tmp,u_ip);
              deep_copy(ugrad_tmp,ugrad_ip);
              
              for (int s=0; s<spaceDim; s++) {
                auto ugrad_sv = subview(ugrad_tmp, ALL(), s);
                if ((w-1) == s) {
                  parallel_for("cell response seed grad 0",
                               RangePolicy<AssemblyExec>(0,ugrad_tmp.extent(0)),
                               KOKKOS_LAMBDA (const size_type var ) {
                    ScalarT tmp = ugrad_sv(var).val();
                    ugrad_sv(var) = u_tmp(var,0);
                    ugrad_sv(var) += -u_tmp(var,0).val() + tmp;
                  });
                }
                else {
                  parallel_for("cell response seed grad 1",
                               RangePolicy<AssemblyExec>(0,ugrad_tmp.extent(0)),
                               KOKKOS_LAMBDA (const size_type var ) {
                    ugrad_sv(var) = ugrad_sv(var).val();
                  });
                }
                
              }
              parallel_for("cell response seed grad 2",
                           RangePolicy<AssemblyExec>(0,u_tmp.extent(0)),
                           KOKKOS_LAMBDA (const size_type var ) {
                for(size_type s=0; s<u_tmp.extent(1); s++ ) {
                  u_tmp(var,s) = u_tmp(var,s).val();
                }
              });
              
              assembler->wkset[block]->setSolutionPoint(u_tmp);
              assembler->wkset[block]->setSolutionGradPoint(ugrad_tmp);
              
            }

            auto rdata = functionManagers[block]->evaluate(objectives[r].name+" response","point");
            AD diff = rdata(0,0) - objectives[r].sensor_data(pt,tindex);
            AD totaldiff = objectives[r].weight*diff*diff;
            

            for (size_type n=0; n<numDOF.extent(0); n++) {
              int bnum = assembler->wkset[block]->usebasis[n];
              
              std::string btype = assembler->wkset[block]->basis_types[bnum];
              if (btype == "HDIV" || btype == "HCURL") {
                if (w==0) {
                  auto cbasis = objectives[r].sensor_basis[pt][bnum];
                  int nn = n; // TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        for (size_type d=0; d<cbasis.extent(3); d++) {
                          local_grad(elem,offsets(nn,j),0) += -totaldiff.fastAccessDx(offsets(nn,i))*cbasis(0,j,s,d);
                        }
                      }
                    }
                  }
                }
              }
              else {
                if (w==0) {
                  auto cbasis = objectives[r].sensor_basis[pt][bnum];
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(elem,offsets(nn,j),0) += -totaldiff.fastAccessDx(offsets(nn,i))*cbasis(0,j,s,0);
                      }
                    }
                  }
                }
                else {
                  auto cbasis = objectives[r].sensor_basis_grad[pt][bnum];
                  auto cbasis_sv = subview(cbasis,ALL(),ALL(),ALL(),w-1);
                  int nn = n; //TMW - temp
                  for (int j=0; j<numDOF(nn); j++) {
                    for (int i=0; i<numDOF(nn); i++) {
                      for (size_type s=0; s<cbasis.extent(2); s++) {
                        local_grad(elem,offsets(nn,j),0) += -totaldiff.fastAccessDx(offsets(nn,i))*cbasis_sv(0,j,s);
                      }
                    }
                  }
                }
              }
            }
          }
          
          assembler->scatterRes(grad_view, local_grad, assembler->groups[block][cell]->LIDs[set]);
          
        }
      }
      
      linalg->exportVectorFromOverlapped(set, grad_tmp, grad_over);
      grad->update(1.0, *grad_tmp, 1.0);
      
    }
  }
  
#endif
  
}


// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeSensitivities(vector<vector_RCP> & u,
                                                    vector<vector_RCP> & adjoint,
                                                    const ScalarT & current_time,
                                                    const ScalarT & deltat,
                                                    vector<ScalarT> & gradient) {
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Starting PostprocessManager::computeSensitivities ..." << std::endl;
    }
  }
  
  typedef typename Node::device_type LA_device;
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
    
  DFAD obj_sens = 0.0;
  if (response_type != "discrete") {
    this->computeObjective(u, current_time, obj_sens);
  }
  
  size_t set = 0; // hard coded for now
  
  auto u_kv = u[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto adjoint_kv = adjoint[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  if (params->num_active_params > 0) {
  
    params->sacadoizeParams(true);
    
    vector<ScalarT> localsens(params->num_active_params);
    
    vector_RCP res = linalg->getNewVector(set,params->num_active_params);
    matrix_RCP J = linalg->getNewMatrix(set);
    vector_RCP res_over = linalg->getNewOverlappedVector(set,params->num_active_params);
    matrix_RCP J_over = linalg->getNewOverlappedMatrix(set);
    
    auto res_kv = res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    
    res_over->putScalar(0.0);
    
    assembler->assembleJacRes(set, u[set], u[set], false, true, false,
                              res_over, J_over, isTD, current_time, false, false, //store_adjPrev,
                              params->num_active_params, params->Psol[0], false, deltat); //is_final_time, deltat);
    
    linalg->exportVectorFromOverlapped(set, res, res_over);
    
    for (int paramiter=0; paramiter<params->num_active_params; paramiter++) {
      // fine-scale
      if (assembler->groups[0][0]->groupData->multiscale) {
        ScalarT subsens = 0.0;
        for (size_t block=0; block<assembler->groups.size(); ++block) {
          for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
            subsens = -assembler->groups[block][grp]->subgradient(0,paramiter);
            localsens[paramiter] += subsens;
          }
        }
      }
      else { // coarse-scale
      
        ScalarT currsens = 0.0;
        for( size_t i=0; i<res_kv.extent(0); i++ ) {
          currsens += adjoint_kv(i,0) * res_kv(i,paramiter);
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
    vector_RCP a_owned = linalg->getNewVector(set);
    auto ao_kv = a_owned->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    for (size_t i=0; i<ao_kv.extent(0); i++) {
      ao_kv(i,0) = adjoint_kv(i,0);
    }
    vector_RCP res_over = linalg->getNewOverlappedVector(set);
    matrix_RCP J = linalg->getNewParamMatrix();
    matrix_RCP J_over = linalg->getNewParamOverlappedMatrix();
    res_over->putScalar(0.0);
    J->setAllToScalar(0.0);
    
    J_over->setAllToScalar(0.0);
    
    assembler->assembleJacRes(set, u[set], u[set], true, false, true,
                              res_over, J_over, isTD, current_time, false, false, //store_adjPrev,
                              params->num_active_params, params->Psol[0], false, deltat); //is_final_time, deltat);
    
    linalg->fillCompleteParam(set, J_over);
    
    vector_RCP sens_over = linalg->getNewParamOverlappedVector(); //Teuchos::rcp(new LA_MultiVector(params->param_overlapped_map,1));
    vector_RCP sens = linalg->getNewParamVector();
    auto sens_kv = sens->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    
    linalg->exportParamMatrixFromOverlapped(J, J_over);
    linalg->fillCompleteParam(set,J);
    
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
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      std::cout << "******** Finished PostprocessManager::computeSensitivities ..." << std::endl;
    }
  }
  
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
    vector<size_t> myElements = disc->myElements[block];
    
    if (myElements.size() > 0) {
        
      for (size_t set=0; set<setnames.size(); ++set) {
    
        assembler->updatePhysicsSet(set);
      
        vector<string> vartypes = phys->types[set][block];
        vector<int> varorders = phys->orders[set][block];
        int numVars = phys->numVars[set][block]; // probably redundant
        
        for (int n = 0; n<numVars; n++) {
          
          if (vartypes[n] == "HGRAD") {
            
            Kokkos::View<ScalarT**,AssemblyDevice> soln_dev = Kokkos::View<ScalarT**,AssemblyDevice>("solution",myElements.size(), numNodesPerElem);
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            std::string var = varlist[set][block][n];
            for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              auto sol = Kokkos::subview(assembler->groups[block][grp]->u[set], Kokkos::ALL(), n, Kokkos::ALL());
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
          else if (vartypes[n] == "HVOL") {
            Kokkos::View<ScalarT*,AssemblyDevice> soln_dev("solution",myElements.size());
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            std::string var = varlist[set][block][n];
            for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              auto sol = Kokkos::subview(assembler->groups[block][grp]->u[set], Kokkos::ALL(), n, Kokkos::ALL());
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
            View_Sc2 sol("average solution",assembler->groupData[block]->numElem,spaceDim);
            
            for (size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              
              assembler->groups[block][grp]->computeSolutionAverage(var,sol);
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
            mesh->stk_mesh->setCellFieldData(var+append+"x", blockID, myElements, soln_x);
            if (spaceDim > 1) {
              mesh->stk_mesh->setCellFieldData(var+append+"y", blockID, myElements, soln_y);
            }
            if (spaceDim > 2) {
              mesh->stk_mesh->setCellFieldData(var+append+"z", blockID, myElements, soln_z);
            }
            
          }
          else if (vartypes[n] == "HFACE" && write_HFACE_variables) {
            
            Kokkos::View<ScalarT*,AssemblyDevice> soln_faceavg_dev("solution",myElements.size());
            auto soln_faceavg = Kokkos::create_mirror_view(soln_faceavg_dev);
            
            Kokkos::View<ScalarT*,AssemblyDevice> face_measure_dev("face measure",myElements.size());
            
            for( size_t grp=0; grp<assembler->groups[block].size(); ++grp ) {
              auto eID = assembler->groups[block][grp]->localElemID;
              for (size_t face=0; face<assembler->groupData[block]->numSides; face++) {
                int seedwhat = 0;
                for (size_t iset=0; iset<assembler->wkset[block]->numSets; ++iset) {
                  assembler->wkset[block]->computeSolnSteadySeeded(iset, assembler->groups[block][grp]->u[iset], seedwhat);
                }
                //assembler->groups[block][grp]->computeSolnFaceIP(face);
                assembler->groups[block][grp]->updateWorksetFace(face);
                auto wts = assembler->wkset[block]->wts_side;
                auto sol = assembler->wkset[block]->getSolutionField(varlist[set][block][n]+" side");
                parallel_for("postproc plot HFACE",
                             RangePolicy<AssemblyExec>(0,eID.extent(0)),
                             KOKKOS_LAMBDA (const int elem ) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    face_measure_dev(eID(elem)) += wts(elem,pt);
#ifndef MrHyDE_NO_AD
                    soln_faceavg_dev(eID(elem)) += sol(elem,pt).val()*wts(elem,pt);
#else
                    soln_faceavg_dev(eID(elem)) += sol(elem,pt)*wts(elem,pt);
#endif
                  }
                });
              }
            }
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
             
             mesh->stk_mesh->setCellFieldData(var+"x", blockID, myElements, soln_x);
             mesh->stk_mesh->setCellFieldData(var+"y", blockID, myElements, soln_y);
             mesh->stk_mesh->setCellFieldData(var+"z", blockID, myElements, soln_z);
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
          
          auto cfields = phys->getExtraFields(b, 0, nodes, currenttime, assembler->wkset[block]);
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
      // Extra cell fields
      ////////////////////////////////////////////////////////////////
      
      if (extracellfields_list[block].size() > 0) {
        Kokkos::View<ScalarT**,AssemblyDevice> ecd_dev("cell data",myElements.size(),
                                                       extracellfields_list[block].size());
        auto ecd = Kokkos::create_mirror_view(ecd_dev);
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto eID = assembler->groups[block][grp]->localElemID;
          
          assembler->groups[block][grp]->updateWorkset(0,true);
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
        Kokkos::View<ScalarT**,AssemblyDevice> dq_dev("cell data",myElements.size(),
                                                      derivedquantities_list[block].size());
        auto dq = Kokkos::create_mirror_view(dq_dev);
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto eID = assembler->groups[block][grp]->localElemID;
          
          assembler->groups[block][grp]->updateWorkset(0,true);
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
      // TMW This is slightly inefficient, but leaving until cell_data_seed is stored differently
      
      if (assembler->groups[block][0]->groupData->have_phi ||
          assembler->groups[block][0]->groupData->have_rotation ||
          assembler->groups[block][0]->groupData->have_extra_data) {
        
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
      // Cell number
      ////////////////////////////////////////////////////////////////
      
      if (write_cell_number) {
        Kokkos::View<ScalarT*,AssemblyDevice> cellnum_dev("cell number",myElements.size());
        auto cellnum = Kokkos::create_mirror_view(cellnum_dev);
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto eID = assembler->groups[block][grp]->localElemID;
          parallel_for("postproc plot param HVOL",
                       RangePolicy<AssemblyExec>(0,eID.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            cellnum_dev(eID(elem)) = grp; // TMW: is this what we want?
          });
        }
        Kokkos::deep_copy(cellnum, cellnum_dev);
        mesh->stk_mesh->setCellFieldData("group number", blockID, myElements, cellnum);
      }
      
      if (write_database_id) {
        Kokkos::View<ScalarT*,AssemblyDevice> jacnum_dev("unique jac ID",myElements.size());
        auto jacnum = Kokkos::create_mirror_view(jacnum_dev);
        
        for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
          auto index = assembler->groups[block][grp]->basis_database_index;
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
  
  if (write_subgrid_solution && multiscale_manager->subgridModels.size() > 0) {
    for (size_t m=0; m<multiscale_manager->subgridModels.size(); m++) {
      multiscale_manager->subgridModels[m]->writeSolution(currenttime, append);
    }
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
  View_Sc2 fields("cell field data",numElem, extracellfields_list[block].size());
  
  for (size_t fnum=0; fnum<extracellfields_list[block].size(); ++fnum) {
    
    auto cfield = subview(fields, ALL(), fnum);
    auto ecf = functionManagers[block]->evaluate(extracellfields_list[block][fnum],"ip");
    
    if (cellfield_reduction == "mean") { // default
      parallel_for("physics get extra cell fields",
                   RangePolicy<AssemblyExec>(0,wts.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        ScalarT cellmeas = 0.0;
        for (size_t pt=0; pt<wts.extent(1); pt++) {
          cellmeas += wts(e,pt);
        }
        for (size_t j=0; j<wts.extent(1); j++) {
#ifndef MrHyDE_NO_AD
          ScalarT val = ecf(e,j).val();
#else
          ScalarT val = ecf(e,j);
#endif
          cfield(e) += val*wts(e,j)/cellmeas;
        }
      });
    }
    else if (cellfield_reduction == "max") {
      parallel_for("physics get extra cell fields",
                   RangePolicy<AssemblyExec>(0,wts.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_t j=0; j<wts.extent(1); j++) {
#ifndef MrHyDE_NO_AD
          ScalarT val = ecf(e,j).val();
#else
          ScalarT val = ecf(e,j);
#endif
          if (val>cfield(e)) {
            cfield(e) = val;
          }
        }
      });
    }
    if (cellfield_reduction == "min") {
      parallel_for("physics get extra cell fields",
                   RangePolicy<AssemblyExec>(0,wts.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_t j=0; j<wts.extent(1); j++) {
#ifndef MrHyDE_NO_AD
          ScalarT val = ecf(e,j).val();
#else
          ScalarT val = ecf(e,j);
#endif
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
  View_Sc2 fields("cell field data",numElem, derivedquantities_list[block].size());
  
  int prog = 0;
  
  for (size_t m=0; m<phys->modules[block].size(); ++m) {
    for (size_t set=0; set<phys->modules.size(); ++set) {
      
      vector<View_AD2> dqvals = phys->modules[set][block][m]->getDerivedValues();
      for (size_t k=0; k<dqvals.size(); k++) {
        auto cfield = subview(fields, ALL(), prog);
        auto cdq = dqvals[k];
        
        if (cellfield_reduction == "mean") { // default
          parallel_for("physics get extra cell fields",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            ScalarT cellmeas = 0.0;
            for (size_t pt=0; pt<wts.extent(1); pt++) {
              cellmeas += wts(e,pt);
            }
            for (size_t j=0; j<wts.extent(1); j++) {
#ifndef MrHyDE_NO_AD
              ScalarT val = cdq(e,j).val();
#else
              ScalarT val = cdq(e,j);
#endif
              cfield(e) += val*wts(e,j)/cellmeas;
            }
          });
        }
        else if (cellfield_reduction == "max") {
          parallel_for("physics get extra cell fields",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_t j=0; j<wts.extent(1); j++) {
#ifndef MrHyDE_NO_AD
              ScalarT val = cdq(e,j).val();
#else
              ScalarT val = cdq(e,j);
#endif
              if (val>cfield(e)) {
                cfield(e) = val;
              }
            }
          });
        }
        else if (cellfield_reduction == "min") {
          parallel_for("physics get extra cell fields",
                       RangePolicy<AssemblyExec>(0,wts.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_t j=0; j<wts.extent(1); j++) {
#ifndef MrHyDE_NO_AD
              ScalarT val = cdq(e,j).val();
#else
              ScalarT val = cdq(e,j);
#endif
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
    vector<size_t> myElements = disc->myElements[block];
    //vector<string> vartypes = phys->types[block];
    //vector<int> varorders = phys->orders[block];
    
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
             
             mesh->setCellFieldData(var+"x", blockID, myElements, soln_x);
             mesh->setCellFieldData(var+"y", blockID, myElements, soln_y);
             mesh->setCellFieldData(var+"z", blockID, myElements, soln_z);
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
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting PostprocessManager::addSensors ..." << endl;
    }
  }
    
  // Reading in sensors from a mesh file only works on a single element block (for now)
  // There isn't any problem with multiple blocks, it just hasn't been generalized for sensors yet
  for (size_t r=0; r<objectives.size(); ++r) {
    if (objectives[r].type == "sensors") {
  
      if (objectives[r].sensor_points_file == "mesh") {
        //Teuchos::TimeMonitor localtimer(*importexodustimer);
        this->importSensorsFromExodus(r);
      }
      else {
        //Teuchos::TimeMonitor localtimer(*importfiletimer);
        this->importSensorsFromFiles(r);
      }
    }
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished PostprocessManager::addSensors ..." << endl;
    }
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
    int numSensorsInCell = mesh->efield_vals[block][i];
    numFound += numSensorsInCell;
  }
  
  objectives[objID].numSensors = numFound;
  
  if (numFound > 0) {
    
    Kokkos::View<ScalarT**,HostDevice> spts_host("exodus sensors on host",numFound,spaceDim);
    Kokkos::View<int*[2],HostDevice> spts_owners("exodus sensor owners",numFound);
    
    // TMW: as far as I can tell, this is limited to steady-state data
    Kokkos::View<ScalarT*,HostDevice> stime_host("sensor times", 1);
    stime_host(0) = 0.0;
    Kokkos::View<ScalarT**,HostDevice> sdat_host("sensor data", numFound, 1);
    
    size_t sprog = 0;
    for (size_t i=0; i<assembler->groups[block].size(); i++) {
      int numSensorsInCell = mesh->efield_vals[block][i];
      
      if (numSensorsInCell > 0) {
        for (int j=0; j<numSensorsInCell; j++) {
          // sensorLocation
          std::stringstream ssSensorNum;
          ssSensorNum << j+1;
          string sensorNum = ssSensorNum.str();
          string fieldLocx = "sensor_" + sensorNum + "_Loc_x";
          ptrdiff_t ind_Locx = std::distance(mesh->efield_names.begin(),
                                             std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocx));
          spts_host(sprog,0) = mesh->efield_vals[ind_Locx][i];
          
          if (spaceDim > 1) {
            string fieldLocy = "sensor_" + sensorNum + "_Loc_y";
            ptrdiff_t ind_Locy = std::distance(mesh->efield_names.begin(),
                                               std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocy));
            spts_host(sprog,1) = mesh->efield_vals[ind_Locy][i];
          }
          if (spaceDim > 2) {
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
    
    Kokkos::View<ScalarT**,AssemblyDevice> spts("sensor point", numFound, spaceDim);
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
    
    for (size_type pt=0; pt<spts.extent(0); ++pt) {
      
      DRV cpt("point",1,1,spaceDim);
      auto cpt_sub = subview(cpt,0,0,ALL());
      auto pp_sub = subview(spts,pt,ALL());
      Kokkos::deep_copy(cpt_sub,pp_sub);
      
      auto nodes = assembler->groups[block][sowners(pt,0)]->nodes;
      auto nodes_sv = subview(nodes,sowners(pt,1),ALL(),ALL());
      DRV cnodes("subnodes",1,nodes.extent(1),nodes.extent(2));
      auto cnodes_sv = subview(cnodes,0,ALL(),ALL());
      deep_copy(cnodes_sv,nodes_sv);
      
      DRV refpt_tmp = assembler->disc->mapPointsToReference(cpt, cnodes, assembler->groupData[block]->cellTopo);
      DRV refpt("refsenspts",1,spaceDim);
      for (size_type d=0; d<refpt_tmp.extent(2); ++d) {
        refpt(0,d) = refpt_tmp(0,0,d);
      }
      
      vector<Kokkos::View<ScalarT****,AssemblyDevice> > csensorBasis;
      vector<Kokkos::View<ScalarT****,AssemblyDevice> > csensorBasisGrad;
      
      auto orient = assembler->groups[block][sowners(pt,0)]->orientation;
      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> corientation("curr orient",1);
      corientation(0) = orient(sowners(pt,1));
      
      for (size_t k=0; k<assembler->disc->basis_pointers[block].size(); k++) {
        auto basis_ptr = assembler->disc->basis_pointers[block][k];
        string basis_type = assembler->disc->basis_types[block][k];
        auto cellTopo = assembler->groupData[block]->cellTopo;
        
        Kokkos::View<ScalarT****,AssemblyDevice> bvals2, bgradvals2;
        
        if (basis_type == "HGRAD" || basis_type == "HVOL") {
          
          DRV bvals = assembler->disc->evaluateBasis(basis_ptr, refpt, corientation);
          
          bvals2 = Kokkos::View<ScalarT****,AssemblyDevice>("sensor basis",bvals.extent(0),bvals.extent(1),
                                                            bvals.extent(2),spaceDim);
          
          auto bvals2_sv = subview(bvals2,ALL(),ALL(),ALL(),0);
          deep_copy(bvals2_sv,bvals);
          
          DRV bgradvals = assembler->disc->evaluateBasisGrads(basis_ptr, cnodes, refpt, cellTopo, corientation);
          bgradvals2 = Kokkos::View<ScalarT****,AssemblyDevice>("sensor basis",bgradvals.extent(0),
                                                                bgradvals.extent(1),bgradvals.extent(2),spaceDim);
          
          deep_copy(bgradvals2,bgradvals);
          
        }
        csensorBasis.push_back(bvals2);
        csensorBasisGrad.push_back(bgradvals2);
      }
      
      objectives[objID].sensor_basis.push_back(csensorBasis);
      objectives[objID].sensor_basis_grad.push_back(csensorBasisGrad);
      
    }
  }
  
  if (debug_level > 0) {
    if (assembler->Comm->getRank() == 0) {
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
    sdata = Data("Sensor Measurements", spaceDim,
                 objectives[objID].sensor_points_file);
  }
  else {
    sdata = Data("Sensor Measurements", spaceDim,
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
  if (spts_host.extent(1) != static_cast<size_type>(spaceDim)) {
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
  
  for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
    
    auto nodes = assembler->groups[block][grp]->nodes;
    auto nodes_host = create_mirror_view(nodes);
    deep_copy(nodes_host,nodes);
    
    // Create a bounding box for the element
    // This serves as a preprocessing check to avoid unnecessary inclusion checks
    // If a sensor point is not in the box, then it is not in the element
    Kokkos::View<double**[2],HostDevice> nodebox("bounding box",nodes_host.extent(0),spaceDim);
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
          bool proceed = true;
          if (spts_host(pt,0)<nodebox(p,0,0) || spts_host(pt,0)>nodebox(p,0,1)) {
            proceed = false;
          }
          if (proceed && spaceDim > 1) {
            if (spts_host(pt,1)<nodebox(p,1,0) || spts_host(pt,1)>nodebox(p,1,1)) {
              proceed = false;
            }
          }
          if (proceed && spaceDim > 2) {
            if (spts_host(pt,2)<nodebox(p,2,0) || spts_host(pt,2)>nodebox(p,2,1)) {
              proceed = false;
            }
          }
          
          if (proceed) {
            // Need to use DRV, which are on AssemblyDevice
            // We have less control here
            DRV phys_pt("phys_pt",1,1,spaceDim);
            auto phys_pt_host = create_mirror_view(phys_pt);
            for (size_type d=0; d<spts_host.extent(1); ++d) {
              phys_pt_host(0,0,d) = spts_host(pt,d);
            }
            deep_copy(phys_pt,phys_pt_host);
            DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
            auto n_sub = subview(nodes,p,ALL(),ALL());
            auto cn_sub = subview(cnodes,0,ALL(),ALL());
            Kokkos::deep_copy(cn_sub,n_sub);
            
            auto inRefCell = assembler->disc->checkInclusionPhysicalData(phys_pt,cnodes,
                                                                         assembler->groupData[block]->cellTopo,
                                                                         1.0e-15);
            auto inRef_host = create_mirror_view(inRefCell);
            deep_copy(inRef_host,inRefCell);
            if (inRef_host(0,0)) {
              spts_found(pt) = true;
              spts_owners(pt,0) = grp;
              spts_owners(pt,1) = p;
            }
          }
        }
      }// found
    } // pt
  } // elem
  
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
    
    Kokkos::View<ScalarT**,AssemblyDevice> spts("sensor point", numFound, spaceDim);
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
    
    for (size_type pt=0; pt<spts.extent(0); ++pt) {
      
      DRV cpt("point",1,1,spaceDim);
      auto cpt_sub = subview(cpt,0,0,ALL());
      auto pp_sub = subview(spts,pt,ALL());
      Kokkos::deep_copy(cpt_sub,pp_sub);
      
      auto nodes = assembler->groups[block][sowners(pt,0)]->nodes;
      auto nodes_sv = subview(nodes,sowners(pt,1),ALL(),ALL());
      DRV cnodes("subnodes",1,nodes.extent(1),nodes.extent(2));
      auto cnodes_sv = subview(cnodes,0,ALL(),ALL());
      deep_copy(cnodes_sv,nodes_sv);
      
      DRV refpt_tmp = assembler->disc->mapPointsToReference(cpt, cnodes, assembler->groupData[block]->cellTopo);
      DRV refpt("refsenspts",1,spaceDim);
      for (size_type d=0; d<refpt_tmp.extent(2); ++d) {
        refpt(0,d) = refpt_tmp(0,0,d);
      }
      
      vector<Kokkos::View<ScalarT****,AssemblyDevice> > csensorBasis;
      vector<Kokkos::View<ScalarT****,AssemblyDevice> > csensorBasisGrad;
      
      auto orient = assembler->groups[block][sowners(pt,0)]->orientation;
      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> corientation("curr orient",1);
      corientation(0) = orient(sowners(pt,1));
      
      for (size_t k=0; k<assembler->disc->basis_pointers[block].size(); k++) {
        auto basis_ptr = assembler->disc->basis_pointers[block][k];
        string basis_type = assembler->disc->basis_types[block][k];
        auto cellTopo = assembler->groupData[block]->cellTopo;
        
        Kokkos::View<ScalarT****,AssemblyDevice> bvals2, bgradvals2;
        
        if (basis_type == "HGRAD" || basis_type == "HVOL") {
          
          DRV bvals = assembler->disc->evaluateBasis(basis_ptr, refpt, corientation);
          
          bvals2 = Kokkos::View<ScalarT****,AssemblyDevice>("sensor basis",bvals.extent(0),bvals.extent(1),
                                                            bvals.extent(2),spaceDim);
          
          auto bvals2_sv = subview(bvals2,ALL(),ALL(),ALL(),0);
          deep_copy(bvals2_sv,bvals);
          
          DRV bgradvals = assembler->disc->evaluateBasisGrads(basis_ptr, cnodes, refpt, cellTopo, corientation);
          bgradvals2 = Kokkos::View<ScalarT****,AssemblyDevice>("sensor basis",bgradvals.extent(0),
                                                                bgradvals.extent(1),bgradvals.extent(2),spaceDim);
          
          deep_copy(bgradvals2,bgradvals);
          
        }
        csensorBasis.push_back(bvals2);
        csensorBasisGrad.push_back(bgradvals2);
      }
      
      objectives[objID].sensor_basis.push_back(csensorBasis);
      objectives[objID].sensor_basis_grad.push_back(csensorBasisGrad);
      
    }
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished SensorManager::importSensorsFromFiles() ..." << endl;
    }
  }
  
}


// Explicit template instantiations
template class MrHyDE::PostprocessManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
  template class MrHyDE::PostprocessManager<SubgridSolverNode>;
#endif
