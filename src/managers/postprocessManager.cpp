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

// Explicit template instantiations
template class MrHyDE::PostprocessManager<SolverNode>;
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA) && !defined(MrHyDE_SOLVERSPACE_CUDA)
  template class MrHyDE::PostprocessManager<SubgridSolverNode>;
#endif


// ========================================================================================
/* Minimal constructor to set up the problem */
// ========================================================================================

template<class Node>
PostprocessManager<Node>::PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                                             Teuchos::RCP<Teuchos::ParameterList> & settings,
                                             Teuchos::RCP<meshInterface> & mesh_,
                                             //Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                                             //Teuchos::RCP<panzer_stk::STK_Interface> & optimization_mesh_,
                                             Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                                             vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                                             Teuchos::RCP<AssemblyManager<Node> > & assembler_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_), //optimization_mesh(optimization_mesh_),
assembler(assembler_), functionManagers(functionManagers_) {
  this->setup(settings);
}

// ========================================================================================
/* Full constructor to set up the problem */
// ========================================================================================

template<class Node>
PostprocessManager<Node>::PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                                             Teuchos::RCP<Teuchos::ParameterList> & settings,
                                             Teuchos::RCP<meshInterface> & mesh_,
                                             //Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                                             //Teuchos::RCP<panzer_stk::STK_Interface> & optimization_mesh_,
                                             Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                                             vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                                             Teuchos::RCP<MultiScale> & multiscale_manager_,
                                             Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                                             Teuchos::RCP<ParameterManager<Node> > & params_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_), //optimization_mesh(optimization_mesh_),
assembler(assembler_), params(params_), functionManagers(functionManagers_), multiscale_manager(multiscale_manager_) {
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
  
  verbosity = settings->get<int>("verbosity",1);
  
  compute_response = settings->sublist("Postprocess").get<bool>("compute responses",false);
  compute_error = settings->sublist("Postprocess").get<bool>("compute errors",false);
  write_solution = settings->sublist("Postprocess").get("write solution",false);
  write_aux_solution = settings->sublist("Postprocess").get("write aux solution",false);
  write_subgrid_solution = settings->sublist("Postprocess").get("write subgrid solution",false);
  write_HFACE_variables = settings->sublist("Postprocess").get("write HFACE variables",false);
  exodus_filename = settings->sublist("Postprocess").get<string>("output file","output")+".exo";
  write_optimization_solution = settings->sublist("Postprocess").get("create optimization movie",false);
  
  if (verbosity > 0 && Comm->getRank() == 0) {
    if (write_solution && !write_HFACE_variables) {
      bool have_HFACE_vars = false;
      vector<vector<string> > types = phys->types;
      for (size_t b=0; b<types.size(); b++) {
        for (size_t var=0; var<types[b].size(); var++) {
          if (types[b][var] == "HFACE") {
            have_HFACE_vars = true;
          }
        }
      }
      if (phys->have_aux) {
        vector<vector<string> > types = phys->aux_types;
        for (size_t b=0; b<types.size(); b++) {
          for (size_t var=0; var<types[b].size(); var++) {
            if (types[b][var] == "HFACE") {
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
  //overlapped_map = solve->LA_overlapped_map;
  //param_overlapped_map = params->param_overlapped_map;
  mesh->stk_mesh->getElementBlockNames(blocknames);
  
  numNodesPerElem = settings->sublist("Mesh").get<int>("numNodesPerElem",4); // actually set by mesh interface
  spaceDim = mesh->stk_mesh->getDimension();
    
  response_type = settings->sublist("Postprocess").get("response type", "pointwise"); // or "global"
  have_sensor_data = settings->sublist("Analysis").get("have sensor data", false); // or "global"
  save_sensor_data = settings->sublist("Analysis").get("save sensor data",false);
  sname = settings->sublist("Analysis").get("sensor prefix","sensor");
    
  stddev = settings->sublist("Analysis").get("additive normal noise standard dev",0.0);
  write_dakota_output = settings->sublist("Postprocess").get("write Dakota output",false);
  
  varlist = phys->varlist;
  aux_varlist = phys->aux_varlist;
  
  for (size_t b=0; b<blocknames.size(); b++) {
        
    if (settings->sublist("Postprocess").isSublist("Responses")) {
      Teuchos::ParameterList resps = settings->sublist("Postprocess").sublist("Responses");
      Teuchos::ParameterList::ConstIterator rsp_itr = resps.begin();
      while (rsp_itr != resps.end()) {
        string entry = resps.get<string>(rsp_itr->first);
        functionManagers[b]->addFunction(rsp_itr->first,entry,"ip");
        rsp_itr++;
      }
    }
    if (settings->sublist("Postprocess").isSublist("Weights")) {
      Teuchos::ParameterList wts = settings->sublist("Postprocess").sublist("Weights");
      Teuchos::ParameterList::ConstIterator wts_itr = wts.begin();
      while (wts_itr != wts.end()) {
        string entry = wts.get<string>(wts_itr->first);
        functionManagers[b]->addFunction(wts_itr->first,entry,"ip");
        wts_itr++;
      }
    }
    if (settings->sublist("Postprocess").isSublist("Targets")) {
      Teuchos::ParameterList tgts = settings->sublist("Postprocess").sublist("Targets");
      Teuchos::ParameterList::ConstIterator tgt_itr = tgts.begin();
      while (tgt_itr != tgts.end()) {
        string entry = tgts.get<string>(tgt_itr->first);
        functionManagers[b]->addFunction(tgt_itr->first,entry,"ip");
        tgt_itr++;
      }
    }
    
    Teuchos::ParameterList blockPhysSettings;
    if (settings->sublist("Physics").isSublist(blocknames[b])) { // adding block overwrites the default
      blockPhysSettings = settings->sublist("Physics").sublist(blocknames[b]);
    }
    else { // default
      blockPhysSettings = settings->sublist("Physics");
    }
    vector<vector<string> > types = phys->types;
    
    // Add true solutions to the function manager for verification studies
    Teuchos::ParameterList true_solns = blockPhysSettings.sublist("True solutions");
    vector<std::pair<size_t,string> > block_error_list = this->addTrueSolutions(true_solns, varlist[b], types[b], b);
    
    error_list.push_back(block_error_list);
    
    if (phys->have_aux) {
      Teuchos::ParameterList blockPhysSettings;
      if (settings->sublist("Aux Physics").isSublist(blocknames[b])) { // adding block overwrites the default
        blockPhysSettings = settings->sublist("Aux Physics").sublist(blocknames[b]);
      }
      else { // default
        blockPhysSettings = settings->sublist("Aux Physics");
      }
      vector<vector<string> > types = phys->aux_types;
      
      // Add true solutions to the function manager for verification studies
      Teuchos::ParameterList true_solns = blockPhysSettings.sublist("True solutions");
      vector<std::pair<size_t,string> > block_error_list = this->addTrueSolutions(true_solns, phys->aux_varlist[b], types[b], b);
      
      aux_error_list.push_back(block_error_list);
    }
  } // end block loop
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished PostprocessManager::setup()" << endl;
    }
  }
  
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<std::pair<size_t,string> > PostprocessManager<Node>::addTrueSolutions(Teuchos::ParameterList & true_solns,
                                                                             vector<string> & vars,
                                                                             vector<string> & types,
                                                                             const int & block) {
  vector<std::pair<size_t,string> > block_error_list;
  for (size_t j=0; j<vars.size(); j++) {
    if (true_solns.isParameter(vars[j])) { // solution at volumetric ip
      if (types[j] == "HGRAD" || types[j] == "HVOL") {
        std::pair<size_t,string> newerr(j,"L2");
        block_error_list.push_back(newerr);
        string expression = true_solns.get<string>(vars[j],"0.0");
        functionManagers[block]->addFunction("true "+vars[j],expression,"ip");
      }
    }
    if (true_solns.isParameter("grad("+vars[j]+")[x]") || true_solns.isParameter("grad("+vars[j]+")[y]") || true_solns.isParameter("grad("+vars[j]+")[z]")) { // GRAD of the solution at volumetric ip
      if (types[j] == "HGRAD") {
        std::pair<size_t,string> newerr(j,"GRAD");
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
      if (types[j] == "HGRAD" || types[j] == "HFACE") {
        std::pair<size_t,string> newerr(j,"L2 FACE");
        block_error_list.push_back(newerr);
        string expression = true_solns.get<string>(vars[j]+" face","0.0");
        functionManagers[block]->addFunction("true "+vars[j],expression,"side ip");
        
      }
    }
    if (true_solns.isParameter(vars[j]+"[x]") || true_solns.isParameter(vars[j]+"[y]") || true_solns.isParameter(vars[j]+"[z]")) { // vector solution at volumetric ip
      if (types[j] == "HDIV" || types[j] == "HCURL") {
        std::pair<size_t,string> newerr(j,"L2 VECTOR");
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
      if (types[j] == "HDIV") {
        std::pair<size_t,string> newerr(j,"DIV");
        block_error_list.push_back(newerr);
        string expression = true_solns.get<string>("div("+vars[j]+")","0.0");
        functionManagers[block]->addFunction("true div("+vars[j]+")",expression,"ip");
        
      }
    }
    if (true_solns.isParameter("curl("+vars[j]+")[x]") || true_solns.isParameter("curl("+vars[j]+")[y]") || true_solns.isParameter("curl("+vars[j]+")[z]")) { // vector solution at volumetric ip
      if (types[j] == "HCURL") {
        std::pair<size_t,string> newerr(j,"CURL");
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
  
  return block_error_list;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::record(const ScalarT & currenttime) {
  if (compute_response) {
    this->computeResponse(currenttime);
  }
  if (compute_error) {
    this->computeError(currenttime);
  }
  if (write_solution) {
    this->writeSolution(currenttime);
  }
  //if (write_optimization_solution) {
  //  this->writeOptimizationSolution(currenttime);
  //}
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::report() {
  
  ////////////////////////////////////////////////////////////////////////////
  // The subgrid models still store everything, so we create the output after the run
  ////////////////////////////////////////////////////////////////////////////
  
  //if (write_subgrid_solution) {
  //  multiscale_manager->writeSolution(exodus_filename, plot_times, Comm->getRank());
  //}
  
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
    //int numresponses = phys->getNumResponses(b);
    int numSensors = 1;
    if (response_type == "pointwise" ) {
      numSensors = sensors->numSensors;
    }
    
    
    if (response_type == "pointwise" && save_sensor_data) {
      
      srand(time(0)); //use current time as seed for random generator for noise
      
      ScalarT err = 0.0;
      
      
      for (int k=0; k<numSensors; k++) {
        std::stringstream ss;
        ss << k;
        string str = ss.str();
        string sname2 = sname + "." + str + ".dat";
        std::ofstream respOUT(sname2.c_str());
        respOUT.precision(16);
        for (size_t tt=0; tt<response_times.size(); tt++) { // skip the initial condition
          if(Comm->getRank() == 0){
            respOUT << response_times[tt] << "  ";
          }
          for (size_type n=0; n<responses[tt].extent(1); n++) {
            ScalarT tmp1 = responses[tt](k,n);
            ScalarT tmp2 = 0.0;
            Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&tmp1,&tmp2);
            //Comm->SumAll(&tmp1, &tmp2, 1);
            err = this->makeSomeNoise(stddev);
            if(Comm->getRank() == 0) {
              respOUT << tmp2+err << "  ";
            }
          }
          if(Comm->getRank() == 0){
            respOUT << endl;
          }
        }
        respOUT.close();
      }
    }
    
    //KokkosTools::print(responses);
    
    if (write_dakota_output) {
      string sname2 = "results.out";
      std::ofstream respOUT(sname2.c_str());
      respOUT.precision(16);
      for (size_type k=0; k<responses[0].extent(0); k++) {// TMW: not correct
        for (size_type n=0; n<responses[0].extent(1); n++) {// TMW: not correct
          for (size_t m=0; m<response_times.size(); m++) {
            ScalarT tmp1 = responses[m](k,n);
            ScalarT tmp2 = 0.0;//globalresp(k,n,tt);
            Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&tmp1,&tmp2);
            //Comm->SumAll(&tmp1, &tmp2, 1);
            if(Comm->getRank() == 0) {
              respOUT << tmp2 << "  ";
            }
          }
        }
      }
      if(Comm->getRank() == 0){
        respOUT << endl;
      }
      respOUT.close();
    }
    
  }
  
  ////////////////////////////////////////////////////////////////////////////
  // Report the errors for verification tests
  ////////////////////////////////////////////////////////////////////////////
  
  if (compute_error) {
    if(Comm->getRank() == 0) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Computing errors ******" << endl << endl;
    }
    
    for (size_t block=0; block<assembler->cells.size(); block++) {// loop over blocks
      for (size_t etype=0; etype<error_list[block].size(); etype++){
        
        //for (size_t et=0; et<error_types.size(); et++){
        for (size_t time=0; time<error_times.size(); time++) {
          //for (int n=0; n<numVars[b]; n++) {
          
          ScalarT lerr = errors[time][block](etype);
          ScalarT gerr = 0.0;
          Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&lerr,&gerr);
          if(Comm->getRank() == 0) {
            string varname = varlist[block][error_list[block][etype].first];
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
          vector<std::pair<size_t,string> > sg_error_list;
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
                
                string varname = sgvars[sg_error_list[etype].first];
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
  
  for (size_t block=0; block<assembler->cells.size(); block++) {// loop over blocks
    
    int altblock; // Needed for subgrid error calculations
    if (assembler->wkset.size()>block) {
      altblock = block;
    }
    else {
      altblock = 0;
    }
    // Cells can use block, but everything else should be altblock
    // This is due to how the subgrid models store the cells
    
    Kokkos::View<ScalarT*,HostDevice> blockerrors("error",error_list[altblock].size());
    
    if (assembler->cells[block].size() > 0) {
      
      assembler->wkset[altblock]->setTime(currenttime);
      
      // Need to use time step solution instead of stage solution
      bool isTransient = assembler->wkset[altblock]->isTransient;
      assembler->wkset[altblock]->isTransient = false;
      
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
      for (size_t cell=0; cell<assembler->cells[block].size(); cell++) {
        if (have_vol_errs) {
          assembler->wkset[altblock]->computeSolnSteadySeeded(assembler->cells[block][cell]->u, seedwhat);
          assembler->cells[block][cell]->computeSolnVolIP();
        }
        auto wts = assembler->cells[block][cell]->wkset->wts;
        
        for (size_t etype=0; etype<error_list[altblock].size(); etype++) {
          int var = error_list[altblock][etype].first;
          string varname = varlist[altblock][var];
          
          if (error_list[altblock][etype].second == "L2") {
            // compute the true solution
            string expression = varname;
            auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
            auto sol = assembler->wkset[altblock]->getData(expression);
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol(elem,pt).val() - tsol(elem,pt).val();
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
          }
          else if (error_list[altblock][etype].second == "GRAD") {
            // compute the true x-component of grad
            string expression = "grad(" + varname + ")[x]";
            auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
            auto sol_x = assembler->wkset[altblock]->getData(expression);
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol_x(elem,pt).val() - tsol(elem,pt).val();
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
            
            if (spaceDim > 1) {
              // compute the true y-component of grad
              string expression = "grad(" + varname + ")[y]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_y = assembler->wkset[altblock]->getData(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_y(elem,pt).val() - tsol(elem,pt).val();
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
            
            if (spaceDim > 2) {
              // compute the true z-component of grad
              string expression = "grad(" + varname + ")[z]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_z = assembler->wkset[altblock]->getData(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_z(elem,pt).val() - tsol(elem,pt).val();
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
            auto sol_div = assembler->wkset[altblock]->getData(expression);
            
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol_div(elem,pt).val() - tsol(elem,pt).val();
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
          }
          else if (error_list[altblock][etype].second == "CURL") {
            // compute the true x-component of grad
            string expression = "curl(" + varlist[altblock][var] + ")[x]";
            auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
            auto sol_curl_x = assembler->wkset[altblock]->getData(expression);
            
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol_curl_x(elem,pt).val() - tsol(elem,pt).val();
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
            
            if (spaceDim > 1) {
              // compute the true y-component of grad
              string expression = "curl(" + varlist[altblock][var] + ")[y]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_curl_y = assembler->wkset[altblock]->getData(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_curl_y(elem,pt).val() - tsol(elem,pt).val();
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
            
            if (spaceDim >2) {
              // compute the true z-component of grad
              string expression = "curl(" + varlist[altblock][var] + ")[z]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_curl_z = assembler->wkset[altblock]->getData(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_curl_z(elem,pt).val() - tsol(elem,pt).val();
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
          }
          else if (error_list[altblock][etype].second == "L2 VECTOR") {
            // compute the true x-component of grad
            string expression = varlist[altblock][var] + "[x]";
            auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
            auto sol_x = assembler->wkset[altblock]->getData(expression);
            
            // add in the L2 difference at the volumetric ip
            ScalarT error = 0.0;
            parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
              for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                ScalarT diff = sol_x(elem,pt).val() - tsol(elem,pt).val();
                update += diff*diff*wts(elem,pt);
              }
            }, error);
            blockerrors(etype) += error;
            
            if (spaceDim > 1) {
              // compute the true y-component of grad
              string expression = varlist[altblock][var] + "[y]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_y = assembler->wkset[altblock]->getData(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_y(elem,pt).val() - tsol(elem,pt).val();
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
            
            if (spaceDim > 2) {
              // compute the true z-component of grad
              string expression = varlist[altblock][var] + "[z]";
              auto tsol = functionManagers[altblock]->evaluate("true "+expression,"ip");
              auto sol_z = assembler->wkset[altblock]->getData(expression);
              
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_z(elem,pt).val() - tsol(elem,pt).val();
                  update += diff*diff*wts(elem,pt);
                }
              }, error);
              blockerrors(etype) += error;
            }
          }
        }
        if (have_face_errs) {
          for (size_t face=0; face<assembler->cells[block][cell]->cellData->numSides; face++) {
            assembler->wkset[altblock]->computeSolnSteadySeeded(assembler->cells[block][cell]->u, seedwhat);
            assembler->cells[block][cell]->computeSolnFaceIP(face);
            //assembler->cells[block][cell]->computeSolnFaceIP(face, seedwhat);
            for (size_t etype=0; etype<error_list[altblock].size(); etype++) {
              int var = error_list[altblock][etype].first;
              if (error_list[altblock][etype].second == "L2 FACE") {
                // compute the true z-component of grad
                string expression = varlist[altblock][var];
                auto tsol = functionManagers[altblock]->evaluate("true "+expression,"side ip");
                auto sol = assembler->wkset[altblock]->getData(expression+" side");
                auto wts = assembler->cells[block][cell]->wkset->wts_side;
                
                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0,wts.extent(0)),
                                KOKKOS_LAMBDA (const int elem, ScalarT& update) {
                  double facemeasure = 0.0;
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    facemeasure += wts(elem,pt);
                  }
                  
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    ScalarT diff = sol(elem,pt).val() - tsol(elem,pt).val();
                    update += 0.5/facemeasure*diff*diff*wts(elem,pt);
                  }
                }, error);
                blockerrors(etype) += error;
              }
            }
          }
        }
      }
      assembler->wkset[altblock]->isTransient = isTransient;
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
      
      for (size_t block=0; block<assembler->cells.size(); block++) {// loop over blocks
        
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
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::computeResponse(const ScalarT & currenttime) {
  
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
  for (size_t b=0; b<assembler->cells.size(); b++) {
    for (size_t e=0; e<assembler->cells[b].size(); e++) {
  
      auto responsevals = assembler->cells[b][e]->computeResponse(0);
      
      //auto host_response = Kokkos::create_mirror_view(responsevals);
      Kokkos::View<AD***,HostDevice> host_response("response on host",responsevals.extent(0),
                                                   responsevals.extent(1), responsevals.extent(2));
      Kokkos::deep_copy(host_response,responsevals);
      
      for (int r=0; r<numresponses; r++) {
        if (response_type == "global" ) {
          auto wts = assembler->cells[b][e]->wts;
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
            vector<int> sensIDs = assembler->cells[b][e]->mySensorIDs;
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
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::writeSolution(const ScalarT & currenttime) {
  
  Teuchos::TimeMonitor localtimer(*writeSolutionTimer);
  
  plot_times.push_back(currenttime);
  
  for (size_t b=0; b<blocknames.size(); b++) {
    std::string blockID = blocknames[b];
    vector<size_t> myElements = disc->myElements[b];
    vector<string> vartypes = phys->types[b];
    vector<int> varorders = phys->orders[b];
    int numVars = phys->numVars[b]; // probably redundant
    
    if (myElements.size() > 0) {
      
      for (int n = 0; n<numVars; n++) {
        
        if (vartypes[n] == "HGRAD") {
          
          Kokkos::View<ScalarT**,AssemblyDevice> soln_dev = Kokkos::View<ScalarT**,AssemblyDevice>("solution",myElements.size(), numNodesPerElem);
          auto soln_computed = Kokkos::create_mirror_view(soln_dev);
          std::string var = varlist[b][n];
          for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
            auto eID = assembler->cells[b][e]->localElemID;
            auto sol = Kokkos::subview(assembler->cells[b][e]->u, Kokkos::ALL(), n, Kokkos::ALL());
            parallel_for("postproc plot HGRAD",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
              for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                soln_dev(eID(elem),i) = sol(elem,i);
              }
            });
          }
          Kokkos::deep_copy(soln_computed, soln_dev);
          
          if (var == "dx") {
            mesh->stk_mesh->setSolutionFieldData("dispx", blockID, myElements, soln_computed);
          }
          if (var == "dy") {
            mesh->stk_mesh->setSolutionFieldData("dispy", blockID, myElements, soln_computed);
          }
          if (var == "dz" || var == "H") {
            mesh->stk_mesh->setSolutionFieldData("dispz", blockID, myElements, soln_computed);
          }
          
          mesh->stk_mesh->setSolutionFieldData(var, blockID, myElements, soln_computed);
        }
        else if (vartypes[n] == "HVOL") {
          Kokkos::View<ScalarT*,AssemblyDevice> soln_dev("solution",myElements.size());
          auto soln_computed = Kokkos::create_mirror_view(soln_dev);
          std::string var = varlist[b][n];
          for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
            auto eID = assembler->cells[b][e]->localElemID;
            auto sol = Kokkos::subview(assembler->cells[b][e]->u, Kokkos::ALL(), n, Kokkos::ALL());
            parallel_for("postproc plot HVOL",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
              soln_dev(eID(elem)) = sol(elem,0);//u_kv(pindex,0);
            });
          }
          Kokkos::deep_copy(soln_computed,soln_dev);
          mesh->stk_mesh->setCellFieldData(var, blockID, myElements, soln_computed);
        }
        else if (vartypes[n] == "HDIV" || vartypes[n] == "HCURL") { // need to project each component onto PW-linear basis and PW constant basis
          Kokkos::View<ScalarT*,AssemblyDevice> soln_x_dev("solution",myElements.size());
          Kokkos::View<ScalarT*,AssemblyDevice> soln_y_dev("solution",myElements.size());
          Kokkos::View<ScalarT*,AssemblyDevice> soln_z_dev("solution",myElements.size());
          auto soln_x = Kokkos::create_mirror_view(soln_x_dev);
          auto soln_y = Kokkos::create_mirror_view(soln_y_dev);
          auto soln_z = Kokkos::create_mirror_view(soln_z_dev);
          std::string var = varlist[b][n];
          for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
            auto eID = assembler->cells[b][e]->localElemID;
            auto sol = Kokkos::subview(assembler->cells[b][e]->u_avg, Kokkos::ALL(), n, Kokkos::ALL());
            parallel_for("postproc plot HDIV/HCURL",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
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
          mesh->stk_mesh->setCellFieldData(var+"x", blockID, myElements, soln_x);
          mesh->stk_mesh->setCellFieldData(var+"y", blockID, myElements, soln_y);
          mesh->stk_mesh->setCellFieldData(var+"z", blockID, myElements, soln_z);
          
        }
        else if (vartypes[n] == "HFACE" && write_HFACE_variables) {
          
          Kokkos::View<ScalarT*,AssemblyDevice> soln_faceavg_dev("solution",myElements.size());
          auto soln_faceavg = Kokkos::create_mirror_view(soln_faceavg_dev);
          
          Kokkos::View<ScalarT*,AssemblyDevice> face_measure_dev("face measure",myElements.size());
          
          for( size_t c=0; c<assembler->cells[b].size(); c++ ) {
            auto eID = assembler->cells[b][c]->localElemID;
            for (size_t face=0; face<assembler->cellData[b]->numSides; face++) {
              int seedwhat = 0;
              assembler->wkset[b]->computeSolnSteadySeeded(assembler->cells[b][c]->u, seedwhat);
              assembler->cells[b][c]->computeSolnFaceIP(face);
              auto wts = assembler->wkset[b]->wts_side;
              auto sol = assembler->wkset[b]->getData(varlist[b][n]+" side");
              parallel_for("postproc plot HFACE",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  face_measure_dev(eID(elem)) += wts(elem,pt);
                  soln_faceavg_dev(eID(elem)) += sol(elem,pt).val()*wts(elem,pt);
                }
              });
            }
          }
          parallel_for("postproc plot HFACE 2",RangePolicy<AssemblyExec>(0,soln_faceavg_dev.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            soln_faceavg_dev(elem) *= 1.0/face_measure_dev(elem);
          });
          Kokkos::deep_copy(soln_faceavg, soln_faceavg_dev);
          mesh->stk_mesh->setCellFieldData(varlist[b][n], blockID, myElements, soln_faceavg);
        }
      }
      
      if (phys->have_aux && write_aux_solution) {
        
        vector<string> vartypes = phys->aux_types[b];
        vector<string> vars = phys->aux_varlist[b];
        vector<int> varorders = phys->aux_orders[b];
        
        for (size_t n=0; n<vars.size(); n++) {
          string var = vars[n];
          if (vartypes[n] == "HGRAD") {
            
            Kokkos::View<ScalarT**,AssemblyDevice> soln_dev = Kokkos::View<ScalarT**,AssemblyDevice>("solution",myElements.size(), numNodesPerElem);
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
              auto eID = assembler->cells[b][e]->localElemID;
              auto sol = Kokkos::subview(assembler->cells[b][e]->aux, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot HGRAD",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
                for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                  soln_dev(eID(elem),i) = sol(elem,i);
                }
              });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            
            if (var == "dx") {
              mesh->stk_mesh->setSolutionFieldData("dispx", blockID, myElements, soln_computed);
            }
            if (var == "dy") {
              mesh->stk_mesh->setSolutionFieldData("dispy", blockID, myElements, soln_computed);
            }
            if (var == "dz" || var == "H") {
              mesh->stk_mesh->setSolutionFieldData("dispz", blockID, myElements, soln_computed);
            }
            
            mesh->stk_mesh->setSolutionFieldData(var, blockID, myElements, soln_computed);
          }
          else if (vartypes[n] == "HVOL") {
            Kokkos::View<ScalarT*,AssemblyDevice> soln_dev("solution",myElements.size());
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
              auto eID = assembler->cells[b][e]->localElemID;
              auto sol = Kokkos::subview(assembler->cells[b][e]->aux, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot HVOL",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
                soln_dev(eID(elem)) = sol(elem,0);//u_kv(pindex,0);
              });
            }
            Kokkos::deep_copy(soln_computed,soln_dev);
            mesh->stk_mesh->setCellFieldData(var, blockID, myElements, soln_computed);
          }
          else if (vartypes[n] == "HDIV" || vartypes[n] == "HCURL") { // need to project each component onto PW-linear basis and PW constant basis
            Kokkos::View<ScalarT*,AssemblyDevice> soln_x_dev("solution",myElements.size());
            Kokkos::View<ScalarT*,AssemblyDevice> soln_y_dev("solution",myElements.size());
            Kokkos::View<ScalarT*,AssemblyDevice> soln_z_dev("solution",myElements.size());
            auto soln_x = Kokkos::create_mirror_view(soln_x_dev);
            auto soln_y = Kokkos::create_mirror_view(soln_y_dev);
            auto soln_z = Kokkos::create_mirror_view(soln_z_dev);
            for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
              auto eID = assembler->cells[b][e]->localElemID;
              auto sol = Kokkos::subview(assembler->cells[b][e]->aux_avg, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot HDIV/HCURL",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
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
            mesh->stk_mesh->setCellFieldData(var+"x", blockID, myElements, soln_x);
            mesh->stk_mesh->setCellFieldData(var+"y", blockID, myElements, soln_y);
            mesh->stk_mesh->setCellFieldData(var+"z", blockID, myElements, soln_z);
            
          }
          else if (vartypes[n] == "HFACE" && write_HFACE_variables) {
            
            Kokkos::View<ScalarT*,AssemblyDevice> soln_faceavg_dev("solution",myElements.size());
            auto soln_faceavg = Kokkos::create_mirror_view(soln_faceavg_dev);
            
            Kokkos::View<ScalarT*,AssemblyDevice> face_measure_dev("face measure",myElements.size());
            
            for( size_t c=0; c<assembler->cells[b].size(); c++ ) {
              auto eID = assembler->cells[b][c]->localElemID;
              for (size_t face=0; face<assembler->cellData[b]->numSides; face++) {
                int seedwhat = 0;
                assembler->wkset[b]->computeAuxSolnSteadySeeded(assembler->cells[b][c]->aux, seedwhat);
                assembler->cells[b][c]->computeAuxSolnFaceIP(face);
                auto wts = assembler->wkset[b]->wts_side;
                auto sol = assembler->wkset[b]->getData(var+" face");
                parallel_for("postproc plot HFACE",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    face_measure_dev(eID(elem)) += wts(elem,pt);
                    soln_faceavg_dev(eID(elem)) += sol(elem,pt).val()*wts(elem,pt);
                  }
                });
              }
            }
            parallel_for("postproc plot HFACE 2",RangePolicy<AssemblyExec>(0,soln_faceavg_dev.extent(0)), KOKKOS_LAMBDA (const int elem ) {
              soln_faceavg_dev(elem) *= 1.0/face_measure_dev(elem);
            });
            Kokkos::deep_copy(soln_faceavg, soln_faceavg_dev);
            mesh->stk_mesh->setCellFieldData(var, blockID, myElements, soln_faceavg);
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
            for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
              auto eID = assembler->cells[b][e]->localElemID;
              auto sol = Kokkos::subview(assembler->cells[b][e]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HGRAD",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
                for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                  soln_dev(eID(elem),i) = sol(elem,i);
                }
              });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->stk_mesh->setSolutionFieldData(dpnames[n], blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HVOL") {
            Kokkos::View<ScalarT*,AssemblyDevice> soln_dev("solution",myElements.size());
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            //std::string var = varlist[b][n];
            for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
              auto eID = assembler->cells[b][e]->localElemID;
              auto sol = Kokkos::subview(assembler->cells[b][e]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HVOL",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
                soln_dev(eID(elem)) = sol(elem,0);
              });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->stk_mesh->setCellFieldData(dpnames[n], blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HDIV" || discParamTypes[n] == "HCURL") {
            // TMW: this is not actually implemented yet ... not hard to do though
            /*
             Kokkos::View<ScalarT*,HostDevice> soln_x("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_y("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_z("solution",myElements.size());
             std::string var = varlist[b][n];
             size_t eprog = 0;
             for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
             Kokkos::View<ScalarT**,AssemblyDevice> sol = assembler->cells[b][e]->param_avg;
             auto host_sol = Kokkos::create_mirror_view(sol);
             Kokkos::deep_copy(host_sol,sol);
             for (int p=0; p<assembler->cells[b][e]->numElem; p++) {
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
      
      vector<string> extrafieldnames = phys->getExtraFieldNames(b);
      for (size_t j=0; j<extrafieldnames.size(); j++) {
        Kokkos::View<ScalarT**,HostDevice> efd("field data",myElements.size(), numNodesPerElem);
        
        for (size_t k=0; k<assembler->cells[b].size(); k++) {
          auto nodes = assembler->cells[b][k]->nodes;
          auto eID = assembler->cells[b][k]->localElemID;
          auto host_eID = Kokkos::create_mirror_view(eID);
          Kokkos::deep_copy(host_eID,eID);
          
          auto cfields = phys->getExtraFields(b, 0, nodes, currenttime, assembler->wkset[b]);
          auto host_cfields = Kokkos::create_mirror_view(cfields);
          Kokkos::deep_copy(host_cfields,cfields);
          for (size_type p=0; p<host_eID.extent(0); p++) {
            for (size_t i=0; i<host_cfields.extent(1); i++) {
              efd(host_eID(p),i) = host_cfields(p,i);
            }
          }
        }
        mesh->stk_mesh->setSolutionFieldData(extrafieldnames[j], blockID, myElements, efd);
      }
      
      ////////////////////////////////////////////////////////////////
      // Extra cell fields
      ////////////////////////////////////////////////////////////////
      
      vector<string> extracellfieldnames = phys->getExtraCellFieldNames(b);
      
      for (size_t j=0; j<extracellfieldnames.size(); j++) {
        Kokkos::View<ScalarT*,AssemblyDevice> ecd_dev("cell data",myElements.size());
        auto ecd = Kokkos::create_mirror_view(ecd_dev);
        for (size_t k=0; k<assembler->cells[b].size(); k++) {
          auto eID = assembler->cells[b][k]->localElemID;
          
          assembler->cells[b][k]->updateData();
          assembler->cells[b][k]->updateWorksetBasis();
          assembler->wkset[b]->setTime(currenttime);
          assembler->wkset[b]->computeSolnSteadySeeded(assembler->cells[b][k]->u, 0);
          assembler->wkset[b]->computeParamSteadySeeded(assembler->cells[b][k]->param, 0);
          assembler->wkset[b]->computeSolnVolIP();
          assembler->wkset[b]->computeParamVolIP();
          
          auto cfields = phys->getExtraCellFields(b, j, assembler->cells[b][k]->wts);
          
          parallel_for("postproc plot param HVOL",RangePolicy<AssemblyExec>(0,eID.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            ecd_dev(eID(elem)) = cfields(elem);
          });
        }
        Kokkos::deep_copy(ecd, ecd_dev);
        mesh->stk_mesh->setCellFieldData(extracellfieldnames[j], blockID, myElements, ecd);
      }
      
      ////////////////////////////////////////////////////////////////
      // Mesh data
      ////////////////////////////////////////////////////////////////
      // TMW This is slightly inefficient, but leaving until cell_data_seed is stored differently
      
      if (assembler->cells[b][0]->cellData->have_cell_phi || assembler->cells[b][0]->cellData->have_cell_rotation || assembler->cells[b][0]->cellData->have_extra_data) {
        
        Kokkos::View<ScalarT*,HostDevice> cdata("cell data",myElements.size());
        Kokkos::View<ScalarT*,HostDevice> cseed("cell data seed",myElements.size());
        for (size_t k=0; k<assembler->cells[b].size(); k++) {
          vector<size_t> cell_data_seed = assembler->cells[b][k]->cell_data_seed;
          vector<size_t> cell_data_seedindex = assembler->cells[b][k]->cell_data_seedindex;
          Kokkos::View<ScalarT**,AssemblyDevice> cell_data = assembler->cells[b][k]->cell_data;
          Kokkos::View<LO*,AssemblyDevice> eID = assembler->cells[b][k]->localElemID;
          auto host_eID = Kokkos::create_mirror_view(eID);
          Kokkos::deep_copy(host_eID,eID);
          
          for (size_type p=0; p<host_eID.extent(0); p++) {
            if (cell_data.extent(1) == 1) {
              cdata(host_eID(p)) = cell_data(p,0);//cell_data_seed[p];
            }
            cseed(host_eID(p)) = cell_data_seedindex[p];
          }
        }
        mesh->stk_mesh->setCellFieldData("mesh_data_seed", blockID, myElements, cseed);
        mesh->stk_mesh->setCellFieldData("mesh_data", blockID, myElements, cdata);
      }
      
      ////////////////////////////////////////////////////////////////
      // Cell number
      ////////////////////////////////////////////////////////////////
      
      Kokkos::View<ScalarT*,AssemblyDevice> cellnum_dev("cell number",myElements.size());
      auto cellnum = Kokkos::create_mirror_view(cellnum_dev);
      
      for (size_t k=0; k<assembler->cells[b].size(); k++) {
        auto eID = assembler->cells[b][k]->localElemID;
        parallel_for("postproc plot param HVOL",
                     RangePolicy<AssemblyExec>(0,eID.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          cellnum_dev(eID(elem)) = elem; // TMW: is this what we want?
        });
      }
      Kokkos::deep_copy(cellnum, cellnum_dev);
      mesh->stk_mesh->setCellFieldData("cell number", blockID, myElements, cellnum);
      
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
      multiscale_manager->subgridModels[m]->writeSolution(currenttime);
    }
  }
}


// ========================================================================================
// ========================================================================================

template<class Node>
void PostprocessManager<Node>::writeOptimizationSolution(const int & numEvaluations) {
  
  Teuchos::TimeMonitor localtimer(*writeSolutionTimer);
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    std::string blockID = blocknames[b];
    vector<vector<int> > curroffsets = disc->offsets[b];
    vector<size_t> myElements = disc->myElements[b];
    vector<string> vartypes = phys->types[b];
    vector<int> varorders = phys->orders[b];
    
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
            for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
              auto eID = assembler->cells[b][e]->localElemID;
              auto sol = Kokkos::subview(assembler->cells[b][e]->param, Kokkos::ALL(), n, Kokkos::ALL());
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
            //std::string var = varlist[b][n];
            for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
              auto eID = assembler->cells[b][e]->localElemID;
              auto sol = Kokkos::subview(assembler->cells[b][e]->param, Kokkos::ALL(), n, Kokkos::ALL());
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
             std::string var = varlist[b][n];
             size_t eprog = 0;
             for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
             Kokkos::View<ScalarT**,AssemblyDevice> sol = assembler->cells[b][e]->param_avg;
             auto host_sol = Kokkos::create_mirror_view(sol);
             Kokkos::deep_copy(host_sol,sol);
             for (int p=0; p<assembler->cells[b][e]->numElem; p++) {
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

