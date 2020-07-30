/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "postprocessManager.hpp"

// ========================================================================================
/* Minimal constructor to set up the problem */
// ========================================================================================

PostprocessManager::PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                                       Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                                       Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                                       vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                                       Teuchos::RCP<AssemblyManager> & assembler_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_),
assembler(assembler_), functionManagers(functionManagers_) {
  this->setup(settings);
}

// ========================================================================================
/* Full constructor to set up the problem */
// ========================================================================================

PostprocessManager::PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                                       Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                                       Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                                       vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                                       Teuchos::RCP<MultiScale> & multiscale_manager_,
                                       Teuchos::RCP<AssemblyManager> & assembler_,
                                       Teuchos::RCP<ParameterManager> & params_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_),
assembler(assembler_), params(params_), //sensors(sensors_),
functionManagers(functionManagers_), multiscale_manager(multiscale_manager_) {
  this->setup(settings);
}

// ========================================================================================
// Setup function used by different constructors
// ========================================================================================

void PostprocessManager::setup(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  
  milo_debug_level = settings->get<int>("debug level",0);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting PostprocessManager::setup()" << endl;
    }
  }
  
  verbosity = settings->get<int>("verbosity",1);
  
  compute_response = settings->sublist("Postprocess").get<bool>("compute responses",false);
  compute_error = settings->sublist("Postprocess").get<bool>("compute errors",false);
  write_solution = settings->sublist("Postprocess").get("write solution",false);
  write_subgrid_solution = settings->sublist("Postprocess").get("write subgrid solution",false);
  
  exodus_filename = settings->sublist("Postprocess").get<string>("output file","output")+".exo";
  
  if (write_solution && Comm->getRank() == 0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Writing the solution to " << exodus_filename << endl;
    cout << "*********************************************************" << endl;
  }
  
  isTD = false;
  if (settings->sublist("Solver").get<string>("solver","steady-state") == "transient") {
    isTD = true;
  }
  
  if (isTD) {
    mesh->setupExodusFile(exodus_filename);
  }
  
  //overlapped_map = solve->LA_overlapped_map;
  //param_overlapped_map = params->param_overlapped_map;
  mesh->getElementBlockNames(blocknames);
  
  numNodesPerElem = settings->sublist("Mesh").get<int>("numNodesPerElem",4); // actually set by mesh interface
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  numVars = phys->numVars; //
  
  response_type = settings->sublist("Postprocess").get("response type", "pointwise"); // or "global"
  have_sensor_data = settings->sublist("Analysis").get("have sensor data", false); // or "global"
  save_sensor_data = settings->sublist("Analysis").get("save sensor data",false);
  sname = settings->sublist("Analysis").get("sensor prefix","sensor");
  stddev = settings->sublist("Analysis").get("additive normal noise standard dev",0.0);
  write_dakota_output = settings->sublist("Postprocess").get("write Dakota output",false);
  
  use_sol_mod_mesh = settings->sublist("Postprocess").get<bool>("solution based mesh mod",false);
  sol_to_mod_mesh = settings->sublist("Postprocess").get<int>("solution for mesh mod",0);
  meshmod_TOL = settings->sublist("Postprocess").get<ScalarT>("solution based mesh mod TOL",1.0);
  layer_size = settings->sublist("Postprocess").get<ScalarT>("solution based mesh mod layer thickness",0.1);
  
  /*
   string error_list = settings->sublist("Postprocess").get<string>("Error type","L2"); // or "H1"
   // Script to break delimited list into pieces
   {
   string delimiter = ", ";
   size_t pos = 0;
   if (error_list.find(delimiter) == string::npos) {
   error_types.push_back(error_list);
   }
   else {
   string token;
   while ((pos = error_list.find(delimiter)) != string::npos) {
   token = error_list.substr(0, pos);
   error_types.push_back(token);
   error_list.erase(0, pos + delimiter.length());
   }
   error_types.push_back(error_list);
   }
   }
   */
  
  /*
   string subgrid_error_list = settings->sublist("Postprocess").get<string>("Subgrid error type","L2"); // or "H1"
   // Script to break delimited list into pieces
   {
   string delimiter = ", ";
   size_t pos = 0;
   if (subgrid_error_list.find(delimiter) == string::npos) {
   subgrid_error_types.push_back(subgrid_error_list);
   }
   else {
   string token;
   while ((pos = subgrid_error_list.find(delimiter)) != string::npos) {
   token = subgrid_error_list.substr(0, pos);
   subgrid_error_types.push_back(token);
   subgrid_error_list.erase(0, pos + delimiter.length());
   }
   subgrid_error_types.push_back(subgrid_error_list);
   }
   }
   */
  
  use_sol_mod_height = settings->sublist("Postprocess").get<bool>("solution based height mod",false);
  sol_to_mod_height = settings->sublist("Postprocess").get<int>("solution for height mod",0);
  
  //have_subgrids = false;
  //if (settings->isSublist("Subgrid"))
  //have_subgrids = true;
  
  plot_response = settings->sublist("Postprocess").get<bool>("plot response",false);
  save_height_file = settings->sublist("Postprocess").get("save height file",false);
  
  vector<vector<int> > cards = disc->cards;
  vector<vector<string> > phys_varlist = phys->varlist;
  
  //offsets = phys->offsets;
  
  for (size_t b=0; b<blocknames.size(); b++) {
    
    vector<int> curruseBasis(numVars[b]);
    vector<int> currnumBasis(numVars[b]);
    vector<string> currvarlist(numVars[b]);
    
    int currmaxbasis = 0;
    for (int j=0; j<numVars[b]; j++) {
      string var = phys_varlist[b][j];
      int vub = phys->getUniqueIndex(b,var);
      //currvarlist[vnum] = var;
      //curruseBasis[vnum] = vub;
      //currnumBasis[vnum] = cards[b][vub];
      currvarlist[j] = var;
      curruseBasis[j] = vub;
      currnumBasis[j] = cards[b][vub];
      currmaxbasis = std::max(currmaxbasis,cards[b][vub]);
    }
    
    //phys->setVars(currvarlist);
    
    varlist.push_back(currvarlist);
    useBasis.push_back(curruseBasis);
    numBasis.push_back(currnumBasis);
    maxbasis.push_back(currmaxbasis);
    
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
    vector<std::pair<size_t,string> > block_error_list;
    for (size_t j=0; j<varlist[b].size(); j++) {
      if (true_solns.isParameter(varlist[b][j])) { // solution at volumetric ip
        if (types[b][j] == "HGRAD" || types[b][j] == "HVOL") {
          std::pair<size_t,string> newerr(j,"L2");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>(varlist[b][j],"0.0");
          functionManagers[b]->addFunction("true "+varlist[b][j],expression,"ip");
        }
      }
      if (true_solns.isParameter("grad("+varlist[b][j]+")_x") || true_solns.isParameter("grad("+varlist[b][j]+")_y") || true_solns.isParameter("grad("+varlist[b][j]+")_z")) { // GRAD of the solution at volumetric ip
        if (types[b][j] == "HGRAD") {
          std::pair<size_t,string> newerr(j,"GRAD");
          block_error_list.push_back(newerr);
          
          string expression = true_solns.get<string>("grad("+varlist[b][j]+")_x","0.0");
          functionManagers[b]->addFunction("true grad("+varlist[b][j]+")_x",expression,"ip");
          if (spaceDim>1) {
            expression = true_solns.get<string>("grad("+varlist[b][j]+")_y","0.0");
            functionManagers[b]->addFunction("true grad("+varlist[b][j]+")_y",expression,"ip");
          }
          if (spaceDim>2) {
            expression = true_solns.get<string>("grad("+varlist[b][j]+")_z","0.0");
            functionManagers[b]->addFunction("true grad("+varlist[b][j]+")_z",expression,"ip");
          }
        }
      }
      if (true_solns.isParameter(varlist[b][j]+" face")) { // solution at face/side ip
        if (types[b][j] == "HGRAD" || types[b][j] == "HFACE") {
          std::pair<size_t,string> newerr(j,"L2 FACE");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>(varlist[b][j]+" face","0.0");
          functionManagers[b]->addFunction("true "+varlist[b][j],expression,"side ip");
          
        }
      }
      if (true_solns.isParameter(varlist[b][j]+"_x") || true_solns.isParameter(varlist[b][j]+"_y") || true_solns.isParameter(varlist[b][j]+"_z")) { // vector solution at volumetric ip
        if (types[b][j] == "HDIV" || types[b][j] == "HCURL") {
          std::pair<size_t,string> newerr(j,"L2 VECTOR");
          block_error_list.push_back(newerr);
          
          string expression = true_solns.get<string>(varlist[b][j]+"_x","0.0");
          functionManagers[b]->addFunction("true "+varlist[b][j]+"_x",expression,"ip");
          
          if (spaceDim>1) {
            expression = true_solns.get<string>(varlist[b][j]+"_y","0.0");
            functionManagers[b]->addFunction("true "+varlist[b][j]+"_y",expression,"ip");
          }
          if (spaceDim>2) {
            expression = true_solns.get<string>(varlist[b][j]+"_z","0.0");
            functionManagers[b]->addFunction("true "+varlist[b][j]+"_z",expression,"ip");
          }
        }
      }
      if (true_solns.isParameter("div("+varlist[b][j]+")")) { // div of solution at volumetric ip
        if (types[b][j] == "HDIV") {
          std::pair<size_t,string> newerr(j,"DIV");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>("div("+varlist[b][j]+")","0.0");
          functionManagers[b]->addFunction("true div("+varlist[b][j]+")",expression,"ip");
          
        }
      }
      if (true_solns.isParameter("curl("+varlist[b][j]+")_x") || true_solns.isParameter("curl("+varlist[b][j]+")_y") || true_solns.isParameter("curl("+varlist[b][j]+")_z")) { // vector solution at volumetric ip
        if (types[b][j] == "HCURL") {
          std::pair<size_t,string> newerr(j,"CURL");
          block_error_list.push_back(newerr);
          
          string expression = true_solns.get<string>("curl("+varlist[b][j]+")_x","0.0");
          functionManagers[b]->addFunction("true curl("+varlist[b][j]+")_x",expression,"ip");
          
          if (spaceDim>1) {
            expression = true_solns.get<string>("curl("+varlist[b][j]+")_y","0.0");
            functionManagers[b]->addFunction("true curl("+varlist[b][j]+")_y",expression,"ip");
          }
          if (spaceDim>2) {
            expression = true_solns.get<string>("curl("+varlist[b][j]+")_z","0.0");
            functionManagers[b]->addFunction("true curl("+varlist[b][j]+")_z",expression,"ip");
          }
        }
      }
    }
    error_list.push_back(block_error_list);
    
  } // end block loop
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished PostprocessManager::setup()" << endl;
    }
  }
  
  
}

// ========================================================================================
// ========================================================================================

void PostprocessManager::record(const ScalarT & currenttime) {
  if (compute_response) {
    this->computeResponse(currenttime);
  }
  if (compute_error) {
    this->computeError(currenttime);
  }
  if (write_solution) {
    this->writeSolution(currenttime);
  }
}

// ========================================================================================
// ========================================================================================

void PostprocessManager::report() {
  
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
        stringstream ss;
        ss << k;
        string str = ss.str();
        string sname2 = sname + "." + str + ".dat";
        ofstream respOUT(sname2.c_str());
        respOUT.precision(16);
        for (size_t tt=0; tt<response_times.size(); tt++) { // skip the initial condition
          if(Comm->getRank() == 0){
            respOUT << response_times[tt] << "  ";
          }
          for (int n=0; n<responses[tt].extent(1); n++) {
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
      ofstream respOUT(sname2.c_str());
      respOUT.precision(16);
      for (int k=0; k<responses[0].extent(0); k++) {// TMW: not correct
        for (int n=0; n<responses[0].extent(1); n++) {// TMW: not correct
          for (int m=0; m<response_times.size(); m++) {
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
          vector<pair<size_t,string> > sg_error_list;
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
              size_t myID = Comm->getRank();
              if (nvars == 0) {
                myID = 100000000;
              }
              size_t gID = 0;
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

void PostprocessManager::computeError(const ScalarT & currenttime) {
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting PostprocessManager::computeError(time)" << endl;
    }
  }
  
  Teuchos::TimeMonitor localtimer(*computeErrorTimer);
  
  error_times.push_back(currenttime);
  
  for (size_t block=0; block<assembler->wkset.size(); block++) {
    assembler->wkset[block]->time = currenttime;
    assembler->wkset[block]->time_KV(0) = currenttime;
  }
  
  // Need to use time step solution instead of stage solution
  bool isTransient = assembler->wkset[0]->isTransient;
  for (size_t block=0; block<assembler->wkset.size(); block++) {
    assembler->wkset[block]->isTransient = false;
  }
  
  vector<Kokkos::View<ScalarT*,AssemblyDevice> > currerror;
  //Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
  int seedwhat = 0;
  
  for (size_t block=0; block<assembler->cells.size(); block++) {// loop over blocks
    
    int altblock; // Needed for subgrid error calculations
    if (assembler->wkset.size()>block) {
      altblock = block;
    }
    else {
      altblock = 0;
    }
    
    Kokkos::View<ScalarT*,AssemblyDevice> blockerrors("error",error_list[altblock].size());
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
      for (size_t etype=0; etype<error_list[altblock].size(); etype++) {
        int var = error_list[altblock][etype].first;
        
        if (error_list[altblock][etype].second == "L2") {
          // compute the true solution
          string expression = "true " + varlist[altblock][var];
          FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
          
          // add in the L2 difference at the volumetric ip
          for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
            for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
              ScalarT diff = assembler->wkset[altblock]->local_soln(elem,var,pt,0).val() - tsol(elem,pt).val();
              blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
            }
          }
        }
        else if (error_list[altblock][etype].second == "GRAD") {
          // compute the true x-component of grad
          string expression = "true grad(" + varlist[altblock][var] + ")_x";
          FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
          
          // add in the L2 difference at the volumetric ip
          for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
            for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
              ScalarT diff = assembler->wkset[altblock]->local_soln_grad(elem,var,pt,0).val() - tsol(elem,pt).val();
              blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
            }
          }
          
          if (spaceDim > 1) {
            // compute the true y-component of grad
            string expression = "true grad(" + varlist[altblock][var] + ")_y";
            FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
            
            // add in the L2 difference at the volumetric ip
            for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
              for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
                ScalarT diff = assembler->wkset[altblock]->local_soln_grad(elem,var,pt,1).val() - tsol(elem,pt).val();
                blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
              }
            }
          }
          
          if (spaceDim >2) {
            // compute the true z-component of grad
            string expression = "true grad(" + varlist[altblock][var] + ")_z";
            FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
            
            // add in the L2 difference at the volumetric ip
            for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
              for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
                ScalarT diff = assembler->wkset[altblock]->local_soln_grad(elem,var,pt,2).val() - tsol(elem,pt).val();
                blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
              }
            }
          }
        }
        else if (error_list[altblock][etype].second == "DIV") {
          // compute the true divergence
          string expression = "true div(" + varlist[altblock][var] + ")";
          FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
          
          // add in the L2 difference at the volumetric ip
          for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
            for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
              ScalarT diff = assembler->wkset[altblock]->local_soln_div(elem,var,pt).val() - tsol(elem,pt).val();
              blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
            }
          }
        }
        else if (error_list[altblock][etype].second == "CURL") {
          // compute the true x-component of grad
          string expression = "true curl(" + varlist[altblock][var] + ")_x";
          FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
          
          // add in the L2 difference at the volumetric ip
          for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
            for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
              ScalarT diff = assembler->wkset[altblock]->local_soln_curl(elem,var,pt,0).val() - tsol(elem,pt).val();
              blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
            }
          }
          
          if (spaceDim > 1) {
            // compute the true y-component of grad
            string expression = "true curl(" + varlist[altblock][var] + ")_y";
            FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
            
            // add in the L2 difference at the volumetric ip
            for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
              for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
                ScalarT diff = assembler->wkset[altblock]->local_soln_curl(elem,var,pt,1).val() - tsol(elem,pt).val();
                blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
              }
            }
          }
          
          if (spaceDim >2) {
            // compute the true z-component of grad
            string expression = "true curl(" + varlist[altblock][var] + ")_z";
            FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
            
            // add in the L2 difference at the volumetric ip
            for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
              for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
                ScalarT diff = assembler->wkset[altblock]->local_soln_curl(elem,var,pt,2).val() - tsol(elem,pt).val();
                blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
              }
            }
          }
        }
        else if (error_list[altblock][etype].second == "L2 VECTOR") {
          // compute the true x-component of grad
          string expression = "true " + varlist[altblock][var] + "_x";
          FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
          
          // add in the L2 difference at the volumetric ip
          for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
            for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
              ScalarT diff = assembler->wkset[altblock]->local_soln(elem,var,pt,0).val() - tsol(elem,pt).val();
              blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
            }
          }
          
          if (spaceDim > 1) {
            // compute the true y-component of grad
            string expression = "true " + varlist[altblock][var] + "_y";
            FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
            
            // add in the L2 difference at the volumetric ip
            for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
              for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
                ScalarT diff = assembler->wkset[altblock]->local_soln(elem,var,pt,1).val() - tsol(elem,pt).val();
                blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
              }
            }
          }
          
          if (spaceDim >2) {
            // compute the true z-component of grad
            string expression = "true " + varlist[altblock][var] + "_z";
            FDATA tsol = functionManagers[altblock]->evaluate(expression,"ip");
            
            // add in the L2 difference at the volumetric ip
            for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
              for( size_t pt=0; pt<assembler->wkset[altblock]->wts.extent(1); pt++ ) {
                ScalarT diff = assembler->wkset[altblock]->local_soln(elem,var,pt,2).val() - tsol(elem,pt).val();
                blockerrors(etype) += diff*diff*assembler->wkset[altblock]->wts(elem,pt);
              }
            }
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
              string expression = "true " + varlist[altblock][var];
              FDATA tsol = functionManagers[altblock]->evaluate(expression,"side ip");
              
              // add in the L2 difference at the volumetric ip
              for (int elem=0; elem<assembler->cells[block][cell]->numElem; elem++) {
                double facemeasure = 0.0;
                for( size_t pt=0; pt<assembler->wkset[altblock]->wts_side.extent(1); pt++ ) {
                  facemeasure += assembler->wkset[altblock]->wts_side(elem,pt);
                }
                
                for( size_t pt=0; pt<assembler->wkset[altblock]->wts_side.extent(1); pt++ ) {
                  ScalarT diff = assembler->wkset[altblock]->local_soln_face(elem,var,pt,0).val() - tsol(elem,pt).val();
                  blockerrors(etype) += 0.5/facemeasure*diff*diff*assembler->wkset[altblock]->wts_side(elem,pt);
                }
              }
            }
          }
        }
      }
    }
    currerror.push_back(blockerrors);
  } // end block loop
  errors.push_back(currerror);
  
  // Reset
  for (size_t b=0; b<assembler->wkset.size(); b++) {
    assembler->wkset[b]->isTransient = isTransient;
  }
  
  if (!(Teuchos::is_null(multiscale_manager))) {
    if (multiscale_manager->subgridModels.size() > 0) {
      // Collect all of the errors for each subgrid model
      vector<vector<Kokkos::View<ScalarT*,AssemblyDevice> > > blocksgerrs;
      
      for (size_t block=0; block<assembler->cells.size(); block++) {// loop over blocks
        
        vector<Kokkos::View<ScalarT*,AssemblyDevice> > sgerrs;
        for (size_t m=0; m<multiscale_manager->subgridModels.size(); m++) {
          //vector<pair<size_t,string> > sub_error_list;
          Kokkos::View<ScalarT*,AssemblyDevice> err = multiscale_manager->subgridModels[m]->computeError(currenttime);
          //Kokkos::View<ScalarT**,AssemblyDevice> err = multiscale_manager->subgridModels[m]->computeError(subgrid_error_types, currenttime);
          sgerrs.push_back(err);
          //subgrid_error_lists.push_back(sub_error_list);
        }
        blocksgerrs.push_back(sgerrs);
      }
      subgrid_errors.push_back(blocksgerrs);
    }
  }
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished PostprocessManager::computeError(time)" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

AD PostprocessManager::computeObjective() {
  /*
   
   if(Comm->getRank() == 0 ) {
   if (verbosity > 0) {
   cout << endl << "*********************************************************" << endl;
   cout << "***** Computing Objective Function ******" << endl << endl;
   }
   }
   
   AD totaldiff = 0.0;
   AD regDomain = 0.0;
   AD regBoundary = 0.0;
   //bvbw    AD classicParamPenalty = 0.0;
   vector<ScalarT> solvetimes = solve->soln->times[0];
   vector<int> domainRegTypes = params->domainRegTypes;
   vector<ScalarT> domainRegConstants = params->domainRegConstants;
   vector<int> domainRegIndices = params->domainRegIndices;
   int numDomainParams = domainRegIndices.size();
   vector<int> boundaryRegTypes = params->boundaryRegTypes;
   vector<ScalarT> boundaryRegConstants = params->boundaryRegConstants;
   vector<int> boundaryRegIndices = params->boundaryRegIndices;
   int numBoundaryParams = boundaryRegIndices.size();
   vector<string> boundaryRegSides = params->boundaryRegSides;
   
   
   params->sacadoizeParams(true);
   int numClassicParams = params->getNumParams(1);
   int numDiscParams = params->getNumParams(4);
   int numParams = numClassicParams + numDiscParams;
   vector<ScalarT> regGradient(numParams);
   vector<ScalarT> dmGradient(numParams);
   vector_RCP P_soln = params->Psol[0];
   vector_RCP u;
   //cout << solvetimes.size() << endl;
   //for (int i=0; i<solvetimes.size(); i++) {
   //  cout << solvetimes[i] << endl;
   //}
   for (size_t tt=0; tt<solvetimes.size(); tt++) {
   for (size_t b=0; b<cells.size(); b++) {
   
   bool fnd = solve->soln->extract(u,tt);
   assembler->performGather(b,u,0,0);
   assembler->performGather(b,P_soln,4,0);
   
   for (size_t e=0; e<cells[b].size(); e++) {
   //cout << e << endl;
   
   Kokkos::View<AD**,AssemblyDevice> obj = cells[b][e]->computeObjective(solvetimes[tt], tt, 0);
   
   int numElem = cells[b][e]->numElem;
   
   vector<vector<int> > paramoffsets = params->paramoffsets;
   //for (size_t tt=0; tt<solvetimes.size(); tt++) { // skip initial condition in 0th position
   if (obj.extent(1) > 0) {
   for (int c=0; c<numElem; c++) {
   for (size_t i=0; i<obj.extent(1); i++) {
   totaldiff += obj(c,i);
   if (numClassicParams > 0) {
   if (obj(c,i).size() > 0) {
   ScalarT val;
   val = obj(c,i).fastAccessDx(0);
   dmGradient[0] += val;
   }
   }
   if (numDiscParams > 0) {
   Kokkos::View<GO**,HostDevice> paramGIDs = cells[b][e]->paramGIDs;
   
   for (int row=0; row<paramoffsets[0].size(); row++) {
   int rowIndex = paramGIDs(c,paramoffsets[0][row]);
   int poffset = paramoffsets[0][row];
   ScalarT val;
   if (obj(c,i).size() > numClassicParams) {
   val = obj(c,i).fastAccessDx(poffset+numClassicParams);
   dmGradient[rowIndex+numClassicParams] += val;
   }
   }
   }
   }
   }
   }
   //}
   if ((numDomainParams > 0) || (numBoundaryParams > 0)) {
   for (int c=0; c<numElem; c++) {
   Kokkos::View<GO**,HostDevice> paramGIDs = cells[b][e]->paramGIDs;
   vector<vector<int> > paramoffsets = params->paramoffsets;
   
   if (numDomainParams > 0) {
   int paramIndex, rowIndex, poffset;
   ScalarT val;
   regDomain = cells[b][e]->computeDomainRegularization(domainRegConstants,
   domainRegTypes, domainRegIndices);
   
   for (size_t p = 0; p < numDomainParams; p++) {
   paramIndex = domainRegIndices[p];
   for( size_t row=0; row<paramoffsets[paramIndex].size(); row++ ) {
   if (regDomain.size() > 0) {
   rowIndex = paramGIDs(c,paramoffsets[paramIndex][row]);
   poffset = paramoffsets[paramIndex][row];
   val = regDomain.fastAccessDx(poffset);
   regGradient[rowIndex+numClassicParams] += val;
   }
   }
   }
   }
   
   
   if (numBoundaryParams > 0) {
   
   //int paramIndex, rowIndex, poffset;
   //ScalarT val;
   //regBoundary = cells[b][e]->computeBoundaryRegularization(boundaryRegConstants,
   //                                                         boundaryRegTypes, boundaryRegIndices,
   //                                                         boundaryRegSides);
   //for (size_t p = 0; p < numBoundaryParams; p++) {
   //  paramIndex = boundaryRegIndices[p];
   //  for( size_t row=0; row<paramoffsets[paramIndex].size(); row++ ) {
   //    if (regBoundary.size() > 0) {
   //      rowIndex = paramGIDs(c,paramoffsets[paramIndex][row]);
   //      poffset = paramoffsets[paramIndex][row];
   //      val = regBoundary.fastAccessDx(poffset);
   //      regGradient[rowIndex+numClassicParams] += val;
   //    }
   //  }
   //}
   }
   
   totaldiff += (regDomain + regBoundary);
   }
   }
   }
   totaldiff += phys->computeTopoResp(b);
   }
   }
   
   //to gather contributions across processors
   ScalarT meep = 0.0;
   Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&totaldiff.val(),&meep);
   //Comm->SumAll(&totaldiff.val(), &meep, 1);
   totaldiff.val() = meep;
   AD fullobj(numParams,meep);
   
   for (size_t j=0; j< numParams; j++) {
   ScalarT dval;
   ScalarT ldval = dmGradient[j] + regGradient[j];
   Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&ldval,&dval);
   //Comm->SumAll(&ldval,&dval,1);
   fullobj.fastAccessDx(j) = dval;
   }
   
   if(Comm->getRank() == 0 ) {
   if (verbosity > 0) {
   cout << "********** Value of Objective Function = " << std::setprecision(16) << fullobj.val() << endl;
   cout << "*********************************************************" << endl;
   }
   }
   
   if(Comm->getRank() == 0) {
   std::string sname2 = "obj.dat";
   ofstream objOUT(sname2.c_str());
   objOUT.precision(16);
   objOUT << fullobj.val() << endl;
   objOUT.close();
   }
   
   
   return fullobj;
   */
}

// ========================================================================================
// ========================================================================================

void PostprocessManager::computeResponse(const ScalarT & currenttime) {
  
  response_times.push_back(currenttime);
  params->sacadoizeParams(false);
  //vector<ScalarT> solvetimes = solve->soln->times[0];
  
  //Kokkos::View<ScalarT**,HostDevice> responses = this->computeResponse(b);
  
  //params->sacadoizeParams(false);
  //vector<ScalarT> solvetimes = solve->soln->times[0];
  
  // TMW: may not work for multi-block
  int numresponses = phys->getNumResponses(0);
  int numSensors = 1;
  if (response_type == "pointwise" ) {
    numSensors = sensors->numSensors;
  }
  
  Kokkos::View<ScalarT**,HostDevice> curr_response("current response",
                                                   numSensors, numresponses);
  //vector_RCP P_soln = params->Psol[0];
  //vector_RCP u;
  
  //  bool fnd = solve->soln->extract(u,tt);
  //  assembler->performGather(b,u,0,0);
  //  assembler->performGather(b,P_soln,4,0);
  for (size_t b=0; b<assembler->cells.size(); b++) {
    for (size_t e=0; e<assembler->cells[b].size(); e++) {
      //assembler->wkset[b]->update(cells[b][e]->ip, cells[b][e]->wts,
      //                            cells[b][e]->jacobian, cells[b][e]->jacobianInv,
      //                            cells[b][e]->jacobianDet, cells[b][e]->orientation);
      
      //Kokkos::View<AD***,AssemblyDevice> responsevals = cells[b][e]->computeResponse(solvetimes[tt], tt, 0);
      Kokkos::View<AD***,AssemblyDevice> responsevals = assembler->cells[b][e]->computeResponse(0);
      
      int numElem = assembler->cells[b][e]->numElem;
      for (int r=0; r<numresponses; r++) {
        if (response_type == "global" ) {
          DRV wts = assembler->wkset[b]->wts;
          for (int p=0; p<numElem; p++) {
            for (size_t j=0; j<wts.extent(1); j++) {
              curr_response(0,r) += responsevals(p,r,j).val() * wts(p,j);
            }
          }
        }
        else if (response_type == "pointwise" ) {
          if (responsevals.extent(1) > 0) {
            vector<int> sensIDs = assembler->cells[b][e]->mySensorIDs;
            for (int p=0; p<numElem; p++) {
              for (size_t j=0; j<responsevals.extent(2); j++) {
                curr_response(sensIDs[j],r) += responsevals(p,r,j).val();
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

vector<ScalarT> PostprocessManager::computeSensitivities() {
  /*
   Teuchos::RCP<Teuchos::Time> sensitivitytimer = Teuchos::rcp(new Teuchos::Time("sensitivity",false));
   sensitivitytimer->start();
   
   vector<string> active_paramnames = params->getParamsNames(1);
   vector<size_t> active_paramlengths = params->getParamsLengths(1);
   
   vector<ScalarT> dwr_sens;
   vector<ScalarT> disc_sens;
   
   int numClassicParams = params->getNumParams(1);
   int numDiscParams = params->getNumParams(4);
   int numParams = numClassicParams + numDiscParams;
   
   vector<ScalarT> gradient(numParams);
   
   AD obj_sens = this->computeObjective();
   
   if (numClassicParams > 0 ) {
   dwr_sens = this->computeParameterSensitivities();
   }
   if (numDiscParams > 0) {
   disc_sens = this->computeDiscretizedSensitivities();
   }
   size_t pprog  = 0;
   for (size_t i=0; i<numClassicParams; i++) {
   ScalarT cobj = 0.0;
   if (i<obj_sens.size()) {
   cobj = obj_sens.fastAccessDx(i);
   }
   gradient[pprog] = cobj + dwr_sens[i];
   pprog++;
   }
   for (size_t i=0; i<numDiscParams; i++) {
   ScalarT cobj = 0.0;
   if (i<obj_sens.size()) {
   cobj = obj_sens.fastAccessDx(i+numClassicParams);
   }
   gradient[pprog] = cobj + disc_sens[i];
   pprog++;
   }
   
   sensitivitytimer->stop();
   
   if(Comm->getRank() == 0 ) {
   if (verbosity > 0) {
   int pprog = 0;
   for (size_t p=0; p < active_paramnames.size(); p++) {
   for (size_t p2=0; p2 < active_paramlengths[p]; p2++) {
   cout << "Sensitivity of the response w.r.t " << active_paramnames[p] << " component " << p2 << " = " << gradient[pprog] << endl;
   pprog++;
   }
   }
   for (size_t p =0; p < numDiscParams; p++)
   cout << "sens w.r.t. discretized param " << p << " is " << gradient[p+numClassicParams] << endl;
   cout << "Sensitivity Calculation Time: " << sensitivitytimer->totalElapsedTime() << endl;
   }
   }
   
   return gradient;
   */
}

// ========================================================================================
// ========================================================================================

void PostprocessManager::writeSolution(const ScalarT & currenttime) {
  
  Teuchos::TimeMonitor localtimer(*writeSolutionTimer);
  
  plot_times.push_back(currenttime);
  
  for (size_t b=0; b<assembler->cells.size(); b++) {
    std::string blockID = blocknames[b];
    vector<vector<int> > curroffsets = phys->offsets[b];
    vector<size_t> myElements = disc->myElements[b];
    vector<string> vartypes = phys->types[b];
    vector<int> varorders = phys->orders[b];
    
    for (int n = 0; n<numVars[b]; n++) {
      
      if (vartypes[n] == "HGRAD") {
        
        if (varorders[n] > 1) { // project to PWlinear for plotting
          
        }
        Kokkos::View<ScalarT**,HostDevice> soln_computed = Kokkos::View<ScalarT**,HostDevice>("solution",myElements.size(), numNodesPerElem);
        
        std::string var = varlist[b][n];
        size_t eprog = 0;
        Kokkos::View<int*,UnifiedDevice> numDOF = assembler->cells[b][0]->cellData->numDOF;
        for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
          int numElem = assembler->cells[b][e]->numElem;
          Kokkos::View<ScalarT***,AssemblyDevice> sol = assembler->cells[b][e]->u;
          auto host_sol = Kokkos::create_mirror_view(sol);
          Kokkos::deep_copy(host_sol,sol);
          for (int p=0; p<numElem; p++) {
            for( int i=0; i<numNodesPerElem; i++ ) {
              soln_computed(eprog,i) = host_sol(p,n,i);
            }
            eprog++;
          }
        }
        if (var == "dx") {
          mesh->setSolutionFieldData("dispx", blockID, myElements, soln_computed);
        }
        if (var == "dy") {
          mesh->setSolutionFieldData("dispy", blockID, myElements, soln_computed);
        }
        if (var == "dz" || var == "H") {
          mesh->setSolutionFieldData("dispz", blockID, myElements, soln_computed);
        }
        
        mesh->setSolutionFieldData(var, blockID, myElements, soln_computed);
      }
      else if (vartypes[n] == "HVOL") {
        Kokkos::View<ScalarT*,HostDevice> soln_computed("solution",myElements.size());
        std::string var = varlist[b][n];
        size_t eprog = 0;
        for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
          int numElem = assembler->cells[b][e]->numElem;
          Kokkos::View<ScalarT***,AssemblyDevice> sol = assembler->cells[b][e]->u;
          auto host_sol = Kokkos::create_mirror_view(sol);
          Kokkos::deep_copy(host_sol,sol);
          for (int p=0; p<numElem; p++) {
            soln_computed(eprog) = host_sol(p,n,0);//u_kv(pindex,0);
            eprog++;
          }
        }
        mesh->setCellFieldData(var, blockID, myElements, soln_computed);
      }
      else if (vartypes[n] == "HDIV" || vartypes[n] == "HCURL") { // need to project each component onto PW-linear basis and PW constant basis
        Kokkos::View<ScalarT*,HostDevice> soln_x("solution",myElements.size());
        Kokkos::View<ScalarT*,HostDevice> soln_y("solution",myElements.size());
        Kokkos::View<ScalarT*,HostDevice> soln_z("solution",myElements.size());
        std::string var = varlist[b][n];
        size_t eprog = 0;
        for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
          Kokkos::View<ScalarT***,AssemblyDevice> sol = assembler->cells[b][e]->u_avg;
          auto host_sol = Kokkos::create_mirror_view(sol);
          Kokkos::deep_copy(host_sol,sol);
          for (int p=0; p<assembler->cells[b][e]->numElem; p++) {
            soln_x(eprog) = host_sol(p,n,0);
            if (spaceDim > 1) {
              soln_y(eprog) = host_sol(p,n,1);
            }
            if (spaceDim > 2) {
              soln_z(eprog) = host_sol(p,n,2);
            }
            eprog++;
          }
        }
        
        mesh->setCellFieldData(var+"x", blockID, myElements, soln_x);
        mesh->setCellFieldData(var+"y", blockID, myElements, soln_y);
        mesh->setCellFieldData(var+"z", blockID, myElements, soln_z);
      }
      else if (vartypes[n] == "HFACE") { // need to project each component onto PW-linear basis and PW constant basis
        
      }
    }
    
    ////////////////////////////////////////////////////////////////
    // Discretized Parameters
    ////////////////////////////////////////////////////////////////
    
    vector<string> dpnames = params->discretized_param_names;
    vector<int> numParamBasis = params->paramNumBasis;
    vector<string> discParamTypes = params->discretized_param_basis_types;
    if (dpnames.size() > 0) {
      for (size_t n=0; n<dpnames.size(); n++) {
        if (discParamTypes[n] == "HGRAD") {
          Kokkos::View<ScalarT**,HostDevice> soln_computed = Kokkos::View<ScalarT**,HostDevice>("solution",myElements.size(), numNodesPerElem);
          size_t eprog = 0;
          for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
            int numElem = assembler->cells[b][e]->numElem;
            Kokkos::View<ScalarT***,AssemblyDevice> sol = assembler->cells[b][e]->param;
            auto host_sol = Kokkos::create_mirror_view(sol);
            Kokkos::deep_copy(host_sol,sol);
            for (int p=0; p<numElem; p++) {
              for( int i=0; i<numNodesPerElem; i++ ) {
                soln_computed(e,i) = host_sol(p,n,i);
              }
              eprog++;
            }
          }
          mesh->setSolutionFieldData(dpnames[n], blockID, myElements, soln_computed);
        }
        else if (discParamTypes[n] == "HVOL") {
          Kokkos::View<ScalarT*,HostDevice> soln_computed("solution",myElements.size());
          std::string var = varlist[b][n];
          size_t eprog = 0;
          for( size_t e=0; e<assembler->cells[b].size(); e++ ) {
            int numElem = assembler->cells[b][e]->numElem;
            Kokkos::View<ScalarT***,AssemblyDevice> sol = assembler->cells[b][e]->param;
            auto host_sol = Kokkos::create_mirror_view(sol);
            Kokkos::deep_copy(host_sol,sol);
            for (int p=0; p<numElem; p++) {
              soln_computed(eprog) = host_sol(p,n,0);//u_kv(pindex,0);
              eprog++;
            }
          }
          mesh->setCellFieldData(var, blockID, myElements, soln_computed);
        }
        else if (discParamTypes[n] == "HDIV" || discParamTypes[n] == "HCURL") {
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
    
    ////////////////////////////////////////////////////////////////
    // Extra nodal fields
    ////////////////////////////////////////////////////////////////
    
    vector<string> extrafieldnames = phys->getExtraFieldNames(b);
    for (size_t j=0; j<extrafieldnames.size(); j++) {
      Kokkos::View<ScalarT**,HostDevice> efdata("field data",myElements.size(), numNodesPerElem);
      size_t eprog = 0;
      for (size_t k=0; k<assembler->cells[b].size(); k++) {
        DRV nodes = assembler->cells[b][k]->nodes;
        Kokkos::View<ScalarT**,AssemblyDevice> cfields = phys->getExtraFields(b, 0, nodes, currenttime, assembler->wkset[b]);
        auto host_cfields = Kokkos::create_mirror_view(cfields);
        Kokkos::deep_copy(host_cfields,cfields);
        for (int p=0; p<assembler->cells[b][k]->numElem; p++) {
          for (size_t i=0; i<host_cfields.extent(1); i++) {
            efdata(eprog,i) = host_cfields(p,i);
          }
          eprog++;
        }
      }
      mesh->setSolutionFieldData(extrafieldnames[j], blockID, myElements, efdata);
    }
    
    ////////////////////////////////////////////////////////////////
    // Extra cell fields
    ////////////////////////////////////////////////////////////////
    
    vector<string> extracellfieldnames = phys->getExtraCellFieldNames(b);
    
    for (size_t j=0; j<extracellfieldnames.size(); j++) {
      Kokkos::View<ScalarT*,HostDevice> efdata("cell data",myElements.size());
      
      int eprog = 0;
      for (size_t k=0; k<assembler->cells[b].size(); k++) {
        
        assembler->cells[b][k]->updateData();
        assembler->cells[b][k]->updateWorksetBasis();
        assembler->wkset[b]->time = currenttime;
        assembler->wkset[b]->computeSolnSteadySeeded(assembler->cells[b][k]->u, 0);
        assembler->wkset[b]->computeSolnVolIP();
        assembler->wkset[b]->computeParamVolIP(assembler->cells[b][k]->param, 0);
        
        Kokkos::View<ScalarT*,AssemblyDevice> cfields = phys->getExtraCellFields(b, j,
                                                                                 assembler->cells[b][k]->wts);
        
        auto host_cfields = Kokkos::create_mirror_view(cfields);
        Kokkos::deep_copy(host_cfields, cfields);
        for (int p=0; p<host_cfields.extent(0); p++) {
          efdata(eprog) = host_cfields(p);
          eprog++;
        }
      }
      mesh->setCellFieldData(extracellfieldnames[j], blockID, myElements, efdata);
    }
    
    ////////////////////////////////////////////////////////////////
    // Mesh data
    ////////////////////////////////////////////////////////////////
    
    if (assembler->cells[b][0]->cellData->have_cell_phi || assembler->cells[b][0]->cellData->have_cell_rotation || assembler->cells[b][0]->cellData->have_extra_data) {
      
      Kokkos::View<ScalarT*,HostDevice> cdata("cell data",myElements.size());
      Kokkos::View<ScalarT*,HostDevice> cseed("cell data seed",myElements.size());
      int eprog = 0;
      for (size_t k=0; k<assembler->cells[b].size(); k++) {
        vector<size_t> cell_data_seed = assembler->cells[b][k]->cell_data_seed;
        vector<size_t> cell_data_seedindex = assembler->cells[b][k]->cell_data_seedindex;
        Kokkos::View<ScalarT**,AssemblyDevice> cell_data = assembler->cells[b][k]->cell_data;
        // TMW: will need to create mirror view to re-enable this
        for (int p=0; p<assembler->cells[b][k]->numElem; p++) {
          
          if (cell_data.extent(1) == 1) {
            cdata(eprog) = cell_data(p,0);//cell_data_seed[p];
          }
          //else if (cell_data.extent(1) == 9) {
          //  cdata(eprog,0) = cell_data(p,0);//cell_data(p,4)*cell_data(p,8);//cell_data_seed[p];
          //}
          
          cseed(eprog) = cell_data_seedindex[p];
          eprog++;
        }
      }
      mesh->setCellFieldData("mesh_data_seed", blockID, myElements, cseed);
      mesh->setCellFieldData("mesh_data", blockID, myElements, cdata);
    }
    
    if (isTD) {
      mesh->writeToExodus(currenttime);
    }
    else {
      mesh->writeToExodus(exodus_filename);
    }
    
  }
  
  if (write_subgrid_solution && multiscale_manager->subgridModels.size() > 0) {
    for (size_t m=0; m<multiscale_manager->subgridModels.size(); m++) {
      multiscale_manager->subgridModels[m]->writeSolution(currenttime);
    }
  }
}


// ========================================================================================
// ========================================================================================

ScalarT PostprocessManager::makeSomeNoise(ScalarT stdev) {
  //generate sample from 0-centered normal with stdev
  //Box-Muller method
  //srand(time(0)); //doing this more frequently than once-per-second results in getting the same numbers...
  ScalarT U1 = rand()/ScalarT(RAND_MAX);
  ScalarT U2 = rand()/ScalarT(RAND_MAX);
  
  return stdev*sqrt(-2*log(U1))*cos(2*PI*U2);
}

// ========================================================================================
// ========================================================================================

vector<ScalarT> PostprocessManager::computeParameterSensitivities() {
  /*
   if(Comm->getRank() == 0 && verbosity>0) {
   cout << endl << "*********************************************************" << endl;
   cout << "***** Computing Sensitivities ******" << endl << endl;
   }
   
   vector_RCP u = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // forward solution
   vector_RCP phi = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // forward solution
   vector_RCP a2 = Teuchos::rcp(new LA_MultiVector(solve->LA_owned_map,1)); // adjoint solution
   vector_RCP u_dot = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // previous solution (can be either fwd or adj)
   
   //auto u_kv = u->getLocalView<HostDevice>();
   auto a2_kv = a2->getLocalView<HostDevice>();
   //auto u_dot_kv = u_dot->getLocalView<HostDevice>();
   
   ScalarT alpha = 0.0;
   ScalarT beta = 1.0;
   
   vector<ScalarT> gradient(params->num_active_params);
   
   params->sacadoizeParams(true);
   
   vector<ScalarT> localsens(params->num_active_params);
   int nsteps = 1;
   if (solve->isTransient) {
   nsteps = solve->soln->times[0].size()-1;
   }
   double current_time = 0.0;
   
   for (int timeiter = 0; timeiter<nsteps; timeiter++) {
   if (solve->isTransient) {
   current_time = solve->soln->times[0][timeiter+1];
   bool fnd = solve->soln->extract(u,timeiter+1);
   bool fndadj = solve->adj_soln->extract(phi,nsteps-timeiter);
   auto phi_kv = phi->getLocalView<HostDevice>();
   
   //for( LO i=0; i<solve->LA_ownedAndShared.size(); i++ ) {
   //  u_dot_kv(i,0) = alpha*(GF_kv(i,timeiter+1) - GF_kv(i,timeiter));
   //  u_kv(i,0) = GF_kv(i,timeiter+1);
   //}
   for( LO i=0; i<solve->LA_owned.size(); i++ ) {
   a2_kv(i,0) = phi_kv(i,0);
   }
   }
   else {
   current_time = solve->soln->times[0][timeiter];
   bool fnd = solve->soln->extract(u,0);
   bool fndadj = solve->adj_soln->extract(phi,0);
   auto phi_kv = phi->getLocalView<HostDevice>();
   
   //for( LO i=0; i<solve->LA_ownedAndShared.size(); i++ ) {
   //  u_kv(i,0) = GF_kv(i,timeiter);
   //}
   for( LO i=0; i<solve->LA_owned.size(); i++ ) {
   a2_kv(i,0) = phi_kv(i,0);
   }
   }
   
   
   vector_RCP res = Teuchos::rcp(new LA_MultiVector(solve->LA_owned_map,params->num_active_params)); // reset residual
   matrix_RCP J = Tpetra::createCrsMatrix<ScalarT>(solve->LA_owned_map); // reset Jacobian
   vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,params->num_active_params)); // reset residual
   matrix_RCP J_over = Tpetra::createCrsMatrix<ScalarT>(solve->LA_overlapped_map); // reset Jacobian
   res_over->putScalar(0.0);
   
   //this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, false, true, false, res_over, J_over);
   assembler->assembleJacRes(u, u, false, true, false,
   res_over, J_over, solve->isTransient, current_time, false, false,
   params->num_active_params, params->Psol[0], false, solve->deltat);
   
   res->putScalar(0.0);
   res->doExport(*res_over, *(solve->exporter), Tpetra::ADD);
   
   auto res_kv = res->getLocalView<HostDevice>();
   
   for (size_t paramiter=0; paramiter < params->num_active_params; paramiter++) {
   ScalarT currsens = 0.0;
   for( LO i=0; i<solve->LA_owned.size(); i++ ) {
   currsens += a2_kv(i,0) * res_kv(i,paramiter);
   }
   localsens[paramiter] -= currsens;
   }
   }
   
   ScalarT localval = 0.0;
   ScalarT globalval = 0.0;
   for (size_t paramiter=0; paramiter < params->num_active_params; paramiter++) {
   localval = localsens[paramiter];
   Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
   //Comm->SumAll(&localval, &globalval, 1);
   gradient[paramiter] = globalval;
   }
   
   if(Comm->getRank() == 0 && solve->batchID == 0) {
   stringstream ss;
   std::string sname2 = "sens.dat";
   ofstream sensOUT(sname2.c_str());
   sensOUT.precision(16);
   for (size_t paramiter=0; paramiter < params->num_active_params; paramiter++) {
   sensOUT << gradient[paramiter] << "  ";
   }
   sensOUT << endl;
   sensOUT.close();
   }
   
   return gradient;
   */
}


// ========================================================================================
// Compute the sensitivity of the objective with respect to discretized parameters
// ========================================================================================

vector<ScalarT> PostprocessManager::computeDiscretizedSensitivities() {
  
  /*
   if(Comm->getRank() == 0 && verbosity>0) {
   cout << endl << "*********************************************************" << endl;
   cout << "***** Computing Discretized Sensitivities ******" << endl << endl;
   }
   //auto F_kv = F_soln->getLocalView<HostDevice>();
   //auto A_kv = A_soln->getLocalView<HostDevice>();
   
   vector_RCP u = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // forward solution
   vector_RCP phi = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // forward solution
   vector_RCP a2 = Teuchos::rcp(new LA_MultiVector(solve->LA_owned_map,1)); // adjoint solution
   
   //auto u_kv = u->getLocalView<HostDevice>();
   auto a2_kv = a2->getLocalView<HostDevice>();
   //auto u_dot_kv = u_dot->getLocalView<HostDevice>();
   
   ScalarT alpha = 0.0;
   ScalarT beta = 1.0;
   
   params->sacadoizeParams(false);
   
   int nsteps = 1;
   if (solve->isTransient) {
   nsteps = solve->soln->times[0].size()-1;
   }
   
   vector_RCP totalsens = Teuchos::rcp(new LA_MultiVector(params->param_owned_map,1));
   auto tsens_kv = totalsens->getLocalView<HostDevice>();
   
   double current_time =0.0;
   
   for (int timeiter = 0; timeiter<nsteps; timeiter++) {
   
   if (solve->isTransient) {
   current_time = solve->soln->times[0][timeiter+1];
   bool fnd = solve->soln->extract(u,timeiter+1);
   bool fndadj = solve->adj_soln->extract(phi,nsteps-timeiter);
   auto phi_kv = phi->getLocalView<HostDevice>();
   
   //for( LO i=0; i<solve->LA_ownedAndShared.size(); i++ ) {
   //  u_dot_kv(i,0) = alpha*(F_kv(i,timeiter+1) - F_kv(i,timeiter));
   //  u_kv(i,0) = F_kv(i,timeiter+1);
   //}
   for( LO i=0; i<solve->LA_owned.size(); i++ ) {
   a2_kv(i,0) = phi_kv(i,0);
   }
   }
   else {
   current_time = solve->soln->times[0][timeiter];
   bool fnd = solve->soln->extract(u,0);
   bool fndadj = solve->adj_soln->extract(phi,0);
   auto phi_kv = phi->getLocalView<HostDevice>();
   
   //for( LO i=0; i<solve->LA_ownedAndShared.size(); i++ ) {
   //  u_kv(i,0) = F_kv(i,timeiter);
   //}
   for( LO i=0; i<solve->LA_owned.size(); i++ ) {
   a2_kv(i,0) = phi_kv(i,0);
   }
   }
   
   // current_time = solvetimes[timeiter+1];
   // for( size_t i=0; i<ownedAndShared.size(); i++ ) {
   // u[0][i] = F_soln[timeiter+1][i];
   // u_dot[0][i] = alpha*(F_soln[timeiter+1][i] - F_soln[timeiter][i]);
   // }
   // for( size_t i=0; i<owned.size(); i++ ) {
   // a2[0][i] = A_soln[nsteps-timeiter][i];
   // }
   //
   
   vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // reset residual
   matrix_RCP J_over = Tpetra::createCrsMatrix<ScalarT>(params->param_overlapped_map); // reset Jacobian
   matrix_RCP J = Tpetra::createCrsMatrix<ScalarT>(params->param_owned_map); // reset Jacobian
   //this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, true, false, true, res_over, J_over);
   assembler->assembleJacRes(u, u, true, false, true,
   res_over, J_over, solve->isTransient, current_time, false, false,
   params->num_active_params, params->Psol[0], false, solve->deltat);
   
   J_over->fillComplete(solve->LA_owned_map, params->param_owned_map);
   vector_RCP sens_over = Teuchos::rcp(new LA_MultiVector(params->param_overlapped_map,1)); // reset residual
   vector_RCP sens = Teuchos::rcp(new LA_MultiVector(params->param_owned_map,1)); // reset residual
   
   J->setAllToScalar(0.0);
   J->doExport(*J_over, *(params->param_exporter), Tpetra::ADD);
   J->fillComplete(solve->LA_owned_map, params->param_owned_map);
   
   J->apply(*a2,*sens);
   
   totalsens->update(1.0, *sens, 1.0);
   }
   
   params->dRdP.push_back(totalsens);
   params->have_dRdP = true;
   
   int numParams = params->getNumParams(4);
   vector<ScalarT> discLocalGradient(numParams);
   vector<ScalarT> discGradient(numParams);
   for (size_t i = 0; i < params->paramOwned.size(); i++) {
   GO gid = params->paramOwned[i];
   discLocalGradient[gid] = tsens_kv(i,0);
   }
   for (size_t i = 0; i < numParams; i++) {
   ScalarT globalval = 0.0;
   ScalarT localval = discLocalGradient[i];
   Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
   //Comm->SumAll(&localval, &globalval, 1);
   discGradient[i] = globalval;
   }
   return discGradient;
   */
}


