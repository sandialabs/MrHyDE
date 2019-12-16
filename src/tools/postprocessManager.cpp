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
/* Constructor to set up the problem */
// ========================================================================================

PostprocessManager::PostprocessManager(const Teuchos::RCP<LA_MpiComm> & Comm_,
                         Teuchos::RCP<Teuchos::ParameterList> & settings,
                         Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                         Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                         Teuchos::RCP<solver> & solve_, Teuchos::RCP<panzer::DOFManager> & DOF_,
                         vector<vector<Teuchos::RCP<cell> > > cells_,
                         Teuchos::RCP<FunctionInterface> & functionManager,
                         Teuchos::RCP<AssemblyManager> & assembler_,
                         Teuchos::RCP<ParameterManager> & params_,
                         Teuchos::RCP<SensorManager> & sensors_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_), solve(solve_),
DOF(DOF_), cells(cells_), assembler(assembler_), params(params_), sensors(sensors_) {
  
  verbosity = settings->sublist("Postprocess").get<int>("Verbosity",1);
  
  overlapped_map = solve->LA_overlapped_map;
  param_overlapped_map = params->param_overlapped_map;
  mesh->getElementBlockNames(blocknames);
  
  numNodesPerElem = settings->sublist("Mesh").get<int>("numNodesPerElem",4);
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  numVars = phys->numVars; //
  
  response_type = settings->sublist("Postprocess").get("response type", "pointwise"); // or "global"
  have_sensor_data = settings->sublist("Analysis").get("Have Sensor Data", false); // or "global"
  save_sensor_data = settings->sublist("Analysis").get("Save Sensor Data",false);
  sname = settings->sublist("Analysis").get("Sensor Prefix","sensor");
  stddev = settings->sublist("Analysis").get("Additive Normal Noise Standard Dev",0.0);
  write_dakota_output = settings->sublist("Postprocess").get("Write Dakota Output",false);
  
  use_sol_mod_mesh = settings->sublist("Postprocess").get<bool>("Solution Based Mesh Mod",false);
  sol_to_mod_mesh = settings->sublist("Postprocess").get<int>("Solution For Mesh Mod",0);
  meshmod_TOL = settings->sublist("Postprocess").get<ScalarT>("Solution Based Mesh Mod TOL",1.0);
  layer_size = settings->sublist("Postprocess").get<ScalarT>("Solution Based Mesh Mod Layer Thickness",0.1);
  compute_subgrid_error = settings->sublist("Postprocess").get<bool>("Subgrid Error",false);
  error_type = settings->sublist("Postprocess").get<string>("Error type","L2"); // or "H1"
  use_sol_mod_height = settings->sublist("Postprocess").get<bool>("Solution Based Height Mod",false);
  sol_to_mod_height = settings->sublist("Postprocess").get<int>("Solution For Height Mod",0);
  
  have_subgrids = false;
  if (settings->isSublist("Subgrid"))
  have_subgrids = true;
  
  isTD = false;
  if (settings->sublist("Solver").get<string>("solver","steady-state") == "transient") {
    isTD = true;
  }
  plot_response = settings->sublist("Postprocess").get<bool>("Plot Response",false);
  save_height_file = settings->sublist("Postprocess").get("Save Height File",false);
  
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
      int vnum = DOF->getFieldNum(var);
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
  
  
    int numElemPerCell = settings->sublist("Solver").get<int>("Workset size",1);
    int numip = disc->ref_ip[0].dimension(0);
    
    if (settings->sublist("Postprocess").isSublist("Responses")) {
      Teuchos::ParameterList resps = settings->sublist("Postprocess").sublist("Responses");
      Teuchos::ParameterList::ConstIterator rsp_itr = resps.begin();
      while (rsp_itr != resps.end()) {
        string entry = resps.get<string>(rsp_itr->first);
        functionManager->addFunction(rsp_itr->first,entry,numElemPerCell,numip,"ip",b);
        rsp_itr++;
      }
    }
    if (settings->sublist("Postprocess").isSublist("Weights")) {
      Teuchos::ParameterList wts = settings->sublist("Postprocess").sublist("Weights");
      Teuchos::ParameterList::ConstIterator wts_itr = wts.begin();
      while (wts_itr != wts.end()) {
        string entry = wts.get<string>(wts_itr->first);
        functionManager->addFunction(wts_itr->first,entry,numElemPerCell,numip,"ip",b);
        wts_itr++;
      }
    }
    if (settings->sublist("Postprocess").isSublist("Targets")) {
      Teuchos::ParameterList tgts = settings->sublist("Postprocess").sublist("Targets");
      Teuchos::ParameterList::ConstIterator tgt_itr = tgts.begin();
      while (tgt_itr != tgts.end()) {
        string entry = tgts.get<string>(tgt_itr->first);
        functionManager->addFunction(tgt_itr->first,entry,numElemPerCell,numip,"ip",b);
        tgt_itr++;
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

void PostprocessManager::computeError() {
  
  
  if(Comm->getRank() == 0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Performing verification ******" << endl << endl;
  }
  
  vector<ScalarT> solvetimes = solve->soln->times[0];
  vector_RCP u;
  
  for (size_t b=0; b<cells.size(); b++) {
    Kokkos::View<ScalarT**,AssemblyDevice> localerror("error",solvetimes.size(),numVars[b]);
    for (size_t t=0; t<solvetimes.size(); t++) {
      bool fnd = solve->soln->extract(u,t);
      assembler->performGather(b,u,0,0);
      for (size_t e=0; e<cells[b].size(); e++) {
        int numElem = cells[b][e]->numElem;
        Kokkos::View<ScalarT**,AssemblyDevice> localerrs = cells[b][e]->computeError(solvetimes[t], t, compute_subgrid_error, error_type);
        
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<numVars[b]; n++) {
            localerror(t,n) += localerrs(p,n);
          }
        }
      }
    }
    
    for (size_t t=0; t<solvetimes.size(); t++) {
      for (int n=0; n<numVars[b]; n++) {
        ScalarT lerr = localerror(t,n);
        ScalarT gerr = 0.0;
        Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&lerr,&gerr);
        if(Comm->getRank() == 0) {
          cout << "***** " << error_type << " norm of the error for " << varlist[b][n] << " = " << sqrt(gerr) << "  (time = " << solvetimes[t] << ")" <<  endl;
        }
      }
    }
    
  }
}

// ========================================================================================
// ========================================================================================

AD PostprocessManager::computeObjective() {
  
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
  
  
  int numSensors = 1;
  if (response_type == "pointwise") {
    numSensors = sensors->numSensors;
  }
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
        if (obj.dimension(1) > 0) {
          for (int c=0; c<numElem; c++) {
            for (size_t i=0; i<obj.dimension(1); i++) {
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
              /*
              int paramIndex, rowIndex, poffset;
              ScalarT val;
              regBoundary = cells[b][e]->computeBoundaryRegularization(boundaryRegConstants,
                                                                       boundaryRegTypes, boundaryRegIndices,
                                                                       boundaryRegSides);
              for (size_t p = 0; p < numBoundaryParams; p++) {
                paramIndex = boundaryRegIndices[p];
                for( size_t row=0; row<paramoffsets[paramIndex].size(); row++ ) {
                  if (regBoundary.size() > 0) {
                    rowIndex = paramGIDs(c,paramoffsets[paramIndex][row]);
                    poffset = paramoffsets[paramIndex][row];
                    val = regBoundary.fastAccessDx(poffset);
                    regGradient[rowIndex+numClassicParams] += val;
                  }
                }
              }*/
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
}

// ========================================================================================
// ========================================================================================

Kokkos::View<ScalarT***,HostDevice> PostprocessManager::computeResponse(const int & b) {
  
  params->sacadoizeParams(false);
  vector<ScalarT> solvetimes = solve->soln->times[0];
  
  //FC responses = this->computeResponse(F_soln);
  
  int numresponses = phys->getNumResponses(b);
  int numSensors = 1;
  if (response_type == "pointwise" ) {
    numSensors = sensors->numSensors;
  }
  
  Kokkos::View<ScalarT***,HostDevice> responses("responses",numSensors, numresponses, solvetimes.size());
  vector_RCP P_soln = params->Psol[0];
  vector_RCP u;
  
  for (size_t tt=0; tt<solvetimes.size(); tt++) {
    bool fnd = solve->soln->extract(u,tt);
    assembler->performGather(b,u,0,0);
    assembler->performGather(b,P_soln,4,0);
    
    for (size_t e=0; e<cells[b].size(); e++) {
      assembler->wkset[b]->update(cells[b][e]->ip, cells[b][e]->ijac, cells[b][e]->orientation);
      
      Kokkos::View<AD***,AssemblyDevice> responsevals = cells[b][e]->computeResponse(solvetimes[tt], tt, 0);
      
      int numElem = cells[b][e]->numElem;
      for (int r=0; r<numresponses; r++) {
        if (response_type == "global" ) {
          DRV wts = assembler->wkset[b]->wts;
          for (int p=0; p<numElem; p++) {
            for (size_t j=0; j<wts.dimension(1); j++) {
              responses(0,r,tt) += responsevals(p,r,j).val() * wts(p,j);
            }
          }
        }
        else if (response_type == "pointwise" ) {
          if (responsevals.dimension(1) > 0) {
            vector<int> sensIDs = cells[b][e]->mySensorIDs;
            for (int p=0; p<numElem; p++) {
              for (size_t j=0; j<responsevals.dimension(2); j++) {
                responses(sensIDs[j],r,tt) += responsevals(p,r,j).val();
              }
            }
          }
        }
      }
      
    }
  }
  //KokkosTools::print(responses);
  
  return responses;
}

// ========================================================================================
// ========================================================================================

void PostprocessManager::computeResponse() {
  
  
  if(Comm->getRank() == 0 ) {
    if (verbosity > 0) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Computing Responses ******" << endl;
      cout << "*********************************************************" << endl;
    }
  }
  
  params->sacadoizeParams(false);
  vector<ScalarT> solvetimes = solve->soln->times[0];
  for (size_t b=0; b<cells.size(); b++) {
    
    Kokkos::View<ScalarT***,HostDevice> responses = this->computeResponse(b);
    
    int numresponses = phys->getNumResponses(b);
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
        for (size_t tt=0; tt<solvetimes.size(); tt++) { // skip the initial condition
          if(Comm->getRank() == 0){
            respOUT << solvetimes[tt] << "  ";
          }
          for (int n=0; n<responses.dimension(1); n++) {
            ScalarT tmp1 = responses(k,n,tt);
            ScalarT tmp2 = 0.0;//globalresp(k,n,tt);
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
      for (int k=0; k<responses.dimension(0); k++) {
        for (int n=0; n<responses.dimension(1); n++) {
          for (int m=0; m<responses.dimension(2); m++) {
            ScalarT tmp1 = responses(k,n,m);
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
}

// ========================================================================================
// ========================================================================================

vector<ScalarT> PostprocessManager::computeSensitivities() {
  
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
}

// ========================================================================================
// ========================================================================================

void PostprocessManager::writeSolution(const std::string & filelabel) {
  
  string filename = filelabel+".exo";
  
  if(Comm->getRank() == 0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Writing the solution to " << filename << endl;
    cout << "*********************************************************" << endl;
  }
  
  //auto E_kv = E_soln->getLocalView<HostDevice>();
  
  //vector<stk_classic::mesh::Entity*> stk_meshElems;
  //mesh->getMyElements(blockID, stk_meshElems);
  
  if (isTD) {
    mesh->setupExodusFile(filename);
  }
  
  //int numSteps = E_soln->getNumVectors();
  vector<ScalarT> solvetimes = solve->soln->times[0];
  int numSteps = solvetimes.size();
  vector_RCP u;
  
  Kokkos::View<ScalarT**,HostDevice> dispz("dispz",cells[0].size(), numNodesPerElem);
  for(int m=0; m<numSteps; m++) {
    bool fnd = solve->soln->extract(u,m);
    auto u_kv = u->getLocalView<HostDevice>();
    for (size_t b=0; b<cells.size(); b++) {
      std::string blockID = blocknames[b];
      vector<vector<int> > curroffsets = phys->offsets[b];
      vector<size_t> myElements = disc->myElements[b];
      for (int n = 0; n<numVars[b]; n++) {
        Kokkos::View<ScalarT**,HostDevice> soln_computed;
        if (numBasis[b][n]>1) {
          soln_computed = Kokkos::View<ScalarT**,HostDevice>("solution",myElements.size(), numBasis[b][n]);
        }
        else {
          soln_computed = Kokkos::View<ScalarT**,HostDevice>("solution",myElements.size(), numNodesPerElem);
        }
        std::string var = varlist[b][n];
        size_t eprog = 0;
        for( size_t e=0; e<cells[b].size(); e++ ) {
          int numElem = cells[b][e]->numElem;
          Kokkos::View<GO**,HostDevice> GIDs = cells[b][e]->GIDs;
          for (int p=0; p<numElem; p++) {
            for( int i=0; i<numBasis[b][n]; i++ ) {
              int pindex = overlapped_map->getLocalElement(GIDs(p,curroffsets[n][i]));
              if (numBasis[b][n] == 1) {
                for( int j=0; j<numNodesPerElem; j++ ) {
                  soln_computed(eprog,j) = u_kv(pindex,0);
                }
              }
              else {
                soln_computed(eprog,i) = u_kv(pindex,0);
              }
              if (use_sol_mod_mesh && sol_to_mod_mesh == n) {
                if (abs(soln_computed(e,i)) >= meshmod_TOL) {
                  dispz(eprog,i) += layer_size;
                }
              }
              else if (use_sol_mod_height && sol_to_mod_height == n) {
                dispz(eprog,i) = 10.0*soln_computed(eprog,i);
              }
            }
            eprog++;
          }
        }
        if (use_sol_mod_mesh && sol_to_mod_mesh == n) {
          mesh->setSolutionFieldData("dispz", blockID, myElements, dispz);
        }
        else if (use_sol_mod_height && sol_to_mod_height == n) {
          mesh->setSolutionFieldData("dispz", blockID, myElements, dispz);
        }
        else {
          if (var == "dx") {
            mesh->setSolutionFieldData("dispx", blockID, myElements, soln_computed);
          }
          if (var == "dy") {
            mesh->setSolutionFieldData("dispy", blockID, myElements, soln_computed);
          }
          if (var == "dz" || var == "H") {
            mesh->setSolutionFieldData("dispz", blockID, myElements, soln_computed);
          }
        }
        
        mesh->setSolutionFieldData(var, blockID, myElements, soln_computed);
      }
      
      ////////////////////////////////////////////////////////////////
      // Discretized Parameters
      ////////////////////////////////////////////////////////////////
      
      vector<string> dpnames = params->discretized_param_names;
      vector<int> numParamBasis = params->paramNumBasis;
      if (dpnames.size() > 0) {
        vector_RCP P_soln = params->Psol[0];
        auto P_kv = P_soln->getLocalView<HostDevice>();
        
        for (size_t n=0; n<dpnames.size(); n++) {
          Kokkos::View<ScalarT**,HostDevice> soln_computed;
          if (numParamBasis[n]>1) {
            soln_computed = Kokkos::View<ScalarT**,HostDevice>("solution",myElements.size(), numParamBasis[n]);
          }
          else {
            soln_computed = Kokkos::View<ScalarT**,HostDevice>("solution",myElements.size(), numNodesPerElem);
          }
          size_t eprog = 0;
          for( size_t e=0; e<cells[b].size(); e++ ) {
            int numElem = cells[b][e]->numElem;
            Kokkos::View<GO**,HostDevice> paramGIDs = cells[b][e]->paramGIDs;
            
            for (int p=0; p<numElem; p++) {
              vector<vector<int> > paramoffsets = params->paramoffsets;
              for( int i=0; i<numParamBasis[n]; i++ ) {
                int pindex = param_overlapped_map->getLocalElement(paramGIDs(p,paramoffsets[n][i]));
                if (numParamBasis[n] == 1) {
                  for( int j=0; j<numNodesPerElem; j++ ) {
                    soln_computed(e,j) = P_kv(pindex,0);
                  }
                }
                else {
                  soln_computed(e,i) = P_kv(pindex,0);
                }
              }
              eprog++;
            }
          }
          mesh->setSolutionFieldData(dpnames[n], blockID, myElements, soln_computed);
        }
        bool have_dRdP = params->have_dRdP;
        if (have_dRdP) {
          vector_RCP P_soln = params->dRdP[0];
          auto P_kv = P_soln->getLocalView<HostDevice>();
          
          for (size_t n=0; n<dpnames.size(); n++) {
            Kokkos::View<ScalarT**,HostDevice> soln_computed;
            if (numParamBasis[n]>1) {
              soln_computed = Kokkos::View<ScalarT**,HostDevice>("solution",myElements.size(), numParamBasis[n]);
            }
            else {
              soln_computed = Kokkos::View<ScalarT**,HostDevice>("solution",myElements.size(), numNodesPerElem);
            }
            size_t eprog = 0;

            for( size_t e=0; e<cells[b].size(); e++ ) {
              int numElem = cells[b][e]->numElem;
              Kokkos::View<GO**,HostDevice> paramGIDs = cells[b][e]->paramGIDs;
              for (int p=0; p<numElem; p++) {
                vector<vector<int> > paramoffsets = params->paramoffsets;
                for( int i=0; i<numParamBasis[n]; i++ ) {
                  int pindex = param_overlapped_map->getLocalElement(paramGIDs(p,paramoffsets[n][i]));
                  if (numParamBasis[n] == 1) {
                    for( int j=0; j<numNodesPerElem; j++ ) {
                      soln_computed(eprog,j) = P_kv(pindex,0);
                    }
                  }
                  else {
                    soln_computed(eprog,i) = P_kv(pindex,0);
                  }
                }
                eprog++;
              }
            }
            mesh->setSolutionFieldData(dpnames[n]+"_dRdP", blockID, myElements, soln_computed);
          }
        }
        
      }
      
      ////////////////////////////////////////////////////////////////
      // Mesh movement
      ////////////////////////////////////////////////////////////////
      
      bool meshpert = false;
      if (meshpert) {
        Kokkos::View<ScalarT**,HostDevice> dispx("dispx",myElements.size(), numNodesPerElem);
        Kokkos::View<ScalarT**,HostDevice> dispy("dispy",myElements.size(), numNodesPerElem);
        Kokkos::View<ScalarT**,HostDevice> dispz("dispz",myElements.size(), numNodesPerElem);
        /* // TMW: commented out for now.  Need a better way to store this data
        size_t eprog = 0;
        for( size_t e=0; e<cells[b].size(); e++ ) {
          DRV nodePert = cells[b][e]->nodepert;
          for (int p=0; p<cells[b][e]->numElem; p++) {
            for( int j=0; j<numNodesPerElem; j++ ) {
              dispx(eprog,j) = nodePert(p,j,0);
              if (spaceDim > 1)
                dispy(eprog,j) = nodePert(p,j,1);
              if (spaceDim > 2)
                dispz(eprog,j) = nodePert(p,j,2);
            }
            eprog++;
          }
        }*/
        mesh->setSolutionFieldData("dispx", blockID, myElements, dispx);
        mesh->setSolutionFieldData("dispy", blockID, myElements, dispy);
        mesh->setSolutionFieldData("dispz", blockID, myElements, dispz);
      }
      
      ////////////////////////////////////////////////////////////////
      // Plot response
      ////////////////////////////////////////////////////////////////
      
      
      if (plot_response) {
        vector_RCP P_soln = params->Psol[0];
        vector<string> responsefieldnames = phys->getResponseFieldNames(b);
        vector<Kokkos::View<ScalarT**,HostDevice> > responsefields;
        for (size_t j=0; j<responsefieldnames.size(); j++) {
          Kokkos::View<ScalarT**,HostDevice> rfdata("response data",myElements.size(), numNodesPerElem);
          responsefields.push_back(rfdata);
        }
        Kokkos::View<AD***,AssemblyDevice> rfields; // response for each cell
        size_t eprog = 0;
        for (size_t k=0; k<cells[b].size(); k++) {
          DRV nodes = cells[b][k]->nodes;
          rfields = cells[b][k]->computeResponseAtNodes(nodes, m, solvetimes[m]);
          for (int p=0; p<cells[b][k]->numElem; p++) {
            for (size_t j=0; j<responsefieldnames.size(); j++) {
              for (size_t i=0; i<nodes.dimension(1); i++) {
                responsefields[j](eprog,i) = rfields(p,j,i).val();
              }
            }
            eprog++;
          }
        }
        for (size_t j=0; j<responsefieldnames.size(); j++) {
          mesh->setSolutionFieldData(responsefieldnames[j], blockID, myElements, responsefields[j]);
        }
      }
      
      
      ////////////////////////////////////////////////////////////////
      // Extra nodal fields
      ////////////////////////////////////////////////////////////////
      
      
      vector<string> extrafieldnames = phys->getExtraFieldNames(b);
      vector<Kokkos::View<ScalarT**,HostDevice> > extrafields;
      for (size_t j=0; j<extrafieldnames.size(); j++) {
        Kokkos::View<ScalarT**,HostDevice> efdata("field data",myElements.size(), numNodesPerElem);
        extrafields.push_back(efdata);
      }
      
      Kokkos::View<ScalarT***,AssemblyDevice> cfields;
      size_t eprog = 0;
      for (size_t k=0; k<cells[b].size(); k++) {
        DRV nodes = cells[b][k]->nodes;
        cfields = phys->getExtraFields(b, nodes, solvetimes[m], assembler->wkset[b]);
        for (int p=0; p<cells[b][k]->numElem; p++) {
          size_t j = 0;
          for (size_t h=0; h<cfields.dimension(1); h++) {
            for (size_t i=0; i<cfields.dimension(2); i++) {
              extrafields[j](eprog,i) = cfields(p,h,i);
            }
            ++j;
          }
          eprog++;
        }
      }
      
      for (size_t j=0; j<extrafieldnames.size(); j++) {
        mesh->setSolutionFieldData(extrafieldnames[j], blockID, myElements, extrafields[j]);
      }
      
      ////////////////////////////////////////////////////////////////
      // Extra cell fields
      ////////////////////////////////////////////////////////////////
      
      
      vector<string> extracellfieldnames = phys->getExtraCellFieldNames(b);
      
      vector<Kokkos::View<ScalarT**,HostDevice> > extracellfields;
      for (size_t j=0; j<extracellfieldnames.size(); j++) {
        Kokkos::View<ScalarT**,HostDevice> efdata("cell data",myElements.size(), 1);
        extracellfields.push_back(efdata);
      }
      eprog = 0;
      for (size_t k=0; k<cells[b].size(); k++) {
        DRV nodes = cells[b][k]->nodes;
        Kokkos::View<ScalarT***,HostDevice> center("center",nodes.dimension(0),1,spaceDim);
        int numnodes = nodes.dimension(1);
        for (int p=0; p<cells[b][k]->numElem; p++) {
          for (int i=0; i<numnodes; i++) {
            for (int d=0; d<spaceDim; d++) {
              center(p,0,d) += nodes(p,i,d) / numnodes;
            }
          }
        }
        cells[b][k]->updateSolnWorkset(u,0); // also updates ip, ijac
        cells[b][k]->updateData();
        assembler->wkset[b]->time = solvetimes[m];
        Kokkos::View<ScalarT***,HostDevice> cfields = phys->getExtraCellFields(b, cells[b][k]->numElem);
        for (int p=0; p<cells[b][k]->numElem; p++) {
          size_t j = 0;
          for (size_t h=0; h<cfields.dimension(1); h++) {
            extracellfields[j](eprog,0) = cfields(p,h,0);
            ++j;
          }
          eprog++;
        }
      }
      for (size_t j=0; j<extracellfieldnames.size(); j++) {
        mesh->setCellFieldData(extracellfieldnames[j], blockID, myElements, extracellfields[j]);
      }
      
      
      if (cells[b][0]->cellData->have_cell_phi || cells[b][0]->cellData->have_cell_rotation) {
        Kokkos::View<ScalarT**,HostDevice> cdata("cell data",myElements.size(), 1);
        int eprog = 0;
        for (size_t k=0; k<cells[b].size(); k++) {
          vector<size_t> cell_data_seed = cells[b][k]->cell_data_seed;
          vector<size_t> cell_data_seedindex = cells[b][k]->cell_data_seedindex;
          Kokkos::View<ScalarT**> cell_data = cells[b][k]->cell_data;
          for (int p=0; p<cells[b][k]->numElem; p++) {
            /*
            if (cell_data.dimension(1) == 3) {
              cdata(eprog,0) = cell_data(p,0);//cell_data_seed[p];
            }
            else if (cell_data.dimension(1) == 9) {
              cdata(eprog,0) = cell_data(p,0);//*cell_data(p,4)*cell_data(p,8);//cell_data_seed[p];
            }
             */
            cdata(eprog,0) = cell_data_seedindex[p];
            eprog++;
          }
        }
        mesh->setCellFieldData("mesh_data_seed", blockID, myElements, cdata);
      }
      
      if (have_subgrids) {
        Kokkos::View<ScalarT**,HostDevice> cdata("cell data",myElements.size(), 1);
        int eprog = 0;
        for (size_t k=0; k<cells[b].size(); k++) {
          vector<vector<size_t> > subgrid_model_index = cells[b][k]->subgrid_model_index;

          for (int p=0; p<cells[b][k]->numElem; p++) {
            cdata(eprog,0) = subgrid_model_index[p][m];
            eprog++;
          }
        }
        mesh->setCellFieldData("subgrid model", blockID, myElements, cdata);
      }
      /*
      if (have_subgrids) {
        Kokkos::View<ScalarT**,HostDevice> subgrid_mean_fields = solve->multiscale_manager->getMeanCellFields(b, m,
                                                                                                             solvetimes[m],
                                                                                                             extracellfieldnames.size());
        Kokkos::View<ScalarT**,HostDevice> csf("csf",myElements.size(),1);
        int eprog = 0;
        for (size_t j=0; j<extracellfieldnames.size(); j++) {
          for (size_t e=0; e<cells[b].size(); e++) {
            for (int p=0; p<cells[b][e]->numElem; p++) {
              csf(eprog,0) = subgrid_mean_fields(e,j);
              eprog++;
            }
          }
          string sgfn = "subgrid_mean_" + extracellfieldnames[j];
          mesh->setCellFieldData(sgfn, blockID, myElements, csf);
        }
       
      }
       */
      
      
      if(isTD) {
        mesh->writeToExodus(solvetimes[m]);
      }
      else
      mesh->writeToExodus(filename);
    }
  }
  
  if (save_height_file) {
    ofstream hOUT("meshpert.dat");
    hOUT.precision(10);
    //int numsteps = E_soln->getNumVectors();
    ScalarT finalt = 0.0;
    bool fnd = solve->soln->extractLast(u,0,finalt);
    auto u_kv = u->getLocalView<HostDevice>();
    for (size_t b=0; b<numBlocks; b++) {
      std::string blockID = blocknames[b];
      vector<vector<int> > curroffsets = phys->offsets[b];
      vector<size_t> myElements = disc->myElements[b];
      for (int n = 0; n<numVars[b]; n++) {
        for( size_t e=0; e<cells[b].size(); e++ ) {
          DRV nodes = cells[b][e]->nodes;
          Kokkos::View<GO**,HostDevice> GIDs = cells[b][e]->GIDs;
          for (int p=0; p<cells[b][e]->numElem; p++) {
            for( int i=0; i<numBasis[b][n]; i++ ) {
              int pindex = overlapped_map->getLocalElement(GIDs(p,curroffsets[n][i]));
              ScalarT soln = u_kv(pindex,0);
              hOUT << nodes(p,i,0) << "  " << nodes(p,i,1) << "  " << soln << endl;
            }
          }
        }
      }
    }
    hOUT.close();
  }
  
  solve->multiscale_manager->writeSolution(filelabel, solvetimes, Comm->getRank());
  
  //for (size_t b=0; b<cells.size(); b++) {
  //  for (size_t e=0; e<cells[b].size(); e++) {
  //    stringstream ss;
  //    ss << Comm->getRank() << "." << e;
  //    string blockname = "subgrid_data/subgrid.exo." + ss.str();// + ".exo";
  //    cells[b][e]->writeSubgridSolution(blockname);
  //  }
  //}
  if(Comm->getRank() == 0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Finished Writing the solution to " << filename << endl;
    cout << "*********************************************************" << endl;
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
// The following function is the adjoint-based error estimate
// Not to be confused with the postprocess::computeError function which uses a true
//   solution to perform verification studies
// ========================================================================================

// TMW Commented out due to lack of use
/*
ScalarT PostprocessManager::computeError() {
  ScalarT errorest = 0.0;
  if(Comm->getRank() == 0 && verbosity>0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Computing Error Estimate ******" << endl << endl;
  }
  
  auto GF_kv = GF_soln->getLocalView<HostDevice>();
  auto GA_kv = GA_soln->getLocalView<HostDevice>();
  
  vector_RCP u = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // forward solution
  //LA_MultiVector A_soln(*LA_overlapped_map,1); // adjoint solution
  vector_RCP a = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // adjoint solution
  vector_RCP a2 = Teuchos::rcp(new LA_MultiVector(solve->LA_owned_map,1)); // adjoint solution
  vector_RCP u_dot = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // previous solution (can be either fwd or adj)
  
  auto u_kv = u->getLocalView<HostDevice>();
  auto a_kv = a->getLocalView<HostDevice>();
  auto a2_kv = a2->getLocalView<HostDevice>();
  auto u_dot_kv = u_dot->getLocalView<HostDevice>();
  
  
  ScalarT deltat = 0.0;
  ScalarT alpha = 0.0;
  ScalarT beta = 1.0;
  if (solve->isTransient) {
    deltat = solve->finaltime / solve->numsteps;
    alpha = 1./deltat;
  }
  
  params->sacadoizeParams(false);
  
  // ******************* ITERATE ON THE Parameters **********************
  
  double current_time = 0.0;
  ScalarT localerror = 0.0;
  for (int timeiter = 0; timeiter<solve->numsteps; timeiter++) {
    
    current_time += deltat;
    
    for( size_t i=0; i<solve->LA_ownedAndShared.size(); i++ ) {
      u_kv(i,0) = GF_kv(i,timeiter+1);
      u_dot_kv(i,0) = alpha*(GF_kv(i,timeiter+1) - GF_kv(i,timeiter));
    }
    for( size_t i=0; i<solve->LA_owned.size(); i++ ) {
      a2_kv(i,0) = GA_kv(i,numsteps-timeiter);
    }
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(solve->LA_owned_map,params->num_active_params)); // reset residual
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,params->num_active_params)); // reset residual
    matrix_RCP J_over = Tpetra::createCrsMatrix<ScalarT>(solve->LA_overlapped_map); // reset Jacobian
    res_over->putScalar(0.0);
    //this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, false, false, false, res_over, J_over);
    assembler->assembleJacRes(u, u_dot, u, u_dot, alpha, beta, false, false, false,
                              res_over, J_over, solve->isTransient, current_time, false, false,
                              params->num_active_params, params->Psol[0], false);
    res->putScalar(0.0);
    res->doExport(*res_over, *(solve->exporter), Tpetra::ADD);
    auto res_kv = res->getLocalView<HostDevice>();
    
    ScalarT currerror = 0.0;
    for( size_t i=0; i<solve->LA_owned.size(); i++ ) {
      currerror += a2_kv(i,0) * res_kv(i,0);
    }
    localerror += currerror;
  }
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localerror,&errorest);
  //Comm->SumAll(&localerror, &errorest, 1);
  
  if(Comm->getRank() == 0 && verbosity>0) {
    cout << "Error estimate = " << errorest << endl;
  }
  return errorest;
}
*/

// ========================================================================================
// ========================================================================================

vector<ScalarT> PostprocessManager::computeParameterSensitivities() {
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
  ScalarT globalsens = 0.0;
  int nsteps = 1;
  if (solve->isTransient) {
    nsteps = solve->soln->times[0].size()-1;
  }
  double current_time = 0.0;
  
  for (int timeiter = 0; timeiter<nsteps; timeiter++) {
    if (solve->isTransient) {
      current_time = solve->soln->times[0][timeiter+1];
      bool fnd = solve->soln->extract(u,timeiter+1);
      bool fnddot = solve->soln_dot->extract(u_dot,timeiter+1);
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
    assembler->assembleJacRes(u, u_dot, u, u_dot, alpha, beta, false, true, false,
                              res_over, J_over, solve->isTransient, current_time, false, false,
                              params->num_active_params, params->Psol[0], false);
    
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
}


// ========================================================================================
// Compute the sensitivity of the objective with respect to discretized parameters
// ========================================================================================

vector<ScalarT> PostprocessManager::computeDiscretizedSensitivities() {
  
  if(Comm->getRank() == 0 && verbosity>0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Computing Discretized Sensitivities ******" << endl << endl;
  }
  //auto F_kv = F_soln->getLocalView<HostDevice>();
  //auto A_kv = A_soln->getLocalView<HostDevice>();
  
  vector_RCP u = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // forward solution
  vector_RCP phi = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // forward solution
  vector_RCP a2 = Teuchos::rcp(new LA_MultiVector(solve->LA_owned_map,1)); // adjoint solution
  vector_RCP u_dot = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // previous solution (can be either fwd or adj)
  
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
      bool fnddot = solve->soln_dot->extract(u_dot,timeiter+1);
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
    /*
     current_time = solvetimes[timeiter+1];
     for( size_t i=0; i<ownedAndShared.size(); i++ ) {
     u[0][i] = F_soln[timeiter+1][i];
     u_dot[0][i] = alpha*(F_soln[timeiter+1][i] - F_soln[timeiter][i]);
     }
     for( size_t i=0; i<owned.size(); i++ ) {
     a2[0][i] = A_soln[nsteps-timeiter][i];
     }
     */
    
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(solve->LA_overlapped_map,1)); // reset residual
    matrix_RCP J_over = Tpetra::createCrsMatrix<ScalarT>(params->param_overlapped_map); // reset Jacobian
    matrix_RCP J = Tpetra::createCrsMatrix<ScalarT>(params->param_owned_map); // reset Jacobian
    //this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, true, false, true, res_over, J_over);
    assembler->assembleJacRes(u, u_dot, u, u_dot, alpha, beta, true, false, true,
                              res_over, J_over, solve->isTransient, current_time, false, false,
                              params->num_active_params, params->Psol[0], false);
    
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
}


