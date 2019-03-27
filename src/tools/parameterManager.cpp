/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "discretizationInterface.hpp"
#include "discretizationTools.hpp"
#include "workset.hpp"
#include "parameterManager.hpp"


// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

ParameterManager::ParameterManager(const Teuchos::RCP<LA_MpiComm> & Comm_,
                                   Teuchos::RCP<Teuchos::ParameterList> & settings,
                                   Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                                   Teuchos::RCP<physics> & phys_,
                                   vector<vector<Teuchos::RCP<cell> > > & cells) :
Comm(Comm_), mesh(mesh_), phys(phys_) {
  
  
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  /////////////////////////////////////////////////////////////////////////////
  
  mesh->getElementBlockNames(blocknames);
  spaceDim = settings->sublist("Mesh").get<int>("dim");
  verbosity = settings->get<int>("verbosity",0);
  
  num_inactive_params = 0;
  num_active_params = 0;
  num_stochastic_params = 0;
  num_discrete_params = 0;
  num_discretized_params = 0;
  globalParamUnknowns = 0;
  have_dRdP = false;
  discretized_stochastic = false;
  
  use_custom_initial_param_guess = settings->sublist("Physics").get<bool>("use custom initial param guess",false);
  
  this->setupParameters(settings);
  this->setupDiscretizedParameters(cells);
  
}

// ========================================================================================
// Set up the parameters (inactive, active, stochastic, discrete)
// Communicate these parameters back to the physics interface and the enabled modules
// ========================================================================================

void ParameterManager::setupParameters(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  Teuchos::ParameterList parameters;
  
  if (settings->isSublist("Parameters")) {
    parameters = settings->sublist("Parameters");
    Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
    while (pl_itr != parameters.end()) {
      Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
      vector<ScalarT> newparamvals;
      int numnewparams = 0;
      if (!newparam.isParameter("type") || !newparam.isParameter("usage")) {
        // print out error message
      }
      
      if (newparam.get<string>("type") == "scalar") {
        newparamvals.push_back(newparam.get<ScalarT>("value"));
        numnewparams = 1;
      }
      else if (newparam.get<string>("type") == "vector") {
        std::string filename = newparam.get<string>("source");
        std::ifstream fin(filename.c_str());
        std::istream_iterator<ScalarT> start(fin), end;
        vector<ScalarT> importedparamvals(start, end);
        for (size_t i=0; i<importedparamvals.size(); i++) {
          newparamvals.push_back(importedparamvals[i]);
        }
      }
      
      paramnames.push_back(pl_itr->first);
      paramvals.push_back(newparamvals);
      
      Teuchos::RCP<vector<AD> > newparam_AD = Teuchos::rcp(new vector<AD>(newparamvals.size()));
      paramvals_AD.push_back(newparam_AD);
      
      //blank bounds
      vector<ScalarT> lo(newparamvals.size(),0.0);
      vector<ScalarT> up(newparamvals.size(),0.0);
      
      if (newparam.get<string>("usage") == "inactive") {
        paramtypes.push_back(0);
        num_inactive_params += newparamvals.size();
      }
      else if (newparam.get<string>("usage") == "active") {
        paramtypes.push_back(1);
        num_active_params += newparamvals.size();
        
        //if active, look for actual bounds
        if(newparam.isParameter("bounds")){
          std::string filename = newparam.get<string>("bounds");
          FILE* BoundsFile = fopen(filename.c_str(),"r");
          float a,b;
          int i = 0;
          while( !feof(BoundsFile) ) {
            char line[100] = "";
            fgets(line,100,BoundsFile);
            if( strcmp(line,"") ) {
              sscanf(line, "%f %f", &a, &b);
              lo[i] = a;
              up[i] = b;
            }
            i++;
          }
        }
      }
      else if (newparam.get<string>("usage") == "stochastic") {
        paramtypes.push_back(2);
        num_stochastic_params += newparamvals.size();
        for (size_t i=0; i<newparamvals.size(); i++) {
          stochastic_distribution.push_back(newparam.get<string>("distribution","uniform"));
          stochastic_mean.push_back(newparam.get<ScalarT>("mean",0.0));
          stochastic_variance.push_back(newparam.get<ScalarT>("variance",1.0));
          stochastic_min.push_back(newparam.get<ScalarT>("min",0.0));
          stochastic_max.push_back(newparam.get<ScalarT>("max",0.0));
        }
      }
      else if (newparam.get<string>("usage") == "discrete") {
        paramtypes.push_back(3);
        num_discrete_params += newparamvals.size();
      }
      else if (newparam.get<string>("usage") == "discretized") {
        paramtypes.push_back(4);
        num_discretized_params += 1;
        if (!discretized_stochastic) { // once this is turned on, it stays on
          discretized_stochastic = newparam.get<bool>("stochastic",false);
        }
        discretized_param_basis_types.push_back(newparam.get<string>("type","HGRAD"));
        discretized_param_basis_orders.push_back(newparam.get<int>("order",1));
        discretized_param_names.push_back(pl_itr->first);
        initialParamValues.push_back(newparam.get<ScalarT>("initial_value",1.0));
        lowerParamBounds.push_back(newparam.get<ScalarT>("lower_bound",-1.0));
        upperParamBounds.push_back(newparam.get<ScalarT>("upper_bound",1.0));
        discparam_distribution.push_back(newparam.get<string>("distribution","uniform"));
        discparamVariance.push_back(newparam.get<ScalarT>("variance",1.0));
        if (newparam.get<bool>("isDomainParam",true)) {
          domainRegTypes.push_back(newparam.get<int>("reg_type",0));
          domainRegConstants.push_back(newparam.get<ScalarT>("reg_constant",0.0));
          domainRegIndices.push_back(num_discretized_params - 1);
        }
        else {
          boundaryRegTypes.push_back(newparam.get<int>("reg_type",0));
          boundaryRegConstants.push_back(newparam.get<ScalarT>("reg_constant",0.0));
          boundaryRegSides.push_back(newparam.get<string>("sides"," "));
          boundaryRegIndices.push_back(num_discretized_params - 1);
        }
      }
      
      paramLowerBounds.push_back(lo);
      paramUpperBounds.push_back(up);
      
      pl_itr++;
    }
    
    TEUCHOS_TEST_FOR_EXCEPTION(num_active_params > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of parameters.");
    
    size_t maxcomp = 0;
    for (size_t k=0; k<paramvals.size(); k++) {
      if (paramvals[k].size() > maxcomp) {
        maxcomp = paramvals[k].size();
      }
    }
    
    paramvals_KVAD = Kokkos::View<AD**,AssemblyDevice>("parameter values (AD)", paramvals.size(), maxcomp);
    
  }
}
  
  // Set up the discretized parameter DOF manager
  
void ParameterManager::setupDiscretizedParameters(vector<vector<Teuchos::RCP<cell> > > & cells) {
  
  if (num_discretized_params > 0) {
    // determine the unique list of basis'
    vector<int> disc_orders;
    vector<string> disc_types;
    vector<int> disc_usebasis;
    
    for (size_t j=0; j<discretized_param_basis_orders.size(); j++) {
      bool is_unique = true;
      for (size_t k=0; k<disc_orders.size(); k++) {
        if (disc_orders[k] == discretized_param_basis_orders[j] &&
            disc_types[k] == discretized_param_basis_types[j]) {
          is_unique = false;
          disc_usebasis.push_back(k);
        }
      }
      if (is_unique) {
        disc_orders.push_back(discretized_param_basis_orders[j]);
        disc_types.push_back(discretized_param_basis_types[j]);
        disc_usebasis.push_back(disc_orders.size()-1);
      }
    }
    
    discretized_param_basis_types = disc_types;
    discretized_param_basis_orders = disc_orders;
    discretized_param_usebasis = disc_usebasis;
    
    for (size_t n=0; n<disc_orders.size(); n++) {
      topo_RCP cellTopo = mesh->getCellTopology(blocknames[0]);
      basis_RCP basis = DiscTools::getBasis(spaceDim, cellTopo, disc_types[n],
                                            disc_orders[n]);
      discretized_param_basis.push_back(basis);
      
    }
    
    paramDOF = Teuchos::rcp(new panzer::DOFManager<int,int>());
    Teuchos::RCP<panzer::ConnManager<int,int> > conn = Teuchos::rcp(new panzer_stk::STKConnManager<int>(mesh));
    paramDOF->setConnManager(conn,*(Comm->getRawMpiComm()));
    
    Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern;
    
    for (size_t b=0; b<blocknames.size(); b++) {
      for (size_t j=0; j<discretized_param_names.size(); j++) {
        
        Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(discretized_param_basis[disc_usebasis[j]]));
        paramDOF->addField(blocknames[b], discretized_param_names[j], Pattern);
      }
    }
    
    paramDOF->buildGlobalUnknowns();
    paramDOF->getOwnedIndices(paramOwned);
    numParamUnknowns = (int)paramOwned.size();
    paramDOF->getOwnedAndGhostedIndices(paramOwnedAndShared);
    numParamUnknownsOS = (int)paramOwnedAndShared.size();
    int localParamUnknowns = numParamUnknowns;
    
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localParamUnknowns,&globalParamUnknowns);
    //Comm->SumAll(&localParamUnknowns, &globalParamUnknowns, 1);
    
    for (size_t j=0; j<discretized_param_names.size(); j++) {
      int num = paramDOF->getFieldNum(discretized_param_names[j]);
      vector<int> poffsets = paramDOF->getGIDFieldOffsets(blocknames[0],num);
      paramoffsets.push_back(poffsets);
      paramNumBasis.push_back(discretized_param_basis[discretized_param_usebasis[j]]->getCardinality());
    }
    
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        vector<vector<int> > GIDs;
        int numElem = cells[b][e]->numElem;
        int numLocalDOF = 0;
        for (int p=0; p<numElem; p++) {
          size_t elemID = cells[b][e]->globalElemID(p);//disc->myElements[b][eprog+p];
          vector<int> localGIDs;
          paramDOF->getElementGIDs(elemID, localGIDs, blocknames[b]);
          GIDs.push_back(localGIDs);
          numLocalDOF = localGIDs.size(); // should be the same for all elements
        }
        Kokkos::View<GO**,HostDevice> hostGIDs("GIDs on host device",numElem,numLocalDOF);
        for (int i=0; i<numElem; i++) {
          for (int j=0; j<numLocalDOF; j++) {
            hostGIDs(i,j) = GIDs[i][j];
          }
        }
        cells[b][e]->paramGIDs = hostGIDs;
        cells[b][e]->setParamUseBasis(disc_usebasis, paramNumBasis);
      }
    }
    
    if (discretized_stochastic) { // add the param DOFs as indep. rv's
      for (size_t j=0; j<numParamUnknownsOS; j++) {
        // hard coding for one disc param just to get something working
        stochastic_distribution.push_back(discparam_distribution[0]);
        stochastic_mean.push_back(initialParamValues[0]);
        stochastic_variance.push_back(discparamVariance[0]);
        stochastic_min.push_back(lowerParamBounds[0]);
        stochastic_max.push_back(upperParamBounds[0]);
        
      }
    }
  }
  
  // Set up the discretized parameter linear algebra objects
  const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();
  
  if (num_discretized_params > 0) {
    
    param_owned_map = Teuchos::rcp(new LA_Map(INVALID, paramOwned, 0, Comm));
    param_overlapped_map = Teuchos::rcp(new LA_Map(INVALID, paramOwnedAndShared, 0, Comm));
    
    param_exporter = Teuchos::rcp(new LA_Export(param_overlapped_map, param_owned_map));
    param_importer = Teuchos::rcp(new LA_Import(param_overlapped_map, param_owned_map));
    
    Kokkos::View<GO**,HostDevice> gids;
    vector< vector<int> > param_nodesOS(numParamUnknownsOS); // should be overlapped
    vector< vector<int> > param_nodes(numParamUnknowns); // not overlapped -- for bounds
    vector< vector< vector<ScalarT> > > param_initial_vals; // custom initial guess set by assembler->cells
    DRV nodes;
    vector_RCP paramVec = this->setInitialParams(); // TMW: this will be deprecated soon
    
    int max_param_basis = 0;
    Kokkos::View<LO*,HostDevice> numDOF_KVhost("number of param DOF per variable",num_discretized_params);
    for (int k=0; k<num_discretized_params; k++) {
      numDOF_KVhost(k) = paramNumBasis[k];
      if (paramNumBasis[k]>max_param_basis){
        max_param_basis = paramNumBasis[k];
      }
    }
    Kokkos::View<LO*,AssemblyDevice> numDOF_KV = Kokkos::create_mirror_view(numDOF_KVhost);
    Kokkos::deep_copy(numDOF_KVhost, numDOF_KV);
    
    for (size_t b=0; b<cells.size(); b++) {
      //vector<vector<int> > curroffsets = phys->offsets[b];
      
      for(size_t e=0; e<cells[b].size(); e++) {
        gids = cells[b][e]->paramGIDs;
        // this should fail on the first iteration through if maxDerivs is not large enough
        TEUCHOS_TEST_FOR_EXCEPTION(gids.dimension(1) > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of parameter degrees of freedom per element.");
        
        int numElem = cells[b][e]->numElem;
        
        //vector<vector<vector<int> > > cellindices;
        Kokkos::View<LO***,AssemblyDevice> cellindices("Local DOF indices", numElem,
                                                       num_discretized_params, max_param_basis);
        
        for (int p=0; p<numElem; p++) {
          
          //vector<vector<int> > indices;
          
          for (int n=0; n<num_discretized_params; n++) {
            //vector<int> cindex;
            for( int i=0; i<paramNumBasis[n]; i++ ) {
              int globalIndexOS = param_overlapped_map->getLocalElement(gids(p,paramoffsets[n][i]));
              //cindex.push_back(globalIndexOS);
              cellindices(p,n,i) = globalIndexOS;
              param_nodesOS[n].push_back(globalIndexOS);
              int globalIndex_owned = param_owned_map->getLocalElement(gids(p,paramoffsets[n][i]));
              param_nodes[n].push_back(globalIndex_owned);
              
            }
            //indices.push_back(cindex);
          }
          //cellindices.push_back(indices);
        }
        cells[b][e]->setParamIndex(cellindices, numDOF_KV);
        /* // needs to be updated
         if (use_custom_initial_param_guess) {
         nodes = assembler->cells[b][e]->nodes;
         param_initial_vals = phys->udfunc->setInitialParams(nodes,cellindices);
         for (int p=0; p<numElem; p++) {
         for (int n = 0; n < num_discretized_params; n++) {
         for (int i = 0; i < cellindices[p][n].size(); i++) {
         paramVec->ReplaceGlobalValue(paramOwnedAndShared[cellindices[p][n][i]]
         ,0,param_initial_vals[p][n][i]);
         }
         }
         }
         }*/
      }
    }
    for (int n=0; n<num_discretized_params; n++) {
      std::sort(param_nodesOS[n].begin(), param_nodesOS[n].end());
      param_nodesOS[n].erase( std::unique(param_nodesOS[n].begin(),
                                          param_nodesOS[n].end()), param_nodesOS[n].end());
      
      std::sort(param_nodes[n].begin(), param_nodes[n].end());
      param_nodes[n].erase( std::unique(param_nodes[n].begin(),
                                        param_nodes[n].end()), param_nodes[n].end());
    }
    for (int n = 0; n < num_discretized_params; n++) {
      if (!use_custom_initial_param_guess) {
        for (size_t i = 0; i < param_nodesOS[n].size(); i++) {
          paramVec->replaceGlobalValue(paramOwnedAndShared[param_nodesOS[n][i]]
                                       ,0,initialParamValues[n]);
        }
      }
      paramNodesOS.push_back(param_nodesOS[n]); // store for later use
      paramNodes.push_back(param_nodes[n]); // store for later use
    }
    Psol.push_back(paramVec);
  }
  else {
    // set up a dummy parameter vector
    paramOwnedAndShared.push_back(0);
    param_overlapped_map = Teuchos::rcp(new LA_Map(INVALID, paramOwnedAndShared, 0, Comm));
    
    vector_RCP paramVec = this->setInitialParams(); // TMW: this will be deprecated soon
    Psol.push_back(paramVec);
  }
  
}
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int ParameterManager::getNumParams(const int & type) {
  int np = 0;
  if (type == 0)
  np = num_inactive_params;
  else if (type == 1)
  np = num_active_params;
  else if (type == 2)
  np = num_stochastic_params;
  else if (type == 3)
  np = num_discrete_params;
  else if (type == 4)
  np = globalParamUnknowns;
  
  return np;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int ParameterManager::getNumParams(const std::string & type) {
  int np = 0;
  if (type == "inactive")
  np = num_inactive_params;
  else if (type == "active")
  np = num_active_params;
  else if (type == "stochastic")
  np = num_stochastic_params;
  else if (type == "discrete")
  np = num_discrete_params;
  else if (type == "discretized")
  np = num_discretized_params;
  
  return np;
}

// ========================================================================================
// return the discretized parameters as vector for use with ROL
// ========================================================================================

vector<ScalarT> ParameterManager::getDiscretizedParamsVector() {
  int numParams = this->getNumParams(4);
  vector<ScalarT> discLocalParams(numParams);
  vector<ScalarT> discParams(numParams);
  auto Psol_2d = Psol[0]->getLocalView<HostDevice>();
  
  for (size_t i = 0; i < paramOwned.size(); i++) {
    int gid = paramOwned[i];
    discLocalParams[gid] = Psol_2d(i,0);
  }
  for (size_t i = 0; i < numParams; i++) {
    ScalarT globalval = 0.0;
    ScalarT localval = discLocalParams[i];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
    //Comm->SumAll(&localval, &globalval, 1);
    discParams[i] = globalval;
  }
  return discParams;
}

// ========================================================================================
// ========================================================================================

vector_RCP ParameterManager::setInitialParams() {
  vector_RCP initial = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
  ScalarT value = 2.0;
  initial->putScalar(value);
  return initial;
}

// ========================================================================================
// ========================================================================================

void ParameterManager::sacadoizeParams(const bool & seed_active) {
  
  //vector<vector<AD> > paramvals_AD;
  if (seed_active) {
    size_t pprog = 0;
    for (size_t i=0; i<paramvals.size(); i++) {
      vector<AD> currparams;
      if (paramtypes[i] == 1) { // active parameters
        for (size_t j=0; j<paramvals[i].size(); j++) {
          //currparams.push_back(Sacado::Fad::DFad<ScalarT>(num_active_params,pprog,paramvals[i][j]));
          paramvals_KVAD(i,j) = AD(maxDerivs,pprog,paramvals[i][j]);
          currparams.push_back(AD(maxDerivs,pprog,paramvals[i][j]));
          pprog++;
        }
      }
      else { // inactive, stochastic, or discrete parameters
        for (size_t j=0; j<paramvals[i].size(); j++) {
          //currparams.push_back(Sacado::Fad::DFad<ScalarT>(paramvals[i][j]));
          paramvals_KVAD(i,j) = AD(paramvals[i][j]);
          currparams.push_back(AD(paramvals[i][j]));
        }
      }
      *(paramvals_AD[i]) = currparams;
    }
  }
  else {
    size_t pprog = 0;
    for (size_t i=0; i<paramvals.size(); i++) {
      vector<AD> currparams;
      for (size_t j=0; j<paramvals[i].size(); j++) {
        //currparams.push_back(Sacado::Fad::DFad<ScalarT>(paramvals[i][j]));
        currparams.push_back(AD(paramvals[i][j]));
        paramvals_KVAD(i,j) = AD(paramvals[i][j]);
      }
      *(paramvals_AD[i]) = currparams;
    }
  }
  
  // TMW: these need to be depracated and removed
  phys->updateParameters(paramvals_AD, paramnames);
  
}

// ========================================================================================
// ========================================================================================

void ParameterManager::updateParams(const vector<ScalarT> & newparams, const int & type) {
  size_t pprog = 0;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        if (Comm->getRank() == 0 && verbosity > 0) {
          cout << "Updated Params: " << paramvals[i][j] << " (old value)   " << newparams[pprog] << " (new value)" << endl;
        }
        paramvals[i][j] = newparams[pprog];
        pprog++;
      }
    }
  }
  if ((type == 4) && (globalParamUnknowns > 0)) {
    int numClassicParams = this->getNumParams(1); // offset for ROL param vector
    for (size_t i = 0; i < paramOwnedAndShared.size(); i++) {
      int gid = paramOwnedAndShared[i];
      Psol[0]->replaceGlobalValue(gid,0,newparams[gid+numClassicParams]);
    }
  }
  if ((type == 2) && (globalParamUnknowns > 0)) {
    int numClassicParams = this->getNumParams(2); // offset for ROL param vector
    for (size_t i=0; i<paramOwnedAndShared.size(); i++) {
      int gid = paramOwnedAndShared[i];
      Psol[0]->replaceGlobalValue(gid,0,newparams[i+numClassicParams]);
    }
  }
}

// ========================================================================================
// ========================================================================================

void ParameterManager::updateParams(const vector<ScalarT> & newparams, const std::string & stype) {
  size_t pprog = 0;
  int type;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  if (stype == "inactive") { type = 0;}
  else if (stype == "active") { type = 1;}
  else if (stype == "stochastic") { type = 2;}
  else if (stype == "discrete") { type = 3;}
  else {
    //complain
  }
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        paramvals[i][j] = newparams[pprog];
        pprog++;
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

vector<ScalarT> ParameterManager::getParams(const int & type) {
  vector<ScalarT> reqparams;
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        reqparams.push_back(paramvals[i][j]);
      }
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

vector<string> ParameterManager::getParamsNames(const int & type) {
  vector<string> reqparams;
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      reqparams.push_back(paramnames[i]);
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

vector<size_t> ParameterManager::getParamsLengths(const int & type) {
  vector<size_t> reqparams;
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      reqparams.push_back(paramvals[i].size());
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

vector<ScalarT> ParameterManager::getParams(const std::string & stype) {
  vector<ScalarT> reqparams;
  int type;
  if (stype == "inactive")
  type = 0;
  else if (stype == "active")
  type = 1;
  else if (stype == "stochastic")
  type = 2;
  else if (stype == "discrete")
  type = 3;
  else
  //complain
  
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        reqparams.push_back(paramvals[i][j]);
      }
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

vector<vector<ScalarT> > ParameterManager::getParamBounds(const std::string & stype) {
  vector<vector<ScalarT> > reqbnds;
  vector<ScalarT> reqlo;
  vector<ScalarT> requp;
  int type;
  if (stype == "inactive") {type = 0;}
  else if (stype == "active") {type = 1;}
  else if (stype == "stochastic") {type = 2;}
  else if (stype == "discrete") {type = 3;}
  else if (stype == "discretized") {type = 4;}
  
  if (type == 0) {
    std::cout << "Bounds for inactive parameters are currently at default of (0,0)" << std::endl;
  }
  
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        reqlo.push_back(paramLowerBounds[i][j]);
        requp.push_back(paramUpperBounds[i][j]);
      }
    }
  }
  
  if (type == 4 && globalParamUnknowns > 0) {
    int numDiscParams = this->getNumParams(4);
    vector<ScalarT> rLocalLo(numDiscParams);
    vector<ScalarT> rLocalUp(numDiscParams);
    vector<ScalarT> rlo(numDiscParams);
    vector<ScalarT> rup(numDiscParams);
    int pindex;
    for (int n = 0; n < num_discretized_params; n++) {
      for (size_t i = 0; i < paramNodes[n].size(); i++) {
        int pnode = paramNodes[n][i];
        if (pnode >= 0) {
          int pindex = paramOwned[pnode];
          rLocalLo[pindex] = lowerParamBounds[n];
          rLocalUp[pindex] = upperParamBounds[n];
        }
      }
    }
    
    for (size_t i = 0; i < numDiscParams; i++) {
      
      ScalarT globalval = 0.0;
      ScalarT localval = rLocalLo[i];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      rlo[i] = globalval;
      
      globalval = 0.0;
      localval = rLocalUp[i];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      rup[i] = globalval;
    }
    
    reqlo = rlo;
    requp = rup;
    
  }
  
  reqbnds.push_back(reqlo);
  reqbnds.push_back(requp);
  return reqbnds;
}

// ========================================================================================
// ========================================================================================

void ParameterManager::stashParams(){
  if (batchID == 0 && Comm->getRank() == 0){
    string outname = "param_stash.dat";
    ofstream respOUT(outname);
    respOUT.precision(16);
    for (size_t i=0; i<paramvals.size(); i++) {
      if (paramtypes[i] == 1) {
        for (size_t j=0; j<paramvals[i].size(); j++) {
          respOUT << paramvals[i][j] << endl;
        }
      }
    }
    respOUT.close();
  }
}

// ========================================================================================
// ========================================================================================

vector<ScalarT> ParameterManager::getStochasticParams(const std::string & whichparam) {
  if (whichparam == "mean")
    return stochastic_mean;
  else if (whichparam == "variance")
    return stochastic_variance;
  else if (whichparam == "min")
    return stochastic_min;
  else if (whichparam == "max")
    return stochastic_max;
  else {
    vector<ScalarT> emptyvec;
    return emptyvec;
  }
}

// ========================================================================================
// ========================================================================================

vector<ScalarT> ParameterManager::getFractionalParams(const std::string & whichparam) {
  if (whichparam == "s-exponent")
    return s_exp;
  else if (whichparam == "mesh-resolution")
    return h_mesh;
  else {
    vector<ScalarT> emptyvec;
    return emptyvec;
  }
}

