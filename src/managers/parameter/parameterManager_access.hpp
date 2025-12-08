/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
int ParameterManager<Node>::getNumParams(const int & type) {
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

template<class Node>
int ParameterManager<Node>::getNumParams(const std::string & type) {
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
// return the discretized parameters as vector for use with ROL (deprecated)
// ========================================================================================

template<class Node>
vector<ScalarT> ParameterManager<Node>::getDiscretizedParamsVector() {
  int numDParams = this->getNumParams(4);
  vector<ScalarT> discLocalParams(numDParams);
  vector<ScalarT> discParams(numDParams);
  auto Psol_2d = discretized_params[0]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto Psol_host = Kokkos::create_mirror_view(Psol_2d);
  for (size_t i = 0; i < paramOwned.size(); i++) {
    int gid = paramOwned[i];
    discLocalParams[gid] = Psol_host(i,0);
    //cout << gid << " " << Psol_2d(i,0) << endl;
  }
  for (int i = 0; i < numDParams; i++) {
    ScalarT globalval = 0.0;
    ScalarT localval = discLocalParams[i];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
    //Comm->SumAll(&localval, &globalval, 1);
    discParams[i] = globalval;
    //cout << i << " " << globalval << "  " << localval << endl;
    
  }
  
  return discParams;
}

// ========================================================================================
// return the discretized parameters as vector of vector_RCPs for use with ROL
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > ParameterManager<Node>::getDiscretizedParams() {
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > cparams;
  int index = 0;
  if (have_dynamic_discretized) {
    index = dynamic_timeindex;
  }
  if (discretized_params.size() > index) {
    cparams = discretized_params[index];
  }
  return cparams;
}

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > ParameterManager<Node>::getDiscretizedParamsOver() {
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > cparams;
  int index = 0;
  if (have_dynamic_discretized) {
    index = dynamic_timeindex;
  }
  
  if (discretized_params_over.size() > index) {
    cparams = discretized_params_over[index];
  }
  return cparams;
}

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > ParameterManager<Node>::getDiscretizedParamsDot() {
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > cparamsdot = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
  cparamsdot->putScalar(0.0);
  
  int index = 0;
  if (have_dynamic_discretized) {
    index = dynamic_timeindex;
  }
  
  if (index > 0 && discretized_params.size() > index+1) {
    cparamsdot->update(1.0/dynamic_dt,*(discretized_params[index]),1.0);
    cparamsdot->update(-1.0/dynamic_dt,*(discretized_params[index-1]),1.0);
  }
  return cparamsdot;
}

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > ParameterManager<Node>::getDiscretizedParamsDotOver() {
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > cparamsdot = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
  
  cparamsdot->putScalar(0.0);
  
  int index = 0;
  if (have_dynamic_discretized) {
    index = dynamic_timeindex;
  }
  
  if (index > 0 && discretized_params_over.size() > index+1) {
    cparamsdot->update(1.0/dynamic_dt,*(discretized_params_over[index]),1.0);
    cparamsdot->update(-1.0/dynamic_dt,*(discretized_params_over[index-1]),1.0);
  }
  return cparamsdot;
}

// ========================================================================================
// ========================================================================================

template<class Node>
MrHyDE_OptVector ParameterManager<Node>::getCurrentVector() {
  
  Teuchos::TimeMonitor localtimer(*getcurrenttimer);

  vector<Teuchos::RCP<vector<ScalarT> > > new_active_params;
  vector<vector_RCP> new_disc_params;
  if (num_active_params > 0) {
    new_active_params = this->getParams(1);
  }
  //else {
    //new_active_params = Teuchos::null;
  //}
  if (globalParamUnknowns > 0) {
    //if (have_dynamic_discretized) {
      new_disc_params = discretized_params;
    //}
    //else {
    //  new_disc_params.push_back(Psol);
    //}
  }
    
  MrHyDE_OptVector newvec(new_disc_params, new_active_params, 1.0, diagParamMass, paramMass, Comm->getRank());
  return newvec;
}

// ========================================================================================
// return the discretized parameters as vector of vector_RCPs for use with ROL
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > ParameterManager<Node>::getDynamicDiscretizedParams() {
  return discretized_params;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<vector<ScalarT> > > ParameterManager<Node>::getParams(const int & type) {
  vector<Teuchos::RCP<vector<ScalarT> > > reqparams;// = Teuchos::rcp(new std::vector<ScalarT>());
  for (size_t i=0; i<paramvals.size(); i++) {
    Teuchos::RCP<vector<ScalarT> > tmpparams = Teuchos::rcp(new std::vector<ScalarT>());
    for (size_t k=0; k<paramvals[i].size(); k++) {
      if (paramtypes[k] == type) {
        for (size_t j=0; j<paramvals[i][k].size(); j++) {
          tmpparams->push_back(paramvals[i][k][j]);
        }
      }
    }
    reqparams.push_back(tmpparams);
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<string> ParameterManager<Node>::getParamsNames(const int & type) {
  vector<string> reqparams;
  if (paramvals.size() > 0) {
    for (size_t i=0; i<paramvals[0].size(); i++) { // just first is fine
      if (paramtypes[i] == type) {
        reqparams.push_back(paramnames[i]);
      }
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<size_t> ParameterManager<Node>::getParamsLengths(const int & type) {
  vector<size_t> reqparams;
  if (paramvals.size() > 0) {
    for (size_t i=0; i<paramvals[0].size(); i++) { // first is fine
      if (paramtypes[i] == type) {
        reqparams.push_back(paramvals[0][i].size());
      }
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<ScalarT> ParameterManager<Node>::getParams(const std::string & stype) {
  vector<ScalarT> reqparams;
  int type = -1;
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
  cout << "Error in parameterManager::getParams: input stype is not valid" << std::endl;

  for (size_t i=0; i<paramvals.size(); i++) {
    for (size_t k=0; k<paramvals[i].size(); k++) {
      if (paramtypes[k] == type) {
        for (size_t j=0; j<paramvals[i][k].size(); j++) {
          reqparams.push_back(paramvals[i][k][j]);
        }
      }
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<ScalarT> ParameterManager<Node>::getParams(const std::string & stype, int dynamic_index) {
  vector<ScalarT> reqparams;
  int type = -1;
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
  cout << "Error in parameterManager::getParams: input stype is not valid" << std::endl;

  for (size_t k=0; k<paramvals[dynamic_index].size(); k++) {
    if (paramtypes[k] == type) {
      for (size_t j=0; j<paramvals[dynamic_index][k].size(); j++) {
        reqparams.push_back(paramvals[dynamic_index][k][j]);
      }
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<vector<ScalarT> > > ParameterManager<Node>::getActiveParamBounds() {
  vector<Teuchos::RCP<vector<ScalarT> > > reqbnds;
  Teuchos::RCP<vector<ScalarT> > reqlo = Teuchos::rcp( new vector<ScalarT> (num_active_params, 0.0) );
  Teuchos::RCP<vector<ScalarT> > requp = Teuchos::rcp( new vector<ScalarT> (num_active_params, 0.0) );

  size_t prog = 0;
  for (size_t i=0; i<paramvals[0].size(); i++) {
    if (paramtypes[i] == 1) {
      for (size_t j=0; j<paramvals[0][i].size(); j++) {
        (*reqlo)[prog] = paramLowerBounds[i][j];
        (*requp)[prog] = paramUpperBounds[i][j];
        prog++;
      }
    }
  }
  reqbnds.push_back(reqlo);
  reqbnds.push_back(requp);

  return reqbnds;

}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > ParameterManager<Node>::getDiscretizedParamBounds() {

  vector<vector_RCP> reqbnds;

  vector_RCP lower, upper, lower_over, upper_over;

  if (globalParamUnknowns > 0) {

    lower = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
    lower_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
    lower->putScalar(0.0); 
    lower_over->putScalar(0.0);
    
    upper = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
    upper_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
    upper->putScalar(0.0); 
    upper_over->putScalar(0.0); // TMW: why is this hard-coded??? 
  
    int numDiscParams = this->getNumParams(4);
    vector<ScalarT> rLocalLo(numDiscParams);
    vector<ScalarT> rLocalUp(numDiscParams);
    vector<ScalarT> rlo(numDiscParams);
    vector<ScalarT> rup(numDiscParams);
    for (size_t n = 0; n < num_discretized_params; n++) {
      for (size_t i = 0; i < paramNodesOS[n].size(); i++) {
        //int pnode = paramNodesOS[n][i];
        lower_over->replaceGlobalValue(paramNodesOS[n][i],0,lowerParamBounds[n]);
        upper_over->replaceGlobalValue(paramNodesOS[n][i],0,upperParamBounds[n]);
        //if (pnode >= 0) {
        //  int pindex = paramOwned[pnode];
        //  rLocalLo[pindex] = lowerParamBounds[n];
        //  rLocalUp[pindex] = upperParamBounds[n];
        //}
      }
    }
    
    lower->doExport(*lower_over, *param_exporter, Tpetra::REPLACE);
    upper->doExport(*upper_over, *param_exporter, Tpetra::REPLACE);

    /*
    for (int i = 0; i < numDiscParams; i++) {
      
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
    */
  }
  
  reqbnds.push_back(lower);
  reqbnds.push_back(upper);
  return reqbnds;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<ScalarT> ParameterManager<Node>::getStochasticParams(const std::string & whichparam) {
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

template<class Node>
vector<ScalarT> ParameterManager<Node>::getFractionalParams(const std::string & whichparam) {
  if (whichparam == "s-exponent")
    return s_exp;
  else if (whichparam == "mesh-resolution")
    return h_mesh;
  else {
    vector<ScalarT> emptyvec;
    return emptyvec;
  }
}
