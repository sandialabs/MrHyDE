/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setInitialParams() {
  
  debugger->print("**** Starting ParameterManager::setInitialParams ...");
  
  //auto Psol = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
  //auto Psol_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
  //Psol->putScalar(0.0);
  //Psol_over->putScalar(0.0); // TMW: why is this hard-coded???
  
  int numsols = 1;
  if (have_dynamic_discretized) {
    numsols = numTimeSteps;
  }
  
  for (int i=0; i<numsols; ++i) {
    vector_RCP dyninit = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
    vector_RCP dyninit_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
    dyninit->putScalar(0.0); // TMW: why is this hard-coded???
    dyninit_over->putScalar(0.0); // TMW: why is this hard-coded???
    discretized_params.push_back(dyninit);
    discretized_params_over.push_back(dyninit_over);
  }
  
  /*
  if (scalarInitialData) {
    // This will be done on the host for now
    auto initial_kv = initial->getLocalView<HostDevice>();
    for (size_t block=0; block<assembler->groups.size(); block++) {
      Kokkos::View<int**,AssemblyDevice> offsets = assembler->wkset[block]->offsets;
      auto host_offsets = Kokkos::create_mirror_view(offsets);
      Kokkos::deep_copy(host_offsets,offsets);
      for (size_t group=0; group<assembler->groups[block].size(); group++) {
        Kokkos::View<LO**,HostDevice> LIDs = assembler->groups[block][group]->LIDs_host;
        Kokkos::View<LO*,HostDevice> numDOF = assembler->groups[block][group]->group_data->numDOF_host;
        //parallel_for("solver initial scalar",RangePolicy<HostExec>(0,LIDs.extent(0)), MRHYDE_LAMBDA (const int e ) {
        for (int e=0; e<LIDs.extent(0); e++) {
          for (size_t n=0; n<numDOF.extent(0); n++) {
            for (size_t i=0; i<numDOF(n); i++ ) {
              initial_kv(LIDs(e,host_offsets(n,i)),0) = scalarInitialValues[block][n];
            }
          }
        }
      }
    }
  }
  else {
  
    initial->putScalar(0.0);
    
    vector_RCP glinitial = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    
    if (initial_type == "L2-projection") {
      
      // Compute the L2 projection of the initial data into the discrete space
      vector_RCP rhs = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // reset residual
      matrix_RCP mass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_overlapped_graph));//Tpetra::createCrsMatrix<ScalarT>(LA_overlapped_map); // reset Jacobian
      vector_RCP glrhs = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1)); // reset residual
      matrix_RCP glmass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_owned_map, maxEntries));//Tpetra::createCrsMatrix<ScalarT>(LA_owned_map); // reset Jacobian
      assembler->setInitial(rhs, mass, useadjoint);
      
      glmass->setAllToScalar(0.0);
      glmass->doExport(*mass, *exporter, Tpetra::ADD);
      
      glrhs->putScalar(0.0);
      glrhs->doExport(*rhs, *exporter, Tpetra::ADD);
      
      glmass->fillComplete();
      
      this->linearSolver(glmass, glrhs, glinitial);
      have_preconditioner = false; // resetting this because mass matrix may not have same connectivity as Jacobians
      initial->doImport(*glinitial, *importer, Tpetra::ADD);
      
    }
    else if (initial_type == "interpolation") {
      
      assembler->setInitial(initial, useadjoint);
      
    }
  }
  */
  
  debugger->print("**** Finished ParameterManager::setInitialParams ...");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setParam(const vector<ScalarT> & newparams, const std::string & name) {
  size_t pprog = 0;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  
  int index = 0;
  if (have_dynamic_scalar) {
    index = dynamic_timeindex;
  }
  
  if (paramvals.size() > index) {
    for (size_t i=0; i<paramvals[index].size(); i++) {
      if (paramnames[i] == name) {
        for (size_t j=0; j<paramvals[index][i].size(); j++) {
          if (Comm->getRank() == 0 && verbosity > 0) {
            cout << "Updated Params: " << paramvals[index][i][j] << " (old value)   " << newparams[pprog] << " (new value)" << endl;
          }
          paramvals[index][i][j] = newparams[pprog];
          pprog++;
        }
      }
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
bool ParameterManager<Node>::isParameter(const string & name) {
  bool isparam = false;
  if (paramvals.size() > 0) {
    for (size_t i=0; i<paramvals[0].size(); i++) { // just first is fine
      if (paramnames[i] == name) {
        isparam = true;
      }
    }
  }
  return isparam;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::stashParams(){
  if (batchID == 0 && Comm->getRank() == 0){
    string outname = "param_stash.dat";
    std::ofstream respOUT(outname);
    respOUT.precision(16);
    for (size_t i=0; i<paramvals.size(); i++) {
      for (size_t k=0; k<paramvals[i].size(); k++) {
        if (paramtypes[k] == 1) {
          for (size_t j=0; j<paramvals[i][k].size(); j++) {
            respOUT << paramvals[i][k][j] << endl;
          }
        }
      }
    }
    respOUT.close();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setParamMass(Teuchos::RCP<LA_MultiVector> diag,
                                          matrix_RCP mass) {
  
  paramMass = mass;
  diagParamMass = diag;
  
}
/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we can get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void ParameterManager<Node>::purgeMemory() {
  // nothing here  
}
