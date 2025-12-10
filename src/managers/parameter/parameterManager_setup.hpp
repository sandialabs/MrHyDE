/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
// Set up the parameters (inactive, active, stochastic, discrete)
// Communicate these parameters back to the physics interface and the enabled modules
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setupParameters() {
  
  debugger->print("**** Starting ParameterManager::setupParameters ... ");
  
  Teuchos::ParameterList parameters;
  vector<vector<ScalarT> > tmp_paramvals;
  if (settings->isSublist("Parameters")) {
    parameters = settings->sublist("Parameters");
    Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
    while (pl_itr != parameters.end()) {
      Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
      vector<ScalarT> newparamvals;
      if (!newparam.isParameter("type") || !newparam.isParameter("usage")) {
        // print out error message
      }
      
      if (newparam.get<string>("type") == "scalar") {
        newparamvals.push_back(newparam.get<ScalarT>("value"));
      }
      else if (newparam.get<string>("type") == "vector") {
        if (newparam.isParameter("number of components")) {
          
        }
        else {
          std::string filename = newparam.get<string>("source");
          std::ifstream fin(filename.c_str());
          std::istream_iterator<ScalarT> start(fin), end;
          vector<ScalarT> importedparamvals(start, end);
          for (size_t i=0; i<importedparamvals.size(); i++) {
            newparamvals.push_back(importedparamvals[i]);
          }
        }
      }
      
      paramnames.push_back(pl_itr->first);
      tmp_paramvals.push_back(newparamvals);
      
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
        bool is_dynamic = newparam.get<bool>("dynamic",false);
        scalar_param_dynamic.push_back(is_dynamic);
        if (is_dynamic) {
          have_dynamic_scalar = true;
        }
        
        //if active, look for actual bounds
        if (newparam.isParameter("bounds")) {
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
        if (newparam.get<string>("type") == "vector") {
          if (newparam.isParameter("mean source")) {
            std::string filename = newparam.get<string>("mean source");
            std::ifstream fin(filename.c_str());
            std::istream_iterator<ScalarT> start(fin), end;
            vector<ScalarT> importedparammean(start, end);
            for (size_t i=0; i<importedparammean.size(); i++) {
              stochastic_mean.push_back(importedparammean[i]);
            }
          }
          else {
            for (size_t i=0; i<newparamvals.size(); i++) {
              stochastic_mean.push_back(newparam.get<ScalarT>("mean",0.0));
            }
          }
          for (size_t i=0; i<newparamvals.size(); i++) {
            stochastic_distribution.push_back(newparam.get<string>("distribution","uniform"));
            stochastic_variance.push_back(newparam.get<ScalarT>("variance",1.0));
            stochastic_min.push_back(newparam.get<ScalarT>("min",-1.0));
            stochastic_max.push_back(newparam.get<ScalarT>("max",1.0));
          }
        }
        else {
          for (size_t i=0; i<newparamvals.size(); i++) {
            stochastic_distribution.push_back(newparam.get<string>("distribution","uniform"));
            stochastic_mean.push_back(newparam.get<ScalarT>("mean",0.0));
            stochastic_variance.push_back(newparam.get<ScalarT>("variance",1.0));
            stochastic_min.push_back(newparam.get<ScalarT>("min",-1.0));
            stochastic_max.push_back(newparam.get<ScalarT>("max",1.0));
          }
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
        
        bool is_dynamic = newparam.get<bool>("dynamic",false);
        discretized_param_dynamic.push_back(is_dynamic);
        if (is_dynamic) {
          have_dynamic_discretized = true;
        }
        
        initialParamValues.push_back(newparam.get<ScalarT>("initial_value",1.0));
        lowerParamBounds.push_back(newparam.get<ScalarT>("lower_bound",-1.0));
        upperParamBounds.push_back(newparam.get<ScalarT>("upper_bound",1.0));
        discparam_distribution.push_back(newparam.get<string>("distribution","uniform"));
        discparamVariance.push_back(newparam.get<ScalarT>("variance",1.0));
        
      }
      
      paramLowerBounds.push_back(lo);
      paramUpperBounds.push_back(up);
      
      pl_itr++;
    }
    
    if (have_dynamic_scalar) {
      for (size_t i=0; i<numTimeSteps; ++i) {
        paramvals.push_back(tmp_paramvals);
      }
    }
    else {
      paramvals.push_back(tmp_paramvals);
    }
    
#ifndef MrHyDE_NO_AD
    for (size_t block=0; block<blocknames.size(); ++block) {
      if ((int)num_active_params>disc->num_derivs_required[block]) {
        disc->num_derivs_required[block] = num_active_params;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(num_active_params > MAXDERIVS,std::runtime_error,"Error: MAXDERIVS is not large enough to support the number of parameters.");
#endif
    
    size_t maxcomp = 1;
    for (size_t k=0; k<paramvals.size(); k++) {
      for (size_t j=0; j<paramvals[k].size(); j++) {
        if (paramvals[k][j].size() > maxcomp) {
          maxcomp = paramvals[k][j].size();
        }
      }
    }
    
    size_t nump = 1;
    if (paramvals.size() > 0) {
      nump = paramvals[0].size();
    }
    paramvals_KV = Kokkos::View<ScalarT**,AssemblyDevice>("parameter values (ScalarT)", nump, maxcomp);
#ifndef MrHyDE_NO_AD
    paramvals_KVAD = Kokkos::View<AD**,AssemblyDevice>("parameter values (AD)", nump, maxcomp);
    paramvals_KVAD2 = Kokkos::View<AD2**,AssemblyDevice>("parameter values (AD2)", nump, maxcomp);
    paramvals_KVAD4 = Kokkos::View<AD4**,AssemblyDevice>("parameter values (AD4)", nump, maxcomp);
    paramvals_KVAD8 = Kokkos::View<AD8**,AssemblyDevice>("parameter values (AD8)", nump, maxcomp);
    paramvals_KVAD16 = Kokkos::View<AD16**,AssemblyDevice>("parameter values (AD16)", nump, maxcomp);
    paramvals_KVAD18 = Kokkos::View<AD18**,AssemblyDevice>("parameter values (AD18)", nump, maxcomp);
    paramvals_KVAD24 = Kokkos::View<AD24**,AssemblyDevice>("parameter values (AD24)", nump, maxcomp);
    paramvals_KVAD32 = Kokkos::View<AD32**,AssemblyDevice>("parameter values (AD32)", nump, maxcomp);
#endif
    
    int numind = 1;
    if (have_dynamic_scalar) {
      numind = numTimeSteps;
    }
    paramvals_KV_ALL = Kokkos::View<ScalarT***,AssemblyDevice>("parameter values (ScalarT)", numind, nump, maxcomp);
#ifndef MrHyDE_NO_AD
    paramvals_KVAD_ALL = Kokkos::View<AD***,AssemblyDevice>("parameter values (AD)", numind, nump, maxcomp);
    paramvals_KVAD2_ALL = Kokkos::View<AD2***,AssemblyDevice>("parameter values (AD2)", numind, nump, maxcomp);
    paramvals_KVAD4_ALL = Kokkos::View<AD4***,AssemblyDevice>("parameter values (AD4)", numind, nump, maxcomp);
    paramvals_KVAD8_ALL = Kokkos::View<AD8***,AssemblyDevice>("parameter values (AD8)", numind, nump, maxcomp);
    paramvals_KVAD16_ALL = Kokkos::View<AD16***,AssemblyDevice>("parameter values (AD16)", numind, nump, maxcomp);
    paramvals_KVAD18_ALL = Kokkos::View<AD18***,AssemblyDevice>("parameter values (AD18)", numind, nump, maxcomp);
    paramvals_KVAD24_ALL = Kokkos::View<AD24***,AssemblyDevice>("parameter values (AD24)", numind, nump, maxcomp);
    paramvals_KVAD32_ALL = Kokkos::View<AD32***,AssemblyDevice>("parameter values (AD32)", numind, nump, maxcomp);
#endif
    
    paramvals_KV = Kokkos::View<ScalarT**,AssemblyDevice>("parameter values (ScalarT)", nump, maxcomp);
#ifndef MrHyDE_NO_AD
    paramvals_KVAD = Kokkos::View<AD**,AssemblyDevice>("parameter values (AD)", nump, maxcomp);
    paramvals_KVAD2 = Kokkos::View<AD2**,AssemblyDevice>("parameter values (AD2)", nump, maxcomp);
    paramvals_KVAD4 = Kokkos::View<AD4**,AssemblyDevice>("parameter values (AD4)", nump, maxcomp);
    paramvals_KVAD8 = Kokkos::View<AD8**,AssemblyDevice>("parameter values (AD8)", nump, maxcomp);
    paramvals_KVAD16 = Kokkos::View<AD16**,AssemblyDevice>("parameter values (AD16)", nump, maxcomp);
    paramvals_KVAD18 = Kokkos::View<AD18**,AssemblyDevice>("parameter values (AD18)", nump, maxcomp);
    paramvals_KVAD24 = Kokkos::View<AD24**,AssemblyDevice>("parameter values (AD24)", nump, maxcomp);
    paramvals_KVAD32 = Kokkos::View<AD32**,AssemblyDevice>("parameter values (AD32)", nump, maxcomp);
#endif
  }
  
  debugger->print("**** Finished ParameterManager::setupParameters");
  
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
// Set up the discretized parameter DOF manager
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void ParameterManager<Node>::setupDiscretizedParameters(vector<vector<Teuchos::RCP<Group> > > & groups,
                                                        vector<vector<Teuchos::RCP<BoundaryGroup> > > & boundary_groups) {
  
  debugger->print("**** Starting ParameterManager::setupDiscretizedParameters ... ");
  
  if (num_discretized_params > 0) {
    // determine the unique list of basis'
    
    vector<int> disc_orders = phys->unique_orders[0];
    vector<string> disc_types = phys->unique_types[0];
    vector<int> disc_usebasis;
    
    for (size_t j=0; j<discretized_param_basis_orders.size(); j++) {
      for (size_t k=0; k<disc_orders.size(); k++) {
        if (discretized_param_basis_orders[j] == disc_orders[k] && discretized_param_basis_types[j] == disc_types[k]) {
          disc_usebasis.push_back(k);
        }
      }
    }
    
    discretized_param_basis_types = disc_types;
    discretized_param_basis_orders = disc_orders;
    discretized_param_usebasis = disc_usebasis;
    
    discretized_param_basis = disc->basis_pointers[0];
    
    paramDOF = Teuchos::rcp(new panzer::DOFManager());
    Teuchos::RCP<panzer::ConnManager> conn = mesh->getSTKConnManager();
    paramDOF->setConnManager(conn,*(Comm->getRawMpiComm()));
    
    Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern;
    
    for (size_t block=0; block<blocknames.size(); ++block) {
      auto cellTopo = mesh->getCellTopology(blocknames[block]);
      for (size_t j=0; j<discretized_param_names.size(); j++) {
        basis_RCP cbasis = disc->getBasis(spaceDim, cellTopo,
                                          disc_types[disc_usebasis[j]],
                                          disc_orders[disc_usebasis[j]]);
        
        Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(cbasis));
        paramDOF->addField(blocknames[block], discretized_param_names[j], Pattern);
      }
    }
    
    paramDOF->buildGlobalUnknowns();
#ifndef MrHyDE_NO_AD
    for (size_t block=0; block<blocknames.size(); ++block) {
      int numGIDs = paramDOF->getElementBlockGIDCount(blocknames[block]);
      if (numGIDs > disc->num_derivs_required[block]) {
        disc->num_derivs_required[block] = numGIDs;
      } 
    
      TEUCHOS_TEST_FOR_EXCEPTION(numGIDs > MAXDERIVS,std::runtime_error,
                                 "Error: MAXDERIVS is not large enough to support the number of discretized parameter degrees of freedom per element on block: " + blocknames[block]);
    }
#endif
    paramDOF->getOwnedIndices(paramOwned);
    numParamUnknowns = (int)paramOwned.size();
    paramDOF->getOwnedAndGhostedIndices(paramOwnedAndShared);
    numParamUnknownsOS = (int)paramOwnedAndShared.size();
    int localParamUnknowns = numParamUnknowns;
    
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localParamUnknowns,&globalParamUnknowns);
    //Comm->SumAll(&localParamUnknowns, &globalParamUnknowns, 1);
    
    for (size_t j=0; j<discretized_param_names.size(); j++) {
      int num = paramDOF->getFieldNum(discretized_param_names[j]);
      vector<int> poffsets = paramDOF->getGIDFieldOffsets(blocknames[0],num); // same for all blocks?
      paramoffsets.push_back(poffsets);
      paramNumBasis.push_back(discretized_param_basis[discretized_param_usebasis[j]]->getCardinality());
    }
    
    Kokkos::View<const LO**,Kokkos::LayoutRight,PHX::Device> LIDs = paramDOF->getLIDs();
        
    for (size_t block=0; block<groups.size(); ++block) {
      if (groups[block].size() > 0) {
        int numLocalDOF = 0;
        Kokkos::View<LO*,AssemblyDevice> numDOF_KV("number of param DOF per variable",num_discretized_params);
        for (size_t k=0; k<num_discretized_params; k++) {
          numDOF_KV(k) = paramNumBasis[k];
          numLocalDOF += paramNumBasis[k];
        }
        groups[block][0]->group_data->num_param_dof = numDOF_KV;
        Kokkos::View<LO*,HostDevice> numDOF_host("numDOF on host",num_discretized_params);
        Kokkos::deep_copy(numDOF_host, numDOF_KV);
        groups[block][0]->group_data->num_param_dof_host = numDOF_host;
        
        auto myElem = disc->my_elements[block];
        Kokkos::View<size_t*,AssemblyDevice> GEIDs("element IDs on device",myElem.size());
        auto host_GEIDs = Kokkos::create_mirror_view(GEIDs);
        for (size_t elem=0; elem<myElem.extent(0); elem++) {
          host_GEIDs(elem) = myElem(elem);
        }
        Kokkos::deep_copy(GEIDs, host_GEIDs);
        
        for (size_t grp=0; grp<groups[block].size(); ++grp) {
          LIDView groupLIDs("parameter LIDs",groups[block][grp]->numElem, LIDs.extent(1));
          Kokkos::View<LO*,AssemblyDevice> EIDs = groups[block][grp]->localElemID;
          parallel_for("paramman copy LIDs",
                       RangePolicy<AssemblyExec>(0,groupLIDs.extent(0)), 
                       KOKKOS_CLASS_LAMBDA (const int c ) {
            size_t elemID = GEIDs(EIDs(c));
            for (size_type j=0; j<LIDs.extent(1); j++) {
              groupLIDs(c,j) = LIDs(elemID,j);
            }
          });
          groups[block][grp]->setParams(groupLIDs);
          groups[block][grp]->setParamUseBasis(disc_usebasis, paramNumBasis);
        }
      }
    }
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      if (boundary_groups[block].size() > 0) {
        int numLocalDOF = 0;
        for (size_t k=0; k<num_discretized_params; k++) {
          numLocalDOF += paramNumBasis[k];
        }
        
        for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
          LIDView groupLIDs("parameter LIDs",boundary_groups[block][grp]->numElem, LIDs.extent(1));
          Kokkos::View<LO*,AssemblyDevice> EIDs = boundary_groups[block][grp]->localElemID;
          parallel_for("paramman copy LIDs bgroups",
                       RangePolicy<AssemblyExec>(0,groupLIDs.extent(0)), 
                       KOKKOS_CLASS_LAMBDA (const int e ) {
            size_t elemID = EIDs(e);
            for (size_type j=0; j<LIDs.extent(1); j++) {
              groupLIDs(e,j) = LIDs(elemID,j);
            }
          });
          boundary_groups[block][grp]->setParams(groupLIDs);
          boundary_groups[block][grp]->setParamUseBasis(disc_usebasis, paramNumBasis);
        }
      }
    }
    if (discretized_stochastic) { // add the param DOFs as indep. rv's
      for (int j=0; j<numParamUnknownsOS; j++) {
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
  
  if (num_discretized_params > 0) {
    
    GO localNumUnknowns = paramOwned.size();
    
    GO globalNumUnknowns = 0;
    Teuchos::reduceAll<LO,GO>(*Comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
    
    param_owned_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, paramOwned, 0, Comm));
    param_overlapped_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, paramOwnedAndShared, 0, Comm));
    
    param_exporter = Teuchos::rcp(new LA_Export(param_overlapped_map, param_owned_map));
    param_importer = Teuchos::rcp(new LA_Import(param_owned_map, param_overlapped_map));
    
    //vector< vector<int> > param_nodesOS(numParamUnknownsOS); // should be overlapped
    //vector< vector<int> > param_nodes(numParamUnknowns); // not overlapped -- for bounds
    vector< vector< vector<ScalarT> > > param_initial_vals; // custom initial guess set by assembler->groups
    
    this->setInitialParams(); 
    
    vector<vector<GO> > param_dofs;//(num_discretized_params);
    vector<vector<GO> > param_dofs_OS;//(num_discretized_params);
    for (size_t num=0; num<num_discretized_params; num++) {
      vector<GO> dofs, dofs_OS;
      param_dofs.push_back(dofs);
      param_dofs_OS.push_back(dofs_OS);
    }
    
    for (size_t block=0; block<blocknames.size(); ++block) {
      auto EIDs = disc->my_elements[block];
      for (size_t e=0; e<EIDs.extent(0); e++) {
        vector<GO> gids;
        size_t elemID = EIDs(e);
        paramDOF->getElementGIDs(elemID, gids, blocknames[block]);
        
        for (size_t num=0; num<num_discretized_params; num++) {
          vector<int> var_offsets = paramDOF->getGIDFieldOffsets(blocknames[block],num);
          for (size_t dof=0; dof<var_offsets.size(); dof++) {
            param_dofs_OS[num].push_back(gids[var_offsets[dof]]);
          }
        }
      }
    }
    
    for (size_t n = 0; n < num_discretized_params; n++) {
      if (!use_custom_initial_param_guess) {
        for (size_t i = 0; i < param_dofs_OS[n].size(); i++) {
          //if (have_dynamic_discretized) {
            for (size_t j=0; j<discretized_params_over.size(); ++j) {
              discretized_params_over[j]->replaceGlobalValue(param_dofs_OS[n][i],0,initialParamValues[n]);
            }
          //}
          //discretized_params_over->replaceGlobalValue(param_dofs_OS[n][i],0,initialParamValues[n]);
        }
      }
      paramNodesOS.push_back(param_dofs_OS[n]); // store for later use
      paramNodes.push_back(param_dofs[n]); // store for later use
    }
    //discretized_params->doExport(*discretized_params_over, *param_exporter, Tpetra::REPLACE);
    //if (have_dynamic_discretized) {
      for (size_t i=0; i<discretized_params_over.size(); ++i) {
        discretized_params[i]->doExport(*(discretized_params_over[i]), *param_exporter, Tpetra::REPLACE);
      }
    //}
  }
  else {
    // set up a dummy parameter vector
    paramOwnedAndShared.push_back(0);
    paramOwned.push_back(0);
    const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();
    
    param_overlapped_map = Teuchos::rcp(new LA_Map(INVALID, paramOwnedAndShared, 0, Comm));
    param_owned_map = Teuchos::rcp(new LA_Map(INVALID, paramOwned, 0, Comm));

    this->setInitialParams(); 
  }
  
  debugger->print("**** Finished ParameterManager::setupDiscretizedParameters");
  
}
