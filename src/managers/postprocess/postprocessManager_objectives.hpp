/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::addObjectiveFunctions(Teuchos::ParameterList &obj_funs,
                                                     const size_t &block)
{
  
  // obj_funs is a sublist of settings that just contains the objective functions
  // Note that objective functions can be defined only on certain blocks
  
  Teuchos::ParameterList::ConstIterator obj_itr = obj_funs.begin();
  while (obj_itr != obj_funs.end())
  {
    Teuchos::ParameterList objsettings = obj_funs.sublist(obj_itr->first);
    
    // Determine if we need to add this obj fun on this block
    bool addobj = true;
    if (objsettings.isParameter("blocks"))
    {
      string blocklist = objsettings.get<string>("blocks");
      std::size_t found = blocklist.find(blocknames[block]);
      if (found == std::string::npos)
      {
        addobj = false;
      }
    }
    
    // If so, then add it and the necessary functions to the function manager
    if (addobj)
    {
      objective newobj(objsettings, obj_itr->first, block);
      objectives.push_back(newobj);
      
      if (newobj.type == "sensors")
      {
        assembler->addFunction(block, newobj.name + " response", newobj.response, "point");
      }
      else if (newobj.type == "integrated response")
      {
        assembler->addFunction(block, newobj.name + " response", newobj.response, "ip");
      }
      else if (newobj.type == "integrated control")
      {
        assembler->addFunction(block, newobj.name, newobj.function, "ip");
      }
      
      // Each objective can be associated with various types of regularizations
      for (size_t r = 0; r < newobj.regularizations.size(); ++r)
      {
        if (newobj.regularizations[r].type == "integrated")
        {
          if (newobj.regularizations[r].location == "volume")
          {
            assembler->addFunction(block, newobj.regularizations[r].name, newobj.regularizations[r].function, "ip");
          }
          else if (newobj.regularizations[r].location == "boundary")
          {
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

template <class Node>
void PostprocessManager<Node>::computeObjective(vector<vector_RCP> &current_soln,
                                                const ScalarT &current_time) {
  
  Teuchos::TimeMonitor localtimer(*objectiveTimer);
  
  debugger->print(1, "******** Starting PostprocessManager::computeObjective ...");
  
  typedef typename Node::execution_space LA_exec;
  typedef typename Node::device_type LA_device;
  
  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }
  
  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> sol_kv;
  for (size_t s = 0; s < current_soln.size(); ++s) {
    auto vec_kv = current_soln[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
    if (data_avail) {
      sol_kv.push_back(vec_slice);
    }
    else {
      auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(), vec_slice);
      Kokkos::deep_copy(vec_dev, vec_slice);
      sol_kv.push_back(vec_dev);
    }
  }
  
  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> params_kv;
  
  auto Psol = params->getDiscretizedParamsOver();
  auto p_kv = Psol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto pslice = Kokkos::subview(p_kv, Kokkos::ALL(), 0);
  
  if (data_avail) {
    params_kv.push_back(pslice);
  }
  else {
    auto p_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(), pslice);
    Kokkos::deep_copy(p_dev, pslice);
    params_kv.push_back(p_dev);
  }
  
  int numParams = params->num_active_params + params->globalParamUnknowns;
  
  // Objective function values
  vector<ScalarT> totaldiff(objectives.size(), 0.0);
  
  for (size_t r = 0; r < objectives.size(); ++r)
  {
    if (objectives[r].type == "integrated control")
    {
      
      size_t block = objectives[r].block;
      
      for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
      {
        
        View_Sc1 objsum_dev("obj func sum as scalar on device", numParams + 1);
        
        auto wts = assembler->groups[block][grp]->wts;
        
        if (!assembler->groups[block][grp]->have_sols)
        {
          for (size_t set = 0; set < sol_kv.size(); ++set)
          {
            assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
          }
          assembler->performGather(0, block, grp, params_kv[0], 4, 0);
        }
        assembler->updateWorkset(block, grp, 0, 0, true);
        
        auto obj_dev = assembler->function_managers[block]->evaluate(objectives[r].name, "ip");
        
        Kokkos::View<ScalarT[1], AssemblyDevice> objsum("sum of objective");
        parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
          ScalarT tmpval = 0.0;
          for (size_type pt=0; pt<wts.extent(1); pt++) {
            tmpval += obj_dev(elem,pt)*wts(elem,pt);
          }
          Kokkos::atomic_add(&(objsum(0)),tmpval); });
        
        parallel_for("grp objective", RangePolicy<AssemblyExec>(0, objsum_dev.extent(0)), MRHYDE_LAMBDA(const size_type p) {
          if (p==0) {
            objsum_dev(p) = objsum(0);
          } });
        
        auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
        Kokkos::deep_copy(objsum_host, objsum_dev);
        
        // Update the objective function value
        totaldiff[r] += objectives[r].weight * objsum_host(0);
      }
    }
    else if (objectives[r].type == "discrete control")
    {
      for (size_t set = 0; set < current_soln.size(); ++set)
      {
        vector_RCP D_soln;
        bool fnd = datagen_soln[set]->extract(D_soln, 0, current_time);
        if (fnd)
        {
          vector_RCP diff = linalg->getNewVector(set);
          vector_RCP F_no = linalg->getNewVector(set);
          vector_RCP D_no = linalg->getNewVector(set);
          F_no->doExport(*(current_soln[set]), *(linalg->exporter[set]), Tpetra::REPLACE);
          D_no->doExport(*D_soln, *(linalg->exporter[set]), Tpetra::REPLACE);
          
          diff->update(1.0, *F_no, 0.0);
          diff->update(-1.0, *D_no, 1.0);
          Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> obj(1);
          diff->norm2(obj);
          if (Comm->getRank() == 0)
          {
            totaldiff[r] += objectives[r].weight * obj[0] * obj[0];
          }
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: did not find a data-generating solution");
        }
      }
    }
    else if (objectives[r].type == "integrated response")
    {
      
      size_t block = objectives[r].block;
      
      ScalarT intresp = 0.0;
      
      for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
      {
        
        auto wts = assembler->groups[block][grp]->wts;
        
        if (!assembler->groups[block][grp]->have_sols)
        {
          for (size_t set = 0; set < sol_kv.size(); ++set)
          {
            assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
          }
          assembler->performGather(0, block, grp, params_kv[0], 4, 0);
        }
        
        assembler->updateWorkset(block, grp, 0, 0, true);
        
        auto obj_dev = assembler->function_managers[block]->evaluate(objectives[r].name + " response", "ip");
        
        Kokkos::View<ScalarT[1], AssemblyDevice> objsum("sum of objective");
        parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
          ScalarT tmpval = 0.0;
          for (size_type pt=0; pt<wts.extent(1); pt++) {
            tmpval += obj_dev(elem,pt)*wts(elem,pt);
          }
          Kokkos::atomic_add(&(objsum(0)),tmpval); });
        
        View_Sc1 objsum_dev("obj func sum as scalar on device", numParams + 1);
        
        parallel_for("grp objective", RangePolicy<AssemblyExec>(0, objsum_dev.extent(0)), MRHYDE_LAMBDA(const size_type p) {
          if (p==0) {
            objsum_dev(p) = objsum(0);
          } });
        
        auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
        Kokkos::deep_copy(objsum_host, objsum_dev);
        
        // Update the objective function value
        intresp += objsum_host(0);
      }
      
      totaldiff[r] += intresp;
      
      if (compute_response)
      {
        if (objectives[r].save_data)
        {
          objectives[r].response_times.push_back(current_time);
          objectives[r].scalar_response_data.push_back(totaldiff[r]);
          if (verbosity >= 10)
          {
            double globalval = 0.0;
            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &intresp, &globalval);
            if (Comm->getRank() == 0)
            {
              cout << objectives[r].name << " on block " << blocknames[objectives[r].block] << ": " << globalval << endl;
            }
          }
        }
      }
      //}
    }
    else if (objectives[r].type == "sensors" || objectives[r].type == "sensor response" || objectives[r].type == "pointwise response")
    {
      
      if (objectives[r].compute_sensor_soln || objectives[r].compute_sensor_average_soln)
      {
        // don't do anything for this use case
      }
      else
      {
        
        Kokkos::View<ScalarT *, HostDevice> sensordat;
        if (compute_response)
        {
          sensordat = Kokkos::View<ScalarT *, HostDevice>("sensor data to save", objectives[r].numSensors);
          objectives[r].response_times.push_back(current_time);
        }
        
        for (size_t pt = 0; pt < objectives[r].numSensors; ++pt)
        {
          size_t tindex = 0;
          bool foundtime = false;
          for (size_type t = 0; t < objectives[r].sensor_times.extent(0); ++t)
          {
            if (std::abs(current_time - objectives[r].sensor_times(t)) < 1.0e-12)
            {
              foundtime = true;
              tindex = t;
            }
          }
          
          if (compute_response || foundtime)
          {
            
            size_t block = objectives[r].block;
            size_t grp = objectives[r].sensor_owners(pt, 0);
            size_t elem = objectives[r].sensor_owners(pt, 1);
            assembler->wkset[block]->isOnPoint = true;
            auto x = assembler->wkset[block]->getScalarField("x");
            x(0, 0) = objectives[r].sensor_points(pt, 0);
            if (dimension > 1)
            {
              auto y = assembler->wkset[block]->getScalarField("y");
              y(0, 0) = objectives[r].sensor_points(pt, 1);
            }
            if (dimension > 2)
            {
              auto z = assembler->wkset[block]->getScalarField("z");
              z(0, 0) = objectives[r].sensor_points(pt, 2);
            }
            
            auto numDOF = assembler->groupData[block]->num_dof;
            if (!assembler->groups[block][grp]->have_sols)
            {
              for (size_t set = 0; set < sol_kv.size(); ++set)
              {
                assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
              }
              assembler->performGather(0, block, grp, params_kv[0], 4, 0);
            }
            
            View_Sc2 u_dof("u_dof", numDOF.extent(0), assembler->groups[block][grp]->LIDs[0].extent(1)); // hard coded
            auto cu = subview(assembler->groupData[block]->sol[0], elem, ALL(), ALL());                  // hard coded
            parallel_for("grp response get u", RangePolicy<AssemblyExec>(0, u_dof.extent(0)), MRHYDE_LAMBDA(const size_type n) {
              for (size_type n=0; n<numDOF.extent(0); n++) {
                for( int i=0; i<numDOF(n); i++ ) {
                  u_dof(n,i) = cu(n,i);
                }
              } });
            
            // Map the local solution to the solution and gradient at ip
            View_Sc2 u_ip("u_ip", numDOF.extent(0), assembler->groupData[block]->dimension);
            View_Sc2 ugrad_ip("ugrad_ip", numDOF.extent(0), assembler->groupData[block]->dimension);
            
            for (size_type var = 0; var < numDOF.extent(0); var++)
            {
              int bnum = assembler->wkset[block]->usebasis[var];
              auto btype = assembler->wkset[block]->basis_types[bnum];
              
              auto cbasis = objectives[r].sensor_basis[bnum];
              auto u_sv = subview(u_ip, var, ALL());
              auto u_dof_sv = subview(u_dof, var, ALL());
              
              parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) {
                u_sv(0) += u_dof_sv(dof)*cbasis(pt,dof,0,0);
              });
              
              if (btype == "HGRAD") {
                auto cbasis_grad = objectives[r].sensor_basis_grad[assembler->wkset[block]->usebasis[var]];
                auto ugrad_sv = subview(ugrad_ip, var, ALL());
                parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) {
                  for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                    ugrad_sv(dim) += u_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                  } });
                
              }
            }
            
            assembler->wkset[block]->setSolutionPoint(u_ip);
            assembler->wkset[block]->setSolutionGradPoint(ugrad_ip);
            
            // Map the local discretized params to param and grad at ip
            if (params->globalParamUnknowns > 0)
            {
              auto numParamDOF = assembler->groupData[block]->num_param_dof;
              if (!assembler->groups[block][grp]->have_sols)
              {
                assembler->performGather(0, block, grp, params_kv[0], 4, 0);
              }
              View_Sc2 p_dof("p_dof", numParamDOF.extent(0), assembler->groups[block][grp]->paramLIDs.extent(1));
              auto cp = subview(assembler->groupData[block]->param, elem, ALL(), ALL());
              parallel_for("grp response get u", RangePolicy<AssemblyExec>(0, p_dof.extent(0)), MRHYDE_LAMBDA(const size_type n) {
                for (size_type n=0; n<numParamDOF.extent(0); n++) {
                  for( int i=0; i<numParamDOF(n); i++ ) {
                    p_dof(n,i) = cp(n,i);
                  }
                } });
              
              View_Sc2 p_ip("p_ip", numParamDOF.extent(0), assembler->groupData[block]->dimension);
              View_Sc2 pgrad_ip("pgrad_ip", numParamDOF.extent(0), assembler->groupData[block]->dimension);
              
              for (size_type var = 0; var < numParamDOF.extent(0); var++)
              {
                int bnum = assembler->wkset[block]->paramusebasis[var];
                auto btype = assembler->wkset[block]->basis_types[bnum];
                auto cbasis = objectives[r].sensor_basis[bnum];
                auto p_sv = subview(p_ip, var, ALL());
                auto p_dof_sv = subview(p_dof, var, ALL());
                
                parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) { p_sv(0) += p_dof_sv(dof) * cbasis(pt, dof, 0, 0); });
                assembler->wkset[block]->setParamPoint(p_ip);
                
                if (btype == "HGRAD")
                {
                  auto cbasis_grad = objectives[r].sensor_basis_grad[bnum];
                  auto pgrad_sv = subview(pgrad_ip, var, ALL());
                  
                  parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) {
                    for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                      pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                    } });
                  assembler->wkset[block]->setParamGradPoint(pgrad_ip);
                }
              }
            }
            
            // Evaluate the response
            auto rdata = assembler->function_managers[block]->evaluate(objectives[r].name + " response", "point");
            
            if (compute_response)
            {
              sensordat(pt) = rdata(0, 0);
            }
            
            if (compute_objective)
            {
              
              // Update the value of the objective
              ScalarT diff = rdata(0, 0) - objectives[r].sensor_data(pt, tindex);
              ScalarT sdiff = objectives[r].weight * diff * diff;
              totaldiff[r] += sdiff;
            }
            assembler->wkset[block]->isOnPoint = false;
            
          } // found time
        } // sensor points
        
        if (compute_response)
        {
          objectives[r].response_data.push_back(sensordat);
        }
      } // objectives
    }
    // ========================================================================================
    // Add regularizations (reg funcs are tied to objectives and objectives can have more than one reg)
    // ========================================================================================
    
    for (size_t reg = 0; reg < objectives[r].regularizations.size(); ++reg)
    {
      if (objectives[r].regularizations[reg].type == "integrated")
      {
        if (objectives[r].regularizations[reg].location == "volume")
        {
          ScalarT regwt = objectives[r].regularizations[reg].weight;
          size_t block = objectives[r].block;
          for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
          {
            
            auto wts = assembler->groups[block][grp]->wts;
            
            if (!assembler->groups[block][grp]->have_sols)
            {
              assembler->performGather(0, block, grp, params_kv[0], 4, 0);
            }
            assembler->updateWorkset(block, grp, 3, 0, true);
            
            auto regvals_tmp = assembler->function_managers[block]->evaluate(objectives[r].regularizations[reg].name, "ip");
            View_Sc2 regvals("regvals", wts.extent(0), wts.extent(1));
            
            parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
              for (size_type pt=0; pt<wts.extent(1); ++pt) {
                regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
              } });
            
            auto regvals_sc_host = create_mirror_view(regvals);
            deep_copy(regvals_sc_host, regvals);
            
            auto poffs = params->paramoffsets;
            for (size_t elem = 0; elem < assembler->groups[block][grp]->numElem; ++elem)
            {
              
              // vector<GO> paramGIDs;
              // params->paramDOF->getElementGIDs(assembler->groups[block][grp]->localElemID(elem),
              //                                  paramGIDs, blocknames[block]);
              
              for (size_type pt = 0; pt < regvals_sc_host.extent(1); ++pt)
              {
                totaldiff[r] += regwt * regvals_sc_host(elem, pt);
              }
            }
          }
        }
        else if (objectives[r].regularizations[reg].location == "boundary")
        {
          string bname = objectives[r].regularizations[reg].boundary_name;
          ScalarT regwt = objectives[r].regularizations[reg].weight;
          size_t block = objectives[r].block;
          assembler->wkset[block]->isOnSide = true;
          for (size_t grp = 0; grp < assembler->boundary_groups[block].size(); ++grp)
          {
            if (assembler->boundary_groups[block][grp]->sidename == bname)
            {
              
              auto wts = assembler->boundary_groups[block][grp]->wts;
              
              assembler->performBoundaryGather(0, block, grp, params_kv[0], 4, 0);
              assembler->updateWorksetBoundary(block, grp, 3, 0, true);
              
              auto regvals_tmp = assembler->function_managers[block]->evaluate(objectives[r].regularizations[reg].name, "side ip");
              View_Sc2 regvals("regvals", wts.extent(0), wts.extent(1));
              
              parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
                for (size_type pt=0; pt<wts.extent(1); ++pt) {
                  regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
                } });
              
              auto regvals_sc_host = create_mirror_view(regvals);
              deep_copy(regvals_sc_host, regvals);
              
              auto poffs = params->paramoffsets;
              for (size_t elem = 0; elem < assembler->boundary_groups[block][grp]->numElem; ++elem)
              {
                
                vector<GO> paramGIDs;
                params->paramDOF->getElementGIDs(assembler->boundary_groups[block][grp]->localElemID(elem),
                                                 paramGIDs, blocknames[block]);
                
                for (size_type pt = 0; pt < regvals_sc_host.extent(1); ++pt)
                {
                  totaldiff[r] += regwt * regvals_sc_host(elem, pt);
                }
              }
            }
          }
          
          assembler->wkset[block]->isOnSide = false;
        }
      }
      else if(objectives[r].regularizations[reg].type == "l2 norm squared")
      {
        if(Comm->getRank() == 0)
        {
          int param_dim = params->getNumParams("active");
          if(param_dim == 0)
          {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: l2 norm squared reguarlization was used but the active parameter dimension is zero");
          }
          ScalarT regwt = objectives[r].regularizations[reg].weight;
          int index = 0;
          if( params->have_dynamic_scalar )
          {
            index = params->dynamic_timeindex;
          }
          std::vector<ScalarT> param_vec = params->getParams("active",index);
          for(int k = 0; k < param_dim; k++)
          {
            totaldiff[r] += regwt * param_vec[k] * param_vec[k];
          }
        }
        Comm->barrier();
      }
    }
  }
  
  for (size_t r = 0; r < totaldiff.size(); ++r)
  {
    objectives[r].objective_values.push_back(totaldiff[r]);
    objectives[r].objective_times.push_back(current_time);
  }
  
  debugger->print(1, "******** Finished PostprocessManager::computeObjective ...");
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::reportObjective(ScalarT &objectiveval)
{
  
  debugger->print(1, "******** Starting PostprocessManager::reportObjective ...");
  
  // For now, we scalarize the objective functions by summing them
  // Also, need to gather contributions across processors
  
  ScalarT totalobj = 0.0;
  
  for (size_t r = 0; r < objectives.size(); ++r)
  {
    ScalarT value = 0.0;
    if (objectives[r].objective_times.size() == 1)
    { // implies steady-state
      ScalarT gcontrib = 0.0;
      ScalarT lcontrib = objectives[r].objective_values[0];
      Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lcontrib, &gcontrib);
      value += gcontrib;
    }
    else
    {
      // Start with t=1 to ignore initial condition
      for (size_t t = 1; t < objectives[r].objective_times.size(); ++t)
      {
        ScalarT gcontrib = 0.0;
        ScalarT lcontrib = objectives[r].objective_values[t];
        Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lcontrib, &gcontrib);
        
        ScalarT dt = 1.0;
        
        dt = objectives[r].objective_times[t] - objectives[r].objective_times[t - 1];
        
        if (objectives[r].type != "sensors")
        {
          gcontrib *= dt;
        }
        value += gcontrib;
      }
    }
    if (objectives[r].type == "integrated response")
    {
      // Right now, totaldiff = response
      // We want    totaldiff = wt*(response-target)^2
      ScalarT diff = value - objectives[r].target;
      value = objectives[r].weight * diff * diff;
    }
    
    totalobj += value;
  }
  
  objectiveval += totalobj;
  
  if (write_objective_to_file) {
    std::stringstream ss;
    ss << objective_storage_file << ".dat";
    
    std::ofstream fout(ss.str());
    if (!fout.is_open()) {
      TEUCHOS_TEST_FOR_EXCEPTION(!fout.is_open(),std::runtime_error,"Error: could not open the data file: " + ss.str());
    }
    fout.precision(12);
    fout << objectiveval << endl;
    fout.close();
  }
  
  debugger->print(1, "******** Finished PostprocessManager::reportObjective ...");
}



// ========================================================================================
// ========================================================================================

// Helper function to save data
template <class Node>
void PostprocessManager<Node>::saveObjectiveData(const ScalarT &obj) {
  if (Comm->getRank() != 0)
    return;
  if (objective_file.length() > 0) {
    std::ofstream obj_out{objective_file};
    TEUCHOS_TEST_FOR_EXCEPTION(!obj_out.is_open(), std::runtime_error, "Could not open file to print objective value");
    obj_out << obj;
  }
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::resetObjectives()
{
  for (size_t r = 0; r < objectives.size(); ++r)
  {
    objectives[r].response_times.clear();
    objectives[r].response_data.clear();
    objectives[r].scalar_response_data.clear();
    objectives[r].objective_times.clear();
    objectives[r].objective_values.clear();
  }
}
