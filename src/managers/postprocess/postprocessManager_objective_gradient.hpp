/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// ========================================================================================

// Helper function to save data
template <class Node>
void PostprocessManager<Node>::saveObjectiveGradientData(const MrHyDE_OptVector &gradient)
{
  if (Comm->getRank() != 0)
    return;
  if (objective_grad_file.length() > 0)
  {
    std::ofstream obj_grad_out{objective_grad_file};
    TEUCHOS_TEST_FOR_EXCEPTION(!obj_grad_out.is_open(), std::runtime_error, "Could not open file to print objective gradient value");
    gradient.print(obj_grad_out);
  }
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::computeObjectiveGradParam(vector<vector_RCP> &current_soln,
                                                         const ScalarT &current_time,
                                                         const ScalarT &dt,
                                                         DFAD &objectiveval)
{

  debugger->print(1, "******** Starting PostprocessManager::computeObjectiveGradParam ...");

#ifndef MrHyDE_NO_AD
  for (size_t r = 0; r < objectives.size(); ++r)
  {
    DFAD newobj = 0.0;
    size_t block = objectives[r].block;
    if (assembler->type_AD == -1)
    {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time, dt,
                                               assembler->wkset_AD[block],
                                               assembler->function_managers_AD[block]);
    }
    else if (assembler->type_AD == 2)
    {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time, dt,
                                               assembler->wkset_AD2[block],
                                               assembler->function_managers_AD2[block]);
    }
    else if (assembler->type_AD == 4)
    {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time, dt,
                                               assembler->wkset_AD4[block],
                                               assembler->function_managers_AD4[block]);
    }
    else if (assembler->type_AD == 8)
    {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time, dt,
                                               assembler->wkset_AD8[block],
                                               assembler->function_managers_AD8[block]);
    }
    else if (assembler->type_AD == 16)
    {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time, dt,
                                               assembler->wkset_AD16[block],
                                               assembler->function_managers_AD16[block]);
    }
    else if (assembler->type_AD == 18)
    {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time, dt,
                                               assembler->wkset_AD18[block],
                                               assembler->function_managers_AD18[block]);
    }
    else if (assembler->type_AD == 24)
    {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time, dt,
                                               assembler->wkset_AD24[block],
                                               assembler->function_managers_AD24[block]);
    }
    else if (assembler->type_AD == 32)
    {
      newobj = this->computeObjectiveGradParam(r, current_soln, current_time, dt,
                                               assembler->wkset_AD32[block],
                                               assembler->function_managers_AD32[block]);
    }

    objectiveval += newobj;
  }

#if defined(MrHyDE_ENABLE_HDSA)
  if (hdsa_solop)
  {
    objectiveval = 0.0;
  }
#endif

  saveObjectiveData(objectiveval.val());
#endif

  debugger->print(1, "******** Finished PostprocessManager::computeObjectiveGradParam ...");
}

// ========================================================================================
// ========================================================================================

template <class Node>
template <class EvalT>
DFAD PostprocessManager<Node>::computeObjectiveGradParam(const size_t &obj, vector<vector_RCP> &current_soln,
                                                         const ScalarT &current_time,
                                                         const ScalarT &dt,
                                                         Teuchos::RCP<Workset<EvalT>> &wset,
                                                         Teuchos::RCP<FunctionManager<EvalT>> &fman)
{

  Teuchos::TimeMonitor localtimer(*objectiveTimer);

  debugger->print(1, "******** Starting PostprocessManager::computeObjectiveGradParam<EvalT> ...");

  DFAD fullobj = 0.0;

#ifndef MrHyDE_NO_AD

  typedef Kokkos::View<EvalT **, ContLayout, AssemblyDevice> View_EvalT2;

  typedef typename Node::execution_space LA_exec;
  typedef typename Node::device_type LA_device;

  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible)
  {
    data_avail = false;
  }

  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> sol_kv;
  for (size_t s = 0; s < current_soln.size(); ++s)
  {
    auto vec_kv = current_soln[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
    if (data_avail)
    {
      sol_kv.push_back(vec_slice);
    }
    else
    {
      auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(), vec_slice);
      Kokkos::deep_copy(vec_dev, vec_slice);
      sol_kv.push_back(vec_dev);
    }
  }

  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> params_kv;

  if (params->num_discretized_params > 0)
  {
    auto Psol = params->getDiscretizedParamsOver();
    auto p_kv = Psol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto pslice = Kokkos::subview(p_kv, Kokkos::ALL(), 0);

    if (data_avail)
    {
      params_kv.push_back(pslice);
    }
    else
    {
      auto p_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(), pslice);
      Kokkos::deep_copy(p_dev, pslice);
      params_kv.push_back(p_dev);
    }
  }

  // Objective function values
  ScalarT objval = 0.0;

  int numParams = params->num_active_params + params->numParamUnknownsOS;
  size_t block = objectives[obj].block;

  // Objective function gradients w.r.t params
  vector<ScalarT> gradient(numParams, 0.0);

  // for (size_t r=0; r<objectives.size(); ++r) {
  if (objectives[obj].type == "integrated control")
  {

    // First, compute objective value and deriv. w.r.t scalar params
    params->sacadoizeParams(true);

    for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
    {

      View_Sc1 objsum_dev("obj func sum as scalar on device", numParams + 1);
      // assembler->computeObjectiveGrad(block, grp, objectives[r].name, objsum_dev);

      auto wts = assembler->groups[block][grp]->wts;

      if (!assembler->groups[block][grp]->have_sols)
      {
        for (size_t set=0; set<sol_kv.size(); ++set) {
          assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
          if (params->num_discretized_params > 0) {
            assembler->performGather(0, block, grp, params_kv[0], 4, 0);
          }
        }
      }
      assembler->updateWorksetAD(block, grp, 0, 0, true);

      auto obj_dev = fman->evaluate(objectives[obj].name, "ip");

      Kokkos::View<EvalT[1], AssemblyDevice> objsum("sum of objective");
      parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
        EvalT tmpval = 0.0;
        for (size_type pt=0; pt<wts.extent(1); pt++) {
          tmpval += obj_dev(elem,pt)*wts(elem,pt);
        }
        Kokkos::atomic_add(&(objsum(0)),tmpval); });

      parallel_for("grp objective", RangePolicy<AssemblyExec>(0, objsum_dev.extent(0)), MRHYDE_LAMBDA(const size_type p) {
        size_t numder = static_cast<size_t>(objsum(0).size());
        if (p==0) {
          objsum_dev(p) = objsum(0).val();
        }
        else if (p <= numder) {
          objsum_dev(p) = objsum(0).fastAccessDx(p-1);
        } });

      auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
      Kokkos::deep_copy(objsum_host, objsum_dev);

      // Update the objective function value
      objval += objectives[obj].weight * objsum_host(0);

      // Update the gradients w.r.t scalar active parameters
      for (size_t p = 0; p < params->num_active_params; p++)
      {
        gradient[p] += objectives[obj].weight * objsum_host(p + 1);
      }
    }

    // Next, deriv w.r.t discretized params
    if (params->globalParamUnknowns > 0)
    {

      params->sacadoizeParams(false);

      for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
      {

        auto wts = assembler->groups[block][grp]->wts;

        if (!assembler->groups[block][grp]->have_sols)
        {
          for (size_t set=0; set<sol_kv.size(); ++set) {
            assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
          }
          assembler->performGather(0, block, grp, params_kv[0], 4, 0);
        }
        assembler->updateWorksetAD(block, grp, 3, 0, true);

        auto obj_dev = fman->evaluate(objectives[obj].name, "ip");

        Kokkos::View<EvalT[1], AssemblyDevice> objsum("sum of objective");
        parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
          EvalT tmpval = 0.0;
          for (size_type pt=0; pt<wts.extent(1); pt++) {
            tmpval += obj_dev(elem,pt)*wts(elem,pt);
          }
          Kokkos::atomic_add(&(objsum(0)),tmpval); });

        View_Sc1 objsum_dev("obj func sum as scalar on device", numParams + 1);

        parallel_for("grp objective", RangePolicy<AssemblyExec>(0, objsum_dev.extent(0)), MRHYDE_LAMBDA(const size_type p) {
          size_t numder = static_cast<size_t>(objsum(0).size());
          if (p==0) {
            objsum_dev(p) = objsum(0).val();
          }
          else if (p <= numder) {
            objsum_dev(p) = objsum(0).fastAccessDx(p-1);
          } });

        auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
        Kokkos::deep_copy(objsum_host, objsum_dev);
        auto poffs = params->paramoffsets;
        auto LIDs = assembler->groups[block][grp]->paramLIDs;

        for (size_t c = 0; c < assembler->groups[block][grp]->numElem; c++)
        {
          // vector<GO> paramGIDs;
          // params->paramDOF->getElementGIDs(assembler->groups[block][grp]->localElemID[c],
          //                                  paramGIDs, blocknames[block]);

          for (size_t pp = 0; pp < poffs.size(); ++pp)
          {
            for (size_t row = 0; row < poffs[pp].size(); row++)
            {
              // GO rowIndex = paramGIDs[poffs[pp][row]];
              LO rowIndex = LIDs(c, poffs[pp][row]); // paramGIDs[poffs[pp][row]];
              int poffset = 1 + poffs[pp][row];
              gradient[rowIndex + params->num_active_params] += objectives[obj].weight * objsum_host(poffset);
            }
          }
        }
      }
    }
    for (size_t i = 0; i < gradient.size(); ++i)
    {
      gradient[i] *= dt;
    }
  }
  else if (objectives[obj].type == "discrete control")
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
        Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> objn(1);
        diff->norm2(objn);
        if (Comm->getRank() == 0)
        {
          objval += objectives[obj].weight * dt * objn[0] * objn[0];
        }
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: did not find a data-generating solution");
      }
    }
  }
  else if (objectives[obj].type == "integrated response")
  {

    ScalarT value = 0.0;
    if (objectives[obj].objective_times.size() == 1)
    { // implies steady-state
      ScalarT gcontrib = 0.0;
      ScalarT lcontrib = objectives[obj].objective_values[0];
      Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lcontrib, &gcontrib);
      value += gcontrib;
    }
    else
    {
      // Start with t=1 to ignore initial condition
      for (size_t t = 1; t < objectives[obj].objective_times.size(); ++t)
      {
        ScalarT gcontrib = 0.0;
        ScalarT lcontrib = objectives[obj].objective_values[t];
        Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lcontrib, &gcontrib);

        ScalarT dt = objectives[obj].objective_times[t] - objectives[obj].objective_times[t - 1];
        gcontrib *= dt;
        value += gcontrib;
      }
    }

    // First, compute objective value and deriv. w.r.t scalar params
    // if (params->num_active_params > 0) {
    params->sacadoizeParams(true); // seed active

    for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
    {

      auto wts = assembler->groups[block][grp]->wts;

      if (!assembler->groups[block][grp]->have_sols)
      {
        for (size_t set=0; set<sol_kv.size(); ++set) {
          assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
        }
        if (params->globalParamUnknowns > 0)
        {
          assembler->performGather(0, block, grp, params_kv[0], 4, 0);
        }
      }
      assembler->updateWorkset(block, grp, 0, 0, true);

      auto obj_dev = fman->evaluate(objectives[obj].name + " response", "ip");

      Kokkos::View<EvalT[1], AssemblyDevice> objsum("sum of objective");
      parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
        EvalT tmpval = 0.0;
        for (size_type pt=0; pt<wts.extent(1); pt++) {
          tmpval += obj_dev(elem,pt)*wts(elem,pt);
        }
        Kokkos::atomic_add(&(objsum(0)),tmpval); });

      View_Sc1 objsum_dev("obj func sum as scalar on device", numParams + 1);

      parallel_for("grp objective", RangePolicy<AssemblyExec>(0, objsum_dev.extent(0)), MRHYDE_LAMBDA(const size_type p) {
        size_t numder = static_cast<size_t>(objsum(0).size());
        if (p==0) {
          objsum_dev(p) = objsum(0).val();
        }
        else if (p <= numder) {
          objsum_dev(p) = objsum(0).fastAccessDx(p-1);
        } });

      auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
      Kokkos::deep_copy(objsum_host, objsum_dev);

      // Update the objective function value
      objval += objsum_host(0);

      // Update the gradients w.r.t scalar active parameters
      for (size_t p = 0; p < params->num_active_params; p++)
      {
        gradient[p] += objsum_host(p + 1);
      }
    }

    if (compute_response)
    {
      if (objectives[obj].save_data)
      {
        objectives[obj].response_times.push_back(current_time);
        objectives[obj].scalar_response_data.push_back(objval);
        if (verbosity >= 10)
        {
          double localval = objval;
          double globalval = 0.0;
          Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &localval, &globalval);
          if (Comm->getRank() == 0)
          {
            cout << objectives[obj].name << " on block " << blocknames[block] << ": " << globalval << endl;
          }
        }
      }
    }
    //}

    // Next, deriv w.r.t discretized params
    if (params->globalParamUnknowns > 0)
    {

      params->sacadoizeParams(false);

      for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
      {

        auto wts = assembler->groups[block][grp]->wts;

        if (!assembler->groups[block][grp]->have_sols)
        {
          for (size_t set=0; set<sol_kv.size(); ++set) {
            assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
          }
          assembler->performGather(0, block, grp, params_kv[0], 4, 0);
        }
        assembler->updateWorksetAD(block, grp, 3, 0, true);

        auto obj_dev = fman->evaluate(objectives[obj].name + " response", "ip");

        Kokkos::View<EvalT[1], AssemblyDevice> objsum("sum of objective");
        parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
          EvalT tmpval = 0.0;
          for (size_type pt=0; pt<wts.extent(1); pt++) {
            tmpval += obj_dev(elem,pt)*wts(elem,pt);
          }
          Kokkos::atomic_add(&(objsum(0)),tmpval); });

        View_Sc1 objsum_dev("obj func sum as scalar on device", numParams + 1);

        parallel_for("grp objective", RangePolicy<AssemblyExec>(0, objsum_dev.extent(0)), MRHYDE_LAMBDA(const size_type p) {
          size_t numder = static_cast<size_t>(objsum(0).size());
          if (p==0) {
            objsum_dev(p) = objsum(0).val();
          }
          else if (p <= numder) {
            objsum_dev(p) = objsum(0).fastAccessDx(p-1);
          } });

        auto objsum_host = Kokkos::create_mirror_view(objsum_dev);
        Kokkos::deep_copy(objsum_host, objsum_dev);
        auto poffs = params->paramoffsets;

        for (size_t c = 0; c < assembler->groups[block][grp]->numElem; c++)
        {
          vector<GO> paramGIDs;
          params->paramDOF->getElementGIDs(assembler->groups[block][grp]->localElemID[c],
                                           paramGIDs, blocknames[block]);

          for (size_t pp = 0; pp < poffs.size(); ++pp)
          {
            for (size_t row = 0; row < poffs[pp].size(); row++)
            {
              GO rowIndex = paramGIDs[poffs[pp][row]];
              int poffset = 1 + poffs[pp][row];
              // gradients[r][rowIndex+params->num_active_params] += objectives[r].weight*objsum_host(poffset);
              gradient[rowIndex + params->num_active_params] += objsum_host(poffset);
            }
          }
        }
      }
    }

    // Right now, totaldiff = response
    //             gradient = dresponse / dp
    // We want    totaldiff = wt*(response-target)^2
    //             gradient = 2*wt*(response-target)*dresponse/dp

    ScalarT diff = value - objectives[obj].target;
    for (size_t g = 0; g < gradient.size(); ++g)
    {
      gradient[g] = 2.0 * dt * objectives[obj].weight * diff * gradient[g];
    }
  }
  else if (objectives[obj].type == "sensors")
  {
    if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln)
    {
      // don't do anything for this use case
    }
    else
    {
      Kokkos::View<ScalarT *, HostDevice> sensordat;
      if (compute_response)
      {
        sensordat = Kokkos::View<ScalarT *, HostDevice>("sensor data to save", objectives[obj].numSensors);
        objectives[obj].response_times.push_back(current_time);
      }

      for (size_t pt = 0; pt < objectives[obj].numSensors; ++pt)
      {
        size_t tindex = 0;
        bool foundtime = false;
        for (size_type t = 0; t < objectives[obj].sensor_times.extent(0); ++t)
        {
          if (std::abs(current_time - objectives[obj].sensor_times(t)) < 1.0e-12)
          {
            foundtime = true;
            tindex = t;
          }
        }

        if (compute_response || foundtime)
        {

          // First compute objective and derivative w.r.t scalar params
          params->sacadoizeParams(true);

          size_t grp = objectives[obj].sensor_owners(pt, 0);
          size_t elem = objectives[obj].sensor_owners(pt, 1);
          wset->isOnPoint = true;
          auto x = wset->getScalarField("x");
          x(0, 0) = objectives[obj].sensor_points(pt, 0);
          if (dimension > 1)
          {
            auto y = wset->getScalarField("y");
            y(0, 0) = objectives[obj].sensor_points(pt, 1);
          }
          if (dimension > 2)
          {
            auto z = wset->getScalarField("z");
            z(0, 0) = objectives[obj].sensor_points(pt, 2);
          }

          for (size_t set=0; set<sol_kv.size(); ++set) {
            assembler->updatePhysicsSet(set);
            auto numDOF = assembler->groupData[block]->num_dof; // filled in properly after updatePhysicsSet gets called
            View_EvalT2 u_dof("u_dof", numDOF.extent(0), assembler->groups[block][grp]->LIDs[set].extent(1)); // hard coded
            if (!assembler->groups[block][grp]->have_sols)
            {
              assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
            }
            auto cu = subview(assembler->groupData[block]->sol[set], elem, ALL(), ALL()); // hard coded
            parallel_for("grp response get u", RangePolicy<AssemblyExec>(0, u_dof.extent(0)), MRHYDE_LAMBDA(const size_type n) {
              for (size_type n=0; n<numDOF.extent(0); n++) {
                for( int i=0; i<numDOF(n); i++ ) {
                  u_dof(n,i) = cu(n,i);
                }
              } });
            
            // Map the local solution to the solution and gradient at ip
            View_EvalT2 u_ip("u_ip", numDOF.extent(0), assembler->groupData[block]->dimension);
            View_EvalT2 ugrad_ip("ugrad_ip", numDOF.extent(0), assembler->groupData[block]->dimension);
            
            for (size_type var = 0; var < numDOF.extent(0); var++)
            {
              auto cbasis = objectives[obj].sensor_basis[wset->usebasis[var]];
              auto cbasis_grad = objectives[obj].sensor_basis_grad[wset->usebasis[var]];
              auto u_sv = subview(u_ip, var, ALL());
              auto u_dof_sv = subview(u_dof, var, ALL());
              auto ugrad_sv = subview(ugrad_ip, var, ALL());
              
              parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) {
                u_sv(0) += u_dof_sv(dof)*cbasis(pt,dof,0,0);
                for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                  ugrad_sv(dim) += u_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                } });
            }
            
            wset->setSolutionPoint(u_ip);
            wset->setSolutionGradPoint(ugrad_ip);
          }
          // Map the local discretized params to param and grad at ip
          if (params->globalParamUnknowns > 0)
          {
            auto numParamDOF = assembler->groupData[block]->num_param_dof;

            View_EvalT2 p_dof("p_dof", numParamDOF.extent(0), assembler->groups[block][grp]->paramLIDs.extent(1));
            if (!assembler->groups[block][grp]->have_sols)
            {
              assembler->performGather(0, block, grp, params_kv[0], 4, 0);
            }
            auto cp = subview(assembler->groupData[block]->param, elem, ALL(), ALL());
            parallel_for("grp response get u", RangePolicy<AssemblyExec>(0, p_dof.extent(0)), MRHYDE_LAMBDA(const size_type n) {
              for (size_type n=0; n<numParamDOF.extent(0); n++) {
                for( int i=0; i<numParamDOF(n); i++ ) {
                  p_dof(n,i) = cp(n,i);
                }
              } });

            View_EvalT2 p_ip("p_ip", numParamDOF.extent(0), assembler->groupData[block]->dimension);
            View_EvalT2 pgrad_ip("pgrad_ip", numParamDOF.extent(0), assembler->groupData[block]->dimension);

            for (size_type var = 0; var < numParamDOF.extent(0); var++)
            {
              int bnum = wset->paramusebasis[var];
              auto btype = wset->basis_types[bnum];

              auto cbasis = objectives[obj].sensor_basis[bnum];
              auto p_sv = subview(p_ip, var, ALL());
              auto p_dof_sv = subview(p_dof, var, ALL());

              parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) { p_sv(0) += p_dof_sv(dof) * cbasis(pt, dof, 0, 0); });
              wset->setParamPoint(p_ip);

              if (btype == "HGRAD")
              {
                auto cbasis_grad = objectives[obj].sensor_basis_grad[bnum];
                auto p_dof_sv = subview(p_dof, var, ALL());
                auto pgrad_sv = subview(pgrad_ip, var, ALL());

                parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) {
                  for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                    pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                  } });
                wset->setParamGradPoint(pgrad_ip);
              }
            }
          }

          // Evaluate the response
          auto rdata = fman->evaluate(objectives[obj].name + " response", "point");

          if (compute_response)
          {
            sensordat(pt) = rdata(0, 0).val();
          }

          if (compute_objective)
          {

            // Update the value of the objective
            EvalT diff = rdata(0, 0) - objectives[obj].sensor_data(pt, tindex);
            EvalT sdiff = objectives[obj].weight * diff * diff;
            objval += sdiff.val();

            // Update the gradient w.r.t scalar active parameters
            for (size_t p = 0; p < params->num_active_params; p++)
            {
              gradient[p] += sdiff.fastAccessDx(p);
            }

            // Discretized parameters
            if (params->globalParamUnknowns > 0)
            {

              // Need to compute derivative w.r.t discretized params
              params->sacadoizeParams(false);

              auto numParamDOF = assembler->groupData[block]->num_param_dof;
              auto poff = wset->paramoffsets;
              View_EvalT2 p_dof("p_dof", numParamDOF.extent(0), assembler->groups[block][grp]->paramLIDs.extent(1));
              if (!assembler->groups[block][grp]->have_sols)
              {
                assembler->performGather(0, block, grp, params_kv[0], 4, 0);
              }
              auto cp = subview(assembler->groupData[block]->param, elem, ALL(), ALL());
              parallel_for("grp response get u", RangePolicy<AssemblyExec>(0, p_dof.extent(0)), MRHYDE_LAMBDA(const size_type n) {
                EvalT dummyval = 0.0;
                for (size_type n=0; n<numParamDOF.extent(0); n++) {
                  for( int i=0; i<numParamDOF(n); i++ ) {
                    p_dof(n,i) = EvalT(dummyval.size(),poff(n,i),cp(n,i));
                  }
                } });

              View_EvalT2 p_ip("p_ip", numParamDOF.extent(0), assembler->groupData[block]->dimension);
              View_EvalT2 pgrad_ip("pgrad_ip", numParamDOF.extent(0), assembler->groupData[block]->dimension);

              for (size_type var = 0; var < numParamDOF.extent(0); var++)
              {
                int bnum = wset->paramusebasis[var];
                auto btype = wset->basis_types[bnum];

                auto cbasis = objectives[obj].sensor_basis[bnum];
                auto p_sv = subview(p_ip, var, ALL());
                auto p_dof_sv = subview(p_dof, var, ALL());

                parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) { p_sv(0) += p_dof_sv(dof) * cbasis(pt, dof, 0, 0); });
                wset->setParamPoint(p_ip);

                if (btype == "HGRAD")
                {
                  auto cbasis_grad = objectives[obj].sensor_basis_grad[bnum];
                  auto p_dof_sv = subview(p_dof, var, ALL());
                  auto pgrad_sv = subview(pgrad_ip, var, ALL());

                  parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) {
                    for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                      pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                    } });
                  wset->setParamGradPoint(pgrad_ip);
                }
              }

              // Evaluate the response
              auto rdata = fman->evaluate(objectives[obj].name + " response", "point");
              EvalT diff = rdata(0, 0) - objectives[obj].sensor_data(pt, tindex);
              EvalT sdiff = objectives[obj].weight * diff * diff;

              auto poffs = params->paramoffsets;
              vector<GO> paramGIDs;
              params->paramDOF->getElementGIDs(assembler->groups[block][grp]->localElemID[elem],
                                               paramGIDs, blocknames[block]);

              for (size_t pp = 0; pp < poffs.size(); ++pp)
              {
                for (size_t row = 0; row < poffs[pp].size(); row++)
                {
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

      if (compute_response)
      {
        objectives[obj].response_data.push_back(sensordat);
      }
    } // objectives
  }
  // ========================================================================================
  // Add regularizations (reg funcs are tied to objectives and objectives can have more than one reg)
  // ========================================================================================

  for (size_t reg = 0; reg < objectives[obj].regularizations.size(); ++reg)
  {
    if (objectives[obj].regularizations[reg].type == "integrated" && params->num_discretized_params > 0)
    {
      if (objectives[obj].regularizations[reg].location == "volume")
      {
        params->sacadoizeParams(false);
        ScalarT regwt = objectives[obj].regularizations[reg].weight;

        for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
        {

          auto wts = assembler->groups[block][grp]->wts;

          if (!assembler->groups[block][grp]->have_sols)
          {
            for (size_t set=0; set<sol_kv.size(); ++set) {
              assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
            }
            assembler->performGather(0, block, grp, params_kv[0], 4, 0);
          }
          assembler->updateWorksetAD(block, grp, 3, 0, true);

          auto regvals_tmp = fman->evaluate(objectives[obj].regularizations[reg].name, "ip");
          View_EvalT2 regvals("regvals", wts.extent(0), wts.extent(1));
          parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
            for (size_type pt=0; pt<wts.extent(1); ++pt) {
              regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
            } });

          EvalT dummyval = 0.0;
          View_Sc3 regvals_sc("scalar version of AD view", wts.extent(0), wts.extent(1), dummyval.size() + 1);
          parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
            for (size_type pt=0; pt<wts.extent(1); ++pt) {
              regvals_sc(elem,pt,0) = regvals(elem,pt).val();
              for (size_type d=0; d<regvals_sc.extent(2)-1; ++d) {
                regvals_sc(elem,pt,d+1) = regvals(elem,pt).fastAccessDx(d);
              }
            } });

          auto regvals_sc_host = create_mirror_view(regvals_sc);
          deep_copy(regvals_sc_host, regvals_sc);

          auto LIDs = assembler->groups[block][grp]->paramLIDs;
          auto poffs = params->paramoffsets;
          for (size_t elem = 0; elem < assembler->groups[block][grp]->numElem; ++elem)
          {

            for (size_type pt = 0; pt < regvals_sc_host.extent(1); ++pt)
            {
              objval += regwt * regvals_sc_host(elem, pt, 0);
              for (size_t pp = 0; pp < poffs.size(); ++pp)
              {
                for (size_t row = 0; row < poffs[pp].size(); row++)
                {
                  LO rowIndex = LIDs(elem, poffs[pp][row]) + params->num_active_params;
                  int poffset = poffs[pp][row] + 1;
                  gradient[rowIndex] += regwt * dt * regvals_sc_host(elem, pt, poffset);
                }
              }
            }
          }
        }
      }
      else if (objectives[obj].regularizations[reg].location == "boundary")
      {
        string bname = objectives[obj].regularizations[reg].boundary_name;
        params->sacadoizeParams(false);
        ScalarT regwt = objectives[obj].regularizations[reg].weight;
        wset->isOnSide = true;
        for (size_t grp = 0; grp < assembler->boundary_groups[block].size(); ++grp)
        {
          if (assembler->boundary_groups[block][grp]->sidename == bname)
          {

            auto wts = assembler->boundary_groups[block][grp]->wts;

            for (size_t set=0; set<sol_kv.size(); ++set) {
              assembler->performBoundaryGather(set, block, grp, sol_kv[set], 0, 0);
            }
            assembler->performBoundaryGather(0, block, grp, params_kv[0], 4, 0);
            assembler->updateWorksetBoundaryAD(block, grp, 3, 0, true);

            auto regvals_tmp = fman->evaluate(objectives[obj].regularizations[reg].name, "side ip");
            View_EvalT2 regvals("regvals", wts.extent(0), wts.extent(1));

            parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
              for (size_type pt=0; pt<wts.extent(1); ++pt) {
                regvals(elem,pt) = wts(elem,pt)*regvals_tmp(elem,pt);
              } });

            EvalT dummyval = 0.0;
            View_Sc3 regvals_sc("scalar version of AD view", wts.extent(0), wts.extent(1), dummyval.size() + 1);
            parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
              for (size_type pt = 0; pt < wts.extent(1); ++pt)
              {
                regvals_sc(elem, pt, 0) = regvals(elem, pt).val();
                for (size_type d = 0; d < regvals_sc.extent(2) - 1; ++d)
                {
                  regvals_sc(elem, pt, d + 1) = regvals(elem, pt).fastAccessDx(d);
                }
              } });

            auto regvals_sc_host = create_mirror_view(regvals_sc);
            deep_copy(regvals_sc_host, regvals_sc);

            auto poffs = params->paramoffsets;
            auto LIDs = assembler->boundary_groups[block][grp]->paramLIDs;

            for (size_t elem = 0; elem < assembler->boundary_groups[block][grp]->numElem; ++elem)
            {

              for (size_type pt = 0; pt < regvals_sc_host.extent(1); ++pt)
              {
                objval += regwt * regvals_sc_host(elem, pt, 0);
                for (size_t pp = 0; pp < poffs.size(); ++pp)
                {
                  for (size_t row = 0; row < poffs[pp].size(); ++row)
                  {
                    // GO rowIndex = paramGIDs[poffs[pp][row]] + params->num_active_params;
                    GO rowIndex = LIDs(elem, poffs[pp][row]) + params->num_active_params;
                    int poffset = poffs[pp][row];
                    gradient[rowIndex] += regwt * dt * regvals_sc_host(elem, pt, poffset + 1);
                  }
                }
              }
            }
          }
        }
        wset->isOnSide = false;
      }
    }
    else if(objectives[obj].regularizations[reg].type == "l2 norm squared")
    {
      int param_dim = params->getNumParams("active");
      if(param_dim != gradient.size())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: l2 norm squared reguarlization assumes that there are no discretized parameters");
      }
      ScalarT regwt = objectives[obj].regularizations[reg].weight;
      int index = 0;
      if (params->have_dynamic_scalar)
      {
        index = params->dynamic_timeindex;
      }
      std::vector<ScalarT> param_vec = params->getParams("active", index);
      for (int k = 0; k < param_dim; k++)
      {
        gradient[k] += regwt * dt * 2.0 * param_vec[k];
      }
    }
  }
  //}

  // to gather contributions across processors
  ScalarT meep = 0.0;
  Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &objval, &meep);

  fullobj = DFAD(numParams, meep);

  for (int j = 0; j < numParams; j++)
  {
    // ScalarT dval = 0.0;
    ScalarT ldval = gradient[j];
    // Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&ldval,&dval);
    fullobj.fastAccessDx(j) = ldval;
  }

  params->sacadoizeParams(false);

#endif

  debugger->print(1, "******** Finished PostprocessManager::computeObjectiveGradParam<EvalT> ...");

  return fullobj;
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::computeObjectiveGradState(const size_t &set,
                                                         const vector_RCP &current_soln,
                                                         const ScalarT &current_time,
                                                         const ScalarT &deltat,
                                                         vector_RCP &grad)
{

  debugger->print(1, "******** Starting PostprocessManager::computeObjectiveGradState ...");

  // Determine if we want to collect QoI, objectives, etc.
  bool write_this_step = false;
  if (time_index % write_frequency == 0)
  {
    write_this_step = true;
  }

#ifndef MrHyDE_NO_AD
  if (write_this_step)
  {

    for (size_t r = 0; r < objectives.size(); ++r)
    {
      size_t block = objectives[r].block;

#if defined(MrHyDE_ENABLE_HDSA)
      if (hdsa_solop)
      {
        vector_RCP D_soln;
        hdsa_solop_data[set]->extract(D_soln, 0, current_time);
        grad->update(1.0, *D_soln, 1.0);
      }
      else
      {
#endif

        if (assembler->type_AD == -1)
        {
          this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                          assembler->wkset_AD[block],
                                          assembler->function_managers_AD[block]);
        }
        else if (assembler->type_AD == 2)
        {
          this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                          assembler->wkset_AD2[block],
                                          assembler->function_managers_AD2[block]);
        }
        else if (assembler->type_AD == 4)
        {
          this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                          assembler->wkset_AD4[block],
                                          assembler->function_managers_AD4[block]);
        }
        else if (assembler->type_AD == 8)
        {
          this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                          assembler->wkset_AD8[block],
                                          assembler->function_managers_AD8[block]);
        }
        else if (assembler->type_AD == 16)
        {
          this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                          assembler->wkset_AD16[block],
                                          assembler->function_managers_AD16[block]);
        }
        else if (assembler->type_AD == 18)
        {
          this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                          assembler->wkset_AD18[block],
                                          assembler->function_managers_AD18[block]);
        }
        else if (assembler->type_AD == 24)
        {
          this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                          assembler->wkset_AD24[block],
                                          assembler->function_managers_AD24[block]);
        }
        else if (assembler->type_AD == 32)
        {
          this->computeObjectiveGradState(set, r, current_soln, current_time, deltat, grad,
                                          assembler->wkset_AD32[block],
                                          assembler->function_managers_AD32[block]);
        }
#if defined(MrHyDE_ENABLE_HDSA)
      }
#endif
    }
  }
#endif

  debugger->print(1, "******** Finished PostprocessManager::computeObjectiveGradState ...");
}


// ========================================================================================
// ========================================================================================

template <class Node>
template <class EvalT>
void PostprocessManager<Node>::computeObjectiveGradState(const size_t &set,
                                                         const size_t &obj,
                                                         const vector_RCP &current_soln,
                                                         const ScalarT &current_time,
                                                         const ScalarT &deltat,
                                                         vector_RCP &grad,
                                                         Teuchos::RCP<Workset<EvalT>> &wset,
                                                         Teuchos::RCP<FunctionManager<EvalT>> &fman)
{

  debugger->print(1, "******** Starting PostprocessManager::computeObjectiveGradState<EvalT> ...");

#ifndef MrHyDE_NO_AD

  typedef Kokkos::View<EvalT **, ContLayout, AssemblyDevice> View_EvalT2;

  DFAD totaldiff = 0.0;

  params->sacadoizeParams(false);

  int numParams = params->num_active_params + params->globalParamUnknowns;
  size_t block = objectives[obj].block;

  vector<ScalarT> regGradient(numParams);
  vector<ScalarT> dmGradient(numParams);

  typedef typename Node::device_type LA_device;
  typedef typename Node::execution_space LA_exec;

  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible)
  {
    data_avail = false;
  }
  // LIDs are on AssemblyDevice.  If the AssemblyDevice memory is accessible, then these are fine.
  // Copy of LIDs is stored on HostDevice.
  bool use_host_LIDs = false;
  if (!data_avail)
  {
    if (Kokkos::SpaceAccessibility<LA_exec, HostDevice::memory_space>::accessible)
    {
      use_host_LIDs = true;
    }
  }

  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> sol_kv;
  auto vec_kv = current_soln->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
  if (data_avail)
  {
    sol_kv.push_back(vec_slice);
  }
  else
  {
    auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(), vec_slice);
    Kokkos::deep_copy(vec_dev, vec_slice);
    sol_kv.push_back(vec_dev);
  }

  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> params_kv;

  auto Psol = params->getDiscretizedParams();
  auto p_kv = Psol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto pslice = Kokkos::subview(p_kv, Kokkos::ALL(), 0);

  if (data_avail)
  {
    params_kv.push_back(pslice);
  }
  else
  {
    auto p_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(), pslice);
    Kokkos::deep_copy(p_dev, pslice);
    params_kv.push_back(p_dev);
  }

  // We are on a given time step
  // Need to find the appropriate dt to scale the objective value and gradient
  ScalarT dt = 1.0;
  if (objectives[obj].objective_times.size() > 1)
  {
    for (size_t t = 1; t < objectives[obj].objective_times.size(); ++t)
    {
      if (std::abs(objectives[obj].objective_times[t] - current_time) / current_time < 1.0e-12)
      {
        dt = objectives[obj].objective_times[t] - objectives[obj].objective_times[t - 1];
      }
    }
  }

  if (objectives[obj].type == "integrated control")
  {
    auto grad_over = linalg->getNewOverlappedVector(set);
    auto grad_tmp = linalg->getNewVector(set);
    auto grad_view = grad_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);

    auto offsets = wset->offsets;
    auto numDOF = assembler->groupData[block]->num_dof;

    for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
    {

      View_Sc3 local_grad("local contrib to dobj/dstate",
                          assembler->groups[block][grp]->numElem,
                          assembler->groups[block][grp]->LIDs[set].extent(1), 1);

      auto local_grad_ladev = create_mirror(LA_exec(), local_grad);

      if (!assembler->groups[block][grp]->have_sols)
      {
        assembler->performGather(set, block, grp, sol_kv[0], 0, 0);
        assembler->performGather(set, block, grp, params_kv[0], 4, 0);
      }
      assembler->updateWorksetAD(block, grp, 1, 0, true);

      // Evaluate the objective
      auto obj_dev = fman->evaluate(objectives[obj].name, "ip");

      // Weight using volumetric integration weights
      auto wts = assembler->groups[block][grp]->wts;
      auto owt = objectives[obj].weight;
      parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
        for (size_type nn=0; nn<numDOF.extent(0); nn++) {
          for (size_type dof=0; dof<numDOF(nn); dof++) {
            for (size_type pt=0; pt<wts.extent(1); pt++) {
              local_grad(elem, offsets(nn,dof),0) += -owt*obj_dev(elem,pt).fastAccessDx(offsets(nn,dof))*wts(elem,pt);
            }
          }
        } });

      if (data_avail)
      {
        assembler->scatterRes(grad_view, local_grad, assembler->groups[block][grp]->LIDs[set]);
      }
      else
      {
        Kokkos::deep_copy(local_grad_ladev, local_grad);

        if (use_host_LIDs)
        { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
          assembler->scatterRes(grad_view, local_grad_ladev, assembler->groups[block][grp]->LIDs_host[set]);
        }
        else
        { // LA_device = CUDA, AssemblyDevice = Host
          // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
          auto LIDs_dev = Kokkos::create_mirror(LA_exec(), assembler->groups[block][grp]->LIDs[set]);
          Kokkos::deep_copy(LIDs_dev, assembler->groups[block][grp]->LIDs[set]);
          assembler->scatterRes(grad_view, local_grad_ladev, LIDs_dev);
        }
      }
    }

    linalg->exportVectorFromOverlapped(set, grad_tmp, grad_over);
    grad->update(dt, *grad_tmp, 1.0);
  }
  else if (objectives[obj].type == "integrated response")
  {
    auto grad_over = linalg->getNewOverlappedVector(set);
    auto grad_tmp = linalg->getNewVector(set);
    auto grad_view = grad_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);

    auto offsets = wset->offsets;
    auto numDOF = assembler->groupData[block]->num_dof;

    ScalarT value = 0.0;
    if (objectives[obj].objective_times.size() == 1)
    { // implies steady-state
      ScalarT gcontrib = 0.0;
      ScalarT lcontrib = objectives[obj].objective_values[0];
      Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lcontrib, &gcontrib);
      value += gcontrib;
    }
    else
    {
      // Start with t=1 to ignore initial condition
      for (size_t t = 1; t < objectives[obj].objective_times.size(); ++t)
      {
        ScalarT gcontrib = 0.0;
        ScalarT lcontrib = objectives[obj].objective_values[t];
        Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lcontrib, &gcontrib);

        ScalarT dt = objectives[obj].objective_times[t] - objectives[obj].objective_times[t - 1];
        gcontrib *= dt;
        value += gcontrib;
      }
    }

    for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
    {

      View_Sc3 local_grad("local contrib to dobj/dstate",
                          assembler->groups[block][grp]->numElem,
                          assembler->groups[block][grp]->LIDs[set].extent(1), 1);

      auto local_grad_ladev = create_mirror(LA_exec(), local_grad);

      // TMW: this gives the correct gradient if we loop w a few times, but not if we only go through one
      //      this is strange because only the first loop contributes to the gradient
      //      come back to this
      for (int w = 0; w < dimension + 1; ++w)
      {
        //{
        // int w=0;
        // Seed the state and compute the solution at the ip
        if (w == 0)
        {
          if (!assembler->groups[block][grp]->have_sols)
          {
            assembler->performGather(set, block, grp, sol_kv[0], 0, 0);
            assembler->performGather(set, block, grp, params_kv[0], 4, 0);
          }
          assembler->updateWorksetAD(block, grp, 1, 0, true);
        }

        // Evaluate the objective
        // Weight using volumetric integration weights
        auto wts = assembler->groups[block][grp]->wts;

        auto obj_dev = fman->evaluate(objectives[obj].name + " response", "ip");

        parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
          for (size_type pt=0; pt<wts.extent(1); pt++) {
            obj_dev(elem,pt) = obj_dev(elem,pt)*wts(elem,pt);
          } });

        if (w == 0)
        {
          parallel_for("grp objective", RangePolicy<AssemblyExec>(0, wts.extent(0)), MRHYDE_LAMBDA(const size_type elem) {
            for (size_type nn=0; nn<numDOF.extent(0); nn++) {
              for (size_type dof=0; dof<numDOF(nn); dof++) {
                for (size_type pt=0; pt<wts.extent(1); pt++) {
                  local_grad(elem, offsets(nn,dof),0) += -obj_dev(elem,pt).fastAccessDx(offsets(nn,dof));
                }
              }
            } });
        }
      }

      if (data_avail)
      {
        assembler->scatterRes(grad_view, local_grad, assembler->groups[block][grp]->LIDs[set]);
      }
      else
      {
        Kokkos::deep_copy(local_grad_ladev, local_grad);

        if (use_host_LIDs)
        { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
          assembler->scatterRes(grad_view, local_grad_ladev, assembler->groups[block][grp]->LIDs_host[set]);
        }
        else
        { // LA_device = CUDA, AssemblyDevice = Host
          // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
          auto LIDs_dev = Kokkos::create_mirror(LA_exec(), assembler->groups[block][grp]->LIDs[set]);
          Kokkos::deep_copy(LIDs_dev, assembler->groups[block][grp]->LIDs[set]);
          assembler->scatterRes(grad_view, local_grad_ladev, LIDs_dev);
        }
      }
    }

    // ScalarT gresp = 0.0;
    // Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&intresp,&gresp);

    // Right now grad_over = dresponse/du
    // We want   grad_over = 2.0*wt*(response - target)*dresponse/du
    grad_over->scale(2.0 * objectives[obj].weight * (value - objectives[obj].target));

    linalg->exportVectorFromOverlapped(set, grad_tmp, grad_over);
    grad->update(dt, *grad_tmp, 1.0);
    // KokkosTools::print(grad);
  }
  else if (objectives[obj].type == "discrete control")
  {
    vector_RCP D_soln;
    bool fnd = datagen_soln[set]->extract(D_soln, 0, current_time);
    if (fnd)
    {
      // TMW: this is unecessarily complicated because we store the overlapped soln
      vector_RCP diff = linalg->getNewVector(set);
      vector_RCP u_no = linalg->getNewVector(set);
      vector_RCP D_no = linalg->getNewVector(set);
      u_no->doExport(*(current_soln), *(linalg->exporter[set]), Tpetra::REPLACE);
      D_no->doExport(*D_soln, *(linalg->exporter[set]), Tpetra::REPLACE);
      diff->update(1.0, *u_no, 0.0);
      diff->update(-1.0, *D_no, 1.0);
      grad->update(-2.0 * dt * objectives[obj].weight, *diff, 1.0);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: did not find a data-generating solution");
    }
  }
  else if (objectives[obj].type == "sensors")
  {

    auto grad_over = linalg->getNewOverlappedVector(set);
    auto grad_tmp = linalg->getNewVector(set);
    auto grad_view = grad_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);

    for (size_t pt = 0; pt < objectives[obj].numSensors; ++pt)
    {
      size_t tindex = 0;
      bool foundtime = false;
      for (size_type t = 0; t < objectives[obj].sensor_times.extent(0); ++t)
      {
        if (std::abs(current_time - objectives[obj].sensor_times(t)) < 1.0e-12)
        {
          foundtime = true;
          tindex = t;
        }
      }

      if (foundtime)
      {

        size_t grp = objectives[obj].sensor_owners(pt, 0);
        size_t elem = objectives[obj].sensor_owners(pt, 1);

        wset->isOnSide = true;

        auto x = wset->getScalarField("x");
        x(0, 0) = objectives[obj].sensor_points(pt, 0);
        if (dimension > 1)
        {
          auto y = wset->getScalarField("y");
          y(0, 0) = objectives[obj].sensor_points(pt, 1);
        }
        if (dimension > 2)
        {
          auto z = wset->getScalarField("z");
          z(0, 0) = objectives[obj].sensor_points(pt, 2);
        }

        auto numDOF = assembler->groupData[block]->num_dof;
        auto offsets = wset->offsets;

        View_EvalT2 u_dof("u_dof", numDOF.extent(0), assembler->groups[block][grp]->LIDs[set].extent(1));
        if (!assembler->groups[block][grp]->have_sols)
        {
          assembler->performGather(set, block, grp, sol_kv[0], 0, 0);
          assembler->performGather(set, block, grp, params_kv[0], 4, 0);
        }
        auto cu = subview(assembler->groupData[block]->sol[set], elem, ALL(), ALL());
        parallel_for("grp response get u", RangePolicy<AssemblyExec>(0, u_dof.extent(0)), MRHYDE_LAMBDA(const size_type n) {
          EvalT dummyval = 0.0;
          for (size_type n=0; n<numDOF.extent(0); n++) {
            for( int i=0; i<numDOF(n); i++ ) {
              u_dof(n,i) = EvalT(dummyval.size(),offsets(n,i),cu(n,i));
            }
          } });

        // Map the local solution to the solution and gradient at ip
        View_EvalT2 u_ip("u_ip", numDOF.extent(0), assembler->groupData[block]->dimension);
        View_EvalT2 ugrad_ip("ugrad_ip", numDOF.extent(0), assembler->groupData[block]->dimension);

        for (size_type var = 0; var < numDOF.extent(0); var++)
        {
          auto cbasis = objectives[obj].sensor_basis[wset->usebasis[var]];
          auto cbasis_grad = objectives[obj].sensor_basis_grad[wset->usebasis[var]];
          auto u_sv = subview(u_ip, var, ALL());
          auto u_dof_sv = subview(u_dof, var, ALL());
          auto ugrad_sv = subview(ugrad_ip, var, ALL());

          parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) {
            u_sv(0) += u_dof_sv(dof)*cbasis(pt,dof,0,0);
            for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
              ugrad_sv(dim) += u_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
            } });
        }

        // Map the local discretized params to param and grad at ip
        if (params->globalParamUnknowns > 0)
        {
          auto numParamDOF = assembler->groupData[block]->num_param_dof;

          View_EvalT2 p_dof("p_dof", numParamDOF.extent(0), assembler->groups[block][grp]->paramLIDs.extent(1));
          if (!assembler->groups[block][grp]->have_sols)
          {
            assembler->performGather(set, block, grp, sol_kv[0], 0, 0);
            assembler->performGather(set, block, grp, params_kv[0], 4, 0);
          }
          auto cp = subview(assembler->groupData[block]->param, elem, ALL(), ALL());
          parallel_for("grp response get u", RangePolicy<AssemblyExec>(0, p_dof.extent(0)), MRHYDE_LAMBDA(const size_type n) {
            for (size_type n=0; n<numParamDOF.extent(0); n++) {
              for( int i=0; i<numParamDOF(n); i++ ) {
                p_dof(n,i) = cp(n,i);
              }
            } });

          View_EvalT2 p_ip("p_ip", numParamDOF.extent(0), assembler->groupData[block]->dimension);
          View_EvalT2 pgrad_ip("pgrad_ip", numParamDOF.extent(0), assembler->groupData[block]->dimension);

          for (size_type var = 0; var < numParamDOF.extent(0); var++)
          {
            int bnum = wset->paramusebasis[var];
            auto btype = wset->basis_types[bnum];

            auto cbasis = objectives[obj].sensor_basis[bnum];
            auto p_sv = subview(p_ip, var, ALL());
            auto p_dof_sv = subview(p_dof, var, ALL());

            parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) { p_sv(0) += p_dof_sv(dof) * cbasis(pt, dof, 0, 0); });

            if (btype == "HGRAD")
            {
              auto cbasis_grad = objectives[obj].sensor_basis_grad[bnum];
              auto p_dof_sv = subview(p_dof, var, ALL());
              auto pgrad_sv = subview(pgrad_ip, var, ALL());

              parallel_for("grp response sensor uip", RangePolicy<AssemblyExec>(0, cbasis.extent(1)), MRHYDE_LAMBDA(const int dof) {
                for (size_t dim=0; dim<cbasis_grad.extent(3); dim++) {
                  pgrad_sv(dim) += p_dof_sv(dof)*cbasis_grad(pt,dof,0,dim);
                } });
            }
          }
          wset->setParamPoint(p_ip);
          wset->setParamGradPoint(pgrad_ip);
        }

        View_Sc3 local_grad("local contrib to dobj/dstate",
                            assembler->groups[block][grp]->numElem,
                            assembler->groups[block][grp]->LIDs[set].extent(1), 1);

        wset->setSolutionPoint(u_ip);
        wset->setSolutionGradPoint(ugrad_ip);

        auto rdata = fman->evaluate(objectives[obj].name + " response", "point");
        EvalT diff = rdata(0, 0) - objectives[obj].sensor_data(pt, tindex);
        EvalT totaldiff = objectives[obj].weight * diff * diff;
        for (size_type nn = 0; nn < numDOF.extent(0); nn++)
        {
          for (size_type dof = 0; dof < numDOF(nn); dof++)
          {
            local_grad(elem, offsets(nn, dof), 0) += -totaldiff.fastAccessDx(offsets(nn, dof)); //*wts(elem,pt);
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

template <class Node>
void PostprocessManager<Node>::computeSensitivities(vector<vector_RCP> &u,
                                                    vector<vector_RCP> &u_stage,
                                                    vector<vector_RCP> &u_prev,
                                                    vector<vector_RCP> &adjoint,
                                                    const ScalarT &current_time,
                                                    const int &tindex,
                                                    const ScalarT &deltat,
                                                    MrHyDE_OptVector &gradient)
{

  debugger->print(1, "******** Starting PostprocessManager::computeSensitivities ...");

  typedef typename Node::device_type LA_device;
  typedef Tpetra::CrsMatrix<ScalarT, LO, GO, Node> LA_CrsMatrix;
  typedef Teuchos::RCP<LA_CrsMatrix> matrix_RCP;

  if (save_adjoint_solution) {
    for (size_t set = 0; set < soln.size(); ++set) {
      adj_soln[set]->store(adjoint[set], current_time, 0);
    }
  }
  DFAD obj_sens = 0.0;
  if (response_type != "discrete") {
    this->computeObjectiveGradParam(u, current_time, deltat, obj_sens);
  }

  size_t set = 0; // hard coded for now

  auto u_kv = u[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto adjoint_kv = adjoint[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);

  // KokkosTools::print(adjoint[0],"adjoint");

  if (params->num_active_params > 0) {

    params->sacadoizeParams(true);
    params->updateDynamicParams(tindex - 1);

    // vector<ScalarT> localsens(params->num_active_params);
    auto sgrad = gradient.getParameter();
    ROL::Ptr<std::vector<ScalarT>> scalar_grad;
    if (gradient.haveDynamicScalar()) {
      scalar_grad = sgrad[tindex - 1]->getVector();
    }
    else {
      scalar_grad = sgrad[0]->getVector();
    }
    vector<ScalarT> local_grad(scalar_grad->size(), 0.0);
    vector_RCP res = linalg->getNewVector(set, params->num_active_params);
    matrix_RCP J = linalg->getNewMatrix(set);
    vector_RCP res_over = linalg->getNewOverlappedVector(set, params->num_active_params);
    matrix_RCP J_over = linalg->getNewOverlappedMatrix(set);

    auto res_kv = res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);

    res_over->putScalar(0.0);
    vector<vector_RCP> zero_vec;
    auto paramvec = params->getDiscretizedParamsOver();
    auto paramdot = params->getDiscretizedParamsDotOver();
    assembler->assembleJacRes(set, 0, u, u_stage, u_prev, u, zero_vec, zero_vec, false, true, false, false, 0,
                              res_over, J_over, isTD, current_time, false, false,            // store_adjPrev,
                              params->num_active_params, paramvec, paramdot, false, deltat); // is_final_time, deltat);

    linalg->exportVectorFromOverlapped(set, res, res_over);

    // KokkosTools::print(res,"dres/dp");

    linalg->writeToFile(J_over, res, u[0], "sens_jacobian.mm",
                        "sens_residual.mm", "sens_solution.mm");

    for (size_t paramiter = 0; paramiter < params->num_active_params; paramiter++) {
      // fine-scale
      if (assembler->groups[0][0]->group_data->multiscale) {
        ScalarT subsens = 0.0;
        for (size_t block = 0; block < assembler->groups.size(); ++block) {
          for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp) {
            subsens = -assembler->groups[block][grp]->subgradient(0, paramiter);
            local_grad[paramiter] += subsens;
          }
        }
      }
      else { // coarse-scale

        ScalarT currsens = 0.0;
        for (size_t i = 0; i < res_kv.extent(0); i++) {
          currsens += adjoint_kv(i, 0) * res_kv(i, paramiter);
        }
        local_grad[paramiter] = -currsens;
      }
    }

    ScalarT localval = 0.0;
    ScalarT globalval = 0.0;
    int numderivs = (int)obj_sens.size();
    for (size_t paramiter = 0; paramiter < params->num_active_params; paramiter++) {
      localval = local_grad[paramiter];
      Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &localval, &globalval);
      // Comm->SumAll(&localval, &globalval, 1);
      ScalarT cobj = 0.0;

      if ((int)paramiter < numderivs) {
        cobj = obj_sens.fastAccessDx(paramiter);
      }
      globalval += cobj;
      (*scalar_grad)[paramiter] += globalval;
    }
    params->sacadoizeParams(false);
    params->updateDynamicParams(tindex - 1);
  }

  int numDiscParams = params->getNumParams(4);

  if (numDiscParams > 0) {

    auto disc_grad = gradient.getField();
    vector_RCP curr_grad;
    if (gradient.haveDynamicField()) {
      curr_grad = disc_grad[tindex - 1]->getVector();
    }
    else {
      curr_grad = disc_grad[0]->getVector();
    }
    
    auto sens = this->computeDiscreteSensitivities(u, adjoint, current_time, tindex, deltat);
    curr_grad->update(1.0, *sens, 1.0);

    vector_RCP sens_over = linalg->getNewOverlappedParamVector();
    auto sens_kv = sens_over->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    
    for (size_t i = 0; i < params->paramOwnedAndShared.size(); i++) {
      ScalarT cobj = 0.0;
      if ((int)(i + params->num_active_params) < obj_sens.size()) {
        cobj = obj_sens.fastAccessDx(i + params->num_active_params);
      }
      sens_kv(i, 0) += cobj;
    }
    
    vector_RCP sensr = linalg->getNewParamVector();
    linalg->exportParamVectorFromOverlapped(sensr, sens_over);
    curr_grad->update(1.0, *sensr, 1.0);

  }
  this->saveObjectiveGradientData(gradient);

  debugger->print(1, "******** Finished PostprocessManager::computeSensitivities ...");
}

// ========================================================================================
// ========================================================================================

template <class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT, LO, GO, Node>>
PostprocessManager<Node>::computeDiscreteSensitivities(vector<vector_RCP> &u,
                                                       vector<vector_RCP> &adjoint,
                                                       const ScalarT &current_time,
                                                       const int &tindex,
                                                       const ScalarT &deltat)
{

  int set = 0; // hard-coded for now

  typedef Tpetra::CrsMatrix<ScalarT, LO, GO, Node> LA_CrsMatrix;
  typedef Teuchos::RCP<LA_CrsMatrix> matrix_RCP;
  
  vector_RCP res_over = linalg->getNewOverlappedVector(set);
  matrix_RCP J = linalg->getNewParamStateMatrix(set);
  matrix_RCP J_over = linalg->getNewOverlappedParamStateMatrix(set);
  
  res_over->putScalar(0.0);
  J->setAllToScalar(0.0);
  J_over->setAllToScalar(0.0);
  vector<vector_RCP> zero_vec;
  params->sacadoizeParams(false);
  params->updateDynamicParams(tindex - 1);

  auto Psol = params->getDiscretizedParamsOver();
  auto Pdot = params->getDiscretizedParamsDotOver();

  assembler->assembleJacRes(set, 0, u, zero_vec, zero_vec, u, zero_vec, zero_vec, true, false, true, false, 0,
                            res_over, J_over, isTD, current_time, false, false,    // store_adjPrev,
                            params->num_active_params, Psol, Pdot, false, deltat); // is_final_time, deltat);

  
  linalg->fillCompleteParamState(set, J_over);
  
  vector_RCP gradient = linalg->getNewParamVector();

  linalg->exportParamStateMatrixFromOverlapped(set, J, J_over);
  
  linalg->fillCompleteParamState(set, J);
  
  vector_RCP adj = linalg->getNewVector(set);
  adj->doExport(*(adjoint[set]), *(linalg->exporter[set]), Tpetra::REPLACE);
  J->apply(*adj, *gradient);

  return gradient;
}

