/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// For verification studies with analytical solutions, set up true solutions
// ========================================================================================

template <class Node>
vector<std::pair<string, string>> PostprocessManager<Node>::addTrueSolutions(Teuchos::ParameterList &true_solns,
                                                                             vector<vector<vector<string>>> &types,
                                                                             const int &block)
{
  // Note: errors can be measured in various norms/seminorms: L2, L2-VECTOR, L2-FACE, HGRAD, HDIV, HCURL
  // true_solns is a sublist from settings that just contains the expression for the given analytical solutions

  // Each block can have different physics, so this needs to be done per block
  vector<std::pair<string, string>> block_error_list;

  // Loop over physics sets
  for (size_t set = 0; set < varlist.size(); ++set) {
    vector<string> vars = varlist[set][block];
    vector<string> ctypes = types[set][block];

    for (size_t j = 0; j < vars.size(); j++) {

      // Different types (scalar versus vector) have different forms
      if (true_solns.isParameter(vars[j])) { // solution at volumetric ip
        if (ctypes[j].substr(0, 5) == "HGRAD" || ctypes[j].substr(0, 5) == "HVOL") {
          std::pair<string, string> newerr(vars[j], "L2");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>(vars[j], "0.0");
          assembler->addFunction(block, "true " + vars[j], expression, "ip");
        }
      }
      if (true_solns.isParameter("grad(" + vars[j] + ")[x]") || true_solns.isParameter("grad(" + vars[j] + ")[y]") || true_solns.isParameter("grad(" + vars[j] + ")[z]")) { // GRAD of the solution at volumetric ip
        if (!true_solns.isParameter("grad(" + vars[j] + ")[x]")) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error in the gradient of " + vars[j] + " but the [x] component is missing a true solution.");
        }
        if (dimension > 1) {
          if (!true_solns.isParameter("grad(" + vars[j] + ")[y]")) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error in the gradient of " + vars[j] + " but the [y] component is missing a true solution.");
          }
          if (dimension > 2) {
            if (!true_solns.isParameter("grad(" + vars[j] + ")[z]")) {
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error in the gradient of " + vars[j] + " but the [z] component is missing a true solution.");
            }
          }
        }
        if (ctypes[j].substr(0, 5) == "HGRAD") {
          std::pair<string, string> newerr(vars[j], "GRAD");
          block_error_list.push_back(newerr);

          string expression = true_solns.get<string>("grad(" + vars[j] + ")[x]", "0.0");
          assembler->addFunction(block, "true grad(" + vars[j] + ")[x]", expression, "ip");
          if (dimension > 1) {
            expression = true_solns.get<string>("grad(" + vars[j] + ")[y]", "0.0");
            assembler->addFunction(block, "true grad(" + vars[j] + ")[y]", expression, "ip");
          }
          if (dimension > 2) {
            expression = true_solns.get<string>("grad(" + vars[j] + ")[z]", "0.0");
            assembler->addFunction(block, "true grad(" + vars[j] + ")[z]", expression, "ip");
          }
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error in the gradient of " + vars[j] + " which does not use an HGRAD basis.");
        }
      }
      if (true_solns.isParameter(vars[j] + " face")) { // solution at face/side ip
        if (ctypes[j].substr(0, 5) == "HGRAD" || ctypes[j].substr(0, 5) == "HFACE") {
          std::pair<string, string> newerr(vars[j], "L2 FACE");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>(vars[j] + " face", "0.0");
          assembler->addFunction(block, "true " + vars[j], expression, "side ip");
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the face norm of the error in " + vars[j] + " which does not use an HGRAD or HFACE basis.");
        }
      }
      if (true_solns.isParameter(vars[j] + "[x]") || true_solns.isParameter(vars[j] + "[y]") || true_solns.isParameter(vars[j] + "[z]")) { // vector solution at volumetric ip
        if (!true_solns.isParameter(vars[j] + "[x]")) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error of " + vars[j] + " but the [x] component is missing a true solution.");
        }
        if (dimension > 1) {
          if (!true_solns.isParameter(vars[j] + "[y]")) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error of " + vars[j] + " but the [y] component is missing a true solution.");
          }
          if (dimension > 2) {
            if (!true_solns.isParameter(vars[j] + "[z]")) {
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error of " + vars[j] + " but the [z] component is missing a true solution.");
            }
          }
        }
        if (ctypes[j].substr(0, 4) == "HDIV" || ctypes[j].substr(0, 5) == "HCURL") {
          std::pair<string, string> newerr(vars[j], "L2 VECTOR");
          block_error_list.push_back(newerr);

          string expression = true_solns.get<string>(vars[j] + "[x]", "0.0");
          assembler->addFunction(block, "true " + vars[j] + "[x]", expression, "ip");

          if (dimension > 1) {
            expression = true_solns.get<string>(vars[j] + "[y]", "0.0");
            assembler->addFunction(block, "true " + vars[j] + "[y]", expression, "ip");
          }
          if (dimension > 2) {
            expression = true_solns.get<string>(vars[j] + "[z]", "0.0");
            assembler->addFunction(block, "true " + vars[j] + "[z]", expression, "ip");
          }
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute a component of the norm of a error for " + vars[j] + " which does not use a vector basis.");
        }
      }
      if (true_solns.isParameter("div(" + vars[j] + ")")) { // div of solution at volumetric ip
        if (ctypes[j].substr(0, 4) == "HDIV") {
          std::pair<string, string> newerr(vars[j], "DIV");
          block_error_list.push_back(newerr);
          string expression = true_solns.get<string>("div(" + vars[j] + ")", "0.0");
          assembler->addFunction(block, "true div(" + vars[j] + ")", expression, "ip");
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error in the divergence of " + vars[j] + " which does not use an HDIV basis.");
        }
      }
      if (true_solns.isParameter("curl(" + vars[j] + ")[x]") || true_solns.isParameter("curl(" + vars[j] + ")[y]") || true_solns.isParameter("curl(" + vars[j] + ")[z]")) { // vector solution at volumetric ip
        if (!true_solns.isParameter("curl(" + vars[j] + ")[x]")) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error in the curl of " + vars[j] + " but the [x] component is missing a true solution.");
        }
        if (dimension > 1) {
          if (!true_solns.isParameter("curl(" + vars[j] + ")[y]")) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error in the curl of " + vars[j] + " but the [y] component is missing a true solution.");
          }
          if (dimension > 2) {
            if (!true_solns.isParameter("curl(" + vars[j] + ")[z]")) {
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error in the curl of " + vars[j] + " but the [z] component is missing a true solution.");
            }
          }
        }
        if (ctypes[j].substr(0, 5) == "HCURL") {
          std::pair<string, string> newerr(vars[j], "CURL");
          block_error_list.push_back(newerr);

          string expression = true_solns.get<string>("curl(" + vars[j] + ")[x]", "0.0");
          assembler->addFunction(block, "true curl(" + vars[j] + ")[x]", expression, "ip");

          if (dimension > 1) {
            expression = true_solns.get<string>("curl(" + vars[j] + ")[y]", "0.0");
            assembler->addFunction(block, "true curl(" + vars[j] + ")[y]", expression, "ip");
          }
          if (dimension > 2) {
            expression = true_solns.get<string>("curl(" + vars[j] + ")[z]", "0.0");
            assembler->addFunction(block, "true curl(" + vars[j] + ")[z]", expression, "ip");
          }
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error: MrHyDE was asked to compute the norm of the error in the curl of " + vars[j] + " which does not use an HCURL basis.");
        }
      }
    }
  }
  return block_error_list;
}

// ========================================================================================
// Compute the error in various requested norms given a user-defined true solution and the current state
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::computeError(vector<vector_RCP> &current_soln, const ScalarT &currenttime)
{

  Teuchos::TimeMonitor localtimer(*computeErrorTimer);

  debugger->print(1, "**** Starting PostprocessManager::computeError(time)");

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

  error_times.push_back(currenttime);

  vector<Kokkos::View<ScalarT *, HostDevice>> currerror;
  int seedwhat = 0;

  for (size_t block = 0; block < assembler->groups.size(); block++) { // loop over blocks

    int altblock; // Needed for subgrid error calculations
    if (assembler->wkset.size() > block && error_list.size() > block) {
      altblock = block;
    }
    else {
      altblock = 0;
    }
    // groups can use block, but everything else should be altblock
    // This is due to how the subgrid models store the groups

    Kokkos::View<ScalarT *, HostDevice> blockerrors("error", error_list[altblock].size());

    if (assembler->groups[block].size() > 0) {

      assembler->wkset[altblock]->setTime(currenttime);

      // Need to use time step solution instead of stage solution
      bool isTransient = assembler->wkset[altblock]->isTransient;
      assembler->wkset[altblock]->isTransient = false;
      assembler->groupData[altblock]->requires_transient = false;

      // Determine what needs to be updated in the workset
      bool have_vol_errs = false, have_face_errs = false;
      for (size_t etype = 0; etype < error_list[altblock].size(); etype++)
      {
        if (error_list[altblock][etype].second == "L2" || error_list[altblock][etype].second == "GRAD" || error_list[altblock][etype].second == "DIV" || error_list[altblock][etype].second == "CURL" || error_list[altblock][etype].second == "L2 VECTOR")
        {
          have_vol_errs = true;
        }
        if (error_list[altblock][etype].second == "L2 FACE")
        {
          have_face_errs = true;
        }
      }
      for (size_t grp = 0; grp < assembler->groups[block].size(); grp++)
      {
        if (assembler->groups[block][grp]->active)
        {
          for (size_t set = 0; set < sol_kv.size(); ++set)
          {
            if (!assembler->groups[block][grp]->have_sols)
            {
              assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
            }
          }
          if (have_vol_errs)
          {
            assembler->updateWorkset(assembler->wkset[altblock], block, grp, seedwhat, 0, true);
            // assembler->updateWorkset(altblock, grp, seedwhat,true);
          }
          // auto wts = assembler->wkset[block]->wts;
          auto wts = assembler->wkset[altblock]->wts;

          for (size_t etype = 0; etype < error_list[altblock].size(); etype++)
          {
            string varname = error_list[altblock][etype].first;

            if (error_list[altblock][etype].second == "L2")
            {
              // compute the true solution
              string name = varname;
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
              auto sol = assembler->wkset[altblock]->getSolutionField(name);

              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                } }, error);
              blockerrors(etype) += error;
            }
            else if (error_list[altblock][etype].second == "GRAD")
            {
              // compute the true x-component of grad
              string name = "grad(" + varname + ")[x]";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
              auto sol_x = assembler->wkset[altblock]->getSolutionField(name);
              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_x(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                } }, error);
              blockerrors(etype) += error;

              if (dimension > 1)
              {
                // compute the true y-component of grad
                string name = "grad(" + varname + ")[y]";
                View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
                auto sol_y = assembler->wkset[altblock]->getSolutionField(name);

                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    ScalarT diff = sol_y(elem,pt) - tsol(elem,pt);
                    update += diff*diff*wts(elem,pt);
                  } }, error);
                blockerrors(etype) += error;
              }

              if (dimension > 2)
              {
                // compute the true z-component of grad
                string name = "grad(" + varname + ")[z]";
                View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
                auto sol_z = assembler->wkset[altblock]->getSolutionField(name);

                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    ScalarT diff = sol_z(elem,pt) - tsol(elem,pt);
                    update += diff*diff*wts(elem,pt);
                  } }, error);
                blockerrors(etype) += error;
              }
            }
            else if (error_list[altblock][etype].second == "DIV")
            {
              // compute the true divergence
              string name = "div(" + varname + ")";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
              auto sol_div = assembler->wkset[altblock]->getSolutionField(name);

              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_div(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                } }, error);
              blockerrors(etype) += error;
            }
            else if (error_list[altblock][etype].second == "CURL")
            {
              // compute the true x-component of grad
              string name = "curl(" + varname + ")[x]";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
              auto sol_curl_x = assembler->wkset[altblock]->getSolutionField(name);

              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_curl_x(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                } }, error);
              blockerrors(etype) += error;

              if (dimension > 1)
              {
                // compute the true y-component of grad
                string name = "curl(" + varname + ")[y]";
                View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
                auto sol_curl_y = assembler->wkset[altblock]->getSolutionField(name);

                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    ScalarT diff = sol_curl_y(elem,pt) - tsol(elem,pt);
                    update += diff*diff*wts(elem,pt);
                  } }, error);
                blockerrors(etype) += error;
              }

              if (dimension > 2)
              {
                // compute the true z-component of grad
                string name = "curl(" + varname + ")[z]";
                View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
                auto sol_curl_z = assembler->wkset[altblock]->getSolutionField(name);

                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    ScalarT diff = sol_curl_z(elem,pt) - tsol(elem,pt);
                    update += diff*diff*wts(elem,pt);
                  } }, error);
                blockerrors(etype) += error;
              }
            }
            else if (error_list[altblock][etype].second == "L2 VECTOR")
            {
              // compute the true x-component of grad
              string name = varname + "[x]";
              View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
              auto sol_x = assembler->wkset[altblock]->getSolutionField(name);

              // add in the L2 difference at the volumetric ip
              ScalarT error = 0.0;
              parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                  ScalarT diff = sol_x(elem,pt) - tsol(elem,pt);
                  update += diff*diff*wts(elem,pt);
                } }, error);
              blockerrors(etype) += error;

              if (dimension > 1)
              {
                // compute the true y-component of grad
                string name = varname + "[y]";
                View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
                auto sol_y = assembler->wkset[altblock]->getSolutionField(name);

                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    ScalarT diff = sol_y(elem,pt) - tsol(elem,pt);
                    update += diff*diff*wts(elem,pt);
                  } }, error);
                blockerrors(etype) += error;
              }

              if (dimension > 2)
              {
                // compute the true z-component of grad
                string name = varname + "[z]";
                View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "ip");
                auto sol_z = assembler->wkset[altblock]->getSolutionField(name);

                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    ScalarT diff = sol_z(elem,pt) - tsol(elem,pt);
                    update += diff*diff*wts(elem,pt);
                  } }, error);
                blockerrors(etype) += error;
              }
            }
          }
        }
        if (have_face_errs)
        {
          assembler->wkset[altblock]->isOnSide = true;
          for (size_t face = 0; face < assembler->groups[block][grp]->group_data->num_sides; face++)
          {
            // TMW - hard coded for now
            for (size_t set = 0; set < assembler->wkset[altblock]->numSets; ++set)
            {
              assembler->wkset[altblock]->computeSolnSteadySeeded(set, assembler->groupData[block]->sol[set], seedwhat);
            }
            assembler->updateWorksetFace(block, grp, face);
            assembler->wkset[altblock]->resetSolutionFields();
            for (size_t etype = 0; etype < error_list[altblock].size(); etype++)
            {
              string varname = error_list[altblock][etype].first;
              if (error_list[altblock][etype].second == "L2 FACE")
              {
                // compute the true z-component of grad
                string name = varname;
                View_Sc2 tsol = assembler->evaluateFunction(altblock, "true " + name, "side ip");

                auto sol = assembler->wkset[altblock]->getSolutionField(name);
                auto wts = assembler->wkset[block]->wts_side;

                // add in the L2 difference at the volumetric ip
                ScalarT error = 0.0;
                parallel_reduce(RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_LAMBDA(const int elem, ScalarT &update) {
                  double facemeasure = 0.0;
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    facemeasure += wts(elem,pt);
                  }
                  
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    ScalarT diff = sol(elem,pt) - tsol(elem,pt);
                    update += 0.5/facemeasure*diff*diff*wts(elem,pt);  // TODO - BWR what is this? why .5?
                  } }, error);
                blockerrors(etype) += error;
              }
            }
          }
          assembler->wkset[altblock]->isOnSide = false;
        }
      }
      assembler->wkset[altblock]->isTransient = isTransient;
      assembler->groupData[altblock]->requires_transient = isTransient;
    }
    currerror.push_back(blockerrors);
  } // end block loop

  // Need to move currerrors to Host
  vector<Kokkos::View<ScalarT *, HostDevice>> host_error;
  for (size_t k = 0; k < currerror.size(); k++)
  {
    Kokkos::View<ScalarT *, HostDevice> host_cerr("error on host", currerror[k].extent(0));
    Kokkos::deep_copy(host_cerr, currerror[k]);
    host_error.push_back(host_cerr);
  }

  errors.push_back(host_error);

  if (!(Teuchos::is_null(multiscale_manager)))
  {
    if (multiscale_manager->getNumberSubgridModels() > 0)
    {
      // Collect all of the errors for each subgrid model
      vector<vector<Kokkos::View<ScalarT *, HostDevice>>> blocksgerrs;

      for (size_t block = 0; block < assembler->groups.size(); block++)
      { // loop over blocks

        vector<Kokkos::View<ScalarT *, HostDevice>> sgerrs;
        for (size_t m = 0; m < multiscale_manager->getNumberSubgridModels(); m++)
        {
          Kokkos::View<ScalarT *, HostDevice> err = multiscale_manager->subgridModels[m]->computeError(currenttime);
          sgerrs.push_back(err);
        }
        blocksgerrs.push_back(sgerrs);
      }

      subgrid_errors.push_back(blocksgerrs);
    }
  }

  debugger->print(1, "**** Finished PostprocessManager::computeError(time)");
}

// ========================================================================================
// ========================================================================================

template <class Node>
ScalarT PostprocessManager<Node>::computeDualWeightedResidual(vector<vector_RCP> &u,
                                                              vector<vector_RCP> &adjoint,
                                                              const ScalarT &current_time,
                                                              const int &tindex,
                                                              const ScalarT &deltat)
{

  debugger->print(1, "******** Starting PostprocessManager::computeDualWeightedResidual ...");

  typedef Tpetra::CrsMatrix<ScalarT, LO, GO, Node> LA_CrsMatrix;
  typedef Teuchos::RCP<LA_CrsMatrix> matrix_RCP;

  size_t set = 0; // hard coded for now
  size_t stage = 0;
  // adjoint solution is overlapped
  vector_RCP adj = linalg->getNewVector(set);
  linalg->exportVectorFromOverlapped(set, adj, adjoint[set]);

  // auto adjoint_kv = adj->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);

  vector_RCP res = linalg->getNewVector(set, params->num_active_params);
  //  matrix_RCP J = linalg->getNewMatrix(set);
  vector_RCP res_over = linalg->getNewOverlappedVector(set, params->num_active_params);
  matrix_RCP J_over; // = linalg->getNewOverlappedMatrix(set);

  res_over->putScalar(0.0);
  vector<vector_RCP> zero_vec;
  auto Psol = params->getDiscretizedParamsOver();
  auto Pdot = params->getDiscretizedParamsDotOver();
  assembler->assembleJacRes(set, stage, u, zero_vec, zero_vec, u, zero_vec, zero_vec, false, false, false, false, 0,
                            res_over, J_over, isTD, current_time, false, false,    // store_adjPrev,
                            params->num_active_params, Psol, Pdot, false, deltat); // is_final_time, deltat);

  linalg->exportVectorFromOverlapped(set, res, res_over);

  res->dot(*adj, dotprod);

  debugger->print(1, "******** Finished PostprocessManager::computeDualWeightedResidual ...");

  return dotprod[0];
}
