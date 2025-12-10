/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// Main visualization routine - writes to an exodus file
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::writeSolution(vector<vector_RCP> &current_soln, const ScalarT &currenttime)
{

  Teuchos::TimeMonitor localtimer(*writeSolutionTimer);

  debugger->print(1, "******** Starting PostprocessManager::writeSolution() ...");

  typedef typename Node::execution_space LA_exec;
  typedef typename Node::device_type LA_device;

  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible)
  {
    data_avail = false;
  }

  // Grab slices of Kokkos Views and push to AssembleDevice one time for each state vector
  vector<Kokkos::View<ScalarT *, AssemblyDevice>> sol_kv, params_kv;
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

  // Grab slices of Kokkos Views and push to AssembleDevice one time for each discretized parameter vector
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

  // Store the current vis time (subset of solve times)
  plot_times.push_back(currenttime);

  // Loop over element blocks
  // Easier to have this as the outer loop
  for (size_t block = 0; block < blocknames.size(); ++block)
  {

    // Get the block name
    std::string blockID = blocknames[block];

    // Create a std::vector of element ids on this block
    // Disc interface stores them as a Kokkos view to track memory
    auto myElements_tmp = disc->my_elements[block];
    vector<size_t> myElements(myElements_tmp.extent(0));
    for (size_t i = 0; i < myElements_tmp.extent(0); ++i)
    {
      myElements[i] = myElements_tmp(i);
    }

    // Nothing is required if this processor does not own any elements on this block
    // This happens all the time
    if (myElements.size() > 0)
    {

      // Loop over physics sets
      for (size_t set = 0; set < setnames.size(); ++set)
      {

        // Make sure everything knows what set we are on
        assembler->updatePhysicsSet(set);

        // Get a few lists from physics for this block/set
        vector<string> vartypes = physics->types[set][block];
        vector<int> varorders = physics->orders[set][block];
        int numVars = physics->num_vars[set][block]; // probably redundant

        // Loop over the state variables
        for (int n = 0; n < numVars; n++)
        {

          if (vartypes[n] == "HGRAD")
          {
            if (assembler->groups[block][0]->group_data->require_basis_at_nodes)
            {
              // The actual solution data (on device)
              Kokkos::View<ScalarT **, AssemblyDevice> soln_dev = Kokkos::View<ScalarT **, AssemblyDevice>("solution", myElements.size(),
                                                                                                           numNodesPerElem);
              // Solution data on host that will be written to the file
              auto soln_computed = Kokkos::create_mirror_view(soln_dev);

              // Fill data on device
              for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
              {
                // Gather is probably necessary (checks internally)
                assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
                
                auto eID = assembler->groups[block][grp]->localElemID;
                auto tmpsol = assembler->getSolutionAtNodes(block, grp, n);
                auto sol = Kokkos::subview(tmpsol, Kokkos::ALL(), Kokkos::ALL(), 0); // last component is dimension, which is 0 for HGRAD
                parallel_for("postproc plot param HGRAD", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
                  for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                    soln_dev(eID(elem),i) = sol(elem,i);
                  } });
              }
              // Copy to host
              Kokkos::deep_copy(soln_computed, soln_dev);
              
              // Write to file
              mesh->setSolutionFieldData(varlist[set][block][n] + append, blockID, myElements, soln_computed);
            }
            else
            {
              // The actual solution data (on device)
              Kokkos::View<ScalarT **, AssemblyDevice> soln_dev = Kokkos::View<ScalarT **, AssemblyDevice>("solution", myElements.size(), numNodesPerElem);

              // Solution data on host that will be written to the file
              auto soln_computed = Kokkos::create_mirror_view(soln_dev);

              // Fill data on device
              for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
              {
                auto eID = assembler->groups[block][grp]->localElemID;

                // Gather is probably necessary (checks internally)
                assembler->performGather(set, block, grp, sol_kv[set], 0, 0);

                // Fill data on device
                auto sol = Kokkos::subview(assembler->groupData[block]->sol[set], Kokkos::ALL(), n, Kokkos::ALL());
                parallel_for("postproc plot HGRAD", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
                  for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                    soln_dev(eID(elem),i) = sol(elem,i);
                  } });
              }

              // Copy to host
              Kokkos::deep_copy(soln_computed, soln_dev);

              // Write to file
              mesh->setSolutionFieldData(varlist[set][block][n] + append, blockID, myElements, soln_computed);
            }
          }
          else if (vartypes[n] == "HVOL")
          {
            // The actual solution data (on device)
            Kokkos::View<ScalarT *, AssemblyDevice> soln_dev("solution", myElements.size());

            // Solution data on host that will be written to the file
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);

            // Fill on device
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
              auto sol = Kokkos::subview(assembler->groupData[block]->sol[set], Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot HVOL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
                soln_dev(eID(elem)) = sol(elem, 0); // u_kv(pindex,0);
              });
            }

            // Copy to host
            Kokkos::deep_copy(soln_computed, soln_dev);

            // Write to file
            mesh->setCellFieldData(varlist[set][block][n] + append, blockID, myElements, soln_computed);
          }
          else if (vartypes[n] == "HDIV" || vartypes[n] == "HCURL")
          { // need to project each component onto PW-linear basis and PW constant basis
            // The actual solution data (on device)
            Kokkos::View<ScalarT *, AssemblyDevice> soln_x_dev("solution", myElements.size());
            Kokkos::View<ScalarT *, AssemblyDevice> soln_y_dev("solution", myElements.size());
            Kokkos::View<ScalarT *, AssemblyDevice> soln_z_dev("solution", myElements.size());

            // Solution data on host that will be written to the file
            auto soln_x = Kokkos::create_mirror_view(soln_x_dev);
            auto soln_y = Kokkos::create_mirror_view(soln_y_dev);
            auto soln_z = Kokkos::create_mirror_view(soln_z_dev);

            // Storage on device for solution averages
            View_Sc2 sol("average solution", assembler->groupData[block]->num_elem, dimension);

            // Fill on device
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
              // Compute the element average
              assembler->computeSolutionAverage(block, grp, varlist[set][block][n], sol);
              parallel_for("postproc plot HDIV/HCURL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
                soln_x_dev(eID(elem)) = sol(elem,0);
                if (sol.extent(1) > 1) {
                  soln_y_dev(eID(elem)) = sol(elem,1);
                }
                if (sol.extent(1) > 2) {
                  soln_z_dev(eID(elem)) = sol(elem,2);
                } });
            }

            // Copy to host
            Kokkos::deep_copy(soln_x, soln_x_dev);
            Kokkos::deep_copy(soln_y, soln_y_dev);
            Kokkos::deep_copy(soln_z, soln_z_dev);

            // Write to file
            mesh->setCellFieldData(varlist[set][block][n] + append + "x", blockID, myElements, soln_x);
            if (dimension > 1)
            {
              mesh->setCellFieldData(varlist[set][block][n] + append + "y", blockID, myElements, soln_y);
            }
            if (dimension > 2)
            {
              mesh->setCellFieldData(varlist[set][block][n] + append + "z", blockID, myElements, soln_z);
            }
          }
          else if (vartypes[n] == "HFACE" && write_HFACE_variables)
          {

            // The actual solution data (on device)
            Kokkos::View<ScalarT *, AssemblyDevice> soln_faceavg_dev("solution", myElements.size());

            // Solution data on host that will be written to the file
            auto soln_faceavg = Kokkos::create_mirror_view(soln_faceavg_dev);

            // Storage for the measure of each face (on device)
            Kokkos::View<ScalarT *, AssemblyDevice> face_measure_dev("face measure", myElements.size());

            // Convince the workset we are working with a side (temporarily)
            assembler->wkset[block]->isOnSide = true;

            // Fill on device
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              for (size_t face = 0; face < assembler->groupData[block]->num_sides; face++)
              {
                int seedwhat = 0;
                for (size_t iset = 0; iset < assembler->wkset[block]->numSets; ++iset)
                {
                  assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
                  assembler->wkset[block]->computeSolnSteadySeeded(iset, assembler->groupData[block]->sol[iset], seedwhat);
                }
                assembler->updateWorksetFace(block, grp, face);
                auto wts = assembler->wkset[block]->wts_side;
                auto sol = assembler->wkset[block]->getSolutionField(varlist[set][block][n]);
                parallel_for("postproc plot HFACE", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
                  for( size_t pt=0; pt<wts.extent(1); pt++ ) {
                    face_measure_dev(eID(elem)) += wts(elem,pt);
                    soln_faceavg_dev(eID(elem)) += sol(elem,pt)*wts(elem,pt);
                  } });
              }
            }

            // Reset the workset to volume instead of side
            assembler->wkset[block]->isOnSide = false;

            // Compute the face average
            parallel_for("postproc plot HFACE 2", RangePolicy<AssemblyExec>(0, soln_faceavg_dev.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) { soln_faceavg_dev(elem) *= 1.0 / face_measure_dev(elem); });

            // Copy to host
            Kokkos::deep_copy(soln_faceavg, soln_faceavg_dev);

            // Write to file
            mesh->setCellFieldData(varlist[set][block][n] + append, blockID, myElements, soln_faceavg);
          }
        }
      }

      ////////////////////////////////////////////////////////////////
      // Discretized Parameters
      ////////////////////////////////////////////////////////////////

      // Grab the list of discretized parameters
      vector<string> dpnames = params->discretized_param_names;

      // Check if we actually have any
      if (dpnames.size() > 0)
      {

        // Grab the actual disc. param. basis information
        vector<int> numParamBasis = params->paramNumBasis;
        vector<int> dp_usebasis = params->discretized_param_usebasis;
        vector<string> discParamTypes = params->discretized_param_basis_types;

        // Loop ove disc. params and add to mesh
        for (size_t n = 0; n < dpnames.size(); n++)
        {
          int bnum = dp_usebasis[n];
          if (discParamTypes[bnum] == "HGRAD")
          {

            // The actual solution data (on device)
            Kokkos::View<ScalarT **, AssemblyDevice> soln_dev = Kokkos::View<ScalarT **, AssemblyDevice>("solution", myElements.size(),
                                                                                                         numNodesPerElem);

            // Solution data on host that will be written to the file
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);

            // Fill on device
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              assembler->performGather(0, block, grp, params_kv[0], 4, 0);

              auto sol = Kokkos::subview(assembler->groupData[block]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HGRAD", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
                for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                  soln_dev(eID(elem),i) = sol(elem,i);
                } });
            }

            // Copy to host
            Kokkos::deep_copy(soln_computed, soln_dev);

            // Write to file
            mesh->setSolutionFieldData(dpnames[n] + append, blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HVOL")
          {

            // The actual solution data (on device)
            Kokkos::View<ScalarT *, AssemblyDevice> soln_dev("solution", myElements.size());

            // Solution data on host that will be written to the file
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);

            // Fill on device
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              assembler->performGather(0, block, grp, params_kv[0], 4, 0);

              auto sol = Kokkos::subview(assembler->groupData[block]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HVOL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) { soln_dev(eID(elem)) = sol(elem, 0); });
            }

            // Copy to host
            Kokkos::deep_copy(soln_computed, soln_dev);

            // Write to file
            mesh->setCellFieldData(dpnames[n] + append, blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HDIV" || discParamTypes[n] == "HCURL")
          {

            // The actual solution data (on device)
            Kokkos::View<ScalarT *, AssemblyDevice> soln_x_dev("solution", myElements.size());
            Kokkos::View<ScalarT *, AssemblyDevice> soln_y_dev("solution", myElements.size());
            Kokkos::View<ScalarT *, AssemblyDevice> soln_z_dev("solution", myElements.size());

            // Solution data on host that will be written to the file
            auto soln_x = Kokkos::create_mirror_view(soln_x_dev);
            auto soln_y = Kokkos::create_mirror_view(soln_y_dev);
            auto soln_z = Kokkos::create_mirror_view(soln_z_dev);

            // Solution average (on device)
            View_Sc2 sol("average solution", assembler->groupData[block]->num_elem, dimension);

            // Fill on device
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              assembler->performGather(0, block, grp, params_kv[0], 4, 0);
              assembler->computeParameterAverage(block, grp, dpnames[n], sol);
              parallel_for("postproc plot HDIV/HCURL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
                soln_x_dev(eID(elem)) = sol(elem,0);
                if (sol.extent(1) > 1) {
                  soln_y_dev(eID(elem)) = sol(elem,1);
                }
                if (sol.extent(1) > 2) {
                  soln_z_dev(eID(elem)) = sol(elem,2);
                } });
            }

            // Copy to host
            Kokkos::deep_copy(soln_x, soln_x_dev);
            Kokkos::deep_copy(soln_y, soln_y_dev);
            Kokkos::deep_copy(soln_z, soln_z_dev);

            // Write to file
            mesh->setCellFieldData(dpnames[n] + append + "x", blockID, myElements, soln_x);
            if (dimension > 1)
            {
              mesh->setCellFieldData(dpnames[n] + append + "y", blockID, myElements, soln_y);
            }
            if (dimension > 2)
            {
              mesh->setCellFieldData(dpnames[n] + append + "z", blockID, myElements, soln_z);
            }
          }
        }
      }

      ////////////////////////////////////////////////////////////////
      // Extra nodal fields (PW linear/bilinear/trilinear fields)
      ////////////////////////////////////////////////////////////////
      // TMW: This needs to be rewritten to actually use integration points
      //      Filling with all zeros for now
      vector<string> extrafieldnames = extrafields_list[block];
      for (size_t j = 0; j < extrafieldnames.size(); j++)
      {
        Kokkos::View<ScalarT **, HostDevice> efd("field data", myElements.size(), numNodesPerElem);
        mesh->setSolutionFieldData(extrafieldnames[j], blockID, myElements, efd);
      }

      ////////////////////////////////////////////////////////////////
      // Extra cell fields (PW constant fields)
      ////////////////////////////////////////////////////////////////

      if (extracellfields_list[block].size() > 0)
      {

        // Storage for field (on device)
        Kokkos::View<ScalarT **, AssemblyDevice> ecd_dev("grp data", myElements.size(),
                                                         extracellfields_list[block].size());

        // Storage for field (on host)
        auto ecd = Kokkos::create_mirror_view(ecd_dev);

        // Fill on device
        for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
        {
          auto eID = assembler->groups[block][grp]->localElemID;
          int set = 0; // TMW: why is this hard-coded?
          assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
          assembler->performGather(0, block, grp, params_kv[0], 4, 0);
          assembler->updateWorkset(block, grp, 0, 0, true);
          assembler->wkset[block]->setTime(currenttime);

          auto cfields = this->getExtraCellFields(block, assembler->groups[block][grp]->wts);

          parallel_for("postproc plot param HVOL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
            for (size_type r=0; r<cfields.extent(1); ++r) {
              ecd_dev(eID(elem),r) = cfields(elem,r);
            } });
        }

        // Copy to host
        Kokkos::deep_copy(ecd, ecd_dev);

        // Write to file
        for (size_t j = 0; j < extracellfields_list[block].size(); j++)
        {
          auto ccd = subview(ecd, ALL(), j);
          Kokkos::View<ScalarT *, HostDevice> tmpccd("temp dq", ccd.extent(0));
          deep_copy(tmpccd, ccd);
          mesh->setCellFieldData(extracellfields_list[block][j] + append, blockID, myElements, tmpccd);
        }
      }

      ////////////////////////////////////////////////////////////////
      // Derived quantities from physics modules
      // Values averaged over each element
      ////////////////////////////////////////////////////////////////

      if (derivedquantities_list[block].size() > 0)
      {

        // Storage for field (on device)
        Kokkos::View<ScalarT **, AssemblyDevice> dq_dev("grp data", myElements.size(),
                                                        derivedquantities_list[block].size());

        // Storage for field (on host)
        auto dq = Kokkos::create_mirror_view(dq_dev);

        // Fill on device
        for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
        {
          auto eID = assembler->groups[block][grp]->localElemID;
          int set = 0; // TMW: why is this hard-coded?
          assembler->performGather(set, block, grp, sol_kv[set], 0, 0);
          assembler->updateWorkset(block, grp, 0, 0, true);
          assembler->wkset[block]->setTime(currenttime);

          auto cfields = this->getDerivedQuantities(block, assembler->groups[block][grp]->wts);

          parallel_for("postproc plot param HVOL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
            for (size_type r=0; r<cfields.extent(1); ++r) {
              dq_dev(eID(elem),r) = cfields(elem,r);
            } });
        }

        // Copy to host
        Kokkos::deep_copy(dq, dq_dev);

        // Write to file
        for (size_t j = 0; j < derivedquantities_list[block].size(); j++)
        {
          auto cdq = subview(dq, ALL(), j);
          Kokkos::View<ScalarT *, HostDevice> tmpcdq("temp dq", cdq.extent(0));
          deep_copy(tmpcdq, cdq);
          mesh->setCellFieldData(derivedquantities_list[block][j] + append, blockID, myElements, tmpcdq);
        }
      }

      ////////////////////////////////////////////////////////////////
      // Seeds for crystal elasticity/plasticity
      ////////////////////////////////////////////////////////////////

      // Check if this data is used
      if (assembler->groups[block][0]->group_data->have_phi ||
          assembler->groups[block][0]->group_data->have_rotation ||
          assembler->groups[block][0]->group_data->have_extra_data)
      {

        // Allocate storage for elements data and seed (on host)
        Kokkos::View<ScalarT *, HostDevice> cdata("data", myElements.size());
        Kokkos::View<ScalarT *, HostDevice> cseed("data seed", myElements.size());

        // Fill on host
        for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
        {
          vector<size_t> data_seed = assembler->groups[block][grp]->data_seed;
          vector<size_t> data_seedindex = assembler->groups[block][grp]->data_seedindex;
          Kokkos::View<ScalarT **, AssemblyDevice> data = assembler->groups[block][grp]->data;
          Kokkos::View<LO *, AssemblyDevice> eID = assembler->groups[block][grp]->localElemID;

          // Copy element IDs to host
          auto host_eID = Kokkos::create_mirror_view(eID);
          Kokkos::deep_copy(host_eID, eID);

          for (size_type p = 0; p < host_eID.extent(0); p++)
          {
            if (data.extent(1) == 1)
            {
              cdata(host_eID(p)) = data(p, 0);
            }
            cseed(host_eID(p)) = data_seedindex[p];
          }
        }

        // Write to file
        string name = "mesh_data_seed";
        mesh->setCellFieldData(name, blockID, myElements, cseed);
        name = "mesh_data";
        mesh->setCellFieldData(name, blockID, myElements, cdata);
      }

      ////////////////////////////////////////////////////////////////
      // Group number
      // Useful to see how elements get grouped together
      ////////////////////////////////////////////////////////////////

      if (write_group_number)
      {

        // Allocate storage (on device)
        Kokkos::View<ScalarT *, AssemblyDevice> grpnum_dev("grp number", myElements.size());

        // Storage on host
        auto grpnum = Kokkos::create_mirror_view(grpnum_dev);

        // Fill on device
        for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
        {
          auto eID = assembler->groups[block][grp]->localElemID;
          parallel_for("postproc plot param HVOL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) { grpnum_dev(eID(elem)) = grp; });
        }

        // Copt to host
        Kokkos::deep_copy(grpnum, grpnum_dev);

        // Write to file
        mesh->setCellFieldData("group number", blockID, myElements, grpnum);
      }

      ////////////////////////////////////////////////////////////////
      // Database IDs
      // Very useful to assess compression
      ////////////////////////////////////////////////////////////////

      if (write_database_id)
      {

        // Allocate storage on device
        Kokkos::View<ScalarT *, AssemblyDevice> jacnum_dev("unique jac ID", myElements.size());

        // Storage on host
        auto jacnum = Kokkos::create_mirror_view(jacnum_dev);

        // Fill on device
        for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
        {
          auto index = assembler->groups[block][grp]->basis_index;
          auto eID = assembler->groups[block][grp]->localElemID;
          parallel_for("postproc plot param HVOL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
            jacnum_dev(eID(elem)) = index(elem); // TMW: is this what we want?
          });
        }

        // Copy to host
        Kokkos::deep_copy(jacnum, jacnum_dev);

        // Write to file
        mesh->setCellFieldData("unique Jacobian ID", blockID, myElements, jacnum);
      }

      ////////////////////////////////////////////////////////////////
      // Subgrid model each coarse element uses
      // Useful for dynamic adaptive subgrid modeling
      ////////////////////////////////////////////////////////////////

      if (write_subgrid_model)
      {

        // Allocate storage on device
        Kokkos::View<ScalarT *, AssemblyDevice> sgmodel_dev("subgrid model", myElements.size());

        // Storage on host
        auto sgmodel = Kokkos::create_mirror_view(sgmodel_dev);

        // Fill on device
        for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
        {
          int sgindex = assembler->groups[block][grp]->subgrid_model_index;
          auto eID = assembler->groups[block][grp]->localElemID;
          parallel_for("postproc plot param HVOL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) { sgmodel_dev(eID(elem)) = sgindex; });
        }

        // Copy to host
        Kokkos::deep_copy(sgmodel, sgmodel_dev);

        // Write to file
        mesh->setCellFieldData("subgrid model", blockID, myElements, sgmodel);
      }
    }
  }

  ////////////////////////////////////////////////////////////////
  // Write to Exodus
  ////////////////////////////////////////////////////////////////

  if (isTD)
  {
    mesh->writeToExodus(currenttime);
  }
  else
  {
    mesh->writeToExodus(exodus_filename);
  }

  // Write the subgrid solutions if in multiscale mode
  if (write_subgrid_solution && multiscale_manager->getNumberSubgridModels() > 0)
  {
    multiscale_manager->writeSolution(currenttime, append);
  }

  debugger->print(1, "******** Finished PostprocessManager::writeSolution() ...");
}

// ========================================================================================
// ========================================================================================

template <class Node>
View_Sc2 PostprocessManager<Node>::getExtraCellFields(const int &block, CompressedView<View_Sc2> &wts)
{

  int numElem = wts.extent(0);
  View_Sc2 fields("grp field data", numElem, extracellfields_list[block].size());

  for (size_t fnum = 0; fnum < extracellfields_list[block].size(); ++fnum)
  {

    auto cfield = subview(fields, ALL(), fnum);
    View_Sc2 ecf = assembler->evaluateFunction(block, extracellfields_list[block][fnum], "ip");

    if (cellfield_reduction == "mean")
    { // default
      parallel_for("physics get extra grp fields", RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_CLASS_LAMBDA(const int e) {
        ScalarT grpmeas = 0.0;
        for (size_t pt=0; pt<wts.extent(1); pt++) {
          grpmeas += wts(e,pt);
        }
        for (size_t j=0; j<wts.extent(1); j++) {
          ScalarT val = ecf(e,j);
          cfield(e) += val*wts(e,j)/grpmeas;
        } });
    }
    else if (cellfield_reduction == "max")
    {
      parallel_for("physics get extra grp fields", RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_CLASS_LAMBDA(const int e) {
        for (size_t j=0; j<wts.extent(1); j++) {
          ScalarT val = ecf(e,j);
          if (val>cfield(e)) {
            cfield(e) = val;
          }
        } });
    }
    if (cellfield_reduction == "min")
    {
      parallel_for("physics get extra grp fields", RangePolicy<AssemblyExec>(0, wts.extent(0)), KOKKOS_CLASS_LAMBDA(const int e) {
        for (size_t j=0; j<wts.extent(1); j++) {
          ScalarT val = ecf(e,j);
          if (val<cfield(e)) {
            cfield(e) = val;
          }
        } });
    }
  }

  return fields;
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::writeOptimizationSolution(const int &numEvaluations)
{

  Teuchos::TimeMonitor localtimer(*writeSolutionTimer);

  typedef typename Node::device_type LA_device;
  typedef typename Node::execution_space LA_exec;

  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible)
  {
    data_avail = false;
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

  for (size_t block = 0; block < assembler->groups.size(); ++block)
  {
    std::string blockID = blocknames[block];
    auto myElements_tmp = disc->my_elements[block];
    vector<size_t> myElements(myElements_tmp.extent(0));
    for (size_t i = 0; i < myElements_tmp.extent(0); ++i)
    {
      myElements[i] = myElements_tmp(i);
    }

    if (myElements.size() > 0)
    {

      ////////////////////////////////////////////////////////////////
      // Discretized Parameters
      ////////////////////////////////////////////////////////////////

      vector<string> dpnames = params->discretized_param_names;
      vector<int> numParamBasis = params->paramNumBasis;
      vector<int> dp_usebasis = params->discretized_param_usebasis;
      vector<string> discParamTypes = params->discretized_param_basis_types;
      if (dpnames.size() > 0)
      {
        for (size_t n = 0; n < dpnames.size(); n++)
        {
          int bnum = dp_usebasis[n];
          if (discParamTypes[bnum] == "HGRAD")
          {
            Kokkos::View<ScalarT **, AssemblyDevice> soln_dev = Kokkos::View<ScalarT **, AssemblyDevice>("solution", myElements.size(), numNodesPerElem);
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              if (!assembler->groups[block][grp]->have_sols)
              {
                assembler->performGather(0, block, grp, params_kv[0], 4, 0);
              }
              auto sol = Kokkos::subview(assembler->groupData[block]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HGRAD", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
                for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                  soln_dev(eID(elem),i) = sol(elem,i);
                } });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->setOptimizationSolutionFieldData(dpnames[n], blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HVOL")
          {
            Kokkos::View<ScalarT *, AssemblyDevice> soln_dev("solution", myElements.size());
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            // std::string var = varlist[block][n];
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              if (!assembler->groups[block][grp]->have_sols)
              {
                assembler->performGather(0, block, grp, params_kv[0], 4, 0);
              }
              auto sol = Kokkos::subview(assembler->groupData[block]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HVOL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) { soln_dev(eID(elem)) = sol(elem, 0); });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->setOptimizationCellFieldData(dpnames[n], blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HDIV" || discParamTypes[n] == "HCURL")
          {
            // TMW: this is not actually implemented yet ... not hard to do though
            /*
             Kokkos::View<ScalarT*,HostDevice> soln_x("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_y("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_z("solution",myElements.size());
             std::string var = varlist[block][n];
             size_t eprog = 0;
             for( size_t e=0; e<assembler->groups[block].size(); e++ ) {
             Kokkos::View<ScalarT**,AssemblyDevice> sol = assembler->groups[block][grp]->param_avg;
             auto host_sol = Kokkos::create_mirror_view(sol);
             Kokkos::deep_copy(host_sol,sol);
             for (int p=0; p<assembler->groups[block][grp]->numElem; p++) {
             soln_x(eprog) = host_sol(p,n,0);
             soln_y(eprog) = host_sol(p,n,1);
             soln_z(eprog) = host_sol(p,n,2);
             eprog++;
             }
             }

             mesh->setcellFieldData(var+"x", blockID, myElements, soln_x);
             mesh->setcellFieldData(var+"y", blockID, myElements, soln_y);
             mesh->setcellFieldData(var+"z", blockID, myElements, soln_z);
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
  mesh->writeToOptimizationExodus(timestamp);
}

#if defined(MrHyDE_ENABLE_HDSA)
template <class Node>
void PostprocessManager<Node>::writeOptimizationSolution(const std::string &filename)
{

  Teuchos::TimeMonitor localtimer(*writeSolutionTimer);

  typedef typename Node::device_type LA_device;
  typedef typename Node::execution_space LA_exec;

  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible)
  {
    data_avail = false;
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

  for (size_t block = 0; block < assembler->groups.size(); ++block)
  {
    std::string blockID = blocknames[block];
    auto myElements_tmp = disc->my_elements[block];
    vector<size_t> myElements(myElements_tmp.extent(0));
    for (size_t i = 0; i < myElements_tmp.extent(0); ++i)
    {
      myElements[i] = myElements_tmp(i);
    }

    if (myElements.size() > 0)
    {

      ////////////////////////////////////////////////////////////////
      // Discretized Parameters
      ////////////////////////////////////////////////////////////////

      vector<string> dpnames = params->discretized_param_names;
      vector<int> numParamBasis = params->paramNumBasis;
      vector<int> dp_usebasis = params->discretized_param_usebasis;
      vector<string> discParamTypes = params->discretized_param_basis_types;
      if (dpnames.size() > 0)
      {
        for (size_t n = 0; n < dpnames.size(); n++)
        {
          int bnum = dp_usebasis[n];
          if (discParamTypes[bnum] == "HGRAD")
          {
            Kokkos::View<ScalarT **, AssemblyDevice> soln_dev = Kokkos::View<ScalarT **, AssemblyDevice>("solution", myElements.size(), numNodesPerElem);
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              if (!assembler->groups[block][grp]->have_sols)
              {
                assembler->performGather(0, block, grp, params_kv[0], 4, 0);
              }
              auto sol = Kokkos::subview(assembler->groupData[block]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HGRAD", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) {
                for( size_type i=0; i<soln_dev.extent(1); i++ ) {
                  soln_dev(eID(elem),i) = sol(elem,i);
                } });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->setOptimizationSolutionFieldData(dpnames[n], blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HVOL")
          {
            Kokkos::View<ScalarT *, AssemblyDevice> soln_dev("solution", myElements.size());
            auto soln_computed = Kokkos::create_mirror_view(soln_dev);
            // std::string var = varlist[block][n];
            for (size_t grp = 0; grp < assembler->groups[block].size(); ++grp)
            {
              auto eID = assembler->groups[block][grp]->localElemID;
              if (!assembler->groups[block][grp]->have_sols)
              {
                assembler->performGather(0, block, grp, params_kv[0], 4, 0);
              }
              auto sol = Kokkos::subview(assembler->groupData[block]->param, Kokkos::ALL(), n, Kokkos::ALL());
              parallel_for("postproc plot param HVOL", RangePolicy<AssemblyExec>(0, eID.extent(0)), KOKKOS_CLASS_LAMBDA(const int elem) { soln_dev(eID(elem)) = sol(elem, 0); });
            }
            Kokkos::deep_copy(soln_computed, soln_dev);
            mesh->setOptimizationCellFieldData(dpnames[n], blockID, myElements, soln_computed);
          }
          else if (discParamTypes[bnum] == "HDIV" || discParamTypes[n] == "HCURL")
          {
            // TMW: this is not actually implemented yet ... not hard to do though
            /*
             Kokkos::View<ScalarT*,HostDevice> soln_x("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_y("solution",myElements.size());
             Kokkos::View<ScalarT*,HostDevice> soln_z("solution",myElements.size());
             std::string var = varlist[block][n];
             size_t eprog = 0;
             for( size_t e=0; e<assembler->groups[block].size(); e++ ) {
             Kokkos::View<ScalarT**,AssemblyDevice> sol = assembler->groups[block][grp]->param_avg;
             auto host_sol = Kokkos::create_mirror_view(sol);
             Kokkos::deep_copy(host_sol,sol);
             for (int p=0; p<assembler->groups[block][grp]->numElem; p++) {
             soln_x(eprog) = host_sol(p,n,0);
             soln_y(eprog) = host_sol(p,n,1);
             soln_z(eprog) = host_sol(p,n,2);
             eprog++;
             }
             }

             mesh->setcellFieldData(var+"x", blockID, myElements, soln_x);
             mesh->setcellFieldData(var+"y", blockID, myElements, soln_y);
             mesh->setcellFieldData(var+"z", blockID, myElements, soln_z);
             */
          }
        }
      }
    }
  }

  ////////////////////////////////////////////////////////////////
  // Write to Exodus
  ////////////////////////////////////////////////////////////////

  mesh->writeToOptimizationExodus(filename);
}

#endif

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::setNewExodusFile(string &newfile)
{
  if (isTD && write_solution)
  {
    mesh->setupExodusFile(newfile);
  }
}
