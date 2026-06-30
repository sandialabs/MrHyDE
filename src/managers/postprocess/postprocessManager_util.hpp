/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

// ========================================================================================
// Called after each time step from the solver manager
// Checks if output is required at this time, and write to file or saves for output later
// ========================================================================================

//void PostprocessManager<Node>::record(vector<vector_RCP> &current_soln, const ScalarT &current_time,
//                                      const int &stepnum) {

template <class Node>
void PostprocessManager<Node>::record(vector<vector_RCP> &current_soln, const ScalarT &current_time,
                                      const int &stepnum, const ScalarT &deltat) { //EB
    
    // Determine if we want to collect QoI, objectives, etc.
    bool write_this_step = false;
    if (stepnum % write_frequency == 0) {
        write_this_step = true;
    }
    
    // Determine if we want to write to exodus on this time step
    bool write_exodus_this_step = false;
    if (stepnum % exodus_write_frequency == 0) {
        write_exodus_this_step = true;
    }
    
    if (nf2ff.save) {
        this->accumulateNF2FF(current_soln, current_time, deltat);
    }
    if (lumped_port_parameters.save) {
        this->accumulateLumpedPortParameters(current_soln, current_time, deltat);
    }

    // Write to exodus if requested and within user-defined time window for output
    if (write_exodus_this_step && current_time + 1.0e-100 >= exodus_record_start && current_time - 1.0e-100 <= exodus_record_stop) {
        if (write_solution) {
            this->writeSolution(current_soln, current_time);
        }
    }
    
    // Write all other output if requested and within user-defined time window for output
    if (write_this_step && current_time + 1.0e-100 >= record_start && current_time - 1.0e-100 <= record_stop) {
        
        if (compute_error) {
            this->computeError(current_soln, current_time);
        }
        if (compute_response || compute_objective) {
            this->computeObjective(current_soln, current_time);
        }
        if (compute_flux_response) {
            this->computeFluxResponse(current_soln, current_time);
        }
        if (compute_integrated_quantities) {
            this->computeIntegratedQuantities(current_soln, current_time);
        }
        if (compute_weighted_norm) {
            this->computeWeightedNorm(current_soln);
        }
        if (store_sensor_solution) {
            this->computeSensorSolution(current_soln, current_time, deltat);
        }
    }
    
    // We only store the full forward state if running optimization, or if user requested it
    if (save_solution) {
        for (size_t set = 0; set < soln.size(); ++set) {
            soln[set]->store(current_soln[set], current_time, 0);
        }
    }
    
    if (write_solution_to_file) {
        linalg->writeStateToFile(current_soln, solution_storage_file, stepnum);
    }
}

// ========================================================================================
// After simulation has completed, write to file or screen output all data saved
// ========================================================================================


// ========================================================================================
// Accumulate frequency-domain field data on the selected NF2FF surface and port volumes.
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::accumulateNF2FF(vector<vector_RCP> &current_soln,
                                                const ScalarT &current_time,
                                                const ScalarT &deltat)
{
  if (!nf2ff.save) {
    return;
  }

  const bool scattering_mode = (nf2ff.mode == "scattering");
  const bool use_manual_incident =
    scattering_mode && !nf2ff.manual_incident_sideset.empty() &&
    nf2ff.manual_incident_sideset == nf2ff.sideset;
  const bool use_automatic_incident =
    scattering_mode && nf2ff.automatic_planewave_index >= 0;

  typedef typename Node::execution_space LA_exec;
  typedef typename Node::device_type LA_device;

  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }

  vector<Kokkos::View<ScalarT *, AssemblyDevice> > sol_kv;
  for (size_t set = 0; set < current_soln.size(); ++set) {
    auto vec_kv = current_soln[set]->template getLocalView<LA_device>(
      Tpetra::Access::ReadWrite);
    auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
    if (data_avail) {
      sol_kv.push_back(vec_slice);
    }
    else {
      auto vec_dev = Kokkos::create_mirror(
        AssemblyDevice::memory_space(), vec_slice);
      Kokkos::deep_copy(vec_dev, vec_slice);
      sol_kv.push_back(vec_dev);
    }
  }

  auto firstValue = [](auto & value) {
    auto data = value.getData();
    auto host = create_mirror_view(data);
    deep_copy(host, data);
    return host(0, 0);
  };

  ScalarT local_constants[2] = {0.0, 0.0};
  int local_constant_count = 0;
  ScalarT local_source_values[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  int local_source_count = 0;

  for (size_t surface_index = 0;
       surface_index < nf2ff_surface_groups.size(); ++surface_index) {
    const size_t block = nf2ff_surface_groups[surface_index].block;
    const size_t group = nf2ff_surface_groups[surface_index].group;
    auto boundary_group = assembler->boundary_groups[block][group];

    assembler->wkset[block]->isOnSide = true;
    assembler->wkset[block]->time = current_time;

    for (size_t set = 0; set < sol_kv.size(); ++set) {
      assembler->performBoundaryGather(set, block, group, sol_kv[set], 0, 0);
    }
    assembler->updateWorksetBoundary(block, group, 0, 0, true);

    auto Ex = assembler->wkset[block]->getSolutionField("E[x]");
    auto Ey = assembler->wkset[block]->getSolutionField("E[y]");
    auto Ez = assembler->wkset[block]->getSolutionField("E[z]");

    auto dft = nf2ff_surface_groups[surface_index].electric_E_dft;
    auto frequencies = nf2ff.frequency_device;
    const ScalarT time = current_time;
    const ScalarT dt = deltat;

    if (scattering_mode) {
      Vista<ScalarT> incidentEx;
      Vista<ScalarT> incidentEy;
      Vista<ScalarT> incidentEz;
      Vista<ScalarT> automaticIncidentEx;
      Vista<ScalarT> automaticIncidentEy;
      Vista<ScalarT> automaticIncidentEz;

      if (use_manual_incident) {
        incidentEx =
          assembler->function_managers[block]->evaluate(
            "incident Ex", "side ip");
        incidentEy =
          assembler->function_managers[block]->evaluate(
            "incident Ey", "side ip");
        incidentEz =
          assembler->function_managers[block]->evaluate(
            "incident Ez", "side ip");
      }

      if (use_automatic_incident) {
        const string prefix = "automatic_planewave_" +
          std::to_string(nf2ff.automatic_planewave_index) + "_";
        automaticIncidentEx =
          assembler->function_managers[block]->evaluate(
            prefix + "Ex", "side ip");
        automaticIncidentEy =
          assembler->function_managers[block]->evaluate(
            prefix + "Ey", "side ip");
        automaticIncidentEz =
          assembler->function_managers[block]->evaluate(
            prefix + "Ez", "side ip");
      }

      parallel_for("PostprocessManager NF2FF scattering DFT",
                   RangePolicy<AssemblyExec>(0, boundary_group->numElem),
                   MRHYDE_LAMBDA(const int elem) {
        for (size_type pt = 0; pt < dft.extent(2); ++pt) {
          ScalarT incident_x = 0.0;
          ScalarT incident_y = 0.0;
          ScalarT incident_z = 0.0;
          if (use_manual_incident) {
            incident_x += incidentEx(elem, pt);
            incident_y += incidentEy(elem, pt);
            incident_z += incidentEz(elem, pt);
          }
          if (use_automatic_incident) {
            incident_x += automaticIncidentEx(elem, pt);
            incident_y += automaticIncidentEy(elem, pt);
            incident_z += automaticIncidentEz(elem, pt);
          }

          const ScalarT field_x = Ex(elem, pt) - incident_x;
          const ScalarT field_y = Ey(elem, pt) - incident_y;
          const ScalarT field_z = Ez(elem, pt) - incident_z;

          for (size_type freq = 0; freq < dft.extent(0); ++freq) {
            const ScalarT omega_t = 2.0*PI*frequencies[freq]*time;
            const ScalarT real_scale = dt*cos(omega_t);
            const ScalarT imag_scale = -dt*sin(omega_t);

            dft(freq, elem, pt, 0, 0) += real_scale*field_x;
            dft(freq, elem, pt, 1, 0) += real_scale*field_y;
            dft(freq, elem, pt, 2, 0) += real_scale*field_z;
            dft(freq, elem, pt, 0, 1) += imag_scale*field_x;
            dft(freq, elem, pt, 1, 1) += imag_scale*field_y;
            dft(freq, elem, pt, 2, 1) += imag_scale*field_z;
          }
        }
      });
    }
    else {
      parallel_for("PostprocessManager NF2FF radiation DFT",
                   RangePolicy<AssemblyExec>(0, boundary_group->numElem),
                   MRHYDE_LAMBDA(const int elem) {
        for (size_type pt = 0; pt < dft.extent(2); ++pt) {
          const ScalarT field_x = Ex(elem, pt);
          const ScalarT field_y = Ey(elem, pt);
          const ScalarT field_z = Ez(elem, pt);

          for (size_type freq = 0; freq < dft.extent(0); ++freq) {
            const ScalarT omega_t = 2.0*PI*frequencies[freq]*time;
            const ScalarT real_scale = dt*cos(omega_t);
            const ScalarT imag_scale = -dt*sin(omega_t);

            dft(freq, elem, pt, 0, 0) += real_scale*field_x;
            dft(freq, elem, pt, 1, 0) += real_scale*field_y;
            dft(freq, elem, pt, 2, 0) += real_scale*field_z;
            dft(freq, elem, pt, 0, 1) += imag_scale*field_x;
            dft(freq, elem, pt, 1, 1) += imag_scale*field_y;
            dft(freq, elem, pt, 2, 1) += imag_scale*field_z;
          }
        }
      });
    }

    if (local_constant_count == 0 && boundary_group->numElem > 0) {
      auto c0 = assembler->function_managers[block]->evaluate("c0", "side ip");
      auto eta0 = assembler->function_managers[block]->evaluate("eta0", "side ip");
      local_constants[0] = firstValue(c0);
      local_constants[1] = firstValue(eta0);
      local_constant_count = 1;
    }

    if (scattering_mode && local_source_count == 0 &&
        boundary_group->numElem > 0) {
      string prefix;
      if (use_automatic_incident) {
        prefix = "automatic_planewave_" +
          std::to_string(nf2ff.automatic_planewave_index) + "_";
      }

      const string waveform_te_name = use_automatic_incident ?
        prefix + "source_waveform_te" : "source_waveform_te";
      const string waveform_tm_name = use_automatic_incident ?
        prefix + "source_waveform_tm" : "source_waveform_tm";
      const string amplitude_name = use_automatic_incident ?
        prefix + "source_amplitude" : "source_amplitude";
      const string te_name = use_automatic_incident ?
        prefix + "source_te" : "source_te";
      const string tm_name = use_automatic_incident ?
        prefix + "source_tm" : "source_tm";

      auto source_waveform_te =
        assembler->function_managers[block]->evaluate(waveform_te_name,
                                                       "side ip");
      auto source_waveform_tm =
        assembler->function_managers[block]->evaluate(waveform_tm_name,
                                                       "side ip");
      auto source_amplitude =
        assembler->function_managers[block]->evaluate(amplitude_name,
                                                       "side ip");
      auto source_te =
        assembler->function_managers[block]->evaluate(te_name, "side ip");
      auto source_tm =
        assembler->function_managers[block]->evaluate(tm_name, "side ip");

      local_source_values[0] = firstValue(source_waveform_te);
      local_source_values[1] = firstValue(source_waveform_tm);
      local_source_values[2] = firstValue(source_amplitude);
      local_source_values[3] = firstValue(source_te);
      local_source_values[4] = firstValue(source_tm);
      local_source_count = 1;
    }
  }

  for (size_t port_group_index = 0;
       port_group_index < nf2ff_port_groups.size(); ++port_group_index) {
    const size_t block = nf2ff_port_groups[port_group_index].block;
    const size_t group = nf2ff_port_groups[port_group_index].group;
    auto element_group = assembler->groups[block][group];

    assembler->wkset[block]->isOnSide = false;
    assembler->wkset[block]->time = current_time;

    for (size_t set = 0; set < sol_kv.size(); ++set) {
      assembler->performGather(set, block, group, sol_kv[set], 0, 0);
    }
    assembler->updateWorkset(block, group, 0, 0, true);

    auto Ex = assembler->wkset[block]->getSolutionField("E[x]");
    auto Ey = assembler->wkset[block]->getSolutionField("E[y]");
    auto Ez = assembler->wkset[block]->getSolutionField("E[z]");

    auto dft = nf2ff_port_groups[port_group_index].electric_E_dft;
    auto frequencies = nf2ff.frequency_device;
    const ScalarT time = current_time;
    const ScalarT dt = deltat;

    parallel_for("PostprocessManager NF2FF port DFT",
                 RangePolicy<AssemblyExec>(0, element_group->numElem),
                 MRHYDE_LAMBDA(const int elem) {
      for (size_type pt = 0; pt < dft.extent(2); ++pt) {
        const ScalarT field_x = Ex(elem, pt);
        const ScalarT field_y = Ey(elem, pt);
        const ScalarT field_z = Ez(elem, pt);

        for (size_type freq = 0; freq < dft.extent(0); ++freq) {
          const ScalarT omega_t = 2.0*PI*frequencies[freq]*time;
          const ScalarT real_scale = dt*cos(omega_t);
          const ScalarT imag_scale = -dt*sin(omega_t);

          dft(freq, elem, pt, 0, 0) += real_scale*field_x;
          dft(freq, elem, pt, 1, 0) += real_scale*field_y;
          dft(freq, elem, pt, 2, 0) += real_scale*field_z;
          dft(freq, elem, pt, 0, 1) += imag_scale*field_x;
          dft(freq, elem, pt, 1, 1) += imag_scale*field_y;
          dft(freq, elem, pt, 2, 1) += imag_scale*field_z;
        }
      }
    });
  }

  for (size_t surface_index = 0;
       surface_index < nf2ff_surface_groups.size(); ++surface_index) {
    assembler->wkset[nf2ff_surface_groups[surface_index].block]->isOnSide = false;
  }

  ScalarT global_constants[2] = {0.0, 0.0};
  int global_constant_count = 0;
  Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 2,
                     local_constants, global_constants);
  Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1,
                     &local_constant_count, &global_constant_count);

  if (global_constant_count > 0 && !nf2ff.constants_initialized) {
    const ScalarT inverse_count =
      1.0/static_cast<ScalarT>(global_constant_count);
    nf2ff.c0 = global_constants[0]*inverse_count;
    nf2ff.eta0 = global_constants[1]*inverse_count;
    TEUCHOS_TEST_FOR_EXCEPTION(nf2ff.c0 <= 0.0 || nf2ff.eta0 <= 0.0,
                               std::runtime_error,
                               "NF2FF requires positive c0 and eta0 on the selected sideset.");
    nf2ff.constants_initialized = true;
  }

  if (scattering_mode) {
    ScalarT global_source_values[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    int global_source_count = 0;
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 5,
                       local_source_values, global_source_values);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1,
                       &local_source_count, &global_source_count);

    if (global_source_count > 0) {
      const ScalarT inverse_count =
        1.0/static_cast<ScalarT>(global_source_count);
      const ScalarT source_waveform_te =
        global_source_values[0]*inverse_count;
      const ScalarT source_waveform_tm =
        global_source_values[1]*inverse_count;

      if (!nf2ff.scattering_source_initialized) {
        nf2ff.source_amplitude = global_source_values[2]*inverse_count;
        nf2ff.source_te = global_source_values[3]*inverse_count;
        nf2ff.source_tm = global_source_values[4]*inverse_count;

        TEUCHOS_TEST_FOR_EXCEPTION(
          std::abs(nf2ff.source_te) <= 1.0e-30 &&
          std::abs(nf2ff.source_tm) <= 1.0e-30, std::runtime_error,
          "NF2FF scattering mode requires a nonzero source_te or source_tm coefficient.");
        TEUCHOS_TEST_FOR_EXCEPTION(
          std::abs(nf2ff.source_amplitude) <= 1.0e-30,
          std::runtime_error,
          "NF2FF scattering mode requires a nonzero source_amplitude.");

        nf2ff.scattering_source_initialized = true;
      }

      for (size_t freq = 0; freq < nf2ff.frequencies.size(); ++freq) {
        const ScalarT omega_t =
          2.0*PI*nf2ff.frequencies[freq]*current_time;
        const std::complex<ScalarT> kernel(
          deltat*cos(omega_t), -deltat*sin(omega_t));
        nf2ff.source_te_dft[freq] += source_waveform_te*kernel;
        nf2ff.source_tm_dft[freq] += source_waveform_tm*kernel;
      }
    }
  }
  else {
    for (size_t port_index = 0; port_index < nf2ff_ports.size();
         ++port_index) {
      const NF2FFPort & port = nf2ff_ports[port_index];
      const ScalarT u = current_time - port.offset;
      const ScalarT a = u/port.tau;
      const ScalarT envelope = exp(-a*a);
      ScalarT waveform = 0.0;

      if (port.source_type == "gaussian") {
        waveform = envelope;
      }
      else if (port.source_type == "gaussian derivative") {
        waveform = -2.0*a*envelope;
      }
      else if (port.source_type == "gaussian sinusoidal") {
        waveform = envelope*cos(2.0*PI*port.carrier_frequency*u);
      }
      else if (port.source_type == "sinusoidal") {
        const ScalarT ramp = (u < 0.0) ? envelope : 1.0;
        waveform = ramp*cos(2.0*PI*port.carrier_frequency*u);
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                   "Unknown lumped-port source type.");
      }

      for (size_t freq = 0; freq < nf2ff.frequencies.size(); ++freq) {
        const ScalarT omega_t =
          2.0*PI*nf2ff.frequencies[freq]*current_time;
        nf2ff_ports[port_index].source_dft[freq] +=
          std::complex<ScalarT>(deltat*cos(omega_t),
                                -deltat*sin(omega_t))*waveform;
      }
    }
  }
}



// ========================================================================================
// Accumulate frequency-domain field data used by lumped-port parameters.
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::accumulateLumpedPortParameters(
  vector<vector_RCP> &current_soln, const ScalarT &current_time,
  const ScalarT &deltat)
{
  if (!lumped_port_parameters.save) {
    return;
  }

  typedef typename Node::execution_space LA_exec;
  typedef typename Node::device_type LA_device;

  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec,
      AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }

  vector<Kokkos::View<ScalarT *, AssemblyDevice> > sol_kv;
  for (size_t set = 0; set < current_soln.size(); ++set) {
    auto vec_kv = current_soln[set]->template getLocalView<LA_device>(
      Tpetra::Access::ReadWrite);
    auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
    if (data_avail) {
      sol_kv.push_back(vec_slice);
    }
    else {
      auto vec_dev = Kokkos::create_mirror(
        AssemblyDevice::memory_space(), vec_slice);
      Kokkos::deep_copy(vec_dev, vec_slice);
      sol_kv.push_back(vec_dev);
    }
  }

  auto firstValue = [](auto & value) {
    auto data = value.getData();
    auto host = create_mirror_view(data);
    deep_copy(host, data);
    return host(0, 0);
  };

  ScalarT local_radiation_constants[2] = {0.0, 0.0};
  int local_radiation_constant_count = 0;

  for (size_t surface_index = 0;
       surface_index < lumped_port_parameter_surface_groups.size();
       ++surface_index) {
    const size_t block =
      lumped_port_parameter_surface_groups[surface_index].block;
    const size_t group =
      lumped_port_parameter_surface_groups[surface_index].group;
    auto boundary_group = assembler->boundary_groups[block][group];

    assembler->wkset[block]->isOnSide = true;
    assembler->wkset[block]->time = current_time;

    for (size_t set = 0; set < sol_kv.size(); ++set) {
      assembler->performBoundaryGather(set, block, group, sol_kv[set], 0, 0);
    }
    assembler->updateWorksetBoundary(block, group, 0, 0, true);

    auto Ex = assembler->wkset[block]->getSolutionField("E[x]");
    auto Ey = assembler->wkset[block]->getSolutionField("E[y]");
    auto Ez = assembler->wkset[block]->getSolutionField("E[z]");

    auto dft =
      lumped_port_parameter_surface_groups[surface_index].electric_E_dft;
    auto frequencies = lumped_port_parameters.frequency_device;
    const ScalarT time = current_time;
    const ScalarT dt = deltat;

    parallel_for("PostprocessManager lumped port surface DFT",
                 RangePolicy<AssemblyExec>(0, boundary_group->numElem),
                 MRHYDE_LAMBDA(const int elem) {
      for (size_type pt = 0; pt < dft.extent(2); ++pt) {
        const ScalarT field_x = Ex(elem, pt);
        const ScalarT field_y = Ey(elem, pt);
        const ScalarT field_z = Ez(elem, pt);

        for (size_type freq = 0; freq < dft.extent(0); ++freq) {
          const ScalarT omega_t = 2.0*PI*frequencies[freq]*time;
          const ScalarT real_scale = dt*cos(omega_t);
          const ScalarT imag_scale = -dt*sin(omega_t);

          dft(freq, elem, pt, 0, 0) += real_scale*field_x;
          dft(freq, elem, pt, 1, 0) += real_scale*field_y;
          dft(freq, elem, pt, 2, 0) += real_scale*field_z;
          dft(freq, elem, pt, 0, 1) += imag_scale*field_x;
          dft(freq, elem, pt, 1, 1) += imag_scale*field_y;
          dft(freq, elem, pt, 2, 1) += imag_scale*field_z;
        }
      }
    });

    if (!lumped_port_parameters.radiation_constants_initialized &&
        local_radiation_constant_count == 0 &&
        boundary_group->numElem > 0) {
      auto c0 = assembler->function_managers[block]->evaluate(
        "c0", "side ip");
      auto eta0 = assembler->function_managers[block]->evaluate(
        "eta0", "side ip");
      local_radiation_constants[0] = firstValue(c0);
      local_radiation_constants[1] = firstValue(eta0);
      local_radiation_constant_count = 1;
    }
  }

  if (lumped_port_parameters.has_radiation_surface &&
      !lumped_port_parameters.radiation_constants_initialized) {
    ScalarT global_radiation_constants[2] = {0.0, 0.0};
    int global_radiation_constant_count = 0;
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 2,
                       local_radiation_constants,
                       global_radiation_constants);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1,
                       &local_radiation_constant_count,
                       &global_radiation_constant_count);

    if (global_radiation_constant_count > 0) {
      const ScalarT inverse_count = 1.0/
        static_cast<ScalarT>(global_radiation_constant_count);
      lumped_port_parameters.radiation_c0 =
        global_radiation_constants[0]*inverse_count;
      lumped_port_parameters.radiation_eta0 =
        global_radiation_constants[1]*inverse_count;
      TEUCHOS_TEST_FOR_EXCEPTION(
        lumped_port_parameters.radiation_c0 <= 0.0 ||
        lumped_port_parameters.radiation_eta0 <= 0.0,
        std::runtime_error,
        "Lumped port radiation sideset requires positive c0 and eta0.");
      lumped_port_parameters.radiation_constants_initialized = true;
    }
  }

  for (size_t port_group_index = 0;
       port_group_index < lumped_port_parameter_port_groups.size();
       ++port_group_index) {
    const size_t block =
      lumped_port_parameter_port_groups[port_group_index].block;
    const size_t group =
      lumped_port_parameter_port_groups[port_group_index].group;
    auto element_group = assembler->groups[block][group];

    assembler->wkset[block]->isOnSide = false;
    assembler->wkset[block]->time = current_time;

    for (size_t set = 0; set < sol_kv.size(); ++set) {
      assembler->performGather(set, block, group, sol_kv[set], 0, 0);
    }
    assembler->updateWorkset(block, group, 0, 0, true);

    auto Ex = assembler->wkset[block]->getSolutionField("E[x]");
    auto Ey = assembler->wkset[block]->getSolutionField("E[y]");
    auto Ez = assembler->wkset[block]->getSolutionField("E[z]");

    auto dft =
      lumped_port_parameter_port_groups[port_group_index].electric_E_dft;
    auto frequencies = lumped_port_parameters.frequency_device;
    const ScalarT time = current_time;
    const ScalarT dt = deltat;

    parallel_for("PostprocessManager lumped port electric-field DFT",
                 RangePolicy<AssemblyExec>(0, element_group->numElem),
                 MRHYDE_LAMBDA(const int elem) {
      for (size_type pt = 0; pt < dft.extent(2); ++pt) {
        const ScalarT field_x = Ex(elem, pt);
        const ScalarT field_y = Ey(elem, pt);
        const ScalarT field_z = Ez(elem, pt);

        for (size_type freq = 0; freq < dft.extent(0); ++freq) {
          const ScalarT omega_t = 2.0*PI*frequencies[freq]*time;
          const ScalarT real_scale = dt*cos(omega_t);
          const ScalarT imag_scale = -dt*sin(omega_t);

          dft(freq, elem, pt, 0, 0) += real_scale*field_x;
          dft(freq, elem, pt, 1, 0) += real_scale*field_y;
          dft(freq, elem, pt, 2, 0) += real_scale*field_z;
          dft(freq, elem, pt, 0, 1) += imag_scale*field_x;
          dft(freq, elem, pt, 1, 1) += imag_scale*field_y;
          dft(freq, elem, pt, 2, 1) += imag_scale*field_z;
        }
      }
    });
  }

  for (size_t volume_index = 0;
       volume_index < lumped_port_parameter_volume_groups.size();
       ++volume_index) {
    auto & volume_group = lumped_port_parameter_volume_groups[volume_index];
    const size_t block = volume_group.block;
    const size_t group = volume_group.group;
    auto element_group = assembler->groups[block][group];

    assembler->wkset[block]->isOnSide = false;
    assembler->wkset[block]->time = current_time;

    for (size_t set = 0; set < sol_kv.size(); ++set) {
      assembler->performGather(set, block, group, sol_kv[set], 0, 0);
    }
    assembler->updateWorkset(block, group, 0, 0, true);

    auto Ex = assembler->wkset[block]->getSolutionField("E[x]");
    auto Ey = assembler->wkset[block]->getSolutionField("E[y]");
    auto Ez = assembler->wkset[block]->getSolutionField("E[z]");
    auto Hx = assembler->wkset[block]->getSolutionField("H[x]");
    auto Hy = assembler->wkset[block]->getSolutionField("H[y]");
    auto Hz = assembler->wkset[block]->getSolutionField("H[z]");

    if (!volume_group.material_energy_initialized) {
      auto c0 = assembler->function_managers[block]->evaluate("c0", "ip");
      auto eta0 = assembler->function_managers[block]->evaluate("eta0", "ip");

      auto epsr_xx = assembler->function_managers[block]->evaluate("epsr_xx", "ip");
      auto epsr_xy = assembler->function_managers[block]->evaluate("epsr_xy", "ip");
      auto epsr_xz = assembler->function_managers[block]->evaluate("epsr_xz", "ip");
      auto epsr_yx = assembler->function_managers[block]->evaluate("epsr_yx", "ip");
      auto epsr_yy = assembler->function_managers[block]->evaluate("epsr_yy", "ip");
      auto epsr_yz = assembler->function_managers[block]->evaluate("epsr_yz", "ip");
      auto epsr_zx = assembler->function_managers[block]->evaluate("epsr_zx", "ip");
      auto epsr_zy = assembler->function_managers[block]->evaluate("epsr_zy", "ip");
      auto epsr_zz = assembler->function_managers[block]->evaluate("epsr_zz", "ip");

      auto mur_xx = assembler->function_managers[block]->evaluate("mur_xx", "ip");
      auto mur_xy = assembler->function_managers[block]->evaluate("mur_xy", "ip");
      auto mur_xz = assembler->function_managers[block]->evaluate("mur_xz", "ip");
      auto mur_yx = assembler->function_managers[block]->evaluate("mur_yx", "ip");
      auto mur_yy = assembler->function_managers[block]->evaluate("mur_yy", "ip");
      auto mur_yz = assembler->function_managers[block]->evaluate("mur_yz", "ip");
      auto mur_zx = assembler->function_managers[block]->evaluate("mur_zx", "ip");
      auto mur_zy = assembler->function_managers[block]->evaluate("mur_zy", "ip");
      auto mur_zz = assembler->function_managers[block]->evaluate("mur_zz", "ip");

      auto xir_xx = assembler->function_managers[block]->evaluate("xir_xx", "ip");
      auto xir_xy = assembler->function_managers[block]->evaluate("xir_xy", "ip");
      auto xir_xz = assembler->function_managers[block]->evaluate("xir_xz", "ip");
      auto xir_yx = assembler->function_managers[block]->evaluate("xir_yx", "ip");
      auto xir_yy = assembler->function_managers[block]->evaluate("xir_yy", "ip");
      auto xir_yz = assembler->function_managers[block]->evaluate("xir_yz", "ip");
      auto xir_zx = assembler->function_managers[block]->evaluate("xir_zx", "ip");
      auto xir_zy = assembler->function_managers[block]->evaluate("xir_zy", "ip");
      auto xir_zz = assembler->function_managers[block]->evaluate("xir_zz", "ip");

      auto zetar_xx = assembler->function_managers[block]->evaluate("zetar_xx", "ip");
      auto zetar_xy = assembler->function_managers[block]->evaluate("zetar_xy", "ip");
      auto zetar_xz = assembler->function_managers[block]->evaluate("zetar_xz", "ip");
      auto zetar_yx = assembler->function_managers[block]->evaluate("zetar_yx", "ip");
      auto zetar_yy = assembler->function_managers[block]->evaluate("zetar_yy", "ip");
      auto zetar_yz = assembler->function_managers[block]->evaluate("zetar_yz", "ip");
      auto zetar_zx = assembler->function_managers[block]->evaluate("zetar_zx", "ip");
      auto zetar_zy = assembler->function_managers[block]->evaluate("zetar_zy", "ip");
      auto zetar_zz = assembler->function_managers[block]->evaluate("zetar_zz", "ip");

      auto coefficients = volume_group.energy_coefficients;
      parallel_for("PostprocessManager lumped port energy coefficients",
                   RangePolicy<AssemblyExec>(0, element_group->numElem),
                   MRHYDE_LAMBDA(const int elem) {
        for (size_type pt = 0; pt < coefficients.extent(1); ++pt) {
          const ScalarT c = c0(elem, pt);
          const ScalarT eta = eta0(elem, pt);
          const ScalarT scale = (c > 0.0 && eta > 0.0) ?
            1.0/(eta*c) : 0.0;

          coefficients(elem, pt, 0) = scale*epsr_xx(elem, pt);
          coefficients(elem, pt, 1) = 0.5*scale*(epsr_xy(elem, pt) + epsr_yx(elem, pt));
          coefficients(elem, pt, 2) = 0.5*scale*(epsr_xz(elem, pt) + epsr_zx(elem, pt));
          coefficients(elem, pt, 3) = scale*epsr_yy(elem, pt);
          coefficients(elem, pt, 4) = 0.5*scale*(epsr_yz(elem, pt) + epsr_zy(elem, pt));
          coefficients(elem, pt, 5) = scale*epsr_zz(elem, pt);

          coefficients(elem, pt, 6) = 0.5*scale*(xir_xx(elem, pt) + zetar_xx(elem, pt));
          coefficients(elem, pt, 7) = 0.5*scale*(xir_xy(elem, pt) + zetar_yx(elem, pt));
          coefficients(elem, pt, 8) = 0.5*scale*(xir_xz(elem, pt) + zetar_zx(elem, pt));
          coefficients(elem, pt, 9) = 0.5*scale*(xir_yx(elem, pt) + zetar_xy(elem, pt));
          coefficients(elem, pt, 10) = 0.5*scale*(xir_yy(elem, pt) + zetar_yy(elem, pt));
          coefficients(elem, pt, 11) = 0.5*scale*(xir_yz(elem, pt) + zetar_zy(elem, pt));
          coefficients(elem, pt, 12) = 0.5*scale*(xir_zx(elem, pt) + zetar_xz(elem, pt));
          coefficients(elem, pt, 13) = 0.5*scale*(xir_zy(elem, pt) + zetar_yz(elem, pt));
          coefficients(elem, pt, 14) = 0.5*scale*(xir_zz(elem, pt) + zetar_zz(elem, pt));

          coefficients(elem, pt, 15) = scale*mur_xx(elem, pt);
          coefficients(elem, pt, 16) = 0.5*scale*(mur_xy(elem, pt) + mur_yx(elem, pt));
          coefficients(elem, pt, 17) = 0.5*scale*(mur_xz(elem, pt) + mur_zx(elem, pt));
          coefficients(elem, pt, 18) = scale*mur_yy(elem, pt);
          coefficients(elem, pt, 19) = 0.5*scale*(mur_yz(elem, pt) + mur_zy(elem, pt));
          coefficients(elem, pt, 20) = scale*mur_zz(elem, pt);

          const ScalarT free_space_error =
            fabs(epsr_xx(elem, pt) - 1.0) + fabs(epsr_xy(elem, pt)) +
            fabs(epsr_xz(elem, pt)) + fabs(epsr_yx(elem, pt)) +
            fabs(epsr_yy(elem, pt) - 1.0) + fabs(epsr_yz(elem, pt)) +
            fabs(epsr_zx(elem, pt)) + fabs(epsr_zy(elem, pt)) +
            fabs(epsr_zz(elem, pt) - 1.0) +
            fabs(mur_xx(elem, pt) - 1.0) + fabs(mur_xy(elem, pt)) +
            fabs(mur_xz(elem, pt)) + fabs(mur_yx(elem, pt)) +
            fabs(mur_yy(elem, pt) - 1.0) + fabs(mur_yz(elem, pt)) +
            fabs(mur_zx(elem, pt)) + fabs(mur_zy(elem, pt)) +
            fabs(mur_zz(elem, pt) - 1.0) +
            fabs(xir_xx(elem, pt)) + fabs(xir_xy(elem, pt)) +
            fabs(xir_xz(elem, pt)) + fabs(xir_yx(elem, pt)) +
            fabs(xir_yy(elem, pt)) + fabs(xir_yz(elem, pt)) +
            fabs(xir_zx(elem, pt)) + fabs(xir_zy(elem, pt)) +
            fabs(xir_zz(elem, pt)) +
            fabs(zetar_xx(elem, pt)) + fabs(zetar_xy(elem, pt)) +
            fabs(zetar_xz(elem, pt)) + fabs(zetar_yx(elem, pt)) +
            fabs(zetar_yy(elem, pt)) + fabs(zetar_yz(elem, pt)) +
            fabs(zetar_zx(elem, pt)) + fabs(zetar_zy(elem, pt)) +
            fabs(zetar_zz(elem, pt));
          coefficients(elem, pt, 21) =
            (free_space_error <= 1.0e-12) ? scale : 0.0;
        }
      });
      volume_group.material_energy_initialized = true;
    }

    auto dft = volume_group.field_dft;
    auto frequencies = lumped_port_parameters.frequency_device;
    const ScalarT time = current_time;
    const ScalarT dt = deltat;

    parallel_for("PostprocessManager lumped port volume-field DFT",
                 RangePolicy<AssemblyExec>(0, element_group->numElem),
                 MRHYDE_LAMBDA(const int elem) {
      for (size_type pt = 0; pt < dft.extent(2); ++pt) {
        const ScalarT fields[6] = {
          Ex(elem, pt), Ey(elem, pt), Ez(elem, pt),
          Hx(elem, pt), Hy(elem, pt), Hz(elem, pt)};

        for (size_type freq = 0; freq < dft.extent(0); ++freq) {
          const ScalarT omega_t = 2.0*PI*frequencies[freq]*time;
          const ScalarT real_scale = dt*cos(omega_t);
          const ScalarT imag_scale = -dt*sin(omega_t);
          for (int component = 0; component < 6; ++component) {
            dft(freq, elem, pt, component, 0) +=
              real_scale*fields[component];
            dft(freq, elem, pt, component, 1) +=
              imag_scale*fields[component];
          }
        }
      }
    });
  }

  for (size_t surface_index = 0;
       surface_index < lumped_port_parameter_surface_groups.size();
       ++surface_index) {
    assembler->wkset[
      lumped_port_parameter_surface_groups[surface_index].block]->isOnSide =
      false;
  }

  for (size_t port_index = 0;
       port_index < lumped_port_parameter_ports.size(); ++port_index) {
    const NF2FFPort & port = lumped_port_parameter_ports[port_index];
    const ScalarT u = current_time - port.offset;
    const ScalarT a = u/port.tau;
    const ScalarT envelope = exp(-a*a);
    ScalarT waveform = 0.0;

    if (port.source_type == "gaussian") {
      waveform = envelope;
    }
    else if (port.source_type == "gaussian derivative") {
      waveform = -2.0*a*envelope;
    }
    else if (port.source_type == "gaussian sinusoidal") {
      waveform = envelope*cos(2.0*PI*port.carrier_frequency*u);
    }
    else if (port.source_type == "sinusoidal") {
      const ScalarT ramp = (u < 0.0) ? envelope : 1.0;
      waveform = ramp*cos(2.0*PI*port.carrier_frequency*u);
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                 "Unknown lumped-port source type.");
    }

    const ScalarT source_value = port.amplitude*waveform;
    for (size_t freq = 0;
         freq < lumped_port_parameters.frequencies.size(); ++freq) {
      const ScalarT omega_t =
        2.0*PI*lumped_port_parameters.frequencies[freq]*current_time;
      lumped_port_parameter_ports[port_index].source_dft[freq] +=
        std::complex<ScalarT>(deltat*cos(omega_t),
                              -deltat*sin(omega_t))*source_value;
    }
  }
}


// ========================================================================================
// Write NF2FF data in CSV format.
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::writeNF2FF()
{
  if (!nf2ff.save) {
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(!nf2ff.constants_initialized,
                             std::runtime_error,
                             "NF2FF did not accumulate surface-field constants.");

  const bool scattering_mode = (nf2ff.mode == "scattering");
  const bool radiation_mode = (nf2ff.mode == "radiation");
  TEUCHOS_TEST_FOR_EXCEPTION(!scattering_mode && !radiation_mode,
                             std::runtime_error,
                             "NF2FF mode must be scattering or radiation.");

  vector<ScalarT> theta_deg(nf2ff.ntheta);
  vector<ScalarT> phi_deg(nf2ff.nphi);
  for (int i = 0; i < nf2ff.ntheta; ++i) {
    theta_deg[i] = (nf2ff.ntheta == 1) ? nf2ff.min_theta :
      nf2ff.min_theta + (nf2ff.max_theta - nf2ff.min_theta) *
      static_cast<ScalarT>(i)/static_cast<ScalarT>(nf2ff.ntheta - 1);
  }
  for (int i = 0; i < nf2ff.nphi; ++i) {
    phi_deg[i] = (nf2ff.nphi == 1) ? nf2ff.min_phi :
      nf2ff.min_phi + (nf2ff.max_phi - nf2ff.min_phi) *
      static_cast<ScalarT>(i)/static_cast<ScalarT>(nf2ff.nphi - 1);
  }

  std::ofstream csv;
  if (Comm->getRank() == 0) {
    std::filesystem::path filename(nf2ff.output_file);
    if (filename.extension() != ".csv") {
      filename += ".csv";
    }

    std::error_code error;
    if (!filename.parent_path().empty()) {
      std::filesystem::create_directories(filename.parent_path(), error);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<bool>(error), std::runtime_error,
                               "Could not create NF2FF output directory for '"
                               << filename.string() << "': " << error.message());
    csv.open(filename.string());
    TEUCHOS_TEST_FOR_EXCEPTION(!csv, std::runtime_error,
                               "Could not open NF2FF output file '"
                               << filename.string() << "'.");

    csv << std::setprecision(17);
    csv << "frequency,theta_deg,phi_deg"
        << ",A_theta_real,A_theta_imag,A_phi_real,A_phi_imag"
        << ",F_theta_real,F_theta_imag,F_phi_real,F_phi_imag"
        << ",E_theta_real,E_theta_imag,E_phi_real,E_phi_imag"
        << ",H_theta_real,H_theta_imag,H_phi_real,H_phi_imag"
        << ",P_theta,P_phi,P_total";

    if (scattering_mode) {
      csv << ",RCS_theta,RCS_phi,RCS_total";
    }
    else {
      csv << ",Directivity_theta,Directivity_phi,Directivity_total"
          << ",Gain_theta,Gain_phi,Gain_total"
          << ",Realized_gain_theta,Realized_gain_phi,Realized_gain_total"
          << ",Radiated_power,Accepted_power";
      for (size_t port_index = 0; port_index < nf2ff_ports.size();
           ++port_index) {
        const string label = "port" + std::to_string(port_index);
        csv << ',' << label << "_source_real"
            << ',' << label << "_source_imag"
            << ',' << label << "_power"
            << ',' << label << "_voltage_real"
            << ',' << label << "_voltage_imag"
            << ',' << label << "_current_real"
            << ',' << label << "_current_imag"
            << ',' << label << "_conductance_power"
            << ',' << label << "_source_power"
            << ',' << label << "_accepted_power";
      }
    }
    csv << '\n';

    if (verbosity > 0) {
      cout << "Writing NF2FF CSV output to " << filename.string() << endl;
    }
  }

  const ScalarT nan = std::numeric_limits<ScalarT>::quiet_NaN();
  const int nangles = nf2ff.ntheta*nf2ff.nphi;

  for (size_t freq_index = 0; freq_index < nf2ff.frequencies.size();
       ++freq_index) {
    Teuchos::Array<ScalarT> local_values(8*nangles, 0.0);
    Teuchos::Array<ScalarT> global_values(8*nangles, 0.0);
    ScalarT local_radiated_power = 0.0;
    ScalarT global_radiated_power = 0.0;

    const ScalarT frequency = nf2ff.frequencies[freq_index];
    const ScalarT k0 = 2.0*PI*frequency/nf2ff.c0;

    for (size_t surface_index = 0;
         surface_index < nf2ff_surface_groups.size(); ++surface_index) {
      const size_t block = nf2ff_surface_groups[surface_index].block;
      const size_t group = nf2ff_surface_groups[surface_index].group;
      auto boundary_group = assembler->boundary_groups[block][group];

      auto dft_host = create_mirror_view(
        nf2ff_surface_groups[surface_index].electric_E_dft);
      deep_copy(dft_host, nf2ff_surface_groups[surface_index].electric_E_dft);

      auto wts_host = create_mirror_view(boundary_group->wts);
      deep_copy(wts_host, boundary_group->wts);

      vector<decltype(create_mirror_view(boundary_group->ip[0]))> ip_host(3);
      vector<decltype(create_mirror_view(boundary_group->normals[0]))>
        normals_host(3);
      for (int d = 0; d < 3; ++d) {
        ip_host[d] = create_mirror_view(boundary_group->ip[d]);
        normals_host[d] = create_mirror_view(boundary_group->normals[d]);
        deep_copy(ip_host[d], boundary_group->ip[d]);
        deep_copy(normals_host[d], boundary_group->normals[d]);
      }

      for (size_type elem = 0; elem < wts_host.extent(0); ++elem) {
        for (size_type pt = 0; pt < wts_host.extent(1); ++pt) {
          const ScalarT nx = normals_host[0](elem, pt);
          const ScalarT ny = normals_host[1](elem, pt);
          const ScalarT nz = normals_host[2](elem, pt);
          const ScalarT Exr = dft_host(freq_index, elem, pt, 0, 0);
          const ScalarT Eyr = dft_host(freq_index, elem, pt, 1, 0);
          const ScalarT Ezr = dft_host(freq_index, elem, pt, 2, 0);
          const ScalarT Exi = dft_host(freq_index, elem, pt, 0, 1);
          const ScalarT Eyi = dft_host(freq_index, elem, pt, 1, 1);
          const ScalarT Ezi = dft_host(freq_index, elem, pt, 2, 1);
          const ScalarT nxe_r[3] = {
            ny*Ezr - nz*Eyr,
            nz*Exr - nx*Ezr,
            nx*Eyr - ny*Exr
          };
          const ScalarT nxe_i[3] = {
            ny*Ezi - nz*Eyi,
            nz*Exi - nx*Ezi,
            nx*Eyi - ny*Exi
          };
          const ScalarT x[3] = {
            ip_host[0](elem, pt),
            ip_host[1](elem, pt),
            ip_host[2](elem, pt)
          };
          const ScalarT weight = wts_host(elem, pt);

          if (radiation_mode) {
            local_radiated_power += 0.5*weight/nf2ff.eta0*
              (nxe_r[0]*nxe_r[0] + nxe_r[1]*nxe_r[1] +
               nxe_r[2]*nxe_r[2] + nxe_i[0]*nxe_i[0] +
               nxe_i[1]*nxe_i[1] + nxe_i[2]*nxe_i[2]);
          }

          for (int iphi = 0; iphi < nf2ff.nphi; ++iphi) {
            const ScalarT phi = phi_deg[iphi]*PI/180.0;
            const ScalarT sin_phi = sin(phi);
            const ScalarT cos_phi = cos(phi);

            for (int itheta = 0; itheta < nf2ff.ntheta; ++itheta) {
              const ScalarT theta = theta_deg[itheta]*PI/180.0;
              const ScalarT sin_theta = sin(theta);
              const ScalarT cos_theta = cos(theta);

              const ScalarT rhat[3] = {
                sin_theta*cos_phi,
                sin_theta*sin_phi,
                cos_theta
              };
              const ScalarT theta_hat[3] = {
                cos_theta*cos_phi,
                cos_theta*sin_phi,
                -sin_theta
              };
              const ScalarT phi_hat[3] = {
                -sin_phi,
                cos_phi,
                0.0
              };
              const ScalarT nx_theta[3] = {
                ny*theta_hat[2] - nz*theta_hat[1],
                nz*theta_hat[0] - nx*theta_hat[2],
                nx*theta_hat[1] - ny*theta_hat[0]
              };
              const ScalarT nx_phi[3] = {
                ny*phi_hat[2] - nz*phi_hat[1],
                nz*phi_hat[0] - nx*phi_hat[2],
                nx*phi_hat[1] - ny*phi_hat[0]
              };

              const ScalarT phase = k0*(rhat[0]*x[0] +
                                        rhat[1]*x[1] +
                                        rhat[2]*x[2]);
              const std::complex<ScalarT> phasor(cos(phase), sin(phase));
              const std::complex<ScalarT> nxe_theta(
                nx_theta[0]*nxe_r[0] + nx_theta[1]*nxe_r[1] +
                nx_theta[2]*nxe_r[2],
                nx_theta[0]*nxe_i[0] + nx_theta[1]*nxe_i[1] +
                nx_theta[2]*nxe_i[2]);
              const std::complex<ScalarT> nxe_phi(
                nx_phi[0]*nxe_r[0] + nx_phi[1]*nxe_r[1] +
                nx_phi[2]*nxe_r[2],
                nx_phi[0]*nxe_i[0] + nx_phi[1]*nxe_i[1] +
                nx_phi[2]*nxe_i[2]);
              const std::complex<ScalarT> theta_nxe(
                theta_hat[0]*nxe_r[0] + theta_hat[1]*nxe_r[1] +
                theta_hat[2]*nxe_r[2],
                theta_hat[0]*nxe_i[0] + theta_hat[1]*nxe_i[1] +
                theta_hat[2]*nxe_i[2]);
              const std::complex<ScalarT> phi_nxe(
                phi_hat[0]*nxe_r[0] + phi_hat[1]*nxe_r[1] +
                phi_hat[2]*nxe_r[2],
                phi_hat[0]*nxe_i[0] + phi_hat[1]*nxe_i[1] +
                phi_hat[2]*nxe_i[2]);

              const std::complex<ScalarT> A_theta =
                -(weight/nf2ff.eta0)*phasor*nxe_theta;
              const std::complex<ScalarT> A_phi =
                -(weight/nf2ff.eta0)*phasor*nxe_phi;
              const std::complex<ScalarT> F_theta =
                -weight*phasor*theta_nxe;
              const std::complex<ScalarT> F_phi =
                -weight*phasor*phi_nxe;

              const int angle = itheta*nf2ff.nphi + iphi;
              local_values[8*angle + 0] += A_theta.real();
              local_values[8*angle + 1] += A_theta.imag();
              local_values[8*angle + 2] += A_phi.real();
              local_values[8*angle + 3] += A_phi.imag();
              local_values[8*angle + 4] += F_theta.real();
              local_values[8*angle + 5] += F_theta.imag();
              local_values[8*angle + 6] += F_phi.real();
              local_values[8*angle + 7] += F_phi.imag();
            }
          }
        }
      }
    }

    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 8*nangles,
                       &local_values[0], &global_values[0]);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1,
                       &local_radiated_power, &global_radiated_power);

    Teuchos::Array<ScalarT> local_port_values(
      static_cast<int>(3*nf2ff_ports.size()), 0.0);
    Teuchos::Array<ScalarT> global_port_values(
      static_cast<int>(3*nf2ff_ports.size()), 0.0);

    if (radiation_mode) {
      for (size_t port_group_index = 0;
           port_group_index < nf2ff_port_groups.size();
           ++port_group_index) {
        const size_t port_index =
          nf2ff_port_groups[port_group_index].port;
        const size_t block = nf2ff_port_groups[port_group_index].block;
        const size_t group = nf2ff_port_groups[port_group_index].group;
        const NF2FFPort & port = nf2ff_ports[port_index];

        auto dft_host = create_mirror_view(
          nf2ff_port_groups[port_group_index].electric_E_dft);
        deep_copy(dft_host,
                  nf2ff_port_groups[port_group_index].electric_E_dft);

        auto wts = assembler->groups[block][group]->getWts();
        auto wts_host = create_mirror_view(wts);
        deep_copy(wts_host, wts);

        for (size_type elem = 0; elem < wts_host.extent(0); ++elem) {
          for (size_type pt = 0; pt < wts_host.extent(1); ++pt) {
            const ScalarT Exr = dft_host(freq_index, elem, pt, 0, 0);
            const ScalarT Eyr = dft_host(freq_index, elem, pt, 1, 0);
            const ScalarT Ezr = dft_host(freq_index, elem, pt, 2, 0);
            const ScalarT Exi = dft_host(freq_index, elem, pt, 0, 1);
            const ScalarT Eyi = dft_host(freq_index, elem, pt, 1, 1);
            const ScalarT Ezi = dft_host(freq_index, elem, pt, 2, 1);
            const ScalarT Epr = port.polarization_x*Exr +
                                 port.polarization_y*Eyr +
                                 port.polarization_z*Ezr;
            const ScalarT Epi = port.polarization_x*Exi +
                                 port.polarization_y*Eyi +
                                 port.polarization_z*Ezi;
            const ScalarT weight = wts_host(elem, pt);

            local_port_values[3*port_index + 0] += weight*Epr;
            local_port_values[3*port_index + 1] += weight*Epi;
            local_port_values[3*port_index + 2] +=
              0.5*port.conductivity*weight*(Epr*Epr + Epi*Epi);
          }
        }
      }

      if (!nf2ff_ports.empty()) {
        Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM,
                           static_cast<int>(local_port_values.size()),
                           &local_port_values[0], &global_port_values[0]);
      }
    }

    if (Comm->getRank() == 0) {
      std::complex<ScalarT> field_norm(1.0, 0.0);
      if (scattering_mode) {
        TEUCHOS_TEST_FOR_EXCEPTION(!nf2ff.scattering_source_initialized,
                                   std::runtime_error,
                                   "NF2FF scattering source data were not initialized.");
        const std::complex<ScalarT> source_dft =
          (std::abs(nf2ff.source_te) > 1.0e-30) ?
          nf2ff.source_te_dft[freq_index] :
          nf2ff.source_tm_dft[freq_index];
        TEUCHOS_TEST_FOR_EXCEPTION(
          std::norm(source_dft) <= 1.0e-30, std::runtime_error,
          "NF2FF scattering normalization failed at frequency " << frequency
          << ". The selected source waveform has a near-zero DFT.");
        field_norm = 1.0/source_dft;
      }
      else if (!nf2ff_ports.empty()) {
        const std::complex<ScalarT> source_dft =
          nf2ff_ports[0].source_dft[freq_index];
        TEUCHOS_TEST_FOR_EXCEPTION(
          std::norm(source_dft) <= 1.0e-30, std::runtime_error,
          "NF2FF radiation normalization failed at frequency " << frequency
          << ". The port source waveform has a near-zero DFT.");
        field_norm = 1.0/source_dft;
      }

      const ScalarT field_scale = k0/(4.0*PI);
      const ScalarT power_scale =
        k0*k0/((4.0*PI)*(4.0*PI)*nf2ff.eta0);
      const ScalarT incident_field_squared =
        nf2ff.source_amplitude*nf2ff.source_amplitude;

      vector<std::complex<ScalarT> > port_voltage(nf2ff_ports.size());
      vector<std::complex<ScalarT> > port_current(nf2ff_ports.size());
      vector<std::complex<ScalarT> > port_input_impedance(nf2ff_ports.size());
      vector<std::complex<ScalarT> > port_reflection_coefficient(nf2ff_ports.size());
      vector<ScalarT> port_conductance_power(nf2ff_ports.size(), nan);
      vector<ScalarT> port_source_power(nf2ff_ports.size(), nan);
      vector<ScalarT> port_accepted_power(nf2ff_ports.size(), nan);
      vector<ScalarT> port_available_power(nf2ff_ports.size(), nan);
      vector<ScalarT> port_mismatch_factor(nf2ff_ports.size(), nan);

      ScalarT accepted_power = nan;
      ScalarT available_power = nan;
      ScalarT mismatch_factor = nan;
      if (radiation_mode && !nf2ff_ports.empty()) {
        for (size_t port_index = 0; port_index < nf2ff_ports.size();
             ++port_index) {
          const NF2FFPort & port = nf2ff_ports[port_index];
          const std::complex<ScalarT> voltage_raw(
            port.height*global_port_values[3*port_index + 0]/port.volume,
            port.height*global_port_values[3*port_index + 1]/port.volume);
          port_voltage[port_index] = field_norm*voltage_raw;
          port_conductance_power[port_index] =
            std::norm(field_norm)*global_port_values[3*port_index + 2];

          const std::complex<ScalarT> source_current(
            -2.0*port.amplitude/std::sqrt(port.impedance), 0.0);
          port_current[port_index] =
            -source_current - port_voltage[port_index]/port.impedance;
          port_source_power[port_index] =
            -0.5*std::real(port_voltage[port_index]*
                           std::conj(source_current));
          port_accepted_power[port_index] =
            0.5*std::real(port_voltage[port_index]*
                          std::conj(port_current[port_index]));
          port_available_power[port_index] =
            std::norm(source_current)*port.impedance/8.0;

          const ScalarT current_norm = std::norm(port_current[port_index]);
          if (current_norm > 1.0e-30) {
            port_input_impedance[port_index] =
              port_voltage[port_index]/port_current[port_index];
            const std::complex<ScalarT> denominator =
              port_input_impedance[port_index] + port.impedance;
            if (std::norm(denominator) > 1.0e-30) {
              port_reflection_coefficient[port_index] =
                (port_input_impedance[port_index] - port.impedance)/
                denominator;
              port_mismatch_factor[port_index] =
                1.0 - std::norm(port_reflection_coefficient[port_index]);
              if (port_mismatch_factor[port_index] < 0.0 &&
                  port_mismatch_factor[port_index] > -1.0e-12) {
                port_mismatch_factor[port_index] = 0.0;
              }
            }
          }
        }

        accepted_power = port_accepted_power[0];
        available_power = port_available_power[0];
        mismatch_factor = port_mismatch_factor[0];
        const ScalarT tolerance =
          1.0e-12*std::max(ScalarT(1.0),
                            std::abs(port_source_power[0]));
        if (accepted_power < 0.0 && accepted_power > -tolerance) {
          accepted_power = 0.0;
          port_accepted_power[0] = 0.0;
        }
      }

      const ScalarT normalized_radiated_power =
        std::norm(field_norm)*global_radiated_power;
      const ScalarT directivity_scale =
        (normalized_radiated_power > 1.0e-30) ?
        4.0*PI/normalized_radiated_power : nan;
      const ScalarT gain_scale =
        (accepted_power > 1.0e-30) ?
        4.0*PI/accepted_power : nan;
      const ScalarT realized_gain_scale =
        (gain_scale == gain_scale && mismatch_factor == mismatch_factor) ?
        gain_scale*mismatch_factor :
        ((available_power > 1.0e-30) ? 4.0*PI/available_power : nan);

      for (int iphi = 0; iphi < nf2ff.nphi; ++iphi) {
        for (int itheta = 0; itheta < nf2ff.ntheta; ++itheta) {
          const int angle = itheta*nf2ff.nphi + iphi;
          const std::complex<ScalarT> A_theta_raw(
            global_values[8*angle + 0], global_values[8*angle + 1]);
          const std::complex<ScalarT> A_phi_raw(
            global_values[8*angle + 2], global_values[8*angle + 3]);
          const std::complex<ScalarT> F_theta_raw(
            global_values[8*angle + 4], global_values[8*angle + 5]);
          const std::complex<ScalarT> F_phi_raw(
            global_values[8*angle + 6], global_values[8*angle + 7]);

          const std::complex<ScalarT> A_theta = field_norm*A_theta_raw;
          const std::complex<ScalarT> A_phi = field_norm*A_phi_raw;
          const std::complex<ScalarT> F_theta = field_norm*F_theta_raw;
          const std::complex<ScalarT> F_phi = field_norm*F_phi_raw;

          const std::complex<ScalarT> E_theta =
            field_scale*(nf2ff.eta0*A_theta + F_phi);
          const std::complex<ScalarT> E_phi =
            field_scale*(nf2ff.eta0*A_phi - F_theta);
          const std::complex<ScalarT> H_theta = -E_phi/nf2ff.eta0;
          const std::complex<ScalarT> H_phi = E_theta/nf2ff.eta0;
          const ScalarT P_theta = power_scale*
            std::norm(nf2ff.eta0*A_theta + F_phi);
          const ScalarT P_phi = power_scale*
            std::norm(nf2ff.eta0*A_phi - F_theta);
          const ScalarT P_total = P_theta + P_phi;

          csv << frequency << ',' << theta_deg[itheta] << ','
              << phi_deg[iphi]
              << ',' << A_theta.real() << ',' << A_theta.imag()
              << ',' << A_phi.real() << ',' << A_phi.imag()
              << ',' << F_theta.real() << ',' << F_theta.imag()
              << ',' << F_phi.real() << ',' << F_phi.imag()
              << ',' << E_theta.real() << ',' << E_theta.imag()
              << ',' << E_phi.real() << ',' << E_phi.imag()
              << ',' << H_theta.real() << ',' << H_theta.imag()
              << ',' << H_phi.real() << ',' << H_phi.imag()
              << ',' << P_theta << ',' << P_phi << ',' << P_total;

          if (scattering_mode) {
            const ScalarT rcs_scale =
              (incident_field_squared > 1.0e-30) ?
              4.0*PI/incident_field_squared : nan;
            const ScalarT RCS_theta = rcs_scale*std::norm(E_theta);
            const ScalarT RCS_phi = rcs_scale*std::norm(E_phi);
            csv << ',' << RCS_theta << ',' << RCS_phi << ','
                << RCS_theta + RCS_phi;
          }
          else {
            const ScalarT Directivity_theta = directivity_scale*P_theta;
            const ScalarT Directivity_phi = directivity_scale*P_phi;
            const ScalarT Directivity_total = directivity_scale*P_total;
            const ScalarT Gain_theta = gain_scale*P_theta;
            const ScalarT Gain_phi = gain_scale*P_phi;
            const ScalarT Gain_total = gain_scale*P_total;
            const ScalarT Realized_gain_theta =
              realized_gain_scale*P_theta;
            const ScalarT Realized_gain_phi =
              realized_gain_scale*P_phi;
            const ScalarT Realized_gain_total =
              realized_gain_scale*P_total;
            csv << ',' << Directivity_theta << ',' << Directivity_phi << ','
                << Directivity_total
                << ',' << Gain_theta << ',' << Gain_phi << ','
                << Gain_total
                << ',' << Realized_gain_theta << ','
                << Realized_gain_phi << ',' << Realized_gain_total
                << ',' << normalized_radiated_power
                << ',' << accepted_power;

            for (size_t port_index = 0;
                 port_index < nf2ff_ports.size(); ++port_index) {
              csv << ',' << nf2ff_ports[port_index].source_dft[freq_index].real()
                  << ',' << nf2ff_ports[port_index].source_dft[freq_index].imag()
                  << ',' << port_accepted_power[port_index]
                  << ',' << port_voltage[port_index].real()
                  << ',' << port_voltage[port_index].imag()
                  << ',' << port_current[port_index].real()
                  << ',' << port_current[port_index].imag()
                  << ',' << port_conductance_power[port_index]
                  << ',' << port_source_power[port_index]
                  << ',' << port_accepted_power[port_index];
            }
          }

          csv << '\n';
        }
      }
    }
  }
}


// ========================================================================================
// Write lumped-port parameters in CSV format.
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::writeLumpedPortParameters()
{
  if (!lumped_port_parameters.save) {
    return;
  }

  const ScalarT nan = std::numeric_limits<ScalarT>::quiet_NaN();
  const size_t nfrequency = lumped_port_parameters.frequencies.size();
  const size_t nports = lumped_port_parameter_ports.size();

  struct LumpedPortResult {
    ScalarT incident_amplitude;
    ScalarT incident_power;
    std::complex<ScalarT> S11;
    std::complex<ScalarT> Zin;
    std::complex<ScalarT> Gamma;
    ScalarT VSWR;
    ScalarT radiation_efficiency;
    ScalarT realized_efficiency;
    ScalarT Q_Z;
    ScalarT Q_rad;
    ScalarT Q_rad_field;
  };

  vector<vector<LumpedPortResult> > results(
    nfrequency, vector<LumpedPortResult>(nports));

  const bool have_radiation_surface =
    lumped_port_parameters.has_radiation_surface;
  const bool have_stored_energy_q =
    lumped_port_parameters.save_stored_energy_q;
  if (have_radiation_surface) {
    TEUCHOS_TEST_FOR_EXCEPTION(
      !lumped_port_parameters.radiation_constants_initialized ||
      lumped_port_parameters.radiation_eta0 <= 0.0,
      std::runtime_error,
      "Lumped port radiation sideset constants were not initialized.");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(
    have_stored_energy_q && !have_radiation_surface,
    std::runtime_error,
    "Stored-energy Q requires a lumped-port radiation sideset.");

  for (size_t freq_index = 0; freq_index < nfrequency; ++freq_index) {
    ScalarT local_radiated_power = 0.0;
    ScalarT global_radiated_power = 0.0;
    ScalarT local_stored_energy = 0.0;
    ScalarT global_stored_energy = 0.0;

    if (have_radiation_surface) {
      for (size_t surface_index = 0;
           surface_index < lumped_port_parameter_surface_groups.size();
           ++surface_index) {
        const size_t block =
          lumped_port_parameter_surface_groups[surface_index].block;
        const size_t group =
          lumped_port_parameter_surface_groups[surface_index].group;
        auto boundary_group = assembler->boundary_groups[block][group];

        auto dft_host = create_mirror_view(
          lumped_port_parameter_surface_groups[surface_index].electric_E_dft);
        deep_copy(
          dft_host,
          lumped_port_parameter_surface_groups[surface_index].electric_E_dft);

        auto wts_host = create_mirror_view(boundary_group->wts);
        deep_copy(wts_host, boundary_group->wts);

        vector<decltype(create_mirror_view(boundary_group->normals[0]))>
          normals_host(3);
        for (int d = 0; d < 3; ++d) {
          normals_host[d] = create_mirror_view(
            boundary_group->normals[d]);
          deep_copy(normals_host[d], boundary_group->normals[d]);
        }

        for (size_type elem = 0; elem < wts_host.extent(0); ++elem) {
          for (size_type pt = 0; pt < wts_host.extent(1); ++pt) {
            const ScalarT nx = normals_host[0](elem, pt);
            const ScalarT ny = normals_host[1](elem, pt);
            const ScalarT nz = normals_host[2](elem, pt);

            const ScalarT Exr = dft_host(freq_index, elem, pt, 0, 0);
            const ScalarT Eyr = dft_host(freq_index, elem, pt, 1, 0);
            const ScalarT Ezr = dft_host(freq_index, elem, pt, 2, 0);
            const ScalarT Exi = dft_host(freq_index, elem, pt, 0, 1);
            const ScalarT Eyi = dft_host(freq_index, elem, pt, 1, 1);
            const ScalarT Ezi = dft_host(freq_index, elem, pt, 2, 1);

            const ScalarT nxe_r_x = ny*Ezr - nz*Eyr;
            const ScalarT nxe_r_y = nz*Exr - nx*Ezr;
            const ScalarT nxe_r_z = nx*Eyr - ny*Exr;
            const ScalarT nxe_i_x = ny*Ezi - nz*Eyi;
            const ScalarT nxe_i_y = nz*Exi - nx*Ezi;
            const ScalarT nxe_i_z = nx*Eyi - ny*Exi;

            local_radiated_power +=
              wts_host(elem, pt)/lumped_port_parameters.radiation_eta0*
              (nxe_r_x*nxe_r_x + nxe_r_y*nxe_r_y +
               nxe_r_z*nxe_r_z + nxe_i_x*nxe_i_x +
               nxe_i_y*nxe_i_y + nxe_i_z*nxe_i_z);
          }
        }
      }

      Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1,
                         &local_radiated_power, &global_radiated_power);
    }

    if (have_stored_energy_q) {
      for (size_t volume_index = 0;
           volume_index < lumped_port_parameter_volume_groups.size();
           ++volume_index) {
        const auto & volume_group =
          lumped_port_parameter_volume_groups[volume_index];
        auto dft_host = create_mirror_view(volume_group.field_dft);
        deep_copy(dft_host, volume_group.field_dft);
        auto coefficients_host = create_mirror_view(
          volume_group.energy_coefficients);
        deep_copy(coefficients_host, volume_group.energy_coefficients);

        auto element_group = assembler->groups[volume_group.block][volume_group.group];
        auto wts = element_group->getWts();
        auto wts_host = create_mirror_view(wts);
        deep_copy(wts_host, wts);
        auto ip = element_group->getIntegrationPts();
        vector<decltype(create_mirror_view(ip[0]))> ip_host(3);
        for (int d = 0; d < 3; ++d) {
          ip_host[d] = create_mirror_view(ip[d]);
          deep_copy(ip_host[d], ip[d]);
        }

        for (size_type elem = 0; elem < wts_host.extent(0); ++elem) {
          for (size_type pt = 0; pt < wts_host.extent(1); ++pt) {
            const ScalarT Exr = dft_host(freq_index, elem, pt, 0, 0);
            const ScalarT Eyr = dft_host(freq_index, elem, pt, 1, 0);
            const ScalarT Ezr = dft_host(freq_index, elem, pt, 2, 0);
            const ScalarT Exi = dft_host(freq_index, elem, pt, 0, 1);
            const ScalarT Eyi = dft_host(freq_index, elem, pt, 1, 1);
            const ScalarT Ezi = dft_host(freq_index, elem, pt, 2, 1);
            const ScalarT Hxr = dft_host(freq_index, elem, pt, 3, 0);
            const ScalarT Hyr = dft_host(freq_index, elem, pt, 4, 0);
            const ScalarT Hzr = dft_host(freq_index, elem, pt, 5, 0);
            const ScalarT Hxi = dft_host(freq_index, elem, pt, 3, 1);
            const ScalarT Hyi = dft_host(freq_index, elem, pt, 4, 1);
            const ScalarT Hzi = dft_host(freq_index, elem, pt, 5, 1);

            const ScalarT field_norm[6] = {
              Exr*Exr + Exi*Exi,
              Eyr*Eyr + Eyi*Eyi,
              Ezr*Ezr + Ezi*Ezi,
              Hxr*Hxr + Hxi*Hxi,
              Hyr*Hyr + Hyi*Hyi,
              Hzr*Hzr + Hzi*Hzi};
            const ScalarT field_dot[15] = {
              Exr*Eyr + Exi*Eyi,
              Exr*Ezr + Exi*Ezi,
              Eyr*Ezr + Eyi*Ezi,
              Exr*Hxr + Exi*Hxi,
              Exr*Hyr + Exi*Hyi,
              Exr*Hzr + Exi*Hzi,
              Eyr*Hxr + Eyi*Hxi,
              Eyr*Hyr + Eyi*Hyi,
              Eyr*Hzr + Eyi*Hzi,
              Ezr*Hxr + Ezi*Hxi,
              Ezr*Hyr + Ezi*Hyi,
              Ezr*Hzr + Ezi*Hzi,
              Hxr*Hyr + Hxi*Hyi,
              Hxr*Hzr + Hxi*Hzi,
              Hyr*Hzr + Hyi*Hzi};

            const ScalarT field_energy = 0.5*(
              coefficients_host(elem, pt, 0)*field_norm[0] +
              2.0*coefficients_host(elem, pt, 1)*field_dot[0] +
              2.0*coefficients_host(elem, pt, 2)*field_dot[1] +
              coefficients_host(elem, pt, 3)*field_norm[1] +
              2.0*coefficients_host(elem, pt, 4)*field_dot[2] +
              coefficients_host(elem, pt, 5)*field_norm[2] +
              2.0*coefficients_host(elem, pt, 6)*field_dot[3] +
              2.0*coefficients_host(elem, pt, 7)*field_dot[4] +
              2.0*coefficients_host(elem, pt, 8)*field_dot[5] +
              2.0*coefficients_host(elem, pt, 9)*field_dot[6] +
              2.0*coefficients_host(elem, pt, 10)*field_dot[7] +
              2.0*coefficients_host(elem, pt, 11)*field_dot[8] +
              2.0*coefficients_host(elem, pt, 12)*field_dot[9] +
              2.0*coefficients_host(elem, pt, 13)*field_dot[10] +
              2.0*coefficients_host(elem, pt, 14)*field_dot[11] +
              coefficients_host(elem, pt, 15)*field_norm[3] +
              2.0*coefficients_host(elem, pt, 16)*field_dot[12] +
              2.0*coefficients_host(elem, pt, 17)*field_dot[13] +
              coefficients_host(elem, pt, 18)*field_norm[4] +
              2.0*coefficients_host(elem, pt, 19)*field_dot[14] +
              coefficients_host(elem, pt, 20)*field_norm[5]);

            ScalarT radiating_energy = 0.0;
            if (coefficients_host(elem, pt, 21) > 0.0) {
              const ScalarT rx = ip_host[0](elem, pt) -
                lumped_port_parameters.radiation_center_x;
              const ScalarT ry = ip_host[1](elem, pt) -
                lumped_port_parameters.radiation_center_y;
              const ScalarT rz = ip_host[2](elem, pt) -
                lumped_port_parameters.radiation_center_z;
              const ScalarT radius = sqrt(rx*rx + ry*ry + rz*rz);
              if (radius > 1.0e-30) {
                const ScalarT poynting_x = Eyr*Hzr + Eyi*Hzi -
                  Ezr*Hyr - Ezi*Hyi;
                const ScalarT poynting_y = Ezr*Hxr + Ezi*Hxi -
                  Exr*Hzr - Exi*Hzi;
                const ScalarT poynting_z = Exr*Hyr + Exi*Hyi -
                  Eyr*Hxr - Eyi*Hxi;
                const ScalarT outward_poynting =
                  (poynting_x*rx + poynting_y*ry + poynting_z*rz)/radius;
                if (outward_poynting > 0.0) {
                  radiating_energy = coefficients_host(elem, pt, 21)*
                    outward_poynting;
                }
              }
            }

            local_stored_energy += wts_host(elem, pt)*
              (field_energy - radiating_energy);
          }
        }
      }

      Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1,
                         &local_stored_energy, &global_stored_energy);
    }

    Teuchos::Array<ScalarT> local_port_values(
      static_cast<int>(3*nports), 0.0);
    Teuchos::Array<ScalarT> global_port_values(
      static_cast<int>(3*nports), 0.0);

    for (size_t port_group_index = 0;
         port_group_index < lumped_port_parameter_port_groups.size();
         ++port_group_index) {
      const size_t port_index =
        lumped_port_parameter_port_groups[port_group_index].port;
      const size_t block =
        lumped_port_parameter_port_groups[port_group_index].block;
      const size_t group =
        lumped_port_parameter_port_groups[port_group_index].group;
      const NF2FFPort & port = lumped_port_parameter_ports[port_index];

      auto dft_host = create_mirror_view(
        lumped_port_parameter_port_groups[port_group_index].electric_E_dft);
      deep_copy(
        dft_host,
        lumped_port_parameter_port_groups[port_group_index].electric_E_dft);

      auto wts = assembler->groups[block][group]->getWts();
      auto wts_host = create_mirror_view(wts);
      deep_copy(wts_host, wts);

      for (size_type elem = 0; elem < wts_host.extent(0); ++elem) {
        for (size_type pt = 0; pt < wts_host.extent(1); ++pt) {
          const ScalarT Exr = dft_host(freq_index, elem, pt, 0, 0);
          const ScalarT Eyr = dft_host(freq_index, elem, pt, 1, 0);
          const ScalarT Ezr = dft_host(freq_index, elem, pt, 2, 0);
          const ScalarT Exi = dft_host(freq_index, elem, pt, 0, 1);
          const ScalarT Eyi = dft_host(freq_index, elem, pt, 1, 1);
          const ScalarT Ezi = dft_host(freq_index, elem, pt, 2, 1);

          const ScalarT Epr =
            port.polarization_x*Exr +
            port.polarization_y*Eyr +
            port.polarization_z*Ezr;
          const ScalarT Epi =
            port.polarization_x*Exi +
            port.polarization_y*Eyi +
            port.polarization_z*Ezi;
          const ScalarT weight = wts_host(elem, pt);

          local_port_values[3*port_index + 0] += weight*Epr;
          local_port_values[3*port_index + 1] += weight*Epi;
          local_port_values[3*port_index + 2] +=
            port.conductivity*weight*(Epr*Epr + Epi*Epi);
        }
      }
    }

    if (nports > 0) {
      Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM,
                         static_cast<int>(local_port_values.size()),
                         &local_port_values[0], &global_port_values[0]);
    }

    if (Comm->getRank() == 0) {
      for (size_t port_index = 0; port_index < nports; ++port_index) {
        const NF2FFPort & port = lumped_port_parameter_ports[port_index];
        LumpedPortResult & result = results[freq_index][port_index];
        result.incident_amplitude = port.amplitude;
        result.incident_power = result.incident_amplitude*
                                result.incident_amplitude;
        result.S11 = std::complex<ScalarT>(nan, nan);
        result.Zin = std::complex<ScalarT>(nan, nan);
        result.Gamma = std::complex<ScalarT>(nan, nan);
        result.VSWR = nan;
        result.radiation_efficiency = nan;
        result.realized_efficiency = nan;
        result.Q_Z = nan;
        result.Q_rad = nan;
        result.Q_rad_field = nan;

        const std::complex<ScalarT> source_dft =
          port.source_dft[freq_index];

        if (std::norm(source_dft) > 1.0e-30) {
          ScalarT port_power = global_port_values[3*port_index + 2];
          const ScalarT tolerance =
            1.0e-12*std::max(ScalarT(1.0), std::abs(port_power));
          if (port_power < 0.0 && port_power > -tolerance) {
            port_power = 0.0;
          }

          TEUCHOS_TEST_FOR_EXCEPTION(
            port_power < 0.0, std::runtime_error,
            "Computed negative lumped-port power. Check the port conductance sign.");

          const std::complex<ScalarT> projection(
            port.height*global_port_values[3*port_index + 0]/port.volume,
            port.height*global_port_values[3*port_index + 1]/port.volume);
          std::complex<ScalarT> phase(1.0, 0.0);
          if (std::norm(projection) > 1.0e-30) {
            phase = projection/std::abs(projection);
          }

          const std::complex<ScalarT> Iinc =
            source_dft/std::sqrt(port.impedance);
          const std::complex<ScalarT> I1 =
            std::sqrt(port_power/port.impedance)*phase;
          const std::complex<ScalarT> Iref = I1 - Iinc;
          result.S11 = Iref/Iinc;

          const std::complex<ScalarT> one(1.0, 0.0);
          const std::complex<ScalarT> zin_denominator = one - result.S11;
          if (std::norm(zin_denominator) > 1.0e-30) {
            result.Zin = port.impedance*(one + result.S11)/zin_denominator;
            const std::complex<ScalarT> gamma_denominator =
              result.Zin + port.impedance;
            if (std::norm(gamma_denominator) > 1.0e-30) {
              result.Gamma =
                (result.Zin - port.impedance)/gamma_denominator;
              const ScalarT gamma_magnitude = std::abs(result.Gamma);
              result.VSWR = (gamma_magnitude < 1.0) ?
                (1.0 + gamma_magnitude)/(1.0 - gamma_magnitude) :
                std::numeric_limits<ScalarT>::infinity();
            }
          }

          if (have_radiation_surface) {
            const ScalarT source_magnitude_squared =
              std::norm(source_dft);
            const ScalarT one_minus_s11 = 1.0 - std::norm(result.S11);
            const ScalarT normalized_radiated_power =
              global_radiated_power*result.incident_power/
              source_magnitude_squared;

            if (result.incident_power > 1.0e-30) {
              result.realized_efficiency =
                normalized_radiated_power/result.incident_power;
            }
            if (one_minus_s11 > 1.0e-30) {
              result.radiation_efficiency =
                normalized_radiated_power/
                (result.incident_power*one_minus_s11);
            }
          }

          if (have_stored_energy_q && global_radiated_power > 1.0e-30 &&
              global_stored_energy > 0.0) {
            const ScalarT omega = 2.0*PI*
              lumped_port_parameters.frequencies[freq_index];
            result.Q_rad_field = omega*global_stored_energy/
              global_radiated_power;
          }
        }
      }
    }
  }

  if (Comm->getRank() == 0) {
    for (size_t port_index = 0; port_index < nports; ++port_index) {
      for (size_t freq_index = 0; freq_index < nfrequency; ++freq_index) {
        LumpedPortResult & result = results[freq_index][port_index];
        const std::complex<ScalarT> Zin = result.Zin;
        const ScalarT resistance = Zin.real();
        std::complex<ScalarT> dZin_domega(nan, nan);

        if (nfrequency == 2) {
          const ScalarT omega0 =
            2.0*PI*lumped_port_parameters.frequencies[0];
          const ScalarT omega1 =
            2.0*PI*lumped_port_parameters.frequencies[1];
          const std::complex<ScalarT> Z0 = results[0][port_index].Zin;
          const std::complex<ScalarT> Z1 = results[1][port_index].Zin;
          if (std::isfinite(Z0.real()) && std::isfinite(Z0.imag()) &&
              std::isfinite(Z1.real()) && std::isfinite(Z1.imag()) &&
              std::abs(omega1 - omega0) > 1.0e-30) {
            dZin_domega = (Z1 - Z0)/(omega1 - omega0);
          }
        }
        else if (nfrequency >= 3) {
          size_t i0 = 0;
          size_t i1 = 1;
          size_t i2 = 2;
          if (freq_index > 0 && freq_index + 1 < nfrequency) {
            i0 = freq_index - 1;
            i1 = freq_index;
            i2 = freq_index + 1;
          }
          else if (freq_index + 1 == nfrequency) {
            i0 = nfrequency - 3;
            i1 = nfrequency - 2;
            i2 = nfrequency - 1;
          }

          const ScalarT omega0 =
            2.0*PI*lumped_port_parameters.frequencies[i0];
          const ScalarT omega1 =
            2.0*PI*lumped_port_parameters.frequencies[i1];
          const ScalarT omega2 =
            2.0*PI*lumped_port_parameters.frequencies[i2];
          const std::complex<ScalarT> Z0 = results[i0][port_index].Zin;
          const std::complex<ScalarT> Z1 = results[i1][port_index].Zin;
          const std::complex<ScalarT> Z2 = results[i2][port_index].Zin;
          const ScalarT denominator0 =
            (omega0 - omega1)*(omega0 - omega2);
          const ScalarT denominator1 =
            (omega1 - omega0)*(omega1 - omega2);
          const ScalarT denominator2 =
            (omega2 - omega0)*(omega2 - omega1);

          if (std::isfinite(Z0.real()) && std::isfinite(Z0.imag()) &&
              std::isfinite(Z1.real()) && std::isfinite(Z1.imag()) &&
              std::isfinite(Z2.real()) && std::isfinite(Z2.imag()) &&
              std::abs(denominator0) > 1.0e-30 &&
              std::abs(denominator1) > 1.0e-30 &&
              std::abs(denominator2) > 1.0e-30) {
            const ScalarT omega =
              2.0*PI*lumped_port_parameters.frequencies[freq_index];
            const ScalarT coefficient0 =
              (2.0*omega - omega1 - omega2)/denominator0;
            const ScalarT coefficient1 =
              (2.0*omega - omega0 - omega2)/denominator1;
            const ScalarT coefficient2 =
              (2.0*omega - omega0 - omega1)/denominator2;
            dZin_domega = coefficient0*Z0 + coefficient1*Z1 +
                           coefficient2*Z2;
          }
        }

        const ScalarT omega =
          2.0*PI*lumped_port_parameters.frequencies[freq_index];
        if (std::isfinite(resistance) && resistance > 1.0e-30 &&
            std::isfinite(dZin_domega.real()) &&
            std::isfinite(dZin_domega.imag())) {
          result.Q_Z = omega*std::abs(dZin_domega)/(2.0*resistance);
        }

        if (std::isfinite(result.Q_Z) &&
            std::isfinite(result.radiation_efficiency) &&
            result.radiation_efficiency > 1.0e-30) {
          result.Q_rad = result.Q_Z/result.radiation_efficiency;
        }
      }
    }

    std::ofstream csv;
    std::filesystem::path filename(lumped_port_parameters.output_file);
    if (filename.extension() != ".csv") {
      filename += ".csv";
    }

    std::error_code error;
    if (!filename.parent_path().empty()) {
      std::filesystem::create_directories(filename.parent_path(), error);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(
      static_cast<bool>(error), std::runtime_error,
      "Could not create lumped-port output directory for '"
      << filename.string() << "': " << error.message());

    csv.open(filename.string());
    TEUCHOS_TEST_FOR_EXCEPTION(
      !csv, std::runtime_error,
      "Could not open lumped-port output file '"
      << filename.string() << "'.");

    csv << std::setprecision(17);
    csv << "frequency,port,incident_amplitude,incident_power"
        << ",S11_real,S11_imag"
        << ",Zin_real,Zin_imag"
        << ",Gamma_real,Gamma_imag"
        << ",VSWR"
        << ",radiation_efficiency"
        << ",realized_efficiency"
        << ",Q_Z"
        << ",Q_rad"
        << ",Q_rad_field\n";

    if (verbosity > 0) {
      cout << "Writing lumped-port parameter CSV output to "
           << filename.string() << endl;
    }

    for (size_t freq_index = 0; freq_index < nfrequency; ++freq_index) {
      for (size_t port_index = 0; port_index < nports; ++port_index) {
        const LumpedPortResult & result = results[freq_index][port_index];
        csv << lumped_port_parameters.frequencies[freq_index]
            << ',' << port_index
            << ',' << result.incident_amplitude
            << ',' << result.incident_power
            << ',' << result.S11.real() << ',' << result.S11.imag()
            << ',' << result.Zin.real() << ',' << result.Zin.imag()
            << ',' << result.Gamma.real() << ',' << result.Gamma.imag()
            << ',' << result.VSWR
            << ',' << result.radiation_efficiency
            << ',' << result.realized_efficiency
            << ',' << result.Q_Z
            << ',' << result.Q_rad
            << ',' << result.Q_rad_field
            << '\n';
      }
    }
  }
}


template <class Node>
void PostprocessManager<Node>::report()
{
    
    Teuchos::TimeMonitor localtimer(*reportTimer);
    
    ////////////////////////////////////////////////////////////////////////////
    // Report the responses
    ////////////////////////////////////////////////////////////////////////////
    
    if (compute_response && write_response)
    {
        
        if (Comm->getRank() == 0)
        {
            if (verbosity > 0)
            {
                cout << endl
                << "*********************************************************" << endl;
                cout << "***** Writing responses ******" << endl;
                cout << "*********************************************************" << endl;
            }
        }
        for (size_t obj = 0; obj < objectives.size(); ++obj)
        {
            if (objectives[obj].type == "sensors") {
                // First case: sensors just computed states (faster than other case)
                if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln) {
                    
                    Kokkos::View<ScalarT ***, HostDevice> sensor_data;
                    Kokkos::View<int *, HostDevice> sensorIDs;
                    size_t numtimes = 0;
                    int numfields = 0;
                    
                    int numsensors = objectives[obj].numSensors;
                    
                    if (numsensors > 0)
                    {
                        sensorIDs = Kokkos::View<int *, HostDevice>("sensor IDs owned by proc", numsensors);
                        size_t sprog = 0;
                        auto sensor_found = objectives[obj].sensor_found;
                        for (size_type s = 0; s < sensor_found.extent(0); ++s)
                        {
                            if (sensor_found(s))
                            {
                                sensorIDs(sprog) = s;
                                ++sprog;
                            }
                        }
                        if (objectives[obj].output_type == "dft") {
                            auto dft_data = objectives[obj].sensor_solution_dft;
                            numtimes = dft_data.extent(3);
                            int numsols = dft_data.extent_int(1);
                            int numdims = dft_data.extent_int(2);
                            size_type numfreq = dft_data.extent(3);
                            numfields = numsols * numdims;
                            sensor_data = Kokkos::View<ScalarT ***, HostDevice>("sensor data", numsensors, numfields, numfreq);
                            for (size_t t = 0; t < numfreq; ++t) {
                                for (int sens = 0; sens < numsensors; ++sens) {
                                    size_t solprog = 0;
                                    for (int sol = 0; sol < numsols; ++sol) {
                                        for (int d = 0; d < numdims; ++d) {
                                            sensor_data(sens, solprog, t) = dft_data(sens, sol, d, t).real();
                                            solprog++;
                                        }
                                    }
                                }
                            }
                        }
                        else if (objectives[obj].output_type == "integrated dft")
                        {
                            
                            auto dft_data = objectives[obj].sensor_solution_dft;
                            numtimes = dft_data.extent(3);
                            
                            // Grab some parameters from settings
                            int numtheta = settings->sublist("Postprocess").sublist("NF2FF").get("number theta", 1);
                            ScalarT mintheta = settings->sublist("Postprocess").sublist("NF2FF").get("min theta", 0.0);
                            ScalarT maxtheta = settings->sublist("Postprocess").sublist("NF2FF").get("max theta", 0.0);
                            int numphi = settings->sublist("Postprocess").sublist("NF2FF").get("number phi", 1);
                            ScalarT minphi = settings->sublist("Postprocess").sublist("NF2FF").get("min phi", 0.0);
                            ScalarT maxphi = settings->sublist("Postprocess").sublist("NF2FF").get("max phi", 0.0);
							
							TEUCHOS_TEST_FOR_EXCEPTION(!nf2ff.constants_initialized, std::runtime_error,
													   "NF2FF constants were not initialized. "
													   "Define c0 and eta0 in the input Functions file "
													   "and enable NF2FF accumulation before reporting.");
							const ScalarT c0 = nf2ff.c0;
							const ScalarT eta0 = nf2ff.eta0;
                            
                            // Create the vectors of PHI and THETA
                            vector<ScalarT> THETA(numtheta), PHI(numphi);
                            if (numtheta>1) {
                                ScalarT dtheta = (maxtheta-mintheta)/(numtheta-1);
                                for (size_t k=0; k<numtheta; ++k) {
                                    THETA[k] = (mintheta + k*dtheta) * PI / 180;
                                }
                            }
                            else {
                                THETA[0] = mintheta * PI / 180;
                            }
                            
                            if (numphi>1) {
                                ScalarT dphi = (maxphi-minphi)/(numphi-1);
                                for (size_t k=0; k<numphi; ++k) {
                                    PHI[k] = (minphi + k*dphi) * PI / 180;
                                }
                            }
                            else {
                                PHI[0] = minphi * PI / 180;
                            }
                            
                            const int numfreq = dft_data.extent(3);
                            // Initialize vector potentials
                            Kokkos::View<std::complex<ScalarT>***,HostDevice> A_th = Kokkos::View<std::complex<ScalarT>***,HostDevice>("NF2FF A_th", numfreq, numtheta, numphi);
                            Kokkos::View<std::complex<ScalarT>***,HostDevice> A_ph = Kokkos::View<std::complex<ScalarT>***,HostDevice>("NF2FF A_ph", numfreq, numtheta, numphi);
                            Kokkos::View<std::complex<ScalarT>***,HostDevice> F_th = Kokkos::View<std::complex<ScalarT>***,HostDevice>("NF2FF F_th", numfreq, numtheta, numphi);
                            Kokkos::View<std::complex<ScalarT>***,HostDevice> F_ph = Kokkos::View<std::complex<ScalarT>***,HostDevice>("NF2FF F_ph", numfreq, numtheta, numphi);
                            
                            //vector<std::complex<ScalarT>> Prad(numfreq, std::complex<ScalarT>(0.0, 0.0)); // radiated power
                            Kokkos::View<std::complex<ScalarT>*,HostDevice> Prad = Kokkos::View<std::complex<ScalarT>*,HostDevice>("NF2FF Prad", numfreq);
                            
                            // Need to make a few assumptions here:
                            // 1. The sensors are the boundary quadrature points on a given block
                            // 2. They are in the same order as looping through the boundary groups and quadrature points
                            // 3. A given block has all of its boundary quadrature points as sensors
                            // 4. The data stored in dft_data is the dft of E at the quadrature points
                            // 5. The problem is 3D, not 1D or 2D
                            
                            size_t iblock = objectives[obj].block;
                            string sidename = objectives[obj].sideset;
                            int prog = 0; // increments sensors
                            for (size_t t=0; t<numfreq; ++t) {
                                
                                ScalarT freq = objectives[obj].dft_frequencies[t];
                                ScalarT k0 = 2.0 * PI * freq / c0;
                                
                                //for (int sens = 0; sens<dft_data.extent(0); ++sens) {
                                //vector<std::complex<ScalarT>> Esrc = {dft_data(sens,0,0,t), dft_data(sens,0,1,t), dft_data(sens,0,2,t)};
                                
                                for (int nt=0; nt<numtheta; ++nt) {
                                    for (int np=0; np<numphi; ++np) {
                                        
                                        prog = 0;
                                        
                                        for (size_t grp=0; grp<assembler->boundary_groups[iblock].size(); ++grp) {
                                            
                                            if (assembler->boundary_groups[iblock][grp]->sidename == sidename) {
                                                // These arrays live on the AssemblyDevice, which may not be the Host, i.e., will not run on GPU without data transfer
                                                vector<View_Sc2> ip = assembler->boundary_groups[iblock][grp]->ip;
                                                vector<View_Sc2> normals = assembler->boundary_groups[iblock][grp]->normals;
                                                View_Sc2 wts = assembler->boundary_groups[iblock][grp]->wts;
                                                
                                                // Cartesian to spherical transform
                                                vector<ScalarT> r_hat = {std::sin(THETA[nt])*std::cos(PHI[np]), std::sin(THETA[nt])*std::sin(PHI[np]), std::cos(THETA[nt])};
                                                vector<ScalarT> theta_hat = {std::cos(THETA[nt])*std::cos(PHI[np]), std::cos(THETA[nt])*std::sin(PHI[np]), -std::sin(THETA[nt])};
                                                vector<ScalarT> phi_hat = {-std::sin(PHI[np]), std::cos(PHI[np]), 0.0};
                                                
                                                // Compute phase at quadrature points (assumes 3D)
                                                for (size_type elem=0; elem<wts.extent(0); ++elem) {
                                                    for (size_type pt=0; pt<wts.extent(1); ++pt) {
                                                        vector<std::complex<ScalarT>> Esrc = {dft_data(prog,0,0,t), dft_data(prog,0,1,t), dft_data(prog,0,2,t)};
                                                        vector<std::complex<ScalarT>> EsrcC = {std::conj(dft_data(prog,0,0,t)), std::conj(dft_data(prog,0,1,t)), std::conj(dft_data(prog,0,2,t))}; //conj(Esrc)
                                                        
                                                        // Compute Phase
                                                        ScalarT phase = k0*(ip[0](elem,pt)*r_hat[0] + ip[1](elem,pt)*r_hat[1] + ip[2](elem,pt)*r_hat[2]);
                                                        std::complex<ScalarT> phasor(std::cos(phase), -std::sin(phase));
                                                        
                                                        vector<std::complex<ScalarT>> E_theta = {theta_hat[0]*phasor, theta_hat[1]*phasor, theta_hat[2]*phasor};
                                                        vector<std::complex<ScalarT>> E_phi = {phi_hat[0]*phasor, phi_hat[1]*phasor, phi_hat[2]*phasor};
                                                        
                                                        // Compute normal x Escr, normal x conj(Escr), normal x E_theta, normal x E_phi
                                                        vector<std::complex<ScalarT>> n_x_Esrc = {normals[1](elem,pt)*Esrc[2] - normals[2](elem,pt)*Esrc[1], normals[2](elem,pt)*Esrc[0] - normals[0](elem,pt)*Esrc[2], normals[0](elem,pt)*Esrc[1] - normals[1](elem,pt)*Esrc[0]};
                                                        vector<std::complex<ScalarT>> n_x_EsrcC = {normals[1](elem,pt)*EsrcC[2] - normals[2](elem,pt)*EsrcC[1], normals[2](elem,pt)*EsrcC[0] - normals[0](elem,pt)*EsrcC[2], normals[0](elem,pt)*EsrcC[1] - normals[1](elem,pt)*EsrcC[0]};
                                                        vector<std::complex<ScalarT>> n_x_E_theta = {normals[1](elem,pt)*E_theta[2] - normals[2](elem,pt)*E_theta[1], normals[2](elem,pt)*E_theta[0] - normals[0](elem,pt)*E_theta[2], normals[0](elem,pt)*E_theta[1] - normals[1](elem,pt)*E_theta[0]};
                                                        vector<std::complex<ScalarT>> n_x_E_phi = {normals[1](elem,pt)*E_phi[2] - normals[2](elem,pt)*E_phi[1], normals[2](elem,pt)*E_phi[0] - normals[0](elem,pt)*E_phi[2], normals[0](elem,pt)*E_phi[1] - normals[1](elem,pt)*E_phi[0]};
                                                        
                                                        // Sum into total radiated power at ABC
                                                        //Prad = (1/eta0) * E * integral( dot( cross(normal,T) , cross(normal,T) ) ) * E';
                                                        if (nt==0 && np==0) {
                                                          Prad(t) += 1.0/eta0*wts(elem,pt)*(n_x_Esrc[0]*n_x_EsrcC[0] + n_x_Esrc[1]*n_x_EsrcC[1] + n_x_Esrc[2]*n_x_EsrcC[2]);
                                                        }
                                                        
                                                        // Sum into the vector potentials
                                                        A_th(t,nt,np) += -1.0/eta0*wts(elem,pt)*(n_x_E_theta[0]*n_x_Esrc[0] + n_x_E_theta[1]*n_x_Esrc[1] + n_x_E_theta[2]*n_x_Esrc[2]);
                                                        A_ph(t,nt,np) += -1.0/eta0*wts(elem,pt)*(n_x_E_phi[0]*n_x_Esrc[0] + n_x_E_phi[1]*n_x_Esrc[1] + n_x_E_phi[2]*n_x_Esrc[2]);
                                                        F_th(t,nt,np) += -wts(elem,pt)*(E_theta[0]*n_x_Esrc[0] + E_theta[1]*n_x_Esrc[1] + E_theta[2]*n_x_Esrc[2]);
                                                        F_ph(t,nt,np) += -wts(elem,pt)*(E_phi[0]*n_x_Esrc[0] + E_phi[1]*n_x_Esrc[1] + E_phi[2]*n_x_Esrc[2]);
                                                        prog++;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            
                            const int numentries = numfreq * numtheta * numphi;
                            
                            // The above calculation provides the processor-local contribution to the integrated quantity
                            // Need to sum over all processors to get the global values
                            Teuchos::Array<ScalarT> local_data_A_th_re(numentries, 0.0), global_data_A_th_re(numentries, 0.0);
                            Teuchos::Array<ScalarT> local_data_A_th_im(numentries, 0.0), global_data_A_th_im(numentries, 0.0);
                            
                            Teuchos::Array<ScalarT> local_data_A_ph_re(numentries, 0.0), global_data_A_ph_re(numentries, 0.0);
                            Teuchos::Array<ScalarT> local_data_A_ph_im(numentries, 0.0), global_data_A_ph_im(numentries, 0.0);
                            
                            Teuchos::Array<ScalarT> local_data_F_th_re(numentries, 0.0), global_data_F_th_re(numentries, 0.0);
                            Teuchos::Array<ScalarT> local_data_F_th_im(numentries, 0.0), global_data_F_th_im(numentries, 0.0);
                            
                            Teuchos::Array<ScalarT> local_data_F_ph_re(numentries, 0.0), global_data_F_ph_re(numentries, 0.0);
                            Teuchos::Array<ScalarT> local_data_F_ph_im(numentries, 0.0), global_data_F_ph_im(numentries, 0.0);
                            
                            Teuchos::Array<ScalarT> local_data_Prad_re(numfreq, 0.0), global_data_Prad_re(numfreq, 0.0);
                            Teuchos::Array<ScalarT> local_data_Prad_im(numfreq, 0.0), global_data_Prad_im(numfreq, 0.0);
                            
                            // Just taking the real component for now
                            prog = 0;
                            for (int t=0; t<numfreq; ++t) {
                                local_data_Prad_re[t] = Prad(t).real();
                                local_data_Prad_im[t] = Prad(t).imag();
                                for (int nt=0; nt<numtheta; ++nt) {
                                    for (int np=0; np<numphi; ++np) {
                                        local_data_A_th_re[prog] = A_th(t,nt,np).real();
                                        local_data_A_th_im[prog] = A_th(t,nt,np).imag();
                                        
                                        local_data_A_ph_re[prog] = A_ph(t,nt,np).real();
                                        local_data_A_ph_im[prog] = A_ph(t,nt,np).imag();
                                        
                                        local_data_F_th_re[prog] = F_th(t,nt,np).real();
                                        local_data_F_th_im[prog] = F_th(t,nt,np).imag();
                                        
                                        local_data_F_ph_re[prog] = F_ph(t,nt,np).real();
                                        local_data_F_ph_im[prog] = F_ph(t,nt,np).imag();
                                        ++prog;
                                    }
                                }
                            }
                            
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &local_data_A_th_re[0], &global_data_A_th_re[0]);
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &local_data_A_th_im[0], &global_data_A_th_im[0]);
                            
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &local_data_A_ph_re[0], &global_data_A_ph_re[0]);
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &local_data_A_ph_im[0], &global_data_A_ph_im[0]);
                            
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &local_data_F_th_re[0], &global_data_F_th_re[0]);
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &local_data_F_th_im[0], &global_data_F_th_im[0]);
                            
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &local_data_F_ph_re[0], &global_data_F_ph_re[0]);
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &local_data_F_ph_im[0], &global_data_F_ph_im[0]);
                            
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numfreq, &local_data_Prad_re[0], &global_data_Prad_re[0]);
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numfreq, &local_data_Prad_im[0], &global_data_Prad_im[0]);
                            
                            prog = 0;
                            for (int t=0; t<numfreq; ++t) {
                                Prad(t) = std::complex<ScalarT>(global_data_Prad_re[t], global_data_Prad_im[t]);
                                for (int nt=0; nt<numtheta; ++nt) {
                                    for (int np=0; np<numphi; ++np) {
                                        A_th(t,nt,np) = std::complex<ScalarT>(global_data_A_th_re[prog], global_data_A_th_im[prog]);
                                        A_ph(t,nt,np) = std::complex<ScalarT>(global_data_A_ph_re[prog], global_data_A_ph_im[prog]);
                                        F_th(t,nt,np) = std::complex<ScalarT>(global_data_F_th_re[prog], global_data_F_th_im[prog]);
                                        F_ph(t,nt,np) = std::complex<ScalarT>(global_data_F_ph_re[prog], global_data_F_ph_im[prog]);
                                        ++prog;
                                    }
                                }
                            }
							
                            Kokkos::View<std::complex<ScalarT>**, HostDevice> sdat("sensor data", numentries, 7);
                            
                            ScalarT r = 1.0; // distance into the far field
                            prog = 0;
                            // ScalarT Pinc = (EPx*EPx + EPy*EPy + EPz*EPz)/(2.0*eta0);
                            for (size_t t=0; t<numfreq; ++t) {
                                ScalarT freq = objectives[obj].dft_frequencies[t];
                                ScalarT k0 = 2.0 * PI * freq / c0;
                                std::complex<ScalarT> Piso = Prad[t]/(4*PI);
                                
                                for (int nt=0; nt<numtheta; ++nt) {
                                    for (int np=0; np<numphi; ++np) {
                                        
                                        // Electric Field
                                        sdat(prog,0) = k0/(4*PI*r)*std::abs(eta0*A_th(t,nt,np) + F_ph(t,nt,np)); //E_theta
                                        sdat(prog,1) = k0/(4*PI*r)*std::abs(eta0*A_ph(t,nt,np) - F_th(t,nt,np)); //E_phi
                                        
                                        // Radiated Power in r_hat
                                        sdat(prog,2) = std::pow(k0/(4*PI),2)*1.0/eta0*std::pow(std::abs(eta0*A_th(t,nt,np) + F_ph(t,nt,np)),2); //P_theta
                                        sdat(prog,3) = std::pow(k0/(4*PI),2)*1.0/eta0*std::pow(std::abs(eta0*A_ph(t,nt,np) - F_th(t,nt,np)),2); //P_phi
                                        
                                        // Gain
                                        sdat(prog,4) = sdat(prog,2)/Piso; //Gain_theta
                                        sdat(prog,5) = sdat(prog,3)/Piso; //Gain_phi
                                        sdat(prog,6) = sdat(prog,4) + sdat(prog,5); //Gain_total
										
                                        ++prog;
                                    }
                                }
                            }
                            
                            if (Comm->getRank() == 0) {
                                string respfile = "integrated_dft_calc.csv";
                                std::ofstream respOUT;
                                
                                bool is_open = false;
                                int attempts = 0;
                                int max_attempts = 100;
                                while (!is_open && attempts < max_attempts) {
                                    respOUT.open(respfile);
                                    is_open = respOUT.is_open();
                                    attempts++;
                                }
                                respOUT.precision(8);
                                
                                // Just writing the real component for now
                                prog = 0;
                                for (size_t t=0; t<numfreq; ++t) {
                                    for (int nt=0; nt<numtheta; ++nt) {
                                        for (int np=0; np<numphi; ++np) {
                                            respOUT << sdat(prog,0).real() << ",  ";
                                            respOUT << sdat(prog,0).imag() << ",  ";
                                            respOUT << sdat(prog,1).real() << ",  ";
                                            respOUT << sdat(prog,1).imag() << ",  ";
                                            respOUT << sdat(prog,2).real() << ",  ";
                                            respOUT << sdat(prog,2).imag() << ",  ";
                                            respOUT << sdat(prog,3).real() << ",  ";
                                            respOUT << sdat(prog,3).imag() << ",  ";
                                            respOUT << sdat(prog,4).real() << ",  ";
                                            respOUT << sdat(prog,4).imag() << ",  ";
                                            respOUT << sdat(prog,5).real() << ",  ";
                                            respOUT << sdat(prog,5).imag() << ",  ";
                                            respOUT << sdat(prog,6).real() << ",  ";
                                            respOUT << sdat(prog,6).imag() << ",  ";
                                            respOUT << endl;
                                            ++prog;
                                        }
                                    }
                                }
                                respOUT.close();
                            }
                        }
                        else
                        {
                            numtimes = objectives[obj].sensor_solution_data.size();              // vector of Kokkos::Views
                            int numsols = objectives[obj].sensor_solution_data[0].extent_int(1); // does assume this does not change in time, which it shouldn't
                            int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
                            numfields = numsols * numdims;
                            sensor_data = Kokkos::View<ScalarT ***, HostDevice>("sensor data", numsensors, numfields, numtimes);
                            for (size_t t = 0; t < numtimes; ++t)
                            {
                                auto sdat = objectives[obj].sensor_solution_data[t];
                                for (int sens = 0; sens < numsensors; ++sens)
                                {
                                    size_t solprog = 0;
                                    for (int sol = 0; sol < numsols; ++sol)
                                    {
                                        for (int d = 0; d < numdims; ++d)
                                        {
                                            sensor_data(sens, solprog, t) = sdat(sens, sol, d);
                                            solprog++;
                                        }
                                    }
                                }
                            }
                            if (objectives[obj].output_type == "fft")
                            {
#if defined(MrHyDE_ENABLE_FFTW)
                                fft->compute(sensor_data, sensorIDs, global_num_sensors);
#endif
                            }
                        }
                    }
                    
                    size_t max_numtimes = 0;
                    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &numtimes, &max_numtimes);
                    
                    int max_numfields = 0;
                    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &numfields, &max_numfields);
                    
                    if (fileoutput == "text" && objectives[obj].output_type != "integrated dft") {
                        
                        for (int field = 0; field < max_numfields; ++field) {
                            std::stringstream ss;
                            size_t blocknum = objectives[obj].block;
                            ss << field;
                            string respfile = "sensor_solution_field." + ss.str() + "." + blocknames[blocknum] + ".out";
                            std::ofstream respOUT;
                            if (Comm->getRank() == 0) {
                                bool is_open = false;
                                int attempts = 0;
                                int max_attempts = 100;
                                while (!is_open && attempts < max_attempts) {
                                    respOUT.open(respfile);
                                    is_open = respOUT.is_open();
                                    attempts++;
                                }
                                respOUT.precision(8);
                                Teuchos::Array<ScalarT> time_data(max_numtimes + dimension, 0.0);
                                for (int dim = 0; dim < dimension; ++dim) {
                                    time_data[dim] = 0.0;
                                }
                                
                                for (size_t tt = 0; tt < max_numtimes; ++tt) {
                                    time_data[tt + dimension] = objectives[obj].response_times[tt];
                                }
                                
                                for (size_t tt = 0; tt < max_numtimes + dimension; ++tt) {
                                    respOUT << time_data[tt] << "  ";
                                }
                                respOUT << endl;
                            }
                            
                            auto spts = objectives[obj].sensor_points;
                            for (size_t ss = 0; ss < objectives[obj].sensor_found.size(); ++ss) {
                                Teuchos::Array<ScalarT> series_data(max_numtimes + dimension, 0.0);
                                Teuchos::Array<ScalarT> gseries_data(max_numtimes + dimension, 0.0);
                                if (objectives[obj].sensor_found[ss]) {
                                    size_t sindex = 0;
                                    for (size_t j = 0; j < ss; ++j) {
                                        if (objectives[obj].sensor_found(j)) {
                                            sindex++;
                                        }
                                    }
                                    for (int dim = 0; dim < dimension; ++dim) {
                                        series_data[dim] = spts(sindex, dim);
                                    }
                                    
                                    for (size_t tt = 0; tt < max_numtimes; ++tt) {
                                        series_data[tt + dimension] = sensor_data(sindex, field, tt);
                                    }
                                }
                                
                                const int numentries = max_numtimes + dimension;
                                
                                Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &series_data[0], &gseries_data[0]);
                                
                                if (Comm->getRank() == 0) {
                                    // respOUT << gseries_data[0] << "  " << gseries_data[1] << "  " << gseries_data[2] << "  ";
                                    for (size_t tt = 0; tt < max_numtimes + dimension; ++tt) {
                                        respOUT << gseries_data[tt] << "  ";
                                    }
                                    respOUT << endl;
                                }
                            }
                            if (Comm->getRank() == 0) {
                                respOUT.close();
                            }
                        }
                    }
#ifdef MrHyDE_USE_HDF5
                    else if (fileoutput == "hdf5") {
                        // PHDF5 creation
                        size_t num_snaps = max_numtimes;
                        const size_t alength = num_snaps;
                        ScalarT *myData = new ScalarT[alength];
                        
                        for (int field = 0; field < max_numfields; ++field) {
                            
                            herr_t err; // HDF5 return value
                            hid_t f_id; // HDF5 file ID
                            
                            // file access property list
                            hid_t fapl_id;
                            fapl_id = H5Pcreate(H5P_FILE_ACCESS);
                            
                            err = H5Pset_fapl_mpio(fapl_id, *(Comm->getRawMpiComm()), MPI_INFO_NULL);
                            
                            // create the file
                            std::stringstream ss;
                            ss << field;
                            string respfile = "sensor_solution_field." + ss.str() + ".h5";
                            
                            f_id = H5Fcreate(respfile.c_str(), H5F_ACC_TRUNC, // overwrites file if it exists
                                             H5P_DEFAULT, fapl_id);
                            
                            // free the file access template
                            err = H5Pclose(fapl_id);
                            
                            // create the dataspace
                            
                            hid_t ds_id;
                            hsize_t dims[2] = {objectives[obj].sensor_found.size(), num_snaps};
                            ds_id = H5Screate_simple(2, dims, // [sensor_id,snap]
                                                     NULL);
                            
                            // need to create a new hdf5 datatype which matches fftw_complex
                            // TODO not sure about this...
                            // TODO change ??
                            hsize_t comp_dims[1] = {1};
                            hid_t complex_id = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, comp_dims);
                            
                            // create the storage
                            hid_t field_id;
                            field_id = H5Dcreate2(f_id, "soln", complex_id, ds_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                            
                            // set up the portion of the files this process will access
                            for (size_type sens = 0; sens < sensorIDs.extent(0); ++sens) {
                                hsize_t myID = sensorIDs(sens);
                                hsize_t start[2] = {myID, 0};
                                hsize_t count[2] = {1, num_snaps};
                                
                                err = H5Sselect_hyperslab(ds_id, H5S_SELECT_SET, start, NULL, // contiguous
                                                          count, NULL);                       // contiguous
                                
                                for (size_t s = 0; s < num_snaps; ++s) {
                                    myData[s] = sensor_data(sensorIDs(sens), field, s);
                                }
                                hsize_t flattened[] = {num_snaps};
                                hid_t ms_id = H5Screate_simple(1, flattened, NULL);
                                err = H5Dwrite(field_id, complex_id, ms_id, ds_id, H5P_DEFAULT, myData);
                            }
                            
                            err = H5Dclose(field_id);
                            
                            if (err > 0) {
                                // say something
                            }
                            H5Sclose(ds_id);
                            H5Fclose(f_id);
                        }
                        delete[] myData;
                    }
#endif // MrHyDE_USE_HDF5
                }
                else { // Second case: sensors computed response functions
                    string respfile = objectives[obj].response_file + "." + blocknames[objectives[obj].block] + ".out";
                    std::ofstream respOUT;
                    if (Comm->getRank() == 0) {
                        bool is_open = false;
                        int attempts = 0;
                        int max_attempts = 100;
                        while (!is_open && attempts < max_attempts) {
                            respOUT.open(respfile);
                            is_open = respOUT.is_open();
                            attempts++;
                        }
                        respOUT.precision(16);
                    }
                    
                    if (Comm->getRank() == 0) {
                        for (size_t tt = 0; tt < objectives[obj].response_times.size(); ++tt) {
                            respOUT << objectives[obj].response_times[tt] << "  ";
                        }
                        respOUT << endl;
                    }
                    for (size_t ss = 0; ss < objectives[obj].sensor_found.size(); ++ss) {
                        for (size_t tt = 0; tt < objectives[obj].response_times.size(); ++tt) {
                            ScalarT sslval = 0.0, ssgval = 0.0;
                            if (objectives[obj].sensor_found[ss]) {
                                size_t sindex = 0;
                                for (size_t j = 0; j < ss; ++j) {
                                    if (objectives[obj].sensor_found(j)) {
                                        sindex++;
                                    }
                                }
                                
                                sslval = objectives[obj].response_data[tt](sindex);
                            }
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &sslval, &ssgval);
                            if (Comm->getRank() == 0) {
                                respOUT << ssgval << "  ";
                            }
                        }
                        if (Comm->getRank() == 0) {
                            respOUT << endl;
                        }
                    }
                    if (Comm->getRank() == 0) {
                        respOUT.close();
                    }
                }
            }
            else if (objectives[obj].type == "integrated response") {
                if (objectives[obj].save_data) {
                    string respfile = objectives[obj].response_file + "." + blocknames[objectives[obj].block] + append + ".out";
                    std::ofstream respOUT;
                    if (Comm->getRank() == 0) {
                        bool is_open = false;
                        int attempts = 0;
                        int max_attempts = 100;
                        while (!is_open && attempts < max_attempts) {
                            respOUT.open(respfile);
                            is_open = respOUT.is_open();
                            attempts++;
                        }
                        respOUT.precision(16);
                    }
                    for (size_t tt = 0; tt < objectives[obj].response_times.size(); ++tt) {
                        
                        if (Comm->getRank() == 0) {
                            respOUT << objectives[obj].response_times[tt] << "  ";
                        }
                        double localval = objectives[obj].scalar_response_data[tt];
                        double globalval = 0.0;
                        Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &localval, &globalval);
                        if (Comm->getRank() == 0) {
                            respOUT << globalval;
                            respOUT << endl;
                        }
                    }
                    if (Comm->getRank() == 0) {
                        respOUT.close();
                    }
                }
            }
        }
    }
    
    if (compute_flux_response)
    {
        if (Comm->getRank() == 0)
        {
            if (verbosity > 0)
            {
                cout << endl
                << "*********************************************************" << endl;
                cout << "***** Computing Flux Responses ******" << endl;
                cout << "*********************************************************" << endl;
            }
        }
        
        vector<ScalarT> gvals;
        
        for (size_t f = 0; f < fluxes.size(); ++f)
        {
            for (size_t tt = 0; tt < fluxes[f].vals.extent(0); ++tt)
            {
                ScalarT lval = fluxes[f].vals(tt);
                ScalarT gval = 0.0;
                Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lval, &gval);
                gvals.push_back(gval);
            }
            Kokkos::deep_copy(fluxes[f].vals, 0.0);
        }
        if (Comm->getRank() == 0)
        {
            string respfile = "flux_response.out";
            std::ofstream respOUT;
            if (Comm->getRank() == 0)
            {
                bool is_open = false;
                int attempts = 0;
                int max_attempts = 100;
                while (!is_open && attempts < max_attempts)
                {
                    respOUT.open(respfile, std::ios_base::app);
                    is_open = respOUT.is_open();
                    attempts++;
                }
                respOUT.precision(16);
            }
            
            for (size_t g = 0; g < gvals.size(); ++g)
            {
                cout << gvals[g] << endl;
                
                respOUT << " " << gvals[g] << "  ";
            }
            respOUT << endl;
            respOUT.close();
        }
    }
    
    if (compute_integrated_quantities)
    {
        if (Comm->getRank() == 0)
        {
            if (verbosity > 0)
            {
                cout << endl
                << "*********************************************************" << endl;
                cout << "****** Storing Integrated Quantities ******" << endl;
                cout << "*********************************************************" << endl;
            }
        }
        
        for (size_t iLocal = 0; iLocal < integratedQuantities.size(); iLocal++)
        {
            
            // iLocal indexes over the number of blocks where IQs are defined and
            // does not necessarily match the global block ID
            
            size_t globalBlock = integratedQuantities[iLocal][0].block; // all IQs with same first index share a block
            
            if (Comm->getRank() == 0)
            {
                cout << endl
                << "*********************************************************" << endl;
                cout << "****** Integrated Quantities on block : " << blocknames[globalBlock] << " ******" << endl;
                cout << "*********************************************************" << endl;
                for (size_t k = 0; k < integratedQuantities[iLocal].size(); ++k)
                {
                    std::cout << integratedQuantities[iLocal][k].name << " : "
                    << integratedQuantities[iLocal][k].val(0) << std::endl;
                }
            }
            
        } // end loop over blocks with IQs requested
        // TODO output something? Make the first print statement true!
        // BWR -- this only happens at end of sim.
    } // end if compute_integrated_quantities
    
    ////////////////////////////////////////////////////////////////////////////
    // Report the errors for verification tests
    ////////////////////////////////////////////////////////////////////////////
    
    if (compute_error)
    {
        if (Comm->getRank() == 0)
        {
            cout << endl
            << "*********************************************************" << endl;
            cout << "***** Computing errors ******" << endl
            << endl;
        }
        
        for (size_t block = 0; block < assembler->groups.size(); block++)
        { // loop over blocks
            for (size_t etype = 0; etype < error_list[block].size(); etype++)
            {
                
                // for (size_t et=0; et<error_types.size(); et++){
                for (size_t time = 0; time < error_times.size(); time++)
                {
                    // for (int n=0; n<numVars[block]; n++) {
                    
                    ScalarT lerr = errors[time][block](etype);
                    ScalarT gerr = 0.0;
                    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lerr, &gerr);
                    if (Comm->getRank() == 0)
                    {
                        string varname = error_list[block][etype].first;
                        if (error_list[block][etype].second == "L2" || error_list[block][etype].second == "L2 VECTOR")
                        {
                            cout << "***** L2 norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                        }
                        else if (error_list[block][etype].second == "L2 FACE")
                        {
                            cout << "***** L2-face norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                        }
                        else if (error_list[block][etype].second == "GRAD")
                        {
                            cout << "***** L2 norm of the error for grad(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                        }
                        else if (error_list[block][etype].second == "DIV")
                        {
                            cout << "***** L2 norm of the error for div(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                        }
                        else if (error_list[block][etype].second == "CURL")
                        {
                            cout << "***** L2 norm of the error for curl(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                        }
                    }
                    //}
                }
            }
        }
        
        // Error in subgrid models
        if (!(Teuchos::is_null(multiscale_manager)))
        {
            if (multiscale_manager->getNumberSubgridModels() > 0)
            {
                
                for (size_t m = 0; m < multiscale_manager->getNumberSubgridModels(); m++)
                {
                    vector<string> sgvars = multiscale_manager->subgridModels[m]->varlist;
                    vector<std::pair<string, string>> sg_error_list;
                    // A given processor may not have any elements that use this subgrid model
                    // In this case, nothing gets initialized so sgvars.size() == 0
                    // Find the global max number of sgvars over all processors
                    size_t nvars = sgvars.size();
                    if (nvars > 0)
                    {
                        sg_error_list = multiscale_manager->subgridModels[m]->getErrorList();
                    }
                    // really only works on one block
                    size_t nerrs = sg_error_list.size();
                    size_t gnerrs = 0;
                    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &nerrs, &gnerrs);
                    
                    for (size_t etype = 0; etype < gnerrs; etype++)
                    {
                        for (size_t time = 0; time < error_times.size(); time++)
                        {
                            // Get the local contribution (if processor uses subgrid model)
                            ScalarT lerr = 0.0;
                            if (subgrid_errors[time][0][m].extent(0) > 0)
                            {
                                lerr = subgrid_errors[time][0][m](etype); // block is not relevant
                            }
                            ScalarT gerr = 0.0;
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &lerr, &gerr);
                            
                            // Figure out who can print the information (lowest rank amongst procs using subgrid model)
                            int myID = Comm->getRank();
                            if (nvars == 0)
                            {
                                myID = 100000000;
                            }
                            int gID = 0;
                            Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MIN, 1, &myID, &gID);
                            
                            if (Comm->getRank() == gID)
                            {
                                
                                string varname = sg_error_list[etype].first;
                                if (sg_error_list[etype].second == "L2" || sg_error_list[etype].second == "L2 VECTOR")
                                {
                                    cout << "***** Subgrid " << m << ": L2 norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                                }
                                else if (sg_error_list[etype].second == "L2 FACE")
                                {
                                    cout << "***** Subgrid " << m << ": L2-face norm of the error for " << varname << " = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                                }
                                else if (sg_error_list[etype].second == "GRAD")
                                {
                                    cout << "***** Subgrid " << m << ": L2 norm of the error for grad(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                                }
                                else if (sg_error_list[etype].second == "DIV")
                                {
                                    cout << "***** Subgrid " << m << ": L2 norm of the error for div(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                                }
                                else if (sg_error_list[etype].second == "CURL")
                                {
                                    cout << "***** Subgrid " << m << ": L2 norm of the error for curl(" << varname << ") = " << sqrt(gerr) << "  (time = " << error_times[time] << ")" << endl;
                                }
                            }
                            //}
                        }
                    }
                }
            }
        }
    }
    
    if (compute_weighted_norm)
    {
        if (Comm->getRank() == 0)
        {
            string respfile = "weighted_norms.out";
            std::ofstream respOUT;
            respOUT.open(respfile);
            for (size_t k = 0; k < weighted_norms.size(); ++k)
            {
                respOUT << weighted_norms[k] << endl;
            }
            respOUT << endl;
            respOUT.close();
        }
    }

    if (nf2ff.save) {
        this->writeNF2FF();
    }
    if (lumped_port_parameters.save) {
        this->writeLumpedPortParameters();
    }
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::computeWeightedNorm(vector<vector_RCP> &current_soln)
{
    
    debugger->print(1, "**** Starting PostprocessManager::computeWeightedNorm()");
    
    Teuchos::TimeMonitor localtimer(*computeWeightedNormTimer);
    
    typedef typename Node::execution_space LA_exec;
    typedef typename Node::device_type LA_device;
    
    if (!have_norm_weights)
    {
        for (size_t set = 0; set < current_soln.size(); ++set)
        {
            vector_RCP wts_over = linalg->getNewOverlappedVector(set);
            assembler->getWeightVector(set, wts_over);
            vector_RCP set_norm_wts = linalg->getNewVector(set);
            set_norm_wts->putScalar(0.0);
            set_norm_wts->doExport(*wts_over, *(linalg->exporter[set]), Tpetra::REPLACE);
            norm_wts.push_back(set_norm_wts);
        }
        have_norm_weights = true;
    }
    
    ScalarT totalnorm = 0.0;
    for (size_t set = 0; set < current_soln.size(); ++set)
    {
        // current_soln is an overlapped vector ... we want
        vector_RCP soln = linalg->getNewVector(set);
        soln->putScalar(0.0);
        soln->doExport(*(current_soln[set]), *(linalg->exporter[set]), Tpetra::REPLACE);
        
        vector_RCP prod = linalg->getNewVector(set);
        
        auto wts_view = norm_wts[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
        auto prod_view = prod->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
        auto soln_view = soln->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
        parallel_for("assembly insert Jac", RangePolicy<LA_exec>(0, prod_view.extent(0)), MRHYDE_LAMBDA(const int k) { prod_view(k, 0) = wts_view(k, 0) * soln_view(k, 0) * soln_view(k, 0); });
        
        Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> l2norm(1);
        prod->norm2(l2norm);
        totalnorm += l2norm[0];
    }
    
    weighted_norms.push_back(totalnorm);
    
    if (verbosity >= 10 && Comm->getRank() == 0)
    {
        cout << "Weighted norm of solution: " << totalnorm << endl;
    }
    
    debugger->print(1, "**** Finished PostprocessManager::computeWeightedNorm()");
}


// ========================================================================================
// ========================================================================================

template <class Node>
ScalarT PostprocessManager<Node>::makeSomeNoise(ScalarT stdev)
{
    // generate sample from 0-centered normal with stdev
    // Box-Muller method
    // srand(time(0)); //doing this more frequently than once-per-second results in getting the same numbers...
    ScalarT U1 = rand() / ScalarT(RAND_MAX);
    ScalarT U2 = rand() / ScalarT(RAND_MAX);
    
    return stdev * sqrt(-2 * log(U1)) * cos(2 * PI * U2);
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::setTimeIndex(const int &cindex)
{
    time_index = cindex;
}

// ========================================================================================
// ========================================================================================

template <class Node>
Teuchos::Array<ScalarT> PostprocessManager<Node>::collectResponses()
{
    
    //
    // May be multiple objectives, which store the responses
    // Each objective can store different types of responses:
    //   1.  Integrated response: scalar over time (each proc stores own contribution)
    //   2.  Sensor response: scalar over time at each sensor (each proc stores only own sensor)
    //   3.  Sensor solution: state variable over time at each sensor (each proc stores only own sensor)
    //
    
    Teuchos::Array<ScalarT> globalarraydata;
    
    ////////////////////////////////
    // First, determine how many responses have been computed
    ////////////////////////////////
    
    int totalresp = 0;
    vector<int> response_sizes;
    for (size_t obj = 0; obj < objectives.size(); ++obj)
    {
        if (objectives[obj].type == "sensors")
        {
            int totalsens = objectives[obj].sensor_found.extent_int(0); // this is actually the global number of sensors
            if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln)
            {
                size_t numtimes = objectives[obj].sensor_solution_data.size();
                int numsols = objectives[obj].sensor_solution_data[0].extent_int(1);
                int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
                totalsens *= numtimes * numsols * numdims;
            }
            else
            {
                size_t numtimes = objectives[obj].response_times.size();
                totalsens *= numtimes;
            }
            response_sizes.push_back(totalsens);
            // totalresp += totalsens;
        }
        else if (objectives[obj].type == "integrated response")
        {
            response_sizes.push_back(objectives[obj].response_times.size());
            // totalresp += objectives[obj].response_times.size();
        }
    }
    
    for (size_t i = 0; i < response_sizes.size(); ++i)
    {
        totalresp += response_sizes[i];
    }
    
    if (totalresp > 0)
    {
        int glbresp = 0;
        Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &totalresp, &glbresp);
        Kokkos::View<ScalarT *, HostDevice> newresp("response", glbresp);
        Teuchos::Array<ScalarT> localarraydata(glbresp, 0.0);
        
        ////////////////////////////////
        // Next, we fill in the responses
        ////////////////////////////////
        
        size_t overallprog = 0;
        for (size_t obj = 0; obj < objectives.size(); ++obj)
        {
            if (objectives[obj].type == "sensors")
            {
                int numsensors = objectives[obj].numSensors;
                if (numsensors > 0)
                {
                    Kokkos::View<int *, HostDevice> sensorIDs("sensor IDs owned by proc", numsensors);
                    size_t sprog = 0;
                    auto sensor_found = objectives[obj].sensor_found;
                    for (size_type s = 0; s < sensor_found.extent(0); ++s)
                    {
                        if (sensor_found(s))
                        {
                            sensorIDs(sprog) = s;
                            ++sprog;
                        }
                    }
                    
                    if (objectives[obj].compute_sensor_soln || objectives[obj].compute_sensor_average_soln)
                    {
                        for (int sens = 0; sens < numsensors; ++sens)
                        {
                            size_t numtimes = objectives[obj].sensor_solution_data.size();
                            int numsols = objectives[obj].sensor_solution_data[0].extent_int(1);
                            int numdims = objectives[obj].sensor_solution_data[0].extent_int(2);
                            int sensnum = sensorIDs(sens);
                            size_t startind = overallprog + sensnum * (numsols * numdims * numtimes);
                            size_t cprog = 0;
                            for (size_t tt = 0; tt < numtimes; ++tt)
                            {
                                for (int ss = 0; ss < numsols; ++ss)
                                {
                                    for (int dd = 0; dd < numdims; ++dd)
                                    {
                                        localarraydata[startind + cprog] = objectives[obj].sensor_solution_data[tt](sens, ss, dd);
                                        cprog++;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int sens = 0; sens < numsensors; ++sens)
                        {
                            size_t numtimes = objectives[obj].response_data.size();
                            int sensnum = sensorIDs(sens);
                            size_t startind = overallprog + sensnum * (numtimes);
                            size_t cprog = 0;
                            for (size_t tt = 0; tt < numtimes; ++tt)
                            {
                                localarraydata[startind + cprog] = objectives[obj].response_data[tt](sens);
                                cprog++;
                            }
                        }
                    }
                }
            }
            else if (objectives[obj].type == "integrated response")
            {
                for (size_t tt = 0; tt < objectives[obj].response_times.size(); ++tt)
                {
                    localarraydata[overallprog + tt] = objectives[obj].scalar_response_data[tt];
                }
            }
            overallprog += response_sizes[obj];
        }
        
        globalarraydata = Teuchos::Array<ScalarT>(glbresp, 0.0);
        
        const int numentries = totalresp;
        Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, numentries, &localarraydata[0], &globalarraydata[0]);
    }
    
    return globalarraydata;
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::resetSolutions() {
    for (size_t set = 0; set < soln.size(); ++set) {
        soln[set]->reset();
    }
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::writeQuadratureData() {
    
    // Separate files for different element blocks - can change later if one file is better
    string ptsfile_base = settings->sublist("Postprocessing").get<string>("quadrature points file", "qpts");
    string wtsfile_base = settings->sublist("Postprocessing").get<string>("quadrature weights file", "qwts");
    
    for (size_t block=0; block<blocknames.size(); ++block) {
        string ptsfile = ptsfile_base + "." + blocknames[block] + ".dat";
        string wtsfile = wtsfile_base + "." + blocknames[block] + ".dat";
        
        // All processors gather their own data
        View_Sc2 qdata = assembler->getQuadratureData(blocknames[block]); // [pts wts]
        Teuchos::Array<ScalarT> data = Teuchos::Array<ScalarT>(qdata.extent(0), 0.0);
        View_Sc2 qdata0;
        
        if (qdata.extent(0) > 0) { // Not all procs will own elements on this block
            // One by one they send to processor 0, which writes the file
            for (size_t rank=0; rank<Comm->getSize(); ++rank) {
                if (rank == 0) {
                    qdata0 = qdata;
                }
                else {
                    const int host = 0;
                    Teuchos::Array<int> numdata(1,qdata.extent(0));
                    Teuchos::Array<int> numdata0(1,0);
                    if (Comm->getRank() == rank && rank>0) {
                        Teuchos::send(*Comm, 1, &numdata[0], host);
                    }
                    if (Comm->getRank() == 0 && rank>0) {
                        Teuchos::receive(*Comm, rank, 1, &numdata0[0]);
                        qdata0 = View_Sc2("Data on proc 0", numdata0[0], dimension+1);
                    }
                    
                    for (size_type k=0; k<qdata.extent(1); ++k) {
                        if (Comm->getRank() == rank && rank>0) {
                            for (size_type i=0; i<qdata.extent(0); ++i) {
                                data[i] = qdata(k,i);
                            }
                            Teuchos::send(*Comm, numdata[0], &data[0], 0);
                        }
                        if (Comm->getRank() == 0 && rank>0) {
                            Teuchos::Array<ScalarT> data0(numdata0[0], 0.0);
                            Teuchos::receive(*Comm, rank, numdata[0], &data0[0]);
                            for (size_type i=0; i<qdata.extent(0); ++i) {
                                qdata0(k,i) = data0[i];
                            }
                        }
                    }
                }
                if (Comm->getRank() == 0) {
                    std::ofstream ptsOUT(ptsfile.c_str());
                    ptsOUT.precision(12);
                    for (size_type i=0; i<qdata0.extent(0); ++i) {
                        for (size_type v=0; v<qdata0.extent(1)-1; ++v) {
                            ptsOUT << qdata0(i, v) << "  ";
                        }
                        ptsOUT << endl;
                    }
                    ptsOUT.close();
                    
                    std::ofstream wtsOUT(wtsfile.c_str());
                    wtsOUT.precision(12);
                    for (size_type i=0; i<qdata0.extent(0); ++i) {
                        wtsOUT << qdata0(i, qdata0.extent(1)-1) << endl;
                    }
                    wtsOUT.close();
                }
            }
            
        }
    }
}

// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::writeBoundaryQuadratureData() {
    
    // Separate files for different side sets (not element blocks) - can change later if one file is better
    string ptsfile_base = settings->sublist("Postprocessing").get<string>("boundary quadrature points file", "qpts");
    string wtsfile_base = settings->sublist("Postprocessing").get<string>("boundary quadrature weights file", "qwts");
    string nsfile_base = settings->sublist("Postprocessing").get<string>("boundary quadrature normals file", "qns");
    
    for (size_t block=0; block<blocknames.size(); ++block) {
        
        for (size_t side=0; side<sideSets.size(); ++side) {
            
            std::stringstream ptsfile_ss, wtsfile_ss, nsfile_ss;
            ptsfile_ss << ptsfile_base << "." << blocknames[block] << "." << sideSets[side] << "." << Comm->getRank() << ".dat";
            wtsfile_ss << wtsfile_base << "." << blocknames[block] << "." << sideSets[side] << "." << Comm->getRank() << ".dat";
            nsfile_ss << nsfile_base << "." << blocknames[block] << "." << sideSets[side] << "." << Comm->getRank() << ".dat";
            
            string ptsfile = ptsfile_ss.str();
            string wtsfile = wtsfile_ss.str();
            string nsfile = nsfile_ss.str();
            
            // All processors gather their own data
            View_Sc2 qdata = assembler->getboundaryQuadratureData(blocknames[block], sideSets[side]); // [pts wts normals]
            Teuchos::Array<ScalarT> data = Teuchos::Array<ScalarT>(qdata.extent(0), 0.0);
            View_Sc2 qdata0;
            
            if (qdata.extent(0) > 0) { // Not all procs will own elements on this block
                // One by one they send to processor 0, which writes the file
                for (size_t rank=0; rank<Comm->getSize(); ++rank) {
                    if (rank == 0) {
                        qdata0 = qdata;
                    }
                    else {
                        const int host = 0;
                        Teuchos::Array<int> numdata(1,qdata.extent(0));
                        Teuchos::Array<int> numdata0(1,0);
                        if (Comm->getRank() == rank && rank>0) {
                            Teuchos::send(*Comm, 1, &numdata[0], host);
                        }
                        if (Comm->getRank() == 0 && rank>0) {
                            Teuchos::receive(*Comm, rank, 1, &numdata0[0]);
                            qdata0 = View_Sc2("Data on proc 0", numdata0[0], 2*dimension+1); // [pts wts normals]
                        }
                        
                        for (size_type k=0; k<qdata.extent(1); ++k) {
                            if (Comm->getRank() == rank && rank>0) {
                                for (size_type i=0; i<qdata.extent(0); ++i) {
                                    data[i] = qdata(k,i);
                                }
                                Teuchos::send(*Comm, numdata[0], &data[0], 0);
                            }
                            if (Comm->getRank() == 0 && rank>0) {
                                Teuchos::Array<ScalarT> data0(numdata0[0], 0.0);
                                Teuchos::receive(*Comm, rank, numdata[0], &data0[0]);
                                for (size_type i=0; i<qdata.extent(0); ++i) {
                                    qdata0(k,i) = data0[i];
                                }
                            }
                        }
                    }
                    //if (Comm->getRank() == 0) {
                    std::ofstream ptsOUT(ptsfile.c_str());
                    ptsOUT.precision(12);
                    for (size_type i=0; i<qdata0.extent(0); ++i) {
                        for (size_type v=0; v<dimension; ++v) {
                            ptsOUT << qdata0(i, v) << "  ";
                        }
                        ptsOUT << endl;
                    }
                    ptsOUT.close();
                    
                    std::ofstream wtsOUT(wtsfile.c_str());
                    wtsOUT.precision(12);
                    for (size_type i=0; i<qdata0.extent(0); ++i) {
                        wtsOUT << qdata0(i, dimension) << endl;
                    }
                    wtsOUT.close();
                    
                    std::ofstream nsOUT(nsfile.c_str());
                    nsOUT.precision(12);
                    for (size_type i=0; i<qdata0.extent(0); ++i) {
                        for (size_type v=0; v<dimension; ++v) {
                            nsOUT << qdata0(i, v+dimension+1) << "  ";
                        }
                        nsOUT << endl;
                    }
                    nsOUT.close();
                    
                }
                //}
                
            }
        }
    }
}


// ========================================================================================
// ========================================================================================

template <class Node>
void PostprocessManager<Node>::setForwardStates(vector<vector<vector_RCP> > & fwd_states, vector<ScalarT> & times) {
    for (size_t set = 0; set < fwd_states.size(); ++set) {
        for (size_t t = 0; t < fwd_states[set].size(); ++t) {
            soln[set]->store(fwd_states[set][t], times[t], 0);
        }
    }
}

