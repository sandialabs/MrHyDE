/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ========================================================================================
/* transient adjoint driven by a Revolve (Algorithm 799) checkpointing schedule */
// ========================================================================================

template<class Node>
void SolverManager<Node>::checkpointedAdjointModel(MrHyDE_OptVector & gradient) {

  debugger->print("**** Starting SolverManager::checkpointedAdjointModel ...");

  Teuchos::TimeMonitor localtimer(*adjointtimer);

  const size_t set = 0;
  const size_t stage = 0;

  // Revolve needs a fixed step count and a state fully described by the vectors
  // we hand it.  Each case below breaks one of those, so refuse rather than
  // return a gradient that is quietly wrong.

  TEUCHOS_TEST_FOR_EXCEPTION(solver_type != "transient", std::runtime_error,
    "Error: checkpointed adjoints require a transient solve.");

  TEUCHOS_TEST_FOR_EXCEPTION(setnames.size() != 1, std::runtime_error,
    "Error: checkpointed adjoints only support a single physics set.");

  // A multistep method would need more than one previous state per reverse step.
  // Guard on maxnumsteps too: BDF2 with a BDF1 startup leaves numsteps at 1 while
  // maxnumsteps stays 2, and that is what sizes the history handed to the adjoint.
  TEUCHOS_TEST_FOR_EXCEPTION(numsteps[set] != 1 || maxnumsteps[set] != 1, std::runtime_error,
    "Error: checkpointed adjoints require a single-step time integrator (BDF1).");

  TEUCHOS_TEST_FOR_EXCEPTION(numstages[set] != 1 || maxnumstages[set] != 1, std::runtime_error,
    "Error: checkpointed adjoints require a single-stage time integrator.");

  // subgrid state lives outside the macro solution vector, so restoring a
  // checkpoint would not restore all of it
  TEUCHOS_TEST_FOR_EXCEPTION(multiscale_manager->subgridModels.size() > 0, std::runtime_error,
    "Error: checkpointed adjoints do not support subgrid models.");

  // Count steps the way transientSolver's loop does.  Rounding
  // (final_time-initial_time)/deltat disagrees with it whenever deltat does not
  // divide the interval, and the reverse sweep would then cover a range the
  // forward pass never took.
  int num_steps = 0;
  {
    const ScalarT timetol = final_time*1.0e-6;
    ScalarT step_time = initial_time;
    while (step_time < final_time - timetol) {
      step_time += deltat;
      ++num_steps;
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION(num_steps < 1, std::runtime_error,
    "Error: checkpointed adjoints need at least one time step.");

  TEUCHOS_TEST_FOR_EXCEPTION(num_checkpoints < 1, std::runtime_error,
    "Error: 'Analysis: number of checkpoints' must be at least 1.");

  // the point of the exercise: num_checkpoints states, not num_steps
  vector<vector_RCP> checkpoint_state(num_checkpoints);
  vector<ScalarT> checkpoint_time(num_checkpoints, initial_time);
  for (int i=0; i<num_checkpoints; ++i) {
    checkpoint_state[i] = linalg->getNewOverlappedVector(set);
  }

  params->sacadoizeParams(false);
  linalg->resetAllJacobian();

  // state at the schedule's current position, starting at u_0
  is_adjoint = false;
  vector<vector_RCP> sol_carried = this->setInitial();
  if (usestrongDBCs) {
    assembler->updatePhysicsSet(set);
    this->setDirichlet(set, sol_carried[set]);
  }
  current_time = initial_time;

  // takeForwardStep shifts this and leaves the state we came from in slot 0,
  // which is what the adjoint step needs as its previous solution
  vector<vector_RCP> step_history;
  for (int i=0; i<maxnumsteps[set]; ++i) {
    step_history.push_back(linalg->getNewOverlappedVector(set));
  }

  vector<vector_RCP> sol, sol_stage, phi, phi_prev, phi_stage;
  sol.push_back(linalg->getNewOverlappedVector(set));
  sol_stage.push_back(linalg->getNewOverlappedVector(set));
  phi.push_back(linalg->getNewOverlappedVector(set));
  phi_prev.push_back(linalg->getNewOverlappedVector(set));
  phi_stage.push_back(linalg->getNewOverlappedVector(set));

  // transient adjoints carry Jacobian-vector products from the step above
  if (previous_adjoints.size() == 0) {
    for (size_t i=0; i<numsteps[set]; ++i) {
      vector<vector_RCP> ivecs;
      for (size_t iset=0; iset<setnames.size(); ++iset) {
        vector_RCP tempvec = linalg->getNewVector(iset);
        tempvec->putScalar(0.0);
        ivecs.push_back(tempvec);
      }
      previous_adjoints.push_back(ivecs);
    }
  }
  else {
    for (size_t i=0; i<numsteps[set]; ++i) {
      for (size_t iset=0; iset<setnames.size(); ++iset) {
        previous_adjoints[i][iset]->putScalar(0.0);
      }
    }
  }

  // Revolve decides what to do; this loop carries it out

  Revolve schedule(num_steps, num_checkpoints);

  num_forward_solves = 0;
  is_final_time = false;
  bool done = false;
  long long guard = 0;
  const long long max_actions = 1000LL*(static_cast<long long>(num_steps) + num_checkpoints);

  while (!done) {

    TEUCHOS_TEST_FOR_EXCEPTION(guard++ > max_actions, std::runtime_error,
      "Error: the Revolve schedule did not terminate.");

    const int prev_position = schedule.getRangeStart();
    RevolveAction action = schedule.next();

    if (action == RevolveAction::Advance) {

      // recompute forward, storing nothing
      for (int k=prev_position+1; k<=schedule.getRangeStart(); ++k) {
        params->updateDynamicParams(k-1);
        assembler->updateTimeStep(k-1);
        int status = this->takeForwardStep(set, sol_carried, step_history, k-1);
        TEUCHOS_TEST_FOR_EXCEPTION(status != 0, std::runtime_error,
          "Error: a forward step failed during checkpointed recomputation. The "
          "schedule assumes a fixed time step, so a step cut cannot be recovered from.");
        current_time += deltat;
        ++num_forward_solves;
      }
    }
    else if (action == RevolveAction::Store) {

      // next() has already claimed the slot, so the count points one past it
      const int slot = schedule.getNumCheckpointsStored() - 1;
      checkpoint_state[slot]->assign(*(sol_carried[set]));
      checkpoint_time[slot] = current_time;
    }
    else if (action == RevolveAction::Restore) {

      const int slot = schedule.getNumCheckpointsStored() - 1;
      sol_carried[set]->assign(*(checkpoint_state[slot]));
      // use the recorded time, not initial_time + k*deltat: the objective is
      // evaluated here and the two round differently
      current_time = checkpoint_time[slot];
    }
    else if (action == RevolveAction::FirstReverse || action == RevolveAction::Reverse) {

      // a reversal at position range_start handles step k, entering with
      // u_{k-1} carried and current_time already at t_{k-1}
      const int k = schedule.getRangeStart() + 1;

      sol[set]->assign(*(sol_carried[set]));
      params->updateDynamicParams(k-1);
      assembler->updateTimeStep(k-1);
      int status = this->takeForwardStep(set, sol, step_history, k-1);
      TEUCHOS_TEST_FOR_EXCEPTION(status != 0, std::runtime_error,
        "Error: a forward step failed during checkpointed recomputation.");
      ++num_forward_solves;

      // one adjoint step, mirroring the stored-state path in transientSolver
      is_adjoint = true;
      is_final_time = (action == RevolveAction::FirstReverse);

      // Drop any cached Jacobian.  The stored-state path assembles only adjoint
      // Jacobians between resets, but this sweep alternates forward and adjoint
      // operators, so with "reuse Jacobian" on the adjoint solve would otherwise
      // reuse the forward matrix.
      linalg->resetJacobian();
      linalg->resetPrevJacobian();

      if (Comm->getRank() == 0 && verbosity > 1) {
        cout << endl << "**** Checkpointed adjoint step " << k
             << " (time " << current_time << ")" << endl;
      }

      phi_prev[set] = linalg->getNewOverlappedVector(set);
      phi_prev[set]->update(1.0, *(phi[0]), 0.0);

      params->updateDynamicParams(k-1);
      postproc->setTimeIndex(k);
      assembler->updateStage(stage, current_time, deltat);

      sol_stage[set]->assign(*(sol[set]));

      status = this->nonlinearSolver(set, stage, sol, sol_stage, step_history,
                                     phi, phi_stage, phi_prev);

      phi[set]->update(1.0, *(phi_stage[0]), 0.0);
      postproc->computeSensitivities(sol, sol_stage, step_history, phi,
                                     current_time, k, deltat, gradient);

      is_adjoint = false;
      linalg->resetJacobian();
      linalg->resetPrevJacobian();
    }
    else if (action == RevolveAction::Terminate) {
      done = true;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "Error: the Revolve schedule reported an inconsistent state (status "
        + std::to_string(schedule.getStatus()) + ").");
    }
  }

  is_final_time = false;

  if (Comm->getRank() == 0 && verbosity > 0) {
    const long extra = Revolve::minExtraForwardSteps(num_steps, num_checkpoints);

    // the trajectory should not be in full storage - that is the memory saving.
    // ask without indexing, since the container may be empty
    size_t states_in_storage = 0;
    vector<vector<ScalarT> > all_times = postproc->soln[set]->extractAllTimes();
    if (all_times.size() > 0) {
      states_in_storage = all_times[0].size();
    }

    cout << endl << "**** Checkpointed adjoint complete" << endl;
    cout << "****   time steps               : " << num_steps << endl;
    cout << "****   checkpoints stored       : " << num_checkpoints << endl;
    cout << "****   forward solves used      : " << num_forward_solves
         << "  (predicted " << num_steps + extra << ")" << endl;
    cout << "****   states in full storage   : " << states_in_storage
         << "  (full storage would hold " << num_steps + 1 << ")" << endl;
  }

  debugger->print("**** Finished SolverManager::checkpointedAdjointModel");

}

// ========================================================================================
/* transient adjoint with sketched windows on the checkpointing axis */
// ========================================================================================

template<class Node>
void SolverManager<Node>::bswAdjointModel(MrHyDE_OptVector & gradient) {

  debugger->print("**** Starting SolverManager::bswAdjointModel ...");

  Teuchos::TimeMonitor localtimer(*adjointtimer);

  const size_t set = 0;
  const size_t stage = 0;

  // same regime as the classic checkpointed adjoint, plus the window manager
  TEUCHOS_TEST_FOR_EXCEPTION(solver_type != "transient", std::runtime_error,
    "Error: sketched checkpointing requires a transient solve.");
  TEUCHOS_TEST_FOR_EXCEPTION(setnames.size() != 1, std::runtime_error,
    "Error: sketched checkpointing only supports a single physics set.");
  TEUCHOS_TEST_FOR_EXCEPTION(numsteps[set] != 1 || maxnumsteps[set] != 1, std::runtime_error,
    "Error: sketched checkpointing requires a single-step time integrator (BDF1).");
  TEUCHOS_TEST_FOR_EXCEPTION(numstages[set] != 1 || maxnumstages[set] != 1, std::runtime_error,
    "Error: sketched checkpointing requires a single-stage time integrator.");
  TEUCHOS_TEST_FOR_EXCEPTION(multiscale_manager->subgridModels.size() > 0, std::runtime_error,
    "Error: sketched checkpointing does not support subgrid models.");
  TEUCHOS_TEST_FOR_EXCEPTION(bsw_manager.is_null(), std::runtime_error,
    "Error: the window manager was never constructed.");

  int num_steps = 0;
  {
    const ScalarT timetol = final_time*1.0e-6;
    ScalarT step_time = initial_time;
    while (step_time < final_time - timetol) {
      step_time += deltat;
      ++num_steps;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(num_steps < 1, std::runtime_error,
    "Error: sketched checkpointing needs at least one time step.");

  // windows the forward sweep committed, and the slot flow-back
  std::vector<std::pair<int,int> > window_spans;
  for (size_t i=0; i<bsw_manager->effectiveWindows().size(); ++i) {
    window_spans.push_back(std::make_pair(bsw_manager->effectiveWindows()[i].a,
                                          bsw_manager->effectiveWindows()[i].b_eff));
  }
  const int num_slots = bsw_manager->numCheckpointSlots();

  vector<vector_RCP> checkpoint_state(num_slots);
  vector<ScalarT> checkpoint_time(num_slots, initial_time);
  for (int i=0; i<num_slots; ++i) {
    checkpoint_state[i] = linalg->getNewOverlappedVector(set);
  }

  params->sacadoizeParams(false);
  linalg->resetAllJacobian();

  is_adjoint = false;
  vector<vector_RCP> sol_carried = this->setInitial();
  if (usestrongDBCs) {
    assembler->updatePhysicsSet(set);
    this->setDirichlet(set, sol_carried[set]);
  }
  current_time = initial_time;

  vector<vector_RCP> step_history;
  for (int i=0; i<maxnumsteps[set]; ++i) {
    step_history.push_back(linalg->getNewOverlappedVector(set));
  }

  vector<vector_RCP> sol, sol_stage, phi, phi_prev, phi_stage;
  sol.push_back(linalg->getNewOverlappedVector(set));
  sol_stage.push_back(linalg->getNewOverlappedVector(set));
  phi.push_back(linalg->getNewOverlappedVector(set));
  phi_prev.push_back(linalg->getNewOverlappedVector(set));
  phi_stage.push_back(linalg->getNewOverlappedVector(set));

  if (previous_adjoints.size() == 0) {
    for (size_t i=0; i<numsteps[set]; ++i) {
      vector<vector_RCP> ivecs;
      for (size_t iset=0; iset<setnames.size(); ++iset) {
        vector_RCP tempvec = linalg->getNewVector(iset);
        tempvec->putScalar(0.0);
        ivecs.push_back(tempvec);
      }
      previous_adjoints.push_back(ivecs);
    }
  }
  else {
    for (size_t i=0; i<numsteps[set]; ++i) {
      for (size_t iset=0; iset<setnames.size(); ++iset) {
        previous_adjoints[i][iset]->putScalar(0.0);
      }
    }
  }

  // raw-column seam between the windows and the linear algebra
  std::vector<double> raw_state, raw_prev;
  auto loadFromRaw = [&](vector_RCP & vec, const std::vector<double> & raw) {
    auto view = vec->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    for (size_t i=0; i<view.extent(0); ++i) {
      view(i,0) = raw[i];
    }
  };
  {
    auto view = sol_carried[set]->template getLocalView<LA_device>(Tpetra::Access::ReadOnly);
    raw_state.resize(view.extent(0));
    raw_prev.resize(view.extent(0));
  }

  // one adjoint step at real step k: sol holds u_k, step_history[0] holds
  // u_{k-1}, current_time is t_{k-1}.  Identical to the classic path.
  auto adjointStepAt = [&](const int & k, const bool & terminal) {
    is_adjoint = true;
    is_final_time = terminal;
    linalg->resetJacobian();
    linalg->resetPrevJacobian();

    phi_prev[set] = linalg->getNewOverlappedVector(set);
    phi_prev[set]->update(1.0, *(phi[0]), 0.0);

    params->updateDynamicParams(k-1);
    postproc->setTimeIndex(k);
    assembler->updateStage(stage, current_time, deltat);

    sol_stage[set]->assign(*(sol[set]));

    this->nonlinearSolver(set, stage, sol, sol_stage, step_history,
                          phi, phi_stage, phi_prev);

    phi[set]->update(1.0, *(phi_stage[0]), 0.0);
    postproc->computeSensitivities(sol, sol_stage, step_history, phi,
                                   current_time, k, deltat, gradient);

    is_adjoint = false;
    linalg->resetJacobian();
    linalg->resetPrevJacobian();
  };

  BswRevolve schedule(num_steps, window_spans, num_slots);

  num_forward_solves = 0;
  is_final_time = false;
  bool done = false;
  long long guard = 0;
  const long long max_actions = 1000LL*(static_cast<long long>(num_steps) + num_slots);

  while (!done) {

    TEUCHOS_TEST_FOR_EXCEPTION(guard++ > max_actions, std::runtime_error,
      "Error: the BSW schedule did not terminate.");

    BswDirective d = schedule.next();

    if (d.action == BswAction::SolveRange) {

      for (int k=d.k1; k<=d.k2; ++k) {
        params->updateDynamicParams(k-1);
        assembler->updateTimeStep(k-1);
        int status = this->takeForwardStep(set, sol_carried, step_history, k-1);
        TEUCHOS_TEST_FOR_EXCEPTION(status != 0, std::runtime_error,
          "Error: a forward step failed during sketched-checkpointed recomputation.");
        current_time += deltat;
        ++num_forward_solves;
      }
    }
    else if (d.action == BswAction::JumpAnchor) {

      // crossing a window forward is free: land on its exact right edge
      bsw_manager->rightEdge(d.win, raw_state.data());
      loadFromRaw(sol_carried[set], raw_state);
      current_time = bsw_manager->getTime(d.step);
    }
    else if (d.action == BswAction::Store) {

      checkpoint_state[d.slot]->assign(*(sol_carried[set]));
      checkpoint_time[d.slot] = current_time;
    }
    else if (d.action == BswAction::Restore) {

      sol_carried[set]->assign(*(checkpoint_state[d.slot]));
      current_time = checkpoint_time[d.slot];
    }
    else if (d.action == BswAction::ReverseStep) {

      // classic reversal: recompute u_k from the carried u_{k-1}, one solve
      const int k = d.step;
      sol[set]->assign(*(sol_carried[set]));
      params->updateDynamicParams(k-1);
      assembler->updateTimeStep(k-1);
      int status = this->takeForwardStep(set, sol, step_history, k-1);
      TEUCHOS_TEST_FOR_EXCEPTION(status != 0, std::runtime_error,
        "Error: a forward step failed during sketched-checkpointed recomputation.");
      ++num_forward_solves;

      adjointStepAt(k, d.is_first);
    }
    else if (d.action == BswAction::ReverseWindow) {

      // window interiors come from reconstructions: zero forward solves.
      // On entry the carried state is the exact u_a.
      const int b_eff = window_spans[d.win].second;
      for (int k=b_eff; k>d.a; --k) {

        if (k == b_eff) {
          bsw_manager->rightEdge(d.win, raw_state.data());
        }
        else {
          bsw_manager->reconstruct(d.win, k, raw_state.data());
        }
        loadFromRaw(sol[set], raw_state);

        if (k-1 == d.a) {
          step_history[0]->assign(*(sol_carried[set]));
        }
        else {
          bsw_manager->reconstruct(d.win, k-1, raw_prev.data());
          loadFromRaw(step_history[0], raw_prev);
        }

        current_time = bsw_manager->getTime(k-1);
        adjointStepAt(k, d.is_first && k == b_eff);
      }
    }
    else if (d.action == BswAction::Terminate) {
      done = true;
    }
  }

  is_final_time = false;

  if (Comm->getRank() == 0 && verbosity > 0) {
    BswRevolve::GradientSolves predicted =
      BswRevolve::predictGradientSolves(num_steps, window_spans, num_slots);
    cout << endl << "**** Sketched-checkpointed adjoint complete" << endl;
    cout << "****   time steps               : " << num_steps << endl;
    cout << "****   committed windows        : " << window_spans.size() << endl;
    cout << "****   checkpoint slots         : " << num_slots << endl;
    cout << "****   forward solves used      : " << num_forward_solves
         << "  (predicted " << predicted.total << ")" << endl;
    cout << "****   window storage           : " << bsw_manager->getSpentSE()
         << " SE spent, " << bsw_manager->getPeakSE() << " SE peak" << endl;
    std::vector<std::string> layout = bsw_manager->layoutReport();
    for (size_t i=0; i<layout.size(); ++i) {
      cout << "****   " << layout[i] << endl;
    }
  }

  debugger->print("**** Finished SolverManager::bswAdjointModel");

}
