/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

/* transient adjoint driven by a Revolve (Algorithm 799) checkpointing schedule */

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
  is_final_time = true;
  bool done = false;
  int guard = 0;
  const int max_actions = 1000*(num_steps + num_checkpoints);

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

      // slots are 1-based in the schedule, 0-based here
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
