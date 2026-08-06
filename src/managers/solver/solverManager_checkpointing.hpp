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

  ////////////////////////////////////////////////////////////////////////////
  // Scope guards.  Revolve assumes a fixed, known number of steps and a state
  // that is fully described by the vectors we hand it.  Each of these breaks
  // one of those assumptions, so refuse loudly rather than return a gradient
  // that is quietly wrong.
  ////////////////////////////////////////////////////////////////////////////

  TEUCHOS_TEST_FOR_EXCEPTION(solver_type != "transient", std::runtime_error,
    "Error: checkpointed adjoints require a transient solve.");

  TEUCHOS_TEST_FOR_EXCEPTION(setnames.size() != 1, std::runtime_error,
    "Error: checkpointed adjoints are only implemented for a single physics set.");

  TEUCHOS_TEST_FOR_EXCEPTION(numsteps[set] != 1, std::runtime_error,
    "Error: checkpointed adjoints require a single-step time integrator (BDF1). "
    "A multistep method needs more than one previous state per reverse step.");

  TEUCHOS_TEST_FOR_EXCEPTION(numstages[set] != 1, std::runtime_error,
    "Error: checkpointed adjoints require a single-stage time integrator.");

  TEUCHOS_TEST_FOR_EXCEPTION(multiscale_manager->subgridModels.size() > 0, std::runtime_error,
    "Error: checkpointed adjoints are not supported with subgrid models. Subgrid "
    "models carry state that is not in the macro solution vector, so restoring a "
    "checkpoint would not restore the full state.");

  const int num_steps = static_cast<int>(std::round((final_time - initial_time)/deltat));

  TEUCHOS_TEST_FOR_EXCEPTION(num_steps < 1, std::runtime_error,
    "Error: checkpointed adjoints need at least one time step.");

  TEUCHOS_TEST_FOR_EXCEPTION(num_checkpoints < 1, std::runtime_error,
    "Error: 'Analysis: number of checkpoints' must be at least 1.");

  ////////////////////////////////////////////////////////////////////////////
  // Storage.  This is the point of the exercise: num_checkpoints states, not
  // num_steps.  Allocate once and reuse the slots.
  ////////////////////////////////////////////////////////////////////////////

  vector<vector_RCP> cp_state(num_checkpoints);
  vector<ScalarT>    cp_time(num_checkpoints, initial_time);
  for (int i=0; i<num_checkpoints; ++i) {
    cp_state[i] = linalg->getNewOverlappedVector(set);
  }

  // The state travelling along the schedule, plus the BDF history that
  // takeForwardStep shifts.  After a step, work_prev[0] holds the state we
  // came from -- which is exactly what the adjoint step needs.
  params->sacadoizeParams(false);
  linalg->resetAllJacobian();

  is_adjoint = false;
  vector<vector_RCP> sol = this->setInitial();
  if (usestrongDBCs) {
    assembler->updatePhysicsSet(set);
    this->setDirichlet(set, sol[set]);
  }
  current_time = initial_time;

  vector<vector_RCP> work_prev;
  for (int i=0; i<maxnumsteps[set]; ++i) {
    work_prev.push_back(linalg->getNewOverlappedVector(set));
  }

  // Scratch for the recomputed u_k, so the carried state u_{k-1} survives.
  vector<vector_RCP> sol_k, sol_stage, phi, phi_prev, phi_stage;
  sol_k.push_back(linalg->getNewOverlappedVector(set));
  sol_stage.push_back(linalg->getNewOverlappedVector(set));
  phi.push_back(linalg->getNewOverlappedVector(set));
  phi_prev.push_back(linalg->getNewOverlappedVector(set));
  phi_stage.push_back(linalg->getNewOverlappedVector(set));

  // Transient adjoints carry Jacobian-vector products from the step above;
  // same allocation as the stored-state path in transientSolver.
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

  ////////////////////////////////////////////////////////////////////////////
  // Run the schedule.  Revolve decides; this loop carries the decisions out.
  ////////////////////////////////////////////////////////////////////////////

  Revolve schedule(num_steps, num_checkpoints);

  num_ckpt_state_solves = 0;
  is_final_time = true;
  bool done = false;
  int guard = 0;
  const int guard_max = 1000*(num_steps + num_checkpoints);

  while (!done) {

    TEUCHOS_TEST_FOR_EXCEPTION(guard++ > guard_max, std::runtime_error,
      "Error: the Revolve schedule did not terminate.");

    const int prev_position = schedule.getRangeStart();
    RevolveAction action = schedule.next();

    if (action == RevolveAction::Advance) {

      // Recompute forward, storing nothing.
      for (int k=prev_position+1; k<=schedule.getRangeStart(); ++k) {
        params->updateDynamicParams(k-1);
        assembler->updateTimeStep(k-1);
        int status = this->takeForwardStep(set, sol, work_prev, k-1);
        TEUCHOS_TEST_FOR_EXCEPTION(status != 0, std::runtime_error,
          "Error: a forward step failed while recomputing for a checkpointed adjoint. "
          "The schedule assumes a fixed time step, so a step cut cannot be recovered from.");
        current_time += deltat;
        ++num_ckpt_state_solves;
      }
    }
    else if (action == RevolveAction::Store) {

      // Slot numbering is 1-based in the schedule, 0-based in the array.
      const int slot = schedule.getNumCheckpointsStored() - 1;
      cp_state[slot]->assign(*(sol[set]));
      cp_time[slot] = current_time;
    }
    else if (action == RevolveAction::Restore) {

      const int slot = schedule.getNumCheckpointsStored() - 1;
      sol[set]->assign(*(cp_state[slot]));
      // Restore the recorded time rather than recomputing it: accumulating
      // initial_time + k*deltat rounds differently than the forward march did,
      // and the objective is evaluated at this time.
      current_time = cp_time[slot];
    }
    else if (action == RevolveAction::FirstReverse || action == RevolveAction::Reverse) {

      // A reversal at position range_start handles time step k = range_start+1.
      // On entry sol holds u_{k-1} and current_time is t_{k-1}.
      const int k = schedule.getRangeStart() + 1;

      // Recompute u_k into scratch, keeping u_{k-1} in sol.  takeForwardStep
      // leaves u_{k-1} in work_prev[0], which is what the adjoint step wants.
      sol_k[set]->assign(*(sol[set]));
      params->updateDynamicParams(k-1);
      assembler->updateTimeStep(k-1);
      int status = this->takeForwardStep(set, sol_k, work_prev, k-1);
      TEUCHOS_TEST_FOR_EXCEPTION(status != 0, std::runtime_error,
        "Error: a forward step failed while recomputing for a checkpointed adjoint.");
      ++num_ckpt_state_solves;

      // ---- one adjoint step, mirroring the stored-state path ----
      is_adjoint = true;
      is_final_time = (action == RevolveAction::FirstReverse);

      if (Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << "**** Checkpointed adjoint step " << k
             << " (time " << current_time << ")" << endl;
      }

      phi_prev[set] = linalg->getNewOverlappedVector(set);
      phi_prev[set]->update(1.0, *(phi[0]), 0.0);

      params->updateDynamicParams(k-1);
      postproc->setTimeIndex(k);
      assembler->updateStage(stage, current_time, deltat);

      sol_stage[set]->assign(*(sol_k[set]));

      status = this->nonlinearSolver(set, stage, sol_k, sol_stage, work_prev,
                                     phi, phi_stage, phi_prev);

      phi[set]->update(1.0, *(phi_stage[0]), 0.0);
      postproc->computeSensitivities(sol_k, sol_stage, work_prev, phi,
                                     current_time, k, deltat, gradient);

      is_adjoint = false;
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

    // The trajectory should NOT be in the postprocessor's storage -- that is the
    // memory saving.  Ask without indexing, since the container may be empty.
    size_t states_in_storage = 0;
    vector<vector<ScalarT> > all_times = postproc->soln[set]->extractAllTimes();
    if (all_times.size() > 0) {
      states_in_storage = all_times[0].size();
    }

    cout << endl << "**** Checkpointed adjoint complete" << endl;
    cout << "****   time steps               : " << num_steps << endl;
    cout << "****   checkpoints stored       : " << num_checkpoints << endl;
    cout << "****   forward solves used      : " << num_ckpt_state_solves
         << "  (predicted " << num_steps + extra << ")" << endl;
    cout << "****   states in full storage   : " << states_in_storage
         << "  (stored-state path would hold " << num_steps + 1 << ")" << endl;
  }

  debugger->print("**** Finished SolverManager::checkpointedAdjointModel");

}
