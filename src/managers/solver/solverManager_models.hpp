/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

namespace {
  // Restores the flag when leaving scope (including if the solve throws), so the
  // next forward/adjoint is not left stuck in incremental mode.
  struct BoolFlagGuard {
    bool & flag_;
    BoolFlagGuard(bool & f) : flag_(f) { flag_ = true; }
    ~BoolFlagGuard() { flag_ = false; }
    BoolFlagGuard(const BoolFlagGuard &) = delete;
    BoolFlagGuard & operator=(const BoolFlagGuard &) = delete;
  };

  // RAII: set scalar param to 0 on entry, 1 on exit (including throw).
  // sacadoizeParams pushes paramvals into the workset View; setParam alone does not.
  template<class ParamsPtr>
  struct ScalarParamGuard {
    ParamsPtr & params_;
    string name_;
    ScalarParamGuard(ParamsPtr & p, const string & name)
      : params_(p), name_(name) {
      vector<ScalarT> off = {0.0};
      params_->setParam(off, name_);
      params_->sacadoizeParams(false);
    }
    ~ScalarParamGuard() {
      vector<ScalarT> on = {1.0};
      params_->setParam(on, name_);
      params_->sacadoizeParams(false);
    }
    ScalarParamGuard(const ScalarParamGuard &) = delete;
    ScalarParamGuard & operator=(const ScalarParamGuard &) = delete;
  };
}

// ========================================================================================
/* given the parameters, solve the forward problem */
// ========================================================================================

template<class Node>
void SolverManager<Node>::forwardModel(ScalarT & objective) {
  
  Teuchos::TimeMonitor localtimer(*forwardtimer);
  
  current_time = initial_time;
  
  debugger->print("**** Starting SolverManager::forwardModel ...");
  
  is_adjoint = false;
  params->sacadoizeParams(false);
  postproc->resetObjectives();
  postproc->resetSolutions();
  linalg->resetAllJacobian();
  
  for (size_t set=0; set<setnames.size(); ++set) {
    if (!scalarDirichletData[set]) {
      if (!staticDirichletData[set]) {
        this->projectDirichlet(set);
      }
      else if (!have_static_Dirichlet_data[set]) {
        this->projectDirichlet(set);
        have_static_Dirichlet_data[set] = true;
      }
    }
  }
  
  vector<vector_RCP> sol = this->setInitial();
    
  if (solver_type == "steady-state") {
    this->steadySolver(sol);
  }
  else if (solver_type == "transient") {
    MrHyDE_OptVector gradient; // not really used here
    this->transientSolver(sol, gradient, initial_time, final_time);
  }
  else {
    // print out an error message
  }
    
  if (postproc->write_optimization_solution) {
    postproc->writeOptimizationSolution(numEvaluations);
  }
  
  postproc->reportObjective(objective);
  
  numEvaluations++;
  
  debugger->print("**** Finished SolverManager::forwardModel");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::adjointModel(MrHyDE_OptVector & gradient) {
  
  debugger->print("**** Starting SolverManager::adjointModel ...");
  
  Teuchos::TimeMonitor localtimer(*adjointtimer);
  
  if (setnames.size()>1 && Comm->getRank() == 0) {
    cout << "MrHyDE WARNING: Adjoints are not yet implemented for multiple physics sets." << endl;
  }
  else {
    
    is_adjoint = true;
    
    params->sacadoizeParams(false);
    linalg->resetAllJacobian();
    
    vector<vector_RCP> phi = setInitial();
    
    if (solver_type == "steady-state") {
      // Since this is the adjoint solve, we loop over the physics sets in reverse order
      for (size_t oset=0; oset<phi.size(); ++oset) {
        size_t set = phi.size()-1-oset;
        vector<vector_RCP> sol, zero_vec;
        for (size_t iset=0; iset<phi.size(); ++iset) { // just collecting states - order doesn't matter
          sol.push_back(linalg->getNewVector(iset));
          bool fnd = postproc->soln[set]->extract(sol[iset], current_time);
          if (!fnd) {
            cout << "UNABLE TO FIND FORWARD SOLUTION" << endl;
          }
        }
        params->updateDynamicParams(0);
        this->nonlinearSolver(set, 0, sol, sol, zero_vec, phi, phi, zero_vec);
        
        postproc->computeSensitivities(sol, zero_vec, zero_vec, phi, 0, current_time, deltat, gradient);
      }
    }
    else if (solver_type == "transient") {
      DFAD obj = 0.0;
      this->transientSolver(phi, gradient, initial_time, final_time);
    }
    else {
      // print out an error message
    }
    
    is_adjoint = false;
  }
  
  debugger->print("**** Finished SolverManager::adjointModel");
  
}

// ========================================================================================
// Tangent sweep (Heinkenschloss 2008, Alg. 4.1 step 3): solve c_y w = -c_u v
// with zero IC. Control set to v; src_gate=0 zeroes the physical source so the
// RHS is -c_u v. Writes w to incr_soln.
// ========================================================================================

template<class Node>
void SolverManager<Node>::incrementalForwardModel(MrHyDE_OptVector & v) {

  Teuchos::TimeMonitor localtimer(*forwardtimer);

  current_time = initial_time;

  debugger->print("**** Starting SolverManager::incrementalForwardModel ...");

  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter("src_gate"), std::runtime_error,
    "SolverManager::incrementalForwardModel requires inactive scalar 'src_gate' "
    "(wrap the physical source as src_gate*(...) in Functions).");

  is_adjoint = false;
  params->sacadoizeParams(false);
  // Without this, a prior adjoint leaves J^T cached and the tangent solve blows up.
  linalg->resetAllJacobian();

  params->updateParams(v);

  // Zero physical source: tangent RHS should be -c_u v only.
  ScalarParamGuard<decltype(params)> gate_guard(params, "src_gate");
  // Route record() to incr_soln so we do not overwrite the cached forward y.
  BoolFlagGuard incr_guard(postproc->is_incremental);

  // store() appends; reset so a second hessVec does not grow the timeline.
  for (size_t set=0; set<postproc->incr_soln.size(); ++set) {
    postproc->incr_soln[set]->reset();
  }

  // Tangent IC is zero even if the deck's forward IC is not.
  vector<vector_RCP> w = this->setInitial();
  for (size_t set=0; set<w.size(); ++set) {
    w[set]->putScalar(0.0);
  }

  if (solver_type == "steady-state") {
    this->steadySolver(w);
  }
  else if (solver_type == "transient") {
    MrHyDE_OptVector dummy_gradient;
    this->transientSolver(w, dummy_gradient, initial_time, final_time);
  }

  debugger->print("**** Finished SolverManager::incrementalForwardModel");

}

// ========================================================================================
// Second-order adjoint on w: assemble Hv. LQ homogeneous-in-state f:
// nabla_y f(w) = nabla_yy f * w; a linear-in-state term breaks that.
// ========================================================================================

template<class Node>
void SolverManager<Node>::incrementalAdjointModel(MrHyDE_OptVector & hessvec) {

  debugger->print("**** Starting SolverManager::incrementalAdjointModel ...");

  Teuchos::TimeMonitor localtimer(*adjointtimer);

  if (setnames.size()>1 && Comm->getRank() == 0) {
    cout << "MrHyDE WARNING: Adjoints are not yet implemented for multiple physics sets." << endl;
  }
  else {

    is_adjoint = true;
    BoolFlagGuard adj_guard(postproc->is_incremental_adjoint);

    // Optionally zero Td on incremental sweep.
    std::unique_ptr<ScalarParamGuard<decltype(params)>> trk_guard;
    if (params->isParameter("trk_gate")) {
      trk_guard = std::make_unique<ScalarParamGuard<decltype(params)>>(params, "trk_gate");
    }

    params->sacadoizeParams(false);
    // Need J^T; do not reuse the tangent-sweep Jacobian.
    linalg->resetAllJacobian();

    // p = second-order adjoint (zero terminal condition).
    vector<vector_RCP> p = setInitial();
    for (size_t set=0; set<p.size(); ++set) {
      p[set]->putScalar(0.0);
    }

    if (solver_type == "steady-state") {
      // Loop over physics sets in reverse order for the adjoint sweep.
      for (size_t oset=0; oset<p.size(); ++oset) {
        size_t set = p.size()-1-oset;
        vector<vector_RCP> sol, zero_vec;
        for (size_t iset=0; iset<p.size(); ++iset) {
          sol.push_back(linalg->getNewVector(iset));
          bool fnd = postproc->incr_soln[set]->extract(sol[iset], current_time);
          if (!fnd) {
            cout << "UNABLE TO FIND INCREMENTAL FORWARD SOLUTION" << endl;
          }
        }
        params->updateDynamicParams(0);
        this->nonlinearSolver(set, 0, sol, sol, zero_vec, p, p, zero_vec);

        // Assemble Hv += c_u^T p + reg Hessian action on v.
        postproc->computeSensitivities(sol, zero_vec, zero_vec, p, 0, current_time, deltat, hessvec);
      }
    }
    else if (solver_type == "transient") {
      this->transientSolver(p, hessvec, initial_time, final_time);
    }

    is_adjoint = false;
  }

  debugger->print("**** Finished SolverManager::incrementalAdjointModel");
}
