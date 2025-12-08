/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

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
  linalg->resetJacobian();
  
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
    linalg->resetJacobian();
    
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
// solve an incremental forward problem for the incremental adjoint
// ========================================================================================

template<class Node>
void SolverManager<Node>::incrementalForwardModel(ScalarT & objective) {
  
}

// ========================================================================================
// solve an incremental adjoint for the hessian-vector product
// This should only be called after a forward solve, an adjoint solve, and an incremental forward solve
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
    
    params->sacadoizeParams(false);
    linalg->resetJacobian();
    
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
        
        //postproc->computeSensitivities(sol, zero_vec, zero_vec, phi, 0, current_time, deltat, gradient);
      }
    }
    else if (solver_type == "transient") {
      DFAD obj = 0.0;
      //this->transientSolver(phi, gradient, initial_time, final_time);
    }
    else {
      // print out an error message
    }
    
    is_adjoint = false;
  }
  
  debugger->print("**** Finished SolverManager::adjointModel");
}
