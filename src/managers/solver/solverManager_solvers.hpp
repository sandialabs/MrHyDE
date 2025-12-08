/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::steadySolver(vector<vector_RCP> & sol) {
  
  debugger->print("**** Starting SolverManager::steadySolver ...");
  
  for (int ss=0; ss<subcycles; ++ss) {
    for (size_t set=0; set<setnames.size(); ++set) {
      assembler->updatePhysicsSet(set);
      vector<vector_RCP> zero_soln;
      if (usestrongDBCs) {
        this->setDirichlet(set, sol[set]);
      }
      params->updateDynamicParams(0);
      this->nonlinearSolver(set, 0, sol, zero_soln, zero_soln,
                            zero_soln, zero_soln, zero_soln);
    }
  }
  postproc->record(sol, current_time, 1);
  
  debugger->print("**** Finished SolverManager::steadySolver");
  
}

// ========================================================================================
/* solve a transient problem */
// ========================================================================================

template<class Node>
void SolverManager<Node>::transientSolver(vector<vector_RCP> & initial, 
                                          MrHyDE_OptVector & gradient,
                                          ScalarT & start_time, ScalarT & end_time) {
  
  Teuchos::TimeMonitor localtimer(*transientsolvertimer);
  
  debugger->print(1, "******** Starting SolverManager::transientSolver ...");
  
  vector<vector_RCP> zero_vec(initial.size());
  
  current_time = start_time;
  if (!is_adjoint) { // forward solve - adaptive time stepping
    is_final_time = false;
    vector<vector_RCP> sol = initial;
    
    if (usestrongDBCs) {
      for (size_t set=0; set<initial.size(); ++set) {
        assembler->updatePhysicsSet(set);
        this->setDirichlet(set,sol[set]);
      }
    }
    
    postproc->record(sol,current_time,0);
    
    vector<vector<vector_RCP> > sol_prev;
    for (size_t set=0; set<sol.size(); ++set) {
      vector<vector_RCP> c_prev;
      for (int step=0; step<maxnumsteps[set]; ++step) {
        c_prev.push_back(linalg->getNewOverlappedVector(set));
      }
      sol_prev.push_back(c_prev);
    }
           
    int stepProg = 0;
    int numCuts = 0;
    int maxCuts = maxTimeStepCuts; // TMW: make this a user-defined input
    double timetol = end_time*1.0e-6; // just need to get close enough to final time
    
    while (current_time < (end_time-timetol) && numCuts<=maxCuts) {
      int status = 0;
      if (Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Time Step " << stepProg+1 << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      params->updateDynamicParams(stepProg);
      assembler->updateTimeStep(stepProg);
      
      //for (int ss=0; ss<subcycles; ++ss) {
        for (size_t set=0; set<sol.size(); ++set) {
          // this needs to come first now, so that updatePhysicsSet can pick out the
          // time integration info
          if (BDForder[set] > 1 && stepProg == startupSteps[set]) {
            // Only overwrite the current set
            this->setBackwardDifference(BDForder,set);
            this->setButcherTableau(ButcherTab,set);
          }

          assembler->updatePhysicsSet(set);
    
          // if num_stages = 1, the sol_stage = sol
          //
          vector<vector_RCP> sol_stage;
          if (maxnumstages[set] == 1) {
            sol_stage.push_back(sol[set]);
          }
          else {
            for (int stage=0; stage<maxnumstages[set]; ++stage) {
              sol_stage.push_back(linalg->getNewOverlappedVector(set));
              sol_stage[stage]->assign(*(sol[set]));
            }
          }
    
          // Increment the previous step solutions (shift history and moves u into first spot)
          for (size_t step=1; step<sol_prev[set].size(); ++step) {
            size_t ind = sol_prev[set].size()-step;
            sol_prev[set][ind]->assign(*(sol_prev[set][ind-1]));
          }
          sol_prev[set][0]->assign(*(sol[set]));
      
          ////////////////////////////////////////////////////////////////////////
          // Allow the groups to change subgrid model
          ////////////////////////////////////////////////////////////////////////
          
          vector<vector<int> > sgmodels = assembler->identifySubgridModels();
          multiscale_manager->update(sgmodels);
          
          for (int stage=0; stage<numstages[set]; stage++) {
            // Need a stage solution
            // Set the initial guess for stage solution
            // sol_stage[stage]->assign(*(sol_prev[0]));
            // Updates the current time and sets the stage number in wksets
            assembler->updateStage(stage, current_time, deltat); 

            if (usestrongDBCs) {
              this->setDirichlet(set, sol_stage[stage]);
            }
  
            if (fully_explicit) {
              status += this->explicitSolver(set, stage, sol, sol_stage, sol_prev[set], 
                                             zero_vec, zero_vec, zero_vec);
            }
            else {
              status += this->nonlinearSolver(set, stage, sol, sol_stage, sol_prev[set], zero_vec, zero_vec, zero_vec);
            }

            // u_{n+1} = u_n + \sum_stage ( u_stage - u_n )
            
            // if num_stages = 1, then we might be able to skip this 
            if (maxnumstages[set] > 1) {
              sol[set]->update(1.0, *(sol_stage[stage]), 1.0);
              sol[set]->update(-1.0, *(sol_prev[set][0]), 1.0);
            }
            multiscale_manager->completeStage();
          }
        }
      //}
      
      if (status == 0) { // NL solver converged
        current_time += deltat;
        stepProg += 1;
        
        // Make sure last step solution is gathered
        // Last set of values is from a stage solution, which is potentially different
        //assembler->performGather(sol, 0, 0);
        //for (size_t set=0; set<u.size(); ++set) {
        //  assembler->updatePhysicsSet(set);
        //  assembler->performGather(set,u[set],0,0);
        //}
        multiscale_manager->completeTimeStep();
        postproc->record(sol,current_time,stepProg);
        
      }
      else { // something went wrong, cut time step and try again
        deltat *= 0.5;
        numCuts += 1;
        for (size_t set=0; set<sol.size(); ++set) {
          assembler->revertSoln(set);
          sol[set]->assign(*(sol_prev[set][0]));
        }
        if (Comm->getRank() == 0 && verbosity > 0) {
          cout << endl << endl << "*******************************************************" << endl;
          cout << endl << "**** Cutting time step to " << deltat << endl;
          cout << "**** Current time is " << current_time << endl << endl;
          cout << "*******************************************************" << endl << endl << endl;
        }
        
      }
    }
    // If the final step doesn't fall when a write is requested, catch that here  
    if (stepProg % postproc->write_frequency != 0 && postproc->write_solution) {
      postproc->writeSolution(sol, current_time);
    }
  }
  else { // adjoint solve - fixed time stepping based on forward solve
  
    current_time = final_time;
    is_final_time = true;
    
    vector<vector_RCP> sol, sol_prev, phi, phi_prev, sol_stage, phi_stage;
    for (size_t set=0; set<1; ++set) { // hard coded for now
      sol.push_back(linalg->getNewOverlappedVector(set));
      sol_prev.push_back(linalg->getNewOverlappedVector(set));
      phi.push_back(linalg->getNewOverlappedVector(set));
      phi_prev.push_back(linalg->getNewOverlappedVector(set));
      sol_stage.push_back(linalg->getNewOverlappedVector(set));
      phi_stage.push_back(linalg->getNewOverlappedVector(set));
    }
    // Transient adjoints require derivatives of Jacobians w.r.t. previous states
    // We store the Jacobian-vector products in a (Nstep x Nstep) matrix
    if (previous_adjoints.size() == 0) {
      for (size_t i=0; i<numsteps[0]; ++i) { // hard-coded for now
        vector<vector_RCP> ivecs;
        for (size_t set=0; set<setnames.size(); ++set) { // hard-coded for now
          vector_RCP tempvec = linalg->getNewVector(set);
          tempvec->putScalar(0.0);
          ivecs.push_back(tempvec);
        }
        previous_adjoints.push_back(ivecs);
      }
    }
    else {
      for (size_t i=0; i<numsteps[0]; ++i) { // hard-coded for now
        for (size_t set=0; set<setnames.size(); ++set) { // hard-coded for now
          previous_adjoints[i][set]->putScalar(0.0);
        }
      }
    }
    
    size_t set = 0, stage = 0;
    // Just getting the number of times from first physics set should be fine
    // TODO will this be affected by having physics sets with different timesteppers?
    int store_index = 0;
    size_t numFwdSteps = postproc->soln[set]->getTotalTimes(store_index)-1; 
    
    for (size_t timeiter = 0; timeiter<numFwdSteps; timeiter++) {
      size_t cindex = numFwdSteps-timeiter;
      phi_prev[set] = linalg->getNewOverlappedVector(set);
      phi_prev[set]->update(1.0,*(phi[0]),0.0);
      if (Comm->getRank() == 0 && verbosity > 0) {
        cout << endl << endl << "*******************************************************" << endl;
        cout << endl << "**** Beginning Adjoint Time Step " << timeiter << endl;
        cout << "**** Current time is " << current_time << endl << endl;
        cout << "*******************************************************" << endl << endl << endl;
      }
      
      // TMW: this is specific to implicit Euler
      // Needs to be generalized
      // Also, need to implement checkpoint/recovery
      bool fndu = postproc->soln[set]->extract(sol[set], cindex);
      if (!fndu) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE was not able to find forward solution");
      }
      bool fndup = postproc->soln[set]->extract(sol_prev[set], cindex-1);
      if (!fndup) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE was not able to find previous forward solution");
      }
      //params->updateDynamicParams(cindex-1);
      params->updateDynamicParams(cindex-1);
      //assembler->performGather(set,u_prev[set],0,0);
      //assembler->resetPrevSoln(set);
      
      int stime_index = cindex-1;
      
      current_time = postproc->soln[set]->getSpecificTime(store_index, stime_index);
      postproc->setTimeIndex(cindex);
      assembler->updateStage(stage, current_time, deltat);
      
      sol_stage[set]->assign(*sol[set]);
      // if multistage, recover forward solution at each stage
      if (numstages[set] == 1) { // No need to re-solve in this case
        int status = this->nonlinearSolver(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev);
        if (status>0) {
          // throw error
        }
        phi[set]->update(1.0,*(phi_stage[0]),0.0);
        postproc->computeSensitivities(sol, sol_stage, sol_prev, phi, current_time, cindex, deltat, gradient);
      }
      else {
        // NEEDS TO BE REWRITTEN
      }
      
      is_final_time = false;
      
    }
    
  }
  
  debugger->print(1, "******** Finished SolverManager::transientSolver");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
int SolverManager<Node>::nonlinearSolver(const size_t & set, const size_t & stage,
                                         vector<vector_RCP> & sol, // [set]
                                         vector<vector_RCP> & sol_stage, // [stage]
                                         vector<vector_RCP> & sol_prev, // [step]
                                         vector<vector_RCP> & phi, // [set]
                                         vector<vector_RCP> & phi_stage, // [stage]
                                         vector<vector_RCP> & phi_prev) { // [step]
   // Goal is to update sol_stage[stage]
   // Assembler will need to gather sol for other physics sets and other step/stage solutions for current set

  Teuchos::TimeMonitor localtimer(*nonlinearsolvertimer);

  debugger->print(1, "******** Starting SolverManager::nonlinearSolver ...");

  int status = 0;
  int NLiter = 0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_first(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_scaled(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm(1);
  resnorm_first[0] = 10*NLtol;
  resnorm_scaled[0] = resnorm_first[0];
  resnorm[0] = resnorm_first[0];
  
  int maxiter = maxNLiter;
  if (is_adjoint) {
    maxiter = 1;//2;
  }
    
  bool proceed = true;
  ScalarT alpha = 1.0;
  
  vector_RCP current_res, current_res_over, current_du, current_du_over;
  if (store_vectors) {
    current_res = res[set];
    current_res_over = res_over[set];
    current_du = du[set];
    current_du_over = du_over[set];
  }
  else {
    current_res = linalg->getNewVector(set);
    current_res_over = linalg->getNewOverlappedVector(set);
    current_du = linalg->getNewVector(set);
    current_du_over = linalg->getNewOverlappedVector(set);
  }
  
  while (proceed) {
    
    multiscale_manager->reset();
    multiscale_manager->macro_nl_iter = NLiter;

    gNLiter = NLiter;
  
    bool build_jacobian = !linalg->getJacobianReuse(set);
    matrix_RCP J, J_over;
    
    J = linalg->getNewMatrix(set);
    if (build_jacobian) {
      J_over = linalg->getNewOverlappedMatrix(set);
      linalg->fillComplete(J_over);
      J_over->resumeFill();
      J_over->setAllToScalar(0.0);
    }
    
    // *********************** COMPUTE THE JACOBIAN AND THE RESIDUAL **************************
    
    current_res_over->putScalar(0.0);

    store_adjPrev = false; //false;
    if ( is_adjoint && (NLiter == 1)) {
      store_adjPrev = true;
    }

    bool use_autotune = true;
    if (assembler->groupData[0]->multiscale) {
      use_autotune = false;
    }

    auto paramvec = params->getDiscretizedParamsOver();
    auto paramdot = params->getDiscretizedParamsDotOver();
    
    // This is where the residual is computed for the forward problem
    // Jacobian is computed only if the residual is large enough to merit a linear solve
    // Adjoint residual is computed below
    if (!is_adjoint) {
      if (!use_autotune) {
        assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, build_jacobian, false, false, false, 0,
                                  current_res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                                  params->num_active_params, paramvec, paramdot, is_final_time, deltat);
      }
      else {
        assembler->assembleRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev,
                               paramvec, paramdot, current_res_over, J_over, isTransient, current_time, deltat);
      }
      linalg->exportVectorFromOverlapped(set, current_res, current_res_over);
    }
    
    // *********************** CHECK THE NORM OF THE RESIDUAL **************************
    
    {
      Teuchos::TimeMonitor localtimer(*normLAtimer);
      current_res->normInf(resnorm);
    }
    
    bool solve = true;
    if (NLiter == 0) {
      resnorm_first[0] = resnorm[0];
      resnorm_scaled[0] = 1.0;
    }
    else {
      resnorm_scaled[0] = resnorm[0]/resnorm_first[0];
    }
    
    // hard code these for adjoint solves since residual is computed below and only one iteration is needed
    if (is_adjoint) {
      resnorm[0] = 1.0;
      resnorm_scaled[0] = 1.0;
    }
    
    if (Comm->getRank() == 0 && verbosity > 1) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Iteration: " << NLiter << endl;
      cout << "***** Norm of nonlinear residual: " << resnorm[0] << endl;
      cout << "***** Scaled Norm of nonlinear residual: " << resnorm_scaled[0] << endl;
      cout << "*********************************************************" << endl;
    }
    
    if (!is_adjoint && allowBacktracking && resnorm_scaled[0] > 1.1) {
      solve = false;
      alpha *= 0.5;
      Teuchos::TimeMonitor localtimer(*updateLAtimer);
      if (sol_stage.size() > 0) {
        sol_stage[stage]->update(-1.0*alpha, *(current_du_over), 1.0);
      }
      else {
        sol[set]->update(-1.0*alpha, *(current_du_over), 1.0);
      }
      if (Comm->getRank() == 0 && verbosity > 1) {
        cout << "***** Backtracking: new learning rate = " << alpha << endl;
      }
    }
    else {
      if (useRelativeTOL) {
        if (resnorm_scaled[0]<NLtol) {
          solve = false;
          proceed = false;
        }
        else if (resnorm[0]<1.0e-100) { // Not sure why this is hard coded
          solve = false;
          proceed = false;
        }
      }
      else if (useAbsoluteTOL && resnorm[0]<NLabstol) {
        solve = false;
        proceed = false;
      }
    }
    if (is_adjoint) { // Always perform one linear solve
      solve = true; // force a solve
      proceed = false; // but only one
    }
    
    // *********************** SOLVE THE LINEAR SYSTEM FOR THE UPDATE **************************
    
    if (solve) {
      
      if (build_jacobian) {
        if (use_autotune) { // If false, J was already computed when the residual was computed - just saves an extra assembly
          auto paramvec = params->getDiscretizedParamsOver();
          auto paramdot = params->getDiscretizedParamsDotOver();
          assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, build_jacobian, false, false, false, 0,
                                    current_res_over, J_over, isTransient, current_time, is_adjoint, store_adjPrev,
                                    params->num_active_params, paramvec, paramdot, is_final_time, deltat);
        }
        linalg->fillComplete(J_over);
        J->resumeFill();
        linalg->exportMatrixFromOverlapped(set, J, J_over);
        linalg->fillComplete(J);
      }
            
      // This is where the adjoint residual is computed
      if (is_adjoint) {
        // First, the derivative of the objective w.r.t. the state
        ScalarT cdt = 0.0;
        if (isTransient) {
          cdt = deltat;
        }
        postproc->computeObjectiveGradState(set, sol[set], current_time+cdt, deltat, current_res);
        
        // We use a true adjoint residual, so we need to Jacobian^T times the current approximation
        
        auto mvprod = linalg->getNewVector(set);
        auto phi_owned = linalg->getNewVector(set);
        linalg->exportVectorFromOverlappedReplace(set, phi_owned, phi_stage[set]);
        J->apply(*phi_owned,*mvprod);
        current_res->update(-1.0, *mvprod, 1.0);
        
        
        // For transient problems, need to update adjoint residual with previous adjoint solutions (multi-step)
        // This is actually a little complicated
        // The jacobians need to be evaluated at the right time and using the right forward solution
        // The Jacobian-vector products we need on this time step should already be computed
        if (isTransient) {
          
          // use the stored Jacobian vector (Jv) products from previous time steps
          for (size_t istep=0; istep<previous_adjoints.size(); ++istep) {
            current_res->update(-1.0, *(previous_adjoints[istep][istep]), 1.0);
          }
          
          // Increment the Jv products
          size_t numSteps = sol_prev.size();
          for (size_t istep=0; istep<numSteps-1; ++istep) {
            for (size_t kstep=0; kstep<numSteps; ++kstep) {
              previous_adjoints[kstep][numSteps-istep] = previous_adjoints[kstep][numSteps-istep-1];
            }
          }
          
          // The next set of Jacobian vector products are calculated below once phi is updated
          
        }
        {
          Teuchos::TimeMonitor localtimer(*normLAtimer);
          current_res->normInf(resnorm);
        }
        
      }
      
      //******************************************************
      // Actual linear solve
      //******************************************************
      
      current_du->putScalar(0.0);
      current_du_over->putScalar(0.0);
      linalg->linearSolver(set, J, current_res, current_du);
      
      // doesn't always write to file - only if requested
      if (is_adjoint) {
        linalg->writeToFile(J, current_res, current_du, "adjoint_jacobian.mm",
                            "adjoint_residual.mm","adjoint_solution.mm");
      }
      else {
        linalg->writeToFile(J, current_res, current_du);
      }
      linalg->importVectorToOverlapped(set, current_du_over, current_du);
      
      alpha = 1.0; // what is the point of alpha?
      if (is_adjoint) {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        if (phi_stage.size() > 0) {
          phi_stage[stage]->update(alpha, *(current_du_over), 1.0);
        }
        else {
          phi[set]->update(alpha, *(current_du_over), 1.0);
        }
        
        if (isTransient) {
          // Fill in prev_mass_adjoints[:][0] - need to create new vectors since these are RCPs
          vector<matrix_RCP> Jprev = linalg->getNewPreviousMatrix(set, phi_prev.size());
          for (size_t step=0; step<phi_prev.size(); ++step) {
            
            if (build_jacobian) {
              auto paramvec = params->getDiscretizedParamsOver();
              auto paramdot = params->getDiscretizedParamsDotOver();
              matrix_RCP currJ = Jprev[step];
              matrix_RCP currJ_over = linalg->getNewOverlappedMatrix(set);
              linalg->fillComplete(currJ_over);
              currJ_over->resumeFill();
              currJ_over->setAllToScalar(0.0);
              auto dummy_res_over = linalg->getNewOverlappedVector(set);
              assembler->assembleJacRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, build_jacobian, false, false, true, step,
                                        dummy_res_over, currJ_over, isTransient, current_time, is_adjoint, store_adjPrev,
                                        params->num_active_params, paramvec, paramdot, is_final_time, deltat);
              linalg->fillComplete(currJ_over);
              currJ->resumeFill();
              linalg->exportMatrixFromOverlapped(set, currJ, currJ_over);
              linalg->fillComplete(currJ);
            }
            
            auto mvprod = linalg->getNewVector(set);
            auto phip_owned = linalg->getNewVector(set);
            
            if (step == 0) {
              if (phi_stage.size() > 0) {
                linalg->exportVectorFromOverlappedReplace(set, phip_owned, phi_stage[stage]);
              }
              else {
                linalg->exportVectorFromOverlappedReplace(set, phip_owned, phi[set]);
              }
            }
            else {
              linalg->exportVectorFromOverlappedReplace(set, phip_owned, phi_prev[step-1]);
            }
            
            Jprev[step]->apply(*phip_owned,*mvprod);
            previous_adjoints[0][0]->update(1.0, *mvprod, 0.0);
            
          }
        }
      }
      else {
        Teuchos::TimeMonitor localtimer(*updateLAtimer);
        if (sol_stage.size() > 0) {
          sol_stage[stage]->update(alpha, *(current_du_over), 1.0);
        }
        else {
          sol[set]->update(alpha, *(current_du_over), 1.0);
        }
      }
    }
    NLiter++; // increment number of nonlinear iterations
    
    if (is_adjoint) {
      //proceed = false;
    }
    else if (NLiter >= maxiter) {
      proceed = false;
    }
  } // while loop
  
  if (verbosity>1) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> normu(1);
    if (sol_stage.size() > 0) {
      sol_stage[stage]->norm2(normu);
    }
    else {
      sol[set]->norm2(normu);
    }
    if (Comm->getRank() == 0) {
      cout << "Norm of solution: " << normu[0] << "    (overlapped vector so results may differ on multiple procs)" << endl;
    }
  }
  if (Comm->getRank() == 0) {
    if (!is_adjoint) {
      if ( (NLiter>maxNLiter) && verbosity > 1) {
        status = 1;
        cout << endl << endl << "********************" << endl;
        cout << endl << "SOLVER FAILED TO CONVERGE CONVERGED in " << NLiter
        << " iterations with residual norm " << resnorm[0] << endl;
        cout << "********************" << endl;
      }
    }
  }
  
  debugger->print(1, "******** Finished SolverManager::nonlinearSolver");
  
  return status;
}

// ========================================================================================
// ========================================================================================

template<class Node>
int SolverManager<Node>::explicitSolver(const size_t & set, const size_t & stage,
                                        vector<vector_RCP> & sol, // [set]
                                        vector<vector_RCP> & sol_stage, // [stage]
                                        vector<vector_RCP> & sol_prev, // [step]
                                        vector<vector_RCP> & phi, // [set]
                                        vector<vector_RCP> & phi_stage, // [stage]
                                        vector<vector_RCP> & phi_prev) { // [step]
  
  
  // Goal is just to update sol_stage[stage]
  // Other solutions are just used in assembler for gather operations

  Teuchos::TimeMonitor localtimer(*explicitsolvertimer);
  
  debugger->print(1, "******** Starting SolverManager::explicitSolver ...");
  
  int status = 0;
  assembler->updatePhysicsSet(set);
  
  if (usestrongDBCs) {
    this->setDirichlet(set,sol_stage[stage]);
  }
  
  vector_RCP current_res, current_res_over;
  if (store_vectors) {
    current_res = res[set];
    current_res_over = res_over[set];
  }
  else {
    current_res = linalg->getNewVector(set);
    if (linalg->getHaveOverlapped()) {
      current_res_over = linalg->getNewOverlappedVector(set);
    }
    else {
      current_res_over = current_res;
    }
  }

  // *********************** COMPUTE THE RESIDUAL **************************
    
  current_res_over->putScalar(0.0);
  matrix_RCP J_over;
  
  auto paramvec = params->getDiscretizedParamsOver();
  auto paramdot = params->getDiscretizedParamsDotOver();
  assembler->assembleRes(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev,
                         paramvec, paramdot, current_res_over, J_over, isTransient, current_time, deltat);
  
  if (linalg->getHaveOverlapped()) {
    linalg->exportVectorFromOverlapped(set, current_res, current_res_over);
  }

  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  current_res->norm2(rnorm);
  //KokkosTools::print(current_res);
  
  // *********************** SOLVE THE LINEAR SYSTEM **************************
  
  if (rnorm[0]>1.0e-100) {
    // Given m = diag(M^-1)
    // Given timewt = b(stage)*deltat
    // Compute du = timewt*m*res
    // Compute u += du
    
    ScalarT wt = deltat*butcher_b(stage);
    
    
    if (!assembler->lump_mass) {
      vector_RCP current_du, current_du_over;
      if (store_vectors) {
        current_du = du[set];
        current_du_over = du_over[set];
      }
      else {
        current_du = linalg->getNewVector(set);
        if (linalg->getHaveOverlapped()) {
          current_du_over = linalg->getNewOverlappedVector(set);
        }
        else {
          current_du_over = current_du;
        }
      }

      current_du_over->putScalar(0.0);
      if (linalg->getHaveOverlapped()) {
        current_du->putScalar(0.0);
      }

      current_res->scale(wt);
      if (assembler->matrix_free) {
        this->matrixFreePCG(set, current_res, current_du, diagMass[set],
                            settings->sublist("Solver").get("linear TOL",1.0e-2),
                            settings->sublist("Solver").get("max linear iters",100));
      }
      else {
        if (use_custom_PCG) {
          this->PCG(set, explicitMass[set], current_res, current_du, diagMass[set],
                    settings->sublist("Solver").get("linear TOL",1.0e-2),
                    settings->sublist("Solver").get("max linear iters",100));
        }
        else {
          linalg->linearSolverL2(set, explicitMass[set], current_res, current_du);
        }
      }
      if (linalg->getHaveOverlapped()) {
        linalg->importVectorToOverlapped(set, current_du_over, current_du);
      }
      
      sol_stage[stage]->update(1.0, *(current_du_over), 1.0);
      
    }
    else {
      typedef typename Node::execution_space LA_exec;
      
      // can probably avoid du in this case
      // sol += sol + wt*res/dm
      vector_RCP current_sol;
      if (linalg->getHaveOverlapped()) {
        current_sol = linalg->getNewVector(set);
        linalg->exportVectorFromOverlappedReplace(set, current_sol, sol_stage[stage]);
      }
      else {
        current_sol = sol_stage[stage];
      }

      //auto du_view = current_du->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto sol_view = current_sol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto res_view = current_res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto dm_view = diagMass[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      
      parallel_for("explicit solver apply invdiag",
                   RangePolicy<LA_exec>(0,sol_view.extent(0)),
                   KOKKOS_LAMBDA (const int k ) {
        sol_view(k,0) += wt*res_view(k,0)/dm_view(k,0);
      });

      if (linalg->getHaveOverlapped()) {
        linalg->importVectorToOverlapped(set, sol_stage[stage], current_sol);
      }
    }
    
  }
  
  if (verbosity>=10) {
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> unorm(1);
    sol_stage[stage]->norm2(unorm);
    if (Comm->getRank() == 0) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Explicit integrator: L2 norm of (overlapped/ghosted) solution: " << unorm[0] << endl;
      cout << "*********************************************************" << endl;
    }
  }
  
  debugger->print(1, "******** Finished SolverManager::explicitSolver");
  
  return status;
}

// ========================================================================================
// Specialized PCG
// ========================================================================================

template<class Node>
void SolverManager<Node>::PCG(const size_t & set, matrix_RCP & J, vector_RCP & b, vector_RCP & x,
                              vector_RCP & M, const ScalarT & tol, const int & maxiter) {
  
  Teuchos::TimeMonitor localtimer(*PCGtimer);
  
  typedef typename Node::execution_space LA_exec;
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);
  
  ScalarT rho = 1.0, rho1 = 0.0, alpha = 0.0, beta = 1.0, pq = 0.0;
  ScalarT one = 1.0, zero = 0.0;
  
  vector_RCP p, q, r, z;
  if (store_vectors) {
    p = p_pcg[set];
    q = q_pcg[set];
    r = r_pcg[set];
    z = z_pcg[set];
  }
  else {
    p = linalg->getNewVector(set);
    q = linalg->getNewVector(set);
    r = linalg->getNewVector(set);
    z = linalg->getNewVector(set);
  }
  
  p->putScalar(zero);
  q->putScalar(zero);
  r->putScalar(zero);
  z->putScalar(zero);
  
  int iter=0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  {
    Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
    J->apply(*x,*q);
  }
  
  r->assign(*b);
  r->update(-one,*q,one);
  
  r->norm2(rnorm);
  ScalarT r0 = rnorm[0];
  
  auto M_view = M->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto r_view = r->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto z_view = z->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  while (iter<maxiter && rnorm[0]/r0>tol) {
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyPrectimer);
      parallel_for("PCG apply prec",
                   RangePolicy<LA_exec>(0,z_view.extent(0)),
                   KOKKOS_LAMBDA (const int k ) {
        z_view(k,0) = r_view(k,0)/M_view(k,0);
      });
    }
    
    rho1 = rho;
    r->dot(*z, dotprod);
    rho = dotprod[0];
    if (iter == 0) {
      p->assign(*z);
    }
    else {
      beta = rho/rho1;
      p->update(one,*z,beta);
    }
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
      J->apply(*p,*q);
    }
    
    p->dot(*q,dotprod);
    pq = dotprod[0];
    alpha = rho/pq;
    
    x->update(alpha,*p,one);
    r->update(-one*alpha,*q,one);
    r->norm2(rnorm);
    
    iter++;
  }
  if (verbosity >= 10 && Comm->getRank() == 0) {
    cout << " ******* PCG Convergence Information: " << endl;
    cout << " *******     Iter: " << iter << "   " << "rnorm = " << rnorm[0]/r0 << endl;
  }
}

// ========================================================================================
// Specialized matrix-free PCG
// ========================================================================================

template<class Node>
void SolverManager<Node>::matrixFreePCG(const size_t & set, vector_RCP & b, vector_RCP & x,
                                        vector_RCP & M, const ScalarT & tol, const int & maxiter) {
  
  Teuchos::TimeMonitor localtimer(*PCGtimer);
  
  typedef typename Node::execution_space LA_exec;
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);
  
  ScalarT rho = 1.0, rho1 = 0.0, alpha = 0.0, beta = 1.0, pq = 0.0;
  ScalarT one = 1.0, zero = 0.0;
  
  vector_RCP p, q, r, z, p_over, q_over;
  if (store_vectors) {
    p = p_pcg[set];
    q = q_pcg[set];
    r = r_pcg[set];
    z = z_pcg[set];
    p_over = p_pcg_over[set];
    q_over = q_pcg_over[set];
  }
  else {
    p = linalg->getNewVector(set);
    q = linalg->getNewVector(set);
    r = linalg->getNewVector(set);
    z = linalg->getNewVector(set);
    if (linalg->getHaveOverlapped()) {
      p_over = linalg->getNewOverlappedVector(set);
      q_over = linalg->getNewOverlappedVector(set);
    }
    else {
      p_over = p;
      q_over = q;
    }
  }
   
  p->putScalar(zero);
  q->putScalar(zero);
  r->putScalar(zero);
  z->putScalar(zero);
  
  if (linalg->getHaveOverlapped()) {
    p_over->putScalar(zero);
    q_over->putScalar(zero);
  }

  int iter=0;
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
  
  {
    Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
    if (linalg->getHaveOverlapped()) {
      linalg->importVectorToOverlapped(set, p_over, x);
    }
    else {
      p_over->assign(*x);
    }
    assembler->applyMassMatrixFree(set, p_over, q_over);
    if (linalg->getHaveOverlapped()) {
      linalg->exportVectorFromOverlapped(set, q, q_over);
    }
  }
  
  r->assign(*b);
  r->update(-one,*q,one);
  
  r->norm2(rnorm);
  ScalarT r0 = rnorm[0];
  
  auto M_view = M->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto r_view = r->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto z_view = z->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  while (iter<maxiter && rnorm[0]/r0>tol) {
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyPrectimer);
      parallel_for("PCG apply prec",
                   RangePolicy<LA_exec>(0,z_view.extent(0)),
                   KOKKOS_LAMBDA (const int k ) {
        z_view(k,0) = r_view(k,0)/M_view(k,0);
      });
    }
    
    rho1 = rho;
    r->dot(*z, dotprod);
    rho = dotprod[0];
    if (iter == 0) {
      p->assign(*z);
    }
    else {
      beta = rho/rho1;
      p->update(one,*z,beta);
    }
    
    {
      Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
      if (linalg->getHaveOverlapped()) {
        linalg->importVectorToOverlapped(set, p_over, p);
      }
      q_over->putScalar(zero);
      assembler->applyMassMatrixFree(set, p_over, q_over);
      if (linalg->getHaveOverlapped()) {
        linalg->exportVectorFromOverlapped(set, q, q_over);
      }
    }
    
    p->dot(*q,dotprod);
    pq = dotprod[0];
    alpha = rho/pq;
    
    x->update(alpha,*p,one);
    r->update(-one*alpha,*q,one);
    r->norm2(rnorm);
    
    iter++;
  }
  if (verbosity >= 10 && Comm->getRank() == 0) {
    cout << " ******* PCG Convergence Information: " << endl;
    cout << " *******     Iter: " << iter << "   " << "rnorm = " << rnorm[0]/r0 << endl;
  }
}
