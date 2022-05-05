/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "solverManager.hpp"
#include "subgridDtN_solver.hpp"

using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


SubGridDtN_Solver::SubGridDtN_Solver(const Teuchos::RCP<MpiComm> & LocalComm,
                                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                     Teuchos::RCP<MeshInterface> & mesh,
                                     Teuchos::RCP<DiscretizationInterface> & disc,
                                     Teuchos::RCP<PhysicsInterface> & physics,
                                     Teuchos::RCP<AssemblyManager<SubgridSolverNode> > & assembler_,
                                     Teuchos::RCP<ParameterManager<SubgridSolverNode>> & params,
                                     size_t & numMacroDOF) :
settings(settings_), assembler(assembler_) {
  
  verbosity = settings->get<int>("verbosity",0);
  debug_level = settings->get<int>("debug level",0);
  dimension = settings->sublist("Mesh").get<int>("dim",2);
  multiscale_method = settings->get<string>("multiscale method","mortar");
  shape = settings->sublist("Mesh").get<string>("shape","quad");
  macroshape = settings->sublist("Mesh").get<string>("macro-shape","quad");
  time_steps = settings->sublist("Solver").get<int>("number of steps",1);
  initial_time = settings->sublist("Solver").get<ScalarT>("initial time",0.0);
  final_time = settings->sublist("Solver").get<ScalarT>("final time",1.0);
  
  have_sym_factor = false;
  sub_NLtol = settings->sublist("Solver").get<ScalarT>("nonlinear TOL",1.0E-12);
  sub_maxNLiter = settings->sublist("Solver").get<int>("max nonlinear iters",10);
  useDirect = settings->sublist("Solver").get<bool>("use direct solver",true);
  amesos_solver_type = settings->sublist("Solver").get<string>("Amesos solver type","KLU2");

  use_preconditioner = settings->sublist("Solver").get<bool>("use preconditioner",true);
  
  store_aux_and_flux = settings->sublist("Postprocess").get<bool>("store aux and flux",false);
  
  solver = Teuchos::rcp( new SolverManager<SubgridSolverNode>(LocalComm, settings, mesh, disc,
                                                              physics, assembler, params) );
  
  res = solver->linalg->getNewVector(0);
  J = solver->linalg->getNewOverlappedMatrix(0);
  
  // Initialize storage for intermediate and final stage solutions
  u = solver->linalg->getNewOverlappedVector(0);
  u_prev = solver->linalg->getNewOverlappedVector(0);
  u_stage = solver->linalg->getNewOverlappedVector(0);
  phi = solver->linalg->getNewOverlappedVector(0);
  
  if (LocalComm->getSize() > 1) {
    res_over = solver->linalg->getNewOverlappedVector(0);
    sub_J_over = solver->linalg->getNewOverlappedMatrix(0);
  }
  else {
    res_over = res;
    sub_J_over = J;
  }
  
  d_um = solver->linalg->getNewVector(0,numMacroDOF);
  d_sub_res_overm = solver->linalg->getNewOverlappedVector(0,numMacroDOF);
  d_sub_resm = solver->linalg->getNewVector(0,numMacroDOF);
  d_sub_u_prevm = solver->linalg->getNewVector(0,numMacroDOF);
  d_sub_u_overm = solver->linalg->getNewOverlappedVector(0,numMacroDOF);
  
  du_glob = solver->linalg->getNewVector(0);
  if (LocalComm->getSize() > 1) {
    du = solver->linalg->getNewOverlappedVector(0);
  }
  else {
    du = du_glob;
  }
  if (useDirect) {
    {
      Teuchos::TimeMonitor amsetuptimer(*sgfemNonlinearSolverAmesosSetupTimer);
      Am2Solver = Amesos2::create<SG_CrsMatrix,SG_MultiVector>(amesos_solver_type, J);
    }
    {
      Teuchos::TimeMonitor amsetuptimer(*sgfemNonlinearSolverAmesosSymbFactTimer);
      Am2Solver->symbolicFactorization();
    }
    have_sym_factor = true;
  }
  else {
    Teuchos::TimeMonitor amsetuptimer(*sgfemNonlinearSolverBelosSetupTimer);
    J->fillComplete(); // temporary
    belos_problem = Teuchos::rcp(new SG_LinearProblem(J, u, res));
    have_belos = true;
    
    // Need to read these in from input file
    belosList = Teuchos::rcp(new Teuchos::ParameterList());
    belosList->set("Maximum Iterations",    50); // Maximum number of iterations allowed
    belosList->set("Convergence Tolerance", 1.0E-10);    // Relative convergence tolerance requested
    belosList->set("Verbosity", Belos::Errors);
    belosList->set("Output Frequency",0);
    
    int numEqns = solver->numVars[0][0];
    belosList->set("number of equations",numEqns);
    
    belosList->set("Output Style",          Belos::Brief);
    belosList->set("Implicit Residual Scaling", "None");
    
    belos_solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT, SG_MultiVector, SG_Operator>(belos_problem, belosList));
    belos_problem->setProblem(u, res);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridDtN_Solver::solve(View_Sc3 coarse_u,
                              View_Sc3 coarse_phi,
                              Teuchos::RCP<SG_MultiVector> & prev_phi,
                              Teuchos::RCP<SG_MultiVector> & disc_params,
                              const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                              const bool & compute_jacobian, const bool & compute_sens,
                              const int & num_active_params,
                              const bool & compute_disc_sens, const bool & compute_aux_sens,
                              workset & macrowkset,
                              const int & macrogrp, const int & macroelemindex,
                              Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) {
  
  Teuchos::TimeMonitor totalsolvertimer(*sgfemSolverTimer);
  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Starting SubGridDtN_Solver::solve ..." << endl;
    }
  }
  ScalarT current_time = time;
  //int macroDOF = macrowkset.numDOF;
  //bool usesubadjoint = false;
  ScalarT macro_deltat = macrowkset.deltat;

  Kokkos::deep_copy(subgradient, 0.0);
  
  if (std::abs(current_time - final_time) < 1.0e-12)
    is_final_time = true;
  else
    is_final_time = false;
  
  ///////////////////////////////////////////////////////////////////////////////////
  // Subgrid transient
  ///////////////////////////////////////////////////////////////////////////////////
  
  ScalarT alpha = 0.0;
  
  ///////////////////////////////////////////////////////////////////////////////////
  // Solve the subgrid problem(s)
  ///////////////////////////////////////////////////////////////////////////////////
  
  View_Sc3 lambda = coarse_u;
  if (isAdjoint) {
    lambda = coarse_phi;
  }
  
  // remove seeding on active params for now
  //if (compute_sens) {
  //  this->sacadoizeParams(false, num_active_params);
  //}
  
  //////////////////////////////////////////////////////////////
  // Set the initial conditions
  //////////////////////////////////////////////////////////////

  // TODO what is going on here?
  // Does this type of structure eliminate the need for what we discussed?
  // getLocalView is the data owned by this MPI process (obtained on the specified memory space)
  // TODO is u here defined over all groups? It seems so given the gather code...
  
  // TODO going to change this to assign but ask about it...
  //auto prev_u_kv = prev_u->getLocalView<SubgridSolverNode::device_type>(Tpetra::Access::ReadWrite);
  //auto u_kv = u->getLocalView<SubgridSolverNode::device_type>(Tpetra::Access::ReadWrite);
  //Kokkos::deep_copy(u_kv, prev_u_kv);

  auto current_stage = macrowkset.current_stage;

  if (current_stage == 0) {

    // TODO is this robust to the timestep cutting that occurs at the coarse scale?
    // I don't think so...
    // But I don't think the code that was there originally (using SolutionStorage) was either... could be wrong
    // TODO I am also concerned that we are not respecting the iterations of the coarse scale solver.
    // E.g., for the first stage, the subgrid solver is called multiple times until convergence
    // In that case, the solution is probably stacking up incorrectly
    u_prev->assign(*u); // overwrite u_n on first stage

    // prepare the group storage by shifting old data and copying u_n into first slot

    this->resetGroupStorage("prev");
    this->resetGroupStorage("stage");

  } 

  u_stage->assign(*u_prev); // set initial guess for stage solution (always u_n)
  
  // this moves u_stage to the groups
  this->performGather(macrogrp, u_stage, 0, 0);

  //////////////////////////////////////////////////////////////
  // Use the coarse scale solution to solve local transient/nonlinear problem
  //////////////////////////////////////////////////////////////
  
  Teuchos::RCP<SG_MultiVector> d_u = d_um;
  if (compute_sens) {
    d_u = solver->linalg->getNewVector(0,num_active_params);
  }
  d_u->putScalar(0.0);
  
  res->putScalar(0.0);

  //ScalarT h = 0.0;
  //assembler->wkset[0]->resetFlux();
  
  if (isTransient) {
    ScalarT sgtime = time - macro_deltat;
    vector<Teuchos::RCP<SG_MultiVector> > curr_fsol;
    vector<ScalarT> subsolvetimes;
    subsolvetimes.push_back(sgtime);

    // TODO I am not touching the adjoint at this point
    // so this may break or behave unexpectedly
    if (isAdjoint) {
      // First, we need to resolve the forward problem
      
      for (int tstep=0; tstep<time_steps; tstep++) {
        Teuchos::RCP<SG_MultiVector> recu = solver->linalg->getNewOverlappedVector(0);
        
        *recu = *u;
        sgtime += macro_deltat/(ScalarT)time_steps;
        subsolvetimes.push_back(sgtime);
        
        // set du/dt and \lambda
        alpha = (ScalarT)time_steps/macro_deltat;
        assembler->wkset[0]->alpha = alpha;
        assembler->wkset[0]->setDeltat(1.0/alpha);
        
        Kokkos::View<ScalarT***,AssemblyDevice> currlambda = coarse_u;
        
        //ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
        
        this->nonlinearSolver(recu, phi, disc_params, currlambda,
                              sgtime, isTransient, false, num_active_params, alpha, macrogrp, false);
        
        curr_fsol.push_back(recu);
        
      }
      
      for (int tstep=0; tstep<time_steps; tstep++) {
        
        size_t numsubtimes = subsolvetimes.size();
        size_t tindex = numsubtimes-1-tstep;
        sgtime = subsolvetimes[tindex];
        // set du/dt and \lambda
        alpha = (ScalarT)time_steps/macro_deltat;
        assembler->wkset[0]->alpha = alpha;
        assembler->wkset[0]->setDeltat(1.0/alpha);
        
        Kokkos::View<ScalarT***,AssemblyDevice> currlambda = lambda;
        
        ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
        
        this->nonlinearSolver(curr_fsol[tindex-1], phi, disc_params, currlambda,
                              sgtime, isTransient, isAdjoint, num_active_params, alpha, macrogrp, store_adjPrev);
        
        this->computeSolnSens(d_u, compute_sens, curr_fsol[tindex-1],
                              phi, disc_params, currlambda,
                              sgtime, isTransient, isAdjoint, num_active_params, alpha, lambda_scale, macrogrp, subgradient);
        
        this->updateFlux(phi, d_u, lambda, disc_params, compute_sens, macroelemindex, time, macrowkset, macrogrp);
        
      }
    }
    else {
      for (int tstep=0; tstep<time_steps; tstep++) {
        // TODO This seems to assume the coarse scale is single-stage
        // and the subgrid is BWE
        // If we update the stage, BDF, RK info correctly, then
        // the correct "z" should get constructed by the nonlinear solver

        // TODO to start we will assume the synchronous case
        // notably, the sub timestepping probably won'twork
        sgtime += macro_deltat/(ScalarT)time_steps;
        // set du/dt and \lambda
        alpha = (ScalarT)time_steps/macro_deltat;
        
        // TODO should we copy every time?
        // TODO device concerns?

        assembler->wkset[0]->butcher_A = macrowkset.butcher_A;
        assembler->wkset[0]->butcher_b = macrowkset.butcher_b;
        assembler->wkset[0]->butcher_c = macrowkset.butcher_c;

        assembler->wkset[0]->BDF_wts = macrowkset.BDF_wts;

        // TODO what is alpha?
        // TODO is this workset guaranteed to be the macro workset?
        assembler->wkset[0]->alpha = alpha;
        assembler->wkset[0]->setDeltat(1.0/alpha);

        Kokkos::View<ScalarT***,AssemblyDevice> currlambda = lambda;
        
        ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
        
        // TODO do we need a d_ustage??
        
        this->nonlinearSolver(u_stage, phi, disc_params, currlambda,
                              sgtime, isTransient, isAdjoint, num_active_params, alpha, macrogrp, false);
        
        this->computeSolnSens(d_u, compute_sens, u_stage,
                              phi, disc_params, currlambda,
                              sgtime, isTransient, isAdjoint, num_active_params, alpha, lambda_scale, macrogrp, subgradient);
        
        this->updateFlux(u_stage, d_u, lambda, disc_params, compute_sens, macroelemindex, time, macrowkset, macrogrp);

        // TODO for the very first iteration, this matches -- then things diverge
        KokkosTools::print(u_stage);

        // TODO update u_{n+1} = u_n + \sum_stage ( u_stage - u_n )
        // TODO BUT recall that u is just a copy of the data that is passed in at
        // the beginning of the routine... we need persistent storage throughout the loop

        // update u_{n+1} = u_n + \sum_stage ( u_stage - u_n )

        u->update(1.0, *u_stage, 1.0);
        u->update(-1.0, *u_prev, 1.0);

        // update the stage storage at the group level

        this->updateGroupStorage("stage");

        // TODO some form of gather step when timestep is completed
        // (Ensure u is correct like in solverManager 1370-ish)

      }
    }
    
  }
  else {
    
    assembler->wkset[0]->setDeltat(1.0);
    
    this->nonlinearSolver(u, phi, disc_params, lambda,
                          current_time, isTransient, isAdjoint, num_active_params, alpha, macrogrp, false);
    
    this->computeSolnSens(d_u, compute_sens, u,
                          phi, disc_params, lambda,
                          current_time, isTransient, isAdjoint, num_active_params, alpha, 1.0, macrogrp, subgradient);
    
    if (isAdjoint) {
      this->updateFlux(phi, d_u, lambda, disc_params, compute_sens, macroelemindex, time, macrowkset, macrogrp);
    }
    else {
      this->updateFlux(u, d_u, lambda, disc_params, compute_sens, macroelemindex, time, macrowkset, macrogrp);
    }
    
  }
  
  if (store_aux_and_flux) {
    this->storeFluxData(lambda, macrowkset.res);
  }
  
  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Finished SubGridDtN_Solver::solve ..." << endl;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Store macro-dofs and flux (for ML-based subgrid)
// Does not work on GPU for obvious reasons
///////////////////////////////////////////////////////////////////////////////////////

void SubGridDtN_Solver::storeFluxData(View_Sc3 lambda, View_AD2 flux) {
  
  //int num_dof_lambda = lambda.extent(1)*lambda.extent(2);
  
  std::ofstream ofs;
  
  // Input data - macro DOFs
  ofs.open ("input_data.txt", std::ofstream::out | std::ofstream::app);
  ofs.precision(10);
  //for (size_t e=0; e<lambda.extent(0); e++) {
    for (size_t i=0; i<lambda.extent(1); i++) {
      for (size_t j=0; j<lambda.extent(2); j++) {
        ofs << lambda(0,i,j) << "  ";
      }
    }
    ofs << endl;
  //}
  ofs.close();
  
  // Output data - upscaled flux
  ofs.open ("output_data.txt", std::ofstream::out | std::ofstream::app);
  ofs.precision(10);
  for (size_t e=0; e<flux.extent(0); e++) {
    //for (size_t i=0; i<flux.extent(1); i++) {
    //for (size_t j=0; j<flux.extent(2); j++) {
#ifndef MrHyDE_NO_AD
    ofs << flux(e,0).val() << "  ";
#else
    ofs << flux(e,0) << "  ";
#endif
    //}
    //}
    ofs << endl;
  }
  ofs.close();
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Subgrid Nonlinear Solver
///////////////////////////////////////////////////////////////////////////////////////

void SubGridDtN_Solver::nonlinearSolver(Teuchos::RCP<SG_MultiVector> & sub_u,
                                               Teuchos::RCP<SG_MultiVector> & sub_phi,
                                               Teuchos::RCP<SG_MultiVector> & sub_params,
                                               Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                               const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                               const int & num_active_params, const ScalarT & alpha, const int & macrogrp,
                                               const bool & store_adjPrev) {
  
  
  Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverTimer);
  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Starting SubGridDtN_Solver::nonlinearSolver ..." << endl;
    }
  }
  
  typedef typename SubgridSolverNode::execution_space SG_exec;
  
  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<SG_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }
  
  // LIDs are on AssemblyDevice.  If the AssemblyDevice memory is accessible, then these are fine.
  // Copy of LIDs is stored on HostDevice.
  bool use_host_LIDs = false;
  if (!data_avail) {
    if (Kokkos::SpaceAccessibility<SG_exec, HostDevice::memory_space>::accessible) {
      use_host_LIDs = true;
    }
  }
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_scaled(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_initial(1);
  resnorm[0] = 10.0*sub_NLtol;
  resnorm_initial[0] = resnorm[0];
  resnorm_scaled[0] = resnorm[0];
  
  int iter = 0;
  Kokkos::View<ScalarT**,AssemblyDevice> aPrev;
  
  auto localMatrix = sub_J_over->getLocalMatrixDevice();
  auto res_view = res_over->template getLocalView<SubgridSolverNode::device_type>(Tpetra::Access::ReadWrite);
  
  while (iter < sub_maxNLiter && resnorm_scaled[0] > sub_NLtol) {
    
    sub_J_over->resumeFill();
    
    sub_J_over->setAllToScalar(0.0);
    res_over->putScalar(0.0);
        
    //assembler->wkset[0]->setTime(time);
    assembler->wkset[0]->time = time;
    assembler->wkset[0]->isTransient = isTransient;
    assembler->wkset[0]->isAdjoint = isAdjoint;
    
    int numElem = assembler->groups[macrogrp][0]->numElem;
    int maxElem = assembler->groups[0][0]->numElem;
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSetSolnTimer);
      this->performGather(macrogrp, sub_u, 0, 0);
      if (isAdjoint) {
        this->performGather(macrogrp, sub_phi, 2, 0);
      }
      //this->performGather(macrogrp, sub_params, 4, 0);
      
      //this->performBoundaryGather(macrogrp, sub_u, 0, 0);
      //if (isAdjoint) {
      //  this->performBoundaryGather(macrogrp, sub_phi, 2, 0);
      //}
      
      for (size_t e=0; e < assembler->boundary_groups[macrogrp].size(); e++) {
        assembler->boundary_groups[macrogrp][e]->aux = lambda;
      }
    }
    
    ////////////////////////////////////////////////
    // Assembly
    ////////////////////////////////////////////////
    
    {
      
      Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverAssemblyTimer);
    
      ////////////////////////////////////////////////
      // volume assembly
      ////////////////////////////////////////////////
      
      auto res = assembler->wkset[0]->res;
      auto offsets = assembler->wkset[0]->offsets;
      auto numDOF = assembler->groupData[0]->numDOF;
    
      for (size_t e=0; e<assembler->groups[macrogrp].size(); e++) {
        
        if (isAdjoint) {
          if (is_final_time) {
            Kokkos::deep_copy(assembler->groups[macrogrp][e]->adj_prev[0], 0.0);
          }
        }
        
        //////////////////////////////////////////////////////////////
        // Compute the AD-seeded solutions at integration points
        //////////////////////////////////////////////////////////////
        
        int seedwhat = 1;
        
        //////////////////////////////////////////////////////////////
        // Compute res and J=dF/du
        //////////////////////////////////////////////////////////////
        
        // Volumetric contribution
        assembler->groups[macrogrp][e]->updateWorkset(seedwhat);
        assembler->phys->volumeResidual(0,0);
                
        //////////////////////////////////////////////////////////////////////////
        // Scatter into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverScatterTimer);
        
          auto LIDs = assembler->groups[macrogrp][e]->LIDs[0];
          
          parallel_for("assembly insert Jac",
                       RangePolicy<SG_exec>(0,LIDs.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                int row = offsets(n,j);
                LO rowIndex = LIDs(elem,row);
#ifndef MrHyDE_NO_AD
                ScalarT val = -res(elem,row).val();
#else
                ScalarT val = -res(elem,row);
#endif
                Kokkos::atomic_add(&(res_view(rowIndex,0)), val);
              }
            }
          });
#ifndef MrHyDE_NO_AD
          parallel_for("assembly insert Jac",
                       RangePolicy<SG_exec>(0,LIDs.extent(0)),
                       KOKKOS_LAMBDA (const int elem ) {
            const size_type numVals = LIDs.extent(1);
            LO cols[maxDerivs];
            ScalarT vals[maxDerivs];
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                int row = offsets(n,j);
                LO rowIndex = LIDs(elem,row);
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    int col = offsets(m,k);
                    vals[col] = res(elem,row).fastAccessDx(col);
                    cols[col] = LIDs(elem,col);
                  }
                }
                localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, true, true); // isSorted, useAtomics
                
              }
            }
          });
#endif
        }
      }
      
      ////////////////////////////////////////////////
      // boundary assembly
      ////////////////////////////////////////////////
      
      for (size_t e=0; e<assembler->boundary_groups[macrogrp].size(); e++) {
        assembler->wkset[0]->isOnSide = true;
        if (assembler->boundary_groups[macrogrp][e]->numElem > 0) {
          
          int seedwhat = 1;
          
          assembler->boundary_groups[macrogrp][e]->updateWorkset(seedwhat);
          assembler->phys->boundaryResidual(0,0);
          
          //////////////////////////////////////////////////////////////////////////
          // Scatter into global matrix/vector
          ///////////////////////////////////////////////////////////////////////////
          
          {
            Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverScatterTimer);
            
            auto LIDs = assembler->boundary_groups[macrogrp][e]->LIDs[0];
            
            parallel_for("assembly insert Jac",
                         RangePolicy<SG_exec>(0,LIDs.extent(0)),
                         KOKKOS_LAMBDA (const int elem ) {
              for (size_type n=0; n<numDOF.extent(0); ++n) {
                for (int j=0; j<numDOF(n); j++) {
                  int row = offsets(n,j);
                  LO rowIndex = LIDs(elem,row);
#ifndef MrHyDE_NO_AD
                  ScalarT val = -res(elem,row).val();
#else
                  ScalarT val = -res(elem,row);
#endif
                  Kokkos::atomic_add(&(res_view(rowIndex,0)), val);
                }
              }
            });

#ifndef MrHyDE_NO_AD
            parallel_for("assembly insert Jac",
                         RangePolicy<SG_exec>(0,LIDs.extent(0)),
                         KOKKOS_LAMBDA (const int elem ) {
              const size_type numVals = LIDs.extent(1);
              LO cols[maxDerivs];
              ScalarT vals[maxDerivs];
              for (size_type n=0; n<numDOF.extent(0); ++n) {
                for (int j=0; j<numDOF(n); j++) {
                  int row = offsets(n,j);
                  LO rowIndex = LIDs(elem,row);
                  for (size_type m=0; m<numDOF.extent(0); m++) {
                    for (int k=0; k<numDOF(m); k++) {
                      int col = offsets(m,k);
                      vals[col] = res(elem,row).fastAccessDx(col);
                      cols[col] = LIDs(elem,col);
                    }
                  }
                  localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, true, true); // isSorted, useAtomics
                  
                }
              }
            });
#endif
          }
        }
        assembler->wkset[0]->isOnSide = false;
      }
      
      //////////////////////////////////////////////////////////////////////////
      // Fix up any empty rows due to workset size
      ///////////////////////////////////////////////////////////////////////////
      
      if (maxElem > numElem) {
        if (data_avail) {
          auto LIDs = assembler->groups[0][0]->LIDs[0];
          this->fixDiagonal(LIDs, localMatrix, numElem);
        }
        else {
          if (use_host_LIDs) {
            auto LIDs = assembler->groups[0][0]->LIDs_host[0];
            this->fixDiagonal(LIDs, localMatrix, numElem);
          }
          else {
            auto LIDs_dev = Kokkos::create_mirror(SG_exec(), assembler->groups[0][0]->LIDs[0]);
            Kokkos::deep_copy(LIDs_dev, assembler->groups[0][0]->LIDs[0]);
            this->fixDiagonal(LIDs_dev, localMatrix, numElem);
          }
        }
        
      }
    }
    
    sub_J_over->fillComplete();
    
    if (solver->Comm->getSize() > 1) {
      J->resumeFill();
      J->setAllToScalar(0.0);
      J->doExport(*sub_J_over, *(solver->linalg->exporter[0]), Tpetra::ADD); // TMW: tmp fix
      J->fillComplete();
    }
    else {
      J = sub_J_over;
    }
    //KokkosTools::print(J);
    
    
    if (solver->Comm->getSize() > 1) {
      res->putScalar(0.0);
      res->doExport(*res_over, *(solver->linalg->exporter[0]), Tpetra::ADD); // TMW: tmp fix
    }
    else {
      res = res_over;
    }
    
    if (iter == 0) {
      res->normInf(resnorm_initial);
      if (resnorm_initial[0] > 0.0)
        resnorm_scaled[0] = 1.0;
      else
        resnorm_scaled[0] = 0.0;
    }
    else {
      res->normInf(resnorm);
      resnorm_scaled[0] = resnorm[0]/resnorm_initial[0];
    }
    if(solver->Comm->getRank() == 0 && verbosity>5) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Subgrid Nonlinear Iteration: " << iter << endl;
      cout << "***** Scaled Norm of nonlinear residual: " << resnorm_scaled << endl;
      cout << "*********************************************************" << endl;
    }
    
    if (resnorm_scaled[0] > sub_NLtol) {
      
      Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSolveTimer);
      du_glob->putScalar(0.0);
      if (useDirect) {
        Am2Solver->setA(J, Amesos2::SYMBFACT);
        Am2Solver->setX(du_glob);
        Am2Solver->setB(res);
        Am2Solver->numericFactorization().solve();
      }
      else {
        if (use_preconditioner) {
          if (have_preconditioner) {
            // TMW: why is this commented?
            //MueLu::ReuseTpetraPreconditioner(J,*belos_M);
          }
          else {
            belos_M = solver->linalg->buildPreconditioner(J,"Preconditioner Settings");
            //belos_problem->setRightPrec(belos_M);
            belos_problem->setLeftPrec(belos_M);
            //have_preconditioner = true;
            
          }
        }
        belos_problem->setProblem(du_glob, res);
        belos_solver->solve();
        
      }
      if (solver->Comm->getSize() > 1) {
        du->putScalar(0.0);
        du->doImport(*du_glob, *(solver->linalg->importer[0]), Tpetra::ADD); // TMW tmp fix
      }
      else {
        du = du_glob;
      }
      if (isAdjoint) {
        
        sub_phi->update(1.0, *du, 1.0);
      }
      else {
        sub_u->update(1.0, *du, 1.0);
      }
    }
    iter++;
    
  }
  //KokkosTools::print(sub_u);
  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Finished SubGridDtN_Solver::nonlinearSolver ..." << endl;
    }
  }
}

//////////////////////////////////////////////////////////////
// Correct the diagonal for certain cases
//////////////////////////////////////////////////////////////

template<class LIDViewType, class MatType>
void SubGridDtN_Solver::fixDiagonal(LIDViewType LIDs, MatType localMatrix, const int startpoint) {
  
  typedef typename SubgridSolverNode::execution_space SG_exec;
  
  parallel_for("subgrid diagonal fix",
               RangePolicy<SG_exec>(startpoint,LIDs.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    ScalarT vals[1];
    LO cols[1];
    for( size_type row=0; row<LIDs.extent(1); row++ ) {
      LO rowIndex = LIDs(elem,row);
      vals[0] = 1.0;
      cols[0] = rowIndex;
      localMatrix.sumIntoValues(rowIndex, cols, 1, vals, true, true); // bools: isSorted, useAtomics
    }
  });
  
}


//////////////////////////////////////////////////////////////
// Compute the derivative of the local solution w.r.t coarse
// solution or w.r.t parameters
//////////////////////////////////////////////////////////////

void SubGridDtN_Solver::computeSolnSens(Teuchos::RCP<SG_MultiVector> & d_sub_u,
                                               const bool & compute_sens,
                                               Teuchos::RCP<SG_MultiVector> & sub_u,
                                               Teuchos::RCP<SG_MultiVector> & sub_phi,
                                               Teuchos::RCP<SG_MultiVector> & sub_param,
                                               Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                               const ScalarT & time,
                                               const bool & isTransient, const bool & isAdjoint,
                                               const int & num_active_params, const ScalarT & alpha,
                                               const ScalarT & lambda_scale, const int & macrogrp,
                                               Kokkos::View<ScalarT**,AssemblyDevice> subgradient) {
  
  Teuchos::TimeMonitor localtimer(*sgfemSolnSensTimer);
  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Starting SubGridDtN_Solver::computeSolnSens ..." << endl;
    }
  }
  
  typedef typename SubgridSolverNode::device_type SG_device;
  typedef typename SubgridSolverNode::execution_space SG_exec;

  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<SG_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }
  
  bool use_host_LIDs = false;
  if (!data_avail) {
    if (Kokkos::SpaceAccessibility<SG_exec, HostDevice::memory_space>::accessible) {
      use_host_LIDs = true;
    }
  }
  
  Teuchos::RCP<SG_MultiVector> d_sub_res_over = d_sub_res_overm;
  Teuchos::RCP<SG_MultiVector> d_sub_res = d_sub_resm;
  Teuchos::RCP<SG_MultiVector> d_sub_u_prev = d_sub_u_prevm;
  Teuchos::RCP<SG_MultiVector> d_sub_u_over = d_sub_u_overm;
  
  if (compute_sens) {
    int numsubDerivs = d_sub_u->getNumVectors();
    d_sub_res_over = solver->linalg->getNewOverlappedVector(0,numsubDerivs);
    d_sub_res = solver->linalg->getNewVector(0,numsubDerivs);
    d_sub_u_prev = solver->linalg->getNewVector(0,numsubDerivs);
    d_sub_u_over = solver->linalg->getNewOverlappedVector(0,numsubDerivs);
  }
  
  d_sub_res_over->putScalar(0.0);
  d_sub_res->putScalar(0.0);
  d_sub_u_prev->putScalar(0.0);
  d_sub_u_over->putScalar(0.0);
  
  //ScalarT scale = -1.0*lambda_scale;
  
  auto dres_view = d_sub_res_over->getLocalView<SG_device>(Tpetra::Access::ReadWrite);
  
  //assembler->wkset[0]->setTime(time);
  assembler->wkset[0]->isTransient = isTransient;
  assembler->wkset[0]->isAdjoint = isAdjoint;
  
  if (compute_sens) {
    
    //this->sacadoizeParams(true, num_active_params);
    
    if (multiscale_method != "mortar") {
      
      int numElem = assembler->groups[macrogrp][0]->numElem;
      int snumDOF = assembler->groups[macrogrp][0]->LIDs[0].extent(1);
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_res("local residual",numElem,snumDOF,num_active_params);
      Kokkos::View<ScalarT***,AssemblyDevice> local_J("local Jacobian",numElem,snumDOF,snumDOF);
      
      for (size_t elem=0; elem<assembler->groups[macrogrp].size(); elem++) {
        
        assembler->wkset[0]->localEID = elem;
        assembler->groups[macrogrp][elem]->updateData();
        Kokkos::deep_copy(local_res, 0.0);
        
        assembler->groups[macrogrp][elem]->computeJacRes(time, isTransient, isAdjoint,
                                                    false, true, num_active_params, false, false, false,
                                                    local_res, local_J,
                                                    assembler->assemble_volume_terms[0][0],
                                                    assembler->assemble_face_terms[0][0]);
        
        this->updateResSens(true, macrogrp, elem, dres_view, local_res,
                            data_avail, use_host_LIDs, true);
          
      }
      
    }
    else {
            
      int numElem = assembler->boundary_groups[macrogrp][0]->numElem;
      int snumDOF = assembler->boundary_groups[macrogrp][0]->LIDs[0].extent(1);
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_res("local residual",numElem,snumDOF,num_active_params);
      Kokkos::View<ScalarT***,AssemblyDevice> local_J("local Jacobian",numElem,snumDOF,snumDOF);
      assembler->wkset[0]->isOnSide = true;
      for (size_t elem=0; elem<assembler->boundary_groups[macrogrp].size(); elem++) {
        
        assembler->boundary_groups[macrogrp][elem]->updateData();
        Kokkos::deep_copy(local_res, 0.0);
        
        assembler->boundary_groups[macrogrp][elem]->computeJacRes(time, isTransient, isAdjoint,
                                                               false, true, num_active_params, false, false, false,
                                                               local_res, local_J);
        
        this->updateResSens(false, macrogrp, elem, dres_view, local_res,
                            data_avail, use_host_LIDs, true);
          
      }
      assembler->wkset[0]->isOnSide = false;
    }
    
    auto sub_phi_kv = sub_phi->getLocalView<SG_device>(Tpetra::Access::ReadWrite);
    auto d_sub_res_over_kv = d_sub_res_over->getLocalView<SG_device>(Tpetra::Access::ReadWrite);
    
    auto subgrad_host = Kokkos::create_mirror_view(subgradient);
    
    for (int p=0; p<num_active_params; p++) {
      auto sub_res_sv = Kokkos::subview(d_sub_res_over_kv,Kokkos::ALL(),p);
      ScalarT subgrad = 0.0;
      parallel_reduce(RangePolicy<SG_exec>(0,sub_phi_kv.extent(0)), 
                      KOKKOS_LAMBDA (const int i, ScalarT& update) {
        update += sub_phi_kv(i,0) * sub_res_sv(i);
      }, subgrad);
      subgrad_host(p,0) = subgrad;
    }
    Kokkos::deep_copy(subgradient,subgrad_host);
  }
  else {
    
    if (multiscale_method != "mortar") {
      
      int numElem = assembler->groups[macrogrp][0]->numElem;
      int snumDOF = assembler->groups[macrogrp][0]->LIDs[0].extent(1);
      int anumDOF = assembler->groups[macrogrp][0]->auxLIDs.extent(1);
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_res("local residual",numElem,snumDOF,1);
      Kokkos::View<ScalarT***,AssemblyDevice> local_J("local Jacobian",numElem,snumDOF,anumDOF);
      
      for (size_t elem=0; elem<assembler->groups[macrogrp].size(); elem++) {
                
        assembler->wkset[0]->localEID = elem;
        
        Kokkos::deep_copy(local_res, 0.0);
        Kokkos::deep_copy(local_J, 0.0);
        
        // TMW: this may not work properly with new version
        assembler->groups[macrogrp][elem]->updateData();
        
        assembler->groups[macrogrp][elem]->computeJacRes(time, isTransient, isAdjoint,
                                                       true, false, num_active_params, false, true, false,
                                                       local_res, local_J,
                                                       assembler->assemble_volume_terms[0][0],
                                                       assembler->assemble_face_terms[0][0]);
        
        this->updateResSens(true, macrogrp, elem, dres_view, local_J,
                            data_avail, use_host_LIDs, false);
        
      }
    }
    else {
      
      //int numElem = assembler->boundary_groups[macrogrp][0]->numElem;
      //int snumDOF = assembler->boundary_groups[macrogrp][0]->LIDs.extent(1);
      //int anumDOF = assembler->boundary_groups[macrogrp][0]->auxLIDs.extent(1);
      
      auto res_AD = assembler->wkset[0]->res;
      auto offsets = assembler->wkset[0]->offsets;
      auto numDOF = assembler->groupData[0]->numDOF;
      auto numAuxDOF = assembler->groupData[0]->numAuxDOF;
      auto aoffsets = assembler->boundary_groups[macrogrp][0]->auxoffsets;
      
      assembler->wkset[0]->isOnSide = true;
      for (size_t elem=0; elem<assembler->boundary_groups[macrogrp].size(); elem++) {
        
        //-----------------------------------------------
        // Prep the workset
        //-----------------------------------------------
        
        int seedwhat = 4;
        assembler->boundary_groups[macrogrp][elem]->updateWorkset(seedwhat);
          
        //-----------------------------------------------
        // Compute the residual
        //-----------------------------------------------
        
        assembler->phys->boundaryResidual(0,0);
          
        
        //-----------------------------------------------
        // Scatter to vectors
        //-----------------------------------------------
        
        auto LIDs = assembler->boundary_groups[macrogrp][elem]->LIDs[0];
        
#ifndef MrHyDE_NO_AD
        parallel_for("bcell update aux jac",
                     RangePolicy<SG_exec>(0,LIDs.extent(0)),
                     KOKKOS_LAMBDA (const int elem) {
          for (size_type n=0; n<numDOF.extent(0); ++n) {
            for (int j=0; j<numDOF(n); ++j) {
              LO row = offsets(n,j);
              LO rowIndex = LIDs(elem,offsets(n,j));
              for (size_type m=0; m<numAuxDOF.extent(0); ++m) {
                for (int k=0; k<numAuxDOF(m); ++k) {
                  LO col = aoffsets(m,k);
                  double val = res_AD(elem,row).fastAccessDx(col);
                  Kokkos::atomic_add(&(dres_view(rowIndex,col)),-1.0*val);
                }
              }
            }
          }
        });
#endif
      }
      assembler->wkset[0]->isOnSide = false;
    }
    
    if (solver->Comm->getSize() > 1) {
      d_sub_res->doExport(*d_sub_res_over, *(solver->linalg->exporter[0]), Tpetra::ADD); // TMW tmp fix
    }
    else {
      d_sub_res = d_sub_res_over;
    }
    
    //KokkosTools::print(d_sub_res);
    
    if (useDirect) {
      
      Teuchos::TimeMonitor localtimer(*sgfemSolnSensLinearSolverTimer);
      
      int numsubDerivs = d_sub_u_over->getNumVectors();
      
      auto d_sub_u_over_kv = d_sub_u_over->getLocalView<SG_device>(Tpetra::Access::ReadWrite);
      auto d_sub_res_kv = d_sub_res->getLocalView<SG_device>(Tpetra::Access::ReadWrite);
      for (int c=0; c<numsubDerivs; c++) {
        Teuchos::RCP<SG_MultiVector> x = solver->linalg->getNewOverlappedVector(0); //Teuchos::rcp(new SG_MultiVector(solver->LA_overlapped_map,1));
        Teuchos::RCP<SG_MultiVector> b = solver->linalg->getNewVector(0); //Teuchos::rcp(new SG_MultiVector(solver->LA_owned_map,1));
        auto b_kv = Kokkos::subview(b->getLocalView<SG_device>(Tpetra::Access::ReadWrite),Kokkos::ALL(),0);
        auto x_kv = Kokkos::subview(x->getLocalView<SG_device>(Tpetra::Access::ReadWrite),Kokkos::ALL(),0);
        
        auto u_sv = Kokkos::subview(d_sub_u_over_kv,Kokkos::ALL(),c);
        auto res_sv = Kokkos::subview(d_sub_res_kv,Kokkos::ALL(),c);
        Kokkos::deep_copy(b_kv,res_sv);
        
        Am2Solver->setX(x);
        Am2Solver->setB(b);
        Am2Solver->solve();
        
        Kokkos::deep_copy(u_sv,x_kv);
        
      }
      
    }
    else {
      
      Teuchos::TimeMonitor localtimer(*sgfemSolnSensLinearSolverTimer);
      
      belos_problem->setProblem(d_sub_u_over, d_sub_res);
      belos_solver->solve();
    }
    
    if (solver->Comm->getSize() > 1) {
      d_sub_u->putScalar(0.0);
      d_sub_u->doImport(*d_sub_u_over, *(solver->linalg->importer[0]), Tpetra::ADD);
    }
    else {
      d_sub_u = d_sub_u_over;
    }
    
  }
  
  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Finished SubGridDtN_Solver::computeSolnSens ..." << endl;
    }
  }
}

//////////////////////////////////////////////////////////////
// Figure out which views are needed to update the residual for the subgrid solution sensitivity wrt coarse DOFs
//////////////////////////////////////////////////////////////

template<class ResViewType, class DataViewType>
void SubGridDtN_Solver::updateResSens(const bool & use_groups, const int & macrogrp, const int & elem, ResViewType dres_view,
                                      DataViewType data, const bool & data_avail,
                                      const bool & use_host_LIDs, const bool & compute_sens) {
  
  typedef typename SubgridSolverNode::execution_space SG_exec;

  if (data_avail) {
    if (use_groups) {
      this->updateResSens(dres_view,data,assembler->groups[macrogrp][elem]->LIDs[0],compute_sens);
    }
    else {
      this->updateResSens(dres_view,data,assembler->boundary_groups[macrogrp][elem]->LIDs[0],compute_sens);
    }
  }
  else { // need to send assembly data to solver device
    auto data_sgladev = Kokkos::create_mirror(SG_exec(), data);
    Kokkos::deep_copy(data_sgladev,data);
    if (use_host_LIDs) { // copy already on host device
      if (use_groups) {
        this->updateResSens(dres_view, data_sgladev,assembler->groups[macrogrp][elem]->LIDs_host[0],compute_sens);
      }
      else {
        this->updateResSens(dres_view, data_sgladev,assembler->boundary_groups[macrogrp][elem]->LIDs_host[0],compute_sens);
      }
    }
    else { // solve on GPU, but assembly on CPU (not common)
      if (use_groups) {
        auto LIDs = assembler->groups[macrogrp][elem]->LIDs[0];
        auto LIDs_sgladev = Kokkos::create_mirror(SG_exec(), LIDs);
        Kokkos::deep_copy(LIDs_sgladev,LIDs);
        this->updateResSens(dres_view,data_sgladev,LIDs_sgladev,compute_sens);
      }
      else {
        auto LIDs = assembler->boundary_groups[macrogrp][elem]->LIDs[0];
        auto LIDs_sgladev = Kokkos::create_mirror(SG_exec(), LIDs);
        Kokkos::deep_copy(LIDs_sgladev,LIDs);
        this->updateResSens(dres_view,data_sgladev,LIDs_sgladev,compute_sens);
      }
    }
  }
}

//////////////////////////////////////////////////////////////
// Update the residual for the subgrid solution sensitivity wrt coarse DOFs
//////////////////////////////////////////////////////////////

template<class LIDViewType, class ResViewType, class DataViewType>
void SubGridDtN_Solver::updateResSens(ResViewType res, DataViewType data, LIDViewType LIDs, const bool & compute_sens ) {
  
  typedef typename SubgridSolverNode::execution_space SG_exec;

  if (compute_sens) {
    parallel_for("subgrid diagonal fix",
                 RangePolicy<SG_exec>(0,LIDs.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(elem,row);
        for (size_type col=0; col<data.extent(2); col++ ) {
          Kokkos::atomic_add(&(res(rowIndex,col)),data(elem,row,col));
        }
      }
    });
  }
  else {
    parallel_for("subgrid diagonal fix",
                 RangePolicy<SG_exec>(0,LIDs.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(elem,row);
        for (size_type col=0; col<data.extent(2); col++ ) {
          ScalarT mult = -1.0;
          Kokkos::atomic_add(&(res(rowIndex,col)),mult*data(elem,row,col));
        }
      }
    });
  
  }
  
}

//////////////////////////////////////////////////////////////
// Update the flux
//////////////////////////////////////////////////////////////

void SubGridDtN_Solver::updateFlux(const Teuchos::RCP<SG_MultiVector> & u,
                                   const Teuchos::RCP<SG_MultiVector> & d_u,
                                   Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                   const Teuchos::RCP<SG_MultiVector> & disc_params,
                                   const bool & compute_sens, const int macroelemindex,
                                   const ScalarT & time, workset & macrowkset,
                                   const int & macrogrp) {
  
  Teuchos::TimeMonitor localtimer(*sgfemFluxTimer);
  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Starting SubGridDtN_Solver::updateFlux (intermediate function) ..." << endl;
    }
  }
    
  typedef typename SubgridSolverNode::memory_space SGS_mem;
  
  auto u_kv = u->getLocalView<SubgridSolverNode::device_type>(Tpetra::Access::ReadWrite);
  auto du_kv = d_u->getLocalView<SubgridSolverNode::device_type>(Tpetra::Access::ReadWrite);
  
  // TMW: The discretized parameters are not fully enabled at the subgrid level
  //      This causes errors if there are no discretized parameters, so it is hacked for now.
  auto dp_kv = u_kv;
  //if (disc_params->getNumVectors() > 0) {
  //  dp_kv = disc_params->getLocalView<SubgridSolverNode::device_type>();
  //}
  {
    Teuchos::TimeMonitor localtimer(*sgfemTemplatedFluxTimer);
    
    if (Kokkos::SpaceAccessibility<AssemblyExec, SGS_mem>::accessible) { // can we avoid a copy?
      this->updateFlux(u_kv, du_kv, lambda, dp_kv, compute_sens, macroelemindex,
                       time, macrowkset, macrogrp);
    }
    else {
      auto u_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),u_kv);
      Kokkos::deep_copy(u_dev,u_kv);
      auto du_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),du_kv);
      Kokkos::deep_copy(du_dev,du_kv);
      auto dp_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),dp_kv);
      Kokkos::deep_copy(dp_dev,dp_kv);
      this->updateFlux(u_dev, du_dev, lambda, dp_dev, compute_sens, macroelemindex,
                       time, macrowkset, macrogrp);
    }
  }
  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Finished SubGridDtN_Solver::updateFlux (intermediate function) ..." << endl;
    }
  }
}

//////////////////////////////////////////////////////////////
// Update the flux
//////////////////////////////////////////////////////////////

template<class ViewType>
void SubGridDtN_Solver::updateFlux(ViewType u_kv,
                                   ViewType du_kv,
                                   Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                   ViewType dp_kv,
                                   const bool & compute_sens, const int macroelemindex,
                                   const ScalarT & time, workset & macrowkset,
                                   const int & macrogrp) {
  
  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Starting SubGridDtN_Solver::updateFlux ..." << endl;
    }
  }
  
  //macrowkset.resetResidual();
  macrowkset.reset();
  assembler->wkset[0]->isOnSide = true;
  for (size_t e=0; e<assembler->boundary_groups[macrogrp].size(); e++) {

    //if (assembler->boundary_groups[macrogrp][e]->sidename == "interior") {
      {
        Teuchos::TimeMonitor localwktimer(*sgfemFluxWksetTimer);
        assembler->boundary_groups[macrogrp][e]->updateData();
        assembler->boundary_groups[macrogrp][e]->updateWorksetBasis();
      }
      
      auto cwts = assembler->wkset[0]->wts_side;
      ScalarT h = 0.0;
      //assembler->wkset[0]->sidename = "interior";
      {
        Teuchos::TimeMonitor localcelltimer(*sgfemFluxCellTimer);
        assembler->boundary_groups[macrogrp][e]->computeFlux(u_kv, du_kv, dp_kv, lambda, time,
                                                          0, h, compute_sens);
      }
      
      {
        Teuchos::TimeMonitor localtimer(*sgfemAssembleFluxTimer);
        
        auto bMIDs = assembler->boundary_groups[macrogrp][e]->auxMIDs_dev;
        for (size_type n=0; n<macrowkset.offsets.extent(0); n++) {
          auto macrobasis_ip = assembler->boundary_groups[macrogrp][e]->auxside_basis[macrowkset.usebasis[n]];
          auto off = Kokkos::subview(macrowkset.offsets, n, Kokkos::ALL());
          auto flux = Kokkos::subview(assembler->wkset[0]->flux, Kokkos::ALL(), n, Kokkos::ALL());
          auto res = macrowkset.res;
          parallel_for("subgrid flux",
                       RangePolicy<AssemblyExec>(0,bMIDs.extent(0)),
                       KOKKOS_LAMBDA (const size_type c ) {
            for (size_type j=0; j<macrobasis_ip.extent(1); j++) {
              for (size_type i=0; i<macrobasis_ip.extent(2); i++) {
                AD val = macrobasis_ip(c,j,i)*flux(c,i)*cwts(c,i);
                Kokkos::atomic_add( &res(bMIDs(c),off(j)), val);
              }
            }
          });
        }
      }
    //}
  }
  assembler->wkset[0]->isOnSide = false;

  if (debug_level > 0) {
    if (solver->Comm->getRank() == 0) {
      cout << "**** Finished SubGridDtN_Solver::updateFlux ..." << endl;
    }
  }
}

//////////////////////////////////////////////////////////////
// Compute the initial values for the subgrid solution
//////////////////////////////////////////////////////////////

void SubGridDtN_Solver::setInitial(Teuchos::RCP<SG_MultiVector> & initial,
                                   const int & macrogrp, const bool & useadjoint) {
  
  // TODO :: BWR is this deprecated now?
  initial->putScalar(0.0);
  // TMW: uncomment if you need a nonzero initial condition
  //      right now, it slows everything down ... especially if using an L2-projection
  
  /*
   bool useL2proj = true;//settings->sublist("Solver").get<bool>("Project initial",true);
   
   if (useL2proj) {
   
   // Compute the L2 projection of the initial data into the discrete space
   Teuchos::RCP<SG_MultiVector> rhs = Teuchos::rcp(new SG_MultiVector(*overlapped_map,1)); // reset residual
   Teuchos::RCP<SG_CrsMatrix>  mass = Teuchos::rcp(new SG_CrsMatrix(Copy, *overlapped_map, -1)); // reset Jacobian
   Teuchos::RCP<SG_MultiVector> glrhs = Teuchos::rcp(new SG_MultiVector(*owned_map,1)); // reset residual
   Teuchos::RCP<SG_CrsMatrix>  glmass = Teuchos::rcp(new SG_CrsMatrix(Copy, *owned_map, -1)); // reset Jacobian
   
   
   //for (size_t block=0; block<groups.size(); ++block) {
   for (size_t e=0; e<groups[macrogrp].size(); e++) {
   int numElem = groups[macrogrp][e]->numElem;
   vector<vector<int> > GIDs = groups[macrogrp][e]->GIDs;
   Kokkos::View<ScalarT**,AssemblyDevice> localrhs = groups[macrogrp][e]->getInitial(true, useadjoint);
   Kokkos::View<ScalarT***,AssemblyDevice> localmass = groups[macrogrp][e]->getMass();
   
   // assemble into global matrix
   for (int c=0; c<numElem; c++) {
   for( size_t row=0; row<GIDs[c].size(); row++ ) {
   int rowIndex = GIDs[c][row];
   ScalarT val = localrhs(c,row);
   rhs->SumIntoGlobalValue(rowIndex,0, val);
   for( size_t col=0; col<GIDs[c].size(); col++ ) {
   int colIndex = GIDs[c][col];
   ScalarT val = localmass(c,row,col);
   mass->InsertGlobalValues(rowIndex, 1, &val, &colIndex);
   }
   }
   }
   }
   //}
   
   
   mass->FillComplete();
   glmass->PutScalar(0.0);
   glmass->Export(*mass, *exporter, Add);
   
   glrhs->PutScalar(0.0);
   glrhs->Export(*rhs, *exporter, Add);
   
   glmass->FillComplete();
   
   Teuchos::RCP<SG_MultiVector> glinitial = Teuchos::rcp(new SG_MultiVector(*overlapped_map,1)); // reset residual
   
   this->linearSolver(glmass, glrhs, glinitial);
   
   initial->Import(*glinitial, *importer, Add);
   
   }
   else {
   
   for (size_t e=0; e<groups[macrogrp].size(); e++) {
   int numElem = groups[macrogrp][e]->numElem;
   vector<vector<int> > GIDs = groups[macrogrp][e]->GIDs;
   Kokkos::View<ScalarT**,AssemblyDevice> localinit = groups[macrogrp][e]->getInitial(false, useadjoint);
   for (int c=0; c<numElem; c++) {
   for( size_t row=0; row<GIDs[c].size(); row++ ) {
   int rowIndex = GIDs[c][row];
   ScalarT val = localinit(c,row);
   initial->SumIntoGlobalValue(rowIndex,0, val);
   }
   }
   }
   
   }*/
  
}

////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> >  SubGridDtN_Solver::getProjectionMatrix() {
  
  // Compute the mass matrix on a reference element
  matrix_RCP mass = solver->linalg->getNewOverlappedMatrix(0);
  //Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(solver->LA_overlapped_graph) );
  
  auto localMatrix = mass->getLocalMatrixDevice();
  
  int macrogrp = 0;
  for (size_t e=0; e<assembler->groups[macrogrp].size(); e++) {
    LIDView LIDs = assembler->groups[macrogrp][e]->LIDs[0];
    Kokkos::View<ScalarT***,AssemblyDevice> localmass = assembler->groups[macrogrp][e]->getMass();
    
    size_type numVals = LIDs.extent(1);
    LO cols[maxDerivs];
    ScalarT vals[maxDerivs];
    for (size_type i=0; i<LIDs.extent(0); i++) { // this should be changed to a Kokkos::parallel_for on host
      for( size_type row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(i,row);
        // add check here for fixedDOF
        for( size_type col=0; col<LIDs.extent(1); col++ ) {
          vals[col] = localmass(i,row,col);
          cols[col] = LIDs(i,col);
        }
        localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, true, false); // isSorted, useAtomics
        // the LIDs are actually not sorted, but this appears to run a little faster
        
      }
    }
  }
  
  mass->fillComplete();
  
  matrix_RCP glmass;
  //size_t maxEntries = 256;
  if (solver->Comm->getSize() > 1) {
    glmass = solver->linalg->getNewMatrix(0);
    //Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(solver->LA_owned_map,maxEntries) );
    glmass->setAllToScalar(0.0);
    glmass->doExport(*mass, *(solver->linalg->exporter[0]), Tpetra::ADD); // TMW: tmp fix
    glmass->fillComplete();
  }
  else {
    glmass = mass;
  }
  return glmass;
}

////////////////////////////////////////////////////////////////////////////////
// Evaluate the basis functions at a set of points
////////////////////////////////////////////////////////////////////////////////

std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridDtN_Solver::evaluateBasis2(const DRV & pts) {
  
  size_t numpts = pts.extent(1);
  size_t dimpts = pts.extent(2);
  size_t numLIDs = assembler->groups[0][0]->LIDs[0].extent(1);
  Kokkos::View<int**,AssemblyDevice> owners("owners",numpts,2+numLIDs);
  
  for (size_t e=0; e<assembler->groups[0].size(); e++) {
    int numElem = assembler->groups[0][e]->numElem;
    DRV nodes = assembler->groups[0][e]->nodes;
    for (int c=0; c<numElem;c++) {
      //DRV refpts("refpts",1, numpts, dimpts);
      //Kokkos::DynRankView<int,PHX::Device> inRefCell("inRefCell",1,numpts);
      DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
      auto tmp0 = subview(cnodes,0,ALL(),ALL());
      auto tmp1 = subview(nodes,c,ALL(),ALL());
      Kokkos::deep_copy(tmp0,tmp1);
      //for (unsigned int i=0; i<nodes.extent(1); i++) {
      //  for (unsigned int j=0; j<nodes.extent(2); j++) {
      //    cnodes(0,i,j) = nodes(c,i,j);
      //  }
      //}
      
      Kokkos::DynRankView<int,PHX::Device> inRefCell = solver->disc->checkInclusionPhysicalData(pts, cnodes, solver->mesh->cellTopo[0], 1.0e-12);
      //CellTools::mapToReferenceFrame(refpts, pts, cnodes, *(solver->mesh->cellTopo[0]));
      //CellTools::checkPointwiseInclusion(inRefCell, refpts, *(solver->mesh->cellTopo[0]), 1.0e-12);
      for (size_t i=0; i<numpts; i++) {
        if (inRefCell(0,i) == 1) {
          owners(i,0) = e;//groups[0][e]->localElemID[c];
          owners(i,1) = c;
          LIDView LIDs = assembler->groups[0][e]->LIDs[0];
          for (size_t j=0; j<numLIDs; j++) {
            owners(i,j+2) = LIDs(c,j);
          }
        }
      }
    }
  }
  
  vector<DRV> ptsBasis;
  for (size_t i=0; i<numpts; i++) {
    vector<DRV> currBasis;
    //DRV refpt_buffer("refpt_buffer",1,1,dimpts);
    DRV cpt("cpt",1,1,dimpts);
    auto tmp0 = subview(cpt,0,0,ALL());
    auto tmp1 = subview(pts,0,i,ALL());
    Kokkos::deep_copy(tmp0,tmp1);
    //for (size_t s=0; s<dimpts; s++) {
    //  cpt(0,0,s) = pts(0,i,s);
    //}
    DRV nodes = assembler->groups[0][owners(i,0)]->nodes;
    DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
    auto tmp2 = subview(cnodes,0,ALL(),ALL());
    auto tmp3 = subview(nodes,owners(i,1),ALL(),ALL());
    Kokkos::deep_copy(tmp2,tmp3);
    //for (unsigned int k=0; k<nodes.extent(1); k++) {
    //  for (unsigned int j=0; j<nodes.extent(2); j++) {
    //    cnodes(0,k,j) = nodes(owners(i,1),k,j);
    //  }
    //}
    DRV refpt_buffer = solver->disc->mapPointsToReference(cpt,cnodes,solver->mesh->cellTopo[0]);
    //CellTools::mapToReferenceFrame(refpt_buffer, cpt, cnodes, *(solver->mesh->cellTopo[0]));
    DRV refpt("refpt",1,dimpts);
    Kokkos::deep_copy(refpt,Kokkos::subdynrankview(refpt_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
    Kokkos::View<int**,AssemblyDevice> offsets = assembler->wkset[0]->offsets;
    vector<int> usebasis = assembler->wkset[0]->usebasis;
    DRV basisvals("basisvals",offsets.extent(0),numLIDs);
    for (size_t n=0; n<offsets.extent(0); n++) {
      DRV bvals = solver->disc->evaluateBasis(solver->disc->basis_pointers[0][usebasis[n]], refpt);
      auto tmpb0 = subview(basisvals,n,ALL());
      auto tmpb1 = subview(bvals,0,ALL(),0);
      auto off = subview(offsets,n,ALL());
      parallel_for("ODE volume resid",
                   RangePolicy<AssemblyExec>(0,off.extent(0)),
                   KOKKOS_LAMBDA (const int m ) {
        tmpb0(off(m)) = tmpb1(m);
      });
    }
    ptsBasis.push_back(basisvals);
    
  }
  std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo(owners, ptsBasis);
  return basisinfo;
  
}


////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> > SubGridDtN_Solver::getProjectionMatrix(DRV & ip, DRV & wts,
                                                                  std::pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo) {
  
  std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > my_basisinfo = this->evaluateBasis2(ip);
  matrix_RCP map_over = solver->linalg->getNewOverlappedMatrix(0);
  //Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(solver->LA_overlapped_graph));
  
  matrix_RCP map;
  if (solver->Comm->getSize() > 1) {
    map = solver->linalg->getNewMatrix(0);
    //Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(solver->LA_overlapped_graph));
    map->setAllToScalar(0.0);
  }
  else {
    map = map_over;
  }
  
  Teuchos::Array<ScalarT> vals(1);
  Teuchos::Array<GO> cols(1);
  
  for (size_t k=0; k<ip.extent(1); k++) {
    for (size_t r=0; r<my_basisinfo.second[k].extent(0);r++) {
      for (size_t p=0; p<my_basisinfo.second[k].extent(1);p++) {
        int igid = my_basisinfo.first(k,p+2);
        for (size_t s=0; s<other_basisinfo.second[k].extent(0);s++) {
          for (size_t q=0; q<other_basisinfo.second[k].extent(1);q++) {
            cols[0] = other_basisinfo.first(k,q+2);
            if (r == s) {
              vals[0] = my_basisinfo.second[k](r,p) * other_basisinfo.second[k](s,q) * wts(0,k);
              map_over->sumIntoGlobalValues(igid, cols, vals);
            }
          }
        }
      }
    }
  }
  
  map_over->fillComplete();
  
  if (solver->Comm->getSize() > 1) {
    map->doExport(*map_over, *(solver->linalg->exporter[0]), Tpetra::ADD);
    map->fillComplete();
  }
  return map;
}

////////////////////////////////////////////////////////////////////////////////
// Get an empty vector
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> > SubGridDtN_Solver::getVector() {
  vector_RCP vec = solver->linalg->getNewOverlappedVector(0); //Teuchos::rcp(new SG_MultiVector(solver->LA_overlapped_map,1));
  return vec;
}

////////////////////////////////////////////////////////////////////////////////
// Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> >  SubGridDtN_Solver::getEvaluationMatrix(const DRV & newip, Teuchos::RCP<SG_Map> & ip_map) {
  matrix_RCP map_over = solver->linalg->getNewOverlappedMatrix(0);
  //Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(solver->LA_overlapped_graph) );
  matrix_RCP map;
  if (solver->Comm->getSize() > 1) {
    //size_t maxEntries = 256;
    map = solver->linalg->getNewMatrix(0);
    //Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(solver->LA_owned_map, maxEntries) );
    
    map->setAllToScalar(0.0);
    map->doExport(*map_over, *(solver->linalg->exporter[0]), Tpetra::ADD);
    map->fillComplete();
  }
  else {
    map = map_over;
  }
  return map;
}

////////////////////////////////////////////////////////////////////////////////
// Update the subgrid parameters (will be depracated)
////////////////////////////////////////////////////////////////////////////////

void SubGridDtN_Solver::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
  for (size_t block=0; block<assembler->wkset.size(); ++block) {
    assembler->wkset[block]->params = params;
    assembler->wkset[block]->paramnames = paramnames;
  }
  solver->phys->updateParameters(params, paramnames);
  
}

// ========================================================================================
//
// ========================================================================================

void SubGridDtN_Solver::performGather(const size_t & block, const vector_RCP & vec,
                                      const size_t & type, const size_t & entry) {
    
  typedef typename SubgridSolverNode::memory_space SGS_mem;
  
  auto vec_kv = vec->getLocalView<SubgridSolverNode::device_type>(Tpetra::Access::ReadWrite);
  auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), entry);
  
  if (Kokkos::SpaceAccessibility<AssemblyExec, SGS_mem>::accessible) { // can we avoid a copy?
    this->performGather(block, vec_slice, type);
    this->performBoundaryGather(block, vec_slice, type);
  }
  else {
    auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
    Kokkos::deep_copy(vec_dev,vec_slice);
    this->performGather(block, vec_dev, type);
    this->performBoundaryGather(block, vec_dev, type);
  }
  
}

// ========================================================================================
//
// ========================================================================================

template<class ViewType>
void SubGridDtN_Solver::performGather(const size_t & block, ViewType vec_dev, const size_t & type) {

  Kokkos::View<LO*,AssemblyDevice> numDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> data;
  Kokkos::View<int**,AssemblyDevice> offsets;
  LIDView LIDs;
  
  for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
    switch(type) {
      case 0 :
        numDOF = assembler->groups[block][grp]->groupData->numDOF;
        data = assembler->groups[block][grp]->u[0];
        LIDs = assembler->groups[block][grp]->LIDs[0];
        offsets = assembler->wkset[0]->offsets;
        break;
      case 1 : // deprecated (was udot)
        break;
      case 2 :
        numDOF = assembler->groups[block][grp]->groupData->numDOF;
        data = assembler->groups[block][grp]->phi[0];
        LIDs = assembler->groups[block][grp]->LIDs[0];
        offsets = assembler->wkset[0]->offsets;
        break;
      case 3 : // deprecated (was phidot)
        break;
      case 4:
        numDOF = assembler->groups[block][grp]->groupData->numParamDOF;
        data = assembler->groups[block][grp]->param;
        LIDs = assembler->groups[block][grp]->paramLIDs;
        offsets = assembler->wkset[0]->paramoffsets;
        break;
      case 5 :
        numDOF = assembler->groups[block][grp]->groupData->numAuxDOF;
        data = assembler->groups[block][grp]->aux;
        LIDs = assembler->groups[block][grp]->auxLIDs;
        offsets = assembler->groups[block][grp]->auxoffsets;
        break;
      default :
        cout << "ERROR - NOTHING WAS GATHERED" << endl;
    }
    
    parallel_for(RangePolicy<AssemblyExec>(0,data.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) {
        for (int i=0; i<numDOF(n); i++ ) {
          data(e,n,i) = vec_dev(LIDs(e,offsets(n,i)));
        }
      }
    });
  }
  
}

// ========================================================================================
//
// ========================================================================================

template<class ViewType>
void SubGridDtN_Solver::performBoundaryGather(const size_t & block, ViewType vec_dev, const size_t & type) {
  
  if (assembler->boundary_groups.size() > block) {
    
    Kokkos::View<LO*,AssemblyDevice> numDOF;
    Kokkos::View<ScalarT***,AssemblyDevice> data;
    Kokkos::View<int**,AssemblyDevice> offsets;
    LIDView LIDs;
    
    for (size_t grp=0; grp<assembler->boundary_groups[block].size(); ++grp) {
      if (assembler->boundary_groups[block][grp]->numElem > 0) {
        
        switch(type) {
          case 0 :
            numDOF = assembler->boundary_groups[block][grp]->groupData->numDOF;
            data = assembler->boundary_groups[block][grp]->u[0];
            LIDs = assembler->boundary_groups[block][grp]->LIDs[0];
            offsets = assembler->wkset[0]->offsets;
            break;
          case 1 : // deprecated (was udot)
            break;
          case 2 :
            numDOF = assembler->boundary_groups[block][grp]->groupData->numDOF;
            data = assembler->boundary_groups[block][grp]->phi[0];
            LIDs = assembler->boundary_groups[block][grp]->LIDs[0];
            offsets = assembler->wkset[0]->offsets;
            break;
          case 3 : // deprecated (was phidot)
            break;
          case 4:
            numDOF = assembler->boundary_groups[block][grp]->groupData->numParamDOF;
            data = assembler->boundary_groups[block][grp]->param;
            LIDs = assembler->boundary_groups[block][grp]->paramLIDs;
            offsets = assembler->wkset[0]->paramoffsets;
            break;
          case 5 :
            numDOF = assembler->boundary_groups[block][grp]->groupData->numAuxDOF;
            data = assembler->boundary_groups[block][grp]->aux;
            LIDs = assembler->boundary_groups[block][grp]->auxLIDs;
            offsets = assembler->boundary_groups[block][grp]->auxoffsets;
            break;
          default :
            cout << "ERROR - NOTHING WAS GATHERED" << endl;
        }
        
        parallel_for(RangePolicy<AssemblyExec>(0,data.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_type n=0; n<numDOF.extent(0); n++) {
            for(int i=0; i<numDOF(n); i++ ) {
              data(e,n,i) = vec_dev(LIDs(e,offsets(n,i)));
            }
          }
        });
      }
    }
  }
  
}

void SubGridDtN_Solver::resetGroupStorage( const string & target ) {

  if ( ! ( (target == "prev") || (target == "stage") ) ) {
    cout << "Error :: SubgridDtN_solver::resetGroupStorage received an unknown target : "
         << target << endl;
  }

  // loop over element block and groups and reset the desired solution storage    

  for (size_t block=0; block<assembler->groups.size(); ++block) {
    for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
      if (target == "prev") {
        assembler->groups[block][grp]->resetPrevSoln(0);
      } 
      else if (target == "stage") {
        assembler->groups[block][grp]->resetStageSoln(0);
      }     
    }
  }

}

void SubGridDtN_Solver::updateGroupStorage( const string & target ) {

  if ( !(target == "stage") ) {
    cout << "Error :: SubgridDtN_solver::updateGroupStorage received an unknown target : "
         << target << endl;
  }

  // loop over element block and groups and update/shift the stage solution storage

  for (size_t block=0; block<assembler->groups.size(); ++block) {
    for (size_t grp=0; grp<assembler->groups[block].size(); ++grp) {
      assembler->groups[block][grp]->updateStageSoln(0);
    }
  }

}