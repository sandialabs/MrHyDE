/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/
#include "solverInterface.hpp"

#include "subgridFEM_solver.hpp"

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


SubGridFEM_Solver::SubGridFEM_Solver(const Teuchos::RCP<MpiComm> & LocalComm,
                                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                     Teuchos::RCP<meshInterface> & mesh,
                                     Teuchos::RCP<discretization> & disc,
                                     Teuchos::RCP<physics> & physics,
                                     Teuchos::RCP<AssemblyManager> & assembler_,
                                     Teuchos::RCP<ParameterManager> & params,
                                     Teuchos::RCP<panzer::DOFManager> & DOF,
                                     ScalarT & macro_deltat_, size_t & numMacroDOF) :
settings(settings_), macro_deltat(macro_deltat_), assembler(assembler_) {
  
  dimension = settings->sublist("Mesh").get<int>("dim",2);
  multiscale_method = settings->get<string>("multiscale method","mortar");
  subgridverbose = settings->sublist("Solver").get<int>("verbosity",0);
  shape = settings->sublist("Mesh").get<string>("shape","quad");
  macroshape = settings->sublist("Mesh").get<string>("macro-shape","quad");
  time_steps = settings->sublist("Solver").get<int>("number of steps",1);
  initial_time = settings->sublist("Solver").get<ScalarT>("initial time",0.0);
  final_time = settings->sublist("Solver").get<ScalarT>("final time",1.0);
  
  have_sym_factor = false;
  sub_NLtol = settings->sublist("Solver").get<ScalarT>("nonlinear TOL",1.0E-12);
  sub_maxNLiter = settings->sublist("Solver").get<int>("max nonlinear iters",10);
  useDirect = settings->sublist("Solver").get<bool>("use direct solver",true);

  store_aux_and_flux = settings->sublist("Postprocess").get<bool>("store aux and flux",false);
  
  milo_solver = Teuchos::rcp( new solver(LocalComm, settings, mesh, disc, physics, DOF, assembler, params) );
  
  res = Teuchos::rcp( new LA_MultiVector(milo_solver->LA_owned_map,1)); // allocate residual
  J = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(milo_solver->LA_overlapped_graph));
  
  u = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_overlapped_map,1));
  phi = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_overlapped_map,1));
  
  if (LocalComm->getSize() > 1) {
    res_over = Teuchos::rcp( new LA_MultiVector(milo_solver->LA_overlapped_map,1)); // allocate residual
    sub_J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(milo_solver->LA_overlapped_graph));
  }
  else {
    res_over = res;
    sub_J_over = J;
  }
  
  d_um = Teuchos::rcp( new LA_MultiVector(milo_solver->LA_owned_map,numMacroDOF)); // reset residual
  d_sub_res_overm = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_overlapped_map,numMacroDOF));
  d_sub_resm = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_owned_map,numMacroDOF));
  d_sub_u_prevm = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_owned_map,numMacroDOF));
  d_sub_u_overm = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_overlapped_map,numMacroDOF));
  
  du_glob = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_owned_map,1));
  if (LocalComm->getSize() > 1) {
    du = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_overlapped_map,1));
  }
  else {
    du = du_glob;
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM_Solver::solve(Kokkos::View<ScalarT***,AssemblyDevice> coarse_u,
                              Kokkos::View<ScalarT***,AssemblyDevice> coarse_phi,
                              Teuchos::RCP<LA_MultiVector> & prev_u,
                              Teuchos::RCP<LA_MultiVector> & prev_phi,
                              //Teuchos::RCP<LA_MultiVector> & u,
                              //Teuchos::RCP<LA_MultiVector> & phi,
                              Teuchos::RCP<LA_MultiVector> & disc_params,
                              Teuchos::RCP<SubGridMacroData> & macroData,
                              const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                              const bool & compute_jacobian, const bool & compute_sens,
                              const int & num_active_params,
                              const bool & compute_disc_sens, const bool & compute_aux_sens,
                              workset & macrowkset,
                              const int & usernum, const int & macroelemindex,
                              Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) {
  
  Teuchos::TimeMonitor totalsolvertimer(*sgfemSolverTimer);
  
  ScalarT current_time = time;
  int macroDOF = macrowkset.numDOF;
  bool usesubadjoint = false;
  
  Kokkos::deep_copy(subgradient, 0.0);
  
  if (abs(current_time - final_time) < 1.0e-12)
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
  
  Kokkos::View<ScalarT***,AssemblyDevice> lambda = coarse_u;
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
  
  auto prev_u_kv = prev_u->getLocalView<HostDevice>();
  auto u_kv = u->getLocalView<HostDevice>();
  Kokkos::deep_copy(u_kv, prev_u_kv);
  
  this->performGather(usernum, prev_u, 0, 0);
  for (size_t b=0; b<assembler->cells.size(); b++) {
    for (size_t e=0; e<assembler->cells[b].size(); e++) {
      assembler->cells[b][e]->resetPrevSoln();
    }
  }
  
  //////////////////////////////////////////////////////////////
  // Use the coarse scale solution to solve local transient/nonlinear problem
  //////////////////////////////////////////////////////////////
  
  Teuchos::RCP<LA_MultiVector> d_u = d_um;
  if (compute_sens) {
    d_u = Teuchos::rcp( new LA_MultiVector(milo_solver->LA_owned_map, num_active_params)); // reset residual
  }
  d_u->putScalar(0.0);
  
  res->putScalar(0.0);
  
  ScalarT h = 0.0;
  //assembler->wkset[0]->resetFlux();
  
  if (isTransient) {
    ScalarT sgtime = 0.0;
    Teuchos::RCP<LA_MultiVector> prev_u = u;
    vector<Teuchos::RCP<LA_MultiVector> > curr_fsol;
    vector<ScalarT> subsolvetimes;
    subsolvetimes.push_back(sgtime);
    if (isAdjoint) {
      // First, we need to resolve the forward problem
      
      for (int tstep=0; tstep<time_steps; tstep++) {
        Teuchos::RCP<LA_MultiVector> recu = Teuchos::rcp( new LA_MultiVector(milo_solver->LA_overlapped_map,1)); // reset residual
        
        *recu = *u;
        sgtime += macro_deltat/(ScalarT)time_steps;
        subsolvetimes.push_back(sgtime);
        
        // set du/dt and \lambda
        alpha = (ScalarT)time_steps/macro_deltat;
        assembler->wkset[0]->alpha = alpha;
        assembler->wkset[0]->deltat= 1.0/alpha;
        assembler->wkset[0]->deltat_KV(0) = 1.0/alpha;
        
        Kokkos::View<ScalarT***,AssemblyDevice> currlambda = coarse_u;
        
        ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
        
        this->nonlinearSolver(recu, phi, disc_params, currlambda,
                              sgtime, isTransient, false, num_active_params, alpha, usernum, false);
        
        curr_fsol.push_back(recu);
        
      }
      
      for (int tstep=0; tstep<time_steps; tstep++) {
        
        size_t numsubtimes = subsolvetimes.size();
        size_t tindex = numsubtimes-1-tstep;
        sgtime = subsolvetimes[tindex];
        // set du/dt and \lambda
        alpha = (ScalarT)time_steps/macro_deltat;
        assembler->wkset[0]->alpha = alpha;
        assembler->wkset[0]->deltat= 1.0/alpha;
        assembler->wkset[0]->deltat_KV(0) = 1.0/alpha;
        
        Kokkos::View<ScalarT***,AssemblyDevice> currlambda = lambda;
        
        ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
        
        this->nonlinearSolver(curr_fsol[tindex-1], phi, disc_params, currlambda,
                              sgtime, isTransient, isAdjoint, num_active_params, alpha, usernum, store_adjPrev);
        
        this->computeSolnSens(d_u, compute_sens, curr_fsol[tindex-1],
                              phi, disc_params, currlambda,
                              sgtime, isTransient, isAdjoint, num_active_params, alpha, lambda_scale, usernum, subgradient);
        
        this->updateFlux(phi, d_u, lambda, disc_params, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0/(ScalarT)time_steps, macroData);
        
      }
    }
    else {
      for (int tstep=0; tstep<time_steps; tstep++) {
        sgtime += macro_deltat/(ScalarT)time_steps;
        // set du/dt and \lambda
        alpha = (ScalarT)time_steps/macro_deltat;
        
        assembler->wkset[0]->BDF_wts(0) = 1.0;//alpha;
        assembler->wkset[0]->BDF_wts(1) = -1.0;//-alpha;
        
        assembler->wkset[0]->alpha = alpha;
        assembler->wkset[0]->deltat= 1.0/alpha;
        assembler->wkset[0]->deltat_KV(0) = 1.0/alpha;
        
        Kokkos::View<ScalarT***,AssemblyDevice> currlambda = lambda;
        
        ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
        
        for (size_t b=0; b<assembler->cells.size(); b++) {
          for (size_t e=0; e<assembler->cells[b].size(); e++) {
            assembler->cells[b][e]->resetPrevSoln();
            assembler->cells[b][e]->resetStageSoln();
          }
        }
        
        this->nonlinearSolver(u, phi, disc_params, currlambda,
                              sgtime, isTransient, isAdjoint, num_active_params, alpha, usernum, false);
        
        
        this->computeSolnSens(d_u, compute_sens, u,
                              phi, disc_params, currlambda,
                              sgtime, isTransient, isAdjoint, num_active_params, alpha, lambda_scale, usernum, subgradient);
        
        this->updateFlux(u, d_u, lambda, disc_params, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0/(ScalarT)time_steps, macroData);
      }
    }
    
  }
  else {
    
    assembler->wkset[0]->deltat = 1.0;
    assembler->wkset[0]->deltat_KV(0) = 1.0;
    
    this->nonlinearSolver(u, phi, disc_params, lambda,
                          current_time, isTransient, isAdjoint, num_active_params, alpha, usernum, false);
    
    //KokkosTools::print(u);
    
    this->computeSolnSens(d_u, compute_sens, u,
                          phi, disc_params, lambda,
                          current_time, isTransient, isAdjoint, num_active_params, alpha, 1.0, usernum, subgradient);
    
    //KokkosTools::print(d_u);
    
    if (isAdjoint) {
      this->updateFlux(phi, d_u, lambda, disc_params, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0, macroData);
    }
    else {
      this->updateFlux(u, d_u, lambda, disc_params, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0, macroData);
    }
    
  }
  
  if (store_aux_and_flux) {
    this->storeFluxData(lambda, macrowkset.res);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Store macro-dofs and flux (for ML-based subgrid)
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM_Solver::storeFluxData(Kokkos::View<ScalarT***,AssemblyDevice> lambda, Kokkos::View<AD**,AssemblyDevice> flux) {
  
  int num_dof_lambda = lambda.extent(1)*lambda.extent(2);
  
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
    ofs << flux(e,0).val() << "  ";
    //}
    //}
    ofs << endl;
  }
  ofs.close();
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Subgrid Nonlinear Solver
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM_Solver::nonlinearSolver(Teuchos::RCP<LA_MultiVector> & sub_u,
                                               Teuchos::RCP<LA_MultiVector> & sub_phi,
                                               Teuchos::RCP<LA_MultiVector> & sub_params,
                                               Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                               const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                               const int & num_active_params, const ScalarT & alpha, const int & usernum,
                                               const bool & store_adjPrev) {
  
  
  Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverTimer);
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_scaled(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_initial(1);
  resnorm[0] = 10.0*sub_NLtol;
  resnorm_initial[0] = resnorm[0];
  resnorm_scaled[0] = resnorm[0];
  
  int iter = 0;
  Kokkos::View<ScalarT**,AssemblyDevice> aPrev;
  
  while (iter < sub_maxNLiter && resnorm_scaled[0] > sub_NLtol) {
    
    sub_J_over->resumeFill();
    
    sub_J_over->setAllToScalar(0.0);
    res_over->putScalar(0.0);
    
    assembler->wkset[0]->time = time;
    assembler->wkset[0]->isTransient = isTransient;
    assembler->wkset[0]->isAdjoint = isAdjoint;
    
    int numElem = assembler->cells[usernum][0]->numElem;
    int maxElem = assembler->cells[0][0]->numElem;
    int numDOF = assembler->cells[usernum][0]->LIDs.extent(1);
    
    Kokkos::View<ScalarT***,AssemblyDevice> local_res, local_J;
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverAllocateTimer);
      local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,numDOF,1);
      local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,numDOF,numDOF);
    }
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSetSolnTimer);
      this->performGather(usernum, sub_u, 0, 0);
      if (isAdjoint) {
        this->performGather(usernum, sub_phi, 2, 0);
      }
      //this->performGather(usernum, sub_params, 4, 0);
      
      this->performBoundaryGather(usernum, sub_u, 0, 0);
      if (isAdjoint) {
        this->performBoundaryGather(usernum, sub_phi, 2, 0);
      }
      
      for (size_t e=0; e < assembler->boundaryCells[usernum].size(); e++) {
        assembler->boundaryCells[usernum][e]->aux = lambda;
      }
    }
    
    ////////////////////////////////////////////////
    // volume assembly
    ////////////////////////////////////////////////
    auto localMatrix = sub_J_over->getLocalMatrix();
    
    for (size_t e=0; e<assembler->cells[usernum].size(); e++) {
      if (isAdjoint) {
        if (is_final_time) {
          Kokkos::deep_copy(assembler->cells[usernum][e]->adj_prev, 0.0);
        }
      }
      
      assembler->wkset[0]->localEID = e;
      assembler->cells[usernum][e]->updateData();
      
      Kokkos::deep_copy(local_res, 0.0);
      Kokkos::deep_copy(local_J, 0.0);
      
      {
        Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverJacResTimer);
        assembler->cells[usernum][e]->computeJacRes(time, isTransient, isAdjoint,
                                              true, false, num_active_params, false, false, false,
                                              local_res, local_J,
                                              assembler->assemble_volume_terms[0],
                                              assembler->assemble_face_terms[0]);
        
      }
      
      //KokkosTools::print(local_res);
      //KokkosTools::print(local_J);
      {
        Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverInsertTimer);
        LIDView LIDs = assembler->cells[usernum][e]->LIDs;
        LO numentries = static_cast<LO>(LIDs.extent(1));
        ScalarT vals[numentries];
        LO cols[numentries];
        for (unsigned int i=0; i<LIDs.extent(0); i++) { // should be Kokkos::parallel_for on SubgridExec
          for( size_t row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(i,row);
            ScalarT val = local_res(i,row,0);
            res_over->sumIntoLocalValue(rowIndex,0, val);
            for( size_t col=0; col<numentries; col++ ) {
              vals[col] = local_J(i,row,col);
              cols[col] = LIDs(i,col);
            }
            localMatrix.sumIntoValues(rowIndex, cols, numentries, vals, true, false); // bools: isSorted, useAtomics
            // indices are not actually sorted, but this seems to run faster
            // may need to set useAtomics = true if subgridexec is not Serial
          }
        }
      }
    }
    
    ////////////////////////////////////////////////
    // boundary assembly
    ////////////////////////////////////////////////
    
    for (size_t e=0; e<assembler->boundaryCells[usernum].size(); e++) {
      
      if (assembler->boundaryCells[usernum][e]->numElem > 0) {
        assembler->wkset[0]->localEID = e;
        
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverAllocateTimer);
          local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",assembler->boundaryCells[usernum][e]->numElem,numDOF,1);
          local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",assembler->boundaryCells[usernum][e]->numElem,numDOF,numDOF);
        }
        
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverJacResTimer);
          
          assembler->boundaryCells[usernum][e]->computeJacRes(time, isTransient, isAdjoint,
                                                              true, false, num_active_params,
                                                              false, false, false,
                                                              local_res, local_J);
          
        }
        
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverInsertTimer);
          LIDView LIDs = assembler->boundaryCells[usernum][e]->LIDs;
          LO numentries = static_cast<LO>(LIDs.extent(1));
          ScalarT vals[numentries];
          LO cols[numentries];
          for (unsigned int i=0; i<LIDs.extent(0); i++) { // should be Kokkos::parallel_for on SubgridExec
            for( size_t row=0; row<LIDs.extent(1); row++ ) {
              LO rowIndex = LIDs(i,row);
              ScalarT val = local_res(i,row,0);
              res_over->sumIntoLocalValue(rowIndex,0, val);
              for( size_t col=0; col<numentries; col++ ) {
                vals[col] = local_J(i,row,col);
                cols[col] = LIDs(i,col);
              }
              localMatrix.sumIntoValues(rowIndex, cols, numentries, vals, true, false); // bools: isSorted, useAtomics
              // indices are not actually sorted, but this seems to run faster
              // may need to set useAtomics = true if subgridexec is not Serial
            }
          }
        }
      }
    }
    
    if (maxElem > numElem) {
      LIDView LIDs = assembler->cells[0][0]->LIDs;
      ScalarT vals[1];
      LO cols[1];
      for (unsigned int i=numElem; i<LIDs.extent(0); i++) { // should be Kokkos::parallel_for on SubgridExec
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(i,row);
          vals[0] = 1.0;
          cols[0] = rowIndex;
          localMatrix.sumIntoValues(rowIndex, cols, 1, vals, true, false); // bools: isSorted, useAtomics
        }
      }
    }
    
    sub_J_over->fillComplete();
    
    if (milo_solver->Comm->getSize() > 1) {
      J->resumeFill();
      J->setAllToScalar(0.0);
      J->doExport(*sub_J_over, *(milo_solver->exporter), Tpetra::ADD);
      J->fillComplete();
    }
    else {
      J = sub_J_over;
    }
    //KokkosTools::print(J);
    
    
    if (milo_solver->Comm->getSize() > 1) {
      res->putScalar(0.0);
      res->doExport(*res_over, *(milo_solver->exporter), Tpetra::ADD);
    }
    else {
      res = res_over;
    }
    
    if (useDirect) {
      if (have_sym_factor) {
        Am2Solver->setA(J, Amesos2::SYMBFACT);
        Am2Solver->setX(du_glob);
        Am2Solver->setB(res);
      }
      else {
        Am2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, du_glob, res);
        Am2Solver->symbolicFactorization();
        have_sym_factor = true;
      }
      //Am2Solver->numericFactorization().solve();
    }
    //KokkosTools::print(res);
    
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
    if(milo_solver->Comm->getRank() == 0 && subgridverbose>5) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Subgrid Nonlinear Iteration: " << iter << endl;
      cout << "***** Scaled Norm of nonlinear residual: " << resnorm_scaled << endl;
      cout << "*********************************************************" << endl;
    }
    
    if (resnorm_scaled[0] > sub_NLtol) {
      
      Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSolveTimer);
      du_glob->putScalar(0.0);
      if (useDirect) {
        Am2Solver->numericFactorization().solve();
      }
      else {
        if (have_belos) {
          //belos_problem->setProblem(du_glob, res);
        }
        else {
          belos_problem = Teuchos::rcp(new LA_LinearProblem(J, du_glob, res));
          have_belos = true;
          
          belosList = Teuchos::rcp(new Teuchos::ParameterList());
          belosList->set("Maximum Iterations",    50); // Maximum number of iterations allowed
          belosList->set("Convergence Tolerance", 1.0E-10);    // Relative convergence tolerance requested
          belosList->set("Verbosity", Belos::Errors);
          belosList->set("Output Frequency",0);
          
          //belosList->set("Verbosity", Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
          //belosList->set("Output Frequency",10);
          
          int numEqns = milo_solver->numVars[0];
          belosList->set("number of equations",numEqns);
          
          belosList->set("Output Style",          Belos::Brief);
          belosList->set("Implicit Residual Scaling", "None");
          
          belos_solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT, LA_MultiVector, LA_Operator>(belos_problem, belosList));
          
        }
        if (have_preconditioner) {
          //MueLu::ReuseTpetraPreconditioner(J,*belos_M);
        }
        else {
          belos_M = milo_solver->buildPreconditioner(J);
          //belos_problem->setRightPrec(belos_M);
          belos_problem->setLeftPrec(belos_M);
          have_preconditioner = true;
          
        }
        belos_problem->setProblem(du_glob, res);
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSolveTimer);
          belos_solver->solve();
          
        }
        //milo_solver->linearSolver(J,res,du_glob);
      }
      if (milo_solver->Comm->getSize() > 1) {
        du->putScalar(0.0);
        du->doImport(*du_glob, *(milo_solver->importer), Tpetra::ADD);
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
  
}

//////////////////////////////////////////////////////////////
// Compute the derivative of the local solution w.r.t coarse
// solution or w.r.t parameters
//////////////////////////////////////////////////////////////

void SubGridFEM_Solver::computeSolnSens(Teuchos::RCP<LA_MultiVector> & d_sub_u,
                                               const bool & compute_sens,
                                               Teuchos::RCP<LA_MultiVector> & sub_u,
                                               Teuchos::RCP<LA_MultiVector> & sub_phi,
                                               Teuchos::RCP<LA_MultiVector> & sub_param,
                                               Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                               const ScalarT & time,
                                               const bool & isTransient, const bool & isAdjoint,
                                               const int & num_active_params, const ScalarT & alpha,
                                               const ScalarT & lambda_scale, const int & usernum,
                                               Kokkos::View<ScalarT**,AssemblyDevice> subgradient) {
  
  Teuchos::TimeMonitor localtimer(*sgfemSolnSensTimer);
  
  Teuchos::RCP<LA_MultiVector> d_sub_res_over = d_sub_res_overm;
  Teuchos::RCP<LA_MultiVector> d_sub_res = d_sub_resm;
  Teuchos::RCP<LA_MultiVector> d_sub_u_prev = d_sub_u_prevm;
  Teuchos::RCP<LA_MultiVector> d_sub_u_over = d_sub_u_overm;
  
  if (compute_sens) {
    int numsubDerivs = d_sub_u->getNumVectors();
    d_sub_res_over = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_overlapped_map,numsubDerivs));
    d_sub_res = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_owned_map,numsubDerivs));
    d_sub_u_prev = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_owned_map,numsubDerivs));
    d_sub_u_over = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_overlapped_map,numsubDerivs));
  }
  
  d_sub_res_over->putScalar(0.0);
  d_sub_res->putScalar(0.0);
  d_sub_u_prev->putScalar(0.0);
  d_sub_u_over->putScalar(0.0);
  
  ScalarT scale = -1.0*lambda_scale;
  
  if (multiscale_method != "mortar") {
    this->performGather(usernum, sub_u, 0, 0);
    if (isAdjoint) {
      this->performGather(usernum, sub_phi, 2, 0);
    }
    for (size_t e=0; e < assembler->cells[usernum].size(); e++) {
      assembler->cells[usernum][e]->aux = lambda;
    }
  }
  else {
    this->performBoundaryGather(usernum, sub_u, 0, 0);
    if (isAdjoint) {
      this->performBoundaryGather(usernum, sub_phi, 2, 0);
    }
    for (size_t e=0; e < assembler->boundaryCells[usernum].size(); e++) {
      assembler->boundaryCells[usernum][e]->aux = lambda;
    }
  }
  //this->performGather(usernum, sub_param, 4, 0);
  
  if (compute_sens) {
    
    //this->sacadoizeParams(true, num_active_params);
    assembler->wkset[0]->time = time;
    assembler->wkset[0]->isTransient = isTransient;
    assembler->wkset[0]->isAdjoint = isAdjoint;
    
    if (multiscale_method != "mortar") {
      int numElem = assembler->cells[usernum][0]->numElem;
      
      int snumDOF = assembler->cells[usernum][0]->LIDs.extent(1);
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_res("local residual",numElem,snumDOF,num_active_params);
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_J("local Jacobian",numElem,snumDOF,snumDOF);
      
      for (size_t e=0; e<assembler->cells[usernum].size(); e++) {
        
        assembler->wkset[0]->localEID = e;
        assembler->cells[usernum][e]->updateData();
        
        Kokkos::deep_copy(local_res, 0.0);
        Kokkos::deep_copy(local_J, 0.0);
        
        assembler->cells[usernum][e]->computeJacRes(time, isTransient, isAdjoint,
                                   false, true, num_active_params, false, false, false,
                                   local_res, local_J,
                                   assembler->assemble_volume_terms[0],
                                   assembler->assemble_face_terms[0]);
        
        LIDView LIDs = assembler->cells[usernum][e]->LIDs;
        for (unsigned int i=0; i<LIDs.extent(0); i++) {
          for( size_t row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(i,row);
            for( size_t col=0; col<num_active_params; col++ ) {
              ScalarT val = local_res(i,row,col);
              d_sub_res_over->sumIntoLocalValue(rowIndex,col, 1.0*val);
            }
          }
        }
      }
      auto sub_phi_kv = sub_phi->getLocalView<HostDevice>();
      auto d_sub_res_over_kv = d_sub_res_over->getLocalView<HostDevice>();
      
      for (int p=0; p<num_active_params; p++) {
        for (int i=0; i<sub_phi->getGlobalLength(); i++) {
          subgradient(p,0) += sub_phi_kv(i,0) * d_sub_res_over_kv(i,p);
        }
      }
    }
    else {
      
      for (size_t e=0; e<assembler->boundaryCells[usernum].size(); e++) {
        int numElem = assembler->boundaryCells[usernum][e]->numElem;
        int snumDOF = assembler->boundaryCells[usernum][e]->LIDs.extent(1);
        
        Kokkos::View<ScalarT***,AssemblyDevice> local_res("local residual",numElem,snumDOF,num_active_params);
        
        Kokkos::View<ScalarT***,AssemblyDevice> local_J("local Jacobian",numElem,snumDOF,snumDOF);
        
        assembler->wkset[0]->localEID = e;
        //wkset[0]->var_bcs = subgridbcs[usernum];
        
        assembler->cells[usernum][e]->updateData();
        
        assembler->boundaryCells[usernum][e]->computeJacRes(time, isTransient, isAdjoint,
                                                            false, true, num_active_params, false, false, false,
                                                            local_res, local_J);
        
        LIDView LIDs = assembler->boundaryCells[usernum][e]->LIDs;
        for (unsigned int i=0; i<LIDs.extent(0); i++) {
          for (size_t row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(i,row);
            for (size_t col=0; col<num_active_params; col++ ) {
              ScalarT val = local_res(i,row,col);
              d_sub_res_over->sumIntoLocalValue(rowIndex,col, 1.0*val);
            }
          }
        }
      }
      auto sub_phi_kv = sub_phi->getLocalView<HostDevice>();
      auto d_sub_res_over_kv = d_sub_res_over->getLocalView<HostDevice>();
      
      for (int p=0; p<num_active_params; p++) {
        for (int i=0; i<sub_phi->getGlobalLength(); i++) {
          subgradient(p,0) += sub_phi_kv(i,0) * d_sub_res_over_kv(i,p);
        }
      }
    }
  }
  else {
    assembler->wkset[0]->time = time;
    assembler->wkset[0]->isTransient = isTransient;
    assembler->wkset[0]->isAdjoint = isAdjoint;
    
    Kokkos::View<ScalarT***,AssemblyDevice> local_res, local_J;
    
    if (multiscale_method != "mortar") {
      
      for (size_t e=0; e<assembler->cells[usernum].size(); e++) {
        
        int numElem = assembler->cells[usernum][e]->numElem;
        int snumDOF = assembler->cells[usernum][e]->LIDs.extent(1);
        int anumDOF = assembler->cells[usernum][e]->auxLIDs.extent(1);
        
        local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,snumDOF,1);
        local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,snumDOF,anumDOF);
        
        assembler->wkset[0]->localEID = e;
        
        // TMW: this may not work properly with new version
        assembler->cells[usernum][e]->updateData();
        
        assembler->cells[usernum][e]->computeJacRes(time, isTransient, isAdjoint,
                                                    true, false, num_active_params, false, true, false,
                                                    local_res, local_J,
                                                    assembler->assemble_volume_terms[0],
                                                    assembler->assemble_face_terms[0]);
        LIDView LIDs = assembler->cells[usernum][e]->LIDs;
        LIDView aLIDs = assembler->cells[usernum][e]->auxLIDs;
        //vector<vector<int> > aoffsets = cells[0][e]->auxoffsets;
        
        for (unsigned int i=0; i<LIDs.extent(0); i++) {
          for (size_t row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(i,row);
            for (size_t col=0; col<aLIDs.extent(1); col++ ) {
              ScalarT val = local_J(i,row,col);
              int colIndex = col;
              d_sub_res_over->sumIntoLocalValue(rowIndex,colIndex, scale*val);
            }
          }
        }
      }
    }
    else {
      
      for (size_t e=0; e<assembler->boundaryCells[usernum].size(); e++) {
        
        int numElem = assembler->boundaryCells[usernum][e]->numElem;
        int snumDOF = assembler->boundaryCells[usernum][e]->LIDs.extent(1);
        int anumDOF = assembler->boundaryCells[usernum][e]->auxLIDs.extent(1);
        
        local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,snumDOF,1);
        local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,snumDOF,anumDOF);
        
        assembler->boundaryCells[usernum][e]->computeJacRes(time, isTransient, isAdjoint,
                                                            true, false, num_active_params, false, true, false,
                                                            local_res, local_J);
        LIDView LIDs = assembler->boundaryCells[usernum][e]->LIDs;
        LIDView aLIDs = assembler->boundaryCells[usernum][e]->auxLIDs;
        for (unsigned int i=0; i<LIDs.extent(0); i++) {
          for (size_t row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(i,row);
            for (size_t col=0; col<aLIDs.extent(1); col++ ) {
              ScalarT val = local_J(i,row,col);
              LO colIndex = col;
              d_sub_res_over->sumIntoLocalValue(rowIndex,colIndex, scale*val);
            }
          }
        }
      }
    }
    
    if (milo_solver->Comm->getSize() > 1) {
      d_sub_res->doExport(*d_sub_res_over, *(milo_solver->exporter), Tpetra::ADD);
    }
    else {
      d_sub_res = d_sub_res_over;
    }
    
    if (useDirect) {
      
      Teuchos::TimeMonitor localtimer(*sgfemSolnSensLinearSolverTimer);
      
      int numsubDerivs = d_sub_u_over->getNumVectors();
      
      auto d_sub_u_over_kv = d_sub_u_over->getLocalView<HostDevice>();
      auto d_sub_res_kv = d_sub_res->getLocalView<HostDevice>();
      for (int c=0; c<numsubDerivs; c++) {
        Teuchos::RCP<LA_MultiVector> x = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_overlapped_map,1));
        Teuchos::RCP<LA_MultiVector> b = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_owned_map,1));
        auto b_kv = b->getLocalView<HostDevice>();
        auto x_kv = x->getLocalView<HostDevice>();
        
        for (int i=0; i<b->getGlobalLength(); i++) {
          b_kv(i,0) += d_sub_res_kv(i,c);
        }
        Am2Solver->setX(x);
        Am2Solver->setB(b);
        Am2Solver->solve();
        
        for (int i=0; i<x->getGlobalLength(); i++) {
          d_sub_u_over_kv(i,c) += x_kv(i,0);
        }
        
      }
    }
    else {
      
      Teuchos::TimeMonitor localtimer(*sgfemSolnSensLinearSolverTimer);
      
      belos_problem->setProblem(d_sub_u_over, d_sub_res);
      belos_solver->solve();
      //milo_solver->linearSolver(J,d_sub_res,d_sub_u_over);
    }
    
    if (milo_solver->Comm->getSize() > 1) {
      d_sub_u->putScalar(0.0);
      d_sub_u->doImport(*d_sub_u_over, *(milo_solver->importer), Tpetra::ADD);
    }
    else {
      d_sub_u = d_sub_u_over;
    }
    
  }
}

//////////////////////////////////////////////////////////////
// Update the flux
//////////////////////////////////////////////////////////////

void SubGridFEM_Solver::updateFlux(const Teuchos::RCP<LA_MultiVector> & u,
                                   const Teuchos::RCP<LA_MultiVector> & d_u,
                                   Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                   const Teuchos::RCP<LA_MultiVector> & disc_params,
                                   const bool & compute_sens, const int macroelemindex,
                                   const ScalarT & time, workset & macrowkset,
                                   const int & usernum, const ScalarT & fwt,
                                   Teuchos::RCP<SubGridMacroData> & macroData) {
  
  Teuchos::TimeMonitor localtimer(*sgfemFluxTimer);
  
  //this->updateLocalData(usernum);
  
  for (size_t e=0; e<assembler->boundaryCells[usernum].size(); e++) {
    
    if (assembler->boundaryCells[usernum][e]->sidename == "interior") {
      {
        Teuchos::TimeMonitor localwktimer(*sgfemFluxWksetTimer);
        assembler->boundaryCells[usernum][e]->updateWorksetBasis();
      }
      
      DRV cwts = assembler->wkset[0]->wts_side;
      ScalarT h = 0.0;
      assembler->wkset[0]->sidename = "interior";
      {
        Teuchos::TimeMonitor localcelltimer(*sgfemFluxCellTimer);
        assembler->boundaryCells[usernum][e]->computeFlux(u, d_u, disc_params, lambda, time,
                                                          0, h, compute_sens);
      }
      
      vector<size_t> bMIDs = assembler->boundaryCells[usernum][e]->auxMIDs;//localData->boundaryMIDs[e];
      for (int c=0; c<assembler->boundaryCells[usernum][e]->numElem; c++) {
        for (int n=0; n<macrowkset.offsets.extent(0); n++) {
          DRV macrobasis_ip = assembler->boundaryCells[usernum][e]->auxside_basis[macrowkset.usebasis[n]];
          for (unsigned int j=0; j<macrobasis_ip.extent(1); j++) {
            for (unsigned int i=0; i<macrobasis_ip.extent(2); i++) {
              macrowkset.res(bMIDs[c],macrowkset.offsets(n,j)) += macrobasis_ip(c,j,i)*(assembler->wkset[0]->flux(c,n,i))*cwts(c,i)*fwt;
            }
          }
        }
      }
    }
  }
  
}

//////////////////////////////////////////////////////////////
// Compute the initial values for the subgrid solution
//////////////////////////////////////////////////////////////

void SubGridFEM_Solver::setInitial(Teuchos::RCP<LA_MultiVector> & initial,
                                   const int & usernum, const bool & useadjoint) {
  
  initial->putScalar(0.0);
  // TMW: uncomment if you need a nonzero initial condition
  //      right now, it slows everything down ... especially if using an L2-projection
  
  /*
   bool useL2proj = true;//settings->sublist("Solver").get<bool>("Project initial",true);
   
   if (useL2proj) {
   
   // Compute the L2 projection of the initial data into the discrete space
   Teuchos::RCP<LA_MultiVector> rhs = Teuchos::rcp(new LA_MultiVector(*overlapped_map,1)); // reset residual
   Teuchos::RCP<LA_CrsMatrix>  mass = Teuchos::rcp(new LA_CrsMatrix(Copy, *overlapped_map, -1)); // reset Jacobian
   Teuchos::RCP<LA_MultiVector> glrhs = Teuchos::rcp(new LA_MultiVector(*owned_map,1)); // reset residual
   Teuchos::RCP<LA_CrsMatrix>  glmass = Teuchos::rcp(new LA_CrsMatrix(Copy, *owned_map, -1)); // reset Jacobian
   
   
   //for (size_t b=0; b<cells.size(); b++) {
   for (size_t e=0; e<cells[usernum].size(); e++) {
   int numElem = cells[usernum][e]->numElem;
   vector<vector<int> > GIDs = cells[usernum][e]->GIDs;
   Kokkos::View<ScalarT**,AssemblyDevice> localrhs = cells[usernum][e]->getInitial(true, useadjoint);
   Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[usernum][e]->getMass();
   
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
   
   Teuchos::RCP<LA_MultiVector> glinitial = Teuchos::rcp(new LA_MultiVector(*overlapped_map,1)); // reset residual
   
   this->linearSolver(glmass, glrhs, glinitial);
   
   initial->Import(*glinitial, *importer, Add);
   
   }
   else {
   
   for (size_t e=0; e<cells[usernum].size(); e++) {
   int numElem = cells[usernum][e]->numElem;
   vector<vector<int> > GIDs = cells[usernum][e]->GIDs;
   Kokkos::View<ScalarT**,AssemblyDevice> localinit = cells[usernum][e]->getInitial(false, useadjoint);
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

Teuchos::RCP<LA_CrsMatrix>  SubGridFEM_Solver::getProjectionMatrix() {
  
  // Compute the mass matrix on a reference element
  matrix_RCP mass = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(milo_solver->LA_overlapped_graph) );
  
  auto localMatrix = mass->getLocalMatrix();
  
  int usernum = 0;
  for (size_t e=0; e<assembler->cells[usernum].size(); e++) {
    LIDView LIDs = assembler->cells[usernum][e]->LIDs;
    Kokkos::View<ScalarT***,AssemblyDevice> localmass = assembler->cells[usernum][e]->getMass();
    
    const int numVals = static_cast<int>(LIDs.extent(1));
    LO cols[numVals];
    ScalarT vals[numVals];
    for (int i=0; i<LIDs.extent(0); i++) { // this should be changed to a Kokkos::parallel_for on host
      for( size_t row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(i,row);
        // add check here for fixedDOF
        for( size_t col=0; col<LIDs.extent(1); col++ ) {
          vals[col] = localmass(i,row,col);
          cols[col] = LIDs(i,col);
        }
        localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, true, false); // isSorted, useAtomics
        // the LIDs are actually not sorted, but this appears to run a little faster
        
      }
    }
    /*
     Kokkos::View<GO**,HostDevice> GIDs = cells[usernum][e]->GIDs;
     Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[usernum][e]->getMass();
     for (int c=0; c<numElem; c++) {
     for( size_t row=0; row<GIDs.extent(1); row++ ) {
     GO rowIndex = GIDs(c,row);
     for( size_t col=0; col<GIDs.extent(1); col++ ) {
     GO colIndex = GIDs(c,col);
     ScalarT val = localmass(c,row,col);
     mass->insertGlobalValues(rowIndex, 1, &val, &colIndex);
     }
     }
     }*/
  }
  
  mass->fillComplete();
  
  matrix_RCP glmass;
  size_t maxEntries = 256;
  if (milo_solver->Comm->getSize() > 1) {
    glmass = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(milo_solver->LA_owned_map,maxEntries) );
    glmass->setAllToScalar(0.0);
    glmass->doExport(*mass, *(milo_solver->exporter), Tpetra::ADD);
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

pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridFEM_Solver::evaluateBasis2(const DRV & pts) {
  
  size_t numpts = pts.extent(1);
  size_t dimpts = pts.extent(2);
  size_t numLIDs = assembler->cells[0][0]->LIDs.extent(1);
  Kokkos::View<int**,AssemblyDevice> owners("owners",numpts,2+numLIDs);
  
  for (size_t e=0; e<assembler->cells[0].size(); e++) {
    int numElem = assembler->cells[0][e]->numElem;
    DRV nodes = assembler->cells[0][e]->nodes;
    for (int c=0; c<numElem;c++) {
      DRV refpts("refpts",1, numpts, dimpts);
      DRVint inRefCell("inRefCell",1,numpts);
      DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
      for (unsigned int i=0; i<nodes.extent(1); i++) {
        for (unsigned int j=0; j<nodes.extent(2); j++) {
          cnodes(0,i,j) = nodes(c,i,j);
        }
      }
      
      CellTools::mapToReferenceFrame(refpts, pts, cnodes, *(milo_solver->mesh->cellTopo[0]));
      CellTools::checkPointwiseInclusion(inRefCell, refpts, *(milo_solver->mesh->cellTopo[0]), 1.0e-12);
      //KokkosTools::print(refpts);
      //KokkosTools::print(inRefCell);
      for (size_t i=0; i<numpts; i++) {
        if (inRefCell(0,i) == 1) {
          owners(i,0) = e;//cells[0][e]->localElemID[c];
          owners(i,1) = c;
          LIDView LIDs = assembler->cells[0][e]->LIDs;
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
    DRV refpt_buffer("refpt_buffer",1,1,dimpts);
    DRV cpt("cpt",1,1,dimpts);
    for (size_t s=0; s<dimpts; s++) {
      cpt(0,0,s) = pts(0,i,s);
    }
    DRV nodes = assembler->cells[0][owners(i,0)]->nodes;
    DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
    for (unsigned int k=0; k<nodes.extent(1); k++) {
      for (unsigned int j=0; j<nodes.extent(2); j++) {
        cnodes(0,k,j) = nodes(owners(i,1),k,j);
      }
    }
    CellTools::mapToReferenceFrame(refpt_buffer, cpt, cnodes, *(milo_solver->mesh->cellTopo[0]));
    DRV refpt("refpt",1,dimpts);
    Kokkos::deep_copy(refpt,Kokkos::subdynrankview(refpt_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
    Kokkos::View<int**,AssemblyDevice> offsets = assembler->wkset[0]->offsets;
    vector<int> usebasis = assembler->wkset[0]->usebasis;
    DRV basisvals("basisvals",offsets.extent(0),numLIDs);
    for (size_t n=0; n<offsets.extent(0); n++) {
      DRV bvals = milo_solver->disc->evaluateBasis(milo_solver->disc->basis_pointers[0][usebasis[n]], refpt);
      for (size_t m=0; m<offsets.extent(1); m++) {
        basisvals(n,offsets(n,m)) = bvals(0,m,0);
      }
    }
    ptsBasis.push_back(basisvals);
    
  }
  pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo(owners, ptsBasis);
  return basisinfo;
  
}


////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<LA_CrsMatrix> SubGridFEM_Solver::getProjectionMatrix(DRV & ip, DRV & wts,
                                                                  pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo) {
  
  pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > my_basisinfo = this->evaluateBasis2(ip);
  matrix_RCP map_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(milo_solver->LA_overlapped_graph));
  
  matrix_RCP map;
  if (milo_solver->Comm->getSize() > 1) {
    map = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(milo_solver->LA_overlapped_graph));
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
  
  if (milo_solver->Comm->getSize() > 1) {
    map->doExport(*map_over, *(milo_solver->exporter), Tpetra::ADD);
    map->fillComplete();
  }
  return map;
}

////////////////////////////////////////////////////////////////////////////////
// Get an empty vector
////////////////////////////////////////////////////////////////////////////////

vector_RCP SubGridFEM_Solver::getVector() {
  vector_RCP vec = Teuchos::rcp(new LA_MultiVector(milo_solver->LA_overlapped_map,1));
  return vec;
}

////////////////////////////////////////////////////////////////////////////////
// Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<LA_CrsMatrix>  SubGridFEM_Solver::getEvaluationMatrix(const DRV & newip, Teuchos::RCP<LA_Map> & ip_map) {
  matrix_RCP map_over = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(milo_solver->LA_overlapped_graph) );
  matrix_RCP map;
  if (milo_solver->Comm->getSize() > 1) {
    size_t maxEntries = 256;
    map = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(milo_solver->LA_owned_map, maxEntries) );
    
    map->setAllToScalar(0.0);
    map->doExport(*map_over, *(milo_solver->exporter), Tpetra::ADD);
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

void SubGridFEM_Solver::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
  for (size_t b=0; b<assembler->wkset.size(); b++) {
    assembler->wkset[b]->params = params;
    assembler->wkset[b]->paramnames = paramnames;
  }
  milo_solver->phys->updateParameters(params, paramnames);
  
}

// ========================================================================================
//
// ========================================================================================

void SubGridFEM_Solver::performGather(const size_t & b, const vector_RCP & vec,
                                      const size_t & type, const size_t & entry) const {
  
  //for (size_t e=0; e < cells[block].size(); e++) {
  //  cells[block][e]->setLocalSoln(vec, type, index);
  //}
  // Get a view of the vector on the HostDevice
  auto vec_kv = vec->getLocalView<HostDevice>();
  
  // Get a corresponding view on the AssemblyDevice
  
  Kokkos::View<LO*,AssemblyDevice> numDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> data;
  Kokkos::View<int**,AssemblyDevice> offsets;
  LIDView LIDs;
  
  for (size_t c=0; c < assembler->cells[b].size(); c++) {
    switch(type) {
      case 0 :
        //index = cells[b][c]->index;
        numDOF = assembler->cells[b][c]->cellData->numDOF;
        data = assembler->cells[b][c]->u;
        LIDs = assembler->cells[b][c]->LIDs;
        offsets = assembler->wkset[0]->offsets;
        break;
      case 1 : // deprecated
        break;
      case 2 :
        numDOF = assembler->cells[b][c]->cellData->numDOF;
        data = assembler->cells[b][c]->phi;
        LIDs = assembler->cells[b][c]->LIDs;
        offsets = assembler->wkset[0]->offsets;
        break;
      case 3 : // deprecated
        break;
      case 4:
        numDOF = assembler->cells[b][c]->cellData->numParamDOF;
        data = assembler->cells[b][c]->param;
        LIDs = assembler->cells[b][c]->paramLIDs;
        offsets = assembler->wkset[0]->paramoffsets;
        break;
      case 5 :
        numDOF = assembler->cells[b][c]->cellData->numAuxDOF;
        data = assembler->cells[b][c]->aux;
        LIDs = assembler->cells[b][c]->auxLIDs;
        offsets = assembler->cells[b][c]->auxoffsets;
        break;
      default :
        cout << "ERROR - NOTHING WAS GATHERED" << endl;
    }
    
    parallel_for(RangePolicy<AssemblyExec>(0,data.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t n=0; n<numDOF.extent(0); n++) {
        for (size_t i=0; i<numDOF(n); i++ ) {
          data(e,n,i) = vec_kv(LIDs(e,offsets(n,i)),entry);
        }
      }
    });
  }
  
}

// ========================================================================================
//
// ========================================================================================

void SubGridFEM_Solver::performBoundaryGather(const size_t & b, const vector_RCP & vec,
                                              const size_t & type, const size_t & entry) const {
  
  if (assembler->boundaryCells.size() > b) {
    
    // Get a view of the vector on the HostDevice
    
    // TMW: this all needs to be updated
    auto vec_kv = vec->getLocalView<HostDevice>();
    
    // Get a corresponding view on the AssemblyDevice
    
    //Kokkos::View<LO***,AssemblyDevice> index;
    Kokkos::View<LO*,UnifiedDevice> numDOF;
    Kokkos::View<ScalarT***,AssemblyDevice> data;
    Kokkos::View<int**,AssemblyDevice> offsets;
    LIDView LIDs;
    
    for (size_t c=0; c < assembler->boundaryCells[b].size(); c++) {
      if (assembler->boundaryCells[b][c]->numElem > 0) {
        
        switch(type) {
          case 0 :
            numDOF = assembler->boundaryCells[b][c]->cellData->numDOF;
            data = assembler->boundaryCells[b][c]->u;
            LIDs = assembler->boundaryCells[b][c]->LIDs;
            offsets = assembler->wkset[0]->offsets;
            break;
          case 1 : // deprecated
            break;
          case 2 :
            numDOF = assembler->boundaryCells[b][c]->cellData->numDOF;
            data = assembler->boundaryCells[b][c]->phi;
            LIDs = assembler->boundaryCells[b][c]->LIDs;
            offsets = assembler->wkset[0]->offsets;
            break;
          case 3 : // deprecated
            break;
          case 4:
            numDOF = assembler->boundaryCells[b][c]->cellData->numParamDOF;
            data = assembler->boundaryCells[b][c]->param;
            LIDs = assembler->boundaryCells[b][c]->paramLIDs;
            offsets = assembler->wkset[0]->paramoffsets;
            break;
          case 5 :
            numDOF = assembler->boundaryCells[b][c]->cellData->numAuxDOF;
            data = assembler->boundaryCells[b][c]->aux;
            LIDs = assembler->boundaryCells[b][c]->auxLIDs;
            offsets = assembler->boundaryCells[b][c]->auxoffsets;
            break;
          default :
            cout << "ERROR - NOTHING WAS GATHERED" << endl;
        }
        
        parallel_for(RangePolicy<AssemblyExec>(0,data.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_t n=0; n<numDOF.extent(0); n++) {
            for(size_t i=0; i<numDOF(n); i++ ) {
              data(e,n,i) = vec_kv(LIDs(e,offsets(n,i)),entry);
            }
          }
        });
      }
    }
  }
  
}
