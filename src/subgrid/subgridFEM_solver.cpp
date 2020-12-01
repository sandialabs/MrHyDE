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
#include "solverInterface.hpp"

#include "subgridFEM_solver.hpp"

using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


SubGridFEM_Solver::SubGridFEM_Solver(const Teuchos::RCP<MpiComm> & LocalComm,
                                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                     Teuchos::RCP<meshInterface> & mesh,
                                     Teuchos::RCP<discretization> & disc,
                                     Teuchos::RCP<physics> & physics,
                                     Teuchos::RCP<AssemblyManager<SubgridSolverNode> > & assembler_,
                                     Teuchos::RCP<ParameterManager<SubgridSolverNode>> & params,
                                     Teuchos::RCP<panzer::DOFManager> & DOF,
                                     ScalarT & macro_deltat_, size_t & numMacroDOF) :
settings(settings_), macro_deltat(macro_deltat_), assembler(assembler_) {
  
  verbosity = settings->get<int>("verbosity",0);
  debug_level = settings->get<int>("debug level",0);
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
  amesos_solver_type = settings->sublist("Solver").get<string>("Amesos solver type","KLU2");

  store_aux_and_flux = settings->sublist("Postprocess").get<bool>("store aux and flux",false);
  
  milo_solver = Teuchos::rcp( new solver<SubgridSolverNode>(LocalComm, settings, mesh, disc, physics, DOF, assembler, params) );
  
  res = Teuchos::rcp( new SG_MultiVector(milo_solver->LA_owned_map,1)); // allocate residual
  J = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(milo_solver->LA_overlapped_graph));
  
  u = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_overlapped_map,1));
  phi = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_overlapped_map,1));
  
  if (LocalComm->getSize() > 1) {
    res_over = Teuchos::rcp( new SG_MultiVector(milo_solver->LA_overlapped_map,1)); // allocate residual
    sub_J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(milo_solver->LA_overlapped_graph));
  }
  else {
    res_over = res;
    sub_J_over = J;
  }
  
  d_um = Teuchos::rcp( new SG_MultiVector(milo_solver->LA_owned_map,numMacroDOF)); // reset residual
  d_sub_res_overm = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_overlapped_map,numMacroDOF));
  d_sub_resm = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_owned_map,numMacroDOF));
  d_sub_u_prevm = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_owned_map,numMacroDOF));
  d_sub_u_overm = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_overlapped_map,numMacroDOF));
  
  du_glob = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_owned_map,1));
  if (LocalComm->getSize() > 1) {
    du = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_overlapped_map,1));
  }
  else {
    du = du_glob;
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM_Solver::solve(Kokkos::View<ScalarT***,AssemblyDevice> coarse_u,
                              Kokkos::View<ScalarT***,AssemblyDevice> coarse_phi,
                              Teuchos::RCP<SG_MultiVector> & prev_u,
                              Teuchos::RCP<SG_MultiVector> & prev_phi,
                              Teuchos::RCP<SG_MultiVector> & disc_params,
                              Teuchos::RCP<SubGridMacroData> & macroData,
                              const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                              const bool & compute_jacobian, const bool & compute_sens,
                              const int & num_active_params,
                              const bool & compute_disc_sens, const bool & compute_aux_sens,
                              workset & macrowkset,
                              const int & usernum, const int & macroelemindex,
                              Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) {
  
  Teuchos::TimeMonitor totalsolvertimer(*sgfemSolverTimer);
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Starting SubgridFEM_Solver::solve ..." << endl;
    }
  }
  ScalarT current_time = time;
  //int macroDOF = macrowkset.numDOF;
  //bool usesubadjoint = false;
  
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
  
  auto prev_u_kv = prev_u->getLocalView<SubgridSolverNode::device_type>();
  auto u_kv = u->getLocalView<SubgridSolverNode::device_type>();
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
  
  Teuchos::RCP<SG_MultiVector> d_u = d_um;
  if (compute_sens) {
    d_u = Teuchos::rcp( new SG_MultiVector(milo_solver->LA_owned_map, num_active_params)); // reset residual
  }
  d_u->putScalar(0.0);
  
  res->putScalar(0.0);
  
  //ScalarT h = 0.0;
  //assembler->wkset[0]->resetFlux();
  
  if (isTransient) {
    ScalarT sgtime = 0.0;
    Teuchos::RCP<SG_MultiVector> prev_u = u;
    vector<Teuchos::RCP<SG_MultiVector> > curr_fsol;
    vector<ScalarT> subsolvetimes;
    subsolvetimes.push_back(sgtime);
    if (isAdjoint) {
      // First, we need to resolve the forward problem
      
      for (int tstep=0; tstep<time_steps; tstep++) {
        Teuchos::RCP<SG_MultiVector> recu = Teuchos::rcp( new SG_MultiVector(milo_solver->LA_overlapped_map,1)); // reset residual
        
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
        assembler->wkset[0]->setDeltat(1.0/alpha);
        
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
        assembler->wkset[0]->setDeltat(1.0/alpha);
        
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
    
    assembler->wkset[0]->setDeltat(1.0);
    
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
  
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Finished SubgridFEM_Solver::solve ..." << endl;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Store macro-dofs and flux (for ML-based subgrid)
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM_Solver::storeFluxData(Kokkos::View<ScalarT***,AssemblyDevice> lambda, Kokkos::View<AD**,AssemblyDevice> flux) {
  
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

void SubGridFEM_Solver::nonlinearSolver(Teuchos::RCP<SG_MultiVector> & sub_u,
                                               Teuchos::RCP<SG_MultiVector> & sub_phi,
                                               Teuchos::RCP<SG_MultiVector> & sub_params,
                                               Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                               const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                               const int & num_active_params, const ScalarT & alpha, const int & usernum,
                                               const bool & store_adjPrev) {
  
  
  Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverTimer);
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Starting SubgridFEM_Solver::nonlinearSolver ..." << endl;
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
  
  while (iter < sub_maxNLiter && resnorm_scaled[0] > sub_NLtol) {
    
    sub_J_over->resumeFill();
    
    sub_J_over->setAllToScalar(0.0);
    res_over->putScalar(0.0);
    
    auto localMatrix = sub_J_over->getLocalMatrix();
    auto res_view = res_over->template getLocalView<SubgridSolverNode::device_type>();
    
    //assembler->wkset[0]->setTime(time);
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
      
      //this->performBoundaryGather(usernum, sub_u, 0, 0);
      //if (isAdjoint) {
      //  this->performBoundaryGather(usernum, sub_phi, 2, 0);
      //}
      
      for (size_t e=0; e < assembler->boundaryCells[usernum].size(); e++) {
        assembler->boundaryCells[usernum][e]->aux = lambda;
      }
    }
    
    auto local_res_ladev = Kokkos::create_mirror(SG_exec(), local_res);
    auto local_J_ladev = Kokkos::create_mirror(SG_exec(), local_J);
    
    ////////////////////////////////////////////////
    // volume assembly
    ////////////////////////////////////////////////
    
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
      
      if (data_avail) {
        assembler->scatter(localMatrix, res_view, local_res, local_J,
                           assembler->cells[usernum][e]->LIDs,
                           assembler->cells[usernum][e]->paramLIDs,
                           true, false);
      }
      else {
        Kokkos::deep_copy(local_J_ladev,local_J);
        Kokkos::deep_copy(local_res_ladev,local_res);
        
        if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
          assembler->scatter(localMatrix, res_view, local_res_ladev, local_J_ladev,
                             assembler->cells[usernum][e]->LIDs_host,
                             assembler->cells[usernum][e]->paramLIDs_host,
                             true, false);
        }
        else { // LA_device = CUDA, AssemblyDevice = Host
          // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
          auto LIDs_dev = Kokkos::create_mirror(SG_exec(), assembler->cells[usernum][e]->LIDs);
          auto paramLIDs_dev = Kokkos::create_mirror(SG_exec(), assembler->cells[usernum][e]->paramLIDs);
          Kokkos::deep_copy(LIDs_dev, assembler->cells[usernum][e]->LIDs);
          Kokkos::deep_copy(paramLIDs_dev, assembler->cells[usernum][e]->paramLIDs);
          
          assembler->scatter(localMatrix, res_view, local_res_ladev, local_J_ladev,
                             LIDs_dev, paramLIDs_dev,
                             true, false);
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
        
        
        //////////////////////////////////////////////////////////////////////////
        // Scatter into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        if (data_avail) {
          assembler->scatter(localMatrix, res_view, local_res, local_J,
                             assembler->boundaryCells[usernum][e]->LIDs,
                             assembler->boundaryCells[usernum][e]->paramLIDs,
                             true, false);
        }
        else {
          
          auto local_res_ladev = Kokkos::create_mirror(SG_exec(),local_res);
          auto local_J_ladev = Kokkos::create_mirror(SG_exec(),local_J);
          
          Kokkos::deep_copy(local_J_ladev,local_J);
          Kokkos::deep_copy(local_res_ladev,local_res);
          
          if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
            assembler->scatter(localMatrix, res_view, local_res_ladev, local_J_ladev,
                               assembler->boundaryCells[usernum][e]->LIDs_host,
                               assembler->boundaryCells[usernum][e]->paramLIDs_host,
                               true, false);
          }
          else { // LA_device = CUDA, AssemblyDevice = Host
            // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
            auto LIDs_dev = Kokkos::create_mirror(SG_exec(), assembler->boundaryCells[usernum][e]->LIDs);
            auto paramLIDs_dev = Kokkos::create_mirror(SG_exec(), assembler->boundaryCells[usernum][e]->paramLIDs);
            Kokkos::deep_copy(LIDs_dev, assembler->boundaryCells[usernum][e]->LIDs);
            Kokkos::deep_copy(paramLIDs_dev, assembler->boundaryCells[usernum][e]->paramLIDs);
            
            assembler->scatter(localMatrix, res_view, local_res_ladev, local_J_ladev,
                               LIDs_dev, paramLIDs_dev,
                               true, false);
          }
          
        }
                
      }
    }
    
    if (maxElem > numElem) {
      if (data_avail) {
        auto LIDs = assembler->cells[0][0]->LIDs;
        this->fixDiagonal(LIDs, localMatrix, numElem);
      }
      else {
        if (use_host_LIDs) {
          auto LIDs = assembler->cells[0][0]->LIDs_host;
          this->fixDiagonal(LIDs, localMatrix, numElem);
        }
        else {
          auto LIDs_dev = Kokkos::create_mirror(SG_exec(), assembler->cells[0][0]->LIDs);
          Kokkos::deep_copy(LIDs_dev, assembler->cells[0][0]->LIDs);
          this->fixDiagonal(LIDs_dev, localMatrix, numElem);
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
        if (have_sym_factor) {
          Am2Solver->setA(J, Amesos2::SYMBFACT);
          Am2Solver->setX(du_glob);
          Am2Solver->setB(res);
        }
        else {
          Teuchos::TimeMonitor amsetuptimer(*sgfemNonlinearSolverAmesosSetupTimer);
          Am2Solver = Amesos2::create<SG_CrsMatrix,SG_MultiVector>(amesos_solver_type, J, du_glob, res);
          Am2Solver->symbolicFactorization();
          have_sym_factor = true;
        }
        Am2Solver->numericFactorization().solve();
      }
      else {
        if (have_belos) {
          //belos_problem->setProblem(du_glob, res);
        }
        else {
          belos_problem = Teuchos::rcp(new SG_LinearProblem(J, du_glob, res));
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
          
          belos_solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT, SG_MultiVector, SG_Operator>(belos_problem, belosList));
          
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
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Finished SubgridFEM_Solver::nonlinearSolver ..." << endl;
    }
  }
}

//////////////////////////////////////////////////////////////
// Correct the diagonal for certain cases
//////////////////////////////////////////////////////////////

template<class LIDViewType, class MatType>
void SubGridFEM_Solver::fixDiagonal(LIDViewType LIDs, MatType localMatrix, const int startpoint) {
  
  typedef typename SubgridSolverNode::execution_space SG_exec;
  
  parallel_for("subgrid diagonal fix",RangePolicy<SG_exec>(startpoint,LIDs.extent(0)), KOKKOS_LAMBDA (const int elem ) {
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

void SubGridFEM_Solver::computeSolnSens(Teuchos::RCP<SG_MultiVector> & d_sub_u,
                                               const bool & compute_sens,
                                               Teuchos::RCP<SG_MultiVector> & sub_u,
                                               Teuchos::RCP<SG_MultiVector> & sub_phi,
                                               Teuchos::RCP<SG_MultiVector> & sub_param,
                                               Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                               const ScalarT & time,
                                               const bool & isTransient, const bool & isAdjoint,
                                               const int & num_active_params, const ScalarT & alpha,
                                               const ScalarT & lambda_scale, const int & usernum,
                                               Kokkos::View<ScalarT**,AssemblyDevice> subgradient) {
  
  Teuchos::TimeMonitor localtimer(*sgfemSolnSensTimer);
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Starting SubgridFEM_Solver::computeSolnSens ..." << endl;
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
    d_sub_res_over = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_overlapped_map,numsubDerivs));
    d_sub_res = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_owned_map,numsubDerivs));
    d_sub_u_prev = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_owned_map,numsubDerivs));
    d_sub_u_over = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_overlapped_map,numsubDerivs));
  }
  
  d_sub_res_over->putScalar(0.0);
  d_sub_res->putScalar(0.0);
  d_sub_u_prev->putScalar(0.0);
  d_sub_u_over->putScalar(0.0);
  
  //ScalarT scale = -1.0*lambda_scale;
  
  auto dres_view = d_sub_res_over->getLocalView<SG_device>();
  
  // Are these gathers even necessary???
  /*
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
    //this->performBoundaryGather(usernum, sub_u, 0, 0);
    //if (isAdjoint) {
    //  this->performBoundaryGather(usernum, sub_phi, 2, 0);
    //}
    for (size_t e=0; e < assembler->boundaryCells[usernum].size(); e++) {
      assembler->boundaryCells[usernum][e]->aux = lambda;
    }
  }
  //this->performGather(usernum, sub_param, 4, 0);
  */
  
  assembler->wkset[0]->setTime(time);
  assembler->wkset[0]->isTransient = isTransient;
  assembler->wkset[0]->isAdjoint = isAdjoint;
  
  if (compute_sens) {
    
    //this->sacadoizeParams(true, num_active_params);
    
    if (multiscale_method != "mortar") {
      
      int numElem = assembler->cells[usernum][0]->numElem;
      int snumDOF = assembler->cells[usernum][0]->LIDs.extent(1);
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_res("local residual",numElem,snumDOF,num_active_params);
      Kokkos::View<ScalarT***,AssemblyDevice> local_J("local Jacobian",numElem,snumDOF,snumDOF);
      
      for (size_t elem=0; elem<assembler->cells[usernum].size(); elem++) {
        
        assembler->wkset[0]->localEID = elem;
        assembler->cells[usernum][elem]->updateData();
        Kokkos::deep_copy(local_res, 0.0);
        
        assembler->cells[usernum][elem]->computeJacRes(time, isTransient, isAdjoint,
                                                    false, true, num_active_params, false, false, false,
                                                    local_res, local_J,
                                                    assembler->assemble_volume_terms[0],
                                                    assembler->assemble_face_terms[0]);
        
        this->updateResSens(true, usernum, elem, dres_view, local_res,
                            data_avail, use_host_LIDs, true);
          
      }
      
    }
    else {
            
      int numElem = assembler->boundaryCells[usernum][0]->numElem;
      int snumDOF = assembler->boundaryCells[usernum][0]->LIDs.extent(1);
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_res("local residual",numElem,snumDOF,num_active_params);
      Kokkos::View<ScalarT***,AssemblyDevice> local_J("local Jacobian",numElem,snumDOF,snumDOF);
      
      for (size_t elem=0; elem<assembler->boundaryCells[usernum].size(); elem++) {
        
        Kokkos::deep_copy(local_res, 0.0);
        
        assembler->boundaryCells[usernum][elem]->computeJacRes(time, isTransient, isAdjoint,
                                                               false, true, num_active_params, false, false, false,
                                                               local_res, local_J);
        
        this->updateResSens(false, usernum, elem, dres_view, local_res,
                            data_avail, use_host_LIDs, true);
          
      }
      
    }
    
    auto sub_phi_kv = sub_phi->getLocalView<SG_device>();
    auto d_sub_res_over_kv = d_sub_res_over->getLocalView<SG_device>();
    
    auto subgrad_host = Kokkos::create_mirror_view(subgradient);
    
    for (int p=0; p<num_active_params; p++) {
      auto sub_res_sv = Kokkos::subview(d_sub_res_over_kv,Kokkos::ALL(),p);
      ScalarT subgrad = 0.0;
      parallel_reduce(RangePolicy<SG_exec>(0,sub_phi_kv.extent(0)), KOKKOS_LAMBDA (const int i, ScalarT& update) {
        update += sub_phi_kv(i,0) * sub_res_sv(i);
      }, subgrad);
      subgrad_host(p,0) = subgrad;
    }
    Kokkos::deep_copy(subgradient,subgrad_host);
  }
  else {
    
    Kokkos::View<ScalarT***,AssemblyDevice> local_res, local_J;
    
    if (multiscale_method != "mortar") {
      
      int numElem = assembler->cells[usernum][0]->numElem;
      int snumDOF = assembler->cells[usernum][0]->LIDs.extent(1);
      int anumDOF = assembler->cells[usernum][0]->auxLIDs.extent(1);
      
      local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,snumDOF,1);
      local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,snumDOF,anumDOF);
      
      for (size_t elem=0; elem<assembler->cells[usernum].size(); elem++) {
                
        assembler->wkset[0]->localEID = elem;
        
        Kokkos::deep_copy(local_res, 0.0);
        Kokkos::deep_copy(local_J, 0.0);
        
        // TMW: this may not work properly with new version
        assembler->cells[usernum][elem]->updateData();
        
        assembler->cells[usernum][elem]->computeJacRes(time, isTransient, isAdjoint,
                                                    true, false, num_active_params, false, true, false,
                                                    local_res, local_J,
                                                    assembler->assemble_volume_terms[0],
                                                    assembler->assemble_face_terms[0]);
        
        this->updateResSens(true, usernum, elem, dres_view, local_J,
                            data_avail, use_host_LIDs, false);
        
      }
    }
    else {
      
      int numElem = assembler->boundaryCells[usernum][0]->numElem;
      int snumDOF = assembler->boundaryCells[usernum][0]->LIDs.extent(1);
      int anumDOF = assembler->boundaryCells[usernum][0]->auxLIDs.extent(1);
      
      local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,snumDOF,1);
      local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,snumDOF,anumDOF);
      
      for (size_t elem=0; elem<assembler->boundaryCells[usernum].size(); elem++) {
        
        Kokkos::deep_copy(local_res, 0.0);
        Kokkos::deep_copy(local_J, 0.0);
        
        assembler->boundaryCells[usernum][elem]->computeJacRes(time, isTransient, isAdjoint,
                                                            true, false, num_active_params, false, true, false,
                                                            local_res, local_J);
        
        this->updateResSens(false, usernum, elem, dres_view, local_J,
                            data_avail, use_host_LIDs, false);
        
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
      
      auto d_sub_u_over_kv = d_sub_u_over->getLocalView<SG_device>();
      auto d_sub_res_kv = d_sub_res->getLocalView<SG_device>();
      for (int c=0; c<numsubDerivs; c++) {
        Teuchos::RCP<SG_MultiVector> x = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_overlapped_map,1));
        Teuchos::RCP<SG_MultiVector> b = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_owned_map,1));
        auto b_kv = Kokkos::subview(b->getLocalView<SG_device>(),Kokkos::ALL(),0);
        auto x_kv = Kokkos::subview(x->getLocalView<SG_device>(),Kokkos::ALL(),0);
        
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
    
    if (milo_solver->Comm->getSize() > 1) {
      d_sub_u->putScalar(0.0);
      d_sub_u->doImport(*d_sub_u_over, *(milo_solver->importer), Tpetra::ADD);
    }
    else {
      d_sub_u = d_sub_u_over;
    }
    
  }
  
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Finished SubgridFEM_Solver::computeSolnSens ..." << endl;
    }
  }
}

//////////////////////////////////////////////////////////////
// Figure out which views are needed to update the residual for the subgrid solution sensitivity wrt coarse DOFs
//////////////////////////////////////////////////////////////

template<class ResViewType, class DataViewType>
void SubGridFEM_Solver::updateResSens(const bool & use_cells, const int & usernum, const int & elem, ResViewType dres_view,
                                      DataViewType data, const bool & data_avail,
                                      const bool & use_host_LIDs, const bool & compute_sens) {
  
  typedef typename SubgridSolverNode::execution_space SG_exec;

  if (data_avail) {
    if (use_cells) {
      this->updateResSens(dres_view,data,
                          assembler->cells[usernum][elem]->LIDs,
                          compute_sens);
    }
    else {
      this->updateResSens(dres_view,data,
                          assembler->boundaryCells[usernum][elem]->LIDs,
                          compute_sens);
    }
  }
  else { // need to send assembly data to solver device
    auto data_sgladev = Kokkos::create_mirror(SG_exec(), data);
    Kokkos::deep_copy(data_sgladev,data);
    if (use_host_LIDs) { // copy already on host device
      if (use_cells) {
        this->updateResSens(dres_view, data_sgladev,
                            assembler->cells[usernum][elem]->LIDs_host,
                            compute_sens);
      }
      else {
        this->updateResSens(dres_view, data_sgladev,
                            assembler->boundaryCells[usernum][elem]->LIDs_host,
                            compute_sens);
      }
    }
    else { // solve on GPU, but assembly on CPU (not common)
      if (use_cells) {
        auto LIDs = assembler->cells[usernum][elem]->LIDs;
        auto LIDs_sgladev = Kokkos::create_mirror(SG_exec(), LIDs);
        Kokkos::deep_copy(LIDs_sgladev,LIDs);
        this->updateResSens(dres_view,data_sgladev,LIDs_sgladev,compute_sens);
      }
      else {
        auto LIDs = assembler->boundaryCells[usernum][elem]->LIDs;
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
void SubGridFEM_Solver::updateResSens(ResViewType res, DataViewType data, LIDViewType LIDs, const bool & compute_sens ) {
  
  typedef typename SubgridSolverNode::execution_space SG_exec;

  if (compute_sens) {
    parallel_for("subgrid diagonal fix",RangePolicy<SG_exec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(elem,row);
        for (size_type col=0; col<data.extent(2); col++ ) {
          Kokkos::atomic_add(&(res(rowIndex,col)),data(elem,row,col));
        }
      }
    });
  }
  else {
    parallel_for("subgrid diagonal fix",RangePolicy<SG_exec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(elem,row);
        for (size_type col=0; col<data.extent(2); col++ ) {
          Kokkos::atomic_add(&(res(rowIndex,col)),-1.0*data(elem,row,col));
        }
      }
    });
  
  }
  
}

//////////////////////////////////////////////////////////////
// Update the flux
//////////////////////////////////////////////////////////////

void SubGridFEM_Solver::updateFlux(const Teuchos::RCP<SG_MultiVector> & u,
                                   const Teuchos::RCP<SG_MultiVector> & d_u,
                                   Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                   const Teuchos::RCP<SG_MultiVector> & disc_params,
                                   const bool & compute_sens, const int macroelemindex,
                                   const ScalarT & time, workset & macrowkset,
                                   const int & usernum, const ScalarT & fwt,
                                   Teuchos::RCP<SubGridMacroData> & macroData) {
  
  Teuchos::TimeMonitor localtimer(*sgfemFluxTimer);
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Starting SubgridFEM_Solver::updateFlux (intermediate function) ..." << endl;
    }
  }
    
  typedef typename SubgridSolverNode::memory_space SGS_mem;
  
  auto u_kv = u->getLocalView<SubgridSolverNode::device_type>();
  auto du_kv = d_u->getLocalView<SubgridSolverNode::device_type>();
  
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
                       time, macrowkset, usernum, fwt, macroData);
    }
    else {
      auto u_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),u_kv);
      Kokkos::deep_copy(u_dev,u_kv);
      auto du_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),du_kv);
      Kokkos::deep_copy(du_dev,du_kv);
      auto dp_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),dp_kv);
      Kokkos::deep_copy(dp_dev,dp_kv);
      this->updateFlux(u_dev, du_dev, lambda, dp_dev, compute_sens, macroelemindex,
                       time, macrowkset, usernum, fwt, macroData);
    }
  }
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Finished SubgridFEM_Solver::updateFlux (intermediate function) ..." << endl;
    }
  }
}

//////////////////////////////////////////////////////////////
// Update the flux
//////////////////////////////////////////////////////////////

template<class ViewType>
void SubGridFEM_Solver::updateFlux(ViewType u_kv,
                                   ViewType du_kv,
                                   Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                   ViewType dp_kv,
                                   const bool & compute_sens, const int macroelemindex,
                                   const ScalarT & time, workset & macrowkset,
                                   const int & usernum, const ScalarT & fwt,
                                   Teuchos::RCP<SubGridMacroData> & macroData) {
  
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Starting SubgridFEM_Solver::updateFlux ..." << endl;
    }
  }
  
  for (size_t e=0; e<assembler->boundaryCells[usernum].size(); e++) {
    
    if (assembler->boundaryCells[usernum][e]->sidename == "interior") {
      {
        Teuchos::TimeMonitor localwktimer(*sgfemFluxWksetTimer);
        assembler->boundaryCells[usernum][e]->updateWorksetBasis();
      }
      
      auto cwts = assembler->wkset[0]->wts_side;
      ScalarT h = 0.0;
      assembler->wkset[0]->sidename = "interior";
      {
        Teuchos::TimeMonitor localcelltimer(*sgfemFluxCellTimer);
        assembler->boundaryCells[usernum][e]->computeFlux(u_kv, du_kv, dp_kv, lambda, time,
                                                          0, h, compute_sens);
      }
      {
        Teuchos::TimeMonitor localtimer(*sgfemAssembleFluxTimer);
        
        auto bMIDs = assembler->boundaryCells[usernum][e]->auxMIDs_dev;
        for (size_type n=0; n<macrowkset.offsets.extent(0); n++) {
          auto macrobasis_ip = assembler->boundaryCells[usernum][e]->auxside_basis[macrowkset.usebasis[n]];
          auto off = Kokkos::subview(macrowkset.offsets, n, Kokkos::ALL());
          auto flux = Kokkos::subview(assembler->wkset[0]->flux, Kokkos::ALL(), n, Kokkos::ALL());
          auto res = macrowkset.res;
          parallel_for("subgrid flux",RangePolicy<AssemblyExec>(0,bMIDs.extent(0)), KOKKOS_LAMBDA (const size_type c ) {
            for (size_type j=0; j<macrobasis_ip.extent(1); j++) {
              for (size_type i=0; i<macrobasis_ip.extent(2); i++) {
                AD val = macrobasis_ip(c,j,i)*flux(c,i)*cwts(c,i);
                Kokkos::atomic_add( &res(bMIDs(c),off(j)), val);
              }
            }
          });
        }
      }
    }
  }
  
  if (debug_level > 0) {
    if (milo_solver->Comm->getRank() == 0) {
      cout << "**** Finished SubgridFEM_Solver::updateFlux ..." << endl;
    }
  }
}

//////////////////////////////////////////////////////////////
// Compute the initial values for the subgrid solution
//////////////////////////////////////////////////////////////

void SubGridFEM_Solver::setInitial(Teuchos::RCP<SG_MultiVector> & initial,
                                   const int & usernum, const bool & useadjoint) {
  
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
   
   Teuchos::RCP<SG_MultiVector> glinitial = Teuchos::rcp(new SG_MultiVector(*overlapped_map,1)); // reset residual
   
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

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> >  SubGridFEM_Solver::getProjectionMatrix() {
  
  // Compute the mass matrix on a reference element
  matrix_RCP mass = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(milo_solver->LA_overlapped_graph) );
  
  auto localMatrix = mass->getLocalMatrix();
  
  int usernum = 0;
  for (size_t e=0; e<assembler->cells[usernum].size(); e++) {
    LIDView LIDs = assembler->cells[usernum][e]->LIDs;
    Kokkos::View<ScalarT***,AssemblyDevice> localmass = assembler->cells[usernum][e]->getMass();
    
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
    glmass = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(milo_solver->LA_owned_map,maxEntries) );
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

std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridFEM_Solver::evaluateBasis2(const DRV & pts) {
  
  size_t numpts = pts.extent(1);
  size_t dimpts = pts.extent(2);
  size_t numLIDs = assembler->cells[0][0]->LIDs.extent(1);
  Kokkos::View<int**,AssemblyDevice> owners("owners",numpts,2+numLIDs);
  
  for (size_t e=0; e<assembler->cells[0].size(); e++) {
    int numElem = assembler->cells[0][e]->numElem;
    DRV nodes = assembler->cells[0][e]->nodes;
    for (int c=0; c<numElem;c++) {
      DRV refpts("refpts",1, numpts, dimpts);
      Kokkos::DynRankView<int,PHX::Device> inRefCell("inRefCell",1,numpts);
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
  std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo(owners, ptsBasis);
  return basisinfo;
  
}


////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> > SubGridFEM_Solver::getProjectionMatrix(DRV & ip, DRV & wts,
                                                                  std::pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo) {
  
  std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > my_basisinfo = this->evaluateBasis2(ip);
  matrix_RCP map_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(milo_solver->LA_overlapped_graph));
  
  matrix_RCP map;
  if (milo_solver->Comm->getSize() > 1) {
    map = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(milo_solver->LA_overlapped_graph));
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

Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> > SubGridFEM_Solver::getVector() {
  vector_RCP vec = Teuchos::rcp(new SG_MultiVector(milo_solver->LA_overlapped_map,1));
  return vec;
}

////////////////////////////////////////////////////////////////////////////////
// Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> >  SubGridFEM_Solver::getEvaluationMatrix(const DRV & newip, Teuchos::RCP<SG_Map> & ip_map) {
  matrix_RCP map_over = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(milo_solver->LA_overlapped_graph) );
  matrix_RCP map;
  if (milo_solver->Comm->getSize() > 1) {
    size_t maxEntries = 256;
    map = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>(milo_solver->LA_owned_map, maxEntries) );
    
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
                                      const size_t & type, const size_t & entry) {
    
  typedef typename SubgridSolverNode::memory_space SGS_mem;
  
  auto vec_kv = vec->getLocalView<SubgridSolverNode::device_type>();
  auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), entry);
  
  if (Kokkos::SpaceAccessibility<AssemblyExec, SGS_mem>::accessible) { // can we avoid a copy?
    this->performGather(b, vec_slice, type);
    this->performBoundaryGather(b, vec_slice, type);
  }
  else {
    auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
    Kokkos::deep_copy(vec_dev,vec_slice);
    this->performGather(b, vec_dev, type);
    this->performBoundaryGather(b, vec_dev, type);
  }
  
}

// ========================================================================================
//
// ========================================================================================

template<class ViewType>
void SubGridFEM_Solver::performGather(const size_t & b, ViewType vec_dev, const size_t & type) {

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
void SubGridFEM_Solver::performBoundaryGather(const size_t & b, ViewType vec_dev, const size_t & type) {
  
  if (assembler->boundaryCells.size() > b) {
    
    Kokkos::View<LO*,AssemblyDevice> numDOF;
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
