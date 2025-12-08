/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ========================================================================================
// Wrapper to the main assembly routine to assemble over all blocks (most common use case)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::assembleJacRes(const size_t & set, const size_t & stage,
                                           vector<vector_RCP> & sol,
                                           vector<vector_RCP> & sol_stage,
                                           vector<vector_RCP> & sol_prev,
                                           vector<vector_RCP> & phi,
                                           vector<vector_RCP> & phi_stage,
                                           vector<vector_RCP> & phi_prev,
                                           const bool & compute_jacobian, const bool & compute_sens,
                                           const bool & compute_disc_sens,
                                           const bool & compute_previous_jac, const size_t & stepindex,
                                           vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                           const ScalarT & current_time,
                                           const bool & useadjoint, const bool & store_adjPrev,
                                           const int & num_active_params,
                                           vector_RCP & Psol,
                                           vector_RCP & Pdot,
                                           const bool & is_final_time,
                                           const ScalarT & deltat) {
  
  debugger->print("******** Starting AssemblyManager::assembleJacRes ...");
  
#ifndef MrHyDE_NO_AD
    
  for (size_t block=0; block<groups.size(); ++block) {
    
    if (groups[block].size() > 0) {
      if (groupData[block]->multiscale) {
        allow_autotune = false;
      }
      if (!allow_autotune) {
        this->assembleJacRes<AD>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                 compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                 current_time, useadjoint, store_adjPrev, num_active_params,
                                 is_final_time, block, deltat);
      }
      else {
        if (type_AD == -1) {
          this->assembleJacRes<AD>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                   compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                   current_time, useadjoint, store_adjPrev, num_active_params,
                                   is_final_time, block, deltat);
        }
        else if (type_AD == 2) {
          this->assembleJacRes<AD2>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                    compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                    current_time, useadjoint, store_adjPrev, num_active_params,
                                    is_final_time, block, deltat);
        }
        else if (type_AD == 4) {
          this->assembleJacRes<AD4>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                    compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                    current_time, useadjoint, store_adjPrev, num_active_params,
                                    is_final_time, block, deltat);
        }
        else if (type_AD == 8) {
          this->assembleJacRes<AD8>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                    compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                    current_time, useadjoint, store_adjPrev, num_active_params,
                                    is_final_time, block, deltat);
        }
        else if (type_AD == 16) {
          this->assembleJacRes<AD16>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                     compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                     current_time, useadjoint, store_adjPrev, num_active_params,
                                     is_final_time, block, deltat);
        }
        else if (type_AD == 18) {
          this->assembleJacRes<AD18>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                     compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                     current_time, useadjoint, store_adjPrev, num_active_params,
                                     is_final_time, block, deltat);
        }
        else if (type_AD == 24) {
          this->assembleJacRes<AD24>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                     compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                     current_time, useadjoint, store_adjPrev, num_active_params,
                                     is_final_time, block, deltat);
        }
        else if (type_AD == 32) {
          this->assembleJacRes<AD32>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                     compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                     current_time, useadjoint, store_adjPrev, num_active_params,
                                     is_final_time, block, deltat);
        }
        else {
          this->assembleJacRes<AD>(set, stage, sol, sol_stage, sol_prev, phi, phi_stage, phi_prev, Psol, Pdot,
                                   compute_jacobian, compute_sens, compute_disc_sens, compute_previous_jac, stepindex, res, J, isTransient,
                                   current_time, useadjoint, store_adjPrev, num_active_params,
                                   is_final_time, block, deltat);
        }
      }
    }
  }
  #endif

  debugger->print("******** Finished AssemblyManager::assembleJacRes");
  
}

// ========================================================================================
// Main assembly routine ... only assembles on a given block (b)
// This routine is the old version that does both Jacobian and residual
// Will eventually be deprecated
// ========================================================================================

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::assembleJacRes(const size_t & set, const size_t & stage,
                                           vector<vector_RCP> & sol,
                                           vector<vector_RCP> & sol_stage,
                                           vector<vector_RCP> & sol_prev,
                                           vector<vector_RCP> & phi,
                                           vector<vector_RCP> & phi_stage,
                                           vector<vector_RCP> & phi_prev,
                                           vector_RCP & param_sol,
                                           vector_RCP & param_dot,
                                           const bool & compute_jacobian, const bool & compute_sens,
                                           const bool & compute_disc_sens,
                                           const bool & compute_previous_jac, const size_t & stepindex,
                                           vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                           const ScalarT & current_time,
                                           const bool & useadjoint, const bool & store_adjPrev,
                                           const int & num_active_params,
                                           const bool & is_final_time,
                                           const int & block, const ScalarT & deltat) {
  
  Teuchos::TimeMonitor localassemblytimer(*assembly_timer);
  using namespace std;

  // Kokkos::CRSMatrix and Kokkos::View for J and res
  // Scatter needs to be on LA_device
  typedef typename Tpetra::CrsMatrix<ScalarT, LO, GO, Node >::local_matrix_device_type local_matrix;
  local_matrix J_kcrs;
  if (compute_jacobian) {
    J_kcrs = J->getLocalMatrixDevice();
  }
  
  auto res_view = res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  typedef typename Node::execution_space LA_exec;
  
  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }
  
  // LIDs are on AssemblyDevice.  If the AssemblyDevice memory is accessible, then these are fine.
  // Copy of LIDs is stored on HostDevice.
  bool use_host_LIDs = false;
  if (!data_avail) {
    if (Kokkos::SpaceAccessibility<LA_exec, HostDevice::memory_space>::accessible) {
      use_host_LIDs = true;
    }
  }
  
  // Determine if we can use the reduced memory version of assembly
  // This is the preferred approach, but not features are enabled yet
  bool reduce_memory = true;
  if (!data_avail || useadjoint || groupData[block]->multiscale || compute_disc_sens || compute_sens) {
    reduce_memory = false;
  }
  
  // Set the seeding flag for AD objects
  int seedwhat = 0;
  int seedindex = 0;
  if (compute_jacobian) {
    if (compute_disc_sens) {
      seedwhat = 3;
    }
    else {
      seedwhat = 1;
    }
  }
  if (compute_previous_jac) {
    seedwhat = 2;
    seedindex = stepindex;
  }
  
  
  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT*,AssemblyDevice> > sol_kv, sol_stage_kv, sol_prev_kv, phi_kv, phi_stage_kv, phi_prev_kv;
  for (size_t s=0; s<sol.size(); ++s) {
    auto vec_kv = sol[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
    if (data_avail) {
      sol_kv.push_back(vec_slice);
    }
    else {
      auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
      Kokkos::deep_copy(vec_dev,vec_slice);
      sol_kv.push_back(vec_dev);
    }
  }
  bool use_only_sol = false;
  if (sol_stage.size() == 0 && sol_prev.size() == 0) {
    use_only_sol = true;
  }
  if (!use_only_sol) {
    for (size_t s=0; s<sol_stage.size(); ++s) {
      auto vec_kv = sol_stage[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
      if (data_avail) {
        sol_stage_kv.push_back(vec_slice);
      }
      else {
        auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
        Kokkos::deep_copy(vec_dev,vec_slice);
        sol_stage_kv.push_back(vec_dev);
      }
    }
    for (size_t s=0; s<sol_prev.size(); ++s) {
      auto vec_kv = sol_prev[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
      if (data_avail) {
        sol_prev_kv.push_back(vec_slice);
      }
      else {
        auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
        Kokkos::deep_copy(vec_dev,vec_slice);
        sol_prev_kv.push_back(vec_dev);
      }
    }
  }
  if (useadjoint) {
    for (size_t s=0; s<phi.size(); ++s) {
      auto vec_kv = phi[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
      if (data_avail) {
        phi_kv.push_back(vec_slice);
      }
      else {
        auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
        Kokkos::deep_copy(vec_dev,vec_slice);
        phi_kv.push_back(vec_dev);
      }
    }
    if (!use_only_sol) {
      for (size_t s=0; s<phi_stage.size(); ++s) {
        auto vec_kv = phi_stage[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
        auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
        if (data_avail) {
          phi_stage_kv.push_back(vec_slice);
        }
        else {
          auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
          Kokkos::deep_copy(vec_dev,vec_slice);
          phi_stage_kv.push_back(vec_dev);
        }
      }
      for (size_t s=0; s<phi_prev.size(); ++s) {
        auto vec_kv = phi_prev[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
        auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
        if (data_avail) {
          phi_prev_kv.push_back(vec_slice);
        }
        else {
          auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
          Kokkos::deep_copy(vec_dev,vec_slice);
          phi_prev_kv.push_back(vec_dev);
        }
      }
    }
  }
    
  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT*,AssemblyDevice> > params_kv, params_dot_kv;
  
  auto p_kv = param_sol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto pslice = Kokkos::subview(p_kv, Kokkos::ALL(), 0);
  
  if (data_avail) {
    params_kv.push_back(pslice);
  }
  else {
    auto p_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),pslice);
    Kokkos::deep_copy(p_dev,pslice);
    params_kv.push_back(p_dev);
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Set up the worksets and allocate the local residual and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////
  
  //  This actually updates all of the worksets, which is fine
  this->updateWorksetTime(block, isTransient, current_time, deltat);

  this->updateWorksetAdjoint(block, useadjoint);
  int numElem = groupData[block]->num_elem;
  int numDOF = groups[block][0]->LIDs[set].extent(1);
  
  int numParamDOF = 0;
  if (compute_disc_sens) {
    numParamDOF = groups[block][0]->paramLIDs.extent(1);
  }
  
  // This data needs to be available on Host and Device
  // Optimizing layout for AssemblyExec
  Kokkos::View<ScalarT***,AssemblyDevice> local_res, local_J;
  
  if (!reduce_memory) {
    if (compute_sens) {
      local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual on device",numElem,numDOF,num_active_params);
    }
    else {
      local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual on device",numElem,numDOF,1);
    }
    
    if (compute_disc_sens) {
      local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian on device",numElem,numDOF,numParamDOF);
    }
    else { // note that this does increase memory as numElem increases
      local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian on device",numElem,numDOF,numDOF);
    }
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Volume contribution
  /////////////////////////////////////////////////////////////////////////////
  
  // Note: Cannot parallelize over groups since data structures are re-used
  
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    
    this->updateWorksetEID(block, grp);
    
    /////////////////////////////////////////////////////////////////////////////
    // Perform the necessary local gathers (now stored in group meta data)
    /////////////////////////////////////////////////////////////////////////////
    {
      Teuchos::TimeMonitor localtimer(*gather_timer);
    
      this->performGather(set, block, grp, useadjoint, stage, use_only_sol,
                          sol_kv, sol_stage_kv, sol_prev_kv,
                          phi_kv, phi_stage_kv, phi_prev_kv, params_kv, params_dot_kv);
    }

    /////////////////////////////////////////////////////////////////////////////
    // Compute the local residual and Jacobian on this group
    /////////////////////////////////////////////////////////////////////////////
    
    bool fixJacDiag = false;
    
    {
      Teuchos::TimeMonitor localtimer(*physics_timer);
      
      //////////////////////////////////////////////////////////////
      // Compute res and J=dF/du
      //////////////////////////////////////////////////////////////
      
      // Volumetric contribution
      if (assemble_volume_terms[set][block]) {
        if (groupData[block]->multiscale) {
          
#ifndef MrHyDE_NO_AD
          // Right now, this can only be called with AD, thus hard-coded
          this->updateWorkset<AD>(block, grp, 0, 0);
          multiscale_manager->evaluateMacroMicroMacroMap(wkset_AD[block], groups[block][grp], groupData[block], set, isTransient, useadjoint,
                                                         compute_jacobian, compute_sens, num_active_params,
                                                         compute_disc_sens, false,
                                                         store_adjPrev);
          
          fixJacDiag = true;
#endif
        }
        else {
          this->updateWorkset<EvalT>(block, grp, seedwhat, seedindex);
          physics->volumeResidual<EvalT>(set, block);
        }
      }
      
      ///////////////////////////////////////////////////////////////////////////
      // Edge/face contribution
      ///////////////////////////////////////////////////////////////////////////
      
      if (assemble_face_terms[set][block]) {
        if (groupData[block]->multiscale) {
          // do nothing
        }
        else {
          this->updateWorksetOnSide(block, true);
          for (size_t s=0; s<groupData[block]->num_sides; s++) {
            this->updateWorksetFace<EvalT>(block, grp, s);
            physics->faceResidual<EvalT>(set,block);
          }
          this->updateWorksetOnSide(block, false);
        }
      }
      
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Scatter into global matrix/vector
    ///////////////////////////////////////////////////////////////////////////
    
    if (reduce_memory) { // skip local_res and local_J
      EvalT dummyval = 0.0;
      this->scatter(set, J_kcrs, res_view,
                    groups[block][grp]->LIDs[set], groups[block][grp]->paramLIDs, block,
                    compute_jacobian, compute_sens, compute_disc_sens, useadjoint, dummyval);
    }
    else { // fill local_res and local_J and then scatter
      
      Teuchos::TimeMonitor localtimer(*scatter_timer);
      
      Kokkos::deep_copy(local_res,0.0);
      Kokkos::deep_copy(local_J,0.0);
      
      // Use AD residual to update local Jacobian
      if (compute_jacobian) {
        if (compute_disc_sens) {
          this->updateParamJac(block, grp, local_J);
        }
        else {
          this->updateJac(block, grp, useadjoint, local_J);
        }
      }
      
      if (compute_jacobian && fixJacDiag) {
        this->fixDiagJac(block, grp, local_J, local_res);
      }
      
      // Update the local residual
      
      this->updateRes(block, grp, compute_sens, local_res);
      
      // Now scatter from local_res and local_J
      
      if (data_avail) {
        this->scatterRes(res_view, local_res, groups[block][grp]->LIDs[set]);
        if (compute_jacobian) {
          this->scatterJac(set, J_kcrs, local_J, groups[block][grp]->LIDs[set], groups[block][grp]->paramLIDs, compute_disc_sens);
        }
      }
      else {
        auto local_res_ladev = create_mirror(LA_exec(),local_res);
        auto local_J_ladev = create_mirror(LA_exec(),local_J);
        
        Kokkos::deep_copy(local_J_ladev,local_J);
        Kokkos::deep_copy(local_res_ladev,local_res);
        
        if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
          this->scatterRes(res_view, local_res_ladev, groups[block][grp]->LIDs_host[set]);
          if (compute_jacobian) {
            this->scatterJac(set, J_kcrs, local_J_ladev, groups[block][grp]->LIDs_host[set], groups[block][grp]->paramLIDs_host, compute_disc_sens);
          }
          
        }
        else { // LA_device = CUDA, AssemblyDevice = Host
          // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
          auto LIDs_dev = Kokkos::create_mirror(LA_exec(), groups[block][grp]->LIDs[set]);
          auto paramLIDs_dev = Kokkos::create_mirror(LA_exec(), groups[block][grp]->paramLIDs);
          Kokkos::deep_copy(LIDs_dev,groups[block][grp]->LIDs[set]);
          Kokkos::deep_copy(paramLIDs_dev,groups[block][grp]->paramLIDs);
          
          this->scatterRes(res_view, local_res_ladev, LIDs_dev);
          if (compute_jacobian) {
            this->scatterJac(set, J_kcrs, local_J_ladev, LIDs_dev, paramLIDs_dev, compute_disc_sens);
          }
        }
        
      }
      
    }
    
  } // group loop
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Boundary terms
  //////////////////////////////////////////////////////////////////////////////////////
  
  if (assemble_boundary_terms[set][block]) {
    
    this->updateWorksetOnSide(block, true);
    
    if (!reduce_memory) {
      if (compute_sens) {
        local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,numDOF,num_active_params);
      }
      else {
        local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,numDOF,1);
      }
      
      if (compute_disc_sens) {
        local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,numDOF,numParamDOF);
      }
      else {
        local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,numDOF,numDOF);
      }
    }
    
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      
      if (boundary_groups[block][grp]->numElem > 0) {
        
        /////////////////////////////////////////////////////////////////////////////
        // Perform the necessary local gathers (now stored in group meta data)
        /////////////////////////////////////////////////////////////////////////////
    
        this->performBoundaryGather(set, block, grp, useadjoint, stage, use_only_sol,
                                    sol_kv, sol_stage_kv, sol_prev_kv,
                                    phi_kv, phi_stage_kv, phi_prev_kv, params_kv, params_dot_kv);
    
        /////////////////////////////////////////////////////////////////////////////
        // Compute the local residual and Jacobian on this boundary group
        /////////////////////////////////////////////////////////////////////////////
        
        this->updateWorksetResidual(block);

        this->updateWorksetBoundary<EvalT>(block, grp, seedwhat);
        
        if (!groupData[block]->multiscale) {
          Teuchos::TimeMonitor localtimer(*physics_timer);
          physics->boundaryResidual<EvalT>(set,block);
        }
        
        {
          physics->fluxConditions<EvalT>(set,block);
        }
        ///////////////////////////////////////////////////////////////////////////
        // Scatter into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        if (reduce_memory) { // skip local_res and local_J
          EvalT dummyval = 0.0;
          this->scatter(set, J_kcrs, res_view,
                        boundary_groups[block][grp]->LIDs[set], boundary_groups[block][grp]->paramLIDs, block,
                        compute_jacobian, compute_sens, compute_disc_sens, useadjoint, dummyval);
        }
        else { // fill local_res and local_J and then scatter
          
          Teuchos::TimeMonitor localtimer(*scatter_timer);
          
          Kokkos::deep_copy(local_res,0.0);
          Kokkos::deep_copy(local_J,0.0);
          
          // Use AD residual to update local Jacobian
          if (compute_jacobian) {
            if (compute_disc_sens) {
              this->updateParamJacBoundary(block, grp, local_J);
            }
            else {
              this->updateJacBoundary(block, grp, useadjoint, local_J);
            }
          }
          
          // Update the local residual (forward mode)
          this->updateResBoundary(block, grp, compute_sens, local_res);
          
          if (data_avail) {
            this->scatterRes(res_view, local_res, boundary_groups[block][grp]->LIDs[set]);
            if (compute_jacobian) {
              this->scatterJac(set, J_kcrs, local_J, boundary_groups[block][grp]->LIDs[set], boundary_groups[block][grp]->paramLIDs, compute_disc_sens);
            }
          }
          else {
            auto local_res_ladev = create_mirror(LA_exec(),local_res);
            auto local_J_ladev = create_mirror(LA_exec(),local_J);
            
            Kokkos::deep_copy(local_J_ladev,local_J);
            Kokkos::deep_copy(local_res_ladev,local_res);
            
            if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
              this->scatterRes(res_view, local_res_ladev, boundary_groups[block][grp]->LIDs_host[set]);
              if (compute_jacobian) {
                this->scatterJac(set, J_kcrs, local_J_ladev,
                                 boundary_groups[block][grp]->LIDs_host[set], boundary_groups[block][grp]->paramLIDs_host,
                                 compute_disc_sens);
              }
            }
            else { // LA_device = CUDA, AssemblyDevice = Host
              // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
              auto LIDs_dev = Kokkos::create_mirror(LA_exec(), boundary_groups[block][grp]->LIDs[set]);
              auto paramLIDs_dev = Kokkos::create_mirror(LA_exec(), boundary_groups[block][grp]->paramLIDs);
              Kokkos::deep_copy(LIDs_dev,boundary_groups[block][grp]->LIDs[set]);
              Kokkos::deep_copy(paramLIDs_dev,boundary_groups[block][grp]->paramLIDs);
              
              this->scatterRes(res_view, local_res_ladev, LIDs_dev);
              if (compute_jacobian) {
                this->scatterJac(set, J_kcrs, local_J_ladev, LIDs_dev, paramLIDs_dev, compute_disc_sens);
              }
            }
            
          }
        }
        
      }
    } // element loop
    this->updateWorksetOnSide(block, false);
  }
  
  // Apply constraints, e.g., strongly imposed Dirichlet
  this->dofConstraints(set, J, res, current_time, compute_jacobian, compute_disc_sens);
  
  
  if (fix_zero_rows) {
    size_t numrows = J->getLocalNumRows();
    
    parallel_for("assembly insert Jac",
                 RangePolicy<LA_exec>(0,numrows),
                 KOKKOS_LAMBDA (const size_t row ) {
      auto rowdata = J_kcrs.row(row);
      ScalarT abssum = 0.0;
      for (int col=0; col<rowdata.length; ++col ) {
        abssum += abs(rowdata.value(col));
      }
      ScalarT val[1];
      LO cols[1];
      if (abssum<1.0e-14) { // needs to be generalized!
        val[0] = 1.0;
        cols[0] = row;
        J_kcrs.replaceValues(row,cols,1,val,false,false);
      }
    });
  }
  
}


// ========================================================================================
// Wrapper to the main assembly routine to assemble over all blocks (most common use case)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::assembleRes(const size_t & set, const size_t & stage,
                                        vector<vector_RCP> & sol,
                                        vector<vector_RCP> & sol_stage,
                                        vector<vector_RCP> & sol_prev,
                                        vector<vector_RCP> & phi,
                                        vector<vector_RCP> & phi_stage,
                                        vector<vector_RCP> & phi_prev,
                                        vector_RCP & param_sol,
                                        vector_RCP & param_dot,
                                        vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                        const ScalarT & current_time, const ScalarT & deltat) {
  
  debugger->print("******** Starting AssemblyManager::assembleRes ...");
    
  for (size_t block=0; block<groups.size(); ++block) {
    if (groups[block].size() > 0) {
      this->assembleRes(set, stage,
                        sol, sol_stage, sol_prev, phi, phi_stage, phi_stage,
                        param_sol, param_dot, res, J, isTransient,
                        current_time, block, deltat);
    }
  }
  
  debugger->print("******** Finished AssemblyManager::assembleRes");
}

// ========================================================================================
// Main assembly routine ... just the residual on a given block (b)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::assembleRes(const size_t & set, const size_t & stage,
                                        vector<vector_RCP> & sol,
                                        vector<vector_RCP> & sol_stage,
                                        vector<vector_RCP> & sol_prev,
                                        vector<vector_RCP> & phi,
                                        vector<vector_RCP> & phi_stage,
                                        vector<vector_RCP> & phi_prev,
                                        vector_RCP & param_sol,
                                        vector_RCP & param_dot,
                                        vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                        const ScalarT & current_time,
                                        const int & block, const ScalarT & deltat) {
  
  Teuchos::TimeMonitor localassemblytimer(*assembly_res_timer);
    
  auto res_view = res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  typedef typename Node::execution_space LA_exec;
  
  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }
  
  // Set the seeding flag for AD objects
  int seedwhat = 0;
    
  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT*,AssemblyDevice> > sol_kv, sol_stage_kv, sol_prev_kv, phi_kv, phi_stage_kv, phi_prev_kv;
  for (size_t s=0; s<sol.size(); ++s) {
    auto vec_kv = sol[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
    if (data_avail) {
      sol_kv.push_back(vec_slice);
    }
    else {
      auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
      Kokkos::deep_copy(vec_dev,vec_slice);
      sol_kv.push_back(vec_dev);
    }
  }
  bool use_only_sol = false;
  if (sol_stage.size() == 0 && sol_prev.size() == 0) {
    use_only_sol = true;
  }
  if (!use_only_sol) {
    for (size_t s=0; s<sol_stage.size(); ++s) {
      auto vec_kv = sol_stage[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
      if (data_avail) {
        sol_stage_kv.push_back(vec_slice);
      }
      else {
        auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
        Kokkos::deep_copy(vec_dev,vec_slice);
        sol_stage_kv.push_back(vec_dev);
      }
    }
    for (size_t s=0; s<sol_prev.size(); ++s) {
      auto vec_kv = sol_prev[s]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
      if (data_avail) {
        sol_prev_kv.push_back(vec_slice);
      }
      else {
        auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
        Kokkos::deep_copy(vec_dev,vec_slice);
        sol_prev_kv.push_back(vec_dev);
      }
    }
  }

  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  vector<Kokkos::View<ScalarT*,AssemblyDevice> > params_kv, param_dot_kv;
  
  auto p_kv = param_sol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto pslice = Kokkos::subview(p_kv, Kokkos::ALL(), 0);
  
  if (data_avail) {
    params_kv.push_back(pslice);
  }
  else {
    auto p_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),pslice);
    Kokkos::deep_copy(p_dev,pslice);
    params_kv.push_back(p_dev);
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Set up the worksets and allocate the local residual and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////
  
  if (isTransient) {
    // TMW: tmp fix
    auto butcher_c = Kokkos::create_mirror_view(wkset[block]->butcher_c);
    Kokkos::deep_copy(butcher_c, wkset[block]->butcher_c);
    ScalarT timeval = current_time + butcher_c(wkset[block]->current_stage)*deltat;
    
    wkset[block]->setTime(timeval);
    wkset[block]->setDeltat(deltat);
    wkset[block]->alpha = 1.0/deltat;
  }
    
  wkset[block]->isTransient = isTransient;
  wkset[block]->isAdjoint = false;
  
  /////////////////////////////////////////////////////////////////////////////
  // Volume contribution
  /////////////////////////////////////////////////////////////////////////////
  
  // Note: Cannot parallelize over groups since data structures are re-used
  
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    
    wkset[block]->localEID = grp;
    
    /////////////////////////////////////////////////////////////////////////////
    // Perform the necessary local gathers (now stored in group meta data)
    /////////////////////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*gather_timer);
    
      this->performGather(set, block, grp, false, stage, use_only_sol,
                          sol_kv, sol_stage_kv, sol_prev_kv,
                          phi_kv, phi_stage_kv, phi_prev_kv,
                          params_kv, param_dot_kv);
    }

    /////////////////////////////////////////////////////////////////////////////
    // Compute the local residual and Jacobian on this group
    /////////////////////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*physics_timer);
      
      //////////////////////////////////////////////////////////////
      // Compute res and J=dF/du
      //////////////////////////////////////////////////////////////
    
      // Volumetric contribution
      if (assemble_volume_terms[set][block]) {
        this->updateWorkset<ScalarT>(block, grp, seedwhat, 0);
        physics->volumeResidual<ScalarT>(set, block);
      }
      
      ///////////////////////////////////////////////////////////////////////////
      // Edge/face contribution
      ///////////////////////////////////////////////////////////////////////////
      
      if (assemble_face_terms[set][block]) {
        wkset[block]->isOnSide = true;
        for (size_t s=0; s<groupData[block]->num_sides; s++) {
          this->updateWorksetFace<ScalarT>(block, grp, s);
          physics->faceResidual<ScalarT>(set,block);
        }
        wkset[block]->isOnSide = false;
      }
      
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Scatter into global matrix/vector
    ///////////////////////////////////////////////////////////////////////////
    
    this->scatterRes(set, res_view, groups[block][grp]->LIDs[set], block);
    
  } // group loop
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Boundary terms
  //////////////////////////////////////////////////////////////////////////////////////
  
  if (assemble_boundary_terms[set][block]) {
    
    wkset[block]->isOnSide = true;
    
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      
      if (boundary_groups[block][grp]->numElem > 0) {
        
        /////////////////////////////////////////////////////////////////////////////
        // Compute the local residual and Jacobian on this boundary group
        /////////////////////////////////////////////////////////////////////////////
        wkset[block]->resetResidual();

        this->performBoundaryGather(set, block, grp, false, stage, use_only_sol,
                                    sol_kv, sol_stage_kv, sol_prev_kv,
                                    phi_kv, phi_stage_kv, phi_prev_kv,
                                    params_kv, param_dot_kv);

        this->updateWorksetBoundary<ScalarT>(block, grp, seedwhat);
        physics->boundaryResidual<ScalarT>(set,block);
        physics->fluxConditions<ScalarT>(set,block);
        
        ///////////////////////////////////////////////////////////////////////////
        // Scatter into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        this->scatterRes(set, res_view, boundary_groups[block][grp]->LIDs[set], block);
        
      }
    } // element loop
    wkset[block]->isOnSide = false;
  }
  
}


template<class Node>
void AssemblyManager<Node>::computeJacResBoundary(const int & block, const size_t & grp,
                                                     const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                                     const bool & compute_jacobian, const bool & compute_sens,
                                                     const int & num_active_params, const bool & compute_disc_sens,
                                                     const bool & compute_aux_sens, const bool & store_adjPrev,
                                                     View_Sc3 local_res, View_Sc3 local_J) {
  
#ifndef MrHyDE_NO_AD

  int seedwhat = 0;
  if (compute_jacobian) {
    if (compute_disc_sens) {
      seedwhat = 3;
    }
    else if (compute_aux_sens) {
      seedwhat = 4;
    }
    else {
      seedwhat = 1;
    }
  }
  this->updateWorksetBoundary<AD>(block, grp, seedwhat);
  physics->boundaryResidual<AD>(wkset_AD[block]->current_set, block);
  
  if (compute_jacobian) {
    if (compute_disc_sens) {
      this->updateParamJacBoundary(block, grp, local_J);
    }
    else if (compute_aux_sens){
      this->updateAuxJacBoundary(block, grp, local_J);
    }
    else {
      this->updateJacBoundary(block, grp, isAdjoint, local_J);
    }
  }
  
  if (!isAdjoint) {
    this->updateResBoundary(block, grp, compute_sens, local_res);
  }
#endif

}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateResBoundary(const int & block, const size_t & grp,
                                      const bool & compute_sens, View_Sc3 local_res) {

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD32[block]);
  }
#endif
}


template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateResBoundary(const int & block, const size_t & grp,
                                      const bool & compute_sens, View_Sc3 local_res,
                                      Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  if (compute_sens) {
    
    parallel_for("bgroup update res sens",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int r=0; r<local_res.extent(2); r++) {
            local_res(elem,offsets(n,j),r) -= res_AD(elem,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("bgroup update res",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(elem,offsets(n,j),0) -= res_AD(elem,offsets(n,j)).val();
        }
      }
    });
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateJacBoundary(const int & block, const size_t & grp,
                                      const bool & useadjoint, View_Sc3 local_J) {

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateJacBoundary(const int & block, const size_t & grp,
                                      const bool & useadjoint, View_Sc3 local_J,
                                      Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  if (useadjoint) {
    parallel_for("bgroup update jac sens",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(m,k),offsets(n,j)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("bgroup update jac",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(n,j),offsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateParamJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J) {

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateParamJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J,
                                                   Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  auto paramoffsets = wset->paramoffsets;
  auto numParamDOF = groupData[block]->num_param_dof;
  
  parallel_for("bgroup update param jac",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VECTORSIZE),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numParamDOF.extent(0); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_J(elem,offsets(n,j),paramoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateAuxJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J) {

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateAuxJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J,
                                                 Teuchos::RCP<Workset<EvalT> > & wset) {
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  auto aoffsets = boundary_groups[block][grp]->auxoffsets;
  auto numAuxDOF = groupData[block]->num_aux_dof;
  
  parallel_for("bgroup update aux jac",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VECTORSIZE),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numAuxDOF.extent(0); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(elem,offsets(n,j),aoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(aoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}


///////////////////////////////////////////////////////////////////////////////////////
// Compute the contribution from this Group to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeJacRes(const int & block, const size_t & grp,
                                          const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                          const bool & compute_jacobian, const bool & compute_sens,
                                          const int & num_active_params, const bool & compute_disc_sens,
                                          const bool & compute_aux_sens, const bool & store_adjPrev,
                                          Kokkos::View<ScalarT***,AssemblyDevice> local_res,
                                          Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                                          const bool & assemble_volume_terms,
                                          const bool & assemble_face_terms) {
  
#ifndef MrHyDE_NO_AD

  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  bool fixJacDiag = false;
  
  //////////////////////////////////////////////////////////////
  // Compute the AD-seeded solutions at integration points
  //////////////////////////////////////////////////////////////
  
  int seedwhat = 0;
  if (compute_jacobian) {
    if (compute_disc_sens) {
      seedwhat = 3;
    }
    else if (compute_aux_sens) {
      seedwhat = 4;
    }
    else {
      seedwhat = 1;
    }
  }
  int seedindex = 0;

  //////////////////////////////////////////////////////////////
  // Compute res and J=dF/du
  //////////////////////////////////////////////////////////////
  
  // Volumetric contribution
  if (assemble_volume_terms) {
    //Teuchos::TimeMonitor localtimer(*volumeResidualTimer);
    if (groupData[block]->multiscale) {
      this->updateWorkset<AD>(block, grp, seedwhat, seedindex);
      if (groups[block][grp]->have_sols) {
        groups[block][grp]->subgridModels[groups[block][grp]->subgrid_model_index]->subgridSolver(groups[block][grp]->sol[0], groups[block][grp]->sol_prev[0],
                                              groups[block][grp]->phi[0], wkset_AD[block]->time, isTransient, isAdjoint,
                                              compute_jacobian, compute_sens, num_active_params,
                                              compute_disc_sens, compute_aux_sens,
                                              *wkset_AD[block], groups[block][grp]->subgrid_usernum, 0,
                                              groups[block][grp]->subgradient, store_adjPrev);
      }
      else {
        groups[block][grp]->subgridModels[groups[block][grp]->subgrid_model_index]->subgridSolver(groups[block][grp]->group_data->sol[0], groups[block][grp]->group_data->sol_prev[0],
                                              groups[block][grp]->group_data->phi[0], wkset_AD[block]->time, isTransient, isAdjoint,
                                              compute_jacobian, compute_sens, num_active_params,
                                              compute_disc_sens, compute_aux_sens,
                                              *wkset_AD[block], groups[block][grp]->subgrid_usernum, 0,
                                              groups[block][grp]->subgradient, store_adjPrev);
      }
      fixJacDiag = true;
    }
    else {
      this->updateWorkset<AD>(block, grp, seedwhat, seedindex);
      physics->volumeResidual<AD>(wkset_AD[block]->current_set,groupData[block]->my_block);
    }
  }
  
  // Edge/face contribution
  if (assemble_face_terms) {
    //Teuchos::TimeMonitor localtimer(*faceResidualTimer);
    if (groupData[block]->multiscale) {
      // do nothing
    }
    else {
      for (size_t s=0; s<groupData[block]->num_sides; s++) {
        this->updateWorksetFace<AD>(block, grp, s);
        physics->faceResidual<AD>(wkset_AD[block]->current_set,groupData[block]->my_block);
      }
    }
  }
  
  {
    //Teuchos::TimeMonitor localtimer(*jacobianFillTimer);
    
    // Use AD residual to update local Jacobian
    if (compute_jacobian) {
      if (compute_disc_sens) {
        this->updateParamJac(block, grp, local_J);
      }
      else if (compute_aux_sens){
        this->updateAuxJac(block, grp, local_J);
      }
      else {
        this->updateJac(block, grp, isAdjoint, local_J);
      }
    }
  }
  
  if (compute_jacobian && fixJacDiag) {
    this->fixDiagJac(block, grp, local_J, local_res);
  }
  
  // Update the local residual
  this->updateRes(block, grp, compute_sens, local_res);
    
#endif

}


///////////////////////////////////////////////////////////////////////////////////////
// Just compute the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateRes(const int & block, const size_t & grp,
                                      const bool & compute_sens, View_Sc3 local_res) {

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateRes(const int & block, const size_t & grp,
                                      const bool & compute_sens, View_Sc3 local_res,
                                      Teuchos::RCP<Workset<EvalT> > & wset) {

#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  if (compute_sens) {

    parallel_for("Group res sens",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type r=0; r<local_res.extent(2); r++) {
            local_res(elem,offsets(n,j),r) -= res_AD(elem,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("Group res",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(elem,offsets(n,j),0) -= res_AD(elem,offsets(n,j)).val();
        }
      }
    });
  }
#endif
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
// This is the wrapper function
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateJac(const int & block, const size_t & grp,
                                      const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J) {

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD32[block]);
  }
  
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
// This is the primary function
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateJac(const int & block, const size_t & grp,
                                      const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                                      Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  if (useadjoint) {
    parallel_for("Group J adj",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(m,k),offsets(n,j)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("Group J",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(n,j),offsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  //AssemblyExec::execution_space().fence();
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Place ones on the diagonal of the Jacobian if
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::fixDiagJac(const int & block, const size_t & grp,
                      Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                      Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;

  using namespace std;

  parallel_for("Group fix diag",
               RangePolicy<AssemblyExec>(0,local_J.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT JTOL = 1.0E-14;
    for (size_type var=0; var<offsets.extent(0); var++) {
      for (int dof=0; dof<numDOF(var); dof++) {
        int diag = offsets(var,dof);
        if (abs(local_J(elem,diag,diag)) < JTOL) {
          local_res(elem,diag,0) = 0.0;//-u(elem,var,dof);
          for (int j=0; j<numDOF(var); j++) {
            ScalarT scale = 1.0/((ScalarT)numDOF(var)-1.0);
            local_J(elem,diag,offsets(var,j)) = -scale;
            //if (j!=dof)
            //  local_res(elem,diag,0) += scale*u(elem,var,j);
          }
          local_J(elem,diag,diag) = 1.0;
        }
      }
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////


template<class Node>
void AssemblyManager<Node>::updateParamJac(const int & block, const size_t & grp,
                                           Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateParamJac(block, grp, local_J, wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->updateParamJac(block, grp, local_J, wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->updateParamJac(block, grp, local_J, wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->updateParamJac(block, grp, local_J, wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->updateParamJac(block, grp, local_J, wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->updateParamJac(block, grp, local_J, wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->updateParamJac(block, grp, local_J, wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->updateParamJac(block, grp, local_J, wkset_AD32[block]);
  }
  #endif
}


template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateParamJac(const int & block, const size_t & grp,
                                           Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                                           Teuchos::RCP<Workset<EvalT> > & wset) {
#ifndef MrHyDE_NO_AD
  auto paramoffsets = wset->paramoffsets;
  auto numParamDOF = groupData[block]->num_param_dof;
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  parallel_for("Group param J",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (size_type m=0; m<numParamDOF.extent(0); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_J(elem,offsets(n,j),paramoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateAuxJac(const int & block, const size_t & grp,
                                           Kokkos::View<ScalarT***,AssemblyDevice> local_J) {

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateAuxJac(block, grp, local_J, wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->updateAuxJac(block, grp, local_J, wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->updateAuxJac(block, grp, local_J, wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->updateAuxJac(block, grp, local_J, wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->updateAuxJac(block, grp, local_J, wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->updateAuxJac(block, grp, local_J, wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->updateAuxJac(block, grp, local_J, wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->updateAuxJac(block, grp, local_J, wkset_AD32[block]);
  }
#endif

}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateAuxJac(const int & block, const size_t & grp,
                                           Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                                           Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto aoffsets = groups[block][grp]->auxoffsets;
  auto numDOF = groupData[block]->num_dof;
  auto numAuxDOF = groupData[block]->num_aux_dof;
  
  parallel_for("Group aux J",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (size_type m=0; m<numAuxDOF.extent(0); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(elem,offsets(n,j),aoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(aoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}

