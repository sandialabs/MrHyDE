/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ========================================================================================
// Gather local solutions on groups.
// This intermediate function allows us to copy the data from LA_device to AssemblyDevice only once (if necessary)
// ========================================================================================
    
template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performGather(const size_t & current_set, const size_t & block, const size_t & grp,
                                          const bool & include_adjoint, const size_t & stage, const bool & use_only_sol,
                                          vector<ViewType> & sol, vector<ViewType> & sol_stage, vector<ViewType> & sol_prev,
                                          vector<ViewType> & phi, vector<ViewType> & phi_stage, vector<ViewType> & phi_prev,
                                          vector<ViewType> & param_sol, vector<ViewType> & param_dot) {

  if (use_only_sol) {
    for (size_t set=0; set<sol.size(); ++set) {
      this->performGather(set, block, grp, sol[set], 0, 0);
    }
    if (include_adjoint) {
      for (size_t set=0; set<phi.size(); ++set) {
        this->performGather(set, block, grp, phi[set], 2, 0);
      }
    }
  }
  else {
    // For most sets, we use the solution
    for (size_t set=0; set<sol.size(); ++set) {
      if (set != current_set) {
        this->performGather(set, block, grp, sol[set], 0, 0);
      }
    }
    // For the current set, we use the stage solution
    this->performGather(current_set, block, grp, sol_stage[stage], 0, 0);

    for (size_t stg=0; stg<sol_stage.size(); ++stg) {
      if (stg < stage) { // no need to fill current stage?
        this->performGather4D(current_set, block, grp, sol_stage[stg], 5, stg);
      }
    }
    for (size_t stp=0; stp<sol_prev.size(); ++stp) {
      this->performGather4D(current_set, block, grp, sol_prev[stp], 6, stp);
    }
    
    if (include_adjoint) {
      for (size_t set=0; set<phi.size(); ++set) {
        if (set != current_set) {
          this->performGather(set, block, grp, phi[set], 2, 0);
        }
      }
      this->performGather(current_set, block, grp, phi_stage[stage], 2, 0);
      for (size_t stg=0; stg<phi_stage.size(); ++stg) {
        if (stg < stage) { // no need to fill current stage?
          this->performGather4D(current_set, block, grp, phi_stage[stg], 7, stg);
        }
      }
      for (size_t stp=0; stp<phi_prev.size(); ++stp) {
        this->performGather4D(current_set, block, grp, phi_prev[stp], 8, stp);
      }
    }
  }
  if (params->num_discretized_params > 0) {
    if (param_sol.size() > 0) {
      this->performGather(current_set, block, grp, param_sol[0], 4, 0);
    }
    if (param_dot.size() > 0) {
      this->performGather(current_set, block, grp, param_dot[0], 9, 0);
    }
  }
}

// ========================================================================================
// Gather local solutions on groups.
// This intermediate function allows us to copy the data from LA_device to AssemblyDevice only once (if necessary)
// ========================================================================================
    
template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performBoundaryGather(const size_t & current_set, const size_t & block, const size_t & grp,
                                                  const bool & include_adjoint, const size_t & stage, const bool & use_only_sol,
                                                  vector<ViewType> & sol, vector<ViewType> & sol_stage, vector<ViewType> & sol_prev,
                                                  vector<ViewType> & phi, vector<ViewType> & phi_stage, vector<ViewType> & phi_prev,
                                                  vector<ViewType> & param_sol, vector<ViewType> & param_dot) {
  if (use_only_sol) {
    for (size_t set=0; set<sol.size(); ++set) {
      this->performBoundaryGather(set, block, grp, sol[set], 0, 0);
    }
    if (include_adjoint) {
      for (size_t set=0; set<phi.size(); ++set) {
        this->performBoundaryGather(set, block, grp, phi[set], 2, 0);
      }
    }
  }
  else {
    // For most sets, we use the solution
    for (size_t set=0; set<sol.size(); ++set) {
      if (set != current_set) {
        this->performBoundaryGather(set, block, grp, sol[set], 0, 0);
      }
    }
    // For the current set, we use the stage solution
    this->performBoundaryGather(current_set, block, grp, sol_stage[stage], 0, 0);

    for (size_t stg=0; stg<sol_stage.size(); ++stg) {
      if (stg < stage) { // no need to fill current stage?
        this->performBoundaryGather4D(current_set, block, grp, sol_stage[stg], 5, stg);
      }
    }
    for (size_t stp=0; stp<sol_prev.size(); ++stp) {
      this->performBoundaryGather4D(current_set, block, grp, sol_prev[stp], 6, stp);
    }
    
    if (include_adjoint) {
      for (size_t set=0; set<phi.size(); ++set) {
        if (set != current_set) {
          this->performBoundaryGather(set, block, grp, phi[set], 2, 0);
        }
      }
      this->performBoundaryGather(current_set, block, grp, phi_stage[stage], 2, 0);
      for (size_t stg=0; stg<phi_stage.size(); ++stg) {
        if (stg < stage) { // no need to fill current stage?
          this->performBoundaryGather4D(current_set, block, grp, phi_stage[stg], 7, stg);
        }
      }
      for (size_t stp=0; stp<phi_prev.size(); ++stp) {
        this->performBoundaryGather4D(current_set, block, grp, phi_prev[stp], 8, stp);
      }
    }
  }
  if (params->num_discretized_params > 0) {
    if (param_sol.size() > 0) {
      this->performBoundaryGather(current_set, block, grp, param_sol[0], 4, 0);
    }
    if (param_dot.size() > 0) {
      this->performBoundaryGather(current_set, block, grp, param_dot[0], 9, 0);
    }
  }
}

// ========================================================================================
//
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::performGather(const size_t & set, const size_t & block, const size_t & grp,
                                          vector_RCP & vec, const int & type, const size_t & local_entry) {

  typedef typename Node::device_type LA_device;
  typedef typename Node::execution_space LA_exec;
  
  // Can the LA_device execution_space access the AseemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }
  
  // Grab slices of Kokkos Views and push to AssembleDevice one time (each)
  auto vec_kv = vec->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), 0);
  if (data_avail) {
    this->performGather(set, block, grp, vec_slice, type, local_entry);
  }
  else {
    auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
    Kokkos::deep_copy(vec_dev,vec_slice);
    this->performGather(set, block, grp, vec_dev, type, local_entry);
  }
  
}

// ========================================================================================
//
// ========================================================================================

template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performGather(const size_t & set, const size_t & block, const size_t & grp,
                                          ViewType vec_dev, const int & type, const size_t & local_entry) {
  
  Kokkos::View<LO*,AssemblyDevice> numDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> data;
  Kokkos::View<int**,AssemblyDevice> offsets;
  LIDView LIDs;
  
  switch(type) {
    case 0 :
      LIDs = groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->sol[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 1 : // deprecated (u_dot)
      break;
    case 2 :
      LIDs = groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->phi[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 3 : // deprecated (phi_dot)
      break;
    case 4:
      LIDs = groups[block][grp]->paramLIDs;
      numDOF = groupData[block]->num_param_dof;
      data = groupData[block]->param;
      offsets = wkset[block]->paramoffsets;
      break;
    case 9:
      LIDs = groups[block][grp]->paramLIDs;
      numDOF = groupData[block]->num_param_dof;
      data = groupData[block]->param_dot;
      offsets = wkset[block]->paramoffsets;
      break;
    default :
      cout << "ERROR - NOTHING WAS GATHERED" << endl;
  }
  auto cvec = vec_dev;
  parallel_for("assembly gather",
               RangePolicy<AssemblyExec>(0,LIDs.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type var=0; var<offsets.extent(0); var++) {
      for(int dof=0; dof<numDOF(var); dof++ ) {
        data(elem,var,dof) = cvec(LIDs(elem,offsets(var,dof)));
      }
    }
  });
  
}

// ========================================================================================
// Specialized gather for 4D views, e.g., stage solutions
// ========================================================================================

template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performGather4D(const size_t & set, const size_t & block, const size_t & grp,
                                            ViewType vec_dev, const int & type, const size_t & local_entry) {
  
  Kokkos::View<LO*,AssemblyDevice> numDOF;
  Kokkos::View<ScalarT****,AssemblyDevice> data;
  Kokkos::View<int**,AssemblyDevice> offsets;
  LIDView LIDs;
  
  
  switch(type) {
    case 5: // sol_stage
      LIDs = groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->sol_stage[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 6: // sol_prev
      LIDs = groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->sol_prev[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 7: // phi_stage
      LIDs = groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->phi_stage[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 8: // phi_prev
      LIDs = groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->phi_prev[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    default :
      cout << "ERROR - NOTHING WAS GATHERED" << endl;
  }
  
  auto cvec = vec_dev;
  parallel_for("assembly gather",
               RangePolicy<AssemblyExec>(0,LIDs.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type var=0; var<offsets.extent(0); var++) {
      for(int dof=0; dof<numDOF(var); dof++ ) {
        data(elem,var,dof,local_entry) = cvec(LIDs(elem,offsets(var,dof)));
      }
    }
  });
  
}

// ========================================================================================
//
// ========================================================================================

template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performBoundaryGather(const size_t & set, const size_t & block, const size_t & grp,
                                                  ViewType vec_dev, const int & type, const size_t & local_entry) {
  
  Kokkos::View<LO*,AssemblyDevice> numDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> data;
  Kokkos::View<int**,AssemblyDevice> offsets;
  LIDView LIDs;
  
  
  switch(type) {
    case 0 :
      LIDs = boundary_groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->sol[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 1 : // deprecated (u_dot)
      break;
    case 2 :
      LIDs = boundary_groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->phi[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 3 : // deprecated (phi_dot)
      break;
    case 4:
      LIDs = boundary_groups[block][grp]->paramLIDs;
      numDOF = groupData[block]->num_param_dof;
      data = groupData[block]->param;
      offsets = wkset[block]->paramoffsets;
      break;
    case 9:
      LIDs = boundary_groups[block][grp]->paramLIDs;
      numDOF = groupData[block]->num_param_dof;
      data = groupData[block]->param_dot;
      offsets = wkset[block]->paramoffsets;
      break;
    default :
      cout << "ERROR - NOTHING WAS GATHERED" << endl;
  }
  
  auto cvec = vec_dev;
  parallel_for("assembly gather",
               RangePolicy<AssemblyExec>(0,LIDs.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type var=0; var<offsets.extent(0); var++) {
      for(int dof=0; dof<numDOF(var); dof++ ) {
        data(elem,var,dof) = cvec(LIDs(elem,offsets(var,dof)));
      }
    }
  });
  
}

// ========================================================================================
//
// ========================================================================================

template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performBoundaryGather4D(const size_t & set, const size_t & block, const size_t & grp,
                                                    ViewType vec_dev, const int & type, const size_t & local_entry) {
  
  Kokkos::View<LO*,AssemblyDevice> numDOF;
  Kokkos::View<ScalarT****,AssemblyDevice> data;
  Kokkos::View<int**,AssemblyDevice> offsets;
  LIDView LIDs;
  
  
  switch(type) {
    case 5: // sol_stage
      LIDs = boundary_groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->sol_stage[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 6: // sol_prev
      LIDs = boundary_groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->sol_prev[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 7: // phi_stage
      LIDs = boundary_groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->phi_stage[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    case 8: // phi_prev
      LIDs = boundary_groups[block][grp]->LIDs[set];
      numDOF = groupData[block]->set_num_dof[set];
      data = groupData[block]->phi_prev[set];
      offsets = wkset[block]->set_offsets[set];
      break;
    default :
      cout << "ERROR - NOTHING WAS GATHERED" << endl;
  }
  
  auto cvec = vec_dev;
  parallel_for("assembly gather",
               RangePolicy<AssemblyExec>(0,LIDs.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type var=0; var<offsets.extent(0); var++) {
      for(int dof=0; dof<numDOF(var); dof++ ) {
        data(elem,var,dof,local_entry) = cvec(LIDs(elem,offsets(var,dof)));
      }
    }
  });
  
}


// ========================================================================================
//
// ========================================================================================

template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performBoundaryGather(const size_t & set, ViewType vec_dev, const int & type) {
  
  /*
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    
    Kokkos::View<LO*,AssemblyDevice> numDOF;
    Kokkos::View<ScalarT***,AssemblyDevice> data;
    Kokkos::View<int**,AssemblyDevice> offsets;
    LIDView LIDs;
    
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      if (boundary_groups[block][grp]->numElem > 0) {
        
        switch(type) {
          case 0 :
            LIDs = boundary_groups[block][grp]->LIDs[set];
            numDOF = boundary_groups[block][grp]->group_data->num_dof;
            data = boundary_groups[block][grp]->sol[set];
            offsets = wkset[block]->offsets;
            break;
          case 1 : // deprecated (u_dot)
            break;
          case 2 :
            LIDs = boundary_groups[block][grp]->LIDs[set];
            numDOF = boundary_groups[block][grp]->group_data->num_dof;
            data = boundary_groups[block][grp]->phi[set];
            offsets = wkset[block]->offsets;
            break;
          case 3 : // deprecated (phi_dot)
            break;
          case 4:
            LIDs = boundary_groups[block][grp]->paramLIDs;
            numDOF = boundary_groups[block][grp]->group_data->num_param_dof;
            data = boundary_groups[block][grp]->param;
            offsets = wkset[block]->paramoffsets;
            break;
          default :
            cout << "ERROR - NOTHING WAS GATHERED" << endl;
        }
        
        parallel_for("assembly boundary gather",
                     RangePolicy<AssemblyExec>(0,data.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_t var=0; var<numDOF.extent(0); var++) {
            for(int dof=0; dof<numDOF(var); dof++ ) {
              data(elem,var,dof) = vec_dev(LIDs(elem,offsets(var,dof)));
            }
          }
        });
      }
    }
  }
  */
}
