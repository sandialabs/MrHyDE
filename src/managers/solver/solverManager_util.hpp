/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// ========================================================================================
// Set up the logicals and data structures for the fixed DOF (Dirichlet and point constraints)
// ========================================================================================

template<class Node>
void SolverManager<Node>::projectDirichlet(const size_t & set) {
  
  Teuchos::TimeMonitor localtimer(*dbcprojtimer);
  
  debugger->print(1, "**** Starting SolverManager::projectDirichlet()");
  
  assembler->updatePhysicsSet(set);
  
  if (usestrongDBCs) {
    
    if (fixedDOF_soln.size() > set) {
      fixedDOF_soln[set] = linalg->getNewOverlappedVector(set);
    }
    else {
      fixedDOF_soln.push_back(linalg->getNewOverlappedVector(set));
    }
    
    vector_RCP glfixedDOF_soln = linalg->getNewVector(set);
    
    vector_RCP rhs = linalg->getNewOverlappedVector(set);
    matrix_RCP mass = linalg->getNewOverlappedMatrix(set);
    vector_RCP glrhs = linalg->getNewVector(set);
    matrix_RCP glmass = linalg->getNewMatrix(set);
    
    assembler->computeConstraintProjection(set, rhs, mass, is_adjoint, current_time);
    
    linalg->exportMatrixFromOverlapped(set, glmass, mass);
    linalg->exportVectorFromOverlapped(set, glrhs, rhs);
    linalg->fillComplete(glmass);
    
    // TODO BWR -- couldn't think of a good way to protect against
    // the preconditioner failing for HFACE, will need to be handled
    // explicitly in the input file for now (State boundary L2 linear solver)
    linalg->linearSolverBoundaryL2(set, glmass, glrhs, glfixedDOF_soln);
    linalg->importVectorToOverlapped(set, fixedDOF_soln[set], glfixedDOF_soln);
    
  }
  
  debugger->print(1, "**** Finished SolverManager::projectDirichlet()");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setDirichlet(const size_t & set, vector_RCP & u) {
  
  debugger->print("**** Starting SolverManager::setDirichlet ...");
  
  Teuchos::TimeMonitor localtimer(*dbcsettimer);
  
  typedef typename Node::execution_space LA_exec;
  
  if (usestrongDBCs) {
    auto u_kv = u->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    //auto meas_kv = meas->getLocalView<HostDevice>();
    
    if (!scalarDirichletData[set]) {
      if (!staticDirichletData[set]) {
        this->projectDirichlet(set);
      }
      else if (!have_static_Dirichlet_data[set]) {
        this->projectDirichlet(set);
        have_static_Dirichlet_data[set] = true;
      }
    }
    
    //if (!scalarDirichletData && transientDirichletData) {
    //  this->projectDirichlet();
    //}
    
    vector<vector<Kokkos::View<LO*,LA_device> > > dbcDOFs = assembler->fixedDOF[set];
    if (scalarDirichletData[set]) {
      
      for (size_t block=0; block<dbcDOFs.size(); ++block) {
        for (size_t v=0; v<dbcDOFs[block].size(); v++) {
          if (dbcDOFs[block][v].extent(0)>0) {
            ScalarT value = scalarDirichletValues[set][block][v];
            auto cdofs = dbcDOFs[block][v];
            parallel_for("solver initial scalar",
                         RangePolicy<LA_exec>(0,cdofs.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const int i ) {
              u_kv(cdofs(i),0) = value;
            });
          }
        }
      }
    }
    else {
      auto dbc_kv = fixedDOF_soln[set]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
      for (size_t block=0; block<dbcDOFs.size(); ++block) {
        for (size_t v=0; v<dbcDOFs[block].size(); v++) {
          if (dbcDOFs[block][v].extent(0)>0) {
            auto cdofs = dbcDOFs[block][v];
            parallel_for("solver initial scalar",
                         RangePolicy<LA_exec>(0,cdofs.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const int i ) {
              u_kv(cdofs(i),0) = dbc_kv(cdofs(i),0);
            });
          }
        }
      }
    }
    
    // set point dbcs
    vector<vector<GO> > pointDOFs = disc->point_dofs[set];
    for (size_t block=0; block<blocknames.size(); ++block) {
      vector<GO> pt_dofs = pointDOFs[block];
      Kokkos::View<LO*,LA_device> ptdofs("pointwise dofs", pointDOFs[block].size());
      auto ptdofs_host = Kokkos::create_mirror_view(ptdofs);
      for (size_t i = 0; i < pt_dofs.size(); i++) {
        LO row = linalg->overlapped_map[set]->getLocalElement(pt_dofs[i]); // TMW: this is a temporary fix
        ptdofs_host(i) = row;
      }
      Kokkos::deep_copy(ptdofs,ptdofs_host);
      parallel_for("solver initial scalar",
                   RangePolicy<LA_exec>(0,ptdofs.extent(0)),
                   KOKKOS_CLASS_LAMBDA (const int i ) {
        LO row = ptdofs(i);
        u_kv(row,0) = 0.0; // fix to zero for now
      });
    }
  }
  
  debugger->print("**** Finished SolverManager::setDirichlet");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > SolverManager<Node>::setInitialParams() {
  vector_RCP initial = linalg->getNewOverlappedParamVector();
  ScalarT value = 2.0;
  initial->putScalar(value);
  return initial;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > SolverManager<Node>::setInitial() {
  
  Teuchos::TimeMonitor localtimer(*initsettimer);
  typedef typename Node::execution_space LA_exec;
  
  debugger->print("**** Starting SolverManager::setInitial ...");
  
  vector<vector_RCP> initial_solns;
  
  if (use_restart) {
    for (size_t set=0; set<restart_solution.size(); ++set) {
      initial_solns.push_back(restart_solution[set]);
    }
  }
  else {

    for (size_t set=0; set<setnames.size(); ++set) {
      assembler->updatePhysicsSet(set);
      
      vector_RCP initial = linalg->getNewOverlappedVector(set);
      initial->putScalar(0.0);
      
      bool samedevice = true;
      bool usehost = false;
      if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyMem>::accessible) {
        samedevice = false;
        if (!Kokkos::SpaceAccessibility<LA_exec, HostMem>::accessible) {
          usehost = true;
        }
        else {
          // output an error
        }
      }
      
      if (have_initial_conditions[set]) {
        if (scalarInitialData[set]) {
          
          auto initial_kv = initial->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
          
          for (size_t block=0; block<assembler->groupData.size(); block++) {
            
            assembler->updatePhysicsSet(set);
            
            if (assembler->groupData[block]->num_elem > 0) {
              
              Kokkos::View<ScalarT*,LA_device> idata("scalar initial data",scalarInitialValues[set][block].size());
              auto idata_host = Kokkos::create_mirror_view(idata);
              for (size_t i=0; i<scalarInitialValues[set][block].size(); i++) {
                idata_host(i) = scalarInitialValues[set][block][i];
              }
              Kokkos::deep_copy(idata,idata_host);
              
              if (samedevice) {
                auto offsets = assembler->wkset[block]->offsets;
                auto numDOF = assembler->groupData[block]->num_dof;
                for (size_t cell=0; cell<assembler->groups[block].size(); cell++) {
                  auto LIDs = assembler->groups[block][cell]->LIDs[set];
                  parallel_for("solver initial scalar",
                               RangePolicy<LA_exec>(0,LIDs.extent(0)),
                               KOKKOS_CLASS_LAMBDA (const int e ) {
                    for (size_type n=0; n<numDOF.extent(0); n++) {
                      for (int i=0; i<numDOF(n); i++ ) {
                        initial_kv(LIDs(e,offsets(n,i)),0) = idata(n);
                      }
                    }
                  });
                }
              }
              else if (usehost) {
                auto offsets = assembler->wkset[block]->offsets;
                auto host_offsets = Kokkos::create_mirror_view(offsets);
                Kokkos::deep_copy(host_offsets,offsets);
                auto numDOF = assembler->groupData[block]->num_dof_host;
                for (size_t cell=0; cell<assembler->groups[block].size(); cell++) {
                  auto LIDs = assembler->groups[block][cell]->LIDs_host[set];
                  parallel_for("solver initial scalar",
                               RangePolicy<LA_exec>(0,LIDs.extent(0)),
                               KOKKOS_CLASS_LAMBDA (const int e ) {
                    for (size_type n=0; n<numDOF.extent(0); n++) {
                      for (int i=0; i<numDOF(n); i++ ) {
                        initial_kv(LIDs(e,host_offsets(n,i)),0) = idata(n);
                      }
                    }
                  });
                }
              }
              
            }
          }
        }
        else {
          
          vector_RCP glinitial = linalg->getNewVector(set);
          
          if (initial_type == "L2-projection") {
            // Compute the L2 projection of the initial data into the discrete space
            vector_RCP rhs = linalg->getNewOverlappedVector(set);
            matrix_RCP mass = linalg->getNewOverlappedMatrix(set);
            vector_RCP glrhs = linalg->getNewVector(set);
            matrix_RCP glmass = linalg->getNewMatrix(set);
            
            assembler->setInitial(set, rhs, mass, is_adjoint);
            
            linalg->exportMatrixFromOverlapped(set, glmass, mass);
            linalg->exportVectorFromOverlapped(set, glrhs, rhs);
            
            linalg->fillComplete(glmass);
            linalg->linearSolverL2(set, glmass, glrhs, glinitial);
            linalg->importVectorToOverlapped(set, initial, glinitial);
            linalg->resetJacobian(set);
          }
          else if (initial_type == "L2-projection-HFACE") {
            // Similar to above, but the basis support only exists on the mesh skeleton
            // The use case is setting the IC at the coarse-scale
            vector_RCP rhs = linalg->getNewOverlappedVector(set);
            matrix_RCP mass = linalg->getNewOverlappedMatrix(set);
            vector_RCP glrhs = linalg->getNewVector(set);
            matrix_RCP glmass = linalg->getNewMatrix(set);
            
            assembler->setInitialFace(set, rhs, mass, is_adjoint);
            
            linalg->exportMatrixFromOverlapped(set, glmass, mass);
            linalg->exportVectorFromOverlapped(set, glrhs, rhs);
            linalg->fillComplete(glmass);
            
            // With HFACE we ensure the preconditioner is not
            // used for this projection (mass matrix is nearly the identity
            // and can cause issues)
            auto origPreconFlag = linalg->options_L2[set]->use_preconditioner;
            linalg->options_L2[set]->use_preconditioner = false;
            // do the solve
            linalg->linearSolverL2(set, glmass, glrhs, glinitial);
            // set back to original
            linalg->options_L2[set]->use_preconditioner = origPreconFlag;
            
            linalg->importVectorToOverlapped(set, initial, glinitial);
            linalg->resetJacobian(set); // TODO not sure of this
            
          }
          else if (initial_type == "interpolation") {
            
            assembler->setInitial(set, initial, is_adjoint);
            
          }
        }
      }
      
      initial_solns.push_back(initial);
    }
  }
  
  debugger->print("**** Finished SolverManager::setInitial ...");
  
  
  return initial_solns;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void SolverManager<Node>::setBatchID(const int & bID){
  batchID = bID;
  params->batchID = bID;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > SolverManager<Node>::blankState(){
  size_t set = 0; // hard coded since somebody uses this
  vector_RCP F_soln = linalg->getNewOverlappedVector(set);
  return F_soln;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > SolverManager<Node>::getRestartSolution() {
  
  if (restart_solution.size() == 0) {
    for (size_t set=0; set<setnames.size(); ++set) {
      vector_RCP F_soln = linalg->getNewOverlappedVector(set);
      restart_solution.push_back(F_soln);
    }
  }
  return restart_solution;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > SolverManager<Node>::getRestartAdjointSolution() {
  
  if (restart_adjoint_solution.size() == 0) {
    for (size_t set=0; set<setnames.size(); ++set) {
      vector_RCP F_soln = linalg->getNewOverlappedVector(set);
      restart_adjoint_solution.push_back(F_soln);
    }
  }
  return restart_adjoint_solution;
}
