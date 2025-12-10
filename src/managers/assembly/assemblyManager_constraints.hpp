/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// Create the constraints
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createConstraints() {
  
  debugger->print("**** Starting AssemblyManager::createFixedDOFs ... ");
  
  // Grab the Dirichlet DOFs from discretization interface
  vector<vector<vector<vector<LO> > > > dbc_dofs = disc->dbc_dofs; // [set][block][var][dof]
  
  // Data is stored per physics set
  for (size_t set=0; set<dbc_dofs.size(); ++set) {
    
    // create a View of bools indicating if a DOF is fixed
    vector<vector<Kokkos::View<LO*,LA_device> > > set_fixedDOF;
    
    // Get the number of DOFs on this set/proc
    int numLocalDof = disc->dof_owned_and_shared[set].size();
    
    // Create storage for logicals (on device)
    Kokkos::View<bool*,LA_device> set_isFixedDOF("logicals for fixed DOFs",numLocalDof);
    
    // Storage on host
    auto fixed_host = Kokkos::create_mirror_view(set_isFixedDOF);
    
    // Fill on host
    for (size_t block=0; block<dbc_dofs[set].size(); block++) {
      for (size_t var=0; var<dbc_dofs[set][block].size(); var++) {
        for (size_t i=0; i<dbc_dofs[set][block][var].size(); i++) {
          LO dof = dbc_dofs[set][block][var][i];
          fixed_host(dof) = true;
        }
      }
    }
    
    // Copy to device
    Kokkos::deep_copy(set_isFixedDOF,fixed_host);
    
    // Store
    isFixedDOF.push_back(set_isFixedDOF);
    
    // Loop over blocks
    for (size_t block=0; block<dbc_dofs[set].size(); block++) {
      
      // Create a View to store DBC DOFs
      vector<Kokkos::View<LO*,LA_device> > block_dofs;
      
      // Loop over variables
      for (size_t var=0; var<dbc_dofs[set][block].size(); var++) {
        
        // Create empty array (var may not have any DBC DOFs)
        Kokkos::View<LO*,LA_device> cfixed;
        
        // Fill array if var has DBC DOFs
        if (dbc_dofs[set][block][var].size()>0) {
          
          // Storage for DOFs (on device)
          cfixed = Kokkos::View<LO*,LA_device>("fixed DOFs",dbc_dofs[set][block][var].size());
          
          // Storage on host
          auto cfixed_host = Kokkos::create_mirror_view(cfixed);
          
          // Fill on host
          for (size_t i=0; i<dbc_dofs[set][block][var].size(); i++) {
            LO dof = dbc_dofs[set][block][var][i];
            cfixed_host(i) = dof;
          }
          
          // Copy to device
          Kokkos::deep_copy(cfixed,cfixed_host);
        }
        // Store var DOFs
        block_dofs.push_back(cfixed);
      }
      // Store block DOFs
      set_fixedDOF.push_back(block_dofs);
    }
    // Store set DOFs
    fixedDOF.push_back(set_fixedDOF);
  }
  
  debugger->print("**** Finished AssemblyManager::createFixedDOFs");
  
}

template<class Node>
void AssemblyManager<Node>::setJacobianConstraints(matrix_RCP & J, const vector<vector<GO> > & dofs,
                                                   const size_t & block, const bool & compute_disc_sens) {
  
  // given a "block" and the unknown field update jacobian to enforce Dirichlet BCs
  for( size_t i=0; i<dofs[block].size(); i++ ) { // for each node
    if (compute_disc_sens) {
      int numcols = globalParamUnknowns; // TMW fix this!
      for( int col=0; col<numcols; col++ ) {
        ScalarT m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        J->replaceGlobalValues(col, 1, &m_val, &dofs[block][i]);
      }
    }
    else {
      GO numcols = J->getGlobalNumCols(); // TMW fix this!
      for( GO col=0; col<numcols; col++ ) {
        ScalarT m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        J->replaceGlobalValues(dofs[block][i], 1, &m_val, &col);
      }
      ScalarT val = 1.0; // set diagonal entry to 1
      J->replaceGlobalValues(dofs[block][i], 1, &val, &dofs[block][i]);
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::setJacobianConstraints(matrix_RCP & J,
                                                   const vector<LO> & dofs,
                                                   const bool & compute_disc_sens) {
  
  if (compute_disc_sens) {
    // nothing to do here
  }
  else {
    for( size_t i=0; i<dofs.size(); i++ ) {
      ScalarT val = 1.0; // set diagonal entry to 1
      J->replaceLocalValues(dofs[i], 1, &val, &dofs[i]);
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::computeConstraintProjection(const size_t & set, vector_RCP & rhs, matrix_RCP & mass,
                                                        const bool & useadjoint,
                                                        const ScalarT & time,
                                                        const bool & lumpmass) {
  
  Teuchos::TimeMonitor localtimer(*set_dbc_timer);
  
  debugger->print("**** Starting AssemblyManager::setDirichlet ...");
  
  // TMW TODO: The Dirichlet BCs are being applied on the host
  //           This is expensive and unnecessary if the LA_Device is not the host device
  //           Will take a fair bit of work to generalize to all cases
  
  auto localMatrix = mass->getLocalMatrixHost();
  
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    //wkset[block]->setTime(time);
    wkset[block]->isOnSide = true;
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      int numElem = boundary_groups[block][grp]->numElem;
      auto LIDs = boundary_groups[block][grp]->LIDs_host[set];
      
      auto localrhs = this->getDirichletBoundary(block, grp, set);
      auto localmass = this->getMassBoundary(block, grp, set);
      auto host_rhs = Kokkos::create_mirror_view(localrhs);
      auto host_mass = Kokkos::create_mirror_view(localmass);
      Kokkos::deep_copy(host_rhs,localrhs);
      Kokkos::deep_copy(host_mass,localmass);
      
      size_t numVals = LIDs.extent(1);
      // assemble into global matrix
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(c,row);
          if (isFixedDOF[set](rowIndex)) {
            ScalarT val = host_rhs(c,row);
            rhs->sumIntoLocalValue(rowIndex,0, val);
            if (lumpmass) {
              LO cols[1];
              ScalarT vals[1];
              
              ScalarT totalval = 0.0;
              for( size_t col=0; col<LIDs.extent(1); col++ ) {
                cols[0] = LIDs(c,col);
                totalval += host_mass(c,row,col);
              }
              vals[0] = totalval;
              localMatrix.sumIntoValues(rowIndex, cols, 1, vals, true, false);
            }
            else {
              LO cols[MAXDERIVS];
              ScalarT vals[MAXDERIVS];
              for( size_t col=0; col<LIDs.extent(1); col++ ) {
                cols[col] = LIDs(c,col);
                vals[col] = host_mass(c,row,col);
              }
              localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, true, false);
              
            }
          }
        }
      }
    }
    wkset[block]->isOnSide = false;
  }
  
  
  // Loop over the groups to put ones on the diagonal for DOFs not on Dirichlet boundaries
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      auto LIDs = groups[block][grp]->LIDs_host[set];
      for (size_t c=0; c<groups[block][grp]->numElem; c++) {
        for( size_type row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(c,row);
          if (!isFixedDOF[set](rowIndex)) {
            ScalarT vals[1];
            LO cols[1];
            vals[0] = 1.0;
            cols[0] = rowIndex;
            localMatrix.replaceValues(rowIndex, cols, 1, vals, true, false);
          }
        }
      }
    }
  }
  
  mass->fillComplete();
  
  debugger->print("**** Finished AssemblyManager::setDirichlet ...");
  
}

// ========================================================================================
// Enforce DOF constraints - includes strong Dirichlet
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::dofConstraints(const size_t & set, matrix_RCP & J, vector_RCP & res,
                                           const ScalarT & current_time,
                                           const bool & compute_jacobian,
                                           const bool & compute_disc_sens) {
  
  debugger->print("******** Starting AssemblyManager::dofConstraints");
  
  Teuchos::TimeMonitor localtimer(*dbc_timer);
  
  if (usestrongDBCs) {
    vector<vector<vector<LO> > > dbcDOFs = disc->dbc_dofs[set];
    for (size_t block=0; block<dbcDOFs.size(); block++) {
      for (size_t var=0; var<dbcDOFs[block].size(); var++) {
        if (compute_jacobian) {
          this->setJacobianConstraints(J,dbcDOFs[block][var],compute_disc_sens);
        }
      }
    }
  }
  
  vector<vector<GO> > fixedDOFs = disc->point_dofs[set];
  for (size_t block=0; block<fixedDOFs.size(); block++) {
    if (compute_jacobian) {
      this->setJacobianConstraints(J,fixedDOFs,block,compute_disc_sens);
    }
  }
  
  debugger->print("******** Finished AssemblyManager::dofConstraints");
  
}

// ========================================================================================
// Get the Dirichlet conditions (RHS for projection)
// ========================================================================================

template<class Node>
View_Sc2 AssemblyManager<Node>::getDirichletBoundary(const int & block, const size_t & grp, const size_t & set) {
  
  View_Sc2 dvals("initial values",boundary_groups[block][grp]->numElem, boundary_groups[block][grp]->LIDs[set].extent(1));
  this->updateWorksetBoundary<ScalarT>(block, grp, 0);
  
  Kokkos::View<string**,HostDevice> bcs = wkset[block]->var_bcs;
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;
  auto cwts = boundary_groups[block][grp]->wts;
  auto cnormals = boundary_groups[block][grp]->normals;

  for (size_t n=0; n<wkset[block]->varlist.size(); n++) {
    if (bcs(n,boundary_groups[block][grp]->sidenum) == "Dirichlet") { // is this a strong DBC for this variable
      int bind = wkset[block]->usebasis[n];
      std::string btype = groupData[block]->basis_types[bind];
      auto cbasis = boundary_groups[block][grp]->basis[bind]; // may fault in memory-saving mode
      
      auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        // Get scalar Dirichlet data for HGRAD/HVOL/HFACE
        auto dip = groupData[block]->physics->getDirichlet(n, set, groupData[block]->my_block, boundary_groups[block][grp]->sidename);
        parallel_for("bgroup fill Dirichlet",
                     RangePolicy<AssemblyExec>(0,cwts.extent(0)),
                     MRHYDE_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cwts.extent(1); j++ ) {
              dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,0)*cwts(e,j);
            }
          }
        });
      }
      else if (btype == "HDIV"){
        // Get scalar Dirichlet data for HDIV (normal component)
        auto dip = groupData[block]->physics->getDirichlet(n, set, groupData[block]->my_block, boundary_groups[block][grp]->sidename);
        
        View_Sc2 nx, ny, nz;
        nx = cnormals[0];
        if (cnormals.size()>1) {
          ny = cnormals[1];
        }
        if (cnormals.size()>2) {
          nz = cnormals[2];
        }
        
        parallel_for("bgroup fill Dirichlet HDIV",
                     RangePolicy<AssemblyExec>(0,dvals.extent(0)),
                     MRHYDE_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cwts.extent(1); j++ ) {
              dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,0)*nx(e,j)*cwts(e,j);
              if (cbasis.extent(3)>1) {
                dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,1)*ny(e,j)*cwts(e,j);
              }
              if (cbasis.extent(3)>2) {
                dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,2)*nz(e,j)*cwts(e,j);
              }
            }
          }
        });
      }
      else if (btype == "HCURL"){
        // Get vector-valued Dirichlet data E = (Ex, Ey, Ez)
        auto dip_vec = groupData[block]->physics->getDirichletVector(n, set, groupData[block]->my_block, boundary_groups[block][grp]->sidename);
        View_Sc2 dip_x = dip_vec[0];
        View_Sc2 dip_y = dip_vec[1];
        View_Sc2 dip_z = dip_vec[2];
        
        // Get normals n = (nx, ny, nz) for tangential projection
        View_Sc2 nx, ny, nz;
        nx = cnormals[0];
        if (cnormals.size()>1) {
          ny = cnormals[1];
        }
        if (cnormals.size()>2) {
          nz = cnormals[2];
        }
        
        // Tangential projection for HCURL basis functions psi.
        // RHS: b_i = \int_\Gamma [E \cdot psi_i - (E \cdot n)(psi_i \cdot n)] dS
        // This projects the tangential component of E onto the HCURL basis.
        // Important for curved elements where psi \cdot n may not be exactly zero.
        parallel_for("bgroup fill Dirichlet HCURL tangential",
                     RangePolicy<AssemblyExec>(0,dvals.extent(0)),
                     MRHYDE_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cwts.extent(1); j++ ) {
              // E \cdot n
              ScalarT E_dot_n = dip_x(e,j)*nx(e,j);
              if (cbasis.extent(3)>1) E_dot_n += dip_y(e,j)*ny(e,j);
              if (cbasis.extent(3)>2) E_dot_n += dip_z(e,j)*nz(e,j);
              
              // psi \cdot n
              ScalarT psi_dot_n = cbasis(e,i,j,0)*nx(e,j);
              if (cbasis.extent(3)>1) psi_dot_n += cbasis(e,i,j,1)*ny(e,j);
              if (cbasis.extent(3)>2) psi_dot_n += cbasis(e,i,j,2)*nz(e,j);
              
              // E \cdot psi
              ScalarT E_dot_psi = dip_x(e,j)*cbasis(e,i,j,0);
              if (cbasis.extent(3)>1) E_dot_psi += dip_y(e,j)*cbasis(e,i,j,1);
              if (cbasis.extent(3)>2) E_dot_psi += dip_z(e,j)*cbasis(e,i,j,2);
              
              // Tangential projection: E \cdot psi - (E \cdot n)(psi \cdot n)
              dvals(e,off(i)) += (E_dot_psi - E_dot_n*psi_dot_n) * cwts(e,j);
            }
          }
        });
      }
    }
  }
  return dvals;
}
