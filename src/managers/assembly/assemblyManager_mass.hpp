/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::getWeightedMass(const size_t & set,
                                            matrix_RCP & mass,
                                            vector_RCP & diagMass) {
  
  Teuchos::TimeMonitor localtimer(*set_init_timer);
  
  using namespace std;

  debugger->print("**** Starting AssemblyManager::getWeightedMass ...");
  
  typedef typename Node::execution_space LA_exec;
  bool use_atomics_ = false;
  if (LA_exec::concurrency() > 1) {
    use_atomics_ = true;
  }
  bool compute_matrix = true;
  if (lump_mass || matrix_free) {
    compute_matrix = false;
  }
  bool use_jacobi = true;
  if (lump_mass) {
    use_jacobi = false;
  }
  
  typedef typename Tpetra::CrsMatrix<ScalarT, LO, GO, Node >::local_matrix_device_type local_matrix;
  local_matrix localMatrix;
  
  // TMW TODO: This probably won't work if the LA_device is not the AssemblyDevice
  
  if (compute_matrix) {
    localMatrix = mass->getLocalMatrixDevice();
  }
  
  auto diag_view = diagMass->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  // Can the LA_device execution_space access the AssemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }
  
  for (size_t block=0; block<groups.size(); ++block) {
    
    auto offsets = wkset[block]->offsets;
    auto numDOF = groupData[block]->num_dof;
    bool sparse_mass = groupData[block]->use_sparse_mass;

    // Create mirrors on LA_Device
    // This might be unnecessary, but it only happens once per block
    auto offsets_ladev = create_mirror(LA_exec(),offsets);
    deep_copy(offsets_ladev,offsets);
    
    auto numDOF_ladev = create_mirror(LA_exec(),numDOF);
    deep_copy(numDOF_ladev,numDOF);
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      
      auto LIDs = groups[block][grp]->LIDs[set];

      if (sparse_mass) {
        auto curr_mass = groupData[block]->sparse_database_mass[set];
        if (!curr_mass->getStatus()) {
          curr_mass->setLocalColumns(offsets,numDOF);
        }
        auto values = curr_mass->getValues();
        auto local_columns = curr_mass->getLocalColumns();
        auto nnz = curr_mass->getNNZPerRow();
        auto index = groups[block][grp]->basis_index;

        parallel_for("assembly insert Jac",
                       RangePolicy<LA_exec>(0,LIDs.extent(0)),
                       KOKKOS_CLASS_LAMBDA (const int elem ) {
          
            LO eindex = index(elem);
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                LO localrow = offsets(n,j);
                LO globalrow = LIDs(elem,localrow);

                ScalarT val = 0.0;
                if (use_jacobi) {
                  for (size_type k=0; k<nnz(eindex,localrow); k++ ) {
                    LO localcol = offsets(n,local_columns(eindex,localrow,k));
                    LO globalcol = LIDs(elem,localcol);
                    if (globalrow == globalcol) {
                      val = values(eindex,localrow,k);
                    }
                  }
                }
                else {
                  for (size_type k=0; k<nnz(eindex,localrow); k++ ) {
                    val += values(eindex,localrow,k);
                  }
                }
                
                if (use_atomics_) {
                  Kokkos::atomic_add(&(diag_view(globalrow,0)), val);
                }
                else {
                  diag_view(globalrow,0) += val;
                }
                
              }
            }
        });
      }
      else {
        auto localmass = this->getWeightedMass(block, grp, physics->mass_wts[set][block]);
      
        if (data_avail) {
        
          // Build the diagonal of the mass matrix
          // Mostly for Jacobi preconditioning
        
          parallel_for("assembly insert Jac",
                       RangePolicy<LA_exec>(0,LIDs.extent(0)),
                       KOKKOS_CLASS_LAMBDA (const int elem ) {
          
            int row = 0;
            LO rowIndex = 0;
          
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                row = offsets(n,j);
                rowIndex = LIDs(elem,row);
              
                ScalarT val = 0.0;
                if (use_jacobi) {
                  val = localmass(elem,row,row);
                }
                else {
                  for (int k=0; k<numDOF(n); k++) {
                    int col = offsets(n,k);
                    val += abs(localmass(elem,row,col));
                  }
                }
              
                if (use_atomics_) {
                  Kokkos::atomic_add(&(diag_view(rowIndex,0)), val);
                }
                else {
                  diag_view(rowIndex,0) += val;
                }
                
              }
            }
          });
        
          // Build the mass matrix if requested
          if (compute_matrix) {
            parallel_for("assembly insert Jac",
                         RangePolicy<LA_exec>(0,LIDs.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const int elem ) {
              
              int row = 0;
              LO rowIndex = 0;
            
              int col = 0;
              LO cols[1028];
              ScalarT vals[1028];
              for (size_type n=0; n<numDOF.extent(0); ++n) {
                const size_type numVals = numDOF(n);
                for (int j=0; j<numDOF(n); j++) {
                  row = offsets(n,j);
                  rowIndex = LIDs(elem,row);
                  for (int k=0; k<numDOF(n); k++) {
                    col = offsets(n,k);
                    vals[k] = localmass(elem,row,col);
                    cols[k] = LIDs(elem,col);
                  }
                
                  localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, false, use_atomics_);
                }
              }
            });
          }
        
        }
      
      else {
        auto localmass_ladev = create_mirror(LA_exec(),localmass.getView());
        deep_copy(localmass_ladev,localmass.getView());
        
        auto LIDs_ladev = create_mirror(LA_exec(),LIDs);
        deep_copy(LIDs_ladev,LIDs);
        
        // Build the diagonal of the mass matrix
        // Mostly for Jacobi preconditioning
        
        parallel_for("assembly insert Jac",
                     RangePolicy<LA_exec>(0,LIDs_ladev.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const int elem ) {
          
          int row = 0;
          LO rowIndex = 0;
          
          for (size_type n=0; n<numDOF_ladev.extent(0); ++n) {
            for (int j=0; j<numDOF_ladev(n); j++) {
              row = offsets_ladev(n,j);
              rowIndex = LIDs_ladev(elem,row);
              
              ScalarT val = 0.0;
              if (use_jacobi) {
                val = localmass_ladev(elem,row,row);
              }
              else {
                for (int k=0; k<numDOF_ladev(n); k++) {
                  int col = offsets_ladev(n,k);
                  val += localmass_ladev(elem,row,col);
                }
              }
              
              if (use_atomics_) {
                Kokkos::atomic_add(&(diag_view(rowIndex,0)), val);
              }
              else {
                diag_view(rowIndex,0) += val;
              }
              
            }
          }
        });
        
        // Build the mass matrix if requested
        if (compute_matrix) {
          parallel_for("assembly insert Jac",
                       RangePolicy<LA_exec>(0,LIDs_ladev.extent(0)),
                       KOKKOS_CLASS_LAMBDA (const int elem ) {
            
            int row = 0;
            LO rowIndex = 0;
            
            int col = 0;
            LO cols[1028];
            ScalarT vals[1028];
            for (size_type n=0; n<numDOF_ladev.extent(0); ++n) {
              const size_type numVals = numDOF_ladev(n);
              for (int j=0; j<numDOF_ladev(n); j++) {
                row = offsets_ladev(n,j);
                rowIndex = LIDs_ladev(elem,row);
                for (int k=0; k<numDOF_ladev(n); k++) {
                  col = offsets_ladev(n,k);
                  vals[k] = localmass_ladev(elem,row,col);
                  cols[k] = LIDs_ladev(elem,col);
                }
                
                localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, false, use_atomics_);
              }
            }
          });
        }
        
      }
      
      }
      
    }
  }
  
  if (compute_matrix) {
    mass->fillComplete();
  }
  
  debugger->print("**** Finished AssemblyManager::getWeightedMass ...");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::getParamMass(matrix_RCP & mass,
                                         vector_RCP & diagMass) {
  
  Teuchos::TimeMonitor localtimer(*set_init_timer);
  
  using namespace std;

  debugger->print("**** Starting AssemblyManager::getParamMass ...");
  
  typedef typename Node::execution_space LA_exec;
  bool use_atomics_ = false;
  if (LA_exec::concurrency() > 1) {
    use_atomics_ = true;
  }
  bool compute_matrix = true;
  if (lump_mass || matrix_free) {
    compute_matrix = false;
  }
  bool use_jacobi = true;
  if (lump_mass) {
    use_jacobi = false;
  }
  
  typedef typename Tpetra::CrsMatrix<ScalarT, LO, GO, Node >::local_matrix_device_type local_matrix;
  local_matrix localMatrix;
  
  // TMW TODO: This probably won't work if the LA_device is not the AssemblyDevice
  
  if (compute_matrix) {
    localMatrix = mass->getLocalMatrixDevice();
  }
  
  auto diag_view = diagMass->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  // Can the LA_device execution_space access the AssemblyDevice data?
  bool data_avail = true;
  if (!Kokkos::SpaceAccessibility<LA_exec, AssemblyDevice::memory_space>::accessible) {
    data_avail = false;
  }
  
  for (size_t block=0; block<groups.size(); ++block) {
    
    auto offsets = wkset[block]->paramoffsets;
    auto numDOF = groupData[block]->num_param_dof;
    bool sparse_mass = false; //groupData[block]->use_sparse_mass;

    // Create mirrors on LA_Device
    // This might be unnecessary, but it only happens once per block
    auto offsets_ladev = create_mirror(LA_exec(),offsets);
    deep_copy(offsets_ladev,offsets);
    
    auto numDOF_ladev = create_mirror(LA_exec(),numDOF);
    deep_copy(numDOF_ladev,numDOF);
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      
      auto LIDs = groups[block][grp]->paramLIDs;

      if (sparse_mass) {
        /*
        auto curr_mass = groupData[block]->sparse_database_mass[set];
        if (!curr_mass->getStatus()) {
          curr_mass->setLocalColumns(offsets,numDOF);
        }
        auto values = curr_mass->getValues();
        auto local_columns = curr_mass->getLocalColumns();
        auto nnz = curr_mass->getNNZPerRow();
        auto index = groups[block][grp]->basis_index;

        parallel_for("assembly insert Jac",
                       RangePolicy<LA_exec>(0,LIDs.extent(0)),
                       KOKKOS_CLASS_LAMBDA (const int elem ) {
          
            LO eindex = index(elem);
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                LO localrow = offsets(n,j);
                LO globalrow = LIDs(elem,localrow);

                ScalarT val = 0.0;
                if (use_jacobi) {
                  for (size_type k=0; k<nnz(eindex,localrow); k++ ) {
                    LO localcol = offsets(n,local_columns(eindex,localrow,k));
                    LO globalcol = LIDs(elem,localcol);
                    if (globalrow == globalcol) {
                      val = values(eindex,localrow,k);
                    }
                  }
                }
                else {
                  for (size_type k=0; k<nnz(eindex,localrow); k++ ) {
                    val += values(eindex,localrow,k);
                  }
                }
                
                if (use_atomics_) {
                  Kokkos::atomic_add(&(diag_view(globalrow,0)), val);
                }
                else {
                  diag_view(globalrow,0) += val;
                }
                
              }
            }
        });
         */
      }
      else {
        auto localmass = this->getParamMass(block, grp);
      
        if (data_avail) {
        
          // Build the diagonal of the mass matrix
          // Mostly for Jacobi preconditioning
        
          parallel_for("assembly insert Jac",
                       RangePolicy<LA_exec>(0,LIDs.extent(0)),
                       KOKKOS_CLASS_LAMBDA (const int elem ) {
          
            int row = 0;
            LO rowIndex = 0;
          
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                row = offsets(n,j);
                rowIndex = LIDs(elem,row);
              
                ScalarT val = 0.0;
                if (use_jacobi) {
                  val = localmass(elem,row,row);
                }
                else {
                  for (int k=0; k<numDOF(n); k++) {
                    int col = offsets(n,k);
                    val += abs(localmass(elem,row,col));
                  }
                }
              
                if (use_atomics_) {
                  Kokkos::atomic_add(&(diag_view(rowIndex,0)), val);
                }
                else {
                  diag_view(rowIndex,0) += val;
                }
                
              }
            }
          });
        
          // Build the mass matrix if requested
          if (compute_matrix) {
            parallel_for("assembly insert Jac",
                         RangePolicy<LA_exec>(0,LIDs.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const int elem ) {
              
              int row = 0;
              LO rowIndex = 0;
            
              int col = 0;
              LO cols[1028];
              ScalarT vals[1028];
              for (size_type n=0; n<numDOF.extent(0); ++n) {
                const size_type numVals = numDOF(n);
                for (int j=0; j<numDOF(n); j++) {
                  row = offsets(n,j);
                  rowIndex = LIDs(elem,row);
                  for (int k=0; k<numDOF(n); k++) {
                    col = offsets(n,k);
                    vals[k] = localmass(elem,row,col);
                    cols[k] = LIDs(elem,col);
                  }
                
                  localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, false, use_atomics_);
                }
              }
            });
          }
        
        }
      
      else {
        auto localmass_ladev = create_mirror(LA_exec(),localmass.getView());
        deep_copy(localmass_ladev,localmass.getView());
        
        auto LIDs_ladev = create_mirror(LA_exec(),LIDs);
        deep_copy(LIDs_ladev,LIDs);
        
        // Build the diagonal of the mass matrix
        // Mostly for Jacobi preconditioning
        
        parallel_for("assembly insert Jac",
                     RangePolicy<LA_exec>(0,LIDs_ladev.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const int elem ) {
          
          int row = 0;
          LO rowIndex = 0;
          
          for (size_type n=0; n<numDOF_ladev.extent(0); ++n) {
            for (int j=0; j<numDOF_ladev(n); j++) {
              row = offsets_ladev(n,j);
              rowIndex = LIDs_ladev(elem,row);
              
              ScalarT val = 0.0;
              if (use_jacobi) {
                val = localmass_ladev(elem,row,row);
              }
              else {
                for (int k=0; k<numDOF_ladev(n); k++) {
                  int col = offsets_ladev(n,k);
                  val += localmass_ladev(elem,row,col);
                }
              }
              
              if (use_atomics_) {
                Kokkos::atomic_add(&(diag_view(rowIndex,0)), val);
              }
              else {
                diag_view(rowIndex,0) += val;
              }
              
            }
          }
        });
        
        // Build the mass matrix if requested
        if (compute_matrix) {
          parallel_for("assembly insert Jac",
                       RangePolicy<LA_exec>(0,LIDs_ladev.extent(0)),
                       KOKKOS_CLASS_LAMBDA (const int elem ) {
            
            int row = 0;
            LO rowIndex = 0;
            
            int col = 0;
            LO cols[1028];
            ScalarT vals[1028];
            for (size_type n=0; n<numDOF_ladev.extent(0); ++n) {
              const size_type numVals = numDOF_ladev(n);
              for (int j=0; j<numDOF_ladev(n); j++) {
                row = offsets_ladev(n,j);
                rowIndex = LIDs_ladev(elem,row);
                for (int k=0; k<numDOF_ladev(n); k++) {
                  col = offsets_ladev(n,k);
                  vals[k] = localmass_ladev(elem,row,col);
                  cols[k] = LIDs_ladev(elem,col);
                }
                
                localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, false, use_atomics_);
              }
            }
          });
        }
        
      }
      
      }
      
    }
  }
  
  if (compute_matrix) {
    mass->fillComplete();
  }
  
  debugger->print("**** Finished AssemblyManager::getParamMass ...");
  
}


// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::applyMassMatrixFree(const size_t & set, const vector_RCP & x, vector_RCP & y) {
  
  typedef typename Node::execution_space LA_exec;
  bool use_atomics_ = false;
  if (LA_exec::concurrency() > 1) {
    use_atomics_ = true;
  }
  
  auto x_kv = x->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto x_slice = Kokkos::subview(x_kv, Kokkos::ALL(), 0);
  
  auto y_kv = y->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto y_slice = Kokkos::subview(y_kv, Kokkos::ALL(), 0);
  
  for (size_t block=0; block<groups.size(); ++block) {
    auto offsets = wkset[block]->offsets;
    auto numDOF = groupData[block]->num_dof;
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      
      auto cLIDs = groups[block][grp]->LIDs[set];
      
      if (!groups[block][grp]->storeMass) {
        auto twts = groups[block][grp]->wts;
        vector<CompressedView<View_Sc4>> tbasis;
        if (groups[block][grp]->storeAll) { // unlikely case, but enabled
          tbasis = groups[block][grp]->basis;
        }
        else {
          vector<View_Sc4> tmpbasis;
          disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->localElemID, tmpbasis);
          for (size_t i=0; i<tmpbasis.size(); ++i) {
            tbasis.push_back(CompressedView<View_Sc4>(tmpbasis[i]));
          }
        }
        
        for (size_type var=0; var<numDOF.extent(0); var++) {
          int bindex = wkset[block]->usebasis[var];
          CompressedView<View_Sc4> cbasis = tbasis[bindex];
          
          string btype = wkset[block]->basis_types[bindex];
          auto off = subview(offsets,var,ALL());
          ScalarT mwt = physics->mass_wts[set][block][var];
          
          if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
            parallel_for("get mass",
                         RangePolicy<AssemblyExec>(0,twts.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const size_type e ) {
              for (size_type i=0; i<cbasis.extent(1); i++ ) {
                for (size_type j=0; j<cbasis.extent(1); j++ ) {
                  ScalarT massval = 0.0;
                  for (size_type k=0; k<cbasis.extent(2); k++ ) {
                    massval += cbasis(e,i,k,0)*cbasis(e,j,k,0)*twts(e,k)*mwt;
                  }
                  LO indi = cLIDs(e,off(i));
                  LO indj = cLIDs(e,off(j));
                  if (use_atomics_) {
                    Kokkos::atomic_add(&(y_slice(indi)), massval*x_slice(indj));
                  }
                  else {
                    y_slice(indi) += massval*x_slice(indj);
                  }
                }
              }
            });
          }
          else if (btype.substr(0,4) == "HDIV" || btype.substr(0,5) == "HCURL") {
            parallel_for("get mass",
                         RangePolicy<AssemblyExec>(0,twts.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const size_type e ) {
              for (size_type i=0; i<cbasis.extent(1); i++ ) {
                for (size_type j=0; j<cbasis.extent(1); j++ ) {
                  ScalarT massval = 0.0;
                  for (size_type k=0; k<cbasis.extent(2); k++ ) {
                    for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                      massval += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*twts(e,k)*mwt;
                    }
                  }
                  LO indi = cLIDs(e,off(i));
                  LO indj = cLIDs(e,off(j));
                  if (use_atomics_) {
                    Kokkos::atomic_add(&(y_slice(indi)), massval*x_slice(indj));
                  }
                  else {
                    y_slice(indi) += massval*x_slice(indj);
                  }
                }
              }
            });
          }
        }
      }
      else {
        
        if (groupData[block]->use_mass_database) {
          
          bool use_sparse = settings->sublist("Solver").get<bool>("sparse mass format",false);
          if (use_sparse) {
            auto curr_mass = groupData[block]->sparse_database_mass[set];
            auto values = curr_mass->getValues();
            auto nnz = curr_mass->getNNZPerRow();
            auto index = groups[block][grp]->basis_index;

            if (!curr_mass->getStatus()) {
              curr_mass->setLocalColumns(offsets, numDOF);
            }
            
            auto local_columns = curr_mass->getLocalColumns();
            
            parallel_for("get mass",
                         RangePolicy<AssemblyExec>(0,index.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const size_type elem ) {
              LO eindex = index(elem);
           
              // New code that uses sparse data structures
              for (size_type var=0; var<numDOF.extent(0); var++) {
                for (int i=0; i<numDOF(var); i++ ) {
                  LO localrow = offsets(var,i);
                  LO globalrow = cLIDs(elem,localrow);
                  for (size_type k=0; k<nnz(eindex,localrow); k++ ) {
                    LO localcol = offsets(var,local_columns(eindex,localrow,k));
                    LO globalcol = cLIDs(elem,localcol);
                    ScalarT matrixval = values(eindex,localrow,k);
                    if (use_atomics_) {
                      Kokkos::atomic_add(&(y_slice(globalrow)), matrixval*x_slice(globalcol));
                    }
                    else {
                      y_slice(globalrow) += matrixval*x_slice(globalcol);
                    }
                  }
                  
                }
              }
            });
          }
          else {

            auto index = groups[block][grp]->basis_index;

            auto curr_mass = groupData[block]->database_mass[set];

            parallel_for("get mass",
                         RangePolicy<AssemblyExec>(0,index.extent(0)),
                         KOKKOS_CLASS_LAMBDA (const size_type elem ) {
              LO eindex = index(elem);
              // Old code that assumed dense data structures
              for (size_type var=0; var<numDOF.extent(0); var++) {
                for (int i=0; i<numDOF(var); i++ ) {
                  LO localrow = offsets(var,i);
                  LO globalrow = cLIDs(elem,localrow);
                  for (int j=0; j<numDOF(var); j++ ) {
                    LO localcol = offsets(var,j);
                    LO globalcol = cLIDs(elem,localcol);
                    if (use_atomics_) {
                      Kokkos::atomic_add(&(y_slice(globalrow)), curr_mass(eindex,localrow,localcol)*x_slice(globalcol));
                    }
                    else {
                      y_slice(globalrow) += curr_mass(eindex,localrow,localcol)*x_slice(globalcol);
                    }
                  }
                }
              }
            });
          }
          
        }
        else {
          auto curr_mass = groups[block][grp]->local_mass[set];
          parallel_for("get mass",
                       RangePolicy<AssemblyExec>(0,curr_mass.extent(0)),
                       KOKKOS_CLASS_LAMBDA (const size_type elem ) {
            for (size_type var=0; var<numDOF.extent(0); var++) {
              for (int i=0; i<numDOF(var); i++ ) {
                for (int j=0; j<numDOF(var); j++ ) {
                  LO indi = cLIDs(elem,offsets(var,i));
                  LO indj = cLIDs(elem,offsets(var,j));
                  if (use_atomics_) {
                    Kokkos::atomic_add(&(y_slice(indi)), curr_mass(elem,offsets(var,i),offsets(var,j))*x_slice(indj));
                  }
                  else {
                    y_slice(indi) += curr_mass(elem,offsets(var,i),offsets(var,j))*x_slice(indj);
                  }
                }
              }
            }
          });
        }
      }
    }
  }
  
}


// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::getWeightVector(const size_t & set, vector_RCP & wts) {
  
  Teuchos::TimeMonitor localtimer(*set_init_timer);
  
  typedef typename Node::execution_space LA_exec;
  
  debugger->print("**** Starting AssemblyManager::getWeightVector ...");
  
  auto wts_view = wts->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  vector<vector<ScalarT> > normwts = physics->norm_wts[set];
  
  for (size_t block=0; block<groups.size(); ++block) {
    
    auto offsets = wkset[block]->offsets;
    auto numDOF = groupData[block]->num_dof;
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      
      for (size_type n=0; n<numDOF.extent(0); ++n) {
        
        ScalarT val = normwts[block][n];
        auto LIDs = groups[block][grp]->LIDs[set];
        parallel_for("assembly insert Jac",
                     RangePolicy<LA_exec>(0,LIDs.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const int elem ) {
          
          int row = 0;
          LO rowIndex = 0;
          
          for (int j=0; j<numDOF(n); j++) {
            row = offsets(n,j);
            rowIndex = LIDs(elem,row);
            wts_view(rowIndex,0) = val;
          }
          
        });
      }
      
    }
  }
  
  debugger->print("**** Finished AssemblyManager::getWeightVector ...");
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
View_Sc3 AssemblyManager<Node>::getMassBoundary(const int & block, const size_t & grp, const size_t & set) {
  
  View_Sc3 mass("local mass", boundary_groups[block][grp]->numElem,
                boundary_groups[block][grp]->LIDs[set].extent(1),
                boundary_groups[block][grp]->LIDs[set].extent(1));
  
  Kokkos::View<string**,HostDevice> bcs = wkset[block]->var_bcs;
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;
  auto cwts = boundary_groups[block][grp]->wts;
  
  for (size_type n=0; n<numDOF.extent(0); n++) {
    if (bcs(n,boundary_groups[block][grp]->sidenum) == "Dirichlet") { // is this a strong DBC for this variable
      int bind = wkset[block]->usebasis[n];
      auto cbasis = boundary_groups[block][grp]->basis[bind];
      auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
      std::string btype = groupData[block]->basis_types[bind];
      
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bgroup compute mass",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cbasis.extent(1); j++ ) {
              for( size_type k=0; k<cbasis.extent(2); k++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k);
              }
            }
          }
        });
      }
      else if (btype == "HDIV") {
        auto cnormals = boundary_groups[block][grp]->normals;
        View_Sc2 nx, ny, nz;
        nx = cnormals[0];
        if (cnormals.size()>1) {
          ny = cnormals[1];
        }
        if (cnormals.size()>2) {
          nz = cnormals[2];
        }
        parallel_for("bgroup compute mass HDIV",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cbasis.extent(1); j++ ) {
              for( size_type k=0; k<cbasis.extent(2); k++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,0)*nx(e,k)*cbasis(e,j,k,0)*nx(e,k)*cwts(e,k);
                if (cbasis.extent(3)>1) {
                  mass(e,off(i),off(j)) += cbasis(e,i,k,1)*ny(e,k)*cbasis(e,j,k,1)*ny(e,k)*cwts(e,k);
                }
                if (cbasis.extent(3)>2) {
                  mass(e,off(i),off(j)) += cbasis(e,i,k,2)*nz(e,k)*cbasis(e,j,k,2)*nz(e,k)*cwts(e,k);
                }
              }
            }
          }
        });
      }
      else if (btype == "HCURL"){
        // Tangential mass matrix for HCURL basis functions psi.
        // M_ij = \int_\Gamma [psi_i \cdot psi_j - (psi_i \cdot n)(psi_j \cdot n)] dS
        // This properly accounts for the tangential component only,
        // important for curved elements where psi \cdot n may not be exactly zero.
        auto cnormals = boundary_groups[block][grp]->normals;
        View_Sc2 nx, ny, nz;
        nx = cnormals[0];
        if (cnormals.size()>1) {
          ny = cnormals[1];
        }
        if (cnormals.size()>2) {
          nz = cnormals[2];
        }
        
        parallel_for("bgroup compute mass HCURL tangential",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cbasis.extent(1); j++ ) {
              for( size_type k=0; k<cbasis.extent(2); k++ ) {
                // Compute psi_i \cdot n
                ScalarT psi_i_n = cbasis(e,i,k,0)*nx(e,k);
                if (cbasis.extent(3)>1) psi_i_n += cbasis(e,i,k,1)*ny(e,k);
                if (cbasis.extent(3)>2) psi_i_n += cbasis(e,i,k,2)*nz(e,k);
                
                // Compute psi_j \cdot n
                ScalarT psi_j_n = cbasis(e,j,k,0)*nx(e,k);
                if (cbasis.extent(3)>1) psi_j_n += cbasis(e,j,k,1)*ny(e,k);
                if (cbasis.extent(3)>2) psi_j_n += cbasis(e,j,k,2)*nz(e,k);
                
                // Compute psi_i \cdot psi_j
                ScalarT psi_dot = cbasis(e,i,k,0)*cbasis(e,j,k,0);
                if (cbasis.extent(3)>1) psi_dot += cbasis(e,i,k,1)*cbasis(e,j,k,1);
                if (cbasis.extent(3)>2) psi_dot += cbasis(e,i,k,2)*cbasis(e,j,k,2);
                
                // Tangential mass: psi_i \cdot psi_j - (psi_i \cdot n)(psi_j \cdot n)
                mass(e,off(i),off(j)) += (psi_dot - psi_i_n*psi_j_n) * cwts(e,k);
              }
            }
          }
        });
      }
    }
  }
  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
CompressedView<View_Sc3> AssemblyManager<Node>::getMass(const int & block, const size_t & grp) {
  
  size_t set = wkset[block]->current_set;
  View_Sc3 mass_view("local mass", groups[block][grp]->numElem,
                     groups[block][grp]->LIDs[set].extent(1),
                     groups[block][grp]->LIDs[set].extent(1));
  CompressedView<View_Sc3> mass(mass_view);
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;
  auto cwts = groups[block][grp]->wts;
  
  vector<CompressedView<View_Sc4>> tbasis;
  if (groups[block][grp]->storeAll) {
    tbasis = groups[block][grp]->basis;
  }
  else { // goes through this more than once, but really shouldn't be used much anyways
    vector<View_Sc4> tmpbasis,tmpbasis_grad, tmpbasis_curl, tmpbasis_nodes;
    vector<View_Sc3> tmpbasis_div;
    disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->localElemID,
                                    tmpbasis, tmpbasis_grad, tmpbasis_curl,
                                    tmpbasis_div, tmpbasis_nodes);
    for (size_t i=0; i<tmpbasis.size(); ++i) {
      tbasis.push_back(CompressedView<View_Sc4>(tmpbasis[i]));
    }
  }
  
  for (size_type n=0; n<numDOF.extent(0); n++) {
    auto cbasis = tbasis[wkset[block]->usebasis[n]];
    string btype = wkset[block]->basis_types[wkset[block]->usebasis[n]];
    auto off = subview(offsets,n,ALL());
    if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
      parallel_for("Group get mass",
                   RangePolicy<AssemblyExec>(0,mass.extent(0)),
                   KOKKOS_CLASS_LAMBDA (const size_type e ) {
        for(size_type i=0; i<cbasis.extent(1); i++ ) {
          for(size_type j=0; j<cbasis.extent(1); j++ ) {
            for(size_type k=0; k<cbasis.extent(2); k++ ) {
              mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k);
            }
          }
        }
      });
    }
    else if (btype.substr(0,4) == "HDIV" || btype.substr(0,5) == "HCURL") {
      parallel_for("Group get mass",
                   RangePolicy<AssemblyExec>(0,mass.extent(0)),
                   KOKKOS_CLASS_LAMBDA (const size_type e ) {
        for (size_type i=0; i<cbasis.extent(1); i++ ) {
          for (size_type j=0; j<cbasis.extent(1); j++ ) {
            for (size_type k=0; k<cbasis.extent(2); k++ ) {
              for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k);
              }
            }
          }
        }
      });
    }
  }
  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get a local weighted mass matrix
// Not currently using weights
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
CompressedView<View_Sc3> AssemblyManager<Node>::getParamMass(const int & block, const size_t & grp) {
  
  auto numDOF = groupData[block]->num_param_dof;
  
  View_Sc3 mass_view("local mass", groups[block][grp]->numElem,
                     groups[block][grp]->paramLIDs.extent(1),
                     groups[block][grp]->paramLIDs.extent(1));
  CompressedView<View_Sc3> mass;

  //if (groupData[block]->use_mass_database) {
  //  mass = CompressedView<View_Sc3>(groupData[block]->database_mass[set], groups[block][grp]->basis_index);
  //}
  //else {
    auto cwts = groups[block][grp]->wts;
    auto offsets = wkset[block]->paramoffsets;
    vector<CompressedView<View_Sc4>> tbasis;
    mass = CompressedView<View_Sc3>(mass_view);

    if (groups[block][grp]->storeAll || groupData[block]->use_basis_database) {
      tbasis = groups[block][grp]->basis;
    }
    else {
      vector<View_Sc4> tmpbasis;
      disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->localElemID, tmpbasis);
      for (size_t i=0; i<tmpbasis.size(); ++i) {
        tbasis.push_back(CompressedView<View_Sc4>(tmpbasis[i]));
      }
    }

    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = tbasis[wkset[block]->paramusebasis[n]];
    
      string btype = wkset[block]->basis_types[wkset[block]->paramusebasis[n]];
      auto off = subview(offsets,n,ALL());
      ScalarT mwt = 1.0; //masswts[n];
    
      if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
        parallel_for("Group get mass",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const size_type e ) {
          for (size_type i=0; i<cbasis.extent(1); i++ ) {
            for (size_type j=0; j<cbasis.extent(1); j++ ) {
              for (size_type k=0; k<cbasis.extent(2); k++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k)*mwt;
              }
            }
          }
        });
      }
      else if (btype.substr(0,4) == "HDIV" || btype.substr(0,5) == "HCURL") {
        parallel_for("Group get mass",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const size_type e ) {
          for (size_type i=0; i<cbasis.extent(1); i++ ) {
            for (size_type j=0; j<cbasis.extent(1); j++ ) {
              for (size_type k=0; k<cbasis.extent(2); k++ ) {
                for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                  mass(e,off(i),off(j)) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k)*mwt;
                }
              }
            }
          }
        });
      }
    }
  
  //}
  
  if (groups[block][grp]->storeMass) {
    // This assumes they are computed in order
    groups[block][grp]->local_param_mass = mass;
  }

  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get a weighted mass matrix
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
CompressedView<View_Sc3> AssemblyManager<Node>::getWeightedMass(const int & block, const size_t & grp, vector<ScalarT> & masswts) {
  
  size_t set = wkset[block]->current_set;
  auto numDOF = groupData[block]->num_dof;
  
  View_Sc3 mass_view("local mass", groups[block][grp]->numElem,
                     groups[block][grp]->LIDs[set].extent(1),
                     groups[block][grp]->LIDs[set].extent(1));
  CompressedView<View_Sc3> mass;

  if (groupData[block]->use_mass_database) {
    mass = CompressedView<View_Sc3>(groupData[block]->database_mass[set], groups[block][grp]->basis_index);
  }
  else {
    auto cwts = groups[block][grp]->wts;
    auto offsets = wkset[block]->offsets;
    vector<CompressedView<View_Sc4>> tbasis;
    mass = CompressedView<View_Sc3>(mass_view);

    if (groups[block][grp]->storeAll || groupData[block]->use_basis_database) {
      tbasis = groups[block][grp]->basis;
    }
    else {
      vector<View_Sc4> tmpbasis;
      disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->localElemID, tmpbasis);
      for (size_t i=0; i<tmpbasis.size(); ++i) {
        tbasis.push_back(CompressedView<View_Sc4>(tmpbasis[i]));
      }
    }

    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = tbasis[wkset[block]->usebasis[n]];
    
      string btype = wkset[block]->basis_types[wkset[block]->usebasis[n]];
      auto off = subview(offsets,n,ALL());
      ScalarT mwt = masswts[n];
    
      if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
        parallel_for("Group get mass",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const size_type e ) {
          for (size_type i=0; i<cbasis.extent(1); i++ ) {
            for (size_type j=0; j<cbasis.extent(1); j++ ) {
              for (size_type k=0; k<cbasis.extent(2); k++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k)*mwt;
              }
            }
          }
        });
      }
      else if (btype.substr(0,4) == "HDIV" || btype.substr(0,5) == "HCURL") {
        parallel_for("Group get mass",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const size_type e ) {
          for (size_type i=0; i<cbasis.extent(1); i++ ) {
            for (size_type j=0; j<cbasis.extent(1); j++ ) {
              for (size_type k=0; k<cbasis.extent(2); k++ ) {
                for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                  mass(e,off(i),off(j)) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k)*mwt;
                }
              }
            }
          }
        });
      }
    }
  
  }
  
  if (groups[block][grp]->storeMass) {
    // This assumes they are computed in order
    groups[block][grp]->local_mass.push_back(mass);
  }

  return mass;
}


///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
CompressedView<View_Sc3> AssemblyManager<Node>::getMassFace(const int & block, const size_t & grp) {
  
  size_t set = wkset[block]->current_set;
  
  View_Sc3 mass_view("local mass", groups[block][grp]->numElem,
                     groups[block][grp]->LIDs[set].extent(1),
                     groups[block][grp]->LIDs[set].extent(1));
  CompressedView<View_Sc3> mass(mass_view);

  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;

  // loop over faces of the reference element
  for (size_t face=0; face<groupData[block]->num_sides; face++) {

    this->updateWorksetFace<ScalarT>(block, grp, face);
    auto cwts = wkset[block]->wts_side; // face weights get put into wts_side after update
    for (size_type n=0; n<numDOF.extent(0); n++) {
      
      auto cbasis = wkset[block]->basis_side[wkset[block]->usebasis[n]]; // face basis put here after update
      string btype = wkset[block]->basis_types[wkset[block]->usebasis[n]]; // TODO does this work in general?
      auto off = subview(offsets,n,ALL());

      if (btype.substr(0,5) == "HFACE") {
        // loop over mesh elements
        parallel_for("Group get mass",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_CLASS_LAMBDA (const size_type e ) {
          for(size_type i=0; i<cbasis.extent(1); i++ ) {
            for(size_type j=0; j<cbasis.extent(1); j++ ) {
              for(size_type k=0; k<cbasis.extent(2); k++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k);
              }
            }
          }
        });
      }
      else {
        // TODO ERROR
        cout << "Group::getMassFace() called with non-HFACE basis type!" << endl;
      }
    }
  }
  return mass;
}

