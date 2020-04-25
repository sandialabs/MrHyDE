/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "assemblyManager.hpp"
#include <boost/algorithm/string.hpp>


// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

AssemblyManager::AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
                                 Teuchos::RCP<panzer_stk::STK_Interface> & mesh_, Teuchos::RCP<discretization> & disc_,
                                 Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager> & DOF_,
                                 vector<vector<Teuchos::RCP<cell> > > & cells_,
                                 vector<vector<Teuchos::RCP<BoundaryCell> > > & boundaryCells_,
                                 Teuchos::RCP<ParameterManager> & params_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_), DOF(DOF_), cells(cells_), boundaryCells(boundaryCells_), params(params_) {
  
  // Get the required information from the settings
  milo_debug_level = settings->get<int>("debug level",0);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting assembly manager constructor ..." << endl;
    }
  }
  
  verbosity = settings->get<int>("verbosity",0);
  usestrongDBCs = settings->sublist("Solver").get<bool>("use strong DBCs",true);
  useNewBCs = settings->sublist("Solver").get<bool>("use new BCs",true);
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("Use Measurements as DBCs", false);
  
  // needed information from the mesh
  mesh->getElementBlockNames(blocknames);
  
  // needed information from the physics interface
  numVars = phys->numVars; //
  varlist = phys->varlist;
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished assembly manager constructor" << endl;
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

void AssemblyManager::createWorkset() {
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createWorkset ..." << endl;
    }
  }
  
  for (size_t b=0; b<cells.size(); b++) {
    wkset.push_back(Teuchos::rcp( new workset(cells[b][0]->getInfo(), disc->ref_ip[b],
                                              disc->ref_wts[b], disc->ref_side_ip[b],
                                              disc->ref_side_wts[b], disc->basis_types[b],
                                              disc->basis_pointers[b],
                                              params->discretized_param_basis,
                                              mesh->getCellTopology(blocknames[b]),
                                              phys->var_bcs[b]) ) );
    
    wkset[b]->isInitialized = true;
    wkset[b]->block = b;
  }
  
  //phys->setWorkset(wkset);
  //params->wkset = wkset;
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::createWorkset" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

void AssemblyManager::updateJacDBC(matrix_RCP & J, size_t & e, size_t & block, int & fieldNum,
                                   size_t & localSideId, const bool & compute_disc_sens) {
  
  // given a "block" and the unknown field update jacobian to enforce Dirichlet BCs
  
  string blockID = blocknames[block];
  vector<GO> GIDs;// = cells[block][e]->GIDs;
  DOF->getElementGIDs(e, GIDs, blockID);
  
  const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fieldNum,
                                                                                   (phys->spaceDim)-1, localSideId);
  
  const vector<int> elmtOffset = SideIndex.first; // local nodes on that side
  const vector<int> basisIdMap = SideIndex.second;
  
  Teuchos::Array<ScalarT> vals(1);
  Teuchos::Array<GO> cols(1);
  
  for( size_t i=0; i<elmtOffset.size(); i++ ) { // for each node
    GO row = GIDs[elmtOffset[i]]; // global row
    if (compute_disc_sens) {
      vector<GO> paramGIDs;// = cells[block][e]->paramGIDs;
      params->paramDOF->getElementGIDs(e, paramGIDs, blockID);
      for( size_t col=0; col<paramGIDs.size(); col++ ) {
        GO ind = paramGIDs[col];
        ScalarT m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        //J.ReplaceGlobalValues(row, 1, &m_val, &ind);
        J->replaceGlobalValues(ind, 1, &m_val, &row);
      }
    }
    else {
      for( size_t col=0; col<GIDs.size(); col++ ) {
        cols[0] = GIDs[col];
        vals[0] = 0.0; // set ALL of the entries to 0 in the Jacobian
        //J->replaceGlobalValues(row, 1, &m_val, &ind);
        J->replaceGlobalValues(row, cols, vals);
      }
      cols[0] = row;
      vals[0] = 1.0; // set diagonal entry to 1
      //J->replaceGlobalValues(row, 1, &val, &row);
      J->replaceGlobalValues(row, cols, vals);
      //cout << Comm->getRank() << "  " << row << "  " << vals[0] << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================

void AssemblyManager::updateJacDBC(matrix_RCP & J, const vector<GO> & dofs, const bool & compute_disc_sens) {
  
  // given a "block" and the unknown field update jacobian to enforce Dirichlet BCs
  
  //size_t numcols = J->getGlobalNumCols();
  for( int i=0; i<dofs.size(); i++ ) { // for each node
    if (compute_disc_sens) {
      int numcols = globalParamUnknowns; // TMW
      for( int col=0; col<numcols; col++ ) {
        ScalarT m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        //J.ReplaceGlobalValues(row, 1, &m_val, &ind);
        J->replaceGlobalValues(col, 1, &m_val, &dofs[i]);
      }
    }
    else {
      GO numcols = J->getGlobalNumCols();
      for( GO col=0; col<numcols; col++ ) {
        ScalarT m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        J->replaceGlobalValues(dofs[i], 1, &m_val, &col);
      }
      ScalarT val = 1.0; // set diagonal entry to 1
      J->replaceGlobalValues(dofs[i], 1, &val, &dofs[i]);
    }
  }
}

// ========================================================================================
// ========================================================================================

void AssemblyManager::updateResDBC(vector_RCP & resid, size_t & e, size_t & block, int & fieldNum,
                  size_t & localSideId) {
  // given a "block" and the unknown field update resid to enforce Dirichlet BCs
  
  string blockID = blocknames[block];
  vector<GO> elemGIDs;
  DOF->getElementGIDs(e, elemGIDs, blockID); // compute element global IDs
  
  int numRes = resid->getNumVectors();
  const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fieldNum,
                                                                                   (phys->spaceDim)-1,
                                                                                   localSideId);
  const vector<int> elmtOffset = SideIndex.first; // local nodes on that side
  const vector<int> basisIdMap = SideIndex.second;
  
  for( size_t i=0; i<elmtOffset.size(); i++ ) { // for each node
    GO row = elemGIDs[elmtOffset[i]]; // global row
    ScalarT r_val = 0.0; // set residual to 0
    for( int j=0; j<numRes; j++ ) {
      resid->replaceGlobalValue(row, j, r_val); // replace the value
    }
  }
}

// ========================================================================================
// ========================================================================================

void AssemblyManager::updateResDBC(vector_RCP & resid, const vector<GO> & dofs) {
  // given a "block" and the unknown field update resid to enforce Dirichlet BCs
  
  int numRes = resid->getNumVectors();
  
  for( size_t i=0; i<dofs.size(); i++ ) { // for each node
    ScalarT r_val = 0.0; // set residual to 0
    for( int j=0; j<numRes; j++ ) {
      resid->replaceGlobalValue(dofs[i], j, r_val); // replace the value
    }
  }
}


// ========================================================================================
// ========================================================================================

void AssemblyManager::updateResDBCsens(vector_RCP & resid, size_t & e, size_t & block, int & fieldNum, size_t & localSideId,
                      const std::string & gside, const ScalarT & current_time) {
  
  
  int fnum = DOF->getFieldNum(varlist[block][fieldNum]);
  string blockID = blocknames[block];
  vector<GO> elemGIDs;// = cells[block][e]->GIDs[p];
  DOF->getElementGIDs(e, elemGIDs, blockID);
  
  int numRes = resid->getNumVectors();
  const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fnum,
                                                                                   (phys->spaceDim)-1,
                                                                                   localSideId);
  const vector<int> elmtOffset = SideIndex.first; // local nodes on that side
  const vector<int> basisIdMap = SideIndex.second;
  
  for( size_t i=0; i<elmtOffset.size(); i++ ) { // for each node
    GO row = elemGIDs[elmtOffset[i]]; // global row
    ScalarT r_val = 0.0;
    for( int j=0; j<numRes; j++ ) {
      resid->replaceGlobalValue(row, j, r_val); // replace the value
    }
  }
}

// ========================================================================================
// ========================================================================================

void AssemblyManager::setInitial(vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                                 const bool & lumpmass) {
  
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      
      int numElem = cells[b][e]->numElem;
      Kokkos::View<GO**,HostDevice> GIDs = cells[b][e]->GIDs;
      
      Kokkos::View<ScalarT**,AssemblyDevice> localrhs = cells[b][e]->getInitial(true, useadjoint);
      Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[b][e]->getMass();
      
      // assemble into global matrix
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<GIDs.extent(1); row++ ) {
          GO rowIndex = GIDs(c,row);
          ScalarT val = localrhs(c,row);
          rhs->sumIntoGlobalValue(rowIndex,0, val);
          if (lumpmass) {
            Teuchos::Array<ScalarT> vals(1);
            Teuchos::Array<GO> cols(1);
            
            ScalarT totalval = 0.0;
            for( size_t col=0; col<GIDs.extent(1); col++ ) {
              cols[0] = GIDs(c,col);
              totalval += localmass(c,row,col);
            }
            vals[0] = totalval;
            mass->sumIntoGlobalValues(rowIndex, cols, vals);
          }
          else {
            Teuchos::Array<ScalarT> vals(GIDs.extent(1));
            Teuchos::Array<GO> cols(GIDs.extent(1));
            
            for( size_t col=0; col<GIDs.extent(1); col++ ) {
              cols[col] = GIDs(c,col);
              vals[col] = localmass(c,row,col);
            }
            mass->sumIntoGlobalValues(rowIndex, cols, vals);
            
          }
        }
      }
    }
  }
  
  mass->fillComplete();
}

// ========================================================================================
// ========================================================================================

void AssemblyManager::setInitial(vector_RCP & initial, const bool & useadjoint) {

  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      Kokkos::View<GO**,HostDevice> GIDs = cells[b][e]->GIDs;
      Kokkos::View<ScalarT**,AssemblyDevice> localinit = cells[b][e]->getInitial(false, useadjoint);
      int numElem = cells[b][e]->numElem;
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<GIDs.extent(1); row++ ) {
          int rowIndex = GIDs(c,row);
          ScalarT val = localinit(c,row);
          initial->replaceGlobalValue(rowIndex,0, val);
        }
      }
    }
  }
  
}

// ========================================================================================
// Wrapper to the main assembly routine to assemble over all blocks (most common use case)
// ========================================================================================

void AssemblyManager::assembleJacRes(vector_RCP & u, vector_RCP & u_dot,
                                     vector_RCP & phi, vector_RCP & phi_dot,
                                     const ScalarT & alpha, const ScalarT & beta,
                                     const bool & compute_jacobian, const bool & compute_sens,
                                     const bool & compute_disc_sens,
                                     vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                     const ScalarT & current_time,
                                     const bool & useadjoint, const bool & store_adjPrev,
                                     const int & num_active_params,
                                     vector_RCP & Psol, const bool & is_final_time) {
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting AssemblyManager::assembleJacRes ..." << endl;
    }
  }
  
  for (size_t b=0; b<cells.size(); b++) {
    this->assembleJacRes(u, u_dot, phi, phi_dot, alpha, beta, compute_jacobian,
                         compute_sens, compute_disc_sens, res, J, isTransient,
                         current_time, useadjoint, store_adjPrev, num_active_params,
                         Psol, is_final_time, b);
  }
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Finished AssemblyManager::assembleJacRes" << endl;
    }
  }
}

// ========================================================================================
// Main assembly routine ... only assembles on a given block (b)
// ========================================================================================

void AssemblyManager::assembleJacRes(vector_RCP & u, vector_RCP & u_dot,
                                     vector_RCP & phi, vector_RCP & phi_dot,
                                     const ScalarT & alpha, const ScalarT & beta,
                                     const bool & compute_jacobian, const bool & compute_sens,
                                     const bool & compute_disc_sens,
                                     vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                     const ScalarT & current_time,
                                     const bool & useadjoint, const bool & store_adjPrev,
                                     const int & num_active_params,
                                     vector_RCP & Psol, const bool & is_final_time,
                                     const int & b) {
    
  
  
  int numRes = res->getNumVectors();
  
  Teuchos::TimeMonitor localassemblytimer(*assemblytimer);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Set up the worksets and allocate the local residual and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////
  
  wkset[b]->time = current_time;
  wkset[b]->time_KV(0) = current_time; // TMW issue -- current_time in not on Device
  wkset[b]->isTransient = isTransient;
  wkset[b]->isAdjoint = useadjoint;
  wkset[b]->alpha = alpha;
  if (isTransient) {
    wkset[b]->deltat = 1.0/alpha;
  }
  else {
    wkset[b]->deltat = 1.0;
  }
  int numElem = cells[b][0]->numElem;
  int numDOF = cells[b][0]->GIDs.extent(1);
  
  int numParamDOF = 0;
  if (compute_disc_sens) {
    numParamDOF = cells[b][0]->paramGIDs.extent(1); // is this on host
  }
  
  // This data needs to be available on Host and Device
  // Optimizing layout for AssemblyExec
  Kokkos::View<ScalarT***,UnifiedDevice> local_res, local_J, local_Jdot;
  
  if (compute_sens) {
    local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,numDOF,num_active_params);
  }
  else {
    local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,numDOF,1);
  }
  
  if (compute_disc_sens) {
    local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,numDOF,numParamDOF);
    local_Jdot = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian dot",numElem,numDOF,numParamDOF);
  }
  else {
    local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,numDOF,numDOF);
    local_Jdot = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian dot",numElem,numDOF,numDOF);
  }
  
  //Kokkos::View<ScalarT**,AssemblyDevice> aPrev;
  
  /////////////////////////////////////////////////////////////////////////////
  // Perform gather to cells
  /////////////////////////////////////////////////////////////////////////////
  
  {
    Teuchos::TimeMonitor localtimer(*gathertimer);
    
    // Local gather of solutions
    this->performGather(b,u,0,0);
    this->performGather(b,u_dot,1,0);
    this->performGather(b,Psol,4,0);
    if (useadjoint) {
      this->performGather(b,phi,2,0);
      this->performGather(b,phi_dot,3,0);
    }
  }
  
  
  /////////////////////////////////////////////////////////////////////////////
  // Volume contribution
  /////////////////////////////////////////////////////////////////////////////
  
  for (size_t e=0; e < cells[b].size(); e++) {
    
    wkset[b]->localEID = e;
    cells[b][e]->updateData();
    
    if (isTransient && useadjoint && !cells[0][0]->cellData->multiscale) {
      if (is_final_time) {
        cells[b][e]->resetAdjPrev(0.0);
      }
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // Compute the local residual and Jacobian on this cell
    /////////////////////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*phystimer);
      
      // This could be done on either host or device
      parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int p ) {
        for (int n=0; n<local_res.extent(1); n++) {
          for (int s=0; s<local_res.extent(2); s++) {
            local_res(p,n,s) = 0.0;
          }
          for (int s=0; s<local_J.extent(2); s++) {
            local_J(p,n,s) = 0.0;
            local_Jdot(p,n,s) = 0.0;
          }
        }
      });
      
      cells[b][e]->computeJacRes(current_time, isTransient, useadjoint, compute_jacobian, compute_sens,
                                 num_active_params, compute_disc_sens, false, store_adjPrev,
                                 local_res, local_J, local_Jdot);
      
    }
    
    // This should be ok if KokkosTools is updated for device
    // GH: commented this out for now; it can't tell DRV from DRVint
    /*
    if (milo_debug_level > 2) {
      if (Comm->getRank() == 0) {
        KokkosTools::print(local_res, "inside assemblyManager.cpp, res after volumetric");
        KokkosTools::print(local_J, "inside assemblyManager.cpp, J after volumetric");
        KokkosTools::print(local_Jdot, "inside assemblyManager.cpp, J_dot after volumetric");
      }
    }*/
    
    ///////////////////////////////////////////////////////////////////////////
    // Insert into global matrix/vector
    ///////////////////////////////////////////////////////////////////////////
    
    this->insert(J, res, local_res, local_J, local_Jdot,
                 cells[b][e]->GIDs, cells[b][e]->paramGIDs,
                 compute_jacobian, compute_disc_sens, alpha);
    
    
  } // element loop
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Boundary terms
  //////////////////////////////////////////////////////////////////////////////////////
  
  
  if (!cells[0][0]->cellData->multiscale) {
    //for (size_t b=0; b<boundaryCells.size(); b++) {
      
      {
        Teuchos::TimeMonitor localtimer(*gathertimer);
        
        // Local gather of solutions
        // Do not need to gather u_dot or phi_dot on boundaries (for now)
        this->performBoundaryGather(b,u,0,0);
        this->performBoundaryGather(b,Psol,4,0);
        if (useadjoint) {
          this->performBoundaryGather(b,phi,2,0);
        }
      }
      
      for (size_t e=0; e < boundaryCells[b].size(); e++) {
        
        if (boundaryCells[b][e]->numElem > 0) {
          wkset[b]->localEID = e;
          
          //////////////////////////////////////////////////////////////////////////////////////
          // Set up the worksets and allocate the local residual and Jacobians
          //////////////////////////////////////////////////////////////////////////////////////
          
          int numElem = boundaryCells[b][e]->numElem;
          int numDOF = boundaryCells[b][e]->GIDs.extent(1);
          
          int numParamDOF = 0;
          if (compute_disc_sens) {
            numParamDOF = boundaryCells[b][e]->paramGIDs.extent(1);
          }
          
          Kokkos::View<ScalarT***,UnifiedDevice> local_res, local_J, local_Jdot;
          
          if (compute_sens) {
            local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,numDOF,num_active_params);
          }
          else {
            local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,numDOF,1);
          }
          
          if (compute_disc_sens) {
            local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,numDOF,numParamDOF);
            local_Jdot = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian dot",numElem,numDOF,numParamDOF);
          }
          else {
            local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,numDOF,numDOF);
            local_Jdot = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian dot",numElem,numDOF,numDOF);
          }
          
          /////////////////////////////////////////////////////////////////////////////
          // Compute the local residual and Jacobian on this cell
          /////////////////////////////////////////////////////////////////////////////
          
          {
            Teuchos::TimeMonitor localtimer(*phystimer);
            
            parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int p ) {
              for (int n=0; n<local_res.extent(1); n++) {
                for (int s=0; s<local_res.extent(2); s++) {
                  local_res(p,n,s) = 0.0;
                }
              }
              for (int n=0; n<local_J.extent(1); n++) {
                for (int s=0; s<local_J.extent(2); s++) {
                  local_J(p,n,s) = 0.0;
                  local_Jdot(p,n,s) = 0.0;
                }
              }
            });
            
            boundaryCells[b][e]->computeJacRes(current_time, isTransient, useadjoint, compute_jacobian, compute_sens,
                                               num_active_params, compute_disc_sens, false, store_adjPrev,
                                               local_res, local_J, local_Jdot);
            
          }
          
          // GH: commented this out for now; it can't tell DRV from DRVint
          /*
          if (milo_debug_level > 2) {
            if (Comm->getRank() == 0) {
              KokkosTools::print(local_res, "inside assemblyManager.cpp, res after boundary");
              KokkosTools::print(local_J, "inside assemblyManager.cpp, J after boundary");
              KokkosTools::print(local_Jdot, "inside assemblyManager.cpp, J_dot after boundary");
            }
          }*/
          
          ///////////////////////////////////////////////////////////////////////////
          // Insert into global matrix/vector
          ///////////////////////////////////////////////////////////////////////////
          
          this->insert(J, res, local_res, local_J, local_Jdot,
                       boundaryCells[b][e]->GIDs, boundaryCells[b][e]->paramGIDs,
                       compute_jacobian, compute_disc_sens, alpha);
          
          
        }
      } // element loop
      
    //} // block loop
  }
  if (usestrongDBCs) {
    this->pointConstraints(J, res, current_time, compute_jacobian, compute_disc_sens);
  }
}


// ************************** STRONGLY ENFORCE DIRICHLET BCs *******************************************
void AssemblyManager::pointConstraints(matrix_RCP & J, vector_RCP & res,
                                       const ScalarT & current_time,
                                       const bool & compute_jacobian,
                                       const bool & compute_disc_sens) {
  Teuchos::TimeMonitor localtimer(*dbctimer);
  vector<vector<GO> > fixedDOFs = phys->dbc_dofs;
  for (size_t b=0; b<cells.size(); b++) {
    vector<size_t> boundDirichletElemIDs;   // list of elements on the Dirichlet boundary
    vector<size_t> localDirichletSideIDs;   // local side numbers for Dirichlet boundary sides
    vector<size_t> globalDirichletSideIDs;   // local side numbers for Dirichlet boundary sides
    for (int n=0; n<numVars[b]; n++) {
      int fnum = DOF->getFieldNum(varlist[b][n]);
      boundDirichletElemIDs = phys->boundDirichletElemIDs[b][n];
      localDirichletSideIDs = phys->localDirichletSideIDs[b][n];
      globalDirichletSideIDs = phys->globalDirichletSideIDs[b][n];
      
      size_t numDBC = boundDirichletElemIDs.size();
      for (size_t e=0; e<numDBC; e++) {
        size_t eindex = boundDirichletElemIDs[e];
        size_t sindex = localDirichletSideIDs[e];
        size_t gside_index = globalDirichletSideIDs[e];
        
        if (compute_jacobian) {
          this->updateJacDBC(J, eindex, b, fnum, sindex, compute_disc_sens);
        }
        std::string gside = phys->sideSets[gside_index];
        this->updateResDBCsens(res, eindex, b, n, sindex, gside, current_time);
      }
    }
    if (compute_jacobian) {
      this->updateJacDBC(J,fixedDOFs[b],compute_disc_sens);
    }
    this->updateResDBC(res,fixedDOFs[b]);
  }
  
}


// ========================================================================================
//
// ========================================================================================

void AssemblyManager::performGather(const size_t & b, const vector_RCP & vec,
                                    const int & type, const size_t & entry) {
  
  // Get a view of the vector on the AssemblyDevice
  // TMW: not sure if this will work correctly.  According to the tpetra documentation, one can request a in either Host or Device
  // This takes care of the memory transfer from Host to Device
  auto vec_kv = vec->getLocalView<AssemblyDevice>();
  
  // Get a corresponding view on the AssemblyDevice
  
  Kokkos::View<LO***,AssemblyDevice> index;
  Kokkos::View<LO*,AssemblyDevice> numDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> data;
  
  // TMW: Does this need to be executed on Device?
  for (size_t c=0; c < cells[b].size(); c++) {
    switch(type) {
      case 0 :
        index = cells[b][c]->index;
        numDOF = cells[b][c]->numDOF;
        data = cells[b][c]->u;
        break;
      case 1 :
        index = cells[b][c]->index;
        numDOF = cells[b][c]->numDOF;
        data = cells[b][c]->u_dot;
        break;
      case 2 :
        index = cells[b][c]->index;
        numDOF = cells[b][c]->numDOF;
        data = cells[b][c]->phi;
        break;
      case 3 :
        index = cells[b][c]->index;
        numDOF = cells[b][c]->numDOF;
        data = cells[b][c]->phi_dot;
        break;
      case 4:
        index = cells[b][c]->paramindex;
        numDOF = cells[b][c]->numParamDOF;
        data = cells[b][c]->param;
        break;
      case 5 :
        index = cells[b][c]->auxindex;
        numDOF = cells[b][c]->numAuxDOF;
        data = cells[b][c]->aux;
        break;
      default :
        cout << "ERROR - NOTHING WAS GATHERED" << endl;
    }
    
    // TMW: commenting this debug statement out for now
    /*
    if (milo_debug_level > 3) {
      if (Comm->getRank() == 0) {
        cout << "inside assemblyManager::gather: type = " << type << endl;
        KokkosTools::print(index, "inside assemblyManager::gather - index");
        KokkosTools::print(numDOF, "inside assemblyManager::gather - numDOF");
        KokkosTools::print(data, "inside assemblyManager::gather - data");
      }
    }*/
    parallel_for(RangePolicy<AssemblyExec>(0,index.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t n=0; n<index.extent(1); n++) {
        for(size_t i=0; i<numDOF(n); i++ ) {
          data(e,n,i) = vec_kv(index(e,n,i),entry);
        }
      }
    });
  }
}

// ========================================================================================
//
// ========================================================================================

void AssemblyManager::performBoundaryGather(const size_t & b, const vector_RCP & vec,
                                            const int & type, const size_t & entry) {
  
  if (boundaryCells.size() > b) {
    
    // Get a view of the vector on the HostDevice
    auto vec_kv = vec->getLocalView<AssemblyDevice>();
    
    // Get a corresponding view on the AssemblyDevice
    
    Kokkos::View<LO***,AssemblyDevice> index;
    Kokkos::View<LO*,AssemblyDevice> numDOF;
    Kokkos::View<ScalarT***,AssemblyDevice> data;
    
    for (size_t c=0; c < boundaryCells[b].size(); c++) {
      if (boundaryCells[b][c]->numElem > 0) {
        
        switch(type) {
          case 0 :
            index = boundaryCells[b][c]->index;
            numDOF = boundaryCells[b][c]->numDOF;
            data = boundaryCells[b][c]->u;
            break;
          case 1 :
            index = boundaryCells[b][c]->index;
            numDOF = boundaryCells[b][c]->numDOF;
            data = boundaryCells[b][c]->u_dot;
            break;
          case 2 :
            index = boundaryCells[b][c]->index;
            numDOF = boundaryCells[b][c]->numDOF;
            data = boundaryCells[b][c]->phi;
            break;
          case 3 :
            index = boundaryCells[b][c]->index;
            numDOF = boundaryCells[b][c]->numDOF;
            data = boundaryCells[b][c]->phi_dot;
            break;
          case 4:
            index = boundaryCells[b][c]->paramindex;
            numDOF = boundaryCells[b][c]->numParamDOF;
            data = boundaryCells[b][c]->param;
            break;
          case 5 :
            index = boundaryCells[b][c]->auxindex;
            numDOF = boundaryCells[b][c]->numAuxDOF;
            data = boundaryCells[b][c]->aux;
            break;
          default :
            cout << "ERROR - NOTHING WAS GATHERED" << endl;
        }
        /*
        if (milo_debug_level > 2) {
          if (Comm->getRank() == 0) {
            KokkosTools::print(index, "inside assemblyManager::boundarygather - index");
            KokkosTools::print(numDOF, "inside assemblyManager::boundarygather - numDOF");
            KokkosTools::print(data, "inside assemblyManager::boundarygather - data");
          }
        }*/
        parallel_for(RangePolicy<AssemblyExec>(0,index.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_t n=0; n<index.extent(1); n++) {
            for(size_t i=0; i<numDOF(n); i++ ) {
              data(e,n,i) = vec_kv(index(e,n,i),entry);
            }
          }
        });
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void AssemblyManager::insert(matrix_RCP & J, vector_RCP & res,
                             Kokkos::View<ScalarT***,UnifiedDevice> & local_res,
                             Kokkos::View<ScalarT***,UnifiedDevice> & local_J,
                             Kokkos::View<ScalarT***,UnifiedDevice> & local_Jdot,
                             Kokkos::View<GO**,HostDevice> & GIDs,
                             Kokkos::View<GO**,HostDevice> & paramGIDs,
                             const bool & compute_jacobian,
                             const bool & compute_disc_sens,
                             const ScalarT & alpha) {

  Teuchos::TimeMonitor localtimer(*inserttimer);
  
  for (int i=0; i<GIDs.extent(0); i++) {
    Teuchos::Array<ScalarT> vals(GIDs.extent(1));
    Teuchos::Array<GO> cols(GIDs.extent(1));
    
    for( size_t row=0; row<GIDs.extent(1); row++ ) {
      GO rowIndex = GIDs(i,row);
      for (int g=0; g<local_res.extent(2); g++) {
        ScalarT val = local_res(i,row,g);
        res->sumIntoGlobalValue(rowIndex,g, val);
      }
      if (compute_jacobian) {
        if (compute_disc_sens) {
          for( size_t col=0; col<paramGIDs.extent(1); col++ ) {
            GO colIndex = paramGIDs(i,col);
            ScalarT val = local_J(i,row,col) + alpha*local_Jdot(i,row,col);
            J->insertGlobalValues(colIndex, 1, &val, &rowIndex);
          }
        }
        else {
          for( size_t col=0; col<GIDs.extent(1); col++ ) {
            vals[col] = local_J(i,row,col) + alpha*local_Jdot(i,row,col);
            cols[col] = GIDs(i,col);
          }
          //J->sumIntoGlobalValues(rowIndex, GIDs[i].size(), &vals[0], &GIDs[i][0]);
          J->sumIntoGlobalValues(rowIndex, cols, vals);
          
        }
      }
    }
  }
}

