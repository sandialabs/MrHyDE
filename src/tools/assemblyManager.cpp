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

AssemblyManager::AssemblyManager(const Teuchos::RCP<LA_MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
               Teuchos::RCP<panzer_stk::STK_Interface> & mesh_, Teuchos::RCP<discretization> & disc_,
               Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager<int,int> > & DOF_,
               vector<vector<Teuchos::RCP<cell> > > & cells_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_), DOF(DOF_), cells(cells_) {
  
  // Get the required information from the settings
  verbosity = settings->get<int>("verbosity",0);
  usestrongDBCs = settings->sublist("Solver").get<bool>("use strong DBCs",true);
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("Use Measurements as DBCs", false);
  
  // needed information from the mesh
  mesh->getElementBlockNames(blocknames);
  
  // needed information from the physics interface
  numVars = phys->numVars; //
  varlist = phys->varlist;
  
  // needed information from the disc interface
  
  for (size_t b=0; b<cells.size(); b++) {
    vector<size_t> localIds;
    DRV blocknodes;
    panzer_stk::workset_utils::getIdsAndVertices(*mesh, blocknames[b], localIds, blocknodes);
    elemnodes.push_back(blocknodes);
    
  }
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

void AssemblyManager::createWorkset(const vector<basis_RCP> & param_basis) {
  
  for (size_t b=0; b<cells.size(); b++) {
    wkset.push_back(Teuchos::rcp( new workset(cells[b][0]->getInfo(), disc->ref_ip[b],
                                              disc->ref_wts[b], disc->ref_side_ip[b],
                                              disc->ref_side_wts[b], disc->basis_types[b],
                                              disc->basis_pointers[b],
                                              param_basis,
                                              mesh->getCellTopology(blocknames[b])) ) );
    
    wkset[b]->isInitialized = true;
    wkset[b]->block = b;
  }
  
  phys->setWorkset(wkset);
  
  
}

// ========================================================================================
// ========================================================================================

void AssemblyManager::updateJacDBC(matrix_RCP & J, size_t & e, size_t & block, int & fieldNum,
                  size_t & localSideId, const bool & compute_disc_sens) {
  
  // given a "block" and the unknown field update jacobian to enforce Dirichlet BCs
  
  string blockID = blocknames[block];
  vector<int> GIDs;// = cells[block][e]->GIDs;
  DOF->getElementGIDs(e, GIDs, blockID);
  
  const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fieldNum,
                                                                                   (phys->spaceDim)-1, localSideId);
  
  const vector<int> elmtOffset = SideIndex.first; // local nodes on that side
  const vector<int> basisIdMap = SideIndex.second;
  
  Teuchos::Array<ScalarT> vals(1);
  Teuchos::Array<GO> cols(1);
  
  for( size_t i=0; i<elmtOffset.size(); i++ ) { // for each node
    int row = GIDs[elmtOffset[i]]; // global row
    if (compute_disc_sens) {
      vector<int> paramGIDs;// = cells[block][e]->paramGIDs;
      paramDOF->getElementGIDs(e, paramGIDs, blockID);
      for( size_t col=0; col<paramGIDs.size(); col++ ) {
        int ind = paramGIDs[col];
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

void AssemblyManager::updateJacDBC(matrix_RCP & J, const vector<int> & dofs, const bool & compute_disc_sens) {
  
  // given a "block" and the unknown field update jacobian to enforce Dirichlet BCs
  
  //size_t numcols = J->getGlobalNumCols();
  for( int i=0; i<dofs.size(); i++ ) { // for each node
    if (compute_disc_sens) {
      size_t numcols = globalParamUnknowns;
      for( int col=0; col<numcols; col++ ) {
        ScalarT m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        //J.ReplaceGlobalValues(row, 1, &m_val, &ind);
        J->replaceGlobalValues(col, 1, &m_val, &dofs[i]);
      }
    }
    else {
      size_t numcols = J->getGlobalNumCols();
      for( int col=0; col<numcols; col++ ) {
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
  vector<int> elemGIDs;
  DOF->getElementGIDs(e, elemGIDs, blockID); // compute element global IDs
  
  int numRes = resid->getNumVectors();
  const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fieldNum,
                                                                                   (phys->spaceDim)-1,
                                                                                   localSideId);
  const vector<int> elmtOffset = SideIndex.first; // local nodes on that side
  const vector<int> basisIdMap = SideIndex.second;
  
  for( size_t i=0; i<elmtOffset.size(); i++ ) { // for each node
    int row = elemGIDs[elmtOffset[i]]; // global row
    ScalarT r_val = 0.0; // set residual to 0
    for( int j=0; j<numRes; j++ ) {
      resid->replaceGlobalValue(row, j, r_val); // replace the value
    }
  }
}

// ========================================================================================
// ========================================================================================

void AssemblyManager::updateResDBC(vector_RCP & resid, const vector<int> & dofs) {
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
  vector<int> elemGIDs;// = cells[block][e]->GIDs[p];
  DOF->getElementGIDs(e, elemGIDs, blockID);
  
  int numRes = resid->getNumVectors();
  const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fnum,
                                                                                   (phys->spaceDim)-1,
                                                                                   localSideId);
  const vector<int> elmtOffset = SideIndex.first; // local nodes on that side
  const vector<int> basisIdMap = SideIndex.second;
  
  DRV I_elemNodes = this->getElemNodes(block,e);//cells[block][e]->nodes;
  
  for( size_t i=0; i<elmtOffset.size(); i++ ) { // for each node
    int row = elemGIDs[elmtOffset[i]]; // global row
    ScalarT x = I_elemNodes(0,basisIdMap[i],0);
    ScalarT y = 0.0;
    if (phys->spaceDim > 1)
      y = I_elemNodes(0,basisIdMap[i],1);
    ScalarT z = 0.0;
    if (phys->spaceDim > 2)
      z = I_elemNodes(0,basisIdMap[i],2);
    
    AD diri_FAD;
    diri_FAD = phys->getDirichletValue(block, x, y, z, current_time, varlist[block][fieldNum], gside, false, wkset[block]);
    ScalarT r_val = 0.0;
    size_t numDerivs = diri_FAD.size();
    for( int j=0; j<numRes; j++ ) {
      if (numDerivs > j)
      r_val = diri_FAD.fastAccessDx(j);
      //cout << "resDBC: " << row << j << r_val << endl;
      resid->replaceGlobalValue(row, j, r_val); // replace the value
    }
  }
}

// ========================================================================================
// ========================================================================================

/*
void AssemblyManager::setDirichlet(vector_RCP & initial) {
  
  auto init_kv = initial->getLocalView<HostDevice>();
  //auto meas_kv = meas->getLocalView<HostDevice>();
  
  // TMW: this function needs to be fixed
  vector<vector<int> > fixedDOFs = phys->dbc_dofs;
  
  for (size_t b=0; b<cells.size(); b++) {
    string blockID = blocknames[b];
    Kokkos::View<int**,HostDevice> side_info;
    
    for (int n=0; n<numVars[b]; n++) {
      
      vector<size_t> localDirichletSideIDs = phys->localDirichletSideIDs[b][n];
      vector<size_t> boundDirichletElemIDs = phys->boundDirichletElemIDs[b][n];
      int fnum = DOF->getFieldNum(varlist[b][n]);
      for( size_t e=0; e<disc->myElements[b].size(); e++ ) { // loop through all the elements
        side_info = phys->getSideInfo(b,n,e);
        int numSides = side_info.dimension(0);
        DRV I_elemNodes = this->getElemNodes(b,e);//cells[b][e]->nodes;
        // enforce the boundary conditions if the element is on the given boundary
        
        for( int i=0; i<numSides; i++ ) {
          if( side_info(i,0)==1 ) {
            vector<int> elemGIDs;
            int gside_index = side_info(i,1);
            string gside = phys->sideSets[gside_index];
            size_t elemID = disc->myElements[b][e];
            DOF->getElementGIDs(elemID, elemGIDs, blockID); // global index of each node
            // get the side index and the node->global mapping for the side that is on the boundary
            const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fnum,
                                                                                             (phys->spaceDim)-1, i);
            const vector<int> elmtOffset = SideIndex.first;
            const vector<int> basisIdMap = SideIndex.second;
            // for each node that is on the boundary side
            for( size_t j=0; j<elmtOffset.size(); j++ ) {
              // get the global row and coordinate
              int row =  LA_overlapped_map->getLocalElement(elemGIDs[elmtOffset[j]]);
              ScalarT x = I_elemNodes(0,basisIdMap[j],0);
              ScalarT y = 0.0;
              if (phys->spaceDim > 1) {
                y = I_elemNodes(0,basisIdMap[j],1);
              }
              ScalarT z = 0.0;
              if (phys->spaceDim > 2) {
                z = I_elemNodes(0,basisIdMap[j],2);
              }
              
              if (use_meas_as_dbcs) {
                //init_kv(row,0) = meas_kv(row,0);
              }
              else {
                // put the value into the soln vector
                AD diri_FAD_tmp;
                diri_FAD_tmp = phys->getDirichletValue(b, x, y, z, current_time, varlist[b][n], gside, useadjoint, wkset[b]);
                
                init_kv(row,0) = diri_FAD_tmp.val();
              }
            }
          }
        }
      }
    }
    // set point dbcs
    vector<int> dbc_dofs = fixedDOFs[b];
    
    for (int i = 0; i < dbc_dofs.size(); i++) {
      int row = LA_overlapped_map->getLocalElement(dbc_dofs[i]);
      init_kv(row,0) = 0.0; // fix to zero for now
    }
    
  }
  
}
*/

// ========================================================================================
// ========================================================================================

void AssemblyManager::setInitial(vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint) {
  
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      
      int numElem = cells[b][e]->numElem;
      vector<vector<int> > GIDs = cells[b][e]->GIDs;
      
      Kokkos::View<ScalarT**,AssemblyDevice> localrhs = cells[b][e]->getInitial(true, useadjoint);
      Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[b][e]->getMass();
      
      // assemble into global matrix
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<GIDs[c].size(); row++ ) {
          int rowIndex = GIDs[c][row];
          ScalarT val = localrhs(c,row);
          rhs->sumIntoGlobalValue(rowIndex,0, val);
          for( size_t col=0; col<GIDs[c].size(); col++ ) {
            int colIndex = GIDs[c][col];
            ScalarT val = localmass(c,row,col);
            mass->insertGlobalValues(rowIndex, 1, &val, &colIndex);
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
      vector<vector<int> > GIDs = cells[b][e]->GIDs;
      Kokkos::View<ScalarT**,AssemblyDevice> localinit = cells[b][e]->getInitial(false, useadjoint);
      int numElem = cells[b][e]->numElem;
      for (int c=0; c<numElem; c++) {
        
        for( size_t row=0; row<GIDs[c].size(); row++ ) {
          int rowIndex = GIDs[c][row];
          ScalarT val = localinit(c,row);
          initial->replaceGlobalValue(rowIndex,0, val);
        }
      }
    }
  }
  
}

// ========================================================================================
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
  
  int numRes = res->getNumVectors();
  
  Teuchos::TimeMonitor localassemblytimer(*assemblytimer);
  
  for (size_t b=0; b<cells.size(); b++) {
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Set up the worksets and allocate the local residual and Jacobians
    //////////////////////////////////////////////////////////////////////////////////////
    
    wkset[b]->time = current_time;
    wkset[b]->time_KV(0) = current_time;
    wkset[b]->isTransient = isTransient;
    wkset[b]->isAdjoint = useadjoint;
    wkset[b]->alpha = alpha;
    if (isTransient)
      wkset[b]->deltat = 1.0/alpha;
    else
      wkset[b]->deltat = 1.0;
    
    int numElem = cells[b][0]->numElem;
    int numDOF = cells[b][0]->GIDs[0].size();
    
    int numParamDOF = 0;
    if (compute_disc_sens) {
      numParamDOF = cells[b][0]->paramGIDs[0].size();
    }
    
    Kokkos::View<ScalarT***,AssemblyDevice> local_res, local_J, local_Jdot;
    
    if (compute_sens) {
      local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,numDOF,num_active_params);
    }
    else {
      local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,numDOF,1);
    }
    
    if (compute_disc_sens) {
      local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,numDOF,numParamDOF);
      local_Jdot = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian dot",numElem,numDOF,numParamDOF);
    }
    else {
      local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,numDOF,numDOF);
      local_Jdot = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian dot",numElem,numDOF,numDOF);
    }
    
    //Kokkos::View<ScalarT**,AssemblyDevice> aPrev;
    
    /////////////////////////////////////////////////////////////////////////////
    // Loop over cells
    /////////////////////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*gathertimer);
    
      // Local gather of solutions (should be a better way to do this)
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
      
      
      if (isTransient && useadjoint && !cells[0][0]->multiscale) {
        //aPrev = cells[b][e]->adjPrev;
        //KokkosTools::print(aPrev);
        // TMW: commented for now
        if (is_final_time) {
          for (int i=0; i<cells[b][e]->adjPrev.dimension(0); i++) {
            for (int j=0; j<cells[b][e]->adjPrev.dimension(1); j++) {
              cells[b][e]->adjPrev(i,j) = 0.0;
            }
          }
          //(cells[b][e]->adjPrev).initialize(0.0);
        }
      }
      
      
      /////////////////////////////////////////////////////////////////////////////
      // Compute the local residual and Jacobian on this cell
      /////////////////////////////////////////////////////////////////////////////
      
      {
        Teuchos::TimeMonitor localtimer(*phystimer);
        
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<numDOF; n++) {
            for (int s=0; s<local_res.dimension(2); s++) {
              local_res(p,n,s) = 0.0;
            }
            for (int s=0; s<local_J.dimension(2); s++) {
              local_J(p,n,s) = 0.0;
              local_Jdot(p,n,s) = 0.0;
            }
          }
        }
        
        cells[b][e]->computeJacRes(current_time, isTransient, useadjoint, compute_jacobian, compute_sens,
                                   num_active_params, compute_disc_sens, false, store_adjPrev,
                                   local_res, local_J, local_Jdot);
        
      }
      
      //KokkosTools::print(local_J);
      //KokkosTools::print(local_res);
      /*
      if (isTransient && useadjoint && !cells[0][0]->multiscale) {
        if (gNLiter == 0)
          cells[b][e]->adjPrev = aPrev;
        else if (!store_adjPrev)
          cells[b][e]->adjPrev = aPrev;
      }
      */
      
      ///////////////////////////////////////////////////////////////////////////
      // Insert into global matrix/vector
      ///////////////////////////////////////////////////////////////////////////
      
      {
        Teuchos::TimeMonitor localtimer(*inserttimer);
        vector<vector<int> > GIDs = cells[b][e]->GIDs;
        
        vector<vector<int> > paramGIDs = cells[b][e]->paramGIDs;
        
        for (int i=0; i<GIDs.size(); i++) {
          Teuchos::Array<ScalarT> vals(GIDs[i].size());
          Teuchos::Array<LO> cols(GIDs[i].size());
          
          for( size_t row=0; row<GIDs[i].size(); row++ ) {
            int rowIndex = GIDs[i][row];
            for (int g=0; g<numRes; g++) {
              ScalarT val = local_res(i,row,g);
              res->sumIntoGlobalValue(rowIndex,g, val);
            }
            if (compute_jacobian) {
              if (compute_disc_sens) {
                for( size_t col=0; col<paramGIDs[i].size(); col++ ) {
                  int colIndex = paramGIDs[i][col];
                  ScalarT val = local_J(i,row,col) + alpha*local_Jdot(i,row,col);
                  J->insertGlobalValues(colIndex, 1, &val, &rowIndex);
                }
              }
              else {
                for( size_t col=0; col<GIDs[i].size(); col++ ) {
                  vals[col] = local_J(i,row,col) + alpha*local_Jdot(i,row,col);
                  cols[col] = GIDs[i][col];
                }
                //J->sumIntoGlobalValues(rowIndex, GIDs[i].size(), &vals[0], &GIDs[i][0]);
                J->sumIntoGlobalValues(rowIndex, cols, vals);
                
              }
            }
          }
        }
      }
      
    } // element loop
    
  } // block loop
  
  
  //if (compute_jacobian) {
    //if (compute_disc_sens) {
    //  J->fillComplete(LA_owned_map, param_owned_map);
    //  J->resumeFill();
    //}
    //else {
  //    J->fillComplete();
  //    J->resumeFill();
    //}
  //}
  
  
  // ************************** STRONGLY ENFORCE DIRICHLET BCs *******************************************
  
  if (usestrongDBCs) {
    Teuchos::TimeMonitor localtimer(*dbctimer);
    vector<vector<int> > fixedDOFs = phys->dbc_dofs;
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
  
  //updateResPin(res_over); //pinning attempt
  //if (compute_jacobian) {
  //    updateJacPin(J_over); //pinning attempt
  //}
  
  //auto out = Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
  
  //res->describe(*out, Teuchos::VERB_EXTREME);
  //KokkosTools::print(Comm,res);
  /*
  if (compute_jacobian) {
    if (compute_disc_sens) {
      J->fillComplete(LA_owned_map, param_owned_map);
    }
    else {
      J->fillComplete();
    }
  }
   */
}


// ========================================================================================
//
// ========================================================================================

void AssemblyManager::performGather(const size_t & block, const vector_RCP & vec, const int & type,
                   const size_t & index) {
  
  for (size_t e=0; e < cells[block].size(); e++) {
    cells[block][e]->setLocalSoln(vec, type, index);
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV AssemblyManager::getElemNodes(const int & block, const int & elemID) {
  int nnodes = elemnodes[block].dimension(1);
  
  DRV cnodes("element nodes",1,nnodes,phys->spaceDim);
  for (int i=0; i<nnodes; i++) {
    for (int j=0; j<phys->spaceDim; j++) {
      cnodes(0,i,j) = elemnodes[block](elemID,i,j);
    }
  }
  return cnodes;
}

