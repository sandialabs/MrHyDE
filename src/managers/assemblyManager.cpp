/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "assemblyManager.hpp"
#include "cellMetaData.hpp"

#include <boost/algorithm/string.hpp>


// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

AssemblyManager::AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                 Teuchos::RCP<panzer_stk::STK_Interface> & mesh_, Teuchos::RCP<discretization> & disc_,
                                 Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager> & DOF_,
                                 Teuchos::RCP<ParameterManager> & params_,
                                 const int & numElemPerCell_) :
Comm(Comm_), settings(settings_), mesh(mesh_), disc(disc_), phys(phys_), DOF(DOF_), params(params_), numElemPerCell(numElemPerCell_) {
  
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
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("use measurements as DBCs", false);
  
  string solver_type = settings->sublist("Solver").get<string>("solver","none"); // or "transient"
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  
  // needed information from the mesh
  mesh->getElementBlockNames(blocknames);
  
  // check if we need to assembly volumetric, boundary and face terms
  for (int b=0; b<blocknames.size(); b++) {
    if (settings->sublist("Physics").isSublist(blocknames[b])) {
      assemble_volume_terms.push_back(settings->sublist("Physics").sublist(blocknames[b]).get<bool>("assemble volume terms",true));
      assemble_boundary_terms.push_back(settings->sublist("Physics").sublist(blocknames[b]).get<bool>("assemble boundary terms",true));
      assemble_face_terms.push_back(settings->sublist("Physics").sublist(blocknames[b]).get<bool>("assemble face terms",false));
    }
    else { // meaning all blocks use the same physics settings
      assemble_volume_terms.push_back(settings->sublist("Physics").get<bool>("assemble volume terms",true));
      assemble_boundary_terms.push_back(settings->sublist("Physics").get<bool>("assemble boundary terms",true));
      assemble_face_terms.push_back(settings->sublist("Physics").get<bool>("assemble face terms",false));
    }
  }
  // overwrite assemble_face_terms if HFACE vars are used
  for (size_t b=0; b<blocknames.size(); b++) {
    vector<string> ctypes = phys->unique_types[b];
    for (size_t n=0; n<ctypes.size(); n++) {
      if (ctypes[n] == "HFACE") {
        assemble_face_terms[b] = true;
      }
    }
  }
  
  // determine if we need to build basis functions
  for (int b=0; b<blocknames.size(); b++) {
    if (assemble_volume_terms[b]) {
      build_volume_terms.push_back(true);
    }
    else {
      if (settings->sublist("Physics").isSublist(blocknames[b])) {
        build_volume_terms.push_back(settings->sublist("Physics").sublist(blocknames[b]).get<bool>("build volume terms",true));
      }
      else { // meaning all blocks use the same physics settings
        build_volume_terms.push_back(settings->sublist("Physics").get<bool>("build volume terms",true));
      }
    }
    if (assemble_boundary_terms[b]) {
      build_boundary_terms.push_back(true);
    }
    else {
      if (settings->sublist("Physics").isSublist(blocknames[b])) {
        build_boundary_terms.push_back(settings->sublist("Physics").sublist(blocknames[b]).get<bool>("build boundary terms",true));
      }
      else { // meaning all blocks use the same physics settings
        build_boundary_terms.push_back(settings->sublist("Physics").get<bool>("build boundary terms",true));
      }
    }
    if (assemble_face_terms[b]) {
      build_face_terms.push_back(true);
    }
    else {
      if (settings->sublist("Physics").isSublist(blocknames[b])) {
        build_face_terms.push_back(settings->sublist("Physics").sublist(blocknames[b]).get<bool>("build face terms",false));
      }
      else { // meaning all blocks use the same physics settings
        build_face_terms.push_back(settings->sublist("Physics").get<bool>("build face terms",false));
      }
    }
    
  }
  
  // needed information from the physics interface
  numVars = phys->numVars; //
  varlist = phys->varlist;
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished assembly manager constructor" << endl;
    }
  }
  
  // Create cells/boundary cells
  this->createCells();
  
  // Set up discretized parameters
  params->setupDiscretizedParameters(cells,boundaryCells);
  
  // Create worksets
  //this->createWorkset();
  
  // create fixedDOF View of bools
  vector<vector<vector<LO> > > dbc_dofs = phys->dbc_dofs; // [block][var][dof]
  int numLocalDof = DOF->getNumOwnedAndGhosted();
  isFixedDOF = Kokkos::View<bool*,HostDevice>("logicals for fixed DOFs",numLocalDof);
  for (size_t block=0; block<dbc_dofs.size(); block++) {
    for (size_t var=0; var<dbc_dofs[block].size(); var++) {
      for (size_t i=0; i<dbc_dofs[block][var].size(); i++) {
        LO dof = dbc_dofs[block][var][i];
        isFixedDOF(dof) = true;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Create the cells
////////////////////////////////////////////////////////////////////////////////

void AssemblyManager::createCells() {
  
  Teuchos::TimeMonitor localtimer(*celltimer);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createCells ..." << endl;
    }
  }
  
  int numElem = numElemPerCell;
  
  vector<stk::mesh::Entity> all_meshElems;
  mesh->getMyElements(all_meshElems);
  
  for (size_t b=0; b<blocknames.size(); b++) {
    vector<Teuchos::RCP<cell> > blockcells;
    vector<stk::mesh::Entity> stk_meshElems;
    mesh->getMyElements(blocknames[b], stk_meshElems);
    
    topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
    int numNodesPerElem = cellTopo->getNodeCount();
    int spaceDim = phys->spaceDim;
    size_t numTotalElem = stk_meshElems.size();
    vector<size_t> localIds;
    
    Kokkos::DynRankView<ScalarT,HostDevice> blocknodes("nodes on block",numTotalElem,numNodesPerElem,spaceDim);
    panzer_stk::workset_utils::getIdsAndVertices(*mesh, blocknames[b], localIds, blocknodes); // fill on host
    
    int elemPerCell = settings->sublist("Solver").get<int>("workset size",1);
    bool memeff = settings->sublist("Solver").get<bool>("memory efficient",false);
    int prog = 0;
    
    vector<string> sideSets;
    mesh->getSidesetNames(sideSets);
    Teuchos::RCP<CellMetaData> cellData = Teuchos::rcp( new CellMetaData(settings, cellTopo,
                                                                         phys, b, 0,
                                                                         build_face_terms[b],
                                                                         assemble_face_terms[b],
                                                                         sideSets, disc->ref_ip[b],
                                                                         disc->ref_wts[b], disc->ref_side_ip[b],
                                                                         disc->ref_side_wts[b], disc->basis_types[b],
                                                                         disc->basis_pointers[b],
                                                                         params->num_discretized_params));
    
    // May need to be PHX::Device
    Kokkos::View<const LO**,Kokkos::LayoutRight,HostDevice> LIDs = DOF->getLIDs();
    
    while (prog < numTotalElem) {
      int currElem = elemPerCell;  // Avoid faults in last iteration
      if (prog+currElem > numTotalElem){
        currElem = numTotalElem-prog;
      }
      
      Kokkos::View<LO*,AssemblyDevice> eIndex("element indices",currElem);
      DRV currnodes("currnodes", currElem, numNodesPerElem, spaceDim);
      
      auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
      auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host

      //DRV currnodepert("currnodepert", currElem, numNodesPerElem, spaceDim);
      // Making this loop explicitly on the host
      for (int e=0; e<host_currnodes.extent(0); e++) {
        for (int n=0; n<host_currnodes.extent(1); n++) {
          for (int m=0; m<host_currnodes.extent(2); m++) {
            host_currnodes(e,n,m) = blocknodes(prog+e,n,m);// + blocknodepert(prog+e,n,m);
            //currnodepert(e,n,m) = blocknodepert(prog+e,n,m);
          }
        }
        host_eIndex(e) = prog+e;
      }
      Kokkos::deep_copy(currnodes,host_currnodes);
      Kokkos::deep_copy(eIndex,host_eIndex);
      
      // Build the Kokkos View of the cell GIDs ------
      vector<vector<GO> > cellGIDs;
      int numLocalDOF = 0;
      for (int i=0; i<currElem; i++) {
        vector<GO> GIDs;
        size_t elemID = disc->myElements[b][prog+i]; // TMW: why here?
        DOF->getElementGIDs(elemID, GIDs, blocknames[b]);
        
        //vector<LO> LIDs;
        //DOF->getElementLIDs(elemID, LIDs);
        
        cellGIDs.push_back(GIDs);
        numLocalDOF = GIDs.size(); // should be the same for all elements
      }
      
      Kokkos::View<GO**,HostDevice> hostGIDs("GIDs on host device",currElem,numLocalDOF);
      for (int i=0; i<currElem; i++) {
        for (int j=0; j<numLocalDOF; j++) {
          hostGIDs(i,j) = cellGIDs[i][j];
        }
      }
      
      Kokkos::View<LO**,HostDevice> hostLIDs("LIDs on host device",currElem,numLocalDOF);
      for (int i=0; i<currElem; i++) {
        size_t elemID = disc->myElements[b][prog+i]; // TMW: why here?
        for (int j=0; j<numLocalDOF; j++) {
          hostLIDs(i,j) = LIDs(elemID,j);
        }
      }
      
      // Set the side information (soon to be removed)-
      Kokkos::View<int****,HostDevice> sideinfo = phys->getSideInfo(b,host_eIndex);
      
      Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices", currElem, numNodesPerElem);
      
      for (int i=0; i<currElem; i++) {
        vector<stk::mesh::EntityId> stk_nodeids;
        mesh->getNodeIdsForElement(stk_meshElems[prog+i], stk_nodeids);
        for (int n=0; n<numNodesPerElem; n++) {
          currind(i,n) = stk_nodeids[n];
        }
      }
      
      Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orient_drv("kv to orients",currElem);
      Intrepid2::OrientationTools<AssemblyDevice>::getOrientation(orient_drv, currind, *cellTopo);
      
      blockcells.push_back(Teuchos::rcp(new cell(cellData, currnodes, eIndex, hostGIDs, hostLIDs,
                                                 sideinfo, orient_drv)));
      prog += elemPerCell;
    }
    cells.push_back(blockcells);
    
    //////////////////////////////////////////////////////////////////////////////////
    // Boundary cells
    //////////////////////////////////////////////////////////////////////////////////
    
    vector<Teuchos::RCP<BoundaryCell> > bcells;
    
    if (build_boundary_terms[b]) {
      // TMW: this is just for ease of use
      int numBoundaryElem = settings->sublist("Solver").get<int>("workset size",1);
      
      ///////////////////////////////////////////////////////////////////////////////////
      // Rules for grouping elements into boundary cells
      //
      // 1.  All elements must be on the same processor
      // 2.  All elements must be on the same physical side
      // 3.  Each edge/face on the side must have the same local ID.
      // 4.  No more than numBoundaryElem (= numElem) in a group
      ///////////////////////////////////////////////////////////////////////////////////
      
      for (size_t side=0; side<sideSets.size(); side++ ) {
        string sideName = sideSets[side];
        
        vector<stk::mesh::Entity> sideEntities;
        mesh->getMySides(sideName, blocknames[b], sideEntities);
        vector<size_t>             local_side_Ids;
        vector<stk::mesh::Entity> side_output;
        vector<size_t>             local_elem_Ids;
        
        panzer_stk::workset_utils::getSideElements(*mesh, blocknames[b], sideEntities, local_side_Ids, side_output);
        
        int numSideElem = local_side_Ids.size();
        
        if (numSideElem > 0) {
          vector<size_t> unique_sides;
          unique_sides.push_back(local_side_Ids[0]);
          for (size_t e=0; e<numSideElem; e++) {
            bool found = false;
            for (size_t j=0; j<unique_sides.size(); j++) {
              if (unique_sides[j] == local_side_Ids[e]) {
                found = true;
              }
            }
            if (!found) {
              unique_sides.push_back(local_side_Ids[e]);
            }
          }
          
          for (size_t j=0; j<unique_sides.size(); j++) {
            vector<size_t> group;
            for (size_t e=0; e<numSideElem; e++) {
              if (local_side_Ids[e] == unique_sides[j]) {
                group.push_back(e);
              }
            }
            
            int prog = 0;
            while (prog < group.size()) {
              int currElem = numBoundaryElem;  // Avoid faults in last iteration
              if (prog+currElem > group.size()){
                currElem = group.size()-prog;
              }
              Kokkos::View<LO*,AssemblyDevice> eIndex("element indices",currElem);
              Kokkos::View<LO*,AssemblyDevice> sideIndex("local side indices",currElem);
              DRV currnodes("currnodes", currElem, numNodesPerElem, spaceDim);
              
              auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
              auto host_sideIndex = Kokkos::create_mirror_view(sideIndex); // mirror on host
              auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
              
              for (int e=0; e<currElem; e++) {
                host_eIndex(e) = mesh->elementLocalId(side_output[group[e+prog]]);
                host_sideIndex(e) = local_side_Ids[group[e+prog]];
                for (int n=0; n<host_currnodes.extent(1); n++) {
                  for (int m=0; m<host_currnodes.extent(2); m++) {
                    host_currnodes(e,n,m) = blocknodes(eIndex(e),n,m);
                  }
                }
              }
              Kokkos::deep_copy(currnodes,host_currnodes);
              Kokkos::deep_copy(eIndex,host_eIndex);
              Kokkos::deep_copy(sideIndex,host_sideIndex);
              
              // Build the Kokkos View of the cell GIDs ------
              vector<vector<GO> > cellGIDs;
              int numLocalDOF = 0;
              for (int i=0; i<currElem; i++) {
                vector<GO> GIDs;
                size_t elemID = host_eIndex(i);
                DOF->getElementGIDs(elemID, GIDs, blocknames[b]);
                cellGIDs.push_back(GIDs);
                numLocalDOF = GIDs.size(); // should be the same for all elements
              }
              Kokkos::View<GO**,HostDevice> hostGIDs("GIDs on host device",currElem,numLocalDOF);
              for (int i=0; i<currElem; i++) {
                for (int j=0; j<numLocalDOF; j++) {
                  hostGIDs(i,j) = cellGIDs[i][j];
                }
              }
              
              Kokkos::View<LO**,HostDevice> hostLIDs("LIDs on host device",currElem,numLocalDOF);
              for (int i=0; i<currElem; i++) {
                size_t elemID = host_eIndex(i);
                for (int j=0; j<numLocalDOF; j++) {
                  hostLIDs(i,j) = LIDs(elemID,j);
                }
              }
              
              //-----------------------------------------------
              // Set the side information (soon to be removed)-
              Kokkos::View<int****,HostDevice> sideinfo = phys->getSideInfo(b,host_eIndex);
              //-----------------------------------------------
              
              // Set the cell orientation ---
              
              Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices",
                                                                              currElem, numNodesPerElem);
              
              for (int i=0; i<currElem; i++) {
                vector<stk::mesh::EntityId> stk_nodeids;
                size_t elemID = host_eIndex(i);
                mesh->getNodeIdsForElement(all_meshElems[elemID], stk_nodeids);
                //mesh->getNodeIdsForElement(elemID, stk_nodeids);
                for (int n=0; n<numNodesPerElem; n++) {
                  currind(i,n) = stk_nodeids[n];
                }
              }
              //KokkosTools::print(currind);
              
              Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orient_drv("kv to orients",currElem);
              //Intrepid2::OrientationTools<AssemblyDevice>::getOrientation(orient_drv, cells[b][e]->nodeIndices, *(cells[b][e]->cellData->cellTopo));
              Intrepid2::OrientationTools<AssemblyDevice>::getOrientation(orient_drv, currind, *cellTopo);
              
              bcells.push_back(Teuchos::rcp(new BoundaryCell(cellData, currnodes, eIndex, sideIndex,
                                                             side, sideName, bcells.size(),
                                                             hostGIDs, hostLIDs,
                                                             sideinfo, orient_drv)));
              prog += currElem;
            }
          }
        }
      }
    }
    
    boundaryCells.push_back(bcells);
    
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::createCells" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

void AssemblyManager::createWorkset() {
  
  Teuchos::TimeMonitor localtimer(*wksettimer);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createWorkset ..." << endl;
    }
  }
  
  for (size_t b=0; b<cells.size(); b++) {
    wkset.push_back(Teuchos::rcp( new workset(cells[b][0]->getInfo(),
                                              isTransient,
                                              disc->ref_ip[b],
                                              disc->ref_wts[b], disc->ref_side_ip[b],
                                              disc->ref_side_wts[b],
                                              disc->basis_types[b],
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

void AssemblyManager::updateJacDBC(matrix_RCP & J,
                                   const vector<GO> & dofs, const bool & compute_disc_sens) {
  
  // given a "block" and the unknown field update jacobian to enforce Dirichlet BCs
  //size_t numcols = J->getGlobalNumCols();
  for( int i=0; i<dofs.size(); i++ ) { // for each node
    if (compute_disc_sens) {
      int numcols = globalParamUnknowns; // TMW fix this!
      for( int col=0; col<numcols; col++ ) {
        ScalarT m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        //J.ReplaceGlobalValues(row, 1, &m_val, &ind);
        J->replaceGlobalValues(col, 1, &m_val, &dofs[i]);
      }
    }
    else {
      GO numcols = J->getGlobalNumCols(); // TMW fix this!
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

void AssemblyManager::updateJacDBC(matrix_RCP & J,
                                   const vector<LO> & dofs, const bool & compute_disc_sens) {
  
  if (compute_disc_sens) {
    // nothing to do here
  }
  else {
    for( int i=0; i<dofs.size(); i++ ) {
      ScalarT val = 1.0; // set diagonal entry to 1
      J->replaceLocalValues(dofs[i], 1, &val, &dofs[i]);
    }
  }
}

// ========================================================================================
// ========================================================================================

void AssemblyManager::setInitial(vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                                 const bool & lumpmass) {
  
  Teuchos::TimeMonitor localtimer(*setinittimer);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::setInitial ..." << endl;
    }
  }
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      
      int numElem = cells[b][e]->numElem;
      Kokkos::View<GO**,HostDevice> GIDs = cells[b][e]->GIDs;
      
      Kokkos::View<ScalarT**,AssemblyDevice> localrhs = cells[b][e]->getInitial(true, useadjoint);
      Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[b][e]->getMass();
      auto host_rhs = Kokkos::create_mirror_view(localrhs);
      auto host_mass = Kokkos::create_mirror_view(localmass);
      Kokkos::deep_copy(host_rhs,localrhs);
      Kokkos::deep_copy(host_mass,localmass);
      
      // assemble into global matrix
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<GIDs.extent(1); row++ ) {
          GO rowIndex = GIDs(c,row);
          ScalarT val = host_rhs(c,row);
          rhs->sumIntoGlobalValue(rowIndex,0, val);
          if (lumpmass) {
            Teuchos::Array<ScalarT> vals(1);
            Teuchos::Array<GO> cols(1);
            
            ScalarT totalval = 0.0;
            for( size_t col=0; col<GIDs.extent(1); col++ ) {
              cols[0] = GIDs(c,col);
              totalval += host_mass(c,row,col);
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
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::setInitial ..." << endl;
    }
  }
  
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
// ========================================================================================

void AssemblyManager::setDirichlet(vector_RCP & rhs, matrix_RCP & mass,
                                   const bool & useadjoint,
                                   const ScalarT & time,
                                   const bool & lumpmass) {
  
  Teuchos::TimeMonitor localtimer(*setdbctimer);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::setDirichlet ..." << endl;
    }
  }
  
  auto localMatrix = mass->getLocalMatrix();
  
  for (size_t b=0; b<boundaryCells.size(); b++) {
    wkset[b]->time = time;
    wkset[b]->time_KV(0) = time;
    for (size_t e=0; e<boundaryCells[b].size(); e++) {
      
      int numElem = boundaryCells[b][e]->numElem;
      Kokkos::View<LO**,HostDevice> LIDs = boundaryCells[b][e]->LIDs;
      
      Kokkos::View<ScalarT**,AssemblyDevice> localrhs = boundaryCells[b][e]->getDirichlet();
      Kokkos::View<ScalarT***,AssemblyDevice> localmass = boundaryCells[b][e]->getMass();
      auto host_rhs = Kokkos::create_mirror_view(localrhs);
      auto host_mass = Kokkos::create_mirror_view(localmass);
      Kokkos::deep_copy(host_rhs,localrhs);
      Kokkos::deep_copy(host_mass,localmass);
      
      const int numVals = static_cast<int>(LIDs.extent(1));
      
      // assemble into global matrix
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(c,row);
          if (isFixedDOF(rowIndex)) {
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
              //mass->sumIntoGlobalValues(rowIndex, cols, vals);
              localMatrix.sumIntoValues(rowIndex, cols, 1, vals, true, false);
            }
            else {
              LO cols[numVals];
              ScalarT vals[numVals];
              for( size_t col=0; col<LIDs.extent(1); col++ ) {
                cols[col] = LIDs(c,col);
                vals[col] = localmass(c,row,col);
              }
              localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, true, false);
              
            }
          }
        }
      }
    }
  }
  
  // Loop over the cells to put ones on the diagonal for DOFs not on Dirichlet boundaries
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      Kokkos::View<LO**,HostDevice> LIDs = cells[b][e]->LIDs;
      for (int c=0; c<cells[b][e]->numElem; c++) {
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(c,row);
          if (!isFixedDOF(rowIndex)) {
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
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::setDirichlet ..." << endl;
    }
  }
  
}


// ========================================================================================
// Wrapper to the main assembly routine to assemble over all blocks (most common use case)
// ========================================================================================

void AssemblyManager::assembleJacRes(vector_RCP & u, vector_RCP & phi,
                                     const bool & compute_jacobian, const bool & compute_sens,
                                     const bool & compute_disc_sens,
                                     vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                     const ScalarT & current_time,
                                     const bool & useadjoint, const bool & store_adjPrev,
                                     const int & num_active_params,
                                     vector_RCP & Psol, const bool & is_final_time,
                                     const ScalarT & deltat) {
  
  if (milo_debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting AssemblyManager::assembleJacRes ..." << endl;
    }
  }
  
  for (size_t b=0; b<cells.size(); b++) {
    this->assembleJacRes(u, phi, compute_jacobian,
                         compute_sens, compute_disc_sens, res, J, isTransient,
                         current_time, useadjoint, store_adjPrev, num_active_params,
                         Psol, is_final_time, b, deltat);
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

void AssemblyManager::assembleJacRes(vector_RCP & u, vector_RCP & phi,
                                     const bool & compute_jacobian, const bool & compute_sens,
                                     const bool & compute_disc_sens,
                                     vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                     const ScalarT & current_time,
                                     const bool & useadjoint, const bool & store_adjPrev,
                                     const int & num_active_params,
                                     vector_RCP & Psol, const bool & is_final_time,
                                     const int & b, const ScalarT & deltat) {
    
  
  
  int numRes = res->getNumVectors();
  
  Teuchos::TimeMonitor localassemblytimer(*assemblytimer);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Set up the worksets and allocate the local residual and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////
  
  // TMW: this won't really work on GPU, need to fix once working
  if (isTransient) {
    ScalarT timeval = current_time + wkset[b]->butcher_c(wkset[b]->current_stage)*deltat;
    
    wkset[b]->time = timeval;
    wkset[b]->time_KV(0) = timeval;
  }
  wkset[b]->isTransient = isTransient;
  wkset[b]->isAdjoint = useadjoint;
  wkset[b]->deltat = deltat;
  wkset[b]->deltat_KV(0) = deltat;
  wkset[b]->alpha = 1.0/deltat;
  
  int numElem = cells[b][0]->numElem;
  int numDOF = cells[b][0]->GIDs.extent(1);
  
  int numParamDOF = 0;
  if (compute_disc_sens) {
    numParamDOF = cells[b][0]->paramGIDs.extent(1); // is this on host
  }
  
  // This data needs to be available on Host and Device
  // Optimizing layout for AssemblyExec
  Kokkos::View<ScalarT***,UnifiedDevice> local_res, local_J;
  
  if (compute_sens) {
    local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,numDOF,num_active_params);
  }
  else {
    local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,numDOF,1);
  }
  
  if (compute_disc_sens) {
    local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,numDOF,numParamDOF);
  }
  else { // note that this does increase memory as numElem increases
    local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,numDOF,numDOF);
  }
  
  //Kokkos::View<ScalarT**,AssemblyDevice> aPrev;
  
  /////////////////////////////////////////////////////////////////////////////
  // Perform gather to cells
  /////////////////////////////////////////////////////////////////////////////
  
  {
    Teuchos::TimeMonitor localtimer(*gathertimer);
    
    // Local gather of solutions
    this->performGather(b,u,0,0);
    this->performGather(b,Psol,4,0);
    if (useadjoint) {
      this->performGather(b,phi,2,0);
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
      Kokkos::deep_copy(local_res,0.0);
      Kokkos::deep_copy(local_J,0.0);
      
      cells[b][e]->computeJacRes(current_time, isTransient, useadjoint, compute_jacobian, compute_sens,
                                 num_active_params, compute_disc_sens, false, store_adjPrev,
                                 local_res, local_J, assemble_volume_terms[b], assemble_face_terms[b]);
      
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
    
    this->insert(J, res, local_res, local_J,
                 cells[b][e]->GIDs, cells[b][e]->LIDs,
                 cells[b][e]->paramGIDs,
                 compute_jacobian, compute_disc_sens);
    
    
  } // element loop
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Boundary terms
  //////////////////////////////////////////////////////////////////////////////////////
  
  
  if (!cells[0][0]->cellData->multiscale && assemble_boundary_terms[b]) {
    
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
    
    if (compute_sens) {
      local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,numDOF,num_active_params);
    }
    else {
      local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,numDOF,1);
    }
    
    if (compute_disc_sens) {
      local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,numDOF,numParamDOF);
    }
    else {
      local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,numDOF,numDOF);
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
        
        /////////////////////////////////////////////////////////////////////////////
        // Compute the local residual and Jacobian on this cell
        /////////////////////////////////////////////////////////////////////////////
        
        {
          Teuchos::TimeMonitor localtimer(*phystimer);
          
          Kokkos::deep_copy(local_res,0.0);
          Kokkos::deep_copy(local_J,0.0);
          
          boundaryCells[b][e]->computeJacRes(current_time, isTransient, useadjoint, compute_jacobian, compute_sens,
                                             num_active_params, compute_disc_sens, false, store_adjPrev,
                                             local_res, local_J);
          
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
        
        this->insert(J, res, local_res, local_J,
                     boundaryCells[b][e]->GIDs,
                     boundaryCells[b][e]->LIDs,
                     boundaryCells[b][e]->paramGIDs,
                     compute_jacobian, compute_disc_sens);
        
        
      }
    } // element loop
  }

  // Apply constraints, e.g., strongly imposed Dirichlet
  this->dofConstraints(J, res, current_time, compute_jacobian, compute_disc_sens);
  
}


// ========================================================================================
// Enforce DOF constraints - includes strong Dirichlet
// ========================================================================================

void AssemblyManager::dofConstraints(matrix_RCP & J, vector_RCP & res,
                                     const ScalarT & current_time,
                                     const bool & compute_jacobian,
                                     const bool & compute_disc_sens) {
  
  Teuchos::TimeMonitor localtimer(*dbctimer);
  
  if (usestrongDBCs) {
    vector<vector<vector<LO> > > dbcDOFs = phys->dbc_dofs;
    for (size_t block=0; block<dbcDOFs.size(); block++) {
      for (size_t var=0; var<dbcDOFs[block].size(); var++) {
        if (compute_jacobian) {
          this->updateJacDBC(J,dbcDOFs[block][var],compute_disc_sens);
        }
        //this->updateResDBC(res,dbcDOFs[block][var]);
      }
    }
  }
  
  vector<vector<GO> > fixedDOFs = phys->point_dofs;
  for (size_t block=0; block<fixedDOFs.size(); block++) {
    if (compute_jacobian) {
      this->updateJacDBC(J,fixedDOFs[block],compute_disc_sens);
    }
    //this->updateResDBC(res,fixedDOFs[block]);
  }
  
}


// ========================================================================================
//
// ========================================================================================

void AssemblyManager::resetPrevSoln() {
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->resetPrevSoln();
    }
  }
}

void AssemblyManager::resetStageSoln() {
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->resetStageSoln();
    }
  }
}

void AssemblyManager::updateStageNumber(const int & stage) {
  for (size_t b=0; b<wkset.size(); b++) {
    wkset[b]->current_stage = stage;
    wkset[b]->current_stage_KV(0) = stage;
  }
}

void AssemblyManager::updateStageSoln()  {
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->updateStageSoln();
    }
  }
}

// ========================================================================================
//
// ========================================================================================

void AssemblyManager::performGather(const size_t & b, const vector_RCP & vec,
                                    const int & type, const size_t & entry) {
  
  // Get a view of the vector on the AssemblyDevice
  // TMW: this manual deep copy is a hack.  There should be a better way to do this.

  auto vec_kv = vec->getLocalView<HostDevice>();
  Kokkos::View<ScalarT**,AssemblyDevice> vec_dev("tpetra vector on device",vec_kv.extent(0),vec_kv.extent(1));
  auto vec_host = Kokkos::create_mirror_view(vec_dev);
  for (int i=0; i<vec_kv.extent(0); i++) {
    for (int j=0; j<vec_kv.extent(1); j++) {
      vec_host(i,j) = vec_kv(i,j);
    }
  }
  Kokkos::deep_copy(vec_dev,vec_host);
  
  Kokkos::View<LO***,AssemblyDevice> index;
  Kokkos::View<LO*,UnifiedDevice> numDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> data;
  
  // TMW: Does this need to be executed on Device?
  for (size_t c=0; c < cells[b].size(); c++) {
    switch(type) {
      case 0 :
        index = cells[b][c]->index;
        numDOF = cells[b][c]->cellData->numDOF;
        data = cells[b][c]->u;
        break;
      case 1 : // deprecated (u_dot)
        break;
      case 2 :
        index = cells[b][c]->index;
        numDOF = cells[b][c]->cellData->numDOF;
        data = cells[b][c]->phi;
        break;
      case 3 : // deprecated (phi_dot)
        break;
      case 4:
        index = cells[b][c]->paramindex;
        numDOF = cells[b][c]->cellData->numParamDOF;
        data = cells[b][c]->param;
        break;
      case 5 :
        index = cells[b][c]->auxindex;
        numDOF = cells[b][c]->cellData->numAuxDOF;
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
    //Kokkos::View<ScalarT***,HostDevice>::HostMirror host_data = Kokkos::create_mirror_view(data);
    parallel_for(RangePolicy<AssemblyExec>(0,index.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t n=0; n<index.extent(1); n++) {
        for(size_t i=0; i<numDOF(n); i++ ) {
          //data(e,n,i) = vec_kv(index(e,n,i),entry);
          data(e,n,i) = vec_dev(index(e,n,i),entry);
        }
      }
    });
    //Kokkos::deep_copy(data,host_data);
  }
}

// ========================================================================================
//
// ========================================================================================

void AssemblyManager::performBoundaryGather(const size_t & b, const vector_RCP & vec,
                                            const int & type, const size_t & entry) {
  
  if (boundaryCells.size() > b) {
    
    // Get a view of the vector on the HostDevice
    auto vec_kv = vec->getLocalView<HostDevice>();
    // TMW: need to move this to the device

    // Get a corresponding view on the AssemblyDevice
    
    Kokkos::View<LO***,AssemblyDevice> index;
    Kokkos::View<LO*,UnifiedDevice> numDOF;
    Kokkos::View<ScalarT***,AssemblyDevice> data;
    
    for (size_t c=0; c < boundaryCells[b].size(); c++) {
      if (boundaryCells[b][c]->numElem > 0) {
        
        switch(type) {
          case 0 :
            index = boundaryCells[b][c]->index;
            numDOF = boundaryCells[b][c]->cellData->numDOF;
            data = boundaryCells[b][c]->u;
            break;
          case 1 : // deprecated (u_dot)
            break;
          case 2 :
            index = boundaryCells[b][c]->index;
            numDOF = boundaryCells[b][c]->cellData->numDOF;
            data = boundaryCells[b][c]->phi;
            break;
          case 3 : // deprecated (phi_dot)
            break;
          case 4:
            index = boundaryCells[b][c]->paramindex;
            numDOF = boundaryCells[b][c]->cellData->numParamDOF;
            data = boundaryCells[b][c]->param;
            break;
          case 5 :
            index = boundaryCells[b][c]->auxindex;
            numDOF = boundaryCells[b][c]->cellData->numAuxDOF;
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
        
        //Kokkos::View<ScalarT***,HostDevice>::HostMirror host_data = Kokkos::create_mirror_view(data);
        parallel_for(RangePolicy<AssemblyExec>(0,index.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_t n=0; n<index.extent(1); n++) {
            for(size_t i=0; i<numDOF(n); i++ ) {
              data(e,n,i) = vec_kv(index(e,n,i),entry);
            }
          }
        });
        //Kokkos::deep_copy(data,host_data);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void AssemblyManager::insert(matrix_RCP & J, vector_RCP & res,
                             Kokkos::View<ScalarT***,UnifiedDevice> local_res,
                             Kokkos::View<ScalarT***,UnifiedDevice> local_J,
                             Kokkos::View<GO**,HostDevice> GIDs,
                             Kokkos::View<LO**,HostDevice> LIDs,
                             Kokkos::View<GO**,HostDevice> paramGIDs,
                             const bool & compute_jacobian,
                             const bool & compute_disc_sens) {

  Teuchos::TimeMonitor localtimer(*inserttimer);
  
  bool useLocal = true;
  if (compute_jacobian && compute_disc_sens) { // fix later
    useLocal = false;
  }
  
  if (useLocal) {
    /////////////////////////////////////
    // Using LIDs
    /////////////////////////////////////
    
    auto localMatrix = J->getLocalMatrix();
    
    const int numVals = static_cast<int>(LIDs.extent(1));
    
    LO cols[numVals];
    ScalarT vals[numVals];
    
    for (int i=0; i<LIDs.extent(0); i++) {
      for( size_t row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(i,row);
        // add check here for fixedDOF
        if (!isFixedDOF(rowIndex)) {
          for (LO g=0; g<local_res.extent(2); g++) {
            ScalarT val = local_res(i,row,g);
            res->sumIntoLocalValue(rowIndex,g, val);
          }
        }
      }
    }
    if (compute_jacobian) {
      for (int i=0; i<LIDs.extent(0); i++) { // this should be changed to a Kokkos::parallel_for on host
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(i,row);
          // add check here for fixedDOF
          if (!isFixedDOF(rowIndex)) {
            for( size_t col=0; col<LIDs.extent(1); col++ ) {
              vals[col] = local_J(i,row,col);
              cols[col] = LIDs(i,col);
            }
            localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, true, false); // isSorted, useAtomics
            // the LIDs are actually not sorted, but this appears to run a little faster
          }
        }
      }
    }
    
  }
  else {
    /////////////////////////////////////
    // Using GIDs
    /////////////////////////////////////
    
    Teuchos::Array<ScalarT> vals(GIDs.extent(1));
    Teuchos::Array<GO> cols(GIDs.extent(1));
    
    for (int i=0; i<GIDs.extent(0); i++) {
      
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
              ScalarT val = local_J(i,row,col);
              J->insertGlobalValues(colIndex, 1, &val, &rowIndex);
            }
          }
          else {
            for( size_t col=0; col<GIDs.extent(1); col++ ) {
              vals[col] = local_J(i,row,col);
              cols[col] = GIDs(i,col);
            }
            J->sumIntoGlobalValues(rowIndex, cols, vals);
          }
        }
      }
    }
  }
}

