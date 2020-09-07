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
  
  assembly_partitioning = settings->sublist("Solver").get<string>("assembly partitioning","sequential"); // "neighbor-avoiding"
  use_atomics = settings->sublist("Solver").get<bool>("use atomics",false); // not need if assembly partitioning is done correctly
  
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
  
  
  // Create cells/boundary cells
  this->createCells();
  
  params->setupDiscretizedParameters(cells, boundaryCells);
  
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
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished assembly manager constructor" << endl;
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
  
  vector<stk::mesh::Entity> all_meshElems;
  mesh->getMyElements(all_meshElems);
  
  // May need to be PHX::Device
  Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device> LIDs = DOF->getLIDs();
  
  for (size_t b=0; b<blocknames.size(); b++) {
    vector<Teuchos::RCP<cell> > blockcells;
    vector<stk::mesh::Entity> stk_meshElems;
    mesh->getMyElements(blocknames[b], stk_meshElems);
    
    topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
    int numNodesPerElem = cellTopo->getNodeCount();
    int spaceDim = phys->spaceDim;
    size_t numTotalElem = stk_meshElems.size();
    vector<size_t> localIds;
    
    Kokkos::DynRankView<ScalarT,AssemblyDevice> blocknodes("nodes on block",numTotalElem,numNodesPerElem,spaceDim);
    auto host_blocknodes = Kokkos::create_mirror_view(blocknodes);
    panzer_stk::workset_utils::getIdsAndVertices(*mesh, blocknames[b], localIds, host_blocknodes); // fill on host
    Kokkos::deep_copy(blocknodes, host_blocknodes);
    
    vector<size_t> myElem = disc->myElements[b];
    Kokkos::View<size_t*,AssemblyDevice> eIDs("local element IDs on device",myElem.size());
    auto host_eIDs = Kokkos::create_mirror_view(eIDs);
    for (size_t elem=0; elem<myElem.size(); elem++) {
      host_eIDs(elem) = myElem[elem];
    }
    Kokkos::deep_copy(eIDs, host_eIDs);
    
    DRV refnodes("nodes on reference element",numNodesPerElem,spaceDim);
    CellTools::getReferenceSubcellVertices(refnodes, spaceDim, 0, *cellTopo);
    
    int elemPerCell = settings->sublist("Solver").get<int>("workset size",1);
    int prog = 0;
    
    vector<string> sideSets;
    mesh->getSidesetNames(sideSets);
    Teuchos::RCP<CellMetaData> blockCellData = Teuchos::rcp( new CellMetaData(settings, cellTopo,
                                                                              phys, b, 0,
                                                                              build_face_terms[b],
                                                                              assemble_face_terms[b],
                                                                              sideSets, disc->ref_ip[b],
                                                                              disc->ref_wts[b], disc->ref_side_ip[b],
                                                                              disc->ref_side_wts[b], disc->basis_types[b],
                                                                              disc->basis_pointers[b],
                                                                              params->num_discretized_params,
                                                                              refnodes));
    
    
    blockCellData->requireBasisAtNodes = settings->sublist("Postprocess").get<bool>("plot solution at nodes",false);
    
    vector<vector<int> > curroffsets = phys->offsets[b];
    Kokkos::View<LO*,AssemblyDevice> numDOF_KV("number of DOF per variable",curroffsets.size());
    for (int k=0; k<curroffsets.size(); k++) {
      numDOF_KV(k) = curroffsets[k].size();
    }
    blockCellData->numDOF = numDOF_KV;
    Kokkos::View<LO*,HostDevice> numDOF_host("numDOF on host",curroffsets.size());// = Kokkos::create_mirror_view(numDOF_KV);
    Kokkos::deep_copy(numDOF_host, numDOF_KV);
    blockCellData->numDOF_host = numDOF_host;
    
    cellData.push_back(blockCellData);
    
    TEUCHOS_TEST_FOR_EXCEPTION(LIDs.extent(1) > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of degrees of freedom per element times the number of time stages.");
    
    if (assembly_partitioning == "sequential") {
      while (prog < numTotalElem) {
        int currElem = elemPerCell;  // Avoid faults in last iteration
        if (prog+currElem > numTotalElem){
          currElem = numTotalElem-prog;
        }
        
        Kokkos::View<LO*,AssemblyDevice> eIndex("element indices",currElem);
        DRV currnodes("currnodes", currElem, numNodesPerElem, spaceDim);
        LIDView cellLIDs("LIDs on device",currElem,LIDs.extent(1));
        
        auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
        Kokkos::View<LO*,HostDevice> host_eIndex2("element indices on host",currElem);
        //auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
        
        auto nodes_sub = Kokkos::subview(blocknodes,std::make_pair(prog, prog+currElem), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::deep_copy(currnodes,nodes_sub);
        
        for (size_t e=0; e<host_eIndex.extent(0); e++) {
          //for (int n=0; n<currnodes.extent(1); n++) {
          //  for (int m=0; m<currnodes.extent(2); m++) {
          //    currnodes(e,n,m) = blocknodes(prog+e,n,m);
          //  }
          //}
          host_eIndex(e) = prog+e;//disc->myElements[b][prog+e]; // TMW: why here?;prog+e;
        }
        //Kokkos::deep_copy(currnodes,host_currnodes);
        Kokkos::deep_copy(eIndex,host_eIndex);
        Kokkos::deep_copy(host_eIndex2,host_eIndex);
        // This subview only works if the cells use a continuous ordering of elements
        // Considering generalizing this to reduce atomic overhead, so performing manual deep copy for now
        //LIDView cellLIDs = Kokkos::subview(LIDs, std::make_pair(prog,prog+currElem), Kokkos::ALL());
        int progend = prog+cellLIDs.extent(0);
        auto celem = Kokkos::subview(eIDs, std::make_pair(prog,progend));
        parallel_for("assembly copy LIDs",RangePolicy<AssemblyExec>(0,cellLIDs.extent(0)), KOKKOS_LAMBDA (const int e ) {
          size_t elemID = celem(e);//disc->myElements[b][prog+e]; // TMW: why here?
          for (int j=0; j<LIDs.extent(1); j++) {
            cellLIDs(e,j) = LIDs(elemID,j);
          }
        });
        
        // Set the side information (soon to be removed)-
        Kokkos::View<int****,HostDevice> sideinfo = phys->getSideInfo(b,host_eIndex2);
        
        Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices", currElem, numNodesPerElem);
        auto host_currind = Kokkos::create_mirror_view(currind);
        
        for (int i=0; i<currElem; i++) {
          vector<stk::mesh::EntityId> stk_nodeids;
          size_t elemID = prog+i;//host_eIndex(i);
          mesh->getNodeIdsForElement(stk_meshElems[elemID], stk_nodeids);
          for (int n=0; n<numNodesPerElem; n++) {
            host_currind(i,n) = stk_nodeids[n];
          }
        }
        Kokkos::deep_copy(currind, host_currind);
        
        Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orient_drv("kv to orients",currElem);
        OrientTools::getOrientation(orient_drv, currind, *cellTopo);
        
        blockcells.push_back(Teuchos::rcp(new cell(blockCellData, currnodes, eIndex,
                                                   cellLIDs, sideinfo, orient_drv)));
        prog += elemPerCell;
      }
    }
    else if (assembly_partitioning == "random") { // not implemented yet
      
    }
    else if (assembly_partitioning == "neighbor-avoiding") { // not implemented yet
     // need neighbor information
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
              Kokkos::View<LO*,HostDevice> host_eIndex2("element indices",currElem);
              auto host_sideIndex = Kokkos::create_mirror_view(sideIndex); // mirror on host
              auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
              
              for (int e=0; e<currElem; e++) {
                host_eIndex(e) = mesh->elementLocalId(side_output[group[e+prog]]);
                host_sideIndex(e) = local_side_Ids[group[e+prog]];
                for (int n=0; n<host_currnodes.extent(1); n++) {
                  for (int m=0; m<host_currnodes.extent(2); m++) {
                    host_currnodes(e,n,m) = blocknodes(host_eIndex(e),n,m);
                  }
                }
              }
              Kokkos::deep_copy(currnodes,host_currnodes);
              Kokkos::deep_copy(eIndex,host_eIndex);
              Kokkos::deep_copy(host_eIndex2,host_eIndex);
              Kokkos::deep_copy(sideIndex,host_sideIndex);
              
              // Build the Kokkos View of the cell GIDs ------
              
              LIDView cellLIDs("LIDs",currElem,LIDs.extent(1));
              parallel_for("assembly copy LIDs bcell",RangePolicy<AssemblyExec>(0,cellLIDs.extent(0)), KOKKOS_LAMBDA (const int e ) {
                size_t elemID = eIndex(e);
                for (int j=0; j<LIDs.extent(1); j++) {
                  cellLIDs(e,j) = LIDs(elemID,j);
                }
              });
              
              //-----------------------------------------------
              // Set the side information (soon to be removed)-
              Kokkos::View<int****,HostDevice> sideinfo = phys->getSideInfo(b,host_eIndex2);
              //-----------------------------------------------
              
              // Set the cell orientation ---
              
              Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices",
                                                                              currElem, numNodesPerElem);
              
              auto host_currind = Kokkos::create_mirror_view(currind);
              
              for (int i=0; i<currElem; i++) {
                vector<stk::mesh::EntityId> stk_nodeids;
                size_t elemID = host_eIndex(i);
                mesh->getNodeIdsForElement(all_meshElems[elemID], stk_nodeids);
                for (int n=0; n<numNodesPerElem; n++) {
                  host_currind(i,n) = stk_nodeids[n];
                }
              }
              Kokkos::deep_copy(currind, host_currind);
                            
              Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orient_drv("kv to orients",currElem);
              OrientTools::getOrientation(orient_drv, currind, *cellTopo);
              
              bcells.push_back(Teuchos::rcp(new BoundaryCell(blockCellData, currnodes, eIndex, sideIndex,
                                                             side, sideName, bcells.size(),
                                                             cellLIDs, sideinfo, orient_drv)));
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
  
  // TMW: ToDo - should add a lumped mass option
  
  Teuchos::TimeMonitor localtimer(*setinittimer);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::setInitial ..." << endl;
    }
  }
  
  auto localMatrix = mass->getLocalMatrix();
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      
      int numElem = cells[b][e]->numElem;
      LIDView_host LIDs = cells[b][e]->LIDs_host;
      
      Kokkos::View<ScalarT**,AssemblyDevice> localrhs = cells[b][e]->getInitial(true, useadjoint);
      Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[b][e]->getMass();
      auto host_rhs = Kokkos::create_mirror_view(localrhs);
      auto host_mass = Kokkos::create_mirror_view(localmass);
      Kokkos::deep_copy(host_rhs,localrhs);
      Kokkos::deep_copy(host_mass,localmass);
      
      // Would prefer to rewrite this
      //parallel_for("assembly copy LIDs",RangePolicy<HostExec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int e=0; e<LIDs.extent(0); e++) {
        //const int numVals = static_cast<int>(LIDs.extent(1));
        //const int numVals = LIDs.extent(1);
        //int numVals = LIDs.extent(1);
        //LO cols[numVals];
        //ScalarT vals[numVals];
        
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(e,row);
          ScalarT val = host_rhs(e,row);
          rhs->sumIntoLocalValue(rowIndex, 0, val);
          for( size_t col=0; col<LIDs.extent(1); col++ ) {
            ScalarT vals = host_mass(e,row,col);
            LO cols = LIDs(e,col);
            localMatrix.sumIntoValues(rowIndex, &cols, 1, &vals, true, false); // isSorted, useAtomics
            // the LIDs are actually not sorted, but this appears to run a little faster
          }
          
        }
      }
      //});
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
      LIDView_host LIDs = cells[b][e]->LIDs_host;
      Kokkos::View<ScalarT**,AssemblyDevice> localinit = cells[b][e]->getInitial(false, useadjoint);
      auto host_init = Kokkos::create_mirror_view(localinit);
      Kokkos::deep_copy(host_init,localinit);
      int numElem = cells[b][e]->numElem;
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(c,row);
          ScalarT val = host_init(c,row);
          initial->replaceLocalValue(rowIndex,0, val);
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
    wkset[b]->setTime(time);
    for (size_t e=0; e<boundaryCells[b].size(); e++) {
      
      int numElem = boundaryCells[b][e]->numElem;
      LIDView LIDs = boundaryCells[b][e]->LIDs;
      
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
      LIDView LIDs = cells[b][e]->LIDs;
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
  
  {
    Teuchos::TimeMonitor localtimer(*gathertimer);
    
    // Local gather of solutions
    this->performGather(u,0,0);
    if (params->num_discretized_params > 0) {
      this->performGather(Psol,4,0);
    }
    if (useadjoint) {
      this->performGather(phi,2,0);
    }
  }
  
  for (size_t b=0; b<cells.size(); b++) {
    this->assembleJacRes(compute_jacobian,
                         compute_sens, compute_disc_sens, res, J, isTransient,
                         current_time, useadjoint, store_adjPrev, num_active_params,
                         is_final_time, b, deltat);
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

void AssemblyManager::assembleJacRes(const bool & compute_jacobian, const bool & compute_sens,
                                     const bool & compute_disc_sens,
                                     vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                     const ScalarT & current_time,
                                     const bool & useadjoint, const bool & store_adjPrev,
                                     const int & num_active_params,
                                     const bool & is_final_time,
                                     const int & b, const ScalarT & deltat) {
    
  
  
  int numRes = res->getNumVectors();
  
  Teuchos::TimeMonitor localassemblytimer(*assemblytimer);
  
  auto J_view = J->getLocalMatrix();
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Set up the worksets and allocate the local residual and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////
  
  if (isTransient) {
    ScalarT timeval = current_time + wkset[b]->butcher_c(wkset[b]->current_stage)*deltat;
    
    wkset[b]->setTime(timeval);
    wkset[b]->setDeltat(deltat);
    wkset[b]->alpha = 1.0/deltat;
  }
  wkset[b]->isTransient = isTransient;
  wkset[b]->isAdjoint = useadjoint;
  
  int numElem = cells[b][0]->numElem;
  int numDOF = cells[b][0]->LIDs.extent(1);
  
  int numParamDOF = 0;
  if (compute_disc_sens) {
    numParamDOF = cells[b][0]->paramLIDs.extent(1); // is this on host
  }
  
  // This data needs to be available on Host and Device
  // Optimizing layout for AssemblyExec
  Kokkos::View<ScalarT***,AssemblyDevice> local_res, local_J;
  
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
  
  Kokkos::View<ScalarT***,HostDevice> local_res_host("local residual on host",numElem,numDOF,local_res.extent(2));// = create_mirror_view(local_res);
  Kokkos::View<ScalarT***,HostDevice> local_J_host("local J on host",numElem,numDOF,local_J.extent(2));// = create_mirror_view(local_J);
  
  //Kokkos::View<ScalarT**,AssemblyDevice> aPrev;
  
  /////////////////////////////////////////////////////////////////////////////
  // Perform gather to cells
  /////////////////////////////////////////////////////////////////////////////
  /*
  Kokkos::View<ScalarT*,AssemblyDevice> u_dev, phi_dev, P_dev;
  {
    Teuchos::TimeMonitor localtimer(*gathertimer);
    
    // Local gather of solutions
    auto u_kv = u->getLocalView<HostDevice>();
    auto u_slice = Kokkos::subview(u_kv, Kokkos::ALL(), 0);
    u_dev = Kokkos::View<ScalarT*,AssemblyDevice>("tpetra vector on device",u_kv.extent(0));
    auto u_host = Kokkos::create_mirror_view(u_dev);
    Kokkos::deep_copy(u_host,u_slice);
    Kokkos::deep_copy(u_dev,u_host);
    this->performGather(b,u_dev,0,0);
    
    if (params->num_discretized_params > 0) {
      auto P_kv = Psol->getLocalView<HostDevice>();
      auto P_slice = Kokkos::subview(P_kv, Kokkos::ALL(), 0);
      P_dev = Kokkos::View<ScalarT*,AssemblyDevice>("tpetra vector on device",P_kv.extent(0));
      auto P_host = Kokkos::create_mirror_view(P_dev);
      Kokkos::deep_copy(P_host,P_slice);
      Kokkos::deep_copy(P_dev,P_host);
      this->performGather(b,P_dev,4,0);
    }
    if (useadjoint) {
      auto phi_kv = phi->getLocalView<HostDevice>();
      auto phi_slice = Kokkos::subview(phi_kv, Kokkos::ALL(), 0);
      phi_dev = Kokkos::View<ScalarT*,AssemblyDevice>("tpetra vector on device",phi_kv.extent(0));
      auto phi_host = Kokkos::create_mirror_view(phi_dev);
      Kokkos::deep_copy(phi_host,phi_slice);
      Kokkos::deep_copy(phi_dev,phi_host);
      this->performGather(b,phi,2,0);
    }
  }
  Kokkos::fence(); 
 */
  
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
 
    Kokkos::fence();

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
    
    Kokkos::deep_copy(local_res_host,local_res);
    Kokkos::deep_copy(local_J_host,local_J);
    
    ///////////////////////////////////////////////////////////////////////////
    // Insert into global matrix/vector
    ///////////////////////////////////////////////////////////////////////////
    this->insert(J_view, res, local_res_host, local_J_host,
                 cells[b][e]->LIDs_host,
                 cells[b][e]->paramLIDs_host,
                 compute_jacobian, compute_disc_sens);
    
  } // element loop
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Boundary terms
  //////////////////////////////////////////////////////////////////////////////////////
  
  if (!cells[0][0]->cellData->multiscale && assemble_boundary_terms[b]) {
    /*
    {
      Teuchos::TimeMonitor localtimer(*gathertimer);
      
      // Local gather of solutions
      this->performBoundaryGather(b,u_dev,0,0);
      if (params->num_discretized_params > 0) {
        this->performBoundaryGather(b,P_dev,4,0);
      }
      if (useadjoint) {
        this->performBoundaryGather(b,phi_dev,2,0);
      }
    }
    Kokkos::fence();   
    */
    
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
    
    Kokkos::View<ScalarT***,HostDevice> local_res_host("local residual on host",numElem,numDOF,local_res.extent(2));// = create_mirror_view(local_res);
    Kokkos::View<ScalarT***,HostDevice> local_J_host("local J on host",numElem,numDOF,local_J.extent(2));// = create_mirror_view(local_J);
    
    for (size_t e=0; e < boundaryCells[b].size(); e++) {
      
      if (boundaryCells[b][e]->numElem > 0) {
        wkset[b]->localEID = e;
        
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
        
        Kokkos::deep_copy(local_res_host,local_res);
        Kokkos::deep_copy(local_J_host,local_J);
        
        ///////////////////////////////////////////////////////////////////////////
        // Insert into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        this->insert(J_view, res, local_res_host, local_J_host,
                     boundaryCells[b][e]->LIDs_host,
                     boundaryCells[b][e]->paramLIDs_host,
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
      }
    }
  }
  
  vector<vector<GO> > fixedDOFs = phys->point_dofs;
  for (size_t block=0; block<fixedDOFs.size(); block++) {
    if (compute_jacobian) {
      this->updateJacDBC(J,fixedDOFs[block],compute_disc_sens);
    }
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
    wkset[b]->setStage(stage);
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

void AssemblyManager::performGather(const vector_RCP & vec, const int & type, const size_t & entry) {
  
  auto vec_kv = vec->getLocalView<HostDevice>();
  auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), entry);
  Kokkos::View<ScalarT*,AssemblyDevice> vec_dev("tpetra vector on device",vec_kv.extent(0));
  auto vec_host = Kokkos::create_mirror_view(vec_dev);
  Kokkos::deep_copy(vec_host,vec_slice);
  Kokkos::deep_copy(vec_dev,vec_host);
  this->performGather(vec_dev, type);
  this->performBoundaryGather(vec_dev, type);
  
}

// ========================================================================================
//
// ========================================================================================

void AssemblyManager::performGather(Kokkos::View<ScalarT*,AssemblyDevice> vec_dev, const int & type) {
  
  Kokkos::View<LO*,AssemblyDevice> numDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> data;
  Kokkos::View<int**,AssemblyDevice> offsets;
  LIDView LIDs;
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t c=0; c<cells[b].size(); c++) {
      switch(type) {
        case 0 :
          LIDs = cells[b][c]->LIDs;
          numDOF = cells[b][c]->cellData->numDOF;
          data = cells[b][c]->u;
          offsets = wkset[b]->offsets;
          break;
        case 1 : // deprecated (u_dot)
          break;
        case 2 :
          LIDs = cells[b][c]->LIDs;
          numDOF = cells[b][c]->cellData->numDOF;
          data = cells[b][c]->phi;
          offsets = wkset[b]->offsets;
          break;
        case 3 : // deprecated (phi_dot)
          break;
        case 4:
          LIDs = cells[b][c]->paramLIDs;
          numDOF = cells[b][c]->cellData->numParamDOF;
          data = cells[b][c]->param;
          offsets = wkset[b]->paramoffsets;
          break;
        default :
          cout << "ERROR - NOTHING WAS GATHERED" << endl;
      }
      
      parallel_for("assembly gather",RangePolicy<AssemblyExec>(0,data.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_t var=0; var<offsets.extent(0); var++) {
          for(size_t dof=0; dof<numDOF(var); dof++ ) {
            data(elem,var,dof) = vec_dev(LIDs(elem,offsets(var,dof)));
          }
        }
      });
      
    }
  }
}

// ========================================================================================
//
// ========================================================================================

void AssemblyManager::performBoundaryGather(Kokkos::View<ScalarT*,AssemblyDevice> vec_dev, const int & type) {
  
  for (size_t b=0; b<boundaryCells.size(); b++) {
    
    Kokkos::View<LO*,AssemblyDevice> numDOF;
    Kokkos::View<ScalarT***,AssemblyDevice> data;
    Kokkos::View<int**,AssemblyDevice> offsets;
    LIDView LIDs;
    
    for (size_t c=0; c < boundaryCells[b].size(); c++) {
      if (boundaryCells[b][c]->numElem > 0) {
        
        switch(type) {
          case 0 :
            LIDs = boundaryCells[b][c]->LIDs;
            numDOF = boundaryCells[b][c]->cellData->numDOF;
            data = boundaryCells[b][c]->u;
            offsets = wkset[b]->offsets;
            break;
          case 1 : // deprecated (u_dot)
            break;
          case 2 :
            LIDs = boundaryCells[b][c]->LIDs;
            numDOF = boundaryCells[b][c]->cellData->numDOF;
            data = boundaryCells[b][c]->phi;
            offsets = wkset[b]->offsets;
            break;
          case 3 : // deprecated (phi_dot)
            break;
          case 4:
            LIDs = boundaryCells[b][c]->paramLIDs;
            numDOF = boundaryCells[b][c]->cellData->numParamDOF;
            data = boundaryCells[b][c]->param;
            offsets = wkset[b]->paramoffsets;
            break;
          default :
            cout << "ERROR - NOTHING WAS GATHERED" << endl;
        }
        
        parallel_for("assembly boundary gather",RangePolicy<AssemblyExec>(0,data.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (size_t var=0; var<numDOF.extent(0); var++) {
            for(size_t dof=0; dof<numDOF(var); dof++ ) {
              data(elem,var,dof) = vec_dev(LIDs(elem,offsets(var,dof)));
            }
          }
        });
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class T>
void AssemblyManager::insert(T J_view, vector_RCP & res,
                             Kokkos::View<ScalarT***,HostDevice> local_res,
                             Kokkos::View<ScalarT***,HostDevice> local_J,
                             LIDView_host LIDs, LIDView_host paramLIDs,
                             const bool & compute_jacobian,
                             const bool & compute_disc_sens) {

  Teuchos::TimeMonitor localtimer(*inserttimer);
  
  /////////////////////////////////////
  // Using LIDs
  /////////////////////////////////////
  
  //parallel_for("assembly insert res",RangePolicy<HostExec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int elem ) {
  for( size_t elem=0; elem<LIDs.extent(0); elem++ ) {
    for( size_t row=0; row<LIDs.extent(1); row++ ) {
      LO rowIndex = LIDs(elem,row);
      if (!isFixedDOF(rowIndex)) {
        for (LO g=0; g<local_res.extent(2); g++) {
          ScalarT val = local_res(elem,row,g);
          res->sumIntoLocalValue(rowIndex,g, val);
        }
      }
    }
  }
  
  if (compute_jacobian) {
    
    //auto localMatrix = J->getLocalMatrix();
    
    if (compute_disc_sens) {
      //parallel_for("assembly insert Jac sens",RangePolicy<HostExec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for( size_t elem=0; elem<LIDs.extent(0); elem++ ) {
        for (size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(elem,row);
          for (size_t col=0; col<paramLIDs.extent(1); col++ ) {
            LO colIndex = paramLIDs(elem,col);
            ScalarT val = local_J(elem,row,col);
            //localMatrix.sumIntoValues(colIndex, &rowIndex, 1, &val, true, use_atomics); // isSorted, useAtomics
            J_view.sumIntoValues(colIndex, &rowIndex, 1, &val, true, use_atomics); // isSorted, useAtomics
          }
        }
      }
    }
    else {
      //parallel_for("assembly insert Jac",RangePolicy<HostExec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for( size_t elem=0; elem<LIDs.extent(0); elem++ ) {
        //const int numVals = static_cast<int>(LIDs.extent(1));
        const int numVals = LIDs.extent(1);
        LO cols[numVals];
        ScalarT vals[numVals];
        
        for (size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(elem,row);
          if (!isFixedDOF(rowIndex)) {
            for (size_t col=0; col<LIDs.extent(1); col++ ) {
              vals[col] = local_J(elem,row,col);
              cols[col] = LIDs(elem,col);
              //ScalarT vals = local_J(elem,row,col);
              //LO cols = LIDs(elem,col);
              //localMatrix.sumIntoValues(rowIndex, &cols, 1, &vals, true, use_atomics); // isSorted, useAtomics
            }
            //localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, true, use_atomics); // isSorted, useAtomics
            J_view.sumIntoValues(rowIndex, cols, numVals, vals, true, use_atomics); // isSorted, useAtomics
          }
        }
      }
    }
  }
}

