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

#include "assemblyManager.hpp"
#include "cellMetaData.hpp"

#include <boost/algorithm/string.hpp>

using namespace MrHyDE;

template class MrHyDE::AssemblyManager<SolverNode>;
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA) && !defined(MrHyDE_SOLVERSPACE_CUDA)
  template class MrHyDE::AssemblyManager<SubgridSolverNode>;
#endif

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class Node>
AssemblyManager<Node>::AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                 Teuchos::RCP<panzer_stk::STK_Interface> & mesh_, Teuchos::RCP<discretization> & disc_,
                                 Teuchos::RCP<physics> & phys_, Teuchos::RCP<ParameterManager<Node>> & params_,
                                 const int & numElemPerCell_) :
Comm(Comm_), settings(settings_), mesh(mesh_), disc(disc_), phys(phys_),
params(params_), numElemPerCell(numElemPerCell_) {
  
  // Get the required information from the settings
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting assembly manager constructor ..." << endl;
    }
  }
  
  verbosity = settings->get<int>("verbosity",0);
  usestrongDBCs = settings->sublist("Solver").get<bool>("use strong DBCs",true);
  
  // TMW: the following flag should only be used if there are extra variables, but no corresponding equation/constraint
  fix_zero_rows = settings->sublist("Solver").get<bool>("fix zero rows",false);
  
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("use measurements as DBCs", false);
  
  assembly_partitioning = settings->sublist("Solver").get<string>("assembly partitioning","sequential"); // "neighbor-avoiding"
  
  // TMW: Do we really want the user to have control over this?  Probably not ... redefining if on Cuda
  //use_atomics = settings->sublist("Solver").get<bool>("use atomics",false); // not needed if assembly partitioning is done correctly
  #if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
    #define use_atomics true
  #else
    #define use_atomics false
  #endif
  string solver_type = settings->sublist("Solver").get<string>("solver","none"); // or "transient"
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  
  // needed information from the mesh
  mesh->getElementBlockNames(blocknames);
  
  // check if we need to assembly volumetric, boundary and face terms
  for (size_t b=0; b<blocknames.size(); b++) {
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
  for (size_t b=0; b<blocknames.size(); b++) {
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
  
  this->createFixedDOFs();
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished assembly manager constructor" << endl;
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////
// Create the cells
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createFixedDOFs() {

  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createFixedDOFs ... " << endl;
    }
  }
  
  // create fixedDOF View of bools
  vector<vector<vector<LO> > > dbc_dofs = disc->dbc_dofs; // [block][var][dof]
  int numLocalDof = disc->DOF->getNumOwnedAndGhosted();
  isFixedDOF = Kokkos::View<bool*,LA_device>("logicals for fixed DOFs",numLocalDof);
  auto fixed_host = Kokkos::create_mirror_view(isFixedDOF);
  for (size_t block=0; block<dbc_dofs.size(); block++) {
    for (size_t var=0; var<dbc_dofs[block].size(); var++) {
      for (size_t i=0; i<dbc_dofs[block][var].size(); i++) {
        LO dof = dbc_dofs[block][var][i];
        fixed_host(dof) = true;
      }
    }
  }
  Kokkos::deep_copy(isFixedDOF,fixed_host);
  
  for (size_t block=0; block<dbc_dofs.size(); block++) {
    vector<Kokkos::View<LO*,LA_device> > block_dofs;
    for (size_t var=0; var<dbc_dofs[block].size(); var++) {
      Kokkos::View<LO*,LA_device> cfixed;
      if (dbc_dofs[block][var].size()>0) {
        cfixed = Kokkos::View<LO*,LA_device>("fixed DOFs",dbc_dofs[block][var].size());
        auto cfixed_host = Kokkos::create_mirror_view(cfixed);
        for (size_t i=0; i<dbc_dofs[block][var].size(); i++) {
          LO dof = dbc_dofs[block][var][i];
          cfixed_host(i) = dof;
        }
        Kokkos::deep_copy(cfixed,cfixed_host);
      }
      block_dofs.push_back(cfixed);
    }
    fixedDOF.push_back(block_dofs);
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::createFixedDOFs" << endl;
    }
  }
  
}
////////////////////////////////////////////////////////////////////////////////
// Create the cells
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createCells() {
  
  Teuchos::TimeMonitor localtimer(*celltimer);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createCells ..." << endl;
    }
  }
  
  vector<stk::mesh::Entity> all_meshElems;
  mesh->getMyElements(all_meshElems);
  
  Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device> LIDs = disc->DOF->getLIDs();
  
  for (size_t b=0; b<blocknames.size(); b++) {
    Teuchos::RCP<CellMetaData> blockCellData;
    vector<Teuchos::RCP<cell> > blockcells;
    vector<Teuchos::RCP<BoundaryCell> > bcells;
    
    vector<stk::mesh::Entity> stk_meshElems;
    mesh->getMyElements(blocknames[b], stk_meshElems);
    
    topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
    int numNodesPerElem = cellTopo->getNodeCount();
    int spaceDim = phys->spaceDim;
    LO numTotalElem = static_cast<LO>(stk_meshElems.size());
    
    if (numTotalElem>0) {
      
      vector<size_t> localIds;
      
      Kokkos::DynRankView<ScalarT,AssemblyDevice> blocknodes("nodes on block",numTotalElem,numNodesPerElem,spaceDim);
      auto host_blocknodes = Kokkos::create_mirror_view(blocknodes);
      panzer_stk::workset_utils::getIdsAndVertices(*mesh, blocknames[b], localIds, host_blocknodes); // fill on host
      Kokkos::deep_copy(blocknodes, host_blocknodes);
      
      vector<size_t> myElem = disc->myElements[b];
      Kokkos::View<LO*,AssemblyDevice> eIDs("local element IDs on device",myElem.size());
      auto host_eIDs = Kokkos::create_mirror_view(eIDs);
      for (size_t elem=0; elem<myElem.size(); elem++) {
        host_eIDs(elem) = static_cast<LO>(myElem[elem]);
      }
      Kokkos::deep_copy(eIDs, host_eIDs);
      
      // LO is int, but just in case that changes ...
      LO elemPerCell = static_cast<LO>(settings->sublist("Solver").get<int>("workset size",100));
      LO prog = 0;
      
      vector<string> sideSets;
      mesh->getSidesetNames(sideSets);
      
      blockCellData = Teuchos::rcp( new CellMetaData(settings, cellTopo,
                                                     phys, b, 0,
                                                     build_face_terms[b],
                                                     assemble_face_terms[b],
                                                     sideSets,
                                                     params->num_discretized_params));
                                                     
      disc->setReferenceData(blockCellData);
      
      blockCellData->requireBasisAtNodes = settings->sublist("Postprocess").get<bool>("plot solution at nodes",false);
      
      vector<vector<int> > curroffsets = disc->offsets[b];
      Kokkos::View<LO*,AssemblyDevice> numDOF_KV("number of DOF per variable",curroffsets.size());
      Kokkos::View<LO*,HostDevice> numDOF_host("numDOF on host",curroffsets.size());
      for (size_t k=0; k<curroffsets.size(); k++) {
        numDOF_host(k) = static_cast<LO>(curroffsets[k].size());
      }
      Kokkos::deep_copy(numDOF_KV, numDOF_host);
      
      blockCellData->numDOF = numDOF_KV;
      blockCellData->numDOF_host = numDOF_host;
      
      if (assembly_partitioning == "sequential") {
        while (prog < numTotalElem) {
          LO currElem = elemPerCell;  // Avoid faults in last iteration
          if (prog+currElem > numTotalElem){
            currElem = numTotalElem-prog;
          }
          
          Kokkos::View<LO*,AssemblyDevice> eIndex("element indices",currElem);
          DRV currnodes("currnodes", currElem, numNodesPerElem, spaceDim);
          LIDView cellLIDs("LIDs on device",currElem,LIDs.extent(1));
          
          auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
          Kokkos::View<LO*,HostDevice> host_eIndex2("element indices on host",currElem);
          
          auto nodes_sub = Kokkos::subview(blocknodes,std::make_pair(prog, prog+currElem), Kokkos::ALL(), Kokkos::ALL());
          Kokkos::deep_copy(currnodes,nodes_sub);
          
          for (size_t e=0; e<host_eIndex.extent(0); e++) {
            host_eIndex(e) = prog+static_cast<LO>(e);//disc->myElements[b][prog+e]; // TMW: why here?;prog+e;
          }
          
          Kokkos::deep_copy(eIndex,host_eIndex);
          Kokkos::deep_copy(host_eIndex2,host_eIndex);
          // This subview only works if the cells use a continuous ordering of elements
          // Considering generalizing this to reduce atomic overhead, so performing manual deep copy for now
          //LIDView cellLIDs = Kokkos::subview(LIDs, std::make_pair(prog,prog+currElem), Kokkos::ALL());
          LO progend = prog + static_cast<LO>(cellLIDs.extent(0));
          auto celem = Kokkos::subview(eIDs, std::make_pair(prog,progend));
          parallel_for("assembly copy LIDs",RangePolicy<AssemblyExec>(0,cellLIDs.extent(0)), KOKKOS_LAMBDA (const int e ) {
            LO elemID = celem(e);//disc->myElements[b][prog+e]; // TMW: why here?
            for (size_type j=0; j<LIDs.extent(1); j++) {
              cellLIDs(e,j) = LIDs(elemID,j);
            }
          });
          
          // Set the side information (soon to be removed)-
          Kokkos::View<int****,HostDevice> sideinfo = disc->getSideInfo(b,host_eIndex2);
          
          blockcells.push_back(Teuchos::rcp(new cell(blockCellData, currnodes, eIndex,
                                                     cellLIDs, sideinfo, disc)));
          prog += elemPerCell;
        }
      }
      else if (assembly_partitioning == "random") { // not implemented yet
        
      }
      else if (assembly_partitioning == "neighbor-avoiding") { // not implemented yet
        // need neighbor information
      }
            
      //////////////////////////////////////////////////////////////////////////////////
      // Boundary cells
      //////////////////////////////////////////////////////////////////////////////////
      
      if (build_boundary_terms[b]) {
        // TMW: this is just for ease of use
        int numBoundaryElem = settings->sublist("Solver").get<int>("workset size",100);
        
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
          
          DRV sidenodes;
          mesh->getElementVertices(side_output, blocknames[b],sidenodes);
          
          size_t numSideElem = local_side_Ids.size();
          
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
              
              size_t prog = 0;
              while (prog < group.size()) {
                size_t currElem = numBoundaryElem;
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
                
                for (size_t e=0; e<currElem; e++) {
                  host_eIndex(e) = mesh->elementLocalId(side_output[group[e+prog]]);
                  host_sideIndex(e) = local_side_Ids[group[e+prog]];
                  for (size_type n=0; n<host_currnodes.extent(1); n++) {
                    for (size_type m=0; m<host_currnodes.extent(2); m++) {
                      host_currnodes(e,n,m) = sidenodes(group[e+prog],n,m);
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
                  for (size_type j=0; j<LIDs.extent(1); j++) {
                    cellLIDs(e,j) = LIDs(elemID,j);
                  }
                });
                
                //-----------------------------------------------
                // Set the side information (soon to be removed)-
                Kokkos::View<int****,HostDevice> sideinfo = disc->getSideInfo(b,host_eIndex2);
                
                bcells.push_back(Teuchos::rcp(new BoundaryCell(blockCellData, currnodes, eIndex, sideIndex,
                                                               side, sideName, bcells.size(),
                                                               cellLIDs, sideinfo, disc)));//, orient_drv)));
                prog += currElem;
              }
            }
          }
        }
      }
    
      
    }
    
    
    cellData.push_back(blockCellData);
    cells.push_back(blockcells);
    boundaryCells.push_back(bcells);
    
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::createCells" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createWorkset() {
  
  Teuchos::TimeMonitor localtimer(*wksettimer);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createWorkset ..." << endl;
    }
  }
  
  for (size_t b=0; b<cells.size(); b++) {
    if (cells[b].size() > 0) {
      vector<int> info;
      info.push_back(cellData[b]->dimension);
      info.push_back(cellData[b]->numDOF.extent(0));
      info.push_back((int)cellData[b]->numDiscParams);
      info.push_back(cellData[b]->numAuxDOF.extent(0));
      info.push_back(cells[b][0]->numElem);
      info.push_back(cellData[b]->numip);
      info.push_back(cellData[b]->numsideip);
      wkset.push_back(Teuchos::rcp( new workset(info,
                                                isTransient,
                                                disc->basis_types[b],
                                                disc->basis_pointers[b],
                                                params->discretized_param_basis,
                                                mesh->getCellTopology(blocknames[b]),
                                                disc->var_bcs[b]) ) );
      
      wkset[b]->isInitialized = true;
      wkset[b]->block = b;
    }
    else {
      wkset.push_back(Teuchos::rcp( new workset()));
      wkset[b]->isInitialized = false;
      wkset[b]->block = b;
    }
  }
  
  //phys->setWorkset(wkset);
  //params->wkset = wkset;
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::createWorkset" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

// TMW: this might be deprecated
template<class Node>
void AssemblyManager<Node>::updateJacDBC(matrix_RCP & J, const vector<vector<GO> > & dofs,
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
void AssemblyManager<Node>::updateJacDBC(matrix_RCP & J,
                                         const vector<LO> & dofs, const bool & compute_disc_sens) {
  
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
void AssemblyManager<Node>::setInitial(vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                                       const bool & lumpmass, const ScalarT & scale) {
  
  // TMW: ToDo - should add a lumped mass option
  
  Teuchos::TimeMonitor localtimer(*setinittimer);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::setInitial ..." << endl;
    }
  }
  
  auto localMatrix = mass->getLocalMatrix();
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      
      LIDView_host LIDs = cells[b][e]->LIDs_host;
      
      Kokkos::View<ScalarT**,AssemblyDevice> localrhs = cells[b][e]->getInitial(true, useadjoint);
      Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[b][e]->getMass();
      auto host_rhs = Kokkos::create_mirror_view(localrhs);
      auto host_mass = Kokkos::create_mirror_view(localmass);
      Kokkos::deep_copy(host_rhs,localrhs);
      Kokkos::deep_copy(host_mass,localmass);
      
      // Would prefer to rewrite this
      //parallel_for("assembly copy LIDs",RangePolicy<HostExec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_type e=0; e<LIDs.extent(0); e++) {
        //const int numVals = static_cast<int>(LIDs.extent(1));
        //const int numVals = LIDs.extent(1);
        //int numVals = LIDs.extent(1);
        //LO cols[numVals];
        //ScalarT vals[numVals];
        
        for( size_type row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(e,row);
          ScalarT val = host_rhs(e,row);
          rhs->sumIntoLocalValue(rowIndex, 0, val);
          for( size_type col=0; col<LIDs.extent(1); col++ ) {
            ScalarT vals = scale*host_mass(e,row,col);
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
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::setInitial ..." << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::setInitial(vector_RCP & initial, const bool & useadjoint) {

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

template<class Node>
void AssemblyManager<Node>::setDirichlet(vector_RCP & rhs, matrix_RCP & mass,
                                   const bool & useadjoint,
                                   const ScalarT & time,
                                   const bool & lumpmass) {
  
  Teuchos::TimeMonitor localtimer(*setdbctimer);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::setDirichlet ..." << endl;
    }
  }
  
  auto localMatrix = mass->getLocalMatrix();
  
  for (size_t b=0; b<boundaryCells.size(); b++) {
    wkset[b]->setTime(time);
    for (size_t e=0; e<boundaryCells[b].size(); e++) {
      
      int numElem = boundaryCells[b][e]->numElem;
      auto LIDs = boundaryCells[b][e]->LIDs_host;
      auto localrhs = boundaryCells[b][e]->getDirichlet();
      auto localmass = boundaryCells[b][e]->getMass();
      auto host_rhs = Kokkos::create_mirror_view(localrhs);
      auto host_mass = Kokkos::create_mirror_view(localmass);
      Kokkos::deep_copy(host_rhs,localrhs);
      Kokkos::deep_copy(host_mass,localmass);
      
      //const int numVals = static_cast<int>(LIDs.extent(1));
      size_t numVals = LIDs.extent(1);
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
              localMatrix.sumIntoValues(rowIndex, cols, 1, vals, true, false);
            }
            else {
              LO cols[maxDerivs];
              ScalarT vals[maxDerivs];
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
  }
  
  // Loop over the cells to put ones on the diagonal for DOFs not on Dirichlet boundaries
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      auto LIDs = cells[b][e]->LIDs_host;
      for (size_t c=0; c<cells[b][e]->numElem; c++) {
        for( size_type row=0; row<LIDs.extent(1); row++ ) {
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
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::setDirichlet ..." << endl;
    }
  }
  
}


// ========================================================================================
// Wrapper to the main assembly routine to assemble over all blocks (most common use case)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::assembleJacRes(vector_RCP & u, vector_RCP & phi,
                                     const bool & compute_jacobian, const bool & compute_sens,
                                     const bool & compute_disc_sens,
                                     vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                     const ScalarT & current_time,
                                     const bool & useadjoint, const bool & store_adjPrev,
                                     const int & num_active_params,
                                     vector_RCP & Psol, const bool & is_final_time,
                                     const ScalarT & deltat) {
  
  if (debug_level > 1) {
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
    if (cells[b].size() > 0) {
      this->assembleJacRes(compute_jacobian,
                           compute_sens, compute_disc_sens, res, J, isTransient,
                           current_time, useadjoint, store_adjPrev, num_active_params,
                           is_final_time, b, deltat);
    }
  }
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Finished AssemblyManager::assembleJacRes" << endl;
    }
  }
}

// ========================================================================================
// Main assembly routine ... only assembles on a given block (b)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::assembleJacRes(const bool & compute_jacobian, const bool & compute_sens,
                                     const bool & compute_disc_sens,
                                     vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                     const ScalarT & current_time,
                                     const bool & useadjoint, const bool & store_adjPrev,
                                     const int & num_active_params,
                                     const bool & is_final_time,
                                     const int & b, const ScalarT & deltat) {
  
  Teuchos::TimeMonitor localassemblytimer(*assemblytimer);
  
  // Kokkos::CRSMatrix and Kokkos::View for J and res
  // Scatter needs to be on LA_device
  auto J_kcrs = J->getLocalMatrix();
  auto res_view = res->template getLocalView<LA_device>();
  
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
  if (!data_avail || useadjoint || cellData[b]->multiscale) {
    reduce_memory = false;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Set up the worksets and allocate the local residual and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////
  
  if (isTransient) {
    // TMW: tmp fix
    auto butcher_c = Kokkos::create_mirror_view(wkset[b]->butcher_c);
    Kokkos::deep_copy(butcher_c, wkset[b]->butcher_c);
    ScalarT timeval = current_time + butcher_c(wkset[b]->current_stage)*deltat;
    
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
  auto local_res_ladev = create_mirror(LA_exec(),local_res);
  auto local_J_ladev = create_mirror(LA_exec(),local_J);
  
  /////////////////////////////////////////////////////////////////////////////
  // Volume contribution
  /////////////////////////////////////////////////////////////////////////////
  
  // Note: Cannot parallelize over cells since data structures are re-used
  
  for (size_t e=0; e < cells[b].size(); e++) {

    wkset[b]->localEID = e;
    cells[b][e]->updateData();
    
    if (isTransient && useadjoint && !cells[b][0]->cellData->multiscale) {
      if (is_final_time) {
        cells[b][e]->resetAdjPrev(0.0);
      }
    }
 
    Kokkos::fence();

    /////////////////////////////////////////////////////////////////////////////
    // Compute the local residual and Jacobian on this cell
    /////////////////////////////////////////////////////////////////////////////
    
    bool fixJacDiag = false;
    
    {
      Teuchos::TimeMonitor localtimer(*phystimer);
      
      wkset[b]->resetResidual();
      
      if (useadjoint) {
        wkset[b]->resetAdjointRHS();
      }
      
      //////////////////////////////////////////////////////////////
      // Compute the AD-seeded solutions at integration points
      //////////////////////////////////////////////////////////////
      
      int seedwhat = 0;
      if (compute_jacobian) {
        if (compute_disc_sens) {
          seedwhat = 3;
        }
        else {
          seedwhat = 1;
        }
      }
      
      if (!(cellData[b]->multiscale)) {
        if (isTransient) {
          wkset[b]->computeSolnTransientSeeded(cells[b][e]->u,
                                               cells[b][e]->u_prev,
                                               cells[b][e]->u_stage,
                                               seedwhat);
        }
        else { // steady-state
          wkset[b]->computeSolnSteadySeeded(cells[b][e]->u, seedwhat);
        }
        if (wkset[b]->numParams > 0) {
          wkset[b]->computeParamSteadySeeded(cells[b][e]->param, seedwhat);
        }
      }
      Kokkos::fence();
      
      //////////////////////////////////////////////////////////////
      // Compute res and J=dF/du
      //////////////////////////////////////////////////////////////
      
      // Volumetric contribution
      if (assemble_volume_terms[b]) {
        if (cellData[b]->multiscale) {
          int sgindex = cells[b][e]->subgrid_model_index[cells[b][e]->subgrid_model_index.size()-1];
          cells[b][e]->subgridModels[sgindex]->subgridSolver(cells[b][e]->u, cells[b][e]->phi, wkset[b]->time, isTransient, useadjoint,
                                                             compute_jacobian, compute_sens, num_active_params,
                                                             compute_disc_sens, false,
                                                             *(wkset[b]), cells[b][e]->subgrid_usernum, 0,
                                                             cells[b][e]->subgradient, store_adjPrev);
          fixJacDiag = true;
        }
        else {
          cells[b][e]->computeSolnVolIP();
          phys->volumeResidual(b);
        }
      }
      Kokkos::fence();
      
      ///////////////////////////////////////////////////////////////////////////
      // Edge/face contribution
      ///////////////////////////////////////////////////////////////////////////
      
      if (assemble_face_terms[b]) {
        if (cellData[b]->multiscale) {
          // do nothing
        }
        else {
          for (size_t s=0; s<cellData[b]->numSides; s++) {
            cells[b][e]->computeSolnFaceIP(s);
            phys->faceResidual(b);
          }
        }
      }
      
    }
        
    ///////////////////////////////////////////////////////////////////////////
    // Scatter into global matrix/vector
    ///////////////////////////////////////////////////////////////////////////
    
    if (reduce_memory) { // skip local_res and local_J
      this->scatter(J_kcrs, res_view,
                    cells[b][e]->LIDs, cells[b][e]->paramLIDs, b,
                    compute_jacobian, compute_sens, compute_disc_sens, useadjoint);
    }
    else { // fill local_res and local_J and then scatter
    
      Kokkos::deep_copy(local_res,0.0);
      Kokkos::deep_copy(local_J,0.0);
      
      // Use AD residual to update local Jacobian
      if (compute_jacobian) {
        if (compute_disc_sens) {
          cells[b][e]->updateParamJac(local_J);
        }
        else {
          cells[b][e]->updateJac(useadjoint, local_J);
        }
      }
      
      if (compute_jacobian && fixJacDiag) {
        cells[b][e]->fixDiagJac(local_J, local_res);
      }
      
      // Update the local residual
      if (useadjoint) {
        cells[b][e]->updateAdjointRes(compute_sens, local_res);
      }
      else {
        cells[b][e]->updateRes(compute_sens, local_res);
      }
      
      if (useadjoint) {
        cells[b][e]->updateAdjointRes(compute_jacobian, isTransient,
                                      false, store_adjPrev,
                                      local_J, local_res);
        
        
      }
      
      // Now scatter from local_res and local_J
      
      if (data_avail) {
        this->scatter(J_kcrs, res_view, local_res, local_J,
                      cells[b][e]->LIDs, cells[b][e]->paramLIDs,
                      compute_jacobian, compute_disc_sens);
      }
      else {
        Kokkos::deep_copy(local_J_ladev,local_J);
        Kokkos::deep_copy(local_res_ladev,local_res);
        
        if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
          this->scatter(J_kcrs, res_view, local_res_ladev, local_J_ladev,
                        cells[b][e]->LIDs_host, cells[b][e]->paramLIDs_host,
                        compute_jacobian, compute_disc_sens);
        }
        else { // LA_device = CUDA, AssemblyDevice = Host
          // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
          auto LIDs_dev = Kokkos::create_mirror(LA_exec(), cells[b][e]->LIDs);
          auto paramLIDs_dev = Kokkos::create_mirror(LA_exec(), cells[b][e]->paramLIDs);
          Kokkos::deep_copy(LIDs_dev,cells[b][e]->LIDs);
          Kokkos::deep_copy(paramLIDs_dev,cells[b][e]->paramLIDs);
          
          this->scatter(J_kcrs, res_view, local_res_ladev, local_J_ladev,
                        LIDs_dev, paramLIDs_dev,
                        compute_jacobian, compute_disc_sens);
        }
        
      }
    }
    
  } // element loop
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Boundary terms
  //////////////////////////////////////////////////////////////////////////////////////
  
  if (!cells[b][0]->cellData->multiscale && assemble_boundary_terms[b]) {
    
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
    auto local_res_ladev = create_mirror(LA_exec(),local_res);
    auto local_J_ladev = create_mirror(LA_exec(),local_J);
    
    for (size_t e=0; e < boundaryCells[b].size(); e++) {
      
      if (boundaryCells[b][e]->numElem > 0) {
        wkset[b]->localEID = e;
        
        /////////////////////////////////////////////////////////////////////////////
        // Compute the local residual and Jacobian on this cell
        /////////////////////////////////////////////////////////////////////////////
        
        {
          Teuchos::TimeMonitor localtimer(*phystimer);
          wkset[b]->sidename = boundaryCells[b][e]->sidename;
          wkset[b]->currentside = boundaryCells[b][e]->sidenum;
          
          int seedwhat = 0;
          if (compute_jacobian) {
            if (compute_disc_sens) {
              seedwhat = 3;
            }
            else {
              seedwhat = 1;
            }
          }
          
          if (isTransient) {
            wkset[b]->computeSolnTransientSeeded(boundaryCells[b][e]->u,
                                                 boundaryCells[b][e]->u_prev,
                                                 boundaryCells[b][e]->u_stage,
                                                 seedwhat);
          }
          else { // steady-state
            wkset[b]->computeSolnSteadySeeded(boundaryCells[b][e]->u, seedwhat);
          }
          if (wkset[b]->numParams > 0) {
            wkset[b]->computeParamSteadySeeded(boundaryCells[b][e]->param, seedwhat);
          }
        
          boundaryCells[b][e]->updateWorksetBasis();
          boundaryCells[b][e]->computeSoln(seedwhat);
          
          wkset[b]->resetResidual();
          
          phys->boundaryResidual(b);
          
        }
        
        ///////////////////////////////////////////////////////////////////////////
        // Scatter into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        if (reduce_memory) { // skip local_res and local_J
          this->scatter(J_kcrs, res_view,
                        boundaryCells[b][e]->LIDs, boundaryCells[b][e]->paramLIDs, b,
                        compute_jacobian, compute_sens, compute_disc_sens, useadjoint);
        }
        else { // fill local_res and local_J and then scatter
        
          Kokkos::deep_copy(local_res,0.0);
          Kokkos::deep_copy(local_J,0.0);
        
          // Use AD residual to update local Jacobian
          if (compute_jacobian) {
            if (compute_disc_sens) {
              boundaryCells[b][e]->updateParamJac(local_J);
            }
            else {
              boundaryCells[b][e]->updateJac(useadjoint, local_J);
            }
          }
          
          // Update the local residual (forward mode)
          if (useadjoint) {
            boundaryCells[b][e]->updateAdjointRes(compute_sens, local_res);
          }
          else {
            boundaryCells[b][e]->updateRes(compute_sens, local_res);
          }
         
          if (data_avail) {
            this->scatter(J_kcrs, res_view, local_res, local_J,
                          boundaryCells[b][e]->LIDs, boundaryCells[b][e]->paramLIDs,
                          compute_jacobian, compute_disc_sens);
          }
          else {
            Kokkos::deep_copy(local_J_ladev,local_J);
            Kokkos::deep_copy(local_res_ladev,local_res);
            
            if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
              this->scatter(J_kcrs, res_view, local_res_ladev, local_J_ladev,
                            boundaryCells[b][e]->LIDs_host, boundaryCells[b][e]->paramLIDs_host,
                            compute_jacobian, compute_disc_sens);
            }
            else { // LA_device = CUDA, AssemblyDevice = Host
              // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
              auto LIDs_dev = Kokkos::create_mirror(LA_exec(), boundaryCells[b][e]->LIDs);
              auto paramLIDs_dev = Kokkos::create_mirror(LA_exec(), boundaryCells[b][e]->paramLIDs);
              Kokkos::deep_copy(LIDs_dev,boundaryCells[b][e]->LIDs);
              Kokkos::deep_copy(paramLIDs_dev,boundaryCells[b][e]->paramLIDs);
              
              this->scatter(J_kcrs, res_view, local_res_ladev, local_J_ladev,
                            LIDs_dev, paramLIDs_dev,
                            compute_jacobian, compute_disc_sens);
            }
            
          }
        }
        
      }
    } // element loop
  }
  
  // Apply constraints, e.g., strongly imposed Dirichlet
  this->dofConstraints(J, res, current_time, compute_jacobian, compute_disc_sens);
  
  
  if (fix_zero_rows) {
    size_t numrows = J->getNodeNumRows();
    
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
// Enforce DOF constraints - includes strong Dirichlet
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::dofConstraints(matrix_RCP & J, vector_RCP & res,
                                     const ScalarT & current_time,
                                     const bool & compute_jacobian,
                                     const bool & compute_disc_sens) {
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Starting AssemblyManager::dofConstraints" << endl;
    }
  }
  
  Teuchos::TimeMonitor localtimer(*dbctimer);
  
  if (usestrongDBCs) {
    vector<vector<vector<LO> > > dbcDOFs = disc->dbc_dofs;
    for (size_t block=0; block<dbcDOFs.size(); block++) {
      for (size_t var=0; var<dbcDOFs[block].size(); var++) {
        if (compute_jacobian) {
          this->updateJacDBC(J,dbcDOFs[block][var],compute_disc_sens);
        }
      }
    }
  }
  
  vector<vector<GO> > fixedDOFs = disc->point_dofs;
  for (size_t block=0; block<fixedDOFs.size(); block++) {
    if (compute_jacobian) {
      this->updateJacDBC(J,fixedDOFs,block,compute_disc_sens);
    }
  }
  
  if (debug_level > 1) {
    if (Comm->getRank() == 0) {
      cout << "******** Finished AssemblyManager::dofConstraints" << endl;
    }
  }
  
}


// ========================================================================================
//
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::resetPrevSoln() {
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->resetPrevSoln();
    }
  }
}

template<class Node>
void AssemblyManager<Node>::resetStageSoln() {
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->resetStageSoln();
    }
  }
}

template<class Node>
void AssemblyManager<Node>::updateStageNumber(const int & stage) {
  for (size_t b=0; b<wkset.size(); b++) {
    wkset[b]->setStage(stage);
  }
}

template<class Node>
void AssemblyManager<Node>::updateStageSoln()  {
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->updateStageSoln();
    }
  }
}

// ========================================================================================
// Gather local solutions on cells.
// This intermediate function allows us to copy the data from LA_device to AssemblyDevice only once (if necessary)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::performGather(const vector_RCP & vec, const int & type, const size_t & entry) {
  
  typedef typename LA_device::memory_space LA_mem;
  
  auto vec_kv = vec->template getLocalView<LA_device>();
  
  // Even if there are multiple vectors, we only use one at a time
  auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), entry);
  
  // vector is on LA_device, but gather must happen on AssemblyDevice
  if (Kokkos::SpaceAccessibility<AssemblyExec, LA_mem>::accessible) { // can we avoid a copy?
    this->performGather(vec_slice, type);
    this->performBoundaryGather(vec_slice, type);
  }
  else { // apparently not
    auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
    Kokkos::deep_copy(vec_dev,vec_slice);
    this->performGather(vec_dev, type);
    this->performBoundaryGather(vec_dev, type);
  }
  
}

// ========================================================================================
//
// ========================================================================================

template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performGather(ViewType vec_dev, const int & type) {

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
        for (size_type var=0; var<offsets.extent(0); var++) {
          for(int dof=0; dof<numDOF(var); dof++ ) {
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

template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performBoundaryGather(ViewType vec_dev, const int & type) {
  
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
            for(int dof=0; dof<numDOF(var); dof++ ) {
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

template<class Node>
template<class MatType, class VecViewType, class LocalViewType, class LIDViewType>
void AssemblyManager<Node>::scatter(MatType J_kcrs, VecViewType res_view,
                                    LocalViewType local_res, LocalViewType local_J,
                                    LIDViewType LIDs, LIDViewType paramLIDs,
                                    const bool & compute_jacobian,
                                    const bool & compute_disc_sens) {

  Teuchos::TimeMonitor localtimer(*scattertimer);
  
  typedef typename Node::execution_space LA_exec;
  
  /////////////////////////////////////
  // This scatter needs to happen on the LA_device due to the use of J_kcrs->sumIntoValues()
  // Could be changed to the AssemblyDevice, but would require a mirror view of this data and filling such a view is nontrivial
  /////////////////////////////////////
  
  auto fixedDOF = isFixedDOF;
  
  if (use_atomics) { // If LA_device = Kokkos::Serial or if Worksets are colored
    parallel_for("assembly scatter res",
                 RangePolicy<LA_exec>(0,LIDs.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for( size_type row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(elem,row);
        if (!fixedDOF(rowIndex)) {
          for (size_type g=0; g<local_res.extent(2); g++) {
            ScalarT val = local_res(elem,row,g);
            Kokkos::atomic_add(&(res_view(rowIndex,g)), val);
          }
        }
      }
    });
  }
  else {
    parallel_for("assembly scatter res",
                 RangePolicy<LA_exec>(0,LIDs.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for( size_type row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(elem,row);
        if (!fixedDOF(rowIndex)) {
          for (size_type g=0; g<local_res.extent(2); g++) {
            ScalarT val = local_res(elem,row,g);
            res_view(rowIndex,g) += val;
          }
        }
      }
    });
  }
  
  if (compute_jacobian) {
    
    if (compute_disc_sens) {
      if (use_atomics) { // If LA_device = Kokkos::Serial or if Worksets are colored
        parallel_for("assembly insert Jac sens",
                     RangePolicy<LA_exec>(0,LIDs.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_t row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(elem,row);
            for (size_t col=0; col<paramLIDs.extent(1); col++ ) {
              LO colIndex = paramLIDs(elem,col);
              ScalarT val = local_J(elem,row,col);
              J_kcrs.sumIntoValues(colIndex, &rowIndex, 1, &val, true, true); // isSorted, useAtomics
            }
          }
        });
      }
      else {
        parallel_for("assembly insert Jac sens",
                     RangePolicy<LA_exec>(0,LIDs.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_t row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(elem,row);
            for (size_t col=0; col<paramLIDs.extent(1); col++ ) {
              LO colIndex = paramLIDs(elem,col);
              ScalarT val = local_J(elem,row,col);
              J_kcrs.sumIntoValues(colIndex, &rowIndex, 1, &val, true, false); // isSorted, useAtomics
            }
          }
        });
      }
      
    }
    else {
      if (use_atomics) { // If LA_device = Kokkos::Serial or if Worksets are colored
        parallel_for("assembly insert Jac",
                     RangePolicy<LA_exec>(0,LIDs.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          const size_type numVals = LIDs.extent(1);
          LO cols[maxDerivs];
          ScalarT vals[maxDerivs];
          for (size_type row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(elem,row);
            if (!fixedDOF(rowIndex)) {
              for (size_type col=0; col<LIDs.extent(1); col++ ) {
                vals[col] = local_J(elem,row,col);
                cols[col] = LIDs(elem,col);
              }
              J_kcrs.sumIntoValues(rowIndex, cols, numVals, vals, true, true); // isSorted, useAtomics
            }
          }
        });
      }
      else {
        parallel_for("assembly insert Jac",
                     RangePolicy<LA_exec>(0,LIDs.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          const size_type numVals = LIDs.extent(1);
          LO cols[maxDerivs];
          ScalarT vals[maxDerivs];
          for (size_type row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(elem,row);
            if (!fixedDOF(rowIndex)) {
              for (size_type col=0; col<LIDs.extent(1); col++ ) {
                vals[col] = local_J(elem,row,col);
                cols[col] = LIDs(elem,col);
              }
              J_kcrs.sumIntoValues(rowIndex, cols, numVals, vals, true, false); // isSorted, useAtomics
            }
          }
        });
      }
    }
  }
}


template<class Node>
template<class MatType, class VecViewType, class LIDViewType>
void AssemblyManager<Node>::scatter(MatType J_kcrs, VecViewType res_view,
                                    LIDViewType LIDs, LIDViewType paramLIDs,
                                    const int & block,
                                    const bool & compute_jacobian,
                                    const bool & compute_sens,
                                    const bool & compute_disc_sens,
                                    const bool & isAdjoint) {

  Teuchos::TimeMonitor localtimer(*scattertimer);
  
  typedef typename Node::execution_space LA_exec;
  
  /////////////////////////////////////
  // This scatter needs to happen on the LA_device due to the use of J_kcrs->sumIntoValues()
  // Could be changed to the AssemblyDevice, but would require a mirror view of this data and filling such a view is nontrivial
  /////////////////////////////////////
  
  auto fixedDOF = isFixedDOF;
  auto res = wkset[block]->res;
  if (isAdjoint) {
    res = wkset[block]->adjrhs;
  }
  auto offsets = wkset[block]->offsets;
  auto numDOF = cellData[block]->numDOF;
  
  if (use_atomics) { // If LA_device = Kokkos::Serial or if Worksets are colored
    if (compute_sens) {
      parallel_for("assembly insert Jac",
                   RangePolicy<LA_exec>(0,LIDs.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type n=0; n<numDOF.extent(0); ++n) {
          for (int j=0; j<numDOF(n); j++) {
            int row = offsets(n,j);
            LO rowIndex = LIDs(elem,row);
            if (!fixedDOF(rowIndex)) {
              for (size_type r=0; r<res_view.extent(1); ++r) {
                ScalarT val = -res(elem,row).fastAccessDx(r);
                Kokkos::atomic_add(&(res_view(rowIndex,r)), val);
              }
            }
          }
        }
      });
    }
    else {
      parallel_for("assembly insert Jac",
                   RangePolicy<LA_exec>(0,LIDs.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type n=0; n<numDOF.extent(0); ++n) {
          for (int j=0; j<numDOF(n); j++) {
            int row = offsets(n,j);
            LO rowIndex = LIDs(elem,row);
            if (!fixedDOF(rowIndex)) {
              ScalarT val = -res(elem,row).val();
              Kokkos::atomic_add(&(res_view(rowIndex,0)), val);
            }
          }
        }
      });
    }
  }
  else {
    if (compute_sens) {
      parallel_for("assembly insert Jac",
                   RangePolicy<LA_exec>(0,LIDs.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type n=0; n<numDOF.extent(0); ++n) {
          for (int j=0; j<numDOF(n); j++) {
            int row = offsets(n,j);
            LO rowIndex = LIDs(elem,row);
            if (!fixedDOF(rowIndex)) {
              for (size_type r=0; r<res_view.extent(1); ++r) {
                ScalarT val = -res(elem,row).fastAccessDx(r);
                res_view(rowIndex,r) += val;
              }
            }
          }
        }
      });
    }
    else {
      parallel_for("assembly insert Jac",
                   RangePolicy<LA_exec>(0,LIDs.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type n=0; n<numDOF.extent(0); ++n) {
          for (int j=0; j<numDOF(n); j++) {
            int row = offsets(n,j);
            LO rowIndex = LIDs(elem,row);
            if (!fixedDOF(rowIndex)) {
              ScalarT val = -res(elem,row).val();
              res_view(rowIndex,0) += val;
            }
          }
        }
      });
    }
  }
  
  if (compute_jacobian) {
    
    if (compute_disc_sens) {
      /*
      parallel_for("assembly insert Jac sens",
                   RangePolicy<LA_exec>(0,LIDs.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(elem,row);
          for (size_t col=0; col<paramLIDs.extent(1); col++ ) {
            LO colIndex = paramLIDs(elem,col);
            ScalarT val = local_J(elem,row,col);
            J_kcrs.sumIntoValues(colIndex, &rowIndex, 1, &val, true, use_atomics); // isSorted, useAtomics
          }
        }
      });
      */
    }
    else {
      if (isAdjoint) {
        parallel_for("assembly insert Jac",
                     RangePolicy<LA_exec>(0,LIDs.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          const size_type numVals = LIDs.extent(1);
          LO cols[maxDerivs];
          ScalarT vals[maxDerivs];
          for (size_type n=0; n<numDOF.extent(0); ++n) {
            for (int j=0; j<numDOF(n); j++) {
              int row = offsets(n,j);
              LO rowIndex = LIDs(elem,row);
              if (!fixedDOF(rowIndex)) {
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    int col = offsets(m,k);
                    vals[col] = res(elem,col).fastAccessDx(row);
                    cols[col] = LIDs(elem,col);
                  }
                }
                J_kcrs.sumIntoValues(rowIndex, cols, numVals, vals, true, use_atomics); // isSorted, useAtomics
              }
            }
          }
        });
      }
      else {
        parallel_for("assembly insert Jac",
                     RangePolicy<LA_exec>(0,LIDs.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          const size_type numVals = LIDs.extent(1);
          LO cols[maxDerivs];
          ScalarT vals[maxDerivs];
          for (size_type n=0; n<numDOF.extent(0); ++n) {
            for (int j=0; j<numDOF(n); j++) {
              int row = offsets(n,j);
              LO rowIndex = LIDs(elem,row);
              if (!fixedDOF(rowIndex)) {
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    int col = offsets(m,k);
                    vals[col] = res(elem,row).fastAccessDx(col);
                    cols[col] = LIDs(elem,col);
                  }
                }
                J_kcrs.sumIntoValues(rowIndex, cols, numVals, vals, true, use_atomics); // isSorted, useAtomics
              }
            }
          }
        });
      }
    }
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we can get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::purgeMemory() {
  bool write_solution = settings->sublist("Postprocess").get("write solution",false);
  if (!write_solution) {
    mesh.reset();
  }
}
