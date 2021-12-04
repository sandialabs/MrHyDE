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

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class Node>
AssemblyManager<Node>::AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_,
                                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                       Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                                       Teuchos::RCP<DiscretizationInterface> & disc_,
                                       Teuchos::RCP<PhysicsInterface> & phys_,
                                       Teuchos::RCP<ParameterManager<Node>> & params_) :
Comm(Comm_), settings(settings_), mesh(mesh_), disc(disc_), phys(phys_), params(params_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
    
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
  
  // Really, this lumps the Jacobian and should only be used in explicit time integration
  lump_mass = settings->sublist("Solver").get<bool>("lump mass",false);
  matrix_free = settings->sublist("Solver").get<bool>("matrix free",false);
  
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("use measurements as DBCs", false);
  
  assembly_partitioning = settings->sublist("Solver").get<string>("assembly partitioning","sequential");
  
  if (settings->isSublist("Subgrid")) {
    assembly_partitioning = "subgrid-preserving";
  }
  
  string solver_type = settings->sublist("Solver").get<string>("solver","none"); // or "transient"
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  
  // needed information from the mesh
  mesh->getElementBlockNames(blocknames);
  
  // check if we need to assembly volumetric, boundary and face terms
  for (size_t set=0; set<phys->setnames.size(); ++set) {
    vector<bool> set_assemble_vol, set_assemble_bndry, set_assemble_face;
    for (size_t b=0; b<blocknames.size(); b++) {
      set_assemble_vol.push_back(phys->setPhysSettings[set][b].template get<bool>("assemble volume terms",true));
      set_assemble_bndry.push_back(phys->setPhysSettings[set][b].template get<bool>("assemble boundary terms",true));
      set_assemble_face.push_back(phys->setPhysSettings[set][b].template get<bool>("assemble face terms",false));
    }
    assemble_volume_terms.push_back(set_assemble_vol);
    assemble_boundary_terms.push_back(set_assemble_bndry);
    assemble_face_terms.push_back(set_assemble_face);
  }
  // overwrite assemble_face_terms if HFACE vars are used
  for (size_t set=0; set<assemble_face_terms.size(); ++set) {
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<string> ctypes = phys->unique_types[b];
      for (size_t n=0; n<ctypes.size(); n++) {
        if (ctypes[n] == "HFACE") {
          assemble_face_terms[set][b] = true;
        }
      }
    }
  }
  
  // determine if we need to build basis functions
  for (size_t b=0; b<blocknames.size(); b++) {
    bool build_volume = false, build_bndry = false, build_face = false;
  
    for (size_t set=0; set<phys->setnames.size(); ++set) {
      
      if (assemble_volume_terms[set][b]) {
        build_volume = true;
      }
      else if (phys->setPhysSettings[set][b].template get<bool>("build volume terms",true) ) {
        build_volume = true;
      }
      
      if (assemble_boundary_terms[set][b]) {
        build_bndry = true;
      }
      else if (phys->setPhysSettings[set][b].template get<bool>("build boundary terms",true)) {
        build_bndry = true;
      }
      
      if (assemble_face_terms[set][b]) {
        build_face = true;
      }
      else if (phys->setPhysSettings[set][b].template get<bool>("build face terms",false)) {
        build_face = true;
      }
    }
    build_volume_terms.push_back(build_volume);
    build_boundary_terms.push_back(build_bndry);
    build_face_terms.push_back(build_face);
  }
  
  // needed information from the physics interface
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
// Create the fixed DOFs
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createFixedDOFs() {

  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createFixedDOFs ... " << endl;
    }
  }
  
  // create fixedDOF View of bools
  vector<vector<vector<vector<LO> > > > dbc_dofs = disc->dbc_dofs; // [set][block][var][dof]
  for (size_t set=0; set<dbc_dofs.size(); ++set) {
    vector<vector<Kokkos::View<LO*,LA_device> > > set_fixedDOF;
    
    int numLocalDof = disc->DOF[set]->getNumOwnedAndGhosted();
    Kokkos::View<bool*,LA_device> set_isFixedDOF("logicals for fixed DOFs",numLocalDof);
    auto fixed_host = Kokkos::create_mirror_view(set_isFixedDOF);
    for (size_t block=0; block<dbc_dofs[set].size(); block++) {
      for (size_t var=0; var<dbc_dofs[set][block].size(); var++) {
        for (size_t i=0; i<dbc_dofs[set][block][var].size(); i++) {
          LO dof = dbc_dofs[set][block][var][i];
          fixed_host(dof) = true;
        }
      }
    }
    Kokkos::deep_copy(set_isFixedDOF,fixed_host);
    isFixedDOF.push_back(set_isFixedDOF);
    
    for (size_t block=0; block<dbc_dofs[set].size(); block++) {
      vector<Kokkos::View<LO*,LA_device> > block_dofs;
      for (size_t var=0; var<dbc_dofs[set][block].size(); var++) {
        Kokkos::View<LO*,LA_device> cfixed;
        if (dbc_dofs[set][block][var].size()>0) {
          cfixed = Kokkos::View<LO*,LA_device>("fixed DOFs",dbc_dofs[set][block][var].size());
          auto cfixed_host = Kokkos::create_mirror_view(cfixed);
          for (size_t i=0; i<dbc_dofs[set][block][var].size(); i++) {
            LO dof = dbc_dofs[set][block][var][i];
            cfixed_host(i) = dof;
          }
          Kokkos::deep_copy(cfixed,cfixed_host);
        }
        block_dofs.push_back(cfixed);
      }
      set_fixedDOF.push_back(block_dofs);
    }
    fixedDOF.push_back(set_fixedDOF);
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
  
  bool storeAll = settings->sublist("Solver").get<bool>("store all cell data",true);
  double storageProportion = 1.0;
  if (!storeAll) {
    storageProportion = settings->sublist("Solver").get<double>("storage proportion",0.0);
  }
  
  vector<stk::mesh::Entity> all_meshElems;
  mesh->getMyElements(all_meshElems);
  
  vector<Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device> > LIDs;
  for (size_t set=0; set<disc->DOF.size(); ++set) {
    Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device> setLIDs = disc->DOF[set]->getLIDs();
    LIDs.push_back(setLIDs);
  }
   
  // Disc manager stores offsets as [set][block][var][dof]
  vector<vector<vector<vector<int> > > > disc_offsets = disc->offsets;
  
  // We want these re-ordered as [block][set][var][dof]
  vector<vector<vector<vector<int> > > > my_offsets;
  for (size_t b=0; b<blocknames.size(); ++b) {
    vector<vector<vector<int> > > block_offsets;
    for (size_t set=0; set<disc_offsets.size(); ++set) {
      block_offsets.push_back(disc_offsets[set][b]);
    }
    my_offsets.push_back(block_offsets);
  }
  
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
    LO processedElem = 0;
    
    if (numTotalElem>0) {
      
      vector<size_t> localIds;
      
      Kokkos::DynRankView<ScalarT,HostDevice> blocknodes;
      panzer_stk::workset_utils::getIdsAndVertices(*mesh, blocknames[b], localIds, blocknodes); // fill on host
      
      vector<size_t> myElem = disc->myElements[b];
      Kokkos::View<LO*,AssemblyDevice> eIDs("local element IDs on device",myElem.size());
      auto host_eIDs = Kokkos::create_mirror_view(eIDs);
      for (size_t elem=0; elem<myElem.size(); elem++) {
        host_eIDs(elem) = static_cast<LO>(myElem[elem]);
      }
      Kokkos::deep_copy(eIDs, host_eIDs);
      
      // LO is int, but just in case that changes ...
      LO elemPerCell = static_cast<LO>(settings->sublist("Solver").get<int>("workset size",100));
      if (elemPerCell == -1) {
        elemPerCell = numTotalElem;
      }
      else {
        elemPerCell = std::min(elemPerCell,numTotalElem);
      }
      
      vector<string> sideSets;
      mesh->getSidesetNames(sideSets);
      vector<bool> aface;
      for (size_t set=0; set<assemble_face_terms.size(); ++set) {
        aface.push_back(assemble_face_terms[set][b]);
      }
      blockCellData = Teuchos::rcp( new CellMetaData(settings, cellTopo,
                                                     phys, b, 0, elemPerCell,
                                                     build_face_terms[b],
                                                     aface,
                                                     sideSets,
                                                     params->num_discretized_params));
                      
      disc->setReferenceData(blockCellData);
      
      blockCellData->requireBasisAtNodes = settings->sublist("Postprocess").get<bool>("plot solution at nodes",false);
      
      
      vector<vector<vector<int> > > curroffsets = my_offsets[b];
      vector<Kokkos::View<LO*,AssemblyDevice> > set_numDOF;
      vector<Kokkos::View<LO*,HostDevice> > set_numDOF_host;
      
      for (size_t set=0; set<curroffsets.size(); ++set) {
        Kokkos::View<LO*,AssemblyDevice> numDOF_KV("number of DOF per variable",curroffsets[set].size());
        Kokkos::View<LO*,HostDevice> numDOF_host("numDOF on host",curroffsets[set].size());
        for (size_t k=0; k<curroffsets[set].size(); k++) {
          numDOF_host(k) = static_cast<LO>(curroffsets[set][k].size());
        }
        Kokkos::deep_copy(numDOF_KV, numDOF_host);
        set_numDOF.push_back(numDOF_KV);
        set_numDOF_host.push_back(numDOF_host);
      }
      
      blockCellData->set_numDOF = set_numDOF;
      blockCellData->set_numDOF_host = set_numDOF_host;
           
      blockCellData->numDOF = set_numDOF[0];
      blockCellData->numDOF_host = set_numDOF_host[0];
      
      //////////////////////////////////////////////////////////////////////////////////
      // Boundary cells
      //////////////////////////////////////////////////////////////////////////////////
      
      if (build_boundary_terms[b]) {
        
        int numBoundaryElem = elemPerCell;
        
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
                
                // Build the Kokkos View of the cell LIDs ------
                vector<LIDView> set_LIDs;
                for (size_t set=0; set<LIDs.size(); ++set) {
                  LIDView cellLIDs("LIDs",currElem,LIDs[set].extent(1));
                  auto currLIDs = LIDs[set];
                  parallel_for("assembly copy LIDs bcell",
                               RangePolicy<AssemblyExec>(0,cellLIDs.extent(0)),
                               KOKKOS_LAMBDA (const int e ) {
                    size_t elemID = eIndex(e);
                    for (size_type j=0; j<currLIDs.extent(1); j++) {
                      cellLIDs(e,j) = currLIDs(elemID,j);
                    }
                  });
                  set_LIDs.push_back(cellLIDs);
                }
                
                //-----------------------------------------------
                // Set the side information (soon to be removed)-
                vector<Kokkos::View<int****,HostDevice> > set_sideinfo;
                for (size_t set=0; set<LIDs.size(); ++set) {
                  Kokkos::View<int****,HostDevice> sideinfo = disc->getSideInfo(set,b,host_eIndex2);
                  set_sideinfo.push_back(sideinfo);
                }
                
                bcells.push_back(Teuchos::rcp(new BoundaryCell(blockCellData, currnodes, eIndex, sideIndex,
                                                               side, sideName, bcells.size(),
                                                               disc, storeAll)));
                bcells[bcells.size()-1]->LIDs = set_LIDs;
                bcells[bcells.size()-1]->createHostLIDs();
                bcells[bcells.size()-1]->sideinfo = set_sideinfo;
                prog += currElem;
              }
            }
          }
        }
      }
      
      //////////////////////////////////////////////////////////////////////////////////
      // Cells
      //////////////////////////////////////////////////////////////////////////////////
      
      LO prog = 0;
      vector<vector<LO> > groups;
      
      if (assembly_partitioning == "sequential") {
        while (prog < numTotalElem) {
          
          vector<LO> newgroup;
          
          LO currElem = elemPerCell;
          if (prog+currElem > numTotalElem){
            currElem = numTotalElem-prog;
          }
          for (LO e=prog; e<prog+currElem; ++e) {
            newgroup.push_back(e);
          }
          groups.push_back(newgroup);
          prog += currElem;
        }
      }
      else if (assembly_partitioning == "random") { // not implemented yet
        
      }
      else if (assembly_partitioning == "neighbor-avoiding") { // not implemented yet
        // need neighbor information
      }
      else if (assembly_partitioning == "subgrid-preserving") {
        
        ///////////////////////////////////////////////////////////////////////////////////
        // Rules for subgrid-preserving grouping
        //
        // 1.  All elements must be on the same processor
        // 2.  All elements must either be interior, or
        // 3.  All elements must have the same boundary edges/faces (this is the key difference)
        // 4.  No more than elemPerCell (= numElem) in a group
        ///////////////////////////////////////////////////////////////////////////////////
        
        if (bcells.size() > 0) {
          Kokkos::View<bool*> beenadded("been processed",numTotalElem);
          deep_copy(beenadded,false);
          
          Kokkos::View<bool**> onbndry("onbndry",numTotalElem,bcells.size());
          deep_copy(onbndry,false);
          
          for (size_t bc=0; bc<bcells.size(); ++bc) {
            auto eind = create_mirror_view(bcells[bc]->localElemID);
            deep_copy(eind,bcells[bc]->localElemID);
            
            for (size_type e=0; e<eind.extent(0); ++e) {
              onbndry(eind(e),bc) = true;
            }
          }
          
          
          LO numAdded=0;
          while (numAdded < numTotalElem) {
            vector<LO> newgroup;
            bool foundind = false;
            LO refind = 0;
            while (!foundind && refind<numTotalElem) {
              if (!beenadded(refind)) {
                foundind = true;
              }
              else {
                refind++;
              }
            }
            newgroup.push_back(refind);
            beenadded(refind) = true;
            numAdded++;
            for (LO j=refind+1; j<numTotalElem; ++j) {
              bool matches = true;
              for (size_type k=0; k<onbndry.extent(1); ++k) {
                if (onbndry(j,k) != onbndry(refind,k)) {
                  matches = false;
                }
              }
              if (matches && static_cast<LO>(newgroup.size()) < elemPerCell) {
                newgroup.push_back(j);
                beenadded(j) = true;
                numAdded++;
              }
            }
            groups.push_back(newgroup);
          }
          
          // Re-order the groups from biggest to smallest
          // This is needed for certain parts of MrHyDE that assume the first
          // cell contains the most elements
          for (size_t grp=0; grp<groups.size()-1; ++grp) {
            size_t mxgrp_ind = grp;
            size_t mxgrp = groups[grp].size();
            bool perform_swap = false;
            for (size_t grp2=grp+1; grp2<groups.size(); ++grp2) {
              if (groups[grp2].size() > mxgrp) {
                mxgrp = groups[grp2].size();
                mxgrp_ind = grp2;
                perform_swap = true;
              }
            }
            if (perform_swap) {
              groups[grp].swap(groups[mxgrp_ind]);
            }
          }
        }
        else {
          while (prog < numTotalElem) {
            
            vector<LO> newgroup;
            
            LO currElem = elemPerCell;
            if (prog+currElem > numTotalElem){
              currElem = numTotalElem-prog;
            }
            for (LO e=prog; e<prog+currElem; ++e) {
              newgroup.push_back(e);
            }
            groups.push_back(newgroup);
            prog += currElem;
          }
        }
        
      }
      
      elemPerCell = std::min(elemPerCell, static_cast<LO>(groups[0].size()));
      
      // Add the cells correspondng to the groups
      for (size_t grp=0; grp<groups.size(); ++grp) {
        LO currElem = groups[grp].size();
        
        bool storeThis = storeAll;
        if (!storeAll) {
          if (static_cast<double>(processedElem)/static_cast<double>(numTotalElem) < storageProportion) {
            storeThis = true;
          }
        }
        processedElem += currElem;
        
        Kokkos::View<LO*,AssemblyDevice> eIndex("element indices",currElem);
        DRV currnodes("currnodes", currElem, numNodesPerElem, spaceDim);
        
        auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
        Kokkos::View<LO*,HostDevice> host_eIndex2("element indices on host",currElem);
        
        for (LO e=0; e<currElem; ++e) {
          host_eIndex(e) = groups[grp][e];
        }
        Kokkos::deep_copy(eIndex,host_eIndex);
        Kokkos::deep_copy(host_eIndex2,host_eIndex);
        
        vector<LIDView> set_LIDs;
        for (size_t set=0; set<LIDs.size(); ++set) {
          LIDView cellLIDs("LIDs on device",currElem,LIDs[set].extent(1));
          auto currLIDs = LIDs[set];
          parallel_for("assembly copy nodes",
                       RangePolicy<AssemblyExec>(0,eIndex.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            LO elemID = eIndex(e);
            for (size_type j=0; j<currLIDs.extent(1); j++) {
              cellLIDs(e,j) = currLIDs(eIDs(elemID),j);
            }
          });
          set_LIDs.push_back(cellLIDs);
        }
        
        parallel_for("assembly copy nodes",
                     RangePolicy<AssemblyExec>(0,eIndex.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          LO elemID = eIndex(e);
          for (size_type pt=0; pt<currnodes.extent(1); ++pt) {
            for (size_type dim=0; dim<currnodes.extent(2); ++dim) {
              currnodes(e,pt,dim) = blocknodes(elemID,pt,dim);
            }
          }
        });
        
        // Set the side information (soon to be removed)-
        vector<Kokkos::View<int****,HostDevice> > set_sideinfo;
        for (size_t set=0; set<LIDs.size(); ++set) {
          Kokkos::View<int****,HostDevice> sideinfo = disc->getSideInfo(set,b,host_eIndex2);
          set_sideinfo.push_back(sideinfo);
        }
        
        blockcells.push_back(Teuchos::rcp(new cell(blockCellData, currnodes, eIndex,
                                                   disc, storeThis)));
        
        blockcells[blockcells.size()-1]->LIDs = set_LIDs;
        blockcells[blockcells.size()-1]->createHostLIDs();
        blockcells[blockcells.size()-1]->sideinfo = set_sideinfo;
        
        prog += elemPerCell;
        
      }
    }
    
    else {
      blockCellData = Teuchos::rcp( new CellMetaData());
    }
    
    cellData.push_back(blockCellData);
    cells.push_back(blockcells);
    boundaryCells.push_back(bcells);
    
  }
}

// =======================================================
// Have the cells compute and store the basis functions
// at the quadrature points (if storage is turned on)
// =======================================================

template<class Node>
void AssemblyManager<Node>::allocateCellStorage() {
  
  bool keepnodes = false;
  // There are a few scenarios where we want the cells to keep their nodes
  if (settings->sublist("Solver").get<string>("initial type","L2-projection") == "interpolation") {
    keepnodes = true;
  }
  if (settings->isSublist("Subgrid")) {
    keepnodes = true;
  }
  if (settings->sublist("Solver").get<bool>("keep nodes",false)) {
    keepnodes = true;
  }
  
  for (size_t b=0; b<cells.size(); ++b) {
    for (size_t c=0; c<cells[b].size(); ++c) {
      cells[b][c]->computeBasis(keepnodes);
    }
  }
  
  for (size_t b=0; b<boundaryCells.size(); ++b) {
    for (size_t c=0; c<boundaryCells[b].size(); ++c) {
      boundaryCells[b][c]->computeBasis(keepnodes);
    }
  }
  
  
  // ==============================================
  // Inform the user how many cells/bcells are on
  // each processor and much memory is utilized by
  // the cells/bcells
  // ==============================================
  
  if (verbosity > 5) {
    
    // Volumetric elements
    size_t numelements = 0;
    double minsize = 1e100;
    double maxsize = 0.0;
    for (size_t b=0; b<cells.size(); ++b) {
      for (size_t c=0; c<cells[b].size(); ++c) {
        numelements += cells[b][c]->numElem;
        if (cells[b][c]->storeAll) {
          auto wts = cells[b][c]->wts;
          auto host_wts = create_mirror_view(wts);
          deep_copy(host_wts,wts);
          for (size_type e=0; e<host_wts.extent(0); ++e) {
            double currsize = 0.0;
            for (size_type pt=0; pt<host_wts.extent(1); ++pt) {
              currsize += host_wts(e,pt);
            }
            maxsize = std::max(currsize,maxsize);
            minsize = std::min(currsize,minsize);
          }
        }
      }
    }
    cout << " - Processor " << Comm->getRank() << " has " << numelements << " elements" << endl;
    cout << " - Processor " << Comm->getRank() << " min element size: " << minsize << endl;
    cout << " - Processor " << Comm->getRank() << " max element size: " << maxsize << endl;
    
    // Boundary elements
    size_t numbndryelements = 0;
    double minbsize = 1e100;
    double maxbsize = 0.0;
    for (size_t b=0; b<boundaryCells.size(); ++b) {
      for (size_t c=0; c<boundaryCells[b].size(); ++c) {
        numbndryelements += boundaryCells[b][c]->numElem;
        if (boundaryCells[b][c]->storeAll) {
          auto wts = boundaryCells[b][c]->wts;
          auto host_wts = create_mirror_view(wts);
          deep_copy(host_wts,wts);
          for (size_type e=0; e<host_wts.extent(0); ++e) {
            double currsize = 0.0;
            for (size_type pt=0; pt<host_wts.extent(1); ++pt) {
              currsize += host_wts(e,pt);
            }
            maxbsize = std::max(currsize,maxbsize);
            minbsize = std::min(currsize,minbsize);
          }
        }
      }
    }
    cout << " - Processor " << Comm->getRank() << " has " << numbndryelements << " boundary elements" << endl;
    cout << " - Processor " << Comm->getRank() << " min boundary element size: " << minbsize << endl;
    cout << " - Processor " << Comm->getRank() << " max boundary element size: " << maxbsize << endl;
    
    // Volumetric ip/basis
    size_t cellstorage = 0;
    for (size_t b=0; b<cells.size(); ++b) {
      for (size_t c=0; c<cells[b].size(); ++c) {
        cellstorage += cells[b][c]->getVolumetricStorage();
      }
    }
    double totalstorage = static_cast<double>(cellstorage)/1.0e6;
    cout << " - Processor " << Comm->getRank() << " is using " << totalstorage << " MB to store cell volumetric data" << endl;
    
    // Face ip/basis
    size_t facestorage = 0;
    for (size_t b=0; b<cells.size(); ++b) {
      for (size_t c=0; c<cells[b].size(); ++c) {
        facestorage += cells[b][c]->getFaceStorage();
      }
    }
    totalstorage = static_cast<double>(facestorage)/1.0e6;
    cout << " - Processor " << Comm->getRank() << " is using " << totalstorage << " MB to store cell face data" << endl;
    
    // Boundary ip/basis
    size_t boundarystorage = 0;
    for (size_t b=0; b<boundaryCells.size(); ++b) {
      for (size_t c=0; c<boundaryCells[b].size(); ++c) {
        boundarystorage += boundaryCells[b][c]->getStorage();
      }
    }
    totalstorage = static_cast<double>(boundarystorage)/1.0e6;
    cout << " - Processor " << Comm->getRank() << " is using " << totalstorage << " MB to store boundary cell data" << endl;
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
      info.push_back((int)cellData[b]->numDiscParams);
      info.push_back(cellData[b]->numElem);
      info.push_back(cellData[b]->numip);
      info.push_back(cellData[b]->numsideip);
      info.push_back(phys->setnames.size());
      vector<size_t> numVars;
      for (size_t set=0; set<cellData[b]->set_numDOF.size(); ++set) {
        numVars.push_back(cellData[b]->set_numDOF[set].extent(0));
      }
      vector<Kokkos::View<string**,HostDevice> > bcs(phys->setnames.size());
      for (size_t set=0; set<phys->setnames.size(); ++set) {
        Kokkos::View<string**,HostDevice> vbcs = disc->getVarBCs(set,b);
        bcs[set] = vbcs;
      }
      wkset.push_back(Teuchos::rcp( new workset(info,
                                                numVars,
                                                isTransient,
                                                disc->basis_types[b],
                                                disc->basis_pointers[b],
                                                params->discretized_param_basis,
                                                mesh->getCellTopology(blocknames[b])) ) );
      
      wkset[b]->block = b;
      wkset[b]->set_var_bcs = bcs;
      wkset[b]->var_bcs = bcs[0];
    }
    else {
      wkset.push_back(Teuchos::rcp( new workset()));
      wkset[b]->isInitialized = false;
      wkset[b]->block = b;
    }
  }
  
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
void AssemblyManager<Node>::setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                                       const bool & lumpmass, const ScalarT & scale) {
  
  Teuchos::TimeMonitor localtimer(*setinittimer);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::setInitial ..." << endl;
    }
  }
  
  for (size_t block=0; block<cells.size(); block++) {
    this->setInitial(set,rhs,mass,useadjoint,lumpmass,scale,block,block);
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
void AssemblyManager<Node>::setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                                       const bool & lumpmass, const ScalarT & scale,
                                       const size_t & block, const size_t & cellblock) {

  typedef typename Node::execution_space LA_exec;
  using namespace std;
  
  bool use_atomics_ = false;
  if (LA_exec::concurrency() > 1) {
    use_atomics_ = true;
  }
  
  bool fix_zero_rows = true;
  
  auto localMatrix = mass->getLocalMatrix();
  auto rhs_view = rhs->template getLocalView<LA_device>();
  bool lump_mass_ = lump_mass;

  wkset[block]->updatePhysicsSet(set);
  cellData[block]->updatePhysicsSet(set);
  
  auto offsets = wkset[block]->offsets;
  auto numDOF = cellData[block]->numDOF;

  for (size_t e=0; e<cells[cellblock].size(); e++) {

    auto LIDs = cells[cellblock][e]->LIDs[set];
    
    auto localrhs = cells[cellblock][e]->getInitial(true, useadjoint);
    auto localmass = cells[cellblock][e]->getMass();

    parallel_for("assembly insert Jac",
                 RangePolicy<LA_exec>(0,LIDs.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      
      int row = 0;
      LO rowIndex = 0;
      
      for (size_type n=0; n<numDOF.extent(0); ++n) {
        for (int j=0; j<numDOF(n); j++) {
          row = offsets(n,j);
          rowIndex = LIDs(elem,row);
          ScalarT val = localrhs(elem,row);
          if (use_atomics_) {
            Kokkos::atomic_add(&(rhs_view(rowIndex,0)), val);
          }
          else {
            rhs_view(rowIndex,0) += val;
          }
        }
      }
      
      const size_type numVals = LIDs.extent(1);
      int col = 0;
      LO cols[maxDerivs];
      ScalarT vals[maxDerivs];
      for (size_type n=0; n<numDOF.extent(0); ++n) {
        for (int j=0; j<numDOF(n); j++) {
          row = offsets(n,j);
          rowIndex = LIDs(elem,row);
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              col = offsets(m,k);
              vals[col] = localmass(elem,row,col);
              if (lump_mass_) {
                cols[col] = rowIndex;
              }
              else {
                cols[col] = LIDs(elem,col);
              }
            }
          }
          localMatrix.sumIntoValues(rowIndex, cols, numVals, vals, false, use_atomics_);
        }
      }
    });
  }
  
  if (fix_zero_rows) {
    size_t numrows = mass->getNodeNumRows();
    
    parallel_for("assembly insert Jac",
                 RangePolicy<LA_exec>(0,numrows),
                 KOKKOS_LAMBDA (const size_t row ) {
      auto rowdata = localMatrix.row(row);
      ScalarT abssum = 0.0;
      for (int col=0; col<rowdata.length; ++col ) {
        abssum += abs(rowdata.value(col));
      }
      ScalarT val[1];
      LO cols[1];
      if (abssum<1.0e-14) { // needs to be generalized!
        val[0] = 1.0;
        cols[0] = row;
        localMatrix.replaceValues(row,cols,1,val,false,false);
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::getWeightedMass(const size_t & set,
                                            matrix_RCP & mass,
                                            vector_RCP & diagMass) {
  
  Teuchos::TimeMonitor localtimer(*setinittimer);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::getWeightedMass ..." << endl;
    }
  }
  
  typedef typename Node::execution_space LA_exec;
  bool use_atomics_ = false;
  if (LA_exec::concurrency() > 1) {
    use_atomics_ = true;
  }
  bool compute_matrix = true;
  if (lump_mass || matrix_free) {
    compute_matrix = false;
  }
  
  typedef typename Tpetra::CrsMatrix<ScalarT, LO, GO, Node >::local_matrix_type local_matrix;
  local_matrix localMatrix;
  
  if (compute_matrix) {
    localMatrix = mass->getLocalMatrix();
  }
  
  auto diag_view = diagMass->template getLocalView<LA_device>();
  
  for (size_t b=0; b<cells.size(); b++) {
    
    auto offsets = wkset[b]->offsets;
    auto numDOF = cellData[b]->numDOF;
    
    for (size_t e=0; e<cells[b].size(); e++) {
      
      auto LIDs = cells[b][e]->LIDs[set];
      
      Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[b][e]->getWeightedMass(phys->masswts[set][b]);
      
      parallel_for("assembly insert Jac",
                   RangePolicy<LA_exec>(0,LIDs.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        
        int row = 0;
        LO rowIndex = 0;
        
        for (size_type n=0; n<numDOF.extent(0); ++n) {
          for (int j=0; j<numDOF(n); j++) {
            row = offsets(n,j);
            rowIndex = LIDs(elem,row);
            ScalarT val = 0.0;
            
            for (int k=0; k<numDOF(n); k++) {
              int col = offsets(n,k);
              val += localmass(elem,row,col);
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
      
      if (compute_matrix) {
        parallel_for("assembly insert Jac",
                     RangePolicy<LA_exec>(0,LIDs.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          
          int row = 0;
          LO rowIndex = 0;
          
          int col = 0;
          LO cols[64];
          ScalarT vals[64];
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
  }
  
  if (compute_matrix) {
    mass->fillComplete();
  }
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::getWeightedMass ..." << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::applyMassMatrixFree(const size_t & set, vector_RCP & x, vector_RCP & y) {
  
  auto x_kv = x->template getLocalView<LA_device>();
  auto x_slice = Kokkos::subview(x_kv, Kokkos::ALL(), 0);
  
  auto y_kv = y->template getLocalView<LA_device>();
  auto y_slice = Kokkos::subview(y_kv, Kokkos::ALL(), 0);
  
  Kokkos::View<LO*,AssemblyDevice> numDOF;
  View_Sc2 x_local, y_local;
  Kokkos::View<int**,AssemblyDevice> offsets;
  LIDView LIDs;
  
  for (size_t b=0; b<cells.size(); b++) {
    offsets = wkset[b]->offsets;
    numDOF = cellData[b]->numDOF;
    int maxdof = 0;
    for (size_type i=0; i<cellData[b]->set_numDOF_host[set].extent(0); i++) {
      maxdof += cellData[b]->set_numDOF_host[set](i);
    }
    x_local = View_Sc2("local x values",wkset[b]->maxElem, maxdof);
    y_local = View_Sc2("local y values",wkset[b]->maxElem, maxdof);
    
    for (size_t c=0; c<cells[b].size(); c++) {
      deep_copy(y_local,0.0);
      LIDs = cells[b][c]->LIDs[set];
      parallel_for("assembly gather",
                   RangePolicy<AssemblyExec>(0,LIDs.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type var=0; var<offsets.extent(0); var++) {
          for(int dof=0; dof<numDOF(var); dof++ ) {
            x_local(elem,offsets(var,dof)) = x_slice(LIDs(elem,offsets(var,dof)));
            //y_local(elem,var,dof) = y_slice(LIDs(elem,offsets(var,dof)));
          }
        }
      });
      
      cells[b][c]->applyLocalMass(set, phys->masswts[set][b], x_local, y_local);
      
    //  KokkosTools::print(y_local);
      
      parallel_for("assembly gather",
                   RangePolicy<AssemblyExec>(0,LIDs.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type var=0; var<offsets.extent(0); var++) {
          for(int dof=0; dof<numDOF(var); dof++ ) {
            y_slice(LIDs(elem,offsets(var,dof))) += y_local(elem,offsets(var,dof));
          }
        }
      });
    }
  }
  
}


// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::getWeightVector(const size_t & set, vector_RCP & wts) {
  
  Teuchos::TimeMonitor localtimer(*setinittimer);
  
  typedef typename Node::execution_space LA_exec;
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::getWeightVector ..." << endl;
    }
  }
  
  auto wts_view = wts->template getLocalView<LA_device>();
  
  vector<vector<ScalarT> > normwts = phys->normwts[set];
  
  for (size_t b=0; b<cells.size(); b++) {
    
    auto offsets = wkset[b]->offsets;
    auto numDOF = cellData[b]->numDOF;
    
    for (size_t e=0; e<cells[b].size(); e++) {
      
      for (size_type n=0; n<numDOF.extent(0); ++n) {
      
        ScalarT val = normwts[b][n];
        auto LIDs = cells[b][e]->LIDs[set];
        parallel_for("assembly insert Jac",
                     RangePolicy<LA_exec>(0,LIDs.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          
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
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::getWeightVector ..." << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::setInitial(const size_t & set, vector_RCP & initial, const bool & useadjoint) {

  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      LIDView_host LIDs = cells[b][e]->LIDs_host[set];
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
void AssemblyManager<Node>::setDirichlet(const size_t & set, vector_RCP & rhs, matrix_RCP & mass,
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
      auto LIDs = boundaryCells[b][e]->LIDs_host[set];
      auto localrhs = boundaryCells[b][e]->getDirichlet(set);
      auto localmass = boundaryCells[b][e]->getMass(set);
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
      auto LIDs = cells[b][e]->LIDs_host[set];
      for (size_t c=0; c<cells[b][e]->numElem; c++) {
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
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::setDirichlet ..." << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::setInitialFace(const size_t & set, vector_RCP & rhs, matrix_RCP & mass,
                                           const bool & lumpmass) {
  
//  // TODO TIMERS BROKEN
//  //Teuchos::TimeMonitor localtimer(*setdbctimer);
//
  
  using namespace std;
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::setInitialFace ..." << endl;
    }
  }
  
  auto localMatrix = mass->getLocalMatrix();
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      int numElem = cells[b][e]->numElem;
      auto LIDs = cells[b][e]->LIDs_host[set];
      // Get the requested IC from the cell
      auto localrhs = cells[b][e]->getInitialFace(true);
      // Create the mass matrix
      auto localmass = cells[b][e]->getMassFace();
      auto host_rhs = Kokkos::create_mirror_view(localrhs);
      auto host_mass = Kokkos::create_mirror_view(localmass);
      Kokkos::deep_copy(host_rhs,localrhs);
      Kokkos::deep_copy(host_mass,localmass);
      
      size_t numVals = LIDs.extent(1);
      // assemble into global matrix
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<LIDs.extent(1); row++ ) {
          LO rowIndex = LIDs(c,row);
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

  // make sure we don't have any rows of all zeroes
  // TODO I don't think this can ever happen?
  // at least globally
  
  typedef typename Node::execution_space LA_exec;
  size_t numrows = mass->getNodeNumRows();
  
  parallel_for("assembly insert Jac",
               RangePolicy<LA_exec>(0,numrows),
               KOKKOS_LAMBDA (const size_t row ) {
    auto rowdata = localMatrix.row(row);
    ScalarT abssum = 0.0;
    for (int col=0; col<rowdata.length; ++col ) {
      abssum += abs(rowdata.value(col));
    }
    ScalarT val[1];
    LO cols[1];
    if (abssum<1.0e-14) { // needs to be generalized!
      val[0] = 1.0;
      cols[0] = row;
      localMatrix.replaceValues(row,cols,1,val,false,false);
    }
  });

  mass->fillComplete();
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::setInitialFace ..." << endl;
    }
  }
  
}

// ========================================================================================
// Wrapper to the main assembly routine to assemble over all blocks (most common use case)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::assembleJacRes(const size_t & set, vector_RCP & u, vector_RCP & phi,
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
    this->performGather(set, u, 0, 0);
    if (params->num_discretized_params > 0) {
      this->performGather(set, Psol, 4, 0);
    }
    if (useadjoint) {
      this->performGather(set, phi, 2, 0);
    }
  }

  
  for (size_t b=0; b<cells.size(); b++) {
    if (cells[b].size() > 0) {
      this->assembleJacRes(set, compute_jacobian,
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
void AssemblyManager<Node>::assembleJacRes(const size_t & set, const bool & compute_jacobian, const bool & compute_sens,
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
  typedef typename Tpetra::CrsMatrix<ScalarT, LO, GO, Node >::local_matrix_type local_matrix;
  local_matrix J_kcrs;
  if (compute_jacobian) {
    J_kcrs = J->getLocalMatrix();
  }
  
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
  if (!data_avail || useadjoint || cellData[b]->multiscale || compute_disc_sens || compute_sens) {
    reduce_memory = false;
  }
  
  // Set the seeding flag for AD objects
  int seedwhat = 0;
  if (compute_jacobian) {
    if (compute_disc_sens) {
      seedwhat = 3;
    }
    else {
      seedwhat = 1;
    }
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
  
  int numElem = cellData[b]->numElem;
  int numDOF = cells[b][0]->LIDs[set].extent(1);
  
  int numParamDOF = 0;
  if (compute_disc_sens) {
    numParamDOF = cells[b][0]->paramLIDs.extent(1);
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
  
  // Note: Cannot parallelize over cells since data structures are re-used
  
  for (size_t e=0; e<cells[b].size(); e++) {

    wkset[b]->localEID = e;
    
    if (isTransient && useadjoint && !cells[b][0]->cellData->multiscale) {
      if (is_final_time) {
        cells[b][e]->resetAdjPrev(set,0.0);
      }
    }
 
    /////////////////////////////////////////////////////////////////////////////
    // Compute the local residual and Jacobian on this cell
    /////////////////////////////////////////////////////////////////////////////
    
    bool fixJacDiag = false;
    
    {
      Teuchos::TimeMonitor localtimer(*phystimer);
      
      //////////////////////////////////////////////////////////////
      // Compute res and J=dF/du
      //////////////////////////////////////////////////////////////
      
      // Volumetric contribution
      if (assemble_volume_terms[set][b]) {
        if (cellData[b]->multiscale) {
          int sgindex = cells[b][e]->subgrid_model_index[cells[b][e]->subgrid_model_index.size()-1];
          cells[b][e]->subgridModels[sgindex]->subgridSolver(cells[b][e]->u[set], cells[b][e]->phi[set], wkset[b]->time, isTransient, useadjoint,
                                                             compute_jacobian, compute_sens, num_active_params,
                                                             compute_disc_sens, false,
                                                             *(wkset[b]), cells[b][e]->subgrid_usernum, 0,
                                                             cells[b][e]->subgradient, store_adjPrev);
          fixJacDiag = true;
        }
        else {
          cells[b][e]->updateWorkset(seedwhat);
          phys->volumeResidual(set,b);
        }
      }
      
      ///////////////////////////////////////////////////////////////////////////
      // Edge/face contribution
      ///////////////////////////////////////////////////////////////////////////
      
      if (assemble_face_terms[set][b]) {
        if (cellData[b]->multiscale) {
          // do nothing
        }
        else {
          for (size_t s=0; s<cellData[b]->numSides; s++) {
            cells[b][e]->updateWorksetFace(s);
            phys->faceResidual(set,b);
          }
        }
      }
      
    }
        
    ///////////////////////////////////////////////////////////////////////////
    // Scatter into global matrix/vector
    ///////////////////////////////////////////////////////////////////////////
    
    if (reduce_memory) { // skip local_res and local_J
      this->scatter(set, J_kcrs, res_view,
                    cells[b][e]->LIDs[set], cells[b][e]->paramLIDs, b,
                    compute_jacobian, compute_sens, compute_disc_sens, useadjoint);
    }
    else { // fill local_res and local_J and then scatter
    
      Teuchos::TimeMonitor localtimer(*scattertimer);
      
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
        cells[b][e]->updateAdjointRes(compute_jacobian, isTransient,
                                      false, store_adjPrev,
                                      local_J, local_res);
      }
      else {
        cells[b][e]->updateRes(compute_sens, local_res);
      }
      
      // Now scatter from local_res and local_J
      
      if (data_avail) {
        this->scatterRes(res_view, local_res, cells[b][e]->LIDs[set]);
        if (compute_jacobian) {
          this->scatterJac(set, J_kcrs, local_J, cells[b][e]->LIDs[set], cells[b][e]->paramLIDs, compute_disc_sens);
        }
      }
      else {
        auto local_res_ladev = create_mirror(LA_exec(),local_res);
        auto local_J_ladev = create_mirror(LA_exec(),local_J);
        
        Kokkos::deep_copy(local_J_ladev,local_J);
        Kokkos::deep_copy(local_res_ladev,local_res);
        
        if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
          this->scatterRes(res_view, local_res_ladev, cells[b][e]->LIDs_host[set]);
          if (compute_jacobian) {
            this->scatterJac(set, J_kcrs, local_J_ladev, cells[b][e]->LIDs_host[set], cells[b][e]->paramLIDs_host, compute_disc_sens);
          }
          
        }
        else { // LA_device = CUDA, AssemblyDevice = Host
          // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
          auto LIDs_dev = Kokkos::create_mirror(LA_exec(), cells[b][e]->LIDs[set]);
          auto paramLIDs_dev = Kokkos::create_mirror(LA_exec(), cells[b][e]->paramLIDs);
          Kokkos::deep_copy(LIDs_dev,cells[b][e]->LIDs[set]);
          Kokkos::deep_copy(paramLIDs_dev,cells[b][e]->paramLIDs);
          
          this->scatterRes(res_view, local_res_ladev, LIDs_dev);
          if (compute_jacobian) {
            this->scatterJac(set, J_kcrs, local_J_ladev, LIDs_dev, paramLIDs_dev, compute_disc_sens);
          }
        }
        
      }
    }
    
  } // cell loop
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Boundary terms
  //////////////////////////////////////////////////////////////////////////////////////
  
  if (!cellData[b]->multiscale && assemble_boundary_terms[set][b]) {
    
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
        
        /////////////////////////////////////////////////////////////////////////////
        // Compute the local residual and Jacobian on this cell
        /////////////////////////////////////////////////////////////////////////////
        
        {
          Teuchos::TimeMonitor localtimer(*phystimer);
          wkset[b]->resetResidual();
          
          boundaryCells[b][e]->updateWorkset(seedwhat);
          phys->boundaryResidual(set,b);
          
        }
        
        ///////////////////////////////////////////////////////////////////////////
        // Scatter into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        if (reduce_memory) { // skip local_res and local_J
          this->scatter(set, J_kcrs, res_view,
                        boundaryCells[b][e]->LIDs[set], boundaryCells[b][e]->paramLIDs, b,
                        compute_jacobian, compute_sens, compute_disc_sens, useadjoint);
        }
        else { // fill local_res and local_J and then scatter
        
          Teuchos::TimeMonitor localtimer(*scattertimer);
          
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
          if (!useadjoint) {
            boundaryCells[b][e]->updateRes(compute_sens, local_res);
          }
         
          if (data_avail) {
            this->scatterRes(res_view, local_res, boundaryCells[b][e]->LIDs[set]);
            if (compute_jacobian) {
              this->scatterJac(set, J_kcrs, local_J, boundaryCells[b][e]->LIDs[set], boundaryCells[b][e]->paramLIDs, compute_disc_sens);
            }
          }
          else {
            Kokkos::deep_copy(local_J_ladev,local_J);
            Kokkos::deep_copy(local_res_ladev,local_res);
            
            if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
              this->scatterRes(res_view, local_res_ladev, boundaryCells[b][e]->LIDs_host[set]);
              if (compute_jacobian) {
                this->scatterJac(set, J_kcrs, local_J_ladev,
                                 boundaryCells[b][e]->LIDs_host[set], boundaryCells[b][e]->paramLIDs_host,
                                 compute_disc_sens);
              }
            }
            else { // LA_device = CUDA, AssemblyDevice = Host
              // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
              auto LIDs_dev = Kokkos::create_mirror(LA_exec(), boundaryCells[b][e]->LIDs[set]);
              auto paramLIDs_dev = Kokkos::create_mirror(LA_exec(), boundaryCells[b][e]->paramLIDs);
              Kokkos::deep_copy(LIDs_dev,boundaryCells[b][e]->LIDs[set]);
              Kokkos::deep_copy(paramLIDs_dev,boundaryCells[b][e]->paramLIDs);
              
              this->scatterRes(res_view, local_res_ladev, LIDs_dev);
              if (compute_jacobian) {
                this->scatterJac(set, J_kcrs, local_J_ladev, LIDs_dev, paramLIDs_dev, compute_disc_sens);
              }
            }
            
          }
        }
        
      }
    } // element loop
    
  }
  
  // Apply constraints, e.g., strongly imposed Dirichlet
  this->dofConstraints(set, J, res, current_time, compute_jacobian, compute_disc_sens);
  
  
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
void AssemblyManager<Node>::dofConstraints(const size_t & set, matrix_RCP & J, vector_RCP & res,
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
    vector<vector<vector<LO> > > dbcDOFs = disc->dbc_dofs[set];
    for (size_t block=0; block<dbcDOFs.size(); block++) {
      for (size_t var=0; var<dbcDOFs[block].size(); var++) {
        if (compute_jacobian) {
          this->updateJacDBC(J,dbcDOFs[block][var],compute_disc_sens);
        }
      }
    }
  }
  
  vector<vector<GO> > fixedDOFs = disc->point_dofs[set];
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
void AssemblyManager<Node>::resetPrevSoln(const size_t & set) {
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->resetPrevSoln(set);
    }
  }
}

template<class Node>
void AssemblyManager<Node>::resetStageSoln(const size_t & set) {
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->resetStageSoln(set);
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
void AssemblyManager<Node>::updateStageSoln(const size_t & set)  {
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->updateStageSoln(set);
    }
  }
}

// ========================================================================================
// Gather local solutions on cells.
// This intermediate function allows us to copy the data from LA_device to AssemblyDevice only once (if necessary)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::performGather(const size_t & set, const vector_RCP & vec, const int & type, const size_t & entry) {
  
  typedef typename LA_device::memory_space LA_mem;
  
  auto vec_kv = vec->template getLocalView<LA_device>();
  
  // Even if there are multiple vectors, we only use one at a time
  auto vec_slice = Kokkos::subview(vec_kv, Kokkos::ALL(), entry);
  
  // vector is on LA_device, but gather must happen on AssemblyDevice
  if (Kokkos::SpaceAccessibility<AssemblyExec, LA_mem>::accessible) { // can we avoid a copy?
    this->performGather(set, vec_slice, type);
    this->performBoundaryGather(set, vec_slice, type);
  }
  else { // apparently not
    auto vec_dev = Kokkos::create_mirror(AssemblyDevice::memory_space(),vec_slice);
    Kokkos::deep_copy(vec_dev,vec_slice);
    this->performGather(set, vec_dev, type);
    this->performBoundaryGather(set, vec_dev, type);
  }
  
}

// ========================================================================================
//
// ========================================================================================

template<class Node>
template<class ViewType>
void AssemblyManager<Node>::performGather(const size_t & set, ViewType vec_dev, const int & type) {

  Kokkos::View<LO*,AssemblyDevice> numDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> data;
  Kokkos::View<int**,AssemblyDevice> offsets;
  LIDView LIDs;
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t c=0; c<cells[b].size(); c++) {
      switch(type) {
        case 0 :
          LIDs = cells[b][c]->LIDs[set];
          numDOF = cells[b][c]->cellData->numDOF;
          data = cells[b][c]->u[set];
          offsets = wkset[b]->offsets;
          break;
        case 1 : // deprecated (u_dot)
          break;
        case 2 :
          LIDs = cells[b][c]->LIDs[set];
          numDOF = cells[b][c]->cellData->numDOF;
          data = cells[b][c]->phi[set];
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
void AssemblyManager<Node>::performBoundaryGather(const size_t & set, ViewType vec_dev, const int & type) {
  
  for (size_t b=0; b<boundaryCells.size(); b++) {
    
    Kokkos::View<LO*,AssemblyDevice> numDOF;
    Kokkos::View<ScalarT***,AssemblyDevice> data;
    Kokkos::View<int**,AssemblyDevice> offsets;
    LIDView LIDs;
    
    for (size_t c=0; c < boundaryCells[b].size(); c++) {
      if (boundaryCells[b][c]->numElem > 0) {
        
        switch(type) {
          case 0 :
            LIDs = boundaryCells[b][c]->LIDs[set];
            numDOF = boundaryCells[b][c]->cellData->numDOF;
            data = boundaryCells[b][c]->u[set];
            offsets = wkset[b]->offsets;
            break;
          case 1 : // deprecated (u_dot)
            break;
          case 2 :
            LIDs = boundaryCells[b][c]->LIDs[set];
            numDOF = boundaryCells[b][c]->cellData->numDOF;
            data = boundaryCells[b][c]->phi[set];
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

//==============================================================
// Scatter just the Jacobian
//==============================================================

template<class Node>
template<class MatType, class LocalViewType, class LIDViewType>
void AssemblyManager<Node>::scatterJac(const size_t & set, MatType J_kcrs, LocalViewType local_J,
                                       LIDViewType LIDs, LIDViewType paramLIDs,
                                       const bool & compute_disc_sens) {

  //Teuchos::TimeMonitor localtimer(*scattertimer);
  
  typedef typename Node::execution_space LA_exec;
  
  /////////////////////////////////////
  // This scatter needs to happen on the LA_device due to the use of J_kcrs->sumIntoValues()
  // Could be changed to the AssemblyDevice, but would require a mirror view of this data and filling such a view is nontrivial
  /////////////////////////////////////
  
  auto fixedDOF = isFixedDOF[set];
  bool use_atomics_ = false;
  if (LA_exec::concurrency() > 1) {
    use_atomics_ = true;
  }
  
  if (compute_disc_sens) {
    parallel_for("assembly insert Jac sens",
                 RangePolicy<LA_exec>(0,LIDs.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_t row=0; row<LIDs.extent(1); row++ ) {
        LO rowIndex = LIDs(elem,row);
        for (size_t col=0; col<paramLIDs.extent(1); col++ ) {
          LO colIndex = paramLIDs(elem,col);
          ScalarT val = local_J(elem,row,col);
          J_kcrs.sumIntoValues(colIndex, &rowIndex, 1, &val, false, use_atomics_); // isSorted, useAtomics
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
          J_kcrs.sumIntoValues(rowIndex, cols, numVals, vals, false, use_atomics_); // isSorted, useAtomics
        }
      }
    });
  }
  
}

//==============================================================
// Scatter just the Residual
//==============================================================

template<class Node>
template<class VecViewType, class LocalViewType, class LIDViewType>
void AssemblyManager<Node>::scatterRes(VecViewType res_view, LocalViewType local_res, LIDViewType LIDs) {

  //Teuchos::TimeMonitor localtimer(*scattertimer);
  
  typedef typename Node::execution_space LA_exec;
  
  /////////////////////////////////////
  // This scatter needs to happen on the LA_device due to the use of J_kcrs->sumIntoValues()
  // Could be changed to the AssemblyDevice, but would require a mirror view of this data and filling such a view is nontrivial
  /////////////////////////////////////
  
  auto fixedDOF = isFixedDOF[0];
  bool use_atomics_ = false;
  if (LA_exec::concurrency() > 1) {
    use_atomics_ = true;
  }
  
  parallel_for("assembly scatter res",
               RangePolicy<LA_exec>(0,LIDs.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    for( size_type row=0; row<LIDs.extent(1); row++ ) {
      LO rowIndex = LIDs(elem,row);
      if (!fixedDOF(rowIndex)) {
        for (size_type g=0; g<local_res.extent(2); g++) {
          ScalarT val = local_res(elem,row,g);
          if (use_atomics_) {
            Kokkos::atomic_add(&(res_view(rowIndex,g)), val);
          }
          else {
            res_view(rowIndex,g) += val;
          }
        }
      }
    }
  });
}

//==============================================================
// Scatter both and use wkset->res
//==============================================================

template<class Node>
template<class MatType, class VecViewType, class LIDViewType>
void AssemblyManager<Node>::scatter(const size_t & set, MatType J_kcrs, VecViewType res_view,
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
  
  // Make sure the functor can access the necessary data
  auto fixedDOF = isFixedDOF[set];
  auto res = wkset[block]->res;
  auto offsets = wkset[block]->offsets;
  auto numDOF = cellData[block]->numDOF;
  bool compute_sens_ = compute_sens;
#ifndef MrHyDE_NO_AD
  bool lump_mass_ = lump_mass, isAdjoint_ = isAdjoint, compute_jacobian_ = compute_jacobian;
#endif
  
  bool use_atomics_ = false;
  if (LA_exec::concurrency() > 1) {
    use_atomics_ = true;
  }
  
  parallel_for("assembly insert Jac",
               RangePolicy<LA_exec>(0,LIDs.extent(0)),
               KOKKOS_LAMBDA (const int elem ) {
    
    int row = 0;
    LO rowIndex = 0;
    
    // Residual scatter
    for (size_type n=0; n<numDOF.extent(0); ++n) {
      for (int j=0; j<numDOF(n); j++) {
        row = offsets(n,j);
        rowIndex = LIDs(elem,row);
        if (!fixedDOF(rowIndex)) {
          if (compute_sens_) {
#ifndef MrHyDE_NO_AD
            if (use_atomics_) {
              for (size_type r=0; r<res_view.extent(1); ++r) {
                ScalarT val = -res(elem,row).fastAccessDx(r);
                Kokkos::atomic_add(&(res_view(rowIndex,r)), val);
              }
            }
            else {
              for (size_type r=0; r<res_view.extent(1); ++r) {
                ScalarT val = -res(elem,row).fastAccessDx(r);
                res_view(rowIndex,r) += val;
              }
            }
#endif
          }
          else {
#ifndef MrHyDE_NO_AD
            ScalarT val = -res(elem,row).val();
#else
            ScalarT val = -res(elem,row);
#endif
            if (use_atomics_) {
              Kokkos::atomic_add(&(res_view(rowIndex,0)), val);
            }
            else {
              res_view(rowIndex,0) += val;
            }
          }
        }
      }
    }
    
#ifndef MrHyDE_NO_AD
    // Jacobian scatter
    if (compute_jacobian_) {
      const size_type numVals = LIDs.extent(1);
      int col = 0;
      LO cols[maxDerivs];
      ScalarT vals[maxDerivs];
      for (size_type n=0; n<numDOF.extent(0); ++n) {
        for (int j=0; j<numDOF(n); j++) {
          row = offsets(n,j);
          rowIndex = LIDs(elem,row);
          if (!fixedDOF(rowIndex)) {
            for (size_type m=0; m<numDOF.extent(0); m++) {
              for (int k=0; k<numDOF(m); k++) {
                col = offsets(m,k);
                if (isAdjoint_) {
                  vals[col] = res(elem,row).fastAccessDx(row);
                }
                else {
                  vals[col] = res(elem,row).fastAccessDx(col);
                }
                if (lump_mass_) {
                  cols[col] = rowIndex;
                }
                else {
                  cols[col] = LIDs(elem,col);
                }
              }
            }
            J_kcrs.sumIntoValues(rowIndex, cols, numVals, vals, false, use_atomics_); // isSorted, useAtomics
          }
        }
      }
    }
#endif
  });
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updatePhysicsSet(const size_t & set) {
  for (size_t block=0; block<blocknames.size(); ++block) {
    if (wkset[block]->isInitialized) {
      wkset[block]->updatePhysicsSet(set);
      cellData[block]->updatePhysicsSet(set);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we can get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::purgeMemory() {
  bool write_solution = settings->sublist("Postprocess").get("write solution",false);
  bool create_optim_movie = settings->sublist("Postprocess").get("create optimization movie",false);
  if (!write_solution && !create_optim_movie) {
    mesh.reset();
  }
  
  for (size_t block=0; block<cellData.size(); ++block) {
    cellData[block]->clearPhysicalData();
  }
  
}


template class MrHyDE::AssemblyManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
  template class MrHyDE::AssemblyManager<SubgridSolverNode>;
#endif
