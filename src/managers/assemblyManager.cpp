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

// Remove this when done testing
#include "Intrepid2_CellTools.hpp"


using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class Node>
AssemblyManager<Node>::AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_,
                                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                       Teuchos::RCP<MeshInterface> & mesh_,
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
  
  //if (settings->isSublist("Subgrid")) {
    //assembly_partitioning = "subgrid-preserving";
  //}
  
  string solver_type = settings->sublist("Solver").get<string>("solver","none"); // or "transient"
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  
  // needed information from the mesh
  //blocknames = mesh->block_names;
  mesh->stk_mesh->getElementBlockNames(blocknames);

  // check if we need to assembly volumetric, boundary and face terms
  for (size_t set=0; set<phys->setnames.size(); ++set) {
    vector<bool> set_assemble_vol, set_assemble_bndry, set_assemble_face;
    for (size_t block=0; block<blocknames.size(); ++block) {
      set_assemble_vol.push_back(phys->setPhysSettings[set][block].template get<bool>("assemble volume terms",true));
      set_assemble_bndry.push_back(phys->setPhysSettings[set][block].template get<bool>("assemble boundary terms",true));
      set_assemble_face.push_back(phys->setPhysSettings[set][block].template get<bool>("assemble face terms",false));
    }
    assemble_volume_terms.push_back(set_assemble_vol);
    assemble_boundary_terms.push_back(set_assemble_bndry);
    assemble_face_terms.push_back(set_assemble_face);
  }
  // overwrite assemble_face_terms if HFACE vars are used
  for (size_t set=0; set<assemble_face_terms.size(); ++set) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      vector<string> ctypes = phys->unique_types[block];
      for (size_t n=0; n<ctypes.size(); n++) {
        if (ctypes[n] == "HFACE") {
          assemble_face_terms[set][block] = true;
        }
      }
    }
  }
  
  // determine if we need to build basis functions
  for (size_t block=0; block<blocknames.size(); ++block) {
    bool build_volume = false, build_bndry = false, build_face = false;
  
    for (size_t set=0; set<phys->setnames.size(); ++set) {
      
      if (assemble_volume_terms[set][block]) {
        build_volume = true;
      }
      else if (phys->setPhysSettings[set][block].template get<bool>("build volume terms",true) ) {
        build_volume = true;
      }
      
      if (assemble_boundary_terms[set][block]) {
        build_bndry = true;
      }
      else if (phys->setPhysSettings[set][block].template get<bool>("build boundary terms",true)) {
        build_bndry = true;
      }
      
      if (assemble_face_terms[set][block]) {
        build_face = true;
      }
      else if (phys->setPhysSettings[set][block].template get<bool>("build face terms",false)) {
        build_face = true;
      }
    }
    build_volume_terms.push_back(build_volume);
    build_boundary_terms.push_back(build_bndry);
    build_face_terms.push_back(build_face);
  }
  
  // needed information from the physics interface
  varlist = phys->varlist;
  
  // Create groups/boundary groups
  this->createGroups();
  
  params->setupDiscretizedParameters(groups, boundary_groups);
  
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
    
    int numLocalDof = disc->DOF_ownedAndShared[set].size();//
    //int numLocalDof = disc->DOF[set]->getNumOwnedAndGhosted();
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
// Create the groups
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createGroups() {
  
  Teuchos::TimeMonitor localtimer(*grouptimer);
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createGroups ..." << endl;
    }
  }
  
  double storageProportion = settings->sublist("Solver").get<double>("storage proportion",1.0);
  
  vector<stk::mesh::Entity> all_meshElems;
  mesh->stk_mesh->getMyElements(all_meshElems);
  
  
  auto LIDs = disc->DOF_LIDs;
   
  // Disc manager stores offsets as [set][block][var][dof]
  vector<vector<vector<vector<int> > > > disc_offsets = disc->offsets;
  
  // We want these re-ordered as [block][set][var][dof]
  vector<vector<vector<vector<int> > > > my_offsets;
  for (size_t block=0; block<blocknames.size(); ++block) {
    vector<vector<vector<int> > > block_offsets;
    for (size_t set=0; set<disc_offsets.size(); ++set) {
      block_offsets.push_back(disc_offsets[set][block]);
    }
    my_offsets.push_back(block_offsets);
  }
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    Teuchos::RCP<GroupMetaData> blockGroupData;
    vector<Teuchos::RCP<Group> > block_groups;
    vector<Teuchos::RCP<BoundaryGroup> > block_boundary_groups;
    
    vector<stk::mesh::Entity> stk_meshElems;
    mesh->stk_mesh->getMyElements(blocknames[block], stk_meshElems);
    
    topo_RCP cellTopo = mesh->stk_mesh->getCellTopology(blocknames[block]);
    int numNodesPerElem = cellTopo->getNodeCount();
    int spaceDim = phys->spaceDim;
    size_t numTotalElem = stk_meshElems.size();
    size_t processedElem = 0;
    
    if (numTotalElem>0) {
      
      //vector<size_t> localIds;
      //Kokkos::DynRankView<ScalarT,HostDevice> blocknodes;
      //panzer_stk::workset_utils::getIdsAndVertices(*(mesh->stk_mesh), blocknames[block], localIds, blocknodes); // fill on host
      
      vector<size_t> myElem = disc->myElements[block];
      Kokkos::View<LO*,AssemblyDevice> eIDs("local element IDs on device",myElem.size());
      auto host_eIDs = Kokkos::create_mirror_view(eIDs);
      for (size_t elem=0; elem<myElem.size(); elem++) {
        host_eIDs(elem) = static_cast<LO>(myElem[elem]);
      }
      Kokkos::deep_copy(eIDs, host_eIDs);
      
      // LO is int, but just in case that changes ...
      size_t elemPerGroup;
      int tmp_elemPerGroup = settings->sublist("Solver").get<int>("workset size",100);
      if (tmp_elemPerGroup == -1) {
        elemPerGroup = numTotalElem;
      }
      else {
        elemPerGroup = std::min(static_cast<size_t>(tmp_elemPerGroup),numTotalElem);
      }
      
      vector<string> sideSets;// = mesh->side_names;
      mesh->stk_mesh->getSidesetNames(sideSets);
      vector<bool> aface;
      for (size_t set=0; set<assemble_face_terms.size(); ++set) {
        aface.push_back(assemble_face_terms[set][block]);
      }
      blockGroupData = Teuchos::rcp( new GroupMetaData(settings, cellTopo,
                                                       phys, block, 0, elemPerGroup,
                                                       build_face_terms[block],
                                                       aface, sideSets,
                                                       params->num_discretized_params));
                      
      disc->setReferenceData(blockGroupData);
      
      blockGroupData->requireBasisAtNodes = settings->sublist("Postprocess").get<bool>("plot solution at nodes",false);
      
      
      vector<vector<vector<int> > > curroffsets = my_offsets[block];
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
      
      blockGroupData->set_numDOF = set_numDOF;
      blockGroupData->set_numDOF_host = set_numDOF_host;
           
      blockGroupData->numDOF = set_numDOF[0];
      blockGroupData->numDOF_host = set_numDOF_host[0];
      
      //////////////////////////////////////////////////////////////////////////////////
      // Boundary groups
      //////////////////////////////////////////////////////////////////////////////////
      
      if (build_boundary_terms[block]) {
        
        int numBoundaryElem = elemPerGroup;
        
        ///////////////////////////////////////////////////////////////////////////////////
        // Rules for grouping elements into boundary groups
        //
        // 1.  All elements must be on the same processor
        // 2.  All elements must be on the same physical side
        // 3.  Each edge/face on the side must have the same local ID.
        // 4.  No more than numBoundaryElem (= numElem) in a group
        ///////////////////////////////////////////////////////////////////////////////////
        
        for (size_t side=0; side<sideSets.size(); side++ ) {
          string sideName = sideSets[side];
          
          vector<stk::mesh::Entity> sideEntities;
          mesh->stk_mesh->getMySides(sideName, blocknames[block], sideEntities);
          vector<size_t>             local_side_Ids;
          vector<stk::mesh::Entity> side_output;
          vector<size_t>             local_elem_Ids;
          
          panzer_stk::workset_utils::getSideElements(*(mesh->stk_mesh), blocknames[block], sideEntities, local_side_Ids, side_output);
          
          DRV sidenodes;
          mesh->stk_mesh->getElementVertices(side_output, blocknames[block],sidenodes);
          
          size_t numSideElem = local_side_Ids.size();
          size_t belemProg = 0;
          
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
                //Kokkos::View<LO*,AssemblyDevice> sideIndex("local side indices",currElem);
                DRV currnodes("currnodes", currElem, numNodesPerElem, spaceDim);
                
                auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
                Kokkos::View<LO*,HostDevice> host_eIndex2("element indices",currElem);
                //auto host_sideIndex = Kokkos::create_mirror_view(sideIndex); // mirror on host
                auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
                LO sideIndex;

                for (size_t e=0; e<currElem; e++) {
                  host_eIndex(e) = mesh->stk_mesh->elementLocalId(side_output[group[e+prog]]);
                  sideIndex = local_side_Ids[group[e+prog]];
                  for (size_type n=0; n<host_currnodes.extent(1); n++) {
                    for (size_type m=0; m<host_currnodes.extent(2); m++) {
                      host_currnodes(e,n,m) = sidenodes(group[e+prog],n,m);
                    }
                  }
                }
                Kokkos::deep_copy(currnodes,host_currnodes);
                Kokkos::deep_copy(eIndex,host_eIndex);
                Kokkos::deep_copy(host_eIndex2,host_eIndex);
                //Kokkos::deep_copy(sideIndex,host_sideIndex);
                
                // Build the Kokkos View of the group LIDs ------
                vector<LIDView> set_LIDs;
                for (size_t set=0; set<LIDs.size(); ++set) {
                  LIDView groupLIDs("LIDs",currElem,LIDs[set].extent(1));
                  auto currLIDs = LIDs[set];
                  parallel_for("assembly copy LIDs bgrp",
                               RangePolicy<AssemblyExec>(0,groupLIDs.extent(0)),
                               KOKKOS_LAMBDA (const int e ) {
                    size_t elemID = eIndex(e);
                    for (size_type j=0; j<currLIDs.extent(1); j++) {
                      groupLIDs(e,j) = currLIDs(elemID,j);
                    }
                  });
                  set_LIDs.push_back(groupLIDs);
                }
                
                //-----------------------------------------------
                // Set the side information (soon to be removed)-
                vector<Kokkos::View<int****,HostDevice> > set_sideinfo;
                for (size_t set=0; set<LIDs.size(); ++set) {
                  Kokkos::View<int****,HostDevice> sideinfo = disc->getSideInfo(set,block,host_eIndex2);
                  set_sideinfo.push_back(sideinfo);
                }
                
                bool storeThis = true;
                if (static_cast<double>(belemProg)/static_cast<double>(numSideElem) >= storageProportion) {
                  storeThis = false;
                }
                
                block_boundary_groups.push_back(Teuchos::rcp(new BoundaryGroup(blockGroupData, currnodes, eIndex, sideIndex,
                                                                               side, sideName, block_boundary_groups.size(),
                                                                               disc, storeThis)));
                size_t cindex = block_boundary_groups.size()-1;
                block_boundary_groups[cindex]->LIDs = set_LIDs;
                block_boundary_groups[cindex]->createHostLIDs();
                block_boundary_groups[cindex]->sideinfo = set_sideinfo;
                prog += currElem;
              }
            }
          }
        }
      }
      
      //////////////////////////////////////////////////////////////////////////////////
      // Groups
      //////////////////////////////////////////////////////////////////////////////////
      
      size_t prog = 0;
      vector<vector<size_t> > elem_groups;
      
      if (assembly_partitioning == "sequential") { // default
        while (prog < numTotalElem) {
          
          vector<size_t> newgroup;
          
          size_t currElem = elemPerGroup;
          if (prog+currElem > numTotalElem){
            currElem = numTotalElem-prog;
          }
          for (size_t e=prog; e<prog+currElem; ++e) {
            newgroup.push_back(e);
          }
          elem_groups.push_back(newgroup);
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
        // 4.  No more than elemPerGroup (= numElem) in a group
        ///////////////////////////////////////////////////////////////////////////////////
        
        if (block_boundary_groups.size() > 0) {
          Kokkos::View<bool*> beenadded("been processed",numTotalElem);
          deep_copy(beenadded,false);
          
          Kokkos::View<bool**> onbndry("onbndry",numTotalElem,block_boundary_groups.size());
          deep_copy(onbndry,false);
          
          for (size_t bc=0; bc<block_boundary_groups.size(); ++bc) {
            auto eind = create_mirror_view(block_boundary_groups[bc]->localElemID);
            deep_copy(eind,block_boundary_groups[bc]->localElemID);
            
            for (size_type e=0; e<eind.extent(0); ++e) {
              onbndry(eind(e),bc) = true;
            }
          }
          
          size_t numAdded=0;
          while (numAdded < numTotalElem) {
            vector<size_t> newgroup;
            bool foundind = false;
            size_t refind = 0;
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
            for (size_t j=refind+1; j<numTotalElem; ++j) {
              bool matches = true;
              for (size_type k=0; k<onbndry.extent(1); ++k) {
                if (onbndry(j,k) != onbndry(refind,k)) {
                  matches = false;
                }
              }
              if (matches && newgroup.size() < elemPerGroup) {
                newgroup.push_back(j);
                beenadded(j) = true;
                numAdded++;
              }
            }
            elem_groups.push_back(newgroup);
          }
          
          // Re-order the groups from biggest to smallest
          // This is needed for certain parts of MrHyDE that assume the first
          // group contains the most elements
          for (size_t grp=0; grp<elem_groups.size()-1; ++grp) {
            size_t mxgrp_ind = grp;
            size_t mxgrp = elem_groups[grp].size();
            bool perform_swap = false;
            for (size_t grp2=grp+1; grp2<elem_groups.size(); ++grp2) {
              if (elem_groups[grp2].size() > mxgrp) {
                mxgrp = elem_groups[grp2].size();
                mxgrp_ind = grp2;
                perform_swap = true;
              }
            }
            if (perform_swap) {
              elem_groups[grp].swap(elem_groups[mxgrp_ind]);
            }
          }
        }
        else {
          while (prog < numTotalElem) {
            
            vector<size_t> newgroup;
            
            size_t currElem = elemPerGroup;
            if (prog+currElem > numTotalElem){
              currElem = numTotalElem-prog;
            }
            for (size_t e=prog; e<prog+currElem; ++e) {
              newgroup.push_back(e);
            }
            elem_groups.push_back(newgroup);
            prog += currElem;
          }
        }
        
      }
      
      elemPerGroup = std::min(elemPerGroup, elem_groups[0].size());
      
      // Add the groups correspondng to the groups
      for (size_t grp=0; grp<elem_groups.size(); ++grp) {
        size_t currElem = elem_groups[grp].size();
        
        bool storeThis = true;
        if (static_cast<double>(processedElem)/static_cast<double>(numTotalElem) >= storageProportion) {
          storeThis = false;
        }
        
        processedElem += currElem;
        
        Kokkos::View<LO*,AssemblyDevice> eIndex("element indices",currElem);
        DRV currnodes("currnodes", currElem, numNodesPerElem, spaceDim);
        
        auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
        Kokkos::View<LO*,HostDevice> host_eIndex2("element indices on host",currElem);
        
        for (size_t e=0; e<currElem; ++e) {
          host_eIndex(e) = elem_groups[grp][e];
        }
        Kokkos::deep_copy(eIndex,host_eIndex);
        Kokkos::deep_copy(host_eIndex2,host_eIndex);
        
        vector<LIDView> set_LIDs;
        for (size_t set=0; set<LIDs.size(); ++set) {
          LIDView groupLIDs("LIDs on device",currElem,LIDs[set].extent(1));
          auto currLIDs = LIDs[set];
          parallel_for("assembly copy nodes",
                       RangePolicy<AssemblyExec>(0,eIndex.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            LO elemID = eIndex(e);
            for (size_type j=0; j<currLIDs.extent(1); j++) {
              groupLIDs(e,j) = currLIDs(eIDs(elemID),j);
            }
          });
          set_LIDs.push_back(groupLIDs);
        }
        
        vector<size_t> local_grp(elem_groups[grp].size());
        for (size_t e=0; e<local_grp.size(); ++e) {
          local_grp[e] = host_eIDs(elem_groups[grp][e]);
        }

        mesh->stk_mesh->getElementVertices(local_grp, blocknames[block], currnodes);
        
        // Set the side information (soon to be removed)-
        vector<Kokkos::View<int****,HostDevice> > set_sideinfo;
        for (size_t set=0; set<LIDs.size(); ++set) {
          Kokkos::View<int****,HostDevice> sideinfo = disc->getSideInfo(set,block,host_eIndex2);
          set_sideinfo.push_back(sideinfo);
        }
        
        block_groups.push_back(Teuchos::rcp(new Group(blockGroupData, currnodes, eIndex,
                                                     disc, storeThis)));
        
        size_t cindex = block_groups.size()-1;
        block_groups[cindex]->LIDs = set_LIDs;
        block_groups[cindex]->createHostLIDs();
        block_groups[cindex]->sideinfo = set_sideinfo;
        
        prog += elemPerGroup;
        
      }
    }    
    else {
      blockGroupData = Teuchos::rcp( new GroupMetaData());
    }
    
    groupData.push_back(blockGroupData);
    groups.push_back(block_groups);
    boundary_groups.push_back(block_boundary_groups);
    
  }
}

// =======================================================
// Have the groups compute and store the basis functions
// at the quadrature points (if storage is turned on)
// =======================================================

template<class Node>
void AssemblyManager<Node>::allocateGroupStorage() {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::allocateGroupStorage" << endl;
    }
  }

  bool keepnodes = false;  
  // There are a few scenarios where we want the groups to keep their nodes
  if (settings->sublist("Solver").get<string>("initial type","L2-projection") == "interpolation") {
    keepnodes = true;
  }
  if (settings->isSublist("Subgrid")) {
    keepnodes = true;
  }
  if (settings->sublist("Solver").get<bool>("keep nodes",false)) {
    keepnodes = true;
  }
  
  for (size_t block=0; block<groups.size(); ++block) {
    
    if (groupData[block]->use_basis_database) {

      vector<std::pair<size_t,size_t> > first_users; // stores <grpID,elemID>
      vector<std::pair<size_t,size_t> > first_boundary_users; // stores <grpID,elemID>
      size_t totalelem = 0, boundaryelem = 0;
      int dimension = groupData[block]->dimension;
      size_type numip = groupData[block]->ref_ip.extent(0);
      size_type numsideip = groupData[block]->ref_side_ip[0].extent(0);

      double database_TOL = settings->sublist("Solver").get<double>("database TOL",1.0e-10);
            
      /////////////////////////////////////////////////////////////////////////////
      // Step 1: determine the unique elements
      /////////////////////////////////////////////////////////////////////////////
      
      /////////////////////////////////////////////////////////////////////////////
      // Step 1a: volumetric elements
      /////////////////////////////////////////////////////////////////////////////
      
      /*
      // For each element, search db for match
      // The match must have the same basis values and the appropriate derivative values
      size_t numbasis = groupData[block]->basis_pointers.size();
      vector<View_Sc4> db_basis(numbasis), db_basis_grad(numbasis), db_basis_curl(numbasis);
      vector<View_Sc3> db_basis_div(numbasis);
      
      {

        Teuchos::TimeMonitor localtimer(*groupdatabaseCreatetimer);

        Kokkos::View<ScalarT*,HostDevice> db_measures("measures for data base",0);
        
        for (size_t grp=0; grp<groups[block].size(); ++grp) {

          groups[block][grp]->storeAll = false;
          totalelem += groups[block][grp]->numElem;
          Kokkos::View<LO*,AssemblyDevice> index("basis database index",groups[block][grp]->numElem);

          // Get the Jacobian for this group
          DRV jacobian("jacobian", groups[block][grp]->numElem, numip, dimension, dimension);
          disc->getJacobian(groupData[block], groups[block][grp]->nodes, jacobian);
          
          // Get the measures for this group
          DRV measure("measure", groups[block][grp]->numElem);
          disc->getMeasure(groupData[block], jacobian, measure);
          auto measure_host = create_mirror_view(measure);
          deep_copy(measure_host,measure);
  
          vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
          vector<View_Sc3> tbasis_div;

          // Create an index for the basis type so we don't need to keep checking
          // 0 = HGRAD, 1 = HDIV, 2 = HCURL, 3 = HVOL, 4 = HFACE
          //vector<size_t> basis_type(tbasis.size());
          
          disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->nodes, 
                                           groups[block][grp]->orientation,
                                           tbasis, tbasis_grad, tbasis_curl,
                                           tbasis_div, tbasis_nodes, true);

          for (size_t e=0; e<groups[block][grp]->numElem; ++e) {

            bool found = false;
            size_t db_itr = 0;
            while (!found && db_itr<first_users.size()) {
              ScalarT refmeas = measure_host(e);//std::pow(measure_host(e),1.0/dimension);
              if (abs(db_measures(db_itr) - measure_host(e))/refmeas < database_TOL) {
                bool ruled_out = false;

                for (size_t i=0; i<groupData[block]->basis_pointers.size(); i++) {

                  // First, check the basis values
                  View_Sc4 cbasis = tbasis[i];
                  View_Sc4 dbasis = db_basis[i];
                  
                  ScalarT scale = 0.0;
                  for (size_type dof=0; dof<cbasis.extent(1); ++dof) {
                    for (size_type pt=0; pt<cbasis.extent(2); ++pt) {
                      for (size_type dim=0; dim<cbasis.extent(3); ++dim) {
                        scale += abs(cbasis(e,dof,pt,dim));
                      }
                    }
                  }
                  size_type dof=0;
                  while (!ruled_out && dof<cbasis.extent(1)) {
                    size_type ip = 0;
                    while (!ruled_out && ip<cbasis.extent(2)) {
                      size_type dim=0;
                      while (!ruled_out && dim<cbasis.extent(3)) {
                        if (abs(cbasis(e,dof,ip,dim)-dbasis(db_itr,dof,ip,dim))/scale < database_TOL) {
                          ++dim;
                        }
                        else {
                          ruled_out = true;
                        }
                      }
                      ++ip;
                    }
                    ++dof;
                  }

                  if (!ruled_out) {

                    // Now check derivatives, but just one per basis type
                    if (tbasis_grad[i].extent(0) == groups[block][grp]->numElem) { // HGRAD basis
                      View_Sc4 cbasis_deriv = tbasis_grad[i];
                      View_Sc4 dbasis_deriv = db_basis_grad[i];
                      size_type dof=0;
                      ScalarT scale = 0.0;
                      for (size_type dof=0; dof<cbasis_deriv.extent(1); ++dof) {
                        for (size_type pt=0; pt<cbasis_deriv.extent(2); ++pt) {
                          for (size_type dim=0; dim<cbasis_deriv.extent(3); ++dim) {
                            scale += abs(cbasis_deriv(e,dof,pt,dim));
                          }
                        }
                      }
                      while (!ruled_out && dof<cbasis_deriv.extent(1)) {
                        size_type ip = 0;
                        while (!ruled_out && ip<cbasis_deriv.extent(2)) {
                          size_type dim=0;
                          while (!ruled_out && dim<cbasis_deriv.extent(3)) {
                            if (abs(cbasis_deriv(e,dof,ip,dim)-dbasis_deriv(db_itr,dof,ip,dim))/scale < database_TOL) {
                              ++dim;
                            }
                            else {
                              ruled_out = true;
                            }
                          }
                          if (!ruled_out) {
                            ++ip;
                          }
                        }
                        if (!ruled_out) {
                          ++dof;
                        }
                      }
                    }
                    else if (tbasis_curl[i].extent(0) == groups[block][grp]->numElem) { // HCURL basis
                      View_Sc4 cbasis_deriv = tbasis_curl[i];
                      View_Sc4 dbasis_deriv = db_basis_curl[i];
                      size_type dof=0;
                      ScalarT scale = 0.0;
                      for (size_type dof=0; dof<cbasis_deriv.extent(1); ++dof) {
                        for (size_type pt=0; pt<cbasis_deriv.extent(2); ++pt) {
                          for (size_type dim=0; dim<cbasis_deriv.extent(3); ++dim) {
                            scale += abs(cbasis_deriv(e,dof,pt,dim));
                          }
                        }
                      }
                      while (!ruled_out && dof<cbasis_deriv.extent(1)) {
                        size_type ip = 0;
                        while (!ruled_out && ip<cbasis_deriv.extent(2)) {
                          size_type dim=0;
                          while (!ruled_out && dim<cbasis_deriv.extent(3)) {
                            if (abs(cbasis_deriv(e,dof,ip,dim)-dbasis_deriv(db_itr,dof,ip,dim))/scale < database_TOL) {
                              ++dim;
                            }
                            else {
                              ruled_out = true;
                            }
                          }
                          if (!ruled_out) {
                            ++ip;
                          }
                        }
                        if (!ruled_out) {
                          ++dof;
                        }
                      }
                    }
                    else if (tbasis_div[i].extent(0) == groups[block][grp]->numElem) { // HDIV basis
                      View_Sc3 cbasis_deriv = tbasis_div[i];
                      View_Sc3 dbasis_deriv = db_basis_div[i];
                      size_type dof=0;
                      ScalarT scale = 0.0;
                      for (size_type dof=0; dof<cbasis_deriv.extent(1); ++dof) {
                        for (size_type pt=0; pt<cbasis_deriv.extent(2); ++pt) {
                          scale += abs(cbasis_deriv(e,dof,pt));
                        }
                      }
                      while (!ruled_out && dof<cbasis_deriv.extent(1)) {
                        size_type ip = 0;
                        while (!ruled_out && ip<cbasis_deriv.extent(2)) {
                          if (abs(cbasis_deriv(e,dof,ip)-dbasis_deriv(db_itr,dof,ip))/scale < database_TOL) {
                            ++ip;
                          }
                          else {
                            ruled_out = true;
                          }
                        }
                        if (!ruled_out) {
                          ++dof;
                        }
                      }
                    }
                  }
                }
                if (ruled_out) {
                  ++db_itr;
                }
                else {
                  found = true;
                  index(e) = db_itr;
                }
              }
              else {
                ++db_itr;
              }
            }
            if (!found) {
              index(e) = first_users.size();
              std::pair<size_t,size_t> newuj{grp,e};
              first_users.push_back(newuj);

              // Resize measures and basis/ and add new one
              size_t pad = 100;
              bool resize = false;
              if (first_users.size() >= db_measures.extent(0)) {
                resize = true;
                Kokkos::resize(db_measures, pad+db_measures.extent(0));
                //cout << "db size = " << db_measures.extent(0) << endl;
              }
                for (size_t n=0; n<db_basis.size(); ++n) {
                  if (resize) {
                    Kokkos::resize(db_basis[n], pad+db_basis[n].extent(0), 
                                   tbasis[n].extent(1), tbasis[n].extent(2), tbasis[n].extent(3) );  
                  }
                  auto db_slice = subview(db_basis[n], first_users.size()-1, ALL(), ALL(), ALL());
                  auto jac_slice = subview(tbasis[n], e, ALL(), ALL(), ALL());
                  deep_copy(db_slice,jac_slice);
                }
                
                // HGRAD
                for (size_t n=0; n<db_basis_grad.size(); ++n) {
                  if (tbasis_grad[n].extent(0) == groups[block][grp]->numElem) {
                    if (resize) {
                      Kokkos::resize(db_basis_grad[n], pad+db_basis_grad[n].extent(0), 
                                     tbasis_grad[n].extent(1), tbasis_grad[n].extent(2), tbasis_grad[n].extent(3) );  
                    }
                    auto db_slice = subview(db_basis_grad[n], first_users.size()-1, ALL(), ALL(), ALL());
                    auto jac_slice = subview(tbasis_grad[n], e, ALL(), ALL(), ALL());
                    deep_copy(db_slice,jac_slice);
                  }
                }
                
                // HCURL
                for (size_t n=0; n<db_basis_curl.size(); ++n) {
                  if (tbasis_curl[n].extent(0) == groups[block][grp]->numElem) {
                    if (resize) {
                      Kokkos::resize(db_basis_curl[n], pad+db_basis_curl[n].extent(0), 
                                     tbasis_curl[n].extent(1), tbasis_curl[n].extent(2), tbasis_curl[n].extent(3) );  
                    }
                    auto db_slice = subview(db_basis_curl[n], first_users.size()-1, ALL(), ALL(), ALL());
                    auto jac_slice = subview(tbasis_curl[n], e, ALL(), ALL(), ALL());
                    deep_copy(db_slice,jac_slice);
                  }
                }
                
                // HDIV
                for (size_t n=0; n<db_basis_div.size(); ++n) {
                  if (tbasis_div[n].extent(0) == groups[block][grp]->numElem) {
                    if (resize) {
                      Kokkos::resize(db_basis_div[n], pad+db_basis_div[n].extent(0), 
                                     tbasis_div[n].extent(1), tbasis_div[n].extent(2) );  
                    }
                    auto db_slice = subview(db_basis_div[n], first_users.size()-1, ALL(), ALL());
                    auto jac_slice = subview(tbasis_div[n], e, ALL(), ALL());
                    deep_copy(db_slice,jac_slice);
                  }
                }
                
                
              //}
              db_measures(first_users.size()-1) = measure_host(e);
            }
            
          }
        
          groups[block][grp]->basis_database_index = index;
        } 
      
        if (first_users.size() < db_measures.extent(0)) {
          Kokkos::resize(db_measures, first_users.size());
          for (size_t n=0; n<db_basis.size(); ++n) {
            Kokkos::resize(db_basis[n], first_users.size(),
                           db_basis[n].extent(1), db_basis[n].extent(2), db_basis[n].extent(3) );  
          }    
          // HGRAD
          for (size_t n=0; n<db_basis_grad.size(); ++n) {
            Kokkos::resize(db_basis_grad[n], first_users.size(),
                           db_basis_grad[n].extent(1), db_basis_grad[n].extent(2), db_basis_grad[n].extent(3) );  
          }      
          // HCURL
          for (size_t n=0; n<db_basis_curl.size(); ++n) {
            Kokkos::resize(db_basis_curl[n], first_users.size(),
                           db_basis_curl[n].extent(1), db_basis_curl[n].extent(2), db_basis_curl[n].extent(3) );  
          }
                
                // HDIV
          for (size_t n=0; n<db_basis_div.size(); ++n) {
            Kokkos::resize(db_basis_div[n], first_users.size(),
                           db_basis_div[n].extent(1), db_basis_div[n].extent(2) );  
          }
        }           

        groupData[block]->database_basis = db_basis;
        groupData[block]->database_basis_grad = db_basis_grad;
        groupData[block]->database_basis_div = db_basis_div;
        groupData[block]->database_basis_curl = db_basis_curl;

      }
*/

      {
        Teuchos::TimeMonitor localtimer(*groupdatabaseCreatetimer);

        Kokkos::View<ScalarT****,HostDevice> db_jacobians("jacobians for data base",1,numip,dimension,dimension);
        Kokkos::View<ScalarT*,HostDevice> db_measures("measures for data base",1);
        
        // There are only so many unique orientation
        // Creating a short list of the unique ones and the index for each element 

        vector<string> unique_orients;
        vector<vector<size_t> > all_orients;
        for (size_t grp=0; grp<groups[block].size(); ++grp) {
          vector<size_t> grp_orient(groups[block][grp]->numElem);
          for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
            string orient = groups[block][grp]->orientation(e).to_string();
            bool found = false;
            size_t oprog = 0;
            while (!found && oprog<unique_orients.size()) {
              if (orient == unique_orients[oprog]) {
                found = true;
              }
              else {
                ++oprog;
              }
            }
            if (found) {
              grp_orient[e] = oprog;
            }
            else {
              unique_orients.push_back(orient);
              grp_orient[e] = unique_orients.size()-1;
            }
          }
          all_orients.push_back(grp_orient);
        }
        
        
        for (size_t grp=0; grp<groups[block].size(); ++grp) {
          groups[block][grp]->storeAll = false;
          totalelem += groups[block][grp]->numElem;
          Kokkos::View<LO*,AssemblyDevice> index("basis database index",groups[block][grp]->numElem);

          // Get the Jacobian for this group
          DRV jacobian("jacobian", groups[block][grp]->numElem, numip, dimension, dimension);
          disc->getJacobian(groupData[block], groups[block][grp]->nodes, jacobian);
          auto jacobian_host = create_mirror_view(jacobian);
          deep_copy(jacobian_host,jacobian);
          
          // Get the measures for this group
          DRV measure("measure", groups[block][grp]->numElem);
          disc->getMeasure(groupData[block], jacobian, measure);
          auto measure_host = create_mirror_view(measure);
          deep_copy(measure_host,measure);
  
          for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
            bool found = false;
            size_t prog = 0;
            while (!found && prog<first_users.size()) {
              size_t refgrp = first_users[prog].first;
              size_t refelem = first_users[prog].second;

              // Check #1: element orientations
              size_t orient = all_orients[grp][e];
              size_t reforient = all_orients[refgrp][refelem];
              if (orient == reforient) {
                
                // Check #2: element measures
                ScalarT diff = abs(measure_host(e)-db_measures(prog));
                ScalarT refmeas = std::pow(db_measures(prog),1.0/dimension);
                if (abs(diff/refmeas)<database_TOL) { 
                
                  // Check #3: element Jacobians
                  ScalarT diff2 = 0.0; 
                  for (size_type pt=0; pt<jacobian_host.extent(1); ++pt) {
                    for (size_type d0=0; d0<jacobian_host.extent(2); ++d0) {
                      for (size_type d1=0; d1<jacobian_host.extent(3); ++d1) {
                        diff2 += abs(jacobian_host(e,pt,d0,d1) - db_jacobians(prog,pt,d0,d1));
                      }
                    }
                  }
                  if (abs(diff2/refmeas)<database_TOL) { 
                    found = true;
                    index(e) = prog;                
                  }
                  else {
                    ++prog;
                  }               

                }
                else {
                  ++prog;
                }
              }
              else {
                ++prog;
              }
            }
            if (!found) {
              index(e) = first_users.size();
              std::pair<size_t,size_t> newuj{grp,e};
              first_users.push_back(newuj);

              // Resize Jacobians and add new one
              if (first_users.size() > db_jacobians.extent(0)) {
                Kokkos::resize(db_jacobians, 2*db_jacobians.extent(0), numip, dimension, dimension);
                //cout << "New db_jacobian size = " << db_jacobians.extent(0) << endl;
              }
              auto db_slice = subview(db_jacobians, first_users.size()-1, ALL(), ALL(), ALL());
              auto jac_slice = subview(jacobian, e, ALL(), ALL(), ALL());
              deep_copy(db_slice,jac_slice);

              // Resize measures and add new one
              if (first_users.size() > db_measures.extent(0)) {
                Kokkos::resize(db_measures, 2*db_measures.extent(0));
                //cout << "New db_measures size = " << db_measures.extent(0) << endl;
              }
              db_measures(first_users.size()-1) = measure(e);
            }
          }
          groups[block][grp]->basis_index = index;
        }
      }

      /////////////////////////////////////////////////////////////////////////////
      // Step 1b: boundary elements
      /////////////////////////////////////////////////////////////////////////////
      
      {
        Teuchos::TimeMonitor localtimer(*groupdatabaseCreatetimer);

        Kokkos::View<ScalarT****,HostDevice> db_jacobians("jacobians for data base",1,numip,dimension,dimension);
        Kokkos::View<ScalarT*,HostDevice> db_measures("measures for data base",1);
      
        // There are only so many unique orientation
        // Creating a short list of the unique ones and the index for each element 
        vector<string> unique_orients;
        vector<vector<size_t> > all_orients;
        for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
          vector<size_t> grp_orient(boundary_groups[block][grp]->numElem);
          for (size_t e=0; e<boundary_groups[block][grp]->numElem; ++e) {
            string orient = boundary_groups[block][grp]->orientation(e).to_string();
            bool found = false;
            size_t oprog = 0;
            while (!found && oprog<unique_orients.size()) {
              if (orient == unique_orients[oprog]) {
                found = true;
              }
              else {
                ++oprog;
              }
            }
            if (found) {
              grp_orient[e] = oprog;
            }
            else {
              unique_orients.push_back(orient);
              grp_orient[e] = unique_orients.size()-1;
            }
          }
          all_orients.push_back(grp_orient);
        }
        
        for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
          boundary_groups[block][grp]->storeAll = false;
          boundaryelem += boundary_groups[block][grp]->numElem;
          Kokkos::View<LO*,AssemblyDevice> index("basis database index",boundary_groups[block][grp]->numElem);

          // Get the Jacobian for this group
          DRV jacobian("jacobian", boundary_groups[block][grp]->numElem, numip, dimension, dimension);
          disc->getJacobian(groupData[block], boundary_groups[block][grp]->nodes, jacobian);
          auto jacobian_host = create_mirror_view(jacobian);
          deep_copy(jacobian_host,jacobian);
          
          // Get the measures for this group
          DRV measure("measure", boundary_groups[block][grp]->numElem);
          disc->getMeasure(groupData[block], jacobian, measure);
          auto measure_host = create_mirror_view(measure);
          deep_copy(measure_host,measure);
          
          for (size_t e=0; e<boundary_groups[block][grp]->numElem; ++e) {
            LO localSideID = boundary_groups[block][grp]->localSideID;
            LO sidenum = boundary_groups[block][grp]->sidenum;
            bool found = false;
            size_t prog = 0;
            while (!found && prog<first_boundary_users.size()) {
              size_t refgrp = first_boundary_users[prog].first;
              size_t refelem = first_boundary_users[prog].second;

              // Check #0: side IDs must match
              if (localSideID == boundary_groups[block][refgrp]->localSideID &&
                  sidenum == boundary_groups[block][refgrp]->sidenum) {

                // Check #1: element orientations

                size_t orient = all_orients[grp][e];
                size_t reforient = all_orients[refgrp][refelem];
                if (orient == reforient) { // if all 3 checks have passed
                    
                  // Check #2: element measures
                  ScalarT diff = abs(measure_host(e)-db_measures(prog));
                  ScalarT refmeas = std::pow(db_measures(prog),1.0/dimension);
            
                  if (abs(diff/refmeas)<database_TOL) { 

                    // Check #3: element Jacobians
  
                    ScalarT diff2 = 0.0;              
                    for (size_type pt=0; pt<jacobian_host.extent(1); ++pt) {
                      for (size_type d0=0; d0<jacobian_host.extent(2); ++d0) {
                        for (size_type d1=0; d1<jacobian_host.extent(3); ++d1) {
                          diff2 += abs(jacobian_host(e,pt,d0,d1) - db_jacobians(prog,pt,d0,d1));
                        }
                      }
                    }

                    if (abs(diff2/refmeas)<database_TOL) { 
                      found = true;
                      index(e) = prog;                
                    }
                    else {
                      ++prog;
                    }

                  }
                  else {
                    ++prog;
                  }
                }
                else {
                  ++prog;
                }
              }
              else {
                ++prog;
              }
            }
            if (!found) {
              index(e) = first_boundary_users.size();
              std::pair<size_t,size_t> newuj{grp,e};
              first_boundary_users.push_back(newuj);
              
              // Resize Jacobians and add new one
              if (first_boundary_users.size() > db_jacobians.extent(0)) {
                Kokkos::resize(db_jacobians, 2*db_jacobians.extent(0), numip, dimension, dimension);
                //cout << "New boundary db_jacobian size = " << db_jacobians.extent(0) << endl;
              }
              auto db_slice = subview(db_jacobians, first_boundary_users.size()-1, ALL(), ALL(), ALL());
              auto jac_slice = subview(jacobian, e, ALL(), ALL(), ALL());
              deep_copy(db_slice,jac_slice);

              // Resize measures and add new one
              if (first_boundary_users.size() > db_measures.extent(0)) {
                Kokkos::resize(db_measures, 2*db_measures.extent(0));
                //cout << "New boundary db_measures size = " << db_measures.extent(0) << endl;
              }
              db_measures(first_boundary_users.size()-1) = measure(e);
            }
          }
          boundary_groups[block][grp]->basis_index = index;
        }
      }
    
      /////////////////////////////////////////////////////////////////////////////
      // Step 2: inform the user about the savings
      /////////////////////////////////////////////////////////////////////////////
      
      if (verbosity > 5) {
        cout << " - Processor " << Comm->getRank() << ": Number of elements on block " << blocknames[block] << ": " << totalelem << endl;
        cout << " - Processor " << Comm->getRank() << ": Number of unique elements on block " << blocknames[block] << ": " << first_users.size() << endl;
        cout << " - Processor " << Comm->getRank() << ": Database memory savings on " << blocknames[block] << ": " 
             << (100.0 - 100.0*((double)first_users.size()/(double)totalelem)) << "%" << endl;
        cout << " - Processor " << Comm->getRank() << ": Number of boundary elements on block " << blocknames[block] << ": " << boundaryelem << endl;
        cout << " - Processor " << Comm->getRank() << ": Number of unique boundary elements on block " << blocknames[block] << ": " << first_boundary_users.size() << endl;
        cout << " - Processor " << Comm->getRank() << ": Database boundary memory savings on " << blocknames[block] << ": " 
             << (100.0 - 100.0*((double)first_boundary_users.size()/(double)boundaryelem)) << "%" << endl;
      }

      /////////////////////////////////////////////////////////////////////////////
      // Step 3: build the database basis
      /////////////////////////////////////////////////////////////////////////////

      {
        Teuchos::TimeMonitor localtimer(*groupdatabaseBasistimer);
      
        /////////////////////////////////////////////////////////////////////////////
        // Step 3a: volumetric database
        /////////////////////////////////////////////////////////////////////////////

        {

          
          size_t database_numElem = first_users.size();
          DRV database_nodes("nodes for the database",database_numElem, groups[block][0]->nodes.extent(1), dimension);
          Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> database_orientation("database orientations",database_numElem);
          View_Sc2 database_wts("physical wts",database_numElem, groupData[block]->ref_ip.extent(0));
        
          auto database_nodes_host = create_mirror_view(database_nodes);
          auto database_orientation_host = create_mirror_view(database_orientation);
          auto database_wts_host = create_mirror_view(database_wts);

          for (size_t e=0; e<first_users.size(); ++e) {
            size_t refgrp = first_users[e].first;
            size_t refelem = first_users[e].second;
          
            // Get the nodes on the host
            auto sub_nodes = subview(groups[block][refgrp]->nodes, refelem, ALL(), ALL());
            auto sub_nodes_host = create_mirror_view(sub_nodes);
            deep_copy(sub_nodes_host, sub_nodes);

            for (size_type node=0; node<database_nodes.extent(1); ++node) {
              for (size_type dim=0; dim<database_nodes.extent(2); ++dim) {
                database_nodes_host(e,node,dim) = sub_nodes_host(node,dim);
              }
            }
          
            // Get the orientations on the host
            auto orientations_host = create_mirror_view(groups[block][refgrp]->orientation);
            deep_copy(orientations_host, groups[block][refgrp]->orientation);
            database_orientation_host(e) = orientations_host(refelem);

            // Get the wts on the host
            auto sub_wts = subview(groups[block][refgrp]->wts, refelem, ALL());
            auto sub_wts_host = create_mirror_view(sub_wts);
            deep_copy(sub_wts_host, sub_wts);
          
            for (size_type pt=0; pt<sub_wts.extent(0); ++pt) {
              database_wts_host(e,pt) = sub_wts_host(pt);
            }
          
          }
        
          deep_copy(database_nodes, database_nodes_host);
          deep_copy(database_orientation, database_orientation_host);
          deep_copy(database_wts, database_wts_host);

          
          vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
          vector<View_Sc3> tbasis_div;
      
          disc->getPhysicalVolumetricBasis(groupData[block], database_nodes, database_orientation,
                                          tbasis, tbasis_grad, tbasis_curl,
                                          tbasis_div, tbasis_nodes, true);
          groupData[block]->database_basis = tbasis;
          groupData[block]->database_basis_grad = tbasis_grad;
          groupData[block]->database_basis_div = tbasis_div;
          groupData[block]->database_basis_curl = tbasis_curl;

          if (groupData[block]->build_face_terms) {
            for (size_type side=0; side<groupData[block]->numSides; side++) {
              vector<View_Sc4> face_basis, face_basis_grad;
          
              disc->getPhysicalFaceBasis(groupData[block], side, database_nodes, database_orientation,
                                         face_basis, face_basis_grad);
          
              groupData[block]->database_face_basis.push_back(face_basis);
              groupData[block]->database_face_basis_grad.push_back(face_basis_grad);
            }
        
          }

          /////////////////////////////////////////////////////////////////////////////
          // Step 3b: database of mass matrices while we have basis and wts
          /////////////////////////////////////////////////////////////////////////////
        
          // Create a database of mass matrices
          if (groupData[block]->use_mass_database) {
            size_t database_numElem = first_users.size();
            for (size_t set=0; set<phys->setnames.size(); ++set) {
              View_Sc3 mass("local mass",database_numElem, groups[block][0]->LIDs[set].extent(1), 
                             groups[block][0]->LIDs[set].extent(1));
  
              auto offsets = wkset[block]->set_offsets[set];
              auto numDOF = groupData[block]->set_numDOF[set];
          
              auto cwts = database_wts;
        
              for (size_type n=0; n<numDOF.extent(0); n++) {
                ScalarT mwt = phys->masswts[set][block][n];
                View_Sc4 cbasis = tbasis[wkset[block]->set_usebasis[set][n]];
                string btype = wkset[block]->basis_types[wkset[block]->set_usebasis[set][n]];
                auto off = subview(offsets,n,ALL());
                if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
                  parallel_for("Group get mass",
                             RangePolicy<AssemblyExec>(0,mass.extent(0)),
                             KOKKOS_LAMBDA (const size_type e ) {
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
                               KOKKOS_LAMBDA (const size_type e ) {
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
              groupData[block]->database_mass.push_back(mass);
            }
          }
        }

        
        /////////////////////////////////////////////////////////////////////////////
        // Step 3c: boundary database
        /////////////////////////////////////////////////////////////////////////////
        
        {
          size_t database_boundary_numElem = first_boundary_users.size();
          vector<View_Sc4> bbasis, bbasis_grad;
          for (size_t i=0; i<groupData[block]->basis_pointers.size(); i++) {
            int numb = groupData[block]->basis_pointers[i]->getCardinality();
            View_Sc4 basis_vals, basis_grad_vals;
            if (groupData[block]->basis_types[i].substr(0,5) == "HGRAD"){
              basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,1);
              basis_grad_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,groupData[block]->dimension);
            }
            else if (groupData[block]->basis_types[i].substr(0,4) == "HVOL"){ // does not require orientations
              basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,1);
            }
            else if (groupData[block]->basis_types[i].substr(0,5) == "HFACE"){
              basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,1);
            }
            else if (groupData[block]->basis_types[i].substr(0,4) == "HDIV"){
              basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,groupData[block]->dimension);
            }
            else if (groupData[block]->basis_types[i].substr(0,5) == "HCURL"){
              basis_vals = View_Sc4("basis vals",database_boundary_numElem,numb,numsideip,groupData[block]->dimension);
            }
            bbasis.push_back(basis_vals);
            bbasis_grad.push_back(basis_grad_vals);
          }
          for (size_t e=0; e<first_boundary_users.size(); ++e) {
            size_t refgrp = first_boundary_users[e].first;
            size_t refelem = first_boundary_users[e].second;

            DRV database_bnodes("nodes for the database", 1, groups[block][0]->nodes.extent(1), dimension);
            Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> database_borientation("database orientations", 1);
        
            auto database_bnodes_host = create_mirror_view(database_bnodes);
            auto database_borientation_host = create_mirror_view(database_borientation);
          
            LO localSideID = boundary_groups[block][refgrp]->localSideID;
            // Get the nodes on the host
            auto sub_nodes = subview(boundary_groups[block][refgrp]->nodes, refelem, ALL(), ALL());
            auto sub_nodes_host = create_mirror_view(sub_nodes);
            deep_copy(sub_nodes_host, sub_nodes);

            for (size_type node=0; node<database_bnodes.extent(1); ++node) {
              for (size_type dim=0; dim<database_bnodes.extent(2); ++dim) {
                database_bnodes_host(0,node,dim) = sub_nodes_host(node,dim);
              }
            }
            deep_copy(database_bnodes, database_bnodes_host);

            // Get the orientations on the host
            auto orientations_host = create_mirror_view(boundary_groups[block][refgrp]->orientation);
            deep_copy(orientations_host, boundary_groups[block][refgrp]->orientation);
            database_borientation_host(0) = orientations_host(refelem);
            deep_copy(database_borientation, database_borientation_host);
            
            vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl;
            vector<View_Sc3> tbasis_div;
            disc->getPhysicalBoundaryBasis(groupData[block], database_bnodes, localSideID, database_borientation,
                                           tbasis, tbasis_grad, tbasis_curl, tbasis_div);

            for (size_t i=0; i<groupData[block]->basis_pointers.size(); i++) {
              auto tbasis_slice = subview(tbasis[i],0,ALL(),ALL(),ALL());
              auto bbasis_slice = subview(bbasis[i],e,ALL(),ALL(),ALL());
              deep_copy(bbasis_slice, tbasis_slice);
            }
          }
                  
          groupData[block]->database_side_basis = bbasis;
          groupData[block]->database_side_basis_grad = bbasis_grad;
        
          
        }
      }
    }
  }    

  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      groups[block][grp]->computeBasis(keepnodes);
    }
  }
  
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      boundary_groups[block][grp]->computeBasis(keepnodes);
    }
  }
  
  
  // ==============================================
  // Inform the user how many groups are on
  // each processor and much memory is utilized by
  // the groups
  // ==============================================
  
  if (verbosity > 5) {
    
    // Volumetric elements
    size_t numelements = 0;
    double minsize = 1e100;
    double maxsize = 0.0;
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        numelements += groups[block][grp]->numElem;
        auto wts = groups[block][grp]->wts;
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
    cout << " - Processor " << Comm->getRank() << " has " << numelements << " elements" << endl;
    cout << " - Processor " << Comm->getRank() << " min element size: " << minsize << endl;
    cout << " - Processor " << Comm->getRank() << " max element size: " << maxsize << endl;
    
    // Boundary elements
    size_t numbndryelements = 0;
    double minbsize = 1e100;
    double maxbsize = 0.0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        numbndryelements += boundary_groups[block][grp]->numElem;
        //if (boundary_groups[block][grp]->storeAll) {
          auto wts = boundary_groups[block][grp]->wts;
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
        //}
      }
    }
    cout << " - Processor " << Comm->getRank() << " has " << numbndryelements << " boundary elements" << endl;
    cout << " - Processor " << Comm->getRank() << " min boundary element size: " << minbsize << endl;
    cout << " - Processor " << Comm->getRank() << " max boundary element size: " << maxbsize << endl;
    
    // Volumetric ip/basis
    size_t groupstorage = 0;
    for (size_t block=0; block<groups.size(); ++block) {
      if (groupData[block]->use_basis_database) {
        groupstorage += groupData[block]->getDatabaseStorage();
      }
      else {
        for (size_t grp=0; grp<groups[block].size(); ++grp) {
          groupstorage += groups[block][grp]->getVolumetricStorage();
        }
      }
    }
    double totalstorage = static_cast<double>(groupstorage)/1.0e6;
    cout << " - Processor " << Comm->getRank() << " is using " << totalstorage << " MB to store volumetric data" << endl;
    
    // Face ip/basis
    size_t facestorage = 0;
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        facestorage += groups[block][grp]->getFaceStorage();
      }
    }
    totalstorage = static_cast<double>(facestorage)/1.0e6;
    cout << " - Processor " << Comm->getRank() << " is using " << totalstorage << " MB to store face data" << endl;
    
    // Boundary ip/basis
    size_t boundarystorage = 0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        boundarystorage += boundary_groups[block][grp]->getStorage();
      }
    }
    totalstorage = static_cast<double>(boundarystorage)/1.0e6;
    cout << " - Processor " << Comm->getRank() << " is using " << totalstorage << " MB to store boundary data" << endl;
  }
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::allocategroupstorage" << endl;
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
  
  for (size_t block=0; block<groups.size(); ++block) {
    if (groups[block].size() > 0) {
      vector<int> info;
      info.push_back(groupData[block]->dimension);
      info.push_back((int)groupData[block]->numDiscParams);
      info.push_back(groupData[block]->numElem);
      info.push_back(groupData[block]->numip);
      info.push_back(groupData[block]->numsideip);
      info.push_back(phys->setnames.size());
      vector<size_t> numVars;
      for (size_t set=0; set<groupData[block]->set_numDOF.size(); ++set) {
        numVars.push_back(groupData[block]->set_numDOF[set].extent(0));
      }
      vector<Kokkos::View<string**,HostDevice> > bcs(phys->setnames.size());
      for (size_t set=0; set<phys->setnames.size(); ++set) {
        Kokkos::View<string**,HostDevice> vbcs = disc->getVarBCs(set,block);
        bcs[set] = vbcs;
      }
      wkset.push_back(Teuchos::rcp( new workset(info,
                                                numVars,
                                                isTransient,
                                                disc->basis_types[block],
                                                disc->basis_pointers[block],
                                                params->discretized_param_basis,
                                                groupData[block]->cellTopo)));
                                                //mesh->cellTopo[block]) ) );
      wkset[block]->block = block;
      wkset[block]->set_var_bcs = bcs;
      wkset[block]->var_bcs = bcs[0];
    }
    else {
      wkset.push_back(Teuchos::rcp( new workset()));
      wkset[block]->isInitialized = false;
      wkset[block]->block = block;
    }
      // this needs to be done even for uninitialized worksets
      // initialize BDF_wts vector (empty views)
      vector<Kokkos::View<ScalarT*,AssemblyDevice> > tmpBDF_wts(phys->setnames.size());
      wkset[block]->set_BDF_wts = tmpBDF_wts;
      // initialize Butcher tableau vectors (empty views);
      vector<Kokkos::View<ScalarT**,AssemblyDevice> > tmpbutcher_A(phys->setnames.size());
      vector<Kokkos::View<ScalarT*,AssemblyDevice> > tmpbutcher_b(phys->setnames.size());
      vector<Kokkos::View<ScalarT*,AssemblyDevice> > tmpbutcher_c(phys->setnames.size());
      wkset[block]->set_butcher_A = tmpbutcher_A;
      wkset[block]->set_butcher_b = tmpbutcher_b;
      wkset[block]->set_butcher_c = tmpbutcher_c;
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
  
  for (size_t block=0; block<groups.size(); block++) {
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
                                       const size_t & block, const size_t & groupblock) {

  typedef typename Node::execution_space LA_exec;
  using namespace std;
  
  bool use_atomics_ = false;
  if (LA_exec::concurrency() > 1) {
    use_atomics_ = true;
  }
  
  bool fix_zero_rows = true;
  
  auto localMatrix = mass->getLocalMatrixHost();
  auto rhs_view = rhs->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  bool lump_mass_ = lump_mass;

  wkset[block]->updatePhysicsSet(set);
  groupData[block]->updatePhysicsSet(set);
  
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->numDOF;

  for (size_t grp=0; grp<groups[groupblock].size(); ++grp) {

    auto LIDs = groups[groupblock][grp]->LIDs[set];
    
    auto localrhs = groups[groupblock][grp]->getInitial(true, useadjoint);
    auto localmass = groups[groupblock][grp]->getMass();

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
  
  for (size_t block=0; block<groups.size(); ++block) {
    
    auto offsets = wkset[block]->offsets;
    auto numDOF = groupData[block]->numDOF;
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      
      auto LIDs = groups[block][grp]->LIDs[set];
      
      Kokkos::View<ScalarT***,AssemblyDevice> localmass = groups[block][grp]->getWeightedMass(phys->masswts[set][block]);
      
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
            if (use_jacobi) {
              val = localmass(elem,row,row);
            }
            else {
              for (int k=0; k<numDOF(n); k++) {
                int col = offsets(n,k);
                val += localmass(elem,row,col);
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
    auto numDOF = groupData[block]->numDOF;
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      
      auto cLIDs = groups[block][grp]->LIDs[set];
      
      if (!groups[block][grp]->storeMass) { //groupData[block]->store_mass) { //groupData->matrix_free) {
        auto twts = groups[block][grp]->wts;
        vector<View_Sc4> tbasis;
        if (groups[block][grp]->storeAll) { // unlikely case, but enabled
          tbasis = groups[block][grp]->basis;
        }
        else {
          disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->nodes,
                                           groups[block][grp]->orientation, tbasis);
        }
        
        for (size_type var=0; var<numDOF.extent(0); var++) {
          int bindex = wkset[block]->usebasis[var];
          View_Sc4 cbasis = tbasis[bindex];
          
          string btype = wkset[block]->basis_types[bindex];
          auto off = subview(offsets,var,ALL());
          ScalarT mwt = phys->masswts[set][block][var];
          
          if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
            parallel_for("get mass",
                         RangePolicy<AssemblyExec>(0,twts.extent(0)),
                         KOKKOS_LAMBDA (const size_type e ) {
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
                         KOKKOS_LAMBDA (const size_type e ) {
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
          auto curr_mass = groupData[block]->database_mass[set];
          auto index = groups[block][grp]->basis_index;
          parallel_for("get mass",
                       RangePolicy<AssemblyExec>(0,index.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            LO eindex = index(elem);
            for (size_type var=0; var<numDOF.extent(0); var++) {
              for (int i=0; i<numDOF(var); i++ ) {
                for (int j=0; j<numDOF(var); j++ ) {
                  LO indi = cLIDs(elem,offsets(var,i));
                  LO indj = cLIDs(elem,offsets(var,j));
                  if (use_atomics_) {
                    Kokkos::atomic_add(&(y_slice(indi)), curr_mass(eindex,offsets(var,i),offsets(var,j))*x_slice(indj));
                  }
                  else {
                    y_slice(indi) += curr_mass(eindex,offsets(var,i),offsets(var,j))*x_slice(indj);
                  }
                }
              }
            }
          });
          
        }
        else {
          auto curr_mass = groups[block][grp]->local_mass[set];
          parallel_for("get mass",
                       RangePolicy<AssemblyExec>(0,curr_mass.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
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
  
  Teuchos::TimeMonitor localtimer(*setinittimer);
  
  typedef typename Node::execution_space LA_exec;
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::getWeightVector ..." << endl;
    }
  }
  
  auto wts_view = wts->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  vector<vector<ScalarT> > normwts = phys->normwts[set];
  
  for (size_t block=0; block<groups.size(); ++block) {
    
    auto offsets = wkset[block]->offsets;
    auto numDOF = groupData[block]->numDOF;
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      
      for (size_type n=0; n<numDOF.extent(0); ++n) {
      
        ScalarT val = normwts[block][n];
        auto LIDs = groups[block][grp]->LIDs[set];
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

  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      LIDView_host LIDs = groups[block][grp]->LIDs_host[set];
      Kokkos::View<ScalarT**,AssemblyDevice> localinit = groups[block][grp]->getInitial(false, useadjoint);
      auto host_init = Kokkos::create_mirror_view(localinit);
      Kokkos::deep_copy(host_init,localinit);
      int numElem = groups[block][grp]->numElem;
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

      auto localrhs = boundary_groups[block][grp]->getDirichlet(set);
      auto localmass = boundary_groups[block][grp]->getMass(set);
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
  
  auto localMatrix = mass->getLocalMatrixHost();
  
  for (size_t block=0; block<groups.size(); ++block) {
    wkset[block]->isOnSide = true;
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      auto LIDs = groups[block][grp]->LIDs_host[set];
      // Get the requested IC from the group
      auto localrhs = groups[block][grp]->getInitialFace(true);
      // Create the mass matrix
      auto localmass = groups[block][grp]->getMassFace();
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
    wkset[block]->isOnSide = false;
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

  
  for (size_t block=0; block<groups.size(); ++block) {
    if (groups[block].size() > 0) {
      this->assembleJacRes(set, compute_jacobian,
                           compute_sens, compute_disc_sens, res, J, isTransient,
                           current_time, useadjoint, store_adjPrev, num_active_params,
                           is_final_time, block, deltat);
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
                                           const int & block, const ScalarT & deltat) {
  
  Teuchos::TimeMonitor localassemblytimer(*assemblytimer);
  
  // Kokkos::CRSMatrix and Kokkos::View for J and res
  // Scatter needs to be on LA_device
  typedef typename Tpetra::CrsMatrix<ScalarT, LO, GO, Node >::local_matrix_device_type local_matrix;
  local_matrix J_kcrs;
  if (compute_jacobian) {
    J_kcrs = J->getLocalMatrixDevice();
  }
  
  auto res_view = res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
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
  if (!data_avail || useadjoint || groupData[block]->multiscale || compute_disc_sens || compute_sens) {
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
    auto butcher_c = Kokkos::create_mirror_view(wkset[block]->butcher_c);
    Kokkos::deep_copy(butcher_c, wkset[block]->butcher_c);
    ScalarT timeval = current_time + butcher_c(wkset[block]->current_stage)*deltat;
    
    wkset[block]->setTime(timeval);
    wkset[block]->setDeltat(deltat);
    wkset[block]->alpha = 1.0/deltat;
  }
  
  wkset[block]->isTransient = isTransient;
  wkset[block]->isAdjoint = useadjoint;
  
  int numElem = groupData[block]->numElem;
  int numDOF = groups[block][0]->LIDs[set].extent(1);
  
  int numParamDOF = 0;
  if (compute_disc_sens) {
    numParamDOF = groups[block][0]->paramLIDs.extent(1);
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
  
  // Note: Cannot parallelize over groups since data structures are re-used
  
  for (size_t grp=0; grp<groups[block].size(); ++grp) {

    wkset[block]->localEID = grp;
    
    if (isTransient && useadjoint && !groups[block][0]->groupData->multiscale) {
      if (is_final_time) {
        groups[block][grp]->resetAdjPrev(set,0.0);
      }
    }
 
    /////////////////////////////////////////////////////////////////////////////
    // Compute the local residual and Jacobian on this group
    /////////////////////////////////////////////////////////////////////////////
    
    bool fixJacDiag = false;
    
    {
      Teuchos::TimeMonitor localtimer(*phystimer);
      
      //////////////////////////////////////////////////////////////
      // Compute res and J=dF/du
      //////////////////////////////////////////////////////////////
      
      // Volumetric contribution
      if (assemble_volume_terms[set][block]) {
        if (groupData[block]->multiscale) {
          int sgindex = groups[block][grp]->subgrid_model_index[groups[block][grp]->subgrid_model_index.size()-1];
          groups[block][grp]->subgridModels[sgindex]->subgridSolver(groups[block][grp]->u[set], groups[block][grp]->phi[set], 
                                                                    wkset[block]->time, isTransient, useadjoint,
                                                                    compute_jacobian, compute_sens, num_active_params,
                                                                    compute_disc_sens, false,
                                                                    *(wkset[block]), groups[block][grp]->subgrid_usernum, 0,
                                                                    groups[block][grp]->subgradient, store_adjPrev);
          fixJacDiag = true;
        }
        else {
          groups[block][grp]->updateWorkset(seedwhat);
          phys->volumeResidual(set,block);
        }
      }
      
      ///////////////////////////////////////////////////////////////////////////
      // Edge/face contribution
      ///////////////////////////////////////////////////////////////////////////
      
      if (assemble_face_terms[set][block]) {
        if (groupData[block]->multiscale) {
          // do nothing
        }
        else {
          wkset[block]->isOnSide = true;
          for (size_t s=0; s<groupData[block]->numSides; s++) {
            groups[block][grp]->updateWorksetFace(s);
            phys->faceResidual(set,block);
          }
          wkset[block]->isOnSide =false;
        }
      }
      
    }
        
    ///////////////////////////////////////////////////////////////////////////
    // Scatter into global matrix/vector
    ///////////////////////////////////////////////////////////////////////////
    
    if (reduce_memory) { // skip local_res and local_J
      this->scatter(set, J_kcrs, res_view,
                    groups[block][grp]->LIDs[set], groups[block][grp]->paramLIDs, block,
                    compute_jacobian, compute_sens, compute_disc_sens, useadjoint);
    }
    else { // fill local_res and local_J and then scatter
    
      Teuchos::TimeMonitor localtimer(*scattertimer);
      
      Kokkos::deep_copy(local_res,0.0);
      Kokkos::deep_copy(local_J,0.0);
      
      // Use AD residual to update local Jacobian
      if (compute_jacobian) {
        if (compute_disc_sens) {
          groups[block][grp]->updateParamJac(local_J);
        }
        else {
          groups[block][grp]->updateJac(useadjoint, local_J);
        }
      }
      
      if (compute_jacobian && fixJacDiag) {
        groups[block][grp]->fixDiagJac(local_J, local_res);
      }
      
      // Update the local residual
      
      if (useadjoint) {
        groups[block][grp]->updateAdjointRes(compute_jacobian, isTransient,
                                      false, store_adjPrev,
                                      local_J, local_res);
      }
      else {
        groups[block][grp]->updateRes(compute_sens, local_res);
      }
      
      // Now scatter from local_res and local_J
      
      if (data_avail) {
        this->scatterRes(res_view, local_res, groups[block][grp]->LIDs[set]);
        if (compute_jacobian) {
          this->scatterJac(set, J_kcrs, local_J, groups[block][grp]->LIDs[set], groups[block][grp]->paramLIDs, compute_disc_sens);
        }
      }
      else {
        auto local_res_ladev = create_mirror(LA_exec(),local_res);
        auto local_J_ladev = create_mirror(LA_exec(),local_J);
        
        Kokkos::deep_copy(local_J_ladev,local_J);
        Kokkos::deep_copy(local_res_ladev,local_res);
        
        if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
          this->scatterRes(res_view, local_res_ladev, groups[block][grp]->LIDs_host[set]);
          if (compute_jacobian) {
            this->scatterJac(set, J_kcrs, local_J_ladev, groups[block][grp]->LIDs_host[set], groups[block][grp]->paramLIDs_host, compute_disc_sens);
          }
          
        }
        else { // LA_device = CUDA, AssemblyDevice = Host
          // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
          auto LIDs_dev = Kokkos::create_mirror(LA_exec(), groups[block][grp]->LIDs[set]);
          auto paramLIDs_dev = Kokkos::create_mirror(LA_exec(), groups[block][grp]->paramLIDs);
          Kokkos::deep_copy(LIDs_dev,groups[block][grp]->LIDs[set]);
          Kokkos::deep_copy(paramLIDs_dev,groups[block][grp]->paramLIDs);
          
          this->scatterRes(res_view, local_res_ladev, LIDs_dev);
          if (compute_jacobian) {
            this->scatterJac(set, J_kcrs, local_J_ladev, LIDs_dev, paramLIDs_dev, compute_disc_sens);
          }
        }
        
      }
    }
    
  } // group loop
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Boundary terms
  //////////////////////////////////////////////////////////////////////////////////////
  
  if (assemble_boundary_terms[set][block]) {
    
    wkset[block]->isOnSide = true;

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
    
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      
      if (boundary_groups[block][grp]->numElem > 0) {
        
        /////////////////////////////////////////////////////////////////////////////
        // Compute the local residual and Jacobian on this boundary group
        /////////////////////////////////////////////////////////////////////////////
        wkset[block]->resetResidual();
        boundary_groups[block][grp]->updateWorkset(seedwhat);
        
        if (!groupData[block]->multiscale) {
          Teuchos::TimeMonitor localtimer(*phystimer);
          phys->boundaryResidual(set,block);
        }
        
        {
          phys->fluxConditions(set,block);
        }
        ///////////////////////////////////////////////////////////////////////////
        // Scatter into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        if (reduce_memory) { // skip local_res and local_J
          this->scatter(set, J_kcrs, res_view,
                        boundary_groups[block][grp]->LIDs[set], boundary_groups[block][grp]->paramLIDs, block,
                        compute_jacobian, compute_sens, compute_disc_sens, useadjoint);
        }
        else { // fill local_res and local_J and then scatter
        
          Teuchos::TimeMonitor localtimer(*scattertimer);
          
          Kokkos::deep_copy(local_res,0.0);
          Kokkos::deep_copy(local_J,0.0);
        
          // Use AD residual to update local Jacobian
          if (compute_jacobian) {
            if (compute_disc_sens) {
              boundary_groups[block][grp]->updateParamJac(local_J);
            }
            else {
              boundary_groups[block][grp]->updateJac(useadjoint, local_J);
            }
          }
          
          // Update the local residual (forward mode)
          if (!useadjoint) {
            boundary_groups[block][grp]->updateRes(compute_sens, local_res);
          }
         
          if (data_avail) {
            this->scatterRes(res_view, local_res, boundary_groups[block][grp]->LIDs[set]);
            if (compute_jacobian) {
              this->scatterJac(set, J_kcrs, local_J, boundary_groups[block][grp]->LIDs[set], boundary_groups[block][grp]->paramLIDs, compute_disc_sens);
            }
          }
          else {
            auto local_res_ladev = create_mirror(LA_exec(),local_res);
            auto local_J_ladev = create_mirror(LA_exec(),local_J);
            
            Kokkos::deep_copy(local_J_ladev,local_J);
            Kokkos::deep_copy(local_res_ladev,local_res);
            
            if (use_host_LIDs) { // LA_device = Host, AssemblyDevice = CUDA (no UVM)
              this->scatterRes(res_view, local_res_ladev, boundary_groups[block][grp]->LIDs_host[set]);
              if (compute_jacobian) {
                this->scatterJac(set, J_kcrs, local_J_ladev,
                                 boundary_groups[block][grp]->LIDs_host[set], boundary_groups[block][grp]->paramLIDs_host,
                                 compute_disc_sens);
              }
            }
            else { // LA_device = CUDA, AssemblyDevice = Host
              // TMW: this should be a very rare instance, so we are just being lazy and copying the data here
              auto LIDs_dev = Kokkos::create_mirror(LA_exec(), boundary_groups[block][grp]->LIDs[set]);
              auto paramLIDs_dev = Kokkos::create_mirror(LA_exec(), boundary_groups[block][grp]->paramLIDs);
              Kokkos::deep_copy(LIDs_dev,boundary_groups[block][grp]->LIDs[set]);
              Kokkos::deep_copy(paramLIDs_dev,boundary_groups[block][grp]->paramLIDs);
              
              this->scatterRes(res_view, local_res_ladev, LIDs_dev);
              if (compute_jacobian) {
                this->scatterJac(set, J_kcrs, local_J_ladev, LIDs_dev, paramLIDs_dev, compute_disc_sens);
              }
            }
            
          }
        }
        
      }
    } // element loop
    wkset[block]->isOnSide = false;
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
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      groups[block][grp]->resetPrevSoln(set);
    }
  }
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      boundary_groups[block][grp]->resetPrevSoln(set);
    }
  }
}

template<class Node>
void AssemblyManager<Node>::revertSoln(const size_t & set) {
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      groups[block][grp]->revertSoln(set);
    }
  }
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      boundary_groups[block][grp]->revertSoln(set);
    }
  }
}

template<class Node>
void AssemblyManager<Node>::resetStageSoln(const size_t & set) {
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      groups[block][grp]->resetStageSoln(set);
    }
  }
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      boundary_groups[block][grp]->resetStageSoln(set);
    }
  }
}

template<class Node>
void AssemblyManager<Node>::updateStage(const int & stage, const ScalarT & current_time,
                                        const ScalarT & deltat) {
  for (size_t block=0; block<wkset.size(); ++block) {
    wkset[block]->setStage(stage);
    auto butcher_c = Kokkos::create_mirror_view(wkset[block]->butcher_c);
    Kokkos::deep_copy(butcher_c, wkset[block]->butcher_c);
    ScalarT timeval = current_time + butcher_c(stage)*deltat;
    wkset[block]->setTime(timeval);
    wkset[block]->setDeltat(deltat);
    wkset[block]->alpha = 1.0/deltat;
  }
  
}

template<class Node>
void AssemblyManager<Node>::updateStageSoln(const size_t & set)  {
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      groups[block][grp]->updateStageSoln(set);
    }
  }
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      boundary_groups[block][grp]->updateStageSoln(set);
    }
  }
}

// ========================================================================================
// Gather local solutions on groups.
// This intermediate function allows us to copy the data from LA_device to AssemblyDevice only once (if necessary)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::performGather(const size_t & set, const vector_RCP & vec, const int & type, const size_t & entry) {
  
  typedef typename LA_device::memory_space LA_mem;
  
  auto vec_kv = vec->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
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
  
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      switch(type) {
        case 0 :
          LIDs = groups[block][grp]->LIDs[set];
          numDOF = groups[block][grp]->groupData->numDOF;
          data = groups[block][grp]->u[set];
          offsets = wkset[block]->offsets;
          break;
        case 1 : // deprecated (u_dot)
          break;
        case 2 :
          LIDs = groups[block][grp]->LIDs[set];
          numDOF = groups[block][grp]->groupData->numDOF;
          data = groups[block][grp]->phi[set];
          offsets = wkset[block]->offsets;
          break;
        case 3 : // deprecated (phi_dot)
          break;
        case 4:
          LIDs = groups[block][grp]->paramLIDs;
          numDOF = groups[block][grp]->groupData->numParamDOF;
          data = groups[block][grp]->param;
          offsets = wkset[block]->paramoffsets;
          break;
        default :
          cout << "ERROR - NOTHING WAS GATHERED" << endl;
      }
      
      parallel_for("assembly gather",
                   RangePolicy<AssemblyExec>(0,data.extent(0)), 
                   KOKKOS_LAMBDA (const int elem ) {
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
            numDOF = boundary_groups[block][grp]->groupData->numDOF;
            data = boundary_groups[block][grp]->u[set];
            offsets = wkset[block]->offsets;
            break;
          case 1 : // deprecated (u_dot)
            break;
          case 2 :
            LIDs = boundary_groups[block][grp]->LIDs[set];
            numDOF = boundary_groups[block][grp]->groupData->numDOF;
            data = boundary_groups[block][grp]->phi[set];
            offsets = wkset[block]->offsets;
            break;
          case 3 : // deprecated (phi_dot)
            break;
          case 4:
            LIDs = boundary_groups[block][grp]->paramLIDs;
            numDOF = boundary_groups[block][grp]->groupData->numParamDOF;
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
  auto numDOF = groupData[block]->numDOF;
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
      groupData[block]->updatePhysicsSet(set);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we can get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::purgeMemory() {
  // nothing here
}


template class MrHyDE::AssemblyManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
  template class MrHyDE::AssemblyManager<SubgridSolverNode>;
#endif
