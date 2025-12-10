/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// Create the groups of elements/cells
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createGroups() {
  
  Teuchos::TimeMonitor localtimer(*group_timer);
  
  debugger->print("**** Starting AssemblyManager::createGroups ...");
  
  double storageProportion = settings->sublist("Solver").get<double>("storage proportion",1.0);
  
  vector<stk::mesh::Entity> all_meshElems = mesh->getMySTKElements();
  
  
  auto LIDs = disc->dof_lids;
  
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
    
    vector<stk::mesh::Entity> stk_meshElems = mesh->getMySTKElements(blocknames[block]);
    
    topo_RCP cellTopo = mesh->getCellTopology(blocknames[block]);
    
    size_t numTotalElem;
    if(mesh->use_stk_mesh)
      numTotalElem = stk_meshElems.size();
    else
      numTotalElem = mesh->simple_mesh->getNumCells();
    size_t processedElem = 0;
    
    if (numTotalElem>0) {
      
      auto myElem = disc->my_elements[block];
      Kokkos::View<LO*,AssemblyDevice> eIDs("local element IDs on device",myElem.size());
      auto host_eIDs = Kokkos::create_mirror_view(eIDs);
      for (size_t elem=0; elem<myElem.size(); elem++) {
        host_eIDs(elem) = static_cast<LO>(myElem(elem));
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
      
      vector<string> sideSets = mesh->getSideNames();
      vector<bool> aface;
      for (size_t set=0; set<assemble_face_terms.size(); ++set) {
        aface.push_back(assemble_face_terms[set][block]);
      }
      blockGroupData = Teuchos::rcp( new GroupMetaData(settings, cellTopo,
                                                       physics, block, 0, elemPerGroup,
                                                       build_face_terms[block],
                                                       aface, sideSets,
                                                       params->num_discretized_params));
      
      disc->setReferenceData(blockGroupData);
      
      blockGroupData->require_basis_at_nodes = settings->sublist("Postprocess").get<bool>("plot solution at nodes",false);
      bool write_solution = settings->sublist("Postprocess").get("write solution",false);
      
      // if any of the discretizations are greater than 1st order and the user requests output, override the input to plot solution at nodes
      // TMW: modified to only override for HGRAD basis'.  HCURL and HDIV only plot cell averages.
      for (size_t i_basis=0; i_basis<physics->unique_orders[block].size(); ++i_basis) {
        if (physics->unique_orders[block][i_basis] > 1 && write_solution) {
          if (physics->unique_types[block][i_basis] == "HGRAD") {
            blockGroupData->require_basis_at_nodes = true;
          }
        }
      }
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
      
      blockGroupData->set_num_dof = set_numDOF;
      blockGroupData->set_num_dof_host = set_numDOF_host;
      
      blockGroupData->num_dof = set_numDOF[0];
      blockGroupData->num_dof_host = set_numDOF_host[0];
      
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
          
          vector<stk::mesh::Entity> sideEntities = mesh->getMySTKSides(sideName, blocknames[block]);
          
          vector<size_t>             local_side_Ids;
          vector<stk::mesh::Entity> side_output;
          vector<size_t>             local_elem_Ids;
          
          
          mesh->getSTKSideElements(blocknames[block], sideEntities, local_side_Ids, side_output);
          DRV sidenodes;
          mesh->getSTKElementVertices(side_output, blocknames[block], sidenodes);
          
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
                DRV currnodes("currnodes", currElem, mesh->num_nodes_per_elem, mesh->dimension);
                
                auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
                Kokkos::View<LO*,HostDevice> host_eIndex2("element indices",currElem);
                auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
                LO sideIndex;
                
                for (size_t e=0; e<currElem; e++) {
                  host_eIndex(e) = mesh->getSTKElementLocalId(side_output[group[e+prog]]);
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
                
                // Build the Kokkos View of the group LIDs ------
                vector<LIDView> set_LIDs;
                for (size_t set=0; set<LIDs.size(); ++set) {
                  LIDView groupLIDs("LIDs",currElem,LIDs[set].extent(1));
                  auto currLIDs = LIDs[set];
                  parallel_for("assembly copy LIDs bgrp",
                               RangePolicy<AssemblyExec>(0,groupLIDs.extent(0)),
                               MRHYDE_LAMBDA (const int e ) {
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
                
                block_boundary_groups.push_back(Teuchos::rcp(new BoundaryGroup(blockGroupData, eIndex, currnodes, sideIndex,
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
        
        auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
        Kokkos::View<LO*,HostDevice> host_eIndex2("element indices on host",currElem);
        
        for (size_t e=0; e<currElem; ++e) {
          host_eIndex(e) = elem_groups[grp][e];
        }
        Kokkos::deep_copy(eIndex,host_eIndex);
        Kokkos::deep_copy(host_eIndex2,host_eIndex);
        vector<LIDView> set_LIDs;
        for (size_t set=0; set<LIDs.size(); ++set) {
          auto currLIDs = LIDs[set];
          LIDView groupLIDs("LIDs on device",currElem,currLIDs.extent(1));
          parallel_for("assembly copy nodes",
                       RangePolicy<AssemblyExec>(0,eIndex.extent(0)),
                       MRHYDE_LAMBDA (const int e ) {
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
        
        // Set the side information (soon to be removed)-
        vector<Kokkos::View<int****,HostDevice> > set_sideinfo;
        if(mesh->use_stk_mesh) {
          for (size_t set=0; set<LIDs.size(); ++set) {
            Kokkos::View<int****,HostDevice> sideinfo = disc->getSideInfo(set,block,host_eIndex2);
            set_sideinfo.push_back(sideinfo);
          }
        }

        if (blockGroupData->multiscale || store_nodes) {
          DRV currnodes("currnodes", currElem, mesh->num_nodes_per_elem, mesh->dimension);
          mesh->getSTKElementVertices(local_grp, blocknames[block], currnodes);
          block_groups.push_back(Teuchos::rcp(new Group(blockGroupData, eIndex, currnodes,
                                                        disc, storeThis)));
        }
        else {
          block_groups.push_back(Teuchos::rcp(new Group(blockGroupData, eIndex,
                                                        disc, storeThis)));
        }
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
  
  debugger->print("**** Starting AssemblyManager::allocateGroupStorage");
  
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
      this->buildDatabase(block);
    }
    else {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        groups[block][grp]->initializeBasisIndex();
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
    //double minsize = 1e100;
    //double maxsize = 0.0;
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        numelements += groups[block][grp]->numElem;
        //auto wts = groups[block][grp]->wts;
        //auto host_wts = wts;//create_mirror_view(wts);
        //deep_copy(host_wts,wts);
        //for (size_type e=0; e<host_wts.extent(0); ++e) {
        //  double currsize = 0.0;
        //  for (size_type pt=0; pt<host_wts.extent(1); ++pt) {
        //    currsize += host_wts(e,pt);
        //  }
        //  maxsize = std::max(currsize,maxsize);
        //  minsize = std::min(currsize,minsize);
        //}
      }
    }
    cout << " - Processor " << comm->getRank() << " has " << numelements << " elements" << endl;
    //cout << " - Processor " << comm->getRank() << " min element size: " << minsize << endl;
    //cout << " - Processor " << comm->getRank() << " max element size: " << maxsize << endl;
    
    // Boundary elements
    size_t numbndryelements = 0;
    //double minbsize = 1e100;
    //double maxbsize = 0.0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        numbndryelements += boundary_groups[block][grp]->numElem;
        //if (boundary_groups[block][grp]->storeAll) {
        //auto wts = boundary_groups[block][grp]->wts;
        //auto host_wts = create_mirror_view(wts);
        //deep_copy(host_wts,wts);
        //for (size_type e=0; e<host_wts.extent(0); ++e) {
        //  double currsize = 0.0;
        //  for (size_type pt=0; pt<host_wts.extent(1); ++pt) {
        //    currsize += host_wts(e,pt);
        //  }
        //  maxbsize = std::max(currsize,maxbsize);
        //  minbsize = std::min(currsize,minbsize);
        //}
        //}
      }
    }
    cout << " - Processor " << comm->getRank() << " has " << numbndryelements << " boundary elements" << endl;
    //cout << " - Processor " << comm->getRank() << " min boundary element size: " << minbsize << endl;
    //cout << " - Processor " << comm->getRank() << " max boundary element size: " << maxbsize << endl;
    
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
    cout << " - Processor " << comm->getRank() << " is using " << totalstorage << " MB to store volumetric data" << endl;
    
    // Face ip/basis
    size_t facestorage = 0;
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        facestorage += groups[block][grp]->getFaceStorage();
      }
    }
    totalstorage = static_cast<double>(facestorage)/1.0e6;
    cout << " - Processor " << comm->getRank() << " is using " << totalstorage << " MB to store face data" << endl;
    
    // Boundary ip/basis
    size_t boundarystorage = 0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        boundarystorage += boundary_groups[block][grp]->getStorage();
      }
    }
    totalstorage = static_cast<double>(boundarystorage)/1.0e6;
    cout << " - Processor " << comm->getRank() << " is using " << totalstorage << " MB to store boundary data" << endl;
  }
  
  debugger->print("**** Finished AssemblyManager::allocategroupstorage");
}


///////////////////////////////////////////////////////////////////////////////////////
// Pass the Group data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateGroupData(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp) {
  
  // hard coded for what I need it for right now
  if (groupData[block]->have_phi) {
    wset->have_rotation_phi = true;
    wset->rotation_phi = groups[block][grp]->data;
    wset->allocateRotations();
  }
  else if (groupData[block]->have_rotation) {
    wset->have_rotation = true;
    wset->allocateRotations();
    auto rot = wset->rotation;
    auto data = groups[block][grp]->data;

    parallel_for("Group update data",
                 RangePolicy<AssemblyExec>(0,data.extent(0)),
                 MRHYDE_LAMBDA (const size_type e ) {
      rot(e,0,0) = data(e,0);
      rot(e,0,1) = data(e,1);
      rot(e,0,2) = data(e,2);
      rot(e,1,0) = data(e,3);
      rot(e,1,1) = data(e,4);
      rot(e,1,2) = data(e,5);
      rot(e,2,0) = data(e,6);
      rot(e,2,1) = data(e,7);
      rot(e,2,2) = data(e,8);
    });

    //KokkosTools::print(wkset[block]->rotation);
    //KokkosTools::print(rot);

  }
  else if (groupData[block]->have_extra_data) {
    wset->extra_data = groups[block][grp]->data;
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

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

