/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "assemblyManager.hpp"

// Remove this when done testing
#include "Intrepid2_CellTools.hpp"


using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class Node>
AssemblyManager<Node>::AssemblyManager(const Teuchos::RCP<MpiComm> & comm_,
                                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                       Teuchos::RCP<MeshInterface> & mesh_,
                                       Teuchos::RCP<DiscretizationInterface> & disc_,
                                       Teuchos::RCP<PhysicsInterface> & physics_,
                                       Teuchos::RCP<ParameterManager<Node>> & params_) :
comm(comm_), settings(settings_), mesh(mesh_), disc(disc_), physics(physics_), params(params_) {
  
  RCP<Teuchos::Time> constructor_time = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager - constructor");
  Teuchos::TimeMonitor constructor_timer(*constructor_time);
  
  // Get the required information from the settings
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
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
  allow_autotune = settings->sublist("Solver").get<bool>("enable autotune",true);
  
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
  blocknames = mesh->getBlockNames();
  
  // check if we need to assembly volumetric, boundary and face terms
  for (size_t set=0; set<physics->set_names.size(); ++set) {
    vector<bool> set_assemble_vol, set_assemble_bndry, set_assemble_face;
    for (size_t block=0; block<blocknames.size(); ++block) {
      set_assemble_vol.push_back(physics->physics_settings[set][block].template get<bool>("assemble volume terms",true));
      set_assemble_bndry.push_back(physics->physics_settings[set][block].template get<bool>("assemble boundary terms",true));
      set_assemble_face.push_back(physics->physics_settings[set][block].template get<bool>("assemble face terms",false));
    }
    assemble_volume_terms.push_back(set_assemble_vol);
    assemble_boundary_terms.push_back(set_assemble_bndry);
    assemble_face_terms.push_back(set_assemble_face);
  }
  // overwrite assemble_face_terms if HFACE vars are used
  for (size_t set=0; set<assemble_face_terms.size(); ++set) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      vector<string> ctypes = physics->unique_types[block];
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
    
    for (size_t set=0; set<physics->set_names.size(); ++set) {
      
      if (assemble_volume_terms[set][block]) {
        build_volume = true;
      }
      else if (physics->physics_settings[set][block].template get<bool>("build volume terms",true) ) {
        build_volume = true;
      }
      
      if (assemble_boundary_terms[set][block]) {
        build_bndry = true;
      }
      else if (physics->physics_settings[set][block].template get<bool>("build boundary terms",true)) {
        build_bndry = true;
      }
      
      if (assemble_face_terms[set][block]) {
        build_face = true;
      }
      else if (physics->physics_settings[set][block].template get<bool>("build face terms",false)) {
        build_face = true;
      }
    }
    build_volume_terms.push_back(build_volume);
    build_boundary_terms.push_back(build_bndry);
    build_face_terms.push_back(build_face);
  }
  
  // needed information from the physics interface
  varlist = physics->var_list;
  
  // Create groups/boundary groups
  this->createGroups();
  
  params->setupDiscretizedParameters(groups, boundary_groups);
  
  this->createFixedDOFs();
  
  this->createFunctions();

  num_derivs_required = disc->num_derivs_required;
  physics->num_derivs_required = disc->num_derivs_required;

  if (debug_level > 0) {
    if (comm->getRank() == 0) {
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
    if (comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createFixedDOFs ... " << endl;
    }
  }
  
  // create fixedDOF View of bools
  vector<vector<vector<vector<LO> > > > dbc_dofs = disc->dbc_dofs; // [set][block][var][dof]
  for (size_t set=0; set<dbc_dofs.size(); ++set) {
    vector<vector<Kokkos::View<LO*,LA_device> > > set_fixedDOF;
    
    int numLocalDof = disc->dof_owned_and_shared[set].size();//
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
    if (comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::createFixedDOFs" << endl;
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////
// Create the groups
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createGroups() {
  
  Teuchos::TimeMonitor localtimer(*group_timer);
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createGroups ..." << endl;
    }
  }
  
  double storageProportion = settings->sublist("Solver").get<double>("storage proportion",1.0);
  double mesh_scale = settings->sublist("Mesh").get<double>("scale factor",1.0);

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
    int numNodesPerElem = cellTopo->getNodeCount();
    int dimension = physics->dimension;
    size_t numTotalElem = stk_meshElems.size();
    size_t processedElem = 0;
    
    if (numTotalElem>0) {
      
      //vector<size_t> localIds;
      //Kokkos::DynRankView<ScalarT,HostDevice> blocknodes;
      //panzer_stk::workset_utils::getIdsAndVertices(*(mesh->stk_mesh), blocknames[block], localIds, blocknodes); // fill on host
      
      vector<size_t> myElem = disc->my_elements[block];
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
                //Kokkos::View<LO*,AssemblyDevice> sideIndex("local side indices",currElem);
                DRV currnodes("currnodes", currElem, numNodesPerElem, dimension);
                
                auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
                Kokkos::View<LO*,HostDevice> host_eIndex2("element indices",currElem);
                //auto host_sideIndex = Kokkos::create_mirror_view(sideIndex); // mirror on host
                auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
                LO sideIndex;
                
                for (size_t e=0; e<currElem; e++) {
                  host_eIndex(e) = mesh->getSTKElementLocalId(side_output[group[e+prog]]);
                  sideIndex = local_side_Ids[group[e+prog]];
                  for (size_type n=0; n<host_currnodes.extent(1); n++) {
                    for (size_type m=0; m<host_currnodes.extent(2); m++) {
                      host_currnodes(e,n,m) = mesh_scale*sidenodes(group[e+prog],n,m);
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
        DRV currnodes("currnodes", currElem, numNodesPerElem, dimension);
        
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
        
        mesh->getSTKElementVertices(local_grp, blocknames[block], currnodes);
        parallel_for("assembly scale nodes",
                     RangePolicy<AssemblyExec>(0,currnodes.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_t pt=0; pt<currnodes.extent(1); ++pt) {
            for (size_t dim=0; dim<currnodes.extent(2); ++dim) {
              currnodes(elem,pt,dim) *= mesh_scale;
            }
          }
        });
          
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
    if (comm->getRank() == 0) {
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
      this->buildDatabase(block);
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
    cout << " - Processor " << comm->getRank() << " has " << numelements << " elements" << endl;
    cout << " - Processor " << comm->getRank() << " min element size: " << minsize << endl;
    cout << " - Processor " << comm->getRank() << " max element size: " << maxsize << endl;
    
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
    cout << " - Processor " << comm->getRank() << " has " << numbndryelements << " boundary elements" << endl;
    cout << " - Processor " << comm->getRank() << " min boundary element size: " << minbsize << endl;
    cout << " - Processor " << comm->getRank() << " max boundary element size: " << maxbsize << endl;
    
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
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::allocategroupstorage" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createWorkset() {
  
  Teuchos::TimeMonitor localtimer(*wkset_timer);
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::createWorkset ..." << endl;
    }
  }
  
  for (size_t block=0; block<groups.size(); ++block) {
    if (groups[block].size() > 0) {
      vector<int> info;
      info.push_back(groupData[block]->dimension);
      info.push_back((int)groupData[block]->num_disc_params);
      info.push_back(groupData[block]->num_elem);
      info.push_back(groupData[block]->num_ip);
      info.push_back(groupData[block]->num_side_ip);
      info.push_back(physics->set_names.size());
      vector<size_t> numVars;
      for (size_t set=0; set<groupData[block]->set_num_dof.size(); ++set) {
        numVars.push_back(groupData[block]->set_num_dof[set].extent(0));
      }
      vector<Kokkos::View<string**,HostDevice> > bcs(physics->set_names.size());
      for (size_t set=0; set<physics->set_names.size(); ++set) {
        Kokkos::View<string**,HostDevice> vbcs = disc->getVarBCs(set,block);
        bcs[set] = vbcs;
      }

      // ScalarT workset, always active unless no elements on proc
      wkset.push_back(Teuchos::rcp( new Workset<ScalarT>(info, numVars, isTransient,
                                                         disc->basis_types[block],
                                                         disc->basis_pointers[block],
                                                         params->discretized_param_basis,
                                                         groupData[block]->cell_topo)));
      wkset[block]->block = block;
      wkset[block]->blockname = blocknames[block];
      wkset[block]->set_var_bcs = bcs;
      wkset[block]->var_bcs = bcs[0];

#ifndef MrHyDE_NO_AD
      bool fully_explicit = settings->sublist("Solver").get<bool>("fully explicit",false);
      string analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
      bool requires_AD = true;
      if (fully_explicit && analysis_type == "forward") {
        requires_AD = false;
      }
      
      bool found = false;
      
      int ndr = num_derivs_required[block];
  
      if (requires_AD && !found && ndr>0 && ndr <= 2 ) {
        // AD2 workset
        wkset_AD2.push_back(Teuchos::rcp( new Workset<AD2>(info, numVars, isTransient,
                                                           disc->basis_types[block],
                                                           disc->basis_pointers[block],
                                                           params->discretized_param_basis,
                                                           groupData[block]->cell_topo)));
        wkset_AD2[block]->block = block;
        wkset_AD2[block]->blockname = blocknames[block];
        wkset_AD2[block]->set_var_bcs = bcs;
        wkset_AD2[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        wkset_AD2.push_back(Teuchos::rcp( new Workset<AD2>(block, physics->set_names.size())));  
      }

      if (requires_AD && !found && ndr>2 && ndr <= 4 ) {
        // AD4 workset
        wkset_AD4.push_back(Teuchos::rcp( new Workset<AD4>(info, numVars, isTransient,
                                                             disc->basis_types[block],
                                                             disc->basis_pointers[block],
                                                             params->discretized_param_basis,
                                                             groupData[block]->cell_topo)));
        wkset_AD4[block]->block = block;
        wkset_AD4[block]->blockname = blocknames[block];
        wkset_AD4[block]->set_var_bcs = bcs;
        wkset_AD4[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        wkset_AD4.push_back(Teuchos::rcp( new Workset<AD4>(block, physics->set_names.size())));  
      }

      if (requires_AD && !found && ndr>4 && ndr <= 8 ) {
        // AD8 workset
        wkset_AD8.push_back(Teuchos::rcp( new Workset<AD8>(info, numVars, isTransient,
                                                             disc->basis_types[block],
                                                             disc->basis_pointers[block],
                                                             params->discretized_param_basis,
                                                             groupData[block]->cell_topo)));
        wkset_AD8[block]->block = block;
        wkset_AD8[block]->blockname = blocknames[block];
        wkset_AD8[block]->set_var_bcs = bcs;
        wkset_AD8[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        wkset_AD8.push_back(Teuchos::rcp( new Workset<AD8>(block, physics->set_names.size())));  
      }

      if (requires_AD && !found && ndr>8 && ndr <= 16 ) {
        // AD16 workset
        wkset_AD16.push_back(Teuchos::rcp( new Workset<AD16>(info, numVars, isTransient,
                                                               disc->basis_types[block],
                                                               disc->basis_pointers[block],
                                                               params->discretized_param_basis,
                                                               groupData[block]->cell_topo)));
        wkset_AD16[block]->block = block;
        wkset_AD16[block]->blockname = blocknames[block];
        wkset_AD16[block]->set_var_bcs = bcs;
        wkset_AD16[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        wkset_AD16.push_back(Teuchos::rcp( new Workset<AD16>(block, physics->set_names.size())));  
      }

      if (requires_AD && !found && ndr>16 && ndr <= 18 ) {
        // AD18 workset
        wkset_AD18.push_back(Teuchos::rcp( new Workset<AD18>(info, numVars, isTransient,
                                                               disc->basis_types[block],
                                                               disc->basis_pointers[block],
                                                               params->discretized_param_basis,
                                                               groupData[block]->cell_topo)));
        wkset_AD18[block]->block = block;
        wkset_AD18[block]->blockname = blocknames[block];
        wkset_AD18[block]->set_var_bcs = bcs;
        wkset_AD18[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        wkset_AD18.push_back(Teuchos::rcp( new Workset<AD18>(block, physics->set_names.size())));  
      }

      if (requires_AD && !found && ndr>18 && ndr <= 24 ) {
        // AD24 workset
        wkset_AD24.push_back(Teuchos::rcp( new Workset<AD24>(info, numVars, isTransient,
                                                               disc->basis_types[block], 
                                                               disc->basis_pointers[block],
                                                               params->discretized_param_basis,
                                                               groupData[block]->cell_topo)));
        wkset_AD24[block]->block = block;
        wkset_AD24[block]->blockname = blocknames[block];
        wkset_AD24[block]->set_var_bcs = bcs;
        wkset_AD24[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        wkset_AD24.push_back(Teuchos::rcp( new Workset<AD24>(block, physics->set_names.size())));  
      }

      if (requires_AD && !found && ndr>24 && ndr <= 32 ) {
        // AD32 workset
        wkset_AD32.push_back(Teuchos::rcp( new Workset<AD32>(info, numVars, isTransient,
                                                               disc->basis_types[block],
                                                               disc->basis_pointers[block],
                                                               params->discretized_param_basis,
                                                               groupData[block]->cell_topo)));
        wkset_AD32[block]->block = block;
        wkset_AD32[block]->blockname = blocknames[block];
        wkset_AD32[block]->set_var_bcs = bcs;
        wkset_AD32[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        wkset_AD32.push_back(Teuchos::rcp( new Workset<AD32>(block, physics->set_names.size())));  
      }

      if ((requires_AD && !found) || groupData[block]->multiscale || !allow_autotune) {
        // AD workset
        wkset_AD.push_back(Teuchos::rcp( new Workset<AD>(info, numVars, isTransient,
                                                         disc->basis_types[block],
                                                         disc->basis_pointers[block],
                                                         params->discretized_param_basis,
                                                         groupData[block]->cell_topo)));
        wkset_AD[block]->block = block;
        wkset_AD[block]->blockname = blocknames[block];
        wkset_AD[block]->set_var_bcs = bcs;
        wkset_AD[block]->var_bcs = bcs[0];
      }
      else {
        wkset_AD.push_back(Teuchos::rcp( new Workset<AD>(block, physics->set_names.size())));  
      }
     
#endif
    }
    else {
      wkset.push_back(Teuchos::rcp( new Workset<ScalarT>(block, physics->set_names.size())));
#ifndef MrHyDE_NO_AD
      wkset_AD.push_back(Teuchos::rcp( new Workset<AD>(block, physics->set_names.size())));
      wkset_AD2.push_back(Teuchos::rcp( new Workset<AD2>(block, physics->set_names.size())));
      wkset_AD4.push_back(Teuchos::rcp( new Workset<AD4>(block, physics->set_names.size())));
      wkset_AD8.push_back(Teuchos::rcp( new Workset<AD8>(block, physics->set_names.size())));
      wkset_AD16.push_back(Teuchos::rcp( new Workset<AD16>(block, physics->set_names.size())));
      wkset_AD18.push_back(Teuchos::rcp( new Workset<AD18>(block, physics->set_names.size())));
      wkset_AD24.push_back(Teuchos::rcp( new Workset<AD24>(block, physics->set_names.size())));
      wkset_AD32.push_back(Teuchos::rcp( new Workset<AD32>(block, physics->set_names.size())));
#endif
    }
    
  }
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::createWorkset" << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::addFunction(const int & block, const string & name, const string & expression, const string & location) {
  function_managers[block]->addFunction(name, expression, location);
#ifndef MrHyDE_NO_AD
  function_managers_AD[block]->addFunction(name, expression, location);
  function_managers_AD2[block]->addFunction(name, expression, location);
  function_managers_AD4[block]->addFunction(name, expression, location);
  function_managers_AD8[block]->addFunction(name, expression, location);
  function_managers_AD16[block]->addFunction(name, expression, location);
  function_managers_AD18[block]->addFunction(name, expression, location);
  function_managers_AD24[block]->addFunction(name, expression, location);
  function_managers_AD32[block]->addFunction(name, expression, location);
#endif
}

template<class Node>
View_Sc2 AssemblyManager<Node>::evaluateFunction(const int & block, const string & name, const string & location) {

  typedef typename Node::execution_space LA_exec;

  auto data = function_managers[block]->evaluate(name, location);
  size_type num_elem = function_managers[block]->num_elem_;
  size_type num_pts = 0;
  if (location == "ip") {
    num_pts = function_managers[block]->num_ip_;
  }
  else if (location == "side ip") {
    num_pts = function_managers[block]->num_ip_side_;
  }
  else if (location == "point") {
    num_pts = 1;
  }

  View_Sc2 outdata("data from function evaluation", num_elem, num_pts);

  parallel_for("assembly eval func",
                 RangePolicy<LA_exec>(0,num_elem),
                 KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<num_pts; ++pt) {
      outdata(elem,pt) = data(elem,pt);
    }
  });

  return outdata;
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
  
  Teuchos::TimeMonitor localtimer(*set_init_timer);
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::setInitial ..." << endl;
    }
  }
  
  for (size_t block=0; block<groups.size(); block++) {
    if (wkset[block]->isInitialized) {
      this->setInitial(set,rhs,mass,useadjoint,lumpmass,scale,block,block);
    }
  }
  
  mass->fillComplete();
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
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
  auto numDOF = groupData[block]->num_dof;
  
  for (size_t grp=0; grp<groups[groupblock].size(); ++grp) {
    
    auto LIDs = groups[groupblock][grp]->LIDs[set];
    
    auto localrhs = this->getInitial(groupblock, grp, true, useadjoint);
    auto localmass = this->getMass(groupblock, grp);
    
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
      LO cols[MAXDERIVS];
      ScalarT vals[MAXDERIVS];
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
    size_t numrows = mass->getLocalNumRows();
    
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
  
  Teuchos::TimeMonitor localtimer(*set_init_timer);
  
  using namespace std;

  if (debug_level > 0) {
    if (comm->getRank() == 0) {
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
                       KOKKOS_LAMBDA (const int elem ) {
          
            LO eindex = index(elem);
            for (size_type n=0; n<numDOF.extent(0); ++n) {
              for (int j=0; j<numDOF(n); j++) {
                LO localrow = offsets(n,j);
                LO globalrow = LIDs(elem,localrow);

                ScalarT val = 0.0;
                if (use_jacobi) {
                  for (int k=0; k<nnz(eindex,localrow); k++ ) {
                    LO localcol = offsets(n,local_columns(eindex,localrow,k));
                    LO globalcol = LIDs(elem,localcol);
                    if (globalrow == globalcol) {
                      val = values(eindex,localrow,k);
                    }
                  }
                }
                else {
                  for (int k=0; k<nnz(eindex,localrow); k++ ) {
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
                         KOKKOS_LAMBDA (const int elem ) {
              
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
                     KOKKOS_LAMBDA (const int elem ) {
          
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
                       KOKKOS_LAMBDA (const int elem ) {
            
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
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
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
          disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->nodes,
                                           groups[block][grp]->orientation, tmpbasis);
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
                         KOKKOS_LAMBDA (const size_type elem ) {
              LO eindex = index(elem);
           
              // New code that uses sparse data structures
              for (size_type var=0; var<numDOF.extent(0); var++) {
                for (int i=0; i<numDOF(var); i++ ) {
                  LO localrow = offsets(var,i);
                  LO globalrow = cLIDs(elem,localrow);
                  for (int k=0; k<nnz(eindex,localrow); k++ ) {
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
                         KOKKOS_LAMBDA (const size_type elem ) {
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
  
  Teuchos::TimeMonitor localtimer(*set_init_timer);
  
  typedef typename Node::execution_space LA_exec;
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::getWeightVector ..." << endl;
    }
  }
  
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
    if (comm->getRank() == 0) {
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
      Kokkos::View<ScalarT**,AssemblyDevice> localinit = this->getInitial(block, grp, false, useadjoint);
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
  
  Teuchos::TimeMonitor localtimer(*set_dbc_timer);
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
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
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::setDirichlet ..." << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::setInitialFace(const size_t & set, vector_RCP & rhs, matrix_RCP & mass,
                                           const bool & lumpmass) {
  
  Teuchos::TimeMonitor localtimer(*set_init_timer);
  
  using namespace std;
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
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
      auto localrhs = this->getInitialFace(block, grp, true);
      // Create the mass matrix
      auto localmass = this->getMassFace(block, grp);
      auto host_rhs = Kokkos::create_mirror_view(localrhs);
      auto host_mass = localmass;//Kokkos::create_mirror_view(localmass);
      Kokkos::deep_copy(host_rhs,localrhs);
      //Kokkos::deep_copy(host_mass,localmass);
      
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
    wkset[block]->isOnSide = false;
  }
  
  // make sure we don't have any rows of all zeroes
  // TODO I don't think this can ever happen?
  // at least globally
  
  typedef typename Node::execution_space LA_exec;
  size_t numrows = mass->getLocalNumRows();
  
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
    if (comm->getRank() == 0) {
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
    if (comm->getRank() == 0) {
      cout << "******** Starting AssemblyManager::assembleJacRes ..." << endl;
    }
  }
#ifndef MrHyDE_NO_AD
  {
    Teuchos::TimeMonitor localtimer(*gather_timer);
    
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
      if (groupData[block]->multiscale) {
        allow_autotune = false;
      }
      if (!allow_autotune) {
        this->assembleJacRes<AD>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
      }
      else {
        int ndr = num_derivs_required[block];
        if (ndr == MAXDERIVS) {
          this->assembleJacRes<AD>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
        }
        else if (ndr>0 && ndr <= 2) {
          this->assembleJacRes<AD2>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
        }
        else if (ndr>2 && ndr <= 4) {
          this->assembleJacRes<AD4>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
        }
        else if (ndr>4 && ndr <= 8) {
          this->assembleJacRes<AD8>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
        }
        else if (ndr>8 && ndr <= 16) {
          this->assembleJacRes<AD16>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
        }
        else if (ndr>16 && ndr <= 18) {
          this->assembleJacRes<AD18>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
        }
        else if (ndr>18 && ndr <= 24) {
          this->assembleJacRes<AD24>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
        }
        else if (ndr>24 && ndr <= 32) {
          this->assembleJacRes<AD32>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
        }
        else {
          this->assembleJacRes<AD>(set, compute_jacobian,
                             compute_sens, compute_disc_sens, res, J, isTransient,
                             current_time, useadjoint, store_adjPrev, num_active_params,
                             is_final_time, block, deltat);
        }
      }
    }
  }
  #endif

  if (debug_level > 1) {
    if (comm->getRank() == 0) {
      cout << "******** Finished AssemblyManager::assembleJacRes" << endl;
    }
  }
}

// ========================================================================================
// Wrapper to the main assembly routine to assemble over all blocks (most common use case)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::assembleRes(const size_t & set, vector_RCP & u, vector_RCP & phi,
                                               const bool & compute_jacobian, const bool & compute_sens,
                                               const bool & compute_disc_sens,
                                               vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                               const ScalarT & current_time,
                                               const bool & useadjoint, const bool & store_adjPrev,
                                               const int & num_active_params,
                                               vector_RCP & Psol, const bool & is_final_time,
                                               const ScalarT & deltat) {
  
  if (debug_level > 1) {
    if (comm->getRank() == 0) {
      cout << "******** Starting AssemblyManager::assembleRes ..." << endl;
    }
  }
  
  {
    Teuchos::TimeMonitor localtimer(*gather_timer);
    
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
      this->assembleRes(set, compute_jacobian,
                           compute_sens, compute_disc_sens, res, J, isTransient,
                           current_time, useadjoint, store_adjPrev, num_active_params,
                           is_final_time, block, deltat);
    }
  }
  
  if (debug_level > 1) {
    if (comm->getRank() == 0) {
      cout << "******** Finished AssemblyManager::assembleRes" << endl;
    }
  }
}

// ========================================================================================
// Main assembly routine ... only assembles on a given block (b)
// This routine is the old version that does both Jacobian and residual
// Will eventually be deprecated
// ========================================================================================

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::assembleJacRes(const size_t & set, const bool & compute_jacobian, const bool & compute_sens,
                                           const bool & compute_disc_sens,
                                           vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                           const ScalarT & current_time,
                                           const bool & useadjoint, const bool & store_adjPrev,
                                           const int & num_active_params,
                                           const bool & is_final_time,
                                           const int & block, const ScalarT & deltat) {
  
  Teuchos::TimeMonitor localassemblytimer(*assembly_timer);
  using namespace std;

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
  
  //  This actually updates all of the worksets, which is fine
  this->updateWorksetTime(block, isTransient, current_time, deltat);

  this->updateWorksetAdjoint(block, useadjoint);
  int numElem = groupData[block]->num_elem;
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
    
    this->updateWorksetEID(block, grp);
    
    if (isTransient && useadjoint && !groups[block][0]->group_data->multiscale) {
      if (is_final_time) {
        groups[block][grp]->resetAdjPrev(set,0.0);
      }
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // Compute the local residual and Jacobian on this group
    /////////////////////////////////////////////////////////////////////////////
    
    bool fixJacDiag = false;
    
    {
      Teuchos::TimeMonitor localtimer(*physics_timer);
      
      //////////////////////////////////////////////////////////////
      // Compute res and J=dF/du
      //////////////////////////////////////////////////////////////
      
      // Volumetric contribution
      if (assemble_volume_terms[set][block]) {
        if (groupData[block]->multiscale) {
          
#ifndef MrHyDE_NO_AD
          // Right now, this can only be called with AD, thus hard-coded
          multiscale_manager->evaluateMacroMicroMacroMap(wkset_AD[block], groups[block][grp], set, isTransient, useadjoint,
                                                         compute_jacobian, compute_sens, num_active_params,
                                                         compute_disc_sens, false,
                                                         store_adjPrev);
          
          fixJacDiag = true;
#endif
        }
        else {
          this->updateWorkset<EvalT>(block, grp, seedwhat, 0);
          physics->volumeResidual<EvalT>(set, block);
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
          this->updateWorksetOnSide(block, true);
          for (size_t s=0; s<groupData[block]->num_sides; s++) {
            this->updateWorksetFace<EvalT>(block, grp, s);
            physics->faceResidual<EvalT>(set,block);
          }
          this->updateWorksetOnSide(block, false);
        }
      }
      
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Scatter into global matrix/vector
    ///////////////////////////////////////////////////////////////////////////
    
    if (reduce_memory) { // skip local_res and local_J
      EvalT dummyval = 0.0;
      this->scatter(set, J_kcrs, res_view,
                    groups[block][grp]->LIDs[set], groups[block][grp]->paramLIDs, block,
                    compute_jacobian, compute_sens, compute_disc_sens, useadjoint, dummyval);
    }
    else { // fill local_res and local_J and then scatter
      
      Teuchos::TimeMonitor localtimer(*scatter_timer);
      
      Kokkos::deep_copy(local_res,0.0);
      Kokkos::deep_copy(local_J,0.0);
      
      // Use AD residual to update local Jacobian
      if (compute_jacobian) {
        if (compute_disc_sens) {
          this->updateParamJac(block, grp, local_J);
        }
        else {
          this->updateJac(block, grp, useadjoint, local_J);
        }
      }
      
      if (compute_jacobian && fixJacDiag) {
        this->fixDiagJac(block, grp, local_J, local_res);
      }
      
      // Update the local residual
      
      if (useadjoint) {
        this->updateAdjointRes(block, grp, compute_jacobian, isTransient,
                                             false, store_adjPrev,
                                             local_J, local_res);
      }
      else {
        this->updateRes(block, grp, compute_sens, local_res);
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
    
    this->updateWorksetOnSide(block, true);
    
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
        
        this->updateWorksetResidual(block);

        this->updateWorksetBoundary<EvalT>(block, grp, seedwhat);
        
        if (!groupData[block]->multiscale) {
          Teuchos::TimeMonitor localtimer(*physics_timer);
          physics->boundaryResidual<EvalT>(set,block);
        }
        
        {
          physics->fluxConditions<EvalT>(set,block);
        }
        ///////////////////////////////////////////////////////////////////////////
        // Scatter into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        if (reduce_memory) { // skip local_res and local_J
          EvalT dummyval = 0.0;
          this->scatter(set, J_kcrs, res_view,
                        boundary_groups[block][grp]->LIDs[set], boundary_groups[block][grp]->paramLIDs, block,
                        compute_jacobian, compute_sens, compute_disc_sens, useadjoint, dummyval);
        }
        else { // fill local_res and local_J and then scatter
          
          Teuchos::TimeMonitor localtimer(*scatter_timer);
          
          Kokkos::deep_copy(local_res,0.0);
          Kokkos::deep_copy(local_J,0.0);
          
          // Use AD residual to update local Jacobian
          if (compute_jacobian) {
            if (compute_disc_sens) {
              this->updateParamJacBoundary(block, grp, local_J);
            }
            else {
              this->updateJacBoundary(block, grp, useadjoint, local_J);
            }
          }
          
          // Update the local residual (forward mode)
          if (!useadjoint) {
            this->updateResBoundary(block, grp, compute_sens, local_res);
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
    this->updateWorksetOnSide(block, false);
  }
  
  // Apply constraints, e.g., strongly imposed Dirichlet
  this->dofConstraints(set, J, res, current_time, compute_jacobian, compute_disc_sens);
  
  
  if (fix_zero_rows) {
    size_t numrows = J->getLocalNumRows();
    
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


/// @brief ////////////////////////////
/// @tparam Node 
/// @param block 
/// @param isTransient 
/// @param current_time 
/// @param deltat /
template<class Node>
void AssemblyManager<Node>::updateWorksetTime(const size_t & block, const bool & isTransient,
                                              const ScalarT & current_time, const ScalarT & deltat) {

  if (wkset[block]->isInitialized) {
    this->updateWorksetTime(wkset[block], isTransient, current_time, deltat);
  }
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateWorksetTime(wkset_AD[block], isTransient, current_time, deltat);
  }
  if (wkset_AD2[block]->isInitialized) {
    this->updateWorksetTime(wkset_AD2[block], isTransient, current_time, deltat);
  }
  if (wkset_AD4[block]->isInitialized) {
    this->updateWorksetTime(wkset_AD4[block], isTransient, current_time, deltat);
  }
  if (wkset_AD8[block]->isInitialized) {
    this->updateWorksetTime(wkset_AD8[block], isTransient, current_time, deltat);
  }
  if (wkset_AD16[block]->isInitialized) {
    this->updateWorksetTime(wkset_AD16[block], isTransient, current_time, deltat);
  }
  if (wkset_AD18[block]->isInitialized) {
    this->updateWorksetTime(wkset_AD18[block], isTransient, current_time, deltat);
  }
  if (wkset_AD24[block]->isInitialized) {
    this->updateWorksetTime(wkset_AD24[block], isTransient, current_time, deltat);
  }
  if (wkset_AD32[block]->isInitialized) {
    this->updateWorksetTime(wkset_AD32[block], isTransient, current_time, deltat);
  }
#endif
}

/// @brief /////////////////////////////
/// @tparam Node 
/// @param wset 
/// @param isTransient 
/// @param current_time 
/// @param deltat /
template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetTime(Teuchos::RCP<Workset<EvalT> > & wset, const bool & isTransient,
                                              const ScalarT & current_time, const ScalarT & deltat) {

  if (isTransient) {
    // TMW: tmp fix
    auto butcher_c = Kokkos::create_mirror_view(wset->butcher_c);
    Kokkos::deep_copy(butcher_c, wset->butcher_c);
    ScalarT timeval = current_time + butcher_c(wset->current_stage)*deltat;
    
    wset->setTime(timeval);
    wset->setDeltat(deltat);
    wset->alpha = 1.0/deltat;
  }
  
  wset->isTransient = isTransient;
}


////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetAdjoint(const size_t & block, const bool & isAdjoint) {
                                                

  if (wkset[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset[block], isAdjoint);
  }
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset_AD[block], isAdjoint);
  }
  if (wkset_AD2[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset_AD2[block], isAdjoint);
  }
  if (wkset_AD4[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset_AD4[block], isAdjoint);
  }
  if (wkset_AD8[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset_AD8[block], isAdjoint);
  }
  if (wkset_AD16[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset_AD16[block], isAdjoint);
  }
  if (wkset_AD18[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset_AD18[block], isAdjoint);
  }
  if (wkset_AD24[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset_AD24[block], isAdjoint);
  }
  if (wkset_AD32[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset_AD32[block], isAdjoint);
  }
#endif
}

////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetAdjoint(Teuchos::RCP<Workset<EvalT> > & wset, const bool & isAdjoint) {
  wset->isAdjoint = isAdjoint;
}

////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetEID(const size_t & block, const size_t & eid) {
                                                

  if (wkset[block]->isInitialized) {
    this->updateWorksetEID(wkset[block], eid);
  }
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateWorksetEID(wkset_AD[block], eid);
  }
  if (wkset_AD2[block]->isInitialized) {
    this->updateWorksetEID(wkset_AD2[block], eid);
  }
  if (wkset_AD4[block]->isInitialized) {
    this->updateWorksetEID(wkset_AD4[block], eid);
  }
  if (wkset_AD8[block]->isInitialized) {
    this->updateWorksetEID(wkset_AD8[block], eid);
  }
  if (wkset_AD16[block]->isInitialized) {
    this->updateWorksetEID(wkset_AD16[block], eid);
  }
  if (wkset_AD18[block]->isInitialized) {
    this->updateWorksetEID(wkset_AD18[block], eid);
  }
  if (wkset_AD24[block]->isInitialized) {
    this->updateWorksetEID(wkset_AD24[block], eid);
  }
  if (wkset_AD32[block]->isInitialized) {
    this->updateWorksetEID(wkset_AD32[block], eid);
  }
#endif
}

////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetEID(Teuchos::RCP<Workset<EvalT> > & wset, const size_t & eid) {
  wset->localEID = eid;
}

////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetOnSide(const size_t & block, const bool & on_side) {
                                                

  if (wkset[block]->isInitialized) {
    this->updateWorksetOnSide(wkset[block], on_side);
  }
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateWorksetOnSide(wkset_AD[block], on_side);
  }
  if (wkset_AD2[block]->isInitialized) {
    this->updateWorksetOnSide(wkset_AD2[block], on_side);
  }
  if (wkset_AD4[block]->isInitialized) {
    this->updateWorksetOnSide(wkset_AD4[block], on_side);
  }
  if (wkset_AD8[block]->isInitialized) {
    this->updateWorksetOnSide(wkset_AD8[block], on_side);
  }
  if (wkset_AD16[block]->isInitialized) {
    this->updateWorksetOnSide(wkset_AD16[block], on_side);
  }
  if (wkset_AD18[block]->isInitialized) {
    this->updateWorksetOnSide(wkset_AD18[block], on_side);
  }
  if (wkset_AD24[block]->isInitialized) {
    this->updateWorksetOnSide(wkset_AD24[block], on_side);
  }
  if (wkset_AD32[block]->isInitialized) {
    this->updateWorksetOnSide(wkset_AD32[block], on_side);
  }
#endif
}

////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetOnSide(Teuchos::RCP<Workset<EvalT> > & wset, const bool & on_side) {
  wset->isOnSide = on_side;
}

////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetResidual(const size_t & block) {
                                                

  if (wkset[block]->isInitialized) {
    this->updateWorksetResidual(wkset[block]);
  }
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateWorksetResidual(wkset_AD[block]);
  }
  if (wkset_AD2[block]->isInitialized) {
    this->updateWorksetResidual(wkset_AD2[block]);
  }
  if (wkset_AD4[block]->isInitialized) {
    this->updateWorksetResidual(wkset_AD4[block]);
  }
  if (wkset_AD8[block]->isInitialized) {
    this->updateWorksetResidual(wkset_AD8[block]);
  }
  if (wkset_AD16[block]->isInitialized) {
    this->updateWorksetResidual(wkset_AD16[block]);
  }
  if (wkset_AD18[block]->isInitialized) {
    this->updateWorksetResidual(wkset_AD18[block]);
  }
  if (wkset_AD24[block]->isInitialized) {
    this->updateWorksetResidual(wkset_AD24[block]);
  }
  if (wkset_AD32[block]->isInitialized) {
    this->updateWorksetResidual(wkset_AD32[block]);
  }
#endif
}

////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetResidual(Teuchos::RCP<Workset<EvalT> > & wset) {
  wset->resetResidual();
}

// ========================================================================================
// Main assembly routine ... just the residual on a given block (b)
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::assembleRes(const size_t & set, const bool & compute_jacobian, const bool & compute_sens,
                                           const bool & compute_disc_sens,
                                           vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                                           const ScalarT & current_time,
                                           const bool & useadjoint, const bool & store_adjPrev,
                                           const int & num_active_params,
                                           const bool & is_final_time,
                                           const int & block, const ScalarT & deltat) {
  
  Teuchos::TimeMonitor localassemblytimer(*assembly_res_timer);
    
  auto res_view = res->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  
  // Set the seeding flag for AD objects
  int seedwhat = 0;
    
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
  wkset[block]->isAdjoint = false;
  
  /////////////////////////////////////////////////////////////////////////////
  // Volume contribution
  /////////////////////////////////////////////////////////////////////////////
  
  // Note: Cannot parallelize over groups since data structures are re-used
  
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    
    wkset[block]->localEID = grp;
    
    /////////////////////////////////////////////////////////////////////////////
    // Compute the local residual and Jacobian on this group
    /////////////////////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*physics_timer);
      
      //////////////////////////////////////////////////////////////
      // Compute res and J=dF/du
      //////////////////////////////////////////////////////////////
    
      // Volumetric contribution
      if (assemble_volume_terms[set][block]) {
        this->updateWorkset<ScalarT>(block, grp, seedwhat, 0);
        physics->volumeResidual<ScalarT>(set, block);
      }
      
      ///////////////////////////////////////////////////////////////////////////
      // Edge/face contribution
      ///////////////////////////////////////////////////////////////////////////
      
      if (assemble_face_terms[set][block]) {
        wkset[block]->isOnSide = true;
        for (size_t s=0; s<groupData[block]->num_sides; s++) {
          this->updateWorksetFace<ScalarT>(block, grp, s);
          physics->faceResidual<ScalarT>(set,block);
        }
        wkset[block]->isOnSide = false;
      }
      
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Scatter into global matrix/vector
    ///////////////////////////////////////////////////////////////////////////
    
    this->scatterRes(set, res_view, groups[block][grp]->LIDs[set], block);
    
  } // group loop
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Boundary terms
  //////////////////////////////////////////////////////////////////////////////////////
  
  
  if (assemble_boundary_terms[set][block]) {
    
    wkset[block]->isOnSide = true;
    
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      
      if (boundary_groups[block][grp]->numElem > 0) {
        
        /////////////////////////////////////////////////////////////////////////////
        // Compute the local residual and Jacobian on this boundary group
        /////////////////////////////////////////////////////////////////////////////
        wkset[block]->resetResidual();
        this->updateWorksetBoundary<ScalarT>(block, grp, seedwhat);
        physics->boundaryResidual<ScalarT>(set,block);
        physics->fluxConditions<ScalarT>(set,block);
        
        ///////////////////////////////////////////////////////////////////////////
        // Scatter into global matrix/vector
        ///////////////////////////////////////////////////////////////////////////
        
        this->scatterRes(set, res_view, boundary_groups[block][grp]->LIDs[set], block);
        
      }
    } // element loop
    wkset[block]->isOnSide = false;
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
    if (comm->getRank() == 0) {
      cout << "******** Starting AssemblyManager::dofConstraints" << endl;
    }
  }
  
  Teuchos::TimeMonitor localtimer(*dbc_timer);
  
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
    if (comm->getRank() == 0) {
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
    groupData[block]->current_stage = stage;
    if (wkset[block]->isInitialized) {
      this->updateStage(wkset[block], stage, current_time, deltat);
    }
#ifndef MrHyDE_NO_AD
    if (wkset_AD[block]->isInitialized) {
      this->updateStage(wkset_AD[block], stage, current_time, deltat);
    }
    if (wkset_AD2[block]->isInitialized) {
      this->updateStage(wkset_AD2[block], stage, current_time, deltat);
    }
    if (wkset_AD4[block]->isInitialized) {
      this->updateStage(wkset_AD4[block], stage, current_time, deltat);
    }
    if (wkset_AD8[block]->isInitialized) {
      this->updateStage(wkset_AD8[block], stage, current_time, deltat);
    }
    if (wkset_AD16[block]->isInitialized) {
      this->updateStage(wkset_AD16[block], stage, current_time, deltat);
    }
    if (wkset_AD18[block]->isInitialized) {
      this->updateStage(wkset_AD18[block], stage, current_time, deltat);
    }
    if (wkset_AD24[block]->isInitialized) {
      this->updateStage(wkset_AD24[block], stage, current_time, deltat);
    }
    if (wkset_AD32[block]->isInitialized) {
      this->updateStage(wkset_AD32[block], stage, current_time, deltat);
    }
#endif
    
  }
}


template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateStage(Teuchos::RCP<Workset<EvalT> > & wset, const int & stage, const ScalarT & current_time,
                                        const ScalarT & deltat) {
  wset->setStage(stage);
  auto butcher_c = Kokkos::create_mirror_view(wset->butcher_c);
  Kokkos::deep_copy(butcher_c, wset->butcher_c);
  ScalarT timeval = current_time + butcher_c(stage)*deltat;
  wset->setTime(timeval);
  wset->setDeltat(deltat);
  wset->alpha = 1.0/deltat;

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

template<class Node>
void AssemblyManager<Node>::updateTimeStep(const int & timestep) {
  for (size_t block=0; block<wkset.size(); ++block) {
    wkset[block]->time_step = timestep;
#ifndef MrHyDE_NO_AD
    wkset_AD[block]->time_step = timestep;
    wkset_AD2[block]->time_step = timestep;
    wkset_AD4[block]->time_step = timestep;
    wkset_AD8[block]->time_step = timestep;
    wkset_AD16[block]->time_step = timestep;
    wkset_AD18[block]->time_step = timestep;
    wkset_AD24[block]->time_step = timestep;
    wkset_AD32[block]->time_step = timestep;
#endif
  }
}
    
template<class Node>
void AssemblyManager<Node>::setWorksetButcher(const size_t & set, const size_t & block, 
                                        Kokkos::View<ScalarT**,AssemblyDevice> butcher_A, 
                                        Kokkos::View<ScalarT*,AssemblyDevice> butcher_b, 
                                        Kokkos::View<ScalarT*,AssemblyDevice> butcher_c) {


  if (wkset[block]->isInitialized) {
    this->setWorksetButcher(set, wkset[block],butcher_A, butcher_b, butcher_c);
  }
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->setWorksetButcher(set, wkset_AD[block],butcher_A, butcher_b, butcher_c);
  }
  if (wkset_AD2[block]->isInitialized) {
    this->setWorksetButcher(set, wkset_AD2[block],butcher_A, butcher_b, butcher_c);
  }
  if (wkset_AD4[block]->isInitialized) {
    this->setWorksetButcher(set, wkset_AD4[block],butcher_A, butcher_b, butcher_c);
  }
  if (wkset_AD8[block]->isInitialized) {
    this->setWorksetButcher(set, wkset_AD8[block],butcher_A, butcher_b, butcher_c);
  }
  if (wkset_AD16[block]->isInitialized) {
    this->setWorksetButcher(set, wkset_AD16[block],butcher_A, butcher_b, butcher_c);
  }
  if (wkset_AD18[block]->isInitialized) {
    this->setWorksetButcher(set, wkset_AD18[block],butcher_A, butcher_b, butcher_c);
  }
  if (wkset_AD24[block]->isInitialized) {
    this->setWorksetButcher(set, wkset_AD24[block],butcher_A, butcher_b, butcher_c);
  }
  if (wkset_AD32[block]->isInitialized) {
    this->setWorksetButcher(set, wkset_AD32[block],butcher_A, butcher_b, butcher_c);
  }
#endif
}

template<class Node>                    
template<class EvalT>
void AssemblyManager<Node>::setWorksetButcher(const size_t & set, Teuchos::RCP<Workset<EvalT> > & wset, 
                                        Kokkos::View<ScalarT**,AssemblyDevice> butcher_A, 
                                        Kokkos::View<ScalarT*,AssemblyDevice> butcher_b, 
                                        Kokkos::View<ScalarT*,AssemblyDevice> butcher_c) {

  wset->set_butcher_A[set] = butcher_A;
  wset->set_butcher_b[set] = butcher_b;
  wset->set_butcher_c[set] = butcher_c;

  // TODO dont like this... but should protect against 1 set errors
  wset->butcher_A = butcher_A;
  wset->butcher_b = butcher_b;
  wset->butcher_c = butcher_c;

}

template<class Node>
void AssemblyManager<Node>::setWorksetBDF(const size_t & set, const size_t & block, 
                                        Kokkos::View<ScalarT*,AssemblyDevice> BDF_wts) {


  if (wkset[block]->isInitialized) {
    this->setWorksetBDF(set, wkset[block], BDF_wts);
  }
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->setWorksetBDF(set, wkset_AD[block], BDF_wts);
  }
  if (wkset_AD2[block]->isInitialized) {
    this->setWorksetBDF(set, wkset_AD2[block], BDF_wts);
  }
  if (wkset_AD4[block]->isInitialized) {
    this->setWorksetBDF(set, wkset_AD4[block], BDF_wts);
  }
  if (wkset_AD8[block]->isInitialized) {
    this->setWorksetBDF(set, wkset_AD8[block], BDF_wts);
  }
  if (wkset_AD16[block]->isInitialized) {
    this->setWorksetBDF(set, wkset_AD16[block], BDF_wts);
  }
  if (wkset_AD18[block]->isInitialized) {
    this->setWorksetBDF(set, wkset_AD18[block], BDF_wts);
  }
  if (wkset_AD24[block]->isInitialized) {
    this->setWorksetBDF(set, wkset_AD24[block], BDF_wts);
  }
  if (wkset_AD32[block]->isInitialized) {
    this->setWorksetBDF(set, wkset_AD32[block], BDF_wts);
  }
#endif
}

template<class Node>                    
template<class EvalT>
void AssemblyManager<Node>::setWorksetBDF(const size_t & set, Teuchos::RCP<Workset<EvalT> > & wset,  
                                        Kokkos::View<ScalarT*,AssemblyDevice> BDF_wts) {

  wset->set_BDF_wts[set] = BDF_wts;
  wset->BDF_wts = BDF_wts;

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
          numDOF = groups[block][grp]->group_data->num_dof;
          data = groups[block][grp]->sol[set];
          offsets = wkset[block]->offsets;
          break;
        case 1 : // deprecated (u_dot)
          break;
        case 2 :
          LIDs = groups[block][grp]->LIDs[set];
          numDOF = groups[block][grp]->group_data->num_dof;
          data = groups[block][grp]->phi[set];
          offsets = wkset[block]->offsets;
          break;
        case 3 : // deprecated (phi_dot)
          break;
        case 4:
          LIDs = groups[block][grp]->paramLIDs;
          numDOF = groups[block][grp]->group_data->num_param_dof;
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
            numDOF = boundary_groups[block][grp]->group_data->num_dof;
            data = boundary_groups[block][grp]->sol[set];
            offsets = wkset[block]->offsets;
            break;
          case 1 : // deprecated (u_dot)
            break;
          case 2 :
            LIDs = boundary_groups[block][grp]->LIDs[set];
            numDOF = boundary_groups[block][grp]->group_data->num_dof;
            data = boundary_groups[block][grp]->phi[set];
            offsets = wkset[block]->offsets;
            break;
          case 3 : // deprecated (phi_dot)
            break;
          case 4:
            LIDs = boundary_groups[block][grp]->paramLIDs;
            numDOF = boundary_groups[block][grp]->group_data->num_param_dof;
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
  
  //Teuchos::TimeMonitor localtimer(*scatter_timer);
  
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
      LO cols[MAXDERIVS];
      ScalarT vals[MAXDERIVS];
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
  
  //Teuchos::TimeMonitor localtimer(*scatter_timer);
  
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
template<class MatType, class VecViewType, class LIDViewType, class EvalT>
void AssemblyManager<Node>::scatter(const size_t & set, MatType J_kcrs, VecViewType res_view,
                                    LIDViewType LIDs, LIDViewType paramLIDs,
                                    const int & block,
                                    const bool & compute_jacobian,
                                    const bool & compute_sens,
                                    const bool & compute_disc_sens,
                                    const bool & isAdjoint, EvalT & dummyval) {
#ifndef MrHyDE_NO_AD
  if (std::is_same<EvalT, AD>::value) {
    this->scatter(wkset_AD[block], set, J_kcrs, res_view, LIDs, paramLIDs, block, 
                  compute_jacobian, compute_sens, compute_disc_sens, isAdjoint);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->scatter(wkset_AD2[block], set, J_kcrs, res_view, LIDs, paramLIDs, block, 
                  compute_jacobian, compute_sens, compute_disc_sens, isAdjoint);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->scatter(wkset_AD4[block], set, J_kcrs, res_view, LIDs, paramLIDs, block, 
                  compute_jacobian, compute_sens, compute_disc_sens, isAdjoint);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->scatter(wkset_AD8[block], set, J_kcrs, res_view, LIDs, paramLIDs, block, 
                  compute_jacobian, compute_sens, compute_disc_sens, isAdjoint);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->scatter(wkset_AD16[block], set, J_kcrs, res_view, LIDs, paramLIDs, block, 
                  compute_jacobian, compute_sens, compute_disc_sens, isAdjoint);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->scatter(wkset_AD18[block], set, J_kcrs, res_view, LIDs, paramLIDs, block, 
                  compute_jacobian, compute_sens, compute_disc_sens, isAdjoint);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->scatter(wkset_AD24[block], set, J_kcrs, res_view, LIDs, paramLIDs, block, 
                  compute_jacobian, compute_sens, compute_disc_sens, isAdjoint);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->scatter(wkset_AD32[block], set, J_kcrs, res_view, LIDs, paramLIDs, block, 
                  compute_jacobian, compute_sens, compute_disc_sens, isAdjoint);
  }
#endif
}

template<class Node>
template<class MatType, class VecViewType, class LIDViewType, class EvalT>
void AssemblyManager<Node>::scatter(Teuchos::RCP<Workset<EvalT> > & wset, const size_t & set, MatType J_kcrs, VecViewType res_view,
                                    LIDViewType LIDs, LIDViewType paramLIDs,
                                    const int & block,
                                    const bool & compute_jacobian,
                                    const bool & compute_sens,
                                    const bool & compute_disc_sens,
                                    const bool & isAdjoint) {

  Teuchos::TimeMonitor localtimer(*scatter_timer);
  
  typedef typename Node::execution_space LA_exec;
  
  /////////////////////////////////////
  // This scatter needs to happen on the LA_device due to the use of J_kcrs->sumIntoValues()
  // Could be changed to the AssemblyDevice, but would require a mirror view of this data and filling such a view is nontrivial
  /////////////////////////////////////
  
  // Make sure the functor can access the necessary data
  auto fixedDOF = isFixedDOF[set];
  auto res = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
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
      LO cols[MAXDERIVS];
      ScalarT vals[MAXDERIVS];
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


//==============================================================
// Scatter res and use wkset->res
//==============================================================

template<class Node>
template<class VecViewType, class LIDViewType>
void AssemblyManager<Node>::scatterRes(const size_t & set, VecViewType res_view,
                                       LIDViewType LIDs, const int & block) {
  
  Teuchos::TimeMonitor localtimer(*scatter_timer);
  
  typedef typename Node::execution_space LA_exec;
  
  /////////////////////////////////////
  // This scatter needs to happen on the LA_device due to the use of J_kcrs->sumIntoValues()
  // Could be changed to the AssemblyDevice, but would require a mirror view of this data and filling such a view is nontrivial
  /////////////////////////////////////
  
  // Make sure the functor can access the necessary data
  auto fixedDOF = isFixedDOF[set];
  auto res = wkset[block]->res;
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;
  
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
            ScalarT val = -res(elem,row);
            if (use_atomics_) {
              Kokkos::atomic_add(&(res_view(rowIndex,0)), val);
            }
            else {
              res_view(rowIndex,0) += val;
            }
          
        }
      }
    }
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
#ifndef MrHyDE_NO_AD
    if (wkset_AD[block]->isInitialized) {
      wkset_AD[block]->updatePhysicsSet(set);
    }
    if (wkset_AD2[block]->isInitialized) {
      wkset_AD2[block]->updatePhysicsSet(set);
    }
    if (wkset_AD4[block]->isInitialized) {
      wkset_AD4[block]->updatePhysicsSet(set);
    }
    if (wkset_AD8[block]->isInitialized) {
      wkset_AD8[block]->updatePhysicsSet(set);
    }
    if (wkset_AD16[block]->isInitialized) {
      wkset_AD16[block]->updatePhysicsSet(set);
    }
    if (wkset_AD18[block]->isInitialized) {
      wkset_AD18[block]->updatePhysicsSet(set);
    }
    if (wkset_AD24[block]->isInitialized) {
      wkset_AD24[block]->updatePhysicsSet(set);
    }
    if (wkset_AD32[block]->isInitialized) {
      wkset_AD32[block]->updatePhysicsSet(set);
    }
#endif
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::buildDatabase(const size_t & block) {
  
  vector<std::pair<size_t,size_t> > first_users; // stores <grpID,elemID>
  vector<std::pair<size_t,size_t> > first_boundary_users; // stores <grpID,elemID>
  
  /////////////////////////////////////////////////////////////////////////////
  // Step 1: identify the duplicate information
  /////////////////////////////////////////////////////////////////////////////
  
  this->identifyVolumetricDatabase(block, first_users);
  
  this->identifyBoundaryDatabase(block, first_boundary_users);
  
  /////////////////////////////////////////////////////////////////////////////
  // Step 2: inform the user about the savings
  /////////////////////////////////////////////////////////////////////////////
  
  size_t totalelem = 0, boundaryelem = 0;
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    totalelem += groups[block][grp]->numElem;
  }
  for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
    boundaryelem += boundary_groups[block][grp]->numElem;
  }
  
  if (verbosity > 5) {
    cout << " - Processor " << comm->getRank() << ": Number of elements on block " << blocknames[block] << ": " << totalelem << endl;
    cout << " - Processor " << comm->getRank() << ": Number of unique elements on block " << blocknames[block] << ": " << first_users.size() << endl;
    cout << " - Processor " << comm->getRank() << ": Database memory savings on " << blocknames[block] << ": "
    << (100.0 - 100.0*((double)first_users.size()/(double)totalelem)) << "%" << endl;
    cout << " - Processor " << comm->getRank() << ": Number of boundary elements on block " << blocknames[block] << ": " << boundaryelem << endl;
    cout << " - Processor " << comm->getRank() << ": Number of unique boundary elements on block " << blocknames[block] << ": " << first_boundary_users.size() << endl;
    cout << " - Processor " << comm->getRank() << ": Database boundary memory savings on " << blocknames[block] << ": "
    << (100.0 - 100.0*((double)first_boundary_users.size()/(double)boundaryelem)) << "%" << endl;
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Step 3: build the database
  /////////////////////////////////////////////////////////////////////////////
  
  this->buildVolumetricDatabase(block, first_users);
  
  this->buildBoundaryDatabase(block, first_boundary_users);
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::identifyVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users) {
  Teuchos::TimeMonitor localtimer(*group_database_create_timer);
  
  double database_TOL = settings->sublist("Solver").get<double>("database TOL",1.0e-10);
  
  int dimension = groupData[block]->dimension;
  size_type numip = groupData[block]->ref_ip.extent(0);
  bool ignore_orientations = true;
  for (size_t i=0; i<groupData[block]->basis_pointers.size(); ++i) {
    if (groupData[block]->basis_pointers[i]->requireOrientation()) {
      ignore_orientations = false;
    }
  }
  
  vector<Kokkos::View<ScalarT***,HostDevice>> db_jacobians;
  vector<ScalarT> db_measures;
  
  // There are only so many unique orientation
  // Creating a short list of the unique ones and the index for each element
  
  vector<string> unique_orients;
  vector<vector<size_t> > all_orients;
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    auto orient_host = create_mirror_view(groups[block][grp]->orientation);
    deep_copy(orient_host, groups[block][grp]->orientation);
    vector<size_t> grp_orient(groups[block][grp]->numElem);
    for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
      string orient = orient_host(e).to_string();
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
  
  // Write data to file (for clustering)
  
  bool write_volumetric_data = settings->sublist("Solver").get<bool>("write volumetric data",false);
  if (write_volumetric_data) {
    this->writeVolumetricData(block, all_orients);
  }
  
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    groups[block][grp]->storeAll = false;
    Kokkos::View<LO*,AssemblyDevice> index("basis database index",groups[block][grp]->numElem);
    auto index_host = create_mirror_view(index);
    
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

      //ScalarT refmeas = std::pow(measure_host(e),1.0/dimension);
            
      while (!found && prog<first_users.size()) {
        size_t refgrp = first_users[prog].first;
        size_t refelem = first_users[prog].second;
        
        // Check #1: element orientations
        size_t orient = all_orients[grp][e];
        size_t reforient = all_orients[refgrp][refelem];
        if (ignore_orientations || orient == reforient) {
          
          // Check #2: element measures
          ScalarT diff = std::abs(measure_host(e)-db_measures[prog]);
          
          if (std::abs(diff/db_measures[prog])<database_TOL) { // abs(measure) is probably unnecessary here
            
            // Check #3: element Jacobians
            size_type pt = 0;
            bool ruled_out = false;
            while (pt<numip && !ruled_out) { 
              ScalarT fronorm = 0.0;
              ScalarT frodiff = 0.0;
              ScalarT diff = 0.0;
              for (size_type d0=0; d0<jacobian_host.extent(2); ++d0) {
                for (size_type d1=0; d1<jacobian_host.extent(3); ++d1) {
                  diff = jacobian_host(e,pt,d0,d1)-db_jacobians[prog](pt,d0,d1);
                  frodiff += diff*diff;
                  fronorm += jacobian_host(e,pt,d0,d1)*jacobian_host(e,pt,d0,d1);
                }
              }
              if (std::sqrt(frodiff)/std::sqrt(fronorm) > database_TOL) {
                ruled_out = true;
              }
              pt++;
            }
            
            if (!ruled_out) {
              found = true;
              index_host(e) = prog;
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
        index_host(e) = first_users.size();
        std::pair<size_t,size_t> newuj{grp,e};
        first_users.push_back(newuj);
        
        Kokkos::View<ScalarT***,HostDevice> new_jac("new db jac",numip, dimension, dimension);
        for (size_type pt=0; pt<new_jac.extent(0); ++pt) {
          for (size_type d0=0; d0<new_jac.extent(1); ++d0) {
            for (size_type d1=0; d1<new_jac.extent(2); ++d1) {
              new_jac(pt,d0,d1) = jacobian(e,pt,d0,d1);
            }
          }
        }
        db_jacobians.push_back(new_jac);
        db_measures.push_back(measure_host(e));
        
      }
    }
    deep_copy(index,index_host);
    groups[block][grp]->basis_index = index;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::identifyBoundaryDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_boundary_users) {
  
  Teuchos::TimeMonitor localtimer(*group_database_create_timer);

  double database_TOL = settings->sublist("Solver").get<double>("database TOL",1.0e-10);
  
  int dimension = groupData[block]->dimension;
  size_type numip = groupData[block]->ref_ip.extent(0);
  
  vector<Kokkos::View<ScalarT***,HostDevice>> db_jacobians;
  vector<ScalarT> db_measures, db_jacobian_norms;
  
  // There are only so many unique orientation
  // Creating a short list of the unique ones and the index for each element
  vector<string> unique_orients;
  vector<vector<size_t> > all_orients;
  for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
    vector<size_t> grp_orient(boundary_groups[block][grp]->numElem);
    auto orient_host = create_mirror_view(boundary_groups[block][grp]->orientation);
    deep_copy(orient_host,boundary_groups[block][grp]->orientation);
    
    for (size_t e=0; e<boundary_groups[block][grp]->numElem; ++e) {
      string orient = orient_host(e).to_string();
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
    Kokkos::View<LO*,AssemblyDevice> index("basis database index",boundary_groups[block][grp]->numElem);
    auto index_host = create_mirror_view(index);
    
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
            ScalarT diff = std::abs(measure_host(e)-db_measures[prog]);
            //ScalarT refmeas = std::pow(db_measures[prog],1.0/dimension);
            ScalarT refnorm = db_jacobian_norms[prog];

            if (std::abs(diff/db_measures[prog])<database_TOL) {
              
              // Check #3: element Jacobians
              
              ScalarT diff2 = 0.0;
              for (size_type pt=0; pt<jacobian_host.extent(1); ++pt) {
                for (size_type d0=0; d0<jacobian_host.extent(2); ++d0) {
                  for (size_type d1=0; d1<jacobian_host.extent(3); ++d1) {
                    diff2 += std::abs(jacobian_host(e,pt,d0,d1) - db_jacobians[prog](pt,d0,d1));
                  }
                }
              }
              
              if (std::abs(diff2/refnorm)<database_TOL) {
                found = true;
                index_host(e) = prog;
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
        index_host(e) = first_boundary_users.size();
        std::pair<size_t,size_t> newuj{grp,e};
        first_boundary_users.push_back(newuj);
        
        ScalarT jnorm = 0.0;
        Kokkos::View<ScalarT***,HostDevice> new_jac("new db jac",numip, dimension, dimension);
        for (size_type pt=0; pt<new_jac.extent(0); ++pt) {
          for (size_type d0=0; d0<new_jac.extent(1); ++d0) {
            for (size_type d1=0; d1<new_jac.extent(2); ++d1) {
              new_jac(pt,d0,d1) = jacobian(e,pt,d0,d1);
              jnorm += std::abs(jacobian(e,pt,d0,d1));
            }
          }
        }
        db_jacobians.push_back(new_jac);
        db_measures.push_back(measure_host(e));
        db_jacobian_norms.push_back(jnorm);
        
      }
    }
    deep_copy(index,index_host);
    boundary_groups[block][grp]->basis_index = index;
  }
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::buildVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users) {
  
  Teuchos::TimeMonitor localtimer(*group_database_basis_timer);
  
  using namespace std;

  int dimension = groupData[block]->dimension;
  
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
    auto nodes_host = create_mirror_view(groups[block][refgrp]->nodes);
    deep_copy(nodes_host, groups[block][refgrp]->nodes);
    
    for (size_type node=0; node<database_nodes.extent(1); ++node) {
      for (size_type dim=0; dim<database_nodes.extent(2); ++dim) {
        database_nodes_host(e,node,dim) = nodes_host(refelem,node,dim);
      }
    }
    
    // Get the orientations on the host
    auto orientations_host = create_mirror_view(groups[block][refgrp]->orientation);
    deep_copy(orientations_host, groups[block][refgrp]->orientation);
    database_orientation_host(e) = orientations_host(refelem);
    
    // Get the wts on the host
    auto wts_host = create_mirror_view(groups[block][refgrp]->wts);
    deep_copy(wts_host, groups[block][refgrp]->wts);
    
    for (size_type pt=0; pt<database_wts_host.extent(1); ++pt) {
      database_wts_host(e,pt) = wts_host(refelem,pt);
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
    for (size_type side=0; side<groupData[block]->num_sides; side++) {
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
    for (size_t set=0; set<physics->set_names.size(); ++set) {
      View_Sc3 mass("local mass",database_numElem, groups[block][0]->LIDs[set].extent(1),
                    groups[block][0]->LIDs[set].extent(1));
      
      auto offsets = wkset[block]->set_offsets[set];
      auto numDOF = groupData[block]->set_num_dof[set];
      
      bool use_sparse_quad = settings->sublist("Solver").get<bool>("use sparsifying mass quadrature",false);
      bool include_high_order = false;

      for (size_type n=0; n<numDOF.extent(0); n++) {
        string btype = wkset[block]->basis_types[wkset[block]->set_usebasis[set][n]];
        
        vector<vector<string> > qrules;

        if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
          // throw an error
        }
        else if (btype.substr(0,4) == "HDIV") {            
          //vector<string> qrule1 = {"GAUSS","GAUSS","GAUSS"};
          //qrules.push_back(qrule1);
          //vector<string> qrule2 = {"GAUSS-LOBATTO","GAUSS","GAUSS"};
          //qrules.push_back(qrule2);
          //vector<string> qrule3 = {"GAUSS","GAUSS-LOBATTO","GAUSS"};
          //qrules.push_back(qrule3);
          //vector<string> qrule4 = {"GAUSS","GAUSS","GAUSS-LOBATTO"};
          //qrules.push_back(qrule4);
        }
        else if (btype.substr(0,5) == "HCURL") {    
          vector<string> qrule1 = {"GAUSS-LOBATTO","GAUSS-LOBATTO","GAUSS-LOBATTO"};
          qrules.push_back(qrule1);
          //vector<string> qrule2 = {"GAUSS","GAUSS-LOBATTO","GAUSS-LOBATTO"};
          //qrules.push_back(qrule2);
          //vector<string> qrule3 = {"GAUSS-LOBATTO","GAUSS","GAUSS-LOBATTO"};
          //qrules.push_back(qrule3);
          //vector<string> qrule4 = {"GAUSS-LOBATTO","GAUSS-LOBATTO","GAUSS"};
          //qrules.push_back(qrule4);
        }
        else {
         // throw an error
        }

        if (use_sparse_quad && qrules.size() > 0) {
          ScalarT mwt = physics->mass_wts[set][block][n];
          View_Sc4 cbasis = tbasis[wkset[block]->set_usebasis[set][n]];
          
          View_Sc3 mass_sparse("local mass", mass.extent(0), cbasis.extent(1), cbasis.extent(1)); 
          
          if (include_high_order) {
            auto cwts = database_wts;
          //for (size_type n=0; n<numDOF.extent(0); n++) {
            ScalarT mwt = physics->mass_wts[set][block][n];
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
                      mass_sparse(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k)*mwt;
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
                        mass_sparse(e,off(i),off(j)) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k)*mwt;
                      }
                    }
                  }
                }
              });
            }
          //}
          }
          for (size_t q=0; q<qrules.size(); ++q) {
            vector<string> qrule = qrules[q];
            DRV cwts;
            DRV cbasis = disc->evaluateBasisNewQuadrature(block, wkset[block]->set_usebasis[set][n], qrule,
                                                          database_nodes, database_orientation, cwts);

            View_Sc3 newmass("local mass", mass.extent(0), cbasis.extent(1), cbasis.extent(1)); 
              
            parallel_for("Group get mass",
                         RangePolicy<AssemblyExec>(0,mass.extent(0)),
                         KOKKOS_LAMBDA (const size_type e ) {
              for (size_type i=0; i<cbasis.extent(1); i++ ) {
                for (size_type j=0; j<cbasis.extent(1); j++ ) {
                  for (size_type k=0; k<cbasis.extent(2); k++ ) {
                    for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                      newmass(e,i,j) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k)*mwt;
                    }
                  }
                }
              }
            });
            if (q==0 && !include_high_order) { // first qrule is the base rule
              deep_copy(mass_sparse,newmass);
            }
            else { // see if any alternative rules are better for each entry
              parallel_for("Group get mass",
                           RangePolicy<AssemblyExec>(0,mass.extent(0)),
                           KOKKOS_LAMBDA (const size_type e ) {
            
                for (size_type i=0; i<newmass.extent(1); i++ ) {
                  for (size_type j=0; j<newmass.extent(2); j++ ) {
                    if (i==j) {
                      if (newmass(e,i,j) > mass_sparse(e,i,j)) { // not using abs since always positive
                        mass_sparse(e,i,j) = newmass(e,i,j);
                      }
                    }
                    else {
                      if (abs(newmass(e,i,j)) < abs(mass_sparse(e,i,j))) {
                        mass_sparse(e,i,j) = newmass(e,i,j);
                      }
                    }
                  }
                }
              });   
            }     
          }
          
          // Permute the entries in mass matrix using the offsets
          auto off = subview(offsets,n,ALL());
          parallel_for("Group get mass",
                       RangePolicy<AssemblyExec>(0,mass.extent(0)),
                       KOKKOS_LAMBDA (const size_type e ) {
            for (size_type i=0; i<mass_sparse.extent(1); i++ ) {
              for (size_type j=0; j<mass_sparse.extent(2); j++ ) {
                mass(e,off(i),off(j)) = mass_sparse(e,i,j);
              }
            }
          });
        }
      
        else {
          auto cwts = database_wts;
          //for (size_type n=0; n<numDOF.extent(0); n++) {
            ScalarT mwt = physics->mass_wts[set][block][n];
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
          //}
        }
      }
      
      bool use_sparse = settings->sublist("Solver").get<bool>("sparse mass format",false);
      if (use_sparse) {
        ScalarT tol = settings->sublist("Solver").get<double>("sparse mass TOL",1.0e-10);
        Teuchos::RCP<Sparse3DView> sparse_mass = Teuchos::rcp( new Sparse3DView(mass,tol) );
        groupData[block]->sparse_database_mass.push_back(sparse_mass);
        groupData[block]->use_sparse_mass = true;
        cout << " - Processor " << comm->getRank() << ": Sparse mass format savings on " << blocknames[block] << ": "
             << (100.0 - 100.0*((double)sparse_mass->size()/(double)mass.size())) << "%" << endl;
      }
      //else {
        groupData[block]->database_mass.push_back(mass);
      //}

      bool write_matrices_to_file = false;
      if (write_matrices_to_file) {
        std::stringstream ss;
        ss << set;
        string filename = "mass_matrices." + ss.str() + ".out";
        KokkosTools::printToFile(mass,filename);
      }
    }
  }
}


/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::buildBoundaryDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_boundary_users) {
  
  int dimension = groupData[block]->dimension;
  size_type numsideip = groupData[block]->ref_side_ip[0].extent(0);
  
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
    auto nodes_host = create_mirror_view(boundary_groups[block][refgrp]->nodes);
    deep_copy(nodes_host, boundary_groups[block][refgrp]->nodes);
    
    for (size_type node=0; node<database_bnodes.extent(1); ++node) {
      for (size_type dim=0; dim<database_bnodes.extent(2); ++dim) {
        database_bnodes_host(0,node,dim) = nodes_host(refelem,node,dim);
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

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::writeVolumetricData(const size_t & block, vector<vector<size_t>> & all_orients) {
  
  vector<vector<ScalarT> > all_meas;
  vector<vector<ScalarT> > all_fros;
  int dimension = groupData[block]->dimension;
  size_type numip = groupData[block]->ref_ip.extent(0);
  
  for (size_t grp=0; grp<groups[block].size(); ++grp) {
    
    // Get the Jacobian for this group
    DRV jacobian("jacobian", groups[block][grp]->numElem, numip, dimension, dimension);
    disc->getJacobian(groupData[block], groups[block][grp]->nodes, jacobian);
    
    // Get the measures for this group
    DRV measure("measure", groups[block][grp]->numElem);
    disc->getMeasure(groupData[block], jacobian, measure);
    
    DRV fro("fro norm of J", groups[block][grp]->numElem);
    disc->getFrobenius(groupData[block], jacobian, fro);
    vector<ScalarT> currmeas;
    for (size_type e=0; e<measure.extent(0); ++e) {
      currmeas.push_back(measure(e));
      //currmeas.push_back(jacobian(e,0,0,0));
    }
    all_meas.push_back(currmeas);
    
    
    vector<ScalarT> currfro;
    for (size_type e=0; e<fro.extent(0); ++e) {
      ScalarT val = 0.0;
      for (size_type d1=0; d1<jacobian.extent(2); ++d1) {
        for (size_type d2=0; d2<jacobian.extent(3); ++d2) {
          for (size_type pt=0; pt<jacobian.extent(1); ++pt) {
            ScalarT cval = jacobian(e,pt,d1,d2) - jacobian(e,pt,d2,d1);
            val += cval*cval;
          }
        }
      }
      currfro.push_back(val);
      //currfro.push_back(jacobian(e,0,1,1));
    }
    all_fros.push_back(currfro);
  }
  
  if (comm->getRank() == 0) {
    
    string outfile = "jacobian_data.out";
    std::ofstream respOUT(outfile.c_str());
    respOUT.precision(16);
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      // Get the Jacobian for this group
      DRV jac("jacobian", groups[block][grp]->numElem, numip, dimension, dimension);
      disc->getJacobian(groupData[block], groups[block][grp]->nodes, jac);
      
      DRV wts("jacobian", groups[block][grp]->numElem, numip);
      disc->getPhysicalWts(groupData[block], groups[block][grp]->nodes, jac, wts);
      
      for (size_t e=0; e<groups[block][grp]->numElem; ++e) {
        /*
        ScalarT j00 = 0.0, j01 = 0.0, j02 = 0.0;
        ScalarT j10 = 0.0, j11 = 0.0, j12 = 0.0;
        ScalarT j20 = 0.0, j21 = 0.0, j22 = 0.0;
        
        for (size_type pt=0; pt<jac.extent(1); ++pt) {
          j00 += jac(e,pt,0,0)*wts(e,pt);
          j01 += jac(e,pt,0,1)*wts(e,pt);
          j02 += jac(e,pt,0,2)*wts(e,pt);
          j10 += jac(e,pt,1,0)*wts(e,pt);
          j11 += jac(e,pt,1,1)*wts(e,pt);
          j12 += jac(e,pt,1,2)*wts(e,pt);
          j20 += jac(e,pt,2,0)*wts(e,pt);
          j21 += jac(e,pt,2,1)*wts(e,pt);
          j22 += jac(e,pt,2,2)*wts(e,pt);
        }
        */

        //respOUT << j00 << ", " << j01 << ", " << j02 << ", " << j10 << ", " << j11 << ", " << j12 << ", " << j20 << ", " << j21 << ", " << j22 << endl;
        respOUT << all_orients[grp][e] << ", " << all_meas[grp][e] << ", " << all_fros[grp][e] << endl;
      }
    }
    respOUT.close();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we finalize the function managers
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::finalizeFunctions() {
  for (size_t block=0; block<wkset.size(); ++block) {
    this->finalizeFunctions(function_managers[block], wkset[block]);
  }
#ifndef MrHyDE_NO_AD
  for (size_t block=0; block<wkset_AD.size(); ++block) {
    this->finalizeFunctions(function_managers_AD[block], wkset_AD[block]);
  }
  for (size_t block=0; block<wkset_AD2.size(); ++block) {
    this->finalizeFunctions(function_managers_AD2[block], wkset_AD2[block]);
  }
  for (size_t block=0; block<wkset_AD4.size(); ++block) {
    this->finalizeFunctions(function_managers_AD4[block], wkset_AD4[block]);
  }
  for (size_t block=0; block<wkset_AD8.size(); ++block) {
    this->finalizeFunctions(function_managers_AD8[block], wkset_AD8[block]);
  }
  for (size_t block=0; block<wkset_AD16.size(); ++block) {
    this->finalizeFunctions(function_managers_AD16[block], wkset_AD16[block]);
  }
  for (size_t block=0; block<wkset_AD18.size(); ++block) {
    this->finalizeFunctions(function_managers_AD18[block], wkset_AD18[block]);
  }
  for (size_t block=0; block<wkset_AD24.size(); ++block) {
    this->finalizeFunctions(function_managers_AD24[block], wkset_AD24[block]);
  }
  for (size_t block=0; block<wkset_AD32.size(); ++block) {
    this->finalizeFunctions(function_managers_AD32[block], wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::finalizeFunctions(Teuchos::RCP<FunctionManager<EvalT> > & fman,
                                              Teuchos::RCP<Workset<EvalT> > & wset) {
  fman->setupLists(params->paramnames);
  fman->wkset = wset;
  if (wset->isInitialized) {
    fman->decomposeFunctions();
    if (verbosity >= 20) {
      fman->printFunctions();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Functionality moved from boundary groups into here
////////////////////////////////////////////////////////////////////////////////


template<class Node>
void AssemblyManager<Node>::computeJacResBoundary(const int & block, const size_t & grp,
                                                     const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                                     const bool & compute_jacobian, const bool & compute_sens,
                                                     const int & num_active_params, const bool & compute_disc_sens,
                                                     const bool & compute_aux_sens, const bool & store_adjPrev,
                                                     View_Sc3 local_res, View_Sc3 local_J) {
  
#ifndef MrHyDE_NO_AD

  int seedwhat = 0;
  if (compute_jacobian) {
    if (compute_disc_sens) {
      seedwhat = 3;
    }
    else if (compute_aux_sens) {
      seedwhat = 4;
    }
    else {
      seedwhat = 1;
    }
  }
  this->updateWorksetBoundary<AD>(block, grp, seedwhat);
  physics->boundaryResidual<AD>(wkset_AD[block]->current_set, block);
  
  if (compute_jacobian) {
    if (compute_disc_sens) {
      this->updateParamJacBoundary(block, grp, local_J);
    }
    else if (compute_aux_sens){
      this->updateAuxJacBoundary(block, grp, local_J);
    }
    else {
      this->updateJacBoundary(block, grp, isAdjoint, local_J);
    }
  }
  
  if (!isAdjoint) {
    this->updateResBoundary(block, grp, compute_sens, local_res);
  }
#endif

}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetBoundary(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
  this->updateWorksetBoundary(wkset[block], block, grp, seedwhat, seedindex, override_transient);
}


template<class Node>
void AssemblyManager<Node>::updateWorksetBoundaryAD(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateWorksetBoundary(wkset_AD[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateWorksetBoundary(wkset_AD2[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateWorksetBoundary(wkset_AD4[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateWorksetBoundary(wkset_AD8[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateWorksetBoundary(wkset_AD16[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateWorksetBoundary(wkset_AD18[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateWorksetBoundary(wkset_AD24[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateWorksetBoundary(wkset_AD32[block], block, grp, seedwhat, seedindex, override_transient);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetBoundary(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {

  
  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateWorksetBoundary(wkset[block], block, grp, seedwhat, seedindex, override_transient);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateWorksetBoundary(wkset_AD[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateWorksetBoundary(wkset_AD2[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateWorksetBoundary(wkset_AD4[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateWorksetBoundary(wkset_AD8[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateWorksetBoundary(wkset_AD16[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateWorksetBoundary(wkset_AD18[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateWorksetBoundary(wkset_AD24[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateWorksetBoundary(wkset_AD32[block], block, grp, seedwhat, seedindex, override_transient);
  }
#endif  
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetBoundary(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp, 
                                                  const int & seedwhat, const int & seedindex, const bool & override_transient) {
  
  ///////////////////////////////////////////////////////////
  // Reset the residual and data in the workset
  
  wset->reset();
  wset->sidename = boundary_groups[block][grp]->sidename;
  wset->currentside = boundary_groups[block][grp]->sidenum;
  wset->numElem = boundary_groups[block][grp]->numElem;

  ///////////////////////////////////////////////////////////
  // Update the observational data stored in the workset
  // This is specific to cases with data-based parameters

  this->updateDataBoundary<EvalT>(block, grp);
  
  ///////////////////////////////////////////////////////////
  // Update the integration info and basis in workset

  this->updateWorksetBasisBoundary<EvalT>(block, grp);
  
  ///////////////////////////////////////////////////////////
  // Map the gathered solution to seeded version in workset
  if (groupData[block]->requires_transient && !override_transient) {
    for (size_t set=0; set<groupData[block]->num_sets; ++set) {
      wset->computeSolnTransientSeeded(set, boundary_groups[block][grp]->sol[set], 
                                        boundary_groups[block][grp]->sol_prev[set], 
                                        boundary_groups[block][grp]->sol_stage[set], 
                                        seedwhat, seedindex);
    }
  }
  else { // steady-state
    for (size_t set=0; set<groupData[block]->num_sets; ++set) {
      wset->computeSolnSteadySeeded(set, boundary_groups[block][grp]->sol[set], seedwhat);
    }
  }
  if (wset->numParams > 0) {
    wset->computeParamSteadySeeded(boundary_groups[block][grp]->param, seedwhat);
  }

  // Aux solutions are still handled separately
  //this->computeSoln(seedwhat);
  if (wset->numAux > 0 && std::is_same<EvalT,AD>::value) {
    
    this->computeBoundaryAux(block, grp, seedwhat);

  }
}


template<class Node>
void AssemblyManager<Node>::computeBoundaryAux(const int & block, const size_t & grp, const int & seedwhat) {

#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD[block]);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD2[block]);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD4[block]);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD8[block]);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD16[block]);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD18[block]);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD24[block]);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD32[block]);
  }

#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::computeBoundaryAux(const int & block, const size_t & grp, const int & seedwhat,
                                               Teuchos::RCP<Workset<EvalT> > & wset) {

#ifndef MrHyDE_NO_AD
  auto numAuxDOF = groupData[block]->num_aux_dof;
    
  for (size_type var=0; var<numAuxDOF.extent(0); var++) {
    auto abasis = boundary_groups[block][grp]->auxside_basis[boundary_groups[block][grp]->auxusebasis[var]];
    auto off = subview(boundary_groups[block][grp]->auxoffsets,var,ALL());
    string varname = wset->aux_varlist[var];
    auto local_aux = wset->getSolutionField("aux "+varname,false);
    Kokkos::deep_copy(local_aux,0.0);
    auto localID = boundary_groups[block][grp]->localElemID;
    auto varaux = subview(boundary_groups[block][grp]->aux, ALL(), var, ALL());
    if (seedwhat == 4) {
      parallel_for("bgroup aux 4",
                   TeamPolicy<AssemblyExec>(localID.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        EvalT dummyval = 0.0;
        for (size_type pt=team.team_rank(); pt<abasis.extent(2); pt+=team.team_size() ) {
          for (size_type dof=0; dof<abasis.extent(1); ++dof) {
            EvalT auxval = EvalT(dummyval.size(),off(dof), varaux(localID(elem),dof));
            local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
          }
        }
      });
    }
    else {
      parallel_for("bgroup aux 5",
                    TeamPolicy<AssemblyExec>(localID.extent(0), Kokkos::AUTO, VECTORSIZE),
                    KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<abasis.extent(2); pt+=team.team_size() ) {
          for (size_type dof=0; dof<abasis.extent(1); ++dof) {
            ScalarT auxval = varaux(localID(elem),dof);
            local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
          }
        }
      });
    }
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateDataBoundaryAD(const int & block, const size_t & grp) {
  this->updateDataBoundary<AD>(block, grp);
}

template<class Node>
void AssemblyManager<Node>::updateDataBoundary(const int & block, const size_t & grp) {
  this->updateDataBoundary<ScalarT>(block, grp);
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateDataBoundary(const int & block, const size_t & grp) {

  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateDataBoundary(wkset[block], block, grp);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateDataBoundary(wkset_AD[block], block, grp);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateDataBoundary(wkset_AD2[block], block, grp);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateDataBoundary(wkset_AD4[block], block, grp);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateDataBoundary(wkset_AD8[block], block, grp);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateDataBoundary(wkset_AD16[block], block, grp);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateDataBoundary(wkset_AD18[block], block, grp);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateDataBoundary(wkset_AD24[block], block, grp);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateDataBoundary(wkset_AD32[block], block, grp);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateDataBoundary(Teuchos::RCP<Workset<EvalT> > & wset,
                                               const int & block, const size_t & grp) {

  // hard coded for what I need it for right now
  if (groupData[block]->have_phi) {
    wset->have_rotation_phi = true;
    wset->rotation_phi = boundary_groups[block][grp]->data;
    wset->allocateRotations();
  }
  else if (groupData[block]->have_rotation) {
    wset->have_rotation = true;
    wset->allocateRotations();
    auto rot = wset->rotation;
    auto data = boundary_groups[block][grp]->data;
    parallel_for("update data",
                 RangePolicy<AssemblyExec>(0,data.extent(0)),
                 KOKKOS_LAMBDA (const size_type e ) {
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
  
  }
  else if (groupData[block]->have_extra_data) {
    wset->extra_data = boundary_groups[block][grp]->data;
  }
  wset->multidata = boundary_groups[block][grp]->multidata;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetBasisBoundary(const int & block, const size_t & grp) {
  this->updateWorksetBasisBoundary<ScalarT>(block, grp);
}

template<class Node>
void AssemblyManager<Node>::updateWorksetBasisBoundaryAD(const int & block, const size_t & grp) {
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateWorksetBasisBoundary(wkset_AD[block], block, grp);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateWorksetBasisBoundary(wkset_AD2[block], block, grp);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateWorksetBasisBoundary(wkset_AD4[block], block, grp);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateWorksetBasisBoundary(wkset_AD8[block], block, grp);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateWorksetBasisBoundary(wkset_AD16[block], block, grp);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateWorksetBasisBoundary(wkset_AD18[block], block, grp);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateWorksetBasisBoundary(wkset_AD24[block], block, grp);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateWorksetBasisBoundary(wkset_AD32[block], block, grp);
  }

#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetBasisBoundary(const int & block, const size_t & grp) {

  
  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateWorksetBasisBoundary(wkset[block], block, grp);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateWorksetBasisBoundary(wkset_AD[block], block, grp);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateWorksetBasisBoundary(wkset_AD2[block], block, grp);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateWorksetBasisBoundary(wkset_AD4[block], block, grp);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateWorksetBasisBoundary(wkset_AD8[block], block, grp);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateWorksetBasisBoundary(wkset_AD16[block], block, grp);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateWorksetBasisBoundary(wkset_AD18[block], block, grp);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateWorksetBasisBoundary(wkset_AD24[block], block, grp);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateWorksetBasisBoundary(wkset_AD32[block], block, grp);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetBasisBoundary(Teuchos::RCP<Workset<EvalT> > & wset,
                                                       const int & block, const size_t & grp) {

  wset->wts_side = boundary_groups[block][grp]->wts;
  wset->h = boundary_groups[block][grp]->hsize;
  
  wset->setScalarField(boundary_groups[block][grp]->ip[0],"x");
  wset->setScalarField(boundary_groups[block][grp]->normals[0],"n[x]");
  wset->setScalarField(boundary_groups[block][grp]->tangents[0],"t[x]");
  if (boundary_groups[block][grp]->ip.size() > 1) {
    wset->setScalarField(boundary_groups[block][grp]->ip[1],"y");
    wset->setScalarField(boundary_groups[block][grp]->normals[1],"n[y]");
    wset->setScalarField(boundary_groups[block][grp]->tangents[1],"t[y]");
  }
  if (boundary_groups[block][grp]->ip.size() > 2) {
    wset->setScalarField(boundary_groups[block][grp]->ip[2],"z");
    wset->setScalarField(boundary_groups[block][grp]->normals[2],"n[z]");
    wset->setScalarField(boundary_groups[block][grp]->tangents[2],"t[z]");
  }

  if (boundary_groups[block][grp]->storeAll || groupData[block]->use_basis_database) {
    wset->basis_side = boundary_groups[block][grp]->basis;
    wset->basis_grad_side = boundary_groups[block][grp]->basis_grad;
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalBoundaryBasis(groupData[block], boundary_groups[block][grp]->nodes, 
                                   boundary_groups[block][grp]->localSideID, 
                                   boundary_groups[block][grp]->orientation,
                                   tbasis, tbasis_grad, tbasis_curl, tbasis_div);
    vector<CompressedView<View_Sc4>> tcbasis, tcbasis_grad;
    for (size_t i=0; i<tbasis.size(); ++i) {
      tcbasis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      tcbasis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
    }
    wset->basis_side = tcbasis;
    wset->basis_grad_side = tcbasis_grad;
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateResBoundary(const int & block, const size_t & grp,
                                      const bool & compute_sens, View_Sc3 local_res) {

#ifndef MrHyDE_NO_AD     
  if (wkset_AD[block]->isInitialized) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD[block]);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD2[block]);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD4[block]);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD8[block]);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD16[block]);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD18[block]);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD24[block]);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateResBoundary(block, grp, compute_sens, local_res, wkset_AD32[block]);
  }
#endif
}


template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateResBoundary(const int & block, const size_t & grp,
                                      const bool & compute_sens, View_Sc3 local_res,
                                      Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  if (compute_sens) {
    
    parallel_for("bgroup update res sens",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int r=0; r<local_res.extent(2); r++) {
            local_res(elem,offsets(n,j),r) -= res_AD(elem,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("bgroup update res",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(elem,offsets(n,j),0) -= res_AD(elem,offsets(n,j)).val();
        }
      }
    });
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateJacBoundary(const int & block, const size_t & grp, 
                                      const bool & useadjoint, View_Sc3 local_J) {

#ifndef MrHyDE_NO_AD     
  if (wkset_AD[block]->isInitialized) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD[block]);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD2[block]);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD4[block]);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD8[block]);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD16[block]);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD18[block]);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD24[block]);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateJacBoundary(block, grp, useadjoint, local_J, wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateJacBoundary(const int & block, const size_t & grp, 
                                      const bool & useadjoint, View_Sc3 local_J,
                                      Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  if (useadjoint) {
    parallel_for("bgroup update jac sens",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(m,k),offsets(n,j)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("bgroup update jac",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(n,j),offsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateParamJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J) {

#ifndef MrHyDE_NO_AD     
  if (wkset_AD[block]->isInitialized) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD[block]);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD2[block]);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD4[block]);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD8[block]);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD16[block]);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD18[block]);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD24[block]);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateParamJacBoundary(block, grp, local_J, wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateParamJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J,
                                                   Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  auto paramoffsets = wset->paramoffsets;
  auto numParamDOF = groupData[block]->num_param_dof;
  
  parallel_for("bgroup update param jac",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VECTORSIZE),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numParamDOF.extent(0); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_J(elem,offsets(n,j),paramoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateAuxJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J) {

#ifndef MrHyDE_NO_AD     
  if (wkset_AD[block]->isInitialized) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD[block]);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD2[block]);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD4[block]);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD8[block]);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD16[block]);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD18[block]);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD24[block]);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateAuxJacBoundary(block, grp, local_J, wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateAuxJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J,
                                                 Teuchos::RCP<Workset<EvalT> > & wset) {
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  auto aoffsets = boundary_groups[block][grp]->auxoffsets;
  auto numAuxDOF = groupData[block]->num_aux_dof;
  
  parallel_for("bgroup update aux jac",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VECTORSIZE),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numAuxDOF.extent(0); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(elem,offsets(n,j),aoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(aoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}


///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition
///////////////////////////////////////////////////////////////////////////////////////

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
      auto dip = groupData[block]->physics->getDirichlet(n, set, groupData[block]->my_block, boundary_groups[block][grp]->sidename);

      int bind = wkset[block]->usebasis[n];
      std::string btype = groupData[block]->basis_types[bind];
      auto cbasis = boundary_groups[block][grp]->basis[bind]; // may fault in memory-saving mode
      
      auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bgroup fill Dirichlet",
                     RangePolicy<AssemblyExec>(0,cwts.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cwts.extent(1); j++ ) {
              dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,0)*cwts(e,j);
            }
          }
        });
      }
      else if (btype == "HDIV"){
        
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
                     KOKKOS_LAMBDA (const int e ) {
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
        // not implemented yet
      }
    }
  }
  return dvals;
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
                     KOKKOS_LAMBDA (const int e ) {
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
                     KOKKOS_LAMBDA (const int e ) {
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
        // not implemented yet
      }
    }
  }
  return mass;
}

////////////////////////////////////////////////////////////////////////////////
// Functionality moved from groups into here
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorkset(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
  this->updateWorkset<ScalarT>(block, grp, seedwhat, seedindex, override_transient);
}

template<class Node>
void AssemblyManager<Node>::updateWorksetAD(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
#ifndef MrHyDE_NO_AD     
  if (wkset_AD[block]->isInitialized) {
    this->updateWorkset(wkset_AD[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateWorkset(wkset_AD2[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateWorkset(wkset_AD4[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateWorkset(wkset_AD8[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateWorkset(wkset_AD16[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateWorkset(wkset_AD18[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateWorkset(wkset_AD24[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateWorkset(wkset_AD32[block], block, grp, seedwhat, seedindex, override_transient);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorkset(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {

  
  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateWorkset(wkset[block], block, grp, seedwhat, seedindex, override_transient);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateWorkset(wkset_AD[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateWorkset(wkset_AD2[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateWorkset(wkset_AD4[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateWorkset(wkset_AD8[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateWorkset(wkset_AD16[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateWorkset(wkset_AD18[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateWorkset(wkset_AD24[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateWorkset(wkset_AD32[block], block, grp, seedwhat, seedindex, override_transient);
  }
#endif
}



template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorkset(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
    
  //Teuchos::TimeMonitor localtimer(*computeSolnVolTimer);
  
  // Reset the residual and data in the workset
  //auto wset = wkset[block];
  wset->reset();
  
  wset->numElem = groups[block][grp]->numElem;
  this->updateGroupData(wset, block, grp);
  
  wset->wts = groups[block][grp]->wts;
  wset->h = groups[block][grp]->hsize;

  wset->setScalarField(groups[block][grp]->ip[0],"x");
  if (groups[block][grp]->ip.size() > 1) {
    wset->setScalarField(groups[block][grp]->ip[1],"y");
  }
  if (groups[block][grp]->ip.size() > 2) {
    wset->setScalarField(groups[block][grp]->ip[2],"z");
  }

  // Update the integration info and basis in workset
  if (groups[block][grp]->storeAll || groups[block][grp]->group_data->use_basis_database) {
    wset->basis = groups[block][grp]->basis;
    wset->basis_grad = groups[block][grp]->basis_grad;
    wset->basis_div = groups[block][grp]->basis_div;
    wset->basis_curl = groups[block][grp]->basis_curl;
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricBasis(groups[block][grp]->group_data, groups[block][grp]->nodes, groups[block][grp]->orientation,
                                    tbasis, tbasis_grad, tbasis_curl,
                                    tbasis_div, tbasis_nodes);

    vector<CompressedView<View_Sc4>> tcbasis, tcbasis_grad, tcbasis_curl;
    vector<CompressedView<View_Sc3>> tcbasis_div;
    for (size_t i=0; i<tbasis.size(); ++i) {
      tcbasis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      tcbasis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
      tcbasis_div.push_back(CompressedView<View_Sc3>(tbasis_div[i]));
      tcbasis_curl.push_back(CompressedView<View_Sc4>(tbasis_curl[i]));
    }
    wset->basis = tcbasis;
    wset->basis_grad = tcbasis_grad;
    wset->basis_div = tcbasis_div;
    wset->basis_curl = tcbasis_curl;
  }
  
  // Map the gathered solution to seeded version in workset
  if (groups[block][grp]->group_data->requires_transient && !override_transient) {
    for (size_t set=0; set<groups[block][grp]->group_data->num_sets; ++set) {
      wset->computeSolnTransientSeeded(set, groups[block][grp]->sol[set], groups[block][grp]->sol_prev[set], 
                                               groups[block][grp]->sol_stage[set], seedwhat, seedindex);
    }
  }
  else { // steady-state
    for (size_t set=0; set<groups[block][grp]->group_data->num_sets; ++set) {
      wset->computeSolnSteadySeeded(set, groups[block][grp]->sol[set], seedwhat);
    }
  }
  if (wset->numParams > 0) {
    wset->computeParamSteadySeeded(groups[block][grp]->param, seedwhat);
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeSolAvg(const int & block, const size_t & grp) {
  
  // THIS FUNCTION ASSUMES THAT THE WORKSET BASIS HAS BEEN UPDATED
  
  //Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  
  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  auto cwts = wkset[block]->wts;
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("Group sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });
  
  for (size_t set=0; set<groups[block][grp]->sol_avg.size(); ++set) {
    
    // HGRAD vars
    vector<int> vars_HGRAD = wkset[block]->vars_HGRAD[set];
    vector<string> varlist_HGRAD = wkset[block]->varlist_HGRAD[set];
    for (size_t i=0; i<vars_HGRAD.size(); ++i) {
      auto sol = wkset[block]->getSolutionField(varlist_HGRAD[i]);
      auto savg = subview(groups[block][grp]->sol_avg[set],ALL(),vars_HGRAD[i],0);
      parallel_for("Group sol avg",
                   RangePolicy<AssemblyExec>(0,savg.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        ScalarT solavg = 0.0;
        for (size_type pt=0; pt<sol.extent(2); pt++) {
          solavg += sol(elem,pt)*cwts(elem,pt);
        }
        savg(elem) = solavg/avgwts(elem);
      });
    }
    
    // HVOL vars
    vector<int> vars_HVOL = wkset[block]->vars_HVOL[set];
    vector<string> varlist_HVOL = wkset[block]->varlist_HVOL[set];
    for (size_t i=0; i<vars_HVOL.size(); ++i) {
      auto sol = wkset[block]->getSolutionField(varlist_HVOL[i]);
      auto savg = subview(groups[block][grp]->sol_avg[set],ALL(),vars_HVOL[i],0);
      parallel_for("Group sol avg",
                   RangePolicy<AssemblyExec>(0,savg.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        ScalarT solavg = 0.0;
        for (size_type pt=0; pt<sol.extent(2); pt++) {
          solavg += sol(elem,pt)*cwts(elem,pt);
        }
        savg(elem) = solavg/avgwts(elem);
      });
    }
    
    // Compute the postfix options for vector vars
    vector<string> postfix = {"[x]"};
    if (groups[block][grp]->sol_avg[set].extent(2) > 1) { // 2D or 3D
      postfix.push_back("[y]");
    }
    if (groups[block][grp]->sol_avg[set].extent(2) > 2) { // 3D
      postfix.push_back("[z]");
    }
    
    // HDIV vars
    vector<int> vars_HDIV = wkset[block]->vars_HDIV[set];
    vector<string> varlist_HDIV = wkset[block]->varlist_HDIV[set];
    for (size_t i=0; i<vars_HDIV.size(); ++i) {
      for (size_t j=0; j<postfix.size(); ++j) {
        auto sol = wkset[block]->getSolutionField(varlist_HDIV[i]+postfix[j]);
        auto savg = subview(groups[block][grp]->sol_avg[set],ALL(),vars_HDIV[i],j);
        parallel_for("Group sol avg",
                     RangePolicy<AssemblyExec>(0,savg.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          ScalarT solavg = 0.0;
          for (size_type pt=0; pt<sol.extent(2); pt++) {
            solavg += sol(elem,pt)*cwts(elem,pt);
          }
          savg(elem) = solavg/avgwts(elem);
        });
      }
    }
    
    // HCURL vars
    vector<int> vars_HCURL = wkset[block]->vars_HCURL[set];
    vector<string> varlist_HCURL = wkset[block]->varlist_HCURL[set];
    for (size_t i=0; i<vars_HCURL.size(); ++i) {
      for (size_t j=0; j<postfix.size(); ++j) {
        auto sol = wkset[block]->getSolutionField(varlist_HCURL[i]+postfix[j]);
        auto savg = subview(groups[block][grp]->sol_avg[set],ALL(),vars_HCURL[i],j);
        parallel_for("Group sol avg",
                     RangePolicy<AssemblyExec>(0,savg.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          ScalarT solavg = 0.0;
          for (size_type pt=0; pt<sol.extent(2); pt++) {
            solavg += sol(elem,pt)*cwts(elem,pt);
          }
          savg(elem) = solavg/avgwts(elem);
        });
      }
    }
  }
  
  /*
  if (param_avg.extent(1) > 0) {
    View_AD4 psol = wkset[block]->local_param;
    auto pavg = param_avg;

    parallel_for("Group param avg",
                 RangePolicy<AssemblyExec>(0,pavg.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      ScalarT avgwt = 0.0;
      for (size_type pt=0; pt<cwts.extent(1); pt++) {
        avgwt += cwts(elem,pt);
      }
      for (size_type dof=0; dof<psol.extent(1); dof++) {
        for (size_type dim=0; dim<psol.extent(3); dim++) {
          ScalarT solavg = 0.0;
          for (size_type pt=0; pt<psol.extent(2); pt++) {
            solavg += psol(elem,dof,pt,dim).val()*cwts(elem,pt);
          }
          pavg(elem,dof,dim) = solavg/avgwt;
        }
      }
    });
  }
   */
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeSolutionAverage(const int & block, const size_t & grp,
                                                  const string & var, View_Sc2 csol) {
  
  //Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  
  // Figure out which basis we need
  int index;
  wkset[block]->isVar(var,index);
  
  CompressedView<View_Sc4> cbasis;
  auto cwts = groups[block][grp]->wts;

  if (groups[block][grp]->storeAll || groups[block][grp]->group_data->use_basis_database) {
    cbasis = groups[block][grp]->basis[wkset[block]->usebasis[index]];
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->nodes, groups[block][grp]->orientation,
                                     tbasis, tbasis_grad, tbasis_curl,
                                     tbasis_div, tbasis_nodes);
    cbasis = CompressedView<View_Sc4>(tbasis[wkset[block]->usebasis[index]]);
  }
  
  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("Group sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });
  
  size_t set = wkset[block]->current_set;
  auto scsol = subview(groups[block][grp]->sol[set],ALL(),index,ALL());
  parallel_for("wkset[block] soln ip HGRAD",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type dim=0; dim<cbasis.extent(3); ++dim) {
      ScalarT avgval = 0.0;
      for (size_type dof=0; dof<cbasis.extent(1); ++dof ) {
        for (size_type pt=0; pt<cbasis.extent(2); ++pt) {
          avgval += scsol(elem,dof)*cbasis(elem,dof,pt,dim)*cwts(elem,pt);
        }
      }
      csol(elem,dim) = avgval/avgwts(elem);
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeParameterAverage(const int & block, const size_t & grp,
                                                    const string & var, View_Sc2 sol) {
  
  //Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  
  // Figure out which basis we need
  int index;
  wkset[block]->isParameter(var,index);
  
  CompressedView<View_Sc4> cbasis;
  auto cwts = groups[block][grp]->wts;

  if (groups[block][grp]->storeAll || groupData[block]->use_basis_database) {
    cbasis = groups[block][grp]->basis[wkset[block]->paramusebasis[index]];
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->nodes, groups[block][grp]->orientation,
                                     tbasis, tbasis_grad, tbasis_curl,
                                     tbasis_div, tbasis_nodes);
    cbasis = CompressedView<View_Sc4>(tbasis[wkset[block]->paramusebasis[index]]);
  }
  
  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("Group sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });
  
  auto csol = subview(groups[block][grp]->param,ALL(),index,ALL());
  parallel_for("wkset[block] soln ip HGRAD",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type dim=0; dim<cbasis.extent(3); ++dim) {
      ScalarT avgval = 0.0;
      for (size_type dof=0; dof<cbasis.extent(1); ++dof ) {
        for (size_type pt=0; pt<cbasis.extent(2); ++pt) {
          avgval += csol(elem,dof)*cbasis(elem,dof,pt,dim)*cwts(elem,pt);
        }
      }
      sol(elem,dim) = avgval/avgwts(elem);
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////


template<class Node>
void AssemblyManager<Node>::updateWorksetFace(const int & block, const size_t & grp,
                                          const size_t & facenum) {
  this->updateWorksetFace<ScalarT>(block, grp, facenum);
}

template<class Node>
void AssemblyManager<Node>::updateWorksetFaceAD(const int & block, const size_t & grp,
                                          const size_t & facenum) {
  this->updateWorksetFace<AD>(block, grp, facenum);
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetFace(const int & block, const size_t & grp,
                                          const size_t & facenum) {

  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateWorksetFace(wkset[block], block, grp, facenum);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateWorksetFace(wkset_AD[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateWorksetFace(wkset_AD2[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateWorksetFace(wkset_AD4[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateWorksetFace(wkset_AD8[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateWorksetFace(wkset_AD16[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateWorksetFace(wkset_AD18[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateWorksetFace(wkset_AD24[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateWorksetFace(wkset_AD32[block], block, grp, facenum);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetFace(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp, const size_t & facenum) {
  
  // IMPORANT NOTE: This function assumes that face contributions are computing IMMEDIATELY after the
  // volumetric contributions, which implies that the seeded solution in the workset is already
  // correct for this Group.  There is currently no use case where this assumption is false.
  
  //Teuchos::TimeMonitor localtimer(*computeSolnFaceTimer);
  
  wset->wts_side = groups[block][grp]->wts_face[facenum];
  wset->h = groups[block][grp]->hsize;

  wset->setScalarField(groups[block][grp]->ip_face[facenum][0],"x");
  wset->setScalarField(groups[block][grp]->normals_face[facenum][0],"n[x]");
  if (groups[block][grp]->ip_face[facenum].size() > 1) {
    wset->setScalarField(groups[block][grp]->ip_face[facenum][1],"y");
    wset->setScalarField(groups[block][grp]->normals_face[facenum][1],"n[y]");
  }
  if (groups[block][grp]->ip_face[facenum].size() > 2) {
    wset->setScalarField(groups[block][grp]->ip_face[facenum][2],"z");
    wset->setScalarField(groups[block][grp]->normals_face[facenum][2],"n[z]");
  }
    
  // Update the face integration points and basis in workset
  if (groups[block][grp]->storeAll || groupData[block]->use_basis_database) {
    wset->basis_side = groups[block][grp]->basis_face[facenum];
    wset->basis_grad_side = groups[block][grp]->basis_grad_face[facenum];
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad;
  
    disc->getPhysicalFaceBasis(groupData[block], facenum, groups[block][grp]->nodes, groups[block][grp]->orientation,
                               tbasis, tbasis_grad);
    vector<CompressedView<View_Sc4>> tcbasis, tcbasis_grad;
    for (size_t i=0; i<tbasis.size(); ++i) {
      tcbasis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      tcbasis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
    }
    wset->basis_side = tcbasis;
    wset->basis_grad_side = tcbasis_grad;
  }
  
  wset->resetSolutionFields();
  
}


///////////////////////////////////////////////////////////////////////////////////////
// Compute the contribution from this Group to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeJacRes(const int & block, const size_t & grp, 
                         const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                         const bool & compute_jacobian, const bool & compute_sens,
                         const int & num_active_params, const bool & compute_disc_sens,
                         const bool & compute_aux_sens, const bool & store_adjPrev,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_res,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                         const bool & assemble_volume_terms,
                         const bool & assemble_face_terms) {
  
#ifndef MrHyDE_NO_AD

  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  bool fixJacDiag = false;
  
  //wkset[block]->resetResidual();
  
  //////////////////////////////////////////////////////////////
  // Compute the AD-seeded solutions at integration points
  //////////////////////////////////////////////////////////////
  
  int seedwhat = 0;
  if (compute_jacobian) {
    if (compute_disc_sens) {
      seedwhat = 3;
    }
    else if (compute_aux_sens) {
      seedwhat = 4;
    }
    else {
      seedwhat = 1;
    }
  }
  int seedindex = 0;

  //////////////////////////////////////////////////////////////
  // Compute res and J=dF/du
  //////////////////////////////////////////////////////////////
  
  // Volumetric contribution
  if (assemble_volume_terms) {
    //Teuchos::TimeMonitor localtimer(*volumeResidualTimer);
    if (groupData[block]->multiscale) {
      this->updateWorkset<AD>(block, grp, seedwhat, seedindex);
      groups[block][grp]->subgridModels[groups[block][grp]->subgrid_model_index]->subgridSolver(groups[block][grp]->sol[0], groups[block][grp]->sol_prev[0], 
                                            groups[block][grp]->phi[0], wkset_AD[block]->time, isTransient, isAdjoint,
                                            compute_jacobian, compute_sens, num_active_params,
                                            compute_disc_sens, compute_aux_sens,
                                            *wkset_AD[block], groups[block][grp]->subgrid_usernum, 0,
                                            groups[block][grp]->subgradient, store_adjPrev);
      fixJacDiag = true;
    }
    else {
      this->updateWorkset<AD>(block, grp, seedwhat, seedindex);
      physics->volumeResidual<AD>(wkset_AD[block]->current_set,groupData[block]->my_block);
    }
  }
  
  // Edge/face contribution
  if (assemble_face_terms) {
    //Teuchos::TimeMonitor localtimer(*faceResidualTimer);
    if (groupData[block]->multiscale) {
      // do nothing
    }
    else {
      for (size_t s=0; s<groupData[block]->num_sides; s++) {
        this->updateWorksetFace<AD>(block, grp, s);
        physics->faceResidual<AD>(wkset_AD[block]->current_set,groupData[block]->my_block);
      }
    }
  }
  
  {
    //Teuchos::TimeMonitor localtimer(*jacobianFillTimer);
    
    // Use AD residual to update local Jacobian
    if (compute_jacobian) {
      if (compute_disc_sens) {
        this->updateParamJac(block, grp, local_J);
      }
      else if (compute_aux_sens){
        this->updateAuxJac(block, grp, local_J);
      }
      else {
        this->updateJac(block, grp, isAdjoint, local_J);
      }
    }
  }
  
  if (compute_jacobian && fixJacDiag) {
    this->fixDiagJac(block, grp, local_J, local_res);
  }
  
  // Update the local residual
  {
    if (isAdjoint) {
      //Teuchos::TimeMonitor localtimer(*adjointResidualTimer);
      this->updateAdjointRes(block, grp, compute_jacobian, isTransient,
                             compute_aux_sens, store_adjPrev,
                             local_J, local_res);
    }
    else {
      //Teuchos::TimeMonitor localtimer(*residualFillTimer);
      this->updateRes(block, grp, compute_sens, local_res);
    }
  }
    
#endif

}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateRes(const int & block, const size_t & grp,
                                      const bool & compute_sens, View_Sc3 local_res) {

#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD[block]);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD2[block]);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD4[block]);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD8[block]);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD16[block]);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD18[block]);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD24[block]);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateRes(block, grp, compute_sens, local_res, wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateRes(const int & block, const size_t & grp,
                                      const bool & compute_sens, View_Sc3 local_res,
                                      Teuchos::RCP<Workset<EvalT> > & wset) {

#ifndef MrHyDE_NO_AD  
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  if (compute_sens) {

    parallel_for("Group res sens",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type r=0; r<local_res.extent(2); r++) {
            local_res(elem,offsets(n,j),r) -= res_AD(elem,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("Group res",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(elem,offsets(n,j),0) -= res_AD(elem,offsets(n,j)).val();
        }
      }
    });
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateAdjointRes(const int & block, const size_t & grp,
                            const bool & compute_jacobian, const bool & isTransient,
                            const bool & compute_aux_sens, const bool & store_adjPrev,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_res) {

#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateAdjointRes(block, grp, compute_jacobian, isTransient, compute_aux_sens,
                           store_adjPrev, local_J, local_res, wkset_AD[block]);
  }
  if (wkset_AD2[block]->isInitialized) {
    this->updateAdjointRes(block, grp, compute_jacobian, isTransient, compute_aux_sens,
                           store_adjPrev, local_J, local_res, wkset_AD2[block]);
  }
  if (wkset_AD4[block]->isInitialized) {
    this->updateAdjointRes(block, grp, compute_jacobian, isTransient, compute_aux_sens,
                           store_adjPrev, local_J, local_res, wkset_AD4[block]);
  }
  if (wkset_AD8[block]->isInitialized) {
    this->updateAdjointRes(block, grp, compute_jacobian, isTransient, compute_aux_sens,
                           store_adjPrev, local_J, local_res, wkset_AD8[block]);
  }
  if (wkset_AD16[block]->isInitialized) {
    this->updateAdjointRes(block, grp, compute_jacobian, isTransient, compute_aux_sens,
                           store_adjPrev, local_J, local_res, wkset_AD16[block]);
  }
  if (wkset_AD18[block]->isInitialized) {
    this->updateAdjointRes(block, grp, compute_jacobian, isTransient, compute_aux_sens,
                           store_adjPrev, local_J, local_res, wkset_AD18[block]);
  }
  if (wkset_AD24[block]->isInitialized) {
    this->updateAdjointRes(block, grp, compute_jacobian, isTransient, compute_aux_sens,
                           store_adjPrev, local_J, local_res, wkset_AD24[block]);
  }
  if (wkset_AD32[block]->isInitialized) {
    this->updateAdjointRes(block, grp, compute_jacobian, isTransient, compute_aux_sens,
                           store_adjPrev, local_J, local_res, wkset_AD32[block]);
  }
#endif
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateAdjointRes(const int & block, const size_t & grp,
                            const bool & compute_jacobian, const bool & isTransient,
                            const bool & compute_aux_sens, const bool & store_adjPrev,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_res,
                            Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD

  // Update residual (adjoint mode)
  // Adjoint residual: -dobj/du - J^T * phi + 1/dt*M^T * phi_prev
  // J = 1/dtM + A
  // adj_prev stores 1/dt*M^T * phi_prev where M is evaluated at appropriate time
  
  // TMW: This will not work on a GPU
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  size_t set = wset->current_set;
  auto cphi = groups[block][grp]->phi[set];
  
  if (compute_jacobian) {
    parallel_for("Group adjust adjoint jac",
                 RangePolicy<AssemblyExec>(0,local_res.extent(0)),
                 KOKKOS_LAMBDA (const size_type e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_res(e,offsets(n,j),0) += -local_J(e,offsets(n,j),offsets(m,k))*cphi(e,m,k);
            }
          }
        }
      }
    });
    
    if (isTransient) {
      
      auto aprev = groups[block][grp]->adj_prev[set];
      
      // Previous step contributions for the residual
      parallel_for("Group adjust transient adjoint jac",
                   RangePolicy<AssemblyExec>(0,local_res.extent(0)),
                   KOKKOS_LAMBDA (const size_type e ) {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),0) += -aprev(e,offsets(n,j),0);
          }
        }
      });
      /*
      // Previous stage contributions for the residual
      if (adj_stage_prev.extent(2) > 0) {
        parallel_for("Group adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
          for (size_type n=0; n<numDOF.extent(0); n++) {
            for (int j=0; j<numDOF(n); j++) {
              local_res(e,offsets(n,j),0) += -adj_stage_prev(e,offsets(n,j),0);
            }
          }
        });
      }
      */
      
      if (!compute_aux_sens && store_adjPrev) {
        
        //////////////////////////////////////////
        // Multi-step
        //////////////////////////////////////////
        
        // Move vectors up
        parallel_for("Group adjust transient adjoint jac",
                     RangePolicy<AssemblyExec>(0,aprev.extent(0)),
                     KOKKOS_LAMBDA (const size_type e ) {
          for (size_type step=1; step<aprev.extent(2); step++) {
            for (size_type n=0; n<aprev.extent(1); n++) {
              aprev(e,n,step-1) = aprev(e,n,step);
            }
          }
          size_type numsteps = aprev.extent(2);
          for (size_type n=0; n<aprev.extent(1); n++) {
            aprev(e,n,numsteps-1) = 0.0;
          }
        });
        
        // Sum new contributions into vectors
        int seedwhat = 2; // 2 for J wrt previous step solutions
        for (size_type step=0; step<groups[block][grp]->sol_prev[set].extent(3); step++) {
          this->updateWorkset<EvalT>(block, grp, seedwhat,step);
          physics->volumeResidual<EvalT>(set, groupData[block]->my_block);
          Kokkos::View<ScalarT***,AssemblyDevice> Jdot("temporary fix for transient adjoint",
                                                       local_J.extent(0), local_J.extent(1), local_J.extent(2));
          this->updateJac(block, grp, true, Jdot);
          
          auto cadj = subview(aprev,ALL(), ALL(), step);
          parallel_for("Group adjust transient adjoint jac 2",
                       RangePolicy<AssemblyExec>(0,Jdot.extent(0)),
                       KOKKOS_LAMBDA (const size_type e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for (int j=0; j<numDOF(n); j++) {
                ScalarT aPrev = 0.0;
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    aPrev += Jdot(e,offsets(n,j),offsets(m,k))*cphi(e,m,k);
                  }
                }
                cadj(e,offsets(n,j)) += aPrev;
              }
            }
          });
        }
        
        //////////////////////////////////////////
        // Multi-stage
        //////////////////////////////////////////
        /*
        // Move vectors up
        parallel_for("Group adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,adj_stage_prev.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
          for (size_type stage=1; stage<adj_stage_prev.extent(2); stage++) {
            for (size_type n=0; n<adj_stage_prev.extent(1); n++) {
              adj_stage_prev(e,n,stage-1) = adj_stage_prev(e,n,stage);
            }
          }
          size_type numstages = adj_stage_prev.extent(2);
          for (size_type n=0; n<adj_stage_prev.extent(1); n++) {
            adj_stage_prev(e,n,numstages-1) = 0.0;
          }
        });
        
        // Sum new contributions into vectors
        seedwhat = 3; // 3 for J wrt previous stage solutions
        for (size_type stage=0; stage<sol_prev.extent(3); stage++) {
          wkset[block]->computeSolnTransientSeeded(u, sol_prev, sol_stage, seedwhat, stage);
          wkset[block]->computeParamVolIP(param, seedwhat);
          this->computeSolnVolIP();
          
          wkset[block]->resetResidual();
          
          group_data->physics->volumeResidual(group_data->myBlock);
          Kokkos::View<ScalarT***,AssemblyDevice> Jdot("temporary fix for transient adjoint",
                                                       local_J.extent(0), local_J.extent(1), local_J.extent(2));
          this->updateJac(true, Jdot);
          
          auto cadj = subview(adj_stage_prev,ALL(), ALL(), stage);
          parallel_for("Group adjust transient adjoint jac 2",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for (int j=0; j<numDOF(n); j++) {
                ScalarT aPrev = 0.0;
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    aPrev += Jdot(e,offsets(n,j),offsets(m,k))*phi(e,m,k);
                  }
                }
                cadj(e,offsets(n,j)) += aPrev;
              }
            }
          });
        }*/
      }
    }
  }
  #endif
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateJac(const int & block, const size_t & grp,
                                      const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J) {

#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD[block]);
  }
  if (wkset_AD2[block]->isInitialized) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD2[block]);
  }
  if (wkset_AD4[block]->isInitialized) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD4[block]);
  }
  if (wkset_AD8[block]->isInitialized) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD8[block]);
  }
  if (wkset_AD16[block]->isInitialized) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD16[block]);
  }
  if (wkset_AD18[block]->isInitialized) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD18[block]);
  }
  if (wkset_AD24[block]->isInitialized) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD24[block]);
  }
  if (wkset_AD32[block]->isInitialized) {
    this->updateJac(block, grp, useadjoint, local_J, wkset_AD32[block]);
  }
  
#endif
}


template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateJac(const int & block, const size_t & grp,
                                      const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                                      Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  if (useadjoint) {
    parallel_for("Group J adj",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(m,k),offsets(n,j)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("Group J",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(n,j),offsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  //AssemblyExec::execution_space().fence();
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Place ones on the diagonal of the Jacobian if
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::fixDiagJac(const int & block, const size_t & grp, 
                      Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                      Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;

  using namespace std;

  parallel_for("Group fix diag",
               RangePolicy<AssemblyExec>(0,local_J.extent(0)), 
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT JTOL = 1.0E-14;
    for (size_type var=0; var<offsets.extent(0); var++) {
      for (int dof=0; dof<numDOF(var); dof++) {
        int diag = offsets(var,dof);
        if (abs(local_J(elem,diag,diag)) < JTOL) {
          local_res(elem,diag,0) = 0.0;//-u(elem,var,dof);
          for (int j=0; j<numDOF(var); j++) {
            ScalarT scale = 1.0/((ScalarT)numDOF(var)-1.0);
            local_J(elem,diag,offsets(var,j)) = -scale;
            //if (j!=dof)
            //  local_res(elem,diag,0) += scale*u(elem,var,j);
          }
          local_J(elem,diag,diag) = 1.0;
        }
      }
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////


template<class Node>
void AssemblyManager<Node>::updateParamJac(const int & block, const size_t & grp,
                                           Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateParamJac(block, grp, local_J, wkset_AD[block]);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateParamJac(block, grp, local_J, wkset_AD2[block]);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateParamJac(block, grp, local_J, wkset_AD4[block]);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateParamJac(block, grp, local_J, wkset_AD8[block]);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateParamJac(block, grp, local_J, wkset_AD16[block]);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateParamJac(block, grp, local_J, wkset_AD18[block]);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateParamJac(block, grp, local_J, wkset_AD24[block]);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateParamJac(block, grp, local_J, wkset_AD32[block]);
  }
  #endif
}


template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateParamJac(const int & block, const size_t & grp,
                                           Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                                           Teuchos::RCP<Workset<EvalT> > & wset) {
#ifndef MrHyDE_NO_AD
  auto paramoffsets = wset->paramoffsets;
  auto numParamDOF = groupData[block]->num_param_dof;
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto numDOF = groupData[block]->num_dof;
  
  parallel_for("Group param J",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (size_type m=0; m<numParamDOF.extent(0); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_J(elem,offsets(n,j),paramoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateAuxJac(const int & block, const size_t & grp,
                                           Kokkos::View<ScalarT***,AssemblyDevice> local_J) {

#ifndef MrHyDE_NO_AD
  if (wkset_AD[block]->isInitialized) {
    this->updateAuxJac(block, grp, local_J, wkset_AD[block]);
  }
  else if (wkset_AD2[block]->isInitialized) {
    this->updateAuxJac(block, grp, local_J, wkset_AD2[block]);
  }
  else if (wkset_AD4[block]->isInitialized) {
    this->updateAuxJac(block, grp, local_J, wkset_AD4[block]);
  }
  else if (wkset_AD8[block]->isInitialized) {
    this->updateAuxJac(block, grp, local_J, wkset_AD8[block]);
  }
  else if (wkset_AD16[block]->isInitialized) {
    this->updateAuxJac(block, grp, local_J, wkset_AD16[block]);
  }
  else if (wkset_AD18[block]->isInitialized) {
    this->updateAuxJac(block, grp, local_J, wkset_AD18[block]);
  }
  else if (wkset_AD24[block]->isInitialized) {
    this->updateAuxJac(block, grp, local_J, wkset_AD24[block]);
  }
  else if (wkset_AD32[block]->isInitialized) {
    this->updateAuxJac(block, grp, local_J, wkset_AD32[block]);
  }
#endif

}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateAuxJac(const int & block, const size_t & grp,
                                           Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                                           Teuchos::RCP<Workset<EvalT> > & wset) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wset->res;
  auto offsets = wset->offsets;
  auto aoffsets = groups[block][grp]->auxoffsets;
  auto numDOF = groupData[block]->num_dof;
  auto numAuxDOF = groupData[block]->num_aux_dof;
  
  parallel_for("Group aux J",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (size_type m=0; m<numAuxDOF.extent(0); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(elem,offsets(n,j),aoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(aoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
View_Sc2 AssemblyManager<Node>::getInitial(const int & block, const size_t & grp,
                                           const bool & project, const bool & isAdjoint) {
  
  size_t set = wkset[block]->current_set;
  View_Sc2 initialvals("initial values",groups[block][grp]->numElem, groups[block][grp]->LIDs[set].extent(1));
  this->updateWorkset<ScalarT>(block, grp, 0,0);
  
  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;
  auto cwts = groups[block][grp]->wts;
  
  if (project) { // works for any basis
    auto initialip = groupData[block]->physics->getInitial(groups[block][grp]->ip, set,
                                                        groupData[block]->my_block,
                                                        project, wkset[block]);

    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = groups[block][grp]->basis[wkset[block]->usebasis[n]];
      auto off = subview(offsets, n, ALL());
      string btype = wkset[block]->basis_types[wkset[block]->usebasis[n]];
      if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
        auto initvar = subview(initialip, ALL(), n, ALL(), 0);
        parallel_for("Group init project",
                     TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type dof=team.team_rank(); dof<cbasis.extent(1); dof+=team.team_size() ) {
            for(size_type pt=0; pt<cwts.extent(1); pt++ ) {
              initialvals(elem,off(dof)) += initvar(elem,pt)*cbasis(elem,dof,pt,0)*cwts(elem,pt);
            }
          }
        });
      }
      else if (btype.substr(0,5) == "HCURL" || btype.substr(0,4) == "HDIV") {
        auto initvar = subview(initialip, ALL(), n, ALL(), ALL());
        parallel_for("Group init project",
                     TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type dof=team.team_rank(); dof<cbasis.extent(1); dof+=team.team_size() ) {
            for (size_type pt=0; pt<cwts.extent(1); pt++ ) {
              for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                initialvals(elem,off(dof)) += initvar(elem,pt,dim)*cbasis(elem,dof,pt,dim)*cwts(elem,pt);
              }
            }
          }
        });
      }
      
    }
  }
  else { // only works if using HGRAD linear basis
    vector<View_Sc2> vnodes;
    View_Sc2 vx,vy,vz;
    vx = View_Sc2("view of nodes", groups[block][grp]->nodes.extent(0), groups[block][grp]->nodes.extent(1));
    auto n_x = subview(groups[block][grp]->nodes,ALL(),ALL(),0);
    deep_copy(vx,n_x);
    vnodes.push_back(vx);
    if (groups[block][grp]->nodes.extent(2) > 1) {
      vy = View_Sc2("view of nodes", groups[block][grp]->nodes.extent(0), groups[block][grp]->nodes.extent(1));
      auto n_y = subview(groups[block][grp]->nodes,ALL(),ALL(),1);
      deep_copy(vy,n_y);
      vnodes.push_back(vy);
    }
    if (groups[block][grp]->nodes.extent(2) > 2) {
      vz = View_Sc2("view of nodes", groups[block][grp]->nodes.extent(0), groups[block][grp]->nodes.extent(1));
      auto n_z = subview(groups[block][grp]->nodes, ALL(), ALL(), 2);
      deep_copy(vz,n_z);
      vnodes.push_back(vz);
    }
    
    auto initialnodes = groupData[block]->physics->getInitial(vnodes, set,
                                                          groupData[block]->my_block,
                                                          project,
                                                          wkset[block]);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto off = subview( offsets, n, ALL());
      auto initvar = subview(initialnodes, ALL(), n, ALL(), 0);
      parallel_for("Group init project",
                   TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<initvar.extent(1); dof+=team.team_size() ) {
          initialvals(elem,off(dof)) = initvar(elem,dof);
        }
      });
    }
  }
  return initialvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition on the faces
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
View_Sc2 AssemblyManager<Node>::getInitialFace(const int & block, const size_t & grp, const bool & project) {
  
  size_t set = wkset[block]->current_set;
  View_Sc2 initialvals("initial values",groups[block][grp]->numElem, groups[block][grp]->LIDs[set].extent(1)); // TODO is this too big?
  this->updateWorkset<ScalarT>(block, grp, 0, 0); // TODO not sure if this is necessary

  auto offsets = wkset[block]->offsets;
  auto numDOF = groupData[block]->num_dof;

  // loop over faces of the reference element
  for (size_t face=0; face<groupData[block]->num_sides; face++) {

    // get basis functions, weights, etc. for that face
    this->updateWorksetFace<ScalarT>(block, grp, face);
    auto cwts = wkset[block]->wts_side; // face weights get put into wts_side after update
    // get data from IC
    auto initialip = groupData[block]->physics->getInitialFace(groups[block][grp]->ip_face[face], set,
                                                           groupData[block]->my_block,
                                                           project,
                                                           wkset[block]);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = wkset[block]->basis_side[wkset[block]->usebasis[n]]; // face basis gets put here after update
      auto off = subview(offsets, n, ALL());
      auto initvar = subview(initialip, ALL(), n, ALL());
      // loop over mesh elements
      parallel_for("Group init project",
                   TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<cbasis.extent(1); dof+=team.team_size() ) {
          for(size_type pt=0; pt<cwts.extent(1); pt++ ) {
            initialvals(elem,off(dof)) += initvar(elem,pt)*cbasis(elem,dof,pt,0)*cwts(elem,pt);
          }
        }
      });
    }
  }
  
  return initialvals;
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
    disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->nodes, groups[block][grp]->orientation,
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
                   KOKKOS_LAMBDA (const size_type e ) {
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
                   KOKKOS_LAMBDA (const size_type e ) {
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
      disc->getPhysicalVolumetricBasis(groupData[block], groups[block][grp]->nodes, groups[block][grp]->orientation,
                                       tmpbasis);
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
                     KOKKOS_LAMBDA (const size_type e ) {
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


///////////////////////////////////////////////////////////////////////////////////////
// Get the solution at the nodes
///////////////////////////////////////////////////////////////////////////////////////

template<class Node>
Kokkos::View<ScalarT***,AssemblyDevice> AssemblyManager<Node>::getSolutionAtNodes(const int & block, const size_t & grp, const int & var) {
  
  size_t set = wkset[block]->current_set;
  
  int bnum = wkset[block]->usebasis[var];
  auto cbasis = groups[block][grp]->basis_nodes[bnum];
  Kokkos::View<ScalarT***,AssemblyDevice> nodesol("solution at nodes",
                                                  cbasis.extent(0), cbasis.extent(2), groupData[block]->dimension);
  auto uvals = subview(groups[block][grp]->sol[set], ALL(), var, ALL());
  parallel_for("Group node sol",
               RangePolicy<AssemblyExec>(0,cbasis.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
      ScalarT uval = uvals(elem,dof);
      for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
        for (size_type s=0; s<cbasis.extent(3); s++ ) {
          nodesol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
        }
      }
    }
  });
  
  return nodesol;
  
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
                 KOKKOS_LAMBDA (const size_type e ) {
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

template<class Node>
vector<vector<int> > AssemblyManager<Node>::identifySubgridModels() {

  vector<vector<int> > sgmodels;

#ifndef MrHyDE_NO_AD

  for (size_t block=0; block<groups.size(); ++block) {
    
    vector<int> block_sgmodels;
    bool uses_subgrid = false;
    for (size_t s=0; s<multiscale_manager->subgridModels.size(); s++) {
      if (multiscale_manager->subgridModels[s]->macro_block == block) {
        uses_subgrid = true;
      }
    }
    
    if (uses_subgrid) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        
        this->updateWorkset<AD>(block, grp, 0, 0);
        
        vector<int> sgvotes(multiscale_manager->subgridModels.size(),0);
        
        for (size_t s=0; s<multiscale_manager->subgridModels.size(); s++) {
          if (multiscale_manager->subgridModels[s]->macro_block == block) {
            std::stringstream ss;
            ss << s;
            auto usagecheck = function_managers_AD[block]->evaluate(multiscale_manager->subgridModels[s]->name + " usage","ip");
            
            Kokkos::View<ScalarT**,AssemblyDevice> usagecheck_tmp("temp usage check",
                                                                  function_managers[block]->num_elem_,
                                                                  function_managers[block]->num_ip_);
                                                                  
            parallel_for("assembly copy LIDs",
                         RangePolicy<AssemblyExec>(0,usagecheck_tmp.extent(0)),
                         KOKKOS_LAMBDA (const int i ) {
              for (size_type j=0; j<usagecheck_tmp.extent(1); j++) {
                usagecheck_tmp(i,j) = usagecheck(i,j).val();
              }
            });
            
            auto host_usagecheck = Kokkos::create_mirror_view(usagecheck_tmp);
            Kokkos::deep_copy(host_usagecheck, usagecheck_tmp);
            for (size_t p=0; p<groups[block][grp]->numElem; p++) {
              for (size_t j=0; j<host_usagecheck.extent(1); j++) {
                if (host_usagecheck(p,j) >= 1.0) {
                  sgvotes[s] += 1;
                }
              }
            }
          }
        }
        
        int maxvotes = -1;
        int sgwinner = 0;
        for (size_t i=0; i<sgvotes.size(); i++) {
          if (sgvotes[i] >= maxvotes) {
            maxvotes = sgvotes[i];
            sgwinner = i;
          }
        }
        block_sgmodels.push_back(sgwinner);
      }
    }
    sgmodels.push_back(block_sgmodels); 
  }
#endif
  return sgmodels;
}

////////////////////////////////////////////////////////////////////////////////
// Create the function managers
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createFunctions() {    
    
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    function_managers.push_back(Teuchos::rcp(new FunctionManager<ScalarT>(blocknames[block],
                                                                     groupData[block]->num_elem,
                                                                     disc->numip[block],
                                                                     disc->numip_side[block])));
  }
  physics->defineFunctions(function_managers);

#ifndef MrHyDE_NO_AD
  for (size_t block=0; block<blocknames.size(); ++block) {
    function_managers_AD.push_back(Teuchos::rcp(new FunctionManager<AD>(blocknames[block],
                                                                     groupData[block]->num_elem,
                                                                     disc->numip[block],
                                                                     disc->numip_side[block])));
  }
  physics->defineFunctions(function_managers_AD);

    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD2.push_back(Teuchos::rcp(new FunctionManager<AD2>(blocknames[block],
                                                                       groupData[block]->num_elem,
                                                                       disc->numip[block],
                                                                       disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD2);
  
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD4.push_back(Teuchos::rcp(new FunctionManager<AD4>(blocknames[block],
                                                                       groupData[block]->num_elem,
                                                                       disc->numip[block],
                                                                       disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD4);
  
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD8.push_back(Teuchos::rcp(new FunctionManager<AD8>(blocknames[block],
                                                                       groupData[block]->num_elem,
                                                                       disc->numip[block],
                                                                       disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD8);
  
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD16.push_back(Teuchos::rcp(new FunctionManager<AD16>(blocknames[block],
                                                                       groupData[block]->num_elem,
                                                                       disc->numip[block],
                                                                       disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD16);
  
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD18.push_back(Teuchos::rcp(new FunctionManager<AD18>(blocknames[block],
                                                                       groupData[block]->num_elem,
                                                                       disc->numip[block],
                                                                       disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD18);
  
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD24.push_back(Teuchos::rcp(new FunctionManager<AD24>(blocknames[block],
                                                                       groupData[block]->num_elem,
                                                                       disc->numip[block],
                                                                       disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD24);
  
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD32.push_back(Teuchos::rcp(new FunctionManager<AD32>(blocknames[block],
                                                                       groupData[block]->num_elem,
                                                                       disc->numip[block],
                                                                       disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD32);
#endif
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::setMeshData() {
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting assembly manager setMeshData" << endl;
    }
  }
  
  if (mesh->have_mesh_data) {
    this->importMeshData();
  }
  else if (mesh->compute_mesh_data) {
    int randSeed = settings->sublist("Mesh").get<int>("random seed", 1234);
    auto seeds = mesh->generateNewMicrostructure(randSeed);
    this->importNewMicrostructure(randSeed, seeds);
  }
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished mesh interface setMeshData" << endl;
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::importMeshData() {
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::importMeshData ..." << endl;
    }
  }
  
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  int numdata = 1;
  if (mesh->have_rotations) {
    numdata = 9;
  }
  else if (mesh->have_rotation_phi) {
    numdata = 3;
  }
  
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      groups[block][grp]->data = cell_data;
      groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      groups[block][grp]->data_seed = vector<size_t>(numElem);
      groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      int numElem = boundary_groups[block][grp]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      boundary_groups[block][grp]->data = cell_data;
      boundary_groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      boundary_groups[block][grp]->data_seed = vector<size_t>(numElem);
      boundary_groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  
  Teuchos::RCP<Data> mesh_data;
  
  string mesh_data_pts_file = mesh->mesh_data_pts_tag + ".dat";
  string mesh_data_file = mesh->mesh_data_tag + ".dat";
  
  bool have_grid_data = settings->sublist("Mesh").get<bool>("data on grid",false);
  if (have_grid_data) {
    int Nx = settings->sublist("Mesh").get<int>("data grid Nx",0);
    int Ny = settings->sublist("Mesh").get<int>("data grid Ny",0);
    int Nz = settings->sublist("Mesh").get<int>("data grid Nz",0);
    mesh_data = Teuchos::rcp(new Data("mesh data", mesh->dimension, mesh_data_pts_file,
                                      mesh_data_file, false, Nx, Ny, Nz));
    
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        DRV nodes = groups[block][grp]->nodes;
        int numElem = groups[block][grp]->numElem;
        
        auto centers = mesh->getElementCenters(nodes, groups[block][grp]->group_data->cell_topo);
        auto centers_host = create_mirror_view(centers);
        deep_copy(centers_host,centers);
        
        for (int c=0; c<numElem; c++) {
          ScalarT distance = 0.0;
          
          // Doesn't use the Compadre interface
          int cnode = mesh_data->findClosestGridPoint(centers_host(c,0), centers_host(c,1),
                                                      centers_host(c,2), distance);
          
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode);
          for (size_type i=0; i<cdata.extent(1); i++) {
            groups[block][grp]->data(c,i) = cdata(0,i);
          }
          groups[block][grp]->group_data->have_extra_data = true;
          groups[block][grp]->group_data->have_rotation = mesh->have_rotations;
          groups[block][grp]->group_data->have_phi = mesh->have_rotation_phi;
          
          groups[block][grp]->data_seed[c] = cnode;
          groups[block][grp]->data_seedindex[c] = cnode % 100;
          groups[block][grp]->data_distance[c] = distance;
        }
      }
    }
    
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = boundary_groups[block][grp]->nodes;
        int numElem = boundary_groups[block][grp]->numElem;
        
        auto centers = mesh->getElementCenters(nodes, boundary_groups[block][grp]->group_data->cell_topo);
        auto centers_host = create_mirror_view(centers);
        deep_copy(centers_host,centers);
        
        for (int c=0; c<numElem; c++) {
          ScalarT distance = 0.0;
          
          int cnode = mesh_data->findClosestGridPoint(centers_host(c,0), centers_host(c,1),
                                                      centers_host(c,2), distance);
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode);
          for (size_type i=0; i<cdata.extent(1); i++) {
            boundary_groups[block][grp]->data(c,i) = cdata(0,i);
          }
          boundary_groups[block][grp]->group_data->have_extra_data = true;
          boundary_groups[block][grp]->group_data->have_rotation = mesh->have_rotations;
          boundary_groups[block][grp]->group_data->have_phi = mesh->have_rotation_phi;
          
          boundary_groups[block][grp]->data_seed[c] = cnode;
          boundary_groups[block][grp]->data_seedindex[c] = cnode % 100;
          boundary_groups[block][grp]->data_distance[c] = distance;
        }
      }
    }
  }
  else {
    mesh_data = Teuchos::rcp(new Data("mesh data", mesh->dimension, mesh_data_pts_file,
                                      mesh_data_file, false));
    
    for (size_t block=0; block<groups.size(); ++block) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        DRV nodes = groups[block][grp]->nodes;
        int numElem = groups[block][grp]->numElem;
        
        auto centers = mesh->getElementCenters(nodes, groups[block][grp]->group_data->cell_topo);
        
        Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
        Kokkos::View<int*, CompadreDevice> cnode("cnode",numElem);
        
        mesh_data->findClosestPoint(centers,cnode,distance);
        
        auto distance_mirror = Kokkos::create_mirror_view(distance);
        auto data_mirror = Kokkos::create_mirror_view(groups[block][grp]->data);

        for (int c=0; c<numElem; c++) {
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode(c));

          for (size_t i=0; i<cdata.extent(1); i++) {
            data_mirror(c,i) = cdata(0,i);
          }

          groups[block][grp]->group_data->have_extra_data = true;
          groups[block][grp]->group_data->have_rotation = mesh->have_rotations;
          groups[block][grp]->group_data->have_phi = mesh->have_rotation_phi;
          
          groups[block][grp]->data_seed[c] = cnode(c);
          groups[block][grp]->data_seedindex[c] = cnode(c) % 100;
          groups[block][grp]->data_distance[c] = distance_mirror(c);

        }
        Kokkos::deep_copy(groups[block][grp]->data, data_mirror);
      }
    }
    
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = boundary_groups[block][grp]->nodes;
        int numElem = boundary_groups[block][grp]->numElem;
        
        auto centers = mesh->getElementCenters(nodes, boundary_groups[block][grp]->group_data->cell_topo);
        
        Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
        Kokkos::View<int*, CompadreDevice> cnode("cnode",numElem);
        
        mesh_data->findClosestPoint(centers,cnode,distance);

        auto distance_mirror = Kokkos::create_mirror_view(distance);
        auto data_mirror = Kokkos::create_mirror_view(boundary_groups[block][grp]->data);
        
        for (int c=0; c<numElem; c++) {
          Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getData(cnode(c));

          for (size_t i=0; i<cdata.extent(1); i++) {
            data_mirror(c,i) = cdata(0,i);
          }

          boundary_groups[block][grp]->group_data->have_extra_data = true;
          boundary_groups[block][grp]->group_data->have_rotation = mesh->have_rotations;
          boundary_groups[block][grp]->group_data->have_phi = mesh->have_rotation_phi;
          
          boundary_groups[block][grp]->data_seed[c] = cnode(c);
          boundary_groups[block][grp]->data_seedindex[c] = cnode(c) % 50;
          boundary_groups[block][grp]->data_distance[c] = distance_mirror(c);
        }
        Kokkos::deep_copy(boundary_groups[block][grp]->data, data_mirror);
      }
    }
  }
  
  
  meshimporttimer.stop();
  if (verbosity>5 && comm->getRank() == 0) {
    cout << "mesh data import time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::meshDataImport" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::importNewMicrostructure(int & randSeed, View_Sc2 seeds) {
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting AssemblyManager::importNewMicrostructure ..." << endl;
    }
  }
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  std::default_random_engine generator(randSeed);
  
  size_type num_seeds = seeds.extent(0);
  std::uniform_int_distribution<int> idistribution(0,100);
  Kokkos::View<int*,HostDevice> seedIndex("seed index",num_seeds);
  for (int i=0; i<num_seeds; i++) {
    int ci = idistribution(generator);
    seedIndex(i) = ci;
  }
  
  //KokkosTools::print(seedIndex);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////
  
  int numdata = 9;
  
  std::normal_distribution<ScalarT> ndistribution(0.0,1.0);
  Kokkos::View<ScalarT**,HostDevice> rotation_data("cell_data",num_seeds,numdata);
  for (int k=0; k<num_seeds; k++) {
    ScalarT x = ndistribution(generator);
    ScalarT y = ndistribution(generator);
    ScalarT z = ndistribution(generator);
    ScalarT w = ndistribution(generator);
    
    ScalarT r = sqrt(x*x + y*y + z*z + w*w);
    x *= 1.0/r;
    y *= 1.0/r;
    z *= 1.0/r;
    w *= 1.0/r;
    
    rotation_data(k,0) = w*w + x*x - y*y - z*z;
    rotation_data(k,1) = 2.0*(x*y - w*z);
    rotation_data(k,2) = 2.0*(x*z + w*y);
    
    rotation_data(k,3) = 2.0*(x*y + w*z);
    rotation_data(k,4) = w*w - x*x + y*y - z*z;
    rotation_data(k,5) = 2.0*(y*z - w*x);
    
    rotation_data(k,6) = 2.0*(x*z - w*y);
    rotation_data(k,7) = 2.0*(y*z + w*x);
    rotation_data(k,8) = w*w - x*x - y*y + z*z;
    
  }
  
  //KokkosTools::print(rotation_data);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Initialize cell data
  ////////////////////////////////////////////////////////////////////////////////
  
  int totalElem = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      totalElem += numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      groups[block][grp]->data = cell_data;
      groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      groups[block][grp]->data_seed = vector<size_t>(numElem);
      groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Create a list of all cell nodes
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV totalNodes("nodes from all groups",totalElem,
                 groups[0][0]->nodes.extent(1),
                 groups[0][0]->nodes.extent(2));
  int prog = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      auto nodes = groups[block][grp]->nodes;
      parallel_for("mesh data cell nodes",
                   RangePolicy<AssemblyExec>(0,nodes.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<nodes.extent(1); ++pt) {
          for (size_type dim=0; dim<nodes.extent(2); ++dim) {
            totalNodes(prog+elem,pt,dim) = nodes(elem,pt,dim);
          }
        }
      });
      prog += groups[block][grp]->numElem;
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Create a list of all cell centers
  ////////////////////////////////////////////////////////////////////////////////
  
  auto centers = mesh->getElementCenters(totalNodes, groups[0][0]->group_data->cell_topo);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Find the closest seeds
  ////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",totalElem);
  Kokkos::View<int*, CompadreDevice> cnode("cnode",totalElem);
  
  Compadre::NeighborLists<Kokkos::View<int*, CompadreDevice> > neighborlists = CompadreInterface_constructNeighborLists(seeds, centers, distance);
  cnode = neighborlists.getNeighborLists();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set group data
  ////////////////////////////////////////////////////////////////////////////////
  
  prog = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      //auto centers = this->getElementCenters(nodes, groups[block][grp]->group_data->cellTopo);
      
      //Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
      //Kokkos::View<int*, AssemblyDevice> cnode("cnode",numElem);
      //Compadre::NeighborLists<Kokkos::View<int*> > neighborlists = CompadreTools_constructNeighborLists(seeds, centers, distance);
      //cnode = neighborlists.getNeighborLists();

      for (int c=0; c<numElem; c++) {
        
        int cpt = cnode(prog);
        prog++;
        
        for (int i=0; i<9; i++) {
          groups[block][grp]->data(c,i) = rotation_data(cpt,i);//rotation_data(cnode(c),i);
        }
        
        groups[block][grp]->group_data->have_rotation = true;
        groups[block][grp]->group_data->have_phi = false;
        
        groups[block][grp]->data_seed[c] = cpt % 100;//cnode(c) % 100;
        groups[block][grp]->data_seedindex[c] = seedIndex(cpt); //seedIndex(cnode(c));
        groups[block][grp]->data_distance[c] = distance(cpt);//distance(c);
        
      }
    }
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Initialize boundary data
  ////////////////////////////////////////////////////////////////////////////////
  
  totalElem = 0;
  for (size_t block=0; block<boundary_groups.size(); ++block) {
    for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
      int numElem = boundary_groups[block][grp]->numElem;
      totalElem += numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      boundary_groups[block][grp]->data = cell_data;
      boundary_groups[block][grp]->data_distance = vector<ScalarT>(numElem);
      boundary_groups[block][grp]->data_seed = vector<size_t>(numElem);
      boundary_groups[block][grp]->data_seedindex = vector<size_t>(numElem);
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Create a list of all cell nodes
  ////////////////////////////////////////////////////////////////////////////////
  
  if (totalElem > 0) {
    
    totalNodes = DRV("nodes from all groups",totalElem,
                     groups[0][0]->nodes.extent(1),
                     groups[0][0]->nodes.extent(2));
    prog = 0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        auto nodes = boundary_groups[block][grp]->nodes;
        parallel_for("mesh data cell nodes",
                     RangePolicy<AssemblyExec>(0,nodes.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<nodes.extent(1); ++pt) {
            for (size_type dim=0; dim<nodes.extent(2); ++dim) {
              totalNodes(prog+elem,pt,dim) = nodes(elem,pt,dim);
            }
          }
        });
        prog += boundary_groups[block][grp]->numElem;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create a list of all cell centers
    ////////////////////////////////////////////////////////////////////////////////
    
    centers = mesh->getElementCenters(totalNodes, groups[0][0]->group_data->cell_topo);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Find the closest seeds
    ////////////////////////////////////////////////////////////////////////////////
    
    distance = Kokkos::View<ScalarT*, AssemblyDevice>("distance",totalElem);
    cnode = Kokkos::View<int*, CompadreDevice>("cnode",totalElem);
    neighborlists = CompadreInterface_constructNeighborLists(seeds, centers, distance);
    cnode = neighborlists.getNeighborLists();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set data
    ////////////////////////////////////////////////////////////////////////////////
    
    prog = 0;
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
        DRV nodes = boundary_groups[block][grp]->nodes;
        int numElem = boundary_groups[block][grp]->numElem;
        
        for (int c=0; c<numElem; c++) {
          
          int cpt = cnode(prog);
          prog++;
          
          for (int i=0; i<9; i++) {
            boundary_groups[block][grp]->data(c,i) = rotation_data(cpt,i);
          }
          
          boundary_groups[block][grp]->group_data->have_rotation = true;
          boundary_groups[block][grp]->group_data->have_phi = false;
          
          boundary_groups[block][grp]->data_seed[c] = cpt % 100;
          boundary_groups[block][grp]->data_seedindex[c] = seedIndex(cpt);
          boundary_groups[block][grp]->data_distance[c] = distance(cpt);
          
        }
      }
    }
    
  }
  
  meshimporttimer.stop();
  if (verbosity>5 && comm->getRank() == 0) {
    cout << "microstructure import time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished AssemblyManager::importNewMicrostructure" << endl;
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
