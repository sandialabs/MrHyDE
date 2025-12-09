/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

/////////////////////////////////////////////////////////////////////////////////////////////
// After the mesh and the discretizations have been defined, we can create and add the physics
// to the DOF manager
/////////////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::buildDOFManagers() {
  
  Teuchos::TimeMonitor localtimer(*dofmgr_timer);
  
  debugger->print("**** Starting discretization::buildDOF ...");
  
  Teuchos::RCP<panzer::ConnManager> conn = mesh->getSTKConnManager();
  
  num_derivs_required = vector<int>(block_names.size(),0);
  
  // DOF manager for the primary variables
  for (size_t set=0; set<physics->set_names.size(); ++set) {
    Teuchos::RCP<panzer::DOFManager> setDOF = Teuchos::rcp(new panzer::DOFManager());
    setDOF->setConnManager(conn,*(comm->getRawMpiComm()));
    setDOF->setOrientationsRequired(true);
    
    for (size_t block=0; block<block_names.size(); ++block) {
      for (size_t j=0; j<physics->var_list[set][block].size(); j++) {
        topo_RCP cellTopo = mesh->getCellTopology(block_names[block]);
        basis_RCP basis_pointer = this->getBasis(dimension, cellTopo,
                                                 physics->types[set][block][j],
                                                 physics->orders[set][block][j]);
        
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_pointer));
        
        if (physics->use_DG[set][block][j]) {
          setDOF->addField(block_names[block], physics->var_list[set][block][j], Pattern, panzer::FieldType::DG);
        }
        else {
          setDOF->addField(block_names[block], physics->var_list[set][block][j], Pattern, panzer::FieldType::CG);
        }
        
      }
    }
    
    setDOF->buildGlobalUnknowns();
#ifndef MrHyDE_NO_AD
    for (size_t block=0; block<block_names.size(); ++block) {
      int numGIDs = setDOF->getElementBlockGIDCount(block_names[block]);
      if (numGIDs > num_derivs_required[block]) {
        num_derivs_required[block] = numGIDs;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(numGIDs > MAXDERIVS,std::runtime_error,"Error: MAXDERIVS is not large enough to support the number of degrees of freedom per element on block: " + block_names[block]);
    }
#endif
    if (verbosity>1) {
      if (comm->getRank() == 0) {
        setDOF->printFieldInformation(cout);
      }
    }

    // Instead of storing the DOF manager, which holds onto the mesh, we extract what we need
    //DOF.push_back(setDOF);
    Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device> setLIDs = setDOF->getLIDs();
    
    dof_lids.push_back(setLIDs);
    
    {
      vector<GO> owned;
      setDOF->getOwnedIndices(owned);
      Kokkos::View<GO*,HostDevice> owned_kv("owned dofs",owned.size());
      for (size_type i=0; i<owned_kv.extent(0); ++i) {
        owned_kv(i) = owned[i];
      }
      dof_owned.push_back(owned_kv);
    }
    {
      vector<GO> ownedAndShared;
      setDOF->getOwnedAndGhostedIndices(ownedAndShared);
      Kokkos::View<GO*,HostDevice> ownedas_kv("owned dofs",ownedAndShared.size());
      for (size_type i=0; i<ownedas_kv.extent(0); ++i) {
        ownedas_kv(i) = ownedAndShared[i];
      }
      dof_owned_and_shared.push_back(ownedas_kv);
    }

    vector<vector<string> > varlist = physics->var_list[set];
    vector<vector<vector<int> > > set_offsets; // [block][var][dof]
    for (size_t block=0; block<block_names.size(); ++block) {
      vector<vector<int> > celloffsets;
      for (size_t j=0; j<varlist[block].size(); j++) {
        string var = varlist[block][j];
        int num = setDOF->getFieldNum(var);
        vector<int> var_offsets = setDOF->getGIDFieldOffsets(block_names[block],num);

        celloffsets.push_back(var_offsets);
      }
      set_offsets.push_back(celloffsets);
    }
    offsets.push_back(set_offsets);
    

    this->setBCData(set,setDOF);

    this->setDirichletData(set,setDOF);

  }

  // Create the vector of panzer orientations
  // Using the panzer orientation interface works, except when also
  // using an MPI subcommunicator, e.g., in the subgrid models
  // Leaving here for testing purposes

  //auto pOInt = panzer::OrientationsInterface(DOF[0]);
  //auto pO_orients = pOInt.getOrientations();
  //panzer_orientations = *pO_orients;
  
  {
    auto oconn = conn->noConnectivityClone();
    
    shards::CellTopology topology;
    std::vector<shards::CellTopology> elementBlockTopologies;
    oconn->getElementBlockTopologies(elementBlockTopologies);

    topology = elementBlockTopologies.at(0);
  
    const int num_nodes_per_cell = topology.getVertexCount();

    size_t totalElem = 0;
    for (size_t block=0; block<block_names.size(); ++block) {
      totalElem += my_elements[block].extent(0);
    }

    // Make sure the conn is setup for a nodal connectivity
    panzer::NodalFieldPattern pattern(topology);
    oconn->buildConnectivity(pattern);

    // Initialize the orientations vector
    //panzer_orientations.clear();
    panzer_orientations = Kokkos::View<Intrepid2::Orientation*,HostDevice>("panzer orients",totalElem);
  
    using NodeView = Kokkos::View<GO*, Kokkos::DefaultHostExecutionSpace>;
    
    // Add owned orientations
    {
      for (size_t block=0; block<block_names.size(); ++block) {
        for (size_t c=0; c<my_elements[block].extent(0); ++c) {
          size_t elemID = my_elements[block](c);
          const GO * nodes = oconn->getConnectivity(elemID);
          NodeView node_view("nodes",num_nodes_per_cell);
          for (int node=0; node<num_nodes_per_cell; ++node) {
            node_view(node) = nodes[node];
          }
          panzer_orientations(elemID) = Intrepid2::Orientation::getOrientation(topology, node_view);
          
        }
      }
    }
  }
  
  debugger->print("**** Finished discretization::buildDOF");
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::setBCData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF) {
  
  Teuchos::TimeMonitor localtimer(*set_bc_timer);
  
  debugger->print("**** Starting DiscretizationInterface::setBCData ...");
  
  bool requires_sideinfo = false;
  if (settings->isSublist("Subgrid")) {
    requires_sideinfo = true;
  }
  
  vector<string> sideSets, nodeSets;
  sideSets = mesh->getSideNames();
  nodeSets = mesh->getNodeNames();
  
  //for (size_t set=0; set<physics->setnames.size(); ++set) {
    vector<vector<string> > varlist = physics->var_list[set];
    //auto currDOF = DOF[set];
    
    vector<Kokkos::View<int****,HostDevice> > set_side_info;
    vector<vector<vector<string> > > set_var_bcs; // [block][var][boundary]
    
    vector<vector<GO> > set_point_dofs;
    vector<vector<vector<LO> > > set_dbc_dofs;
    
    for (size_t block=0; block<block_names.size(); ++block) {
      
      vector<vector<string> > block_var_bcs; // [var][boundary]
      
      topo_RCP cellTopo = mesh->getCellTopology(block_names[block]);
      int numSidesPerElem = 2; // default to 1D for some reason
      if (dimension == 2) {
        numSidesPerElem = cellTopo->getEdgeCount();
      }
      else if (dimension == 3) {
        numSidesPerElem = cellTopo->getFaceCount();
      }
      
      std::string blockID = block_names[block];
      vector<stk::mesh::Entity> stk_meshElems = mesh->getMySTKElements(blockID);
      size_t maxElemLID = 0;
      for (size_t i=0; i<stk_meshElems.size(); i++) {
        size_t lid = mesh->getSTKElementLocalId(stk_meshElems[i]);
        maxElemLID = std::max(lid,maxElemLID);
      }
      std::vector<size_t> localelemmap(maxElemLID+1);
      for (size_t i=0; i<stk_meshElems.size(); i++) {
        size_t lid = mesh->getSTKElementLocalId(stk_meshElems[i]);
        localelemmap[lid] = i;
      }

      Teuchos::ParameterList blocksettings = physics->physics_settings[set][block];
    
      Teuchos::ParameterList dbc_settings = blocksettings.sublist("Dirichlet conditions");
      Teuchos::ParameterList nbc_settings = blocksettings.sublist("Neumann conditions");
      Teuchos::ParameterList fbc_settings = blocksettings.sublist("Far-field conditions");
      Teuchos::ParameterList sbc_settings = blocksettings.sublist("Slip conditions");
      Teuchos::ParameterList flux_settings = blocksettings.sublist("Flux conditions");
      bool use_weak_dbcs = dbc_settings.get<bool>("use weak Dirichlet",false);
      
      Kokkos::View<int****,HostDevice> currside_info;
      if (requires_sideinfo) {
        currside_info = Kokkos::View<int****,HostDevice>("side info",stk_meshElems.size(),
                                                         varlist[block].size(),numSidesPerElem,2);
      }
      else {
        currside_info = Kokkos::View<int****,HostDevice>("side info",1,1,1,2);
      }

      std::vector<int> block_dbc_dofs;
      
      std::string perBCs = settings->sublist("Mesh").get<string>("Periodic Boundaries","");

      for (size_t j=0; j<varlist[block].size(); j++) {
        string var = varlist[block][j];
        vector<string> current_var_bcs(sideSets.size(),"none"); // [boundary]
        
        for (size_t side=0; side<sideSets.size(); side++ ) {
          string sideName = sideSets[side];
          
          vector<stk::mesh::Entity> sideEntities = mesh->getMySTKSides(sideName, blockID);
          
          bool isDiri = false;
          bool isNeum = false;
          bool isFar  = false;
          bool isSlip = false;
          bool isFlux = false;

          // Check for scalar Dirichlet BC (e.g., "E: all boundaries: ...")
          bool hasScalarDiri = dbc_settings.sublist(var).isParameter("all boundaries") ||
                               dbc_settings.sublist(var).isParameter(sideName);
          if (hasScalarDiri) {
            isDiri = true;
          }
          
          // For HCURL/HDIV variables, also check for component sublists (e.g., "Ex:", "Ey:", "Ez:")
          // Component based values take precedence over scalar values when both are specified.
          std::string vartype = physics->types[set][block][j];
          bool is_vector_type = (vartype.substr(0,5) == "HCURL" || vartype.substr(0,4) == "HDIV");
          bool hasComponentDiri = false;
          if (is_vector_type) {
            // Check if any component has a Dirichlet BC on this boundary
            std::vector<std::string> components = {"x", "y", "z"};
            for (const auto& comp : components) {
              std::string var_comp = var + comp;
              if (dbc_settings.sublist(var_comp).isParameter("all boundaries") ||
                  dbc_settings.sublist(var_comp).isParameter(sideName)) {
                hasComponentDiri = true;
                if (!isDiri) {
                  isDiri = true;
                }
              }
            }
            
            if (hasScalarDiri && hasComponentDiri) {
              cout << "WARNING: Dirichlet condition for variable '" << var
                        << "' on boundary '" << sideName << "' has both scalar ('" << var
                        << ":') and component ('" << var << "x/y/z:') entries. " << endl;
            }
          }
          
          if (isDiri) {
            if (use_weak_dbcs) {
              current_var_bcs[side] = "weak Dirichlet";
            }
            else {
              current_var_bcs[side] = "Dirichlet";
            }
          }
          if (nbc_settings.sublist(var).isParameter("all boundaries") || nbc_settings.sublist(var).isParameter(sideName)) {
            isNeum = true;
            current_var_bcs[side] = "Neumann";
          }
          if (fbc_settings.sublist(var).isParameter("all boundaries") || fbc_settings.sublist(var).isParameter(sideName)) {
            isFar = true;
            current_var_bcs[side] = "Far-field";
          }
          if (sbc_settings.sublist(var).isParameter("all boundaries") || sbc_settings.sublist(var).isParameter(sideName)) {
            isSlip = true;
            current_var_bcs[side] = "Slip";
          }
          if (flux_settings.sublist(var).isParameter("all boundaries") || flux_settings.sublist(var).isParameter(sideName)) {
            isFlux = true;
            current_var_bcs[side] = "Flux";
          }

          if (requires_sideinfo) {
            vector<size_t>             local_side_Ids;
            vector<stk::mesh::Entity> side_output;
            vector<size_t>             local_elem_Ids;
            mesh->getSTKSideElements(blockID, sideEntities, local_side_Ids, side_output);
            //panzer_stk::workset_utils::getSideElements(*mesh, blockID, sideEntities, local_side_Ids, side_output);
            
            for (size_t i=0; i<side_output.size(); i++ ) {
              local_elem_Ids.push_back(mesh->getSTKElementLocalId(side_output[i]));
              size_t localid = localelemmap[local_elem_Ids[i]];
              if (isDiri) {
                if (use_weak_dbcs) {
                  currside_info(localid, j, local_side_Ids[i], 0) = 4;
                }
                else {
                  currside_info(localid, j, local_side_Ids[i], 0) = 1;
                }
                currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
              }
              else if (isNeum) { // Neumann or Robin
                currside_info(localid, j, local_side_Ids[i], 0) = 2;
                currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
              }
              else if (isFar) { // Far-field
                currside_info(localid, j, local_side_Ids[i], 0) = 6;
                currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
              }
              else if (isSlip) { // Slip
                currside_info(localid, j, local_side_Ids[i], 0) = 7;
                currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
              }
              else if (isFlux) { // Flux
                currside_info(localid, j, local_side_Ids[i], 0) = 8;
                currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
              }
            }
          }
        }
      
        block_var_bcs.push_back(current_var_bcs);
        
        // nodeset loop
        string point_DBCs = blocksettings.get<std::string>(var+"_point_DBCs","");
        
        vector<int> dbc_nodes;
        for( size_t node=0; node<nodeSets.size(); node++ ) {
          string nodeName = nodeSets[node];
          std::size_t found = point_DBCs.find(nodeName);
          bool isDiri = false;
          if (found!=std::string::npos) {
            isDiri = true;
          }
          
          if (isDiri && !use_weak_dbcs) {
            vector<stk::mesh::Entity> nodeEntities = mesh->getMySTKNodes(nodeName, blockID);
            vector<GO> elemGIDs;
            
            vector<size_t> local_elem_Ids;
            vector<size_t> local_node_Ids;
            vector<stk::mesh::Entity> side_output;
            mesh->getSTKNodeElements(blockID, nodeEntities, local_node_Ids, side_output);

            for( size_t i=0; i<side_output.size(); i++ ) {
              local_elem_Ids.push_back(mesh->getSTKElementLocalId(side_output[i]));
              size_t localid = localelemmap[local_elem_Ids[i]];
              for (size_t k=0; k<dof_lids[set].extent(1); ++k) {
                GO gid = dof_owned_and_shared[set](dof_lids[set](localid,k));
                //GO gid = dof_owned_and_shared[set][dof_lids[set](localid,k)];
                elemGIDs.push_back(gid);
                //elemGIDs.push_back(dof_gids[set](localid,k));
              }
              //elemGIDs = dof_gids[set][localid];
              //currDOF->getElementGIDs(localid,elemGIDs,blockID);
              block_dbc_dofs.push_back(elemGIDs[offsets[set][block][j][local_node_Ids[i]]]);
            }
          }
          
        }
      }
    
      
      set_var_bcs.push_back(block_var_bcs);
      set_side_info.push_back(currside_info);
      
      std::sort(block_dbc_dofs.begin(), block_dbc_dofs.end());
      block_dbc_dofs.erase(std::unique(block_dbc_dofs.begin(),
                                       block_dbc_dofs.end()), block_dbc_dofs.end());
      
      int localsize = (int)block_dbc_dofs.size();
      int globalsize = 0;
      
      Teuchos::reduceAll<int,int>(*comm,Teuchos::REDUCE_SUM,1,&localsize,&globalsize);
      int gathersize = comm->getSize()*globalsize;
      int *block_dbc_dofs_local = new int [globalsize];
      int *block_dbc_dofs_global = new int [gathersize];
      
      int mxdof = (int) block_dbc_dofs.size();
      for (int i = 0; i < globalsize; i++) {
        if ( i < mxdof) {
          block_dbc_dofs_local[i] = (int) block_dbc_dofs[i];
        }
        else {
          block_dbc_dofs_local[i] = -1;
        }
      }
      
      Teuchos::gatherAll(*comm, globalsize, &block_dbc_dofs_local[0], gathersize, &block_dbc_dofs_global[0]);
      vector<GO> all_dbcs;
      
      for (int i = 0; i < gathersize; i++) {
        all_dbcs.push_back(block_dbc_dofs_global[i]);
      }
      delete [] block_dbc_dofs_local;
      delete [] block_dbc_dofs_global;
      
      vector<GO> dbc_final;
      {
        vector<GO> ownedAndShared(dof_owned_and_shared[set].extent(0));
        for (size_t i=0; i<ownedAndShared.size(); ++i) {
          ownedAndShared[i] = dof_owned_and_shared[set](i);
        }
      
        sort(all_dbcs.begin(),all_dbcs.end());
        sort(ownedAndShared.begin(),ownedAndShared.end());
        set_intersection(all_dbcs.begin(),all_dbcs.end(),
                         ownedAndShared.begin(),ownedAndShared.end(),
                         back_inserter(dbc_final));
        
        //sort(dof_owned_and_shared[set].begin(),dof_owned_and_shared[set].end());
        //set_intersection(all_dbcs.begin(),all_dbcs.end(),
        //                 dof_owned_and_shared[set].begin(),dof_owned_and_shared[set].end(),
        //                 back_inserter(dbc_final));
      
        set_point_dofs.push_back(dbc_final);
      }
    } // blocks
    
    var_bcs.push_back(set_var_bcs);
    side_info.push_back(set_side_info);
    point_dofs.push_back(set_point_dofs);
    
  //} // sets
  
  debugger->print("**** Finished DiscretizationInterface::setBCData");
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::setDirichletData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF) {
  
  Teuchos::TimeMonitor localtimer(*set_dbc_timer);
  
  debugger->print("**** Starting DiscretizationInterface::setDirichletData ...");
  
  //vector<string> side_names;
  //mesh->getSidesetNames(side_names);
  
  //for (size_t set=0; set<physics->setnames.size(); ++set) {
    
    vector<vector<string> > varlist = physics->var_list[set];
    //auto currDOF = DOF[set];
    
    std::vector<std::vector<std::vector<LO> > > set_dbc_dofs;
    
    for (size_t block=0; block<block_names.size(); ++block) {
      
      std::string blockID = block_names[block];
      
      Teuchos::ParameterList dbc_settings = physics->physics_settings[set][block].sublist("Dirichlet conditions");
      bool use_weak_dbcs = dbc_settings.get<bool>("use weak Dirichlet",false);

      std::vector<std::vector<LO> > block_dbc_dofs;
      
      for (size_t j=0; j<varlist[block].size(); j++) {
        std::string var = varlist[block][j];
        
        int fieldnum = DOF->getFieldNum(var);

        std::vector<LO> var_dofs;
        for (size_t side=0; side<side_names.size(); side++ ) {
          std::string sideName = side_names[side];
          vector<stk::mesh::Entity> sideEntities = mesh->getMySTKSides(sideName, blockID);
          
          bool isDiri = false;
          if (dbc_settings.sublist(var).isParameter("all boundaries") || dbc_settings.sublist(var).isParameter(sideName)) {
            isDiri = true;
            have_dirichlet = true;
          }
          
          // For HCURL/HDIV variables, also check for component sublists (Ex, Ey, Ez)
          // This allows for component-only BC specification without requiring the base variable entry
          std::string vartype = physics->types[set][block][j];
          bool is_vector_type = (vartype.substr(0,5) == "HCURL" || vartype.substr(0,4) == "HDIV");
          if (is_vector_type && !isDiri) {
            std::vector<std::string> components = {"x", "y", "z"};
            for (const auto& comp : components) {
              std::string var_comp = var + comp;
              if (dbc_settings.sublist(var_comp).isParameter("all boundaries") ||
                  dbc_settings.sublist(var_comp).isParameter(sideName)) {
                isDiri = true;
                have_dirichlet = true;
                break;
              }
            }
          }
          
          if (isDiri  && !use_weak_dbcs) {
            
            vector<size_t>             local_side_Ids;
            vector<stk::mesh::Entity>  side_output;
            vector<size_t>             local_elem_Ids;
            mesh->getSTKSideElements(blockID, sideEntities, local_side_Ids, side_output);
            //panzer_stk::workset_utils::getSideElements(*mesh, blockID, sideEntities,
            //                                           local_side_Ids, side_output);

            for( size_t i=0; i<side_output.size(); i++ ) {
              LO local_EID = mesh->getSTKElementLocalId(side_output[i]);
              auto elemLIDs = DOF->getElementLIDs(local_EID);
              const std::pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fieldnum,
                                                                                                        dimension-1,
                                                                                                        local_side_Ids[i]);
              const vector<int> sideOffset = SideIndex.first;
              
              for( size_t i=0; i<sideOffset.size(); i++ ) { // for each node
                var_dofs.push_back(elemLIDs(sideOffset[i]));
              }
            }
          }
        }
        std::sort(var_dofs.begin(), var_dofs.end());
        var_dofs.erase(std::unique(var_dofs.begin(), var_dofs.end()), var_dofs.end());
        block_dbc_dofs.push_back(var_dofs);
      }
      set_dbc_dofs.push_back(block_dbc_dofs);
      
    }
    
    dbc_dofs.push_back(set_dbc_dofs);
    
  //}
  
  debugger->print("**** Finished DiscretizationInterface::setDirichletData");
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<int****,HostDevice> DiscretizationInterface::getSideInfo(const size_t & set, const size_t & block,
                                                                      Kokkos::View<int*,HostDevice> elem) {
  
  Kokkos::View<int****,HostDevice> currsi;
  
  int maxe = 0;
  for (size_type e=0; e<elem.extent(0); ++e) {
    maxe = std::max(elem(e),maxe);
  }
  if (maxe < (int)side_info[set][block].extent(0)) {
    size_type nelem = elem.extent(0);
    size_type nvars = side_info[set][block].extent(1);
    size_type nelemsides = side_info[set][block].extent(2);
    currsi = Kokkos::View<int****,HostDevice>("side info for cell",nelem,nvars,nelemsides, 2);
    for (size_type e=0; e<nelem; e++) {
      for (size_type j=0; j<nelemsides; j++) {
        for (size_type i=0; i<nvars; i++) {
          int sidetype = side_info[set][block](elem(e),i,j,0);
          if (sidetype > 0) {
            currsi(e,i,j,0) = sidetype;
            currsi(e,i,j,1) = side_info[set][block](elem(e),i,j,1);
          }
          else {
            currsi(e,i,j,0) = sidetype;
            currsi(e,i,j,1) = 0;
          }
        }
      }
    }
  }
  return currsi;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<vector<int> > DiscretizationInterface::getOffsets(const int & set, const int & block) {
  return offsets[set][block];
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<GO> DiscretizationInterface::getGIDs(const size_t & set, const size_t & block, const size_t & elem) {
  vector<GO> gids;
  for (size_t k=0; k<dof_lids[set].extent(1); ++k) {
    GO gid = dof_owned_and_shared[set](dof_lids[set](elem,k));
    //GO gid = dof_owned_and_shared[set][dof_lids[set](elem,k)];
    gids.push_back(gid);
    //gids.push_back(dof_gids[set](elem,k));
  }
  return gids;//dof_gids[set][elem];
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<string**,HostDevice> DiscretizationInterface::getVarBCs(const size_t & set, const size_t & block) {
  
  
  size_t numvars = var_bcs[set][block].size();
  Kokkos::View<string**,HostDevice> bcs("BCs for each variable",numvars, side_names.size());
  for (size_t var=0; var<numvars; ++var) {
    for (size_t side=0; side<side_names.size(); ++side) {
      bcs(var,side) = var_bcs[set][block][var][side];
    }
  }
  return bcs;
}

