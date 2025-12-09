/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/


DiscretizationInterface::DiscretizationInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                                 Teuchos::RCP<MpiComm> & Comm_,
                                                 Teuchos::RCP<MeshInterface> & mesh_,
                                                 Teuchos::RCP<PhysicsInterface> & physics_) :
settings(settings_), comm(Comm_), mesh(mesh_), physics(physics_) {
  
  RCP<Teuchos::Time> constructor_time = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface - constructor");
  Teuchos::TimeMonitor constructor_timer(*constructor_time);
    
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), comm));
  
  verbosity = settings->get<int>("verbosity",0);
  minimize_memory = settings->sublist("Solver").get<bool>("minimize memory",false);
  
  debugger->print("**** Starting DiscretizationInterface constructor...");
  
  ////////////////////////////////////////////////////////////////////////////////
  // Collect some information
  ////////////////////////////////////////////////////////////////////////////////
  
  dimension = mesh->getDimension();
  block_names = mesh->getBlockNames();
  side_names = mesh->getSideNames();

  ////////////////////////////////////////////////////////////////////////////////
  // Assemble the information we always store
  ////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<int> > orders = physics->unique_orders;
  vector<vector<string> > types = physics->unique_types;
  
  for (size_t block=0; block<block_names.size(); ++block) {
    
    string blockID = block_names[block];
    topo_RCP cellTopo = mesh->getCellTopology(blockID);
    string shape = cellTopo->getName();
    
    if (mesh->use_stk_mesh) {
      vector<stk::mesh::Entity> stk_meshElems = mesh->getMySTKElements(blockID);
      
      // list of all elements on this processor
      Kokkos::View<LO*,HostDevice> blockmy_elements("list of elements",stk_meshElems.size());
      for( size_t e=0; e<stk_meshElems.size(); e++ ) {
        blockmy_elements(e) = mesh->getSTKElementLocalId(stk_meshElems[e]);
      }
      my_elements.push_back(blockmy_elements);
    } else {
      Kokkos::View<LO*,HostDevice> blockmy_elements("list of elements",mesh->simple_mesh->getNumCells());
      for(unsigned int i=0; i<blockmy_elements.size(); ++i)
        blockmy_elements(i) = i;
      my_elements.push_back(blockmy_elements);
      //cout << blockmy_elements.size() << endl;
      
    }
    
    vector<int> blockcards;
    vector<basis_RCP> blockbasis;
    
    vector<int> doneorders;
    vector<string> donetypes;
    
    for (size_t set=0; set<physics->set_names.size(); ++set) {
      Teuchos::ParameterList db_settings = physics->disc_settings[set][block];
      
      ///////////////////////////////////////////////////////////////////////////
      // Get the cardinality of the basis functions  on this block
      ///////////////////////////////////////////////////////////////////////////
      
      for (size_t n=0; n<orders[block].size(); n++) {
        bool go = true;
        for (size_t i=0; i<doneorders.size(); i++){
          if (doneorders[i] == orders[block][n] && donetypes[i] == types[block][n]) {
            go = false;
          }
        }
        if (go) {
          basis_RCP basis = this->getBasis(dimension, cellTopo, types[block][n], orders[block][n]);
          int bsize = basis->getCardinality();
          blockcards.push_back(bsize); // cardinality of the basis
          blockbasis.push_back(basis);
          doneorders.push_back(orders[block][n]);
          donetypes.push_back(types[block][n]);
        }
      }
    }
    basis_types.push_back(donetypes);
    cards.push_back(blockcards);
    
    ///////////////////////////////////////////////////////////////////////////
    // Quadrature
    ///////////////////////////////////////////////////////////////////////////
    
    int mxorder = 0;
    for (size_t i=0; i<orders[block].size(); i++) {
      if (orders[block][i]>mxorder) {
        mxorder = orders[block][i];
      }
    }
    
    DRV qpts, qwts;
    quadorder = physics->disc_settings[0][block].get<int>("quadrature",2*mxorder); // hard coded
    this->getQuadrature(cellTopo, quadorder, qpts, qwts);
    
    ///////////////////////////////////////////////////////////////////////////
    // Side Quadrature
    ///////////////////////////////////////////////////////////////////////////
    
    topo_RCP sideTopo;
    
    if (dimension == 1) {
      sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Node >() ));
    }
    if (dimension == 2) {
      if (shape == "Quadrilateral_4") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
      if (shape == "Triangle_3") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
    }
    if (dimension == 3) {
      if (shape == "Hexahedron_8") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ));
      }
      if (shape == "Tetrahedron_4") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() ));
      }
    }
    
    DRV side_qpts, side_qwts;
    if (dimension == 1) {
      side_qpts = DRV("side qpts",1,1);
      Kokkos::deep_copy(side_qpts,-1.0);
      side_qwts = DRV("side wts",1,1);
      Kokkos::deep_copy(side_qwts,1.0);
    }
    else {
      int side_quadorder = physics->disc_settings[0][block].get<int>("side quadrature",2*mxorder); // hard coded
      this->getQuadrature(sideTopo, side_quadorder, side_qpts, side_qwts);
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Store locally
    ///////////////////////////////////////////////////////////////////////////
    
    basis_pointers.push_back(blockbasis);
    ref_ip.push_back(qpts);
    ref_wts.push_back(qwts);
    ref_side_ip.push_back(side_qpts);
    ref_side_wts.push_back(side_qwts);
    
    numip.push_back(qpts.extent(0));
    numip_side.push_back(side_qpts.extent(0));
    
  } // block loop
  
  // We do not actually store the DOF or Connectivity managers
  // Probably require:
  // std::vector<Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device>> dof_lids; [set](elem, dof)
  // std::vector<std::vector<GO> > dof_owned, dof_owned_and_shared; // list of degrees of freedom on processor
  // std::vector<std::vector<std::vector<GO>>> dof_gids; // [set][elem][dof]
  // vector<vector<vector<vector<int> > > > offsets; // [set][block][var][dof]

  // May also need to fill:
  // std::vector<Intrepid2::Orientation> panzer_orientations; [elem]
  // vector<int> num_derivs_required; [block] (takes max over sets)
     
  if (mesh->use_stk_mesh) {
    this->buildDOFManagers();
  }
  else {
    // GHDR: need to fill in the objects listed above (try it without the orientations and num_derivs_required)

    // GH: this simply pushes back DOFs 0,1,...,N-1 where N is the number of nodes for owned and ownedAndShared
    //vector<GO> owned;
    //for(unsigned int i=0; i < (unsigned int) mesh->simple_mesh->getNumNodes(); ++i)
    //  owned.push_back(((GO) i));
    size_t num_owned = 0;
    for (unsigned int i=0; i < (unsigned int) mesh->simple_mesh->getNumNodes(); ++i) {
      bool isshared = mesh->simple_mesh->isShared(i);
      if (!isshared) {
        num_owned++;
      }
    }
    
    Kokkos::View<GO*,HostDevice> owned("owned dofs",num_owned);
    size_t prog = 0;
    for (unsigned int i=0; i < (unsigned int) mesh->simple_mesh->getNumNodes(); ++i) {
      bool isshared = mesh->simple_mesh->isShared(i);
      if (!isshared) {
        owned(prog) = mesh->simple_mesh->localToGlobal(i);
        ++prog;
      }
    }
    dof_owned.push_back(owned);
    
    //for (size_type i=0; i<owned.extent(0); ++i) {
    //  cout << comm->getRank() << "  " << owned(i) << endl;
    //}
    
    Kokkos::View<GO*,HostDevice> owned_shared("owned and shared dofs",mesh->simple_mesh->getNumNodes());
    for (unsigned int i=0; i < (unsigned int) mesh->simple_mesh->getNumNodes(); ++i) {
      owned_shared(i) = mesh->simple_mesh->localToGlobal(i);
    }
    
    //for (size_type i=0; i<owned_shared.extent(0); ++i) {
    //  cout << comm->getRank() << "  " << owned_shared(i) << endl;
    //}
    
    dof_owned_and_shared.push_back(owned_shared);

    dof_lids.push_back(mesh->simple_mesh->getCellToNodeMap()); // [set](elem, dof)
    
    /*
    //std::vector<std::vector<std::vector<GO>>> dof_gids; // [set][elem][dof]
    Kokkos::View<GO**,HostDevice> elemids("dof gids", dof_lids[dof_lids.size()-1].extent(0), dof_lids[dof_lids.size()-1].extent(1));
    for(unsigned int e=0; e<dof_lids[0].extent(0); ++e) {
      std::vector<GO> localelemids;
      for(unsigned int i=0; i<dof_lids[0].extent(1); ++i) {
        //localelemids.push_back(dof_lids[0](e,i));
        elemids(e,i) = dof_lids[0](e,i);
      }
      //elemids.push_back(localelemids);
    }
    dof_gids.push_back(elemids);
    */

    // vector<vector<vector<vector<int> > > > offsets; // [set][block][var][dof]
    for (size_t set=0; set<physics->set_names.size(); ++set) {
      vector<vector<string> > varlist = physics->var_list[set];
      vector<vector<vector<int> > > set_offsets; // [block][var][dof]
      for (size_t block=0; block<block_names.size(); ++block) {
        vector<vector<int> > celloffsets;
        for (size_t j=0; j<varlist[block].size(); j++) {
          string var = varlist[block][j];
          //int num = setDOF->getFieldNum(var);
          vector<int> var_offsets = {0, 1, 3, 2}; // GH: super hacky???

          celloffsets.push_back(var_offsets);
        }
        set_offsets.push_back(celloffsets);
      }
      offsets.push_back(set_offsets);

      // more hacky stuff; can't set dbcs without dof manager, but we don't have a dof manager
      std::vector<std::vector<std::vector<LO> > > set_dbc_dofs;
      std::vector<std::vector<LO> > block_dbc_dofs;
      std::vector<LO> var_dofs;
      //var_dofs.push_back(0);
      block_dbc_dofs.push_back(var_dofs);
      set_dbc_dofs.push_back(block_dbc_dofs);
      dbc_dofs.push_back(set_dbc_dofs);

      // parameter manager wants num_derivs_required
      num_derivs_required = std::vector<int>(1);

    }
    
    //panzer_orientations = Kokkos::View<Intrepid2::Orientation*,HostDevice>("panzer orient",mesh->simple_mesh->getNumCells());
    panzer_orientations = Kokkos::View<Intrepid2::Orientation*,HostDevice>("panzer orient",1);

  }
  
  //for (size_type i=0; i<dof_lids[0].extent(0); ++i) {
  //  cout << i << "  ";
  //  for (size_type j=0; j<dof_lids[0].extent(1); ++j) {
  //    cout << dof_lids[0](i,j) << " ";
  //  }
  //  cout << endl;
  //}
  
  debugger->print("**** Finished DiscretizationInterface constructor");
  
}


