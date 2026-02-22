/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ========================================================================================
// Constructor to set up a mesh interface that builds an stk or simple mesh
// ========================================================================================

MeshInterface::MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                             const Teuchos::RCP<MpiComm> & comm_) :
settings(settings_), comm(comm_) {
  
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::meshInterface - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), comm));
  
  debugger->print("**** Starting mesh interface constructor ...");
  
  dimension = settings->sublist("Mesh").get<int>("dimension",0);
  verbosity = settings->get<int>("verbosity",0);
  
  use_stk_mesh = settings->sublist("Mesh").get<bool>("use STK mesh",true);
  use_simple_mesh = settings->sublist("Mesh").get<bool>("use simple mesh",false);
  
  if (use_simple_mesh) {
    use_stk_mesh = false;

    Teuchos::ParameterList pl;
    pl.sublist("Geometry").set("X0",     settings->sublist("Mesh").get("xmin",0.0));
    pl.sublist("Geometry").set("Width",  settings->sublist("Mesh").get("xmax",1.0)-settings->sublist("Mesh").get("xmin",0.0));
    pl.sublist("Geometry").set("NX",     settings->sublist("Mesh").get("NX",20));
    // if dim>1
    pl.sublist("Geometry").set("Y0",     settings->sublist("Mesh").get("ymin",0.0));
    pl.sublist("Geometry").set("Height", settings->sublist("Mesh").get("ymax",1.0)-settings->sublist("Mesh").get("ymin",0.0));
    pl.sublist("Geometry").set("NY",     settings->sublist("Mesh").get("NY",20));
    // if dim>2
    pl.sublist("Geometry").set("Z0",     settings->sublist("Mesh").get("zmin",0.0));
    pl.sublist("Geometry").set("Depth",  settings->sublist("Mesh").get("zmax",1.0)-settings->sublist("Mesh").get("zmin",0.0));
    pl.sublist("Geometry").set("NZ",     settings->sublist("Mesh").get("NZ",20));

    if (comm->getSize() == 1) {
      if (dimension == 2) {
        simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle<ScalarT>>(new SimpleMeshManager_Rectangle<ScalarT>(pl));
      }
      else if (dimension == 3) {
        simple_mesh = Teuchos::RCP<SimpleMeshManager_Brick<ScalarT>>(new SimpleMeshManager_Brick<ScalarT>(pl));
      }
    }
    else {
      if (dimension == 2) {
        int xprocs = settings->sublist("Mesh").get("Xprocs",comm->getSize());
        int yprocs = settings->sublist("Mesh").get("Yprocs",1);
        if (xprocs*yprocs != comm->getSize()) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: number of xprocs*yprocs not equal to MPI Comm size");
        }
        simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle_Parallel<ScalarT>>(new SimpleMeshManager_Rectangle_Parallel<ScalarT>(pl, comm->getRank(), xprocs, yprocs));
      }
      else if (dimension == 3) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: parallel simple mesh not implemented yet");
      }
    }
  }
  shape = settings->sublist("Mesh").get<string>("shape","none");
  if (shape == "none") { // new keywords, but allowing BWDS compat.
    shape = settings->sublist("Mesh").get<string>("element type","quad");
  }
  
  have_mesh_data = false;
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  have_rotations = false;
  have_rotation_phi = false;
  have_quadrature_data = false;
  mesh_data_file_tag = "none";
  mesh_data_pts_tag = "none";
  
  mesh_data_tag = settings->sublist("Mesh").get<string>("data file","none");
  if (mesh_data_tag != "none") {
    mesh_data_pts_tag = settings->sublist("Mesh").get<string>("data points file","mesh_data_pts");
    
    have_mesh_data = true;
    have_rotation_phi = settings->sublist("Mesh").get<bool>("have mesh data phi",false);
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",false);
    have_quadrature_data = settings->sublist("Mesh").get<bool>("have mesh quadrature data",false);
  }
  
  meshmod_xvar = settings->sublist("Solver").get<int>("solution for x-mesh mod",-1);
  meshmod_yvar = settings->sublist("Solver").get<int>("solution for y-mesh mod",-1);
  meshmod_zvar = settings->sublist("Solver").get<int>("solution for z-mesh mod",-1);
  meshmod_TOL = settings->sublist("Solver").get<ScalarT>("solution based mesh mod TOL",1.0);
  meshmod_usesmoother = settings->sublist("Solver").get<bool>("solution based mesh mod smoother",false);
  meshmod_center = settings->sublist("Solver").get<ScalarT>("solution based mesh mod param",0.1);
  meshmod_layer_size = settings->sublist("Solver").get<ScalarT>("solution based mesh mod layer thickness",0.1);
  
  shards::CellTopology cTopo;
  shards::CellTopology sTopo;
  
  if (dimension == 1) {
    cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );// lin. cell topology on the interior
    sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Node>() );          // line cell topology on the boundary
  }
  if (dimension == 2) {
    if (shape == "quad") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );          // line cell topology on the boundary
    }
    if (shape == "tri") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );          // line cell topology on the boundary
    }
  }
  if (dimension == 3) {
    if (shape == "hex") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() );          // line cell topology on the boundary
    }
    if (shape == "tet") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() );          // line cell topology on the boundary
    }
    
  }
  // Get dimensions
  num_nodes_per_elem = cTopo.getNodeCount();
  settings->sublist("Mesh").set("numNodesPerElem",num_nodes_per_elem,"number of nodes per element");
  
  if (use_stk_mesh) {
    // Define a parameter list with the required fields for the panzer_stk mesh factory
    RCP<Teuchos::ParameterList> pl = rcp(new Teuchos::ParameterList);
  
    if (settings->sublist("Mesh").get<std::string>("source","Internal") ==  "Exodus") {
      mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory());
      pl->set("File Name",settings->sublist("Mesh").get<std::string>("mesh file","mesh.exo"));
    }
    else {
      pl->set("X Blocks",settings->sublist("Mesh").get("Xblocks",1));
      pl->set("X Elements",settings->sublist("Mesh").get("NX",20));
      pl->set("X0",settings->sublist("Mesh").get("xmin",0.0));
      pl->set("Xf",settings->sublist("Mesh").get("xmax",1.0));
      if (dimension > 1) {
        pl->set("X Procs", settings->sublist("Mesh").get("Xprocs",comm->getSize()));
        pl->set("Y Blocks",settings->sublist("Mesh").get("Yblocks",1));
        pl->set("Y Elements",settings->sublist("Mesh").get("NY",20));
        pl->set("Y0",settings->sublist("Mesh").get("ymin",0.0));
        pl->set("Yf",settings->sublist("Mesh").get("ymax",1.0));
        pl->set("Y Procs", settings->sublist("Mesh").get("Yprocs",1));
      }
      if (dimension > 2) {
        pl->set("Z Blocks",settings->sublist("Mesh").get("Zblocks",1));
        pl->set("Z Elements",settings->sublist("Mesh").get("NZ",20));
        pl->set("Z0",settings->sublist("Mesh").get("zmin",0.0));
        pl->set("Zf",settings->sublist("Mesh").get("zmax",1.0));
        pl->set("Z Procs", settings->sublist("Mesh").get("Zprocs",1));
      }
      if (dimension == 1) {
        mesh_factory = Teuchos::rcp(new panzer_stk::LineMeshFactory());
      }
      else if (dimension == 2) {
        if (shape == "quad") {
          mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory());
        }
        if (shape == "tri") {
            mesh_factory = Teuchos::rcp(new panzer_stk::SquareTriMeshFactory());
          }
        }
      else if (dimension == 3) {
        if (shape == "hex") {
          mesh_factory = Teuchos::rcp(new panzer_stk::CubeHexMeshFactory());
        }
        if (shape == "tet") {
          mesh_factory = Teuchos::rcp(new panzer_stk::CubeTetMeshFactory());
        }
      }
    }

    // Syntax for periodic BCs ... must be set in the mesh input file
    if (settings->sublist("Mesh").isSublist("Periodic BCs")) {
      pl->sublist("Periodic BCs").setParameters( settings->sublist("Mesh").sublist("Periodic BCs") );
    }
  
    mesh_factory->setParameterList(pl);
  
    // create the mesh
    stk_mesh = mesh_factory->buildUncommitedMesh(*(comm->getRawMpiComm()));
  
    // create a mesh for an optmization movie
    if (settings->sublist("Postprocess").get("create optimization movie",false)) {
      stk_optimization_mesh = mesh_factory->buildUncommitedMesh(*(comm->getRawMpiComm()));
    }
  
    stk_mesh->getElementBlockNames(block_names);
    stk_mesh->getSidesetNames(side_names);
    stk_mesh->getNodesetNames(node_names);
  }
  else if (use_simple_mesh) {
    block_names = { "eblock-0_0" };
    // GHDR: Need to define block_names, side_names, node_names
    if (dimension == 2) {
      cell_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<>>())));
      side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<>>())));
    }
    else if (dimension == 3) {
      cell_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<> >())));
      side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >())));
                                                                
    }
  }

  if(use_stk_mesh) {
    for (size_t b=0; b<block_names.size(); b++) {
      cell_topo.push_back(stk_mesh->getCellTopology(block_names[b]));
    }
  
    for (size_t b=0; b<block_names.size(); b++) {
      topo_RCP cell_topo = stk_mesh->getCellTopology(block_names[b]);
      string shape = cell_topo->getName();
      if (dimension == 1) {
        // nothing to do here
      }
      if (dimension == 2) {
        if (shape == "Quadrilateral_4") {
          side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
        }
        if (shape == "Triangle_3") {
          side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
        }
      }
      if (dimension == 3) {
        if (shape == "Hexahedron_8") {
          side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() )));
        }
        if (shape == "Tetrahedron_4") {
          side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() )));
        }
      }
    }
  }

  debugger->print("**** Finished mesh interface constructor");
}

////////////////////////////////////////////////////////////////////////////////
// Use an existing stk mesh/factory to create a mesh interface
////////////////////////////////////////////////////////////////////////////////

MeshInterface::MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                             const Teuchos::RCP<MpiComm> & comm_,
                             Teuchos::RCP<panzer_stk::STK_MeshFactory> & mesh_factory_,
                             Teuchos::RCP<panzer_stk::STK_Interface> & stk_mesh_) :
settings(settings_), comm(comm_), mesh_factory(mesh_factory_), stk_mesh(stk_mesh_) {
  
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), comm));
  
  debugger->print("**** Starting mesh interface constructor ...");
  
  shape = settings->sublist("Mesh").get<string>("shape","none");
  if (shape == "none") { // new keywords, but allowing BWDS compat.
    shape = settings->sublist("Mesh").get<string>("element type","quad");
  }
  dimension = settings->sublist("Mesh").get<int>("dim",0);
  if (dimension == 0) {
    dimension = settings->sublist("Mesh").get<int>("dimension",2);
  }
  
  have_mesh_data = false;
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  have_rotations = false;
  have_rotation_phi = false;
  have_quadrature_data = false;
  mesh_data_file_tag = "none";
  mesh_data_pts_tag = "none";
  
  mesh_data_tag = settings->sublist("Mesh").get<string>("data file","none");
  if (mesh_data_tag != "none") {
    mesh_data_pts_tag = settings->sublist("Mesh").get<string>("data points file","mesh_data_pts");
    
    have_mesh_data = true;
    have_rotation_phi = settings->sublist("Mesh").get<bool>("have mesh data phi",false);
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",true);
    have_quadrature_data = settings->sublist("Mesh").get<bool>("have mesh quadrature data",false);
  }
  
  meshmod_xvar = settings->sublist("Solver").get<int>("solution for x-mesh mod",-1);
  meshmod_yvar = settings->sublist("Solver").get<int>("solution for y-mesh mod",-1);
  meshmod_zvar = settings->sublist("Solver").get<int>("solution for z-mesh mod",-1);
  meshmod_TOL = settings->sublist("Solver").get<ScalarT>("solution based mesh mod TOL",1.0);
  meshmod_usesmoother = settings->sublist("Solver").get<bool>("solution based mesh mod smoother",false);
  meshmod_center = settings->sublist("Solver").get<ScalarT>("solution based mesh mod param",0.1);
  meshmod_layer_size = settings->sublist("Solver").get<ScalarT>("solution based mesh mod layer thickness",0.1);
  
  stk_mesh->getElementBlockNames(block_names);
  stk_mesh->getSidesetNames(side_names);
  stk_mesh->getNodesetNames(node_names);

  for (size_t b=0; b<block_names.size(); b++) {
    cell_topo.push_back(stk_mesh->getCellTopology(block_names[b]));
  }
  
  for (size_t b=0; b<block_names.size(); b++) {
    topo_RCP cell_topo = stk_mesh->getCellTopology(block_names[b]);
    string shape = cell_topo->getName();
    if (dimension == 1) {
      // nothing to do here?
    }
    if (dimension == 2) {
      if (shape == "Quadrilateral_4") {
        side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
      if (shape == "Triangle_3") {
        side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
    }
    if (dimension == 3) {
      if (shape == "Hexahedron_8") {
        side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() )));
      }
      if (shape == "Tetrahedron_4") {
        side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() )));
      }
    }
    
  }
  
  debugger->print("**** Finished mesh interface constructor");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void MeshInterface::finalize(std::vector<std::vector<std::vector<string> > > varlist,
                             std::vector<std::vector<std::vector<string> > > vartypes,
                             std::vector<std::vector<std::vector<std::vector<string> > > > derivedList) {
  
  debugger->print("**** Starting mesh interface finalize ...");
  
  ////////////////////////////////////////////////////////////////////////////////
  // Add fields to the mesh
  ////////////////////////////////////////////////////////////////////////////////
  if (settings->sublist("Postprocess").get("write solution",false)) {
    std::vector<std::string> appends;
    if (settings->sublist("Analysis").get<std::string>("analysis type","forward") == "UQ") {
      if (settings->sublist("Postprocess").get("write solution",false)) {
        int numsamples = settings->sublist("Analysis").sublist("UQ").get<int>("samples",100);
        for (int j=0; j<numsamples; ++j) {
          std::stringstream ss;
          ss << "_" << j;
          appends.push_back(ss.str());
        }
      }
      else {
        appends = {""};
      }
    }
    else {
      appends = {""};
    }
    
    for (size_t app=0; app<appends.size(); ++app) {
      
      std::string append = appends[app];
      for (std::size_t set=0; set<varlist.size(); ++set) {
        
        for (std::size_t blk=0;blk<block_names.size(); ++blk) {
          
          //std::vector<string> varlist = phys->var_list[set][i];
          //std::vector<string> vartype = phys->types[set][i];
          
          for (size_t var=0; var<varlist[set][blk].size(); var++) {
            if (vartypes[set][blk][var] == "HGRAD") {
              stk_mesh->addSolutionField(varlist[set][blk][var]+append, block_names[blk]);
            }
            else if (vartypes[set][blk][var] == "HVOL") { // PW constant
              stk_mesh->addCellField(varlist[set][blk][var]+append, block_names[blk]);
            }
            else if (vartypes[set][blk][var] == "HFACE") { // hybridized variable
              stk_mesh->addCellField(varlist[set][blk][var]+append, block_names[blk]);
            }
            else if (vartypes[set][blk][var] == "HDIV" || vartypes[set][blk][var] == "HCURL") { // HDIV or HCURL
              stk_mesh->addCellField(varlist[set][blk][var]+append+"x", block_names[blk]);
              if (dimension > 1) {
                stk_mesh->addCellField(varlist[set][blk][var]+append+"y", block_names[blk]);
              }
              if (dimension > 2) {
                stk_mesh->addCellField(varlist[set][blk][var]+append+"z", block_names[blk]);
              }
            }
          }
          
          //stk_mesh->addSolutionField("disp"+append+"x", block_names[i]);
          //stk_mesh->addSolutionField("disp"+append+"y", block_names[i]);
          //stk_mesh->addSolutionField("disp"+append+"z", block_names[i]);
          
          
          Teuchos::ParameterList efields;
          if (settings->sublist("Postprocess").isSublist(block_names[blk])) {
            efields = settings->sublist("Postprocess").sublist(block_names[blk]).sublist("Extra fields");
          }
          else {
            efields = settings->sublist("Postprocess").sublist("Extra fields");
          }
          Teuchos::ParameterList::ConstIterator ef_itr = efields.begin();
          while (ef_itr != efields.end()) {
            stk_mesh->addSolutionField(ef_itr->first+append, block_names[blk]);
            ef_itr++;
          }
          
          Teuchos::ParameterList ecfields;
          if (settings->sublist("Postprocess").isSublist(block_names[blk])) {
            ecfields = settings->sublist("Postprocess").sublist(block_names[blk]).sublist("Extra cell fields");
          }
          else {
            ecfields = settings->sublist("Postprocess").sublist("Extra cell fields");
          }
          Teuchos::ParameterList::ConstIterator ecf_itr = ecfields.begin();
          while (ecf_itr != ecfields.end()) {
            stk_mesh->addCellField(ecf_itr->first+append, block_names[blk]);
            if (settings->isSublist("Subgrid")) {
              string sgfn = "subgrid_mean_" + ecf_itr->first;
              stk_mesh->addCellField(sgfn+append, block_names[blk]);
            }
            ecf_itr++;
          }
          
          for (size_t var=0; var<derivedList[set][blk].size(); ++var) {
            //std::vector<string> derivedlist = phys->modules[set][i][j]->getDerivedNames();
            for (size_t k=0; k<derivedList[set][blk][var].size(); ++k) {
              stk_mesh->addCellField(derivedList[set][blk][var][k]+append, block_names[blk]);
            }
          }
          
          if (have_mesh_data || compute_mesh_data) {
            stk_mesh->addCellField("mesh_data_seed", block_names[blk]);
            stk_mesh->addCellField("mesh_data", block_names[blk]);
          }
          
          if (settings->isSublist("Subgrid")) {
            stk_mesh->addCellField("subgrid model", block_names[blk]);
          }
          
          if (settings->sublist("Postprocess").get("write group number",false)) {
            stk_mesh->addCellField("group number", block_names[blk]);
          }
          if (settings->sublist("Solver").get<bool>("use basis database",false)) {
            stk_mesh->addCellField("unique Jacobian ID", block_names[blk]);
          }
          if (settings->isSublist("Parameters")) {
            Teuchos::ParameterList parameters = settings->sublist("Parameters");
            Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
            while (pl_itr != parameters.end()) {
              Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
              if (newparam.get<string>("usage") == "discretized") {
                if (newparam.get<string>("type") == "HGRAD") {
                  stk_mesh->addSolutionField(pl_itr->first+append, block_names[blk]);
                }
                else if (newparam.get<string>("type") == "HVOL") {
                  stk_mesh->addCellField(pl_itr->first+append, block_names[blk]);
                }
                else if (newparam.get<string>("type") == "HDIV" || newparam.get<string>("type") == "HCURL") {
                  stk_mesh->addCellField(pl_itr->first+append+"x", block_names[blk]);
                  if (dimension > 1) {
                    stk_mesh->addCellField(pl_itr->first+append+"y", block_names[blk]);
                  }
                  if (dimension > 2) {
                    stk_mesh->addCellField(pl_itr->first+append+"z", block_names[blk]);
                  }
                }
              }
              pl_itr++;
            }
          }
        }
      }
      
    }
  }
  
  if (use_stk_mesh) {
    mesh_factory->completeMeshConstruction(*stk_mesh,*(comm->getRawMpiComm()));
  }

  if (verbosity>1) {
    if(use_stk_mesh) {
      if (comm->getRank() == 0) {
        stk_mesh->printMetaData(std::cout);
      }
    }
  }
  
  if (settings->sublist("Postprocess").get("create optimization movie",false)) {
    
    for(std::size_t blk=0; blk<block_names.size(); ++blk) {
      
      if (settings->isSublist("Parameters")) {
        Teuchos::ParameterList parameters = settings->sublist("Parameters");
        Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
        while (pl_itr != parameters.end()) {
          Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
          if (newparam.get<string>("usage") == "discretized") {
            if (newparam.get<string>("type") == "HGRAD") {
              stk_optimization_mesh->addSolutionField(pl_itr->first, block_names[blk]);
            }
            else if (newparam.get<string>("type") == "HVOL") {
              stk_optimization_mesh->addCellField(pl_itr->first, block_names[blk]);
            }
            else if (newparam.get<string>("type") == "HDIV" || newparam.get<string>("type") == "HCURL") {
              stk_optimization_mesh->addCellField(pl_itr->first+"x", block_names[blk]);
              if (dimension > 1) {
                stk_optimization_mesh->addCellField(pl_itr->first+"y", block_names[blk]);
              }
              if (dimension > 2) {
                stk_optimization_mesh->addCellField(pl_itr->first+"z", block_names[blk]);
              }
            }
          }
          pl_itr++;
        }
      }
    }
    
    mesh_factory->completeMeshConstruction(*stk_optimization_mesh,*(comm->getRawMpiComm()));
    if (verbosity>1) {
      stk_optimization_mesh->printMetaData(std::cout);
    }
  }
  
  if (settings->sublist("Mesh").get<bool>("have element data", false) ||
      settings->sublist("Mesh").get<bool>("have nodal data", false)) {
    this->readExodusData();
  }
  
  debugger->print("**** Finished mesh interface finalize");
  
}

