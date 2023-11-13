/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "meshInterface.hpp"
#include "exodusII.h"

using namespace MrHyDE;

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
  
  debug_level = settings->get<int>("debug level",0);
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting mesh interface constructor ..." << endl;
    }
  }
  
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

    simple_mesh = Teuchos::RCP<SimpleMeshManager_Rectangle<ScalarT>>(new SimpleMeshManager_Rectangle<ScalarT>(pl));
  }
  shape = settings->sublist("Mesh").get<string>("shape","none");
  if (shape == "none") { // new keywords, but allowing BWDS compat.
    shape = settings->sublist("Mesh").get<string>("element type","quad");
  }
  dimension = settings->sublist("Mesh").get<int>("dim",0);
  if (dimension == 0) {
    dimension = settings->sublist("Mesh").get<int>("dimension",2);
  }
  verbosity = settings->get<int>("verbosity",0);
  
  have_mesh_data = false;
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  have_rotations = false;
  have_rotation_phi = false;
  mesh_data_file_tag = "none";
  mesh_data_pts_tag = "mesh_data_pts";
  
  mesh_data_tag = settings->sublist("Mesh").get<string>("data file","none");
  if (mesh_data_tag != "none") {
    mesh_data_pts_tag = settings->sublist("Mesh").get<string>("data points file","mesh_data_pts");
    
    have_mesh_data = true;
    have_rotation_phi = settings->sublist("Mesh").get<bool>("have mesh data phi",false);
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",false);
    
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
    cell_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<>>())));
    side_topo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<>>())));
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

  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished mesh interface constructor" << endl;
    }
  }
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
  
  debug_level = settings->get<int>("debug level",0);
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting mesh interface constructor ..." << endl;
    }
  }
  
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
  mesh_data_file_tag = "none";
  mesh_data_pts_tag = "mesh_data_pts";
  
  mesh_data_tag = settings->sublist("Mesh").get<string>("data file","none");
  if (mesh_data_tag != "none") {
    mesh_data_pts_tag = settings->sublist("Mesh").get<string>("data points file","mesh_data_pts");
    
    have_mesh_data = true;
    have_rotation_phi = settings->sublist("Mesh").get<bool>("have mesh data phi",false);
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",true);
    
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
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished mesh interface constructor" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void MeshInterface::finalize(std::vector<std::vector<std::vector<string> > > varlist,
                             std::vector<std::vector<std::vector<string> > > vartypes,
                             std::vector<std::vector<std::vector<std::vector<string> > > > derivedList) {
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting mesh interface finalize ..." << endl;
    }
  }
  
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
  
  if(use_stk_mesh) {
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
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished mesh interface finalize" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV MeshInterface::perturbMesh(const int & b, DRV & blocknodes) {
  
  ////////////////////////////////////////////////////////////////////////////////
  // Perturb the mesh (if requested)
  ////////////////////////////////////////////////////////////////////////////////
  
  //for (size_t b=0; b<block_names.size(); b++) {
    //vector<size_t> localIds;
    //DRV blocknodes;
    //panzer_stk::workset_utils::getIdsAndVertices(*mesh, block_names[b], localIds, blocknodes);
    int numNodesPerElem = blocknodes.extent(1);
    DRV blocknodePert("blocknodePert",blocknodes.extent(0),numNodesPerElem,dimension);
    
    if (settings->sublist("Mesh").get("modify mesh height",false)) {
      vector<vector<ScalarT> > values;
      
      string ptsfile = settings->sublist("Mesh").get("mesh pert file","meshpert.dat");
      std::ifstream fin(ptsfile.c_str());
      
      for (string line; getline(fin, line); )
      {
        replace(line.begin(), line.end(), ',', ' ');
        std::istringstream in(line);
        values.push_back(vector<ScalarT>(std::istream_iterator<ScalarT>(in),
                                         std::istream_iterator<ScalarT>()));
      }
      
      DRV pertdata("pertdata",values.size(),3);
      for (size_t i=0; i<values.size(); i++) {
        for (size_t j=0; j<3; j++) {
          pertdata(i,j) = values[i][j];
        }
      }
      //int Nz = settings->sublist("Mesh").get<int>("NZ",1);
      ScalarT zmin = settings->sublist("Mesh").get<ScalarT>("zmin",0.0);
      ScalarT zmax = settings->sublist("Mesh").get<ScalarT>("zmax",1.0);
      for (size_type k=0; k<blocknodes.extent(0); k++) {
        for (int i=0; i<numNodesPerElem; i++){
          ScalarT x = blocknodes(k,i,0);
          ScalarT y = blocknodes(k,i,1);
          ScalarT z = blocknodes(k,i,2);
          int node = -1;
          ScalarT dist = (ScalarT)RAND_MAX;
          for( size_type j=0; j<pertdata.extent(0); j++ ) {
            ScalarT xhat = pertdata(j,0);
            ScalarT yhat = pertdata(j,1);
            ScalarT d = std::sqrt((x-xhat)*(x-xhat) + (y-yhat)*(y-yhat));
            if( d<dist ) {
              node = j;
              dist = d;
            }
          }
          if (node > 0) {
            ScalarT ch = pertdata(node,2);
            blocknodePert(k,i,0) = 0.0;
            blocknodePert(k,i,1) = 0.0;
            blocknodePert(k,i,2) = (ch)*(z-zmin)/(zmax-zmin);
          }
        }
        //for (int k=0; k<blocknodeVert.extent(0); k++) {
        //  for (int i=0; i<numNodesPerElem; i++){
        //    for (int s=0; s<dimension; s++) {
        //      blocknodeVert(k,i,s) += blocknodePert(k,i,s);
        //    }
        //  }
        //}
      }
    }
    
    if (settings->sublist("Mesh").get("modify mesh",false)) {
      for (size_type k=0; k<blocknodes.extent(0); k++) {
        for (int i=0; i<numNodesPerElem; i++){
          blocknodePert(k,i,0) = 0.0;
          blocknodePert(k,i,1) = 0.0;
          blocknodePert(k,i,2) = 0.0 + 0.2*sin(2*3.14159*blocknodes(k,i,0))*sin(2*3.14159*blocknodes(k,i,1));
        }
      }
      //for (int k=0; k<blocknodeVert.extent(0); k++) {
      //  for (int i=0; i<numNodesPerElem; i++){
      //    for (int s=0; s<dimension; s++) {
      //      blocknodeVert(k,i,s) += blocknodePert(k,i,s);
      //    }
      //  }
      //}
    }
    //nodepert.push_back(blocknodePert);
  //}
  return blocknodePert;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
    
void MeshInterface::setupExodusFile(const string & filename) {
  stk_mesh->setupExodusFile(filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void MeshInterface::setupOptimizationExodusFile(const string & filename) {
  stk_optimization_mesh->setupExodusFile(filename);
}
    

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

View_Sc2 MeshInterface::getElementCenters(DRV nodes, topo_RCP & reftopo) {
  
  typedef Intrepid2::CellTools<PHX::Device::execution_space> CellTools;

  DRV tmp_refCenter("cell center", dimension);
  CellTools::getReferenceCellCenter(tmp_refCenter, *reftopo);
  DRV refCenter("cell center", 1, dimension);
  auto cent_sv = subview(refCenter,0, ALL());
  deep_copy(cent_sv, tmp_refCenter);
  DRV tmp_centers("tmp physical cell centers", nodes.extent(0), 1, dimension);
  CellTools::mapToPhysicalFrame(tmp_centers, refCenter, nodes, *reftopo);
  View_Sc2 centers("physics cell centers", nodes.extent(0), dimension);
  auto tmp_centers_sv = subview(tmp_centers, ALL(), 0, ALL());
  deep_copy(centers, tmp_centers_sv);
  
  return centers;
  
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

View_Sc2 MeshInterface::generateNewMicrostructure(int & randSeed) {
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting mesh::generateNewMicrostructure ..." << endl;
    }
  }
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  have_rotations = true;
  have_rotation_phi = false;
  
  View_Sc2 seeds;
  random_seeds.push_back(randSeed);
  std::default_random_engine generator(randSeed);
  num_seeds = 0;
  
  ////////////////////////////////////////////////////////////////////////////////
  // Generate the micro-structure using seeds and nearest neighbors
  ////////////////////////////////////////////////////////////////////////////////
  
  bool fast_and_crude = settings->sublist("Mesh").get<bool>("fast and crude microstructure",false);
  
  if (fast_and_crude) {
    int numxSeeds = settings->sublist("Mesh").get<int>("number of xseeds",10);
    int numySeeds = settings->sublist("Mesh").get<int>("number of yseeds",10);
    int numzSeeds = settings->sublist("Mesh").get<int>("number of zseeds",10);
    
    ScalarT xmin = settings->sublist("Mesh").get<ScalarT>("x min",0.0);
    ScalarT ymin = settings->sublist("Mesh").get<ScalarT>("y min",0.0);
    ScalarT zmin = settings->sublist("Mesh").get<ScalarT>("z min",0.0);
    ScalarT xmax = settings->sublist("Mesh").get<ScalarT>("x max",1.0);
    ScalarT ymax = settings->sublist("Mesh").get<ScalarT>("y max",1.0);
    ScalarT zmax = settings->sublist("Mesh").get<ScalarT>("z max",1.0);
    
    ScalarT dx = (xmax-xmin)/(ScalarT)(numxSeeds+1);
    ScalarT dy = (ymax-ymin)/(ScalarT)(numySeeds+1);
    ScalarT dz = (zmax-zmin)/(ScalarT)(numzSeeds+1);
    
    ScalarT maxpert = 0.25;
    
    Kokkos::View<ScalarT*,HostDevice> xseeds("xseeds",numxSeeds);
    Kokkos::View<ScalarT*,HostDevice> yseeds("yseeds",numySeeds);
    Kokkos::View<ScalarT*,HostDevice> zseeds("zseeds",numzSeeds);
    
    for (int k=0; k<numxSeeds; k++) {
      xseeds(k) = xmin + (k+1)*dx;
    }
    for (int k=0; k<numySeeds; k++) {
      yseeds(k) = ymin + (k+1)*dy;
    }
    for (int k=0; k<numzSeeds; k++) {
      zseeds(k) = zmin + (k+1)*dz;
    }
    
    std::uniform_real_distribution<ScalarT> pdistribution(-maxpert,maxpert);
    num_seeds = numxSeeds*numySeeds*numzSeeds;
    seeds = View_Sc2("seeds",num_seeds,3);
    auto seeds_host = create_mirror_view(seeds);
    
    int prog = 0;
    for (int i=0; i<numxSeeds; i++) {
      for (int j=0; j<numySeeds; j++) {
        for (int k=0; k<numzSeeds; k++) {
          ScalarT xp = pdistribution(generator);
          ScalarT yp = pdistribution(generator);
          ScalarT zp = pdistribution(generator);
          seeds_host(prog,0) = xseeds(i) + xp*dx;
          seeds_host(prog,1) = yseeds(j) + yp*dy;
          seeds_host(prog,2) = zseeds(k) + zp*dz;
          prog += 1;
        }
      }
    }
    deep_copy(seeds,seeds_host);
    
  }
  else {
    num_seeds = settings->sublist("Mesh").get<int>("number of seeds",10);
    seeds = View_Sc2("seeds",num_seeds,3);
    auto seeds_host = create_mirror_view(seeds);
    
    ScalarT xwt = settings->sublist("Mesh").get<ScalarT>("x weight",1.0);
    ScalarT ywt = settings->sublist("Mesh").get<ScalarT>("y weight",1.0);
    ScalarT zwt = settings->sublist("Mesh").get<ScalarT>("z weight",1.0);
    ScalarT nwt = sqrt(xwt*xwt+ywt*ywt+zwt*zwt);
    xwt *= 3.0/nwt;
    ywt *= 3.0/nwt;
    zwt *= 3.0/nwt;
    
    ScalarT xmin = settings->sublist("Mesh").get<ScalarT>("x min",0.0);
    ScalarT ymin = settings->sublist("Mesh").get<ScalarT>("y min",0.0);
    ScalarT zmin = settings->sublist("Mesh").get<ScalarT>("z min",0.0);
    ScalarT xmax = settings->sublist("Mesh").get<ScalarT>("x max",1.0);
    ScalarT ymax = settings->sublist("Mesh").get<ScalarT>("y max",1.0);
    ScalarT zmax = settings->sublist("Mesh").get<ScalarT>("z max",1.0);
    
    std::uniform_real_distribution<ScalarT> xdistribution(xmin,xmax);
    std::uniform_real_distribution<ScalarT> ydistribution(ymin,ymax);
    std::uniform_real_distribution<ScalarT> zdistribution(zmin,zmax);
    
    bool wellspaced = settings->sublist("Mesh").get<bool>("well spaced seeds",true);
    if (wellspaced) {
      // we use a relatively crude algorithm to obtain well-spaced points
      int batch_size = 10;
      int prog = 0;
      Kokkos::View<ScalarT**,HostDevice> cseeds("cand seeds",batch_size,3);
      
      while (prog<num_seeds) {
        // fill in the candidate seeds
        for (int k=0; k<batch_size; k++) {
          ScalarT x = xdistribution(generator);
          cseeds(k,0) = x;
          ScalarT y = ydistribution(generator);
          cseeds(k,1) = y;
          ScalarT z = zdistribution(generator);
          cseeds(k,2) = z;
        }
        int bestpt = 0;
        if (prog > 0) { // for prog = 0, just take the first one
          ScalarT maxdist = 0.0;
          for (int k=0; k<batch_size; k++) {
            ScalarT cmindist = 1.0e200;
            for (int j=0; j<prog; j++) {
              ScalarT dx = cseeds(k,0)-seeds(j,0);
              ScalarT dy = cseeds(k,1)-seeds(j,1);
              ScalarT dz = cseeds(k,2)-seeds(j,2);
              ScalarT cval = xwt*dx*dx + ywt*dy*dy + zwt*dz*dz;
              if (cval < cmindist) {
                cmindist = cval;
              }
            }
            if (cmindist > maxdist) {
              maxdist = cmindist;
              bestpt = k;
            }
          }
        }
        for (int j=0; j<3; j++) {
          seeds_host(prog,j) = cseeds(bestpt,j);
        }
        prog += 1;
      }
    }
    else {
      for (int k=0; k<num_seeds; k++) {
        ScalarT x = xdistribution(generator);
        seeds_host(k,0) = x;
        ScalarT y = ydistribution(generator);
        seeds_host(k,1) = y;
        ScalarT z = zdistribution(generator);
        seeds_host(k,2) = z;
      }
    }
    deep_copy(seeds, seeds_host);
    
  }
  //KokkosTools::print(seeds);
  
  meshimporttimer.stop();
  if (verbosity>5 && comm->getRank() == 0) {
    cout << "microstructure regeneration time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished mesh::generateNewMicrostructure ..." << endl;
    }
  }
  
  return seeds;
}




////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV MeshInterface::getElemNodes(const int & block, const int & elemID) {
  vector<size_t> localIds;
  DRV blocknodes;
  int nnodes = 0;
  if (use_stk_mesh) {
    panzer_stk::workset_utils::getIdsAndVertices(*stk_mesh, block_names[block], localIds, blocknodes);
    nnodes = blocknodes.extent(1);
  }
  else if (use_simple_mesh) {
    //nnodes = simple_mesh->getNumNodes();
    //blocknodes = simple_mesh->getCellNodes({elemID});
  }
  
  DRV cnodes("element nodes",1,nnodes,dimension);
  for (int i=0; i<nnodes; i++) {
    for (int j=0; j<dimension; j++) {
      cnodes(0,i,j) = blocknodes(elemID,i,j);
    }
  }
  return cnodes;
}

DRV MeshInterface::getMyNodes(const size_t & block, vector<size_t> & elemIDs) {
  
  DRV currnodes("current nodes", elemIDs.size(), num_nodes_per_elem, dimension);
  this->getSTKElementVertices(elemIDs, block_names[block], currnodes);
  return currnodes;
}
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<string> MeshInterface::breakupList(const string & list, const string & delimiter) {
  // Script to break delimited list into pieces
  string tmplist = list;
  vector<string> terms;
  size_t pos = 0;
  if (tmplist.find(delimiter) == string::npos) {
    terms.push_back(tmplist);
  }
  else {
    string token;
    while ((pos = tmplist.find(delimiter)) != string::npos) {
      token = tmplist.substr(0, pos);
      terms.push_back(token);
      tmplist.erase(0, pos + delimiter.length());
    }
    terms.push_back(tmplist);
  }
  return terms;
}

/////////////////////////////////////////////////////////////////////////////
// Read in discretized data from an exodus mesh
/////////////////////////////////////////////////////////////////////////////

void MeshInterface::readExodusData() {
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Starting mesh::readExodusData ..." << endl;
    }
  }
  
  string exofile;
  string fname;
  
  exofile = settings->sublist("Mesh").get<std::string>("mesh file","mesh.exo");
  
  if (comm->getSize() > 1) {
    std::stringstream ssProc, ssPID;
    ssProc << comm->getSize();
    ssPID << comm->getRank();
    string strProc = ssProc.str();
    string strPID = ssPID.str();
    // this section may need tweaking if the input exodus mesh is
    // spread across 10's, 100's, or 1000's (etc) of processors
    //if (Comm->MyPID() < 10)
    if (false)
      fname = exofile + "." + strProc + ".0" + strPID;
    else
      fname = exofile + "." + strProc + "." + strPID;
  }
  else {
    fname = exofile;
  }
  
  // open exodus file
  int CPU_word_size, IO_word_size, exoid, exo_error;
  int num_dim, num_nods, num_el, num_el_blk, num_ns, num_ss;
  char title[MAX_STR_LENGTH+1];
  float exo_version;
  CPU_word_size = sizeof(ScalarT);
  IO_word_size = 0;
  exoid = ex_open(fname.c_str(), EX_READ, &CPU_word_size,&IO_word_size,
                  &exo_version);
  exo_error = ex_get_init(exoid, title, &num_dim, &num_nods, &num_el,
                          &num_el_blk, &num_ns, &num_ss);
  
  if (exo_error>0) {
    // need some debug statement
  }
  int id = 1; // only one blkid
  int step = 1; // only one time step (for now)
  ex_block eblock;
  eblock.id = id;
  eblock.type = EX_ELEM_BLOCK;
  
  exo_error = ex_get_block_param(exoid, &eblock);
  
  int num_el_in_blk = eblock.num_entry;
  //int num_node_per_el = eblock.num_nodes_per_entry;
  
  
  // get elem vars
  if (settings->sublist("Mesh").get<bool>("have element data", false)) {
    int num_elem_vars = 0;
    int var_ind;
    numResponses = 1;
    //exo_error = ex_get_var_param(exoid, "e", &num_elem_vars); // TMW: this is depracated
    exo_error = ex_get_variable_param(exoid, EX_ELEM_BLOCK, &num_elem_vars); // TMW: this is depracated
    // This turns off this feature
    for (int i=0; i<num_elem_vars; i++) {
      char varname[MAX_STR_LENGTH+1];
      ScalarT *var_vals = new ScalarT[num_el_in_blk];
      var_ind = i+1;
      exo_error = ex_get_variable_name(exoid, EX_ELEM_BLOCK, var_ind, varname);
      string vname(varname);
      efield_names.push_back(vname);
      size_t found = vname.find("Val");
      if (found != std::string::npos) {
        vector<string> results;
        std::stringstream sns, snr;
        int nr;
        results = this->breakupList(vname,"_");
        //boost::split(results, vname, [](char u){return u == '_';});
        snr << results[3];
        snr >> nr;
        numResponses = std::max(numResponses,nr);
      }
      efield_vals.push_back(vector<ScalarT>(num_el_in_blk));
      exo_error = ex_get_var(exoid,step,EX_ELEM_BLOCK,var_ind,id,num_el_in_blk,var_vals);
      for (int j=0; j<num_el_in_blk; j++) {
        efield_vals[i][j] = var_vals[j];
      }
      delete [] var_vals;
    }
  }
  
  /*
  // assign nodal vars to meas multivector
  if (settings->sublist("Mesh").get<bool>("have nodal data", false)) {
    int *connect = new int[num_el_in_blk*num_node_per_el];
    int edgeconn, faceconn;
    //exo_error = ex_get_elem_conn(exoid, id, connect);
    exo_error = ex_get_conn(exoid, EX_ELEM_BLOCK, id, connect, &edgeconn, &faceconn);
    
    // get nodal vars
    int num_node_vars = 0;
    int var_ind;
    //exo_error = ex_get_variable_param(exoid, EX_NODAL, &num_node_vars);
    // This turns off this feature
    for (int i=0; i<num_node_vars; i++) {
      char varname[MAX_STR_LENGTH+1];
      ScalarT *var_vals = new ScalarT[num_nods];
      var_ind = i+1;
      exo_error = ex_get_variable_name(exoid, EX_NODAL, var_ind, varname);
      string vname(varname);
      nfield_names.push_back(vname);
      nfield_vals.push_back(vector<ScalarT>(num_nods));
      exo_error = ex_get_var(exoid,step,EX_NODAL,var_ind,0,num_nods,var_vals);
      for (int j=0; j<num_nods; j++) {
        nfield_vals[i][j] = var_vals[j];
      }
      delete [] var_vals;
    }
    
    
    meas = Teuchos::rcp(new Tpetra::MultiVector<ScalarT,LO,GO,SolverNode>(LA_overlapped_map,1)); // empty solution
    size_t b = 0;
    //meas->sync<HostDevice>();
    auto meas_kv = meas->getLocalView<HostDevice>();
    
    //meas.modify_host();
    int index, dindex;
    
    auto dev_offsets = groups[b][0]->wkset->offsets;
    auto offsets = Kokkos::create_mirror_view(dev_offsets);
    Kokkos::deep_copy(offsets,dev_offsets);
    
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      //cindex = groups[block][grp]->index;
      auto LIDs = groups[block][grp]->LIDs_host;
      auto nDOF = groups[block][grp]->group_data->numDOF_host;
      
      for (int n=0; n<nDOF(0); n++) {
        //Kokkos::View<GO**,HostDevice> GIDs = assembler->groups[block][grp]->GIDs;
        for (size_t p=0; p<groups[block][grp]->numElem; p++) {
          for( int i=0; i<nDOF(n); i++ ) {
            index = LIDs(p,offsets(n,i));//cindex(p,n,i);//LA_overlapped_map->getLocalElement(GIDs(p,curroffsets[n][i]));
            dindex = connect[e*num_node_per_el + i] - 1;
            meas_kv(index,0) = nfield_vals[n][dindex];
            //(*meas)[0][index] = nfield_vals[n][dindex];
          }
        }
      }
    }
    //meas.sync<>();
    delete [] connect;
    
  }
   */
  exo_error = ex_close(exoid);
  
  if (debug_level > 0) {
    if (comm->getRank() == 0) {
      cout << "**** Finished mesh::readExodusData" << endl;
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we might be able to get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

void MeshInterface::purgeMesh() {
  stk_mesh = Teuchos::null;
  mesh_factory = Teuchos::null;
  simple_mesh = Teuchos::null;
}

void MeshInterface::purgeMemory() {
  nfield_vals.clear();
  efield_vals.clear();
  meas = Teuchos::null;
}

////////////////////////////////////////////////////////////////////////////////
// Access function (mostly) for the stk mesh
////////////////////////////////////////////////////////////////////////////////
    
void MeshInterface::setSolutionFieldData(string var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT**,HostDevice> soln) {
  if (use_stk_mesh) {
    stk_mesh->setSolutionFieldData(var, blockID, myElements, soln);
  }
}

void MeshInterface::setCellFieldData(string var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT*,HostDevice> soln) {
  if (use_stk_mesh) {
    stk_mesh->setCellFieldData(var, blockID, myElements, soln);
  }
}

void MeshInterface::setOptimizationSolutionFieldData(string & var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT**,HostDevice> soln) {
  if (use_stk_mesh) {
    stk_optimization_mesh->setSolutionFieldData(var, blockID, myElements, soln);
  }
}

void MeshInterface::setOptimizationCellFieldData(string & var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT*,HostDevice> soln) {
  if (use_stk_mesh) {
    stk_optimization_mesh->setCellFieldData(var, blockID, myElements, soln);
  }
}

void MeshInterface::writeToExodus(const double & currenttime) {
  if (use_stk_mesh) {
    stk_mesh->writeToExodus(currenttime);
  }
}

void MeshInterface::writeToExodus(const string & filename) {
  if (use_stk_mesh) {
    stk_mesh->writeToExodus(filename);
  }
}
    
void MeshInterface::writeToOptimizationExodus(const double & currenttime) {
  if (use_stk_mesh) {
    stk_optimization_mesh->writeToExodus(currenttime);
  }
}

void MeshInterface::writeToOptimizationExodus(const string & filename) {
  if (use_stk_mesh) {
    stk_optimization_mesh->writeToExodus(filename);
  }
}
    
vector<string> MeshInterface::getBlockNames() {
  return block_names;
}

vector<string> MeshInterface::getSideNames() {
  return side_names;
}
    
vector<string> MeshInterface::getNodeNames() {
  return node_names;
}

int MeshInterface::getDimension() {
  return dimension;
}
    
topo_RCP MeshInterface::getCellTopology(string & blockID) {
  topo_RCP currtopo;
  for (size_t blk=0; blk<block_names.size(); ++blk) {
    if (block_names[blk] == blockID) {
      currtopo = cell_topo[blk];
    }
  }
  return currtopo;
}
    
Teuchos::RCP<panzer::ConnManager> MeshInterface::getSTKConnManager() {
  Teuchos::RCP<panzer::ConnManager> conn;
  if (use_stk_mesh) {
    conn = Teuchos::rcp(new panzer_stk::STKConnManager(stk_mesh));
  }
  return conn;
}

void MeshInterface::setSTKMesh(Teuchos::RCP<panzer_stk::STK_Interface> & new_mesh) {
  stk_mesh = new_mesh;
  stk_mesh->getElementBlockNames(block_names);
  stk_mesh->getSidesetNames(side_names);
  stk_mesh->getNodesetNames(node_names);
}

  
vector<stk::mesh::Entity> MeshInterface::getMySTKElements() {
  vector<stk::mesh::Entity> stk_meshElems;
  if (use_stk_mesh) {
    stk_mesh->getMyElements(stk_meshElems);
  }
  return stk_meshElems;
}
  
vector<stk::mesh::Entity> MeshInterface::getMySTKElements(string & blockID) {
  vector<stk::mesh::Entity> stk_meshElems;
  if (use_stk_mesh) {
    stk_mesh->getMyElements(blockID, stk_meshElems);
  }
  return stk_meshElems;
}
  
void MeshInterface::getSTKNodeIdsForElement(stk::mesh::Entity & stk_meshElem, vector<stk::mesh::EntityId> & stk_nodeids) {
  if (use_stk_mesh) {
    stk_mesh->getNodeIdsForElement(stk_meshElem, stk_nodeids);
  }
}

vector<stk::mesh::Entity> MeshInterface::getMySTKSides(string & sideName, string & blockname) {
  vector<stk::mesh::Entity> sideEntities;
  if (use_stk_mesh) {
    stk_mesh->getMySides(sideName, blockname, sideEntities);
  }
  return sideEntities;
}

vector<stk::mesh::Entity> MeshInterface::getMySTKNodes(string & nodeName, string & blockID) {
  vector<stk::mesh::Entity> nodeEntities;
  if (use_stk_mesh) {
    stk_mesh->getMyNodes(nodeName, blockID, nodeEntities);
  }
  return nodeEntities;
}

void MeshInterface::getSTKSideElements(string & blockname, vector<stk::mesh::Entity> & sideEntities, 
                                       vector<size_t> & local_side_Ids, vector<stk::mesh::Entity> & side_output) {
  if (use_stk_mesh) {
    panzer_stk::workset_utils::getSideElements(*stk_mesh, blockname, sideEntities, local_side_Ids, side_output);
  }
}

void MeshInterface::getSTKElementVertices(vector<stk::mesh::Entity> & side_output, string & blockname, DRV & sidenodes) {
  if (use_stk_mesh) {
    stk_mesh->getElementVertices(side_output, blockname, sidenodes);
  }
}
    
LO MeshInterface::getSTKElementLocalId(stk::mesh::Entity & elem) {
  LO id = 0;
  if (use_stk_mesh) {
    id = stk_mesh->elementLocalId(elem);
  }
  return id;
}

void MeshInterface::getSTKElementVertices(vector<size_t> & local_grp, string & blockname, DRV & currnodes) {
  if (use_stk_mesh) {
    stk_mesh->getElementVertices(local_grp, blockname, currnodes);
  } else {
    currnodes = simple_mesh->getCellNodes(local_grp);
  }
}

void MeshInterface::getSTKNodeElements(string & blockname, vector<stk::mesh::Entity> & nodeEntities, 
                                       vector<size_t> & local_node_Ids, vector<stk::mesh::Entity> & side_output) {
  if (use_stk_mesh) {
    panzer_stk::workset_utils::getNodeElements(*stk_mesh, blockname, nodeEntities, local_node_Ids, side_output);
  }
}
