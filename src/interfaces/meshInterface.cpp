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

#include "meshInterface.hpp"
#include "exodusII.h"

#include <boost/algorithm/string.hpp>

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

meshInterface::meshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                             const Teuchos::RCP<MpiComm> & Commptr_) :
settings(settings_), Commptr(Commptr_) {
  
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  milo_debug_level = settings->get<int>("debug level",0);
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh interface constructor ..." << endl;
    }
  }
  
  shape = settings->sublist("Mesh").get<string>("shape","none");
  if (shape == "none") { // new keywords, but allowing BWDS compat.
    shape = settings->sublist("Mesh").get<string>("element type","quad");
  }
  spaceDim = settings->sublist("Mesh").get<int>("dim",0);
  if (spaceDim == 0) {
    spaceDim = settings->sublist("Mesh").get<int>("dimension",2);
  }
  verbosity = settings->get<int>("verbosity",0);
  
  have_mesh_data = false;
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  have_rotations = false;
  have_rotation_phi = false;
  have_multiple_data_files = false;
  mesh_data_file_tag = "none";
  mesh_data_pts_tag = "mesh_data_pts";
  number_mesh_data_files = 1;
  
  mesh_data_tag = settings->sublist("Mesh").get<string>("data file","none");
  if (mesh_data_tag != "none") {
    mesh_data_pts_tag = settings->sublist("Mesh").get<string>("data points file","mesh_data_pts");
    
    have_mesh_data = true;
    have_rotation_phi = settings->sublist("Mesh").get<bool>("have mesh data phi",false);
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",false);
    have_multiple_data_files = settings->sublist("Mesh").get<bool>("have multiple mesh data files",false);
    number_mesh_data_files = settings->sublist("Mesh").get<int>("number mesh data files",1);
    
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
  
  if (spaceDim == 1) {
    cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );// lin. cell topology on the interior
    sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Node>() );          // line cell topology on the boundary
  }
  if (spaceDim == 2) {
    if (shape == "quad") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );          // line cell topology on the boundary
    }
    if (shape == "tri") {
      cTopo = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() );// lin. cell topology on the interior
      sTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );          // line cell topology on the boundary
    }
  }
  if (spaceDim == 3) {
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
  numNodesPerElem = cTopo.getNodeCount();
  settings->sublist("Mesh").set("numNodesPerElem",numNodesPerElem,"number of nodes per element");
  sideDim = 0;
  if (spaceDim > 1) {
    sTopo.getDimension();
  }
  settings->sublist("Mesh").set("sideDim",sideDim,"dimension of the sides of each element");
  numSides = cTopo.getSideCount();
  numFaces = cTopo.getFaceCount();
  if (spaceDim == 1)
    settings->sublist("Mesh").set("numSidesPerElem",2,"number of sides per element");
  if (spaceDim == 2)
    settings->sublist("Mesh").set("numSidesPerElem",numSides,"number of sides per element");
  if (spaceDim == 3)
    settings->sublist("Mesh").set("numSidesPerElem",numFaces,"number of sides per element");
  
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
    if (spaceDim > 1) {
      pl->set("X Procs", settings->sublist("Mesh").get("Xprocs",Commptr->getSize()));
      pl->set("Y Blocks",settings->sublist("Mesh").get("Yblocks",1));
      pl->set("Y Elements",settings->sublist("Mesh").get("NY",20));
      pl->set("Y0",settings->sublist("Mesh").get("ymin",0.0));
      pl->set("Yf",settings->sublist("Mesh").get("ymax",1.0));
      pl->set("Y Procs", settings->sublist("Mesh").get("Yprocs",1));
    }
    if (spaceDim > 2) {
      pl->set("Z Blocks",settings->sublist("Mesh").get("Zblocks",1));
      pl->set("Z Elements",settings->sublist("Mesh").get("NZ",20));
      pl->set("Z0",settings->sublist("Mesh").get("zmin",0.0));
      pl->set("Zf",settings->sublist("Mesh").get("zmax",1.0));
      pl->set("Z Procs", settings->sublist("Mesh").get("Zprocs",1));
    }
    if (spaceDim == 1) {
      mesh_factory = Teuchos::rcp(new panzer_stk::LineMeshFactory());
    }
    else if (spaceDim == 2) {
      if (shape == "quad") {
        mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory());
      }
      if (shape == "tri") {
        mesh_factory = Teuchos::rcp(new panzer_stk::SquareTriMeshFactory());
      }
    }
    else if (spaceDim == 3) {
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
  mesh = mesh_factory->buildUncommitedMesh(*(Commptr->getRawMpiComm()));
  
  // create a mesh for an optmization movie
  if (settings->sublist("Postprocess").get("create optimization movie",false)) {
    optimization_mesh = mesh_factory->buildUncommitedMesh(*(Commptr->getRawMpiComm()));
  }
  
  vector<string> eBlocks;
  mesh->getElementBlockNames(eBlocks);

  for (size_t b=0; b<eBlocks.size(); b++) {
    cellTopo.push_back(mesh->getCellTopology(eBlocks[b]));
  }
  
  for (size_t b=0; b<eBlocks.size(); b++) {
    topo_RCP cellTopo = mesh->getCellTopology(eBlocks[b]);
    string shape = cellTopo->getName();
    if (spaceDim == 1) {
      // nothing to do here
    }
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
      if (shape == "Triangle_3") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() )));
      }
      if (shape == "Tetrahedron_4") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() )));
      }
    }
    
  }

  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh interface constructor" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

meshInterface::meshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                             const Teuchos::RCP<MpiComm> & Commptr_,
                             Teuchos::RCP<panzer_stk::STK_MeshFactory> & mesh_factory_,
                             Teuchos::RCP<panzer_stk::STK_Interface> & mesh_) :
mesh_factory(mesh_factory_), mesh(mesh_), settings(settings_), Commptr(Commptr_) {
  
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  milo_debug_level = settings->get<int>("debug level",0);
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh interface constructor ..." << endl;
    }
  }
  
  shape = settings->sublist("Mesh").get<string>("shape","none");
  if (shape == "none") { // new keywords, but allowing BWDS compat.
    shape = settings->sublist("Mesh").get<string>("element type","quad");
  }
  spaceDim = settings->sublist("Mesh").get<int>("dim",0);
  if (spaceDim == 0) {
    spaceDim = settings->sublist("Mesh").get<int>("dimension",2);
  }
  
  have_mesh_data = false;
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  have_rotations = false;
  have_rotation_phi = false;
  have_multiple_data_files = false;
  mesh_data_file_tag = "none";
  mesh_data_pts_tag = "mesh_data_pts";
  number_mesh_data_files = 1;
  
  mesh_data_tag = settings->sublist("Mesh").get<string>("data file","none");
  if (mesh_data_tag != "none") {
    mesh_data_pts_tag = settings->sublist("Mesh").get<string>("data points file","mesh_data_pts");
    
    have_mesh_data = true;
    have_rotation_phi = settings->sublist("Mesh").get<bool>("have mesh data phi",false);
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",true);
    have_multiple_data_files = settings->sublist("Mesh").get<bool>("have multiple mesh data files",false);
    number_mesh_data_files = settings->sublist("Mesh").get<int>("number mesh data files",1);
    
  }
  
  meshmod_xvar = settings->sublist("Solver").get<int>("solution for x-mesh mod",-1);
  meshmod_yvar = settings->sublist("Solver").get<int>("solution for y-mesh mod",-1);
  meshmod_zvar = settings->sublist("Solver").get<int>("solution for z-mesh mod",-1);
  meshmod_TOL = settings->sublist("Solver").get<ScalarT>("solution based mesh mod TOL",1.0);
  meshmod_usesmoother = settings->sublist("Solver").get<bool>("solution based mesh mod smoother",false);
  meshmod_center = settings->sublist("Solver").get<ScalarT>("solution based mesh mod param",0.1);
  meshmod_layer_size = settings->sublist("Solver").get<ScalarT>("solution based mesh mod layer thickness",0.1);
  
  vector<string> eBlocks;
  mesh->getElementBlockNames(eBlocks);
  
  for (size_t b=0; b<eBlocks.size(); b++) {
    cellTopo.push_back(mesh->getCellTopology(eBlocks[b]));
  }
  
  for (size_t b=0; b<eBlocks.size(); b++) {
    topo_RCP cellTopo = mesh->getCellTopology(eBlocks[b]);
    string shape = cellTopo->getName();
    if (spaceDim == 1) {
      // nothing to do here?
    }
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
      if (shape == "Triangle_3") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() )));
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() )));
      }
      if (shape == "Tetrahedron_4") {
        sideTopo.push_back(Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() )));
      }
    }
    
  }
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh interface constructor" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void meshInterface::finalize(Teuchos::RCP<physics> & phys) {
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh interface finalize ..." << endl;
    }
  }
  ////////////////////////////////////////////////////////////////////////////////
  // Add fields to the mesh
  ////////////////////////////////////////////////////////////////////////////////
  
  vector<string> eBlocks;
  mesh->getElementBlockNames(eBlocks);
  
  for(std::size_t i=0;i<eBlocks.size();i++) {
    
    std::vector<string> varlist = phys->varlist[i];
    std::vector<string> vartypes = phys->types[i];
    
    for (size_t j=0; j<varlist.size(); j++) {
      if (vartypes[j] == "HGRAD") {
        mesh->addSolutionField(varlist[j], eBlocks[i]);
      }
      else if (vartypes[j] == "HVOL") { // PW constant
        mesh->addCellField(varlist[j], eBlocks[i]);
      }
      else if (vartypes[j] == "HFACE") { // hybridized variable
        mesh->addCellField(varlist[j], eBlocks[i]);
      }
      else if (vartypes[j] == "HDIV" || vartypes[j] == "HCURL") { // HDIV or HCURL
        mesh->addSolutionField(varlist[j]+"x", eBlocks[i]);
        mesh->addSolutionField(varlist[j]+"y", eBlocks[i]);
        mesh->addSolutionField(varlist[j]+"z", eBlocks[i]);
        mesh->addCellField(varlist[j]+"x", eBlocks[i]);
        mesh->addCellField(varlist[j]+"y", eBlocks[i]);
        mesh->addCellField(varlist[j]+"z", eBlocks[i]);
      }
    }
    
    if (phys->have_aux) {
      std::vector<string> varlist = phys->aux_varlist[i];
      std::vector<string> vartypes = phys->aux_types[i];
      for (size_t j=0; j<varlist.size(); j++) {
        if (vartypes[j] == "HGRAD") {
          mesh->addSolutionField(varlist[j], eBlocks[i]);
        }
        else if (vartypes[j] == "HVOL") { // PW constant
          mesh->addCellField(varlist[j], eBlocks[i]);
        }
        else if (vartypes[j] == "HFACE") { // hybridized variable
          mesh->addCellField(varlist[j], eBlocks[i]);
        }
        else if (vartypes[j] == "HDIV" || vartypes[j] == "HCURL") { // HDIV or HCURL
          mesh->addSolutionField(varlist[j]+"x", eBlocks[i]);
          mesh->addSolutionField(varlist[j]+"y", eBlocks[i]);
          mesh->addSolutionField(varlist[j]+"z", eBlocks[i]);
          mesh->addCellField(varlist[j]+"x", eBlocks[i]);
          mesh->addCellField(varlist[j]+"y", eBlocks[i]);
          mesh->addCellField(varlist[j]+"z", eBlocks[i]);
        }
      }
    }
    
    mesh->addSolutionField("dispx", eBlocks[i]);
    mesh->addSolutionField("dispy", eBlocks[i]);
    mesh->addSolutionField("dispz", eBlocks[i]);
    
    
    Teuchos::ParameterList efields;
    if (settings->sublist("Physics").isSublist(eBlocks[i])) {
      efields = settings->sublist("Physics").sublist(eBlocks[i]).sublist("Extra fields");
    }
    else {
      efields = settings->sublist("Physics").sublist("Extra fields");
    }
    Teuchos::ParameterList::ConstIterator ef_itr = efields.begin();
    while (ef_itr != efields.end()) {
      mesh->addSolutionField(ef_itr->first, eBlocks[i]);
      ef_itr++;
    }
    
    Teuchos::ParameterList ecfields;
    if (settings->sublist("Physics").isSublist(eBlocks[i])) {
      ecfields = settings->sublist("Physics").sublist(eBlocks[i]).sublist("Extra cell fields");
    }
    else {
      ecfields = settings->sublist("Physics").sublist("Extra cell fields");
    }
    Teuchos::ParameterList::ConstIterator ecf_itr = ecfields.begin();
    while (ecf_itr != ecfields.end()) {
      mesh->addCellField(ecf_itr->first, eBlocks[i]);
      if (settings->isSublist("Subgrid")) {
        string sgfn = "subgrid_mean_" + ecf_itr->first;
        mesh->addCellField(sgfn, eBlocks[i]);
      }
      ecf_itr++;
    }
    /*
    std::vector<string> extrafields = phys->getExtraFieldNames(i);
    for (size_t j=0; j<extrafields.size(); j++) {
      mesh->addSolutionField(extrafields[j], eBlocks[i]);
    }
    
    std::vector<string> extracellfields = phys->getExtraCellFieldNames(i);
    for (size_t j=0; j<extracellfields.size(); j++) {
      mesh->addCellField(extracellfields[j], eBlocks[i]);
    }
    if (settings->isSublist("Subgrid")) {
      for (size_t j=0; j<extracellfields.size(); j++) {
        string sgfn = "subgrid_mean_" + extracellfields[j];
        mesh->addCellField(sgfn, eBlocks[i]);
      }
    }
    */
    if (have_mesh_data || compute_mesh_data) {
      mesh->addCellField("mesh_data_seed", eBlocks[i]);
      mesh->addCellField("mesh_data", eBlocks[i]);
    }
    
    mesh->addCellField("subgrid model", eBlocks[i]);
    
    mesh->addCellField("cell number", eBlocks[i]);
    
    if (settings->isSublist("Parameters")) {
      Teuchos::ParameterList parameters = settings->sublist("Parameters");
      Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
      while (pl_itr != parameters.end()) {
        Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
        if (newparam.get<string>("usage") == "discretized") {
          if (newparam.get<string>("type") == "HGRAD") {
            mesh->addSolutionField(pl_itr->first, eBlocks[i]);
          }
          else if (newparam.get<string>("type") == "HVOL") {
            mesh->addCellField(pl_itr->first, eBlocks[i]);
          }
          else if (newparam.get<string>("type") == "HDIV" || newparam.get<string>("type") == "HCURL") {
            mesh->addCellField(pl_itr->first+"_x", eBlocks[i]);
            mesh->addCellField(pl_itr->first+"_y", eBlocks[i]);
            mesh->addCellField(pl_itr->first+"_z", eBlocks[i]);
          }
        }
        pl_itr++;
      }
    }
  }
  
  //mesh_factory->completeMeshConstruction(*mesh,Commptr->Comm());
  mesh_factory->completeMeshConstruction(*mesh,*(Commptr->getRawMpiComm()));
  
  //int refinements = settings->sublist("Mesh").get<int>("refinements",0);
  //if (refinements>0) {
  //  mesh->refineMesh(refinements, true);
  //}
  
  //this->perturbMesh();
  
  if (verbosity>1) {
    if (Commptr->getRank() == 0) {
      mesh->printMetaData(std::cout);
    }
  }
  
  if (settings->sublist("Postprocess").get("create optimization movie",false)) {
    vector<string> eBlocks;
    mesh->getElementBlockNames(eBlocks);
    
    for(std::size_t i=0;i<eBlocks.size();i++) {
      
      if (settings->isSublist("Parameters")) {
        Teuchos::ParameterList parameters = settings->sublist("Parameters");
        Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
        while (pl_itr != parameters.end()) {
          Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
          if (newparam.get<string>("usage") == "discretized") {
            if (newparam.get<string>("type") == "HGRAD") {
              optimization_mesh->addSolutionField(pl_itr->first, eBlocks[i]);
            }
            else if (newparam.get<string>("type") == "HVOL") {
              optimization_mesh->addCellField(pl_itr->first, eBlocks[i]);
            }
            else if (newparam.get<string>("type") == "HDIV" || newparam.get<string>("type") == "HCURL") {
              optimization_mesh->addCellField(pl_itr->first+"_x", eBlocks[i]);
              optimization_mesh->addCellField(pl_itr->first+"_y", eBlocks[i]);
              optimization_mesh->addCellField(pl_itr->first+"_z", eBlocks[i]);
            }
          }
          pl_itr++;
        }
      }
    }
    
    mesh_factory->completeMeshConstruction(*optimization_mesh,*(Commptr->getRawMpiComm()));
    if (verbosity>1) {
      optimization_mesh->printMetaData(std::cout);
    }
  }
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh interface finalize" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV meshInterface::perturbMesh(const int & b, DRV & blocknodes) {
  
  ////////////////////////////////////////////////////////////////////////////////
  // Perturb the mesh (if requested)
  ////////////////////////////////////////////////////////////////////////////////
  
  vector<string> eBlocks;
  mesh->getElementBlockNames(eBlocks);
  
  //for (size_t b=0; b<eBlocks.size(); b++) {
    //vector<size_t> localIds;
    //DRV blocknodes;
    //panzer_stk::workset_utils::getIdsAndVertices(*mesh, eBlocks[b], localIds, blocknodes);
    int numNodesPerElem = blocknodes.extent(1);
    DRV blocknodePert("blocknodePert",blocknodes.extent(0),numNodesPerElem,spaceDim);
    
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
        //    for (int s=0; s<spaceDim; s++) {
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
      //    for (int s=0; s<spaceDim; s++) {
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

void meshInterface::setMeshData(vector<vector<Teuchos::RCP<cell> > > & cells) {
  if (have_mesh_data) {
    this->importMeshData(cells);
  }
  else if (compute_mesh_data) {
    this->computeMeshData(cells);
  }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void meshInterface::importMeshData(vector<vector<Teuchos::RCP<cell> > > & cells) {
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh::importMeshData ..." << endl;
    }
  }
  
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  int numdata = 1;
  if (have_rotations) {
    numdata = 9;
  }
  else if (have_rotation_phi) {
    numdata = 3;
  }
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      int numElem = cells[b][e]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      cells[b][e]->cell_data = cell_data;
      cells[b][e]->cell_data_distance = vector<ScalarT>(numElem);
      cells[b][e]->cell_data_seed = vector<size_t>(numElem);
      cells[b][e]->cell_data_seedindex = vector<size_t>(numElem);
    }
  }
  
  for (int p=0; p<number_mesh_data_files; p++) {
        
    Teuchos::RCP<data> mesh_data;
    
    string mesh_data_pts_file;
    string mesh_data_file;
    
    if (have_multiple_data_files) {
      std::stringstream ss;
      ss << p+1;
      mesh_data_pts_file = mesh_data_pts_tag + "." + ss.str() + ".dat";
      mesh_data_file = mesh_data_tag + "." + ss.str() + ".dat";
    }
    else {
      mesh_data_pts_file = mesh_data_pts_tag + ".dat";
      mesh_data_file = mesh_data_tag + ".dat";
    }
    
    bool have_grid_data = settings->sublist("Mesh").get<bool>("data on grid",false);
    if (have_grid_data) {
      int Nx = settings->sublist("Mesh").get<int>("data grid Nx",0);
      int Ny = settings->sublist("Mesh").get<int>("data grid Ny",0);
      int Nz = settings->sublist("Mesh").get<int>("data grid Nz",0);
      mesh_data = Teuchos::rcp(new data("mesh data", spaceDim, mesh_data_pts_file,
                                        mesh_data_file, false, Nx, Ny, Nz));
      
      for (size_t b=0; b<cells.size(); b++) {
        for (size_t e=0; e<cells[b].size(); e++) {
          DRV nodes = cells[b][e]->nodes;
          int numElem = cells[b][e]->numElem;
          
          for (int c=0; c<numElem; c++) {
            Kokkos::View<ScalarT[1][3],HostDevice> center("center");
            for (size_type i=0; i<nodes.extent(1); i++) {
              for (int j=0; j<spaceDim; j++) {
                center(0,j) += nodes(c,i,j)/(ScalarT)nodes.extent(1);
              }
            }
            ScalarT distance = 0.0;
            
            int cnode = mesh_data->findClosestGridNode(center(0,0), center(0,1), center(0,2), distance);
            bool iscloser = true;
            if (p>0){
              if (cells[b][e]->cell_data_distance[c] < distance) {
                iscloser = false;
              }
            }
            if (iscloser) {
              Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getdata(cnode);
              for (size_type i=0; i<cdata.extent(1); i++) {
                cells[b][e]->cell_data(c,i) = cdata(0,i);
              }
              cells[b][e]->cellData->have_extra_data = true;
              if (have_rotations)
                cells[b][e]->cellData->have_cell_rotation = true;
              if (have_rotation_phi)
                cells[b][e]->cellData->have_cell_phi = true;
              
              cells[b][e]->cell_data_seed[c] = cnode;
              cells[b][e]->cell_data_seedindex[c] = cnode % 50;
              cells[b][e]->cell_data_distance[c] = distance;
            }
          }
        }
      }
    }
    else {
      mesh_data = Teuchos::rcp(new data("mesh data", spaceDim, mesh_data_pts_file,
                                        mesh_data_file, false));
      
      for (size_t b=0; b<cells.size(); b++) {
        for (size_t e=0; e<cells[b].size(); e++) {
          DRV nodes = cells[b][e]->nodes;
          int numElem = cells[b][e]->numElem;
          
          Kokkos::View<ScalarT**, AssemblyDevice> center("center",numElem,spaceDim);
          for (int c=0; c<numElem; c++) {
            for (size_t i=0; i<nodes.extent(1); i++) {
              for (int j=0; j<spaceDim; j++) {
                center(c,j) += nodes(c,i,j)/(ScalarT)nodes.extent(1);
              }
            }
	  }

	  Kokkos::View<ScalarT*, AssemblyDevice> distance("distance",numElem);
	  Kokkos::View<int*, AssemblyDevice> cnode("cnode",numElem);  
          mesh_data->findClosestNode(center,cnode,distance);

          bool iscloser = true;
          for (int c=0; c<numElem; c++) {
            if (p>0){
              if (cells[b][e]->cell_data_distance[c] < distance(c)) {
                iscloser = false;
              }
            }
            if (iscloser) {
              Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getdata(cnode(c));
              for (size_t i=0; i<cdata.extent(1); i++) {
                cells[b][e]->cell_data(c,i) = cdata(0,i);
              }
              cells[b][e]->cellData->have_extra_data = true;
              if (have_rotations)
                cells[b][e]->cellData->have_cell_rotation = true;
              if (have_rotation_phi)
                cells[b][e]->cellData->have_cell_phi = true;
            
              cells[b][e]->cell_data_seed[c] = cnode(c);
              cells[b][e]->cell_data_seedindex[c] = cnode(c) % 50;
              cells[b][e]->cell_data_distance[c] = distance(c);
            }
          }
        }
      }
    }
    
  }
  
  meshimporttimer.stop();
  if (verbosity>5 && Commptr->getRank() == 0) {
    cout << "mesh data import time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh::meshDataImport" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void meshInterface::computeMeshData(vector<vector<Teuchos::RCP<cell> > > & cells) {
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh::computeMeshData ..." << endl;
    }
  }
  Teuchos::Time meshimporttimer("mesh import", false);
  meshimporttimer.start();
  
  have_rotations = true;
  have_rotation_phi = false;
  
  Kokkos::View<ScalarT**,HostDevice> seeds;
  int randSeed = settings->sublist("Mesh").get<int>("random seed",1234);
  randomSeeds.push_back(randSeed);
  std::default_random_engine generator(randSeed);
  numSeeds = 0;
  
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
    numSeeds = numxSeeds*numySeeds*numzSeeds;
    seeds = Kokkos::View<ScalarT**,HostDevice>("seeds",numSeeds,3);
    int prog = 0;
    for (int i=0; i<numxSeeds; i++) {
      for (int j=0; j<numySeeds; j++) {
        for (int k=0; k<numzSeeds; k++) {
          ScalarT xp = pdistribution(generator);
          ScalarT yp = pdistribution(generator);
          ScalarT zp = pdistribution(generator);
          seeds(prog,0) = xseeds(i) + xp*dx;
          seeds(prog,1) = yseeds(j) + yp*dy;
          seeds(prog,2) = zseeds(k) + zp*dz;
          prog += 1;
        }
      }
    }
  }
  else {
    numSeeds = settings->sublist("Mesh").get<int>("number of seeds",1000);
    seeds = Kokkos::View<ScalarT**,HostDevice>("seeds",numSeeds,3);
    
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
    
    
    // we use a relatively crude algorithm to obtain well-spaced points
    int batch_size = 10;
    int prog = 0;
    Kokkos::View<ScalarT**,HostDevice> cseeds("cand seeds",batch_size,3);
    
    while (prog<numSeeds) {
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
        ScalarT mindist = 1.0e6;
        for (int k=0; k<batch_size; k++) {
          ScalarT cmindist = 1.0e6;
          for (int j=0; j<prog; j++) {
            ScalarT dx = cseeds(k,0)-seeds(j,0);
            ScalarT dy = cseeds(k,1)-seeds(j,1);
            ScalarT dz = cseeds(k,2)-seeds(j,2);
            ScalarT cval = sqrt(xwt*dx*dx + ywt*dy*dy + zwt*dz*dz);
            if (cval < cmindist) {
              cmindist = cval;
            }
          }
          if (cmindist<mindist) {
            mindist = cmindist;
            bestpt = k;
          }
        }
      }
      for (int j=0; j<3; j++) {
        seeds(prog,j) = cseeds(bestpt,j);
      }
      prog += 1;
    }
  }
  //KokkosTools::print(seeds);
  
  std::uniform_int_distribution<int> idistribution(0,50);
  Kokkos::View<int*,HostDevice> seedIndex("seed index",numSeeds);
  for (int i=0; i<numSeeds; i++) {
    int ci = idistribution(generator);
    seedIndex(i) = ci;
  }
  
  //KokkosTools::print(seedIndex);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////
  
  int numdata = 9;
  
  std::normal_distribution<ScalarT> ndistribution(0.0,1.0);
  Kokkos::View<ScalarT**,HostDevice> rotation_data("cell_data",numSeeds,numdata);
  for (int k=0; k<numSeeds; k++) {
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
  
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      int numElem = cells[b][e]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      cells[b][e]->cell_data = cell_data;
      cells[b][e]->cell_data_distance = vector<ScalarT>(numElem);
      cells[b][e]->cell_data_seed = vector<size_t>(numElem);
      cells[b][e]->cell_data_seedindex = vector<size_t>(numElem);
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set cell data
  ////////////////////////////////////////////////////////////////////////////////
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      DRV nodes = cells[b][e]->nodes;
      int numElem = cells[b][e]->numElem;
      for (int c=0; c<numElem; c++) {
        Kokkos::View<ScalarT[1][3],HostDevice> center("center");
        for (size_t i=0; i<nodes.extent(1); i++) {
          for (int j=0; j<spaceDim; j++) {
            center(0,j) += nodes(c,i,j)/(ScalarT)nodes.extent(1);
          }
        }
        ScalarT distance = 1.0e6;
        int cnode = 0;
        for (int k=0; k<numSeeds; k++) {
          ScalarT dx = center(0,0)-seeds(k,0);
          ScalarT dy = center(0,1)-seeds(k,1);
          ScalarT dz = center(0,2)-seeds(k,2);
          ScalarT cdist = sqrt(dx*dx + dy*dy + dz*dz);
          if (cdist<distance) {
            cnode = k;
            distance = cdist;
          }
        }
        
        for (int i=0; i<9; i++) {
          cells[b][e]->cell_data(c,i) = rotation_data(cnode,i);
        }
        
        cells[b][e]->cellData->have_cell_rotation = true;
        cells[b][e]->cellData->have_cell_phi = false;
        
        cells[b][e]->cell_data_seed[c] = cnode;
        cells[b][e]->cell_data_seedindex[c] = seedIndex(cnode);
        cells[b][e]->cell_data_distance[c] = distance;
        
      }
    }
    
  }
  
  meshimporttimer.stop();
  if (verbosity>5 && Commptr->getRank() == 0) {
    cout << "mesh data compute time: " << meshimporttimer.totalElapsedTime(false) << endl;
  }
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh:computeMeshData" << endl;
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV meshInterface::getElemNodes(const int & block, const int & elemID) {
  vector<size_t> localIds;
  DRV blocknodes;
  vector<string> eBlocks;
  mesh->getElementBlockNames(eBlocks);
  
  panzer_stk::workset_utils::getIdsAndVertices(*mesh, eBlocks[block], localIds, blocknodes);
  int nnodes = blocknodes.extent(1);
  
  DRV cnodes("element nodes",1,nnodes,spaceDim);
  for (int i=0; i<nnodes; i++) {
    for (int j=0; j<spaceDim; j++) {
      cnodes(0,i,j) = blocknodes(elemID,i,j);
    }
  }
  return cnodes;
}


// ========================================================================================
// ========================================================================================

template<class V>
void meshInterface::remesh(const Teuchos::RCP<V> & u, vector<vector<Teuchos::RCP<cell> > > & cells) {
  
  /*
  auto u_kv = u->getLocalView<HostDevice>();
  
  for (size_t b=0; b<cells.size(); b++) {
    for( size_t e=0; e<cells[b].size(); e++ ) {
      Kokkos::View<LO***,AssemblyDevice> index = cells[b][e]->index;
      DRV nodes = cells[b][e]->nodes;
      bool changed = false;
      for (int p=0; p<cells[b][e]->numElem; p++) {
        
        for( int i=0; i<nodes.extent(1); i++ ) {
          if (meshmod_xvar >= 0) {
            int pindex = index(p,meshmod_xvar,i);
            ScalarT xval = u_kv(pindex,0);
            ScalarT xpert = xval;
            if (meshmod_usesmoother)
              xpert = meshmod_layer_size*(1.0/3.14159*atan(100.0*(xval-meshmod_center)+0.5));
            
            if (xpert > meshmod_TOL) {
              nodes(p,i,0) += xpert;
              changed = true;
            }
          }
          if (meshmod_yvar >= 0) {
            int pindex = index(p,meshmod_yvar,i);
            ScalarT yval = u_kv(pindex,0);
            ScalarT ypert = yval;
            if (meshmod_usesmoother)
              ypert = meshmod_layer_size*(1.0/3.14159*atan(100.0*(yval-meshmod_center)+0.5));
            
            if (ypert > meshmod_TOL) {
              nodes(p,i,1) += ypert;
              changed = true;
            }
          }
          if (meshmod_zvar >= 0) {
            int pindex = index(p,meshmod_zvar,i);
            ScalarT zval = u_kv(pindex,0);
            ScalarT zpert = zval;
            if (meshmod_usesmoother)
              zpert = meshmod_layer_size*(1.0/3.14159*atan(100.0*(zval-meshmod_center)+0.5));
            
            if (zpert > meshmod_TOL) {
              nodes(p,i,2) += zpert;
              changed = true;
            }
          }
          if (changed) {
            cells[b][e]->nodes = nodes;
          }
        }
        
      }
    }
  }
   */
}

/////////////////////////////////////////////////////////////////////////////
// Read in discretized data from an exodus mesh
/////////////////////////////////////////////////////////////////////////////

void meshInterface::readMeshData() {
  //Teuchos::RCP<const Tpetra::Map<LO, GO, SolverNode> > & LA_overlapped_map,
                                 //vector<vector<Teuchos::RCP<cell> > > & cells) {
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting mesh::readMeshData ..." << endl;
    }
  }
  
  string exofile;
  string fname;
  
  exofile = settings->sublist("Mesh").get<std::string>("mesh file","mesh.exo");
  
  if (Commptr->getSize() > 1) {
    std::stringstream ssProc, ssPID;
    ssProc << Commptr->getSize();
    ssPID << Commptr->getRank();
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
  int num_node_per_el = eblock.num_nodes_per_entry;
  
  
  // get elem vars
  if (settings->sublist("Mesh").get<bool>("have element data", false)) {
    int num_elem_vars = 0;
    int var_ind;
    numResponses = 1;
    exo_error = ex_get_var_param(exoid, "e", &num_elem_vars); // TMW: this is depracated
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
        boost::split(results, vname, [](char u){return u == '_';});
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
    
    auto dev_offsets = cells[b][0]->wkset->offsets;
    auto offsets = Kokkos::create_mirror_view(dev_offsets);
    Kokkos::deep_copy(offsets,dev_offsets);
    
    for (size_t e=0; e<cells[b].size(); e++) {
      //cindex = cells[b][e]->index;
      auto LIDs = cells[b][e]->LIDs_host;
      auto nDOF = cells[b][e]->cellData->numDOF_host;
      
      for (int n=0; n<nDOF(0); n++) {
        //Kokkos::View<GO**,HostDevice> GIDs = assembler->cells[b][e]->GIDs;
        for (size_t p=0; p<cells[b][e]->numElem; p++) {
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
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished mesh::readMeshData" << endl;
    }
  }
  
}


// ========================================================================================
// ========================================================================================

void meshInterface::updateMeshData(const int & newrandseed,
                                   vector<vector<Teuchos::RCP<cell> > > & cells,
                                   Teuchos::RCP<MultiScale> & multiscale_manager) {
  
  // Determine how many seeds there are
  size_t localnumSeeds = 0;
  size_t numSeeds = 0;
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      for (size_t k=0; k<cells[b][e]->numElem; k++) {
        if (cells[b][e]->cell_data_seed[k] > localnumSeeds) {
          localnumSeeds = cells[b][e]->cell_data_seed[k];
        }
      }
    }
  }
  //Comm->MaxAll(&localnumSeeds, &numSeeds, 1);
  Teuchos::reduceAll<int,size_t>(*Commptr,Teuchos::REDUCE_MAX,1,&localnumSeeds,&numSeeds);
  numSeeds += 1; //To properly allocate and iterate
  
  // Create a random number generator
  std::default_random_engine generator(newrandseed);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////
  
  int numdata = 9;
  
  //cout << "solver numSeeds = " << numSeeds << endl;
  
  std::normal_distribution<ScalarT> ndistribution(0.0,1.0);
  Kokkos::View<ScalarT**,HostDevice> rotation_data("cell_data",numSeeds,numdata);
  for (size_t k=0; k<numSeeds; k++) {
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
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set cell data
  ////////////////////////////////////////////////////////////////////////////////
  
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      int numElem = cells[b][e]->numElem;
      for (int c=0; c<numElem; c++) {
        int cnode = cells[b][e]->cell_data_seed[c];
        for (int i=0; i<9; i++) {
          cells[b][e]->cell_data(c,i) = rotation_data(cnode,i);
        }
      }
    }
  }
  
  multiscale_manager->updateMeshData(rotation_data);
}
