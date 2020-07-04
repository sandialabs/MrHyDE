/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "subgridFEM.hpp"
#include "cell.hpp"

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

SubGridFEM::SubGridFEM(const Teuchos::RCP<MpiComm> & LocalComm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                       topo_RCP & macro_cellTopo_, int & num_macro_time_steps_,
                       ScalarT & macro_deltat_) :
settings(settings_), macro_cellTopo(macro_cellTopo_),
num_macro_time_steps(num_macro_time_steps_), macro_deltat(macro_deltat_) {
  
  LocalComm = LocalComm_;
  dimension = settings->sublist("Mesh").get<int>("dim",2);
  subgridverbose = settings->sublist("Solver").get<int>("verbosity",0);
  multiscale_method = settings->get<string>("multiscale method","mortar");
  numrefine = settings->sublist("Mesh").get<int>("refinements",0);
  shape = settings->sublist("Mesh").get<string>("shape","quad");
  macroshape = settings->sublist("Mesh").get<string>("macro-shape","quad");
  time_steps = settings->sublist("Solver").get<int>("number of steps",1);
  initial_time = settings->sublist("Solver").get<ScalarT>("initial time",0.0);
  final_time = settings->sublist("Solver").get<ScalarT>("final time",1.0);
  write_subgrid_state = settings->sublist("Solver").get<bool>("write subgrid state",true);
  error_type = settings->sublist("Postprocess").get<string>("error type","L2"); // or "H1"
  store_aux_and_flux = settings->sublist("Postprocess").get<bool>("store aux and flux",false);
  string solver = settings->sublist("Solver").get<string>("solver","steady-state");
  if (solver == "steady-state") {
    final_time = 0.0;
  }
  
  soln = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  adjsoln = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  solndot = Teuchos::rcp(new SolutionStorage<LA_MultiVector>(settings));
  
  have_sym_factor = false;
  sub_NLtol = settings->sublist("Solver").get<ScalarT>("nonlinear TOL",1.0E-12);
  sub_maxNLiter = settings->sublist("Solver").get<int>("max nonlinear iters",10);
  useDirect = settings->sublist("Solver").get<bool>("use direct solver",true);
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid physics
  /////////////////////////////////////////////////////////////////////////////////////
  
  if (settings->isParameter("Functions input file")) {
    std::string filename = settings->get<std::string>("Functions input file");
    ifstream fn(filename.c_str());
    if (fn.good()) {
      Teuchos::RCP<Teuchos::ParameterList> functions_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
      Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*functions_parlist) );
      settings->setParameters( *functions_parlist );
    }
    else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
      TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the functions input file: " + filename);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Read-in any mesh-dependent data (from file)
  ////////////////////////////////////////////////////////////////////////////////
  
  have_mesh_data = false;
  have_rotation_phi = false;
  have_rotations = false;
  have_multiple_data_files = false;
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
  
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

int SubGridFEM::addMacro(DRV & macronodes_,
                         Kokkos::View<int****,HostDevice> & macrosideinfo_,
                         Kokkos::View<GO**,HostDevice> & macroGIDs_,
                         Kokkos::View<LO***,AssemblyDevice> & macroindex_,
                         Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & macroorientation_) {
  
  Teuchos::TimeMonitor localmeshtimer(*sgfemTotalAddMacroTimer);
  
  Teuchos::RCP<SubGridLocalData> newdata = Teuchos::rcp( new SubGridLocalData(macronodes_,
                                                                              macrosideinfo_,
                                                                              macroGIDs_,
                                                                              macroindex_,
                                                                              macroorientation_) );
  localData.push_back(newdata);
  int bnum = localData.size()-1;
  return bnum;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::setUpSubgridModels() {
  
  Teuchos::TimeMonitor subgridsetuptimer(*sgfemTotalSetUpTimer);
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid mesh
  /////////////////////////////////////////////////////////////////////////////////////
  
  string blockID = "eblock";
  
  vector<vector<ScalarT> > nodes;
  vector<vector<GO> > connectivity;
  Kokkos::View<int****,HostDevice> sideinfo;
  
  vector<string> eBlocks;
  
  SubGridTools sgt(LocalComm, macroshape, shape, localData[0]->macronodes,
                   localData[0]->macrosideinfo);
  
  {
    Teuchos::TimeMonitor localmeshtimer(*sgfemSubMeshTimer);
    
    sgt.createSubMesh(numrefine);
    
    nodes = sgt.getNodes(localData[0]->macronodes);
    int reps = localData[0]->macronodes.extent(0);
    connectivity = sgt.getSubConnectivity(reps);
    sideinfo = sgt.getNewSideinfo(localData[0]->macrosideinfo);
    
    for (size_t c=0; c<sideinfo.extent(0); c++) { // number of elem in cell
      for (size_t i=0; i<sideinfo.extent(1); i++) { // number of variables
        for (size_t j=0; j<sideinfo.extent(2); j++) { // number of sides per element
          if (sideinfo(c,i,j,0) == 1) {
            sideinfo(c,i,j,0) = 5;
            sideinfo(c,i,j,1) = -1;
          }
        }
      }
    }
    
    panzer_stk::SubGridMeshFactory meshFactory(shape, nodes, connectivity, blockID);
    
    Teuchos::RCP<panzer_stk::STK_Interface> mesh = meshFactory.buildMesh(*(LocalComm->getRawMpiComm()));
    
    mesh->getElementBlockNames(eBlocks);
    
    meshFactory.completeMeshConstruction(*mesh,*(LocalComm->getRawMpiComm()));
    sub_mesh = Teuchos::rcp(new meshInterface(settings, LocalComm) );
    sub_mesh->mesh = mesh;
  }
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid physics
  /////////////////////////////////////////////////////////////////////////////////////
  sub_physics = Teuchos::rcp( new physics(settings, LocalComm, sub_mesh->cellTopo,
                                          sub_mesh->sideTopo, sub_mesh->mesh) );
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the subgrid discretizations
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_disc = Teuchos::rcp( new discretization(settings, LocalComm, sub_mesh->mesh, sub_physics->unique_orders,
                                              sub_physics->unique_types) );
  
  
  int numSubElem = connectivity.size();
  
  settings->sublist("Solver").set<int>("workset size",numSubElem);
  vector<Teuchos::RCP<FunctionManager> > functionManagers;
  functionManagers.push_back(Teuchos::rcp(new FunctionManager(blockID,
                                                              numSubElem,
                                                              sub_disc->numip[0],
                                                              sub_disc->numip_side[0])));
  
  ////////////////////////////////////////////////////////////////////////////////
  // Define the functions on each block
  ////////////////////////////////////////////////////////////////////////////////
  
  sub_physics->defineFunctions(functionManagers);
  
  ////////////////////////////////////////////////////////////////////////////////
  // The DOF-manager needs to be aware of the physics and the discretization(s)
  ////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<panzer::DOFManager> DOF = sub_disc->buildDOF(sub_mesh->mesh,
                                                            sub_physics->varlist,
                                                            sub_physics->types,
                                                            sub_physics->orders,
                                                            sub_physics->useDG);
  
  sub_physics->setBCData(settings, sub_mesh->mesh, DOF, sub_disc->cards);
  //sub_disc->setIntegrationInfo(cells, boundaryCells, DOF, sub_physics);
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the parameter manager, the assembler and the solver
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_params = Teuchos::rcp( new ParameterManager(LocalComm, settings, sub_mesh->mesh,
                                                  sub_physics, sub_disc));
  
  sub_assembler = Teuchos::rcp( new AssemblyManager(LocalComm, settings, sub_mesh->mesh,
                                                    sub_disc, sub_physics, DOF,
                                                    sub_params, numSubElem));
  
  cells = sub_assembler->cells;
  
  Teuchos::RCP<CellMetaData> cellData = cells[0][0]->cellData;
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Boundary cells are not set up properly due to the lack of side sets in the subgrid mesh
  // These just need to be defined once though
  /////////////////////////////////////////////////////////////////////////////////////
  
  int numNodesPerElem = sub_mesh->cellTopo[0]->getNodeCount();
  vector<Teuchos::RCP<BoundaryCell> > newbcells;
  
  int numLocalBoundaries = localData[0]->macrosideinfo.extent(2);
  
  vector<int> unique_sides;
  vector<int> unique_local_sides;
  vector<string> unique_names;
  vector<vector<size_t> > boundary_groups;
  
  sgt.getUniqueSides(sideinfo, unique_sides, unique_local_sides, unique_names,
                     macrosidenames, boundary_groups);
  
  vector<stk::mesh::Entity> stk_meshElems;
  sub_mesh->mesh->getMyElements(blockID, stk_meshElems);
  
  // May need to be PHX::Device
  Kokkos::View<const LO**,Kokkos::LayoutRight,HostDevice> LIDs = DOF->getLIDs();
  
  for (size_t s=0; s<unique_sides.size(); s++) {
    
    string sidename = unique_names[s];
    vector<size_t> group = boundary_groups[s];
    
    int prog = 0;
    while (prog < group.size()) {
      int currElem = numSubElem;  // Avoid faults in last iteration
      if (prog+currElem > group.size()){
        currElem = group.size()-prog;
      }
      Kokkos::View<int*,AssemblyDevice> eIndex("element indices",currElem);
      Kokkos::View<int*,AssemblyDevice> sideIndex("local side indices",currElem);
      DRV currnodes("currnodes", currElem, numNodesPerElem, dimension);

      auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
      auto host_sideIndex = Kokkos::create_mirror_view(sideIndex); // mirror on host
      auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
      for (int e=0; e<currElem; e++) {
        host_eIndex(e) = group[e+prog];
        host_sideIndex(e) = unique_local_sides[s];
        for (int n=0; n<numNodesPerElem; n++) {
          for (int m=0; m<dimension; m++) {
            host_currnodes(e,n,m) = nodes[connectivity[eIndex(e)][n]][m];
          }
        }
      }
      int sideID = s;
     
      Kokkos::deep_copy(currnodes,host_currnodes);
      Kokkos::deep_copy(eIndex,host_eIndex);
      Kokkos::deep_copy(sideIndex,host_sideIndex); 
      
      // Build the Kokkos View of the cell GIDs ------
      vector<vector<GO> > cellGIDs;
      int numLocalDOF = 0;
      for (int i=0; i<currElem; i++) {
        vector<GO> GIDs;
        size_t elemID = eIndex(i);
        DOF->getElementGIDs(elemID, GIDs, blockID);
        cellGIDs.push_back(GIDs);
        numLocalDOF = GIDs.size(); // should be the same for all elements
      }
      Kokkos::View<GO**,HostDevice> hostGIDs("GIDs on host device",currElem,numLocalDOF);
      for (int i=0; i<currElem; i++) {
        for (int j=0; j<numLocalDOF; j++) {
          hostGIDs(i,j) = cellGIDs[i][j];
        }
      }
      
      Kokkos::View<LO**,HostDevice> hostLIDs("LIDs on host device",currElem,numLocalDOF);
      for (int i=0; i<currElem; i++) {
        size_t elemID = eIndex(i);
        for (int j=0; j<numLocalDOF; j++) {
          hostLIDs(i,j) = LIDs(elemID,j);
        }
      }
      
      //-----------------------------------------------
      // Set the side information (soon to be removed)-
      Kokkos::View<int****,HostDevice> sideinfo = sub_physics->getSideInfo(0,host_eIndex);
      
      //-----------------------------------------------
      // Set the cell orientation ---
      Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices",
                                                                      currElem, numNodesPerElem);
      for (int i=0; i<currElem; i++) {
        vector<stk::mesh::EntityId> stk_nodeids;
        size_t elemID = eIndex(i);
        sub_mesh->mesh->getNodeIdsForElement(stk_meshElems[elemID], stk_nodeids);
        for (int n=0; n<numNodesPerElem; n++) {
          currind(i,n) = stk_nodeids[n];
        }
      }
      
      Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orient_drv("kv to orients",currElem);
      Intrepid2::OrientationTools<AssemblyDevice>::getOrientation(orient_drv, currind, *(sub_mesh->cellTopo[0]));
      
      newbcells.push_back(Teuchos::rcp(new BoundaryCell(cellData,currnodes,eIndex,sideIndex,
                                                        sideID,sidename, newbcells.size(),
                                                        hostGIDs, hostLIDs, sideinfo, orient_drv)));
      
      prog += currElem;
    }
    
    
  }
  
  boundaryCells.push_back(newbcells);
  
  sub_assembler->boundaryCells = boundaryCells;
  
  sub_solver = Teuchos::rcp( new solver(LocalComm, settings, sub_mesh, sub_disc, sub_physics,
                                        DOF, sub_assembler, sub_params) );
  
  sub_postproc = Teuchos::rcp( new PostprocessManager(LocalComm, settings, sub_mesh->mesh, sub_disc, sub_physics,
                                                      functionManagers, sub_assembler) );
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Create a subgrid function mananger
  /////////////////////////////////////////////////////////////////////////////////////
  
  {
    Teuchos::TimeMonitor localtimer(*sgfemLinearAlgebraSetupTimer);
    
    varlist = sub_physics->varlist[0];
    functionManagers[0]->setupLists(sub_physics->varlist[0], macro_paramnames,
                                macro_disc_paramnames);
    sub_assembler->wkset[0]->params_AD = paramvals_KVAD;
    
    functionManagers[0]->wkset = sub_assembler->wkset[0];
    
    functionManagers[0]->validateFunctions();
    functionManagers[0]->decomposeFunctions();
  }
  
  /////////////////////////////////////////////////////////////////////////////////////
  // A bunch of stuff that needs to be removed
  /////////////////////////////////////////////////////////////////////////////////////
  
  cost_estimate = 1.0*cells[0].size()*(cells[0][0]->numElem)*time_steps;
  res = Teuchos::rcp( new LA_MultiVector(sub_solver->LA_owned_map,1)); // allocate residual
  J = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_overlapped_graph));
  
  if (LocalComm->getSize() > 1) {
    res_over = Teuchos::rcp( new LA_MultiVector(sub_solver->LA_overlapped_map,1)); // allocate residual
    sub_J_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_overlapped_graph));
  }
  else {
    res_over = res;
    sub_J_over = J;
  }
  u = Teuchos::rcp( new LA_MultiVector(sub_solver->LA_overlapped_map,1));
  phi = Teuchos::rcp( new LA_MultiVector(sub_solver->LA_overlapped_map,1));
  
  int nmacroDOF = localData[0]->macroGIDs.extent(1);
  d_um = Teuchos::rcp( new LA_MultiVector(sub_solver->LA_owned_map,nmacroDOF)); // reset residual
  d_sub_res_overm = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,nmacroDOF));
  d_sub_resm = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_owned_map,nmacroDOF));
  d_sub_u_prevm = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_owned_map,nmacroDOF));
  d_sub_u_overm = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,nmacroDOF));
  
  du_glob = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_owned_map,1));
  if (LocalComm->getSize() > 1) {
    du = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,1));
  }
  else {
    du = du_glob;
  }
  
  wkset = sub_assembler->wkset;
  
  wkset[0]->addAux(macro_varlist.size());
  for(size_t e=0; e<boundaryCells[0].size(); e++) {
    boundaryCells[0][e]->addAuxVars(macro_varlist);
    boundaryCells[0][e]->setAuxUseBasis(macro_usebasis);
    boundaryCells[0][e]->auxoffsets = macro_offsets;
    boundaryCells[0][e]->wkset = wkset[0];
  }
  
  // TMW: would like to remove these since most of this is stored by the
  //      parameter manager
  
  vector<GO> params;
  if (sub_params->paramOwnedAndShared.size() == 0) {
    params.push_back(0);
  }
  else {
    params = sub_params->paramOwnedAndShared;
  }
  
  const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();
  
  param_overlapped_map = Teuchos::rcp(new LA_Map(INVALID, params, 0, LocalComm));
  
  num_active_params = sub_params->getNumParams(1);
  num_stochclassic_params = sub_params->getNumParams(2);
  stochclassic_param_names = sub_params->getParamsNames(2);
  
  stoch_param_types = sub_params->stochastic_distribution;
  stoch_param_means = sub_params->getStochasticParams("mean");
  stoch_param_vars = sub_params->getStochasticParams("variance");
  stoch_param_mins = sub_params->getStochasticParams("min");
  stoch_param_maxs = sub_params->getStochasticParams("max");
  discparamnames = sub_params->discretized_param_names;
  
  
  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////
  // Go through all of the macro-elements using this subgrid model and store
  // all of the local information
  /////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////
  
  for (size_t mindex = 0; mindex<localData.size(); mindex++) {
    
    /////////////////////////////////////////////////////////////////////////////////////
    // Define the local nodes
    /////////////////////////////////////////////////////////////////////////////////////
    
    localData[mindex]->nodes = sgt.getNewNodes(localData[mindex]->macronodes);
    
    localData[mindex]->setIP(cells[0][0]->cellData, cells[0][0]->orientation);
    
    vector<size_t> gids;
    for (size_t e=0; e<cells[0][0]->numElem; e++){
      size_t id = localData[mindex]->getMacroID(e);
      gids.push_back(id);
    }
    localData[mindex]->macroIDs = gids;
    
    /////////////////////////////////////////////////////////////////////////////////////
    // Define the local sideinfo
    /////////////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<int****,HostDevice> newsideinfo = sgt.getNewSideinfo(localData[mindex]->macrosideinfo);
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubSideinfoTimer);
      
      int sprog = 0;
      // Redefine the sideinfo for the subcells
      Kokkos::View<int****,HostDevice> subsideinfo("subcell side info", cells[0][0]->numElem, newsideinfo.extent(1),
                                                   newsideinfo.extent(2), newsideinfo.extent(3));
      
      for (size_t c=0; c<cells[0][0]->numElem; c++) { // number of elem in cell
        for (size_t i=0; i<newsideinfo.extent(1); i++) { // number of variables
          for (size_t j=0; j<newsideinfo.extent(2); j++) { // number of sides per element
            for (size_t k=0; k<newsideinfo.extent(3); k++) { // boundary information
              subsideinfo(c,i,j,k) = newsideinfo(sprog,i,j,k);
            }
            if (subsideinfo(c,i,j,0) == 1) {
              subsideinfo(c,i,j,0) = 5;
              subsideinfo(c,i,j,1) = -1;
            }
          }
        }
        sprog += 1;
        
      }
      localData[mindex]->sideinfo = subsideinfo;
      //KokkosTools::print(macrosideinfo[mindex]);
      //KokkosTools::print(subsideinfo);
      
      vector<int> unique_sides;
      vector<int> unique_local_sides;
      vector<string> unique_names;
      vector<vector<size_t> > boundary_groups;
      
      sgt.getUniqueSides(subsideinfo, unique_sides, unique_local_sides, unique_names,
                         macrosidenames, boundary_groups);
      
      
      vector<string> bnames;
      vector<DRV> boundaryNodes;
      vector<vector<size_t> > boundaryMIDs;
      // Number of cells in each group is less than workset size, so just add the groups
      // without breaking into subgroups
      for (size_t s=0; s<unique_sides.size(); s++) {
        vector<size_t> group = boundary_groups[s];
        DRV currnodes("currnodes", group.size(), numNodesPerElem, dimension);
        vector<size_t> mIDs;
        for (int e=0; e<group.size(); e++) {
          size_t eIndex = group[e];
          size_t mID = localData[mindex]->getMacroID(eIndex);
          
          mIDs.push_back(mID);
          for (int n=0; n<numNodesPerElem; n++) {
            for (int m=0; m<dimension; m++) {
              currnodes(e,n,m) = localData[mindex]->nodes(eIndex,n,m);//newnodes[connectivity[eIndex][n]][m];
            }
          }
        }
        boundaryNodes.push_back(currnodes);
        bnames.push_back(unique_names[s]);
        boundaryMIDs.push_back(mIDs);
        
      }
      localData[mindex]->boundaryNodes = boundaryNodes;
      localData[mindex]->boundaryNames = bnames;
      localData[mindex]->boundaryMIDs = boundaryMIDs;
      localData[mindex]->setBoundaryIndexGIDs(); // must be done after boundaryMIDs are set
      
      Kokkos::View<int**,UnifiedDevice> currbcs("boundary conditions",subsideinfo.extent(1),
                                                 localData[mindex]->macrosideinfo.extent(2));
      for (size_t i=0; i<subsideinfo.extent(1); i++) { // number of variables
        for (size_t j=0; j<localData[mindex]->macrosideinfo.extent(2); j++) { // number of sides per element
          currbcs(i,j) = 5;
        }
      }
      for (size_t c=0; c<subsideinfo.extent(0); c++) {
        for (size_t i=0; i<subsideinfo.extent(1); i++) { // number of variables
          for (size_t j=0; j<subsideinfo.extent(2); j++) { // number of sides per element
            if (subsideinfo(c,i,j,0) > 1) { // TMW: should != 5
              for (size_t p=0; p<unique_sides.size(); p++) {
                if (unique_sides[p] == subsideinfo(c,i,j,1)) {
                  currbcs(i,p) = subsideinfo(c,i,j,0);
                }
              }
            }
          }
        }
      }
      localData[mindex]->bcs = currbcs;
     
    }
    
    // This can only be done after the boundary nodes have been set up
    vector<Kokkos::View<LO*,AssemblyDevice> > localSideIDs;
    vector<Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> > borientation;
    for (size_t bcell=0; bcell<sub_assembler->boundaryCells[0].size(); bcell++) {
      localSideIDs.push_back(sub_assembler->boundaryCells[0][bcell]->localSideID);
      borientation.push_back(sub_assembler->boundaryCells[0][bcell]->orientation);
    }
    localData[mindex]->setBoundaryIP(cells[0][0]->cellData, localSideIDs, borientation);
    
    /////////////////////////////////////////////////////////////////////////////////////
    // Add sub-grid discretizations
    /////////////////////////////////////////////////////////////////////////////////////
    
   
    /*
    vector<int> BIDs;
    for (size_t bb=0; bb<boundaryCells[0].size(); bb++) {
      
      int cBID = wkset[0]->addSide(localData[mindex]->boundaryNodes[bb], boundaryCells[0][bb]->sidenum,
                                   boundaryCells[0][bb]->localSideID,
                                   boundaryCells[0][bb]->orientation);
      BIDs.push_back(cBID);
      
    }
    localData[mindex]->BIDs = BIDs;
    */
    
    //////////////////////////////////////////////////////////////
    // Set the initial conditions
    //////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubICTimer);
      
      Teuchos::RCP<LA_MultiVector> init = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,1));
      this->setInitial(init, mindex, false);
      soln->store(init,initial_time,mindex);
      
      Teuchos::RCP<LA_MultiVector> inita = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,1));
      adjsoln->store(inita,final_time,mindex);
    }
  
    ////////////////////////////////////////////////////////////////////////////////
    // The current macro-element will store the values of its own basis functions
    // at the sub-grid integration points
    // Used to map the macro-scale solution to the sub-grid evaluation/integration pts
    ////////////////////////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor auxbasistimer(*sgfemComputeAuxBasisTimer);
      
      nummacroVars = macro_varlist.size();
      if (mindex == 0) {
        if (multiscale_method != "mortar" ) {
          localData[mindex]->computeMacroBasisVolIP(macro_cellTopo, macro_basis_pointers, sub_disc);
        }
        else {
          localData[mindex]->computeMacroBasisBoundaryIP(macro_cellTopo, macro_basis_pointers, sub_disc);//, wkset[0]);
        }
      }
      else {
        localData[mindex]->aux_side_basis = localData[0]->aux_side_basis;
        localData[mindex]->aux_side_basis_grad = localData[0]->aux_side_basis_grad;
      }
    }
  }
  
  sub_physics->setWorkset(wkset);
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::finalize() {
  if (localData.size() > 0) {
    this->setUpSubgridModels();
    
    size_t defblock = 0;
    if (cells.size() > 0) {
      sub_physics->setAuxVars(defblock, macro_varlist);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::addMeshData() {
  
  Teuchos::TimeMonitor localmeshtimer(*sgfemMeshDataTimer);
  
  if (have_mesh_data) {
    
    int numdata = 0;
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
    
    for (size_t b=0; b<localData.size(); b++) {
      int numElem = cells[0][0]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      localData[b]->cell_data = cell_data;
      localData[b]->cell_data_distance = vector<ScalarT>(numElem);
      localData[b]->cell_data_seed = vector<size_t>(numElem);
      localData[b]->cell_data_seedindex = vector<size_t>(numElem);
    }
    
    for (int p=0; p<number_mesh_data_files; p++) {
      
      Teuchos::RCP<data> mesh_data;
      
      string mesh_data_pts_file;
      string mesh_data_file;
      
      if (have_multiple_data_files) {
        stringstream ss;
        ss << p+1;
        mesh_data_pts_file = mesh_data_pts_tag + "." + ss.str() + ".dat";
        mesh_data_file = mesh_data_tag + "." + ss.str() + ".dat";
      }
      else {
        mesh_data_pts_file = mesh_data_pts_tag + ".dat";
        mesh_data_file = mesh_data_tag + ".dat";
      }
      
      mesh_data = Teuchos::rcp(new data("mesh data", dimension, mesh_data_pts_file,
                                        mesh_data_file, false));
      
      for (size_t b=0; b<localData.size(); b++) {
        for (size_t e=0; e<cells[0].size(); e++) {
          int numElem = cells[0][e]->numElem;
          DRV nodes = localData[b]->nodes;
          for (int c=0; c<numElem; c++) {
            Kokkos::View<ScalarT**,AssemblyDevice> center("center",1,3);
            int numnodes = nodes.extent(1);
            for (size_t i=0; i<numnodes; i++) {
              for (size_t j=0; j<dimension; j++) {
                center(0,j) += nodes(c,i,j)/(ScalarT)numnodes;
              }
            }
            ScalarT distance = 0.0;
            
            int cnode = mesh_data->findClosestNode(center(0,0), center(0,1), center(0,2), distance);
            
            bool iscloser = true;
            if (p>0){
              if (localData[b]->cell_data_distance[c] < distance) {
                iscloser = false;
              }
            }
            if (iscloser) {
              Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getdata(cnode);
              
              for (unsigned int i=0; i<cdata.extent(1); i++) {
                localData[b]->cell_data(c,i) = cdata(0,i);
              }
              
              if (have_rotations)
                cells[0][0]->cellData->have_cell_rotation = true;
              if (have_rotation_phi)
                cells[0][0]->cellData->have_cell_phi = true;
              
              localData[b]->cell_data_seed[c] = cnode % 50;
              localData[b]->cell_data_distance[c] = distance;
            }
          }
        }
      }
    }
  }
  
  if (compute_mesh_data) {
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
      
      ScalarT maxpert = 0.2;
      
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
      size_t prog = 0;
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
    
    for (size_t b=0; b<localData.size(); b++) {
      int numElem = cells[0][0]->numElem;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data("cell_data",numElem,numdata);
      localData[b]->cell_data = cell_data;
      localData[b]->cell_data_distance = vector<ScalarT>(numElem);
      localData[b]->cell_data_seed = vector<size_t>(numElem);
      localData[b]->cell_data_seedindex = vector<size_t>(numElem);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set cell data
    ////////////////////////////////////////////////////////////////////////////////
    
    for (size_t b=0; b<localData.size(); b++) {
      for (size_t e=0; e<cells[0].size(); e++) {
        DRV nodes = localData[b]->nodes;
        
        int numElem = cells[0][e]->numElem;
        for (int c=0; c<numElem; c++) {
          Kokkos::View<ScalarT[1][3],HostDevice> center("center");
          for (size_t i=0; i<nodes.extent(1); i++) {
            for (size_t j=0; j<nodes.extent(2); j++) {
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
            localData[b]->cell_data(c,i) = rotation_data(cnode,i);
          }
          
          cells[0][0]->cellData->have_cell_rotation = true;
          cells[0][0]->cellData->have_cell_phi = false;
          
          localData[b]->cell_data_seed[c] = cnode;
          localData[b]->cell_data_seedindex[c] = seedIndex(cnode);
          localData[b]->cell_data_distance[c] = distance;
          
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::subgridSolver(Kokkos::View<ScalarT***,AssemblyDevice> gl_u,
                               Kokkos::View<ScalarT***,AssemblyDevice> gl_phi,
                               const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                               const bool & compute_jacobian, const bool & compute_sens,
                               const int & num_active_params,
                               const bool & compute_disc_sens, const bool & compute_aux_sens,
                               workset & macrowkset,
                               const int & usernum, const int & macroelemindex,
                               Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) {
  
  Teuchos::TimeMonitor totalsolvertimer(*sgfemSolverTimer);
  
  this->updateLocalData(usernum);
  
  ScalarT current_time = time;
  int macroDOF = macrowkset.numDOF;
  bool usesubadjoint = false;
  for (unsigned int i=0; i<subgradient.extent(0); i++) {
    for (unsigned int j=0; j<subgradient.extent(1); j++) {
      subgradient(i,j) = 0.0;
    }
  }
  if (abs(current_time - final_time) < 1.0e-12)
    is_final_time = true;
  else
    is_final_time = false;
  
  ///////////////////////////////////////////////////////////////////////////////////
  // Subgrid transient
  ///////////////////////////////////////////////////////////////////////////////////
  
  ScalarT alpha = 0.0;
  
  ///////////////////////////////////////////////////////////////////////////////////
  // Solve the subgrid problem(s)
  ///////////////////////////////////////////////////////////////////////////////////
  int cnumElem = cells[0][0]->numElem;
  
  Kokkos::View<ScalarT***,AssemblyDevice> cg_u("local u",cnumElem,
                                               gl_u.extent(1),gl_u.extent(2));
  Kokkos::View<ScalarT***,AssemblyDevice> cg_phi("local phi",cnumElem,
                                                 gl_phi.extent(1),gl_phi.extent(2));
  
  for (int e=0; e<cnumElem; e++) {
    for (unsigned int i=0; i<gl_u.extent(1); i++) {
      for (unsigned int j=0; j<gl_u.extent(2); j++) {
        //cg_u(e,i,j) = gl_u(macroelemindex,i,j);
        cg_u(e,i,j) = gl_u(localData[usernum]->macroIDs[e],i,j);
      }
    }
  }
  for (int e=0; e<cnumElem; e++) {
    for (unsigned int i=0; i<gl_phi.extent(1); i++) {
      for (unsigned int j=0; j<gl_phi.extent(2); j++) {
        //cg_phi(e,i,j) = gl_phi(macroelemindex,i,j);
        cg_phi(e,i,j) = gl_phi(localData[usernum]->macroIDs[e],i,j);
      }
    }
  }
  //KokkosTools::print(cg_u);
  
  Kokkos::View<ScalarT***,AssemblyDevice> lambda = cg_u;
  if (isAdjoint) {
    lambda = cg_phi;
    //lambda_dot = gl_phi_dot;
  }
  
  // remove seeding on active params for now
  if (compute_sens) {
    this->sacadoizeParams(false, num_active_params);
  }
  
  //////////////////////////////////////////////////////////////
  // Set the initial conditions
  //////////////////////////////////////////////////////////////
  
  ScalarT prev_time = 0.0; // TMW: is this actually used???
  Teuchos::RCP<LA_MultiVector> prev_u;
  {
    Teuchos::TimeMonitor localtimer(*sgfemInitialTimer);
    
    size_t numtimes = soln->times[usernum].size();
    if (isAdjoint) {
      if (isTransient) {
        bool foundfwd = soln->extractPrevious(prev_u, usernum, current_time, prev_time);
        bool foundadj = adjsoln->extract(phi, usernum, current_time);
      }
      else {
        bool foundfwd = soln->extract(prev_u, usernum, current_time);
        bool foundadj = adjsoln->extract(phi, usernum, current_time);
      }
    }
    else { // forward or compute sens
      if (isTransient) {
        bool foundfwd = soln->extractPrevious(prev_u, usernum, current_time, prev_time);
        if (!foundfwd) { // this subgrid has not been solved at this time yet
          foundfwd = soln->extractLast(prev_u, usernum, prev_time);
        }
      }
      else {
        bool foundfwd = soln->extractLast(prev_u,usernum,prev_time);
      }
      if (compute_sens) {
        double nexttime = 0.0;
        bool foundadj = adjsoln->extractNext(phi,usernum,current_time,nexttime);
      }
    }
  }
  
  auto prev_u_kv = prev_u->getLocalView<HostDevice>();
  auto u_kv = u->getLocalView<HostDevice>();
  
  for (size_t i=0; i<u_kv.extent(0); i++) {
    for (size_t j=0; j<u_kv.extent(1); j++) {
      u_kv(i,j) = prev_u_kv(i,j);
    }
  }
  
  this->performGather(0, prev_u, 0, 0);
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->resetPrevSoln();
    }
  }
  
  //////////////////////////////////////////////////////////////
  // Use the coarse scale solution to solve local transient/nonlinear problem
  //////////////////////////////////////////////////////////////
  
  Teuchos::RCP<LA_MultiVector> d_u = d_um;
  if (compute_sens) {
    d_u = Teuchos::rcp( new LA_MultiVector(sub_solver->LA_owned_map, num_active_params)); // reset residual
  }
  d_u->putScalar(0.0);
  
  res->putScalar(0.0);
  //J->setAllToScalar(0.0);
  
  ScalarT h = 0.0;
  wkset[0]->resetFlux();
  
  if (isTransient) {
    ScalarT sgtime = prev_time;
    Teuchos::RCP<LA_MultiVector> prev_u = u;
    vector<Teuchos::RCP<LA_MultiVector> > curr_fsol;
    vector<ScalarT> subsolvetimes;
    subsolvetimes.push_back(sgtime);
    if (isAdjoint) {
      // First, we need to resolve the forward problem
      
      for (int tstep=0; tstep<time_steps; tstep++) {
        Teuchos::RCP<LA_MultiVector> recu = Teuchos::rcp( new LA_MultiVector(sub_solver->LA_overlapped_map,1)); // reset residual
        
        *recu = *u;
        sgtime += macro_deltat/(ScalarT)time_steps;
        subsolvetimes.push_back(sgtime);
        
        // set du/dt and \lambda
        alpha = (ScalarT)time_steps/macro_deltat;
        wkset[0]->alpha = alpha;
        wkset[0]->deltat= 1.0/alpha;
        wkset[0]->deltat_KV(0) = 1.0/alpha;
        
        Kokkos::View<ScalarT***,AssemblyDevice> currlambda = cg_u;
        
        ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
        
        this->subGridNonlinearSolver(recu, phi, Psol[0], currlambda,
                                     sgtime, isTransient, false, num_active_params, alpha, usernum, false);
        
        curr_fsol.push_back(recu);
        
      }
      
      for (int tstep=0; tstep<time_steps; tstep++) {
        
        size_t numsubtimes = subsolvetimes.size();
        size_t tindex = numsubtimes-1-tstep;
        sgtime = subsolvetimes[tindex];
        // set du/dt and \lambda
        alpha = (ScalarT)time_steps/macro_deltat;
        wkset[0]->alpha = alpha;
        wkset[0]->deltat= 1.0/alpha;
        wkset[0]->deltat_KV(0) = 1.0/alpha;
        
        Kokkos::View<ScalarT***,AssemblyDevice> currlambda = lambda;
        
        ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
        
        this->subGridNonlinearSolver(curr_fsol[tindex-1], phi, Psol[0], currlambda,
                                     sgtime, isTransient, isAdjoint, num_active_params, alpha, usernum, store_adjPrev);
        
        this->computeSubGridSolnSens(d_u, compute_sens, curr_fsol[tindex-1],
                                     phi, Psol[0], currlambda,
                                     sgtime, isTransient, isAdjoint, num_active_params, alpha, lambda_scale, usernum, subgradient);
        
        this->updateFlux(phi, d_u, lambda, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0/(ScalarT)time_steps);
        
      }
    }
    else {
      for (int tstep=0; tstep<time_steps; tstep++) {
        sgtime += macro_deltat/(ScalarT)time_steps;
        // set du/dt and \lambda
        alpha = (ScalarT)time_steps/macro_deltat;
        
        wkset[0]->BDF_wts(0) = 1.0;//alpha;
        wkset[0]->BDF_wts(1) = -1.0;//-alpha;
        
        wkset[0]->alpha = alpha;
        wkset[0]->deltat= 1.0/alpha;
        wkset[0]->deltat_KV(0) = 1.0/alpha;
        
        Kokkos::View<ScalarT***,AssemblyDevice> currlambda = lambda;
        
        ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
        
        for (size_t b=0; b<cells.size(); b++) {
          for (size_t e=0; e<cells[b].size(); e++) {
            cells[b][e]->resetPrevSoln();
            cells[b][e]->resetStageSoln();
          }
        }
        
        //vector_RCP u_prev = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,1));
        //u_prev->update(1.0,*u,0.0);
        //vector_RCP u_stage = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,1));
        //u_stage->update(1.0,*u,0.0);
        
        this->subGridNonlinearSolver(u, phi, Psol[0], currlambda,
                                     sgtime, isTransient, isAdjoint, num_active_params, alpha, usernum, false);
        
        //u->update(1.0, *u_stage, 1.0);
        //u->update(-1.0, *u_prev, 1.0);
        
        this->computeSubGridSolnSens(d_u, compute_sens, u,
                                     phi, Psol[0], currlambda,
                                     sgtime, isTransient, isAdjoint, num_active_params, alpha, lambda_scale, usernum, subgradient);
        
        this->updateFlux(u, d_u, lambda, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0/(ScalarT)time_steps);
      }
    }
    
  }
  else {
    
    wkset[0]->deltat = 1.0;
    wkset[0]->deltat_KV(0) = 1.0;
    
    this->subGridNonlinearSolver(u, phi, Psol[0], lambda,
                                 current_time, isTransient, isAdjoint, num_active_params, alpha, usernum, false);
    
    this->computeSubGridSolnSens(d_u, compute_sens, u,
                                 phi, Psol[0], lambda,
                                 current_time, isTransient, isAdjoint, num_active_params, alpha, 1.0, usernum, subgradient);
    
    if (isAdjoint) {
      this->updateFlux(phi, d_u, lambda, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0);
    }
    else {
      this->updateFlux(u, d_u, lambda, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0);
    }
    
  }
  
  if (isAdjoint) {
    adjsoln->store(phi,current_time,usernum);
  }
  else if (!compute_sens) {
    soln->store(u,current_time,usernum);
  }
  
  if (store_aux_and_flux) {
    this->storeFluxData(lambda, macrowkset.res);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Store macro-dofs and flux (for ML-based subgrid)
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::storeFluxData(Kokkos::View<ScalarT***,AssemblyDevice> lambda, Kokkos::View<AD**,AssemblyDevice> flux) {
  
  int num_dof_lambda = lambda.extent(1)*lambda.extent(2);
  
  std::ofstream ofs;
  
  // Input data - macro DOFs
  ofs.open ("input_data.txt", std::ofstream::out | std::ofstream::app);
  ofs.precision(10);
  for (size_t e=0; e<lambda.extent(0); e++) {
    for (size_t i=0; i<lambda.extent(1); i++) {
      for (size_t j=0; j<lambda.extent(2); j++) {
        ofs << lambda(e,i,j) << "  ";
      }
    }
    ofs << endl;
  }
  ofs.close();
  
  // Output data - upscaled flux
  ofs.open ("output_data.txt", std::ofstream::out | std::ofstream::app);
  ofs.precision(10);
  for (size_t e=0; e<flux.extent(0); e++) {
    //for (size_t i=0; i<flux.extent(1); i++) {
    //for (size_t j=0; j<flux.extent(2); j++) {
    ofs << flux(e,0).val() << "  ";
    //}
    //}
    ofs << endl;
  }
  ofs.close();
  
  // Output derivatives
  /*Kokkos::View<int**,AssemblyDevice> offsets = macrowkset->offsets;
   ofs.open ("output_gradients.txt", std::ofstream::out | std::ofstream::app);
   ofs.precision(10);
   for (size_t e=0; e<flux.extent(0); e++) {
   for (size_t i=0; i<offsets.extent(0); i++) {
   //for (size_t j=0; j<flux.extent(2); j++) {
   for (size_t k=0; k<num_dof_lambda; k++) {
   ofs << flux(e,0).fastAccessDx(k) << "  ";
   }
   //ofs << endl;
   //}
   ofs << endl;
   }
   ofs.close();
   */
}

///////////////////////////////////////////////////////////////////////////////////////
// Re-seed the global parameters
///////////////////////////////////////////////////////////////////////////////////////


void SubGridFEM::sacadoizeParams(const bool & seed_active, const int & num_active_params) {
  
  /*
   if (seed_active) {
   size_t pprog = 0;
   for (size_t i=0; i<paramvals_KVAD.extent(0); i++) {
   if (paramtypes[i] == 1) { // active parameters
   for (size_t j=0; j<paramvals_KVAD.extent(1); j++) {
   paramvals_KVAD(i,j) = AD(maxDerivs,pprog,paramvals_KVAD(i,j).val());
   pprog++;
   }
   }
   else { // inactive, stochastic, or discrete parameters
   for (size_t j=0; j<paramvals_KVAD.extent(1); j++) {
   paramvals_KVAD(i,j) = AD(paramvals_KVAD(i,j).val());
   }
   }
   }
   }
   else {
   size_t pprog = 0;
   for (size_t i=0; i<paramvals_KVAD.extent(0); i++) {
   for (size_t j=0; j<paramvals_KVAD.extent(1); j++) {
   paramvals_KVAD(i,j) = AD(paramvals_KVAD(i,j).val());
   }
   }
   }
   */
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Subgrid Nonlinear Solver
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::subGridNonlinearSolver(Teuchos::RCP<LA_MultiVector> & sub_u,
                                        Teuchos::RCP<LA_MultiVector> & sub_phi,
                                        Teuchos::RCP<LA_MultiVector> & sub_params,
                                        Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                        const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                        const int & num_active_params, const ScalarT & alpha, const int & usernum,
                                        const bool & store_adjPrev) {
  
  
  Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverTimer);
  
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_scaled(1);
  Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> resnorm_initial(1);
  resnorm[0] = 10.0*sub_NLtol;
  resnorm_initial[0] = resnorm[0];
  resnorm_scaled[0] = resnorm[0];
  
  int iter = 0;
  Kokkos::View<ScalarT**,AssemblyDevice> aPrev;
  
  while (iter < sub_maxNLiter && resnorm_scaled[0] > sub_NLtol) {
    
    sub_J_over->resumeFill();
    
    sub_J_over->setAllToScalar(0.0);
    res_over->putScalar(0.0);
    
    wkset[0]->time = time;
    wkset[0]->isTransient = isTransient;
    wkset[0]->isAdjoint = isAdjoint;
    
    int numElem = cells[0][0]->numElem;
    int numDOF = cells[0][0]->GIDs.extent(1);
    
    Kokkos::View<ScalarT***,UnifiedDevice> local_res, local_J, local_Jdot;
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverAllocateTimer);
      local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,numDOF,1);
      local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,numDOF,numDOF);
      //local_Jdot = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian dot",numElem,numDOF,numDOF);
    }
    {
      Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSetSolnTimer);
      this->performGather(0, sub_u, 0, 0);
      if (isAdjoint) {
        this->performGather(0, sub_phi, 2, 0);
      }
      //this->performGather(usernum, sub_params, 4, 0);
      
      this->performBoundaryGather(0, sub_u, 0, 0);
      if (isAdjoint) {
        this->performBoundaryGather(0, sub_phi, 2, 0);
      }
      
      for (size_t e=0; e < boundaryCells[0].size(); e++) {
        boundaryCells[0][e]->aux = lambda;
      }
    }
    
    ////////////////////////////////////////////////
    // volume assembly
    ////////////////////////////////////////////////
    
    for (size_t e=0; e<cells[0].size(); e++) {
      if (isAdjoint) {
        // TMW: this may not work properly using the new formulation
        aPrev = cells[0][e]->adjPrev;
        if (is_final_time) {
          for (unsigned int i=0; i<aPrev.extent(0); i++) {
            for (unsigned int j=0; j<aPrev.extent(1); j++) {
              cells[0][e]->adjPrev(i,j) = 0.0;
            }
          }
        }
      }
      
      wkset[0]->localEID = e;
      cells[0][e]->updateData();
      
      for (int p=0; p<numElem; p++) {
        for (int n=0; n<numDOF; n++) {
          for (unsigned int s=0; s<local_res.extent(2); s++) {
            local_res(p,n,s) = 0.0;
          }
          for (unsigned int s=0; s<local_J.extent(2); s++) {
            local_J(p,n,s) = 0.0;
          }
        }
      }
      //KokkosTools::print(local_J);
      {
        Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverJacResTimer);
        
        cells[0][e]->computeJacRes(time, isTransient, isAdjoint,
                                   true, false, num_active_params, false, false, false,
                                   local_res, local_J,
                                   sub_assembler->assemble_volume_terms[0],
                                   sub_assembler->assemble_face_terms[0]);
        
      }
      //KokkosTools::print(local_J);
      //KokkosTools::print(local_res);
      
      {
        Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverInsertTimer);
        auto localMatrix = sub_J_over->getLocalMatrix();
        Kokkos::View<LO**,HostDevice> LIDs = cells[0][e]->LIDs;
        LO numentries = static_cast<LO>(LIDs.extent(1));
        ScalarT vals[numentries];
        LO cols[numentries];
        for (unsigned int i=0; i<LIDs.extent(0); i++) { // should be Kokkos::parallel_for on SubgridExec
          for( size_t row=0; row<LIDs.extent(1); row++ ) {
            LO rowIndex = LIDs(i,row);
            ScalarT val = local_res(i,row,0);
            res_over->sumIntoLocalValue(rowIndex,0, val);
            for( size_t col=0; col<numentries; col++ ) {
              vals[col] = local_J(i,row,col);
              cols[col] = LIDs(i,col);
            }
            localMatrix.sumIntoValues(rowIndex, cols, numentries, vals, true, false); // bools: isSorted, useAtomics
            // indices are not actually sorted, but this seems to run faster
            // may need to set useAtomics = true if subgridexec is not Serial
          }
        }
      }
    }
    
    // KokkosTools::print(sub_J_over);
    
    ////////////////////////////////////////////////
    // boundary assembly
    ////////////////////////////////////////////////
    
    for (size_t e=0; e<boundaryCells[0].size(); e++) {
      
      if (boundaryCells[0][e]->numElem > 0) {
        wkset[0]->localEID = e;
        
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverAllocateTimer);
          local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",boundaryCells[0][e]->numElem,numDOF,1);
          local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",boundaryCells[0][e]->numElem,numDOF,numDOF);
        }
        
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverJacResTimer);
          
          boundaryCells[0][e]->computeJacRes(time, isTransient, isAdjoint,
                                             true, false, num_active_params, false, false, false,
                                             local_res, local_J);
          
        }
        
        //KokkosTools::print(local_J);
        //KokkosTools::print(local_res);
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverInsertTimer);
          Kokkos::View<GO**,HostDevice> GIDs = boundaryCells[0][e]->GIDs;
          //KokkosTools::print(GIDs);
          
          for (unsigned int i=0; i<GIDs.extent(0); i++) {
            Teuchos::Array<ScalarT> vals(GIDs.extent(1));
            Teuchos::Array<GO> cols(GIDs.extent(1));
            for (size_t row=0; row<GIDs.extent(1); row++ ) {
              GO rowIndex = GIDs(i,row);
              ScalarT val = local_res(i,row,0);
              res_over->sumIntoGlobalValue(rowIndex,0, val);
              for (size_t col=0; col<GIDs.extent(1); col++ ) {
                vals[col] = local_J(i,row,col);
                cols[col] = GIDs(i,col);
              }
              sub_J_over->sumIntoGlobalValues(rowIndex, cols, vals);
            }
          }
        }
      }
    }
    
    sub_J_over->fillComplete();
    
    if (LocalComm->getSize() > 1) {
      J->resumeFill();
      J->setAllToScalar(0.0);
      J->doExport(*sub_J_over, *(sub_solver->exporter), Tpetra::ADD);
      J->fillComplete();
    }
    else {
      J = sub_J_over;
    }
    //KokkosTools::print(J);
    
    
    if (LocalComm->getSize() > 1) {
      res->putScalar(0.0);
      res->doExport(*res_over, *(sub_solver->exporter), Tpetra::ADD);
    }
    else {
      res = res_over;
    }
    
    if (useDirect) {
      if (have_sym_factor) {
        Am2Solver->setA(J, Amesos2::SYMBFACT);
        Am2Solver->setX(du_glob);
        Am2Solver->setB(res);
      }
      else {
        Am2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", J, du_glob, res);
        Am2Solver->symbolicFactorization();
        have_sym_factor = true;
      }
      //Am2Solver->numericFactorization().solve();
    }
    //KokkosTools::print(res);
    
    if (iter == 0) {
      res->normInf(resnorm_initial);
      if (resnorm_initial[0] > 0.0)
        resnorm_scaled[0] = 1.0;
      else
        resnorm_scaled[0] = 0.0;
    }
    else {
      res->normInf(resnorm);
      resnorm_scaled[0] = resnorm[0]/resnorm_initial[0];
    }
    if(LocalComm->getRank() == 0 && subgridverbose>5) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Subgrid Nonlinear Iteration: " << iter << endl;
      cout << "***** Scaled Norm of nonlinear residual: " << resnorm_scaled << endl;
      cout << "*********************************************************" << endl;
    }
    
    if (resnorm_scaled[0] > sub_NLtol) {
      
      Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSolveTimer);
      du_glob->putScalar(0.0);
      if (useDirect) {
        Am2Solver->numericFactorization().solve();
      }
      else {
        if (have_belos) {
          //belos_problem->setProblem(du_glob, res);
        }
        else {
          belos_problem = Teuchos::rcp(new LA_LinearProblem(J, du_glob, res));
          have_belos = true;
          
          belosList = Teuchos::rcp(new Teuchos::ParameterList());
          belosList->set("Maximum Iterations",    50); // Maximum number of iterations allowed
          belosList->set("Convergence Tolerance", 1.0E-10);    // Relative convergence tolerance requested
          belosList->set("Verbosity", Belos::Errors);
          belosList->set("Output Frequency",0);
          
          //belosList->set("Verbosity", Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
          //belosList->set("Output Frequency",10);
          
          int numEqns = sub_solver->numVars[0];
          belosList->set("number of equations",numEqns);
          
          belosList->set("Output Style",          Belos::Brief);
          belosList->set("Implicit Residual Scaling", "None");
          
          belos_solver = Teuchos::rcp(new Belos::BlockGmresSolMgr<ScalarT, LA_MultiVector, LA_Operator>(belos_problem, belosList));
          
        }
        if (have_preconditioner) {
          //MueLu::ReuseTpetraPreconditioner(J,*belos_M);
        }
        else {
          belos_M = sub_solver->buildPreconditioner(J);
          //belos_problem->setRightPrec(belos_M);
          belos_problem->setLeftPrec(belos_M);
          have_preconditioner = true;
          
        }
        belos_problem->setProblem(du_glob, res);
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSolveTimer);
          belos_solver->solve();
          
        }
        //sub_solver->linearSolver(J,res,du_glob);
      }
      if (LocalComm->getSize() > 1) {
        du->putScalar(0.0);
        du->doImport(*du_glob, *(sub_solver->importer), Tpetra::ADD);
      }
      else {
        du = du_glob;
      }
      if (isAdjoint) {
        
        sub_phi->update(1.0, *du, 1.0);
      }
      else {
        sub_u->update(1.0, *du, 1.0);
      }
    }
    iter++;
    
  }
  //KokkosTools::print(sub_u);
  
}

//////////////////////////////////////////////////////////////
// Compute the derivative of the local solution w.r.t coarse
// solution or w.r.t parameters
//////////////////////////////////////////////////////////////

void SubGridFEM::computeSubGridSolnSens(Teuchos::RCP<LA_MultiVector> & d_sub_u,
                                        const bool & compute_sens,
                                        Teuchos::RCP<LA_MultiVector> & sub_u,
                                        Teuchos::RCP<LA_MultiVector> & sub_phi,
                                        Teuchos::RCP<LA_MultiVector> & sub_param,
                                        Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                                        const ScalarT & time,
                                        const bool & isTransient, const bool & isAdjoint,
                                        const int & num_active_params, const ScalarT & alpha,
                                        const ScalarT & lambda_scale, const int & usernum,
                                        Kokkos::View<ScalarT**,AssemblyDevice> subgradient) {
  
  Teuchos::TimeMonitor localtimer(*sgfemSolnSensTimer);
  
  Teuchos::RCP<LA_MultiVector> d_sub_res_over = d_sub_res_overm;
  Teuchos::RCP<LA_MultiVector> d_sub_res = d_sub_resm;
  Teuchos::RCP<LA_MultiVector> d_sub_u_prev = d_sub_u_prevm;
  Teuchos::RCP<LA_MultiVector> d_sub_u_over = d_sub_u_overm;
  
  if (compute_sens) {
    int numsubDerivs = d_sub_u->getNumVectors();
    d_sub_res_over = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,numsubDerivs));
    d_sub_res = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_owned_map,numsubDerivs));
    d_sub_u_prev = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_owned_map,numsubDerivs));
    d_sub_u_over = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,numsubDerivs));
  }
  
  d_sub_res_over->putScalar(0.0);
  d_sub_res->putScalar(0.0);
  d_sub_u_prev->putScalar(0.0);
  d_sub_u_over->putScalar(0.0);
  
  ScalarT scale = -1.0*lambda_scale;
  
  if (multiscale_method != "mortar") {
    this->performGather(0, sub_u, 0, 0);
    if (isAdjoint) {
      this->performGather(0, sub_phi, 2, 0);
    }
    for (size_t e=0; e < cells[0].size(); e++) {
      cells[0][e]->aux = lambda;
    }
  }
  else {
    this->performBoundaryGather(0, sub_u, 0, 0);
    if (isAdjoint) {
      this->performBoundaryGather(0, sub_phi, 2, 0);
    }
    for (size_t e=0; e < boundaryCells[0].size(); e++) {
      boundaryCells[0][e]->aux = lambda;
    }
  }
  //this->performGather(usernum, sub_param, 4, 0);
  
  
  
  
  if (compute_sens) {
    
    this->sacadoizeParams(true, num_active_params);
    wkset[0]->time = time;
    wkset[0]->isTransient = isTransient;
    wkset[0]->isAdjoint = isAdjoint;
    
    if (multiscale_method != "mortar") {
      int numElem = cells[0][0]->numElem;
      
      int snumDOF = cells[0][0]->GIDs.extent(1);
      
      Kokkos::View<ScalarT***,UnifiedDevice> local_res, local_J, local_Jdot;
      
      local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,snumDOF,num_active_params);
      
      local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,snumDOF,snumDOF);
      
      for (size_t e=0; e<cells[0].size(); e++) {
        
        wkset[0]->localEID = e;
        cells[0][e]->updateData();
        
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<snumDOF; n++) {
            for (unsigned int s=0; s<local_res.extent(2); s++) {
              local_res(p,n,s) = 0.0;
            }
            for (unsigned int s=0; s<local_J.extent(2); s++) {
              local_J(p,n,s) = 0.0;
            }
          }
        }
        
        cells[0][e]->computeJacRes(time, isTransient, isAdjoint,
                                   false, true, num_active_params, false, false, false,
                                   local_res, local_J,
                                   sub_assembler->assemble_volume_terms[0],
                                   sub_assembler->assemble_face_terms[0]);
        
        Kokkos::View<GO**,HostDevice>  GIDs = cells[0][e]->GIDs;
        for (unsigned int i=0; i<GIDs.extent(0); i++) {
          for( size_t row=0; row<GIDs.extent(1); row++ ) {
            int rowIndex = GIDs(i,row);
            for( size_t col=0; col<num_active_params; col++ ) {
              ScalarT val = local_res(i,row,col);
              d_sub_res_over->sumIntoGlobalValue(rowIndex,col, 1.0*val);
            }
          }
        }
      }
      auto sub_phi_kv = sub_phi->getLocalView<HostDevice>();
      auto d_sub_res_over_kv = d_sub_res_over->getLocalView<HostDevice>();
      
      for (int p=0; p<num_active_params; p++) {
        for (int i=0; i<sub_phi->getGlobalLength(); i++) {
          subgradient(p,0) += sub_phi_kv(i,0) * d_sub_res_over_kv(i,p);
        }
      }
    }
    else {
      
      for (size_t e=0; e<boundaryCells[0].size(); e++) {
        int numElem = boundaryCells[0][e]->numElem;
        int snumDOF = boundaryCells[0][e]->GIDs.extent(1);
        
        Kokkos::View<ScalarT***,UnifiedDevice> local_res, local_J, local_Jdot;
        
        local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,snumDOF,num_active_params);
        
        local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,snumDOF,snumDOF);
        
        wkset[0]->localEID = e;
        //wkset[0]->var_bcs = subgridbcs[usernum];
        
        cells[0][e]->updateData();
        
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<snumDOF; n++) {
            for (unsigned int s=0; s<local_res.extent(2); s++) {
              local_res(p,n,s) = 0.0;
            }
            for (unsigned int s=0; s<local_J.extent(2); s++) {
              local_J(p,n,s) = 0.0;
            }
          }
        }
        
        boundaryCells[0][e]->computeJacRes(time, isTransient, isAdjoint,
                                           false, true, num_active_params, false, false, false,
                                           local_res, local_J);
        
        Kokkos::View<GO**,HostDevice>  GIDs = boundaryCells[usernum][e]->GIDs;
        for (unsigned int i=0; i<GIDs.extent(0); i++) {
          for (size_t row=0; row<GIDs.extent(1); row++ ) {
            int rowIndex = GIDs(i,row);
            for (size_t col=0; col<num_active_params; col++ ) {
              ScalarT val = local_res(i,row,col);
              d_sub_res_over->sumIntoGlobalValue(rowIndex,col, 1.0*val);
            }
          }
        }
      }
      auto sub_phi_kv = sub_phi->getLocalView<HostDevice>();
      auto d_sub_res_over_kv = d_sub_res_over->getLocalView<HostDevice>();
      
      for (int p=0; p<num_active_params; p++) {
        for (int i=0; i<sub_phi->getGlobalLength(); i++) {
          subgradient(p,0) += sub_phi_kv(i,0) * d_sub_res_over_kv(i,p);
        }
      }
    }
  }
  else {
    wkset[0]->time = time;
    wkset[0]->isTransient = isTransient;
    wkset[0]->isAdjoint = isAdjoint;
    
    Kokkos::View<ScalarT***,UnifiedDevice> local_res, local_J, local_Jdot;
    
    if (multiscale_method != "mortar") {
      
      for (size_t e=0; e<cells[0].size(); e++) {
        
        int numElem = cells[0][e]->numElem;
        int snumDOF = cells[0][e]->GIDs.extent(1);
        int anumDOF = cells[0][e]->auxGIDs.extent(1);
        
        local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,snumDOF,1);
        local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,snumDOF,anumDOF);
        
        wkset[0]->localEID = e;
        
        // TMW: this may not work properly with new version
        cells[0][e]->updateData();
        
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<snumDOF; n++) {
            for (unsigned int s=0; s<local_res.extent(2); s++) {
              local_res(p,n,s) = 0.0;
            }
            for (unsigned int s=0; s<local_J.extent(2); s++) {
              local_J(p,n,s) = 0.0;
            }
          }
        }
        
        cells[0][e]->computeJacRes(time, isTransient, isAdjoint,
                                   true, false, num_active_params, false, true, false,
                                   local_res, local_J,
                                   sub_assembler->assemble_volume_terms[0],
                                   sub_assembler->assemble_face_terms[0]);
        Kokkos::View<GO**,HostDevice> GIDs = cells[0][e]->GIDs;
        Kokkos::View<GO**,HostDevice> aGIDs = cells[0][e]->auxGIDs;
        //vector<vector<int> > aoffsets = cells[0][e]->auxoffsets;
        
        for (unsigned int i=0; i<GIDs.extent(0); i++) {
          for (size_t row=0; row<GIDs.extent(1); row++ ) {
            int rowIndex = GIDs(i,row);
            for (size_t col=0; col<aGIDs.extent(1); col++ ) {
              ScalarT val = local_J(i,row,col);
              int colIndex = col;
              d_sub_res_over->sumIntoGlobalValue(rowIndex,colIndex, scale*val);
            }
          }
        }
      }
    }
    else {
      for (size_t e=0; e<boundaryCells[0].size(); e++) {
        
        int numElem = boundaryCells[0][e]->numElem;
        int snumDOF = boundaryCells[0][e]->GIDs.extent(1);
        int anumDOF = boundaryCells[0][e]->auxGIDs.extent(1);
        
        local_res = Kokkos::View<ScalarT***,UnifiedDevice>("local residual",numElem,snumDOF,1);
        local_J = Kokkos::View<ScalarT***,UnifiedDevice>("local Jacobian",numElem,snumDOF,anumDOF);
        
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<snumDOF; n++) {
            for (unsigned int s=0; s<local_res.extent(2); s++) {
              local_res(p,n,s) = 0.0;
            }
            for (unsigned int s=0; s<local_J.extent(2); s++) {
              local_J(p,n,s) = 0.0;
            }
          }
        }
        
        boundaryCells[0][e]->computeJacRes(time, isTransient, isAdjoint,
                                           true, false, num_active_params, false, true, false,
                                           local_res, local_J);
        Kokkos::View<GO**,HostDevice> GIDs = boundaryCells[0][e]->GIDs;
        Kokkos::View<GO**,HostDevice> aGIDs = boundaryCells[0][e]->auxGIDs;
        for (unsigned int i=0; i<GIDs.extent(0); i++) {
          for (size_t row=0; row<GIDs.extent(1); row++ ) {
            int rowIndex = GIDs(i,row);
            for (size_t col=0; col<aGIDs.extent(1); col++ ) {
              ScalarT val = local_J(i,row,col);
              int colIndex = col;
              d_sub_res_over->sumIntoGlobalValue(rowIndex,colIndex, scale*val);
            }
          }
        }
      }
    }
    
    if (LocalComm->getSize() > 1) {
      d_sub_res->doExport(*d_sub_res_over, *(sub_solver->exporter), Tpetra::ADD);
    }
    else {
      d_sub_res = d_sub_res_over;
    }
    
    if (useDirect) {
      
      Teuchos::TimeMonitor localtimer(*sgfemSolnSensLinearSolverTimer);
      
      int numsubDerivs = d_sub_u_over->getNumVectors();
      
      auto d_sub_u_over_kv = d_sub_u_over->getLocalView<HostDevice>();
      auto d_sub_res_kv = d_sub_res->getLocalView<HostDevice>();
      for (int c=0; c<numsubDerivs; c++) {
        Teuchos::RCP<LA_MultiVector> x = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,1));
        Teuchos::RCP<LA_MultiVector> b = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_owned_map,1));
        auto b_kv = b->getLocalView<HostDevice>();
        auto x_kv = x->getLocalView<HostDevice>();
        
        for (int i=0; i<b->getGlobalLength(); i++) {
          b_kv(i,0) += d_sub_res_kv(i,c);
        }
        Am2Solver->setX(x);
        Am2Solver->setB(b);
        Am2Solver->solve();
        
        for (int i=0; i<x->getGlobalLength(); i++) {
          d_sub_u_over_kv(i,c) += x_kv(i,0);
        }
        
      }
    }
    else {
      
      Teuchos::TimeMonitor localtimer(*sgfemSolnSensLinearSolverTimer);
      
      belos_problem->setProblem(d_sub_u_over, d_sub_res);
      belos_solver->solve();
      //sub_solver->linearSolver(J,d_sub_res,d_sub_u_over);
    }
    
    if (LocalComm->getSize() > 1) {
      d_sub_u->putScalar(0.0);
      d_sub_u->doImport(*d_sub_u_over, *(sub_solver->importer), Tpetra::ADD);
    }
    else {
      d_sub_u = d_sub_u_over;
    }
    
  }
}

//////////////////////////////////////////////////////////////
// Update the flux
//////////////////////////////////////////////////////////////

void SubGridFEM::updateFlux(const Teuchos::RCP<LA_MultiVector> & u,
                            const Teuchos::RCP<LA_MultiVector> & d_u,
                            Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                            const bool & compute_sens, const int macroelemindex,
                            const ScalarT & time, workset & macrowkset,
                            const int & usernum, const ScalarT & fwt) {
  
  Teuchos::TimeMonitor localtimer(*sgfemFluxTimer);
  
  //this->updateLocalData(usernum);
  
  for (size_t e=0; e<boundaryCells[0].size(); e++) {
    
    if (boundaryCells[0][e]->sidename == "interior") {
      {
        Teuchos::TimeMonitor localwktimer(*sgfemFluxWksetTimer);
        boundaryCells[0][e]->updateWorksetBasis();
        
        //boundaryCells[0][e]->updateWorksetBasis();
        //wkset[0]->updateSide(boundaryCells[0][e]->sidenum,
        //                     boundaryCells[0][e]->wksetBID);
        
      }
      
      DRV cwts = wkset[0]->wts_side;
      ScalarT h = 0.0;
      wkset[0]->sidename = "interior";
      {
        Teuchos::TimeMonitor localcelltimer(*sgfemFluxCellTimer);
        //boundaryCells[usernum][e]->updateData();
        boundaryCells[0][e]->computeFlux(u, d_u, Psol[0], lambda, time,
                                         0, h, compute_sens);
      }
  
      //KokkosTools::print(wkset[0]->flux);
      
      vector<size_t> bMIDs = localData[usernum]->boundaryMIDs[e];
      for (int c=0; c<boundaryCells[0][e]->numElem; c++) {
        for (int n=0; n<nummacroVars; n++) {
          DRV macrobasis_ip = boundaryCells[0][e]->auxside_basis[macrowkset.usebasis[n]];
          //KokkosTools::print(macrobasis_ip);
          for (unsigned int j=0; j<macrobasis_ip.extent(1); j++) {
            for (unsigned int i=0; i<macrobasis_ip.extent(2); i++) {
              //macrowkset.res(macroelemindex,macrowkset.offsets(n,j)) += mortarbasis_ip(c,j,i)*(wkset[0]->flux(c,n,i))*cwts(c,i)*fwt;
              macrowkset.res(bMIDs[c],macrowkset.offsets(n,j)) += macrobasis_ip(c,j,i)*(wkset[0]->flux(c,n,i))*cwts(c,i)*fwt;
            }
          }
        }
      }
    }
  }
  
}

//////////////////////////////////////////////////////////////
// Compute the initial values for the subgrid solution
//////////////////////////////////////////////////////////////

void SubGridFEM::setInitial(Teuchos::RCP<LA_MultiVector> & initial,
                            const int & usernum, const bool & useadjoint) {
  
  initial->putScalar(0.0);
  // TMW: uncomment if you need a nonzero initial condition
  //      right now, it slows everything down ... especially if using an L2-projection
  
  /*
   bool useL2proj = true;//settings->sublist("Solver").get<bool>("Project initial",true);
   
   if (useL2proj) {
   
   // Compute the L2 projection of the initial data into the discrete space
   Teuchos::RCP<LA_MultiVector> rhs = Teuchos::rcp(new LA_MultiVector(*overlapped_map,1)); // reset residual
   Teuchos::RCP<LA_CrsMatrix>  mass = Teuchos::rcp(new LA_CrsMatrix(Copy, *overlapped_map, -1)); // reset Jacobian
   Teuchos::RCP<LA_MultiVector> glrhs = Teuchos::rcp(new LA_MultiVector(*owned_map,1)); // reset residual
   Teuchos::RCP<LA_CrsMatrix>  glmass = Teuchos::rcp(new LA_CrsMatrix(Copy, *owned_map, -1)); // reset Jacobian
   
   
   //for (size_t b=0; b<cells.size(); b++) {
   for (size_t e=0; e<cells[usernum].size(); e++) {
   int numElem = cells[usernum][e]->numElem;
   vector<vector<int> > GIDs = cells[usernum][e]->GIDs;
   Kokkos::View<ScalarT**,AssemblyDevice> localrhs = cells[usernum][e]->getInitial(true, useadjoint);
   Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[usernum][e]->getMass();
   
   // assemble into global matrix
   for (int c=0; c<numElem; c++) {
   for( size_t row=0; row<GIDs[c].size(); row++ ) {
   int rowIndex = GIDs[c][row];
   ScalarT val = localrhs(c,row);
   rhs->SumIntoGlobalValue(rowIndex,0, val);
   for( size_t col=0; col<GIDs[c].size(); col++ ) {
   int colIndex = GIDs[c][col];
   ScalarT val = localmass(c,row,col);
   mass->InsertGlobalValues(rowIndex, 1, &val, &colIndex);
   }
   }
   }
   }
   //}
   
   
   mass->FillComplete();
   glmass->PutScalar(0.0);
   glmass->Export(*mass, *exporter, Add);
   
   glrhs->PutScalar(0.0);
   glrhs->Export(*rhs, *exporter, Add);
   
   glmass->FillComplete();
   
   Teuchos::RCP<LA_MultiVector> glinitial = Teuchos::rcp(new LA_MultiVector(*overlapped_map,1)); // reset residual
   
   this->linearSolver(glmass, glrhs, glinitial);
   
   initial->Import(*glinitial, *importer, Add);
   
   }
   else {
   
   for (size_t e=0; e<cells[usernum].size(); e++) {
   int numElem = cells[usernum][e]->numElem;
   vector<vector<int> > GIDs = cells[usernum][e]->GIDs;
   Kokkos::View<ScalarT**,AssemblyDevice> localinit = cells[usernum][e]->getInitial(false, useadjoint);
   for (int c=0; c<numElem; c++) {
   for( size_t row=0; row<GIDs[c].size(); row++ ) {
   int rowIndex = GIDs[c][row];
   ScalarT val = localinit(c,row);
   initial->SumIntoGlobalValue(rowIndex,0, val);
   }
   }
   }
   
   }*/
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the error for verification
///////////////////////////////////////////////////////////////////////////////////////

//Kokkos::View<ScalarT**,AssemblyDevice> SubGridFEM::computeError(const ScalarT & time, const int & usernum) {
Kokkos::View<ScalarT**,AssemblyDevice> SubGridFEM::computeError(vector<pair<size_t, string> > & sub_error_list,
                                                                const vector<ScalarT> & times) {
  
  //Kokkos::View<ScalarT***,AssemblyDevice> errors("error",solvetimes.size(), sub_physics->varlist[0].size(), error_types.size());
  Kokkos::View<ScalarT**,AssemblyDevice> errors;
  if (localData.size() > 0) {
    
    errors = Kokkos::View<ScalarT**,AssemblyDevice>("error", times.size(), sub_postproc->error_list[0].size());
    sub_error_list = sub_postproc->error_list[0];
  
    for (size_t t=0; t<times.size(); t++) {
      for (size_t b=0; b<localData.size(); b++) {// loop over coarse scale elements
        bool compute = false;
        if (subgrid_static) {
          compute = true;
        }
        else if (active[t][b]) {
          compute = true;
        }
        if (compute) {
          size_t usernum = b;
          Teuchos::RCP<LA_MultiVector> currsol;
          
          //size_t tindex = t;
          //bool found = soln->extract(currsol, usernum, solvetimes[t]);//, tindex);
          bool found = soln->extract(currsol, usernum, times[t]);//, tindex);
          //bool found = soln->extract(currsol, tindex, usernum);//, tindex);
          
          if (found) {
            this->updateLocalData(usernum);
            
            //Kokkos::View<ScalarT***,AssemblyDevice> localerror("error",solvetimes.size(),numVars[b],error_types.size());
            //bool fnd = solve->soln->extract(u,t);
            this->performGather(0,currsol,0,0);
            sub_postproc->computeError(times[t]);
            
            size_t numerrs = sub_postproc->errors.size();
            
            Kokkos::View<ScalarT*,AssemblyDevice> cerr = sub_postproc->errors[0][0];//sub_postproc->errors[0][numerrs-1];
            for (size_t etype=0; etype<cerr.extent(0); etype++) {
              errors(t,etype) += cerr(etype);
            }
            sub_postproc->errors.clear();
            
          }
        }
      }
    }
  }
  
  return errors;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the objective function
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD*,AssemblyDevice> SubGridFEM::computeObjective(const string & response_type, const int & seedwhat,
                                                              const ScalarT & time, const int & usernum) {
  
  int tindex = -1;
  //for (int tt=0; tt<soln[usernum].size(); tt++) {
  //  if (abs(soln[usernum][tt].first - time)<1.0e-10) {
  //    tindex = tt;
  //  }
  //}
  
  Teuchos::RCP<LA_MultiVector> currsol;
  bool found = soln->extract(currsol,usernum,time,tindex);
  
  Kokkos::View<AD*,AssemblyDevice> objective;
  if (found) {
    this->updateLocalData(usernum);
    bool beensized = false;
    this->performGather(0, currsol, 0,0);
    //this->performGather(usernum, Psol[0], 4, 0);
    
    for (size_t e=0; e<cells[0].size(); e++) {
      Kokkos::View<AD**,AssemblyDevice> curr_obj = cells[0][e]->computeObjective(time, tindex, seedwhat);
      if (!beensized && curr_obj.extent(1)>0) {
        objective = Kokkos::View<AD*,AssemblyDevice>("objective", curr_obj.extent(1));
        beensized = true;
      }
      for (int c=0; c<cells[0][e]->numElem; c++) {
        for (size_t i=0; i<curr_obj.extent(1); i++) {
          objective(i) += curr_obj(c,i);
        }
      }
    }
  }
  
  return objective;
}

///////////////////////////////////////////////////////////////////////////////////////
// Write the solution to a file
///////////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::writeSolution(const string & filename, const int & usernum) {
  
  bool isTD = false;
  if (soln->times[usernum].size() > 1) {
    isTD = true;
  }
  
  string blockID = "eblock";
  
  //////////////////////////////////////////////////////////////
  // Re-create the subgrid mesh
  //////////////////////////////////////////////////////////////
  
  SubGridTools sgt(LocalComm, macroshape, shape, localData[usernum]->macronodes,
                   localData[usernum]->macrosideinfo);
  sgt.createSubMesh(numrefine);
  vector<vector<ScalarT> > nodes = sgt.getSubNodes();
  vector<vector<GO> > connectivity = sgt.getSubConnectivity();
  Kokkos::View<int****,HostDevice> sideinfo = sgt.getSubSideinfo();
  
  size_t numNodesPerElem = connectivity[0].size();
  
  panzer_stk::SubGridMeshFactory submeshFactory(shape, nodes, connectivity, blockID);
  Teuchos::RCP<panzer_stk::STK_Interface> submesh = submeshFactory.buildMesh(*(LocalComm->getRawMpiComm()));
  
  //////////////////////////////////////////////////////////////
  // Add in the necessary fields for plotting
  //////////////////////////////////////////////////////////////
  
  vector<string> vartypes = sub_physics->types[0];
  
  vector<string> subeBlocks;
  submesh->getElementBlockNames(subeBlocks);
  for (size_t j=0; j<sub_physics->varlist[0].size(); j++) {
    if (vartypes[j] == "HGRAD") {
      submesh->addSolutionField(sub_physics->varlist[0][j], subeBlocks[0]);
    }
    else if (vartypes[j] == "HVOL"){
      submesh->addCellField(sub_physics->varlist[0][j], subeBlocks[0]);
    }
    else if (vartypes[j] == "HDIV" || vartypes[j] == "HCURL"){
      submesh->addCellField(sub_physics->varlist[0][j]+"x", subeBlocks[0]);
      submesh->addCellField(sub_physics->varlist[0][j]+"y", subeBlocks[0]);
      submesh->addCellField(sub_physics->varlist[0][j]+"z", subeBlocks[0]);
    }
  }
  vector<string> subextrafieldnames = sub_physics->getExtraFieldNames(0);
  for (size_t j=0; j<subextrafieldnames.size(); j++) {
    submesh->addSolutionField(subextrafieldnames[j], subeBlocks[0]);
  }
  vector<string> subextracellfields = sub_physics->getExtraCellFieldNames(0);
  for (size_t j=0; j<subextracellfields.size(); j++) {
    submesh->addCellField(subextracellfields[j], subeBlocks[0]);
  }
  submesh->addCellField("mesh_data_seed", subeBlocks[0]);
  
  if (discparamnames.size() > 0) {
    for (size_t n=0; n<discparamnames.size(); n++) {
      int paramnumbasis = cells[0][0]->paramindex.extent(1);
      if (paramnumbasis==1) {
        submesh->addCellField(discparamnames[n], subeBlocks[0]);
      }
      else {
        submesh->addSolutionField(discparamnames[n], subeBlocks[0]);
      }
    }
  }
  
  submeshFactory.completeMeshConstruction(*submesh,*(LocalComm->getRawMpiComm()));
  
  //////////////////////////////////////////////////////////////
  // Add fields to mesh
  //////////////////////////////////////////////////////////////
  
  if(isTD) {
    submesh->setupExodusFile(filename);
  }
  int numSteps = soln->times[usernum].size();
  
  for (int m=0; m<numSteps; m++) {
    
    vector<size_t> myElements;
    size_t eprog = 0;
    for (size_t e=0; e<cells[0].size(); e++) {
      for (size_t p=0; p<cells[0][e]->numElem; p++) {
        myElements.push_back(eprog);
        eprog++;
      }
    }
    
    vector_RCP u;
    bool fnd = soln->extract(u,usernum,soln->times[usernum][m],m);
    auto u_kv = u->getLocalView<HostDevice>();
    
    vector<vector<int> > suboffsets = sub_physics->offsets[0];
    // Collect the subgrid solution
    for (int n = 0; n<sub_physics->varlist[0].size(); n++) {
      if (vartypes[n] == "HGRAD") {
        //size_t numsb = cells[usernum][0]->numDOF(n);//index[0][n].size();
        Kokkos::View<ScalarT**,HostDevice> soln_computed("soln",cells[0][0]->numElem, numNodesPerElem); // TMW temp. fix
        string var = sub_physics->varlist[0][n];
        size_t pprog = 0;
        
        for( size_t e=0; e<cells[0].size(); e++ ) {
          int numElem = cells[0][e]->numElem;
          Kokkos::View<GO**,HostDevice> GIDs = cells[0][e]->GIDs;
          for (int p=0; p<numElem; p++) {
            
            for( int i=0; i<numNodesPerElem; i++ ) {
              int pindex = sub_solver->LA_overlapped_map->getLocalElement(GIDs(p,suboffsets[n][i]));
              soln_computed(pprog,i) = u_kv(pindex,0);
            }
            pprog += 1;
          }
        }
        
        submesh->setSolutionFieldData(var, blockID, myElements, soln_computed);
      }
      else if (vartypes[n] == "HVOL") {
        Kokkos::View<ScalarT**,HostDevice> soln_computed("soln",cells[0][0]->numElem, 1);
        string var = sub_physics->varlist[0][n];
        size_t pprog = 0;
        
        for( size_t e=0; e<cells[0].size(); e++ ) {
          int numElem = cells[0][e]->numElem;
          Kokkos::View<GO**,HostDevice> GIDs = cells[0][e]->GIDs;
          for (int p=0; p<numElem; p++) {
            int pindex = sub_solver->LA_overlapped_map->getLocalElement(GIDs(p,suboffsets[n][0]));
            soln_computed(pprog,0) = u_kv(pindex,0);
            pprog += 1;
          }
        }
        submesh->setCellFieldData(var, blockID, myElements, soln_computed);
      }
      else if (vartypes[n] == "HDIV" || vartypes[n] == "HCURL") {
        Kokkos::View<ScalarT**,HostDevice> soln_x("soln",cells[usernum].size(), 1);
        Kokkos::View<ScalarT**,HostDevice> soln_y("soln",cells[usernum].size(), 1);
        Kokkos::View<ScalarT**,HostDevice> soln_z("soln",cells[usernum].size(), 1);
        string var = sub_physics->varlist[0][n];
        size_t pprog = 0;
        
        this->updateLocalData(usernum);
        for( size_t e=0; e<cells[0].size(); e++ ) {
          cells[0][e]->updateWorksetBasis();
          //wkset[0]->update(cells[0][e]->ip,cells[0][e]->wts,
          //                 cells[0][e]->jacobian,cells[0][e]->jacobianInv,
          //                 cells[0][e]->jacobianDet,cells[0][e]->orientation);
          //Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
          //seedwhat(0) = 0;
          wkset[0]->computeSolnVolIP(cells[0][e]->u, cells[0][e]->u_prev,
                                     cells[0][e]->u_stage, 0);
          
          int numElem = cells[0][e]->numElem;
          Kokkos::View<GO**,HostDevice> GIDs = cells[0][e]->GIDs;
          
          for (int p=0; p<numElem; p++) {
            ScalarT avgxval = 0.0;
            ScalarT avgyval = 0.0;
            ScalarT avgzval = 0.0;
            ScalarT avgwt = 0.0;
            for (int j=0; j<suboffsets[n].size(); j++) {
              ScalarT xval = wkset[0]->local_soln(p,n,j,0).val();
              avgxval += xval*wkset[0]->wts(p,j);
              if (dimension > 1) {
                ScalarT yval = wkset[0]->local_soln(p,n,j,1).val();
                avgyval += yval*wkset[0]->wts(p,j);
              }
              if (dimension > 2) {
                ScalarT zval = wkset[0]->local_soln(p,n,j,2).val();
                avgzval += zval*wkset[0]->wts(p,j);
              }
              avgwt += wkset[0]->wts(p,j);
            }
            soln_x(pprog,0) = avgxval/avgwt;
            soln_y(pprog,0) = avgyval/avgwt;
            soln_z(pprog,0) = avgzval/avgwt;
            pprog += 1;
          }
        }
        submesh->setCellFieldData(var+"x", blockID, myElements, soln_x);
        submesh->setCellFieldData(var+"y", blockID, myElements, soln_y);
        submesh->setCellFieldData(var+"z", blockID, myElements, soln_z);
      }
    }
    
    ////////////////////////////////////////////////////////////////
    // Discretized Parameters
    ////////////////////////////////////////////////////////////////
    
    /*
     if (discparamnames.size() > 0) {
     for (size_t n=0; n<discparamnames.size(); n++) {
     FC soln_computed;
     bool isConstant = false;
     DRV subnodes = cells[usernum][0]->nodes;
     int numSubNodes = subnodes.extent(0);
     int paramnumbasis =cells[usernum][0]->paramindex[n].size();
     if (paramnumbasis>1)
     soln_computed = FC(cells[usernum].size(), paramnumbasis);
     else {
     isConstant = true;
     soln_computed = FC(cells[usernum].size(), numSubNodes);
     }
     for( size_t e=0; e<cells[usernum].size(); e++ ) {
     vector<int> paramGIDs = cells[usernum][e]->paramGIDs;
     vector<vector<int> > paramoffsets = wkset[0]->paramoffsets;
     for( int i=0; i<paramnumbasis; i++ ) {
     int pindex = param_overlapped_map->LID(paramGIDs[paramoffsets[n][i]]);
     if (isConstant) {
     for( int j=0; j<numSubNodes; j++ ) {
     soln_computed(e,j) = (*(Psol[0]))[0][pindex];
     }
     }
     else
     soln_computed(e,i) = (*(Psol[0]))[0][pindex];
     }
     }
     if (isConstant) {
     submesh->setCellFieldData(discparamnames[n], blockID, myElements, soln_computed);
     }
     else {
     submesh->setSolutionFieldData(discparamnames[n], blockID, myElements, soln_computed);
     }
     }
     }
     */
    
    // Collect the subgrid extra fields (material coefficients)
    //TMW: ADD
    
    // vector<FC> subextrafields;// = phys->getExtraFields(b);
    // DRV rnodes = cells[b][0]->nodes;
    // for (size_t j=0; j<subextrafieldnames.size(); j++) {
    // FC efdata(cells[b].size(), rnodes.extent(1));
    // subextrafields.push_back(efdata);
    // }
    
    // vector<FC> cfields;
    // for (size_t k=0; k<cells[b].size(); k++) {
    // DRV snodes = cells[b][k]->nodes;
    // cfields = sub_physics->getExtraFields(b, snodes, solvetimes[m]);
    // for (size_t j=0; j<subextrafieldnames.size(); j++) {
    // for (size_t i=0; i<snodes.extent(1); i++) {
    // subextrafields[j](k,i) = cfields[j](0,i);
    // }
    // }
    // //vcfields.push_back(cfields[0]);
    // }
    
    //bvbw added block to pushd extrafields to a vector<FC>,
    //     originally cfields did not have more than one vector
    //	vector<FC> vcfields;
    //	vector<FC> cfields;
    //	for (size_t j=0; j<subextrafieldnames.size(); j++) {
    //	  for (size_t k=0; k<subcells[level][b].size(); k++) {
    //	    DRV snodes = subcells[level][b][k]->getNodes();
    //	    cfields = sub_sub_physics[level]->getExtraFields(b, snodes, 0.0);
    //	  }
    //	  vcfields.push_back(cfields[0]);
    //	}
    
    //        for (size_t k=0; k<subcells[level][b].size(); k++) {
    //         DRV snodes = subcells[level][b][k]->getNodes();
    //vector<FC> newbasis, newbasisGrad;
    //for (size_t b=0; b<basisTypes.size(); b++) {
    //  newbasis.push_back(DiscTools::evaluateBasis(basisTypes[b], snodes));
    //  newbasisGrad.push_back(DiscTools::evaluateBasisGrads(basisTypes[b], snodes, snodes, cellTopo));
    //}
    //          vector<FC> cfields = sub_sub_physics[level]->getExtraFields(b, snodes, 0.0);//subgrid_solvetimes[m]);
    //          for (size_t j=0; j<subextrafieldnames.size(); j++) {
    //            for (size_t i=0; i<snodes.extent(1); i++) {
    //              subextrafields[j](k,i) = vcfields[j](i);
    //            }
    //          }
    //        }
    
    /*
     vector<string> extracellfieldnames = sub_physics->getExtraCellFieldNames(0);
     vector<FC> extracellfields;// = phys->getExtraFields(b);
     for (size_t j=0; j<extracellfieldnames.size(); j++) {
     FC efdata(cells[usernum].size(), 1);
     extracellfields.push_back(efdata);
     }
     for (size_t k=0; k<cells[usernum].size(); k++) {
     cells[usernum][k]->updateSolnWorkset(soln[usernum][m].second, 0); // also updates ip, ijac
     cells[usernum][k]->updateData();
     wkset[0]->time = soln[usernum][m].first;
     vector<FC> cfields = sub_physics->getExtraCellFields(0);
     size_t j = 0;
     for (size_t g=0; g<cfields.size(); g++) {
     for (size_t h=0; h<cfields[g].extent(0); h++) {
     extracellfields[j](k,0) = cfields[g](h,0);
     ++j;
     }
     }
     }
     for (size_t j=0; j<extracellfieldnames.size(); j++) {
     submesh->setCellFieldData(extracellfieldnames[j], blockID, myElements, extracellfields[j]);
     }
     */
    
    
    //Kokkos::View<ScalarT**,HostDevice> cdata("cell data",cells[usernum][0]->numElem, 1);
    Kokkos::View<ScalarT**,HostDevice> cdata("cell data",cells[0].size(), 1);
    if (cells[0][0]->cellData->have_cell_phi || cells[0][0]->cellData->have_cell_rotation) {
      int eprog = 0;
      vector<size_t> cell_data_seed = localData[usernum]->cell_data_seed;
      vector<size_t> cell_data_seedindex = localData[usernum]->cell_data_seedindex;
      Kokkos::View<ScalarT**,AssemblyDevice> cell_data = localData[usernum]->cell_data;
      // TMW: need to use a mirror view here
      for (int p=0; p<cells[0][0]->numElem; p++) {
        //cdata(eprog,0) = cell_data_seedindex[p];
        //eprog++;
      }
    }
    submesh->setCellFieldData("mesh_data_seed", blockID, myElements, cdata);
  
    if(isTD) {
      submesh->writeToExodus(soln->times[usernum][m]);
    }
    else {
      submesh->writeToExodus(filename);
    }
    
  }
  
}

////////////////////////////////////////////////////////////////////////////////
// Add in the sensor data
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                            const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                            const vector<basis_RCP> & basisTypes, const int & usernum) {
  for (size_t e=0; e<cells[usernum].size(); e++) {
    cells[usernum][e]->addSensors(sensor_points,sensor_loc_tol,sensor_data,
                                  have_sensor_data, sub_disc, basisTypes, basisTypes);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<LA_CrsMatrix>  SubGridFEM::getProjectionMatrix() {
  
  // Compute the mass matrix on a reference element
  matrix_RCP mass = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_overlapped_graph) );
  
  int usernum = 0;
  for (size_t e=0; e<cells[usernum].size(); e++) {
    int numElem = cells[usernum][e]->numElem;
    Kokkos::View<GO**,HostDevice> GIDs = cells[usernum][e]->GIDs;
    Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[usernum][e]->getMass();
    for (int c=0; c<numElem; c++) {
      for( size_t row=0; row<GIDs.extent(1); row++ ) {
        GO rowIndex = GIDs(c,row);
        for( size_t col=0; col<GIDs.extent(1); col++ ) {
          GO colIndex = GIDs(c,col);
          ScalarT val = localmass(c,row,col);
          mass->insertGlobalValues(rowIndex, 1, &val, &colIndex);
        }
      }
    }
  }
  
  mass->fillComplete();
  
  matrix_RCP glmass;
  size_t maxEntries = 256;
  if (LocalComm->getSize() > 1) {
    glmass = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_owned_map,maxEntries) );
    glmass->setAllToScalar(0.0);
    glmass->doExport(*mass, *(sub_solver->exporter), Tpetra::ADD);
    glmass->fillComplete();
  }
  else {
    glmass = mass;
  }
  return glmass;
}

////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<LA_CrsMatrix> SubGridFEM::getProjectionMatrix(DRV & ip, DRV & wts,
                                                           pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo) {
  
  pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > my_basisinfo = this->evaluateBasis2(ip);
  matrix_RCP map_over = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_overlapped_graph));
  
  matrix_RCP map;
  if (LocalComm->getSize() > 1) {
    map = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_overlapped_graph));
    map->setAllToScalar(0.0);
  }
  else {
    map = map_over;
  }
  
  Teuchos::Array<ScalarT> vals(1);
  Teuchos::Array<GO> cols(1);
  
  for (size_t k=0; k<ip.extent(1); k++) {
    for (size_t r=0; r<my_basisinfo.second[k].extent(0);r++) {
      for (size_t p=0; p<my_basisinfo.second[k].extent(1);p++) {
        int igid = my_basisinfo.first(k,p+2);
        for (size_t s=0; s<other_basisinfo.second[k].extent(0);s++) {
          for (size_t q=0; q<other_basisinfo.second[k].extent(1);q++) {
            cols[0] = other_basisinfo.first(k,q+2);
            if (r == s) {
              vals[0] = my_basisinfo.second[k](r,p) * other_basisinfo.second[k](s,q) * wts(0,k);
              map_over->sumIntoGlobalValues(igid, cols, vals);
            }
          }
        }
      }
    }
  }
  
  map_over->fillComplete();
  
  if (LocalComm->getSize() > 1) {
    map->doExport(*map_over, *(sub_solver->exporter), Tpetra::ADD);
    map->fillComplete();
  }
  return map;
}

////////////////////////////////////////////////////////////////////////////////
// Get an empty vector
////////////////////////////////////////////////////////////////////////////////

vector_RCP SubGridFEM::getVector() {
  vector_RCP vec = Teuchos::rcp(new LA_MultiVector(sub_solver->LA_overlapped_map,1));
  return vec;
}

////////////////////////////////////////////////////////////////////////////////
// Get the integration points
////////////////////////////////////////////////////////////////////////////////

DRV SubGridFEM::getIP() {
  int numip_per_cell = wkset[0]->numip;
  int usernum = 0; // doesn't really matter
  int totalip = 0;
  for (size_t e=0; e<cells[usernum].size(); e++) {
    totalip += numip_per_cell*cells[usernum][e]->numElem;
  }
  
  DRV refip = DRV("refip",1,totalip,dimension);
  int prog = 0;
  for (size_t e=0; e<cells[usernum].size(); e++) {
    int numElem = cells[usernum][e]->numElem;
    DRV ip = cells[usernum][e]->ip;
    for (size_t c=0; c<numElem; c++) {
      for (size_t i=0; i<ip.extent(1); i++) {
        for (size_t j=0; j<ip.extent(2); j++) {
          refip(0,prog,j) = ip(c,i,j);
        }
        prog++;
      }
    }
  }
  return refip;
  
}

////////////////////////////////////////////////////////////////////////////////
// Get the integration weights
////////////////////////////////////////////////////////////////////////////////

DRV SubGridFEM::getIPWts() {
  int numip_per_cell = wkset[0]->numip;
  int usernum = 0; // doesn't really matter
  int totalip = 0;
  for (size_t e=0; e<cells[usernum].size(); e++) {
    totalip += numip_per_cell*cells[usernum][e]->numElem;
  }
  DRV refwts = DRV("refwts",1,totalip);
  int prog = 0;
  for (size_t e=0; e<cells[usernum].size(); e++) {
    DRV wts = cells[0][e]->cellData->ref_wts;//wkset[0]->ref_wts;//cells[usernum][e]->ijac;
    int numElem = cells[usernum][e]->numElem;
    for (size_t c=0; c<numElem; c++) {
      for (size_t i=0; i<wts.extent(0); i++) {
        refwts(0,prog) = wts(i);//sref_ip_tmp(0,i,j);
        prog++;
      }
    }
  }
  return refwts;
  
}


////////////////////////////////////////////////////////////////////////////////
// Evaluate the basis functions at a set of points
////////////////////////////////////////////////////////////////////////////////

pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridFEM::evaluateBasis2(const DRV & pts) {
  
  size_t numpts = pts.extent(1);
  size_t dimpts = pts.extent(2);
  size_t numGIDs = cells[0][0]->GIDs.extent(1);
  Kokkos::View<int**,AssemblyDevice> owners("owners",numpts,2+numGIDs);
  
  for (size_t e=0; e<cells[0].size(); e++) {
    int numElem = cells[0][e]->numElem;
    DRV nodes = cells[0][e]->nodes;
    for (int c=0; c<numElem;c++) {
      DRV refpts("refpts",1, numpts, dimpts);
      DRVint inRefCell("inRefCell",1,numpts);
      DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
      for (unsigned int i=0; i<nodes.extent(1); i++) {
        for (unsigned int j=0; j<nodes.extent(2); j++) {
          cnodes(0,i,j) = nodes(c,i,j);
        }
      }
      
      CellTools::mapToReferenceFrame(refpts, pts, cnodes, *(sub_mesh->cellTopo[0]));
      CellTools::checkPointwiseInclusion(inRefCell, refpts, *(sub_mesh->cellTopo[0]), 1.0e-12);
      //KokkosTools::print(refpts);
      //KokkosTools::print(inRefCell);
      for (size_t i=0; i<numpts; i++) {
        if (inRefCell(0,i) == 1) {
          owners(i,0) = e;//cells[0][e]->localElemID[c];
          owners(i,1) = c;
          Kokkos::View<GO**,HostDevice> GIDs = cells[0][e]->GIDs;
          for (size_t j=0; j<numGIDs; j++) {
            owners(i,j+2) = GIDs(c,j);
          }
        }
      }
    }
  }
  
  vector<DRV> ptsBasis;
  for (size_t i=0; i<numpts; i++) {
    vector<DRV> currBasis;
    DRV refpt_buffer("refpt_buffer",1,1,dimpts);
    DRV cpt("cpt",1,1,dimpts);
    for (size_t s=0; s<dimpts; s++) {
      cpt(0,0,s) = pts(0,i,s);
    }
    DRV nodes = cells[0][owners(i,0)]->nodes;
    DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
    for (unsigned int k=0; k<nodes.extent(1); k++) {
      for (unsigned int j=0; j<nodes.extent(2); j++) {
        cnodes(0,k,j) = nodes(owners(i,1),k,j);
      }
    }
    CellTools::mapToReferenceFrame(refpt_buffer, cpt, cnodes, *(sub_mesh->cellTopo[0]));
    DRV refpt("refpt",1,dimpts);
    Kokkos::deep_copy(refpt,Kokkos::subdynrankview(refpt_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
    Kokkos::View<int**,AssemblyDevice> offsets = wkset[0]->offsets;
    vector<int> usebasis = wkset[0]->usebasis;
    DRV basisvals("basisvals",offsets.extent(0),numGIDs);
    for (size_t n=0; n<offsets.extent(0); n++) {
      DRV bvals = sub_disc->evaluateBasis(sub_disc->basis_pointers[0][usebasis[n]], refpt);
      for (size_t m=0; m<offsets.extent(1); m++) {
        basisvals(n,offsets(n,m)) = bvals(0,m,0);
      }
    }
    ptsBasis.push_back(basisvals);
    
  }
  pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo(owners, ptsBasis);
  return basisinfo;
  
}

////////////////////////////////////////////////////////////////////////////////
// Evaluate the basis functions at a set of points
// TMW: what is this function for???
////////////////////////////////////////////////////////////////////////////////

pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridFEM::evaluateBasis(const DRV & pts) {
  
  /*
   size_t numpts = pts.extent(1);
   size_t dimpts = pts.extent(2);
   size_t numGIDs = cells[0][0]->GIDs.size();
   FCint owners(numpts,1+numGIDs);
   
   for (size_t e=0; e<cells[0].size(); e++) {
   DRV refpts("refpts",1, numpts, dimpts);
   DRVint inRefCell("inRefCell",1,numpts);
   CellTools<PHX::Device>::mapToReferenceFrame(refpts, pts, cells[0][e]->nodes, *cellTopo);
   CellTools<PHX::Device>::checkPointwiseInclusion(inRefCell, refpts, *cellTopo, 0.0);
   
   for (size_t i=0; i<numpts; i++) {
   if (inRefCell(0,i) == 1) {
   owners(i,0) = e;
   vector<int> GIDs = cells[0][e]->GIDs;
   for (size_t j=0; j<numGIDs; j++) {
   owners(i,j+1) = GIDs[j];
   }
   }
   }
   }
   
   vector<DRV> ptsBasis;
   for (size_t i=0; i<numpts; i++) {
   vector<DRV> currBasis;
   DRV refpt_buffer("refpt_buffer",1,1,dimpts);
   DRV cpt("cpt",1,1,dimpts);
   for (size_t s=0; s<dimpts; s++) {
   cpt(0,0,s) = pts(0,i,s);
   }
   CellTools<PHX::Device>::mapToReferenceFrame(refpt_buffer, cpt, cells[0][owners(i,0)]->nodes, *cellTopo);
   DRV refpt("refpt",1,dimpts);
   Kokkos::deep_copy(refpt,Kokkos::subdynrankview(refpt_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
   
   vector<vector<int> > offsets = wkset[0]->offsets;
   vector<int> usebasis = wkset[0]->usebasis;
   DRV basisvals("basisvals",numGIDs);
   for (size_t n=0; n<offsets.size(); n++) {
   DRV bvals = DiscTools::evaluateBasis(basis_pointers[usebasis[n]], refpt);
   for (size_t m=0; m<offsets[n].size(); m++) {
   basisvals(offsets[n][m]) = bvals(0,m,0);
   }
   }
   ptsBasis.push_back(basisvals);
   }
   
   pair<FCint, vector<DRV> > basisinfo(owners, ptsBasis);
   return basisinfo;
   */
}


////////////////////////////////////////////////////////////////////////////////
// Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<LA_CrsMatrix>  SubGridFEM::getEvaluationMatrix(const DRV & newip, Teuchos::RCP<LA_Map> & ip_map) {
  matrix_RCP map_over = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_overlapped_graph) );
  matrix_RCP map;
  if (LocalComm->getSize() > 1) {
    size_t maxEntries = 256;
    map = Teuchos::rcp( new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(sub_solver->LA_owned_map, maxEntries) );
    
    map->setAllToScalar(0.0);
    map->doExport(*map_over, *(sub_solver->exporter), Tpetra::ADD);
    map->fillComplete();
  }
  else {
    map = map_over;
  }
  return map;
}

////////////////////////////////////////////////////////////////////////////////
// Get the subgrid cell GIDs
////////////////////////////////////////////////////////////////////////////////

Kokkos::View<GO**,HostDevice> SubGridFEM::getCellGIDs(const int & cellnum) {
  return cells[0][cellnum]->GIDs;
}

////////////////////////////////////////////////////////////////////////////////
// Update the subgrid parameters (will be depracated)
////////////////////////////////////////////////////////////////////////////////

void SubGridFEM::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
  for (size_t b=0; b<wkset.size(); b++) {
    wkset[b]->params = params;
    wkset[b]->paramnames = paramnames;
  }
  sub_physics->updateParameters(params, paramnames);
  
}

////////////////////////////////////////////////////////////////////////////////
// TMW: Is the following functions used/required ???
////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,AssemblyDevice> SubGridFEM::getCellFields(const int & usernum, const ScalarT & time) {
  
  /*
   vector<string> extracellfieldnames = sub_physics->getExtraCellFieldNames(0);
   FC extracellfields(cells[usernum].size(),extracellfieldnames.size());
   
   int timeindex = 0;
   for (size_t k=0; k<soln[usernum].size(); k++) {
   if (abs(time-soln[usernum][k].first)<1.0e-10) {
   timeindex = k;
   }
   }
   
   for (size_t k=0; k<cells[usernum].size(); k++) {
   cells[usernum][k]->updateSolnWorkset(soln[usernum][timeindex].second, 0); // also updates ip, ijac
   wkset[0]->time = soln[usernum][timeindex].first;
   cells[usernum][k]->updateData();
   vector<FC> cfields = sub_physics->getExtraCellFields(0);
   size_t j = 0;
   for (size_t g=0; g<cfields.size(); g++) {
   for (size_t h=0; h<cfields[g].extent(0); h++) {
   extracellfields(k,j) = cfields[g](h,0);
   ++j;
   }
   }
   }
   
   return extracellfields;
   */
}

// ========================================================================================
//
// ========================================================================================

void SubGridFEM::performGather(const size_t & b, const vector_RCP & vec,
                               const size_t & type, const size_t & entry) const {
  
  //for (size_t e=0; e < cells[block].size(); e++) {
  //  cells[block][e]->setLocalSoln(vec, type, index);
  //}
  // Get a view of the vector on the HostDevice
  auto vec_kv = vec->getLocalView<HostDevice>();
  
  // Get a corresponding view on the AssemblyDevice
  
  Kokkos::View<LO***,AssemblyDevice> index;
  Kokkos::View<LO*,UnifiedDevice> numDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> data;
  
  for (size_t c=0; c < cells[b].size(); c++) {
    switch(type) {
      case 0 :
        index = cells[b][c]->index;
        numDOF = cells[b][c]->cellData->numDOF;
        data = cells[b][c]->u;
        break;
      case 1 :
        //index = cells[b][c]->index;
        //numDOF = cells[b][c]->cellData->numDOF;
        //data = cells[b][c]->u_dot;
        break;
      case 2 :
        index = cells[b][c]->index;
        numDOF = cells[b][c]->cellData->numDOF;
        data = cells[b][c]->phi;
        break;
      case 3 :
        //index = cells[b][c]->index;
        //numDOF = cells[b][c]->cellData->numDOF;
        //data = cells[b][c]->phi_dot;
        break;
      case 4:
        index = cells[b][c]->paramindex;
        numDOF = cells[b][c]->cellData->numParamDOF;
        data = cells[b][c]->param;
        break;
      case 5 :
        index = cells[b][c]->auxindex;
        numDOF = cells[b][c]->cellData->numAuxDOF;
        data = cells[b][c]->aux;
        break;
      default :
        cout << "ERROR - NOTHING WAS GATHERED" << endl;
    }
    
    parallel_for(RangePolicy<AssemblyExec>(0,index.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t n=0; n<index.extent(1); n++) {
        for (size_t i=0; i<numDOF(n); i++ ) {
          data(e,n,i) = vec_kv(index(e,n,i),entry);
        }
      }
    });
  }
  
}

// ========================================================================================
//
// ========================================================================================

void SubGridFEM::performBoundaryGather(const size_t & b, const vector_RCP & vec,
                                       const size_t & type, const size_t & entry) const {
  
  if (boundaryCells.size() > b) {
    
    // Get a view of the vector on the HostDevice
    
    // TMW: this all needs to be updated
    auto vec_kv = vec->getLocalView<HostDevice>();
    
    // Get a corresponding view on the AssemblyDevice
    
    Kokkos::View<LO***,AssemblyDevice> index;
    Kokkos::View<LO*,UnifiedDevice> numDOF;
    Kokkos::View<ScalarT***,AssemblyDevice> data;
    
    for (size_t c=0; c < boundaryCells[b].size(); c++) {
      if (boundaryCells[b][c]->numElem > 0) {
        
        switch(type) {
          case 0 :
            index = boundaryCells[b][c]->index;
            numDOF = boundaryCells[b][c]->cellData->numDOF;
            data = boundaryCells[b][c]->u;
            break;
          case 1 :
            //index = boundaryCells[b][c]->index;
            //numDOF = boundaryCells[b][c]->cellData->numDOF;
            //data = boundaryCells[b][c]->u_dot;
            break;
          case 2 :
            index = boundaryCells[b][c]->index;
            numDOF = boundaryCells[b][c]->cellData->numDOF;
            data = boundaryCells[b][c]->phi;
            break;
          case 3 :
            //index = boundaryCells[b][c]->index;
            //numDOF = boundaryCells[b][c]->cellData->numDOF;
            //data = boundaryCells[b][c]->phi_dot;
            break;
          case 4:
            index = boundaryCells[b][c]->paramindex;
            numDOF = boundaryCells[b][c]->cellData->numParamDOF;
            data = boundaryCells[b][c]->param;
            break;
          case 5 :
            index = boundaryCells[b][c]->auxindex;
            numDOF = boundaryCells[b][c]->cellData->numAuxDOF;
            data = boundaryCells[b][c]->aux;
            break;
          default :
            cout << "ERROR - NOTHING WAS GATHERED" << endl;
        }
        
        parallel_for(RangePolicy<AssemblyExec>(0,index.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_t n=0; n<index.extent(1); n++) {
            for(size_t i=0; i<numDOF(n); i++ ) {
              data(e,n,i) = vec_kv(index(e,n,i),entry);
            }
          }
        });
      }
    }
  }
  
}

// ========================================================================================
//
// ========================================================================================

void SubGridFEM::updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) {
  for (size_t b=0; b<localData.size(); b++) {
    for (size_t e=0; e<cells[0].size(); e++) {
      int numElem = cells[0][e]->numElem;
      for (int c=0; c<numElem; c++) {
        int cnode = localData[b]->cell_data_seed[c];
        for (int i=0; i<9; i++) {
          cells[0][e]->cell_data(c,i) = rotation_data(cnode,i);
        }
      }
    }
  }
}

// ========================================================================================
//
// ========================================================================================

void SubGridFEM::updateLocalData(const int & usernum) {
  
  wkset[0]->var_bcs = localData[usernum]->bcs;
  
  for (size_t e=0; e<cells[0].size(); e++) {
    cells[0][e]->nodes = localData[usernum]->nodes;
    cells[0][e]->ip = localData[usernum]->ip;
    cells[0][e]->wts = localData[usernum]->wts;
    cells[0][e]->hsize = localData[usernum]->hsize;
    cells[0][e]->basis = localData[usernum]->basis;
    cells[0][e]->basis_grad = localData[usernum]->basis_grad;
    cells[0][e]->basis_div = localData[usernum]->basis_div;
    cells[0][e]->basis_curl = localData[usernum]->basis_curl;
    //cells[0][e]->jacobian = localData[usernum]->jacobian;
    //cells[0][e]->jacobianInv = localData[usernum]->jacobianInv;
    //cells[0][e]->jacobianDet = localData[usernum]->jacobianDet;
    cells[0][e]->sideinfo = localData[usernum]->sideinfo;
    cells[0][e]->cell_data = localData[usernum]->cell_data;
  }
  
  for (size_t e=0; e<boundaryCells[0].size(); e++) {
    //boundaryCells[0][e]->wksetBID = localData[usernum]->BIDs[e];
    boundaryCells[0][e]->nodes = localData[usernum]->boundaryNodes[e];
    boundaryCells[0][e]->sidename = localData[usernum]->boundaryNames[e];
    boundaryCells[0][e]->ip = localData[usernum]->boundaryIP[e];
    boundaryCells[0][e]->wts = localData[usernum]->boundaryWts[e];
    boundaryCells[0][e]->normals = localData[usernum]->boundaryNormals[e];
    boundaryCells[0][e]->hsize = localData[usernum]->boundaryHsize[e];
    boundaryCells[0][e]->basis = localData[usernum]->boundaryBasis[e];
    boundaryCells[0][e]->basis_grad = localData[usernum]->boundaryBasisGrad[e];
    
    boundaryCells[0][e]->addAuxDiscretization(macro_basis_pointers,
                                              localData[usernum]->aux_side_basis[e],
                                              localData[usernum]->aux_side_basis_grad[e]);
    
    //boundaryCells[0][e]->setAuxIndex(localData[usernum]->macroindex);
    //boundaryCells[0][e]->auxGIDs = localData[usernum]->macroGIDs;
    boundaryCells[0][e]->setAuxIndex(localData[usernum]->boundaryMacroindex[e]);
    boundaryCells[0][e]->auxGIDs = localData[usernum]->boundaryMacroGIDs[e];
    boundaryCells[0][e]->auxMIDs = localData[usernum]->boundaryMIDs[e];
  }
  
}
