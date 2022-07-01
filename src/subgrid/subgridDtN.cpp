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

#include "subgridDtN.hpp"

using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

SubGridDtN::SubGridDtN(const Teuchos::RCP<MpiComm> & LocalComm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                       topo_RCP & macro_cellTopo_) :
settings(settings_), macro_cellTopo(macro_cellTopo_) {
  
  name = settings->name();
  LocalComm = LocalComm_;
  verbosity = settings->get<int>("verbosity",0);
  debug_level = settings->get<int>("debug level",0);
  dimension = settings->sublist("Mesh").get<int>("dimension",2);
  multiscale_method = settings->get<string>("multiscale method","mortar");
  mesh_type = settings->sublist("Mesh").get<string>("mesh type","inline"); // or "Exodus" or "panzer"
  mesh_file = settings->sublist("Mesh").get<string>("mesh file","mesh.exo"); // or "Exodus" or "panzer"
  numrefine = settings->sublist("Mesh").get<int>("refinements",0);
  shape = settings->sublist("Mesh").get<string>("element type","quad");
  macroshape = settings->sublist("Mesh").get<string>("macro element type","quad");
  time_steps = settings->sublist("Solver").get<int>("number of steps",1);
  initial_time = settings->sublist("Solver").get<ScalarT>("initial time",0.0);
  final_time = settings->sublist("Solver").get<ScalarT>("final time",1.0);
  isSynchronous = settings->sublist("Solver").get<bool>("synchronous time stepping",true);
  
  string solver = settings->sublist("Solver").get<string>("solver","steady-state");
  if (solver == "steady-state") {
    final_time = 0.0;
  }
  soln = Teuchos::rcp(new SolutionStorage<SubgridSolverNode>(settings));
  adjsoln = Teuchos::rcp(new SolutionStorage<SubgridSolverNode>(settings));
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid physics
  /////////////////////////////////////////////////////////////////////////////////////
  
  if (settings->isParameter("Functions input file")) {
    std::string filename = settings->get<std::string>("Functions input file");
    std::ifstream fn(filename.c_str());
    if (fn.good()) {
      Teuchos::RCP<Teuchos::ParameterList> functions_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
      Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*functions_parlist) );
      settings->setParameters( *functions_parlist );
    }
    else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
      TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MrHyDE could not find the functions input file: " + filename);
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
    have_rotations = settings->sublist("Mesh").get<bool>("have mesh data rotations",false);
    have_multiple_data_files = settings->sublist("Mesh").get<bool>("have multiple mesh data files",false);
    number_mesh_data_files = settings->sublist("Mesh").get<int>("number mesh data files",1);
  }
  
  compute_mesh_data = settings->sublist("Mesh").get<bool>("compute mesh data",false);
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

int SubGridDtN::addMacro(DRV & macronodes_,
                         Kokkos::View<int****,HostDevice> & macrosideinfo_,
                         LIDView macroLIDs_,
                         Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & macroorientation_) {
  
  Teuchos::TimeMonitor localmeshtimer(*sgfemTotalAddMacroTimer);
  
  Teuchos::RCP<SubGridMacroData> newdata = Teuchos::rcp( new SubGridMacroData(macronodes_,
                                                                              macrosideinfo_,
                                                                              macroLIDs_,
                                                                              macroorientation_) );
  macroData.push_back(newdata);
  
  int mID = macroData.size()-1;
  return mID;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void SubGridDtN::setUpSubgridModels() {
  
  Teuchos::TimeMonitor subgridsetuptimer(*sgfemTotalSetUpTimer);
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Starting SubGridDtN::setupSubgridModels ..." << endl;
    }
  }
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid mesh
  /////////////////////////////////////////////////////////////////////////////////////
 
  string blockID = "eblock";
  
  Kokkos::View<ScalarT**,HostDevice> nodes;
  vector<vector<GO> > connectivity;
  Kokkos::View<int****,HostDevice> sideinfo;
  
  vector<string> eBlocks;
  Teuchos::RCP<DiscretizationInterface> tmp_disc = Teuchos::rcp(new DiscretizationInterface());
  
  DRV refnodes = tmp_disc->getReferenceNodes(macro_cellTopo);
  
  SubGridTools sgt(LocalComm, macroshape, shape, refnodes,
                   macroData[0]->macrosideinfo, mesh_type, mesh_file);
  
  {
    Teuchos::TimeMonitor localmeshtimer(*sgfemSubMeshTimer);
    
    sgt.createSubMesh(numrefine);
    
    Teuchos::RCP<DiscretizationInterface> tmp_disc = Teuchos::rcp( new DiscretizationInterface());
    nodes = sgt.getListOfPhysicalNodes(macroData[0]->macronodes, macro_cellTopo, tmp_disc);
    
    int reps = macroData[0]->macronodes.extent(0);
    connectivity = sgt.getPhysicalConnectivity(reps);
    
    sideinfo = sgt.getPhysicalSideinfo(macroData[0]->macrosideinfo);
    
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
    
    sub_mesh = Teuchos::rcp(new MeshInterface(settings, LocalComm) );
    sub_mesh->stk_mesh = mesh;
    if (debug_level > 1) {
      if (LocalComm->getRank() == 0) {
        mesh->printMetaData(std::cout);
      }
    }
    
  }
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Define the sub-grid physics
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_physics = Teuchos::rcp( new PhysicsInterface(settings, LocalComm, sub_mesh->stk_mesh) );
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the subgrid discretizations
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_disc = Teuchos::rcp( new DiscretizationInterface(settings, LocalComm, sub_mesh->stk_mesh, sub_physics) );
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the function managers
  /////////////////////////////////////////////////////////////////////////////////////
  
  // Note that the workset size is determined by the number of elements per macro-element
  // times te number of macro-elements
  size_t numSubElem = connectivity.size();
  
  settings->sublist("Solver").set<int>("workset size",(int)numSubElem);
  vector<Teuchos::RCP<FunctionManager> > functionManagers;
  functionManagers.push_back(Teuchos::rcp(new FunctionManager(blockID,
                                                              numSubElem,
                                                              sub_disc->numip[0],
                                                              sub_disc->numip_side[0])));
  
  
  ////////////////////////////////////////////////////////////////////////////////
  // Define the functions on each block
  ////////////////////////////////////////////////////////////////////////////////
  
  sub_physics->sidenames = macrosidenames;
  sub_physics->defineFunctions(functionManagers);
  
  ////////////////////////////////////////////////////////////////////////////////
  // The DOF-manager needs to be aware of the physics and the discretization(s)
  ////////////////////////////////////////////////////////////////////////////////
  
  //Teuchos::RCP<panzer::DOFManager> DOF = sub_disc->buildDOF(sub_mesh->stk_mesh,
  //                                                          sub_physics->varlist,
  //                                                          sub_physics->types,
  //                                                          sub_physics->orders,
  //                                                          sub_physics->useDG);
  
  //sub_physics->setBCData(settings, sub_mesh->stk_mesh, DOF, sub_disc->cards);
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Set up the parameter manager, the assembler and the solver
  /////////////////////////////////////////////////////////////////////////////////////
  
  sub_params = Teuchos::rcp( new ParameterManager<SubgridSolverNode>(LocalComm, settings, sub_mesh->stk_mesh,
                                                                     sub_physics, sub_disc));
  
  sub_assembler = Teuchos::rcp( new AssemblyManager<SubgridSolverNode>(LocalComm, settings, sub_mesh,
                                                                       sub_disc, sub_physics, sub_params));
  
  sub_assembler->allocateGroupStorage();
  
  groups = sub_assembler->groups;
  
  Teuchos::RCP<GroupMetaData> groupData = sub_assembler->groupData[0];
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Boundary groups are not set up properly due to the lack of side sets in the subgrid mesh
  // These just need to be defined once though
  /////////////////////////////////////////////////////////////////////////////////////
  
  int numNodesPerElem = sub_mesh->cellTopo[0]->getNodeCount();
  vector<Teuchos::RCP<BoundaryGroup> > newbgroups;
  
  //int numLocalBoundaries = macroData[0]->macrosideinfo.extent(2);
  
  vector<int> unique_sides;
  vector<int> unique_local_sides;
  vector<string> unique_names;
  vector<vector<size_t> > elem_groups;
  
  sgt.getUniqueSides(sideinfo, unique_sides, unique_local_sides, unique_names,
                     macrosidenames, elem_groups);
  
  vector<stk::mesh::Entity> stk_meshElems;
  sub_mesh->stk_mesh->getMyElements(blockID, stk_meshElems);
  
  // Does need to be PHX::Device
  Kokkos::View<const LO**,Kokkos::LayoutRight, PHX::Device> LIDs = sub_disc->DOF_LIDs[0];//->getLIDs(); // hard coded
    
  for (size_t s=0; s<unique_sides.size(); s++) {
    
    string sidename = unique_names[s];
    vector<size_t> cgroup = elem_groups[s];
    
    size_t prog = 0;
    while (prog < cgroup.size()) {
      size_t currElem = numSubElem;  // Avoid faults in last iteration
      if (prog+currElem > cgroup.size()){
        currElem = cgroup.size()-prog;
      }
      Kokkos::View<int*,AssemblyDevice> eIndex("element indices",currElem);
      LO sideIndex = unique_local_sides[s];
      DRV currnodes("currnodes", currElem, numNodesPerElem, dimension);
      auto host_eIndex = Kokkos::create_mirror_view(eIndex); // mirror on host
      Kokkos::View<int*,HostDevice> host_eIndex2("element indices",currElem);
      auto host_currnodes = Kokkos::create_mirror_view(currnodes); // mirror on host
      for (size_t e=0; e<currElem; e++) {
        host_eIndex(e) = cgroup[e+prog];
        for (int n=0; n<numNodesPerElem; n++) {
          for (int m=0; m<dimension; m++) {
            host_currnodes(e,n,m) = nodes(connectivity[host_eIndex(e)][n],m);
          }
        }
      }
      int sideID = s;
     
      Kokkos::deep_copy(currnodes,host_currnodes);
      Kokkos::deep_copy(eIndex,host_eIndex);
      Kokkos::deep_copy(host_eIndex2,host_eIndex);
      
      // Build the Kokkos View of the cell LIDs ------
      vector<LIDView> vLIDs;
      LIDView cellLIDs("LIDs on device", currElem,LIDs.extent(1));
      parallel_for("assembly copy LIDs",
                   RangePolicy<AssemblyExec>(0,cellLIDs.extent(0)), 
                   KOKKOS_LAMBDA (const int i ) {
        size_t elemID = eIndex(i);
        for (size_type j=0; j<LIDs.extent(1); j++) {
          cellLIDs(i,j) = LIDs(elemID,j);
        }
      });
      vLIDs.push_back(cellLIDs);
      
      //-----------------------------------------------
      // Set the side information (soon to be removed)-
      vector<Kokkos::View<int****,HostDevice> > sideinfo;
      sideinfo.push_back(sub_disc->getSideInfo(0,0,host_eIndex2));
      
      //-----------------------------------------------
      // Set the cell orientation ---
      Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices",
                                                                      currElem, numNodesPerElem);
      auto host_currind = Kokkos::create_mirror_view(currind);
      for (size_t i=0; i<currElem; i++) {
        vector<stk::mesh::EntityId> stk_nodeids;
        size_t elemID = host_eIndex(i);
        sub_mesh->stk_mesh->getNodeIdsForElement(stk_meshElems[elemID], stk_nodeids);
        for (int n=0; n<numNodesPerElem; n++) {
          host_currind(i,n) = stk_nodeids[n];
        }
      }
      Kokkos::deep_copy(currind, host_currind);
      
      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orient_drv("kv to orients",currElem);
      Intrepid2::OrientationTools<AssemblyDevice>::getOrientation(orient_drv, currind, *(sub_mesh->cellTopo[0]));
      
      newbgroups.push_back(Teuchos::rcp(new BoundaryGroup(groupData,currnodes,eIndex,sideIndex,
                                                          sideID, sidename, newbgroups.size(),
                                                          sub_disc, true)));
      newbgroups[newbgroups.size()-1]->LIDs = vLIDs;
      newbgroups[newbgroups.size()-1]->createHostLIDs();
      newbgroups[newbgroups.size()-1]->sideinfo = sideinfo;
      
      prog += currElem;
    }
    
  }
  
  boundary_groups.push_back(newbgroups);
  
  sub_assembler->boundary_groups = boundary_groups;

  Kokkos::View<string**,HostDevice> currbcs("boundary conditions",
                                            sideinfo.extent(1),
                                            macroData[0]->macrosideinfo.extent(2));
  for (size_t i=0; i<sideinfo.extent(1); i++) { // number of variables
    for (size_t j=0; j<macroData[0]->macrosideinfo.extent(2); j++) { // number of sides per element
      currbcs(i,j) = "interface";
    }
  }
  
  for (size_t c=0; c<sideinfo.extent(0); c++) {
    for (size_t i=0; i<sideinfo.extent(1); i++) { // number of variables
      for (size_t j=0; j<sideinfo.extent(2); j++) { // number of sides per element
        if (sideinfo(c,i,j,0) > 1) { // TMW: should != 5
          for (size_t p=0; p<unique_sides.size(); p++) {
            if (unique_sides[p] == sideinfo(c,i,j,1)) {
              if (sideinfo(c,i,j,0) == 1) {
                currbcs(i,p) = "Dirichlet";
              }
              else if (sideinfo(c,i,j,0) == 2) {
                currbcs(i,p) = "Neumann";
              }
              else if (sideinfo(c,i,j,0) == 4) {
                currbcs(i,p) = "weak Dirichlet";
              }
              else if (sideinfo(c,i,j,0) == 5) {
                currbcs(i,p) = "interface";
              }
              else if (sideinfo(c,i,j,0) == 6) {
                currbcs(i,p) = "Far-field";
              }
              else if (sideinfo(c,i,j,0) == 7) {
                currbcs(i,p) = "Slip";
              }
              //currbcs(i,p) = sideinfo(c,i,j,0);
            }
          }
        }
      }
    }
  }
  macroData[0]->bcs = currbcs;

  size_t numMacroDOF = macroData[0]->macroLIDs.extent(1);
  sub_solver = Teuchos::rcp( new SubGridDtN_Solver(LocalComm, settings, sub_mesh, sub_disc, sub_physics,
                                                   sub_assembler, sub_params, numMacroDOF) );
  
  sub_postproc = Teuchos::rcp( new PostprocessManager<SubgridSolverNode>(LocalComm, settings, sub_mesh,
                                                                         sub_disc, sub_physics,
                                                                         functionManagers, sub_assembler) );
  
  
  sub_assembler->allocateGroupStorage();
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Create a subgrid function mananger
  /////////////////////////////////////////////////////////////////////////////////////
  
  {
    varlist = sub_physics->varlist[0][0];
    functionManagers[0]->setupLists(macro_paramnames, macro_disc_paramnames);
    sub_assembler->wkset[0]->params_AD = paramvals_KVAD;
    
    functionManagers[0]->wkset = sub_assembler->wkset[0];
    
    //functionManagers[0]->validateFunctions();
    functionManagers[0]->decomposeFunctions();
  }
  
  wkset = sub_assembler->wkset;
  
  wkset[0]->addAux(macro_varlist, macro_offsets);
  Kokkos::View<int*,HostDevice> macro_numDOF_host("aux DOF on host",macro_numDOF.extent(0));
  auto macro_numDOF_m = Kokkos::create_mirror_view(macro_numDOF);
  Kokkos::deep_copy(macro_numDOF_m, macro_numDOF);
  Kokkos::deep_copy(macro_numDOF_host,macro_numDOF_m);
  
  for(size_t grp=0; grp<boundary_groups[0].size(); ++grp) {
    boundary_groups[0][grp]->addAuxVars(macro_varlist);
    boundary_groups[0][grp]->groupData->numAuxDOF = macro_numDOF;
    boundary_groups[0][grp]->groupData->numAuxDOF_host = macro_numDOF_host;
    //boundary_groups[0][grp]->groupData->numAuxDOF = macro_numDOF;
    boundary_groups[0][grp]->setAuxUseBasis(macro_usebasis);
    boundary_groups[0][grp]->auxoffsets = macro_offsets;
    boundary_groups[0][grp]->wkset = wkset[0];
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
  
  for (size_t mindex = 0; mindex<macroData.size(); mindex++) {
 
    // Define the groups and boundary groups for mindex>0
    if (mindex > 0) {
      
      // Use the subgrid mesh interface to define new nodes
      DRV newnodes = sgt.getPhysicalNodes(macroData[mindex]->macronodes,
                                          macro_cellTopo, sub_disc);
      
      //DRV newnodes = sgt.getNewNodes(macroData[mindex]->macronodes);
      
      int sprog = 0;
      
      // Redefine the sideinfo for the subgroups
      Kokkos::View<int****,HostDevice> newsideinfo = sgt.getPhysicalSideinfo(macroData[mindex]->macrosideinfo);
      
      Kokkos::View<int****,HostDevice> subsideinfo("subcell side info", newsideinfo.extent(0), newsideinfo.extent(1),
                                                   newsideinfo.extent(2), newsideinfo.extent(3));
      
      for (size_t c=0; c<newsideinfo.extent(0); c++) { // number of elem in cell
        for (size_t i=0; i<newsideinfo.extent(1); i++) { // number of variables
          for (size_t j=0; j<newsideinfo.extent(2); j++) { // number of sides per element
            for (size_t k=0; k<newsideinfo.extent(3); k++) { // boundary information
              subsideinfo(c,i,j,k) = newsideinfo(sprog,i,j,k);
            }
            // TODO what is the significance of this?
            if (subsideinfo(c,i,j,0) == 1) {
              subsideinfo(c,i,j,0) = 5;
              subsideinfo(c,i,j,1) = -1;
            }
          }
        }
        sprog += 1;
        
      }
      
      vector<Kokkos::View<int****,HostDevice> > vsideinfo;
      vsideinfo.push_back(subsideinfo);
      
      ///////////////////////////////////////////////////////////
      // New groups
      ///////////////////////////////////////////////////////////
      
      vector<Teuchos::RCP<Group> > newgroups;
      int numElem = newnodes.extent(0);
      int maxElem = groups[0][0]->numElem;
      
      Kokkos::View<LO*,AssemblyDevice> localID;
      LIDView LIDs;
      
      if (numElem == maxElem) { // reuse if possible
        localID = groups[0][0]->localElemID;
        LIDs = groups[0][0]->LIDs[0];
      }
      else { // subviews do not work, so performing a deep copy (should only be on last cell)
        localID = Kokkos::View<LO*,AssemblyDevice>("local elem ids",numElem);
        LIDs = LIDView("LIDs",numElem,groups[0][0]->LIDs[0].extent(1));
        Kokkos::View<LO*,AssemblyDevice> localID_0 = groups[0][0]->localElemID;
        LIDView LIDs_0 = groups[0][0]->LIDs[0];
        parallel_for("subgrid LIDs",
                     RangePolicy<AssemblyExec>(0,numElem),
                     KOKKOS_LAMBDA (const int e ) {
          localID(e) = localID_0(e);
          for (size_t j=0; j<LIDs_0.extent(1); j++) {
            LIDs(e,j) = LIDs_0(e,j);
          }
        });
          
      }
      
      vector<LIDView> vLIDs;
      vLIDs.push_back(LIDs);
      
      newgroups.push_back(Teuchos::rcp(new Group(sub_assembler->groupData[0],
                                                 newnodes, localID,
                                                 sub_disc, true)));
      
      newgroups[newgroups.size()-1]->LIDs = vLIDs;
      newgroups[newgroups.size()-1]->createHostLIDs();
      newgroups[newgroups.size()-1]->sideinfo = vsideinfo;
      newgroups[newgroups.size()-1]->computeBasis(true);
      
      //////////////////////////////////////////////////////////////
      // New boundary groups (more complicated than interior groups)
      //////////////////////////////////////////////////////////////
      
      vector<int> unique_sides;
      vector<int> unique_local_sides;
      vector<string> unique_names;
      vector<vector<size_t> > elem_groups;
      
      sgt.getUniqueSides(subsideinfo, unique_sides, unique_local_sides, unique_names,
                         macrosidenames, elem_groups);
      
      vector<Teuchos::RCP<BoundaryGroup> > newbgroups;
      for (size_t s=0; s<unique_sides.size(); s++) {
        
        vector<size_t> cgroup = elem_groups[s];
        Kokkos::View<size_t*,AssemblyDevice> group_KV("group members on device",cgroup.size());
        auto group_KV_host = Kokkos::create_mirror_view(group_KV);
        for (size_t e=0; e<cgroup.size(); e++) {
          group_KV_host(e) = cgroup[e];
        }
        Kokkos::deep_copy(group_KV, group_KV_host);
        
        DRV currnodes("currnodes", cgroup.size(), numNodesPerElem, dimension);
        
        parallel_for("subgrid bcell group",
                     RangePolicy<AssemblyExec>(0,currnodes.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          size_t eIndex = group_KV(e);
          for (size_type n=0; n<currnodes.extent(1); n++) {
            for (size_type m=0; m<currnodes.extent(2); m++) {
              currnodes(e,n,m) = newnodes(eIndex,n,m);
            }
          }
        });
        
        int numElem = currnodes.extent(0);
        int maxElem = boundary_groups[0][s]->numElem;
        
        Kokkos::View<LO*,AssemblyDevice> localID;
        LIDView LIDs;
        Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation;
        LO sideID;
        int sidenum = boundary_groups[0][s]->sidenum;
        
        if (numElem == maxElem) { // reuse if possible
          localID = boundary_groups[0][s]->localElemID;
          LIDs = boundary_groups[0][s]->LIDs[0];
          orientation = boundary_groups[0][s]->orientation;
          sideID = boundary_groups[0][s]->localSideID;
        }
        else { // subviews do not work, so performing a deep copy (should only be on last cell)
          localID = Kokkos::View<LO*,AssemblyDevice>("local elem ids",numElem);
          LIDs = LIDView("LIDs",numElem,boundary_groups[0][s]->LIDs[0].extent(1));
          orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("orientation",numElem);
          
          Kokkos::View<LO*,AssemblyDevice> localID_0 = boundary_groups[0][s]->localElemID;
          LIDView LIDs_0 = boundary_groups[0][s]->LIDs[0];
          Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation_0 = boundary_groups[0][s]->orientation;
          sideID = boundary_groups[0][s]->localSideID;
          
          parallel_for("subgrid LIDs",
                       RangePolicy<AssemblyExec>(0,numElem),
                       KOKKOS_LAMBDA (const int e ) {
            localID(e) = localID_0(e);
            orientation(e) = orientation_0(e);
            //sideID(e) = sideID_0(e);
            for (size_type j=0; j<LIDs_0.extent(1); j++) {
              LIDs(e,j) = LIDs_0(e,j);
            }
          });
          
        }
                
        vector<LIDView> vLIDs;
        vLIDs.push_back(LIDs);
        
        newbgroups.push_back(Teuchos::rcp(new BoundaryGroup(sub_assembler->groupData[0], currnodes,
                                                            localID, sideID, sidenum, unique_names[s],
                                                            newbgroups.size(), sub_disc, true)));
        
        newbgroups[newbgroups.size()-1]->LIDs = vLIDs;
        newbgroups[newbgroups.size()-1]->createHostLIDs();
        newbgroups[newbgroups.size()-1]->sideinfo = vsideinfo;
        newbgroups[newbgroups.size()-1]->computeBasis(true);
        
        newbgroups[s]->addAuxVars(macro_varlist);
        newbgroups[s]->groupData->numAuxDOF = macro_numDOF;
        newbgroups[s]->setAuxUseBasis(macro_usebasis);
        newbgroups[s]->auxoffsets = macro_offsets;
        newbgroups[s]->wkset = wkset[0];
        
      }
      
      groups.push_back(newgroups);
      
      boundary_groups.push_back(newbgroups);
      
      Kokkos::View<string**,HostDevice> currbcs("boundary conditions",
                                             subsideinfo.extent(1),
                                             macroData[mindex]->macrosideinfo.extent(2));
      for (size_t i=0; i<subsideinfo.extent(1); i++) { // number of variables
        for (size_t j=0; j<macroData[mindex]->macrosideinfo.extent(2); j++) { // number of sides per element
          currbcs(i,j) = "interface";
        }
      }
      for (size_type c=0; c<subsideinfo.extent(0); c++) {
        for (size_type i=0; i<subsideinfo.extent(1); i++) { // number of variables
          for (size_type j=0; j<subsideinfo.extent(2); j++) { // number of sides per element
            if (subsideinfo(c,i,j,0) > 1) { // TMW: should != 5
              for (size_t p=0; p<unique_sides.size(); p++) {
                if (unique_sides[p] == subsideinfo(c,i,j,1)) {
                  if (subsideinfo(c,i,j,0) == 1) {
                    currbcs(i,p) = "Dirichlet";
                  }
                  else if (subsideinfo(c,i,j,0) == 2) {
                    currbcs(i,p) = "Neumann";
                  }
                  else if (subsideinfo(c,i,j,0) == 4) {
                    currbcs(i,p) = "weak Dirichlet";
                  }
                  else if (subsideinfo(c,i,j,0) == 5) {
                    currbcs(i,p) = "interface";
                  }
                  else if (subsideinfo(c,i,j,0) == 6) {
                    currbcs(i,p) = "Far-field";
                  }
                  else if (subsideinfo(c,i,j,0) == 7) {
                    currbcs(i,p) = "Slip";
                  }
                }
              }
            }
          }
        }
      }
      macroData[mindex]->bcs = currbcs;
      
      for (size_t grp=0; grp<groups[mindex].size(); ++grp) {
        groups[mindex][grp]->setWorkset(sub_assembler->wkset[0]);
        groups[mindex][grp]->setUseBasis(sub_solver->solver->useBasis[0],
                                      sub_solver->solver->maxnumsteps,
                                      sub_solver->solver->maxnumstages);
        groups[mindex][grp]->setUpAdjointPrev(sub_solver->solver->maxnumsteps,
                                           sub_solver->solver->maxnumstages);
        groups[mindex][grp]->setUpSubGradient(sub_solver->solver->params->num_active_params);
      }
      if (boundary_groups.size() > mindex) { // should always be true here
        for (size_t grp=0; grp<boundary_groups[mindex].size(); ++grp) {
          if (boundary_groups[mindex][grp]->numElem > 0) {
            boundary_groups[mindex][grp]->setWorkset(sub_assembler->wkset[0]);
            boundary_groups[mindex][grp]->setUseBasis(sub_solver->solver->useBasis[0],
                                                  sub_solver->solver->maxnumsteps,
                                                  sub_solver->solver->maxnumstages);
          }
        }
      }
      
    } // end if mindex > 0
    
    macroData[mindex]->setMacroIDs(groups[mindex][0]->numElem);
    
    // For all groups, define the macro basis functions at subgrid ip
    for (size_t grp=0; grp<boundary_groups[mindex].size(); ++grp) {
      vector<size_t> mIDs;
      Kokkos::View<size_t*,AssemblyDevice> mID_dev("mID device",boundary_groups[mindex][grp]->localElemID.extent(0));
      auto mID_host = Kokkos::create_mirror_view(mID_dev);
      auto localEID = boundary_groups[mindex][grp]->localElemID;
      auto macroIDs = macroData[mindex]->macroIDs;
      parallel_for("subgrid bcell mIDs",RangePolicy<AssemblyExec>(0,mID_dev.extent(0)), KOKKOS_LAMBDA (const int e ) {
        mID_dev(e) = macroIDs(localEID(e));
      });
      Kokkos::deep_copy(mID_host,mID_dev);
      for (size_type c=0; c<mID_host.extent(0); c++) {
        mIDs.push_back(mID_host(c));
      }
      boundary_groups[mindex][grp]->auxMIDs = mIDs;
      boundary_groups[mindex][grp]->auxMIDs_dev = mID_dev;
      size_t numElem = boundary_groups[mindex][grp]->numElem;
      // define the macro LIDs
      LIDView cLIDs("boundary macro LIDs",numElem,
                    macroData[mindex]->macroLIDs.extent(1));
      auto cLIDs_host = Kokkos::create_mirror_view(cLIDs);
      auto macroLIDs_host = Kokkos::create_mirror_view(macroData[mindex]->macroLIDs);
      Kokkos::deep_copy(macroLIDs_host,macroData[mindex]->macroLIDs);
      for (size_t c=0; c<numElem; c++) {
        size_t mid = mIDs[c];
        for (size_type i=0; i<cLIDs.extent(1); i++) {
          cLIDs_host(c,i) = macroLIDs_host(mid,i);
        }
      }
      Kokkos::deep_copy(cLIDs,cLIDs_host);
      boundary_groups[mindex][grp]->auxLIDs = cLIDs;
      LIDView_host cLIDs_host2("LIDs on host",cLIDs_host.extent(0),cLIDs_host.extent(1));
      Kokkos::deep_copy(cLIDs_host2,cLIDs_host);
      boundary_groups[mindex][grp]->auxLIDs_host = cLIDs_host2;
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
          // nothin yet
        }
        else {
          for (size_t grp=0; grp<boundary_groups[mindex].size(); ++grp) {
            
            size_t numElem = boundary_groups[mindex][grp]->numElem;
            
            View_Sc2 sside_ip_x, sside_ip_y, sside_ip_z;
            sside_ip_x = boundary_groups[mindex][grp]->ip[0]; // just x-component
            if (dimension>1) {
              sside_ip_y = boundary_groups[mindex][grp]->ip[1];
            }
            if (dimension>2) {
              sside_ip_z = boundary_groups[mindex][grp]->ip[2];
            }
            
            vector<DRV> currside_basis;
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV tmp_basis = DRV("basis values",numElem,macro_basis_pointers[i]->getCardinality(),sside_ip_x.extent(1));
              currside_basis.push_back(tmp_basis);
            }
            int mcount = 0;
            for (size_t c=0; c<numElem; c++) {
              size_t mID = boundary_groups[mindex][grp]->auxMIDs[c];
              if (mID == 0) {
                mcount++;
              }
            }
            vector<DRV> refbasis;
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV tmp_basis = DRV("basis values",mcount,macro_basis_pointers[i]->getCardinality(),sside_ip_x.extent(1));
              refbasis.push_back(tmp_basis);
            }
            DRV sref_side_ip("sref_side_ip", sside_ip_x.extent(1), dimension);
            DRV side_ip_e("side_ip_e",1, sside_ip_x.extent(1), dimension);
            //DRV sref_side_ip_tmp("sref_side_ip_tmp",1, sside_ip.extent(1), sside_ip.extent(2));
            DRV cnodes("tmp nodes",1,macroData[mindex]->macronodes.extent(1),
                       macroData[mindex]->macronodes.extent(2));
            
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV basisvals("basisvals", macro_basis_pointers[i]->getCardinality(), sref_side_ip.extent(0));
              DRV basisvals_Transformed("basisvals_Transformed", 1, macro_basis_pointers[i]->getCardinality(), sref_side_ip.extent(0));
              for (int c=0; c<mcount; c++) {
                auto cip = Kokkos::subview(sside_ip_x,c,Kokkos::ALL());
                auto sip = Kokkos::subview(side_ip_e,0,Kokkos::ALL(),0);
                Kokkos::deep_copy(sip,cip);
                if (dimension>1) {
                  auto cip = Kokkos::subview(sside_ip_y,c,Kokkos::ALL());
                  auto sip = Kokkos::subview(side_ip_e,0,Kokkos::ALL(),1);
                  Kokkos::deep_copy(sip,cip);
                }
                if (dimension>2) {
                  auto cip = Kokkos::subview(sside_ip_z,c,Kokkos::ALL());
                  auto sip = Kokkos::subview(side_ip_e,0,Kokkos::ALL(),2);
                  Kokkos::deep_copy(sip,cip);
                }
                
                auto mnodes = Kokkos::subview(macroData[mindex]->macronodes,0,Kokkos::ALL(),Kokkos::ALL());
                auto cnodes0 = Kokkos::subview(cnodes,0,Kokkos::ALL(), Kokkos::ALL());
                Kokkos::deep_copy(cnodes0,mnodes);
              
                DRV sref_side_ip_tmp = sub_disc->mapPointsToReference(side_ip_e,cnodes,macro_cellTopo);
                //CellTools::mapToReferenceFrame(sref_side_ip_tmp, side_ip_e, cnodes, *macro_cellTopo);
                auto sip_tmp0 = Kokkos::subview(sref_side_ip_tmp,0,Kokkos::ALL(),Kokkos::ALL());
                Kokkos::deep_copy(sref_side_ip,sip_tmp0);
                
                macro_basis_pointers[i]->getValues(basisvals, sref_side_ip, Intrepid2::OPERATOR_VALUE);
                
                //FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
                auto crefbasis = Kokkos::subview(refbasis[i],c,Kokkos::ALL(),Kokkos::ALL());
                //auto bvt0 = Kokkos::subview(basisvals_Transformed,0,Kokkos::ALL(),Kokkos::ALL());
                //Kokkos::deep_copy(crefbasis,bvt0);
                Kokkos::deep_copy(crefbasis,basisvals);
              }
            }
            int numIDs = numElem / mcount;
      
            Kokkos::View<int[1],PHX::Device> mcount_kv("view of mcount");
            Kokkos::deep_copy(mcount_kv,mcount);
            Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> corientation("tmp orientation",numElem);
            auto morient = macroData[mindex]->macroorientation;
            parallel_for("subgrid macro basis",RangePolicy<PHX::Device::execution_space>(0,numIDs), KOKKOS_LAMBDA (const int m ) {
              int mcount = mcount_kv(0);
              for (int n=0; n<mcount; n++) {
                int index= m*mcount+n;
                corientation(index) = morient(m);
              }
            });
            
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV tmp_basis("basis values",numElem,macro_basis_pointers[i]->getCardinality(),sside_ip_x.extent(1));
              auto rbasis = refbasis[i];
              parallel_for("subgrid macro basis",RangePolicy<PHX::Device::execution_space>(0,numIDs), KOKKOS_LAMBDA (const int m ) {
                int mcount = mcount_kv(0);
                for (int n=0; n<mcount; n++) {
                  int index= m*mcount+n;
                  for (size_type dof=0; dof<rbasis.extent(1); dof++) {
                    for (size_type pt=0; pt<rbasis.extent(2); pt++) {
                       tmp_basis(index,dof,pt) = rbasis(n,dof,pt);
                    }
                  }
                }
              });
     
              currside_basis[i] = sub_disc->applyOrientation(tmp_basis,corientation,macro_basis_pointers[i]);
              //OrientTools::modifyBasisByOrientation(currside_basis[i], tmp_basis,
              //                                      corientation, macro_basis_pointers[i].get());
                
            }

            boundary_groups[mindex][grp]->auxside_basis = currside_basis;
          }
        }
      }
      else {
        if (multiscale_method != "mortar" ) {
          // nothin yet
        }
        else {
          for (size_t grp=0; grp<boundary_groups[mindex].size(); ++grp) {
            boundary_groups[mindex][grp]->auxside_basis = boundary_groups[0][grp]->auxside_basis;
          }
        }
      }
    }
    
  }
  
  sub_assembler->groups = groups;
  sub_assembler->boundary_groups = boundary_groups;
  sub_physics->setWorkset(wkset);
  
  //////////////////////////////////////////////////////////////
  // Set the initial conditions
  //////////////////////////////////////////////////////////////
  
  {
    Teuchos::TimeMonitor localtimer(*sgfemSubICTimer);
    for (size_t mindex = 0; mindex<macroData.size(); mindex++) {
      
      Teuchos::RCP<SG_MultiVector> init = sub_solver->solver->linalg->getNewOverlappedVector(0);
      this->setInitial(init, mindex, false);
      prev_soln.push_back(init);
      
      Teuchos::RCP<SG_MultiVector> curr = sub_solver->solver->linalg->getNewOverlappedVector(0);
      curr->assign(*init);
      curr_soln.push_back(curr);
      
      Teuchos::RCP<SG_MultiVector> stage = sub_solver->solver->linalg->getNewOverlappedVector(0);
      stage->assign(*init);
      stage_soln.push_back(stage);
      
      soln->store(init,initial_time,mindex);
      sub_solver->performGather(mindex, init, 0, 0);
      Teuchos::RCP<SG_MultiVector> inita = sub_solver->solver->linalg->getNewOverlappedVector(0);
      adjsoln->store(inita,final_time,mindex);
    }
  }
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Finished SubGridDtN::setupSubgridModels ..." << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridDtN::finalize(const int & globalSize, const int & globalPID,
                          const bool & write_subgrid_soln, vector<string> & appends) {
  
  // globalRank and globalPID are associated with the global MPI communicator
  // only needed to define a unique output file
  
  if (macroData.size() > 0) {

    this->setUpSubgridModels();
    sub_assembler->updatePhysicsSet(0);
    //size_t defblock = 0;
    //if (groups.size() > 0) {
    //  sub_physics->setAuxVars(defblock, macro_varlist);
    //}
  }

  if (write_subgrid_soln) {
    std::stringstream ss;
    if (globalSize > 1) {
      ss << "_" << name << ".exo." << globalSize << "." << globalPID;
      combined_mesh_filename = "subgrid_output" + ss.str();
    }
    else {
      ss << "_" << name << ".exo";
      combined_mesh_filename = "subgrid_output" + ss.str();
    }
    
    this->setupCombinedExodus(appends);
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridDtN::addMeshData() {
  
  Teuchos::TimeMonitor localmeshtimer(*sgfemMeshDataTimer);
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Starting SubGridDtN::addMeshData ..." << endl;
    }
  }
  
  if (groups.size() > 0) {
    sub_mesh->setMeshData(groups, boundary_groups);
  }
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Finished SubGridDtN::addMeshData ..." << endl;
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SubGridDtN::subgridSolver(View_Sc3 coarse_fwdsoln,
                               View_Sc4 coarse_prevsoln,
                               View_Sc3 coarse_adjsoln,
                               const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                               const bool & compute_jacobian, const bool & compute_sens,
                               const int & num_active_params,
                               const bool & compute_disc_sens, const bool & compute_aux_sens,
                               workset & macrowkset,
                               const int & macrogrp, const int & macroelemindex,
                               Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) {
  
  Teuchos::TimeMonitor totalsolvertimer(*sgfemSolverTimer);
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Starting SubGridDtN::subgridSolver ..." << endl;
    }
  }
  
  // Update the groups for this macro-element (or set of macro-elements)
  this->updateLocalData(macrogrp);
  
  // Copy the locak data (look into using subviews for this)
  // Solver does not know about localData
  Kokkos::View<ScalarT***,AssemblyDevice> coarse_u("local u",groups[macrogrp][0]->numElem,
                                                   coarse_fwdsoln.extent(1),
                                                   coarse_fwdsoln.extent(2));
  Kokkos::View<ScalarT***,AssemblyDevice> coarse_phi("local phi",groups[macrogrp][0]->numElem,
                                                     coarse_adjsoln.extent(1),
                                                     coarse_adjsoln.extent(2));
  
  // TMW: update for device (subgrid or assembly?)
  // Need to move localData[]->macroIDs to a Kokkos::View on the appropriate device
  auto macroIDs = macroData[macrogrp]->macroIDs;
  parallel_for("subgrid set coarse sol",
               RangePolicy<AssemblyExec>(0,coarse_u.extent(0)),
               KOKKOS_LAMBDA (const size_type e ) {
    for (size_type i=0; i<coarse_u.extent(1); i++) {
      for (size_type j=0; j<coarse_u.extent(2); j++) {
        coarse_u(e,i,j) = coarse_fwdsoln(macroIDs(e),i,j);
      }
    }
  });
  if (isAdjoint) {
    parallel_for("subgrid set coarse adj",
                 RangePolicy<AssemblyExec>(0,coarse_phi.extent(0)),
                 KOKKOS_LAMBDA (const size_type e ) {
      for (size_type i=0; i<coarse_phi.extent(1); i++) {
        for (size_type j=0; j<coarse_phi.extent(2); j++) {
          coarse_phi(e,i,j) = coarse_adjsoln(macroIDs(e),i,j);
        }
      }
    });
  }
  
  // Extract the previous solution as the initial guess/condition for subgrid problems
  Teuchos::RCP<SG_MultiVector> prev_fwdsoln, prev_adjsoln;
  {
    Teuchos::TimeMonitor localtimer(*sgfemInitialTimer);
    
    ScalarT prev_time = 0.0; // TMW: needed?
    //size_t numtimes = soln->times[macrogrp].size();
    if (isAdjoint) {
      if (isTransient) {
        bool foundfwd = soln->extractPrevious(prev_fwdsoln, macrogrp, time, prev_time);
        bool foundadj = adjsoln->extract(prev_adjsoln, macrogrp, time);
        if (!foundfwd || !foundadj) {
          // throw error
        }
      }
      else {
        bool foundfwd = soln->extract(prev_fwdsoln, macrogrp, time);
        bool foundadj = adjsoln->extract(prev_adjsoln, macrogrp, time);
        if (!foundfwd || !foundadj) {
          // throw error
        }
      }
    }
    else { // forward or compute sens
      if (isTransient) {
        bool foundfwd = soln->extractPrevious(prev_fwdsoln, macrogrp, time, prev_time);
        if (!foundfwd) { // this subgrid has not been solved at this time yet
          foundfwd = soln->extractLast(prev_fwdsoln, macrogrp, prev_time);
        }
      }
      else {
        bool foundfwd = soln->extractLast(prev_fwdsoln,macrogrp,prev_time);
        if (!foundfwd) {
          // throw error
        }
      }
      if (compute_sens) {
        ScalarT nexttime = 0.0;
        bool foundadj = adjsoln->extractNext(prev_adjsoln,macrogrp,time,nexttime);
        if (!foundadj) {
          // throw error
        }
      }
    }
  }
  Teuchos::RCP<SG_MultiVector> curr_adjsoln;
  
  // Containers for current forward/adjoint solutions
  //Teuchos::RCP<LA_MultiVector> curr_fwdsoln = Teuchos::rcp(new LA_MultiVector(sub_solver->solver->LA_overlapped_map,1));
  //Teuchos::RCP<LA_MultiVector> curr_adjsoln = Teuchos::rcp(new LA_MultiVector(sub_solver->solver->LA_overlapped_map,1));
  
  // Solve the local subgrid problem and fill in the coarse macrowkset->res;
  sub_solver->solve(coarse_u, coarse_prevsoln, coarse_phi,
                    prev_soln[macrogrp], curr_soln[macrogrp], stage_soln[macrogrp],
                    prev_adjsoln, curr_adjsoln,
                    //prev_fwdsoln, prev_adjsoln, //curr_fwdsoln, curr_adjsoln,
                    Psol[0],
                    time, isTransient, isAdjoint, compute_jacobian,
                    compute_sens, num_active_params, compute_disc_sens, compute_aux_sens,
                    macrowkset, macrogrp, macroelemindex, subgradient, store_adjPrev);
  
  //sub_solver->performGather(macrogrp, curr_soln[macrogrp], 0, 0);

  // Store the subgrid fwd or adj solution
  //if (isAdjoint) {
  //  adjsoln->store(sub_solver->phi,time,macrogrp);
  //}
  //else if (!compute_sens) {
  //  soln->store(sub_solver->u,time,macrogrp);
  //}
  
  if (debug_level > 0) {
    if (LocalComm->getRank() == 0) {
      cout << "**** Finished SubGridDtN::subgridSolver ..." << endl;
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Store macro-dofs and flux (for ML-based subgrid)
///////////////////////////////////////////////////////////////////////////////////////

void SubGridDtN::storeFluxData(Kokkos::View<ScalarT***,AssemblyDevice> lambda, Kokkos::View<AD**,AssemblyDevice> flux) {
  
  //int num_dof_lambda = lambda.extent(1)*lambda.extent(2);
  
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
#ifndef MrHyDE_NO_AD
    ofs << flux(e,0).val() << "  ";
#else
    ofs << flux(e,0) << "  ";
#endif
    //}
    //}
    ofs << endl;
  }
  ofs.close();
  
}
 

//////////////////////////////////////////////////////////////
// Compute the initial values for the subgrid solution
//////////////////////////////////////////////////////////////

void SubGridDtN::setInitial(Teuchos::RCP<SG_MultiVector> & initial,
                            const int & macrogrp, const bool & useadjoint) {
  
  initial->putScalar(0.0);
  // TMW: uncomment if you need a nonzero initial condition
  //      right now, it slows everything down ... especially if using an L2-projection
  
  
  bool useL2proj = sub_solver->solver->have_initial_conditions[0];//true;//settings->sublist("Solver").get<bool>("Project initial",true);

  auto glinitial = sub_solver->solver->linalg->getNewVector(0);
   
  if (useL2proj) {
   
    // Compute the L2 projection of the initial data into the discrete space
    auto rhs = sub_solver->solver->linalg->getNewOverlappedVector(0);
    auto mass = sub_solver->solver->linalg->getNewOverlappedMatrix(0);
    auto glrhs = sub_solver->solver->linalg->getNewVector(0);
    auto glmass = sub_solver->solver->linalg->getNewMatrix(0);

    sub_assembler->setInitial(0, rhs, mass, false, false, 1.0, 0, (size_t)macrogrp);
    
    sub_solver->solver->linalg->exportMatrixFromOverlapped(0,glmass, mass);
    sub_solver->solver->linalg->exportVectorFromOverlapped(0,glrhs, rhs);
    
    sub_solver->solver->linalg->fillComplete(glmass);
    sub_solver->solver->linalg->linearSolverL2(0,glmass, glrhs, glinitial);
    sub_solver->solver->linalg->importVectorToOverlapped(0,initial, glinitial);
    sub_solver->solver->linalg->resetJacobian(0);
   
   }
   //else {
   // not re-implemented yet
   //}
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the error for verification
///////////////////////////////////////////////////////////////////////////////////////

vector<std::pair<string, string> > SubGridDtN::getErrorList() {
  return sub_postproc->error_list[0];
}

///////////////////////////////////////////////////////////////////////////////////////
// These views are on the Host since we are using the postproc mananger
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT*,HostDevice> SubGridDtN::computeError(const ScalarT & time) {
  Kokkos::View<ScalarT*,HostDevice> errors;
  
  if (macroData.size() > 0) {
    
    errors = Kokkos::View<ScalarT*,HostDevice>("error", sub_postproc->error_list[0].size());
    
    bool compute = false;
    if (subgrid_static) {
      compute = true;
    }
    if (compute) {
      sub_postproc->computeError(time);
      for (size_t block=0; block<sub_postproc->errors[0].size(); ++block) {
        Kokkos::View<ScalarT*,HostDevice> cerr = sub_postproc->errors[0][block];
        for (size_t etype=0; etype<cerr.extent(0); etype++) {
          errors(etype) += cerr(etype);
        }
      }
      sub_postproc->errors.clear();
    }
  }
  
  return errors;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,HostDevice> SubGridDtN::computeError(vector<std::pair<string, string> > & sub_error_list,
                                                            const vector<ScalarT> & times) {
  
  Kokkos::View<ScalarT**,HostDevice> errors;
  if (macroData.size() > 0) {
    
    errors = Kokkos::View<ScalarT**,HostDevice>("error", times.size(), sub_postproc->error_list[0].size());
    
    
    for (size_t t=0; t<times.size(); t++) {
      bool compute = false;
      if (subgrid_static) {
        compute = true;
      }
      if (compute) {
        sub_postproc->computeError(times[t]);
        for (size_t b=0; b<sub_postproc->errors[0].size(); b++) {
          Kokkos::View<ScalarT*,HostDevice> cerr = sub_postproc->errors[0][b];
          for (size_t etype=0; etype<cerr.extent(0); etype++) {
            errors(t,etype) += cerr(etype);
          }
        }
        sub_postproc->errors.clear();
      }
    }
  }
  
  return errors;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the objective function
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD*,AssemblyDevice> SubGridDtN::computeObjective(const string & response_type, const int & seedwhat,
                                                              const ScalarT & time, const int & macrogrp) {
  
  Kokkos::View<AD*,AssemblyDevice> objective;
  /*
  int tindex = -1;
  //for (int tt=0; tt<soln[macrogrp].size(); tt++) {
  //  if (abs(soln[macrogrp][tt].first - time)<1.0e-10) {
  //    tindex = tt;
  //  }
  //}
  
  Teuchos::RCP<SG_MultiVector> currsol;
  bool found = soln->extract(currsol,macrogrp,time,tindex);
  
  if (found) {
    this->updateLocalData(macrogrp);
    bool beensized = false;
    sub_solver->performGather(0, currsol, 0,0);
    //this->performGather(macrogrp, Psol[0], 4, 0);
    
    for (size_t e=0; e<groups[0].size(); e++) {
      auto curr_obj = groups[0][e]->computeObjective(time, tindex, seedwhat);
      if (!beensized && curr_obj.extent(1)>0) {
        objective = Kokkos::View<AD*,AssemblyDevice>("objective", curr_obj.extent(1));
        beensized = true;
      }
      for (size_t c=0; c<groups[0][e]->numElem; c++) {
        for (size_type i=0; i<curr_obj.extent(1); i++) {
          objective(i) += curr_obj(c,i);
        }
      }
    }
  }
  */
  return objective;
}

///////////////////////////////////////////////////////////////////////////////////////
// Write the solution to a file
///////////////////////////////////////////////////////////////////////////////////////

void SubGridDtN::setupCombinedExodus(vector<string> & appends) {
  
  Teuchos::TimeMonitor meshsetuptimer(*sgfemCombinedMeshSetupTimer);
  
  if (macroData.size() > 0) {
    bool isTD = false;
    string solver = settings->sublist("Solver").get<string>("solver","steady-state");
    if (solver == "transient") {
      isTD = true;
    }
    
    string blockID = "eblock";
    
    //////////////////////////////////////////////////////////////
    // Create a combined subgrid mesh
    //////////////////////////////////////////////////////////////
    
    // Create an initial mesh using the first macroelem
    //DRV refnodes("nodes on reference element",macroData[0]->macronodes.extent(1), dimension);
    //CellTools::getReferenceSubcellVertices(refnodes, dimension, 0, *macro_cellTopo);
    DRV refnodes = sub_disc->getReferenceNodes(macro_cellTopo);
    
    SubGridTools sgt(LocalComm, macroshape, shape, refnodes,//macroData[0]->macronodes,
                     macroData[0]->macrosideinfo, mesh_type, mesh_file);
    sgt.createSubMesh(numrefine);
    
    size_t numRefNodes = sgt.subnodes_list.extent(0);
    size_t numTotalNodes = 0;
    for (size_t macrogrp=0; macrogrp<macroData.size(); macrogrp++) {
      for (size_t e=0; e<macroData[macrogrp]->macronodes.extent(0); e++) {
        numTotalNodes += numRefNodes;
      }
    }
    //vector<vector<ScalarT> > comb_nodes;
    Kokkos::View<ScalarT**,HostDevice> comb_nodes("combined nodes",numTotalNodes,dimension);
    vector<vector<GO> > comb_connectivity;
    size_t nprog = 0;
    for (size_t macrogrp=0; macrogrp<macroData.size(); macrogrp++) {
      //vector<vector<ScalarT> > nodes = sgt.getNodes(macroData[macrogrp]->macronodes);
      Kokkos::View<ScalarT**,HostDevice> nodes = sgt.getListOfPhysicalNodes(macroData[macrogrp]->macronodes, macro_cellTopo, sub_disc);
      for (size_type n=0; n<nodes.extent(0); n++) {
        for (int s=0; s<dimension; s++) {
          comb_nodes(nprog+n,s) = nodes(n,s);
        }
      }
      GO num_prev_nodes = nprog;
      
      nprog += nodes.extent(0);
      //GO num_prev_nodes = static_cast<GO>(comb_nodes.size());
      //for (size_t n=0; n<nodes.size(); n++) {
      //  comb_nodes.push_back(nodes[n]);
      //}
      
      int reps = macroData[macrogrp]->macronodes.extent(0);
      vector<vector<GO> > connectivity = sgt.getPhysicalConnectivity(reps);
      for (size_t c=0; c<connectivity.size(); c++) {
        vector<GO> mod_elem;
        for (size_t n=0; n<connectivity[c].size(); n++) {
          mod_elem.push_back(connectivity[c][n]+num_prev_nodes);
        }
        comb_connectivity.push_back(mod_elem);
      }
    }
    //Kokkos::View<int****,HostDevice> sideinfo = sgt.getNewSideinfo(localData[macrogrp]->macrosideinfo);
    
    //size_t numNodesPerElem = comb_connectivity[0].size();
    
    panzer_stk::SubGridMeshFactory submeshFactory(shape, comb_nodes, comb_connectivity, blockID);
    combined_mesh = submeshFactory.buildMesh(*(LocalComm->getRawMpiComm()));
    
    //////////////////////////////////////////////////////////////
    // Add in the necessary fields for plotting
    //////////////////////////////////////////////////////////////
    
    vector<string> vartypes = sub_physics->types[0][0];
    vector<string> subeBlocks;
    combined_mesh->getElementBlockNames(subeBlocks);
      
    for (size_t app=0; app<appends.size(); ++app) {
      string capp = appends[app];
      for (size_t j=0; j<sub_physics->varlist[0][0].size(); j++) {
        if (vartypes[j] == "HGRAD") {
          combined_mesh->addSolutionField(sub_physics->varlist[0][0][j]+capp, subeBlocks[0]);
        }
        else if (vartypes[j] == "HVOL"){
          combined_mesh->addCellField(sub_physics->varlist[0][0][j]+capp, subeBlocks[0]);
        }
        else if (vartypes[j] == "HDIV" || vartypes[j] == "HCURL"){
          combined_mesh->addCellField(sub_physics->varlist[0][0][j]+capp+"x", subeBlocks[0]);
          combined_mesh->addCellField(sub_physics->varlist[0][0][j]+capp+"y", subeBlocks[0]);
          combined_mesh->addCellField(sub_physics->varlist[0][0][j]+capp+"z", subeBlocks[0]);
        
          combined_mesh->addSolutionField(sub_physics->varlist[0][0][j]+capp+"x", subeBlocks[0]);
          combined_mesh->addSolutionField(sub_physics->varlist[0][0][j]+capp+"y", subeBlocks[0]);
          combined_mesh->addSolutionField(sub_physics->varlist[0][0][j]+capp+"z", subeBlocks[0]);
        }
      }
    
      Teuchos::ParameterList efields;
      if (settings->sublist("Postprocess").isSublist(subeBlocks[0])) {
        efields = settings->sublist("Postprocess").sublist(subeBlocks[0]).sublist("Extra fields");
      }
      else {
        efields = settings->sublist("Postprocess").sublist("Extra fields");
      }
      Teuchos::ParameterList::ConstIterator ef_itr = efields.begin();
      while (ef_itr != efields.end()) {
        combined_mesh->addSolutionField(ef_itr->first+capp, subeBlocks[0]);
        ef_itr++;
      }
    
      Teuchos::ParameterList ecfields;
      if (settings->sublist("Postprocess").isSublist(subeBlocks[0])) {
        ecfields = settings->sublist("Postprocess").sublist(subeBlocks[0]).sublist("Extra cell fields");
      }
      else {
        ecfields = settings->sublist("Postprocess").sublist("Extra cell fields");
      }
      Teuchos::ParameterList::ConstIterator ecf_itr = ecfields.begin();
      while (ecf_itr != ecfields.end()) {
        combined_mesh->addCellField(ecf_itr->first+capp, subeBlocks[0]);
        ecf_itr++;
      }
    
    
      // Add derived quantities from physics modules
      for (size_t j=0; j<sub_physics->modules[0][0].size(); ++j) {
        std::vector<string> derivedlist = sub_physics->modules[0][0][j]->getDerivedNames();
        for (size_t k=0; k<derivedlist.size(); ++k) {
          combined_mesh->addCellField(derivedlist[k]+capp, subeBlocks[0]);
        }
      }
    }
        
    combined_mesh->addCellField("mesh_data_seed", subeBlocks[0]);
    combined_mesh->addCellField("mesh_data", subeBlocks[0]);
    
    if (discparamnames.size() > 0) {
      for (size_t n=0; n<discparamnames.size(); n++) {
        int paramnumbasis = groups[0][0]->groupData->numParamDOF.extent(0);
        if (paramnumbasis==1) {
          combined_mesh->addCellField(discparamnames[n], subeBlocks[0]);
        }
        else {
          combined_mesh->addSolutionField(discparamnames[n], subeBlocks[0]);
        }
      }
    }
    
    //////////////////////////////////////////////////////////////
    // Finalize the mesh
    //////////////////////////////////////////////////////////////
    
    combined_mesh->initialize(*(LocalComm->getRawMpiComm()));
    submeshFactory.modifyMesh(*combined_mesh);
    combined_mesh->buildLocalElementIDs();
    
    //////////////////////////////////////////////////////////////
    // Set up the output for transient data
    //////////////////////////////////////////////////////////////
    
    if (isTD) {
      combined_mesh->setupExodusFile(combined_mesh_filename);
    }
    
  }
}

//////////////////////////////////////////////////////////////
// Write the current states to the combined output file
//////////////////////////////////////////////////////////////

void SubGridDtN::writeSolution(const ScalarT & time, const string & append) {

  Teuchos::TimeMonitor outputtimer(*sgfemCombinedMeshOutputTimer);
  
  if (macroData.size()>0) {
    
    bool isTD = false;
    string solver = settings->sublist("Solver").get<string>("solver","steady-state");
    if (solver == "transient") {
      isTD = true;
    }
    
    vector<size_t> myElements;
    size_t eprog = 0;
    for (size_t macrogrp=0; macrogrp<groups.size(); macrogrp++) {
      for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
        for (size_t p=0; p<groups[macrogrp][grp]->numElem; p++) {
          myElements.push_back(eprog);
          eprog++;
        }
      }
    }
    
    string blockID = "eblock";
    topo_RCP cellTopo = combined_mesh->getCellTopology(blockID);
    size_t numNodesPerElem = cellTopo->getNodeCount();
    
    Kokkos::View<int**,AssemblyDevice> offsets = wkset[0]->offsets;
    Kokkos::View<int*,AssemblyDevice> numDOF = sub_assembler->groupData[0]->numDOF;
    vector<string> vartypes = sub_physics->types[0][0];
    vector<string> varlist = sub_physics->varlist[0][0];
    
    // Collect the subgrid solution
    for (size_t n = 0; n<varlist.size(); n++) {
      
      if (vartypes[n] == "HGRAD") {
        Kokkos::View<ScalarT**,HostDevice> soln_computed("soln",myElements.size(), numNodesPerElem);
        
        size_t pprog = 0;
        for (size_t macrogrp=0; macrogrp<groups.size(); macrogrp++) {
          for( size_t grp=0; grp<groups[macrogrp].size(); ++grp ) {
            Kokkos::View<ScalarT***,AssemblyDevice> sol = groups[macrogrp][grp]->u[0];
            auto host_sol = Kokkos::create_mirror_view(sol);
            Kokkos::deep_copy(host_sol,sol);
            for (size_t p=0; p<groups[macrogrp][grp]->numElem; p++) {
              for( size_t i=0; i<numNodesPerElem; i++ ) {
                soln_computed(pprog,i) = host_sol(p,n,i);
              }
              pprog += 1;
            }
          }
        }
        combined_mesh->setSolutionFieldData(varlist[n]+append, blockID, myElements, soln_computed);
      }
      else if (vartypes[n] == "HVOL") {
        
        Kokkos::View<ScalarT*,HostDevice> soln_computed("soln",myElements.size());
        size_t pprog = 0;
        for( size_t macrogrp=0; macrogrp<groups.size(); macrogrp++ ) {
          for( size_t grp=0; grp<groups[macrogrp].size(); ++grp ) {
            Kokkos::View<ScalarT***,AssemblyDevice> sol = groups[macrogrp][grp]->u[0];
            auto host_sol = Kokkos::create_mirror_view(sol);
            Kokkos::deep_copy(host_sol,sol);
            for (size_t p=0; p<groups[macrogrp][grp]->numElem; p++) {
              soln_computed(pprog) = host_sol(p,n,0);
              pprog++;
            }
          }
        }
        combined_mesh->setCellFieldData(varlist[n]+append, blockID, myElements, soln_computed);
        
      }
      else if (vartypes[n] == "HDIV" || vartypes[n] == "HCURL") {
        
        Kokkos::View<ScalarT*,HostDevice> soln_x("soln",myElements.size());
        Kokkos::View<ScalarT*,HostDevice> soln_y("soln",myElements.size());
        Kokkos::View<ScalarT*,HostDevice> soln_z("soln",myElements.size());
        size_t pprog = 0;
        
        for( size_t macrogrp=0; macrogrp<groups.size(); macrogrp++ ) {
          for( size_t grp=0; grp<groups[macrogrp].size(); ++grp ) {
            Kokkos::View<ScalarT***,AssemblyDevice> sol = groups[macrogrp][grp]->u_avg[0];
            auto host_sol = Kokkos::create_mirror_view(sol);
            Kokkos::deep_copy(host_sol,sol);
            for (size_t p=0; p<groups[macrogrp][grp]->numElem; p++) {
              soln_x(pprog) = host_sol(p,n,0);
              if (dimension > 1) {
                soln_y(pprog) = host_sol(p,n,1);
              }
              if (dimension > 2) {
                soln_z(pprog) = host_sol(p,n,2);
              }
              pprog++;
            }
          }
        }
        combined_mesh->setCellFieldData(varlist[n]+append+"x", blockID, myElements, soln_x);
        combined_mesh->setCellFieldData(varlist[n]+append+"y", blockID, myElements, soln_y);
        combined_mesh->setCellFieldData(varlist[n]+append+"z", blockID, myElements, soln_z);
        
        if (sub_assembler->groupData[0]->requireBasisAtNodes) {
          Kokkos::View<ScalarT**,HostDevice> soln_nx("soln",myElements.size(), numNodesPerElem);
          Kokkos::View<ScalarT**,HostDevice> soln_ny("soln",myElements.size(), numNodesPerElem);
          Kokkos::View<ScalarT**,HostDevice> soln_nz("soln",myElements.size(), numNodesPerElem);
          
          pprog = 0;
          for (size_t macrogrp=0; macrogrp<groups.size(); macrogrp++) {
            for( size_t grp=0; grp<groups[macrogrp].size(); ++grp ) {
              Kokkos::View<ScalarT***,AssemblyDevice> sol = groups[macrogrp][grp]->getSolutionAtNodes(n);
              auto host_sol = Kokkos::create_mirror_view(sol);
              Kokkos::deep_copy(host_sol,sol);
              for (size_t p=0; p<groups[macrogrp][grp]->numElem; p++) {
                for( size_t i=0; i<numNodesPerElem; i++ ) {
                  soln_nx(pprog,i) = host_sol(p,i,0);
                  if (dimension > 1) {
                    soln_ny(pprog,i) = host_sol(p,i,1);
                  }
                  if (dimension > 2) {
                    soln_nz(pprog,i) = host_sol(p,i,2);
                  }
                }
                pprog += 1;
              }
            }
          }
          combined_mesh->setSolutionFieldData(varlist[n]+append+"x", blockID, myElements, soln_nx);
          combined_mesh->setSolutionFieldData(varlist[n]+append+"y", blockID, myElements, soln_ny);
          combined_mesh->setSolutionFieldData(varlist[n]+append+"z", blockID, myElements, soln_nz);
        }
      }
    }
    
    ////////////////////////////////////////////////////////////////
    // Mesh data
    ////////////////////////////////////////////////////////////////
    
    
    Kokkos::View<ScalarT*,HostDevice> cseeds("cell data seeds",myElements.size());
    Kokkos::View<ScalarT*,HostDevice> cdata("cell data",myElements.size());
    
    if (groups[0][0]->groupData->have_phi || groups[0][0]->groupData->have_rotation || groups[0][0]->groupData->have_extra_data) {
      int eprog = 0;
      // TMW: need to use a mirror view here
      for (size_t macrogrp=0; macrogrp<groups.size(); macrogrp++) {
        for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
          vector<size_t> data_seed = groups[macrogrp][grp]->data_seed;
          vector<size_t> data_seedindex = groups[macrogrp][grp]->data_seedindex;
          Kokkos::View<ScalarT**,AssemblyDevice> data = groups[macrogrp][grp]->data;
          for (size_t p=0; p<groups[macrogrp][0]->numElem; p++) {
            cseeds(eprog) = data_seedindex[p];
            cdata(eprog) = data(p,0);
            eprog++;
          }
        }
      }
    }
    
    combined_mesh->setCellFieldData("mesh_data_seed", blockID, myElements, cseeds);
    combined_mesh->setCellFieldData("mesh_data", blockID, myElements, cdata);
    
    
    ////////////////////////////////////////////////////////////////
    // Extra nodal fields
    ////////////////////////////////////////////////////////////////
    
    vector<string> extrafieldnames = sub_postproc->extrafields_list[0];
    for (size_t j=0; j<extrafieldnames.size(); j++) {
      Kokkos::View<ScalarT**,HostDevice> efdata("field data",myElements.size(), numNodesPerElem);
      /*
      size_t eprog = 0;
      for (size_t macrogrp=0; macrogrp<groups.size(); macrogrp++) {
        for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
          DRV nodes = groups[macrogrp][grp]->nodes;
          Kokkos::View<ScalarT**,AssemblyDevice> cfields = sub_physics->getExtraFields(0, 0, nodes, time, wkset[0]);
          auto host_cfields = Kokkos::create_mirror_view(cfields);
          Kokkos::deep_copy(host_cfields,cfields);
          for (size_t p=0; p<groups[macrogrp][grp]->numElem; p++) {
            for (size_t i=0; i<host_cfields.extent(1); i++) {
              efdata(eprog,i) = host_cfields(p,i);
            }
            eprog++;
          }
        }
      }
       */
      combined_mesh->setSolutionFieldData(extrafieldnames[j]+append, blockID, myElements, efdata);
    }
    
    ////////////////////////////////////////////////////////////////
    // Extra cell fields
    ////////////////////////////////////////////////////////////////
    
    vector<string> extracellfieldnames = sub_postproc->extracellfields_list[0];
    
    Kokkos::View<ScalarT**,HostDevice> ecd("cell data",myElements.size(),
                                           extracellfieldnames.size());
    
    eprog = 0;
    for (size_t macrogrp=0; macrogrp<groups.size(); macrogrp++) {
      for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
    
        groups[macrogrp][grp]->updateWorkset(0,true);
        wkset[0]->time = time;
               
        auto cfields = sub_postproc->getExtraCellFields(0, groups[macrogrp][grp]->wts);
      
        auto host_cfields = Kokkos::create_mirror_view(cfields);
        Kokkos::deep_copy(host_cfields, cfields);
        for (size_type p=0; p<host_cfields.extent(0); p++) {
          for (size_type r=0; r<host_cfields.extent(1); ++r) {
            ecd(eprog,r) = cfields(p,r);
          }
          eprog++;
        }
      }
    }
    //Kokkos::deep_copy(ecd, ecd_dev);
    
    for (size_t j=0; j<extracellfieldnames.size(); j++) {
      auto ccd = subview(ecd,ALL(),j);
      combined_mesh->setCellFieldData(extracellfieldnames[j]+append, blockID, myElements, ccd);
    }
    
    ////////////////////////////////////////////////////////////////
    // Derived quantities from physics modules, e.g., stress functionals
    ////////////////////////////////////////////////////////////////
    
    vector<string> dqnames = sub_postproc->derivedquantities_list[0];
    
    Kokkos::View<ScalarT**,HostDevice> dq("cell data",myElements.size(),
                                          dqnames.size());
    
    eprog = 0;
    for (size_t macrogrp=0; macrogrp<groups.size(); macrogrp++) {
      for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
    
        groups[macrogrp][grp]->updateWorkset(0,true);
        wkset[0]->time = time;
               
        auto cfields = sub_postproc->getDerivedQuantities(0, groups[macrogrp][grp]->wts);
      
        auto host_cfields = Kokkos::create_mirror_view(cfields);
        Kokkos::deep_copy(host_cfields, cfields);
        for (size_type p=0; p<host_cfields.extent(0); p++) {
          for (size_type r=0; r<host_cfields.extent(1); ++r) {
            dq(eprog,r) = cfields(p,r);
          }
          eprog++;
        }
      }
    }
    //Kokkos::deep_copy(ecd, ecd_dev);
    
    for (size_t j=0; j<dqnames.size(); j++) {
      auto ccd = subview(dq,ALL(),j);
      combined_mesh->setCellFieldData(dqnames[j]+append, blockID, myElements, ccd);
    }
    /*
    for (size_t j=0; j<extracellfieldnames.size(); j++) {
      Kokkos::View<ScalarT*,HostDevice> efdata("cell data",myElements.size());
      
      int eprog = 0;
      for (size_t macrogrp=0; macrogrp<groups.size(); macrogrp++) {
        for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
        
          groups[macrogrp][grp]->updateData();
          groups[macrogrp][grp]->updateWorksetBasis();
          wkset[0]->time = time;
          wkset[0]->computeSolnSteadySeeded(groups[macrogrp][grp]->u, 0);
          wkset[0]->computeParamSteadySeeded(groups[macrogrp][grp]->param, 0);
          wkset[0]->computeSolnVolIP();
          wkset[0]->computeParamVolIP();
          
          Kokkos::View<ScalarT*,AssemblyDevice> cfields = sub_physics->getExtraCellFields(0, j, groups[macrogrp][grp]->wts);
          
          auto host_cfields = Kokkos::create_mirror_view(cfields);
          Kokkos::deep_copy(host_cfields, cfields);
          for (size_type p=0; p<host_cfields.extent(0); p++) {
            efdata(eprog) = host_cfields(p);
            eprog++;
          }
        }
      }
      combined_mesh->setCellFieldData(extracellfieldnames[j], blockID, myElements, efdata);
    }
    */
    
    if (isTD) {
      combined_mesh->writeToExodus(time);
    }
    else {
      combined_mesh->writeToExodus(combined_mesh_filename);
    }
    
  }
  
}


////////////////////////////////////////////////////////////////////////////////
// Add in the sensor data
////////////////////////////////////////////////////////////////////////////////

void SubGridDtN::addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                            const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                            const vector<basis_RCP> & basisTypes, const int & macrogrp) {
  for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
    //groups[macrogrp][grp]->addSensors(sensor_points,sensor_loc_tol,sensor_data,
    //                              have_sensor_data, sub_disc, basisTypes, basisTypes);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
// This function needs to exist in a subgrid model, but the solver does the real work
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> >  SubGridDtN::getProjectionMatrix() {
  
  return sub_solver->getProjectionMatrix();
  
}

////////////////////////////////////////////////////////////////////////////////
// Assemble the projection (mass) matrix
// This function needs to exist in a subgrid model, but the solver does the real work
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> > SubGridDtN::getProjectionMatrix(DRV & ip, DRV & wts,
                                                           std::pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo) {
  
  return sub_solver->getProjectionMatrix(ip, wts, other_basisinfo);
  
}

////////////////////////////////////////////////////////////////////////////////
// Get an empty vector
// This function needs to exist in a subgrid model, but the solver does the real work
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> > SubGridDtN::getVector() {
  return sub_solver->getVector();
}

////////////////////////////////////////////////////////////////////////////////
// Get the integration points
////////////////////////////////////////////////////////////////////////////////

DRV SubGridDtN::getIP() {
  int numip_per_cell = wkset[0]->numip;
  int macrogrp = 0; // doesn't really matter
  int totalip = 0;
  for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
    totalip += numip_per_cell*groups[macrogrp][grp]->numElem;
  }
  
  DRV refip = DRV("refip",1,totalip,dimension);
  int prog = 0;
  for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
    size_t numElem = groups[macrogrp][grp]->numElem;
    View_Sc2 x,y,z;
    x = groups[macrogrp][grp]->ip[0];
    if (dimension>1) {
      y = groups[macrogrp][grp]->ip[1];
    }
    if (dimension>2) {
      z = groups[macrogrp][grp]->ip[2];
    }
    for (size_t c=0; c<numElem; c++) {
      for (size_type i=0; i<x.extent(1); i++) {
        refip(0,prog,0) = x(c,i);
        if (dimension>1) {
          refip(0,prog,1) = y(c,i);
        }
        if (dimension>2) {
          refip(0,prog,2) = z(c,i);
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

DRV SubGridDtN::getIPWts() {
  int numip_per_cell = wkset[0]->numip;
  int macrogrp = 0; // doesn't really matter
  int totalip = 0;
  for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
    totalip += numip_per_cell*groups[macrogrp][grp]->numElem;
  }
  DRV refwts = DRV("refwts",1,totalip);
  int prog = 0;
  for (size_t grp=0; grp<groups[macrogrp].size(); ++grp) {
    DRV wts = groups[0][grp]->groupData->ref_wts;
    size_t numElem = groups[macrogrp][grp]->numElem;
    for (size_t c=0; c<numElem; c++) {
      for (size_type i=0; i<wts.extent(0); i++) {
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

std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridDtN::evaluateBasis2(const DRV & pts) {
  
  size_t numpts = pts.extent(1);
  size_t dimpts = pts.extent(2);
  size_t numLIDs = groups[0][0]->LIDs[0].extent(1);
  Kokkos::View<int**,AssemblyDevice> owners("owners",numpts,2+numLIDs);
  
  for (size_t grp=0; grp<groups[0].size(); ++grp) {
    int numElem = groups[0][grp]->numElem;
    DRV nodes = groups[0][grp]->nodes;
    for (int c=0; c<numElem;c++) {
      DRV refpts("refpts",1, numpts, dimpts);
      //Kokkos::DynRankView<int,PHX::Device> inRefCell("inRefCell",1,numpts);
      DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
      for (unsigned int i=0; i<nodes.extent(1); i++) {
        for (unsigned int j=0; j<nodes.extent(2); j++) {
          cnodes(0,i,j) = nodes(c,i,j);
        }
      }
      
      Kokkos::DynRankView<int,PHX::Device> inRefCell = sub_disc->checkInclusionPhysicalData(pts,cnodes,
                                                                                            sub_mesh->cellTopo[0], 1.0e-12);
      //CellTools::mapToReferenceFrame(refpts, pts, cnodes, *(sub_mesh->cellTopo[0]));
      //CellTools::checkPointwiseInclusion(inRefCell, refpts, *(sub_mesh->cellTopo[0]), 1.0e-12);
      for (size_t i=0; i<numpts; i++) {
        if (inRefCell(0,i) == 1) {
          owners(i,0) = grp;
          owners(i,1) = c;
          LIDView LIDs = groups[0][grp]->LIDs[0];
          for (size_t j=0; j<numLIDs; j++) {
            owners(i,j+2) = LIDs(c,j);
          }
        }
      }
    }
  }
  
  vector<DRV> ptsBasis;
  for (size_t i=0; i<numpts; i++) {
    vector<DRV> currBasis;
    //DRV refpt_buffer("refpt_buffer",1,1,dimpts);
    DRV cpt("cpt",1,1,dimpts);
    for (size_t s=0; s<dimpts; s++) {
      cpt(0,0,s) = pts(0,i,s);
    }
    DRV nodes = groups[0][owners(i,0)]->nodes;
    DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
    for (unsigned int k=0; k<nodes.extent(1); k++) {
      for (unsigned int j=0; j<nodes.extent(2); j++) {
        cnodes(0,k,j) = nodes(owners(i,1),k,j);
      }
    }
    DRV refpt_buffer = sub_disc->mapPointsToReference(cpt,cnodes,sub_mesh->cellTopo[0]);
    //CellTools::mapToReferenceFrame(refpt_buffer, cpt, cnodes, *(sub_mesh->cellTopo[0]));
    DRV refpt("refpt",1,dimpts);
    Kokkos::deep_copy(refpt,Kokkos::subdynrankview(refpt_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
    Kokkos::View<int**,AssemblyDevice> offsets = wkset[0]->offsets;
    vector<int> usebasis = wkset[0]->usebasis;
    DRV basisvals("basisvals",offsets.extent(0),numLIDs);
    for (size_t n=0; n<offsets.extent(0); n++) {
      DRV bvals = sub_disc->evaluateBasis(sub_disc->basis_pointers[0][usebasis[n]], refpt);
      for (size_t m=0; m<offsets.extent(1); m++) {
        basisvals(n,offsets(n,m)) = bvals(0,m,0);
      }
    }
    ptsBasis.push_back(basisvals);
    
  }
  std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo(owners, ptsBasis);
  return basisinfo;
  
}

////////////////////////////////////////////////////////////////////////////////
// Evaluate the basis functions at a set of points
// TMW: what is this function for???
////////////////////////////////////////////////////////////////////////////////

//std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > SubGridDtN::evaluateBasis(const DRV & pts) {
  // this function is deprecated
//}


////////////////////////////////////////////////////////////////////////////////
// Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode> >  SubGridDtN::getEvaluationMatrix(const DRV & newip, Teuchos::RCP<SG_Map> & ip_map) {
  return sub_solver->getEvaluationMatrix(newip, ip_map);
  /*
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
   */
}

////////////////////////////////////////////////////////////////////////////////
// Get the subgrid cell GIDs
////////////////////////////////////////////////////////////////////////////////

LIDView SubGridDtN::getCellLIDs(const int & cellnum) {
  return groups[0][cellnum]->LIDs[0];
}

////////////////////////////////////////////////////////////////////////////////
// Update the subgrid parameters (will be depracated)
////////////////////////////////////////////////////////////////////////////////

void SubGridDtN::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
  for (size_t block=0; block<wkset.size(); ++block) {
    wkset[block]->params = params;
    wkset[block]->paramnames = paramnames;
  }
  sub_physics->updateParameters(params, paramnames);
  
}

// ========================================================================================
//
// ========================================================================================

void SubGridDtN::updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) {
  /*
  for (size_t block=0; block<groups.size(); ++block) {
    for (size_t grp=0; grp<groups[block].size(); ++grp) {
      int numElem = groups[block][grp]->numElem;
      for (int c=0; c<numElem; c++) {
        int cnode = loc[block]->cell_data_seed[c];
        for (int i=0; i<9; i++) {
          groups[0][grp]->cell_data(c,i) = rotation_data(cnode,i);
        }
      }
    }
  }*/
  
}

// ========================================================================================
//
// ========================================================================================

void SubGridDtN::updateLocalData(const int & macrogrp) {
  
  // Update the boundary condition definitions in the workset
  // to match the current (set of) macroelement(s)
  wkset[0]->var_bcs = macroData[macrogrp]->bcs;

}

// ========================================================================================
//
// ========================================================================================
    
void SubGridDtN::advance() {
  sub_solver->previous_time = sub_solver->current_time;
  for (size_t macrogrp=0; macrogrp<curr_soln.size(); ++macrogrp) {
    prev_soln[macrogrp]->assign(*(curr_soln[macrogrp]));
    sub_solver->performGather(macrogrp, curr_soln[macrogrp], 0, 0);
  }
}

// ========================================================================================
//
// ========================================================================================
    
void SubGridDtN::advanceStage() {
  if (isSynchronous) {
    for (size_t macrogrp=0; macrogrp<curr_soln.size(); ++macrogrp) {
      curr_soln[macrogrp]->update(1.0, *(stage_soln[macrogrp]), 1.0);
      curr_soln[macrogrp]->update(-1.0, *(prev_soln[macrogrp]), 1.0);
    }
  }
  else {
    sub_solver->previous_time = sub_solver->current_time;
    for (size_t macrogrp=0; macrogrp<curr_soln.size(); ++macrogrp) {
      prev_soln[macrogrp]->assign(*curr_soln[macrogrp]);
    }
  }
}
