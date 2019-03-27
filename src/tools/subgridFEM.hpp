/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SUBGRIDFEM_H
#define SUBGRIDFEM_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "cell.hpp"
#include "subgridMeshFactory.hpp"

#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "discretizationTools.hpp"
#include "assemblyManager.hpp"
#include "solverInterface.hpp"
#include "subgridTools.hpp"
#include "parameterManager.hpp"

class SubGridFEM : public SubGridModel {
public:
  
  SubGridFEM() {} ;
  
  ~SubGridFEM() {};
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  SubGridFEM(const Teuchos::RCP<LA_MpiComm> & LocalComm_,
             Teuchos::RCP<Teuchos::ParameterList> & settings_,
             topo_RCP & macro_cellTopo_, int & num_macro_time_steps_,
             ScalarT & macro_deltat_) :
  settings(settings_), macro_cellTopo(macro_cellTopo_),
  num_macro_time_steps(num_macro_time_steps_), macro_deltat(macro_deltat_) {
    
    LocalComm = LocalComm_;
    dimension = settings->sublist("Mesh").get<int>("dim",2);
    subgridverbose = settings->sublist("Solver").get<int>("Verbosity",10);
    multiscale_method = settings->get<string>("Multiscale Method","mortar");
    numrefine = settings->sublist("Mesh").get<int>("refinements",0);
    shape = settings->sublist("Mesh").get<string>("shape","quad");
    macroshape = settings->sublist("Mesh").get<string>("macro-shape","quad");
    time_steps = settings->sublist("Solver").get<int>("numSteps",1);
    initial_time = settings->sublist("Solver").get<ScalarT>("Initial time",0.0);
    final_time = settings->sublist("Solver").get<ScalarT>("finaltime",1.0);
    write_subgrid_state = settings->sublist("Solver").get<bool>("write subgrid state",true);
    error_type = settings->sublist("Postprocess").get<string>("Error type","L2"); // or "H1"
    
    string solver = settings->sublist("Solver").get<string>("solver","steady-state");
    if (solver == "steady-state") {
      final_time = 0.0;
    }
    
    // Solver settings
    useDirect = settings->sublist("Solver").get<bool>("use direct solver",true);
    use_amesos2 = settings->sublist("Solver").get<bool>("use amesos2",false);
    
    lintol = settings->sublist("Solver").get<ScalarT>("lintol",1.0E-7);
    liniter = settings->sublist("Solver").get<int>("liniter",100);
    
    Amesos AmFactory;
    char* SolverType = "Amesos_Klu";
    AmSolver = AmFactory.Create(SolverType, LinSys);
    have_sym_factor = false;
    have_preconditioner = false;
    sub_NLtol = settings->sublist("Solver").get<ScalarT>("NLtol",1.0E-12);
    sub_maxNLiter = settings->sublist("Solver").get<int>("MaxNLiter",10);
    
    /////////////////////////////////////////////////////////////////////////////////////
    // Define the sub-grid physics
    /////////////////////////////////////////////////////////////////////////////////////
    
    if (settings->isParameter("Functions Settings File")) {
      std::string filename = settings->get<std::string>("Functions Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> functions_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*functions_parlist) );
        settings->setParameters( *functions_parlist );
      }
      else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the functions settings file: " + filename);
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
    
    mesh_data_tag = settings->sublist("Mesh").get<string>("Data file","none");
    if (mesh_data_tag != "none") {
      mesh_data_pts_tag = settings->sublist("Mesh").get<string>("Data points file","mesh_data_pts");
      have_mesh_data = true;
      have_rotation_phi = settings->sublist("Mesh").get<bool>("Have mesh data phi",false);
      have_rotations = settings->sublist("Mesh").get<bool>("Have mesh data rotations",true);
      have_multiple_data_files = settings->sublist("Mesh").get<bool>("Have multiple mesh data files",false);
      number_mesh_data_files = settings->sublist("Mesh").get<int>("Number mesh data files",1);
    }
    
    compute_mesh_data = settings->sublist("Mesh").get<bool>("Compute mesh data",false);
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  int addMacro(const DRV macronodes_, Kokkos::View<int****,HostDevice> macrosideinfo_,
               vector<string> & macrosidenames,
               Kokkos::View<GO**,HostDevice> & macroGIDs, Kokkos::View<LO***,HostDevice> & macroindex) {
    
    bool first_time = false;
    if (cells.size() == 0) {
      first_time = true;
    }
    
    /////////////////////////////////////////////////////////////////////////////////////
    // Define the sub-grid mesh
    /////////////////////////////////////////////////////////////////////////////////////
    
    string blockID = "eblock";
    Teuchos::TimeMonitor localmeshtimer(*sgfemTotalAddMacroTimer);
    
    // Use the macro-element nodes to create the initial sub-grid element
    macronodes.push_back(macronodes_);
    macrosideinfo.push_back(macrosideinfo_);
    
    vector<vector<ScalarT> > nodes;
    vector<vector<int> > connectivity;
    Kokkos::View<int****,HostDevice> sideinfo;
    
    vector<string> eBlocks;
    
    {
      Teuchos::TimeMonitor localmeshtimer(*sgfemSubMeshTimer);
      
      SubGridTools sgt(LocalComm, macroshape, shape, macronodes_, macrosideinfo_);
      
      sgt.createSubMesh(numrefine);
    
      nodes = sgt.getSubNodes();
      connectivity = sgt.getSubConnectivity();
      sideinfo = sgt.getSubSideinfo();
    
      panzer_stk::SubGridMeshFactory meshFactory(shape, nodes, connectivity, blockID);
    
      mesh = meshFactory.buildMesh(*(LocalComm->getRawMpiComm()));
      //mesh = meshFactory.buildMesh(LocalComm->Comm());
      
      mesh->getElementBlockNames(eBlocks);
    
      //meshFactory.completeMeshConstruction(*mesh,LocalComm->Comm());
      meshFactory.completeMeshConstruction(*mesh,*(LocalComm->getRawMpiComm()));
    }
    
    /////////////////////////////////////////////////////////////////////////////////////
    // Set up the sub-cells
    /////////////////////////////////////////////////////////////////////////////////////
    
    cellTopo = mesh->getCellTopology(eBlocks[0]);
    vector<vector<Teuchos::RCP<cell> > > currcells;
    vector<vector<int> > orders;
    vector<vector<string> > types;
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubCellTimer);
      
      int numNodesPerElem = cellTopo->getNodeCount();
      int numSubElem = 1;
      
      if (first_time) { // first time through
        functionManager = Teuchos::rcp(new FunctionInterface(settings));
        
        vector<topo_RCP> cellTopo;
        cellTopo.push_back(DiscTools::getCellTopology(dimension, shape));
        
        vector<topo_RCP> sideTopo;
        sideTopo.push_back(DiscTools::getCellSideTopology(dimension, shape));
        physics_RCP = Teuchos::rcp( new physics(settings, LocalComm, cellTopo, sideTopo,
                                                functionManager, mesh) );
      }
      
      orders = physics_RCP->unique_orders;
      types = physics_RCP->unique_types;
      varlist = physics_RCP->varlist[0];
      
      // The convention will be that each subgrid model uses only 1 cell
      // with multiple elements - this will help expose subgrid/local parallelism
      
      vector<Teuchos::RCP<cell> > newcells;
      
      for (size_t e=0; e<connectivity.size(); e++) {
        DRV currnodes("currnodes",1,numNodesPerElem,dimension);
        Kokkos::View<int*> eIndex("element indices",1);
        for (int n=0; n<numNodesPerElem; n++) {
          for (int m=0; m<dimension; m++) {
            currnodes(0,n,m) = nodes[connectivity[e][n]][m];
          }
        }
        eIndex(0) = e;
        newcells.push_back(Teuchos::rcp(new cell(settings, LocalComm, cellTopo, physics_RCP,
                                                 currnodes, 0, eIndex, 0, false)));
      }
      currcells.push_back(newcells);
    }
    
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubSideinfoTimer);
      
      
      for (size_t e=0; e<currcells[0].size(); e++) {
        // Redefine the sideinfo for the subcells
        Kokkos::View<int****,HostDevice> subsideinfo("subcell side info", 1, sideinfo.dimension(1),
                                                     sideinfo.dimension(2), sideinfo.dimension(3));
        
        for (size_t i=0; i<sideinfo.dimension(1); i++) {
          for (size_t j=0; j<sideinfo.dimension(2); j++) {
            for (size_t k=0; k<sideinfo.dimension(3); k++) {
              subsideinfo(0,i,j,k) = sideinfo(e,i,j,k);
            }
            if (subsideinfo(0,i,j,0) == 1) { // redefine for weak Dirichlet conditions
              subsideinfo(0,i,j,0) = 4;
              subsideinfo(0,i,j,1) = -1;
            }
          }
        }
        currcells[0][e]->sideinfo = subsideinfo;
        currcells[0][e]->sidenames = macrosidenames;
      }
    }
    
    
    /////////////////////////////////////////////////////////////////////////////////////
    // Add sub-grid discretizations
    /////////////////////////////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubDiscTimer);
      
      if (first_time) {
        
        disc = Teuchos::rcp( new discretization(settings, mesh, orders, types, currcells) );
        
        vector<vector<int> > cards = disc->cards;
        vector<vector<int> > varowned = physics_RCP->varowned;
        
        ////////////////////////////////////////////////////////////////////////////////
        // The DOF-manager needs to be aware of the physics and the discretization(s)
        ////////////////////////////////////////////////////////////////////////////////
        
        DOF = physics_RCP->buildDOF(mesh);
        
        physics_RCP->setBCData(settings, mesh, DOF, cards);
        
        vector<string> blocknames;
        mesh->getElementBlockNames(blocknames);
        for (size_t b=0; b<currcells.size(); b++) {
          int eprog = 0;
          for (size_t e=0; e<currcells[b].size(); e++) {
            int numElem = currcells[b][e]->numElem;
            int nDOF = 0;
            vector<vector<int> > currGids;
            for (int p=0; p<numElem; p++) {
              vector<int> GIDs;
              size_t elemID = e;
              DOF->getElementGIDs(elemID, GIDs, blocknames[b]);
              currGids.push_back(GIDs);
              nDOF = GIDs.size();
            }
            Kokkos::View<GO**,HostDevice> currGids_KV("GIDs",numElem,nDOF);
            for (int p=0; p<numElem; p++) {
              for (int n=0; n<nDOF; n++) {
                currGids_KV(p,n) = currGids[p][n];
              }
            }
            currcells[b][e]->GIDs = currGids_KV;
            vector<vector<int> > offsets = physics_RCP->getOffsets(b, DOF);
            currcells[b][e]->sidenames = physics_RCP->sideSets;
            eprog += numElem;
          }
          
        }
        
        for (size_t e=0; e<currcells[0].size(); e++) {
          currcells[0][e]->setIP(disc->ref_ip[0]);
          currcells[0][e]->setSideIP(disc->ref_side_ip[0], disc->ref_side_wts[0]);
        }
      }
      else {
        
        for (size_t e=0; e<currcells[0].size(); e++) {
          currcells[0][e]->setIP(disc->ref_ip[0]);
          currcells[0][e]->setSideIP(disc->ref_side_ip[0], disc->ref_side_wts[0]);
          currcells[0][e]->GIDs = cells[0][e]->GIDs;
        }
      }
    }
    
    // Set up the linear algebra objects
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubSolverTimer);
      if (first_time) {
        
        sub_params = Teuchos::rcp( new ParameterManager(LocalComm, settings, mesh,
                                                        physics_RCP, currcells));
        
        sub_assembler = Teuchos::rcp( new AssemblyManager(LocalComm, settings, mesh,
                                                          disc, physics_RCP, DOF,
                                                          currcells, sub_params));
        
        subsolver = Teuchos::rcp( new solver(LocalComm, settings, mesh, disc, physics_RCP,
                                             DOF, sub_assembler, sub_params) );
        
      }
      else { // perform updates to currcells from solver interface
        for (size_t e=0; e<cells[0].size(); e++) {
          currcells[0][e]->setIndex(cells[0][e]->index, cells[0][e]->numDOF);
          currcells[0][e]->setParamIndex(cells[0][e]->paramindex, cells[0][e]->numParamDOF);
          currcells[0][e]->paramGIDs = cells[0][e]->paramGIDs;
          currcells[0][e]->setParamUseBasis(wkset[0]->paramusebasis, sub_params->paramNumBasis);
          int numDOF = currcells[0][0]->GIDs.dimension(1);
          currcells[0][e]->wkset = wkset[0];
          currcells[0][e]->setUseBasis(subsolver->useBasis[0],1);
          currcells[0][e]->setUpAdjointPrev(numDOF);
          currcells[0][e]->setUpSubGradient(sub_params->num_active_params);
        }
      }
    }
    
    if (first_time) { // first time through
      
      Teuchos::TimeMonitor localtimer(*sgfemLinearAlgebraSetupTimer);
      
      functionManager->setupLists(physics_RCP->varlist[0], macro_paramnames,
                                  macro_disc_paramnames);
      sub_assembler->wkset[0]->params_AD = paramvals_KVAD;
      
      functionManager->wkset = sub_assembler->wkset[0];
      
      functionManager->validateFunctions();
      functionManager->decomposeFunctions();
      
      cost_estimate = 1.0*currcells[0].size()*(currcells[0][0]->numElem)*time_steps;
      basis_pointers = disc->basis_pointers[0];
      useBasis = subsolver->useBasis;
      
      // Need to create Epetra versions of these maps (should be easy)
      
      Epetra_MpiComm EP_Comm(*(LocalComm->getRawMpiComm()));
      owned_map = Teuchos::rcp(new Epetra_Map(-1, subsolver->LA_owned.size(), &(subsolver->LA_owned[0]), 0, EP_Comm));
      if (LocalComm->getSize() > 1) {
        overlapped_map = Teuchos::rcp(new Epetra_Map(-1, subsolver->LA_ownedAndShared.size(), &(subsolver->LA_ownedAndShared[0]), 0, EP_Comm));
      }
      else {
        overlapped_map = owned_map;
      }
      
      owned_graph = subsolver->buildEpetraOwnedGraph(EP_Comm);
      if (LocalComm->getSize() > 1) {
        exporter = Teuchos::rcp(new Epetra_Export(*overlapped_map, *owned_map));
        importer = Teuchos::rcp(new Epetra_Import(*overlapped_map, *owned_map));
        overlapped_graph = subsolver->buildEpetraOverlappedGraph(EP_Comm);
      }
      else {
        overlapped_graph = owned_graph;
      }
      
      
      //owned_map = subsolver->LA_owned_map;
      //overlapped_map = subsolver->LA_overlapped_map;
      //exporter = subsolver->exporter;
      //importer = subsolver->importer;
      //overlapped_graph = subsolver->LA_overlapped_graph;
      
      res = Teuchos::rcp( new Epetra_MultiVector(*(owned_map),1)); // reset residual
      J = Teuchos::rcp( new Epetra_CrsMatrix(Copy, *owned_graph)); // reset Jacobian
      M = Teuchos::rcp( new Epetra_CrsMatrix(Copy, *owned_graph)); // reset Mass
      
      if (LocalComm->getSize() > 1) {
        res_over = Teuchos::rcp( new Epetra_MultiVector(*(overlapped_map),1)); // reset residual
        sub_J_over = Teuchos::rcp( new Epetra_CrsMatrix(Copy, *(overlapped_graph))); // reset Jacobian
        sub_M_over = Teuchos::rcp( new Epetra_CrsMatrix(Copy, *(overlapped_graph))); // reset Jacobian
      }
      else {
        res_over = res;
        sub_J_over = J;
        sub_M_over = M;
      }
      u = Teuchos::rcp( new Epetra_MultiVector(*(overlapped_map),1)); // reset residual
      u_dot = Teuchos::rcp( new Epetra_MultiVector(*(overlapped_map),1)); // reset residual
      phi = Teuchos::rcp( new Epetra_MultiVector(*(overlapped_map),1)); // reset residual
      phi_dot = Teuchos::rcp( new Epetra_MultiVector(*(overlapped_map),1)); // reset residual
      
      int nmacroDOF = macroGIDs.dimension(1);
      d_um = Teuchos::rcp( new Epetra_MultiVector(*(owned_map),nmacroDOF)); // reset residual
      d_sub_res_overm = Teuchos::rcp(new Epetra_MultiVector(*(overlapped_map),nmacroDOF));
      d_sub_resm = Teuchos::rcp(new Epetra_MultiVector(*(owned_map),nmacroDOF));
      d_sub_u_prevm = Teuchos::rcp(new Epetra_MultiVector(*(owned_map),nmacroDOF));
      d_sub_u_overm = Teuchos::rcp(new Epetra_MultiVector(*(overlapped_map),nmacroDOF));
      
      du_glob = Teuchos::rcp(new Epetra_MultiVector(*(owned_map),1));
      if (LocalComm->getSize() > 1) {
        du = Teuchos::rcp(new Epetra_MultiVector(*(overlapped_map),1));
      }
      else {
        du = du_glob;
      }
      
      filledJ = false;
      filledM = false;
      
      wkset = sub_assembler->wkset;
      //paramvals_AD = subsolver->paramvals_AD;
      
      // Need to create Epetra versions of these maps
      
      vector<int> params;
      if (sub_params->paramOwnedAndShared.size() == 0) {
        params.push_back(0);
      }
      else {
        params = sub_params->paramOwnedAndShared;
      }
      
      
      //param_owned_map = Teuchos::rcp(new Epetra_Map(-1, subsolver->numParamUnknowns, &(subsolver->paramOwned[0]), 0, EP_Comm));
      param_overlapped_map = Teuchos::rcp(new Epetra_Map(-1, params.size(), &params[0], 0, EP_Comm));
      //param_exporter = Teuchos::rcp(new Epetra_Export(*param_overlapped_map, *param_owned_map));
      //param_importer = Teuchos::rcp(new Epetra_Import(*param_overlapped_map, *param_owned_map));
    
      //param_owned_map = subsolver->param_owned_map;
      //param_overlapped_map = subsolver->param_overlapped_map;
      //param_exporter = subsolver->param_exporter;
      //param_importer = subsolver->param_importer;
      
      //TMW: this may not initialize properly
      //Psol.push_back(Teuchos::rcp(new Epetra_MultiVector(*param_overlapped_map,1)));
      
      num_active_params = sub_params->getNumParams(1);
      num_stochclassic_params = sub_params->getNumParams(2);
      stochclassic_param_names = sub_params->getParamsNames(2);
      
      stoch_param_types = sub_params->stochastic_distribution;
      stoch_param_means = sub_params->getStochasticParams("mean");
      stoch_param_vars = sub_params->getStochasticParams("variance");
      stoch_param_mins = sub_params->getStochasticParams("min");
      stoch_param_maxs = sub_params->getStochasticParams("max");
      discparamnames = sub_params->discretized_param_names;
      
    }
    
    cells.push_back(currcells[0]);
    
    int block = cells.size()-1;
    
    //////////////////////////////////////////////////////////////
    // Set the initial conditions
    //////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemSubICTimer);
      
      Teuchos::RCP<Epetra_MultiVector> init = Teuchos::rcp(new Epetra_MultiVector(*(overlapped_map),1));
      this->setInitial(init, block, false);
      vector<pair<ScalarT, Teuchos::RCP<Epetra_MultiVector> > > initvec;
      pair<ScalarT, Teuchos::RCP<Epetra_MultiVector> > initsol(initial_time, init);
      initvec.push_back(initsol);
      soln.push_back(initvec);
      
      Teuchos::RCP<Epetra_MultiVector> inita = Teuchos::rcp(new Epetra_MultiVector(*(overlapped_map),1));
      vector<pair<ScalarT, Teuchos::RCP<Epetra_MultiVector> > > initadjvec;
      pair<ScalarT, Teuchos::RCP<Epetra_MultiVector> > initadjsol(final_time, inita);
      initadjvec.push_back(initadjsol);
      adjsoln.push_back(initadjvec);
      
    }
    ////////////////////////////////////////////////////////////////////////////////
    // The current macro-element will store the values of its own basis functions
    // at the sub-grid integration points
    // Used to map the macro-scale solution to the sub-grid evaluation/integration pts
    ////////////////////////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor auxbasistimer(*sgfemComputeAuxBasisTimer);
      
      nummacroVars = macro_varlist.size();
      
      for(size_t e=0; e<cells[block].size(); e++) {
        
        
        int numElem = 1;
        
        // Volumetric ip (not used for mortar-ms)
        
        vector<DRV> currcell_basis, currcell_basisGrad;
        
        if (multiscale_method != "mortar" ) {
          
          DRV ip = cells[block][e]->ip;//wkset[b]->ip;
          DRV sref_ip_tmp("sref_ip_tmp",numElem, ip.dimension(1), ip.dimension(2));
          DRV sref_ip("sref_ip",ip.dimension(1), ip.dimension(2));
          CellTools<AssemblyDevice>::mapToReferenceFrame(sref_ip_tmp, ip, macronodes[block], *macro_cellTopo);
          for (size_t i=0; i<ip.dimension(1); i++) {
            for (size_t j=0; j<ip.dimension(2); j++) {
              sref_ip(i,j) = sref_ip_tmp(0,i,j);
            }
          }
          for (size_t i=0; i<macro_basis_pointers.size(); i++) {
            currcell_basis.push_back(DiscTools::evaluateBasis(macro_basis_pointers[i], sref_ip));
            currcell_basisGrad.push_back(DiscTools::evaluateBasisGrads(macro_basis_pointers[i], macronodes[block],
                                                                       sref_ip, macro_cellTopo));
          }
        }
        
        vector<vector<DRV> > currcell_side_basis, currcell_side_basisGrad;
        for (size_t s=0; s<sideinfo.dimension(2); s++) { // number of sides
          
          vector<DRV> currside_basis;
          bool compute = false;
          for (size_t n=0; n<sideinfo.dimension(1); n++) {
            if (cells[block][e]->sideinfo(0,n,s,0) > 0) {
              compute = true;
            }
          }
          
          if (compute) {
            DRV sside_ip = cells[block][e]->sideip[s];
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV tmp_basis = DRV("basis values",numElem,macro_basis_pointers[i]->getCardinality(),sside_ip.dimension(1));
              currside_basis.push_back(tmp_basis);
            }
            
            DRV side_ip_e("side_ip_e",1, sside_ip.dimension(1), sside_ip.dimension(2));
            for (int i=0; i<sside_ip.dimension(1); i++) {
              for (int j=0; j<sside_ip.dimension(2); j++) {
                side_ip_e(0,i,j) = sside_ip(0,i,j);
              }
            }
            DRV sref_side_ip_tmp("sref_side_ip_tmp",1, sside_ip.dimension(1), sside_ip.dimension(2));
            DRV sref_side_ip("sref_side_ip",sside_ip.dimension(1), sside_ip.dimension(2));
            
            CellTools<AssemblyDevice>::mapToReferenceFrame(sref_side_ip_tmp, side_ip_e, macronodes[block], *macro_cellTopo);
            for (size_t i=0; i<sside_ip.dimension(1); i++) {
              for (size_t j=0; j<sside_ip.dimension(2); j++) {
                sref_side_ip(i,j) = sref_side_ip_tmp(0,i,j);
              }
            }
            
            for (size_t i=0; i<macro_basis_pointers.size(); i++) {
              DRV tmp_basis = DiscTools::evaluateBasis(macro_basis_pointers[i], sref_side_ip);
              for (int k=0; k<tmp_basis.dimension(1); k++) {
                for (int j=0; j<tmp_basis.dimension(2); j++) {
                  currside_basis[i](0,k,j) = tmp_basis(0,k,j);
                }
              }
              
            }
          }
          currcell_side_basis.push_back(currside_basis);
        }
        
        cells[block][e]->addAuxDiscretization(macro_basis_pointers, currcell_basis, currcell_basisGrad,
                                              currcell_side_basis, currcell_side_basisGrad);
        
      }
      
      if (block == 0) { // first time through
        wkset[0]->addAux(macro_varlist.size());
      }
      for(size_t e=0; e<cells[block].size(); e++) {
        cells[block][e]->addAuxVars(macro_varlist);
        cells[block][e]->setAuxIndex(macroindex);
        cells[block][e]->setAuxUseBasis(macro_usebasis);
        cells[block][e]->auxGIDs = macroGIDs;
        cells[block][e]->auxoffsets = macro_offsets;
        cells[block][e]->wkset = wkset[0];
      }
    }
    physics_RCP->setWorkset(wkset);
    
    return block;
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void addMeshData() {
    
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
          Kokkos::View<ScalarT**,HostDevice> cell_data("cell_data",numElem,numdata);
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
        
        for (size_t b=0; b<cells.size(); b++) {
          for (size_t e=0; e<cells[b].size(); e++) {
            int numElem = cells[b][e]->numElem;
            DRV nodes = cells[b][e]->nodes;
            for (int c=0; c<numElem; c++) {
              Kokkos::View<ScalarT**,AssemblyDevice> center("center",1,3);
              int numnodes = nodes.dimension(1);
              for (size_t i=0; i<numnodes; i++) {
                for (size_t j=0; j<dimension; j++) {
                  center(0,j) += nodes(c,i,j)/(ScalarT)numnodes;
                }
              }
              ScalarT distance = 0.0;
              
              int cnode = mesh_data->findClosestNode(center(0,0), center(0,1), center(0,2), distance);
              
              bool iscloser = true;
              if (p>0){
                if (cells[b][e]->cell_data_distance[c] < distance) {
                  iscloser = false;
                }
              }
              if (iscloser) {
                Kokkos::View<ScalarT**,HostDevice> cdata = mesh_data->getdata(cnode);
                
                for (int i=0; i<cdata.dimension(1); i++) {
                  cells[b][e]->cell_data(c,i) = cdata(0,i);
                }
                
                if (have_rotations)
                  cells[b][e]->have_cell_rotation = true;
                if (have_rotation_phi)
                  cells[b][e]->have_cell_phi = true;
                
                cells[b][e]->cell_data_seed[c] = cnode % 50;
                cells[b][e]->cell_data_distance[c] = distance;
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
      int randSeed = settings->sublist("Mesh").get<int>("Random seed",1234);
      randomSeeds.push_back(randSeed);
      
      std::default_random_engine generator(randSeed);
      numSeeds = 0;
      
      ////////////////////////////////////////////////////////////////////////////////
      // Generate the micro-structure using seeds and nearest neighbors
      ////////////////////////////////////////////////////////////////////////////////
      
      bool fast_and_crude = settings->sublist("Mesh").get<bool>("Fast and crude microstructure",false);
      
      if (fast_and_crude) {
        int numxSeeds = settings->sublist("Mesh").get<int>("Number of xseeds",10);
        int numySeeds = settings->sublist("Mesh").get<int>("Number of yseeds",10);
        int numzSeeds = settings->sublist("Mesh").get<int>("Number of zseeds",10);
        
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
        numSeeds = settings->sublist("Mesh").get<int>("Number of seeds",1000);
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
          Kokkos::View<ScalarT**,HostDevice> cell_data("cell_data",numElem,numdata);
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
            for (size_t i=0; i<nodes.dimension(1); i++) {
              for (size_t j=0; j<nodes.dimension(2); j++) {
                center(0,j) += nodes(c,i,j)/(ScalarT)nodes.dimension(1);
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
            
            cells[b][e]->have_cell_rotation = true;
            cells[b][e]->have_cell_phi = false;
            
            cells[b][e]->cell_data_seed[c] = cnode;
            cells[b][e]->cell_data_seedindex[c] = seedIndex(cnode);
            cells[b][e]->cell_data_distance[c] = distance;
            
          }
        }
      }
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void subgridSolver(Kokkos::View<ScalarT***,AssemblyDevice> gl_u,
                     Kokkos::View<ScalarT***,AssemblyDevice> gl_phi,
                     const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                     const bool & compute_jacobian, const bool & compute_sens,
                     const int & num_active_params,
                     const bool & compute_disc_sens, const bool & compute_aux_sens,
                     workset & macrowkset,
                     const int & usernum, const int & macroelemindex,
                     Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) {
    
    Teuchos::TimeMonitor totalsolvertimer(*sgfemSolverTimer);
    
    ScalarT current_time = time;
    int macroDOF = macrowkset.numDOF;
    bool usesubadjoint = false;
    for (int i=0; i<subgradient.dimension(0); i++) {
      for (int j=0; j<subgradient.dimension(1); j++) {
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
    int cnumElem = 1;//cells[usernum].size();//[0]->numElem;
    Kokkos::View<ScalarT***,AssemblyDevice> cg_u("local u",cnumElem,
                                                gl_u.dimension(1),gl_u.dimension(2));
    Kokkos::View<ScalarT***,AssemblyDevice> cg_phi("local phi",cnumElem,
                                                  gl_phi.dimension(1),gl_phi.dimension(2));
    
    for (int e=0; e<cnumElem; e++) {
      for (int i=0; i<gl_u.dimension(1); i++) {
        for (int j=0; j<gl_u.dimension(2); j++) {
          cg_u(e,i,j) = gl_u(macroelemindex,i,j);
        }
      }
    }
    for (int e=0; e<cnumElem; e++) {
      for (int i=0; i<gl_phi.dimension(1); i++) {
        for (int j=0; j<gl_phi.dimension(2); j++) {
          cg_phi(e,i,j) = gl_phi(macroelemindex,i,j);
        }
      }
    }
    
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
    
    ScalarT prev_time = 0.0;
    
    {
      Teuchos::TimeMonitor localtimer(*sgfemInitialTimer);
      
      size_t numtimes = soln[usernum].size();
      if (isAdjoint) {
        if (isTransient) {
          for (size_t i=0; i<soln[usernum].size(); i++) {
            if (abs(current_time - soln[usernum][i].first) < 1.0e-12) {
              *u = *(soln[usernum][i-1].second);
              prev_time = soln[usernum][i-1].first;
            }
          }
          for (size_t i=0; i<adjsoln[usernum].size(); i++) {
            if (abs(current_time - adjsoln[usernum][i].first) < 1.0e-12) {
              *phi = *(adjsoln[usernum][i].second);
            }
          }
        }
        else {
          numtimes = soln[usernum].size();
          *u = *(soln[usernum][numtimes-1].second);
          prev_time = soln[usernum][numtimes-1].first;
          numtimes = adjsoln[usernum].size();
          *phi = *(adjsoln[usernum][numtimes-1].second);
        }
      }
      else { // forward or compute sens
        if (isTransient) {
          bool found_time = false;
          for (size_t i=0; i<soln[usernum].size(); i++) {
            if (abs(current_time - soln[usernum][i].first) < 1.0e-12) {
              *u = *(soln[usernum][i-1].second);
              prev_time = soln[usernum][i-1].first;
              found_time = true;
            }
          }
          if (!found_time) {
            *u = *(soln[usernum][numtimes-1].second);
            prev_time = soln[usernum][numtimes-1].first;
          }
        }
        else {
          *u = *(soln[usernum][numtimes-1].second);
          prev_time = soln[usernum][numtimes-1].first;
        }
        if (compute_sens) {
          for (size_t i=0; i<adjsoln[usernum].size(); i++) {
            if (abs(current_time - adjsoln[usernum][i].first) < 1.0e-12) {
              *phi = *(adjsoln[usernum][i+1].second);
            }
          }
        }
      }
    }
    
    
    //////////////////////////////////////////////////////////////
    // Use the coarse scale solution to solve local transient/nonlinear problem
    //////////////////////////////////////////////////////////////
    
    Teuchos::RCP<Epetra_MultiVector> d_u = d_um;
    if (compute_sens) {
      d_u = Teuchos::rcp( new Epetra_MultiVector(*(owned_map), num_active_params)); // reset residual
    }
    d_u->PutScalar(0.0);
    
    res->PutScalar(0.0);
    J->PutScalar(0.0);
    
    ScalarT h = 0.0;
    wkset[0]->resetFlux();
    
    if (isTransient) {
      ScalarT sgtime = prev_time;
      Teuchos::RCP<Epetra_MultiVector> prev_u = u;
      vector<Teuchos::RCP<Epetra_MultiVector> > curr_fsol;
      vector<Teuchos::RCP<Epetra_MultiVector> > curr_fsol_dot;
      vector<ScalarT> subsolvetimes;
      subsolvetimes.push_back(sgtime);
      if (isAdjoint) {
        // First, we need to resolve the forward problem
        
        for (int tstep=0; tstep<time_steps; tstep++) {
          Teuchos::RCP<Epetra_MultiVector> recu = Teuchos::rcp( new Epetra_MultiVector(*(overlapped_map),1)); // reset residual
          Teuchos::RCP<Epetra_MultiVector> recu_dot = Teuchos::rcp( new Epetra_MultiVector(*(overlapped_map),1)); // reset residual
          
          *recu = *u;
          sgtime += macro_deltat/(ScalarT)time_steps;
          subsolvetimes.push_back(sgtime);
          
          // set du/dt and \lambda
          alpha = (ScalarT)time_steps/macro_deltat;
          wkset[0]->alpha = alpha;
          wkset[0]->deltat= 1.0/alpha;
          
          Kokkos::View<ScalarT***,AssemblyDevice> currlambda = cg_u;
          
          ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
          
          recu_dot->PutScalar(0.0);
          
          this->subGridNonlinearSolver(recu, recu_dot, phi, phi_dot, Psol[0], currlambda,
                                       sgtime, isTransient, false, num_active_params, alpha, usernum, false);
          
          curr_fsol.push_back(recu);
          curr_fsol_dot.push_back(recu_dot);
          
        }
        
        for (int tstep=0; tstep<time_steps; tstep++) {
          
          size_t numsubtimes = subsolvetimes.size();
          size_t tindex = numsubtimes-1-tstep;
          sgtime = subsolvetimes[tindex];
          // set du/dt and \lambda
          alpha = (ScalarT)time_steps/macro_deltat;
          wkset[0]->alpha = alpha;
          wkset[0]->deltat= 1.0/alpha;
          
          Kokkos::View<ScalarT***,AssemblyDevice> currlambda = lambda;
          
          ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
          
          if (isAdjoint) {
            phi_dot->PutScalar(0.0);
          }
          
          this->subGridNonlinearSolver(curr_fsol[tindex-1], curr_fsol_dot[tindex-1], phi, phi_dot, Psol[0], currlambda,
                                       sgtime, isTransient, isAdjoint, num_active_params, alpha, usernum, store_adjPrev);
          
          this->computeSubGridSolnSens(d_u, compute_sens, curr_fsol[tindex-1],
                                       curr_fsol_dot[tindex-1], phi, phi_dot, Psol[0], currlambda,
                                       sgtime, isTransient, isAdjoint, num_active_params, alpha, lambda_scale, usernum, subgradient);
          
          this->updateFlux(phi, d_u, lambda, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0/(ScalarT)time_steps);
          
        }
      }
      else {
        for (int tstep=0; tstep<time_steps; tstep++) {
          sgtime += macro_deltat/(ScalarT)time_steps;
          // set du/dt and \lambda
          alpha = (ScalarT)time_steps/macro_deltat;
          wkset[0]->alpha = alpha;
          wkset[0]->deltat= 1.0/alpha;
          
          Kokkos::View<ScalarT***,AssemblyDevice> currlambda = lambda;
          
          ScalarT lambda_scale = 1.0;//-(current_time-sgtime)/deltat;
          
          u_dot->PutScalar(0.0);
          if (isAdjoint) {
            phi_dot->PutScalar(0.0);
          }
          
          this->subGridNonlinearSolver(u, u_dot, phi, phi_dot, Psol[0], currlambda,
                                       sgtime, isTransient, isAdjoint, num_active_params, alpha, usernum, false);
          
          this->computeSubGridSolnSens(d_u, compute_sens, u,
                                       u_dot, phi, phi_dot, Psol[0], currlambda,
                                       sgtime, isTransient, isAdjoint, num_active_params, alpha, lambda_scale, usernum, subgradient);
          
          this->updateFlux(u, d_u, lambda, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0/(ScalarT)time_steps);
        }
      }
      
    }
    else {
      
      wkset[0]->deltat = 1.0;
      this->subGridNonlinearSolver(u, u_dot, phi, phi_dot, Psol[0], lambda,
                                   current_time, isTransient, isAdjoint, num_active_params, alpha, usernum, false);
      
      this->computeSubGridSolnSens(d_u, compute_sens, u,
                                   u_dot, phi, phi_dot, Psol[0], lambda,
                                   current_time, isTransient, isAdjoint, num_active_params, alpha, 1.0, usernum, subgradient);
      
      if (isAdjoint) {
        this->updateFlux(phi, d_u, lambda, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0);
      }
      else {
        this->updateFlux(u, d_u, lambda, compute_sens, macroelemindex, time, macrowkset, usernum, 1.0);
      }
      
    }
    
    if (isAdjoint) {
      this->solutionStorage(phi, current_time, isAdjoint, usernum);
    }
    else if (!compute_sens) {
      this->solutionStorage(u, current_time, isAdjoint, usernum);
    }
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Re-seed the global parameters
  ///////////////////////////////////////////////////////////////////////////////////////
  
  
  void sacadoizeParams(const bool & seed_active, const int & num_active_params) {
    
    /*
    if (seed_active) {
      size_t pprog = 0;
      for (size_t i=0; i<paramvals_KVAD.dimension(0); i++) {
        if (paramtypes[i] == 1) { // active parameters
          for (size_t j=0; j<paramvals_KVAD.dimension(1); j++) {
            paramvals_KVAD(i,j) = AD(maxDerivs,pprog,paramvals_KVAD(i,j).val());
            pprog++;
          }
        }
        else { // inactive, stochastic, or discrete parameters
          for (size_t j=0; j<paramvals_KVAD.dimension(1); j++) {
            paramvals_KVAD(i,j) = AD(paramvals_KVAD(i,j).val());
          }
        }
      }
    }
    else {
      size_t pprog = 0;
      for (size_t i=0; i<paramvals_KVAD.dimension(0); i++) {
        for (size_t j=0; j<paramvals_KVAD.dimension(1); j++) {
          paramvals_KVAD(i,j) = AD(paramvals_KVAD(i,j).val());
        }
      }
    }
     */
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Subgrid Nonlinear Solver
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void subGridNonlinearSolver(Teuchos::RCP<Epetra_MultiVector> & sub_u, Teuchos::RCP<Epetra_MultiVector> & sub_u_dot,
                              Teuchos::RCP<Epetra_MultiVector> & sub_phi, Teuchos::RCP<Epetra_MultiVector> & sub_phi_dot,
                              Teuchos::RCP<Epetra_MultiVector> & sub_params, Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                              const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                              const int & num_active_params, const ScalarT & alpha, const int & usernum,
                              const bool & store_adjPrev) {
    
    
    Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverTimer);
    
    ScalarT resnorm = 10.0*sub_NLtol;
    ScalarT resnorm_initial = resnorm;
    ScalarT resnorm_scaled = resnorm;
    int iter = 0;
    Kokkos::View<ScalarT**,AssemblyDevice> aPrev;
    
    while (iter < sub_maxNLiter && resnorm_scaled > sub_NLtol) {
      
      sub_J_over->PutScalar(0.0);
      sub_M_over->PutScalar(0.0);
      res_over->PutScalar(0.0);
      
      wkset[0]->time = time;
      wkset[0]->isTransient = isTransient;
      wkset[0]->isAdjoint = isAdjoint;
      
      int numElem = 1;
      int numDOF = cells[usernum][0]->GIDs.dimension(1);
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_res, local_J, local_Jdot;
      
      {
        Teuchos::TimeMonitor localtimer2(*sgfemNonlinearSolverAllocateTimer);
        local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,numDOF,1);
        local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,numDOF,numDOF);
        local_Jdot = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian dot",numElem,numDOF,numDOF);
      }
      {
        Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSetSolnTimer);
        for (size_t e=0; e < cells[usernum].size(); e++) {
          cells[usernum][e]->setLocalSoln(sub_u,0,0);
          cells[usernum][e]->setLocalSoln(sub_u_dot,1,0);
          if (isAdjoint) {
            cells[usernum][e]->setLocalSoln(sub_phi,2,0);
            cells[usernum][e]->setLocalSoln(sub_phi_dot,3,0);
          }
          cells[usernum][e]->setLocalSoln(sub_params,4,0);
          cells[usernum][e]->aux = lambda;
        }
      }
      // volume assembly
      
      for (size_t e=0; e<cells[usernum].size(); e++) {
        if (isAdjoint) {
          aPrev = cells[usernum][e]->adjPrev;
          if (is_final_time) {
            for (int i=0; i<aPrev.dimension(0); i++) {
              for (int j=0; j<aPrev.dimension(1); j++) {
                cells[usernum][e]->adjPrev(i,j) = 0.0;
              }
            }
          }
        }
        
        wkset[0]->localEID = e;
        cells[usernum][e]->updateData();
        
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<numDOF; n++) {
            for (int s=0; s<local_res.dimension(2); s++) {
              local_res(p,n,s) = 0.0;
            }
            for (int s=0; s<local_J.dimension(2); s++) {
              local_J(p,n,s) = 0.0;
              local_Jdot(p,n,s) = 0.0;
            }
          }
        }
        
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverJacResTimer);
        
          cells[usernum][e]->computeJacRes(time, isTransient, isAdjoint,
                                           true, false, num_active_params, false, false, false,
                                           local_res, local_J, local_Jdot);
        
        }
        {
          Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverInsertTimer);
          Kokkos::View<GO**,HostDevice> GIDs = cells[usernum][e]->GIDs;
          for (int i=0; i<GIDs.dimension(0); i++) {
            vector<ScalarT> vals(GIDs.dimension(1));
            vector<int> cols(GIDs.dimension(1));
            for( size_t row=0; row<GIDs.dimension(1); row++ ) {
              int rowIndex = GIDs(i,row);
              ScalarT val = local_res(i,row,0);
              res_over->SumIntoGlobalValue(rowIndex,0, val);
              for( size_t col=0; col<GIDs.dimension(1); col++ ) {
                vals[col] = local_J(i,row,col) + alpha*local_Jdot(i,row,col);
                cols[col] = GIDs(i,col);
              }
              sub_J_over->SumIntoGlobalValues(rowIndex, GIDs.dimension(1), &vals[0], &cols[0]);
              for( size_t col=0; col<GIDs.dimension(1); col++ ) {
                vals[col] = local_Jdot(i,row,col);
              }
              sub_M_over->SumIntoGlobalValues(rowIndex, GIDs.dimension(1), &vals[0], &cols[0]);
              
            }
          }
        }
      }
      
      
      
      sub_J_over->FillComplete();
      sub_M_over->FillComplete();
      
      if (LocalComm->getSize() > 1) {
        J->PutScalar(0.0);
        J->Export(*sub_J_over, *(exporter), Add);
        M->PutScalar(0.0);
        M->Export(*sub_M_over, *(exporter), Add);
        if (!filledJ) {
          J->FillComplete();
          filledJ = true;
        }
        if (!filledM) {
          M->FillComplete();
          filledM = true;
        }
      }
      else {
        J = sub_J_over;
        M = sub_M_over;
      }
      LinSys.SetOperator(J.get());
      
      if (useDirect && !have_sym_factor) {
        if (use_amesos2) {
          Am2Solver = Amesos2::create<Epetra_CrsMatrix,Epetra_MultiVector>("KLU2", J, res, du_glob);
          Am2Solver->symbolicFactorization();
        }
        else {
          AmSolver->SymbolicFactorization();
        }
        have_sym_factor = true;
      }
      if (use_amesos2) {
        Am2Solver->setX(du_glob);
        Am2Solver->setB(res);
      }
      if (!useDirect && !have_preconditioner) {
        this->buildPreconditioner();
      }
      if (LocalComm->getSize() > 1) {
        res->PutScalar(0.0);
        res->Export(*res_over, *(exporter), Add);
      }
      else {
        res = res_over;
      }
      if (iter == 0) {
        res->NormInf(&resnorm_initial);
        if (resnorm_initial > 0.0)
          resnorm_scaled = 1.0;
        else
          resnorm_scaled = 0.0;
      }
      else {
        res->NormInf(&resnorm);
        resnorm_scaled = resnorm/resnorm_initial;
      }
      if(LocalComm->getRank() == 0 && subgridverbose>5) {
        cout << endl << "*********************************************************" << endl;
        cout << "***** Subgrid Nonlinear Iteration: " << iter << endl;
        cout << "***** Scaled Norm of nonlinear residual: " << resnorm_scaled << endl;
        cout << "*********************************************************" << endl;
      }
      
      if (resnorm_scaled > sub_NLtol) {
        
        Teuchos::TimeMonitor localtimer(*sgfemNonlinearSolverSolveTimer);
        
        du_glob->PutScalar(0.0);
        
        LinSys.SetRHS(res.get());
        LinSys.SetLHS(du_glob.get());
        if (useDirect) {
          if (use_amesos2) {
            Am2Solver->numericFactorization().solve();
          }
          else {
            AmSolver->NumericFactorization();
            AmSolver->Solve();
          }
        }
        else {
          AztecOO itersolver(LinSys);
          itersolver.SetAztecOption(AZ_output,0);
          itersolver.SetPrecOperator(MLPrec);
          itersolver.Iterate(liniter,lintol);
        }
        if (LocalComm->getSize() > 1) {
          du->PutScalar(0.0);
          du->Import(*du_glob, *(importer), Add);
        }
        else {
          du = du_glob;
        }
        
        if (isAdjoint) {
          
          sub_phi->Update(1.0, *du, 1.0);
          sub_phi_dot->Update(alpha, *du, 1.0);
        }
        else {
          sub_u->Update(1.0, *du, 1.0);
          sub_u_dot->Update(alpha, *du, 1.0);
        }
        
      }
      iter++;
      
    }
  }
  
  //////////////////////////////////////////////////////////////
  // Decide if we need to save the current solution
  //////////////////////////////////////////////////////////////
  
  void solutionStorage(Teuchos::RCP<Epetra_MultiVector> & newvec,
                       const ScalarT & time, const bool & isAdjoint,
                       const int & usernum) {
    
    Teuchos::TimeMonitor localtimer(*sgfemSolnStorageTimer);
    
    Teuchos::RCP<Epetra_MultiVector> vectostore = Teuchos::rcp( new Epetra_MultiVector(*(overlapped_map),1));
    *vectostore = *newvec;
    if (isAdjoint) {
      size_t findex;
      bool foundtime = false;
      ScalarT adjtime = time - macro_deltat;
      for (size_t j=0; j<adjsoln[usernum].size(); j++) {
        if (abs(adjsoln[usernum][j].first - adjtime) < 1.0e-13) {
          foundtime = true;
          findex = j;
        }
      }
      pair<ScalarT, Teuchos::RCP<Epetra_MultiVector> > time_u(adjtime,vectostore);
      if (foundtime) {
        adjsoln[usernum][findex] = time_u;
      }
      else {
        adjsoln[usernum].push_back(time_u);
      }
      
    }
    else {
      size_t findex;
      bool foundtime = false;
      for (size_t j=0; j<soln[usernum].size(); j++) {
        if (abs(soln[usernum][j].first - time) < 1.0e-13) {
          foundtime = true;
          findex = j;
        }
      }
      pair<ScalarT, Teuchos::RCP<Epetra_MultiVector> > time_u(time,vectostore);
      if (foundtime) {
        soln[usernum][findex] = time_u;
      }
      else {
        soln[usernum].push_back(time_u);
      }
    }
  }
  
  //////////////////////////////////////////////////////////////
  // Compute the derivative of the local solution w.r.t coarse
  // solution or w.r.t parameters
  //////////////////////////////////////////////////////////////
  
  void computeSubGridSolnSens(Teuchos::RCP<Epetra_MultiVector> & d_sub_u, const bool & compute_sens,
                              Teuchos::RCP<Epetra_MultiVector> & sub_u, Teuchos::RCP<Epetra_MultiVector> & sub_u_dot,
                              Teuchos::RCP<Epetra_MultiVector> & sub_phi, Teuchos::RCP<Epetra_MultiVector> & sub_phi_dot,
                              Teuchos::RCP<Epetra_MultiVector> & sub_param, Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                              const ScalarT & time,
                              const bool & isTransient, const bool & isAdjoint, const int & num_active_params, const ScalarT & alpha,
                              const ScalarT & lambda_scale, const int & usernum,
                              Kokkos::View<ScalarT**,AssemblyDevice> subgradient) {
    
    Teuchos::TimeMonitor localtimer(*sgfemSolnSensTimer);
    
    Teuchos::RCP<Epetra_MultiVector> d_sub_res_over = d_sub_res_overm;
    Teuchos::RCP<Epetra_MultiVector> d_sub_res = d_sub_resm;
    Teuchos::RCP<Epetra_MultiVector> d_sub_u_prev = d_sub_u_prevm;
    Teuchos::RCP<Epetra_MultiVector> d_sub_u_over = d_sub_u_overm;
    
    if (compute_sens) {
      int numsubDerivs = d_sub_u->NumVectors();
      d_sub_res_over = Teuchos::rcp(new Epetra_MultiVector(*(overlapped_map),numsubDerivs));
      d_sub_res = Teuchos::rcp(new Epetra_MultiVector(*(owned_map),numsubDerivs));
      d_sub_u_prev = Teuchos::rcp(new Epetra_MultiVector(*(owned_map),numsubDerivs));
      d_sub_u_over = Teuchos::rcp(new Epetra_MultiVector(*(overlapped_map),numsubDerivs));
    }
    
    d_sub_res_over->PutScalar(0.0);
    d_sub_res->PutScalar(0.0);
    d_sub_u_prev->PutScalar(0.0);
    d_sub_u_over->PutScalar(0.0);
    
    ScalarT scale = -1.0*lambda_scale;
    
    for (size_t e=0; e < cells[usernum].size(); e++) {
      cells[usernum][e]->setLocalSoln(sub_u,0,0);
      cells[usernum][e]->setLocalSoln(sub_u_dot,1,0);
      if (isAdjoint) {
        cells[usernum][e]->setLocalSoln(sub_phi,2,0);
        cells[usernum][e]->setLocalSoln(sub_phi_dot,3,0);
      }
      cells[usernum][e]->setLocalSoln(sub_param,4,0);
      cells[usernum][e]->aux = lambda;
    }
    
    int numElem = 1;
    
    if (compute_sens) {
      
      this->sacadoizeParams(true, num_active_params);
      wkset[0]->time = time;
      wkset[0]->isTransient = isTransient;
      wkset[0]->isAdjoint = isAdjoint;
      
      int snumDOF = cells[usernum][0]->GIDs.dimension(1);
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_res, local_J, local_Jdot;
      
      local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,snumDOF,num_active_params);
      
      local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,snumDOF,snumDOF);
      local_Jdot = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian dot",numElem,snumDOF,snumDOF);
      
      for (size_t e=0; e<cells[usernum].size(); e++) {
        
        wkset[0]->localEID = e;
        cells[usernum][e]->updateData();
        
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<snumDOF; n++) {
            for (int s=0; s<local_res.dimension(2); s++) {
              local_res(p,n,s) = 0.0;
            }
            for (int s=0; s<local_J.dimension(2); s++) {
              local_J(p,n,s) = 0.0;
              local_Jdot(p,n,s) = 0.0;
            }
          }
        }
        
        cells[usernum][e]->computeJacRes(time, isTransient, isAdjoint,
                                         false, true, num_active_params, false, false, false,
                                         local_res, local_J, local_Jdot);
        
        Kokkos::View<GO**,HostDevice>  GIDs = cells[usernum][e]->GIDs;
        for (int i=0; i<GIDs.dimension(0); i++) {
          for( size_t row=0; row<GIDs.dimension(1); row++ ) {
            int rowIndex = GIDs(i,row);
            for( size_t col=0; col<num_active_params; col++ ) {
              ScalarT val = local_res(i,row,col);
              d_sub_res_over->SumIntoGlobalValue(rowIndex,col, 1.0*val);
            }
          }
        }
      }
      for (int p=0; p<num_active_params; p++) {
        for (int i=0; i<sub_phi->GlobalLength(); i++) {
          subgradient(p,0) += (*sub_phi)[0][i] * (*d_sub_res_over)[p][i];
        }
      }
      
    }
    else {
      wkset[0]->time = time;
      wkset[0]->isTransient = isTransient;
      wkset[0]->isAdjoint = isAdjoint;
      
      Kokkos::View<ScalarT***,AssemblyDevice> local_res, local_J, local_Jdot;
      
      int snumDOF = cells[usernum][0]->GIDs.dimension(1);
      int anumDOF = cells[usernum][0]->auxGIDs.dimension(1);
      
      local_res = Kokkos::View<ScalarT***,AssemblyDevice>("local residual",numElem,snumDOF,1);
      local_J = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian",numElem,snumDOF,anumDOF);
      local_Jdot = Kokkos::View<ScalarT***,AssemblyDevice>("local Jacobian dot",numElem,snumDOF,anumDOF);
      
      //volume assembly
      
      for (size_t e=0; e<cells[usernum].size(); e++) {
        
        wkset[0]->localEID = e;
        cells[usernum][e]->updateData();
        
        for (int p=0; p<numElem; p++) {
          for (int n=0; n<snumDOF; n++) {
            for (int s=0; s<local_res.dimension(2); s++) {
              local_res(p,n,s) = 0.0;
            }
            for (int s=0; s<local_J.dimension(2); s++) {
              local_J(p,n,s) = 0.0;
              local_Jdot(p,n,s) = 0.0;
            }
          }
        }
        
        cells[usernum][e]->computeJacRes(time, isTransient, isAdjoint,
                                         true, false, num_active_params, false, true, false,
                                         local_res, local_J, local_Jdot);
        Kokkos::View<GO**,HostDevice> GIDs = cells[usernum][e]->GIDs;
        Kokkos::View<GO**,HostDevice> aGIDs = cells[usernum][e]->auxGIDs;
        vector<vector<int> > aoffsets = cells[usernum][e]->auxoffsets;
        
        for (int i=0; i<GIDs.dimension(0); i++) {
          for( size_t row=0; row<GIDs.dimension(1); row++ ) {
            int rowIndex = GIDs(i,row);
            for( size_t col=0; col<aGIDs.dimension(1); col++ ) {
              ScalarT val = local_J(i,row,col);
              d_sub_res_over->SumIntoGlobalValue(rowIndex,col, scale*val);
            }
          }
        }
      }
      
      M->Apply(*d_sub_u,*d_sub_u_prev);
      if (LocalComm->getSize() > 1) {
        d_sub_res->Export(*d_sub_res_over, *(exporter), Add);
      }
      else {
        d_sub_res = d_sub_res_over;
      }
      d_sub_res->Update(1.0*alpha, *d_sub_u_prev, 1.0);
      
      LinSys.SetOperator(J.get());
      if (useDirect) {
        LinSys.SetRHS(d_sub_res.get());
        LinSys.SetLHS(d_sub_u_over.get());
        
        if (use_amesos2) {
          Am2Solver->setX(d_sub_u_over);
          Am2Solver->setB(d_sub_res);
          Am2Solver->solve();
        }
        else {
          AmSolver->NumericFactorization();
          AmSolver->Solve();
        }
      }
      else {
        Teuchos::RCP<Epetra_MultiVector> cd_sub_res = Teuchos::rcp(new Epetra_MultiVector(*(owned_map),1));
        Teuchos::RCP<Epetra_MultiVector> cd_sub_u_over = Teuchos::rcp(new Epetra_MultiVector(*(overlapped_map),1));
        
        for (size_t i=0; i<d_sub_u->NumVectors(); i++) {
          for (int j=0; j<d_sub_u->MyLength(); j++) {
            (*cd_sub_res)[0][j] = (*d_sub_res)[i][j];
          }
          cd_sub_u_over->PutScalar(0.0);
          LinSys.SetRHS(cd_sub_res.get());
          LinSys.SetLHS(cd_sub_u_over.get());
          
          AztecOO itersolver(LinSys);
          itersolver.SetAztecOption(AZ_output,0);
          itersolver.SetPrecOperator(MLPrec);
          itersolver.Iterate(liniter,lintol);
          for (int j=0; j<d_sub_u_over->MyLength(); j++) {
            (*d_sub_u_over)[i][j] = (*cd_sub_u_over)[0][j];
          }
        }
      }
      if (LocalComm->getSize() > 1) {
        d_sub_u->PutScalar(0.0);
        d_sub_u->Import(*d_sub_u_over, *(importer), Add);
      }
      else {
        d_sub_u = d_sub_u_over;
      }
    }
  }
  
  //////////////////////////////////////////////////////////////
  // Update the flux
  //////////////////////////////////////////////////////////////
  
  void updateFlux(const Teuchos::RCP<Epetra_MultiVector> & u, const Teuchos::RCP<Epetra_MultiVector> & d_u,
                  Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                  const bool & compute_sens, const int macroelemindex,
                  const ScalarT & time, workset & macrowkset,
                  const int & usernum, const ScalarT & fwt) {
    
    Teuchos::TimeMonitor localtimer(*sgfemFluxTimer);
    
    for (size_t e=0; e<cells[usernum].size(); e++) {
      
      for (size_t s=0; s<cells[usernum][e]->sideip.size(); s++) {
        if (cells[usernum][e]->sideinfo(0,0,s,1) == -1) {
          {
            Teuchos::TimeMonitor localwktimer(*sgfemFluxWksetTimer);
            
            wkset[0]->updateSide(cells[usernum][e]->nodes, cells[usernum][e]->sideip[s],
                                 cells[usernum][e]->sidewts[s],
                                 cells[usernum][e]->normals[s],
                                 cells[usernum][e]->sideijac[s], s);
          }
          DRV cwts = wkset[0]->wts_side;
          ScalarT h = 0.0;
          wkset[0]->sidename = "interior";
          {
            Teuchos::TimeMonitor localcelltimer(*sgfemFluxCellTimer);
            
            cells[usernum][e]->updateData();
            cells[usernum][e]->computeFlux(u, d_u, Psol[0], lambda, time,
                                           s, h, compute_sens);
          }
          for (int c=0; c<cells[usernum][e]->numElem; c++) {
            for (int n=0; n<nummacroVars; n++) {
              if (cells[usernum][e]->sideinfo(c,n,s,1) == -1) {
                DRV mortarbasis_ip = cells[usernum][e]->auxside_basis[s][macrowkset.usebasis[n]];
                for (int j=0; j<mortarbasis_ip.dimension(1); j++) {
                  for (int i=0; i<mortarbasis_ip.dimension(2); i++) {
                    macrowkset.res(macroelemindex,macrowkset.offsets(n,j)) += mortarbasis_ip(c,j,i)*(wkset[0]->flux(c,n,i))*cwts(c,i)*fwt;
                  }
                }
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
  
  void setInitial(Teuchos::RCP<Epetra_MultiVector> & initial, const int & usernum, const bool & useadjoint) {
    
    initial->PutScalar(0.0);
     // TMW: uncomment if you need a nonzero initial condition
     //      right now, it slows everything down ... especially if using an L2-projection
    
    /*
    bool useL2proj = true;//settings->sublist("Solver").get<bool>("Project initial",true);
    
    if (useL2proj) {
      
      // Compute the L2 projection of the initial data into the discrete space
      Teuchos::RCP<Epetra_MultiVector> rhs = Teuchos::rcp(new Epetra_MultiVector(*overlapped_map,1)); // reset residual
      Teuchos::RCP<Epetra_CrsMatrix>  mass = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *overlapped_map, -1)); // reset Jacobian
      Teuchos::RCP<Epetra_MultiVector> glrhs = Teuchos::rcp(new Epetra_MultiVector(*owned_map,1)); // reset residual
      Teuchos::RCP<Epetra_CrsMatrix>  glmass = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *owned_map, -1)); // reset Jacobian
      
      
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
      
      Teuchos::RCP<Epetra_MultiVector> glinitial = Teuchos::rcp(new Epetra_MultiVector(*overlapped_map,1)); // reset residual
      
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
  // Subgrid Linear Solver
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void linearSolver(Teuchos::RCP<Epetra_CrsMatrix>  & M, Teuchos::RCP<Epetra_MultiVector> & r, Teuchos::RCP<Epetra_MultiVector> & sol) {
    Epetra_LinearProblem cLinSys(M.get(), sol.get(), r.get());
    Amesos_BaseSolver * AmSolverT;
    Amesos AmFactoryT;
    char* SolverType = "Amesos_Klu";
    AmSolverT = AmFactoryT.Create(SolverType, cLinSys);
    AmSolverT->SymbolicFactorization();
    AmSolverT->NumericFactorization();
    AmSolverT->Solve();
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the error for verification
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT**,AssemblyDevice> computeError(const ScalarT & time, const int & usernum) {
    
    size_t numVars = varlist.size();
    int tindex = -1;
    for (int tt=0; tt<soln[usernum].size(); tt++) {
      if (abs(soln[usernum][tt].first - time)<1.0e-10) {
        tindex = tt;
      }
    }
    
    Kokkos::View<ScalarT**,AssemblyDevice> errors("error",cells[usernum].size(), numVars);
    if (tindex != -1) {
      Kokkos::View<ScalarT**,AssemblyDevice> curr_errors;
      performGather(usernum, soln[usernum][tindex].second, 0, 0);
      for (size_t e=0; e<cells[usernum].size(); e++) {
        curr_errors = cells[usernum][e]->computeError(time,tindex,false,error_type);
        for (int c=0; c<curr_errors.dimension(0); c++) {
          for (size_t i=0; i < numVars; i++) {
            errors(e,i) += curr_errors(c,i);
          }
        }
      }
    }
    return errors;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the objective function
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD*,AssemblyDevice> computeObjective(const string & response_type, const int & seedwhat,
                                                    const ScalarT & time, const int & usernum) {
    
    int tindex = -1;
    for (int tt=0; tt<soln[usernum].size(); tt++) {
      if (abs(soln[usernum][tt].first - time)<1.0e-10) {
        tindex = tt;
      }
    }
    
    Kokkos::View<AD*,AssemblyDevice> objective;
    bool beensized = false;
    performGather(usernum, soln[usernum][tindex].second, 0,0);
    performGather(usernum, Psol[0], 4, 0);
    
    for (size_t e=0; e<cells[usernum].size(); e++) {
      Kokkos::View<AD**,AssemblyDevice> curr_obj = cells[usernum][e]->computeObjective(time, tindex, seedwhat);
      if (!beensized && curr_obj.dimension(1)>0) {
        objective = Kokkos::View<AD*,AssemblyDevice>("objective", curr_obj.dimension(1));
        beensized = true;
      }
      for (int c=0; c<cells[usernum][e]->numElem; c++) {
        for (size_t i=0; i<curr_obj.dimension(1); i++) {
          objective(i) += curr_obj(c,i);
        }
      }
    }
    
    return objective;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Write the solution to a file
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void writeSolution(const string & filename, const int & usernum) {
    
    
    bool isTD = false;
    if (soln[usernum].size() > 1) {
      isTD = true;
    }
    
    string blockID = "eblock";
    
    //////////////////////////////////////////////////////////////
    // Re-create the subgrid mesh
    //////////////////////////////////////////////////////////////
    
    SubGridTools sgt(LocalComm, macroshape, shape, macronodes[usernum], macrosideinfo[usernum]);
    sgt.createSubMesh(numrefine);
    vector<vector<ScalarT> > nodes = sgt.getSubNodes();
    vector<vector<int> > connectivity = sgt.getSubConnectivity();
    Kokkos::View<int****,HostDevice> sideinfo = sgt.getSubSideinfo();
    
    panzer_stk::SubGridMeshFactory submeshFactory(shape, nodes, connectivity, blockID);
    Teuchos::RCP<panzer_stk::STK_Interface> submesh = submeshFactory.buildMesh(*(LocalComm->getRawMpiComm()));
    
    
    //////////////////////////////////////////////////////////////
    // Add in the necessary fields for plotting
    //////////////////////////////////////////////////////////////
    
    vector<string> subeBlocks;
    submesh->getElementBlockNames(subeBlocks);
    for (size_t j=0; j<varlist.size(); j++) {
      submesh->addSolutionField(varlist[j], subeBlocks[0]);
    }
    vector<string> subextrafieldnames = physics_RCP->getExtraFieldNames(0);
    for (size_t j=0; j<subextrafieldnames.size(); j++) {
      submesh->addSolutionField(subextrafieldnames[j], subeBlocks[0]);
    }
    vector<string> subextracellfields = physics_RCP->getExtraCellFieldNames(0);
    for (size_t j=0; j<subextracellfields.size(); j++) {
      submesh->addCellField(subextracellfields[j], subeBlocks[0]);
    }
    submesh->addCellField("mesh_data_seed", subeBlocks[0]);
    
    if (discparamnames.size() > 0) {
      for (size_t n=0; n<discparamnames.size(); n++) {
        int paramnumbasis = cells[0][0]->paramindex.dimension(1);
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
    int numSteps = soln[usernum].size();
    
    for (int m=0; m<numSteps; m++) {
      
      vector<size_t> myElements;
      size_t eprog = 0;
      for (size_t e=0; e<cells[usernum].size(); e++) {
        for (size_t p=0; p<cells[usernum][e]->numElem; p++) {
          myElements.push_back(eprog);
          eprog++;
        }
      }
      
      vector<vector<int> > suboffsets = physics_RCP->offsets[0];
      // Collect the subgrid solution
      for (int n = 0; n<varlist.size(); n++) { // change to subgrid numVars
        size_t numsb = cells[usernum][0]->numDOF(n);//index[0][n].size();
        Kokkos::View<ScalarT**,HostDevice> soln_computed("soln",cells[usernum].size(), numsb); // TMW temp. fix
        string var = varlist[n];
        for( size_t e=0; e<cells[usernum].size(); e++ ) {
          int numElem = cells[usernum][e]->numElem;
          Kokkos::View<GO**,HostDevice> GIDs = cells[usernum][e]->GIDs;
          for (int p=0; p<numElem; p++) {
            
            for( int i=0; i<numsb; i++ ) {
              int pindex = overlapped_map->LID(GIDs(p,suboffsets[n][i]));
              if (write_subgrid_state)
                soln_computed(p,i) = (*(soln[usernum][m].second))[0][pindex];
              else
                soln_computed(p,i) = (*(adjsoln[usernum][m].second))[0][pindex];
            }
          }
        }
        submesh->setSolutionFieldData(var, blockID, myElements, soln_computed);
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
          int numSubNodes = subnodes.dimension(0);
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
      // FC efdata(cells[b].size(), rnodes.dimension(1));
      // subextrafields.push_back(efdata);
      // }
      
      // vector<FC> cfields;
      // for (size_t k=0; k<cells[b].size(); k++) {
      // DRV snodes = cells[b][k]->nodes;
      // cfields = physics_RCP->getExtraFields(b, snodes, solvetimes[m]);
      // for (size_t j=0; j<subextrafieldnames.size(); j++) {
      // for (size_t i=0; i<snodes.dimension(1); i++) {
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
      //	    cfields = sub_physics_RCP[level]->getExtraFields(b, snodes, 0.0);
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
      //          vector<FC> cfields = sub_physics_RCP[level]->getExtraFields(b, snodes, 0.0);//subgrid_solvetimes[m]);
      //          for (size_t j=0; j<subextrafieldnames.size(); j++) {
      //            for (size_t i=0; i<snodes.dimension(1); i++) {
      //              subextrafields[j](k,i) = vcfields[j](i);
      //            }
      //          }
      //        }
      
      /*
      vector<string> extracellfieldnames = physics_RCP->getExtraCellFieldNames(0);
      vector<FC> extracellfields;// = phys->getExtraFields(b);
      for (size_t j=0; j<extracellfieldnames.size(); j++) {
        FC efdata(cells[usernum].size(), 1);
        extracellfields.push_back(efdata);
      }
      for (size_t k=0; k<cells[usernum].size(); k++) {
        cells[usernum][k]->updateSolnWorkset(soln[usernum][m].second, 0); // also updates ip, ijac
        cells[usernum][k]->updateData();
        wkset[0]->time = soln[usernum][m].first;
        vector<FC> cfields = physics_RCP->getExtraCellFields(0);
        size_t j = 0;
        for (size_t g=0; g<cfields.size(); g++) {
          for (size_t h=0; h<cfields[g].dimension(0); h++) {
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
      Kokkos::View<ScalarT**,HostDevice> cdata("cell data",cells[usernum].size(), 1);
      if (cells[usernum][0]->have_cell_phi || cells[usernum][0]->have_cell_rotation) {
        int eprog = 0;
        for (size_t k=0; k<cells[usernum].size(); k++) {
          vector<size_t> cell_data_seed = cells[usernum][k]->cell_data_seed;
          vector<size_t> cell_data_seedindex = cells[usernum][k]->cell_data_seedindex;
          Kokkos::View<ScalarT**> cell_data = cells[usernum][k]->cell_data;
          for (int p=0; p<cells[usernum][k]->numElem; p++) {
            /*
            if (cell_data.dimension(1) == 3) {
              cdata(eprog,0) = cell_data(p,0);//cell_data_seed[p];
            }
            else if (cell_data.dimension(1) == 9) {
              cdata(eprog,0) = cell_data(p,0)*cell_data(p,4)*cell_data(p,8);//cell_data_seed[p];
            }
             */
            cdata(eprog,0) = cell_data_seedindex[p];
            eprog++;
          }
          //for (int p=0; p<cells[usernum][k]->numElem; p++) {
          //  cdata(eprog,0) = cell_data_seed[p];
          //  eprog++;
          //}
        }
      }
      submesh->setCellFieldData("mesh_data_seed", blockID, myElements, cdata);
      
      if(isTD) {
        submesh->writeToExodus(soln[usernum][m].first);
      }
      else {
        submesh->writeToExodus(filename);
      }
     
    }
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Write the solution to a file
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void writeSolution(const string & filename) {
    
    
    bool isTD = false;
    if (soln[0].size() > 1) {
      isTD = true;
    }
    
    string blockID = "eblock";
    
    panzer_stk::SubGridMeshFactory submeshFactory(shape);
    
    //////////////////////////////////////////////////////////////
    // Re-create the subgrid mesh
    //////////////////////////////////////////////////////////////
    
    for (size_t e=0; e<macronodes.size(); e++) {
      SubGridTools sgt(LocalComm, macroshape, shape, macronodes[e], macrosideinfo[e]);
      sgt.createSubMesh(numrefine);
      vector<vector<ScalarT> > nodes = sgt.getSubNodes();
      vector<vector<int> > connectivity = sgt.getSubConnectivity();
      Kokkos::View<int****,HostDevice> sideinfo = sgt.getSubSideinfo();
      
      submeshFactory.addElems(nodes, connectivity);
    }
    
    Teuchos::RCP<panzer_stk::STK_Interface> submesh = submeshFactory.buildMesh(*(LocalComm->getRawMpiComm()));
    
    //////////////////////////////////////////////////////////////
    // Add in the necessary fields for plotting
    //////////////////////////////////////////////////////////////
    
    vector<string> subeBlocks;
    submesh->getElementBlockNames(subeBlocks);
    
    for (size_t b=0; b<subeBlocks.size(); b++) {
      for (size_t j=0; j<varlist.size(); j++) {
        submesh->addSolutionField(varlist[j], subeBlocks[b]);
      }
      vector<string> subextrafieldnames = physics_RCP->getExtraFieldNames(0);
      for (size_t j=0; j<subextrafieldnames.size(); j++) {
        submesh->addSolutionField(subextrafieldnames[j], subeBlocks[b]);
      }
      vector<string> subextracellfields = physics_RCP->getExtraCellFieldNames(0);
      for (size_t j=0; j<subextracellfields.size(); j++) {
        submesh->addCellField(subextracellfields[j], subeBlocks[b]);
      }
      if (cells[b][0]->have_cell_phi || cells[b][0]->have_cell_rotation) {
        submesh->addCellField("mesh_data_seed", subeBlocks[b]);
      }
      
      if (discparamnames.size() > 0) {
        for (size_t n=0; n<discparamnames.size(); n++) {
          int paramnumbasis = cells[0][0]->numParamDOF(n);//paramindex[n].size();
          if (paramnumbasis==1) {
            submesh->addCellField(discparamnames[n], subeBlocks[b]);
          }
          else {
            submesh->addSolutionField(discparamnames[n], subeBlocks[b]);
          }
        }
      }
    }
    submeshFactory.completeMeshConstruction(*submesh,*(LocalComm->getRawMpiComm()));
    
    //////////////////////////////////////////////////////////////
    // Add fields to mesh
    //////////////////////////////////////////////////////////////
    
    
    if(isTD)
      submesh->setupExodusFile(filename);
    
    int numSteps = soln[0].size();
    
    vector<size_t> myElements;
    size_t eprog = 0;
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        for (size_t p=0; p<cells[b][e]->numElem; p++) {
          myElements.push_back(eprog);
          eprog++;
        }
      }
    }
    
    for (int m=0; m<numSteps; m++) {
      
      
      //for (size_t b=0; b<cells.size(); b++) {
      
      vector<vector<int> > suboffsets = physics_RCP->offsets[0];
      // Collect the subgrid solution
      for (int n = 0; n<varlist.size(); n++) { // change to subgrid numVars
        size_t numsb = cells[0][0]->numDOF(n);//index[0][n].size();
        Kokkos::View<ScalarT**,HostDevice> soln_computed("soln",myElements.size(), numsb); // TMW temp. fix
        string var = varlist[n];
        size_t eprog = 0;
        for (size_t b=0; b<cells.size(); b++) {
          
          for( size_t e=0; e<cells[b].size(); e++ ) {
            int numElem = cells[b][e]->numElem;
            Kokkos::View<GO**,HostDevice> GIDs = cells[b][e]->GIDs;
            for (int p=0; p<numElem; p++) {
              
              for( int i=0; i<numsb; i++ ) {
                int pindex = overlapped_map->LID(GIDs(p,suboffsets[n][i]));
                if (write_subgrid_state)
                  soln_computed(eprog,i) = (*(soln[b][m].second))[0][pindex];
                else
                  soln_computed(eprog,i) = (*(adjsoln[b][m].second))[0][pindex];
              }
              eprog++;
            }
          }
        }
        submesh->setSolutionFieldData(var, blockID, myElements, soln_computed);
      }
      
      if (cells[0][0]->have_cell_phi || cells[0][0]->have_cell_rotation) {
        Kokkos::View<ScalarT**,HostDevice> cdata("cell data",myElements.size(), 1);
        int eprog = 0;
        for (size_t b=0; b<cells.size(); b++) {
          for (size_t k=0; k<cells[b].size(); k++) {
            vector<size_t> cell_data_seed = cells[b][k]->cell_data_seed;
            for (int p=0; p<cells[b][k]->numElem; p++) {
              cdata(eprog,0) = cell_data_seed[p];
              eprog++;
            }
          }
        }
        submesh->setCellFieldData("mesh_data_seed", blockID, myElements, cdata);
      }
    
      
      if(isTD) {
        submesh->writeToExodus(soln[0][m].first);
      }
      else {
        submesh->writeToExodus(filename);
      }
    }
    
  }

  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Write the solution to a file at a specific time
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void writeSolution(const string & filename, const int & usernum, const int & timeindex) {
    
    /*
     string blockID = "eblock";
     
     //////////////////////////////////////////////////////////////
     // Re-create the subgrid mesh
     //////////////////////////////////////////////////////////////
     
     SubGridTools sgt(LocalComm, macroshape, shape, macronodes[usernum], macrosideinfo[usernum]);
     sgt.createSubMesh(numrefine);
     vector<vector<ScalarT> > nodes = sgt.getSubNodes();
     vector<vector<int> > connectivity = sgt.getSubConnectivity();
     vector<FCint> sideinfo = sgt.getSubSideinfo();
     
     panzer_stk::SubGridMeshFactory submeshFactory(shape, nodes, connectivity, blockID);
     Teuchos::RCP<panzer_stk::STK_Interface> submesh = submeshFactory.buildMesh(LocalComm->Comm());
     
     //////////////////////////////////////////////////////////////
     // Add in the necessary fields for plotting
     //////////////////////////////////////////////////////////////
     
     vector<string> subeBlocks;
     submesh->getElementBlockNames(subeBlocks);
     for (size_t j=0; j<varlist.size(); j++) {
     submesh->addSolutionField(varlist[j], subeBlocks[0]);
     }
     vector<string> subextrafieldnames = physics_RCP->getExtraFieldNames(0);
     for (size_t j=0; j<subextrafieldnames.size(); j++) {
     submesh->addSolutionField(subextrafieldnames[j], subeBlocks[0]);
     }
     vector<string> subextracellfields = physics_RCP->getExtraCellFieldNames(0);
     for (size_t j=0; j<subextracellfields.size(); j++) {
     submesh->addCellField(subextracellfields[j], subeBlocks[0]);
     }
     
     
     
     if (discparamnames.size() > 0) {
     for (size_t n=0; n<discparamnames.size(); n++) {
     int paramnumbasis = cells[0][0]->paramindex[n].size();
     if (paramnumbasis==1) {
     submesh->addCellField(discparamnames[n], subeBlocks[0]);
     }
     else {
     submesh->addSolutionField(discparamnames[n], subeBlocks[0]);
     }
     }
     }
     
     submeshFactory.completeMeshConstruction(*submesh,LocalComm->Comm());
     
     //////////////////////////////////////////////////////////////
     // Add fields to mesh
     //////////////////////////////////////////////////////////////
     
     
     vector<size_t> myElements;
     for (size_t e=0; e<cells[usernum].size(); e++) {
     myElements.push_back(e);
     }
     
     // Collect the subgrid solution
     for (int n = 0; n<varlist.size(); n++) { // change to subgrid numVars
     size_t numsb = cells[usernum][0]->index[n].size();
     FC soln_computed = FC(cells[usernum].size(), numsb);
     string var = varlist[n];
     for( size_t e=0; e<cells[usernum].size(); e++ ) {
     vector<int> GIDs = cells[usernum][e]->GIDs;
     vector<vector<int> > suboffsets = wkset[0]->offsets;
     
     for( int i=0; i<numsb; i++ ) {
     int pindex = overlapped_map->LID(GIDs[suboffsets[n][i]]);
     soln_computed(e,i) = (*(soln[usernum][timeindex].second))[0][pindex];
     //soln_computed(e,i) = (*(adjsoln[usernum][timeindex].second))[0][pindex];
     }
     }
     submesh->setSolutionFieldData(var, blockID, myElements, soln_computed);
     }
     
     ////////////////////////////////////////////////////////////////
     // Discretized Parameters
     ////////////////////////////////////////////////////////////////
     
     if (discparamnames.size() > 0) {
     for (size_t n=0; n<discparamnames.size(); n++) {
     FC soln_computed;
     bool isConstant = false;
     DRV subnodes = cells[usernum][0]->nodes;
     int numSubNodes = subnodes.dimension(0);
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
     
     vector<string> extracellfieldnames = physics_RCP->getExtraCellFieldNames(0);
     vector<FC> extracellfields;// = phys->getExtraFields(b);
     for (size_t j=0; j<extracellfieldnames.size(); j++) {
     FC efdata(cells[usernum].size(), 1);
     extracellfields.push_back(efdata);
     }
     for (size_t k=0; k<cells[usernum].size(); k++) {
     cells[usernum][k]->updateSolnWorkset(soln[usernum][timeindex].second, 0); // also updates ip, ijac
     cells[usernum][k]->updateData();
     wkset[0]->time = soln[usernum][timeindex].first;
     vector<FC> cfields = physics_RCP->getExtraCellFields(0);
     size_t j = 0;
     for (size_t g=0; g<cfields.size(); g++) {
     for (size_t h=0; h<cfields[g].dimension(0); h++) {
     extracellfields[j](k,0) = cfields[g](h,0);
     ++j;
     }
     }
     }
     
     for (size_t j=0; j<extracellfieldnames.size(); j++) {
     submesh->setCellFieldData(extracellfieldnames[j], blockID, myElements, extracellfields[j]);
     }
     
     submesh->writeToExodus(filename);
     */
  }
  
  
  ////////////////////////////////////////////////////////////////////////////////
  // Add in the sensor data
  ////////////////////////////////////////////////////////////////////////////////
  
  void addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                  const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                  const vector<basis_RCP> & basisTypes, const int & usernum) {
    for (size_t e=0; e<cells[usernum].size(); e++) {
      cells[usernum][e]->addSensors(sensor_points,sensor_loc_tol,sensor_data,
                                    have_sensor_data, basisTypes, basisTypes);
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Assemble the projection (mass) matrix
  ////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<Epetra_CrsMatrix>  getProjectionMatrix() {
    
    // Compute the mass matrix on a reference element
    Teuchos::RCP<Epetra_CrsMatrix>  mass = Teuchos::rcp( new Epetra_CrsMatrix(Copy, *overlapped_map, -1) );
    
    int usernum = 0;
    for (size_t e=0; e<cells[usernum].size(); e++) {
      int numElem = cells[usernum][e]->numElem;
      Kokkos::View<GO**,HostDevice> GIDs = cells[usernum][e]->GIDs;
      Kokkos::View<ScalarT***,AssemblyDevice> localmass = cells[usernum][e]->getMass();
      for (int c=0; c<numElem; c++) {
        for( size_t row=0; row<GIDs.dimension(1); row++ ) {
          int rowIndex = GIDs(c,row);
          for( size_t col=0; col<GIDs.dimension(1); col++ ) {
            int colIndex = GIDs(c,col);
            ScalarT val = localmass(c,row,col);
            mass->InsertGlobalValues(rowIndex, 1, &val, &colIndex);
          }
        }
      }
    }
    
    mass->FillComplete();
    
    Teuchos::RCP<Epetra_CrsMatrix>  glmass;
    if (LocalComm->getSize() > 1) {
      glmass = Teuchos::rcp( new Epetra_CrsMatrix(Copy, *owned_map, -1) );
      glmass->PutScalar(0.0);
      glmass->Export(*mass, *exporter, Add);
      glmass->FillComplete();
    }
    else {
      glmass = mass;
    }
    return glmass;
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Get the integration points
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV getIP() {
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
        for (size_t i=0; i<ip.dimension(1); i++) {
          for (size_t j=0; j<ip.dimension(2); j++) {
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
  
  DRV getIPWts() {
    int numip_per_cell = wkset[0]->numip;
    int usernum = 0; // doesn't really matter
    int totalip = 0;
    for (size_t e=0; e<cells[usernum].size(); e++) {
      totalip += numip_per_cell*cells[usernum][e]->numElem;
    }
    DRV refwts = DRV("refwts",1,totalip);
    int prog = 0;
    for (size_t e=0; e<cells[usernum].size(); e++) {
      DRV wts = wkset[0]->ref_wts;//cells[usernum][e]->ijac;
      int numElem = cells[usernum][e]->numElem;
      for (size_t c=0; c<numElem; c++) {
        for (size_t i=0; i<wts.dimension(0); i++) {
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
  
  pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis2(const DRV & pts) {
    
    size_t numpts = pts.dimension(1);
    size_t dimpts = pts.dimension(2);
    size_t numGIDs = cells[0][0]->GIDs.dimension(1);
    Kokkos::View<int**,AssemblyDevice> owners("owners",numpts,1+numGIDs);
    
    for (size_t e=0; e<cells[0].size(); e++) {
      int numElem = cells[0][e]->numElem;
      DRV nodes = cells[0][e]->nodes;
      for (int c=0; c<numElem;c++) {
        DRV refpts("refpts",1, numpts, dimpts);
        DRVint inRefCell("inRefCell",1,numpts);
        DRV cnodes("current nodes",1,nodes.dimension(1), nodes.dimension(2));
        for (int i=0; i<nodes.dimension(1); i++) {
          for (int j=0; j<nodes.dimension(2); j++) {
            cnodes(0,i,j) = nodes(e,i,j);
          }
        }
        
        CellTools<AssemblyDevice>::mapToReferenceFrame(refpts, pts, cnodes, *cellTopo);
        CellTools<AssemblyDevice>::checkPointwiseInclusion(inRefCell, refpts, *cellTopo, 0.0);
        
        for (size_t i=0; i<numpts; i++) {
          if (inRefCell(0,i) == 1) {
            owners(i,0) = c;//cells[0][e]->globalElemID[c];
            Kokkos::View<GO**,HostDevice> GIDs = cells[0][e]->GIDs;
            for (size_t j=0; j<numGIDs; j++) {
              owners(i,j+1) = GIDs(c,j);
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
      DRV nodes = cells[0][0]->nodes;
      DRV cnodes("current nodes",1,nodes.dimension(1), nodes.dimension(2));
      for (int i=0; i<nodes.dimension(1); i++) {
        for (int j=0; j<nodes.dimension(2); j++) {
          cnodes(0,i,j) = nodes(owners(i,0),i,j);
        }
      }
      CellTools<AssemblyDevice>::mapToReferenceFrame(refpt_buffer, cpt, cnodes, *cellTopo);
      DRV refpt("refpt",1,dimpts);
      Kokkos::deep_copy(refpt,Kokkos::subdynrankview(refpt_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
      
      Kokkos::View<int**,AssemblyDevice> offsets = wkset[0]->offsets;
      vector<int> usebasis = wkset[0]->usebasis;
      DRV basisvals("basisvals",offsets.dimension(0),numGIDs);
      for (size_t n=0; n<offsets.dimension(0); n++) {
        DRV bvals = DiscTools::evaluateBasis(basis_pointers[usebasis[n]], refpt);
        for (size_t m=0; m<offsets.dimension(1); m++) {
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
  
  pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis(const DRV & pts) {
    
    /*
     size_t numpts = pts.dimension(1);
     size_t dimpts = pts.dimension(2);
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
  
  Teuchos::RCP<Epetra_CrsMatrix>  getEvaluationMatrix(const DRV & newip, Teuchos::RCP<Epetra_Map> & ip_map) {
    Teuchos::RCP<Epetra_CrsMatrix>  map_over = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *overlapped_map, -1));
    Teuchos::RCP<Epetra_CrsMatrix>  map;
    if (LocalComm->getSize() > 1) {
      map = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *owned_map, -1));
    
      map->PutScalar(0.0);
      map->Export(*map_over, *exporter, Add);
      map->FillComplete(*owned_map, *owned_map);
    }
    else {
      map = map_over;
    }
    return map;
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Get the subgrid cell GIDs
  ////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<GO**,HostDevice> getCellGIDs(const int & cellnum) {
    return cells[0][cellnum]->GIDs;
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Update the subgrid parameters (will be depracated)
  ////////////////////////////////////////////////////////////////////////////////
  
  void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
    for (size_t b=0; b<wkset.size(); b++) {
      wkset[b]->params = params;
      wkset[b]->paramnames = paramnames;
    }
    physics_RCP->updateParameters(params, paramnames);
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // TMW: Is the following functions used/required ???
  ////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT**,AssemblyDevice> getCellFields(const int & usernum, const ScalarT & time) {
    
    /*
     vector<string> extracellfieldnames = physics_RCP->getExtraCellFieldNames(0);
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
     vector<FC> cfields = physics_RCP->getExtraCellFields(0);
     size_t j = 0;
     for (size_t g=0; g<cfields.size(); g++) {
     for (size_t h=0; h<cfields[g].dimension(0); h++) {
     extracellfields(k,j) = cfields[g](h,0);
     ++j;
     }
     }
     }
     
     return extracellfields;
     */
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void buildPreconditioner() {
    Teuchos::ParameterList MLList;
    ML_Epetra::SetDefaults("SA",MLList);
    MLList.set("ML output", 0);
    MLList.set("max levels",5);
    MLList.set("increasing or decreasing","increasing");
    int numEqns = varlist.size();
    
    MLList.set("PDE equations",numEqns);
    MLList.set("aggregation: type", "Uncoupled");
    MLList.set("smoother: type","IFPACK");
    MLList.set("smoother: sweeps",1);
    MLList.set("smoother: ifpack type","ILU");
    MLList.set("smoother: ifpack overlap",1);
    MLList.set("smoother: pre or post", "both");
    MLList.set("coarse: type","Amesos-KLU");
    MLPrec = new ML_Epetra::MultiLevelPreconditioner(*J, MLList);
    
  }
  
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void performGather(const size_t & block, const Teuchos::RCP<Epetra_MultiVector> & vec, const size_t & type,
                     const size_t & index) const {
    
    for (size_t e=0; e < cells[block].size(); e++) {
      cells[block][e]->setLocalSoln(vec, type, index);
    }
    
  }
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) {
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
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  // Static - do not depend on macro-element
  int dimension, time_steps;
  ScalarT initial_time, final_time;
  //Teuchos::RCP<LA_MpiComm> LocalComm;
  Teuchos::RCP<Teuchos::ParameterList> settings;
  string macroshape, shape, multiscale_method, error_type;
  int nummacroVars, subgridverbose, numrefine;
  topo_RCP cellTopo, macro_cellTopo;
  vector<basis_RCP> basis_pointers;
  
  vector<vector<int> > useBasis;
  
  Teuchos::RCP<Epetra_Map> param_owned_map;
  Teuchos::RCP<Epetra_Map> param_overlapped_map;
  Teuchos::RCP<Epetra_Export> param_exporter;
  Teuchos::RCP<Epetra_Import> param_importer;
  
  Teuchos::RCP<Epetra_MultiVector> res, res_over, d_um, du, du_glob;//, d_up;//,
  Teuchos::RCP<Epetra_MultiVector> u, u_dot, phi, phi_dot;
  Teuchos::RCP<Epetra_MultiVector> d_sub_res_overm, d_sub_resm, d_sub_u_prevm, d_sub_u_overm;
  
  Teuchos::RCP<Epetra_CrsGraph> owned_graph, overlapped_graph;
  Teuchos::RCP<Epetra_CrsMatrix>  J, sub_J_over, M, sub_M_over;
  
  bool filledJ, filledM, useDirect;
  vector<string> stoch_param_types;
  vector<ScalarT> stoch_param_means, stoch_param_vars, stoch_param_mins, stoch_param_maxs;
  int num_stochclassic_params, num_active_params;
  vector<string> stochclassic_param_names;
  
  Epetra_LinearProblem LinSys;
  
  ScalarT sub_NLtol, lintol;
  int sub_maxNLiter, liniter;
  
  Amesos_BaseSolver * AmSolver;
  Teuchos::RCP<Amesos2::Solver<Epetra_CrsMatrix,Epetra_MultiVector> > Am2Solver;
  Teuchos::RCP<Epetra_MultiVector> LA_rhs, LA_lhs;
  
  bool have_sym_factor, have_preconditioner, use_amesos2;
  ML_Epetra::MultiLevelPreconditioner * MLPrec;
  
  vector<string> varlist;
  vector<string> discparamnames;
  Teuchos::RCP<physics> physics_RCP;
  Teuchos::RCP<panzer::DOFManager<int,int> > DOF;
  Teuchos::RCP<AssemblyManager> sub_assembler;
  Teuchos::RCP<ParameterManager> sub_params;
  Teuchos::RCP<solver> subsolver;
  Teuchos::RCP<panzer_stk::STK_Interface> mesh;
  Teuchos::RCP<discretization> disc;
  Teuchos::RCP<FunctionInterface> functionManager;
  
  vector<Teuchos::RCP<Epetra_MultiVector> > Psol;
  
  // Dynamic - depend on the macro-element
  vector<DRV> macronodes;
  vector<Kokkos::View<int****,HostDevice> > macrosideinfo;
  int num_macro_time_steps;
  ScalarT macro_deltat;
  bool write_subgrid_state;
  
  // Collection of users
  vector<vector<Teuchos::RCP<cell> > > cells;
  
  bool have_mesh_data, have_rotations, have_rotation_phi, compute_mesh_data;
  bool have_multiple_data_files;
  string mesh_data_tag, mesh_data_pts_tag;
  int number_mesh_data_files, numSeeds;
  bool is_final_time;
  vector<int> randomSeeds;
  
  // Timers
  Teuchos::RCP<Teuchos::Time> sgfemSolverTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridSolver()");
  Teuchos::RCP<Teuchos::Time> sgfemInitialTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridSolver - set initial conditions");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver()");
  Teuchos::RCP<Teuchos::Time> sgfemSolnSensTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridSolnSens()");
  Teuchos::RCP<Teuchos::Time> sgfemFluxTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::updateFlux()");
  Teuchos::RCP<Teuchos::Time> sgfemFluxWksetTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::updateFlux - update workset");
  Teuchos::RCP<Teuchos::Time> sgfemFluxCellTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::updateFlux - cell computation");
  Teuchos::RCP<Teuchos::Time> sgfemSolnStorageTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::solutionStorage()");
  Teuchos::RCP<Teuchos::Time> sgfemComputeAuxBasisTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - compute aux basis functions");
  Teuchos::RCP<Teuchos::Time> sgfemSubMeshTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create subgrid meshes");
  Teuchos::RCP<Teuchos::Time> sgfemLinearAlgebraSetupTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - setup linear algebra");
  Teuchos::RCP<Teuchos::Time> sgfemTotalAddMacroTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro()");
  Teuchos::RCP<Teuchos::Time> sgfemMeshDataTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMeshData()");
  Teuchos::RCP<Teuchos::Time> sgfemSubCellTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create subcells");
  Teuchos::RCP<Teuchos::Time> sgfemSubDiscTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create disc. interface");
  Teuchos::RCP<Teuchos::Time> sgfemSubSolverTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create solver interface");
  Teuchos::RCP<Teuchos::Time> sgfemSubICTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create vectors");
  Teuchos::RCP<Teuchos::Time> sgfemSubSideinfoTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create side info");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverAllocateTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - allocate objects");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverSetSolnTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - set local soln");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverJacResTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - Jacobian/residual");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverInsertTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - insert");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverSolveTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - solve");
  
  
  
};
#endif

