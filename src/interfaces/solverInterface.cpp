/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "multiscaleInterface.hpp"
#include "solverInterface.hpp"
#include "discretizationTools.hpp"
#include "workset.hpp"
#include <boost/algorithm/string.hpp>


// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

solver::solver(const Teuchos::RCP<LA_MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
               Teuchos::RCP<panzer_stk::STK_Interface> & mesh_, Teuchos::RCP<discretization> & disc_,
               Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager<int,int> > & DOF_,
               vector<vector<Teuchos::RCP<cell> > > & cells_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_), DOF(DOF_), cells(cells_) {
  
  // Get the required information from the settings
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  numsteps = settings->sublist("Solver").get("numSteps",1);
  verbosity = settings->get<int>("verbosity",0);
  usestrongDBCs = settings->sublist("Solver").get<bool>("use strong DBCs",true);
  use_meas_as_dbcs = settings->sublist("Mesh").get<bool>("Use Measurements as DBCs", false);
  solver_type = settings->sublist("Solver").get<string>("solver","none"); // or "transient"
  allow_remesh = settings->sublist("Solver").get<bool>("Remesh",false);
  finaltime = settings->sublist("Solver").get<double>("finaltime",1.0);
  time_order = settings->sublist("Solver").get<int>("time order",1);
  NLtol = settings->sublist("Solver").get<double>("NLtol",1.0E-6);
  MaxNLiter = settings->sublist("Solver").get<int>("MaxNLiter",10);
  NLsolver = settings->sublist("Solver").get<string>("Nonlinear Solver","Newton");
  line_search = false;//settings->sublist("Solver").get<bool>("Use Line Search","false");
  store_adjPrev = false;
  
  isTransient = false;
  if (solver_type == "transient") {
    isTransient = true;
  }
  else {
    numsteps = 1;
  }
  
  isInitial = false;
  current_time = settings->sublist("Solver").get<double>("Initial Time",0.0);
  solvetimes.push_back(current_time);
  
  if (isTransient) {
    double deltat = finaltime / numsteps;
    double ctime = current_time; // local current time
    for (int timeiter = 0; timeiter < numsteps; timeiter++) {
      ctime += deltat;
      solvetimes.push_back(ctime);
    }
  }
  
  response_type = settings->sublist("Postprocess").get("response type", "pointwise"); // or "global"
  compute_objective = settings->sublist("Postprocess").get("compute objective",false);
  compute_sensitivity = settings->sublist("Postprocess").get("compute sensitivities",false);
  
  meshmod_xvar = settings->sublist("Solver").get<int>("Solution For x-Mesh Mod",-1);
  meshmod_yvar = settings->sublist("Solver").get<int>("Solution For y-Mesh Mod",-1);
  meshmod_zvar = settings->sublist("Solver").get<int>("Solution For z-Mesh Mod",-1);
  meshmod_TOL = settings->sublist("Solver").get<double>("Solution Based Mesh Mod TOL",1.0);
  meshmod_usesmoother = settings->sublist("Solver").get<bool>("Solution Based Mesh Mod Smoother",false);
  meshmod_center = settings->sublist("Solver").get<double>("Solution Based Mesh Mod Param",0.1);
  meshmod_layer_size = settings->sublist("Solver").get<double>("Solution Based Mesh Mod Layer Thickness",0.1);
  
  initial_type = settings->sublist("Solver").get<string>("Initial type","L2-projection");
  
  lintol = settings->sublist("Solver").get<double>("lintol",1.0E-7);
  liniter = settings->sublist("Solver").get<int>("liniter",100);
  kspace = settings->sublist("Solver").get<int>("krylov vectors",100);
  useDomDecomp = settings->sublist("Solver").get<bool>("use dom decomp",false);
  useDirect = settings->sublist("Solver").get<bool>("use direct solver",false);
  usePrec = settings->sublist("Solver").get<bool>("use preconditioner",true);
  dropTol = settings->sublist("Solver").get<double>("ILU drop tol",0.0); //defaults to AztecOO default
  fillParam = settings->sublist("Solver").get<double>("ILU fill param",3.0); //defaults to AztecOO default
  
  use_custom_initial_param_guess= settings->sublist("Physics").get<bool>("use custom initial param guess",false);
  
  // needed information from the mesh
  mesh->getElementBlockNames(blocknames);
  
  // needed information from the physics interface
  numVars = phys->numVars; //
  vector<vector<string> > phys_varlist = phys->varlist;
  //offsets = phys->offsets;
  
  // Set up the time integrator
  string timeinttype = settings->sublist("Solver").get<string>("Time integrator","RK");
  string timeintmethod = settings->sublist("Solver").get<string>("Time method","Implicit");
  int timeintorder = settings->sublist("Solver").get<int>("Time order",1);
  bool timeintstagger = settings->sublist("Solver").get<bool>("Stagger solutions",true);

  //if (timeinttype == "RK") {
  //  timeInt = Teuchos::rcp(new RungeKutta(timeintmethod,timeintorder,timeintstagger));
  //}
  
  // needed information from the DOF manager
  DOF->getOwnedIndices(LA_owned);
  numUnknowns = (int)LA_owned.size();
  DOF->getOwnedAndGhostedIndices(LA_ownedAndShared);
  numUnknownsOS = (int)LA_ownedAndShared.size();
  int localNumUnknowns = numUnknowns;
  
  DOF->getOwnedIndices(owned);
  DOF->getOwnedAndGhostedIndices(ownedAndShared);
  
  int nstages = 1;//timeInt->num_stages;
  bool sol_staggered = true;//timeInt->sol_staggered;
  /*
  LA_owned = vector(numUnknowns);
  LA_ownedAndShared = vector(numUnknownsOS);
  for (size_t i=0; i<numUnknowns; i++) {
    LA_owned[i] = owned[i];
  }
  for (size_t i=0; i<numUnknownsOS; i++) {
    LA_ownedAndShared[i] = ownedAndShared[i];
  }
   */
  //for (size_t i=0; i<numUnknowns; i++) {
    //for (int s=0; s<nstages; s++) {
    //  LA_owned[i*(nstages)+s] = owned[i]*nstages+s;
    //}
  //}
  //for (size_t i=0; i<numUnknownsOS; i++) {
  //  for (int s=0; s<nstages; s++) {
  //    LA_ownedAndShared[i*(nstages)+s] = ownedAndShared[i]*nstages+s;
  //  }
  //}
  
  globalNumUnknowns = 0;
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
  //Comm->SumAll(&localNumUnknowns, &globalNumUnknowns, 1);
  
  // needed information from the disc interface
  vector<vector<int> > cards = disc->cards;
  
  for (size_t b=0; b<cells.size(); b++) {
    
    vector<int> curruseBasis(numVars[b]);
    vector<int> currnumBasis(numVars[b]);
    vector<string> currvarlist(numVars[b]);
    
    int currmaxbasis = 0;
    for (int j=0; j<numVars[b]; j++) {
      string var = phys_varlist[b][j];
      int vnum = DOF->getFieldNum(var);
      int vub = phys->getUniqueIndex(b,var);
      currvarlist[j] = var;
      curruseBasis[j] = vub;
      currnumBasis[j] = cards[b][vub];
      //currvarlist[vnum] = var;
      //curruseBasis[vnum] = vub;
      //currnumBasis[vnum] = cards[b][vub];
      currmaxbasis = std::max(currmaxbasis,cards[b][vub]);
    }
    
    phys->setVars(b,currvarlist);
    
    varlist.push_back(currvarlist);
    useBasis.push_back(curruseBasis);
    numBasis.push_back(currnumBasis);
    maxbasis.push_back(currmaxbasis);
    
    vector<size_t> localIds;
    DRV blocknodes;
    panzer_stk::workset_utils::getIdsAndVertices(*mesh, blocknames[b], localIds, blocknodes);
    elemnodes.push_back(blocknodes);
    
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  /////////////////////////////////////////////////////////////////////////////
  
  num_inactive_params = 0;
  num_active_params = 0;
  num_stochastic_params = 0;
  num_discrete_params = 0;
  num_discretized_params = 0;
  globalParamUnknowns = 0;
  have_dRdP = false;
  discretized_stochastic = false;
  
  this->setupParameters(settings);
  
  /////////////////////////////////////////////////////////////////////////////
  // Epetra maps
  /////////////////////////////////////////////////////////////////////////////
  
  this->setupLinearAlgebra();
  
  /////////////////////////////////////////////////////////////////////////////
  // Worksets
  /////////////////////////////////////////////////////////////////////////////
  
  for (size_t b=0; b<cells.size(); b++) {
    wkset.push_back(Teuchos::rcp( new workset(cells[b][0]->getInfo(), disc->ref_ip[b],
                                              disc->ref_wts[b], disc->ref_side_ip[b],
                                              disc->ref_side_wts[b], disc->basis_types[b],
                                              disc->basis_pointers[b],
                                              discretized_param_basis,
                                              mesh->getCellTopology(blocknames[b])) ) );
    
    wkset[b]->isInitialized = true;
    wkset[b]->block = b;
    //wkset[b]->num_stages = nstages;
    vector<vector<int> > voffsets = phys->offsets[b];
    size_t maxoff = 0;
    for (size_t i=0; i<voffsets.size(); i++) {
      if (voffsets[i].size() > maxoff) {
        maxoff = voffsets[i].size();
      }
    }
    Kokkos::View<int**,HostDevice> offsets_host("offsets on host device",voffsets.size(),maxoff);
    for (size_t i=0; i<voffsets.size(); i++) {
      for (size_t j=0; j<voffsets[i].size(); j++) {
        offsets_host(i,j) = voffsets[i][j];
      }
    }
    Kokkos::View<int**,AssemblyDevice>::HostMirror offsets_device = Kokkos::create_mirror_view(offsets_host);
    Kokkos::deep_copy(offsets_host, offsets_device);
    wkset[b]->offsets = offsets_device;//phys->voffsets[b];
    
    size_t maxpoff = 0;
    for (size_t i=0; i<paramoffsets.size(); i++) {
      if (paramoffsets[i].size() > maxpoff) {
        maxpoff = paramoffsets[i].size();
      }
      //maxpoff = max(maxpoff,paramoffsets[i].size());
    }
    Kokkos::View<int**,HostDevice> poffsets_host("param offsets on host device",paramoffsets.size(),maxpoff);
    for (size_t i=0; i<paramoffsets.size(); i++) {
      for (size_t j=0; j<paramoffsets[i].size(); j++) {
        poffsets_host(i,j) = paramoffsets[i][j];
      }
    }
    Kokkos::View<int**,AssemblyDevice>::HostMirror poffsets_device = Kokkos::create_mirror_view(poffsets_host);
    Kokkos::deep_copy(poffsets_host, poffsets_device);
    
    wkset[b]->usebasis = useBasis[b];
    wkset[b]->paramusebasis = discretized_param_usebasis;
    wkset[b]->paramoffsets = poffsets_device;//paramoffsets;
    wkset[b]->varlist = varlist[b];
    int numDOF = cells[b][0]->GIDs[0].size();
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->wkset = wkset[b];
      cells[b][e]->setUseBasis(useBasis[b],nstages);
      cells[b][e]->setUpAdjointPrev(numDOF);
      cells[b][e]->setUpSubGradient(num_active_params);
    }
    
    wkset[b]->params = paramvals_AD;
    wkset[b]->params_AD = paramvals_KVAD;
    wkset[b]->paramnames = paramnames;
    
  }
  phys->setWorkset(wkset);
  
  if (settings->sublist("Mesh").get<bool>("Have Element Data", false) ||
      settings->sublist("Mesh").get<bool>("Have Nodal Data", false)) {
    this->readMeshData(settings);
  }
  
  /////////////////////////////////////////////////////////////////////////////
  
}

// ========================================================================================
// Set up the Epetra objects (maps, importers, exporters and graphs)
// These do need to be recomputed whenever the mesh changes */
// ========================================================================================

void solver::setupLinearAlgebra() {
  
  // Need to construct two types of vectors
  // One for storing the end-stage solutions of length N
  // Another for the linear algebra objects which are length N*s or (N*s)x(N*s)
  
  //sol_overlapped_map = Teuchos::rcp(new LA_Map(-1, (int)ownedAndShared.size(), &ownedAndShared[0], 0, *Comm));
  //sol_overlapped_graph = Teuchos::rcp(new LA_CrsGraph(Copy, *sol_overlapped_map, 0));
  
  //int nstages = timeInt->num_stages;
  //bool sol_staggered = timeInt->sol_staggered;
  
  
  if (USE_TPETRA) {
    
  }
  else {
    Epetra_MpiComm EP_Comm(*(Comm->getRawMpiComm()));
    LA_owned_map = Teuchos::rcp(new LA_Map(-1, (int)LA_owned.size(), &LA_owned[0], 0, EP_Comm));
    LA_overlapped_map = Teuchos::rcp(new LA_Map(-1, (int)LA_ownedAndShared.size(), &LA_ownedAndShared[0], 0, EP_Comm));
    
    LA_owned_graph = Teuchos::rcp(new LA_CrsGraph(Copy, *LA_owned_map, 0));
    LA_overlapped_graph = Teuchos::rcp(new LA_CrsGraph(Copy, *LA_overlapped_map, 0));
    
    exporter = Teuchos::rcp(new LA_Export(*LA_overlapped_map, *LA_owned_map));
    importer = Teuchos::rcp(new LA_Import(*LA_overlapped_map, *LA_owned_map));
  }
  vector<vector<int> > gids;
  
  for (size_t b=0; b<cells.size(); b++) {
    vector<vector<int> > curroffsets = phys->offsets[b];
    for(size_t e=0; e<cells[b].size(); e++) {
      gids = cells[b][e]->GIDs;
      
      int numElem = cells[b][e]->numElem;
      
      // this should fail on the first iteration through if maxDerivs is not large enough
      TEUCHOS_TEST_FOR_EXCEPTION(gids[0].size() > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of degrees of freedom per element times the number of time stages.");
      vector<vector<vector<int> > > cellindices;
      for (int p=0; p<numElem; p++) {
        vector<vector<int> > indices;
        for (int n=0; n<numVars[b]; n++) {
          vector<int> cindex;
          for( int i=0; i<numBasis[b][n]; i++ ) {
            int cgid = gids[p][curroffsets[n][i]]; // now an index into a block of DOFs
            cindex.push_back(LA_overlapped_map->LID(cgid));
          }
          indices.push_back(cindex);
        }
        
        for (size_t i=0; i<gids[p].size(); i++) {
          for (size_t j=0; j<gids[p].size(); j++) {
            //int err = LA_owned_graph->InsertGlobalIndices(gids[i],1,&gids[j]);
            //int err = LA_overlapped_graph->InsertGlobalIndices(gids[p][i],1,&gids[p][j]);
            int ind1 = gids[p][i];
            int ind2 = gids[p][j];
            int err = LA_overlapped_graph->InsertGlobalIndices(ind1,1,&ind2);
            /*for (int si=0; si<nstages; si++) {
              int ind1 = gids[p][i]*nstages+si;
              for (int sj=0; sj<nstages; sj++) {
                int ind2 = gids[p][j]*nstages+sj;
                int err = LA_overlapped_graph->InsertGlobalIndices(ind1,1,&ind2);
              }
            }*/
          }
        }
        cellindices.push_back(indices);
        
      }
      cells[b][e]->setIndex(cellindices);
    }
  }
  
  //LA_owned_graph->FillComplete();
  //sol_overlapped_graph->FillComplete();
  LA_overlapped_graph->FillComplete();
  
  if (num_discretized_params > 0) {
    if (USE_TPETRA) {
      
    }
    else {
      Epetra_MpiComm EP_Comm(*(Comm->getRawMpiComm()));
      param_owned_map = Teuchos::rcp(new LA_Map(-1, numParamUnknowns, &paramOwned[0], 0, EP_Comm));
      param_overlapped_map = Teuchos::rcp(new LA_Map(-1, (int)paramOwnedAndShared.size(), &paramOwnedAndShared[0], 0, EP_Comm));
    }
    
    param_exporter = Teuchos::rcp(new LA_Export(*param_overlapped_map, *param_owned_map));
    param_importer = Teuchos::rcp(new LA_Import(*param_overlapped_map, *param_owned_map));
    
    vector<vector<int> > gids;
    vector< vector<int> > param_nodesOS(numParamUnknownsOS); // should be overlapped
    vector< vector<int> > param_nodes(numParamUnknowns); // not overlapped -- for bounds
    vector< vector< vector<double> > > param_initial_vals; // custom initial guess set by cells
    DRV nodes;
    vector_RCP paramVec = this->setInitialParams(); // TMW: this will be deprecated soon
    
    for (size_t b=0; b<cells.size(); b++) {
      vector<vector<int> > curroffsets = phys->offsets[b];
      for(size_t e=0; e<cells[b].size(); e++) {
        gids = cells[b][e]->paramGIDs;
        // this should fail on the first iteration through if maxDerivs is not large enough
        TEUCHOS_TEST_FOR_EXCEPTION(gids[0].size() > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of parameter degrees of freedom per element.");
        
        vector<vector<vector<int> > > cellindices;
        int numElem = cells[b][e]->numElem;
        for (int p=0; p<numElem; p++) {
          
          vector<vector<int> > indices;
          
          for (int n=0; n<num_discretized_params; n++) {
            vector<int> cindex;
            for( int i=0; i<paramNumBasis[n]; i++ ) {
              int globalIndexOS = param_overlapped_map->LID(gids[p][paramoffsets[n][i]]);
              cindex.push_back(globalIndexOS);
              param_nodesOS[n].push_back(globalIndexOS);
              int globalIndex_owned = param_owned_map->LID(gids[p][paramoffsets[n][i]]);
              param_nodes[n].push_back(globalIndex_owned);
            }
            indices.push_back(cindex);
          }
          cellindices.push_back(indices);
        }
        cells[b][e]->setParamIndex(cellindices);
        /* // needs to be updated
        if (use_custom_initial_param_guess) {
          nodes = cells[b][e]->nodes;
          param_initial_vals = phys->udfunc->setInitialParams(nodes,cellindices);
          for (int p=0; p<numElem; p++) {
            for (int n = 0; n < num_discretized_params; n++) {
              for (int i = 0; i < cellindices[p][n].size(); i++) {
                paramVec->ReplaceGlobalValue(paramOwnedAndShared[cellindices[p][n][i]]
                                             ,0,param_initial_vals[p][n][i]);
              }
            }
          }
        }*/
      }
    }
    for (int n=0; n<num_discretized_params; n++) {
      std::sort(param_nodesOS[n].begin(), param_nodesOS[n].end());
      param_nodesOS[n].erase( std::unique(param_nodesOS[n].begin(),
                                          param_nodesOS[n].end()), param_nodesOS[n].end());
      
      std::sort(param_nodes[n].begin(), param_nodes[n].end());
      param_nodes[n].erase( std::unique(param_nodes[n].begin(),
                                        param_nodes[n].end()), param_nodes[n].end());
    }
    for (int n = 0; n < num_discretized_params; n++) {
      if (!use_custom_initial_param_guess) {
        for (size_t i = 0; i < param_nodesOS[n].size(); i++) {
          paramVec->ReplaceGlobalValue(paramOwnedAndShared[param_nodesOS[n][i]]
                                       ,0,initialParamValues[n]);
        }
      }
      paramNodesOS.push_back(param_nodesOS[n]); // store for later use
      paramNodes.push_back(param_nodes[n]); // store for later use
    }
    Psol.push_back(paramVec);
  }
  else {
    // set up a dummy parameter vector
    paramOwnedAndShared.push_back(0);
    if (USE_TPETRA) {
      
    }
    else {
      Epetra_MpiComm EP_Comm(*(Comm->getRawMpiComm()));
      param_overlapped_map = Teuchos::rcp(new LA_Map(-1, 1, &paramOwnedAndShared[0], 0, EP_Comm));
    }
    vector_RCP paramVec = this->setInitialParams(); // TMW: this will be deprecated soon
    Psol.push_back(paramVec);
  }
}

// ========================================================================================
// Set up the parameters (inactive, active, stochastic, discrete)
// Communicate these parameters back to the physics interface and the enabled modules
// ========================================================================================

void solver::setupParameters(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  Teuchos::ParameterList parameters;
  
  if (settings->isSublist("Parameters")) {
    parameters = settings->sublist("Parameters");
    Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
    while (pl_itr != parameters.end()) {
      Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
      vector<double> newparamvals;
      int numnewparams = 0;
      if (!newparam.isParameter("type") || !newparam.isParameter("usage")) {
        // print out error message
      }
      
      if (newparam.get<string>("type") == "scalar") {
        newparamvals.push_back(newparam.get<double>("value"));
        numnewparams = 1;
      }
      else if (newparam.get<string>("type") == "vector") {
        std::string filename = newparam.get<string>("source");
        std::ifstream fin(filename.c_str());
        std::istream_iterator<double> start(fin), end;
        vector<double> importedparamvals(start, end);
        for (size_t i=0; i<importedparamvals.size(); i++) {
          newparamvals.push_back(importedparamvals[i]);
        }
      }
      
      paramnames.push_back(pl_itr->first);
      paramvals.push_back(newparamvals);
      
      Teuchos::RCP<vector<AD> > newparam_AD = Teuchos::rcp(new vector<AD>(newparamvals.size()));
      paramvals_AD.push_back(newparam_AD);
      
      //blank bounds
      vector<double> lo(newparamvals.size(),0.0);
      vector<double> up(newparamvals.size(),0.0);
      
      if (newparam.get<string>("usage") == "inactive") {
        paramtypes.push_back(0);
        num_inactive_params += newparamvals.size();
      }
      else if (newparam.get<string>("usage") == "active") {
        paramtypes.push_back(1);
        num_active_params += newparamvals.size();
        
        //if active, look for actual bounds
        if(newparam.isParameter("bounds")){
          std::string filename = newparam.get<string>("bounds");
          FILE* BoundsFile = fopen(filename.c_str(),"r");
          float a,b;
          int i = 0;
          while( !feof(BoundsFile) ) {
            char line[100] = "";
            fgets(line,100,BoundsFile);
            if( strcmp(line,"") ) {
              sscanf(line, "%f %f", &a, &b);
              lo[i] = a;
              up[i] = b;
            }
            i++;
          }
        }
      }
      else if (newparam.get<string>("usage") == "stochastic") {
        paramtypes.push_back(2);
        num_stochastic_params += newparamvals.size();
        for (size_t i=0; i<newparamvals.size(); i++) {
          stochastic_distribution.push_back(newparam.get<string>("distribution","uniform"));
          stochastic_mean.push_back(newparam.get<double>("mean",0.0));
          stochastic_variance.push_back(newparam.get<double>("variance",1.0));
          stochastic_min.push_back(newparam.get<double>("min",0.0));
          stochastic_max.push_back(newparam.get<double>("max",0.0));
        }
      }
      else if (newparam.get<string>("usage") == "discrete") {
        paramtypes.push_back(3);
        num_discrete_params += newparamvals.size();
      }
      else if (newparam.get<string>("usage") == "discretized") {
        paramtypes.push_back(4);
        num_discretized_params += 1;
        if (!discretized_stochastic) { // once this is turned on, it stays on
          discretized_stochastic = newparam.get<bool>("stochastic",false);
        }
        discretized_param_basis_types.push_back(newparam.get<string>("type","HGRAD"));
        discretized_param_basis_orders.push_back(newparam.get<int>("order",1));
        discretized_param_names.push_back(pl_itr->first);
        initialParamValues.push_back(newparam.get<double>("initial_value",1.0));
        lowerParamBounds.push_back(newparam.get<double>("lower_bound",-1.0));
        upperParamBounds.push_back(newparam.get<double>("upper_bound",1.0));
        discparam_distribution.push_back(newparam.get<string>("distribution","uniform"));
        discparamVariance.push_back(newparam.get<double>("variance",1.0));
        if (newparam.get<bool>("isDomainParam",true)) {
          domainRegTypes.push_back(newparam.get<int>("reg_type",0));
          domainRegConstants.push_back(newparam.get<double>("reg_constant",0.0));
          domainRegIndices.push_back(num_discretized_params - 1);
        }
        else {
          boundaryRegTypes.push_back(newparam.get<int>("reg_type",0));
          boundaryRegConstants.push_back(newparam.get<double>("reg_constant",0.0));
          boundaryRegSides.push_back(newparam.get<string>("sides"," "));
          boundaryRegIndices.push_back(num_discretized_params - 1);
        }
      }
      
      paramLowerBounds.push_back(lo);
      paramUpperBounds.push_back(up);
      
      pl_itr++;
    }
    
    TEUCHOS_TEST_FOR_EXCEPTION(num_active_params > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of parameters.");
    
    if (num_discretized_params > 0) {
      // determine the unique list of basis'
      vector<int> disc_orders;
      vector<string> disc_types;
      vector<int> disc_usebasis;
      
      for (size_t j=0; j<discretized_param_basis_orders.size(); j++) {
        bool is_unique = true;
        for (size_t k=0; k<disc_orders.size(); k++) {
          if (disc_orders[k] == discretized_param_basis_orders[j] &&
              disc_types[k] == discretized_param_basis_types[j]) {
            is_unique = false;
            disc_usebasis.push_back(k);
          }
        }
        if (is_unique) {
          disc_orders.push_back(discretized_param_basis_orders[j]);
          disc_types.push_back(discretized_param_basis_types[j]);
          disc_usebasis.push_back(disc_orders.size()-1);
        }
      }
      
      discretized_param_basis_types = disc_types;
      discretized_param_basis_orders = disc_orders;
      discretized_param_usebasis = disc_usebasis;
      
      for (size_t n=0; n<disc_orders.size(); n++) {
        topo_RCP cellTopo = mesh->getCellTopology(blocknames[0]);
        basis_RCP basis = DiscTools::getBasis(spaceDim, cellTopo, disc_types[n],
                                              disc_orders[n]);
        discretized_param_basis.push_back(basis);
        
      }
      
      paramDOF = Teuchos::rcp(new panzer::DOFManager<int,int>());
      Teuchos::RCP<panzer::ConnManager<int,int> > conn = Teuchos::rcp(new panzer_stk::STKConnManager<int>(mesh));
      paramDOF->setConnManager(conn,*(Comm->getRawMpiComm()));
      
      Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern;
      
      for (size_t b=0; b<blocknames.size(); b++) {
        for (size_t j=0; j<discretized_param_names.size(); j++) {
          
          Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(discretized_param_basis[disc_usebasis[j]]));
          paramDOF->addField(blocknames[b], discretized_param_names[j], Pattern);
        }
      }
      
      paramDOF->buildGlobalUnknowns();
      paramDOF->getOwnedIndices(paramOwned);
      numParamUnknowns = (int)paramOwned.size();
      paramDOF->getOwnedAndGhostedIndices(paramOwnedAndShared);
      numParamUnknownsOS = (int)paramOwnedAndShared.size();
      int localParamUnknowns = numParamUnknowns;
      
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localParamUnknowns,&globalParamUnknowns);
      //Comm->SumAll(&localParamUnknowns, &globalParamUnknowns, 1);
      
      for (size_t j=0; j<discretized_param_names.size(); j++) {
        int num = paramDOF->getFieldNum(discretized_param_names[j]);
        vector<int> poffsets = paramDOF->getGIDFieldOffsets(blocknames[0],num);
        paramoffsets.push_back(poffsets);
        paramNumBasis.push_back(discretized_param_basis[discretized_param_usebasis[j]]->getCardinality());
      }
      
      for (size_t b=0; b<cells.size(); b++) {
        int eprog = 0;
        for (size_t e=0; e<cells[b].size(); e++) {
          vector<vector<int> > GIDs;
          int numElem = cells[b][e]->numElem;
          for (int p=0; p<numElem; p++) {
            size_t elemID = disc->myElements[b][eprog+p];
            vector<int> localGIDs;
            paramDOF->getElementGIDs(elemID, localGIDs, blocknames[b]);
            GIDs.push_back(localGIDs);
          }
          eprog += numElem;
          cells[b][e]->paramGIDs = GIDs;
          cells[b][e]->setParamUseBasis(disc_usebasis, paramNumBasis);
        }
      }
      
      if (discretized_stochastic) { // add the param DOFs as indep. rv's
        for (size_t j=0; j<numParamUnknownsOS; j++) {
          // hard coding for one disc param just to get something working
          stochastic_distribution.push_back(discparam_distribution[0]);
          stochastic_mean.push_back(initialParamValues[0]);
          stochastic_variance.push_back(discparamVariance[0]);
          stochastic_min.push_back(lowerParamBounds[0]);
          stochastic_max.push_back(upperParamBounds[0]);
          
        }
      }
    }
  }
  
  size_t maxcomp = 0;
  for (size_t k=0; k<paramvals.size(); k++) {
    if (paramvals[k].size() > maxcomp) {
      maxcomp = paramvals[k].size();
    }
  }
  
  paramvals_KVAD = Kokkos::View<AD**,AssemblyDevice>("parameter values (AD)", paramvals.size(), maxcomp);
  
  // Go through the physics interface to setup the parameters in the physics modules
  // phys->setParameters(paramnames);
  
}

/////////////////////////////////////////////////////////////////////////////
// Read in discretized data from an exodus mesh
/////////////////////////////////////////////////////////////////////////////

void solver::readMeshData(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  string exofile;
  string fname;
  
  
  exofile = settings->sublist("Mesh").get<std::string>("Mesh_File","mesh.exo");
  
  if (Comm->getSize() > 1) {
    stringstream ssProc, ssPID;
    ssProc << Comm->getSize();
    ssPID << Comm->getRank();
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
  CPU_word_size = sizeof(double);
  IO_word_size = 0;
  exoid = ex_open(fname.c_str(), EX_READ, &CPU_word_size,&IO_word_size,
                  &exo_version);
  exo_error = ex_get_init(exoid, title, &num_dim, &num_nods, &num_el,
                          &num_el_blk, &num_ns, &num_ss);
  
  int id = 1; // only one blkid
  int step = 1; // only one time step (for now)
  char elem_type[MAX_STR_LENGTH+1];
  ex_block eblock;
  eblock.id = id;
  eblock.type = EX_ELEM_BLOCK;
  
  exo_error = ex_get_block_param(exoid, &eblock);
  
  int num_el_in_blk = eblock.num_entry;
  int num_node_per_el = eblock.num_nodes_per_entry;
  
  
  // get elem vars
  if (settings->sublist("Mesh").get<bool>("Have Element Data", false)) {
    int num_elem_vars;
    int var_ind;
    numResponses = 1;
    exo_error = ex_get_var_param(exoid, "e", &num_elem_vars);
    for (int i=0; i<num_elem_vars; i++) {
      char varname[MAX_STR_LENGTH+1];
      double *var_vals = new double[num_el_in_blk];
      var_ind = i+1;
      exo_error = ex_get_variable_name(exoid, EX_ELEM_BLOCK, var_ind, varname);
      string vname(varname);
      efield_names.push_back(vname);
      size_t found = vname.find("Val");
      if (found != std::string::npos) {
        vector<string> results;
        stringstream sns, snr;
        int ns, nr;
        boost::split(results, vname, [](char u){return u == '_';});
        snr << results[3];
        snr >> nr;
        numResponses = std::max(numResponses,nr);
      }
      efield_vals.push_back(vector<double>(num_el_in_blk));
      exo_error = ex_get_var(exoid,step,EX_ELEM_BLOCK,var_ind,id,num_el_in_blk,var_vals);
      for (int j=0; j<num_el_in_blk; j++) {
        efield_vals[i][j] = var_vals[j];
      }
      delete [] var_vals;
    }
  }
  
  // assign nodal vars to meas multivector
  if (settings->sublist("Mesh").get<bool>("Have Nodal Data", false)) {
    int *connect = new int[num_el_in_blk*num_node_per_el];
    int edgeconn, faceconn;
    //exo_error = ex_get_elem_conn(exoid, id, connect);
    exo_error = ex_get_conn(exoid, EX_ELEM_BLOCK, id, connect, &edgeconn, &faceconn);
    
    // get nodal vars
    int num_node_vars;
    int var_ind;
    exo_error = ex_get_variable_param(exoid, EX_NODAL, &num_node_vars);
    for (int i=0; i<num_node_vars; i++) {
      char varname[MAX_STR_LENGTH+1];
      double *var_vals = new double[num_nods];
      var_ind = i+1;
      exo_error = ex_get_variable_name(exoid, EX_NODAL, var_ind, varname);
      string vname(varname);
      nfield_names.push_back(vname);
      nfield_vals.push_back(vector<double>(num_nods));
      exo_error = ex_get_var(exoid,step,EX_NODAL,var_ind,0,num_nods,var_vals);
      for (int j=0; j<num_nods; j++) {
        nfield_vals[i][j] = var_vals[j];
      }
      delete [] var_vals;
    }
    
    meas = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
    size_t b = 0;
    int index, dindex;
    vector<vector<int> > curroffsets = phys->offsets[b];
    for( size_t e=0; e<cells[b].size(); e++ ) {
      for (int n=0; n<numVars[b]; n++) {
        vector<vector<int> > GIDs = cells[b][e]->GIDs;
        for (int p=0; p<cells[b][e]->numElem; p++) {
          for( int i=0; i<numBasis[b][n]; i++ ) {
            index = LA_overlapped_map->LID(GIDs[p][curroffsets[n][i]]);
            dindex = connect[e*num_node_per_el + i] - 1;
            (*meas)[0][index] = nfield_vals[n][dindex];
          }
        }
      }
    }
    
    delete [] connect;
    
  }
  exo_error = ex_close(exoid);
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void solver::setupSensors(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  
  have_sensor_data = false;
  have_sensor_points = false;
  numSensors = 0;
  
  if (settings->sublist("Mesh").get<bool>("Have Element Data", false)) {
    
    for (size_t i=0; i<cells[0].size(); i++) {
      vector<Kokkos::View<double**,HostDevice> > sensorLocations;
      vector<Kokkos::View<double**,HostDevice> > sensorData;
      int numSensorsInCell = efield_vals[0][i];
      if (numSensorsInCell > 0) {
        cells[0][i]->mySensorIDs.push_back(numSensors); // hack for dakota
        for (size_t j=0; j<numSensorsInCell; j++) {
          // sensorLocation
          Kokkos::View<double**,HostDevice> sensor_loc("sensor location",1,spaceDim);
          stringstream ssSensorNum;
          ssSensorNum << j+1;
          string sensorNum = ssSensorNum.str();
          string fieldLocx = "sensor_" + sensorNum + "_Loc_x";
          ptrdiff_t ind_Locx = std::distance(efield_names.begin(), std::find(efield_names.begin(), efield_names.end(), fieldLocx));
          string fieldLocy = "sensor_" + sensorNum + "_Loc_y";
          ptrdiff_t ind_Locy = std::distance(efield_names.begin(), std::find(efield_names.begin(), efield_names.end(), fieldLocy));
          sensor_loc(0,0) = efield_vals[ind_Locx][i];
          sensor_loc(0,1) = efield_vals[ind_Locy][i];
          if (spaceDim > 2) {
            string fieldLocz = "sensor_" + sensorNum + "_Loc_z";
            ptrdiff_t ind_Locz = std::distance(efield_names.begin(), std::find(efield_names.begin(), efield_names.end(), fieldLocz));
            sensor_loc(0,2) = efield_vals[ind_Locz][i];
          }
          // sensorData
          Kokkos::View<double**,HostDevice> sensor_data("sensor data",1,numResponses+1);
          sensor_data(0,0) = 0.0; // time index
          for (size_t k=1; k<numResponses+1; k++) {
            stringstream ssRespNum;
            ssRespNum << k;
            string respNum = ssRespNum.str();
            string fieldResp = "sensor_" + sensorNum + "_Val_" + respNum;
            ptrdiff_t ind_Resp = std::distance(efield_names.begin(), std::find(efield_names.begin(), efield_names.end(), fieldResp));
            sensor_data(0,k) = efield_vals[ind_Resp][i];
          }
          sensorLocations.push_back(sensor_loc);
          sensorData.push_back(sensor_data);
          numSensors += 1; // solver variable (total number of sensors)
        }
      }
      cells[0][i]->exodus_sensors = true;
      cells[0][i]->numSensors = numSensorsInCell;
      cells[0][i]->sensorLocations = sensorLocations;
      cells[0][i]->sensorData = sensorData;
    }
    
    Kokkos::View<double**,HostDevice> tmp_sensor_points;
    vector<Kokkos::View<double**,HostDevice> > tmp_sensor_data;
    bool have_sensor_data = true;
    double sensor_loc_tol = 1.0;
    // only needed for passing of basis pointers
    for (size_t j=0; j<cells[0].size(); j++) {
      cells[0][j]->addSensors(sensor_points, sensor_loc_tol, sensor_data, have_sensor_data, disc->basis_pointers[0], discretized_param_basis);
    }
  }
  else {
    if (settings->sublist("Analysis").get("Have Sensor Data",false)) {
      data sdata("Sensor Measurements", spaceDim, settings->sublist("Analysis").get("Sensor Location File","sensor_points.dat"), settings->sublist("Analysis").get("Sensor Prefix","sensor"));
      sensor_data = sdata.getdata();
      sensor_points = sdata.getpoints();
      numSensors = sensor_points.dimension(0);
      have_sensor_data = true;
      have_sensor_points = true;
    }
    else if (settings->sublist("Analysis").get("Have Sensor Points",false)) {
      data sdata("Sensor Points", spaceDim, settings->sublist("Analysis").get("Sensor Location File","sensor_points.dat"));
      sensor_points = sdata.getpoints();
      numSensors = sensor_points.dimension(0);
      have_sensor_data = false;
      have_sensor_points = true;
    }
    
    if (settings->sublist("Analysis").get("Have Sensor Points",false)) {
      //sensor_locations = FCint(sensor_points.dimension(0),2);
      double sensor_loc_tol = settings->sublist("Analysis").get("Sensor location tol",1.0E-6);
      for (size_t b=0; b<cells.size(); b++) {
        for (size_t j=0; j<cells[b].size(); j++) {
          cells[b][j]->addSensors(sensor_points, sensor_loc_tol, sensor_data, have_sensor_data, disc->basis_pointers[b], discretized_param_basis);
        }
      }
    }
  }  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int solver::getNumParams(const int & type) {
  int np = 0;
  if (type == 0)
  np = num_inactive_params;
  else if (type == 1)
  np = num_active_params;
  else if (type == 2)
  np = num_stochastic_params;
  else if (type == 3)
  np = num_discrete_params;
  else if (type == 4)
  np = globalParamUnknowns;
  
  return np;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int solver::getNumParams(const std::string & type) {
  int np = 0;
  if (type == "inactive")
  np = num_inactive_params;
  else if (type == "active")
  np = num_active_params;
  else if (type == "stochastic")
  np = num_stochastic_params;
  else if (type == "discrete")
  np = num_discrete_params;
  else if (type == "discretized")
  np = num_discretized_params;
  
  return np;
}

// ========================================================================================
// return the discretized parameters as vector for use with ROL
// ========================================================================================

vector<double> solver::getDiscretizedParamsVector() {
  int numParams = this->getNumParams(4);
  vector<double> discLocalParams(numParams);
  vector<double> discParams(numParams);
  for (size_t i = 0; i < paramOwned.size(); i++) {
    int gid = paramOwned[i];
    discLocalParams[gid] = (*Psol[0])[0][i];
  }
  for (size_t i = 0; i < numParams; i++) {
    double globalval = 0.0;
    double localval = discLocalParams[i];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
    //Comm->SumAll(&localval, &globalval, 1);
    discParams[i] = globalval;
  }
  return discParams;
}

// ========================================================================================
/* given the parameters, solve the forward  problem */
// ========================================================================================

vector_RCP solver::forwardModel(DFAD & obj) {
  useadjoint = false;
  
  this->sacadoizeParams(false);
  
  // Set the initial condition
  //isInitial = true;
  
  vector_RCP initial = this->setInitial(); // TMW: this will be deprecated soon
  
  vector_RCP I_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
  int numsols = 1;
  if (solver_type == "transient") {
    numsols = numsteps+1;
  }
  
  vector_RCP F_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,numsols)); // empty solution
  vector_RCP zero_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
  if (solver_type == "transient") {
    for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
      (*F_soln)[0][i] = (*initial)[0][i];
    }
  }
  if (solver_type == "steady-state") {
    
    vector_RCP SS_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
    for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
      (*SS_soln)[0][i] = (*initial)[0][i];
    }
    
    this->nonlinearSolver(SS_soln, zero_soln, zero_soln, zero_soln, 0.0, 1.0);
    for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
      (*F_soln)[0][i] = (*SS_soln)[0][i];
    }
    
    if (compute_objective) {
      obj = this->computeObjective(F_soln, 0.0, 0);
    }
    
  }
  else if (solver_type == "transient") {
    vector<double> gradient; // not really used here
    this->transientSolver(initial, I_soln, F_soln, obj, gradient);
  }
  else {
    // print out an error message
  }
  
  return F_soln;
}

// ========================================================================================
/* given the parameters, solve the fractional forward  problem */
// ========================================================================================

vector_RCP solver::forwardModel_fr(DFAD & obj, double yt, double st) {
  useadjoint = false;
  wkset[0]->y = yt;
  wkset[0]->s = st;
  this->sacadoizeParams(false);
  
  // Set the initial condition
  //isInitial = true;
  
  vector_RCP initial = this->setInitial(); // TMW: this will be deprecated soon
  
  vector_RCP I_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
  int numsols = 1;
  if (solver_type == "transient") {
    numsols = numsteps+1;
  }
  
  vector_RCP F_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,numsols)); // empty solution
  vector_RCP zero_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
  if (solver_type == "transient") {
    for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
      (*F_soln)[0][i] = (*initial)[0][i];
    }
  }
  if (solver_type == "steady-state") {
    
    vector_RCP SS_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
    for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
      (*SS_soln)[0][i] = (*initial)[0][i];
    }
    
    this->nonlinearSolver(SS_soln, zero_soln, zero_soln, zero_soln, 0.0, 1.0);
    for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
      (*F_soln)[0][i] = (*SS_soln)[0][i];
    }
    
    if (compute_objective) {
      obj = this->computeObjective(F_soln, 0.0, 0);
    }
    
  }
  else if (solver_type == "transient") {
    vector<double> gradient; // not really used here
    this->transientSolver(initial, I_soln, F_soln, obj, gradient);
  }
  else {
    // print out an error message
  }
  
  return F_soln;
}

// ========================================================================================
// ========================================================================================

vector_RCP solver::adjointModel(vector_RCP & F_soln, vector<double> & gradient) {
  useadjoint = true;
  
  this->sacadoizeParams(false);
  
  //isInitial = true;
  vector_RCP initial = setInitial(); // does this need
  // to be updated for adjoint model?
  
  // Solve the forward problem
  int numsols = 1;
  if (solver_type == "transient") {
    numsols = numsteps+1;
  }
  
  vector_RCP zero_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
  vector_RCP A_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,numsols)); // empty solution
  for( size_t i=0; i<ownedAndShared.size(); i++ ) {
    (*A_soln)[0][i] = (*initial)[0][i];
  }
  
  if (solver_type == "steady-state") {
    vector_RCP L_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
    vector_RCP SS_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // empty solution
    for( size_t i=0; i<ownedAndShared.size(); i++ ) {
      (*L_soln)[0][i] = (*F_soln)[0][i];
    }
    this->nonlinearSolver(L_soln, zero_soln, SS_soln, zero_soln, 0.0, 1.0);
    for( size_t i=0; i<ownedAndShared.size(); i++ ) {
      (*A_soln)[0][i] = (*SS_soln)[0][i];
    }
    this->computeSensitivities(F_soln, zero_soln, A_soln, gradient, 0.0, 1.0);
    
  }
  else if (solver_type == "transient") {
    DFAD obj = 0.0;
    this->transientSolver(initial, F_soln, A_soln, obj, gradient);
  }
  else {
    // print out an error message
  }
  
  useadjoint = false;
  return A_soln;
}


// ========================================================================================
/* solve the problem */
// ========================================================================================

void solver::transientSolver(vector_RCP & initial, vector_RCP & L_soln,
                     vector_RCP & SolMat, DFAD & obj, vector<double> & gradient) {
  vector_RCP u = initial;
  vector_RCP u_dot = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1));
  vector_RCP phi = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1));
  vector_RCP phi_dot = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1));
  
  //int numSteps = 1;
  //double finaltime = 0.0;
  double deltat = 0.0;
  
  double alpha = 0.0;
  double beta = 1.0;
  
  deltat = finaltime / numsteps;
  if (time_order == 1){
    alpha = 1./deltat;
  }
  else if (time_order == 2) {
    alpha = 3.0/2.0/deltat;
  }
  else {
    alpha = 0.0; // would be better to print out an error message
  }
  
  int numivec = L_soln->NumVectors();
  
  //double current_time = 0.0;
  if (useadjoint) {
    current_time = finaltime;
    is_final_time = true;
  }
  else {
    current_time = solvetimes[0];
    is_final_time = false;
  }
  
  // ******************* ITERATE ON THE TIME STEPS **********************
  
  obj = 0.0;
  for (int timeiter = 0; timeiter<numsteps; timeiter++) {
    
    {
      Teuchos::TimeMonitor localtimer(*msprojtimer);
      msprojtimer->start();
      double my_cost = multiscale_manager->update();
      double gmin = 0.0;
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
      //Comm->MinAll(&my_cost, &gmin, 1);
      double gmax = 0.0;
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);
      //Comm->MaxAll(&my_cost, &gmax, 1);
      
      if(Comm->getRank() == 0 && verbosity>0) {
        cout << "***** Load Balancing Factor " << gmax/gmin <<  endl;
      }
    }
    
    if (!useadjoint) {
      current_time += deltat;
    }
    
    if(Comm->getRank() == 0 && verbosity > 0) {
      cout << endl << endl << "*******************************************************" << endl;
      cout << endl << "**** Beginning Time Step " << timeiter << endl;
      cout << "**** Current time is " << current_time << endl << endl;
      cout << "*******************************************************" << endl << endl << endl;
    }
    
    if (useadjoint) {
      // phi is updated automatically
      // need to update phi_dot, u, u_dot
      for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
        (*u)[0][i] = (*L_soln)[numivec-timeiter-1][i];
      }
      if (time_order == 1) {
        for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
          (*u_dot)[0][i] = alpha*(*L_soln)[numivec-timeiter-1][i] - alpha*(*L_soln)[numivec-timeiter-2][i];
          //phi_dot[0][i] = alpha*phi[0][i] - alpha*SolMat[timeiter][i];
        }
        phi_dot->PutScalar(0.0);
        
      }
      //else if (time_order == 2) { // TMW: not re-implemented yet
      //  for( size_t i=0; i<ownedAndShared.size(); i++ ) {
      //    u_dot[0][i] = alpha*L_soln[numivec-timeiter-1][i] - alpha*L_soln[numivec-timeiter-2][i];
      //    phi_dot[0][i] = alpha*phi[0][i] - alpha*SolMat[timeiter][i];
      //  }
    }
    else {
      // u is updated automatically
      // need to update u_dot (no need to update phi or phi_dot)
      if (time_order == 1 || timeiter == 0) {
        for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
          (*u_dot)[0][i] = alpha*(*u)[0][i] - alpha*(*SolMat)[timeiter][i];
        }
      }
      else if (time_order == 2) {
        for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
          (*u_dot)[0][i] = alpha*(*u)[0][i] - alpha*4.0/3.0*(*SolMat)[timeiter][i] + alpha*1.0/3.0*(*SolMat)[timeiter-1][i];
        }
      }
    }
    
    this->nonlinearSolver(u, u_dot, phi, phi_dot, alpha, beta);
    
    if (!useadjoint) {
      for( int i=0; i<LA_ownedAndShared.size(); i++ ) {
        (*SolMat)[timeiter+1][i] = (*u)[0][i];
      }
    }
    else {
      for( int i=0; i<LA_ownedAndShared.size(); i++ ) {
        (*SolMat)[timeiter+1][i] = (*phi)[0][i];
      }
    }
    
    
    //solvetimes.push_back(current_time); - This was causing a bug
    
    if (allow_remesh && !useadjoint) {
      this->remesh(u);
    }
    
    if (useadjoint) { // fill in the gradient
      this->computeSensitivities(u,u_dot,phi,gradient,alpha,beta);
      this->sacadoizeParams(false);
    }
    else if (compute_objective) { // fill in the objective function
      DFAD cobj = this->computeObjective(u, current_time, timeiter);
      obj += cobj;
      this->sacadoizeParams(false);
    }
    
    if (useadjoint) {
      current_time -= deltat;
      is_final_time = false;
    }
    
    //if (subgridModels.size() > 0) { // meaning we have multiscale turned on
    //  // give the cells the opportunity to change subgrid models for the next time step
    //  for (size_t b=0; b<cells.size(); b++) {
    //    for (size_t e=0; e<cells[b].size(); e++) {
    //      cells[b][e]->updateSubgridModel(subgridModels, phys->udfunc, *(wkset[b]));
    //    }
    //  }
    //}
    //isInitial = false; // only true on first time step
  }
}

// ========================================================================================
// ========================================================================================


void solver::nonlinearSolver(vector_RCP & u, vector_RCP & u_dot,
                     vector_RCP & phi, vector_RCP & phi_dot,
                     const double & alpha, const double & beta) {
  // current_time, alpha, and beta are determined by the time integrator / steady state solver
  
  double NLerr_first = 10*NLtol;
  double NLerr_scaled = NLerr_first;
  double NLerr = NLerr_first;
  int NLiter = 0;
  
  if (usestrongDBCs) {
    this->setDirichlet(u);
  }
  
  //this->setConstantPin(u); //pinning attempt
  int maxiter = MaxNLiter;
  if (useadjoint) {
    maxiter = 2;
  }
  
  while( NLerr_scaled>NLtol && NLiter<maxiter ) { // while not converged
    
    gNLiter = NLiter;
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,1));
    matrix_RCP J = Teuchos::rcp(new LA_CrsMatrix(Copy, *LA_owned_map, -1));
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1));
    matrix_RCP J_over = Teuchos::rcp(new LA_CrsMatrix(Copy, *LA_overlapped_graph));
    vector_RCP du = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1));
    vector_RCP du_over = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,1));
    
    // *********************** COMPUTE THE JACOBIAN AND THE RESIDUAL **************************
    
    bool build_jacobian = true;
    if (NLsolver == "AA")
    build_jacobian = false;
    
    res_over->PutScalar(0.0);
    J_over->PutScalar(0.0);
    if ( useadjoint && (NLiter == 1))
      store_adjPrev = true;
    else
      store_adjPrev = false;
    
    this->computeJacRes(u, u_dot, phi, phi_dot, alpha, beta, build_jacobian, false, false, res_over, J_over);
  
    J->PutScalar(0.0);
    J->Export(*J_over, *exporter, Add);
    J->FillComplete();
    res->PutScalar(0.0);
    res->Export(*res_over, *exporter, Add);
    
    // *********************** CHECK THE NORM OF THE RESIDUAL **************************
    if (NLiter == 0) {
      res->NormInf(&NLerr_first);
      if (NLerr_first > 1.0e-14)
      NLerr_scaled = 1.0;
      else
      NLerr_scaled = 0.0;
    }
    else {
      res->NormInf(&NLerr);
      NLerr_scaled = NLerr/NLerr_first;
    }
    
    if(Comm->getRank() == 0 && verbosity > 1) {
      cout << endl << "*********************************************************" << endl;
      cout << "***** Iteration: " << NLiter << endl;
      cout << "***** Norm of nonlinear residual: " << NLerr << endl;
      cout << "***** Scaled Norm of nonlinear residual: " << NLerr_scaled << endl;
      cout << "*********************************************************" << endl;
    }
    
    // *********************** SOLVE THE LINEAR SYSTEM **************************
    
    if (NLerr_scaled > NLtol) {
      
      this->linearSolver(J, res, du_over);
      
      du->Import(*du_over, *importer, Add);
      
      if (useadjoint) {
        phi->Update(1.0, *du, 1.0);
        phi_dot->Update(alpha, *du, 1.0);
      }
      else {
        u->Update(1.0, *du, 1.0);
        u_dot->Update(alpha, *du, 1.0);
      }
      
      /*
       if (line_search) {
       double err0 = NLerr;
       res_over.PutScalar(0.0);
       this->computeJacRes(u, u_dot, phi, phi_dot, alpha, beta, false, false, false, res_over, J_over);
       res.PutScalar(0.0);
       res.Export(res_over, *exporter, Add);
       
       double err1;
       res.NormInf(&err1);
       
       if (useadjoint) {
       phi.Update(-0.5, du, 1.0);
       phi_dot.Update(-0.5*alpha, du, 1.0);
       }
       else {
       u.Update(-0.5, du, 1.0);
       u_dot.Update(-0.5*alpha, du, 1.0);
       }
       res_over.PutScalar(0.0);
       this->computeJacRes(u, u_dot, phi, phi_dot, alpha, beta, false, false, false, res_over, J_over);
       res.PutScalar(0.0);
       res.Export(res_over, *exporter, Add);
       
       double errhalf;
       res.NormInf(&errhalf);
       
       double opt_alpha = -(-3.0*err0+4.0*errhalf-err1) / (2.0*(2.0*err0-4.0*errhalf+2.9*err1));
       if (opt_alpha > 1.0)
       opt_alpha = 1.0;
       else if (opt_alpha < 0.0)
       opt_alpha = 0.1;
       
       if(Comm->MyPID() == 0 && verbosity > 10) {
       cout << "Optimal step size: " << opt_alpha << endl;
       cout << "err0 " << err0 << endl;
       cout << "errhalf " << errhalf << endl;
       cout << "err1 " << err1 << endl;
       }
       
       if (useadjoint) {
       phi.Update(opt_alpha-0.5, du, 1.0);
       phi_dot.Update(alpha*(opt_alpha-0.5), du, 1.0);
       }
       else {
       u.Update(opt_alpha-0.5, du, 1.0);
       u_dot.Update(alpha*(opt_alpha-0.5), du, 1.0);
       }
       
       }*/
      
    }
    
    NLiter++; // increment number of iterations
  } // while loop
  
  if(Comm->getRank() == 0) {
    if (!useadjoint) {
      if( (NLiter>MaxNLiter || NLerr_scaled>NLtol) && verbosity > 1) {
        cout << endl << endl << "********************" << endl;
        cout << endl << "SOLVER FAILED TO CONVERGE CONVERGED in " << NLiter
        << " iterations with residual norm " << NLerr << endl;
        cout << "********************" << endl;
      }
    }
  }
  
}


// ========================================================================================
// ========================================================================================

void solver::remesh(const vector_RCP & u) {
  
  for (size_t b=0; b<cells.size(); b++) {
    for( size_t e=0; e<cells[b].size(); e++ ) {
      vector<vector<int> > GIDs = cells[b][e]->GIDs;
      DRV nodes = cells[b][e]->nodes;
      vector<vector<int> > offsets = phys->offsets[b];
      bool changed = false;
      for (int p=0; p<cells[b][e]->numElem; p++) {
        
        for( int i=0; i<nodes.dimension(1); i++ ) {
          if (meshmod_xvar >= 0) {
            int pindex = LA_overlapped_map->LID(GIDs[p][offsets[meshmod_xvar][i]]);
            double xval = (*u)[0][pindex];
            double xpert = xval;
            if (meshmod_usesmoother)
            xpert = meshmod_layer_size*(1.0/3.14159*atan(100.0*(xval-meshmod_center)+0.5));
            
            if (xpert > meshmod_TOL) {
              nodes(p,i,0) += xpert;
              changed = true;
            }
          }
          if (meshmod_yvar >= 0) {
            int pindex = LA_overlapped_map->LID(GIDs[p][offsets[meshmod_yvar][i]]);
            double yval = (*u)[0][pindex];
            double ypert = yval;
            if (meshmod_usesmoother)
            ypert = meshmod_layer_size*(1.0/3.14159*atan(100.0*(yval-meshmod_center)+0.5));
            
            if (ypert > meshmod_TOL) {
              nodes(p,i,1) += ypert;
              changed = true;
            }
          }
          if (meshmod_zvar >= 0) {
            int pindex = LA_overlapped_map->LID(GIDs[p][offsets[meshmod_zvar][i]]);
            double zval = (*u)[0][pindex];
            double zpert = zval;
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
}

// ========================================================================================
// ========================================================================================

DFAD solver::computeObjective(const vector_RCP & F_soln, const double & time, const size_t & tindex) {
  
  DFAD totaldiff = 0.0;
  AD regDomain = 0.0;
  AD regBoundary = 0.0;
  int numDomainParams = domainRegIndices.size();
  int numBoundaryParams = boundaryRegIndices.size();
  
  this->sacadoizeParams(true);
  
  int numParams = num_active_params + globalParamUnknowns;
  vector<double> regGradient(numParams);
  vector<double> dmGradient(numParams);
  
  for (size_t b=0; b<cells.size(); b++) {
    
    this->performGather(b, F_soln, 0, 0);
    this->performGather(b, Psol[0], 4, 0);
    
    for (size_t e=0; e<cells[b].size(); e++) {
      
      Kokkos::View<AD**,AssemblyDevice> obj = cells[b][e]->computeObjective(time, tindex, 0);
      vector<vector<int> > paramGIDs = cells[b][e]->paramGIDs;
      int numElem = cells[b][e]->numElem;
      
      if (obj.dimension(1) > 0) {
        for (int c=0; c<numElem; c++) {
          for (size_t i=0; i<obj.dimension(1); i++) {
            totaldiff += obj(c,i);
            if (num_active_params > 0) {
              if (obj(c,i).size() > 0) {
                double val;
                val = obj(c,i).fastAccessDx(0);
                dmGradient[0] += val;
              }
            }
            
            if (globalParamUnknowns > 0) {
              for (int row=0; row<paramoffsets[0].size(); row++) {
                int rowIndex = paramGIDs[c][paramoffsets[0][row]];
                int poffset = paramoffsets[0][row];
                double val;
                if (obj(c,i).size() > num_active_params) {
                  val = obj(c,i).fastAccessDx(poffset+num_active_params);
                  dmGradient[rowIndex+num_active_params] += val;
                }
              }
            }
          }
        }
      }
      
      if ((numDomainParams > 0) || (numBoundaryParams > 0)) {
        
        vector<vector<int> > paramGIDs = cells[b][e]->paramGIDs;
        
        if (numDomainParams > 0) {
          int paramIndex, rowIndex, poffset;
          double val;
          regDomain = cells[b][e]->computeDomainRegularization(domainRegConstants,
                                                               domainRegTypes, domainRegIndices);
          
          for (int c=0; c<numElem; c++) {
            for (size_t p = 0; p < numDomainParams; p++) {
              paramIndex = domainRegIndices[p];
              for( size_t row=0; row<paramoffsets[paramIndex].size(); row++ ) {
                if (regDomain.size() > 0) {
                  rowIndex = paramGIDs[c][paramoffsets[paramIndex][row]];
                  poffset = paramoffsets[paramIndex][row];
                  val = regDomain.fastAccessDx(poffset);
                  regGradient[rowIndex+num_active_params] += val;
                }
              }
            }
          }
        }
        
      
        if (numBoundaryParams > 0) {
          int paramIndex, rowIndex, poffset;
          double val;
          
          regBoundary = cells[b][e]->computeBoundaryRegularization(boundaryRegConstants,
                                                                   boundaryRegTypes, boundaryRegIndices,
                                                                   boundaryRegSides);
          for (int c=0; c<numElem; c++) {
            for (size_t p = 0; p < numBoundaryParams; p++) {
              paramIndex = boundaryRegIndices[p];
              for( size_t row=0; row<paramoffsets[paramIndex].size(); row++ ) {
                if (regBoundary.size() > 0) {
                  rowIndex = paramGIDs[c][paramoffsets[paramIndex][row]];
                  poffset = paramoffsets[paramIndex][row];
                  val = regBoundary.fastAccessDx(poffset);
                  regGradient[rowIndex+num_active_params] += val;
                }
              }
            }
          }
        }
        
        
        totaldiff += (regDomain + regBoundary);
        
      }
      
    }
    //totaldiff += phys->computeTopoResp(b);
  }
  
  //to gather contributions across processors
  double meep = 0.0;
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&totaldiff.val(),&meep);
  //Comm->SumAll(&totaldiff.val(), &meep, 1);
  totaldiff.val() = meep;
  
  DFAD fullobj(numParams,meep);
  
  for (size_t j=0; j< numParams; j++) {
    double dval;
    double ldval = dmGradient[j] + regGradient[j];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&ldval,&dval);
    //Comm->SumAll(&ldval,&dval,1);
    fullobj.fastAccessDx(j) = dval;
  }
  
  return fullobj;
  
}

// ========================================================================================
// ========================================================================================

vector<double> solver::computeSensitivities(const vector_RCP & GF_soln,
                                    const vector_RCP & GA_soln) {
  if(Comm->getRank() == 0 && verbosity>0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Computing Sensitivities ******" << endl << endl;
  }
  
  vector_RCP u = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // forward solution
  vector_RCP a2 = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,1)); // adjoint solution
  vector_RCP u_dot = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // previous solution (can be either fwd or adj)
  
  double alpha = 0.0;
  double beta = 1.0;
  
  vector<double> gradient(num_active_params);
  
  this->sacadoizeParams(true);
  
  vector<double> localsens(num_active_params);
  double globalsens = 0.0;
  int nsteps = 1;
  if (isTransient)
  nsteps = solvetimes.size()-1;
  
  for (int timeiter = 0; timeiter<nsteps; timeiter++) {
    
    if (isTransient) {
      current_time = solvetimes[timeiter+1];
      for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
        (*u_dot)[0][i] = alpha*((*GF_soln)[timeiter+1][i] - (*GF_soln)[timeiter][i]);
        (*u)[0][i] = (*GF_soln)[timeiter+1][i];
      }
      for( size_t i=0; i<LA_owned.size(); i++ ) {
        (*a2)[0][i] = (*GA_soln)[nsteps-timeiter][i];
      }
    }
    else {
      current_time = solvetimes[timeiter];
      for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
        (*u)[0][i] = (*GF_soln)[timeiter][i];
      }
      for( size_t i=0; i<LA_owned.size(); i++ ) {
        (*a2)[0][i] = (*GA_soln)[nsteps-timeiter-1][i];
      }
    }
    
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,num_active_params)); // reset residual
    matrix_RCP J = Teuchos::rcp(new LA_CrsMatrix(Copy, *LA_owned_map, -1)); // reset Jacobian
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,num_active_params)); // reset residual
    matrix_RCP J_over = Teuchos::rcp(new LA_CrsMatrix(Copy, *LA_overlapped_map, -1)); // reset Jacobian
    res_over->PutScalar(0.0);
    
    this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, false, true, false, res_over, J_over);
    
    res->PutScalar(0.0);
    res->Export(*res_over, *exporter, Add);
    
    for (size_t paramiter=0; paramiter < num_active_params; paramiter++) {
      double currsens = 0.0;
      for( size_t i=0; i<LA_owned.size(); i++ ) {
        currsens += (*a2)[0][i] * (*res)[paramiter][i];
      }
      localsens[paramiter] -= currsens;
    }
  }
  
  double localval = 0.0;
  double globalval = 0.0;
  for (size_t paramiter=0; paramiter < num_active_params; paramiter++) {
    localval = localsens[paramiter];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
    //Comm->SumAll(&localval, &globalval, 1);
    gradient[paramiter] = globalval;
  }
  
  if(Comm->getRank() == 0 && batchID == 0) {
    stringstream ss;
    std::string sname2 = "sens.dat";
    ofstream sensOUT(sname2.c_str());
    sensOUT.precision(16);
    for (size_t paramiter=0; paramiter < num_active_params; paramiter++) {
      sensOUT << gradient[paramiter] << "  ";
    }
    sensOUT << endl;
    sensOUT.close();
  }
  
  return gradient;
}


// ========================================================================================
// Compute the sensitivity of the objective with respect to discretized parameters
// ========================================================================================

vector<double> solver::computeDiscretizedSensitivities(const vector_RCP & F_soln,
                                               const vector_RCP & A_soln) {
  
  if(Comm->getRank() == 0 && verbosity>0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Computing Discretized Sensitivities ******" << endl << endl;
  }
  
  vector_RCP u = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // forward solution
  vector_RCP a2 = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,1)); // adjoint solution
  vector_RCP u_dot = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // previous solution (can be either fwd or adj)
  
  double alpha = 0.0;
  double beta = 1.0;
  
  this->sacadoizeParams(false);
  
  int nsteps = 1;
  if (isTransient)
  nsteps = solvetimes.size()-1;
  
  vector_RCP totalsens = Teuchos::rcp(new LA_MultiVector(*param_owned_map,1));
  
  
  for (int timeiter = 0; timeiter<nsteps; timeiter++) {
    
    if (isTransient) {
      current_time = solvetimes[timeiter+1];
      for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
        (*u_dot)[0][i] = alpha*((*F_soln)[timeiter+1][i] - (*F_soln)[timeiter][i]);
        (*u)[0][i] = (*F_soln)[timeiter+1][i];
      }
      for( size_t i=0; i<LA_owned.size(); i++ ) {
        (*a2)[0][i] = (*A_soln)[nsteps-timeiter][i];
      }
    }
    else {
      current_time = solvetimes[timeiter];
      for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
        (*u)[0][i] = (*F_soln)[timeiter][i];
      }
      for( size_t i=0; i<LA_owned.size(); i++ ) {
        (*a2)[0][i] = (*A_soln)[nsteps-timeiter-1][i];
      }
    }
    /*
     current_time = solvetimes[timeiter+1];
     for( size_t i=0; i<ownedAndShared.size(); i++ ) {
     u[0][i] = F_soln[timeiter+1][i];
     u_dot[0][i] = alpha*(F_soln[timeiter+1][i] - F_soln[timeiter][i]);
     }
     for( size_t i=0; i<owned.size(); i++ ) {
     a2[0][i] = A_soln[nsteps-timeiter][i];
     }
     */
    
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // reset residual
    matrix_RCP J_over = Teuchos::rcp(new LA_CrsMatrix(Copy, *param_overlapped_map, -1)); // reset Jacobian
    matrix_RCP J = Teuchos::rcp(new LA_CrsMatrix(Copy, *param_owned_map, -1)); // reset Jacobian
    this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, true, false, true, res_over, J_over);
    
    vector_RCP sens_over = Teuchos::rcp(new LA_MultiVector(*param_overlapped_map,1)); // reset residual
    vector_RCP sens = Teuchos::rcp(new LA_MultiVector(*param_owned_map,1)); // reset residual
    
    J->PutScalar(0.0);
    J->Export(*J_over, *param_exporter, Add);
    J->FillComplete(*LA_owned_map, *param_owned_map);
    
    J->Apply(*a2,*sens);
    
    totalsens->Update(1.0, *sens, 1.0);
  }
  
  dRdP.push_back(totalsens);
  have_dRdP = true;
  
  int numParams = this->getNumParams(4);
  vector<double> discLocalGradient(numParams);
  vector<double> discGradient(numParams);
  for (size_t i = 0; i < paramOwned.size(); i++) {
    int gid = paramOwned[i];
    discLocalGradient[gid] = (*totalsens)[0][i];
  }
  for (size_t i = 0; i < numParams; i++) {
    double globalval = 0.0;
    double localval = discLocalGradient[i];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
    //Comm->SumAll(&localval, &globalval, 1);
    discGradient[i] = globalval;
  }
  return discGradient;
}


// ========================================================================================
// ========================================================================================

void solver::computeSensitivities(vector_RCP & u, vector_RCP & u_dot,
                          vector_RCP & a2, vector<double> & gradient,
                          const double & alpha, const double & beta) {
  
  DFAD obj_sens = this->computeObjective(u, current_time, 0);
  
  if (num_active_params > 0) {
  
    this->sacadoizeParams(true);
    
    vector<double> localsens(num_active_params);
    double globalsens = 0.0;
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,num_active_params)); // reset residual
    matrix_RCP J = Teuchos::rcp(new LA_CrsMatrix(Copy, *LA_owned_map, -1)); // reset Jacobian
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,num_active_params)); // reset residual
    matrix_RCP J_over = Teuchos::rcp(new LA_CrsMatrix(Copy, *LA_overlapped_map, -1)); // reset Jacobian
    res_over->PutScalar(0.0);
    
    bool curradjstatus = useadjoint;
    useadjoint = false;
    
    this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, false, true, false, res_over, J_over);
    useadjoint = curradjstatus;
    
    res->PutScalar(0.0);
    res->Export(*res_over, *exporter, Add);
  
    for (size_t paramiter=0; paramiter < num_active_params; paramiter++) {
      // fine-scale
      if (cells[0][0]->multiscale) {
        double subsens = 0.0;
        for (size_t b=0; b<cells.size(); b++) {
          for (size_t e=0; e<cells[b].size(); e++) {
            subsens = -cells[b][e]->subgradient(0,paramiter);
            localsens[paramiter] += subsens;
          }
        }
      }
      else { // coarse-scale
      
        double currsens = 0.0;
        for( size_t i=0; i<LA_owned.size(); i++ ) {
          currsens += (*a2)[0][i] * (*res)[paramiter][i];
        }
        localsens[paramiter] = -currsens;
      }
      
    }
    
    
    double localval = 0.0;
    double globalval = 0.0;
    for (size_t paramiter=0; paramiter < num_active_params; paramiter++) {
      localval = localsens[paramiter];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      double cobj = 0.0;
      if (paramiter<obj_sens.size()) {
        cobj = obj_sens.fastAccessDx(paramiter);
      }
      globalval += cobj;
      if (gradient.size()<=paramiter) {
        gradient.push_back(globalval);
      }
      else {
        gradient[paramiter] += globalval;
      }
    }
  }
  
  int numDiscParams = this->getNumParams(4);
  
  if (numDiscParams > 0) {
    this->sacadoizeParams(false);
    
    
    vector_RCP a_owned = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,1)); // adjoint solution
    for( size_t i=0; i<LA_owned.size(); i++ ) {
      (*a_owned)[0][i] = (*a2)[0][i];
    }
    
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // reset residual
    matrix_RCP J_over = Teuchos::rcp(new LA_CrsMatrix(Copy, *param_overlapped_map, -1)); // reset Jacobian
    matrix_RCP J = Teuchos::rcp(new LA_CrsMatrix(Copy, *param_owned_map, -1)); // reset Jacobian
    
    res_over->PutScalar(0.0);
    J->PutScalar(0.0);
    J_over->PutScalar(0.0);
    
    this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, true, false, true, res_over, J_over);
    
    
    vector_RCP sens_over = Teuchos::rcp(new LA_MultiVector(*param_overlapped_map,1)); // reset residual
    vector_RCP sens = Teuchos::rcp(new LA_MultiVector(*param_owned_map,1)); // reset residual
    
    J->PutScalar(0.0);
    J->Export(*J_over, *param_exporter, Add);
    J->FillComplete(*LA_owned_map, *param_owned_map);
    
    J->Apply(*a_owned,*sens);
    
    vector<double> discLocalGradient(numDiscParams);
    vector<double> discGradient(numDiscParams);
    for (size_t i = 0; i < paramOwned.size(); i++) {
      int gid = paramOwned[i];
      discLocalGradient[gid] = (*sens)[0][i];
    }
    for (size_t i = 0; i < numDiscParams; i++) {
      double globalval = 0.0;
      double localval = discLocalGradient[i];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      double cobj = 0.0;
      if ((i+num_active_params)<obj_sens.size()) {
        cobj = obj_sens.fastAccessDx(i+num_active_params);
      }
      globalval += cobj;
      if (gradient.size()<=num_active_params+i) {
        gradient.push_back(globalval);
      }
      else {
        gradient[num_active_params+i] += globalval;
      }
    }
  }
}

// ========================================================================================
// The following function is the adjoint-based error estimate
// Not to be confused with the postprocess::computeError function which uses a true
//   solution to perform verification studies
// ========================================================================================

double solver::computeError(const vector_RCP & GF_soln, const vector_RCP & GA_soln) {
  if(Comm->getRank() == 0 && verbosity>0) {
    cout << endl << "*********************************************************" << endl;
    cout << "***** Computing Error Estimate ******" << endl << endl;
  }
  
  vector_RCP u = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // forward solution
  //LA_MultiVector A_soln(*LA_overlapped_map,1); // adjoint solution
  vector_RCP a = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // adjoint solution
  vector_RCP a2 = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,1)); // adjoint solution
  vector_RCP u_dot = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // previous solution (can be either fwd or adj)
  
  
  double deltat = 0.0;
  double alpha = 0.0;
  double beta = 1.0;
  if (isTransient) {
    deltat = finaltime / numsteps;
    alpha = 1./deltat;
  }
  
  double errorest = 0.0;
  this->sacadoizeParams(false);
  
  // ******************* ITERATE ON THE Parameters **********************
  
  current_time = 0.0;
  double localerror = 0.0;
  for (int timeiter = 0; timeiter<numsteps; timeiter++) {
    
    current_time += deltat;
    
    for( size_t i=0; i<LA_ownedAndShared.size(); i++ ) {
      (*u)[0][i] = (*GF_soln)[timeiter+1][i];
      (*u_dot)[0][i] = alpha*((*GF_soln)[timeiter+1][i] - (*GF_soln)[timeiter][i]);
    }
    for( size_t i=0; i<LA_owned.size(); i++ ) {
      (*a2)[0][i] = (*GA_soln)[numsteps-timeiter][i];
    }
    
    vector_RCP res = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,num_active_params)); // reset residual
    vector_RCP res_over = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,num_active_params)); // reset residual
    matrix_RCP J_over = Teuchos::rcp(new LA_CrsMatrix(Copy, *LA_overlapped_map, -1)); // reset Jacobian
    res_over->PutScalar(0.0);
    this->computeJacRes(u, u_dot, u, u_dot, alpha, beta, false, false, false, res_over, J_over);
    res->PutScalar(0.0);
    res->Export(*res_over, *exporter, Add);
    
    double currerror = 0.0;
    for( size_t i=0; i<LA_owned.size(); i++ ) {
      currerror += (*a2)[0][i] * (*res)[0][i];
    }
    localerror += currerror;
  }
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localerror,&errorest);
  //Comm->SumAll(&localerror, &errorest, 1);
  
  if(Comm->getRank() == 0 && verbosity>0) {
    cout << "Error estimate = " << errorest << endl;
  }
  return errorest;
}

// ========================================================================================
// ========================================================================================

void solver::updateJacDBC(matrix_RCP & J, size_t & e, size_t & block, int & fieldNum,
                  size_t & localSideId, const bool & compute_disc_sens) {
  
  // given a "block" and the unknown field update jacobian to enforce Dirichlet BCs
  
  string blockID = blocknames[block];
  vector<int> GIDs;// = cells[block][e]->GIDs;
  DOF->getElementGIDs(e, GIDs, blockID);
  
  const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fieldNum, spaceDim-1, localSideId);
  
  const vector<int> elmtOffset = SideIndex.first; // local nodes on that side
  const vector<int> basisIdMap = SideIndex.second;
  
  for( size_t i=0; i<elmtOffset.size(); i++ ) { // for each node
    int row = GIDs[elmtOffset[i]]; // global row
    if (compute_disc_sens) {
      vector<int> paramGIDs;// = cells[block][e]->paramGIDs;
      paramDOF->getElementGIDs(e, paramGIDs, blockID);
      for( size_t col=0; col<paramGIDs.size(); col++ ) {
        int ind = paramGIDs[col];
        double m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        //J.ReplaceGlobalValues(row, 1, &m_val, &ind);
        J->ReplaceGlobalValues(ind, 1, &m_val, &row);
      }
    }
    else {
      for( size_t col=0; col<GIDs.size(); col++ ) {
        int ind = GIDs[col];
        double m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        J->ReplaceGlobalValues(row, 1, &m_val, &ind);
      }
      double val = 1.0; // set diagonal entry to 1
      J->ReplaceGlobalValues(row, 1, &val, &row);
    }
  }
}

// ========================================================================================
// ========================================================================================

void solver::updateJacDBC(matrix_RCP & J, const vector<int> & dofs, const bool & compute_disc_sens) {
  
  // given a "block" and the unknown field update jacobian to enforce Dirichlet BCs
  
  for( int i=0; i<dofs.size(); i++ ) { // for each node
    if (compute_disc_sens) {
      for( int col=0; col<globalParamUnknowns; col++ ) {
        double m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        //J.ReplaceGlobalValues(row, 1, &m_val, &ind);
        J->ReplaceGlobalValues(col, 1, &m_val, &dofs[i]);
      }
    }
    else {
      for( int col=0; col<globalNumUnknowns; col++ ) {
        double m_val = 0.0; // set ALL of the entries to 0 in the Jacobian
        J->ReplaceGlobalValues(dofs[i], 1, &m_val, &col);
      }
      double val = 1.0; // set diagonal entry to 1
      J->ReplaceGlobalValues(dofs[i], 1, &val, &dofs[i]);
    }
  }
}

// ========================================================================================
// ========================================================================================

void solver::updateResDBC(vector_RCP & resid, size_t & e, size_t & block, int & fieldNum,
                  size_t & localSideId) {
  // given a "block" and the unknown field update resid to enforce Dirichlet BCs
  
  string blockID = blocknames[block];
  vector<int> elemGIDs;
  DOF->getElementGIDs(e, elemGIDs, blockID); // compute element global IDs
  
  int numRes = resid->NumVectors();
  const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fieldNum, spaceDim-1, localSideId);
  const vector<int> elmtOffset = SideIndex.first; // local nodes on that side
  const vector<int> basisIdMap = SideIndex.second;
  
  for( size_t i=0; i<elmtOffset.size(); i++ ) { // for each node
    int row = elemGIDs[elmtOffset[i]]; // global row
    double r_val = 0.0; // set residual to 0
    for( int j=0; j<numRes; j++ ) {
      resid->ReplaceGlobalValue(row, j, r_val); // replace the value
    }
  }
}

// ========================================================================================
// ========================================================================================

void solver::updateResDBC(vector_RCP & resid, const vector<int> & dofs) {
  // given a "block" and the unknown field update resid to enforce Dirichlet BCs
  
  int numRes = resid->NumVectors();
  
  for( size_t i=0; i<dofs.size(); i++ ) { // for each node
    double r_val = 0.0; // set residual to 0
    for( int j=0; j<numRes; j++ ) {
      resid->ReplaceGlobalValue(dofs[i], j, r_val); // replace the value
    }
  }
}


// ========================================================================================
// ========================================================================================

void solver::updateResDBCsens(vector_RCP & resid, size_t & e, size_t & block, int & fieldNum, size_t & localSideId,
                      const std::string & gside, const double & current_time) {
  
  //DOES NOT WORK FOR SENSITIVITIES...even in theory...and e is probably not used correctly either...though that only affects x,y,z...
  // given a "block" and the unknown field update resid derivatives with respect to parameters to account for parameter-dependent DBCs
  
  int fnum = DOF->getFieldNum(varlist[block][fieldNum]);
  string blockID = blocknames[block];
  vector<int> elemGIDs;// = cells[block][e]->GIDs[p];
  DOF->getElementGIDs(e, elemGIDs, blockID);
  
  int numRes = resid->NumVectors();
  const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fnum, spaceDim-1, localSideId);
  const vector<int> elmtOffset = SideIndex.first; // local nodes on that side
  const vector<int> basisIdMap = SideIndex.second;
  
  DRV I_elemNodes = this->getElemNodes(block,e);//cells[block][e]->nodes;
  
  for( size_t i=0; i<elmtOffset.size(); i++ ) { // for each node
    int row = elemGIDs[elmtOffset[i]]; // global row
    double x = I_elemNodes(0,basisIdMap[i],0);
    double y = 0.0;
    if (spaceDim > 1)
    y = I_elemNodes(0,basisIdMap[i],1);
    double z = 0.0;
    if (spaceDim > 2)
    z = I_elemNodes(0,basisIdMap[i],2);
    
    AD diri_FAD;
    diri_FAD = phys->getDirichletValue(block, x, y, z, current_time, varlist[block][fieldNum], gside, false, wkset[block]);
    double r_val = 0.0;
    size_t numDerivs = diri_FAD.size();
    for( int j=0; j<numRes; j++ ) {
      if (numDerivs > j)
      r_val = diri_FAD.fastAccessDx(j);
      
      resid->ReplaceGlobalValue(row, j, r_val); // replace the value
    }
  }
}

// ========================================================================================
// ========================================================================================

void solver::setDirichlet(vector_RCP & initial) {
  
  // TMW: this function needs to be fixed
  vector<vector<int> > fixedDOFs = phys->dbc_dofs;
  
  for (size_t b=0; b<cells.size(); b++) {
    string blockID = blocknames[b];
    Kokkos::View<int**,HostDevice> side_info;
    
    for (int n=0; n<numVars[b]; n++) {
      
      vector<size_t> localDirichletSideIDs = phys->localDirichletSideIDs[b][n];
      vector<size_t> boundDirichletElemIDs = phys->boundDirichletElemIDs[b][n];
      int fnum = DOF->getFieldNum(varlist[b][n]);
      for( size_t e=0; e<disc->myElements[b].size(); e++ ) { // loop through all the elements
        side_info = phys->getSideInfo(b,n,e);
        int numSides = side_info.dimension(0);
        DRV I_elemNodes = this->getElemNodes(b,e);//cells[b][e]->nodes;
        // enforce the boundary conditions if the element is on the given boundary
        
        for( int i=0; i<numSides; i++ ) {
          if( side_info(i,0)==1 ) {
            vector<int> elemGIDs;
            int gside_index = side_info(i,1);
            string gside = phys->sideSets[gside_index];
            size_t elemID = disc->myElements[b][e];
            DOF->getElementGIDs(elemID, elemGIDs, blockID); // global index of each node
            // get the side index and the node->global mapping for the side that is on the boundary
            const pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fnum, spaceDim-1, i);
            const vector<int> elmtOffset = SideIndex.first;
            const vector<int> basisIdMap = SideIndex.second;
            // for each node that is on the boundary side
            for( size_t j=0; j<elmtOffset.size(); j++ ) {
              // get the global row and coordinate
              int row =  LA_overlapped_map->LID(elemGIDs[elmtOffset[j]]);
              double x = I_elemNodes(0,basisIdMap[j],0);
              double y = 0.0;
              if (spaceDim > 1) {
                y = I_elemNodes(0,basisIdMap[j],1);
              }
              double z = 0.0;
              if (spaceDim > 2) {
                z = I_elemNodes(0,basisIdMap[j],2);
              }
              
              if (use_meas_as_dbcs) {
                (*initial)[0][row] = (*meas)[0][row];
              }
              else {
                // put the value into the soln vector
                AD diri_FAD_tmp;
                diri_FAD_tmp = phys->getDirichletValue(b, x, y, z, current_time, varlist[b][n], gside, useadjoint, wkset[b]);
                
                (*initial)[0][row] = diri_FAD_tmp.val();
              }
            }
          }
        }
      }
    }
    // set point dbcs
    vector<int> dbc_dofs = fixedDOFs[b];
    
    for (int i = 0; i < dbc_dofs.size(); i++) {
      int row = LA_overlapped_map->LID(dbc_dofs[i]);
      (*initial)[0][row] = 0.0; // fix to zero for now
    }
    
  }
  
}

// ========================================================================================
// ========================================================================================

vector_RCP solver::setInitialParams() {
  vector_RCP initial = Teuchos::rcp(new LA_MultiVector(*param_overlapped_map,1));
  double value = 2.0;
  initial->PutScalar(value);
  return initial;
}

// ========================================================================================
// ========================================================================================

vector_RCP solver::setInitial() {
  
  vector_RCP initial = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1));
  vector_RCP glinitial = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,1));
  initial->PutScalar(0.0);
  
  if (initial_type == "L2-projection") {
    
    // Compute the L2 projection of the initial data into the discrete space
    vector_RCP rhs = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,1)); // reset residual
    matrix_RCP mass = Teuchos::rcp(new LA_CrsMatrix(Copy, *LA_overlapped_map, -1)); // reset Jacobian
    vector_RCP glrhs = Teuchos::rcp(new LA_MultiVector(*LA_owned_map,1)); // reset residual
    matrix_RCP glmass = Teuchos::rcp(new LA_CrsMatrix(Copy, *LA_owned_map, -1)); // reset Jacobian
    
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        
        int numElem = cells[b][e]->numElem;
        vector<vector<int> > GIDs = cells[b][e]->GIDs;
        
        Kokkos::View<double**,AssemblyDevice> localrhs = cells[b][e]->getInitial(true, useadjoint);
        Kokkos::View<double***,AssemblyDevice> localmass = cells[b][e]->getMass();
        
        // assemble into global matrix
        for (int c=0; c<numElem; c++) {
          for( size_t row=0; row<GIDs[c].size(); row++ ) {
            int rowIndex = GIDs[c][row];
            double val = localrhs(c,row);
            rhs->SumIntoGlobalValue(rowIndex,0, val);
            for( size_t col=0; col<GIDs[c].size(); col++ ) {
              int colIndex = GIDs[c][col];
              double val = localmass(c,row,col);
              mass->InsertGlobalValues(rowIndex, 1, &val, &colIndex);
            }
          }
        }
      }
    }
    
    mass->FillComplete();
    glmass->PutScalar(0.0);
    glmass->Export(*mass, *exporter, Add);
    
    glrhs->PutScalar(0.0);
    glrhs->Export(*rhs, *exporter, Add);
    
    glmass->FillComplete();
    
    this->linearSolver(glmass, glrhs, glinitial);
    
    initial->Import(*glinitial, *importer, Add);
    
  }
  else if (initial_type == "interpolation") {
    
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        vector<vector<int> > GIDs = cells[b][e]->GIDs;
        Kokkos::View<double**,AssemblyDevice> localinit = cells[b][e]->getInitial(false, useadjoint);
        int numElem = cells[b][e]->numElem;
        for (int c=0; c<numElem; c++) {
          
          for( size_t row=0; row<GIDs[c].size(); row++ ) {
            int rowIndex = GIDs[c][row];
            double val = localinit(c,row);
            initial->ReplaceGlobalValue(rowIndex,0, val);
          }
        }
      }
    }
    
  }
  
  return initial;
}

// ========================================================================================
// ========================================================================================

void solver::computeJacRes(vector_RCP & u, vector_RCP & u_dot,
                   vector_RCP & phi, vector_RCP & phi_dot,
                   const double & alpha, const double & beta,
                   const bool & compute_jacobian, const bool & compute_sens,
                   const bool & compute_disc_sens,
                   vector_RCP & res, matrix_RCP & J) {
  
  int numRes = res->NumVectors();
  
  Teuchos::TimeMonitor localassemblytimer(*assemblytimer);
  
  for (size_t b=0; b<cells.size(); b++) {
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Set up the worksets and allocate the local residual and Jacobians
    //////////////////////////////////////////////////////////////////////////////////////
    
    wkset[b]->time = current_time;
    wkset[b]->time_KV(0) = current_time;
    wkset[b]->isTransient = isTransient;
    wkset[b]->isAdjoint = useadjoint;
    wkset[b]->alpha = alpha;
    if (isTransient)
    wkset[b]->deltat = 1.0/alpha;
    else
    wkset[b]->deltat = 1.0;
    
    int numElem = cells[b][0]->numElem;
    int numDOF = cells[b][0]->GIDs[0].size();
    
    int numParamDOF = 0;
    if (compute_disc_sens) {
      numParamDOF = cells[b][0]->paramGIDs[0].size();
    }
    
    Kokkos::View<double***,AssemblyDevice> local_res, local_J, local_Jdot;
    
    if (compute_sens) {
      local_res = Kokkos::View<double***,AssemblyDevice>("local residual",numElem,numDOF,num_active_params);
    }
    else {
      local_res = Kokkos::View<double***,AssemblyDevice>("local residual",numElem,numDOF,1);
    }
    
    if (compute_disc_sens) {
      local_J = Kokkos::View<double***,AssemblyDevice>("local Jacobian",numElem,numDOF,numParamDOF);
      local_Jdot = Kokkos::View<double***,AssemblyDevice>("local Jacobian dot",numElem,numDOF,numParamDOF);
    }
    else {
      local_J = Kokkos::View<double***,AssemblyDevice>("local Jacobian",numElem,numDOF,numDOF);
      local_Jdot = Kokkos::View<double***,AssemblyDevice>("local Jacobian dot",numElem,numDOF,numDOF);
    }
    
    //Kokkos::View<double**,AssemblyDevice> aPrev;
    
    /////////////////////////////////////////////////////////////////////////////
    // Loop over cells
    /////////////////////////////////////////////////////////////////////////////
    
    {
      Teuchos::TimeMonitor localtimer(*gathertimer);
    
      // Local gather of solutions (should be a better way to do this)
      this->performGather(b,u,0,0);
      this->performGather(b,u_dot,1,0);
      this->performGather(b,Psol[0],4,0);
      if (useadjoint) {
        this->performGather(b,phi,2,0);
        this->performGather(b,phi_dot,3,0);
      }
    }
    
    /////////////////////////////////////////////////////////////////////////////
    // Volume contribution
    /////////////////////////////////////////////////////////////////////////////
    
    for (size_t e=0; e < cells[b].size(); e++) {
      
      wkset[b]->localEID = e;
      cells[b][e]->updateData();
      
      
      if (isTransient && useadjoint && !cells[0][0]->multiscale) {
        //aPrev = cells[b][e]->adjPrev;
        //KokkosTools::print(aPrev);
        if (is_final_time) {
          for (int i=0; i<cells[b][e]->adjPrev.dimension(0); i++) {
            for (int j=0; j<cells[b][e]->adjPrev.dimension(1); j++) {
              cells[b][e]->adjPrev(i,j) = 0.0;
            }
          }
          //(cells[b][e]->adjPrev).initialize(0.0);
        }
      }
      
      
      /////////////////////////////////////////////////////////////////////////////
      // Compute the local residual and Jacobian on this cell
      /////////////////////////////////////////////////////////////////////////////
      
      {
        Teuchos::TimeMonitor localtimer(*phystimer);
        
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
        
        cells[b][e]->computeJacRes(paramvals, paramtypes, paramnames,
                                   current_time, isTransient, useadjoint, compute_jacobian, compute_sens,
                                   num_active_params, compute_disc_sens, false, store_adjPrev,
                                   local_res, local_J, local_Jdot);
        
      }
      
      //KokkosTools::print(local_J);
      //KokkosTools::print(local_res);
      /*
      if (isTransient && useadjoint && !cells[0][0]->multiscale) {
        if (gNLiter == 0)
        cells[b][e]->adjPrev = aPrev;
        else if (!store_adjPrev)
        cells[b][e]->adjPrev = aPrev;
      }
      */
      
      ///////////////////////////////////////////////////////////////////////////
      // Insert into global matrix/vector
      ///////////////////////////////////////////////////////////////////////////
      
      {
        Teuchos::TimeMonitor localtimer(*inserttimer);
        vector<vector<int> > GIDs = cells[b][e]->GIDs;
        
        vector<vector<int> > paramGIDs = cells[b][e]->paramGIDs;
        
        for (int i=0; i<GIDs.size(); i++) {
          vector<double> vals(GIDs[i].size());
          
          for( size_t row=0; row<GIDs[i].size(); row++ ) {
            int rowIndex = GIDs[i][row];
            for (int g=0; g<numRes; g++) {
              double val = local_res(i,row,g);
              res->SumIntoGlobalValue(rowIndex,g, val);
            }
            if (compute_jacobian) {
              if (compute_disc_sens) {
                for( size_t col=0; col<paramGIDs[i].size(); col++ ) {
                  int colIndex = paramGIDs[i][col];
                  double val = local_J(i,row,col) + alpha*local_Jdot(i,row,col);
                  J->InsertGlobalValues(colIndex, 1, &val, &rowIndex);
                }
              }
              else {
                for( size_t col=0; col<GIDs[i].size(); col++ ) {
                  vals[col] = local_J(i,row,col) + alpha*local_Jdot(i,row,col);
                }
                J->SumIntoGlobalValues(rowIndex, GIDs[i].size(), &vals[0], &GIDs[i][0]);
              }
            }
          }
        }
      }
      
    } // element loop
    
  } // block loop
  
  // ************************** STRONGLY ENFORCE DIRICHLET BCs *******************************************
  
  if (compute_jacobian) {
    if (compute_disc_sens) {
      J->FillComplete(*LA_owned_map, *param_owned_map);
    }
    else {
      J->FillComplete();
    }
  }
  
  if (usestrongDBCs) {
    Teuchos::TimeMonitor localtimer(*dbctimer);
    vector<vector<int> > fixedDOFs = phys->dbc_dofs;
    for (size_t b=0; b<cells.size(); b++) {
      vector<size_t> boundDirichletElemIDs;   // list of elements on the Dirichlet boundary
      vector<size_t> localDirichletSideIDs;   // local side numbers for Dirichlet boundary sides
      vector<size_t> globalDirichletSideIDs;   // local side numbers for Dirichlet boundary sides
      for (int n=0; n<numVars[b]; n++) {
        int fnum = DOF->getFieldNum(varlist[b][n]);
        boundDirichletElemIDs = phys->boundDirichletElemIDs[b][n];
        localDirichletSideIDs = phys->localDirichletSideIDs[b][n];
        globalDirichletSideIDs = phys->globalDirichletSideIDs[b][n];
        
        size_t numDBC = boundDirichletElemIDs.size();
        for (size_t e=0; e<numDBC; e++) {
          size_t eindex = boundDirichletElemIDs[e];
          size_t sindex = localDirichletSideIDs[e];
          size_t gside_index = globalDirichletSideIDs[e];
          
          if (compute_jacobian) {
            this->updateJacDBC(J, eindex, b, fnum, sindex, compute_disc_sens);
          }
          std::string gside = phys->sideSets[gside_index];
          this->updateResDBCsens(res, eindex, b, n, sindex, gside, current_time);
        }
      }
      
      this->updateJacDBC(J,fixedDOFs[b],compute_disc_sens);
      this->updateResDBC(res,fixedDOFs[b]);
    }
  }
  
  //updateResPin(res_over); //pinning attempt
  //if (compute_jacobian) {
  //    updateJacPin(J_over); //pinning attempt
  //}
  
}


// ========================================================================================
// ========================================================================================

void solver::linearSolver(matrix_RCP & J, vector_RCP & r, vector_RCP & soln)  {
  Teuchos::TimeMonitor localtimer(*linearsolvertimer);
  
  LA_LinearProblem LinSys(J.get(), soln.get(), r.get());
  
  // SOLVE ....
  if (useDirect) {
    Amesos AmFactory;
    char* SolverType = "Amesos_Klu";
    Amesos_BaseSolver * AmSolver = AmFactory.Create(SolverType, LinSys);
    AmSolver->SymbolicFactorization();
    AmSolver->NumericFactorization();
    AmSolver->Solve();
    delete AmSolver;
  }
  else {
    AztecOO linsolver(LinSys);
    
    // Set up the preconditioner
    ML_Epetra::MultiLevelPreconditioner* MLPrec;
    
    linsolver.SetAztecOption(AZ_solver,AZ_gmres);
    if(useDomDecomp){ //domain decomposition preconditioner, specific to Helmholtz at high frequencies
      linsolver.SetAztecOption(AZ_precond,AZ_dom_decomp);
      linsolver.SetAztecOption(AZ_subdomain_solve,AZ_ilut);
      linsolver.SetAztecParam(AZ_drop,dropTol);
      linsolver.SetAztecParam(AZ_ilut_fill,fillParam);
      
      if(verbosity == 0)
      linsolver.SetAztecOption(AZ_diagnostics,AZ_none);
      
      double condest = 0.0;
      linsolver.ConstructPreconditioner(condest);
      if(condest > 1.e13 || condest < 1.0){
        linsolver.DestroyPreconditioner();
        linsolver.SetAztecParam(AZ_athresh,1.e-5);
        linsolver.SetAztecParam(AZ_rthresh,0.0);
        linsolver.ConstructPreconditioner(condest);
        if(condest > 1.e13 || condest < 1.0){
          linsolver.DestroyPreconditioner();
          linsolver.SetAztecParam(AZ_athresh,1.e-5);
          linsolver.SetAztecParam(AZ_rthresh,0.01);
          linsolver.ConstructPreconditioner(condest);
          if(condest > 1.e13 || condest < 1.0){
            linsolver.DestroyPreconditioner();
            linsolver.SetAztecParam(AZ_athresh,1.e-2);
            linsolver.SetAztecParam(AZ_rthresh,0.0);
            linsolver.ConstructPreconditioner(condest);
            if(condest > 1.e13 || condest < 1.0){
              linsolver.DestroyPreconditioner();
              linsolver.SetAztecParam(AZ_athresh,1.e-2);
              linsolver.SetAztecParam(AZ_rthresh,0.01);
              linsolver.ConstructPreconditioner(condest);
              if(condest > 1.e13){
                cout << "SAD PRECONDITIONER: condition number " << condest << endl;
              }
            }
          }
        }
      }
    }
    
    else if (usePrec) { //multi-level preconditioner
      MLPrec = buildPreconditioner(J);
      linsolver.SetPrecOperator(MLPrec);
    }
    else {
      linsolver.SetAztecOption(AZ_precond, AZ_none);
    }
    linsolver.SetAztecOption(AZ_kspace,kspace);
    
    if (verbosity > 8)
    linsolver.SetAztecOption(AZ_output,10);
    else
    linsolver.SetAztecOption(AZ_output,0);
    
    linsolver.Iterate(liniter,lintol);
    
    if(!useDomDecomp && usePrec)
    delete MLPrec;
  }
  
  //return soln;
}


// ========================================================================================
// ========================================================================================

ML_Epetra::MultiLevelPreconditioner* solver::buildPreconditioner(const matrix_RCP & J) {
  Teuchos::ParameterList MLList;
  ML_Epetra::SetDefaults("SA",MLList);
  MLList.set("ML output", 0);
  MLList.set("max levels",5);
  MLList.set("increasing or decreasing","increasing");
  int numEqns;
  if (cells.size() == 1)
  numEqns = numVars[0];
  else
  numEqns = 1;
  
  MLList.set("PDE equations",numEqns);
  MLList.set("aggregation: type", "Uncoupled");
  MLList.set("smoother: type","IFPACK");
  MLList.set("smoother: sweeps",1);
  MLList.set("smoother: ifpack type","ILU");
  MLList.set("smoother: ifpack overlap",1);
  MLList.set("smoother: pre or post", "both");
  MLList.set("coarse: type","Amesos-KLU");
  ML_Epetra::MultiLevelPreconditioner* MLPrec =
  new ML_Epetra::MultiLevelPreconditioner(*J, MLList);
  
  return MLPrec;
}


// ========================================================================================
// ========================================================================================

void solver::sacadoizeParams(const bool & seed_active) {
  
  //vector<vector<AD> > paramvals_AD;
  if (seed_active) {
    size_t pprog = 0;
    for (size_t i=0; i<paramvals.size(); i++) {
      vector<AD> currparams;
      if (paramtypes[i] == 1) { // active parameters
        for (size_t j=0; j<paramvals[i].size(); j++) {
          //currparams.push_back(Sacado::Fad::DFad<double>(num_active_params,pprog,paramvals[i][j]));
          paramvals_KVAD(i,j) = AD(maxDerivs,pprog,paramvals[i][j]);
          currparams.push_back(AD(maxDerivs,pprog,paramvals[i][j]));
          pprog++;
        }
      }
      else { // inactive, stochastic, or discrete parameters
        for (size_t j=0; j<paramvals[i].size(); j++) {
          //currparams.push_back(Sacado::Fad::DFad<double>(paramvals[i][j]));
          paramvals_KVAD(i,j) = AD(paramvals[i][j]);
          currparams.push_back(AD(paramvals[i][j]));
        }
      }
      *(paramvals_AD[i]) = currparams;
    }
  }
  else {
    size_t pprog = 0;
    for (size_t i=0; i<paramvals.size(); i++) {
      vector<AD> currparams;
      for (size_t j=0; j<paramvals[i].size(); j++) {
        //currparams.push_back(Sacado::Fad::DFad<double>(paramvals[i][j]));
        currparams.push_back(AD(paramvals[i][j]));
        paramvals_KVAD(i,j) = AD(paramvals[i][j]);
      }
      *(paramvals_AD[i]) = currparams;
    }
  }
  
  phys->updateParameters(paramvals_AD, paramnames);
  multiscale_manager->updateParameters(paramvals_AD, paramnames);
  
}

// ========================================================================================
// ========================================================================================

void solver::updateParams(const vector<double> & newparams, const int & type) {
  size_t pprog = 0;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        if (Comm->getRank() == 0 && verbosity > 0) {
          cout << "Updated Params: " << paramvals[i][j] << " (old value)   " << newparams[pprog] << " (new value)" << endl;
        }
        paramvals[i][j] = newparams[pprog];
        pprog++;
      }
    }
  }
  if ((type == 4) && (globalParamUnknowns > 0)) {
    int numClassicParams = this->getNumParams(1); // offset for ROL param vector
    for (size_t i = 0; i < paramOwnedAndShared.size(); i++) {
      int gid = paramOwnedAndShared[i];
      Psol[0]->ReplaceGlobalValue(gid,0,newparams[gid+numClassicParams]);
    }
  }
  if ((type == 2) && (globalParamUnknowns > 0)) {
    int numClassicParams = this->getNumParams(2); // offset for ROL param vector
    for (size_t i=0; i<paramOwnedAndShared.size(); i++) {
      int gid = paramOwnedAndShared[i];
      Psol[0]->ReplaceGlobalValue(gid,0,newparams[i+numClassicParams]);
    }
  }
}

// ========================================================================================
// ========================================================================================

void solver::updateParams(const vector<double> & newparams, const std::string & stype) {
  size_t pprog = 0;
  int type;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  if (stype == "inactive") { type = 0;}
  else if (stype == "active") { type = 1;}
  else if (stype == "stochastic") { type = 2;}
  else if (stype == "discrete") { type = 3;}
  else {
    //complain
  }
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        paramvals[i][j] = newparams[pprog];
        pprog++;
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

void solver::updateMeshData(const int & newrandseed) {
  
  // Determine how many seeds there are
  int localnumSeeds = 0;
  int numSeeds = 0;
  for (int b=0; b<cells.size(); b++) {
    for (int e=0; e<cells[b].size(); e++) {
      for (int k=0; k<cells[b][e]->numElem; k++) {
        if (cells[b][e]->cell_data_seed[k] > localnumSeeds) {
          localnumSeeds = cells[b][e]->cell_data_seed[k];
        }
      }
    }
  }
  //Comm->MaxAll(&localnumSeeds, &numSeeds, 1);
  Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&localnumSeeds,&numSeeds);
  numSeeds += 1; //To properly allocate and iterate
  
  // Create a random number generator
  std::default_random_engine generator(newrandseed);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set seed data
  ////////////////////////////////////////////////////////////////////////////////
  
  int numdata = 9;
  
  //cout << "solver numSeeds = " << numSeeds << endl;
  
  std::normal_distribution<double> ndistribution(0.0,1.0);
  Kokkos::View<double**,HostDevice> rotation_data("cell_data",numSeeds,numdata);
  for (int k=0; k<numSeeds; k++) {
    double x = ndistribution(generator);
    double y = ndistribution(generator);
    double z = ndistribution(generator);
    double w = ndistribution(generator);
    
    double r = sqrt(x*x + y*y + z*z + w*w);
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
  
  ////////////////////////////////////////////////////////////////////////////////
  // Update subgrid elements
  ////////////////////////////////////////////////////////////////////////////////
  
  //multiscale_manager->updateMeshData(rotation_data);
  
}

// ========================================================================================
// ========================================================================================

vector<double> solver::getParams(const int & type) {
  vector<double> reqparams;
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        reqparams.push_back(paramvals[i][j]);
      }
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

vector<string> solver::getParamsNames(const int & type) {
  vector<string> reqparams;
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      reqparams.push_back(paramnames[i]);
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

vector<size_t> solver::getParamsLengths(const int & type) {
  vector<size_t> reqparams;
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      reqparams.push_back(paramvals[i].size());
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

vector<double> solver::getParams(const std::string & stype) {
  vector<double> reqparams;
  int type;
  if (stype == "inactive")
  type = 0;
  else if (stype == "active")
  type = 1;
  else if (stype == "stochastic")
  type = 2;
  else if (stype == "discrete")
  type = 3;
  else
  //complain
  
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        reqparams.push_back(paramvals[i][j]);
      }
    }
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

vector<vector<double> > solver::getParamBounds(const std::string & stype) {
  vector<vector<double> > reqbnds;
  vector<double> reqlo;
  vector<double> requp;
  int type;
  if (stype == "inactive") {type = 0;}
  else if (stype == "active") {type = 1;}
  else if (stype == "stochastic") {type = 2;}
  else if (stype == "discrete") {type = 3;}
  else if (stype == "discretized") {type = 4;}
  
  if (type == 0) {
    std::cout << "Bounds for inactive parameters are currently at default of (0,0)" << std::endl;
  }
  
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        reqlo.push_back(paramLowerBounds[i][j]);
        requp.push_back(paramUpperBounds[i][j]);
      }
    }
  }
  
  if (type == 4 && globalParamUnknowns > 0) {
    int numDiscParams = this->getNumParams(4);
    vector<double> rLocalLo(numDiscParams);
    vector<double> rLocalUp(numDiscParams);
    vector<double> rlo(numDiscParams);
    vector<double> rup(numDiscParams);
    int pindex;
    for (int n = 0; n < num_discretized_params; n++) {
      for (size_t i = 0; i < paramNodes[n].size(); i++) {
        int pnode = paramNodes[n][i];
        if (pnode >= 0) {
          int pindex = paramOwned[pnode];
          rLocalLo[pindex] = lowerParamBounds[n];
          rLocalUp[pindex] = upperParamBounds[n];
        }
      }
    }
    
    for (size_t i = 0; i < numDiscParams; i++) {
      
      double globalval = 0.0;
      double localval = rLocalLo[i];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      rlo[i] = globalval;
      
      globalval = 0.0;
      localval = rLocalUp[i];
      Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
      //Comm->SumAll(&localval, &globalval, 1);
      rup[i] = globalval;
    }
    
    reqlo = rlo;
    requp = rup;
    
  }
  
  reqbnds.push_back(reqlo);
  reqbnds.push_back(requp);
  return reqbnds;
}

// ========================================================================================
// ========================================================================================

void solver::setBatchID(const int & bID){
  batchID = bID;
}

// ========================================================================================
// ========================================================================================

void solver::stashParams(){
  if (batchID == 0 && Comm->getRank() == 0){
    string outname = "param_stash.dat";
    ofstream respOUT(outname);
    respOUT.precision(16);
    for (size_t i=0; i<paramvals.size(); i++) {
      if (paramtypes[i] == 1) {
        for (size_t j=0; j<paramvals[i].size(); j++) {
          respOUT << paramvals[i][j] << endl;
        }
      }
    }
    respOUT.close();
  }
}

// ========================================================================================
// ========================================================================================

vector<double> solver::getStochasticParams(const std::string & whichparam) {
  if (whichparam == "mean")
    return stochastic_mean;
  else if (whichparam == "variance")
    return stochastic_variance;
  else if (whichparam == "min")
    return stochastic_min;
  else if (whichparam == "max")
    return stochastic_max;
  else {
    vector<double> emptyvec;
    return emptyvec;
  }
}

// ========================================================================================
// ========================================================================================

vector<double> solver::getFractionalParams(const std::string & whichparam) {
  if (whichparam == "s-exponent")
    return s_exp;
  else if (whichparam == "mesh-resolution")
    return h_mesh;
  else {
    vector<double> emptyvec;
    return emptyvec;
  }
}


// ========================================================================================
// ========================================================================================

vector_RCP solver::blankState(){
  vector_RCP F_soln = Teuchos::rcp(new LA_MultiVector(*LA_overlapped_map,numsteps+1)); // empty solution
  return F_soln;
}

// ========================================================================================
//
// ========================================================================================

void solver::performGather(const size_t & block, const vector_RCP & vec, const int & type,
                   const size_t & index) {
  
  for (size_t e=0; e < cells[block].size(); e++) {
    cells[block][e]->setLocalSoln(vec, type, index);
  }
  
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

DRV solver::getElemNodes(const int & block, const int & elemID) {
  int nnodes = elemnodes[block].dimension(1);
  
  DRV cnodes("element nodes",1,nnodes,spaceDim);
  for (int i=0; i<nnodes; i++) {
    for (int j=0; j<spaceDim; j++) {
      cnodes(0,i,j) = elemnodes[block](elemID,i,j);
    }
  }
  return cnodes;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void solver::finalizeMultiscale() {
  if (multiscale_manager->subgridModels.size() > 0 ) {
    for (size_t k=0; k<multiscale_manager->subgridModels.size(); k++) {
      multiscale_manager->subgridModels[k]->paramvals_KVAD = paramvals_KVAD;
    //  multiscale_manager->subgridModels[k]->wkset[0]->paramnames = paramnames;
    }
    
    multiscale_manager->setMacroInfo(disc->basis_pointers, disc->basis_types,
                                     phys->varlist, useBasis, phys->offsets,
                                     paramnames, discretized_param_names);
    
    multiscale_manager->macro_wkset = wkset;
    double my_cost = multiscale_manager->initialize();
    double gmin = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
    //Comm->MinAll(&my_cost, &gmin, 1);
    double gmax = 0.0;
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);
    //Comm->MaxAll(&my_cost, &gmax, 1);
    
    if(Comm->getRank() == 0 && verbosity>0) {
      cout << "***** Load Balancing Factor " << gmax/gmin <<  endl;
    }
    
    
  }

}
