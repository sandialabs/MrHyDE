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

#include "workset.hpp"
#include "parameterManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_STKConnManager.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class Node>
ParameterManager<Node>::ParameterManager(const Teuchos::RCP<MpiComm> & Comm_,
                                   Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                   Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                                   Teuchos::RCP<physics> & phys_,
                                   Teuchos::RCP<discretization> & disc_) :
Comm(Comm_), mesh(mesh_), disc(disc_), phys(phys_), settings(settings_) {
  
  milo_debug_level = settings->get<int>("debug level",0);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting ParameterManager constructor ... " << endl;
    }
  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  /////////////////////////////////////////////////////////////////////////////
  
  mesh->getElementBlockNames(blocknames);
  spaceDim = mesh->getDimension();
  verbosity = settings->get<int>("verbosity",0);
  milo_debug_level = settings->get<int>("debug level",0);
  
  num_inactive_params = 0;
  num_active_params = 0;
  num_stochastic_params = 0;
  num_discrete_params = 0;
  num_discretized_params = 0;
  globalParamUnknowns = 0;
  have_dRdP = false;
  discretized_stochastic = false;
  
  use_custom_initial_param_guess = settings->sublist("Physics").get<bool>("use custom initial param guess",false);
  
  this->setupParameters();
  //this->setupDiscretizedParameters(cells,boundaryCells);
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished ParameterManager constructor" << endl;
    }
  }
  
}

// ========================================================================================
// Set up the parameters (inactive, active, stochastic, discrete)
// Communicate these parameters back to the physics interface and the enabled modules
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setupParameters() {
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting ParameterManager::setupParameters ... " << endl;
    }
  }
  
  Teuchos::ParameterList parameters;
  
  if (settings->isSublist("Parameters")) {
    parameters = settings->sublist("Parameters");
    Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
    while (pl_itr != parameters.end()) {
      Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
      vector<ScalarT> newparamvals;
      if (!newparam.isParameter("type") || !newparam.isParameter("usage")) {
        // print out error message
      }
      
      if (newparam.get<string>("type") == "scalar") {
        newparamvals.push_back(newparam.get<ScalarT>("value"));
      }
      else if (newparam.get<string>("type") == "vector") {
        std::string filename = newparam.get<string>("source");
        std::ifstream fin(filename.c_str());
        std::istream_iterator<ScalarT> start(fin), end;
        vector<ScalarT> importedparamvals(start, end);
        for (size_t i=0; i<importedparamvals.size(); i++) {
          newparamvals.push_back(importedparamvals[i]);
        }
      }
      
      paramnames.push_back(pl_itr->first);
      paramvals.push_back(newparamvals);
      
      Teuchos::RCP<vector<AD> > newparam_AD = Teuchos::rcp(new vector<AD>(newparamvals.size()));
      paramvals_AD.push_back(newparam_AD);
      
      //blank bounds
      vector<ScalarT> lo(newparamvals.size(),0.0);
      vector<ScalarT> up(newparamvals.size(),0.0);
      
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
          stochastic_mean.push_back(newparam.get<ScalarT>("mean",0.0));
          stochastic_variance.push_back(newparam.get<ScalarT>("variance",1.0));
          stochastic_min.push_back(newparam.get<ScalarT>("min",-1.0));
          stochastic_max.push_back(newparam.get<ScalarT>("max",1.0));
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
        initialParamValues.push_back(newparam.get<ScalarT>("initial_value",1.0));
        lowerParamBounds.push_back(newparam.get<ScalarT>("lower_bound",-1.0));
        upperParamBounds.push_back(newparam.get<ScalarT>("upper_bound",1.0));
        discparam_distribution.push_back(newparam.get<string>("distribution","uniform"));
        discparamVariance.push_back(newparam.get<ScalarT>("variance",1.0));
        if (newparam.get<bool>("isDomainParam",true)) {
          domainRegTypes.push_back(newparam.get<int>("reg_type",0));
          domainRegConstants.push_back(newparam.get<ScalarT>("reg_constant",0.0));
          domainRegIndices.push_back(num_discretized_params - 1);
        }
        else {
          boundaryRegTypes.push_back(newparam.get<int>("reg_type",0));
          boundaryRegConstants.push_back(newparam.get<ScalarT>("reg_constant",0.0));
          boundaryRegSides.push_back(newparam.get<string>("sides"," "));
          boundaryRegIndices.push_back(num_discretized_params - 1);
        }
      }
      
      paramLowerBounds.push_back(lo);
      paramUpperBounds.push_back(up);
      
      pl_itr++;
    }
    
    TEUCHOS_TEST_FOR_EXCEPTION(num_active_params > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of parameters.");
    
    size_t maxcomp = 0;
    for (size_t k=0; k<paramvals.size(); k++) {
      if (paramvals[k].size() > maxcomp) {
        maxcomp = paramvals[k].size();
      }
    }

    Kokkos::View<ScalarT**,AssemblyDevice> test("parameter values (AD)", paramvals.size(), maxcomp);
 
    paramvals_KVAD = Kokkos::View<AD**,AssemblyDevice>("parameter values (AD)", paramvals.size(), maxcomp);

  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished ParameterManager::setupParameters" << endl;
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
// Set up the discretized parameter DOF manager
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void ParameterManager<Node>::setupDiscretizedParameters(vector<vector<Teuchos::RCP<cell> > > & cells,
                                                  vector<vector<Teuchos::RCP<BoundaryCell> > > & boundaryCells) {
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting ParameterManager::setupDiscretizedParameters ... " << endl;
    }
  }
  
  if (num_discretized_params > 0) {
    // determine the unique list of basis'
    
    vector<int> disc_orders = phys->unique_orders[0];
    vector<string> disc_types = phys->unique_types[0];
    vector<int> disc_usebasis;
    
    for (size_t j=0; j<discretized_param_basis_orders.size(); j++) {
      for (size_t k=0; k<disc_orders.size(); k++) {
        if (discretized_param_basis_orders[j] == disc_orders[k] && discretized_param_basis_types[j] == disc_types[k]) {
          disc_usebasis.push_back(k);
        }
      }
    }
    
    discretized_param_basis_types = disc_types;
    discretized_param_basis_orders = disc_orders;
    discretized_param_usebasis = disc_usebasis;
    
    discretized_param_basis = disc->basis_pointers[0];
    
    paramDOF = Teuchos::rcp(new panzer::DOFManager());
    Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
    paramDOF->setConnManager(conn,*(Comm->getRawMpiComm()));
    
    Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern;
    
    for (size_t b=0; b<blocknames.size(); b++) {
      for (size_t j=0; j<discretized_param_names.size(); j++) {
        
        Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(discretized_param_basis[disc_usebasis[j]]));
        paramDOF->addField(blocknames[b], discretized_param_names[j], Pattern);
      }
    }
    
    paramDOF->buildGlobalUnknowns();
    
    for (size_t b=0; b<blocknames.size(); b++) {
      int numGIDs = paramDOF->getElementBlockGIDCount(blocknames[b]);
      TEUCHOS_TEST_FOR_EXCEPTION(numGIDs > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of discretized parameter degrees of freedom per element on block: " + blocknames[b]);
    }
    
    paramDOF->getOwnedIndices(paramOwned);
    numParamUnknowns = (int)paramOwned.size();
    paramDOF->getOwnedAndGhostedIndices(paramOwnedAndShared);
    numParamUnknownsOS = (int)paramOwnedAndShared.size();
    int localParamUnknowns = numParamUnknowns;
    
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localParamUnknowns,&globalParamUnknowns);
    //Comm->SumAll(&localParamUnknowns, &globalParamUnknowns, 1);
    
    for (size_t j=0; j<discretized_param_names.size(); j++) {
      int num = paramDOF->getFieldNum(discretized_param_names[j]);
      vector<int> poffsets = paramDOF->getGIDFieldOffsets(blocknames[0],num); // same for all blocks?
      paramoffsets.push_back(poffsets);
      paramNumBasis.push_back(discretized_param_basis[discretized_param_usebasis[j]]->getCardinality());
    }
    
    Kokkos::View<const LO**,Kokkos::LayoutRight,PHX::Device> LIDs = paramDOF->getLIDs();
        
    for (size_t b=0; b<cells.size(); b++) {
      if (cells[b].size() > 0) {
        int numLocalDOF = 0;
        Kokkos::View<LO*,AssemblyDevice> numDOF_KV("number of param DOF per variable",num_discretized_params);
        for (int k=0; k<num_discretized_params; k++) {
          numDOF_KV(k) = paramNumBasis[k];
          numLocalDOF += paramNumBasis[k];
        }
        cells[b][0]->cellData->numParamDOF = numDOF_KV;
        Kokkos::View<LO*,HostDevice> numDOF_host("numDOF on host",num_discretized_params);
        Kokkos::deep_copy(numDOF_host, numDOF_KV);
        cells[b][0]->cellData->numParamDOF_host = numDOF_host;
        
        vector<size_t> myElem = disc->myElements[b];
        Kokkos::View<size_t*,AssemblyDevice> GEIDs("element IDs on device",myElem.size());
        auto host_GEIDs = Kokkos::create_mirror_view(GEIDs);
        for (size_t elem=0; elem<myElem.size(); elem++) {
          host_GEIDs(elem) = myElem[elem];
        }
        Kokkos::deep_copy(GEIDs, host_GEIDs);
        
        for (size_t e=0; e<cells[b].size(); e++) {
          LIDView cellLIDs("cell parameter LIDs",cells[b][e]->numElem, LIDs.extent(1));
          Kokkos::View<LO*,AssemblyDevice> EIDs = cells[b][e]->localElemID;
          parallel_for("paramman copy LIDs",RangePolicy<AssemblyExec>(0,cellLIDs.extent(0)), KOKKOS_LAMBDA (const int c ) {
            size_t elemID = GEIDs(EIDs(c));
            for (size_type j=0; j<LIDs.extent(1); j++) {
              cellLIDs(c,j) = LIDs(elemID,j);
            }
          });
          cells[b][e]->setParams(cellLIDs);
          cells[b][e]->setParamUseBasis(disc_usebasis, paramNumBasis);
        }
      }
    }
    for (size_t b=0; b<boundaryCells.size(); b++) {
      if (boundaryCells[b].size() > 0) {
        int numLocalDOF = 0;
        for (int k=0; k<num_discretized_params; k++) {
          numLocalDOF += paramNumBasis[k];
        }
        
        for (size_t e=0; e<boundaryCells[b].size(); e++) {
          LIDView cellLIDs("bcell parameter LIDs",boundaryCells[b][e]->numElem, LIDs.extent(1));
          Kokkos::View<LO*,AssemblyDevice> EIDs = boundaryCells[b][e]->localElemID;
          parallel_for("paramman copy LIDs bcells",RangePolicy<AssemblyExec>(0,cellLIDs.extent(0)), KOKKOS_LAMBDA (const int e ) {
            size_t elemID = EIDs(e);
            for (size_type j=0; j<LIDs.extent(1); j++) {
              cellLIDs(e,j) = LIDs(elemID,j);
            }
          });
          boundaryCells[b][e]->setParams(cellLIDs);
          boundaryCells[b][e]->setParamUseBasis(disc_usebasis, paramNumBasis);
        }
      }
    }
    if (discretized_stochastic) { // add the param DOFs as indep. rv's
      for (int j=0; j<numParamUnknownsOS; j++) {
        // hard coding for one disc param just to get something working
        stochastic_distribution.push_back(discparam_distribution[0]);
        stochastic_mean.push_back(initialParamValues[0]);
        stochastic_variance.push_back(discparamVariance[0]);
        stochastic_min.push_back(lowerParamBounds[0]);
        stochastic_max.push_back(upperParamBounds[0]);
        
      }
    }
  }
  
  // Set up the discretized parameter linear algebra objects
  
  if (num_discretized_params > 0) {
    
    GO localNumUnknowns = paramOwned.size();
    
    GO globalNumUnknowns = 0;
    Teuchos::reduceAll<LO,GO>(*Comm,Teuchos::REDUCE_SUM,1,&localNumUnknowns,&globalNumUnknowns);
    
    param_owned_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, paramOwned, 0, Comm));
    param_overlapped_map = Teuchos::rcp(new LA_Map(globalNumUnknowns, paramOwnedAndShared, 0, Comm));
    
    param_exporter = Teuchos::rcp(new LA_Export(param_overlapped_map, param_owned_map));
    param_importer = Teuchos::rcp(new LA_Import(param_overlapped_map, param_owned_map));
    
    //vector< vector<int> > param_nodesOS(numParamUnknownsOS); // should be overlapped
    //vector< vector<int> > param_nodes(numParamUnknowns); // not overlapped -- for bounds
    vector< vector< vector<ScalarT> > > param_initial_vals; // custom initial guess set by assembler->cells
    
    vector_RCP paramVec = this->setInitialParams(); // TMW: this will be deprecated soon
    
    vector<vector<GO> > param_dofs;//(num_discretized_params);
    vector<vector<GO> > param_dofs_OS;//(num_discretized_params);
    for (int num=0; num<num_discretized_params; num++) {
      vector<GO> dofs, dofs_OS;
      param_dofs.push_back(dofs);
      param_dofs_OS.push_back(dofs_OS);
    }
    
    /*
    Kokkos::View<const LO**,AssemblyDevice> LIDs = paramDOF->getLIDs();
    
    auto host_LIDs = Kokkos::create_mirror_view(LIDs);
    for (size_t b=0; b<blocknames.size(); b++) {
      for (int num=0; num<num_discretized_params; num++) {
        vector<int> var_offsets = paramDOF->getGIDFieldOffsets(blocknames[b],num);
        for (size_t e=0; e<host_LIDs.extent(0); e++) {
          for (size_t dof=0; dof<var_offsets.size(); dof++) {
            param_dofs_OS[num].push_back(host_LIDs(e,var_offsets[dof]));
          }
        }
      }
    }
    */
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<size_t> EIDs = disc->myElements[b];
      for (size_t e=0; e<EIDs.size(); e++) {
        vector<GO> gids;
        size_t elemID = EIDs[e];
        paramDOF->getElementGIDs(elemID, gids, blocknames[b]);
        
        for (int num=0; num<num_discretized_params; num++) {
          vector<int> var_offsets = paramDOF->getGIDFieldOffsets(blocknames[b],num);
          for (size_t dof=0; dof<var_offsets.size(); dof++) {
            param_dofs_OS[num].push_back(gids[var_offsets[dof]]);
          }
        }
      }
    }
    
    //vector<GO> paramOwned_tmp = paramOwned;
    for (int n=0; n<num_discretized_params; n++) {
      //std::sort(param_dofs_OS[n].begin(), param_dofs_OS[n].end());
      //param_dofs_OS[n].erase( std::unique(param_dofs_OS[n].begin(),
      //                                    param_dofs_OS[n].end()), param_dofs_OS[n].end());
      //sort(param_dofs_OS[n].begin(),param_dofs_OS[n].end());
      //sort(paramOwned_tmp.begin(),paramOwned_tmp.end());
      
      //set_intersection(param_dofs_OS[n].begin(),param_dofs_OS[n].end(),
      //                 paramOwned.begin(),paramOwned.end(),
      //                 back_inserter(param_dofs[n]));
      
    }
    
    for (int n = 0; n < num_discretized_params; n++) {
      if (!use_custom_initial_param_guess) {
        for (size_t i = 0; i < param_dofs_OS[n].size(); i++) {
          paramVec->replaceGlobalValue(param_dofs_OS[n][i],0,initialParamValues[n]);
        }
      }
      paramNodesOS.push_back(param_dofs_OS[n]); // store for later use
      paramNodes.push_back(param_dofs[n]); // store for later use
    }
    
    /*
    for (int n=0; n<num_discretized_params; n++) {
      std::sort(param_nodesOS[n].begin(), param_nodesOS[n].end());
      param_nodesOS[n].erase( std::unique(param_nodesOS[n].begin(),
                                          param_nodesOS[n].end()), param_nodesOS[n].end());
      
      std::sort(param_nodes[n].begin(), param_nodes[n].end());
      param_nodes[n].erase( std::unique(param_nodes[n].begin(),
                                        param_nodes[n].end()), param_nodes[n].end());
    }*/
    /*for (int n = 0; n < num_discretized_params; n++) {
      if (!use_custom_initial_param_guess) {
        for (size_t i = 0; i < param_nodesOS[n].size(); i++) {
          paramVec->replaceGlobalValue(paramOwnedAndShared[param_nodesOS[n][i]]
                                       ,0,initialParamValues[n]);
        }
      }
      paramNodesOS.push_back(param_nodesOS[n]); // store for later use
      paramNodes.push_back(param_nodes[n]); // store for later use
    }*/
    //KokkosTools::print(paramVec);
    Psol.push_back(paramVec);
  }
  else {
    // set up a dummy parameter vector
    paramOwnedAndShared.push_back(0);
    const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();
    
    param_overlapped_map = Teuchos::rcp(new LA_Map(INVALID, paramOwnedAndShared, 0, Comm));
    
    vector_RCP paramVec = this->setInitialParams(); // TMW: this will be deprecated soon
    Psol.push_back(paramVec);
  }
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished ParameterManager::setupDiscretizedParameters" << endl;
    }
  }
  
}
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
int ParameterManager<Node>::getNumParams(const int & type) {
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

template<class Node>
int ParameterManager<Node>::getNumParams(const std::string & type) {
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

template<class Node>
vector<ScalarT> ParameterManager<Node>::getDiscretizedParamsVector() {
  int numParams = this->getNumParams(4);
  vector<ScalarT> discLocalParams(numParams);
  vector<ScalarT> discParams(numParams);
  //auto Psol_2d = Psol[0]->getLocalView<Kokkos::Device<Node::execution_space,Node::memory_space>>();
  auto Psol_2d = Psol[0]->template getLocalView<LA_device>();
  auto Psol_host = Kokkos::create_mirror_view(Psol_2d);
  for (size_t i = 0; i < paramOwned.size(); i++) {
    int gid = paramOwned[i];
    discLocalParams[gid] = Psol_host(i,0);
    //cout << gid << " " << Psol_2d(i,0) << endl;
  }
  for (int i = 0; i < numParams; i++) {
    ScalarT globalval = 0.0;
    ScalarT localval = discLocalParams[i];
    Teuchos::reduceAll(*Comm,Teuchos::REDUCE_SUM,1,&localval,&globalval);
    //Comm->SumAll(&localval, &globalval, 1);
    discParams[i] = globalval;
    //cout << i << " " << globalval << "  " << localval << endl;
    
  }
  
  return discParams;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > ParameterManager<Node>::setInitialParams() {
//vector_RCP ParameterManager<Node>::setInitialParams() {
  
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting ParameterManager::setInitialParams ..." << endl;
    }
  }
  
  vector_RCP initial = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
  initial->putScalar(2.0);
  
  /*
  if (scalarInitialData) {
    // This will be done on the host for now
    auto initial_kv = initial->getLocalView<HostDevice>();
    for (size_t block=0; block<assembler->cells.size(); block++) {
      Kokkos::View<int**,AssemblyDevice> offsets = assembler->wkset[block]->offsets;
      auto host_offsets = Kokkos::create_mirror_view(offsets);
      Kokkos::deep_copy(host_offsets,offsets);
      for (size_t cell=0; cell<assembler->cells[block].size(); cell++) {
        Kokkos::View<LO**,HostDevice> LIDs = assembler->cells[block][cell]->LIDs_host;
        Kokkos::View<LO*,HostDevice> numDOF = assembler->cells[block][cell]->cellData->numDOF_host;
        //parallel_for("solver initial scalar",RangePolicy<HostExec>(0,LIDs.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int e=0; e<LIDs.extent(0); e++) {
          for (size_t n=0; n<numDOF.extent(0); n++) {
            for (size_t i=0; i<numDOF(n); i++ ) {
              initial_kv(LIDs(e,host_offsets(n,i)),0) = scalarInitialValues[block][n];
            }
          }
        }
      }
    }
  }
  else {
  
    initial->putScalar(0.0);
    
    vector_RCP glinitial = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    
    if (initial_type == "L2-projection") {
      
      // Compute the L2 projection of the initial data into the discrete space
      vector_RCP rhs = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // reset residual
      matrix_RCP mass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_overlapped_graph));//Tpetra::createCrsMatrix<ScalarT>(LA_overlapped_map); // reset Jacobian
      vector_RCP glrhs = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1)); // reset residual
      matrix_RCP glmass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_owned_map, maxEntries));//Tpetra::createCrsMatrix<ScalarT>(LA_owned_map); // reset Jacobian
      assembler->setInitial(rhs, mass, useadjoint);
      
      glmass->setAllToScalar(0.0);
      glmass->doExport(*mass, *exporter, Tpetra::ADD);
      
      glrhs->putScalar(0.0);
      glrhs->doExport(*rhs, *exporter, Tpetra::ADD);
      
      glmass->fillComplete();
      
      this->linearSolver(glmass, glrhs, glinitial);
      have_preconditioner = false; // resetting this because mass matrix may not have connectivity as Jacobians
      initial->doImport(*glinitial, *importer, Tpetra::ADD);
      
    }
    else if (initial_type == "interpolation") {
      
      assembler->setInitial(initial, useadjoint);
      
    }
  }
  */
  if (milo_debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished ParameterManager::setInitialParams ..." << endl;
    }
  }
  
  return initial;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::sacadoizeParams(const bool & seed_active) {
  
  size_t maxlength = paramvals_KVAD.extent(1);
  
  Kokkos::View<int*,AssemblyDevice> ptypes("parameter types",paramtypes.size());
  auto host_ptypes = Kokkos::create_mirror_view(ptypes);
  for (size_t i=0; i<paramtypes.size(); i++) {
    host_ptypes(i) = paramtypes[i];
  }
  Kokkos::deep_copy(ptypes, host_ptypes);
  
  Kokkos::View<size_t*,AssemblyDevice> plengths("parameter lengths",paramvals.size());
  auto host_plengths = Kokkos::create_mirror_view(plengths);
  for (size_t i=0; i<paramvals.size(); i++) {
    host_plengths(i) = paramvals[i].size();
  }
  Kokkos::deep_copy(plengths, host_plengths);
  
  size_t prog = 0;
  Kokkos::View<size_t**,AssemblyDevice> pseed("parameter seed index",paramvals.size(),maxlength);
  auto host_pseed = Kokkos::create_mirror_view(pseed);
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == 1) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        host_pseed(i,j) = prog;
        prog++;
      }
    }
  }
  Kokkos::deep_copy(pseed,host_pseed);
  
  Kokkos::View<ScalarT**,AssemblyDevice> pvals("parameter values",paramvals.size(), maxlength);
  auto host_pvals = Kokkos::create_mirror_view(pvals);
  for (size_t i=0; i<paramvals.size(); i++) {
    for (size_t j=0; j<paramvals[i].size(); j++) {
      host_pvals(i,j) = paramvals[i][j];
    }
  }
  Kokkos::deep_copy(pvals, host_pvals);
  
  if (seed_active) {
    size_t pprog = 0;
    for (size_t i=0; i<paramvals.size(); i++) {
      vector<AD> currparams;
      if (paramtypes[i] == 1) { // active parameters
        for (size_t j=0; j<paramvals[i].size(); j++) {
          currparams.push_back(AD(maxDerivs,pprog,paramvals[i][j]));
          pprog++;
        }
      }
      else { // inactive, stochastic, or discrete parameters
        for (size_t j=0; j<paramvals[i].size(); j++) {
          //host_params(i,j) = AD(paramvals[i][j]);
          currparams.push_back(AD(paramvals[i][j]));
        }
      }
      *(paramvals_AD[i]) = currparams;
    }
    parallel_for("parameter manager sacadoize - seed active",RangePolicy<AssemblyExec>(0,pvals.extent(0)), KOKKOS_LAMBDA (const size_type i ) {
      if (ptypes(i) == 1) { // active params
        for (size_t j=0; j<plengths(i); j++) {
          paramvals_KVAD(i,j) = AD(maxDerivs, pseed(i,j), pvals(i,j));
        }
      }
      else {
        for (size_t j=0; j<plengths(i); j++) {
          paramvals_KVAD(i,j) = AD(pvals(i,j));
        }
      }
    });
  }
  else {
    for (size_t i=0; i<paramvals.size(); i++) {
      vector<AD> currparams;
      for (size_t j=0; j<paramvals[i].size(); j++) {
        currparams.push_back(AD(paramvals[i][j]));
        //host_params(i,j) = AD(paramvals[i][j]);
      }
      *(paramvals_AD[i]) = currparams;
    }
    parallel_for("parameter manager sacadoize - no seeding",RangePolicy<AssemblyExec>(0,pvals.extent(0)), KOKKOS_LAMBDA (const size_type i ) {
      for (size_t j=0; j<plengths(i); j++) {
        paramvals_KVAD(i,j) = AD(pvals(i,j));
      }
    });
  }
  AssemblyExec::execution_space().fence();
  phys->updateParameters(paramvals_AD, paramnames);
   
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateParams(const vector<ScalarT> & newparams, const int & type) {
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
      Psol[0]->replaceGlobalValue(gid,0,newparams[gid+numClassicParams]);
      //Psol[0]->replaceGlobalValue(gid,0,newparams[i+numClassicParams]);
    }
  }
  if ((type == 2) && (globalParamUnknowns > 0)) {
    int numClassicParams = this->getNumParams(2); // offset for ROL param vector
    for (size_t i=0; i<paramOwnedAndShared.size(); i++) {
      int gid = paramOwnedAndShared[i];
      Psol[0]->replaceGlobalValue(gid,0,newparams[i+numClassicParams]);
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setParam(const vector<ScalarT> & newparams, const std::string & name) {
  size_t pprog = 0;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramnames[i] == name) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        if (Comm->getRank() == 0 && verbosity > 0) {
          cout << "Updated Params: " << paramvals[i][j] << " (old value)   " << newparams[pprog] << " (new value)" << endl;
        }
        paramvals[i][j] = newparams[pprog];
        pprog++;
      }
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateParams(const vector<ScalarT> & newparams, const std::string & stype) {
  size_t pprog = 0;
  int type = -1;
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

template<class Node>
vector<ScalarT> ParameterManager<Node>::getParams(const int & type) {
  vector<ScalarT> reqparams;
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

template<class Node>
vector<string> ParameterManager<Node>::getParamsNames(const int & type) {
  vector<string> reqparams;
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      reqparams.push_back(paramnames[i]);
    }
  }
  return reqparams;
}

template<class Node>
bool ParameterManager<Node>::isParameter(const string & name) {
  bool isparam = false;
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramnames[i] == name) {
      isparam = true;
    }
  }
  return isparam;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<size_t> ParameterManager<Node>::getParamsLengths(const int & type) {
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

template<class Node>
vector<ScalarT> ParameterManager<Node>::getParams(const std::string & stype) {
  vector<ScalarT> reqparams;
  int type = -1;
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

template<class Node>
vector<vector<ScalarT> > ParameterManager<Node>::getParamBounds(const std::string & stype) {
  vector<vector<ScalarT> > reqbnds;
  vector<ScalarT> reqlo;
  vector<ScalarT> requp;
  int type = -1;
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
    vector<ScalarT> rLocalLo(numDiscParams);
    vector<ScalarT> rLocalUp(numDiscParams);
    vector<ScalarT> rlo(numDiscParams);
    vector<ScalarT> rup(numDiscParams);
    for (int n = 0; n < num_discretized_params; n++) {
      for (size_t i = 0; i < paramNodesOS[n].size(); i++) {
        int pnode = paramNodesOS[n][i];
        if (pnode >= 0) {
          int pindex = paramOwned[pnode];
          rLocalLo[pindex] = lowerParamBounds[n];
          rLocalUp[pindex] = upperParamBounds[n];
        }
      }
    }
    
    for (int i = 0; i < numDiscParams; i++) {
      
      ScalarT globalval = 0.0;
      ScalarT localval = rLocalLo[i];
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

template<class Node>
void ParameterManager<Node>::stashParams(){
  if (batchID == 0 && Comm->getRank() == 0){
    string outname = "param_stash.dat";
    std::ofstream respOUT(outname);
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

template<class Node>
vector<ScalarT> ParameterManager<Node>::getStochasticParams(const std::string & whichparam) {
  if (whichparam == "mean")
    return stochastic_mean;
  else if (whichparam == "variance")
    return stochastic_variance;
  else if (whichparam == "min")
    return stochastic_min;
  else if (whichparam == "max")
    return stochastic_max;
  else {
    vector<ScalarT> emptyvec;
    return emptyvec;
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<ScalarT> ParameterManager<Node>::getFractionalParams(const std::string & whichparam) {
  if (whichparam == "s-exponent")
    return s_exp;
  else if (whichparam == "mesh-resolution")
    return h_mesh;
  else {
    vector<ScalarT> emptyvec;
    return emptyvec;
  }
}



