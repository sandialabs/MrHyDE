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
                                   Teuchos::RCP<PhysicsInterface> & phys_,
                                   Teuchos::RCP<DiscretizationInterface> & disc_) :
Comm(Comm_), disc(disc_), phys(phys_), settings(settings_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParameterManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  mesh = mesh_;
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0) {
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
  debug_level = settings->get<int>("debug level",0);
  
  num_inactive_params = 0;
  num_active_params = 0;
  num_stochastic_params = 0;
  num_discrete_params = 0;
  num_discretized_params = 0;
  globalParamUnknowns = 0;
  discretized_stochastic = false;
  
  use_custom_initial_param_guess = settings->sublist("Physics").get<bool>("use custom initial param guess",false);
  
  if (settings->sublist("Solver").isParameter("number of steps")) {
    numTimeSteps = settings->sublist("Solver").get<int>("number of steps",1);
  }
  else {
    double deltat = settings->sublist("Solver").get<double>("delta t",1.0);
    double final_time = settings->sublist("Solver").get<double>("final time",1.0);
    numTimeSteps = (int)final_time/deltat;
  }
  
  this->setupParameters();
  
  have_dynamic = false;
  for (size_t dp=0; dp<discretized_param_dynamic.size(); ++dp) {
    if (discretized_param_dynamic[dp]) {
      have_dynamic = true;
    }
  }

  if (debug_level > 0) {
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
  
  if (debug_level > 0) {
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
        if (newparam.isParameter("number of components")) {

        }
        else {
          std::string filename = newparam.get<string>("source");
          std::ifstream fin(filename.c_str());
          std::istream_iterator<ScalarT> start(fin), end;
          vector<ScalarT> importedparamvals(start, end);
          for (size_t i=0; i<importedparamvals.size(); i++) {
            newparamvals.push_back(importedparamvals[i]);
          }
        }
      }
      
      paramnames.push_back(pl_itr->first);
      paramvals.push_back(newparamvals);
      
      Teuchos::RCP<vector<AD> > newparam_AD = Teuchos::rcp(new vector<AD>(newparamvals.size()));
      paramvals_AD.push_back(newparam_AD);
      
      Teuchos::RCP<vector<ScalarT> > newparam_Sc = Teuchos::rcp(new vector<ScalarT>(newparamvals.size()));
      paramvals_Sc.push_back(newparam_Sc);
      
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
        discretized_param_dynamic.push_back(newparam.get<bool>("dynamic",false));

        initialParamValues.push_back(newparam.get<ScalarT>("initial_value",1.0));
        lowerParamBounds.push_back(newparam.get<ScalarT>("lower_bound",-1.0));
        upperParamBounds.push_back(newparam.get<ScalarT>("upper_bound",1.0));
        discparam_distribution.push_back(newparam.get<string>("distribution","uniform"));
        discparamVariance.push_back(newparam.get<ScalarT>("variance",1.0));
        /*
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
        }*/
      }
      
      paramLowerBounds.push_back(lo);
      paramUpperBounds.push_back(up);
      
      pl_itr++;
    }

    for (size_t block=0; block<blocknames.size(); ++block) {
      if (num_active_params>disc->num_derivs_required[block]) {
        disc->num_derivs_required[block] = num_active_params;
      } 
    }
    
#ifndef MrHyDE_NO_AD
    TEUCHOS_TEST_FOR_EXCEPTION(num_active_params > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of parameters.");
#endif
    size_t maxcomp = 0;
    for (size_t k=0; k<paramvals.size(); k++) {
      if (paramvals[k].size() > maxcomp) {
        maxcomp = paramvals[k].size();
      }
    }

    Kokkos::View<ScalarT**,AssemblyDevice> test("parameter values (AD)", paramvals.size(), maxcomp);
 
    paramvals_KVAD = Kokkos::View<AD**,AssemblyDevice>("parameter values (AD)", paramvals.size(), maxcomp);
    paramvals_KV = Kokkos::View<ScalarT**,AssemblyDevice>("parameter values (AD)", paramvals.size(), maxcomp);
  }
  
  if (debug_level > 0) {
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
void ParameterManager<Node>::setupDiscretizedParameters(vector<vector<Teuchos::RCP<Group> > > & groups,
                                                        vector<vector<Teuchos::RCP<BoundaryGroup> > > & boundary_groups) {
  
  if (debug_level > 0) {
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
    
    for (size_t block=0; block<blocknames.size(); ++block) {
      auto cellTopo = mesh->getCellTopology(blocknames[block]);
      for (size_t j=0; j<discretized_param_names.size(); j++) {
        basis_RCP cbasis = disc->getBasis(spaceDim, cellTopo,
                                          disc_types[disc_usebasis[j]],
                                          disc_orders[disc_usebasis[j]]);
        
        Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(cbasis));
        paramDOF->addField(blocknames[block], discretized_param_names[j], Pattern);
      }
    }
    
    paramDOF->buildGlobalUnknowns();
#ifndef MrHyDE_NO_AD
    for (size_t block=0; block<blocknames.size(); ++block) {
      int numGIDs = paramDOF->getElementBlockGIDCount(blocknames[block]);
      if (numGIDs > disc->num_derivs_required[block]) {
        disc->num_derivs_required[block] = numGIDs;
      } 
    
      TEUCHOS_TEST_FOR_EXCEPTION(numGIDs > maxDerivs,std::runtime_error,
                                 "Error: maxDerivs is not large enough to support the number of discretized parameter degrees of freedom per element on block: " + blocknames[block]);
    }
#endif
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
        
    for (size_t block=0; block<groups.size(); ++block) {
      if (groups[block].size() > 0) {
        int numLocalDOF = 0;
        Kokkos::View<LO*,AssemblyDevice> numDOF_KV("number of param DOF per variable",num_discretized_params);
        for (size_t k=0; k<num_discretized_params; k++) {
          numDOF_KV(k) = paramNumBasis[k];
          numLocalDOF += paramNumBasis[k];
        }
        groups[block][0]->group_data->num_param_dof = numDOF_KV;
        Kokkos::View<LO*,HostDevice> numDOF_host("numDOF on host",num_discretized_params);
        Kokkos::deep_copy(numDOF_host, numDOF_KV);
        groups[block][0]->group_data->num_param_dof_host = numDOF_host;
        
        vector<size_t> myElem = disc->my_elements[block];
        Kokkos::View<size_t*,AssemblyDevice> GEIDs("element IDs on device",myElem.size());
        auto host_GEIDs = Kokkos::create_mirror_view(GEIDs);
        for (size_t elem=0; elem<myElem.size(); elem++) {
          host_GEIDs(elem) = myElem[elem];
        }
        Kokkos::deep_copy(GEIDs, host_GEIDs);
        
        for (size_t grp=0; grp<groups[block].size(); ++grp) {
          LIDView groupLIDs("parameter LIDs",groups[block][grp]->numElem, LIDs.extent(1));
          Kokkos::View<LO*,AssemblyDevice> EIDs = groups[block][grp]->localElemID;
          parallel_for("paramman copy LIDs",
                       RangePolicy<AssemblyExec>(0,groupLIDs.extent(0)), 
                       KOKKOS_LAMBDA (const int c ) {
            size_t elemID = GEIDs(EIDs(c));
            for (size_type j=0; j<LIDs.extent(1); j++) {
              groupLIDs(c,j) = LIDs(elemID,j);
            }
          });
          groups[block][grp]->setParams(groupLIDs);
          groups[block][grp]->setParamUseBasis(disc_usebasis, paramNumBasis);
        }
      }
    }
    for (size_t block=0; block<boundary_groups.size(); ++block) {
      if (boundary_groups[block].size() > 0) {
        int numLocalDOF = 0;
        for (size_t k=0; k<num_discretized_params; k++) {
          numLocalDOF += paramNumBasis[k];
        }
        
        for (size_t grp=0; grp<boundary_groups[block].size(); ++grp) {
          LIDView groupLIDs("parameter LIDs",boundary_groups[block][grp]->numElem, LIDs.extent(1));
          Kokkos::View<LO*,AssemblyDevice> EIDs = boundary_groups[block][grp]->localElemID;
          parallel_for("paramman copy LIDs bgroups",
                       RangePolicy<AssemblyExec>(0,groupLIDs.extent(0)), 
                       KOKKOS_LAMBDA (const int e ) {
            size_t elemID = EIDs(e);
            for (size_type j=0; j<LIDs.extent(1); j++) {
              groupLIDs(e,j) = LIDs(elemID,j);
            }
          });
          boundary_groups[block][grp]->setParams(groupLIDs);
          boundary_groups[block][grp]->setParamUseBasis(disc_usebasis, paramNumBasis);
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
    param_importer = Teuchos::rcp(new LA_Import(param_owned_map, param_overlapped_map));
    
    //vector< vector<int> > param_nodesOS(numParamUnknownsOS); // should be overlapped
    //vector< vector<int> > param_nodes(numParamUnknowns); // not overlapped -- for bounds
    vector< vector< vector<ScalarT> > > param_initial_vals; // custom initial guess set by assembler->groups
    
    this->setInitialParams(); 
    
    vector<vector<GO> > param_dofs;//(num_discretized_params);
    vector<vector<GO> > param_dofs_OS;//(num_discretized_params);
    for (size_t num=0; num<num_discretized_params; num++) {
      vector<GO> dofs, dofs_OS;
      param_dofs.push_back(dofs);
      param_dofs_OS.push_back(dofs_OS);
    }
    
    for (size_t block=0; block<blocknames.size(); ++block) {
      vector<size_t> EIDs = disc->my_elements[block];
      for (size_t e=0; e<EIDs.size(); e++) {
        vector<GO> gids;
        size_t elemID = EIDs[e];
        paramDOF->getElementGIDs(elemID, gids, blocknames[block]);
        
        for (size_t num=0; num<num_discretized_params; num++) {
          vector<int> var_offsets = paramDOF->getGIDFieldOffsets(blocknames[block],num);
          for (size_t dof=0; dof<var_offsets.size(); dof++) {
            param_dofs_OS[num].push_back(gids[var_offsets[dof]]);
          }
        }
      }
    }
    
    for (size_t n = 0; n < num_discretized_params; n++) {
      if (!use_custom_initial_param_guess) {
        for (size_t i = 0; i < param_dofs_OS[n].size(); i++) {
          if (have_dynamic) {
            for (size_t j=0; j<dynamic_Psol_over.size(); ++j) {
              dynamic_Psol_over[j]->replaceGlobalValue(param_dofs_OS[n][i],0,initialParamValues[n]);
            }
          }
          Psol_over->replaceGlobalValue(param_dofs_OS[n][i],0,initialParamValues[n]);
        }
      }
      paramNodesOS.push_back(param_dofs_OS[n]); // store for later use
      paramNodes.push_back(param_dofs[n]); // store for later use
    }
    Psol->doExport(*Psol_over, *param_exporter, Tpetra::REPLACE);
    if (have_dynamic) {
      for (size_t i=0; i<dynamic_Psol_over.size(); ++i) {
        dynamic_Psol[i]->doExport(*(dynamic_Psol_over[i]), *param_exporter, Tpetra::REPLACE);
      }
    }
  }
  else {
    // set up a dummy parameter vector
    paramOwnedAndShared.push_back(0);
    paramOwned.push_back(0);
    const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();
    
    param_overlapped_map = Teuchos::rcp(new LA_Map(INVALID, paramOwnedAndShared, 0, Comm));
    param_owned_map = Teuchos::rcp(new LA_Map(INVALID, paramOwned, 0, Comm));

    this->setInitialParams(); 
  }
  
  if (debug_level > 0) {
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
  auto Psol_2d = Psol->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
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
// return the discretized parameters as vector of vector_RCPs for use with ROL
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > ParameterManager<Node>::getDiscretizedParams() {
  return Psol;
}

template<class Node>
MrHyDE_OptVector ParameterManager<Node>::getCurrentVector() {
  
  Teuchos::RCP<vector<ScalarT> > new_active_params;
  vector<vector_RCP> new_disc_params;
  if (num_active_params > 0) {
    new_active_params = this->getParams(1);
  }
  else {
    new_active_params = Teuchos::null;
  }
  if (globalParamUnknowns > 0) {
    if (have_dynamic) {
      new_disc_params = dynamic_Psol;
    }
    else {
      new_disc_params.push_back(Psol);
    }
  }
    
  MrHyDE_OptVector newvec(new_disc_params, new_active_params, Comm->getRank());
  return newvec;
}

// ========================================================================================
// return the discretized parameters as vector of vector_RCPs for use with ROL
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > ParameterManager<Node>::getDynamicDiscretizedParams() {
  return dynamic_Psol;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setInitialParams() {
  
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Starting ParameterManager::setInitialParams ..." << endl;
    }
  }
  
  
  Psol = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
  Psol_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
  Psol->putScalar(0.0); 
  Psol_over->putScalar(0.0); // TMW: why is this hard-coded??? 
  if (have_dynamic) {
    for (int i=0; i<numTimeSteps; ++i) {
      vector_RCP dyninit = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
      vector_RCP dyninit_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
      dyninit->putScalar(0.0); // TMW: why is this hard-coded??? 
      dyninit_over->putScalar(0.0); // TMW: why is this hard-coded??? 
      dynamic_Psol.push_back(dyninit);
      dynamic_Psol_over.push_back(dyninit_over);
    }
  }
  /*
  if (scalarInitialData) {
    // This will be done on the host for now
    auto initial_kv = initial->getLocalView<HostDevice>();
    for (size_t block=0; block<assembler->groups.size(); block++) {
      Kokkos::View<int**,AssemblyDevice> offsets = assembler->wkset[block]->offsets;
      auto host_offsets = Kokkos::create_mirror_view(offsets);
      Kokkos::deep_copy(host_offsets,offsets);
      for (size_t group=0; group<assembler->groups[block].size(); group++) {
        Kokkos::View<LO**,HostDevice> LIDs = assembler->groups[block][group]->LIDs_host;
        Kokkos::View<LO*,HostDevice> numDOF = assembler->groups[block][group]->group_data->numDOF_host;
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
      have_preconditioner = false; // resetting this because mass matrix may not have same connectivity as Jacobians
      initial->doImport(*glinitial, *importer, Tpetra::ADD);
      
    }
    else if (initial_type == "interpolation") {
      
      assembler->setInitial(initial, useadjoint);
      
    }
  }
  */
  if (debug_level > 0) {
    if (Comm->getRank() == 0) {
      cout << "**** Finished ParameterManager::setInitialParams ..." << endl;
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::sacadoizeParams(const bool & seed_active) {
  
  
  if (paramvals.size()>0) {
    
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
        vector<ScalarT> currparams_Sc;
        if (paramtypes[i] == 1) { // active parameters
          for (size_t j=0; j<paramvals[i].size(); j++) {
#ifndef MrHyDE_NO_AD
            currparams.push_back(AD(maxDerivs,pprog,paramvals[i][j]));
#else
            currparams.push_back(paramvals[i][j]);
#endif
            currparams_Sc.push_back(paramvals[i][j]);
            pprog++;
          }
        }
        else { // inactive, stochastic, or discrete parameters
          for (size_t j=0; j<paramvals[i].size(); j++) {
            //host_params(i,j) = AD(paramvals[i][j]);
            currparams.push_back(AD(paramvals[i][j]));
            currparams_Sc.push_back(paramvals[i][j]);
          }
        }
        *(paramvals_AD[i]) = currparams;
        *(paramvals_Sc[i]) = currparams_Sc;
      }
      parallel_for("parameter manager sacadoize - seed active",
                   RangePolicy<AssemblyExec>(0,pvals.extent(0)),
                   KOKKOS_LAMBDA (const size_type i ) {
        if (ptypes(i) == 1) { // active params
          for (size_t j=0; j<plengths(i); j++) {
#ifndef MrHyDE_NO_AD
            paramvals_KVAD(i,j) = AD(maxDerivs, pseed(i,j), pvals(i,j));
#else
            paramvals_KVAD(i,j) = pvals(i,j);
#endif
            paramvals_KV(i,j) = pvals(i,j);
          }
        }
        else {
          for (size_t j=0; j<plengths(i); j++) {
            paramvals_KVAD(i,j) = AD(pvals(i,j));
            paramvals_KV(i,j) = pvals(i,j);
          }
        }
      });
    }
    else {
      for (size_t i=0; i<paramvals.size(); i++) {
        vector<AD> currparams;
        vector<ScalarT> currparams_Sc;
        for (size_t j=0; j<paramvals[i].size(); j++) {
          currparams.push_back(AD(paramvals[i][j]));
          currparams_Sc.push_back(paramvals[i][j]);
          //host_params(i,j) = AD(paramvals[i][j]);
        }
        *(paramvals_AD[i]) = currparams;
        *(paramvals_Sc[i]) = currparams_Sc;
      }
      parallel_for("parameter manager sacadoize - no seeding",
                   RangePolicy<AssemblyExec>(0,pvals.extent(0)),
                   KOKKOS_LAMBDA (const size_type i ) {
        for (size_t j=0; j<plengths(i); j++) {
          paramvals_KVAD(i,j) = AD(pvals(i,j));
          paramvals_KV(i,j) = pvals(i,j);
        }
      });
    }
    AssemblyExec::execution_space().fence();
    phys->updateParameters(paramvals_AD, paramnames);
    phys->updateParameters(paramvals_Sc, paramnames);
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateParams(MrHyDE_OptVector & newparams) {
  
  if (newparams.haveScalar()) {
    auto scalar_params = newparams.getParameter();

    size_t pprog = 0;
    // perhaps add a check that the size of newparams equals the number of parameters of the
    // requested type
  
    for (size_t i=0; i<paramvals.size(); i++) {
      if (paramtypes[i] == 1) {
        for (size_t j=0; j<paramvals[i].size(); j++) {
          if (Comm->getRank() == 0 && verbosity > 0) {
            cout << "Updated Params: " << paramvals[i][j] << " (old value)   " << (*scalar_params)[pprog] << " (new value)" << endl;
          }
          paramvals[i][j] = (*scalar_params)[pprog];
          pprog++;
        }
      }
    }
  }

  if (newparams.haveField()) {
    auto disc_params = newparams.getField();
    
    if (have_dynamic) {
      for (size_t i=0; i<disc_params.size(); ++i) {
        auto owned_vec = disc_params[i]->getVector();
        dynamic_Psol[i]->assign(*owned_vec);
        dynamic_Psol_over[i]->putScalar(0.0);
        dynamic_Psol_over[i]->doImport(*owned_vec, *param_importer, Tpetra::ADD);
      }
    }
    else {
      auto owned_vec = disc_params[0]->getVector();
      Psol->assign(*owned_vec);
      Psol_over->putScalar(0.0);
      Psol_over->doImport(*owned_vec, *param_importer, Tpetra::ADD);
      //Psol->assign(*(disc_params[0]->getVector()));
    }
  }

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
      Psol->replaceGlobalValue(gid,0,newparams[gid+numClassicParams]);
    }
  }
  if ((type == 2) && (globalParamUnknowns > 0)) {
    int numClassicParams = this->getNumParams(2); // offset for ROL param vector
    for (size_t i=0; i<paramOwnedAndShared.size(); i++) {
      int gid = paramOwnedAndShared[i];
      Psol->replaceGlobalValue(gid,0,newparams[i+numClassicParams]);
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateDynamicParams(const int & timestep) {
  
  if ((int)dynamic_Psol.size() > timestep) {
    Psol_over->assign(*dynamic_Psol_over[timestep]);
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
Teuchos::RCP<vector<ScalarT> > ParameterManager<Node>::getParams(const int & type) {
  Teuchos::RCP<vector<ScalarT> > reqparams = Teuchos::rcp(new std::vector<ScalarT>());
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        reqparams->push_back(paramvals[i][j]);
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
vector<Teuchos::RCP<vector<ScalarT> > > ParameterManager<Node>::getActiveParamBounds() {
  vector<Teuchos::RCP<vector<ScalarT> > > reqbnds;
  Teuchos::RCP<vector<ScalarT> > reqlo = Teuchos::rcp( new vector<ScalarT> (num_active_params, 0.0) );
  Teuchos::RCP<vector<ScalarT> > requp = Teuchos::rcp( new vector<ScalarT> (num_active_params, 0.0) );

  size_t prog = 0;
  for (size_t i=0; i<paramvals.size(); i++) {
    if (paramtypes[i] == 1) {
      for (size_t j=0; j<paramvals[i].size(); j++) {
        (*reqlo)[prog] = paramLowerBounds[i][j];
        (*requp)[prog] = paramUpperBounds[i][j];
        prog++;
      }
    }
  }
  reqbnds.push_back(reqlo);
  reqbnds.push_back(requp);

  return reqbnds;

}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > ParameterManager<Node>::getDiscretizedParamBounds() {

  vector<vector_RCP> reqbnds;

  vector_RCP lower, upper, lower_over, upper_over;

  if (globalParamUnknowns > 0) {

    lower = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
    lower_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
    lower->putScalar(0.0); 
    lower_over->putScalar(0.0);
    
    upper = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
    upper_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
    upper->putScalar(0.0); 
    upper_over->putScalar(0.0); // TMW: why is this hard-coded??? 
  
    int numDiscParams = this->getNumParams(4);
    vector<ScalarT> rLocalLo(numDiscParams);
    vector<ScalarT> rLocalUp(numDiscParams);
    vector<ScalarT> rlo(numDiscParams);
    vector<ScalarT> rup(numDiscParams);
    for (size_t n = 0; n < num_discretized_params; n++) {
      for (size_t i = 0; i < paramNodesOS[n].size(); i++) {
        //int pnode = paramNodesOS[n][i];
        lower_over->replaceGlobalValue(paramNodesOS[n][i],0,lowerParamBounds[n]);
        upper_over->replaceGlobalValue(paramNodesOS[n][i],0,upperParamBounds[n]);
        //if (pnode >= 0) {
        //  int pindex = paramOwned[pnode];
        //  rLocalLo[pindex] = lowerParamBounds[n];
        //  rLocalUp[pindex] = upperParamBounds[n];
        //}
      }
    }
    
    lower->doExport(*lower_over, *param_exporter, Tpetra::REPLACE);
    upper->doExport(*upper_over, *param_exporter, Tpetra::REPLACE);

    /*
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
    */
  }
  
  reqbnds.push_back(lower);
  reqbnds.push_back(upper);
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

template<class Node>
void ParameterManager<Node>::setWorkset(vector<Teuchos::RCP<Workset<AD> > > & wkset_) {
  for (size_t block = 0; block<wkset_.size(); block++) {
    wkset.push_back(wkset_[block]);
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we can get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void ParameterManager<Node>::purgeMemory() {
  // nothing here  
}


// Explicit template instantiations
template class MrHyDE::ParameterManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
  template class MrHyDE::ParameterManager<SubgridSolverNode>;
#endif
