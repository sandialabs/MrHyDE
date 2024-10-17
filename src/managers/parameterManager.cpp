/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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
                                   Teuchos::RCP<MeshInterface> & mesh_,
                                   Teuchos::RCP<PhysicsInterface> & phys_,
                                   Teuchos::RCP<DiscretizationInterface> & disc_) :
Comm(Comm_), disc(disc_), phys(phys_), settings(settings_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParameterManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  mesh = mesh_;
  
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), Comm));
  
  debugger->print("**** Starting ParameterManager constructor ... ");
  
  /////////////////////////////////////////////////////////////////////////////
  // Parameters
  /////////////////////////////////////////////////////////////////////////////
  
  blocknames = mesh->getBlockNames();
  spaceDim = mesh->getDimension();
  verbosity = settings->get<int>("verbosity",0);
  debug_level = settings->get<int>("debug level",0);
  
  num_inactive_params = 0;
  num_active_params = 0;
  num_stochastic_params = 0;
  num_discrete_params = 0;
  num_discretized_params = 0;
  globalParamUnknowns = 0;
  numParamUnknowns = 0;
  numParamUnknownsOS = 0;
  discretized_stochastic = false;
  have_dynamic_scalar = false;
  have_dynamic_discretized = false;
  
  use_custom_initial_param_guess = settings->sublist("Physics").get<bool>("use custom initial param guess",false);
  
  if (settings->sublist("Solver").isParameter("delta t")) {
    dynamic_dt = settings->sublist("Solver").get<double>("delta t");
  }
  else {
    dynamic_dt = 1.0;
  }
  dynamic_timeindex = 0; // starting point
  
  // Need number of time steps
  if (settings->sublist("Solver").isParameter("number of steps")) {
    numTimeSteps = settings->sublist("Solver").get<int>("number of steps",1);
  }
  else {
    double initial_time = settings->sublist("Solver").get<double>("initial time",0.0);
    double final_time = settings->sublist("Solver").get<double>("final time",1.0);
    double deltat = settings->sublist("Solver").get<double>("delta t",1.0);
    
    numTimeSteps = 0;
    double ctime = initial_time;
    while (ctime < final_time) {
      numTimeSteps++;
      ctime += deltat;
    }
    
  }
  
  this->setupParameters();
  
  debugger->print("**** Finished ParameterManager constructor");
  
}

// ========================================================================================
// Set up the parameters (inactive, active, stochastic, discrete)
// Communicate these parameters back to the physics interface and the enabled modules
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setupParameters() {
  
  debugger->print("**** Starting ParameterManager::setupParameters ... ");
  
  Teuchos::ParameterList parameters;
  vector<vector<ScalarT> > tmp_paramvals;
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
      tmp_paramvals.push_back(newparamvals);
      
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
        bool is_dynamic = newparam.get<bool>("dynamic",false);
        scalar_param_dynamic.push_back(is_dynamic);
        if (is_dynamic) {
          have_dynamic_scalar = true;
        }
        
        //if active, look for actual bounds
        if (newparam.isParameter("bounds")) {
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
        
        bool is_dynamic = newparam.get<bool>("dynamic",false);
        discretized_param_dynamic.push_back(is_dynamic);
        if (is_dynamic) {
          have_dynamic_discretized = true;
        }
        
        initialParamValues.push_back(newparam.get<ScalarT>("initial_value",1.0));
        lowerParamBounds.push_back(newparam.get<ScalarT>("lower_bound",-1.0));
        upperParamBounds.push_back(newparam.get<ScalarT>("upper_bound",1.0));
        discparam_distribution.push_back(newparam.get<string>("distribution","uniform"));
        discparamVariance.push_back(newparam.get<ScalarT>("variance",1.0));
        
      }
      
      paramLowerBounds.push_back(lo);
      paramUpperBounds.push_back(up);
      
      pl_itr++;
    }
    
    if (have_dynamic_scalar) {
      for (size_t i=0; i<numTimeSteps; ++i) {
        paramvals.push_back(tmp_paramvals);
      }
    }
    else {
      paramvals.push_back(tmp_paramvals);
    }
    
#ifndef MrHyDE_NO_AD
    for (size_t block=0; block<blocknames.size(); ++block) {
      if ((int)num_active_params>disc->num_derivs_required[block]) {
        disc->num_derivs_required[block] = num_active_params;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(num_active_params > MAXDERIVS,std::runtime_error,"Error: MAXDERIVS is not large enough to support the number of parameters.");
#endif
    
    size_t maxcomp = 1;
    for (size_t k=0; k<paramvals.size(); k++) {
      for (size_t j=0; j<paramvals[k].size(); j++) {
        if (paramvals[k][j].size() > maxcomp) {
          maxcomp = paramvals[k][j].size();
        }
      }
    }
    
    size_t nump = 1;
    if (paramvals.size() > 0) {
      nump = paramvals[0].size();
    }
    paramvals_KV = Kokkos::View<ScalarT**,AssemblyDevice>("parameter values (ScalarT)", nump, maxcomp);
#ifndef MrHyDE_NO_AD
    paramvals_KVAD = Kokkos::View<AD**,AssemblyDevice>("parameter values (AD)", nump, maxcomp);
    paramvals_KVAD2 = Kokkos::View<AD2**,AssemblyDevice>("parameter values (AD2)", nump, maxcomp);
    paramvals_KVAD4 = Kokkos::View<AD4**,AssemblyDevice>("parameter values (AD4)", nump, maxcomp);
    paramvals_KVAD8 = Kokkos::View<AD8**,AssemblyDevice>("parameter values (AD8)", nump, maxcomp);
    paramvals_KVAD16 = Kokkos::View<AD16**,AssemblyDevice>("parameter values (AD16)", nump, maxcomp);
    paramvals_KVAD18 = Kokkos::View<AD18**,AssemblyDevice>("parameter values (AD18)", nump, maxcomp);
    paramvals_KVAD24 = Kokkos::View<AD24**,AssemblyDevice>("parameter values (AD24)", nump, maxcomp);
    paramvals_KVAD32 = Kokkos::View<AD32**,AssemblyDevice>("parameter values (AD32)", nump, maxcomp);
#endif
    
    int numind = 1;
    if (have_dynamic_scalar) {
      numind = numTimeSteps;
    }
    paramvals_KV_ALL = Kokkos::View<ScalarT***,AssemblyDevice>("parameter values (ScalarT)", numind, nump, maxcomp);
#ifndef MrHyDE_NO_AD
    paramvals_KVAD_ALL = Kokkos::View<AD***,AssemblyDevice>("parameter values (AD)", numind, nump, maxcomp);
    paramvals_KVAD2_ALL = Kokkos::View<AD2***,AssemblyDevice>("parameter values (AD2)", numind, nump, maxcomp);
    paramvals_KVAD4_ALL = Kokkos::View<AD4***,AssemblyDevice>("parameter values (AD4)", numind, nump, maxcomp);
    paramvals_KVAD8_ALL = Kokkos::View<AD8***,AssemblyDevice>("parameter values (AD8)", numind, nump, maxcomp);
    paramvals_KVAD16_ALL = Kokkos::View<AD16***,AssemblyDevice>("parameter values (AD16)", numind, nump, maxcomp);
    paramvals_KVAD18_ALL = Kokkos::View<AD18***,AssemblyDevice>("parameter values (AD18)", numind, nump, maxcomp);
    paramvals_KVAD24_ALL = Kokkos::View<AD24***,AssemblyDevice>("parameter values (AD24)", numind, nump, maxcomp);
    paramvals_KVAD32_ALL = Kokkos::View<AD32***,AssemblyDevice>("parameter values (AD32)", numind, nump, maxcomp);
#endif
    
    paramvals_KV = Kokkos::View<ScalarT**,AssemblyDevice>("parameter values (ScalarT)", nump, maxcomp);
#ifndef MrHyDE_NO_AD
    paramvals_KVAD = Kokkos::View<AD**,AssemblyDevice>("parameter values (AD)", nump, maxcomp);
    paramvals_KVAD2 = Kokkos::View<AD2**,AssemblyDevice>("parameter values (AD2)", nump, maxcomp);
    paramvals_KVAD4 = Kokkos::View<AD4**,AssemblyDevice>("parameter values (AD4)", nump, maxcomp);
    paramvals_KVAD8 = Kokkos::View<AD8**,AssemblyDevice>("parameter values (AD8)", nump, maxcomp);
    paramvals_KVAD16 = Kokkos::View<AD16**,AssemblyDevice>("parameter values (AD16)", nump, maxcomp);
    paramvals_KVAD18 = Kokkos::View<AD18**,AssemblyDevice>("parameter values (AD18)", nump, maxcomp);
    paramvals_KVAD24 = Kokkos::View<AD24**,AssemblyDevice>("parameter values (AD24)", nump, maxcomp);
    paramvals_KVAD32 = Kokkos::View<AD32**,AssemblyDevice>("parameter values (AD32)", nump, maxcomp);
#endif
  }
  
  debugger->print("**** Finished ParameterManager::setupParameters");
  
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
// Set up the discretized parameter DOF manager
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void ParameterManager<Node>::setupDiscretizedParameters(vector<vector<Teuchos::RCP<Group> > > & groups,
                                                        vector<vector<Teuchos::RCP<BoundaryGroup> > > & boundary_groups) {
  
  debugger->print("**** Starting ParameterManager::setupDiscretizedParameters ... ");
  
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
    Teuchos::RCP<panzer::ConnManager> conn = mesh->getSTKConnManager();
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
    
      TEUCHOS_TEST_FOR_EXCEPTION(numGIDs > MAXDERIVS,std::runtime_error,
                                 "Error: MAXDERIVS is not large enough to support the number of discretized parameter degrees of freedom per element on block: " + blocknames[block]);
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
        
        auto myElem = disc->my_elements[block];
        Kokkos::View<size_t*,AssemblyDevice> GEIDs("element IDs on device",myElem.size());
        auto host_GEIDs = Kokkos::create_mirror_view(GEIDs);
        for (size_t elem=0; elem<myElem.extent(0); elem++) {
          host_GEIDs(elem) = myElem(elem);
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
      auto EIDs = disc->my_elements[block];
      for (size_t e=0; e<EIDs.extent(0); e++) {
        vector<GO> gids;
        size_t elemID = EIDs(e);
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
          //if (have_dynamic_discretized) {
            for (size_t j=0; j<discretized_params_over.size(); ++j) {
              discretized_params_over[j]->replaceGlobalValue(param_dofs_OS[n][i],0,initialParamValues[n]);
            }
          //}
          //discretized_params_over->replaceGlobalValue(param_dofs_OS[n][i],0,initialParamValues[n]);
        }
      }
      paramNodesOS.push_back(param_dofs_OS[n]); // store for later use
      paramNodes.push_back(param_dofs[n]); // store for later use
    }
    //discretized_params->doExport(*discretized_params_over, *param_exporter, Tpetra::REPLACE);
    //if (have_dynamic_discretized) {
      for (size_t i=0; i<discretized_params_over.size(); ++i) {
        discretized_params[i]->doExport(*(discretized_params_over[i]), *param_exporter, Tpetra::REPLACE);
      }
    //}
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
  
  debugger->print("**** Finished ParameterManager::setupDiscretizedParameters");
  
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
// return the discretized parameters as vector for use with ROL (deprecated)
// ========================================================================================

template<class Node>
vector<ScalarT> ParameterManager<Node>::getDiscretizedParamsVector() {
  int numDParams = this->getNumParams(4);
  vector<ScalarT> discLocalParams(numDParams);
  vector<ScalarT> discParams(numDParams);
  auto Psol_2d = discretized_params[0]->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
  auto Psol_host = Kokkos::create_mirror_view(Psol_2d);
  for (size_t i = 0; i < paramOwned.size(); i++) {
    int gid = paramOwned[i];
    discLocalParams[gid] = Psol_host(i,0);
    //cout << gid << " " << Psol_2d(i,0) << endl;
  }
  for (int i = 0; i < numDParams; i++) {
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
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > cparams;
  int index = 0;
  if (have_dynamic_discretized) {
    index = dynamic_timeindex;
  }
  if (discretized_params.size() > index) {
    cparams = discretized_params[index];
  }
  return cparams;
}

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > ParameterManager<Node>::getDiscretizedParamsOver() {
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > cparams;
  int index = 0;
  if (have_dynamic_discretized) {
    index = dynamic_timeindex;
  }
  
  if (discretized_params_over.size() > index) {
    cparams = discretized_params_over[index];
  }
  return cparams;
}

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > ParameterManager<Node>::getDiscretizedParamsDot() {
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > cparamsdot = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
  cparamsdot->putScalar(0.0);
  
  int index = 0;
  if (have_dynamic_discretized) {
    index = dynamic_timeindex;
  }
  
  if (index > 0 && discretized_params.size() > index+1) {
    cparamsdot->update(1.0/dynamic_dt,*(discretized_params[index]),1.0);
    cparamsdot->update(-1.0/dynamic_dt,*(discretized_params[index-1]),1.0);
  }
  return cparamsdot;
}

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > ParameterManager<Node>::getDiscretizedParamsDotOver() {
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > cparamsdot = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
  
  cparamsdot->putScalar(0.0);
  
  int index = 0;
  if (have_dynamic_discretized) {
    index = dynamic_timeindex;
  }
  
  if (index > 0 && discretized_params_over.size() > index+1) {
    cparamsdot->update(1.0/dynamic_dt,*(discretized_params_over[index]),1.0);
    cparamsdot->update(-1.0/dynamic_dt,*(discretized_params_over[index-1]),1.0);
  }
  return cparamsdot;
}

// ========================================================================================
// ========================================================================================

template<class Node>
MrHyDE_OptVector ParameterManager<Node>::getCurrentVector() {
  
  vector<Teuchos::RCP<vector<ScalarT> > > new_active_params;
  vector<vector_RCP> new_disc_params;
  if (num_active_params > 0) {
    new_active_params = this->getParams(1);
  }
  //else {
    //new_active_params = Teuchos::null;
  //}
  if (globalParamUnknowns > 0) {
    //if (have_dynamic_discretized) {
      new_disc_params = discretized_params;
    //}
    //else {
    //  new_disc_params.push_back(Psol);
    //}
  }
    
  MrHyDE_OptVector newvec(new_disc_params, new_active_params, Comm->getRank());
  return newvec;
}

// ========================================================================================
// return the discretized parameters as vector of vector_RCPs for use with ROL
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > > ParameterManager<Node>::getDynamicDiscretizedParams() {
  return discretized_params;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setInitialParams() {
  
  debugger->print("**** Starting ParameterManager::setInitialParams ...");
  
  //auto Psol = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
  //auto Psol_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
  //Psol->putScalar(0.0);
  //Psol_over->putScalar(0.0); // TMW: why is this hard-coded???
  
  int numsols = 1;
  if (have_dynamic_discretized) {
    numsols = numTimeSteps;
  }
  
  for (int i=0; i<numsols; ++i) {
    vector_RCP dyninit = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
    vector_RCP dyninit_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
    dyninit->putScalar(0.0); // TMW: why is this hard-coded???
    dyninit_over->putScalar(0.0); // TMW: why is this hard-coded???
    discretized_params.push_back(dyninit);
    discretized_params_over.push_back(dyninit_over);
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
  
  debugger->print("**** Finished ParameterManager::setInitialParams ...");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::sacadoizeParams(const bool & seed_active) {
  
  if (paramvals.size() > 0) {
    if (paramvals[0].size() > 0) {
      
      size_t maxlength = paramvals_KV.extent(1);
      
      Kokkos::View<int*,AssemblyDevice> ptypes("parameter types",paramtypes.size());
      auto host_ptypes = Kokkos::create_mirror_view(ptypes);
      for (size_t i=0; i<paramtypes.size(); i++) {
        host_ptypes(i) = paramtypes[i];
      }
      Kokkos::deep_copy(ptypes, host_ptypes);
      
      Kokkos::View<size_t*,AssemblyDevice> plengths("parameter lengths",paramvals[0].size());
      auto host_plengths = Kokkos::create_mirror_view(plengths);
      for (size_t i=0; i<paramvals[0].size(); i++) {
        host_plengths(i) = paramvals[0][i].size();
      }
      Kokkos::deep_copy(plengths, host_plengths);
      
      size_t prog = 0;
      Kokkos::View<size_t**,AssemblyDevice> pseed("parameter seed index",paramvals[0].size(),maxlength);
      auto host_pseed = Kokkos::create_mirror_view(pseed);
      for (size_t i=0; i<paramvals[0].size(); i++) {
        if (paramtypes[i] == 1) {
          for (size_t j=0; j<paramvals[0][i].size(); j++) {
            host_pseed(i,j) = prog;
            prog++;
          }
        }
      }
      Kokkos::deep_copy(pseed,host_pseed);
      //KokkosTools::print(pseed);
      
      Kokkos::View<ScalarT***,AssemblyDevice> pvals("parameter values",paramvals.size(), paramvals[0].size(), maxlength);
      auto host_pvals = Kokkos::create_mirror_view(pvals);
      for (size_t k=0; k<paramvals.size(); k++) {
        for (size_t i=0; i<paramvals[k].size(); i++) {
          for (size_t j=0; j<paramvals[k][i].size(); j++) {
            host_pvals(k,i,j) = paramvals[k][i][j];
          }
        }
      }
      Kokkos::deep_copy(pvals, host_pvals);
      
      this->sacadoizeParamsSc(seed_active, ptypes, plengths, pseed, pvals, paramvals_KV_ALL);
#ifndef MrHyDE_NO_AD
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD2_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD4_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD8_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD16_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD18_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD24_ALL);
      this->sacadoizeParams(seed_active, ptypes, plengths, pseed, pvals, paramvals_KVAD32_ALL);
#endif
    }
  }
}

template<class Node>
void ParameterManager<Node>::sacadoizeParamsSc(const bool & seed_active,
                                             Kokkos::View<int*,AssemblyDevice> ptypes,
                                             Kokkos::View<size_t*,AssemblyDevice> plengths,
                                             Kokkos::View<size_t**,AssemblyDevice> pseed,
                                             Kokkos::View<ScalarT***,AssemblyDevice> pvals,
                                             Kokkos::View<ScalarT***,AssemblyDevice> kv_pvals) {
  
  parallel_for("parameter manager sacadoize - no seeding",
               RangePolicy<AssemblyExec>(0,pvals.extent(0)),
               KOKKOS_LAMBDA (const size_type i ) {
    for (size_t j=0; j<pvals.extent(1); j++) {
      for (size_t k=0; k<pvals.extent(2); k++) {
        kv_pvals(i,j,k) = pvals(i,j,k);
      }
    }
  });
  
}

template<class Node>
template<class EvalT>
void ParameterManager<Node>::sacadoizeParams(const bool & seed_active,
                                             Kokkos::View<int*,AssemblyDevice> ptypes,
                                             Kokkos::View<size_t*,AssemblyDevice> plengths,
                                             Kokkos::View<size_t**,AssemblyDevice> pseed,
                                             Kokkos::View<ScalarT***,AssemblyDevice> pvals,
                                             Kokkos::View<EvalT***,AssemblyDevice> kv_pvals) {
  
  
  if (paramvals.size() > 0) {
    if (seed_active) {
      
      parallel_for("parameter manager sacadoize - seed active",
                   RangePolicy<AssemblyExec>(0,pvals.extent(0)),
                   KOKKOS_LAMBDA (const size_type k ) {
        for (size_t i=0; i<plengths.extent(0); i++) {
          if (ptypes(i) == 1) { // active params
            for (size_t j=0; j<plengths(i); j++) {
              EvalT dummyval = 0.0;
              if (dummyval.size() > pseed(i,j)) {
                kv_pvals(k,i,j) = EvalT(dummyval.size(), pseed(i,j), pvals(k,i,j));
              }
              else {
                kv_pvals(k,i,j) = EvalT(pvals(k,i,j));
              }
            }
          }
          else {
            for (size_t j=0; j<plengths(i); j++) {
              kv_pvals(k,i,j) = EvalT(pvals(k,i,j));
            }
          }
        }
      });
    }
    else {
      parallel_for("parameter manager sacadoize - no seeding",
                   RangePolicy<AssemblyExec>(0,pvals.extent(0)),
                   KOKKOS_LAMBDA (const size_type index ) {
        for (size_t i=0; i<plengths.extent(0); i++) {
          for (size_t j=0; j<plengths(i); j++) {
            kv_pvals(index,i,j) = EvalT(pvals(index,i,j));
          }
        }
      });
    }
  }
}

// ========================================================================================
// ========================================================================================
#if defined(MrHyDE_ENABLE_HDSA)
template<class Node>
void ParameterManager<Node>::updateParams(const vector_RCP & newparams) {

        // only for steady state
        discretized_params[0]->assign(*newparams);
        discretized_params_over[0]->putScalar(0.0);
        discretized_params_over[0]->doImport(*newparams, *param_importer, Tpetra::ADD);
}
#endif
// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateParams(MrHyDE_OptVector & newparams) {
  
  if (newparams.haveScalar()) {
    auto scalar_params = newparams.getParameter();
    
    for (size_t i=0; i<paramvals.size(); i++) {
      size_t pprog = 0;
      for (size_t k=0; k<paramvals[i].size(); k++) {
        if (paramtypes[k] == 1) {
          auto cparams = scalar_params[i];
          for (size_t j=0; j<paramvals[i][k].size(); j++) {
            if (Comm->getRank() == 0 && verbosity > 0) {
              cout << "Updated Params: " << paramvals[i][k][j] << " (old value)   " << (*cparams)[pprog] << " (new value)" << endl;
            }
            paramvals[i][k][j] = (*cparams)[pprog];
            pprog++;
          }
        }
      }
    }
    
  }
  
  if (newparams.haveField()) {
    auto disc_params = newparams.getField();
    
    for (size_t i=0; i<disc_params.size(); ++i) {
      auto owned_vec = disc_params[i]->getVector();
      discretized_params[i]->assign(*owned_vec);
      discretized_params_over[i]->putScalar(0.0);
      discretized_params_over[i]->doImport(*owned_vec, *param_importer, Tpetra::ADD);
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
  
  for (size_t i=0; i<paramvals[0].size(); i++) {
    if (paramtypes[i] == type) {
      for (size_t j=0; j<paramvals[0][i].size(); j++) {
        if (Comm->getRank() == 0 && verbosity > 0) {
          cout << "Updated Params: " << paramvals[0][i][j] << " (old value)   " << newparams[pprog] << " (new value)" << endl;
        }
        paramvals[0][i][j] = newparams[pprog];
        pprog++;
      }
    }
  }
  
  /*
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
  */
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::updateDynamicParams(const int & timestep) {
  
  dynamic_timeindex = timestep;
  size_t index = 0;
  if (have_dynamic_scalar) {
    index = dynamic_timeindex;
  }
  
  auto pslice = subview(paramvals_KV_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KV, pslice);
  
#ifndef MrHyDE_NO_AD
  auto pslice_AD = subview(paramvals_KVAD_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD, pslice_AD);
  auto pslice_AD2 = subview(paramvals_KVAD2_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD2, pslice_AD2);
  auto pslice_AD4 = subview(paramvals_KVAD4_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD4, pslice_AD4);
  auto pslice_AD8 = subview(paramvals_KVAD8_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD8, pslice_AD8);
  auto pslice_AD16 = subview(paramvals_KVAD16_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD16, pslice_AD16);
  auto pslice_AD18 = subview(paramvals_KVAD18_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD18, pslice_AD18);
  auto pslice_AD24 = subview(paramvals_KVAD24_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD24, pslice_AD24);
  auto pslice_AD32 = subview(paramvals_KVAD32_ALL, index, ALL(), ALL());
  deep_copy(paramvals_KVAD32, pslice_AD32);
#endif
  
  if (index == 0) {
    parallel_for("paramman copy zero",
                 RangePolicy<AssemblyExec>(0,paramdot_KV.extent(0)),
                 KOKKOS_LAMBDA (const int c ) {
      for (size_type j=0; j<paramdot_KV.extent(1); j++) {
        paramdot_KV(c,j) = 0.0;
#ifndef MrHyDE_NO_AD
        paramdot_KVAD(c,j) = 0.0;
        paramdot_KVAD2(c,j) = 0.0;
        paramdot_KVAD4(c,j) = 0.0;
        paramdot_KVAD8(c,j) = 0.0;
        paramdot_KVAD16(c,j) = 0.0;
        paramdot_KVAD18(c,j) = 0.0;
        paramdot_KVAD24(c,j) = 0.0;
        paramdot_KVAD32(c,j) = 0.0;
#endif
      }
    });
  }
  else {
    parallel_for("paramman copy zero",
                 RangePolicy<AssemblyExec>(0,paramdot_KV.extent(0)),
                 KOKKOS_LAMBDA (const int c ) {
      for (size_type j=0; j<paramdot_KV.extent(1); j++) {
        paramdot_KV(c,j) = (paramvals_KV_ALL(index,c,j) - paramvals_KV_ALL(index-1,c,j))/dynamic_dt;
#ifndef MrHyDE_NO_AD
        paramdot_KVAD(c,j) = (paramvals_KVAD_ALL(index,c,j) - paramvals_KVAD_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD2(c,j) = (paramvals_KVAD2_ALL(index,c,j) - paramvals_KVAD2_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD4(c,j) = (paramvals_KVAD4_ALL(index,c,j) - paramvals_KVAD4_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD8(c,j) = (paramvals_KVAD8_ALL(index,c,j) - paramvals_KVAD8_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD16(c,j) = (paramvals_KVAD16_ALL(index,c,j) - paramvals_KVAD16_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD18(c,j) = (paramvals_KVAD18_ALL(index,c,j) - paramvals_KVAD18_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD24(c,j) = (paramvals_KVAD24_ALL(index,c,j) - paramvals_KVAD24_ALL(index-1,c,j).val())/dynamic_dt;
        paramdot_KVAD32(c,j) = (paramvals_KVAD32_ALL(index,c,j) - paramvals_KVAD32_ALL(index-1,c,j).val())/dynamic_dt;
#endif
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setParam(const vector<ScalarT> & newparams, const std::string & name) {
  size_t pprog = 0;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  
  int index = 0;
  if (have_dynamic_scalar) {
    index = dynamic_timeindex;
  }
  
  if (paramvals.size() > index) {
    for (size_t i=0; i<paramvals[index].size(); i++) {
      if (paramnames[i] == name) {
        for (size_t j=0; j<paramvals[index][i].size(); j++) {
          if (Comm->getRank() == 0 && verbosity > 0) {
            cout << "Updated Params: " << paramvals[index][i][j] << " (old value)   " << newparams[pprog] << " (new value)" << endl;
          }
          paramvals[index][i][j] = newparams[pprog];
          pprog++;
        }
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
  
  int index = 0;
  if (have_dynamic_scalar) {
    index = dynamic_timeindex;
  }
  
  if (paramvals.size() > index) {
    for (size_t i=0; i<paramvals[index].size(); i++) {
      if (paramtypes[i] == type) {
        for (size_t j=0; j<paramvals[index][i].size(); j++) {
          paramvals[index][i][j] = newparams[pprog];
          pprog++;
        }
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<Teuchos::RCP<vector<ScalarT> > > ParameterManager<Node>::getParams(const int & type) {
  vector<Teuchos::RCP<vector<ScalarT> > > reqparams;// = Teuchos::rcp(new std::vector<ScalarT>());
  for (size_t i=0; i<paramvals.size(); i++) {
    Teuchos::RCP<vector<ScalarT> > tmpparams = Teuchos::rcp(new std::vector<ScalarT>());
    for (size_t k=0; k<paramvals[i].size(); k++) {
      if (paramtypes[k] == type) {
        for (size_t j=0; j<paramvals[i][k].size(); j++) {
          tmpparams->push_back(paramvals[i][k][j]);
        }
      }
    }
    reqparams.push_back(tmpparams);
  }
  return reqparams;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<string> ParameterManager<Node>::getParamsNames(const int & type) {
  vector<string> reqparams;
  if (paramvals.size() > 0) {
    for (size_t i=0; i<paramvals[0].size(); i++) { // just first is fine
      if (paramtypes[i] == type) {
        reqparams.push_back(paramnames[i]);
      }
    }
  }
  return reqparams;
}

template<class Node>
bool ParameterManager<Node>::isParameter(const string & name) {
  bool isparam = false;
  if (paramvals.size() > 0) {
    for (size_t i=0; i<paramvals[0].size(); i++) { // just first is fine
      if (paramnames[i] == name) {
        isparam = true;
      }
    }
  }
  return isparam;
}

// ========================================================================================
// ========================================================================================

template<class Node>
vector<size_t> ParameterManager<Node>::getParamsLengths(const int & type) {
  vector<size_t> reqparams;
  if (paramvals.size() > 0) {
    for (size_t i=0; i<paramvals[0].size(); i++) { // first is fine
      if (paramtypes[i] == type) {
        reqparams.push_back(paramvals[0][i].size());
      }
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
    for (size_t k=0; k<paramvals[i].size(); k++) {
      if (paramtypes[k] == type) {
        for (size_t j=0; j<paramvals[i][k].size(); j++) {
          reqparams.push_back(paramvals[i][k][j]);
        }
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
  for (size_t i=0; i<paramvals[0].size(); i++) {
    if (paramtypes[i] == 1) {
      for (size_t j=0; j<paramvals[0][i].size(); j++) {
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
      for (size_t k=0; k<paramvals[i].size(); k++) {
        if (paramtypes[k] == 1) {
          for (size_t j=0; j<paramvals[i][k].size(); j++) {
            respOUT << paramvals[i][k][j] << endl;
          }
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
