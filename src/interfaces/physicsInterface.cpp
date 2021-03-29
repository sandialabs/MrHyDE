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

#include "physicsInterface.hpp"
#include "physicsImporter.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

physics::physics(Teuchos::RCP<Teuchos::ParameterList> & settings_, Teuchos::RCP<MpiComm> & Comm_,
                 Teuchos::RCP<panzer_stk::STK_Interface> & mesh) :
settings(settings_), Commptr(Comm_){
  
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics constructor ..." << endl;
    }
  }
  
  mesh->getElementBlockNames(blocknames);
  mesh->getSidesetNames(sideNames);
  
  numBlocks = blocknames.size();
  spaceDim = mesh->getDimension();
  
  for (size_t b=0; b<blocknames.size(); b++) {
    if (settings->sublist("Physics").isSublist(blocknames[b])) { // adding block overwrites the default
      blockPhysSettings.push_back(settings->sublist("Physics").sublist(blocknames[b]));
    }
    else { // default
      blockPhysSettings.push_back(settings->sublist("Physics"));
    }
    
    if (settings->sublist("Discretization").isSublist(blocknames[b])) { // adding block overwrites default
      blockDiscSettings.push_back(settings->sublist("Discretization").sublist(blocknames[b]));
    }
    else { // default
      blockDiscSettings.push_back(settings->sublist("Discretization"));
    }
  }
  
  this->importPhysics(false);
  
  if (settings->isSublist("Aux Physics") && settings->isSublist("Aux Discretization")) {
    have_aux = true;
    for (size_t b=0; b<blocknames.size(); b++) {
      if (settings->sublist("Aux Physics").isSublist(blocknames[b])) { // adding block overwrites the default
        aux_blockPhysSettings.push_back(settings->sublist("Aux Physics").sublist(blocknames[b]));
      }
      else { // default
        aux_blockPhysSettings.push_back(settings->sublist("Aux Physics"));
      }
      
      if (settings->sublist("Aux Discretization").isSublist(blocknames[b])) { // adding block overwrites default
        aux_blockDiscSettings.push_back(settings->sublist("Aux Discretization").sublist(blocknames[b]));
      }
      else { // default
        aux_blockDiscSettings.push_back(settings->sublist("Aux Discretization"));
      }
    }
    this->importPhysics(true);
  }
  else {
    for (size_t b=0; b<blocknames.size(); b++) {
      vector<string> avars;
      aux_varlist.push_back(avars);
    }
  }

  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics constructor" << endl;
    }
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////
// Add the functions to the function managers
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::defineFunctions(vector<Teuchos::RCP<FunctionManager> > & functionManagers_) {
  
  functionManagers = functionManagers_;
  
  for (size_t b=0; b<blocknames.size(); b++) {
    Teuchos::ParameterList fs;
    if (settings->sublist("Functions").isSublist(blocknames[b])) {
      fs = settings->sublist("Functions").sublist(blocknames[b]);
    }
    else {
      fs = settings->sublist("Functions");
    }
    
    for (size_t n=0; n<modules[b].size(); n++) {
      modules[b][n]->defineFunctions(fs, functionManagers[b]);
    }
    
    // Add initial conditions
    Teuchos::ParameterList initial_conds = blockPhysSettings[b].sublist("Initial conditions");
    for (size_t j=0; j<varlist[b].size(); j++) {
      string expression;
      if (initial_conds.isType<string>(varlist[b][j])) {
        expression = initial_conds.get<string>(varlist[b][j]);
      }
      else if (initial_conds.isType<double>(varlist[b][j])) {
        double value = initial_conds.get<double>(varlist[b][j]);
        expression = std::to_string(value);
      }
      else {
        expression = "0.0";
      }
      functionManagers[b]->addFunction("initial "+varlist[b][j],expression,"ip");
      functionManagers[b]->addFunction("initial "+varlist[b][j],expression,"point");
    }
    
    // Dirichlet conditions
    Teuchos::ParameterList dbcs = blockPhysSettings[b].sublist("Dirichlet conditions");
    for (size_t j=0; j<varlist[b].size(); j++) {
      if (dbcs.isSublist(varlist[b][j])) {
        if (dbcs.sublist(varlist[b][j]).isType<string>("all boundaries")) {
          string entry = dbcs.sublist(varlist[b][j]).get<string>("all boundaries");
          for (size_t s=0; s<sideNames.size(); s++) {
            string label = "Dirichlet " + varlist[b][j] + " " + sideNames[s];
            functionManagers[b]->addFunction(label,entry,"side ip");
          }
        }
        else if (dbcs.sublist(varlist[b][j]).isType<double>("all boundaries")) {
          double value = dbcs.sublist(varlist[b][j]).get<double>("all boundaries");
          string entry = std::to_string(value);
          for (size_t s=0; s<sideNames.size(); s++) {
            string label = "Dirichlet " + varlist[b][j] + " " + sideNames[s];
            functionManagers[b]->addFunction(label,entry,"side ip");
          }
        }
        else {
          Teuchos::ParameterList currdbcs = dbcs.sublist(varlist[b][j]);
          Teuchos::ParameterList::ConstIterator d_itr = currdbcs.begin();
          while (d_itr != currdbcs.end()) {
            if (currdbcs.isType<string>(d_itr->first)) {
              string entry = currdbcs.get<string>(d_itr->first);
              string label = "Dirichlet " + varlist[b][j] + " " + d_itr->first;
              functionManagers[b]->addFunction(label,entry,"side ip");
            }
            else if (currdbcs.isType<double>(d_itr->first)) {
              double value = currdbcs.get<double>(d_itr->first);
              string entry = std::to_string(value);
              string label = "Dirichlet " + varlist[b][j] + " " + d_itr->first;
              functionManagers[b]->addFunction(label,entry,"side ip");
            }
            d_itr++;
          }
        }
      }
    }
    
    // Neumann/robin conditions
    Teuchos::ParameterList nbcs = blockPhysSettings[b].sublist("Neumann conditions");
    for (size_t j=0; j<varlist[b].size(); j++) {
      if (nbcs.isSublist(varlist[b][j])) {
        if (nbcs.sublist(varlist[b][j]).isParameter("all boundaries")) {
          string entry = nbcs.sublist(varlist[b][j]).get<string>("all boundaries");
          for (size_t s=0; s<sideNames.size(); s++) {
            string label = "Neumann " + varlist[b][j] + " " + sideNames[s];
            functionManagers[b]->addFunction(label,entry,"side ip");
          }
        }
        else {
          Teuchos::ParameterList currnbcs = nbcs.sublist(varlist[b][j]);
          Teuchos::ParameterList::ConstIterator n_itr = currnbcs.begin();
          while (n_itr != currnbcs.end()) {
            string entry = currnbcs.get<string>(n_itr->first);
            string label = "Neumann " + varlist[b][j] + " " + n_itr->first;
            functionManagers[b]->addFunction(label,entry,"side ip");
            n_itr++;
          }
        }
      }
    }
    
    /*
    vector<string> block_ef;
    Teuchos::ParameterList efields = blockPhysSettings[b].sublist("Extra fields");
    Teuchos::ParameterList::ConstIterator ef_itr = efields.begin();
    while (ef_itr != efields.end()) {
      string entry = efields.get<string>(ef_itr->first);
      block_ef.push_back(ef_itr->first);
      functionManagers[b]->addFunction(ef_itr->first,entry,"ip");
      functionManagers[b]->addFunction(ef_itr->first,entry,"point");
      ef_itr++;
    }
    extrafields_list.push_back(block_ef);
    
    vector<string> block_ecf;
    Teuchos::ParameterList ecfields = blockPhysSettings[b].sublist("Extra cell fields");
    Teuchos::ParameterList::ConstIterator ecf_itr = ecfields.begin();
    while (ecf_itr != ecfields.end()) {
      string entry = ecfields.get<string>(ecf_itr->first);
      block_ecf.push_back(ecf_itr->first);
      functionManagers[b]->addFunction(ecf_itr->first,entry,"ip");
      ecf_itr++;
    }
    extracellfields_list.push_back(block_ecf);
    */
    
    /*
    vector<string> block_resp;
    Teuchos::ParameterList rfields = blockPhysSettings[b].sublist("Responses");
    Teuchos::ParameterList::ConstIterator r_itr = rfields.begin();
    while (r_itr != rfields.end()) {
      string entry = rfields.get<string>(r_itr->first);
      block_resp.push_back(r_itr->first);
      functionManagers[b]->addFunction(r_itr->first,entry,"point");
      functionManagers[b]->addFunction(r_itr->first,entry,"ip");
      r_itr++;
    }
    response_list.push_back(block_resp);
    
    vector<string> block_targ;
    Teuchos::ParameterList tfields = blockPhysSettings[b].sublist("Targets");
    Teuchos::ParameterList::ConstIterator t_itr = tfields.begin();
    while (t_itr != tfields.end()) {
      string entry = tfields.get<string>(t_itr->first);
      block_targ.push_back(t_itr->first);
      functionManagers[b]->addFunction(t_itr->first,entry,"point");
      functionManagers[b]->addFunction(t_itr->first,entry,"ip");
      t_itr++;
    }
    target_list.push_back(block_targ);
    
    vector<string> block_wts;
    Teuchos::ParameterList wfields = blockPhysSettings[b].sublist("Weights");
    Teuchos::ParameterList::ConstIterator w_itr = wfields.begin();
    while (w_itr != wfields.end()) {
      string entry = wfields.get<string>(w_itr->first);
      block_wts.push_back(w_itr->first);
      functionManagers[b]->addFunction(w_itr->first,entry,"point");
      functionManagers[b]->addFunction(w_itr->first,entry,"ip");
      w_itr++;
    }
    weight_list.push_back(block_wts);
    */
    
  }
  
  for (size_t b=0; b<blocknames.size(); b++) {
    Teuchos::ParameterList functions;
    if (settings->sublist("Functions").isSublist(blocknames[b])) {
      functions = settings->sublist("Functions").sublist(blocknames[b]);
    }
    else {
      functions = settings->sublist("Functions");
    }
    Teuchos::ParameterList::ConstIterator fnc_itr = functions.begin();
    while (fnc_itr != functions.end()) {
      string entry = functions.get<string>(fnc_itr->first);
      functionManagers[b]->addFunction(fnc_itr->first,entry,"ip");
      functionManagers[b]->addFunction(fnc_itr->first,entry,"side ip");
      functionManagers[b]->addFunction(fnc_itr->first,entry,"point");
      fnc_itr++;
    }
  }
    
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Add the requested physics modules, variables, discretization types
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::importPhysics(const bool & isaux) {
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics::importPhysics ..." << endl;
    }
  }
  
  for (size_t b=0; b<blocknames.size(); b++) {
    vector<int> currorders;
    vector<string> currtypes;
    vector<string> currvarlist;
    vector<int> currvarowned;
    
    vector<Teuchos::RCP<physicsbase> > currmodules;
    vector<bool> currSubgrid, curruseDG;
    std::string var;
    int default_order = 1;
    std::string default_type = "HGRAD";
    string module_list;
    if (isaux) {
      module_list = aux_blockPhysSettings[b].get<string>("modules","");
    }
    else {
      module_list = blockPhysSettings[b].get<string>("modules","");
    }
    
    vector<string> enabled_modules;
    // Script to break delimited list into pieces
    {
      string delimiter = ", ";
      size_t pos = 0;
      if (module_list.find(delimiter) == string::npos) {
        enabled_modules.push_back(module_list);
      }
      else {
        string token;
        while ((pos = module_list.find(delimiter)) != string::npos) {
          token = module_list.substr(0, pos);
          enabled_modules.push_back(token);
          module_list.erase(0, pos + delimiter.length());
        }
        enabled_modules.push_back(module_list);
      }
    }
    
    physicsImporter physimp = physicsImporter();
    currmodules = physimp.import(enabled_modules, settings, isaux, Commptr);
    
    if (isaux) {
      aux_modules.push_back(currmodules);
    }
    else {
      modules.push_back(currmodules);
      //useSubgrid.push_back(currSubgrid);
    }
    
    for (size_t m=0; m<currmodules.size(); m++) {
      vector<string> cvars = currmodules[m]->myvars;
      vector<string> ctypes = currmodules[m]->mybasistypes;
      for (size_t v=0; v<cvars.size(); v++) {
        currvarlist.push_back(cvars[v]);
        
        if (ctypes[v] == "HGRAD-DG") {
          currtypes.push_back("HGRAD");
          curruseDG.push_back(true);
        }
        else if (ctypes[v] == "HDIV-DG") {
          currtypes.push_back("HDIV");
          curruseDG.push_back(true);
        }
        else if (ctypes[v] == "HCURL-DG") {
          currtypes.push_back("HCURL");
          curruseDG.push_back(true);
        }
        else {
          currtypes.push_back(ctypes[v]);
          curruseDG.push_back(false);
        }
        currvarowned.push_back(m);
        currorders.push_back(blockDiscSettings[b].sublist("order").get<int>(cvars[v],default_order));
      }
    }
    
    if (isaux) {
      if (aux_blockPhysSettings[b].isSublist("Extra variables")) {
        Teuchos::ParameterList evars = aux_blockPhysSettings[b].sublist("Extra variables");
        Teuchos::ParameterList::ConstIterator pl_itr = evars.begin();
        while (pl_itr != evars.end()) {
          string newvar = pl_itr->first;
          currvarlist.push_back(newvar);
          
          string newtype = evars.get<string>(pl_itr->first);
          if (newtype == "HGRAD-DG") {
            currtypes.push_back("HGRAD");
            curruseDG.push_back(true);
          }
          else if (newtype == "HDIV-DG") {
            currtypes.push_back("HDIV");
            curruseDG.push_back(true);
          }
          else if (newtype == "HCURL-DG") {
            currtypes.push_back("HCURL");
            curruseDG.push_back(true);
          }
          else {
            currtypes.push_back(newtype);
            curruseDG.push_back(false);
          }
          //currvarowned.push_back(m);
          
          int neworder = default_order;
          if (aux_blockDiscSettings[b].sublist("order").isSublist("Extra variables")) {
            neworder = aux_blockDiscSettings[b].sublist("order").sublist("Extra variables").get<int>(newvar,default_order);
          }
          currorders.push_back(neworder);
          pl_itr++;
        }
      }
    }
    else {
      if (blockPhysSettings[b].isSublist("Extra variables")) {
        Teuchos::ParameterList evars = blockPhysSettings[b].sublist("Extra variables");
        Teuchos::ParameterList::ConstIterator pl_itr = evars.begin();
        while (pl_itr != evars.end()) {
          string newvar = pl_itr->first;
          currvarlist.push_back(newvar);
          
          string newtype = evars.get<string>(pl_itr->first);
          if (newtype == "HGRAD-DG") {
            currtypes.push_back("HGRAD");
            curruseDG.push_back(true);
          }
          else if (newtype == "HDIV-DG") {
            currtypes.push_back("HDIV");
            curruseDG.push_back(true);
          }
          else if (newtype == "HCURL-DG") {
            currtypes.push_back("HCURL");
            curruseDG.push_back(true);
          }
          else {
            currtypes.push_back(newtype);
            curruseDG.push_back(false);
          }
          //currvarowned.push_back(m);
          
          int neworder = default_order;
          if (blockDiscSettings[b].sublist("order").isSublist("Extra variables")) {
            neworder = blockDiscSettings[b].sublist("order").sublist("Extra variables").get<int>(newvar,default_order);
          }
          currorders.push_back(neworder);
          pl_itr++;
        }
      }
    }
    
    int currnumVars = currvarlist.size();
    TEUCHOS_TEST_FOR_EXCEPTION(currnumVars==0,std::runtime_error,"Error: no variable were added on block: " + blocknames[b]);
    
    std::vector<int> currunique_orders;
    std::vector<string> currunique_types;
    std::vector<int> currunique_index;
    
    for (size_t j=0; j<currorders.size(); j++) {
      bool is_unique = true;
      for (size_t k=0; k<currunique_orders.size(); k++) {
        if (currunique_orders[k] == currorders[j] && currunique_types[k] == currtypes[j]) {
          is_unique = false;
          currunique_index.push_back(k);
        }
      }
      if (is_unique) {
        currunique_orders.push_back(currorders[j]);
        currunique_types.push_back(currtypes[j]);
        currunique_index.push_back(currunique_orders.size()-1);
      }
    }
    
    vector<string> discretized_param_basis_types;
    vector<int> discretized_param_basis_orders;
    if (settings->isSublist("Parameters")) {
      Teuchos::ParameterList parameters = settings->sublist("Parameters");
      Teuchos::ParameterList::ConstIterator pl_itr = parameters.begin();
      while (pl_itr != parameters.end()) {
        Teuchos::ParameterList newparam = parameters.sublist(pl_itr->first);
        if (newparam.get<string>("usage") == "discretized") {
          discretized_param_basis_types.push_back(newparam.get<string>("type","HGRAD"));
          discretized_param_basis_orders.push_back(newparam.get<int>("order",1));
        }
        pl_itr++;
      }
    }
    
    for (size_t j=0; j<discretized_param_basis_orders.size(); j++) {
      bool is_unique = true;
      for (size_t k=0; k<currunique_orders.size(); k++) {
        if (currunique_orders[k] == discretized_param_basis_orders[j] && currunique_types[k] == discretized_param_basis_types[j]) {
          is_unique = false;
          //currunique_index.push_back(k);
        }
      }
      if (is_unique) {
        currunique_orders.push_back(discretized_param_basis_orders[j]);
        currunique_types.push_back(discretized_param_basis_types[j]);
      //  currunique_index.push_back(currunique_orders.size()-1);
      }
    }
    if (isaux) {
      aux_orders.push_back(currorders);
      aux_types.push_back(currtypes);
      aux_varlist.push_back(currvarlist);
      aux_varowned.push_back(currvarowned);
      aux_numVars.push_back(currnumVars);
      aux_useDG.push_back(curruseDG);
      aux_unique_orders.push_back(currunique_orders);
      aux_unique_types.push_back(currunique_types);
      aux_unique_index.push_back(currunique_index);
    }
    else {
      orders.push_back(currorders);
      types.push_back(currtypes);
      varlist.push_back(currvarlist);
      varowned.push_back(currvarowned);
      numVars.push_back(currnumVars);
      useDG.push_back(curruseDG);
      unique_orders.push_back(currunique_orders);
      unique_types.push_back(currunique_types);
      unique_index.push_back(currunique_index);
    }
  }
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics::importPhysics ..." << endl;
    }
  }
  
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int physics::getvarOwner(const int & block, const string & var) {
  int owner = 0;
  for (size_t k=0; k<varlist[block].size(); k++) {
    if (varlist[block][k] == var) {
      owner = varowned[block][k];
    }
  }
  return owner;
  
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

// TMW: this function is probably never used

AD physics::getDirichletValue(const int & block, const ScalarT & x, const ScalarT & y,
                              const ScalarT & z, const ScalarT & t, const string & var,
                              const string & gside, const bool & useadjoint,
                              Teuchos::RCP<workset> & wkset) {
  
  // update point in wkset
  auto xpt = wkset->getDataSc("x point");
  Kokkos::deep_copy(xpt,x);
  
  auto ypt = wkset->getDataSc("y point");
  Kokkos::deep_copy(ypt,y);
  
  if (spaceDim == 3) {
    auto zpt = wkset->getDataSc("z point");
    Kokkos::deep_copy(zpt,z);
  }
  
  wkset->setTime(t);
  
  // evaluate the response
  auto ddata = functionManagers[block]->evaluate("Dirichlet " + var + " " + gside,"point");
  AD val = 0.0;
  return ddata(0,0);
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

ScalarT physics::getInitialValue(const int & block, const ScalarT & x, const ScalarT & y,
                                const ScalarT & z, const string & var, const bool & useadjoint) {
  
  /*
  // update point in wkset
  wkset->point_KV(0,0,0) = x;
  wkset->point_KV(0,0,1) = y;
  wkset->point_KV(0,0,2) = z;
  
  // evaluate the response
  View_AD2_sv idata = functionManager->evaluate("initial " + var,"point",block);
  return idata(0,0).val();
  */
  return 0.0;
}

/////////////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////////////

bool physics::checkFace(const size_t & block){
  bool include_face = false;
  for (size_t i=0; i<modules[block].size(); i++) {
    bool cuseef = modules[block][i]->include_face;
    if (cuseef) {
      include_face = true;
    }
  }
  
  return include_face;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc3 physics::getInitial(const View_Sc3 ip, const int & block,
                             const bool & project, Teuchos::RCP<workset> & wkset) {
  
  
  size_t numElem = ip.extent(0);
  size_t numVars = varlist[block].size();
  size_t numip = ip.extent(1);
  
  View_Sc3 ivals("temp invals", numElem, numVars, numip);
  
  
  if (project) {
    // ip in wkset are set in cell::getInitial
    for (size_t n=0; n<varlist[block].size(); n++) {
  
      auto ivals_AD = functionManagers[block]->evaluate("initial " + varlist[block][n],"ip");
      auto cvals = subview( ivals, ALL(), n, ALL());
      //copy
      parallel_for("physics fill initial values",RangePolicy<AssemblyExec>(0,cvals.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t i=0; i<cvals.extent(1); i++) {
          cvals(e,i) = ivals_AD(e,i).val();
        }
      });
    }
  }
  else {
    // TMW: will not work on device yet
    
    size_type dim = ip.extent(2);
    View_Sc2 x,y,z;
    x = wkset->getDataSc("x point");
    if (dim > 1) {
      y = wkset->getDataSc("y point");
    }
    if (dim > 2) {
      z = wkset->getDataSc("z point");
    }
    //auto point_KV = wkset->point;
    auto host_ivals = Kokkos::create_mirror_view(ivals);
    for (size_t e=0; e<numElem; e++) {
      for (size_t i=0; i<numip; i++) {
        // set the node in wkset
        auto node = subview( ip, e, i, ALL());
        
        parallel_for("physics initial set point",RangePolicy<AssemblyExec>(0,node.extent(0)), KOKKOS_LAMBDA (const int s ) {
          x(0,0) = node(0);
          if (dim > 1) {
            y(0,0) = node(1);
          }
          if (dim > 2) {
            z(0,0) = node(2);
          }
          
        });
        
        for (size_t n=0; n<varlist[block].size(); n++) {
          // evaluate
          auto ivals_AD = functionManagers[block]->evaluate("initial " + varlist[block][n],"point");
          
          ivals(e,n,i) = ivals_AD(0,0).val();
          // copy
          //auto iv = Kokkos::subview( ivals, e, n, i);
          //parallel_for("physics initial set point",RangePolicy<AssemblyExec>(0,1), KOKKOS_LAMBDA (const int s ) {
          //  iv(0) = ivals_AD(0,0).val();
          //});
        }
      }
    }
  }
   
  //KokkosTools::print(ivals);
  return ivals;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc2 physics::getDirichlet(const View_Sc3 ip, const int & var,
                               const int & block,
                               const std::string & sidename,
                               Teuchos::RCP<workset> & wkset) {
  
  
  size_t numElem = ip.extent(0);
  size_t numip = ip.extent(1);
  
  View_Sc2 dvals("temp dnvals", numElem, numip);
  
  // evaluate
  auto dvals_AD = functionManagers[block]->evaluate("Dirichlet " + varlist[block][var] + " " + sidename,"side ip");
  
  // copy values
  parallel_for("physics fill Dirichlet values",RangePolicy<AssemblyExec>(0,dvals.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (size_t i=0; i<dvals.extent(1); i++) {
      dvals(e,i) = dvals_AD(e,i).val();
    }
  });
  return dvals;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::setVars() {
  for (size_t block=0; block<modules.size(); ++block) {
    for (size_t i=0; i<modules[block].size(); ++i) {
      if (varlist[block].size() > 0){
        //modules[block][i]->setVars(varlist[block]);
      }
    }
  }
}

void physics::setAuxVars(size_t & block, vector<string> & vars) {
  for (size_t i=0; i<modules[block].size(); i++) {
    //modules[block][i]->setAuxVars(vars);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params,
                               const vector<string> & paramnames) {
  
  //needs to be deprecated
  //udfunc->updateParameters(params,paramnames);
  
  for (size_t b=0; b<modules.size(); b++) {
    for (size_t i=0; i<modules[b].size(); i++) {
      modules[b][i]->updateParameters(params, paramnames);
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int physics::getUniqueIndex(const int & block, const std::string & var) {
  int index = 0;
  for (int j=0; j<numVars[block]; j++) {
    if (varlist[block][j] == var)
    index = unique_index[block][j];
  }
  return index;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::volumeResidual(const size_t block) {
  if (debug_level > 1) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics volume residual ..." << endl;
    }
  }
  for (size_t i=0; i<modules[block].size(); i++) {
    modules[block][i]->volumeResidual();
  }
  if (debug_level > 1) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics volume residual" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::boundaryResidual(const size_t block) {
  for (size_t i=0; i<modules[block].size(); i++) {
    modules[block][i]->boundaryResidual();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::computeFlux(const size_t block) {
  for (size_t i=0; i<modules[block].size(); i++) {
    modules[block][i]->computeFlux();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::setWorkset(vector<Teuchos::RCP<workset> > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t i=0; i<modules[block].size(); i++) {
        modules[block][i]->setWorkset(wkset[block]);//setWorkset(wkset[block]);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void physics::faceResidual(const size_t block) {
  for (size_t i=0; i<modules[block].size(); i++) {
    modules[block][i]->faceResidual();
  }
}
