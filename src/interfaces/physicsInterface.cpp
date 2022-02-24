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

PhysicsInterface::PhysicsInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                   Teuchos::RCP<MpiComm> & Comm_,
                                   Teuchos::RCP<panzer_stk::STK_Interface> & mesh) :
settings(settings_), Commptr(Comm_){
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debug_level = settings->get<int>("debug level",0);
  
  if (debug_level > 0 && Commptr->getRank() == 0) {
    cout << "**** Starting PhysicsInterface constructor ..." << endl;
  }
  
  mesh->getElementBlockNames(blocknames);
  mesh->getSidesetNames(sidenames);
  
  spaceDim = mesh->getDimension();
  
  if (settings->sublist("Physics").isParameter("physics set names")) {
    string names = settings->sublist("Physics").get<string>("physics set names");
    setnames = this->breakupList(names,", ");
  }
  else {
    setnames.push_back("default");
  }
  
  for (size_t set=0; set<setnames.size(); ++set) {
    Teuchos::ParameterList psetlist, dsetlist;
    if (setnames[set] == "default") {
      psetlist = settings->sublist("Physics");
      dsetlist = settings->sublist("Discretization");
    }
    else {
      psetlist = settings->sublist("Physics").sublist(setnames[set]);
      dsetlist = settings->sublist("Discretization").sublist(setnames[set]);
    }
    
    vector<Teuchos::ParameterList> currpsettings, currdsettings;
    for (size_t block=0; block<blocknames.size(); ++block) {
      if (psetlist.isSublist(blocknames[block])) { // adding block overwrites the default
        currpsettings.push_back(psetlist.sublist(blocknames[block]));
      }
      else { // default
        currpsettings.push_back(psetlist);
      }
      
      if (dsetlist.isSublist(blocknames[block])) { // adding block overwrites default
        currdsettings.push_back(dsetlist.sublist(blocknames[block]));
      }
      else { // default
        currdsettings.push_back(dsetlist);
      }
    }
    setPhysSettings.push_back(currpsettings);
    setDiscSettings.push_back(currdsettings);
  }
    
  this->importPhysics();
  
  if (debug_level > 0 && Commptr->getRank() == 0) {
    cout << "**** Finished physics constructor" << endl;
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////
// Add the functions to the function managers
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager> > & functionManagers_) {

  functionManagers = functionManagers_;
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    Teuchos::ParameterList fs;
    if (settings->sublist("Functions").isSublist(blocknames[block])) {
      fs = settings->sublist("Functions").sublist(blocknames[block]);
    }
    else {
      fs = settings->sublist("Functions");
    }
    
    for (size_t set=0; set<modules.size(); set++) {
      for (size_t n=0; n<modules[set][block].size(); n++) {
        modules[set][block][n]->defineFunctions(fs, functionManagers[block]);
      }
    }
  }
  
  // Add initial conditions
  for (size_t set=0; set<setnames.size(); set++) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      if (setPhysSettings[set][block].isSublist("Initial conditions")) {
        Teuchos::ParameterList initial_conds = setPhysSettings[set][block].sublist("Initial conditions");
        for (size_t j=0; j<varlist[set][block].size(); j++) {
          string expression;
          string var = varlist[set][block][j];
          if (initial_conds.isType<string>(var)) {
            expression = initial_conds.get<string>(var);
          }
          else if (initial_conds.isType<double>(var)) {
            double value = initial_conds.get<double>(var);
            expression = std::to_string(value);
          }
          else {
            expression = "0.0";
          }
          functionManagers[block]->addFunction("initial "+var,expression,"ip");
          functionManagers[block]->addFunction("initial "+var,expression,"point");
          if (types[set][block][j] == "HFACE") {
            // we have found an HFACE variable and need to have side ip evaluations
            // TODO check aux, etc?
            functionManagers[block]->addFunction("initial "+var,expression,"side ip");
          }
        }
      }
    }
  }
    
  // Dirichlet conditions
  for (size_t set=0; set<setnames.size(); set++) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      if (setPhysSettings[set][block].isSublist("Dirichlet conditions")) {
        Teuchos::ParameterList dbcs = setPhysSettings[set][block].sublist("Dirichlet conditions");
        for (size_t j=0; j<varlist[set][block].size(); j++) {
          string var = varlist[set][block][j];
          if (dbcs.isSublist(var)) {
            if (dbcs.sublist(var).isType<string>("all boundaries")) {
              string entry = dbcs.sublist(var).get<string>("all boundaries");
              for (size_t s=0; s<sidenames.size(); s++) {
                string label = "Dirichlet " + var + " " + sidenames[s];
                functionManagers[block]->addFunction(label,entry,"side ip");
              }
            }
            else if (dbcs.sublist(var).isType<double>("all boundaries")) {
              double value = dbcs.sublist(var).get<double>("all boundaries");
              string entry = std::to_string(value);
              for (size_t s=0; s<sidenames.size(); s++) {
                string label = "Dirichlet " + var + " " + sidenames[s];
                functionManagers[block]->addFunction(label,entry,"side ip");
              }
            }
            else {
              Teuchos::ParameterList currdbcs = dbcs.sublist(var);
              Teuchos::ParameterList::ConstIterator d_itr = currdbcs.begin();
              while (d_itr != currdbcs.end()) {
                if (currdbcs.isType<string>(d_itr->first)) {
                  string entry = currdbcs.get<string>(d_itr->first);
                  string label = "Dirichlet " + var + " " + d_itr->first;
                  functionManagers[block]->addFunction(label,entry,"side ip");
                }
                else if (currdbcs.isType<double>(d_itr->first)) {
                  double value = currdbcs.get<double>(d_itr->first);
                  string entry = std::to_string(value);
                  string label = "Dirichlet " + var + " " + d_itr->first;
                  functionManagers[block]->addFunction(label,entry,"side ip");
                }
                d_itr++;
              }
            }
          }
        }
      }
    }
  }
  
  // Neumann/robin conditions
  for (size_t set=0; set<setnames.size(); set++) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      Teuchos::ParameterList nbcs = setPhysSettings[set][block].sublist("Neumann conditions");
      for (size_t j=0; j<varlist[set][block].size(); j++) {
        string var = varlist[set][block][j];
        if (nbcs.isSublist(var)) {
          if (nbcs.sublist(var).isParameter("all boundaries")) {
            string entry = nbcs.sublist(var).get<string>("all boundaries");
            for (size_t s=0; s<sidenames.size(); s++) {
              string label = "Neumann " + var + " " + sidenames[s];
              functionManagers[block]->addFunction(label,entry,"side ip");
            }
          }
          else {
            Teuchos::ParameterList currnbcs = nbcs.sublist(var);
            Teuchos::ParameterList::ConstIterator n_itr = currnbcs.begin();
            while (n_itr != currnbcs.end()) {
              string entry = currnbcs.get<string>(n_itr->first);
              string label = "Neumann " + var + " " + n_itr->first;
              functionManagers[block]->addFunction(label,entry,"side ip");
              n_itr++;
            }
          }
        }
      }
    }
  }

  // Far-field conditions
  for (size_t set=0; set<setnames.size(); set++) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      Teuchos::ParameterList fbcs = setPhysSettings[set][block].sublist("Far-field conditions");
      for (size_t j=0; j<varlist[set][block].size(); j++) {
        string var = varlist[set][block][j];
        if (fbcs.isSublist(var)) {
          if (fbcs.sublist(var).isParameter("all boundaries")) {
            string entry = fbcs.sublist(var).get<string>("all boundaries");
            for (size_t s=0; s<sidenames.size(); s++) {
              string label = "Far-field " + var + " " + sidenames[s];
              functionManagers[block]->addFunction(label,entry,"side ip");
            }
          }
          else {
            Teuchos::ParameterList currfbcs = fbcs.sublist(var);
            Teuchos::ParameterList::ConstIterator f_itr = currfbcs.begin();
            while (f_itr != currfbcs.end()) {
              string entry = currfbcs.get<string>(f_itr->first);
              string label = "Far-field " + var + " " + f_itr->first;
              functionManagers[block]->addFunction(label,entry,"side ip");
              f_itr++;
            }
          }
        }
      }
    }
  }

  // Slip conditions
  for (size_t set=0; set<setnames.size(); set++) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      Teuchos::ParameterList sbcs = setPhysSettings[set][block].sublist("Slip conditions");
      for (size_t j=0; j<varlist[set][block].size(); j++) {
        string var = varlist[set][block][j];
        if (sbcs.isSublist(var)) {
          if (sbcs.sublist(var).isParameter("all boundaries")) {
            string entry = sbcs.sublist(var).get<string>("all boundaries");
            for (size_t s=0; s<sidenames.size(); s++) {
              string label = "Slip " + var + " " + sidenames[s];
              functionManagers[block]->addFunction(label,entry,"side ip");
            }
          }
          else {
            Teuchos::ParameterList currsbcs = sbcs.sublist(var);
            Teuchos::ParameterList::ConstIterator s_itr = currsbcs.begin();
            while (s_itr != currsbcs.end()) {
              string entry = currsbcs.get<string>(s_itr->first);
              string label = "Slip " + var + " " + s_itr->first;
              functionManagers[block]->addFunction(label,entry,"side ip");
              s_itr++;
            }
          }
        }
      }
    }
  }

  // Flux conditions
  for (size_t set=0; set<setnames.size(); set++) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      Teuchos::ParameterList fbcs = setPhysSettings[set][block].sublist("Flux conditions");
      for (size_t j=0; j<varlist[set][block].size(); j++) {
        string var = varlist[set][block][j];
        if (fbcs.isSublist(var)) {
          if (fbcs.sublist(var).isParameter("all boundaries")) {
            string entry = fbcs.sublist(var).get<string>("all boundaries");
            for (size_t s=0; s<sidenames.size(); s++) {
              string label = "Flux " + var + " " + sidenames[s];
              functionManagers[block]->addFunction(label,entry,"side ip");
            }
          }
          else {
            Teuchos::ParameterList currfbcs = fbcs.sublist(var);
            Teuchos::ParameterList::ConstIterator f_itr = currfbcs.begin();
            while (f_itr != currfbcs.end()) {
              string entry = currfbcs.get<string>(f_itr->first);
              string label = "Flux " + var + " " + f_itr->first;
              functionManagers[block]->addFunction(label,entry,"side ip");
              f_itr++;
            }
          }
        }
      }
    }
  }
  
  // Add mass scalings
  for (size_t set=0; set<setnames.size(); set++) {
    vector<vector<ScalarT> > setwts;
    for (size_t block=0; block<blocknames.size(); ++block) {
      Teuchos::ParameterList wts_list = setPhysSettings[set][block].sublist("Mass weights");
      vector<ScalarT> blockwts;
      for (size_t j=0; j<varlist[set][block].size(); j++) {
        ScalarT wval = 1.0;
        if (wts_list.isType<ScalarT>(varlist[set][block][j])) {
          wval = wts_list.get<ScalarT>(varlist[set][block][j]);
        }
        blockwts.push_back(wval);
      }
      setwts.push_back(blockwts);
    }
    masswts.push_back(setwts);
  }
 
  // Add norm weights
  for (size_t set=0; set<setnames.size(); set++) {
    vector<vector<ScalarT> > setwts;
    for (size_t block=0; block<blocknames.size(); ++block) {
      Teuchos::ParameterList nwts_list = setPhysSettings[set][block].sublist("Norm weights");
      vector<ScalarT> blockwts;
      for (size_t j=0; j<varlist[set][block].size(); j++) {
        ScalarT wval = 1.0;
        if (nwts_list.isType<ScalarT>(varlist[set][block][j])) {
          wval = nwts_list.get<ScalarT>(varlist[set][block][j]);
        }
        blockwts.push_back(wval);
      }
      setwts.push_back(blockwts);
    }
    normwts.push_back(setwts);
  }
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    Teuchos::ParameterList functions;
    if (settings->sublist("Functions").isSublist(blocknames[block])) {
      functions = settings->sublist("Functions").sublist(blocknames[block]);
    }
    else {
      functions = settings->sublist("Functions");
    }
    Teuchos::ParameterList::ConstIterator fnc_itr = functions.begin();
    while (fnc_itr != functions.end()) {
      string entry = functions.get<string>(fnc_itr->first);
      functionManagers[block]->addFunction(fnc_itr->first,entry,"ip");
      functionManagers[block]->addFunction(fnc_itr->first,entry,"side ip");
      functionManagers[block]->addFunction(fnc_itr->first,entry,"point");
      fnc_itr++;
    }
  }
    
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Add the requested physics modules, variables, discretization types
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::importPhysics() {
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting PhysicsInterface::importPhysics ..." << endl;
    }
  }
  
  //-----------------------------------------------------------------
  // Step 1: load the enabled modules
  //-----------------------------------------------------------------
  
  for (size_t set=0; set<setnames.size(); set++) { // physics sets
    
    vector<vector<int> > set_orders;
    vector<vector<string> > set_types;
    vector<vector<string> > set_varlist;
    vector<vector<int> > set_varowned;
    
    vector<vector<Teuchos::RCP<physicsbase> > > set_modules;
    vector<vector<bool> > set_useSubgrid, set_useDG;
    
    for (size_t block=0; block<blocknames.size(); ++block) { // element blocks
      vector<int> block_orders;
      vector<string> block_types;
      vector<string> block_varlist;
      vector<int> block_varowned;
      
      vector<Teuchos::RCP<physicsbase> > block_modules;
      vector<bool> block_useSubgrid, block_useDG;
      
      std::string var;
      std::string default_type = "HGRAD";
      string module_list = setPhysSettings[set][block].get<string>("modules","");
      
      vector<string> enabled_modules = this->breakupList(module_list, ", ");
      
      physicsImporter physimp = physicsImporter();
      setPhysSettings[set][block].set<int>("verbosity",settings->get<int>("verbosity",0));
      block_modules = physimp.import(enabled_modules, setPhysSettings[set][block],
                                     spaceDim, Commptr);
      
      set_modules.push_back(block_modules);
    }
    modules.push_back(set_modules);
  }
  
  //-----------------------------------------------------------------
  // Step 2: get the variable names, type, etc.
  //-----------------------------------------------------------------
  
  for (size_t set=0; set<setnames.size(); set++) { // physics sets
    
    vector<vector<int> > set_orders;
    vector<vector<string> > set_types;
    vector<vector<string> > set_varlist;
    vector<vector<int> > set_varowned;
    
    vector<vector<bool> > set_useDG;
    vector<size_t> set_numVars;
    
    for (size_t block=0; block<blocknames.size(); ++block) { // element blocks
      vector<int> block_orders;
      vector<string> block_types;
      vector<string> block_varlist;
      vector<int> block_varowned;
      vector<bool> block_useDG;
      
      for (size_t m=0; m<modules[set][block].size(); m++) {
        vector<string> cvars = modules[set][block][m]->myvars;
        vector<string> ctypes = modules[set][block][m]->mybasistypes;
        for (size_t v=0; v<cvars.size(); v++) {
          block_varlist.push_back(cvars[v]);
          string DGflag("-DG");
          size_t found = ctypes[v].find(DGflag);
          if (found!=string::npos) {
            string name = ctypes[v];
            name.erase(name.end()-3, name.end());
            block_types.push_back(name);
            block_useDG.push_back(true);
          }
          else {
            block_types.push_back(ctypes[v]);
            block_useDG.push_back(false);
          }
          
          block_varowned.push_back(m);
          block_orders.push_back(setDiscSettings[set][block].sublist("order").get<int>(cvars[v],1));
        }
      }
      
      Teuchos::ParameterList evars;
      bool have_extra_vars = false;
      if (setPhysSettings[set][block].isSublist("Extra variables")) {
        have_extra_vars = true;
        evars = setPhysSettings[set][block].sublist("Extra variables");
      }
      if (have_extra_vars) {
        Teuchos::ParameterList::ConstIterator pl_itr = evars.begin();
        while (pl_itr != evars.end()) {
          string newvar = pl_itr->first;
          block_varlist.push_back(newvar);
          
          string newtype = evars.get<string>(pl_itr->first);
          string DGflag("-DG");
          size_t found = newtype.find(DGflag);
          if (found!=string::npos) {
            string name = newtype;
            name.erase(name.end()-3, name.end());
            block_types.push_back(name);
            block_useDG.push_back(true);
          }
          else {
            block_types.push_back(newtype);
            block_useDG.push_back(false);
          }
          block_orders.push_back(setDiscSettings[set][block].sublist("order").get<int>(newvar,1));
          pl_itr++;
        }
      }     
      
      set_orders.push_back(block_orders);
      set_types.push_back(block_types);
      set_varlist.push_back(block_varlist);
      set_varowned.push_back(block_varowned);
      set_numVars.push_back(block_varlist.size());
      set_useDG.push_back(block_useDG);
      
    }
    orders.push_back(set_orders);
    types.push_back(set_types);
    varlist.push_back(set_varlist);
    varowned.push_back(set_varowned);
    numVars.push_back(set_numVars);
    useDG.push_back(set_useDG);
  }
    
  //-----------------------------------------------------------------
  // Step 3: get the unique information on each block
  //-----------------------------------------------------------------
  
  for (size_t block=0; block<blocknames.size(); ++block) { // element blocks

    std::vector<int> block_unique_orders;
    std::vector<string> block_unique_types;
    std::vector<int> block_unique_index;
    
    int currnumVars = 0;
    for (size_t set=0; set<setnames.size(); set++) { // physics sets
      currnumVars += varlist[set][block].size();
    }
    TEUCHOS_TEST_FOR_EXCEPTION(currnumVars==0,std::runtime_error,"Error: no variable were added on block: " + blocknames[block]);
    
    for (size_t set=0; set<setnames.size(); set++) { // physics sets
      for (size_t j=0; j<orders[set][block].size(); j++) {
        bool is_unique = true;
        for (size_t k=0; k<block_unique_orders.size(); k++) {
          if (block_unique_orders[k] == orders[set][block][j] && block_unique_types[k] == types[set][block][j]) {
            is_unique = false;
            block_unique_index.push_back(k);
          }
        }
        if (is_unique) {
          block_unique_orders.push_back(orders[set][block][j]);
          block_unique_types.push_back(types[set][block][j]);
          block_unique_index.push_back(block_unique_orders.size()-1);
        }
      }
    }
    
    
    // Discretized parameters currently get added on all blocks
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
      for (size_t k=0; k<block_unique_orders.size(); k++) {
        if (block_unique_orders[k] == discretized_param_basis_orders[j] && block_unique_types[k] == discretized_param_basis_types[j]) {
          is_unique = false;
        }
      }
      if (is_unique) {
        block_unique_orders.push_back(discretized_param_basis_orders[j]);
        block_unique_types.push_back(discretized_param_basis_types[j]);
      }
    }
    
    unique_orders.push_back(block_unique_orders);
    unique_types.push_back(block_unique_types);
    unique_index.push_back(block_unique_index);
    
  }
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished PhysicsInterface::importPhysics ..." << endl;
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<string> PhysicsInterface::breakupList(const string & list, const string & delimiter) {
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

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int PhysicsInterface::getvarOwner(const int & set, const int & block, const string & var) {
  int owner = 0;
  for (size_t k=0; k<varlist[set][block].size(); k++) {
    if (varlist[set][block][k] == var) {
      owner = varowned[set][block][k];
    }
  }
  return owner;
  
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

// TMW: this function is probably never used

AD PhysicsInterface::getDirichletValue(const int & block, const ScalarT & x, const ScalarT & y,
                                       const ScalarT & z, const ScalarT & t, const string & var,
                                       const string & gside, const bool & useadjoint,
                                       Teuchos::RCP<workset> & wkset) {
  
  // update point in wkset
  auto xpt = wkset->getScalarField("x point");
  Kokkos::deep_copy(xpt,x);
  
  auto ypt = wkset->getScalarField("y point");
  Kokkos::deep_copy(ypt,y);
  
  if (spaceDim == 3) {
    auto zpt = wkset->getScalarField("z point");
    Kokkos::deep_copy(zpt,z);
  }
  
  //wkset->setTime(t);
  
  // evaluate the response
  auto ddata = functionManagers[block]->evaluate("Dirichlet " + var + " " + gside,"point");
  
  return ddata(0,0);
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

ScalarT PhysicsInterface::getInitialValue(const int & block, const ScalarT & x, const ScalarT & y,
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

bool PhysicsInterface::checkFace(const size_t & set, const size_t & block){
  bool include_face = false;
  for (size_t i=0; i<modules[set][block].size(); i++) {
    bool cuseef = modules[set][block][i]->include_face;
    if (cuseef) {
      include_face = true;
    }
  }
  
  return include_face;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc3 PhysicsInterface::getInitial(vector<View_Sc2> & pts, const int & set, const int & block,
                                      const bool & project, Teuchos::RCP<workset> & wkset) {
  
  
  size_t currnumVars = varlist[set][block].size();
  
  View_Sc3 ivals;
  
  if (project) {
    
    ivals = View_Sc3("tmp ivals",pts[0].extent(0), currnumVars, pts[0].extent(1));
    
    // ip in wkset are set in cell::getInitial
    for (size_t n=0; n<varlist[set][block].size(); n++) {
  
      auto ivals_AD = functionManagers[block]->evaluate("initial " + varlist[set][block][n],"ip");
      auto cvals = subview( ivals, ALL(), n, ALL());
      //copy
      parallel_for("physics fill initial values",
                   RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_t i=0; i<cvals.extent(1); i++) {
#ifndef MrHyDE_NO_AD
          cvals(e,i) = ivals_AD(e,i).val();
#else
          cvals(e,i) = ivals_AD(e,i);
#endif
        }
      });
    }
  }
  else {
    // TMW: will not work on device yet
    
    size_type dim = wkset->dimension;
    size_type Nelem = pts[0].extent(0);
    size_type Npts = pts[0].extent(1);
    
    View_Sc2 ptx("ptx",Nelem,Npts), pty("pty",Nelem,Npts), ptz("ptz",Nelem,Npts);
    ptx = pts[0];
    
    wkset->isOnPoint = true;

    View_Sc2 x,y,z;
    x = wkset->getScalarField("x");
    if (dim > 1) {
      pty = pts[1];
      y = wkset->getScalarField("y");
    }
    if (dim > 2) {
      ptz = pts[2];
      z = wkset->getScalarField("z");
    }
    
    
    ivals = View_Sc3("tmp ivals",Nelem,currnumVars,Npts);
    for (size_t e=0; e<ptx.extent(0); e++) {
      for (size_t i=0; i<ptx.extent(1); i++) {
        // set the node in wkset
        int dim_ = spaceDim;
        parallel_for("physics initial set point",
                     RangePolicy<AssemblyExec>(0,1),
                     KOKKOS_LAMBDA (const int s ) {
          x(0,0) = ptx(e,i); // TMW: this might be ok
          if (dim_ > 1) {
            y(0,0) = pty(e,i);
          }
          if (dim_ > 2) {
            z(0,0) = ptz(e,i);
          }
          
        });
        
        for (size_t n=0; n<varlist[set][block].size(); n++) {
          // evaluate
          auto ivals_AD = functionManagers[block]->evaluate("initial " + varlist[set][block][n],"point");
        
          // Also might be ok (terribly inefficient though)
          parallel_for("physics initial set point",
                       RangePolicy<AssemblyExec>(0,1),
                       KOKKOS_LAMBDA (const int s ) {
#ifndef MrHyDE_NO_AD
            ivals(e,n,i) = ivals_AD(0,0).val();
#else
            ivals(e,n,i) = ivals_AD(0,0);
#endif
          });
          
        }
      }
    }
    wkset->isOnPoint = false;
  }
   
  return ivals;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc3 PhysicsInterface::getInitialFace(vector<View_Sc2> & pts, const int & set,
                                          const int & block, const bool & project, Teuchos::RCP<workset> & wkset) {
  
  size_t currnumVars = varlist[set][block].size();
  
  View_Sc3 ivals;
  
  if (project) {
    
    ivals = View_Sc3("tmp ivals",pts[0].extent(0), currnumVars, pts[0].extent(1));
    
    // ip in wkset are set in cell::getInitial
    for (size_t n=0; n<varlist[set][block].size(); n++) {

      auto ivals_AD = functionManagers[block]->evaluate("initial " + varlist[set][block][n],"side ip");
      auto cvals = subview( ivals, ALL(), n, ALL());
      //copy
      parallel_for("physics fill initial values",
                   RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_t i=0; i<cvals.extent(1); i++) {
#ifndef MrHyDE_NO_AD
          cvals(e,i) = ivals_AD(e,i).val();
#else
          cvals(e,i) = ivals_AD(e,i);
#endif
        }
      });
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(!project,std::runtime_error,"MyHyDE Error: HFACE variables need to use an L2-projection for the initial conditions");
  }
   
  return ivals;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

View_Sc2 PhysicsInterface::getDirichlet(const int & var,
                                        const int & set,
                                        const int & block,
                                        const std::string & sidename) {
  
  // evaluate
  
  auto dvals_AD = functionManagers[block]->evaluate("Dirichlet " + varlist[set][block][var] + " " + sidename,"side ip");
  
  //View_Sc2 dvals("temp dnvals", dvals_AD.extent(0), dvals_AD.extent(1));
  View_Sc2 dvals("temp dnvals", functionManagers[block]->numElem, functionManagers[block]->numip_side);
  
  // copy values
  parallel_for("physics fill Dirichlet values",
               RangePolicy<AssemblyExec>(0,dvals.extent(0)),
               KOKKOS_LAMBDA (const int e ) {
    for (size_t i=0; i<dvals.extent(1); i++) {
#ifndef MrHyDE_NO_AD
      dvals(e,i) = dvals_AD(e,i).val();
#else
      dvals(e,i) = dvals_AD(e,i);
#endif
    }
  });
  return dvals;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params,
                                        const vector<string> & paramnames) {
  
  for (size_t set=0; set<modules.size(); set++) {
    for (size_t block=0; block<modules[set].size(); ++block) {
      for (size_t i=0; i<modules[set][block].size(); i++) {
        modules[set][block][i]->updateParameters(params, paramnames);
      }
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int PhysicsInterface::getUniqueIndex(const int & set, const int & block, const std::string & var) {
  int index = 0;
  size_t prog = 0;
  for (size_t set=0; set<numVars.size(); ++set) {
    for (size_t j=0; j<numVars[set][block]; j++) {
      if (varlist[set][block][j] == var) {
        index = unique_index[block][j+prog];
      }
    }
    prog += numVars[set][block];
  }
  return index;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::volumeResidual(const size_t & set, const size_t block) {
  if (debug_level > 1 && Commptr->getRank() == 0) {
    cout << "**** Starting PhysicsInterface volume residual ..." << endl;
  }
  for (size_t i=0; i<modules[set][block].size(); i++) {
    modules[set][block][i]->volumeResidual();
  }
  if (debug_level > 1 && Commptr->getRank() == 0) {
    cout << "**** Finished PhysicsInterface volume residual" << endl;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::boundaryResidual(const size_t & set, const size_t block) {
  if (debug_level > 1 && Commptr->getRank() == 0) {
    cout << "**** Starting PhysicsInterface boundary residual ..." << endl;
  }
  for (size_t i=0; i<modules[set][block].size(); i++) {
    modules[set][block][i]->boundaryResidual();
  }
  if (debug_level > 1 && Commptr->getRank() == 0) {
    cout << "**** Finished PhysicsInterface boundary residual" << endl;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::computeFlux(const size_t & set, const size_t block) {
  if (debug_level > 1 && Commptr->getRank() == 0) {
    cout << "**** Starting PhysicsInterface compute flux ..." << endl;
  }
  for (size_t i=0; i<modules[set][block].size(); i++) {
    modules[set][block][i]->computeFlux();
  }
  if (debug_level > 1 && Commptr->getRank() == 0) {
    cout << "**** Finished PhysicsInterface compute flux" << endl;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<workset> > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules[set][block].size(); i++) {
          modules[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::faceResidual(const size_t & set, const size_t block) {
  for (size_t i=0; i<modules[set][block].size(); i++) {
    modules[set][block][i]->faceResidual();
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::fluxConditions(const size_t & set, const size_t block) {
  for (size_t var=0; var<varlist[set][block].size(); ++var) {
    int cside = functionManagers[block]->wkset->currentside;
    string bctype = functionManagers[block]->wkset->var_bcs(var,cside);
    if (bctype == "Flux") {
      string varname = varlist[set][block][var];
      string sidename = functionManagers[block]->wkset->sidename;
      string label = "Flux " + varname + " " + sidename;
      auto fluxvals = functionManagers[block]->evaluate(label,"side ip");
      
      auto basis = functionManagers[block]->wkset->getBasisSide(varname);
      auto wts = functionManagers[block]->wkset->wts_side;
      auto res = functionManagers[block]->wkset->res;
      auto off = functionManagers[block]->wkset->getOffsets(varname);
      
      
      parallel_for("physics flux condition",
                   TeamPolicy<AssemblyExec>(wts.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += -fluxvals(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
          }
        }
      });
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
    
void PhysicsInterface::purgeMemory() {
  // nothing here
}
    