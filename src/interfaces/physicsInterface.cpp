/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "physicsInterface.hpp"
#include "physicsImporter.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

PhysicsInterface::PhysicsInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                   Teuchos::RCP<MpiComm> & comm_,
                                   std::vector<string> block_names_,
                                   std::vector<string> side_names_,
                                   int dimension_) :
settings(settings_), comm(comm_), dimension(dimension_), block_names(block_names_), side_names(side_names_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), comm));
  
  debugger->print("**** Starting PhysicsInterface constructor ...");
  
  type_AD = 0; // Will get redefined later
  
  if (settings->sublist("Physics").isParameter("physics set names")) {
    string names = settings->sublist("Physics").get<string>("physics set names");
    set_names = this->breakupList(names,", ");
  }
  else {
    set_names.push_back("default");
  }
  
  for (size_t set=0; set<set_names.size(); ++set) {
    Teuchos::ParameterList psetlist, dsetlist, ssetlist;
    if (set_names[set] == "default") {
      psetlist = settings->sublist("Physics");
      dsetlist = settings->sublist("Discretization");
      ssetlist = settings->sublist("Solver");
    }
    else {
      psetlist = settings->sublist("Physics").sublist(set_names[set]);
      dsetlist = settings->sublist("Discretization").sublist(set_names[set]);
      ssetlist = settings->sublist("Solver").sublist(set_names[set]);
    }
    
    vector<Teuchos::ParameterList> currpsettings, currdsettings, currssettings;
    for (size_t block=0; block<block_names.size(); ++block) {
      if (psetlist.isSublist(block_names[block])) { // adding block overwrites the default
        currpsettings.push_back(psetlist.sublist(block_names[block]));
      }
      else { // default
        currpsettings.push_back(psetlist);
      }
      
      if (dsetlist.isSublist(block_names[block])) { // adding block overwrites default
        currdsettings.push_back(dsetlist.sublist(block_names[block]));
      }
      else { // default
        currdsettings.push_back(dsetlist);
      }

      if (ssetlist.isSublist(block_names[block])) { // adding block overwrites default
        currssettings.push_back(ssetlist.sublist(block_names[block]));
      }
      else { // default
        currssettings.push_back(ssetlist);
      }
    }
    physics_settings.push_back(currpsettings);
    disc_settings.push_back(currdsettings);
    solver_settings.push_back(currssettings);
  }
    
  this->importPhysics();
  
  debugger->print("**** Finished physics constructor");
  
}


/////////////////////////////////////////////////////////////////////////////////////////////
// Add the functions to the function managers
// Function managers are set up in the assembly manager
/////////////////////////////////////////////////////////////////////////////////////////////

// Externally called functions that need to be specialized for each evaluation type
// In other words, templating only saves space in the header

///////////////////////////////////////////////////
// ScalarT and AD are (almost) always defined
///////////////////////////////////////////////////

// Avoid duplication when AD=ScalarT
#ifndef MrHyDE_NO_AD
template<>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<AD> > > & function_managers_) {
  function_managers_AD = function_managers_;
  this->defineFunctions(function_managers_AD, modules_AD);
}
#endif

template<>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<ScalarT> > > & function_managers_) {
  function_managers = function_managers_;
  this->defineFunctions(function_managers, modules);
}

///////////////////////////////////////////////////
// The rest are used if needed
///////////////////////////////////////////////////

#ifndef MrHyDE_NO_AD
template<>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<AD2> > > & function_managers_) {
  function_managers_AD2 = function_managers_;
  this->defineFunctions(function_managers_AD2, modules_AD2);
}

template<>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<AD4> > > & function_managers_) {
  function_managers_AD4 = function_managers_;
  this->defineFunctions(function_managers_AD4, modules_AD4);
}

template<>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<AD8> > > & function_managers_) {
  function_managers_AD8 = function_managers_;
  this->defineFunctions(function_managers_AD8, modules_AD8);
}

template<>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<AD16> > > & function_managers_) {
  function_managers_AD16 = function_managers_;
  this->defineFunctions(function_managers_AD16, modules_AD16);
}

template<>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<AD18> > > & function_managers_) {
  function_managers_AD18 = function_managers_;
  this->defineFunctions(function_managers_AD18, modules_AD18);
}

template<>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<AD24> > > & function_managers_) {
  function_managers_AD24 = function_managers_;
  this->defineFunctions(function_managers_AD24, modules_AD24);
}

template<>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<AD32> > > & function_managers_) {
  function_managers_AD32 = function_managers_;
  this->defineFunctions(function_managers_AD32, modules_AD32);
}
#endif

///////////////////////////////////////////////////
// Main define functions routine that actually does work
///////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::defineFunctions(vector<Teuchos::RCP<FunctionManager<EvalT> > > & func_managers,
                                       vector<vector<vector<Teuchos::RCP<PhysicsBase<EvalT> > > > > & mods) {

  
  for (size_t block=0; block<block_names.size(); ++block) {
    Teuchos::ParameterList fs;
    if (settings->sublist("Functions").isSublist(block_names[block])) {
      fs = settings->sublist("Functions").sublist(block_names[block]);
    }
    else {
      fs = settings->sublist("Functions");
    }
    
    for (size_t set=0; set<mods.size(); set++) {
      for (size_t n=0; n<mods[set][block].size(); n++) {
        mods[set][block][n]->defineFunctions(fs, func_managers[block]);
      }
    }
  }
  
  // Add initial conditions
  for (size_t set=0; set<set_names.size(); set++) {
    for (size_t block=0; block<block_names.size(); ++block) {
      if (physics_settings[set][block].isSublist("Initial conditions")) {
        Teuchos::ParameterList initial_conds = physics_settings[set][block].sublist("Initial conditions");
        for (size_t j=0; j<var_list[set][block].size(); j++) {
          string var = var_list[set][block][j];
          if (types[set][block][j].substr(0,5) == "HGRAD" || types[set][block][j].substr(0,4) == "HVOL" || types[set][block][j].substr(0,5) == "HFACE") {
            string expression;
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
            func_managers[block]->addFunction("initial "+var,expression,"ip");
            func_managers[block]->addFunction("initial "+var,expression,"point");
            if (types[set][block][j] == "HFACE") {
              // we have found an HFACE variable and need to have side ip evaluations
              // TODO check aux, etc?
              func_managers[block]->addFunction("initial "+var,expression,"side ip");
            }
          }
          else if (types[set][block][j].substr(0,5) == "HCURL" || types[set][block][j].substr(0,4) == "HDIV") {
            string expressionx, expressiony, expressionz;          
            if (initial_conds.isType<string>(var+"[x]")) {
              expressionx = initial_conds.get<string>(var+"[x]");
            }
            else if (initial_conds.isType<double>(var+"[x]")) {
              double value = initial_conds.get<double>(var+"[x]");
              expressionx = std::to_string(value);
            }
            else {
              expressionx = "0.0";
            }
            func_managers[block]->addFunction("initial "+var+"[x]",expressionx,"ip");
            func_managers[block]->addFunction("initial "+var+"[x]",expressionx,"point");
          
            if (initial_conds.isType<string>(var+"[y]")) {
              expressiony = initial_conds.get<string>(var+"[y]");
            }
            else if (initial_conds.isType<double>(var+"[y]")) {
              double value = initial_conds.get<double>(var+"[y]");
              expressiony = std::to_string(value);
            }
            else {
              expressiony = "0.0";
            }
            func_managers[block]->addFunction("initial "+var+"[y]",expressiony,"ip");
            func_managers[block]->addFunction("initial "+var+"[y]",expressiony,"point");
          
            if (initial_conds.isType<string>(var+"[z]")) {
              expressionz = initial_conds.get<string>(var+"[z]");
            }
            else if (initial_conds.isType<double>(var+"[z]")) {
              double value = initial_conds.get<double>(var+"[z]");
              expressionz = std::to_string(value);
            }
            else {
              expressionz = "0.0";
            }
            func_managers[block]->addFunction("initial "+var+"[z]",expressionz,"ip");
            func_managers[block]->addFunction("initial "+var+"[z]",expressionz,"point");
          
          }
          
        }
      }
    }
  }
    
  // Dirichlet conditions
  for (size_t set=0; set<set_names.size(); set++) {
    for (size_t block=0; block<block_names.size(); ++block) {
      if (physics_settings[set][block].isSublist("Dirichlet conditions")) {
        Teuchos::ParameterList dbcs = physics_settings[set][block].sublist("Dirichlet conditions");
        for (size_t j=0; j<var_list[set][block].size(); j++) {
          string var = var_list[set][block][j];
          bool is_vector_type = (types[set][block][j].substr(0,5) == "HCURL" || types[set][block][j].substr(0,4) == "HDIV");
          
          // check for scalar variable entry (e.g., "E:")
          if (dbcs.isSublist(var)) {
            if (dbcs.sublist(var).isType<string>("all boundaries")) {
              string entry = dbcs.sublist(var).get<string>("all boundaries");
              for (size_t s=0; s<side_names.size(); s++) {
                string label = "Dirichlet " + var + " " + side_names[s];
                func_managers[block]->addFunction(label,entry,"side ip");
              }
            }
            else if (dbcs.sublist(var).isType<double>("all boundaries")) {
              double value = dbcs.sublist(var).get<double>("all boundaries");
              string entry = std::to_string(value);
              for (size_t s=0; s<side_names.size(); s++) {
                string label = "Dirichlet " + var + " " + side_names[s];
                func_managers[block]->addFunction(label,entry,"side ip");
              }
            }
            else {
              Teuchos::ParameterList currdbcs = dbcs.sublist(var);
              Teuchos::ParameterList::ConstIterator d_itr = currdbcs.begin();
              while (d_itr != currdbcs.end()) {
                if (currdbcs.isType<string>(d_itr->first)) {
                  string entry = currdbcs.get<string>(d_itr->first);
                  string label = "Dirichlet " + var + " " + d_itr->first;
                  func_managers[block]->addFunction(label,entry,"side ip");
                }
                else if (currdbcs.isType<double>(d_itr->first)) {
                  double value = currdbcs.get<double>(d_itr->first);
                  string entry = std::to_string(value);
                  string label = "Dirichlet " + var + " " + d_itr->first;
                  func_managers[block]->addFunction(label,entry,"side ip");
                }
                d_itr++;
              }
            }
          }
          
          // check for vector component entries for HCURL/HDIV variables (e.g., "Ex:", "Ey:", "Ez:")
          if (is_vector_type) {
            std::vector<string> components = {"x", "y", "z"};
            for (const auto& comp : components) {
              string var_comp = var + comp;
              if (dbcs.isSublist(var_comp)) {
                if (dbcs.sublist(var_comp).isType<string>("all boundaries")) {
                  string entry = dbcs.sublist(var_comp).get<string>("all boundaries");
                  for (size_t s=0; s<side_names.size(); s++) {
                    string label = "Dirichlet " + var_comp + " " + side_names[s];
                    func_managers[block]->addFunction(label,entry,"side ip");
                  }
                }
                else if (dbcs.sublist(var_comp).isType<double>("all boundaries")) {
                  double value = dbcs.sublist(var_comp).get<double>("all boundaries");
                  string entry = std::to_string(value);
                  for (size_t s=0; s<side_names.size(); s++) {
                    string label = "Dirichlet " + var_comp + " " + side_names[s];
                    func_managers[block]->addFunction(label,entry,"side ip");
                  }
                }
                else {
                  Teuchos::ParameterList currdbcs = dbcs.sublist(var_comp);
                  Teuchos::ParameterList::ConstIterator d_itr = currdbcs.begin();
                  while (d_itr != currdbcs.end()) {
                    if (currdbcs.isType<string>(d_itr->first)) {
                      string entry = currdbcs.get<string>(d_itr->first);
                      string label = "Dirichlet " + var_comp + " " + d_itr->first;
                      func_managers[block]->addFunction(label,entry,"side ip");
                    }
                    else if (currdbcs.isType<double>(d_itr->first)) {
                      double value = currdbcs.get<double>(d_itr->first);
                      string entry = std::to_string(value);
                      string label = "Dirichlet " + var_comp + " " + d_itr->first;
                      func_managers[block]->addFunction(label,entry,"side ip");
                    }
                    d_itr++;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Neumann/robin conditions
  for (size_t set=0; set<set_names.size(); set++) {
    for (size_t block=0; block<block_names.size(); ++block) {
      Teuchos::ParameterList nbcs = physics_settings[set][block].sublist("Neumann conditions");
      for (size_t j=0; j<var_list[set][block].size(); j++) {
        string var = var_list[set][block][j];
        if (nbcs.isSublist(var)) {
          if (nbcs.sublist(var).isParameter("all boundaries")) {
            string entry = nbcs.sublist(var).get<string>("all boundaries");
            for (size_t s=0; s<side_names.size(); s++) {
              string label = "Neumann " + var + " " + side_names[s];
              func_managers[block]->addFunction(label,entry,"side ip");
            }
          }
          else {
            Teuchos::ParameterList currnbcs = nbcs.sublist(var);
            Teuchos::ParameterList::ConstIterator n_itr = currnbcs.begin();
            while (n_itr != currnbcs.end()) {
              string entry = currnbcs.get<string>(n_itr->first);
              string label = "Neumann " + var + " " + n_itr->first;
              func_managers[block]->addFunction(label,entry,"side ip");
              n_itr++;
            }
          }
        }
      }
    }
  }

  // Far-field conditions
  for (size_t set=0; set<set_names.size(); set++) {
    for (size_t block=0; block<block_names.size(); ++block) {
      Teuchos::ParameterList fbcs = physics_settings[set][block].sublist("Far-field conditions");
      for (size_t j=0; j<var_list[set][block].size(); j++) {
        string var = var_list[set][block][j];
        if (fbcs.isSublist(var)) {
          if (fbcs.sublist(var).isParameter("all boundaries")) {
            string entry = fbcs.sublist(var).get<string>("all boundaries");
            for (size_t s=0; s<side_names.size(); s++) {
              string label = "Far-field " + var + " " + side_names[s];
              func_managers[block]->addFunction(label,entry,"side ip");
            }
          }
          else {
            Teuchos::ParameterList currfbcs = fbcs.sublist(var);
            Teuchos::ParameterList::ConstIterator f_itr = currfbcs.begin();
            while (f_itr != currfbcs.end()) {
              string entry = currfbcs.get<string>(f_itr->first);
              string label = "Far-field " + var + " " + f_itr->first;
              func_managers[block]->addFunction(label,entry,"side ip");
              f_itr++;
            }
          }
        }
      }
    }
  }

  // Slip conditions
  for (size_t set=0; set<set_names.size(); set++) {
    for (size_t block=0; block<block_names.size(); ++block) {
      Teuchos::ParameterList sbcs = physics_settings[set][block].sublist("Slip conditions");
      for (size_t j=0; j<var_list[set][block].size(); j++) {
        string var = var_list[set][block][j];
        if (sbcs.isSublist(var)) {
          if (sbcs.sublist(var).isParameter("all boundaries")) {
            string entry = sbcs.sublist(var).get<string>("all boundaries");
            for (size_t s=0; s<side_names.size(); s++) {
              string label = "Slip " + var + " " + side_names[s];
              func_managers[block]->addFunction(label,entry,"side ip");
            }
          }
          else {
            Teuchos::ParameterList currsbcs = sbcs.sublist(var);
            Teuchos::ParameterList::ConstIterator s_itr = currsbcs.begin();
            while (s_itr != currsbcs.end()) {
              string entry = currsbcs.get<string>(s_itr->first);
              string label = "Slip " + var + " " + s_itr->first;
              func_managers[block]->addFunction(label,entry,"side ip");
              s_itr++;
            }
          }
        }
      }
    }
  }

  // Flux conditions
  for (size_t set=0; set<set_names.size(); set++) {
    for (size_t block=0; block<block_names.size(); ++block) {
      Teuchos::ParameterList fbcs = physics_settings[set][block].sublist("Flux conditions");
      for (size_t j=0; j<var_list[set][block].size(); j++) {
        string var = var_list[set][block][j];
        if (fbcs.isSublist(var)) {
          if (fbcs.sublist(var).isParameter("all boundaries")) {
            string entry = fbcs.sublist(var).get<string>("all boundaries");
            for (size_t s=0; s<side_names.size(); s++) {
              string label = "Flux " + var + " " + side_names[s];
              func_managers[block]->addFunction(label,entry,"side ip");
            }
          }
          else {
            Teuchos::ParameterList currfbcs = fbcs.sublist(var);
            Teuchos::ParameterList::ConstIterator f_itr = currfbcs.begin();
            while (f_itr != currfbcs.end()) {
              string entry = currfbcs.get<string>(f_itr->first);
              string label = "Flux " + var + " " + f_itr->first;
              func_managers[block]->addFunction(label,entry,"side ip");
              f_itr++;
            }
          }
        }
      }
    }
  }
  
  // Add mass scalings
  for (size_t set=0; set<set_names.size(); set++) {
    vector<vector<ScalarT> > setwts;
    for (size_t block=0; block<block_names.size(); ++block) {
      Teuchos::ParameterList wts_list = physics_settings[set][block].sublist("Mass weights");
      vector<ScalarT> blockwts;
      for (size_t j=0; j<var_list[set][block].size(); j++) {
        ScalarT wval = 1.0;
        if (wts_list.isType<ScalarT>(var_list[set][block][j])) {
          wval = wts_list.get<ScalarT>(var_list[set][block][j]);
        }
        blockwts.push_back(wval);
      }
      setwts.push_back(blockwts);
    }
    mass_wts.push_back(setwts);
  }
 
  // Add norm weights
  for (size_t set=0; set<set_names.size(); set++) {
    vector<vector<ScalarT> > setwts;
    for (size_t block=0; block<block_names.size(); ++block) {
      Teuchos::ParameterList nwts_list = physics_settings[set][block].sublist("Norm weights");
      vector<ScalarT> blockwts;
      for (size_t j=0; j<var_list[set][block].size(); j++) {
        ScalarT wval = 1.0;
        if (nwts_list.isType<ScalarT>(var_list[set][block][j])) {
          wval = nwts_list.get<ScalarT>(var_list[set][block][j]);
        }
        blockwts.push_back(wval);
      }
      setwts.push_back(blockwts);
    }
    norm_wts.push_back(setwts);
  }
  
  for (size_t block=0; block<block_names.size(); ++block) {
    Teuchos::ParameterList functions;
    if (settings->sublist("Functions").isSublist(block_names[block])) {
      functions = settings->sublist("Functions").sublist(block_names[block]);
    }
    else {
      functions = settings->sublist("Functions");
    }
    Teuchos::ParameterList::ConstIterator fnc_itr = functions.begin();
    while (fnc_itr != functions.end()) {
      string entry = functions.get<string>(fnc_itr->first);
      func_managers[block]->addFunction(fnc_itr->first,entry,"ip");
      func_managers[block]->addFunction(fnc_itr->first,entry,"side ip");
      func_managers[block]->addFunction(fnc_itr->first,entry,"point");
      fnc_itr++;
    }
  }
    
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Add the requested physics modules, variables, discretization types
/////////////////////////////////////////////////////////////////////////////////////////////

void PhysicsInterface::importPhysics() {
  
  debugger->print("**** Starting PhysicsInterface::importPhysics ...");
  
  //-----------------------------------------------------------------
  // Step 1: load the enabled modules
  //-----------------------------------------------------------------
  
  for (size_t set=0; set<set_names.size(); set++) { // physics sets
    
    vector<vector<int> > set_orders;
    vector<vector<string> > set_types;
    vector<vector<string> > set_var_list;
    vector<vector<int> > set_var_owned;
    
    vector<vector<Teuchos::RCP<PhysicsBase<ScalarT> > > > set_modules;

    vector<vector<bool> > set_use_subgrid, set_use_DG;
    
    for (size_t block=0; block<block_names.size(); ++block) { // element blocks
      vector<int> block_orders;
      vector<string> block_types;
      vector<string> block_var_list;
      vector<int> block_var_owned;
      
      vector<bool> block_use_subgrid, block_use_DG;
      
      std::string var;
      std::string default_type = "HGRAD";
      string module_list = physics_settings[set][block].get<string>("modules","");
      
      vector<string> enabled_modules = this->breakupList(module_list, ", ");
      
      physics_settings[set][block].set<int>("verbosity",settings->get<int>("verbosity",0));
      
      { 
        vector<Teuchos::RCP<PhysicsBase<ScalarT> > > block_modules;
        PhysicsImporter<ScalarT> physimp = PhysicsImporter<ScalarT>();
        block_modules = physimp.import(enabled_modules, physics_settings[set][block],
                                       dimension, comm);
      
        set_modules.push_back(block_modules);
      }
    }
    modules.push_back(set_modules);
  }
  
  //-----------------------------------------------------------------
  // Step 2: get the variable names, type, etc.
  //-----------------------------------------------------------------
  
  for (size_t set=0; set<set_names.size(); set++) { // physics sets
    
    vector<vector<int> > set_orders;
    vector<vector<string> > set_types;
    vector<vector<string> > set_var_list;
    vector<vector<int> > set_var_owned;
    
    vector<vector<bool> > set_use_DG;
    vector<size_t> set_num_vars;
    
    for (size_t block=0; block<block_names.size(); ++block) { // element blocks
      vector<int> block_orders;
      vector<string> block_types;
      vector<string> block_var_list;
      vector<int> block_var_owned;
      vector<bool> block_use_DG;
      
      for (size_t m=0; m<modules[set][block].size(); m++) {
        vector<string> cvars = modules[set][block][m]->myvars;
        vector<string> ctypes = modules[set][block][m]->mybasistypes;
        for (size_t v=0; v<cvars.size(); v++) {
          block_var_list.push_back(cvars[v]);
          string DGflag("-DG");
          size_t found = ctypes[v].find(DGflag);
          if (found!=string::npos) {
            string name = ctypes[v];
            name.erase(name.end()-3, name.end());
            block_types.push_back(name);
            block_use_DG.push_back(true);
          }
          else {
            block_types.push_back(ctypes[v]);
            block_use_DG.push_back(false);
          }
          
          block_var_owned.push_back(m);
          block_orders.push_back(disc_settings[set][block].sublist("order").get<int>(cvars[v],1));
        }
      }
      
      Teuchos::ParameterList evars;
      bool have_extra_vars = false;
      if (physics_settings[set][block].isSublist("Extra variables")) {
        have_extra_vars = true;
        evars = physics_settings[set][block].sublist("Extra variables");
      }
      if (have_extra_vars) {
        Teuchos::ParameterList::ConstIterator pl_itr = evars.begin();
        while (pl_itr != evars.end()) {
          string newvar = pl_itr->first;
          block_var_list.push_back(newvar);
          
          string newtype = evars.get<string>(pl_itr->first);
          string DGflag("-DG");
          size_t found = newtype.find(DGflag);
          if (found!=string::npos) {
            string name = newtype;
            name.erase(name.end()-3, name.end());
            block_types.push_back(name);
            block_use_DG.push_back(true);
          }
          else {
            block_types.push_back(newtype);
            block_use_DG.push_back(false);
          }
          block_orders.push_back(disc_settings[set][block].sublist("order").get<int>(newvar,1));
          pl_itr++;
        }
      }     
      
      set_orders.push_back(block_orders);
      set_types.push_back(block_types);
      set_var_list.push_back(block_var_list);
      set_var_owned.push_back(block_var_owned);
      set_num_vars.push_back(block_var_list.size());
      set_use_DG.push_back(block_use_DG);
      
    }
    orders.push_back(set_orders);
    types.push_back(set_types);
    var_list.push_back(set_var_list);
    var_owned.push_back(set_var_owned);
    num_vars.push_back(set_num_vars);
    use_DG.push_back(set_use_DG);
  }
    
  //-----------------------------------------------------------------
  // Step 3: get the unique information on each block
  //-----------------------------------------------------------------
  
  for (size_t block=0; block<block_names.size(); ++block) { // element blocks

    std::vector<int> block_unique_orders;
    std::vector<string> block_unique_types;
    std::vector<int> block_unique_index;
    
    int currnum_vars = 0;
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
      currnum_vars += var_list[set][block].size();
    }
    TEUCHOS_TEST_FOR_EXCEPTION(currnum_vars==0,std::runtime_error,"Error: no variable were added on block: " + block_names[block]);
    
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
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
  
  debugger->print("**** Finished PhysicsInterface::importPhysics ...");
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
void PhysicsInterface::importPhysicsAD(int & type_AD_) {

  type_AD = type_AD_;
  
#ifndef MrHyDE_NO_AD

  if (type_AD == -1) {
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
      vector<vector<Teuchos::RCP<PhysicsBase<AD> > > > set_modules_AD;
      for (size_t block=0; block<block_names.size(); ++block) { // element blocks
        string module_list = physics_settings[set][block].get<string>("modules","");
        vector<string> enabled_modules = this->breakupList(module_list, ", ");
        
        vector<Teuchos::RCP<PhysicsBase<AD> > > block_modules_AD;
        PhysicsImporter<AD> physimp = PhysicsImporter<AD>();
        block_modules_AD = physimp.import(enabled_modules, physics_settings[set][block],
                                          dimension, comm);
        
        set_modules_AD.push_back(block_modules_AD);
        
      }
      modules_AD.push_back(set_modules_AD);
    }
  }
  else if (type_AD == 2) {
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
      vector<vector<Teuchos::RCP<PhysicsBase<AD2> > > > set_modules_AD;
      for (size_t block=0; block<block_names.size(); ++block) { // element blocks
        string module_list = physics_settings[set][block].get<string>("modules","");
        vector<string> enabled_modules = this->breakupList(module_list, ", ");
        
        vector<Teuchos::RCP<PhysicsBase<AD2> > > block_modules_AD;
        PhysicsImporter<AD2> physimp = PhysicsImporter<AD2>();
        block_modules_AD = physimp.import(enabled_modules, physics_settings[set][block],
                                          dimension, comm);
        
        set_modules_AD.push_back(block_modules_AD);
        
      }
      modules_AD2.push_back(set_modules_AD);
    }
  }
  else if (type_AD == 4) {
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
      vector<vector<Teuchos::RCP<PhysicsBase<AD4> > > > set_modules_AD;
      for (size_t block=0; block<block_names.size(); ++block) { // element blocks
        string module_list = physics_settings[set][block].get<string>("modules","");
        vector<string> enabled_modules = this->breakupList(module_list, ", ");
        
        vector<Teuchos::RCP<PhysicsBase<AD4> > > block_modules_AD;
        PhysicsImporter<AD4> physimp = PhysicsImporter<AD4>();
        block_modules_AD = physimp.import(enabled_modules, physics_settings[set][block],
                                          dimension, comm);
        
        set_modules_AD.push_back(block_modules_AD);
        
      }
      modules_AD4.push_back(set_modules_AD);
    }
  }
  else if (type_AD == 8) {
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
      vector<vector<Teuchos::RCP<PhysicsBase<AD8> > > > set_modules_AD;
      for (size_t block=0; block<block_names.size(); ++block) { // element blocks
        string module_list = physics_settings[set][block].get<string>("modules","");
        vector<string> enabled_modules = this->breakupList(module_list, ", ");
        
        vector<Teuchos::RCP<PhysicsBase<AD8> > > block_modules_AD;
        PhysicsImporter<AD8> physimp = PhysicsImporter<AD8>();
        block_modules_AD = physimp.import(enabled_modules, physics_settings[set][block],
                                          dimension, comm);
        
        set_modules_AD.push_back(block_modules_AD);
        
      }
      modules_AD8.push_back(set_modules_AD);
    }
  }
  else if (type_AD == 16) {
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
      vector<vector<Teuchos::RCP<PhysicsBase<AD16> > > > set_modules_AD;
      for (size_t block=0; block<block_names.size(); ++block) { // element blocks
        string module_list = physics_settings[set][block].get<string>("modules","");
        vector<string> enabled_modules = this->breakupList(module_list, ", ");
        
        vector<Teuchos::RCP<PhysicsBase<AD16> > > block_modules_AD;
        PhysicsImporter<AD16> physimp = PhysicsImporter<AD16>();
        block_modules_AD = physimp.import(enabled_modules, physics_settings[set][block],
                                          dimension, comm);
        
        set_modules_AD.push_back(block_modules_AD);
        
      }
      modules_AD16.push_back(set_modules_AD);
    }
  }
  else if (type_AD == 18) {
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
      vector<vector<Teuchos::RCP<PhysicsBase<AD18> > > > set_modules_AD;
      for (size_t block=0; block<block_names.size(); ++block) { // element blocks
        string module_list = physics_settings[set][block].get<string>("modules","");
        vector<string> enabled_modules = this->breakupList(module_list, ", ");
        
        vector<Teuchos::RCP<PhysicsBase<AD18> > > block_modules_AD;
        PhysicsImporter<AD18> physimp = PhysicsImporter<AD18>();
        block_modules_AD = physimp.import(enabled_modules, physics_settings[set][block],
                                          dimension, comm);
        
        set_modules_AD.push_back(block_modules_AD);
        
      }
      modules_AD18.push_back(set_modules_AD);
    }
  }
  else if (type_AD == 24) {
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
      vector<vector<Teuchos::RCP<PhysicsBase<AD24> > > > set_modules_AD;
      for (size_t block=0; block<block_names.size(); ++block) { // element blocks
        string module_list = physics_settings[set][block].get<string>("modules","");
        vector<string> enabled_modules = this->breakupList(module_list, ", ");
        
        vector<Teuchos::RCP<PhysicsBase<AD24> > > block_modules_AD;
        PhysicsImporter<AD24> physimp = PhysicsImporter<AD24>();
        block_modules_AD = physimp.import(enabled_modules, physics_settings[set][block],
                                          dimension, comm);
        
        set_modules_AD.push_back(block_modules_AD);
        
      }
      modules_AD24.push_back(set_modules_AD);
    }
  }
  else if (type_AD == 32) {
    for (size_t set=0; set<set_names.size(); set++) { // physics sets
      vector<vector<Teuchos::RCP<PhysicsBase<AD32> > > > set_modules_AD;
      for (size_t block=0; block<block_names.size(); ++block) { // element blocks
        string module_list = physics_settings[set][block].get<string>("modules","");
        vector<string> enabled_modules = this->breakupList(module_list, ", ");
        
        vector<Teuchos::RCP<PhysicsBase<AD32> > > block_modules_AD;
        PhysicsImporter<AD32> physimp = PhysicsImporter<AD32>();
        block_modules_AD = physimp.import(enabled_modules, physics_settings[set][block],
                                          dimension, comm);
        
        set_modules_AD.push_back(block_modules_AD);
        
      }
      modules_AD32.push_back(set_modules_AD);
    }
  }
  
#endif

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
  for (size_t k=0; k<var_list[set][block].size(); k++) {
    if (var_list[set][block][k] == var) {
      owner = var_owned[set][block][k];
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
                                       Teuchos::RCP<Workset<AD> > & wkset) {
  
  // update point in wkset
  auto xpt = wkset->getScalarField("x point");
  Kokkos::deep_copy(xpt,x);
  
  auto ypt = wkset->getScalarField("y point");
  Kokkos::deep_copy(ypt,y);
  
  if (dimension == 3) {
    auto zpt = wkset->getScalarField("z point");
    Kokkos::deep_copy(zpt,z);
  }
  
  //wkset->setTime(t);
  
  // evaluate the response
#ifndef MrHyDE_NO_AD
  auto ddata = function_managers_AD[block]->evaluate("Dirichlet " + var + " " + gside,"point");
  return ddata(0,0);
#else
  return 0.0;
#endif

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
  View_AD2_sv idata = function_manager->evaluate("initial " + var,"point",block);
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

View_Sc4 PhysicsInterface::getInitial(vector<View_Sc2> & pts, const int & set, const int & block,
                                      const bool & project, Teuchos::RCP<Workset<ScalarT> > & wkset) {
  
  
  size_t currnum_vars = var_list[set][block].size();
  
  View_Sc4 ivals;
  
  if (project) {
    
    ivals = View_Sc4("tmp ivals",pts[0].extent(0), currnum_vars, pts[0].extent(1),dimension);
    
    // ip in wkset are set in cell::getInitial
    for (size_t n=0; n<var_list[set][block].size(); n++) {
      if (types[set][block][n].substr(0,5) == "HGRAD" || types[set][block][n].substr(0,4) == "HVOL") {
        auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n],"ip");
        auto cvals = subview( ivals, ALL(), n, ALL(), 0);
        //copy
        parallel_for("physics fill initial values",
                     RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_t i=0; i<cvals.extent(1); i++) {
            cvals(e,i) = tivals(e,i);
          }
        });
      }
      else if (types[set][block][n].substr(0,5) == "HCURL" || types[set][block][n].substr(0,4) == "HDIV") {
        auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n] + "[x]","ip");
        auto cvals = subview( ivals, ALL(), n, ALL(), 0);
        //copy
        parallel_for("physics fill initial values",
                     RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_t i=0; i<cvals.extent(1); i++) {
            cvals(e,i) = tivals(e,i);
          }
        });
        if (dimension > 1) {
          auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n] + "[y]","ip");
          auto cvals = subview( ivals, ALL(), n, ALL(), 1);
          //copy
          parallel_for("physics fill initial values",
                       RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_t i=0; i<cvals.extent(1); i++) {
              cvals(e,i) = tivals(e,i);
            }
          });
        }
        if (dimension>2) {
          auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n] + "[z]","ip");
          auto cvals = subview( ivals, ALL(), n, ALL(), 2);
          //copy
          parallel_for("physics fill initial values",
                       RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_t i=0; i<cvals.extent(1); i++) {
              cvals(e,i) = tivals(e,i);
            }
          });
        }
      }
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
    
    
    ivals = View_Sc4("tmp ivals",Nelem,currnum_vars,Npts,dimension);
    for (size_t e=0; e<ptx.extent(0); e++) {
      for (size_t i=0; i<ptx.extent(1); i++) {
        // set the node in wkset
        int dim_ = dimension;
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
        
        for (size_t n=0; n<var_list[set][block].size(); n++) {
          // evaluate
          auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n],"point");
        
          // Also might be ok (terribly inefficient though)
          parallel_for("physics initial set point",
                       RangePolicy<AssemblyExec>(0,1),
                       KOKKOS_LAMBDA (const int s ) {
            ivals(e,n,i,0) = tivals(0,0);
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
                                          const int & block, const bool & project, Teuchos::RCP<Workset<ScalarT> > & wkset) {
  
  size_t currnum_vars = var_list[set][block].size();
  
  View_Sc3 ivals;
  
  if (project) {
    
    ivals = View_Sc3("tmp ivals",pts[0].extent(0), currnum_vars, pts[0].extent(1));
    
    // ip in wkset are set in cell::getInitial
    for (size_t n=0; n<var_list[set][block].size(); n++) {

      auto tivals = function_managers[block]->evaluate("initial " + var_list[set][block][n],"side ip");
      auto cvals = subview( ivals, ALL(), n, ALL());
      //copy
      parallel_for("physics fill initial values",
                   RangePolicy<AssemblyExec>(0,cvals.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_t i=0; i<cvals.extent(1); i++) {
          cvals(e,i) = tivals(e,i);
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
  
  auto tdvals = function_managers[block]->evaluate("Dirichlet " + var_list[set][block][var] + " " + sidename,"side ip");
  
  View_Sc2 dvals("temp dnvals", function_managers[block]->num_elem_, function_managers[block]->num_ip_side_);
  
  // copy values
  parallel_for("physics fill Dirichlet values",
               RangePolicy<AssemblyExec>(0,dvals.extent(0)),
               KOKKOS_LAMBDA (const int e ) {
    for (size_t i=0; i<dvals.extent(1); i++) {
      dvals(e,i) = tdvals(e,i);
    }
  });
  return dvals;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

std::vector<View_Sc2> PhysicsInterface::getDirichletVector(const int & var,
                                                           const int & set,
                                                           const int & block,
                                                           const std::string & sidename) {
  
  std::vector<View_Sc2> dvals_vec(3);
  string varname = var_list[set][block][var];
  std::vector<string> components = {"x", "y", "z"};
  
  // check if vector component functions exist (e.g., "Dirichlet Ex bottom")
  bool has_components = function_managers[block]->hasFunction("Dirichlet " + varname + "x " + sidename);
  
  if (has_components) {
    // use component-wise Dirichlet data
    for (size_t d=0; d<3; d++) {
      string label = "Dirichlet " + varname + components[d] + " " + sidename;
      if (function_managers[block]->hasFunction(label)) {
        auto tdvals = function_managers[block]->evaluate(label, "side ip");
        View_Sc2 dvals("dirichlet component", function_managers[block]->num_elem_, function_managers[block]->num_ip_side_);
        parallel_for("physics fill Dirichlet vector component",
                     RangePolicy<AssemblyExec>(0,dvals.extent(0)),
                     KOKKOS_LAMBDA (const int e) {
          for (size_t i=0; i<dvals.extent(1); i++) {
            dvals(e,i) = tdvals(e,i);
          }
        });
        dvals_vec[d] = dvals;
      }
      else {
        // use zero, if component not specified
        View_Sc2 dvals("dirichlet component zero", function_managers[block]->num_elem_, function_managers[block]->num_ip_side_);
        Kokkos::deep_copy(dvals, 0.0);
        dvals_vec[d] = dvals;
      }
    }
  }
  else {
    // fall back to scalar Dirichlet data broadcast to all components
    auto tdvals = function_managers[block]->evaluate("Dirichlet " + varname + " " + sidename, "side ip");
    for (size_t d=0; d<3; d++) {
      View_Sc2 dvals("dirichlet component broadcast", function_managers[block]->num_elem_, function_managers[block]->num_ip_side_);
      parallel_for("physics fill Dirichlet broadcast",
                   RangePolicy<AssemblyExec>(0,dvals.extent(0)),
                   KOKKOS_LAMBDA (const int e) {
        for (size_t i=0; i<dvals.extent(1); i++) {
          dvals(e,i) = tdvals(e,i);
        }
      });
      dvals_vec[d] = dvals;
    }
  }
  
  return dvals_vec;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

int PhysicsInterface::getUniqueIndex(const int & set, const int & block, const std::string & var) {
  int index = 0;
  size_t prog = 0;
  for (size_t set=0; set<num_vars.size(); ++set) {
    for (size_t j=0; j<num_vars[set][block]; j++) {
      if (var_list[set][block][j] == var) {
        index = unique_index[block][j+prog];
      }
    }
    prog += num_vars[set][block];
  }
  return index;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::volumeResidual(const size_t & set, const size_t block) {

  debugger->print(1, "**** Starting PhysicsInterface volume residual ...");

  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t i=0; i<modules[set][block].size(); i++) {
      modules[set][block][i]->volumeResidual();
    }
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t i=0; i<modules_AD[set][block].size(); i++) {
      modules_AD[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD2>::value) {
    for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
      modules_AD2[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD4>::value) {
    for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
      modules_AD4[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD8>::value) {
    for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
      modules_AD8[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD16>::value) {
    for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
      modules_AD16[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD18>::value) {
    for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
      modules_AD18[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD24>::value) {
    for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
      modules_AD24[set][block][i]->volumeResidual();
    }
  }
  else if (std::is_same<EvalT, AD32>::value) {
    for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
      modules_AD32[set][block][i]->volumeResidual();
    }
  }
#endif
  debugger->print(1, "**** Finished PhysicsInterface volume residual");

}

template void PhysicsInterface::volumeResidual<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::volumeResidual<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::volumeResidual<AD32>(const size_t & set, const size_t block);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::boundaryResidual(const size_t & set, const size_t block) {
  debugger->print(1, "**** Starting PhysicsInterface boundary residual ...");
  
  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t i=0; i<modules[set][block].size(); i++) {
      modules[set][block][i]->boundaryResidual();
    }
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t i=0; i<modules_AD[set][block].size(); i++) {
      modules_AD[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD2>::value) {
    for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
      modules_AD2[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD4>::value) {
    for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
      modules_AD4[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD8>::value) {
    for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
      modules_AD8[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD16>::value) {
    for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
      modules_AD16[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD18>::value) {
    for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
      modules_AD18[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD24>::value) {
    for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
      modules_AD24[set][block][i]->boundaryResidual();
    }
  }
  else if (std::is_same<EvalT, AD32>::value) {
    for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
      modules_AD32[set][block][i]->boundaryResidual();
    }
  }
#endif
  debugger->print(1, "**** Finished PhysicsInterface boundary residual");
  
}

template void PhysicsInterface::boundaryResidual<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::boundaryResidual<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::boundaryResidual<AD32>(const size_t & set, const size_t block);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::computeFlux(const size_t & set, const size_t block) {
  debugger->print(1, "**** Starting PhysicsInterface compute flux ...");
  
  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t i=0; i<modules[set][block].size(); i++) {
      modules[set][block][i]->computeFlux();
    }
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t i=0; i<modules_AD[set][block].size(); i++) {
      modules_AD[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD2>::value) {
    for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
      modules_AD2[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD4>::value) {
    for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
      modules_AD4[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD8>::value) {
    for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
      modules_AD8[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD16>::value) {
    for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
      modules_AD16[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD18>::value) {
    for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
      modules_AD18[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD24>::value) {
    for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
      modules_AD24[set][block][i]->computeFlux();
    }
  }
  else if (std::is_same<EvalT, AD32>::value) {
    for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
      modules_AD32[set][block][i]->computeFlux();
    }
  }
#endif
  debugger->print(1, "**** Finished PhysicsInterface compute flux");
  
}

template void PhysicsInterface::computeFlux<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::computeFlux<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::computeFlux<AD32>(const size_t & set, const size_t block);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

// These cannot be templated (unfortunately)

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<ScalarT> > > & wkset) {
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

#ifndef MrHyDE_NO_AD
void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD[set][block].size(); i++) {
          modules_AD[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD2> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD2.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
          modules_AD2[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD4> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD4.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
          modules_AD4[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD8> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD8.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
          modules_AD8[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD16> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD16.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
          modules_AD16[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD18> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD18.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
          modules_AD18[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD24> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD24.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
          modules_AD24[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}

void PhysicsInterface::setWorkset(vector<Teuchos::RCP<Workset<AD32> > > & wkset) {
  for (size_t block = 0; block<wkset.size(); block++) {
    if (wkset[block]->isInitialized) {
      for (size_t set=0; set<modules_AD32.size(); set++) {
        wkset[block]->updatePhysicsSet(set);
        for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
          modules_AD32[set][block][i]->setWorkset(wkset[block]);
        }
      }
    }
  }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::faceResidual(const size_t & set, const size_t block) {
  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t i=0; i<modules[set][block].size(); i++) {
      modules[set][block][i]->faceResidual();
    }
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t i=0; i<modules_AD[set][block].size(); i++) {
      modules_AD[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD2>::value) {
    for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
      modules_AD2[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD4>::value) {
    for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
      modules_AD4[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD8>::value) {
    for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
      modules_AD8[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD16>::value) {
    for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
      modules_AD16[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD18>::value) {
    for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
      modules_AD18[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD24>::value) {
    for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
      modules_AD24[set][block][i]->faceResidual();
    }
  }
  else if (std::is_same<EvalT, AD32>::value) {
    for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
      modules_AD32[set][block][i]->faceResidual();
    }
  }
#endif
}

template void PhysicsInterface::faceResidual<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::faceResidual<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::faceResidual<AD32>(const size_t & set, const size_t block);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
    
void PhysicsInterface::updateFlags(vector<bool> & newflags) {
  for (size_t set=0; set<modules.size(); set++) {
    for (size_t block=0; block<modules[set].size(); block++) {
      for (size_t i=0; i<modules[set][block].size(); i++) {
        modules[set][block][i]->updateFlags(newflags);
      }
    }
  }
#ifndef MrHyDE_NO_AD
  for (size_t set=0; set<modules_AD.size(); set++) {
    for (size_t block=0; block<modules_AD[set].size(); block++) {
      for (size_t i=0; i<modules_AD[set][block].size(); i++) {
        modules_AD[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD2.size(); set++) {
    for (size_t block=0; block<modules_AD2[set].size(); block++) {
      for (size_t i=0; i<modules_AD2[set][block].size(); i++) {
        modules_AD2[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD4.size(); set++) {
    for (size_t block=0; block<modules_AD4[set].size(); block++) {
      for (size_t i=0; i<modules_AD4[set][block].size(); i++) {
        modules_AD4[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD8.size(); set++) {
    for (size_t block=0; block<modules_AD8[set].size(); block++) {
      for (size_t i=0; i<modules_AD8[set][block].size(); i++) {
        modules_AD8[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD16.size(); set++) {
    for (size_t block=0; block<modules_AD16[set].size(); block++) {
      for (size_t i=0; i<modules_AD16[set][block].size(); i++) {
        modules_AD16[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD18.size(); set++) {
    for (size_t block=0; block<modules_AD18[set].size(); block++) {
      for (size_t i=0; i<modules_AD18[set][block].size(); i++) {
        modules_AD18[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD24.size(); set++) {
    for (size_t block=0; block<modules_AD24[set].size(); block++) {
      for (size_t i=0; i<modules_AD24[set][block].size(); i++) {
        modules_AD24[set][block][i]->updateFlags(newflags);
      }
    }
  }
  for (size_t set=0; set<modules_AD32.size(); set++) {
    for (size_t block=0; block<modules_AD32[set].size(); block++) {
      for (size_t i=0; i<modules_AD32[set][block].size(); i++) {
        modules_AD32[set][block][i]->updateFlags(newflags);
      }
    }
  }
#endif
}
    
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void PhysicsInterface::fluxConditions(const size_t & set, const size_t block) {
  if (std::is_same<EvalT, ScalarT>::value) {
    for (size_t var=0; var<var_list[set][block].size(); ++var) {
      int cside = function_managers[block]->wkset->currentside;
      string bctype = function_managers[block]->wkset->var_bcs(var,cside);
      if (bctype == "Flux") {
        string varname = var_list[set][block][var];
        string sidename = function_managers[block]->wkset->sidename;
        string label = "Flux " + varname + " " + sidename;
        auto fluxvals = function_managers[block]->evaluate(label,"side ip");
      
        auto basis = function_managers[block]->wkset->getBasisSide(varname);
        auto wts = function_managers[block]->wkset->wts_side;
        auto res = function_managers[block]->wkset->res;
        auto off = function_managers[block]->wkset->getOffsets(varname);
      
        parallel_for("physics flux condition",
                     TeamPolicy<AssemblyExec>(wts.extent(0), Kokkos::AUTO, VECTORSIZE),
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
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    for (size_t var=0; var<var_list[set][block].size(); ++var) {
      int cside = function_managers_AD[block]->wkset->currentside;
      string bctype = function_managers_AD[block]->wkset->var_bcs(var,cside);
      if (bctype == "Flux") {
        string varname = var_list[set][block][var];
        string sidename = function_managers_AD[block]->wkset->sidename;
        string label = "Flux " + varname + " " + sidename;
        auto fluxvals = function_managers_AD[block]->evaluate(label,"side ip");
      
        auto basis = function_managers_AD[block]->wkset->getBasisSide(varname);
        auto wts = function_managers_AD[block]->wkset->wts_side;
        auto res = function_managers_AD[block]->wkset->res;
        auto off = function_managers_AD[block]->wkset->getOffsets(varname);
      
        parallel_for("physics flux condition",
                     TeamPolicy<AssemblyExec>(wts.extent(0), Kokkos::AUTO, VECTORSIZE),
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
#endif
}

template void PhysicsInterface::fluxConditions<ScalarT>(const size_t & set, const size_t block);
#ifndef MrHyDE_NO_AD
template void PhysicsInterface::fluxConditions<AD>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD2>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD4>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD8>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD16>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD18>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD24>(const size_t & set, const size_t block);
template void PhysicsInterface::fluxConditions<AD32>(const size_t & set, const size_t block);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
    
void PhysicsInterface::purgeMemory() {
  // nothing here
}
