/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

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

