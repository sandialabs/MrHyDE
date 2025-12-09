/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/


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

