/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ========================================================================================
// ========================================================================================

template<class Node>
void AssemblyManager<Node>::addFunction(const int & block, const string & name, const string & expression, const string & location) {
  function_managers[block]->addFunction(name, expression, location);
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    function_managers_AD[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 2) {
    function_managers_AD2[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 4) {
    function_managers_AD4[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 8) {
    function_managers_AD8[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 16) {
    function_managers_AD16[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 18) {
    function_managers_AD18[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 24) {
    function_managers_AD24[block]->addFunction(name, expression, location);
  }
  else if (type_AD == 32) {
    function_managers_AD32[block]->addFunction(name, expression, location);
  }
#endif
}

// ========================================================================================
// ========================================================================================

template<class Node>
View_Sc2 AssemblyManager<Node>::evaluateFunction(const int & block, const string & name, const string & location) {

  typedef typename Node::execution_space LA_exec;

  auto data = function_managers[block]->evaluate(name, location);
  size_type num_elem = function_managers[block]->num_elem_;
  size_type num_pts = 0;
  if (location == "ip") {
    num_pts = function_managers[block]->num_ip_;
  }
  else if (location == "side ip") {
    num_pts = function_managers[block]->num_ip_side_;
  }
  else if (location == "point") {
    num_pts = 1;
  }

  View_Sc2 outdata("data from function evaluation", num_elem, num_pts);

  parallel_for("assembly eval func",
                 RangePolicy<LA_exec>(0,num_elem),
                 KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<num_pts; ++pt) {
      outdata(elem,pt) = data(elem,pt);
    }
  });

  return outdata;
}
    

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we finalize the function managers
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::finalizeFunctions() {
  
  debugger->print("**** Starting AssemblyManager::finalizeFunctions()");
  
  for (size_t block=0; block<wkset.size(); ++block) {
    this->finalizeFunctions(function_managers[block], wkset[block]);
  }
  
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    for (size_t block=0; block<wkset_AD.size(); ++block) {
      this->finalizeFunctions(function_managers_AD[block], wkset_AD[block]);
    }
  }
  else if (type_AD == 2) {
    for (size_t block=0; block<wkset_AD2.size(); ++block) {
      this->finalizeFunctions(function_managers_AD2[block], wkset_AD2[block]);
    }
  }
  else if (type_AD == 4) {
    for (size_t block=0; block<wkset_AD4.size(); ++block) {
      this->finalizeFunctions(function_managers_AD4[block], wkset_AD4[block]);
    }
  }
  else if (type_AD == 8) {
    for (size_t block=0; block<wkset_AD8.size(); ++block) {
      this->finalizeFunctions(function_managers_AD8[block], wkset_AD8[block]);
    }
  }
  else if (type_AD == 16) {
    for (size_t block=0; block<wkset_AD16.size(); ++block) {
      this->finalizeFunctions(function_managers_AD16[block], wkset_AD16[block]);
    }
  }
  else if (type_AD == 18) {
    for (size_t block=0; block<wkset_AD18.size(); ++block) {
      this->finalizeFunctions(function_managers_AD18[block], wkset_AD18[block]);
    }
  }
  else if (type_AD == 24) {
    for (size_t block=0; block<wkset_AD24.size(); ++block) {
      this->finalizeFunctions(function_managers_AD24[block], wkset_AD24[block]);
    }
  }
  else if (type_AD == 32) {
    for (size_t block=0; block<wkset_AD32.size(); ++block) {
      this->finalizeFunctions(function_managers_AD32[block], wkset_AD32[block]);
    }
  }
#endif
  
  debugger->print("**** Finished AssemblyManager::finalizeFunctions()");
  
}

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::finalizeFunctions(Teuchos::RCP<FunctionManager<EvalT> > & fman,
                                              Teuchos::RCP<Workset<EvalT> > & wset) {
  fman->setupLists(params->paramnames);
  fman->wkset = wset;
  if (wset->isInitialized) {
    fman->decomposeFunctions();
    if (verbosity >= 20) {
      fman->printFunctions();
      wset->printSolutionFields();
      wset->printScalarFields();
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Create the function managers
////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createFunctions() {
    
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    function_managers.push_back(Teuchos::rcp(new FunctionManager<ScalarT>(blocknames[block],
                                                                     groupData[block]->num_elem,
                                                                     disc->numip[block],
                                                                     disc->numip_side[block])));
  }
  physics->defineFunctions(function_managers);

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD.push_back(Teuchos::rcp(new FunctionManager<AD>(blocknames[block],
                                                                          groupData[block]->num_elem,
                                                                          disc->numip[block],
                                                                          disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD);
  }
  else if (type_AD == 2) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD2.push_back(Teuchos::rcp(new FunctionManager<AD2>(blocknames[block],
                                                                            groupData[block]->num_elem,
                                                                            disc->numip[block],
                                                                            disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD2);
  }
  else if (type_AD == 4) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD4.push_back(Teuchos::rcp(new FunctionManager<AD4>(blocknames[block],
                                                                            groupData[block]->num_elem,
                                                                            disc->numip[block],
                                                                            disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD4);
  }
  else if (type_AD == 8) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD8.push_back(Teuchos::rcp(new FunctionManager<AD8>(blocknames[block],
                                                                            groupData[block]->num_elem,
                                                                            disc->numip[block],
                                                                            disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD8);
  }
  else if (type_AD == 16) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD16.push_back(Teuchos::rcp(new FunctionManager<AD16>(blocknames[block],
                                                                              groupData[block]->num_elem,
                                                                              disc->numip[block],
                                                                              disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD16);
  }
  else if (type_AD == 18) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD18.push_back(Teuchos::rcp(new FunctionManager<AD18>(blocknames[block],
                                                                              groupData[block]->num_elem,
                                                                              disc->numip[block],
                                                                              disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD18);
  }
  else if (type_AD == 24) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD24.push_back(Teuchos::rcp(new FunctionManager<AD24>(blocknames[block],
                                                                              groupData[block]->num_elem,
                                                                              disc->numip[block],
                                                                              disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD24);
  }
  else if (type_AD == 32) {
    for (size_t block=0; block<blocknames.size(); ++block) {
      function_managers_AD32.push_back(Teuchos::rcp(new FunctionManager<AD32>(blocknames[block],
                                                                              groupData[block]->num_elem,
                                                                              disc->numip[block],
                                                                              disc->numip_side[block])));
    }
    physics->defineFunctions(function_managers_AD32);
  }
#endif
}

