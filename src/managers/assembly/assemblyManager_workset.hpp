/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

/////////////////////////////////////////////////////////////////////////////
// Worksets
/////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::createWorkset() {
  
  Teuchos::TimeMonitor localtimer(*wkset_timer);
  
  debugger->print("**** Starting AssemblyManager::createWorkset ...");
  
  for (size_t block=0; block<groups.size(); ++block) {
    if (groups[block].size() > 0) {
      vector<int> info;
      info.push_back(groupData[block]->dimension);
      info.push_back((int)groupData[block]->num_disc_params);
      info.push_back(groupData[block]->num_elem);
      info.push_back(groupData[block]->num_ip);
      info.push_back(groupData[block]->num_side_ip);
      info.push_back(physics->set_names.size());
      info.push_back(params->num_active_params);
      vector<size_t> numVars;
      for (size_t set=0; set<groupData[block]->set_num_dof.size(); ++set) {
        numVars.push_back(groupData[block]->set_num_dof[set].extent(0));
      }
      vector<Kokkos::View<string**,HostDevice> > bcs(physics->set_names.size());
      if (mesh->use_stk_mesh) {
        for (size_t set=0; set<physics->set_names.size(); ++set) {
          Kokkos::View<string**,HostDevice> vbcs = disc->getVarBCs(set,block);
          bcs[set] = vbcs;
        }
      }

      // ScalarT workset, always active unless no elements on proc
      wkset.push_back(Teuchos::rcp( new Workset<ScalarT>(info, numVars, isTransient,
                                                         disc->basis_types[block],
                                                         disc->basis_pointers[block],
                                                         params->discretized_param_basis,
                                                         groupData[block]->cell_topo)));
      wkset[block]->block = block;
      wkset[block]->blockname = blocknames[block];
      wkset[block]->set_var_bcs = bcs;
      wkset[block]->var_bcs = bcs[0];

#ifndef MrHyDE_NO_AD
      bool fully_explicit = settings->sublist("Solver").get<bool>("fully explicit",false);
      string analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
      bool requires_AD = true;
      if (fully_explicit && (analysis_type == "forward" || analysis_type == "dry-run")) {
        requires_AD = false;
      }
      
      bool found = false;
      
      if (requires_AD && !found && type_AD == 2 ) {
        // AD2 workset
        wkset_AD2.push_back(Teuchos::rcp( new Workset<AD2>(info, numVars, isTransient,
                                                           disc->basis_types[block],
                                                           disc->basis_pointers[block],
                                                           params->discretized_param_basis,
                                                           groupData[block]->cell_topo)));
        wkset_AD2[block]->block = block;
        wkset_AD2[block]->blockname = blocknames[block];
        wkset_AD2[block]->set_var_bcs = bcs;
        wkset_AD2[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        //wkset_AD2.push_back(Teuchos::rcp( new Workset<AD2>(block, physics->set_names.size())));
      }

      if (requires_AD && !found && type_AD == 4 ) {
        // AD4 workset
        wkset_AD4.push_back(Teuchos::rcp( new Workset<AD4>(info, numVars, isTransient,
                                                             disc->basis_types[block],
                                                             disc->basis_pointers[block],
                                                             params->discretized_param_basis,
                                                             groupData[block]->cell_topo)));
        wkset_AD4[block]->block = block;
        wkset_AD4[block]->blockname = blocknames[block];
        wkset_AD4[block]->set_var_bcs = bcs;
        wkset_AD4[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        //wkset_AD4.push_back(Teuchos::rcp( new Workset<AD4>(block, physics->set_names.size())));
      }

      if (requires_AD && !found && type_AD == 8 ) {
        // AD8 workset
        wkset_AD8.push_back(Teuchos::rcp( new Workset<AD8>(info, numVars, isTransient,
                                                             disc->basis_types[block],
                                                             disc->basis_pointers[block],
                                                             params->discretized_param_basis,
                                                             groupData[block]->cell_topo)));
        wkset_AD8[block]->block = block;
        wkset_AD8[block]->blockname = blocknames[block];
        wkset_AD8[block]->set_var_bcs = bcs;
        wkset_AD8[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        //wkset_AD8.push_back(Teuchos::rcp( new Workset<AD8>(block, physics->set_names.size())));
      }

      if (requires_AD && !found && type_AD == 16 ) {
        // AD16 workset
        wkset_AD16.push_back(Teuchos::rcp( new Workset<AD16>(info, numVars, isTransient,
                                                               disc->basis_types[block],
                                                               disc->basis_pointers[block],
                                                               params->discretized_param_basis,
                                                               groupData[block]->cell_topo)));
        wkset_AD16[block]->block = block;
        wkset_AD16[block]->blockname = blocknames[block];
        wkset_AD16[block]->set_var_bcs = bcs;
        wkset_AD16[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        //wkset_AD16.push_back(Teuchos::rcp( new Workset<AD16>(block, physics->set_names.size())));
      }

      if (requires_AD && !found && type_AD == 18 ) {
        // AD18 workset
        wkset_AD18.push_back(Teuchos::rcp( new Workset<AD18>(info, numVars, isTransient,
                                                               disc->basis_types[block],
                                                               disc->basis_pointers[block],
                                                               params->discretized_param_basis,
                                                               groupData[block]->cell_topo)));
        wkset_AD18[block]->block = block;
        wkset_AD18[block]->blockname = blocknames[block];
        wkset_AD18[block]->set_var_bcs = bcs;
        wkset_AD18[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        //wkset_AD18.push_back(Teuchos::rcp( new Workset<AD18>(block, physics->set_names.size())));
      }

      if (requires_AD && !found && type_AD == 24 ) {
        // AD24 workset
        wkset_AD24.push_back(Teuchos::rcp( new Workset<AD24>(info, numVars, isTransient,
                                                               disc->basis_types[block],
                                                               disc->basis_pointers[block],
                                                               params->discretized_param_basis,
                                                               groupData[block]->cell_topo)));
        wkset_AD24[block]->block = block;
        wkset_AD24[block]->blockname = blocknames[block];
        wkset_AD24[block]->set_var_bcs = bcs;
        wkset_AD24[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        //wkset_AD24.push_back(Teuchos::rcp( new Workset<AD24>(block, physics->set_names.size())));
      }

      if (requires_AD && !found && type_AD == 32 ) {
        // AD32 workset
        wkset_AD32.push_back(Teuchos::rcp( new Workset<AD32>(info, numVars, isTransient,
                                                               disc->basis_types[block],
                                                               disc->basis_pointers[block],
                                                               params->discretized_param_basis,
                                                               groupData[block]->cell_topo)));
        wkset_AD32[block]->block = block;
        wkset_AD32[block]->blockname = blocknames[block];
        wkset_AD32[block]->set_var_bcs = bcs;
        wkset_AD32[block]->var_bcs = bcs[0];
        found = true;
      }
      else {
        //wkset_AD32.push_back(Teuchos::rcp( new Workset<AD32>(block, physics->set_names.size())));
      }

      if ((requires_AD && !found) || groupData[block]->multiscale || !allow_autotune) {
        // AD workset
        wkset_AD.push_back(Teuchos::rcp( new Workset<AD>(info, numVars, isTransient,
                                                         disc->basis_types[block],
                                                         disc->basis_pointers[block],
                                                         params->discretized_param_basis,
                                                         groupData[block]->cell_topo)));
        wkset_AD[block]->block = block;
        wkset_AD[block]->blockname = blocknames[block];
        wkset_AD[block]->set_var_bcs = bcs;
        wkset_AD[block]->var_bcs = bcs[0];
        
      }
      else {
        //wkset_AD.push_back(Teuchos::rcp( new Workset<AD>(block, physics->set_names.size())));
      }
     
#endif
    }
    else {
      wkset.push_back(Teuchos::rcp( new Workset<ScalarT>(block, physics->set_names.size())));
#ifndef MrHyDE_NO_AD
      if (type_AD == -1) {
        wkset_AD.push_back(Teuchos::rcp( new Workset<AD>(block, physics->set_names.size())));
      }
      else if (type_AD == 2) {
        wkset_AD2.push_back(Teuchos::rcp( new Workset<AD2>(block, physics->set_names.size())));
      }
      else if (type_AD == 4) {
        wkset_AD4.push_back(Teuchos::rcp( new Workset<AD4>(block, physics->set_names.size())));
      }
      else if (type_AD == 8) {
        wkset_AD8.push_back(Teuchos::rcp( new Workset<AD8>(block, physics->set_names.size())));
      }
      else if (type_AD == 16) {
        wkset_AD16.push_back(Teuchos::rcp( new Workset<AD16>(block, physics->set_names.size())));
      }
      else if (type_AD == 18) {
        wkset_AD18.push_back(Teuchos::rcp( new Workset<AD18>(block, physics->set_names.size())));
      }
      else if (type_AD == 24) {
        wkset_AD24.push_back(Teuchos::rcp( new Workset<AD24>(block, physics->set_names.size())));
      }
      else if (type_AD == 32) {
        wkset_AD32.push_back(Teuchos::rcp( new Workset<AD32>(block, physics->set_names.size())));
      }
#endif
    }
    
  }
  
  debugger->print("**** Finished AssemblyManager::createWorkset");
  
}

/// @brief ////////////////////////////
/// @tparam Node
/// @param block
/// @param isTransient
/// @param current_time
/// @param deltat /
template<class Node>
void AssemblyManager<Node>::updateWorksetTime(const size_t & block, const bool & isTransient,
                                              const ScalarT & current_time, const ScalarT & deltat) {

  if (wkset[block]->isInitialized) {
    this->updateWorksetTime(wkset[block], isTransient, current_time, deltat);
  }
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateWorksetTime(wkset_AD[block], isTransient, current_time, deltat);
  }
  else if (type_AD == 2) {
    this->updateWorksetTime(wkset_AD2[block], isTransient, current_time, deltat);
  }
  else if (type_AD == 4) {
    this->updateWorksetTime(wkset_AD4[block], isTransient, current_time, deltat);
  }
  else if (type_AD == 8) {
    this->updateWorksetTime(wkset_AD8[block], isTransient, current_time, deltat);
  }
  else if (type_AD == 16) {
    this->updateWorksetTime(wkset_AD16[block], isTransient, current_time, deltat);
  }
  else if (type_AD == 18) {
    this->updateWorksetTime(wkset_AD18[block], isTransient, current_time, deltat);
  }
  else if (type_AD == 24) {
    this->updateWorksetTime(wkset_AD24[block], isTransient, current_time, deltat);
  }
  else if (type_AD == 32) {
    this->updateWorksetTime(wkset_AD32[block], isTransient, current_time, deltat);
  }
#endif
}

/// @brief /////////////////////////////
/// @tparam Node
/// @param wset
/// @param isTransient
/// @param current_time
/// @param deltat /
template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetTime(Teuchos::RCP<Workset<EvalT> > & wset, const bool & isTransient,
                                              const ScalarT & current_time, const ScalarT & deltat) {

  if (isTransient) {
    // TMW: tmp fix
    auto butcher_c = Kokkos::create_mirror_view(wset->butcher_c);
    Kokkos::deep_copy(butcher_c, wset->butcher_c);
    ScalarT timeval = current_time + butcher_c(wset->current_stage)*deltat;
    
    wset->setTime(timeval);
    wset->setDeltat(deltat);
    wset->alpha = 1.0/deltat;
  }
  
  wset->isTransient = isTransient;
}


////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetAdjoint(const size_t & block, const bool & isAdjoint) {
                                                

  if (wkset[block]->isInitialized) {
    this->updateWorksetAdjoint(wkset[block], isAdjoint);
  }
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateWorksetAdjoint(wkset_AD[block], isAdjoint);
  }
  else if (type_AD == 2) {
    this->updateWorksetAdjoint(wkset_AD2[block], isAdjoint);
  }
  else if (type_AD == 4) {
    this->updateWorksetAdjoint(wkset_AD4[block], isAdjoint);
  }
  else if (type_AD == 8) {
    this->updateWorksetAdjoint(wkset_AD8[block], isAdjoint);
  }
  else if (type_AD == 16) {
    this->updateWorksetAdjoint(wkset_AD16[block], isAdjoint);
  }
  else if (type_AD == 18) {
    this->updateWorksetAdjoint(wkset_AD18[block], isAdjoint);
  }
  else if (type_AD == 24) {
    this->updateWorksetAdjoint(wkset_AD24[block], isAdjoint);
  }
  else if (type_AD == 32) {
    this->updateWorksetAdjoint(wkset_AD32[block], isAdjoint);
  }
#endif
}

////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetAdjoint(Teuchos::RCP<Workset<EvalT> > & wset, const bool & isAdjoint) {
  wset->isAdjoint = isAdjoint;
}

////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetEID(const size_t & block, const size_t & eid) {
                                                

  if (wkset[block]->isInitialized) {
    this->updateWorksetEID(wkset[block], eid);
  }
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateWorksetEID(wkset_AD[block], eid);
  }
  else if (type_AD == 2) {
    this->updateWorksetEID(wkset_AD2[block], eid);
  }
  else if (type_AD == 4) {
    this->updateWorksetEID(wkset_AD4[block], eid);
  }
  else if (type_AD == 8) {
    this->updateWorksetEID(wkset_AD8[block], eid);
  }
  else if (type_AD == 16) {
    this->updateWorksetEID(wkset_AD16[block], eid);
  }
  else if (type_AD == 18) {
    this->updateWorksetEID(wkset_AD18[block], eid);
  }
  else if (type_AD == 24) {
    this->updateWorksetEID(wkset_AD24[block], eid);
  }
  else if (type_AD == 32) {
    this->updateWorksetEID(wkset_AD32[block], eid);
  }
#endif
}

////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetEID(Teuchos::RCP<Workset<EvalT> > & wset, const size_t & eid) {
  wset->localEID = eid;
}

////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetOnSide(const size_t & block, const bool & on_side) {
                                                

  this->updateWorksetOnSide(wkset[block], on_side);
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateWorksetOnSide(wkset_AD[block], on_side);
  }
  else if (type_AD == 2) {
    this->updateWorksetOnSide(wkset_AD2[block], on_side);
  }
  else if (type_AD == 4) {
    this->updateWorksetOnSide(wkset_AD4[block], on_side);
  }
  else if (type_AD == 8) {
    this->updateWorksetOnSide(wkset_AD8[block], on_side);
  }
  else if (type_AD == 16) {
    this->updateWorksetOnSide(wkset_AD16[block], on_side);
  }
  else if (type_AD == 18) {
    this->updateWorksetOnSide(wkset_AD18[block], on_side);
  }
  else if (type_AD == 24) {
    this->updateWorksetOnSide(wkset_AD24[block], on_side);
  }
  else if (type_AD == 32) {
    this->updateWorksetOnSide(wkset_AD32[block], on_side);
  }
#endif
}

////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetOnSide(Teuchos::RCP<Workset<EvalT> > & wset, const bool & on_side) {
  wset->isOnSide = on_side;
}

////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetResidual(const size_t & block) {
                                                

  this->updateWorksetResidual(wkset[block]);
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateWorksetResidual(wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->updateWorksetResidual(wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->updateWorksetResidual(wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->updateWorksetResidual(wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->updateWorksetResidual(wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->updateWorksetResidual(wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->updateWorksetResidual(wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->updateWorksetResidual(wkset_AD32[block]);
  }
#endif
}

////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetResidual(Teuchos::RCP<Workset<EvalT> > & wset) {
  wset->resetResidual();
}

////////////////////////////////////////////////
///
template<class Node>
void AssemblyManager<Node>::setWorksetButcher(const size_t & set, const size_t & block,
                                        Kokkos::View<ScalarT**,AssemblyDevice> butcher_A,
                                        Kokkos::View<ScalarT*,AssemblyDevice> butcher_b,
                                        Kokkos::View<ScalarT*,AssemblyDevice> butcher_c) {


  this->setWorksetButcher(set, wkset[block],butcher_A, butcher_b, butcher_c);
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->setWorksetButcher(set, wkset_AD[block],butcher_A, butcher_b, butcher_c);
  }
  else if (type_AD == 2) {
    this->setWorksetButcher(set, wkset_AD2[block],butcher_A, butcher_b, butcher_c);
  }
  else if (type_AD == 4) {
    this->setWorksetButcher(set, wkset_AD4[block],butcher_A, butcher_b, butcher_c);
  }
  else if (type_AD == 8) {
    this->setWorksetButcher(set, wkset_AD8[block],butcher_A, butcher_b, butcher_c);
  }
  else if (type_AD == 16) {
    this->setWorksetButcher(set, wkset_AD16[block],butcher_A, butcher_b, butcher_c);
  }
  else if (type_AD == 18) {
    this->setWorksetButcher(set, wkset_AD18[block],butcher_A, butcher_b, butcher_c);
  }
  else if (type_AD == 24) {
    this->setWorksetButcher(set, wkset_AD24[block],butcher_A, butcher_b, butcher_c);
  }
  else if (type_AD == 32) {
    this->setWorksetButcher(set, wkset_AD32[block],butcher_A, butcher_b, butcher_c);
  }
#endif
}

////////////////////////////////////////////////
///
template<class Node>
template<class EvalT>
void AssemblyManager<Node>::setWorksetButcher(const size_t & set, Teuchos::RCP<Workset<EvalT> > & wset,
                                        Kokkos::View<ScalarT**,AssemblyDevice> butcher_A,
                                        Kokkos::View<ScalarT*,AssemblyDevice> butcher_b,
                                        Kokkos::View<ScalarT*,AssemblyDevice> butcher_c) {

  wset->set_butcher_A[set] = butcher_A;
  wset->set_butcher_b[set] = butcher_b;
  wset->set_butcher_c[set] = butcher_c;

  // TODO dont like this... but should protect against 1 set errors
  wset->butcher_A = butcher_A;
  wset->butcher_b = butcher_b;
  wset->butcher_c = butcher_c;

}

////////////////////////////////////////////////
///
template<class Node>
void AssemblyManager<Node>::setWorksetBDF(const size_t & set, const size_t & block,
                                        Kokkos::View<ScalarT*,AssemblyDevice> BDF_wts) {


  this->setWorksetBDF(set, wkset[block], BDF_wts);
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->setWorksetBDF(set, wkset_AD[block], BDF_wts);
  }
  else if (type_AD == 2) {
    this->setWorksetBDF(set, wkset_AD2[block], BDF_wts);
  }
  else if (type_AD == 4) {
    this->setWorksetBDF(set, wkset_AD4[block], BDF_wts);
  }
  else if (type_AD == 8) {
    this->setWorksetBDF(set, wkset_AD8[block], BDF_wts);
  }
  else if (type_AD == 16) {
    this->setWorksetBDF(set, wkset_AD16[block], BDF_wts);
  }
  else if (type_AD == 18) {
    this->setWorksetBDF(set, wkset_AD18[block], BDF_wts);
  }
  else if (type_AD == 24) {
    this->setWorksetBDF(set, wkset_AD24[block], BDF_wts);
  }
  else if (type_AD == 32) {
    this->setWorksetBDF(set, wkset_AD32[block], BDF_wts);
  }
#endif
}

////////////////////////////////////////////////
///
template<class Node>
template<class EvalT>
void AssemblyManager<Node>::setWorksetBDF(const size_t & set, Teuchos::RCP<Workset<EvalT> > & wset,
                                        Kokkos::View<ScalarT*,AssemblyDevice> BDF_wts) {

  wset->set_BDF_wts[set] = BDF_wts;
  wset->BDF_wts = BDF_wts;

}


/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetBoundary(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
  this->updateWorksetBoundary(wkset[block], block, grp, seedwhat, seedindex, override_transient);
}


/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetBoundaryAD(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateWorksetBoundary(wkset_AD[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 2) {
    this->updateWorksetBoundary(wkset_AD2[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 4) {
    this->updateWorksetBoundary(wkset_AD4[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 8) {
    this->updateWorksetBoundary(wkset_AD8[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 16) {
    this->updateWorksetBoundary(wkset_AD16[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 18) {
    this->updateWorksetBoundary(wkset_AD18[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 24) {
    this->updateWorksetBoundary(wkset_AD24[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 32) {
    this->updateWorksetBoundary(wkset_AD32[block], block, grp, seedwhat, seedindex, override_transient);
  }
#endif
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetBoundary(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {

  
  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateWorksetBoundary(wkset[block], block, grp, seedwhat, seedindex, override_transient);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateWorksetBoundary(wkset_AD[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateWorksetBoundary(wkset_AD2[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateWorksetBoundary(wkset_AD4[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateWorksetBoundary(wkset_AD8[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateWorksetBoundary(wkset_AD16[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateWorksetBoundary(wkset_AD18[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateWorksetBoundary(wkset_AD24[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateWorksetBoundary(wkset_AD32[block], block, grp, seedwhat, seedindex, override_transient);
  }
#endif
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetBoundary(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp,
                                                  const int & seedwhat, const int & seedindex, const bool & override_transient) {
  
  ///////////////////////////////////////////////////////////
  // Reset the residual and data in the workset
  
  wset->reset();
  wset->sidename = boundary_groups[block][grp]->sidename;
  wset->currentside = boundary_groups[block][grp]->sidenum;
  wset->numElem = boundary_groups[block][grp]->numElem;

  ///////////////////////////////////////////////////////////
  // Update the observational data stored in the workset
  // This is specific to cases with data-based parameters

  this->updateDataBoundary<EvalT>(block, grp);
  
  ///////////////////////////////////////////////////////////
  // Update the integration info and basis in workset

  this->updateWorksetBasisBoundary<EvalT>(block, grp);
  
  ///////////////////////////////////////////////////////////
  // Map the gathered solution to seeded version in workset
  if (groupData[block]->requires_transient && !override_transient) {
    for (size_t set=0; set<groupData[block]->num_sets; ++set) {
      if (boundary_groups[block][grp]->have_sols) {
        wset->computeSolnTransientSeeded(set, boundary_groups[block][grp]->sol[set],
                                          boundary_groups[block][grp]->sol_prev[set],
                                          boundary_groups[block][grp]->sol_stage[set],
                                          seedwhat, seedindex);
      }
      else {
        wset->computeSolnTransientSeeded(set, groupData[block]->sol[set],
                                          groupData[block]->sol_prev[set],
                                          groupData[block]->sol_stage[set],
                                          seedwhat, seedindex);
      }
    }
  }
  else { // steady-state
    for (size_t set=0; set<groupData[block]->num_sets; ++set) {
      if (boundary_groups[block][grp]->have_sols) {
        wset->computeSolnSteadySeeded(set, boundary_groups[block][grp]->sol[set], seedwhat);
      }
      else {
        wset->computeSolnSteadySeeded(set, groupData[block]->sol[set], seedwhat);
      }
    }
  }
  if (wset->numDiscParams > 0) {
    if (boundary_groups[block][grp]->have_sols) {
      wset->computeParamSteadySeeded(boundary_groups[block][grp]->param, seedwhat);
    }
    else {
      wset->computeParamSteadySeeded(groupData[block]->param, seedwhat);
    }
  }

  // Aux solutions are still handled separately
  //this->computeSoln(seedwhat);
  if (wset->numAux > 0 && std::is_same<EvalT,AD>::value) {
       this->computeBoundaryAux(block, grp, seedwhat);
  }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetBasisBoundary(const int & block, const size_t & grp) {
  this->updateWorksetBasisBoundary<ScalarT>(block, grp);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetBasisBoundaryAD(const int & block, const size_t & grp) {
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateWorksetBasisBoundary(wkset_AD[block], block, grp);
  }
  else if (type_AD == 2) {
    this->updateWorksetBasisBoundary(wkset_AD2[block], block, grp);
  }
  else if (type_AD == 4) {
    this->updateWorksetBasisBoundary(wkset_AD4[block], block, grp);
  }
  else if (type_AD == 8) {
    this->updateWorksetBasisBoundary(wkset_AD8[block], block, grp);
  }
  else if (type_AD == 16) {
    this->updateWorksetBasisBoundary(wkset_AD16[block], block, grp);
  }
  else if (type_AD == 18) {
    this->updateWorksetBasisBoundary(wkset_AD18[block], block, grp);
  }
  else if (type_AD == 24) {
    this->updateWorksetBasisBoundary(wkset_AD24[block], block, grp);
  }
  else if (type_AD == 32) {
    this->updateWorksetBasisBoundary(wkset_AD32[block], block, grp);
  }

#endif
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetBasisBoundary(const int & block, const size_t & grp) {

  
  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateWorksetBasisBoundary(wkset[block], block, grp);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateWorksetBasisBoundary(wkset_AD[block], block, grp);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateWorksetBasisBoundary(wkset_AD2[block], block, grp);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateWorksetBasisBoundary(wkset_AD4[block], block, grp);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateWorksetBasisBoundary(wkset_AD8[block], block, grp);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateWorksetBasisBoundary(wkset_AD16[block], block, grp);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateWorksetBasisBoundary(wkset_AD18[block], block, grp);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateWorksetBasisBoundary(wkset_AD24[block], block, grp);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateWorksetBasisBoundary(wkset_AD32[block], block, grp);
  }
#endif
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetBasisBoundary(Teuchos::RCP<Workset<EvalT> > & wset,
                                                       const int & block, const size_t & grp) {

  wset->wts_side = boundary_groups[block][grp]->wts;
  //wset->h = boundary_groups[block][grp]->hsize;
  
  wset->setScalarField(boundary_groups[block][grp]->ip[0],"x");
  wset->setScalarField(boundary_groups[block][grp]->normals[0],"n[x]");
  wset->setScalarField(boundary_groups[block][grp]->tangents[0],"t[x]");
  if (boundary_groups[block][grp]->ip.size() > 1) {
    wset->setScalarField(boundary_groups[block][grp]->ip[1],"y");
    wset->setScalarField(boundary_groups[block][grp]->normals[1],"n[y]");
    wset->setScalarField(boundary_groups[block][grp]->tangents[1],"t[y]");
  }
  if (boundary_groups[block][grp]->ip.size() > 2) {
    wset->setScalarField(boundary_groups[block][grp]->ip[2],"z");
    wset->setScalarField(boundary_groups[block][grp]->normals[2],"n[z]");
    wset->setScalarField(boundary_groups[block][grp]->tangents[2],"t[z]");
  }

  if (boundary_groups[block][grp]->storeAll || groupData[block]->use_basis_database) {
    wset->basis_side = boundary_groups[block][grp]->basis;
    wset->basis_grad_side = boundary_groups[block][grp]->basis_grad;
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalBoundaryBasis(groupData[block], boundary_groups[block][grp]->localElemID,
                                   boundary_groups[block][grp]->localSideID,
                                   tbasis, tbasis_grad, tbasis_curl, tbasis_div);
    vector<CompressedView<View_Sc4>> tcbasis, tcbasis_grad;
    for (size_t i=0; i<tbasis.size(); ++i) {
      tcbasis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      tcbasis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
    }
    wset->basis_side = tcbasis;
    wset->basis_grad_side = tcbasis_grad;
  }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorkset(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
  this->updateWorkset<ScalarT>(block, grp, seedwhat, seedindex, override_transient);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetAD(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->updateWorkset(wkset_AD[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 2) {
    this->updateWorkset(wkset_AD2[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 4) {
    this->updateWorkset(wkset_AD4[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 8) {
    this->updateWorkset(wkset_AD8[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 16) {
    this->updateWorkset(wkset_AD16[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 18) {
    this->updateWorkset(wkset_AD18[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 24) {
    this->updateWorkset(wkset_AD24[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (type_AD == 32) {
    this->updateWorkset(wkset_AD32[block], block, grp, seedwhat, seedindex, override_transient);
  }
#endif
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorkset(const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {

  
  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateWorkset(wkset[block], block, grp, seedwhat, seedindex, override_transient);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateWorkset(wkset_AD[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateWorkset(wkset_AD2[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateWorkset(wkset_AD4[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateWorkset(wkset_AD8[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateWorkset(wkset_AD16[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateWorkset(wkset_AD18[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateWorkset(wkset_AD24[block], block, grp, seedwhat, seedindex, override_transient);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateWorkset(wkset_AD32[block], block, grp, seedwhat, seedindex, override_transient);
  }
#endif
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorkset(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp,
                                          const int & seedwhat, const int & seedindex,
                                          const bool & override_transient) {
    
  //Teuchos::TimeMonitor localtimer(*computeSolnVolTimer);
  
  // Reset the residual and data in the workset
  //auto wset = wkset[block];
  wset->reset();
  
  wset->numElem = groups[block][grp]->numElem;
  this->updateGroupData(wset, block, grp);
  
  wset->wts = groups[block][grp]->wts;
  //wset->h = groups[block][grp]->hsize;
  vector<View_Sc2> ip = groups[block][grp]->getIntegrationPts();
  wset->setScalarField(ip[0],"x");
  if (ip.size() > 1) {
    wset->setScalarField(ip[1],"y");
  }
  if (ip.size() > 2) {
    wset->setScalarField(ip[2],"z");
  }

  // Update the integration info and basis in workset
  if (groups[block][grp]->storeAll || groups[block][grp]->group_data->use_basis_database) {
    wset->basis = groups[block][grp]->basis;
    wset->basis_grad = groups[block][grp]->basis_grad;
    wset->basis_div = groups[block][grp]->basis_div;
    wset->basis_curl = groups[block][grp]->basis_curl;
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricBasis(groups[block][grp]->group_data, groups[block][grp]->localElemID,
                                    tbasis, tbasis_grad, tbasis_curl,
                                    tbasis_div, tbasis_nodes);

    vector<CompressedView<View_Sc4>> tcbasis, tcbasis_grad, tcbasis_curl;
    vector<CompressedView<View_Sc3>> tcbasis_div;
    for (size_t i=0; i<tbasis.size(); ++i) {
      tcbasis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      tcbasis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
      tcbasis_div.push_back(CompressedView<View_Sc3>(tbasis_div[i]));
      tcbasis_curl.push_back(CompressedView<View_Sc4>(tbasis_curl[i]));
    }
    wset->basis = tcbasis;
    wset->basis_grad = tcbasis_grad;
    wset->basis_div = tcbasis_div;
    wset->basis_curl = tcbasis_curl;
  }
  
  // Map the gathered solution to seeded version in workset
  if (groups[block][grp]->group_data->requires_transient && !override_transient) {
    for (size_t set=0; set<groups[block][grp]->group_data->num_sets; ++set) {
      if (groups[block][grp]->have_sols) {
        wset->computeSolnTransientSeeded(set, groups[block][grp]->sol[set], groups[block][grp]->sol_prev[set],
                                         groups[block][grp]->sol_stage[set], seedwhat, seedindex);
      }
      else {
        wset->computeSolnTransientSeeded(set, groupData[block]->sol[set], groupData[block]->sol_prev[set],
                                         groupData[block]->sol_stage[set], seedwhat, seedindex);
      }
    }
  }
  else { // steady-state
    for (size_t set=0; set<groups[block][grp]->group_data->num_sets; ++set) {
      if (groups[block][grp]->have_sols) {
        wset->computeSolnSteadySeeded(set, groups[block][grp]->sol[set], seedwhat);
      }
      else {
        wset->computeSolnSteadySeeded(set, groupData[block]->sol[set], seedwhat);
      }
    }
  }
  if (wset->numDiscParams > 0) {
    if (groups[block][grp]->have_sols) {
      wset->computeParamSteadySeeded(groups[block][grp]->param, seedwhat);
    }
    else {
      wset->computeParamSteadySeeded(groupData[block]->param, seedwhat);
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetFace(const int & block, const size_t & grp,
                                          const size_t & facenum) {
  this->updateWorksetFace<ScalarT>(block, grp, facenum);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateWorksetFaceAD(const int & block, const size_t & grp,
                                          const size_t & facenum) {
  this->updateWorksetFace<AD>(block, grp, facenum);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetFace(const int & block, const size_t & grp,
                                          const size_t & facenum) {

  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateWorksetFace(wkset[block], block, grp, facenum);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateWorksetFace(wkset_AD[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateWorksetFace(wkset_AD2[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateWorksetFace(wkset_AD4[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateWorksetFace(wkset_AD8[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateWorksetFace(wkset_AD16[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateWorksetFace(wkset_AD18[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateWorksetFace(wkset_AD24[block], block, grp, facenum);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateWorksetFace(wkset_AD32[block], block, grp, facenum);
  }
#endif
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateWorksetFace(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp, const size_t & facenum) {
  
  // IMPORANT NOTE: This function assumes that face contributions are computing IMMEDIATELY after the
  // volumetric contributions, which implies that the seeded solution in the workset is already
  // correct for this Group.  There is currently no use case where this assumption is false.
  
  //Teuchos::TimeMonitor localtimer(*computeSolnFaceTimer);
  
  wset->wts_side = groups[block][grp]->wts_face[facenum];
  //wset->h = groups[block][grp]->hsize;

  wset->setScalarField(groups[block][grp]->ip_face[facenum][0],"x");
  wset->setScalarField(groups[block][grp]->normals_face[facenum][0],"n[x]");
  if (groups[block][grp]->ip_face[facenum].size() > 1) {
    wset->setScalarField(groups[block][grp]->ip_face[facenum][1],"y");
    wset->setScalarField(groups[block][grp]->normals_face[facenum][1],"n[y]");
  }
  if (groups[block][grp]->ip_face[facenum].size() > 2) {
    wset->setScalarField(groups[block][grp]->ip_face[facenum][2],"z");
    wset->setScalarField(groups[block][grp]->normals_face[facenum][2],"n[z]");
  }
    
  // Update the face integration points and basis in workset
  if (groups[block][grp]->storeAll || groupData[block]->use_basis_database) {
    wset->basis_side = groups[block][grp]->basis_face[facenum];
    wset->basis_grad_side = groups[block][grp]->basis_grad_face[facenum];
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad;
  
    disc->getPhysicalFaceBasis(groupData[block], facenum, groups[block][grp]->localElemID,
                               tbasis, tbasis_grad);
    vector<CompressedView<View_Sc4>> tcbasis, tcbasis_grad;
    for (size_t i=0; i<tbasis.size(); ++i) {
      tcbasis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      tcbasis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
    }
    wset->basis_side = tcbasis;
    wset->basis_grad_side = tcbasis_grad;
  }
  
  wset->resetSolutionFields();
  
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateStage(const int & stage, const ScalarT & current_time,
                                        const ScalarT & deltat) {

  for (size_t block=0; block<wkset.size(); ++block) {
    groupData[block]->current_stage = stage;
    this->updateStage(wkset[block], stage, current_time, deltat);
#ifndef MrHyDE_NO_AD
    if (type_AD == -1) {
      this->updateStage(wkset_AD[block], stage, current_time, deltat);
    }
    else if (type_AD == 2) {
      this->updateStage(wkset_AD2[block], stage, current_time, deltat);
    }
    else if (type_AD == 4) {
      this->updateStage(wkset_AD4[block], stage, current_time, deltat);
    }
    else if (type_AD == 8) {
      this->updateStage(wkset_AD8[block], stage, current_time, deltat);
    }
    else if (type_AD == 16) {
      this->updateStage(wkset_AD16[block], stage, current_time, deltat);
    }
    else if (type_AD == 18) {
      this->updateStage(wkset_AD18[block], stage, current_time, deltat);
    }
    else if (type_AD == 24) {
      this->updateStage(wkset_AD24[block], stage, current_time, deltat);
    }
    else if (type_AD == 32) {
      this->updateStage(wkset_AD32[block], stage, current_time, deltat);
    }
#endif
    
  }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateStage(Teuchos::RCP<Workset<EvalT> > & wset, const int & stage, const ScalarT & current_time,
                                        const ScalarT & deltat) {
  wset->setStage(stage);
  auto butcher_c = Kokkos::create_mirror_view(wset->butcher_c);
  Kokkos::deep_copy(butcher_c, wset->butcher_c);
  ScalarT timeval = current_time + butcher_c(stage)*deltat;
  wset->setTime(timeval);
  wset->setDeltat(deltat);
  wset->alpha = 1.0/deltat;

}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateTimeStep(const int & timestep) {
  for (size_t block=0; block<wkset.size(); ++block) {
    wkset[block]->time_step = timestep;
#ifndef MrHyDE_NO_AD
    if (type_AD == -1) {
      wkset_AD[block]->time_step = timestep;
    }
    else if (type_AD == 2) {
      wkset_AD2[block]->time_step = timestep;
    }
    else if (type_AD == 4) {
      wkset_AD4[block]->time_step = timestep;
    }
    else if (type_AD == 8) {
      wkset_AD8[block]->time_step = timestep;
    }
    else if (type_AD == 16) {
      wkset_AD16[block]->time_step = timestep;
    }
    else if (type_AD == 18) {
      wkset_AD18[block]->time_step = timestep;
    }
    else if (type_AD == 24) {
      wkset_AD24[block]->time_step = timestep;
    }
    else if (type_AD == 32) {
      wkset_AD32[block]->time_step = timestep;
    }
#endif
  }
}
    
/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updatePhysicsSet(const size_t & set) {
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    
    if (wkset[block]->isInitialized) {
      wkset[block]->updatePhysicsSet(set);
      groupData[block]->updatePhysicsSet(set);
    }
#ifndef MrHyDE_NO_AD
    if (type_AD == -1) {
      wkset_AD[block]->updatePhysicsSet(set);
    }
    else if (type_AD == 2) {
      wkset_AD2[block]->updatePhysicsSet(set);
    }
    else if (type_AD == 4) {
      wkset_AD4[block]->updatePhysicsSet(set);
    }
    else if (type_AD == 8) {
      wkset_AD8[block]->updatePhysicsSet(set);
    }
    else if (type_AD == 16) {
      wkset_AD16[block]->updatePhysicsSet(set);
    }
    else if (type_AD == 18) {
      wkset_AD18[block]->updatePhysicsSet(set);
    }
    else if (type_AD == 24) {
      wkset_AD24[block]->updatePhysicsSet(set);
    }
    else if (type_AD == 32) {
      wkset_AD32[block]->updatePhysicsSet(set);
    }
#endif
  }
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::computeBoundaryAux(const int & block, const size_t & grp, const int & seedwhat) {

#ifndef MrHyDE_NO_AD
  if (type_AD == -1) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD[block]);
  }
  else if (type_AD == 2) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD2[block]);
  }
  else if (type_AD == 4) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD4[block]);
  }
  else if (type_AD == 8) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD8[block]);
  }
  else if (type_AD == 16) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD16[block]);
  }
  else if (type_AD == 18) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD18[block]);
  }
  else if (type_AD == 24) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD24[block]);
  }
  else if (type_AD == 32) {
    this->computeBoundaryAux(block, grp, seedwhat, wkset_AD32[block]);
  }

#endif
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::computeBoundaryAux(const int & block, const size_t & grp, const int & seedwhat,
                                               Teuchos::RCP<Workset<EvalT> > & wset) {

#ifndef MrHyDE_NO_AD
  auto numAuxDOF = groupData[block]->num_aux_dof;
    
  for (size_type var=0; var<numAuxDOF.extent(0); var++) {
    auto abasis = boundary_groups[block][grp]->auxside_basis[boundary_groups[block][grp]->auxusebasis[var]];
    auto off = subview(boundary_groups[block][grp]->auxoffsets,var,ALL());
    string varname = wset->aux_varlist[var];
    auto local_aux = wset->getSolutionField("aux "+varname,false);
    Kokkos::deep_copy(local_aux,0.0);
    auto localID = boundary_groups[block][grp]->localElemID;
    auto varaux = subview(boundary_groups[block][grp]->aux, ALL(), var, ALL());
    if (seedwhat == 4) {
      parallel_for("bgroup aux 4",
                   TeamPolicy<AssemblyExec>(localID.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        EvalT dummyval = 0.0;
        for (size_type pt=team.team_rank(); pt<abasis.extent(2); pt+=team.team_size() ) {
          for (size_type dof=0; dof<abasis.extent(1); ++dof) {
            EvalT auxval = EvalT(dummyval.size(),off(dof), varaux(localID(elem),dof));
            local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
          }
        }
      });
    }
    else {
      parallel_for("bgroup aux 5",
                    TeamPolicy<AssemblyExec>(localID.extent(0), Kokkos::AUTO, VECTORSIZE),
                    KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<abasis.extent(2); pt+=team.team_size() ) {
          for (size_type dof=0; dof<abasis.extent(1); ++dof) {
            ScalarT auxval = varaux(localID(elem),dof);
            local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
          }
        }
      });
    }
  }
#endif

}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateDataBoundaryAD(const int & block, const size_t & grp) {
  this->updateDataBoundary<AD>(block, grp);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
void AssemblyManager<Node>::updateDataBoundary(const int & block, const size_t & grp) {
  this->updateDataBoundary<ScalarT>(block, grp);
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateDataBoundary(const int & block, const size_t & grp) {

  if (std::is_same<EvalT, ScalarT>::value) {
    this->updateDataBoundary(wkset[block], block, grp);
  }
#ifndef MrHyDE_NO_AD
  else if (std::is_same<EvalT, AD>::value) {
    this->updateDataBoundary(wkset_AD[block], block, grp);
  }
  else if (std::is_same<EvalT, AD2>::value) {
    this->updateDataBoundary(wkset_AD2[block], block, grp);
  }
  else if (std::is_same<EvalT, AD4>::value) {
    this->updateDataBoundary(wkset_AD4[block], block, grp);
  }
  else if (std::is_same<EvalT, AD8>::value) {
    this->updateDataBoundary(wkset_AD8[block], block, grp);
  }
  else if (std::is_same<EvalT, AD16>::value) {
    this->updateDataBoundary(wkset_AD16[block], block, grp);
  }
  else if (std::is_same<EvalT, AD18>::value) {
    this->updateDataBoundary(wkset_AD18[block], block, grp);
  }
  else if (std::is_same<EvalT, AD24>::value) {
    this->updateDataBoundary(wkset_AD24[block], block, grp);
  }
  else if (std::is_same<EvalT, AD32>::value) {
    this->updateDataBoundary(wkset_AD32[block], block, grp);
  }
#endif
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////

template<class Node>
template<class EvalT>
void AssemblyManager<Node>::updateDataBoundary(Teuchos::RCP<Workset<EvalT> > & wset,
                                               const int & block, const size_t & grp) {

  // hard coded for what I need it for right now
  if (groupData[block]->have_phi) {
    wset->have_rotation_phi = true;
    wset->rotation_phi = boundary_groups[block][grp]->data;
    wset->allocateRotations();
  }
  else if (groupData[block]->have_rotation) {
    wset->have_rotation = true;
    wset->allocateRotations();
    auto rot = wset->rotation;
    auto data = boundary_groups[block][grp]->data;
    parallel_for("update data",
                 RangePolicy<AssemblyExec>(0,data.extent(0)),
                 KOKKOS_LAMBDA (const size_type e ) {
      rot(e,0,0) = data(e,0);
      rot(e,0,1) = data(e,1);
      rot(e,0,2) = data(e,2);
      rot(e,1,0) = data(e,3);
      rot(e,1,1) = data(e,4);
      rot(e,1,2) = data(e,5);
      rot(e,2,0) = data(e,6);
      rot(e,2,1) = data(e,7);
      rot(e,2,2) = data(e,8);
    });
  
  }
  else if (groupData[block]->have_extra_data) {
    wset->extra_data = boundary_groups[block][grp]->data;
  }
  wset->multidata = boundary_groups[block][grp]->multidata;
}

