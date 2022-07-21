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

#include "multiscaleManager.hpp"
#include "split_mpi_communicators.hpp"
#include "subgridDtN.hpp"
#include "subgridDtN2.hpp"
#include <random>

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

MultiscaleManager::MultiscaleManager(const Teuchos::RCP<MpiComm> & MacroComm_,
                                     Teuchos::RCP<MeshInterface> & mesh_,
                                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                     vector<vector<Teuchos::RCP<Group> > > & groups_,
                                     vector<Teuchos::RCP<FunctionManager> > macro_functionManagers_ ) :
MacroComm(MacroComm_), settings(settings_), groups(groups_), macro_functionManagers(macro_functionManagers_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debug_level = settings->get<int>("debug level",0);
  verbosity = settings->get<int>("verbosity",0);
  if (debug_level > 0) {
    if (MacroComm->getRank() == 0) {
      cout << "**** Starting MultiscaleManager manager constructor ..." << endl;
    }
  }
  
  // Create subcommunicators for the subgrid models (this isn't really used much)
  Teuchos::RCP<MpiComm> unusedComm;
  SplitComm(settings, *MacroComm, unusedComm, Comm);
  ml_training = true;
  max_training_steps = settings->sublist("Solver").get<int>("max subgrid ML training steps",10);
  num_training_steps = 0;
  have_ml_models = settings->sublist("Solver").get<bool>("have ML models",false);

  if (settings->isSublist("Subgrid")) {
    
    string subgrid_selection = settings->sublist("Solver").get<string>("subgrid model selection","user defined");
    if (subgrid_selection == "user defined") {
      subgrid_model_selection = 0;
    }
    else if (subgrid_selection == "hierarchical") {
      subgrid_model_selection = 1;
    }
    else if (subgrid_selection == "ML") {
      subgrid_model_selection = 2;
    }

    reltol = settings->sublist("Solver").get<ScalarT>("subgrid error tolerance",1.0e-6);
    abstol = settings->sublist("Solver").get<ScalarT>("subgrid absolute error tolerance",1.0e-12);
    vector<Teuchos::RCP<Teuchos::ParameterList> > subgrid_model_pls;
    
    bool single_model = false;
    Teuchos::ParameterList::ConstIterator sub_itr = settings->sublist("Subgrid").begin();
    while (sub_itr != settings->sublist("Subgrid").end()) {
      if (sub_itr->first == "Mesh") {
        single_model = true;
      }
      sub_itr++;
    }
    if (single_model) {
      Teuchos::RCP<Teuchos::ParameterList> subgrid_pl = rcp(new Teuchos::ParameterList("Subgrid"));
      subgrid_pl->setParameters(settings->sublist("Subgrid"));
      subgrid_model_pls.push_back(subgrid_pl);
    }
    else {
      Teuchos::ParameterList::ConstIterator sub_itr = settings->sublist("Subgrid").begin();
      while (sub_itr != settings->sublist("Subgrid").end()) {
        if (settings->sublist("Subgrid").isSublist(sub_itr->first)) {
          Teuchos::RCP<Teuchos::ParameterList> subgrid_pl = rcp(new Teuchos::ParameterList(sub_itr->first));
          subgrid_pl->setParameters(settings->sublist("Subgrid").sublist(sub_itr->first));
          subgrid_model_pls.push_back(subgrid_pl);
        }
        sub_itr++;
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the subgrid models specified in the input file
    ////////////////////////////////////////////////////////////////////////////////
    
    if (single_model) {
      Teuchos::RCP<Teuchos::ParameterList> subgrid_pl = subgrid_model_pls[0];
      string subgrid_model_type = subgrid_pl->get<string>("subgrid model","DtN2");
      string macro_block_name = subgrid_pl->get<string>("macro block","eblock-0_0_0");
      std::vector<string> macro_blocknames;
      //mesh_->stk_mesh->getElementBlockNames(macro_blocknames);
      macro_blocknames = mesh_->block_names;
      int macro_block = 0; // default to single block case
      for (size_t m=0; m<macro_blocknames.size(); ++m) {
        if (macro_blocknames[m] == macro_block_name) {
          macro_block = m;
        }
      }
      topo_RCP macro_topo = mesh_->cellTopo[macro_block]; //mesh_->stk_mesh->getCellTopology(macro_blocknames[macro_block]);
      if (subgrid_model_type == "DtN") {
        subgridModels.push_back(Teuchos::rcp( new SubGridDtN(Comm, subgrid_pl, macro_topo) ) );
      }
      else if (subgrid_model_type == "Explicit FEM") {
        // not implemented
      }
      else if (subgrid_model_type == "DtN2") {
        subgridModels.push_back(Teuchos::rcp( new SubGridDtN2(Comm, subgrid_pl, macro_topo) ) );
      }
      subgridModels[subgridModels.size()-1]->macro_block = macro_block;
      subgridModels[subgridModels.size()-1]->usage = "1.0";
    }
    else {
      for (size_t j=0; j<subgrid_model_pls.size(); j++) {
        Teuchos::RCP<Teuchos::ParameterList> subgrid_pl = subgrid_model_pls[j];
        string subgrid_model_type = subgrid_pl->get<string>("subgrid model","DtN2");
        string macro_block_name = subgrid_pl->get<string>("macro block","eblock-0_0_0");
        std::vector<string> macro_blocknames;
        //mesh_->stk_mesh->getElementBlockNames(macro_blocknames);
        macro_blocknames = mesh_->block_names;
        int macro_block = 0; // default to single block case
        for (size_t m=0; m<macro_blocknames.size(); ++m) {
          if (macro_blocknames[m] == macro_block_name) {
            macro_block = m;
          }
        }
        topo_RCP macro_topo = mesh_->cellTopo[macro_block]; //stk_mesh->getCellTopology(macro_blocknames[macro_block]);
        
        if (subgrid_model_type == "DtN") {
          subgridModels.push_back(Teuchos::rcp( new SubGridDtN(Comm, subgrid_pl, macro_topo) ) );
        }
        else if (subgrid_model_type == "Explicit FEM") {
          // not implemented
        }
        else if (subgrid_model_type == "DtN2") {
          subgridModels.push_back(Teuchos::rcp( new SubGridDtN2(Comm, subgrid_pl, macro_topo) ) );
        }
        subgridModels[subgridModels.size()-1]->macro_block = macro_block;
        string usage;
        if (j==0) {// to enable default behavior
          usage = subgrid_pl->get<string>("usage","1.0");
        }
        else {
          usage = subgrid_pl->get<string>("usage","0.0");
        }
        subgridModels[subgridModels.size()-1]->usage = usage;
      }
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the subgrid models specified in the input file
    ////////////////////////////////////////////////////////////////////////////////
    
    subgrid_static = settings->sublist("Subgrid").get<bool>("static subgrids",true);
    for (size_t n=0; n<subgridModels.size(); n++) {
      std::stringstream ss;
      ss << n;
      int macro_block = subgridModels[n]->macro_block;
      //macro_functionManagers[macro_block]->addFunction("Subgrid " + ss.str() + " usage",subgridModels[n]->usage, "ip");
      macro_functionManagers[macro_block]->addFunction(subgridModels[n]->name + " usage",subgridModels[n]->usage, "ip");
    }
    
  }
  else {
    subgrid_static = true;
  }
  
  if (debug_level > 0) {
    if (MacroComm->getRank() == 0) {
      cout << "**** Finished MultiscaleManager manager constructor" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Set the information from the macro-scale that does not depend on the specific group
////////////////////////////////////////////////////////////////////////////////

void MultiscaleManager::setMacroInfo(vector<vector<basis_RCP> > & macro_basis_pointers,
                                     vector<vector<string> > & macro_basis_types,
                                     vector<vector<string> > & macro_varlist,
                                     vector<vector<int> > macro_usebasis,
                                     vector<vector<vector<int> > > & macro_offsets,
                                     vector<Kokkos::View<int*,AssemblyDevice>> & macro_numDOF,
                                     vector<string> & macro_paramnames,
                                     vector<string> & macro_disc_paramnames) {
  
  for (size_t j=0; j<subgridModels.size(); j++) {
    int mblock = subgridModels[j]->macro_block;
    subgridModels[j]->macro_basis_pointers = macro_basis_pointers[mblock];
    subgridModels[j]->macro_basis_types = macro_basis_types[mblock];
    subgridModels[j]->macro_varlist = macro_varlist[mblock];
    subgridModels[j]->macro_usebasis = macro_usebasis[mblock];
    subgridModels[j]->macro_offsets = macro_wkset[mblock]->offsets;
    subgridModels[j]->macro_numDOF = macro_numDOF[mblock];
    subgridModels[j]->macro_paramnames = macro_paramnames;
    subgridModels[j]->macro_disc_paramnames = macro_disc_paramnames;
    subgridModels[j]->subgrid_static = subgrid_static;
    subgridModels[j]->macrosidenames = groups[0][0]->groupData->sidenames;
  }
  
}

////////////////////////////////////////////////////////////////////////////////
// Initial assignment of subgrid models to groups
////////////////////////////////////////////////////////////////////////////////

ScalarT MultiscaleManager::initialize() {
  
  Teuchos::TimeMonitor localtimer(*initializetimer);
  
  if (debug_level > 0) {
    if (MacroComm->getRank() == 0) {
      cout << "**** Starting MultiscaleManager manager initialize" << endl;
    }
  }
  ScalarT my_cost = 0.0;
  size_t numusers = 0;
  for (size_t block=0; block<groups.size(); ++block) {
    
    bool uses_subgrid = false;
    for (size_t s=0; s<subgridModels.size(); s++) {
      if (subgridModels[s]->macro_block == block) {
        uses_subgrid = true;
      }
    }
    if (uses_subgrid) {
      for (size_t grp=0; grp<groups[block].size(); ++grp) {
        
        groups[block][grp]->updateWorkset(0,0);
        
        vector<int> sgvotes(subgridModels.size(),0);
        
        for (size_t s=0; s<subgridModels.size(); s++) {
          if (subgridModels[s]->macro_block == block) {
            std::stringstream ss;
            ss << s;
            auto usagecheck = macro_functionManagers[block]->evaluate(subgridModels[s]->name + " usage","ip");
            
            Kokkos::View<ScalarT**,AssemblyDevice> usagecheck_tmp("temp usage check",
                                                                  macro_functionManagers[block]->numElem,
                                                                  macro_functionManagers[block]->numip);
                                                                  
            parallel_for("assembly copy LIDs",
                         RangePolicy<AssemblyExec>(0,usagecheck_tmp.extent(0)),
                         KOKKOS_LAMBDA (const int i ) {
              for (size_type j=0; j<usagecheck_tmp.extent(1); j++) {
#ifndef MrHyDE_NO_AD
                usagecheck_tmp(i,j) = usagecheck(i,j).val();
#else
                usagecheck_tmp(i,j) = usagecheck(i,j);
#endif
              }
            });
            
            auto host_usagecheck = Kokkos::create_mirror_view(usagecheck_tmp);
            Kokkos::deep_copy(host_usagecheck, usagecheck_tmp);
            for (size_t p=0; p<groups[block][grp]->numElem; p++) {
              for (size_t j=0; j<host_usagecheck.extent(1); j++) {
                if (host_usagecheck(p,j) >= 1.0) {
                  sgvotes[s] += 1;
                }
              }
            }
          }
        }
        
        int maxvotes = -1;
        int sgwinner = 0;
        for (size_t i=0; i<sgvotes.size(); i++) {
          if (sgvotes[i] >= maxvotes) {
            maxvotes = sgvotes[i];
            sgwinner = i;
          }
        }

        size_t sgusernum = 0;
        if (subgrid_static && subgrid_model_selection == 0) { // only add each group to one subgrid model
          
          sgusernum = subgridModels[sgwinner]->addMacro(groups[block][grp]->nodes,
                                                        groups[block][grp]->sideinfo[0],
                                                        groups[block][grp]->LIDs[0],
                                                        groups[block][grp]->orientation);
          
        }
        else {
          for (size_t s=0; s<subgridModels.size(); s++) { // needs to add this group info to all of them (sgusernum is same for all)
            sgusernum = subgridModels[s]->addMacro(groups[block][grp]->nodes,
                                                   groups[block][grp]->sideinfo[0],
                                                   groups[block][grp]->LIDs[0],
                                                   groups[block][grp]->orientation);
          }
        }
        groups[block][grp]->subgridModels = subgridModels;
        groups[block][grp]->subgrid_model_index = sgwinner;
        groups[block][grp]->subgrid_usernum = sgusernum;
        groups[block][grp]->groupData->multiscale = true;
        my_cost = subgridModels[sgwinner]->cost_estimate * groups[block][grp]->numElem;
        numusers += 1;
      }
    }
  }
  
  std::vector<std::string> appends;
  if (settings->sublist("Analysis").get<std::string>("analysis type","forward") == "UQ") {
    if (settings->sublist("Postprocess").get("write subgrid solution",false)) {
      int numsamples = settings->sublist("Analysis").sublist("UQ").get<int>("samples",100);
      for (int j=0; j<numsamples; ++j) {
        std::stringstream ss;
        ss << "_" << j;
        appends.push_back(ss.str());
      }
    }
    else {
      appends = {""};
    }
  }
  else {
    appends = {""};
  }

  bool write_subgrid_soln = settings->sublist("Postprocess").get<bool>("write subgrid solution",false);
  for (size_t s=0; s<subgridModels.size(); s++) {
    subgridModels[s]->finalize(MacroComm->getSize(), MacroComm->getRank(), write_subgrid_soln, appends);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // If the subgrid models are not static, then we need projection maps between
  // the various subgrid models.
  // Since we only store N subgrid models, we only require (N-1)^2 maps
  ////////////////////////////////////////////////////////////////////////////////
  
  if (!subgrid_static) {
    
    for (size_t s=0; s<subgridModels.size(); s++) {
      if (subgrid_model_selection == 0) {
        vector<bool> active(numusers,false);
        size_t numactive = 0;
        for (size_t block=0; block<groups.size(); ++block) {
          for (size_t grp=0; grp<groups[block].size(); ++grp) {
            if (groups[block][grp]->subgrid_model_index == s) {
              size_t usernum = groups[block][grp]->subgrid_usernum;
              active[usernum] = true;
              numactive += 1;
            }
          }
        }
        subgridModels[s]->updateActive(active);
      }
      else {
        vector<bool> active(numusers,true);
        subgridModels[s]->updateActive(active);
      }
    }
    
    for (size_t i=0; i<subgridModels.size(); i++) {
      auto ip = subgridModels[i]->getIP();
      auto wts = subgridModels[i]->getIPWts();
      std::pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > basisinfo_i = subgridModels[i]->evaluateBasis2(ip);
      vector<Teuchos::RCP<SGLA_CrsMatrix> > currmaps;
      for (size_t j=0; j<subgridModels.size(); j++) {
        std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo_j = subgridModels[j]->evaluateBasis2(ip);
        Teuchos::RCP<SGLA_CrsMatrix> map = subgridModels[i]->getProjectionMatrix(ip, wts, subgridModels[j]->owned_map, subgridModels[j]->overlapped_map, basisinfo_j);
        currmaps.push_back(map);
      }
      subgrid_projection_maps.push_back(currmaps);
    }
    
    for (size_t i=0; i<subgridModels.size(); i++) {
      vector_RCP dummy_vec = subgridModels[i]->getVector();
      vector_RCP dummy_vec2 = subgridModels[i]->getVector();
      Teuchos::RCP<Amesos2::Solver<SGLA_CrsMatrix,SGLA_MultiVector> > Am2Solver = Amesos2::create<SGLA_CrsMatrix,SGLA_MultiVector>("KLU2",subgrid_projection_maps[i][i], dummy_vec, dummy_vec2);
      Am2Solver->symbolicFactorization();
      Am2Solver->numericFactorization();
      subgrid_projection_solvers.push_back(Am2Solver);
    }
  }
  
  // add mesh data
  
  for (size_t s=0; s< subgridModels.size(); s++) {
    subgridModels[s]->addMeshData();
  }
  
  if (debug_level > 0) {
    if (MacroComm->getRank() == 0) {
      cout << "**** Finished MultiscaleManager manager initialize" << endl;
    }
  }
  
  //subgrid_static = true;
  return my_cost;
}

////////////////////////////////////////////////////////////////////////////////
// Re-assignment of subgrid models to groups
////////////////////////////////////////////////////////////////////////////////

void MultiscaleManager::update() {
  
  Teuchos::TimeMonitor localtimer(*updatetimer);
  
  ScalarT my_cost = 1.0;
  
  if (subgridModels.size() > 0) {
  
    if (subgrid_static) {
      for (size_t block=0; block<groups.size(); ++block) {
        for (size_t grp=0; grp<groups[block].size(); ++grp) {
          if (groups[block][grp]->groupData->multiscale) {
            int currmodel = groups[block][grp]->subgrid_model_index;
            my_cost += subgridModels[currmodel]->cost_estimate * groups[block][grp]->numElem;
          }
        }
      }
    }
    else if (subgrid_model_selection == 0){
      for (size_t block=0; block<groups.size(); ++block) {
        for (size_t grp=0; grp<groups[block].size(); ++grp) {
          if (groups[block][grp]->groupData->multiscale) {
          
            groups[block][grp]->updateWorkset(0,0);
          
            vector<int> sgvotes(subgridModels.size(),0);
          
            for (size_t s=0; s<subgridModels.size(); s++) {
              if (subgridModels[s]->macro_block == block) {
                std::stringstream ss;
                ss << s;
                auto usagecheck = macro_functionManagers[block]->evaluate(subgridModels[s]->name + " usage","ip");
                Kokkos::View<ScalarT**,AssemblyDevice> usagecheck_tmp("temp usage check",
                                                                      macro_functionManagers[block]->numElem,
                                                                      macro_functionManagers[block]->numip);
                                                                    
                parallel_for("assembly copy LIDs",
                             RangePolicy<AssemblyExec>(0,usagecheck_tmp.extent(0)),
                             KOKKOS_LAMBDA (const int i ) {
                  for (size_type j=0; j<usagecheck_tmp.extent(1); j++) {
#ifndef MrHyDE_NO_AD
                    usagecheck_tmp(i,j) = usagecheck(i,j).val();
#else
                    usagecheck_tmp(i,j) = usagecheck(i,j);
#endif
                  }
                });
              
                auto host_usagecheck = Kokkos::create_mirror_view(usagecheck_tmp);
                Kokkos::deep_copy(host_usagecheck, usagecheck_tmp);
              
                for (size_t p=0; p<groups[block][grp]->numElem; p++) {
                  for (size_t j=0; j<host_usagecheck.extent(1); j++) {
                    if (host_usagecheck(p,j) >= 1.0) {
                      sgvotes[s] += 1;
                    }
                  }
                }
              }
            }
            int maxvotes = -1;
            int sgwinner = 0;
            for (size_t i=0; i<sgvotes.size(); i++) {
              if (sgvotes[i] >= maxvotes) {
                maxvotes = sgvotes[i];
                sgwinner = i;
              }
            }
          
            int oldmodel = groups[block][grp]->subgrid_model_index;
            if (sgwinner != oldmodel) {
            
              size_t usernum = groups[block][grp]->subgrid_usernum;
          
              // get the time/solution from old subgrid model at last time step
              Teuchos::RCP<SGLA_MultiVector> lastsol = subgridModels[oldmodel]->prev_soln[usernum];
              
              Teuchos::RCP<SGLA_MultiVector> projvec = subgridModels[sgwinner]->getVector();
              subgrid_projection_maps[sgwinner][oldmodel]->apply(*lastsol, *projvec);
            
              Teuchos::RCP<SGLA_MultiVector> newvec = subgridModels[sgwinner]->prev_soln[usernum];
              subgrid_projection_solvers[sgwinner]->setB(projvec);
              subgrid_projection_solvers[sgwinner]->setX(newvec);
              subgrid_projection_solvers[sgwinner]->solve();
            
              ScalarT ptime = subgridModels[oldmodel]->getPreviousTime();
              subgridModels[sgwinner]->setPreviousTime(ptime);
            }
            my_cost += subgridModels[sgwinner]->cost_estimate * groups[block][grp]->numElem;
            groups[block][grp]->subgrid_model_index = sgwinner;
          
          }
        }
      }
      
      for (size_t s=0; s<subgridModels.size(); s++) {
        vector<bool> active(subgridModels[s]->active.size(),false);
        size_t numactive = 0;
        for (size_t block=0; block<groups.size(); ++block) {
          for (size_t grp=0; grp<groups[block].size(); ++grp) {
            if (groups[block][grp]->subgrid_model_index == s) {
              size_t usernum = groups[block][grp]->subgrid_usernum;
              active[usernum] = true;
              numactive += 1;
            }
          }
        }
        subgridModels[s]->updateActive(active);
      }
    }
    else if (subgrid_model_selection == 1) {
      // nothing to do here
    }
    else if (subgrid_model_selection == 2) {
      if (num_training_steps < max_training_steps) {
        ++num_training_steps;
      }
      else if (have_ml_models) {

        // generate new data as input for the ML model
        std::stringstream inss;
        inss << "new_input_";
        inss << MacroComm->getRank();
        string infile = inss.str() + ".txt";
          
        std::ofstream inputOUT;
        bool is_open = false;
        int attempts = 0;
        int max_attempts = 10;
        while (!is_open && attempts < max_attempts) {
          inputOUT.open(infile);
          is_open = inputOUT.is_open();
          attempts++;
        }
        inputOUT.precision(12);
        
        int set = 0; // hard coded for now
        for (size_t block=0; block<groups.size(); ++block) {
          for (size_t grp=0; grp<groups[block].size(); ++grp) {
            groups[block][grp]->wkset->reset();
  
            auto u_curr = groups[block][grp]->u[set];

            // Get the coarse time derivative
            bool include_timederiv = true;
            View_Sc3 udot_sc("coarse udot unseeded",u_curr.extent(0),u_curr.extent(1),u_curr.extent(2));
            if (include_timederiv) {              

              for (size_type var=0; var<u_curr.extent(1); ++var) {
            
                size_t uindex = groups[block][grp]->wkset->uvals_index[set][var];
                auto uvals_AD = groups[block][grp]->wkset->u_dotvals[uindex];
                auto udot_sc_sv = subview(udot_sc,ALL(),var,ALL());
                parallel_for("assembly compute coarse sol",
                             RangePolicy<AssemblyExec>(0,u_curr.extent(0)),
                             KOKKOS_LAMBDA (const size_type elem ) {
                  for (size_type dof=0; dof<uvals_AD.extent(1); ++dof) {
    #ifndef MrHyDE_NO_AD
                    udot_sc_sv(elem,dof) = uvals_AD(elem,dof).val();
    #else
                    udot_sc_sv(elem,dof) = uvals_AD(elem,dof);
    #endif
                  }
                }); 
              }
  
            }

            // Map the gathered solution to seeded version in workset
            if (groups[block][grp]->groupData->requiresTransient) {
              for (size_t iset=0; iset<groups[block][grp]->groupData->numSets; ++iset) {
                groups[block][grp]->wkset->computeSolnTransientSeeded(iset, groups[block][grp]->u[iset], groups[block][grp]->u_prev[iset], 
                                                                      groups[block][grp]->u_stage[iset], 0);
              }
            }
            else { // steady-state
              for (size_t iset=0; iset<groups[block][grp]->groupData->numSets; ++iset) {
                groups[block][grp]->wkset->computeSolnSteadySeeded(iset, groups[block][grp]->u[iset], 0);
              }
            }
          
            // Get the coarse state
            View_Sc3 uvals_sc("coarse vals unseeded",u_curr.extent(0),u_curr.extent(1),u_curr.extent(2));

            for (size_type var=0; var<u_curr.extent(1); ++var) {
            
              size_t uindex = groups[block][grp]->wkset->uvals_index[set][var];
              auto uvals_AD = groups[block][grp]->wkset->uvals[uindex];
              auto uvals_sc_sv = subview(uvals_sc,ALL(),var,ALL());
              parallel_for("assembly compute coarse sol",
                           RangePolicy<AssemblyExec>(0,u_curr.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                for (size_type dof=0; dof<uvals_AD.extent(1); ++dof) {
    #ifndef MrHyDE_NO_AD
                  uvals_sc_sv(elem,dof) = uvals_AD(elem,dof).val();
    #else
                  uvals_sc_sv(elem,dof) = uvals_AD(elem,dof);
    #endif
                }
              }); 
            }

            // Get the average x,y,z locations
            bool include_xyz = true;
            View_Sc2 avg_xyz("average spatial locations",u_curr.extent(0),groups[block][grp]->ip.size());
            if (include_xyz) {
              auto wts = groups[block][grp]->wts;
              View_Sc1 avg_wts("average wts",u_curr.extent(0));
              parallel_for("assembly compute coarse sol",
                           RangePolicy<AssemblyExec>(0,u_curr.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                for (size_type pt=0; pt<wts.extent(1); ++pt) {
                  avg_wts(elem) += wts(elem,pt);
                }
              });
              auto ip_x = groups[block][grp]->ip[0];
              auto ip_y = groups[block][grp]->ip[1];
              parallel_for("assembly compute coarse sol",
                           RangePolicy<AssemblyExec>(0,u_curr.extent(0)),
                           KOKKOS_LAMBDA (const size_type elem ) {
                for (size_type pt=0; pt<ip_x.extent(1); ++pt) {
                  avg_xyz(elem,0) = ip_x(elem,pt)*wts(elem,pt)/avg_wts(elem);
                  avg_xyz(elem,1) = ip_y(elem,pt)*wts(elem,pt)/avg_wts(elem);
                }
              });
            }
            // Write out the inputs
            for (size_type elem=0; elem<uvals_sc.extent(0); ++elem) {
                
              for (size_type var=0; var<uvals_sc.extent(1); ++var) {
                for (size_type dof=0; dof<uvals_sc.extent(2); ++dof) {
                  inputOUT << uvals_sc(elem,var,dof) << "  ";
                }
                if (include_timederiv) {
                  for (size_type dof=0; dof<udot_sc.extent(2); ++dof) {
                    inputOUT << udot_sc(elem,var,dof) << "  ";
                  }  
                }
                
              }
              if (include_xyz) {
                for (size_type dim=0; dim<avg_xyz.extent(1); ++dim) {
                  inputOUT << avg_xyz(elem,dim) << "  ";  
                }
              }
              inputOUT << endl;
            }
          }//grp
        }//block
        inputOUT.close();
          

        // Evaluate each of the ML models
        vector<vector<int> > sg_model_pred;
        for (size_t model=0; model<subgridModels.size()-1; ++model) {
          // Build the pytorch models
          string filename = "nn_predict.py";
          string command = settings->get<string>("python","python3");
          string name = subgridModels[model]->name;

          std::stringstream nnss;
          nnss << name;
          nnss << "_";
          nnss << MacroComm->getRank();
          string nnfile = name + "_nn.pt";
          string predfile = nnss.str() + "_pred.txt";

          command += " ";
          command += filename;
          command += " --model " + nnfile;
          command += " --data " + infile;
          command += " --predictions " + predfile;
              
          cout << "========== Calling: " << command << endl;
          system(command.c_str());
          cout << "========== Done" << endl;

          vector<int> nn_pred;
          std::ifstream nn_out(predfile);
          std::string line;
          while (std::getline(nn_out,line)){
            std::istringstream tmp(line);
            int val;
            tmp >> val;
            nn_pred.push_back(val);
          }
          nn_out.close();
          sg_model_pred.push_back(nn_pred);
        }
        
        //size_t num_pred = sg_model_pred[0].size();
        
        //for (size_t j=0; j<num_pred; ++j) {
        //  for (size_t k=0; k<sg_model_pred.size(); ++k) {
        //    cout << sg_model_pred[k][j] << "  ";
        //  }
        //  cout << endl;
        //}
        // Figure out which subgrid each group should use
        size_t prog = 0;
        for (size_t block=0; block<groups.size(); ++block) {
          for (size_t grp=0; grp<groups[block].size(); ++grp) {
            int numElem = groups[block][grp]->numElem;
            bool found = false;
            size_t sgtest = 0;
            int sgwinner = 0;
            while (!found) {
              bool ok = true;
              for (int elem=0; elem<numElem; ++elem) {
                if (sg_model_pred[sgtest][prog+elem] == 0) {
                  ok = false;
                }
              }
              if (ok) {
                found = true;
                sgwinner = sgtest;
              }
              else if (sgtest == sg_model_pred.size()-1) {
                found = true;
                sgwinner = sg_model_pred.size();
              }
              else {
                sgtest++;
              }
            }
            prog += numElem;

            int oldmodel = groups[block][grp]->subgrid_model_index;
            if (sgwinner != oldmodel) {
            
              size_t usernum = groups[block][grp]->subgrid_usernum;
          
              // get the time/solution from old subgrid model at last time step
              Teuchos::RCP<SGLA_MultiVector> lastsol = subgridModels[oldmodel]->prev_soln[usernum];
              
              Teuchos::RCP<SGLA_MultiVector> projvec = subgridModels[sgwinner]->getVector();
              subgrid_projection_maps[sgwinner][oldmodel]->apply(*lastsol, *projvec);
            
              Teuchos::RCP<SGLA_MultiVector> newvec = subgridModels[sgwinner]->prev_soln[usernum];
              subgrid_projection_solvers[sgwinner]->setB(projvec);
              subgrid_projection_solvers[sgwinner]->setX(newvec);
              subgrid_projection_solvers[sgwinner]->solve();
            
              ScalarT ptime = subgridModels[oldmodel]->getPreviousTime();
              subgridModels[sgwinner]->setPreviousTime(ptime);
            }
            my_cost += subgridModels[sgwinner]->cost_estimate * groups[block][grp]->numElem;
            groups[block][grp]->subgrid_model_index = sgwinner;
          }
        }

        for (size_t s=0; s<subgridModels.size(); s++) {
        vector<bool> active(subgridModels[s]->active.size(),false);
        size_t numactive = 0;
        for (size_t block=0; block<groups.size(); ++block) {
          for (size_t grp=0; grp<groups[block].size(); ++grp) {
            if (groups[block][grp]->subgrid_model_index == s) {
              size_t usernum = groups[block][grp]->subgrid_usernum;
              active[usernum] = true;
              numactive += 1;
            }
          }
        }
        subgridModels[s]->updateActive(active);
      }
    }
    else {
      have_ml_models = true;
      ml_training = false;
      for (size_t model=0; model<subgridModels.size()-1; ++model) {

        string name = subgridModels[model]->name;

        ScalarT mean = 0.0, meansqr = 0.0;
        ScalarT numdata = (ScalarT)ml_model_extradata[model].size();
        for (size_t d=0; d<ml_model_extradata[model].size(); ++d) {
          mean += ml_model_extradata[model][d]/numdata;
          meansqr += ml_model_extradata[model][d]*ml_model_extradata[model][d]/numdata;
        }
        ScalarT var = meansqr - mean*mean;
        cout << name << ":" << endl;
        cout << "      mean: " << mean << "   var: " << var << endl;

        string infile;

        vector<bool> keep(ml_model_inputs[model].size(),true);
        ScalarT thresh = 0.5;
        int seed = 123;
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<ScalarT> distribution(0.0,1.0);
        for (size_t k=0; k<keep.size(); k++) {
          ScalarT number = distribution(generator);
          if (number<thresh) {
            keep[k] = false;
          }
        }
        {
          std::stringstream ss;
          ss << name;
          ss << "_";
          ss << MacroComm->getRank();
          infile = ss.str() + "_input.txt";
          std::ofstream inputOUT;
          bool is_open = false;
          int attempts = 0;
          int max_attempts = 10;
          while (!is_open && attempts < max_attempts) {
            inputOUT.open(infile);
            is_open = inputOUT.is_open();
            attempts++;
          }
              
          inputOUT.precision(12);
          for (size_t d=0; d<ml_model_inputs[model].size(); ++d) {
            if (keep[d]) {
              for (size_t d2=0; d2<ml_model_inputs[model][d].size(); ++d2) {
                inputOUT << ml_model_inputs[model][d][d2] << "  ";
              }
              inputOUT << endl;
            }
          }
          inputOUT.close();
        }

        string outfile;
        {
          
          std::stringstream ss;
          ss << name;
          ss << "_";
          ss << MacroComm->getRank();
          outfile = ss.str() + "_output.txt";
          std::ofstream outputOUT;
          bool is_open = false;
          int attempts = 0;
          int max_attempts = 10;
          while (!is_open && attempts < max_attempts) {
            outputOUT.open(outfile);
            is_open = outputOUT.is_open();
            attempts++;
          }
              
          outputOUT.precision(12);
              
          for (size_t d=0; d<ml_model_outputs[model].size(); ++d) {
            if (keep[d]) {
              outputOUT << ml_model_outputs[model][d] << "  ";
              outputOUT << endl;
            }
          }
          outputOUT.close();
        }

        MacroComm->barrier();
        
        if (MacroComm->getRank() == 0) {
          // Build the pytorch models
          //string filename = "binary_classifier.py";
          string filename = "classifier2.py";
          string command = settings->get<string>("python","python3");
          
          std::stringstream nnss;
          nnss << name;
          //nnss << "_";
          //nnss << MacroComm->getRank();
          string nnfile = nnss.str() + "_nn.pt";

          std::stringstream procss;
          procss << MacroComm->getSize();

          command += " ";
          command += filename;
          //command += " --input-data " + infile;
          //command += " --output-data " + outfile;
          command += " --model-name " + name;
          command += " --Nproc " + procss.str();
          command += " --nn-model " + nnfile;
              
          system(command.c_str());
        }
      } //model

          

      } // done training
      
    }
  }
      
  ScalarT gmin = 0.0;
  Teuchos::reduceAll(*MacroComm,Teuchos::REDUCE_MIN,1,&my_cost,&gmin);
  ScalarT gmax = 0.0;
  Teuchos::reduceAll(*MacroComm,Teuchos::REDUCE_MAX,1,&my_cost,&gmin);

  if (MacroComm->getRank() == 0 && verbosity > 10) {
    cout << "***** Multiscale Load Balancing Factor " << gmax/gmin <<  endl;
  }
  
}

////////////////////////////////////////////////////////////////////////////////
// Compute the macro->micro->macro map and Jacobian
////////////////////////////////////////////////////////////////////////////////

void MultiscaleManager::evaluateMacroMicroMacroMap(Teuchos::RCP<workset> & wkset, Teuchos::RCP<Group> & group,
                                                   const int & set, 
                                                   const bool & isTransient, const bool & isAdjoint,
                                                   const bool & compute_jacobian, const bool & compute_sens,
                                                   const int & num_active_params,
                                                   const bool & compute_disc_sens, const bool & compute_aux_sens,
                                                   const bool & store_adjPrev) {
  
  wkset->reset();
  
  auto u_curr = group->u[set];
  size_type numElem = u_curr.extent(0);
  // Map the gathered solution to seeded version in workset
  if (group->groupData->requiresTransient) {
    for (size_t iset=0; iset<group->groupData->numSets; ++iset) {
      wkset->computeSolnTransientSeeded(iset, group->u[iset], group->u_prev[iset], 
                                        group->u_stage[iset], 0);
    }
  }
  else { // steady-state
    for (size_t iset=0; iset<group->groupData->numSets; ++iset) {
      wkset->computeSolnSteadySeeded(iset, group->u[iset], 0);
    }
  }
          
  View_Sc3 uvals_sc("coarse vals unseeded",u_curr.extent(0),u_curr.extent(1),u_curr.extent(2));

  for (size_type var=0; var<u_curr.extent(1); ++var) {
            
    size_t uindex = wkset->uvals_index[set][var];
    auto uvals_AD = wkset->uvals[uindex];
    auto uvals_sc_sv = subview(uvals_sc,ALL(),var,ALL());
    parallel_for("assembly compute coarse sol",
                 RangePolicy<AssemblyExec>(0,u_curr.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type dof=0; dof<uvals_AD.extent(1); ++dof) {
#ifndef MrHyDE_NO_AD
        uvals_sc_sv(elem,dof) = uvals_AD(elem,dof).val();
#else
        uvals_sc_sv(elem,dof) = uvals_AD(elem,dof);
#endif
      }
    }); 
  }

  if (subgrid_model_selection == 0) { // user defined
    int sgindex = group->subgrid_model_index;

    subgridModels[sgindex]->subgridSolver(uvals_sc, group->u_prev[set], 
                                          group->phi[set], wkset->time, isTransient, isAdjoint,
                                          compute_jacobian, compute_sens, num_active_params,
                                          compute_disc_sens, false,
                                          *wkset, group->subgrid_usernum, 0,
                                          group->subgradient, store_adjPrev);
  }   
  else if (subgrid_model_selection == 1) { // hierarchical - assumes order is in complexity/fidelity
    subgridModels[0]->subgridSolver(uvals_sc, group->u_prev[set], 
                                    group->phi[set], wkset->time, isTransient, isAdjoint,
                                    compute_jacobian, compute_sens, num_active_params,
                                    compute_disc_sens, false,
                                    *wkset, group->subgrid_usernum, 0,
                                    group->subgradient, store_adjPrev);
    if (subgridModels.size() > 1) {
      size_t cmodel = 1;
      bool satisfied = false;
      while (!satisfied) {
        View_AD2 prev_res = View_AD2("prev res",wkset->res.extent(0),wkset->res.extent(1));
        auto res = wkset->res;
        parallel_for("ms man copy res",
                     RangePolicy<AssemblyExec>(0,res.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<res.extent(1); ++dof) {
            prev_res(elem,dof) = res(elem,dof);
          }
        });
        wkset->resetResidual();
        subgridModels[cmodel]->subgridSolver(uvals_sc, group->u_prev[set], 
                                             group->phi[set], wkset->time, isTransient, isAdjoint,
                                             compute_jacobian, compute_sens, num_active_params,
                                             compute_disc_sens, false,
                                             *wkset, group->subgrid_usernum, 0,
                                             group->subgradient, store_adjPrev);
        if (cmodel == subgridModels.size()-1) {
          satisfied = true;
          group->subgrid_model_index = cmodel;
        }
        else {
          ScalarT resnorm = 0.0;
          for (size_type elem=0; elem<res.extent(0); ++elem) {
              for (size_type dof=0; dof<res.extent(1); ++dof) {
#ifndef MrHyDE_NO_AD
              resnorm += res(elem,dof).val()*res(elem,dof).val();
#else
              resnorm += res(elem,dof)*res(elem,dof);
#endif
            }
          }
          resnorm = std::sqrt(resnorm);

          ScalarT resdiff = 0.0;
          for (size_type elem=0; elem<res.extent(0); ++elem) {
            for (size_type dof=0; dof<res.extent(1); ++dof) {
#ifndef MrHyDE_NO_AD
              ScalarT diff = res(elem,dof).val() - prev_res(elem,dof).val();
#else
              ScalarT diff = res(elem,dof) - prev_res(elem,dof);
#endif
              resdiff += diff*diff;
            }
          }

          ScalarT error = std::sqrt(resdiff)/resnorm;
          if (error < reltol) {
            satisfied = true;
            group->subgrid_model_index = cmodel;
          }
          else {
            cmodel++;
          }
        }
      }
      
      
    }
  }
  else if (subgrid_model_selection == 2) { // ML - assumes order is in complexity/fidelity
    
    if (ml_training  && macro_nl_iter>0) {

      if (ml_model_inputs.size() == 0) {
        for (size_t model=0; model<subgridModels.size(); ++model) {
          vector<vector<ScalarT> > in_data;
          ml_model_inputs.push_back(in_data);
        }
      }
      if (ml_model_outputs.size() == 0) {
        for (size_t model=0; model<subgridModels.size(); ++model) {
          vector<ScalarT> out_data;
          ml_model_outputs.push_back(out_data);
        }
      }
      if (ml_model_extradata.size() == 0) {
        for (size_t model=0; model<subgridModels.size(); ++model) {
          vector<ScalarT> out_data;
          ml_model_extradata.push_back(out_data);
        }
      }
      
      // Get the coarse time derivative
      bool include_timederiv = true;
      View_Sc3 udot_sc("coarse udot unseeded",u_curr.extent(0),u_curr.extent(1),u_curr.extent(2));
      if (include_timederiv) {              

        for (size_type var=0; var<u_curr.extent(1); ++var) {
            
          size_t uindex = group->wkset->uvals_index[set][var];
          auto uvals_AD = group->wkset->u_dotvals[uindex];
          auto udot_sc_sv = subview(udot_sc,ALL(),var,ALL());
          parallel_for("assembly compute coarse sol",
                       RangePolicy<AssemblyExec>(0,u_curr.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type dof=0; dof<uvals_AD.extent(1); ++dof) {
    #ifndef MrHyDE_NO_AD
              udot_sc_sv(elem,dof) = uvals_AD(elem,dof).val();
    #else
              udot_sc_sv(elem,dof) = uvals_AD(elem,dof);
    #endif
            }
          }); 
        }
  
      }

      // Get the average x,y,z locations
      bool include_xyz = true;
      View_Sc2 avg_xyz("average spatial locations",u_curr.extent(0),group->ip.size());
      if (include_xyz) {
        auto wts = group->wts;
        View_Sc1 avg_wts("average wts",u_curr.extent(0));
        parallel_for("assembly compute coarse sol",
                     RangePolicy<AssemblyExec>(0,u_curr.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type pt=0; pt<wts.extent(1); ++pt) {
            avg_wts(elem) += wts(elem,pt);
          }
        });
        auto ip_x = group->ip[0];
        auto ip_y = group->ip[1];
        parallel_for("assembly compute coarse sol",
                     RangePolicy<AssemblyExec>(0,u_curr.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type pt=0; pt<ip_x.extent(1); ++pt) {
            avg_xyz(elem,0) = ip_x(elem,pt)*wts(elem,pt)/avg_wts(elem);
            avg_xyz(elem,1) = ip_y(elem,pt)*wts(elem,pt)/avg_wts(elem);
          }
        });
      }

      vector<vector<ScalarT> > in_data;
      for (size_type elem=0; elem<uvals_sc.extent(0); ++elem) {
        vector<ScalarT> data;
        for (size_type var=0; var<uvals_sc.extent(1); ++var) {
          for (size_type dof=0; dof<uvals_sc.extent(2); ++dof) {
            data.push_back(uvals_sc(elem,var,dof));
          }
          if (include_timederiv) {
            for (size_type dof=0; dof<udot_sc.extent(2); ++dof) {
              data.push_back(udot_sc(elem,var,dof));
            }  
          }
          if (include_xyz) {
            for (size_type dim=0; dim<avg_xyz.extent(1); ++dim) {
              data.push_back(avg_xyz(elem,dim));
            }
          }
        }
        in_data.push_back(data);
      }
      for (size_t model=0; model<subgridModels.size(); ++model) {
        for (size_t d=0; d<in_data.size(); ++d) {
          ml_model_inputs[model].push_back(in_data[d]);
        }
      }

      size_t num_models = subgridModels.size();
      subgridModels[num_models-1]->subgridSolver(uvals_sc, group->u_prev[set], 
                                                 group->phi[set], wkset->time, isTransient, isAdjoint,
                                                 compute_jacobian, compute_sens, num_active_params,
                                                 compute_disc_sens, false,
                                                 *wkset, group->subgrid_usernum, 0,
                                                 group->subgradient, store_adjPrev);
    
      auto res = wkset->res;
      View_AD2 ref_res = View_AD2("prev res",numElem,res.extent(1));
      parallel_for("ms man copy res",
                   RangePolicy<AssemblyExec>(0,numElem),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<res.extent(1); ++dof) {
          ref_res(elem,dof) = res(elem,dof);
        }
      });
        
      vector<ScalarT> resnorm(numElem,0.0);
      for (size_type elem=0; elem<numElem; ++elem) {
        for (size_type dof=0; dof<res.extent(1); ++dof) {
#ifndef MrHyDE_NO_AD
          resnorm[elem] += ref_res(elem,dof).val()*ref_res(elem,dof).val();
#else
          resnorm[elem] += ref_res(elem,dof)*ref_res(elem,dof);
#endif
        }
        resnorm[elem] = std::sqrt(resnorm[elem]);
      }
      

      for (size_t cmodel=0; cmodel<subgridModels.size()-1; ++cmodel) {
        wkset->resetResidual();
        subgridModels[cmodel]->subgridSolver(uvals_sc, group->u_prev[set], 
                                             group->phi[set], wkset->time, isTransient, isAdjoint,
                                             compute_jacobian, compute_sens, num_active_params,
                                             compute_disc_sens, false,
                                             *wkset, group->subgrid_usernum, 0,
                                             group->subgradient, store_adjPrev);
        
        vector<ScalarT> resdiff(numElem,0.0);
        for (size_type elem=0; elem<numElem; ++elem) {
          for (size_type dof=0; dof<res.extent(1); ++dof) {
#ifndef MrHyDE_NO_AD
            ScalarT diff = res(elem,dof).val() - ref_res(elem,dof).val();
#else
            ScalarT diff = res(elem,dof) - ref_res(elem,dof);
#endif
            resdiff[elem] += diff*diff;
          }
        }

        vector<ScalarT> out_data(numElem,0.0);
        vector<ScalarT> extra_data(numElem,0.0);
        //ScalarT abstol = 1.0e-8;

        for (size_type elem=0; elem<numElem; ++elem) {
          ScalarT error = std::sqrt(resdiff[elem]);
          extra_data[elem] = error/resnorm[elem];
          if (error < abstol || error/resnorm[elem] < reltol) {
            out_data[elem] = 1.0;
          }
        }
        for (size_t k=0; k<out_data.size(); ++k) {
          ml_model_outputs[cmodel].push_back(out_data[k]);
          ml_model_extradata[cmodel].push_back(extra_data[k]);
        }
      } // models
      parallel_for("ms man copy res",
                   RangePolicy<AssemblyExec>(0,numElem),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<res.extent(1); ++dof) {
          res(elem,dof) = ref_res(elem,dof);
        }
      });
    }
    else {

      int sgindex = group->subgrid_model_index;
      subgridModels[sgindex]->subgridSolver(uvals_sc, group->u_prev[set], 
                                                 group->phi[set], wkset->time, isTransient, isAdjoint,
                                                 compute_jacobian, compute_sens, num_active_params,
                                                 compute_disc_sens, false,
                                                 *wkset, group->subgrid_usernum, 0,
                                                 group->subgradient, store_adjPrev);
    
      
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Reset the time step
////////////////////////////////////////////////////////////////////////////////

void MultiscaleManager::reset() {
  Teuchos::TimeMonitor localtimer(*resettimer);
  
  //for (size_t j=0; j<subgridModels.size(); j++) {
  //  subgridModels[j]->reset();
  //}
}

////////////////////////////////////////////////////////////////////////////////
// Update parameters
////////////////////////////////////////////////////////////////////////////////

void MultiscaleManager::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params,
                                         const vector<string> & paramnames) {
  for (size_t i=0; i<subgridModels.size(); i++) {
    //subgridModels[i]->paramvals_AD = params;
    subgridModels[i]->updateParameters(params, paramnames);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Get the mean subgrid cell fields
////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,HostDevice> MultiscaleManager::getMeanCellFields(const size_t & block, const int & timeindex,
                                                                        const ScalarT & time, const int & numfields) {
  
  Kokkos::View<ScalarT**,HostDevice> subgrid_cell_fields("subgrid cell fields",groups[block].size(),numfields);
  return subgrid_cell_fields;
}

////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////

void MultiscaleManager::updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) {
  for (size_t i=0; i<subgridModels.size(); i++) {
    subgridModels[i]->updateMeshData(rotation_data);
  }
}

////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////

void MultiscaleManager::completeTimeStep() {
  for (size_t i=0; i<subgridModels.size(); i++) {
    subgridModels[i]->advance();
  }
}

////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////

void MultiscaleManager::completeStage() {
  for (size_t i=0; i<subgridModels.size(); i++) {
    subgridModels[i]->advanceStage();
  }
}
