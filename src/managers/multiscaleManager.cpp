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
#include "subgridFEM.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

MultiscaleManager::MultiscaleManager(const Teuchos::RCP<MpiComm> & MacroComm_,
                                     Teuchos::RCP<MeshInterface> & mesh_,
                                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                     vector<vector<Teuchos::RCP<cell> > > & cells_,
                                     vector<Teuchos::RCP<FunctionManager> > macro_functionManagers_ ) :
MacroComm(MacroComm_), settings(settings_), cells(cells_), macro_functionManagers(macro_functionManagers_) {
  
  debug_level = settings->get<int>("debug level",0);
  if (debug_level > 0) {
    if (MacroComm->getRank() == 0) {
      cout << "**** Starting MultiscaleManager manager constructor ..." << endl;
    }
  }
  
  Teuchos::RCP<MpiComm> unusedComm;
  SplitComm(settings, *MacroComm, unusedComm, Comm);
  
  //subgridModels = subgridGenerator(Comm, settings, mesh_);
  
  if (settings->isSublist("Subgrid")) {
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the subgrid models specified in the input file
    ////////////////////////////////////////////////////////////////////////////////
    
    int nummodels = settings->sublist("Subgrid").get<int>("number of models",1);
    int  num_macro_time_steps = settings->sublist("Solver").get("number of steps",1);
    ScalarT finaltime = settings->sublist("Solver").get<ScalarT>("final time",1.0);
    ScalarT macro_deltat = finaltime/num_macro_time_steps;
    if (nummodels == 1) {
      Teuchos::RCP<Teuchos::ParameterList> subgrid_pl = rcp(new Teuchos::ParameterList("Subgrid"));
      subgrid_pl->setParameters(settings->sublist("Subgrid"));
      string subgrid_model_type = subgrid_pl->get<string>("subgrid model","FEM");
      string macro_block_name = subgrid_pl->get<string>("macro block","eblock-0_0_0");
      std::vector<string> macro_blocknames;
      mesh_->stk_mesh->getElementBlockNames(macro_blocknames);
      int macro_block = 0; // default to single block case
      for (size_t m=0; m<macro_blocknames.size(); ++m) {
        if (macro_blocknames[m] == macro_block_name) {
          macro_block = m;
        }
      }
      topo_RCP macro_topo = mesh_->stk_mesh->getCellTopology(macro_blocknames[macro_block]);
      if (subgrid_model_type == "FEM") {
        subgridModels.push_back(Teuchos::rcp( new SubGridFEM(Comm, subgrid_pl, macro_topo,
                                                             num_macro_time_steps,
                                                             macro_deltat) ) );
      }
      else if (subgrid_model_type == "Explicit FEM") {
        // subgridModels.push_back(Teuchos::rcp( new SubGridExpFEM(Comm, subgrid_pl, macro_topo,
        //                                                         num_macro_time_steps,
        //                                                         macro_deltat) ) );
      }
      else if (subgrid_model_type == "FEM2") {
        //subgridModels.push_back(Teuchos::rcp( new SubGridFEM2(Comm, subgrid_pl, macro_topo, num_macro_time_steps, macro_deltat) ) );
      }
      subgridModels[subgridModels.size()-1]->macro_block = macro_block;
      subgridModels[subgridModels.size()-1]->usage = "1.0";
    }
    else {
      for (int j=0; j<nummodels; j++) {
        std::stringstream ss;
        ss << j;
        if (settings->sublist("Subgrid").isSublist("Model" + ss.str())) {
          Teuchos::RCP<Teuchos::ParameterList> subgrid_pl = rcp(new Teuchos::ParameterList("Subgrid"));
          subgrid_pl->setParameters(settings->sublist("Subgrid").sublist("Model" + ss.str()));
          string subgrid_model_type = subgrid_pl->get<string>("subgrid model","FEM");
          string macro_block_name = subgrid_pl->get<string>("macro block","eblock-0_0_0");
          std::vector<string> macro_blocknames;
          mesh_->stk_mesh->getElementBlockNames(macro_blocknames);
          int macro_block = 0; // default to single block case
          for (size_t m=0; m<macro_blocknames.size(); ++m) {
            if (macro_blocknames[m] == macro_block_name) {
              macro_block = m;
            }
          }
          topo_RCP macro_topo = mesh_->stk_mesh->getCellTopology(macro_blocknames[macro_block]);
          
          if (subgrid_model_type == "FEM") {
            subgridModels.push_back(Teuchos::rcp( new SubGridFEM(Comm, subgrid_pl, macro_topo,
                                                                 num_macro_time_steps,
                                                                 macro_deltat ) ) );
          }
          else if (subgrid_model_type == "Explicit FEM") {
            // subgridModels.push_back(Teuchos::rcp( new SubGridExpFEM(Comm, subgrid_pl, macro_topo,
            //                                                          num_macro_time_steps,
            //                                                          macro_deltat ) ) );
          }
          else if (subgrid_model_type == "FEM2") {
            //subgridModels.push_back(Teuchos::rcp( new SubGridFEM2(Comm, subgrid_pl, macro_topo, num_macro_time_steps, macro_deltat ) ) );
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
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the subgrid models specified in the input file
    ////////////////////////////////////////////////////////////////////////////////
    
    subgrid_static = settings->sublist("Subgrid").get<bool>("static subgrids",true);
    for (size_t n=0; n<subgridModels.size(); n++) {
      std::stringstream ss;
      ss << n;
      int macro_block = subgridModels[n]->macro_block;
      macro_functionManagers[macro_block]->addFunction("Subgrid " + ss.str() + " usage",subgridModels[n]->usage, "ip");
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
// Set the information from the macro-scale that does not depend on the specific cell
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
    subgridModels[j]->macrosidenames = cells[0][0]->cellData->sidenames;
  }
  
}

////////////////////////////////////////////////////////////////////////////////
// Initial assignment of subgrid models to cells
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
  for (size_t b=0; b<cells.size(); b++) {
    
    bool uses_subgrid = false;
    for (size_t s=0; s<subgridModels.size(); s++) {
      if (subgridModels[s]->macro_block == b) {
        uses_subgrid = true;
      }
    }
    if (uses_subgrid) {
      for (size_t e=0; e<cells[b].size(); e++) {
        
        cells[b][e]->updateWorksetBasis();
        macro_wkset[b]->computeSolnSteadySeeded(cells[b][e]->u, 0);
        macro_wkset[b]->computeParamSteadySeeded(cells[b][e]->param, 0);
        cells[b][e]->computeSolnVolIP();
        
        vector<int> sgvotes(subgridModels.size(),0);
        
        for (size_t s=0; s<subgridModels.size(); s++) {
          if (subgridModels[s]->macro_block == b) {
            std::stringstream ss;
            ss << s;
            auto usagecheck = macro_functionManagers[b]->evaluate("Subgrid " + ss.str() + " usage","ip");
            Kokkos::View<ScalarT**,AssemblyDevice> usagecheck_tmp("temp usage check",usagecheck.extent(0),usagecheck.extent(1));
            parallel_for("assembly copy LIDs",RangePolicy<AssemblyExec>(0,usagecheck.extent(0)), KOKKOS_LAMBDA (const int i ) {
              for (size_type j=0; j<usagecheck.extent(1); j++) {
                usagecheck_tmp(i,j) = usagecheck(i,j).val();
              }
            });
            
            auto host_usagecheck = Kokkos::create_mirror_view(usagecheck_tmp);
            Kokkos::deep_copy(host_usagecheck, usagecheck_tmp);
            for (size_t p=0; p<cells[b][e]->numElem; p++) {
              for (size_t j=0; j<host_usagecheck.extent(1); j++) {
                if (host_usagecheck(p,j) >= 1.0) {
                  sgvotes[s] += 1;
                }
              }
            }
          }
        }
        
        int maxvotes = -1;
        int sgwinner = -1;
        for (size_t i=0; i<sgvotes.size(); i++) {
          if (sgvotes[i] >= maxvotes) {
            maxvotes = sgvotes[i];
            sgwinner = i;
          }
        }
        
        size_t sgusernum = 0;
        if (subgrid_static) { // only add each cell to one subgrid model
          
          sgusernum = subgridModels[sgwinner]->addMacro(cells[b][e]->nodes,
                                                        cells[b][e]->sideinfo,
                                                        cells[b][e]->LIDs,
                                                        cells[b][e]->orientation);
          
        }
        else {
          for (size_t s=0; s<subgridModels.size(); s++) { // needs to add this cell info to all of them (sgusernum is same for all)
            sgusernum = subgridModels[s]->addMacro(cells[b][e]->nodes,
                                                   cells[b][e]->sideinfo,
                                                   cells[b][e]->LIDs,
                                                   cells[b][e]->orientation);
          }
        }
        cells[b][e]->subgridModels = subgridModels;
        cells[b][e]->subgrid_model_index.push_back(sgwinner);
        cells[b][e]->subgrid_usernum = sgusernum;
        cells[b][e]->cellData->multiscale = true;
        my_cost = subgridModels[sgwinner]->cost_estimate * cells[b][e]->numElem;
        numusers += 1;
      }
    }
  }
  
  for (size_t s=0; s<subgridModels.size(); s++) {
    subgridModels[s]->finalize(MacroComm->getSize(), MacroComm->getRank());
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // If the subgrid models are not static, then we need projection maps between
  // the various subgrid models.
  // Since we only store N subgrid models, we only require (N-1)^2 maps
  ////////////////////////////////////////////////////////////////////////////////
  
  if (!subgrid_static) {
    
    for (size_t s=0; s<subgridModels.size(); s++) {
      vector<bool> active(numusers,false);
      size_t numactive = 0;
      for (size_t b=0; b<cells.size(); b++) {
        for (size_t e=0; e<cells[b].size(); e++) {
          //for (int c=0; c<cells[b][e]->numElem; c++) {
          if (cells[b][e]->subgrid_model_index[0] == s) {
            size_t usernum = cells[b][e]->subgrid_usernum;
            active[usernum] = true;
            numactive += 1;
          }
        }
        //}
      }
      subgridModels[s]->active.push_back(active);
    }
    
    for (size_t i=0; i<subgridModels.size(); i++) {
      auto ip = subgridModels[i]->getIP();
      auto wts = subgridModels[i]->getIPWts();
      
      std::pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > basisinfo_i = subgridModels[i]->evaluateBasis2(ip);
      vector<Teuchos::RCP<SGLA_CrsMatrix> > currmaps;
      for (size_t j=0; j<subgridModels.size(); j++) {
        std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo_j = subgridModels[j]->evaluateBasis2(ip);
        Teuchos::RCP<SGLA_CrsMatrix> map = subgridModels[i]->getProjectionMatrix(ip, wts, basisinfo_j);
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
// Re-assignment of subgrid models to cells
////////////////////////////////////////////////////////////////////////////////

ScalarT MultiscaleManager::update() {
  
  Teuchos::TimeMonitor localtimer(*updatetimer);
  
  ScalarT my_cost = 1.0;
  
  if (subgrid_static) {
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        if (cells[b][e]->cellData->multiscale) {
          //int numElem = cells[b][e]->numElem;
          //for (int c=0;c<numElem; c++) {
          int nummod = cells[b][e]->subgrid_model_index.size();
          int oldmodel = cells[b][e]->subgrid_model_index[nummod-1];
          cells[b][e]->subgrid_model_index.push_back(oldmodel);
          my_cost += subgridModels[oldmodel]->cost_estimate;
          //}
        }
      }
    }
  }
  else {
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        if (cells[b][e]->cellData->multiscale) {
          
          cells[b][e]->updateWorksetBasis();
          macro_wkset[b]->computeSolnSteadySeeded(cells[b][e]->u, 0);
          macro_wkset[b]->computeParamSteadySeeded(cells[b][e]->param, 0);
          cells[b][e]->computeSolnVolIP();
          
          vector<int> sgvotes(subgridModels.size(),0);
          
          for (size_t s=0; s<subgridModels.size(); s++) {
            if (subgridModels[s]->macro_block == b) {
              std::stringstream ss;
              ss << s;
              auto usagecheck = macro_functionManagers[b]->evaluate("Subgrid " + ss.str() + " usage","ip");
              for (size_t p=0; p<cells[b][e]->numElem; p++) {
                for (size_t j=0; j<usagecheck.extent(1); j++) {
                  if (usagecheck(p,j).val() >= 1.0) {
                    sgvotes[s] += 1;
                  }
                }
              }
            }
          }
          int maxvotes = -1;
          int sgwinner = -1;
          for (size_t i=0; i<sgvotes.size(); i++) {
            if (sgvotes[i] >= maxvotes) {
              maxvotes = sgvotes[i];
              sgwinner = i;
            }
          }
          
          
          //for (int c=0;c<numElem; c++) {
          
          int nummod = cells[b][e]->subgrid_model_index.size();
          int oldmodel = cells[b][e]->subgrid_model_index[nummod-1];
          if (sgwinner != oldmodel) {
            
            size_t usernum = cells[b][e]->subgrid_usernum;
            // get the time/solution from old subgrid model at last time step
            int lastindex = subgridModels[oldmodel]->soln->times[usernum].size()-1;
            Teuchos::RCP<SGLA_MultiVector> lastsol = subgridModels[oldmodel]->soln->data[usernum][lastindex];
            ScalarT lasttime = subgridModels[oldmodel]->soln->times[usernum][lastindex];
            Teuchos::RCP<SGLA_MultiVector> projvec = subgridModels[sgwinner]->getVector();
            subgrid_projection_maps[sgwinner][oldmodel]->apply(*lastsol, *projvec);
            
            Teuchos::RCP<SGLA_MultiVector> newvec = subgridModels[sgwinner]->getVector();
            subgrid_projection_solvers[sgwinner]->setB(projvec);
            subgrid_projection_solvers[sgwinner]->setX(newvec);
            
            subgrid_projection_solvers[sgwinner]->solve();
            subgridModels[sgwinner]->soln->store(newvec, lasttime, usernum);
            
          }
          my_cost += subgridModels[sgwinner]->cost_estimate * cells[b][e]->numElem;
          cells[b][e]->subgrid_model_index.push_back(sgwinner);
          //}
        }
      }
    }
    
    for (size_t s=0; s<subgridModels.size(); s++) {
      vector<bool> active(subgridModels[s]->active[0].size(),false);
      size_t numactive = 0;
      for (size_t b=0; b<cells.size(); b++) {
        for (size_t e=0; e<cells[b].size(); e++) {
          //for (int c=0; c<cells[b][e]->numElem; c++) {
          size_t nindex = cells[b][e]->subgrid_model_index.size();
          if (cells[b][e]->subgrid_model_index[nindex-1] == s) {
            size_t usernum = cells[b][e]->subgrid_usernum;
            active[usernum] = true;
            numactive += 1;
          }
          //}
        }
      }
      subgridModels[s]->active.push_back(active);
    }
  }
  
  return my_cost;
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
  
  Kokkos::View<ScalarT**,HostDevice> subgrid_cell_fields("subgrid cell fields",cells[block].size(),numfields);
  /*
   if (subgridModels.size() > 0) {
   for (size_t e=0; e<cells[block].size(); e++) {
   int sgmodelnum = cells[block][e]->subgrid_model_index[timeindex];
   FC cfields = subgridModels[sgmodelnum]->getCellFields(cells[block][e]->subgrid_usernum, time);
   size_t nsgc = cfields.extent(0);
   for (size_t k=0; k<cfields.extent(1); k++) {
   ScalarT cval = 0.0;
   for (size_t j=0; j<nsgc; j++) {
   cval += cfields(j,k)/(ScalarT)nsgc;
   }
   subgrid_cell_fields(e,k) = cval;
   }
   }
   }
   */
  
  return subgrid_cell_fields;
}

void MultiscaleManager::updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) {
  for (size_t i=0; i<subgridModels.size(); i++) {
    subgridModels[i]->updateMeshData(rotation_data);
  }
}
