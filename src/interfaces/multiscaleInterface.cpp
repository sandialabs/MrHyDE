/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "multiscaleInterface.hpp"

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

MultiScale::MultiScale(const Teuchos::RCP<MpiComm> & MacroComm_,
                       const Teuchos::RCP<MpiComm> & Comm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                       vector<vector<Teuchos::RCP<cell> > > & cells_,
                       vector<Teuchos::RCP<SubGridModel> > subgridModels_,
                       vector<Teuchos::RCP<FunctionManager> > macro_functionManagers_ ) :
MacroComm(MacroComm_), Comm(Comm_), settings(settings_), cells(cells_), subgridModels(subgridModels_),
macro_functionManagers(macro_functionManagers_) {
  
  milo_debug_level = settings->get<int>("debug level",0);
  if (milo_debug_level > 0) {
    if (MacroComm->getRank() == 0) {
      cout << "**** Starting multiscale manager constructor ..." << endl;
    }
  }
  if (settings->isSublist("Subgrid")) {
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the subgrid models specified in the input file
    ////////////////////////////////////////////////////////////////////////////////
    
    int nummodels = settings->sublist("Subgrid").get<int>("Number of Models",1);
    subgrid_static = settings->sublist("Subgrid").get<bool>("Static Subgrids",true);
    macro_concurrency = settings->sublist("Subgrid").get<int>("Macro-element concurrency",1);
    int numElem = settings->sublist("Solver").get<int>("Workset size",1);
    for (size_t n=0; n<subgridModels.size(); n++) {
      stringstream ss;
      ss << n;
      macro_functionManagers[0]->addFunction("Subgrid " + ss.str() + " usage",subgridModels[n]->usage, "ip");
    }
     
  }
  else {
    subgrid_static = true;
  }
  
  if (milo_debug_level > 0) {
    if (MacroComm->getRank() == 0) {
      cout << "**** Finished multiscale manager constructor" << endl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Set the information from the macro-scale that does not depend on the specific cell
////////////////////////////////////////////////////////////////////////////////

void MultiScale::setMacroInfo(vector<vector<basis_RCP> > & macro_basis_pointers,
                              vector<vector<string> > & macro_basis_types,
                              vector<vector<string> > & macro_varlist,
                              vector<vector<int> > macro_usebasis,
                              vector<vector<vector<int> > > & macro_offsets,
                              vector<string> & macro_paramnames,
                              vector<string> & macro_disc_paramnames) {
  
  for (int j=0; j<subgridModels.size(); j++) {
    int mblock = subgridModels[j]->macro_block;
    subgridModels[j]->macro_basis_pointers = macro_basis_pointers[mblock];
    subgridModels[j]->macro_basis_types = macro_basis_types[mblock];
    subgridModels[j]->macro_varlist = macro_varlist[mblock];
    subgridModels[j]->macro_usebasis = macro_usebasis[mblock];
    subgridModels[j]->macro_offsets = macro_wkset[mblock]->offsets;//macro_offsets[mblock];
    subgridModels[j]->macro_paramnames = macro_paramnames;
    subgridModels[j]->macro_disc_paramnames = macro_disc_paramnames;
    subgridModels[j]->subgrid_static = subgrid_static;
    subgridModels[j]->macrosidenames = cells[0][0]->cellData->sidenames;
  }
  
}

////////////////////////////////////////////////////////////////////////////////
// Initial assignment of subgrid models to cells
////////////////////////////////////////////////////////////////////////////////

ScalarT MultiScale::initialize() {
  if (milo_debug_level > 0) {
    if (MacroComm->getRank() == 0) {
      cout << "**** Starting multiscale manager initialize" << endl;
    }
  }
  ScalarT my_cost = 0.0;
  size_t numusers = 0;
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      // needs to be updated
      //vector<size_t> sgnum = udfunc->getSubgridModel(cells[b][e]->nodes, macro_wkset[b],
      //                                               cells[b][e]->u, subgridModels.size());
      //vector<size_t> usernum;
      //int numElem = cells[b][e]->numElem;
      
      macro_wkset[b]->update(cells[b][e]->ip,cells[b][e]->wts,
                             cells[b][e]->jacobian,cells[b][e]->jacobianInv,
                             cells[b][e]->jacobianDet,cells[b][e]->orientation);
      Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
      seedwhat(0) = 0;
      macro_wkset[b]->computeSolnVolIP(cells[b][e]->u, cells[b][e]->u_dot, seedwhat);
      macro_wkset[b]->computeParamVolIP(cells[b][e]->param, seedwhat);
      
      //vector<size_t> sgnum(numElem,0);
      vector<int> sgvotes(subgridModels.size(),0);
      
      for (size_t s=0; s<subgridModels.size(); s++) {
        stringstream ss;
        ss << s;
        FDATA usagecheck = macro_functionManagers[0]->evaluate("Subgrid " + ss.str() + " usage","ip");
        
        for (int p=0; p<cells[b][e]->numElem; p++) {
          for (size_t j=0; j<usagecheck.extent(1); j++) {
            if (usagecheck(p,j).val() >= 1.0) {
              sgvotes[s] += 1;
            }
          }
        }
        //cout << "s = " << s << "  " << sgvotes[s] << endl;
      }
      int maxvotes = -1;
      int sgwinner = -1;
      for (size_t i=0; i<sgvotes.size(); i++) {
        if (sgvotes[i] >= maxvotes) {
          maxvotes = sgvotes[i];
          sgwinner = i;
        }
      }
      if (maxvotes < cells[b][e]->numElem) {
        //output a warning
      }
      
      int sgusernum;
      if (subgrid_static) { // only add each cell to one subgrid model
        
        sgusernum = subgridModels[sgwinner]->addMacro(cells[b][e]->nodes,
                                                      cells[b][e]->sideinfo,
                                                      cells[b][e]->GIDs,
                                                      cells[b][e]->index,
                                                      cells[b][e]->orientation);
        
        //int cnum = subgridModels[sgnum[c]]->addMacro(cnodes, csideinfo,
        //                                             cGIDs, cindex, orientation);
        
        
      }
      else {
        for (size_t s=0; s<subgridModels.size(); s++) { // needs to add this cell info to all of them (sgusernum is same for all)
          sgusernum = subgridModels[s]->addMacro(cells[b][e]->nodes,
                                                 cells[b][e]->sideinfo,
                                                 cells[b][e]->GIDs,
                                                 cells[b][e]->index,
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
  
  for (size_t s=0; s<subgridModels.size(); s++) {
    subgridModels[s]->finalize();
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
      DRV ip = subgridModels[i]->getIP();
      DRV wts = subgridModels[i]->getIPWts();
      
      pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > basisinfo_i = subgridModels[i]->evaluateBasis2(ip);
      vector<Teuchos::RCP<LA_CrsMatrix> > currmaps;
      for (size_t j=0; j<subgridModels.size(); j++) {
        pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo_j = subgridModels[j]->evaluateBasis2(ip);
        Teuchos::RCP<LA_CrsMatrix> map = subgridModels[i]->getProjectionMatrix(ip, wts, basisinfo_j);
        currmaps.push_back(map);
      }
      subgrid_projection_maps.push_back(currmaps);
    }
    
    for (size_t i=0; i<subgridModels.size(); i++) {
      vector_RCP dummy_vec = subgridModels[i]->getVector();
      vector_RCP dummy_vec2 = subgridModels[i]->getVector();
      Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > Am2Solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2",subgrid_projection_maps[i][i], dummy_vec, dummy_vec2);
      Am2Solver->symbolicFactorization();
      Am2Solver->numericFactorization();
      subgrid_projection_solvers.push_back(Am2Solver);
    }
  }
  
  // add mesh data
  
  for (size_t s=0; s< subgridModels.size(); s++) {
    subgridModels[s]->addMeshData();
  }
  
  if (milo_debug_level > 0) {
    if (MacroComm->getRank() == 0) {
      cout << "**** Finished multiscale manager initialize" << endl;
    }
  }
  
  //subgrid_static = true;
  return my_cost;
}

////////////////////////////////////////////////////////////////////////////////
// Re-assignment of subgrid models to cells
////////////////////////////////////////////////////////////////////////////////

ScalarT MultiScale::update() {
  ScalarT my_cost = 1.0;
  
  if (subgrid_static) {
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        if (cells[b][e]->cellData->multiscale) {
          int numElem = cells[b][e]->numElem;
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
          
          macro_wkset[b]->update(cells[b][e]->ip,cells[b][e]->wts,
                                 cells[b][e]->jacobian,cells[b][e]->jacobianInv,
                                 cells[b][e]->jacobianDet,cells[b][e]->orientation);
          Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
          seedwhat(0) = 0;
          macro_wkset[b]->computeSolnVolIP(cells[b][e]->u, cells[b][e]->u_dot, seedwhat);
          macro_wkset[b]->computeParamVolIP(cells[b][e]->param, seedwhat);
          
          vector<int> sgvotes(subgridModels.size(),0);
          
          for (size_t s=0; s<subgridModels.size(); s++) {
            stringstream ss;
            ss << s;
            FDATA usagecheck = macro_functionManagers[0]->evaluate("Subgrid " + ss.str() + " usage","ip");
            for (int p=0; p<cells[b][e]->numElem; p++) {
              for (size_t j=0; j<usagecheck.extent(1); j++) {
                if (usagecheck(p,j).val() >= 1.0) {
                  sgvotes[s] += 1;
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
          if (maxvotes < cells[b][e]->numElem) {
            //output a warning
          }
          
          //for (int c=0;c<numElem; c++) {
            
            int nummod = cells[b][e]->subgrid_model_index.size();
            int oldmodel = cells[b][e]->subgrid_model_index[nummod-1];
            if (sgwinner != oldmodel) {
              
              int usernum = cells[b][e]->subgrid_usernum;
              // get the time/solution from old subgrid model at last time step
              int lastindex = subgridModels[oldmodel]->soln->times[usernum].size()-1;
              Teuchos::RCP<LA_MultiVector> lastsol = subgridModels[oldmodel]->soln->data[usernum][lastindex];
              ScalarT lasttime = subgridModels[oldmodel]->soln->times[usernum][lastindex];
              //Teuchos::RCP<LA_MultiVector> projvec = Teuchos::rcp(new LA_MultiVector(subgridModels[newmodel[c]]->owned_map,1));
              vector_RCP projvec = subgridModels[sgwinner]->getVector();//Teuchos::rcp(new LA_MultiVector(subgridModels[newmodel[c]]->owned_map,1));
              subgrid_projection_maps[sgwinner][oldmodel]->apply(*lastsol, *projvec);
              
              //Teuchos::RCP<LA_MultiVector> newvec = Teuchos::rcp(new LA_MultiVector(subgridModels[newmodel[c]]->owned_map,1));
              vector_RCP newvec = subgridModels[sgwinner]->getVector();
              subgrid_projection_solvers[sgwinner]->setB(projvec);
              subgrid_projection_solvers[sgwinner]->setX(newvec);
              
              subgrid_projection_solvers[sgwinner]->solve();
              subgridModels[sgwinner]->soln->store(newvec, lasttime, usernum);
              //subgridModels[newmodel[c]]->solutionStorage(newvec, lastsol.first, false, usernum);
              
              // update the cell
              //cells[b][e]->subgridModel = subgridModels[newmodel];
              
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

void MultiScale::reset() {
  //for (size_t j=0; j<subgridModels.size(); j++) {
  //  subgridModels[j]->reset();
  //}
}

////////////////////////////////////////////////////////////////////////////////
// Post-processing
////////////////////////////////////////////////////////////////////////////////

void MultiScale::writeSolution(const string & macrofilename, const vector<ScalarT> & solvetimes,
                               const int & globalPID) {
  
  
  //vector<FC> subgrid_cell_fields;
  if (subgridModels.size() > 0) {
    if (subgrid_static) {
      /*
      for (size_t s=0; s<subgridModels.size(); s++) {
        stringstream ss;
        ss << s << "." << globalPID;
        string filename = "subgrid_data/subgrid_"+macrofilename+".exo." + ss.str();// + ".exo";
        //cells[b][e]->writeSubgridSolution(blockname);
        subgridModels[s]->writeSolution(filename);
        
      }
       */
      
      for (size_t b=0; b<cells.size(); b++) {
        for (size_t e=0; e<cells[b].size(); e++) {
          //for (size_t c=0; c<cells[b][e]->numElem; c++) {
            
            stringstream ss;
            ss << globalPID << "." << e;
            string filename = "subgrid_data/subgrid_"+macrofilename+".exo." + ss.str();// + ".exo";
            //cells[b][e]->writeSubgridSolution(blockname);
            int sgmodelnum = cells[b][e]->subgrid_model_index[0];
            subgridModels[sgmodelnum]->writeSolution(filename, cells[b][e]->subgrid_usernum);
          //}
        }
      }
    }
    else {
      /*
      for (size_t i=0; i<solvetimes.size(); i++) {
        for (size_t b=0; b<cells.size(); b++) {
          for (size_t e=0; e<cells[b].size(); e++) {
            for (size_t c=0; c<cells[b][e]->numElem; c++) {
              
              int usernum = cells[b][e]->subgrid_usernum[c];
              int timeindex = 0;
              int currsgmodel = cells[b][e]->subgrid_model_index[c][i];
              for (size_t k=0; k<subgridModels[currsgmodel]->soln[usernum].size(); k++) {
                if (abs(solvetimes[i]-subgridModels[currsgmodel]->soln[usernum][k].first)<1.0e-10) {
                  timeindex = k;
                }
              }
              
              stringstream ss, ss2;
              ss << globalPID << "." << e;
              ss2 << i;
              string filename = "subgrid_data/subgrid_"+macrofilename+ss2.str() + ".exo." + ss.str();// + ".exo";
              //cells[b][e]->writeSubgridSolution(blockname);
              subgridModels[currsgmodel]->writeSolution(filename, usernum, timeindex);
            }
          }
        }
      }*/
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////
// Update parameters
////////////////////////////////////////////////////////////////////////////////

void MultiScale::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params,
                                  const vector<string> & paramnames) {
  for (size_t i=0; i<subgridModels.size(); i++) {
    //subgridModels[i]->paramvals_AD = params;
    subgridModels[i]->updateParameters(params, paramnames);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Get the mean subgrid cell fields
////////////////////////////////////////////////////////////////////////////////


Kokkos::View<ScalarT**,HostDevice> MultiScale::getMeanCellFields(const size_t & block, const int & timeindex,
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

void MultiScale::updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) {
  for (size_t i=0; i<subgridModels.size(); i++) {
    subgridModels[i]->updateMeshData(rotation_data);
  }
}
