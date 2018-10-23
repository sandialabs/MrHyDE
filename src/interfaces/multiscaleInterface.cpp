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

MultiScale::MultiScale(const Teuchos::RCP<Epetra_MpiComm> & Comm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings_,
                       vector<vector<Teuchos::RCP<cell> > > & cells_,
                       vector<Teuchos::RCP<SubGridModel> > subgridModels_,
                       Teuchos::RCP<FunctionInterface> macro_functionManager_ ) :
Comm(Comm_), settings(settings_), cells(cells_), subgridModels(subgridModels_),
macro_functionManager(macro_functionManager_) {
  
  if (settings->isSublist("Subgrid")) {
    
    ////////////////////////////////////////////////////////////////////////////////
    // Define the subgrid models specified in the input file
    ////////////////////////////////////////////////////////////////////////////////
    
    int nummodels = settings->sublist("Subgrid").get<int>("Number of Models",1);
    subgrid_static = settings->sublist("Subgrid").get<bool>("Static Subgrids",true);
    
    
    for (size_t n=0; n<subgridModels.size(); n++) {
      stringstream ss;
      ss << n;
      macro_functionManager->addFunction("Subgrid " + ss.str() + " usage",subgridModels[n]->usage,
                                         cells[0][0]->numElem,cells[0][0]->ip.dimension(1),"ip",0);
    }
     
  }
  else {
    subgrid_static = true;
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
    subgridModels[j]->macro_offsets = macro_offsets[mblock];
    subgridModels[j]->macro_paramnames = macro_paramnames;
    subgridModels[j]->macro_disc_paramnames = macro_disc_paramnames;
  }
  
}

////////////////////////////////////////////////////////////////////////////////
// Initial assignment of subgrid models to cells
////////////////////////////////////////////////////////////////////////////////

double MultiScale::initialize() {
  double my_cost = 0.0;
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      // needs to be updated
      //vector<size_t> sgnum = udfunc->getSubgridModel(cells[b][e]->nodes, macro_wkset[b],
      //                                               cells[b][e]->u, subgridModels.size());
      vector<size_t> usernum;
      int numElem = cells[b][e]->numElem;
      
      
      vector<size_t> sgnum(numElem,0);
      
      
      macro_wkset[b]->update(cells[b][e]->ip,cells[b][e]->ijac,cells[b][e]->orientation);
      macro_wkset[b]->computeSolnVolIP(cells[b][e]->u, cells[b][e]->u_dot, false, false);
      macro_wkset[b]->computeParamVolIP(cells[b][e]->param, false);
      
      
      for (size_t s=0; s<subgridModels.size(); s++) {
        stringstream ss;
        ss << s;
        FDATA usagecheck = macro_functionManager->evaluate("Subgrid " + ss.str() + " usage","ip",0);
        
        for (int p=0; p<numElem; p++) {
          for (size_t j=0; j<usagecheck.dimension(1); j++) {
            if (usagecheck(p,j).val() >= 1.0) {
              sgnum[p] = s;
            }
          }
        }
      }
      
      if (subgrid_static) { // only add each cell to one subgrid model
        DRV cellnodes = cells[b][e]->nodes;
        Kokkos::View<int****,HostDevice> cellsideinfo = cells[b][e]->sideinfo;
        for (int c=0; c<numElem; c++) {
          DRV cnodes("cnodes",1,cellnodes.dimension(1),cellnodes.dimension(2));
          Kokkos::View<int****,HostDevice> csideinfo("csideinfo",1,cellsideinfo.dimension(1),
                                                     cellsideinfo.dimension(2),
                                                     cellsideinfo.dimension(3));
          for (int i=0; i<cellnodes.dimension(1); i++) {
            for (int j=0; j<cellnodes.dimension(2); j++) {
              cnodes(0,i,j) = cellnodes(c,i,j);
            }
          }
          for (int i=0; i<cellsideinfo.dimension(1); i++) {
            for (int j=0; j<cellsideinfo.dimension(2); j++) {
              for (int k=0; k<cellsideinfo.dimension(3); k++) {
                csideinfo(0,i,j,k) = cellsideinfo(c,i,j,k);
              }
            }
          }
          // needs to be updated
          int cnum = subgridModels[sgnum[c]]->addMacro(cnodes, csideinfo, cells[b][e]->sidenames,
                                                       cells[b][e]->GIDs[c], cells[b][e]->index[c]);
          usernum.push_back(cnum);
        }
      }
      else {
        // usernum is the same for all subgrid models
        DRV cellnodes = cells[b][e]->nodes;
        Kokkos::View<int****,HostDevice> cellsideinfo = cells[b][e]->sideinfo;
        for (int c=0; c<numElem; c++) {
          DRV cnodes("cnodes",1,cellnodes.dimension(1),cellnodes.dimension(2));
          Kokkos::View<int****,HostDevice> csideinfo("csideinfo",1,cellsideinfo.dimension(1),
                                                     cellsideinfo.dimension(2),
                                                     cellsideinfo.dimension(3));
          
          for (int i=0; i<cellnodes.dimension(1); i++) {
            for (int j=0; j<cellnodes.dimension(2); j++) {
              cnodes(0,i,j) = cellnodes(c,i,j);
            }
          }
          for (int i=0; i<cellsideinfo.dimension(1); i++) {
            for (int j=0; j<cellsideinfo.dimension(2); j++) {
              for (int k=0; k<cellsideinfo.dimension(3); k++) {
                csideinfo(0,i,j,k) = cellsideinfo(c,i,j,k);
              }
            }
          }
          for (size_t s=0; s<subgridModels.size(); s++) {
            int cnum = subgridModels[s]->addMacro(cnodes, csideinfo,
                                                  cells[b][e]->sidenames,
                                                  cells[b][e]->GIDs[c], cells[b][e]->index[c]);
            usernum.push_back(cnum);
          }
        }
      }
      cells[b][e]->subgridModels = subgridModels;
      cells[b][e]->subgrid_model_index.push_back(sgnum);
      cells[b][e]->subgrid_usernum = usernum;
      cells[b][e]->multiscale = true;
      for (int c=0; c<numElem; c++) {
        my_cost += subgridModels[sgnum[c]]->cost_estimate;
      }
    }
  }
  
  
  ////////////////////////////////////////////////////////////////////////////////
  // If the subgrid models are not static, then we need projection maps between
  // the various subgrid models.
  // Since we only store N subgrid models, we only require (N-1)^2 maps
  ////////////////////////////////////////////////////////////////////////////////
  
  if (!subgrid_static) {
    for (size_t i=0; i<subgridModels.size(); i++) {
      //vector<vector<vector<FC> > > curr_sg_basis;
      DRV ip = subgridModels[i]->getIP();
      DRV wts = subgridModels[i]->getIPWts();
      pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > basisinfo_i = subgridModels[i]->evaluateBasis2(ip);
      vector<Teuchos::RCP<Epetra_CrsMatrix> > currmaps;
      for (size_t j=0; j<subgridModels.size(); j++) {
        pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > basisinfo_j = subgridModels[j]->evaluateBasis2(ip);
        matrix_RCP map_over = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *(subgridModels[i]->overlapped_map), -1)); // reset Jacobian
        matrix_RCP map = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *(subgridModels[i]->owned_map), -1)); // reset Jacobian
        
        for (size_t k=0; k<ip.dimension(1); k++) {
          int icell = basisinfo_i.first(k,0);
          int jcell = basisinfo_j.first(k,0);
          for (size_t r=0; r<basisinfo_i.second[k].dimension(0);r++) {
            for (size_t p=0; p<basisinfo_i.second[k].dimension(1);p++) {
              int igid = basisinfo_i.first(k,p+1);
              for (size_t s=0; s<basisinfo_j.second[k].dimension(0);s++) {
                for (size_t q=0; q<basisinfo_j.second[k].dimension(1);q++) {
                  int jgid = basisinfo_j.first(k,q+1);
                  if (r == s) {
                    double val = basisinfo_i.second[k](r,p) * basisinfo_j.second[k](s,q) * wts(0,k);
                    map_over->InsertGlobalValues(igid, 1, &val, &jgid);
                  }
                }
              }
            }
          }
        }
        
        map_over->FillComplete(*(subgridModels[j]->overlapped_map), *(subgridModels[i]->overlapped_map));
        
        map->PutScalar(0.0);
        map->Export(*map_over, *(subgridModels[i]->exporter), Add);
        map->FillComplete(*(subgridModels[j]->owned_map), *(subgridModels[i]->owned_map));
        /*
         if (i == j) {
         cout << "i is " << i << endl;
         cout << *map << endl;
         }
         */
        
        currmaps.push_back(map);
      }
      subgrid_projection_maps.push_back(currmaps);
    }
    
    for (size_t i=0; i<subgridModels.size(); i++) {
      
      Teuchos::RCP<Epetra_LinearProblem> SgLinSys = Teuchos::rcp(new Epetra_LinearProblem());
      Amesos SgAmFactory;
      char* SolverType = "Amesos_Klu";
      Amesos_BaseSolver * SgAmSolver = SgAmFactory.Create(SolverType, *SgLinSys);
      SgLinSys->SetOperator(subgrid_projection_maps[i][i].get());
      SgAmSolver->SymbolicFactorization();
      SgAmSolver->NumericFactorization();
      
      /*
       Epetra_MultiVector newlhs(*(subgridModels[i]->owned_map),1);
       Epetra_MultiVector newrhs(*(subgridModels[i]->owned_map),1);
       SgLinSys->SetLHS(&newlhs);
       SgLinSys->SetRHS(&newrhs);
       SgAmSolver->Solve();
       */
      
      subgrid_projection_linsys.push_back(SgLinSys);
      subgrid_projection_solvers.push_back(SgAmSolver);
    }
  }
  
  // add mesh data
  
  for (size_t s=0; s< subgridModels.size(); s++) {
    subgridModels[s]->addMeshData();
  }
  
  //subgrid_static = true;
  return my_cost;
}

////////////////////////////////////////////////////////////////////////////////
// Re-assignment of subgrid models to cells
////////////////////////////////////////////////////////////////////////////////

double MultiScale::update() {
  double my_cost = 1.0;
  
  if (subgrid_static) {
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        if (cells[b][e]->multiscale) {
          int numElem = cells[b][e]->numElem;
          for (int c=0;c<numElem; c++) {
            int nummod = cells[b][e]->subgrid_model_index[c].size();
            int oldmodel = cells[b][e]->subgrid_model_index[c][nummod-1];
            cells[b][e]->subgrid_model_index[c].push_back(oldmodel);
            my_cost += subgridModels[oldmodel]->cost_estimate;
          }
        }
      }
    }
  }
  else {
    for (size_t b=0; b<cells.size(); b++) {
      for (size_t e=0; e<cells[b].size(); e++) {
        if (cells[b][e]->multiscale) {
          // needs to be updated
          //vector<size_t> newmodel = udfunc->getSubgridModel(cells[b][e]->nodes, macro_wkset[0],
          //                                               cells[b][e]->u, subgridModels.size());
          
          int numElem = cells[b][e]->numElem;
          vector<size_t> newmodel(numElem,0);
          
          macro_wkset[b]->update(cells[b][e]->ip,cells[b][e]->ijac,cells[b][e]->orientation);
          macro_wkset[b]->computeSolnVolIP(cells[b][e]->u, cells[b][e]->u_dot, false, false);
          macro_wkset[b]->computeParamVolIP(cells[b][e]->param, false);
          
          for (size_t s=0; s<subgridModels.size(); s++) {
            stringstream ss;
            ss << s;
            FDATA usagecheck = macro_functionManager->evaluate("Subgrid " + ss.str() + " usage","ip",0);
            
            for (int p=0; p<numElem; p++) {
              for (size_t j=0; j<usagecheck.dimension(1); j++) {
                if (usagecheck(p,j).val() >= 1.0) {
                  newmodel[p] = s;
                }
              }
            }
          }

          for (int c=0;c<numElem; c++) {
            
            int nummod = cells[b][e]->subgrid_model_index[c].size();
            int oldmodel = cells[b][e]->subgrid_model_index[c][nummod-1];
            if (newmodel[c] != oldmodel) {
              
              int usernum = cells[b][e]->subgrid_usernum[c];
              // get the time/solution from old subgrid model at last time step
              int lastindex = subgridModels[oldmodel]->soln[usernum].size()-1;
              pair<double, vector_RCP> lastsol = subgridModels[oldmodel]->soln[usernum][lastindex];
              
              vector_RCP projvec = Teuchos::rcp(new Epetra_MultiVector(*(subgridModels[newmodel[c]]->owned_map),1));
              subgrid_projection_maps[newmodel[c]][oldmodel]->Apply(*(lastsol.second), *projvec);
              
              vector_RCP newvec = Teuchos::rcp(new Epetra_MultiVector(*(subgridModels[newmodel[c]]->owned_map),1));
              subgrid_projection_linsys[newmodel[c]]->SetRHS(projvec.get());
              subgrid_projection_linsys[newmodel[c]]->SetLHS(newvec.get());
              
              subgrid_projection_solvers[newmodel[c]]->Solve();
              
              subgridModels[newmodel[c]]->solutionStorage(newvec, lastsol.first, false, usernum);
              
              // update the cell
              //cells[b][e]->subgridModel = subgridModels[newmodel];
              
            }
            my_cost += subgridModels[newmodel[c]]->cost_estimate;
            cells[b][e]->subgrid_model_index[c].push_back(newmodel[c]);
          }
        }
      }
    }
  }
  
  return my_cost;
}

////////////////////////////////////////////////////////////////////////////////
// Post-processing
////////////////////////////////////////////////////////////////////////////////

void MultiScale::writeSolution(const string & macrofilename, const vector<double> & solvetimes,
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
          for (size_t c=0; c<cells[b][e]->numElem; c++) {
            
            stringstream ss;
            ss << globalPID << "." << e;
            string filename = "subgrid_data/subgrid_"+macrofilename+".exo." + ss.str();// + ".exo";
            //cells[b][e]->writeSubgridSolution(blockname);
            int sgmodelnum = cells[b][e]->subgrid_model_index[c][0];
            subgridModels[sgmodelnum]->writeSolution(filename, cells[b][e]->subgrid_usernum[c]);
          }
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


Kokkos::View<double**,HostDevice> MultiScale::getMeanCellFields(const size_t & block, const int & timeindex,
                                                                const double & time, const int & numfields) {
  
  Kokkos::View<double**,HostDevice> subgrid_cell_fields("subgrid cell fields",cells[block].size(),numfields);
  /*
  if (subgridModels.size() > 0) {
    for (size_t e=0; e<cells[block].size(); e++) {
      int sgmodelnum = cells[block][e]->subgrid_model_index[timeindex];
      FC cfields = subgridModels[sgmodelnum]->getCellFields(cells[block][e]->subgrid_usernum, time);
      size_t nsgc = cfields.dimension(0);
      for (size_t k=0; k<cfields.dimension(1); k++) {
        double cval = 0.0;
        for (size_t j=0; j<nsgc; j++) {
          cval += cfields(j,k)/(double)nsgc;
        }
        subgrid_cell_fields(e,k) = cval;
      }
    }
  }
  */
  
  return subgrid_cell_fields;
}

void MultiScale::updateMeshData(Kokkos::View<double**,HostDevice> & rotation_data) {
  for (size_t i=0; i<subgridModels.size(); i++) {
    subgridModels[i]->updateMeshData(rotation_data);
  }
}
