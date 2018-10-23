/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "cell.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "solverInterface.hpp"
#include "uqInterface.hpp"
#include "subgridFEM.hpp"

#include <iostream>
#include <iterator>

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux basis functions at the integration points.
// This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
///////////////////////////////////////////////////////////////////////////////////////

void cell::addAuxDiscretization(const vector<basis_RCP> & abasis_pointers, const vector<DRV> & abasis,
                                const vector<DRV> & abasisGrad, const vector<vector<DRV> > & asideBasis,
                                const vector<vector<DRV> > & asideBasisGrad) {
  
  for (size_t b=0; b<abasis_pointers.size(); b++) {
    auxbasisPointers.push_back(abasis_pointers[b]);
  }
  for (size_t b=0; b<abasis.size(); b++) {
    auxbasis.push_back(abasis[b]);
    //auxbasisGrad.push_back(abasisGrad[b]);
  }
  
  for (size_t s=0; s<asideBasis.size(); s++) {
    auxside_basis.push_back(asideBasis[s]);
    //auxside_basisGrad.push_back(asideBasisGrad[s]);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux variables
///////////////////////////////////////////////////////////////////////////////////////

void cell::addAuxVars(const vector<string> & auxlist_) {
  auxlist = auxlist_;
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the regular parameters (everything but discretized)
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
  physics_RCP->updateParameters(params, paramnames);
}


///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each variable will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setUseBasis(vector<int> & usebasis_, const int & nstages_) {
  vector<int> usebasis = usebasis_;
  //num_stages = nstages;
  
  // Set up the containers for usual solution storage
  size_t maxnbasis = 0;
  for (size_t i=0; i<numDOF.size(); i++) {
    if (numDOF(i) > maxnbasis) {
      maxnbasis = numDOF(i);
    }
  }
  //maxnbasis *= nstages;
  u = Kokkos::View<double***,AssemblyDevice>("u",numElem,numDOF.size(),maxnbasis);
  u_dot = Kokkos::View<double***,AssemblyDevice>("u_dot",numElem,numDOF.size(),maxnbasis);
  phi = Kokkos::View<double***,AssemblyDevice>("phi",numElem,numDOF.size(),maxnbasis);
  phi_dot = Kokkos::View<double***,AssemblyDevice>("phi_dot",numElem,numDOF.size(),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
  Kokkos::View<int*,HostDevice> numParamDOF_host("numParamDOF on host",paramusebasis.size());
  for (int i=0; i<paramusebasis.size(); i++) {
    numParamDOF_host(i) = paramnumbasis_[paramusebasis[i]];
  }
  numParamDOF = Kokkos::create_mirror_view(numParamDOF_host);
  Kokkos::deep_copy(numParamDOF_host, numParamDOF);
  
  
  size_t maxnbasis = 0;
  for (size_t i=0; i<numParamDOF.size(); i++) {
    if (numParamDOF(i) > maxnbasis) {
      maxnbasis = numParamDOF(i);
    }
  }
  param = Kokkos::View<double***,AssemblyDevice>("param",numElem,numParamDOF.size(),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setAuxUseBasis(vector<int> & ausebasis_) {
  auxusebasis = ausebasis_;
  size_t maxnbasis = 0;
  for (size_t i=0; i<numAuxDOF.size(); i++) {
    if (numAuxDOF(i) > maxnbasis) {
      maxnbasis = numAuxDOF(i);
    }
  }
  aux = Kokkos::View<double***,AssemblyDevice>("aux",numElem,numAuxDOF.size(),maxnbasis);
  
}


///////////////////////////////////////////////////////////////////////////////////////
// Set one of the local solution
///////////////////////////////////////////////////////////////////////////////////////

void cell::setLocalSoln(const vector_RCP & gl_vec, const int & type,
                        const size_t & entry){ //, const int & nstages) {
  
  // Here, nstages refers to the number of stages in gl_vec
  // which may be different from num_stages, but always nstages <= num_stages
  
  // In general, gl_vec will reside in host memory
  // We may want to thread this gather on host
  
  switch(type) {
    case 0 :
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index[e].size(); n++) {
          for(size_t i=0; i<index[e][n].size(); i++ ) {
            u(e,n,i) = (*gl_vec)[entry][index[e][n][i]];
          }
        }
      }
      break;
    case 1 :
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index[e].size(); n++) {
          for(size_t i=0; i<index[e][n].size(); i++ ) {
            u_dot(e,n,i) = (*gl_vec)[entry][index[e][n][i]];
          }
        }
      }
      break;
    case 2 :
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index[e].size(); n++) {
          for(size_t i=0; i<index[e][n].size(); i++ ) {
            phi(e,n,i) = (*gl_vec)[entry][index[e][n][i]];
          }
        }
      }
      break;
    case 3 :
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index[e].size(); n++) {
          for(size_t i=0; i<index[e][n].size(); i++ ) {
            phi_dot(e,n,i) = (*gl_vec)[entry][index[e][n][i]];
          }
        }
      }
      break;
    case 4 :
      for (int e=0; e<numElem; e++) {
        if (paramindex.size()>e) {
          for (size_t n=0; n<paramindex[e].size(); n++) {
            for(size_t i=0; i<paramindex[e][n].size(); i++ ) {
              param(e,n,i) = (*gl_vec)[entry][paramindex[e][n][i]];
            }
          }
        }
      }
      break;
    case 5 :
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<auxindex.size(); n++) {
          for(size_t i=0; i<auxindex[n].size(); i++ ) {
            aux(e,n,i) = (*gl_vec)[entry][auxindex[n][i]];
          }
        }
      }
      break;
    default :
      cout << "ERROR - NOTHING WAS GATHERED" << endl;
      
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolnVolIP(const bool & seedu, const bool & seedudot, const bool & seedparams,
                            const bool & seedaux) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnVolTimer);
  
  wkset->update(ip,ijac,orientation);
  wkset->computeSolnVolIP(u, u_dot, seedu, seedudot);
  wkset->computeParamVolIP(param, seedparams);
  
  if (wkset->numAux > 0) {
    wkset->resetAux();
    
    AD auxval;
    for (int e=0; e<numElem; e++) {
      if (auxindex.size() > e ) {
        for (size_t k=0; k<auxindex.size(); k++) {
          if (auxusebasis[k] < auxbasis.size()) {
            for( int i=0; i<auxindex[k].size(); i++ ) {
              
              if (seedaux) {
                auxval = AD(maxDerivs,auxoffsets[k][i],aux(e,k,i));
              }
              else {
                auxval = aux(e,k,i);
              }
              for( size_t j=0; j<ip.dimension(1); j++ ) {
                wkset->local_aux(e,k,j) += auxval*auxbasis[auxusebasis[k]](e,i,j);
                //for( int s=0; s<dimension; s++ ) {
                //  wkset->local_aux_grad(e,k,j,s) += auxval*auxbasisGrad[auxusebasis[k]](e,i,j,s);
                //}
              }
            }
          }
        }
      }
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the coarse grid solution to the fine grid integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolnSideIP(const int & side,
                             const bool & seedu, const bool & seedudot, const bool & seedparams,
                             const bool & seedaux) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnSideTimer);
  
  wkset->updateSide(nodes, sideip[side], sidewts[side],normals[side],sideijac[side], side);
  wkset->computeSolnSideIP(side, u, u_dot, seedu, seedudot);
  wkset->computeParamSideIP(side, param, seedparams);
  
  if (wkset->numAux > 0) {
    
    wkset->resetAuxSide();
    
    size_t numip = wkset->numsideip;
    AD auxval;
    
    for (int e=0; e<numElem; e++) {
      for (size_t k=0; k<auxindex.size(); k++) {
        for(size_t i=0; i<auxindex[k].size(); i++ ) {
          if (seedaux) {
            auxval = AD(maxDerivs,auxoffsets[k][i],aux(e,k,i));
          }
          else {
            auxval = aux(e,k,i);
          }
          for( size_t j=0; j<numip; j++ ) {
            wkset->local_aux_side(e,k,j) += auxval*auxside_basis[side][auxusebasis[k]](e,i,j);
            //for( int s=0; s<dimension; s++ ) {
            //  wkset->local_aux_grad_side(e,k,j,s) += auxval*auxside_basisGrad[side][auxusebasis[k]](e,i,j,s);
            //}
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the solution variables in the workset
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateSolnWorkset(const vector_RCP & gl_u, int tindex) {
  Kokkos::View<double***,AssemblyDevice> ulocal("tempory u", numElem,u.dimension(1),u.dimension(2));
  for (int e=0; e<numElem; e++) {
    for (size_t n=0; n<index[e].size(); n++) {
      for(size_t i=0; i<index[e][n].size(); i++ ) {
        ulocal(e,n,i) = (*gl_u)[tindex][index[e][n][i]];
      }
    }
  }
  wkset->update(ip,ijac,orientation);
  wkset->computeSolnVolIP(ulocal);
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the contribution from this cell to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeJacRes(const vector<vector<double> > & paramvals,
                         const vector<int> & paramtypes, const vector<string> & paramnames,
                         const double & time, const bool & isTransient, const bool & isAdjoint,
                         const bool & compute_jacobian, const bool & compute_sens,
                         const int & num_active_params, const bool & compute_disc_sens,
                         const bool & compute_aux_sens, const bool & store_adjPrev,
                         Kokkos::View<double***,AssemblyDevice> local_res,
                         Kokkos::View<double***,AssemblyDevice> local_J,
                         Kokkos::View<double***,AssemblyDevice> local_Jdot) {
  current_time = time;
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  if (multiscale) {
    
    wkset->resetResidual();
    
    for (int e=0; e<numElem; e++) {
      int sgindex = subgrid_model_index[e][subgrid_model_index.size()-1];
      
      subgridModels[sgindex]->subgridSolver(u, phi,
                                            paramvals, paramtypes, paramnames,time, isTransient, isAdjoint,
                                            compute_jacobian, compute_sens,num_active_params,
                                            compute_disc_sens, compute_aux_sens,
                                            *wkset, //local_res, local_J, local_Jdot,
                                            subgrid_usernum[e], e,
                                            subgradient, store_adjPrev);
      
    }
    
    //////////////////////////////////////////////////////////////
    // Fill in the coarse scale res and J
    //////////////////////////////////////////////////////////////
    
    this->updateRes(compute_sens, local_res);
    if (compute_jacobian) {
      if (isAdjoint) { // && !useSubGridAdjoint) {
        this->updateJac(false, local_J);
      }
      else {
        this->updateJac(false, local_J);
      }
    }
    
    if (compute_jacobian) {
      bool fixJacDiag = true;
      if (fixJacDiag) {
        double JTOL = 1.0E-8;
        
        for (int e=0; e<numElem; e++) {
          for (size_t n=0; n<wkset->offsets.dimension(0); n++) {
            for (size_t i=0; i<wkset->offsets.dimension(1); i++) {
              if (abs(local_J(e,wkset->offsets(n,i),wkset->offsets(n,i))) < JTOL) {
                local_res(e,wkset->offsets(n,i),0) = -u(e,n,i);
                
                
                for (size_t j=0; j<wkset->offsets.dimension(0); j++) {
                  double scale = 1.0/((double)wkset->offsets.dimension(1)-1.0);
                  local_J(e,wkset->offsets(n,i),wkset->offsets(n,j)) = -scale;
                  //local_J(wkset->offsets[n][j],wkset->offsets[n][i]) = 0.0;
                  
                  if (j!=i)
                    local_res(e,wkset->offsets(n,i),0) += scale*u(e,n,j);
                }
                local_J(e,wkset->offsets(n,i),wkset->offsets(n,i)) = 1.0;
              }
            }
          }
        }
        
      }
    }
    
    //timers[5]->stop();
    
    /* // TMW: not re-implemented yet
     if (mortar_objective && isAdjoint) {
     wkset->res.initialize(0.0);
     this->setLocalADSolns(false,false,false,false, u_AD, u_dot_AD, param_AD, aux_AD);
     this->computeSolnVolIP(u_AD, u_dot_AD, param_AD, aux_AD);
     int w = 1;
     FCAD obj = computeObjective(current_time, 0, w);
     int numDerivs;
     if (useSensors) {
     if (numSensors > 0) {
     for (int n=0; n<index.size(); n++) {
     for (int j=0; j<index[n].size(); j++) {
     for (int i=0; i<index[n].size(); i++) {
     for (int s=0; s<numSensors; s++) {
     if (w == 1) {
     local_res(wkset->offsets[n][j],0) += -obj(s).fastAccessDx(wkset->offsets[n][i])*sensorBasis[wkset->usebasis[n]](0,j,s);
     }
     else {
     local_res(wkset->offsets[n][j],0) += -obj(s).fastAccessDx(wkset->offsets[n][i])*sensorBasisGrad[wkset->usebasis[n]](0,j,s,w-2);
     }
     }
     }
     }
     }
     }
     }
     else {
     for (int n=0; n<index.size(); n++) {
     for (int j=0; j<index[n].size(); j++) {
     for (int i=0; i<index[n].size(); i++) {
     for (int s=0; s<wkset->numip; s++) {
     if (w == 1) {
     local_res(wkset->offsets[n][j],0) += -obj(s).fastAccessDx(wkset->offsets[n][i])*wkset->ref_basis[wkset->usebasis[n]](0,j,s);
     }
     else {
     local_res(wkset->offsets[n][j],0) += -obj(s).fastAccessDx(wkset->offsets[n][i])*wkset->basis_grad_uw[wkset->usebasis[n]](0,j,s,w-2);
     }
     }
     }
     }
     }
     }
     //this->updateRes(wkset->res, compute_sens, wkset->offsets, local_res);
     }
     */
    
  }
  else { // NON-MULTISCALE RESIDUAL, JACOBIAN, ETC.
    
    wkset->resetResidual();
    
    if (isAdjoint) {
      wkset->resetAdjointRHS();
    }
    
    //////////////////////////////////////////////////////////////
    // COmpute the AD-seeded solutions at integration points
    //////////////////////////////////////////////////////////////
    
    if (compute_jacobian) {
      if (compute_disc_sens) {
        this->computeSolnVolIP(false,false,true,false);
      }
      else if (compute_aux_sens) {
        this->computeSolnVolIP(false,false,false,true);
      }
      else {
        this->computeSolnVolIP(true,false,false,false);
      }
    }
    else {
      this->computeSolnVolIP(false,false,false,false);
    }
    
    //////////////////////////////////////////////////////////////
    // Compute res and J=dF/du
    //////////////////////////////////////////////////////////////
    
    // Volumetric contribution
    
    {
      Teuchos::TimeMonitor localtimer(*volumeResidualTimer);
      physics_RCP->volumeResidual(myBlock);
    }
    
    // Boundary contribution
    
    
    {
      Teuchos::TimeMonitor localtimer(*boundaryResidualTimer);
      
      
      for (int side=0; side<numSides; side++) {
        bool compute = false; // not going to work if Host!=Assembly
        string gsideid;
        for (int e=0; e<sideinfo.dimension(0); e++) {
          for (int n=0; n<sideinfo.dimension(1); n++) {
            if (sideinfo(e,n,side,0) > 0) {
              compute = true;
              if (sideinfo(e,n,side,1) >= 0) {
                gsideid = sidenames[sideinfo(e,n,side,1)];
              }
              else { // = -1
                gsideid = "interior";
              }
            }
          }
        }
        if (compute) {
          
          wkset->sideinfo = sideinfo;
          wkset->currentside = side;
          // if (sideinfo[e](side,1) == -1) {
          //   wkset->sidename = "interior";
          //   wkset->sidetype = -1;
          // }
          // else {
          wkset->sidename = gsideid;
          //wkset->sidetype = sideinfo[e](side,0);
          // }
          
          if (compute_jacobian) {
            if (compute_disc_sens) {
              this->computeSolnSideIP(side,false,false,true,false);
            }
            else if (compute_aux_sens) {
              this->computeSolnSideIP(side,false,false,false,true);
            }
            else {
              this->computeSolnSideIP(side,true,false,false,false);
            }
          }
          else {
            this->computeSolnSideIP(side,false,false,false,false);
          }
          
          physics_RCP->boundaryResidual(myBlock);
          
        }
      }
    }
    
    // Edge contribution
    //for (int side=0; side<numSides; side++) {
    //  bool compute_eres = true;
    //  if (sideinfo[e](side,0) == 1) {
    //    compute_eres = false;
    //  }
    //  if (compute_eres) {
    //    // not implemented yet
    //  }
    //}
    
    {
      Teuchos::TimeMonitor localtimer(*jacobianFillTimer);
      
      // Use AD residual to update local Jacobian
      if (compute_jacobian) {
        if (compute_disc_sens) {
          this->updateParamJac(local_J);
        }
        else if (compute_aux_sens){
          this->updateAuxJac(local_J);
        }
        else {
          this->updateJac(isAdjoint, local_J);
        }
      }
    }
    
    {
      Teuchos::TimeMonitor localtimer(*residualFillTimer);
      
      // Update the local residual (forward mode)
      if (isAdjoint) {
        this->updateAdjointRes(compute_sens, local_res);
      }
      else {
        this->updateRes(compute_sens, local_res);
      }
      
    }
    
    {
      Teuchos::TimeMonitor localtimer(*transientResidualTimer);
      if (isTransient && compute_jacobian) {
        //this->resetWorkset(wkset);
        if (compute_jacobian) {
          if (compute_disc_sens) {
            this->computeSolnVolIP(false,false,true,false);
            //this->setLocalADSolns(false,false,true,false, u_AD, u_dot_AD, param_AD, aux_AD);
          }
          else if (compute_aux_sens) {
            this->computeSolnVolIP(false,false,false,true);
            //  this->setLocalADSolns(false,false,false,true, u_AD, u_dot_AD, param_AD, aux_AD);
          }
          else {
            this->computeSolnVolIP(false,true,false,false);
            //this->setLocalADSolns(false,true,false,false, u_AD, u_dot_AD, param_AD, aux_AD);
          }
        }
        else {
          this->computeSolnVolIP(false,false,false,false);
          //this->setLocalADSolns(false,false,false,false, u_AD, u_dot_AD, param_AD, aux_AD);
        }
        wkset->resetResidual();//res.initialize(0.0);// = FCAD(GIDs.size());
        
        
        // evaluate the local solutions at the volumetric integration points
        
        //this->computeSolnVolIP(u_AD, u_dot_AD, param_AD, aux_AD);
        
        physics_RCP->volumeResidual(myBlock);
        
        // Update the local transient Jacobian
        if (compute_disc_sens) {
          this->updateParamJacDot(local_Jdot);
        }
        else if (compute_aux_sens) {
          //  this->updateAuxJacDot(wkset->res);
        }
        else {
          this->updateJacDot(isAdjoint, local_Jdot);
        }
      }
    }
    
    {
      Teuchos::TimeMonitor localtimer(*adjointResidualTimer);
      // Update residual (adjoint mode)
      if (isAdjoint) {
        if (!mortar_objective) {
          for (int w=1; w < dimension+2; w++) {
            
            Kokkos::View<AD**,AssemblyDevice> obj = computeObjective(current_time, 0, w);
            
            int numDerivs;
            if (useSensors) {
              if (numSensors > 0) {
                //for (int e=0; e<numSensors; e++) {
                for (int s=0; s<numSensors; s++) {
                  int e = sensorElem[s];
                  for (int n=0; n<index[e].size(); n++) {
                    for (int j=0; j<index[e][n].size(); j++) {
                      for (int i=0; i<index[e][n].size(); i++) {
                        if (w == 1) {
                          local_res(e,wkset->offsets(n,j),0) += -obj(e,s).fastAccessDx(wkset->offsets(n,i))*sensorBasis[s][wkset->usebasis[n]](0,j,s);
                        }
                        else {
                          local_res(e,wkset->offsets(n,j),0) += -obj(e,s).fastAccessDx(wkset->offsets(n,i))*sensorBasisGrad[s][wkset->usebasis[n]](0,j,s,w-2);
                        }
                      }
                    }
                  }
                }
              }
            }
            else {
              for (int e=0; e<numElem; e++) {
                for (int n=0; n<index[e].size(); n++) {
                  for (int j=0; j<index[e][n].size(); j++) {
                    for (int i=0; i<index[e][n].size(); i++) {
                      for (int s=0; s<wkset->numip; s++) {
                        if (w == 1) {
                          local_res(e,wkset->offsets(n,j),0) += -obj(e,s).fastAccessDx(wkset->offsets(n,i))*wkset->ref_basis[wkset->usebasis[n]](e,j,s);
                        }
                        else {
                          local_res(e,wkset->offsets(n,j),0) += -obj(e,s).fastAccessDx(wkset->offsets(n,i))*wkset->basis_grad_uw[wkset->usebasis[n]](e,j,s,w-2);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (compute_jacobian) {
          for (int e=0; e<numElem; e++) {
            for (int n=0; n<index[e].size(); n++) {
              for (int j=0; j<index[e][n].size(); j++) {
                for (int m=0; m<index[e].size(); m++) {
                  for (int k=0; k<index[e][m].size(); k++) {
                    local_res(e,wkset->offsets(n,j),0) += -local_J(e,wkset->offsets(n,j),wkset->offsets(m,k))*phi(e,m,k);
                  }
                }
              }
            }
          }
          if (isTransient) {
            for (int e=0; e<numElem; e++) {
              for (int n=0; n<index[e].size(); n++) {
                for (int j=0; j<index[e][n].size(); j++) {
                  double aPrev = 0.0;
                  for (int m=0; m<index[e].size(); m++) {
                    for (int k=0; k<index[e][m].size(); k++) {
                      local_res(e,wkset->offsets(n,j),0) += -wkset->alpha*local_Jdot(e,wkset->offsets(n,j),wkset->offsets(m,k))*phi(e,m,k);
                      aPrev += wkset->alpha*local_Jdot(e,wkset->offsets(n,j),wkset->offsets(m,k))*phi(e,m,k);
                    }
                  }
                  local_res(e,wkset->offsets(n,j),0) += this->adjPrev(e,wkset->offsets(n,j));
                  if (!compute_aux_sens && store_adjPrev) {
                    adjPrev(e,wkset->offsets(n,j)) = aPrev;
                  }
                }
              }
            }
          }
        }
        for (int side=0; side<numSides; side++) {
          
          /*
           if (sideinfo(side,0)>0 && sideinfo(side,1) == -1) {
           
           wkset->sidename = "interior";
           wkset->sidetype = -1;
           
           //this->computeSolnSideIP(side, u_AD, u_dot_AD, param_AD, aux_AD, wkset);
           wkset->resetSide();
           wkset->res.initialize(0.0);
           
           //wkset->computeSolnSideIP(side, u_AD, u_dot_AD);
           
           wkset->local_aux_side.initialize(0.0);
           wkset->local_aux_grad_side.initialize(0.0);
           
           size_t numip = wkset->numsideip;
           for (size_t k=0; k<auxindex.size(); k++) {
           for(size_t i=0; i<auxindex[k].size(); i++ ) {
           for( size_t j=0; j<numip; j++ ) {
           wkset->local_aux_side(k,j) += aux_AD[k][i]*auxside_basis[side][auxusebasis[k]](0,i,j);
           for( int s=0; s<dimension; s++ ) {
           wkset->local_aux_grad_side(k,j,s) += aux_AD[k][i]*auxside_basisGrad[side][auxusebasis[k]](0,i,j,s);
           }
           }
           }
           }
           
           physics_RCP->boundaryResidual(wkset);
           //cout << "wkset->res = " << wkset->res << endl;
           //cout << "local_res = " << local_res << endl;
           
           //->updateRes(wkset->res, compute_sens, wkset->offsets, local_res);
           
           //physics_RCP->computeFlux(wkset);
           //cout << "flux = " << wkset->flux << endl;
           }
           //timers[2]->stop();
           //cout << "timer 2: " << timers[2]->totalElapsedTime() << endl;
           
           */
        }
        
        
      }
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateRes(const bool & compute_sens, Kokkos::View<double***,AssemblyDevice> local_res) {
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  if (compute_sens) {
    for (int e=0; e<numElem; e++) {
      for (int r=0; r<local_res.dimension(2); r++) {
        for (int n=0; n<index[e].size(); n++) {
          for (int j=0; j<index[e][n].size(); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    }
  }
  else {
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index[e].size(); n++) {
        for (int j=0; j<index[e][n].size(); j++) {
          local_res(e,offsets(n,j),0) -= res_AD(e,offsets(n,j)).val();
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAdjointRes(const bool & compute_sens, Kokkos::View<double***,AssemblyDevice> local_res) {
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->adjrhs;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  if (compute_sens) {
    for (int e=0; e<numElem; e++) {
      for (int r=0; r<maxDerivs; r++) {
        for (int n=0; n<index[e].size(); n++) {
          for (int j=0; j<index[e][n].size(); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    }
  }
  else {
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index[e].size(); n++) {
        for (int j=0; j<index[e][n].size(); j++) {
          local_res(e,offsets(n,j),0) -= res_AD(e,offsets(n,j)).val();
        }
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateJac(const bool & useadjoint, Kokkos::View<double***,AssemblyDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  
  if (useadjoint) {
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index[e].size(); n++) {
        for (int j=0; j<index[e][n].size(); j++) {
          for (int m=0; m<index[e].size(); m++) {
            for (int k=0; k<index[e][m].size(); k++) {
              local_J(e,offsets(m,k),offsets(n,j)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    }
  }
  else {
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index[e].size(); n++) {
        for (int j=0; j<index[e][n].size(); j++) {
          for (int m=0; m<index[e].size(); m++) {
            for (int k=0; k<index[e][m].size(); k++) {
              local_J(e,offsets(n,j),offsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jdot
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateJacDot(const bool & useadjoint, Kokkos::View<double***,AssemblyDevice> local_Jdot) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  
  if (useadjoint) {
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index[e].size(); n++) {
        for (int j=0; j<index[e][n].size(); j++) {
          for (int m=0; m<index[e].size(); m++) {
            for (int k=0; k<index[e][m].size(); k++) {
              local_Jdot(e,offsets(m,k),offsets(n,j)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    }
  }
  else {
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index[e].size(); n++) {
        for (int j=0; j<index[e][n].size(); j++) {
          for (int m=0; m<index[e].size(); m++) {
            for (int k=0; k<index[e][m].size(); k++) {
              local_Jdot(e,offsets(n,j),offsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    }
  }
  bool lumpmass = false;
  /* // TMW: Commented this out since have it hard-coded to false
   if (lumpmass) {
   FC Jdotold = local_Jdot;
   local_Jdot.initialize(0.0);
   //his->resetJacDot();
   for (int e=0; e<numElem; e++) {
   for (int n=0; n<GIDs[e].size(); n++) {
   for (int m=0; m<GIDs[e].size(); m++) {
   local_Jdot(e,n,n) += Jdotold(e,n,m);
   }
   }
   }
   }*/
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateParamJac(Kokkos::View<double***,AssemblyDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<int**,AssemblyDevice> paramoffsets = wkset->paramoffsets;
  
  for (int e=0; e<numElem; e++) {
    for (int n=0; n<index[e].size(); n++) {
      for (int j=0; j<index[e][n].size(); j++) {
        for (int m=0; m<paramindex[e].size(); m++) {
          for (int k=0; k<paramindex[e][m].size(); k++) {
            local_J(e,offsets(n,j),paramoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparamdot
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateParamJacDot(Kokkos::View<double***,AssemblyDevice> local_Jdot) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<int**,AssemblyDevice> paramoffsets = wkset->paramoffsets;
  
  for (int e=0; e<numElem; e++) {
    for (int n=0; n<index[e].size(); n++) {
      for (int j=0; j<index[e][n].size(); j++) {
        for (int m=0; m<paramindex[e].size(); m++) {
          for (int k=0; k<paramindex[e][m].size(); k++) {
            local_Jdot(e,offsets(n,j),paramoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAuxJac(Kokkos::View<double***,AssemblyDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  
  for (int e=0; e<numElem; e++) {
    for (int n=0; n<index[e].size(); n++) {
      for (int j=0; j<index[e][n].size(); j++) {
        for (int m=0; m<auxindex.size(); m++) {
          for (int k=0; k<auxindex[m].size(); k++) {
            local_J(e,offsets(n,j),auxoffsets[m][k]) += res_AD(e,offsets(n,j)).fastAccessDx(auxoffsets[m][k]);
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparamdot
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAuxJacDot(Kokkos::View<double***,AssemblyDevice> local_Jdot) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  
  for (int e=0; e<numElem; e++) {
    for (int n=0; n<index[e].size(); n++) {
      for (int j=0; j<index[e][n].size(); j++) {
        for (int m=0; m<auxindex.size(); m++) {
          for (int k=0; k<auxindex[m].size(); k++) {
            local_Jdot(e,offsets(n,j),auxoffsets[m][k]) += res_AD(e,offsets(n,j)).fastAccessDx(auxoffsets[m][k]);
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<double**,AssemblyDevice> cell::getInitial(const bool & project, const bool & isAdjoint) {
  Kokkos::View<double**,AssemblyDevice> initialvals("initial values",numElem,GIDs[0].size());
  wkset->update(ip,ijac,orientation);
  if (project) { // works for any basis
    for (int n=0; n<wkset->varlist.size(); n++) {
      Kokkos::View<double**,AssemblyDevice> initialip = physics_RCP->getInitial(wkset->ip,
                                                                                wkset->varlist[n],
                                                                                current_time,
                                                                                isAdjoint,
                                                                                wkset);
      for (int e=0; e<numElem; e++) {
        for( int i=0; i<index[e][n].size(); i++ ) {
          for( size_t j=0; j<wkset->numip; j++ ) {
            initialvals(e,wkset->offsets(n,i)) += initialip(e,j)*wkset->basis[wkset->usebasis[n]](e,i,j);
          }
        }
      }
    }
  }
  else { // only works if using HGRAD linear basis
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index[e].size(); n++) {
        Kokkos::View<double**,AssemblyDevice> initialnodes = physics_RCP->getInitial(nodes,
                                                                                     wkset->varlist[n],
                                                                                     current_time,
                                                                                     isAdjoint,
                                                                                     wkset);
        for( int i=0; i<index[e][n].size(); i++ ) {
          initialvals(e,wkset->offsets(n,i)) = initialnodes(e,i);
        }
      }
    }
  }
  return initialvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<double***,AssemblyDevice> cell::getMass() {
  Kokkos::View<double***,AssemblyDevice> mass("local mass",numElem,GIDs[0].size(), GIDs[0].size());
  wkset->update(ip,ijac,orientation);
  vector<string> basis_types = wkset->basis_types;
  
  for (int e=0; e<numElem; e++) {
    for (int n=0; n<index[e].size(); n++) {
      for( int i=0; i<index[e][n].size(); i++ ) {
        for (int m=0; m<index[e].size(); m++) {
          if (n == m) {
            for( int j=0; j<index[e][m].size(); j++ ) {
              for( size_t k=0; k<wkset->numip; k++ ) {
                mass(e,wkset->offsets(n,i),wkset->offsets(m,j)) += wkset->basis_uw[wkset->usebasis[n]](e,i,k)*wkset->basis[wkset->usebasis[m]](e,j,k);
              }
            }
          }
        }
      }
    }
  }
  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the error at the integration points given the solution and solve times
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<double**,AssemblyDevice> cell::computeError(const double & solvetime,
                                                         const size_t & tindex,
                                                         const bool compute_subgrid,
                                                         const string & error_type) {
  
  // Assumes that u has been updated already
  wkset->time = solvetime;
  wkset->time_KV(0) = solvetime;
  
  Kokkos::View<double**,AssemblyDevice> errors("errors",numElem,index[0].size());
  if (!compute_subgrid) {
    wkset->update(ip,ijac,orientation);
    wkset->computeSolnVolIP(u, u_dot, false, false);
    size_t numip = wkset->numip;
    
    /*
     size_t numip = wkset->numip;
     Kokkos::View<double***,AssemblyDevice> u_ip("u_ip",numElem,index[0].size(),numip);
     for (int e=0; e<numElem; e++) {
     for (int n=0; n<index[e].size(); n++) {
     for( int i=0; i<index[e][n].size(); i++ ) {
     for( size_t j=0; j<numip; j++ ) {
     u_ip(e,n,j) += u(e,n,i)*wkset->ref_basis[wkset->usebasis[n]](e,i,j);
     }
     }
     }
     }
     */
    
    if (error_type == "L2") {
      Kokkos::View<double***,AssemblyDevice> truesol("true solution",numElem,index[0].size(),numip);
      physics_RCP->trueSolution(myBlock, solvetime, truesol);
      //KokkosTools::print(wkset->local_soln);
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<index[e].size(); n++) {
          for( size_t j=0; j<numip; j++ ) {
            double diff = wkset->local_soln(e,n,j,0).val() - truesol(e,n,j);
            errors(e,n) += diff*diff*wkset->wts(e,j);
          }
        }
      }
    }
    if (error_type == "H1") {
      Kokkos::View<double****,AssemblyDevice> truesol("true solution",numElem,index[0].size(),numip,dimension);
      physics_RCP->trueSolutionGrad(myBlock, solvetime, truesol);
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<index[e].size(); n++) {
          for( size_t j=0; j<numip; j++ ) {
            for (int s=0; s<dimension; s++) {
              double diff = wkset->local_soln_grad(e,n,j,s).val() - truesol(e,n,j,s);
              errors(e,n) += diff*diff*wkset->wts(e,j);
            }
          }
        }
      }
    }
    
    /*
     for (int e=0; e<numElem; e++) {
     for (int n=0; n<index[e].size(); n++) {
     for( size_t j=0; j<numip; j++ ) {
     double x = wkset->ip(e,j,0);
     double y = 0.0;
     if (dimension > 1) {
     y = wkset->ip(e,j,1);
     }
     double z = 0.0;
     if (dimension > 2) {
     z = wkset->ip(e,j,2);
     }
     double truesol = physics_RCP->trueSolution(myBlock, wkset->varlist[n], x, y, z, solvetime, error_type);
     errors(e,n) += (u_ip(e,n,j)-truesol)*(u_ip(e,n,j)-truesol)*wkset->wts(e,j);
     }
     }
     }
     */
    
  }
  else if (multiscale) {
    
    for (int e=0; e<numElem; e++) {
      
      Kokkos::View<double**,AssemblyDevice> currerrors = subgridModels[subgrid_model_index[e][tindex]]->computeError(solvetime, subgrid_usernum[e]);
      
      for (int c=0; c<currerrors.dimension(0); c++) { // loop over subgrid elements
        for (int n=0; n<index[e].size(); n++) {
          errors(e,n) += currerrors(c,n);
        }
      }
    }
    
  }
  return errors;
}


///////////////////////////////////////////////////////////////////////////////////////
// Compute the response at a given set of points and time
///////////////////////////////////////////////////////////////////////////////////////
Kokkos::View<AD***,AssemblyDevice> cell::computeResponseAtNodes(const DRV & nodes,
                                                                const int tindex,
                                                                const double & time) {
  
  
  // TMW: this whole function needs to be rewritten to use worksets properly
  /*
   size_t numip = nodes.dimension(1);
   wkset->update(ip,ijac,orientation);
   
   // Map the local solution to the solution and gradient at ip
   FCAD u_ip(numElem,index[0].size(),numip);
   FCAD ugrad_ip(numElem,index[0].size(),numip,dimension);
   for (int e=0; e<numElem; e++) {
   for (int n=0; n<index[e].size(); n++) {
   for( int i=0; i<index[e][n].size(); i++ ) {
   for( size_t j=0; j<numip; j++ ) {
   u_ip(e,n,j) += u[e][n][i]*wkset->ref_basis[wkset->usebasis[n]](e,i,j);
   for (int s=0; s<dimension; s++) {
   ugrad_ip(e,n,j,s) += u[e][n][i]*wkset->basis_grad_uw[wkset->usebasis[n]](e,i,j,s);
   }
   }
   }
   }
   }
   
   bool seedParams = false;
   //vector<vector<AD> > param_AD;
   //for (int n=0; n<paramindex.size(); n++) {
   //  param_AD.push_back(vector<AD>(paramindex[n].size()));
   //}
   //this->setLocalADParams(param_AD,seedParams);//, wkset);
   //FCAD p_ip(numElem,paramindex[e].size(), numip);
   //FCAD pgrad_ip(numElem,paramindex[e].size(), numip, dimension);
   wkset->computeParamVolIP(param,seedParams);
   
   FCAD response = physics_RCP->getResponse(myBlock, u_ip, ugrad_ip, wkset->local_param,
   wkset->local_param_grad, wkset->ip, time);
   return response;
   */
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the response at the integration points given the solution and solve time
///////////////////////////////////////////////////////////////////////////////////////
//
Kokkos::View<AD***,AssemblyDevice> cell::computeResponse(const double & solvetime,
                                                         const size_t & tindex,
                                                         const int & seedwhat) {
  
  // Assumes that u has already been filled
  
  // seedwhat indicates what needs to be seeded
  // seedwhat == 0 => seed nothing
  // seedwhat == 1 => seed sol
  // seedwhat == j (j>1) => seed (j-1)-derivative of sol
  
  
  Kokkos::View<AD***,AssemblyDevice> response;
  bool useSensors = false;
  if (response_type == "pointwise") {
    useSensors = true;
  }
  
  size_t numip = wkset->ip.dimension(1);
  if (useSensors) {
    numip = sensorLocations.size();
  }
  
  wkset->update(ip,ijac,orientation);
  
  //KokkosTools::print(u);
  if (numip > 0) {
    // Extract the local solution at this time
    // We automatically seed the AD and adjust it below
    Kokkos::View<AD***,AssemblyDevice> u_dof("u_dof",numElem,index[0].size(),GIDs[0].size());
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index[e].size(); n++) {
        for( int i=0; i<index[e][n].size(); i++ ) {
          u_dof(e,n,i) = AD(maxDerivs,wkset->offsets(n,i),u(e,n,i));
        }
      }
    }
    
    // Map the local solution to the solution and gradient at ip
    Kokkos::View<AD****,AssemblyDevice> u_ip("u_ip",numElem,index[0].size(),numip,dimension);
    Kokkos::View<AD****,AssemblyDevice> ugrad_ip("ugrad_ip",numElem,index[0].size(),numip,dimension);
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index[e].size(); n++) {
        for( int i=0; i<index[e][n].size(); i++ ) {
          if (useSensors) {
            for (int ee=0; ee<numSensors; ee++) {
              int eind = sensorElem[ee];
              if (eind ==e) {
                u_ip(eind,n,ee,0) += u_dof(eind,n,i)*sensorBasis[ee][wkset->usebasis[n]](0,i,0);
              }
            }
          }
          else {
            for( size_t j=0; j<numip; j++ ) {
              u_ip(e,n,j,0) += u_dof(e,n,i)*wkset->ref_basis[wkset->usebasis[n]](e,i,j);
            }
          }
          for (int s=0; s<dimension; s++) {
            if (useSensors) {
              for (int ee=0; ee<numSensors; ee++) {
                int eind = sensorElem[ee];
                if (eind == e) {
                  ugrad_ip(eind,n,ee,s) += u_dof(eind,n,i)*sensorBasisGrad[ee][wkset->usebasis[n]](0,i,0,s);
                }
              }
            }
            else {
              for( size_t j=0; j<numip; j++ ) {
                ugrad_ip(e,n,j,s) += u_dof(e,n,i)*wkset->basis_grad_uw[wkset->usebasis[n]](e,i,j,s);
              }
            }
          }
        }
      }
    }
    
    // Adjust the AD based on seedwhat
    if (seedwhat == 0) { // remove all seeding
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<index[e].size(); n++) {
          for( size_t j=0; j<numip; j++ ) {
            u_ip(e,n,j,0) = u_ip(e,n,j,0).val();
            for (int s=0; s<dimension; s++) {
              ugrad_ip(e,n,j,s) = ugrad_ip(e,n,j,s).val();
            }
          }
        }
      }
    }
    else if (seedwhat == 1) { // remove seeding on gradient
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<index[e].size(); n++) {
          for( size_t j=0; j<numip; j++ ) {
            for (int s=0; s<dimension; s++) {
              ugrad_ip(e,n,j,s) = ugrad_ip(e,n,j,s).val();
            }
          }
        }
      }
      //KokkosTools::print(u_ip);
      
    }
    else {
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<index[e].size(); n++) {
          for( size_t j=0; j<numip; j++ ) {
            for (int s=0; s<dimension; s++) {
              if ((seedwhat-2) == s) {
                double tmp = ugrad_ip(e,n,j,s).val();
                ugrad_ip(e,n,j,s) = u_ip(e,n,j,0);
                ugrad_ip(e,n,j,s) += -u_ip(e,n,j,0).val() + tmp;
              }
              else {
                ugrad_ip(e,n,j,s) = ugrad_ip(e,n,j,s).val();
              }
            }
            u_ip(e,n,j,0) = u_ip(e,n,j,0).val();
          }
        }
      }
    }
    
    bool seedParams = false;
    if (seedwhat == 0) {
      seedParams = true;
    }
    
    Kokkos::View<AD****,AssemblyDevice> param_ip;
    Kokkos::View<AD****,AssemblyDevice> paramgrad_ip;
    
    if (paramindex.size() > 0) {
      // Extract the local solution at this time
      // We automatically seed the AD and adjust it below
      Kokkos::View<AD***,AssemblyDevice> param_dof("param dof",numElem,paramindex[0].size(),paramGIDs[0].size());
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<paramindex[e].size(); n++) {
          for( int i=0; i<paramindex[e][n].size(); i++ ) {
            param_dof(e,n,i) = AD(maxDerivs,wkset->paramoffsets(n,i),param(e,n,i));
          }
        }
      }
      
      // Map the local solution to the solution and gradient at ip
      param_ip = Kokkos::View<AD****,AssemblyDevice>("u_ip",numElem,paramindex[0].size(),numip,dimension);
      paramgrad_ip = Kokkos::View<AD****,AssemblyDevice>("ugrad_ip",numElem,paramindex[0].size(),numip,dimension);
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<paramindex[e].size(); n++) {
          for( int i=0; i<paramindex[e][n].size(); i++ ) {
            if (useSensors) {
              for (int ee=0; ee<numSensors; ee++) {
                int eind = sensorElem[ee];
                if (eind ==e) {
                  param_ip(eind,n,ee,0) += param_dof(eind,n,i)*param_sensorBasis[ee][wkset->paramusebasis[n]](0,i,0);
                }
              }
            }
            else {
              for( size_t j=0; j<numip; j++ ) {
                param_ip(e,n,j,0) += param_dof(e,n,i)*wkset->param_basis[wkset->paramusebasis[n]](e,i,j);
              }
            }
            for (int s=0; s<dimension; s++) {
              if (useSensors) {
                for (int ee=0; ee<numSensors; ee++) {
                  int eind = sensorElem[ee];
                  if (eind == e) {
                    paramgrad_ip(eind,n,ee,s) += param_dof(eind,n,i)*param_sensorBasisGrad[ee][wkset->paramusebasis[n]](0,i,0,s);
                  }
                }
              }
              else {
                for( size_t j=0; j<numip; j++ ) {
                  paramgrad_ip(e,n,j,s) += param_dof(e,n,i)*wkset->param_basis_grad[wkset->paramusebasis[n]](e,i,j,s);
                }
              }
            }
          }
        }
      }
      
      // Adjust the AD based on seedwhat
      if (seedwhat == 0) { // remove seeding on grad
        for (int e=0; e<numElem; e++) {
          for (int n=0; n<paramindex[e].size(); n++) {
            for( size_t j=0; j<numip; j++ ) {
              for (int s=0; s<dimension; s++) {
                paramgrad_ip(e,n,j,s) = paramgrad_ip(e,n,j,s).val();
              }
            }
          }
        }
      }
      else {
        for (int e=0; e<numElem; e++) {
          for (int n=0; n<paramindex[e].size(); n++) {
            for( size_t j=0; j<numip; j++ ) {
              param_ip(e,n,j,0) = param_ip(e,n,j,0).val();
              for (int s=0; s<dimension; s++) {
                paramgrad_ip(e,n,j,s) = paramgrad_ip(e,n,j,s).val();
              }
            }
          }
        }
      }
    }
    
    if (useSensors) {
      if (sensorLocations.size() > 0){
        response = physics_RCP->getResponse(myBlock, u_ip, ugrad_ip, param_ip,
                                            paramgrad_ip, sensorPoints,
                                            solvetime, wkset);
      }
    }
    else {
      response = physics_RCP->getResponse(myBlock, u_ip, ugrad_ip, param_ip,
                                          paramgrad_ip, wkset->ip,
                                          solvetime, wkset);
    }
  }
  if (seedwhat == 1) {
    //  KokkosTools::print(response);
    //  KokkosTools::print(wkset->local_soln_point);
  }
  
  return response;
  
}


///////////////////////////////////////////////////////////////////////////////////////
// Compute the objective function given the solution and solve times
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD**,AssemblyDevice> cell::computeObjective(const double & solvetime,
                                                         const size_t & tindex,
                                                         const int & seedwhat) {
  
  // assumes the params have been seeded elsewhere (solver, postprocess interfaces)
  Kokkos::View<AD**,AssemblyDevice> objective;
  
  if (!multiscale || mortar_objective) {
    
    Kokkos::View<AD***,AssemblyDevice> responsevals = computeResponse(solvetime,tindex,seedwhat);
    
    if (response_type == "pointwise") { // uses sensor data
      
      double TOL = 1.0e-6; // tolerance for comparing sensor times and simulation times
      objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,numSensors);
      
      if (numSensors > 0) { // if this element has any sensors
        for (size_t s=0; s<numSensors; s++) {
          bool foundtime = false;
          size_t ftime;
          
          for (size_t t2=0; t2<sensorData[s].dimension(0); t2++) {
            double stime = sensorData[s](t2,0);
            if (abs(stime-solvetime) < TOL) {
              foundtime = true;
              ftime = t2;
            }
          }
          
          if (foundtime) {
            int ee = sensorElem[s];
            for (size_t r=0; r<responsevals.dimension(1); r++) {
              AD rval = responsevals(ee,r,s);
              double sval = sensorData[s](ftime,r+1);
              if(compute_diff) {
                objective(ee,s) += 0.5*wkset->deltat*(rval-sval) * (rval-sval);
              }
              else {
                objective(ee,s) += wkset->deltat*rval;
              }
            }
          }
        }
      }
      
    }
    else if (response_type == "global") { // uses physicsmodules->target
      objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,wkset->ip.dimension(1));
      Kokkos::View<AD***,AssemblyDevice> ctarg = computeTarget(solvetime);
      Kokkos::View<AD***,AssemblyDevice> cweight = computeWeight(solvetime);
      
      for (int e=0; e<numElem; e++) {
        for (size_t r=0; r<responsevals.dimension(1); r++) {
          for (size_t k=0; k<wkset->ip.dimension(1); k++) {
            AD diff = responsevals(e,r,k)-ctarg(e,r,k);
            if(compute_diff) {
              objective(e,k) += 0.5*wkset->deltat*cweight(e,r,k)*(diff)*(diff)*wkset->wts(e,k);
              //objective(e,k) += 0.5*wkset->deltat*(diff)*(diff)*wkset->wts(e,k);
            }
            else {
              objective(e,k) += wkset->deltat*responsevals(e,r,k)*wkset->wts(e,k);
            }
          }
        }
      }
    }
    
  }
  else {
    
    for (int e=0; e<numElem; e++) {
      int sgindex = subgrid_model_index[e][tindex];
      Kokkos::View<AD*,AssemblyDevice> cobj = subgridModels[sgindex]->computeObjective(response_type,seedwhat,
                                                                                       solvetime,subgrid_usernum[e]);
      
      if (e == 0) {
        objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,cobj.dimension(0));
      }
      for (int i=0; i<cobj.dimension(0); i++) {
        objective(e,i) += cobj(i);
      }
      
      
    }
  }
  
  return objective;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute boundary regularization given the boundary discretized parameters
///////////////////////////////////////////////////////////////////////////////////////
AD cell::computeBoundaryRegularization(const vector<double> reg_constants, const vector<int> reg_types,
                                       const vector<int> reg_indices, const vector<string> reg_sides) {
  
  AD reg;
  bool seedParams = true;
  //vector<vector<AD> > param_AD;
  //for (int n=0; n<paramindex.size(); n++) {
  //  param_AD.push_back(vector<AD>(paramindex[n].size()));
  //}
  //this->setLocalADParams(param_AD,seedParams);
  int numip = wkset->numip;
  int numParams = reg_indices.size();
  bool onside = false;
  string sname;
  for (int side=0; side<numSides; side++) {
    for (int e=0; e<numElem; e++) {
      if (sideinfo(e,0,side,0) > 0) { // Just checking the first variable should be sufficient
        onside = true;
        sname = sidenames[sideinfo(e,0,side,1)];
      }
    }
    
    if (onside) {
      //int sidetype = sideinfo[e](side,0); // 0-not on bndry, 1-Dirichlet bndry, 2-Neumann bndry
      //if (sidetype > 0) {
      //wkset->updateSide(nodes, sideip[side], sideijac[side], side);
      
      wkset->updateSide(nodes, sideip[side], sidewts[side],normals[side],sideijac[side], side);
      
      int numip = wkset->numsideip;
      //int gside = sideinfo[e](side,1); // =-1 if is an interior edge
      
      DRV side_weights = wkset->wts_side;
      int paramIndex, reg_type;
      double reg_constant;
      string reg_side;
      size_t found;
      
      for (int i = 0; i < numParams; i++) {
        paramIndex = reg_indices[i];
        reg_constant = reg_constants[i];
        reg_type = reg_types[i];
        reg_side = reg_sides[i];
        found = reg_side.find(sname);
        if (found != string::npos) {
          
          wkset->computeParamSideIP(side, param, seedParams);
          
          AD p, dpdx, dpdy, dpdz; // parameters
          double offset = 1.0e-5;
          for (int e=0; e<numElem; e++) {
            if (sideinfo(e,0,side,0) > 0) {
              for (int k = 0; k < numip; k++) {
                p = wkset->local_param_side(e,paramIndex,k);
                // L2
                if (reg_type == 0) {
                  reg += 0.5*reg_constant*p*p*side_weights(e,k);
                }
                else {
                  AD sx, sy ,sz;
                  AD normal_dot;
                  dpdx = wkset->local_param_grad_side(e,paramIndex,k,0); // param 0 in single trac inversion
                  if (dimension > 1) {
                    dpdy = wkset->local_param_grad_side(e,paramIndex,k,1);
                  }
                  if (dimension > 2) {
                    dpdz = wkset->local_param_grad_side(e,paramIndex,k,2);
                  }
                  if (dimension == 1) {
                    normal_dot = dpdx*wkset->normals(e,k,0);
                    sx = dpdx - normal_dot*wkset->normals(e,k,0);
                  }
                  else if (dimension == 2) {
                    normal_dot = dpdx*wkset->normals(e,k,0) + dpdy*wkset->normals(e,k,1);
                    sx = dpdx - normal_dot*wkset->normals(e,k,0);
                    sy = dpdy - normal_dot*wkset->normals(e,k,1);
                  }
                  else if (dimension == 3) {
                    normal_dot = dpdx*wkset->normals(e,k,0) + dpdy*wkset->normals(e,k,1) + dpdz*wkset->normals(e,k,2);
                    sx = dpdx - normal_dot*wkset->normals(e,k,0);
                    sy = dpdy - normal_dot*wkset->normals(e,k,1);
                    sz = dpdz - normal_dot*wkset->normals(e,k,2);
                  }
                  // H1
                  if (reg_type == 1) {
                    reg += 0.5*reg_constant*(sx*sx + sy*sy + sz*sz)*side_weights(e,k);
                  }
                  // TV
                  else if (reg_type == 2) {
                    reg += reg_constant*sqrt(sx*sx + sy*sy + sz*sz + offset*offset)*side_weights(e,k);
                  }
                }
              }
            }
          }
        }
      }
      //}
    }
  }
  
  //cout << "reg = " << reg << endl;
  
  return reg;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the regularization over the domain given the domain discretized parameters
///////////////////////////////////////////////////////////////////////////////////////
AD cell::computeDomainRegularization(const vector<double> reg_constants, const vector<int> reg_types,
                                     const vector<int> reg_indices) {
  
  AD reg;
  
  bool seedParams = true;
  //vector<vector<AD> > param_AD;
  //for (int n=0; n<paramindex.size(); n++) {
  //  param_AD.push_back(vector<AD>(paramindex[n].size()));
  //}
  //this->setLocalADParams(param_AD,seedParams);
  int numip = wkset->numip;
  wkset->update(ip,ijac,orientation);
  wkset->computeParamVolIP(param, seedParams);
  
  AD p, dpdx, dpdy, dpdz; // parameters
  double regoffset = 1.0e-5;
  int numParams = reg_indices.size();
  int paramIndex, reg_type;
  double reg_constant;
  for (int i = 0; i < numParams; i++) {
    reg_constant = reg_constants[i];
    reg_type = reg_types[i];
    paramIndex = reg_indices[i];
    for (int e=0; e<numElem; e++) {
      for (int k = 0; k < numip; k++) {
        p = wkset->local_param(e,paramIndex,k);
        // L2
        if (reg_type == 0) {
          reg += 0.5*reg_constant*p*p*wkset->wts(e,k);
        }
        else {
          dpdx = wkset->local_param_grad(e,paramIndex,k,0);
          if (dimension > 1)
            dpdy = wkset->local_param_grad(e,paramIndex,k,1);
          if (dimension > 2)
            dpdz = wkset->local_param_grad(e,paramIndex,k,2);
          // H1
          if (reg_type == 1) {
            reg += 0.5*reg_constant*(dpdx*dpdx + dpdy*dpdy + dpdz*dpdz)*wkset->wts(e,k);
          }
          // TV
          else if (reg_type == 2) {
            reg += reg_constant*sqrt(dpdx*dpdx + dpdy*dpdy + dpdz*dpdz + regoffset*regoffset)*wkset->wts(e,k);
          }
        }
      }
    }
  }
  
  return reg;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the target at the integration points given the solve times
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> cell::computeTarget(const double & solvetime) {
  return physics_RCP->target(myBlock, wkset->ip, solvetime, wkset);
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the weighting function at the integration points given the solve times
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> cell::computeWeight(const double & solvetime) {
  return physics_RCP->weight(myBlock, wkset->ip, solvetime, wkset);
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the response at the integration points given the solution and solve times
///////////////////////////////////////////////////////////////////////////////////////

void cell::addSensors(const Kokkos::View<double**,HostDevice> sensor_points, const double & sensor_loc_tol,
                      const vector<Kokkos::View<double**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                      const vector<basis_RCP> & basis_pointers,
                      const vector<basis_RCP> & param_basis_pointers) {
  
  
  // If we have sensors, then we set the response type to pointwise
  response_type = "pointwise";
  useSensors = true;
  useFineScale = true;
  if (!multiscale || mortar_objective) {
    useFineScale = false;
  }
  
  if (exodus_sensors) {
    // don't use sensor_points
    // set sensorData and sensorLocations from exodus file
    if (sensorLocations.size() > 0) {
      sensorPoints = DRV("sensorPoints",1,sensorLocations.size(),dimension);
      for (size_t i=0; i<sensorLocations.size(); i++) {
        for (int j=0; j<dimension; j++) {
          sensorPoints(0,i,j) = sensorLocations[i](0,j);
        }
        sensorElem.push_back(0);
      }
      DRV refsenspts_buffer("refsenspts_buffer",1,sensorLocations.size(),dimension);
      CellTools<PHX::Device>::mapToReferenceFrame(refsenspts_buffer, sensorPoints, nodes, *cellTopo);
      DRV refsenspts("refsenspts",sensorLocations.size(),dimension);
      Kokkos::deep_copy(refsenspts,Kokkos::subdynrankview(refsenspts_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
      
      vector<DRV> csensorBasis;
      vector<DRV> csensorBasisGrad;
      
      for (size_t b=0; b<basis_pointers.size(); b++) {
        csensorBasis.push_back(DiscTools::evaluateBasis(basis_pointers[b], refsenspts));
        csensorBasisGrad.push_back(DiscTools::evaluateBasisGrads(basis_pointers[b], nodes, refsenspts, cellTopo));
      }
      
      sensorBasis.push_back(csensorBasis);
      sensorBasisGrad.push_back(csensorBasisGrad);
      
      
      vector<DRV> cpsensorBasis;
      vector<DRV> cpsensorBasisGrad;
      
      for (size_t b=0; b<param_basis_pointers.size(); b++) {
        cpsensorBasis.push_back(DiscTools::evaluateBasis(param_basis_pointers[b], refsenspts));
        cpsensorBasisGrad.push_back(DiscTools::evaluateBasisGrads(param_basis_pointers[b], nodes, refsenspts, cellTopo));
      }
      
      param_sensorBasis.push_back(cpsensorBasis);
      param_sensorBasisGrad.push_back(cpsensorBasisGrad);
    }
    
  }
  else {
    if (useFineScale) {
      
      for (size_t i=0; i<subgridModels.size(); i++) {
        //if (subgrid_model_index[0] == i) {
        subgridModels[i]->addSensors(sensor_points,sensor_loc_tol,sensor_data,have_sensor_data,
                                     basis_pointers, subgrid_usernum[0]);
        //}
      }
      
    }
    else {
      DRV phys_points("phys_points",1,sensor_points.dimension(0),dimension);
      for (size_t i=0; i<sensor_points.dimension(0); i++) {
        for (int j=0; j<dimension; j++) {
          phys_points(0,i,j) = sensor_points(i,j);
        }
      }
      
      if (!loadSensorFiles) {
        for (int e=0; e<numElem; e++) {
          
          DRV refpts("refpts", 1, sensor_points.dimension(0), sensor_points.dimension(1));
          DRVint inRefCell("inRefCell", 1, sensor_points.dimension(0));
          DRV cnodes("current nodes",1,nodes.dimension(1), nodes.dimension(2));
          for (int i=0; i<nodes.dimension(1); i++) {
            for (int j=0; j<nodes.dimension(2); j++) {
              cnodes(0,i,j) = nodes(e,i,j);
            }
          }
          CellTools<AssemblyDevice>::mapToReferenceFrame(refpts, phys_points, cnodes, *cellTopo);
          CellTools<AssemblyDevice>::checkPointwiseInclusion(inRefCell, refpts, *cellTopo, sensor_loc_tol);
          
          for (size_t i=0; i<sensor_points.dimension(0); i++) {
            if (inRefCell(0,i) == 1) {
              
              Kokkos::View<double**,HostDevice> newsenspt("new sensor point",1,dimension);
              for (int j=0; j<dimension; j++) {
                newsenspt(0,j) = sensor_points(i,j);
              }
              sensorLocations.push_back(newsenspt);
              mySensorIDs.push_back(i);
              sensorElem.push_back(e);
              if (have_sensor_data) {
                sensorData.push_back(sensor_data[i]);
              }
              if (writeSensorFiles) {
                stringstream ss;
                ss << globalElemID(e);
                string str = ss.str();
                string fname = "sdat." + str + ".dat";
                ofstream outfile(fname.c_str());
                outfile.precision(8);
                outfile << i << "  ";
                outfile << sensor_points(i,0) << "  " << sensor_points(i,1) << "  ";
                //outfile << sensor_data[i](0,0) << "  " << sensor_data[i](0,1) << "  " << sensor_data[i](0,2) << "  " ;
                outfile << endl;
                outfile.close();
              }
            }
          }
        }
      }
      
      if (loadSensorFiles) {
        for (int e=0; e<numElem; e++) {
          stringstream ss;
          ss << globalElemID(e);
          string str = ss.str();
          ifstream sfile;
          sfile.open("sensorLocations/sdat." + str + ".dat");
          int cID;
          //double l1, l2, t1, d1, d2;
          double l1, l2;
          sfile >> cID;
          sfile >> l1;
          sfile >> l2;
          
          sfile.close();
          
          Kokkos::View<double**,HostDevice> newsenspt("sensor point",1,dimension);
          //FC newsensdat(1,3);
          newsenspt(0,0) = l1;
          newsenspt(0,1) = l2;
          sensorLocations.push_back(newsenspt);
          mySensorIDs.push_back(cID);
          sensorElem.push_back(e);
        }
      }
      
      numSensors = sensorLocations.size();
      
      // Evaluate the basis functions and derivatives at sensor points
      if (numSensors > 0) {
        sensorPoints = DRV("sensorPoints",numElem,numSensors,dimension);
        
        for (size_t i=0; i<numSensors; i++) {
          
          DRV csensorPoints("sensorPoints",1,1,dimension);
          DRV cnodes("current nodes",1,nodes.dimension(1), nodes.dimension(2));
          for (int j=0; j<dimension; j++) {
            csensorPoints(0,0,j) = sensorLocations[i](0,j);
            sensorPoints(0,i,j) = sensorLocations[i](0,j);
            for (int k=0; k<nodes.dimension(1); k++) {
              cnodes(0,k,j) = nodes(sensorElem[i],k,j);
            }
          }
          
          
          DRV refsenspts_buffer("refsenspts_buffer",1,1,dimension);
          DRV refsenspts("refsenspts",1,dimension);
          
          CellTools<AssemblyDevice>::mapToReferenceFrame(refsenspts_buffer, csensorPoints, cnodes, *cellTopo);
          //CellTools<AssemblyDevice>::mapToReferenceFrame(refsenspts, csensorPoints, cnodes, *cellTopo);
          Kokkos::deep_copy(refsenspts,Kokkos::subdynrankview(refsenspts_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
          
          vector<DRV> csensorBasis;
          vector<DRV> csensorBasisGrad;
          
          for (size_t b=0; b<basis_pointers.size(); b++) {
            csensorBasis.push_back(DiscTools::evaluateBasis(basis_pointers[b], refsenspts));
            csensorBasisGrad.push_back(DiscTools::evaluateBasisGrads(basis_pointers[b], cnodes, refsenspts, cellTopo));
          }
          sensorBasis.push_back(csensorBasis);
          sensorBasisGrad.push_back(csensorBasisGrad);
          
          
          vector<DRV> cpsensorBasis;
          vector<DRV> cpsensorBasisGrad;
          
          for (size_t b=0; b<param_basis_pointers.size(); b++) {
            cpsensorBasis.push_back(DiscTools::evaluateBasis(param_basis_pointers[b], refsenspts));
            cpsensorBasisGrad.push_back(DiscTools::evaluateBasisGrads(param_basis_pointers[b], nodes, refsenspts, cellTopo));
          }
          
          param_sensorBasis.push_back(cpsensorBasis);
          param_sensorBasisGrad.push_back(cpsensorBasisGrad);
        }
        
      }
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Subgrid Plotting
///////////////////////////////////////////////////////////////////////////////////////

void cell::writeSubgridSolution(const std::string & filename) {
  //if (multiscale) {
  //  subgridModel->writeSolution(filename, subgrid_usernum);
  //}
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute flux and sensitivity wrt params
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeFlux(//vector<vector<AD> > & u_AD, vector<vector<AD> > & u_dot_AD,
                       //vector<vector<AD> > & param_AD, vector<vector<AD> > & lambda_AD,
                       const vector_RCP & gl_u, const vector_RCP & gl_du,
                       const vector_RCP & params,
                       Kokkos::View<double***,AssemblyDevice> lambda,
                       const double & time, const int & side, const double & coarse_h,
                       const bool & compute_sens) {
  
  wkset->time = time;
  wkset->time_KV(0) = time;
  
  Kokkos::View<AD***,AssemblyDevice> u_AD("temp u AD",u.dimension(0),u.dimension(1),u.dimension(2));
  Kokkos::View<AD***,AssemblyDevice> u_dot_AD("temp u AD",u.dimension(0),u.dimension(1),u.dimension(2));
  Kokkos::View<AD***,AssemblyDevice> param_AD("temp u AD",param.dimension(0),param.dimension(1),param.dimension(2));
  
  {
    Teuchos::TimeMonitor localtimer(*cellFluxGatherTimer);
    
    if (compute_sens) {
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index[e].size(); n++) {
          for( size_t i=0; i<index[e][n].size(); i++ ) {
            u_AD(e,n,i) = AD((*gl_u)[0][index[e][n][i]]);
          }
        }
      }
    }
    else {
      size_t numDerivs = gl_du->NumVectors();
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index[e].size(); n++) {
          for( size_t i=0; i<index[e][n].size(); i++ ) {
            u_AD(e,n,i) = AD(maxDerivs, 0, (*gl_u)[0][index[e][n][i]]);
            for( size_t p=0; p<numDerivs; p++ ) {
              u_AD(e,n,i).fastAccessDx(p) = (*gl_du)[p][index[e][n][i]];
            }
          }
        }
      }
    }
    for (int e=0; e<paramindex.size(); e++) {
      for (size_t n=0; n<paramindex[e].size(); n++) {
        for( size_t i=0; i<paramindex[e][n].size(); i++ ) {
          param_AD(e,n,i) = AD((*params)[0][paramindex[e][n][i]]);
        }
      }
    }
  }
  
  {
    Teuchos::TimeMonitor localtimer(*cellFluxWksetTimer);
    
    wkset->computeSolnSideIP(side, u_AD, u_dot_AD, param_AD);
  }
  if (wkset->numAux > 0) {
    
    Teuchos::TimeMonitor localtimer(*cellFluxAuxTimer);
    
    wkset->resetAuxSide();
    
    size_t numip = wkset->numsideip;
    AD auxval;
    
    for (int e=0; e<numElem; e++) {
      for (size_t k=0; k<auxindex.size(); k++) {
        for(size_t i=0; i<auxindex[k].size(); i++ ) {
          auxval = AD(maxDerivs, auxoffsets[k][i], lambda(0,k,i));
          for( size_t j=0; j<numip; j++ ) {
            wkset->local_aux_side(e,k,j) += auxval*auxside_basis[side][auxusebasis[k]](e,i,j);
          }
        }
      }
    }
  }
  
  wkset->resetFlux();
  {
    Teuchos::TimeMonitor localtimer(*cellFluxEvalTimer);
    
    physics_RCP->computeFlux(myBlock);
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the subgrid model
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateSubgridModel(vector<Teuchos::RCP<SubGridModel> > & models) {
  
  /*
   wkset->update(ip,ijac);
   int newmodel = udfunc->getSubgridModel(nodes, wkset, models.size());
   if (newmodel != subgrid_model_index) {
   // then we need:
   // 1. To add the macro-element to the new model
   // 2. Project the most recent solutions onto the new model grid
   // 3. Update this cell to use the new model
   
   // Step 1:
   int newusernum = models[newmodel]->addMacro(nodes, sideinfo, sidenames,
   GIDs, index);
   
   // Step 2:
   
   // Step 3:
   subgridModel = models[newmodel];
   subgrid_model_index = newmodel;
   subgrid_usernum = newusernum;
   
   
   }*/
}

///////////////////////////////////////////////////////////////////////////////////////
// Pass the cell data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateData() {
  
  // hard coded for what I need it for right now
  if (have_cell_phi) {
    wkset->have_rotation_phi = true;
    wkset->rotation_phi = cell_data;
  }
  else if (have_cell_rotation) {
    wkset->have_rotation = true;
    Kokkos::View<double***,AssemblyDevice> rotmat("rotation matrix",numElem,3,3);
    for (int e=0; e<numElem; e++) {
      rotmat(e,0,0) = cell_data(e,0);
      rotmat(e,0,1) = cell_data(e,1);
      rotmat(e,0,2) = cell_data(e,2);
      rotmat(e,1,0) = cell_data(e,3);
      rotmat(e,1,1) = cell_data(e,4);
      rotmat(e,1,2) = cell_data(e,5);
      rotmat(e,2,0) = cell_data(e,6);
      rotmat(e,2,1) = cell_data(e,7);
      rotmat(e,2,2) = cell_data(e,8);
    }
    wkset->rotation = rotmat;
  }
}
