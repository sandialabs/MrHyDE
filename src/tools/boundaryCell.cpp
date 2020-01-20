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

void BoundaryCell::addAuxDiscretization(const vector<basis_RCP> & abasis_pointers,
                                        const vector<DRV> & asideBasis,
                                        const vector<DRV> & asideBasisGrad) {
  
  for (size_t b=0; b<abasis_pointers.size(); b++) {
    auxbasisPointers.push_back(abasis_pointers[b]);
  }
  //for (size_t b=0; b<abasis.size(); b++) {
  //  auxbasis.push_back(abasis[b]);
    //auxbasisGrad.push_back(abasisGrad[b]);
  //}
  
  for (size_t s=0; s<asideBasis.size(); s++) {
    auxside_basis.push_back(asideBasis[s]);
    //auxside_basisGrad.push_back(asideBasisGrad[s]);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux variables
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::addAuxVars(const vector<string> & auxlist_) {
  auxlist = auxlist_;
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each variable will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setUseBasis(vector<int> & usebasis_, const int & nstages_) {
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
  u = Kokkos::View<ScalarT***,AssemblyDevice>("u",numElem,numDOF.size(),maxnbasis);
  u_dot = Kokkos::View<ScalarT***,AssemblyDevice>("u_dot",numElem,numDOF.size(),maxnbasis);
  phi = Kokkos::View<ScalarT***,AssemblyDevice>("phi",numElem,numDOF.size(),maxnbasis);
  phi_dot = Kokkos::View<ScalarT***,AssemblyDevice>("phi_dot",numElem,numDOF.size(),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
  Kokkos::View<int*,HostDevice> numParamDOF_host("numParamDOF on host",paramusebasis.size());
  for (unsigned int i=0; i<paramusebasis.size(); i++) {
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
  param = Kokkos::View<ScalarT***,AssemblyDevice>("param",numElem,numParamDOF.size(),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setAuxUseBasis(vector<int> & ausebasis_) {
  auxusebasis = ausebasis_;
  size_t maxnbasis = 0;
  for (size_t i=0; i<numAuxDOF.size(); i++) {
    if (numAuxDOF(i) > maxnbasis) {
      maxnbasis = numAuxDOF(i);
    }
  }
  aux = Kokkos::View<ScalarT***,AssemblyDevice>("aux",numElem,numAuxDOF.size(),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the coarse grid solution to the fine grid integration points
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::computeSoln(const bool & seedu, const bool & seedudot, const bool & seedparams,
                               const bool & seedaux) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnSideTimer);
  
  wkset->computeSolnSideIP(sidenum, u, u_dot, seedu, seedudot);
  wkset->computeParamSideIP(sidenum, param, seedparams);
  
  if (wkset->numAux > 0) {
    
    wkset->resetAuxSide();
    
    size_t numip = wkset->numsideip;
    AD auxval;
    
    for (int e=0; e<numElem; e++) {
      for (size_t k=0; k<auxindex.dimension(1); k++) {
        for(size_t i=0; i<numAuxDOF(k); i++ ) {
          if (seedaux) {
            auxval = AD(maxDerivs,auxoffsets[k][i],aux(e,k,i));
          }
          else {
            auxval = aux(e,k,i);
          }
          for( size_t j=0; j<numip; j++ ) {
            wkset->local_aux_side(e,k,j) += auxval*auxside_basis[auxusebasis[k]](e,i,j);
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
// Compute the contribution from this cell to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                         const bool & compute_jacobian, const bool & compute_sens,
                         const int & num_active_params, const bool & compute_disc_sens,
                         const bool & compute_aux_sens, const bool & store_adjPrev,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_res,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot) {
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  // Boundary contribution
  {
    Teuchos::TimeMonitor localtimer(*boundaryResidualTimer);
    
    wkset->updateSide(sidenum, wksetBID);
    wkset->sidename = sidename;
    wkset->currentside = sidenum;
    
    
    //    wkset->sideinfo = sideinfo;
    //    wkset->currentside = side;
    //    wkset->sidetype = sidetype;
    // if (sideinfo[e](side,1) == -1) {
    //   wkset->sidename = "interior";
    //   wkset->sidetype = -1;
    // }
    // else {
    //    wkset->sidename = gsideid;
    //wkset->sidetype = sideinfo[e](side,0);
    // }
    
    if (compute_jacobian) {
      if (compute_disc_sens) {
        this->computeSoln(false,false,true,false);
      }
      else if (compute_aux_sens) {
        this->computeSoln(false,false,false,true);
      }
      else {
        this->computeSoln(true,false,false,false);
      }
    }
    else {
      this->computeSoln(false,false,false,false);
    }
    
    //wkset->resetResidual(numElem);
    wkset->resetResidual();
    
    cellData->physics_RCP->boundaryResidual(cellData->myBlock);
    
  }
  
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
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  if (compute_sens) {
    parallel_for(RangePolicy<AssemblyDevice>(0,local_res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int r=0; r<local_res.dimension(2); r++) {
        for (unsigned int n=0; n<index.dimension(1); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyDevice>(0,local_res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.dimension(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(e,offsets(n,j),0) -= res_AD(e,offsets(n,j)).val();
        }
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateAdjointRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->adjrhs;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  if (compute_sens) {
    parallel_for(RangePolicy<AssemblyDevice>(0,local_res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int r=0; r<maxDerivs; r++) {
        for (unsigned int n=0; n<index.dimension(1); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyDevice>(0,local_res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.dimension(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(e,offsets(n,j),0) -= res_AD(e,offsets(n,j)).val();
        }
      }
    });
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  
  if (useadjoint) {
    parallel_for(RangePolicy<AssemblyDevice>(0,local_J.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.dimension(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<index.dimension(1); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(e,offsets(m,k),offsets(n,j)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyDevice>(0,local_J.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.dimension(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<index.dimension(1); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(e,offsets(n,j),offsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jdot
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateJacDot(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  
  if (useadjoint) {
    parallel_for(RangePolicy<AssemblyDevice>(0,local_Jdot.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.dimension(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<index.dimension(1); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_Jdot(e,offsets(m,k),offsets(n,j)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyDevice>(0,local_Jdot.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.dimension(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<index.dimension(1); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_Jdot(e,offsets(n,j),offsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  bool lumpmass = false;
  /* // TMW: Commented this out since have it hard-coded to false
   if (lumpmass) {
   FC Jdotold = local_Jdot;
   local_Jdot.initialize(0.0);
   //his->resetJacDot();
   for (int e=0; e<numElem; e++) {
   for (unsigned int n=0; n<GIDs[e].size(); n++) {
   for (unsigned int m=0; m<GIDs[e].size(); m++) {
   local_Jdot(e,n,n) += Jdotold(e,n,m);
   }
   }
   }
   }*/
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateParamJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<int**,AssemblyDevice> paramoffsets = wkset->paramoffsets;
  
  parallel_for(RangePolicy<AssemblyDevice>(0,local_J.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int n=0; n<index.dimension(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<paramindex.dimension(1); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_J(e,offsets(n,j),paramoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparamdot
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateParamJacDot(Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<int**,AssemblyDevice> paramoffsets = wkset->paramoffsets;
  
  parallel_for(RangePolicy<AssemblyDevice>(0,local_Jdot.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int n=0; n<index.dimension(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<paramindex.dimension(1); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_Jdot(e,offsets(n,j),paramoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateAuxJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  
  parallel_for(RangePolicy<AssemblyDevice>(0,local_J.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int n=0; n<index.dimension(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<auxindex.dimension(1); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(e,offsets(n,j),auxoffsets[m][k]) += res_AD(e,offsets(n,j)).fastAccessDx(auxoffsets[m][k]);
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparamdot
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateAuxJacDot(Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  
  parallel_for(RangePolicy<AssemblyDevice>(0,local_Jdot.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int n=0; n<index.dimension(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<auxindex.dimension(1); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_Jdot(e,offsets(n,j),auxoffsets[m][k]) += res_AD(e,offsets(n,j)).fastAccessDx(auxoffsets[m][k]);
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute boundary regularization given the boundary discretized parameters
///////////////////////////////////////////////////////////////////////////////////////

AD BoundaryCell::computeBoundaryRegularization(const vector<ScalarT> reg_constants,
                                               const vector<int> reg_types,
                                               const vector<int> reg_indices,
                                               const vector<string> reg_sides) {
  
  AD reg;
  
  bool seedParams = true;
  //vector<vector<AD> > param_AD;
  //for (int n=0; n<paramindex.size(); n++) {
  //  param_AD.push_back(vector<AD>(paramindex[n].size()));
  //}
  //this->setLocalADParams(param_AD,seedParams);
  //int numip = wkset->numip;
  int numParams = reg_indices.size();
  /*
  for (int side=0; side<cellData->numSides; side++) {
    for (int e=0; e<numElem; e++) {
      if (sideinfo(e,0,side,0) > 0) { // Just checking the first variable should be sufficient
        onside = true;
        sname = sidenames[sideinfo(e,0,side,1)];
      }
    }
    
    if (onside) {*/
  
      //int sidetype = sideinfo[e](side,0); // 0-not on bndry, 1-Dirichlet bndry, 2-Neumann bndry
      //if (sidetype > 0) {
      //wkset->updateSide(nodes, sideip[side], sideijac[side], side);
      
  //    wkset->updateSide(nodes, sideip[side], sidewts[side],normals[side],sideijac[side], side);
      
      int numip = wkset->numsideip;
      //int gside = sideinfo[e](side,1); // =-1 if is an interior edge
      
      DRV side_weights = wkset->wts_side;
      int paramIndex, reg_type;
      ScalarT reg_constant;
      string reg_side;
      size_t found;
      
      for (int i = 0; i < numParams; i++) {
        paramIndex = reg_indices[i];
        reg_constant = reg_constants[i];
        reg_type = reg_types[i];
        reg_side = reg_sides[i];
        found = reg_side.find(sidename);
        if (found != string::npos) {
          
          wkset->updateSide(sidenum, cellID);
          wkset->computeParamSideIP(sidenum, param, seedParams);
          
          AD p, dpdx, dpdy, dpdz; // parameters
          ScalarT offset = 1.0e-5;
          for (int e=0; e<numElem; e++) {
            //if (sideinfo(e,0,side,0) > 0) {
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
                  if (cellData->dimension > 1) {
                    dpdy = wkset->local_param_grad_side(e,paramIndex,k,1);
                  }
                  if (cellData->dimension > 2) {
                    dpdz = wkset->local_param_grad_side(e,paramIndex,k,2);
                  }
                  if (cellData->dimension == 1) {
                    normal_dot = dpdx*wkset->normals(e,k,0);
                    sx = dpdx - normal_dot*wkset->normals(e,k,0);
                  }
                  else if (cellData->dimension == 2) {
                    normal_dot = dpdx*wkset->normals(e,k,0) + dpdy*wkset->normals(e,k,1);
                    sx = dpdx - normal_dot*wkset->normals(e,k,0);
                    sy = dpdy - normal_dot*wkset->normals(e,k,1);
                  }
                  else if (cellData->dimension == 3) {
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
            //}
          }
        }
      }
      //}
    //}
  //}
  
  //cout << "reg = " << reg << endl;
  
  return reg;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute flux and sensitivity wrt params
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::computeFlux(const vector_RCP & gl_u,
                               const vector_RCP & gl_du,
                               const vector_RCP & params,
                               Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                               const ScalarT & time, const int & side, const ScalarT & coarse_h,
                               const bool & compute_sens) {
  
  wkset->time = time;
  wkset->time_KV(0) = time;
  
  auto u_kv = gl_u->getLocalView<HostDevice>();
  auto du_kv = gl_du->getLocalView<HostDevice>();
  //auto params_kv = params->getLocalView<HostDevice>();
  
  Kokkos::View<AD***,AssemblyDevice> u_AD("temp u AD",u.dimension(0),u.dimension(1),u.dimension(2));
  Kokkos::View<AD***,AssemblyDevice> u_dot_AD("temp u AD",u.dimension(0),u.dimension(1),u.dimension(2));
  //Kokkos::View<AD***,AssemblyDevice> param_AD("temp u AD",param.dimension(0),param.dimension(1),param.dimension(2));
  Kokkos::View<AD***,AssemblyDevice> param_AD("temp u AD",1,1,1);
  
  {
    Teuchos::TimeMonitor localtimer(*cellFluxGatherTimer);
    
    if (compute_sens) {
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index.dimension(1); n++) {
          for( size_t i=0; i<numDOF(n); i++ ) {
            u_AD(e,n,i) = AD(u_kv(index(e,n,i),0));
          }
        }
      }
    }
    else {
      size_t numDerivs = gl_du->getNumVectors();
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index.dimension(1); n++) {
          for( size_t i=0; i<numDOF(n); i++ ) {
            u_AD(e,n,i) = AD(maxDerivs, 0, u_kv(index(e,n,i),0));
            for( size_t p=0; p<numDerivs; p++ ) {
              u_AD(e,n,i).fastAccessDx(p) = du_kv(index(e,n,i),p);
            }
          }
        }
      }
    }
    /*
     for (int e=0; e<paramindex.size(); e++) {
     for (size_t n=0; n<paramindex.dimension(1); n++) {
     for( size_t i=0; i<numParamDOF(n); i++ ) {
     param_AD(e,n,i) = AD(params_kv(paramindex(e,n,i),0));
     }
     }
     }*/
  }
  
  {
    Teuchos::TimeMonitor localtimer(*cellFluxWksetTimer);
    
    wkset->computeSolnSideIP(sidenum, u_AD, u_dot_AD, param_AD);
  }
  if (wkset->numAux > 0) {
    
    Teuchos::TimeMonitor localtimer(*cellFluxAuxTimer);
    
    wkset->resetAuxSide();
    
    size_t numip = wkset->numsideip;
    AD auxval;
    
    for (int e=0; e<numElem; e++) {
      for (size_t k=0; k<auxindex.dimension(1); k++) {
        for(size_t i=0; i<numAuxDOF(k); i++ ) {
          auxval = AD(maxDerivs, auxoffsets[k][i], lambda(0,k,i));
          for( size_t j=0; j<numip; j++ ) {
            wkset->local_aux_side(e,k,j) += auxval*auxside_basis[auxusebasis[k]](e,i,j);
          }
        }
      }
    }
  }
  
  wkset->resetFlux();
  {
    Teuchos::TimeMonitor localtimer(*cellFluxEvalTimer);
    
    cellData->physics_RCP->computeFlux(cellData->myBlock);
  }
  
}
