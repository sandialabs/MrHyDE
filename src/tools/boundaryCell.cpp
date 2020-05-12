/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "boundaryCell.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

BoundaryCell::BoundaryCell(const Teuchos::RCP<CellMetaData> & cellData_,
                           const DRV & nodes_,
                           const Kokkos::View<LO*,AssemblyDevice> & localID_,
                           const Kokkos::View<LO*,AssemblyDevice> & sideID_,
                           const int & sidenum_, const string & sidename_,
                           const int & cellID_,
                           Kokkos::View<GO**,HostDevice> GIDs_,
                           Kokkos::View<int****,HostDevice> sideinfo_,
                           Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orientation_) :
cellData(cellData_), localElemID(localID_), localSideID(sideID_), nodes(nodes_),
sidenum(sidenum_), sidename(sidename_), cellID(cellID_), GIDs(GIDs_), sideinfo(sideinfo_), orientation(orientation_) {
  
  numElem = nodes.extent(0);
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setIndex(Kokkos::View<LO***,AssemblyDevice> & index_) {
  
  index = Kokkos::View<LO***,AssemblyDevice>("local index",index_.extent(0),
                                             index_.extent(1), index_.extent(2));
  
  // Need to copy the data since index_ is rewritten for each cell
  
  parallel_for(RangePolicy<AssemblyExec>(0,index_.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int j=0; j<index_.extent(1); j++) {
      for (unsigned int k=0; k<index_.extent(2); k++) {
        index(e,j,k) = index_(e,j,k);
      }
    }
  });
  
  
  // This is common to all cells (within the same block), so a view copy will do
  //numDOF = numDOF_;
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setParamIndex(Kokkos::View<LO***,AssemblyDevice> & pindex_) {
  
  paramindex = Kokkos::View<LO***,AssemblyDevice>("local param index",pindex_.extent(0),
                                                  pindex_.extent(1), pindex_.extent(2));
  
  // Need to copy the data since index_ is rewritten for each cell
  
  parallel_for(RangePolicy<AssemblyExec>(0,pindex_.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int j=0; j<pindex_.extent(1); j++) {
      for (unsigned int k=0; k<pindex_.extent(2); k++) {
        paramindex(e,j,k) = pindex_(e,j,k);
      }
    }
  });
  
  // This is common to all cells, so a view copy will do
  //numParamDOF = pnumDOF_;
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setAuxIndex(Kokkos::View<LO***,AssemblyDevice> & aindex_) {
  
  auxindex = Kokkos::View<LO***,AssemblyDevice>("local aux index",
                                                aindex_.extent(0),
                                                aindex_.extent(1),
                                                aindex_.extent(2));
  
  // Need to copy the data since index_ is rewritten for each cell
  //Kokkos::deep_copy(auxindex,aindex_);
  
  parallel_for(RangePolicy<AssemblyExec>(0,aindex_.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int j=0; j<aindex_.extent(1); j++) {
      for (unsigned int k=0; k<aindex_.extent(2); k++) {
        auxindex(e,j,k) = aindex_(e,j,k);
      }
    }
  });
  
  // This is common to all cells, so a view copy will do
  // This is excessive storage, please remove
  //numAuxDOF = anumDOF_;
  
  // Temp. fix
  Kokkos::View<int*,UnifiedDevice> numAuxDOF("numAuxDOF",auxindex.extent(1));
  for (unsigned int i=0; i<auxindex.extent(1); i++) {
    numAuxDOF(i) = auxindex.extent(2);
  }
  cellData->numAuxDOF = numAuxDOF;
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux basis functions at the integration points.
// This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::addAuxDiscretization(const vector<basis_RCP> & abasis_pointers,
                                        const vector<DRV> & asideBasis,
                                        const vector<DRV> & asideBasisGrad) {
  
  auxbasisPointers = abasis_pointers;
  auxside_basis = asideBasis;
  
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
  for (size_t i=0; i<cellData->numDOF.extent(0); i++) {
    if (cellData->numDOF(i) > maxnbasis) {
      maxnbasis = cellData->numDOF(i);
    }
  }
  //maxnbasis *= nstages;
  u = Kokkos::View<ScalarT***,AssemblyDevice>("u",numElem,cellData->numDOF.extent(0),maxnbasis);
  //u_dot = Kokkos::View<ScalarT***,AssemblyDevice>("u_dot",numElem,cellData->numDOF.extent(0),maxnbasis);
  phi = Kokkos::View<ScalarT***,AssemblyDevice>("phi",numElem,cellData->numDOF.extent(0),maxnbasis);
  //phi_dot = Kokkos::View<ScalarT***,AssemblyDevice>("phi_dot",numElem,cellData->numDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
  /*
  Kokkos::View<int*,HostDevice> numParamDOF_host("numParamDOF on host",paramusebasis.size());
  for (unsigned int i=0; i<paramusebasis.size(); i++) {
    numParamDOF_host(i) = paramnumbasis_[paramusebasis[i]];
  }
  numParamDOF = Kokkos::create_mirror_view(numParamDOF_host);
  Kokkos::deep_copy(numParamDOF_host, numParamDOF);
  */
  
  size_t maxnbasis = 0;
  for (size_t i=0; i<cellData->numParamDOF.extent(0); i++) {
    if (cellData->numParamDOF(i) > maxnbasis) {
      maxnbasis = cellData->numParamDOF(i);
    }
  }
  param = Kokkos::View<ScalarT***,AssemblyDevice>("param",numElem,cellData->numParamDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setAuxUseBasis(vector<int> & ausebasis_) {
  auxusebasis = ausebasis_;
  size_t maxnbasis = 0;
  for (size_t i=0; i<cellData->numAuxDOF.extent(0); i++) {
    if (cellData->numAuxDOF(i) > maxnbasis) {
      maxnbasis = cellData->numAuxDOF(i);
    }
  }
  aux = Kokkos::View<ScalarT***,AssemblyDevice>("aux",numElem,cellData->numAuxDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the coarse grid solution to the fine grid integration points
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::computeSoln(Kokkos::View<int*,UnifiedDevice> seedwhat) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnSideTimer);
  
  wkset->computeSolnSideIP(u, seedwhat);
  wkset->computeParamSideIP(sidenum, param, seedwhat);
  
  if (wkset->numAux > 0) {
    
    wkset->resetAuxSide();
    
    size_t numip = wkset->numsideip;
    AD auxval;
    
    for (int e=0; e<numElem; e++) {
      
      for (size_t k=0; k<auxindex.extent(1); k++) {
        for(size_t i=0; i<cellData->numAuxDOF(k); i++ ) {
          ScalarT auxtmp = aux(localElemID[e],k,i);
          if (seedwhat(0) == 4) {
            auxval = AD(maxDerivs,auxoffsets(k,i),auxtmp);
            //auxval = AD(maxDerivs,auxoffsets[k][i],aux(e,k,i));
          }
          else {
            auxval = auxtmp;
            //auxval = aux(e,k,i);
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
                                 Kokkos::View<ScalarT***,UnifiedDevice> local_res,
                                 Kokkos::View<ScalarT***,UnifiedDevice> local_J) {
  
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
    Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
    if (compute_jacobian) {
      if (compute_disc_sens) {
        seedwhat(0) = 3;
        this->computeSoln(seedwhat);
      }
      else if (compute_aux_sens) {
        seedwhat(0) = 4;
        this->computeSoln(seedwhat);
      }
      else {
        seedwhat(0) = 1;
        this->computeSoln(seedwhat);
      }
    }
    else {
      seedwhat(0) = 0;
      this->computeSoln(seedwhat);
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

void BoundaryCell::updateRes(const bool & compute_sens, Kokkos::View<ScalarT***,UnifiedDevice> local_res) {
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  
  if (compute_sens) {
    parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int r=0; r<local_res.extent(2); r++) {
        for (unsigned int n=0; n<index.extent(1); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.extent(1); n++) {
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

void BoundaryCell::updateAdjointRes(const bool & compute_sens, Kokkos::View<ScalarT***,UnifiedDevice> local_res) {
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->adjrhs;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  
  if (compute_sens) {
    parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int r=0; r<maxDerivs; r++) {
        for (unsigned int n=0; n<index.extent(1); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.extent(1); n++) {
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

void BoundaryCell::updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,UnifiedDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  
  if (useadjoint) {
    parallel_for(RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.extent(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<index.extent(1); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(e,offsets(m,k),offsets(n,j)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<index.extent(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<index.extent(1); m++) {
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
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateParamJac(Kokkos::View<ScalarT***,UnifiedDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<int**,AssemblyDevice> paramoffsets = wkset->paramoffsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  Kokkos::View<LO*,UnifiedDevice> numParamDOF = cellData->numParamDOF;
  
  parallel_for(RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int n=0; n<index.extent(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<paramindex.extent(1); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_J(e,offsets(n,j),paramoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateAuxJac(Kokkos::View<ScalarT***,UnifiedDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  Kokkos::View<LO*,UnifiedDevice> numAuxDOF = cellData->numAuxDOF;
  
  parallel_for(RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int n=0; n<index.extent(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<auxindex.extent(1); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(e,offsets(n,j),auxoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(auxoffsets(m,k));
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
          Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
          seedwhat(0) = 3;
          wkset->computeParamSideIP(sidenum, param, seedwhat);
          
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
  
  //KokkosTools::print(lambda);
  
  Kokkos::View<AD***,AssemblyDevice> u_AD("temp u AD",u.extent(0),u.extent(1),u.extent(2));
  //Kokkos::View<AD***,AssemblyDevice> u_dot_AD("temp u AD",u.extent(0),u.extent(1),u.extent(2));
  //Kokkos::View<AD***,AssemblyDevice> param_AD("temp u AD",param.extent(0),param.extent(1),param.extent(2));
  Kokkos::View<AD***,AssemblyDevice> param_AD("temp u AD",1,1,1);
  
  {
    Teuchos::TimeMonitor localtimer(*cellFluxGatherTimer);
    
    if (compute_sens) {
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index.extent(1); n++) {
          for( size_t i=0; i<cellData->numDOF(n); i++ ) {
            u_AD(e,n,i) = AD(u_kv(index(e,n,i),0));
          }
        }
      }
    }
    else {
      size_t numDerivs = gl_du->getNumVectors();
      for (int e=0; e<numElem; e++) {
        for (size_t n=0; n<index.extent(1); n++) {
          for( size_t i=0; i<cellData->numDOF(n); i++ ) {
            u_AD(e,n,i) = AD(maxDerivs, 0, u_kv(index(e,n,i),0));
            for( size_t p=0; p<numDerivs; p++ ) {
              u_AD(e,n,i).fastAccessDx(p) = du_kv(index(e,n,i),p);
            }
          }
        }
      }
    }
  }
  
  {
    Teuchos::TimeMonitor localtimer(*cellFluxWksetTimer);
    
    wkset->computeSolnSideIP(sidenum, u_AD, param_AD);
  }
  if (wkset->numAux > 0) {
    
    Teuchos::TimeMonitor localtimer(*cellFluxAuxTimer);
    
    wkset->resetAuxSide();
    
    size_t numip = wkset->numsideip;
    AD auxval;
    for (int e=0; e<numElem; e++) {
      for (size_t k=0; k<auxindex.extent(1); k++) {
        for(size_t i=0; i<cellData->numAuxDOF(k); i++ ) {
          auxval = AD(maxDerivs, auxoffsets(k,i), lambda(localElemID[e],k,i));
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
