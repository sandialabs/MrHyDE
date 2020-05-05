/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "cell.hpp"
#include "discretizationTools.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

cell::cell(const Teuchos::RCP<CellMetaData> & cellData_,
           const DRV & nodes_,
           const Kokkos::View<LO*,AssemblyDevice> & localID_,
           Kokkos::View<GO**,HostDevice> GIDs_,
           Kokkos::View<int****,HostDevice> sideinfo_,
           Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orientation_) :
cellData(cellData_), localElemID(localID_), nodes(nodes_), GIDs(GIDs_), sideinfo(sideinfo_), orientation(orientation_)
{
  
  numElem = nodes.extent(0);
  useSensors = false;
  
  this->setIP();
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::setIP() {
  // ip and ref_ip will live on the assembly device
  ip = DRV("ip", numElem, cellData->ref_ip.extent(0), cellData->dimension);
  CellTools::mapToPhysicalFrame(ip, cellData->ref_ip, nodes, *(cellData->cellTopo));
  
  jacobian = DRV("jacobian", numElem, cellData->ref_ip.extent(0), cellData->dimension, cellData->dimension);
  CellTools::setJacobian(jacobian, cellData->ref_ip, nodes, *(cellData->cellTopo));
  
  jacobianDet = DRV("determinant of jacobian", numElem, cellData->ref_ip.extent(0));
  jacobianInv = DRV("inverse of jacobian", numElem, cellData->ref_ip.extent(0), cellData->dimension, cellData->dimension);
  CellTools::setJacobianDet(jacobianDet, jacobian);
  CellTools::setJacobianInv(jacobianInv, jacobian);
  
  wts = DRV("ip wts", numElem, cellData->ref_ip.extent(0));
  FuncTools::computeCellMeasure(wts, jacobianDet, cellData->ref_wts);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::setIndex(Kokkos::View<LO***,AssemblyDevice> & index_) {
  
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

void cell::setParamIndex(Kokkos::View<LO***,AssemblyDevice> & pindex_) {
  
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

void cell::setAuxIndex(Kokkos::View<LO***,AssemblyDevice> & aindex_) {
  
  auxindex = Kokkos::View<LO***,AssemblyDevice>("local aux index",1,aindex_.extent(1),
                                            aindex_.extent(2));
  
  // Need to copy the data since index_ is rewritten for each cell
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
  Kokkos::View<int*,AssemblyDevice> numAuxDOF("numAuxDOF",auxindex.extent(1));
  for (unsigned int i=0; i<auxindex.extent(1); i++) {
    numAuxDOF(i) = auxindex.extent(2);
  }
  cellData->numAuxDOF = numAuxDOF;
}

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
  cellData->physics_RCP->updateParameters(params, paramnames);
}


///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each variable will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setUseBasis(vector<int> & usebasis_, const int & nstages_) {
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
  u_dot = Kokkos::View<ScalarT***,AssemblyDevice>("u_dot",numElem,cellData->numDOF.extent(0),maxnbasis);
  phi = Kokkos::View<ScalarT***,AssemblyDevice>("phi",numElem,cellData->numDOF.extent(0),maxnbasis);
  phi_dot = Kokkos::View<ScalarT***,AssemblyDevice>("phi_dot",numElem,cellData->numDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
  /*
  Kokkos::View<int*,HostDevice> numParamDOF_host("numParamDOF on host",paramusebasis.size());
  for (int i=0; i<paramusebasis.size(); i++) {
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

void cell::setAuxUseBasis(vector<int> & ausebasis_) {
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
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolnVolIP(Kokkos::View<int*,UnifiedDevice> seedwhat) {
  // seedwhat key: 0-nothing; 1-sol; 2-soldot; 3-disc.params.; 4-aux.vars
  
  Teuchos::TimeMonitor localtimer(*computeSolnVolTimer);
  
  wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  wkset->computeSolnVolIP(u, u_dot, seedwhat);
  wkset->computeParamVolIP(param, seedwhat);
  
  /*
  if (wkset->numAux > 0) {
    wkset->resetAux();
    
    AD auxval;
    for (int e=0; e<numElem; e++) {
      if (auxindex.size() > e ) {
        for (size_t k=0; k<auxindex.extent(1); k++) {
          if (auxusebasis[k] < auxbasis.size()) {
            for( int i=0; i<numAuxDOF(k); i++ ) {
              
              if (seedaux) {
                auxval = AD(maxDerivs,auxoffsets[k][i],aux(e,k,i));
              }
              else {
                auxval = aux(e,k,i);
              }
              for( size_t j=0; j<ip.extent(1); j++ ) {
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
  }*/
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolnFaceIP(const size_t & facenum, Kokkos::View<int*,UnifiedDevice> seedwhat) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnFaceTimer);
  
  wkset->updateFace(nodes, orientation, facenum);
  wkset->computeSolnFaceIP(u, u_dot, seedwhat);
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the solution variables in the workset
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateSolnWorkset(const vector_RCP & gl_u, int tindex) {
  Kokkos::View<ScalarT***,AssemblyDevice> ulocal("tempory u", numElem,u.extent(1),u.extent(2));
  auto u_kv = gl_u->getLocalView<HostDevice>();
  
  for (int e=0; e<numElem; e++) {
    for (size_t n=0; n<index.extent(1); n++) {
      for(size_t i=0; i<cellData->numDOF(n); i++ ) {
        ulocal(e,n,i) = u_kv(index(e,n,i),tindex);
      }
    }
  }
  wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  wkset->computeSolnVolIP(ulocal);
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the contribution from this cell to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                         const bool & compute_jacobian, const bool & compute_sens,
                         const int & num_active_params, const bool & compute_disc_sens,
                         const bool & compute_aux_sens, const bool & store_adjPrev,
                         Kokkos::View<ScalarT***,UnifiedDevice> local_res,
                         Kokkos::View<ScalarT***,UnifiedDevice> local_J,
                         Kokkos::View<ScalarT***,UnifiedDevice> local_Jdot) {
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  bool fixJacDiag = false;
  
  wkset->resetResidual();
  
  if (isAdjoint) {
    wkset->resetAdjointRHS();
  }
  
  //////////////////////////////////////////////////////////////
  // Compute the AD-seeded solutions at integration points
  //////////////////////////////////////////////////////////////
  
  // Putting this on UVM memory so anyone can access it
  Kokkos::View<int*,UnifiedDevice> seedwhat("int indicating seeding",1);
  if (compute_jacobian) {
    if (compute_disc_sens) {
      seedwhat(0) = 3;
      this->computeSolnVolIP(seedwhat);
    }
    else if (compute_aux_sens) {
      seedwhat(0) = 4;
      this->computeSolnVolIP(seedwhat);
    }
    else {
      seedwhat(0) = 1;
      this->computeSolnVolIP(seedwhat);
    }
  }
  else {
    seedwhat(0) = 0; // seed nothing
    this->computeSolnVolIP(seedwhat);
  }
  
  //////////////////////////////////////////////////////////////
  // Compute res and J=dF/du
  //////////////////////////////////////////////////////////////
  
  // Volumetric contribution
  
  {
    Teuchos::TimeMonitor localtimer(*volumeResidualTimer);
    if (cellData->multiscale) {
      //for (int e=0; e<numElem; e++) {
        int sgindex = subgrid_model_index[subgrid_model_index.size()-1];
        subgridModels[sgindex]->subgridSolver(u, phi, time, isTransient, isAdjoint,
                                              compute_jacobian, compute_sens, num_active_params,
                                              compute_disc_sens, compute_aux_sens,
                                              *wkset, subgrid_usernum, 0,
                                              subgradient, store_adjPrev);
      //}
      fixJacDiag = true;
    }
    else {
      cellData->physics_RCP->volumeResidual(cellData->myBlock);
    }
  }
  
  // Edge/face contribution
  
  {
    Teuchos::TimeMonitor localtimer(*faceResidualTimer);
    if (cellData->multiscale) {
      // do nothing
    }
    else {
      // determine if we need to include edge/face terms
      bool include_face = cellData->physics_RCP->checkFace(cellData->myBlock);
      
      if (include_face) {
        for (size_t s=0; s<cellData->numSides; s++) {
          if (compute_jacobian) {
            if (compute_disc_sens) {
              seedwhat(0) = 3;
              this->computeSolnFaceIP(s,seedwhat);
            }
            else if (compute_aux_sens) {
              seedwhat(0) = 4;
              this->computeSolnFaceIP(s,seedwhat);
            }
            else {
              seedwhat(0) = 1;
              this->computeSolnFaceIP(s,seedwhat);
            }
          }
          else {
            seedwhat(0) = 0;
            this->computeSolnFaceIP(s,seedwhat);
          }
          cellData->physics_RCP->faceResidual(cellData->myBlock);
        }
      }
    }
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
  
  if (compute_jacobian && fixJacDiag) {
    ScalarT JTOL = 1.0E-8;
    
    for (int e=0; e<numElem; e++) {
      for (size_t n=0; n<wkset->offsets.extent(0); n++) {
        for (size_t i=0; i<wkset->offsets.extent(1); i++) {
          if (abs(local_J(e,wkset->offsets(n,i),wkset->offsets(n,i))) < JTOL) {
            local_res(e,wkset->offsets(n,i),0) = -u(e,n,i);
            
            
            for (size_t j=0; j<wkset->offsets.extent(0); j++) {
              ScalarT scale = 1.0/((ScalarT)wkset->offsets.extent(1)-1.0);
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
    if (isTransient && compute_jacobian && !cellData->multiscale) {
      if (compute_jacobian) {
        if (compute_disc_sens) {
          seedwhat(0) = 3;
          this->computeSolnVolIP(seedwhat);
        }
        else if (compute_aux_sens) {
          seedwhat(0) = 4;
          this->computeSolnVolIP(seedwhat);
        }
        else {
          seedwhat(0) = 2;
          this->computeSolnVolIP(seedwhat);
        }
      }
      else {
        seedwhat(0) = 0;
        this->computeSolnVolIP(seedwhat);
      }
      wkset->resetResidual();//res.initialize(0.0);// = FCAD(GIDs.size());
      
      
      // evaluate the local solutions at the volumetric integration points
      
      cellData->physics_RCP->volumeResidual(cellData->myBlock);
      
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
      if (!(cellData->mortar_objective)) {
        for (int w=1; w < cellData->dimension+2; w++) {
          
          Kokkos::View<AD**,AssemblyDevice> obj = computeObjective(wkset->time, 0, w);
          
          int numDerivs;
          if (useSensors) {
            if (numSensors > 0) {
              //for (int e=0; e<numSensors; e++) {
              for (int s=0; s<numSensors; s++) {
                int e = sensorElem[s];
                for (int n=0; n<index.extent(1); n++) {
                  for (int j=0; j<cellData->numDOF(n); j++) {
                    for (int i=0; i<cellData->numDOF(n); i++) {
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
              for (int n=0; n<index.extent(1); n++) {
                for (int j=0; j<cellData->numDOF(n); j++) {
                  for (int i=0; i<cellData->numDOF(n); i++) {
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
          for (int n=0; n<index.extent(1); n++) {
            for (int j=0; j<cellData->numDOF(n); j++) {
              for (int m=0; m<index.extent(1); m++) {
                for (int k=0; k<cellData->numDOF(m); k++) {
                  local_res(e,wkset->offsets(n,j),0) += -local_J(e,wkset->offsets(n,j),wkset->offsets(m,k))*phi(e,m,k);
                }
              }
            }
          }
        }
        if (isTransient) {
          for (int e=0; e<numElem; e++) {
            for (int n=0; n<index.extent(1); n++) {
              for (int j=0; j<cellData->numDOF(n); j++) {
                ScalarT aPrev = 0.0;
                for (int m=0; m<index.extent(1); m++) {
                  for (int k=0; k<cellData->numDOF(m); k++) {
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
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  
  if (compute_sens) {
    parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int r=0; r<local_res.extent(2); r++) {
        for (int n=0; n<index.extent(1); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<index.extent(1); n++) {
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

void cell::updateAdjointRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->adjrhs;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  
  if (compute_sens) {
    parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int r=0; r<maxDerivs; r++) {
        for (int n=0; n<index.extent(1); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<index.extent(1); n++) {
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

void cell::updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  
  if (useadjoint) {
    parallel_for(RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<index.extent(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (int m=0; m<index.extent(1); m++) {
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
      for (int n=0; n<index.extent(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (int m=0; m<index.extent(1); m++) {
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

void cell::updateJacDot(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  
  if (useadjoint) {
    parallel_for(RangePolicy<AssemblyExec>(0,local_Jdot.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<index.extent(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (int m=0; m<index.extent(1); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_Jdot(e,offsets(m,k),offsets(n,j)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for(RangePolicy<AssemblyExec>(0,local_Jdot.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<index.extent(1); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (int m=0; m<index.extent(1); m++) {
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

void cell::updateParamJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<int**,AssemblyDevice> paramoffsets = wkset->paramoffsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  Kokkos::View<LO*,UnifiedDevice> numParamDOF = cellData->numParamDOF;
  
  parallel_for(RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<index.extent(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (int m=0; m<paramindex.extent(1); m++) {
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

void cell::updateParamJacDot(Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<int**,AssemblyDevice> paramoffsets = wkset->paramoffsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  Kokkos::View<LO*,UnifiedDevice> numParamDOF = cellData->numParamDOF;
  
  parallel_for(RangePolicy<AssemblyExec>(0,local_Jdot.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<index.extent(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (int m=0; m<paramindex.extent(1); m++) {
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

void cell::updateAuxJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  Kokkos::View<LO*,UnifiedDevice> numAuxDOF = cellData->numAuxDOF;
  
  parallel_for(RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<index.extent(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (int m=0; m<auxindex.extent(1); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(e,offsets(n,j),auxoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(auxoffsets(m,k));
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparamdot
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAuxJacDot(Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot) {
  
  Kokkos::View<AD**,AssemblyDevice> res_AD = wkset->res;
  Kokkos::View<int**,AssemblyDevice> offsets = wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  Kokkos::View<LO*,UnifiedDevice> numAuxDOF = cellData->numAuxDOF;
  
  parallel_for(RangePolicy<AssemblyExec>(0,local_Jdot.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<index.extent(1); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (int m=0; m<auxindex.extent(1); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_Jdot(e,offsets(n,j),auxoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(auxoffsets(m,k));
          }
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,AssemblyDevice> cell::getInitial(const bool & project, const bool & isAdjoint) {
  Kokkos::View<ScalarT**,AssemblyDevice> initialvals("initial values",numElem,GIDs.extent(1));
  wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  if (project) { // works for any basis
    for (int n=0; n<wkset->varlist.size(); n++) {
      Kokkos::View<ScalarT**,AssemblyDevice> initialip = cellData->physics_RCP->getInitial(wkset->ip,
                                                                                           wkset->varlist[n],
                                                                                           wkset->time,
                                                                                           isAdjoint,
                                                                                           wkset);
      for (int e=0; e<numElem; e++) {
        for( int i=0; i<cellData->numDOF(n); i++ ) {
          for( size_t j=0; j<wkset->numip; j++ ) {
            initialvals(e,wkset->offsets(n,i)) += initialip(e,j)*wkset->basis[wkset->usebasis[n]](e,i,j);
          }
        }
      }
    }
  }
  else { // only works if using HGRAD linear basis
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index.extent(1); n++) {
        Kokkos::View<ScalarT**,AssemblyDevice> initialnodes = cellData->physics_RCP->getInitial(nodes,
                                                                                                wkset->varlist[n],
                                                                                                wkset->time,
                                                                                                isAdjoint,
                                                                                                wkset);
        for( int i=0; i<cellData->numDOF(n); i++ ) {
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

Kokkos::View<ScalarT***,AssemblyDevice> cell::getMass() {
  Kokkos::View<ScalarT***,AssemblyDevice> mass("local mass",numElem,GIDs.extent(1), GIDs.extent(1));
  wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  vector<string> basis_types = wkset->basis_types;
  Kokkos::View<LO**,AssemblyDevice> offsets= wkset->offsets;
  Kokkos::View<LO*,UnifiedDevice> numDOF = cellData->numDOF;
  
  for (int n=0; n<index.extent(1); n++) {
    DRV basis_uw = wkset->basis_uw[wkset->usebasis[n]];
    DRV basis = wkset->basis[wkset->usebasis[n]];
    //auto cndof = Kokkos::subview(numDOF, n);
    parallel_for(RangePolicy<AssemblyExec>(0,mass.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for( int i=0; i<numDOF(n); i++ ) {
        for( int j=0; j<numDOF(n); j++ ) {
          for( size_t k=0; k<wkset->numip; k++ ) {
            mass(e,offsets(n,i),offsets(n,j)) += basis_uw(e,i,k)*basis(e,j,k);
          }
        }
      }
    });
  }
  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the error at the integration points given the solution and solve times
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT***,AssemblyDevice> cell::computeError(const ScalarT & solvetime,
                                                          const size_t & tindex,
                                                          //const bool compute_subgrid,
                                                          const vector<string> & error_types) {
  
  // Assumes that u has been updated already
  wkset->time = solvetime;
  wkset->time_KV(0) = solvetime;
  
  Kokkos::View<ScalarT***,AssemblyDevice> errors("errors",numElem,index.extent(1), error_types.size());
  //if (!compute_subgrid) {
  
    for (size_t et=0; et<error_types.size(); et++){
      if (error_types[et] == "L2") {
        wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
        Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
        seedwhat(0) = 0;
        wkset->computeSolnVolIP(u, u_dot, seedwhat);
        size_t numip = wkset->numip;
        
        Kokkos::View<ScalarT****,AssemblyDevice> truesol("true solution",numElem,index.extent(1),
                                                         numip,cellData->dimension);
        cellData->physics_RCP->trueSolution(cellData->myBlock, solvetime, truesol);
        //KokkosTools::print(truesol);
        //KokkosTools::print(wkset->local_soln);
        
        for (int e=0; e<numElem; e++) {
          for (int n=0; n<index.extent(1); n++) {
            for( size_t j=0; j<numip; j++ ) {
              for (size_t d=0; d<wkset->local_soln.extent(3); d++) {
                ScalarT diff = wkset->local_soln(e,n,j,d).val() - truesol(e,n,j,d);
                errors(e,n,et) += diff*diff*wkset->wts(e,j);
              }
            }
          }
        }
      }
      else if (error_types[et] == "H1") {
        wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
        Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
        seedwhat(0) = 0;
        wkset->computeSolnVolIP(u, u_dot, seedwhat);
        size_t numip = wkset->numip;
        
        Kokkos::View<ScalarT****,AssemblyDevice> truesol("true solution",numElem,index.extent(1),
                                                         numip,cellData->dimension);
        cellData->physics_RCP->trueSolutionGrad(cellData->myBlock, solvetime, truesol);
        for (int e=0; e<numElem; e++) {
          for (int n=0; n<index.extent(1); n++) {
            for( size_t j=0; j<numip; j++ ) {
              for (int s=0; s<cellData->dimension; s++) {
                ScalarT diff = wkset->local_soln_grad(e,n,j,s).val() - truesol(e,n,j,s);
                errors(e,n,et) += diff*diff*wkset->wts(e,j);
              }
            }
          }
        }
      }
      else if (error_types[et] == "L2-face") {
        for (size_t s=0; s<cellData->numSides; s++) {
          
          wkset->updateFace(nodes, orientation, s);
          Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
          seedwhat(0) = 0;
          wkset->computeSolnFaceIP(u, u_dot, seedwhat);

          size_t numip = wkset->numsideip;
        
          Kokkos::View<ScalarT****,AssemblyDevice> truesol("true solution",numElem,index.extent(1),
                                                           numip,cellData->dimension);
          cellData->physics_RCP->trueSolutionFace(cellData->myBlock, solvetime, truesol);
          for (int e=0; e<numElem; e++) {
            double edgelength = 0.0;
            for( size_t j=0; j<numip; j++ ) {
              edgelength += wkset->wts_side(e,j);
            }
            for (int n=0; n<index.extent(1); n++) {
              for( size_t j=0; j<numip; j++ ) {
                //for (size_t d=0; d<wkset->local_soln_face.extent(3); d++) { // this loop might be unnecessary ... any vector HFACE variables?
                  ScalarT diff = wkset->local_soln_face(e,n,j,0).val() - truesol(e,n,j,0);
                errors(e,n,et) += 0.5/edgelength*diff*diff*wkset->wts_side(e,j);
                //}
              }
            }
          }
        }
      }
    }
  //}
  /*
  else if (cellData->multiscale) {
    
    for (int e=0; e<numElem; e++) {
      
      Kokkos::View<ScalarT**,AssemblyDevice> currerrors = subgridModels[subgrid_model_index[e][tindex]]->computeError(solvetime, subgrid_usernum[e]);
      
      for (int c=0; c<currerrors.extent(0); c++) { // loop over subgrid elements
        for (int n=0; n<index.extent(1); n++) {
          errors(e,n) += currerrors(c,n);
        }
      }
    }
    
  }*/
  
  return errors;
}


///////////////////////////////////////////////////////////////////////////////////////
// Compute the response at a given set of points and time
///////////////////////////////////////////////////////////////////////////////////////
Kokkos::View<AD***,AssemblyDevice> cell::computeResponseAtNodes(const DRV & nodes,
                                                                const int tindex,
                                                                const ScalarT & time) {
  
  
  // TMW: this whole function needs to be rewritten to use worksets properly
  /*
   size_t numip = nodes.extent(1);
   wkset->update(ip,jacobian,orientation);
   
   // Map the local solution to the solution and gradient at ip
   FCAD u_ip(numElem,index[0].size(),numip);
   FCAD ugrad_ip(numElem,index[0].size(),numip,dimension);
   for (int e=0; e<numElem; e++) {
   for (int n=0; n<index.extent(1); n++) {
   for( int i=0; i<index.extent(2); i++ ) {
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
   //FCAD p_ip(numElem,paramindex.extent(1), numip);
   //FCAD pgrad_ip(numElem,paramindex.extent(1), numip, dimension);
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
Kokkos::View<AD***,AssemblyDevice> cell::computeResponse(const ScalarT & solvetime,
                                                         const size_t & tindex,
                                                         const int & seedwhat) {
  
  // Assumes that u has already been filled
  
  // seedwhat indicates what needs to be seeded
  // seedwhat == 0 => seed nothing
  // seedwhat == 1 => seed sol
  // seedwhat == j (j>1) => seed (j-1)-derivative of sol
  
  
  Kokkos::View<AD***,AssemblyDevice> response;
  bool useSensors = false;
  if (cellData->response_type == "pointwise") {
    useSensors = true;
  }
  
  size_t numip = wkset->ip.extent(1);
  if (useSensors) {
    numip = sensorLocations.size();
  }
  
  wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  
  //KokkosTools::print(u);
  if (numip > 0) {
    // Extract the local solution at this time
    // We automatically seed the AD and adjust it below
    Kokkos::View<AD***,AssemblyDevice> u_dof("u_dof",numElem,index.extent(1),GIDs.extent(1));
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index.extent(1); n++) {
        for( int i=0; i<cellData->numDOF(n); i++ ) {
          u_dof(e,n,i) = AD(maxDerivs,wkset->offsets(n,i),u(e,n,i));
        }
      }
    }
    
    // Map the local solution to the solution and gradient at ip
    Kokkos::View<AD****,AssemblyDevice> u_ip("u_ip",numElem,index.extent(1),
                                             numip,cellData->dimension);
    Kokkos::View<AD****,AssemblyDevice> ugrad_ip("ugrad_ip",numElem,index.extent(1),
                                                 numip,cellData->dimension);
    for (int e=0; e<numElem; e++) {
      for (int n=0; n<index.extent(1); n++) {
        for( int i=0; i<cellData->numDOF(n); i++ ) {
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
          for (int s=0; s<cellData->dimension; s++) {
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
        for (int n=0; n<index.extent(1); n++) {
          for( size_t j=0; j<numip; j++ ) {
            u_ip(e,n,j,0) = u_ip(e,n,j,0).val();
            for (int s=0; s<cellData->dimension; s++) {
              ugrad_ip(e,n,j,s) = ugrad_ip(e,n,j,s).val();
            }
          }
        }
      }
    }
    else if (seedwhat == 1) { // remove seeding on gradient
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<index.extent(1); n++) {
          for( size_t j=0; j<numip; j++ ) {
            for (int s=0; s<cellData->dimension; s++) {
              ugrad_ip(e,n,j,s) = ugrad_ip(e,n,j,s).val();
            }
          }
        }
      }
      //KokkosTools::print(u_ip);
      
    }
    else {
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<index.extent(1); n++) {
          for( size_t j=0; j<numip; j++ ) {
            for (int s=0; s<cellData->dimension; s++) {
              if ((seedwhat-2) == s) {
                ScalarT tmp = ugrad_ip(e,n,j,s).val();
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
    
    if (paramindex.extent(0) > 0) {
      // Extract the local solution at this time
      // We automatically seed the AD and adjust it below
      Kokkos::View<AD***,AssemblyDevice> param_dof("param dof",numElem,paramindex.extent(1),paramGIDs.extent(1));
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<paramindex.extent(1); n++) {
          for( int i=0; i<cellData->numParamDOF(n); i++ ) {
            param_dof(e,n,i) = AD(maxDerivs,wkset->paramoffsets(n,i),param(e,n,i));
          }
        }
      }
      
      // Map the local solution to the solution and gradient at ip
      param_ip = Kokkos::View<AD****,AssemblyDevice>("u_ip",numElem,paramindex.extent(1),
                                                     numip,cellData->dimension);
      paramgrad_ip = Kokkos::View<AD****,AssemblyDevice>("ugrad_ip",numElem,paramindex.extent(1),
                                                         numip,cellData->dimension);
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<paramindex.extent(1); n++) {
          for( int i=0; i<cellData->numParamDOF(n); i++ ) {
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
            for (int s=0; s<cellData->dimension; s++) {
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
          for (int n=0; n<paramindex.extent(1); n++) {
            for( size_t j=0; j<numip; j++ ) {
              for (int s=0; s<cellData->dimension; s++) {
                paramgrad_ip(e,n,j,s) = paramgrad_ip(e,n,j,s).val();
              }
            }
          }
        }
      }
      else {
        for (int e=0; e<numElem; e++) {
          for (int n=0; n<paramindex.extent(1); n++) {
            for( size_t j=0; j<numip; j++ ) {
              param_ip(e,n,j,0) = param_ip(e,n,j,0).val();
              for (int s=0; s<cellData->dimension; s++) {
                paramgrad_ip(e,n,j,s) = paramgrad_ip(e,n,j,s).val();
              }
            }
          }
        }
      }
    }
    
    if (useSensors) {
      if (sensorLocations.size() > 0){
        response = cellData->physics_RCP->getResponse(cellData->myBlock, u_ip, ugrad_ip, param_ip,
                                                      paramgrad_ip, sensorPoints,
                                                      solvetime, wkset);
      }
    }
    else {
      response = cellData->physics_RCP->getResponse(cellData->myBlock, u_ip, ugrad_ip, param_ip,
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

Kokkos::View<AD**,AssemblyDevice> cell::computeObjective(const ScalarT & solvetime,
                                                         const size_t & tindex,
                                                         const int & seedwhat) {
  
  // assumes the params have been seeded elsewhere (solver, postprocess interfaces)
  Kokkos::View<AD**,AssemblyDevice> objective;
  
  if (!(cellData->multiscale) || cellData->mortar_objective) {
    
    Kokkos::View<AD***,AssemblyDevice> responsevals = computeResponse(solvetime,tindex,seedwhat);
    
    if (cellData->response_type == "pointwise") { // uses sensor data
      
      ScalarT TOL = 1.0e-6; // tolerance for comparing sensor times and simulation times
      objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,numSensors);
      
      if (numSensors > 0) { // if this element has any sensors
        for (size_t s=0; s<numSensors; s++) {
          bool foundtime = false;
          size_t ftime;
          
          for (size_t t2=0; t2<sensorData[s].extent(0); t2++) {
            ScalarT stime = sensorData[s](t2,0);
            if (abs(stime-solvetime) < TOL) {
              foundtime = true;
              ftime = t2;
            }
          }
          
          if (foundtime) {
            int ee = sensorElem[s];
            for (size_t r=0; r<responsevals.extent(1); r++) {
              AD rval = responsevals(ee,r,s);
              ScalarT sval = sensorData[s](ftime,r+1);
              if(cellData->compute_diff) {
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
    else if (cellData->response_type == "global") { // uses physicsmodules->target
      objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,wkset->ip.extent(1));
      Kokkos::View<AD***,AssemblyDevice> ctarg = computeTarget(solvetime);
      Kokkos::View<AD***,AssemblyDevice> cweight = computeWeight(solvetime);
      
      for (int e=0; e<numElem; e++) {
        for (size_t r=0; r<responsevals.extent(1); r++) {
          for (size_t k=0; k<wkset->ip.extent(1); k++) {
            AD diff = responsevals(e,r,k)-ctarg(e,r,k);
            if(cellData->compute_diff) {
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
    
    //for (int e=0; e<numElem; e++) {
      int sgindex = subgrid_model_index[tindex];
      Kokkos::View<AD*,AssemblyDevice> cobj = subgridModels[sgindex]->computeObjective(cellData->response_type,seedwhat,
                                                                                       solvetime,subgrid_usernum);
      
      //if (e == 0) {
        objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,cobj.extent(0));
      //}
      for (int i=0; i<cobj.extent(0); i++) {
        objective(0,i) += cobj(i); // TMW: tempory fix
      }
      
      
    //}
  }
  
  return objective;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the regularization over the domain given the domain discretized parameters
///////////////////////////////////////////////////////////////////////////////////////
AD cell::computeDomainRegularization(const vector<ScalarT> reg_constants, const vector<int> reg_types,
                                     const vector<int> reg_indices) {
  
  AD reg;
  
  bool seedParams = true;
  //vector<vector<AD> > param_AD;
  //for (int n=0; n<paramindex.size(); n++) {
  //  param_AD.push_back(vector<AD>(paramindex[n].size()));
  //}
  //this->setLocalADParams(param_AD,seedParams);
  int numip = wkset->numip;
  wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  Kokkos::View<int*,UnifiedDevice> seedwhat("int for seeding",1);
  seedwhat(0) = 3;  
  wkset->computeParamVolIP(param, seedwhat);
  
  AD p, dpdx, dpdy, dpdz; // parameters
  ScalarT regoffset = 1.0e-5;
  int numParams = reg_indices.size();
  int paramIndex, reg_type;
  ScalarT reg_constant;
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
          if (cellData->dimension > 1)
            dpdy = wkset->local_param_grad(e,paramIndex,k,1);
          if (cellData->dimension > 2)
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

Kokkos::View<AD***,AssemblyDevice> cell::computeTarget(const ScalarT & solvetime) {
  return cellData->physics_RCP->target(cellData->myBlock, wkset->ip, solvetime, wkset);
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the weighting function at the integration points given the solve times
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> cell::computeWeight(const ScalarT & solvetime) {
  return cellData->physics_RCP->weight(cellData->myBlock, wkset->ip, solvetime, wkset);
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the response at the integration points given the solution and solve times
///////////////////////////////////////////////////////////////////////////////////////

void cell::addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                      const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                      const vector<basis_RCP> & basis_pointers,
                      const vector<basis_RCP> & param_basis_pointers) {
  
  
  // If we have sensors, then we set the response type to pointwise
  cellData->response_type = "pointwise";
  useSensors = true;
  bool useFineScale = true;
  if (!(cellData->multiscale) || cellData->mortar_objective) {
    useFineScale = false;
  }
  
  Teuchos::RCP<DiscTools> discTools = Teuchos::rcp( new DiscTools() );
  
  if (cellData->exodus_sensors) {
    // don't use sensor_points
    // set sensorData and sensorLocations from exodus file
    if (sensorLocations.size() > 0) {
      sensorPoints = DRV("sensorPoints",1,sensorLocations.size(),cellData->dimension);
      for (size_t i=0; i<sensorLocations.size(); i++) {
        for (int j=0; j<cellData->dimension; j++) {
          sensorPoints(0,i,j) = sensorLocations[i](0,j);
        }
        sensorElem.push_back(0);
      }
      DRV refsenspts_buffer("refsenspts_buffer",1,sensorLocations.size(),cellData->dimension);
      Intrepid2::CellTools<PHX::Device>::mapToReferenceFrame(refsenspts_buffer, sensorPoints, nodes, *(cellData->cellTopo));
      DRV refsenspts("refsenspts",sensorLocations.size(),cellData->dimension);
      Kokkos::deep_copy(refsenspts,Kokkos::subdynrankview(refsenspts_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
      
      vector<DRV> csensorBasis;
      vector<DRV> csensorBasisGrad;
      
      for (size_t b=0; b<basis_pointers.size(); b++) {
        csensorBasis.push_back(discTools->evaluateBasis(basis_pointers[b], refsenspts, orientation));
        csensorBasisGrad.push_back(discTools->evaluateBasisGrads(basis_pointers[b], nodes, refsenspts,
                                                                 cellData->cellTopo, orientation));
      }
      
      sensorBasis.push_back(csensorBasis);
      sensorBasisGrad.push_back(csensorBasisGrad);
      
      
      vector<DRV> cpsensorBasis;
      vector<DRV> cpsensorBasisGrad;
      
      for (size_t b=0; b<param_basis_pointers.size(); b++) {
        cpsensorBasis.push_back(discTools->evaluateBasis(param_basis_pointers[b], refsenspts, orientation));
        cpsensorBasisGrad.push_back(discTools->evaluateBasisGrads(param_basis_pointers[b], nodes,
                                                                  refsenspts, cellData->cellTopo, orientation));
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
                                     basis_pointers, subgrid_usernum);
        //}
      }
      
    }
    else {
      DRV phys_points("phys_points",1,sensor_points.extent(0),cellData->dimension);
      for (size_t i=0; i<sensor_points.extent(0); i++) {
        for (int j=0; j<cellData->dimension; j++) {
          phys_points(0,i,j) = sensor_points(i,j);
        }
      }
      
      if (!(cellData->loadSensorFiles)) {
        for (int e=0; e<numElem; e++) {
          
          DRV refpts("refpts", 1, sensor_points.extent(0), sensor_points.extent(1));
          DRVint inRefCell("inRefCell", 1, sensor_points.extent(0));
          DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
          for (int i=0; i<nodes.extent(1); i++) {
            for (int j=0; j<nodes.extent(2); j++) {
              cnodes(0,i,j) = nodes(e,i,j);
            }
          }
          CellTools::mapToReferenceFrame(refpts, phys_points, cnodes, *(cellData->cellTopo));
          CellTools::checkPointwiseInclusion(inRefCell, refpts, *(cellData->cellTopo), sensor_loc_tol);
          
          for (size_t i=0; i<sensor_points.extent(0); i++) {
            if (inRefCell(0,i) == 1) {
              
              Kokkos::View<ScalarT**,HostDevice> newsenspt("new sensor point",1,cellData->dimension);
              for (int j=0; j<cellData->dimension; j++) {
                newsenspt(0,j) = sensor_points(i,j);
              }
              sensorLocations.push_back(newsenspt);
              mySensorIDs.push_back(i);
              sensorElem.push_back(e);
              if (have_sensor_data) {
                sensorData.push_back(sensor_data[i]);
              }
              if (cellData->writeSensorFiles) {
                stringstream ss;
                ss << localElemID(e);
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
      
      if (cellData->loadSensorFiles) {
        for (int e=0; e<numElem; e++) {
          stringstream ss;
          ss << localElemID(e);
          string str = ss.str();
          ifstream sfile;
          sfile.open("sensorLocations/sdat." + str + ".dat");
          int cID;
          //ScalarT l1, l2, t1, d1, d2;
          ScalarT l1, l2;
          sfile >> cID;
          sfile >> l1;
          sfile >> l2;
          
          sfile.close();
          
          Kokkos::View<ScalarT**,HostDevice> newsenspt("sensor point",1,cellData->dimension);
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
        sensorPoints = DRV("sensorPoints",numElem,numSensors,cellData->dimension);
        
        for (size_t i=0; i<numSensors; i++) {
          
          DRV csensorPoints("sensorPoints",1,1,cellData->dimension);
          DRV cnodes("current nodes",1,nodes.extent(1), nodes.extent(2));
          for (int j=0; j<cellData->dimension; j++) {
            csensorPoints(0,0,j) = sensorLocations[i](0,j);
            sensorPoints(0,i,j) = sensorLocations[i](0,j);
            for (int k=0; k<nodes.extent(1); k++) {
              cnodes(0,k,j) = nodes(sensorElem[i],k,j);
            }
          }
          
          
          DRV refsenspts_buffer("refsenspts_buffer",1,1,cellData->dimension);
          DRV refsenspts("refsenspts",1,cellData->dimension);
          
          CellTools::mapToReferenceFrame(refsenspts_buffer, csensorPoints, cnodes, *(cellData->cellTopo));
          //CellTools<AssemblyDevice>::mapToReferenceFrame(refsenspts, csensorPoints, cnodes, *cellTopo);
          Kokkos::deep_copy(refsenspts,Kokkos::subdynrankview(refsenspts_buffer,0,Kokkos::ALL(),Kokkos::ALL()));
          
          vector<DRV> csensorBasis;
          vector<DRV> csensorBasisGrad;
          
          for (size_t b=0; b<basis_pointers.size(); b++) {
            csensorBasis.push_back(discTools->evaluateBasis(basis_pointers[b], refsenspts, orientation));
            csensorBasisGrad.push_back(discTools->evaluateBasisGrads(basis_pointers[b], cnodes,
                                                                     refsenspts, cellData->cellTopo, orientation));
          }
          sensorBasis.push_back(csensorBasis);
          sensorBasisGrad.push_back(csensorBasisGrad);
          
          
          vector<DRV> cpsensorBasis;
          vector<DRV> cpsensorBasisGrad;
          
          for (size_t b=0; b<param_basis_pointers.size(); b++) {
            cpsensorBasis.push_back(discTools->evaluateBasis(param_basis_pointers[b], refsenspts, orientation));
            cpsensorBasisGrad.push_back(discTools->evaluateBasisGrads(param_basis_pointers[b], nodes,
                                                                      refsenspts, cellData->cellTopo, orientation));
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
// Update the subgrid model
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateSubgridModel(vector<Teuchos::RCP<SubGridModel> > & models) {
  
  /*
   wkset->update(ip,jacobian);
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
  if (cellData->have_cell_phi) {
    wkset->have_rotation_phi = true;
    wkset->rotation_phi = cell_data;
  }
  else if (cellData->have_cell_rotation) {
    wkset->have_rotation = true;
    //Kokkos::View<ScalarT***,AssemblyDevice> rotmat("rotation matrix",numElem,3,3);
    for (int e=0; e<numElem; e++) {
      wkset->rotation(e,0,0) = cell_data(e,0);
      wkset->rotation(e,0,1) = cell_data(e,1);
      wkset->rotation(e,0,2) = cell_data(e,2);
      wkset->rotation(e,1,0) = cell_data(e,3);
      wkset->rotation(e,1,1) = cell_data(e,4);
      wkset->rotation(e,1,2) = cell_data(e,5);
      wkset->rotation(e,2,0) = cell_data(e,6);
      wkset->rotation(e,2,1) = cell_data(e,7);
      wkset->rotation(e,2,2) = cell_data(e,8);
    }
    /*
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
     }*/
    //wkset->rotation = rotmat;
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Pass the cell data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetAdjPrev(const ScalarT & val) {
  parallel_for(RangePolicy<AssemblyExec>(0,adjPrev.extent(0)), KOKKOS_LAMBDA (const int i ) {
    for (int j=0; j<adjPrev.extent(1); j++) {
      adjPrev(i,j) = val;
    }
  });
}
