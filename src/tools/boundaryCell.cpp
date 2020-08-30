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
                           const DRV nodes_,
                           const Kokkos::View<LO*,AssemblyDevice> localID_,
                           const Kokkos::View<LO*,AssemblyDevice> sideID_,
                           const int & sidenum_, const string & sidename_,
                           const int & cellID_,
                           LIDView LIDs_,
                           Kokkos::View<int****,HostDevice> sideinfo_,
                           Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orientation_) :
cellData(cellData_), localElemID(localID_), localSideID(sideID_), nodes(nodes_),
sidenum(sidenum_), sidename(sidename_), cellID(cellID_), 
LIDs(LIDs_), sideinfo(sideinfo_), orientation(orientation_) {
  
  numElem = nodes.extent(0);
  
  LIDs_host = Kokkos::create_mirror_view(LIDs);
  Kokkos::deep_copy(LIDs_host,LIDs);
  
  {
    Teuchos::TimeMonitor localtimer(*buildBasisTimer);
    
    DRV ref_ip = cellData->ref_side_ip[localSideID(0)];
    DRV ref_wts = cellData->ref_side_wts[localSideID(0)];
    
    int dimension = cellData->dimension;
    int numip = ref_ip.extent(0);
    
    ip = DRV("boundary ip", numElem, numip, dimension);
    DRV ijac("bijac", numElem, numip, dimension, dimension);
    DRV ijacDet("bijacDet", numElem, numip);
    DRV ijacInv("bijacInv", numElem, numip, dimension, dimension);
    wts = DRV("boundary wts", numElem, numip);
    normals = DRV("boundary normals", numElem, numip, dimension);
    tangents = DRV("boundary tangents", numElem, numip, dimension);
    
    {
      //Teuchos::TimeMonitor dbgtimer(*worksetDebugTimer0);
      CellTools::mapToPhysicalFrame(ip, ref_ip, nodes, *(cellData->cellTopo));
      CellTools::setJacobian(ijac, ref_ip, nodes, *(cellData->cellTopo));
      CellTools::setJacobianInv(ijacInv, ijac);
      CellTools::setJacobianDet(ijacDet, ijac);
    }
    
    {
      //Teuchos::TimeMonitor dbgtimer(*worksetDebugTimer1);
      
      if (dimension == 2) {
        DRV ref_tangents = cellData->ref_side_tangents[localSideID(0)];
        Intrepid2::RealSpaceTools<AssemblyExec>::matvec(tangents, ijac, ref_tangents);
        
        DRV rotation("rotation matrix",dimension,dimension);
        rotation(0,0) = 0;  rotation(0,1) = 1;
        rotation(1,0) = -1; rotation(1,1) = 0;
        Intrepid2::RealSpaceTools<AssemblyExec>::matvec(normals, rotation, tangents);
        
        Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(wts, tangents, Intrepid2::NORM_TWO);
        Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(wts, wts, ref_wts);
        
      }
      else if (dimension == 3) {
        
        DRV ref_tangentsU = cellData->ref_side_tangentsU[localSideID(0)];
        DRV ref_tangentsV = cellData->ref_side_tangentsV[localSideID(0)];
        
        DRV faceTanU("face tangent U", numElem, numip, dimension);
        DRV faceTanV("face tangent V", numElem, numip, dimension);
        
        Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanU, ijac, ref_tangentsU);
        Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanV, ijac, ref_tangentsV);
        
        Intrepid2::RealSpaceTools<AssemblyExec>::vecprod(normals, faceTanU, faceTanV);
        
        Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(wts, normals, Intrepid2::NORM_TWO);
        Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(wts, wts, ref_wts);
        
      }
      
    }
    
    hsize = Kokkos::View<ScalarT*,AssemblyDevice>("element sizes",numElem);
    parallel_for("bcell hsize",RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int e ) {
      ScalarT vol = 0.0;
      for (int i=0; i<wts.extent(1); i++) {
        vol += wts(e,i);
      }
      ScalarT dimscl = 1.0/((ScalarT)ip.extent(2)-1.0);
      hsize(e) = std::pow(vol,dimscl);
    });
    
    // TMW: this might not be needed
    // scale the normal vector (we need unit normal...)
    parallel_for("bcell normal rescale",RangePolicy<AssemblyExec>(0,normals.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int j=0; j<normals.extent(1); j++ ) {
        ScalarT normalLength = 0.0;
        for (int sd=0; sd<normals.extent(2); sd++) {
          normalLength += normals(e,j,sd)*normals(e,j,sd);
        }
        normalLength = std::sqrt(normalLength);
        for (int sd=0; sd<normals.extent(2); sd++) {
          normals(e,j,sd) = normals(e,j,sd) / normalLength;
        }
      }
    });
    
    {
      
      for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
        
        int numb = cellData->basis_pointers[i]->getCardinality();
        DRV basis_vals, basis_grad_vals, basis_div_vals, basis_curl_vals;
        
        DRV ref_basis_vals = cellData->ref_side_basis[localSideID(0)][i];
        
        if (cellData->basis_types[i] == "HGRAD"){
          
          DRV basis_vals_tmp("tmp basis_vals",numElem, numb, numip);
          FuncTools::HGRADtransformVALUE(basis_vals_tmp, ref_basis_vals);
          basis_vals = DRV("basis_vals",numElem, numb, numip);
          OrientTools::modifyBasisByOrientation(basis_vals, basis_vals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          
          DRV ref_basis_grad_vals = cellData->ref_side_basis_grad[localSideID(0)][i];
          
          DRV basis_grad_vals_tmp("basis_grad_side tmp",numElem,numb,numip,dimension);
          FuncTools::HGRADtransformGRAD(basis_grad_vals_tmp, ijacInv, ref_basis_grad_vals);
          
          basis_grad_vals = DRV("basis_grad_vals",numElem,numb,numip,dimension);
          OrientTools::modifyBasisByOrientation(basis_grad_vals, basis_grad_vals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          
        }
        else if (cellData->basis_types[i] == "HVOL"){ // does not require orientations
          
          basis_vals = DRV("basis_vals",numElem, numb, numip);
          FuncTools::HGRADtransformVALUE(basis_vals, ref_basis_vals);
          
        }
        else if (cellData->basis_types[i] == "HFACE"){
          
          DRV basis_vals_tmp("tmp basis_vals",numElem, numb, numip);
          FuncTools::HGRADtransformVALUE(basis_vals_tmp, ref_basis_vals);
          basis_vals = DRV("basis_vals",numElem, numb, numip);
          OrientTools::modifyBasisByOrientation(basis_vals, basis_vals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
        }
        else if (cellData->basis_types[i] == "HDIV"){
          
          DRV basis_vals_tmp("tmp basis_vals",numElem, numb, numip, dimension);
          
          FuncTools::HDIVtransformVALUE(basis_vals_tmp, ijac, ijacDet, ref_basis_vals);
          basis_vals = DRV("basis_vals",numElem, numb, numip, dimension);
          OrientTools::modifyBasisByOrientation(basis_vals, basis_vals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
        }
        else if (cellData->basis_types[i] == "HCURL"){
          
        }
        basis.push_back(basis_vals);
        basis_grad.push_back(basis_grad_vals);
        basis_div.push_back(basis_div_vals);
        basis_curl.push_back(basis_curl_vals);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setWorkset(Teuchos::RCP<workset> & wkset_) {
  
  wkset = wkset_;
  
  // Frequently used Views
  res_AD = wkset->res;
  offsets = wkset->offsets;
  //paramoffsets = wkset->paramoffsets;
  
  numDOF = cellData->numDOF;
  //numParamDOF = cellData->numParamDOF;
  numAuxDOF = cellData->numAuxDOF;
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setParams(LIDView paramLIDs_) {
  paramLIDs = paramLIDs_;
  paramLIDs_host = Kokkos::create_mirror_view(paramLIDs);
  Kokkos::deep_copy(paramLIDs_host, paramLIDs);
  
  // This has now been set
  numParamDOF = cellData->numParamDOF;
  
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

void BoundaryCell::setUseBasis(vector<int> & usebasis_, const int & numsteps, const int & numstages) {
  vector<int> usebasis = usebasis_;
  
  // Set up the containers for usual solution storage
  size_t maxnbasis = 0;
  for (size_t i=0; i<cellData->numDOF.extent(0); i++) {
    if (cellData->numDOF(i) > maxnbasis) {
      maxnbasis = cellData->numDOF(i);
    }
  }
  u = Kokkos::View<ScalarT***,AssemblyDevice>("u",numElem,cellData->numDOF.extent(0),maxnbasis);
  phi = Kokkos::View<ScalarT***,AssemblyDevice>("phi",numElem,cellData->numDOF.extent(0),maxnbasis);
  u_prev = Kokkos::View<ScalarT****,AssemblyDevice>("u previous",numElem,cellData->numDOF.extent(0),maxnbasis,numsteps);
  u_stage = Kokkos::View<ScalarT****,AssemblyDevice>("u stages",numElem,cellData->numDOF.extent(0),maxnbasis,numstages);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
  numParamDOF = cellData->numParamDOF;
  
  size_t maxnbasis = 0;
  for (size_t i=0; i<numParamDOF.extent(0); i++) {
    if (numParamDOF(i) > maxnbasis) {
      maxnbasis = numParamDOF(i);
    }
  }
  param = Kokkos::View<ScalarT***,AssemblyDevice>("param",numElem,
                                                  numParamDOF.extent(0),maxnbasis);
  
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
// Update the workset
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateWorksetBasis() {
  wkset->ip_side = ip;
  wkset->wts_side = wts;
  wkset->normals = normals;
  wkset->h = hsize;
  
  // TMW: can these be avoided??
  Kokkos::deep_copy(wkset->normals_KV,normals);
  Kokkos::deep_copy(wkset->ip_side_KV,ip);
  
  wkset->basis_side = basis;
  wkset->basis_grad_side = basis_grad;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the coarse grid solution to the fine grid integration points
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::computeSoln(const int & seedwhat) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnSideTimer);
  
  wkset->computeSolnSideIP();
  //wkset->computeParamSideIP(sidenum, param, seedwhat);
  
  if (wkset->numAux > 0) {
    
    wkset->resetAuxSide();
    
    size_t numip = wkset->numsideip;
    AD auxval;
    
    // TMW: this will not work on GPU
    for (int e=0; e<numElem; e++) {
      
      for (size_t k=0; k<numAuxDOF.extent(0); k++) {
        for(size_t i=0; i<numAuxDOF(k); i++ ) {
          ScalarT auxtmp = aux(localElemID[e],k,i);
          if (seedwhat == 4) {
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
                                 Kokkos::View<ScalarT***,AssemblyDevice> local_res,
                                 Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  // Boundary contribution
  {
    Teuchos::TimeMonitor localtimer(*boundaryResidualTimer);
    
    this->updateWorksetBasis();
    
    wkset->sidename = sidename;
    wkset->currentside = sidenum;
    
    int seedwhat = 0;
    if (compute_jacobian) {
      if (compute_disc_sens) {
        seedwhat = 3;
      }
      else if (compute_aux_sens) {
        seedwhat = 4;
      }
      else {
        seedwhat = 1;
      }
    }
    
    if (isTransient) {
      wkset->computeSolnTransientSeeded(u, u_prev, u_stage, seedwhat);
    }
    else { // steady-state
      wkset->computeSolnSteadySeeded(u, seedwhat);
    }
    
    this->computeSoln(seedwhat);
    wkset->computeParamSideIP(sidenum, param, seedwhat);
    
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
  
  if (compute_sens) {
    parallel_for("bcell update res sens",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int r=0; r<local_res.extent(2); r++) {
        for (unsigned int n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("bcell update res",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<numDOF.extent(0); n++) {
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
  Kokkos::View<AD**,AssemblyDevice> adjres_AD = wkset->adjrhs;
  
  if (compute_sens) {
    parallel_for("bcell update res adjoint sens",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int r=0; r<maxDerivs; r++) {
        for (unsigned int n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= adjres_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("bcell update adjoint res",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(e,offsets(n,j),0) -= adjres_AD(e,offsets(n,j)).val();
        }
      }
    });
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  if (useadjoint) {
    parallel_for("bcell update jac adjoint",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(e,offsets(m,k),offsets(n,j)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("bcell update jac",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
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

void BoundaryCell::updateParamJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  paramoffsets = wkset->paramoffsets;
  numParamDOF = cellData->numParamDOF;
  
  parallel_for("bcell update param jac",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int n=0; n<numDOF.extent(0); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numParamDOF.extent(0); m++) {
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

void BoundaryCell::updateAuxJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
  numAuxDOF = cellData->numAuxDOF;
  
  parallel_for("bcell update aux jac",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (unsigned int n=0; n<numDOF.extent(0); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numAuxDOF.extent(0); m++) {
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
      
      //wkset->updateSide(sidenum, cellID);
      this->updateWorksetBasis();
      wkset->computeParamSideIP(sidenum, param, 3);
      
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
  
  wkset->setTime(time);
  
  auto u_kv = gl_u->getLocalView<AssemblyDevice>();
  auto du_kv = gl_du->getLocalView<AssemblyDevice>();
  
  Kokkos::View<AD***,AssemblyDevice> u_AD("temp u AD",u.extent(0),u.extent(1),u.extent(2));
  Kokkos::View<AD***,AssemblyDevice> param_AD("temp u AD",1,1,1);
  
  {
    Teuchos::TimeMonitor localtimer(*cellFluxGatherTimer);
    
    if (compute_sens) {
      parallel_for("bcell flux gather",RangePolicy<AssemblyExec>(0,u_AD.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_t var=0; var<u_AD.extent(1); var++) {
          for( size_t dof=0; dof<u_AD.extent(2); dof++ ) {
            u_AD(elem,var,dof) = AD(u_kv(LIDs(elem,offsets(var,dof)),0));
          }
        }
      });
    }
    else {
      parallel_for("bcell flux gather",RangePolicy<AssemblyExec>(0,u_AD.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_t var=0; var<u_AD.extent(1); var++) {
          for( size_t dof=0; dof<u_AD.extent(2); dof++ ) {
            u_AD(elem,var,dof) = AD(maxDerivs, 0, u_kv(LIDs(elem,offsets(var,dof)),0));
            for( size_t p=0; p<du_kv.extent(1); p++ ) {
              u_AD(elem,var,dof).fastAccessDx(p) = du_kv(LIDs(elem,offsets(var,dof)),p);
            }
          }
        }
      });
    }
  }
  
  {
    Teuchos::TimeMonitor localtimer(*cellFluxWksetTimer);
    
    wkset->computeSolnSideIP(sidenum, u_AD, param_AD);
  }
  numAuxDOF = cellData->numAuxDOF;
  
  if (wkset->numAux > 0) {
    
    Teuchos::TimeMonitor localtimer(*cellFluxAuxTimer);
    
    wkset->resetAuxSide();
    
    size_t numip = wkset->numsideip;
    AD auxval;
    for (int e=0; e<numElem; e++) {
      for (size_t k=0; k<numAuxDOF.extent(0); k++) {
        for(size_t i=0; i<numAuxDOF(k); i++ ) {
          auxval = AD(maxDerivs, auxoffsets(k,i), lambda(localElemID[e],k,i));
          for( size_t j=0; j<numip; j++ ) {
            wkset->local_aux_side(e,k,j) += auxval*auxside_basis[auxusebasis[k]](e,i,j);
          }
        }
      }
    }
  }
  
  //wkset->resetFlux();
  {
    Teuchos::TimeMonitor localtimer(*cellFluxEvalTimer);
    
    cellData->physics_RCP->computeFlux(cellData->myBlock);
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,AssemblyDevice> BoundaryCell::getDirichlet() {
  
  Kokkos::View<ScalarT**,AssemblyDevice> dvals("initial values",numElem,LIDs.extent(1));
  this->updateWorksetBasis();
  //wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  Kokkos::View<int[1],AssemblyDevice> currind("current index");
  auto currind_host = Kokkos::create_mirror_view(currind);
  
  Kokkos::View<int**,HostDevice> bcs = wkset->var_bcs;
  
  for (int n=0; n<wkset->varlist.size(); n++) {
    if (bcs(n,sidenum) == 1) { // is this a strong DBC for this variable
      Kokkos::View<ScalarT**,AssemblyDevice> dip = cellData->physics_RCP->getDirichlet(ip,n,
                                                                                        cellData->myBlock,
                                                                                        sidename,
                                                                                        wkset);
      int bind = wkset->usebasis[n];
      std::string btype = cellData->basis_types[bind];
      currind_host(0) = n;
      Kokkos::deep_copy(currind,currind_host);
      DRV cbasis = basis[bind];
      
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bcell fill Dirichlet",RangePolicy<AssemblyExec>(0,dvals.extent(0)), KOKKOS_LAMBDA (const int e ) {
          int n = currind(0);
          for( int i=0; i<numDOF(n); i++ ) {
            for( size_t j=0; j<dip.extent(1); j++ ) {
              dvals(e,offsets(n,i)) += dip(e,j)*cbasis(e,i,j)*wts(e,j);
            }
          }
        });
      }
      else if (btype == "HDIV"){
        parallel_for("bcell fill Dirichlet HDIV",RangePolicy<AssemblyExec>(0,dvals.extent(0)), KOKKOS_LAMBDA (const int e ) {
          int n = currind(0);
          for( int i=0; i<numDOF(n); i++ ) {
            for( size_t j=0; j<dip.extent(1); j++ ) {
              for (size_t s=0; s<cbasis.extent(3); s++) {
                dvals(e,offsets(n,i)) += dip(e,j)*cbasis(e,i,j,s)*normals(e,j,s)*wts(e,j);
              }
            }
          }
        });
      }
      else if (btype == "HCURL"){
        // not implemented yet
      }
    }
  }
  return dvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT***,AssemblyDevice> BoundaryCell::getMass() {
  
  Kokkos::View<ScalarT***,AssemblyDevice> mass("local mass",numElem,
                                               LIDs.extent(1), LIDs.extent(1));
  
  Kokkos::View<int**,AssemblyDevice> bcs = wkset->var_bcs;
  Kokkos::View<int[1],AssemblyDevice> currind("current index");
  auto currind_host = Kokkos::create_mirror_view(currind);
  
  for (int n=0; n<numDOF.extent(0); n++) {
    if (bcs(n,sidenum) == 1) { // is this a strong DBC for this variable
      currind_host(0) = n;
      Kokkos::deep_copy(currind,currind_host);
      int bind = wkset->usebasis[n];
      
      DRV cbasis = basis[bind];
      std::string btype = cellData->basis_types[bind];
      
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bcell compute mass",RangePolicy<AssemblyExec>(0,mass.extent(0)), KOKKOS_LAMBDA (const int e ) {
          int n = currind(0);
          for( int i=0; i<numDOF(n); i++ ) {
            for( int j=0; j<numDOF(n); j++ ) {
              for( size_t k=0; k<cbasis.extent(2); k++ ) {
                mass(e,offsets(n,i),offsets(n,j)) += cbasis(e,i,k)*cbasis(e,j,k)*wts(e,k);
              }
            }
          }
        });
      }
      else if (btype == "HDIV"){
        parallel_for("bcell compute mass HDIV",RangePolicy<AssemblyExec>(0,mass.extent(0)), KOKKOS_LAMBDA (const int e ) {
          int n = currind(0);
          for( int i=0; i<numDOF(n); i++ ) {
            for( int j=0; j<numDOF(n); j++ ) {
              for( size_t k=0; k<cbasis.extent(2); k++ ) {
                for (size_t s=0; s<cbasis.extent(3); s++) {
                  mass(e,offsets(n,i),offsets(n,j)) += cbasis(e,i,k,s)*normals(e,k,s)*cbasis(e,j,k)*normals(e,k,s)*wts(e,k);
                }
              }
            }
          }
        });
      }
      else if (btype == "HCURL"){
        // not implemented yet
      }
    }
  }
  return mass;
}

