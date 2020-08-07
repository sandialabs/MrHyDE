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

#include <iostream>
#include <iterator>

cell::cell(const Teuchos::RCP<CellMetaData> & cellData_,
           const DRV nodes_,
           const Kokkos::View<LO*,AssemblyDevice> localID_,
           LIDView LIDs_,
           Kokkos::View<int****,HostDevice> sideinfo_,
           Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orientation_) :
cellData(cellData_), localElemID(localID_), nodes(nodes_), 
LIDs(LIDs_), sideinfo(sideinfo_), orientation(orientation_)
{
  
  LIDs_host = Kokkos::create_mirror_view(LIDs);
  Kokkos::deep_copy(LIDs_host,LIDs);
  
  numElem = nodes.extent(0);
  useSensors = false;
  
  {
    Teuchos::TimeMonitor localtimer(*buildBasisTimer);
    
    int dimension = cellData->dimension;
    int numip = cellData->ref_ip.extent(0);
    
    ip = DRV("ip", numElem, numip, dimension);
    CellTools::mapToPhysicalFrame(ip, cellData->ref_ip, nodes, *(cellData->cellTopo));
    
    DRV jacobian("jacobian", numElem, numip, dimension, dimension);
    CellTools::setJacobian(jacobian, cellData->ref_ip, nodes, *(cellData->cellTopo));
    
    DRV jacobianDet("determinant of jacobian", numElem, numip);
    DRV jacobianInv("inverse of jacobian", numElem, numip, dimension, dimension);
    CellTools::setJacobianDet(jacobianDet, jacobian);
    CellTools::setJacobianInv(jacobianInv, jacobian);
    
    wts = DRV("ip wts", numElem, numip);
    FuncTools::computeCellMeasure(wts, jacobianDet, cellData->ref_wts);
    
    hsize = Kokkos::View<ScalarT*,AssemblyDevice>("element sizes", numElem);
    parallel_for("cell hsize",RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int e ) {
      ScalarT vol = 0.0;
      for (int i=0; i<wts.extent(1); i++) {
        vol += wts(e,i);
      }
      ScalarT dimscl = 1.0/(ScalarT)ip.extent(2);
      hsize(e) = std::pow(vol,dimscl);
    });
    
    for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
      
      int numb = cellData->basis_pointers[i]->getCardinality();
      
      DRV basis_vals, basis_grad_vals, basis_div_vals, basis_curl_vals, basis_node_vals;
      
      if (cellData->basis_types[i] == "HGRAD"){
        basis_vals = DRV("basis",numElem,numb,numip);
        DRV tmp_basis_vals("basis",numElem,numb,numip);
        FuncTools::HGRADtransformVALUE(tmp_basis_vals, cellData->ref_basis[i]);
        OrientTools::modifyBasisByOrientation(basis_vals, tmp_basis_vals, orientation,
                                              cellData->basis_pointers[i].get());
        
        DRV basis_grad_tmp("basis grad tmp",numElem,numb,numip,dimension);
        basis_grad_vals = DRV("basis grad",numElem,numb,numip,dimension);
        FuncTools::HGRADtransformGRAD(basis_grad_tmp, jacobianInv, cellData->ref_basis_grad[i]);
        OrientTools::modifyBasisByOrientation(basis_grad_vals, basis_grad_tmp, orientation,
                                              cellData->basis_pointers[i].get());
      }
      else if (cellData->basis_types[i] == "HVOL"){
        basis_vals = DRV("basis",numElem,numb,numip);
        FuncTools::HGRADtransformVALUE(basis_vals, cellData->ref_basis[i]);
      }
      else if (cellData->basis_types[i] == "HDIV"){
        
        basis_vals = DRV("basis",numElem,numb,numip,dimension);
        DRV basis_tmp("basis tmp",numElem,numb,numip,dimension);
        FuncTools::HDIVtransformVALUE(basis_tmp, jacobian, jacobianDet, cellData->ref_basis[i]);
        OrientTools::modifyBasisByOrientation(basis_vals, basis_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        
        if (cellData->requireBasisAtNodes) {
          basis_node_vals = DRV("basis",numElem,numb,nodes.extent(1),dimension);
          DRV basis_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
          FuncTools::HDIVtransformVALUE(basis_tmp, jacobian, jacobianDet, cellData->ref_basis_nodes[i]);
          OrientTools::modifyBasisByOrientation(basis_node_vals, basis_tmp, orientation,
                                                cellData->basis_pointers[i].get());
        }
        
        basis_div_vals = DRV("basis div",numElem,numb,numip);
        DRV basis_div_vals_tmp("basis div tmp",numElem,numb,numip);
        FuncTools::HDIVtransformDIV(basis_div_vals_tmp, jacobianDet, cellData->ref_basis_div[i]);
        OrientTools::modifyBasisByOrientation(basis_div_vals, basis_div_vals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        
      }
      else if (cellData->basis_types[i] == "HCURL"){
        
        basis_vals = DRV("basis",numElem,numb,numip,dimension);
        DRV basis_tmp("basis tmp",numElem,numb,numip,dimension);
        FuncTools::HCURLtransformVALUE(basis_tmp, jacobianInv, cellData->ref_basis[i]);
        OrientTools::modifyBasisByOrientation(basis_vals, basis_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        
        if (cellData->requireBasisAtNodes) {
          basis_node_vals = DRV("basis",numElem,numb,nodes.extent(1),dimension);
          DRV basis_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
          FuncTools::HCURLtransformVALUE(basis_tmp, jacobianInv, cellData->ref_basis_nodes[i]);
          OrientTools::modifyBasisByOrientation(basis_node_vals, basis_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          
        }
        
        basis_curl_vals = DRV("basis curl",numElem,numb,numip,dimension);
        DRV basis_curl_vals_tmp("basis curl tmp",numElem,numb,numip,dimension);
        FuncTools::HCURLtransformCURL(basis_curl_vals_tmp, jacobian, jacobianDet, cellData->ref_basis_curl[i]);
        OrientTools::modifyBasisByOrientation(basis_curl_vals, basis_curl_vals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        
      }
      basis.push_back(basis_vals);
      basis_grad.push_back(basis_grad_vals);
      basis_div.push_back(basis_div_vals);
      basis_curl.push_back(basis_curl_vals);
      basis_nodes.push_back(basis_node_vals);
    }
  }
  
  if (cellData->build_face_terms) {
    Teuchos::TimeMonitor localtimer(*buildFaceBasisTimer);
    for (int side=0; side<cellData->numSides; side++) {
      DRV ref_ip = cellData->ref_side_ip[side];
      DRV ref_wts = cellData->ref_side_wts[side];
      
      int dimension = cellData->dimension;
      int numip = ref_ip.extent(0);
      
      // Step 1: fill in ip_side, wts_side and normals
      DRV sip("side ip", numElem, numip, dimension);
      DRV jac("bijac", numElem, numip, dimension, dimension);
      DRV jacDet("bijacDet", numElem, numip);
      DRV jacInv("bijacInv", numElem, numip, dimension, dimension);
      DRV swts = DRV("wts_side", numElem, numip);
      DRV snormals = DRV("normals", numElem, numip, dimension);
      DRV tangents("tangents", numElem, numip, dimension);
      
      {
        //Teuchos::TimeMonitor updatetimer(*worksetFaceUpdateIPTimer);
        
        CellTools::mapToPhysicalFrame(sip, ref_ip, nodes, *(cellData->cellTopo));
        CellTools::setJacobian(jac, ref_ip, nodes, *(cellData->cellTopo));
        CellTools::setJacobianInv(jacInv, jac);
        CellTools::setJacobianDet(jacDet, jac);
        
        if (dimension == 2) {
          DRV ref_tangents = cellData->ref_side_tangents[side];
          Intrepid2::RealSpaceTools<AssemblyExec>::matvec(tangents, jac, ref_tangents);
          
          DRV rotation("rotation matrix",dimension,dimension);
          rotation(0,0) = 0;  rotation(0,1) = 1;
          rotation(1,0) = -1; rotation(1,1) = 0;
          Intrepid2::RealSpaceTools<AssemblyExec>::matvec(snormals, rotation, tangents);
          
          Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(swts, tangents, Intrepid2::NORM_TWO);
          Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(swts, swts, ref_wts);
          
        }
        else if (dimension == 3) {
          
          DRV ref_tangentsU = cellData->ref_side_tangentsU[side];
          DRV ref_tangentsV = cellData->ref_side_tangentsV[side];
          
          DRV faceTanU("face tangent U", numElem, numip, dimension);
          DRV faceTanV("face tangent V", numElem, numip, dimension);
          
          Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanU, jac, ref_tangentsU);
          Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanV, jac, ref_tangentsV);
          
          Intrepid2::RealSpaceTools<AssemblyExec>::vecprod(snormals, faceTanU, faceTanV);
          
          Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(swts, snormals, Intrepid2::NORM_TWO);
          Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(swts, swts, ref_wts);
          
        }
        
        // scale the normal vector (we need unit normal...)
        parallel_for("cell normal unnecessary rescale",RangePolicy<AssemblyExec>(0,snormals.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (int j=0; j<snormals.extent(1); j++ ) {
            ScalarT normalLength = 0.0;
            for (int sd=0; sd<snormals.extent(2); sd++) {
              normalLength += snormals(e,j,sd)*snormals(e,j,sd);
            }
            normalLength = std::sqrt(normalLength);
            for (int sd=0; sd<snormals.extent(2); sd++) {
              snormals(e,j,sd) = snormals(e,j,sd) / normalLength;
            }
          }
        });
        
        ip_face.push_back(sip);
        wts_face.push_back(swts);
        normals_face.push_back(snormals);
      }
      
      // Step 2: define basis functionsat these integration points
      {
        //Teuchos::TimeMonitor updatetimer(*worksetFaceUpdateBasisTimer);
        vector<DRV> currbasis, currbasisgrad;
        for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
          int numb = cellData->basis_pointers[i]->getCardinality();
          DRV basis_vals, basis_grad_vals, basis_div_vals, basis_curl_vals;
          
          DRV ref_basis_vals = cellData->ref_side_basis[side][i];
          
          if (cellData->basis_types[i] == "HGRAD"){
            
            DRV basis_vals_tmp("tmp basis_vals",numElem, numb, numip);
            basis_vals = DRV("basis_vals",numElem, numb, numip);
            FuncTools::HGRADtransformVALUE(basis_vals_tmp, ref_basis_vals);
            OrientTools::modifyBasisByOrientation(basis_vals, basis_vals_tmp, orientation,
                                                  cellData->basis_pointers[i].get());
          
            DRV ref_basis_grad_vals = cellData->ref_side_basis_grad[side][i];
            DRV basis_grad_vals_tmp("tmp basis_grad_vals",numElem, numb, numip, dimension);
            basis_grad_vals = DRV("basis_grad_vals",numElem, numb, numip, dimension);
            FuncTools::HGRADtransformGRAD(basis_grad_vals_tmp, jacInv, ref_basis_grad_vals);
            OrientTools::modifyBasisByOrientation(basis_grad_vals, basis_grad_vals_tmp, orientation,
                                                  cellData->basis_pointers[i].get());
            
          }
          else if (cellData->basis_types[i] == "HVOL"){
            
            basis_vals = DRV("basis_vals",numElem, numb, numip);
            FuncTools::HGRADtransformVALUE(basis_vals, ref_basis_vals);
            
          }
          else if (cellData->basis_types[i] == "HDIV"){
            
            DRV basis_vals_tmp("tmp basis_vals",numElem, numb, numip, dimension);
            basis_vals = DRV("basis_vals",numElem, numb, numip, dimension);
            
            FuncTools::HDIVtransformVALUE(basis_vals_tmp, jac, jacDet, ref_basis_vals);
            OrientTools::modifyBasisByOrientation(basis_vals, basis_vals_tmp, orientation,
                                                  cellData->basis_pointers[i].get());
            
          }
          else if (cellData->basis_types[i] == "HCURL"){
            //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
          }
          else if (cellData->basis_types[i] == "HFACE"){
            
            basis_vals = DRV("basis_vals",numElem, numb, numip);
            DRV basis_vals_tmp("basisvals_Transformed",numElem, numb, numip);
            FuncTools::HGRADtransformVALUE(basis_vals_tmp, ref_basis_vals);
            OrientTools::modifyBasisByOrientation(basis_vals, basis_vals_tmp, orientation,
                                                  cellData->basis_pointers[i].get());
            
          }
          
          currbasis.push_back(basis_vals);
          currbasisgrad.push_back(basis_grad_vals);
        }
        basis_face.push_back(currbasis);
        basis_grad_face.push_back(currbasisgrad);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::setWorkset(Teuchos::RCP<workset> & wkset_) {
  
  wkset = wkset_;

  // Frequently used Views 
  res_AD = wkset->res;
  offsets = wkset->offsets;
  //
  
  numDOF = cellData->numDOF;
  //
  numAuxDOF = cellData->numAuxDOF;

}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::setParams(LIDView paramLIDs_) {
  
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

void cell::setUseBasis(vector<int> & usebasis_, const int & numsteps, const int & numstages) {
  vector<int> usebasis = usebasis_;
  //num_stages = nstages;
  
  // Set up the containers for usual solution storage
  size_t maxnbasis = 0;
  for (size_t i=0; i<cellData->numDOF_host.extent(0); i++) {
    if (cellData->numDOF_host(i) > maxnbasis) {
      maxnbasis = cellData->numDOF_host(i);
    }
  }
  //maxnbasis *= nstages;
  u = Kokkos::View<ScalarT***,AssemblyDevice>("u",numElem,cellData->numDOF.extent(0),maxnbasis);
  phi = Kokkos::View<ScalarT***,AssemblyDevice>("phi",numElem,cellData->numDOF.extent(0),maxnbasis);
  
  // This does add a little extra un-used memory for steady-state problems, but not a concern
  u_prev = Kokkos::View<ScalarT****,AssemblyDevice>("u previous",numElem,cellData->numDOF.extent(0),maxnbasis,numsteps);
  phi_prev = Kokkos::View<ScalarT****,AssemblyDevice>("phi previous",numElem,cellData->numDOF.extent(0),maxnbasis,numsteps);
  
  u_stage = Kokkos::View<ScalarT****,AssemblyDevice>("u stages",numElem,cellData->numDOF.extent(0),maxnbasis,numstages);
  phi_stage = Kokkos::View<ScalarT****,AssemblyDevice>("phi stages",numElem,cellData->numDOF.extent(0),maxnbasis,numstages);
 
  u_avg = Kokkos::View<ScalarT***,AssemblyDevice>("u spatial average",numElem,cellData->numDOF.extent(0),cellData->dimension);
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  //wkset->paramusebasis = pusebasis_;
  
  size_t maxnbasis = 0;
  for (size_t i=0; i<cellData->numParamDOF.extent(0); i++) {
    if (cellData->numParamDOF(i) > maxnbasis) {
      maxnbasis = cellData->numParamDOF(i);
    }
  }
  param = Kokkos::View<ScalarT***,AssemblyDevice>("param",numElem,cellData->numParamDOF.extent(0),maxnbasis);
  param_avg = Kokkos::View<ScalarT**,AssemblyDevice>("param",numElem,cellData->numParamDOF.extent(0));
  
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
// Update the workset
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateWorksetBasis() {
  wkset->ip = ip;
  wkset->wts = wts;
  wkset->h = hsize;
  
  // TMW: can this be avoided??
  Kokkos::deep_copy(wkset->ip_KV,ip);
  
  wkset->basis = basis;
  wkset->basis_grad = basis_grad;
  wkset->basis_div = basis_div;
  wkset->basis_curl = basis_curl;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolnVolIP() {
  // seedwhat key: 0-nothing; 1-sol; 2-soldot; 3-disc.params.; 4-aux.vars
  // Note: seeding u_dot is now deprecated
  
  Teuchos::TimeMonitor localtimer(*computeSolnVolTimer);
  //wkset->update(ip,wts,jacobian,jacobianInv,jacobianDet,orientation);
  this->updateWorksetBasis();
  wkset->computeSolnVolIP();
  
  //wkset->computeParamVolIP(param, seedwhat);
  if (cellData->compute_sol_avg) {
    this->computeSolAvg();
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolAvg() {
  
  // THIS FUNCTION ASSUMES THAT THE WORKSET BASIS HAS BEEN UPDATED
  // AND THE SOLUTION HAS BEEN COMPUTED AT THE VOLUMETRIC IP
  
  Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  
  Kokkos::View<AD****,AssemblyDevice> sol = wkset->local_soln;
  
  parallel_for("cell sol avg",RangePolicy<AssemblyExec>(0,u_avg.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    ScalarT avgwt = 0.0;
    for (int pt=0; pt<wts.extent(1); pt++) {
      avgwt += wts(elem,pt);
    }
    for (int dof=0; dof<sol.extent(1); dof++) {
      for (int dim=0; dim<sol.extent(3); dim++) {
        ScalarT solavg = 0.0;
        for (int pt=0; pt<sol.extent(2); pt++) {
          solavg += sol(elem,dof,pt,dim).val()*wts(elem,pt);
        }
        u_avg(elem,dof,dim) = solavg/avgwt;
      }
    }
  });
  
  if (param_avg.extent(1) > 0) {
    Kokkos::View<AD***,AssemblyDevice> psol = wkset->local_param;
    
    parallel_for("cell param avg",RangePolicy<AssemblyExec>(0,param_avg.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      ScalarT avgwt = 0.0;
      for (int pt=0; pt<wts.extent(1); pt++) {
        avgwt += wts(elem,pt);
      }
      for (int dof=0; dof<psol.extent(1); dof++) {
        ScalarT solavg = 0.0;
        for (int pt=0; pt<psol.extent(2); pt++) {
          solavg += psol(elem,dof,pt).val()*wts(elem,pt);
        }
        param_avg(elem,dof) = solavg/avgwt;
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the workset
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateWorksetFaceBasis(const size_t & facenum) {
  
  wkset->ip_side = ip_face[facenum];
  wkset->wts_side = wts_face[facenum];
  wkset->normals = normals_face[facenum];
  wkset->h = hsize;
  
  // TMW: can this be avoided??
  Kokkos::deep_copy(wkset->ip_side_KV,ip_face[facenum]);
  Kokkos::deep_copy(wkset->normals_KV,normals_face[facenum]);
  
  wkset->basis_face = basis_face[facenum];
  wkset->basis_grad_face = basis_grad_face[facenum];
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolnFaceIP(const size_t & facenum) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnFaceTimer);
  
  this->updateWorksetFaceBasis(facenum);
  //wkset->updateFace(nodes, orientation, facenum);
  wkset->computeSolnFaceIP();
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous step solutions
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetPrevSoln() {
  
  // shift previous step solns
  if (u_prev.extent(3)>1) {
    parallel_for("cell shift prev soln",RangePolicy<AssemblyExec>(0,u_prev.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int i=0; i<u_prev.extent(1); i++) {
        for (int j=0; j<u_prev.extent(2); j++) {
          for (int s=u_prev.extent(3)-1; s>0; s--) {
            u_prev(e,i,j,s) = u_prev(e,i,j,s-1);
          }
        }
      }
    });
  }
  
  // copy current u into first step
  parallel_for("cell copy prev soln",RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int i=0; i<u.extent(1); i++) {
      for (int j=0; j<u.extent(2); j++) {
        u_prev(e,i,j,0) = u(e,i,j);
      }
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetStageSoln() {
  
  parallel_for("cell reset stage soln",RangePolicy<AssemblyExec>(0,u_stage.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int i=0; i<u_stage.extent(1); i++) {
      for (int j=0; j<u_stage.extent(2); j++) {
        for (int k=0; k<u_stage.extent(3); k++) {
          u_stage(e,i,j,k) = u(e,i,j);
        }
      }
    }
  });
  //KokkosTools::print(u_stage);
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateStageSoln() {
  
  
  // add u into the current stage soln (done after stage solution is computed)
  Kokkos::View<int*,UnifiedDevice> snum = wkset->current_stage_KV;
  parallel_for("cell update stage soln",RangePolicy<AssemblyExec>(0,u_stage.extent(0)), KOKKOS_LAMBDA (const int e ) {
    int stage = snum(0);
    for (int i=0; i<u_stage.extent(1); i++) {
      for (int j=0; j<u_stage.extent(2); j++) {
        u_stage(e,i,j,stage) = u(e,i,j);
      }
    }
  });
  //KokkosTools::print(u_stage);
  
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
                         const bool & assemble_volume_terms,
                         const bool & assemble_face_terms) {
  
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
  
  if (!(cellData->multiscale)) {
    if (isTransient) {
      wkset->computeSolnTransientSeeded(u, u_prev, u_stage, seedwhat);
    }
    else { // steady-state
      wkset->computeSolnSteadySeeded(u, seedwhat);
    }
    
    this->computeSolnVolIP();
    wkset->computeParamVolIP(param, seedwhat);
    
  }
  
  //////////////////////////////////////////////////////////////
  // Compute res and J=dF/du
  //////////////////////////////////////////////////////////////
  
  // Volumetric contribution
  if (assemble_volume_terms) {
    Teuchos::TimeMonitor localtimer(*volumeResidualTimer);
    if (cellData->multiscale) {
      int sgindex = subgrid_model_index[subgrid_model_index.size()-1];
      subgridModels[sgindex]->subgridSolver(u, phi, wkset->time, isTransient, isAdjoint,
                                            compute_jacobian, compute_sens, num_active_params,
                                            compute_disc_sens, compute_aux_sens,
                                            *wkset, subgrid_usernum, 0,
                                            subgradient, store_adjPrev);
      fixJacDiag = true;
    }
    else {
      cellData->physics_RCP->volumeResidual(cellData->myBlock);
    }
  }
  
  // Edge/face contribution
  if (assemble_face_terms) {
    Teuchos::TimeMonitor localtimer(*faceResidualTimer);
    if (cellData->multiscale) {
      // do nothing
    }
    else {
      for (size_t s=0; s<cellData->numSides; s++) {
        this->computeSolnFaceIP(s);
        cellData->physics_RCP->faceResidual(cellData->myBlock);
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
    this->fixDiagJac(local_J, local_res);
  }
  
  
  // Update the local residual
  {
    Teuchos::TimeMonitor localtimer(*residualFillTimer);
    if (isAdjoint) {
      this->updateAdjointRes(compute_sens, local_res);
    }
    else {
      this->updateRes(compute_sens, local_res);
    }
  }
  
  {
    if (isAdjoint) {
      Teuchos::TimeMonitor localtimer(*adjointResidualTimer);
      this->updateAdjointRes(compute_jacobian, isTransient,
                             compute_aux_sens, store_adjPrev,
                             local_J, local_res);
      
      
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateRes(const bool & compute_sens, Kokkos::View<ScalarT***,UnifiedDevice> local_res) {
  
  if (compute_sens) {
    parallel_for("cell update res sens",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int r=0; r<local_res.extent(2); r++) {
        for (int n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),r) -= res_AD(e,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
  }
  else {
    parallel_for("cell update res",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<numDOF.extent(0); n++) {
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

void cell::updateAdjointRes(const bool & compute_sens, Kokkos::View<ScalarT***,UnifiedDevice> local_res) {
  Kokkos::View<AD**,AssemblyDevice> adjres_AD = wkset->adjrhs;
  
  if (compute_sens) {
    parallel_for("cell update adjoint res sens",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
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
    parallel_for("cell update adjoint res",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          local_res(e,offsets(n,j),0) -= adjres_AD(e,offsets(n,j)).val();
        }
      }
    });
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAdjointRes(const bool & compute_jacobian, const bool & isTransient,
                            const bool & compute_aux_sens, const bool & store_adjPrev,
                            Kokkos::View<ScalarT***,UnifiedDevice> local_J,
                            Kokkos::View<ScalarT***,UnifiedDevice> local_res) {
  
  // Update residual (adjoint mode)
  // Adjoint residual: -dobj/du - J^T * phi + 1/dt*M^T * phi_prev
  // J = 1/dtM + A
  // adj_prev stores 1/dt*M^T * phi_prev where M is evaluated at appropriate time
  
  // TMW: This will not work on a GPU
  
  if (!(cellData->mortar_objective)) {
    for (int w=1; w < cellData->dimension+2; w++) {
      
      Kokkos::View<AD**,AssemblyDevice> obj = computeObjective(wkset->time, 0, w);
      
      int numDerivs;
      if (useSensors) {
        if (numSensors > 0) {
          
          for (int s=0; s<numSensors; s++) {
            int e = sensorElem[s];
            auto cobj = Kokkos::subview(obj,Kokkos::ALL(), s);
            for (int n=0; n<numDOF.extent(0); n++) {
              auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
              Kokkos::View<int[2],AssemblyDevice> scratch("scratch pad");
              auto scratch_host = Kokkos::create_mirror_view(scratch);
              scratch_host(0) = n;
              scratch_host(1) = e;
              Kokkos::deep_copy(scratch,scratch_host);
              auto sres = Kokkos::subview(local_res,e,Kokkos::ALL(),0);
              if (w == 1) {
                auto sbasis = Kokkos::subview(sensorBasis[s][wkset->usebasis[n]],0,Kokkos::ALL(),s);
                parallel_for("cell adjust adjoint res sensor",RangePolicy<AssemblyExec>(0,cellData->numDOF_host(n)), KOKKOS_LAMBDA (const int j ) {
                  int nn = scratch(0);
                  int elem = scratch(1);
                  for (int i=0; i<numDOF(nn); i++) {
                    sres(off(j)) += -cobj(elem).fastAccessDx(off(i))*sbasis(j);
                  }
                });
              }
              else {
                auto sbasis = Kokkos::subview(sensorBasisGrad[s][wkset->usebasis[n]],0,Kokkos::ALL(),s,w-2);
                parallel_for("cell adjust adjoint res sensor grad", RangePolicy<AssemblyExec>(0,cellData->numDOF_host(n)), KOKKOS_LAMBDA (const int j ) {
                  int nn = scratch(0);
                  int elem = scratch(1);
                  for (int i=0; i<numDOF(nn); i++) {
                    sres(off(j)) += -cobj(elem).fastAccessDx(off(i))*sbasis(j);
                  }
                });
              }
            }
          }
        }
      }
      else {
        for (int n=0; n<numDOF.extent(0); n++) {
          Kokkos::View<int[2],AssemblyDevice> scratch("scratch pad");
          auto scratch_host = Kokkos::create_mirror_view(scratch);
          scratch_host(0) = n;
          Kokkos::deep_copy(scratch,scratch_host);
          if (w==1) {
            DRV sbasis = basis[wkset->usebasis[n]];
            parallel_for("cell adjust adjoint res",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
              int nn = scratch(0);
              for (int j=0; j<numDOF(nn); j++) {
                for (int i=0; i<numDOF(nn); i++) {
                  for (int s=0; s<sbasis.extent(2); s++) {
                    local_res(e,offsets(nn,j),0) += -obj(e,s).fastAccessDx(offsets(nn,i))*sbasis(e,j,s);
                  }
                }
              }
            });
          }
          else {
            auto sbasis = Kokkos::subview(basis_grad[wkset->usebasis[n]],Kokkos::ALL(),
                                          Kokkos::ALL(), Kokkos::ALL(), w-2);
            parallel_for("cell adjust adjoint res grad",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
              int nn = scratch(0);
              for (int j=0; j<numDOF(nn); j++) {
                for (int i=0; i<numDOF(nn); i++) {
                  for (int s=0; s<sbasis.extent(2); s++) {
                    local_res(e,offsets(nn,j),0) += -obj(e,s).fastAccessDx(offsets(nn,i))*sbasis(e,j,s);
                  }
                }
              }
            });
          }
        }
      }
    }
  }
  if (compute_jacobian) {
    parallel_for("cell adjust adjoint jac",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_res(e,offsets(n,j),0) += -local_J(e,offsets(n,j),offsets(m,k))*phi(e,m,k);
            }
          }
        }
      }
    });
    if (isTransient) {
      int seedwhat = 2;
      //this->computeSolnVolIP(seedwhat);
      wkset->computeSolnTransientSeeded(u, u_prev, u_stage, seedwhat);
      wkset->computeParamVolIP(param, seedwhat);
      this->computeSolnVolIP();
      
      wkset->resetResidual();
      
      cellData->physics_RCP->volumeResidual(cellData->myBlock);
      Kokkos::View<ScalarT***,AssemblyDevice> Jdot("temporary fix for transient adjoint",
                                                   local_J.extent(0), local_J.extent(1), local_J.extent(2));
      this->updateJac(true, Jdot);
      Kokkos::View<ScalarT[1],AssemblyDevice> dscratch("double scratch pad");
      Kokkos::View<bool[2],AssemblyDevice> bscratch("bool scratch pad");
      auto dscratch_host = Kokkos::create_mirror_view(dscratch);
      auto bscratch_host = Kokkos::create_mirror_view(bscratch);
      dscratch_host(0) = wkset->alpha;
      bscratch_host(0) = compute_aux_sens;
      bscratch_host(1) = store_adjPrev;
      Kokkos::deep_copy(dscratch,dscratch_host);
      Kokkos::deep_copy(bscratch,bscratch_host);
      parallel_for("cell adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            ScalarT aPrev = 0.0;
            for (int m=0; m<numDOF.extent(0); m++) {
              for (int k=0; k<numDOF(m); k++) {
                aPrev += dscratch(0)*Jdot(e,offsets(n,j),offsets(m,k))*phi(e,m,k);
              }
            }
            local_res(e,offsets(n,j),0) += adjPrev(e,offsets(n,j));
            if (!bscratch(0) && bscratch(1)) {
              adjPrev(e,offsets(n,j)) = aPrev;
            }
          }
        }
      });
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,UnifiedDevice> local_J) {
  
  if (useadjoint) {
    parallel_for("cell update jac adj",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(e,offsets(m,k),offsets(n,j)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("cell update jac",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(e,offsets(n,j),offsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  AssemblyExec::execution_space().fence();
}

///////////////////////////////////////////////////////////////////////////////////////
// Place ones on the diagonal of the Jacobian if
///////////////////////////////////////////////////////////////////////////////////////

void cell::fixDiagJac(Kokkos::View<ScalarT***,UnifiedDevice> local_J,
                      Kokkos::View<ScalarT***,UnifiedDevice> local_res) {
  
  ScalarT JTOL = 1.0E-14;
  
  parallel_for("cell fix diag",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (size_t var=0; var<offsets.extent(0); var++) {
      for (size_t dof=0; dof<numDOF(var); dof++) {
        int diag = offsets(var,dof);
        if (abs(local_J(elem,diag,diag)) < JTOL) {
          local_res(elem,diag,0) = -u(elem,var,dof);
          for (size_t j=0; j<numDOF(var); j++) {
            ScalarT scale = 1.0/((ScalarT)numDOF(var)-1.0);
            local_J(elem,diag,offsets(var,j)) = -scale;
            if (j!=dof)
              local_res(elem,diag,0) += scale*u(elem,var,j);
          }
          local_J(elem,diag,diag) = 1.0;
        }
      }
    }
  });
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateParamJac(Kokkos::View<ScalarT***,UnifiedDevice> local_J) {
  paramoffsets = wkset->paramoffsets;
  numParamDOF = cellData->numParamDOF;
  
  parallel_for("cell update param jac",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<numDOF.extent(0); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (int m=0; m<numParamDOF.extent(0); m++) {
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

void cell::updateAuxJac(Kokkos::View<ScalarT***,UnifiedDevice> local_J) {
  
  parallel_for("cell update aux jac",RangePolicy<AssemblyExec>(0,local_J.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<numDOF.extent(0); n++) {
      for (int j=0; j<numDOF(n); j++) {
        for (int m=0; m<numAuxDOF.extent(0); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(e,offsets(n,j),auxoffsets(m,k)) += res_AD(e,offsets(n,j)).fastAccessDx(auxoffsets(m,k));
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
  Kokkos::View<ScalarT**,AssemblyDevice> initialvals("initial values",numElem,LIDs.extent(1));
  this->updateWorksetBasis();
  Kokkos::View<int[1],UnifiedDevice> iscratch("current index");
  auto iscratch_host = Kokkos::create_mirror_view(iscratch);
  if (project) { // works for any basis
    Kokkos::View<ScalarT***,AssemblyDevice> initialip = cellData->physics_RCP->getInitial(wkset->ip,
                                                                                          cellData->myBlock,
                                                                                          project,
                                                                                          wkset);
    for (int n=0; n<numDOF.extent(0); n++) {
      DRV cbasis = basis[wkset->usebasis[n]];
      iscratch_host(0) = n;
      Kokkos::deep_copy(iscratch,iscratch_host);
      parallel_for("cell get init",RangePolicy<AssemblyExec>(0,initialvals.extent(0)), KOKKOS_LAMBDA (const int e ) {
        int nn = iscratch(0);
        for( int i=0; i<numDOF(nn); i++ ) {
          for( size_t j=0; j<initialip.extent(2); j++ ) {
            initialvals(e,offsets(nn,i)) += initialip(e,nn,j)*cbasis(e,i,j)*wts(e,j);
          }
        }
      });
    }
  }
  else { // only works if using HGRAD linear basis
    
    Kokkos::View<ScalarT***,AssemblyDevice> initialnodes = cellData->physics_RCP->getInitial(nodes,
                                                                                             cellData->myBlock,
                                                                                             project,
                                                                                             wkset);
    for (int n=0; n<numDOF.extent(0); n++) {
      auto off = Kokkos::subview( offsets, n, Kokkos::ALL());
      iscratch_host(0) = n;
      Kokkos::deep_copy(iscratch,iscratch_host);
      parallel_for("cell get init interp",RangePolicy<AssemblyExec>(0,initialnodes.extent(0)), KOKKOS_LAMBDA (const int e ) {
        int nn = iscratch(0);
        for( int i=0; i<numDOF(nn); i++ ) {
          initialvals(e,off(i)) = initialnodes(e,nn,i);
        }
      });
    }
  }
  return initialvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT***,AssemblyDevice> cell::getMass() {
  Kokkos::View<ScalarT***,AssemblyDevice> mass("local mass",numElem,
                                               LIDs.extent(1),
                                               LIDs.extent(1));
  Kokkos::View<int[1],UnifiedDevice> iscratch("current index");
  auto iscratch_host = Kokkos::create_mirror_view(iscratch);
  
  for (int n=0; n<numDOF.extent(0); n++) {
    DRV cbasis = basis[wkset->usebasis[n]];
    iscratch_host(0) = n;
    Kokkos::deep_copy(iscratch,iscratch_host);
    parallel_for("cell get mass",RangePolicy<AssemblyExec>(0,mass.extent(0)), KOKKOS_LAMBDA (const int e ) {
      int nn  = iscratch(0);
      for( int i=0; i<numDOF(nn); i++ ) {
        for( int j=0; j<numDOF(nn); j++ ) {
          for( size_t k=0; k<cbasis.extent(2); k++ ) {
            mass(e,offsets(nn,i),offsets(nn,j)) += cbasis(e,i,k)*cbasis(e,j,k)*wts(e,k);
          }
        }
      }
    });
  }
  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the response at the integration points given the solution and solve time
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<AD***,AssemblyDevice> cell::computeResponse(const int & seedwhat) {
  
  // Assumes that u has already been filled
  
  // seedwhat indicates what needs to be seeded
  // seedwhat == 0 => seed nothing
  // seedwhat == 1 => seed sol
  // seedwhat == j (j>1) => seed (j-1)-derivative of sol
  
  paramoffsets = wkset->paramoffsets;
  numParamDOF = cellData->numParamDOF;
  
  Kokkos::View<AD***,AssemblyDevice> response;
  bool useSensors = false;
  if (cellData->response_type == "pointwise") {
    useSensors = true;
  }
  
  size_t numip = wkset->ip.extent(1);
  if (useSensors) {
    numip = sensorLocations.size();
  }
  
  
  if (numip > 0) {
    
    this->updateWorksetBasis();
    
    // Extract the local solution at this time
    // We automatically seed the AD and adjust it below
    Kokkos::View<AD***,AssemblyDevice> u_dof("u_dof",numElem,numDOF.extent(0),LIDs.extent(1));
    parallel_for("cell response get u",RangePolicy<AssemblyExec>(0,u_dof.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<numDOF.extent(0); n++) {
        for( int i=0; i<numDOF(n); i++ ) {
          u_dof(e,n,i) = AD(maxDerivs,offsets(n,i),u(e,n,i));
        }
      }
    });
    
    // Map the local solution to the solution and gradient at ip
    Kokkos::View<AD****,AssemblyDevice> u_ip("u_ip",numElem,numDOF.extent(0),
                                             numip,cellData->dimension);
    Kokkos::View<AD****,AssemblyDevice> ugrad_ip("ugrad_ip",numElem,numDOF.extent(0),
                                                 numip,cellData->dimension);
    
    Kokkos::View<int*,AssemblyDevice> iscratch("int scratch",sensorElem.size());
    auto iscratch_host = Kokkos::create_mirror_view(iscratch);
    for (size_t i=0; i<sensorElem.size(); i++) {
      iscratch_host(i) = sensorElem[i];
    }
    Kokkos::deep_copy(iscratch,iscratch_host);
    
    // Need to rewrite this using useSensors on outside
    
    parallel_for(RangePolicy<AssemblyExec>(0,u_ip.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<numDOF.extent(0); n++) {
        for( int i=0; i<numDOF(n); i++ ) {
          if (useSensors) {
            for (int ee=0; ee<iscratch.extent(0); ee++) {
              int eind = iscratch(ee);
              if (eind == e) {
                u_ip(eind,n,ee,0) += u_dof(eind,n,i)*sensorBasis[ee][wkset->usebasis[n]](0,i,0);
              }
            }
          }
          else {
            for( size_t j=0; j<numip; j++ ) {
              u_ip(e,n,j,0) += u_dof(e,n,i)*wkset->basis[wkset->usebasis[n]](e,i,j);
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
                ugrad_ip(e,n,j,s) += u_dof(e,n,i)*wkset->basis_grad[wkset->usebasis[n]](e,i,j,s);
              }
            }
          }
        }
      }
    });
    
    // Adjust the AD based on seedwhat
    if (seedwhat == 0) { // remove all seeding
      parallel_for(RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int n=0; n<numDOF.extent(0); n++) {
          for( size_t j=0; j<numip; j++ ) {
            u_ip(e,n,j,0) = u_ip(e,n,j,0).val();
            for (int s=0; s<ugrad_ip.extent(3); s++) {
              ugrad_ip(e,n,j,s) = ugrad_ip(e,n,j,s).val();
            }
          }
        }
      });
    }
    else if (seedwhat == 1) { // remove seeding on gradient
      parallel_for(RangePolicy<AssemblyExec>(0,ugrad_ip.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int n=0; n<numDOF.extent(0); n++) {
          for( size_t j=0; j<numip; j++ ) {
            for (int s=0; s<ugrad_ip.extent(3); s++) {
              ugrad_ip(e,n,j,s) = ugrad_ip(e,n,j,s).val();
            }
          }
        }
      });
      
    }
    else {
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<numDOF.extent(0); n++) {
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
    
    if (numParamDOF.extent(0) > 0) {
      // Extract the local solution at this time
      // We automatically seed the AD and adjust it below
      Kokkos::View<AD***,AssemblyDevice> param_dof("param dof",numElem,numParamDOF.extent(0),paramLIDs.extent(1));
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<numParamDOF.extent(0); n++) {
          for( int i=0; i<numParamDOF(n); i++ ) {
            param_dof(e,n,i) = AD(maxDerivs,paramoffsets(n,i),param(e,n,i));
          }
        }
      }
      
      // Map the local solution to the solution and gradient at ip
      param_ip = Kokkos::View<AD****,AssemblyDevice>("u_ip",numElem,numParamDOF.extent(0),
                                                     numip,cellData->dimension);
      paramgrad_ip = Kokkos::View<AD****,AssemblyDevice>("ugrad_ip",numElem,numParamDOF.extent(0),
                                                         numip,cellData->dimension);
      for (int e=0; e<numElem; e++) {
        for (int n=0; n<numParamDOF.extent(0); n++) {
          for( int i=0; i<numParamDOF(n); i++ ) {
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
                param_ip(e,n,j,0) += param_dof(e,n,i)*basis[wkset->paramusebasis[n]](e,i,j);
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
                  paramgrad_ip(e,n,j,s) += param_dof(e,n,i)*basis_grad[wkset->paramusebasis[n]](e,i,j,s);
                }
              }
            }
          }
        }
      }
      
      // Adjust the AD based on seedwhat
      if (seedwhat == 0) { // remove seeding on grad
        for (int e=0; e<numElem; e++) {
          for (int n=0; n<numParamDOF.extent(0); n++) {
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
          for (int n=0; n<numParamDOF.extent(0); n++) {
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
                                                      wkset->time, wkset);
      }
    }
    else {
      response = cellData->physics_RCP->getResponse(cellData->myBlock, u_ip, ugrad_ip, param_ip,
                                                    paramgrad_ip, wkset->ip,
                                                    wkset->time, wkset);
    }
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
    
    //Kokkos::View<AD***,AssemblyDevice> responsevals = computeResponse(solvetime,tindex,seedwhat);
    Kokkos::View<AD***,AssemblyDevice> responsevals = computeResponse(seedwhat);
    
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
          for (size_t k=0; k<ip.extent(1); k++) {
            AD diff = responsevals(e,r,k)-ctarg(e,r,k);
            if(cellData->compute_diff) {
              objective(e,k) += 0.5*wkset->deltat*cweight(e,r,k)*(diff)*(diff)*wkset->wts(e,k);
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
    
    int sgindex = subgrid_model_index[tindex];
    Kokkos::View<AD*,AssemblyDevice> cobj = subgridModels[sgindex]->computeObjective(cellData->response_type,seedwhat,
                                                                                     solvetime,subgrid_usernum);
    
    objective = Kokkos::View<AD**,AssemblyDevice>("objective",numElem,cobj.extent(0));
    for (int i=0; i<cobj.extent(0); i++) {
      objective(0,i) += cobj(i); // TMW: tempory fix
    }
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
  int numip = wkset->numip;
  this->updateWorksetBasis();
  wkset->computeParamVolIP(param, 3);
  
  Kokkos::View<AD[1],AssemblyDevice> adscratch("scratch for AD");
  auto adscratch_host = Kokkos::create_mirror_view(adscratch);
  
  Kokkos::View<int[2],AssemblyDevice> iscratch("scratch for ints");
  auto iscratch_host = Kokkos::create_mirror_view(iscratch);
  
  Kokkos::View<ScalarT[2],AssemblyDevice> dscratch("scratch for ScalarT");
  auto dscratch_host = Kokkos::create_mirror_view(dscratch);
  
  int numParams = reg_indices.size();
  ScalarT reg_offset = 1.0e-5;
  Kokkos::View<AD***,AssemblyDevice> par = wkset->local_param;
  Kokkos::View<AD****,AssemblyDevice> par_grad = wkset->local_param_grad;
  for (int i = 0; i < numParams; i++) {
    dscratch_host(0) = reg_constants[i];
    dscratch_host(1) = reg_offset;
    iscratch_host(0) = reg_types[i];
    iscratch_host(1) = reg_indices[i];
    Kokkos::deep_copy(dscratch,dscratch_host);
    Kokkos::deep_copy(iscratch,iscratch_host);
    parallel_for("cell domain reg",RangePolicy<AssemblyExec>(0,par.extent(0)), KOKKOS_LAMBDA (const int e ) {
      int pindex = iscratch(1);
      int rtype = iscratch(0);
      ScalarT reg_const = dscratch(0);
      ScalarT reg_off = dscratch(1);
      for (int k = 0; k < par.extent(2); k++) {
        AD p = par(e,pindex,k);
        // L2
        if (rtype == 0) {
          adscratch(0) += 0.5*reg_const*p*p*wts(e,k);
        }
        else {
          AD dpdx = par_grad(e,pindex,k,0);
          AD dpdy = 0.0;
          AD dpdz = 0.0;
          if (par_grad.extent(3) > 1)
            dpdy = par_grad(e,pindex,k,1);
          if (par_grad.extent(3) > 2)
            dpdz = par_grad(e,pindex,k,2);
          // H1
          if (rtype == 1) {
            adscratch(0) += 0.5*reg_const*(dpdx*dpdx + dpdy*dpdy + dpdz*dpdz)*wts(e,k);
          }
          // TV
          else if (rtype == 2) {
            adscratch(0) += reg_const*sqrt(dpdx*dpdx + dpdy*dpdy + dpdz*dpdz + reg_off*reg_off)*wts(e,k);
          }
        }
      }
    });
  }
  Kokkos::deep_copy(adscratch_host,adscratch);
  return adscratch_host(0);
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
                      Teuchos::RCP<discretization> & disc,
                      const vector<basis_RCP> & basis_pointers,
                      const vector<basis_RCP> & param_basis_pointers) {
  
  
  // If we have sensors, then we set the response type to pointwise
  cellData->response_type = "pointwise";
  useSensors = true;
  bool useFineScale = true;
  if (!(cellData->multiscale) || cellData->mortar_objective) {
    useFineScale = false;
  }
  
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
        csensorBasis.push_back(disc->evaluateBasis(basis_pointers[b], refsenspts, orientation));
        csensorBasisGrad.push_back(disc->evaluateBasisGrads(basis_pointers[b], nodes, refsenspts,
                                                            cellData->cellTopo, orientation));
      }
      
      sensorBasis.push_back(csensorBasis);
      sensorBasisGrad.push_back(csensorBasisGrad);
      
      
      vector<DRV> cpsensorBasis;
      vector<DRV> cpsensorBasisGrad;
      
      for (size_t b=0; b<param_basis_pointers.size(); b++) {
        cpsensorBasis.push_back(disc->evaluateBasis(param_basis_pointers[b], refsenspts, orientation));
        cpsensorBasisGrad.push_back(disc->evaluateBasisGrads(param_basis_pointers[b], nodes,
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
            csensorBasis.push_back(disc->evaluateBasis(basis_pointers[b], refsenspts, orientation));
            csensorBasisGrad.push_back(disc->evaluateBasisGrads(basis_pointers[b], cnodes,
                                                                refsenspts, cellData->cellTopo, orientation));
          }
          sensorBasis.push_back(csensorBasis);
          sensorBasisGrad.push_back(csensorBasisGrad);
          
          
          vector<DRV> cpsensorBasis;
          vector<DRV> cpsensorBasisGrad;
          
          for (size_t b=0; b<param_basis_pointers.size(); b++) {
            cpsensorBasis.push_back(disc->evaluateBasis(param_basis_pointers[b], refsenspts, orientation));
            cpsensorBasisGrad.push_back(disc->evaluateBasisGrads(param_basis_pointers[b], nodes,
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
    Kokkos::View<ScalarT***,AssemblyDevice> rot = wkset->rotation;
    parallel_for("cell update data", RangePolicy<AssemblyExec>(0,cell_data.extent(0)), KOKKOS_LAMBDA (const int e ) {
      rot(e,0,0) = cell_data(e,0);
      rot(e,0,1) = cell_data(e,1);
      rot(e,0,2) = cell_data(e,2);
      rot(e,1,0) = cell_data(e,3);
      rot(e,1,1) = cell_data(e,4);
      rot(e,1,2) = cell_data(e,5);
      rot(e,2,0) = cell_data(e,6);
      rot(e,2,1) = cell_data(e,7);
      rot(e,2,2) = cell_data(e,8);
    });
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
  else if (cellData->have_extra_data) {
    wkset->extra_data = cell_data;
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Pass the cell data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetAdjPrev(const ScalarT & val) {
  Kokkos::deep_copy(adjPrev,val);
}


///////////////////////////////////////////////////////////////////////////////////////
// Get the discretization/physics info (used for workset construction)
///////////////////////////////////////////////////////////////////////////////////////

vector<int> cell::getInfo() {
  vector<int> info;
  info.push_back(cellData->dimension);
  info.push_back(cellData->numDOF.extent(0));
  info.push_back((int)cellData->numDiscParams);
  info.push_back(numAuxDOF.extent(0));
  info.push_back(LIDs.extent(1));
  info.push_back(numElem);
  return info;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the solution at the nodes
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT***,AssemblyDevice> cell::getSolutionAtNodes(const int & var) {
  
  Teuchos::TimeMonitor nodesoltimer(*computeNodeSolTimer);
  
  int bnum = wkset->usebasis[var];
  DRV cbasis = basis_nodes[bnum];
  Kokkos::View<ScalarT***,AssemblyDevice> nodesol("solution at nodes",
                                                  cbasis.extent(0), cbasis.extent(2), cellData->dimension);
  auto uvals = Kokkos::subview(u,Kokkos::ALL(), var, Kokkos::ALL());
  parallel_for("cell node sol",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (int dof=0; dof<cbasis.extent(1); dof++ ) {
      ScalarT uval = uvals(elem,dof);
      for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
        for (int s=0; s<cbasis.extent(3); s++ ) {
          nodesol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
        }
      }
    }
  });
  
  return nodesol;
  
}
