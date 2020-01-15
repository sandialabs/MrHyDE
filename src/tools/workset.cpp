/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "workset.hpp"

////////////////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////////////////

workset::workset(const vector<int> & cellinfo, const DRV & ref_ip_, const DRV & ref_wts_, const DRV & ref_side_ip_,
                 const DRV & ref_side_wts_, const vector<string> & basis_types_,
                 const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
                 const topo_RCP & topo, Kokkos::View<int**,AssemblyDevice> & var_bcs_) : //, const Teuchos::RCP<TimeIntegrator> & timeInt_) :
ref_ip(ref_ip_), ref_wts(ref_wts_), ref_side_ip(ref_side_ip_), ref_side_wts(ref_side_wts_),
basis_types(basis_types_), basis_pointers(basis_pointers_), param_basis_pointers(param_basis_),
celltopo(topo), var_bcs(var_bcs_)  { //, timeInt(timeInt_) {
  
  // Settings that should not change
  dimension = cellinfo[0];
  numVars = cellinfo[1];
  numParams = cellinfo[2];
  numAux = cellinfo[3];
  numDOF = cellinfo[4];
  numElem = cellinfo[5];
  usebcs = true;
  
  /*
   num_stages = 1;//timeInt->num_stages;
   Kokkos::View<ScalarT**> newip("ip for stages",ref_ip.dimension(0)*num_stages, ref_ip.dimension(1));
   for (int i=0; i<ref_ip.dimension(0); i++) {
   for (int s=0; s<num_stages; s++) {
   for (int d=0; d<ref_ip.dimension(1); d++) {
   newip(i*num_stages+s,d) = ref_ip(i,d);
   }
   }
   }
   ref_ip = newip;
   Kokkos::View<ScalarT**> newsideip("ip for stages",ref_side_ip.dimension(0)*num_stages, ref_side_ip.dimension(1));
   for (int i=0; i<ref_side_ip.dimension(0); i++) {
   for (int s=0; s<num_stages; s++) {
   for (int d=0; d<ref_side_ip.dimension(1); d++) {
   newsideip(i*num_stages+s,d) = ref_side_ip(i,d);
   }
   }
   }
   ref_ip = newip;
   */
  
  // Integration information
  numip = ref_ip.dimension(0);
  numsideip = ref_side_ip.dimension(0);
  numsides = celltopo->getSideCount();
  
  time_KV = Kokkos::View<ScalarT*,AssemblyDevice>("time",1);
  //sidetype = Kokkos::View<int*,AssemblyDevice>("side types",numElem);
  ip = DRV("ip", numElem,numip, dimension);
  wts = DRV("wts", numElem, numip);
  ip_side = DRV("ip_side", numElem,numsideip,dimension);
  wts_side = DRV("wts_side", numElem,numsideip);
  normals = DRV("normals", numElem,numsideip,dimension);
  
  ip_KV = Kokkos::View<ScalarT***,AssemblyDevice>("ip stored in KV",numElem,numip,dimension);
  ip_side_KV = Kokkos::View<ScalarT***,AssemblyDevice>("side ip stored in KV",numElem,numsideip,dimension);
  normals_KV = Kokkos::View<ScalarT***,AssemblyDevice>("side normals stored in normals KV",numElem,numsideip,dimension);
  point_KV = Kokkos::View<ScalarT***,AssemblyDevice>("ip stored in point KV",1,1,dimension);
  
  //ip_side_vec = vector<DRV>(numBound);
  //wts_side_vec = vector<DRV>(numBound);
  //normals_side_vec = vector<DRV>(numBound);
  
  h = Kokkos::View<ScalarT*,AssemblyDevice>("h",numElem);
  res = Kokkos::View<AD**,AssemblyDevice>("residual",numElem,numDOF);
  adjrhs = Kokkos::View<AD**,AssemblyDevice>("adjoint RHS",numElem,numDOF);
  
  jacobDet = DRV("jacobDet",numElem, numip);
  jacobInv = DRV("jacobInv",numElem, numip, dimension, dimension);
  weightedMeasure = DRV("weightedMeasure",numElem, numip);
  
  sidejacobDet = DRV("sidejacobDet",numElem, numsideip);
  sidejacobInv = DRV("sidejacobInv",numElem, numsideip, dimension, dimension);
  sideweightedMeasure = DRV("sideweightedMeasure",numElem, numsideip);
  
  flux = Kokkos::View<AD***,AssemblyDevice>("flux",numElem,numVars,numsideip);
  
  have_rotation = false;
  have_rotation_phi = false;
  rotation = Kokkos::View<ScalarT***,AssemblyDevice>("rotation matrix",numElem,3,3);
  
  // Local solution with grad, div, curl
  local_soln = Kokkos::View<AD****, AssemblyDevice>("local_soln",numElem, numVars, numip, dimension);
  local_soln_grad = Kokkos::View<AD****, AssemblyDevice>("local_soln_grad",numElem, numVars, numip, dimension);
  local_soln_div = Kokkos::View<AD***, AssemblyDevice>("local_soln_div",numElem, numVars, numip);
  local_soln_curl = Kokkos::View<AD****, AssemblyDevice>("local_soln_curl",numElem, numVars, numip, dimension);
  local_soln_dot = Kokkos::View<AD****, AssemblyDevice>("local_soln_dot",numElem, numVars, numip, dimension);
  local_soln_dot_grad = Kokkos::View<AD****, AssemblyDevice>("local_soln_dot_grad",numElem, numVars, numip, dimension);
  
  local_param = Kokkos::View<AD***, AssemblyDevice>("local_param",numElem, numParams, numip);
  local_param_grad = Kokkos::View<AD****, AssemblyDevice>("local_param_grad",numElem, numParams, numip, dimension);
  
  local_aux = Kokkos::View<AD***, AssemblyDevice>("local_aux",numElem, numAux, numip);
  //local_aux_grad = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad",numElem, numAux, numip, dimension);
  
  local_soln_side = Kokkos::View<AD****, AssemblyDevice>("local_soln_side",numElem, numVars, numsideip, dimension);
  local_soln_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_soln_grad_side",numElem, numVars, numsideip, dimension);
  local_soln_dot_side = Kokkos::View<AD****, AssemblyDevice>("local_soln_dot_side",numElem, numVars, numsideip, dimension);
  
  local_param_side = Kokkos::View<AD***, AssemblyDevice>("local_param_side",numElem, numParams, numsideip);
  local_param_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_param_grad_side",numElem, numParams, numsideip, dimension);
  local_aux_side = Kokkos::View<AD***, AssemblyDevice>("local_aux_side",numElem, numAux, numsideip);
  //local_aux_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad_side",numElem, numAux, numsideip, dimension);
  
  local_soln_point = Kokkos::View<AD****, AssemblyDevice>("local_soln point",1, numVars, 1, dimension);
  local_soln_grad_point = Kokkos::View<AD****, AssemblyDevice>("local_soln point",1, numVars, 1, dimension);
  local_param_point = Kokkos::View<AD***, AssemblyDevice>("local_soln point",1, numParams, 1);
  local_param_grad_point = Kokkos::View<AD****, AssemblyDevice>("local_soln point",1, numParams, 1, dimension);
  
  // Compute the basis value and basis grad values on reference element
  // at volumetric ip
  /*
   int maxb = 0;
   for (size_t i=0; i<basis_pointers.size(); i++) {
   int numb = basis_pointers[i]->getCardinality();
   maxb = max(maxb, numb);
   }
   */
  
  this->setupBasis();
  this->setupParamBasis();
}

////////////////////////////////////////////////////////////////////////////////////
// Public functions
////////////////////////////////////////////////////////////////////////////////////

void workset::setupBasis() {
  
  for (size_t i=0; i<basis_pointers.size(); i++) {
    
    int numb = basis_pointers[i]->getCardinality();
    basis_grad.push_back(DRV("basis_grad",numElem,numb,numip,dimension));
    basis_grad_uw.push_back(DRV("basis_grad_uw",numElem,numb,numip,dimension));
    basis_div.push_back(DRV("basis_div",numElem,numb,numip));
    basis_div_uw.push_back(DRV("basis_div_uw",numElem,numb,numip));
    basis_curl.push_back(DRV("basis_curl",numElem,numb,numip,dimension));
    basis_curl_uw.push_back(DRV("basis_curl_uw",numElem,numb,numip,dimension));
    numbasis.push_back(numb);
    
    
    if (basis_types[i] == "HGRAD") {
      DRV basisvals("basisvals",numb, numip);
      basis_pointers[i]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
      DRV basisvals_Transformed("basisvals_Transformed",numElem, numb, numip);
      Intrepid2::FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
      ref_basis.push_back(basisvals_Transformed);
      basis.push_back(DRV("basis",numElem,numb,numip));
      basis_uw.push_back(DRV("basis_uw",numElem,numb,numip));
      
      
      DRV basisgrad("basisgrad",numb, numip, dimension);
      basis_pointers[i]->getValues(basisgrad, ref_ip, Intrepid2::OPERATOR_GRAD);
      ref_basis_grad.push_back(basisgrad);
      
      DRV basisdiv("basisdiv",numb, numip);
      ref_basis_div.push_back(basisdiv);
      
      DRV basiscurl("basiscurl",numb, numip, dimension);
      ref_basis_curl.push_back(basiscurl);
      
    }
    else if (basis_types[i] == "HDIV"){
      
      DRV basisvals("basisvals",numb, numip, dimension);
      basis_pointers[i]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
      
      DRV basisvals_Transformed("basisvals_Transformed",numElem, numb, numip, dimension);
      
      parallel_for(RangePolicy<AssemblyDevice>(0,basisvals_Transformed.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t m=0; m<numb; m++) {
          for (size_t j=0; j<numip; j++) {
            for (size_t k=0; k<dimension; k++) {
              basisvals_Transformed(e,m,j,k) = basisvals(m,j,k);
            }
          }
        }
      });
      
      ref_basis.push_back(basisvals_Transformed);
      basis.push_back(DRV("basis",numElem,numb,numip,dimension));
      basis_uw.push_back(DRV("basis_uw",numElem,numb,numip,dimension));
      
      DRV basisdiv("basisdiv",numb, numip);
      basis_pointers[i]->getValues(basisdiv, ref_ip, Intrepid2::OPERATOR_DIV);
      
      ref_basis_div.push_back(basisdiv);
      
      DRV basisgrad("basisgrad",numb, numip, dimension);
      ref_basis_grad.push_back(basisgrad);
      
      DRV basiscurl("basiscurl",numb, numip, dimension);
      ref_basis_curl.push_back(basiscurl);
      
    }
    else if (basis_types[i] == "HCURL"){
      DRV basisvals("basisvals",numb, numip, dimension);
      basis_pointers[i]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
      
      DRV basisvals_Transformed("basisvals_Transformed", numElem, numb, numip, dimension);
      parallel_for(RangePolicy<AssemblyDevice>(0,basisvals_Transformed.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_t m=0; m<numb; m++) {
          for (size_t j=0; j<numip; j++) {
            for (size_t k=0; k<dimension; k++) {
              basisvals_Transformed(e,m,j,k) = basisvals(m,j,k);
            }
          }
        }
      });
      
      ref_basis.push_back(basisvals_Transformed);
      basis.push_back(DRV("basis",numElem,numb,numip,dimension));
      basis_uw.push_back(DRV("basis_uw",numElem,numb,numip,dimension));
      
      DRV basiscurl("basiscurl",numb, numip, dimension);
      basis_pointers[i]->getValues(basiscurl, ref_ip, Intrepid2::OPERATOR_CURL);
      
      ref_basis_curl.push_back(basiscurl);
      
      DRV basisgrad("basisgrad",numb, numip, dimension);
      ref_basis_grad.push_back(basisgrad);
      
      DRV basisdiv("basisdiv",numb, numip);
      ref_basis_div.push_back(basisdiv);
      
    }
  }
  
  // Compute the basis value and basis grad values on reference element
  // at side ip
  
  for (size_t s=0; s<numsides; s++) {
    vector<DRV> csbasis;
    vector<DRV> csbasisgrad;
    for (size_t i=0; i<basis_pointers.size(); i++) {
      int numb = basis_pointers[i]->getCardinality();
      if (s==0) {
        basis_grad_side.push_back(DRV("basis_grad_side",numElem,numb,numsideip,dimension));
        basis_grad_side_uw.push_back(DRV("basis_grad_side_uw",numElem,numb,numsideip,dimension));
        basis_div_side.push_back(DRV("basis_div_side",numElem,numb,numsideip));
        basis_div_side_uw.push_back(DRV("basis_div_side_uw",numElem,numb,numsideip));
        basis_curl_side.push_back(DRV("basis_curl_side",numElem,numb,numsideip,dimension));
        basis_curl_side_uw.push_back(DRV("basis_curl_side_uw",numElem,numb,numsideip,dimension));
      }
      DRV refSidePoints("refSidePoints",numsideip, dimension);
      Intrepid2::CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, ref_side_ip, dimension-1, s, *celltopo);
      ref_side_ip_vec.push_back(refSidePoints);
      
      if (basis_types[i] == "HGRAD"){
        
        DRV basisvals("basisvals",numb, numsideip);
        basis_pointers[i]->getValues(basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
        DRV basisvals_Transformed("basisvals_Transformed",numElem, numb, numsideip);
        Intrepid2::FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
        csbasis.push_back(basisvals_Transformed);
        
        DRV basisgrad("basisgrad",numb, numsideip, dimension);
        basis_pointers[i]->getValues(basisgrad, refSidePoints, Intrepid2::OPERATOR_GRAD);
        csbasisgrad.push_back(basisgrad);
        
        if (s==0) {
          basis_side.push_back(DRV("basis_side",numElem,numb,numsideip)); // allocate weighted basis
          basis_side_uw.push_back(DRV("basis_side_uw",numElem,numb,numsideip)); // allocate un-weighted basis
        }
      }
      else if (basis_types[i] == "HDIV"){
        DRV basisvals("basisvals",numb, numsideip, dimension);
        basis_pointers[i]->getValues(basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
        DRV basisvals_Transformed("basisvals_Transformed",numElem, numb, numsideip, dimension);
        //FunctionSpaceTools<AssemblyDevice>::HDIVtransformVALUE(basisvals_Transformed, basisvals);
        csbasis.push_back(basisvals_Transformed);
        
        DRV basisgrad("basisgrad",numb, numsideip, dimension);
        csbasisgrad.push_back(basisgrad);
        
      }
      else if (basis_types[i] == "HCURL"){
        DRV basisvals("basisvals",numb, numsideip, dimension);
        basis_pointers[i]->getValues(basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
        DRV basisvals_Transformed("basisvals_Transformed",numElem, numb, numsideip, dimension);
        //FunctionSpaceTools<AssemblyDevice>::HCURLtransformVALUE(basisvals_Transformed, basisvals);
        csbasis.push_back(basisvals_Transformed);
        
        DRV basisgrad("basisgrad",numb, numsideip, dimension);
        csbasisgrad.push_back(basisgrad);
        
      }
    }
    ref_basis_side.push_back(csbasis);
    ref_basis_grad_side.push_back(csbasisgrad);
  }
  
  //basis_side_vec = vector<vector<DRV> >(numBound);
  //basis_grad_side_vec = vector<vector<DRV> >(numBound);
  //basis_side_uw_vec = vector<vector<DRV> >(numBound);
  //basis_grad_side_uw_vec = vector<vector<DRV> >(numBound);
  
}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

void workset::setupParamBasis() {
  
  // Compute the discretized parameter basis value and basis grad values on reference element
  // at volumetric ip
  for (size_t i=0; i<param_basis_pointers.size(); i++) {
    int numb = param_basis_pointers[i]->getCardinality();
    numparambasis.push_back(numb);
    DRV basisvals("basisvals",numb, numip);
    param_basis_pointers[i]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
    DRV basisvals_Transformed("basisvals_Transformed",numElem, numb, numip);
    Intrepid2::FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
    param_basis.push_back(basisvals_Transformed);
    
    DRV basisgrad("basisgrad",numb, numip, dimension);
    param_basis_pointers[i]->getValues(basisgrad, ref_ip, Intrepid2::OPERATOR_GRAD);
    param_basis_grad_ref.push_back(basisgrad);
    param_basis_grad.push_back(DRV("param_basis_grad",numElem,numb,numip,dimension));
  }
  
  // Compute the discretized parameter basis value and basis grad values on reference element
  // at side ip
  
  for (size_t s=0; s<numsides; s++) {
    vector<DRV> csbasis;
    vector<DRV> csbasisgrad;
    for (size_t i=0; i<param_basis_pointers.size(); i++) {
      int numb = param_basis_pointers[i]->getCardinality();
      DRV refSidePoints("refSidePoints", numsideip, dimension);
      Intrepid2::CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, ref_side_ip, dimension-1, s, *celltopo);
      
      DRV basisvals("basisvals", numb, numsideip);
      param_basis_pointers[i]->getValues(basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
      DRV basisvals_Transformed("basisvals_Transformed", numElem, numb, numsideip);
      Intrepid2::FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
      csbasis.push_back(basisvals_Transformed);
      
      DRV basisgrad("basisgrad", numb, numsideip, dimension);
      param_basis_pointers[i]->getValues(basisgrad, refSidePoints, Intrepid2::OPERATOR_GRAD);
      csbasisgrad.push_back(basisgrad);
      
      if (s==0) {
        param_basis_side.push_back(DRV("param_basis_side", numElem,numb,numsideip));
        param_basis_grad_side.push_back(DRV("param_basis_grad_side", numElem,numb,numsideip,dimension));
      }
    }
    param_basis_side_ref.push_back(csbasis);
    param_basis_grad_side_ref.push_back(csbasisgrad);
  }
  
  //param_basis_side_vec = vector<vector<DRV> >(numBound);
  //param_basis_grad_side_vec = vector<vector<DRV> >(numBound);
}

////////////////////////////////////////////////////////////////////////////////////
// Update the nodes and the basis functions at the volumetric ip
////////////////////////////////////////////////////////////////////////////////////

void workset::update(const DRV & ip_, const DRV & jacobian, const vector<vector<ScalarT> > & orientation) {
  
  using namespace Intrepid2;
  
  {
    
    Teuchos::TimeMonitor updatetimer(*worksetUpdateIPTimer);
    ip = ip_;
    
    
    parallel_for(RangePolicy<AssemblyDevice>(0,ip.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<numip; j++) {
        for (size_t k=0; k<dimension; k++) {
          ip_KV(e,j,k) = ip(e,j,k);
        }
      }
    });
    
    CellTools<AssemblyDevice>::setJacobianDet(jacobDet, jacobian);
    CellTools<AssemblyDevice>::setJacobianInv(jacobInv, jacobian);
    FunctionSpaceTools<AssemblyDevice>::computeCellMeasure(wts, jacobDet, ref_wts);
    
    parallel_for(RangePolicy<AssemblyDevice>(0,wts.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      ScalarT vol = 0.0;
      for (int i=0; i<numip; i++) {
        vol += wts(e,i);
      }
      h(e) = pow(vol,1.0/(ScalarT)dimension);
    });
  }
  
  {
    for (size_t i=0; i<basis_pointers.size(); i++) {
      
      if (basis_types[i] == "HGRAD"){
        basis_uw[i] = ref_basis[i];
        {
          Teuchos::TimeMonitor updatetimer(*worksetUpdateBasisMMTimer);
          FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis[i], wts, ref_basis[i]);
        }
        {
          Teuchos::TimeMonitor updatetimer(*worksetUpdateBasisHGTGTimer);
          FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(basis_grad_uw[i], jacobInv, ref_basis_grad[i]);
        }
        {
          Teuchos::TimeMonitor updatetimer(*worksetUpdateBasisMMTimer);
          FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_grad[i], wts, basis_grad_uw[i]);
        }
      }
      else if (basis_types[i] == "HDIV"){
        
        FunctionSpaceTools<AssemblyDevice>::HDIVtransformVALUE(basis_uw[i], jacobian, jacobDet, ref_basis[i]);
        FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis[i], wts, basis_uw[i]);
        FunctionSpaceTools<AssemblyDevice>::HDIVtransformDIV(basis_div_uw[i], jacobDet, ref_basis_div[i]);
        FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_div[i], wts, basis_div_uw[i]);
        /*
        vector<ScalarT> orient = {-1.0, -1.0, 1.0, 1.0};
        for (int e=0; e<numElem; e++) {
          for (size_t j=0; j<orient.size(); j++) {
            for (size_t k=0; k<numip; k++) {
              for (int s=0; s<dimension; s++) {
                basis_uw[i](e,j,k,s) *= orient[j];
                basis[i](e,j,k,s) *= orient[j];
              }
              basis_div_uw[i](e,j,k) *= orient[j];
              basis_div[i](e,j,k) *= orient[j];
            }
          }
        }*/
        
      }
      else if (basis_types[i] == "HCURL"){
        FunctionSpaceTools<AssemblyDevice>::HCURLtransformVALUE(basis_uw[i], jacobInv, ref_basis[i]);
        FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis[i], wts, basis_uw[i]);
        FunctionSpaceTools<AssemblyDevice>::HCURLtransformCURL(basis_curl_uw[i], jacobian, jacobDet, ref_basis_curl[i]);
        FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_curl[i], wts, basis_curl_uw[i]);
        
      }
    }
    
    for (size_t i=0; i<param_basis_pointers.size(); i++) {
      FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(param_basis_grad[i], jacobInv, param_basis_grad_ref[i]);
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
// Update the nodes and the basis functions at the side ip
////////////////////////////////////////////////////////////////////////////////////

int workset::addSide(const DRV & nodes, const int & sidenum,
                     Kokkos::View<int*> & localSideID) {
  
  using namespace Intrepid2;
  Teuchos::TimeMonitor updatetimer(*worksetAddSideTimer);
  int BID = 0;
  
  // check that all localSideIDs are the same
  for (size_t i=0; i<localSideID.size(); i++) {
    TEUCHOS_TEST_FOR_EXCEPTION(localSideID(i) != localSideID(0),
                               std::runtime_error,
                               "Workset Error: all elements do not share the same local side ID");
  }
  
  // ----------------------------------------------------------
  // Need to store ip, normals, basis, basis_uw, basis_grad, basis_grad_uw, param_basis, param_basis_grad
  // ----------------------------------------------------------
  int numBElem = nodes.dimension(0);
  
  DRV bip("sip", numBElem, ref_side_ip.dimension(0), dimension);
  DRV bijac("sijac", numBElem, ref_side_ip.dimension(0), dimension, dimension);
  DRV bijacInv("sijac", numBElem, ref_side_ip.dimension(0), dimension, dimension);
  DRV bwts("wts_side", numBElem, ref_side_ip.dimension(0));
  DRV bnormals("normals", numBElem, ref_side_ip.dimension(0), dimension);
  
  vector<DRV> cbasis, cbasis_uw, cbasis_grad, cbasis_grad_uw;
  
  for (size_t i=0; i<basis_pointers.size(); i++) {
    int numb = basis_pointers[i]->getCardinality();
    cbasis.push_back(DRV("basis_side",numBElem,numb,numsideip));
    cbasis_uw.push_back(DRV("basis_side",numBElem,numb,numsideip));
    cbasis_grad.push_back(DRV("basis_grad_side",numBElem,numb,numsideip,dimension));
    cbasis_grad_uw.push_back(DRV("basis_grad_side_uw",numBElem,numb,numsideip,dimension));
  }
  
  
  DRV refSidePoints("refSidePoints", ref_side_ip.dimension(0), dimension);
  
  CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, ref_side_ip,
                                                   dimension-1, localSideID(0), *celltopo);
  
    
  CellTools<AssemblyDevice>::mapToPhysicalFrame(bip, refSidePoints, nodes, *celltopo);
  CellTools<AssemblyDevice>::setJacobian(bijac, refSidePoints, nodes, *celltopo);
  CellTools<AssemblyDevice>::setJacobianInv(bijacInv, bijac);
  DRV temporary_buffer("temporary_buffer",numBElem*ref_side_ip.dimension(0)*dimension*dimension);
  
  if (dimension == 2) {
    FunctionSpaceTools<AssemblyDevice>::computeEdgeMeasure(bwts, bijac, ref_side_wts, localSideID(0),
                                                           *celltopo, temporary_buffer);
  }
  else if (dimension == 3) {
    FunctionSpaceTools<AssemblyDevice>::computeFaceMeasure(bwts, bijac, ref_side_wts, localSideID(0),
                                                           *celltopo, temporary_buffer);
  }
  CellTools<AssemblyDevice>::getPhysicalSideNormals(bnormals, bijac, localSideID(0), *celltopo);
  
  BID = ip_side_vec.size();
  // ----------------------------------------------------------
  // store ip
  //if (ip_side_vec.size() < cnum+1) {
  //  ip_side_vec.resize(cnum+1);
  //}
  ip_side_vec.push_back(bip);
  // ----------------------------------------------------------
  
  // ----------------------------------------------------------
  // store wts
  //if (wts_side_vec.size() < cnum+1) {
  //  wts_side_vec.resize(cnum+1);
  //}
  wts_side_vec.push_back(bwts);
  
  // ----------------------------------------------------------
  
  // scale the normal vector (we need unit normal...)
  for (int e=0; e<bnormals.dimension(0); e++) {
    for( int j=0; j<bnormals.dimension(1); j++ ) {
      ScalarT normalLength = 0.0;
      for (int sd=0; sd<dimension; sd++) {
        normalLength += bnormals(e,j,sd)*bnormals(e,j,sd);
      }
      normalLength = sqrt(normalLength);
      for (int sd=0; sd<dimension; sd++) {
        bnormals(e,j,sd) = bnormals(e,j,sd) / normalLength;
      }
    }
  }
  
  // ----------------------------------------------------------
  // store normals
  //if (normals_side_vec.size() < cnum+1) {
  //  normals_side_vec.resize(cnum+1);
  //}
  normals_side_vec.push_back(bnormals);
  // ----------------------------------------------------------
  
  // ----------------------------------------------------------
  // evaluate basis vectors
  for (size_t i=0; i<basis_pointers.size(); i++) {
    if (basis_types[i] == "HGRAD"){
      int numb = basis_pointers[i]->getCardinality();
      
      DRV basisvals("basisvals",numb, numsideip);
      basis_pointers[i]->getValues(basisvals, refSidePoints, OPERATOR_VALUE);
      
      DRV basisvals_trans("basisvals_Transformed",numBElem, numb, numsideip);
      FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(cbasis_uw[i], basisvals);
      
      DRV basisgrad("basisgrad",numb, numsideip, dimension);
      basis_pointers[i]->getValues(basisgrad, refSidePoints, OPERATOR_GRAD);
      
      FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(cbasis[i], bwts, cbasis_uw[i]);
      FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(cbasis_grad_uw[i], bijacInv, basisgrad);
      FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(cbasis_grad[i], bwts, cbasis_grad_uw[i]);
    }
    else if (basis_types[i] == "HDIV"){
      //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
    }
    else if (basis_types[i] == "HCURL"){
      //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
    }
  }
  // ----------------------------------------------------------
  
  // ----------------------------------------------------------
  // store basis values
  //if (basis_side_vec.size() < cnum+1) {
  //  basis_side_vec.resize(cnum+1);
  //}
  basis_side_vec.push_back(cbasis);
  
  //if (basis_side_uw_vec.size() < cnum+1) {
  //  basis_side_uw_vec.resize(cnum+1);
  //}
  basis_side_uw_vec.push_back(cbasis_uw);
  
  //if (basis_grad_side_vec.size() < cnum+1) {
  //  basis_grad_side_vec.resize(cnum+1);
  //}
  basis_grad_side_vec.push_back(cbasis_grad);
  
  //if (basis_grad_side_uw_vec.size() < cnum+1) {
  //  basis_grad_side_uw_vec.resize(cnum+1);
  //}
  basis_grad_side_uw_vec.push_back(cbasis_grad_uw);
  // ----------------------------------------------------------
  
  // ----------------------------------------------------------
  // evaluate param basis vectors
  
  vector<DRV> cpbasis, cpbasis_grad;
  
  for (size_t i=0; i<param_basis_pointers.size(); i++) {
    int numb = param_basis_pointers[i]->getCardinality();
    cpbasis.push_back(DRV("basis_side",numBElem,numb,numsideip));
    cpbasis_grad.push_back(DRV("basis_grad_side",numBElem,numb,numsideip,dimension));
  }
  for (size_t i=0; i<param_basis_pointers.size(); i++) {
    int numb = param_basis_pointers[i]->getCardinality();
    DRV basisvals("basisvals",numb, numsideip);
    param_basis_pointers[i]->getValues(basisvals, refSidePoints, OPERATOR_VALUE);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(cpbasis[i], basisvals);
    DRV basisgrad("basisgrad",numb, numsideip, dimension);
    basis_pointers[i]->getValues(basisgrad, refSidePoints, OPERATOR_GRAD);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(cpbasis_grad[i], bijacInv, basisgrad);
  }
  // ----------------------------------------------------------
  
  // ----------------------------------------------------------
  // store param basis values
  //if (param_basis_side_vec.size() < cnum+1) {
  //  param_basis_side_vec.resize(cnum+1);
  //}
  param_basis_side_vec.push_back(cpbasis);
  
  //if (param_basis_grad_side_vec.size() < cnum+1) {
  //  param_basis_grad_side_vec.resize(cnum+1);
  //}
  param_basis_grad_side_vec.push_back(cpbasis_grad);
  // ----------------------------------------------------------
  return BID;
}


////////////////////////////////////////////////////////////////////////////////////
// Update the nodes and the basis functions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::updateSide(const DRV & nodes, const DRV & ip_side_, const DRV & wts_side_,
                         const DRV & normals_, const DRV & sidejacobian, const int & s) {
  
  using namespace Intrepid2;
  
  {
    Teuchos::TimeMonitor updatetimer(*worksetSideUpdateIPTimer);
    
    ip_side = ip_side_;
    wts_side = wts_side_;
    normals = normals_;
    
    parallel_for(RangePolicy<AssemblyDevice>(0,normals.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<numsideip; j++) {
        for (size_t k=0; k<dimension; k++) {
          ip_side_KV(e,j,k) = ip_side(e,j,k);
          normals_KV(e,j,k) = normals(e,j,k);
        }
      }
    });
    
    CellTools<AssemblyDevice>::setJacobianInv(sidejacobInv, sidejacobian);
    
  }
  
  {
    Teuchos::TimeMonitor updatetimer(*worksetSideUpdateBasisTimer);
    
    for (size_t i=0; i<basis_pointers.size(); i++) {
      if (basis_types[i] == "HGRAD"){
        basis_side_uw[i] = ref_basis_side[s][i];
        FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
        FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(basis_grad_side_uw[i], sidejacobInv, ref_basis_grad_side[s][i]);
        FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_grad_side[i], wts_side, basis_grad_side_uw[i]);
      }
      else if (basis_types[i] == "HDIV"){
        //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
      }
      else if (basis_types[i] == "HCURL"){
        //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
      }
    }
    
    for (size_t i=0; i<param_basis_pointers.size(); i++) {
      param_basis_side[i] = param_basis_side_ref[s][i];
      FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(param_basis_grad_side[i], sidejacobInv, param_basis_grad_side_ref[s][i]);
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
// Update the nodes and the basis functions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::updateSide(const int & sidenum, const int & cnum) {
  
  currentside = sidenum;
  
  
  {
    Teuchos::TimeMonitor updatetimer(*worksetSideUpdateIPTimer);
  
    ip_side = ip_side_vec[cnum];
    wts_side = wts_side_vec[cnum];
    normals = normals_side_vec[cnum];
    
    parallel_for(RangePolicy<AssemblyDevice>(0,normals.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<numsideip; j++) {
        for (size_t k=0; k<dimension; k++) {
          ip_side_KV(e,j,k) = ip_side(e,j,k);
          normals_KV(e,j,k) = normals(e,j,k);
        }
      }
    });
  }
  
  {
    Teuchos::TimeMonitor updatetimer(*worksetSideUpdateBasisTimer);
    
    basis_side = basis_side_vec[cnum];
    basis_side_uw = basis_side_uw_vec[cnum];
    basis_grad_side_uw = basis_grad_side_uw_vec[cnum];
    basis_grad_side = basis_grad_side_vec[cnum];
    
    param_basis_side = param_basis_side_vec[cnum];
    param_basis_grad_side = param_basis_grad_side_vec[cnum];
    
  }
  
}



////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetResidual() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<res.dimension(1); n++) {
      res(e,n) = 0.0;
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetResidual(const int & numE) {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<res.dimension(1); n++) {
      res(e,n) = 0.0;
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetFlux() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  parallel_for(RangePolicy<AssemblyDevice>(0,flux.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<flux.dimension(1); n++) {
      for (int k=0; k<flux.dimension(2); k++) {
        flux(e,n,k) = 0.0;
      }
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetAux() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  parallel_for(RangePolicy<AssemblyDevice>(0,local_aux.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<local_aux.dimension(1); n++) {
      for (int k=0; k<local_aux.dimension(2); k++) {
        local_aux(e,n,k) = 0.0;
        //for (int s=0; s<local_aux_grad.dimension(3); s++) {
        //  local_aux_grad(e,n,k,s) = 0.0;
        //}
      }
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetAuxSide() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  parallel_for(RangePolicy<AssemblyDevice>(0,local_aux_side.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<local_aux_side.dimension(1); n++) {
      for (int k=0; k<local_aux_side.dimension(2); k++) {
        local_aux_side(e,n,k) = 0.0;
        //for (int s=0; s<local_aux_grad_side.dimension(3); s++) {
        //  local_aux_grad_side(e,n,k,s) = 0.0;
        //}
      }
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetAdjointRHS() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  parallel_for(RangePolicy<AssemblyDevice>(0,adjrhs.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<adjrhs.dimension(1); n++) {
      adjrhs(e,n) = 0.0;
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the volumetric ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnVolIP(Kokkos::View<ScalarT***,AssemblyDevice> u) {
  
  // Reset the values
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    parallel_for(RangePolicy<AssemblyDevice>(0,local_soln.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_soln.dimension(1); k++) {
        for (int i=0; i<local_soln.dimension(2); i++) {
          for (int s=0; s<local_soln.dimension(3); s++) {
            local_soln(e,k,i,s) = 0.0;
            local_soln_grad(e,k,i,s) = 0.0;
            local_soln_curl(e,k,i,s) = 0.0;
          }
          local_soln_div(e,k,i) = 0.0;
        }
      }
    });
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnVolTimer);
    AD uval;
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      int knbasis = numbasis[kubasis];
      string kutype = basis_types[kubasis];
      
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_grad_uw = basis_grad_uw[kubasis];
        
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u(e,k,i);
            for( size_t j=0; j<numip; j++ ) {
              local_soln(e,k,j,0) += uval*kbasis_uw(e,i,j);
              for( int s=0; s<dimension; s++ ) {
                local_soln_grad(e,k,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (basis_types[usebasis[k]] == "HDIV"){
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_div_uw = basis_div_uw[kubasis];
        
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u(e,k,i);
            for( size_t j=0; j<numip; j++ ) {
              for( int s=0; s<dimension; s++ ) {
                local_soln(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
              }
              local_soln_div(e,k,j) += uval*kbasis_div_uw(e,i,j);
            }
          }
        }
      }
      else if (basis_types[usebasis[k]] == "HCURL"){
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_curl_uw = basis_curl_uw[kubasis];
        
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u(e,k,i);
            for( size_t j=0; j<numip; j++ ) {
              for( int s=0; s<dimension; s++ ) {
                local_soln(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
                local_soln_curl(e,k,j,s) += uval*kbasis_curl_uw(e,i,j,s);
              }
            }
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the volumetric ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnVolIP(Kokkos::View<ScalarT***,AssemblyDevice> u,
                               Kokkos::View<ScalarT***,AssemblyDevice> u_dot,
                               const bool & seedu, const bool & seedudot) {
  
  // Reset the values (may combine with next loop when parallelized)
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    parallel_for(RangePolicy<AssemblyDevice>(0,local_soln.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_soln.dimension(1); k++) {
        for (int i=0; i<local_soln.dimension(2); i++) {
          for (int s=0; s<local_soln.dimension(3); s++) {
            local_soln(e,k,i,s) = 0.0;
            local_soln_dot(e,k,i,s) = 0.0;
            local_soln_grad(e,k,i,s) = 0.0;
            //local_soln_dot_grad(e,k,i,s) = 0.0;
            local_soln_curl(e,k,i,s) = 0.0;
          }
          local_soln_div(e,k,i) = 0.0;
        }
      }
    });
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnVolTimer);
    AD uval, u_dotval;
    
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      int knbasis = numbasis[kubasis];
      string kutype = basis_types[kubasis];
      
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_grad_uw = basis_grad_uw[kubasis];
        
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            if (seedu) {
              uval = AD(maxDerivs,offsets(k,i),u(e,k,i));
            }
            else {
              uval = u(e,k,i);
            }
            if (seedudot) {
              u_dotval = AD(maxDerivs,offsets(k,i),u_dot(e,k,i));
            }
            else {
              u_dotval = u_dot(e,k,i);
            }
            
            for( size_t j=0; j<numip; j++ ) {
              local_soln(e,k,j,0) += uval*kbasis_uw(e,i,j);
              local_soln_dot(e,k,j,0) += u_dotval*kbasis_uw(e,i,j);
              for( int s=0; s<dimension; s++ ) {
                local_soln_grad(e,k,j,s) += uval*kbasis_grad_uw(e,i,j,s);
                //local_soln_dot_grad(e,k,j,s) += u_dotval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HDIV"){
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_div_uw = basis_div_uw[kubasis];
        
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            
            if (seedu) {
              uval = AD(maxDerivs,offsets(k,i),u(e,k,i));
            }
            else {
              uval = u(e,k,i);
            }
            if (seedudot) {
              u_dotval = AD(maxDerivs,offsets(k,i),u_dot(e,k,i));
            }
            else {
              u_dotval = u_dot(e,k,i);
            }
            
            for( size_t j=0; j<numip; j++ ) {
              for( int s=0; s<dimension; s++ ) {
                local_soln(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
                local_soln_dot(e,k,j,s) += u_dotval*kbasis_uw(e,i,j,s);
              }
              local_soln_div(e,k,j) += uval*kbasis_div_uw(e,i,j);
            }
          }
        }
      }
      else if (basis_types[usebasis[k]] == "HCURL"){
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_curl_uw = basis_curl_uw[kubasis];
        
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            
            if (seedu) {
              uval = AD(maxDerivs,offsets(k,i),u(e,k,i));
            }
            else {
              uval = u(e,k,i);
            }
            if (seedudot) {
              u_dotval = AD(maxDerivs,offsets(k,i),u_dot(e,k,i));
            }
            else {
              u_dotval = u_dot(e,k,i);
            }
            
            for( size_t j=0; j<numip; j++ ) {
              for( int s=0; s<dimension; s++ ) {
                local_soln(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
                local_soln_dot(e,k,j,s) += u_dotval*kbasis_uw(e,i,j,s);
                local_soln_curl(e,k,j,s) += uval*kbasis_curl_uw(e,i,j,s);
              }
            }
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the discretized parameters at the volumetric ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamVolIP(Kokkos::View<ScalarT***,AssemblyDevice> param, const bool & seedparams) {
  
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    // Reset the values (may combine with next loop when parallelized)
    parallel_for(RangePolicy<AssemblyDevice>(0,local_param.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_param.dimension(1); k++) {
        for (int i=0; i<local_param.dimension(2); i++) {
          local_param(e,k,i) = 0.0;
          for (int s=0; s<local_param_grad.dimension(3); s++) {
            local_param_grad(e,k,i,s) = 0.0;
          }
        }
      }
    });
  }
  
  //local_param.initialize(0.0);
  //local_param_grad.initialize(0.0);
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeParamVolTimer);
    AD paramval;
    for (int k=0; k<numParams; k++) {
      int kpbasis = paramusebasis[k];
      int knpbasis = numparambasis[kpbasis];
      
      DRV pbasis = param_basis[kpbasis];
      DRV pbasis_grad = param_basis_grad[kpbasis];
      
      for( int i=0; i<knpbasis; i++ ) {
        for (int e=0; e<numElem; e++) {
          
          if (seedparams) {
            paramval = AD(maxDerivs,paramoffsets(k,i),param(e,k,i));
          }
          else {
            paramval = param(e,k,i);
          }
          for( size_t j=0; j<numip; j++ ) {
            local_param(e,k,j) += paramval*pbasis(e,i,j);
            for( int s=0; s<dimension; s++ ) {
              local_param_grad(e,k,j,s) += paramval*pbasis_grad(e,i,j,s);
            }
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSideIP(const int & side, Kokkos::View<ScalarT***,AssemblyDevice> u,
                                Kokkos::View<ScalarT***,AssemblyDevice> u_dot,
                                const bool & seedu, const bool& seedudot) {
  
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    // Reset the values (may combine with next loop when parallelized)
    parallel_for(RangePolicy<AssemblyDevice>(0,local_soln_side.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_soln_side.dimension(1); k++) {
        for (int i=0; i<local_soln_side.dimension(2); i++) {
          for (int s=0; s<local_soln_side.dimension(3); s++) {
            local_soln_side(e,k,i,s) = 0.0;
            local_soln_grad_side(e,k,i,s) = 0.0;
            //local_soln_dot_side(e,k,i,s) = 0.0;
          }
        }
      }
    });
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    AD uval, u_dotval;
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      int knbasis = numbasis[kubasis];
      string kutype = basis_types[kubasis];
      
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_side_uw[kubasis];
        DRV kbasis_grad_uw = basis_grad_side_uw[kubasis];
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            
            if (seedu) {
              uval = AD(maxDerivs,offsets(k,i),u(e,k,i));
            }
            else {
              uval = u(e,k,i);
            }
            //if (seedudot) {
            //  u_dotval = AD(maxDerivs,offsets(k,i),u_dot(e,k,i));
            //}
            //else {
            //  u_dotval = u_dot(e,k,i);
            //}
            for( size_t j=0; j<numsideip; j++ ) {
              local_soln_side(e,k,j,0) += uval*kbasis_uw(e,i,j);
              //local_soln_dot_side(e,k,j,0) += u_dotval*kbasis_uw(e,i,j);
              for( int s=0; s<dimension; s++ ) {
                local_soln_grad_side(e,k,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HDIV"){
        
        /*
         DRV kbasis_uw = ref_basis_side[side][kubasis];
         
         for( int i=0; i<knbasis; i++ ) {
         for (int e=0; e<numElem; e++) {
         
         if (seedu) {
         uval = AD(maxDerivs,offsets(k,i),u(e,k,i));
         }
         else {
         uval = u(e,k,i);
         }
         if (seedudot) {
         u_dotval = AD(maxDerivs,offsets(k,i),u_dot(e,k,i));
         }
         else {
         u_dotval = u_dot(e,k,i);
         }
         for( size_t j=0; j<numsideip; j++ ) {
         for( int s=0; s<dimension; s++ ) {
         local_soln_side(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
         local_soln_dot_side(e,k,j,s) += u_dotval*kbasis_uw(e,i,j,s);
         }
         }
         }
         }
         */
        
      }
      else if (kutype == "HCURL"){
        
        /*
         DRV kbasis_uw = ref_basis_side[side][kubasis];
         
         for( int i=0; i<knbasis; i++ ) {
         for (int e=0; e<numElem; e++) {
         
         if (seedu) {
         uval = AD(maxDerivs,offsets(k,i),u(e,k,i));
         }
         else {
         uval = u(e,k,i);
         }
         if (seedudot) {
         u_dotval = AD(maxDerivs,offsets(k,i),u_dot(e,k,i));
         }
         else {
         u_dotval = u_dot(e,k,i);
         }
         for( size_t j=0; j<numsideip; j++ ) {
         for( int s=0; s<dimension; s++ ) {
         local_soln_side(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
         local_soln_dot_side(e,k,j,s) += u_dotval*kbasis_uw(e,i,j,s);
         }
         }
         }
         }
         */
      }
      
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the discretized parameters at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamSideIP(const int & side, Kokkos::View<ScalarT***,AssemblyDevice> param,
                                 const bool & seedparams) {
  
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    // Reset the values (may combine with next loop when parallelized)
    parallel_for(RangePolicy<AssemblyDevice>(0,local_param_side.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_param_side.dimension(1); k++) {
        for (int i=0; i<local_param_side.dimension(2); i++) {
          local_param_side(e,k,i) = 0.0;
          for (int s=0; s<local_param_grad_side.dimension(3); s++) {
            local_param_grad_side(e,k,i,s) = 0.0;
          }
        }
      }
    });
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeParamSideTimer);
    
    AD paramval;
    
    for (int k=0; k<numParams; k++) {
      int kpbasis = paramusebasis[k];
      int knpbasis = numparambasis[kpbasis];
      
      DRV pbasis = param_basis_side[kpbasis];
      DRV pbasis_grad = param_basis_grad_side[kpbasis];
      
      for( int i=0; i<knpbasis; i++ ) {
        for (int e=0; e<numElem; e++) {
          if (seedparams) {
            paramval = AD(maxDerivs,paramoffsets(k,i),param(e,k,i));
          }
          else {
            paramval = param(e,k,i);
          }
          
          for( size_t j=0; j<numsideip; j++ ) {
            local_param_side(e,k,j) += paramval*pbasis(e,i,j);
            for( int s=0; s<dimension; s++ ) {
              local_param_grad_side(e,k,j,s) += paramval*pbasis_grad(e,i,j,s);
            }
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSideIP(const int & side, Kokkos::View<AD***,AssemblyDevice> u_AD,
                                Kokkos::View<AD***,AssemblyDevice> u_dot_AD,
                                Kokkos::View<AD***,AssemblyDevice> param_AD) {
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    // Reset the values (may combine with next loop when parallelized)
    parallel_for(RangePolicy<AssemblyDevice>(0,local_soln_side.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_soln_side.dimension(1); k++) {
        for (int i=0; i<local_soln_side.dimension(2); i++) {
          for (int s=0; s<local_soln_side.dimension(3); s++) {
            local_soln_side(e,k,i,s) = 0.0;
            local_soln_grad_side(e,k,i,s) = 0.0;
            local_soln_dot_side(e,k,i,s) = 0.0;
          }
        }
      }
    });
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    AD uval, u_dotval;
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      int knbasis = numbasis[kubasis];
      string kutype = basis_types[kubasis];
      
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_side_uw[kubasis];
        DRV kbasis_grad_uw = basis_grad_side_uw[kubasis];
        
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u_AD(e,k,i);
            u_dotval = u_dot_AD(e,k,i);
            for( size_t j=0; j<numsideip; j++ ) {
              local_soln_side(e,k,j,0) += uval*kbasis_uw(e,i,j);
              local_soln_dot_side(e,k,j,0) += u_dotval*kbasis_uw(e,i,j);
              for( int s=0; s<dimension; s++ ) {
                local_soln_grad_side(e,k,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HDIV"){
        DRV kbasis_uw = ref_basis_side[side][kubasis];
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u_AD(e,k,i);
            u_dotval = u_dot_AD(e,k,i);
            for( size_t j=0; j<numsideip; j++ ) {
              for( int s=0; s<dimension; s++ ) {
                local_soln_side(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
                local_soln_dot_side(e,k,j,s) += u_dotval*kbasis_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HCURL"){
        DRV kbasis_uw = ref_basis_side[side][kubasis];
        
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u_AD(e,k,i);
            u_dotval = u_dot_AD(e,k,i);
            for( size_t j=0; j<numsideip; j++ ) {
              for( int s=0; s<dimension; s++ ) {
                local_soln_side(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
                local_soln_dot_side(e,k,j,s) += u_dotval*kbasis_uw(e,i,j,s);
              }
            }
          }
        }
      }
      
    }
  }
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    // Reset the values (may combine with next loop when parallelized)
    parallel_for(RangePolicy<AssemblyDevice>(0,local_param_side.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_param_side.dimension(1); k++) {
        for (int i=0; i<local_param_side.dimension(2); i++) {
          local_param_side(e,k,i) = 0.0;
          for (int s=0; s<local_param_grad_side.dimension(3); s++) {
            local_param_grad_side(e,k,i,s) = 0.0;
          }
        }
      }
    });
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeParamSideTimer);
    
    AD paramval;
    
    for (int k=0; k<numParams; k++) {
      int kpbasis = paramusebasis[k];
      int knpbasis = numparambasis[kpbasis];
      
      DRV pbasis = param_basis[kpbasis];
      DRV pbasis_grad = param_basis_grad[kpbasis];
      
      for( int i=0; i<knpbasis; i++ ) {
        for (int e=0; e<numElem; e++) {
          paramval = param_AD(e,k,i);
          for( size_t j=0; j<numsideip; j++ ) {
            local_param_side(e,k,j) += paramval*pbasis(e,i,j);
            for( int s=0; s<dimension; s++ ) {
              local_param_grad_side(e,k,j,s) += paramval*pbasis_grad(e,i,j,s);
            }
          }
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////
// Add Aux
//////////////////////////////////////////////////////////////

void workset::addAux(const size_t & naux) {
  numAux = naux;
  local_aux = Kokkos::View<AD***, AssemblyDevice>("local_aux",numElem, numAux, numip);
  //local_aux_grad = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad",numElem, numAux, numip, dimension);
  local_aux_side = Kokkos::View<AD***, AssemblyDevice>("local_aux_side",numElem, numAux, numsideip);
  //local_aux_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad_side",numElem, numAux, numsideip, dimension);
}

//////////////////////////////////////////////////////////////
// Get a pointer to vector of parameters
//////////////////////////////////////////////////////////////

vector<AD> workset::getParam(const string & name, bool & found) {
  found = false;
  int iter=0;
  vector<AD> pvec;
  while (!found && iter<paramnames.size()) {
    if (paramnames[iter] == name) {
      found  = true;
      pvec = *(params[iter]);
    }
    else {
      iter++;
    }
  }
  if (!found) {
    pvec = vector<AD>(1);
  }
  return pvec;
}
