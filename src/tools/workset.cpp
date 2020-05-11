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
   Kokkos::View<ScalarT**> newip("ip for stages",ref_ip.extent(0)*num_stages, ref_ip.extent(1));
   for (int i=0; i<ref_ip.extent(0); i++) {
   for (int s=0; s<num_stages; s++) {
   for (int d=0; d<ref_ip.extent(1); d++) {
   newip(i*num_stages+s,d) = ref_ip(i,d);
   }
   }
   }
   ref_ip = newip;
   Kokkos::View<ScalarT**> newsideip("ip for stages",ref_side_ip.extent(0)*num_stages, ref_side_ip.extent(1));
   for (int i=0; i<ref_side_ip.extent(0); i++) {
   for (int s=0; s<num_stages; s++) {
   for (int d=0; d<ref_side_ip.extent(1); d++) {
   newsideip(i*num_stages+s,d) = ref_side_ip(i,d);
   }
   }
   }
   ref_ip = newip;
   */
  
  deltat = 1.0;
  // Integration information
  numip = ref_ip.extent(0);
  numsideip = ref_side_ip.extent(0);
  if (dimension == 2) {
    numsides = celltopo->getSideCount();
  }
  else if (dimension == 3) {
    numsides = celltopo->getFaceCount();
  }
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
  
  for (size_t s=0; s<numsides; s++) {
    DRV refSidePoints("refSidePoints",numsideip, dimension);
    CellTools::mapToReferenceSubcell(refSidePoints, ref_side_ip, dimension-1, s, *celltopo);
    ref_side_ip_vec.push_back(refSidePoints);
    
    DRV refSideNormals("refSideNormals",numsideip, dimension);
    DRV refSideTangents("refSideTangents", dimension);
    DRV refSideTangentsU("refSideTangents U", dimension);
    DRV refSideTangentsV("refSideTangents V", dimension);
    if (dimension == 2) {
      CellTools::getReferenceSideNormal(refSideNormals,s,*celltopo);
      CellTools::getReferenceEdgeTangent(refSideTangents,s,*celltopo);
    }
    else if (dimension == 3) {
      //CellTools::getReferenceFaceNormal(refSideNormals,s,*celltopo);
      CellTools::getReferenceFaceTangents(refSideTangentsU, refSideTangentsV, s, *celltopo);
      //CellTools::getReferenceFaceTangents(refSideTangents,s,*celltopo);
    }
    ref_side_normals_vec.push_back(refSideNormals);
    ref_side_tangents_vec.push_back(refSideTangents);
    ref_side_tangentsU_vec.push_back(refSideTangentsU);
    ref_side_tangentsV_vec.push_back(refSideTangentsV);
  }
  
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
  
  local_soln_face = Kokkos::View<AD****, AssemblyDevice>("local_soln_face",numElem, numVars, numsideip, dimension);
  local_soln_grad_face = Kokkos::View<AD****, AssemblyDevice>("local_soln_grad_face",numElem, numVars, numsideip, dimension);
  
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
    
    
    if (basis_types[i] == "HGRAD" || basis_types[i] == "HVOL") {
      DRV basisvals("basisvals",numb, numip);
      basis_pointers[i]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
      DRV basisvals_Transformed("basisvals_Transformed",numElem, numb, numip);
      FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
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
      
      ref_basis.push_back(basisvals);
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
      
      ref_basis.push_back(basisvals);
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
    else if (basis_types[i] == "HFACE"){
      DRV basisvals("basisvals",numb, numip, dimension);
      ref_basis.push_back(basisvals);
      
      basis.push_back(DRV("basis",numElem,numb,numip,dimension));
      basis_uw.push_back(DRV("basis_uw",numElem,numb,numip,dimension));
      
      DRV basiscurl("basiscurl",numb, numip, dimension);
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
    DRV refSidePoints = ref_side_ip_vec[s];
    
    for (size_t i=0; i<basis_pointers.size(); i++) {
      int numb = basis_pointers[i]->getCardinality();
      if (s==0) {
        basis_side.push_back(DRV("basis_side",numElem,numb,numsideip)); // allocate weighted basis
        basis_side_uw.push_back(DRV("basis_side_uw",numElem,numb,numsideip)); // allocate un-weighted basis
        basis_grad_side.push_back(DRV("basis_grad_side",numElem,numb,numsideip,dimension));
        basis_grad_side_uw.push_back(DRV("basis_grad_side_uw",numElem,numb,numsideip,dimension));
        basis_div_side.push_back(DRV("basis_div_side",numElem,numb,numsideip));
        basis_div_side_uw.push_back(DRV("basis_div_side_uw",numElem,numb,numsideip));
        basis_curl_side.push_back(DRV("basis_curl_side",numElem,numb,numsideip,dimension));
        basis_curl_side_uw.push_back(DRV("basis_curl_side_uw",numElem,numb,numsideip,dimension));
        
        basis_face.push_back(DRV("basis_face",numElem,numb,numsideip)); // allocate weighted basis
        basis_face_uw.push_back(DRV("basis_face_uw",numElem,numb,numsideip)); // allocate un-weighted basis
        basis_grad_face.push_back(DRV("basis_grad_face",numElem,numb,numsideip,dimension));
        basis_grad_face_uw.push_back(DRV("basis_grad_face_uw",numElem,numb,numsideip,dimension));
        
      }
      
      
      if (basis_types[i] == "HGRAD" || basis_types[i] == "HVOL" || basis_types[i] == "HFACE"){
        
        DRV basisvals("basisvals",numb, numsideip);
        basis_pointers[i]->getValues(basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
        DRV basisvals_Transformed("basisvals_Transformed",numElem, numb, numsideip);
        FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
        csbasis.push_back(basisvals_Transformed);
        
        DRV basisgrad("basisgrad",numb, numsideip, dimension);
        basis_pointers[i]->getValues(basisgrad, refSidePoints, Intrepid2::OPERATOR_GRAD);
        csbasisgrad.push_back(basisgrad);
        
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
    param_basis_ref.push_back(basisvals);
    param_basis.push_back(DRV("param_basis",numElem,numb,numip));
    //DRV basisvals_Transformed("basisvals_Transformed",numElem, numb, numip);
    //Intrepid2::FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
    //param_basis.push_back(basisvals_Transformed);
    
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
      CellTools::mapToReferenceSubcell(refSidePoints, ref_side_ip, dimension-1, s, *celltopo);
      
      DRV basisvals("basisvals", numb, numsideip);
      param_basis_pointers[i]->getValues(basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
      DRV basisvals_Transformed("basisvals_Transformed", numElem, numb, numsideip);
      FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
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

void workset::update(const DRV & ip_, const DRV & wts_, const DRV & jacobian,
                     const DRV & jacobianInv, const DRV & jacobianDet,
                     Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & orientation) {
  
  size_t activeElem = ip_.extent(0);
  {
    
    Teuchos::TimeMonitor updatetimer(*worksetUpdateIPTimer);
    ip = ip_;
    wts = wts_;
    
    parallel_for(RangePolicy<AssemblyExec>(0,activeElem), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<numip; j++) {
        for (size_t k=0; k<dimension; k++) {
          ip_KV(e,j,k) = ip(e,j,k);
        }
      }
    });
    //KokkosTools::print(jacobianDet);
    
    //CellTools::setJacobianDet(jacobDet, jacobian);
    //CellTools::setJacobianInv(jacobInv, jacobian);
    //FuncTools::computeCellMeasure(wts, jacobDet, ref_wts);
    
    parallel_for(RangePolicy<AssemblyExec>(0,activeElem), KOKKOS_LAMBDA (const int e ) {
      ScalarT vol = 0.0;
      for (int i=0; i<numip; i++) {
        vol += wts(e,i);
      }
      h(e) = pow(vol,1.0/(ScalarT)dimension);
    });
  }
  
  //ots::modifyBasisByOrientation(basis_out, basis_in, orientations[i],basis_pointers[i]);
  
  {
    
    Teuchos::TimeMonitor dbgtimer(*worksetDebugTimer1);
    
    for (size_t i=0; i<basis_pointers.size(); i++) {
      
      int numb = basis_pointers[i]->getCardinality();
      
      if (basis_types[i] == "HGRAD"){
        OrientTools::modifyBasisByOrientation(basis_uw[i], ref_basis[i], orientation, basis_pointers[i].get());
        {
          Teuchos::TimeMonitor updatetimer(*worksetUpdateBasisMMTimer);
          FuncTools::multiplyMeasure(basis[i], wts, basis_uw[i]);
        }
        {
          Teuchos::TimeMonitor updatetimer(*worksetUpdateBasisHGTGTimer);
          DRV basis_grad_tmp("basis grad tmp",activeElem,numb,numip,dimension);
          FuncTools::HGRADtransformGRAD(basis_grad_tmp, jacobianInv, ref_basis_grad[i]);
          OrientTools::modifyBasisByOrientation(basis_grad_uw[i], basis_grad_tmp, orientation, basis_pointers[i].get());
        }
        {
          Teuchos::TimeMonitor updatetimer(*worksetUpdateBasisMMTimer);
          FuncTools::multiplyMeasure(basis_grad[i], wts, basis_grad_uw[i]);
        }
      }
      else if (basis_types[i] == "HVOL"){
        basis_uw[i] = ref_basis[i];
        {
          Teuchos::TimeMonitor updatetimer(*worksetUpdateBasisMMTimer);
          FuncTools::multiplyMeasure(basis[i], wts, ref_basis[i]);
        }
      }
      else if (basis_types[i] == "HDIV"){
        
        DRV basis_tmp("basis tmp",activeElem,numb,numip,dimension);
        FuncTools::HDIVtransformVALUE(basis_tmp, jacobian, jacobianDet, ref_basis[i]);
        //basis_uw[i] = basis_tmp;
        OrientTools::modifyBasisByOrientation(basis_uw[i], basis_tmp, orientation, basis_pointers[i].get());
        FuncTools::multiplyMeasure(basis[i], wts, basis_uw[i]);
        
        
        DRV basis_div_tmp("basis div tmp",activeElem,numb,numip);
        FuncTools::HDIVtransformDIV(basis_div_tmp, jacobianDet, ref_basis_div[i]);
        //basis_div_uw[i] = basis_div_tmp;
        {
          Teuchos::TimeMonitor dbgtimer(*worksetDebugTimer2);
          OrientTools::modifyBasisByOrientation(basis_div_uw[i], basis_div_tmp, orientation, basis_pointers[i].get());
        FuncTools::multiplyMeasure(basis_div[i], wts, basis_div_uw[i]);
        }
      }
      else if (basis_types[i] == "HCURL"){
        DRV basis_tmp("basis tmp",activeElem,numb,numip,dimension);
        FuncTools::HCURLtransformVALUE(basis_tmp, jacobianInv, ref_basis[i]);
        OrientTools::modifyBasisByOrientation(basis_uw[i], basis_tmp, orientation, basis_pointers[i].get());
        FuncTools::multiplyMeasure(basis[i], wts, basis_uw[i]);
        
        DRV basis_curl_tmp("basis curl tmp",activeElem,numb,numip,dimension);
        FuncTools::HCURLtransformCURL(basis_curl_tmp, jacobian, jacobianDet, ref_basis_curl[i]);
        OrientTools::modifyBasisByOrientation(basis_curl_uw[i], basis_curl_tmp, orientation, basis_pointers[i].get());
        FuncTools::multiplyMeasure(basis_curl[i], wts, basis_curl_uw[i]);
        
      }
    }
    
    for (size_t i=0; i<param_basis_pointers.size(); i++) {
      int numb = param_basis_pointers[i]->getCardinality();
      DRV basisvals("basisvals_Transformed",activeElem, numb, numip);
      FuncTools::HGRADtransformVALUE(basisvals, param_basis_ref[i]);
      OrientTools::modifyBasisByOrientation(param_basis[i], basisvals, orientation, param_basis_pointers[i].get());
      DRV basis_grad_tmp("basis grad tmp",activeElem,numb,numip,dimension);
      FuncTools::HGRADtransformGRAD(basis_grad_tmp, jacobianInv, param_basis_grad_ref[i]);
      OrientTools::modifyBasisByOrientation(param_basis_grad[i], basis_grad_tmp, orientation, param_basis_pointers[i].get());
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
// Update the nodes and the basis functions at the side ip
////////////////////////////////////////////////////////////////////////////////////

int workset::addSide(const DRV & nodes, const int & sidenum,
                     Kokkos::View<LO*,AssemblyDevice> & localSideID,
                     Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & orientation) {
  
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
  int numBElem = nodes.extent(0);
  
  DRV bip("bip", numBElem, ref_side_ip.extent(0), dimension);
  DRV bijac("bijac", numBElem, ref_side_ip.extent(0), dimension, dimension);
  DRV bijacDet("bijacDet", numBElem, ref_side_ip.extent(0));
  DRV bijacInv("bijacInv", numBElem, ref_side_ip.extent(0), dimension, dimension);
  DRV bwts("wts_side", numBElem, ref_side_ip.extent(0));
  DRV bnormals("normals", numBElem, ref_side_ip.extent(0), dimension);
  DRV btangents("tangents", numBElem, ref_side_ip.extent(0), dimension);
  
  DRV refSidePoints = ref_side_ip_vec[localSideID(0)];
  {
    //Teuchos::TimeMonitor dbgtimer(*worksetDebugTimer0);
    CellTools::mapToPhysicalFrame(bip, refSidePoints, nodes, *celltopo);
    CellTools::setJacobian(bijac, refSidePoints, nodes, *celltopo);
    CellTools::setJacobianInv(bijacInv, bijac);
    CellTools::setJacobianDet(bijacDet, bijac);
  }
  
  {
    //Teuchos::TimeMonitor dbgtimer(*worksetDebugTimer1);
    
    if (dimension == 2) {
      Intrepid2::RealSpaceTools<AssemblyExec>::matvec(btangents, bijac, ref_side_tangents_vec[localSideID(0)]);
      
      DRV rotation("rotation matrix",dimension,dimension);
      rotation(0,0) = 0;  rotation(0,1) = 1;
      rotation(1,0) = -1; rotation(1,1) = 0;
      Intrepid2::RealSpaceTools<AssemblyExec>::matvec(bnormals, rotation, btangents);
      
      Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(bwts, btangents, Intrepid2::NORM_TWO);
      Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(bwts, bwts, ref_side_wts);
      
    }
    else if (dimension == 3) {
      
      DRV faceTanU("face tangent U", numBElem, ref_side_ip.extent(0), dimension);
      DRV faceTanV("face tangent V", numBElem, ref_side_ip.extent(0), dimension);
      
      Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanU, bijac, ref_side_tangentsU_vec[localSideID(0)]);
      Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanV, bijac, ref_side_tangentsV_vec[localSideID(0)]);
      
      Intrepid2::RealSpaceTools<AssemblyExec>::vecprod(bnormals, faceTanU, faceTanV);
      
      Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(bwts, bnormals, Intrepid2::NORM_TWO);
      Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(bwts, bwts, ref_side_wts);
      
    }
    
  }
  
  BID = ip_side_vec.size();
  
  // ----------------------------------------------------------
  // store ip
  ip_side_vec.push_back(bip);
  // ----------------------------------------------------------
  
  // ----------------------------------------------------------
  // store wts
  wts_side_vec.push_back(bwts);
  
  // ----------------------------------------------------------
  
  // scale the normal vector (we need unit normal...)
  parallel_for(RangePolicy<AssemblyExec>(0,bnormals.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int j=0; j<bnormals.extent(1); j++ ) {
      ScalarT normalLength = 0.0;
      for (int sd=0; sd<dimension; sd++) {
        normalLength += bnormals(e,j,sd)*bnormals(e,j,sd);
      }
      normalLength = sqrt(normalLength);
      for (int sd=0; sd<dimension; sd++) {
        bnormals(e,j,sd) = bnormals(e,j,sd) / normalLength;
      }
    }
  });
  
  // ----------------------------------------------------------
  // store normals
  normals_side_vec.push_back(bnormals);
  // ----------------------------------------------------------
  
  // ----------------------------------------------------------
  // evaluate basis vectors (just values and grads for now)
  
  vector<DRV> cbasis, cbasis_uw, cbasis_grad, cbasis_grad_uw;
  
  {
    //Teuchos::TimeMonitor dbgtimer(*worksetDebugTimer2);
  
  for (size_t i=0; i<basis_pointers.size(); i++) {
    if (basis_types[i] == "HGRAD"){
      int numb = basis_pointers[i]->getCardinality();
      
      DRV ref_basisvals("basisvals",numb, numsideip);
      basis_pointers[i]->getValues(ref_basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
      
      DRV basisvals_trans("basisvals_Transformed",numBElem, numb, numsideip);
      FuncTools::HGRADtransformVALUE(basisvals_trans, ref_basisvals);
      DRV basisvals_to("basisvals_Transformed",numBElem, numb, numsideip);
      OrientTools::modifyBasisByOrientation(basisvals_to, basisvals_trans, orientation, basis_pointers[i].get());
      
      cbasis_uw.push_back(basisvals_to);
      
      DRV basis_wtd("basis_side",numBElem,numb,numsideip);
      FuncTools::multiplyMeasure(basis_wtd, bwts, cbasis_uw[i]);
      cbasis.push_back(basis_wtd);
      
      DRV ref_basisgrad("basisgrad",numb, numsideip, dimension);
      basis_pointers[i]->getValues(ref_basisgrad, refSidePoints, Intrepid2::OPERATOR_GRAD);
      
      DRV basis_grad_trans("basis_grad_side_uw",numBElem,numb,numsideip,dimension);
      FuncTools::HGRADtransformGRAD(basis_grad_trans, bijacInv, ref_basisgrad);
      
      DRV basis_grad_to("basis_grad_side_uw",numBElem,numb,numsideip,dimension);
      OrientTools::modifyBasisByOrientation(basis_grad_to, basis_grad_trans, orientation, basis_pointers[i].get());
      
      cbasis_grad_uw.push_back(basis_grad_to);
      
      DRV basis_grad_wtd("basis_grad_side_uw",numBElem,numb,numsideip,dimension);
      FuncTools::multiplyMeasure(basis_grad_wtd, bwts, cbasis_grad_uw[i]);
      cbasis_grad.push_back(basis_grad_wtd);
    }
    else if (basis_types[i] == "HVOL"){ // does not require orientations
      int numb = basis_pointers[i]->getCardinality();
      
      DRV ref_basisvals("basisvals",numb, numsideip);
      basis_pointers[i]->getValues(ref_basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
      
      DRV basisvals_trans("basisvals_Transformed",numBElem, numb, numsideip);
      FuncTools::HGRADtransformVALUE(basisvals_trans, ref_basisvals);
      cbasis_uw.push_back(basisvals_trans);
      
      DRV basis_wtd("basis_side",numBElem,numb,numsideip);
      FuncTools::multiplyMeasure(basis_wtd, bwts, cbasis_uw[i]);
      cbasis.push_back(basis_wtd);
      
      DRV basis_grad_trans("basis_grad_side_uw",numBElem,numb,numsideip,dimension);
      cbasis_grad_uw.push_back(basis_grad_trans);
      
      DRV basis_grad_wtd("basis_grad_side_uw",numBElem,numb,numsideip,dimension);
      cbasis_grad.push_back(basis_grad_wtd);
      
    }
    else if (basis_types[i] == "HFACE"){
      int numb = basis_pointers[i]->getCardinality();
      
      DRV ref_basisvals("basisvals",numb, numsideip);
      basis_pointers[i]->getValues(ref_basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
      
      DRV basisvals_trans("basisvals_Transformed",numBElem, numb, numsideip);
      FuncTools::HGRADtransformVALUE(basisvals_trans, ref_basisvals);
      DRV basisvals_to("basisvals_Transformed",numBElem, numb, numsideip);
      OrientTools::modifyBasisByOrientation(basisvals_to, basisvals_trans, orientation, basis_pointers[i].get());
      
      cbasis_uw.push_back(basisvals_to);
      
      DRV basis_wtd("basis_side",numBElem,numb,numsideip);
      FuncTools::multiplyMeasure(basis_wtd, bwts, cbasis_uw[i]);
      cbasis.push_back(basis_wtd);
      
      DRV basis_grad_trans("basis_grad_side_uw",numBElem,numb,numsideip,dimension);
      cbasis_grad_uw.push_back(basis_grad_trans);
      
      DRV basis_grad_wtd("basis_grad_side_uw",numBElem,numb,numsideip,dimension);
      cbasis_grad.push_back(basis_grad_wtd);
      
    }
    else if (basis_types[i] == "HDIV"){
      int numb = basis_pointers[i]->getCardinality();
      
      DRV ref_basisvals("basisvals",numb, numsideip, dimension);
      basis_pointers[i]->getValues(ref_basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
      
      DRV basisvals_trans("basisvals_Transformed",numBElem, numb, numsideip, dimension);
      
      FuncTools::HDIVtransformVALUE(basisvals_trans, bijac, bijacDet, ref_basisvals);
      DRV basisvals_to("basisvals_Transformed",numBElem, numb, numsideip, dimension);
      OrientTools::modifyBasisByOrientation(basisvals_to, basisvals_trans, orientation, basis_pointers[i].get());
      
      DRV basis_wtd("basis_side",numBElem,numb,numsideip,dimension);
      FuncTools::multiplyMeasure(basis_wtd, bwts, basisvals_to);
      
      cbasis_uw.push_back(basisvals_to);
      cbasis.push_back(basis_wtd);
      
      DRV basis_grad_trans("basis_grad_side_uw",numBElem,numb,numsideip,dimension);
      cbasis_grad_uw.push_back(basis_grad_trans);
      
      DRV basis_grad_wtd("basis_grad_side_uw",numBElem,numb,numsideip,dimension);
      cbasis_grad.push_back(basis_grad_wtd);
      //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
    }
    else if (basis_types[i] == "HCURL"){
      //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
    }
  }
  }
  // ----------------------------------------------------------
  
  // ----------------------------------------------------------
  // store basis values
  basis_side_vec.push_back(cbasis);
  basis_side_uw_vec.push_back(cbasis_uw);
  basis_grad_side_vec.push_back(cbasis_grad);
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
    param_basis_pointers[i]->getValues(basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
    DRV basisvals_trans("basisvals_Transformed",numBElem, numb, numsideip);
    FuncTools::HGRADtransformVALUE(basisvals_trans, basisvals);
    OrientTools::modifyBasisByOrientation(cpbasis[i], basisvals_trans, orientation, basis_pointers[i].get());
    
    DRV basisgrad("basisgrad",numb, numsideip, dimension);
    basis_pointers[i]->getValues(basisgrad, refSidePoints, Intrepid2::OPERATOR_GRAD);
    DRV basis_grad_trans("basisvals_Transformed",numBElem, numb, numsideip, dimension);
    
    FuncTools::HGRADtransformGRAD(basis_grad_trans, bijacInv, basisgrad);
    OrientTools::modifyBasisByOrientation(cpbasis_grad[i], basis_grad_trans, orientation, basis_pointers[i].get());
    
  }
  // ----------------------------------------------------------
  
  // ----------------------------------------------------------
  // store param basis values
  param_basis_side_vec.push_back(cpbasis);
  param_basis_grad_side_vec.push_back(cpbasis_grad);
  // ----------------------------------------------------------
  return BID;
}


////////////////////////////////////////////////////////////////////////////////////
// Update the nodes and the basis functions at the face ip
////////////////////////////////////////////////////////////////////////////////////

void workset::updateFace(const DRV & nodes, Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & orientation,
                         //const vector<vector<ScalarT> > & orientation,
                         const size_t & facenum) {
  
  // Step 1: fill in ip_side, wts_side and normals
  ip_side = DRV("side ip", numElem, ref_side_ip.extent(0), dimension);
  DRV jac("bijac", numElem, ref_side_ip.extent(0), dimension, dimension);
  DRV jacDet("bijacDet", numElem, ref_side_ip.extent(0));
  DRV jacInv("bijacInv", numElem, ref_side_ip.extent(0), dimension, dimension);
  wts_side = DRV("wts_side", numElem, ref_side_ip.extent(0));
  normals = DRV("normals", numElem, ref_side_ip.extent(0), dimension);
  DRV tangents("tangents", numElem, ref_side_ip.extent(0), dimension);
  DRV refSidePoints;//("refSidePoints", ref_side_ip.extent(0), dimension);
  
  {
    Teuchos::TimeMonitor updatetimer(*worksetFaceUpdateIPTimer);
    
    refSidePoints = ref_side_ip_vec[facenum];
    //CellTools::mapToReferenceSubcell(refSidePoints, ref_side_ip,
    //                                                 dimension-1, facenum, *celltopo);
    
    
    CellTools::mapToPhysicalFrame(ip_side, refSidePoints, nodes, *celltopo);
    CellTools::setJacobian(jac, refSidePoints, nodes, *celltopo);
    CellTools::setJacobianInv(jacInv, jac);
    CellTools::setJacobianDet(jacDet, jac);
    
    if (dimension == 2) {
      Intrepid2::RealSpaceTools<AssemblyExec>::matvec(tangents, jac, ref_side_tangents_vec[facenum]);
      
      DRV rotation("rotation matrix",dimension,dimension);
      rotation(0,0) = 0;  rotation(0,1) = 1;
      rotation(1,0) = -1; rotation(1,1) = 0;
      Intrepid2::RealSpaceTools<AssemblyExec>::matvec(normals, rotation, tangents);
      
      Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(wts_side, tangents, Intrepid2::NORM_TWO);
      Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(wts_side, wts_side, ref_side_wts);
      
    }
    else if (dimension == 3) {
      
      DRV faceTanU("face tangent U", numElem, ref_side_ip.extent(0), dimension);
      DRV faceTanV("face tangent V", numElem, ref_side_ip.extent(0), dimension);
      
      Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanU, jac, ref_side_tangentsU_vec[facenum]);
      Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanV, jac, ref_side_tangentsV_vec[facenum]);
      
      Intrepid2::RealSpaceTools<AssemblyExec>::vecprod(normals, faceTanU, faceTanV);
      
      Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(wts_side, normals, Intrepid2::NORM_TWO);
      Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(wts_side, wts_side, ref_side_wts);
      
    }
    
    // scale the normal vector (we need unit normal...)
    parallel_for(RangePolicy<AssemblyExec>(0,normals.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int j=0; j<normals.extent(1); j++ ) {
        ScalarT normalLength = 0.0;
        for (int sd=0; sd<dimension; sd++) {
          normalLength += normals(e,j,sd)*normals(e,j,sd);
        }
        normalLength = sqrt(normalLength);
        for (int sd=0; sd<dimension; sd++) {
          normals(e,j,sd) = normals(e,j,sd) / normalLength;
        }
      }
    });
    
    
    parallel_for(RangePolicy<AssemblyExec>(0,normals.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<numsideip; j++) {
        for (size_t k=0; k<dimension; k++) {
          ip_side_KV(e,j,k) = ip_side(e,j,k);
          normals_KV(e,j,k) = normals(e,j,k);
        }
      }
    });
    
  }
  
  // Step 2: define basis functionsat these integration points
  {
    Teuchos::TimeMonitor updatetimer(*worksetFaceUpdateBasisTimer);
    
    for (size_t i=0; i<basis_pointers.size(); i++) {
      int numb = basis_pointers[i]->getCardinality();
      
      DRV basis_trans("basis_side_uw",numElem,numb,numsideip);
      DRV basis_wtd("basis_side",numElem,numb,numsideip);
      DRV basis_grad_trans("basis_grad_side_uw",numElem,numb,numsideip,dimension);
      DRV basis_grad_wtd("basis_grad_side_uw",numElem,numb,numsideip,dimension);
      
      if (basis_types[i] == "HGRAD"){
        
        DRV ref_basisvals("basisvals",numb, numsideip);
        basis_pointers[i]->getValues(ref_basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
        
        DRV basis_tmp("basisvals_Transformed",numElem, numb, numsideip);
        FuncTools::HGRADtransformVALUE(basis_tmp, ref_basisvals);
        OrientTools::modifyBasisByOrientation(basis_trans, basis_tmp, orientation, basis_pointers[i].get());
        
        FuncTools::multiplyMeasure(basis_wtd, wts_side, basis_trans);
        
        DRV ref_basisgrad("basisgrad",numb, numsideip, dimension);
        basis_pointers[i]->getValues(ref_basisgrad, refSidePoints, Intrepid2::OPERATOR_GRAD);
        DRV basis_grad_tmp("basisvals_Transformed",numElem, numb, numsideip, dimension);
        FuncTools::HGRADtransformGRAD(basis_grad_tmp, jacInv, ref_basisgrad);
        OrientTools::modifyBasisByOrientation(basis_grad_trans, basis_grad_tmp, orientation, basis_pointers[i].get());
        FuncTools::multiplyMeasure(basis_grad_wtd, wts_side, basis_grad_trans);
        
      }
      else if (basis_types[i] == "HVOL"){
        
        DRV ref_basisvals("basisvals",numb, numsideip);
        basis_pointers[i]->getValues(ref_basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
        FuncTools::HGRADtransformVALUE(basis_trans, ref_basisvals);
        FuncTools::multiplyMeasure(basis_wtd, wts_side, basis_trans);
        
      }
      else if (basis_types[i] == "HDIV"){
        
        DRV ref_basisvals("basisvals",numb, numsideip, dimension);
        basis_pointers[i]->getValues(ref_basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
        
        DRV basis_tmp("basisvals_Transformed",numElem, numb, numsideip, dimension);
        basis_trans = DRV("basisvals_Transformed",numElem, numb, numsideip, dimension);
        
        FuncTools::HDIVtransformVALUE(basis_tmp, jac, jacDet, ref_basisvals);
        OrientTools::modifyBasisByOrientation(basis_trans, basis_tmp, orientation, basis_pointers[i].get());
        
        basis_wtd = DRV("basis_side",numElem,numb,numsideip,dimension);
        FuncTools::multiplyMeasure(basis_wtd, wts_side, basis_trans);
        
        
      }
      else if (basis_types[i] == "HCURL"){
        //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
      }
      else if (basis_types[i] == "HFACE"){
        
        DRV ref_basisvals("basisvals",numb, numsideip);
        basis_pointers[i]->getValues(ref_basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
        
        
        DRV basis_tmp("basisvals_Transformed",numElem, numb, numsideip);
        FuncTools::HGRADtransformVALUE(basis_tmp, ref_basisvals);
        OrientTools::modifyBasisByOrientation(basis_trans, basis_tmp, orientation, basis_pointers[i].get());
        
        //FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basis_trans, ref_basisvals);
        FuncTools::multiplyMeasure(basis_wtd, wts_side, basis_trans);
        
      }
      
      basis_face_uw[i] = basis_trans;
      basis_face[i] = basis_wtd;
      basis_grad_face_uw[i] = basis_grad_trans;
      basis_grad_face[i] = basis_grad_wtd;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Update the nodes and the basis functions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::updateSide(const int & sidenum, const int & cnum) {
  
  currentside = sidenum;
  
  {
    
    //Teuchos::TimeMonitor updatetimer(*worksetSideUpdateIPTimer);
  
    ip_side = ip_side_vec[cnum];
    wts_side = wts_side_vec[cnum];
    normals = normals_side_vec[cnum];
    {
      Teuchos::TimeMonitor updatetimer(*worksetSideUpdateIPTimer);
      
    
    parallel_for(RangePolicy<AssemblyExec>(0,normals.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (size_t j=0; j<numsideip; j++) {
        for (size_t k=0; k<dimension; k++) {
          ip_side_KV(e,j,k) = ip_side(e,j,k);
          normals_KV(e,j,k) = normals(e,j,k);
        }
      }
    });
      
    }
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
  parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<res.extent(1); n++) {
      res(e,n) = 0.0;
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetResidual(const int & numE) {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  
  parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<res.extent(1); n++) {
      res(e,n) = 0.0;
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetFlux() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  parallel_for(RangePolicy<AssemblyExec>(0,flux.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<flux.extent(1); n++) {
      for (int k=0; k<flux.extent(2); k++) {
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
  parallel_for(RangePolicy<AssemblyExec>(0,local_aux.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<local_aux.extent(1); n++) {
      for (int k=0; k<local_aux.extent(2); k++) {
        local_aux(e,n,k) = 0.0;
        //for (int s=0; s<local_aux_grad.extent(3); s++) {
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
  parallel_for(RangePolicy<AssemblyExec>(0,local_aux_side.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<local_aux_side.extent(1); n++) {
      for (int k=0; k<local_aux_side.extent(2); k++) {
        local_aux_side(e,n,k) = 0.0;
        //for (int s=0; s<local_aux_grad_side.extent(3); s++) {
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
  parallel_for(RangePolicy<AssemblyExec>(0,adjrhs.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<adjrhs.extent(1); n++) {
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
    parallel_for(RangePolicy<AssemblyExec>(0,local_soln.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_soln.extent(1); k++) {
        for (int i=0; i<local_soln.extent(2); i++) {
          for (int s=0; s<local_soln.extent(3); s++) {
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
        
        for (int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u(e,k,i);
            for (size_t j=0; j<numip; j++ ) {
              local_soln(e,k,j,0) += uval*kbasis_uw(e,i,j);
              for (int s=0; s<dimension; s++ ) {
                local_soln_grad(e,k,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HVOL") {
        DRV kbasis_uw = basis_uw[kubasis];
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u(e,k,i);
            for( size_t j=0; j<numip; j++ ) {
              local_soln(e,k,j,0) += uval*kbasis_uw(e,i,j);
            }
          }
        }
      }
      else if (basis_types[usebasis[k]] == "HDIV"){
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_div_uw = basis_div_uw[kubasis];
        
        for (int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u(e,k,i);
            for (size_t j=0; j<numip; j++ ) {
              for (int s=0; s<dimension; s++ ) {
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
        
        for (int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u(e,k,i);
            for (size_t j=0; j<numip; j++ ) {
              for (int s=0; s<dimension; s++ ) {
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
                               Kokkos::View<ScalarT****,AssemblyDevice> u_prev,
                               Kokkos::View<int*,UnifiedDevice> seedwhat) {
  
  // Reset the values (may combine with next loop when parallelized)
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    parallel_for(RangePolicy<AssemblyExec>(0,local_soln.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<local_soln.extent(1); k++) {
        for (int i=0; i<local_soln.extent(2); i++) {
          for (int s=0; s<local_soln.extent(3); s++) {
            local_soln(e,k,i,s) = value;
            local_soln_dot(e,k,i,s) = value;
            local_soln_grad(e,k,i,s) = value;
            local_soln_curl(e,k,i,s) = value;
          }
          local_soln_div(e,k,i) = value;
        }
      }
    });
    
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnVolTimer);
    Kokkos::View<int*,UnifiedDevice> bind("basis index",1);
    
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      int knbasis = numbasis[kubasis];
      string kutype = basis_types[kubasis];
      bind(0) = k;
      
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_grad_uw = basis_grad_uw[kubasis];
        
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for (int i=0; i<kbasis_uw.extent(1); i++ ) {
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            AD u_dotval = soldot_wts(0)*uval;
            if (soldot_wts.extent(0)>1) {
              for (int s=1; s<soldot_wts.extent(0); s++) {
                u_dotval += soldot_wts(s)*u_prev(e,kk,i,s-1);
              }
            }
            if (seedwhat(0) == 2) {
              ScalarT val = u_dotval.val();
              u_dotval = AD(maxDerivs,offsets(kk,i),val);
            }
            
            for (size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              local_soln(e,kk,j,0) += uval*kbasis_uw(e,i,j);
              local_soln_dot(e,kk,j,0) += u_dotval*kbasis_uw(e,i,j);
              for (int s=0; s<kbasis_grad_uw.extent(3); s++ ) {
                local_soln_grad(e,kk,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        });
      }
      else if (kutype == "HVOL") {
        DRV kbasis_uw = basis_uw[kubasis];
        
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for( int i=0; i<kbasis_uw.extent(1); i++ ) {
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            AD u_dotval = soldot_wts(0)*uval;
            if (soldot_wts.extent(0)>1) {
              for (int s=1; s<soldot_wts.extent(0); s++) {
                u_dotval += soldot_wts(s)*u_prev(e,kk,i,s);
              }
            }
            
            for( size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              local_soln(e,kk,j,0) += uval*kbasis_uw(e,i,j);
              local_soln_dot(e,kk,j,0) += u_dotval*kbasis_uw(e,i,j);
            }
          }
        });
      }
      else if (kutype == "HDIV"){
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_div_uw = basis_div_uw[kubasis];
        
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for (int i=0; i<kbasis_uw.extent(1); i++ ) {
            
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            
            AD u_dotval = soldot_wts(0)*uval;
            if (soldot_wts.extent(0)>1) {
              for (int s=1; s<soldot_wts.extent(0); s++) {
                u_dotval += soldot_wts(s)*u_prev(e,kk,i,s);
              }
            }
            
            for (size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              for (int s=0; s<kbasis_uw.extent(3); s++ ) {
                local_soln(e,kk,j,s) += uval*kbasis_uw(e,i,j,s);
                local_soln_dot(e,kk,j,s) += u_dotval*kbasis_uw(e,i,j,s);
              }
              local_soln_div(e,kk,j) += uval*kbasis_div_uw(e,i,j);
            }
          }
        });
      }
      else if (basis_types[usebasis[k]] == "HCURL"){
        DRV kbasis_uw = basis_uw[kubasis];
        DRV kbasis_curl_uw = basis_curl_uw[kubasis];
        
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for (int i=0; i<kbasis_uw.extent(1); i++ ) {
            
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            
            AD u_dotval = soldot_wts(0)*uval;
            if (soldot_wts.extent(0)>1) {
              for (int s=1; s<soldot_wts.extent(0); s++) {
                u_dotval += soldot_wts(s)*u_prev(e,kk,i,s);
              }
            }
            
            for (size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              for (int s=0; s<kbasis_uw.extent(3); s++ ) {
                local_soln(e,kk,j,s) += uval*kbasis_uw(e,i,j,s);
                local_soln_dot(e,kk,j,s) += u_dotval*kbasis_uw(e,i,j,s);
                local_soln_curl(e,kk,j,s) += uval*kbasis_curl_uw(e,i,j,s);
              }
            }
          }
        });
      }
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the discretized parameters at the volumetric ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamVolIP(Kokkos::View<ScalarT***,AssemblyDevice> param,
                                Kokkos::View<int*,UnifiedDevice> seedwhat) {
  
  if (numParams > 0) {
    {
      Teuchos::TimeMonitor resettimer(*worksetResetTimer);
      // Reset the values (may combine with next loop when parallelized)
      parallel_for(RangePolicy<AssemblyExec>(0,local_param.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<local_param.extent(1); k++) {
          for (int i=0; i<local_param.extent(2); i++) {
            local_param(e,k,i) = 0.0;
            for (int s=0; s<local_param_grad.extent(3); s++) {
              local_param_grad(e,k,i,s) = 0.0;
            }
          }
        }
      });
    }
    
    {
      Teuchos::TimeMonitor basistimer(*worksetComputeParamVolTimer);
      Kokkos::View<int*,UnifiedDevice> bind("basis index",1);
      
      for (int k=0; k<numParams; k++) {
        int kpbasis = paramusebasis[k];
        
        DRV pbasis = param_basis[kpbasis];
        DRV pbasis_grad = param_basis_grad[kpbasis];
        bind(0) = k;
        
        parallel_for(RangePolicy<AssemblyExec>(0,pbasis.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD paramval;
          int kk = bind(0);
          for (int i=0; i<pbasis.extent(1); i++ ) {
            
            if (seedwhat(0) == 3) {
              paramval = AD(maxDerivs,paramoffsets(kk,i),param(e,kk,i));
            }
            else {
              paramval = param(e,kk,i);
            }
            for (size_t j=0; j<pbasis.extent(2); j++ ) {
              local_param(e,k,j) += paramval*pbasis(e,i,j);
              for (int s=0; s<pbasis_grad.extent(3); s++ ) {
                local_param_grad(e,kk,j,s) += paramval*pbasis_grad(e,i,j,s);
              }
            }
          }
        });
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnFaceIP(Kokkos::View<ScalarT***,AssemblyDevice> u,
                                Kokkos::View<int*,UnifiedDevice> seedwhat) {
  
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    // Reset the values (may combine with next loop when parallelized)
    parallel_for(RangePolicy<AssemblyExec>(0,local_soln_face.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_soln_face.extent(1); k++) {
        for (int i=0; i<local_soln_face.extent(2); i++) {
          for (int s=0; s<local_soln_face.extent(3); s++) {
            local_soln_face(e,k,i,s) = 0.0;
            local_soln_grad_face(e,k,i,s) = 0.0;
          }
        }
      }
    });
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    Kokkos::View<int*,UnifiedDevice> bind("basis index",1);
    
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      string kutype = basis_types[kubasis];
      bind(0) = k;
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_face_uw[kubasis];
        DRV kbasis_grad_uw = basis_grad_face_uw[kubasis];
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for (int i=0; i<kbasis_uw.extent(1); i++ ) {
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            for (size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              local_soln_face(e,kk,j,0) += uval*kbasis_uw(e,i,j);
              for (int s=0; s<kbasis_grad_uw.extent(3); s++ ) {
                local_soln_grad_face(e,kk,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        });
      }
      else if (kutype == "HVOL") {
        DRV kbasis_uw = basis_face_uw[kubasis];
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for( int i=0; i<kbasis_uw.extent(1); i++ ) {
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            for( size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              local_soln_face(e,kk,j,0) += uval*kbasis_uw(e,i,j);
            }
          }
        });
      }
      else if (kutype == "HDIV"){
        DRV kbasis_uw = basis_face_uw[kubasis];
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for( int i=0; i<kbasis_uw.extent(1); i++ ) {
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            for( size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              for (size_t s=0; s<kbasis_uw.extent(3); s++) {
                local_soln_face(e,kk,j,s) += uval*kbasis_uw(e,i,j,s);
              }
            }
          }
        });
      }
      else if (kutype == "HCURL"){
        
      }
      else if (kutype == "HFACE") {
        
        DRV kbasis_uw = basis_face_uw[kubasis];
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for( int i=0; i<kbasis_uw.extent(1); i++ ) {
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            for( size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              local_soln_face(e,kk,j,0) += uval*kbasis_uw(e,i,j);
            }
          }
        });
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSideIP(Kokkos::View<ScalarT***,AssemblyDevice> u,
                                Kokkos::View<int*,UnifiedDevice> seedwhat) {
  
  {// Reset the values (may combine with next loop when parallelized)
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    parallel_for(RangePolicy<AssemblyExec>(0,local_soln_side.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_soln_side.extent(1); k++) {
        for (int i=0; i<local_soln_side.extent(2); i++) {
          for (int s=0; s<local_soln_side.extent(3); s++) {
            local_soln_side(e,k,i,s) = 0.0;
            local_soln_grad_side(e,k,i,s) = 0.0;
          }
        }
      }
    });
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    Kokkos::View<int*,UnifiedDevice> bind("basis index",1);
    
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      string kutype = basis_types[kubasis];
      bind(0) = k;
      
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_side_uw[kubasis];
        DRV kbasis_grad_uw = basis_grad_side_uw[kubasis];
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for (int i=0; i<kbasis_uw.extent(1); i++ ) {
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            for (size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              local_soln_side(e,kk,j,0) += uval*kbasis_uw(e,i,j);
              for (int s=0; s<kbasis_grad_uw.extent(3); s++ ) {
                local_soln_grad_side(e,kk,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        });
      }
      else if (kutype == "HVOL") {
        DRV kbasis_uw = basis_side_uw[kubasis];
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for( int i=0; i<kbasis_uw.extent(1); i++ ) {
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            for( size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              local_soln_side(e,kk,j,0) += uval*kbasis_uw(e,i,j);
            }
          }
        });
      }
      else if (kutype == "HDIV"){
        DRV kbasis_uw = basis_side_uw[kubasis];
        parallel_for(RangePolicy<AssemblyExec>(0,kbasis_uw.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD uval;
          int kk = bind(0);
          for( int i=0; i<kbasis_uw.extent(1); i++ ) {
            if (seedwhat(0) == 1) {
              uval = AD(maxDerivs,offsets(kk,i),u(e,kk,i));
            }
            else {
              uval = u(e,kk,i);
            }
            for( size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              for (size_t s=0; s<kbasis_uw.extent(3); s++) {
                local_soln_side(e,kk,j,s) += uval*kbasis_uw(e,i,j,s);
              }
            }
          }
        });
      }
      else if (kutype == "HCURL"){
        
      }
      else if (kutype == "HFACE") {
       
      }
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the discretized parameters at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamSideIP(const int & side, Kokkos::View<ScalarT***,AssemblyDevice> param,
                                 Kokkos::View<int*,UnifiedDevice> seedwhat) {
  
  if (numParams>0) {
    {// reset the local params
      Teuchos::TimeMonitor resettimer(*worksetResetTimer);
      // Reset the values (may combine with next loop when parallelized)
      parallel_for(RangePolicy<AssemblyExec>(0,local_param_side.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<local_param_side.extent(1); k++) {
          for (int i=0; i<local_param_side.extent(2); i++) {
            local_param_side(e,k,i) = 0.0;
            for (int s=0; s<local_param_grad_side.extent(3); s++) {
              local_param_grad_side(e,k,i,s) = 0.0;
            }
          }
        }
      });
    }
    
    {
      Teuchos::TimeMonitor basistimer(*worksetComputeParamSideTimer);
      
      Kokkos::View<int*,UnifiedDevice> bind("basis index",1);
      
      for (int k=0; k<numParams; k++) {
        int kpbasis = paramusebasis[k];
        bind(0) = k;
        DRV pbasis = param_basis_side[kpbasis];
        DRV pbasis_grad = param_basis_grad_side[kpbasis];
        
        parallel_for(RangePolicy<AssemblyExec>(0,pbasis.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (int i=0; i<pbasis.extent(1); i++ ) {
            AD paramval;
            int kk = bind(0);
            if (seedwhat(0) == 3) {
              paramval = AD(maxDerivs,paramoffsets(kk,i),param(e,kk,i));
            }
            else {
              paramval = param(e,kk,i);
            }
            
            for (size_t j=0; j<pbasis.extent(2); j++ ) {
              local_param_side(e,kk,j) += paramval*pbasis(e,i,j);
              for (int s=0; s<pbasis_grad.extent(3); s++ ) {
                local_param_grad_side(e,kk,j,s) += paramval*pbasis_grad(e,i,j,s);
              }
            }
          }
        });
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSideIP(const int & side, Kokkos::View<AD***,AssemblyDevice> u_AD,
                                Kokkos::View<AD***,AssemblyDevice> param_AD) {
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    // Reset the values (may combine with next loop when parallelized)
    parallel_for(RangePolicy<AssemblyExec>(0,local_soln_side.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_soln_side.extent(1); k++) {
        for (int i=0; i<local_soln_side.extent(2); i++) {
          for (int s=0; s<local_soln_side.extent(3); s++) {
            local_soln_side(e,k,i,s) = 0.0;
            local_soln_grad_side(e,k,i,s) = 0.0;
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
        
        for (int i=0; i<knbasis; i++ ) {
          for (int e=0; e<kbasis_uw.extent(0); e++) {
            uval = u_AD(e,k,i);
            for (size_t j=0; j<numsideip; j++ ) {
              local_soln_side(e,k,j,0) += uval*kbasis_uw(e,i,j);
              for (int s=0; s<dimension; s++ ) {
                local_soln_grad_side(e,k,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HVOL") {
        DRV kbasis_uw = basis_side_uw[kubasis];
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u_AD(e,k,i);
            for( size_t j=0; j<numsideip; j++ ) {
              local_soln_side(e,k,j,0) += uval*kbasis_uw(e,i,j);
            }
          }
        }
      }
      else if (kutype == "HDIV"){
        DRV kbasis_uw = basis_side_uw[kubasis];
        for (int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u_AD(e,k,i);
            for (size_t j=0; j<numsideip; j++ ) {
              for (int s=0; s<dimension; s++ ) {
                local_soln_side(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HCURL"){
        DRV kbasis_uw = basis_side_uw[kubasis];
        
        for (int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u_AD(e,k,i);
            for (size_t j=0; j<numsideip; j++ ) {
              for (int s=0; s<dimension; s++ ) {
                local_soln_side(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
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
    parallel_for(RangePolicy<AssemblyExec>(0,local_param_side.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (int k=0; k<local_param_side.extent(1); k++) {
        for (int i=0; i<local_param_side.extent(2); i++) {
          local_param_side(e,k,i) = 0.0;
          //for (int s=0; s<local_param_grad_side.extent(3); s++) {
          //  local_param_grad_side(e,k,i,s) = 0.0;
          //}
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
      //DRV pbasis_grad = param_basis_grad[kpbasis];
      
      for (int i=0; i<knpbasis; i++ ) {
        for (int e=0; e<numElem; e++) {
          paramval = param_AD(e,k,i);
          for (size_t j=0; j<numsideip; j++ ) {
            local_param_side(e,k,j) += paramval*pbasis(e,i,j);
            //for (int s=0; s<dimension; s++ ) {
            //  local_param_grad_side(e,k,j,s) += paramval*pbasis_grad(e,i,j,s);
            //}
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
  flux = Kokkos::View<AD***,AssemblyDevice>("flux",numElem,numAux,numsideip);
  
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
