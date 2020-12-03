/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "workset.hpp"
using namespace MrHyDE;

////////////////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////////////////

workset::workset(const vector<int> & cellinfo, const bool & isTransient_,
                 const vector<string> & basis_types_,
                 const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
                 const topo_RCP & topo, Kokkos::View<int**,HostDevice> & var_bcs_) :
var_bcs(var_bcs_), isTransient(isTransient_), celltopo(topo),
basis_types(basis_types_), basis_pointers(basis_pointers_) {

  // Settings that should not change
  dimension = cellinfo[0];
  numVars = cellinfo[1];
  numParams = cellinfo[2];
  numAux = cellinfo[3];
  numElem = cellinfo[4];
  usebcs = true;
  numip = cellinfo[5];
  numsideip = cellinfo[6];
  if (dimension == 2) {
    numsides = celltopo->getSideCount();
  }
  else if (dimension == 3) {
    numsides = celltopo->getFaceCount();
  }
  
  deltat = 1.0;
  deltat_KV = Kokkos::View<ScalarT*,AssemblyDevice>("deltat",1);
  Kokkos::deep_copy(deltat_KV,deltat);
  
  current_stage_KV = Kokkos::View<int*,AssemblyDevice>("stage number on device",1);
  Kokkos::deep_copy(current_stage_KV,0);
  // Integration information
  time_KV = Kokkos::View<ScalarT*,AssemblyDevice>("time",1); // defaults to 0.0
  
  // these cannot point to different arrays ... data must be deep copied into them
  ip = Kokkos::View<ScalarT***,AssemblyDevice>("ip",numElem,numip,dimension);
  ip_side = Kokkos::View<ScalarT***,AssemblyDevice>("side ip",numElem,numsideip,dimension);
  normals = Kokkos::View<ScalarT***,AssemblyDevice>("side normals",numElem,numsideip,dimension);
  point = Kokkos::View<ScalarT***,AssemblyDevice>("point",1,1,dimension);
  res = Kokkos::View<AD**,AssemblyDevice>("residual",numElem, maxDerivs, maxDerivs);
  adjrhs = Kokkos::View<AD**,AssemblyDevice>("adjoint RHS",numElem, maxDerivs, maxDerivs);
  
  // these can point to different arrays
  wts = Kokkos::View<ScalarT**,AssemblyDevice>("ip wts",numElem,numip);
  wts_side = Kokkos::View<ScalarT**,AssemblyDevice>("ip side wts",numElem,numsideip);
    
  have_rotation = false;
  have_rotation_phi = false;
  rotation = Kokkos::View<ScalarT***,AssemblyDevice>("rotation matrix",numElem,3,3);
  
  int maxb = 0;
  for (size_t i=0; i<basis_pointers.size(); i++) {
    int numb = basis_pointers[i]->getCardinality();
    maxb = std::max(maxb,numb);
  }
  
  uvals = Kokkos::View<AD***,AssemblyDevice>("seeded uvals",numElem, numVars, maxb, maxDerivs);
  if (isTransient) {
    u_dotvals = Kokkos::View<AD***,AssemblyDevice>("seeded uvals",numElem, numVars, maxb, maxDerivs);
  }
}


////////////////////////////////////////////////////////////////////////////////////
// Public functions
////////////////////////////////////////////////////////////////////////////////////

void workset::createSolns() {

  for (size_t i=0; i<usebasis.size(); i++) {
    int bind = usebasis[i];
    if (basis_types[bind] == "HGRAD") {
      vars_HGRAD.push_back(i);
    }
    else if (basis_types[bind] == "HDIV") {
      vars_HDIV.push_back(i);
    }
    else if (basis_types[bind] == "HVOL") {
      vars_HVOL.push_back(i);
    }
    else if (basis_types[bind] == "HCURL") {
      vars_HCURL.push_back(i);
    }
    else if (basis_types[bind] == "HFACE") {
      vars_HFACE.push_back(i);
    }
  }
  
  // Local solutions that are always used
  local_soln = Kokkos::View<AD****, AssemblyDevice>("local_soln",numElem, numVars, numip, dimension, maxDerivs);
  local_soln_side = Kokkos::View<AD****, AssemblyDevice>("local_soln_side",numElem, numVars, numsideip, dimension, maxDerivs);
  local_soln_face = Kokkos::View<AD****, AssemblyDevice>("local_soln_face",numElem, numVars, numsideip, dimension, maxDerivs);
  local_soln_point = Kokkos::View<AD****, AssemblyDevice>("local_soln point",1, numVars, 1, dimension, maxDerivs);
  local_soln_dot = Kokkos::View<AD****, AssemblyDevice>("local_soln_dot",numElem, numVars, numip, dimension, maxDerivs);
  
  if (vars_HGRAD.size() > 0) {
    local_soln_grad = Kokkos::View<AD****, AssemblyDevice>("local_soln_grad",numElem, numVars, numip, dimension, maxDerivs);
    local_soln_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_soln_grad_side",numElem, numVars, numsideip, dimension, maxDerivs);
    local_soln_grad_face = Kokkos::View<AD****, AssemblyDevice>("local_soln_grad_face",numElem, numVars, numsideip, dimension, maxDerivs);
    local_soln_grad_point = Kokkos::View<AD****, AssemblyDevice>("local_soln point",1, numVars, 1, dimension, maxDerivs);
  }
  if (vars_HDIV.size() > 0) {
    local_soln_div = Kokkos::View<AD***, AssemblyDevice>("local_soln_div",numElem, numVars, numip, maxDerivs);
  }
  if (vars_HCURL.size() > 0) {
    local_soln_curl = Kokkos::View<AD****, AssemblyDevice>("local_soln_curl",numElem, numVars, numip, dimension, maxDerivs);
  }
  
  if (paramusebasis.size()>0) {
    for (size_t i=0; i<paramusebasis.size(); i++) {
      int bind = paramusebasis[i];
      if (basis_types[bind] == "HGRAD") {
        paramvars_HGRAD.push_back(i);
      }
      else if (basis_types[bind] == "HDIV") {
        paramvars_HDIV.push_back(i);
      }
      else if (basis_types[bind] == "HVOL") {
        paramvars_HVOL.push_back(i);
      }
      else if (basis_types[bind] == "HCURL") {
        paramvars_HCURL.push_back(i);
      }
    }
    local_param = Kokkos::View<AD****, AssemblyDevice>("local_param",numElem, numParams, numip, dimension, maxDerivs);
    local_param_side = Kokkos::View<AD****, AssemblyDevice>("local_param_side",numElem, numParams, numsideip, dimension, maxDerivs);
    local_param_point = Kokkos::View<AD****, AssemblyDevice>("local_soln point",1, numParams, 1, dimension, maxDerivs);
    
    if (paramvars_HGRAD.size() > 0) {
      local_param_grad = Kokkos::View<AD****, AssemblyDevice>("local_param_grad",numElem, numParams, numip, dimension, maxDerivs);
      local_param_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_param_grad_side",numElem, numParams, numsideip, dimension, maxDerivs);
      local_param_grad_point = Kokkos::View<AD****, AssemblyDevice>("local_param grad point",1, numParams, 1, dimension, maxDerivs);
    }
    if (paramvars_HDIV.size() > 0) {
      local_param_div = Kokkos::View<AD***, AssemblyDevice>("local_param_div",numElem, numVars, numip, maxDerivs);
    }
    if (paramvars_HCURL.size() > 0) {
      local_param_curl = Kokkos::View<AD****, AssemblyDevice>("local_param_curl",numElem, numVars, numip, dimension, maxDerivs);
    }
    pvals = Kokkos::View<AD***,AssemblyDevice>("seeded uvals",numElem, paramusebasis.size(), uvals.extent(2), maxDerivs);
  }
  
  if (numAux>0) {
    local_aux = Kokkos::View<AD***, AssemblyDevice>("local_aux",numElem, numAux, numip, maxDerivs);
    local_aux_side = Kokkos::View<AD***, AssemblyDevice>("local_aux_side",numElem, numAux, numsideip, maxDerivs);
  }
  
  // Arrays that are not currently used for anything
  /*
  local_aux_grad = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad",numElem, numAux, numip, dimension);
  local_aux_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad_side",numElem, numAux, numsideip, dimension);
  local_soln_dot_grad = Kokkos::View<AD****, AssemblyDevice>("local_soln_dot_grad",numElem, numVars, numip, dimension);
  local_soln_dot_side = Kokkos::View<AD****, AssemblyDevice>("local_soln_dot_side",numElem, numVars, numsideip, dimension);
  */
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetResidual() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  Kokkos::deep_copy(res,0.0);
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetResidual(const int & numE) {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  Kokkos::deep_copy(res,0.0);
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
// TMW: I believe this can be deprecated
////////////////////////////////////////////////////////////////////////////////////

void workset::resetFlux() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  Kokkos::deep_copy(flux,0.0);
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetAux() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  Kokkos::deep_copy(local_aux,0.0);
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetAuxSide() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  Kokkos::deep_copy(local_aux_side,0.0);
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetAdjointRHS() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  Kokkos::deep_copy(adjrhs,0.0);
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the volumetric ip
////////////////////////////////////////////////////////////////////////////////////

// TMW: This function should be deprecated
void workset::computeSolnVolIP(Kokkos::View<ScalarT***,AssemblyDevice> u) {
  
  // Reset the values
  {
    Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    Kokkos::deep_copy(local_soln,0.0);
    Kokkos::deep_copy(local_soln_grad,0.0);
    Kokkos::deep_copy(local_soln_curl,0.0);
    Kokkos::deep_copy(local_soln_div,0.0);
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnVolTimer);
    AD uval = 0.0;
    for (size_t k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      int knbasis = numbasis[kubasis];
      string kutype = basis_types[kubasis];
      
      if (kutype == "HGRAD") {
        auto kbasis_uw = basis[kubasis];
        auto kbasis_grad_uw = basis_grad[kubasis];
        
        for (int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u(e,k,i);
            for (size_t j=0; j<numip; j++ ) {
              local_soln(e,k,j,0) += uval*kbasis_uw(e,i,j,0);
              for (int s=0; s<dimension; s++ ) {
                local_soln_grad(e,k,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HVOL") {
        auto kbasis_uw = basis[kubasis];
        for( int i=0; i<knbasis; i++ ) {
          for (int e=0; e<numElem; e++) {
            uval = u(e,k,i);
            for( size_t j=0; j<numip; j++ ) {
              local_soln(e,k,j,0) += uval*kbasis_uw(e,i,j,0);
            }
          }
        }
      }
      else if (basis_types[usebasis[k]] == "HDIV"){
        auto kbasis_uw = basis[kubasis];
        auto kbasis_div_uw = basis_div[kubasis];
        
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
        auto kbasis_uw = basis[kubasis];
        auto kbasis_curl_uw = basis_curl[kubasis];
        
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
// Compute the seeded solutions for general transient problems
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnTransientSeeded(Kokkos::View<ScalarT***,AssemblyDevice> u,
                                         Kokkos::View<ScalarT****,AssemblyDevice> u_prev,
                                         Kokkos::View<ScalarT****,AssemblyDevice> u_stage,
                                         const int & seedwhat,
                                         const int & index) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
  // These need to be set locally to be available to AssemblyDevice
  auto u_AD = uvals;
  auto u_dot_AD = u_dotvals;
  auto off = offsets;
  auto dt = deltat_KV;
  auto curr_stage = current_stage_KV;
  auto b_A = butcher_A;
  auto b_b = butcher_b;
  auto b_c = butcher_c;
  auto BDF = BDF_wts;

  // Seed the current stage solution
  if (seedwhat == 1) {
    parallel_for("wkset transient soln 1",MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{u.extent(0),u.extent(1),u.extent(2)}), KOKKOS_LAMBDA (const size_type elem, const size_type var, const size_type dof ) {
      
      ScalarT beta_u, beta_t;
      int stage = curr_stage(0);
      ScalarT deltat = dt(0);
      ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
      ScalarT timewt = 1.0/deltat/b_b(stage);
      ScalarT alpha_t = BDF(0)*timewt;
      //for (size_type var=0; var<u.extent(1); var++ ) {
      //  for (size_type dof=0; dof<u.extent(2); dof++ ) {
          // Seed the stage solution
          AD stageval = AD(maxDerivs,off(var,dof),u(elem,var,dof));
          
          // Compute the evaluating solution
          beta_u = (1.0-alpha_u)*u_prev(elem,var,dof,0);
          for (int s=0; s<stage; s++) {
            beta_u += b_A(stage,s)/b_b(s) * (u_stage(elem,var,dof,s) - u_prev(elem,var,dof,0));
          }
          u_AD(elem,var,dof) = alpha_u*stageval+beta_u;
          
          // Compute the time derivative
          beta_t = 0.0;
          for (size_type s=1; s<BDF.extent(0); s++) {
            beta_t += BDF(s)*u_prev(elem,var,dof,s-1);
          }
          beta_t *= timewt;
          u_dot_AD(elem,var,dof) = alpha_t*stageval + beta_t;
        //}
      //}
    });
  }
  else if (seedwhat == 2) { // Seed one of the previous step solutions
    Kokkos::View<size_t[1],AssemblyDevice> sindex("seed index on device");
    auto host_sindex = Kokkos::create_mirror_view(sindex);
    host_sindex(0) = index;
    Kokkos::deep_copy(sindex,host_sindex);
    parallel_for("wkset transient soln 2",MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{u.extent(0),u.extent(1),u.extent(2)}), KOKKOS_LAMBDA (const size_type elem, const size_type var, const size_type dof ) {
      AD beta_u, beta_t;
      int stage = curr_stage(0);
      ScalarT deltat = dt(0);
      ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
      ScalarT timewt = 1.0/deltat/b_b(stage);
      ScalarT alpha_t = BDF(0)*timewt;
      //for (size_type var=0; var<u.extent(1); var++ ) {
      //  for (size_type dof=0; dof<u.extent(2); dof++ ) {
          // Get the stage solution
          ScalarT stageval = u(elem,var,dof);
          
          // Compute the evaluating solution
          AD u_prev_val = u_prev(elem,var,dof,0);
          if (sindex(0) == 0) {
            u_prev_val = AD(maxDerivs,off(var,dof),u_prev(elem,var,dof,0));
          }
          
          beta_u = (1.0-alpha_u)*u_prev_val;
          for (int s=0; s<stage; s++) {
            beta_u += b_A(stage,s)/b_b(s) * (u_stage(elem,var,dof,s) - u_prev_val);
          }
          u_AD(elem,var,dof) = alpha_u*stageval+beta_u;
          
          // Compute and seed the time derivative
          beta_t = 0.0;
          for (size_type s=1; s<BDF.extent(0); s++) {
            AD u_prev_val = u_prev(elem,var,dof,s-1);
            if (sindex(0) == (s-1)) {
              u_prev_val = AD(maxDerivs,off(var,dof),u_prev(elem,var,dof,s-1));
            }
            beta_t += BDF(s)*u_prev_val;
          }
          beta_t *= timewt;
          u_dot_AD(elem,var,dof) = alpha_t*stageval + beta_t;
      //  }
      //}
    });
  }
  else if (seedwhat == 3) { // Seed one of the previous stage solutions
    Kokkos::View<int[1],AssemblyDevice> sindex("seed index on device");
    auto host_sindex = Kokkos::create_mirror_view(sindex);
    host_sindex(0) = index;
    Kokkos::deep_copy(sindex,host_sindex);
    parallel_for("wkset transient soln 2",MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{u.extent(0),u.extent(1),u.extent(2)}), KOKKOS_LAMBDA (const size_type elem, const size_type var, const size_type dof ) {
      AD beta_u, beta_t;
      int stage = curr_stage(0);
      ScalarT deltat = dt(0);
      ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
      ScalarT timewt = 1.0/deltat/b_b(stage);
      ScalarT alpha_t = BDF(0)*timewt;
      //for (size_type var=0; var<u.extent(1); var++ ) {
      //  for (size_type dof=0; dof<u.extent(2); dof++ ) {
          // Get the stage solution
          ScalarT stageval = u(elem,var,dof);
          
          // Compute the evaluating solution
          ScalarT u_prev_val = u_prev(elem,var,dof,0);
          
          beta_u = (1.0-alpha_u)*u_prev_val;
          for (int s=0; s<stage; s++) {
            AD u_stage_val = u_stage(elem,var,dof,s);
            if (sindex(0) == s) {
              u_stage_val = AD(maxDerivs,off(var,dof),u_stage(elem,var,dof,s));
            }
            beta_u += b_A(stage,s)/b_b(s) * (u_stage_val - u_prev_val);
          }
          u_AD(elem,var,dof) = alpha_u*stageval+beta_u;
          
          // Compute and seed the time derivative
          beta_t = 0.0;
          for (size_type s=1; s<BDF.extent(0); s++) {
            ScalarT u_prev_val = u_prev(elem,var,dof,s-1);
            beta_t += BDF(s)*u_prev_val;
          }
          beta_t *= timewt;
          u_dot_AD(elem,var,dof) = alpha_t*stageval + beta_t;
      //  }
      //}
    });
  }
  else { // Seed nothing
    parallel_for("wkset transient soln",MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{u.extent(0),u.extent(1),u.extent(2)}), KOKKOS_LAMBDA (const size_type elem , const size_type var, const size_type dof) {
      
      ScalarT beta_u, beta_t;
      int stage = curr_stage(0);
      ScalarT deltat = dt(0);
      ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
      ScalarT timewt = 1.0/deltat/b_b(stage);
      ScalarT alpha_t = BDF(0)*timewt;
      //for (size_type var=0; var<u.extent(1); var++ ) {
      //  for (size_type dof=0; dof<u.extent(2); dof++ ) {
          // Get the stage solution
          ScalarT stageval = u(elem,var,dof);
          
          // Compute the evaluating solution
          beta_u = (1.0-alpha_u)*u_prev(elem,var,dof,0);
          for (int s=0; s<stage; s++) {
            beta_u += b_A(stage,s)/b_b(s) * (u_stage(elem,var,dof,s) - u_prev(elem,var,dof,0));
          }
          u_AD(elem,var,dof) = alpha_u*stageval+beta_u;
          
          // Compute the time derivative
          beta_t = 0.0;
          for (size_type s=1; s<BDF.extent(0); s++) {
            beta_t += BDF(s)*u_prev(elem,var,dof,s-1);
          }
          beta_t *= timewt;
          u_dot_AD(elem,var,dof) = alpha_t*stageval + beta_t;
      //  }
      //}
    });
  }
  Kokkos::fence();         
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for steady-state problems
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSteadySeeded(Kokkos::View<ScalarT***,AssemblyDevice> u,
                                      const int & seedwhat) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);

  // Needed so the device can access data-members (may be a better way)
  auto u_AD = uvals;
  auto off = offsets;

  if (seedwhat == 1) {
    parallel_for("wkset steady soln",MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{u.extent(0),u.extent(1),u.extent(2)}), KOKKOS_LAMBDA (const size_type elem, const size_type var, const size_type dof ) {
      u_AD(elem,var,dof) = AD(maxDerivs,off(var,dof),u(elem,var,dof));
    });
  }
  else {
    parallel_for("wkset steady soln",MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{u.extent(0),u.extent(1),u.extent(2)}), KOKKOS_LAMBDA (const size_type elem, const size_type var, const size_type dof) {
      u_AD(elem,var,dof) = u(elem,var,dof);
    });
  }
  Kokkos::fence();
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for steady-state problems
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamSteadySeeded(Kokkos::View<ScalarT***,AssemblyDevice> param,
                                      const int & seedwhat) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
  // Needed so the device can access data-members (may be a better way)
  auto p_AD = pvals;
  auto off = paramoffsets;
  
  if (seedwhat == 3) { // There is an inconsistency here with seeded the stage solutions ... not sure if this matters
    parallel_for("wkset param seeded",MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{param.extent(0),param.extent(1),param.extent(2)}), KOKKOS_LAMBDA (const size_type elem, const size_type var, const size_type dof ) {
      //for (size_type var=0; var<param.extent(1); var++ ) {
      //  for (size_type dof=0; dof<param.extent(2); dof++ ) {
          p_AD(elem,var,dof) = AD(maxDerivs,off(var,dof),param(elem,var,dof));
      //  }
      //}
    });
  }
  else {
    parallel_for("wkset param unseeded",MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{param.extent(0),param.extent(1),param.extent(2)}), KOKKOS_LAMBDA (const size_type elem, const size_type var, const size_type dof ) {
      //for (size_type var=0; var<param.extent(1); var++ ) {
      //  for (size_type dof=0; dof<param.extent(2); dof++ ) {
          p_AD(elem,var,dof) = param(elem,var,dof);
      //  }
      //}
    });
  }
  Kokkos::fence();
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions at volumetric ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnVolIP() {
  this->computeSoln(1);
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions at specified ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSoln(const int & type) {
    
  // type specifications: 1 - volume; 2 - boundary; 3 - face
    
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnVolTimer);
    
    /////////////////////////////////////////////////////////////////////
    // HGRAD
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HGRAD.size(); i++) {
      int var = vars_HGRAD[i];
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis, cbasis_grad;
      Kokkos::View<AD**, Kokkos::LayoutStride, AssemblyDevice> csol;
      Kokkos::View<AD***, Kokkos::LayoutStride, AssemblyDevice> csol_grad;
      auto cuvals = Kokkos::subview(uvals,Kokkos::ALL(),var,Kokkos::ALL());
      
      if (type == 1) { // volumetric ip
        csol = Kokkos::subview(local_soln,Kokkos::ALL(),var,Kokkos::ALL(),0);
        csol_grad = Kokkos::subview(local_soln_grad,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis[usebasis[var]];
        cbasis_grad = basis_grad[usebasis[var]];
      }
      else if (type == 2) { // boundary ip
        csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
        csol_grad = Kokkos::subview(local_soln_grad_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis_side[usebasis[var]];
        cbasis_grad = basis_grad_side[usebasis[var]];
      }
      else if (type == 3) { // face ip
        csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),0);
        csol_grad = Kokkos::subview(local_soln_grad_face,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis_face[usebasis[var]];
        cbasis_grad = basis_grad_face[usebasis[var]];
      }
      
      
      /*
      parallel_for("wkset soln ip HGRAD",
                   MDRangePolicy<AssemblyExec,Rank<2>>({0,0},{csol.extent(0),csol.extent(1)}),
                   KOKKOS_LAMBDA (const size_type elem , const size_type pt) {
        csol(elem,pt) = 0.0;
      });
      
      parallel_for("wkset soln ip HGRAD",
                   MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{csol.extent(0),csol.extent(1),csol.extent(2)}),
                   KOKKOS_LAMBDA (const size_type elem , const size_type pt, const size_type dim) {
        csol_grad(elem,pt,dim) = 0.0;
      });
      
      parallel_for("wkset soln ip HGRAD",
                   MDRangePolicy<AssemblyExec,Rank<3>>({0,0,0},{cbasis.extent(0),cbasis.extent(1),cbasis.extent(2)}),
                   KOKKOS_LAMBDA (const size_type elem , const size_type dof, const size_type pt) {
        AD uval = cuvals(elem,dof);
        Kokkos::atomic_add(&(csol(elem,pt)), uval*cbasis(elem,dof,pt,0));
        for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
          Kokkos::atomic_add(&(csol_grad(elem,pt,s)), uval*cbasis_grad(elem,dof,pt,s));
        }
      });
       */

       
      
      parallel_for("wkset soln ip HGRAD",
                   MDRangePolicy<AssemblyExec,Rank<2>>({0,0},{cbasis.extent(0),cbasis.extent(2)}),
                   KOKKOS_LAMBDA (const size_type elem , const size_type pt) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          AD uval = cuvals(elem,dof);
          if (dof == 0) {
            csol(elem,pt) = uval*cbasis(elem,dof,pt,0);
            for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
              csol_grad(elem,pt,s) = uval*cbasis_grad(elem,dof,pt,s);
            }
          }
          else {
            csol(elem,pt) += uval*cbasis(elem,dof,pt,0);
            for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
              csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
            }
          }
        }
      });
        
      if (isTransient && type == 1) { // transient terms only needed at volumetric ip (for now)
        auto csol_dot = Kokkos::subview(local_soln_dot, Kokkos::ALL(),var,Kokkos::ALL(),0);
        auto cu_dotvals = Kokkos::subview(u_dotvals,Kokkos::ALL(),var,Kokkos::ALL());
        parallel_for("wkset soln ip HGRAD transient",MDRangePolicy<AssemblyExec,Rank<2>>({0,0},{cbasis.extent(0),cbasis.extent(2)}), KOKKOS_LAMBDA (const size_type elem , const size_type pt) {
          csol_dot(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csol_dot(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,0);
          }
        });
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HVOL
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HVOL.size(); i++) {
      int var = vars_HVOL[i];
      Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> csol;
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis;
      if (type == 1) {
        csol = Kokkos::subview(local_soln,Kokkos::ALL(),var,Kokkos::ALL(),0);
        cbasis = basis[usebasis[var]];
      }
      else if (type == 2) {
        csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
        cbasis = basis_side[usebasis[var]];
      }
      else if (type == 3) {
        csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),0);
        cbasis = basis_face[usebasis[var]];
      }
      auto cuvals = Kokkos::subview(uvals,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset soln ip HVOL",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) = cuvals(elem,dof)*cbasis(elem,dof,pt,0);
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
            }
          }
        }
      });
      
      if (isTransient && type == 1) {
        auto csol_dot = Kokkos::subview(local_soln_dot, Kokkos::ALL(),var,Kokkos::ALL(),0);
        auto cu_dotvals = Kokkos::subview(u_dotvals,Kokkos::ALL(),var,Kokkos::ALL());
        parallel_for("wkset soln ip HVOL transient",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            if ( dof == 0) {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                csol_dot(elem,pt) = cu_dotvals(elem,dof)*cbasis(elem,dof,pt,0);
              }
            }
            else {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                csol_dot(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,0);
              }
            }
          }
        });
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HDIV
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HDIV.size(); i++) {
      int var = vars_HDIV[i];
      Kokkos::View<AD***,Kokkos::LayoutStride,AssemblyDevice> csol;
      Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> csol_div;
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis;
      Kokkos::View<ScalarT***,AssemblyDevice> cbasis_div;
      if (type == 1) {
        csol = Kokkos::subview(local_soln,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        csol_div = Kokkos::subview(local_soln_div,Kokkos::ALL(),var,Kokkos::ALL());
        cbasis = basis[usebasis[var]];
        cbasis_div = basis_div[usebasis[var]];
      }
      else if (type == 2) {
        csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis_side[usebasis[var]];
      }
      else if (type == 3) {
        csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis_face[usebasis[var]];
      }
      auto cuvals = Kokkos::subview(uvals,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset soln ip HDIV",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) = cbasis(elem,dof,pt,s)*cuvals(elem,dof);
              }
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += cbasis(elem,dof,pt,s)*cuvals(elem,dof);
              }
            }
          }
        }
      });
      
      if (type == 1) {
        parallel_for("wkset soln ip HDIV",RangePolicy<AssemblyExec>(0,cbasis_div.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<cbasis_div.extent(1); dof++ ) {
            if ( dof == 0) {
              for (size_type pt=0; pt<cbasis_div.extent(2); pt++ ) {
                csol_div(elem,pt) = cbasis_div(elem,dof,pt)*cuvals(elem,dof);
              }
            }
            else {
              for (size_t pt=0; pt<cbasis_div.extent(2); pt++ ) {
                csol_div(elem,pt) += cbasis_div(elem,dof,pt)*cuvals(elem,dof);
              }
            }
          }
        });
      }
      
      if (isTransient && type == 1) {
        auto csol_dot = Kokkos::subview(local_soln_dot, Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        auto cu_dotvals = Kokkos::subview(u_dotvals,Kokkos::ALL(),var,Kokkos::ALL());
        parallel_for("wkset soln ip HDIV transient",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            if ( dof == 0) {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++ ) {
                  csol_dot(elem,pt,s) = cbasis(elem,dof,pt,s)*cu_dotvals(elem,dof);
                }
              }
            }
            else {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++ ) {
                  csol_dot(elem,pt,s) += cbasis(elem,dof,pt,s)*cu_dotvals(elem,dof);
                }
              }
            }
          }
        });
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HCURL
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HCURL.size(); i++) {
      int var = vars_HCURL[i];
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis, cbasis_curl;
      Kokkos::View<AD***,Kokkos::LayoutStride,AssemblyDevice> csol, csol_curl;
      if (type == 1) {
        csol = Kokkos::subview(local_soln,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        csol_curl = Kokkos::subview(local_soln_curl,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis[usebasis[var]];
        cbasis_curl = basis_curl[usebasis[var]];
      }
      else if (type == 2) {
        csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis_side[usebasis[var]];
      }
      else if (type == 3) {
        csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis_face[usebasis[var]];
      }
      auto cuvals = Kokkos::subview(uvals,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset soln ip HCURL",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) = cbasis(elem,dof,pt,s)*cuvals(elem,dof);
              }
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += cbasis(elem,dof,pt,s)*cuvals(elem,dof);
              }
            }
          }
        }
      });
      
      if (type == 1) {
        parallel_for("wkset soln ip HCURL",RangePolicy<AssemblyExec>(0,cbasis_curl.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            if ( dof == 0) {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++ ) {
                  csol_curl(elem,pt,s) = cbasis_curl(elem,dof,pt,s)*cuvals(elem,dof);
                }
              }
            }
            else {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++ ) {
                  csol_curl(elem,pt,s) += cbasis_curl(elem,dof,pt,s)*cuvals(elem,dof);
                }
              }
            }
          }
        });
      }
      if (isTransient && type == 1) {
        auto csol_dot = Kokkos::subview(local_soln_dot, Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        auto cu_dotvals = Kokkos::subview(u_dotvals,Kokkos::ALL(),var,Kokkos::ALL());
        parallel_for("wkset soln ip HCURL transient",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            if ( dof == 0) {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                for(size_type s=0; s<cbasis.extent(3); s++ ) {
                  csol_dot(elem,pt,s) = cbasis(elem,dof,pt,s)*cu_dotvals(elem,dof);
                }
              }
            }
            else {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++ ) {
                  csol_dot(elem,pt,s) += cbasis(elem,dof,pt,s)*cu_dotvals(elem,dof);
                }
              }
            }
          }
        });
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HFACE
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HFACE.size(); i++) {
      int var = vars_HFACE[i];
      Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> csol;
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis;
      if (type == 1) {
        // not defined
      }
      else if (type == 2) {
        csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
        cbasis = basis_side[usebasis[var]];
      }
      else if (type == 3) {
        csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),0);
        cbasis = basis_face[usebasis[var]];
      }
      auto cuvals = Kokkos::subview(uvals,Kokkos::ALL(),var,Kokkos::ALL());
      
      if (type == 2 || type == 3) {
        parallel_for("wkset soln ip HFACE",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            if ( dof == 0) {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) = cuvals(elem,dof)*cbasis(elem,dof,pt,0);
              }
            }
            else {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
              }
            }
          }
        });
      }
    }
    
  }  
}
  
////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions at specified ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParam(const int & type) {
  
  // type specifications: 1 - volume; 2 - boundary; 3 - face
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnVolTimer);
    
    /////////////////////////////////////////////////////////////////////
    // HGRAD
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<paramvars_HGRAD.size(); i++) {
      int var = paramvars_HGRAD[i];
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis, cbasis_grad;
      Kokkos::View<AD**, Kokkos::LayoutStride, AssemblyDevice> csol;
      Kokkos::View<AD***, Kokkos::LayoutStride, AssemblyDevice> csol_grad;
      auto cuvals = Kokkos::subview(pvals,Kokkos::ALL(),var,Kokkos::ALL());
      
      if (type == 1) { // volumetric ip
        csol = Kokkos::subview(local_param,Kokkos::ALL(),var,Kokkos::ALL(),0);
        csol_grad = Kokkos::subview(local_param_grad,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis[paramusebasis[var]];
        cbasis_grad = basis_grad[paramusebasis[var]];
      }
      else if (type == 2) { // boundary ip
        csol = Kokkos::subview(local_param_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
        csol_grad = Kokkos::subview(local_param_grad_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis_side[paramusebasis[var]];
        cbasis_grad = basis_grad_side[paramusebasis[var]];
      }
      
      parallel_for("wkset soln ip HGRAD",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          AD uval = cuvals(elem,dof);
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) = uval*cbasis(elem,dof,pt,0);
              for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) = uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt,0);
              for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        }
      });
      
    }
    
    /////////////////////////////////////////////////////////////////////
    // HVOL
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<paramvars_HVOL.size(); i++) {
      int var = paramvars_HVOL[i];
      Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> csol;
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis;
      if (type == 1) {
        csol = Kokkos::subview(local_param,Kokkos::ALL(),var,Kokkos::ALL(),0);
        cbasis = basis[paramusebasis[var]];
      }
      else if (type == 2) {
        csol = Kokkos::subview(local_param_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
        cbasis = basis_side[paramusebasis[var]];
      }
      auto cuvals = Kokkos::subview(pvals,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset soln ip HVOL",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) = cuvals(elem,dof)*cbasis(elem,dof,pt,0);
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
            }
          }
        }
      });
    }
    
    /////////////////////////////////////////////////////////////////////
    // HDIV
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<paramvars_HDIV.size(); i++) {
      int var = paramvars_HDIV[i];
      Kokkos::View<AD***,Kokkos::LayoutStride,AssemblyDevice> csol;
      Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> csol_div;
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis;
      Kokkos::View<ScalarT***,AssemblyDevice> cbasis_div;
      if (type == 1) {
        csol = Kokkos::subview(local_param,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        csol_div = Kokkos::subview(local_param_div,Kokkos::ALL(),var,Kokkos::ALL());
        cbasis = basis[paramusebasis[var]];
        cbasis_div = basis_div[paramusebasis[var]];
      }
      else if (type == 2) {
        csol = Kokkos::subview(local_param_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis_side[paramusebasis[var]];
      }
      auto cuvals = Kokkos::subview(pvals,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset soln ip HDIV",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) = cbasis(elem,dof,pt,s)*cuvals(elem,dof);
              }
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += cbasis(elem,dof,pt,s)*cuvals(elem,dof);
              }
            }
          }
        }
      });
      
      if (type == 1) {
        parallel_for("wkset soln ip HDIV",RangePolicy<AssemblyExec>(0,cbasis_div.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<cbasis_div.extent(1); dof++ ) {
            if ( dof == 0) {
              for (size_type pt=0; pt<cbasis_div.extent(2); pt++ ) {
                csol_div(elem,pt) = cbasis_div(elem,dof,pt)*cuvals(elem,dof);
              }
            }
            else {
              for (size_type pt=0; pt<cbasis_div.extent(2); pt++ ) {
                csol_div(elem,pt) += cbasis_div(elem,dof,pt)*cuvals(elem,dof);
              }
            }
          }
        });
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HCURL
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<paramvars_HCURL.size(); i++) {
      int var = paramvars_HCURL[i];
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis, cbasis_curl;
      Kokkos::View<AD***,Kokkos::LayoutStride,AssemblyDevice> csol, csol_curl;
      if (type == 1) {
        csol = Kokkos::subview(local_param,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        csol_curl = Kokkos::subview(local_param_curl,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis[paramusebasis[var]];
        cbasis_curl = basis_curl[paramusebasis[var]];
      }
      else if (type == 2) {
        csol = Kokkos::subview(local_param_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        cbasis = basis_side[paramusebasis[var]];
      }
      auto cuvals = Kokkos::subview(pvals,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset soln ip HCURL",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) = cbasis(elem,dof,pt,s)*cuvals(elem,dof);
              }
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += cbasis(elem,dof,pt,s)*cuvals(elem,dof);
              }
            }
          }
        }
      });
      
      if (type == 1) {
        parallel_for("wkset soln ip HCURL",RangePolicy<AssemblyExec>(0,cbasis_curl.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            if ( dof == 0) {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++ ) {
                  csol_curl(elem,pt,s) = cbasis_curl(elem,dof,pt,s)*cuvals(elem,dof);
                }
              }
            }
            else {
              for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
                for (size_type s=0; s<cbasis.extent(3); s++ ) {
                  csol_curl(elem,pt,s) += cbasis_curl(elem,dof,pt,s)*cuvals(elem,dof);
                }
              }
            }
          }
        });
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HFACE
    /////////////////////////////////////////////////////////////////////
    // Not currently used
  }
}


////////////////////////////////////////////////////////////////////////////////////
// Compute the discretized parameters at the volumetric ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamVolIP(Kokkos::View<ScalarT***,AssemblyDevice> param,
                                const int & seedwhat) {
  
  if (numParams > 0) {
    
    this->computeParamSteadySeeded(param,seedwhat);
    this->computeParam(1);
    
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnFaceIP() {
  this->computeSoln(3);
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSideIP() {
  this->computeSoln(2);
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the discretized parameters at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamSideIP(const int & side, Kokkos::View<ScalarT***,AssemblyDevice> param,
                                 const int & seedwhat) {
  
  if (numParams>0) {
    this->computeParamSteadySeeded(param,seedwhat);
    this->computeParam(2);
  }
  AssemblyExec::execution_space().fence();
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

// TMW: this function should be deprecated
// Gets used only in the boundaryCell flux calculation
// Will not work properly for multi-stage or multi-step
void workset::computeSolnSideIP(const int & side) { //, Kokkos::View<AD***,AssemblyDevice> u_AD_old,
                                //Kokkos::View<AD***,AssemblyDevice> param_AD) {
  {
    //Teuchos::TimeMonitor resettimer(*worksetResetTimer);
    //Kokkos::deep_copy(local_soln_side, 0.0);
    //Kokkos::deep_copy(local_soln_grad_side, 0.0);
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    
    auto u_AD = uvals;

    /////////////////////////////////////////////////////////////////////
    // HGRAD
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HGRAD.size(); i++) {
      int var = vars_HGRAD[i];
      
      auto csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
      auto csol_grad = Kokkos::subview(local_soln_grad_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cbasis = basis_side[usebasis[var]];
      auto cbasis_grad = basis_grad_side[usebasis[var]];
      auto cuvals = Kokkos::subview(u_AD,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset flux soln ip HGRAD",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          AD uval = cuvals(elem,dof);
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) = uval*cbasis(elem,dof,pt,0);
              for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) = uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt,0);
              for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        }
      });
    }
      
    /////////////////////////////////////////////////////////////////////
    // HVOL
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HVOL.size(); i++) {
      int var = vars_HVOL[i];
      
      auto csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
      auto cbasis = basis_side[usebasis[var]];
      auto cuvals = Kokkos::subview(u_AD,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset flux soln ip HVOL",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          AD uval = cuvals(elem,dof);
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) = uval*cbasis(elem,dof,pt,0);
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt,0);
            }
          }
        }
      });
    }
    
    /////////////////////////////////////////////////////////////////////
    // HDIV
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HDIV.size(); i++) {
      int var = vars_HDIV[i];
      
      auto csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cbasis = basis_side[usebasis[var]];
      auto cuvals = Kokkos::subview(u_AD,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset flux soln ip HDIV",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          AD uval = cuvals(elem,dof);
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) = uval*cbasis(elem,dof,pt,s);
              }
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        }
      });
    }
    
    /////////////////////////////////////////////////////////////////////
    // HCURL
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HCURL.size(); i++) {
      int var = vars_HCURL[i];
      
      auto csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cbasis = basis_side[usebasis[var]];
      auto cuvals = Kokkos::subview(u_AD,Kokkos::ALL(),var,Kokkos::ALL());
      
      parallel_for("wkset flux soln ip HCURL",RangePolicy<AssemblyExec>(0,cbasis.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          AD uval = cuvals(elem,dof);
          if ( dof == 0) {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) = uval*cbasis(elem,dof,pt,s);
              }
            }
          }
          else {
            for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
              for (size_type s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        }
      });
    }
  }
}

//////////////////////////////////////////////////////////////
// Add Aux
//////////////////////////////////////////////////////////////

void workset::addAux(const size_t & naux) {
  numAux = naux;
  local_aux = Kokkos::View<AD***, AssemblyDevice>("local_aux",numElem, numAux, numip, maxDerivs);
  //local_aux_grad = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad",numElem, numAux, numip, dimension);
  local_aux_side = Kokkos::View<AD***, AssemblyDevice>("local_aux_side",numElem, numAux, numsideip, maxDerivs);
  flux = Kokkos::View<AD***,AssemblyDevice>("flux",numElem,numAux,numsideip, maxDerivs);
  
  //local_aux_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad_side",numElem, numAux, numsideip, dimension);
}

//////////////////////////////////////////////////////////////
// Get a pointer to vector of parameters
//////////////////////////////////////////////////////////////

vector<AD> workset::getParam(const string & name, bool & found) {
  found = false;
  size_t iter=0;
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


//////////////////////////////////////////////////////////////
// Set the time
//////////////////////////////////////////////////////////////

void workset::setTime(const ScalarT & newtime) {
  time = newtime;
  Kokkos::deep_copy(time_KV,time);
}

//////////////////////////////////////////////////////////////
// Set deltat
//////////////////////////////////////////////////////////////

void workset::setDeltat(const ScalarT & newdt) {
  deltat = newdt;
  Kokkos::deep_copy(deltat_KV,deltat);
}

//////////////////////////////////////////////////////////////
// Set the stage index
//////////////////////////////////////////////////////////////

void workset::setStage(const int & newstage) {
  current_stage = newstage;
  Kokkos::deep_copy(current_stage_KV, newstage);
}
