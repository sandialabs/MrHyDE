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

workset::workset(const vector<int> & cellinfo, const bool & isTransient_,
                 const DRV & ref_ip_, const DRV & ref_wts_, const DRV & ref_side_ip_,
                 const DRV & ref_side_wts_,
                 const vector<string> & basis_types_,
                 const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
                 const topo_RCP & topo, Kokkos::View<int**,UnifiedDevice> & var_bcs_) : //, const Teuchos::RCP<TimeIntegrator> & timeInt_) :
//ref_ip(ref_ip_), ref_wts(ref_wts_), ref_side_ip(ref_side_ip_), ref_side_wts(ref_side_wts_),
basis_types(basis_types_), basis_pointers(basis_pointers_), 
celltopo(topo), var_bcs(var_bcs_), isTransient(isTransient_)  { //, timeInt(timeInt_) {
  
  // Settings that should not change
  dimension = cellinfo[0];
  numVars = cellinfo[1];
  numParams = cellinfo[2];
  numAux = cellinfo[3];
  numDOF = cellinfo[4];
  numElem = cellinfo[5];
  usebcs = true;
  
  deltat = 1.0;
  deltat_KV = Kokkos::View<ScalarT*,AssemblyDevice>("deltat",1);
  deltat_KV(0) = deltat;
  
  current_stage = 0;
  current_stage_KV = Kokkos::View<int*,UnifiedDevice>("stage number on device",1);
  current_stage_KV(0) = 0;
  // Integration information
  numip = ref_ip_.extent(0);
  numsideip = ref_side_ip_.extent(0);
  if (dimension == 2) {
    numsides = celltopo->getSideCount();
  }
  else if (dimension == 3) {
    numsides = celltopo->getFaceCount();
  }
  time_KV = Kokkos::View<ScalarT*,AssemblyDevice>("time",1); // defaults to 0.0
  //sidetype = Kokkos::View<int*,AssemblyDevice>("side types",numElem);
  
  // these can point to different arrays ... data must be deep copied into them
  ip = DRV("ip", numElem,numip, dimension);
  wts = DRV("wts", numElem, numip);
  ip_side = DRV("ip_side", numElem,numsideip,dimension);
  wts_side = DRV("wts_side", numElem,numsideip);
  normals = DRV("normals", numElem,numsideip,dimension);
  
  // these cannot point to different arrays ... data must be deep copied into them
  ip_KV = Kokkos::View<ScalarT***,AssemblyDevice>("ip stored in KV",numElem,numip,dimension);
  ip_side_KV = Kokkos::View<ScalarT***,AssemblyDevice>("side ip stored in KV",numElem,numsideip,dimension);
  normals_KV = Kokkos::View<ScalarT***,AssemblyDevice>("side normals stored in normals KV",numElem,numsideip,dimension);
  point_KV = Kokkos::View<ScalarT***,AssemblyDevice>("ip stored in point KV",1,1,dimension);
  
  
  //h = Kokkos::View<ScalarT*,AssemblyDevice>("h",numElem);
  res = Kokkos::View<AD**,AssemblyDevice>("residual",numElem,numDOF);
  adjrhs = Kokkos::View<AD**,AssemblyDevice>("adjoint RHS",numElem,numDOF);
  
  have_rotation = false;
  have_rotation_phi = false;
  rotation = Kokkos::View<ScalarT***,AssemblyDevice>("rotation matrix",numElem,3,3);
  
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
  local_soln = Kokkos::View<AD****, AssemblyDevice>("local_soln",numElem, numVars, numip, dimension);
  local_soln_side = Kokkos::View<AD****, AssemblyDevice>("local_soln_side",numElem, numVars, numsideip, dimension);
  local_soln_face = Kokkos::View<AD****, AssemblyDevice>("local_soln_face",numElem, numVars, numsideip, dimension);
  local_param_side = Kokkos::View<AD***, AssemblyDevice>("local_param_side",numElem, numParams, numsideip);
  local_soln_point = Kokkos::View<AD****, AssemblyDevice>("local_soln point",1, numVars, 1, dimension);
  local_param_point = Kokkos::View<AD***, AssemblyDevice>("local_soln point",1, numParams, 1);
  local_soln_dot = Kokkos::View<AD****, AssemblyDevice>("local_soln_dot",numElem, numVars, numip, dimension);
  
  //if (vars_HGRAD.size() > 0) {
    local_soln_grad = Kokkos::View<AD****, AssemblyDevice>("local_soln_grad",numElem, numVars, numip, dimension);
    local_soln_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_soln_grad_side",numElem, numVars, numsideip, dimension);
    local_soln_grad_face = Kokkos::View<AD****, AssemblyDevice>("local_soln_grad_face",numElem, numVars, numsideip, dimension);
    local_soln_grad_point = Kokkos::View<AD****, AssemblyDevice>("local_soln point",1, numVars, 1, dimension);
  //}
  //if (vars_HDIV.size() > 0) {
    local_soln_div = Kokkos::View<AD***, AssemblyDevice>("local_soln_div",numElem, numVars, numip);
  //}
  //if (vars_HCURL.size() > 0) {
    local_soln_curl = Kokkos::View<AD****, AssemblyDevice>("local_soln_curl",numElem, numVars, numip, dimension);
  //}
  
  //if (numParams>0) {
    local_param = Kokkos::View<AD***, AssemblyDevice>("local_param",numElem, numParams, numip);
  //}
  
  //if (numAux>0) {
    local_aux = Kokkos::View<AD***, AssemblyDevice>("local_aux",numElem, numAux, numip);
    local_aux_side = Kokkos::View<AD***, AssemblyDevice>("local_aux_side",numElem, numAux, numsideip);
  //}
  
  // Arrays that are not currently used for anything
  local_aux_grad = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad",numElem, numAux, numip, dimension);
  local_param_grad = Kokkos::View<AD****, AssemblyDevice>("local_param_grad",numElem, numParams, numip, dimension);
  local_soln_dot_grad = Kokkos::View<AD****, AssemblyDevice>("local_soln_dot_grad",numElem, numVars, numip, dimension);
  local_soln_dot_side = Kokkos::View<AD****, AssemblyDevice>("local_soln_dot_side",numElem, numVars, numsideip, dimension);
  local_param_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_param_grad_side",numElem, numParams, numsideip, dimension);
  local_aux_grad_side = Kokkos::View<AD****, AssemblyDevice>("local_aux_grad_side",numElem, numAux, numsideip, dimension);
  local_param_grad_point = Kokkos::View<AD****, AssemblyDevice>("local_soln point",1, numParams, 1, dimension);
  
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetResidual() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  AD val = 0.0;
  //res.assign_data(&val);
  parallel_for(RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    for (int n=0; n<res.extent(1); n++) {
      res(e,n) = val;
    }
  });
  AssemblyExec::execution_space().fence();
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
  AssemblyExec::execution_space().fence();
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
// TMW: I believe this can be deprecated
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
  AssemblyExec::execution_space().fence();
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
  AssemblyExec::execution_space().fence();
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
  AssemblyExec::execution_space().fence();
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
  AssemblyExec::execution_space().fence();
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the volumetric ip
////////////////////////////////////////////////////////////////////////////////////

// TMW: This funciton should be deprecated
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
    AssemblyExec::execution_space().fence();
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnVolTimer);
    AD uval;
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      int knbasis = numbasis[kubasis];
      string kutype = basis_types[kubasis];
      
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis[kubasis];
        DRV kbasis_grad_uw = basis_grad[kubasis];
        
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
        DRV kbasis_uw = basis[kubasis];
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
        DRV kbasis_uw = basis[kubasis];
        DRV kbasis_div_uw = basis_div[kubasis];
        
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
        DRV kbasis_uw = basis[kubasis];
        DRV kbasis_curl_uw = basis_curl[kubasis];
        
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
                               Kokkos::View<ScalarT****,AssemblyDevice> u_stage,
                               const int & seedwhat) {
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnVolTimer);
    
    // Reset/fill operations specialized for each basis type
    
    /////////////////////////////////////////////////////////////////////
    // HGRAD
    /////////////////////////////////////////////////////////////////////
    
    for (int i=0; i<vars_HGRAD.size(); i++) {
      int var = vars_HGRAD[i];
      auto csol = Kokkos::subview(local_soln,Kokkos::ALL(),var,Kokkos::ALL(),0);
      auto csol_grad = Kokkos::subview(local_soln_grad,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
        AD value = 0.0;
        for (int k=0; k<csol.extent(1); k++) {
          csol(e,k) = value;
        }
        for (int k=0; k<csol_grad.extent(1); k++) {
          for (int j=0; j<csol_grad.extent(2); j++) {
            csol_grad(e,k,j) = value;
          }
        }
      });
      if (isTransient) {
        auto csol_dot = Kokkos::subview(local_soln_dot,Kokkos::ALL(),var,Kokkos::ALL(),0);
        parallel_for(RangePolicy<AssemblyExec>(0,csol_dot.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD value = 0.0;
          for (int k=0; k<csol_dot.extent(1); k++) {
            csol_dot(e,k) = value;
          }
        });
      }
      
      DRV cbasis = basis[usebasis[var]];
      DRV cbasis_grad = basis_grad[usebasis[var]];
      auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
      auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
      
      if (isTransient) { // transient problem
        auto csol_dot = Kokkos::subview(local_soln_dot, Kokkos::ALL(),var,Kokkos::ALL(),0);
        auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        if (seedwhat == 1) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
                csol_dot(elem,pt) += u_dotval*cbasis(elem,dof,pt);
                for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                  csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
                }
              }
            }
          });
          
        }
        else if (seedwhat == 2) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = cu(elem,dof);
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              //cout << "uval2 = " << uval << endl;
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              ScalarT val = u_dotval.val();
              u_dotval = AD(maxDerivs,coff(dof),val);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
                csol_dot(elem,pt) += u_dotval*cbasis(elem,dof,pt);
                for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                  csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
                }
              }
            }
          });
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = cu(elem,dof);
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
                csol_dot(elem,pt) += u_dotval*cbasis(elem,dof,pt);
                for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                  csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
                }
              }
            }
          });
        }
      }
      else { // steady-state
        if (seedwhat == 1) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              AD uval = AD(maxDerivs,coff(dof),cu(elem,dof));
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
                for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                  csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
                }
              }
            }
          });
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval;
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              uval = cu(elem,dof);
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
                for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                  csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
                }
              }
            }
          });
        }
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HVOL
    /////////////////////////////////////////////////////////////////////
    
    for (int i=0; i<vars_HVOL.size(); i++) {
      int var = vars_HVOL[i];
      auto csol = Kokkos::subview(local_soln,Kokkos::ALL(),var,Kokkos::ALL(),0);
      parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
        AD value = 0.0;
        for (int k=0; k<csol.extent(1); k++) {
          csol(e,k) = value;
        }
      });
      if (isTransient) {
        auto csol_dot = Kokkos::subview(local_soln_dot,Kokkos::ALL(),var,Kokkos::ALL(),0);
        parallel_for(RangePolicy<AssemblyExec>(0,csol_dot.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD value = 0.0;
          for (int k=0; k<csol_dot.extent(1); k++) {
            csol_dot(e,k) = value;
          }
        });
      }
      
      DRV cbasis = basis[usebasis[var]];
      auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
      auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
      
      if (isTransient) { // transient problem
        auto csol_dot = Kokkos::subview(local_soln_dot, Kokkos::ALL(),var,Kokkos::ALL(),0);
        auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        if (seedwhat == 1) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
                csol_dot(elem,pt) += u_dotval*cbasis(elem,dof,pt);
              }
            }
          });
          
        }
        else if (seedwhat == 2) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = cu(elem,dof);
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              //cout << "uval2 = " << uval << endl;
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              ScalarT val = u_dotval.val();
              u_dotval = AD(maxDerivs,coff(dof),val);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
                csol_dot(elem,pt) += u_dotval*cbasis(elem,dof,pt);
              }
            }
          });
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = cu(elem,dof);
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
                csol_dot(elem,pt) += u_dotval*cbasis(elem,dof,pt);
              }
            }
          });
        }
      }
      else { // steady-state
        if (seedwhat == 1) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval;
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              uval = AD(maxDerivs,coff(dof),cu(elem,dof));
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
              }
            }
          });
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval;
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              uval = cu(elem,dof);
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                csol(elem,pt) += uval*cbasis(elem,dof,pt);
              }
            }
          });
        }
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HDIV
    /////////////////////////////////////////////////////////////////////
    
    for (int i=0; i<vars_HDIV.size(); i++) {
      int var = vars_HDIV[i];
      auto csol = Kokkos::subview(local_soln,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto csol_div = Kokkos::subview(local_soln_div,Kokkos::ALL(),var,Kokkos::ALL());
      parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
        AD value = 0.0;
        for (int k=0; k<csol.extent(1); k++) {
          csol_div(e,k) = value;
          for (int j=0; j<csol.extent(2); j++) {
            csol(e,k,j) = value;
          }
        }
      });
      if (isTransient) {
        auto csol_dot = Kokkos::subview(local_soln_dot,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        parallel_for(RangePolicy<AssemblyExec>(0,csol_dot.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD value = 0.0;
          for (int k=0; k<csol_dot.extent(1); k++) {
            for (int j=0; j<csol_dot.extent(2); j++) {
              csol_dot(e,k,j) = value;
            }
          }
        });
      }
      
      DRV cbasis = basis[usebasis[var]];
      DRV cbasis_div = basis_div[usebasis[var]];
      auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
      auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
      
      if (isTransient) { // transient problem
        auto csol_dot = Kokkos::subview(local_soln_dot, Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        if (seedwhat == 1) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                  csol_dot(elem,pt,s) += u_dotval*cbasis(elem,dof,pt,s);
                }
                csol_div(elem,pt) += uval*cbasis_div(elem,dof,pt);
              }
            }
          });
          
        }
        else if (seedwhat == 2) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = cu(elem,dof);
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              //cout << "uval2 = " << uval << endl;
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              ScalarT val = u_dotval.val();
              u_dotval = AD(maxDerivs,coff(dof),val);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                  csol_dot(elem,pt,s) += u_dotval*cbasis(elem,dof,pt,s);
                }
                csol_div(elem,pt) += uval*cbasis_div(elem,dof,pt);
              }
            }
          });
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = cu(elem,dof);
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                  csol_dot(elem,pt,s) += u_dotval*cbasis(elem,dof,pt,s);
                }
                csol_div(elem,pt) += uval*cbasis_div(elem,dof,pt);
              }
            }
          });
        }
      }
      else { // steady-state
        if (seedwhat == 1) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              AD uval = AD(maxDerivs,coff(dof),cu(elem,dof));
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                }
                csol_div(elem,pt) += uval*cbasis_div(elem,dof,pt);
              }
            }
          });
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              AD uval = cu(elem,dof);
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                }
                csol_div(elem,pt) += uval*cbasis_div(elem,dof,pt);
              }
            }
          });
        }
      }
      
    }
    
    /////////////////////////////////////////////////////////////////////
    // HCURL
    /////////////////////////////////////////////////////////////////////
    
    for (int i=0; i<vars_HCURL.size(); i++) {
      int var = vars_HCURL[i];
      auto csol = Kokkos::subview(local_soln,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto csol_curl = Kokkos::subview(local_soln_grad,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
        AD value = 0.0;
        for (int k=0; k<csol.extent(1); k++) {
          for (int j=0; j<csol.extent(2); j++) {
            csol(e,k,j) = value;
            csol_curl(e,k,j) = value;
          }
        }
      });
      if (isTransient) {
        auto csol_dot = Kokkos::subview(local_soln_dot,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        parallel_for(RangePolicy<AssemblyExec>(0,csol_dot.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD value = 0.0;
          for (int k=0; k<csol_dot.extent(1); k++) {
            for (int j=0; j<csol_dot.extent(2); j++) {
              csol_dot(e,k,j) = value;
            }
          }
        });
      }
      
      DRV cbasis = basis[usebasis[var]];
      DRV cbasis_curl = basis_curl[usebasis[var]];
      auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
      auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
      
      if (isTransient) { // transient problem
        auto csol_dot = Kokkos::subview(local_soln_dot, Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
        if (seedwhat == 1) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                  csol_dot(elem,pt,s) += u_dotval*cbasis(elem,dof,pt,s);
                  csol_curl(elem,pt,s) += uval*cbasis_curl(elem,dof,pt,s);
                }
              }
            }
          });
          
        }
        else if (seedwhat == 2) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = cu(elem,dof);
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              //cout << "uval2 = " << uval << endl;
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              ScalarT val = u_dotval.val();
              u_dotval = AD(maxDerivs,coff(dof),val);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                  csol_dot(elem,pt,s) += u_dotval*cbasis(elem,dof,pt,s);
                  csol_curl(elem,pt,s) += uval*cbasis_curl(elem,dof,pt,s);
                }
              }
            }
          });
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            AD uval, u_dotval, stageval;
            int current_stage = current_stage_KV(0);
            ScalarT deltat = deltat_KV(0);
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              stageval = cu(elem,dof);
              uval = cu_prev(elem,dof,0);
              uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
              for (int s=0; s<current_stage; s++) {
                uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              u_dotval = BDF_wts(0)*stageval;
              for (int s=1; s<BDF_wts.extent(0); s++) {
                u_dotval += BDF_wts(s)*cu_prev(elem,dof,s-1);
              }
              u_dotval *= 1.0/deltat;
              u_dotval *= 1.0/butcher_b(current_stage);
              
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                  csol_dot(elem,pt,s) += u_dotval*cbasis(elem,dof,pt,s);
                  csol_curl(elem,pt,s) += uval*cbasis_curl(elem,dof,pt,s);
                }
              }
            }
          });
        }
      }
      else { // steady-state
        if (seedwhat == 1) {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              AD uval = AD(maxDerivs,coff(dof),cu(elem,dof));
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                  csol_curl(elem,pt,s) += uval*cbasis_curl(elem,dof,pt,s);
                }
              }
            }
          });
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            for (int dof=0; dof<cbasis.extent(1); dof++ ) {
              AD uval = cu(elem,dof);
              for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
                for (int s=0; s<cbasis.extent(3); s++ ) {
                  csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
                  csol_curl(elem,pt,s) += uval*cbasis_curl(elem,dof,pt,s);
                }
              }
            }
          });
        }
      }
    }
    
    // HFACE variables have no volumetric support
    
  }
  
}
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the discretized parameters at the volumetric ip
  ////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamVolIP(Kokkos::View<ScalarT***,AssemblyDevice> param,
                                const int & seedwhat) {
  
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
      AssemblyExec::execution_space().fence();
    }
    
    {
      Teuchos::TimeMonitor basistimer(*worksetComputeParamVolTimer);
      Kokkos::View<int*,UnifiedDevice> bind("basis index",1);
      
      for (int k=0; k<numParams; k++) {
        int kpbasis = paramusebasis[k];
        
        DRV pbasis = basis[kpbasis];
        DRV pbasis_grad = basis_grad[kpbasis];
        bind(0) = k;
        
        parallel_for(RangePolicy<AssemblyExec>(0,pbasis.extent(0)), KOKKOS_LAMBDA (const int e ) {
          AD paramval;
          int kk = bind(0);
          for (int i=0; i<pbasis.extent(1); i++ ) {
            
            if (seedwhat == 3) {
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
        AssemblyExec::execution_space().fence();
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnFaceIP(Kokkos::View<ScalarT***,AssemblyDevice> u,
                                Kokkos::View<ScalarT****,AssemblyDevice> u_prev,
                                Kokkos::View<ScalarT****,AssemblyDevice> u_stage,
                                const int & seedwhat) {
  
  /////////////////////////////////////////////////////////////////////
  // HGRAD
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HGRAD.size(); i++) {
    int var = vars_HGRAD[i];
    auto csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),0);
    auto csol_grad = Kokkos::subview(local_soln_grad_face,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        csol(e,k) = value;
      }
      for (int k=0; k<csol_grad.extent(1); k++) {
        for (int j=0; j<csol_grad.extent(2); j++) {
          csol_grad(e,k,j) = value;
        }
      }
    });
    
    DRV cbasis = basis_face[usebasis[var]];
    DRV cbasis_grad = basis_grad_face[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, u_dotval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
              for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
              for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
              for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
              for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
  }
  
  /////////////////////////////////////////////////////////////////////
  // HVOL
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HVOL.size(); i++) {
    int var = vars_HVOL[i];
    auto csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),0);
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        csol(e,k) = value;
      }
    });
    
    DRV cbasis = basis_face[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
        
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
    }
  }
  
  /////////////////////////////////////////////////////////////////////
  // HDIV
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HDIV.size(); i++) {
    int var = vars_HDIV[i];
    auto csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        for (int j=0; j<csol.extent(2); j++) {
          csol(e,k,j) = value;
        }
      }
    });
    
    DRV cbasis = basis_face[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, u_dotval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
        
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
    
  }
  
  /////////////////////////////////////////////////////////////////////
  // HCURL
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HCURL.size(); i++) {
    int var = vars_HCURL[i];
    auto csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        for (int j=0; j<csol.extent(2); j++) {
          csol(e,k,j) = value;
        }
      }
    });
    
    DRV cbasis = basis_face[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
        
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
  }
  
  /////////////////////////////////////////////////////////////////////
  // HFACE
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HFACE.size(); i++) {
    int var = vars_HFACE[i];
    auto csol = Kokkos::subview(local_soln_face,Kokkos::ALL(),var,Kokkos::ALL(),0);
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        csol(e,k) = value;
      }
    });
    
    DRV cbasis = basis_face[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
        
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
    }
  }
  
  
  /*
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
    AssemblyExec::execution_space().fence();
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    Kokkos::View<int*,UnifiedDevice> bind("basis index",1);
    
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      string kutype = basis_types[kubasis];
      bind(0) = k;
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_face[kubasis];
        DRV kbasis_grad_uw = basis_grad_face[kubasis];
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
        DRV kbasis_uw = basis_face[kubasis];
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
        DRV kbasis_uw = basis_face[kubasis];
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
        
        DRV kbasis_uw = basis_face[kubasis];
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
  AssemblyExec::execution_space().fence();
   */
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSideIP(Kokkos::View<ScalarT***,AssemblyDevice> u,
                                Kokkos::View<ScalarT****,AssemblyDevice> u_prev,
                                Kokkos::View<ScalarT****,AssemblyDevice> u_stage,
                                const int & seedwhat) {
  
  /////////////////////////////////////////////////////////////////////
  // HGRAD
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HGRAD.size(); i++) {
    int var = vars_HGRAD[i];
    auto csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
    auto csol_grad = Kokkos::subview(local_soln_grad_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        csol(e,k) = value;
      }
      for (int k=0; k<csol_grad.extent(1); k++) {
        for (int j=0; j<csol_grad.extent(2); j++) {
          csol_grad(e,k,j) = value;
        }
      }
    });
    
    DRV cbasis = basis_side[usebasis[var]];
    DRV cbasis_grad = basis_grad_side[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, u_dotval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
              for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
              for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
              for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
              for (int s=0; s<cbasis_grad.extent(3); s++ ) {
                csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
  }
  
  /////////////////////////////////////////////////////////////////////
  // HVOL
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HVOL.size(); i++) {
    int var = vars_HVOL[i];
    auto csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        csol(e,k) = value;
      }
    });
    
    DRV cbasis = basis_side[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
        
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
    }
  }
  
  /////////////////////////////////////////////////////////////////////
  // HDIV
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HDIV.size(); i++) {
    int var = vars_HDIV[i];
    auto csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        for (int j=0; j<csol.extent(2); j++) {
          csol(e,k,j) = value;
        }
      }
    });
    
    DRV cbasis = basis_side[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, u_dotval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
        
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
    
  }
  
  /////////////////////////////////////////////////////////////////////
  // HCURL
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HCURL.size(); i++) {
    int var = vars_HCURL[i];
    auto csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        for (int j=0; j<csol.extent(2); j++) {
          csol(e,k,j) = value;
        }
      }
    });
    
    DRV cbasis = basis_side[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
        
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            AD uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              for (int s=0; s<cbasis.extent(3); s++ ) {
                csol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
              }
            }
          }
        });
      }
    }
  }
  
  /////////////////////////////////////////////////////////////////////
  // HFACE
  /////////////////////////////////////////////////////////////////////
  
  for (int i=0; i<vars_HFACE.size(); i++) {
    int var = vars_HFACE[i];
    auto csol = Kokkos::subview(local_soln_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
    parallel_for(RangePolicy<AssemblyExec>(0,csol.extent(0)), KOKKOS_LAMBDA (const int e ) {
      AD value = 0.0;
      for (int k=0; k<csol.extent(1); k++) {
        csol(e,k) = value;
      }
    });
    
    DRV cbasis = basis_side[usebasis[var]];
    auto coff = Kokkos::subview(offsets,var,Kokkos::ALL());
    auto cu = Kokkos::subview(u,Kokkos::ALL(),var,Kokkos::ALL());
    
    if (isTransient) { // transient problem
      auto cu_prev = Kokkos::subview(u_prev,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      auto cu_stage = Kokkos::subview(u_stage,Kokkos::ALL(),var,Kokkos::ALL(),Kokkos::ALL());
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = AD(maxDerivs,coff(dof),cu(elem,dof));
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
        
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval, stageval;
          int current_stage = current_stage_KV(0);
          ScalarT deltat = deltat_KV(0);
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            stageval = cu(elem,dof);
            uval = cu_prev(elem,dof,0);
            uval += butcher_A(current_stage,current_stage)/butcher_b(current_stage)*(stageval - cu_prev(elem,dof,0));
            for (int s=0; s<current_stage; s++) {
              uval += butcher_A(current_stage,s)/butcher_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
    }
    else { // steady-state
      if (seedwhat == 1) {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = AD(maxDerivs,coff(dof),cu(elem,dof));
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyExec>(0,u.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          AD uval;
          for (int dof=0; dof<cbasis.extent(1); dof++ ) {
            uval = cu(elem,dof);
            for (size_t pt=0; pt<cbasis.extent(2); pt++ ) {
              csol(elem,pt) += uval*cbasis(elem,dof,pt);
            }
          }
        });
      }
    }
  }
  
  
  /*
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
    AssemblyExec::execution_space().fence();
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    Kokkos::View<int*,UnifiedDevice> bind("basis index",1);
    
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      string kutype = basis_types[kubasis];
      bind(0) = k;
      
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_side[kubasis];
        DRV kbasis_grad_uw = basis_grad_side[kubasis];
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
        DRV kbasis_uw = basis_side[kubasis];
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
        DRV kbasis_uw = basis_side[kubasis];
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
  AssemblyExec::execution_space().fence();
   */
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the discretized parameters at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamSideIP(const int & side, Kokkos::View<ScalarT***,AssemblyDevice> param,
                                 const int & seedwhat) {
  
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
      AssemblyExec::execution_space().fence();
    }
    
    {
      Teuchos::TimeMonitor basistimer(*worksetComputeParamSideTimer);
      
      Kokkos::View<int*,UnifiedDevice> bind("basis index",1);
      
      for (int k=0; k<numParams; k++) {
        int kpbasis = paramusebasis[k];
        bind(0) = k;
        DRV pbasis = basis_side[kpbasis];
        DRV pbasis_grad = basis_grad_side[kpbasis];
        
        parallel_for(RangePolicy<AssemblyExec>(0,pbasis.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (int i=0; i<pbasis.extent(1); i++ ) {
            AD paramval;
            int kk = bind(0);
            if (seedwhat == 3) {
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
  AssemblyExec::execution_space().fence();
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

// TMW: this function should be deprecated
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
    AssemblyExec::execution_space().fence();
  }
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    AD uval, u_dotval;
    for (int k=0; k<numVars; k++) {
      int kubasis = usebasis[k];
      //int knbasis = numbasis[kubasis];
      string kutype = basis_types[kubasis];
      
      if (kutype == "HGRAD") {
        DRV kbasis_uw = basis_side[kubasis];
        DRV kbasis_grad_uw = basis_grad_side[kubasis];
        
        for (int i=0; i<kbasis_uw.extent(1); i++ ) {
          for (int e=0; e<kbasis_uw.extent(0); e++) {
            uval = u_AD(e,k,i);
            for (size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              local_soln_side(e,k,j,0) += uval*kbasis_uw(e,i,j);
              for (int s=0; s<kbasis_grad_uw.extent(3); s++ ) {
                local_soln_grad_side(e,k,j,s) += uval*kbasis_grad_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HVOL") {
        DRV kbasis_uw = basis_side[kubasis];
        for( int i=0; i<kbasis_uw.extent(1); i++ ) {
          for (int e=0; e<kbasis_uw.extent(0); e++) {
            uval = u_AD(e,k,i);
            for( size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              local_soln_side(e,k,j,0) += uval*kbasis_uw(e,i,j);
            }
          }
        }
      }
      else if (kutype == "HDIV"){
        DRV kbasis_uw = basis_side[kubasis];
        for (int i=0; i<kbasis_uw.extent(1); i++ ) {
          for (int e=0; e<kbasis_uw.extent(0); e++) {
            uval = u_AD(e,k,i);
            for (size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              for (int s=0; s<kbasis_uw.extent(3); s++ ) {
                local_soln_side(e,k,j,s) += uval*kbasis_uw(e,i,j,s);
              }
            }
          }
        }
      }
      else if (kutype == "HCURL"){
        DRV kbasis_uw = basis_side[kubasis];
        
        for (int i=0; i<kbasis_uw.extent(1); i++ ) {
          for (int e=0; e<kbasis_uw.extent(0); e++) {
            uval = u_AD(e,k,i);
            for (size_t j=0; j<kbasis_uw.extent(2); j++ ) {
              for (int s=0; s<kbasis_uw.extent(3); s++ ) {
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
  
  /*
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
  }*/
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
