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
                 const topo_RCP & topo, Kokkos::View<string**,HostDevice> & var_bcs_) :
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
  
  maxElem = numElem;
  deltat = 1.0;
  deltat_KV = Kokkos::View<ScalarT*,AssemblyDevice>("deltat",1);
  Kokkos::deep_copy(deltat_KV,deltat);
  
  current_stage_KV = Kokkos::View<int*,AssemblyDevice>("stage number on device",1);
  Kokkos::deep_copy(current_stage_KV,0);
  time_KV = Kokkos::View<ScalarT*,AssemblyDevice>("time",1); // defaults to 0.0
  
  // Add scalar views to store ips
  this->addDataSc("x",numElem,numip);
  this->addDataSc("y",numElem,numip);
  this->addDataSc("z",numElem,numip);
  
  this->addDataSc("x side",numElem,numsideip);
  this->addDataSc("y side",numElem,numsideip);
  this->addDataSc("z side",numElem,numsideip);
  
  this->addDataSc("nx side",numElem,numsideip);
  this->addDataSc("ny side",numElem,numsideip);
  this->addDataSc("nz side",numElem,numsideip);
  
  this->addDataSc("tx side",numElem,numsideip);
  this->addDataSc("ty side",numElem,numsideip);
  this->addDataSc("tz side",numElem,numsideip);
  
  this->addDataSc("x point",1,1);
  this->addDataSc("y point",1,1);
  this->addDataSc("z point",1,1);
  
  // these can point to different arrays
  wts = View_Sc2("ip wts",numElem,numip);
  wts_side = View_Sc2("ip side wts",numElem,numsideip);
    
  have_rotation = false;
  have_rotation_phi = false;
  rotation = View_Sc3("rotation matrix",numElem,3,3);
  
  int maxb = 0;
  for (size_t i=0; i<basis_pointers.size(); i++) {
    int numb = basis_pointers[i]->getCardinality();
    maxb = std::max(maxb,numb);
  }
  
  // These are stored as vector<View_AD2> instead of View_AD3 to avoid subviews
  for (size_t k=0; k<numVars; ++k) {
    uvals.push_back(View_AD2("seeded uvals",numElem, maxb, maxDerivs));
    if (isTransient) {
      u_dotvals.push_back(View_AD2("seeded uvals",numElem, maxb, maxDerivs));
    }
  }
    
  
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  maxTeamSize = 256 / VectorSize;
#else
  maxTeamSize = 1;
#endif
  
}

////////////////////////////////////////////////////////////////////////////////////
// Public functions
////////////////////////////////////////////////////////////////////////////////////

void workset::createSolns() {

  // Need to first allocate the residual view
  // This is the largest view in the code (due to the AD) so we are careful with the size
  
  
  maxRes = std::max(offsets.extent(0)*offsets.extent(1),numParams);
  if (paramusebasis.size() > 0) {
    maxRes = std::max(maxRes,paramoffsets.extent(0)*paramoffsets.extent(1));
  }
  if (auxusebasis.size() > 0) {
    maxRes = std::max(maxRes,aux_offsets.extent(0)*aux_offsets.extent(1));
  }
  
  res = View_AD2("residual",numElem, maxRes, maxDerivs);

  for (size_t i=0; i<usebasis.size(); i++) {
    int bind = usebasis[i];
    string var = varlist[i];
    int numb = basis_pointers[bind]->getCardinality();
    uvals.push_back(View_AD2("seeded uvals",numElem, numb, maxDerivs));
    if (isTransient) {
      u_dotvals.push_back(View_AD2("seeded uvals",numElem, numb, maxDerivs));
    }
    
    if (basis_types[bind].substr(0,5) == "HGRAD") {
      vars_HGRAD.push_back(i);
      varlist_HGRAD.push_back(var);
      this->addData(var,numElem,numip);
      this->addData("grad("+var+")[x]",numElem,numip);
      this->addData("grad("+var+")[y]",numElem,numip);
      this->addData("grad("+var+")[z]",numElem,numip);
      this->addData(var+"_t",numElem,numip);
      this->addData(var+" side",numElem,numsideip);
      this->addData("grad("+var+")[x] side",numElem,numsideip);
      this->addData("grad("+var+")[y] side",numElem,numsideip);
      this->addData("grad("+var+")[z] side",numElem,numsideip);
      this->addData(var+" point",1,1);
      this->addData("grad("+var+")[x] point",1,1);
      this->addData("grad("+var+")[y] point",1,1);
      this->addData("grad("+var+")[z] point",1,1);
    }
    else if (basis_types[bind].substr(0,4) == "HDIV" ) {
      vars_HDIV.push_back(i);
      varlist_HDIV.push_back(var);
      this->addData(var+"[x]",numElem,numip);
      this->addData(var+"[y]",numElem,numip);
      this->addData(var+"[z]",numElem,numip);
      this->addData("div("+var+")",numElem,numip);
      this->addData(var+"_t[x]",numElem,numip);
      this->addData(var+"_t[y]",numElem,numip);
      this->addData(var+"_t[z]",numElem,numip);
      this->addData(var+"[x] side",numElem,numsideip);
      this->addData(var+"[y] side",numElem,numsideip);
      this->addData(var+"[z] side",numElem,numsideip);
      this->addData(var+"[x] point",1,1);
      this->addData(var+"[y] point",1,1);
      this->addData(var+"[z] point",1,1);
    }
    else if (basis_types[bind].substr(0,4) == "HVOL") {
      vars_HVOL.push_back(i);
      varlist_HVOL.push_back(var);
      this->addData(var,numElem,numip);
      this->addData(var+"_t",numElem,numip);
      this->addData(var+" side",numElem,numsideip);
      this->addData(var+" point",1,1);
    }
    else if (basis_types[bind].substr(0,5) == "HCURL") {
      vars_HCURL.push_back(i);
      varlist_HCURL.push_back(var);
      this->addData(var+"[x]",numElem,numip);
      this->addData(var+"[y]",numElem,numip);
      this->addData(var+"[z]",numElem,numip);
      this->addData("curl("+var+")[x]",numElem,numip);
      this->addData("curl("+var+")[y]",numElem,numip);
      this->addData("curl("+var+")[z]",numElem,numip);
      this->addData(var+"_t[x]",numElem,numip);
      this->addData(var+"_t[y]",numElem,numip);
      this->addData(var+"_t[z]",numElem,numip);
      this->addData(var+"[x] side",numElem,numsideip);
      this->addData(var+"[y] side",numElem,numsideip);
      this->addData(var+"[z] side",numElem,numsideip);
      this->addData(var+"[x] point",1,1);
      this->addData(var+"[y] point",1,1);
      this->addData(var+"[z] point",1,1);
    }
    else if (basis_types[bind].substr(0,5) == "HFACE") {
      vars_HFACE.push_back(i);
      varlist_HFACE.push_back(var);
      this->addData(var+" side",numElem,numsideip);
    }
  }
  
  
  for (size_t i=0; i<paramusebasis.size(); i++) {
    int bind = paramusebasis[i];
    string var = param_varlist[i];
    int numb = basis_pointers[bind]->getCardinality();
    pvals.push_back(View_AD2("seeded uvals",numElem, numb, maxDerivs));
    
    if (basis_types[bind].substr(0,5) == "HGRAD") {
      paramvars_HGRAD.push_back(i);
      paramvarlist_HGRAD.push_back(var);
      this->addData(var,numElem,numip);
      this->addData("grad("+var+")[x]",numElem,numip);
      this->addData("grad("+var+")[y]",numElem,numip);
      this->addData("grad("+var+")[z]",numElem,numip);
      this->addData(var+"_t",numElem,numip);
      this->addData(var+" side",numElem,numsideip);
      this->addData("grad("+var+")[x] side",numElem,numsideip);
      this->addData("grad("+var+")[y] side",numElem,numsideip);
      this->addData("grad("+var+")[z] side",numElem,numsideip);
      this->addData(var+" point",1,1);
      this->addData("grad("+var+")[x] point",1,1);
      this->addData("grad("+var+")[y] point",1,1);
      this->addData("grad("+var+")[z] point",1,1);
    }
    else if (basis_types[bind].substr(0,4) == "HDIV") {
      paramvars_HDIV.push_back(i);
      paramvarlist_HDIV.push_back(var);
      this->addData(var+"[x]",numElem,numip);
      this->addData(var+"[y]",numElem,numip);
      this->addData(var+"[z]",numElem,numip);
      this->addData("div("+var+")",numElem,numip);
      this->addData(var+"_t[x]",numElem,numip);
      this->addData(var+"_t[y]",numElem,numip);
      this->addData(var+"_t[z]",numElem,numip);
      this->addData(var+"[x] side",numElem,numsideip);
      this->addData(var+"[y] side",numElem,numsideip);
      this->addData(var+"[z] side",numElem,numsideip);
      this->addData(var+"[x] point",1,1);
      this->addData(var+"[y] point",1,1);
      this->addData(var+"[z] point",1,1);
    }
    else if (basis_types[bind].substr(0,4) == "HVOL") {
      paramvars_HVOL.push_back(i);
      paramvarlist_HVOL.push_back(var);
      this->addData(var,numElem,numip);
      this->addData(var+"_t",numElem,numip);
      this->addData(var+" side",numElem,numsideip);
      this->addData(var+" point",1,1);
    }
    else if (basis_types[bind].substr(0,5) == "HCURL") {
      paramvars_HCURL.push_back(i);
      paramvarlist_HCURL.push_back(var);
      this->addData(var+"[x]",numElem,numip);
      this->addData(var+"[y]",numElem,numip);
      this->addData(var+"[z]",numElem,numip);
      this->addData("curl("+var+")[x]",numElem,numip);
      this->addData("curl("+var+")[y]",numElem,numip);
      this->addData("curl("+var+")[z]",numElem,numip);
      this->addData(var+"_t[x]",numElem,numip);
      this->addData(var+"_t[y]",numElem,numip);
      this->addData(var+"_t[z]",numElem,numip);
      this->addData(var+"[x] side",numElem,numsideip);
      this->addData(var+"[y] side",numElem,numsideip);
      this->addData(var+"[z] side",numElem,numsideip);
      this->addData(var+"[x] point",1,1);
      this->addData(var+"[y] point",1,1);
      this->addData(var+"[z] point",1,1);
    }
    else if (basis_types[bind].substr(0,5) == "HFACE") {
      paramvars_HFACE.push_back(i);
      paramvarlist_HFACE.push_back(var);
      this->addData(var+" side",numElem,numsideip);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution to zero
////////////////////////////////////////////////////////////////////////////////////

void workset::resetResidual() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  //Kokkos::deep_copy(res,0.0);
  
  size_t maxRes_ = maxRes;
  
  parallel_for("wkset reset res",
               TeamPolicy<AssemblyExec>(res.extent(0), Kokkos::AUTO),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type dof=team.team_rank(); dof<maxRes_; dof+=team.team_size() ) {
      res(elem,dof).val() = 0.0;
      for (size_type d=0; d<maxRes_; ++d) {
        res(elem,dof).fastAccessDx(d) = 0.0;
      }
      //for (size_type var=0; var<off.extent(0); ++var) {
      //  res(elem,off(var,dof)) = 0.0;
      //}
    }
  });
  
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for general transient problems
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnTransientSeeded(View_Sc3 u,
                                         View_Sc4 u_prev,
                                         View_Sc4 u_stage,
                                         const int & seedwhat,
                                         const int & index) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
  
  // These need to be set locally to be available to AssemblyDevice
  auto dt = deltat_KV;
  auto curr_stage = current_stage_KV;
  auto b_A = butcher_A;
  auto b_b = butcher_b;
  auto b_c = butcher_c;
  auto BDF = BDF_wts;

  // Seed the current stage solution
  if (seedwhat == 1) {
    for (size_type var=0; var<u.extent(1); var++ ) {
      auto u_AD = uvals[var];
      auto u_dot_AD = u_dotvals[var];
      auto off = subview(offsets,var,ALL());
      auto cu = subview(u,ALL(),var,ALL());
      auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
      auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        ScalarT beta_u, beta_t;
        int stage = curr_stage(0);
        ScalarT deltat = dt(0);
        ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
        ScalarT timewt = 1.0/deltat/b_b(stage);
        ScalarT alpha_t = BDF(0)*timewt;
        for (size_type dof=team.team_rank(); dof<cu.extent(1); dof+=team.team_size() ) {
      
          // Seed the stage solution
          AD stageval = AD(maxDerivs,off(dof),cu(elem,dof));
          
          // Compute the evaluating solution
          beta_u = (1.0-alpha_u)*cu_prev(elem,dof,0);
          for (int s=0; s<stage; s++) {
            beta_u += b_A(stage,s)/b_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
          }
          u_AD(elem,dof) = alpha_u*stageval+beta_u;
          
          // Compute the time derivative
          beta_t = 0.0;
          for (size_type s=1; s<BDF.extent(0); s++) {
            beta_t += BDF(s)*cu_prev(elem,dof,s-1);
          }
          beta_t *= timewt;
          u_dot_AD(elem,dof) = alpha_t*stageval + beta_t;
        }
        
      });
    }
  }
  else if (seedwhat == 2) { // Seed one of the previous step solutions
    Kokkos::View<size_t[1],AssemblyDevice> sindex("seed index on device");
    auto host_sindex = Kokkos::create_mirror_view(sindex);
    host_sindex(0) = index;
    Kokkos::deep_copy(sindex,host_sindex);
    for (size_type var=0; var<u.extent(1); var++ ) {
      auto u_AD = uvals[var];
      auto u_dot_AD = u_dotvals[var];
      auto off = subview(offsets,var,ALL());
      auto cu = subview(u,ALL(),var,ALL());
      auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
      auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
    
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        AD beta_u, beta_t;
        int stage = curr_stage(0);
        ScalarT deltat = dt(0);
        ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
        ScalarT timewt = 1.0/deltat/b_b(stage);
        ScalarT alpha_t = BDF(0)*timewt;
        for (size_type dof=team.team_rank(); dof<cu.extent(1); dof+=team.team_size() ) {
          
          // Get the stage solution
          ScalarT stageval = cu(elem,dof);
          
          // Compute the evaluating solution
          AD u_prev_val = cu_prev(elem,dof,0);
          if (sindex(0) == 0) {
            u_prev_val = AD(maxDerivs,off(dof),cu_prev(elem,dof,0));
          }
          
          beta_u = (1.0-alpha_u)*u_prev_val;
          for (int s=0; s<stage; s++) {
            beta_u += b_A(stage,s)/b_b(s) * (cu_stage(elem,dof,s) - u_prev_val);
          }
          u_AD(elem,dof) = alpha_u*stageval+beta_u;
          
          // Compute and seed the time derivative
          beta_t = 0.0;
          for (size_type s=1; s<BDF.extent(0); s++) {
            AD u_prev_val = cu_prev(elem,dof,s-1);
            if (sindex(0) == (s-1)) {
              u_prev_val = AD(maxDerivs,off(dof),cu_prev(elem,dof,s-1));
            }
            beta_t += BDF(s)*u_prev_val;
          }
          beta_t *= timewt;
          u_dot_AD(elem,dof) = alpha_t*stageval + beta_t;
        }
      });
    }
  }
  else if (seedwhat == 3) { // Seed one of the previous stage solutions
    Kokkos::View<int[1],AssemblyDevice> sindex("seed index on device");
    auto host_sindex = Kokkos::create_mirror_view(sindex);
    host_sindex(0) = index;
    Kokkos::deep_copy(sindex,host_sindex);
    for (size_type var=0; var<u.extent(1); var++ ) {
      auto u_AD = uvals[var];
      auto u_dot_AD = u_dotvals[var];
      auto off = subview(offsets,var,ALL());
      auto cu = subview(u,ALL(),var,ALL());
      auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
      auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
    
      
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        AD beta_u, beta_t;
        int stage = curr_stage(0);
        ScalarT deltat = dt(0);
        ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
        ScalarT timewt = 1.0/deltat/b_b(stage);
        ScalarT alpha_t = BDF(0)*timewt;
        for (size_type dof=team.team_rank(); dof<cu.extent(1); dof+=team.team_size() ) {
          
          // Get the stage solution
          ScalarT stageval = cu(elem,dof);
          
          // Compute the evaluating solution
          ScalarT u_prev_val = cu_prev(elem,dof,0);
          
          beta_u = (1.0-alpha_u)*u_prev_val;
          for (int s=0; s<stage; s++) {
            AD u_stage_val = cu_stage(elem,dof,s);
            if (sindex(0) == s) {
              u_stage_val = AD(maxDerivs,off(dof),cu_stage(elem,dof,s));
            }
            beta_u += b_A(stage,s)/b_b(s) * (u_stage_val - u_prev_val);
          }
          u_AD(elem,dof) = alpha_u*stageval+beta_u;
          
          // Compute and seed the time derivative
          beta_t = 0.0;
          for (size_type s=1; s<BDF.extent(0); s++) {
            ScalarT u_prev_val = cu_prev(elem,dof,s-1);
            beta_t += BDF(s)*u_prev_val;
          }
          beta_t *= timewt;
          u_dot_AD(elem,dof) = alpha_t*stageval + beta_t;
        }
      });
    }
  }
  else { // Seed nothing
    for (size_type var=0; var<u.extent(1); var++ ) {
      auto u_AD = uvals[var];
      auto u_dot_AD = u_dotvals[var];
      auto off = subview(offsets,var,ALL());
      auto cu = subview(u,ALL(),var,ALL());
      auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
      auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
    
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        ScalarT beta_u, beta_t;
        int stage = curr_stage(0);
        ScalarT deltat = dt(0);
        ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
        ScalarT timewt = 1.0/deltat/b_b(stage);
        ScalarT alpha_t = BDF(0)*timewt;
        for (size_type dof=team.team_rank(); dof<cu.extent(1); dof+=team.team_size() ) {
          // Get the stage solution
          ScalarT stageval = cu(elem,dof);
          
          // Compute the evaluating solution
          beta_u = (1.0-alpha_u)*cu_prev(elem,dof,0);
          for (int s=0; s<stage; s++) {
            beta_u += b_A(stage,s)/b_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
          }
          u_AD(elem,dof) = alpha_u*stageval+beta_u;
          
          // Compute the time derivative
          beta_t = 0.0;
          for (size_type s=1; s<BDF.extent(0); s++) {
            beta_t += BDF(s)*cu_prev(elem,dof,s-1);
          }
          beta_t *= timewt;
          u_dot_AD(elem,dof) = alpha_t*stageval + beta_t;
        }
      });
    }
  }
  //Kokkos::fence();
  //AssemblyExec::execution_space().fence();
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for steady-state problems
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSteadySeeded(View_Sc3 u,
                                      const int & seedwhat) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
  for (size_type var=0; var<u.extent(1); var++ ) {
  
    auto u_AD = uvals[var];
    auto off = subview(offsets,var,ALL());
    auto cu = subview(u,ALL(),var,ALL());
    if (seedwhat == 1) {
      parallel_for("wkset steady soln",
                   RangePolicy<AssemblyExec>(0,u.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<u_AD.extent(1); dof++ ) {
          u_AD(elem,dof) = AD(maxDerivs,off(dof),cu(elem,dof));
        }
      });
    }
    else {
      parallel_for("wkset steady soln",
                   RangePolicy<AssemblyExec>(0,u.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<u_AD.extent(1); dof++ ) {
          u_AD(elem,dof) = cu(elem,dof);
        }
      });
    }
  }
  //Kokkos::fence();
  //AssemblyExec::execution_space().fence();
}


////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for steady-state problems
////////////////////////////////////////////////////////////////////////////////////

void workset::computeAuxSolnSteadySeeded(View_Sc3 aux,
                                         const int & seedwhat) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);

  /*
  // Needed so the device can access data-members (may be a better way)
  auto aux_AD = auxvals;
  auto off = aux_offsets;

  if (seedwhat == 1) {
    parallel_for(RangePolicy<AssemblyExec>(0,aux.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type var=0; var<aux.extent(1); var++ ) {
        for (size_type dof=0; dof<aux.extent(2); dof++ ) {
          aux_AD(elem,var,dof) = AD(maxDerivs,off(var,dof),aux(elem,var,dof));
        }
      }
    });
  }
  else {
    parallel_for("wkset steady soln",RangePolicy<AssemblyExec>(0,aux.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type var=0; var<aux.extent(1); var++ ) {
        for (size_type dof=0; dof<aux.extent(2); dof++ ) {
          aux_AD(elem,var,dof) = aux(elem,var,dof);
        }
      }
    });
  }
   */
  //Kokkos::fence();
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for steady-state problems
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamSteadySeeded(View_Sc3 param,
                                      const int & seedwhat) {
  
  if (numParams>0) {
    Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
    for (size_type var=0; var<param.extent(1); var++ ) {
      
      auto p_AD = pvals[var];
      auto off = subview(paramoffsets,var,ALL());
      auto cp = subview(param,ALL(),var,ALL());
      if (seedwhat == 3) {
        parallel_for("wkset steady soln",
                     RangePolicy<AssemblyExec>(0,param.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<p_AD.extent(1); dof++ ) {
            p_AD(elem,dof) = AD(maxDerivs,off(dof),cp(elem,dof));
          }
        });
      }
      else {
        parallel_for("wkset steady soln",
                     RangePolicy<AssemblyExec>(0,param.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          for (size_type dof=0; dof<p_AD.extent(1); dof++ ) {
            p_AD(elem,dof) = cp(elem,dof);
          }
        });
      }
    }
    //Kokkos::fence();
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions at volumetric ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnVolIP() {
  this->computeSoln(1,false);
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions at specified ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSoln(const int & type, const bool & onside) {
    
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnVolTimer);
    
    vector<string> list_HGRAD, list_HVOL, list_HDIV, list_HCURL, list_HFACE;
    vector<int> index_HGRAD, index_HVOL, index_HDIV, index_HCURL, index_HFACE;
    vector<int> basis_index;
    vector<View_AD2> solvals, soldotvals;
    bool include_transient = isTransient;
    if (onside) {
      include_transient = false;
    }
    
    if (type == 1) { // soln
      list_HGRAD = varlist_HGRAD;
      list_HVOL = varlist_HVOL;
      list_HDIV = varlist_HDIV;
      list_HCURL = varlist_HCURL;
      list_HFACE = varlist_HFACE;
      index_HGRAD = vars_HGRAD;
      index_HVOL = vars_HVOL;
      index_HDIV = vars_HDIV;
      index_HCURL = vars_HCURL;
      index_HFACE = vars_HFACE;
      basis_index = usebasis;
      solvals = uvals;
      soldotvals = u_dotvals;
    }
    else if (type == 2) { // discr. params
      list_HGRAD = paramvarlist_HGRAD;
      list_HVOL = paramvarlist_HVOL;
      list_HDIV = paramvarlist_HDIV;
      list_HCURL = paramvarlist_HCURL;
      list_HFACE = paramvarlist_HFACE;
      index_HGRAD = paramvars_HGRAD;
      index_HVOL = paramvars_HVOL;
      index_HDIV = paramvars_HDIV;
      index_HCURL = paramvars_HCURL;
      index_HFACE = paramvars_HFACE;
      include_transient = false;
      basis_index = paramusebasis;
      solvals = pvals;
    }
    else if (type == 3) { // aux
      list_HGRAD = auxvarlist_HGRAD;
      list_HVOL = auxvarlist_HVOL;
      list_HDIV = auxvarlist_HDIV;
      list_HCURL = auxvarlist_HCURL;
      list_HFACE = auxvarlist_HFACE;
      index_HGRAD = auxvars_HGRAD;
      index_HVOL = auxvars_HVOL;
      index_HDIV = auxvars_HDIV;
      index_HCURL = auxvars_HCURL;
      index_HFACE = auxvars_HFACE;
      basis_index = auxusebasis;
      solvals = auxvals;
      soldotvals = aux_dotvals;
    }
    
    /////////////////////////////////////////////////////////////////////
    // HGRAD
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<list_HGRAD.size(); i++) {
      string var = list_HGRAD[i];
      int varind = index_HGRAD[i];
      View_AD2 csol, csol_x, csol_y, csol_z, csol_t;
      View_Sc4 cbasis;
      View_Sc4 cbasis_grad;
      
      auto cuvals = solvals[varind];
      
      if (!onside) { // volumetric ip
        csol = this->getData(var);
        csol_x = this->getData("grad("+var+")[x]");
        if (dimension > 1) {
          csol_y = this->getData("grad("+var+")[y]");
        }
        if (dimension > 2) {
          csol_z = this->getData("grad("+var+")[z]");
        }
        cbasis = basis[basis_index[varind]];
        cbasis_grad = basis_grad[basis_index[varind]];
      }
      else { // boundary ip
        csol = this->getData(var+" side");
        csol_x = this->getData("grad("+var+")[x] side");
        if (dimension > 1) {
          csol_y = this->getData("grad("+var+")[y] side");
        }
        if (dimension > 2) {
          csol_z = this->getData("grad("+var+")[z] side");
        }
        cbasis = basis_side[basis_index[varind]];
        cbasis_grad = basis_grad_side[basis_index[varind]];
      }
      
      size_t teamSize = std::min(maxTeamSize,cbasis.extent(2));
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), teamSize, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        size_type dim = cbasis_grad.extent(3);
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csol(elem,pt) = 0.0;
          csol_x(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
            csol_x(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,0);
          }
          if (dim>1) {
            csol_y(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_y(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,1);
            }
          }
          if (dim>2) {
            csol_z(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_z(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,2);
            }
          }
        }
      });
      
      if (include_transient) { // transient terms only needed at volumetric ip
        auto csol_t = this->getData(var+"_t");
        auto cu_dotvals = soldotvals[varind];
        parallel_for("wkset soln ip HGRAD transient",
                     TeamPolicy<AssemblyExec>(cbasis.extent(0), teamSize, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            csol_t(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_t(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,0);
            }
          }
        });
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HVOL
    /////////////////////////////////////////////////////////////////////
    for (size_t i=0; i<list_HVOL.size(); i++) {
      
      string var = list_HVOL[i];
      int varind = index_HVOL[i];
      View_AD2 csol, csol_t;
      View_Sc4 cbasis;
      
      auto cuvals = solvals[varind];
      
      if (!onside) { // volumetric ip
        csol = this->getData(var);
        cbasis = basis[basis_index[varind]];
      }
      else { // boundary ip
        csol = this->getData(var+" side");
        cbasis = basis_side[basis_index[varind]];
      }
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csol(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
          }
        }
      });
      
      if (include_transient) { // transient terms only need at volumetric ip
        auto csol_t = this->getData(var+"_t");
        auto cu_dotvals = soldotvals[varind];
        parallel_for("wkset soln ip HGRAD transient",
                     TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            csol_t(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_t(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,0);
            }
          }
        });
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HDIV
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<list_HDIV.size(); i++) {
      string var = list_HDIV[i];
      int varind = index_HDIV[i];
      
      View_AD2 csolx, csoly, csolz, csol_div;
      View_Sc4 cbasis;
      View_Sc3 cbasis_div;
      
      if (!onside) {
        csolx = this->getData(var+"[x]");
        if (dimension > 1) {
          csoly = this->getData(var+"[y]");
        }
        if (dimension > 2) {
          csolz = this->getData(var+"[z]");
        }
        csol_div = this->getData("div("+var+")");
        cbasis = basis[basis_index[varind]];
        cbasis_div = basis_div[basis_index[varind]];
      }
      else {
        csolx = this->getData(var+"[x] side");
        if (dimension > 1) {
          csoly = this->getData(var+"[y] side");
        }
        if (dimension > 2) {
          csolz = this->getData(var+"[z] side");
        }
        cbasis = basis_side[basis_index[varind]];
      }
      
      auto cuvals = solvals[varind];
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        size_type dim = cbasis.extent(3);
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csolx(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csolx(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
          }
          if (dim>1) {
            csoly(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csoly(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,1);
            }
          }
          if (dim>2) {
            csolz(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csolz(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,2);
            }
          }
        }
      });
      if (!onside) {
        parallel_for("wkset soln ip HGRAD",
                     TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            csol_div(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_div(elem,pt) += cuvals(elem,dof)*cbasis_div(elem,dof,pt);
            }
          }
        });
      }
      
      if (include_transient) {
        View_AD2 csol_tx, csol_ty, csol_tz;
        csol_tx = this->getData(var+"_t[x]");
        if (dimension > 1) {
          csol_ty = this->getData(var+"_t[y]");
        }
        if (dimension > 2) {
          csol_tz = this->getData(var+"_t[z]");
        }
        auto cu_dotvals = soldotvals[varind];
        parallel_for("wkset soln ip HGRAD transient",
                     TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          size_type dim = cbasis.extent(3);
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            csol_tx(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_tx(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,0);
            }
            if (dim>1) {
              csol_ty(elem,pt) = 0.0;
              for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
                csol_ty(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,1);
              }
            }
            if (dim>2) {
              csol_tz(elem,pt) = 0.0;
              for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
                csol_tz(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,2);
              }
            }
          }
        });
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HCURL
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<list_HCURL.size(); i++) {
      string var = list_HCURL[i];
      int varind = index_HCURL[i];
      
      View_AD2 csolx, csoly, csolz, csol_curlx,csol_curly,csol_curlz;
      View_Sc4 cbasis, cbasis_curl;
      
      if (!onside) {
        csolx = this->getData(var+"[x]");
        csol_curlx = this->getData("curl("+var+")[x]");
        if (dimension > 1) {
          csoly = this->getData(var+"[y]");
          csol_curly = this->getData("curl("+var+")[y]");
        }
        if (dimension > 2) {
          csolz = this->getData(var+"[z]");
          csol_curlz = this->getData("curl("+var+")[z]");
        }
        cbasis = basis[basis_index[varind]];
        cbasis_curl = basis_curl[basis_index[varind]];
      }
      else {
        csolx = this->getData(var+"[x] side");
        if (dimension > 1) {
          csoly = this->getData(var+"[y] side");
        }
        if (dimension > 1) {
          csolz = this->getData(var+"[z] side");
        }
        cbasis = basis_side[basis_index[varind]];
      }
      
      auto cuvals = solvals[varind];
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        size_type dim = cbasis.extent(3);
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csolx(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csolx(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
          }
          if (dim>1) {
            csoly(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csoly(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,1);
            }
          }
          if (dim>2) {
            csolz(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csolz(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,2);
            }
          }
        }
      });
      
      if (!onside) { // no curl on boundary?
        parallel_for("wkset soln ip HGRAD",
                     TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          size_type dim = cbasis.extent(3);
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            csol_curlx(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_curlx(elem,pt) += cuvals(elem,dof)*cbasis_curl(elem,dof,pt,0);
            }
            if (dim>1) {
              csol_curly(elem,pt) = 0.0;
              for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
                csol_curly(elem,pt) += cuvals(elem,dof)*cbasis_curl(elem,dof,pt,1);
              }
            }
            if (dim>2) {
              csol_curlz(elem,pt) = 0.0;
              for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
                csol_curlz(elem,pt) += cuvals(elem,dof)*cbasis_curl(elem,dof,pt,2);
              }
            }
          }
        });
      }
      if (include_transient) {
        View_AD2 csol_tx, csol_ty, csol_tz;
        csol_tx = this->getData(var+"_t[x]");
        if (dimension > 1) {
          csol_ty = this->getData(var+"_t[y]");
        }
        if (dimension > 2) {
          csol_tz = this->getData(var+"_t[z]");
        }
        auto cu_dotvals = soldotvals[varind];
        parallel_for("wkset soln ip HGRAD transient",
                     TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          size_type dim = cbasis.extent(3);
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            csol_tx(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_tx(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,0);
            }
            if (dim>1) {
              csol_ty(elem,pt) = 0.0;
              for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
                csol_ty(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,1);
              }
            }
            if (dim>2) {
              csol_tz(elem,pt) = 0.0;
              for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
                csol_tz(elem,pt) += cu_dotvals(elem,dof)*cbasis(elem,dof,pt,2);
              }
            }
          }
        });
        
      }
    }
    
    /////////////////////////////////////////////////////////////////////
    // HFACE
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<list_HFACE.size(); i++) {
      if (onside) {
        string var = list_HFACE[i];
        int varind = index_HFACE[i];
        View_AD2 csol;
        View_Sc4 cbasis;
        
        auto cuvals = solvals[varind];
        
        csol = this->getData(var+" side");
        cbasis = basis_side[basis_index[varind]];
        
        parallel_for("wkset soln ip HFACE",
                     TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            csol(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
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

void workset::computeParamVolIP() {
  
  if (numParams > 0) {
    
    //this->computeParamSteadySeeded(param,seedwhat);
    this->computeSoln(2,false);
    
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSideIP() {
  this->computeSoln(1,true);
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the discretized parameters at the side ip
////////////////////////////////////////////////////////////////////////////////////

void workset::computeParamSideIP() {
  
  if (numParams>0) {
    //this->computeParamSteadySeeded(param,seedwhat);
    this->computeSoln(2,true);
  }
  
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
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    
    /////////////////////////////////////////////////////////////////////
    // HGRAD
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<varlist_HGRAD.size(); i++) {
      string var = varlist_HGRAD[i];
      int varind = vars_HGRAD[i];
      
      auto cuvals = uvals[varind];
      
      auto csol = this->getData(var+" side");
      auto csol_x = this->getData("grad("+var+")[x] side");
      auto csol_y = this->getData("grad("+var+")[y] side");
      auto csol_z = this->getData("grad("+var+")[z] side");
      auto cbasis = basis_side[usebasis[varind]];
      auto cbasis_grad = basis_grad_side[usebasis[varind]];
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        size_type dim = cbasis_grad.extent(3);
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csol(elem,pt) = 0.0;
          csol_x(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
            csol_x(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,0);
          }
          if (dim>1) {
            csol_y(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_y(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,1);
            }
          }
          if (dim>2) {
            csol_z(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_z(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,2);
            }
          }
        }
      });
    }
    
    /////////////////////////////////////////////////////////////////////
    // HVOL
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HVOL.size(); i++) {
      string var = varlist_HVOL[i];
      int varind = vars_HVOL[i];
      
      auto cuvals = uvals[varind];
      
      auto csol = this->getData(var+" side");
      auto cbasis = basis_side[usebasis[varind]];
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csol(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
          }
        }
      });
    }
    
    /////////////////////////////////////////////////////////////////////
    // HDIV
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HDIV.size(); i++) {
      string var = varlist_HDIV[i];
      int varind = vars_HDIV[i];
      
      auto cuvals = uvals[varind];
      
      auto csolx = this->getData(var+"[x] side");
      auto csoly = this->getData(var+"[y] side");
      auto csolz = this->getData(var+"[z] side");
      auto cbasis = basis_side[usebasis[varind]];
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        size_type dim = cbasis.extent(3);
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csolx(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csolx(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
          }
          if (dim>1) {
            csoly(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csoly(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,1);
            }
          }
          if (dim>2) {
            csolz(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csolz(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,2);
            }
          }
        }
      });
    }
    
    /////////////////////////////////////////////////////////////////////
    // HCURL
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<vars_HDIV.size(); i++) {
      string var = varlist_HDIV[i];
      int varind = vars_HDIV[i];
      
      auto cuvals = uvals[varind];
      
      auto csolx = this->getData(var+"[x] side");
      auto csoly = this->getData(var+"[y] side");
      auto csolz = this->getData(var+"[z] side");
      auto cbasis = basis_side[usebasis[varind]];
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        size_type dim = cbasis.extent(3);
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csolx(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csolx(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
          }
          if (dim>1) {
            csoly(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csoly(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,1);
            }
          }
          if (dim>2) {
            csolz(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csolz(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,2);
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

void workset::addAux(const vector<string> & auxvars, Kokkos::View<int**,AssemblyDevice> aoffs) {
  aux_offsets = aoffs;
  aux_varlist = auxvars;
  numAux = aux_varlist.size();
  flux = View_AD3("flux",numElem,numAux,numsideip, maxDerivs);
  
  if (numAux > 0) {
    size_t maxAux = aux_offsets.extent(0)*aux_offsets.extent(1);
    if (maxAux > maxRes) {
      maxRes = maxAux;
      res = View_AD2("residual",numElem, maxRes, maxDerivs);
    }
  }

  for (size_t i=0; i<aux_varlist.size(); ++i) {
    string var = aux_varlist[i];
    this->addData("aux "+var,numElem,numip);
    this->addData("aux "+var+" side",numElem,numsideip);
  }
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

//////////////////////////////////////////////////////////////
// Add a data view
//////////////////////////////////////////////////////////////

void workset::addData(const string & label, const int & dim0, const int & dim1) {
  data.push_back(View_AD2(label,0,dim1));
  data_labels.push_back(label);
  data_usage.push_back(0);
}

void workset::addDataSc(const string & label, const int & dim0, const int & dim1) {
  data_Sc.push_back(View_Sc2(label,0,dim1));
  data_Sc_labels.push_back(label);
  data_Sc_usage.push_back(0);
}

int workset::addIntegratedQuantities(const int & nRequested) {

  int startingIndex = this->integrated_quantities.extent(0);

  // this should only be called when setting up the physics module
  // in the case of multiple physics defined on the same block requesting IQs,
  // integrated_quantities will get re-initialized until it's big
  // enough for all of them (we anticipate nTotal to be small here).

  this->integrated_quantities = 
    View_Sc1("integrated quantities",startingIndex+nRequested);

  return startingIndex;

}

//////////////////////////////////////////////////////////////
// Extract a VIEW_AD2 (2-dimensional array with AD entries)
//////////////////////////////////////////////////////////////

View_AD2 workset::getData(const string & label) {
  
  //Teuchos::TimeMonitor basistimer(*worksetgetDataTimer);
  
  bool found = false;
  size_t ind = 0;
  while (!found && ind<data_labels.size()) {
    if (label == data_labels[ind]) {
      found = true;
      data_usage[ind] += 1;
    }
    else {
      ++ind;
    }
  }
  if (!found) {
    std::cout << "Error: could not find data named " << label << std::endl;
    this->printMetaData();
  }
  this->checkDataAllocation(ind);
  return data[ind];
  
}

void workset::checkDataAllocation(const size_t & ind) {
  if (data[ind].extent(0) < maxElem) {
    Kokkos::resize(data[ind],maxElem,data[ind].extent(1));
  }
}

void workset::checkDataScAllocation(const size_t & ind) {
  if (data_Sc[ind].extent(0) < maxElem) {
    Kokkos::resize(data_Sc[ind],maxElem,data_Sc[ind].extent(1));
  }
}

//////////////////////////////////////////////////////////////
// Extract a View_Sc2 (2-dimensional array with ScalarT entries)
//////////////////////////////////////////////////////////////

View_Sc2 workset::getDataSc(const string & label) {
  
  //Teuchos::TimeMonitor basistimer(*worksetgetDataScTimer);
  
  bool found = false;
  size_t ind = 0;
  while (!found && ind<data_Sc_labels.size()) {
    if (label == data_Sc_labels[ind]) {
      found = true;
      data_Sc_usage[ind] += 1;
    }
    else {
      ++ind;
    }
  }
  
  if (!found) {
    std::cout << "Error: could not find scalar data named " << label << std::endl;
  }
  
  this->checkDataScAllocation(ind);
  return data_Sc[ind];
  
}

//////////////////////////////////////////////////////////////
// Extract a View_Sc2 (2-dimensional array with ScalarT entries)
//////////////////////////////////////////////////////////////

size_t workset::getDataScIndex(const string & label) {
  
  //Teuchos::TimeMonitor basistimer(*worksetgetDataScIndexTimer);
  
  bool found = false;
  size_t ind = 0;
  while (!found && ind<data_Sc_labels.size()) {
    if (label == data_Sc_labels[ind]) {
      found = true;
      data_Sc_usage[ind] += 1;
    }
    else {
      ++ind;
    }
  }
  
  if (!found) {
    std::cout << "Error: could not find scalar data named " << label << std::endl;
  }
  
  this->checkDataScAllocation(ind);
  
  return ind;
  
}


//////////////////////////////////////////////////////////////
// Another method to extract a View_AD2
//////////////////////////////////////////////////////////////

void workset::get(const string & label, View_AD2 & dataout) {
  
  //Teuchos::TimeMonitor basistimer(*worksetgetTimer);
  
  bool found = false;
  size_t ind = 0;
  while (!found && ind<data_labels.size()) {
    if (label == data_labels[ind]) {
      found = true;
      data_usage[ind] += 1;
      this->checkDataAllocation(ind);
      dataout = data[ind];
    }
    else {
      ++ind;
    }
  }
  if (!found) {
    std::cout << "Warning: could not find data with label: " << label << std::endl;
    std::cout << "An error will probably occur if this view is accessed" << std::endl;
    this->printMetaData();
  }
}

//////////////////////////////////////////////////////////////
// Another method to extract a View_Sc2
//////////////////////////////////////////////////////////////

void workset::get(const string & label, View_Sc2 & dataout) {
  
  //Teuchos::TimeMonitor basistimer(*worksetgetTimer);
  
  bool found = false;
  size_t ind = 0;
  while (!found && ind<data_Sc_labels.size()) {
    if (label == data_Sc_labels[ind]) {
      found = true;
      data_Sc_usage[ind] += 1;
      this->checkDataScAllocation(ind);
      dataout = data_Sc[ind];
    }
    else {
      ++ind;
    }
  }
  
  if (!found) {
    std::cout << "Warning: could not find data with label: " << label << std::endl;
    std::cout << "An error will probably occur if this view is accessed" << std::endl;
    this->printMetaData();
  }
}

//////////////////////////////////////////////////////////////
// Function to determine which basis a variable uses
//////////////////////////////////////////////////////////////

bool workset::findBasisIndex(const string & var, int & basisindex) {
  bool found = false;
  int index;
  found = this->isVar(var,index);
  if (found) {
    basisindex = usebasis[index];
  }
  else {
    found = this->isParameter(var,index);
    if (found) {
      basisindex = paramusebasis[index];
    }
    else {
      found = this->isAux(var,index);
      if (found) {
        basisindex = auxusebasis[index];
      }
      else {
        std::cout << "Warning: could not find basis for: " << var << std::endl;
        std::cout << "An error will probably occur if this view is accessed" << std::endl;
      }
    }
  }
  return found;
}

//////////////////////////////////////////////////////////////
// Check if a string is a variable
//////////////////////////////////////////////////////////////

bool workset::isVar(const string & var, int & index) {
  bool found = false;
  size_t varindex = 0;
  while (!found && varindex<varlist.size()) {
    if (varlist[varindex] == var) {
      found = true;
      index = varindex;
    }
  }
  return found;
}

//////////////////////////////////////////////////////////////
// Check if a string is a discretized parameter
//////////////////////////////////////////////////////////////

bool workset::isParameter(const string & var, int & index) {
  bool found = false;
  size_t varindex = 0;
  while (!found && varindex<param_varlist.size()) {
    if (param_varlist[varindex] == var) {
      found = true;
      index = varindex;
    }
  }
  return found;
}

//////////////////////////////////////////////////////////////
// Check if a string is an aux variable
//////////////////////////////////////////////////////////////

bool workset::isAux(const string & var, int & index) {
  bool found = false;
  size_t varindex = 0;
  while (!found && varindex<aux_varlist.size()) {
    if (aux_varlist[varindex] == var) {
      found = true;
      index = varindex;
    }
  }
  return found;
}

//////////////////////////////////////////////////////////////
// Get the AD residual
//////////////////////////////////////////////////////////////

View_AD2 workset::getResidual() {
  return res;
}

//////////////////////////////////////////////////////////////
// Get the integration weights (interior)
//////////////////////////////////////////////////////////////

View_Sc2 workset::getWeights() {
  return wts;
}

//////////////////////////////////////////////////////////////
// Get the integration weights (boundary)
//////////////////////////////////////////////////////////////

View_Sc2 workset::getSideWeights() {
  return wts_side;
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasis(const string & var) {

  //Teuchos::TimeMonitor basistimer(*worksetgetBasisTimer);
  
  View_Sc4 dataout;
  int basisindex;
  
  bool found = this->findBasisIndex(var, basisindex);
  if (found) {
    dataout = this->getBasis(basisindex);
  }
  return dataout;
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by an index (faster)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasis(const int & index) {
  return basis[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisGrad(const string & var) {

  //Teuchos::TimeMonitor basistimer(*worksetgetBasisTimer);
  
  View_Sc4 dataout;
  int basisindex;
  
  bool found = this->findBasisIndex(var, basisindex);
  if (found) {
    dataout = this->getBasisGrad(basisindex);
  }
  return dataout;
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by an index (faster)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisGrad(const int & index) {
  return basis_grad[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

View_Sc3 workset::getBasisDiv(const string & var) {

  //Teuchos::TimeMonitor basistimer(*worksetgetBasisTimer);
  
  View_Sc3 dataout;
  int basisindex;
  
  bool found = this->findBasisIndex(var, basisindex);
  if (found) {
    dataout = this->getBasisDiv(basisindex);
  }
  return dataout;
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by an index (faster)
//////////////////////////////////////////////////////////////

View_Sc3 workset::getBasisDiv(const int & index) {
  return basis_div[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisCurl(const string & var) {

  //Teuchos::TimeMonitor basistimer(*worksetgetBasisTimer);
  
  View_Sc4 dataout;
  int basisindex;
  
  bool found = this->findBasisIndex(var, basisindex);
  if (found) {
    dataout = this->getBasisCurl(basisindex);
  }
  return dataout;
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by an index (faster)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisCurl(const int & index) {
  return basis_curl[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisSide(const string & var) {

  //Teuchos::TimeMonitor basistimer(*worksetgetBasisTimer);
  
  View_Sc4 dataout;
  int basisindex;
  
  bool found = this->findBasisIndex(var, basisindex);
  if (found) {
    dataout = this->getBasisSide(basisindex);
  }
  return dataout;
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by an index (faster)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisSide(const int & index) {
  return basis_side[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisGradSide(const string & var) {

  //Teuchos::TimeMonitor basistimer(*worksetgetBasisTimer);
  
  View_Sc4 dataout;
  int basisindex;
  
  bool found = this->findBasisIndex(var, basisindex);
  if (found) {
    dataout = this->getBasisGradSide(basisindex);
  }
  return dataout;
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by an index (faster)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisGradSide(const int & index) {
  return basis_grad_side[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisCurlSide(const string & var) {

  //Teuchos::TimeMonitor basistimer(*worksetgetBasisTimer);
  
  View_Sc4 dataout;
  int basisindex;
  
  bool found = this->findBasisIndex(var, basisindex);
  if (found) {
    dataout = this->getBasisCurlSide(basisindex);
  }
  return dataout;
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by an index (faster)
//////////////////////////////////////////////////////////////

View_Sc4 workset::getBasisCurlSide(const int & index) {
  return basis_curl_side[index];
}

//////////////////////////////////////////////////////////////
// Extract all of the offsets
//////////////////////////////////////////////////////////////

Kokkos::View<int**,AssemblyDevice> workset::getOffsets() {
  return offsets;
}

//////////////////////////////////////////////////////////////
// Extract the offsets for a particular variable
//////////////////////////////////////////////////////////////

Kokkos::View<int*,Kokkos::LayoutStride,AssemblyDevice> workset::getOffsets(const string & var) {
  
  Kokkos::View<int*,Kokkos::LayoutStride,AssemblyDevice> reqdata;
  
  int index;
  bool found = this->isVar(var, index);
  if (found) {
    reqdata = subview(offsets,index,ALL());
  }
  else {
    std::cout << "Warning: could not find variable: " << var << std::endl;
    std::cout << "An error will probably occur if this view is accessed" << std::endl;
  }
  return reqdata;
}

//////////////////////////////////////////////////////////////
// Print the meta-data associated with the stored View_AD2 and View_Sc2
//////////////////////////////////////////////////////////////

void workset::printMetaData() {
  std::cout << "Number of View_AD2 stored: " << data.size() << std::endl << std::endl;
  std::cout << "Label                | dim0 | dim1 | Num. requests" << std::endl;
  
  for (size_t i=0; i<data.size(); ++i) {
    string pad = "";
    for (size_t j=0; j<20-data_labels[i].size(); ++j) {
      pad += " ";
    }
    std::cout << data_labels[i] << pad << "   " << data[i].extent(0) << "     " << data[i].extent(1) << "     " << data_usage[i] << std::endl;
  }
  std::cout << std::endl << std::endl;
  
  std::cout << "Number of View_Sc2 stored: " << data_Sc.size() << std::endl;
  std::cout << "Label                | dim0 | dim1 | Num. requests" << std::endl;
  
  for (size_t i=0; i<data_Sc.size(); ++i) {
    string pad = "";
    for (size_t j=0; j<20-data_Sc_labels[i].size(); ++j) {
      pad += " ";
    }
    std::cout << data_Sc_labels[i] << pad << "   " << data_Sc[i].extent(0) << "     " << data_Sc[i].extent(1) << "     " << data_Sc_usage[i] << std::endl;
  }
  
}

//////////////////////////////////////////////////////////////
// Copy data carefully
//////////////////////////////////////////////////////////////

template<class V1, class V2>
void workset::copyData(V1 view1, V2 view2) {
  
  //Teuchos::TimeMonitor functimer(*worksetcopyDataTimer);
  
  // Copy data from view2 into view1
  // Both are rank-2 and second dimensions are the same
  // However, view2 may be shorter in first dimension
  if (view1.extent(0) == view2.extent(0)) {
    deep_copy(view1,view2);
  }
  else {
    //deep_copy(view1,0.0);
    parallel_for("wkset copy data",
                 RangePolicy<AssemblyExec>(0,view2.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<view1.extent(1); ++pt) {
        view1(elem,pt) = view2(elem,pt);
      }
    });
  }
}

//////////////////////////////////////////////////////////////
// Set the integration points
//////////////////////////////////////////////////////////////

void workset::setIP(vector<View_Sc2> & newip, const string & pfix) {
  size_t dim = newip.size();
  
  size_t xind = this->getDataScIndex("x"+pfix);
  data_Sc[xind] = newip[0];
  
  if (dim > 1) {
    size_t yind = this->getDataScIndex("y"+pfix);
    data_Sc[yind] = newip[1];
  }
  if (dim > 2) {
    size_t zind = this->getDataScIndex("z"+pfix);
    data_Sc[zind] = newip[2];
  }
}

//////////////////////////////////////////////////////////////
// Set the side/face normals
//////////////////////////////////////////////////////////////

void workset::setNormals(vector<View_Sc2> & newnormals) {
  size_t dim = newnormals.size();
  
  size_t nxind = this->getDataScIndex("nx side");
  data_Sc[nxind] = newnormals[0];
  
  if (dim > 1) {
    size_t nyind = this->getDataScIndex("ny side");
    data_Sc[nyind] = newnormals[1];
  }
  if (dim > 2) {
    size_t nzind = this->getDataScIndex("nz side");
    data_Sc[nzind] = newnormals[2];
  }
}

//////////////////////////////////////////////////////////////
// Set the side/face tangents
//////////////////////////////////////////////////////////////

void workset::setTangents(vector<View_Sc2> & newtangents) {
  size_t dim = newtangents.size();
  
  size_t txind = this->getDataScIndex("tx side");
  data_Sc[txind] = newtangents[0];
  
  if (dim > 1) {
    size_t tyind = this->getDataScIndex("ty side");
    data_Sc[tyind] = newtangents[1];
  }
  if (dim > 2) {
    size_t tzind = this->getDataScIndex("tz side");
    data_Sc[tzind] = newtangents[2];
  }
}


/////////////////////////////////////////////////////////////
// Set the integration points
//////////////////////////////////////////////////////////////

//void workset::seth(View_Sc2 newh, const string & pfix) {
//  size_t hind = this->getDataScIndex("h"+pfix);
//  data_Sc[hind] = newh;
//  h = newh;
//}

//////////////////////////////////////////////////////////////
// Set the solutions
//////////////////////////////////////////////////////////////

void workset::setSolution(View_AD4 newsol, const string & pfix) {
  // newsol has dims numElem x numvars x numip x dimension
  // however, this numElem may be smaller than the size of the data arrays
  
  for (size_t i=0; i<varlist_HGRAD.size(); i++) {
    string var = varlist_HGRAD[i];
    int varind = vars_HGRAD[i];
    auto csol = this->getData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<varlist_HVOL.size(); i++) {
    string var = varlist_HVOL[i];
    int varind = vars_HVOL[i];
    auto csol = this->getData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<varlist_HFACE.size(); i++) {
    string var = varlist_HFACE[i];
    int varind = vars_HFACE[i];
    auto csol = this->getData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<varlist_HDIV.size(); i++) {
    string var = varlist_HDIV[i];
    int varind = vars_HDIV[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData(var+"[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getData(var+"[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getData(var+"[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  for (size_t i=0; i<varlist_HCURL.size(); i++) {
    string var = varlist_HCURL[i];
    int varind = vars_HCURL[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData(var+"[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getData(var+"[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getData(var+"[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  
}

//////////////////////////////////////////////////////////////
// Set the solution GRADs
//////////////////////////////////////////////////////////////

void workset::setSolutionGrad(View_AD4 newsol, const string & pfix) {
  for (size_t i=0; i<varlist_HGRAD.size(); i++) {
    string var = varlist_HGRAD[i];
    int varind = vars_HGRAD[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData("grad("+var+")[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getData("grad("+var+")[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getData("grad("+var+")[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
}

//////////////////////////////////////////////////////////////
// Set the solution DIVs
//////////////////////////////////////////////////////////////

void workset::setSolutionDiv(View_AD3 newsol, const string & pfix) {
  for (size_t i=0; i<varlist_HDIV.size(); i++) {
    string var = varlist_HDIV[i];
    int varind = vars_HDIV[i];
    auto csol = this->getData("div("+var+")");
    auto cnsol = subview(newsol,ALL(),varind,ALL());
    this->copyData(csol,cnsol);
  }
}

//////////////////////////////////////////////////////////////
// Set the solution CURLs
//////////////////////////////////////////////////////////////

void workset::setSolutionCurl(View_AD4 newsol, const string & pfix) {
  for (size_t i=0; i<varlist_HCURL.size(); i++) {
    string var = varlist_HCURL[i];
    int varind = vars_HCURL[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData("curl("+var+")[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getData("curl("+var+")[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getData("curl("+var+")[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
}

//////////////////////////////////////////////////////////////
// Set the solution at a point
//////////////////////////////////////////////////////////////

void workset::setSolutionPoint(View_AD2 newsol) {
  // newsol has dims numElem x numvars x numip x dimension
  for (size_t i=0; i<varlist_HGRAD.size(); i++) {
    string var = varlist_HGRAD[i];
    int varind = vars_HGRAD[i];
    auto csol = this->getData(var+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
  }
  for (size_t i=0; i<varlist_HVOL.size(); i++) {
    string var = varlist_HVOL[i];
    int varind = vars_HVOL[i];
    auto csol = this->getData(var+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
  }
  for (size_t i=0; i<varlist_HDIV.size(); i++) {
    string var = varlist_HDIV[i];
    int varind = vars_HDIV[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData(var+"[x] point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getData(var+"[y] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getData(var+"[z] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(2);
      });
    }
  }
  for (size_t i=0; i<varlist_HCURL.size(); i++) {
    string var = varlist_HCURL[i];
    int varind = vars_HCURL[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData(var+"[x] point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getData(var+"[y] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getData(var+"[z] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(2);
      });
    }
  }
  
}

//////////////////////////////////////////////////////////////
// Set the solution at a point
//////////////////////////////////////////////////////////////

void workset::setSolutionGradPoint(View_AD2 newsol) {
  // newsol has dims numElem x numvars x numip x dimension
  for (size_t i=0; i<varlist_HGRAD.size(); i++) {
    string var = varlist_HGRAD[i];
    int varind = vars_HGRAD[i];
    size_type dim = newsol.extent(1);
    auto csol = this->getData("grad("+var+")[x]"+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getData("grad("+var+")[y]"+" point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getData("grad("+var+")[z]"+" point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(2);
      });
    }
  }

}

//////////////////////////////////////////////////////////////
// Set the parameter solutions
//////////////////////////////////////////////////////////////

void workset::setParam(View_AD4 newsol, const string & pfix) {
  // newsol has dims numElem x numvars x numip x dimension
  // however, this numElem may be smaller than the size of the data arrays
  
  for (size_t i=0; i<paramvarlist_HGRAD.size(); i++) {
    string var = paramvarlist_HGRAD[i];
    int varind = paramvars_HGRAD[i];
    auto csol = this->getData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<paramvarlist_HVOL.size(); i++) {
    string var = paramvarlist_HVOL[i];
    int varind = paramvars_HVOL[i];
    auto csol = this->getData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<paramvarlist_HFACE.size(); i++) {
    string var = paramvarlist_HFACE[i];
    int varind = paramvars_HFACE[i];
    auto csol = this->getData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<paramvarlist_HDIV.size(); i++) {
    string var = paramvarlist_HDIV[i];
    int varind = paramvars_HDIV[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData(var+"[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getData(var+"[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getData(var+"[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  for (size_t i=0; i<varlist_HCURL.size(); i++) {
    string var = paramvarlist_HCURL[i];
    int varind = paramvars_HCURL[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData(var+"[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getData(var+"[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getData(var+"[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  
}

//////////////////////////////////////////////////////////////
// Set the solution at a point
//////////////////////////////////////////////////////////////

void workset::setParamPoint(View_AD2 newsol) {
  // newsol has dims numElem x numvars x numip x dimension
  for (size_t i=0; i<paramvarlist_HGRAD.size(); i++) {
    string var = paramvarlist_HGRAD[i];
    int varind = paramvars_HGRAD[i];
    auto csol = this->getData(var+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
  }
  for (size_t i=0; i<paramvarlist_HVOL.size(); i++) {
    string var = paramvarlist_HVOL[i];
    int varind = paramvars_HVOL[i];
    auto csol = this->getData(var+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
  }
  for (size_t i=0; i<paramvarlist_HDIV.size(); i++) {
    string var = paramvarlist_HDIV[i];
    int varind = paramvars_HDIV[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData(var+"[x] point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getData(var+"[y] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getData(var+"[z] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(2);
      });
    }
  }
  for (size_t i=0; i<paramvarlist_HCURL.size(); i++) {
    string var = paramvarlist_HCURL[i];
    int varind = paramvars_HCURL[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getData(var+"[x] point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getData(var+"[y] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getData(var+"[z] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(2);
      });
    }
  }
  
}

//////////////////////////////////////////////////////////////
// Set the solution at a point
//////////////////////////////////////////////////////////////

void workset::setParamGradPoint(View_AD2 newsol) {
  // newsol has dims numElem x numvars x numip x dimension
  for (size_t i=0; i<paramvarlist_HGRAD.size(); i++) {
    string var = paramvarlist_HGRAD[i];
    int varind = paramvars_HGRAD[i];
    size_type dim = newsol.extent(1);
    auto csol = this->getData("grad("+var+")[x]"+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getData("grad("+var+")[y]"+" point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getData("grad("+var+")[z]"+" point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(2);
      });
    }
  }

}
