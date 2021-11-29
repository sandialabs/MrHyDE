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

workset::workset(const vector<int> & cellinfo,
                 const vector<size_t> & numVars_,
                 const bool & isTransient_,
                 const vector<string> & basis_types_,
                 const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
                 const topo_RCP & topo) :
isTransient(isTransient_), celltopo(topo),
basis_types(basis_types_), basis_pointers(basis_pointers_) {

  isInitialized = true;
  
  // Settings that should not change
  dimension = cellinfo[0];
  numVars = numVars_;
  numParams = cellinfo[1];
  numAux = 0;
  numElem = cellinfo[2];
  usebcs = true;
  numip = cellinfo[3];
  numsideip = cellinfo[4];
  numSets = cellinfo[5];
  
  if (dimension == 2) {
    numsides = celltopo->getSideCount();
  }
  else if (dimension == 3) {
    numsides = celltopo->getFaceCount();
  }
  
  maxElem = numElem;
  time = 0.0;
  deltat = 1.0;
  current_stage = 0;
  current_set = 0;
  //var_bcs = set_var_bcs[0];
  
  // Add scalar fields to store ip, normals, etc.
  scalar_fields.push_back(ScalarField("x",0,numip,false,false));
  scalar_fields.push_back(ScalarField("y",0,numip,false,false));
  scalar_fields.push_back(ScalarField("z",0,numip,false,false));
  
  scalar_fields.push_back(ScalarField("x side",0,numsideip,true,false));
  scalar_fields.push_back(ScalarField("y side",0,numsideip,true,false));
  scalar_fields.push_back(ScalarField("z side",0,numsideip,true,false));
  
  scalar_fields.push_back(ScalarField("nx side",0,numsideip,true,false));
  scalar_fields.push_back(ScalarField("ny side",0,numsideip,true,false));
  scalar_fields.push_back(ScalarField("nz side",0,numsideip,true,false));
  
  scalar_fields.push_back(ScalarField("tx side",0,numsideip,true,false));
  scalar_fields.push_back(ScalarField("ty side",0,numsideip,true,false));
  scalar_fields.push_back(ScalarField("tz side",0,numsideip,true,false));
  
  scalar_fields.push_back(ScalarField("x point",0,1,false,true));
  scalar_fields.push_back(ScalarField("y point",0,1,false,true));
  scalar_fields.push_back(ScalarField("z point",0,1,false,true));
  
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
  
  basis = vector<View_Sc4>(basis_pointers.size());
  basis_grad = vector<View_Sc4>(basis_pointers.size());
  basis_curl = vector<View_Sc4>(basis_pointers.size());
  basis_div = vector<View_Sc3>(basis_pointers.size());
  
  basis_side = vector<View_Sc4>(basis_pointers.size());
  basis_grad_side = vector<View_Sc4>(basis_pointers.size());
  basis_curl_side = vector<View_Sc4>(basis_pointers.size());
  
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
  
  // Start with the number of active scalar parameters (typically small)
  maxRes = numParams;
  
  // Check the number of DOF for each variable
  for (size_t set=0; set<numSets; ++set) {
    maxRes = std::max(maxRes, set_offsets[set].extent(0)*set_offsets[set].extent(1));
  }
  
  // Check the number of DOF for each discretized parameter
  if (paramusebasis.size() > 0) {
    maxRes = std::max(maxRes,paramoffsets.extent(0)*paramoffsets.extent(1));
  }
  
  size_t totalvars = 0;
  for (size_t set=0; set<numSets; ++set) {
    totalvars += set_varlist[set].size();
  }
  
  uvals = vector<View_AD2>(totalvars);
  if (isTransient) {
    u_dotvals = vector<View_AD2>(totalvars);
  }
  
  res = View_AD2("residual",numElem, maxRes);
  
  size_t uprog = 0;
  for (size_t set=0; set<numSets; ++set) {
    
    vector<size_t> set_uindex;

    vector<int> set_vars_HGRAD, set_vars_HVOL, set_vars_HDIV, set_vars_HCURL, set_vars_HFACE;
    vector<string> set_varlist_HGRAD, set_varlist_HVOL, set_varlist_HDIV, set_varlist_HCURL, set_varlist_HFACE;
    
    for (size_t i=0; i<set_usebasis[set].size(); i++) {
      int bind = set_usebasis[set][i];
      string var = set_varlist[set][i];
      
      int numb = basis_pointers[bind]->getCardinality();
      View_AD2 newsol("seeded uvals",numElem, numb);
      //set_uvals[i] = newsol;
      uvals[uprog] = newsol;
      if (isTransient) {
        View_AD2 newtsol("seeded uvals",numElem, numb);
        u_dotvals[uprog] = newtsol;
      }
      
      set_uindex.push_back(uprog);
      
      uprog++;
      
      if (basis_types[bind].substr(0,5) == "HGRAD") {
        set_vars_HGRAD.push_back(i);
        set_varlist_HGRAD.push_back(var);
        
        soln_fields.push_back(SolutionField(var,set,"solution",i,"HGRAD",bind,"",0,0,numip,false,false));
        soln_fields.push_back(SolutionField("grad("+var+")[x]",set,"solution",i,"HGRAD",bind,"grad",0,0,numip,false,false));
        soln_fields.push_back(SolutionField("grad("+var+")[y]",set,"solution",i,"HGRAD",bind,"grad",1,0,numip,false,false));
        soln_fields.push_back(SolutionField("grad("+var+")[z]",set,"solution",i,"HGRAD",bind,"grad",2,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"_t",set,"solution",i,"HGRAD",bind,"time",0,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+" side",set,"solution",i,"HGRAD",bind,"",0,0,numsideip,true,false));
        soln_fields.push_back(SolutionField("grad("+var+")[x] side",set,"solution",i,"HGRAD",bind,"grad",0,0,numsideip,true,false));
        soln_fields.push_back(SolutionField("grad("+var+")[y] side",set,"solution",i,"HGRAD",bind,"grad",1,0,numsideip,true,false));
        soln_fields.push_back(SolutionField("grad("+var+")[z] side",set,"solution",i,"HGRAD",bind,"grad",2,0,numsideip,true,false));
        soln_fields.push_back(SolutionField(var+" point",set,"solution",i,"HGRAD",bind,"grad",0,0,1,false,true));
        soln_fields.push_back(SolutionField("grad("+var+")[x] point",set,"solution",i,"HGRAD",bind,"grad",0,0,1,false,true));
        soln_fields.push_back(SolutionField("grad("+var+")[y] point",set,"solution",i,"HGRAD",bind,"grad",1,0,1,false,true));
        soln_fields.push_back(SolutionField("grad("+var+")[z] point",set,"solution",i,"HGRAD",bind,"grad",2,0,1,false,true));
      }
      else if (basis_types[bind].substr(0,4) == "HDIV" ) {
        set_vars_HDIV.push_back(i);
        set_varlist_HDIV.push_back(var);
        
        soln_fields.push_back(SolutionField(var+"[x]",set,"solution",i,"HDIV",bind,"",0,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"[y]",set,"solution",i,"HDIV",bind,"",1,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"[z]",set,"solution",i,"HDIV",bind,"",2,0,numip,false,false));
        soln_fields.push_back(SolutionField("div("+var+")",set,"solution",i,"HDIV",bind,"div",0,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"_t[x]",set,"solution",i,"HDIV",bind,"time",0,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"_t[y]",set,"solution",i,"HDIV",bind,"time",1,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"_t[z]",set,"solution",i,"HDIV",bind,"time",2,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"[x] side",set,"solution",i,"HDIV",bind,"",0,0,numsideip,true,false));
        soln_fields.push_back(SolutionField(var+"[y] side",set,"solution",i,"HDIV",bind,"",1,0,numsideip,true,false));
        soln_fields.push_back(SolutionField(var+"[z] side",set,"solution",i,"HDIV",bind,"",2,0,numsideip,true,false));
        soln_fields.push_back(SolutionField(var+"[x] point",set,"solution",i,"HDIV",bind,"",0,0,1,false,true));
        soln_fields.push_back(SolutionField(var+"[y] point",set,"solution",i,"HDIV",bind,"",1,0,1,false,true));
        soln_fields.push_back(SolutionField(var+"[z] point",set,"solution",i,"HDIV",bind,"",2,0,1,false,true));
        
      }
      else if (basis_types[bind].substr(0,4) == "HVOL") {
        set_vars_HVOL.push_back(i);
        set_varlist_HVOL.push_back(var);
        soln_fields.push_back(SolutionField(var,set,"solution",i,"HVOL",bind,"",0,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"_t",set,"solution",i,"HVOL",bind,"time",0,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+" side",set,"solution",i,"HVOL",bind,"",0,0,numsideip,false,false));
        soln_fields.push_back(SolutionField(var+" point",set,"solution",i,"HVOL",bind,"",0,0,1,false,true));
        
      }
      else if (basis_types[bind].substr(0,5) == "HCURL") {
        set_vars_HCURL.push_back(i);
        set_varlist_HCURL.push_back(var);
        
        soln_fields.push_back(SolutionField(var+"[x]",set,"solution",i,"HCURL",bind,"",0,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"[y]",set,"solution",i,"HCURL",bind,"",1,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"[z]",set,"solution",i,"HCURL",bind,"",2,0,numip,false,false));
        soln_fields.push_back(SolutionField("curl("+var+")[x]",set,"solution",i,"HCURL",bind,"curl",0,0,numip,false,false));
        soln_fields.push_back(SolutionField("curl("+var+")[y]",set,"solution",i,"HCURL",bind,"curl",1,0,numip,false,false));
        soln_fields.push_back(SolutionField("curl("+var+")[z]",set,"solution",i,"HCURL",bind,"curl",2,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"_t[x]",set,"solution",i,"HCURL",bind,"time",0,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"_t[y]",set,"solution",i,"HCURL",bind,"time",1,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"_t[z]",set,"solution",i,"HCURL",bind,"time",2,0,numip,false,false));
        soln_fields.push_back(SolutionField(var+"[x] side",set,"solution",i,"HCURL",bind,"",0,0,numsideip,true,false));
        soln_fields.push_back(SolutionField(var+"[y] side",set,"solution",i,"HCURL",bind,"",1,0,numsideip,true,false));
        soln_fields.push_back(SolutionField(var+"[z] side",set,"solution",i,"HCURL",bind,"",2,0,numsideip,true,false));
        soln_fields.push_back(SolutionField(var+"[x] point",set,"solution",i,"HCURL",bind,"",0,0,1,false,true));
        soln_fields.push_back(SolutionField(var+"[y] point",set,"solution",i,"HCURL",bind,"",0,0,1,false,true));
        soln_fields.push_back(SolutionField(var+"[z] point",set,"solution",i,"HCURL",bind,"",0,0,1,false,true));
        
      }
      else if (basis_types[bind].substr(0,5) == "HFACE") {
        set_vars_HFACE.push_back(i);
        set_varlist_HFACE.push_back(var);
        
        soln_fields.push_back(SolutionField(var+" side",set,"solution",i,"HFACE",bind,"",0,0,numsideip,true,false));
        
      }
    }
    uvals_index.push_back(set_uindex);
    vars_HGRAD.push_back(set_vars_HGRAD);
    vars_HVOL.push_back(set_vars_HVOL);
    vars_HDIV.push_back(set_vars_HDIV);
    vars_HCURL.push_back(set_vars_HCURL);
    vars_HFACE.push_back(set_vars_HFACE);
    varlist_HGRAD.push_back(set_varlist_HGRAD);
    varlist_HVOL.push_back(set_varlist_HVOL);
    varlist_HDIV.push_back(set_varlist_HDIV);
    varlist_HCURL.push_back(set_varlist_HCURL);
    varlist_HFACE.push_back(set_varlist_HFACE);
    
  }
  
  for (size_t i=0; i<paramusebasis.size(); i++) {
    size_t set = 0;
    int bind = paramusebasis[i];
    string var = param_varlist[i];
    int numb = basis_pointers[bind]->getCardinality();
    View_AD2 newpsol("seeded uvals",numElem, numb);
    pvals.push_back(newpsol);
    
    if (basis_types[bind].substr(0,5) == "HGRAD") {
      paramvars_HGRAD.push_back(i);
      paramvarlist_HGRAD.push_back(var);
      
      soln_fields.push_back(SolutionField(var,set,"param",i,"HGRAD",bind,"",0,0,numip,false,false));
      soln_fields.push_back(SolutionField("grad("+var+")[x]",set,"param",i,"HGRAD",bind,"grad",0,0,numip,false,false));
      soln_fields.push_back(SolutionField("grad("+var+")[y]",set,"param",i,"HGRAD",bind,"grad",1,0,numip,false,false));
      soln_fields.push_back(SolutionField("grad("+var+")[z]",set,"param",i,"HGRAD",bind,"grad",2,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"_t",set,"param",i,"HGRAD",bind,"time",0,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+" side",set,"param",i,"HGRAD",bind,"",0,0,numsideip,true,false));
      soln_fields.push_back(SolutionField("grad("+var+")[x] side",set,"param",i,"HGRAD",bind,"grad",0,0,numsideip,true,false));
      soln_fields.push_back(SolutionField("grad("+var+")[y] side",set,"param",i,"HGRAD",bind,"grad",1,0,numsideip,true,false));
      soln_fields.push_back(SolutionField("grad("+var+")[z] side",set,"param",i,"HGRAD",bind,"grad",2,0,numsideip,true,false));
      soln_fields.push_back(SolutionField(var+" point",set,"param",i,"HGRAD",bind,"",0,0,1,false,true));
      soln_fields.push_back(SolutionField("grad("+var+")[x] point",set,"param",i,"HGRAD",bind,"grad",0,0,1,false,true));
      soln_fields.push_back(SolutionField("grad("+var+")[y] point",set,"param",i,"HGRAD",bind,"grad",1,0,1,false,true));
      soln_fields.push_back(SolutionField("grad("+var+")[z] point",set,"param",i,"HGRAD",bind,"grad",2,0,1,false,true));
      
    }
    else if (basis_types[bind].substr(0,4) == "HDIV") {
      paramvars_HDIV.push_back(i);
      paramvarlist_HDIV.push_back(var);
      
      soln_fields.push_back(SolutionField(var+"[x]",set,"param",i,"HDIV",bind,"",0,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"[y]",set,"param",i,"HDIV",bind,"",1,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"[z]",set,"param",i,"HDIV",bind,"",2,0,numip,false,false));
      soln_fields.push_back(SolutionField("div("+var+")",set,"param",i,"HDIV",bind,"div",0,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"_t[x]",set,"param",i,"HDIV",bind,"time",0,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"_t[y]",set,"param",i,"HDIV",bind,"time",1,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"_t[z]",set,"param",i,"HDIV",bind,"time",2,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"[x] side",set,"param",i,"HDIV",bind,"",0,0,numsideip,true,false));
      soln_fields.push_back(SolutionField(var+"[y] side",set,"param",i,"HDIV",bind,"",1,0,numsideip,true,false));
      soln_fields.push_back(SolutionField(var+"[z] side",set,"param",i,"HDIV",bind,"",2,0,numsideip,true,false));
      soln_fields.push_back(SolutionField(var+"[x] point",set,"param",i,"HDIV",bind,"",0,0,1,false,true));
      soln_fields.push_back(SolutionField(var+"[y] point",set,"param",i,"HDIV",bind,"",1,0,1,false,true));
      soln_fields.push_back(SolutionField(var+"[z] point",set,"param",i,"HDIV",bind,"",2,0,1,false,true));
      
    }
    else if (basis_types[bind].substr(0,4) == "HVOL") {
      paramvars_HVOL.push_back(i);
      paramvarlist_HVOL.push_back(var);
      
      soln_fields.push_back(SolutionField(var,set,"param",i,"HVOL",bind,"",0,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"_t",set,"param",i,"HVOL",bind,"time",0,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+" side",set,"param",i,"HVOL",bind,"",0,0,numsideip,true,false));
      soln_fields.push_back(SolutionField(var+" point",set,"param",i,"HVOL",bind,"",0,0,1,false,true));
      
    }
    else if (basis_types[bind].substr(0,5) == "HCURL") {
      paramvars_HCURL.push_back(i);
      paramvarlist_HCURL.push_back(var);
      
      soln_fields.push_back(SolutionField(var+"[x]",set,"param",i,"HCURL",bind,"",0,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"[y]",set,"param",i,"HCURL",bind,"",1,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"[z]",set,"param",i,"HCURL",bind,"",2,0,numip,false,false));
      soln_fields.push_back(SolutionField("curl("+var+")[x]",set,"param",i,"HCURL",bind,"curl",0,0,numip,false,false));
      soln_fields.push_back(SolutionField("curl("+var+")[y]",set,"param",i,"HCURL",bind,"curl",1,0,numip,false,false));
      soln_fields.push_back(SolutionField("curl("+var+")[z]",set,"param",i,"HCURL",bind,"curl",2,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"_t[x]",set,"param",i,"HCURL",bind,"time",0,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"_t[y]",set,"param",i,"HCURL",bind,"time",1,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"_t[z]",set,"param",i,"HCURL",bind,"time",2,0,numip,false,false));
      soln_fields.push_back(SolutionField(var+"[x] side",set,"param",i,"HCURL",bind,"",0,0,numsideip,true,false));
      soln_fields.push_back(SolutionField(var+"[y] side",set,"param",i,"HCURL",bind,"",1,0,numsideip,true,false));
      soln_fields.push_back(SolutionField(var+"[z] side",set,"param",i,"HCURL",bind,"",2,0,numsideip,true,false));
      soln_fields.push_back(SolutionField(var+"[x] point",set,"param",i,"HCURL",bind,"",0,0,1,false,true));
      soln_fields.push_back(SolutionField(var+"[y] point",set,"param",i,"HCURL",bind,"",1,0,1,false,true));
      soln_fields.push_back(SolutionField(var+"[z] point",set,"param",i,"HCURL",bind,"",2,0,1,false,true));
      
    }
    else if (basis_types[bind].substr(0,5) == "HFACE") {
      paramvars_HFACE.push_back(i);
      paramvarlist_HFACE.push_back(var);
      
      soln_fields.push_back(SolutionField(var+" side",set,"param",i,"HFACE",bind,"",0,0,numsideip,true,false));
      
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Reset
////////////////////////////////////////////////////////////////////////////////////

void workset::reset() {
  this->resetResidual();
  this->resetSolutionFields();
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution fields
////////////////////////////////////////////////////////////////////////////////////

void workset::resetSolutionFields() {
  for (size_t f=0; f<soln_fields.size(); ++f) {
    soln_fields[f].isUpdated = false;
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Reset residuals
////////////////////////////////////////////////////////////////////////////////////

void workset::resetResidual() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  //Kokkos::deep_copy(res,0.0);
  
  size_t maxRes_ = maxRes;
  ScalarT zero = 0.0;
#ifndef MrHyDE_NO_AD
  parallel_for("wkset reset res",
               TeamPolicy<AssemblyExec>(res.extent(0), Kokkos::AUTO),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type dof=team.team_rank(); dof<maxRes_; dof+=team.team_size() ) {
      res(elem,dof).val() = zero;
      for (size_type d=0; d<maxRes_; ++d) {
        res(elem,dof).fastAccessDx(d) = zero;
      }
    }
  });
#else
  parallel_for("wkset reset res",
               TeamPolicy<AssemblyExec>(res.extent(0), Kokkos::AUTO),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type dof=team.team_rank(); dof<maxRes_; dof+=team.team_size() ) {
      res(elem,dof) = zero;
    }
  });
#endif
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for general transient problems
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnTransientSeeded(const size_t & set,
                                         View_Sc3 u,
                                         View_Sc4 u_prev,
                                         View_Sc4 u_stage,
                                         const int & seedwhat,
                                         const int & index) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
  
  // These need to be set locally to be available to AssemblyDevice
  ScalarT dt = deltat;
  int stage = current_stage;
  auto b_A = butcher_A;
  auto b_b = butcher_b;
  auto BDF = BDF_wts;

  ScalarT one = 1.0;
  ScalarT zero = 0.0;
 
  // Seed the current stage solution
  if (set == current_set) {
    if (seedwhat == 1) {
      for (size_type var=0; var<u.extent(1); var++ ) {
        size_t uindex = uvals_index[set][var];
        auto u_AD = uvals[uindex];
        auto u_dot_AD = u_dotvals[uindex];
        auto off = subview(set_offsets[set],var,ALL());
        auto cu = subview(u,ALL(),var,ALL());
        auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
        auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());

        parallel_for("wkset transient sol seedwhat 1",
                     TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          ScalarT beta_u, beta_t;
          ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
          ScalarT timewt = one/dt/b_b(stage);
          ScalarT alpha_t = BDF(0)*timewt;

          for (size_type dof=team.team_rank(); dof<u_AD.extent(1); dof+=team.team_size() ) {
            
            // Seed the stage solution
#ifndef MrHyDE_NO_AD
            AD stageval = AD(maxDerivs,off(dof),cu(elem,dof));
#else
            AD stageval = cu(elem,dof);
#endif
            // Compute the evaluating solution
            beta_u = (one-alpha_u)*cu_prev(elem,dof,0);
            for (int s=0; s<stage; s++) {
              beta_u += b_A(stage,s)/b_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            u_AD(elem,dof) = alpha_u*stageval+beta_u;
            
            // Compute the time derivative
            beta_t = zero;
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
      for (size_type var=0; var<u.extent(1); var++ ) {
        size_t uindex = uvals_index[set][var];
        auto u_AD = uvals[uindex];
        auto u_dot_AD = u_dotvals[uindex];
        auto off = subview(set_offsets[set],var,ALL());
        auto cu = subview(u,ALL(),var,ALL());
        auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
        auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
        
        parallel_for("wkset transient sol seedwhat 1",
                     TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          AD beta_u, beta_t;
          ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
          ScalarT timewt = one/dt/b_b(stage);
          ScalarT alpha_t = BDF(0)*timewt;
          for (size_type dof=team.team_rank(); dof<u_AD.extent(1); dof+=team.team_size() ) {
            
            // Get the stage solution
            ScalarT stageval = cu(elem,dof);
            
            // Compute the evaluating solution
            AD u_prev_val = cu_prev(elem,dof,0);
            if (index == 0) {
#ifndef MrHyDE_NO_AD
              u_prev_val = AD(maxDerivs,off(dof),cu_prev(elem,dof,0));
#else
              u_prev_val = cu_prev(elem,dof,0);
#endif
            }
            
            beta_u = (one-alpha_u)*u_prev_val;
            for (int s=0; s<stage; s++) {
              beta_u += b_A(stage,s)/b_b(s) * (cu_stage(elem,dof,s) - u_prev_val);
            }
            u_AD(elem,dof) = alpha_u*stageval+beta_u;
            
            // Compute and seed the time derivative
            beta_t = zero;
            for (int s=1; s<BDF.extent_int(0); s++) {
              AD u_prev_val = cu_prev(elem,dof,s-1);
              if (index == (s-1)) {
#ifndef MrHyDE_NO_AD
                u_prev_val = AD(maxDerivs,off(dof),cu_prev(elem,dof,s-1));
#else
                u_prev_val = cu_prev(elem,dof,s-1);
#endif
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
      for (size_type var=0; var<u.extent(1); var++ ) {
        size_t uindex = uvals_index[set][var];
        auto u_AD = uvals[uindex];
        auto u_dot_AD = u_dotvals[uindex];
        auto off = subview(set_offsets[set],var,ALL());
        auto cu = subview(u,ALL(),var,ALL());
        auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
        auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
        parallel_for("wkset transient sol seedwhat 1",
                     TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          AD beta_u, beta_t;
          ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
          ScalarT timewt = one/dt/b_b(stage);
          ScalarT alpha_t = BDF(0)*timewt;
          for (size_type dof=team.team_rank(); dof<u_AD.extent(1); dof+=team.team_size() ) {
            
            // Get the stage solution
            ScalarT stageval = cu(elem,dof);
            
            // Compute the evaluating solution
            ScalarT u_prev_val = cu_prev(elem,dof,0);
            
            beta_u = (one-alpha_u)*u_prev_val;
            for (int s=0; s<stage; s++) {
              AD u_stage_val = cu_stage(elem,dof,s);
              if (index == s) {
#ifndef MrHyDE_NO_AD
                u_stage_val = AD(maxDerivs,off(dof),cu_stage(elem,dof,s));
#else
                u_stage_val = cu_stage(elem,dof,s);
#endif
              }
              beta_u += b_A(stage,s)/b_b(s) * (u_stage_val - u_prev_val);
            }
            u_AD(elem,dof) = alpha_u*stageval+beta_u;
            
            // Compute and seed the time derivative
            beta_t = zero;
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
        size_t uindex = uvals_index[set][var];
        auto u_AD = uvals[uindex];
        auto u_dot_AD = u_dotvals[uindex];
        auto off = subview(set_offsets[set],var,ALL());
        auto cu = subview(u,ALL(),var,ALL());
        auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
        auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
        
        parallel_for("wkset transient sol seedwhat 1",
                     TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          ScalarT beta_u, beta_t;
          ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
          ScalarT timewt = one/dt/b_b(stage);
          ScalarT alpha_t = BDF(0)*timewt;
          for (size_type dof=team.team_rank(); dof<u_AD.extent(1); dof+=team.team_size() ) {
            // Get the stage solution
            ScalarT stageval = cu(elem,dof);
            // Compute the evaluating solution
            beta_u = (one-alpha_u)*cu_prev(elem,dof,0);
            for (int s=0; s<stage; s++) {
              beta_u += b_A(stage,s)/b_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
            }
            u_AD(elem,dof) = alpha_u*stageval+beta_u;
            
            // Compute the time derivative
            beta_t = zero;
            for (size_type s=1; s<BDF.extent(0); s++) {
              beta_t += BDF(s)*cu_prev(elem,dof,s-1);
            }
            beta_t *= timewt;
            u_dot_AD(elem,dof) = alpha_t*stageval + beta_t;
          }
        });
      }
    }
  }
  else {
    for (size_type var=0; var<u.extent(1); var++ ) {
      size_t uindex = uvals_index[set][var];
      auto u_AD = uvals[uindex];
      auto cu = subview(u,ALL(),var,ALL());
      
      parallel_for("wkset steady soln",
                   RangePolicy<AssemblyExec>(0,u.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<u_AD.extent(1); dof++ ) {
          u_AD(elem,dof) = cu(elem,dof);
        }
      });
    }
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for steady-state problems
////////////////////////////////////////////////////////////////////////////////////

void workset::computeSolnSteadySeeded(const size_t & set,
                                      View_Sc3 u,
                                      const int & seedwhat) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
  for (size_type var=0; var<u.extent(1); var++ ) {
    
    size_t uindex = uvals_index[set][var];
    auto u_AD = uvals[uindex];
    auto off = subview(set_offsets[set],var,ALL());
    auto cu = subview(u,ALL(),var,ALL());
    if (seedwhat == 1 && set == current_set) {
      parallel_for("wkset steady soln",
                   RangePolicy<AssemblyExec>(0,u.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<u_AD.extent(1); dof++ ) {
#ifndef MrHyDE_NO_AD
          u_AD(elem,dof) = AD(maxDerivs,off(dof),cu(elem,dof));
#else
          u_AD(elem,dof) = cu(elem,dof);
#endif
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
#ifndef MrHyDE_NO_AD
            p_AD(elem,dof) = AD(maxDerivs,off(dof),cp(elem,dof));
#else
            p_AD(elem,dof) = cp(elem,dof);
#endif
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
// Compute the seeded solutions at specified ip
////////////////////////////////////////////////////////////////////////////////////

void workset::evaluateSolutionField(const int & fieldnum) {
  
  auto fielddata = soln_fields[fieldnum].data;
  
  bool proceed = true;
  if (soln_fields[fieldnum].derivative_type == "time" ) {
    if (!isTransient) {
      proceed = false;
    }
    else if (soln_fields[fieldnum].isOnSide) {
      proceed = false;
    }
    else if (soln_fields[fieldnum].variable_type == "param") {
      proceed = false;
    }
  }
  if (soln_fields[fieldnum].variable_type == "aux") {
    proceed = false;
  }
  if (soln_fields[fieldnum].isPoint) {
    proceed = false;
  }
  
  if (proceed) {
    //-----------------------------------------------------
    // Get the appropriate view of seeded solution values
    //-----------------------------------------------------
    
    View_AD2 solvals;
    size_t uindex = uvals_index[soln_fields[fieldnum].set_index][soln_fields[fieldnum].variable_index];
    if (soln_fields[fieldnum].variable_type == "solution") { // solution
      if (soln_fields[fieldnum].derivative_type == "time" ) {
        solvals = u_dotvals[uindex];
      }
      else {
        solvals = uvals[uindex];
      }
    }
    if (soln_fields[fieldnum].variable_type == "param") { // discr. params
      solvals = pvals[soln_fields[fieldnum].variable_index];
    }
    
    //-----------------------------------------------------
    // Get the appropriate basis values and evaluate the fields
    //-----------------------------------------------------
    
    int component = soln_fields[fieldnum].component;
    int basis_index = soln_fields[fieldnum].basis_index;
    
    if (soln_fields[fieldnum].derivative_type == "div") {
      auto sbasis = basis_div[basis_index];
      size_t teamSize = std::min(maxTeamSize,sbasis.extent(2));
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(sbasis.extent(0), teamSize, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<sbasis.extent(2); pt+=team.team_size() ) {
          fielddata(elem,pt) = solvals(elem,0)*sbasis(elem,0,pt);
        }
        for (size_type dof=1; dof<sbasis.extent(1); dof++ ) {
          for (size_type pt=team.team_rank(); pt<sbasis.extent(2); pt+=team.team_size() ) {
            fielddata(elem,pt) += solvals(elem,dof)*sbasis(elem,dof,pt);
          }
        }
      });
      
    }
    else {
      View_Sc4 cbasis;
      if (soln_fields[fieldnum].derivative_type == "grad") {
        if (soln_fields[fieldnum].isOnSide) {
          cbasis = basis_grad_side[basis_index];
        }
        else {
          cbasis = basis_grad[basis_index];
        }
      }
      else if (soln_fields[fieldnum].derivative_type == "curl") {
        if (soln_fields[fieldnum].isOnSide) {
          // not implemented
        }
        else {
          cbasis = basis_curl[basis_index];
        }
      }
      else {
        if (soln_fields[fieldnum].isOnSide) {
          cbasis = basis_side[basis_index];
        }
        else {
          cbasis = basis[basis_index];
        }
      }
      
      auto sbasis = subview(cbasis, ALL(), ALL(), ALL(), component);
      size_t teamSize = std::min(maxTeamSize,sbasis.extent(2));
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(sbasis.extent(0), teamSize, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<sbasis.extent(2); pt+=team.team_size() ) {
          fielddata(elem,pt) = solvals(elem,0)*sbasis(elem,0,pt);
        }
        for (size_type dof=1; dof<sbasis.extent(1); dof++ ) {
          for (size_type pt=team.team_rank(); pt<sbasis.extent(2); pt+=team.team_size() ) {
            fielddata(elem,pt) += solvals(elem,dof)*sbasis(elem,dof,pt);
          }
        }
      });
    }
    
    soln_fields[fieldnum].isUpdated = true;
  }
  
}


////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

// TMW: this function should be deprecated
// Gets used only in the boundaryCell flux calculation
// Will not work properly for multi-stage or multi-step

void workset::computeSolnSideIP(const int & side) { 
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    
    /////////////////////////////////////////////////////////////////////
    // HGRAD
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
      string var = varlist_HGRAD[current_set][i];
      int varind = vars_HGRAD[current_set][i];
      
      auto cuvals = uvals[uvals_index[current_set][varind]];
      
      auto csol = this->findData(var+" side");
      auto csol_x = this->findData("grad("+var+")[x] side");
      auto csol_y = this->findData("grad("+var+")[y] side");
      auto csol_z = this->findData("grad("+var+")[z] side");
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
    
    for (size_t i=0; i<vars_HVOL[current_set].size(); i++) {
      string var = varlist_HVOL[current_set][i];
      int varind = vars_HVOL[current_set][i];
      
      auto cuvals = uvals[uvals_index[current_set][varind]];
      
      auto csol = this->findData(var+" side");
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
    
    for (size_t i=0; i<vars_HDIV[current_set].size(); i++) {
      string var = varlist_HDIV[current_set][i];
      int varind = vars_HDIV[current_set][i];
      
      auto cuvals = uvals[uvals_index[current_set][varind]];
      
      auto csolx = this->findData(var+"[x] side");
      auto csoly = this->findData(var+"[y] side");
      auto csolz = this->findData(var+"[z] side");
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
    
    for (size_t i=0; i<vars_HDIV[current_set].size(); i++) {
      string var = varlist_HDIV[current_set][i];
      int varind = vars_HDIV[current_set][i];
      
      auto cuvals = uvals[uvals_index[current_set][varind]];
      
      auto csolx = this->findData(var+"[x] side");
      auto csoly = this->findData(var+"[y] side");
      auto csolz = this->findData(var+"[z] side");
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
  flux = View_AD3("flux",numElem,numAux,numsideip);
  
  if (numAux > 0) {
    size_t maxAux = aux_offsets.extent(0)*aux_offsets.extent(1);
    if (maxAux > maxRes) {
      maxRes = maxAux;
      res = View_AD2("residual",numElem, maxRes);
    }
  }

  for (size_t i=0; i<aux_varlist.size(); ++i) {
    string var = aux_varlist[i];
    
    soln_fields.push_back(SolutionField("aux "+var,0,"aux",i,"HGRAD",0,"",0,0,numip,false,false));
    soln_fields.push_back(SolutionField("aux "+var+" side",0,"aux",i,"HGRAD",0,"",0,0,numsideip,true,false));
    
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
// Get a subview associated with a vector of parameters
//////////////////////////////////////////////////////////////

Kokkos::View<AD*,Kokkos::LayoutStride,AssemblyDevice> workset::getParameter(const string & name, bool & found) {
  found = false;
  size_t iter=0;
  Kokkos::View<AD*,Kokkos::LayoutStride,AssemblyDevice> pvals;
  while (!found && iter<paramnames.size()) {
    if (paramnames[iter] == name) {
      found  = true;
      pvals = subview(params_AD,iter,ALL());
    }
    else {
      iter++;
    }
  }
  return pvals;
}

//////////////////////////////////////////////////////////////
// Set the time
//////////////////////////////////////////////////////////////

void workset::setTime(const ScalarT & newtime) {
  time = newtime;
}

//////////////////////////////////////////////////////////////
// Set deltat
//////////////////////////////////////////////////////////////

void workset::setDeltat(const ScalarT & newdt) {
  deltat = newdt;
}

//////////////////////////////////////////////////////////////
// Set the stage index
//////////////////////////////////////////////////////////////

void workset::setStage(const int & newstage) {
  current_stage = newstage;
}

//////////////////////////////////////////////////////////////
// Add a data view
//////////////////////////////////////////////////////////////

//void workset::addData(const string & label, const int & dim0, const int & dim1) {
  //data.push_back(View_AD2(label,0,dim1));
  //data_labels.push_back(label);
  //data_usage.push_back(0);
//}

//void workset::addDataSc(const string & label, const int & dim0, const int & dim1) {
  //data_Sc.push_back(View_Sc2(label,1,dim1));
  //data_Sc_labels.push_back(label);
  //data_Sc_usage.push_back(0);
//}

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

View_AD2 workset::findData(const string & label) {
  
  Teuchos::TimeMonitor basistimer(*worksetgetDataTimer);
  
  View_AD2 outdata;
  
  bool found = false;
  size_t ind = 0;
  while (!found && ind<soln_fields.size()) {
    if (label == soln_fields[ind].expression) {
      found = true;
    }
    else {
      ++ind;
    }
  }
  if (!found) {
    std::cout << "Error: could not find a field named " << label << std::endl;
  }
  else {
    this->checkDataAllocation(ind);
  }
  outdata = soln_fields[ind].data;
  
  return outdata;
  
}

//----------------------------------------------------------------

void workset::printSolutionFields() {
  cout << "Currently defined fields are: " << endl;
  for (size_t f=0; f<soln_fields.size(); ++f) {
    cout << soln_fields[f].expression << endl;
  }
}

void workset::printScalarFields() {
  cout << "Currently defined scalar fields are: " << endl;
  for (size_t f=0; f<scalar_fields.size(); ++f) {
    cout << scalar_fields[f].expression << endl;
  }
}
//----------------------------------------------------------------

View_AD2 workset::getData(const string & label) {
  
  Teuchos::TimeMonitor basistimer(*worksetgetDataTimer);
  
  View_AD2 outdata;
  
  bool found = false;
  size_t ind = 0;
  while (!found && ind<soln_fields.size()) {
    if (label == soln_fields[ind].expression) {
      found = true;
    }
    else {
      ++ind;
    }
  }
  if (!found) {
    std::cout << "Error: could not find a field named " << label << std::endl;
    this->printSolutionFields();
  }
  else {
    this->checkDataAllocation(ind);
    if (!soln_fields[ind].isUpdated) {
      this->evaluateSolutionField(ind);
    }
  }
  outdata = soln_fields[ind].data;
  
  return outdata;
  
}

void workset::checkDataAllocation(const size_t & ind) {
  
  if (!soln_fields[ind].isInitialized) {
    soln_fields[ind].initialize(maxElem);
  }
  
}

void workset::checkDataScAllocation(const size_t & ind) {
  
  if (!scalar_fields[ind].isInitialized) {
    scalar_fields[ind].initialize(maxElem);
  }
  
}

//////////////////////////////////////////////////////////////
// Extract a View_Sc2 (2-dimensional array with ScalarT entries)
//////////////////////////////////////////////////////////////

View_Sc2 workset::getDataSc(const string & label) {
  
  Teuchos::TimeMonitor basistimer(*worksetgetDataScTimer);
  View_Sc2 outdata;
  
  bool found = false;
  size_t ind = 0;
  while (!found && ind<scalar_fields.size()) {
    if (label == scalar_fields[ind].expression) {
      found = true;
    }
    else {
      ++ind;
    }
  }
  if (!found) {
    std::cout << "Error: could not find a scalar field named " << label << std::endl;
    this->printScalarFields();
  }
  else {
    this->checkDataScAllocation(ind);
    outdata = scalar_fields[ind].data;
  }
  
  return outdata;
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
      std::cout << "Warning: could not find basis for: " << var << std::endl;
      std::cout << "An error will probably occur if this view is accessed" << std::endl;
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
    else {
      varindex++;
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
  std::cout << "THIS SHOULD NOT BE CALLED" << std::endl;
  
  /*
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
  */
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
// Set the data is a scalar field
//////////////////////////////////////////////////////////////

void workset::setScalarField(View_Sc2 newdata, const string & expression) {
  
  bool found = false;
  size_t ind = 0;
  while (!found && ind<scalar_fields.size()) {
    if (expression == scalar_fields[ind].expression) {
      found = true;
    }
    else {
      ++ind;
    }
  }
  if (!found) {
    std::cout << "Error: could not find a scalar field named " << expression << std::endl;
    this->printScalarFields();
  }
  else {
    scalar_fields[ind].data = newdata;
  }
}

//////////////////////////////////////////////////////////////
// Set the solutions
//////////////////////////////////////////////////////////////

void workset::setSolution(View_AD4 newsol, const string & pfix) {
  // newsol has dims numElem x numvars x numip x dimension
  // however, this numElem may be smaller than the size of the data arrays
  
  for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
    string var = varlist_HGRAD[current_set][i];
    int varind = vars_HGRAD[current_set][i];
    auto csol = this->findData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<varlist_HVOL[current_set].size(); i++) {
    string var = varlist_HVOL[current_set][i];
    int varind = vars_HVOL[current_set][i];
    auto csol = this->findData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<varlist_HFACE[current_set].size(); i++) {
    string var = varlist_HFACE[current_set][i];
    int varind = vars_HFACE[current_set][i];
    auto csol = this->findData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<varlist_HDIV[current_set].size(); i++) {
    string var = varlist_HDIV[current_set][i];
    int varind = vars_HDIV[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->findData(var+"[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->findData(var+"[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->findData(var+"[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  for (size_t i=0; i<varlist_HCURL[current_set].size(); i++) {
    string var = varlist_HCURL[current_set][i];
    int varind = vars_HCURL[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->findData(var+"[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->findData(var+"[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->findData(var+"[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  
}

//////////////////////////////////////////////////////////////
// Set the solution GRADs
//////////////////////////////////////////////////////////////

void workset::setSolutionGrad(View_AD4 newsol, const string & pfix) {
  for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
    string var = varlist_HGRAD[current_set][i];
    int varind = vars_HGRAD[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->findData("grad("+var+")[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->findData("grad("+var+")[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->findData("grad("+var+")[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
}

//////////////////////////////////////////////////////////////
// Set the solution DIVs
//////////////////////////////////////////////////////////////

void workset::setSolutionDiv(View_AD3 newsol, const string & pfix) {
  for (size_t i=0; i<varlist_HDIV[current_set].size(); i++) {
    string var = varlist_HDIV[current_set][i];
    int varind = vars_HDIV[current_set][i];
    auto csol = this->findData("div("+var+")");
    auto cnsol = subview(newsol,ALL(),varind,ALL());
    this->copyData(csol,cnsol);
  }
}

//////////////////////////////////////////////////////////////
// Set the solution CURLs
//////////////////////////////////////////////////////////////

void workset::setSolutionCurl(View_AD4 newsol, const string & pfix) {
  for (size_t i=0; i<varlist_HCURL[current_set].size(); i++) {
    string var = varlist_HCURL[current_set][i];
    int varind = vars_HCURL[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->findData("curl("+var+")[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->findData("curl("+var+")[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->findData("curl("+var+")[z]"+pfix);
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
  for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
    string var = varlist_HGRAD[current_set][i];
    int varind = vars_HGRAD[current_set][i];
    auto csol = this->findData(var+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
  }
  for (size_t i=0; i<varlist_HVOL[current_set].size(); i++) {
    string var = varlist_HVOL[current_set][i];
    int varind = vars_HVOL[current_set][i];
    auto csol = this->findData(var+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
  }
  for (size_t i=0; i<varlist_HDIV[current_set].size(); i++) {
    string var = varlist_HDIV[current_set][i];
    int varind = vars_HDIV[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->findData(var+"[x] point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->findData(var+"[y] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->findData(var+"[z] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(2);
      });
    }
  }
  for (size_t i=0; i<varlist_HCURL[current_set].size(); i++) {
    string var = varlist_HCURL[current_set][i];
    int varind = vars_HCURL[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->findData(var+"[x] point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->findData(var+"[y] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->findData(var+"[z] point");
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
  for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
    string var = varlist_HGRAD[current_set][i];
    int varind = vars_HGRAD[current_set][i];
    size_type dim = newsol.extent(1);
    auto csol = this->findData("grad("+var+")[x]"+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->findData("grad("+var+")[y]"+" point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->findData("grad("+var+")[z]"+" point");
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
    auto csol = this->findData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<paramvarlist_HVOL.size(); i++) {
    string var = paramvarlist_HVOL[i];
    int varind = paramvars_HVOL[i];
    auto csol = this->findData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<paramvarlist_HFACE.size(); i++) {
    string var = paramvarlist_HFACE[i];
    int varind = paramvars_HFACE[i];
    auto csol = this->findData(var+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<paramvarlist_HDIV.size(); i++) {
    string var = paramvarlist_HDIV[i];
    int varind = paramvars_HDIV[i];
    size_type dim = newsol.extent(3);
    auto csol = this->findData(var+"[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->findData(var+"[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->findData(var+"[z]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  for (size_t i=0; i<varlist_HCURL.size(); i++) {
    string var = paramvarlist_HCURL[i];
    int varind = paramvars_HCURL[i];
    size_type dim = newsol.extent(3);
    auto csol = this->findData(var+"[x]"+pfix);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->findData(var+"[y]"+pfix);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->findData(var+"[z]"+pfix);
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
    auto csol = this->findData(var+" point");
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
    auto csol = this->findData(var+" point");
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
    auto csol = this->findData(var+"[x] point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->findData(var+"[y] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->findData(var+"[z] point");
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
    auto csol = this->findData(var+"[x] point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->findData(var+"[y] point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->findData(var+"[z] point");
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
    auto csol = this->findData("grad("+var+")[x]"+" point");
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->findData("grad("+var+")[y]"+" point");
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->findData("grad("+var+")[z]"+" point");
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

void workset::updatePhysicsSet(const size_t & current_set_) {
  if (isInitialized) {
    if (numSets>1) {
      current_set = current_set_;
      offsets = set_offsets[current_set];
      usebasis = set_usebasis[current_set];
      varlist = set_varlist[current_set];
      var_bcs = set_var_bcs[current_set];
    }
  }
}
