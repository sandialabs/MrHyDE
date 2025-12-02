/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "workset.hpp"
#include <iostream>
using namespace MrHyDE;

////////////////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
Workset<EvalT>::Workset(const vector<int> & cellinfo,
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
  numDiscParams = cellinfo[1];
  numAux = 0;
  numElem = cellinfo[2];
  usebcs = true;
  numip = cellinfo[3];
  numsideip = cellinfo[4];
  numSets = cellinfo[5];
  numScalarParams = cellinfo[6];
  
  isOnSide = false;
  isOnPoint = false;

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
  scalar_fields.push_back(ScalarField("x"));
  scalar_fields.push_back(ScalarField("y"));
  scalar_fields.push_back(ScalarField("z"));
  
  side_scalar_fields.push_back(ScalarField("x"));
  side_scalar_fields.push_back(ScalarField("y"));
  side_scalar_fields.push_back(ScalarField("z"));
  
  side_scalar_fields.push_back(ScalarField("n[x]"));
  side_scalar_fields.push_back(ScalarField("n[y]"));
  side_scalar_fields.push_back(ScalarField("n[z]"));
  
  side_scalar_fields.push_back(ScalarField("t[x]"));
  side_scalar_fields.push_back(ScalarField("t[y]"));
  side_scalar_fields.push_back(ScalarField("t[z]"));
  
  point_scalar_fields.push_back(ScalarField("x"));
  point_scalar_fields.push_back(ScalarField("y"));
  point_scalar_fields.push_back(ScalarField("z"));
    
  have_rotation = false;
  have_rotation_phi = false;
  rotation = View_Sc3("rotation matrix",1,3,3);
  
  int maxb = 0;
  for (size_t i=0; i<basis_pointers.size(); i++) {
    int numb = basis_pointers[i]->getCardinality();
    maxb = std::max(maxb,numb);
  }
  
  basis = vector<CompressedView<View_Sc4>>(basis_pointers.size());
  basis_grad = vector<CompressedView<View_Sc4>>(basis_pointers.size());
  basis_curl = vector<CompressedView<View_Sc4>>(basis_pointers.size());
  basis_div = vector<CompressedView<View_Sc3>>(basis_pointers.size());
  
  basis_side = vector<CompressedView<View_Sc4>>(basis_pointers.size());
  basis_grad_side = vector<CompressedView<View_Sc4>>(basis_pointers.size());
  basis_curl_side = vector<CompressedView<View_Sc4>>(basis_pointers.size());
  
  set_BDF_wts = vector<Kokkos::View<ScalarT*,AssemblyDevice> >(numSets);
  set_butcher_A = vector<Kokkos::View<ScalarT**,AssemblyDevice> >(numSets);
  set_butcher_b = vector<Kokkos::View<ScalarT*,AssemblyDevice> >(numSets);
  set_butcher_c = vector<Kokkos::View<ScalarT*,AssemblyDevice> >(numSets);
    
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  maxTeamSize = 256 / VECTORSIZE;
#else
  maxTeamSize = 1;
#endif
  
  bool is_same_eval = std::is_same<EvalT, ScalarT>::value;
  if (is_same_eval) {
    only_scalar = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Public functions
////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::createSolutionFields() {

  // Need to first allocate the residual view
  // This is the largest view in the code (due to the AD) so we are careful with the size
  
  // Start with the number of active scalar parameters (typically small)
  maxRes = numScalarParams;
  
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
  
  sol_vals = vector<View_EvalT2>(totalvars);
  if (isTransient) {
    sol_dot_vals = vector<View_EvalT2>(totalvars);
  }
  
  res = View_EvalT2("residual",numElem, maxRes);
  
  size_t uprog = 0;
  string soltype = "solution";
  
  for (size_t set=0; set<numSets; ++set) {
    
    vector<size_t> set_uindex;

    vector<int> set_vars_HGRAD, set_vars_HVOL, set_vars_HDIV, set_vars_HCURL, set_vars_HFACE;
    vector<string> set_varlist_HGRAD, set_varlist_HVOL, set_varlist_HDIV, set_varlist_HCURL, set_varlist_HFACE;
    
    for (size_t i=0; i<set_usebasis[set].size(); i++) {
      int bind = set_usebasis[set][i];
      string var = set_varlist[set][i];
      
      int numb = basis_pointers[bind]->getCardinality();
      View_EvalT2 newsol("seeded sol_vals",numElem, numb);
      sol_vals[uprog] = newsol;
      if (isTransient) {
        View_EvalT2 newtsol("seeded sol_vals",numElem, numb);
        sol_dot_vals[uprog] = newtsol;
      }
      
      set_uindex.push_back(uprog);
      
      uprog++;
      
      this->addSolutionField(var, set, i, basis_types[bind], soltype);
      
      if (basis_types[bind].substr(0,5) == "HGRAD") {
        set_vars_HGRAD.push_back(i);
        set_varlist_HGRAD.push_back(var);
      }
      else if (basis_types[bind].substr(0,4) == "HDIV" ) {
        set_vars_HDIV.push_back(i);
        set_varlist_HDIV.push_back(var);
      }
      else if (basis_types[bind].substr(0,4) == "HVOL") {
        set_vars_HVOL.push_back(i);
        set_varlist_HVOL.push_back(var);
      }
      else if (basis_types[bind].substr(0,5) == "HCURL") {
        set_vars_HCURL.push_back(i);
        set_varlist_HCURL.push_back(var);
      }
      else if (basis_types[bind].substr(0,5) == "HFACE") {
        set_vars_HFACE.push_back(i);
        set_varlist_HFACE.push_back(var);
      }
    }
    sol_vals_index.push_back(set_uindex);
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
  
  soltype = "param";
  
  for (size_t i=0; i<paramusebasis.size(); i++) {
    size_t set = 0;
    int bind = paramusebasis[i];
    string var = param_varlist[i];
    int numb = basis_pointers[bind]->getCardinality();
    View_EvalT2 newpsol("seeded sol_vals",numElem, numb);
    pvals.push_back(newpsol);
    
    this->addSolutionField(var, set, i, basis_types[bind], soltype);
    
    if (basis_types[bind].substr(0,5) == "HGRAD") {
      paramvars_HGRAD.push_back(i);
      paramvarlist_HGRAD.push_back(var);
    }
    else if (basis_types[bind].substr(0,4) == "HDIV") {
      paramvars_HDIV.push_back(i);
      paramvarlist_HDIV.push_back(var);
    }
    else if (basis_types[bind].substr(0,4) == "HVOL") {
      paramvars_HVOL.push_back(i);
      paramvarlist_HVOL.push_back(var);
    }
    else if (basis_types[bind].substr(0,5) == "HCURL") {
      paramvars_HCURL.push_back(i);
      paramvarlist_HCURL.push_back(var);
    }
    else if (basis_types[bind].substr(0,5) == "HFACE") {
      paramvars_HFACE.push_back(i);
      paramvarlist_HFACE.push_back(var);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Add solution fields
////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::addSolutionFields(vector<string> & vars, vector<string> & types, vector<int> & basis_indices) {
  
  vector<int> set_vars_HGRAD, set_vars_HVOL, set_vars_HDIV, set_vars_HCURL, set_vars_HFACE;
  vector<string> set_varlist_HGRAD, set_varlist_HVOL, set_varlist_HDIV, set_varlist_HCURL, set_varlist_HFACE;
  
  for (size_t i=0; i<vars.size(); ++i) {
    size_t set = 0;
    if (set_usebasis.size() == 0) {
      vector<int> cusebasis;
      set_usebasis.push_back(cusebasis);
    }
    size_t varind = set_usebasis[set].size();
    string soltype = "solution";
    int bind = basis_indices[i];
    string type = types[bind];
    string var = vars[i];
    
    this->addSolutionField(var, set, varind, type, soltype);
    set_usebasis[set].push_back(basis_indices[i]);
    
    if (type.substr(0,5) == "HGRAD") {
      set_vars_HGRAD.push_back(i);
      set_varlist_HGRAD.push_back(var);
    }
    else if (type.substr(0,4) == "HDIV" ) {
      set_vars_HDIV.push_back(i);
      set_varlist_HDIV.push_back(var);
    }
    else if (type.substr(0,4) == "HVOL") {
      set_vars_HVOL.push_back(i);
      set_varlist_HVOL.push_back(var);
    }
    else if (type.substr(0,5) == "HCURL") {
      set_vars_HCURL.push_back(i);
      set_varlist_HCURL.push_back(var);
    }
    else if (type.substr(0,5) == "HFACE") {
      set_vars_HFACE.push_back(i);
      set_varlist_HFACE.push_back(var);
    }
  }
  
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


////////////////////////////////////////////////////////////////////////////////////
// Add solution fields
////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::addSolutionField(string & var, size_t & set_index,
                               size_t & var_index, string & basistype, string & soltype) {
  
  if (basistype.substr(0,5) == "HGRAD") {
    
    soln_fields.push_back(SolutionField<EvalT>(var, set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>("grad("+var+")[x]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>("grad("+var+")[y]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>("grad("+var+")[z]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"_t", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>(var, set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>("grad("+var+")[x]", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>("grad("+var+")[y]", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>("grad("+var+")[z]", set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>(var, set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>("grad("+var+")[x]", set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>("grad("+var+")[y]", set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>("grad("+var+")[z]", set_index, soltype, var_index));
  
  }
  else if (basistype.substr(0,4) == "HDIV" ) {
    
    soln_fields.push_back(SolutionField<EvalT>(var+"[x]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"[y]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"[z]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>("div("+var+")", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"_t[x]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"_t[y]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"_t[z]", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>(var+"[x]", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>(var+"[y]", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>(var+"[z]", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>("div("+var+")", set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>(var+"[x]", set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>(var+"[y]", set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>(var+"[z]", set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>("div("+var+")", set_index, soltype, var_index));
    
  }
  else if (basistype.substr(0,4) == "HVOL") {
    
    soln_fields.push_back(SolutionField<EvalT>(var, set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"_t", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>(var, set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>(var, set_index, soltype, var_index));
    
  }
  else if (basistype.substr(0,5) == "HCURL") {
    
    soln_fields.push_back(SolutionField<EvalT>(var+"[x]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"[y]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"[z]", set_index, soltype, var_index));
    if (dimension == 2) {
      soln_fields.push_back(SolutionField<EvalT>("curl("+var+")", set_index, soltype, var_index));
    }
    else {
      soln_fields.push_back(SolutionField<EvalT>("curl("+var+")[x]", set_index, soltype, var_index));
      soln_fields.push_back(SolutionField<EvalT>("curl("+var+")[y]", set_index, soltype, var_index));
      soln_fields.push_back(SolutionField<EvalT>("curl("+var+")[z]", set_index, soltype, var_index));
    }
    soln_fields.push_back(SolutionField<EvalT>(var+"_t[x]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"_t[y]", set_index, soltype, var_index));
    soln_fields.push_back(SolutionField<EvalT>(var+"_t[z]", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>(var+"[x]", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>(var+"[y]", set_index, soltype, var_index));
    side_soln_fields.push_back(SolutionField<EvalT>(var+"[z]", set_index, soltype, var_index));
    if (dimension == 2) {
      side_soln_fields.push_back(SolutionField<EvalT>("curl("+var+")", set_index, soltype, var_index));
    }
    else {
      side_soln_fields.push_back(SolutionField<EvalT>("curl("+var+")[x]", set_index, soltype, var_index));
      side_soln_fields.push_back(SolutionField<EvalT>("curl("+var+")[y]", set_index, soltype, var_index));
      side_soln_fields.push_back(SolutionField<EvalT>("curl("+var+")[z]", set_index, soltype, var_index));
    }
    point_soln_fields.push_back(SolutionField<EvalT>(var+"[x]", set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>(var+"[y]", set_index, soltype, var_index));
    point_soln_fields.push_back(SolutionField<EvalT>(var+"[z]", set_index, soltype, var_index));
    if (dimension == 2) {
      point_soln_fields.push_back(SolutionField<EvalT>("curl("+var+")", set_index, soltype, var_index));
    }
    else {
      point_soln_fields.push_back(SolutionField<EvalT>("curl("+var+")[x]", set_index, soltype, var_index));
      point_soln_fields.push_back(SolutionField<EvalT>("curl("+var+")[y]", set_index, soltype, var_index));
      point_soln_fields.push_back(SolutionField<EvalT>("curl("+var+")[z]", set_index, soltype, var_index));
    }
    soln_fields.push_back(SolutionField<EvalT>("div("+var+")", set_index, soltype, var_index));
  }
  else if (basistype.substr(0,5) == "HFACE") {
    
    side_soln_fields.push_back(SolutionField<EvalT>(var, set_index, soltype, var_index));
    
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Add scalar fields
////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::addScalarFields(vector<string> & fields) {
  for (size_t i=0; i<fields.size(); ++i) {
    scalar_fields.push_back(ScalarField(fields[i]));
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Reset
////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::reset() {
  this->resetResidual();
  this->resetSolutionFields();
}

////////////////////////////////////////////////////////////////////////////////////
// Reset solution fields
////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::resetSolutionFields() {
  for (size_t f=0; f<soln_fields.size(); ++f) {
    soln_fields[f].is_updated_ = false;
  }
  for (size_t f=0; f<side_soln_fields.size(); ++f) {
    side_soln_fields[f].is_updated_ = false;
  }
  for (size_t f=0; f<point_soln_fields.size(); ++f) {
    point_soln_fields[f].is_updated_ = false;
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Reset residuals
////////////////////////////////////////////////////////////////////////////////////

template<>
void Workset<ScalarT>::resetResidual() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  
  if (isInitialized) {
    size_t maxRes_ = maxRes;
    parallel_for("wkset reset res",
                 TeamPolicy<AssemblyExec>(res.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type dof=team.team_rank(); dof<maxRes_; dof+=team.team_size() ) {
        res(elem,dof) = 0.0;
      }
    });
  }
}

template<class EvalT>
void Workset<EvalT>::resetResidual() {
  Teuchos::TimeMonitor resettimer(*worksetResetTimer);
  
  if (isInitialized) {
    size_t maxRes_ = maxRes;
    ScalarT zero = 0.0;
    parallel_for("wkset reset res",
                 TeamPolicy<AssemblyExec>(res.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type dof=team.team_rank(); dof<maxRes_; dof+=team.team_size() ) {
        res(elem,dof) = zero;
      }
    });
  }
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for general transient problems
////////////////////////////////////////////////////////////////////////////////////


template<>
void Workset<ScalarT>::computeSolnTransientSeeded(const size_t & set,
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
    for (size_type var=0; var<u.extent(1); var++ ) {
      size_t uindex = sol_vals_index[set][var];
      auto u_AD = sol_vals[uindex];
      auto u_dot_AD = sol_dot_vals[uindex];
      auto off = subview(set_offsets[set],var,ALL());
      auto cu = subview(u,ALL(),var,ALL());
      auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
      auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
        
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VECTORSIZE),
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
  else {
    for (size_type var=0; var<u.extent(1); var++ ) {
      size_t uindex = sol_vals_index[set][var];
      auto u_AD = sol_vals[uindex];
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



template<class EvalT>
void Workset<EvalT>::computeSolnTransientSeeded(const size_t & set,
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
        size_t uindex = sol_vals_index[set][var];
        auto u_AD = sol_vals[uindex];
        auto u_dot_AD = sol_dot_vals[uindex];
        auto off = subview(set_offsets[set],var,ALL());
        auto cu = subview(u,ALL(),var,ALL());
        auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
        auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
        parallel_for("wkset transient sol seedwhat 1",
                     TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VECTORSIZE),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          ScalarT beta_u, beta_t;
          ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
          ScalarT timewt = one/dt/b_b(stage);
          ScalarT alpha_t = BDF(0)*timewt;
          EvalT dummyval = 0.0;
          for (size_type dof=team.team_rank(); dof<u_AD.extent(1); dof+=team.team_size() ) {
            
            // Seed the stage solution
#ifndef MrHyDE_NO_AD
            EvalT stageval = EvalT(dummyval.size(),off(dof),cu(elem,dof));
#else
            EvalT stageval = cu(elem,dof);
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
        size_t uindex = sol_vals_index[set][var];
        auto u_AD = sol_vals[uindex];
        auto u_dot_AD = sol_dot_vals[uindex];
        auto off = subview(set_offsets[set],var,ALL());
        auto cu = subview(u,ALL(),var,ALL());
        auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
        auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
        
        parallel_for("wkset transient sol seedwhat 1",
                     TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VECTORSIZE),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          EvalT beta_u, beta_t;
          ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
          ScalarT timewt = one/dt/b_b(stage);
          ScalarT alpha_t = BDF(0)*timewt;
          for (size_type dof=team.team_rank(); dof<u_AD.extent(1); dof+=team.team_size() ) {
            
            // Get the stage solution
            ScalarT stageval = cu(elem,dof);
            
            // Compute the evaluating solution
            EvalT u_prev_val = cu_prev(elem,dof,0);
            if (index == 0) {
#ifndef MrHyDE_NO_AD
              u_prev_val = EvalT(u_prev_val.size(),off(dof),cu_prev(elem,dof,0));
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
              EvalT u_prev_val = cu_prev(elem,dof,s-1);
              if (index == (s-1)) {
#ifndef MrHyDE_NO_AD
                u_prev_val = EvalT(u_prev_val.size(),off(dof),cu_prev(elem,dof,s-1));
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
        size_t uindex = sol_vals_index[set][var];
        auto u_AD = sol_vals[uindex];
        auto u_dot_AD = sol_dot_vals[uindex];
        auto off = subview(set_offsets[set],var,ALL());
        auto cu = subview(u,ALL(),var,ALL());
        auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
        auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
        parallel_for("wkset transient sol seedwhat 1",
                     TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VECTORSIZE),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          EvalT beta_u, beta_t;
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
              EvalT u_stage_val = cu_stage(elem,dof,s);
              if (index == s) {
#ifndef MrHyDE_NO_AD
                u_stage_val = EvalT(u_stage_val.size(),off(dof),cu_stage(elem,dof,s));
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
        size_t uindex = sol_vals_index[set][var];
        auto u_AD = sol_vals[uindex];
        auto u_dot_AD = sol_dot_vals[uindex];
        auto off = subview(set_offsets[set],var,ALL());
        auto cu = subview(u,ALL(),var,ALL());
        auto cu_prev = subview(u_prev,ALL(),var,ALL(),ALL());
        auto cu_stage = subview(u_stage,ALL(),var,ALL(),ALL());
        
        parallel_for("wkset transient sol seedwhat 1",
                     TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VECTORSIZE),
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
      size_t uindex = sol_vals_index[set][var];
      auto u_AD = sol_vals[uindex];
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


template<>
void Workset<ScalarT>::computeSolnSteadySeeded(const size_t & set,
                                      View_Sc3 u,
                                      const int & seedwhat) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
  for (size_type var=0; var<u.extent(1); var++ ) {
    
    size_t uindex = sol_vals_index[set][var];
    auto u_AD = sol_vals[uindex];
    auto off = subview(set_offsets[set],var,ALL());
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

template<class EvalT>
void Workset<EvalT>::computeSolnSteadySeeded(const size_t & set,
                                      View_Sc3 u,
                                      const int & seedwhat) {
  
  Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
  for (size_type var=0; var<u.extent(1); var++ ) {
    
    size_t uindex = sol_vals_index[set][var];
    auto u_AD = sol_vals[uindex];
    auto off = subview(set_offsets[set],var,ALL());
    auto cu = subview(u,ALL(),var,ALL());
    if (seedwhat == 1 && set == current_set) {
      parallel_for("wkset steady soln",
                   RangePolicy<AssemblyExec>(0,u.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        EvalT dummyval = 0.0;
        for (size_type dof=0; dof<u_AD.extent(1); dof++ ) {
#ifndef MrHyDE_NO_AD
          u_AD(elem,dof) = EvalT(dummyval.size(), off(dof), cu(elem,dof));
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
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions for steady-state problems
////////////////////////////////////////////////////////////////////////////////////

template<>
void Workset<ScalarT>::computeParamSteadySeeded(View_Sc3 param,
                                      const int & seedwhat) {
  
  if (numDiscParams>0) {
    Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
    for (size_type var=0; var<param.extent(1); var++ ) {
      
      auto p_AD = pvals[var];
      auto off = subview(paramoffsets,var,ALL());
      auto cp = subview(param,ALL(),var,ALL());
      
      parallel_for("wkset steady soln",
                   RangePolicy<AssemblyExec>(0,param.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<p_AD.extent(1); dof++ ) {
          p_AD(elem,dof) = cp(elem,dof);
        }
      });
    }
  }
  
}


template<class EvalT>
void Workset<EvalT>::computeParamSteadySeeded(View_Sc3 param,
                                      const int & seedwhat) {
  
  if (numDiscParams>0) {
    Teuchos::TimeMonitor seedtimer(*worksetComputeSolnSeededTimer);
  
    for (size_type var=0; var<param.extent(1); var++ ) {
      
      auto p_AD = pvals[var];
      auto off = subview(paramoffsets,var,ALL());
      auto cp = subview(param,ALL(),var,ALL());
      if (seedwhat == 3) {
        parallel_for("wkset steady soln",
                     RangePolicy<AssemblyExec>(0,param.extent(0)),
                     KOKKOS_LAMBDA (const size_type elem ) {
          EvalT dummyval = 0.0;
          for (size_type dof=0; dof<p_AD.extent(1); dof++ ) {
#ifndef MrHyDE_NO_AD
            p_AD(elem,dof) = EvalT(dummyval.size(), off(dof), cp(elem,dof));
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

template<class EvalT>
void Workset<EvalT>::evaluateSolutionField(const int & fieldnum) {

  auto fielddata = soln_fields[fieldnum].data_;

  bool proceed = true;
  if (soln_fields[fieldnum].derivative_type_ == "time" ) {
    if (!isTransient) {
      proceed = false;
    }
    else if (isOnSide) {
      proceed = false;
    }
    else if (soln_fields[fieldnum].variable_type_ == "param") {
      proceed = false;
    }
  }
  if (soln_fields[fieldnum].variable_type_ == "aux") {
    proceed = false;
  }
  if (isOnPoint) {
    proceed = false;
  }
  
  if (proceed) {
    
    //-----------------------------------------------------
    // Get the appropriate view of seeded solution values
    //-----------------------------------------------------
    
    size_t sindex = soln_fields[fieldnum].set_index_;
    size_t vindex = soln_fields[fieldnum].variable_index_;
    
    View_EvalT2 solvals;
    size_t uindex = sol_vals_index[soln_fields[fieldnum].set_index_][soln_fields[fieldnum].variable_index_];
    if (soln_fields[fieldnum].variable_type_ == "solution") { // solution
      if (soln_fields[fieldnum].derivative_type_ == "time" ) {
        solvals = sol_dot_vals[uindex];
      }
      else {
        solvals = sol_vals[uindex];
      }
    }

    int basis_id;
    
    if (soln_fields[fieldnum].variable_type_ == "param") { // discr. params
      solvals = pvals[soln_fields[fieldnum].variable_index_];
      basis_id = paramusebasis[vindex];
    }
    else {
      basis_id = set_usebasis[sindex][vindex];
    }
    
    //-----------------------------------------------------
    // Get the appropriate basis values and evaluate the fields
    //-----------------------------------------------------
    
    int component = soln_fields[fieldnum].component_;
    
    if (soln_fields[fieldnum].derivative_type_ == "div" || (soln_fields[fieldnum].derivative_type_ == "div" && dimension == 2)) {
      auto sbasis = basis_div[basis_id];
      size_t teamSize = std::min(maxTeamSize,sbasis.extent(2));
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(sbasis.extent(0), teamSize, VECTORSIZE),
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
      CompressedView<View_Sc4> cbasis;
      if (soln_fields[fieldnum].derivative_type_ == "grad") {
        if (isOnSide) {
          cbasis = basis_grad_side[basis_id];
        }
        else {
          cbasis = basis_grad[basis_id];
        }
      }
      else if (soln_fields[fieldnum].derivative_type_ == "curl") {
        if (isOnSide) {
          // not implemented
        }
        else {
          cbasis = basis_curl[basis_id];
        }
      }
      else {
        if (isOnSide) {
          cbasis = basis_side[basis_id];
        }
        else {
          cbasis = basis[basis_id];
        }
      }
      
      size_t teamSize = std::min(maxTeamSize,cbasis.extent(2));
      size_type basis_dim = cbasis.extent(3);
      
      if (component >= basis_dim) {
        Kokkos::deep_copy(fielddata, 0.0);
        soln_fields[fieldnum].is_updated_ = true;
        return;
      }
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), teamSize, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          fielddata(elem,pt) = solvals(elem,0)*cbasis(elem,0,pt,component);
        }
        for (size_type dof=1; dof<cbasis.extent(1); dof++ ) {
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            fielddata(elem,pt) += solvals(elem,dof)*cbasis(elem,dof,pt,component);
          }
        }
      });
    }
    
    soln_fields[fieldnum].is_updated_ = true;
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the seeded solutions at specified side ip
////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::evaluateSideSolutionField(const int & fieldnum) {
  
  auto fielddata = side_soln_fields[fieldnum].data_;
  
  bool proceed = true;
  if (side_soln_fields[fieldnum].derivative_type_ == "time" ) {
    if (!isTransient) {
      proceed = false;
    }
    else if (isOnSide) {
      proceed = false;
    }
    else if (side_soln_fields[fieldnum].variable_type_ == "param") {
      proceed = false;
    }
  }
  if (side_soln_fields[fieldnum].variable_type_ == "aux") {
    proceed = false;
  }
  if (isOnPoint) {
    proceed = false;
  }
  
  if (proceed) {
    
    //-----------------------------------------------------
    // Get the appropriate view of seeded solution values
    //-----------------------------------------------------
    
    size_t sindex = side_soln_fields[fieldnum].set_index_;
    size_t vindex = side_soln_fields[fieldnum].variable_index_;
    
    View_EvalT2 solvals;
    size_t uindex = sol_vals_index[side_soln_fields[fieldnum].set_index_][side_soln_fields[fieldnum].variable_index_];
    if (side_soln_fields[fieldnum].variable_type_ == "solution") { // solution
      if (side_soln_fields[fieldnum].derivative_type_ == "time" ) {
        solvals = sol_dot_vals[uindex];
      }
      else {
        solvals = sol_vals[uindex];
      }
    }

    int basis_id;
    
    if (side_soln_fields[fieldnum].variable_type_ == "param") { // discr. params
      solvals = pvals[side_soln_fields[fieldnum].variable_index_];
      basis_id = paramusebasis[vindex];
    }
    else {
      basis_id = set_usebasis[sindex][vindex];
    }
    
    //-----------------------------------------------------
    // Get the appropriate basis values and evaluate the fields
    //-----------------------------------------------------
    
    int component = side_soln_fields[fieldnum].component_;
    
    if (side_soln_fields[fieldnum].derivative_type_ == "div") {
      auto sbasis = basis_div[basis_id];
      size_t teamSize = std::min(maxTeamSize,sbasis.extent(2));
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(sbasis.extent(0), teamSize, VECTORSIZE),
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
      CompressedView<View_Sc4> cbasis;
      if (side_soln_fields[fieldnum].derivative_type_ == "grad") {
        cbasis = basis_grad_side[basis_id];
      }
      else {
        cbasis = basis_side[basis_id];
      }
      
      size_t teamSize = std::min(maxTeamSize,cbasis.extent(2));
      size_type basis_dim = cbasis.extent(3);
      
      if (component >= basis_dim) {
        Kokkos::deep_copy(fielddata, 0.0);
        side_soln_fields[fieldnum].is_updated_ = true;
        return;
      }
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), teamSize, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          fielddata(elem,pt) = solvals(elem,0)*cbasis(elem,0,pt,component);
        }
        for (size_type dof=1; dof<cbasis.extent(1); dof++ ) {
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            fielddata(elem,pt) += solvals(elem,dof)*cbasis(elem,dof,pt,component);
          }
        }
      });
    }
    
    side_soln_fields[fieldnum].is_updated_ = true;
  }
  
}

////////////////////////////////////////////////////////////////////////////////////
// Compute the solutions at the side ip
////////////////////////////////////////////////////////////////////////////////////

// TMW: this function should be deprecated
// Gets used only in the boundaryCell flux calculation
// Will not work properly for multi-stage or multi-step

template<class EvalT>
void Workset<EvalT>::computeSolnSideIP(const int & side) { 
  
  {
    Teuchos::TimeMonitor basistimer(*worksetComputeSolnSideTimer);
    
    /////////////////////////////////////////////////////////////////////
    // HGRAD
    /////////////////////////////////////////////////////////////////////
    
    for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
      string var = varlist_HGRAD[current_set][i];
      int varind = vars_HGRAD[current_set][i];
      
      auto csol_vals = sol_vals[sol_vals_index[current_set][varind]];
      
      auto csol = this->getSolutionField(var,false);
      auto csol_x = this->getSolutionField("grad("+var+")[x]",false);
      auto csol_y = this->getSolutionField("grad("+var+")[y]",false);
      auto csol_z = this->getSolutionField("grad("+var+")[z]",false);
      auto cbasis = basis_side[usebasis[varind]];
      auto cbasis_grad = basis_grad_side[usebasis[varind]];
      
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        size_type dim = cbasis_grad.extent(3);
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csol(elem,pt) = 0.0;
          csol_x(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csol(elem,pt) += csol_vals(elem,dof)*cbasis(elem,dof,pt,0);
            csol_x(elem,pt) += csol_vals(elem,dof)*cbasis_grad(elem,dof,pt,0);
          }
          if (dim>1) {
            csol_y(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_y(elem,pt) += csol_vals(elem,dof)*cbasis_grad(elem,dof,pt,1);
            }
          }
          if (dim>2) {
            csol_z(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csol_z(elem,pt) += csol_vals(elem,dof)*cbasis_grad(elem,dof,pt,2);
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
      
      auto csol_vals = sol_vals[sol_vals_index[current_set][varind]];
      
      auto csol = this->getSolutionField(var,false);
      auto cbasis = basis_side[usebasis[varind]];
      
      parallel_for("wkset soln ip HVOL",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csol(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csol(elem,pt) += csol_vals(elem,dof)*cbasis(elem,dof,pt,0);
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
      
      auto csol_vals = sol_vals[sol_vals_index[current_set][varind]];
      
      auto csolx = this->getSolutionField(var+"[x]",false);
      auto csoly = this->getSolutionField(var+"[y]",false);
      auto csolz = this->getSolutionField(var+"[z]",false);
      auto cbasis = basis_side[usebasis[varind]];
      
      parallel_for("wkset soln ip HDIV",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        size_type dim = cbasis.extent(3);
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csolx(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csolx(elem,pt) += csol_vals(elem,dof)*cbasis(elem,dof,pt,0);
          }
          if (dim>1) {
            csoly(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csoly(elem,pt) += csol_vals(elem,dof)*cbasis(elem,dof,pt,1);
            }
          }
          if (dim>2) {
            csolz(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csolz(elem,pt) += csol_vals(elem,dof)*cbasis(elem,dof,pt,2);
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
      
      auto csol_vals = sol_vals[sol_vals_index[current_set][varind]];
      
      auto csolx = this->getSolutionField(var+"[x]",false);
      auto csoly = this->getSolutionField(var+"[y]",false);
      auto csolz = this->getSolutionField(var+"[z]",false);
      auto cbasis = basis_side[usebasis[varind]];
      
      parallel_for("wkset soln ip HCURL",
                   TeamPolicy<AssemblyExec>(cbasis.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        size_type dim = cbasis.extent(3);
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csolx(elem,pt) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csolx(elem,pt) += csol_vals(elem,dof)*cbasis(elem,dof,pt,0);
          }
          if (dim>1) {
            csoly(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csoly(elem,pt) += csol_vals(elem,dof)*cbasis(elem,dof,pt,1);
            }
          }
          if (dim>2) {
            csolz(elem,pt) = 0.0;
            for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
              csolz(elem,pt) += csol_vals(elem,dof)*cbasis(elem,dof,pt,2);
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

template<class EvalT>
void Workset<EvalT>::addAux(const vector<string> & auxvars, Kokkos::View<int**,AssemblyDevice> aoffs) {
  aux_offsets = aoffs;
  aux_varlist = auxvars;
  numAux = aux_varlist.size();
  flux = View_EvalT3("flux",numElem,numAux,numsideip);
  
  if (numAux > 0) {
    size_t maxAux = aux_offsets.extent(0)*aux_offsets.extent(1);
    if (maxAux > maxRes) {
      maxRes = maxAux;
      res = View_EvalT2("residual",numElem, maxRes);
    }
  }

  for (size_t i=0; i<aux_varlist.size(); ++i) {
    string var = aux_varlist[i];
    
    soln_fields.push_back(SolutionField<EvalT>("aux "+var,0,"aux",i)); // TMW: I think this is hard-coded for one basis type
    side_soln_fields.push_back(SolutionField<EvalT>("aux "+var,0,"aux",i));
    
    //soln_fields.push_back(SolutionField("aux "+var,0,"aux",i,"HGRAD",0,"",0,0,numip,false,false));
    //soln_fields.push_back(SolutionField("aux "+var+" side",0,"aux",i,"HGRAD",0,"",0,0,numsideip,true,false));
    
  }
}

//////////////////////////////////////////////////////////////
// Get a subview associated with a vector of parameters
//////////////////////////////////////////////////////////////

template<class EvalT>
Kokkos::View<EvalT*,Kokkos::LayoutStride,AssemblyDevice> Workset<EvalT>::getParameter(const string & name, bool & found) {
  found = false;
  size_t iter=0;
  Kokkos::View<EvalT*,Kokkos::LayoutStride,AssemblyDevice> pvals;
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

template<class EvalT>
void Workset<EvalT>::setTime(const ScalarT & newtime) {
  time = newtime;
}

//////////////////////////////////////////////////////////////
// Set deltat
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::setDeltat(const ScalarT & newdt) {
  deltat = newdt;
}

//////////////////////////////////////////////////////////////
// Set the stage index
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::setStage(const int & newstage) {
  current_stage = newstage;
}

template<class EvalT>
int Workset<EvalT>::addIntegratedQuantities(const int & nRequested) {

  int startingIndex = this->integrated_quantities.extent(0);

  // this should only be called when setting up the physics module
  // in the case of multiple physics defined on the same block requesting IQs,
  // integrated_quantities will get re-initialized until it's big
  // enough for all of them (we anticipate nTotal to be small here).

  this->integrated_quantities = 
    View_Sc1("integrated quantities",startingIndex+nRequested);

  return startingIndex;

}

//----------------------------------------------------------------

template<class EvalT>
void Workset<EvalT>::printSolutionFields() {
  cout << "Currently defined solution fields are: " << endl;
  for (size_t f=0; f<soln_fields.size(); ++f) {
    cout << soln_fields[f].expression_ << endl;
  }
  cout << "Currently defined side solution fields are: " << endl;
  for (size_t f=0; f<side_soln_fields.size(); ++f) {
    cout << side_soln_fields[f].expression_ << endl;
  }
  cout << "Currently defined point solution fields are: " << endl;
  for (size_t f=0; f<point_soln_fields.size(); ++f) {
    cout << point_soln_fields[f].expression_ << endl;
  }
}

template<class EvalT>
void Workset<EvalT>::printScalarFields() {
  cout << "Currently defined side scalar fields are: " << endl;
  for (size_t f=0; f<side_scalar_fields.size(); ++f) {
    cout << side_scalar_fields[f].expression_ << endl;
  }
  cout << "Currently defined point scalar fields are: " << endl;
  for (size_t f=0; f<point_scalar_fields.size(); ++f) {
    cout << point_scalar_fields[f].expression_ << endl;
  }
  cout << "Currently defined scalar fields are: " << endl;
  for (size_t f=0; f<scalar_fields.size(); ++f) {
    cout << scalar_fields[f].expression_ << endl;
  }
}

//////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////

template<class EvalT>
Kokkos::View<EvalT**,ContLayout,AssemblyDevice> Workset<EvalT>::getSolutionField(const string & label, const bool & evaluate, 
                                                                                 const bool & markUpdated) {
  
  Teuchos::TimeMonitor basistimer(*worksetgetDataTimer);
  
  View_EvalT2 outdata;
  
  if (isOnSide) {
    bool found = false;
    size_t ind = 0;
    while (!found && ind<side_soln_fields.size()) {
      if (label == side_soln_fields[ind].expression_) {
        found = true;
      }
      else {
        ++ind;
      }
    }
    if (!found) {
      std::cout << "Error: could not find a side solution field named " << label << std::endl;
      this->printSolutionFields();
    }
    else {
      this->checkSolutionFieldAllocation(ind);
      if (evaluate && !side_soln_fields[ind].is_updated_) {
        this->evaluateSideSolutionField(ind);
      }
      else if (markUpdated) {
        side_soln_fields[ind].is_updated_ = true;
      }
    }
    outdata = side_soln_fields[ind].data_;
  }
  else if (isOnPoint) {
    bool found = false;
    size_t ind = 0;
    while (!found && ind<point_soln_fields.size()) {
      if (label == point_soln_fields[ind].expression_) {
        found = true;
      }
      else {
        ++ind;
      }
    }
    if (!found) {
      std::cout << "Error: could not find a point solution field named " << label << std::endl;
      this->printSolutionFields();
    }
    else {
      this->checkSolutionFieldAllocation(ind);
      if (evaluate && !point_soln_fields[ind].is_updated_) {
        this->evaluateSolutionField(ind);
      }
      else if (markUpdated) {
        point_soln_fields[ind].is_updated_ = true;
      }
    }
    outdata = point_soln_fields[ind].data_;
  }
  else {
    bool found = false;
    size_t ind = 0;
    while (!found && ind<soln_fields.size()) {
      if (label == soln_fields[ind].expression_) {
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
      this->checkSolutionFieldAllocation(ind);
      if (evaluate && !soln_fields[ind].is_updated_) {
        this->evaluateSolutionField(ind);
      }
      else if (markUpdated) {
        soln_fields[ind].is_updated_ = true;
      }
    }
    outdata = soln_fields[ind].data_;
  }
  return outdata;
  
}

//////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::checkSolutionFieldAllocation(const size_t & ind) {
  
  if (isOnSide) {
    if (!side_soln_fields[ind].is_initialized_) {
      side_soln_fields[ind].initialize(maxElem,numsideip);
    }
  }
  else if (isOnPoint) {
    if (!point_soln_fields[ind].is_initialized_) {
      point_soln_fields[ind].initialize(maxElem,1);
    }
  }
  else {
    if (!soln_fields[ind].is_initialized_) {
      soln_fields[ind].initialize(maxElem,numip);
    }
  }
  
}

//////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::checkScalarFieldAllocation(const size_t & ind) {
  
  if (isOnSide) {
    if (!side_scalar_fields[ind].is_initialized_) {
      side_scalar_fields[ind].initialize(maxElem,numsideip);
    }
  }
  else if (isOnPoint) {
    if (!point_scalar_fields[ind].is_initialized_) {
      point_scalar_fields[ind].initialize(maxElem,1);
    }
  }
  else {
    if (!scalar_fields[ind].is_initialized_) {
      scalar_fields[ind].initialize(maxElem,numip);
    }
  }
}

//////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////

template<class EvalT>
View_Sc2 Workset<EvalT>::getScalarField(const string & label) {
  
  Teuchos::TimeMonitor basistimer(*worksetgetDataScTimer);
  View_Sc2 outdata;
  bool found = false;
  size_t ind = 0;
    
  if (isOnSide) {
    while (!found && ind<side_scalar_fields.size()) {
      if (label == side_scalar_fields[ind].expression_) {
        found = true;
      }
      else {
        ++ind;
      }
    }
    if (!found) {
      std::cout << "Error: could not find a side scalar field named " << label << std::endl;
      this->printScalarFields();
    }
    else {
      this->checkScalarFieldAllocation(ind);
      outdata = side_scalar_fields[ind].data_;
    }
  
  }
  else if (isOnPoint) {
    while (!found && ind<point_scalar_fields.size()) {
      if (label == point_scalar_fields[ind].expression_) {
        found = true;
      }
      else {
        ++ind;
      }
    }
    if (!found) {
      std::cout << "Error: could not find a side scalar field named " << label << std::endl;
      this->printScalarFields();
    }
    else {
      this->checkScalarFieldAllocation(ind);
      outdata = point_scalar_fields[ind].data_;
    }
  
  }
  else {
    while (!found && ind<scalar_fields.size()) {
      if (label == scalar_fields[ind].expression_) {
        found = true;
      }
      else {
        ++ind;
      }
    }
    if (!found) {
      std::cout << "Error: could not find a side scalar field named " << label << std::endl;
      this->printScalarFields();
    }
    else {
      this->checkScalarFieldAllocation(ind);
      outdata = scalar_fields[ind].data_;
    }
  
  }

  return outdata;
}

//////////////////////////////////////////////////////////////
// Function to determine which basis a variable uses
//////////////////////////////////////////////////////////////

template<class EvalT>
bool Workset<EvalT>::findBasisIndex(const string & var, int & basisindex) {
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

template<class EvalT>
bool Workset<EvalT>::isVar(const string & var, int & index) {
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

template<class EvalT>
bool Workset<EvalT>::isParameter(const string & var, int & index) {
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

template<class EvalT>
Kokkos::View<EvalT**,ContLayout,AssemblyDevice> Workset<EvalT>::getResidual() {
  return res;
}

//////////////////////////////////////////////////////////////
// Get the integration weights (interior)
//////////////////////////////////////////////////////////////

template<class EvalT>
CompressedView<View_Sc2> Workset<EvalT>::getWeights() {
  return wts;
}

//////////////////////////////////////////////////////////////
// Get the integration weights (boundary)
//////////////////////////////////////////////////////////////

template<class EvalT>
View_Sc2 Workset<EvalT>::getSideWeights() {
  return wts_side;
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasis(const string & var) {

  CompressedView<View_Sc4> dataout;
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

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasis(const int & index) {
  return basis[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisGrad(const string & var) {
  
  CompressedView<View_Sc4> dataout;
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

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisGrad(const int & index) {
  return basis_grad[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

template<class EvalT>
CompressedView<View_Sc3> Workset<EvalT>::getBasisDiv(const string & var) {
  
  CompressedView<View_Sc3> dataout;
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

template<class EvalT>
CompressedView<View_Sc3> Workset<EvalT>::getBasisDiv(const int & index) {
  return basis_div[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisCurl(const string & var) {
  
  CompressedView<View_Sc4> dataout;
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

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisCurl(const int & index) {
  return basis_curl[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisSide(const string & var) {
  
  CompressedView<View_Sc4> dataout;
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

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisSide(const int & index) {
  return basis_side[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisGradSide(const string & var) {
  
  CompressedView<View_Sc4> dataout;
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

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisGradSide(const int & index) {
  return basis_grad_side[index];
}

//////////////////////////////////////////////////////////////
// Extract a basis identified by a variable name (slower)
//////////////////////////////////////////////////////////////

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisCurlSide(const string & var) {
  
  CompressedView<View_Sc4> dataout;
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

template<class EvalT>
CompressedView<View_Sc4> Workset<EvalT>::getBasisCurlSide(const int & index) {
  return basis_curl_side[index];
}

//////////////////////////////////////////////////////////////
// Extract all of the offsets
//////////////////////////////////////////////////////////////

template<class EvalT>
Kokkos::View<int**,AssemblyDevice> Workset<EvalT>::getOffsets() {
  return offsets;
}

//////////////////////////////////////////////////////////////
// Extract the offsets for a particular variable
//////////////////////////////////////////////////////////////

template<class EvalT>
Kokkos::View<int*,Kokkos::LayoutStride,AssemblyDevice> Workset<EvalT>::getOffsets(const string & var) {
  
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
// Copy data carefully
//////////////////////////////////////////////////////////////

template<class EvalT>
template<class V1, class V2>
void Workset<EvalT>::copyData(V1 view1, V2 view2) {
  
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

template<class EvalT>
void Workset<EvalT>::setScalarField(View_Sc2 newdata, const string & expression) {
  
  bool found = false;
  size_t ind = 0;
  if (isOnSide) {
    while (!found && ind<side_scalar_fields.size()) {
      if (expression == side_scalar_fields[ind].expression_) {
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
      side_scalar_fields[ind].data_ = newdata;
      side_scalar_fields[ind].is_initialized_ = true;
    }
  }
  else if (isOnPoint) {
    while (!found && ind<point_scalar_fields.size()) {
      if (expression == point_scalar_fields[ind].expression_) {
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
      point_scalar_fields[ind].data_ = newdata;
      point_scalar_fields[ind].is_initialized_ = true;
    }
  }
  else {
    while (!found && ind<scalar_fields.size()) {
      if (expression == scalar_fields[ind].expression_) {
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
      scalar_fields[ind].data_ = newdata;
      scalar_fields[ind].is_initialized_ = true;
    }  
  }
  
}

//////////////////////////////////////////////////////////////
// Set the solutions
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::setSolution(View_EvalT4 newsol, const string & pfix) {
  // newsol has dims numElem x numvars x numip x dimension
  // however, this numElem may be smaller than the size of the data arrays
  
  for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
    string var = varlist_HGRAD[current_set][i];
    int varind = vars_HGRAD[current_set][i];
    auto csol = this->getSolutionField(var+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<varlist_HVOL[current_set].size(); i++) {
    string var = varlist_HVOL[current_set][i];
    int varind = vars_HVOL[current_set][i];
    auto csol = this->getSolutionField(var+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<varlist_HFACE[current_set].size(); i++) {
    string var = varlist_HFACE[current_set][i];
    int varind = vars_HFACE[current_set][i];
    auto csol = this->getSolutionField(var+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<varlist_HDIV[current_set].size(); i++) {
    string var = varlist_HDIV[current_set][i];
    int varind = vars_HDIV[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->getSolutionField(var+"[x]"+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getSolutionField(var+"[y]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getSolutionField(var+"[z]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  for (size_t i=0; i<varlist_HCURL[current_set].size(); i++) {
    string var = varlist_HCURL[current_set][i];
    int varind = vars_HCURL[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->getSolutionField(var+"[x]"+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getSolutionField(var+"[y]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getSolutionField(var+"[z]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  
}

//////////////////////////////////////////////////////////////
// Set the solution GRADs
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::setSolutionGrad(View_EvalT4 newsol, const string & pfix) {
  for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
    string var = varlist_HGRAD[current_set][i];
    int varind = vars_HGRAD[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->getSolutionField("grad("+var+")[x]"+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getSolutionField("grad("+var+")[y]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getSolutionField("grad("+var+")[z]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
}

//////////////////////////////////////////////////////////////
// Set the solution DIVs
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::setSolutionDiv(View_EvalT3 newsol, const string & pfix) {
  for (size_t i=0; i<varlist_HDIV[current_set].size(); i++) {
    string var = varlist_HDIV[current_set][i];
    int varind = vars_HDIV[current_set][i];
    auto csol = this->getSolutionField("div("+var+")",false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL());
    this->copyData(csol,cnsol);
  }
}

//////////////////////////////////////////////////////////////
// Set the solution CURLs
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::setSolutionCurl(View_EvalT4 newsol, const string & pfix) {
  for (size_t i=0; i<varlist_HCURL[current_set].size(); i++) {
    string var = varlist_HCURL[current_set][i];
    int varind = vars_HCURL[current_set][i];
    size_type dim = newsol.extent(3);
    auto csol = this->getSolutionField("curl("+var+")[x]"+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getSolutionField("curl("+var+")[y]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getSolutionField("curl("+var+")[z]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
}

//////////////////////////////////////////////////////////////
// Set the solution at a point
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::setSolutionPoint(View_EvalT2 newsol) {
  // newsol has dims numElem x numvars x numip x dimension
  for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
    string var = varlist_HGRAD[current_set][i];
    int varind = vars_HGRAD[current_set][i];
    auto csol = this->getSolutionField(var,false,true);
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
    auto csol = this->getSolutionField(var,false,true);
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
    auto csol = this->getSolutionField(var+"[x]",false,true);
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getSolutionField(var+"[y]",false,true);
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getSolutionField(var+"[z]",false,true);
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
    auto csol = this->getSolutionField(var+"[x]",false,true);
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getSolutionField(var+"[y]",false,true);
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getSolutionField(var+"[z]",false,true);
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

template<class EvalT>
void Workset<EvalT>::setSolutionGradPoint(View_EvalT2 newsol) {
  // newsol has dims numElem x numvars x numip x dimension
  for (size_t i=0; i<varlist_HGRAD[current_set].size(); i++) {
    string var = varlist_HGRAD[current_set][i];
    int varind = vars_HGRAD[current_set][i];
    size_type dim = newsol.extent(1);
    auto csol = this->getSolutionField("grad("+var+")[x]",false,true);
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getSolutionField("grad("+var+")[y]",false,true);
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getSolutionField("grad("+var+")[z]",false,true);
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

template<class EvalT>
void Workset<EvalT>::setParam(View_EvalT4 newsol, const string & pfix) {
  // newsol has dims numElem x numvars x numip x dimension
  // however, this numElem may be smaller than the size of the data arrays
  
  for (size_t i=0; i<paramvarlist_HGRAD.size(); i++) {
    string var = paramvarlist_HGRAD[i];
    int varind = paramvars_HGRAD[i];
    auto csol = this->getSolutionField(var+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<paramvarlist_HVOL.size(); i++) {
    string var = paramvarlist_HVOL[i];
    int varind = paramvars_HVOL[i];
    auto csol = this->getSolutionField(var+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<paramvarlist_HFACE.size(); i++) {
    string var = paramvarlist_HFACE[i];
    int varind = paramvars_HFACE[i];
    auto csol = this->getSolutionField(var+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
  }
  for (size_t i=0; i<paramvarlist_HDIV.size(); i++) {
    string var = paramvarlist_HDIV[i];
    int varind = paramvars_HDIV[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getSolutionField(var+"[x]"+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getSolutionField(var+"[y]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getSolutionField(var+"[z]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  for (size_t i=0; i<varlist_HCURL.size(); i++) {
    string var = paramvarlist_HCURL[i];
    int varind = paramvars_HCURL[i];
    size_type dim = newsol.extent(3);
    auto csol = this->getSolutionField(var+"[x]"+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),varind,ALL(),0);
    this->copyData(csol,cnsol);
    if (dim>1) {
      auto csol = this->getSolutionField(var+"[y]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),1);
      this->copyData(csol,cnsol);
    }
    if (dim>2) {
      auto csol = this->getSolutionField(var+"[z]"+pfix,false,true);
      auto cnsol = subview(newsol,ALL(),varind,ALL(),2);
      this->copyData(csol,cnsol);
    }
  }
  
}

//////////////////////////////////////////////////////////////
// Set the solution at a point
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::setParamPoint(View_EvalT2 newsol) {
  // newsol has dims numElem x numvars x numip x dimension
  for (size_t i=0; i<paramvarlist_HGRAD.size(); i++) {
    string var = paramvarlist_HGRAD[i];
    int varind = paramvars_HGRAD[i];
    auto csol = this->getSolutionField(var,false,true);
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
    auto csol = this->getSolutionField(var,false,true);
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
    auto csol = this->getSolutionField(var+"[x]",false,true);
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getSolutionField(var+"[y]",false,true);
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getSolutionField(var+"[z]",false,true);
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
    auto csol = this->getSolutionField(var+"[x]",false,true);
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getSolutionField(var+"[y]",false,true);
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getSolutionField(var+"[z]",false,true);
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

template<class EvalT>
void Workset<EvalT>::setParamGradPoint(View_EvalT2 newsol) {
  // newsol has dims numElem x numvars x numip x dimension
  for (size_t i=0; i<paramvarlist_HGRAD.size(); i++) {
    string var = paramvarlist_HGRAD[i];
    int varind = paramvars_HGRAD[i];
    size_type dim = newsol.extent(1);
    auto csol = this->getSolutionField("grad("+var+")[x]",false,true);
    auto cnsol = subview(newsol,varind,ALL());
    parallel_for("physics point response",
                 RangePolicy<AssemblyExec>(0,1),
                 KOKKOS_LAMBDA (const int elem ) {
      csol(0,0) = cnsol(0);
    });
    if (dim>1) {
      auto csol = this->getSolutionField("grad("+var+")[y]",false,true);
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(1);
      });
    }
    if (dim>2) {
      auto csol = this->getSolutionField("grad("+var+")[z]",false,true);
      parallel_for("physics point response",
                   RangePolicy<AssemblyExec>(0,1),
                   KOKKOS_LAMBDA (const int elem ) {
        csol(0,0) = cnsol(2);
      });
    }
  }

}

template<class EvalT>
void Workset<EvalT>::setAux(View_EvalT4 newsol, const string & pfix) {
  // newsol has dims numElem x numvars x numip x dimension
  // however, this numElem may be smaller than the size of the data arrays

  // currently the new solution must be ordered appropriately
  
  for (size_t i=0; i<aux_varlist.size(); i++) {
    string var = aux_varlist[i];
    auto csol = this->getSolutionField("aux "+var+pfix,false,true);
    auto cnsol = subview(newsol,ALL(),i,ALL(),0);
    this->copyData(csol,cnsol);
  }
  
}

template<class EvalT>
string Workset<EvalT>::getParamBasisType(string & name) {
  string type = "none";

  bool found = false; 
  for (size_t i=0; i<paramvarlist_HGRAD.size(); ++i) {
    if (!found && paramvarlist_HGRAD[i] == name) {
      type = "HGRAD";
      found = true;
    }
  }
  for (size_t i=0; i<paramvarlist_HVOL.size(); ++i) {
    if (!found && paramvarlist_HVOL[i] == name) {
      type = "HVOL";
      found = true;
    }
  }
  for (size_t i=0; i<paramvarlist_HDIV.size(); ++i) {
    if (!found && paramvarlist_HDIV[i] == name) {
      type = "HDIV";
      found = true;
    }
  }
  for (size_t i=0; i<paramvarlist_HCURL.size(); ++i) {
    if (!found && paramvarlist_HCURL[i] == name) {
      type = "HCURL";
      found = true;
    }
  }
  for (size_t i=0; i<paramvarlist_HFACE.size(); ++i) {
    if (!found && paramvarlist_HFACE[i] == name) {
      type = "HFACE";
      found = true;
    }
  }
  
  return type;

}

//////////////////////////////////////////////////////////////
// Update the set-specific workset attributes
//////////////////////////////////////////////////////////////

/**
 * @brief Update the set-specific workset attributes
 * 
 * @param[in] current_set_ The index of the current physics set
 */

template<class EvalT>
void Workset<EvalT>::updatePhysicsSet(const size_t & current_set_) {
  if (isInitialized) {
    if (numSets>1) {
      current_set = current_set_;
      offsets = set_offsets[current_set];
      usebasis = set_usebasis[current_set];
      varlist = set_varlist[current_set];
      var_bcs = set_var_bcs[current_set];
      butcher_A = set_butcher_A[current_set];
      butcher_b = set_butcher_b[current_set];
      butcher_c = set_butcher_c[current_set];
      BDF_wts = set_BDF_wts[current_set];
    }
  }
}

//////////////////////////////////////////////////////////////
// Allocate the rotation tensor
//////////////////////////////////////////////////////////////

template<class EvalT>
void Workset<EvalT>::allocateRotations() {
  if (rotation.extent_int(0) < numElem) {
    rotation = View_Sc3("rotations", numElem, 3, 3);
  }
}

//////////////////////////////////////////////////////////////
// Allocate and return the element sizes, h
//////////////////////////////////////////////////////////////

template<class EvalT>
View_Sc1 Workset<EvalT>::getElementSize() {
  View_Sc1 hsize("tmp hsize",wts.extent(0));
  parallel_for("elem size",
               RangePolicy<AssemblyExec>(0,wts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT vol = 0.0;
    for (size_type i=0; i<wts.extent(1); i++) {
      vol += wts(elem,i);
    }
    ScalarT dimscl = 1.0/(ScalarT)dimension;
    hsize(elem) = std::pow(vol,dimscl);
  });
  return hsize;
}

//////////////////////////////////////////////////////////////
// Allocate and return the element sizes, h
//////////////////////////////////////////////////////////////

template<class EvalT>
View_Sc1 Workset<EvalT>::getSideElementSize() {
  
  View_Sc1 hsize("tmp hsize",wts_side.extent(0));
  parallel_for("elem size",
               RangePolicy<AssemblyExec>(0,wts_side.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT vol = 0.0;
    for (size_type i=0; i<wts_side.extent(1); i++) {
      vol += wts_side(elem,i);
    }
    ScalarT dimscl = 1.0/((ScalarT)dimension-1.0);
    hsize(elem) = std::pow(vol,dimscl);
  });
  return hsize;
}

//////////////////////////////////////////////////////////////
// Get the view containing the current integrated quantities
//////////////////////////////////////////////////////////////

template<class EvalT>
View_Sc1 Workset<EvalT>::getIntegratedQuantities() {
  return integrated_quantities;
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::Workset<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::Workset<AD>;

// Standard built-in types
template class MrHyDE::Workset<AD2>;
template class MrHyDE::Workset<AD4>;
template class MrHyDE::Workset<AD8>;
template class MrHyDE::Workset<AD16>;
template class MrHyDE::Workset<AD18>;
template class MrHyDE::Workset<AD24>;
template class MrHyDE::Workset<AD32>;
#endif
