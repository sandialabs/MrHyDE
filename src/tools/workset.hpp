/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef WKSET_H
#define WKSET_H

#include "trilinos.hpp"
#include "preferences.hpp"

class workset {
  public:
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Constructors
  ////////////////////////////////////////////////////////////////////////////////////
  
  workset() {};
  
  workset(const vector<int> & cellinfo, const bool & isTransient_,
          const DRV & ref_ip_, const DRV & ref_wts_, const DRV & ref_side_ip_,
          const DRV & ref_side_wts_,
          const vector<string> & basis_types_,
          const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
          const topo_RCP & topo, Kokkos::View<int**,HostDevice> & var_bcs_);

  //KOKKOS_INLINE_FUNCTION
  void createViews();// {
  /*
  deltat = 1.0;
  deltat_KV = Kokkos::View<ScalarT*,AssemblyDevice>("deltat",1);
  Kokkos::deep_copy(deltat_KV,deltat);
  
  current_stage_KV = Kokkos::View<int*,AssemblyDevice>("stage number on device",1);
  Kokkos::deep_copy(current_stage_KV,0);
  // Integration information
  time_KV = Kokkos::View<ScalarT*,AssemblyDevice>("time",1); // defaults to 0.0
  
  // these can point to different arrays
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
  auto host_res = Kokkos::create_mirror_view(res);
  parallel_for(RangePolicy<HostExec>(0,host_res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (int dof=0; dof<host_res.extent(1); dof++) {
      host_res(elem,dof) = 0.0;
    }
  });
  Kokkos::deep_copy(res,host_res);

  have_rotation = false;
  have_rotation_phi = false;
  rotation = Kokkos::View<ScalarT***,AssemblyDevice>("rotation matrix",numElem,3,3);
  
  int maxb = 0;
  for (size_t i=0; i<basis_pointers.size(); i++) {
    int numb = basis_pointers[i]->getCardinality();
    maxb = std::max(maxb,numb);
  }
  
  uvals = Kokkos::View<AD***,AssemblyDevice>("seeded uvals",numElem, numVars, maxb);
  auto host_uvals = Kokkos::create_mirror_view(uvals);
  parallel_for(RangePolicy<HostExec>(0,host_uvals.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (int var=0; var<host_uvals.extent(1); var++) {
      for (int dof=0; dof<host_uvals.extent(2); dof++) {
        host_uvals(elem,var,dof) = 0.0;
      }
    }
  });
  Kokkos::deep_copy(uvals,host_uvals);
  if (isTransient) {
    u_dotvals = Kokkos::View<AD***,AssemblyDevice>("seeded uvals",numElem, numVars, maxb);
  }
  }
i */

  ////////////////////////////////////////////////////////////////////////////////////
  // Public functions
  ////////////////////////////////////////////////////////////////////////////////////
  
  void createSolns();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Reset solution to zero
  ////////////////////////////////////////////////////////////////////////////////////
  
  void resetResidual();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Reset solution to zero
  ////////////////////////////////////////////////////////////////////////////////////
  
  void resetResidual(const int & numE);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Reset solution to zero
  ////////////////////////////////////////////////////////////////////////////////////
  
  void resetFlux();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Reset solution to zero
  ////////////////////////////////////////////////////////////////////////////////////
  
  void resetAux();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Reset solution to zero
  ////////////////////////////////////////////////////////////////////////////////////
  
  void resetAuxSide();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Reset solution to zero
  ////////////////////////////////////////////////////////////////////////////////////
  
  void resetAdjointRHS();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the solutions at the volumetric ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnVolIP(Kokkos::View<ScalarT***,AssemblyDevice> u);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the seeded solutions for general transient problems
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnTransientSeeded(Kokkos::View<ScalarT***,AssemblyDevice> u,
                                  Kokkos::View<ScalarT****,AssemblyDevice> u_prev,
                                  Kokkos::View<ScalarT****,AssemblyDevice> u_stage,
                                  const int & seedwhat);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the seeded solutions for steady-state problems
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnSteadySeeded(Kokkos::View<ScalarT***,AssemblyDevice> u,
                               const int & seedwhat);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the solutions at general set of points
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSoln(const int & type);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the solutions at the volumetric ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnVolIP();

  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the discretized parameters at the volumetric ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeParamVolIP(Kokkos::View<ScalarT***,AssemblyDevice> param,
                         const int & seedwhat);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the solutions at the side ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnSideIP();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the solutions at the face ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnFaceIP();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the discretized parameters at the side ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeParamSideIP(const int & side, Kokkos::View<ScalarT***,AssemblyDevice> param,
                          const int & seedwhat);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the solutions at the side ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnSideIP(const int & side, Kokkos::View<AD***,AssemblyDevice> u_AD,
                         Kokkos::View<AD***,AssemblyDevice> param_AD);
  
  //////////////////////////////////////////////////////////////
  // Add Aux
  //////////////////////////////////////////////////////////////
  
  void addAux(const size_t & naux);
  
  //////////////////////////////////////////////////////////////
  // Get a pointer to vector of parameters
  //////////////////////////////////////////////////////////////
  
  vector<AD> getParam(const string & name, bool & found);
  
  //////////////////////////////////////////////////////////////
  // Set the time
  //////////////////////////////////////////////////////////////
  
  void setTime(const ScalarT & newtime);
  
  //////////////////////////////////////////////////////////////
  // Set deltat
  //////////////////////////////////////////////////////////////
  
  void setDeltat(const ScalarT & newdt);
  
  //////////////////////////////////////////////////////////////
  // Set the stage index
  //////////////////////////////////////////////////////////////
  
  void setStage(const int & newstage);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Public data
  ////////////////////////////////////////////////////////////////////////////////////
  
  // Should be the only view stored on Host
  // Used by physics modules to determine the proper contribution to the boundary residual
  Kokkos::View<int**,HostDevice> var_bcs;
  
  Kokkos::View<int**,AssemblyDevice> offsets, paramoffsets;
  vector<string> varlist;
  Kokkos::View<ScalarT**,AssemblyDevice> butcher_A;
  Kokkos::View<ScalarT*,AssemblyDevice> butcher_b, butcher_c, BDF_wts;
  
  vector<int> usebasis, paramusebasis;
  vector<int> vars_HGRAD, vars_HVOL, vars_HDIV, vars_HCURL, vars_HFACE;
  bool isAdjoint, onlyTransient, isTransient;
  bool isInitialized, usebcs;
  topo_RCP celltopo;
  size_t numsides, numip, numsideip, numVars, numParams, numAux, numDOF;
  int dimension, numElem, current_stage;
  Kokkos::View<int*,AssemblyDevice> current_stage_KV; // for access on device
  
  vector<string> basis_types;
  vector<int> numbasis;
  vector<basis_RCP> basis_pointers;
  
  vector<Teuchos::RCP<vector<AD> > > params;
  Kokkos::View<AD**,AssemblyDevice> params_AD;
  vector<string> paramnames;
  
  ScalarT time, alpha, deltat;
  Kokkos::View<ScalarT*,AssemblyDevice> time_KV;
  Kokkos::View<ScalarT*,AssemblyDevice> deltat_KV;
  
  Kokkos::View<ScalarT*,AssemblyDevice> h;
  size_t block, localEID, globalEID;
  DRV ip, ip_side, wts, wts_side, normals;
  Kokkos::View<ScalarT***,AssemblyDevice> ip_KV, ip_side_KV, normals_KV, point_KV;
  
  Kokkos::View<AD***,AssemblyDevice> uvals, u_dotvals;
  
  vector<DRV> basis;
  vector<DRV> basis_grad;
  vector<DRV> basis_div;
  vector<DRV> basis_curl;
  vector<DRV> basis_side, basis_grad_side;
  vector<DRV> basis_face, basis_grad_face;
  vector<DRV> basis_div_side, basis_curl_side;
  
  Kokkos::View<AD****, AssemblyDevice> local_soln, local_soln_grad, local_soln_dot, local_soln_dot_grad, local_soln_curl;
  Kokkos::View<AD***, AssemblyDevice> local_soln_div, local_param, local_aux, local_param_side, local_aux_side;
  Kokkos::View<AD****, AssemblyDevice> local_param_grad, local_aux_grad, local_param_grad_side, local_aux_grad_side;
  Kokkos::View<AD****, AssemblyDevice> local_soln_side, local_soln_grad_side, local_soln_dot_side;
  Kokkos::View<AD****, AssemblyDevice> local_soln_face, local_soln_grad_face;
  
  //Kokkos::View<AD****, AssemblyDevice> local_soln_reset, local_soln_grad_reset, local_soln_dot_reset, local_soln_curl_reset;
  //Kokkos::View<AD***, AssemblyDevice> local_soln_div_reset;
  
  Kokkos::View<AD****, AssemblyDevice> local_soln_point, local_soln_grad_point, local_param_grad_point;
  Kokkos::View<AD***, AssemblyDevice> local_param_point;
  
  int sidetype;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  string sidename, var;
  int currentside;
  Kokkos::View<AD**,AssemblyDevice> res, adjrhs;
  Kokkos::View<AD***,AssemblyDevice> flux;
  
  bool have_rotation, have_rotation_phi;
  Kokkos::View<ScalarT***,AssemblyDevice> rotation;
  Kokkos::View<ScalarT**,AssemblyDevice> rotation_phi, extra_data;
  
  // Profile timers
  Teuchos::RCP<Teuchos::Time> worksetUpdateIPTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::update - integration data");
  Teuchos::RCP<Teuchos::Time> worksetUpdateBasisMMTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::update::multiplyMeasure");
  Teuchos::RCP<Teuchos::Time> worksetUpdateBasisHGTGTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::update::HGRADTransformGrad");
  Teuchos::RCP<Teuchos::Time> worksetAddSideTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::addSide");
  Teuchos::RCP<Teuchos::Time> worksetSideUpdateIPTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::updateSide - integration data");
  Teuchos::RCP<Teuchos::Time> worksetSideUpdateBasisTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::updateSide - basis data");
  Teuchos::RCP<Teuchos::Time> worksetFaceUpdateIPTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::updateFace - integration data");
  Teuchos::RCP<Teuchos::Time> worksetFaceUpdateBasisTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::updateFace - basis data");
  Teuchos::RCP<Teuchos::Time> worksetResetTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::reset*");
  Teuchos::RCP<Teuchos::Time> worksetComputeSolnVolTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::computeSolnVolIP - compute seeded sol at ip");
  Teuchos::RCP<Teuchos::Time> worksetComputeSolnSeededTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::computeSolnVolIP - allocate/compute seeded");
  Teuchos::RCP<Teuchos::Time> worksetComputeSolnSideTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::computeSolnSideIP");
  Teuchos::RCP<Teuchos::Time> worksetComputeParamVolTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::computeParamVolIP");
  Teuchos::RCP<Teuchos::Time> worksetComputeParamSideTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::computeParamSideIP");
  
  Teuchos::RCP<Teuchos::Time> worksetDebugTimer0 = Teuchos::TimeMonitor::getNewCounter("MILO::workset::debug0");
  Teuchos::RCP<Teuchos::Time> worksetDebugTimer1 = Teuchos::TimeMonitor::getNewCounter("MILO::workset::debug1");
  Teuchos::RCP<Teuchos::Time> worksetDebugTimer2 = Teuchos::TimeMonitor::getNewCounter("MILO::workset::debug2");
  
};

#endif
