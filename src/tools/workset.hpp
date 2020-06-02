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
  
  workset(const vector<int> & cellinfo, const DRV & ref_ip_, const DRV & ref_wts_, const DRV & ref_side_ip_,
          const DRV & ref_side_wts_, const vector<string> & basis_types_,
          const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
          const topo_RCP & topo, Kokkos::View<int**,UnifiedDevice> & var_bcs_);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Public functions
  ////////////////////////////////////////////////////////////////////////////////////
  
  void setupBasis();
  
  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////
  
  void setupParamBasis();
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Update the nodes and the basis functions at the volumetric ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void update(const DRV & ip_, const DRV & wts_, const DRV & jacobian,
              const DRV & jacobianInv, const DRV & jacobianDet,
              Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & orientation);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Add a side information
  ////////////////////////////////////////////////////////////////////////////////////
  
  int addSide(const DRV & nodes, const int & sidenum, Kokkos::View<LO*,AssemblyDevice> & localSideID,
              Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & orientation);

  ////////////////////////////////////////////////////////////////////////////////////
  // Update the nodes and the basis functions at the side ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void updateFace(const DRV & nodes,
                  Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & orientation,
                  const size_t & sidenum);

  ////////////////////////////////////////////////////////////////////////////////////
  // Update the nodes and the basis functions at the side ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void updateSide(const int & sidenum, const int & cnum);

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
  // Compute the solutions at the volumetric ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnVolIP(Kokkos::View<ScalarT***,AssemblyDevice> u,
                        Kokkos::View<ScalarT****,AssemblyDevice> u_prev,
                        Kokkos::View<ScalarT****,AssemblyDevice> u_stage,
                        Kokkos::View<int*,UnifiedDevice> seedwhat);

  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the discretized parameters at the volumetric ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeParamVolIP(Kokkos::View<ScalarT***,AssemblyDevice> param,
                         Kokkos::View<int*,UnifiedDevice> seedwhat);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the solutions at the side ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnSideIP(Kokkos::View<ScalarT***,AssemblyDevice> u,
                         Kokkos::View<int*,UnifiedDevice> seedwhat);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the solutions at the face ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolnFaceIP(Kokkos::View<ScalarT***,AssemblyDevice> u,
                         Kokkos::View<int*,UnifiedDevice> seedwhat);
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Compute the discretized parameters at the side ip
  ////////////////////////////////////////////////////////////////////////////////////
  
  void computeParamSideIP(const int & side, Kokkos::View<ScalarT***,AssemblyDevice> param,
                          Kokkos::View<int*,UnifiedDevice> seedwhat);
  
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
  
  ////////////////////////////////////////////////////////////////////////////////////
  // Public data
  ////////////////////////////////////////////////////////////////////////////////////
  
  // Data filled by constructor
  //vector<vector<int> > offsets, paramoffsets;
  Kokkos::View<int**,AssemblyDevice> offsets, paramoffsets;
  vector<string> varlist;
  //Teuchos::RCP<TimeIntegrator> timeInt;
  // ScalarT wts to compute u and udot for transient problems
  Kokkos::View<ScalarT**,UnifiedDevice> butcher_A;
  Kokkos::View<ScalarT*,UnifiedDevice> butcher_b, butcher_c, BDF_wts;
  
  vector<int> usebasis, paramusebasis;
  bool isAdjoint, onlyTransient, isTransient;
  bool isInitialized, usebcs;
  topo_RCP celltopo;
  size_t numsides, numip, numsideip, numVars, numParams, numAux, numDOF;
  int dimension, numElem, current_stage;
  Kokkos::View<int*,UnifiedDevice> current_stage_KV; // for access on device
  
  DRV ref_ip, ref_side_ip, ref_wts, ref_side_wts;
  vector<DRV> ref_side_ip_vec, ref_side_normals_vec, ref_side_tangents_vec, ref_side_tangentsU_vec, ref_side_tangentsV_vec;
  vector<string> basis_types;
  vector<int> numbasis, numparambasis;
  vector<basis_RCP> basis_pointers, param_basis_pointers;
  vector<DRV> param_basis, param_basis_grad;
  vector<DRV> param_basis_ref, param_basis_grad_ref;
  vector<DRV> param_basis_grad_side, param_basis_side;
  
  // Scalar and vector parameters
  // Stored as a vector of pointers (vector does not change but the data does)
  vector<Teuchos::RCP<vector<AD> > > params;
  Kokkos::View<AD**,AssemblyDevice> params_AD;
  
  vector<string> paramnames;
  
  // Data computed after construction (depends only on reference element)
  vector<DRV> ref_basis, ref_basis_grad, ref_basis_div, ref_basis_curl;
  vector<vector<DRV> > ref_basis_side, ref_basis_grad_side, ref_basis_div_side, ref_basis_curl_side;
  vector<vector<DRV> > param_basis_side_ref, param_basis_grad_side_ref;
  
  // Data recomputed often (depends on physical element and time)
  ScalarT time, alpha, deltat;
  Kokkos::View<ScalarT*,AssemblyDevice> time_KV;
  Kokkos::View<ScalarT*,AssemblyDevice> deltat_KV;
  
  Kokkos::View<ScalarT*,AssemblyDevice> h;
  size_t block, localEID, globalEID;
  DRV ip, ip_side, wts, wts_side, normals;
  Kokkos::View<ScalarT***,AssemblyDevice> ip_KV, ip_side_KV, normals_KV, point_KV;
  
  // experimental data structures
  vector<DRV> ip_side_vec, wts_side_vec, normals_side_vec;
  vector<vector<DRV> > param_basis_grad_side_vec, param_basis_side_vec;
  vector<vector<DRV> > basis_side_vec, basis_side_uw_vec, basis_grad_side_vec, basis_grad_side_uw_vec;
  Kokkos::View<int**,UnifiedDevice> var_bcs;
  
  vector<DRV> basis; // weighted by volumetric integration weights
  vector<DRV> basis_uw; // un-weighted
  vector<DRV> basis_grad; // weighted by volumetric integration weights
  vector<DRV> basis_grad_uw; // un-weighted
  vector<DRV> basis_div; // weighted by volumetric integration weights
  vector<DRV> basis_div_uw; // un-weighted
  vector<DRV> basis_curl; // weighted by volumetric integration weights
  vector<DRV> basis_curl_uw; // un-weighted
  
  vector<DRV> basis_side, basis_side_uw, basis_grad_side, basis_grad_side_uw;
  vector<DRV> basis_face, basis_face_uw, basis_grad_face, basis_grad_face_uw;
  vector<DRV> basis_div_side, basis_div_side_uw, basis_curl_side, basis_curl_side_uw;
  
  Kokkos::View<AD****, AssemblyDevice> local_soln, local_soln_grad, local_soln_dot, local_soln_dot_grad, local_soln_curl;
  Kokkos::View<AD***, AssemblyDevice> local_soln_div, local_param, local_aux, local_param_side, local_aux_side;
  Kokkos::View<AD****, AssemblyDevice> local_param_grad, local_aux_grad, local_param_grad_side, local_aux_grad_side;
  Kokkos::View<AD****, AssemblyDevice> local_soln_side, local_soln_grad_side, local_soln_dot_side;
  Kokkos::View<AD****, AssemblyDevice> local_soln_face, local_soln_grad_face;
  
  Kokkos::View<AD****, AssemblyDevice> local_soln_point, local_soln_grad_point, local_param_grad_point;
  Kokkos::View<AD***, AssemblyDevice> local_param_point;
  
  //DRV jacobian, jacobDet, jacobInv, weightedMeasure;
  //DRV sidejacobian, sidejacobDet, sidejacobInv, sideweightedMeasure;
  DRV jacobDet, jacobInv, weightedMeasure;
  DRV sidejacobDet, sidejacobInv, sideweightedMeasure;
  //vector<DRV> param_basis_grad; // parameter basis function are never weighted
  // Dynamic data (changes multiple times per element)
  int sidetype;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  string sidename, var;
  int currentside;
  Kokkos::View<AD**,AssemblyDevice> res, adjrhs;
  Kokkos::View<AD***,AssemblyDevice> flux;
  //FCAD scratch, sidescratch;
  
  bool have_rotation, have_rotation_phi;
  Kokkos::View<ScalarT***,AssemblyDevice> rotation;
  Kokkos::View<ScalarT**,AssemblyDevice> rotation_phi;
  
  ScalarT y; // index parameter for fractional operators
  ScalarT s; // fractional exponent
  
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
  Teuchos::RCP<Teuchos::Time> worksetComputeSolnVolTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::computeSolnVolIP");
  Teuchos::RCP<Teuchos::Time> worksetComputeSolnSideTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::computeSolnSideIP");
  Teuchos::RCP<Teuchos::Time> worksetComputeParamVolTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::computeParamVolIP");
  Teuchos::RCP<Teuchos::Time> worksetComputeParamSideTimer = Teuchos::TimeMonitor::getNewCounter("MILO::workset::computeParamSideIP");
  
  Teuchos::RCP<Teuchos::Time> worksetDebugTimer0 = Teuchos::TimeMonitor::getNewCounter("MILO::workset::debug0");
  Teuchos::RCP<Teuchos::Time> worksetDebugTimer1 = Teuchos::TimeMonitor::getNewCounter("MILO::workset::debug1");
  Teuchos::RCP<Teuchos::Time> worksetDebugTimer2 = Teuchos::TimeMonitor::getNewCounter("MILO::workset::debug2");
  
};

#endif
