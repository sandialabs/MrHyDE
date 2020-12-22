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

#ifndef WKSET_H
#define WKSET_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  class workset {
  public:
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    ////////////////////////////////////////////////////////////////////////////////////
    
    workset() {};
    
    workset(const vector<int> & cellinfo, const bool & isTransient_,
            const vector<string> & basis_types_,
            const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
            const topo_RCP & topo, Kokkos::View<int**,HostDevice> & var_bcs_);
    
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
    
    void computeSolnVolIP(View_Sc3 u);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the seeded solutions for general transient problems
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnTransientSeeded(View_Sc3 u,
                                    View_Sc4 u_prev,
                                    View_Sc4 u_stage,
                                    const int & seedwhat,
                                    const int & index=0);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the seeded solutions for steady-state problems
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnSteadySeeded(View_Sc3 u, const int & seedwhat);
    
    void computeParamSteadySeeded(View_Sc3 params, const int & seedwhat);
    
    void computeAuxSolnSteadySeeded(View_Sc3 aux, const int & seedwhat);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at general set of points
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSoln(const int & type);
    
    void computeParam(const int & type);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at the volumetric ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnVolIP();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the discretized parameters at the volumetric ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeParamVolIP(View_Sc3 param, const int & seedwhat);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at the side ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnSideIP();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at the face ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnFaceIP();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at the face ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeAuxSolnFaceIP();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the discretized parameters at the side ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeParamSideIP(const int & side, View_Sc3 param, const int & seedwhat);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at the side ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnSideIP(const int & side);
    
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
    Kokkos::View<int**,HostDevice> var_bcs, aux_var_bcs;
    
    Kokkos::View<int**,AssemblyDevice> offsets, paramoffsets, aux_offsets;
    vector<string> varlist, aux_varlist;
    Kokkos::View<ScalarT**,AssemblyDevice> butcher_A, aux_butcher_A;
    Kokkos::View<ScalarT*,AssemblyDevice> butcher_b, butcher_c, BDF_wts, aux_butcher_b, aux_butcher_c, aux_BDF_wts;
    
    vector<int> usebasis, paramusebasis, aux_usebasis;
    vector<int> vars_HGRAD, vars_HVOL, vars_HDIV, vars_HCURL, vars_HFACE;
    vector<int> paramvars_HGRAD, paramvars_HVOL, paramvars_HDIV, paramvars_HCURL, paramvars_HFACE;
    
    bool isAdjoint, onlyTransient, isTransient;
    bool isInitialized, usebcs;
    topo_RCP celltopo;
    size_t numsides, numip, numsideip, numVars, numParams, numAux;
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
    
    size_t block, localEID, globalEID;
    
    // Views that use ContLayout for hierarchical parallelism
    View_Sc1 h;
    View_Sc3 ip, ip_side, normals, point;
    View_Sc2 wts, wts_side;
    
    View_AD3 uvals, u_dotvals, pvals, auxvals, aux_dotvals;
    
    vector<View_Sc4> basis, basis_grad, basis_curl, basis_side, basis_face, basis_grad_side, basis_grad_face, basis_curl_side, basis_curl_face;
    vector<View_Sc3> basis_div, basis_div_side;
    
    View_AD4 local_soln, local_soln_grad, local_soln_dot, local_aux_dot, local_soln_dot_grad, local_soln_curl;
    View_AD3 local_soln_div, local_param_div, local_aux_div;
    View_AD4 local_param, local_param_side, local_param_curl;
    View_AD4 local_aux, local_aux_side, local_aux_curl;
    View_AD4 local_param_grad, local_aux_grad, local_param_grad_side, local_aux_grad_side;
    View_AD4 local_soln_side, local_soln_grad_side, local_soln_dot_side;
    View_AD4 local_soln_face, local_soln_grad_face, local_aux_face, local_aux_grad_face;
    
    View_AD4 local_soln_point, local_soln_grad_point, local_param_grad_point;
    View_AD4 local_param_point;
    
    View_AD3 scratch, flux;
    View_AD2 res, adjrhs;
    
    int sidetype;
    Kokkos::View<int****,AssemblyDevice> sideinfo;
    string sidename, var;
    int currentside;
    
    bool have_rotation, have_rotation_phi;
    View_Sc3 rotation;
    View_Sc2 rotation_phi, extra_data;
    
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
  
}

#endif
