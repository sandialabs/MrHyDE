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

#include "cell.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

using namespace MrHyDE;

cell::cell(const Teuchos::RCP<CellMetaData> & cellData_,
           const DRV nodes_,
           const Kokkos::View<LO*,AssemblyDevice> localID_,
           LIDView LIDs_,
           Kokkos::View<int****,HostDevice> sideinfo_,
           Teuchos::RCP<DiscretizationInterface> & disc_) :
LIDs(LIDs_), cellData(cellData_), localElemID(localID_), sideinfo(sideinfo_), nodes(nodes_), disc(disc_)
{
  numElem = nodes.extent(0);
  useSensors = false;

  auto LIDs_tmp = create_mirror_view(LIDs);
  deep_copy(LIDs_tmp,LIDs);
 
  LIDs_host = LIDView_host("LIDs on host",LIDs.extent(0), LIDs.extent(1));
  deep_copy(LIDs_host,LIDs_tmp);
  
  // Compute integration data and basis functions
  if (cellData->storeAll) {
    size_type numip = cellData->ref_ip.extent(0);
    wts = View_Sc2("physical wts",numElem, numip);
    hsize = View_Sc1("physical meshsize",numElem);
    orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
    disc->getPhysicalVolumetricData(cellData, nodes, localElemID,
                                    ip, wts, hsize, orientation,
                                    basis, basis_grad, basis_curl,
                                    basis_div, basis_nodes,true,true);
    
    
    
    if (cellData->build_face_terms) {
      for (size_type side=0; side<cellData->numSides; side++) {
        int numip = cellData->ref_side_ip[side].extent(0);
        vector<View_Sc2> face_ip;
        vector<View_Sc2> face_normals;
        View_Sc2 face_wts("face wts", numElem, numip);
        View_Sc1 face_hsize("face hsize", numElem);
        vector<View_Sc4> face_basis, face_basis_grad;
                
        disc->getPhysicalFaceData(cellData, side, nodes, localElemID, orientation,
                                  face_ip, face_wts, face_normals, face_hsize,
                                  face_basis, face_basis_grad,true,false);
        
        
        ip_face.push_back(face_ip);
        wts_face.push_back(face_wts);
        normals_face.push_back(face_normals);
        hsize_face.push_back(face_hsize);
        basis_face.push_back(face_basis);
        basis_grad_face.push_back(face_basis_grad);
      }
      
    }
  }
  else {
    orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
    disc->getPhysicalOrientations(cellData, localElemID,
                                  orientation, true);
    
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::setWorkset(Teuchos::RCP<workset> & wkset_) {
  
  wkset = wkset_;

}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::setParams(LIDView paramLIDs_) {
  
  paramLIDs = paramLIDs_;
  paramLIDs_host = LIDView_host("param LIDs on host", paramLIDs.extent(0), paramLIDs.extent(1));//create_mirror_view(paramLIDs);
  deep_copy(paramLIDs_host, paramLIDs);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux basis functions at the integration points.
// This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
///////////////////////////////////////////////////////////////////////////////////////

void cell::addAuxDiscretization(const vector<basis_RCP> & abasis_pointers, const vector<DRV> & abasis,
                                const vector<DRV> & abasisGrad, const vector<vector<DRV> > & asideBasis,
                                const vector<vector<DRV> > & asideBasisGrad) {
  
  for (size_t b=0; b<abasis_pointers.size(); b++) {
    auxbasisPointers.push_back(abasis_pointers[b]);
  }
  for (size_t b=0; b<abasis.size(); b++) {
    auxbasis.push_back(abasis[b]);
    //auxbasisGrad.push_back(abasisGrad[b]);
  }
  
  for (size_t s=0; s<asideBasis.size(); s++) {
    auxside_basis.push_back(asideBasis[s]);
    //auxside_basisGrad.push_back(asideBasisGrad[s]);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux variables
///////////////////////////////////////////////////////////////////////////////////////

void cell::addAuxVars(const vector<string> & auxlist_) {
  auxlist = auxlist_;
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the regular parameters (everything but discretized)
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
  cellData->physics_RCP->updateParameters(params, paramnames);
}


///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each variable will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setUseBasis(vector<int> & usebasis_, const int & numsteps, const int & numstages) {
  vector<int> usebasis = usebasis_;
  //num_stages = nstages;
  
  // Set up the containers for usual solution storage
  int maxnbasis = 0;
  for (size_type i=0; i<cellData->numDOF_host.extent(0); i++) {
    if (cellData->numDOF_host(i) > maxnbasis) {
      maxnbasis = cellData->numDOF_host(i);
    }
  }
  //maxnbasis *= nstages;
  u = View_Sc3("u",numElem,cellData->numDOF.extent(0),maxnbasis);
  if (cellData->requiresAdjoint) {
    phi = View_Sc3("phi",numElem,cellData->numDOF.extent(0),maxnbasis);
  }
  
  // This does add a little extra un-used memory for steady-state problems, but not a concern
  if (cellData->requiresTransient) {
    u_prev = View_Sc4("u previous",numElem,cellData->numDOF.extent(0),maxnbasis,numsteps);
    u_stage = View_Sc4("u stages",numElem,cellData->numDOF.extent(0),maxnbasis,numstages);
    if (cellData->requiresAdjoint) {
      phi_prev = View_Sc4("phi previous",numElem,cellData->numDOF.extent(0),maxnbasis,numsteps);
      phi_stage = View_Sc4("phi stages",numElem,cellData->numDOF.extent(0),maxnbasis,numstages);
    }
  }
  
  if (cellData->compute_sol_avg) {
    u_avg = View_Sc3("u spatial average",numElem,cellData->numDOF.extent(0),cellData->dimension);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
  int maxnbasis = 0;
  for (size_type i=0; i<cellData->numParamDOF.extent(0); i++) {
    if (cellData->numParamDOF(i) > maxnbasis) {
      maxnbasis = cellData->numParamDOF(i);
    }
  }
  param = View_Sc3("param",numElem,cellData->numParamDOF.extent(0),maxnbasis);
  
  if (cellData->compute_sol_avg) {
    param_avg = View_Sc3("param",numElem,cellData->numParamDOF.extent(0), cellData->dimension);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void cell::setAuxUseBasis(vector<int> & ausebasis_) {
  auxusebasis = ausebasis_;
  int maxnbasis = 0;
  for (size_type i=0; i<cellData->numAuxDOF.extent(0); i++) {
    if (cellData->numAuxDOF(i) > maxnbasis) {
      maxnbasis = cellData->numAuxDOF(i);
    }
  }
  aux = View_Sc3("aux",numElem,cellData->numAuxDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateWorkset(const int & seedwhat, const bool & override_transient) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnVolTimer);
  
  // Reset the residual and data in the workset
  wkset->resetResidual();
  wkset->numElem = numElem;
  this->updateData();
  
  // Update the integration info and basis in workset
  if (cellData->storeAll) {
    wkset->wts = wts;
    wkset->h = hsize;
    wkset->setIP(ip);
    wkset->basis = basis;
    wkset->basis_grad = basis_grad;
    wkset->basis_div = basis_div;
    wkset->basis_curl = basis_curl;
  }
  else {
    vector<View_Sc2> tip;
    View_Sc2 twts("physical wts",numElem, cellData->ref_ip.extent(0));
    View_Sc1 thsize("physical meshsize",numElem);
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricData(cellData, nodes, localElemID,
                                    tip, twts, thsize, orientation,
                                    tbasis, tbasis_grad, tbasis_curl,
                                    tbasis_div, tbasis_nodes,true,false);
    wkset->wts = twts;
    wkset->h = thsize;
    wkset->setIP(tip);
    wkset->basis = tbasis;
    wkset->basis_grad = tbasis_grad;
    wkset->basis_div = tbasis_div;
    wkset->basis_curl = tbasis_curl;
  }
  
  // Map the gathered solution to seeded version in workset
  if (cellData->requiresTransient && !override_transient) {
    wkset->computeSolnTransientSeeded(u, u_prev, u_stage, seedwhat);
  }
  else { // steady-state
    wkset->computeSolnSteadySeeded(u, seedwhat);
  }
  if (wkset->numParams > 0) {
    wkset->computeParamSteadySeeded(param, seedwhat);
  }
  /* // TMW: not implemented yet
  if (wkset->numAux > 0) {
    if (cellData->requiresTransient) {
      wkset->computeAuxTransientSeeded(aux, aux_prev, aux_stage, seedwhat);
    }
    else { // steady-state
      wkset->computeAuxSteadySeeded(aux, seedwhat);
    }
  }
  */
  
  // Map the AD solutions to the aolutions at the volumetric ip
  wkset->computeSolnVolIP();
  wkset->computeParamVolIP();
  //wkset->computeAuxVolIP();
    
  if (cellData->compute_sol_avg) {
    this->computeSolAvg();
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeSolAvg() {
  
  // THIS FUNCTION ASSUMES THAT THE WORKSET BASIS HAS BEEN UPDATED
  // AND THE SOLUTION HAS BEEN COMPUTED AT THE VOLUMETRIC IP
  
  Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);

  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  auto cwts = wkset->wts;
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("cell sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });

  // HGRAD vars
  vector<int> vars_HGRAD = wkset->vars_HGRAD;
  vector<string> varlist_HGRAD = wkset->varlist_HGRAD;
  for (size_t i=0; i<vars_HGRAD.size(); ++i) {
    auto sol = wkset->getData(varlist_HGRAD[i]);
    auto savg = subview(u_avg,ALL(),vars_HGRAD[i],0);
    parallel_for("cell sol avg",
                 RangePolicy<AssemblyExec>(0,savg.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      ScalarT solavg = 0.0;
      for (size_type pt=0; pt<sol.extent(2); pt++) {
#ifndef MrHyDE_NO_AD
        solavg += sol(elem,pt).val()*cwts(elem,pt);
#else
        solavg += sol(elem,pt)*cwts(elem,pt);
#endif
      }
      savg(elem) = solavg/avgwts(elem);
    });
  }
  
  // HVOL vars
  vector<int> vars_HVOL = wkset->vars_HVOL;
  vector<string> varlist_HVOL = wkset->varlist_HVOL;
  for (size_t i=0; i<vars_HVOL.size(); ++i) {
    auto sol = wkset->getData(varlist_HVOL[i]);
    auto savg = subview(u_avg,ALL(),vars_HVOL[i],0);
    parallel_for("cell sol avg",
                 RangePolicy<AssemblyExec>(0,savg.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      ScalarT solavg = 0.0;
      for (size_type pt=0; pt<sol.extent(2); pt++) {
#ifndef MrHyDE_NO_AD
        solavg += sol(elem,pt).val()*cwts(elem,pt);
#else
        solavg += sol(elem,pt)*cwts(elem,pt);
#endif
      }
      savg(elem) = solavg/avgwts(elem);
    });
  }
  
  // Compute the postfix options for vector vars
  vector<string> postfix = {"[x]"};
  if (u_avg.extent(2) > 1) { // 2D or 3D
    postfix.push_back("[y]");
  }
  if (u_avg.extent(2) > 2) { // 3D
    postfix.push_back("[z]");
  }
  
  // HDIV vars
  vector<int> vars_HDIV = wkset->vars_HDIV;
  vector<string> varlist_HDIV = wkset->varlist_HDIV;
  for (size_t i=0; i<vars_HDIV.size(); ++i) {
    for (size_t j=0; j<postfix.size(); ++j) {
      auto sol = wkset->getData(varlist_HDIV[i]+postfix[j]);
      auto savg = subview(u_avg,ALL(),vars_HDIV[i],j);
      parallel_for("cell sol avg",
                   RangePolicy<AssemblyExec>(0,savg.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        ScalarT solavg = 0.0;
        for (size_type pt=0; pt<sol.extent(2); pt++) {
#ifndef MrHyDE_NO_AD
          solavg += sol(elem,pt).val()*cwts(elem,pt);
#else
          solavg += sol(elem,pt)*cwts(elem,pt);
#endif
        }
        savg(elem) = solavg/avgwts(elem);
      });
    }
  }
  
  // HCURL vars
  vector<int> vars_HCURL = wkset->vars_HCURL;
  vector<string> varlist_HCURL = wkset->varlist_HCURL;
  for (size_t i=0; i<vars_HCURL.size(); ++i) {
    for (size_t j=0; j<postfix.size(); ++j) {
      auto sol = wkset->getData(varlist_HCURL[i]+postfix[j]);
      auto savg = subview(u_avg,ALL(),vars_HCURL[i],j);
      parallel_for("cell sol avg",
                   RangePolicy<AssemblyExec>(0,savg.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        ScalarT solavg = 0.0;
        for (size_type pt=0; pt<sol.extent(2); pt++) {
#ifndef MrHyDE_NO_AD
          solavg += sol(elem,pt).val()*cwts(elem,pt);
#else
          solavg += sol(elem,pt)*cwts(elem,pt);
#endif
        }
        savg(elem) = solavg/avgwts(elem);
      });
    }
  }
  
  /*
  if (param_avg.extent(1) > 0) {
    View_AD4 psol = wkset->local_param;
    auto pavg = param_avg;

    parallel_for("cell param avg",
                 RangePolicy<AssemblyExec>(0,pavg.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      ScalarT avgwt = 0.0;
      for (size_type pt=0; pt<cwts.extent(1); pt++) {
        avgwt += cwts(elem,pt);
      }
      for (size_type dof=0; dof<psol.extent(1); dof++) {
        for (size_type dim=0; dim<psol.extent(3); dim++) {
          ScalarT solavg = 0.0;
          for (size_type pt=0; pt<psol.extent(2); pt++) {
            solavg += psol(elem,dof,pt,dim).val()*cwts(elem,pt);
          }
          pavg(elem,dof,dim) = solavg/avgwt;
        }
      }
    });
  }
   */
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateWorksetFace(const size_t & facenum) {
  
  // IMPORANT NOTE: This function assumes that face contributions are computing IMMEDIATELY after the
  // volumetric contributions, which implies that the seeded solution in the workset is already
  // correct for this cell.  There is currently no use case where this assumption is false.
  
  Teuchos::TimeMonitor localtimer(*computeSolnFaceTimer);
  
  // Update the face integration points and basis in workset
  if (cellData->storeAll) {
    wkset->wts_side = wts_face[facenum];
    wkset->h = hsize;
    wkset->setIP(ip_face[facenum]," side");
    wkset->setNormals(normals_face[facenum]);
    wkset->basis_side = basis_face[facenum];
    wkset->basis_grad_side = basis_grad_face[facenum];
  }
  else {
    int numip = cellData->ref_side_ip[facenum].extent(0);
    vector<View_Sc2> tip;
    vector<View_Sc2> tnormals;
    View_Sc2 twts("face wts", numElem, numip);
    View_Sc1 thsize("face hsize", numElem);
    vector<View_Sc4> tbasis, tbasis_grad;
  
    disc->getPhysicalFaceData(cellData, facenum, nodes, localElemID, orientation,
                              tip, twts, tnormals, thsize, tbasis, tbasis_grad, true, false);
    
    wkset->wts_side = twts;
    wkset->h = thsize;
    wkset->setIP(tip," side");
    wkset->setNormals(tnormals);
    wkset->basis_side = tbasis;
    wkset->basis_grad_side = tbasis_grad;
  }
  
  // Map the seeded solution in workset to solution at face ip
  wkset->computeSolnSideIP();
  wkset->computeParamSideIP();
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeAuxSolnFaceIP(const size_t & facenum) {

  Teuchos::TimeMonitor localtimer(*computeSolnFaceTimer);
  this->updateWorksetFace(facenum);

}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous step solutions
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetPrevSoln() {
  
  auto sol = u;
  auto sol_prev = u_prev;
  
  // shift previous step solns
  if (sol_prev.extent(3)>1) {
    parallel_for("cell reset prev soln 1",
                 TeamPolicy<AssemblyExec>(sol_prev.extent(0), Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type i=team.team_rank(); i<sol_prev.extent(1); i+=team.team_size() ) {
        for (size_type j=0; j<sol_prev.extent(2); j++) {
          for (size_type s=sol_prev.extent(3)-1; s>0; s--) {
            sol_prev(elem,i,j,s) = sol_prev(elem,i,j,s-1);
          }
        }
      }
    });
  }
  
  // copy current u into first step
  parallel_for("cell reset prev soln 2",
               TeamPolicy<AssemblyExec>(sol_prev.extent(0), Kokkos::AUTO, VectorSize),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type i=team.team_rank(); i<sol_prev.extent(1); i+=team.team_size() ) {
      for (size_type j=0; j<sol.extent(2); j++) {
        sol_prev(elem,i,j,0) = sol(elem,i,j);
      }
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetStageSoln() {
  
  auto sol = u;
  auto sol_stage = u_stage;
  
  parallel_for("cell reset stage 1",
               TeamPolicy<AssemblyExec>(sol_stage.extent(0), Kokkos::AUTO, VectorSize),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type i=team.team_rank(); i<sol_stage.extent(1); i+=team.team_size() ) {
      for (size_type j=0; j<sol_stage.extent(2); j++) {
        for (size_type k=0; k<sol_stage.extent(3); k++) {
          sol_stage(elem,i,j,k) = sol(elem,i,j);
        }
      }
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateStageSoln() {
  
  auto sol = u;
  auto sol_stage = u_stage;
  
  // add u into the current stage soln (done after stage solution is computed)
  auto stage = wkset->current_stage;
  parallel_for("wkset transient sol seedwhat 1",
               TeamPolicy<AssemblyExec>(sol_stage.extent(0), Kokkos::AUTO, VectorSize),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    //int stage = snum(0);
    for (size_type i=team.team_rank(); i<sol_stage.extent(1); i+=team.team_size() ) {
      for (size_type j=0; j<sol_stage.extent(2); j++) {
        sol_stage(elem,i,j,stage) = sol(elem,i,j);
      }
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the contribution from this cell to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

void cell::computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                         const bool & compute_jacobian, const bool & compute_sens,
                         const int & num_active_params, const bool & compute_disc_sens,
                         const bool & compute_aux_sens, const bool & store_adjPrev,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_res,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                         const bool & assemble_volume_terms,
                         const bool & assemble_face_terms) {
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  bool fixJacDiag = false;
  
  //wkset->resetResidual();
  
  //////////////////////////////////////////////////////////////
  // Compute the AD-seeded solutions at integration points
  //////////////////////////////////////////////////////////////
  
  int seedwhat = 0;
  if (compute_jacobian) {
    if (compute_disc_sens) {
      seedwhat = 3;
    }
    else if (compute_aux_sens) {
      seedwhat = 4;
    }
    else {
      seedwhat = 1;
    }
  }
  
  //////////////////////////////////////////////////////////////
  // Compute res and J=dF/du
  //////////////////////////////////////////////////////////////
  
  // Volumetric contribution
  if (assemble_volume_terms) {
    Teuchos::TimeMonitor localtimer(*volumeResidualTimer);
    if (cellData->multiscale) {
      this->updateWorkset(seedwhat);
      int sgindex = subgrid_model_index[subgrid_model_index.size()-1];
      subgridModels[sgindex]->subgridSolver(u, phi, wkset->time, isTransient, isAdjoint,
                                            compute_jacobian, compute_sens, num_active_params,
                                            compute_disc_sens, compute_aux_sens,
                                            *wkset, subgrid_usernum, 0,
                                            subgradient, store_adjPrev);
      fixJacDiag = true;
    }
    else {
      this->updateWorkset(seedwhat);
      cellData->physics_RCP->volumeResidual(cellData->myBlock);
    }
  }
  
  // Edge/face contribution
  if (assemble_face_terms) {
    Teuchos::TimeMonitor localtimer(*faceResidualTimer);
    if (cellData->multiscale) {
      // do nothing
    }
    else {
      for (size_t s=0; s<cellData->numSides; s++) {
        this->updateWorksetFace(s);
        cellData->physics_RCP->faceResidual(cellData->myBlock);
      }
    }
  }
  
  {
    Teuchos::TimeMonitor localtimer(*jacobianFillTimer);
    
    // Use AD residual to update local Jacobian
    if (compute_jacobian) {
      if (compute_disc_sens) {
        this->updateParamJac(local_J);
      }
      else if (compute_aux_sens){
        this->updateAuxJac(local_J);
      }
      else {
        this->updateJac(isAdjoint, local_J);
      }
    }
  }
  
  if (compute_jacobian && fixJacDiag) {
    this->fixDiagJac(local_J, local_res);
  }
  
  // Update the local residual
  {
    if (isAdjoint) {
      Teuchos::TimeMonitor localtimer(*adjointResidualTimer);
      this->updateAdjointRes(compute_jacobian, isTransient,
                             compute_aux_sens, store_adjPrev,
                             local_J, local_res);
    }
    else {
      Teuchos::TimeMonitor localtimer(*residualFillTimer);
      this->updateRes(compute_sens, local_res);
    }
  }
    
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateRes(const bool & compute_sens, View_Sc3 local_res) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (compute_sens) {
#ifndef MrHyDE_NO_AD
    parallel_for("cell res sens",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type r=0; r<local_res.extent(2); r++) {
            local_res(elem,offsets(n,j),r) -= res_AD(elem,offsets(n,j)).fastAccessDx(r);
          }
        }
      }
    });
#endif
  }
  else {
    parallel_for("cell res",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
#ifndef MrHyDE_NO_AD
          local_res(elem,offsets(n,j),0) -= res_AD(elem,offsets(n,j)).val();
#else
          local_res(elem,offsets(n,j),0) -= res_AD(elem,offsets(n,j));
#endif
        }
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAdjointRes(const bool & compute_jacobian, const bool & isTransient,
                            const bool & compute_aux_sens, const bool & store_adjPrev,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  
  // Update residual (adjoint mode)
  // Adjoint residual: -dobj/du - J^T * phi + 1/dt*M^T * phi_prev
  // J = 1/dtM + A
  // adj_prev stores 1/dt*M^T * phi_prev where M is evaluated at appropriate time
  
  // TMW: This will not work on a GPU
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  
  if (compute_jacobian) {
    parallel_for("cell adjust adjoint jac",
                 RangePolicy<AssemblyExec>(0,local_res.extent(0)),
                 KOKKOS_LAMBDA (const size_type e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_res(e,offsets(n,j),0) += -local_J(e,offsets(n,j),offsets(m,k))*phi(e,m,k);
            }
          }
        }
      }
    });
    
    if (isTransient) {
      
      // Previous step contributions for the residual
      parallel_for("cell adjust transient adjoint jac",
                   RangePolicy<AssemblyExec>(0,local_res.extent(0)),
                   KOKKOS_LAMBDA (const size_type e ) {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),0) += -adj_prev(e,offsets(n,j),0);
          }
        }
      });
      /*
      // Previous stage contributions for the residual
      if (adj_stage_prev.extent(2) > 0) {
        parallel_for("cell adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
          for (size_type n=0; n<numDOF.extent(0); n++) {
            for (int j=0; j<numDOF(n); j++) {
              local_res(e,offsets(n,j),0) += -adj_stage_prev(e,offsets(n,j),0);
            }
          }
        });
      }
      */
      
      if (!compute_aux_sens && store_adjPrev) {
        
        //////////////////////////////////////////
        // Multi-step
        //////////////////////////////////////////
        
        // Move vectors up
        parallel_for("cell adjust transient adjoint jac",
                     RangePolicy<AssemblyExec>(0,adj_prev.extent(0)),
                     KOKKOS_LAMBDA (const size_type e ) {
          for (size_type step=1; step<adj_prev.extent(2); step++) {
            for (size_type n=0; n<adj_prev.extent(1); n++) {
              adj_prev(e,n,step-1) = adj_prev(e,n,step);
            }
          }
          size_type numsteps = adj_prev.extent(2);
          for (size_type n=0; n<adj_prev.extent(1); n++) {
            adj_prev(e,n,numsteps-1) = 0.0;
          }
        });
        
        // Sum new contributions into vectors
        int seedwhat = 2; // 2 for J wrt previous step solutions
        for (size_type step=0; step<u_prev.extent(3); step++) {
          this->updateWorkset(seedwhat);
          cellData->physics_RCP->volumeResidual(cellData->myBlock);
          Kokkos::View<ScalarT***,AssemblyDevice> Jdot("temporary fix for transient adjoint",
                                                       local_J.extent(0), local_J.extent(1), local_J.extent(2));
          this->updateJac(true, Jdot);
          
          auto cadj = subview(adj_prev,ALL(), ALL(), step);
          parallel_for("cell adjust transient adjoint jac 2",RangePolicy<AssemblyExec>(0,Jdot.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for (int j=0; j<numDOF(n); j++) {
                ScalarT aPrev = 0.0;
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    aPrev += Jdot(e,offsets(n,j),offsets(m,k))*phi(e,m,k);
                  }
                }
                cadj(e,offsets(n,j)) += aPrev;
              }
            }
          });
        }
        
        //////////////////////////////////////////
        // Multi-stage
        //////////////////////////////////////////
        /*
        // Move vectors up
        parallel_for("cell adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,adj_stage_prev.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
          for (size_type stage=1; stage<adj_stage_prev.extent(2); stage++) {
            for (size_type n=0; n<adj_stage_prev.extent(1); n++) {
              adj_stage_prev(e,n,stage-1) = adj_stage_prev(e,n,stage);
            }
          }
          size_type numstages = adj_stage_prev.extent(2);
          for (size_type n=0; n<adj_stage_prev.extent(1); n++) {
            adj_stage_prev(e,n,numstages-1) = 0.0;
          }
        });
        
        // Sum new contributions into vectors
        seedwhat = 3; // 3 for J wrt previous stage solutions
        for (size_type stage=0; stage<u_prev.extent(3); stage++) {
          wkset->computeSolnTransientSeeded(u, u_prev, u_stage, seedwhat, stage);
          wkset->computeParamVolIP(param, seedwhat);
          this->computeSolnVolIP();
          
          wkset->resetResidual();
          
          cellData->physics_RCP->volumeResidual(cellData->myBlock);
          Kokkos::View<ScalarT***,AssemblyDevice> Jdot("temporary fix for transient adjoint",
                                                       local_J.extent(0), local_J.extent(1), local_J.extent(2));
          this->updateJac(true, Jdot);
          
          auto cadj = subview(adj_stage_prev,ALL(), ALL(), stage);
          parallel_for("cell adjust transient adjoint jac 2",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for (int j=0; j<numDOF(n); j++) {
                ScalarT aPrev = 0.0;
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    aPrev += Jdot(e,offsets(n,j),offsets(m,k))*phi(e,m,k);
                  }
                }
                cadj(e,offsets(n,j)) += aPrev;
              }
            }
          });
        }*/
      }
    }
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (useadjoint) {
    parallel_for("cell J adj",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(m,k),offsets(n,j)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("cell J",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(n,j),offsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  //AssemblyExec::execution_space().fence();
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Place ones on the diagonal of the Jacobian if
///////////////////////////////////////////////////////////////////////////////////////

void cell::fixDiagJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                      Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;

  using namespace std;

  parallel_for("cell fix diag",
               RangePolicy<AssemblyExec>(0,local_J.extent(0)), 
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT JTOL = 1.0E-14;
    for (size_type var=0; var<offsets.extent(0); var++) {
      for (int dof=0; dof<numDOF(var); dof++) {
        int diag = offsets(var,dof);
        if (abs(local_J(elem,diag,diag)) < JTOL) {
          local_res(elem,diag,0) = 0.0;//-u(elem,var,dof);
          for (int j=0; j<numDOF(var); j++) {
            ScalarT scale = 1.0/((ScalarT)numDOF(var)-1.0);
            local_J(elem,diag,offsets(var,j)) = -scale;
            //if (j!=dof)
            //  local_res(elem,diag,0) += scale*u(elem,var,j);
          }
          local_J(elem,diag,diag) = 1.0;
        }
      }
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateParamJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
#ifndef MrHyDE_NO_AD
  auto paramoffsets = wkset->paramoffsets;
  auto numParamDOF = cellData->numParamDOF;
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  parallel_for("cell param J",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (size_type m=0; m<numParamDOF.extent(0); m++) {
          for (int k=0; k<numParamDOF(m); k++) {
            local_J(elem,offsets(n,j),paramoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(paramoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jaux
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateAuxJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto aoffsets = auxoffsets;
  auto numDOF = cellData->numDOF;
  auto numAuxDOF = cellData->numAuxDOF;
  
  parallel_for("cell aux J",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (size_type m=0; m<numAuxDOF.extent(0); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(elem,offsets(n,j),auxoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(aoffsets(m,k));
          }
        }
      }
    }
  });
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the initial condition
///////////////////////////////////////////////////////////////////////////////////////

View_Sc2 cell::getInitial(const bool & project, const bool & isAdjoint) {
  
  View_Sc2 initialvals("initial values",numElem,LIDs.extent(1));
  this->updateWorkset(0);
  
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto cwts = wts;
  
  if (project) { // works for any basis
    auto initialip = cellData->physics_RCP->getInitial(ip,
                                                       cellData->myBlock,
                                                       project,
                                                       wkset);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = basis[wkset->usebasis[n]];
      auto off = subview(offsets, n, ALL());
      auto initvar = subview(initialip, ALL(), n, ALL());
      parallel_for("cell init project",
                   TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<cbasis.extent(1); dof+=team.team_size() ) {
          for(size_type pt=0; pt<cwts.extent(1); pt++ ) {
            initialvals(elem,off(dof)) += initvar(elem,pt)*cbasis(elem,dof,pt,0)*cwts(elem,pt);
          }
        }
      });
    }
  }
  else { // only works if using HGRAD linear basis
    vector<View_Sc2> vnodes;
    View_Sc2 vx,vy,vz;
    vx = View_Sc2("view of nodes",nodes.extent(0),nodes.extent(1));
    auto n_x = subview(nodes,ALL(),ALL(),0);
    deep_copy(vx,n_x);
    vnodes.push_back(vx);
    if (nodes.extent(2) > 1) {
      vy = View_Sc2("view of nodes",nodes.extent(0),nodes.extent(1));
      auto n_y = subview(nodes,ALL(),ALL(),1);
      deep_copy(vy,n_y);
      vnodes.push_back(vy);
    }
    if (nodes.extent(2) > 2) {
      vz = View_Sc2("view of nodes",nodes.extent(0),nodes.extent(1));
      auto n_z = subview(nodes,ALL(),ALL(),2);
      deep_copy(vz,n_z);
      vnodes.push_back(vz);
    }
    
    auto initialnodes = cellData->physics_RCP->getInitial(vnodes,
                                                          cellData->myBlock,
                                                          project,
                                                          wkset);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto off = subview( offsets, n, ALL());
      auto initvar = subview(initialnodes, ALL(), n, ALL());
      parallel_for("cell init project",
                   TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<initvar.extent(1); dof+=team.team_size() ) {
          initialvals(elem,off(dof)) = initvar(elem,dof);
        }
      });
    }
  }
  return initialvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

View_Sc3 cell::getMass() {
  
  View_Sc3 mass("local mass",numElem, LIDs.extent(1), LIDs.extent(1));
  
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto cwts = wts;
  for (size_type n=0; n<numDOF.extent(0); n++) {
    View_Sc4 cbasis;
    if (cellData->storeAll) {
      cbasis = basis[wkset->usebasis[n]];
    }
    else { // goes through this more than once, but really shouldn't be used much anyways
      vector<View_Sc2> tip;
      View_Sc2 twts("physical wts",numElem, cellData->ref_ip.extent(0));
      View_Sc1 thsize("physical meshsize",numElem);
      vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
      vector<View_Sc3> tbasis_div;
      disc->getPhysicalVolumetricData(cellData, nodes, localElemID,
                                      tip, twts, thsize, orientation,
                                      tbasis, tbasis_grad, tbasis_curl,
                                      tbasis_div, tbasis_nodes,true,false);
      cbasis = tbasis[wkset->usebasis[n]];
    }
      
    string btype = wkset->basis_types[wkset->usebasis[n]];
    auto off = subview(offsets,n,ALL());
    if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
      parallel_for("cell get mass",
                   RangePolicy<AssemblyExec>(0,mass.extent(0)),
                   KOKKOS_LAMBDA (const size_type e ) {
        for(size_type i=0; i<cbasis.extent(1); i++ ) {
          for(size_type j=0; j<cbasis.extent(1); j++ ) {
            for(size_type k=0; k<cbasis.extent(2); k++ ) {
              mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k);
            }
          }
        }
      });
    }
    else if (btype.substr(0,4) == "HDIV" || btype.substr(0,5) == "HCURL") {
      parallel_for("cell get mass",
                   RangePolicy<AssemblyExec>(0,mass.extent(0)),
                   KOKKOS_LAMBDA (const size_type e ) {
        for (size_type i=0; i<cbasis.extent(1); i++ ) {
          for (size_type j=0; j<cbasis.extent(1); j++ ) {
            for (size_type k=0; k<cbasis.extent(2); k++ ) {
              for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k);
              }
            }
          }
        }
      });
    }
  }
  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get a weighted mass matrix
///////////////////////////////////////////////////////////////////////////////////////

View_Sc3 cell::getWeightedMass(vector<ScalarT> & masswts) {
  
  View_Sc3 mass("local mass",numElem, LIDs.extent(1), LIDs.extent(1));
  
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto cwts = wts;
  for (size_type n=0; n<numDOF.extent(0); n++) {
    View_Sc4 cbasis;
    if (cellData->storeAll) {
      cbasis = basis[wkset->usebasis[n]];
    }
    else { // goes through this more than once, but really shouldn't be used much anyways
      vector<View_Sc2> tip;
      View_Sc2 twts("physical wts",numElem, cellData->ref_ip.extent(0));
      View_Sc1 thsize("physical meshsize",numElem);
      vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
      vector<View_Sc3> tbasis_div;
      disc->getPhysicalVolumetricData(cellData, nodes, localElemID,
                                      tip, twts, thsize, orientation,
                                      tbasis, tbasis_grad, tbasis_curl,
                                      tbasis_div, tbasis_nodes,true,false);
      cbasis = tbasis[wkset->usebasis[n]];
    }
      
    string btype = wkset->basis_types[wkset->usebasis[n]];
    auto off = subview(offsets,n,ALL());
    ScalarT mwt = masswts[n];
    
    if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
      parallel_for("cell get mass",
                   RangePolicy<AssemblyExec>(0,mass.extent(0)),
                   KOKKOS_LAMBDA (const size_type e ) {
        for(size_type i=0; i<cbasis.extent(1); i++ ) {
          for(size_type j=0; j<cbasis.extent(1); j++ ) {
            for(size_type k=0; k<cbasis.extent(2); k++ ) {
              mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k)*mwt;
            }
          }
        }
      });
    }
    else if (btype.substr(0,4) == "HDIV" || btype.substr(0,5) == "HCURL") {
      parallel_for("cell get mass",
                   RangePolicy<AssemblyExec>(0,mass.extent(0)),
                   KOKKOS_LAMBDA (const size_type e ) {
        for (size_type i=0; i<cbasis.extent(1); i++ ) {
          for (size_type j=0; j<cbasis.extent(1); j++ ) {
            for (size_type k=0; k<cbasis.extent(2); k++ ) {
              for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,dim)*cbasis(e,j,k,dim)*cwts(e,k)*mwt;
              }
            }
          }
        }
      });
    }
  }
  return mass;
}


///////////////////////////////////////////////////////////////////////////////////////
// Subgrid Plotting
///////////////////////////////////////////////////////////////////////////////////////

void cell::writeSubgridSolution(const std::string & filename) {
  //if (multiscale) {
  //  subgridModel->writeSolution(filename, subgrid_usernum);
  //}
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the subgrid model
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateSubgridModel(vector<Teuchos::RCP<SubGridModel> > & models) {
  
  /*
   wkset->update(ip,jacobian);
   int newmodel = udfunc->getSubgridModel(nodes, wkset, models.size());
   if (newmodel != subgrid_model_index) {
   // then we need:
   // 1. To add the macro-element to the new model
   // 2. Project the most recent solutions onto the new model grid
   // 3. Update this cell to use the new model
   
   // Step 1:
   int newusernum = models[newmodel]->addMacro(nodes, sideinfo, sidenames,
   GIDs, index);
   
   // Step 2:
   
   // Step 3:
   subgridModel = models[newmodel];
   subgrid_model_index = newmodel;
   subgrid_usernum = newusernum;
   
   
   }*/
}

///////////////////////////////////////////////////////////////////////////////////////
// Pass the cell data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void cell::updateData() {
  
  // hard coded for what I need it for right now
  if (cellData->have_cell_phi) {
    wkset->have_rotation_phi = true;
    wkset->rotation_phi = cell_data;
  }
  else if (cellData->have_cell_rotation) {
    wkset->have_rotation = true;
    auto rot = wkset->rotation;
    parallel_for("cell update data",
                 RangePolicy<AssemblyExec>(0,cell_data.extent(0)),
                 KOKKOS_LAMBDA (const size_type e ) {
      rot(e,0,0) = cell_data(e,0);
      rot(e,0,1) = cell_data(e,1);
      rot(e,0,2) = cell_data(e,2);
      rot(e,1,0) = cell_data(e,3);
      rot(e,1,1) = cell_data(e,4);
      rot(e,1,2) = cell_data(e,5);
      rot(e,2,0) = cell_data(e,6);
      rot(e,2,1) = cell_data(e,7);
      rot(e,2,2) = cell_data(e,8);
    });
    /*
     for (int e=0; e<numElem; e++) {
     rotmat(e,0,0) = cell_data(e,0);
     rotmat(e,0,1) = cell_data(e,1);
     rotmat(e,0,2) = cell_data(e,2);
     rotmat(e,1,0) = cell_data(e,3);
     rotmat(e,1,1) = cell_data(e,4);
     rotmat(e,1,2) = cell_data(e,5);
     rotmat(e,2,0) = cell_data(e,6);
     rotmat(e,2,1) = cell_data(e,7);
     rotmat(e,2,2) = cell_data(e,8);
     }*/
    //wkset->rotation = rotmat;
  }
  else if (cellData->have_extra_data) {
    wkset->extra_data = cell_data;
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Pass the cell data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void cell::resetAdjPrev(const ScalarT & val) {
  deep_copy(adj_prev,val);
}


///////////////////////////////////////////////////////////////////////////////////////
// Get the solution at the nodes
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT***,AssemblyDevice> cell::getSolutionAtNodes(const int & var) {
  
  Teuchos::TimeMonitor nodesoltimer(*computeNodeSolTimer);
  
  int bnum = wkset->usebasis[var];
  auto cbasis = basis_nodes[bnum];
  Kokkos::View<ScalarT***,AssemblyDevice> nodesol("solution at nodes",
                                                  cbasis.extent(0), cbasis.extent(2), cellData->dimension);
  auto uvals = subview(u, ALL(), var, ALL());
  parallel_for("cell node sol",
               RangePolicy<AssemblyExec>(0,cbasis.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
      ScalarT uval = uvals(elem,dof);
      for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
        for (size_type s=0; s<cbasis.extent(3); s++ ) {
          nodesol(elem,pt,s) += uval*cbasis(elem,dof,pt,s);
        }
      }
    }
  });
  
  return nodesol;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the storage required for the integration/basis info
///////////////////////////////////////////////////////////////////////////////////////

size_t cell::getVolumetricStorage() {
  size_t mystorage = 0;
  if (cellData->storeAll) {
    size_t scalarcost = 8; // 8 bytes per double
    for (size_t k=0; k<ip.size(); ++k) {
      mystorage += scalarcost*ip[k].size();
    }
    mystorage += scalarcost*wts.size();
    for (size_t k=0; k<basis.size(); ++k) {
      mystorage += scalarcost*basis[k].size();
    }
    for (size_t k=0; k<basis_grad.size(); ++k) {
      mystorage += scalarcost*basis_grad[k].size();
    }
    for (size_t k=0; k<basis_curl.size(); ++k) {
      mystorage += scalarcost*basis_curl[k].size();
    }
    for (size_t k=0; k<basis_div.size(); ++k) {
      mystorage += scalarcost*basis_div[k].size();
    }
    for (size_t k=0; k<basis_nodes.size(); ++k) {
      mystorage += scalarcost*basis_nodes[k].size();
    }
  }
  return mystorage;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the storage required for the face integration/basis info
///////////////////////////////////////////////////////////////////////////////////////

size_t cell::getFaceStorage() {
  size_t mystorage = 0;
  if (cellData->storeAll) {
    size_t scalarcost = 8; // 8 bytes per double
    for (size_t f=0; f<ip_face.size(); ++f) {
      for (size_t k=0; k<ip_face[f].size(); ++k) {
        mystorage += scalarcost*ip_face[f][k].size();
      }
    }
    for (size_t f=0; f<normals_face.size(); ++f) {
      for (size_t k=0; k<normals_face[f].size(); ++k) {
        mystorage += scalarcost*normals_face[f][k].size();
      }
    }
    for (size_t f=0; f<wts_face.size(); ++f) {
      mystorage += scalarcost*wts_face[f].size();
    }
    for (size_t f=0; f<basis_face.size(); ++f) {
      for (size_t k=0; k<basis_face[f].size(); ++k) {
        mystorage += scalarcost*basis_face[f][k].size();
      }
    }
    for (size_t f=0; f<basis_grad_face.size(); ++f) {
      for (size_t k=0; k<basis_grad_face[f].size(); ++k) {
        mystorage += scalarcost*basis_grad_face[f][k].size();
      }
    }
  }
  return mystorage;
}
