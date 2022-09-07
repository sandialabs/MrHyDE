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

#include "boundaryGroup.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

BoundaryGroup::BoundaryGroup(const Teuchos::RCP<GroupMetaData> & groupData_,
                             const DRV nodes_,
                             const Kokkos::View<LO*,AssemblyDevice> localID_,
                             LO & sideID_,
                             const int & sidenum_, const string & sidename_,
                             const int & groupID_,
                             Teuchos::RCP<DiscretizationInterface> & disc_,
                             const bool & storeAll_) :
groupData(groupData_), localElemID(localID_), localSideID(sideID_),
sidenum(sidenum_), groupID(groupID_), nodes(nodes_), 
sidename(sidename_), disc(disc_)   {
  
  numElem = nodes.extent(0);
  
  storeAll = storeAll_;
  
  haveBasis = false;
  
  // Orientations are always stored
  orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
  disc->getPhysicalOrientations(groupData, localElemID, orientation, false);
  
  // Integration points, weights, normals, tangents and element sizes are always stored
  int numip = groupData->ref_side_ip[0].extent(0);
  wts = View_Sc2("physical wts",numElem, numip);
  hsize = View_Sc1("physical meshsize",numElem);
  
  disc->getPhysicalBoundaryIntegrationData(groupData, nodes, localSideID, ip,
                                           wts, normals, tangents);
  
  this->computeSize();
  this->initializeBasisIndex();
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::computeSize() {

  size_t dimension = groupData->dimension;

  parallel_for("compute hsize",
               RangePolicy<AssemblyExec>(0,wts.extent(0)),
               KOKKOS_LAMBDA (const int e ) {
    ScalarT vol = 0.0;
    for (size_type i=0; i<wts.extent(1); i++) {
      vol += wts(e,i);
    }
    ScalarT dimscl = 1.0/((ScalarT)dimension-1.0);
    hsize(e) = std::pow(vol,dimscl);
  });
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::initializeBasisIndex() {
  
  basis_index = Kokkos::View<LO*,AssemblyDevice>("basis index",numElem);
  parallel_for("compute hsize",
               RangePolicy<AssemblyExec>(0,basis_index.extent(0)),
               KOKKOS_LAMBDA (const int e ) {
    basis_index(e) = e;
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::computeBasis(const bool & keepnodes) {
  
  if (storeAll && !haveBasis) {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl;
    vector<View_Sc3> tbasis_div;
    
    disc->getPhysicalBoundaryBasis(groupData, nodes, localSideID, orientation,
                                   tbasis, tbasis_grad, tbasis_curl, tbasis_div);
    for (size_t i=0; i<tbasis.size(); ++i) {
      basis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      basis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
      basis_div.push_back(CompressedView<View_Sc3>(tbasis_div[i]));
      basis_curl.push_back(CompressedView<View_Sc4>(tbasis_curl[i]));
    }
    haveBasis = true;
    if (!keepnodes) {
      nodes = DRV("dummy nodes",1);
    }
  }
  else if (groupData->use_basis_database) {
    for (size_t i=0; i<groupData->database_side_basis.size(); ++i) {
      basis.push_back(CompressedView<View_Sc4>(groupData->database_side_basis[i],basis_index));
      basis_grad.push_back(CompressedView<View_Sc4>(groupData->database_side_basis_grad[i],basis_index));
    }
    
    if (!keepnodes) {
      nodes = DRV("empty nodes",1);
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::createHostLIDs() {
  //bool data_avail = true;
  //if (!Kokkos::SpaceAccessibility<HostExec, AssemblyDevice::memory_space>::accessible) {
  //  data_avail = false;
  //}
  
  LIDs_host = vector<LIDView_host>(LIDs.size());
  for (size_t set=0; set<LIDs.size(); ++set) {
    //if (data_avail) {
    //  LIDs_host[set] = LIDs[set];
    //}
    //else {
      auto LIDs_tmp = Kokkos::create_mirror_view(LIDs[set]);
      Kokkos::deep_copy(LIDs_tmp,LIDs[set]);
      LIDView_host currLIDs_host("LIDs on host",LIDs[set].extent(0), LIDs[set].extent(1));
      Kokkos::deep_copy(currLIDs_host,LIDs_tmp);
      LIDs_host[set] = currLIDs_host;
    //}
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::setWorkset(Teuchos::RCP<workset> & wkset_) {
  
  wkset = wkset_;
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::setParams(LIDView paramLIDs_) {
  paramLIDs = paramLIDs_;
  paramLIDs_host = LIDView_host("param LIDs on host", paramLIDs.extent(0), paramLIDs.extent(1));
  Kokkos::deep_copy(paramLIDs_host, paramLIDs);
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux basis functions at the integration points.
// This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::addAuxDiscretization(const vector<basis_RCP> & abasis_pointers,
                                        const vector<DRV> & asideBasis,
                                        const vector<DRV> & asideBasisGrad) {
  
  auxbasisPointers = abasis_pointers;
  auxside_basis = asideBasis;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux variables
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::addAuxVars(const vector<string> & auxlist_) {
  auxlist = auxlist_;
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each variable will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::setUseBasis(vector<vector<int> > & usebasis_, const vector<int> & maxnumsteps, const vector<int> & maxnumstages) {
  vector<vector<int> > usebasis = usebasis_;
  
  // Set up the containers for usual solution storage
  for (size_t set=0; set<usebasis.size(); ++set) {
    int maxnbasis = 0;
    for (size_type i=0; i<groupData->set_numDOF_host[set].extent(0); i++) {
      if (groupData->set_numDOF_host[set](i) > maxnbasis) {
        maxnbasis = groupData->set_numDOF_host[set](i);
      }
    }
    View_Sc3 newu("u bgrp",numElem,groupData->set_numDOF[set].extent(0),maxnbasis);
    u.push_back(newu);
    if (groupData->requiresAdjoint) {
      View_Sc3 newphi("phi bgrp",numElem,groupData->set_numDOF[set].extent(0),maxnbasis);
      phi.push_back(newphi);
    }
    if (groupData->requiresTransient) {
      View_Sc4 newuprev("u previous bgrp",numElem,groupData->set_numDOF[set].extent(0),maxnbasis,maxnumsteps[set]);
      u_prev.push_back(newuprev);
      View_Sc4 newustage("u stages bgrp",numElem,groupData->set_numDOF[set].extent(0),maxnbasis,maxnumstages[set]-1);
      u_stage.push_back(newustage);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
  auto numParamDOF = groupData->numParamDOF;
  
  int maxnbasis = 0;
  for (size_type i=0; i<numParamDOF.extent(0); i++) {
    if (numParamDOF(i) > maxnbasis) {
      maxnbasis = numParamDOF(i);
    }
  }
  param = View_Sc3("param",numElem,numParamDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::setAuxUseBasis(vector<int> & ausebasis_) {
  auxusebasis = ausebasis_;
  auto numAuxDOF = Kokkos::create_mirror_view(groupData->numAuxDOF);
  Kokkos::deep_copy(numAuxDOF,groupData->numAuxDOF);
  int maxnbasis = 0;
  for (size_type i=0; i<numAuxDOF.extent(0); i++) {
    if (numAuxDOF(i) > maxnbasis) {
      maxnbasis = numAuxDOF(i);
    }
  }
  aux = View_Sc3("aux",numElem,numAuxDOF.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the workset
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::updateWorkset(const int & seedwhat, const int & seedindex,
                                  const bool & override_transient) {
  
  // Reset the residual and data in the workset
  wkset->reset();
  wkset->sidename = sidename;
  wkset->currentside = sidenum;
  wkset->numElem = numElem;

  this->updateData();
  
  // Update the integration info and basis in workset
  this->updateWorksetBasis();
  // Map the gathered solution to seeded version in workset
  if (groupData->requiresTransient && !override_transient) {
    for (size_t set=0; set<groupData->numSets; ++set) {
      wkset->computeSolnTransientSeeded(set, u[set], u_prev[set], u_stage[set], seedwhat, seedindex);
    }
  }
  else { // steady-state
    for (size_t set=0; set<groupData->numSets; ++set) {
      wkset->computeSolnSteadySeeded(set, u[set], seedwhat);
    }
  }
  if (wkset->numParams > 0) {
    wkset->computeParamSteadySeeded(param, seedwhat);
  }

  // Map the AD solutions to the aolutions at the boundary ip
  this->computeSoln(seedwhat);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the workset basis and ip
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::updateWorksetBasis() {

  wkset->numElem = numElem;
  
  wkset->wts_side = wts;
  wkset->h = hsize;
  //wkset->basis_index = basis_index;

  wkset->setScalarField(ip[0],"x");
  wkset->setScalarField(normals[0],"n[x]");
  wkset->setScalarField(tangents[0],"t[x]");
  if (ip.size() > 1) {
    wkset->setScalarField(ip[1],"y");
    wkset->setScalarField(normals[1],"n[y]");
    wkset->setScalarField(tangents[1],"t[y]");
  }
  if (ip.size() > 2) {
    wkset->setScalarField(ip[2],"z");
    wkset->setScalarField(normals[2],"n[z]");
    wkset->setScalarField(tangents[2],"t[z]");
  }

  if (storeAll || groupData->use_basis_database) {
    wkset->basis_side = basis;
    wkset->basis_grad_side = basis_grad;
  }
  //else if (groupData->use_basis_database) {
  //  //disc->copySideBasisFromDatabase(groupData, basis_database_index, orientation, false, false);
  //  wkset->basis_side = groupData->database_side_basis;//physical_side_basis;
  //  wkset->basis_grad_side = groupData->database_side_basis_grad;//physical_side_basis_grad;
  //}
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalBoundaryBasis(groupData, nodes, localSideID, orientation,
                                   tbasis, tbasis_grad, tbasis_curl, tbasis_div);
    vector<CompressedView<View_Sc4>> tcbasis, tcbasis_grad;
    for (size_t i=0; i<tbasis.size(); ++i) {
      tcbasis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      tcbasis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
    }
    wkset->basis_side = tcbasis;
    wkset->basis_grad_side = tcbasis_grad;
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the coarse grid solution to the fine grid integration points
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::computeSoln(const int & seedwhat) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnSideTimer);
  
  if (wkset->numAux > 0) {
    
    auto numAuxDOF = groupData->numAuxDOF;
    
    for (size_type var=0; var<numAuxDOF.extent(0); var++) {
      auto abasis = auxside_basis[auxusebasis[var]];
      auto off = subview(auxoffsets,var,ALL());
      string varname = wkset->aux_varlist[var];
      auto local_aux = wkset->getSolutionField("aux "+varname,false);
      Kokkos::deep_copy(local_aux,0.0);
      auto localID = localElemID;
      auto varaux = subview(aux,ALL(),var,ALL());
      if (seedwhat == 4) {
        parallel_for("bgroup aux 4",
                     TeamPolicy<AssemblyExec>(localID.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<abasis.extent(2); pt+=team.team_size() ) {
            for (size_type dof=0; dof<abasis.extent(1); ++dof) {
#ifndef MrHyDE_NO_AD
              AD auxval = AD(maxDerivs,off(dof), varaux(localID(elem),dof));
#else
              AD auxval = varaux(localID(elem),dof);
#endif
              local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
            }
          }
        });
      }
      else {
        parallel_for("bgroup aux 5",
                     TeamPolicy<AssemblyExec>(localID.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<abasis.extent(2); pt+=team.team_size() ) {
            for (size_type dof=0; dof<abasis.extent(1); ++dof) {
              AD auxval = varaux(localID(elem),dof);
              local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
            }
          }
        });
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous step solutions
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::resetPrevSoln(const size_t & set) {
  
  if (groupData->requiresTransient) {
    auto sol = u[set];
    auto sol_prev = u_prev[set];
    
    // shift previous step solns
    if (sol_prev.extent(3)>1) {
      parallel_for("Group reset prev soln 1",
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
    parallel_for("Group reset prev soln 2",
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
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Revert the solution (time step failed)
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::revertSoln(const size_t & set) {
  
  if (groupData->requiresTransient) {
    auto sol = u[set];
    auto sol_prev = u_prev[set];
    
    // copy current u into first step
    parallel_for("Group reset prev soln 2",
                 TeamPolicy<AssemblyExec>(sol_prev.extent(0), Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type i=team.team_rank(); i<sol_prev.extent(1); i+=team.team_size() ) {
        for (size_type j=0; j<sol.extent(2); j++) {
          sol(elem,i,j) = sol_prev(elem,i,j,0);
        }
      }
    });
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::resetStageSoln(const size_t & set) {
  
  if (groupData->requiresTransient) {
    auto sol = u[set];
    auto sol_stage = u_stage[set];
    
    parallel_for("Group reset stage 1",
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
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::updateStageSoln(const size_t & set) {
  
  if (groupData->requiresTransient) {
    auto sol = u[set];
    auto sol_stage = u_stage[set];
    // add u into the current stage soln (done after stage solution is computed)
    auto stage = wkset->current_stage;
    if (stage < sol_stage.extent_int(3)) {
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(sol_stage.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type i=team.team_rank(); i<sol_stage.extent(1); i+=team.team_size() ) {
          for (size_type j=0; j<sol_stage.extent(2); j++) {
            sol_stage(elem,i,j,stage) = sol(elem,i,j);
          }
        }
      });
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the contribution from this group to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                 const bool & compute_jacobian, const bool & compute_sens,
                                 const int & num_active_params, const bool & compute_disc_sens,
                                 const bool & compute_aux_sens, const bool & store_adjPrev,
                                 View_Sc3 local_res,
                                 View_Sc3 local_J) {
  
  /////////////////////////////////////////////////////////////////////////////////////
  // Compute the local contribution to the global residual and Jacobians
  /////////////////////////////////////////////////////////////////////////////////////
  
  // Boundary contribution
  {
    Teuchos::TimeMonitor localtimer(*boundaryResidualTimer);
    
    
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
    this->updateWorkset(seedwhat);
    groupData->physics_RCP->boundaryResidual(wkset->current_set,groupData->myBlock);
    
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
  
  {
    Teuchos::TimeMonitor localtimer(*residualFillTimer);
    
    // Update the local residual (forward mode)
    if (!isAdjoint) {
      this->updateRes(compute_sens, local_res);
    }
    
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT res
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::updateRes(const bool & compute_sens, View_Sc3 local_res) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = groupData->numDOF;
  
  if (compute_sens) {
    
    parallel_for("bgroup update res sens",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int r=0; r<local_res.extent(2); r++) {
#ifndef MrHyDE_NO_AD
            local_res(elem,offsets(n,j),r) -= res_AD(elem,offsets(n,j)).fastAccessDx(r);
#endif
          }
        }
      }
    });
  }
  else {
    parallel_for("bgroup update res",
                 TeamPolicy<AssemblyExec>(local_res.extent(0), Kokkos::AUTO, VectorSize),
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
// Use the AD res to update the scalarT J
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::updateJac(const bool & useadjoint, View_Sc3 local_J) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = groupData->numDOF;
  
  if (useadjoint) {
    parallel_for("bgroup update jac sens",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(m,k),offsets(n,j)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
  else {
    parallel_for("bgroup update jac",
                 TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
        for (int j=0; j<numDOF(n); j++) {
          for (unsigned int m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_J(elem,offsets(n,j),offsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(offsets(m,k));
            }
          }
        }
      }
    });
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////
// Use the AD res to update the scalarT Jparam
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::updateParamJac(View_Sc3 local_J) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = groupData->numDOF;
  auto paramoffsets = wkset->paramoffsets;
  auto numParamDOF = groupData->numParamDOF;
  
  parallel_for("bgroup update param jac",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VectorSize),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numParamDOF.extent(0); m++) {
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

void BoundaryGroup::updateAuxJac(View_Sc3 local_J) {
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = groupData->numDOF;
  auto aoffsets = auxoffsets;
  auto numAuxDOF = groupData->numAuxDOF;
  
  parallel_for("bgroup update aux jac",
               TeamPolicy<AssemblyExec>(local_J.extent(0), Kokkos::AUTO, VectorSize),
               KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
    int elem = team.league_rank();
    for (size_type n=team.team_rank(); n<numDOF.extent(0); n+=team.team_size() ) {
      for (int j=0; j<numDOF(n); j++) {
        for (unsigned int m=0; m<numAuxDOF.extent(0); m++) {
          for (int k=0; k<numAuxDOF(m); k++) {
            local_J(elem,offsets(n,j),aoffsets(m,k)) += res_AD(elem,offsets(n,j)).fastAccessDx(aoffsets(m,k));
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

View_Sc2 BoundaryGroup::getDirichlet(const size_t & set) {
  
  View_Sc2 dvals("initial values",numElem,LIDs[set].extent(1));
  this->updateWorkset(0);
  
  Kokkos::View<string**,HostDevice> bcs = wkset->var_bcs;
  auto offsets = wkset->offsets;
  auto numDOF = groupData->numDOF;
  auto cwts = wts;
  auto cnormals = normals;

  for (size_t n=0; n<wkset->varlist.size(); n++) {
    if (bcs(n,sidenum) == "Dirichlet") { // is this a strong DBC for this variable
      auto dip = groupData->physics_RCP->getDirichlet(n,set,groupData->myBlock, sidename);

      int bind = wkset->usebasis[n];
      std::string btype = groupData->basis_types[bind];
      auto cbasis = basis[bind]; // may fault in memory-saving mode
      
      auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bgroup fill Dirichlet",
                     RangePolicy<AssemblyExec>(0,cwts.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cwts.extent(1); j++ ) {
              dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,0)*cwts(e,j);
            }
          }
        });
      }
      else if (btype == "HDIV"){
        
        View_Sc2 nx, ny, nz;
        nx = cnormals[0];
        if (cnormals.size()>1) {
          ny = cnormals[1];
        }
        if (cnormals.size()>2) {
          nz = cnormals[2];
        }
        
        parallel_for("bgroup fill Dirichlet HDIV",
                     RangePolicy<AssemblyExec>(0,dvals.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cwts.extent(1); j++ ) {
              dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,0)*nx(e,j)*cwts(e,j);
              if (cbasis.extent(3)>1) {
                dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,1)*ny(e,j)*cwts(e,j);
              }
              if (cbasis.extent(3)>2) {
                dvals(e,off(i)) += dip(e,j)*cbasis(e,i,j,2)*nz(e,j)*cwts(e,j);
              }
            }
          }
        });
      }
      else if (btype == "HCURL"){
        // not implemented yet
      }
    }
  }
  return dvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

View_Sc3 BoundaryGroup::getMass(const size_t & set) {
  
  View_Sc3 mass("local mass", numElem, LIDs[set].extent(1), LIDs[set].extent(1));
  
  Kokkos::View<string**,HostDevice> bcs = wkset->var_bcs;
  auto offsets = wkset->offsets;
  auto numDOF = groupData->numDOF;
  auto cwts = wts;
  
  for (size_type n=0; n<numDOF.extent(0); n++) {
    if (bcs(n,sidenum) == "Dirichlet") { // is this a strong DBC for this variable
      int bind = wkset->usebasis[n];
      auto cbasis = basis[bind];
      auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
      std::string btype = groupData->basis_types[bind];
      
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bgroup compute mass",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cbasis.extent(1); j++ ) {
              for( size_type k=0; k<cbasis.extent(2); k++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k);
              }
            }
          }
        });
      }
      else if (btype == "HDIV") {
        auto cnormals = normals;
        View_Sc2 nx, ny, nz;
        nx = cnormals[0];
        if (cnormals.size()>1) {
          ny = cnormals[1];
        }
        if (cnormals.size()>2) {
          nz = cnormals[2];
        }
        parallel_for("bgroup compute mass HDIV",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_LAMBDA (const int e ) {
          for( size_type i=0; i<cbasis.extent(1); i++ ) {
            for( size_type j=0; j<cbasis.extent(1); j++ ) {
              for( size_type k=0; k<cbasis.extent(2); k++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,0)*nx(e,k)*cbasis(e,j,k,0)*nx(e,k)*cwts(e,k);
                if (cbasis.extent(3)>1) {
                  mass(e,off(i),off(j)) += cbasis(e,i,k,1)*ny(e,k)*cbasis(e,j,k,1)*ny(e,k)*cwts(e,k);
                }
                if (cbasis.extent(3)>2) {
                  mass(e,off(i),off(j)) += cbasis(e,i,k,2)*nz(e,k)*cbasis(e,j,k,2)*nz(e,k)*cwts(e,k);
                }
              }
            }
          }
        });
      }
      else if (btype == "HCURL"){
        // not implemented yet
      }
    }
  }
  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Pass the group data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::updateData() {
  
  // hard coded for what I need it for right now
  if (groupData->have_phi) {
    wkset->have_rotation_phi = true;
    wkset->rotation_phi = data;
    wkset->allocateRotations();
  }
  else if (groupData->have_rotation) {
    wkset->have_rotation = true;
    wkset->allocateRotations();
    auto rot = wkset->rotation;
    
    parallel_for("update data",
                 RangePolicy<AssemblyExec>(0,data.extent(0)),
                 KOKKOS_LAMBDA (const size_type e ) {
      rot(e,0,0) = data(e,0);
      rot(e,0,1) = data(e,1);
      rot(e,0,2) = data(e,2);
      rot(e,1,0) = data(e,3);
      rot(e,1,1) = data(e,4);
      rot(e,1,2) = data(e,5);
      rot(e,2,0) = data(e,6);
      rot(e,2,1) = data(e,7);
      rot(e,2,2) = data(e,8);
    });
  
  }
  else if (groupData->have_extra_data) {
    wkset->extra_data = data;
  }
  
}


size_t BoundaryGroup::getStorage() {
  size_t mystorage = 0;
  if (storeAll) {
    size_t scalarcost = sizeof(ScalarT); // 8 bytes per double
    
    for (size_t k=0; k<ip.size(); ++k) {
      mystorage += scalarcost*ip[k].size();
    }
    for (size_t k=0; k<normals.size(); ++k) {
      mystorage += scalarcost*normals[k].size();
    }
    for (size_t k=0; k<tangents.size(); ++k) {
      mystorage += scalarcost*tangents[k].size();
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
  }
  return mystorage;
}
