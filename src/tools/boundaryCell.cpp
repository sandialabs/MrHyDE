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

#include "boundaryCell.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

BoundaryCell::BoundaryCell(const Teuchos::RCP<CellMetaData> & cellData_,
                           const DRV nodes_,
                           const Kokkos::View<LO*,AssemblyDevice> localID_,
                           const Kokkos::View<LO*,AssemblyDevice> sideID_,
                           const int & sidenum_, const string & sidename_,
                           const int & cellID_,
                           LIDView LIDs_,
                           Kokkos::View<int****,HostDevice> sideinfo_,
                           Teuchos::RCP<DiscretizationInterface> & disc_) :
cellData(cellData_), localElemID(localID_), localSideID(sideID_),
sidenum(sidenum_), cellID(cellID_), nodes(nodes_), sideinfo(sideinfo_), sidename(sidename_), LIDs(LIDs_), disc(disc_)   {

  numElem = nodes.extent(0);

  auto LIDs_tmp = Kokkos::create_mirror_view(LIDs);
  Kokkos::deep_copy(LIDs_tmp,LIDs);  
  LIDs_host = LIDView_host("LIDs on host",LIDs.extent(0), LIDs.extent(1));
  Kokkos::deep_copy(LIDs_host,LIDs_tmp);
  
  if (cellData->storeAll) {
    int numip = cellData->ref_side_ip[0].extent(0);
    wts = View_Sc2("physical wts",numElem, numip);
    hsize = View_Sc1("physical meshsize",numElem);
    orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
    disc->getPhysicalBoundaryData(cellData, nodes, localElemID, localSideID, orientation,
                                  ip, wts, normals, tangents, hsize,
                                  basis, basis_grad, basis_curl, basis_div, true, true);
    
  }
  else {
    orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
    disc->getPhysicalOrientations(cellData, localElemID, orientation, false);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setWorkset(Teuchos::RCP<workset> & wkset_) {
  
  wkset = wkset_;
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setParams(LIDView paramLIDs_) {
  paramLIDs = paramLIDs_;
  paramLIDs_host = LIDView_host("param LIDs on host", paramLIDs.extent(0), paramLIDs.extent(1));
  Kokkos::deep_copy(paramLIDs_host, paramLIDs);
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux basis functions at the integration points.
// This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::addAuxDiscretization(const vector<basis_RCP> & abasis_pointers,
                                        const vector<DRV> & asideBasis,
                                        const vector<DRV> & asideBasisGrad) {
  
  auxbasisPointers = abasis_pointers;
  auxside_basis = asideBasis;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux variables
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::addAuxVars(const vector<string> & auxlist_) {
  auxlist = auxlist_;
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each variable will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setUseBasis(vector<int> & usebasis_, const int & numsteps, const int & numstages) {
  vector<int> usebasis = usebasis_;
  
  // Set up the containers for usual solution storage
  int maxnbasis = 0;
  for (size_type i=0; i<cellData->numDOF_host.extent(0); i++) {
    if (cellData->numDOF_host(i) > maxnbasis) {
      maxnbasis = cellData->numDOF_host(i);
    }
  }
  u = View_Sc3("u",numElem,cellData->numDOF.extent(0),maxnbasis);
  if (cellData->requiresAdjoint) {
    phi = View_Sc3("phi",numElem,cellData->numDOF.extent(0),maxnbasis);
  }
  if (cellData->requiresTransient) {
    u_prev = View_Sc4("u previous",numElem,cellData->numDOF.extent(0),maxnbasis,numsteps);
    u_stage = View_Sc4("u stages",numElem,cellData->numDOF.extent(0),maxnbasis,numstages);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
  auto numParamDOF = cellData->numParamDOF;
  
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

void BoundaryCell::setAuxUseBasis(vector<int> & ausebasis_) {
  auxusebasis = ausebasis_;
  auto numAuxDOF = Kokkos::create_mirror_view(cellData->numAuxDOF);
  Kokkos::deep_copy(numAuxDOF,cellData->numAuxDOF);
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

void BoundaryCell::updateWorkset(const int & seedwhat, const bool & override_transient) {
  
  // Reset the residual and data in the workset
  wkset->resetResidual();
  wkset->sidename = sidename;
  wkset->currentside = sidenum;
  wkset->numElem = numElem;
  this->updateData();
  
  // Update the integration info and basis in workset
  this->updateWorksetBasis();
  
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

  // Map the AD solutions to the aolutions at the boundary ip
  this->computeSoln(seedwhat);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the workset basis and ip
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateWorksetBasis() {

  wkset->numElem = numElem;
  
  if (cellData->storeAll) {
    wkset->wts_side = wts;
    wkset->h = hsize;
    wkset->setIP(ip," side");
    wkset->setNormals(normals);
    wkset->setTangents(tangents);
    wkset->basis_side = basis;
    wkset->basis_grad_side = basis_grad;
  }
  else {
    int numip = cellData->ref_side_ip[0].extent(0);
    vector<View_Sc2> tip;
    vector<View_Sc2> tnormals;
    vector<View_Sc2> ttangents;
    View_Sc2 twts("physical wts",numElem, numip);
    View_Sc1 thsize("physical meshsize",numElem);
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalBoundaryData(cellData, nodes, localElemID,
                                  localSideID, orientation,
                                  tip, twts, tnormals, ttangents, thsize,
                                  tbasis, tbasis_grad, tbasis_curl, tbasis_div, true, false);
    
    wkset->wts_side = twts;
    wkset->h = thsize;
    wkset->setIP(tip," side");
    wkset->setNormals(tnormals);
    wkset->setTangents(ttangents);
    wkset->basis_side = tbasis;
    wkset->basis_grad_side = tbasis_grad;
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the coarse grid solution to the fine grid integration points
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::computeSoln(const int & seedwhat) {
  
  Teuchos::TimeMonitor localtimer(*computeSolnSideTimer);
  
  wkset->computeSolnSideIP();
  wkset->computeParamSideIP();
  
  
  if (wkset->numAux > 0) {
    
    auto numAuxDOF = cellData->numAuxDOF;
    
    for (size_type var=0; var<numAuxDOF.extent(0); var++) {
      auto abasis = auxside_basis[auxusebasis[var]];
      auto off = subview(auxoffsets,var,ALL());
      string varname = wkset->aux_varlist[var];
      auto local_aux = wkset->getData("aux "+varname+" side");
      Kokkos::deep_copy(local_aux,0.0);
      auto localID = localElemID;
      auto varaux = subview(aux,ALL(),var,ALL());
      if (seedwhat == 4) {
        parallel_for("bcell aux 4",
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
        parallel_for("bcell aux 5",
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
// Compute the contribution from this cell to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
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
    cellData->physics_RCP->boundaryResidual(cellData->myBlock);
    
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

void BoundaryCell::updateRes(const bool & compute_sens, View_Sc3 local_res) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (compute_sens) {
    
    parallel_for("bcell update res sens",
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
    parallel_for("bcell update res",
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

void BoundaryCell::updateJac(const bool & useadjoint, View_Sc3 local_J) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  
  if (useadjoint) {
    parallel_for("bcell update jac sens",
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
    parallel_for("bcell update jac",
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

void BoundaryCell::updateParamJac(View_Sc3 local_J) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto paramoffsets = wkset->paramoffsets;
  auto numParamDOF = cellData->numParamDOF;
  
  parallel_for("bcell update param jac",
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

void BoundaryCell::updateAuxJac(View_Sc3 local_J) {
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto aoffsets = auxoffsets;
  auto numAuxDOF = cellData->numAuxDOF;
  
  parallel_for("bcell update aux jac",
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

View_Sc2 BoundaryCell::getDirichlet() {
  
  View_Sc2 dvals("initial values",numElem,LIDs.extent(1));
  this->updateWorkset(0);
  
  Kokkos::View<string**,HostDevice> bcs = wkset->var_bcs;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto cwts = wts;
  auto cnormals = normals;
  
  for (size_t n=0; n<wkset->varlist.size(); n++) {
    if (bcs(n,sidenum) == "Dirichlet") { // is this a strong DBC for this variable
      auto dip = cellData->physics_RCP->getDirichlet(n,cellData->myBlock, sidename);
      int bind = wkset->usebasis[n];
      std::string btype = cellData->basis_types[bind];
      auto cbasis = basis[bind]; // may fault in memory-saving mode
      
      auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bcell fill Dirichlet",
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
        
        parallel_for("bcell fill Dirichlet HDIV",
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

View_Sc3 BoundaryCell::getMass() {
  
  View_Sc3 mass("local mass", numElem, LIDs.extent(1), LIDs.extent(1));
  
  Kokkos::View<string**,HostDevice> bcs = wkset->var_bcs;
  auto offsets = wkset->offsets;
  auto numDOF = cellData->numDOF;
  auto cwts = wts;
  
  for (size_type n=0; n<numDOF.extent(0); n++) {
    if (bcs(n,sidenum) == "Dirichlet") { // is this a strong DBC for this variable
      int bind = wkset->usebasis[n];
      auto cbasis = basis[bind];
      auto off = Kokkos::subview(offsets,n,Kokkos::ALL());
      std::string btype = cellData->basis_types[bind];
      
      if (btype == "HGRAD" || btype == "HVOL" || btype == "HFACE"){
        parallel_for("bcell compute mass",
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
        parallel_for("bcell compute mass HDIV",
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
// Pass the cell data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryCell::updateData() {
  
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


size_t BoundaryCell::getStorage() {
  size_t mystorage = 0;
  if (cellData->storeAll) {
    size_t scalarcost = 8; // 8 bytes per double
    
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
