/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "boundaryGroup.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

BoundaryGroup::BoundaryGroup(const Teuchos::RCP<GroupMetaData> & group_data_,
                             const Kokkos::View<LO*,AssemblyDevice> localID_,
                             LO & sideID_,
                             const int & sidenum_, const string & sidename_,
                             const int & groupID_,
                             Teuchos::RCP<DiscretizationInterface> & disc_,
                             const bool & storeAll_) :
group_data(group_data_), localElemID(localID_), localSideID(sideID_),
sidenum(sidenum_), groupID(groupID_), 
sidename(sidename_), disc(disc_)   {
  
  numElem = localElemID.extent(0);
  
  storeAll = storeAll_;
  
  haveBasis = false;
  have_nodes = false;
  //nodes = disc->mesh->getMyNodes(group_data->my_block, localElemID);

  // Orientations are always stored
  orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
  disc->getPhysicalOrientations(group_data, localElemID, orientation, false);
  
  // Integration points, weights, normals, tangents and element sizes are always stored
  int numip = group_data->ref_side_ip[0].extent(0);
  wts = View_Sc2("physical wts",numElem, numip);
  hsize = View_Sc1("physical meshsize",numElem);
  
  disc->getPhysicalBoundaryIntegrationData(group_data, localElemID, localSideID, ip,
                                           wts, normals, tangents);
  
  this->computeSize();
  this->initializeBasisIndex();
  if (group_data->have_multidata) {
    multidata = View_Sc3("multidata array",numElem,wts.extent(1),54);
  }
  else {
    multidata = View_Sc3("multidata array",0,0,0);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

BoundaryGroup::BoundaryGroup(const Teuchos::RCP<GroupMetaData> & group_data_,
                             const Kokkos::View<LO*,AssemblyDevice> localID_,
                             DRV nodes_, LO & sideID_,
                             const int & sidenum_, const string & sidename_,
                             const int & groupID_,
                             Teuchos::RCP<DiscretizationInterface> & disc_,
                             const bool & storeAll_) :
group_data(group_data_), localElemID(localID_), localSideID(sideID_),
sidenum(sidenum_), groupID(groupID_), nodes(nodes_),
sidename(sidename_), disc(disc_)   {
  
  numElem = localElemID.extent(0);
  
  storeAll = storeAll_;
  
  haveBasis = false;
  have_nodes = true;

  // Orientations are always stored
  orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
  disc->getPhysicalOrientations(group_data, localElemID, orientation, false);
  
  // Integration points, weights, normals, tangents and element sizes are always stored
  int numip = group_data->ref_side_ip[0].extent(0);
  wts = View_Sc2("physical wts",numElem, numip);
  hsize = View_Sc1("physical meshsize",numElem);
  
  disc->getPhysicalBoundaryIntegrationData(group_data, nodes, localSideID, ip,
                                           wts, normals, tangents);
  
  this->computeSize();
  this->initializeBasisIndex();
  if (group_data->have_multidata) {
    multidata = View_Sc3("multidata array",numElem,wts.extent(1),54);
  }
  else {
    multidata = View_Sc3("multidata array",0,0,0);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::computeSize() {

  size_t dimension = group_data->dimension;

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
    
    if (have_nodes) {
      disc->getPhysicalBoundaryBasis(group_data, nodes, localSideID, orientation,
                                     tbasis, tbasis_grad, tbasis_curl, tbasis_div);
    }
    else {
      disc->getPhysicalBoundaryBasis(group_data, localElemID, localSideID, 
                                     tbasis, tbasis_grad, tbasis_curl, tbasis_div);
    }
    for (size_t i=0; i<tbasis.size(); ++i) {
      basis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      basis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
      basis_div.push_back(CompressedView<View_Sc3>(tbasis_div[i]));
      basis_curl.push_back(CompressedView<View_Sc4>(tbasis_curl[i]));
    }
    haveBasis = true;
    if (!keepnodes) {
      //nodes = DRV("dummy nodes",1);
    }
  }
  else if (group_data->use_basis_database) {
    for (size_t i=0; i<group_data->database_side_basis.size(); ++i) {
      basis.push_back(CompressedView<View_Sc4>(group_data->database_side_basis[i],basis_index));
      basis_grad.push_back(CompressedView<View_Sc4>(group_data->database_side_basis_grad[i],basis_index));
    }
    
    if (!keepnodes) {
      //nodes = DRV("empty nodes",1);
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::createHostLIDs() {
  bool data_avail = false;
  if (Kokkos::SpaceAccessibility<HostExec, AssemblyDevice::memory_space>::accessible) {
    data_avail = true;
  }
  
  LIDs_host = vector<LIDView_host>(LIDs.size());
  for (size_t set=0; set<LIDs.size(); ++set) {
    if (data_avail) {
      LIDs_host[set] = LIDs[set];
    }
    else {
      auto LIDs_tmp = Kokkos::create_mirror_view(LIDs[set]);
      Kokkos::deep_copy(LIDs_tmp,LIDs[set]);
      LIDView_host currLIDs_host("LIDs on host",LIDs[set].extent(0), LIDs[set].extent(1));
      Kokkos::deep_copy(currLIDs_host,LIDs_tmp);
      LIDs_host[set] = currLIDs_host;
    }
  }
  
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

void BoundaryGroup::setUseBasis(vector<vector<int> > & usebasis_, const vector<int> & maxnumsteps, 
                                const vector<int> & maxnumstages, const bool & allocate_storage) {
  vector<vector<int> > usebasis = usebasis_;
  
  if (allocate_storage) {
    have_sols = true;
    // Set up the containers for usual solution storage
    for (size_t set=0; set<usebasis.size(); ++set) {
      int maxnbasis = 0;
      for (size_type i=0; i<group_data->set_num_dof_host[set].extent(0); i++) {
        if (group_data->set_num_dof_host[set](i) > maxnbasis) {
          maxnbasis = group_data->set_num_dof_host[set](i);
        }
      }
      View_Sc3 newu("u bgrp",numElem,group_data->set_num_dof[set].extent(0),maxnbasis);
      sol.push_back(newu);
      if (group_data->requires_adjoint) {
        View_Sc3 newphi("phi bgrp",numElem,group_data->set_num_dof[set].extent(0),maxnbasis);
        phi.push_back(newphi);
      }
      if (group_data->requires_transient) {
        View_Sc4 newuprev("u previous bgrp",numElem,group_data->set_num_dof[set].extent(0),maxnbasis,maxnumsteps[set]);
        sol_prev.push_back(newuprev);
        View_Sc4 newustage("u stages bgrp",numElem,group_data->set_num_dof[set].extent(0),maxnbasis,maxnumstages[set]-1);
        sol_stage.push_back(newustage);
      }
    }
  }
  else {
    have_sols = false;
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_,
                                     const bool & allocate_storage) {
  vector<int> paramusebasis = pusebasis_;
  
  if (allocate_storage) {
    auto numParamDOF = group_data->num_param_dof;
    int maxnbasis = 0;
    for (size_type i=0; i<numParamDOF.extent(0); i++) {
      if (numParamDOF(i) > maxnbasis) {
        maxnbasis = numParamDOF(i);
      }
    }
    param = View_Sc3("param",numElem,numParamDOF.extent(0),maxnbasis);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::setAuxUseBasis(vector<int> & ausebasis_, const bool & allocate_storage) {
  auxusebasis = ausebasis_;

  if (allocate_storage) {
    auto numAuxDOF = Kokkos::create_mirror_view(group_data->num_aux_dof);
    Kokkos::deep_copy(numAuxDOF,group_data->num_aux_dof);
    int maxnbasis = 0;
    for (size_type i=0; i<numAuxDOF.extent(0); i++) {
      if (numAuxDOF(i) > maxnbasis) {
        maxnbasis = numAuxDOF(i);
      }
    }
    aux = View_Sc3("aux",numElem,numAuxDOF.extent(0),maxnbasis);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous step solutions
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::resetPrevSoln(const size_t & set) {
  
  if (group_data->requires_transient && sol.size() > set && sol_prev.size() > set) {
    auto csol = sol[set];
    auto csol_prev = sol_prev[set];
    
    // shift previous step solns
    if (csol_prev.extent(3)>1) {
      parallel_for("Group reset prev soln 1",
                   TeamPolicy<AssemblyExec>(csol_prev.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type i=team.team_rank(); i<csol_prev.extent(1); i+=team.team_size() ) {
          for (size_type j=0; j<csol_prev.extent(2); j++) {
            for (size_type s=csol_prev.extent(3)-1; s>0; s--) {
              csol_prev(elem,i,j,s) = csol_prev(elem,i,j,s-1);
            }
          }
        }
      });
    }
    
    // copy current u into first step
    parallel_for("Group reset prev soln 2",
                 TeamPolicy<AssemblyExec>(csol_prev.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type i=team.team_rank(); i<csol_prev.extent(1); i+=team.team_size() ) {
        for (size_type j=0; j<csol.extent(2); j++) {
          csol_prev(elem,i,j,0) = csol(elem,i,j);
        }
      }
    });
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Revert the solution (time step failed)
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::revertSoln(const size_t & set) {
  
  if (group_data->requires_transient && sol.size() > set && sol_prev.size() > set) {
    auto csol = sol[set];
    auto csol_prev = sol_prev[set];
    
    // copy current u into first step
    parallel_for("Group reset prev soln 2",
                 TeamPolicy<AssemblyExec>(csol_prev.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type i=team.team_rank(); i<csol_prev.extent(1); i+=team.team_size() ) {
        for (size_type j=0; j<csol.extent(2); j++) {
          csol(elem,i,j) = csol_prev(elem,i,j,0);
        }
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void BoundaryGroup::resetStageSoln(const size_t & set) {
  
  if (group_data->requires_transient && sol.size() > set && sol_stage.size() > set) {
    auto csol = sol[set];
    auto csol_stage = sol_stage[set];
    
    parallel_for("Group reset stage 1",
                 TeamPolicy<AssemblyExec>(csol_stage.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type i=team.team_rank(); i<csol_stage.extent(1); i+=team.team_size() ) {
        for (size_type j=0; j<csol_stage.extent(2); j++) {
          for (size_type k=0; k<csol_stage.extent(3); k++) {
            csol_stage(elem,i,j,k) = csol(elem,i,j);
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
  
  if (group_data->requires_transient && sol.size() > set && sol_stage.size() > set) {
    auto csol = sol[set];
    auto csol_stage = sol_stage[set];
    // add u into the current stage soln (done after stage solution is computed)
    auto stage = group_data->current_stage;
    if (stage < csol_stage.extent(3)) {
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(csol_stage.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type i=team.team_rank(); i<csol_stage.extent(1); i+=team.team_size() ) {
          for (size_type j=0; j<csol_stage.extent(2); j++) {
            csol_stage(elem,i,j,stage) = csol(elem,i,j);
          }
        }
      });
    }
  }
  
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Compute the storage costs
///////////////////////////////////////////////////////////////////////////////////////

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
