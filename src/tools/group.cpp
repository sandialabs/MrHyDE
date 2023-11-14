/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "group.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

using namespace MrHyDE;

Group::Group(const Teuchos::RCP<GroupMetaData> & group_data_,
           const Kokkos::View<LO*,AssemblyDevice> localID_,
           Teuchos::RCP<DiscretizationInterface> & disc_,
           const bool & storeAll_) :
group_data(group_data_), localElemID(localID_), disc(disc_)
{
  numElem = localElemID.extent(0);
  
  active = true;
  storeAll = storeAll_;
  haveBasis = false;
  storeMass = true;
  have_nodes = false;

  // Even if we don't store the basis or integration info, we still store
  // the orientations since these are small, but expensive to recompute (for some reason)
  //orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
  //disc->getPhysicalOrientations(group_data, localElemID,
    //                            orientation, true);
  //nodes = disc->mesh->getMyNodes(group_data->my_block, localElemID);
  //View_Sc2 twts = this->getWts();
  //hsize = View_Sc1("physical hsize",numElem);
  
  //this->computeSize(twts);

  if (group_data->build_face_terms) {
    for (size_type side=0; side<group_data->num_sides; side++) {
      int numfip = group_data->ref_side_ip[side].extent(0);
      vector<View_Sc2> face_ip;
      vector<View_Sc2> face_normals;
      View_Sc2 face_wts("face wts", numElem, numfip);
      vector<View_Sc4> face_basis, face_basis_grad;
      disc->getPhysicalFaceIntegrationData(group_data, side, localElemID, 
                                           face_ip, face_wts, face_normals);
          
          
      ip_face.push_back(face_ip);
      wts_face.push_back(face_wts);
      normals_face.push_back(face_normals);
      
    }
    this->computeFaceSize();   
  }

  this->initializeBasisIndex();
  
}

Group::Group(const Teuchos::RCP<GroupMetaData> & group_data_,
           const Kokkos::View<LO*,AssemblyDevice> localID_,
           DRV nodes_,
           Teuchos::RCP<DiscretizationInterface> & disc_,
           const bool & storeAll_) :
group_data(group_data_), localElemID(localID_), nodes(nodes_), disc(disc_)
{
  numElem = localElemID.extent(0);
  
  active = true;
  storeAll = storeAll_;
  haveBasis = false;
  storeMass = true;
  have_nodes = true;

  // Even if we don't store the basis or integration info, we still store
  // the orientations since these are small, but expensive to recompute (for some reason)
  orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
  disc->getPhysicalOrientations(group_data, localElemID,
                                orientation, true);
  
  //View_Sc2 twts = this->getWts();
  //hsize = View_Sc1("physical hsize",numElem);
  
  //this->computeSize(twts);

  if (group_data->build_face_terms) {
    for (size_type side=0; side<group_data->num_sides; side++) {
      int numfip = group_data->ref_side_ip[side].extent(0);
      vector<View_Sc2> face_ip;
      vector<View_Sc2> face_normals;
      View_Sc2 face_wts("face wts", numElem, numfip);
      vector<View_Sc4> face_basis, face_basis_grad;
      disc->getPhysicalFaceIntegrationData(group_data, side, nodes,
                                           face_ip, face_wts, face_normals);
          
          
      ip_face.push_back(face_ip);
      wts_face.push_back(face_wts);
      normals_face.push_back(face_normals);
      
    }
    this->computeFaceSize();   
  }

  this->initializeBasisIndex();
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::computeSize(View_Sc2 twts) {
  /*
  // -------------------------------------------------
  // Compute the element sizes (h = vol^(1/dimension))
  // -------------------------------------------------
  size_t dimension = group_data->dimension;

  parallel_for("elem size",
               RangePolicy<AssemblyExec>(0,twts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT vol = 0.0;
    for (size_type i=0; i<twts.extent(1); i++) {
      vol += twts(elem,i);
    }
    ScalarT dimscl = 1.0/(ScalarT)dimension;
    hsize(elem) = std::pow(vol,dimscl);
  });
  */
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::computeFaceSize() {

  size_t dimension = group_data->dimension;

  for (size_t side=0; side<wts_face.size(); side++) {
    auto cwts = wts_face[side];
    View_Sc1 face_hsize("face hsize", numElem);
    parallel_for("bcell hsize",
                 RangePolicy<AssemblyExec>(0,cwts.extent(0)),
                 KOKKOS_LAMBDA (const int e ) {
      ScalarT vol = 0.0;
      for (size_type i=0; i<cwts.extent(1); i++) {
        vol += cwts(e,i);
      }
      ScalarT dimscl = 1.0/((ScalarT)dimension-1.0);
      face_hsize(e) = pow(vol,dimscl);
    });
    hsize_face.push_back(face_hsize);
  }   
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::initializeBasisIndex() {

  basis_index = Kokkos::View<LO*,AssemblyDevice>("basis index",numElem);
  parallel_for("compute hsize",
               RangePolicy<AssemblyExec>(0,basis_index.extent(0)),
               KOKKOS_LAMBDA (const int e ) {
    basis_index(e) = e;
  });
}  
  
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::computeBasis(const bool & keepnodes) {
  
  // Set up the integration points
  
  if (!have_ip) {
    if (group_data->use_ip_database) {
      CompressedView<View_Sc2> ip_x(group_data->database_x, ip_x_index);
      ip.push_back(ip_x);
      if (group_data->dimension > 1) {
        CompressedView<View_Sc2> ip_y(group_data->database_y, ip_y_index);
        ip.push_back(ip_y);
      }
      if (group_data->dimension > 2) {
        CompressedView<View_Sc2> ip_z(group_data->database_z, ip_z_index);
        ip.push_back(ip_z);
      }
    }
    else {
      vector<View_Sc2> newip;
      if (have_nodes) {
        disc->getPhysicalIntegrationPts(group_data, nodes, newip);
      }
      else {
        disc->getPhysicalIntegrationPts(group_data, localElemID, newip);
      }
      CompressedView<View_Sc2> ip_x(newip[0]);
      ip.push_back(ip_x);
      if (group_data->dimension > 1) {
        CompressedView<View_Sc2> ip_y(newip[1]);
        ip.push_back(ip_y);
      }
      if (group_data->dimension > 2) {
        CompressedView<View_Sc2> ip_z(newip[2]);
        ip.push_back(ip_z);
      }
    }
    have_ip = true;
  }
  
  if (storeAll) {
    
    View_Sc2 twts = this->getWts();
    wts = CompressedView<View_Sc2>(twts);
  
    if (!haveBasis) {
      // Compute integration data and basis functions
      vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
      vector<View_Sc3> tbasis_div;
      if (have_nodes) {
        disc->getPhysicalVolumetricBasis(group_data, nodes, orientation,
                                         tbasis, tbasis_grad, tbasis_curl,
                                         tbasis_div, tbasis_nodes, true);
      }
      else {
        disc->getPhysicalVolumetricBasis(group_data, localElemID,
                                         tbasis, tbasis_grad, tbasis_curl,
                                         tbasis_div, tbasis_nodes, true);
      }

      for (size_t i=0; i<tbasis.size(); ++i) {
        basis.push_back(CompressedView<View_Sc4>(tbasis[i]));
        basis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
        basis_div.push_back(CompressedView<View_Sc3>(tbasis_div[i]));
        basis_curl.push_back(CompressedView<View_Sc4>(tbasis_curl[i]));
        basis_nodes.push_back(CompressedView<View_Sc4>(tbasis_nodes[i]));
      }
      if (group_data->build_face_terms) {
        for (size_type side=0; side<group_data->num_sides; side++) {
          vector<View_Sc4> face_basis, face_basis_grad;
          
          if (have_nodes) {
            disc->getPhysicalFaceBasis(group_data, side, nodes, orientation,
                                       face_basis, face_basis_grad);
          }
          else {
            disc->getPhysicalFaceBasis(group_data, side, localElemID,
                                      face_basis, face_basis_grad);
          }
          vector<CompressedView<View_Sc4>> newf_basis, newf_basis_grad;
          for (size_t i=0; i<face_basis.size(); ++i) {
            newf_basis.push_back(CompressedView<View_Sc4>(face_basis[i]));
            newf_basis_grad.push_back(CompressedView<View_Sc4>(face_basis_grad[i]));
          }
          basis_face.push_back(newf_basis);
          basis_grad_face.push_back(newf_basis_grad);
        }
        
      }
      haveBasis = true;
    }    
    if (!keepnodes) {
      //nodes = DRV("empty nodes",1);
      orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",1);
    }
  }
  else if (group_data->use_basis_database) {
    wts = CompressedView<View_Sc2>(group_data->database_wts,basis_index);
    for (size_t i=0; i<group_data->database_basis.size(); ++i) {
      basis.push_back(CompressedView<View_Sc4>(group_data->database_basis[i],basis_index));
      basis_grad.push_back(CompressedView<View_Sc4>(group_data->database_basis_grad[i],basis_index));
      basis_div.push_back(CompressedView<View_Sc3>(group_data->database_basis_div[i],basis_index));
      basis_curl.push_back(CompressedView<View_Sc4>(group_data->database_basis_curl[i],basis_index));
    }
    if (!keepnodes) {
      //nodes = DRV("empty nodes",1);
      orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",1);
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::createHostLIDs() {
  
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
      auto LIDs_tmp = create_mirror_view(LIDs[set]);
      deep_copy(LIDs_tmp,LIDs[set]);
      
      LIDView_host currLIDs_host("LIDs on host",LIDs[set].extent(0), LIDs[set].extent(1));
      deep_copy(currLIDs_host,LIDs_tmp);
      LIDs_host[set] = currLIDs_host;
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::setParams(LIDView paramLIDs_) {
  
  paramLIDs = paramLIDs_;
  paramLIDs_host = LIDView_host("param LIDs on host", paramLIDs.extent(0), paramLIDs.extent(1));//create_mirror_view(paramLIDs);
  deep_copy(paramLIDs_host, paramLIDs);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Add the aux basis functions at the integration points.
// This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
///////////////////////////////////////////////////////////////////////////////////////

void Group::addAuxDiscretization(const vector<basis_RCP> & abasis_pointers, const vector<DRV> & abasis,
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

void Group::addAuxVars(const vector<string> & auxlist_) {
  auxlist = auxlist_;
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the regular parameters (everything but discretized)
///////////////////////////////////////////////////////////////////////////////////////

void Group::updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) {
  group_data->physics->updateParameters(params, paramnames);
}


///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each variable will use
///////////////////////////////////////////////////////////////////////////////////////

void Group::setUseBasis(vector<vector<int> > & usebasis_, const vector<int> & maxnumsteps, 
                        const vector<int> & maxnumstages, const bool & allocate_storage) {
  vector<vector<int> > usebasis = usebasis_;

  if (allocate_storage) {
    have_sols = true;
    // Set up the containers for usual solution storage
    sol = vector<View_Sc3>(group_data->num_sets);
    phi = vector<View_Sc3>(group_data->num_sets);
  
    sol_prev = vector<View_Sc4>(group_data->num_sets);
    sol_stage = vector<View_Sc4>(group_data->num_sets);
    phi_prev = vector<View_Sc4>(group_data->num_sets);
    phi_stage = vector<View_Sc4>(group_data->num_sets);
  
    sol_avg = vector<View_Sc3>(group_data->num_sets);
  
    for (size_t set=0; set<group_data->num_sets; ++set) {
      int maxnbasis = 0;
      for (size_type i=0; i<group_data->set_num_dof_host[set].extent(0); i++) {
        if (group_data->set_num_dof_host[set](i) > maxnbasis) {
          maxnbasis = group_data->set_num_dof_host[set](i);
        }
      }
    
      // Storage for gathered forward (state) solutions
      View_Sc3 newu("u",numElem,group_data->set_num_dof[set].extent(0),maxnbasis);
      sol[set] = newu;
    
      // Storage for adjoint solutions
      View_Sc3 newphi;
      if (group_data->requires_adjoint) {
        newphi = View_Sc3("phi",numElem,group_data->set_num_dof[set].extent(0),maxnbasis);
      }
      else {
        newphi = View_Sc3("phi",1,1,1); // just a placeholder
      }
      phi[set] = newphi;
    
      // Storage for transient data for forward and adjoint solutions
      View_Sc4 newuprev, newustage, newphiprev, newphistage;
    
      if (group_data->requires_transient) {
        newuprev = View_Sc4("u previous",numElem,group_data->set_num_dof[set].extent(0),maxnbasis,maxnumsteps[set]);
        newustage = View_Sc4("u stages",numElem,group_data->set_num_dof[set].extent(0),maxnbasis,maxnumstages[set]-1);
        if (group_data->requires_adjoint) {
          newphiprev = View_Sc4("phi previous",numElem,group_data->set_num_dof[set].extent(0),maxnbasis,maxnumsteps[set]);
          newphistage = View_Sc4("phi stages",numElem,group_data->set_num_dof[set].extent(0),maxnbasis,maxnumstages[set]-1);
        }
        else {
          newphiprev = View_Sc4("phi previous",1,1,1,1);
          newphistage = View_Sc4("phi stages",1,1,1,1);
        }
      }
      else {
        newuprev = View_Sc4("u previous",1,1,1,1);
        newustage = View_Sc4("u stages",1,1,1,1);
        newphiprev = View_Sc4("phi previous",1,1,1,1);
        newphistage = View_Sc4("phi stages",1,1,1,1);
      }
      sol_prev[set] = newuprev;
      sol_stage[set] = newustage;
      phi_prev[set] = newphiprev;
      phi_stage[set] = newphistage;
    
      // Storage for average solutions
      View_Sc3 newuavg;
      if (group_data->compute_sol_avg) {
        newuavg = View_Sc3("u spatial average",numElem,group_data->set_num_dof[set].extent(0),group_data->dimension);
      }
      else {
        newuavg = View_Sc3("u spatial average",1,1,1);
      }
      sol_avg[set] = newuavg;
    }
  }
  else {
    have_sols = false;
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void Group::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_,
                             const bool & allocate_storage) {
  vector<int> paramusebasis = pusebasis_;
  if (allocate_storage) {
    int maxnbasis = 0;
    for (size_type i=0; i<group_data->num_param_dof.extent(0); i++) {
      if (group_data->num_param_dof(i) > maxnbasis) {
        maxnbasis = group_data->num_param_dof(i);
      }
    }
    param = View_Sc3("param",numElem,group_data->num_param_dof.extent(0),maxnbasis);
  
    if (group_data->compute_sol_avg) {
      param_avg = View_Sc3("param",numElem,group_data->num_param_dof.extent(0), group_data->dimension);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void Group::setAuxUseBasis(vector<int> & ausebasis_,
                           const bool & allocate_storage) {
  auxusebasis = ausebasis_;
  if (allocate_storage) {
    int maxnbasis = 0;
    for (size_type i=0; i<group_data->num_aux_dof.extent(0); i++) {
      if (group_data->num_aux_dof(i) > maxnbasis) {
        maxnbasis = group_data->num_aux_dof(i);
      }
    }
    aux = View_Sc3("aux",numElem,group_data->num_aux_dof.extent(0),maxnbasis);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous step solutions
///////////////////////////////////////////////////////////////////////////////////////

void Group::resetPrevSoln(const size_t & set) {
  
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
    
    // copy current sol into first step
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

void Group::revertSoln(const size_t & set) {
  
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

void Group::resetStageSoln(const size_t & set) {
  
  
  if (group_data->requires_transient && sol.size() > set && sol_stage.size() > set) {
    auto csol = sol[set];
    auto csol_stage = sol_stage[set];
    
    if (csol_stage.extent(3) > 0) {
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
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the data stored in the previous stage solutions
///////////////////////////////////////////////////////////////////////////////////////

void Group::updateStageSoln(const size_t & set) {
  
  if (group_data->requires_transient && sol.size() > set && sol_stage.size() > set) {
    auto csol = sol[set];
    auto csol_stage = sol_stage[set];
    
    // add u into the current stage soln (done after stage solution is computed)
    auto stage = group_data->current_stage;
    if (stage < csol_stage.extent_int(3)) {
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
// Subgrid Plotting
///////////////////////////////////////////////////////////////////////////////////////

void Group::writeSubgridSolution(const std::string & filename) {
  //if (multiscale) {
  //  subgridModel->writeSolution(filename, subgrid_usernum);
  //}
}

///////////////////////////////////////////////////////////////////////////////////////
// Update the subgrid model
///////////////////////////////////////////////////////////////////////////////////////

void Group::updateSubgridModel(vector<Teuchos::RCP<SubGridModel> > & models) {
  
  /*
   wkset->update(ip,jacobian);
   int newmodel = udfunc->getSubgridModel(nodes, wkset, models.size());
   if (newmodel != subgrid_model_index) {
   // then we need:
   // 1. To add the macro-element to the new model
   // 2. Project the most recent solutions onto the new model grid
   // 3. Update this Group to use the new model
   
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
// 
///////////////////////////////////////////////////////////////////////////////////////

void Group::resetAdjPrev(const size_t & set, const ScalarT & val) {
  if (group_data->requires_adjoint && group_data->requires_transient) {
    deep_copy(adj_prev[set],val);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the storage required for the integration/basis info
///////////////////////////////////////////////////////////////////////////////////////

size_t Group::getVolumetricStorage() {
  size_t mystorage = 0;
  if (storeAll) {
    size_t scalarcost = sizeof(ScalarT); // 8 bytes per double
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

size_t Group::getFaceStorage() {
  size_t mystorage = 0;
  if (storeAll) {
    size_t scalarcost = sizeof(ScalarT); // 8 bytes per double
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

///////////////////////////////////////////////////////////////////////////////////////
// Recompute the wts or decompress them
///////////////////////////////////////////////////////////////////////////////////////

View_Sc2 Group::getWts() {
  View_Sc2 newwts;
  if (wts.extent(0) > 0) {
    if (wts.getHaveKey()) {
      auto vdata = wts.getView();
      auto vkey = wts.getKey();
      newwts = View_Sc2("temp wts",numElem, vdata.extent(1));
      parallel_for("grp wts decompress",
               RangePolicy<AssemblyExec>(0,numElem),
               KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type i=0; i<vdata.extent(1); ++i) {
          newwts(elem,i) = vdata(vkey(elem),i);
        }
      });
    }
    else {
      newwts = wts.getView();
    }
  }
  else {
    size_type numip = group_data->ref_ip.extent(0);
    newwts = View_Sc2("temp physical wts",numElem, numip);
    vector<View_Sc2> tip;
    if (have_nodes) {
      disc->getPhysicalIntegrationData(group_data, nodes, tip, newwts);
    }
    else {
      disc->getPhysicalIntegrationData(group_data, localElemID, tip, newwts);
    }
  }
  return newwts;
}

///////////////////////////////////////////////////////////////////////////////////////
// Recompute the wts or decompress them
///////////////////////////////////////////////////////////////////////////////////////

vector<View_Sc2> Group::getIntegrationPts() {
  vector<View_Sc2> newpts;
  for (size_t dim=0; dim<ip.size(); ++dim) {
    View_Sc2 pts;
    if (ip[dim].getHaveKey()) {
      auto vdata = ip[dim].getView();
      auto vkey = ip[dim].getKey();
      pts = View_Sc2("temp pts",numElem, vdata.extent(1));
      parallel_for("grp pts decompress",
               RangePolicy<AssemblyExec>(0,numElem),
               KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type i=0; i<vdata.extent(1); ++i) {
          pts(elem,i) = vdata(vkey(elem),i);
        }
      });
    }
    else {
      pts = ip[dim].getView();
    }
    newpts.push_back(pts);
  }
  return newpts;
}