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

#include "group.hpp"
#include "physicsInterface.hpp"

#include <iostream>
#include <iterator>

using namespace MrHyDE;

Group::Group(const Teuchos::RCP<GroupMetaData> & group_data_,
           const DRV nodes_,
           const Kokkos::View<LO*,AssemblyDevice> localID_,
           Teuchos::RCP<DiscretizationInterface> & disc_,
           const bool & storeAll_) :
group_data(group_data_), localElemID(localID_), nodes(nodes_), disc(disc_)
{
  numElem = nodes.extent(0);
  
  active = true;
  storeAll = storeAll_;
  haveBasis = false;
  storeMass = true;
  
  // Even if we don't store the basis or integration info, we still store
  // the orientations since these are small, but expensive to recompute (for some reason)
  orientation = Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device>("kv to orients",numElem);
  disc->getPhysicalOrientations(group_data, localElemID,
                                orientation, true);
  
  size_type numip = group_data->ref_ip.extent(0);
  wts = View_Sc2("physical wts",numElem, numip);
  hsize = View_Sc1("physical hsize",numElem);

  disc->getPhysicalIntegrationData(group_data, nodes, ip, wts);
  
  this->computeSize();

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

void Group::computeSize() {

  // -------------------------------------------------
  // Compute the element sizes (h = vol^(1/dimension))
  // -------------------------------------------------
  size_t dimension = group_data->dimension;

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
  
  if (storeAll) {
    
    if (!haveBasis) {
      // Compute integration data and basis functions
      vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
      vector<View_Sc3> tbasis_div;
      disc->getPhysicalVolumetricBasis(group_data, nodes, orientation,
                                       tbasis, tbasis_grad, tbasis_curl,
                                       tbasis_div, tbasis_nodes, true);

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
          
          disc->getPhysicalFaceBasis(group_data, side, nodes, orientation,
                                    face_basis, face_basis_grad);
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
      nodes = DRV("empty nodes",1);
    }
  }
  else if (group_data->use_basis_database) {
    for (size_t i=0; i<group_data->database_basis.size(); ++i) {
      basis.push_back(CompressedView<View_Sc4>(group_data->database_basis[i],basis_index));
      basis_grad.push_back(CompressedView<View_Sc4>(group_data->database_basis_grad[i],basis_index));
      basis_div.push_back(CompressedView<View_Sc3>(group_data->database_basis_div[i],basis_index));
      basis_curl.push_back(CompressedView<View_Sc4>(group_data->database_basis_curl[i],basis_index));
    }
    if (!keepnodes) {
      nodes = DRV("empty nodes",1);
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::createHostLIDs() {
  
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
    
      auto LIDs_tmp = create_mirror_view(LIDs[set]);
      deep_copy(LIDs_tmp,LIDs[set]);
      
      LIDView_host currLIDs_host("LIDs on host",LIDs[set].extent(0), LIDs[set].extent(1));
      deep_copy(currLIDs_host,LIDs_tmp);
      LIDs_host[set] = currLIDs_host;
    //}
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::setWorkset(Teuchos::RCP<Workset<AD> > & wkset_) {
  
  wkset = wkset_;

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

void Group::setUseBasis(vector<vector<int> > & usebasis_, const vector<int> & maxnumsteps, const vector<int> & maxnumstages) {
  vector<vector<int> > usebasis = usebasis_;
  
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

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each discretized parameter will use
///////////////////////////////////////////////////////////////////////////////////////

void Group::setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_) {
  vector<int> paramusebasis = pusebasis_;
  
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

///////////////////////////////////////////////////////////////////////////////////////
// Define which basis each aux variable will use
///////////////////////////////////////////////////////////////////////////////////////

void Group::setAuxUseBasis(vector<int> & ausebasis_) {
  auxusebasis = ausebasis_;
  int maxnbasis = 0;
  for (size_type i=0; i<group_data->num_aux_dof.extent(0); i++) {
    if (group_data->num_aux_dof(i) > maxnbasis) {
      maxnbasis = group_data->num_aux_dof(i);
    }
  }
  aux = View_Sc3("aux",numElem,group_data->num_aux_dof.extent(0),maxnbasis);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void Group::updateWorkset(const int & seedwhat, const int & seedindex,
                          const bool & override_transient) {
    
  Teuchos::TimeMonitor localtimer(*computeSolnVolTimer);
  
  // Reset the residual and data in the workset
  wkset->reset();
  
  wkset->numElem = numElem;
  this->updateData();
  
  wkset->wts = wts;
  wkset->h = hsize;
  //wkset->basis_index = basis_index;

  wkset->setScalarField(ip[0],"x");
  
  if (ip.size() > 1) {
    wkset->setScalarField(ip[1],"y");
  }
  if (ip.size() > 2) {
    wkset->setScalarField(ip[2],"z");
  }

  // Update the integration info and basis in workset
  if (storeAll || group_data->use_basis_database) {
    wkset->basis = basis;
    wkset->basis_grad = basis_grad;
    wkset->basis_div = basis_div;
    wkset->basis_curl = basis_curl;
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricBasis(group_data, nodes, orientation,
                                    tbasis, tbasis_grad, tbasis_curl,
                                    tbasis_div, tbasis_nodes);

    vector<CompressedView<View_Sc4>> tcbasis, tcbasis_grad, tcbasis_curl;
    vector<CompressedView<View_Sc3>> tcbasis_div;
    for (size_t i=0; i<tbasis.size(); ++i) {
      tcbasis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      tcbasis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
      tcbasis_div.push_back(CompressedView<View_Sc3>(tbasis_div[i]));
      tcbasis_curl.push_back(CompressedView<View_Sc4>(tbasis_curl[i]));
    }
    wkset->basis = tcbasis;
    wkset->basis_grad = tcbasis_grad;
    wkset->basis_div = tcbasis_div;
    wkset->basis_curl = tcbasis_curl;
  }
  
  // Map the gathered solution to seeded version in workset
  if (group_data->requires_transient && !override_transient) {
    for (size_t set=0; set<group_data->num_sets; ++set) {
      wkset->computeSolnTransientSeeded(set, sol[set], sol_prev[set], sol_stage[set], seedwhat, seedindex);
    }
  }
  else { // steady-state
    for (size_t set=0; set<group_data->num_sets; ++set) {
      wkset->computeSolnSteadySeeded(set, sol[set], seedwhat);
    }
  }
  if (wkset->numParams > 0) {
    wkset->computeParamSteadySeeded(param, seedwhat);
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::computeSolAvg() {
  
  // THIS FUNCTION ASSUMES THAT THE WORKSET BASIS HAS BEEN UPDATED
  
  Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  
  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  auto cwts = wkset->wts;
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("Group sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });
  
  for (size_t set=0; set<sol_avg.size(); ++set) {
    
    // HGRAD vars
    vector<int> vars_HGRAD = wkset->vars_HGRAD[set];
    vector<string> varlist_HGRAD = wkset->varlist_HGRAD[set];
    for (size_t i=0; i<vars_HGRAD.size(); ++i) {
      auto sol = wkset->getSolutionField(varlist_HGRAD[i]);
      auto savg = subview(sol_avg[set],ALL(),vars_HGRAD[i],0);
      parallel_for("Group sol avg",
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
    vector<int> vars_HVOL = wkset->vars_HVOL[set];
    vector<string> varlist_HVOL = wkset->varlist_HVOL[set];
    for (size_t i=0; i<vars_HVOL.size(); ++i) {
      auto sol = wkset->getSolutionField(varlist_HVOL[i]);
      auto savg = subview(sol_avg[set],ALL(),vars_HVOL[i],0);
      parallel_for("Group sol avg",
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
    if (sol_avg[set].extent(2) > 1) { // 2D or 3D
      postfix.push_back("[y]");
    }
    if (sol_avg[set].extent(2) > 2) { // 3D
      postfix.push_back("[z]");
    }
    
    // HDIV vars
    vector<int> vars_HDIV = wkset->vars_HDIV[set];
    vector<string> varlist_HDIV = wkset->varlist_HDIV[set];
    for (size_t i=0; i<vars_HDIV.size(); ++i) {
      for (size_t j=0; j<postfix.size(); ++j) {
        auto sol = wkset->getSolutionField(varlist_HDIV[i]+postfix[j]);
        auto savg = subview(sol_avg[set],ALL(),vars_HDIV[i],j);
        parallel_for("Group sol avg",
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
    vector<int> vars_HCURL = wkset->vars_HCURL[set];
    vector<string> varlist_HCURL = wkset->varlist_HCURL[set];
    for (size_t i=0; i<vars_HCURL.size(); ++i) {
      for (size_t j=0; j<postfix.size(); ++j) {
        auto sol = wkset->getSolutionField(varlist_HCURL[i]+postfix[j]);
        auto savg = subview(sol_avg[set],ALL(),vars_HCURL[i],j);
        parallel_for("Group sol avg",
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
  }
  
  /*
  if (param_avg.extent(1) > 0) {
    View_AD4 psol = wkset->local_param;
    auto pavg = param_avg;

    parallel_for("Group param avg",
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
///////////////////////////////////////////////////////////////////////////////////////

void Group::computeSolutionAverage(const string & var, View_Sc2 csol) {
  
  Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  
  // Figure out which basis we need
  int index;
  wkset->isVar(var,index);
  
  CompressedView<View_Sc4> cbasis;
  auto cwts = wts;

  if (storeAll || group_data->use_basis_database) {
    cbasis = basis[wkset->usebasis[index]];
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricBasis(group_data, nodes, orientation,
                                     tbasis, tbasis_grad, tbasis_curl,
                                     tbasis_div, tbasis_nodes);
    cbasis = CompressedView<View_Sc4>(tbasis[wkset->usebasis[index]]);
  }
  
  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("Group sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });
  
  size_t set = wkset->current_set;
  auto scsol = subview(sol[set],ALL(),index,ALL());
  parallel_for("wkset soln ip HGRAD",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type dim=0; dim<cbasis.extent(3); ++dim) {
      ScalarT avgval = 0.0;
      for (size_type dof=0; dof<cbasis.extent(1); ++dof ) {
        for (size_type pt=0; pt<cbasis.extent(2); ++pt) {
          avgval += scsol(elem,dof)*cbasis(elem,dof,pt,dim)*cwts(elem,pt);
        }
      }
      csol(elem,dim) = avgval/avgwts(elem);
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void Group::computeParameterAverage(const string & var, View_Sc2 sol) {
  
  Teuchos::TimeMonitor localtimer(*computeSolAvgTimer);
  
  // Figure out which basis we need
  int index;
  wkset->isParameter(var,index);
  
  CompressedView<View_Sc4> cbasis;
  auto cwts = wts;

  if (storeAll || group_data->use_basis_database) {
    cbasis = basis[wkset->paramusebasis[index]];
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad, tbasis_curl, tbasis_nodes;
    vector<View_Sc3> tbasis_div;
    disc->getPhysicalVolumetricBasis(group_data, nodes, orientation,
                                     tbasis, tbasis_grad, tbasis_curl,
                                     tbasis_div, tbasis_nodes);
    cbasis = CompressedView<View_Sc4>(tbasis[wkset->paramusebasis[index]]);
  }
  
  // Compute the average weight, i.e., the size of each elem
  // May consider storing this
  View_Sc1 avgwts("elem size",cwts.extent(0));
  parallel_for("Group sol avg",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    ScalarT avgwt = 0.0;
    for (size_type pt=0; pt<cwts.extent(1); pt++) {
      avgwt += cwts(elem,pt);
    }
    avgwts(elem) = avgwt;
  });
  
  auto csol = subview(param,ALL(),index,ALL());
  parallel_for("wkset soln ip HGRAD",
               RangePolicy<AssemblyExec>(0,cwts.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type dim=0; dim<cbasis.extent(3); ++dim) {
      ScalarT avgval = 0.0;
      for (size_type dof=0; dof<cbasis.extent(1); ++dof ) {
        for (size_type pt=0; pt<cbasis.extent(2); ++pt) {
          avgval += csol(elem,dof)*cbasis(elem,dof,pt,dim)*cwts(elem,pt);
        }
      }
      sol(elem,dim) = avgval/avgwts(elem);
    }
  });
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void Group::updateWorksetFace(const size_t & facenum) {
  
  // IMPORANT NOTE: This function assumes that face contributions are computing IMMEDIATELY after the
  // volumetric contributions, which implies that the seeded solution in the workset is already
  // correct for this Group.  There is currently no use case where this assumption is false.
  
  Teuchos::TimeMonitor localtimer(*computeSolnFaceTimer);
  
  wkset->wts_side = wts_face[facenum];
  wkset->h = hsize;
  //wkset->basis_index = basis_index;

  wkset->setScalarField(ip_face[facenum][0],"x");
  wkset->setScalarField(normals_face[facenum][0],"n[x]");
  if (ip_face[facenum].size() > 1) {
    wkset->setScalarField(ip_face[facenum][1],"y");
    wkset->setScalarField(normals_face[facenum][1],"n[y]");
  }
  if (ip_face[facenum].size() > 2) {
    wkset->setScalarField(ip_face[facenum][2],"z");
    wkset->setScalarField(normals_face[facenum][2],"n[z]");
  }
    
  // Update the face integration points and basis in workset
  if (storeAll || group_data->use_basis_database) {
    wkset->basis_side = basis_face[facenum];
    wkset->basis_grad_side = basis_grad_face[facenum];
  }
  else {
    vector<View_Sc4> tbasis, tbasis_grad;
  
    disc->getPhysicalFaceBasis(group_data, facenum, nodes, orientation,
                               tbasis, tbasis_grad);
    vector<CompressedView<View_Sc4>> tcbasis, tcbasis_grad;
    for (size_t i=0; i<tbasis.size(); ++i) {
      tcbasis.push_back(CompressedView<View_Sc4>(tbasis[i]));
      tcbasis_grad.push_back(CompressedView<View_Sc4>(tbasis_grad[i]));
    }
    wkset->basis_side = tcbasis;
    wkset->basis_grad_side = tcbasis_grad;
  }
  
  wkset->resetSolutionFields();
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Map the AD degrees of freedom to integration points
///////////////////////////////////////////////////////////////////////////////////////

void Group::computeAuxSolnFaceIP(const size_t & facenum) {

  Teuchos::TimeMonitor localtimer(*computeSolnFaceTimer);
  this->updateWorksetFace(facenum);

}

///////////////////////////////////////////////////////////////////////////////////////
// Reset the data stored in the previous step solutions
///////////////////////////////////////////////////////////////////////////////////////

void Group::resetPrevSoln(const size_t & set) {
  
  if (group_data->requires_transient) {
    auto csol = sol[set];
    auto csol_prev = sol_prev[set];
    
    // shift previous step solns
    if (csol_prev.extent(3)>1) {
      parallel_for("Group reset prev soln 1",
                   TeamPolicy<AssemblyExec>(csol_prev.extent(0), Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(csol_prev.extent(0), Kokkos::AUTO, VectorSize),
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
  
  if (group_data->requires_transient) {
    auto csol = sol[set];
    auto csol_prev = sol_prev[set];
    
    // copy current u into first step
    parallel_for("Group reset prev soln 2",
                 TeamPolicy<AssemblyExec>(csol_prev.extent(0), Kokkos::AUTO, VectorSize),
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
  
  if (group_data->requires_transient) {
    auto csol = sol[set];
    auto csol_stage = sol_stage[set];
    
    if (csol_stage.extent(3) > 0) {
      parallel_for("Group reset stage 1",
                   TeamPolicy<AssemblyExec>(csol_stage.extent(0), Kokkos::AUTO, VectorSize),
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
  
  if (group_data->requires_transient) {
    auto csol = sol[set];
    auto csol_stage = sol_stage[set];
    
    // add u into the current stage soln (done after stage solution is computed)
    auto stage = wkset->current_stage;
    if (stage < csol_stage.extent_int(3)) {
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(csol_stage.extent(0), Kokkos::AUTO, VectorSize),
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
// Compute the contribution from this Group to the global res, J, Jdot
///////////////////////////////////////////////////////////////////////////////////////

void Group::computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
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
  int seedindex = 0;

  //////////////////////////////////////////////////////////////
  // Compute res and J=dF/du
  //////////////////////////////////////////////////////////////
  
  // Volumetric contribution
  if (assemble_volume_terms) {
    Teuchos::TimeMonitor localtimer(*volumeResidualTimer);
    if (group_data->multiscale) {
      this->updateWorkset(seedwhat,seedindex);
      subgridModels[subgrid_model_index]->subgridSolver(sol[0], sol_prev[0], phi[0], wkset->time, isTransient, isAdjoint,
                                            compute_jacobian, compute_sens, num_active_params,
                                            compute_disc_sens, compute_aux_sens,
                                            *wkset, subgrid_usernum, 0,
                                            subgradient, store_adjPrev);
      fixJacDiag = true;
    }
    else {
      this->updateWorkset(seedwhat,seedindex);
      group_data->physics->volumeResidual(wkset->current_set,group_data->my_block);
    }
  }
  
  // Edge/face contribution
  if (assemble_face_terms) {
    Teuchos::TimeMonitor localtimer(*faceResidualTimer);
    if (group_data->multiscale) {
      // do nothing
    }
    else {
      for (size_t s=0; s<group_data->num_sides; s++) {
        this->updateWorksetFace(s);
        group_data->physics->faceResidual(wkset->current_set,group_data->my_block);
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

void Group::updateRes(const bool & compute_sens, View_Sc3 local_res) {
  
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = group_data->num_dof;
  
  if (compute_sens) {
#ifndef MrHyDE_NO_AD
    parallel_for("Group res sens",
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
    parallel_for("Group res",
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

void Group::updateAdjointRes(const bool & compute_jacobian, const bool & isTransient,
                            const bool & compute_aux_sens, const bool & store_adjPrev,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  
  // Update residual (adjoint mode)
  // Adjoint residual: -dobj/du - J^T * phi + 1/dt*M^T * phi_prev
  // J = 1/dtM + A
  // adj_prev stores 1/dt*M^T * phi_prev where M is evaluated at appropriate time
  
  // TMW: This will not work on a GPU
  auto offsets = wkset->offsets;
  auto numDOF = group_data->num_dof;
  
  size_t set = wkset->current_set;
  auto cphi = phi[set];
  
  if (compute_jacobian) {
    parallel_for("Group adjust adjoint jac",
                 RangePolicy<AssemblyExec>(0,local_res.extent(0)),
                 KOKKOS_LAMBDA (const size_type e ) {
      for (size_type n=0; n<numDOF.extent(0); n++) {
        for (int j=0; j<numDOF(n); j++) {
          for (size_type m=0; m<numDOF.extent(0); m++) {
            for (int k=0; k<numDOF(m); k++) {
              local_res(e,offsets(n,j),0) += -local_J(e,offsets(n,j),offsets(m,k))*cphi(e,m,k);
            }
          }
        }
      }
    });
    
    if (isTransient) {
      
      auto aprev = adj_prev[set];
      
      // Previous step contributions for the residual
      parallel_for("Group adjust transient adjoint jac",
                   RangePolicy<AssemblyExec>(0,local_res.extent(0)),
                   KOKKOS_LAMBDA (const size_type e ) {
        for (size_type n=0; n<numDOF.extent(0); n++) {
          for (int j=0; j<numDOF(n); j++) {
            local_res(e,offsets(n,j),0) += -aprev(e,offsets(n,j),0);
          }
        }
      });
      /*
      // Previous stage contributions for the residual
      if (adj_stage_prev.extent(2) > 0) {
        parallel_for("Group adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
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
        parallel_for("Group adjust transient adjoint jac",
                     RangePolicy<AssemblyExec>(0,aprev.extent(0)),
                     KOKKOS_LAMBDA (const size_type e ) {
          for (size_type step=1; step<aprev.extent(2); step++) {
            for (size_type n=0; n<aprev.extent(1); n++) {
              aprev(e,n,step-1) = aprev(e,n,step);
            }
          }
          size_type numsteps = aprev.extent(2);
          for (size_type n=0; n<aprev.extent(1); n++) {
            aprev(e,n,numsteps-1) = 0.0;
          }
        });
        
        // Sum new contributions into vectors
        int seedwhat = 2; // 2 for J wrt previous step solutions
        for (size_type step=0; step<sol_prev[set].extent(3); step++) {
          this->updateWorkset(seedwhat,step);
          group_data->physics->volumeResidual(set,group_data->my_block);
          Kokkos::View<ScalarT***,AssemblyDevice> Jdot("temporary fix for transient adjoint",
                                                       local_J.extent(0), local_J.extent(1), local_J.extent(2));
          this->updateJac(true, Jdot);
          
          auto cadj = subview(aprev,ALL(), ALL(), step);
          parallel_for("Group adjust transient adjoint jac 2",
                       RangePolicy<AssemblyExec>(0,Jdot.extent(0)),
                       KOKKOS_LAMBDA (const size_type e ) {
            for (size_type n=0; n<numDOF.extent(0); n++) {
              for (int j=0; j<numDOF(n); j++) {
                ScalarT aPrev = 0.0;
                for (size_type m=0; m<numDOF.extent(0); m++) {
                  for (int k=0; k<numDOF(m); k++) {
                    aPrev += Jdot(e,offsets(n,j),offsets(m,k))*cphi(e,m,k);
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
        parallel_for("Group adjust transient adjoint jac",RangePolicy<AssemblyExec>(0,adj_stage_prev.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
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
        for (size_type stage=0; stage<sol_prev.extent(3); stage++) {
          wkset->computeSolnTransientSeeded(u, sol_prev, sol_stage, seedwhat, stage);
          wkset->computeParamVolIP(param, seedwhat);
          this->computeSolnVolIP();
          
          wkset->resetResidual();
          
          group_data->physics->volumeResidual(group_data->myBlock);
          Kokkos::View<ScalarT***,AssemblyDevice> Jdot("temporary fix for transient adjoint",
                                                       local_J.extent(0), local_J.extent(1), local_J.extent(2));
          this->updateJac(true, Jdot);
          
          auto cadj = subview(adj_stage_prev,ALL(), ALL(), stage);
          parallel_for("Group adjust transient adjoint jac 2",RangePolicy<AssemblyExec>(0,local_res.extent(0)), KOKKOS_LAMBDA (const size_type e ) {
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

void Group::updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = group_data->num_dof;
  
  if (useadjoint) {
    parallel_for("Group J adj",
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
    parallel_for("Group J",
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

void Group::fixDiagJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                      Kokkos::View<ScalarT***,AssemblyDevice> local_res) {
  
  auto offsets = wkset->offsets;
  auto numDOF = group_data->num_dof;

  using namespace std;

  parallel_for("Group fix diag",
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

void Group::updateParamJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
#ifndef MrHyDE_NO_AD
  auto paramoffsets = wkset->paramoffsets;
  auto numParamDOF = group_data->num_param_dof;
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto numDOF = group_data->num_dof;
  
  parallel_for("Group param J",
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

void Group::updateAuxJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J) {
  
#ifndef MrHyDE_NO_AD
  auto res_AD = wkset->res;
  auto offsets = wkset->offsets;
  auto aoffsets = auxoffsets;
  auto numDOF = group_data->num_dof;
  auto numAuxDOF = group_data->num_aux_dof;
  
  parallel_for("Group aux J",
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

View_Sc2 Group::getInitial(const bool & project, const bool & isAdjoint) {
  
  size_t set = wkset->current_set;
  View_Sc2 initialvals("initial values",numElem,LIDs[set].extent(1));
  this->updateWorkset(0,0);
  
  auto offsets = wkset->offsets;
  auto numDOF = group_data->num_dof;
  auto cwts = wts;
  
  if (project) { // works for any basis
    auto initialip = group_data->physics->getInitial(ip, set,
                                                        group_data->my_block,
                                                        project, wkset);

    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = basis[wkset->usebasis[n]];
      auto off = subview(offsets, n, ALL());
      string btype = wkset->basis_types[wkset->usebasis[n]];
      if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
        auto initvar = subview(initialip, ALL(), n, ALL(), 0);
        parallel_for("Group init project",
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
      else if (btype.substr(0,5) == "HCURL" || btype.substr(0,4) == "HDIV") {
        auto initvar = subview(initialip, ALL(), n, ALL(), ALL());
        parallel_for("Group init project",
                     TeamPolicy<AssemblyExec>(initvar.extent(0), Kokkos::AUTO),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type dof=team.team_rank(); dof<cbasis.extent(1); dof+=team.team_size() ) {
            for (size_type pt=0; pt<cwts.extent(1); pt++ ) {
              for (size_type dim=0; dim<cbasis.extent(3); dim++ ) {
                initialvals(elem,off(dof)) += initvar(elem,pt,dim)*cbasis(elem,dof,pt,dim)*cwts(elem,pt);
              }
            }
          }
        });
      }
      
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
    
    auto initialnodes = group_data->physics->getInitial(vnodes, set,
                                                          group_data->my_block,
                                                          project,
                                                          wkset);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto off = subview( offsets, n, ALL());
      auto initvar = subview(initialnodes, ALL(), n, ALL(), 0);
      parallel_for("Group init project",
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
// Get the initial condition on the faces
///////////////////////////////////////////////////////////////////////////////////////

View_Sc2 Group::getInitialFace(const bool & project) {
  
  size_t set = wkset->current_set;
  View_Sc2 initialvals("initial values",numElem,LIDs[set].extent(1)); // TODO is this too big?
  this->updateWorkset(0,0); // TODO not sure if this is necessary

  auto offsets = wkset->offsets;
  auto numDOF = group_data->num_dof;

  // loop over faces of the reference element
  for (size_t face=0; face<group_data->num_sides; face++) {

    // get basis functions, weights, etc. for that face
    this->updateWorksetFace(face);
    auto cwts = wkset->wts_side; // face weights get put into wts_side after update
    // get data from IC
    auto initialip = group_data->physics->getInitialFace(ip_face[face], set,
                                                           group_data->my_block,
                                                           project,
                                                           wkset);
    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = wkset->basis_side[wkset->usebasis[n]]; // face basis gets put here after update
      auto off = subview(offsets, n, ALL());
      auto initvar = subview(initialip, ALL(), n, ALL());
      // loop over mesh elements
      parallel_for("Group init project",
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
  
  return initialvals;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

CompressedView<View_Sc3> Group::getMass() {
  
  size_t set = wkset->current_set;
  View_Sc3 mass_view("local mass",numElem, LIDs[set].extent(1), LIDs[set].extent(1));
  CompressedView<View_Sc3> mass(mass_view);
  auto offsets = wkset->offsets;
  auto numDOF = group_data->num_dof;
  auto cwts = wts;
  
  vector<CompressedView<View_Sc4>> tbasis;
  if (storeAll) {
    tbasis = basis;
  }
  else { // goes through this more than once, but really shouldn't be used much anyways
    vector<View_Sc4> tmpbasis,tmpbasis_grad, tmpbasis_curl, tmpbasis_nodes;
    vector<View_Sc3> tmpbasis_div;
    disc->getPhysicalVolumetricBasis(group_data, nodes, orientation,
                                    tmpbasis, tmpbasis_grad, tmpbasis_curl,
                                    tmpbasis_div, tmpbasis_nodes);
    for (size_t i=0; i<tmpbasis.size(); ++i) {
      tbasis.push_back(CompressedView<View_Sc4>(tmpbasis[i]));
    }
  }
  
  for (size_type n=0; n<numDOF.extent(0); n++) {
    auto cbasis = tbasis[wkset->usebasis[n]];
    string btype = wkset->basis_types[wkset->usebasis[n]];
    auto off = subview(offsets,n,ALL());
    if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
      parallel_for("Group get mass",
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
      parallel_for("Group get mass",
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

CompressedView<View_Sc3> Group::getWeightedMass(vector<ScalarT> & masswts) {
  
  size_t set = wkset->current_set;
  auto numDOF = group_data->num_dof;
  
  View_Sc3 mass_view("local mass",numElem, LIDs[set].extent(1), LIDs[set].extent(1));
  CompressedView<View_Sc3> mass;

  if (group_data->use_mass_database) {
    mass = CompressedView<View_Sc3>(group_data->database_mass[set], basis_index);
  }
  else {
    auto cwts = wts;
    auto offsets = wkset->offsets;
    vector<CompressedView<View_Sc4>> tbasis;
    mass = CompressedView<View_Sc3>(mass_view);

    if (storeAll || group_data->use_basis_database) {
      tbasis = basis;
    }
    else {
      vector<View_Sc4> tmpbasis;
      disc->getPhysicalVolumetricBasis(group_data, nodes, orientation,
                                       tmpbasis);
      for (size_t i=0; i<tmpbasis.size(); ++i) {
        tbasis.push_back(CompressedView<View_Sc4>(tmpbasis[i]));
      }
    }

    for (size_type n=0; n<numDOF.extent(0); n++) {
      auto cbasis = tbasis[wkset->usebasis[n]];
    
      string btype = wkset->basis_types[wkset->usebasis[n]];
      auto off = subview(offsets,n,ALL());
      ScalarT mwt = masswts[n];
    
      if (btype.substr(0,5) == "HGRAD" || btype.substr(0,4) == "HVOL") {
        parallel_for("Group get mass",
                     RangePolicy<AssemblyExec>(0,mass.extent(0)),
                     KOKKOS_LAMBDA (const size_type e ) {
          for (size_type i=0; i<cbasis.extent(1); i++ ) {
            for (size_type j=0; j<cbasis.extent(1); j++ ) {
              for (size_type k=0; k<cbasis.extent(2); k++ ) {
                mass(e,off(i),off(j)) += cbasis(e,i,k,0)*cbasis(e,j,k,0)*cwts(e,k)*mwt;
              }
            }
          }
        });
      }
      else if (btype.substr(0,4) == "HDIV" || btype.substr(0,5) == "HCURL") {
        parallel_for("Group get mass",
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
  
  }
  
  if (storeMass) {
    // This assumes they are computed in order
    local_mass.push_back(mass);
  }

  return mass;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the mass matrix
///////////////////////////////////////////////////////////////////////////////////////

CompressedView<View_Sc3> Group::getMassFace() {
  
  size_t set = wkset->current_set;
  
  View_Sc3 mass_view("local mass",numElem, LIDs[set].extent(1), LIDs[set].extent(1));
  CompressedView<View_Sc3> mass(mass_view);

  auto offsets = wkset->offsets;
  auto numDOF = group_data->num_dof;

  // loop over faces of the reference element
  for (size_t face=0; face<group_data->num_sides; face++) {

    this->updateWorksetFace(face);
    auto cwts = wkset->wts_side; // face weights get put into wts_side after update
    for (size_type n=0; n<numDOF.extent(0); n++) {
      
      auto cbasis = wkset->basis_side[wkset->usebasis[n]]; // face basis put here after update
      string btype = wkset->basis_types[wkset->usebasis[n]]; // TODO does this work in general?
      auto off = subview(offsets,n,ALL());

      if (btype.substr(0,5) == "HFACE") {
        // loop over mesh elements
        parallel_for("Group get mass",
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
      else { 
        // TODO ERROR 
        cout << "Group::getMassFace() called with non-HFACE basis type!" << endl;
      }
    }
  }
  return mass;
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
// Pass the Group data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void Group::updateData() {
  
  // hard coded for what I need it for right now
  if (group_data->have_phi) {
    wkset->have_rotation_phi = true;
    wkset->rotation_phi = data;
    wkset->allocateRotations();
  }
  else if (group_data->have_rotation) {
    wkset->have_rotation = true;
    wkset->allocateRotations();
    auto rot = wkset->rotation;
    
    parallel_for("Group update data",
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

    //KokkosTools::print(wkset->rotation);
    //KokkosTools::print(rot);

  }
  else if (group_data->have_extra_data) {
    wkset->extra_data = data;
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Pass the Group data to the wkset
///////////////////////////////////////////////////////////////////////////////////////

void Group::resetAdjPrev(const size_t & set, const ScalarT & val) {
  if (group_data->requires_adjoint && group_data->requires_transient) {
    deep_copy(adj_prev[set],val);
  }
}


///////////////////////////////////////////////////////////////////////////////////////
// Get the solution at the nodes
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT***,AssemblyDevice> Group::getSolutionAtNodes(const int & var) {
  
  Teuchos::TimeMonitor nodesoltimer(*computeNodeSolTimer);
  size_t set = wkset->current_set;
  
  int bnum = wkset->usebasis[var];
  auto cbasis = basis_nodes[bnum];
  Kokkos::View<ScalarT***,AssemblyDevice> nodesol("solution at nodes",
                                                  cbasis.extent(0), cbasis.extent(2), group_data->dimension);
  auto uvals = subview(sol[set], ALL(), var, ALL());
  parallel_for("Group node sol",
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
