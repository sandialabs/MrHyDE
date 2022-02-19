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

#ifndef MRHYDE_BOUNDARYGROUP_H
#define MRHYDE_BOUNDARYGROUP_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"
#include "groupMetaData.hpp"
#include "discretizationInterface.hpp"

#include <iostream>     
#include <iterator>     

namespace MrHyDE {
  
  class BoundaryGroup {
    
    typedef Tpetra::MultiVector<ScalarT,LO,GO,AssemblyNode> SG_MultiVector;
    typedef Teuchos::RCP<SG_MultiVector> SG_vector_RCP;
    
  public:
    
    BoundaryGroup() {} ;
    
    ~BoundaryGroup() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    BoundaryGroup(const Teuchos::RCP<GroupMetaData> & groupData_,
                  const DRV nodes_,
                  const Kokkos::View<LO*,AssemblyDevice> localID_,
                  LO & sideID_,
                  const int & sidenum_, const string & sidename_,
                  const int & groupID_,
                  Teuchos::RCP<DiscretizationInterface> & disc_,
                  const bool & storeAll_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeBasis(const bool & keepnodes);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void createHostLIDs();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeSizeNormals();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setWorkset(Teuchos::RCP<workset> & wkset_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setParams(LIDView paramLIDs_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Add the aux basis functions at the integration points.
    // This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void addAuxDiscretization(const vector<basis_RCP> & abasis_pointers,
                              const vector<DRV> & asideBasis,
                              const vector<DRV> & asideBasisGrad);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Add the aux variables
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void addAuxVars(const vector<string> & auxlist_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Define which basis each variable will use
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setUseBasis(vector<vector<int> > & usebasis_, const int & numsteps, const int & numstages);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Define which basis each discretized parameter will use
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Define which basis each aux variable will use
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setAuxUseBasis(vector<int> & ausebasis_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Update the workset
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateWorkset(const int & seedwhat, const bool & override_transient=false);
    
    void updateWorksetBasis();
      
    ///////////////////////////////////////////////////////////////////////////////////////
    // Map the coarse grid solution to the fine grid integration points
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeSoln(const int & seedwhat);
    
    void resetPrevSoln(const size_t & set);

    void revertSoln(const size_t & set);

    void resetStageSoln(const size_t & set);

    void updateStageSoln(const size_t & set);

    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute the contribution from this group to the global res, J, Jdot
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                       const bool & compute_jacobian, const bool & compute_sens,
                       const int & num_active_params, const bool & compute_disc_sens,
                       const bool & compute_aux_sens, const bool & store_adjPrev,
                       View_Sc3 res,
                       View_Sc3 local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT res
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateRes(const bool & compute_sens, View_Sc3 local_res);
        
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT J
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateJac(const bool & useadjoint, View_Sc3 local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jdot
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateJacDot(const bool & useadjoint, View_Sc3 local_Jdot);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jparam
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateParamJac(View_Sc3 local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jdot
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateParamJacDot(View_Sc3 local_Jdot);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jaux
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateAuxJac(View_Sc3 local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jdot
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateAuxJacDot(View_Sc3 local_Jdot);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute boundary contribution to the regularization and nodes located on the boundary
    ///////////////////////////////////////////////////////////////////////////////////////
    
    AD computeBoundaryRegularization(const vector<ScalarT> reg_constants, const vector<int> reg_types,
                                     const vector<int> reg_indices, const vector<string> reg_sides);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute flux and sensitivity wrt params
    ///////////////////////////////////////////////////////////////////////////////////////
    
    template<class ViewType>
    void computeFlux(ViewType u_kv, ViewType du_kv, ViewType dp_kv, View_Sc3 lambda,
                     const ScalarT & time, const int & side, const ScalarT & coarse_h,
                     const bool & compute_sens) {
      
      wkset->setTime(time);
      wkset->sidename = sidename;
      wkset->currentside = sidenum;
      wkset->numElem = numElem;
      //wkset->h = hsize;
      //this->updateWorksetBasis();
      
      // Currently hard coded to one physics sets
      vector<View_AD2> uvals = wkset->uvals;
      //auto param_AD = wkset->pvals;
      auto ulocal = u[0];
      auto currLIDs = LIDs[0];
      {
        Teuchos::TimeMonitor localtimer(*fluxGatherTimer);
        
        if (compute_sens) {
          for (size_t var=0; var<ulocal.extent(1); var++) {
            auto u_AD = uvals[var];
            auto offsets = subview(wkset->offsets,var,ALL());
            parallel_for("flux gather",
                         RangePolicy<AssemblyExec>(0,u_AD.extent(0)),
                         KOKKOS_LAMBDA (const int elem ) {
              for( size_t dof=0; dof<u_AD.extent(1); dof++ ) {
                u_AD(elem,dof) = AD(u_kv(currLIDs(elem,offsets(dof)),0));
              }
            });
          }
        }
        else {
          for (size_t var=0; var<ulocal.extent(1); var++) {
            auto u_AD = uvals[var];
            auto offsets = subview(wkset->offsets,var,ALL());
            parallel_for("flux gather",
                         RangePolicy<AssemblyExec>(0,ulocal.extent(0)),
                         KOKKOS_LAMBDA (const int elem ) {
              for( size_t dof=0; dof<u_AD.extent(1); dof++ ) {
#ifndef MrHyDE_NO_AD
                u_AD(elem,dof) = AD(maxDerivs, 0, u_kv(currLIDs(elem,offsets(dof)),0));
                for( size_t p=0; p<du_kv.extent(1); p++ ) {
                  u_AD(elem,dof).fastAccessDx(p) = du_kv(currLIDs(elem,offsets(dof)),p);
                }
#else
                u_AD(elem,dof) = u_kv(currLIDs(elem,offsets(dof)),0);
#endif
              }
            });
          }
        }
      }
      
      {
        Teuchos::TimeMonitor localtimer(*fluxWksetTimer);
        wkset->computeSolnSideIP(sidenum);//, u_AD, param_AD);
      }
      
      if (wkset->numAux > 0) {
        
        Teuchos::TimeMonitor localtimer(*fluxAuxTimer);
      
        auto numAuxDOF = groupData->numAuxDOF;
        
        for (size_type var=0; var<numAuxDOF.extent(0); var++) {
          auto abasis = auxside_basis[auxusebasis[var]];
          auto off = subview(auxoffsets,var,ALL());
          string varname = wkset->aux_varlist[var];
          auto local_aux = wkset->getSolutionField("aux "+varname+" side",false);
          Kokkos::deep_copy(local_aux,0.0);
          //auto local_aux = Kokkos::subview(wkset->local_aux_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
          auto localID = localElemID;
          auto varaux = subview(lambda,ALL(),var,ALL());
          parallel_for("flux aux",
                       RangePolicy<AssemblyExec>(0,localID.extent(0)),
                       KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type dof=0; dof<abasis.extent(1); ++dof) {
#ifndef MrHyDE_NO_AD
              AD auxval = AD(maxDerivs,off(dof), varaux(localID(elem),dof));
#else
              AD auxval = varaux(localID(elem),dof);
#endif
              for (size_type pt=0; pt<abasis.extent(2); ++pt) {
                local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
              }
            }
          });
        }
        
      }
      
      {
        Teuchos::TimeMonitor localtimer(*fluxEvalTimer);
        groupData->physics_RCP->computeFlux(0,groupData->myBlock);
      }
      
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    View_Sc2 getDirichlet(const size_t & set);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    View_Sc3 getMass(const size_t & set);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateData();
    
    size_t getStorage();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
      
    // Public data 
    Teuchos::RCP<GroupMetaData> groupData;
    Teuchos::RCP<workset> wkset;
    
    Kokkos::View<LO*,AssemblyDevice> localElemID;
    LO localSideID;
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation;
    
    // Geometry Information
    size_t numElem = 0; // default value ... used to check if proc. has elements on boundary
    int sidenum, groupID, wksetBID;
    DRV nodes;
    vector<View_Sc2> ip, normals, tangents;
    View_Sc2 wts;
    View_Sc1 hsize;
    bool storeAll, haveBasis;
    Kokkos::View<LO*,AssemblyDevice> basis_database_index;
    
    vector<Kokkos::View<int****,HostDevice> > sideinfo; // may need to move this to Assembly
    string sidename;
        
    // DOF information
    LIDView paramLIDs, auxLIDs;
    vector<LIDView> LIDs;
    
    Teuchos::RCP<DiscretizationInterface> disc;
    
    // Creating LIDs on host device for host assembly
    LIDView_host paramLIDs_host, auxLIDs_host;
    vector<LIDView_host> LIDs_host;
    
    vector<View_Sc3> u, phi;
    View_Sc3 param, aux;
    
    vector<View_Sc4> u_prev, phi_prev, u_stage, phi_stage; // (elem,var,numdof,step or stage)
    
    // basis information
    vector<View_Sc4> basis, basis_grad, basis_curl;
    vector<View_Sc3> basis_div;
    
    // Aux variable Information
    vector<string> auxlist;
    Kokkos::View<LO**,AssemblyDevice> auxoffsets;
    vector<int> auxusebasis;
    vector<basis_RCP> auxbasisPointers;
    vector<DRV> auxbasis, auxbasisGrad;
    vector<DRV> auxside_basis, auxside_basisGrad;
    vector<size_t> auxMIDs;
    Kokkos::View<size_t*,AssemblyDevice> auxMIDs_dev;
    
    vector<size_t> data_seed, data_seedindex;
    Kokkos::View<ScalarT**,AssemblyDevice> data;
    vector<ScalarT> data_distance;
    
    // Boundary groups do not have sensors
    
    // Profile timers
    Teuchos::RCP<Teuchos::Time> computeSolnSideTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeSolnSideIP()");
    Teuchos::RCP<Teuchos::Time> boundaryResidualTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeJacRes() - boundary residual");
    Teuchos::RCP<Teuchos::Time> jacobianFillTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeJacRes() - fill local Jacobian");
    Teuchos::RCP<Teuchos::Time> residualFillTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeJacRes() - fill local residual");
    Teuchos::RCP<Teuchos::Time> transientResidualTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeJacRes() - transient residual");
    Teuchos::RCP<Teuchos::Time> adjointResidualTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeJacRes() - adjoint residual");
    Teuchos::RCP<Teuchos::Time> fluxGatherTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeFlux - gather solution");
    Teuchos::RCP<Teuchos::Time> fluxWksetTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeFlux - update wkset");
    Teuchos::RCP<Teuchos::Time> fluxAuxTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeFlux - compute aux solution");
    Teuchos::RCP<Teuchos::Time> fluxEvalTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup::computeFlux - physics evaluation");
    Teuchos::RCP<Teuchos::Time> buildBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::BoundaryGroup - build basis");
    
  };
  
}

#endif