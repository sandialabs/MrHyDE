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
#include "compressedView.hpp"

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
    
    BoundaryGroup(const Teuchos::RCP<GroupMetaData> & group_data_,
                  const DRV nodes_,
                  const Kokkos::View<LO*,AssemblyDevice> localID_,
                  LO & sideID_,
                  const int & sidenum_, const string & sidename_,
                  const int & groupID_,
                  Teuchos::RCP<DiscretizationInterface> & disc_,
                  const bool & storeAll_);

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////

    void computeSize();

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////

    void initializeBasisIndex();

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
    
    void setWorkset(Teuchos::RCP<Workset> & wkset_);
    
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

    /**
     * @brief Define which basis each variable will use.
     * 
     * @todo Is that really true? Seems like this allocates scalar storage for the solution
     * and required solution history.
     * 
     * @param[in] usebasis_ Which basis should each variable use for each physics set
     * @param[in] maxnumsteps  Maximum number of BDF steps for each physics set
     * @param[in] maxnumstages Maximum number of RK stages for each physics set
     * 
     */   

    void setUseBasis(vector<vector<int> > & usebasis_, const vector<int> & maxnumsteps, 
                     const vector<int> & maxnumstages);
    
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
    
    void updateWorkset(const int & seedwhat, const int & seedindex=0,
                       const bool & override_transient=false);
    
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
                     const bool & compute_sens, const ScalarT & fluxwt,
                     bool & useTransientSol) {
      
      wkset->setTime(time);
      wkset->sidename = sidename;
      wkset->currentside = sidenum;
      wkset->numElem = numElem;
      //wkset->isOnSide = true;
      //wkset->h = hsize;
      //this->updateWorksetBasis();
      
      // Currently hard coded to one physics sets
      int set = 0;

      vector<View_AD2> sol_vals = wkset->sol_vals;
      //auto param_AD = wkset->pvals;
      auto ulocal = sol[set];
      auto currLIDs = LIDs[set];

      if (useTransientSol) { //wkset->isTransient) {
        //ScalarT dt = wkset->deltat;
        int stage = wkset->current_stage;
        auto b_A = wkset->butcher_A;
        auto b_b = wkset->butcher_b;
        auto BDF = wkset->BDF_wts;

        ScalarT one = 1.0;
        
        for (size_type var=0; var<ulocal.extent(1); var++ ) {
          size_t uindex = wkset->sol_vals_index[set][var];
          auto u_AD = sol_vals[uindex];
          auto off = subview(wkset->set_offsets[set],var,ALL());
          auto cu = subview(ulocal,ALL(),var,ALL());
          auto cu_prev = subview(sol_prev[set],ALL(),var,ALL(),ALL());
          auto cu_stage = subview(sol_stage[set],ALL(),var,ALL(),ALL());

          parallel_for("wkset transient sol seedwhat 1",
                       TeamPolicy<AssemblyExec>(cu.extent(0), Kokkos::AUTO, VectorSize),
                       KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
            int elem = team.league_rank();
            ScalarT beta_u;//, beta_t;
            ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
            //ScalarT timewt = one/dt/b_b(stage);
            //ScalarT alpha_t = BDF(0)*timewt;

            for (size_type dof=team.team_rank(); dof<u_AD.extent(1); dof+=team.team_size() ) {
            
              // Seed the stage solution
#ifndef MrHyDE_NO_AD
              AD stageval = AD(maxDerivs,0,cu(elem,dof));
              for( size_t p=0; p<du_kv.extent(1); p++ ) {
                stageval.fastAccessDx(p) = fluxwt*du_kv(currLIDs(elem,off(dof)),p);
              }
#else
              AD stageval = cu(elem,dof);
#endif
              // Compute the evaluating solution
              beta_u = (one-alpha_u)*cu_prev(elem,dof,0);
              for (int s=0; s<stage; s++) {
                beta_u += b_A(stage,s)/b_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
              }
              u_AD(elem,dof) = alpha_u*stageval+beta_u;
            
              // Compute the time derivative
              //beta_t = zero;
              //for (size_type s=1; s<BDF.extent(0); s++) {
              //  beta_t += BDF(s)*cu_prev(elem,dof,s-1);
              //}
              //beta_t *= timewt;
              //u_dot_AD(elem,dof) = alpha_t*stageval + beta_t;
            }
          
          });

        }
      }
      else {
        Teuchos::TimeMonitor localtimer(*fluxGatherTimer);
        
        if (compute_sens) {
          for (size_t var=0; var<ulocal.extent(1); var++) {
            auto u_AD = sol_vals[var];
            auto offsets = subview(wkset->offsets,var,ALL());
            parallel_for("flux gather",
                         RangePolicy<AssemblyExec>(0,ulocal.extent(0)),
                         KOKKOS_LAMBDA (const int elem ) {
              for( size_t dof=0; dof<u_AD.extent(1); dof++ ) {
                u_AD(elem,dof) = AD(u_kv(currLIDs(elem,offsets(dof)),0));
              }
            });
          }
        }
        else {
          for (size_t var=0; var<ulocal.extent(1); var++) {
            auto u_AD = sol_vals[var];
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
      
        auto numAuxDOF = group_data->num_aux_dof;
        
        for (size_type var=0; var<numAuxDOF.extent(0); var++) {
          auto abasis = auxside_basis[auxusebasis[var]];
          auto off = subview(auxoffsets,var,ALL());
          string varname = wkset->aux_varlist[var];
          auto local_aux = wkset->getSolutionField("aux "+varname,false);
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
              auxval.fastAccessDx(off(dof)) *= fluxwt;
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
        group_data->physics->computeFlux(0,group_data->my_block);
      }
      //wkset->isOnSide = false;
      
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
    Teuchos::RCP<GroupMetaData> group_data;
    Teuchos::RCP<Workset> wkset;
    
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
    Kokkos::View<LO*,AssemblyDevice> basis_index;
    
    vector<Kokkos::View<int****,HostDevice> > sideinfo; // may need to move this to Assembly
    string sidename;
        
    // DOF information
    LIDView paramLIDs, auxLIDs;
    vector<LIDView> LIDs;
    
    Teuchos::RCP<DiscretizationInterface> disc;
    
    // Creating LIDs on host device for host assembly
    LIDView_host paramLIDs_host, auxLIDs_host;
    vector<LIDView_host> LIDs_host;
    
    vector<View_Sc3> sol, phi;
    View_Sc3 param, aux;
    
    vector<View_Sc4> sol_prev, phi_prev, sol_stage, phi_stage; // (elem,var,numdof,step or stage)
    
    // basis information
    vector<CompressedView<View_Sc4>> basis, basis_grad, basis_curl;
    vector<CompressedView<View_Sc3>> basis_div;
    
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
    View_Sc2 data;
    vector<ScalarT> data_distance;
    View_Sc3 multidata;
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
