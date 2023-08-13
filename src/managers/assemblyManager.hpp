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

/** \file   analysisManager.hpp
 \brief  Contains all of the assembly routines in MrHyDE.  Also creates the elements groups and the worksets.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_ASSEMBLY_MANAGER_H
#define MRHYDE_ASSEMBLY_MANAGER_H

#include "trilinos.hpp"
#include "Panzer_DOFManager.hpp"

#include "preferences.hpp"
#include "groupMetaData.hpp"
#include "group.hpp"
#include "boundaryGroup.hpp"
#include "workset.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "parameterManager.hpp"
#include "multiscaleManager.hpp"
#include "functionManager.hpp"

namespace MrHyDE {
  
  /** \class  MrHyDE::AssemblyManager
   \brief  Provides the functionality for the MrHyDE-specific assembly routines for both implicit and explicit methods.
   */
  
  template< class Node>
  class AssemblyManager {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
    typedef Tpetra::CrsGraph<LO,GO,Node>            LA_CrsGraph;
    typedef Tpetra::Export<LO, GO, Node>            LA_Export;
    typedef Tpetra::Import<LO, GO, Node>            LA_Import;
    typedef Tpetra::Map<LO, GO, Node>               LA_Map;
    typedef Tpetra::Operator<ScalarT,LO,GO,Node>    LA_Operator;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;
    typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
    typedef typename Node::device_type              LA_device;
    typedef typename Node::memory_space             LA_mem;
    
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    AssemblyManager() {};
    
    ~AssemblyManager() {};
    
    AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_,
                    Teuchos::RCP<Teuchos::ParameterList> & settings,
                    Teuchos::RCP<MeshInterface> & mesh_,
                    Teuchos::RCP<DiscretizationInterface> & disc_,
                    Teuchos::RCP<PhysicsInterface> & phys_,
                    Teuchos::RCP<ParameterManager<Node> > & params_);
    
    
    // ========================================================================================
    // ========================================================================================
    
    void createFixedDOFs();

    // ========================================================================================
    // ========================================================================================
    
    void createGroups();
    
    void allocateGroupStorage();
      
    // ========================================================================================
    // ========================================================================================
    
    void createWorkset();
    
    // ========================================================================================
    // ========================================================================================
    
    void updateJacDBC(matrix_RCP & J, const std::vector<std::vector<GO> > & dofs,
                      const size_t & block, const bool & compute_disc_sens);
    
    void updateJacDBC(matrix_RCP & J, const std::vector<LO> & dofs, const bool & compute_disc_sens);
    
    
    void setDirichlet(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                      const ScalarT & time, const bool & lumpmass=false);
    
    // ========================================================================================
    // ========================================================================================
    
    void setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                    const bool & lumpmass=false, const ScalarT & scale = 1.0);
    
    void setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                    const bool & lumpmass, const ScalarT & scale,
                    const size_t & block, const size_t & groupblock);
    
    void setInitial(const size_t & set, vector_RCP & initial, const bool & useadjoint);

    // TODO BWR -- finish when appropriate
    /* @brief Create the mass matrix and RHS for an L2 projection of the initial
     * condition over the faces.
     *
     * @param[inout] rhs  RHS vector
     * @param[inout] mass Mass matrix
     * @param[in] lumpmass Bool indicating if a lumped mass matrix approximation is requested
     *
     * @details The current use case is for projection the coarse-scale initial condition on the 
     * mesh skeleton (HFACE).
     *
     * @warning BWR -- Under development, I am trying to take things from setDirichlet and setInitial.
     * I think some combination of the two should work, but need to better understand.
     */
    
    void setInitialFace(const size_t & set, vector_RCP & rhs, matrix_RCP & mass,const bool & lumpmass=false);

    void getWeightedMass(const size_t & set, matrix_RCP & mass, vector_RCP & massdiag);
    
    void getWeightVector(const size_t & set, vector_RCP & wts);
    
    // ========================================================================================
    // ========================================================================================
    
    void assembleJacRes(const size_t & set, vector_RCP & u, vector_RCP & phi,
                        const bool & compute_jacobian, const bool & compute_sens,
                        const bool & compute_disc_sens,
                        vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                        const ScalarT & current_time, const bool & useadjoint,
                        const bool & store_adjPrev,
                        const int & num_active_params, vector_RCP & Psol,
                        const bool & is_final_time,
                        const ScalarT & deltat);
    
    
    void assembleJacRes(const size_t & set, const bool & compute_jacobian, const bool & compute_sens,
                        const bool & compute_disc_sens,
                        vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                        const ScalarT & current_time, const bool & useadjoint,
                        const bool & store_adjPrev,
                        const int & num_active_params,
                        const bool & is_final_time, const int & block,
                        const ScalarT & deltat);
    
    void assembleRes(const size_t & set, vector_RCP & u, vector_RCP & phi,
                        const bool & compute_jacobian, const bool & compute_sens,
                        const bool & compute_disc_sens,
                        vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                        const ScalarT & current_time, const bool & useadjoint,
                        const bool & store_adjPrev,
                        const int & num_active_params, vector_RCP & Psol,
                        const bool & is_final_time,
                        const ScalarT & deltat);
    
    void assembleRes(const size_t & set, const bool & compute_jacobian, const bool & compute_sens,
                        const bool & compute_disc_sens,
                        vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                        const ScalarT & current_time, const bool & useadjoint,
                        const bool & store_adjPrev,
                        const int & num_active_params,
                        const bool & is_final_time, const int & block,
                        const ScalarT & deltat);
    
  
    // ========================================================================================
    //
    // ========================================================================================
    
    void dofConstraints(const size_t & set, matrix_RCP & J, vector_RCP & res, const ScalarT & current_time,
                        const bool & compute_jacobian, const bool & compute_disc_sens);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    void resetPrevSoln(const size_t & set);
    
    void revertSoln(const size_t & set);
    
    void resetStageSoln(const size_t & set);
    
    void updateStage(const int & stage, const ScalarT & current_time, const ScalarT & deltat);
    
    void updateStageSoln(const size_t & set);
    
    void updatePhysicsSet(const size_t & set);
    
    void updateTimeStep(const int & timestep);
    
    // ========================================================================================
    // Gather 
    // ========================================================================================
    
    void performGather(const size_t & set, const vector_RCP & vec, const int & type, const size_t & index);
    
    template<class ViewType>
    void performGather(const size_t & set, ViewType vec_dev, const int & type);
        
    template<class ViewType>
    void performBoundaryGather(const size_t & set, ViewType vec_dev, const int & type);
    
    // ========================================================================================
    // Scatter 
    // ========================================================================================
    
    template<class MatType, class LocalViewType, class LIDViewType>
    void scatterJac(const size_t & set, MatType J_kcrs, LocalViewType local_J,
                    LIDViewType LIDs, LIDViewType paramLIDs,
                    const bool & compute_disc_sens);

    template<class VecViewType, class LocalViewType, class LIDViewType>
    void scatterRes(VecViewType res_view,
                    LocalViewType local_res, LIDViewType LIDs);

    template<class MatType, class VecViewType, class LIDViewType>
    void scatter(const size_t & set,MatType J_kcrs, VecViewType res_view,
                 LIDViewType LIDs, LIDViewType paramLIDs,
                 const int & block,
                 const bool & compute_jacobian,
                 const bool & compute_sens,
                 const bool & compute_disc_sens,
                 const bool & isAdjoint);
    
    template<class VecViewType, class LIDViewType>
    void scatterRes(const size_t & set, VecViewType res_view,
                 LIDViewType LIDs, 
                 const int & block);
    
    // Computes y = M*x
    void applyMassMatrixFree(const size_t & set, vector_RCP & x, vector_RCP & y);
    

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    void buildDatabase(const size_t & block);

    void writeVolumetricData(const size_t & block, vector<vector<size_t>> & all_orients);

    void identifyVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users);

    void identifyBoundaryDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_boundary_users);
    
    void buildVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users);

    void buildBoundaryDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_boundary_users);
    
    void finalizeFunctions();

    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute flux and sensitivity wrt params
    ///////////////////////////////////////////////////////////////////////////////////////
    
    template<class ViewType>
    void computeFlux(const int & block, const int & grp, ViewType u_kv, 
                     ViewType du_kv, ViewType dp_kv, View_Sc3 lambda,
                     const ScalarT & time, const int & side, const ScalarT & coarse_h,
                     const bool & compute_sens, const ScalarT & fluxwt,
                     bool & useTransientSol) {

      AD dummyval = 0.0;
      this->computeFluxEvalT(dummyval, block, grp, u_kv, du_kv, dp_kv, lambda, time, side, coarse_h,
                             compute_sens, fluxwt, useTransientSol);

    }

    template<class ViewType, class EvalT>
    void computeFluxEvalT(EvalT & dummyval, const int & block, const int & grp,  
                          ViewType u_kv, ViewType du_kv, ViewType dp_kv, View_Sc3 lambda,
                          const ScalarT & time, const int & side, const ScalarT & coarse_h,
                          const bool & compute_sens, const ScalarT & fluxwt,
                          bool & useTransientSol) {


      //#ifndef MrHyDE_NO_AD
        typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_AD2;
      //#else
      //  typedef View_Sc2 View_AD2;
      //#endif
      int wkblock = 0;

      wkset[wkblock]->setTime(time);
      wkset[wkblock]->sidename = boundary_groups[block][grp]->sidename;
      wkset[wkblock]->currentside = boundary_groups[block][grp]->sidenum;
      wkset[wkblock]->numElem = boundary_groups[block][grp]->numElem;
      
      // Currently hard coded to one physics sets
      int set = 0;

      vector<View_AD2> sol_vals = wkset[wkblock]->sol_vals;
      //auto param_AD = wkset->pvals;
      auto ulocal = boundary_groups[block][grp]->sol[set];
      auto currLIDs = boundary_groups[block][grp]->LIDs[set];

      if (useTransientSol) { 
        int stage = wkset[wkblock]->current_stage;
        auto b_A = wkset[wkblock]->butcher_A;
        auto b_b = wkset[wkblock]->butcher_b;
        auto BDF = wkset[wkblock]->BDF_wts;

        ScalarT one = 1.0;
        
        for (size_type var=0; var<ulocal.extent(1); var++ ) {
          size_t uindex = wkset[wkblock]->sol_vals_index[set][var];
          auto u_AD = sol_vals[uindex];
          auto off = subview(wkset[wkblock]->set_offsets[set],var,ALL());
          auto cu = subview(ulocal,ALL(),var,ALL());
          auto cu_prev = subview(boundary_groups[block][grp]->sol_prev[set],ALL(),var,ALL(),ALL());
          auto cu_stage = subview(boundary_groups[block][grp]->sol_stage[set],ALL(),var,ALL(),ALL());

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
        //Teuchos::TimeMonitor localtimer(*fluxGatherTimer);
        
        if (compute_sens) {
          for (size_t var=0; var<ulocal.extent(1); var++) {
            auto u_AD = sol_vals[var];
            auto offsets = subview(wkset[wkblock]->offsets,var,ALL());
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
            auto offsets = subview(wkset[wkblock]->offsets,var,ALL());
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
        //Teuchos::TimeMonitor localtimer(*fluxWksetTimer);
        wkset[wkblock]->computeSolnSideIP(boundary_groups[block][grp]->sidenum);//, u_AD, param_AD);
      }
      
      if (wkset[wkblock]->numAux > 0) {
        
       // Teuchos::TimeMonitor localtimer(*fluxAuxTimer);
      
        auto numAuxDOF = groupData[wkblock]->num_aux_dof;
        
        for (size_type var=0; var<numAuxDOF.extent(0); var++) {
          auto abasis = boundary_groups[block][grp]->auxside_basis[boundary_groups[block][grp]->auxusebasis[var]];
          auto off = subview(boundary_groups[block][grp]->auxoffsets,var,ALL());
          string varname = wkset[wkblock]->aux_varlist[var];
          auto local_aux = wkset[wkblock]->getSolutionField("aux "+varname,false);
          Kokkos::deep_copy(local_aux,0.0);
          //auto local_aux = Kokkos::subview(wkset->local_aux_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
          auto localID = boundary_groups[block][grp]->localElemID;
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
        //Teuchos::TimeMonitor localtimer(*fluxEvalTimer);
        groupData[block]->physics->computeFlux(0,groupData[block]->my_block);
      }
      //wkset->isOnSide = false;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    // Functionality moved from boundary groups into here
    ////////////////////////////////////////////////////////////////////////////////
    
    void computeJacResBoundary(const int & block, const size_t & grp,
                                                     const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                                                     const bool & compute_jacobian, const bool & compute_sens,
                                                     const int & num_active_params, const bool & compute_disc_sens,
                                                     const bool & compute_aux_sens, const bool & store_adjPrev,
                                                     View_Sc3 local_res, View_Sc3 local_J);

    // Backwards compatible function call
    // Calls fully templated version with AD
    void updateWorksetBoundary(const int & block, const size_t & grp, const int & seedwhat, 
                               const int & seedindex=0, const bool & override_transient=false);

    // Partially templated version that pick the appropriate workset to use
    // and calls fully templated version
    template<class EvalT>
    void updateWorksetBoundary(const int & block, const size_t & grp, const int & seedwhat,
                               const int & seedindex=0, const bool & override_transient=false);

    // Fully templated version
    // Actually does the work
    template<class EvalT>
    void updateWorksetBoundary(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp,
                               const int & seedwhat, const int & seedindex,
                               const bool & override_transient);

    void computeBoundaryAux(const int & block, const size_t & grp, const int & seedwhat);

    // Backwards compatible function call
    // Calls fully templated version with AD
    void updateDataBoundary(const int & block, const size_t & grp);

    // Partially templated version that pick the appropriate workset to use
    // and calls fully templated version
    template<class EvalT>
    void updateDataBoundary(const int & block, const size_t & grp);

    // Fully templated version
    // Actually does the work
    template<class EvalT>
    void updateDataBoundary(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp);

    // Backwards compatible function call
    // Calls fully templated version with AD
    void updateWorksetBasisBoundary(const int & block, const size_t & grp);

    // Partially templated version that pick the appropriate workset to use
    // and calls fully templated version
    template<class EvalT>
    void updateWorksetBasisBoundary(const int & block, const size_t & grp);

    // Fully templated version
    // Actually does the work
    template<class EvalT>
    void updateWorksetBasisBoundary(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp);

    void updateResBoundary(const int & block, const size_t & grp,
                           const bool & compute_sens, View_Sc3 local_res);

    void updateJacBoundary(const int & block, const size_t & grp, 
                           const bool & useadjoint, View_Sc3 local_J);           

    void updateParamJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J);

    void updateAuxJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J);

    View_Sc2 getDirichletBoundary(const int & block, const size_t & grp, const size_t & set);

    View_Sc3 getMassBoundary(const int & block, const size_t & grp, const size_t & set);
        
    ////////////////////////////////////////////////////////////////////////////////
    // Functionality moved from groups into here
    ////////////////////////////////////////////////////////////////////////////////
    
    // Backwards compatible function call
    // Calls fully templated version with AD
    void updateWorkset(const int & block, const size_t & grp, const int & seedwhat,
                       const int & seedindex, const bool & override_transient=false);

    // Partially templated version that pick the appropriate workset to use
    // and calls fully templated version
    template<class EvalT>
    void updateWorkset(const int & block, const size_t & grp, const int & seedwhat,
                       const int & seedindex, const bool & override_transient=false);

    // Fully templated version
    // Actually does the work
    template<class EvalT>
    void updateWorkset(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp,
                       const int & seedwhat, const int & seedindex,
                       const bool & override_transient);

    void computeSolAvg(const int & block, const size_t & grp);

    void computeSolutionAverage(const int & block, const size_t & grp,
                                const string & var, View_Sc2 csol);

    void computeParameterAverage(const int & block, const size_t & grp,
                                 const string & var, View_Sc2 sol);

    // Backwards compatible function call
    // Calls fully templated version with AD
    void updateWorksetFace(const int & block, const size_t & grp, const size_t & facenum);

    // Partially templated version that pick the appropriate workset to use
    // and calls fully templated version
    template<class EvalT>
    void updateWorksetFace(const int & block, const size_t & grp, const size_t & facenum);

    // Fully templated version
    // Actually does the work
    template<class EvalT>
    void updateWorksetFace(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp, const size_t & facenum);

    void computeJacRes(const int & block, const size_t & grp, 
                         const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                         const bool & compute_jacobian, const bool & compute_sens,
                         const int & num_active_params, const bool & compute_disc_sens,
                         const bool & compute_aux_sens, const bool & store_adjPrev,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_res,
                         Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                         const bool & assemble_volume_terms,
                         const bool & assemble_face_terms);

    void updateRes(const int & block, const size_t & grp,
                   const bool & compute_sens, View_Sc3 local_res);

    void updateAdjointRes(const int & block, const size_t & grp,
                            const bool & compute_jacobian, const bool & isTransient,
                            const bool & compute_aux_sens, const bool & store_adjPrev,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                            Kokkos::View<ScalarT***,AssemblyDevice> local_res);

    void updateJac(const int & block, const size_t & grp,
                   const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J);

    void fixDiagJac(const int & block, const size_t & grp, 
                      Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                      Kokkos::View<ScalarT***,AssemblyDevice> local_res);

    void updateParamJac(const int & block, const size_t & grp,
                        Kokkos::View<ScalarT***,AssemblyDevice> local_J);

    void updateAuxJac(const int & block, const size_t & grp,
                      Kokkos::View<ScalarT***,AssemblyDevice> local_J);

    View_Sc2 getInitial(const int & block, const size_t & grp,
                        const bool & project, const bool & isAdjoint);

    View_Sc2 getInitialFace(const int & block, const size_t & grp, const bool & project);


    CompressedView<View_Sc3> getMass(const int & block, const size_t & grp);

    CompressedView<View_Sc3> getWeightedMass(const int & block, const size_t & grp, vector<ScalarT> & masswts);

    CompressedView<View_Sc3> getMassFace(const int & block, const size_t & grp);

    Kokkos::View<ScalarT***,AssemblyDevice> getSolutionAtNodes(const int & block, const size_t & grp, const int & var);

    template<class EvalT>
    void updateGroupData(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp);

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    vector<vector<int> > identifySubgridModels();

    void createFunctions();

    void purgeMemory();
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<MpiComm> comm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    
    // Need
    std::vector<std::string> blocknames;
    std::vector<std::vector<std::vector<std::string> > > varlist; // [set][block][var]
    
    //Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
    Teuchos::RCP<MeshInterface>  mesh;
    Teuchos::RCP<DiscretizationInterface> disc;
    Teuchos::RCP<PhysicsInterface> physics;
    Teuchos::RCP<MultiscaleManager> multiscale_manager;
    std::vector<Teuchos::RCP<FunctionManager<AD> > > function_managers;
    std::vector<Teuchos::RCP<FunctionManager<ScalarT> > > function_managers_Sc;

    size_t globalParamUnknowns;
    int verbosity, debug_level;
    
    // Groups and worksets are unique to each block, but span the physics sets
    std::vector<Teuchos::RCP<GroupMetaData> > groupData;
    std::vector<std::vector<Teuchos::RCP<Group> > > groups;
    std::vector<std::vector<Teuchos::RCP<BoundaryGroup> > > boundary_groups;
    std::vector<Teuchos::RCP<Workset<AD> > > wkset;
    std::vector<Teuchos::RCP<Workset<AD4> > > wkset_AD4;
    std::vector<Teuchos::RCP<Workset<ScalarT> > > wkset_Sc;
    
    bool usestrongDBCs, use_meas_as_dbcs, multiscale, isTransient, fix_zero_rows, lump_mass, matrix_free;
    
    std::string assembly_partitioning;
    std::vector<std::vector<bool> > assemble_volume_terms, assemble_boundary_terms, assemble_face_terms; // use basis functions in assembly [block][set]
    std::vector<bool> build_volume_terms, build_boundary_terms, build_face_terms; // set up basis function [block]
    std::vector<Kokkos::View<bool*,LA_device> > isFixedDOF; // [set]
    std::vector<vector<vector<Kokkos::View<LO*,LA_device> > > > fixedDOF; // [set][block][var]
    Teuchos::RCP<ParameterManager<Node> > params;
      
  private:

    Teuchos::RCP<Teuchos::Time> assembly_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - total assembly");
    Teuchos::RCP<Teuchos::Time> assembly_res_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeRes() - residual assembly");
    Teuchos::RCP<Teuchos::Time> assembly_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJac() - Jacobian assembly");
    Teuchos::RCP<Teuchos::Time> gather_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::gather()");
    Teuchos::RCP<Teuchos::Time> physics_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - physics evaluation");
    Teuchos::RCP<Teuchos::Time> boundary_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - boundary evaluation");
    Teuchos::RCP<Teuchos::Time> scatter_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::scatter()");
    Teuchos::RCP<Teuchos::Time> dbc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::dofConstraints()");
    Teuchos::RCP<Teuchos::Time> complete_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - fill complete");
    Teuchos::RCP<Teuchos::Time> ms_proj_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - multiscale projection");
    Teuchos::RCP<Teuchos::Time> set_init_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::setInitial()");
    Teuchos::RCP<Teuchos::Time> set_dbc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::setDirichlet()");
    Teuchos::RCP<Teuchos::Time> group_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::createGroups()");
    Teuchos::RCP<Teuchos::Time> wkset_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::createWorkset()");
    Teuchos::RCP<Teuchos::Time> group_database_create_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::dataBase - assignment");
    Teuchos::RCP<Teuchos::Time> group_database_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::dataBase - basis");
  };
  
}

#endif
