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

#ifndef BOUNDCELL_H
#define BOUNDCELL_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"
#include "cellMetaData.hpp"

#include <iostream>     
#include <iterator>     

namespace MrHyDE {
  /*
  static void boundaryCellHelp(const string & details) {
    cout << "********** Help and Documentation for the cells **********" << endl;
  }
  */
  
  class BoundaryCell {
    
    typedef Tpetra::MultiVector<ScalarT,LO,GO,AssemblyNode> SG_MultiVector;
    typedef Teuchos::RCP<SG_MultiVector> SG_vector_RCP;
    
  public:
    
    BoundaryCell() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    BoundaryCell(const Teuchos::RCP<CellMetaData> & cellData_,
                 const DRV nodes_,
                 const Kokkos::View<LO*,AssemblyDevice> localID_,
                 const Kokkos::View<LO*,AssemblyDevice> sideID_,
                 const int & sidenum_, const string & sidename_,
                 const int & cellID_,
                 LIDView LIDs_,
                 Kokkos::View<int****,HostDevice> sideinfo_,
                 Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation_);
    
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
    
    void setUseBasis(vector<int> & usebasis_, const int & numsteps, const int & numstages);
    
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
    
    void updateWorksetBasis();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Map the coarse grid solution to the fine grid integration points
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeSoln(const int & seedwhat);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute the contribution from this cell to the global res, J, Jdot
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                       const bool & compute_jacobian, const bool & compute_sens,
                       const int & num_active_params, const bool & compute_disc_sens,
                       const bool & compute_aux_sens, const bool & store_adjPrev,
                       Kokkos::View<ScalarT***,AssemblyDevice> res,
                       Kokkos::View<ScalarT***,AssemblyDevice> local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT res
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Update the adjoint res
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateAdjointRes(const bool & compute_sens, Kokkos::View<ScalarT***,AssemblyDevice> local_res);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT J
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateJac(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jdot
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateJacDot(const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jparam
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateParamJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jdot
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateParamJacDot(Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jaux
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateAuxJac(Kokkos::View<ScalarT***,AssemblyDevice> local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jdot
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateAuxJacDot(Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute boundary contribution to the regularization and nodes located on the boundary
    ///////////////////////////////////////////////////////////////////////////////////////
    
    AD computeBoundaryRegularization(const vector<ScalarT> reg_constants, const vector<int> reg_types,
                                     const vector<int> reg_indices, const vector<string> reg_sides);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute flux and sensitivity wrt params
    ///////////////////////////////////////////////////////////////////////////////////////
    
    template<class ViewType>
    void computeFlux(ViewType u_kv, ViewType du_kv, ViewType dp_kv,
                     Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                     const ScalarT & time, const int & side, const ScalarT & coarse_h,
                     const bool & compute_sens) {
      //wkset->setTime(time);
        
      auto u_AD = wkset->uvals;
      //auto param_AD = wkset->pvals;
      auto offsets = wkset->offsets;
      auto ulocal = u;

      {
        Teuchos::TimeMonitor localtimer(*cellFluxGatherTimer);
        
        if (compute_sens) {
          parallel_for("bcell flux gather",RangePolicy<AssemblyExec>(0,ulocal.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            for (size_t var=0; var<ulocal.extent(1); var++) {
              for( size_t dof=0; dof<ulocal.extent(2); dof++ ) {
                u_AD(elem,var,dof) = AD(u_kv(LIDs(elem,offsets(var,dof)),0));
              }
            }
          });
        }
        else {
          parallel_for("bcell flux gather",RangePolicy<AssemblyExec>(0,ulocal.extent(0)), KOKKOS_LAMBDA (const int elem ) {
            for (size_t var=0; var<ulocal.extent(1); var++) {
              for( size_t dof=0; dof<ulocal.extent(2); dof++ ) {
                u_AD(elem,var,dof) = AD(maxDerivs, 0, u_kv(LIDs(elem,offsets(var,dof)),0));
                for( size_t p=0; p<du_kv.extent(1); p++ ) {
                  u_AD(elem,var,dof).fastAccessDx(p) = du_kv(LIDs(elem,offsets(var,dof)),p);
                }
              }
            }
          });
        }
      }
      
      {
        Teuchos::TimeMonitor localtimer(*cellFluxWksetTimer);
        wkset->computeSolnSideIP(sidenum);//, u_AD, param_AD);
      }
      
      
      if (wkset->numAux > 0) {
        
        Teuchos::TimeMonitor localtimer(*cellFluxAuxTimer);
      
        auto numAuxDOF = cellData->numAuxDOF;
        
        wkset->resetAuxSide();
        
        for (size_type var=0; var<numAuxDOF.extent(0); var++) {
          auto abasis = auxside_basis[auxusebasis[var]];
          auto off = Kokkos::subview(auxoffsets,var,Kokkos::ALL());
          auto local_aux = Kokkos::subview(wkset->local_aux_side,Kokkos::ALL(),var,Kokkos::ALL());
          auto localID = localElemID;
          auto varaux = Kokkos::subview(lambda,Kokkos::ALL(),var,Kokkos::ALL());
          parallel_for("bcell aux",RangePolicy<AssemblyExec>(0,localID.extent(0)), KOKKOS_LAMBDA (const size_type elem ) {
            for (size_type dof=0; dof<abasis.extent(1); ++dof) {
              AD auxval = AD(maxDerivs,off(dof), varaux(localID(elem),dof));
              for (size_type pt=0; pt<abasis.extent(2); ++pt) {
                local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
              }
            }
          });
        }
        
      }
      
      {
        Teuchos::TimeMonitor localtimer(*cellFluxEvalTimer);
        cellData->physics_RCP->computeFlux(cellData->myBlock);
      }
      
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the discretization/physics info (used for workset construction)
    ///////////////////////////////////////////////////////////////////////////////////////
    
    vector<int> getInfo() {
      vector<int> info;
      info.push_back(cellData->dimension);
      info.push_back(cellData->numDOF.extent(0));
      info.push_back(cellData->numParamDOF.extent(0));
      info.push_back(cellData->numAuxDOF.extent(0));
      info.push_back(LIDs.extent(1));
      info.push_back(numElem);
      return info;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<ScalarT**,AssemblyDevice> getDirichlet();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<ScalarT***,AssemblyDevice> getMass();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    

    // Public data 
    Teuchos::RCP<CellMetaData> cellData;
    Teuchos::RCP<workset> wkset;
    
    Kokkos::View<LO*,AssemblyDevice> localElemID, localSideID;
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation;
    
    // Geometry Information
    size_t numElem = 0; // default value ... used to check if proc. has elements on boundary
    int sidenum, cellID, wksetBID;
    DRV nodes;
    Kokkos::View<ScalarT***,AssemblyDevice> ip, normals, tangents;
    Kokkos::View<ScalarT**,AssemblyDevice> wts;
    Kokkos::View<ScalarT*,AssemblyDevice> hsize;
    
    Kokkos::View<int****,HostDevice> sideinfo; // may need to move this to Assembly
    string sidename;
    
    // Frequently used Views (None of these are allocated in the cells)
    //Kokkos::View<AD**,AssemblyDevice> res_AD;
    //Kokkos::View<int**,AssemblyDevice> offsets, paramoffsets;
    //Kokkos::View<LO*,AssemblyDevice> numDOF, numParamDOF, numAuxDOF;
    
    // DOF information
    LIDView LIDs, paramLIDs, auxLIDs;
    
    // Creating LIDs on host device for host assembly
    LIDView_host LIDs_host, paramLIDs_host, auxLIDs_host;
    
    Kokkos::View<ScalarT***,AssemblyDevice> u, phi, aux, param;
    Kokkos::View<ScalarT****,AssemblyDevice> u_prev, phi_prev, u_stage, phi_stage; // (elem,var,numdof,step or stage)
    //Kokkos::View<AD***,AssemblyDevice> u_AD, param_AD;
    
    // basis information
    //vector<DRV> basis, basis_grad, basis_div, basis_curl;
    vector<Kokkos::View<ScalarT****,AssemblyDevice> > basis, basis_grad, basis_curl;
    vector<Kokkos::View<ScalarT***,AssemblyDevice> > basis_div;
    
    // Aux variable Information
    vector<string> auxlist;
    Kokkos::View<LO**,AssemblyDevice> auxoffsets;
    vector<int> auxusebasis;
    vector<basis_RCP> auxbasisPointers;
    vector<DRV> auxbasis, auxbasisGrad;
    vector<DRV> auxside_basis, auxside_basisGrad;
    vector<size_t> auxMIDs;
    Kokkos::View<size_t*,AssemblyDevice> auxMIDs_dev;
    
    // Boundary cells do not have sensors
    
    // Profile timers
    Teuchos::RCP<Teuchos::Time> computeSolnSideTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeSolnSideIP()");
    Teuchos::RCP<Teuchos::Time> boundaryResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeJacRes() - boundary residual");
    Teuchos::RCP<Teuchos::Time> jacobianFillTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeJacRes() - fill local Jacobian");
    Teuchos::RCP<Teuchos::Time> residualFillTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeJacRes() - fill local residual");
    Teuchos::RCP<Teuchos::Time> transientResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeJacRes() - transient residual");
    Teuchos::RCP<Teuchos::Time> adjointResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeJacRes() - adjoint residual");
    Teuchos::RCP<Teuchos::Time> cellFluxGatherTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeFlux - gather solution");
    Teuchos::RCP<Teuchos::Time> cellFluxWksetTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeFlux - update wkset");
    Teuchos::RCP<Teuchos::Time> cellFluxAuxTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeFlux - compute aux solution");
    Teuchos::RCP<Teuchos::Time> cellFluxEvalTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell::computeFlux - physics evaluation");
    Teuchos::RCP<Teuchos::Time> buildBasisTimer = Teuchos::TimeMonitor::getNewCounter("MILO::boundaryCell - build basis");  
    
  };
  
}

#endif
