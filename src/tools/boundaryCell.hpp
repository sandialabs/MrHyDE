/***********************************************************************
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

static void boundaryCellHelp(const string & details) {
  cout << "********** Help and Documentation for the cells **********" << endl;
}

class BoundaryCell {
public:
  
  BoundaryCell() {} ;
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  BoundaryCell(const Teuchos::RCP<CellMetaData> & cellData_,
               const DRV & nodes_,
               const Kokkos::View<int*> & localID_,
               const Kokkos::View<int*> & sideID_,
               const int & sidenum_, const string & sidename_,
               const int & cellID_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setIndex(Kokkos::View<LO***,AssemblyDevice> & index_, Kokkos::View<LO*,AssemblyDevice> & numDOF_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setParamIndex(Kokkos::View<LO***,AssemblyDevice> & pindex_,
                     Kokkos::View<LO*,AssemblyDevice> & pnumDOF_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setAuxIndex(Kokkos::View<LO***,AssemblyDevice> & aindex_);
  
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
  
  void setUseBasis(vector<int> & usebasis_, const int & nstages_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Define which basis each discretized parameter will use
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_);
    
  ///////////////////////////////////////////////////////////////////////////////////////
  // Define which basis each aux variable will use
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setAuxUseBasis(vector<int> & ausebasis_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Map the coarse grid solution to the fine grid integration points
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeSoln(Kokkos::View<int*,UnifiedDevice> seedwhat);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the contribution from this cell to the global res, J, Jdot
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                     const bool & compute_jacobian, const bool & compute_sens,
                     const int & num_active_params, const bool & compute_disc_sens,
                     const bool & compute_aux_sens, const bool & store_adjPrev,
                     Kokkos::View<ScalarT***,UnifiedDevice> res,
                     Kokkos::View<ScalarT***,UnifiedDevice> local_J,
                     Kokkos::View<ScalarT***,UnifiedDevice> local_Jdot);
  
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

  void computeFlux(const vector_RCP & gl_u, const vector_RCP & gl_du, const vector_RCP & params,
                  Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                  const ScalarT & time, const int & side, const ScalarT & coarse_h,
                   const bool & compute_sens);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the discretization/physics info (used for workset construction)
  ///////////////////////////////////////////////////////////////////////////////////////

  vector<int> getInfo() {
    vector<int> info;
    int nparams = 0;
    if (paramindex.extent(0)>0) {
      nparams = paramindex.extent(1);
    }
    info.push_back(cellData->dimension);
    info.push_back(numDOF.extent(0));
    info.push_back(nparams);
    info.push_back(auxindex.extent(1));
    info.push_back(GIDs.extent(1));
    info.push_back(numElem);
    return info;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  // Public data 
  Teuchos::RCP<CellMetaData> cellData;
  Teuchos::RCP<workset> wkset;
  
  Kokkos::View<LO*> localElemID, localSideID;
  Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> orientation;
  
  // Geometry Information
  int numElem = 0; // default value ... used to check if proc. has elements on boundary
  int sidenum, cellID, wksetBID;
  DRV nodes, ip, wts, ijac, normals;
  Kokkos::View<int****,HostDevice> sideinfo; // may need to move this to Assembly
  string sidename;
  
  // DOF information
  Kokkos::View<GO**,HostDevice> GIDs, paramGIDs, auxGIDs;
  Kokkos::View<LO***,AssemblyDevice> index, paramindex, auxindex;
  Kokkos::View<int*,AssemblyDevice> numDOF, numParamDOF, numAuxDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> u, u_dot, phi, phi_dot, aux;
  
  // Discretized Parameter Information
  Kokkos::View<ScalarT***,AssemblyDevice> param;
  
  // Aux variable Information
  vector<string> auxlist;
  vector<vector<int> > auxoffsets;
  vector<int> auxusebasis;
  vector<basis_RCP> auxbasisPointers;
  vector<DRV> auxbasis, auxbasisGrad;
  vector<DRV> auxside_basis, auxside_basisGrad;
  vector<size_t> auxMIDs;
  
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
  
  
};

#endif
