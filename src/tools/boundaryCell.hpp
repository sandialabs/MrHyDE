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
#include "discretizationTools.hpp"
#include "boundaryWorkset.hpp"
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
               const int & sidenum_, const string & sidename_) :
  cellData(cellData_), localElemID(localID_), nodes(nodes_),
  sidenum(sidenum_), sidename(sidename_) {
  
    numElem = nodes.dimension(0);
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setIP(const DRV & ref_side_ip, const DRV & ref_side_wts) {
    
    ip = DRV("sip", numElem, ref_side_ip.dimension(0), cellData->dimension);
    ijac = DRV("sijac", numElem, ref_side_ip.dimension(0), cellData->dimension, cellData->dimension);
    wts = DRV("wts_side", numElem, ref_side_ip.dimension(0));
    normals = DRV("normals", numElem, ref_side_ip.dimension(0), cellData->dimension);
    
    DRV refSidePoints("refSidePoints", ref_side_ip.dimension(0), cellData->dimension);
    CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, ref_side_ip,
                                                     cellData->dimension-1, sidenum, *(cellData->cellTopo));
    
    CellTools<AssemblyDevice>::mapToPhysicalFrame(ip, refSidePoints, nodes, *(cellData->cellTopo));
    
    CellTools<AssemblyDevice>::setJacobian(ijac, refSidePoints, nodes, *(cellData->cellTopo));
    
    DRV ijacInv("sidejacobInv",numElem, ref_side_ip.dimension(0), cellData->dimension, cellData->dimension);
    
    CellTools<AssemblyDevice>::setJacobianInv(ijacInv, ijac);
    
    DRV temporary_buffer("temporary_buffer",numElem,ref_side_ip.dimension(0)*cellData->dimension*cellData->dimension);
    
    if (cellData->dimension == 2) {
      FunctionSpaceTools<AssemblyDevice>::computeEdgeMeasure(wts, ijac, ref_side_wts, sidenum,
                                                             *(cellData->cellTopo), temporary_buffer);
    }
    if (cellData->dimension == 3) {
      FunctionSpaceTools<AssemblyDevice>::computeFaceMeasure(wts, ijac, ref_side_wts, sidenum,
                                                             *(cellData->cellTopo), temporary_buffer);
    }
    CellTools<AssemblyDevice>::getPhysicalSideNormals(normals, ijac, sidenum, *(cellData->cellTopo));
    
    // scale the normal vector (we need unit normal...)
    
    for (int e=0; e<numElem; e++) {
      for( int j=0; j<ref_side_ip.dimension(0); j++ ) {
        ScalarT normalLength = 0.0;
        for (int sd=0; sd<cellData->dimension; sd++) {
          normalLength += normals(e,j,sd)*normals(e,j,sd);
        }
        normalLength = sqrt(normalLength);
        for (int sd=0; sd<cellData->dimension; sd++) {
          normals(e,j,sd) = normals(e,j,sd) / normalLength;
        }
      }
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  //void setIndex(const vector<vector<vector<int> > > & index_) {
  void setIndex(Kokkos::View<LO***,AssemblyDevice> & index_, Kokkos::View<LO*,AssemblyDevice> & numDOF_) {
    
    index = Kokkos::View<LO***,AssemblyDevice>("local index",index_.dimension(0),
                                               index_.dimension(1), index_.dimension(2));
    
    // Need to copy the data since index_ is rewritten for each cell
    parallel_for(RangePolicy<AssemblyDevice>(0,index_.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int j=0; j<index_.dimension(1); j++) {
        for (int k=0; k<index_.dimension(2); k++) {
          index(e,j,k) = index_(e,j,k);
        }
      }
    });
    
    // This is common to all cells (within the same block), so a view copy will do
    numDOF = numDOF_;
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  //void setParamIndex(const vector<vector<vector<int> > > & pindex_) {
  void setParamIndex(Kokkos::View<LO***,AssemblyDevice> & pindex_,
                     Kokkos::View<LO*,AssemblyDevice> & pnumDOF_) {
    
    paramindex = Kokkos::View<LO***,AssemblyDevice>("local param index",pindex_.dimension(0),
                                                    pindex_.dimension(1), pindex_.dimension(2));
    
    // Need to copy the data since index_ is rewritten for each cell
    parallel_for(RangePolicy<AssemblyDevice>(0,pindex_.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int j=0; j<pindex_.dimension(1); j++) {
        for (int k=0; k<pindex_.dimension(2); k++) {
          paramindex(e,j,k) = pindex_(e,j,k);
        }
      }
    });
    
    // This is common to all cells, so a view copy will do
    numParamDOF = pnumDOF_;
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setAuxIndex(Kokkos::View<LO***,AssemblyDevice> & aindex_) {
    
    auxindex = Kokkos::View<LO***,AssemblyDevice>("local aux index",1,aindex_.dimension(1),
                                                  aindex_.dimension(2));
    
    // Need to copy the data since index_ is rewritten for each cell
    parallel_for(RangePolicy<AssemblyDevice>(0,aindex_.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      for (int j=0; j<aindex_.dimension(1); j++) {
        for (int k=0; k<aindex_.dimension(2); k++) {
          auxindex(e,j,k) = aindex_(e,j,k);
        }
      }
    });
    
    // This is common to all cells, so a view copy will do
    // This is excessive storage, please remove
    //numAuxDOF = anumDOF_;
    // Temp. fix
    numAuxDOF = Kokkos::View<int*,HostDevice>("numAuxDOF",auxindex.dimension(1));
    for (int i=0; i<auxindex.dimension(1); i++) {
      numAuxDOF(i) = auxindex.dimension(2);
    }
    
  }
  
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
  // Set the local solutions
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setLocalSoln(const Teuchos::RCP<Epetra_MultiVector> & gl_u, const int & type,
                    const size_t & entry);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Map the coarse grid solution to the fine grid integration points
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeSoln(const bool & seedu, const bool & seedudot, const bool & seedparams,
                        const bool & seedaux);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the contribution from this cell to the global res, J, Jdot
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                     const bool & compute_jacobian, const bool & compute_sens,
                     const int & num_active_params, const bool & compute_disc_sens,
                     const bool & compute_aux_sens, const bool & store_adjPrev,
                     Kokkos::View<ScalarT***,AssemblyDevice> res,
                     Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                     Kokkos::View<ScalarT***,AssemblyDevice> local_Jdot);
  
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

  void computeFlux(const Teuchos::RCP<Epetra_MultiVector> & u,
                   const Teuchos::RCP<Epetra_MultiVector> & du,
                   const Teuchos::RCP<Epetra_MultiVector> & sub_param,
                   Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                   const ScalarT & time, const int & s, const ScalarT & coarse_h,
                   const bool & compute_sens);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the discretization/physics info (used for workset construction)
  ///////////////////////////////////////////////////////////////////////////////////////

  vector<int> getInfo() {
    vector<int> info;
    int nparams = 0;
    if (paramindex.dimension(0)>0) {
      nparams = paramindex.dimension(1);
    }
    info.push_back(cellData->dimension);
    info.push_back(numDOF.dimension(0));
    info.push_back(nparams);
    info.push_back(auxindex.dimension(1));
    info.push_back(GIDs.dimension(1));
    info.push_back(numElem);
    return info;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  // Public data 
  Teuchos::RCP<CellMetaData> cellData;
  Teuchos::RCP<BoundaryWorkset> wkset;
  
  Kokkos::View<LO*> localElemID;
  
  // Geometry Information
  int numElem, sidenum;
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
