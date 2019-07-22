/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef CELL_H
#define CELL_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "physics_base.hpp"
#include "discretizationTools.hpp"
#include "physicsInterface.hpp"
#include "workset.hpp"
#include "subgridModel.hpp"
#include "cellMetaData.hpp"

#include <iostream>     
#include <iterator>     

static void cellHelp(const string & details) {
  cout << "********** Help and Documentation for the cells **********" << endl;
}

class cell {
public:
  
  cell() {} ;
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  cell(const Teuchos::RCP<CellMetaData> & cellData_,
       const DRV & nodes_,
       const Kokkos::View<int*> & localID_) :
  cellData(cellData_), localElemID(localID_), nodes(nodes_){
  
    numElem = nodes.dimension(0);
    active = true;
    useSensors = false;
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setIP(const DRV & ref_ip) {
    // ip and ref_ip will live on the assembly device
    ip = DRV("ip", numElem, ref_ip.dimension(0), cellData->dimension);
    CellTools<AssemblyDevice>::mapToPhysicalFrame(ip, ref_ip, nodes, *(cellData->cellTopo));
    
    ijac = DRV("ijac", numElem, ref_ip.dimension(0), cellData->dimension, cellData->dimension);
    CellTools<AssemblyDevice>::setJacobian(ijac, ref_ip, nodes, *(cellData->cellTopo));
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
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

  void addAuxDiscretization(const vector<basis_RCP> & abasis_pointers, const vector<DRV> & abasis,
                            const vector<DRV> & abasisGrad, const vector<vector<DRV> > & asideBasis, 
                            const vector<vector<DRV> > & asideBasisGrad);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Update the regular parameters (everything but discretized)
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames);
                        
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
  
  void computeSolnVolIP(const bool & seedu, const bool & seedudot, const bool & seedparams,
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
  // Update the solution variables in the workset
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateSolnWorkset(const vector_RCP & gl_u, const int tindex);
  
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
  // Get the initial condition 
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT**,AssemblyDevice> getInitial(const bool & project, const bool & isAdjoint);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the mass matrix 
  ///////////////////////////////////////////////////////////////////////////////////////
    
  Kokkos::View<ScalarT***,AssemblyDevice> getMass();
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the error at the integration points given the solution and solve times
  ///////////////////////////////////////////////////////////////////////////////////////

  Kokkos::View<ScalarT**,AssemblyDevice> computeError(const ScalarT & solvetime, const size_t & tindex,
                                                     const bool compute_subgrid, const string & error_type);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the response at a given set of points and time
  ///////////////////////////////////////////////////////////////////////////////////////

  Kokkos::View<AD***,AssemblyDevice> computeResponseAtNodes(const DRV & nodes,
                                                            const int tindex,
                                                            const ScalarT & time);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the response at the integration points given the solution and solve times
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD***,AssemblyDevice> computeResponse(const ScalarT & solvetime,
                                                     const size_t & tindex,
                                                     const int & seedwhat);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute volumetric contribution to the regularization
  ///////////////////////////////////////////////////////////////////////////////////////

  AD computeDomainRegularization(const vector<ScalarT> reg_constants, const vector<int> reg_types,
                                 const vector<int> reg_indices);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the objective function given the solution and solve times
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD**,AssemblyDevice> computeObjective(const ScalarT & solvetime,
                                                     const size_t & tindex,
                                                     const int & seedwhat);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the target function given the solve times
  ///////////////////////////////////////////////////////////////////////////////////////

  Kokkos::View<AD***,AssemblyDevice> computeTarget(const ScalarT & solvetime);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the weight functino given the solve times
  ///////////////////////////////////////////////////////////////////////////////////////

  Kokkos::View<AD***,AssemblyDevice> computeWeight(const ScalarT & solvetime);

  ///////////////////////////////////////////////////////////////////////////////////////
  // Add sensor information
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                  const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                  const vector<basis_RCP> & basis_pointers,
                  const vector<basis_RCP> & param_basis_pointers);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Subgrid Plotting
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void writeSubgridSolution(const std::string & filename);
    
  ///////////////////////////////////////////////////////////////////////////////////////
  // Subgrid Plotting
  ///////////////////////////////////////////////////////////////////////////////////////

  void writeSubgridSolution(Teuchos::RCP<panzer_stk::STK_Interface> & globalmesh,
                            string & subblockname, bool & isTD, int & offset);

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

  void setUpAdjointPrev(const int & numDOF) {
    adjPrev = Kokkos::View<ScalarT**,AssemblyDevice>("previous adjoint",numElem,numDOF);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  void setUpSubGradient(const int & numParams) {
    subgradient = Kokkos::View<ScalarT**,AssemblyDevice>("subgrid gradient",numElem,numParams);
  }
   
  ///////////////////////////////////////////////////////////////////////////////////////
  // Update the subgrid model
  ///////////////////////////////////////////////////////////////////////////////////////

  void updateSubgridModel(vector<Teuchos::RCP<SubGridModel> > & models);
  
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Pass cell data to wkset
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void updateData();
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  void resetAdjPrev(const ScalarT & val);

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  // Public data 
  Teuchos::RCP<CellMetaData> cellData;
  Teuchos::RCP<workset> wkset;
  
  bool active;
  Kokkos::View<LO*> localElemID;
  
  // Geometry Information
  int numElem;
  DRV nodes, ip, ijac;
  //vector<DRV> sideip, sideijac, normals, sidewts;
  Kokkos::View<int****,HostDevice> sideinfo; // may need to move this to Assembly
  vector<string> sidenames;
  
  // DOF information
  Kokkos::View<GO**,HostDevice> GIDs, paramGIDs, auxGIDs;
  Kokkos::View<LO***,AssemblyDevice> index, paramindex, auxindex;
  vector<vector<ScalarT> > orientation;
  Kokkos::View<int*,AssemblyDevice> numDOF, numParamDOF, numAuxDOF;
  Kokkos::View<ScalarT***,AssemblyDevice> u, u_dot, phi, phi_dot, aux;
  
  // Subgrid information
  vector<Teuchos::RCP<SubGridModel> > subgridModels;
  vector<size_t> subgrid_usernum, cell_data_seed, cell_data_seedindex;
  vector<vector<size_t> > subgrid_model_index;
  
  // Discretized Parameter Information
  Kokkos::View<ScalarT***,AssemblyDevice> param;
  
  // Aux variable Information
  vector<string> auxlist;
  vector<vector<int> > auxoffsets;
  vector<int> auxusebasis;
  vector<basis_RCP> auxbasisPointers;
  vector<DRV> auxbasis, auxbasisGrad;
  vector<vector<DRV> > auxside_basis, auxside_basisGrad;
  
  // Sensor information
  bool useSensors;
  size_t numSensors;
  vector<Kokkos::View<ScalarT**,HostDevice> > sensorLocations, sensorData;
  DRV sensorPoints;
  vector<int> sensorElem, mySensorIDs;
  vector<vector<DRV> > sensorBasis, param_sensorBasis, sensorBasisGrad, param_sensorBasisGrad;
  Kokkos::View<ScalarT**,AssemblyDevice> adjPrev, subgradient, cell_data;
  vector<ScalarT> cell_data_distance;
  
  // Profile timers
  Teuchos::RCP<Teuchos::Time> computeSolnVolTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeSolnVolIP()");
  Teuchos::RCP<Teuchos::Time> computeSolnSideTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeSolnSideIP()");
  Teuchos::RCP<Teuchos::Time> volumeResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - volume residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - boundary residual");
  Teuchos::RCP<Teuchos::Time> jacobianFillTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - fill local Jacobian");
  Teuchos::RCP<Teuchos::Time> residualFillTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - fill local residual");
  Teuchos::RCP<Teuchos::Time> transientResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - transient residual");
  Teuchos::RCP<Teuchos::Time> adjointResidualTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeJacRes() - adjoint residual");
  Teuchos::RCP<Teuchos::Time> cellFluxGatherTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeFlux - gather solution");
  Teuchos::RCP<Teuchos::Time> cellFluxWksetTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeFlux - update wkset");
  Teuchos::RCP<Teuchos::Time> cellFluxAuxTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeFlux - compute aux solution");
  Teuchos::RCP<Teuchos::Time> cellFluxEvalTimer = Teuchos::TimeMonitor::getNewCounter("MILO::cell::computeFlux - physics evaluation");
  
  
};

#endif
