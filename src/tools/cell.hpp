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

#ifndef CELL_H
#define CELL_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"
#include "subgridModel.hpp"
#include "cellMetaData.hpp"
#include "discretizationInterface.hpp"

#include <iostream>     
#include <iterator>     

namespace MrHyDE {
  /*
  static void cellHelp(const string & details) {
    cout << "********** Help and Documentation for the cells **********" << endl;
  }
  */
  
  class cell {
  public:
    
    cell() {} ;
    
    ~cell() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    cell(const Teuchos::RCP<CellMetaData> & cellData_,
         const DRV nodes_,
         const Kokkos::View<LO*,AssemblyDevice> localID_,
         Teuchos::RCP<DiscretizationInterface> & disc_,
         const bool & storeAll_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeBasis();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void createHostLIDs();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setIP();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setWorkset(Teuchos::RCP<workset> & wkset_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setParams(LIDView paramLIDS_);
    
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
    
    //void updateWorksetIP();
    
    void updateWorkset(const int & seedwhat, const bool & override_transient=false);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Map the solution to the volumetric integration points
    ///////////////////////////////////////////////////////////////////////////////////////
    
    //void computeSolnVolIP();
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Update the workset
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateWorksetFace(const size_t & facenum);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Map the solution to the face integration points
    ///////////////////////////////////////////////////////////////////////////////////////
    
    //void computeSolnFaceIP(const size_t & facenum);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Map the solution to the face integration points
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeAuxSolnFaceIP(const size_t & facenum);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Reset the data stored in the previous step/stage solutions
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void resetPrevSoln(const size_t & set);
    
    void resetStageSoln(const size_t & set);
    
    void updateStageSoln(const size_t & set);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute the contribution from this cell to the global res, J, Jdot
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeJacRes(const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                       const bool & compute_jacobian, const bool & compute_sens,
                       const int & num_active_params, const bool & compute_disc_sens,
                       const bool & compute_aux_sens, const bool & store_adjPrev,
                       View_Sc3 res,
                       View_Sc3 local_J,
                       const bool & assemble_volume_terms,
                       const bool & assemble_face_terms);
        
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT res
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateRes(const bool & compute_sens, View_Sc3 local_res);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Update the adjoint res
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateAdjointRes(const bool & compute_jacobian, const bool & isTransient,
                          const bool & compute_aux_sens, const bool & store_adjPrev,
                          View_Sc3 local_J,
                          View_Sc3 local_res);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT J
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateJac(const bool & useadjoint, View_Sc3 local_J);
    
    void fixDiagJac(View_Sc3 local_J,
                    View_Sc3 local_res);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jparam
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateParamJac(View_Sc3 local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Use the AD res to update the scalarT Jaux
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateAuxJac(View_Sc3 local_J);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the initial condition
    ///////////////////////////////////////////////////////////////////////////////////////
    
    View_Sc2 getInitial(const bool & project, const bool & isAdjoint);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the initial condition on the faces
    ///////////////////////////////////////////////////////////////////////////////////////

    /* @brief Project the initial condition on the faces
     *
     * @param[in] project  Flag for L2 projection
     *
     * @returns View_Sc2 of projected data
     *
     * @warning BWR -- under development, can only project, etc. 
     */
    
    View_Sc2 getInitialFace(const bool & project);

    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the mass matrix
    ///////////////////////////////////////////////////////////////////////////////////////
    
    View_Sc3 getMass();
    
    View_Sc3 getWeightedMass(vector<ScalarT> & masswts);

    /* @brief Assemble the local mass matrix on faces
     *
     * @warning BWR -- under development. Will (and should) only work for HFACE vars
     */

    View_Sc3 getMassFace();
    
    void applyLocalMass(const size_t & set, vector<ScalarT> & masswts,
                        View_Sc2 x_local, View_Sc2 y_local);
    
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
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setUpAdjointPrev(const int & numsteps, const int & numstages) {
      if (cellData->requiresTransient && cellData->requiresAdjoint) {
        for (size_t set=0; set<LIDs.size(); ++set) {
          View_Sc3 newaprev("previous step adjoint",numElem,LIDs[set].extent(1),numsteps);
          adj_prev.push_back(newaprev);
          View_Sc3 newastage("previous stage adjoint",numElem,LIDs[set].extent(1),numstages);
          adj_stage_prev.push_back(newastage);
        }
      }
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setUpSubGradient(const int & numParams) {
      if (cellData->requiresAdjoint) {
        subgradient = View_Sc2("subgrid gradient",numElem,numParams);
      }
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
    
    void resetAdjPrev(const size_t & set, const ScalarT & val);
    
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolAvg();
    
    void computeSolutionAverage(const string & var, View_Sc2 sol);
    
    Kokkos::View<ScalarT***,AssemblyDevice> getSolutionAtNodes(const int & var);
    
    size_t getVolumetricStorage();
    
    size_t getFaceStorage();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    // Public data
    
    // Data created elsewhere
    vector<LIDView> LIDs;
    LIDView paramLIDs, auxLIDs;
    
    // Creating LIDs on host device for host assembly
    vector<LIDView_host> LIDs_host;
    LIDView_host paramLIDs_host;
    
    Teuchos::RCP<CellMetaData> cellData;
    Teuchos::RCP<workset> wkset;
    vector<Teuchos::RCP<SubGridModel> > subgridModels;
    Kokkos::View<LO*,AssemblyDevice> localElemID;
    vector<Kokkos::View<int****,HostDevice> > sideinfo; // may need to move this to Assembly
    DRV nodes;
    vector<size_t> cell_data_seed, cell_data_seedindex;
    vector<size_t> subgrid_model_index; // which subgrid model is used for each time step
    size_t subgrid_usernum; // what is the index for this cell in the subgrid model (should be deprecated)
    
    Teuchos::RCP<DiscretizationInterface> disc;
    
    // Data created here (Views should all be AssemblyDevice)
    size_t numElem;
    vector<View_Sc2> ip;
    View_Sc2 wts; 
    vector<vector<View_Sc2>> ip_face, normals_face;
    vector<View_Sc2> wts_face;
    vector<View_Sc1> hsize_face;
    
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation;
    vector<View_Sc3> u, phi;
    View_Sc3 param, aux; // (elem,var,numdof)
    vector<View_Sc3> u_avg, u_alt;
    View_Sc3 param_avg, aux_avg; // (elem,var,dim)
    vector<View_Sc4> u_prev, phi_prev, aux_prev, u_stage, phi_stage, aux_stage; // (elem,var,numdof,step or stage)
    
    // basis information
    vector<View_Sc4> basis, basis_grad, basis_curl, basis_nodes;
    vector<View_Sc3> basis_div, local_mass;
    
    //vector<vector<DRV> > basis_face, basis_grad_face;
    vector<vector<View_Sc4>> basis_face, basis_grad_face;
    View_Sc1 hsize;
    
    // Aux variable Information
    vector<string> auxlist;
    Kokkos::View<LO**,AssemblyDevice> auxoffsets;
    vector<int> auxusebasis;
    vector<basis_RCP> auxbasisPointers;
    vector<DRV> auxbasis, auxbasisGrad; // this does cause a problem
    vector<vector<DRV> > auxside_basis, auxside_basisGrad;
    
    // Sensor information
    bool storeAll, storeMass, useSensors, usealtsol = false, haveBasis;
    size_t numSensors;
    vector<Kokkos::View<ScalarT**,HostDevice> > sensorLocations, sensorData;
    View_Sc3 sensorPoints;
    vector<int> sensorElem, mySensorIDs;
    vector<vector<DRV> > sensorBasis, param_sensorBasis, sensorBasisGrad, param_sensorBasisGrad;
    Kokkos::View<ScalarT**,AssemblyDevice> subgradient, cell_data;
    vector<Kokkos::View<ScalarT***,AssemblyDevice> > adj_prev, adj_stage_prev;
    vector<ScalarT> cell_data_distance;
    
    // Profile timers
    Teuchos::RCP<Teuchos::Time> computeSolnVolTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeSolnVolIP()");
    Teuchos::RCP<Teuchos::Time> computeSolnFaceTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeSolnFaceIP()");
    Teuchos::RCP<Teuchos::Time> volumeResidualTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeJacRes() - volume residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeJacRes() - boundary residual");
    Teuchos::RCP<Teuchos::Time> faceResidualTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeJacRes() - edge/face residual");
    Teuchos::RCP<Teuchos::Time> jacobianFillTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeJacRes() - fill local Jacobian");
    Teuchos::RCP<Teuchos::Time> residualFillTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeJacRes() - fill local residual");
    Teuchos::RCP<Teuchos::Time> transientResidualTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeJacRes() - transient residual");
    Teuchos::RCP<Teuchos::Time> adjointResidualTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeJacRes() - adjoint residual");
    Teuchos::RCP<Teuchos::Time> cellFluxGatherTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeFlux - gather solution");
    Teuchos::RCP<Teuchos::Time> cellFluxWksetTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeFlux - update wkset");
    Teuchos::RCP<Teuchos::Time> cellFluxAuxTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeFlux - compute aux solution");
    Teuchos::RCP<Teuchos::Time> cellFluxEvalTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeFlux - physics evaluation");
    Teuchos::RCP<Teuchos::Time> computeSolAvgTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::computeSolAvg()");
    Teuchos::RCP<Teuchos::Time> computeNodeSolTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::getSolutionAtNodes()");
    Teuchos::RCP<Teuchos::Time> buildBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::constructor - build basis");
    Teuchos::RCP<Teuchos::Time> buildFaceBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::constructor - build face basis");
    Teuchos::RCP<Teuchos::Time> objectiveTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::objective");
    Teuchos::RCP<Teuchos::Time> responseTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cell::response");
    
  };
  
}

#endif
