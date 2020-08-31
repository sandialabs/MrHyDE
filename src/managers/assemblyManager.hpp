/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include "trilinos.hpp"
#include "Panzer_DOFManager.hpp"

#include "preferences.hpp"
#include "cell.hpp"
#include "boundaryCell.hpp"
#include "workset.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "parameterManager.hpp"


void static assemblyHelp(const string & details) {
  cout << "********** Help and Documentation for the Assembly Manager **********" << endl;
}

class AssemblyManager {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
                  Teuchos::RCP<panzer_stk::STK_Interface> & mesh_, Teuchos::RCP<discretization> & disc_,
                  Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager> & DOF_,
                  Teuchos::RCP<ParameterManager> & params_,
                  const int & numElemPerCell_);
  
  
  // ========================================================================================
  // ========================================================================================
  
  void createCells();
    
  // ========================================================================================
  // ========================================================================================
  
  void createWorkset();
  
  // ========================================================================================
  // ========================================================================================
  
  void updateJacDBC(matrix_RCP & J, const vector<GO> & dofs, const bool & compute_disc_sens);
  
  void updateJacDBC(matrix_RCP & J, const vector<LO> & dofs, const bool & compute_disc_sens);
    
  // ========================================================================================
  // ========================================================================================

  void setInitial(vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                  const bool & lumpmass=false);

  // ========================================================================================
  // ========================================================================================
  
  void setDirichlet(vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                    const ScalarT & time, const bool & lumpmass=false);
  
  void setInitial(vector_RCP & initial, const bool & useadjoint);

  // ========================================================================================
  // ========================================================================================
  
  void assembleJacRes(vector_RCP & u, vector_RCP & phi,
                      const bool & compute_jacobian, const bool & compute_sens,
                      const bool & compute_disc_sens,
                      vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                      const ScalarT & current_time, const bool & useadjoint,
                      const bool & store_adjPrev,
                      const int & num_active_params, vector_RCP & Psol,
                      const bool & is_final_time,
                      const ScalarT & deltat);
  
  
  void assembleJacRes(vector_RCP & u, vector_RCP & phi,
                      const bool & compute_jacobian, const bool & compute_sens,
                      const bool & compute_disc_sens,
                      vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                      const ScalarT & current_time, const bool & useadjoint,
                      const bool & store_adjPrev,
                      const int & num_active_params, vector_RCP & Psol,
                      const bool & is_final_time, const int & block,
                      const ScalarT & deltat);
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void dofConstraints(matrix_RCP & J, vector_RCP & res, const ScalarT & current_time,
                      const bool & compute_jacobian, const bool & compute_disc_sens);
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void resetPrevSoln();
  
  void resetStageSoln();
  
  void updateStageNumber(const int & stage);
  
  void updateStageSoln();
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void performGather(const size_t & block, const vector_RCP & vec, const int & type,
                     const size_t & index);
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void performBoundaryGather(const size_t & block, const vector_RCP & vec, const int & type,
                             const size_t & index);
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void insert(matrix_RCP & J, vector_RCP & res,
              Kokkos::View<ScalarT***,HostDevice> local_res,
              Kokkos::View<ScalarT***,HostDevice> local_J,
              LIDView_host LIDs, LIDView_host paramLIDs,
              const bool & compute_jacobian, const bool & compute_disc_sens);
    
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<Teuchos::ParameterList> settings;
  
  // Need
  vector<string> blocknames;
  vector<vector<string> > varlist;
  vector<LO> numVars;
  int numElemPerCell;
  
  Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
  Teuchos::RCP<discretization> disc;
  Teuchos::RCP<physics> phys;
  
  size_t globalParamUnknowns;
  int verbosity, milo_debug_level;
  
  vector<Teuchos::RCP<CellMetaData> > cellData;
  vector<vector<Teuchos::RCP<cell> > > cells;
  vector<vector<Teuchos::RCP<BoundaryCell> > > boundaryCells;
  vector<Teuchos::RCP<workset> > wkset;
  
  bool usestrongDBCs, use_meas_as_dbcs, multiscale, useNewBCs, isTransient, use_atomics;
  string assembly_partitioning;
  Teuchos::RCP<panzer::DOFManager> DOF;
  vector<bool> assemble_volume_terms, assemble_boundary_terms, assemble_face_terms; // use basis functions in assembly
  vector<bool> build_volume_terms, build_boundary_terms, build_face_terms; // set up basis function
  Kokkos::View<bool*,HostDevice> isFixedDOF;
  
private:
  
  Teuchos::RCP<MpiComm> Comm;
  Teuchos::RCP<ParameterManager> params;
  
  Teuchos::RCP<Teuchos::Time> assemblytimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - total assembly");
  Teuchos::RCP<Teuchos::Time> gathertimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - gather");
  Teuchos::RCP<Teuchos::Time> phystimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - physics evaluation");
  Teuchos::RCP<Teuchos::Time> boundarytimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - boundary evaluation");
  Teuchos::RCP<Teuchos::Time> inserttimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - insert");
  Teuchos::RCP<Teuchos::Time> dbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::dofConstraints()");
  Teuchos::RCP<Teuchos::Time> completetimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - fill complete");
  Teuchos::RCP<Teuchos::Time> msprojtimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - multiscale projection");
  Teuchos::RCP<Teuchos::Time> setinittimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::setInitial()");
  Teuchos::RCP<Teuchos::Time> setdbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::setDirichlet()");
  Teuchos::RCP<Teuchos::Time> celltimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::createCells()");
  Teuchos::RCP<Teuchos::Time> wksettimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::createWorkset()");
  
};

#endif
