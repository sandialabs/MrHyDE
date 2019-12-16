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
  
  AssemblyManager(const Teuchos::RCP<LA_MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
                  Teuchos::RCP<panzer_stk::STK_Interface> & mesh_, Teuchos::RCP<discretization> & disc_,
                  Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager> & DOF_,
                  vector<vector<Teuchos::RCP<cell> > > & cells_,
                  vector<vector<Teuchos::RCP<BoundaryCell> > > & boundaryCells_,
                  Teuchos::RCP<ParameterManager> & params_);
  
  // ========================================================================================
  // ========================================================================================
  
  void createWorkset();
  
  // ========================================================================================
  // ========================================================================================
  
  void updateJacDBC(matrix_RCP & J, size_t & e, size_t & block, int & fieldNum,
                    size_t & localSideId, const bool & compute_disc_sens);
  
  // ========================================================================================
  // ========================================================================================
  
  void updateJacDBC(matrix_RCP & J, const vector<GO> & dofs, const bool & compute_disc_sens);
  
  // ========================================================================================
  // ========================================================================================
  
  void updateResDBC(vector_RCP & resid, size_t & e, size_t & block, int & fieldNum,
                    size_t & localSideId);
  
  // ========================================================================================
  // ========================================================================================
  
  void updateResDBC(vector_RCP & resid, const vector<GO> & dofs);
  
  
  // ========================================================================================
  // ========================================================================================
  
  void updateResDBCsens(vector_RCP & resid, size_t & e, size_t & block, int & fieldNum, size_t & localSideId,
                        const std::string & gside, const ScalarT & current_time);
  
  // ========================================================================================
  // ========================================================================================
  
  //void setDirichlet(vector_RCP & initial);

  // ========================================================================================
  // ========================================================================================

  void setInitial(vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint);

  void setInitial(vector_RCP & initial, const bool & useadjoint);

  // ========================================================================================
  // ========================================================================================
  
  void assembleJacRes(vector_RCP & u, vector_RCP & u_dot,
                      vector_RCP & phi, vector_RCP & phi_dot,
                      const ScalarT & alpha, const ScalarT & beta,
                      const bool & compute_jacobian, const bool & compute_sens,
                      const bool & compute_disc_sens,
                      vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                      const ScalarT & current_time, const bool & useadjoint,
                      const bool & store_adjPrev,
                      const int & num_active_params, vector_RCP & Psol,
                      const bool & is_final_time);
  
  
  void assembleJacRes(vector_RCP & u, vector_RCP & u_dot,
                      vector_RCP & phi, vector_RCP & phi_dot,
                      const ScalarT & alpha, const ScalarT & beta,
                      const bool & compute_jacobian, const bool & compute_sens,
                      const bool & compute_disc_sens,
                      vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                      const ScalarT & current_time, const bool & useadjoint,
                      const bool & store_adjPrev,
                      const int & num_active_params, vector_RCP & Psol,
                      const bool & is_final_time, const int & block);
  
  
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
  
  void insert(matrix_RCP & J, vector_RCP & res, Kokkos::View<ScalarT***,AssemblyDevice> & local_res,
              Kokkos::View<ScalarT***,AssemblyDevice> & local_J, Kokkos::View<ScalarT***,AssemblyDevice> & local_Jdot,
              Kokkos::View<GO**,HostDevice> & GIDs, Kokkos::View<GO**,HostDevice> & paramGIDs,
              const bool & compute_jacobian, const bool & compute_disc_sens, const ScalarT & alpha);
    
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  // Need
  vector<string> blocknames;
  vector<vector<string> > varlist;
  vector<LO> numVars;
  
  Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
  Teuchos::RCP<discretization> disc;
  Teuchos::RCP<physics> phys;
  
  size_t globalParamUnknowns;
  int verbosity, milo_debug_level;
  vector<vector<Teuchos::RCP<cell> > > cells;
  vector<vector<Teuchos::RCP<BoundaryCell> > > boundaryCells;
  vector<Teuchos::RCP<workset> > wkset;
  
  bool usestrongDBCs, use_meas_as_dbcs, multiscale, useNewBCs;
  Teuchos::RCP<const panzer::DOFManager> DOF;
  
private:
  
  Teuchos::RCP<LA_MpiComm> Comm;
  Teuchos::RCP<ParameterManager> params;
  
  Teuchos::RCP<Teuchos::Time> assemblytimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - total assembly");
  Teuchos::RCP<Teuchos::Time> gathertimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - gather");
  Teuchos::RCP<Teuchos::Time> phystimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - physics evaluation");
  Teuchos::RCP<Teuchos::Time> boundarytimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - boundary evaluation");
  Teuchos::RCP<Teuchos::Time> inserttimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - insert");
  Teuchos::RCP<Teuchos::Time> dbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - strong Dirichlet BCs");
  Teuchos::RCP<Teuchos::Time> completetimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - fill complete");
  Teuchos::RCP<Teuchos::Time> msprojtimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - multiscale projection");
  
};

#endif
