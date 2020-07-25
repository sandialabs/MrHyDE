/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SUBGRIDFEM_H
#define SUBGRIDFEM_H

#include "trilinos.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"

#include "preferences.hpp"
#include "cell.hpp"
#include "boundaryCell.hpp"
#include "subgridMeshFactory.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "assemblyManager.hpp"
#include "subgridTools.hpp"
#include "parameterManager.hpp"
#include "subgridLocalData.hpp"
#include "subgridFEM_solver.hpp"
#include "postprocessManager.hpp"

class SubGridFEM : public SubGridModel {
public:
  
  SubGridFEM() {} ;
  
  ~SubGridFEM() {};
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  SubGridFEM(const Teuchos::RCP<MpiComm> & LocalComm_,
             Teuchos::RCP<Teuchos::ParameterList> & settings_,
             topo_RCP & macro_cellTopo_, int & num_macro_time_steps_,
             ScalarT & macro_deltat_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  int addMacro(DRV & macronodes_,
               Kokkos::View<int****,HostDevice> & macrosideinfo_,
               LIDView macroLIDs,
               Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & macroorientation);
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void setUpSubgridModels();
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void finalize();
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void addMeshData();
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void subgridSolver(Kokkos::View<ScalarT***,AssemblyDevice> gl_u,
                     Kokkos::View<ScalarT***,AssemblyDevice> gl_phi,
                     const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                     const bool & compute_jacobian, const bool & compute_sens,
                     const int & num_active_params,
                     const bool & compute_disc_sens, const bool & compute_aux_sens,
                     workset & macrowkset,
                     const int & usernum, const int & macroelemindex,
                     Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Re-seed the global parameters
  ///////////////////////////////////////////////////////////////////////////////////////
  
  
  void sacadoizeParams(const bool & seed_active, const int & num_active_params);
    
  ///////////////////////////////////////////////////////////////////////////////////////
  // Store macro-dofs and flux (for ML-based subgrid)
  ///////////////////////////////////////////////////////////////////////////////////////

  void storeFluxData(Kokkos::View<ScalarT***,AssemblyDevice> lambda, Kokkos::View<AD**,AssemblyDevice> flux);
  
  //////////////////////////////////////////////////////////////
  // Compute the initial values for the subgrid solution
  //////////////////////////////////////////////////////////////
  
  void setInitial(Teuchos::RCP<LA_MultiVector> & initial, const int & usernum, const bool & useadjoint);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the error for verification
  ///////////////////////////////////////////////////////////////////////////////////////
  
  //Kokkos::View<ScalarT**,AssemblyDevice> computeError(const ScalarT & time, const int & usernum);
  Kokkos::View<ScalarT**,AssemblyDevice> computeError(vector<pair<size_t, string> > & sub_error_list,
                                                      const vector<ScalarT> & times);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the objective function
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD*,AssemblyDevice> computeObjective(const string & response_type, const int & seedwhat,
                                                    const ScalarT & time, const int & usernum);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Write the solution to a file
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void writeSolution(const string & filename, const int & usernum);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Add in the sensor data
  ////////////////////////////////////////////////////////////////////////////////
  
  void addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                  const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                  const vector<basis_RCP> & basisTypes, const int & usernum);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Assemble the projection (mass) matrix
  ////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<LA_CrsMatrix>  getProjectionMatrix();

  ////////////////////////////////////////////////////////////////////////////////
  // Assemble the projection matrix using ip and basis values from another subgrid model
  ////////////////////////////////////////////////////////////////////////////////

  Teuchos::RCP<LA_CrsMatrix> getProjectionMatrix(DRV & ip, DRV & wts,
                                                 pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo);

  
  ////////////////////////////////////////////////////////////////////////////////
  // Get an empty vector
  ////////////////////////////////////////////////////////////////////////////////

  vector_RCP getVector();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Get the integration points
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV getIP();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Get the integration weights
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV getIPWts();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Evaluate the basis functions at a set of points
  ////////////////////////////////////////////////////////////////////////////////
  
  pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis2(const DRV & pts);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Evaluate the basis functions at a set of points
  // TMW: what is this function for???
  ////////////////////////////////////////////////////////////////////////////////
  
  pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis(const DRV & pts);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
  ////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<LA_CrsMatrix>  getEvaluationMatrix(const DRV & newip, Teuchos::RCP<LA_Map> & ip_map);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Get the subgrid cell GIDs
  ////////////////////////////////////////////////////////////////////////////////
  
  LIDView getCellLIDs(const int & cellnum);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Update the subgrid parameters (will be depracated)
  ////////////////////////////////////////////////////////////////////////////////
  
  void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames);
  
  ////////////////////////////////////////////////////////////////////////////////
  // TMW: Is the following functions used/required ???
  ////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT**,AssemblyDevice> getCellFields(const int & usernum, const ScalarT & time);
  
  // ========================================================================================
  //
  // ========================================================================================
  
  //void performGather(const size_t & block, const Teuchos::RCP<LA_MultiVector> & vec, const size_t & type,
  //                   const size_t & index) const ;
  
  // ========================================================================================
  //
  // ========================================================================================
  
  //void performBoundaryGather(const size_t & block, const Teuchos::RCP<LA_MultiVector> & vec, const size_t & type,
  //                           const size_t & index) const ;
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data);
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void updateLocalData(const int & usernum);
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  // Static - do not depend on macro-element
  int dimension, time_steps;
  ScalarT initial_time, final_time;
  Teuchos::RCP<Teuchos::ParameterList> settings;
  string macroshape, shape, multiscale_method, error_type;
  int nummacroVars, subgridverbose, numrefine, assemble_together;
  topo_RCP cellTopo, macro_cellTopo;
  
  vector<string> stoch_param_types;
  vector<ScalarT> stoch_param_means, stoch_param_vars, stoch_param_mins, stoch_param_maxs;
  int num_stochclassic_params, num_active_params;
  vector<string> stochclassic_param_names;
    
  vector<string> discparamnames;
  Teuchos::RCP<physics> sub_physics;
  Teuchos::RCP<AssemblyManager> sub_assembler;
  Teuchos::RCP<ParameterManager> sub_params;
  Teuchos::RCP<SubGridFEM_Solver> sub_solver;
  Teuchos::RCP<meshInterface> sub_mesh;
  Teuchos::RCP<discretization> sub_disc;
  Teuchos::RCP<PostprocessManager> sub_postproc;
  vector<Teuchos::RCP<LA_MultiVector> > Psol;
  
  // Dynamic - depend on the macro-element
  vector<Teuchos::RCP<SubGridLocalData> > localData;
  
  int num_macro_time_steps;
  ScalarT macro_deltat;
  bool write_subgrid_state;
  
  // Collection of users
  vector<vector<Teuchos::RCP<cell> > > cells;
  vector<vector<Teuchos::RCP<BoundaryCell> > > boundaryCells;
  
  bool have_mesh_data, have_rotations, have_rotation_phi, compute_mesh_data;
  bool have_multiple_data_files;
  string mesh_data_tag, mesh_data_pts_tag;
  int number_mesh_data_files, numSeeds;
  bool is_final_time;
  vector<int> randomSeeds;
  
  // Storage of macro solution and flux (with derivatives)
  //Teuchos::RCP<SolutionStorage<LA_MultiVector> > fluxdata;
  bool store_aux_and_flux = false;
  vector<Kokkos::View<ScalarT***,AssemblyDevice> > auxdata;
  vector<Kokkos::View<AD***,AssemblyDevice> > fluxdata;
  //vector<vector<std:pair<size_t,string> > > subgrid_error_list;
  
  
  // Timers
  Teuchos::RCP<Teuchos::Time> sgfemSolverTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridSolver()");
  Teuchos::RCP<Teuchos::Time> sgfemInitialTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridSolver - set initial conditions");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver()");
  Teuchos::RCP<Teuchos::Time> sgfemSolnSensTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridSolnSens()");
  Teuchos::RCP<Teuchos::Time> sgfemSolnSensLinearSolverTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridSolnSens - linear solver");
  Teuchos::RCP<Teuchos::Time> sgfemFluxTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::updateFlux()");
  Teuchos::RCP<Teuchos::Time> sgfemFluxWksetTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::updateFlux - update workset");
  Teuchos::RCP<Teuchos::Time> sgfemFluxCellTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::updateFlux - cell computation");
  Teuchos::RCP<Teuchos::Time> sgfemComputeAuxBasisTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - compute aux basis functions");
  Teuchos::RCP<Teuchos::Time> sgfemSubMeshTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create subgrid meshes");
  Teuchos::RCP<Teuchos::Time> sgfemLinearAlgebraSetupTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - setup linear algebra");
  Teuchos::RCP<Teuchos::Time> sgfemTotalAddMacroTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro()");
  Teuchos::RCP<Teuchos::Time> sgfemTotalSetUpTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::setUpSubgridModels()");
  Teuchos::RCP<Teuchos::Time> sgfemMeshDataTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMeshData()");
  Teuchos::RCP<Teuchos::Time> sgfemSubCellTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create subcells");
  Teuchos::RCP<Teuchos::Time> sgfemSubDiscTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create disc. interface");
  Teuchos::RCP<Teuchos::Time> sgfemSubSolverTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create solver interface");
  Teuchos::RCP<Teuchos::Time> sgfemSubICTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create vectors");
  Teuchos::RCP<Teuchos::Time> sgfemSubSideinfoTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::addMacro - create side info");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverAllocateTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - allocate objects");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverSetSolnTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - set local soln");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverJacResTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - Jacobian/residual");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverInsertTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - insert");
  Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverSolveTimer = Teuchos::TimeMonitor::getNewCounter("MILO::subgridFEM::subgridNonlinearSolver - solve");
  
  
  
};
#endif
