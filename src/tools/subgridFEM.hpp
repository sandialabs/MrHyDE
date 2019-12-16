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
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"

#include "preferences.hpp"
#include "cell.hpp"
#include "boundaryCell.hpp"
#include "subgridMeshFactory.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "discretizationTools.hpp"
#include "assemblyManager.hpp"
#include "solverInterface.hpp"
#include "subgridTools.hpp"
#include "parameterManager.hpp"

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosBlockGmresSolMgr.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

// Amesos includes
#include "Amesos2.hpp"

typedef Belos::LinearProblem<ScalarT, LA_MultiVector, LA_Operator> LA_LinearProblem;

class SubGridFEM : public SubGridModel {
public:
  
  SubGridFEM() {} ;
  
  ~SubGridFEM() {};
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  SubGridFEM(const Teuchos::RCP<LA_MpiComm> & LocalComm_,
             Teuchos::RCP<Teuchos::ParameterList> & settings_,
             topo_RCP & macro_cellTopo_, int & num_macro_time_steps_,
             ScalarT & macro_deltat_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  int addMacro(const DRV macronodes_, Kokkos::View<int****,HostDevice> macrosideinfo_,
               vector<string> & macrosidenames,
               Kokkos::View<GO**,HostDevice> & macroGIDs, Kokkos::View<LO***,HostDevice> & macroindex);
  
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
  // Subgrid Nonlinear Solver
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void subGridNonlinearSolver(Teuchos::RCP<LA_MultiVector> & sub_u, Teuchos::RCP<LA_MultiVector> & sub_u_dot,
                              Teuchos::RCP<LA_MultiVector> & sub_phi, Teuchos::RCP<LA_MultiVector> & sub_phi_dot,
                              Teuchos::RCP<LA_MultiVector> & sub_params, Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                              const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                              const int & num_active_params, const ScalarT & alpha, const int & usernum,
                              const bool & store_adjPrev);
  
  //////////////////////////////////////////////////////////////
  // Compute the derivative of the local solution w.r.t coarse
  // solution or w.r.t parameters
  //////////////////////////////////////////////////////////////
  
  void computeSubGridSolnSens(Teuchos::RCP<LA_MultiVector> & d_sub_u, const bool & compute_sens,
                              Teuchos::RCP<LA_MultiVector> & sub_u, Teuchos::RCP<LA_MultiVector> & sub_u_dot,
                              Teuchos::RCP<LA_MultiVector> & sub_phi, Teuchos::RCP<LA_MultiVector> & sub_phi_dot,
                              Teuchos::RCP<LA_MultiVector> & sub_param, Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                              const ScalarT & time,
                              const bool & isTransient, const bool & isAdjoint, const int & num_active_params, const ScalarT & alpha,
                              const ScalarT & lambda_scale, const int & usernum,
                              Kokkos::View<ScalarT**,AssemblyDevice> subgradient);
  
  //////////////////////////////////////////////////////////////
  // Update the flux
  //////////////////////////////////////////////////////////////
  
  void updateFlux(const Teuchos::RCP<LA_MultiVector> & u,
                  const Teuchos::RCP<LA_MultiVector> & d_u,
                  Kokkos::View<ScalarT***,AssemblyDevice> lambda,
                  const bool & compute_sens, const int macroelemindex,
                  const ScalarT & time, workset & macrowkset,
                  const int & usernum, const ScalarT & fwt);
  
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
  
  Kokkos::View<ScalarT**,AssemblyDevice> computeError(const ScalarT & time, const int & usernum);
  
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
  
  Kokkos::View<GO**,HostDevice> getCellGIDs(const int & cellnum);
  
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
  
  void performGather(const size_t & block, const Teuchos::RCP<LA_MultiVector> & vec, const size_t & type,
                     const size_t & index) const ;
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void performBoundaryGather(const size_t & block, const Teuchos::RCP<LA_MultiVector> & vec, const size_t & type,
                             const size_t & index) const ;
  
  // ========================================================================================
  //
  // ========================================================================================
  
  void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data);
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  // Static - do not depend on macro-element
  int dimension, time_steps;
  ScalarT initial_time, final_time;
  Teuchos::RCP<Teuchos::ParameterList> settings;
  string macroshape, shape, multiscale_method, error_type;
  int nummacroVars, subgridverbose, numrefine;
  topo_RCP cellTopo, macro_cellTopo;
  vector<basis_RCP> basis_pointers;
  
  vector<Kokkos::View<int**,AssemblyDevice> > subgridbcs;
  vector<vector<int> > useBasis;
  
  // Linear algebra / solver objects
  Teuchos::RCP<LA_Map> param_owned_map;
  Teuchos::RCP<LA_Map> param_overlapped_map;
  Teuchos::RCP<LA_Export> param_exporter;
  Teuchos::RCP<LA_Import> param_importer;
  
  Teuchos::RCP<LA_MultiVector> res, res_over, d_um, du, du_glob;//, d_up;//,
  Teuchos::RCP<LA_MultiVector> u, u_dot, phi, phi_dot;
  Teuchos::RCP<LA_MultiVector> d_sub_res_overm, d_sub_resm, d_sub_u_prevm, d_sub_u_overm;
  
  //Teuchos::RCP<LA_CrsGraph> owned_graph, overlapped_graph;
  Teuchos::RCP<LA_CrsMatrix>  J, sub_J_over, M, sub_M_over;
  
  //LA_LinearProblem LinSys;
  
  //Amesos_BaseSolver * AmSolver;
  Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > Am2Solver;
  Teuchos::RCP<LA_MultiVector> LA_rhs, LA_lhs;
  
  Teuchos::RCP<LA_LinearProblem> belos_problem;
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > belos_M;
  Teuchos::RCP<Teuchos::ParameterList> belosList;
  Teuchos::RCP<Belos::SolverManager<ScalarT, LA_MultiVector, LA_Operator> > belos_solver;
  bool have_belos = false;
  
  bool filledJ, filledM;
  vector<string> stoch_param_types;
  vector<ScalarT> stoch_param_means, stoch_param_vars, stoch_param_mins, stoch_param_maxs;
  int num_stochclassic_params, num_active_params;
  vector<string> stochclassic_param_names;
  
  
  ScalarT sub_NLtol;
  int sub_maxNLiter;
  
  
  bool have_sym_factor, useDirect;
  
  vector<string> varlist;
  vector<string> discparamnames;
  Teuchos::RCP<physics> physics_RCP;
  Teuchos::RCP<panzer::DOFManager> DOF;
  Teuchos::RCP<AssemblyManager> sub_assembler;
  Teuchos::RCP<ParameterManager> sub_params;
  Teuchos::RCP<solver> sub_solver;
  Teuchos::RCP<meshInterface> mesh_interface;
  Teuchos::RCP<panzer_stk::STK_Interface> mesh;
  Teuchos::RCP<discretization> disc;
  Teuchos::RCP<FunctionInterface> functionManager;
  
  vector<Teuchos::RCP<LA_MultiVector> > Psol;
  
  // Dynamic - depend on the macro-element
  vector<DRV> macronodes;
  vector<Kokkos::View<int****,HostDevice> > macrosideinfo;
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

