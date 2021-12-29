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

#ifndef SUBGRIDFEM_SOLVER_H
#define SUBGRIDFEM_SOLVER_H

#include "trilinos.hpp"

#include "preferences.hpp"
#include "assemblyManager.hpp"
#include "solverManager.hpp" // includes belos, muelu, amesos2
#include "parameterManager.hpp"
#include "subgridMacroData.hpp"
#include <BelosBlockGmresSolMgr.hpp>

namespace MrHyDE {
  
  class SubGridFEM_Solver {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>   SG_CrsMatrix;
    typedef Tpetra::CrsGraph<LO,GO,SubgridSolverNode>            SG_CrsGraph;
    typedef Tpetra::Export<LO, GO, SubgridSolverNode>            SG_Export;
    typedef Tpetra::Import<LO, GO, SubgridSolverNode>            SG_Import;
    typedef Tpetra::Map<LO, GO, SubgridSolverNode>               SG_Map;
    typedef Tpetra::Operator<ScalarT,LO,GO,SubgridSolverNode>    SG_Operator;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> SG_MultiVector;
    typedef Teuchos::RCP<SG_MultiVector>                         vector_RCP;
    typedef Teuchos::RCP<SG_CrsMatrix>                           matrix_RCP;
    typedef Belos::LinearProblem<ScalarT, SG_MultiVector, SG_Operator> SG_LinearProblem;

    
  public:
    
    SubGridFEM_Solver() {} ;
    
    ~SubGridFEM_Solver() {};
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    SubGridFEM_Solver(const Teuchos::RCP<MpiComm> & LocalComm_,
                      Teuchos::RCP<Teuchos::ParameterList> & settings_,
                      Teuchos::RCP<MeshInterface> & mesh,
                      Teuchos::RCP<DiscretizationInterface> & disc,
                      Teuchos::RCP<PhysicsInterface> & physics,
                      Teuchos::RCP<AssemblyManager<SubgridSolverNode> > & assembler,
                      Teuchos::RCP<ParameterManager<SubgridSolverNode> > & params,
                      ScalarT & macro_deltat_,
                      size_t & numMacroDOF);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void solve(View_Sc3 coarse_u,
               View_Sc3 coarse_phi,
               Teuchos::RCP<SG_MultiVector> & prev_u,
               Teuchos::RCP<SG_MultiVector> & prev_phi,
               Teuchos::RCP<SG_MultiVector> & disc_params,
               const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
               const bool & compute_jacobian, const bool & compute_sens,
               const int & num_active_params,
               const bool & compute_disc_sens, const bool & compute_aux_sens,
               workset & macrowkset,
               const int & usernum, const int & macroelemindex,
               Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Subgrid Nonlinear Solver
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void nonlinearSolver(Teuchos::RCP<SG_MultiVector> & sub_u,
                         Teuchos::RCP<SG_MultiVector> & sub_phi,
                         Teuchos::RCP<SG_MultiVector> & sub_params, View_Sc3 lambda,
                         const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                         const int & num_active_params, const ScalarT & alpha, const int & usernum,
                         const bool & store_adjPrev);

    //////////////////////////////////////////////////////////////
    // Fix the diagonal
    //////////////////////////////////////////////////////////////

    template<class LIDViewType, class MatType>
    void fixDiagonal(LIDViewType LIDs, MatType localMatrix, const int startpoint);

    //////////////////////////////////////////////////////////////
    // Compute the derivative of the local solution w.r.t coarse
    // solution or w.r.t parameters
    //////////////////////////////////////////////////////////////
    
    void computeSolnSens(Teuchos::RCP<SG_MultiVector> & d_sub_u, const bool & compute_sens,
                         Teuchos::RCP<SG_MultiVector> & sub_u,
                         Teuchos::RCP<SG_MultiVector> & sub_phi,
                         Teuchos::RCP<SG_MultiVector> & sub_param, View_Sc3 lambda,
                         const ScalarT & time,
                         const bool & isTransient, const bool & isAdjoint, const int & num_active_params, const ScalarT & alpha,
                         const ScalarT & lambda_scale, const int & usernum,
                         Kokkos::View<ScalarT**,AssemblyDevice> subgradient);
    
    //////////////////////////////////////////////////////////////
    // Update the residual for the subgrid solution sensitivity wrt coarse DOFs
    //////////////////////////////////////////////////////////////

    template<class ResViewType, class DataViewType>
    void updateResSens(const bool & use_cells, const int & usernum, const int & elem, ResViewType dres_view,
                       DataViewType data, const bool & data_avail,
                       const bool & use_host_LIDs, const bool & compute_sens);
    
    template<class LIDViewType, class ResViewType, class DataViewType>
    void updateResSens(ResViewType res, DataViewType data, LIDViewType LIDs, const bool & compute_sens );

    //////////////////////////////////////////////////////////////
    // Update the flux
    //////////////////////////////////////////////////////////////
    
    void updateFlux(const Teuchos::RCP<SG_MultiVector> & u,
                    const Teuchos::RCP<SG_MultiVector> & d_u,
                    View_Sc3 lambda,
                    const Teuchos::RCP<SG_MultiVector> & disc_params,
                    const bool & compute_sens, const int macroelemindex,
                    const ScalarT & time, workset & macrowkset,
                    const int & usernum);
    
    
    template<class ViewType>
    void updateFlux(ViewType u_kv,
                    ViewType du_kv,
                    View_Sc3 lambda,
                    ViewType dp_kv,
                    const bool & compute_sens, const int macroelemindex,
                    const ScalarT & time, workset & macrowkset,
                    const int & usernum);

    ///////////////////////////////////////////////////////////////////////////////////////
    // Store macro-dofs and flux (for ML-based subgrid)
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void storeFluxData(View_Sc3 lambda, View_AD2 flux);
    
    //////////////////////////////////////////////////////////////
    // Compute the initial values for the subgrid solution
    //////////////////////////////////////////////////////////////
    
    void setInitial(Teuchos::RCP<SG_MultiVector> & initial, const int & usernum, const bool & useadjoint);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Add in the sensor data
    ////////////////////////////////////////////////////////////////////////////////
    
    void addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                    const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                    const vector<basis_RCP> & basisTypes, const int & usernum);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Assemble the projection (mass) matrix
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<SG_CrsMatrix>  getProjectionMatrix();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Assemble the projection matrix using ip and basis values from another subgrid model
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<SG_CrsMatrix> getProjectionMatrix(DRV & ip, DRV & wts,
                                                   std::pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo);
    
    
    ////////////////////////////////////////////////////////////////////////////////
    // Get an empty vector
    ////////////////////////////////////////////////////////////////////////////////
    
    vector_RCP getVector();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Evaluate the basis functions at a set of points
    ////////////////////////////////////////////////////////////////////////////////
    
    std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis2(const DRV & pts);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<SG_CrsMatrix>  getEvaluationMatrix(const DRV & newip, Teuchos::RCP<SG_Map> & ip_map);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Update the subgrid parameters (will be depracated)
    ////////////////////////////////////////////////////////////////////////////////
    
    void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    void performGather(const size_t & block, const Teuchos::RCP<SG_MultiVector> & vec, const size_t & type,
                       const size_t & index);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    template<class ViewType>
    void performGather(const size_t & block, ViewType vec_dev, const size_t & type);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    template<class ViewType>
    void performBoundaryGather(const size_t & block, ViewType vec_dev, const size_t & type);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    // Static - do not depend on macro-element
    int dimension, time_steps, verbosity, debug_level;
    ScalarT initial_time, final_time;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    string macroshape, shape, multiscale_method, error_type;
    int nummacroVars, numrefine, assemble_together;
    topo_RCP cellTopo, macro_cellTopo;
    ScalarT macro_deltat;
    
    // Linear algebra / solver objects
    Teuchos::RCP<SG_Map> param_overlapped_map;
    Teuchos::RCP<SG_MultiVector> res, res_over, d_um, du, du_glob;
    Teuchos::RCP<SG_MultiVector> u, phi;
    Teuchos::RCP<SG_MultiVector> d_sub_res_overm, d_sub_resm, d_sub_u_prevm, d_sub_u_overm;
    Teuchos::RCP<SG_CrsMatrix>  J, sub_J_over;
    
    string amesos_solver_type;
    Teuchos::RCP<Amesos2::Solver<SG_CrsMatrix,SG_MultiVector> > Am2Solver;
    Teuchos::RCP<SG_MultiVector> SG_rhs, SG_lhs;
    
    Teuchos::RCP<SG_LinearProblem> belos_problem;
    Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, SubgridSolverNode> > belos_M;
    Teuchos::RCP<Teuchos::ParameterList> belosList;
    Teuchos::RCP<Belos::SolverManager<ScalarT, SG_MultiVector, SG_Operator> > belos_solver;
    bool have_belos = false;
    bool have_preconditioner = false, use_preconditioner = true;
    
    ScalarT sub_NLtol;
    int sub_maxNLiter;
    bool have_sym_factor, useDirect;
    
    Teuchos::RCP<SolverManager<SubgridSolverNode> > solver;
    Teuchos::RCP<AssemblyManager<SubgridSolverNode> > assembler;
    
    int num_macro_time_steps;
    bool write_subgrid_state;
    
    bool have_mesh_data, have_rotations, have_rotation_phi, compute_mesh_data;
    bool have_multiple_data_files;
    string mesh_data_tag, mesh_data_pts_tag;
    int number_mesh_data_files, numSeeds;
    bool is_final_time;
    vector<int> randomSeeds;
    
    // Storage of macro solution and flux (with derivatives)
    //Teuchos::RCP<SolutionStorage<SG_MultiVector> > fluxdata;
    bool store_aux_and_flux = false;
    vector<Kokkos::View<ScalarT***,AssemblyDevice> > auxdata;
    vector<Kokkos::View<AD***,AssemblyDevice> > fluxdata;
    
    // Timers
    Teuchos::RCP<Teuchos::Time> sgfemSolverTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridSolver()");
    Teuchos::RCP<Teuchos::Time> sgfemInitialTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridSolver - set initial conditions");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver()");
    Teuchos::RCP<Teuchos::Time> sgfemSolnSensTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridSolnSens()");
    Teuchos::RCP<Teuchos::Time> sgfemSolnSensLinearSolverTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridSolnSens - linear solver");
    Teuchos::RCP<Teuchos::Time> sgfemFluxTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::updateFlux()");
    Teuchos::RCP<Teuchos::Time> sgfemTemplatedFluxTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::updateFlux() - templated");
    Teuchos::RCP<Teuchos::Time> sgfemFluxWksetTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::updateFlux - update workset");
    Teuchos::RCP<Teuchos::Time> sgfemFluxCellTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::updateFlux - cell computation");
    Teuchos::RCP<Teuchos::Time> sgfemAssembleFluxTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::updateFlux - assemble flux");
    Teuchos::RCP<Teuchos::Time> sgfemLinearAlgebraSetupTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::addMacro - setup linear algebra");
    Teuchos::RCP<Teuchos::Time> sgfemSubSolverTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::addMacro - create solver interface");
    Teuchos::RCP<Teuchos::Time> sgfemSubICTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::addMacro - create vectors");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverAllocateTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver - allocate objects");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverSetSolnTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver - set local soln");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverAssemblyTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver - volume/bndry assembly");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverScatterTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver - scatter");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverInsertTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver - insert");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverSolveTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver - solve");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverAmesosSetupTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver - setup Amesos");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverAmesosSymbFactTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver - symbolic factor");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverBelosSetupTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::subgridFEM_solver::subgridNonlinearSolver - setup Belos");
  };
  
}

#endif

