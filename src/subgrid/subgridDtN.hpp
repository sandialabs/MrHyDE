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

#ifndef MRHYDE_SUBGRIDDTN_H
#define MRHYDE_SUBGRIDDTN_H

#include "trilinos.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"

#include "preferences.hpp"
#include "group.hpp"
#include "boundaryGroup.hpp"
#include "subgridMeshFactory.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "assemblyManager.hpp"
#include "subgridTools.hpp"
#include "parameterManager.hpp"
#include "subgridMacroData.hpp"
#include "subgridDtN_solver.hpp"
#include "postprocessManager.hpp"

namespace MrHyDE {
  
  class SubGridDtN : public SubGridModel {
        
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>   SG_CrsMatrix;
    typedef Tpetra::CrsGraph<LO,GO,SubgridSolverNode>            SG_CrsGraph;
    typedef Tpetra::Export<LO, GO, SubgridSolverNode>            SG_Export;
    typedef Tpetra::Import<LO, GO, SubgridSolverNode>            SG_Import;
    typedef Tpetra::Map<LO, GO, SubgridSolverNode>               SG_Map;
    typedef Tpetra::Operator<ScalarT,LO,GO,SubgridSolverNode>    SG_Operator;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> SG_MultiVector;
    typedef Teuchos::RCP<SG_MultiVector>                         vector_RCP;
    typedef Teuchos::RCP<SG_CrsMatrix>                           matrix_RCP;
    
  public:
    
    SubGridDtN() {} ;
    
    ~SubGridDtN() {};
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    SubGridDtN(const Teuchos::RCP<MpiComm> & LocalComm_,
               Teuchos::RCP<Teuchos::ParameterList> & settings_,
               topo_RCP & macro_cellTopo_, int & num_macro_time_steps_,
               ScalarT & macro_deltat_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    int addMacro(DRV & macronodes_,
                 Kokkos::View<int****,HostDevice> & macrosideinfo_,
                 LIDView macroLIDs,
                 Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & macroorientation);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void setUpSubgridModels();
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void finalize(const int & globalSize, const int & globalPID, const bool & write_subgrid_soln,
                  vector<string> & appends);
    
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
                       const int & macrogrp, const int & macroelemindex,
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
    
    void setInitial(Teuchos::RCP<SG_MultiVector> & initial, const int & macrogrp, const bool & useadjoint);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute the error for verification
    ///////////////////////////////////////////////////////////////////////////////////////
    
    vector<std::pair<string, string> > getErrorList();
    
    //Kokkos::View<ScalarT**,AssemblyDevice> computeError(const ScalarT & time, const int & macrogrp);
    Kokkos::View<ScalarT**,HostDevice> computeError(vector<std::pair<string, string> > & sub_error_list,
                                                    const vector<ScalarT> & times);
    
    Kokkos::View<ScalarT*,HostDevice> computeError(const ScalarT & times);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Compute the objective function
    ///////////////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<AD*,AssemblyDevice> computeObjective(const string & response_type, const int & seedwhat,
                                                      const ScalarT & time, const int & macrogrp);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Write the solution to a file
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setupCombinedExodus(vector<string> & appends);
    
    //void writeSolution(const string & filename, const int & macrogrp);
    
    void writeSolution(const string & filename);
    
    void writeSolution(const ScalarT & time, const string & append="");
    
    ////////////////////////////////////////////////////////////////////////////////
    // Add in the sensor data
    ////////////////////////////////////////////////////////////////////////////////
    
    void addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points, const ScalarT & sensor_loc_tol,
                    const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data, const bool & have_sensor_data,
                    const vector<basis_RCP> & basisTypes, const int & macrogrp);
    
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
    
    std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis2(const DRV & pts);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Evaluate the basis functions at a set of points
    // TMW: what is this function for???
    ////////////////////////////////////////////////////////////////////////////////
    
    //std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis(const DRV & pts);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Get the matrix mapping the DOFs to a set of integration points on a reference macro-element
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<SG_CrsMatrix>  getEvaluationMatrix(const DRV & newip, Teuchos::RCP<SG_Map> & ip_map);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Get the subgrid cell GIDs
    ////////////////////////////////////////////////////////////////////////////////
    
    LIDView getCellLIDs(const int & cellnum);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Update the subgrid parameters (will be depracated)
    ////////////////////////////////////////////////////////////////////////////////
    
    void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    void updateLocalData(const int & macrogrp);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    // Static - do not depend on macro-element
    int dimension, time_steps, verbosity, debug_level;
    ScalarT initial_time, final_time;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    string macroshape, shape, multiscale_method, combined_mesh_filename, mesh_type, mesh_file;
    int nummacroVars, numrefine, assemble_together;
    topo_RCP cellTopo, macro_cellTopo;
    
    vector<string> stoch_param_types;
    vector<ScalarT> stoch_param_means, stoch_param_vars, stoch_param_mins, stoch_param_maxs;
    int num_stochclassic_params, num_active_params;
    vector<string> stochclassic_param_names;
    
    vector<string> discparamnames;
    Teuchos::RCP<PhysicsInterface> sub_physics;
    Teuchos::RCP<AssemblyManager<SubgridSolverNode> > sub_assembler;
    Teuchos::RCP<ParameterManager<SubgridSolverNode> > sub_params;
    Teuchos::RCP<SubGridDtN_Solver> sub_solver;
    Teuchos::RCP<MeshInterface> sub_mesh;
    Teuchos::RCP<panzer_stk::STK_Interface> combined_mesh;
    Teuchos::RCP<DiscretizationInterface> sub_disc;
    Teuchos::RCP<PostprocessManager<SubgridSolverNode> > sub_postproc;
    vector<Teuchos::RCP<SG_MultiVector> > Psol;
    
    // Dynamic - depend on the macro-element
    vector<Teuchos::RCP<SubGridMacroData> > macroData;
    
    int num_macro_time_steps;
    ScalarT macro_deltat;
    
    // Collection of users
    vector<vector<Teuchos::RCP<Group> > > groups;
    vector<vector<Teuchos::RCP<BoundaryGroup> > > boundary_groups;
    
    bool have_mesh_data, have_rotations, have_rotation_phi, compute_mesh_data;
    bool have_multiple_data_files;
    string mesh_data_tag, mesh_data_pts_tag;
    int number_mesh_data_files, numSeeds;
    bool is_final_time;
    vector<int> randomSeeds;
    
    // Storage of macro solution and flux (with derivatives)
    vector<Kokkos::View<ScalarT***,AssemblyDevice> > auxdata;
    vector<Kokkos::View<AD***,AssemblyDevice> > fluxdata;
    
    // Timers
    Teuchos::RCP<Teuchos::Time> sgfemSolverTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridSolver()");
    Teuchos::RCP<Teuchos::Time> sgfemInitialTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridSolver - set initial conditions");
    Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridNonlinearSolver()");
    Teuchos::RCP<Teuchos::Time> sgfemSolnSensTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridSolnSens()");
    Teuchos::RCP<Teuchos::Time> sgfemSolnSensLinearSolverTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridSolnSens - linear solver");
    Teuchos::RCP<Teuchos::Time> sgfemFluxTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::updateFlux()");
    Teuchos::RCP<Teuchos::Time> sgfemFluxWksetTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::updateFlux - update workset");
    Teuchos::RCP<Teuchos::Time> sgfemFluxCellTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::updateFlux - cell computation");
    Teuchos::RCP<Teuchos::Time> sgfemComputeAuxBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMacro - compute aux basis functions");
    Teuchos::RCP<Teuchos::Time> sgfemSubMeshTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMacro - create subgrid meshes");
    Teuchos::RCP<Teuchos::Time> sgfemLinearAlgebraSetupTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMacro - setup linear algebra");
    Teuchos::RCP<Teuchos::Time> sgfemTotalAddMacroTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMacro()");
    Teuchos::RCP<Teuchos::Time> sgfemTotalSetUpTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::setUpSubgridModels()");
    Teuchos::RCP<Teuchos::Time> sgfemMeshDataTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMeshData()");
    Teuchos::RCP<Teuchos::Time> sgfemSubCellTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMacro - create subgroups");
    Teuchos::RCP<Teuchos::Time> sgfemSubDiscTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMacro - create disc. interface");
    Teuchos::RCP<Teuchos::Time> sgfemSubSolverTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMacro - create solver interface");
    Teuchos::RCP<Teuchos::Time> sgfemSubICTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMacro - create vectors");
    Teuchos::RCP<Teuchos::Time> sgfemSubSideinfoTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::addMacro - create side info");
    //Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverAllocateTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridNonlinearSolver - allocate objects");
    //Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverSetSolnTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridNonlinearSolver - set local soln");
    //Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverJacResTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridNonlinearSolver - Jacobian/residual");
    //Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverInsertTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridNonlinearSolver - insert");
    //Teuchos::RCP<Teuchos::Time> sgfemNonlinearSolverSolveTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::subgridNonlinearSolver - solve");
    Teuchos::RCP<Teuchos::Time> sgfemCombinedMeshSetupTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::setupCombinedExodus()");
    Teuchos::RCP<Teuchos::Time> sgfemCombinedMeshOutputTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SubGridDtN::writeSolution() - combined version");
    
  };
  
}

#endif

