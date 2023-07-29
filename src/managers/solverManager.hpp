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

#ifndef MRHYDE_SOLVER_MANAGER
#define MRHYDE_SOLVER_MANAGER

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "multiscaleManager.hpp"
#include "discretizationInterface.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "postprocessManager.hpp"
#include "solutionStorage.hpp"
#include "linearAlgebraInterface.hpp"

namespace MrHyDE {
  
  template<class Node>
  class SolverManager {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;
    typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
    typedef typename Node::device_type              LA_device;
    
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    SolverManager() {};
    
    ~SolverManager() {};
    
    SolverManager(const Teuchos::RCP<MpiComm> & Comm_,
                  Teuchos::RCP<Teuchos::ParameterList> & settings_,
                  Teuchos::RCP<MeshInterface> & mesh_,
                  Teuchos::RCP<DiscretizationInterface> & disc_,
                  Teuchos::RCP<PhysicsInterface> & phys_,
                  Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                  Teuchos::RCP<ParameterManager<Node> > & params_);
    
    // ========================================================================================
    // ========================================================================================
    
    void completeSetup();

    // ========================================================================================
    // ========================================================================================
    
    void setupExplicitMass();

    // ========================================================================================
    // ========================================================================================

    /**
     * @brief Set the RK Butcher tableau.
     * 
     * The RK scheme can change for each physics set, if desired.
     * This is done within the input file with the physics set sublist under the Solver section.
     * 
     * @todo Add ability to change RK scheme for different blocks.
     * 
     * @param[in] tableau The requested Butcher tableau
     * @param[in] set The set index
     */
    
    void setButcherTableau(const vector<string> & tableau, const int & set);
    
    // ========================================================================================
    // ========================================================================================

    /**
     * @brief Set the BDF weights given a requested order.
     * 
     * The BDF integration weights can be specified for each physics set, if desired.
     * This is done within the input file with the physics set sublist under the Solver section.
     * 
     * @todo Add ability to change time integration for different blocks.
     * 
     * @param[in] order The requested BDF order
     * @param[in] set The set index
     */
    
    void setBackwardDifference(const vector<int> & order, const int & set);
    
    // ========================================================================================
    // ========================================================================================
    
    void setupFixedDOFs(Teuchos::RCP<Teuchos::ParameterList> & settings);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void finalizeWorkset();
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void forwardModel(DFAD & objective);
    
    // ========================================================================================
    // ========================================================================================
    
    void adjointModel(MrHyDE_OptVector & gradient);
    
    // ========================================================================================
    /* solve the problem */
    // ========================================================================================
    
    void steadySolver(DFAD & objective, vector<vector_RCP> & u);
    
    // ========================================================================================
    // ========================================================================================
    
    void transientSolver(vector<vector_RCP> & initial, DFAD & obj, 
                         MrHyDE_OptVector & gradient,
                         ScalarT & start_time, ScalarT & end_time);
    
    // ========================================================================================
    // ========================================================================================
    
    int nonlinearSolver(const size_t & set, vector_RCP & u, vector_RCP & phi);
    
    int explicitSolver(const size_t & set, vector_RCP & u, vector_RCP & phi, const int & stage);

    // ========================================================================================
    // ========================================================================================
    
    void setDirichlet(const size_t & set, vector_RCP & u);
    
    // ========================================================================================
    // ========================================================================================
    
    void projectDirichlet(const size_t & set);
    
    // ========================================================================================
    // ========================================================================================
    
    vector_RCP setInitialParams();
    
    // ========================================================================================
    // ========================================================================================
    
    vector<vector_RCP> setInitial();
    
    // ========================================================================================
    // ========================================================================================
    
    void setBatchID(const LO & bID);
    
    // ========================================================================================
    // ========================================================================================
    
    vector_RCP blankState();
    
    vector<vector_RCP> getRestartSolution();

    vector<vector_RCP> getRestartAdjointSolution();

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void finalizeParams() ;
    
    void finalizeMultiscale() ;
    
    
    void PCG(const size_t & set, matrix_RCP & J, vector_RCP & b, vector_RCP & x, vector_RCP & Minv,
             const ScalarT & tol, const int & maxiter);
    
    void matrixFreePCG(const size_t & set, vector_RCP & b, vector_RCP & x, vector_RCP & Minv,
                       const ScalarT & tol, const int & maxiter);
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<MeshInterface>  mesh;
    Teuchos::RCP<DiscretizationInterface> disc;
    Teuchos::RCP<PhysicsInterface> physics;
    Teuchos::RCP<LinearAlgebraInterface<Node> > linalg;
    Teuchos::RCP<AssemblyManager<Node> > assembler;
    Teuchos::RCP<ParameterManager<Node> > params;
    Teuchos::RCP<PostprocessManager<Node> > postproc;
    Teuchos::RCP<MultiscaleManager> multiscale_manager;
    
    int verbosity, batchID, dimension, gNLiter, debug_level, maxNLiter, subcycles;
    
    // numsteps of BDF scheme
    // numstages of RK
    vector<int> BDForder, startupBDForder, startupSteps, numsteps, numstages; // [set]
    // maximum number of steps and stages required by time integrator
    vector<int> maxnumsteps, maxnumstages; // [set]
    int numEvaluations, maxTimeStepCuts;
    vector<string> ButcherTab, startupButcherTab; // [set]
    
    ScalarT NLtol, NLabstol,final_time, lintol, current_time, initial_time, deltat, amplification_factor;
    
    string solver_type, initial_type;
    
    bool line_search, useL2proj, discretized_stochastic, fully_explicit, use_custom_PCG;
    bool isInitial, isTransient, is_adjoint, is_final_time, usestrongDBCs, use_restart=false;
    bool compute_objective, use_custom_initial_param_guess, store_adjPrev, use_meas_as_dbcs, compute_fwd_sens;
    vector<bool> scalarDirichletData, staticDirichletData, scalarInitialData;
    vector<bool> have_initial_conditions, have_static_Dirichlet_data;
    bool useRelativeTOL, useAbsoluteTOL, allowBacktracking, store_vectors;
    vector<vector<vector<ScalarT> > > scalarDirichletValues, scalarInitialValues; // [set][block][var]
    vector<Teuchos::RCP<LA_MultiVector> > fixedDOF_soln, invdiagMass, diagMass;
    vector<matrix_RCP> explicitMass;
    
    vector<string> blocknames, setnames;
    vector<vector<vector<string> > > varlist;
    
    vector<vector<vector<LO> > > numBasis, useBasis;
    vector<vector<size_t> > maxBasis, numVars;
    
    vector<vector_RCP> res, res_over, du, du_over, restart_solution, restart_adjoint_solution;
    vector<vector_RCP> q_pcg, z_pcg, p_pcg, r_pcg, p_pcg_over, q_pcg_over;
    
    Kokkos::View<ScalarT**,HostDevice> butcher_A; 
    Kokkos::View<ScalarT*,HostDevice> butcher_b, butcher_c;
    
  private:

    Teuchos::RCP<Teuchos::Time> transientsolvertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::transientSolver()");
    Teuchos::RCP<Teuchos::Time> nonlinearsolvertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::nonlinearSolver()");
    Teuchos::RCP<Teuchos::Time> explicitsolvertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::explicitSolver()");
    
    Teuchos::RCP<Teuchos::Time> initsettimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::setInitial()");
    Teuchos::RCP<Teuchos::Time> dbcsettimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::setDirichlet()");
    Teuchos::RCP<Teuchos::Time> dbcprojtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::projectDirichlet()");
    Teuchos::RCP<Teuchos::Time> fixeddofsetuptimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::setupFixedDOFs()");
    Teuchos::RCP<Teuchos::Time> msprojtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::projectDirichlet()");
    Teuchos::RCP<Teuchos::Time> normLAtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::nonlinearSolver() - norm LA");
    Teuchos::RCP<Teuchos::Time> updateLAtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SolverManager::nonlinearSolver() - update LA");
    Teuchos::RCP<Teuchos::Time> PCGtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::PCG - total");
    Teuchos::RCP<Teuchos::Time> PCGApplyOptimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::PCG - apply Op");
    Teuchos::RCP<Teuchos::Time> PCGApplyPrectimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::PCG - apply prec");
    
  };
  
}

#endif
