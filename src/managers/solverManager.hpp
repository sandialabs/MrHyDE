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
  /*
  void static solverHelp(const string & details) {
    cout << "********** Help and Documentation for the Solver Interface **********" << endl;
  }
  */
  template<class Node>
  class solver {

    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;
    typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
    typedef typename Node::device_type              LA_device;
  
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    solver(const Teuchos::RCP<MpiComm> & Comm_,
           Teuchos::RCP<Teuchos::ParameterList> & settings_,
           Teuchos::RCP<meshInterface> & mesh_,
           Teuchos::RCP<discretization> & disc_,
           Teuchos::RCP<physics> & phys_,
           Teuchos::RCP<AssemblyManager<Node> > & assembler_,
           Teuchos::RCP<ParameterManager<Node> > & params_);
    
    // ========================================================================================
    // ========================================================================================
    
    void setButcherTableau(const string & tableau);
    
    // ========================================================================================
    // ========================================================================================
    
    void setBackwardDifference(const int & order);
    
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
    
    void adjointModel(vector<ScalarT> & gradient);
    
    // ========================================================================================
    /* solve the problem */
    // ========================================================================================
    
    void steadySolver(DFAD & objective, vector_RCP & u);
    
    void transientSolver(vector_RCP & initial, DFAD & obj, vector<ScalarT> & gradient,
                         ScalarT & start_time, ScalarT & end_time);
    
    // ========================================================================================
    // ========================================================================================
    
    int nonlinearSolver(vector_RCP & u, vector_RCP & phi);
    
    // ========================================================================================
    // ========================================================================================
    
    DFAD computeObjective(const vector_RCP & F_soln, const ScalarT & time, const size_t & tindex);
    
    // ========================================================================================
    // ========================================================================================
    
    void computeSensitivities(vector_RCP & u,
                              vector_RCP & a2, vector<ScalarT> & gradient);
    
    
    // ========================================================================================
    // ========================================================================================
    
    void setDirichlet(vector_RCP & u);
    
    void projectDirichlet();
    
    // ========================================================================================
    // ========================================================================================
    
    vector_RCP setInitialParams();
    
    // ========================================================================================
    // ========================================================================================
    
    vector_RCP setInitial();
    
    // ========================================================================================
    // ========================================================================================
    
    void setBatchID(const LO & bID);
    
    // ========================================================================================
    // ========================================================================================
    
    vector_RCP blankState();
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void finalizeParams() ;
    
    void finalizeMultiscale() ;
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<meshInterface>  mesh;
    Teuchos::RCP<discretization> disc;
    Teuchos::RCP<physics> phys;
    Teuchos::RCP<linearAlgebra<Node> > linalg;
    Teuchos::RCP<AssemblyManager<Node> > assembler;
    Teuchos::RCP<ParameterManager<Node> > params;
    Teuchos::RCP<PostprocessManager<Node> > postproc;
    Teuchos::RCP<MultiScale> multiscale_manager;
    
    int verbosity, batchID, spaceDim, numsteps, numstages, gNLiter, milo_debug_level, maxNLiter, time_order;
    
    bool save_solution=false;
    
    int BDForder, startupBDForder, startupSteps, numEvaluations;
    string ButcherTab, startupButcherTab;
    
    ScalarT NLtol, final_time, lintol, current_time, initial_time, deltat;
    
    string solver_type, initial_type, response_type;
    
    bool line_search, useL2proj, discretized_stochastic;
    bool isInitial, isTransient, useadjoint, is_final_time, usestrongDBCs;
    bool compute_objective, use_custom_initial_param_guess, store_adjPrev, use_meas_as_dbcs;
    bool scalarDirichletData, transientDirichletData, scalarInitialData;
    bool have_initial_conditions;
    ScalarT discrete_objective_scale_factor;
    
    vector<vector<ScalarT> > scalarDirichletValues, scalarInitialValues; //[block][var]
    Teuchos::RCP<SolutionStorage<Node> > soln, adj_soln, datagen_soln;
    Teuchos::RCP<LA_MultiVector> fixedDOF_soln;
    vector<string> blocknames;
    vector<vector<string> > varlist;
    
    vector<vector<LO> > numBasis, useBasis;
    vector<LO> maxBasis, numVars;
        
    Teuchos::RCP<Teuchos::Time> transientsolvertimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::transientSolver()");
    Teuchos::RCP<Teuchos::Time> nonlinearsolvertimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::nonlinearSolver()");
    
    Teuchos::RCP<Teuchos::Time> initsettimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::setInitial()");
    Teuchos::RCP<Teuchos::Time> dbcsettimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::setDirichlet()");
    Teuchos::RCP<Teuchos::Time> dbcprojtimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::projectDirichlet()");
    Teuchos::RCP<Teuchos::Time> fixeddofsetuptimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::setupFixedDOFs()");
    Teuchos::RCP<Teuchos::Time> msprojtimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::projectDirichlet()");
    Teuchos::RCP<Teuchos::Time> normLAtimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::nonlinearSolver() - norm LA");
    Teuchos::RCP<Teuchos::Time> updateLAtimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::nonlinearSolver() - update LA");
    
  };
  
}

#endif
