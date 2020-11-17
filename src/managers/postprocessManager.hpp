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

#ifndef PP_H
#define PP_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "functionManager.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "sensorManager.hpp"

//using namespace std;

namespace MrHyDE {
  /*
  void static postprocessHelp(const std::string & details) {
    cout << "********** Help and Documentation for the Postprocess Interface **********" << endl;
  }
  */
  
  template<class Node>
  class PostprocessManager {
  public:
    
    // ========================================================================================
    /* Minimal constructor to set up the problem */
    // ========================================================================================
    
    PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                       Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                       Teuchos::RCP<panzer_stk::STK_Interface> & optimization_mesh_,
                       Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                       std::vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                       Teuchos::RCP<AssemblyManager<Node> > & assembler_);
    
    // ========================================================================================
    /* Full constructor to set up the problem */
    // ========================================================================================
    
    PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                       Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                       Teuchos::RCP<panzer_stk::STK_Interface> & optimization_mesh_,
                       Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                       std::vector<Teuchos::RCP<FunctionManager> > & functionManagers,
                       Teuchos::RCP<MultiScale> & multiscale_manager_,
                       Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                       Teuchos::RCP<ParameterManager<Node> > & params_);
    
    // ========================================================================================
    // ========================================================================================
    
    void setup(Teuchos::RCP<Teuchos::ParameterList> & settings);
    
    // ========================================================================================
    // ========================================================================================
    
    void record(const ScalarT & currenttime);
    
    // ========================================================================================
    // ========================================================================================
    
    void report();
    
    // ========================================================================================
    // ========================================================================================
    
    void computeError(const ScalarT & currenttime);
    
    // ========================================================================================
    // ========================================================================================
    
    void computeResponse(const ScalarT & currenttime);
    
    // ========================================================================================
    // ========================================================================================
    
    void writeSolution(const ScalarT & currenttime);
    
    // ========================================================================================
    // ========================================================================================
    
    void writeOptimizationSolution(const int & numEvaluations);
    
    // ========================================================================================
    // ========================================================================================
    
    ScalarT makeSomeNoise(ScalarT stdev);
    
    // ========================================================================================
    // ========================================================================================
    
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
    Teuchos::RCP<discretization> disc;
    Teuchos::RCP<physics> phys;
    
    Teuchos::RCP<panzer_stk::STK_Interface> optimization_mesh; // Needs to be set manually (for now)
    
    //Teuchos::RCP<const panzer::DOFManager> DOF;
    //Teuchos::RCP<solver> solve;
    Teuchos::RCP<AssemblyManager<Node> > assembler;
    Teuchos::RCP<ParameterManager<Node> > params;
    Teuchos::RCP<SensorManager> sensors;
    std::vector<Teuchos::RCP<FunctionManager> > functionManagers;
    Teuchos::RCP<MultiScale> multiscale_manager;
    
    bool compute_response, compute_error, compute_subgrid_error;
    bool write_solution, write_subgrid_solution, write_HFACE_variables, write_optimization_solution;
    std::string exodus_filename;
    int spaceDim;                                                // spatial dimension
    //int numNodes;                                              // total number of nodes in the mesh
    int numNodesPerElem;                                         // nodes on each element
    int numCells;                                                // number of domain cells (normall it is 1)
    size_t numBlocks;                                            // number of element blocks
    
    std::vector<std::vector<int> > numBasis;
    std::vector<std::vector<int> > useBasis;
    std::vector<int> maxbasis;
    bool have_sensor_data, save_sensor_data, write_dakota_output, isTD;
    bool plot_response, save_height_file;
    std::string sname;
    ScalarT stddev;
    
    std::vector<std::string> blocknames, error_types, subgrid_error_types;
    std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice> > > errors; // [time][block](error_list)
    std::vector<Kokkos::View<ScalarT**,HostDevice> > responses; // [time](sensors,response)
    std::vector<std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice> > > > subgrid_errors; // extra std::vector for multiple subgrid models [time][block][sgmodel](error_list)
    
    std::vector<int> numVars; // Number of variables used by the application (may not be used yet)
    int numsteps;
    std::vector<std::vector<std::string> > varlist;
    
    bool use_sol_mod_mesh, use_sol_mod_height;
    int sol_to_mod_mesh, sol_to_mod_height;
    ScalarT meshmod_TOL, layer_size;
    //bool compute_subgrid_error, have_subgrids;
    
    
    
    //Teuchos::RCP<const LA_Map> overlapped_map;
    //Teuchos::RCP<const LA_Map> param_overlapped_map;
    std::string response_type, error_type;
    std::vector<ScalarT> plot_times, response_times, error_times; // probably always the same
    
    //std::vector<std::vector<Teuchos::RCP<cell> > > cells;
    int verbosity, milo_debug_level;
    
    std::vector<std::vector<std::pair<size_t,std::string> > > error_list; // [block][errors]
    std::vector<std::vector<std::vector<std::pair<size_t,std::string> > > > subgrid_error_lists; // [block][sgmodel][errors]
    
    // Timers
    Teuchos::RCP<Teuchos::Time> computeErrorTimer = Teuchos::TimeMonitor::getNewCounter("MILO::postprocess::computeError");
    Teuchos::RCP<Teuchos::Time> writeSolutionTimer = Teuchos::TimeMonitor::getNewCounter("MILO::postprocess::writeSolution");
    Teuchos::RCP<Teuchos::Time> writeSolutionSolIPTimer = Teuchos::TimeMonitor::getNewCounter("MILO::postprocess::writeSolution - solution to ip");
  };
  
  // Explicit template instantiations
  template class PostprocessManager<SolverNode>;
  #if !defined(MrHyDE_SOLVERSPACE_CUDA)
    #if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
      template class PostprocessManager<SubgridSolverNode>;
    #endif
  #endif

}

#endif
