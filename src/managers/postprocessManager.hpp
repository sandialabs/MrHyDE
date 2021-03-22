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

#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "functionManager.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "sensorManager.hpp"

namespace MrHyDE {
  
  // ========================================================================================
  // Class for storing an objective function settings
  // ========================================================================================
  
  class objective {
  public:
    
    objective(Teuchos::ParameterList & objsettings) {
      type = objsettings.get<string>("type","none");
      weight = objsettings.get<ScalarT>("weight",1.0);
      
      sensor_points_file = "";
      sensor_data_file = "";
      save_data = false;
      response = "";
      target = 0.0;
      function = "";
      
      if (type == "sensors") {
        sensor_points_file = objsettings.get<string>("sensor points file","sensor_points.dat");
        sensor_data_file = objsettings.get<string>("sensor data file","sensor_data.dat");
        save_data = objsettings.get<bool>("save sensor data",false);
        response = objsettings.get<string>("response","0.0");
      }
      else if (type == "integrated response") {
        response = objsettings.get<string>("response","0.0");
        target = objsettings.get<ScalarT>("target",0.0);
        save_data = objsettings.get<bool>("save response data",false);
      }
      else if (type == "integrated control") {
        function = objsettings.get<string>("function","0.0");
      }
      else if (type == "discrete control") {
        // nothing else is needed
      }
    }
    
    string type, location, response, function, boundary_name, response_file, sensor_points_file, sensor_data_file;
    ScalarT weight, target;
    bool save_data;
  };
  
  // ========================================================================================
  // Class for storing a regularization function settings
  // ========================================================================================
  
  class regularization {
  public:
    
    regularization(Teuchos::ParameterList & regsettings) {
      type = regsettings.get<string>("type","integrated");
      function = regsettings.get<string>("function","0.0");
      location = regsettings.get<string>("location","volume");
      boundary_name = regsettings.get<string>("boundary name","");
      weight = regsettings.get<ScalarT>("weight",1.0);
    }
    
    string type, location, function, boundary_name;
    ScalarT weight;
    
  };
  
  template<class Node>
  class PostprocessManager {
  public:
    
    // ========================================================================================
    /* Minimal constructor to set up the problem */
    // ========================================================================================
    
    PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                       Teuchos::RCP<meshInterface> & mesh_,
                       Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                       std::vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                       Teuchos::RCP<AssemblyManager<Node> > & assembler_);
    
    // ========================================================================================
    /* Full constructor to set up the problem */
    // ========================================================================================
    
    PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                       Teuchos::RCP<meshInterface> & mesh_,
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

    vector<std::pair<size_t,string> > addTrueSolutions(Teuchos::ParameterList & true_solns,
                                                       vector<string> & vars,
                                                       vector<string> & types,
                                                       const int & block);

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
    Teuchos::RCP<meshInterface>  mesh;
    //Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
    Teuchos::RCP<discretization> disc;
    Teuchos::RCP<physics> phys;
    //Teuchos::RCP<panzer_stk::STK_Interface> optimization_mesh; // Needs to be set manually (for now)
    Teuchos::RCP<AssemblyManager<Node> > assembler;
    Teuchos::RCP<ParameterManager<Node> > params;
    Teuchos::RCP<SensorManager<Node> > sensors;
    std::vector<Teuchos::RCP<FunctionManager> > functionManagers;
    Teuchos::RCP<MultiScale> multiscale_manager;
    
    vector<vector<objective> > objectives;
    vector<vector<regularization> > regularizations;
    
    bool compute_response, compute_error, compute_subgrid_error, compute_aux_error;
    bool write_solution, write_aux_solution, write_subgrid_solution, write_HFACE_variables, write_optimization_solution;
    std::string exodus_filename;
    int spaceDim;                                                // spatial dimension
    //int numNodes;                                              // total number of nodes in the mesh
    int numNodesPerElem;                                         // nodes on each element
    int numCells;                                                // number of domain cells (normall it is 1)
    size_t numBlocks;                                            // number of element blocks
    
    bool have_sensor_data, save_sensor_data, write_dakota_output, isTD;
    std::string sname;
    ScalarT stddev;
    
    std::vector<std::string> blocknames, error_types, subgrid_error_types;
    std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice> > > errors; // [time][block](error_list)
    std::vector<Kokkos::View<ScalarT**,HostDevice> > responses; // [time](sensors,response)
    std::vector<std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice> > > > subgrid_errors; // extra std::vector for multiple subgrid models [time][block][sgmodel](error_list)
    
    int numsteps;
    std::vector<std::vector<std::string> > varlist, aux_varlist; // TMW: remove these at some point
    
    std::string response_type, error_type;
    std::vector<ScalarT> plot_times, response_times, error_times; // probably always the same
    
    int verbosity, debug_level;
    
    std::vector<std::vector<std::pair<size_t,std::string> > > error_list, aux_error_list; // [block][errors]
    std::vector<std::vector<std::vector<std::pair<size_t,std::string> > > > subgrid_error_lists; // [block][sgmodel][errors]
    
    // Timers
    Teuchos::RCP<Teuchos::Time> computeErrorTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::postprocess::computeError");
    Teuchos::RCP<Teuchos::Time> writeSolutionTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::postprocess::writeSolution");
    Teuchos::RCP<Teuchos::Time> writeSolutionSolIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::postprocess::writeSolution - solution to ip");
  };
  
  // Explicit template instantiations
  //template class PostprocessManager<SolverNode>;
  //#if defined(MrHyDE_ASSEMBLYSPACE_CUDA) && !defined(MrHyDE_SOLVERSPACE_CUDA)
  //  template class PostprocessManager<SubgridSolverNode>;
  //#endif

}

#endif
