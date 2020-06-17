/***********************************************************************
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
//#include "solverInterface.hpp"

using namespace std;

void static postprocessHelp(const string & details) {
  cout << "********** Help and Documentation for the Postprocess Interface **********" << endl;
}

class PostprocessManager {
public:
  
  // ========================================================================================
  /* Minimal constructor to set up the problem */
  // ========================================================================================
  
  PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                     Teuchos::RCP<Teuchos::ParameterList> & settings,
                     Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                     Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                     vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                     Teuchos::RCP<AssemblyManager> & assembler_);
  
  // ========================================================================================
  /* Full constructor to set up the problem */
  // ========================================================================================
  
  PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                     Teuchos::RCP<Teuchos::ParameterList> & settings,
                     Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                     Teuchos::RCP<discretization> & disc_, Teuchos::RCP<physics> & phys_,
                     vector<Teuchos::RCP<FunctionManager> > & functionManagers,
                     Teuchos::RCP<MultiScale> & multiscale_manager_,
                     Teuchos::RCP<AssemblyManager> & assembler_,
                     Teuchos::RCP<ParameterManager> & params_,
                     Teuchos::RCP<SensorManager> & sensors_);
  
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
  
  AD computeObjective();
  
  // ========================================================================================
  // ========================================================================================
  
  void computeResponse(const ScalarT & currenttime);
  
  // ========================================================================================
  // ========================================================================================
  
  vector<ScalarT> computeSensitivities();
  
  // ========================================================================================
  // ========================================================================================
  
  void writeSolution(const ScalarT & currenttime);
  
  // ========================================================================================
  // ========================================================================================
  
  ScalarT makeSomeNoise(ScalarT stdev);
  
  // ========================================================================================
  // The following function is the adjoint-based error estimate
  // Not to be confused with the postprocess::computeError function which uses a true
  //   solution to perform verification studies
  // ========================================================================================
  
  //ScalarT computeError();
  
  // ========================================================================================
  // ========================================================================================
  
  vector<ScalarT> computeParameterSensitivities();
  
  // ========================================================================================
  // Compute the sensitivity of the objective with respect to discretized parameters
  // ========================================================================================
  
  vector<ScalarT> computeDiscretizedSensitivities();
  
//protected:
  
  Teuchos::RCP<MpiComm> Comm;
  Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
  Teuchos::RCP<discretization> disc;
  Teuchos::RCP<physics> phys;
  //Teuchos::RCP<const panzer::DOFManager> DOF;
  //Teuchos::RCP<solver> solve;
  Teuchos::RCP<AssemblyManager> assembler;
  Teuchos::RCP<ParameterManager> params;
  Teuchos::RCP<SensorManager> sensors;
  Teuchos::RCP<MultiScale> multiscale_manager;
  vector<Teuchos::RCP<FunctionManager> > functionManagers;
  
  bool compute_response, compute_error, write_solution, write_subgrid_solution;
  string exodus_filename;
  int spaceDim;                                                // spatial dimension
  //int numNodes;                                              // total number of nodes in the mesh
  int numNodesPerElem;                                         // nodes on each element
  int numCells;                                                // number of domain cells (normall it is 1)
  size_t numBlocks;                                            // number of element blocks
  
  vector<vector<int> > numBasis;
  vector<vector<int> > useBasis;
  vector<int> maxbasis;
  bool have_sensor_data, save_sensor_data, write_dakota_output, isTD;
  bool plot_response, save_height_file;
  string sname;
  ScalarT stddev;
  
  vector<string> blocknames, error_types, subgrid_error_types;
  vector<vector<Kokkos::View<ScalarT*,AssemblyDevice> > > errors; // [time][block](error_list)
  vector<Kokkos::View<ScalarT**,AssemblyDevice> > responses; // [time](sensors,response)
  vector<vector<Kokkos::View<ScalarT**,AssemblyDevice> > > subgrid_errors; // extra vector for multiple subgrid models
  
  vector<int> numVars; // Number of variables used by the application (may not be used yet)
  int numsteps;
  vector<vector<string> > varlist;
  
  bool use_sol_mod_mesh, use_sol_mod_height;
  int sol_to_mod_mesh, sol_to_mod_height;
  ScalarT meshmod_TOL, layer_size;
  //bool compute_subgrid_error, have_subgrids;
  
  
  
  //Teuchos::RCP<const LA_Map> overlapped_map;
  //Teuchos::RCP<const LA_Map> param_overlapped_map;
  string response_type, error_type;
  vector<ScalarT> plot_times, response_times, error_times; // probably always the same
  
  //vector<vector<Teuchos::RCP<cell> > > cells;
  int verbosity;
  
  vector<vector<pair<size_t,string> > > error_list;
  
  // Timers
  Teuchos::RCP<Teuchos::Time> computeErrorTimer = Teuchos::TimeMonitor::getNewCounter("MILO::postprocess::computeError");
  Teuchos::RCP<Teuchos::Time> writeSolutionTimer = Teuchos::TimeMonitor::getNewCounter("MILO::postprocess::writeSolution");
  Teuchos::RCP<Teuchos::Time> writeSolutionSolIPTimer = Teuchos::TimeMonitor::getNewCounter("MILO::postprocess::writeSolution - solution to ip");
};

#endif
