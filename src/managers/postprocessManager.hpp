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

#ifndef MRHYDE_POSTPROCESS_MANAGER_H
#define MRHYDE_POSTPROCESS_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "functionManager.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "multiscaleManager.hpp"
#include "linearAlgebraInterface.hpp"

namespace MrHyDE {
  
  
  // ========================================================================================
  // Class for storing a regularization function settings
  // ========================================================================================
  
  class regularization {
  public:
    
    regularization() {};
    
    ~regularization() {};
    
    regularization(Teuchos::ParameterList & regsettings, const string name_,
                   const size_t & block_, Teuchos::RCP<FunctionManager> & functionManager_) {
      name = name_;
      block = block_;
      
      type = regsettings.get<string>("type","integrated");
      function = regsettings.get<string>("function","0.0");
      location = regsettings.get<string>("location","volume");
      boundary_name = regsettings.get<string>("boundary name","");
      weight = regsettings.get<double>("weight",1.0);
      objective_name = regsettings.get<string>("objective name","");
      
      if (type == "integrated") {
        if (location == "volume") {
          functionManager_->addFunction(name,function,"ip");
        }
        else if (location == "boundary") {
          functionManager_->addFunction(name,function,"side ip");
        }
      }
    }
    
    string type, name, location, function, boundary_name, objective_name;
    ScalarT weight;
    size_t block;
  };
  
  // ========================================================================================
  // Class for storing an objective function settings
  // ========================================================================================
  
  class objective {
  public:
    
    objective() {};
    
    ~objective() {};
    
    objective(Teuchos::ParameterList & objsettings, const string name_,
              const size_t & block_, Teuchos::RCP<FunctionManager> & functionManager_) {
      name = name_;
      block = block_;
      type = objsettings.get<string>("type","none");
      weight = objsettings.get<double>("weight",1.0);
      
      sensor_points_file = "";
      sensor_data_file = "";
      save_data = false;
      response = "";
      target = 0.0;
      function = "";
      use_sensor_grid = false;

      if (type == "sensors") {
        sensor_points_file = objsettings.get<string>("sensor points file","sensor_points.dat");
        sensor_data_file = objsettings.get<string>("sensor data file","");
        save_data = objsettings.get<bool>("save sensor data",false);
        use_sensor_grid = objsettings.get<bool>("use sensor grid",false);
        if (use_sensor_grid) {
          sensor_grid_Nx = objsettings.get<int>("grid Nx");
          sensor_grid_Ny = objsettings.get<int>("grid Ny");
          sensor_grid_Nz = objsettings.get<int>("grid Nz");

          sensor_grid_xmin = objsettings.get<double>("grid xmin");
          sensor_grid_xmax = objsettings.get<double>("grid xmax");
          sensor_grid_ymin = objsettings.get<double>("grid ymin");
          sensor_grid_ymax = objsettings.get<double>("grid ymax");
          sensor_grid_zmin = objsettings.get<double>("grid zmin");
          sensor_grid_zmax = objsettings.get<double>("grid zmax");
        }
        response = objsettings.get<string>("response","0.0");
        functionManager_->addFunction(name+" response",response,"point");
        response_file = objsettings.get<string>("response file","sensor."+name);
        compute_sensor_soln = objsettings.get<bool>("compute sensor solution",false);
        compute_sensor_average_soln = objsettings.get<bool>("compute sensor average solution",false);
        if (compute_sensor_soln && compute_sensor_average_soln) {
          // throw an error
        }      
      }
      else if (type == "integrated response") {
        response = objsettings.get<string>("response","0.0");
        target = objsettings.get<double>("target",0.0);
        functionManager_->addFunction(name+" response",response,"ip");
        save_data = objsettings.get<bool>("save response data",false);
        response_file = objsettings.get<string>("response file","response."+name);
      }
      else if (type == "integrated control") {
        function = objsettings.get<string>("function","0.0");
        functionManager_->addFunction(name,function,"ip");
      }
      else if (type == "discrete control") {
        // nothing else is needed
      }
    
      if (objsettings.isSublist("Regularization functions")) {
        Teuchos::ParameterList reg_funs = objsettings.sublist("Regularization functions");
        Teuchos::ParameterList::ConstIterator reg_itr = reg_funs.begin();
        while (reg_itr != reg_funs.end()) {
          Teuchos::ParameterList regsettings = reg_funs.sublist(reg_itr->first);
          regularization newreg(regsettings,reg_itr->first,block,functionManager_);
          regularizations.push_back(newreg);
          reg_itr++;
        }
      }
    }
    
    size_t block;
    string name, type, location, response, function, boundary_name, response_file;
    ScalarT weight, target;
    bool save_data;
    vector<regularization> regularizations;
    vector<ScalarT> response_times;
    vector<ScalarT> scalar_response_data;
    vector<Kokkos::View<ScalarT*,HostDevice> > response_data;
    
    // Data specific to sensors
    string sensor_points_file, sensor_data_file;
    size_t numSensors;
    bool use_sensor_grid, compute_sensor_soln, compute_sensor_average_soln;
    int sensor_grid_Nx, sensor_grid_Ny, sensor_grid_Nz;
    double sensor_grid_xmin, sensor_grid_xmax, sensor_grid_ymin, sensor_grid_ymax, sensor_grid_zmin, sensor_grid_zmax;
    Kokkos::View<ScalarT**,AssemblyDevice> sensor_data;   // Ns x Nt
    Kokkos::View<ScalarT**,AssemblyDevice> sensor_points; // Ns x dim
    Kokkos::View<ScalarT*,AssemblyDevice>  sensor_times;  // Nt
    Kokkos::View<int*[2],HostDevice>       sensor_owners; // Ns x (cell elem)
    Kokkos::View<bool*,HostDevice>         sensor_found;
    vector<Kokkos::View<ScalarT****,AssemblyDevice> > sensor_basis;       //[basis](Ns,dof,pt,dim)
    vector<Kokkos::View<ScalarT****,AssemblyDevice> > sensor_basis_grad;  // [basis](Ns,dof,pt,dim)
    //vector<vector<Kokkos::View<ScalarT***,AssemblyDevice> > >  sensor_basis_div;   // [Ns][basis](elem,dof,pt)
    //vector<vector<Kokkos::View<ScalarT****,AssemblyDevice> > > sensor_basis_curl;  // [Ns][basis](elem,dof,pt,dim)
    vector<Kokkos::View<ScalarT***,HostDevice> > sensor_solution_data; // [time] (sensor,sol,dim)
  };
  
  // ========================================================================================
  // Class for storing a flux response (not for optimization)
  // ========================================================================================
  
  class fluxResponse {
  public:
    
    fluxResponse() {};
    
    ~fluxResponse() {};
    
    fluxResponse(Teuchos::ParameterList & frsettings, const string & name_,
                 const size_t & block_, Teuchos::RCP<FunctionManager> & functionManager_) {
      name = name_;
      block = block_;
      
      sidesets = frsettings.get<string>("side sets","all");
      weight = frsettings.get<string>("weight","1.0");
      int numfluxes = frsettings.get<int>("number",1);
      
      vals = Kokkos::View<ScalarT*,HostDevice>("flux data",numfluxes);
      
      functionManager_->addFunction("flux weight "+name,weight,"side ip");
      
    }
    
    string name, sidesets, weight;
    size_t block;
    Kokkos::View<ScalarT*,HostDevice> vals;
  };
  
  
  // ========================================================================================
  // Class for storing an integrated quantity
  // ========================================================================================

  /** integratedQuantity class
   * 
   * Holds the information necessary to compute an integrated quantity along with
   * its value. This is not for optimization.
   *
   */
  
  // TODO -- BWR could potentially make this a parent class and have the others inherit.
  class integratedQuantity {
  public:
    /**
     * @brief Construct storage and information for an integrated quantity requested
     * in the input file
     *
     * @param[in]  iqsettings  Parameter list with the settings from the input file
     * @param[in]  name_  Name for the quantity
     * @param[in]  block_  Mesh block on which to compute
     * @param[in]  functionManager_  The function manager used to store the integrand 
     *
     */
    
    integratedQuantity() {};
    
    ~integratedQuantity() {};
    
    integratedQuantity(Teuchos::ParameterList & iqsettings, const string & name_,
                 const size_t & block_, Teuchos::RCP<FunctionManager> & functionManager_) {
      name = name_;
      block = block_;
      
      // We assume a volume integral by default
      location = iqsettings.get<string>("location","volume"); 
      // Only used in case of boundary integral
      boundarynames = iqsettings.get<string>("boundary names","all");
      
      val = Kokkos::View<ScalarT*,HostDevice>("integrated quantity data",1);
     
      integrand = iqsettings.get<string>("integrand","0.0");
      
      // Integrand is kept at the appropriate integration points
      if (location == "volume") {
        functionManager_->addFunction(name+" integrand",integrand,"ip"); 
      } else if (location == "boundary") {
        functionManager_->addFunction(name+" integrand",integrand,"side ip");
      }

    }
    
    /**
     * @brief Construct storage and information for an integrated quantity requested
     * from the physics module
     *
     * @param[in]  integrand  The integrand to be added to the function manager
     * @param[in]  name_  Name for the quantity
     * @param[in]  integralType  The type of integral (boundary or volume)
     * @param[in]  block_  Mesh block on which to compute
     * @param[in]  functionManager_  The function manager used to store the integrand 
     *
     */
    
    integratedQuantity(const string & integrand_, const string & name_, const string & integralType, 
                 const size_t & block_, Teuchos::RCP<FunctionManager> & functionManager_) {
      integrand = integrand_;
      name = name_;
      block = block_;
      location = integralType; // for consistency with above

      // Only used in case of boundary integral
      boundarynames = "all"; // TODO this could be expanded in the future (again for consistency)
      
      val = Kokkos::View<ScalarT*,HostDevice>("integrated quantity data",1);
     
      // Integrand is kept at the appropriate integration points
      if (location == "volume") {
        functionManager_->addFunction(name+" integrand",integrand,"ip"); 
      } else if (location == "boundary") {
        functionManager_->addFunction(name+" integrand",integrand,"side ip");
      } else {
        // TODO add error message
      }

    }

    string name, boundarynames, integrand, location;
    size_t block;
    Kokkos::View<ScalarT*,HostDevice> val;
  };

  // ========================================================================================
  // ========================================================================================
  
  template<class Node>
  class PostprocessManager {
    
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;
    
  public:
    
    // ========================================================================================
    /* Minimal constructor to set up the problem */
    // ========================================================================================
    
    PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                       Teuchos::RCP<MeshInterface> & mesh_,
                       Teuchos::RCP<DiscretizationInterface> & disc_,
                       Teuchos::RCP<PhysicsInterface> & phys_,
                       std::vector<Teuchos::RCP<FunctionManager> > & functionManagers_,
                       Teuchos::RCP<AssemblyManager<Node> > & assembler_);
    
    // ========================================================================================
    /* Full constructor to set up the problem */
    // ========================================================================================
    
    PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                       Teuchos::RCP<MeshInterface> & mesh_,
                       Teuchos::RCP<DiscretizationInterface> & disc_,
                       Teuchos::RCP<PhysicsInterface> & phys_,
                       std::vector<Teuchos::RCP<FunctionManager> > & functionManagers,
                       Teuchos::RCP<MultiscaleManager> & multiscale_manager_,
                       Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                       Teuchos::RCP<ParameterManager<Node> > & params_);
    
    // ========================================================================================
    // ========================================================================================
    
    void setup(Teuchos::RCP<Teuchos::ParameterList> & settings);
    
    // ========================================================================================
    // ========================================================================================

    vector<std::pair<string,string> > addTrueSolutions(Teuchos::ParameterList & true_solns,
                                                       vector<vector<vector<string> > > & types,
                                                       const int & block);

    // ========================================================================================
    // ========================================================================================
    
    void record(vector<vector_RCP> & current_soln, const ScalarT & current_time,
                const bool & write_this_step, DFAD & objectiveval);
    
    // ========================================================================================
    // ========================================================================================
    
    void report();
    
    // ========================================================================================
    // ========================================================================================
    
    void computeError(const ScalarT & current_time);
    
    // ========================================================================================
    // ========================================================================================
    
    void computeResponse(const ScalarT & current_time);
    
    // ========================================================================================
    // ========================================================================================
    
    void computeFluxResponse(const ScalarT & current_time);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief Updates the value of integrated quantities requested in the input file
     *
     * @param[in]  current_time  Current simulation time
     */
    
    void computeIntegratedQuantities(const ScalarT & current_time);

    // ========================================================================================
    // ========================================================================================
    
    void computeObjective(vector<vector_RCP> & current_soln, const ScalarT & current_time,
                          DFAD & objectiveval);

    void computeObjectiveGradState(const size_t & set, vector_RCP & current_soln, const ScalarT & current_time,
                                   const ScalarT & deltat, vector_RCP & grad);

    void computeWeightedNorm(vector<vector_RCP> & current_soln);
    
    void computeSensorSolution(vector<vector_RCP> & current_soln, const ScalarT & current_time);

    // ========================================================================================
    // ========================================================================================

    void computeSensitivities(vector<vector_RCP> & u, vector<vector_RCP> & adjoint,
                              const ScalarT & current_time, const ScalarT & deltat,
                              vector<ScalarT> & gradient);

    // ========================================================================================
    // ========================================================================================
    
    void writeSolution(const ScalarT & current_time);
    
    // ========================================================================================
    // ========================================================================================
    
    void writeOptimizationSolution(const int & numEvaluations);
    
    // ========================================================================================
    // ========================================================================================
    
    ScalarT makeSomeNoise(ScalarT stdev);
    
    // ========================================================================================
    // ========================================================================================
    
    void addObjectiveFunctions(Teuchos::ParameterList & obj_funs, const size_t & block);
    
    // ========================================================================================
    // ========================================================================================
    
    void addFluxResponses(Teuchos::ParameterList & flux_resp, const size_t & block);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief Parses the integrated quantities parameter list and constructs the requested
     * IQs.
     *
     * @param[in]  iqs  Integrated quantities parameter list
     * @param[in]  block  Mesh block
     *
     * @return A vector of integrated quantities defined on the block
     */

    vector<integratedQuantity> addIntegratedQuantities(Teuchos::ParameterList & iqs, const size_t & block);
    
    // ========================================================================================
    // ========================================================================================

    /**
     * @brief Add integrated quantities to the postprocessing manager which are required
     * by the physics module defined on a block.
     *
     * @param[in]  integrandsNamesAndTypes  Integrated quantity information from the physics module
     * @param[in]  block  Mesh block
     *
     * @return A vector of integrated quantities defined on the block
     */

    vector<integratedQuantity> addIntegratedQuantities(vector< vector<string> > & integrandsNamesAndTypes, const size_t & block);
    
    // ========================================================================================
    // ========================================================================================

    View_Sc2 getExtraCellFields(const int & block, View_Sc2 wts);
    
    // ========================================================================================
    // ========================================================================================
    
    View_Sc2 getDerivedQuantities(const int & block, View_Sc2 wts);
    
    // ========================================================================================
    // ========================================================================================
    
    void addSensors();
    
    // ========================================================================================
    // ========================================================================================
    
    void importSensorsFromExodus(const int & objID);
    
    // ========================================================================================
    // ========================================================================================
    
    void importSensorsFromFiles(const int & objID);

    // ========================================================================================
    // ========================================================================================

    void importSensorsOnGrid(const int & objID);
    
    // ========================================================================================
    // ========================================================================================

    void computeSensorBasis(const int & objID);
   
    // ========================================================================================
    // ========================================================================================

    void locateSensorPoints(const int & block, Kokkos::View<ScalarT**,HostDevice> spts_host,
                            Kokkos::View<int*[2],HostDevice> spts_owners, 
                            Kokkos::View<bool*,HostDevice> spts_found);

    // ========================================================================================
    // ========================================================================================
        
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<MeshInterface>  mesh;
    Teuchos::RCP<DiscretizationInterface> disc;
    Teuchos::RCP<PhysicsInterface> phys;
    Teuchos::RCP<AssemblyManager<Node> > assembler;
    Teuchos::RCP<ParameterManager<Node> > params;
    std::vector<Teuchos::RCP<FunctionManager> > functionManagers;
    Teuchos::RCP<MultiscaleManager> multiscale_manager;
    Teuchos::RCP<LinearAlgebraInterface<Node> > linalg;
    
    vector<objective> objectives;
    vector<regularization> regularizations;
    vector<Teuchos::RCP<SolutionStorage<Node> > > soln, adj_soln, datagen_soln;
    bool save_solution=false;
    vector<fluxResponse> fluxes;
    vector< vector<integratedQuantity> > integratedQuantities; /// A vector of integrated quantities for each block
    
    bool compute_objective, compute_flux_response, compute_integrated_quantities;
    ScalarT discrete_objective_scale_factor;
    vector<vector<string> > extrafields_list, extracellfields_list, derivedquantities_list;
    vector<ScalarT> weighted_norms;
    vector<vector_RCP> norm_wts;
    bool have_norm_weights = false;
    
    bool compute_response, compute_error, compute_subgrid_error, compute_weighted_norm;
    bool write_solution, write_subgrid_solution, write_HFACE_variables, write_optimization_solution;
    int write_frequency, write_cell_number, write_database_id;  ///< Solution write frequency (1/timesteps) 
    std::string exodus_filename, cellfield_reduction;
    int spaceDim;                                                // spatial dimension
    int numNodesPerElem;                                         // nodes on each element
    int numCells;                                                // number of domain cells (normall it is 1)
    size_t numBlocks;                                            // number of element blocks
    
    bool have_sensor_data, save_sensor_data, write_dakota_output, isTD, store_sensor_solution;
    std::string sname;
    ScalarT stddev;
    
    std::vector<std::string> blocknames, setnames, sideSets, error_types, subgrid_error_types;
    std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice> > > errors; // [time][block](error_type)
    std::vector<Kokkos::View<ScalarT**,HostDevice> > responses; // [time](sensors,response)
    std::vector<std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice> > > > subgrid_errors; // extra std::vector for multiple subgrid models [time][block][sgmodel](error_list)
    
    int numsteps;
    std::vector<std::vector<std::vector<std::string> > > varlist; // [set][block][var]
    
    std::string response_type, error_type, append;
    std::vector<ScalarT> plot_times, response_times, error_times; // probably always the same
    
    int verbosity, debug_level;
    
    std::vector<std::vector<std::pair<std::string,std::string> > > error_list; // [block][errors] <varname,type>
    std::vector<std::vector<std::vector<std::pair<std::string,std::string> > > > subgrid_error_lists; // [block][sgmodel][errors]
    
    
    // Timers
    Teuchos::RCP<Teuchos::Time> computeErrorTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeError");
    Teuchos::RCP<Teuchos::Time> writeSolutionTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::writeSolution");
    Teuchos::RCP<Teuchos::Time> writeSolutionSolIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::writeSolution - solution to ip");
    Teuchos::RCP<Teuchos::Time> objectiveTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeObjective()");
    Teuchos::RCP<Teuchos::Time> computeWeightedNormTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeWeightedNorm()");
  };
  
}

#endif
