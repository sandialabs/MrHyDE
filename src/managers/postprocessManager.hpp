/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_POSTPROCESS_MANAGER_H
#define MRHYDE_POSTPROCESS_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
//#include "functionManager.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "multiscaleManager.hpp"
#include "linearAlgebraInterface.hpp"
#include "MrHyDE_OptVector.hpp"
#include "postprocessTools.hpp"
#include "MrHyDE_Debugger.hpp"

#if defined(MrHyDE_ENABLE_FFTW)
#include "fftInterface.hpp"
#endif

namespace MrHyDE {
  
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
                       Teuchos::RCP<AssemblyManager<Node> > & assembler_);
    
    // ========================================================================================
    /* Full constructor to set up the problem */
    // ========================================================================================
    
    PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                       Teuchos::RCP<Teuchos::ParameterList> & settings,
                       Teuchos::RCP<MeshInterface> & mesh_,
                       Teuchos::RCP<DiscretizationInterface> & disc_,
                       Teuchos::RCP<PhysicsInterface> & phys_,
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
                const int & stepnum, DFAD & objectiveval);
    
    // ========================================================================================
    // ========================================================================================
    
    void report();
    
    // ========================================================================================
    // ========================================================================================
    
    void computeError(vector<vector_RCP> & current_soln, const ScalarT & current_time);
    
    // ========================================================================================
    // ========================================================================================
    
    void computeFluxResponse(vector<vector_RCP> & current_soln, const ScalarT & current_time);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief Updates the value of integrated quantities requested in the input file
     *
     * @param[in]  current_time  Current simulation time
     */
    
    void computeIntegratedQuantities(vector<vector_RCP> & current_soln, const ScalarT & current_time);

    // ========================================================================================
    // ========================================================================================
    
    void computeObjective(vector<vector_RCP> & current_soln, const ScalarT & current_time,
                          DFAD & objectiveval);

    void resetObjectives();
    
    // ========================================================================================
    // ========================================================================================

    void computeObjectiveGradParam(vector<vector_RCP> & current_soln, const ScalarT & current_time,
                                   DFAD & objectiveval);

    // ========================================================================================
    // ========================================================================================

    template<class EvalT>
    DFAD computeObjectiveGradParam(const size_t & obj, vector<vector_RCP> & current_soln,
                                                         const ScalarT & current_time,
                                                         Teuchos::RCP<Workset<EvalT> > & wset,
                                                         Teuchos::RCP<FunctionManager<EvalT> > & fman);

    // ========================================================================================
    // ========================================================================================

    void computeObjectiveGradState(const size_t & set, vector_RCP & current_soln, const ScalarT & current_time,
                                   const ScalarT & deltat, vector_RCP & grad);

    // ========================================================================================
    // ========================================================================================

    template<class EvalT>
    void computeObjectiveGradState(const size_t & set, const size_t & obj, vector_RCP & current_soln,
                                   const ScalarT & current_time, const ScalarT & deltat, vector_RCP & grad,
                                   Teuchos::RCP<Workset<EvalT> > & wset,
                                   Teuchos::RCP<FunctionManager<EvalT> > & fman);

    // ========================================================================================
    // ========================================================================================

    void computeWeightedNorm(vector<vector_RCP> & current_soln);
    
    // ========================================================================================
    // ========================================================================================

    void computeSensorSolution(vector<vector_RCP> & current_soln, const ScalarT & current_time);

    // ========================================================================================
    // ========================================================================================

    void computeSensitivities(vector<vector_RCP> & u, vector<vector_RCP> & sol_stage, vector<vector_RCP> & sol_prev, vector<vector_RCP> & adjoint,
                              const ScalarT & current_time, const int & tindex, const ScalarT & deltat,
                              MrHyDE_OptVector & gradient);

    // ========================================================================================
    // ========================================================================================

    ScalarT computeDualWeightedResidual(vector<vector_RCP> & u, vector<vector_RCP> & adjoint,
                                        const ScalarT & current_time, const int & tindex, const ScalarT & deltat);

    // ========================================================================================
    // ========================================================================================

    Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > 
    computeDiscreteSensitivities(vector<vector_RCP> & u,
                                 vector<vector_RCP> & adjoint,
                                 const ScalarT & current_time,
                                 const ScalarT & deltat);

    // ========================================================================================
    // ========================================================================================
    
    void writeSolution(vector<vector_RCP> & current_soln, const ScalarT & current_time);
    
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

    View_Sc2 getExtraCellFields(const int & block, CompressedView<View_Sc2> & wts);
    
    // ========================================================================================
    // ========================================================================================
    
    View_Sc2 getDerivedQuantities(const int & block, CompressedView<View_Sc2> & wts);
    
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

    void setNewExodusFile(string & newfile);

    // ========================================================================================
    // ========================================================================================

    void saveObjectiveData(const DFAD& objVal);

    // ========================================================================================
    // ========================================================================================

    void saveObjectiveGradientData(const MrHyDE_OptVector& gradient);
    
    // ========================================================================================
    // ========================================================================================

    Teuchos::Array<ScalarT> collectResponses();

    // ========================================================================================
    // ========================================================================================
        
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<MeshInterface>  mesh;
    Teuchos::RCP<DiscretizationInterface> disc;
    Teuchos::RCP<PhysicsInterface> physics;
    Teuchos::RCP<AssemblyManager<Node> > assembler;
    Teuchos::RCP<ParameterManager<Node> > params;
    Teuchos::RCP<MultiscaleManager> multiscale_manager;
    Teuchos::RCP<LinearAlgebraInterface<Node> > linalg;
    Teuchos::RCP<MrHyDE_Debugger> debugger;
    
    vector<objective> objectives;
    vector<regularization> regularizations;
    vector<Teuchos::RCP<SolutionStorage<Node> > > soln, adj_soln, datagen_soln;
    bool save_solution=false, save_adjoint_solution=false;
    vector<fluxResponse> fluxes;
    vector< vector<integratedQuantity> > integratedQuantities; /// A vector of integrated quantities for each block
    
    ScalarT record_start, record_stop, exodus_record_start, exodus_record_stop;
    bool compute_objective, compute_flux_response, compute_integrated_quantities;
    ScalarT discrete_objective_scale_factor;
    vector<vector<string> > extrafields_list, extracellfields_list, derivedquantities_list;
    vector<ScalarT> weighted_norms;
    vector<vector_RCP> norm_wts;
    bool have_norm_weights = false;
    
    bool compute_response, write_response, compute_error, compute_subgrid_error, compute_weighted_norm;
    bool write_solution, write_subgrid_solution, write_HFACE_variables, write_optimization_solution, write_subgrid_model;
    int write_frequency, exodus_write_frequency, write_group_number, write_database_id;  ///< Solution write frequency (1/timesteps) 
    std::string exodus_filename, cellfield_reduction;
    int dimension;                                                // spatial dimension
    int numNodesPerElem;                                         // nodes on each element
    int numCells;                                                // number of domain cells (normall it is 1)
    size_t numBlocks;                                            // number of element blocks
    
    bool have_sensor_data, save_sensor_data, write_dakota_output, isTD, store_sensor_solution;
    std::string sname, fileoutput, objective_file, objective_grad_file;
    ScalarT stddev;
    size_type global_num_sensors;
    
    std::vector<std::string> blocknames, setnames, sideSets, error_types, subgrid_error_types;
    std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice> > > errors; // [time][block](error_type)
    std::vector<Kokkos::View<ScalarT**,HostDevice> > responses; // [time](sensors,response)
    std::vector<std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice> > > > subgrid_errors; // extra std::vector for multiple subgrid models [time][block][sgmodel](error_list)
    
    int numsteps;
    std::vector<std::vector<std::vector<std::string> > > varlist; // [set][block][var]
    
    std::string response_type, error_type, append;
    std::vector<ScalarT> plot_times, response_times, error_times; // probably always the same
    
    int verbosity;
    
    std::vector<std::vector<std::pair<std::string,std::string> > > error_list; // [block][errors] <varname,type>
    std::vector<std::vector<std::vector<std::pair<std::string,std::string> > > > subgrid_error_lists; // [block][sgmodel][errors]
    
#if defined(MrHyDE_ENABLE_FFTW)
    Teuchos::RCP<fftInterface> fft;
#endif

  private:

    // Timers
    Teuchos::RCP<Teuchos::Time> computeErrorTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeError");
    Teuchos::RCP<Teuchos::Time> writeSolutionTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::writeSolution");
    Teuchos::RCP<Teuchos::Time> writeSolutionSolIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::writeSolution - solution to ip");
    Teuchos::RCP<Teuchos::Time> objectiveTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeObjective()");
    Teuchos::RCP<Teuchos::Time> sensorSolutionTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeSensorSolution()");
    Teuchos::RCP<Teuchos::Time> reportTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::report()");
    Teuchos::RCP<Teuchos::Time> computeWeightedNormTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeWeightedNorm()");
  };
  
}

#endif
