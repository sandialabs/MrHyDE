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

/**
 * @class PostprocessManager
 * @brief Handles postprocessing tasks such as computing errors, fluxes, objectives,
 *        and integrated quantities.
 *
 * @tparam Node  Kokkos node type
 */
template<class Node>
class PostprocessManager {
  
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector; /**< MultiVector alias */
  typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;     /**< RCP to MultiVector */
  
public:
  
  /**
   * @brief Minimal constructor
   * @param Comm_     MPI communicator
   * @param settings_ User-provided settings
   * @param mesh_     Mesh interface
   * @param disc_     Discretization interface
   * @param phys_     Physics interface
   * @param assembler_ Assembly manager
   */
  PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                     Teuchos::RCP<MeshInterface> & mesh_,
                     Teuchos::RCP<DiscretizationInterface> & disc_,
                     Teuchos::RCP<PhysicsInterface> & phys_,
                     Teuchos::RCP<AssemblyManager<Node> > & assembler_);
  
  /**
   * @brief Full constructor including multiscale and parameter managers
   */
  PostprocessManager(const Teuchos::RCP<MpiComm> & Comm_,
                     Teuchos::RCP<Teuchos::ParameterList> & settings_,
                     Teuchos::RCP<MeshInterface> & mesh_,
                     Teuchos::RCP<DiscretizationInterface> & disc_,
                     Teuchos::RCP<PhysicsInterface> & phys_,
                     Teuchos::RCP<MultiscaleManager> & multiscale_manager_,
                     Teuchos::RCP<AssemblyManager<Node> > & assembler_,
                     Teuchos::RCP<ParameterManager<Node> > & params_);
  
  /**
   * @brief General setup routine
   */
  void setup();
  
  /**
   * @brief Registers true solution expressions from input file
   * @param true_solns Parameter list of true solutions
   * @param types      Variable types per block
   * @param block      Block index
   * @return List of (variable, expression) pairs
   */
  vector<std::pair<string,string> > addTrueSolutions(Teuchos::ParameterList & true_solns,
                                                     vector<vector<vector<string> > > & types,
                                                     const int & block);
  
  /**
   * @brief Records solution fields for postprocessing
   * @param current_soln Current solution vectors
   * @param current_time Current time value
   * @param stepnum      Time step number
   */
  void record(vector<vector_RCP> & current_soln, const ScalarT & current_time,
              const int & stepnum);
  
  /**
   * @brief Reports all collected postprocessing data
   */
  void report();
  
  /**
   * @brief Computes error quantities between numerical and true solutions
   */
  void computeError(vector<vector_RCP> & current_soln, const ScalarT & current_time);
  
  /**
   * @brief Computes flux-based responses
   */
  void computeFluxResponse(vector<vector_RCP> & current_soln, const ScalarT & current_time);
  
  /**
   * @brief Computes integrated quantities requested in the input file
   * @param current_soln Current solution vectors
   * @param current_time Current simulation time
   */
  void computeIntegratedQuantities(vector<vector_RCP> & current_soln, const ScalarT & current_time);
  
  /**
   * @brief Computes objective function values
   */
  void computeObjective(vector<vector_RCP> & current_soln, const ScalarT & current_time);
  
  
  /**
   * @brief Resets objective accumulators
   */
  void resetObjectives();
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Reports the current value of the objective function.
   *
   * @param[out] objectiveval  The computed value of the objective function.
   */
  void reportObjective(ScalarT & objectiveval);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the gradient of the objective with respect to parameters.
   *
   * @param[in]  current_soln  Current solution vectors (per block and variable).
   * @param[in]  current_time  Current simulation time.
   * @param[in]  deltat        Time step size.
   * @param[out] objectiveval  Objective value equipped with derivative information (DFAD).
   */
  void computeObjectiveGradParam(vector<vector_RCP> & current_soln,
                                 const ScalarT & current_time,
                                 const ScalarT & deltat,
                                 DFAD & objectiveval);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the parameter gradient contribution for a specific objective.
   *
   * @tparam EvalT             Evaluation type (Residual, Jacobian, etc.).
   *
   * @param[in]  obj           Objective index.
   * @param[in]  current_soln  Current solution vectors.
   * @param[in]  current_time  Current simulation time.
   * @param[in]  deltat        Time step size.
   * @param[in]  wset          Workset for local assembly.
   * @param[in]  fman          Function manager for evaluating objective expressions.
   *
   * @return The objective contribution with derivative information as DFAD.
   */
  template<class EvalT>
  DFAD computeObjectiveGradParam(const size_t & obj,
                                 vector<vector_RCP> & current_soln,
                                 const ScalarT & current_time,
                                 const ScalarT & deltat,
                                 Teuchos::RCP<Workset<EvalT> > & wset,
                                 Teuchos::RCP<FunctionManager<EvalT> > & fman);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes gradient of the objective with respect to the solution vector (state).
   *
   * @param[in]  set           Block or physics set index.
   * @param[in]  current_soln  Current solution vector for the set.
   * @param[in]  current_time  Current simulation time.
   * @param[in]  deltat        Time step size.
   * @param[out] grad          Output gradient vector.
   */
  void computeObjectiveGradState(const size_t & set,
                                 const vector_RCP & current_soln,
                                 const ScalarT & current_time,
                                 const ScalarT & deltat,
                                 vector_RCP & grad);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the state gradient contribution for a specific objective.
   *
   * @tparam EvalT             Evaluation type.
   *
   * @param[in]  set           Block/set index.
   * @param[in]  obj           Objective index.
   * @param[in]  current_soln  Current solution vector.
   * @param[in]  current_time  Current simulation time.
   * @param[in]  deltat        Time step size.
   * @param[out] grad          Output gradient vector.
   * @param[in]  wset          Workset for element computations.
   * @param[in]  fman          Function manager for objective evaluation.
   */
  template<class EvalT>
  void computeObjectiveGradState(const size_t & set,
                                 const size_t & obj,
                                 const vector_RCP & current_soln,
                                 const ScalarT & current_time,
                                 const ScalarT & deltat,
                                 vector_RCP & grad,
                                 Teuchos::RCP<Workset<EvalT> > & wset,
                                 Teuchos::RCP<FunctionManager<EvalT> > & fman);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes weighted norms of the solution, as specified in the input file.
   *
   * @param[in] current_soln  Current multi-block solution vectors.
   */
  void computeWeightedNorm(vector<vector_RCP> & current_soln);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes sensor-based quantities from the solution.
   *
   * @param[in] current_soln  Current solution vectors.
   * @param[in] current_time  Current time.
   */
  void computeSensorSolution(vector<vector_RCP> & current_soln,
                             const ScalarT & current_time);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes sensitivities of objectives with respect to parameters using adjoints.
   *
   * @param[in]  u            Forward solution values.
   * @param[in]  sol_stage    Stage solution values (Runge–Kutta, etc.).
   * @param[in]  sol_prev     Previous time-step solutions.
   * @param[in]  adjoint      Adjoint solution vectors.
   * @param[in]  current_time Current time.
   * @param[in]  tindex       Time-step index.
   * @param[in]  deltat       Time-step size.
   * @param[out] gradient     Output gradient container for optimization.
   */
  void computeSensitivities(vector<vector_RCP> & u,
                            vector<vector_RCP> & sol_stage,
                            vector<vector_RCP> & sol_prev,
                            vector<vector_RCP> & adjoint,
                            const ScalarT & current_time,
                            const int & tindex,
                            const ScalarT & deltat,
                            MrHyDE_OptVector & gradient);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the dual-weighted residual (DWR) error indicator.
   *
   * @param[in] u            Forward solution.
   * @param[in] adjoint      Adjoint solution.
   * @param[in] current_time Current time.
   * @param[in] tindex       Time-step index.
   * @param[in] deltat       Time-step size.
   *
   * @return Dual-weighted residual value.
   */
  ScalarT computeDualWeightedResidual(vector<vector_RCP> & u,
                                      vector<vector_RCP> & adjoint,
                                      const ScalarT & current_time,
                                      const int & tindex,
                                      const ScalarT & deltat);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes discrete sensitivities (dJ/du) useful for optimization or DWR.
   *
   * @param[in] u            Forward solution.
   * @param[in] adjoint      Adjoint solution.
   * @param[in] current_time Current time.
   * @param[in] tindex       Time-step index.
   * @param[in] deltat       Time-step size.
   *
   * @return A multivector containing discrete sensitivity values.
   */
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> >
  computeDiscreteSensitivities(vector<vector_RCP> & u,
                               vector<vector_RCP> & adjoint,
                               const ScalarT & current_time,
                               const int & tindex,
                               const ScalarT & deltat);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Writes the current solution fields to output (Exodus, HDF5, etc.).
   *
   * @param[in] current_soln Current block-structured solution.
   * @param[in] current_time Current time.
   */
  void writeSolution(vector<vector_RCP> & current_soln,
                     const ScalarT & current_time);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Writes optimization-related output such as intermediate solutions.
   *
   * @param[in] numEvaluations Number of objective evaluations performed.
   */
  void writeOptimizationSolution(const int & numEvaluations);
  
#if defined(MrHyDE_ENABLE_HDSA)
  /**
   * @brief Writes optimization solution to a specific file.
   *
   * @param[in] filename  Output file name.
   */
  void writeOptimizationSolution(const string & filename);
#endif
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Adds random noise to a value (used for synthetic measurements).
   *
   * @param[in] stdev  Standard deviation of noise.
   *
   * @return Value perturbed by Gaussian noise.
   */
  ScalarT makeSomeNoise(ScalarT stdev);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Parses and constructs objective function evaluators for a given block.
   *
   * @param[in] obj_funs  User-specified objective function parameter list.
   * @param[in] block     Block index.
   */
  void addObjectiveFunctions(Teuchos::ParameterList & obj_funs,
                             const size_t & block);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Parses and constructs flux response function evaluators for a block.
   *
   * @param[in] flux_resp  Parameter list of flux responses.
   * @param[in] block      Block index.
   */
  void addFluxResponses(Teuchos::ParameterList & flux_resp,
                        const size_t & block);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Parses integrated quantity definitions and constructs the objects.
   *
   * @param[in]  iqs    Parameter list defining integrated quantities.
   * @param[in]  block  Block on which IQs are defined.
   *
   * @return Vector of integrated quantity structures created for the block.
   */
  vector<integratedQuantity>
  addIntegratedQuantities(Teuchos::ParameterList & iqs,
                          const size_t & block);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Adds integrated quantities required by the physics module for a given block.
   *
   * This version is used when the physics module provides a list of integrand names
   * and types. Each entry describes a quantity that must be integrated over the block.
   *
   * @param[in] integrandsNamesAndTypes
   *        A vector of {name, type} pairs describing each integrated quantity
   *        required by the physics module.
   * @param[in] block
   *        Index of the mesh block on which the integrated quantities are defined.
   *
   * @return Vector of integratedQuantity objects created for the block.
   */
  vector<integratedQuantity>
  addIntegratedQuantities(vector<vector<string>> & integrandsNamesAndTypes,
                          const size_t & block);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Retrieves extra cell-based fields defined by the physics or discretization.
   *
   * @param[in] block
   *        Index of the mesh block.
   * @param[in] wts
   *        Compressed view of quadrature weights or related data.
   *
   * @return A 2D view (cell × field) containing evaluated cell-level quantities.
   */
  View_Sc2 getExtraCellFields(const int & block,
                              CompressedView<View_Sc2> & wts);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes derived quantities (e.g., fluxes, invariants) for a block.
   *
   * @param[in] block
   *        Index of the mesh block.
   * @param[in] wts
   *        Compressed quadrature weight view used for integration.
   *
   * @return A 2D view storing the evaluated derived quantities.
   */
  View_Sc2 getDerivedQuantities(const int & block,
                                CompressedView<View_Sc2> & wts);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Adds all sensor definitions specified in the user input file.
   *
   * This initializes sensor locations, basis forms, and storage for sensor responses.
   */
  void addSensors();
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Imports sensor point locations and values from an Exodus file.
   *
   * @param[in] objID
   *        Objective index associated with these sensors.
   */
  void importSensorsFromExodus(const int & objID);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Imports sensor data from a set of user-provided files.
   *
   * @param[in] objID
   *        Objective index whose sensors are loaded.
   */
  void importSensorsFromFiles(const int & objID);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Populates sensor locations on a structured grid specified by the user.
   *
   * @param[in] objID
   *        Objective identifier for which grid-based sensors are generated.
   */
  void importSensorsOnGrid(const int & objID);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the basis representation for a set of sensors.
   *
   * This precomputes shape function values or other basis data used when mapping
   * solution fields to sensor points.
   *
   * @param[in] objID
   *        Objective index for which basis data is constructed.
   */
  void computeSensorBasis(const int & objID);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Locates sensor points within mesh elements.
   *
   * @param[in]  block        Mesh block index.
   * @param[in]  spts_host    Host view of sensor point coordinates.
   * @param[out] spts_owners  Output: cell/element ownership for each sensor point.
   * @param[out] spts_found   Output: Boolean flags indicating whether points were found.
   */
  void locateSensorPoints(const int & block,
                          Kokkos::View<ScalarT**,HostDevice> spts_host,
                          Kokkos::View<int*[2],HostDevice> spts_owners,
                          Kokkos::View<bool*,HostDevice> spts_found);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Changes the Exodus output file used for storing postprocessing results.
   *
   * @param[in] newfile
   *        Name of the new Exodus output file.
   */
  void setNewExodusFile(string & newfile);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Saves the value of the objective function for later reporting.
   *
   * @param[in] objVal
   *        Objective value to be recorded.
   */
  void saveObjectiveData(const ScalarT & objVal);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Saves the full objective gradient vector used by optimization routines.
   *
   * @param[in] gradient
   *        Gradient stored in an optimization vector wrapper.
   */
  void saveObjectiveGradientData(const MrHyDE_OptVector & gradient);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Collects and returns all current response function values.
   *
   * @return Array of response values (one entry per response).
   */
  Teuchos::Array<ScalarT> collectResponses();
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Sets the internal time-step index for postprocessing operations.
   *
   * @param[in] ctime
   *        New time-step index.
   */
  void setTimeIndex(const int & ctime);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Resets stored solution fields between time steps or evaluations.
   */
  void resetSolutions();
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Writes quadrature-related data (element integrals, weights, etc.) to output.
   */
  void writeQuadratureData();
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Writes boundary quadrature data, typically used in flux or boundary integral outputs.
   */
  void writeBoundaryQuadratureData();
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Completes any remaining initialization after all postprocessing inputs are parsed.
   *
   * This should be called after sensors, derived quantities, and integrated quantities
   * have been added.
   */
  void completeSetup();
  
  void setForwardStates(vector<vector<vector_RCP> > & fwd_states, vector<ScalarT> & times);
  
  Teuchos::RCP<MpiComm> Comm; ///< MPI communicator used throughout the postprocessing module.
  Teuchos::RCP<MeshInterface> mesh; ///< Mesh interface providing access to mesh blocks, sets, and geometry.
  Teuchos::RCP<DiscretizationInterface> disc; ///< Discretization interface used for finite element or finite volume data.
  Teuchos::RCP<PhysicsInterface> physics; ///< Physics interface providing PDE operators and physical models.
  Teuchos::RCP<AssemblyManager<Node>> assembler; ///< Assembler for residuals, Jacobians, and operators.
  Teuchos::RCP<ParameterManager<Node>> params; ///< Parameter manager for optimization/UQ parameters.
  Teuchos::RCP<MultiscaleManager> multiscale_manager; ///< Manager for multiscale coupled models.
  Teuchos::RCP<LinearAlgebraInterface<Node>> linalg; ///< Linear algebra interface for vectors, matrices, solvers.
  Teuchos::RCP<MrHyDE_Debugger> debugger; ///< Debugger utility for logging and inspection.
  Teuchos::RCP<Teuchos::ParameterList> settings; ///< User-defined and internal settings.
  
  vector<objective> objectives; ///< List of defined objective terms.
  vector<regularization> regularizations; ///< Regularization contributions to objectives.
  vector<Teuchos::RCP<SolutionStorage<Node>>> soln, adj_soln, incr_soln, incr_adj_soln, datagen_soln; ///< Solution storage objects.
  
  bool save_solution = false; ///< Whether to write primal solutions.
  bool save_adjoint_solution = false; ///< Whether to write adjoint solutions.
  
  vector<fluxResponse> fluxes; ///< Flux responses computed on boundaries.
  vector<vector<integratedQuantity>> integratedQuantities; ///< Integrated quantities for each block.
  
  ScalarT record_start; ///< Start time for data recording.
  ScalarT record_stop; ///< Stop time for data recording.
  ScalarT exodus_record_start; ///< Exodus output start time.
  ScalarT exodus_record_stop; ///< Exodus output stop time.
  
  bool compute_objective; ///< Whether to compute objective values.
  bool compute_flux_response; ///< Whether to compute flux responses.
  bool compute_integrated_quantities; ///< Whether to compute integrated quantities.
  bool write_solution_to_file;
  string solution_storage_file;
  ScalarT discrete_objective_scale_factor; ///< Scaling factor for the discrete objective.
  
  vector<vector<string>> extrafields_list, extracellfields_list, derivedquantities_list; ///< Extra field names requested by the user.
  
  vector<ScalarT> weighted_norms; ///< Weighted norms used for error or misfit.
  vector<vector_RCP> norm_wts; ///< Weight vectors for norm computations.
  bool have_norm_weights = false; ///< Whether norm weights have been initialized.
  
  bool compute_response, write_response, compute_error, compute_subgrid_error, compute_weighted_norm; ///< Flags for various postprocessing tasks.
  bool compute_objective_grad_param, write_objective_to_file;
  bool write_solution, write_subgrid_solution, write_HFACE_variables, write_optimization_solution, write_subgrid_model; ///< Output control flags.
  bool write_qdata, write_bqdata; ///< Flags for writing quadrature data.
  string objective_storage_file;
  
  int write_frequency, exodus_write_frequency, write_group_number, write_database_id; ///< Solution write frequency and grouping.
  
  std::string exodus_filename, cellfield_reduction; ///< Exodus output filename and reduction strategy.
  
  int dimension; ///< Spatial dimension.
  int numNodesPerElem; ///< Number of nodes per element.
  int numCells; ///< Number of domain cells.
  size_t numBlocks; ///< Number of mesh blocks.
  int time_index; ///< Current time index.
  
  bool have_sensor_data, save_sensor_data, write_dakota_output, isTD, store_sensor_solution; ///< Sensor and time-dependent flags.
  
  std::string sname, fileoutput, objective_file, objective_grad_file; ///< Output filenames.
  ScalarT stddev; ///< Standard deviation for UQ or noise models.
  size_type global_num_sensors; ///< Total number of sensors.
  
  std::vector<std::string> blocknames, setnames, sideSets, error_types, subgrid_error_types; ///< Various name lists.
  
  std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice>>> errors; ///< Error metrics per time/block.
  std::vector<Kokkos::View<ScalarT**,HostDevice>> responses; ///< Sensor responses.
  std::vector<std::vector<std::vector<Kokkos::View<ScalarT*,HostDevice>>>> subgrid_errors; ///< Subgrid model errors.
  
  int numsteps; ///< Number of steps in the simulation.
  std::vector<std::vector<std::vector<std::string>>> varlist; ///< Variable lists per set/block.
  
  std::string response_type, error_type, append; ///< Output naming flags.
  std::vector<ScalarT> plot_times, response_times, error_times; ///< Time vectors.
  
  int verbosity; ///< Verbosity level.
  
  std::vector<std::vector<std::pair<std::string,std::string>>> error_list; ///< Error variable lists.
  std::vector<std::vector<std::vector<std::pair<std::string,std::string>>>> subgrid_error_lists; ///< Subgrid error variable lists.
  
#if defined(MrHyDE_ENABLE_FFTW)
  Teuchos::RCP<fftInterface> fft; ///< FFT interface.
#endif
  
#if defined(MrHyDE_ENABLE_HDSA)
  bool hdsa_solop; ///< Whether HDSA is active.
  vector<Teuchos::RCP<SolutionStorage<Node>>> hdsa_solop_data; ///< HDSA output data.
#endif
  
  bool is_hdsa_analysis; ///< Whether HDSA analysis is performed.
  
private:
  
  Teuchos::RCP<Teuchos::Time> computeErrorTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeError"); ///< Timer for error computation.
  Teuchos::RCP<Teuchos::Time> writeSolutionTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::writeSolution"); ///< Timer for solution writing.
  Teuchos::RCP<Teuchos::Time> writeSolutionSolIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::writeSolution - solution to ip"); ///< Timer for IP solution writing.
  Teuchos::RCP<Teuchos::Time> objectiveTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeObjective()"); ///< Timer for objective computation.
  Teuchos::RCP<Teuchos::Time> sensorSolutionTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeSensorSolution()"); ///< Timer for sensor solution computation.
  Teuchos::RCP<Teuchos::Time> reportTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::report()"); ///< Timer for reporting.
  Teuchos::RCP<Teuchos::Time> computeWeightedNormTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeWeightedNorm()"); ///< Timer for weighted norm computation.
  Teuchos::RCP<Teuchos::Time> computeGradientTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeSensitivities()"); ///< Timer for
  Teuchos::RCP<Teuchos::Time> computeDiscreteGradientTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeDiscreteSensitivities()"); ///< Timer for
  Teuchos::RCP<Teuchos::Time> objectiveGradParamTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeObjectiveGradParam()"); ///< Timer for objective computation.
  Teuchos::RCP<Teuchos::Time> objectiveGradStateTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Postprocess::computeObjectiveGradState()"); ///< Timer for objective computation.
  
};
}


#endif
