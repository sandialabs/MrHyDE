/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/
#ifndef MRHYDE_PARAMETER_MANAGER_H
#define MRHYDE_PARAMETER_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "group.hpp"
#include "boundaryGroup.hpp"
#include "Panzer_STK_Interface.hpp"
#include "discretizationInterface.hpp"
#include "MrHyDE_OptVector.hpp"
#include "MrHyDE_Debugger.hpp"

namespace MrHyDE {
  
  /**
   * @class ParameterManager
   * @brief Handles creation, storage, communication, and updating of all parameter types
   *        (active, inactive, stochastic, discrete, discretized) used throughout MrHyDE.
   *
   * This class manages parameter lists, discretized parameter fields, adjoint-related data, and
   * provides Sacado-enabled seeding for automatic differentiation. It interacts with the physics,
   * mesh, and discretization interfaces to ensure consistent parameter propagation.
   */
  template<class Node>
  class ParameterManager {
    
    // --- Typedefs for linear algebra interfaces ---
    typedef Tpetra::Export<LO, GO, Node>            LA_Export;   ///< Export operator for parameter communication.
    typedef Tpetra::Import<LO, GO, Node>            LA_Import;   ///< Import operator for parameter communication.
    typedef Tpetra::Map<LO, GO, Node>               LA_Map;      ///< Map describing parameter DOF layout.
    typedef Tpetra::CrsGraph<LO,GO,Node>            LA_CrsGraph; ///< Graph structure for sparse matrices.
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;///< Matrix storing parameter mass or projection operators.
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector; ///< MultiVector storing discretized parameter values.
    typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;  ///< Reference-counted pointer to parameter vectors.
    typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;  ///< Reference-counted pointer to parameter matrices.
    typedef typename Node::device_type              LA_device;   ///< Device type used for Kokkos views.
    
  public:
    
    /** @brief Default constructor (creates an empty manager). */
    ParameterManager() {};
    
    /** @brief Destructor. */
    ~ParameterManager() {};

    /**
     * @brief Full constructor initializing the parameter manager.
     * @param[in] Comm_  MPI communicator.
     * @param[in] settings User input parameter list.
     * @param[in] mesh_ Mesh interface for parameter fields.
     * @param[in] phys_ Physics interface owning active/inactive parameters.
     * @param[in] disc_ Discretization interface defining basis and DOF structure.
     */
    ParameterManager(const Teuchos::RCP<MpiComm> & Comm_,
                     Teuchos::RCP<Teuchos::ParameterList> & settings,
                     Teuchos::RCP<MeshInterface> & mesh_,
                     Teuchos::RCP<PhysicsInterface> & phys_,
                     Teuchos::RCP<DiscretizationInterface> & disc_);
    
    /** @brief Initializes parameter lists, types, and assignments. */
    void setupParameters();

    /**
     * @brief Builds discretized parameter fields over groups and boundary groups.
     */
    void setupDiscretizedParameters(
      std::vector<std::vector<Teuchos::RCP<Group> > > & groups,
      std::vector<std::vector<Teuchos::RCP<BoundaryGroup> > > & boundary_groups);
    
    /** @brief Returns number of parameters of a given type (int identifier). */
    int getNumParams(const int & type);

    /** @brief Returns number of parameters of a given type (string identifier). */
    int getNumParams(const std::string & type);

    /** @brief Returns discretized parameters as a flattened std::vector for ROL. */
    std::vector<ScalarT> getDiscretizedParamsVector();

    /** @brief Returns the primary discretized parameter multivector. */
    vector_RCP getDiscretizedParams();

    /** @brief Returns the "over" version of discretized parameters. */
    vector_RCP getDiscretizedParamsOver();

    /** @brief Returns discretized parameter time derivatives. */
    vector_RCP getDiscretizedParamsDot();

    /** @brief Returns "over" version of time derivatives. */
    vector_RCP getDiscretizedParamsDotOver();

    /** @brief Returns a vector of dynamic parameter fields over time. */
    std::vector<vector_RCP> getDynamicDiscretizedParams();

    /** @brief Converts parameters to Sacado AD types and seeds derivatives. */
    void sacadoizeParams(const bool & seed_active);

    /** @brief Sacado-enabled parameter seeding for scalar AD types. */
    void sacadoizeParamsSc(const bool & seed_active,
                           Kokkos::View<int*,AssemblyDevice> ptypes,
                           Kokkos::View<size_t*,AssemblyDevice> plengths,
                           Kokkos::View<size_t**,AssemblyDevice> pseed,
                           Kokkos::View<ScalarT***,AssemblyDevice> pvals,
                           Kokkos::View<ScalarT***,AssemblyDevice> kv_pvals);

    /** @brief Sacado-enabled seeding for general AD types. */
    template<class EvalT>
    void sacadoizeParams(const bool & seed_active,
                         Kokkos::View<int*,AssemblyDevice> ptypes,
                         Kokkos::View<size_t*,AssemblyDevice> plengths,
                         Kokkos::View<size_t**,AssemblyDevice> pseed,
                         Kokkos::View<ScalarT***,AssemblyDevice> pvals,
                         Kokkos::View<EvalT***,AssemblyDevice> kv_pvals);

    /** @brief Update parameters using a ROL-style optimization vector. */
    void updateParams(MrHyDE_OptVector & newparams);

#if defined(MrHyDE_ENABLE_HDSA)
    /** @brief Update parameters using a multivector for HDSA workflows. */
    void updateParams(const vector_RCP & newparams);
#endif

    /** @brief Update parameters by numerical list and type ID. */
    void updateParams(const std::vector<ScalarT> & newparams, const int & type);

    /** @brief Update dynamic parameters for the given time step. */
    void updateDynamicParams(const int & timestep);

    /** @brief Update parameters by name string type. */
    void updateParams(const std::vector<ScalarT> & newparams, const std::string & stype);

    /** @brief Set specific parameter values by parameter name. */
    void setParam(const std::vector<ScalarT> & newparams, const std::string & name);

    /** @brief Returns parameter arrays (active, inactive, stochastic...). */
    std::vector<Teuchos::RCP<std::vector<ScalarT> > > getParams(const int & type);

    /** @brief Returns the names of parameters of a given type. */
    std::vector<std::string> getParamsNames(const int & type);

    /** @brief Check if a string corresponds to a parameter name. */
    bool isParameter(const std::string & name);

    /** @brief Returns lengths of parameter vectors for each parameter. */
    std::vector<size_t> getParamsLengths(const int & type);

    /** @brief Returns parameters of a given type (string version). */
    std::vector<ScalarT> getParams(const std::string & stype);

    /** @brief Returns dynamic parameter values for a specific index. */
    vector<ScalarT> getParams(const std::string & stype, int dynamic_index);

    /** @brief Forms the combined optimization vector of current parameters. */
    MrHyDE_OptVector getCurrentVector();

    /** @brief Returns bounds for active parameters (ROL usage). */
    std::vector<Teuchos::RCP<std::vector<ScalarT> > > getActiveParamBounds();

    /** @brief Returns bounds for discretized parameters. */
    std::vector<vector_RCP> getDiscretizedParamBounds();

    /** @brief Stores the current parameter state in memory. */
    void stashParams();

    /** @brief Set parameters to their initial (input-file) values. */
    void setInitialParams();

    /** @brief Retrieve stochastic parameter values. */
    std::vector<ScalarT> getStochasticParams(const std::string & whichparam);

    /** @brief Retrieve fractional-order model parameters. */
    std::vector<ScalarT> getFractionalParams(const std::string & whichparam);

    /** @brief Constructs parameter mass matrix and diagonal scaling. */
    void setParamMass(Teuchos::RCP<LA_MultiVector> diag,
                      matrix_RCP mass);

    /** @brief Releases stored memory and cleans up data structures. */
    void purgeMemory();

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members (with inline comments)
    ///////////////////////////////////////////////////////////////////////////////////////////

    std::vector<std::string> blocknames;                       // Names of mesh blocks associated with parameters
    int spaceDim;                                               // Spatial dimension of the problem
    int debug_level;                                            // Verbosity/debugging level
    int numTimeSteps;                                           // Number of time steps for dynamic parameters

    Teuchos::RCP<const LA_Map> param_owned_map;                // Map for owned parameter DOFs
    Teuchos::RCP<const LA_Map> param_overlapped_map;           // Map for overlapped (ghosted) DOFs
    Teuchos::RCP<LA_CrsGraph> param_overlapped_graph;          // Sparsity graph for overlapped parameter operations

    Teuchos::RCP<LA_Export> param_exporter;                    // Exporter (owned -> overlapped)
    Teuchos::RCP<LA_Import> param_importer;                    // Importer (overlapped -> owned)

    std::vector<std::string> paramnames;                       // Names of all parameters

    std::vector<std::vector<std::vector<ScalarT>>> paramvals;  // Parameter values [dynamic][param][entry]

    Kokkos::View<ScalarT**,AssemblyDevice> paramvals_KV;       // Device view of scalar parameters

#ifndef MrHyDE_NO_AD
    Kokkos::View<AD**,AssemblyDevice> paramvals_KVAD;          // AD order-1 device view
    Kokkos::View<AD2**,AssemblyDevice> paramvals_KVAD2;        // AD order-2 device view
    Kokkos::View<AD4**,AssemblyDevice> paramvals_KVAD4;        // AD order-4 device view
    Kokkos::View<AD8**,AssemblyDevice> paramvals_KVAD8;        // AD order-8 device view
    Kokkos::View<AD16**,AssemblyDevice> paramvals_KVAD16;      // AD order-16 device view
    Kokkos::View<AD18**,AssemblyDevice> paramvals_KVAD18;      // AD order-18 device view
    Kokkos::View<AD24**,AssemblyDevice> paramvals_KVAD24;      // AD order-24 device view
    Kokkos::View<AD32**,AssemblyDevice> paramvals_KVAD32;      // AD order-32 device view
#endif

    Kokkos::View<ScalarT***,AssemblyDevice> paramvals_KV_ALL;  // All dynamic parameters [dynamic][param][entry]

#ifndef MrHyDE_NO_AD
    Kokkos::View<AD***,AssemblyDevice> paramvals_KVAD_ALL;     // AD: aggregated dynamic parameters
    Kokkos::View<AD2***,AssemblyDevice> paramvals_KVAD2_ALL;
    Kokkos::View<AD4***,AssemblyDevice> paramvals_KVAD4_ALL;
    Kokkos::View<AD8***,AssemblyDevice> paramvals_KVAD8_ALL;
    Kokkos::View<AD16***,AssemblyDevice> paramvals_KVAD16_ALL;
    Kokkos::View<AD18***,AssemblyDevice> paramvals_KVAD18_ALL;
    Kokkos::View<AD24***,AssemblyDevice> paramvals_KVAD24_ALL;
    Kokkos::View<AD32***,AssemblyDevice> paramvals_KVAD32_ALL;
#endif

    Kokkos::View<ScalarT**,AssemblyDevice> paramdot_KV;        // Time derivative of parameters
#ifndef MrHyDE_NO_AD
    Kokkos::View<AD**,AssemblyDevice> paramdot_KVAD;           // AD version of parameter derivatives
    Kokkos::View<AD2**,AssemblyDevice> paramdot_KVAD2;
    Kokkos::View<AD4**,AssemblyDevice> paramdot_KVAD4;
    Kokkos::View<AD8**,AssemblyDevice> paramdot_KVAD8;
    Kokkos::View<AD16**,AssemblyDevice> paramdot_KVAD16;
    Kokkos::View<AD18**,AssemblyDevice> paramdot_KVAD18;
    Kokkos::View<AD24**,AssemblyDevice> paramdot_KVAD24;
    Kokkos::View<AD32**,AssemblyDevice> paramdot_KVAD32;
#endif

    std::vector<vector_RCP> discretized_params;                // Parameter fields stored on owned DOFs
    std::vector<vector_RCP> discretized_params_over;           // Parameter fields stored on overlapped DOFs

    bool have_dynamic_discretized;                             // Whether discretized parameters change in time
    bool have_dynamic_scalar;                                  // Whether scalar parameters change in time
    int dynamic_timeindex;                                     // Current index for dynamic params
    ScalarT dynamic_dt;                                        // Time-step size for dynamic parameters

    Teuchos::RCP<const panzer::DOFManager> discparamDOF;       // DOF manager for discretized parameters
    std::vector<Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device>> DOF_LIDs; // Local DOF indices
    std::vector<std::vector<GO>> DOF_owned;                    // DOFs owned locally
    std::vector<std::vector<GO>> DOF_ownedAndShared;           // DOFs owned or shared locally

    std::vector<std::vector<std::vector<std::vector<GO>>>> DOF_GIDs; // GIDs for parameter DOFs [set][block][elem][gid]

    std::vector<std::vector<ScalarT>> paramLowerBounds;        // Lower bounds for scalar parameters
    std::vector<std::vector<ScalarT>> paramUpperBounds;        // Upper bounds for scalar parameters

    std::vector<std::string> discretized_param_basis_types;    // Basis functions used per parameter
    std::vector<int> discretized_param_basis_orders;           // Polynomial orders for basis
    std::vector<int> discretized_param_usebasis;               // Flags for enabling basis usage
    std::vector<std::string> discretized_param_names;          // Names of discretized parameters
    std::vector<std::string> discparam_distribution;           // Distribution types for discretized params
    
    std::vector<basis_RCP> discretized_param_basis;            // Basis objects for discretized parameters
    std::vector<bool> discretized_param_dynamic;               // Flags for time-dependent fields
    std::vector<bool> scalar_param_dynamic;                    // Flags for time-dependent scalars

    Teuchos::RCP<panzer::DOFManager> paramDOF;                 // DOF manager for all parameters
    std::vector<std::vector<int>> paramoffsets;                // Offsets in DOF structure
    std::vector<int> paramNumBasis;                            // Number of basis functions per parameter

    int numParamUnknowns;                                      // Total parameter DOFs (owned)
    int numParamUnknownsOS;                                    // Total parameter DOFs (owned + shared)
    int globalParamUnknowns;                                   // Global number of DOFs

    std::vector<GO> paramOwned;                                // Owned GIDs
    std::vector<GO> paramOwnedAndShared;                       // Owned + shared GIDs

    std::vector<int> paramtypes;                               // Type category for each parameter
    std::vector<std::vector<GO>> paramNodes;                   // Node lists per parameter
    std::vector<std::vector<GO>> paramNodesOS;                 // Node lists including overlaps

    size_t num_inactive_params;                                // Number of inactive parameters
    size_t num_active_params;                                  // Number of active parameters
    size_t num_stochastic_params;                              // Number of stochastic parameters
    size_t num_discrete_params;                                // Number of discrete parameters
    size_t num_discretized_params;                             // Number of spatially discretized parameters

    std::vector<ScalarT> initialParamValues;                   // Initial values for parameters
    std::vector<ScalarT> lowerParamBounds;                     // Global lower bounds
    std::vector<ScalarT> upperParamBounds;                     // Global upper bounds
    std::vector<ScalarT> discparamVariance;                    // Variance used for discretized stochastic params

    int verbosity;                                             // Level of runtime output
    std::string response_type;                                 // Response functional type
    std::string multigrid_type;                                // Multigrid configuration
    std::string smoother_type;                                 // Smoother used in multigrid
    bool discretized_stochastic;                               // Whether stochastic parameters are spatial fields
    bool use_custom_initial_param_guess;                       // Whether user provided an initial guess

    std::vector<std::string> stochastic_distribution;          // Distribution types for stochastic params
    std::vector<ScalarT> stochastic_mean;                      // Means of stochastic parameters
    std::vector<ScalarT> stochastic_variance;                  // Variances for stochastic parameters
    std::vector<ScalarT> stochastic_min;                       // Minimum bounds
    std::vector<ScalarT> stochastic_max;                       // Maximum bounds

    int batchID;                                               // Batch index for sampling or UQ workflows

    std::vector<ScalarT> s_exp;                                // Exponents for fractional parameters
    std::vector<ScalarT> h_mesh;                               // Mesh sizes for fractional parameters

    Teuchos::RCP<MpiComm> Comm;                                // MPI communicator
    Teuchos::RCP<MeshInterface>  mesh;                         // Mesh interface
    Teuchos::RCP<DiscretizationInterface> disc;                // Discretization interface
    Teuchos::RCP<PhysicsInterface> phys;                       // Physics interface
    Teuchos::RCP<Teuchos::ParameterList> settings;             // User-specified settings
    Teuchos::RCP<MrHyDE_Debugger> debugger;                    // Debugging utility

    Teuchos::RCP<LA_MultiVector> diagParamMass;                // Diagonal mass matrix entries
    matrix_RCP paramMass;                                      // Full mass matrix for parameter DOFs

    Teuchos::RCP<Teuchos::Time> constructortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParameterManager::constructor()");
    Teuchos::RCP<Teuchos::Time> updatetimer      = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParameterManager::updateParams()");
    Teuchos::RCP<Teuchos::Time> getcurrenttimer  = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParameterManager::getCurrentParams()");

  };
  
}

#endif
