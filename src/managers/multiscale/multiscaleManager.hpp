/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_MULTISCALE_MANAGER_H
#define MRHYDE_MULTISCALE_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "group.hpp"
#include "subgridModel.hpp"
#include "Amesos2.hpp"
#include "meshInterface.hpp"
#include "workset.hpp"
#include "MrHyDE_Debugger.hpp"

namespace MrHyDE {
  
  /**
   * @class MultiscaleManager
   * @brief Handles macro–micro coupling, subgrid model assignment, data transfer,
   *        and training/management of subgrid (micro-scale) models.
   */
  class MultiscaleManager {
    
    // --- Type aliases for readability ---
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>   SGLA_CrsMatrix;  ///< CRS matrix type for subgrid linear algebra
    typedef Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> SGLA_MultiVector; ///< MultiVector type for subgrid linear algebra
    typedef Teuchos::RCP<SGLA_MultiVector> vector_RCP; ///< RCP to subgrid MultiVector
    typedef Teuchos::RCP<SGLA_CrsMatrix>   matrix_RCP; ///< RCP to subgrid CRS matrix
    
    #ifndef MrHyDE_NO_AD
      typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2; ///< 2D AD view for micro-scale quantities
    #else
      typedef View_Sc2 View_AD2; ///< 2D scalar view if AD disabled
    #endif

  public:
    
    /**
     * @brief Default constructor (does not fully initialize anything)
     */
    MultiscaleManager() {};
    
    /**
     * @brief Destructor
     */
    ~MultiscaleManager() {};

    /**
     * @brief Constructor for fully building a multiscale manager.
     *
     * @param[in] MacroComm_   MPI communicator for the macro-problem
     * @param[in] mesh_        Pointer to macro mesh interface
     * @param[in] settings_    ParameterList containing multiscale settings
     * @param[in] groups_      Hierarchical element grouping for macro side
     * @param[in] macro_functionManagers_ Function managers for macro physics
     */
    MultiscaleManager(const Teuchos::RCP<MpiComm> & MacroComm_,
                      Teuchos::RCP<MeshInterface> & mesh_,
                      Teuchos::RCP<Teuchos::ParameterList> & settings_,
                      std::vector<std::vector<Teuchos::RCP<Group> > > & groups_,
                      std::vector<Teuchos::RCP<FunctionManager<AD> > > macro_functionManagers_);
    
    /**
     * @brief Set macro-scale data used by all groups.
     */
    void setMacroInfo(std::vector<std::vector<basis_RCP> > & macro_basis_pointers,
                      std::vector<std::vector<std::string> > & macro_basis_types,
                      std::vector<std::vector<std::string> > & macro_varlist,
                      std::vector<std::vector<int> > macro_usebasis,
                      std::vector<std::vector<std::vector<int> > > & macro_offsets,
                      std::vector<Kokkos::View<int*,AssemblyDevice>> & macro_numDOF,
                      std::vector<std::string> & macro_paramnames,
                      std::vector<std::string> & macro_disc_paramnames);
    
    /**
     * @brief Initial assignment of subgrid models to macro groups.
     * @param[in] sgmodels  Mapping from group/block → model index
     * @return A scalar diagnostic (e.g., initialization cost)
     */
    ScalarT initialize(vector<vector<int> > & sgmodels);

    /**
     * @brief Perform macro→micro→macro data transfer and micro-scale solve.
     */
    void evaluateMacroMicroMacroMap(Teuchos::RCP<Workset<AD>> & wkset,
                                    Teuchos::RCP<Group> & group,
                                    Teuchos::RCP<GroupMetaData> & groupData,
                                    const int & set,
                                    const bool & isTransient,
                                    const bool & isAdjoint,
                                    const bool & compute_jacobian,
                                    const bool & compute_sens,
                                    const int & num_active_params,
                                    const bool & compute_disc_sens,
                                    const bool & compute_aux_sens,
                                    const bool & store_adjPrev);

    /**
     * @brief Update assignment of groups → subgrid models.
     */
    void update(vector<vector<int> > & sgmodels);

    /**
     * @brief Refresh macro workset pointers for fast access.
     */
    void updateMacroWorkset(const int & block, const int & grp);

    /** @brief Reset micro-scale state for new time step. */
    void reset();

    /** @brief Advance subgrid models to next time step. */
    void completeTimeStep();

    /** @brief Advance subgrid models to next stage in multi-stage integrator. */
    void completeStage();
    
    /**
     * @brief Extract mean cell fields from subgrid models.
     */
    Kokkos::View<ScalarT**,HostDevice> getMeanCellFields(const size_t & block, const int & timeindex,
                                                         const ScalarT & time, const int & numfields);

    /** @brief Update microstructure geometry/rotation. */
    void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data);

    /** @brief Get number of subgrid models available. */
    size_t getNumberSubgridModels();

    /** @brief Write multiscale solution to Exodus file. */
    void writeSolution(const ScalarT & time, string & append);

    // -----------------------------------------------------------------------------
    // Public data members (should ideally be private, but keeping structure intact)
    // -----------------------------------------------------------------------------

    bool subgrid_static;              ///< Whether subgrid models are time-independent
    bool ml_training;                 ///< Whether ML training is active
    bool have_ml_models;              ///< Whether ML models are available
    int verbosity;                    ///< Output verbosity level
    int subgrid_model_selection;      ///< Strategy for choosing subgrid model

    size_t num_training_steps;        ///< Current ML training steps performed
    size_t max_training_steps;        ///< Maximum number of allowed ML training steps
    size_t macro_nl_iter;             ///< Nonlinear iteration counter on macro side

    ScalarT reltol;                   ///< Relative tolerance for micro-solvers
    ScalarT abstol;                   ///< Absolute tolerance for micro-solvers

    std::vector<Teuchos::RCP<SubGridModel> > subgridModels; ///< All available micro-scale models
    Teuchos::RCP<MpiComm> Comm;       ///< Communicator for subgrid computations
    Teuchos::RCP<MpiComm> MacroComm;  ///< Communicator for macro problem
    Teuchos::RCP<MeshInterface> macro_mesh; ///< Macro mesh interface
    Teuchos::RCP<Teuchos::ParameterList> settings; ///< Input-file settings
    Teuchos::RCP<MrHyDE_Debugger> debugger; ///< Debugging/diagnostics handler

    std::vector<std::vector<Teuchos::RCP<Group> > > groups; ///< Macro-level groups
    std::vector<Teuchos::RCP<Workset<AD> > > macro_wkset;   ///< Workset per macro block
    std::vector<std::vector<Teuchos::RCP<SGLA_CrsMatrix> > > subgrid_projection_maps; ///< Projection operators
    std::vector<Teuchos::RCP<Amesos2::Solver<SGLA_CrsMatrix,SGLA_MultiVector> > > subgrid_projection_solvers; ///< Linear solvers
    std::vector<Teuchos::RCP<FunctionManager<AD> > > macro_functionManagers; ///< Function managers

    vector<vector<vector<ScalarT> > > ml_model_inputs; ///< ML input datasets [model][point][features]
    vector<vector<ScalarT> > ml_model_outputs;         ///< ML output datasets [model][point]
    vector<vector<ScalarT> > ml_model_extradata;       ///< Additional ML data

  private:
    
    Teuchos::RCP<Teuchos::Time> resettimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager::reset()"); ///< Timer for reset()
    Teuchos::RCP<Teuchos::Time> initializetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager::initialize()"); ///< Timer for initialize()
    Teuchos::RCP<Teuchos::Time> updatetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager::update()"); ///< Timer for update()
  };
}

#endif
