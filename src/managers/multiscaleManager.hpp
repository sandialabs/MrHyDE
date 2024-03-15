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
  
  class MultiscaleManager {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>   SGLA_CrsMatrix;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> SGLA_MultiVector;
    typedef Teuchos::RCP<SGLA_MultiVector> vector_RCP;
    typedef Teuchos::RCP<SGLA_CrsMatrix>   matrix_RCP;
    
    #ifndef MrHyDE_NO_AD
      typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2;
    #else
      typedef View_Sc2 View_AD2;
    #endif

  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    MultiscaleManager() {};
    
    // ========================================================================================
    // ========================================================================================
    
    ~MultiscaleManager() {};

    /** @brief Constructor for MultiscaleManager
     *
     * @param[in] MacroComm_  MpiCommunicator from the macroscale
     * @param[in] mesh_  Macroscopic mesh
     * @param[in] settings_  ParameterList of settings from the input file
     * @param[in] groups_  Groups (collections of elements) from the macroscale 
     * @param[in] macro_functionManagers_  Macroscale function managers
     */
    
    /** @brief Constructor for MultiscaleManager
     *
     * @param[in] MacroComm_  MpiCommunicator from the macroscale
     * @param[in] mesh_  Macroscopic mesh
     * @param[in] settings_  ParameterList of settings from the input file
     * @param[in] groups_  Groups (collections of elements) from the macroscale 
     * @param[in] macro_functionManagers_  Macroscale function managers
     */
    
    MultiscaleManager(const Teuchos::RCP<MpiComm> & MacroComm_,
                      Teuchos::RCP<MeshInterface> & mesh_,
                      Teuchos::RCP<Teuchos::ParameterList> & settings_,
                      std::vector<std::vector<Teuchos::RCP<Group> > > & groups_,
                      std::vector<Teuchos::RCP<FunctionManager<AD> > > macro_functionManagers_);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /** @brief Set the information from the macro-scale that does not depend on the specific group
     */
    
    void setMacroInfo(std::vector<std::vector<basis_RCP> > & macro_basis_pointers,
                      std::vector<std::vector<std::string> > & macro_basis_types,
                      std::vector<std::vector<std::string> > & macro_varlist,
                      std::vector<std::vector<int> > macro_usebasis,
                      std::vector<std::vector<std::vector<int> > > & macro_offsets,
                      std::vector<Kokkos::View<int*,AssemblyDevice>> & macro_numDOF,
                      std::vector<std::string> & macro_paramnames,
                      std::vector<std::string> & macro_disc_paramnames);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /** @brief Initial assignment of subgrid models to groups
     */
    
    ScalarT initialize(vector<vector<int> > & sgmodels);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /** @brief Communicate macro information to subgrids, solve subgrid problem, communicate upscaled information to macro
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

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /** @brief   Re-assignment of subgrid models to groups
     */
    
    void update(vector<vector<int> > & sgmodels);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////

    /** @brief   Reset the time step
     */
    
    void reset();
    
    /** @brief   Tell subgrid models to move to the next time step
     */
    
    void completeTimeStep();

    /** @brief   Tell subgrid models to move to the next time stage
     */
    
    void completeStage();

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /** @brief   Update parameters
     */
    
    void updateParameters(std::vector<Teuchos::RCP<std::vector<AD> > > & params,
                          const std::vector<std::string> & paramnames);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /** @brief  Get the mean subgrid cell fields
     */
    
    Kokkos::View<ScalarT**,HostDevice> getMeanCellFields(const size_t & block, const int & timeindex,
                                                         const ScalarT & time, const int & numfields);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /** @brief   Update the microstructure
     */
    
    void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /** @brief  Return the total number of subgrid models defined by the user
     */
    
    size_t getNumberSubgridModels();

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /** @brief  Write to exodus
     */
    
    void writeSolution(const ScalarT & time, string & append);

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    bool subgrid_static, ml_training, have_ml_models;
    int verbosity, subgrid_model_selection;

    size_t num_training_steps, max_training_steps, macro_nl_iter;
    ScalarT reltol, abstol;
    std::vector<Teuchos::RCP<SubGridModel> > subgridModels;
    Teuchos::RCP<MpiComm> Comm, MacroComm;
    Teuchos::RCP<MeshInterface> macro_mesh;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<MrHyDE_Debugger> debugger;
    
    std::vector<std::vector<Teuchos::RCP<Group> > > groups;
    std::vector<Teuchos::RCP<Workset<AD> > > macro_wkset;
    std::vector<std::vector<Teuchos::RCP<SGLA_CrsMatrix> > > subgrid_projection_maps;
    std::vector<Teuchos::RCP<Amesos2::Solver<SGLA_CrsMatrix,SGLA_MultiVector> > > subgrid_projection_solvers;
    std::vector<Teuchos::RCP<FunctionManager<AD> > > macro_functionManagers;
    
    vector<vector<vector<ScalarT> > > ml_model_inputs; // [model][datapt][data]
    vector<vector<ScalarT> > ml_model_outputs, ml_model_extradata; // [model][datapt] 

  private:
  
    Teuchos::RCP<Teuchos::Time> resettimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager::reset()");
    Teuchos::RCP<Teuchos::Time> initializetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager::initialize()");
    Teuchos::RCP<Teuchos::Time> updatetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager::update()");
  };
}

#endif
