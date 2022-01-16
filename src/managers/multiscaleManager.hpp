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

#ifndef MRHYDE_MULTISCALE_MANAGER_H
#define MRHYDE_MULTISCALE_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "group.hpp"
#include "subgridModel.hpp"
#include "Amesos2.hpp"
#include "meshInterface.hpp"

namespace MrHyDE {
  
  class MultiscaleManager {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>   SGLA_CrsMatrix;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> SGLA_MultiVector;
    typedef Teuchos::RCP<SGLA_MultiVector> vector_RCP;
    typedef Teuchos::RCP<SGLA_CrsMatrix>   matrix_RCP;
    
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    MultiscaleManager() {};
    
    ~MultiscaleManager() {};

    /* @brief Constructor for MultiscaleManager
     *
     * @param[in] MacroComm_  MpiCommunicator from the macroscale
     * @param[in] mesh_  Macroscopic mesh
     * @param[in] settings_  ParameterList of settings from the input file
     * @param[in] groups_  Groups (collections of elements) from the macroscale 
     * @param[in] macro_functionManagers_  Macroscale function managers
     */
    
    /* @brief Constructor for MultiscaleManager
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
                      std::vector<Teuchos::RCP<FunctionManager> > macro_functionManagers_);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set the information from the macro-scale that does not depend on the specific group
    ////////////////////////////////////////////////////////////////////////////////
    
    void setMacroInfo(std::vector<std::vector<basis_RCP> > & macro_basis_pointers,
                      std::vector<std::vector<std::string> > & macro_basis_types,
                      std::vector<std::vector<std::string> > & macro_varlist,
                      std::vector<std::vector<int> > macro_usebasis,
                      std::vector<std::vector<std::vector<int> > > & macro_offsets,
                      std::vector<Kokkos::View<int*,AssemblyDevice>> & macro_numDOF,
                      std::vector<std::string> & macro_paramnames,
                      std::vector<std::string> & macro_disc_paramnames);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Initial assignment of subgrid models to groups
    ////////////////////////////////////////////////////////////////////////////////
    
    ScalarT initialize();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Re-assignment of subgrid models to groups
    ////////////////////////////////////////////////////////////////////////////////
    
    ScalarT update();
    
    void reset();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Update parameters
    ////////////////////////////////////////////////////////////////////////////////
    
    void updateParameters(std::vector<Teuchos::RCP<std::vector<AD> > > & params,
                          const std::vector<std::string> & paramnames);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Get the mean subgrid cell fields
    ////////////////////////////////////////////////////////////////////////////////
    
    
    Kokkos::View<ScalarT**,HostDevice> getMeanCellFields(const size_t & block, const int & timeindex,
                                                         const ScalarT & time, const int & numfields);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Update the mesh data (for UQ studies)
    ////////////////////////////////////////////////////////////////////////////////
    
    void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    bool subgrid_static;
    int debug_level;
    std::vector<Teuchos::RCP<SubGridModel> > subgridModels;
    Teuchos::RCP<MpiComm> Comm, MacroComm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    std::vector<std::vector<Teuchos::RCP<Group> > > groups;
    std::vector<Teuchos::RCP<workset> > macro_wkset;
    std::vector<std::vector<Teuchos::RCP<SGLA_CrsMatrix> > > subgrid_projection_maps;
    std::vector<Teuchos::RCP<Amesos2::Solver<SGLA_CrsMatrix,SGLA_MultiVector> > > subgrid_projection_solvers;
    std::vector<Teuchos::RCP<FunctionManager> > macro_functionManagers;
    
    Teuchos::RCP<Teuchos::Time> resettimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager::reset()");
    Teuchos::RCP<Teuchos::Time> initializetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager::initialize()");
    Teuchos::RCP<Teuchos::Time> updatetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MultiscaleManager::update()");
  };
}

#endif
