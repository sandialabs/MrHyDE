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

#ifndef SENSORMANAGER_H
#define SENSORMANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "assemblyManager.hpp"

namespace MrHyDE {
  
  template< class Node>
  class SensorManager {
  public:
    
    // ========================================================================================
    // Constructor 
    // ========================================================================================
    
    SensorManager(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                  Teuchos::RCP<meshInterface> & mesh_,
                  Teuchos::RCP<AssemblyManager<Node> > & assembler_);
    
    // ========================================================================================
    // ========================================================================================
    
    void importSensorsFromExodus();
    
    // ========================================================================================
    // ========================================================================================
    
    void importSensorsFromFiles();
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<meshInterface> mesh;
    Teuchos::RCP<AssemblyManager<Node> > assembler;
    
    bool have_sensor_data, have_sensor_points;
    int spaceDim, numSensors, debug_level, verbosity;
    vector<Kokkos::View<ScalarT**,HostDevice> > sensor_data;
    Kokkos::View<ScalarT**,HostDevice> sensor_points;
    
    Teuchos::RCP<Teuchos::Time> importexodustimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SensorManager::importSensorsFromExodus()");
    Teuchos::RCP<Teuchos::Time> importfiletimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::SensorManager::importSensorsFromFiles()");
    
  };
  
  template class SensorManager<SolverNode>;
  #if defined(MrHyDE_ASSEMBLYSPACE_CUDA) && !defined(MrHyDE_SOLVERSPACE_CUDA)
    template class SensorManager<SubgridSolverNode>;
  #endif
  
}

#endif
