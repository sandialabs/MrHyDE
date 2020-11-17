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
#include "cell.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"
#include "meshInterface.hpp"
#include "discretizationInterface.hpp"

namespace MrHyDE {
  
  class SensorManager {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    SensorManager(Teuchos::RCP<Teuchos::ParameterList> & settings,
                  Teuchos::RCP<meshInterface> mesh,
                  Teuchos::RCP<discretization> & disc,
                  Teuchos::RCP<AssemblyManager<SolverNode> > & assembler,
                  Teuchos::RCP<ParameterManager<SolverNode> > & params) {
      
      int milo_debug_level = settings->get<int>("debug level",0);
      
      if (milo_debug_level > 0) {
        cout << "**** Starting SensorManager::constructor ..." << endl;
      }
      
      spaceDim = settings->sublist("Mesh").get<int>("dim",2);
      have_sensor_data = false;
      have_sensor_points = false;
      numSensors = 0;
      
      if (settings->sublist("Mesh").get<bool>("have element data", false)) {
        
        for (size_t i=0; i<assembler->cells[0].size(); i++) {
          vector<Kokkos::View<ScalarT**,HostDevice> > sensorLocations;
          vector<Kokkos::View<ScalarT**,HostDevice> > sensorData;
          int numSensorsInCell = mesh->efield_vals[0][i];
          
          if (numSensorsInCell > 0) {
            assembler->cells[0][i]->mySensorIDs.push_back(numSensors); // hack for dakota
            for (int j=0; j<numSensorsInCell; j++) {
              // sensorLocation
              Kokkos::View<ScalarT**,HostDevice> sensor_loc("sensor location",1,spaceDim);
              std::stringstream ssSensorNum;
              ssSensorNum << j+1;
              string sensorNum = ssSensorNum.str();
              string fieldLocx = "sensor_" + sensorNum + "_Loc_x";
              ptrdiff_t ind_Locx = std::distance(mesh->efield_names.begin(), std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocx));
              string fieldLocy = "sensor_" + sensorNum + "_Loc_y";
              ptrdiff_t ind_Locy = std::distance(mesh->efield_names.begin(), std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocy));
              sensor_loc(0,0) = mesh->efield_vals[ind_Locx][i];
              sensor_loc(0,1) = mesh->efield_vals[ind_Locy][i];
              if (spaceDim > 2) {
                string fieldLocz = "sensor_" + sensorNum + "_Loc_z";
                ptrdiff_t ind_Locz = std::distance(mesh->efield_names.begin(), std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldLocz));
                sensor_loc(0,2) = mesh->efield_vals[ind_Locz][i];
              }
              // sensorData
              Kokkos::View<ScalarT**,HostDevice> sensor_data("sensor data",1,mesh->numResponses+1);
              sensor_data(0,0) = 0.0; // time index
              for (int k=1; k<mesh->numResponses+1; k++) {
                std::stringstream ssRespNum;
                ssRespNum << k;
                string respNum = ssRespNum.str();
                string fieldResp = "sensor_" + sensorNum + "_Val_" + respNum;
                ptrdiff_t ind_Resp = std::distance(mesh->efield_names.begin(), std::find(mesh->efield_names.begin(), mesh->efield_names.end(), fieldResp));
                sensor_data(0,k) = mesh->efield_vals[ind_Resp][i];
              }
              sensorLocations.push_back(sensor_loc);
              sensorData.push_back(sensor_data);
              numSensors += 1; // solver variable (total number of sensors)
            }
          }
          assembler->cells[0][i]->cellData->exodus_sensors = true;
          assembler->cells[0][i]->numSensors = numSensorsInCell;
          assembler->cells[0][i]->sensorLocations = sensorLocations;
          assembler->cells[0][i]->sensorData = sensorData;
        }
        
        Kokkos::View<ScalarT**,HostDevice> tmp_sensor_points;
        vector<Kokkos::View<ScalarT**,HostDevice> > tmp_sensor_data;
        bool have_sensor_data = true;
        ScalarT sensor_loc_tol = 1.0;
        // only needed for passing of basis pointers
        for (size_t j=0; j<assembler->cells[0].size(); j++) {
          assembler->cells[0][j]->addSensors(sensor_points, sensor_loc_tol, sensor_data, have_sensor_data, disc, disc->basis_pointers[0], params->discretized_param_basis);
        }
      }
      else {
        if (settings->sublist("Analysis").get("have sensor data",false)) {
          data sdata("Sensor Measurements", spaceDim, settings->sublist("Analysis").get("sensor location file","sensor_points.dat"), settings->sublist("Analysis").get("sensor prefix","sensor"));
          sensor_data = sdata.getdata();
          sensor_points = sdata.getpoints();
          numSensors = sensor_points.extent(0);
          have_sensor_data = true;
          have_sensor_points = true;
        }
        else if (settings->sublist("Analysis").get("have sensor points",false)) {
          data sdata("Sensor Points", spaceDim, settings->sublist("Analysis").get("sensor location file","sensor_points.dat"));
          sensor_points = sdata.getpoints();
          numSensors = sensor_points.extent(0);
          have_sensor_data = false;
          have_sensor_points = true;
        }
        
        if (settings->sublist("Analysis").get("have sensor points",false)) {
          //sensor_locations = FCint(sensor_points.extent(0),2);
          ScalarT sensor_loc_tol = settings->sublist("Analysis").get("sensor location TOL",1.0E-6);
          for (size_t b=0; b<assembler->cells.size(); b++) {
            for (size_t j=0; j<assembler->cells[b].size(); j++) {
              assembler->cells[b][j]->addSensors(sensor_points, sensor_loc_tol, sensor_data, have_sensor_data, disc, disc->basis_pointers[b], params->discretized_param_basis);
            }
          }
        }
      }
      
      if (milo_debug_level > 0) {
        cout << "**** Finished SensorManager::constructor ..." << endl;
      }
      
    }
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    bool have_sensor_data, have_sensor_points;
    int spaceDim, numSensors;
    vector<Kokkos::View<ScalarT**,HostDevice> > sensor_data;
    Kokkos::View<ScalarT**,HostDevice> sensor_points;
    
    
    
  };
}

#endif
