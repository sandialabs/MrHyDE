/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef CELLMETA_H
#define CELLMETA_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "physics_base.hpp"
#include "physicsInterface.hpp"

#include <iostream>     
#include <iterator>     

class CellMetaData {
public:
  
  CellMetaData() {} ;
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  CellMetaData(const Teuchos::RCP<Teuchos::ParameterList> & settings,
               const topo_RCP & cellTopo_,
               const Teuchos::RCP<physics> & physics_RCP_, const size_t & myBlock_,
               const size_t & myLevel_, const bool & memeff_) :
  cellTopo(cellTopo_), physics_RCP(physics_RCP_), myBlock(myBlock_),
  myLevel(myLevel_), memory_efficient(memeff_) {
  
    
    compute_diff = settings->sublist("Postprocess").get<bool>("Compute Difference in Objective", true);
    useFineScale = settings->sublist("Postprocess").get<bool>("Use fine scale sensors",true);
    loadSensorFiles = settings->sublist("Analysis").get<bool>("Load Sensor Files",false);
    writeSensorFiles = settings->sublist("Analysis").get<bool>("Write Sensor Files",false);
    mortar_objective = settings->sublist("Solver").get<bool>("Use Mortar Objective",false);
    
    multiscale = false;
    numnodes = cellTopo->getNodeCount();
    dimension = cellTopo->getDimension();
    
    if (dimension == 2) {
      numSides = cellTopo->getSideCount();
    }
    else if (dimension == 3) {
      numSides = cellTopo->getFaceCount();
    }
    response_type = "global";
    
    have_cell_phi = false;
    have_cell_rotation = false;
    
  }
  
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  bool memory_efficient;
  size_t myBlock, myLevel;
  Teuchos::RCP<physics> physics_RCP;
  string response_type;
  
  // Geometry Information
  size_t numnodes, numSides, dimension;
  topo_RCP cellTopo;
  
  bool compute_diff, useFineScale, loadSensorFiles, writeSensorFiles;
  bool mortar_objective;
  bool exodus_sensors = false;
  bool multiscale, have_cell_phi, have_cell_rotation;
  
};

#endif
