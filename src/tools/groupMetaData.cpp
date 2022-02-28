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

#include "groupMetaData.hpp"
using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

GroupMetaData::GroupMetaData(const Teuchos::RCP<Teuchos::ParameterList> & settings,
                             const topo_RCP & cellTopo_,
                             const Teuchos::RCP<PhysicsInterface> & physics_RCP_,
                             const size_t & myBlock_,
                             const size_t & myLevel_, const int & numElem_,
                             const bool & build_face_terms_,
                             const vector<bool> & assemble_face_terms_,
                             const vector<string> & sidenames_,
                             const size_t & num_params) :
assemble_face_terms(assemble_face_terms_), build_face_terms(build_face_terms_),
myBlock(myBlock_), myLevel(myLevel_), numElem(numElem_),
physics_RCP(physics_RCP_), sidenames(sidenames_), numDiscParams(num_params),
cellTopo(cellTopo_) {

  Teuchos::TimeMonitor localtimer(*grptimer);
  
  compute_diff = settings->sublist("Postprocess").get<bool>("Compute Difference in Objective", true);
  useFineScale = settings->sublist("Postprocess").get<bool>("Use fine scale sensors",true);
  loadSensorFiles = settings->sublist("Analysis").get<bool>("Load Sensor Files",false);
  writeSensorFiles = settings->sublist("Analysis").get<bool>("Write Sensor Files",false);
  mortar_objective = settings->sublist("Solver").get<bool>("Use Mortar Objective",false);
  //storeAll = false;//settings->sublist("Solver").get<bool>("store all cell data",true);
  matrix_free = settings->sublist("Solver").get<bool>("matrix free",false);
  use_basis_database = settings->sublist("Solver").get<bool>("use basis database",false);
  use_mass_database = settings->sublist("Solver").get<bool>("use mass database",false);
  store_mass = settings->sublist("Solver").get<bool>("store mass",true);

  requiresTransient = true;
  if (settings->sublist("Solver").get<string>("solver","steady-state") == "steady-state") {
    requiresTransient = false;
  }
  
  requiresAdjoint = true;
  if (settings->sublist("Analysis").get<string>("analysis type","forward") == "forward") {
    requiresAdjoint = false;
  }
  
  compute_sol_avg = true;
  if (!(settings->sublist("Postprocess").get<bool>("write solution", false))) {
    compute_sol_avg = false;
  }
  
  multiscale = false;
  numnodes = cellTopo->getNodeCount();
  dimension = cellTopo->getDimension();
  
  if (dimension == 1) {
    numSides = 2;
  }
  else if (dimension == 2) {
    numSides = cellTopo->getSideCount();
  }
  else if (dimension == 3) {
    numSides = cellTopo->getFaceCount();
  }
  //response_type = "global";
  response_type = settings->sublist("Postprocess").get("response type", "pointwise");
  have_phi = false;
  have_rotation = false;
  have_extra_data = false;
  
  numSets = physics_RCP->setnames.size();
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void GroupMetaData::updatePhysicsSet(const size_t & set) {
  if (numSets> 1) {
    numDOF = set_numDOF[set];
    numDOF_host = set_numDOF_host[set];
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the storage required for the integration/basis info
///////////////////////////////////////////////////////////////////////////////////////

size_t GroupMetaData::getDatabaseStorage() {
  size_t mystorage = 0;
  size_t scalarcost = sizeof(ScalarT); // 8 bytes per double
  for (size_t k=0; k<database_basis.size(); ++k) {
    mystorage += scalarcost*database_basis[k].size();
  }
  for (size_t k=0; k<database_basis_grad.size(); ++k) {
    mystorage += scalarcost*database_basis_grad[k].size();
  }
  for (size_t k=0; k<database_basis_curl.size(); ++k) {
    mystorage += scalarcost*database_basis_curl[k].size();
  }
  for (size_t k=0; k<database_basis_div.size(); ++k) {
    mystorage += scalarcost*database_basis_div[k].size();
  }
  for (size_t k=0; k<database_side_basis.size(); ++k) {
    mystorage += scalarcost*database_side_basis[k].size();
  }
  for (size_t k=0; k<database_side_basis_grad.size(); ++k) {
    mystorage += scalarcost*database_side_basis_grad[k].size();
  }
  for (size_t k=0; k<database_face_basis.size(); ++k) {
    mystorage += scalarcost*database_face_basis[k].size();
  }
  for (size_t k=0; k<database_face_basis_grad.size(); ++k) {
    mystorage += scalarcost*database_face_basis_grad[k].size();
  }
  return mystorage;
}
