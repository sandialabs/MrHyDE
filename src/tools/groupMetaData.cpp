/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "groupMetaData.hpp"
using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

GroupMetaData::GroupMetaData(const Teuchos::RCP<Teuchos::ParameterList> & settings,
                             const topo_RCP & cellTopo_,
                             const Teuchos::RCP<PhysicsInterface> & physics_,
                             const size_t & myBlock_,
                             const size_t & myLevel_, const int & numElem_,
                             const bool & build_face_terms_,
                             const vector<bool> & assemble_face_terms_,
                             const vector<string> & sidenames_,
                             const size_t & num_params) :
assemble_face_terms(assemble_face_terms_), build_face_terms(build_face_terms_),
my_block(myBlock_), my_level(myLevel_), num_elem(numElem_),
physics(physics_), side_names(sidenames_), num_disc_params(num_params),
cell_topo(cellTopo_) {

  Teuchos::TimeMonitor localtimer(*grp_timer);
  
  compute_diff = settings->sublist("Postprocess").get<bool>("Compute Difference in Objective", true);
  use_fine_scale = settings->sublist("Postprocess").get<bool>("Use fine scale sensors",true);
  load_sensor_files = settings->sublist("Analysis").get<bool>("Load Sensor Files",false);
  write_sensor_files = settings->sublist("Analysis").get<bool>("Write Sensor Files",false);
  mortar_objective = settings->sublist("Solver").get<bool>("Use Mortar Objective",false);
  //storeAll = false;//settings->sublist("Solver").get<bool>("store all cell data",true);
  matrix_free = settings->sublist("Solver").get<bool>("matrix free",false);
  use_basis_database = settings->sublist("Solver").get<bool>("use basis database",false);
  use_mass_database = settings->sublist("Solver").get<bool>("use mass database",false);
  store_mass = settings->sublist("Solver").get<bool>("store mass",true);
  use_sparse_mass = false;

  requires_transient = true;
  if (settings->sublist("Solver").get<string>("solver","steady-state") == "steady-state") {
    requires_transient = false;
  }
  
  requires_adjoint = true;
  if (settings->sublist("Analysis").get<string>("analysis type","forward") == "forward") {
    requires_adjoint = false;
  }
  
  compute_sol_avg = true;
  if (!(settings->sublist("Postprocess").get<bool>("write solution", false))) {
    compute_sol_avg = false;
  }
  multiscale = false;
  if (settings->isSublist("Subgrid")) {
    multiscale = true;
  }
  
  num_nodes = cell_topo->getNodeCount();
  dimension = cell_topo->getDimension();
  
  if (dimension == 1) {
    num_sides = 2;
  }
  else if (dimension == 2) {
    num_sides = cell_topo->getSideCount();
  }
  else if (dimension == 3) {
    num_sides = cell_topo->getFaceCount();
  }
  //response_type = "global";
  response_type = settings->sublist("Postprocess").get("response type", "pointwise");
  have_phi = false;
  have_rotation = false;
  have_extra_data = false;
  have_multidata = false;
  if (settings->sublist("Solver").get("have multidata", false)) {
    have_multidata = true;
  }
  num_sets = physics->set_names.size();
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void GroupMetaData::updatePhysicsSet(const size_t & set) {
  if (num_sets> 1) {
    num_dof = set_num_dof[set];
    num_dof_host = set_num_dof_host[set];
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
