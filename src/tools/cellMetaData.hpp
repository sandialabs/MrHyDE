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
               const size_t & myLevel_, const bool & assemble_face_terms_,
               const vector<string> & sidenames_, DRV ref_ip_, DRV ref_wts_,
               DRV ref_side_ip_, DRV ref_side_wts_, vector<string> & basis_types_,
               vector<basis_RCP> & basis_pointers_);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void setupReferenceBasis();

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  bool assemble_face_terms;
  size_t myBlock, myLevel;
  Teuchos::RCP<physics> physics_RCP;
  string response_type;
  vector<string> sidenames;
  
  // Geometry Information
  size_t numnodes, numSides, dimension, numip, numsideip;
  topo_RCP cellTopo;
  
  // Reference element integration and basis data
  DRV ref_ip, ref_wts;
  vector<DRV> ref_side_ip, ref_side_wts, ref_side_normals, ref_side_tangents, ref_side_tangentsU, ref_side_tangentsV;
  vector<DRV> ref_side_ip_vec, ref_side_normals_vec, ref_side_tangents_vec, ref_side_tangentsU_vec, ref_side_tangentsV_vec;
  
  vector<string> basis_types, disc_param_basis_types;
  vector<basis_RCP> basis_pointers;
  vector<DRV> ref_basis, ref_basis_grad, ref_basis_div, ref_basis_curl;
  vector<vector<DRV> > ref_side_basis, ref_side_basis_grad, ref_side_basis_div, ref_side_basis_curl;
  vector<vector<DRV> > param_basis_side_ref, param_basis_grad_side_ref;
  
  bool compute_diff, useFineScale, loadSensorFiles, writeSensorFiles;
  bool mortar_objective;
  bool exodus_sensors = false, compute_sol_avg = false;
  bool multiscale, have_cell_phi, have_cell_rotation;
  
  // these are common to all elements/cells and are often used on both devices
  Kokkos::View<int*,UnifiedDevice> numDOF, numParamDOF, numAuxDOF;
  
};

#endif
