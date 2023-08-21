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

#ifndef MRHYDE_GROUPMETA_H
#define MRHYDE_GROUPMETA_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "physicsBase.hpp"
#include "physicsInterface.hpp"
#include "sparse3DView.hpp"

#include <iostream>     
#include <iterator>     

namespace MrHyDE {
  
  class GroupMetaData {
  
  public:
    
    GroupMetaData() {} ;
    
    ~GroupMetaData() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    GroupMetaData(const Teuchos::RCP<Teuchos::ParameterList> & settings,
                  const topo_RCP & cellTopo_,
                  const Teuchos::RCP<PhysicsInterface> & physics_RCP_,
                  const size_t & myBlock_,
                  const size_t & myLevel_, const int & numElem_,
                  const bool & build_face_terms_,
                  const vector<bool> & assemble_face_terms_,
                  const vector<string> & sidenames_,
                  const size_t & num_params);
    
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updatePhysicsSet(const size_t & set);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    size_t getDatabaseStorage();

    ///////////////////////////////////////////////////////////////////////////////////////
    // This class is really just for storing common meta-data for groups and bondary groups
    // As such, all data is public
    ///////////////////////////////////////////////////////////////////////////////////////
    
    vector<bool> assemble_face_terms;
    bool store_basis_at_ip = true, require_basis_at_nodes = false, build_face_terms;
    
    size_t my_block, my_level, num_sets;
    int num_elem=0; // safeguard against case where a proc does not own any elem on a block
    Teuchos::RCP<PhysicsInterface> physics;
    string response_type;
    vector<string> side_names;
    bool requires_transient, requires_adjoint, matrix_free, use_sparse_mass;
    
    // Geometry Information
    size_t num_nodes, num_sides, dimension, num_ip, num_side_ip, num_disc_params, current_stage=0;
    topo_RCP cell_topo;
    DRV ref_nodes;
    
    // Reference element integration and basis data
    DRV ref_ip, ref_wts;
    vector<DRV> ref_side_ip, ref_side_wts, ref_side_normals, ref_side_tangents, ref_side_tangentsU, ref_side_tangentsV;
    vector<DRV> ref_side_ip_vec, ref_side_normals_vec, ref_side_tangents_vec, ref_side_tangentsU_vec, ref_side_tangentsV_vec;
        
    vector<string> basis_types;
    vector<basis_RCP> basis_pointers;
    vector<DRV> ref_basis, ref_basis_grad, ref_basis_div, ref_basis_curl;
    vector<vector<DRV> > ref_side_basis, ref_side_basis_grad, ref_side_basis_div, ref_side_basis_curl;
    vector<DRV> ref_basis_nodes; // basis functions at nodes (mostly for plotting)
        
    bool compute_diff, use_fine_scale, load_sensor_files, write_sensor_files, use_basis_database = false, use_mass_database = false;
    bool mortar_objective;
    bool exodus_sensors = false, compute_sol_avg = false, store_mass = true;
    bool multiscale = false, have_phi, have_rotation, have_extra_data, have_multidata;
    
    // database of database basis information (optional)
    // Note that these are not CompressedViews.  CompressedViews use these.
    vector<View_Sc4> database_basis, database_basis_grad, database_basis_curl; // [basis type]
    vector<View_Sc3> database_basis_div;  // [basis type]
    vector<View_Sc4> database_side_basis, database_side_basis_grad;
    vector<vector<View_Sc4> > database_face_basis, database_face_basis_grad;

    // database of mass matrices
    vector<View_Sc3> database_mass;  // [set](dof,dof) 
    vector<Teuchos::RCP<Sparse3DView > > sparse_database_mass;  // [set](dof,dof) 

    // these are common to all elements/groups and are often used on both devices
    vector<Kokkos::View<int*,AssemblyDevice> > set_num_dof;
    vector<Kokkos::View<int*,HostDevice> > set_num_dof_host;
    
    Kokkos::View<int*,AssemblyDevice> num_dof, num_param_dof, num_aux_dof;
    Kokkos::View<int*,HostDevice> num_dof_host, num_param_dof_host, num_aux_dof_host;
    
    Teuchos::RCP<Teuchos::Time> grp_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::groupMetaData::constructor()");
  };
  
}

#endif
