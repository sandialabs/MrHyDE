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

#ifndef CELLMETA_H
#define CELLMETA_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "physicsBase.hpp"
#include "physicsInterface.hpp"

#include <iostream>     
#include <iterator>     

namespace MrHyDE {
  
  class CellMetaData {
  public:
    
    CellMetaData() {} ;
    
    ~CellMetaData() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    CellMetaData(const Teuchos::RCP<Teuchos::ParameterList> & settings,
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
    
    //void clearPhysicalData();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    vector<bool> assemble_face_terms;
    bool storeBasisAtIP = true, requireBasisAtNodes = false, build_face_terms;
    
    size_t myBlock, myLevel, numSets;
    int numElem=0; // safeguard against case where a proc does not own any elem on a block
    Teuchos::RCP<PhysicsInterface> physics_RCP;
    string response_type;
    vector<string> sidenames;
    bool requiresTransient, requiresAdjoint, matrix_free;
    
    // Geometry Information
    size_t numnodes, numSides, dimension, numip, numsideip, numDiscParams;
    topo_RCP cellTopo;
    DRV refnodes;
    
    // Reference element integration and basis data
    DRV ref_ip, ref_wts;
    vector<DRV> ref_side_ip, ref_side_wts, ref_side_normals, ref_side_tangents, ref_side_tangentsU, ref_side_tangentsV;
    vector<DRV> ref_side_ip_vec, ref_side_normals_vec, ref_side_tangents_vec, ref_side_tangentsU_vec, ref_side_tangentsV_vec;
    
    // Physical element integration containers
    //DRV jacobian, jacobianDet, jacobianInv, physip, physwts;
    //DRV side_jacobian, side_jacobianDet, side_jacobianInv, side_physip, side_physwts, side_physnormals, side_phystangents;
    
    vector<string> basis_types;
    vector<basis_RCP> basis_pointers;
    vector<DRV> ref_basis, ref_basis_grad, ref_basis_div, ref_basis_curl;
    vector<vector<DRV> > ref_side_basis, ref_side_basis_grad, ref_side_basis_div, ref_side_basis_curl;
    vector<DRV> ref_basis_nodes; // basis functions at nodes (mostly for plotting)
    
    //vector<DRV> phys_basis1, phys_basisgrad1, phys_basisdiv1, phys_basiscurl1;
    //vector<DRV> phys_basis2, phys_basisgrad2, phys_basisdiv2, phys_basiscurl2;
    //vector<DRV> side_phys_basis1, side_phys_basisgrad1, side_phys_basisdiv1, side_phys_basiscurl1;
    //vector<DRV> side_phys_basis2, side_phys_basisgrad2, side_phys_basisdiv2, side_phys_basiscurl2;
    
    bool compute_diff, useFineScale, loadSensorFiles, writeSensorFiles;
    bool mortar_objective;
    bool exodus_sensors = false, compute_sol_avg = false;
    bool multiscale, have_cell_phi, have_cell_rotation, have_extra_data;
    
    // these are common to all elements/cells and are often used on both devices
    vector<Kokkos::View<int*,AssemblyDevice> > set_numDOF;
    vector<Kokkos::View<int*,HostDevice> > set_numDOF_host;
    
    Kokkos::View<int*,AssemblyDevice> numDOF, numParamDOF, numAuxDOF;
    Kokkos::View<int*,HostDevice> numDOF_host, numParamDOF_host, numAuxDOF_host;
    
    Teuchos::RCP<Teuchos::Time> celltimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cellMetaData::constructor()");
  };
  
}

#endif
