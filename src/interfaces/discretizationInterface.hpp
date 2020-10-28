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

#ifndef DISCINTERFACE_H
#define DISCINTERFACE_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "Panzer_DOFManager.hpp"
#include "Panzer_STK_Interface.hpp"

namespace MrHyDE {
  /*
  void static discretizationHelp(const string & details) {
    cout << "********** Help and Documentation for the Discretization Interface **********" << endl;
  }
  */
  
  class discretization {
  public:
    
    discretization() {} ;
    
    discretization(Teuchos::RCP<Teuchos::ParameterList> & settings,
                   Teuchos::RCP<MpiComm> & Comm_,
                   Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                   vector<vector<int> > & orders, vector<vector<string> > & types);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Create a pointer to an Intrepid or Panzer basis
    //////////////////////////////////////////////////////////////////////////////////////
    
    basis_RCP getBasis(const int & spaceDim, const topo_RCP & cellTopo,
                       const string & type, const int & degree);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getQuadrature(const topo_RCP & cellTopo, const int & order, DRV & ip, DRV & wts);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    DRV evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    DRV evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts,
                      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    DRV evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
                           const DRV & evalpts, const topo_RCP & cellTopo,
                           Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // After the mesh and the discretizations have been defined, we can create and add the physics
    // to the DOF manager
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<panzer::DOFManager> buildDOF(Teuchos::RCP<panzer_stk::STK_Interface> & mesh,
                                              vector<vector<string> > & varlist,
                                              vector<vector<string> > & types,
                                              vector<vector<int> > & orders,
                                              vector<vector<bool> > & useDG);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Public data
    ////////////////////////////////////////////////////////////////////////////////
    
    int milo_debug_level, spaceDim;
    Teuchos::RCP<MpiComm> Commptr;
    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
    vector<vector<basis_RCP> > basis_pointers;
    vector<vector<string> > basis_types;
    
    vector<DRV> ref_ip, ref_wts, ref_side_ip, ref_side_wts;
    vector<size_t> numip, numip_side;
    
    vector<vector<int> > cards;
    vector<vector<size_t> > myElements;
    
    
  };
  
}

#endif
