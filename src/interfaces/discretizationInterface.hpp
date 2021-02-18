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
#include "physicsInterface.hpp"
#include "cellMetaData.hpp"

namespace MrHyDE {
  /*
  void static discretizationHelp(const string & details) {
    cout << "********** Help and Documentation for the Discretization Interface **********" << endl;
  }
  */
  
  class discretization {
  public:
    
    discretization() {} ;
    
    discretization(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                   Teuchos::RCP<MpiComm> & Comm_,
                   Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                   Teuchos::RCP<physics> & phys_);
                   
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
    
    void setReferenceData(Teuchos::RCP<CellMetaData> & cellData);
    
    void getPhysicalVolumetricData(Teuchos::RCP<CellMetaData> & cellData,
                                   DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                   View_Sc3 ip, View_Sc2 wts, View_Sc1 hsize,
                                   Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                   vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                   vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                   vector<View_Sc4> & basis_nodes);
    
    void getPhysicalFaceData(Teuchos::RCP<CellMetaData> & cellData, const int & side,
                             DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                             Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                             View_Sc3 face_ip, View_Sc2 face_wts, View_Sc3 face_normals, View_Sc1 face_hsize,
                             vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad);
    
    void getPhysicalBoundaryData(Teuchos::RCP<CellMetaData> & cellData,
                                 DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                 Kokkos::View<LO*,AssemblyDevice> localSideID,
                                 Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                 View_Sc3 ip, View_Sc2 wts, View_Sc3 normals, View_Sc3 tangents, View_Sc1 hsize,
                                 vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                 vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div);

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
                           const DRV & evalpts, const topo_RCP & cellTopo);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    DRV evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
                           const DRV & evalpts, const topo_RCP & cellTopo,
                           Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // After the mesh and the discretizations have been defined, we can create and add the physics
    // to the DOF manager
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void buildDOFManagers();
    
    void setBCData(const bool & isaux);
    
    void setDirichletData(const bool & isaux);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    Kokkos::View<int****,HostDevice> getSideInfo(const size_t & block, Kokkos::View<int*,HostDevice> elem);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    vector<vector<int> > getOffsets(const int & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV mapPointsToReference(DRV phys_pts, DRV nodes, topo_RCP & cellTopo);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV getReferenceNodes(topo_RCP & cellTopo);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV mapPointsToPhysical(DRV ref_pts, DRV nodes, topo_RCP & cellTopo);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    Kokkos::DynRankView<int,PHX::Device> checkInclusionPhysicalData(DRV phys_pts, DRV nodes,
                                                                    topo_RCP & cellTopo,
                                                                    const ScalarT & tol);
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV applyOrientation(DRV basis, Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                         basis_RCP & basis_pointer);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    void purgeMemory();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Public data
    ////////////////////////////////////////////////////////////////////////////////
    
    int debug_level, verbosity, spaceDim;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<MpiComm> Commptr;
    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
    Teuchos::RCP<physics> phys;
    vector<vector<basis_RCP> > basis_pointers;
    vector<vector<string> > basis_types;
    Teuchos::RCP<panzer::DOFManager> DOF, auxDOF;
    vector<vector<GO> > point_dofs, aux_point_dofs;
    vector<vector<vector<LO> > > dbc_dofs, aux_dbc_dofs;
    
    vector<stk::mesh::Entity> all_stkElems;
    vector<vector<stk::mesh::Entity> > block_stkElems;
    
    vector<DRV> ref_ip, ref_wts, ref_side_ip, ref_side_wts;
    vector<size_t> numip, numip_side;
    
    vector<vector<int> > cards;
    vector<vector<size_t> > myElements;
    
    vector<Kokkos::View<int****,HostDevice> > side_info;
    vector<Kokkos::View<string**,HostDevice> > var_bcs, aux_var_bcs;
    vector<vector<vector<int> > > offsets, aux_offsets;
    bool haveDirichlet = false, haveAuxDirichlet = false;
    
    Teuchos::RCP<Teuchos::Time> setbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::setBCData()");
    Teuchos::RCP<Teuchos::Time> setdbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::setDirichletData()");
    
    Teuchos::RCP<Teuchos::Time> physVolDataTotalTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalVolumetricData - total");
    Teuchos::RCP<Teuchos::Time> physVolDataIPTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalVolumetricData - ip");
    Teuchos::RCP<Teuchos::Time> physVolDataSetJacTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalVolumetricData - set Jac");
    Teuchos::RCP<Teuchos::Time> physVolDataOtherJacTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalVolumetricData - other Jac");
    Teuchos::RCP<Teuchos::Time> physVolDataHsizeTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalVolumetricData - hsize");
    Teuchos::RCP<Teuchos::Time> physVolDataOrientTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalVolumetricData - orientations");
    Teuchos::RCP<Teuchos::Time> physVolDataWtsTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalVolumetricData - wts");
    Teuchos::RCP<Teuchos::Time> physVolDataBasisTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalVolumetricData - basis");
    
    Teuchos::RCP<Teuchos::Time> physFaceDataTotalTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalFaceData - total");
    Teuchos::RCP<Teuchos::Time> physFaceDataIPTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalFaceData - ip");
    Teuchos::RCP<Teuchos::Time> physFaceDataSetJacTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalFaceData - set Jac");
    Teuchos::RCP<Teuchos::Time> physFaceDataOtherJacTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalFaceData - other Jac");
    Teuchos::RCP<Teuchos::Time> physFaceDataHsizeTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalFaceData - hsize");
    Teuchos::RCP<Teuchos::Time> physFaceDataOrientTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalFaceData - orientations");
    Teuchos::RCP<Teuchos::Time> physFaceDataWtsTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalFaceData - wts");
    Teuchos::RCP<Teuchos::Time> physFaceDataBasisTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalFaceData - basis");
    
    Teuchos::RCP<Teuchos::Time> physBndryDataTotalTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalBoundaryData - total");
    Teuchos::RCP<Teuchos::Time> physBndryDataIPTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalBoundaryData - ip");
    Teuchos::RCP<Teuchos::Time> physBndryDataSetJacTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalBoundaryData - set Jac");
    Teuchos::RCP<Teuchos::Time> physBndryDataOtherJacTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalBoundaryData - other Jac");
    Teuchos::RCP<Teuchos::Time> physBndryDataHsizeTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalBoundaryData - hsize");
    Teuchos::RCP<Teuchos::Time> physBndryDataOrientTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalBoundaryData - orientations");
    Teuchos::RCP<Teuchos::Time> physBndryDataWtsTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalBoundaryData - wts");
    Teuchos::RCP<Teuchos::Time> physBndryDataBasisTimer = Teuchos::TimeMonitor::getNewCounter("MILO::discretization::getPhysicalBoundaryData - basis");
  };
  
}

#endif
