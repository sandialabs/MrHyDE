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
  
  class DiscretizationInterface {
  public:
    
    DiscretizationInterface() {} ;
    
    ~DiscretizationInterface() {} ;
    
    DiscretizationInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                   Teuchos::RCP<MpiComm> & Comm_,
                   Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                   Teuchos::RCP<PhysicsInterface> & phys_);
                   
    //////////////////////////////////////////////////////////////////////////////////////
    // Create a pointer to an Intrepid or Panzer basis
    //////////////////////////////////////////////////////////////////////////////////////
    
    basis_RCP getBasis(const int & spaceDim, const topo_RCP & cellTopo, const string & type, const int & degree);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getQuadrature(const topo_RCP & cellTopo, const int & order, DRV & ip, DRV & wts);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void setReferenceData(Teuchos::RCP<CellMetaData> & cellData);
    
    void getPhysicalVolumetricData(Teuchos::RCP<CellMetaData> & cellData,
                                   DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                   vector<View_Sc2> & ip, View_Sc2 wts, View_Sc1 hsize,
                                   Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                   vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                   vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                   vector<View_Sc4> & basis_nodes,
                                   const bool & recompute_jac = true,
                                   const bool & recompute_orient = true);
    
    void getPhysicalVolumetricBasis(Teuchos::RCP<CellMetaData> & cellData,
                                    DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                    View_Sc2 wts,
                                    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                    vector<View_Sc4> & basis,
                                    const bool & recompute_jac,
                                    const bool & recompute_orient);

    void getPhysicalOrientations(Teuchos::RCP<CellMetaData> & cellData,
                                 Kokkos::View<LO*,AssemblyDevice> eIndex,
                                 Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                 const bool & use_block);
    
    void getPhysicalFaceData(Teuchos::RCP<CellMetaData> & cellData, const int & side,
                             DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                             Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                             vector<View_Sc2> & face_ip, View_Sc2 face_wts, vector<View_Sc2> & face_normals, View_Sc1 face_hsize,
                             vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                             const bool & recompute_jac = true,
                             const bool & recompute_orient = true);
    
    void getPhysicalBoundaryData(Teuchos::RCP<CellMetaData> & cellData,
                                 DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                 Kokkos::View<LO*,AssemblyDevice> localSideID,
                                 Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                 vector<View_Sc2> & ip, View_Sc2 wts, vector<View_Sc2> & normals,
                                 vector<View_Sc2> & tangents, View_Sc1 hsize,
                                 vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                 vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                 const bool & recompute_jac = true,
                                 const bool & recompute_orient = true);

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
    
    void setBCData();
    
    void setDirichletData();
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    Kokkos::View<int****,HostDevice> getSideInfo(const size_t & set,
                                                 const size_t & block,
                                                 Kokkos::View<int*,HostDevice> elem);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    vector<vector<int> > getOffsets(const int & set, const int & block);
    
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

    Kokkos::View<string**,HostDevice> getVarBCs(const size_t & set, const size_t & block);
    
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
    Teuchos::RCP<PhysicsInterface> phys;
    vector<vector<basis_RCP> > basis_pointers; // [block][basis]
    vector<vector<string> > basis_types; // [block][basis]
    
    vector<vector<vector<GO> > > point_dofs; // [set][block][dof]
    vector<vector<vector<vector<LO> > > > dbc_dofs; // [set][block][dof]
    vector<string> blocknames;
    
    // Purgable
    vector<stk::mesh::Entity> all_stkElems;
    vector<vector<stk::mesh::Entity> > block_stkElems;
    vector<Teuchos::RCP<panzer::DOFManager> > DOF;
    
    vector<DRV> ref_ip, ref_wts, ref_side_ip, ref_side_wts;
    vector<size_t> numip, numip_side;
    
    vector<vector<int> > cards;
    vector<vector<size_t> > myElements;
        
    vector<vector<Kokkos::View<int****,HostDevice> > > side_info;
    vector<vector<vector<vector<string> > > > var_bcs; // [set][block][var][boundary]
    vector<vector<vector<vector<int> > > > offsets; // [set][block][var][dof]
    
    
    bool haveDirichlet = false, minimize_memory;
    
    Teuchos::RCP<Teuchos::Time> setbctimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::setBCData()");
    Teuchos::RCP<Teuchos::Time> setdbctimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::setDirichletData()");
    Teuchos::RCP<Teuchos::Time> dofmgrtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::buildDOFManagers()");
    
    Teuchos::RCP<Teuchos::Time> physVolDataTotalTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - total");
    Teuchos::RCP<Teuchos::Time> physVolDataIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - ip");
    Teuchos::RCP<Teuchos::Time> physVolDataSetJacTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - set Jac");
    Teuchos::RCP<Teuchos::Time> physVolDataOtherJacTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - other Jac");
    Teuchos::RCP<Teuchos::Time> physVolDataHsizeTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - hsize");
    Teuchos::RCP<Teuchos::Time> physOrientTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalOrientations");
    Teuchos::RCP<Teuchos::Time> physVolDataWtsTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - wts");
    Teuchos::RCP<Teuchos::Time> physVolDataBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - basis");
    
    Teuchos::RCP<Teuchos::Time> physVolDataBasisDivValTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HDIV-VALUE");
    Teuchos::RCP<Teuchos::Time> physVolDataBasisDivDivTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HDIV-DIV");
    Teuchos::RCP<Teuchos::Time> physVolDataBasisCurlValTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HCURL-VALUE");
    Teuchos::RCP<Teuchos::Time> physVolDataBasisCurlCurlTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HCURL-CURL");
    
    Teuchos::RCP<Teuchos::Time> physFaceDataTotalTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - total");
    Teuchos::RCP<Teuchos::Time> physFaceDataIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - ip");
    Teuchos::RCP<Teuchos::Time> physFaceDataSetJacTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - set Jac");
    Teuchos::RCP<Teuchos::Time> physFaceDataOtherJacTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - other Jac");
    Teuchos::RCP<Teuchos::Time> physFaceDataHsizeTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - hsize");
    Teuchos::RCP<Teuchos::Time> physFaceDataWtsTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - wts");
    Teuchos::RCP<Teuchos::Time> physFaceDataBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - basis");
    
    Teuchos::RCP<Teuchos::Time> physBndryDataTotalTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - total");
    Teuchos::RCP<Teuchos::Time> physBndryDataIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - ip");
    Teuchos::RCP<Teuchos::Time> physBndryDataSetJacTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - set Jac");
    Teuchos::RCP<Teuchos::Time> physBndryDataOtherJacTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - other Jac");
    Teuchos::RCP<Teuchos::Time> physBndryDataHsizeTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - hsize");
    Teuchos::RCP<Teuchos::Time> physBndryDataWtsTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - wts");
    Teuchos::RCP<Teuchos::Time> physBndryDataBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - basis");
  };
  
}

#endif
