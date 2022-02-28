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

/** \file   discretizationInterface.hpp
 \brief  Contains the interface to the discretization tools in Intrepid2 and the degree-of-freedom managers in Panzer.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_DISCINTERFACE_H
#define MRHYDE_DISCINTERFACE_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "Panzer_DOFManager.hpp"
#include "Panzer_BlockedDOFManager.hpp"
#include "Panzer_ConnManager.hpp"
#include "Panzer_STK_Interface.hpp"
#include "physicsInterface.hpp"
#include "groupMetaData.hpp"

namespace MrHyDE {
  
  /** \class  MrHyDE::DiscretizationInterface
   \brief  Provides the interface to the functions and classes in the Trilinos packages (panzer, Intrepid2) that handle the discretizations and degrees-of-freedom.
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
    
    void setReferenceData(Teuchos::RCP<GroupMetaData> & groupData);
    
    void getPhysicalIntegrationData(Teuchos::RCP<GroupMetaData> & groupData,
                                    DRV nodes, vector<View_Sc2> & ip, View_Sc2 wts);

    void getJacobian(Teuchos::RCP<GroupMetaData> & groupData,
                     DRV nodes, DRV jacobian);

    void getMeasure(Teuchos::RCP<GroupMetaData> & groupData,
                    DRV jacobian, DRV measure);

    void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes, 
                                    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                    vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                    vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                    vector<View_Sc4> & basis_nodes,
                                    const bool & apply_orientations = true);
    
    void copyBasisFromDatabase(Teuchos::RCP<GroupMetaData> & groupData,
                               Kokkos::View<LO*,AssemblyDevice> basis_database_index, 
                               Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                               const bool & apply_orientation = false,
                               const bool & just_basis = false);

    void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                    vector<View_Sc4> & basis);                                    

    void getPhysicalOrientations(Teuchos::RCP<GroupMetaData> & groupData,
                                 Kokkos::View<LO*,AssemblyDevice> eIndex,
                                 Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                 const bool & use_block);
    
    void getPhysicalFaceIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, const int & side, DRV nodes, 
                                        vector<View_Sc2> & face_ip, View_Sc2 face_wts, vector<View_Sc2> & face_normals);

    void getPhysicalFaceBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & side, DRV nodes, 
                              Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                              vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad);

    void copyFaceBasisFromDatabase(Teuchos::RCP<GroupMetaData> & groupData,
                                   Kokkos::View<LO*,AssemblyDevice> basis_database_index, 
                                   Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                   const size_t & facenum,
                                   const bool & apply_orientation = false,
                                   const bool & just_basis = false);
                         
    void getPhysicalBoundaryIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes, 
                                            LO & localSideID,
                                            vector<View_Sc2> & ip, View_Sc2 wts, vector<View_Sc2> & normals,
                                            vector<View_Sc2> & tangents);
                                            
    void getPhysicalBoundaryBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes, 
                                  LO & localSideID,
                                  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                  vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                  vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div);

    void copySideBasisFromDatabase(Teuchos::RCP<GroupMetaData> & groupData,
                                   Kokkos::View<LO*,AssemblyDevice> basis_database_index, 
                                   Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                   const bool & apply_orientation = false,
                                   const bool & just_basis = false);
    
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
    
    void setBCData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF);
    
    void setDirichletData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    Kokkos::View<int****,HostDevice> getSideInfo(const size_t & set,
                                                 const size_t & block,
                                                 Kokkos::View<int*,HostDevice> elem);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    vector<vector<int> > getOffsets(const int & set, const int & block);
    
    vector<GO> getGIDs(const size_t & set, const size_t & block, const size_t & elem);

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

    void purgeLIDs();

    void purgeMemory();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Public data
    ////////////////////////////////////////////////////////////////////////////////
    
    int debug_level, verbosity, spaceDim;
    double storage_proportion;
    
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<MpiComm> Commptr;
    Teuchos::RCP<panzer_stk::STK_Interface> mesh;
    Teuchos::RCP<PhysicsInterface> phys;
    vector<vector<basis_RCP> > basis_pointers; // [block][basis]
    vector<vector<string> > basis_types; // [block][basis]
    
    vector<vector<vector<GO> > > point_dofs; // [set][block][dof]
    vector<vector<vector<vector<LO> > > > dbc_dofs; // [set][block][dof]
    vector<string> blocknames, sidenames;
    
    // Purgable
    std::vector<Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device>> DOF_LIDs;
    std::vector<std::vector<GO> > DOF_owned, DOF_ownedAndShared;
    std::vector<std::vector<std::vector<GO>>> DOF_GIDs; // [set][elem][gid] may consider a different storage strategy
    
    std::vector<Intrepid2::Orientation> panzer_orientations;

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
    Teuchos::RCP<Teuchos::Time> databaseCopyBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - copy");
    Teuchos::RCP<Teuchos::Time> databaseOrientTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - apply orient");
    Teuchos::RCP<Teuchos::Time> databaseTotalTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - total");
    Teuchos::RCP<Teuchos::Time> databaseAllocateTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - allocate memory");
  };
  
}

#endif
