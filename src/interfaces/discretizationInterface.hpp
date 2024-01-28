/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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
#include "meshInterface.hpp"
#include "groupMetaData.hpp"
#include "MrHyDE_Debugger.hpp"

namespace MrHyDE {
  
  /** \class  MrHyDE::DiscretizationInterface
   \brief  Provides the interface to the functions and classes in the Trilinos packages (panzer, Intrepid2) that handle the discretizations and degrees-of-freedom.
   */
  
  class DiscretizationInterface {
  public:
    
    DiscretizationInterface() {} ;
    
    ~DiscretizationInterface() {} ;
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Primary constructor for the MrHyDE Discretization Interface.  
     *
     * @param[in]  settings_  Main Teuchos Parameter List for MrHyDE
     * @param[in]  Comm_ Global MPI communicator
     * @param[in]  mesh_ MrHyDE Mesh Interface
     * @param[in]  physics_ MrHyDE Physics Interface
     *
     */

    DiscretizationInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                            Teuchos::RCP<MpiComm> & Comm_,
                            Teuchos::RCP<MeshInterface> & mesh_,
                            Teuchos::RCP<PhysicsInterface> & physics_);
                   
    //////////////////////////////////////////////////////////////////////////////////////
    // Create a pointer to an Intrepid or Panzer basis
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Defines an Intrepid2 basis.
     *
     * @param[in]  spaceDim  Spatial dimension (1, 2 or 3)
     * @param[in]  cellTopo  Cell/element topology (quad, tet, etc.)
     * @param[in]  type      String defining the basis type (HGRAD, HVOL, etc)
     * @param[in]  degree    Basis order
     */

    basis_RCP getBasis(const int & spaceDim, const topo_RCP & cellTopo, const string & type, const int & degree);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Defines the quadrature points and weights
     *
     * @param[in]  cellTopo  Cell/element topology (quad, tet, etc.)
     * @param[in]  order     Quadrature order
     * 
     * @param[out]  ip       Kokkos::DRV containing quadrature points on reference element.  Dims = (numip)x(spatial dimension)
     * @param[out]  wts      Kokkos::DRV containing quadrature weights on reference element.  Dims = (numip)
     */

    void getQuadrature(const topo_RCP & cellTopo, const int & order, DRV & ip, DRV & wts);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Sets up the quadrature and basis on the reference element
     *
     * @param[in]  groupData    Group meta data object shared by all elements on a block
     */

    void setReferenceData(Teuchos::RCP<GroupMetaData> & groupData);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Computes the physical integration points and weights for a group of elements
     *
     * @param[in]  groupData    Group meta data object shared by all elements on a block
     * @param[in]  elemIDs        List of element IDs.  Used to compute nodes and then call function using nodes.
     *
     * @param[out]  ip       vector<Kokkos::View> containing quadrature points on physical element.
     * @param[out]  wts     Kokkos::View containing quadrature weights on physical element.
     */

    void getPhysicalIntegrationData(Teuchos::RCP<GroupMetaData> & groupData,
                                    Kokkos::View<LO*,AssemblyDevice> elemIDs, vector<View_Sc2> & ip, View_Sc2 wts);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Computes the physical integration points and weights for a group of elements
     *
     * @param[in]  groupData    Group meta data object shared by all elements on a block
     * @param[in]  nodes        Kokkos::DRV containing the current set of nodes for the physical elements
     *
     * @param[out]  ip       vector<Kokkos::View> containing quadrature points on physical element.
     * @param[out]  wts     Kokkos::View containing quadrature weights on physical element.
     */
    
    void getPhysicalIntegrationData(Teuchos::RCP<GroupMetaData> & groupData,
                                    DRV nodes, vector<View_Sc2> & ip, View_Sc2 wts);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Computes the physical integration points (not weights) for a group of elements
     *
     * @param[in]  groupData    Group meta data object shared by all elements on a block
     * @param[in]  elemIDs        List of element IDs.  Used to compute nodes and then call function using nodes.
     *
     * @param[out]  ip       vector<Kokkos::View> containing quadrature points on physical element.
     */

    void getPhysicalIntegrationPts(Teuchos::RCP<GroupMetaData> & groupData,
                                    Kokkos::View<LO*,AssemblyDevice> elemIDs, vector<View_Sc2> & ip);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Computes the physical integration points (not weights) for a group of elements
     *
     * @param[in]  groupData    Group meta data object shared by all elements on a block
     * @param[in]  nodes        Kokkos::DRV containing the current set of nodes for the physical elements
     *
     * @param[out]  ip       vector<Kokkos::View> containing quadrature points on physical element.
     */
    
    void getPhysicalIntegrationPts(Teuchos::RCP<GroupMetaData> & groupData,
                                    DRV nodes, vector<View_Sc2> & ip);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Computes the physical Jacobian for a group of elements
     *
     * @param[in]  groupData    Group meta data object shared by all elements on a block
     * @param[in]  elemIDs        List of element IDs.  Used to compute nodes and then call function using nodes.
     *
     * @param[out]  jacobian     Kokkos::DRV containing the Jacobian for each element
     */
    
    void getJacobian(Teuchos::RCP<GroupMetaData> & groupData,
                     Kokkos::View<LO*,AssemblyDevice> elemIDs, DRV jacobian);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Computes the physical Jacobian for a group of elements
     *
     * @param[in]  groupData    Group meta data object shared by all elements on a block
     * @param[in]  nodes        Kokkos::DRV containing the current set of nodes for the physical elements
     *
     * @param[out]  jacobian     Kokkos::DRV containing the Jacobian for each element
     */
    
    void getJacobian(Teuchos::RCP<GroupMetaData> & groupData,
                     DRV nodes, DRV jacobian);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Computes the physical Jacobian for a group of elements
     *
     * @param[in]  groupData    Group meta data object shared by all elements on a block
     * @param[in]  elemIDs        List of element IDs.  Used to compute nodes and then call function using nodes.
     *
     * @param[out]  jacobian     Kokkos::DRV containing the Jacobian for each element
     * @param[out]  wts     Kokkos::DRV containing quadrature weights on physical element.
     */
    
    void getPhysicalWts(Teuchos::RCP<GroupMetaData> & groupData,
                        Kokkos::View<LO*,AssemblyDevice> elemIDs, DRV jacobian, DRV wts);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getMeasure(Teuchos::RCP<GroupMetaData> & groupData,
                    DRV jacobian, DRV measure);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getFrobenius(Teuchos::RCP<GroupMetaData> & groupData,
                      DRV jacobian, DRV fro);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, Kokkos::View<LO*,AssemblyDevice> elemIDs, 
                                    vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                    vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                    vector<View_Sc4> & basis_nodes,
                                    const bool & apply_orientations = true);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                    vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                    vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                    vector<View_Sc4> & basis_nodes,
                                    const bool & apply_orientations = true);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                    vector<View_Sc4> & basis);                                    
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                    vector<View_Sc4> & basis);                                    
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getPhysicalOrientations(Teuchos::RCP<GroupMetaData> & groupData,
                                 Kokkos::View<LO*,AssemblyDevice> eIndex,
                                 Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                 const bool & use_block);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getPhysicalFaceIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, const int & side, Kokkos::View<LO*,AssemblyDevice> elemIDs, 
                                        vector<View_Sc2> & face_ip, View_Sc2 face_wts, vector<View_Sc2> & face_normals);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getPhysicalFaceIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, const int & side, DRV nodes,
                                        vector<View_Sc2> & face_ip, View_Sc2 face_wts, vector<View_Sc2> & face_normals);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getPhysicalFaceBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & side, Kokkos::View<LO*,AssemblyDevice> elemIDs, 
                              vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void getPhysicalFaceBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & side, DRV nodes,
                              Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                              vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
                     
    void getPhysicalBoundaryIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, Kokkos::View<LO*,AssemblyDevice> elemIDs, 
                                            LO & localSideID,
                                            vector<View_Sc2> & ip, View_Sc2 wts, vector<View_Sc2> & normals,
                                            vector<View_Sc2> & tangents);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    void getPhysicalBoundaryIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                            LO & localSideID,
                                            vector<View_Sc2> & ip, View_Sc2 wts, vector<View_Sc2> & normals,
                                            vector<View_Sc2> & tangents);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
                                            
    void getPhysicalBoundaryBasis(Teuchos::RCP<GroupMetaData> & groupData, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                  LO & localSideID,
                                  vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                  vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    void getPhysicalBoundaryBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                  LO & localSideID,
                                  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
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
    
    DRV evaluateBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & block, const int & basisID, 
                      const Kokkos::View<LO*,AssemblyDevice> elemIDs,
                      const DRV & evalpts, topo_RCP & cellTopo);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV evaluateBasis(const int & block, const int & basisID, DRV nodes,
                      const DRV & evalpts, topo_RCP & cellTopo,
                      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    DRV evaluateBasisNewQuadrature(Teuchos::RCP<GroupMetaData> & groupData, const int & block, const int & basisID, vector<string> & quad_rules,
                                   Kokkos::View<LO*,AssemblyDevice> elemIDs, 
                                   DRV & wts);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV evaluateBasisNewQuadrature(const int & block, const int & basisID, vector<string> & quad_rules,
                                   DRV nodes, 
                                   Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                   DRV & wts);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    DRV evaluateBasisGrads(const size_t & block, const basis_RCP & basis_pointer, const Kokkos::View<LO*,AssemblyDevice> elemIDs,
                           const DRV & evalpts, const topo_RCP & cellTopo);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV evaluateBasisGrads(const basis_RCP & basis_pointer, DRV nodes,
                           const DRV & evalpts, const topo_RCP & cellTopo);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    DRV evaluateBasisGrads2(Teuchos::RCP<GroupMetaData> & groupData, const size_t & block, const basis_RCP & basis_pointer, const Kokkos::View<LO*,AssemblyDevice> elemIDs,
                           const DRV & evalpts, const topo_RCP & cellTopo);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV evaluateBasisGrads2(const basis_RCP & basis_pointer, DRV nodes,
                           const DRV & evalpts, const topo_RCP & cellTopo,
                           Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // After the mesh and the discretizations have been defined, we can create and add the physics
    // to the DOF manager
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void buildDOFManagers();
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void setBCData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void setDirichletData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    Kokkos::View<int****,HostDevice> getSideInfo(const size_t & set,
                                                 const size_t & block,
                                                 Kokkos::View<int*,HostDevice> elem);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    vector<vector<int> > getOffsets(const int & set, const int & block);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    vector<GO> getGIDs(const size_t & set, const size_t & block, const size_t & elem);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV mapPointsToReference(DRV phys_pts, Kokkos::View<LO*,AssemblyDevice> elemIDs, const size_t & block, topo_RCP & cellTopo);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV mapPointsToReference(DRV phys_pts, DRV nodes, topo_RCP & cellTopo);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV getReferenceNodes(topo_RCP & cellTopo);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV getMyNodes(const size_t & block, Kokkos::View<LO*,AssemblyDevice> elemIDs);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV mapPointsToPhysical(DRV ref_pts, Kokkos::View<LO*,AssemblyDevice> elemIDs, const size_t & block, topo_RCP & cellTopo);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV mapPointsToPhysical(DRV ref_pts, DRV nodes, topo_RCP & cellTopo);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    Kokkos::DynRankView<int,PHX::Device> checkInclusionPhysicalData(DRV phys_pts, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                                    topo_RCP & cellTopo, const size_t & block,
                                                                    const ScalarT & tol);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    Kokkos::DynRankView<int,PHX::Device> checkInclusionPhysicalData(DRV phys_pts, DRV nodes,
                                                                    topo_RCP & cellTopo, 
                                                                    const ScalarT & tol);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    DRV applyOrientation(DRV basis, Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                         basis_RCP & basis_pointer);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<string**,HostDevice> getVarBCs(const size_t & set, const size_t & block);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    ScalarT computeRelativeDifference(DRV data1, DRV data2);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    void purgeLIDs();
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    void purgeMemory();

    void purgeOrientations();
    
    ////////////////////////////////////////////////////////////////////////////////
    // Public data
    ////////////////////////////////////////////////////////////////////////////////
    
  //private:
    int verbosity, dimension, quadorder;
    double storage_proportion;
    
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<MpiComm> comm;
    Teuchos::RCP<MeshInterface> mesh;
    Teuchos::RCP<PhysicsInterface> physics;
    Teuchos::RCP<MrHyDE_Debugger> debugger;
    
    vector<vector<basis_RCP> > basis_pointers; // [block][basis]
    vector<vector<string> > basis_types; // [block][basis]
    
    vector<vector<vector<GO> > > point_dofs; // [set][block][dof]
    vector<vector<vector<vector<LO> > > > dbc_dofs; // [set][block][dof]
    vector<string> block_names, side_names;
    
    // Purgable
    std::vector<Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device>> dof_lids;
    std::vector<Kokkos::View<GO*,HostDevice> > dof_owned, dof_owned_and_shared;
    Kokkos::View<Intrepid2::Orientation*,HostDevice> panzer_orientations;

    vector<DRV> ref_ip, ref_wts, ref_side_ip, ref_side_wts;
    vector<size_t> numip, numip_side;
    vector<int> num_derivs_required;

    vector<vector<int> > cards;
    vector<Kokkos::View<LO*,HostDevice> > my_elements;
        
    vector<vector<Kokkos::View<int****,HostDevice> > > side_info; // rarely used
    vector<vector<vector<vector<string> > > > var_bcs; // [set][block][var][boundary]
    vector<vector<vector<vector<int> > > > offsets; // [set][block][var][dof]
    
    bool have_dirichlet = false, minimize_memory;

  private:
    
    Teuchos::RCP<Teuchos::Time> set_bc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::setBCData()");
    Teuchos::RCP<Teuchos::Time> set_dbc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::setDirichletData()");
    Teuchos::RCP<Teuchos::Time> dofmgr_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::buildDOFManagers()");
    
    Teuchos::RCP<Teuchos::Time> phys_vol_IP_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalIntegrationData()");
    Teuchos::RCP<Teuchos::Time> get_nodes_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getMyNodes()");

    Teuchos::RCP<Teuchos::Time> phys_vol_data_total_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - total");
    Teuchos::RCP<Teuchos::Time> phys_vol_data_IP_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - ip");
    Teuchos::RCP<Teuchos::Time> phys_vol_data_set_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - set Jac");
    Teuchos::RCP<Teuchos::Time> phys_vol_data_other_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - other Jac");
    Teuchos::RCP<Teuchos::Time> phys_vol_data_hsize_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - hsize");
    Teuchos::RCP<Teuchos::Time> phys_orient_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalOrientations");
    Teuchos::RCP<Teuchos::Time> phys_vol_data_wts_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - wts");
    Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - basis");
    
    Teuchos::RCP<Teuchos::Time> phys_basis_new_quad_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::evaluateBasisNewQuadrature()");
    
    Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_div_val_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HDIV-VALUE");
    Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_div_div_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HDIV-DIV");
    Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_curl_val_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HCURL-VALUE");
    Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_curl_curl_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HCURL-CURL");
    
    Teuchos::RCP<Teuchos::Time> phys_face_data_total_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - total");
    Teuchos::RCP<Teuchos::Time> phys_face_data_IP_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - ip");
    Teuchos::RCP<Teuchos::Time> phys_face_data_set_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - set Jac");
    Teuchos::RCP<Teuchos::Time> phys_face_data_other_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - other Jac");
    Teuchos::RCP<Teuchos::Time> phys_face_data_hsize_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - hsize");
    Teuchos::RCP<Teuchos::Time> phys_face_data_wts_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - wts");
    Teuchos::RCP<Teuchos::Time> phys_face_data_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - basis");
    
    Teuchos::RCP<Teuchos::Time> phys_bndry_data_total_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - total");
    Teuchos::RCP<Teuchos::Time> phys_bndry_data_IP_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - ip");
    Teuchos::RCP<Teuchos::Time> phys_bndry_data_set_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - set Jac");
    Teuchos::RCP<Teuchos::Time> phys_bndry_data_other_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - other Jac");
    Teuchos::RCP<Teuchos::Time> phys_bndry_data_hsize_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - hsize");
    Teuchos::RCP<Teuchos::Time> phys_bndry_data_wts_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - wts");
    Teuchos::RCP<Teuchos::Time> phys_bndry_data_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - basis");
    Teuchos::RCP<Teuchos::Time> database_copy_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - copy");
    Teuchos::RCP<Teuchos::Time> database_orient_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - apply orient");
    Teuchos::RCP<Teuchos::Time> database_total_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - total");
    Teuchos::RCP<Teuchos::Time> database_allocate_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - allocate memory");
  };
  
}

#endif
