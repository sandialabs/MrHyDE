/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

/** \file discretizationInterface.hpp
 *  \brief Provides the interface to discretization tools in Intrepid2 and DOF managers in Panzer.
 *
 *  This file defines the MrHyDE::DiscretizationInterface class, which
 *  manages finite element basis construction, quadrature rules, and
 *  mapping between reference and physical elements using Intrepid2 and Panzer.
 *
 *  \author Created by T. Wildey, expanded with full Doxygen documentation.
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

/**
 * \class DiscretizationInterface
 * \brief Provides access to Trilinos/Intrepid2/Panzer discretization tools.
 *
 * This interface handles finite element basis definitions, quadrature rule
 * creation, and transformations between reference and physical elements.
 */
class DiscretizationInterface {
public:
  
  /** \brief Default empty constructor. */
  DiscretizationInterface() {}
  
  /** \brief Default destructor. */
  ~DiscretizationInterface() {}
  
  /**
   * \brief Primary constructor for the discretization interface.
   *
   * Initializes mesh, physics, and parameter settings required for computing
   * basis functions, quadrature rules, and DOF layouts.
   *
   * \param[in] settings_ Global MrHyDE parameter list
   * \param[in] Comm_ MPI communicator
   * \param[in] mesh_ Mesh interface used for geometry and topology queries
   * \param[in] physics_ Physics interface defining variable types
   */
  DiscretizationInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                          Teuchos::RCP<MpiComm> & Comm_,
                          Teuchos::RCP<MeshInterface> & mesh_,
                          Teuchos::RCP<PhysicsInterface> & physics_);
  
  /**
   * \brief Creates an Intrepid2 basis object.
   *
   * \param[in] spaceDim Spatial dimension (1,2,3)
   * \param[in] cellTopo Cell topology (e.g. quad, hex, tet)
   * \param[in] type Basis type ("HGRAD", "HCURL", "HVOL", etc.)
   * \param[in] degree Polynomial degree
   *
   * \return A reference-counted pointer to the constructed basis
   */
  basis_RCP getBasis(const int & spaceDim,
                     const topo_RCP & cellTopo,
                     const string & type,
                     const int & degree);
  
  /**
   * \brief Generates quadrature points and weights on the reference element.
   *
   * \param[in] cellTopo Cell topology
   * \param[in] order Quadrature order
   * \param[out] ip Quadrature points (ref element)
   * \param[out] wts Quadrature weights
   */
  void getQuadrature(const topo_RCP & cellTopo,
                     const int & order,
                     DRV & ip,
                     DRV & wts);
  
  /**
   * \brief Sets up basis and quadrature data stored in GroupMetaData.
   *
   * \param[in] groupData Metadata shared across element blocks
   */
  void setReferenceData(Teuchos::RCP<GroupMetaData> & groupData);
  
  /**
   * \brief Computes physical quadrature points and weights via element IDs.
   *
   * \param[in] groupData Block metadata
   * \param[in] elemIDs List of element IDs
   * \param[out] ip Quadrature points on physical elements
   * \param[out] wts Quadrature weights on physical elements
   */
  void getPhysicalIntegrationData(Teuchos::RCP<GroupMetaData> & groupData,
                                  Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                  vector<View_Sc2> & ip,
                                  View_Sc2 wts);
  
  /**
   * \brief Computes physical quadrature points and weights via provided nodes.
   *
   * \param[in] groupData Block metadata
   * \param[in] nodes Coordinates of physical element nodes
   * \param[out] ip Quadrature points
   * \param[out] wts Quadrature weights
   */
  void getPhysicalIntegrationData(Teuchos::RCP<GroupMetaData> & groupData,
                                  DRV nodes,
                                  vector<View_Sc2> & ip,
                                  View_Sc2 wts);
  
  /**
   * \brief Computes only physical quadrature points (not weights).
   *
   * \param[in] groupData Block metadata
   * \param[in] elemIDs Element IDs used to extract node coordinates
   * \param[out] ip Physical quadrature points
   */
  void getPhysicalIntegrationPts(Teuchos::RCP<GroupMetaData> & groupData,
                                 Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                 vector<View_Sc2> & ip);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the physical integration points (not weights) for a group of elements.
   *
   * @param[in] groupData  Group metadata shared by all elements on a block.
   * @param[in] nodes      Kokkos::DRV containing nodal coordinates for the physical elements.
   * @param[out] ip        Vector of Kokkos::View objects containing quadrature points on each element.
   */
  void getPhysicalIntegrationPts(Teuchos::RCP<GroupMetaData> & groupData,
                                 DRV nodes, vector<View_Sc2> & ip);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the physical Jacobian for a group of elements using element IDs.
   *
   * @param[in] groupData  Group metadata shared by all elements on a block.
   * @param[in] elemIDs    List of element IDs used to gather nodes.
   * @param[out] jacobian  Kokkos::DRV containing the Jacobian matrices for each element.
   */
  void getJacobian(Teuchos::RCP<GroupMetaData> & groupData,
                   Kokkos::View<LO*,AssemblyDevice> elemIDs, DRV jacobian);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the physical Jacobian given explicit nodal coordinates.
   *
   * @param[in] groupData  Group metadata shared by all elements on a block.
   * @param[in] nodes      Kokkos::DRV containing nodal coordinates.
   * @param[out] jacobian  Kokkos::DRV containing the Jacobian matrices.
   */
  void getJacobian(Teuchos::RCP<GroupMetaData> & groupData,
                   DRV nodes, DRV jacobian);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the physical quadrature weights for a group of elements.
   *
   * @param[in] groupData  Group metadata shared by all elements on a block.
   * @param[in] elemIDs    List of element IDs used to compute nodal positions.
   * @param[out] jacobian  Kokkos::DRV containing the Jacobian matrices.
   * @param[out] wts       Kokkos::DRV containing physical quadrature weights.
   */
  void getPhysicalWts(Teuchos::RCP<GroupMetaData> & groupData,
                      Kokkos::View<LO*,AssemblyDevice> elemIDs, DRV jacobian, DRV wts);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the element measure (determinant-based) for each element.
   *
   * @param[in] groupData  Group metadata shared by all elements on a block.
   * @param[in] jacobian   Kokkos::DRV containing element Jacobians.
   * @param[out] measure   Kokkos::DRV storing computed measures.
   */
  void getMeasure(Teuchos::RCP<GroupMetaData> & groupData,
                  DRV jacobian, DRV measure);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the Frobenius norm of Jacobian matrices.
   *
   * @param[in] groupData  Group metadata shared by all elements on a block.
   * @param[in] jacobian   Kokkos::DRV containing Jacobian matrices.
   * @param[out] fro       Kokkos::DRV storing Frobenius norms.
   */
  void getFrobenius(Teuchos::RCP<GroupMetaData> & groupData,
                    DRV jacobian, DRV fro);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the physical volumetric basis and its derivatives using element IDs.
   *
   * @param[in] groupData           Group metadata shared across elements.
   * @param[in] elemIDs             Element IDs used to gather nodes.
   * @param[out] basis              Vector of Kokkos::Views containing basis values.
   * @param[out] basis_grad         Vector containing basis gradients.
   * @param[out] basis_curl         Vector containing basis curls.
   * @param[out] basis_div          Vector containing basis divergences.
   * @param[out] basis_nodes        Vector containing basis evaluated at nodes.
   * @param[in] apply_orientations  Whether to apply orientation corrections.
   */
  void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                  vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                  vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                  vector<View_Sc4> & basis_nodes,
                                  const bool & apply_orientations = true);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes the physical volumetric basis and derivatives from explicit nodal coordinates.
   *
   * @param[in] groupData           Group metadata shared across elements.
   * @param[in] nodes               DRV containing nodal coordinates.
   * @param[in] orientation         Orientation data for each element.
   * @param[out] basis              Vector containing basis values.
   * @param[out] basis_grad         Vector containing basis gradients.
   * @param[out] basis_curl         Vector containing basis curls.
   * @param[out] basis_div          Vector containing basis divergences.
   * @param[out] basis_nodes        Vector containing basis values at nodes.
   * @param[in] apply_orientations  Whether orientation correction should be applied.
   */
  void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                  vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                  vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                  vector<View_Sc4> & basis_nodes,
                                  const bool & apply_orientations = true);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes physical basis values (no derivatives) using element IDs.
   *
   * @param[in] groupData  Group metadata.
   * @param[in] elemIDs    Element IDs.
   * @param[out] basis     Vector storing basis values.
   */
  void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                  vector<View_Sc4> & basis);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes physical basis values (no derivatives) from explicit nodal coordinates.
   *
   * @param[in] groupData    Group metadata.
   * @param[in] nodes        Nodal coordinates.
   * @param[in] orientation  Orientation information.
   * @param[out] basis       Vector storing basis values.
   */
  void getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                  vector<View_Sc4> & basis);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes physical orientations for a set of elements.
   *
   * @param[in] groupData    Group metadata.
   * @param[in] eIndex       Element indices.
   * @param[out] orientation Output orientation values.
   * @param[in] use_block    Whether to use block-level orientation computation.
   */
  void getPhysicalOrientations(Teuchos::RCP<GroupMetaData> & groupData,
                               Kokkos::View<LO*,AssemblyDevice> eIndex,
                               Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                               const bool & use_block);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes face integration data for a given side using element IDs.
   *
   * @param[in] groupData    Group metadata.
   * @param[in] side         Local face/side ID.
   * @param[in] elemIDs      Element IDs.
   * @param[out] face_ip     Integration points on the face.
   * @param[out] face_wts    Integration weights.
   * @param[out] face_normals Computed normals on the face.
   */
  void getPhysicalFaceIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, const int & side, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                      vector<View_Sc2> & face_ip, View_Sc2 face_wts, vector<View_Sc2> & face_normals);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes face integration data from nodal coordinates.
   *
   * @param[in] groupData    Group metadata.
   * @param[in] side         Local face/side ID.
   * @param[in] nodes        Nodal coordinates.
   * @param[out] face_ip     Integration points.
   * @param[out] face_wts    Integration weights.
   * @param[out] face_normals Face normals.
   */
  void getPhysicalFaceIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, const int & side, DRV nodes,
                                      vector<View_Sc2> & face_ip, View_Sc2 face_wts, vector<View_Sc2> & face_normals);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes physical face basis values and gradients using element IDs.
   *
   * @param[in] groupData    Group metadata.
   * @param[in] side         Face/side index.
   * @param[in] elemIDs      Element IDs.
   * @param[out] basis       Basis values on the face.
   * @param[out] basis_grad  Basis gradients on the face.
   */
  void getPhysicalFaceBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & side, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                            vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes physical face basis and gradients using explicit nodal coordinates.
   *
   * @param[in] groupData    Group metadata.
   * @param[in] side         Local face/side ID.
   * @param[in] nodes        Nodal coordinates.
   * @param[in] orientation  Orientation information.
   * @param[out] basis       Basis values.
   * @param[out] basis_grad  Basis gradients.
   */
  void getPhysicalFaceBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & side, DRV nodes,
                            Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                            vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad);
  
  // ========================================================================================
  // ========================================================================================
  
  /**
   * @brief Computes boundary integration data using element IDs.
   *
   * @param[in] groupData  Group metadata.
   * @param[in] elemIDs    Element IDs.
   * @param[in] localSideID Local side ID for the boundary.
   * @param[out] ip        Integration points.
   * @param[out] wts       Quadrature weights.
   * @param[out] normals   Boundary normals.
   * @param[out] tangents  Boundary tangents.
   */
  void getPhysicalBoundaryIntegrationData(Teuchos::RCP<GroupMetaData> & groupData,
                                          Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                          LO & localSideID, vector<View_Sc2> & ip, View_Sc2 wts,
                                          vector<View_Sc2> & normals, vector<View_Sc2> & tangents);
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
  * @brief Computes physical boundary integration data including IPs, weights, normals, and tangents.
  *
  * @param[in,out] groupData Group metadata container receiving integration data.
  * @param[in] nodes Physical node coordinates for the boundary element.
  * @param[in] localSideID Local side index of the element.
  * @param[out] ip Integration point coordinates (per variable or component).
  * @param[out] wts Quadrature weights.
  * @param[out] normals Outward normals at boundary integration points.
  * @param[out] tangents Tangent vectors at boundary integration points.
  */
  void getPhysicalBoundaryIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                          LO & localSideID, vector<View_Sc2> & ip, View_Sc2 wts,
                                          vector<View_Sc2> & normals, vector<View_Sc2> & tangents);
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * @brief Computes physical boundary basis functions and related derivative data for a group of elements.
   *
   * @param[in]  groupData     Group meta data object shared by all elements on a block.
   * @param[in]  elemIDs       List of element IDs for which boundary basis is computed.
   * @param[in]  localSideID   Local ID of the boundary side.
   * @param[out] basis         Evaluated basis functions on boundary.
   * @param[out] basis_grad    Evaluated gradients of basis functions on boundary.
   * @param[out] basis_curl    Evaluated curls of basis functions on boundary.
   * @param[out] basis_div     Evaluated divergences of basis functions on boundary.
   */
  void getPhysicalBoundaryBasis(Teuchos::RCP<GroupMetaData> & groupData,
                                Kokkos::View<LO*,AssemblyDevice> elemIDs, LO & localSideID,
                                vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div);
  
  /**
   * @brief Computes physical boundary basis functions using explicit node data and orientation.
   *
   * @param[in]  groupData     Group meta data object shared by all elements on a block.
   * @param[in]  nodes         Node positions for the physical elements.
   * @param[in]  localSideID   Local ID of the boundary side.
   * @param[in]  orientation   Basis function orientation information.
   * @param[out] basis         Evaluated basis functions on boundary.
   * @param[out] basis_grad    Evaluated gradients of basis functions on boundary.
   * @param[out] basis_curl    Evaluated curls of basis functions on boundary.
   * @param[out] basis_div     Evaluated divergences of basis functions on boundary.
   */
  void getPhysicalBoundaryBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                LO & localSideID,
                                Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div);
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * @brief Evaluates basis functions at given reference points.
   *
   * @param[in]  basis_pointer   Pointer to the Intrepid2 basis.
   * @param[in]  evalpts         Points at which to evaluate basis functions.
   * @return A DRV containing basis values at the requested points.
   */
  DRV evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts);
  
  /**
   * @brief Evaluates oriented basis functions at given reference points.
   *
   * @param[in]  basis_pointer   Pointer to the Intrepid2 basis.
   * @param[in]  evalpts         Reference points for evaluating the basis.
   * @param[in]  orientation     Basis function orientation information.
   * @return A DRV containing oriented basis values.
   */
  DRV evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts,
                    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation);
  
  /**
   * @brief Evaluates basis functions for a specific block and basis ID using element IDs.
   *
   * @param[in]  groupData   Group meta data object.
   * @param[in]  block       Block index.
   * @param[in]  basisID     Basis identifier.
   * @param[in]  elemIDs     IDs of elements on which to evaluate basis.
   * @param[in]  evalpts     Reference evaluation points.
   * @param[in]  cellTopo    Cell topology object.
   * @return A DRV with evaluated basis functions.
   */
  DRV evaluateBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & block, const int & basisID,
                    const Kokkos::View<LO*,AssemblyDevice> elemIDs,
                    const DRV & evalpts, topo_RCP & cellTopo);
  
  /**
   * @brief Evaluates oriented basis functions using explicit node and orientation data.
   *
   * @param[in]  block       Block index.
   * @param[in]  basisID     Basis identifier.
   * @param[in]  nodes       Physical element nodes.
   * @param[in]  evalpts     Reference evaluation points.
   * @param[in]  cellTopo    Cell topology object.
   * @param[in]  orientation Orientation information for basis functions.
   * @return A DRV with oriented basis values.
   */
  DRV evaluateBasis(const int & block, const int & basisID, DRV nodes,
                    const DRV & evalpts, topo_RCP & cellTopo,
                    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation);
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * @brief Evaluates basis functions for elements using newly specified quadrature rules.
   *
   * @param[in]  groupData   Group meta data object.
   * @param[in]  block       Block index.
   * @param[in]  basisID     Basis identifier.
   * @param[in]  quad_rules  Requested quadrature rules.
   * @param[in]  elemIDs     List of elements.
   * @param[out] wts         Quadrature weights associated with evaluation.
   * @return A DRV with evaluated basis values.
   */
  DRV evaluateBasisNewQuadrature(Teuchos::RCP<GroupMetaData> & groupData, const int & block, const int & basisID, vector<string> & quad_rules,
                                 Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                 DRV & wts);
  
  /**
   * @brief Evaluates basis using nodes and orientation under new quadrature rules.
   *
   * @param[in]  block       Block index.
   * @param[in]  basisID     Basis identifier.
   * @param[in]  quad_rules  Quadrature rules.
   * @param[in]  nodes       Physical element nodes.
   * @param[in]  orientation Orientation information.
   * @param[out] wts         Quadrature weight output.
   * @return DRV containing basis evaluation output.
   */
  DRV evaluateBasisNewQuadrature(const int & block, const int & basisID, vector<string> & quad_rules,
                                 DRV nodes,
                                 Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                 DRV & wts);
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * @brief Evaluates gradients of basis functions for a block using element IDs.
   *
   * @param[in]  block        Block index.
   * @param[in]  basis_pointer Pointer to the basis.
   * @param[in]  elemIDs      Element IDs.
   * @param[in]  evalpts      Reference evaluation points.
   * @param[in]  cellTopo     Cell topology.
   * @return A DRV with gradient values.
   */
  DRV evaluateBasisGrads(const size_t & block, const basis_RCP & basis_pointer, const Kokkos::View<LO*,AssemblyDevice> elemIDs,
                         const DRV & evalpts, const topo_RCP & cellTopo);
  
  /**
   * @brief Evaluates basis gradients using explicit node data.
   *
   * @param[in]  basis_pointer Pointer to the basis.
   * @param[in]  nodes         Physical element nodes.
   * @param[in]  evalpts       Points at which gradients are evaluated.
   * @param[in]  cellTopo      Cell topology.
   * @return A DRV of gradient values.
   */
  DRV evaluateBasisGrads(const basis_RCP & basis_pointer, DRV nodes,
                         const DRV & evalpts, const topo_RCP & cellTopo);
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * @brief Second version of gradient evaluation, using group meta data.
   *
   * @param[in]  groupData     Group meta data.
   * @param[in]  block         Block index.
   * @param[in]  basis_pointer Basis pointer.
   * @param[in]  elemIDs       Element IDs.
   * @param[in]  evalpts       Evaluation points.
   * @param[in]  cellTopo      Topology.
   * @return A DRV of gradient values.
   */
  DRV evaluateBasisGrads2(Teuchos::RCP<GroupMetaData> & groupData, const size_t & block, const basis_RCP & basis_pointer, const Kokkos::View<LO*,AssemblyDevice> elemIDs,
                          const DRV & evalpts, const topo_RCP & cellTopo);
  
  /**
   * @brief Second version of gradient evaluation using nodes and orientation.
   *
   * @param[in]  basis_pointer Basis pointer.
   * @param[in]  nodes         Physical nodes.
   * @param[in]  evalpts       Evaluation points.
   * @param[in]  cellTopo      Topology information.
   * @param[in]  orientation   Orientation information.
   * @return DRV containing gradient evaluations.
   */
  DRV evaluateBasisGrads2(const basis_RCP & basis_pointer, DRV nodes,
                          const DRV & evalpts, const topo_RCP & cellTopo,
                          Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation);
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * @brief Builds the DOF managers after mesh and discretization setup is complete.
   */
  void buildDOFManagers();
  
  /**
   * @brief Sets boundary condition information on the DOF manager.
   *
   * @param[in]  set   Boundary set index.
   * @param[in]  DOF   DOF manager object.
   */
  void setBCData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF);
  
  /**
   * @brief Sets Dirichlet boundary condition data on the DOF manager.
   *
   * @param[in]  set   Boundary set index.
   * @param[in]  DOF   DOF manager object.
   */
  void setDirichletData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF);
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * @brief Retrieves side connectivity and topology information.
   *
   * @param[in]  set    Side set index.
   * @param[in]  block  Block index.
   * @param[in]  elem   Elements for which side info is computed.
   * @return A 4D View containing side info.
   */
  Kokkos::View<int****,HostDevice> getSideInfo(const size_t & set,
                                               const size_t & block,
                                               Kokkos::View<int*,HostDevice> elem);
  
  /**
   * @brief Computes offsets for DOFs on a given set and block.
   *
   * @param[in]  set    Set index.
   * @param[in]  block  Block index.
   * @return 2D vector of offsets.
   */
  vector<vector<int> > getOffsets(const int & set, const int & block);
  
  /**
   * @brief Retrieves global IDs for DOFs for an element.
   *
   * @param[in]  set    Set index.
   * @param[in]  block  Block index.
   * @param[in]  elem   Element index.
   * @return Vector of GIDs.
   */
  vector<GO> getGIDs(const size_t & set, const size_t & block, const size_t & elem);
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
   * @brief Maps physical points to reference coordinates for each element.
   *
   * @param[in]  phys_pts   Physical points to be mapped.
   * @param[in]  elemIDs    Element IDs.
   * @param[in]  block      Block index.
   * @param[in]  cellTopo   Cell topology.
   * @return DRV containing mapped reference coordinates.
   */
  DRV mapPointsToReference(DRV phys_pts, Kokkos::View<LO*,AssemblyDevice> elemIDs, const size_t & block, topo_RCP & cellTopo);
  
  /**
   * @brief Maps physical points to reference coordinates using provided node data.
   *
   * @param[in] phys_pts Physical coordinates to map.
   * @param[in] nodes    Coordinates of element nodes.
   * @param[in] cellTopo Cell topology of the elements.
   * @return DRV         Reference coordinates.
   */
  DRV mapPointsToReference(DRV phys_pts, DRV nodes, topo_RCP & cellTopo);
  
  /**
   * @brief Returns reference element node coordinates.
   *
   * @param[in] cellTopo Cell topology.
   * @return DRV         Reference node coordinates.
   */
  DRV getReferenceNodes(topo_RCP & cellTopo);
  
  /**
   * @brief Retrieves the physical node coordinates for a block over a set of elements.
   *
   * @param[in] block   Block index.
   * @param[in] elemIDs Element IDs for which nodes are requested.
   * @return DRV        Node coordinate array.
   */
  DRV getMyNodes(const size_t & block, Kokkos::View<LO*,AssemblyDevice> elemIDs);
  
  /**
   * @brief Maps reference points to physical space for a set of elements.
   *
   * @param[in] ref_pts Reference points.
   * @param[in] elemIDs Element IDs used to retrieve nodal data.
   * @param[in] block   Block index.
   * @param[in] cellTopo Cell topology.
   * @return DRV         Physical coordinates.
   */
  DRV mapPointsToPhysical(DRV ref_pts, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                          const size_t & block, topo_RCP & cellTopo);
  
  /**
   * @brief Maps reference points to physical coordinates using explicit node data.
   *
   * @param[in] ref_pts Reference points.
   * @param[in] nodes   Physical node coordinates.
   * @param[in] cellTopo Cell topology.
   * @return DRV         Physical points.
   */
  DRV mapPointsToPhysical(DRV ref_pts, DRV nodes, topo_RCP & cellTopo);
  
  /**
   * @brief Determines whether physical points lie within each physical element.
   *
   * @param[in] phys_pts Physical points to test.
   * @param[in] elemIDs  Element IDs.
   * @param[in] cellTopo Cell topology.
   * @param[in] block    Block index.
   * @param[in] tol      Inclusion tolerance.
   * @return DynRankView<int> Array containing inclusion flags per point per element.
   */
  Kokkos::DynRankView<int,PHX::Device> checkInclusionPhysicalData(
                                                                  DRV phys_pts, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                                  topo_RCP & cellTopo, const size_t & block,
                                                                  const ScalarT & tol);
  
  /**
   * @brief Determines point inclusion using explicit nodal coordinates.
   *
   * @param[in] phys_pts Physical coordinates.
   * @param[in] nodes    Node coordinates of the element.
   * @param[in] cellTopo Cell topology.
   * @param[in] tol      Tolerance for inclusion check.
   * @return DynRankView<int> Inclusion boolean values.
   */
  Kokkos::DynRankView<int,PHX::Device> checkInclusionPhysicalData(
                                                                  DRV phys_pts, DRV nodes,
                                                                  topo_RCP & cellTopo,
                                                                  const ScalarT & tol);
  
  /**
   * @brief Applies orientation data to a basis evaluation.
   *
   * @param[in] basis         Basis values before orientation.
   * @param[in] orientation   Intrepid2 orientation information.
   * @param[in] basis_pointer Pointer to basis object.
   * @return DRV              Oriented basis.
   */
  DRV applyOrientation(DRV basis,
                       Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                       basis_RCP & basis_pointer);
  
  /**
   * @brief Retrieves variable boundary condition information.
   *
   * @param[in] set   Set index.
   * @param[in] block Block index.
   * @return Kokkos::View<string**,HostDevice> Boundary condition variable matrix.
   */
  Kokkos::View<string**,HostDevice> getVarBCs(const size_t & set, const size_t & block);
  
  /**
   * @brief Computes the relative difference between two data arrays.
   *
   * @param[in] data1 First dataset.
   * @param[in] data2 Second dataset.
   * @return ScalarT  Relative difference measure.
   */
  ScalarT computeRelativeDifference(DRV data1, DRV data2);
  
  /**
   * @brief Clears locally stored LIDs for memory renewal.
   */
  void purgeLIDs();
  
  /**
   * @brief Purges internal memory used for discretization-related data.
   */
  void purgeMemory();
  
  /**
   * @brief Removes cached orientation data.
   */
  void purgeOrientations();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Public data
  ////////////////////////////////////////////////////////////////////////////////
  
  int verbosity; /**< Verbosity level for output and debugging. */
  int dimension; /**< Spatial dimension of the problem. */
  int quadorder; /**< Quadrature order used for integration. */
  
  double storage_proportion; /**< Fraction of data stored for memory optimization. */
  
  Teuchos::RCP<Teuchos::ParameterList> settings; /**< Global settings for discretization. */
  Teuchos::RCP<MpiComm> comm; /**< MPI communicator for parallel execution. */
  Teuchos::RCP<MeshInterface> mesh; /**< Mesh interface providing topology and geometry. */
  Teuchos::RCP<PhysicsInterface> physics; /**< Physics interface for material and model data. */
  Teuchos::RCP<MrHyDE_Debugger> debugger; /**< Debugging utility for diagnostics. */
  
  vector<vector<basis_RCP>> basis_pointers; /**< Basis function pointers per block and type. */
  vector<vector<string>> basis_types; /**< Basis type names per block. */
  
  vector<vector<vector<GO>>> point_dofs; /**< Point DOF IDs per set, block, and DOF. */
  vector<vector<vector<vector<LO>>>> dbc_dofs; /**< Dirichlet boundary condition DOF lists. */
  vector<string> block_names; /**< Names of mesh blocks. */
  vector<string> side_names; /**< Names of mesh side sets. */
  
  // Purgeable
  typedef Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device> lids_view_t;
  std::vector<lids_view_t> dof_lids; /**< Local ID lists for DOFs, purgeable. */
  std::vector<Kokkos::View<GO*,HostDevice>> dof_owned; /**< Owned DOF global IDs. */
  std::vector<Kokkos::View<GO*,HostDevice>> dof_owned_and_shared; /**< Owned + shared DOF global IDs. */
  Kokkos::View<Intrepid2::Orientation*,HostDevice> panzer_orientations; /**< Orientation data for basis functions. */
  
  vector<DRV> ref_ip; /**< Reference integration points per block. */
  vector<DRV> ref_wts; /**< Reference integration weights per block. */
  vector<DRV> ref_side_ip; /**< Reference integration points on sides. */
  vector<DRV> ref_side_wts; /**< Reference integration weights on sides. */
  
  vector<size_t> numip; /**< Number of volume integration points per block. */
  vector<size_t> numip_side; /**< Number of side integration points per block. */
  vector<int> num_derivs_required; /**< Number of derivatives required for basis evaluation. */
  
  vector<vector<int>> cards; /**< Basis cardinals per block and basis. */
  vector<Kokkos::View<LO*,HostDevice>> my_elements; /**< Local element IDs for each block. */
  
  vector<vector<Kokkos::View<int****,HostDevice>>> side_info; /**< Side information, rarely used. */
  vector<vector<vector<vector<string>>>> var_bcs; /**< Variable boundary condition names. */
  vector<vector<vector<vector<int>>>> offsets; /**< DOF offsets per set/block/variable. */
  
  bool have_dirichlet = false; /**< Whether Dirichlet BCs exist. */
  bool minimize_memory; /**< Whether to minimize memory footprint. */
  
  ////////////////////////////////////////////////////////////////////////////////
  // Private timers
  ////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<Teuchos::Time> set_bc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::setBCData()"); /**< Timer for BC setup. */
  Teuchos::RCP<Teuchos::Time> set_dbc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::setDirichletData()"); /**< Timer for Dirichlet BC setup. */
  Teuchos::RCP<Teuchos::Time> dofmgr_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::buildDOFManagers()"); /**< Timer for DOF manager construction. */
  
  Teuchos::RCP<Teuchos::Time> phys_vol_IP_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalIntegrationData()"); /**< Timer for physical IP computation. */
  Teuchos::RCP<Teuchos::Time> get_nodes_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getMyNodes()"); /**< Timer for node retrieval. */
  
  Teuchos::RCP<Teuchos::Time> phys_vol_data_total_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - total"); /**< Total volumetric data time. */
  Teuchos::RCP<Teuchos::Time> phys_vol_data_IP_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - ip"); /**< IP computation time. */
  Teuchos::RCP<Teuchos::Time> phys_vol_data_set_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - set Jac"); /**< Jacobian setup time. */
  Teuchos::RCP<Teuchos::Time> phys_vol_data_other_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - other Jac"); /**< Jacobian secondary computation time. */
  Teuchos::RCP<Teuchos::Time> phys_vol_data_hsize_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - hsize"); /**< Element size computation time. */
  Teuchos::RCP<Teuchos::Time> phys_orient_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalOrientations"); /**< Orientation computation time. */
  Teuchos::RCP<Teuchos::Time> phys_vol_data_wts_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - wts"); /**< Weight computation time. */
  Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - basis"); /**< Basis computation time. */
  
  Teuchos::RCP<Teuchos::Time> phys_basis_new_quad_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::evaluateBasisNewQuadrature()"); /**< Quadrature update timer. */
  
  Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_div_val_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HDIV-VALUE"); /**< HDIV basis value time. */
  Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_div_div_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HDIV-DIV"); /**< HDIV div time. */
  Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_curl_val_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HCURL-VALUE"); /**< HCURL basis value time. */
  Teuchos::RCP<Teuchos::Time> phys_vol_data_basis_curl_curl_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalVolumetricData - HCURL-CURL"); /**< HCURL curl time. */
  
  Teuchos::RCP<Teuchos::Time> phys_face_data_total_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - total"); /**< Total face data time. */
  Teuchos::RCP<Teuchos::Time> phys_face_data_IP_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - ip"); /**< Face IP computation time. */
  Teuchos::RCP<Teuchos::Time> phys_face_data_set_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - set Jac"); /**< Face Jacobian setup. */
  Teuchos::RCP<Teuchos::Time> phys_face_data_other_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - other Jac"); /**< Secondary face Jacobian. */
  Teuchos::RCP<Teuchos::Time> phys_face_data_hsize_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - hsize"); /**< Face element size. */
  Teuchos::RCP<Teuchos::Time> phys_face_data_wts_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - wts"); /**< Face weights. */
  Teuchos::RCP<Teuchos::Time> phys_face_data_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalFaceData - basis"); /**< Face basis evaluation. */
  
  Teuchos::RCP<Teuchos::Time> phys_bndry_data_total_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - total"); /**< Total boundary data time. */
  Teuchos::RCP<Teuchos::Time> phys_bndry_data_IP_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - ip"); /**< Boundary IP. */
  Teuchos::RCP<Teuchos::Time> phys_bndry_data_set_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - set Jac"); /**< Boundary Jacobian setup. */
  Teuchos::RCP<Teuchos::Time> phys_bndry_data_other_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - other Jac"); /**< Secondary boundary Jacobian. */
  Teuchos::RCP<Teuchos::Time> phys_bndry_data_hsize_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - hsize"); /**< Boundary element size. */
  Teuchos::RCP<Teuchos::Time> phys_bndry_data_wts_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - wts"); /**< Boundary weights. */
  Teuchos::RCP<Teuchos::Time> phys_bndry_data_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::getPhysicalBoundaryData - basis"); /**< Boundary basis. */
  
  Teuchos::RCP<Teuchos::Time> database_copy_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - copy"); /**< Database basis copy. */
  Teuchos::RCP<Teuchos::Time> database_orient_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - apply orient"); /**< Database orientation application. */
  Teuchos::RCP<Teuchos::Time> database_total_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - total"); /**< Total DB copy time. */
  Teuchos::RCP<Teuchos::Time> database_allocate_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface::copyDataFromDatabase() - allocate memory"); /**< DB allocation time. */
  };
  
  }
  
#endif
