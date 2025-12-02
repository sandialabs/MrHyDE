/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

/** \file   meshInterface.hpp
 *  \brief Interface to the mesh objects, which includes a Panzer STK interface, and various functionality
 *  for modifying and extracting data from meshes.
 *  \author Created by T. Wildey
 */

#ifndef MRHYDE_MESHINTERFACE_H
#define MRHYDE_MESHINTERFACE_H

#include "trilinos.hpp"
#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_LineMeshFactory.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SquareTriMeshFactory.hpp"
#include "Panzer_STK_CubeHexMeshFactory.hpp"
#include "Panzer_STK_CubeTetMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "simplemeshmanager.hpp"
#include "MrHyDE_Debugger.hpp"
#include "preferences.hpp"
#include "data.hpp"

namespace MrHyDE {

/** \class MeshInterface
 *  \brief Interface to the Trilinos packages (Panzer, STK) for mesh construction, modification, and queries.
 *  \details Provides mesh initialization, field registration, I/O operations, access to STK entities, and mesh utilities.
 */
class MeshInterface {
public:
  
  /** \brief Default constructor.
   *  \details Initializes an empty MeshInterface; mesh is not yet constructed.
   */
  MeshInterface() {};
  
  /** \brief Default destructor. */
  ~MeshInterface() {};
  
  /** \brief Standard constructor.
   *  \param settings_ Parameter list containing mesh and problem configuration.
   *  \param Commptr_ MPI communicator.
   *  \details Mesh remains uninitialized after construction; user must call finalize().
   */
  MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_, const Teuchos::RCP<MpiComm> & Commptr_);
  
  /** \brief Constructor that reuses an existing mesh factory and STK interface.
   *  \param settings_ Parameter list for mesh/problem configuration.
   *  \param Commptr_ MPI communicator.
   *  \param mesh_factory_ Pre-existing STK mesh factory.
   *  \param mesh_ Pre-built STK mesh interface.
   *  \details Useful for reading meshes from disk or sharing STK interfaces across modules.
   */
  MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                const Teuchos::RCP<MpiComm> & Commptr_,
                Teuchos::RCP<panzer_stk::STK_MeshFactory> & mesh_factory_,
                Teuchos::RCP<panzer_stk::STK_Interface> & mesh_);
  
  /** \brief Finalizes mesh setup by registering fields and completing mesh construction.
   *  \param varlist Nested list of variable names defined by the physics interface.
   *  \param vartypes Variable type specification parallel to varlist.
   *  \param derivedlist Lists of derived field names for each block.
   *  \details This method must be called before any mesh queries or data operations.
   */
  void finalize(std::vector<std::vector<std::vector<string> > > varlist,
                std::vector<std::vector<std::vector<string> > > vartypes,
                std::vector<std::vector<std::vector<std::vector<string> > > > derivedlist);
  
  
  /**
   * \brief Perturbs mesh nodes for ALE (Arbitrary Lagrangian–Eulerian) motion.
   * \param b Block index identifying which block of nodes to perturb.
   * \param blocknodes View containing the coordinates of nodes in the block.
   * \return Updated node coordinate view after perturbation.
   * \details This operation invalidates cached basis and integration data; they must be recomputed.
   */
  DRV perturbMesh(const int & b, DRV & blocknodes);
  
  /** \brief Initializes an Exodus output file for solution writes.
   *  \param filename Name of the Exodus file to create.
   */
  void setupExodusFile(const string & filename);
  
  /** \brief Initializes an Exodus file for optimization output.
   *  \param filename Target Exodus filename.
   */
  void setupOptimizationExodusFile(const string & filename);
  
  /** \brief Writes solution field data (nodal) to a specific block.
   *  \param var Field name.
   *  \param blockID Block to which data belongs.
   *  \param myElements Local element IDs.
   *  \param soln 2D view of nodal solution values.
   */
  void setSolutionFieldData(string var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT**,HostDevice> soln);
  
  /** \brief Writes cell-based solution field data.
   *  \param var Field name.
   *  \param blockID Block identifier.
   *  \param myElements Local element IDs.
   *  \param soln 1D view of cell values.
   */
  void setCellFieldData(string var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT*,HostDevice> soln);
  
  /** \brief Writes optimization-based nodal field data.
   *  \param var Field name.
   *  \param blockID Block name.
   *  \param myElements Local element IDs.
   *  \param soln 2D view of nodal optimization values.
   */
  void setOptimizationSolutionFieldData(string & var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT**,HostDevice> soln);
  
  /** \brief Writes optimization cell field data.
   *  \param var Field name.
   *  \param blockID Block name.
   *  \param myElements Local element IDs.
   *  \param soln 1D view of cell values.
   */
  void setOptimizationCellFieldData(string & var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT*,HostDevice> soln);
  
  /** \brief Writes to the main Exodus file at a given time step.
   *  \param currenttime Simulation time.
   */
  void writeToExodus(const double & currenttime);
  
  /** \brief Writes solution fields to a specific Exodus file.
   *  \param filename File to write to.
   */
  void writeToExodus(const string & filename);
  
  /** \brief Writes optimization fields to the optimization Exodus file.
   *  \param currenttime Time stamp.
   */
  void writeToOptimizationExodus(const double & currenttime);
  
  /** \brief Writes optimization fields to a named Exodus file.
   *  \param filename Output file name.
   */
  void writeToOptimizationExodus(const string & filename);
  
  /** \brief Gets list of block names in the mesh.
   *  \return Vector of block names.
   */
  vector<string> getBlockNames();
  
  /** \brief Gets list of side set names.
   *  \return Vector of side names.
   */
  vector<string> getSideNames();
  
  /** \brief Gets list of node set names.
   *  \return Vector of node names.
   */
  vector<string> getNodeNames();
  
  /** \brief Returns spatial dimension of the mesh.
   *  \return Dimension (1, 2, or 3).
   */
  int getDimension();
  
  /** \brief Returns cell topology for a block.
   *  \param blockID Block identifier.
   *  \return Cell topology reference-counted pointer.
   */
  topo_RCP getCellTopology(string & blockID);
  
  /** \brief Accessor for the STK connection manager.
   *  \return RCP to panzer::ConnManager.
   */
  Teuchos::RCP<panzer::ConnManager> getSTKConnManager();
  
  
  
  /** \brief Set a new STK mesh interface.
   *  \param new_mesh RCP to the new STK mesh interface.
   */
  /**
   * @brief Set the underlying STK mesh interface.
   * @param new_mesh Reference-counted pointer to an STK_Interface instance.
   */
  void setSTKMesh(Teuchos::RCP<panzer_stk::STK_Interface> & new_mesh);
  
  /**
   * @brief Get all STK elements owned by this MPI rank.
   * @return Vector of STK element entities.
   */
  vector<stk::mesh::Entity> getMySTKElements();
  
  /**
   * @brief Get all STK elements for a specific block owned by this rank.
   * @param blockID Block identifier.
   * @return Vector of STK element entities.
   */
  vector<stk::mesh::Entity> getMySTKElements(string & blockID);
  
  /**
   * @brief Retrieve node IDs for an STK element.
   * @param stk_meshElem The STK element.
   * @param stk_nodeids Output vector of node IDs.
   */
  void getSTKNodeIdsForElement(stk::mesh::Entity & stk_meshElem, vector<stk::mesh::EntityId> & stk_nodeids);
  
  /**
   * @brief Retrieve sides belonging to a block by side name.
   * @param sideName Name of the side set.
   * @param blockname Block identifier.
   * @return Vector of STK side entities.
   */
  vector<stk::mesh::Entity> getMySTKSides(string & sideName, string & blockname);
  
  /**
   * @brief Retrieve nodes belonging to a node set.
   * @param nodeName Name of the node set.
   * @param blockID Block identifier.
   * @return Vector of STK node entities.
   */
  vector<stk::mesh::Entity> getMySTKNodes(string & nodeName, string & blockID);
  
  /**
   * @brief Retrieve elements attached to sides.
   * @param blockname Block identifier.
   * @param sideEntities Input side entities.
   * @param local_side_Ids Local side IDs.
   * @param side_output Output vector of elements.
   */
  void getSTKSideElements(string & blockname, vector<stk::mesh::Entity> & sideEntities,
                          vector<size_t> & local_side_Ids, vector<stk::mesh::Entity> & side_output);
  
  /**
   * @brief Retrieve vertices for STK elements.
   * @param side_output Elements of interest.
   * @param blockname Block identifier.
   * @param sidenodes Output data structure for node coordinates.
   */
  void getSTKElementVertices(vector<stk::mesh::Entity> & side_output, string & blockname, DRV & sidenodes);
  
  /**
   * @brief Get local element ID.
   * @param elem STK element.
   * @return Local index.
   */
  LO getSTKElementLocalId(stk::mesh::Entity & elem);
  
  /**
   * @brief Retrieve element vertices from a list of element local IDs.
   * @param local_grp Local element indices.
   * @param blockname Block identifier.
   * @param currnodes Output node coordinates.
   */
  void getSTKElementVertices(vector<size_t> & local_grp, string & blockname, DRV & currnodes);
  
  /**
   * @brief Get elements attached to nodes.
   * @param blockname Block identifier.
   * @param nodeEntities Node entities.
   * @param local_node_Ids Local node indices.
   * @param side_output Output element list.
   */
  void getSTKNodeElements(string & blockname, vector<stk::mesh::Entity> & nodeEntities,
                          vector<size_t> & local_node_Ids, vector<stk::mesh::Entity> & side_output);
  
  /**
   * @brief Retrieve nodes for selected elements.
   * @param block Block index.
   * @param elemIDs Element IDs.
   * @return Node coordinate data.
   */
  DRV getMyNodes(const size_t & block, vector<size_t> & elemIDs);
  
  /**
   * @brief Allocate internal mesh data structures.
   */
  void allocateMeshDataStructures();
  
  /**
   * @brief Clear maps used internally.
   */
  void purgeMaps();
  
  /**
   * @brief Compute element centers using Intrepid2 utilities.
   * @param nodes Node coordinate data.
   * @param reftopo Reference topology.
   * @return Array of element center coordinates.
   */
  View_Sc2 getElementCenters(DRV nodes, topo_RCP & reftopo);
  
  /**
   * @brief Get nodes for a specific element.
   * @param block Block index.
   * @param elemID Element ID.
   * @return Node coordinate data.
   */
  DRV getElemNodes(const int & block, const int & elemID);
  
  /**
   * @brief Split a delimited string into a list of entries.
   * @param list Input string.
   * @param delimiter Delimiter string.
   * @return Vector of extracted substrings.
   */
  vector<string> breakupList(const string & list, const string & delimiter);
  
  /**
   * @brief Read data from the Exodus mesh file.
   */
  void readExodusData();
  
  /**
   * @brief Purge unnecessary data before solve stage.
   */
  void purgeMemory();
  
  /**
   * @brief Purge all mesh data.
   */
  void purgeMesh();
  
  /**
   * @brief Generate a new realization of the microstructure.
   *
   * @param randSeed Seed for the random-number generator (may be modified).
   * @return View_Sc2 2D view containing the generated microstructure field values.
   *
   * @details
   *   Produces a randomized microstructure realization using the provided seed so
   *   that repeated calls with the same seed produce identical output.
   */
  View_Sc2 generateNewMicrostructure(int & randSeed);
  
  /** @name Public Data Members */
  ///@{
  Teuchos::RCP<Teuchos::ParameterList>  settings; ///< RCP to the main MrHyDE parameter list
  Teuchos::RCP<MpiComm> comm; ///< RCP to the MPIComm
  Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory; ///< RCP to the Panzer STK Mesh Factory
  
  bool have_mesh_data, compute_mesh_data, have_rotations, have_rotation_phi, have_quadrature_data; ///< Mesh state flags
  string shape, mesh_data_file_tag, mesh_data_pts_tag, mesh_data_tag; ///< Mesh metadata tags
  int dimension, verbosity; ///< Geometric dimension and verbosity level
  int num_nodes_per_elem, side_dim, num_sides, num_faces, num_seeds; ///< Mesh topology counts
  vector<int> random_seeds; ///< Random seeds for microstructure
  vector<topo_RCP> cell_topo, side_topo; ///< Cell and side topologies
  int meshmod_xvar, meshmod_yvar, meshmod_zvar; ///< Mesh modification axes
  bool meshmod_usesmoother, use_stk_mesh, use_simple_mesh; ///< Backend selection flags
  ScalarT meshmod_TOL, meshmod_center, meshmod_layer_size; ///< Mesh modification parameters
  
  vector<string> block_names, side_names, node_names, nfield_names, efield_names; ///< STK and field name lists
  int numResponses; ///< Number of responses
  std::default_random_engine generator; ///< RNG engine
  
  Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > meas; ///< Measurement vector
  vector<vector<ScalarT> > nfield_vals, efield_vals; ///< Field value arrays
  
  Teuchos::RCP<SimpleMeshManager<ScalarT>> simple_mesh; ///< Simple mesh manager
  ///@}
  
  /** @name Private Mesh Objects */
  ///@{
  Teuchos::RCP<panzer_stk::STK_Interface> stk_mesh; ///< STK Mesh interface
  Teuchos::RCP<panzer_stk::STK_Interface> stk_optimization_mesh; ///< Optimization STK Mesh interface
  Teuchos::RCP<MrHyDE_Debugger> debugger; ///< Debugger instance
  ///@}
  
  
}; // class MeshInterface

} // namespace MrHyDE

#endif // MRHYDE_MESHINTERFACE_H
