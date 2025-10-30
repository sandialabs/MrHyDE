/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/** \file   meshInterface.hpp
 \brief  Interface to the mesh objects, which includes a Panzer STK interface, and various functionality
 for modifying and extracting data from meshes.
 \author Created by T. Wildey
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
//#include "physicsInterface.hpp"
//#include "group.hpp"
#include "data.hpp"
//#include "boundaryGroup.hpp"

namespace MrHyDE {
  
  /** \class  MrHyDE::MeshInterface
   \brief  Interface to the Trilinos packages (panzer, stk) that handle the mesh objects.
   */
  
  class MeshInterface {
    
  public:
    
    /**
     * Default constructor
     */
    
    MeshInterface() {};
    
    /**
     * Default destructor
     */
    
    ~MeshInterface() {};
    
    /**
     * Standard constructor.  The mesh is still uninitialized after this function.  Need to call finalize to finish setting up the mesh.
     */
    
    MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_, const Teuchos::RCP<MpiComm> & Commptr_);
    
    /**
     * Constructor that reuses an existing MesfFactory and STK Interface.
     */
    
    MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                  const Teuchos::RCP<MpiComm> & Commptr_,
                  Teuchos::RCP<panzer_stk::STK_MeshFactory> & mesh_factory_,
                  Teuchos::RCP<panzer_stk::STK_Interface> & mesh_);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /**
     * This function uses the vriables defined in the physics interface to add fields to the mesh and complete the mesh construction.
     */
    
    void finalize(std::vector<std::vector<std::vector<string> > > varlist,
                  std::vector<std::vector<std::vector<string> > > vartypes,
                  std::vector<std::vector<std::vector<std::vector<string> > > > derivedlist);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /**
     * This function perturbs the mesh nodes, as required for ALE methods.  Note that the basis functions and integration information will need to be recomputed if they have been stored.
     */
    
    DRV perturbMesh(const int & b, DRV & blocknodes);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Access function (mostly) for the stk mesh
    ////////////////////////////////////////////////////////////////////////////////
    
    void setupExodusFile(const string & filename);

    // ========================================================================================
    // ========================================================================================
    
    void setupOptimizationExodusFile(const string & filename);
    
    // ========================================================================================
    // ========================================================================================
    
    void setSolutionFieldData(string var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT**,HostDevice> soln);

    // ========================================================================================
    // ========================================================================================
    
    void setCellFieldData(string var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT*,HostDevice> soln);

    // ========================================================================================
    // ========================================================================================
    
    void setOptimizationSolutionFieldData(string & var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT**,HostDevice> soln);

    // ========================================================================================
    // ========================================================================================
    
    void setOptimizationCellFieldData(string & var, string & blockID, vector<size_t> & myElements, Kokkos::View<ScalarT*,HostDevice> soln);

    // ========================================================================================
    // ========================================================================================
    
    void writeToExodus(const double & currenttime);

    // ========================================================================================
    // ========================================================================================
    
    void writeToExodus(const string & filename);
    
    // ========================================================================================
    // ========================================================================================
    
    void writeToOptimizationExodus(const double & currenttime);

    // ========================================================================================
    // ========================================================================================
    
    void writeToOptimizationExodus(const string & filename);
    
    // ========================================================================================
    // ========================================================================================
    
    vector<string> getBlockNames();

    // ========================================================================================
    // ========================================================================================
    
    vector<string> getSideNames();
    
    // ========================================================================================
    // ========================================================================================
    
    vector<string> getNodeNames();

    // ========================================================================================
    // ========================================================================================
    
    int getDimension();
    
    // ========================================================================================
    // ========================================================================================
    
    topo_RCP getCellTopology(string & blockID);
    
    // ========================================================================================
    // ========================================================================================
    
    Teuchos::RCP<panzer::ConnManager> getSTKConnManager();

    // ========================================================================================
    // ========================================================================================
    
    void setSTKMesh(Teuchos::RCP<panzer_stk::STK_Interface> & new_mesh);

    // ========================================================================================
    // ========================================================================================
    
    vector<stk::mesh::Entity> getMySTKElements();
  
    // ========================================================================================
    // ========================================================================================
    
    vector<stk::mesh::Entity> getMySTKElements(string & blockID);
  
    // ========================================================================================
    // ========================================================================================
    
    void getSTKNodeIdsForElement(stk::mesh::Entity & stk_meshElem, vector<stk::mesh::EntityId> & stk_nodeids);

    // ========================================================================================
    // ========================================================================================
    
    vector<stk::mesh::Entity> getMySTKSides(string & sideName, string & blockname);

    // ========================================================================================
    // ========================================================================================
    
    vector<stk::mesh::Entity> getMySTKNodes(string & nodeName, string & blockID);

    // ========================================================================================
    // ========================================================================================
    
    void getSTKSideElements(string & blockname, vector<stk::mesh::Entity> & sideEntities,
                            vector<size_t> & local_side_Ids, vector<stk::mesh::Entity> & side_output);

    // ========================================================================================
    // ========================================================================================
    
    void getSTKElementVertices(vector<stk::mesh::Entity> & side_output, string & blockname, DRV & sidenodes);
    
    // ========================================================================================
    // ========================================================================================
    
    LO getSTKElementLocalId(stk::mesh::Entity & elem);

    // ========================================================================================
    // ========================================================================================
    
    void getSTKElementVertices(vector<size_t> & local_grp, string & blockname, DRV & currnodes);

    // ========================================================================================
    // ========================================================================================
    
    void getSTKNodeElements(string & blockname, vector<stk::mesh::Entity> & nodeEntities,
                            vector<size_t> & local_node_Ids, vector<stk::mesh::Entity> & side_output);

    // ========================================================================================
    // ========================================================================================
    
    DRV getMyNodes(const size_t & block, vector<size_t> & elemIDs);

    // ========================================================================================
    // ========================================================================================
    
    void allocateMeshDataStructures();

    // ========================================================================================
    // ========================================================================================
    
    void purgeMaps();
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /**
     * This function uses Intrepid2 tools to determine the cell center for each of the elements in nodes.
     */
    
    View_Sc2 getElementCenters(DRV nodes, topo_RCP & reftopo);
      
    /**
     * This function returns the nodes associated with a particular element on a particular block.
     */
    
    DRV getElemNodes(const int & block, const int & elemID);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This function takes a string containing a list of entries separated by a delmiter, and returns a vector of strings containing the entries.
     *
     * This is similar in functionality to boost split
     */
    
    vector<string> breakupList(const string & list, const string & delimiter);

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /**
     * This function imports data from the Exodus mesh.
     */
    
    void readExodusData();
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /**
     * Removes any unnecessary objects before the cells allocate storing the basis functions and before the solve phase.
     * Only removes the Panzer_STK interface if visualization is not required.
     *
     * This function only gets called outside of this class in the driver/main.
     */
     
    void purgeMemory();
    
    // ========================================================================================
    // ========================================================================================
    
    void purgeMesh();
    
    /**
     * Generate a new realization of the microstructue.
     */
    
    View_Sc2 generateNewMicrostructure(int & randSeed);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    // Public data members
    Teuchos::RCP<Teuchos::ParameterList>  settings; ///< RCP to the main MrHyDE parameter list
    Teuchos::RCP<MpiComm> comm; ///< RCP to the MPIComm
    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory; ///< RCP to the Panzer STK Mesh Factory
    
    bool have_mesh_data, compute_mesh_data, have_rotations, have_rotation_phi, have_quadrature_data;
    string shape, mesh_data_file_tag, mesh_data_pts_tag, mesh_data_tag;
    int dimension, verbosity;
    int num_nodes_per_elem, side_dim, num_sides, num_faces, num_seeds;
    vector<int> random_seeds;
    vector<topo_RCP> cell_topo, side_topo;
    int meshmod_xvar, meshmod_yvar, meshmod_zvar;
    bool meshmod_usesmoother, use_stk_mesh, use_simple_mesh;
    ScalarT meshmod_TOL, meshmod_center, meshmod_layer_size;
    
    vector<string> block_names, side_names, node_names, nfield_names, efield_names;
    int numResponses;
    std::default_random_engine generator;
    
    Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > meas;
    vector<vector<ScalarT> > nfield_vals, efield_vals;
    
    Teuchos::RCP<SimpleMeshManager<ScalarT>> simple_mesh; ///< RCP to the SimpleMeshManager
    ///<
  private:
    Teuchos::RCP<panzer_stk::STK_Interface> stk_mesh; ///< RCP to the Panzer STK Mesh
    Teuchos::RCP<panzer_stk::STK_Interface> stk_optimization_mesh; ///< RCP to the Panzer STK Mesh used to visualize an optmization history.
    Teuchos::RCP<MrHyDE_Debugger> debugger;
  };
  
}

#endif
