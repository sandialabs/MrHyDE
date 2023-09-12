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

#include "preferences.hpp"
#include "physicsInterface.hpp"
#include "group.hpp"
#include "data.hpp"
#include "boundaryGroup.hpp"

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
    
    void finalize(Teuchos::RCP<PhysicsInterface> & phys);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /**
     * This function perturbs the mesh nodes, as required for ALE methods.  Note that the basis functions and integration information will need to be recomputed if they have been stored.
     */
    
    DRV perturbMesh(const int & b, DRV & blocknodes);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /**
     * This function calls the functions associated with microstructure and sets the data in cells and boundary cells.
     */
    
    void setMeshData(vector<vector<Teuchos::RCP<Group> > > & groups,
                     vector<vector<Teuchos::RCP<BoundaryGroup>>> & boundary_groups);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    /**
     * This function eads in a microstructure from a file and sets the data in the cells and boundary cells.
     */
    
    void importMeshData(vector<vector<Teuchos::RCP<Group> > > & groups,
                        vector<vector<Teuchos::RCP<BoundaryGroup> > > & boundary_groups);
    
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
    
    /**
     * Generate a new realization of the microstructue.
     */
    
    View_Sc2 generateNewMicrostructure(int & randSeed);

    /**
     * Determine which grain contains each cell and boundary cell.
     */
    
    void importNewMicrostructure(int & randSeed, View_Sc2 seeds,
                                 vector<vector<Teuchos::RCP<Group> > > & groups,
                                 vector<vector<Teuchos::RCP<BoundaryGroup> > > & boundary_groups);

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    // Public data members
    Teuchos::RCP<Teuchos::ParameterList>  settings; ///< RCP to the main MrHyDE parameter list
    Teuchos::RCP<MpiComm> comm; ///< RCP to the MPIComm
    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory; ///< RCP to the Panzer STK Mesh Factory
    Teuchos::RCP<panzer_stk::STK_Interface> stk_mesh; ///< RCP to the Panzer STK Mesh
    Teuchos::RCP<panzer_stk::STK_Interface> stk_optimization_mesh; ///< RCP to the Panzer STK Mesh used to visualize an optmization history.
    
    bool have_mesh_data, compute_mesh_data, have_rotations, have_rotation_phi;
    string shape, mesh_data_file_tag, mesh_data_pts_tag, mesh_data_tag;
    int dimension, verbosity, debug_level;
    int num_nodes_per_elem, side_dim, num_sides, num_faces, num_seeds;
    vector<int> random_seeds;
    vector<topo_RCP> cell_topo, side_topo;
    int meshmod_xvar, meshmod_yvar, meshmod_zvar;
    bool meshmod_usesmoother;
    ScalarT meshmod_TOL, meshmod_center, meshmod_layer_size;
    
    vector<string> block_names, side_names, node_sets, nfield_names, efield_names;
    int numResponses;
    std::default_random_engine generator;
    
    Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > meas;
    vector<vector<ScalarT> > nfield_vals, efield_vals;
    
  };
  
}

#endif
