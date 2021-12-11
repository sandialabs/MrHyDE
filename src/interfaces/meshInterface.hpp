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

#ifndef MESHINTERFACE_H
#define MESHINTERFACE_H

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
#include "cell.hpp"
#include "data.hpp"
#include "boundaryCell.hpp"

namespace MrHyDE {
  /*
  void static meshHelp(const string & details) {
    cout << "********** Help and Documentation for the Mesh Interface **********" << endl;
  }
  */
  
  class MeshInterface {
    
  public:
    
    MeshInterface() {};
    
    ~MeshInterface() {};
    
    MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_, const Teuchos::RCP<MpiComm> & Commptr_);
    
    MeshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                  const Teuchos::RCP<MpiComm> & Commptr_,
                  Teuchos::RCP<panzer_stk::STK_MeshFactory> & mesh_factory_,
                  Teuchos::RCP<panzer_stk::STK_Interface> & mesh_);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void finalize(Teuchos::RCP<PhysicsInterface> & phys);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    DRV perturbMesh(const int & b, DRV & blocknodes);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void setMeshData(vector<vector<Teuchos::RCP<cell> > > & cells,
                     vector<vector<Teuchos::RCP<BoundaryCell>>> & bcells);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void importMeshData(vector<vector<Teuchos::RCP<cell> > > & cells,
                        vector<vector<Teuchos::RCP<BoundaryCell> > > & bcells);
    
    //void importMeshData(vector<vector<Teuchos::RCP<BoundaryCell> > > & bcells);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    //void computeMeshData(vector<vector<Teuchos::RCP<cell> > > & cells,
    //                     vector<vector<Teuchos::RCP<BoundaryCell> > > & bcells);
    
    //void computeMeshData(vector<vector<Teuchos::RCP<BoundaryCell> > > & bcells);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    View_Sc2 getElementCenters(DRV nodes, topo_RCP & reftopo);
      
    DRV getElemNodes(const int & block, const int & elemID);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    vector<string> breakupList(const string & list, const string & delimiter);

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void readMeshData();
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void purgeMemory();
    
    View_Sc2 generateNewMicrostructure(int & randSeed);

    void importNewMicrostructure(int & randSeed, View_Sc2 seeds,
                                 vector<vector<Teuchos::RCP<cell> > > & cells,
                                 vector<vector<Teuchos::RCP<BoundaryCell> > > & bcells);

    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    // Public data members
    Teuchos::RCP<Teuchos::ParameterList>  settings;
    Teuchos::RCP<MpiComm> Commptr;
    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
    Teuchos::RCP<panzer_stk::STK_Interface> stk_mesh, stk_optimization_mesh;
    
    bool have_mesh_data, compute_mesh_data, have_rotations, have_rotation_phi;
    string shape, mesh_data_file_tag, mesh_data_pts_tag, mesh_data_tag;
    int spaceDim, verbosity, debug_level;
    int numNodesPerElem, sideDim, numSides, numFaces, numSeeds;
    vector<int> randomSeeds;
    vector<topo_RCP> cellTopo, sideTopo;
    int meshmod_xvar, meshmod_yvar, meshmod_zvar;
    bool meshmod_usesmoother;
    ScalarT meshmod_TOL, meshmod_center, meshmod_layer_size;
    
    vector<string> block_names, nfield_names, efield_names;
    int numResponses;
    std::default_random_engine generator;
    
    Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > meas;
    vector<vector<ScalarT> > nfield_vals, efield_vals;
    
  };
  
}

#endif
