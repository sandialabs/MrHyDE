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
//#include "Panzer_STK_PamgenReaderFactory.hpp"

#include "preferences.hpp"
#include "physicsInterface.hpp"
#include "cell.hpp"
#include "boundaryCell.hpp"
#include "multiscaleManager.hpp"

namespace MrHyDE {
  
  void static meshHelp(const string & details) {
    cout << "********** Help and Documentation for the Mesh Interface **********" << endl;
  }
  
  class meshInterface {
  public:
    
    meshInterface() {};
    
    //~meshInterface();
    
    meshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_, const Teuchos::RCP<MpiComm> & Commptr_);
    
    meshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                  const Teuchos::RCP<MpiComm> & Commptr_,
                  Teuchos::RCP<panzer_stk::STK_MeshFactory> & mesh_factory_,
                  Teuchos::RCP<panzer_stk::STK_Interface> & mesh_);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void finalize(Teuchos::RCP<physics> & phys);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    DRV perturbMesh(const int & b, DRV & blocknodes);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void setMeshData(vector<vector<Teuchos::RCP<cell> > > & cells);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void importMeshData(vector<vector<Teuchos::RCP<cell> > > & cells);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void computeMeshData(vector<vector<Teuchos::RCP<cell> > > & cells);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    DRV getElemNodes(const int & block, const int & elemID);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void remesh(const vector_RCP & u, vector<vector<Teuchos::RCP<cell> > > & cells);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void readMeshData(Teuchos::RCP<const LA_Map> & LA_overlapped_map,
                      vector<vector<Teuchos::RCP<cell> > > & cells);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void updateMeshData(const int & newrandseed, vector<vector<Teuchos::RCP<cell> > > & cells,
                        Teuchos::RCP<MultiScale> & multiscale_manager);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    // Public data members
    Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
    Teuchos::RCP<panzer_stk::STK_Interface> mesh, optimization_mesh;
    Teuchos::RCP<Teuchos::ParameterList>  settings;
    Teuchos::RCP<MpiComm> Commptr;
    bool have_mesh_data, compute_mesh_data, have_rotations, have_rotation_phi, have_multiple_data_files;
    string shape, mesh_data_file_tag, mesh_data_pts_tag, mesh_data_tag;
    int spaceDim, verbosity, number_mesh_data_files, milo_debug_level;
    int numNodesPerElem, sideDim, numSides, numFaces, numSeeds;
    vector<int> randomSeeds;
    vector<topo_RCP> cellTopo, sideTopo;
    int meshmod_xvar, meshmod_yvar, meshmod_zvar;
    bool meshmod_usesmoother;
    ScalarT meshmod_TOL, meshmod_center, meshmod_layer_size;
    
    // variables read in from an exodus mesh
    vector_RCP meas;
    vector<vector<ScalarT> > nfield_vals, efield_vals;
    vector<string> nfield_names, efield_names;
    int numResponses;
    
  };
  
}

#endif
