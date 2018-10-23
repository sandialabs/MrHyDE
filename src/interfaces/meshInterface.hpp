/***********************************************************************
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
#include "preferences.hpp"
#include "physicsInterface.hpp"
#include "cell.hpp"

void static meshHelp(const string & details) {
  cout << "********** Help and Documentation for the Mesh Interface **********" << endl;
}

class meshInterface {
  public:
  
  meshInterface() {};
  
  //~meshInterface();
  
  meshInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_, const Teuchos::RCP<Epetra_MpiComm> & Commptr_);
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void finalize(Teuchos::RCP<physics> & phys);
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV perturbMesh(const int & b, DRV & blocknodes);

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void createCells(Teuchos::RCP<physics> & phys, vector<vector<Teuchos::RCP<cell> > > & cells);
  
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
  
  // Public data members
  Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
  Teuchos::RCP<panzer_stk::STK_Interface> mesh;
  Teuchos::RCP<Teuchos::ParameterList>  settings;
  Teuchos::RCP<Epetra_MpiComm> Commptr;
  bool have_mesh_data, compute_mesh_data, have_rotations, have_rotation_phi, have_multiple_data_files;
  string shape, mesh_data_file_tag, mesh_data_pts_tag, mesh_data_tag;
  int spaceDim, verbosity, number_mesh_data_files;
  int numNodesPerElem, sideDim, numSides, numFaces, numSeeds;
  vector<int> randomSeeds;
  vector<topo_RCP> cellTopo, sideTopo;
};
#endif
