/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

// ============================================================
// ============================================================

Teuchos::RCP<panzer::ConnManager> MeshInterface::getSTKConnManager() {
  Teuchos::RCP<panzer::ConnManager> conn;
  if (use_stk_mesh) {
    conn = Teuchos::rcp(new panzer_stk::STKConnManager(stk_mesh));
  }
  return conn;
}

// ============================================================
// ============================================================

void MeshInterface::setSTKMesh(Teuchos::RCP<panzer_stk::STK_Interface> & new_mesh) {
  stk_mesh = new_mesh;
  stk_mesh->getElementBlockNames(block_names);
  stk_mesh->getSidesetNames(side_names);
  stk_mesh->getNodesetNames(node_names);
}

// ============================================================
// ============================================================

vector<stk::mesh::Entity> MeshInterface::getMySTKElements() {
  vector<stk::mesh::Entity> stk_meshElems;
  if (use_stk_mesh) {
    stk_mesh->getMyElements(stk_meshElems);
  }
  return stk_meshElems;
}
 
// ============================================================
// ============================================================

vector<stk::mesh::Entity> MeshInterface::getMySTKElements(string & blockID) {
  vector<stk::mesh::Entity> stk_meshElems;
  if (use_stk_mesh) {
    stk_mesh->getMyElements(blockID, stk_meshElems);
  }
  return stk_meshElems;
}
 
// ============================================================
// ============================================================

void MeshInterface::getSTKNodeIdsForElement(stk::mesh::Entity & stk_meshElem, vector<stk::mesh::EntityId> & stk_nodeids) {
  if (use_stk_mesh) {
    stk_mesh->getNodeIdsForElement(stk_meshElem, stk_nodeids);
  }
}

// ============================================================
// ============================================================

vector<stk::mesh::Entity> MeshInterface::getMySTKSides(string & sideName, string & blockname) {
  vector<stk::mesh::Entity> sideEntities;
  if (use_stk_mesh) {
    stk_mesh->getMySides(sideName, blockname, sideEntities);
  }
  return sideEntities;
}

// ============================================================
// ============================================================

vector<stk::mesh::Entity> MeshInterface::getMySTKNodes(string & nodeName, string & blockID) {
  vector<stk::mesh::Entity> nodeEntities;
  if (use_stk_mesh) {
    stk_mesh->getMyNodes(nodeName, blockID, nodeEntities);
  }
  return nodeEntities;
}

// ============================================================
// ============================================================

void MeshInterface::getSTKSideElements(string & blockname, vector<stk::mesh::Entity> & sideEntities,
                                       vector<size_t> & local_side_Ids, vector<stk::mesh::Entity> & side_output) {
  if (use_stk_mesh) {
    panzer_stk::workset_utils::getSideElements(*stk_mesh, blockname, sideEntities, local_side_Ids, side_output);
  }
}

// ============================================================
// ============================================================

void MeshInterface::getSTKElementVertices(vector<stk::mesh::Entity> & side_output, string & blockname, DRV & sidenodes) {
  if (use_stk_mesh) {
    stk_mesh->getElementVertices(side_output, blockname, sidenodes);
  }
}
  
// ============================================================
// ============================================================

LO MeshInterface::getSTKElementLocalId(stk::mesh::Entity & elem) {
  LO id = 0;
  if (use_stk_mesh) {
    id = stk_mesh->elementLocalId(elem);
  }
  return id;
}

// ============================================================
// ============================================================

void MeshInterface::getSTKElementVertices(vector<size_t> & local_grp, string & blockname, DRV & currnodes) {
  if (use_stk_mesh) {
    stk_mesh->getElementVertices(local_grp, blockname, currnodes);
  } else {
    currnodes = simple_mesh->getCellNodes(local_grp);
  }
}

// ============================================================
// ============================================================

void MeshInterface::getSTKNodeElements(string & blockname, vector<stk::mesh::Entity> & nodeEntities,
                                       vector<size_t> & local_node_Ids, vector<stk::mesh::Entity> & side_output) {
  if (use_stk_mesh) {
    panzer_stk::workset_utils::getNodeElements(*stk_mesh, blockname, nodeEntities, local_node_Ids, side_output);
  }
}
