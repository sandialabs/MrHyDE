/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "subgridTools2.hpp"
#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_LineMeshFactory.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SquareTriMeshFactory.hpp"
#include "Panzer_STK_CubeHexMeshFactory.hpp"
#include "Panzer_STK_CubeTetMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"

using namespace MrHyDE;

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

SubGridTools2::SubGridTools2(const Teuchos::RCP<MpiComm> & local_comm, const string & shape,
                           const string & subshape, const DRV nodes,
                           std::string & mesh_type, std::string & mesh_file) :
local_comm_(local_comm), shape_(shape), subshape_(subshape),
mesh_type_(mesh_type), mesh_file_(mesh_file) {
  
  nodes_ = Kokkos::View<ScalarT**,HostDevice>("nodes on host",nodes.extent(0),nodes.extent(1));
  auto tmp_nodes = Kokkos::create_mirror_view(nodes);
  Kokkos::deep_copy(tmp_nodes,nodes);
  Kokkos::deep_copy(nodes_,tmp_nodes);
  
  dimension_ = nodes_.extent(1);
  
  if (dimension_ == 1) {
    num_macro_sides_ = 2;
  }
  else if (dimension_ == 2) {
    if (shape_ == "tri") {
      num_macro_sides_ = 3;
    }
    if (shape_ == "quad") {
      num_macro_sides_ = 4;
    }
  }
  if (dimension_ == 3) {
    if (shape_ == "tet") {
      num_macro_sides_ = 4;
    }
    if (shape_ == "hex") {
      num_macro_sides_ = 6;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Given the coarse grid nodes and shape_, define the subgrid nodes, connectivity, and sideinfo
//////////////////////////////////////////////////////////////////////////////////////

void SubGridTools2::createSubMesh(const int & numrefine) {
  
  if (mesh_type_ == "Exodus" || mesh_type_ == "panzer") {
    
    if (mesh_type_ == "Exodus") {
      // Read in the mesh
      Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::rcp(new Teuchos::ParameterList);
      Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory());
      pl->set("File Name",mesh_file_);
      mesh_factory->setParameterList(pl);
      ref_mesh_ = mesh_factory->buildUncommitedMesh(*(local_comm_->getRawMpiComm()));
      mesh_factory->completeMeshConstruction(*ref_mesh_,*(local_comm_->getRawMpiComm()));
    }
    else {
      
      if (shape_ == "tri" || shape_ == "tet") {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the panzer subgrid meshes cannot be used for triangles or tetrahedrons.");
      }
      
      Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::rcp(new Teuchos::ParameterList);
      
      int numEPerD = static_cast<int>(std::pow(2,numrefine));
      pl->set("X Blocks", 1);
      pl->set("X Elements", numEPerD);
      pl->set("X0", -1.0);
      pl->set("Xf", 1.0);
      if (dimension_ > 1) {
        pl->set("Y Blocks", 1);
        pl->set("Y Elements", numEPerD);
        pl->set("Y0", -1.0);
        pl->set("Yf", 1.0);
        pl->set("X Procs", 1);
        pl->set("Y Procs", 1);
      }
      if (dimension_ > 2) {
        pl->set("Z Blocks", 1);
        pl->set("Z Elements", numEPerD);
        pl->set("Z0", -1.0);
        pl->set("Zf", 1.0);
        pl->set("Z Procs", 1);
      }
      
      Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
      if (dimension_ == 1)
        mesh_factory = Teuchos::rcp(new panzer_stk::LineMeshFactory());
      else if (dimension_ == 2) {
        if (subshape_ == "quad") {
          mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory());
        }
        if (subshape_ == "tri") {
          mesh_factory = Teuchos::rcp(new panzer_stk::SquareTriMeshFactory());
        }
      }
      else if (dimension_ == 3) {
        if (subshape_ == "hex") {
          mesh_factory = Teuchos::rcp(new panzer_stk::CubeHexMeshFactory());
        }
        if (subshape_ == "tet") {
          mesh_factory = Teuchos::rcp(new panzer_stk::CubeTetMeshFactory());
        }
      }
      mesh_factory->setParameterList(pl);
      ref_mesh_ = mesh_factory->buildUncommitedMesh(*(local_comm_->getRawMpiComm()));
      mesh_factory->completeMeshConstruction(*ref_mesh_,*(local_comm_->getRawMpiComm()));
    }
    
    std::vector<string> blocknames;
    ref_mesh_->getElementBlockNames(blocknames);
    
    topo_RCP cellTopo = ref_mesh_->getCellTopology(blocknames[0]);
    size_t numNodesPerElem = cellTopo->getNodeCount();
    size_t numSides = 0;
    if (dimension_ == 1) {
      numSides = 2;
    }
    else if (dimension_ == 2) {
      numSides = cellTopo->getSideCount();
    }
    else if (dimension_ == 3) {
      numSides = cellTopo->getFaceCount();
    }
    
    std::vector<stk::mesh::Entity> ref_elements;
    ref_mesh_->getMyElements(ref_elements);
    
    // TMW: may fail on device
    DRV vertices_dev("element vertices",ref_elements.size(), numNodesPerElem, dimension_);
    ref_mesh_->getElementVertices_FromCoordsNoResize(ref_elements, vertices_dev);
    auto vertices = Kokkos::create_mirror_view(vertices_dev);
    Kokkos::deep_copy(vertices,vertices_dev);
    
    // extract the nodes
    size_t numTotalNodes = ref_mesh_->getMaxEntityId(0);
    
    subnodes_list_ = DRV("DRV of subgrid nodes on ref elem", numTotalNodes, dimension_);
    auto subnodes_host = Kokkos::create_mirror_view(subnodes_list_);
    
    vector<bool> beenAdded(numTotalNodes,false);
    
    // Extract the connectivity
    for (size_t elem=0; elem<ref_elements.size(); elem++) {
      std::vector< stk::mesh::EntityId > nodeIds;
      ref_mesh_->getNodeIdsForElement(ref_elements[elem],nodeIds);
      vector<GO> conn;
      for (size_t i=0; i<nodeIds.size(); i++) {
        conn.push_back(nodeIds[i]-1);
        if (!beenAdded[nodeIds[i]-1]) {
          for (int s=0; s<dimension_; s++) {
            subnodes_host(nodeIds[i]-1,s) = vertices(elem,i,s);
          }
          beenAdded[nodeIds[i]-1] = true;
        }
      }
      subconnectivity_.push_back(conn);
      
      Kokkos::View<bool*,HostDevice> newsidemap("new side map",numSides);
      subsidemap_.push_back(newsidemap);
      
    }
    
    Kokkos::deep_copy(subnodes_list_, subnodes_host);
    
    vector<string> sideSets;
    ref_mesh_->getSidesetNames(sideSets);
    
    if (dimension_ == 2) {
      if (sideSets.size() < 4) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the subgrid mesh requires at least 4 sidesets.");
      }
    }
    else if (dimension_ == 3) {
      if (sideSets.size() < 6) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the subgrid mesh requires at least 6 sidesets.");
      }
    }
    for( size_t side=0; side<sideSets.size(); side++ ) {
      string sideName = sideSets[side];
      
      vector<stk::mesh::Entity> sideEntities;
      ref_mesh_->getMySides(sideName, blocknames[0], sideEntities);
      
      vector<size_t>             local_side_Ids;
      vector<stk::mesh::Entity>  side_output;
      vector<size_t>             local_elem_Ids;
      panzer_stk::workset_utils::getSideElements(*ref_mesh_, blocknames[0], sideEntities, local_side_Ids, side_output);
      
      for( size_t i=0; i<side_output.size(); i++ ) {
        local_elem_Ids.push_back(ref_mesh_->elementLocalId(side_output[i]));
        size_t localid = local_elem_Ids[i];
        subsidemap_[localid](local_side_Ids[i]) = true;
        //subsidemap_[localid](local_side_Ids[i], 1) = local_side_Ids[i];//side;
        //subsidemap_[localid](side, 1) = local_side_Ids[i];
      }
    }
    
  }
  else if (mesh_type_ == "inline") {
    if (subshape_ == shape_) {
      vector<GO> newconn;
      for (size_type i=0; i<nodes_.extent(0); i++) {
        vector<ScalarT> newnode;
        for (int s=0; s<dimension_; s++) {
          newnode.push_back(nodes_(i,s));
        }
        subnodes_.push_back(newnode);
        newconn.push_back(i);
      }
      subconnectivity_.push_back(newconn);
      
      size_t numSides = 0;
      if (dimension_ == 1) {
        numSides = 2;
      }
      else if (dimension_ == 2) {
        if (subshape_ == "tri") {
          numSides = 3;
        }
        else if (subshape_ == "quad"){
          numSides = 4;
        }
      }
      else if (dimension_ == 3) {
        if (subshape_ == "tet") {
          numSides = 4;
        }
        else if (subshape_ == "hex"){
          numSides = 6;
        }
      }
      
      Kokkos::View<bool*,HostDevice> newsidemap("newsidemap",numSides);
      for (size_t s=0; s<numSides; s++) {
        newsidemap(s) = true;
        //newsidemap(s,1) = s;
      }
      subsidemap_.push_back(newsidemap);
      
    }
    else {
      if (dimension_ == 1) {
        // output an error message
      }
      else if (dimension_ == 2) {
        if (shape_ == "quad" && subshape_ == "tri") {
          
          for (size_type i=0; i<nodes_.extent(0); i++) {
            vector<ScalarT> newnode;
            for (int s=0; s<dimension_; s++) {
              newnode.push_back(nodes_(i,s));
            }
            subnodes_.push_back(newnode);
          }
          vector<ScalarT> midnode;
          midnode.push_back(0.25*(nodes_(0,0)+nodes_(1,0)+nodes_(2,0)+nodes_(3,0)));
          midnode.push_back(0.25*(nodes_(0,1)+nodes_(1,1)+nodes_(2,1)+nodes_(3,1)));
          subnodes_.push_back(midnode);
          
          vector<GO> newconn0 = {0,1,4};
          subconnectivity_.push_back(newconn0);
          
          vector<GO> newconn1 = {1,2,4};
          subconnectivity_.push_back(newconn1);
          
          vector<GO> newconn2 = {2,3,4};
          subconnectivity_.push_back(newconn2);
          
          vector<GO> newconn3 = {3,0,4};
          subconnectivity_.push_back(newconn3);
          
          Kokkos::View<bool*,HostDevice> newsidemap0("newsi",3);
          Kokkos::View<bool*,HostDevice> newsidemap1("newsi",3);
          Kokkos::View<bool*,HostDevice> newsidemap2("newsi",3);
          Kokkos::View<bool*,HostDevice> newsidemap3("newsi",3);
          
          newsidemap0(0) = true;
          newsidemap1(0) = true;
          newsidemap2(0) = true;
          newsidemap3(0) = true;
          
          subsidemap_.push_back(newsidemap0);
          subsidemap_.push_back(newsidemap1);
          subsidemap_.push_back(newsidemap2);
          subsidemap_.push_back(newsidemap3);
          
        }
        else if (shape_ == "tri" && subshape_ == "quad") {
          
          for (size_type i=0; i<nodes_.extent(0); i++) {
            vector<ScalarT> newnode;
            
            for (int s=0; s<dimension_; s++) {
              newnode.push_back(nodes_(i,s));
            }
            subnodes_.push_back(newnode);
            
          }
          
          vector<ScalarT> center, mid01, mid12, mid02;
          center.push_back(1.0/3.0*(nodes_(0,0)+nodes_(1,0)+nodes_(2,0)));
          center.push_back(1.0/3.0*(nodes_(0,1)+nodes_(1,1)+nodes_(2,1)));
          mid01.push_back(0.5*(nodes_(0,0)+nodes_(1,0)));
          mid01.push_back(0.5*(nodes_(0,1)+nodes_(1,1)));
          mid12.push_back(0.5*(nodes_(1,0)+nodes_(2,0)));
          mid12.push_back(0.5*(nodes_(1,1)+nodes_(2,1)));
          mid02.push_back(0.5*(nodes_(0,0)+nodes_(2,0)));
          mid02.push_back(0.5*(nodes_(0,1)+nodes_(2,1)));
          subnodes_.push_back(center);
          subnodes_.push_back(mid01);
          subnodes_.push_back(mid12);
          subnodes_.push_back(mid02);
          
          vector<GO> newconn0 = {0,4,3,6};
          subconnectivity_.push_back(newconn0);
          
          vector<GO> newconn1 = {1,5,3,4};
          subconnectivity_.push_back(newconn1);
          
          vector<GO> newconn2 = {2,6,3,5};
          subconnectivity_.push_back(newconn2);
          
          Kokkos::View<bool*,HostDevice> newsidemap0("newsi",4);
          Kokkos::View<bool*,HostDevice> newsidemap1("newsi",4);
          Kokkos::View<bool*,HostDevice> newsidemap2("newsi",4);
          
          newsidemap0(0) = true;
          newsidemap0(3) = true;
          newsidemap1(0) = true;
          newsidemap1(3) = true;
          newsidemap2(0) = true;
          newsidemap2(3) = true;
          
          subsidemap_.push_back(newsidemap0);
          subsidemap_.push_back(newsidemap1);
          subsidemap_.push_back(newsidemap2);
          
        }
        else {
          // output an error message
        }
      }
      else if (dimension_ == 3) {
        // conversions from het to tet or tet to hex are not added yet
      }
      
    }
    
    // Recursively refine the elements
    for (int r=0; r<numrefine; r++) {
      size_t numelem = subconnectivity_.size();
      for (size_t e=0; e<numelem; e++) {
        refineSubCell(e); // adds new nodes and new elements (does not delete old elements)
      }
      subconnectivity_.erase(subconnectivity_.begin(), subconnectivity_.begin()+numelem);
      subsidemap_.erase(subsidemap_.begin(), subsidemap_.begin()+numelem);
    }
    
    // Create the list of nodes
    subnodes_list_ = DRV("DRV of subgrid nodes on ref elem",subnodes_.size(), subnodes_[0].size());
    auto subnodes_host = Kokkos::create_mirror_view(subnodes_list_);
    for (size_t node=0; node<subnodes_.size(); node++) {
      for (size_t dim=0; dim<subnodes_[0].size(); dim++) {
        subnodes_host(node,dim) = subnodes_[node][dim];
      }
    }
    Kokkos::deep_copy(subnodes_list_,subnodes_host);
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized subgrid mesh type: " + mesh_type_);
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Uniformly refine an element
//////////////////////////////////////////////////////////////////////////////////////

void SubGridTools2::refineSubCell(const int & e) {
  
  if (dimension_ == 1) {
    vector<ScalarT> node0 = subnodes_[subconnectivity_[e][0]];
    vector<ScalarT> node1 = subnodes_[subconnectivity_[e][1]];
    
    ScalarT midx = 0.5*(node0[0]+node1[0]);
    vector<ScalarT> mid(1,midx);
    
    subnodes_.push_back(mid);
    
    vector<GO> newelem0, newelem1;
    newelem0.push_back(subconnectivity_[e][0]);
    newelem0.push_back(subnodes_.size());
    newelem1.push_back(subnodes_.size());
    newelem1.push_back(subconnectivity_[e][1]);
    
    subconnectivity_.push_back(newelem0);
    subconnectivity_.push_back(newelem1);
    
    Kokkos::View<bool*,HostDevice> oldmap = subsidemap_[e];
    Kokkos::View<bool*,HostDevice> newsm0("newsi",2);
    Kokkos::View<bool*,HostDevice> newsm1("newsi",2);
    Kokkos::deep_copy(newsm0,oldmap);
    Kokkos::deep_copy(newsm1,oldmap);
    
    newsm0(1) = false;
    newsm1(0) = false;
    
    subsidemap_.push_back(newsm0);
    subsidemap_.push_back(newsm1);
    
  }
  if (dimension_ == 2) {
    if (subshape_ == "tri") {
      // Extract the existing nodes
      vector<ScalarT> node0 = subnodes_[subconnectivity_[e][0]];
      vector<ScalarT> node1 = subnodes_[subconnectivity_[e][1]];
      vector<ScalarT> node2 = subnodes_[subconnectivity_[e][2]];
      
      // Compute the candidate new nodes
      vector<ScalarT> mid01, mid12, mid02;
      mid01.push_back(0.5*(node0[0]+node1[0]));
      mid01.push_back(0.5*(node0[1]+node1[1]));
      mid12.push_back(0.5*(node1[0]+node2[0]));
      mid12.push_back(0.5*(node1[1]+node2[1]));
      mid02.push_back(0.5*(node0[0]+node2[0]));
      mid02.push_back(0.5*(node0[1]+node2[1]));
      
      // Check if these nodes have been added and add if not
      ScalarT tol=1.0e-10;
      int mid01_ind, mid12_ind, mid02_ind;
      bool found;
      
      found = checkExistingSubNodes(mid01,tol,mid01_ind);
      if (!found) {
        subnodes_.push_back(mid01);
        mid01_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid12,tol,mid12_ind);
      if (!found) {
        subnodes_.push_back(mid12);
        mid12_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid02,tol,mid02_ind);
      if (!found) {
        subnodes_.push_back(mid02);
        mid02_ind = subnodes_.size()-1;
      }
      
      // Define the new elements (appended to end of list)
      vector<GO> elem0, elem1, elem2, elem3;
      
      elem0.push_back(subconnectivity_[e][0]);
      elem0.push_back(mid01_ind);
      elem0.push_back(mid02_ind);
      subconnectivity_.push_back(elem0);
      
      elem1.push_back(subconnectivity_[e][1]);
      elem1.push_back(mid01_ind);
      elem1.push_back(mid12_ind);
      subconnectivity_.push_back(elem1);
      
      elem2.push_back(subconnectivity_[e][2]);
      elem2.push_back(mid02_ind);
      elem2.push_back(mid12_ind);
      subconnectivity_.push_back(elem2);
      
      elem3.push_back(mid01_ind);
      elem3.push_back(mid12_ind);
      elem3.push_back(mid02_ind);
      subconnectivity_.push_back(elem3);
      
      
      Kokkos::View<bool*,HostDevice> oldmap = subsidemap_[e];
      Kokkos::View<bool*,HostDevice> newsm0("newsi",3);
      Kokkos::View<bool*,HostDevice> newsm1("newsi",3);
      Kokkos::View<bool*,HostDevice> newsm2("newsi",3);
      Kokkos::View<bool*,HostDevice> newsm3("newsi",3);
      Kokkos::deep_copy(newsm0,oldmap);
      Kokkos::deep_copy(newsm1,oldmap);
      Kokkos::deep_copy(newsm2,oldmap);
      Kokkos::deep_copy(newsm3,oldmap);
      
      newsm0(1) = false;
      newsm1(1) = false;
      newsm2(1) = false;
      newsm3(0) = false;
      newsm3(1) = false;
      newsm3(2) = false;
      
      subsidemap_.push_back(newsm0);
      subsidemap_.push_back(newsm1);
      subsidemap_.push_back(newsm2);
      subsidemap_.push_back(newsm3);
      
      
    }
    else if (subshape_ == "quad") {
      // Extract the existing nodes
      vector<ScalarT> node0 = subnodes_[subconnectivity_[e][0]];
      vector<ScalarT> node1 = subnodes_[subconnectivity_[e][1]];
      vector<ScalarT> node2 = subnodes_[subconnectivity_[e][2]];
      vector<ScalarT> node3 = subnodes_[subconnectivity_[e][3]];
      
      // Compute the candidate new nodes
      vector<ScalarT> center, mid01, mid12, mid23, mid03;
      center.push_back(0.25*(node0[0]+node1[0]+node2[0]+node3[0]));
      center.push_back(0.25*(node0[1]+node1[1]+node2[1]+node3[1]));
      mid01.push_back(0.5*(node0[0]+node1[0]));
      mid01.push_back(0.5*(node0[1]+node1[1]));
      mid12.push_back(0.5*(node1[0]+node2[0]));
      mid12.push_back(0.5*(node1[1]+node2[1]));
      mid23.push_back(0.5*(node2[0]+node3[0]));
      mid23.push_back(0.5*(node2[1]+node3[1]));
      mid03.push_back(0.5*(node0[0]+node3[0]));
      mid03.push_back(0.5*(node0[1]+node3[1]));
      
      // Check if these nodes have been added and add if not
      ScalarT tol=1.0e-6;
      int center_ind, mid01_ind, mid12_ind, mid23_ind, mid03_ind;
      bool found;
      
      found = checkExistingSubNodes(center,tol,center_ind);
      if (!found) {
        subnodes_.push_back(center);
        center_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid01,tol,mid01_ind);
      if (!found) {
        subnodes_.push_back(mid01);
        mid01_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid12,tol,mid12_ind);
      if (!found) {
        subnodes_.push_back(mid12);
        mid12_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid23,tol,mid23_ind);
      if (!found) {
        subnodes_.push_back(mid23);
        mid23_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid03,tol,mid03_ind);
      if (!found) {
        subnodes_.push_back(mid03);
        mid03_ind = subnodes_.size()-1;
      }
      
      // Define the new elements (appended to end of list)
      vector<GO> elem0, elem1, elem2, elem3;
      
      elem0.push_back(subconnectivity_[e][0]);
      elem0.push_back(mid01_ind);
      elem0.push_back(center_ind);
      elem0.push_back(mid03_ind);
      subconnectivity_.push_back(elem0);
      
      elem1.push_back(mid01_ind);
      elem1.push_back(subconnectivity_[e][1]);
      elem1.push_back(mid12_ind);
      elem1.push_back(center_ind);
      subconnectivity_.push_back(elem1);
      
      elem2.push_back(center_ind);
      elem2.push_back(mid12_ind);
      elem2.push_back(subconnectivity_[e][2]);
      elem2.push_back(mid23_ind);
      subconnectivity_.push_back(elem2);
      
      elem3.push_back(mid03_ind);
      elem3.push_back(center_ind);
      elem3.push_back(mid23_ind);
      elem3.push_back(subconnectivity_[e][3]);
      subconnectivity_.push_back(elem3);
      
      Kokkos::View<bool*,HostDevice> oldmap = subsidemap_[e];
      Kokkos::View<bool*,HostDevice> newsm0("newsm",4);
      Kokkos::View<bool*,HostDevice> newsm1("newsm",4);
      Kokkos::View<bool*,HostDevice> newsm2("newsm",4);
      Kokkos::View<bool*,HostDevice> newsm3("newsm",4);
      Kokkos::deep_copy(newsm0,oldmap);
      Kokkos::deep_copy(newsm1,oldmap);
      Kokkos::deep_copy(newsm2,oldmap);
      Kokkos::deep_copy(newsm3,oldmap);
      
      newsm0(1) = false;
      newsm0(2) = false;
      
      newsm1(2) = false;
      newsm1(3) = false;
      
      newsm2(0) = false;
      newsm2(3) = false;
      
      newsm3(0) = false;
      newsm3(1) = false;
      
      subsidemap_.push_back(newsm0);
      subsidemap_.push_back(newsm1);
      subsidemap_.push_back(newsm2);
      subsidemap_.push_back(newsm3);
      
    }
    else {
      // add error
    }
  }
  if (dimension_ == 3) {
    if (subshape_ == "tet") {
      // Extract the existing nodes
      vector<ScalarT> node0 = subnodes_[subconnectivity_[e][0]];
      vector<ScalarT> node1 = subnodes_[subconnectivity_[e][1]];
      vector<ScalarT> node2 = subnodes_[subconnectivity_[e][2]];
      vector<ScalarT> node3 = subnodes_[subconnectivity_[e][3]];
      
      // Compute the candidate new nodes
      vector<ScalarT> mid01, mid12, mid02, mid03, mid13, mid23;
      mid01.push_back(0.5*(node0[0]+node1[0]));
      mid01.push_back(0.5*(node0[1]+node1[1]));
      mid01.push_back(0.5*(node0[2]+node1[2]));
      mid12.push_back(0.5*(node1[0]+node2[0]));
      mid12.push_back(0.5*(node1[1]+node2[1]));
      mid12.push_back(0.5*(node1[2]+node2[2]));
      mid02.push_back(0.5*(node0[0]+node2[0]));
      mid02.push_back(0.5*(node0[1]+node2[1]));
      mid02.push_back(0.5*(node0[2]+node2[2]));
      mid03.push_back(0.5*(node0[0]+node3[0]));
      mid03.push_back(0.5*(node0[1]+node3[1]));
      mid03.push_back(0.5*(node0[2]+node3[2]));
      mid13.push_back(0.5*(node1[0]+node3[0]));
      mid13.push_back(0.5*(node1[1]+node3[1]));
      mid13.push_back(0.5*(node1[2]+node3[2]));
      mid23.push_back(0.5*(node2[0]+node3[0]));
      mid23.push_back(0.5*(node2[1]+node3[1]));
      mid23.push_back(0.5*(node2[2]+node3[2]));
      
      // Check if these nodes have been added and add if not
      ScalarT tol=1.0e-10;
      int mid01_ind, mid12_ind, mid02_ind;
      int mid03_ind, mid13_ind, mid23_ind;
      bool found;
      
      found = checkExistingSubNodes(mid01,tol,mid01_ind);
      if (!found) {
        subnodes_.push_back(mid01);
        mid01_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid12,tol,mid12_ind);
      if (!found) {
        subnodes_.push_back(mid12);
        mid12_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid02,tol,mid02_ind);
      if (!found) {
        subnodes_.push_back(mid02);
        mid02_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid03,tol,mid03_ind);
      if (!found) {
        subnodes_.push_back(mid03);
        mid03_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid13,tol,mid13_ind);
      if (!found) {
        subnodes_.push_back(mid13);
        mid13_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid23,tol,mid23_ind);
      if (!found) {
        subnodes_.push_back(mid23);
        mid23_ind = subnodes_.size()-1;
      }
      
      // Define the new elements (appended to end of list)
      vector<GO> elem0, elem1, elem2, elem3, elem4, elem5, elem6, elem7;
      
      elem0.push_back(subconnectivity_[e][0]);
      elem0.push_back(mid01_ind);
      elem0.push_back(mid02_ind);
      elem0.push_back(mid03_ind);
      subconnectivity_.push_back(elem0);
      
      elem1.push_back(subconnectivity_[e][1]);
      elem1.push_back(mid12_ind);
      elem1.push_back(mid01_ind);
      elem1.push_back(mid13_ind);
      subconnectivity_.push_back(elem1);
      
      elem2.push_back(subconnectivity_[e][2]);
      elem2.push_back(mid02_ind);
      elem2.push_back(mid12_ind);
      elem2.push_back(mid23_ind);
      subconnectivity_.push_back(elem2);
      
      elem3.push_back(mid03_ind);
      elem3.push_back(mid13_ind);
      elem3.push_back(mid23_ind);
      elem3.push_back(subconnectivity_[e][3]);
      subconnectivity_.push_back(elem3);
      
      elem4.push_back(mid01_ind);
      elem4.push_back(mid12_ind);
      elem4.push_back(mid02_ind);
      elem4.push_back(mid03_ind);
      subconnectivity_.push_back(elem4);
      
      elem5.push_back(mid03_ind);
      elem5.push_back(mid13_ind);
      elem5.push_back(mid23_ind);
      elem5.push_back(mid12_ind);
      subconnectivity_.push_back(elem5);
      
      elem6.push_back(mid03_ind);
      elem6.push_back(mid13_ind);
      elem6.push_back(mid23_ind);
      elem6.push_back(mid12_ind);
      subconnectivity_.push_back(elem6);
      
      elem7.push_back(mid03_ind);
      elem7.push_back(mid13_ind);
      elem7.push_back(mid23_ind);
      elem7.push_back(mid12_ind);
      subconnectivity_.push_back(elem7);
      
      Kokkos::View<bool*,HostDevice> oldmap = subsidemap_[e];
      Kokkos::View<bool*,HostDevice> newsm0("newsi",4);
      Kokkos::View<bool*,HostDevice> newsm1("newsi",4);
      Kokkos::View<bool*,HostDevice> newsm2("newsi",4);
      Kokkos::View<bool*,HostDevice> newsm3("newsi",4);
      Kokkos::View<bool*,HostDevice> newsm4("newsi",4);
      Kokkos::View<bool*,HostDevice> newsm5("newsi",4);
      Kokkos::View<bool*,HostDevice> newsm6("newsi",4);
      Kokkos::View<bool*,HostDevice> newsm7("newsi",4);
      
      Kokkos::deep_copy(newsm0,oldmap);
      Kokkos::deep_copy(newsm1,oldmap);
      Kokkos::deep_copy(newsm2,oldmap);
      Kokkos::deep_copy(newsm3,oldmap);
      Kokkos::deep_copy(newsm4,oldmap);
      Kokkos::deep_copy(newsm5,oldmap);
      Kokkos::deep_copy(newsm6,oldmap);
      Kokkos::deep_copy(newsm7,oldmap);
      
      newsm0(2) = false;
      newsm1(2) = false;
      newsm2(2) = false;
      newsm3(0) = false;
      
      newsm4(1) = false;
      newsm4(2) = false;
      newsm4(3) = false;
      newsm5(2) = false;
      
      subsidemap_.push_back(newsm0);
      subsidemap_.push_back(newsm1);
      subsidemap_.push_back(newsm2);
      subsidemap_.push_back(newsm3);
      subsidemap_.push_back(newsm4);
      subsidemap_.push_back(newsm5);
      subsidemap_.push_back(newsm6);
      subsidemap_.push_back(newsm7);
      
    }
    else if (subshape_ == "hex") {
      // Extract the existing nodes
      vector<ScalarT> node0 = subnodes_[subconnectivity_[e][0]];
      vector<ScalarT> node1 = subnodes_[subconnectivity_[e][1]];
      vector<ScalarT> node2 = subnodes_[subconnectivity_[e][2]];
      vector<ScalarT> node3 = subnodes_[subconnectivity_[e][3]];
      vector<ScalarT> node4 = subnodes_[subconnectivity_[e][4]];
      vector<ScalarT> node5 = subnodes_[subconnectivity_[e][5]];
      vector<ScalarT> node6 = subnodes_[subconnectivity_[e][6]];
      vector<ScalarT> node7 = subnodes_[subconnectivity_[e][7]];
      
      // Compute the candidate new nodes
      vector<ScalarT> mid0123, mid01, mid12, mid23, mid03;
      vector<ScalarT> center, mid04, mid15, mid26, mid37, mid0145, mid1256, mid2367, mid0347;
      vector<ScalarT> mid4567, mid45, mid56, mid67, mid47;
      
      center.push_back(0.125*(node0[0]+node1[0]+node2[0]+node3[0]+node4[0]+node5[0]+node6[0]+node7[0]));
      center.push_back(0.125*(node0[1]+node1[1]+node2[1]+node3[1]+node4[1]+node5[1]+node6[1]+node7[1]));
      center.push_back(0.125*(node0[2]+node1[2]+node2[2]+node3[2]+node4[2]+node5[2]+node6[2]+node7[2]));
      
      mid01.push_back(0.5*(node0[0]+node1[0]));
      mid01.push_back(0.5*(node0[1]+node1[1]));
      mid01.push_back(0.5*(node0[2]+node1[2]));
      mid12.push_back(0.5*(node1[0]+node2[0]));
      mid12.push_back(0.5*(node1[1]+node2[1]));
      mid12.push_back(0.5*(node1[2]+node2[2]));
      mid23.push_back(0.5*(node2[0]+node3[0]));
      mid23.push_back(0.5*(node2[1]+node3[1]));
      mid23.push_back(0.5*(node2[2]+node3[2]));
      mid03.push_back(0.5*(node0[0]+node3[0]));
      mid03.push_back(0.5*(node0[1]+node3[1]));
      mid03.push_back(0.5*(node0[2]+node3[2]));
      mid04.push_back(0.5*(node0[0]+node4[0]));
      mid04.push_back(0.5*(node0[1]+node4[1]));
      mid04.push_back(0.5*(node0[2]+node4[2]));
      mid15.push_back(0.5*(node1[0]+node5[0]));
      mid15.push_back(0.5*(node1[1]+node5[1]));
      mid15.push_back(0.5*(node1[2]+node5[2]));
      mid26.push_back(0.5*(node2[0]+node6[0]));
      mid26.push_back(0.5*(node2[1]+node6[1]));
      mid26.push_back(0.5*(node2[2]+node6[2]));
      mid37.push_back(0.5*(node3[0]+node7[0]));
      mid37.push_back(0.5*(node3[1]+node7[1]));
      mid37.push_back(0.5*(node3[2]+node7[2]));
      mid45.push_back(0.5*(node4[0]+node5[0]));
      mid45.push_back(0.5*(node4[1]+node5[1]));
      mid45.push_back(0.5*(node4[2]+node5[2]));
      mid56.push_back(0.5*(node5[0]+node6[0]));
      mid56.push_back(0.5*(node5[1]+node6[1]));
      mid56.push_back(0.5*(node5[2]+node6[2]));
      mid67.push_back(0.5*(node6[0]+node7[0]));
      mid67.push_back(0.5*(node6[1]+node7[1]));
      mid67.push_back(0.5*(node6[2]+node7[2]));
      mid47.push_back(0.5*(node4[0]+node7[0]));
      mid47.push_back(0.5*(node4[1]+node7[1]));
      mid47.push_back(0.5*(node4[2]+node7[2]));
      
      
      mid0123.push_back(0.25*(node0[0]+node1[0]+node2[0]+node3[0]));
      mid0123.push_back(0.25*(node0[1]+node1[1]+node2[1]+node3[1]));
      mid0123.push_back(0.25*(node0[2]+node1[2]+node2[2]+node3[2]));
      mid0145.push_back(0.25*(node0[0]+node1[0]+node4[0]+node5[0]));
      mid0145.push_back(0.25*(node0[1]+node1[1]+node4[1]+node5[1]));
      mid0145.push_back(0.25*(node0[2]+node1[2]+node4[2]+node5[2]));
      mid1256.push_back(0.25*(node1[0]+node2[0]+node5[0]+node6[0]));
      mid1256.push_back(0.25*(node1[1]+node2[1]+node5[1]+node6[1]));
      mid1256.push_back(0.25*(node1[2]+node2[2]+node5[2]+node6[2]));
      mid2367.push_back(0.25*(node2[0]+node3[0]+node6[0]+node7[0]));
      mid2367.push_back(0.25*(node2[1]+node3[1]+node6[1]+node7[1]));
      mid2367.push_back(0.25*(node2[2]+node3[2]+node6[2]+node7[2]));
      mid0347.push_back(0.25*(node0[0]+node3[0]+node4[0]+node7[0]));
      mid0347.push_back(0.25*(node0[1]+node3[1]+node4[1]+node7[1]));
      mid0347.push_back(0.25*(node0[2]+node3[2]+node4[2]+node7[2]));
      mid4567.push_back(0.25*(node4[0]+node5[0]+node6[0]+node7[0]));
      mid4567.push_back(0.25*(node4[1]+node5[1]+node6[1]+node7[1]));
      mid4567.push_back(0.25*(node4[2]+node5[2]+node6[2]+node7[2]));
      
      // Check if these nodes have been added and add if not
      ScalarT tol=1.0e-6;
      
      int mid0123_ind, mid01_ind, mid12_ind, mid23_ind, mid03_ind;
      int center_ind, mid04_ind, mid15_ind, mid26_ind, mid37_ind;
      int mid0145_ind, mid1256_ind, mid2367_ind, mid0347_ind;
      int mid4567_ind, mid45_ind, mid56_ind, mid67_ind, mid47_ind;
      bool found;
      
      found = checkExistingSubNodes(mid0123,tol,mid0123_ind);
      if (!found) {
        subnodes_.push_back(mid0123);
        mid0123_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid01,tol,mid01_ind);
      if (!found) {
        subnodes_.push_back(mid01);
        mid01_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid12,tol,mid12_ind);
      if (!found) {
        subnodes_.push_back(mid12);
        mid12_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid23,tol,mid23_ind);
      if (!found) {
        subnodes_.push_back(mid23);
        mid23_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid03,tol,mid03_ind);
      if (!found) {
        subnodes_.push_back(mid03);
        mid03_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(center,tol,center_ind);
      if (!found) {
        subnodes_.push_back(center);
        center_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid04,tol,mid04_ind);
      if (!found) {
        subnodes_.push_back(mid04);
        mid04_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid15,tol,mid15_ind);
      if (!found) {
        subnodes_.push_back(mid15);
        mid15_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid26,tol,mid26_ind);
      if (!found) {
        subnodes_.push_back(mid26);
        mid26_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid37,tol,mid37_ind);
      if (!found) {
        subnodes_.push_back(mid37);
        mid37_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid0145,tol,mid0145_ind);
      if (!found) {
        subnodes_.push_back(mid0145);
        mid0145_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid1256,tol,mid1256_ind);
      if (!found) {
        subnodes_.push_back(mid1256);
        mid1256_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid2367,tol,mid2367_ind);
      if (!found) {
        subnodes_.push_back(mid2367);
        mid2367_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid0347,tol,mid0347_ind);
      if (!found) {
        subnodes_.push_back(mid0347);
        mid0347_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid4567,tol,mid4567_ind);
      if (!found) {
        subnodes_.push_back(mid4567);
        mid4567_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid45,tol,mid45_ind);
      if (!found) {
        subnodes_.push_back(mid45);
        mid45_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid56,tol,mid56_ind);
      if (!found) {
        subnodes_.push_back(mid56);
        mid56_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid67,tol,mid67_ind);
      if (!found) {
        subnodes_.push_back(mid67);
        mid67_ind = subnodes_.size()-1;
      }
      
      found = checkExistingSubNodes(mid47,tol,mid47_ind);
      if (!found) {
        subnodes_.push_back(mid47);
        mid47_ind = subnodes_.size()-1;
      }
      
      // Define the new elements (appended to end of list)
      vector<GO> elem0, elem1, elem2, elem3, elem4, elem5, elem6, elem7;
      
      elem0.push_back(subconnectivity_[e][0]);
      elem0.push_back(mid01_ind);
      elem0.push_back(mid0123_ind);
      elem0.push_back(mid03_ind);
      elem0.push_back(mid04_ind);
      elem0.push_back(mid0145_ind);
      elem0.push_back(center_ind);
      elem0.push_back(mid0347_ind);
      
      subconnectivity_.push_back(elem0);
      
      elem1.push_back(mid01_ind);
      elem1.push_back(subconnectivity_[e][1]);
      elem1.push_back(mid12_ind);
      elem1.push_back(mid0123_ind);
      elem1.push_back(mid0145_ind);
      elem1.push_back(mid15_ind);
      elem1.push_back(mid1256_ind);
      elem1.push_back(center_ind);
      
      subconnectivity_.push_back(elem1);
      
      elem2.push_back(mid0123_ind);
      elem2.push_back(mid12_ind);
      elem2.push_back(subconnectivity_[e][2]);
      elem2.push_back(mid23_ind);
      elem2.push_back(center_ind);
      elem2.push_back(mid1256_ind);
      elem2.push_back(mid26_ind);
      elem2.push_back(mid2367_ind);
      
      subconnectivity_.push_back(elem2);
      
      elem3.push_back(mid03_ind);
      elem3.push_back(mid0123_ind);
      elem3.push_back(mid23_ind);
      elem3.push_back(subconnectivity_[e][3]);
      elem3.push_back(mid0347_ind);
      elem3.push_back(center_ind);
      elem3.push_back(mid2367_ind);
      elem3.push_back(mid37_ind);
      
      subconnectivity_.push_back(elem3);
      
      elem4.push_back(mid04_ind);
      elem4.push_back(mid0145_ind);
      elem4.push_back(center_ind);
      elem4.push_back(mid0347_ind);
      elem4.push_back(subconnectivity_[e][4]);
      elem4.push_back(mid45_ind);
      elem4.push_back(mid4567_ind);
      elem4.push_back(mid47_ind);
      
      subconnectivity_.push_back(elem4);
      
      elem5.push_back(mid0145_ind);
      elem5.push_back(mid15_ind);
      elem5.push_back(mid1256_ind);
      elem5.push_back(center_ind);
      elem5.push_back(mid45_ind);
      elem5.push_back(subconnectivity_[e][5]);
      elem5.push_back(mid56_ind);
      elem5.push_back(mid4567_ind);
      
      subconnectivity_.push_back(elem5);
      
      elem6.push_back(center_ind);
      elem6.push_back(mid1256_ind);
      elem6.push_back(mid26_ind);
      elem6.push_back(mid2367_ind);
      elem6.push_back(mid4567_ind);
      elem6.push_back(mid56_ind);
      elem6.push_back(subconnectivity_[e][6]);
      elem6.push_back(mid67_ind);
      
      subconnectivity_.push_back(elem6);
      
      elem7.push_back(mid0347_ind);
      elem7.push_back(center_ind);
      elem7.push_back(mid2367_ind);
      elem7.push_back(mid37_ind);
      elem7.push_back(mid47_ind);
      elem7.push_back(mid4567_ind);
      elem7.push_back(mid67_ind);
      elem7.push_back(subconnectivity_[e][7]);
      
      subconnectivity_.push_back(elem7);
      
      Kokkos::View<bool*,HostDevice> oldmap = subsidemap_[e];
      Kokkos::View<bool*,HostDevice> newsm0("newsi",6);
      Kokkos::View<bool*,HostDevice> newsm1("newsi",6);
      Kokkos::View<bool*,HostDevice> newsm2("newsi",6);
      Kokkos::View<bool*,HostDevice> newsm3("newsi",6);
      Kokkos::View<bool*,HostDevice> newsm4("newsi",6);
      Kokkos::View<bool*,HostDevice> newsm5("newsi",6);
      Kokkos::View<bool*,HostDevice> newsm6("newsi",6);
      Kokkos::View<bool*,HostDevice> newsm7("newsi",6);
      
      Kokkos::deep_copy(newsm0,oldmap);
      Kokkos::deep_copy(newsm1,oldmap);
      Kokkos::deep_copy(newsm2,oldmap);
      Kokkos::deep_copy(newsm3,oldmap);
      Kokkos::deep_copy(newsm4,oldmap);
      Kokkos::deep_copy(newsm5,oldmap);
      Kokkos::deep_copy(newsm6,oldmap);
      Kokkos::deep_copy(newsm7,oldmap);
      
      // order = 0145, 1256, 2367, 0367, 0123, 4567
      // order = bottom, right, top, left, back, front
      newsm0(1) = false;
      newsm0(2) = false;
      newsm0(5) = false;
      
      newsm1(2) = false;
      newsm1(3) = false;
      newsm1(5) = false;
      
      newsm2(0) = false;
      newsm2(3) = false;
      newsm2(5) = false;
      
      newsm3(0) = false;
      newsm3(1) = false;
      newsm3(5) = false;
      
      newsm4(1) = false;
      newsm4(2) = false;
      newsm4(4) = false;
      
      newsm5(2) = false;
      newsm5(3) = false;
      newsm5(4) = false;
      
      newsm6(0) = false;
      newsm6(3) = false;
      newsm6(4) = false;
      
      newsm7(0) = false;
      newsm7(1) = false;
      newsm7(4) = false;
      
      subsidemap_.push_back(newsm0);
      subsidemap_.push_back(newsm1);
      subsidemap_.push_back(newsm2);
      subsidemap_.push_back(newsm3);
      subsidemap_.push_back(newsm4);
      subsidemap_.push_back(newsm5);
      subsidemap_.push_back(newsm6);
      subsidemap_.push_back(newsm7);
      
    }
    else {
      // add error
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Check if a sub-grid nodes has already been added to the list
///////////////////////////////////////////////////////////////////////////////////////

bool SubGridTools2::checkExistingSubNodes(const vector<ScalarT> & newpt,
                                         const ScalarT & tol, int & index) {
  bool found = false;
  int dimension_ = newpt.size();
  for (unsigned int i=0; i<subnodes_.size(); i++) {
    if (!found) {
      ScalarT val = 0.0;
      for (int j=0; j<dimension_; j++) {
        val += (subnodes_[i][j]-newpt[j])*(subnodes_[i][j]-newpt[j]);
      }
      if (sqrt(val)<tol) {
        found = true;
        index = i;
      }
    }
  }
  return found;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the sub-grid nodes as a list: output is (Nnodes x dimension_)
///////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<ScalarT**,HostDevice> SubGridTools2::getListOfPhysicalNodes(DRV newmacronodes, topo_RCP & macro_topo,
                                                                        Teuchos::RCP<DiscretizationInterface> & disc) {
  
  //DRV newnodes("nodes on phys elem", newmacronodes.extent(0), subnodes_list_.extent(0), dimension_);
  //CellTools::mapToPhysicalFrame(newnodes, subnodes_list_, newmacronodes, *macro_topo);
  DRV newnodes = disc->mapPointsToPhysical(subnodes_list_, newmacronodes, macro_topo);
  
  auto newnodes_host = Kokkos::create_mirror_view(newnodes);
  Kokkos::deep_copy(newnodes_host,newnodes);
  
  Kokkos::View<ScalarT**,HostDevice> currnodes("currnodes",newmacronodes.extent(0)*subnodes_list_.extent(0), dimension_);
  
  for (size_type melem=0; melem<newmacronodes.extent(0); melem++) {
    for (size_type elem=0; elem<subnodes_list_.extent(0); elem++) {
      for (int dim=0; dim<dimension_; dim++) {
        size_t index = melem*subnodes_list_.extent(0)+elem;
        currnodes(index,dim) = newnodes_host(melem,elem,dim);
      }
    }
  }
  
  return currnodes;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the sub-grid nodes on each element: output is (Nelem x Nnperelem x dimension_)
///////////////////////////////////////////////////////////////////////////////////////

DRV SubGridTools2::getPhysicalNodes(DRV newmacronodes, topo_RCP & macro_topo,
                                   Teuchos::RCP<DiscretizationInterface> & disc) {
  
  DRV newnodes = disc->mapPointsToPhysical(subnodes_list_, newmacronodes, macro_topo);
  auto newnodes_host = Kokkos::create_mirror_view(newnodes);
  Kokkos::deep_copy(newnodes_host, newnodes);
  
  DRV currnodes("currnodes",newmacronodes.extent(0)*subconnectivity_.size(),
                subconnectivity_[0].size(),
                dimension_);
  auto currnodes_host = Kokkos::create_mirror_view(currnodes);
  
  for (size_type melem=0; melem<newmacronodes.extent(0); melem++) {
    for (size_type elem=0; elem<subconnectivity_.size(); elem++) {
      for (size_t node=0; node<subconnectivity_[elem].size(); node++) {
        for (int dim=0; dim<dimension_; dim++) {
          size_t index = melem*subconnectivity_.size()+elem;
          currnodes_host(index,node,dim) = newnodes_host(melem,subconnectivity_[elem][node],dim);
        }
      }
    }
  }
  Kokkos::deep_copy(currnodes, currnodes_host);
  
  return currnodes;
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the sub-grid connectivity
///////////////////////////////////////////////////////////////////////////////////////

vector<vector<GO> > SubGridTools2::getPhysicalConnectivity(int & reps) {
  vector<vector<GO> > newconn;
  
  int prog = 0;
  for (int k=0; k<reps; k++) {
    for (size_t i=0; i<subconnectivity_.size(); i++) {
      vector<GO> cc;
      for (size_t j=0; j<subconnectivity_[i].size(); j++) {
        cc.push_back(subconnectivity_[i][j]+prog);
      }
      newconn.push_back(cc);
    }
    prog += subnodes_list_.extent(0);//.size();
  }
  return newconn;
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the groups of boundary elements
///////////////////////////////////////////////////////////////////////////////////////

void SubGridTools2::getBoundaryGroups(size_t & numMacro,
                                      vector<vector<size_t> > & boundary_groups) {
  
  for (size_t s=0; s<subsidemap_[0].extent(0); s++) {
    vector<size_t> group;
    for (size_t m=0; m<numMacro; ++m) {
      for (size_t c=0; c<subsidemap_.size(); c++) {
        if (subsidemap_[c](s)) {
          size_t eindex = m*subsidemap_.size() + c;
          group.push_back(eindex);
        }
      }
    }
    boundary_groups.push_back(group);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
// Get the number of nodes on the reference element
///////////////////////////////////////////////////////////////////////////////////////

size_t SubGridTools2::getNumRefNodes() {
  return subnodes_list_.extent(0);
}

