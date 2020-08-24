/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SUBGRIDTOOLS_H
#define SUBGRIDTOOLS_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_LineMeshFactory.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SquareTriMeshFactory.hpp"
#include "Panzer_STK_CubeHexMeshFactory.hpp"
#include "Panzer_STK_CubeTetMeshFactory.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"

class SubGridTools {
public:
  
  SubGridTools() {} ;
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  SubGridTools(const Teuchos::RCP<MpiComm> & LocalComm_, const string & shape_,
               const string & subshape_, const DRV nodes_,
               Kokkos::View<int****,HostDevice> sideinfo_,
               std::string & mesh_type_, std::string & mesh_file_) :
  LocalComm(LocalComm_), shape(shape_), subshape(subshape_), sideinfo(sideinfo_),
  mesh_type(mesh_type_), mesh_file(mesh_file_) {
    
    nodes = nodes_;
    
    dimension = nodes.extent(1);
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Given the coarse grid nodes and shape, define the subgrid nodes, connectivity, and sideinfo
  //////////////////////////////////////////////////////////////////////////////////////
  
  void createSubMesh(const int & numrefine) {
    
    if (mesh_type == "Exodus" || mesh_type == "panzer") {
    
      if (mesh_type == "Exodus") {
        // Read in the mesh
        Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::rcp(new Teuchos::ParameterList);
        Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory = Teuchos::rcp(new panzer_stk::STK_ExodusReaderFactory());
        pl->set("File Name",mesh_file);
        mesh_factory->setParameterList(pl);
        ref_mesh = mesh_factory->buildUncommitedMesh(*(LocalComm->getRawMpiComm()));
        mesh_factory->completeMeshConstruction(*ref_mesh,*(LocalComm->getRawMpiComm()));
      }
      else {
        
        if (shape == "tri" || shape == "tet") {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the panzer subgrid meshes cannot be used for triangles or tetrahedrons.");
        }
        
        Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::rcp(new Teuchos::ParameterList);
        
        int numEPerD = static_cast<int>(std::pow(2,numrefine));
        pl->set("X Blocks", 1);
        pl->set("X Elements", numEPerD);
        pl->set("X0", -1.0);
        pl->set("Xf", 1.0);
        pl->set("X Procs", 1);
        if (dimension > 1) {
          pl->set("Y Blocks", 1);
          pl->set("Y Elements", numEPerD);
          pl->set("Y0", -1.0);
          pl->set("Yf", 1.0);
          pl->set("Y Procs", 1);
        }
        if (dimension > 2) {
          pl->set("Z Blocks", 1);
          pl->set("Z Elements", numEPerD);
          pl->set("Z0", -1.0);
          pl->set("Zf", 1.0);
          pl->set("Z Procs", 1);
        }
        
        Teuchos::RCP<panzer_stk::STK_MeshFactory> mesh_factory;
        if (dimension == 1)
          mesh_factory = Teuchos::rcp(new panzer_stk::LineMeshFactory());
        else if (dimension == 2) {
          if (subshape == "quad") {
            mesh_factory = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory());
          }
          if (subshape == "tri") {
            mesh_factory = Teuchos::rcp(new panzer_stk::SquareTriMeshFactory());
          }
        }
        else if (dimension == 3) {
          if (subshape == "hex") {
            mesh_factory = Teuchos::rcp(new panzer_stk::CubeHexMeshFactory());
          }
          if (subshape == "tet") {
            mesh_factory = Teuchos::rcp(new panzer_stk::CubeTetMeshFactory());
          }
        }
        mesh_factory->setParameterList(pl);
        ref_mesh = mesh_factory->buildUncommitedMesh(*(LocalComm->getRawMpiComm()));
        mesh_factory->completeMeshConstruction(*ref_mesh,*(LocalComm->getRawMpiComm()));
      }
      
      std::vector<string> blocknames;
      ref_mesh->getElementBlockNames(blocknames);
      
      topo_RCP cellTopo = ref_mesh->getCellTopology(blocknames[0]);
      size_t numNodesPerElem = cellTopo->getNodeCount();
      size_t numSides;
      if (dimension == 1) {
        numSides = 2;
      }
      else if (dimension == 2) {
        numSides = cellTopo->getSideCount();
      }
      else if (dimension == 3) {
        numSides = cellTopo->getFaceCount();
      }
      
      std::vector<stk::mesh::Entity> ref_elements;
      ref_mesh->getMyElements(ref_elements);
      DRV vertices("element vertices",ref_elements.size(), numNodesPerElem, dimension);
      ref_mesh->getElementVertices_FromCoordsNoResize(ref_elements, vertices);
      
      // extract the nodes
      size_t numTotalNodes = ref_mesh->getMaxEntityId(0);
      
      subnodes_list = DRV("DRV of subgrid nodes on ref elem", numTotalNodes, dimension);
      vector<bool> beenAdded(numTotalNodes,false);
      
      // Extract the connectivity
      for (size_t elem=0; elem<ref_elements.size(); elem++) {
        std::vector< stk::mesh::EntityId > nodeIds;
        ref_mesh->getNodeIdsForElement(ref_elements[elem],nodeIds);
        vector<GO> conn;
        for (size_t i=0; i<nodeIds.size(); i++) {
          conn.push_back(nodeIds[i]-1);
          if (!beenAdded[nodeIds[i]-1]) {
            for (size_t s=0; s<dimension; s++) {
              subnodes_list(nodeIds[i]-1,s) = vertices(elem,i,s);
            }
            beenAdded[nodeIds[i]-1] = true;
          }
        }
        subconnectivity.push_back(conn);
        
        Kokkos::View<int**,AssemblyDevice> newsidemap("new side map",numSides,2);
        subsidemap.push_back(newsidemap);
        
      }
      
      vector<string> sideSets;
      ref_mesh->getSidesetNames(sideSets);
      
      if (dimension == 2) {
        if (sideSets.size() < 4) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the subgrid mesh requires at least 4 sidesets.");
        }
      }
      else if (dimension == 3) {
        if (sideSets.size() < 6) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the subgrid mesh requires at least 6 sidesets.");
        }
      }
      for( size_t side=0; side<sideSets.size(); side++ ) {
        string sideName = sideSets[side];
        
        vector<stk::mesh::Entity> sideEntities;
        ref_mesh->getMySides(sideName, blocknames[0], sideEntities);
        
        vector<size_t>             local_side_Ids;
        vector<stk::mesh::Entity>  side_output;
        vector<size_t>             local_elem_Ids;
        panzer_stk::workset_utils::getSideElements(*ref_mesh, blocknames[0], sideEntities, local_side_Ids, side_output);
        
        for( size_t i=0; i<side_output.size(); i++ ) {
          local_elem_Ids.push_back(ref_mesh->elementLocalId(side_output[i]));
          size_t localid = local_elem_Ids[i];
          subsidemap[localid](local_side_Ids[i], 0) = 1;
          subsidemap[localid](local_side_Ids[i], 1) = side;
        }
      }
      
    }
    else if (mesh_type == "inline") {
      if (subshape == shape) {
        vector<GO> newconn;
        for (unsigned int i=0; i<nodes.extent(0); i++) {
          vector<ScalarT> newnode;
          for (unsigned int s=0; s<dimension; s++) {
            newnode.push_back(nodes(i,s));
          }
          subnodes.push_back(newnode);
          newconn.push_back(i);
        }
        subconnectivity.push_back(newconn);
        
        Kokkos::View<int**,AssemblyDevice> newsidemap("newsidemap",sideinfo.extent(2),2);
        for (size_t s=0; s<sideinfo.extent(2); s++) {
          newsidemap(s,0) = 1;
          newsidemap(s,1) = s;
        }
        subsidemap.push_back(newsidemap);
        
      }
      else {
        if (dimension == 1) {
          // output an error message
        }
        else if (dimension == 2) {
          if (shape == "quad" && subshape == "tri") {
            
            for (unsigned int i=0; i<nodes.extent(0); i++) {
              vector<ScalarT> newnode;
              for (unsigned int s=0; s<dimension; s++) {
                newnode.push_back(nodes(i,s));
              }
              subnodes.push_back(newnode);
            }
            vector<ScalarT> midnode;
            midnode.push_back(0.25*(nodes(0,0)+nodes(1,0)+nodes(2,0)+nodes(3,0)));
            midnode.push_back(0.25*(nodes(0,1)+nodes(1,1)+nodes(2,1)+nodes(3,1)));
            subnodes.push_back(midnode);
            
            vector<GO> newconn0 = {0,1,4};
            subconnectivity.push_back(newconn0);
            
            vector<GO> newconn1 = {1,2,4};
            subconnectivity.push_back(newconn1);
            
            vector<GO> newconn2 = {2,3,4};
            subconnectivity.push_back(newconn2);
            
            vector<GO> newconn3 = {3,0,4};
            subconnectivity.push_back(newconn3);
            
            Kokkos::View<int**,AssemblyDevice> newsidemap0("newsi",3,2);
            Kokkos::View<int**,AssemblyDevice> newsidemap1("newsi",3,2);
            Kokkos::View<int**,AssemblyDevice> newsidemap2("newsi",3,2);
            Kokkos::View<int**,AssemblyDevice> newsidemap3("newsi",3,2);
            
            newsidemap0(0,0) = 1;
            if (sideinfo(0,0,0,0) > 0)
              newsidemap0(0,1) = 0;
            else
              newsidemap0(0,1) = -1;
            
            newsidemap1(0,0) = 1;
            if (sideinfo(0,0,1,0) > 0)
              newsidemap1(0,1) = 1;
            else
              newsidemap1(0,1) = -1;
            
            newsidemap2(0,0) = 1;
            if (sideinfo(0,0,2,0) > 0)
              newsidemap2(0,1) = 2;
            else
              newsidemap2(0,1) = -1;
            
            newsidemap3(0,0) = 1;
            if (sideinfo(0,0,3,0) > 0)
              newsidemap3(0,1) = 3;
            else
              newsidemap3(0,1) = -1;
            
            subsidemap.push_back(newsidemap0);
            subsidemap.push_back(newsidemap1);
            subsidemap.push_back(newsidemap2);
            subsidemap.push_back(newsidemap3);
            
          }
          else if (shape == "tri" && subshape == "quad") {
            
            for (unsigned int i=0; i<nodes.extent(0); i++) {
              vector<ScalarT> newnode;
              
              for (unsigned int s=0; s<dimension; s++) {
                newnode.push_back(nodes(i,s));
              }
              subnodes.push_back(newnode);
              
            }
            
            vector<ScalarT> center, mid01, mid12, mid02;
            center.push_back(1.0/3.0*(nodes(0,0,0)+nodes(0,1,0)+nodes(0,2,0)));
            center.push_back(1.0/3.0*(nodes(0,0,1)+nodes(0,1,1)+nodes(0,2,1)));
            mid01.push_back(0.5*(nodes(0,0,0)+nodes(0,1,0)));
            mid01.push_back(0.5*(nodes(0,0,1)+nodes(0,1,1)));
            mid12.push_back(0.5*(nodes(0,1,0)+nodes(0,2,0)));
            mid12.push_back(0.5*(nodes(0,1,1)+nodes(0,2,1)));
            mid02.push_back(0.5*(nodes(0,0,0)+nodes(0,2,0)));
            mid02.push_back(0.5*(nodes(0,0,1)+nodes(0,2,1)));
            subnodes.push_back(center);
            subnodes.push_back(mid01);
            subnodes.push_back(mid12);
            subnodes.push_back(mid02);
            
            vector<GO> newconn0 = {0,4,3,6};
            subconnectivity.push_back(newconn0);
            
            vector<GO> newconn1 = {1,5,3,4};
            subconnectivity.push_back(newconn1);
            
            vector<GO> newconn2 = {2,6,3,5};
            subconnectivity.push_back(newconn2);
            
            Kokkos::View<int**,AssemblyDevice> newsidemap0("newsi",4,2);
            Kokkos::View<int**,AssemblyDevice> newsidemap1("newsi",4,2);
            Kokkos::View<int**,AssemblyDevice> newsidemap2("newsi",4,2);
            
            newsidemap0(0,0) = 1;
            if (sideinfo(0,0,0,0) > 0)
              newsidemap0(0,1) = 0;
            else
              newsidemap0(0,1) = -1;
            
            newsidemap0(3,0) = 1;
            if (sideinfo(0,0,2,0) > 0)
              newsidemap0(3,1) = 2;
            else
              newsidemap0(3,1) = -1;
            
            newsidemap1(0,0) = 1;
            if (sideinfo(0,0,1,0) > 0)
              newsidemap1(0,1) = 1;
            else
              newsidemap1(0,1) = -1;
            
            newsidemap1(3,0) = 1;
            if (sideinfo(0,0,0,0) > 0)
              newsidemap1(3,1) = 0;
            else
              newsidemap1(3,1) = -1;
            
            newsidemap2(0,0) = 1;
            if (sideinfo(0,0,2,0) > 0)
              newsidemap2(0,1) = 2;
            else
              newsidemap2(0,1) = -1;
            
            newsidemap2(3,0) = 1;
            if (sideinfo(0,0,1,0) > 0)
              newsidemap2(3,1) = 1;
            else
              newsidemap2(3,1) = -1;
            
            subsidemap.push_back(newsidemap0);
            subsidemap.push_back(newsidemap1);
            subsidemap.push_back(newsidemap2);
            
          }
          else {
            // output an error message
          }
        }
        else if (dimension == 3) {
          // conversions from het to tet or tet to hex are not added yet
        }
        
      }
      
      // Recursively refine the elements
      for (int r=0; r<numrefine; r++) {
        size_t numelem = subconnectivity.size();
        for (int e=0; e<numelem; e++) {
          refineSubCell(e); // adds new nodes and new elements (does not delete old elements)
        }
        subconnectivity.erase(subconnectivity.begin(), subconnectivity.begin()+numelem);
        subsidemap.erase(subsidemap.begin(), subsidemap.begin()+numelem);
      }
      
      // Create the DRV list of nodes
      subnodes_list = DRV("DRV of subgrid nodes on ref elem",subnodes.size(), subnodes[0].size());
      for (size_t node=0; node<subnodes.size(); node++) {
        for (size_t dim=0; dim<subnodes[0].size(); dim++) {
          subnodes_list(node,dim) = subnodes[node][dim];
        }
      }
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized subgrid mesh type: " + mesh_type);
    }
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Uniformly refine an element
  //////////////////////////////////////////////////////////////////////////////////////
  
  void refineSubCell(const int & e) {
    
    if (dimension == 1) {
      vector<ScalarT> node0 = subnodes[subconnectivity[e][0]];
      vector<ScalarT> node1 = subnodes[subconnectivity[e][1]];
      
      ScalarT midx = 0.5*(node0[0]+node1[0]);
      vector<ScalarT> mid(1,midx);
      
      subnodes.push_back(mid);
      
      vector<GO> newelem0, newelem1;
      newelem0.push_back(subconnectivity[e][0]);
      newelem0.push_back(subnodes.size());
      newelem1.push_back(subnodes.size());
      newelem1.push_back(subconnectivity[e][1]);
      
      subconnectivity.push_back(newelem0);
      subconnectivity.push_back(newelem1);
      
      Kokkos::View<int**,AssemblyDevice> oldmap = subsidemap[e];
      Kokkos::View<int**,AssemblyDevice> newsm0("newsi",2,2);
      Kokkos::View<int**,AssemblyDevice> newsm1("newsi",2,2);
      Kokkos::deep_copy(newsm0,oldmap);
      Kokkos::deep_copy(newsm1,oldmap);
      
      newsm0(1,0) = 0;
      newsm0(1,1) = 0;
      newsm1(0,0) = 0;
      newsm1(0,1) = 0;
      subsidemap.push_back(newsm0);
      subsidemap.push_back(newsm1);
      
    }
    if (dimension == 2) {
      if (subshape == "tri") {
        // Extract the existing nodes
        vector<ScalarT> node0 = subnodes[subconnectivity[e][0]];
        vector<ScalarT> node1 = subnodes[subconnectivity[e][1]];
        vector<ScalarT> node2 = subnodes[subconnectivity[e][2]];
        
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
          subnodes.push_back(mid01);
          mid01_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid12,tol,mid12_ind);
        if (!found) {
          subnodes.push_back(mid12);
          mid12_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid02,tol,mid02_ind);
        if (!found) {
          subnodes.push_back(mid02);
          mid02_ind = subnodes.size()-1;
        }
        
        // Define the new elements (appended to end of list)
        vector<GO> elem0, elem1, elem2, elem3;
        
        elem0.push_back(subconnectivity[e][0]);
        elem0.push_back(mid01_ind);
        elem0.push_back(mid02_ind);
        subconnectivity.push_back(elem0);
        
        elem1.push_back(subconnectivity[e][1]);
        elem1.push_back(mid01_ind);
        elem1.push_back(mid12_ind);
        subconnectivity.push_back(elem1);
        
        elem2.push_back(subconnectivity[e][2]);
        elem2.push_back(mid02_ind);
        elem2.push_back(mid12_ind);
        subconnectivity.push_back(elem2);
        
        elem3.push_back(mid01_ind);
        elem3.push_back(mid12_ind);
        elem3.push_back(mid02_ind);
        subconnectivity.push_back(elem3);
        
        
        Kokkos::View<int**,AssemblyDevice> oldmap = subsidemap[e];
        Kokkos::View<int**,AssemblyDevice> newsm0("newsi",3,2);
        Kokkos::View<int**,AssemblyDevice> newsm1("newsi",3,2);
        Kokkos::View<int**,AssemblyDevice> newsm2("newsi",3,2);
        Kokkos::View<int**,AssemblyDevice> newsm3("newsi",3,2);
        Kokkos::deep_copy(newsm0,oldmap);
        Kokkos::deep_copy(newsm1,oldmap);
        Kokkos::deep_copy(newsm2,oldmap);
        Kokkos::deep_copy(newsm3,oldmap);
        
        newsm0(1,0) = 0;
        newsm0(1,1) = 0;
        newsm1(1,0) = 0;
        newsm1(1,1) = 0;
        newsm2(1,0) = 0;
        newsm2(1,1) = 0;
        newsm3(0,0) = 0;
        newsm3(0,1) = 0;
        newsm3(1,0) = 0;
        newsm3(1,1) = 0;
        newsm3(2,0) = 0;
        newsm3(2,1) = 0;
        
        subsidemap.push_back(newsm0);
        subsidemap.push_back(newsm1);
        subsidemap.push_back(newsm2);
        subsidemap.push_back(newsm3);
        
        
      }
      else if (subshape == "quad") {
        // Extract the existing nodes
        vector<ScalarT> node0 = subnodes[subconnectivity[e][0]];
        vector<ScalarT> node1 = subnodes[subconnectivity[e][1]];
        vector<ScalarT> node2 = subnodes[subconnectivity[e][2]];
        vector<ScalarT> node3 = subnodes[subconnectivity[e][3]];
        
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
          subnodes.push_back(center);
          center_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid01,tol,mid01_ind);
        if (!found) {
          subnodes.push_back(mid01);
          mid01_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid12,tol,mid12_ind);
        if (!found) {
          subnodes.push_back(mid12);
          mid12_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid23,tol,mid23_ind);
        if (!found) {
          subnodes.push_back(mid23);
          mid23_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid03,tol,mid03_ind);
        if (!found) {
          subnodes.push_back(mid03);
          mid03_ind = subnodes.size()-1;
        }
        
        // Define the new elements (appended to end of list)
        vector<GO> elem0, elem1, elem2, elem3;
        
        elem0.push_back(subconnectivity[e][0]);
        elem0.push_back(mid01_ind);
        elem0.push_back(center_ind);
        elem0.push_back(mid03_ind);
        subconnectivity.push_back(elem0);
        
        elem1.push_back(mid01_ind);
        elem1.push_back(subconnectivity[e][1]);
        elem1.push_back(mid12_ind);
        elem1.push_back(center_ind);
        subconnectivity.push_back(elem1);
        
        elem2.push_back(center_ind);
        elem2.push_back(mid12_ind);
        elem2.push_back(subconnectivity[e][2]);
        elem2.push_back(mid23_ind);
        subconnectivity.push_back(elem2);
        
        elem3.push_back(mid03_ind);
        elem3.push_back(center_ind);
        elem3.push_back(mid23_ind);
        elem3.push_back(subconnectivity[e][3]);
        subconnectivity.push_back(elem3);
        
        Kokkos::View<int**,AssemblyDevice> oldmap = subsidemap[e];
        Kokkos::View<int**,AssemblyDevice> newsm0("newsm",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm1("newsm",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm2("newsm",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm3("newsm",4,2);
        Kokkos::deep_copy(newsm0,oldmap);
        Kokkos::deep_copy(newsm1,oldmap);
        Kokkos::deep_copy(newsm2,oldmap);
        Kokkos::deep_copy(newsm3,oldmap);
        
        newsm0(1,0) = 0;
        newsm0(1,1) = 0;
        newsm0(2,0) = 0;
        newsm0(2,1) = 0;
        
        newsm1(2,0) = 0;
        newsm1(2,1) = 0;
        newsm1(3,0) = 0;
        newsm1(3,1) = 0;
        
        newsm2(0,0) = 0;
        newsm2(0,1) = 0;
        newsm2(3,0) = 0;
        newsm2(3,1) = 0;
        
        newsm3(0,0) = 0;
        newsm3(0,1) = 0;
        newsm3(1,0) = 0;
        newsm3(1,1) = 0;
      
        subsidemap.push_back(newsm0);
        subsidemap.push_back(newsm1);
        subsidemap.push_back(newsm2);
        subsidemap.push_back(newsm3);
        
      }
      else {
        // add error
      }
    }
    if (dimension == 3) {
      if (subshape == "tet") {
        // Extract the existing nodes
        vector<ScalarT> node0 = subnodes[subconnectivity[e][0]];
        vector<ScalarT> node1 = subnodes[subconnectivity[e][1]];
        vector<ScalarT> node2 = subnodes[subconnectivity[e][2]];
        vector<ScalarT> node3 = subnodes[subconnectivity[e][3]];
        
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
          subnodes.push_back(mid01);
          mid01_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid12,tol,mid12_ind);
        if (!found) {
          subnodes.push_back(mid12);
          mid12_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid02,tol,mid02_ind);
        if (!found) {
          subnodes.push_back(mid02);
          mid02_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid03,tol,mid03_ind);
        if (!found) {
          subnodes.push_back(mid03);
          mid03_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid13,tol,mid13_ind);
        if (!found) {
          subnodes.push_back(mid13);
          mid13_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid23,tol,mid23_ind);
        if (!found) {
          subnodes.push_back(mid23);
          mid23_ind = subnodes.size()-1;
        }
        
        // Define the new elements (appended to end of list)
        vector<GO> elem0, elem1, elem2, elem3, elem4, elem5, elem6, elem7;
        
        elem0.push_back(subconnectivity[e][0]);
        elem0.push_back(mid01_ind);
        elem0.push_back(mid02_ind);
        elem0.push_back(mid03_ind);
        subconnectivity.push_back(elem0);
        
        elem1.push_back(subconnectivity[e][1]);
        elem1.push_back(mid12_ind);
        elem1.push_back(mid01_ind);
        elem1.push_back(mid13_ind);
        subconnectivity.push_back(elem1);
        
        elem2.push_back(subconnectivity[e][2]);
        elem2.push_back(mid02_ind);
        elem2.push_back(mid12_ind);
        elem2.push_back(mid23_ind);
        subconnectivity.push_back(elem2);
        
        elem3.push_back(mid03_ind);
        elem3.push_back(mid13_ind);
        elem3.push_back(mid23_ind);
        elem3.push_back(subconnectivity[e][3]);
        subconnectivity.push_back(elem3);
        
        elem4.push_back(mid01_ind);
        elem4.push_back(mid12_ind);
        elem4.push_back(mid02_ind);
        elem4.push_back(mid03_ind);
        subconnectivity.push_back(elem4);
        
        elem5.push_back(mid03_ind);
        elem5.push_back(mid13_ind);
        elem5.push_back(mid23_ind);
        elem5.push_back(mid12_ind);
        subconnectivity.push_back(elem5);
        
        elem6.push_back(mid03_ind);
        elem6.push_back(mid13_ind);
        elem6.push_back(mid23_ind);
        elem6.push_back(mid12_ind);
        subconnectivity.push_back(elem6);
        
        elem7.push_back(mid03_ind);
        elem7.push_back(mid13_ind);
        elem7.push_back(mid23_ind);
        elem7.push_back(mid12_ind);
        subconnectivity.push_back(elem7);
        
        Kokkos::View<int**,AssemblyDevice> oldmap = subsidemap[e];
        Kokkos::View<int**,AssemblyDevice> newsm0("newsi",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm1("newsi",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm2("newsi",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm3("newsi",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm4("newsi",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm5("newsi",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm6("newsi",4,2);
        Kokkos::View<int**,AssemblyDevice> newsm7("newsi",4,2);
        
        Kokkos::deep_copy(newsm0,oldmap);
        Kokkos::deep_copy(newsm1,oldmap);
        Kokkos::deep_copy(newsm2,oldmap);
        Kokkos::deep_copy(newsm3,oldmap);
        Kokkos::deep_copy(newsm4,oldmap);
        Kokkos::deep_copy(newsm5,oldmap);
        Kokkos::deep_copy(newsm6,oldmap);
        Kokkos::deep_copy(newsm7,oldmap);
        
        newsm0(2,0) = 0;
        newsm0(2,1) = 0;
        newsm1(2,0) = 0;
        newsm1(2,1) = 0;
        newsm2(2,0) = 0;
        newsm2(2,1) = 0;
        newsm3(0,0) = 0;
        newsm3(0,1) = 0;
        
        newsm4(1,0) = 0;
        newsm4(1,1) = 0;
        newsm4(2,0) = 0;
        newsm4(2,1) = 0;
        newsm4(3,0) = 0;
        newsm4(3,1) = 0;
        newsm5(2,0) = 0;
        newsm5(2,1) = 0;
      
        subsidemap.push_back(newsm0);
        subsidemap.push_back(newsm1);
        subsidemap.push_back(newsm2);
        subsidemap.push_back(newsm3);
        subsidemap.push_back(newsm4);
        subsidemap.push_back(newsm5);
        subsidemap.push_back(newsm6);
        subsidemap.push_back(newsm7);
        
      }
      else if (subshape == "hex") {
        // Extract the existing nodes
        vector<ScalarT> node0 = subnodes[subconnectivity[e][0]];
        vector<ScalarT> node1 = subnodes[subconnectivity[e][1]];
        vector<ScalarT> node2 = subnodes[subconnectivity[e][2]];
        vector<ScalarT> node3 = subnodes[subconnectivity[e][3]];
        vector<ScalarT> node4 = subnodes[subconnectivity[e][4]];
        vector<ScalarT> node5 = subnodes[subconnectivity[e][5]];
        vector<ScalarT> node6 = subnodes[subconnectivity[e][6]];
        vector<ScalarT> node7 = subnodes[subconnectivity[e][7]];
        
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
          subnodes.push_back(mid0123);
          mid0123_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid01,tol,mid01_ind);
        if (!found) {
          subnodes.push_back(mid01);
          mid01_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid12,tol,mid12_ind);
        if (!found) {
          subnodes.push_back(mid12);
          mid12_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid23,tol,mid23_ind);
        if (!found) {
          subnodes.push_back(mid23);
          mid23_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid03,tol,mid03_ind);
        if (!found) {
          subnodes.push_back(mid03);
          mid03_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(center,tol,center_ind);
        if (!found) {
          subnodes.push_back(center);
          center_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid04,tol,mid04_ind);
        if (!found) {
          subnodes.push_back(mid04);
          mid04_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid15,tol,mid15_ind);
        if (!found) {
          subnodes.push_back(mid15);
          mid15_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid26,tol,mid26_ind);
        if (!found) {
          subnodes.push_back(mid26);
          mid26_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid37,tol,mid37_ind);
        if (!found) {
          subnodes.push_back(mid37);
          mid37_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid0145,tol,mid0145_ind);
        if (!found) {
          subnodes.push_back(mid0145);
          mid0145_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid1256,tol,mid1256_ind);
        if (!found) {
          subnodes.push_back(mid1256);
          mid1256_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid2367,tol,mid2367_ind);
        if (!found) {
          subnodes.push_back(mid2367);
          mid2367_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid0347,tol,mid0347_ind);
        if (!found) {
          subnodes.push_back(mid0347);
          mid0347_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid4567,tol,mid4567_ind);
        if (!found) {
          subnodes.push_back(mid4567);
          mid4567_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid45,tol,mid45_ind);
        if (!found) {
          subnodes.push_back(mid45);
          mid45_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid56,tol,mid56_ind);
        if (!found) {
          subnodes.push_back(mid56);
          mid56_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid67,tol,mid67_ind);
        if (!found) {
          subnodes.push_back(mid67);
          mid67_ind = subnodes.size()-1;
        }
        
        found = checkExistingSubNodes(mid47,tol,mid47_ind);
        if (!found) {
          subnodes.push_back(mid47);
          mid47_ind = subnodes.size()-1;
        }
        
        // Define the new elements (appended to end of list)
        vector<GO> elem0, elem1, elem2, elem3, elem4, elem5, elem6, elem7;
        
        elem0.push_back(subconnectivity[e][0]);
        elem0.push_back(mid01_ind);
        elem0.push_back(mid0123_ind);
        elem0.push_back(mid03_ind);
        elem0.push_back(mid04_ind);
        elem0.push_back(mid0145_ind);
        elem0.push_back(center_ind);
        elem0.push_back(mid0347_ind);
        
        subconnectivity.push_back(elem0);
        
        elem1.push_back(mid01_ind);
        elem1.push_back(subconnectivity[e][1]);
        elem1.push_back(mid12_ind);
        elem1.push_back(mid0123_ind);
        elem1.push_back(mid0145_ind);
        elem1.push_back(mid15_ind);
        elem1.push_back(mid1256_ind);
        elem1.push_back(center_ind);
        
        subconnectivity.push_back(elem1);
        
        elem2.push_back(mid0123_ind);
        elem2.push_back(mid12_ind);
        elem2.push_back(subconnectivity[e][2]);
        elem2.push_back(mid23_ind);
        elem2.push_back(center_ind);
        elem2.push_back(mid1256_ind);
        elem2.push_back(mid26_ind);
        elem2.push_back(mid2367_ind);
        
        subconnectivity.push_back(elem2);
        
        elem3.push_back(mid03_ind);
        elem3.push_back(mid0123_ind);
        elem3.push_back(mid23_ind);
        elem3.push_back(subconnectivity[e][3]);
        elem3.push_back(mid0347_ind);
        elem3.push_back(center_ind);
        elem3.push_back(mid2367_ind);
        elem3.push_back(mid37_ind);
        
        subconnectivity.push_back(elem3);
        
        elem4.push_back(mid04_ind);
        elem4.push_back(mid0145_ind);
        elem4.push_back(center_ind);
        elem4.push_back(mid0347_ind);
        elem4.push_back(subconnectivity[e][4]);
        elem4.push_back(mid45_ind);
        elem4.push_back(mid4567_ind);
        elem4.push_back(mid47_ind);
        
        subconnectivity.push_back(elem4);
        
        elem5.push_back(mid0145_ind);
        elem5.push_back(mid15_ind);
        elem5.push_back(mid1256_ind);
        elem5.push_back(center_ind);
        elem5.push_back(mid45_ind);
        elem5.push_back(subconnectivity[e][5]);
        elem5.push_back(mid56_ind);
        elem5.push_back(mid4567_ind);
        
        subconnectivity.push_back(elem5);
        
        elem6.push_back(center_ind);
        elem6.push_back(mid1256_ind);
        elem6.push_back(mid26_ind);
        elem6.push_back(mid2367_ind);
        elem6.push_back(mid4567_ind);
        elem6.push_back(mid56_ind);
        elem6.push_back(subconnectivity[e][6]);
        elem6.push_back(mid67_ind);
        
        subconnectivity.push_back(elem6);
        
        elem7.push_back(mid0347_ind);
        elem7.push_back(center_ind);
        elem7.push_back(mid2367_ind);
        elem7.push_back(mid37_ind);
        elem7.push_back(mid47_ind);
        elem7.push_back(mid4567_ind);
        elem7.push_back(mid67_ind);
        elem7.push_back(subconnectivity[e][7]);
        
        subconnectivity.push_back(elem7);
        
        Kokkos::View<int**,AssemblyDevice> oldmap = subsidemap[e];
        Kokkos::View<int**,AssemblyDevice> newsm0("newsi",6,2);
        Kokkos::View<int**,AssemblyDevice> newsm1("newsi",6,2);
        Kokkos::View<int**,AssemblyDevice> newsm2("newsi",6,2);
        Kokkos::View<int**,AssemblyDevice> newsm3("newsi",6,2);
        Kokkos::View<int**,AssemblyDevice> newsm4("newsi",6,2);
        Kokkos::View<int**,AssemblyDevice> newsm5("newsi",6,2);
        Kokkos::View<int**,AssemblyDevice> newsm6("newsi",6,2);
        Kokkos::View<int**,AssemblyDevice> newsm7("newsi",6,2);
        
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
        newsm0(1,0) = 0;
        newsm0(1,1) = 0;
        newsm0(2,0) = 0;
        newsm0(2,1) = 0;
        newsm0(5,0) = 0;
        newsm0(5,1) = 0;
        
        newsm1(2,0) = 0;
        newsm1(2,1) = 0;
        newsm1(3,0) = 0;
        newsm1(3,1) = 0;
        newsm1(5,0) = 0;
        newsm1(5,1) = 0;
        
        newsm2(0,0) = 0;
        newsm2(0,1) = 0;
        newsm2(3,0) = 0;
        newsm2(3,1) = 0;
        newsm2(5,0) = 0;
        newsm2(5,1) = 0;
        
        newsm3(0,0) = 0;
        newsm3(0,1) = 0;
        newsm3(1,0) = 0;
        newsm3(1,1) = 0;
        newsm3(5,0) = 0;
        newsm3(5,1) = 0;
        
        newsm4(1,0) = 0;
        newsm4(1,1) = 0;
        newsm4(2,0) = 0;
        newsm4(2,1) = 0;
        newsm4(4,0) = 0;
        newsm4(4,1) = 0;
        
        newsm5(2,0) = 0;
        newsm5(2,1) = 0;
        newsm5(3,0) = 0;
        newsm5(3,1) = 0;
        newsm5(4,0) = 0;
        newsm5(4,1) = 0;
        
        newsm6(0,0) = 0;
        newsm6(0,1) = 0;
        newsm6(3,0) = 0;
        newsm6(3,1) = 0;
        newsm6(4,0) = 0;
        newsm6(4,1) = 0;
        
        newsm7(0,0) = 0;
        newsm7(0,1) = 0;
        newsm7(1,0) = 0;
        newsm7(1,1) = 0;
        newsm7(4,0) = 0;
        newsm7(4,1) = 0;
        
        subsidemap.push_back(newsm0);
        subsidemap.push_back(newsm1);
        subsidemap.push_back(newsm2);
        subsidemap.push_back(newsm3);
        subsidemap.push_back(newsm4);
        subsidemap.push_back(newsm5);
        subsidemap.push_back(newsm6);
        subsidemap.push_back(newsm7);
        
      }
      else {
        // add error
      }
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Check if a sub-grid nodes has already been added to the list
  ///////////////////////////////////////////////////////////////////////////////////////
  
  bool checkExistingSubNodes(const vector<ScalarT> & newpt,
                             const ScalarT & tol, int & index) {
    bool found = false;
    int dimension = newpt.size();
    for (unsigned int i=0; i<subnodes.size(); i++) {
      if (!found) {
        ScalarT val = 0.0;
        for (int j=0; j<dimension; j++) {
          val += (subnodes[i][j]-newpt[j])*(subnodes[i][j]-newpt[j]);
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
  // Get the sub-grid nodes as a list: output is (Nnodes x dimension)
  ///////////////////////////////////////////////////////////////////////////////////////
  
  DRV getListOfPhysicalNodes(DRV newmacronodes, topo_RCP & macro_topo) {
    
    DRV newnodes("nodes on phys elem", newmacronodes.extent(0), subnodes_list.extent(0), dimension);
    CellTools::mapToPhysicalFrame(newnodes, subnodes_list, newmacronodes, *macro_topo);
    
    DRV currnodes("currnodes",newmacronodes.extent(0)*subnodes_list.extent(0), dimension);
    size_t eprog = 0;
    
    for (size_t melem=0; melem<newmacronodes.extent(0); melem++) {
      for (size_t elem=0; elem<subnodes_list.extent(0); elem++) {
        for (size_t dim=0; dim<dimension; dim++) {
          size_t index = melem*subnodes_list.extent(0)+elem;
          currnodes(index,dim) = newnodes(melem,elem,dim);
        }
      }
    }
    
    return currnodes;
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid nodes on each element: output is (Nelem x Nnperelem x dimension)
  ///////////////////////////////////////////////////////////////////////////////////////
  
  DRV getPhysicalNodes(DRV newmacronodes, topo_RCP & macro_topo) {
    
    DRV newnodes("nodes on phys elem", newmacronodes.extent(0), subnodes_list.extent(0), subnodes_list.extent(1));
    CellTools::mapToPhysicalFrame(newnodes, subnodes_list, newmacronodes, *macro_topo);
    
    DRV currnodes("currnodes",newmacronodes.extent(0)*subconnectivity.size(),
                  subconnectivity[0].size(),
                  dimension);
    size_t eprog = 0;
    
    for (size_t melem=0; melem<newmacronodes.extent(0); melem++) {
      for (size_t elem=0; elem<subconnectivity.size(); elem++) {
        for (size_t node=0; node<subconnectivity[elem].size(); node++) {
          for (size_t dim=0; dim<dimension; dim++) {
            size_t index = melem*subconnectivity.size()+elem;
            currnodes(index,node,dim) = newnodes(melem,subconnectivity[elem][node],dim);
          }
        }
      }
    }
  
    return currnodes;
    
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid side info
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<int****,HostDevice> getPhysicalSideinfo(Kokkos::View<int****,HostDevice> macrosideinfo) {
    Kokkos::View<int****,HostDevice> ksubsideinfo("subgrid side info",
                                                  macrosideinfo.extent(0)*subconnectivity.size(), // macroelem*subelem
                                                  sideinfo.extent(1),
                                                  sideinfo.extent(2),
                                                  2);
    
    int prog = 0;
    for (unsigned int k=0; k<macrosideinfo.extent(0); k++) {
      for (unsigned int e=0; e<subsidemap.size(); e++) {
        for (unsigned int j=0; j<subsidemap[e].extent(0); j++) {
          if (subsidemap[e](j,0)>0) {
            int sideindex = subsidemap[e](j,1);
            for (unsigned int i=0; i<ksubsideinfo.extent(1); i++) {
              if (macrosideinfo(k,i,sideindex,0)>1) {
                ksubsideinfo(e+prog,i,j,0) = macrosideinfo(k,i,sideindex,0);
                ksubsideinfo(e+prog,i,j,1) = macrosideinfo(k,i,sideindex,1);
              }
              else { // default to weak Dirichlet
                ksubsideinfo(e+prog,i,j,0) = 5;
                ksubsideinfo(e+prog,i,j,1) = -1;
              }
            }
          }
        }
      }
      prog += subsidemap.size();
    }
    
    return ksubsideinfo;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid connectivity
  ///////////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<GO> > getPhysicalConnectivity(int & reps) {
    vector<vector<GO> > newconn;
    
    int prog = 0;
    for (int k=0; k<reps; k++) {
      for (int i=0; i<subconnectivity.size(); i++) {
        vector<GO> cc;
        for (int j=0; j<subconnectivity[i].size(); j++) {
          cc.push_back(subconnectivity[i][j]+prog);
        }
        newconn.push_back(cc);
      }
      prog += subnodes_list.extent(0);//.size();
    }
    return newconn;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the unique subgrid side names and indices
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void getUniqueSides(Kokkos::View<int****,HostDevice> & newsi, vector<int> & unique_sides,
                      vector<int> & unique_local_sides, vector<string> & unique_names,
                      vector<string> & macrosidenames,
                      vector<vector<size_t> > & boundary_groups) {
    
    for (size_t c=0; c<newsi.extent(0); c++) { // number of elem in cell
      for (size_t i=0; i<newsi.extent(1); i++) { // number of variables
        for (size_t j=0; j<newsi.extent(2); j++) { // number of sides per element
          if (newsi(c,i,j,0) > 0) {
            bool found = false;
            for (size_t s=0; s<unique_sides.size(); s++) {
              if (newsi(c,i,j,1) == unique_sides[s]) {
                if (j == unique_local_sides[s]) {
                  found = true;
                }
              }
            }
            if (!found) {
              unique_sides.push_back(newsi(c,i,j,1));
              unique_local_sides.push_back(j);
              if (newsi(c,i,j,1) == -1) {
                unique_names.push_back("interior");
              }
              else {
                unique_names.push_back(macrosidenames[newsi(c,i,j,1)]);
              }
            }
          }
        }
      }
    }
    
    for (size_t s=0; s<unique_sides.size(); s++) {
      
      int clside = unique_local_sides[s];
      string sidename = unique_names[s];
      vector<size_t> group;
      for (size_t c=0; c<newsi.extent(0); c++) {
        bool addthis = false;
        for (size_t i=0; i<newsi.extent(1); i++) { // number of variables
          if (newsi(c,i,clside,0) > 0) {
            if (newsi(c,i,clside,1) == unique_sides[s]) {
              addthis = true;
            }
          }
        }
        if (addthis) {
          group.push_back(c);
        }
      }
      boundary_groups.push_back(group);
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  int dimension;
  Teuchos::RCP<MpiComm> LocalComm;
  string shape, subshape, mesh_type, mesh_file;
  DRV nodes;
  Kokkos::View<int****,HostDevice> sideinfo;
  vector<vector<ScalarT> > subnodes;
  DRV subnodes_list;
  vector<Kokkos::View<int**,AssemblyDevice> > subsidemap;
  vector<vector<GO> > subconnectivity;
  
  Teuchos::RCP<panzer_stk::STK_Interface> ref_mesh; // used for Exodus and panzer meshes
  
};
#endif

