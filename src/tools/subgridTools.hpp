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


class SubGridTools {
public:
  
  SubGridTools() {} ;
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  SubGridTools(const Teuchos::RCP<MpiComm> & LocalComm_, const string & shape_,
               const string & subshape_, const DRV nodes_, Kokkos::View<int****,HostDevice> sideinfo_) :
  LocalComm(LocalComm_), shape(shape_), subshape(subshape_), nodes(nodes_), sideinfo(sideinfo_) {
    
    dimension = nodes.extent(2);
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Given the coarse grid nodes and shape, define the subgrid nodes, connectivity, and sideinfo
  //////////////////////////////////////////////////////////////////////////////////////
  
  void createSubMesh(const int & numrefine) {
    
    if (subshape == shape) {
      vector<GO> newconn;
      for (unsigned int i=0; i<nodes.extent(1); i++) {
        vector<ScalarT> newnode;
        for (unsigned int s=0; s<dimension; s++) {
          newnode.push_back(nodes(0,i,s));
        }
        subnodes.push_back(newnode);
        newconn.push_back(i);
        
        Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
        for (unsigned int j=0; j<nodes.extent(1); j++) {
          if (i==j) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 1.0;
            }
          }
        }
        subnodemap.push_back(newmap);
      }
      subconnectivity.push_back(newconn);
      
      Kokkos::View<int****,AssemblyDevice> newsi("newsi",1,sideinfo.extent(1),sideinfo.extent(2),2);
      for (size_t n=0; n<sideinfo.extent(1); n++) {
        for (size_t s=0; s<sideinfo.extent(2); s++) {
          if (sideinfo(0,n,s,0) > 0) {
            newsi(0,n,s,0) = sideinfo(0,n,s,0);
            newsi(0,n,s,1) = sideinfo(0,n,s,1);
          }
          else {
            newsi(0,n,s,0) = 1;
            newsi(0,n,s,1) = -1;
          }
        }
      }
      subsideinfo.push_back(newsi);
      
      Kokkos::View<int***,AssemblyDevice> newsidemap("newsidemap",sideinfo.extent(1),sideinfo.extent(2),2);
      for (size_t n=0; n<sideinfo.extent(1); n++) {
        for (size_t s=0; s<sideinfo.extent(2); s++) {
          //if (sideinfo(0,n,s,0) > 0) {
            newsidemap(n,s,0) = 1;
            newsidemap(n,s,1) = s;
          //}
          //else {
          //  newsidemap(n,s,0) = 1;
          //  newsidemap(n,s,1) = -1;
          //}
        }
      }
      subsidemap.push_back(newsidemap);
      
    }
    else {
      if (dimension == 1) {
        // output an error message
      }
      else if (dimension == 2) {
        if (shape == "quad" && subshape == "tri") {
          vector<GO> newconn0, newconn1, newconn2, newconn3;
          for (unsigned int i=0; i<nodes.extent(1); i++) {
            vector<ScalarT> newnode;
            for (unsigned int s=0; s<dimension; s++) {
              newnode.push_back(nodes(0,i,s));
            }
            subnodes.push_back(newnode);
            
            Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
            for (unsigned int j=0; j<nodes.extent(1); j++) {
              if (i==j) {
                for (unsigned int s=0; s<dimension; s++) {
                  newmap(j,s) = 1.0;
                }
              }
            }
            subnodemap.push_back(newmap);
          }
          vector<ScalarT> midnode;
          midnode.push_back(0.25*(nodes(0,0,0)+nodes(0,1,0)+nodes(0,2,0)+nodes(0,3,0)));
          midnode.push_back(0.25*(nodes(0,0,1)+nodes(0,1,1)+nodes(0,2,1)+nodes(0,3,1)));
          subnodes.push_back(midnode);
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.25;
            }
          }
          subnodemap.push_back(newmap);
          
          newconn0.push_back(0);
          newconn0.push_back(1);
          newconn0.push_back(4);
          subconnectivity.push_back(newconn0);
          
          newconn1.push_back(1);
          newconn1.push_back(2);
          newconn1.push_back(4);
          subconnectivity.push_back(newconn1);
          
          newconn2.push_back(2);
          newconn2.push_back(3);
          newconn2.push_back(4);
          subconnectivity.push_back(newconn2);
          
          newconn3.push_back(3);
          newconn3.push_back(0);
          newconn3.push_back(4);
          subconnectivity.push_back(newconn3);
          
          Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,sideinfo.extent(1),3,2);
          Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,sideinfo.extent(1),3,2);
          Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,sideinfo.extent(1),3,2);
          Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,sideinfo.extent(1),3,2);
          
          for (size_t n=0; n<sideinfo.extent(1); n++) {
            
            newsi0(0,n,0,0) = 1;
            if (sideinfo(0,n,0,0) > 0)
              newsi0(0,n,0,1) = sideinfo(0,n,0,1);
            else
              newsi0(0,n,0,1) = -1;
            
            newsi1(0,n,0,0) = 1;
            if (sideinfo(0,n,1,0) > 0)
              newsi1(0,n,0,1) = sideinfo(0,n,1,1);
            else
              newsi1(0,n,0,1) = -1;
            
            newsi2(0,n,0,0) = 1;
            if (sideinfo(0,n,2,0) > 0)
              newsi2(0,n,0,1) = sideinfo(0,n,2,1);
            else
              newsi2(0,n,0,1) = -1;
            
            newsi3(0,n,0,0) = 1;
            if (sideinfo(0,n,3,0) > 0)
              newsi3(0,n,0,1) = sideinfo(0,n,3,1);
            else
              newsi3(0,n,0,1) = -1;
          }
          subsideinfo.push_back(newsi0);
          subsideinfo.push_back(newsi1);
          subsideinfo.push_back(newsi2);
          subsideinfo.push_back(newsi3);
          
          Kokkos::View<int***,AssemblyDevice> newsidemap0("newsi",sideinfo.extent(1),3,2);
          Kokkos::View<int***,AssemblyDevice> newsidemap1("newsi",sideinfo.extent(1),3,2);
          Kokkos::View<int***,AssemblyDevice> newsidemap2("newsi",sideinfo.extent(1),3,2);
          Kokkos::View<int***,AssemblyDevice> newsidemap3("newsi",sideinfo.extent(1),3,2);
          
          for (size_t n=0; n<sideinfo.extent(1); n++) {
            
            newsidemap0(n,0,0) = 1;
            if (sideinfo(0,n,0,0) > 0)
              newsidemap0(n,0,1) = 0;
            else
              newsidemap0(n,0,1) = -1;
            
            newsidemap1(n,0,0) = 1;
            if (sideinfo(0,n,1,0) > 0)
              newsidemap1(n,0,1) = 1;
            else
              newsidemap1(n,0,1) = -1;
            
            newsidemap2(n,0,0) = 1;
            if (sideinfo(0,n,2,0) > 0)
              newsidemap2(n,0,1) = 2;
            else
              newsidemap2(n,0,1) = -1;
            
            newsidemap3(n,0,0) = 1;
            if (sideinfo(0,n,3,0) > 0)
              newsidemap3(n,0,1) = 3;
            else
              newsidemap3(n,0,1) = -1;
          }
          subsidemap.push_back(newsidemap0);
          subsidemap.push_back(newsidemap1);
          subsidemap.push_back(newsidemap2);
          subsidemap.push_back(newsidemap3);
          
        }
        else if (shape == "tri" && subshape == "quad") {
          vector<GO> newconn0, newconn1, newconn2;
          for (unsigned int i=0; i<nodes.extent(1); i++) {
            vector<ScalarT> newnode;
            
            for (unsigned int s=0; s<dimension; s++) {
              newnode.push_back(nodes(0,i,s));
            }
            subnodes.push_back(newnode);
          
            Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
            for (unsigned int j=0; j<nodes.extent(1); j++) {
              if (i==j) {
                for (unsigned int s=0; s<dimension; s++) {
                  newmap(j,s) = 1.0;
                }
              }
            }
            subnodemap.push_back(newmap);
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
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmapc("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmapc(j,s) = 1.0/3.0;
            }
          }
          subnodemap.push_back(newmapc);
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmapm01("node map", nodes.extent(1), dimension);
          for (unsigned int s=0; s<dimension; s++) {
            newmapm01(0,s) = 0.5;
            newmapm01(1,s) = 0.5;
          }
          subnodemap.push_back(newmapm01);
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmapm12("node map", nodes.extent(1), dimension);
          for (unsigned int s=0; s<dimension; s++) {
            newmapm12(1,s) = 0.5;
            newmapm12(2,s) = 0.5;
          }
          subnodemap.push_back(newmapm12);
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmapm02("node map", nodes.extent(1), dimension);
          for (unsigned int s=0; s<dimension; s++) {
            newmapm02(0,s) = 0.5;
            newmapm02(2,s) = 0.5;
          }
          subnodemap.push_back(newmapm02);
          
          newconn0.push_back(0);
          newconn0.push_back(4);
          newconn0.push_back(3);
          newconn0.push_back(6);
          subconnectivity.push_back(newconn0);
          
          newconn1.push_back(1);
          newconn1.push_back(5);
          newconn1.push_back(3);
          newconn1.push_back(4);
          subconnectivity.push_back(newconn1);
          
          newconn2.push_back(2);
          newconn2.push_back(6);
          newconn2.push_back(3);
          newconn2.push_back(5);
          subconnectivity.push_back(newconn2);
          
          Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,sideinfo.extent(1),4,2);
          Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,sideinfo.extent(1),4,2);
          Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,sideinfo.extent(1),4,2);
          
          for (size_t n=0; n<sideinfo.extent(1); n++) {
            
            newsi0(0,n,0,0) = 1;
            if (sideinfo(0,n,0,0) > 0)
              newsi0(0,n,0,1) = sideinfo(0,n,0,1);
            else
              newsi0(0,n,0,1) = -1;
            
            newsi0(0,n,3,0) = 1;
            if (sideinfo(0,n,2,0) > 0)
              newsi0(0,n,3,1) = sideinfo(0,n,2,1);
            else
              newsi0(0,n,3,1) = -1;
            
            newsi1(0,n,0,0) = 1;
            if (sideinfo(0,n,1,0) > 0)
              newsi1(0,n,0,1) = sideinfo(0,n,1,1);
            else
              newsi1(0,n,0,1) = -1;
            
            newsi1(0,n,3,0) = 1;
            if (sideinfo(0,n,0,0) > 0)
              newsi1(0,n,3,1) = sideinfo(0,n,0,1);
            else
              newsi1(0,n,3,1) = -1;
            
            newsi2(0,n,0,0) = 1;
            if (sideinfo(0,n,2,0) > 0)
              newsi2(0,n,0,1) = sideinfo(0,n,2,1);
            else
              newsi2(0,n,0,1) = -1;
            
            newsi2(0,n,3,0) = 1;
            if (sideinfo(0,n,1,0) > 0)
              newsi2(0,n,3,1) = sideinfo(0,n,1,1);
            else
              newsi2(0,n,3,1) = -1;
            
          }
          subsideinfo.push_back(newsi0);
          subsideinfo.push_back(newsi1);
          subsideinfo.push_back(newsi2);
          
          Kokkos::View<int***,AssemblyDevice> newsidemap0("newsi",sideinfo.extent(1),4,2);
          Kokkos::View<int***,AssemblyDevice> newsidemap1("newsi",sideinfo.extent(1),4,2);
          Kokkos::View<int***,AssemblyDevice> newsidemap2("newsi",sideinfo.extent(1),4,2);
          
          for (size_t n=0; n<sideinfo.extent(1); n++) {
            
            newsidemap0(n,0,0) = 1;
            if (sideinfo(0,n,0,0) > 0)
              newsidemap0(n,0,1) = 0;
            else
              newsidemap0(n,0,1) = -1;
            
            newsidemap0(n,3,0) = 1;
            if (sideinfo(0,n,2,0) > 0)
              newsidemap0(n,3,1) = 2;
            else
              newsidemap0(n,3,1) = -1;
            
            newsidemap1(n,0,0) = 1;
            if (sideinfo(0,n,1,0) > 0)
              newsidemap1(n,0,1) = 1;
            else
              newsidemap1(n,0,1) = -1;
            
            newsidemap1(n,3,0) = 1;
            if (sideinfo(0,n,0,0) > 0)
              newsidemap1(n,3,1) = 0;
            else
              newsidemap1(n,3,1) = -1;
            
            newsidemap2(n,0,0) = 1;
            if (sideinfo(0,n,2,0) > 0)
              newsidemap2(n,0,1) = 2;
            else
              newsidemap2(n,0,1) = -1;
            
            newsidemap2(n,3,0) = 1;
            if (sideinfo(0,n,1,0) > 0)
              newsidemap2(n,3,1) = 1;
            else
              newsidemap2(n,3,1) = -1;
            
          }
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
      subsideinfo.erase(subsideinfo.begin(), subsideinfo.begin()+numelem);
      subsidemap.erase(subsidemap.begin(), subsidemap.begin()+numelem);
    }
    
    //this->checkNodeMap();
    //this->checkSideMap();
    
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
      
      Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
      for (unsigned int s=0; s<dimension; s++) {
        newmap(0,s) = 0.5;
        newmap(1,s) = 0.5;
      }
      subnodemap.push_back(newmap);
      
      vector<GO> newelem0, newelem1;
      newelem0.push_back(subconnectivity[e][0]);
      newelem0.push_back(subnodes.size());
      newelem1.push_back(subnodes.size());
      newelem1.push_back(subconnectivity[e][1]);
      
      subconnectivity.push_back(newelem0);
      subconnectivity.push_back(newelem1);
      
      
      Kokkos::View<int****,AssemblyDevice> oldsi = subsideinfo[e];
      Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.extent(1),2,2);
      Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.extent(1),2,2);
      Kokkos::deep_copy(newsi0,oldsi);
      Kokkos::deep_copy(newsi1,oldsi);
      
      for (unsigned int n=0; n<oldsi.extent(1); n++) {
        newsi0(0,n,1,0) = 0;
        newsi0(0,n,1,1) = 0;
        newsi1(0,n,0,0) = 0;
        newsi1(0,n,0,1) = 0;
      }
      subsideinfo.push_back(newsi0);
      subsideinfo.push_back(newsi1);
      
      Kokkos::View<int***,AssemblyDevice> oldmap = subsidemap[e];
      Kokkos::View<int***,AssemblyDevice> newsm0("newsi",oldmap.extent(0),2,2);
      Kokkos::View<int***,AssemblyDevice> newsm1("newsi",oldmap.extent(0),2,2);
      Kokkos::deep_copy(newsm0,oldmap);
      Kokkos::deep_copy(newsm1,oldmap);
      
      for (unsigned int n=0; n<oldmap.extent(0); n++) {
        newsm0(n,1,0) = 0;
        newsm0(n,1,1) = 0;
        newsm1(n,0,0) = 0;
        newsm1(n,0,1) = 0;
      }
      subsidemap.push_back(newsm0);
      subsidemap.push_back(newsm1);
      
    }
    if (dimension == 2) {
      if (subshape == "tri") {
        // Extract the existing nodes
        vector<ScalarT> node0 = subnodes[subconnectivity[e][0]];
        vector<ScalarT> node1 = subnodes[subconnectivity[e][1]];
        vector<ScalarT> node2 = subnodes[subconnectivity[e][2]];
        
        Kokkos::View<ScalarT**,AssemblyDevice> map0 = subnodemap[subconnectivity[e][0]];
        Kokkos::View<ScalarT**,AssemblyDevice> map1 = subnodemap[subconnectivity[e][1]];
        Kokkos::View<ScalarT**,AssemblyDevice> map2 = subnodemap[subconnectivity[e][2]];
        
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
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map0(j,s) + 0.5*map1(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid12,tol,mid12_ind);
        if (!found) {
          subnodes.push_back(mid12);
          mid12_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map1(j,s) + 0.5*map2(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid02,tol,mid02_ind);
        if (!found) {
          subnodes.push_back(mid02);
          mid02_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map0(j,s) + 0.5*map2(j,s);
            }
          }
          subnodemap.push_back(newmap);
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
        
        
        Kokkos::View<int****,AssemblyDevice> oldsi = subsideinfo[e];
        Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.extent(1),3,2);
        Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.extent(1),3,2);
        Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,oldsi.extent(1),3,2);
        Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,oldsi.extent(1),3,2);
        Kokkos::deep_copy(newsi0,oldsi);
        Kokkos::deep_copy(newsi1,oldsi);
        Kokkos::deep_copy(newsi2,oldsi);
        Kokkos::deep_copy(newsi3,oldsi);
        
        for (unsigned int n=0; n<oldsi.extent(1); n++) {
          newsi0(0,n,1,0) = 0;
          newsi0(0,n,1,1) = 0;
          newsi1(0,n,1,0) = 0;
          newsi1(0,n,1,1) = 0;
          newsi2(0,n,1,0) = 0;
          newsi2(0,n,1,1) = 0;
          newsi3(0,n,0,0) = 0;
          newsi3(0,n,0,1) = 0;
          newsi3(0,n,1,0) = 0;
          newsi3(0,n,1,1) = 0;
          newsi3(0,n,2,0) = 0;
          newsi3(0,n,2,1) = 0;
        }
        subsideinfo.push_back(newsi0);
        subsideinfo.push_back(newsi1);
        subsideinfo.push_back(newsi2);
        subsideinfo.push_back(newsi3);
        
        Kokkos::View<int***,AssemblyDevice> oldmap = subsidemap[e];
        Kokkos::View<int***,AssemblyDevice> newsm0("newsi",oldmap.extent(0),3,2);
        Kokkos::View<int***,AssemblyDevice> newsm1("newsi",oldmap.extent(0),3,2);
        Kokkos::View<int***,AssemblyDevice> newsm2("newsi",oldmap.extent(0),3,2);
        Kokkos::View<int***,AssemblyDevice> newsm3("newsi",oldmap.extent(0),3,2);
        Kokkos::deep_copy(newsm0,oldmap);
        Kokkos::deep_copy(newsm1,oldmap);
        Kokkos::deep_copy(newsm2,oldmap);
        Kokkos::deep_copy(newsm3,oldmap);
        
        for (unsigned int n=0; n<oldmap.extent(0); n++) {
          newsm0(n,1,0) = 0;
          newsm0(n,1,1) = 0;
          newsm1(n,1,0) = 0;
          newsm1(n,1,1) = 0;
          newsm2(n,1,0) = 0;
          newsm2(n,1,1) = 0;
          newsm3(n,0,0) = 0;
          newsm3(n,0,1) = 0;
          newsm3(n,1,0) = 0;
          newsm3(n,1,1) = 0;
          newsm3(n,2,0) = 0;
          newsm3(n,2,1) = 0;
        }
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
        
        Kokkos::View<ScalarT**,AssemblyDevice> map0 = subnodemap[subconnectivity[e][0]];
        Kokkos::View<ScalarT**,AssemblyDevice> map1 = subnodemap[subconnectivity[e][1]];
        Kokkos::View<ScalarT**,AssemblyDevice> map2 = subnodemap[subconnectivity[e][2]];
        Kokkos::View<ScalarT**,AssemblyDevice> map3 = subnodemap[subconnectivity[e][3]];
        
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
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.25*map0(j,s) + 0.25*map1(j,s) + 0.25*map2(j,s) + 0.25*map3(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid01,tol,mid01_ind);
        if (!found) {
          subnodes.push_back(mid01);
          mid01_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map0(j,s) + 0.5*map1(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid12,tol,mid12_ind);
        if (!found) {
          subnodes.push_back(mid12);
          mid12_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map1(j,s) + 0.5*map2(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid23,tol,mid23_ind);
        if (!found) {
          subnodes.push_back(mid23);
          mid23_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map2(j,s) + 0.5*map3(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid03,tol,mid03_ind);
        if (!found) {
          subnodes.push_back(mid03);
          mid03_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map0(j,s) + 0.5*map3(j,s);
            }
          }
          subnodemap.push_back(newmap);
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
        
        
        Kokkos::View<int****,AssemblyDevice> oldsi = subsideinfo[e];
        Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,oldsi.extent(1),4,2);
        Kokkos::deep_copy(newsi0,oldsi);
        Kokkos::deep_copy(newsi1,oldsi);
        Kokkos::deep_copy(newsi2,oldsi);
        Kokkos::deep_copy(newsi3,oldsi);
        
        for (unsigned int n=0; n<oldsi.extent(1); n++) {
          newsi0(0,n,1,0) = 0;
          newsi0(0,n,1,1) = 0;
          newsi0(0,n,2,0) = 0;
          newsi0(0,n,2,1) = 0;
          
          newsi1(0,n,2,0) = 0;
          newsi1(0,n,2,1) = 0;
          newsi1(0,n,3,0) = 0;
          newsi1(0,n,3,1) = 0;
          
          newsi2(0,n,0,0) = 0;
          newsi2(0,n,0,1) = 0;
          newsi2(0,n,3,0) = 0;
          newsi2(0,n,3,1) = 0;
          
          newsi3(0,n,0,0) = 0;
          newsi3(0,n,0,1) = 0;
          newsi3(0,n,1,0) = 0;
          newsi3(0,n,1,1) = 0;
        }
        subsideinfo.push_back(newsi0);
        subsideinfo.push_back(newsi1);
        subsideinfo.push_back(newsi2);
        subsideinfo.push_back(newsi3);
        
        Kokkos::View<int***,AssemblyDevice> oldmap = subsidemap[e];
        Kokkos::View<int***,AssemblyDevice> newsm0("newsm",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm1("newsm",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm2("newsm",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm3("newsm",oldmap.extent(0),4,2);
        Kokkos::deep_copy(newsm0,oldmap);
        Kokkos::deep_copy(newsm1,oldmap);
        Kokkos::deep_copy(newsm2,oldmap);
        Kokkos::deep_copy(newsm3,oldmap);
        
        for (unsigned int n=0; n<oldmap.extent(0); n++) {
          newsm0(n,1,0) = 0;
          newsm0(n,1,1) = 0;
          newsm0(n,2,0) = 0;
          newsm0(n,2,1) = 0;
          
          newsm1(n,2,0) = 0;
          newsm1(n,2,1) = 0;
          newsm1(n,3,0) = 0;
          newsm1(n,3,1) = 0;
          
          newsm2(n,0,0) = 0;
          newsm2(n,0,1) = 0;
          newsm2(n,3,0) = 0;
          newsm2(n,3,1) = 0;
          
          newsm3(n,0,0) = 0;
          newsm3(n,0,1) = 0;
          newsm3(n,1,0) = 0;
          newsm3(n,1,1) = 0;
        }
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
        
        Kokkos::View<ScalarT**,AssemblyDevice> map0 = subnodemap[subconnectivity[e][0]];
        Kokkos::View<ScalarT**,AssemblyDevice> map1 = subnodemap[subconnectivity[e][1]];
        Kokkos::View<ScalarT**,AssemblyDevice> map2 = subnodemap[subconnectivity[e][2]];
        Kokkos::View<ScalarT**,AssemblyDevice> map3 = subnodemap[subconnectivity[e][3]];
        
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
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map0(j,s) + 0.5*map1(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid12,tol,mid12_ind);
        if (!found) {
          subnodes.push_back(mid12);
          mid12_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map1(j,s) + 0.5*map2(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid02,tol,mid02_ind);
        if (!found) {
          subnodes.push_back(mid02);
          mid02_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map0(j,s) + 0.5*map2(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid03,tol,mid03_ind);
        if (!found) {
          subnodes.push_back(mid03);
          mid03_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map0(j,s) + 0.5*map3(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid13,tol,mid13_ind);
        if (!found) {
          subnodes.push_back(mid13);
          mid13_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map1(j,s) + 0.5*map3(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid23,tol,mid23_ind);
        if (!found) {
          subnodes.push_back(mid23);
          mid23_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map2(j,s) + 0.5*map3(j,s);
            }
          }
          subnodemap.push_back(newmap);
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
        
        Kokkos::View<int****,AssemblyDevice> oldsi = subsideinfo[e];
        Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi4("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi5("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi6("newsi",1,oldsi.extent(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi7("newsi",1,oldsi.extent(1),4,2);
        
        Kokkos::deep_copy(newsi0,oldsi);
        Kokkos::deep_copy(newsi1,oldsi);
        Kokkos::deep_copy(newsi2,oldsi);
        Kokkos::deep_copy(newsi3,oldsi);
        Kokkos::deep_copy(newsi4,oldsi);
        Kokkos::deep_copy(newsi5,oldsi);
        Kokkos::deep_copy(newsi6,oldsi);
        Kokkos::deep_copy(newsi7,oldsi);
        
        for (unsigned int n=0; n<oldsi.extent(1); n++) {
          newsi0(0,n,2,0) = 0;
          newsi0(0,n,2,1) = 0;
          newsi1(0,n,2,0) = 0;
          newsi1(0,n,2,1) = 0;
          newsi2(0,n,2,0) = 0;
          newsi2(0,n,2,1) = 0;
          newsi3(0,n,0,0) = 0;
          newsi3(0,n,0,1) = 0;
          
          newsi4(0,n,1,0) = 0;
          newsi4(0,n,1,1) = 0;
          newsi4(0,n,2,0) = 0;
          newsi4(0,n,2,1) = 0;
          newsi4(0,n,3,0) = 0;
          newsi4(0,n,3,1) = 0;
          newsi5(0,n,2,0) = 0;
          newsi5(0,n,2,1) = 0;
        }
        subsideinfo.push_back(newsi0);
        subsideinfo.push_back(newsi1);
        subsideinfo.push_back(newsi2);
        subsideinfo.push_back(newsi3);
        subsideinfo.push_back(newsi4);
        subsideinfo.push_back(newsi5);
        subsideinfo.push_back(newsi6);
        subsideinfo.push_back(newsi7);
        
        Kokkos::View<int***,AssemblyDevice> oldmap = subsidemap[e];
        Kokkos::View<int***,AssemblyDevice> newsm0("newsi",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm1("newsi",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm2("newsi",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm3("newsi",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm4("newsi",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm5("newsi",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm6("newsi",oldmap.extent(0),4,2);
        Kokkos::View<int***,AssemblyDevice> newsm7("newsi",oldmap.extent(0),4,2);
        
        Kokkos::deep_copy(newsm0,oldmap);
        Kokkos::deep_copy(newsm1,oldmap);
        Kokkos::deep_copy(newsm2,oldmap);
        Kokkos::deep_copy(newsm3,oldmap);
        Kokkos::deep_copy(newsm4,oldmap);
        Kokkos::deep_copy(newsm5,oldmap);
        Kokkos::deep_copy(newsm6,oldmap);
        Kokkos::deep_copy(newsm7,oldmap);
        
        for (unsigned int n=0; n<oldmap.extent(0); n++) {
          newsm0(n,2,0) = 0;
          newsm0(n,2,1) = 0;
          newsm1(n,2,0) = 0;
          newsm1(n,2,1) = 0;
          newsm2(n,2,0) = 0;
          newsm2(n,2,1) = 0;
          newsm3(n,0,0) = 0;
          newsm3(n,0,1) = 0;
          
          newsm4(n,1,0) = 0;
          newsm4(n,1,1) = 0;
          newsm4(n,2,0) = 0;
          newsm4(n,2,1) = 0;
          newsm4(n,3,0) = 0;
          newsm4(n,3,1) = 0;
          newsm5(n,2,0) = 0;
          newsm5(n,2,1) = 0;
        }
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
        
        Kokkos::View<ScalarT**,AssemblyDevice> map0 = subnodemap[subconnectivity[e][0]];
        Kokkos::View<ScalarT**,AssemblyDevice> map1 = subnodemap[subconnectivity[e][1]];
        Kokkos::View<ScalarT**,AssemblyDevice> map2 = subnodemap[subconnectivity[e][2]];
        Kokkos::View<ScalarT**,AssemblyDevice> map3 = subnodemap[subconnectivity[e][3]];
        Kokkos::View<ScalarT**,AssemblyDevice> map4 = subnodemap[subconnectivity[e][4]];
        Kokkos::View<ScalarT**,AssemblyDevice> map5 = subnodemap[subconnectivity[e][5]];
        Kokkos::View<ScalarT**,AssemblyDevice> map6 = subnodemap[subconnectivity[e][6]];
        Kokkos::View<ScalarT**,AssemblyDevice> map7 = subnodemap[subconnectivity[e][7]];
        
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
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.25*map0(j,s) + 0.25*map1(j,s) + 0.25*map2(j,s) + 0.25*map3(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid01,tol,mid01_ind);
        if (!found) {
          subnodes.push_back(mid01);
          mid01_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map0(j,s) + 0.5*map1(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid12,tol,mid12_ind);
        if (!found) {
          subnodes.push_back(mid12);
          mid12_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map1(j,s) + 0.5*map2(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid23,tol,mid23_ind);
        if (!found) {
          subnodes.push_back(mid23);
          mid23_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map2(j,s) + 0.5*map3(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid03,tol,mid03_ind);
        if (!found) {
          subnodes.push_back(mid03);
          mid03_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map2(j,s) + 0.5*map3(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(center,tol,center_ind);
        if (!found) {
          subnodes.push_back(center);
          center_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.125*map0(j,s) + 0.125*map1(j,s) + 0.125*map2(j,s) + 0.125*map3(j,s)
                          + 0.125*map4(j,s) + 0.125*map5(j,s) + 0.125*map6(j,s) + 0.125*map7(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid04,tol,mid04_ind);
        if (!found) {
          subnodes.push_back(mid04);
          mid04_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map0(j,s) + 0.5*map4(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid15,tol,mid15_ind);
        if (!found) {
          subnodes.push_back(mid15);
          mid15_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map1(j,s) + 0.5*map5(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid26,tol,mid26_ind);
        if (!found) {
          subnodes.push_back(mid26);
          mid26_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map2(j,s) + 0.5*map6(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid37,tol,mid37_ind);
        if (!found) {
          subnodes.push_back(mid37);
          mid37_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map3(j,s) + 0.5*map7(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid0145,tol,mid0145_ind);
        if (!found) {
          subnodes.push_back(mid0145);
          mid0145_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.25*map0(j,s) + 0.25*map1(j,s) + 0.25*map4(j,s) + 0.25*map5(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid1256,tol,mid1256_ind);
        if (!found) {
          subnodes.push_back(mid1256);
          mid1256_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.25*map1(j,s) + 0.25*map2(j,s) + 0.25*map5(j,s) + 0.25*map6(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid2367,tol,mid2367_ind);
        if (!found) {
          subnodes.push_back(mid2367);
          mid2367_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.25*map2(j,s) + 0.25*map3(j,s) + 0.25*map6(j,s) + 0.25*map7(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid0347,tol,mid0347_ind);
        if (!found) {
          subnodes.push_back(mid0347);
          mid0347_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.25*map0(j,s) + 0.25*map3(j,s) + 0.25*map4(j,s) + 0.25*map7(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid4567,tol,mid4567_ind);
        if (!found) {
          subnodes.push_back(mid4567);
          mid4567_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.25*map4(j,s) + 0.25*map5(j,s) + 0.25*map6(j,s) + 0.25*map7(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid45,tol,mid45_ind);
        if (!found) {
          subnodes.push_back(mid45);
          mid45_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map4(j,s) + 0.5*map5(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid56,tol,mid56_ind);
        if (!found) {
          subnodes.push_back(mid56);
          mid56_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map5(j,s) + 0.5*map6(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid67,tol,mid67_ind);
        if (!found) {
          subnodes.push_back(mid67);
          mid67_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map6(j,s) + 0.5*map7(j,s);
            }
          }
          subnodemap.push_back(newmap);
        }
        
        found = checkExistingSubNodes(mid47,tol,mid47_ind);
        if (!found) {
          subnodes.push_back(mid47);
          mid47_ind = subnodes.size()-1;
          
          Kokkos::View<ScalarT**,AssemblyDevice> newmap("node map", nodes.extent(1), dimension);
          for (unsigned int j=0; j<nodes.extent(1); j++) {
            for (unsigned int s=0; s<dimension; s++) {
              newmap(j,s) = 0.5*map4(j,s) + 0.5*map7(j,s);
            }
          }
          subnodemap.push_back(newmap);
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
        
        Kokkos::View<int****,AssemblyDevice> oldsi = subsideinfo[e];
        Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.extent(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.extent(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,oldsi.extent(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,oldsi.extent(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi4("newsi",1,oldsi.extent(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi5("newsi",1,oldsi.extent(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi6("newsi",1,oldsi.extent(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi7("newsi",1,oldsi.extent(1),6,2);
        
        Kokkos::deep_copy(newsi0,oldsi);
        Kokkos::deep_copy(newsi1,oldsi);
        Kokkos::deep_copy(newsi2,oldsi);
        Kokkos::deep_copy(newsi3,oldsi);
        Kokkos::deep_copy(newsi4,oldsi);
        Kokkos::deep_copy(newsi5,oldsi);
        Kokkos::deep_copy(newsi6,oldsi);
        Kokkos::deep_copy(newsi7,oldsi);
        
        // order = 0145, 1256, 2367, 0367, 0123, 4567
        // order = bottom, right, top, left, back, front
        for (unsigned int n=0; n<oldsi.extent(1); n++) {
          newsi0(0,n,1,0) = 0;
          newsi0(0,n,1,1) = 0;
          newsi0(0,n,2,0) = 0;
          newsi0(0,n,2,1) = 0;
          newsi0(0,n,5,0) = 0;
          newsi0(0,n,5,1) = 0;
          
          newsi1(0,n,2,0) = 0;
          newsi1(0,n,2,1) = 0;
          newsi1(0,n,3,0) = 0;
          newsi1(0,n,3,1) = 0;
          newsi1(0,n,5,0) = 0;
          newsi1(0,n,5,1) = 0;
          
          newsi2(0,n,0,0) = 0;
          newsi2(0,n,0,1) = 0;
          newsi2(0,n,3,0) = 0;
          newsi2(0,n,3,1) = 0;
          newsi2(0,n,5,0) = 0;
          newsi2(0,n,5,1) = 0;
          
          newsi3(0,n,0,0) = 0;
          newsi3(0,n,0,1) = 0;
          newsi3(0,n,1,0) = 0;
          newsi3(0,n,1,1) = 0;
          newsi3(0,n,5,0) = 0;
          newsi3(0,n,5,1) = 0;
          
          newsi4(0,n,1,0) = 0;
          newsi4(0,n,1,1) = 0;
          newsi4(0,n,2,0) = 0;
          newsi4(0,n,2,1) = 0;
          newsi4(0,n,4,0) = 0;
          newsi4(0,n,4,1) = 0;
          
          newsi5(0,n,2,0) = 0;
          newsi5(0,n,2,1) = 0;
          newsi5(0,n,3,0) = 0;
          newsi5(0,n,3,1) = 0;
          newsi5(0,n,4,0) = 0;
          newsi5(0,n,4,1) = 0;
          
          newsi6(0,n,0,0) = 0;
          newsi6(0,n,0,1) = 0;
          newsi6(0,n,3,0) = 0;
          newsi6(0,n,3,1) = 0;
          newsi6(0,n,4,0) = 0;
          newsi6(0,n,4,1) = 0;
          
          newsi7(0,n,0,0) = 0;
          newsi7(0,n,0,1) = 0;
          newsi7(0,n,1,0) = 0;
          newsi7(0,n,1,1) = 0;
          newsi7(0,n,4,0) = 0;
          newsi7(0,n,4,1) = 0;
        }
        subsideinfo.push_back(newsi0);
        subsideinfo.push_back(newsi1);
        subsideinfo.push_back(newsi2);
        subsideinfo.push_back(newsi3);
        subsideinfo.push_back(newsi4);
        subsideinfo.push_back(newsi5);
        subsideinfo.push_back(newsi6);
        subsideinfo.push_back(newsi7);
        
        
        Kokkos::View<int***,AssemblyDevice> oldmap = subsidemap[e];
        Kokkos::View<int***,AssemblyDevice> newsm0("newsi",oldmap.extent(0),6,2);
        Kokkos::View<int***,AssemblyDevice> newsm1("newsi",oldmap.extent(0),6,2);
        Kokkos::View<int***,AssemblyDevice> newsm2("newsi",oldmap.extent(0),6,2);
        Kokkos::View<int***,AssemblyDevice> newsm3("newsi",oldmap.extent(0),6,2);
        Kokkos::View<int***,AssemblyDevice> newsm4("newsi",oldmap.extent(0),6,2);
        Kokkos::View<int***,AssemblyDevice> newsm5("newsi",oldmap.extent(0),6,2);
        Kokkos::View<int***,AssemblyDevice> newsm6("newsi",oldmap.extent(0),6,2);
        Kokkos::View<int***,AssemblyDevice> newsm7("newsi",oldmap.extent(0),6,2);
        
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
        for (unsigned int n=0; n<oldmap.extent(0); n++) {
          newsm0(n,1,0) = 0;
          newsm0(n,1,1) = 0;
          newsm0(n,2,0) = 0;
          newsm0(n,2,1) = 0;
          newsm0(n,5,0) = 0;
          newsm0(n,5,1) = 0;
          
          newsm1(n,2,0) = 0;
          newsm1(n,2,1) = 0;
          newsm1(n,3,0) = 0;
          newsm1(n,3,1) = 0;
          newsm1(n,5,0) = 0;
          newsm1(n,5,1) = 0;
          
          newsm2(n,0,0) = 0;
          newsm2(n,0,1) = 0;
          newsm2(n,3,0) = 0;
          newsm2(n,3,1) = 0;
          newsm2(n,5,0) = 0;
          newsm2(n,5,1) = 0;
          
          newsm3(n,0,0) = 0;
          newsm3(n,0,1) = 0;
          newsm3(n,1,0) = 0;
          newsm3(n,1,1) = 0;
          newsm3(n,5,0) = 0;
          newsm3(n,5,1) = 0;
          
          newsm4(n,1,0) = 0;
          newsm4(n,1,1) = 0;
          newsm4(n,2,0) = 0;
          newsm4(n,2,1) = 0;
          newsm4(n,4,0) = 0;
          newsm4(n,4,1) = 0;
          
          newsm5(n,2,0) = 0;
          newsm5(n,2,1) = 0;
          newsm5(n,3,0) = 0;
          newsm5(n,3,1) = 0;
          newsm5(n,4,0) = 0;
          newsm5(n,4,1) = 0;
          
          newsm6(n,0,0) = 0;
          newsm6(n,0,1) = 0;
          newsm6(n,3,0) = 0;
          newsm6(n,3,1) = 0;
          newsm6(n,4,0) = 0;
          newsm6(n,4,1) = 0;
          
          newsm7(n,0,0) = 0;
          newsm7(n,0,1) = 0;
          newsm7(n,1,0) = 0;
          newsm7(n,1,1) = 0;
          newsm7(n,4,0) = 0;
          newsm7(n,4,1) = 0;
        }
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
  // Make sure the node map is correct
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void checkNodeMap() {
    ScalarT tol = 1.0e-12;
    for (size_t i=0; i<subnodes.size(); i++) {
      for (size_t s=0; s<dimension; s++) {
        ScalarT refval = subnodes[i][s];
        ScalarT mapval = 0.0;
        for (size_t j=0; j<nodes.extent(1); j++) {
          mapval += subnodemap[i](j,s)*nodes(0,j,s);
        }
        if (abs(mapval-refval)>tol) {
          std::cout << "Error in subgrid node mapping *******" << std::endl;
          std::cout << "  refval = " << refval << std::endl;
          std::cout << "  mapval = " << mapval << std::endl;
        }
      }
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Make sure the node map is correct
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void checkSideMap() {
    for (size_t i=0; i<subsideinfo.size(); i++) {
      for (size_t n=0; n<subsideinfo[i].extent(1); n++) {
        for (size_t s=0; s<subsideinfo[i].extent(2); s++) {
          int sval0 = subsideinfo[i](0,n,s,0);
          int sval1 = subsideinfo[i](0,n,s,1);
          int cval0 = 0;
          int cval1 = 0;
          if (subsidemap[i](n,s,0)>0) {
            //if (subsidemap[i](n,s,1)>=0) {
            if (sideinfo(0,n,subsidemap[i](n,s,1),0)>0) {
              cval0 = sideinfo(0,n,subsidemap[i](n,s,1),0);
              cval1 = sideinfo(0,n,subsidemap[i](n,s,1),1);
            }
            else {
              cval0 = 1;
              cval1 = -1;
            }
          }
          else {
            cval0 = 1;
            cval1 = -1;
          }
          
          if (sval0 != cval0 || sval1 != cval1) {
            std::cout << "Error in subgrid side mapping *******" << std::endl;
            std::cout << "  refval0 = " << sval0 << std::endl;
            std::cout << "  mapval0 = " << cval0 << std::endl;
            std::cout << "  refval1 = " << sval1 << std::endl;
            std::cout << "  mapval1 = " << cval1 << std::endl;
          }
        }
      }
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid nodes
  ///////////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<ScalarT> > getNodes(DRV & newmacronodes) {
    vector<vector<ScalarT> > newnodes;
    
    for (size_t e=0; e<newmacronodes.extent(0); e++) {
      for (size_t i=0; i<subnodemap.size(); i++) {
        vector<ScalarT> newnode(dimension,0.0);
        for (size_t j=0; j<newmacronodes.extent(1); j++) {
          for (size_t s=0; s<dimension; s++) {
            newnode[s] += subnodemap[i](j,s)*newmacronodes(e,j,s);
          }
        }
        newnodes.push_back(newnode);
      }
    }
    return newnodes;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid nodes
  ///////////////////////////////////////////////////////////////////////////////////////
  
  DRV getNewNodes(DRV & newmacronodes) {
    vector<vector<ScalarT> > newnodes;
    //KokkosTools::print(newmacronodes,"new macro nodes");
    for (size_t e=0; e<newmacronodes.extent(0); e++) {
      for (size_t i=0; i<subnodemap.size(); i++) {
        vector<ScalarT> newnode(dimension,0.0);
        for (size_t j=0; j<newmacronodes.extent(1); j++) {
          for (size_t s=0; s<dimension; s++) {
            newnode[s] += subnodemap[i](j,s)*newmacronodes(e,j,s);
          }
        }
        newnodes.push_back(newnode);
      }
    }
    
    DRV currnodes("currnodes",newmacronodes.extent(0)*subconnectivity.size(),
                  subconnectivity[0].size(),
                  dimension);
    int prog = 0, prog2 = 0;
    
    for (size_t k=0; k<newmacronodes.extent(0); k++) { // number of macro elements
      for (size_t e=0; e<subconnectivity.size(); e++) { // number of elements
        for (int n=0; n<subconnectivity[e].size(); n++) { // number of nodes/element
          for (int m=0; m<dimension; m++) {
            currnodes(prog+e,n,m) = newnodes[prog2+subconnectivity[e][n]][m];
          }
        }
      }
      prog += subconnectivity.size();
      prog2 += subnodes.size();
    }
    //KokkosTools::print(currnodes,"new subgrid nodes");
    
    return currnodes;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid side info
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<int****,HostDevice> getNewSideinfo(Kokkos::View<int****,HostDevice> & macrosideinfo) {
    Kokkos::View<int****,HostDevice> ksubsideinfo("subgrid side info",
                                                  macrosideinfo.extent(0)*subsideinfo.size(), // macroelem*subelem
                                                  sideinfo.extent(1),
                                                  sideinfo.extent(2),
                                                  2);
    
    //KokkosTools::print(subsidemap[0]);
    int prog = 0;
    for (unsigned int k=0; k<macrosideinfo.extent(0); k++) {
      
      for (unsigned int e=0; e<subsidemap.size(); e++) {
        for (unsigned int i=0; i<ksubsideinfo.extent(1); i++) {
          for (unsigned int j=0; j<ksubsideinfo.extent(2); j++) {
            if (subsidemap[e](i,j,0)>0) {
              int sideindex = subsidemap[e](i,j,1);
              if (macrosideinfo(k,i,sideindex,0)>0) {
                ksubsideinfo(e+prog,i,j,0) = macrosideinfo(k,i,sideindex,0);
                ksubsideinfo(e+prog,i,j,1) = macrosideinfo(k,i,sideindex,1);
              }
              else {
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
  // Get the sub-grid nodes
  ///////////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<ScalarT> > getSubNodes() {
    return subnodes;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid connectivity
  ///////////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<GO> > getSubConnectivity() {
    return subconnectivity;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid connectivity
  ///////////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<GO> > getSubConnectivity(int & reps) {
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
      prog += subnodes.size();
    }
    return newconn;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid sideinfo
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<int****,HostDevice> getSubSideinfo() {
    Kokkos::View<int****,HostDevice> ksubsideinfo("subgrid side info",subsideinfo.size(),
                                                  sideinfo.extent(1),sideinfo.extent(2),2);
    for (unsigned int e=0; e<ksubsideinfo.extent(0); e++) {
      for (unsigned int i=0; i<ksubsideinfo.extent(1); i++) {
        for (unsigned int j=0; j<ksubsideinfo.extent(2); j++) {
          for (unsigned int k=0; k<ksubsideinfo.extent(3); k++) {
            ksubsideinfo(e,i,j,k) = subsideinfo[e](0,i,j,k);
          }
        }
      }
    }
    return ksubsideinfo;
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
  
protected:
  
  int dimension;
  Teuchos::RCP<MpiComm> LocalComm;
  string shape, subshape;
  DRV nodes;
  Kokkos::View<int****,HostDevice> sideinfo;
  vector<vector<ScalarT> > subnodes;
  vector<Kokkos::View<ScalarT**,AssemblyDevice> > subnodemap;
  vector<Kokkos::View<int***,AssemblyDevice> > subsidemap;
  vector<vector<GO> > subconnectivity;
  vector<Kokkos::View<int****,AssemblyDevice> > subsideinfo;
  
  
};
#endif

