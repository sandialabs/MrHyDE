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
  
  SubGridTools(const Teuchos::RCP<Epetra_MpiComm> & LocalComm_, const string & shape_,
               const string & subshape_, const DRV nodes_, Kokkos::View<int****,HostDevice> sideinfo_) :
  LocalComm(LocalComm_), shape(shape_), subshape(subshape_), nodes(nodes_), sideinfo(sideinfo_) {
    
    dimension = nodes.dimension(2);
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Given the coarse grid nodes and shape, define the subgrid nodes, connectivity, and sideinfo
  //////////////////////////////////////////////////////////////////////////////////////
  
  void createSubMesh(const int & numrefine) {
    
    if (subshape == shape) {
      vector<int> newconn;
      for (int i=0; i<nodes.dimension(1); i++) {
        vector<double> newnode;
        for (int s=0; s<dimension; s++) {
          newnode.push_back(nodes(0,i,s));
        }
        subnodes.push_back(newnode);
        newconn.push_back(i);
      }
      subconnectivity.push_back(newconn);
      
      Kokkos::View<int****,AssemblyDevice> newsi("newsi",1,sideinfo.dimension(1),sideinfo.dimension(2),2);
      for (size_t n=0; n<sideinfo.dimension(1); n++) {
        for (size_t s=0; s<sideinfo.dimension(2); s++) {
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
      
    }
    else {
      if (dimension == 1) {
        // output an error message
      }
      else if (dimension == 2) {
        if (shape == "quad" && subshape == "tri") {
          vector<int> newconn0, newconn1, newconn2, newconn3;
          for (int i=0; i<nodes.dimension(1); i++) {
            vector<double> newnode;
            for (int s=0; s<dimension; s++) {
              newnode.push_back(nodes(0,i,s));
            }
            subnodes.push_back(newnode);
          }
          vector<double> midnode;
          midnode.push_back(0.25*(nodes(0,0,0)+nodes(0,1,0)+nodes(0,2,0)+nodes(0,3,0)));
          midnode.push_back(0.25*(nodes(0,0,1)+nodes(0,1,1)+nodes(0,2,1)+nodes(0,3,1)));
          subnodes.push_back(midnode);
          
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
          
          Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,sideinfo.dimension(1),3,2);
          Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,sideinfo.dimension(1),3,2);
          Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,sideinfo.dimension(1),3,2);
          Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,sideinfo.dimension(1),3,2);
          
          for (size_t n=0; n<sideinfo.dimension(1); n++) {
            
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
        }
        else if (shape == "tri" && subshape == "quad") {
          vector<int> newconn0, newconn1, newconn2;
          for (int i=0; i<nodes.dimension(1); i++) {
            vector<double> newnode;
            
            for (int s=0; s<dimension; s++) {
              newnode.push_back(nodes(0,i,s));
            }
            subnodes.push_back(newnode);
          }
          vector<double> center, mid01, mid12, mid02;
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
          
          Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,sideinfo.dimension(1),4,2);
          Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,sideinfo.dimension(1),4,2);
          Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,sideinfo.dimension(1),4,2);
          
          for (size_t n=0; n<sideinfo.dimension(1); n++) {
            
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
    }
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Uniformly refine an element
  //////////////////////////////////////////////////////////////////////////////////////
  
  void refineSubCell(const int & e) {
    
    if (dimension == 1) {
      vector<double> node0 = subnodes[subconnectivity[e][0]];
      vector<double> node1 = subnodes[subconnectivity[e][1]];
      
      double midx = 0.5*(node0[0]+node1[0]);
      vector<double> mid(1,midx);
      
      subnodes.push_back(mid);
      
      vector<int> newelem0, newelem1;
      newelem0.push_back(subconnectivity[e][0]);
      newelem0.push_back(subnodes.size());
      newelem1.push_back(subnodes.size());
      newelem1.push_back(subconnectivity[e][1]);
      
      subconnectivity.push_back(newelem0);
      subconnectivity.push_back(newelem1);
      
      
      Kokkos::View<int****,AssemblyDevice> oldsi = subsideinfo[e];
      Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.dimension(1),2,2);
      Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.dimension(1),2,2);
      Kokkos::deep_copy(newsi0,oldsi);
      Kokkos::deep_copy(newsi1,oldsi);
      
      for (int n=0; n<oldsi.dimension(1); n++) {
        newsi0(0,n,1,0) = 0;
        newsi0(0,n,1,1) = 0;
        newsi1(0,n,0,0) = 0;
        newsi1(0,n,0,1) = 0;
      }
      subsideinfo.push_back(newsi0);
      subsideinfo.push_back(newsi1);
      
    }
    if (dimension == 2) {
      if (subshape == "tri") {
        // Extract the existing nodes
        vector<double> node0 = subnodes[subconnectivity[e][0]];
        vector<double> node1 = subnodes[subconnectivity[e][1]];
        vector<double> node2 = subnodes[subconnectivity[e][2]];
        
        // Compute the candidate new nodes
        vector<double> mid01, mid12, mid02;
        mid01.push_back(0.5*(node0[0]+node1[0]));
        mid01.push_back(0.5*(node0[1]+node1[1]));
        mid12.push_back(0.5*(node1[0]+node2[0]));
        mid12.push_back(0.5*(node1[1]+node2[1]));
        mid02.push_back(0.5*(node0[0]+node2[0]));
        mid02.push_back(0.5*(node0[1]+node2[1]));
        
        // Check if these nodes have been added and add if not
        double tol=1.0e-10;
        int mid01_ind, mid12_ind, mid02_ind;
        double check;
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
        vector<int> elem0, elem1, elem2, elem3;
        
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
        Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.dimension(1),3,2);
        Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.dimension(1),3,2);
        Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,oldsi.dimension(1),3,2);
        Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,oldsi.dimension(1),3,2);
        Kokkos::deep_copy(newsi0,oldsi);
        Kokkos::deep_copy(newsi1,oldsi);
        Kokkos::deep_copy(newsi2,oldsi);
        Kokkos::deep_copy(newsi3,oldsi);
        
        for (int n=0; n<oldsi.dimension(1); n++) {
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
        
        
      }
      else if (subshape == "quad") {
        // Extract the existing nodes
        vector<double> node0 = subnodes[subconnectivity[e][0]];
        vector<double> node1 = subnodes[subconnectivity[e][1]];
        vector<double> node2 = subnodes[subconnectivity[e][2]];
        vector<double> node3 = subnodes[subconnectivity[e][3]];
        
        // Compute the candidate new nodes
        vector<double> center, mid01, mid12, mid23, mid03;
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
        double tol=1.0e-6;
        int center_ind, mid01_ind, mid12_ind, mid23_ind, mid03_ind;
        double check;
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
        vector<int> elem0, elem1, elem2, elem3;
        
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
        Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::deep_copy(newsi0,oldsi);
        Kokkos::deep_copy(newsi1,oldsi);
        Kokkos::deep_copy(newsi2,oldsi);
        Kokkos::deep_copy(newsi3,oldsi);
        
        for (int n=0; n<oldsi.dimension(1); n++) {
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
        
      }
      else {
        // add error
      }
    }
    if (dimension == 3) {
      if (subshape == "tet") {
        // Extract the existing nodes
        vector<double> node0 = subnodes[subconnectivity[e][0]];
        vector<double> node1 = subnodes[subconnectivity[e][1]];
        vector<double> node2 = subnodes[subconnectivity[e][2]];
        vector<double> node3 = subnodes[subconnectivity[e][3]];
        
        // Compute the candidate new nodes
        vector<double> mid01, mid12, mid02, mid03, mid13, mid23;
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
        double tol=1.0e-10;
        int mid01_ind, mid12_ind, mid02_ind;
        int mid03_ind, mid13_ind, mid23_ind;
        double check;
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
        vector<int> elem0, elem1, elem2, elem3, elem4, elem5, elem6, elem7;
        
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
        Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi4("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi5("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi6("newsi",1,oldsi.dimension(1),4,2);
        Kokkos::View<int****,AssemblyDevice> newsi7("newsi",1,oldsi.dimension(1),4,2);
        
        Kokkos::deep_copy(newsi0,oldsi);
        Kokkos::deep_copy(newsi1,oldsi);
        Kokkos::deep_copy(newsi2,oldsi);
        Kokkos::deep_copy(newsi3,oldsi);
        Kokkos::deep_copy(newsi4,oldsi);
        Kokkos::deep_copy(newsi5,oldsi);
        Kokkos::deep_copy(newsi6,oldsi);
        Kokkos::deep_copy(newsi7,oldsi);
        
        for (int n=0; n<oldsi.dimension(1); n++) {
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
        
      }
      else if (subshape == "hex") {
        // Extract the existing nodes
        vector<double> node0 = subnodes[subconnectivity[e][0]];
        vector<double> node1 = subnodes[subconnectivity[e][1]];
        vector<double> node2 = subnodes[subconnectivity[e][2]];
        vector<double> node3 = subnodes[subconnectivity[e][3]];
        vector<double> node4 = subnodes[subconnectivity[e][4]];
        vector<double> node5 = subnodes[subconnectivity[e][5]];
        vector<double> node6 = subnodes[subconnectivity[e][6]];
        vector<double> node7 = subnodes[subconnectivity[e][7]];
        
        // Compute the candidate new nodes
        vector<double> mid0123, mid01, mid12, mid23, mid03;
        vector<double> center, mid04, mid15, mid26, mid37, mid0145, mid1256, mid2367, mid0347;
        vector<double> mid4567, mid45, mid56, mid67, mid47;
        
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
        double tol=1.0e-6;
        
        int mid0123_ind, mid01_ind, mid12_ind, mid23_ind, mid03_ind;
        int center_ind, mid04_ind, mid15_ind, mid26_ind, mid37_ind;
        int mid0145_ind, mid1256_ind, mid2367_ind, mid0347_ind;
        int mid4567_ind, mid45_ind, mid56_ind, mid67_ind, mid47_ind;
        double check;
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
        vector<int> elem0, elem1, elem2, elem3, elem4, elem5, elem6, elem7;
        
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
        Kokkos::View<int****,AssemblyDevice> newsi0("newsi",1,oldsi.dimension(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi1("newsi",1,oldsi.dimension(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi2("newsi",1,oldsi.dimension(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi3("newsi",1,oldsi.dimension(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi4("newsi",1,oldsi.dimension(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi5("newsi",1,oldsi.dimension(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi6("newsi",1,oldsi.dimension(1),6,2);
        Kokkos::View<int****,AssemblyDevice> newsi7("newsi",1,oldsi.dimension(1),6,2);
        
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
        for (int n=0; n<oldsi.dimension(1); n++) {
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
        
      }
      else {
        // add error
      }
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Check if a sub-grid nodes has already been added to the list
  ///////////////////////////////////////////////////////////////////////////////////////
  
  bool checkExistingSubNodes(const vector<double> & newpt,
                             const double & tol, int & index) {
    bool found = false;
    int dimension = newpt.size();
    for (int i=0; i<subnodes.size(); i++) {
      if (!found) {
        double val = 0.0;
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
  // Get the sub-grid nodes
  ///////////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<double> > getSubNodes() {
    return subnodes;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid connectivity
  ///////////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<int> > getSubConnectivity() {
    return subconnectivity;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the sub-grid sideinfo
  ///////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<int****,HostDevice> getSubSideinfo() {
    Kokkos::View<int****,HostDevice> ksubsideinfo("subgrid side info",subsideinfo.size(),
                                                  sideinfo.dimension(1),sideinfo.dimension(2),2);
    for (int e=0; e<ksubsideinfo.dimension(0); e++) {
      for (int i=0; i<ksubsideinfo.dimension(1); i++) {
        for (int j=0; j<ksubsideinfo.dimension(2); j++) {
          for (int k=0; k<ksubsideinfo.dimension(3); k++) {
            ksubsideinfo(e,i,j,k) = subsideinfo[e](0,i,j,k);
          }
        }
      }
    }
    return ksubsideinfo;
  }
  
protected:
  
  int dimension;
  Teuchos::RCP<Epetra_MpiComm> LocalComm;
  string shape, subshape;
  DRV nodes;
  Kokkos::View<int****,HostDevice> sideinfo;
  vector<vector<double> > subnodes;
  vector<vector<int> > subconnectivity;
  vector<Kokkos::View<int****,AssemblyDevice> > subsideinfo;
  
  
};
#endif

