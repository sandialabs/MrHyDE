/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SUBGRIDLOCALDATA_H
#define SUBGRIDLOCALDATA_H

#include "trilinos.hpp"
#include "preferences.hpp"

class SubGridLocalData {
public:
  
  SubGridLocalData() {} ;
  
  ~SubGridLocalData() {} ;
  
  SubGridLocalData(DRV & macronodes_, Kokkos::View<int****,HostDevice> & macrosideinfo_,
                   Kokkos::View<GO**,HostDevice> & macroGIDs_,
                   Kokkos::View<LO***,HostDevice> & macroindex_,
                   Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & macroorientation_)
  : macronodes(macronodes_), macrosideinfo(macrosideinfo_), macroGIDs(macroGIDs_),
  macroindex(macroindex_), macroorientation(macroorientation_) {
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void setIP(DRV & ref_ip, topo_RCP & cellTopo) {
    ip = DRV("ip", nodes.dimension(0), ref_ip.dimension(0), nodes.dimension(2));
    Intrepid2::CellTools<AssemblyDevice>::mapToPhysicalFrame(ip, ref_ip, nodes, *cellTopo);
    ijac = DRV("ijac", nodes.dimension(0), ref_ip.dimension(0), nodes.dimension(2), nodes.dimension(2));
    Intrepid2::CellTools<AssemblyDevice>::setJacobian(ijac, ref_ip, nodes, *cellTopo);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void computeMacroBasisVolIP(topo_RCP & macro_cellTopo, vector<basis_RCP> & macro_basis_pointers,
                              Teuchos::RCP<DiscTools> & discTools) {
    vector<DRV> currcell_basis, currcell_basisGrad;
    
    // Already have ip
    DRV sref_ip_tmp("sref_ip_tmp", nodes.dimension(0), ip.dimension(1), ip.dimension(2));
    DRV sref_ip("sref_ip",ip.dimension(1), ip.dimension(2));
    Intrepid2::CellTools<AssemblyDevice>::mapToReferenceFrame(sref_ip_tmp, ip, macronodes, *macro_cellTopo);
    for (size_t i=0; i<ip.dimension(1); i++) {
      for (size_t j=0; j<ip.dimension(2); j++) {
        sref_ip(i,j) = sref_ip_tmp(0,i,j);
      }
    }
    for (size_t i=0; i<macro_basis_pointers.size(); i++) {
      currcell_basis.push_back(discTools->evaluateBasis(macro_basis_pointers[i], sref_ip, macroorientation));
      currcell_basisGrad.push_back(discTools->evaluateBasisGrads(macro_basis_pointers[i], macronodes,
                                                                 sref_ip, macro_cellTopo, macroorientation));
    }
    
    aux_basis.push_back(currcell_basis);
    aux_basis_grad.push_back(currcell_basisGrad);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void computeMacroBasisBoundaryIP(topo_RCP & macro_cellTopo, vector<basis_RCP> & macro_basis_pointers,
                                   Teuchos::RCP<DiscTools> & discTools, Teuchos::RCP<workset> & wkset) {
    
    for (size_t e=0; e<BIDs.size(); e++) {
      
      int numElem = boundaryNodes[e].dimension(0);
      
      //DRV sside_ip = wkset[0]->ip_side_vec[boundaryCells[mindex][e]->wksetBID];
      DRV sside_ip = wkset->ip_side_vec[BIDs[e]];
      vector<DRV> currside_basis, currside_basis_grad;
      for (size_t i=0; i<macro_basis_pointers.size(); i++) {
        DRV tmp_basis = DRV("basis values",numElem,macro_basis_pointers[i]->getCardinality(),sside_ip.dimension(1));
        currside_basis.push_back(tmp_basis);
      }
      
      for (size_t c=0; c<numElem; c++) {
        //DRV side_ip_e("side_ip_e",cells[block][e]->numElem, sside_ip.dimension(1), sside_ip.dimension(2));
        DRV side_ip_e("side_ip_e",1, sside_ip.dimension(1), sside_ip.dimension(2));
        for (unsigned int i=0; i<sside_ip.dimension(1); i++) {
          for (unsigned int j=0; j<sside_ip.dimension(2); j++) {
            side_ip_e(0,i,j) = sside_ip(c,i,j);
          }
        }
        //DRV sref_side_ip_tmp("sref_side_ip_tmp",sside_ip.dimension(0), sside_ip.dimension(1), sside_ip.dimension(2));
        DRV sref_side_ip_tmp("sref_side_ip_tmp",1, sside_ip.dimension(1), sside_ip.dimension(2));
        
        //CellTools<AssemblyDevice>::mapToReferenceFrame(sref_side_ip_tmp, side_ip_e, macronodes[block], *macro_cellTopo);
        DRV sref_side_ip("sref_side_ip", sside_ip.dimension(1), sside_ip.dimension(2));
        Intrepid2::CellTools<AssemblyDevice>::mapToReferenceFrame(sref_side_ip_tmp, side_ip_e, macronodes, *macro_cellTopo);
        for (size_t i=0; i<sside_ip.dimension(1); i++) {
          for (size_t j=0; j<sside_ip.dimension(2); j++) {
            sref_side_ip(i,j) = sref_side_ip_tmp(0,i,j);
          }
        }
        for (size_t i=0; i<macro_basis_pointers.size(); i++) {
          DRV bvals = discTools->evaluateBasis(macro_basis_pointers[i], sref_side_ip, macroorientation);
          for (unsigned int k=0; k<bvals.dimension(1); k++) {
            for (unsigned int j=0; j<bvals.dimension(2); j++) {
              currside_basis[i](c,k,j) = bvals(0,k,j);
            }
          }
          
        }
      }
      aux_side_basis.push_back(currside_basis);
      aux_side_basis_grad.push_back(currside_basis_grad);
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV macronodes, nodes, ip, ijac;
  Kokkos::View<int****,HostDevice> macrosideinfo, sideinfo;
  Kokkos::View<GO**,HostDevice> macroGIDs;
  Kokkos::View<LO***,HostDevice> macroindex;
  Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> macroorientation;
    
  Kokkos::View<int**,AssemblyDevice> bcs;
  vector<vector<DRV> > aux_basis, aux_basis_grad, aux_side_basis, aux_side_basis_grad;
  vector<int> BIDs;
  vector<DRV> boundaryNodes;
  vector<string> boundaryNames;
  
  vector<Kokkos::View<ScalarT**,HostDevice> > sensorLocations, sensorData;
  DRV sensorPoints;
  vector<int> sensorElem, mySensorIDs;
  vector<vector<DRV> > sensorBasis, param_sensorBasis, sensorBasisGrad, param_sensorBasisGrad;
  
  Kokkos::View<ScalarT**,AssemblyDevice> cell_data;
  vector<size_t> cell_data_seed, cell_data_seedindex;
  vector<ScalarT> cell_data_distance;
  
};
#endif

