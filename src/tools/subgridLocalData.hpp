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
                   Kokkos::View<LO***,AssemblyDevice> & macroindex_,
                   Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & macroorientation_)
  : macronodes(macronodes_), macrosideinfo(macrosideinfo_), macroGIDs(macroGIDs_),
  macroindex(macroindex_), macroorientation(macroorientation_) {
    
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void setIP(DRV & ref_ip, DRV & ref_wts, topo_RCP & cellTopo) {
    ip = DRV("ip", nodes.extent(0), ref_ip.extent(0), nodes.extent(2));
    CellTools::mapToPhysicalFrame(ip, ref_ip, nodes, *cellTopo);
    jacobian = DRV("jacobian", nodes.extent(0), ref_ip.extent(0), nodes.extent(2), nodes.extent(2));
    CellTools::setJacobian(jacobian, ref_ip, nodes, *cellTopo);
    
    jacobianDet = DRV("determinant of jacobian", nodes.extent(0), ref_ip.extent(0));
    jacobianInv = DRV("inverse of jacobian", nodes.extent(0), ref_ip.extent(0), nodes.extent(2), nodes.extent(2));
    CellTools::setJacobianDet(jacobianDet, jacobian);
    CellTools::setJacobianInv(jacobianInv, jacobian);
    
    wts = DRV("ip wts", nodes.extent(0), ref_ip.extent(0));
    FuncTools::computeCellMeasure(wts, jacobianDet, ref_wts);
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void computeMacroBasisVolIP(topo_RCP & macro_cellTopo, vector<basis_RCP> & macro_basis_pointers,
                              Teuchos::RCP<DiscTools> & discTools) {
    vector<DRV> currcell_basis, currcell_basisGrad;
    
    // Already have ip
    DRV sref_ip_tmp("sref_ip_tmp", nodes.extent(0), ip.extent(1), ip.extent(2));
    DRV sref_ip("sref_ip",ip.extent(1), ip.extent(2));
    CellTools::mapToReferenceFrame(sref_ip_tmp, ip, macronodes, *macro_cellTopo);
    for (size_t i=0; i<ip.extent(1); i++) {
      for (size_t j=0; j<ip.extent(2); j++) {
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
      
      int numElem = boundaryNodes[e].extent(0);
      
      //DRV sside_ip = wkset[0]->ip_side_vec[boundaryCells[mindex][e]->wksetBID];
      DRV sside_ip = wkset->ip_side_vec[BIDs[e]];
      vector<DRV> currside_basis, currside_basis_grad;
      for (size_t i=0; i<macro_basis_pointers.size(); i++) {
        DRV tmp_basis = DRV("basis values",numElem,macro_basis_pointers[i]->getCardinality(),sside_ip.extent(1));
        currside_basis.push_back(tmp_basis);
      }
      //KokkosTools::print(sside_ip);
      for (size_t c=0; c<numElem; c++) {
        //DRV side_ip_e("side_ip_e",cells[block][e]->numElem, sside_ip.extent(1), sside_ip.extent(2));
        DRV side_ip_e("side_ip_e",1, sside_ip.extent(1), sside_ip.extent(2));
        for (unsigned int i=0; i<sside_ip.extent(1); i++) {
          for (unsigned int j=0; j<sside_ip.extent(2); j++) {
            side_ip_e(0,i,j) = sside_ip(c,i,j);
          }
        }
        //DRV sref_side_ip_tmp("sref_side_ip_tmp",sside_ip.extent(0), sside_ip.extent(1), sside_ip.extent(2));
        DRV sref_side_ip_tmp("sref_side_ip_tmp",1, sside_ip.extent(1), sside_ip.extent(2));
        
        //CellTools<AssemblyDevice>::mapToReferenceFrame(sref_side_ip_tmp, side_ip_e, macronodes[block], *macro_cellTopo);
        DRV sref_side_ip("sref_side_ip", sside_ip.extent(1), sside_ip.extent(2));
        
        // issue is here
        // need to know which macro-element each (set of) ip belongs to
        // this is trivial for volumetric ip since nummacro = macronodes.extent(0)
        // and the number of total elements divides evenly.
        // this assumption is not nec. true for the boundary cells
        size_t mID = boundaryMIDs[e][c];
        DRV cnodes("tmp nodes",1,macronodes.extent(1),macronodes.extent(2));
        for (size_t i=0; i<macronodes.extent(1); i++) {
          for (size_t j=0; j<macronodes.extent(2); j++) {
            cnodes(0,i,j) = macronodes(mID,i,j);
          }
        }
        CellTools::mapToReferenceFrame(sref_side_ip_tmp, side_ip_e, cnodes, *macro_cellTopo);
        for (size_t i=0; i<sside_ip.extent(1); i++) {
          for (size_t j=0; j<sside_ip.extent(2); j++) {
            sref_side_ip(i,j) = sref_side_ip_tmp(0,i,j);
          }
        }
        Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> corientation("tmp orientation",1);
        corientation(0) = macroorientation(mID);
        for (size_t i=0; i<macro_basis_pointers.size(); i++) {
          DRV bvals = discTools->evaluateBasis(macro_basis_pointers[i], sref_side_ip, corientation);
          for (unsigned int k=0; k<bvals.extent(1); k++) {
            for (unsigned int j=0; j<bvals.extent(2); j++) {
              currside_basis[i](c,k,j) = bvals(0,k,j);
            }
          }
          
        }
      }
      aux_side_basis.push_back(currside_basis);
      aux_side_basis_grad.push_back(currside_basis_grad);
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Get the macro element for a sub-element
  ///////////////////////////////////////////////////////////////////////////////////////
  size_t getMacroID(const size_t & eID) {
    size_t numMacro = macronodes.extent(0);
    size_t numElem = nodes.extent(0);
    size_t numEperM = numElem/numMacro;
    size_t mID = 0;
    size_t prog = 0;
    for (size_t i=0; i<numMacro; i++) {
      for (size_t j=0; j<numEperM; j++) {
        if (prog+j == eID) {
          mID = i;
        }
      }
      prog+=numEperM;
    }
    return mID;
  }

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void setBoundaryIndexGIDs() {
    for (size_t b=0; b<boundaryMIDs.size(); b++) {
      Kokkos::View<GO**,HostDevice> cGIDs("boundary macro GIDs",boundaryMIDs[b].size(),macroGIDs.extent(1));
      Kokkos::View<LO***,AssemblyDevice> cindex("boundary macro GIDs",boundaryMIDs[b].size(),macroindex.extent(1),
                                            macroindex.extent(2));
      for (size_t e=0; e<boundaryMIDs[b].size(); e++) {
        size_t mid = boundaryMIDs[b][e];
        for (size_t i=0; i<cGIDs.extent(1); i++) {
          cGIDs(e,i) = macroGIDs(mid,i);
        }
        for (size_t i=0; i<cindex.extent(1); i++) {
          for (size_t j=0; j<cindex.extent(2); j++) {
            cindex(e,i,j) = macroindex(mid,i,j);
          }
        }
      }
      boundaryMacroGIDs.push_back(cGIDs);
      boundaryMacroindex.push_back(cindex);
    }
  }
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  DRV macronodes, nodes, ip, wts, jacobian, jacobianInv, jacobianDet;
  Kokkos::View<int****,HostDevice> macrosideinfo, sideinfo;
  Kokkos::View<GO**,HostDevice> macroGIDs;
  Kokkos::View<LO***,AssemblyDevice> macroindex;
  vector<Kokkos::View<GO**,HostDevice> > boundaryMacroGIDs;
  vector<Kokkos::View<LO***,AssemblyDevice> > boundaryMacroindex;
  
  Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> macroorientation;
    
  Kokkos::View<int**,AssemblyDevice> bcs;
  vector<vector<DRV> > aux_basis, aux_basis_grad, aux_side_basis, aux_side_basis_grad;
  vector<int> BIDs;
  vector<DRV> boundaryNodes;
  vector<string> boundaryNames;
  vector<vector<size_t> > boundaryMIDs;
  vector<size_t> macroIDs;
  
  vector<Kokkos::View<ScalarT**,HostDevice> > sensorLocations, sensorData;
  DRV sensorPoints;
  vector<int> sensorElem, mySensorIDs;
  vector<vector<DRV> > sensorBasis, param_sensorBasis, sensorBasisGrad, param_sensorBasisGrad;
  
  Kokkos::View<ScalarT**,AssemblyDevice> cell_data;
  vector<size_t> cell_data_seed, cell_data_seedindex;
  vector<ScalarT> cell_data_distance;
  
};
#endif

