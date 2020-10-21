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

#ifndef SUBGRIDLOCALDATA_H
#define SUBGRIDLOCALDATA_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "discretizationInterface.hpp"

namespace MrHyDE {
  
  class SubGridLocalData {
  public:
    
    SubGridLocalData() {} ;
    
    ~SubGridLocalData() {} ;
    
    SubGridLocalData(DRV & macronodes_, Kokkos::View<int****,HostDevice> & macrosideinfo_,
                     LIDView macroLIDs_,
                     Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & macroorientation_)
    : macronodes(macronodes_), macrosideinfo(macrosideinfo_), macroLIDs(macroLIDs_),
    macroorientation(macroorientation_) {
      
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void setIP(const Teuchos::RCP<CellMetaData> & cellData,
               Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & orientation) {
      
      // By convention, each local subgrid only has one cell that contains all of the elements
      // Thus, there is no loop over cells here
      
      int numElem = nodes.extent(0);
      int dimension = cellData->dimension;
      int numip = cellData->ref_ip.extent(0);
      
      ////////////////////////////////////////////////////////////////////////////////
      // create the volumetric integration information
      ////////////////////////////////////////////////////////////////////////////////
      
      ip = DRV("ip", numElem, numip, dimension);
      CellTools::mapToPhysicalFrame(ip, cellData->ref_ip, nodes, *(cellData->cellTopo));
      
      DRV jacobian("jacobian", numElem, numip, dimension, dimension);
      CellTools::setJacobian(jacobian, cellData->ref_ip, nodes, *(cellData->cellTopo));
      
      DRV jacobianDet("determinant of jacobian", numElem, numip);
      DRV jacobianInv("inverse of jacobian", numElem, numip, dimension, dimension);
      CellTools::setJacobianDet(jacobianDet, jacobian);
      CellTools::setJacobianInv(jacobianInv, jacobian);
      
      wts = DRV("ip wts", numElem, numip);
      FuncTools::computeCellMeasure(wts, jacobianDet, cellData->ref_wts);
      
      hsize = Kokkos::View<ScalarT*,AssemblyDevice>("element sizes", numElem);
      parallel_for(RangePolicy<AssemblyExec>(0,wts.extent(0)), KOKKOS_LAMBDA (const int e ) {
        ScalarT vol = 0.0;
        for (int i=0; i<wts.extent(1); i++) {
          vol += wts(e,i);
        }
        ScalarT dimscl = 1.0/(ScalarT)ip.extent(2);
        hsize(e) = std::pow(vol,dimscl);
      });
      
      ////////////////////////////////////////////////////////////////////////////////
      // create the basis function at volumetric ip
      ////////////////////////////////////////////////////////////////////////////////
      
      for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
        
        int numb = cellData->basis_pointers[i]->getCardinality();
        
        DRV basis_vals, basis_grad_vals, basis_div_vals, basis_curl_vals;
        
        if (cellData->basis_types[i] == "HGRAD"){
          basis_vals = DRV("basis",numElem,numb,numip);
          DRV tmp_basis_vals("basis",numElem,numb,numip);
          FuncTools::HGRADtransformVALUE(tmp_basis_vals, cellData->ref_basis[i]);
          OrientTools::modifyBasisByOrientation(basis_vals, tmp_basis_vals, orientation,
                                                cellData->basis_pointers[i].get());
          
          DRV basis_grad_tmp("basis grad tmp",numElem,numb,numip,dimension);
          basis_grad_vals = DRV("basis grad",numElem,numb,numip,dimension);
          FuncTools::HGRADtransformGRAD(basis_grad_tmp, jacobianInv, cellData->ref_basis_grad[i]);
          OrientTools::modifyBasisByOrientation(basis_grad_vals, basis_grad_tmp, orientation,
                                                cellData->basis_pointers[i].get());
        }
        else if (cellData->basis_types[i] == "HVOL"){
          basis_vals = DRV("basis",numElem,numb,numip);
          FuncTools::HGRADtransformVALUE(basis_vals, cellData->ref_basis[i]);
        }
        else if (cellData->basis_types[i] == "HDIV"){
          
          basis_vals = DRV("basis",numElem,numb,numip,dimension);
          DRV basis_tmp("basis tmp",numElem,numb,numip,dimension);
          FuncTools::HDIVtransformVALUE(basis_tmp, jacobian, jacobianDet, cellData->ref_basis[i]);
          //basis_uw[i] = basis_tmp;
          OrientTools::modifyBasisByOrientation(basis_vals, basis_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          
          
          basis_div_vals = DRV("basis div",numElem,numb,numip);
          DRV basis_div_vals_tmp("basis div tmp",numElem,numb,numip);
          FuncTools::HDIVtransformDIV(basis_div_vals_tmp, jacobianDet, cellData->ref_basis_div[i]);
          OrientTools::modifyBasisByOrientation(basis_div_vals, basis_div_vals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          
        }
        else if (cellData->basis_types[i] == "HCURL"){
          
          basis_vals = DRV("basis",numElem,numb,numip,dimension);
          DRV basis_tmp("basis tmp",numElem,numb,numip,dimension);
          FuncTools::HCURLtransformVALUE(basis_tmp, jacobianInv, cellData->ref_basis[i]);
          OrientTools::modifyBasisByOrientation(basis_vals, basis_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          
          basis_curl_vals = DRV("basis curl",numElem,numb,numip,dimension);
          DRV basis_curl_vals_tmp("basis curl tmp",numElem,numb,numip,dimension);
          FuncTools::HCURLtransformCURL(basis_curl_vals_tmp, jacobian, jacobianDet, cellData->ref_basis_curl[i]);
          OrientTools::modifyBasisByOrientation(basis_curl_vals, basis_curl_vals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          
        }
        basis.push_back(basis_vals);
        basis_grad.push_back(basis_grad_vals);
        basis_div.push_back(basis_div_vals);
        basis_curl.push_back(basis_curl_vals);
        
      }
      
      ////////////////////////////////////////////////////////////////////////////////
      // create the basis functions at face ip (not added yet)
      ////////////////////////////////////////////////////////////////////////////////
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void setBoundaryIP(const Teuchos::RCP<CellMetaData> & cellData,
                       vector<Kokkos::View<LO*,AssemblyDevice> > & localSideIDs,
                       vector<Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> > & orientation) {
      
      // By convention, each local subgrid can have multiple boundary cells and each
      // typically contains all of the elements on a given boundary
      // Thus, there is a loop over boundary cells here
      
      for (size_t bcell=0; bcell<boundaryNodes.size(); bcell++) {
        
        DRV bnodes = boundaryNodes[bcell];
        
        int numElem = bnodes.extent(0);
        LO localSID = localSideIDs[bcell](0);
        DRV ref_ip = cellData->ref_side_ip[localSID];
        DRV ref_wts = cellData->ref_side_wts[localSID];
        
        int dimension = cellData->dimension;
        int numip = ref_ip.extent(0);
        
        DRV bip("boundary ip", numElem, numip, dimension);
        DRV bijac("bijac", numElem, numip, dimension, dimension);
        DRV bijacDet("bijacDet", numElem, numip);
        DRV bijacInv("bijacInv", numElem, numip, dimension, dimension);
        DRV bwts("boundary wts", numElem, numip);
        DRV bnormals("boundary normals", numElem, numip, dimension);
        DRV btangents("boundary tangents", numElem, numip, dimension);
        
        {
          //Teuchos::TimeMonitor dbgtimer(*worksetDebugTimer0);
          CellTools::mapToPhysicalFrame(bip, ref_ip, bnodes, *(cellData->cellTopo));
          CellTools::setJacobian(bijac, ref_ip, bnodes, *(cellData->cellTopo));
          CellTools::setJacobianInv(bijacInv, bijac);
          CellTools::setJacobianDet(bijacDet, bijac);
        }
        
        boundaryIP.push_back(bip);
        
        {
          //Teuchos::TimeMonitor dbgtimer(*worksetDebugTimer1);
          
          if (dimension == 2) {
            DRV ref_tangents = cellData->ref_side_tangents[localSID];
            Intrepid2::RealSpaceTools<AssemblyExec>::matvec(btangents, bijac, ref_tangents);
            
            DRV rotation("rotation matrix",dimension,dimension);
            rotation(0,0) = 0;  rotation(0,1) = 1;
            rotation(1,0) = -1; rotation(1,1) = 0;
            Intrepid2::RealSpaceTools<AssemblyExec>::matvec(bnormals, rotation, btangents);
            
            Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(bwts, btangents, Intrepid2::NORM_TWO);
            Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(bwts, bwts, ref_wts);
            
          }
          else if (dimension == 3) {
            
            DRV ref_tangentsU = cellData->ref_side_tangentsU[localSID];
            DRV ref_tangentsV = cellData->ref_side_tangentsV[localSID];
            
            DRV faceTanU("face tangent U", numElem, numip, dimension);
            DRV faceTanV("face tangent V", numElem, numip, dimension);
            
            Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanU, bijac, ref_tangentsU);
            Intrepid2::RealSpaceTools<AssemblyExec>::matvec(faceTanV, bijac, ref_tangentsV);
            
            Intrepid2::RealSpaceTools<AssemblyExec>::vecprod(bnormals, faceTanU, faceTanV);
            
            Intrepid2::RealSpaceTools<AssemblyExec>::vectorNorm(bwts, bnormals, Intrepid2::NORM_TWO);
            Intrepid2::ArrayTools<AssemblyExec>::scalarMultiplyDataData(bwts, bwts, ref_wts);
            
          }
          
          boundaryWts.push_back(bwts);
        }
        
        Kokkos::View<ScalarT*,AssemblyDevice> bhsize("element sizes",numElem);
        parallel_for(RangePolicy<AssemblyExec>(0,bwts.extent(0)), KOKKOS_LAMBDA (const int e ) {
          ScalarT vol = 0.0;
          for (int i=0; i<bwts.extent(1); i++) {
            vol += bwts(e,i);
          }
          ScalarT dimscl = 1.0/((ScalarT)ip.extent(2)-1.0);
          bhsize(e) = std::pow(vol,dimscl);
        });
        boundaryHsize.push_back(bhsize);
        
        // TMW: this might not be needed
        // scale the normal vector (we need unit normal...)
        parallel_for(RangePolicy<AssemblyExec>(0,bnormals.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (int j=0; j<bnormals.extent(1); j++ ) {
            ScalarT normalLength = 0.0;
            for (int sd=0; sd<bnormals.extent(2); sd++) {
              normalLength += bnormals(e,j,sd)*bnormals(e,j,sd);
            }
            normalLength = std::sqrt(normalLength);
            for (int sd=0; sd<bnormals.extent(2); sd++) {
              bnormals(e,j,sd) = bnormals(e,j,sd) / normalLength;
            }
          }
        });
        boundaryNormals.push_back(bnormals);
        
        {
          vector<DRV> cbasis, cbasis_grad;
          
          for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
            
            int numb = cellData->basis_pointers[i]->getCardinality();
            DRV basis_vals, basis_grad_vals;//, basis_div_vals, basis_curl_vals;
            
            DRV ref_basis_vals = cellData->ref_side_basis[localSID][i];
            
            if (cellData->basis_types[i] == "HGRAD"){
              
              DRV basis_vals_tmp("tmp basis_vals",numElem, numb, numip);
              FuncTools::HGRADtransformVALUE(basis_vals_tmp, ref_basis_vals);
              basis_vals = DRV("basis_vals",numElem, numb, numip);
              OrientTools::modifyBasisByOrientation(basis_vals, basis_vals_tmp, orientation[bcell],
                                                    cellData->basis_pointers[i].get());
              
              DRV ref_basis_grad_vals = cellData->ref_side_basis_grad[localSID][i];
              
              DRV basis_grad_vals_tmp("basis_grad_side tmp",numElem,numb,numip,dimension);
              FuncTools::HGRADtransformGRAD(basis_grad_vals_tmp, bijacInv, ref_basis_grad_vals);
              
              basis_grad_vals = DRV("basis_grad_vals",numElem,numb,numip,dimension);
              OrientTools::modifyBasisByOrientation(basis_grad_vals, basis_grad_vals_tmp, orientation[bcell],
                                                    cellData->basis_pointers[i].get());
              
            }
            else if (cellData->basis_types[i] == "HVOL"){ // does not require orientations
              
              basis_vals = DRV("basis_vals",numElem, numb, numip);
              FuncTools::HGRADtransformVALUE(basis_vals, ref_basis_vals);
              
            }
            else if (cellData->basis_types[i] == "HFACE"){
              
              DRV basis_vals_tmp("tmp basis_vals",numElem, numb, numip);
              FuncTools::HGRADtransformVALUE(basis_vals_tmp, ref_basis_vals);
              basis_vals = DRV("basis_vals",numElem, numb, numip);
              OrientTools::modifyBasisByOrientation(basis_vals, basis_vals_tmp, orientation[bcell],
                                                    cellData->basis_pointers[i].get());
              
            }
            else if (cellData->basis_types[i] == "HDIV"){
              
              DRV basis_vals_tmp("tmp basis_vals",numElem, numb, numip, dimension);
              
              FuncTools::HDIVtransformVALUE(basis_vals_tmp, bijac, bijacDet, ref_basis_vals);
              basis_vals = DRV("basis_vals",numElem, numb, numip, dimension);
              OrientTools::modifyBasisByOrientation(basis_vals, basis_vals_tmp, orientation[bcell],
                                                    cellData->basis_pointers[i].get());
            }
            else if (cellData->basis_types[i] == "HCURL"){
              
            }
            cbasis.push_back(basis_vals);
            cbasis_grad.push_back(basis_grad_vals);
            //basis_div.push_back(basis_div_vals);
            //basis_curl.push_back(basis_curl_vals);
          }
          boundaryBasis.push_back(cbasis);
          boundaryBasisGrad.push_back(cbasis_grad);
        }
      }
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void computeMacroBasisVolIP(topo_RCP & macro_cellTopo, vector<basis_RCP> & macro_basis_pointers,
                                Teuchos::RCP<discretization> & disc) {
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
        currcell_basis.push_back(disc->evaluateBasis(macro_basis_pointers[i], sref_ip, macroorientation));
        currcell_basisGrad.push_back(disc->evaluateBasisGrads(macro_basis_pointers[i], macronodes,
                                                              sref_ip, macro_cellTopo, macroorientation));
      }
      
      aux_basis.push_back(currcell_basis);
      aux_basis_grad.push_back(currcell_basisGrad);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void computeMacroBasisBoundaryIP(topo_RCP & macro_cellTopo, vector<basis_RCP> & macro_basis_pointers,
                                     Teuchos::RCP<discretization> & disc) {
      //Teuchos::RCP<workset> & wkset) {
      
      for (size_t e=0; e<boundaryNodes.size(); e++) {
        
        int numElem = boundaryNodes[e].extent(0);
        
        //DRV sside_ip = wkset[0]->ip_side_vec[boundaryCells[mindex][e]->wksetBID];
        DRV sside_ip = boundaryIP[e];//wkset->ip_side_vec[BIDs[e]];
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
            DRV bvals = disc->evaluateBasis(macro_basis_pointers[i], sref_side_ip, corientation);
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
    
    void setBoundaryIndexLIDs() {
      for (size_t b=0; b<boundaryMIDs.size(); b++) {
        LIDView cLIDs("boundary macro LIDs",boundaryMIDs[b].size(),macroLIDs.extent(1));
        //Kokkos::View<LO***,AssemblyDevice> cindex("boundary macro GIDs",boundaryMIDs[b].size(),macroindex.extent(1),
        //                                      macroindex.extent(2));
        for (size_t e=0; e<boundaryMIDs[b].size(); e++) {
          size_t mid = boundaryMIDs[b][e];
          for (size_t i=0; i<cLIDs.extent(1); i++) {
            cLIDs(e,i) = macroLIDs(mid,i);
          }
          //for (size_t i=0; i<cindex.extent(1); i++) {
          //  for (size_t j=0; j<cindex.extent(2); j++) {
          //    cindex(e,i,j) = macroindex(mid,i,j);
          //  }
          //}
        }
        boundaryMacroLIDs.push_back(cLIDs);
        //boundaryMacroindex.push_back(cindex);
      }
    }
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    DRV macronodes, nodes, ip, wts;//jacobian, jacobianInv, jacobianDet;
    vector<DRV> boundaryIP, boundaryWts, boundaryNormals;
    Kokkos::View<ScalarT*,AssemblyDevice> hsize;
    vector<Kokkos::View<ScalarT*,AssemblyDevice> > boundaryHsize;
    
    Kokkos::View<int****,HostDevice> macrosideinfo, sideinfo;
    LIDView macroLIDs;
    vector<LIDView> boundaryMacroLIDs;
    vector<Kokkos::View<LO***,AssemblyDevice> > boundaryMacroindex;
    
    Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> macroorientation;
    
    vector<DRV> basis, basis_grad, basis_div, basis_curl;
    vector<vector<DRV> > boundaryBasis, boundaryBasisGrad;
    
    Kokkos::View<int**,UnifiedDevice> bcs;
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
  
}

#endif

