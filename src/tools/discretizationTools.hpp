/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef DISCTOOLS_H
#define DISCTOOLS_H

#include "trilinos.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid2_HVOL_C0_FEM.hpp"

// HDIV functionality
#include "Intrepid2_HDIV_QUAD_I1_FEM.hpp"
#include "Intrepid2_HDIV_QUAD_In_FEM.hpp"
#include "Intrepid2_HDIV_HEX_I1_FEM.hpp"
#include "Intrepid2_HDIV_HEX_In_FEM.hpp"
#include "Intrepid2_HDIV_TRI_I1_FEM.hpp"
#include "Intrepid2_HDIV_TRI_In_FEM.hpp"
#include "Intrepid2_HDIV_TET_I1_FEM.hpp"
#include "Intrepid2_HDIV_TET_In_FEM.hpp"

// HCURL functionality
#include "Intrepid2_HCURL_QUAD_I1_FEM.hpp"
#include "Intrepid2_HCURL_QUAD_In_FEM.hpp"
#include "Intrepid2_HCURL_HEX_I1_FEM.hpp"
#include "Intrepid2_HCURL_HEX_In_FEM.hpp"
#include "Intrepid2_HCURL_TRI_I1_FEM.hpp"
#include "Intrepid2_HCURL_TRI_In_FEM.hpp"
#include "Intrepid2_HCURL_TET_I1_FEM.hpp"
#include "Intrepid2_HCURL_TET_In_FEM.hpp"

#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Utils.hpp"

#include "preferences.hpp"
typedef Kokkos::DynRankView<ScalarT,AssemblyDevice> DRV;
typedef Kokkos::DynRankView<int,AssemblyDevice> DRVint;
typedef Teuchos::RCP<Intrepid2::Basis<AssemblyDevice, ScalarT, ScalarT > > basis_RCP;

class DiscTools {
  
public:
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate basis at reference element integration points (should be deprecated)
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts) {
    using namespace Intrepid2;
    int numCells = 1;//evalpts.dimension(0);
    int numpts = evalpts.dimension(0);
    int numBasis = basis_pointer->getCardinality();
    DRV basisvals("basisvals", numBasis, numpts); 
    basis_pointer->getValues(basisvals, evalpts, OPERATOR_VALUE);
    
    DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
    
    return basisvals_Transformed;
  }
  
  static Teuchos::RCP<DRV> evaluateBasisRCP(const basis_RCP & basis_pointer, const DRV & evalpts) {
    using namespace Intrepid2;

    int numCells = 1;//evalpts.dimension(0);
    int numpts = evalpts.dimension(0);
    int numBasis = basis_pointer->getCardinality();
    DRV basisvals("basisvals", numBasis, numpts);
    basis_pointer->getValues(basisvals, evalpts, OPERATOR_VALUE);
    
    Teuchos::RCP<DRV> basisvals_Transformed = Teuchos::rcp( new DRV("basisvals_Transformed", numCells, numBasis, numpts));
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(*basisvals_Transformed, basisvals);
    
    return basisvals_Transformed;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate basis functions at side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV evaluateSideBasis(const basis_RCP & basis_pointer, const DRV & evalpts,
                              const topo_RCP & cellTopo, const int & side) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1)+1;
    
    DRV refSidePoints("refSidePoints", numpts, spaceDim);
    
    CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
    
    int numBasis = basis_pointer->getCardinality();
    DRV basisvals("basisvals", numBasis, numpts); 
    basis_pointer->getValues(basisvals, refSidePoints, OPERATOR_VALUE);
    
    DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
    
    return basisvals_Transformed;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate weighted basis functions at integration points (single element)
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV evaluateBasisWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                                  const DRV & evalpts, const DRV & evalwts, 
                                  const topo_RCP & cellTopo) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1);
    int numBasis = basis_pointer->getCardinality();
    DRV basisvals("basisvals", numBasis, numpts); 
    basis_pointer->getValues(basisvals, evalpts, OPERATOR_VALUE);
    DRV weightedMeasure("weightedMeasure", numCells, numpts);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobDet("jacobDet", numCells, numpts);
    CellTools<AssemblyDevice>::setJacobian(jacobian, evalpts, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianDet(jacobDet, jacobian);
    FunctionSpaceTools<AssemblyDevice>::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, evalwts);
    
    DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
    DRV basisvals_TransformedWeighted("basisvals_TransformedWeighted", numCells, numBasis, numpts);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
    FunctionSpaceTools<AssemblyDevice>::multiplyMeasure<ScalarT>(basisvals_TransformedWeighted, weightedMeasure, basisvals_Transformed);
    
    return basisvals_TransformedWeighted;
  }
  
  static Teuchos::RCP<DRV> evaluateBasisWeightedRCP(const basis_RCP & basis_pointer, const DRV & nodes,
                                  const DRV & evalpts, const DRV & evalwts,
                                  const topo_RCP & cellTopo) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1);
    int numBasis = basis_pointer->getCardinality();
    DRV basisvals("basisvals", numBasis, numpts);
    basis_pointer->getValues(basisvals, evalpts, OPERATOR_VALUE);
    DRV weightedMeasure("weightedMeasure", numCells, numpts);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobDet("jacobDet", numCells, numpts);
    CellTools<AssemblyDevice>::setJacobian(jacobian, evalpts, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianDet(jacobDet, jacobian);
    FunctionSpaceTools<AssemblyDevice>::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, evalwts);
    
    DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
    Teuchos::RCP<DRV> basisvals_TransformedWeighted = Teuchos::rcp(new DRV("basisvals_TransformedWeighted", numCells, numBasis, numpts));
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
    FunctionSpaceTools<AssemblyDevice>::multiplyMeasure<ScalarT>(*basisvals_TransformedWeighted, weightedMeasure, basisvals_Transformed);
    
    return basisvals_TransformedWeighted;
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate weighted basis functions at side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV evaluateSideBasisWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                                      const DRV & evalpts, const DRV & evalwts,
                                      const topo_RCP & cellTopo, const int & side) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1)+1;
    DRV refSidePoints("refSidePoints",numpts, spaceDim);
    CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
    
    int numBasis = basis_pointer->getCardinality();
    DRV basisvals("basisvals",numBasis, numpts); 
    basis_pointer->getValues(basisvals, refSidePoints, OPERATOR_VALUE);
    DRV sideweightedMeasure("sideweightedMeasure", numCells, numpts);
    
    DRV sideJacobian("sideJacobian",numCells, numpts, spaceDim, spaceDim);
    DRV sideJacobDet("sideJacobDet",numCells, numpts);
    
    CellTools<AssemblyDevice>::setJacobian(sideJacobian, refSidePoints, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianDet(sideJacobDet, sideJacobian);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobDet("jacobDet", numCells, numpts);

    DRV temporary_buffer("temporary_buffer",numCells*numpts*spaceDim*spaceDim);
    //CellTools<AssemblyDevice>::setJacobian(jacobian, evalpts, nodes, *cellTopo);
   
    if (spaceDim == 2) 
      FunctionSpaceTools<AssemblyDevice>::computeEdgeMeasure(sideweightedMeasure, sideJacobian, evalwts, side, *cellTopo, temporary_buffer);
    if (spaceDim == 3) 
      FunctionSpaceTools<AssemblyDevice>::computeFaceMeasure<ScalarT>(sideweightedMeasure, sideJacobian, evalwts, side, *cellTopo, temporary_buffer);
    
    DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
    DRV basisvals_TransformedWeighted("basisvals_TransformedWeighted", numCells, numBasis, numpts);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformVALUE(basisvals_Transformed, basisvals);
    FunctionSpaceTools<AssemblyDevice>::multiplyMeasure<ScalarT>(basisvals_TransformedWeighted, sideweightedMeasure, basisvals_Transformed);
    
    return basisvals_TransformedWeighted;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate basis derivaties at integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes, 
                               const DRV & evalpts, const topo_RCP & cellTopo) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1);
    int numBasis = basis_pointer->getCardinality();
    DRV basisgrads("basisgrads", numBasis, numpts, spaceDim); 
    DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
    basis_pointer->getValues(basisgrads, evalpts, OPERATOR_GRAD);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
    CellTools<AssemblyDevice>::setJacobian(jacobian, evalpts, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianInv(jacobInv, jacobian);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
    
    return basisgrads_Transformed;
  }
  
  static Teuchos::RCP<DRV> evaluateBasisGradsRCP(const basis_RCP & basis_pointer, const DRV & nodes,
                               const DRV & evalpts, const topo_RCP & cellTopo) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1);
    int numBasis = basis_pointer->getCardinality();
    DRV basisgrads("basisgrads", numBasis, numpts, spaceDim);
    Teuchos::RCP<DRV> basisgrads_Transformed = Teuchos::rcp(new DRV("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim));
    basis_pointer->getValues(basisgrads, evalpts, OPERATOR_GRAD);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
    CellTools<AssemblyDevice>::setJacobian(jacobian, evalpts, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianInv(jacobInv, jacobian);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(*basisgrads_Transformed, jacobInv, basisgrads);
    
    return basisgrads_Transformed;
  }

  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate basis derivatives at side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV evaluateSideBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes, 
                                   const DRV & evalpts, const topo_RCP & cellTopo, const int & side) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1)+1;
    DRV refSidePoints("refSidePoints", numpts, spaceDim);
    CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
    
    int numBasis = basis_pointer->getCardinality();
    DRV basisgrads("basisgrads", numBasis, numpts, spaceDim); 
    DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
    basis_pointer->getValues(basisgrads, refSidePoints, OPERATOR_GRAD);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
    //CellTools<AssemblyDevice>::setJacobian(jacobian, cubSidePoints, I_elemNodes, *cellTopo);
    //CellTools<AssemblyDevice>::setJacobian(jacobian, refSidePoints, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobian(jacobian, refSidePoints, nodes, *cellTopo);
    
    CellTools<AssemblyDevice>::setJacobianInv(jacobInv, jacobian);
    
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
    
    return basisgrads_Transformed;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate weighted basis derivatives at integration points (single element)
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV evaluateBasisGradsWeighted(const basis_RCP & basis_pointer, const DRV & nodes, 
                                       const DRV & evalpts, const DRV & evalwts,
                                       const topo_RCP & cellTopo) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1);
    int numBasis = basis_pointer->getCardinality();
    DRV basisgrads("basisgrads", numBasis, numpts, spaceDim); 
    DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
    DRV basisgrads_TransformedWeighted("basisgrads_TransformedWeighted", numCells, numBasis, numpts, spaceDim);
    basis_pointer->getValues(basisgrads, evalpts, OPERATOR_GRAD);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
    DRV jacobDet("jacobDet", numCells, numpts);
    CellTools<AssemblyDevice>::setJacobian(jacobian, evalpts, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianInv(jacobInv, jacobian);
    CellTools<AssemblyDevice>::setJacobianDet(jacobDet, jacobian);
    DRV weightedMeasure("weightedMeasure", numCells, numpts);
    
    FunctionSpaceTools<AssemblyDevice>::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, evalwts);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
    FunctionSpaceTools<AssemblyDevice>::multiplyMeasure<ScalarT>(basisgrads_TransformedWeighted, weightedMeasure, basisgrads_Transformed);
    
    return basisgrads_TransformedWeighted;
  }
  
  static Teuchos::RCP<DRV> evaluateBasisGradsWeightedRCP(const basis_RCP & basis_pointer, const DRV & nodes,
                                       const DRV & evalpts, const DRV & evalwts,
                                       const topo_RCP & cellTopo) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1);
    int numBasis = basis_pointer->getCardinality();
    DRV basisgrads("basisgrads", numBasis, numpts, spaceDim);
    DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
    Teuchos::RCP<DRV> basisgrads_TransformedWeighted = Teuchos::rcp( new DRV("basisgrads_TransformedWeighted", numCells, numBasis, numpts, spaceDim));
    basis_pointer->getValues(basisgrads, evalpts, OPERATOR_GRAD);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
    DRV jacobDet("jacobDet", numCells, numpts);
    CellTools<AssemblyDevice>::setJacobian(jacobian, evalpts, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianInv(jacobInv, jacobian);
    CellTools<AssemblyDevice>::setJacobianDet(jacobDet, jacobian);
    DRV weightedMeasure("weightedMeasure", numCells, numpts);
    
    FunctionSpaceTools<AssemblyDevice>::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, evalwts);
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
    FunctionSpaceTools<AssemblyDevice>::multiplyMeasure<ScalarT>(*basisgrads_TransformedWeighted, weightedMeasure, basisgrads_Transformed);
    
    return basisgrads_TransformedWeighted;
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate weighted basis derivatives at side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV evaluateSideBasisGradsWeighted(const basis_RCP & basis_pointer, const DRV & nodes, 
                                           const DRV & evalpts, const DRV & evalwts,
                                           const topo_RCP & cellTopo, const int & side) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1)+1;
    DRV refSidePoints("refSidePoints", numpts, spaceDim);
    CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
    
    int numBasis = basis_pointer->getCardinality();
    DRV basisgrads("basisgrads", numBasis, numpts, spaceDim); 
    DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
    DRV basisgrads_TransformedWeighted("basisgrads_TransformedWeighted", numCells, numBasis, numpts, spaceDim);
    basis_pointer->getValues(basisgrads, refSidePoints, OPERATOR_GRAD);
    DRV sideweightedMeasure("sideweightedMeasure", numCells, numpts);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
    //CellTools<AssemblyDevice>::setJacobian(jacobian, cubSidePoints, I_elemNodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobian(jacobian, refSidePoints, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianInv(jacobInv, jacobian);

    DRV temporary_buffer("temporary_buffer",numCells*numpts*spaceDim*spaceDim);
    
    FunctionSpaceTools<AssemblyDevice>::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
    if (spaceDim == 2) 
      FunctionSpaceTools<AssemblyDevice>::computeEdgeMeasure<ScalarT>(sideweightedMeasure, jacobian, evalwts, side, *cellTopo, temporary_buffer);
    if (spaceDim == 3) 
      FunctionSpaceTools<AssemblyDevice>::computeFaceMeasure<ScalarT>(sideweightedMeasure, jacobian, evalwts, side, *cellTopo, temporary_buffer);

    FunctionSpaceTools<AssemblyDevice>::multiplyMeasure<ScalarT>(basisgrads_TransformedWeighted, sideweightedMeasure, basisgrads_Transformed);
    
    return basisgrads_TransformedWeighted;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Compute the normals at the side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV evaluateSideNormals(const DRV & nodes, const DRV & evalpts,
                                const topo_RCP & cellTopo, const int & side) {
    using namespace Intrepid2;

    int numCells = 1;
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1)+1;
    DRV sideJacobian("sideJacobian", numCells, numpts, spaceDim, spaceDim);
    DRV refSidePoints("refSidePoints", numpts, spaceDim);
    CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
    CellTools<AssemblyDevice>::setJacobian(sideJacobian, refSidePoints, nodes, *cellTopo);
    
    // compute normal vector
    DRV normal("normal", numCells, numpts, spaceDim);
    CellTools<AssemblyDevice>::getPhysicalSideNormals(normal, sideJacobian, side, *cellTopo);
    
    // scale the normal vector (we need unit normal...)
    for( int j=0; j<numpts; j++ ) {
      ScalarT normalLength = 0.0;
      for (int sd=0; sd<spaceDim; sd++) {
        normalLength += normal(0,j,sd)*normal(0,j,sd);
      }
      normalLength = sqrt(normalLength);
      for (int sd=0; sd<spaceDim; sd++) {
        normal(0,j,sd) = normal(0,j,sd) / normalLength;
      }
    } 
    
    return normal;
  }
  
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Compute the physical integration weights on one element
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV getPhysicalWts(const DRV & nodes, const DRV & evalpts, const DRV & evalwts, 
                           const topo_RCP & cellTopo) {
    using namespace Intrepid2;

    int numCells = 1;//evalpts.dimension(0);
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1);
    DRV wts("wts", numCells,numpts);
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    DRV jacobDet("jacobDet", numCells, numpts);
    
    // Compute cell Jacobians, their inverses and their determinants
    CellTools<AssemblyDevice>::setJacobian(jacobian, evalpts, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianDet(jacobDet, jacobian);
    
    // compute weighted measure
    FunctionSpaceTools<AssemblyDevice>::computeCellMeasure<ScalarT>(wts, jacobDet, evalwts);
    return wts;      
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the physical integration points on one element
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV getPhysicalIP(const DRV & nodes, const DRV & evalpts, const topo_RCP & cellTopo) {
    using namespace Intrepid2;

    int numCells = 1;//evalpts.dimension(0);
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1);
    DRV ip("ip",numCells,numpts,spaceDim);
    CellTools<AssemblyDevice>::mapToPhysicalFrame(ip, evalpts, nodes, *cellTopo);
    return ip;      
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the physical side integration points on one element/side
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV getPhysicalSideIP(const DRV & nodes, const DRV & evalpts, 
                              const topo_RCP & cellTopo, const int & s) {
    using namespace Intrepid2;

    int numCells = 1;//evalpts.dimension(0);
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1)+1;
    DRV ip("ip", numCells,numpts,spaceDim);
    DRV refSidePoints("refSidePoints", numpts, spaceDim);
    CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, s, *cellTopo);
    CellTools<AssemblyDevice>::mapToPhysicalFrame(ip, refSidePoints, nodes, *cellTopo);
    return ip;      
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the physical side integration weights on one element/side
  //////////////////////////////////////////////////////////////////////////////////////
  
  static DRV getPhysicalSideWts(const DRV & nodes, const DRV & evalpts, const DRV & evalwts,
                              const topo_RCP & cellTopo, const int & s) {
    using namespace Intrepid2;

    int numCells = 1;//evalpts.dimension(0);
    int numpts = evalpts.dimension(0);
    int spaceDim = evalpts.dimension(1)+1;
    DRV ip("ip", numCells,numpts,spaceDim);
    DRV refSidePoints("refSidePoints", numpts, spaceDim);
    CellTools<AssemblyDevice>::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, s, *cellTopo);
    CellTools<AssemblyDevice>::mapToPhysicalFrame(ip, refSidePoints, nodes, *cellTopo);
    DRV wts("wts", numCells, numpts);
    
    DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
    CellTools<AssemblyDevice>::setJacobian(jacobian, refSidePoints, nodes, *cellTopo);

    DRV temporary_buffer("temporary_buffer",numCells*numpts*spaceDim*spaceDim);

    if (spaceDim == 2)
      FunctionSpaceTools<AssemblyDevice>::computeEdgeMeasure<ScalarT>(wts, jacobian, evalwts, s, *cellTopo, temporary_buffer);
    else if (spaceDim ==3)
      FunctionSpaceTools<AssemblyDevice>::computeFaceMeasure<ScalarT>(wts, jacobian, evalwts, s, *cellTopo, temporary_buffer);

    return wts;      
  }

  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
  
  static ScalarT getElementSize(const DRV & nodes, const DRV & ip, const DRV & wts,
                               const topo_RCP & cellTopo) {
    using namespace Intrepid2;

    int numip = ip.dimension(0);
    int spaceDim = ip.dimension(1);
    DRV jacobian("jacobian", 1, numip, spaceDim, spaceDim);
    DRV jacobDet("jacobDet", 1, numip);
    DRV weightedMeasure("weightedMeasure", 1, numip);
    CellTools<AssemblyDevice>::setJacobian(jacobian, ip, nodes, *cellTopo);
    CellTools<AssemblyDevice>::setJacobianDet(jacobDet, jacobian);
    FunctionSpaceTools<AssemblyDevice>::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, wts);
    
    ScalarT vol = 0.0;
    ScalarT h;
    for(int i=0; i<numip; i++) {
      vol += weightedMeasure(0,i);
    }
    if (spaceDim == 1) 
      h = abs(vol);
    else if (spaceDim == 2) 
      h = sqrt(vol);
    else if (spaceDim == 3) 
      h = pow(vol,1.0/3.0);
    
    return h;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
  
  static void getQuadrature(const topo_RCP & cellTopo, const int & order, DRV & ip, DRV & wts) {
    using namespace Intrepid2;

    DefaultCubatureFactory cubFactory;
    Teuchos::RCP<Cubature<AssemblyDevice> > basisCub  = cubFactory.create<AssemblyDevice, ScalarT, ScalarT>(*cellTopo, order); // TMW: the mesh sublist is not the correct place
    int cubDim  = basisCub->getDimension();
    int numCubPoints = basisCub->getNumPoints();
    ip = DRV("ip", numCubPoints, cubDim);
    wts = DRV("wts", numCubPoints);
    basisCub->getCubature(ip, wts);
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Create a pointer to an Intrepid or Panzer basis
  //////////////////////////////////////////////////////////////////////////////////////
  
  static basis_RCP getBasis(const int & spaceDim, const topo_RCP & cellTopo, 
                            const string & type, const int & degree) {
    using namespace Intrepid2;

    basis_RCP basis;
   
    string shape = cellTopo->getName();
 
    if (type == "HGRAD") {
      if (spaceDim == 1) {
        basis = Teuchos::rcp(new Basis_HGRAD_LINE_C1_FEM<AssemblyDevice>() ); 
      }
      if (spaceDim == 2) { 
        if (shape == "Quadrilateral_4") {
          if (degree == 0)
            basis = Teuchos::rcp(new Basis_HVOL_C0_FEM<AssemblyDevice>(*cellTopo));
          else if (degree == 1) {
            basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C1_FEM<AssemblyDevice>() );
          }
          else if (degree == 2) {
            basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C2_FEM<AssemblyDevice>() );
          }
          else {
            basis = Teuchos::rcp(new Basis_HGRAD_QUAD_Cn_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
        if (shape == "Triangle_3") {
          if (degree == 1) 
            basis = Teuchos::rcp(new Basis_HGRAD_TRI_C1_FEM<AssemblyDevice>() ); 
          else if (degree == 2)
            basis = Teuchos::rcp(new Basis_HGRAD_TRI_C2_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HGRAD_TRI_Cn_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        } 
      }
      if (spaceDim == 3) { 
        if (shape == "Hexahedron_8") {
          if (degree == 0)
            basis = Teuchos::rcp(new Basis_HVOL_C0_FEM<AssemblyDevice>(*cellTopo));
          else if (degree  == 1)
            basis = Teuchos::rcp(new Basis_HGRAD_HEX_C1_FEM<AssemblyDevice>() ); 
          else if (degree  == 2)
            basis = Teuchos::rcp(new Basis_HGRAD_HEX_C2_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HGRAD_HEX_Cn_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
        if (shape == "Tetrahedron_4") {
          if (degree == 1) 
            basis = Teuchos::rcp(new Basis_HGRAD_TET_C1_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HGRAD_TET_Cn_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
      }
    }
    else if (type == "HDIV") {
      if (spaceDim == 1) {
        // need to throw an error
      }
      else if (spaceDim == 2) {
        if (shape == "Quadrilateral_4") {
          if (degree == 1)
            basis = Teuchos::rcp(new Basis_HDIV_QUAD_I1_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HDIV_QUAD_In_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
        else if (shape == "Triangle_3") {
          if (degree == 1)
            basis = Teuchos::rcp(new Basis_HDIV_TRI_I1_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HDIV_TRI_In_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
      }
      else if (spaceDim == 3) {
        if (shape == "Hexahedron_8") {
          if (degree  == 1)
            basis = Teuchos::rcp(new Basis_HDIV_HEX_I1_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HDIV_HEX_In_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
        else if (shape == "Tetrahedron_4") {
          if (degree == 1)
            basis = Teuchos::rcp(new Basis_HDIV_TET_I1_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HDIV_TET_In_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
      }

    }
    else if (type == "HCURL") {
      if (spaceDim == 1) {
        // need to throw an error
      }
      else if (spaceDim == 2) {
        if (shape == "Quadrilateral_4") {
          if (degree == 1)
            basis = Teuchos::rcp(new Basis_HCURL_QUAD_I1_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HCURL_QUAD_In_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
        else if (shape == "Triangle_3") {
          if (degree == 1)
            basis = Teuchos::rcp(new Basis_HCURL_TRI_I1_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HCURL_TRI_In_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
      }
      else if (spaceDim == 3) {
        if (shape == "Hexahedron_8") {
          if (degree  == 1)
            basis = Teuchos::rcp(new Basis_HCURL_HEX_I1_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HCURL_HEX_In_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
        else if (shape == "Tetrahedron_4") {
          if (degree == 1)
            basis = Teuchos::rcp(new Basis_HCURL_TET_I1_FEM<AssemblyDevice>() );
          else {
            basis = Teuchos::rcp(new Basis_HCURL_TET_In_FEM<AssemblyDevice>(degree,POINTTYPE_EQUISPACED) );
          }
        }
      }

    }
    else if (type == "HMORT") {
      
    }
    
    
    return basis;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get a cell topology
  //////////////////////////////////////////////////////////////////////////////////////

  static topo_RCP getCellTopology(const int & dimension, const string & shape) {
    
    topo_RCP cellTopo;
    if (dimension == 1) {
      cellTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ) );// lin. cell topology on the interior
    }
    if (dimension == 2) {
      if (shape == "quad") {
        cellTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ) );// lin. cell topology on the interior
      }
      if (shape == "tri") {
        cellTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() ) );// lin. cell topology on the interior
      }
    }
    if (dimension == 3) {
      if (shape == "hex") {
        cellTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<> >() ) );// lin. cell topology on the interior
      }
      if (shape == "tet") {
        cellTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<> >() ) );// lin. cell topology on the interior
      }
      
    }
    return cellTopo;
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Get a cell side topology
  //////////////////////////////////////////////////////////////////////////////////////
  
  static topo_RCP getCellSideTopology(const int & dimension, const string & shape) {

    topo_RCP sideTopo;
    
    if (dimension == 1) {
      //sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Node<> >() ));
    }
    if (dimension == 2) {
      if (shape == "quad") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
      if (shape == "tri") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
    }
    if (dimension == 3) {
      if (shape == "hex") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ));
      }
      if (shape == "tet") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() ));
      }
    }
    return sideTopo;
  }
  
};
#endif

