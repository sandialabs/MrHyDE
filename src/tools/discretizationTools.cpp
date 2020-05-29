/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "discretizationTools.hpp"

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

// HFACE (experimental) functionality
#include "Intrepid2_HFACE_QUAD_In_FEM.hpp"
#include "Intrepid2_HFACE_TRI_In_FEM.hpp"
#include "Intrepid2_HFACE_HEX_In_FEM.hpp"

#include "Intrepid2_PointTools.hpp"
//#include "Intrepid2_FunctionSpaceTools.hpp"
//#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Utils.hpp"


//////////////////////////////////////////////////////////////////////////////////////
// Evaluate basis at reference element integration points (should be deprecated)
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts) {
  int numCells = 1;//evalpts.extent(0);
  int numpts = evalpts.extent(0);
  int numBasis = basis_pointer->getCardinality();
  DRV basisvals("basisvals", numBasis, numpts);
  basis_pointer->getValues(basisvals, evalpts, Intrepid2::OPERATOR_VALUE);
  
  DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
  FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
  
  return basisvals_Transformed;
}

DRV DiscTools::evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts,
                             Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & orientation) {
  int numCells = 1;//evalpts.extent(0);
  int numpts = evalpts.extent(0);
  int numBasis = basis_pointer->getCardinality();
  DRV basisvals("basisvals", numBasis, numpts);
  basis_pointer->getValues(basisvals, evalpts, Intrepid2::OPERATOR_VALUE);
  
  DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
  FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
  DRV basisvals_to("basisvals_Transformed", numCells, numBasis, numpts);
  OrientTools::modifyBasisByOrientation(basisvals_to, basisvals_Transformed,
                                        orientation, basis_pointer.get());
  
  return basisvals_to;
}

Teuchos::RCP<DRV> DiscTools::evaluateBasisRCP(const basis_RCP & basis_pointer, const DRV & evalpts) {
  
  int numCells = 1;//evalpts.extent(0);
  int numpts = evalpts.extent(0);
  int numBasis = basis_pointer->getCardinality();
  DRV basisvals("basisvals", numBasis, numpts);
  basis_pointer->getValues(basisvals, evalpts, Intrepid2::OPERATOR_VALUE);
  
  Teuchos::RCP<DRV> basisvals_Transformed = Teuchos::rcp( new DRV("basisvals_Transformed", numCells, numBasis, numpts));
  FuncTools::HGRADtransformVALUE(*basisvals_Transformed, basisvals);
  
  return basisvals_Transformed;
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate basis functions at side integration points
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::evaluateSideBasis(const basis_RCP & basis_pointer, const DRV & evalpts,
                                 const topo_RCP & cellTopo, const int & side) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1)+1;
  
  DRV refSidePoints("refSidePoints", numpts, spaceDim);
  
  CellTools::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
  
  int numBasis = basis_pointer->getCardinality();
  DRV basisvals("basisvals", numBasis, numpts);
  basis_pointer->getValues(basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
  
  DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
  FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
  
  return basisvals_Transformed;
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate weighted basis functions at integration points (single element)
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::evaluateBasisWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                                     const DRV & evalpts, const DRV & evalwts,
                                     const topo_RCP & cellTopo) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1);
  int numBasis = basis_pointer->getCardinality();
  DRV basisvals("basisvals", numBasis, numpts);
  basis_pointer->getValues(basisvals, evalpts, Intrepid2::OPERATOR_VALUE);
  DRV weightedMeasure("weightedMeasure", numCells, numpts);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobDet("jacobDet", numCells, numpts);
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianDet(jacobDet, jacobian);
  FuncTools::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, evalwts);
  
  DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
  DRV basisvals_TransformedWeighted("basisvals_TransformedWeighted", numCells, numBasis, numpts);
  FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
  FuncTools::multiplyMeasure<ScalarT>(basisvals_TransformedWeighted, weightedMeasure, basisvals_Transformed);
  
  return basisvals_TransformedWeighted;
}

Teuchos::RCP<DRV> DiscTools::evaluateBasisWeightedRCP(const basis_RCP & basis_pointer, const DRV & nodes,
                                                      const DRV & evalpts, const DRV & evalwts,
                                                      const topo_RCP & cellTopo) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1);
  int numBasis = basis_pointer->getCardinality();
  DRV basisvals("basisvals", numBasis, numpts);
  basis_pointer->getValues(basisvals, evalpts, Intrepid2::OPERATOR_VALUE);
  DRV weightedMeasure("weightedMeasure", numCells, numpts);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobDet("jacobDet", numCells, numpts);
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianDet(jacobDet, jacobian);
  FuncTools::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, evalwts);
  
  DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
  Teuchos::RCP<DRV> basisvals_TransformedWeighted = Teuchos::rcp(new DRV("basisvals_TransformedWeighted", numCells, numBasis, numpts));
  FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
  FuncTools::multiplyMeasure<ScalarT>(*basisvals_TransformedWeighted, weightedMeasure, basisvals_Transformed);
  
  return basisvals_TransformedWeighted;
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate weighted basis functions at side integration points
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::evaluateSideBasisWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                                         const DRV & evalpts, const DRV & evalwts,
                                         const topo_RCP & cellTopo, const int & side) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1)+1;
  DRV refSidePoints("refSidePoints",numpts, spaceDim);
  CellTools::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
  
  int numBasis = basis_pointer->getCardinality();
  DRV basisvals("basisvals",numBasis, numpts);
  basis_pointer->getValues(basisvals, refSidePoints, Intrepid2::OPERATOR_VALUE);
  DRV sideweightedMeasure("sideweightedMeasure", numCells, numpts);
  
  DRV sideJacobian("sideJacobian",numCells, numpts, spaceDim, spaceDim);
  DRV sideJacobDet("sideJacobDet",numCells, numpts);
  
  CellTools::setJacobian(sideJacobian, refSidePoints, nodes, *cellTopo);
  CellTools::setJacobianDet(sideJacobDet, sideJacobian);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobDet("jacobDet", numCells, numpts);
  
  DRV temporary_buffer("temporary_buffer",numCells*numpts*spaceDim*spaceDim);
  //CellTools<AssemblyDevice>::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  
  if (spaceDim == 2)
    FuncTools::computeEdgeMeasure(sideweightedMeasure, sideJacobian, evalwts, side, *cellTopo, temporary_buffer);
  if (spaceDim == 3)
    FuncTools::computeFaceMeasure<ScalarT>(sideweightedMeasure, sideJacobian, evalwts, side, *cellTopo, temporary_buffer);
  
  DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
  DRV basisvals_TransformedWeighted("basisvals_TransformedWeighted", numCells, numBasis, numpts);
  FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
  FuncTools::multiplyMeasure<ScalarT>(basisvals_TransformedWeighted, sideweightedMeasure, basisvals_Transformed);
  
  return basisvals_TransformedWeighted;
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate basis derivaties at integration points
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
                                  const DRV & evalpts, const topo_RCP & cellTopo) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1);
  int numBasis = basis_pointer->getCardinality();
  DRV basisgrads("basisgrads", numBasis, numpts, spaceDim);
  DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
  basis_pointer->getValues(basisgrads, evalpts, Intrepid2::OPERATOR_GRAD);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianInv(jacobInv, jacobian);
  FuncTools::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
  
  return basisgrads_Transformed;
}

DRV DiscTools::evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
                                  const DRV & evalpts, const topo_RCP & cellTopo,
                                  Kokkos::DynRankView<Intrepid2::Orientation,AssemblyDevice> & orientation) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1);
  int numBasis = basis_pointer->getCardinality();
  DRV basisgrads("basisgrads", numBasis, numpts, spaceDim);
  DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
  basis_pointer->getValues(basisgrads, evalpts, Intrepid2::OPERATOR_GRAD);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianInv(jacobInv, jacobian);
  FuncTools::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
  DRV basisgrads_to("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
  OrientTools::modifyBasisByOrientation(basisgrads_to, basisgrads_Transformed,
                                        orientation, basis_pointer.get());
  
  return basisgrads_to;
}

Teuchos::RCP<DRV> DiscTools::evaluateBasisGradsRCP(const basis_RCP & basis_pointer, const DRV & nodes,
                                                   const DRV & evalpts, const topo_RCP & cellTopo) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1);
  int numBasis = basis_pointer->getCardinality();
  DRV basisgrads("basisgrads", numBasis, numpts, spaceDim);
  Teuchos::RCP<DRV> basisgrads_Transformed = Teuchos::rcp(new DRV("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim));
  basis_pointer->getValues(basisgrads, evalpts, Intrepid2::OPERATOR_GRAD);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianInv(jacobInv, jacobian);
  FuncTools::HGRADtransformGRAD(*basisgrads_Transformed, jacobInv, basisgrads);
  
  return basisgrads_Transformed;
}


//////////////////////////////////////////////////////////////////////////////////////
// Evaluate basis derivatives at side integration points
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::evaluateSideBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
                                      const DRV & evalpts, const topo_RCP & cellTopo, const int & side) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1)+1;
  DRV refSidePoints("refSidePoints", numpts, spaceDim);
  CellTools::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
  
  int numBasis = basis_pointer->getCardinality();
  DRV basisgrads("basisgrads", numBasis, numpts, spaceDim);
  DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
  basis_pointer->getValues(basisgrads, refSidePoints, Intrepid2::OPERATOR_GRAD);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
  //CellTools<AssemblyDevice>::setJacobian(jacobian, cubSidePoints, I_elemNodes, *cellTopo);
  //CellTools<AssemblyDevice>::setJacobian(jacobian, refSidePoints, nodes, *cellTopo);
  CellTools::setJacobian(jacobian, refSidePoints, nodes, *cellTopo);
  
  CellTools::setJacobianInv(jacobInv, jacobian);
  
  FuncTools::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
  
  return basisgrads_Transformed;
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate weighted basis derivatives at integration points (single element)
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::evaluateBasisGradsWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                                          const DRV & evalpts, const DRV & evalwts,
                                          const topo_RCP & cellTopo) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1);
  int numBasis = basis_pointer->getCardinality();
  DRV basisgrads("basisgrads", numBasis, numpts, spaceDim);
  DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
  DRV basisgrads_TransformedWeighted("basisgrads_TransformedWeighted", numCells, numBasis, numpts, spaceDim);
  basis_pointer->getValues(basisgrads, evalpts, Intrepid2::OPERATOR_GRAD);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
  DRV jacobDet("jacobDet", numCells, numpts);
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianInv(jacobInv, jacobian);
  CellTools::setJacobianDet(jacobDet, jacobian);
  DRV weightedMeasure("weightedMeasure", numCells, numpts);
  
  FuncTools::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, evalwts);
  FuncTools::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
  FuncTools::multiplyMeasure<ScalarT>(basisgrads_TransformedWeighted, weightedMeasure, basisgrads_Transformed);
  
  return basisgrads_TransformedWeighted;
}

Teuchos::RCP<DRV> DiscTools::evaluateBasisGradsWeightedRCP(const basis_RCP & basis_pointer, const DRV & nodes,
                                                           const DRV & evalpts, const DRV & evalwts,
                                                           const topo_RCP & cellTopo) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1);
  int numBasis = basis_pointer->getCardinality();
  DRV basisgrads("basisgrads", numBasis, numpts, spaceDim);
  DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
  Teuchos::RCP<DRV> basisgrads_TransformedWeighted = Teuchos::rcp( new DRV("basisgrads_TransformedWeighted", numCells, numBasis, numpts, spaceDim));
  basis_pointer->getValues(basisgrads, evalpts, Intrepid2::OPERATOR_GRAD);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
  DRV jacobDet("jacobDet", numCells, numpts);
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianInv(jacobInv, jacobian);
  CellTools::setJacobianDet(jacobDet, jacobian);
  DRV weightedMeasure("weightedMeasure", numCells, numpts);
  
  FuncTools::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, evalwts);
  FuncTools::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
  FuncTools::multiplyMeasure<ScalarT>(*basisgrads_TransformedWeighted, weightedMeasure, basisgrads_Transformed);
  
  return basisgrads_TransformedWeighted;
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate weighted basis derivatives at side integration points
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::evaluateSideBasisGradsWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                                              const DRV & evalpts, const DRV & evalwts,
                                              const topo_RCP & cellTopo, const int & side) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1)+1;
  DRV refSidePoints("refSidePoints", numpts, spaceDim);
  CellTools::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
  
  int numBasis = basis_pointer->getCardinality();
  DRV basisgrads("basisgrads", numBasis, numpts, spaceDim);
  DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, spaceDim);
  DRV basisgrads_TransformedWeighted("basisgrads_TransformedWeighted", numCells, numBasis, numpts, spaceDim);
  basis_pointer->getValues(basisgrads, refSidePoints, Intrepid2::OPERATOR_GRAD);
  DRV sideweightedMeasure("sideweightedMeasure", numCells, numpts);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobInv("jacobInv", numCells, numpts, spaceDim, spaceDim);
  //CellTools<AssemblyDevice>::setJacobian(jacobian, cubSidePoints, I_elemNodes, *cellTopo);
  CellTools::setJacobian(jacobian, refSidePoints, nodes, *cellTopo);
  CellTools::setJacobianInv(jacobInv, jacobian);
  
  DRV temporary_buffer("temporary_buffer",numCells*numpts*spaceDim*spaceDim);
  
  FuncTools::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
  if (spaceDim == 2)
    FuncTools::computeEdgeMeasure<ScalarT>(sideweightedMeasure, jacobian, evalwts, side, *cellTopo, temporary_buffer);
  if (spaceDim == 3)
    FuncTools::computeFaceMeasure<ScalarT>(sideweightedMeasure, jacobian, evalwts, side, *cellTopo, temporary_buffer);
  
  FuncTools::multiplyMeasure<ScalarT>(basisgrads_TransformedWeighted, sideweightedMeasure, basisgrads_Transformed);
  
  return basisgrads_TransformedWeighted;
}

//////////////////////////////////////////////////////////////////////////////////////
// Compute the normals at the side integration points
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::evaluateSideNormals(const DRV & nodes, const DRV & evalpts,
                                   const topo_RCP & cellTopo, const int & side) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1)+1;
  DRV sideJacobian("sideJacobian", numCells, numpts, spaceDim, spaceDim);
  DRV refSidePoints("refSidePoints", numpts, spaceDim);
  CellTools::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, side, *cellTopo);
  CellTools::setJacobian(sideJacobian, refSidePoints, nodes, *cellTopo);
  
  // compute normal vector
  DRV normal("normal", numCells, numpts, spaceDim);
  CellTools::getPhysicalSideNormals(normal, sideJacobian, side, *cellTopo);
  
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

DRV DiscTools::getPhysicalWts(const DRV & nodes, const DRV & evalpts, const DRV & evalwts,
                              const topo_RCP & cellTopo) {
  
  int numCells = 1;//evalpts.extent(0);
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1);
  DRV wts("wts", numCells,numpts);
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  DRV jacobDet("jacobDet", numCells, numpts);
  
  // Compute cell Jacobians, their inverses and their determinants
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianDet(jacobDet, jacobian);
  
  // compute weighted measure
  FuncTools::computeCellMeasure<ScalarT>(wts, jacobDet, evalwts);
  return wts;
}

//////////////////////////////////////////////////////////////////////////////////////
// Get the physical integration points on one element
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::getPhysicalIP(const DRV & nodes, const DRV & evalpts, const topo_RCP & cellTopo) {
  
  int numCells = 1;//evalpts.extent(0);
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1);
  DRV ip("ip",numCells,numpts,spaceDim);
  CellTools::mapToPhysicalFrame(ip, evalpts, nodes, *cellTopo);
  return ip;
}

//////////////////////////////////////////////////////////////////////////////////////
// Get the physical side integration points on one element/side
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::getPhysicalSideIP(const DRV & nodes, const DRV & evalpts,
                                 const topo_RCP & cellTopo, const int & s) {
  
  int numCells = 1;//evalpts.extent(0);
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1)+1;
  DRV ip("ip", numCells,numpts,spaceDim);
  DRV refSidePoints("refSidePoints", numpts, spaceDim);
  CellTools::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, s, *cellTopo);
  CellTools::mapToPhysicalFrame(ip, refSidePoints, nodes, *cellTopo);
  return ip;
}

//////////////////////////////////////////////////////////////////////////////////////
// Get the physical side integration weights on one element/side
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscTools::getPhysicalSideWts(const DRV & nodes, const DRV & evalpts, const DRV & evalwts,
                                  const topo_RCP & cellTopo, const int & s) {
  
  int numCells = 1;//evalpts.extent(0);
  int numpts = evalpts.extent(0);
  int spaceDim = evalpts.extent(1)+1;
  DRV ip("ip", numCells,numpts,spaceDim);
  DRV refSidePoints("refSidePoints", numpts, spaceDim);
  CellTools::mapToReferenceSubcell(refSidePoints, evalpts, spaceDim-1, s, *cellTopo);
  CellTools::mapToPhysicalFrame(ip, refSidePoints, nodes, *cellTopo);
  DRV wts("wts", numCells, numpts);
  
  DRV jacobian("jacobian", numCells, numpts, spaceDim, spaceDim);
  CellTools::setJacobian(jacobian, refSidePoints, nodes, *cellTopo);
  
  DRV temporary_buffer("temporary_buffer",numCells*numpts*spaceDim*spaceDim);
  
  if (spaceDim == 2)
    FuncTools::computeEdgeMeasure<ScalarT>(wts, jacobian, evalwts, s, *cellTopo, temporary_buffer);
  else if (spaceDim ==3)
    FuncTools::computeFaceMeasure<ScalarT>(wts, jacobian, evalwts, s, *cellTopo, temporary_buffer);
  
  return wts;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

ScalarT DiscTools::getElementSize(const DRV & nodes, const DRV & ip, const DRV & wts,
                                  const topo_RCP & cellTopo) {
  
  int numip = ip.extent(0);
  int spaceDim = ip.extent(1);
  DRV jacobian("jacobian", 1, numip, spaceDim, spaceDim);
  DRV jacobDet("jacobDet", 1, numip);
  DRV weightedMeasure("weightedMeasure", 1, numip);
  CellTools::setJacobian(jacobian, ip, nodes, *cellTopo);
  CellTools::setJacobianDet(jacobDet, jacobian);
  FuncTools::computeCellMeasure<ScalarT>(weightedMeasure, jacobDet, wts);
  
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

void DiscTools::getQuadrature(const topo_RCP & cellTopo, const int & order, DRV ip, DRV wts) {
  
  Intrepid2::DefaultCubatureFactory cubFactory;
  Teuchos::RCP<Intrepid2::Cubature<AssemblyExec> > basisCub  = cubFactory.create<AssemblyExec, ScalarT, ScalarT>(*cellTopo, order); // TMW: the mesh sublist is not the correct place
  int cubDim  = basisCub->getDimension();
  int numCubPoints = basisCub->getNumPoints();
  ip = DRV("ip", numCubPoints, cubDim);
  wts = DRV("wts", numCubPoints);
  basisCub->getCubature(ip, wts);
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Create a pointer to an Intrepid or Panzer basis
//////////////////////////////////////////////////////////////////////////////////////

basis_RCP DiscTools::getBasis(const int & spaceDim, const topo_RCP & cellTopo,
                              const string & type, const int & degree) {
  using namespace Intrepid2;
  
  basis_RCP basis;
  
  string shape = cellTopo->getName();
  
  if (type == "HGRAD") {
    if (spaceDim == 1) {
      basis = Teuchos::rcp(new Basis_HGRAD_LINE_C1_FEM<AssemblyExec>() );
    }
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        if (degree == 1) {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C1_FEM<AssemblyExec>() );
        }
        else if (degree == 2) {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C2_FEM<AssemblyExec>() );
        }
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_Cn_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      if (shape == "Triangle_3") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C1_FEM<AssemblyExec>() );
        else if (degree == 2)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C2_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_Cn_FEM<AssemblyExec>(degree,POINTTYPE_WARPBLEND) );
        }
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        if (degree  == 1)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C1_FEM<AssemblyExec>() );
        else if (degree  == 2)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C2_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_Cn_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      if (shape == "Tetrahedron_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HGRAD_TET_C1_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_TET_Cn_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
  }
  else if (type == "HVOL") {
    basis = Teuchos::rcp(new Basis_HVOL_C0_FEM<AssemblyExec>(*cellTopo));
  }
  else if (type == "HDIV") {
    if (spaceDim == 1) {
      // need to throw an error
    }
    else if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HDIV_QUAD_I1_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_QUAD_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Triangle_3") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HDIV_TRI_I1_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_TRI_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    else if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        if (degree  == 1)
          basis = Teuchos::rcp(new Basis_HDIV_HEX_I1_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_HEX_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Tetrahedron_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HDIV_TET_I1_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_TET_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
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
          basis = Teuchos::rcp(new Basis_HCURL_QUAD_I1_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_QUAD_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Triangle_3") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HCURL_TRI_I1_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_TRI_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    else if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        if (degree  == 1)
          basis = Teuchos::rcp(new Basis_HCURL_HEX_I1_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_HEX_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Tetrahedron_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HCURL_TET_I1_FEM<AssemblyExec>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_TET_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    
  }
  else if (type == "HFACE") {
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        basis = Teuchos::rcp(new Basis_HFACE_QUAD_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
      }
      else if (shape == "Triangle_3") {
        basis = Teuchos::rcp(new Basis_HFACE_TRI_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        basis = Teuchos::rcp(new Basis_HFACE_HEX_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
      }
      if (shape == "Tetrahedron_4") {
        
      }
    }
  }
  
  
  return basis;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Get a cell topology
//////////////////////////////////////////////////////////////////////////////////////

topo_RCP DiscTools::getCellTopology(const int & dimension, const string & shape) {
  
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

topo_RCP DiscTools::getCellSideTopology(const int & dimension, const string & shape) {
  
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

