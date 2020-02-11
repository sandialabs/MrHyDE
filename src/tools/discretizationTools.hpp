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
#include "preferences.hpp"

class DiscTools {
  
public:
  
  DiscTools() {};
  
  ~DiscTools() {};
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate basis at reference element integration points (should be deprecated)
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts);
  
  Teuchos::RCP<DRV> evaluateBasisRCP(const basis_RCP & basis_pointer, const DRV & evalpts);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate basis functions at side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV evaluateSideBasis(const basis_RCP & basis_pointer, const DRV & evalpts,
                        const topo_RCP & cellTopo, const int & side);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate weighted basis functions at integration points (single element)
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV evaluateBasisWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                            const DRV & evalpts, const DRV & evalwts,
                            const topo_RCP & cellTopo);
  
  Teuchos::RCP<DRV> evaluateBasisWeightedRCP(const basis_RCP & basis_pointer, const DRV & nodes,
                                             const DRV & evalpts, const DRV & evalwts,
                                             const topo_RCP & cellTopo);

  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate weighted basis functions at side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV evaluateSideBasisWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                                const DRV & evalpts, const DRV & evalwts,
                                const topo_RCP & cellTopo, const int & side);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate basis derivaties at integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
                         const DRV & evalpts, const topo_RCP & cellTopo);
  
  Teuchos::RCP<DRV> evaluateBasisGradsRCP(const basis_RCP & basis_pointer, const DRV & nodes,
                                          const DRV & evalpts, const topo_RCP & cellTopo);

  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate basis derivatives at side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV evaluateSideBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
                             const DRV & evalpts, const topo_RCP & cellTopo, const int & side);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate weighted basis derivatives at integration points (single element)
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV evaluateBasisGradsWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                                 const DRV & evalpts, const DRV & evalwts,
                                 const topo_RCP & cellTopo);
  
  Teuchos::RCP<DRV> evaluateBasisGradsWeightedRCP(const basis_RCP & basis_pointer, const DRV & nodes,
                                                  const DRV & evalpts, const DRV & evalwts,
                                                  const topo_RCP & cellTopo);

  //////////////////////////////////////////////////////////////////////////////////////
  // Evaluate weighted basis derivatives at side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV evaluateSideBasisGradsWeighted(const basis_RCP & basis_pointer, const DRV & nodes,
                                     const DRV & evalpts, const DRV & evalwts,
                                     const topo_RCP & cellTopo, const int & side);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Compute the normals at the side integration points
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV evaluateSideNormals(const DRV & nodes, const DRV & evalpts,
                          const topo_RCP & cellTopo, const int & side);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Compute the physical integration weights on one element
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV getPhysicalWts(const DRV & nodes, const DRV & evalpts, const DRV & evalwts,
                     const topo_RCP & cellTopo);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the physical integration points on one element
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV getPhysicalIP(const DRV & nodes, const DRV & evalpts, const topo_RCP & cellTopo);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the physical side integration points on one element/side
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV getPhysicalSideIP(const DRV & nodes, const DRV & evalpts,
                        const topo_RCP & cellTopo, const int & s);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get the physical side integration weights on one element/side
  //////////////////////////////////////////////////////////////////////////////////////
  
  DRV getPhysicalSideWts(const DRV & nodes, const DRV & evalpts, const DRV & evalwts,
                         const topo_RCP & cellTopo, const int & s);

  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
  
  ScalarT getElementSize(const DRV & nodes, const DRV & ip, const DRV & wts,
                         const topo_RCP & cellTopo);
  
  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
  
  void getQuadrature(const topo_RCP & cellTopo, const int & order, DRV & ip, DRV & wts);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Create a pointer to an Intrepid or Panzer basis
  //////////////////////////////////////////////////////////////////////////////////////
  
  basis_RCP getBasis(const int & spaceDim, const topo_RCP & cellTopo,
                     const string & type, const int & degree);
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Get a cell topology
  //////////////////////////////////////////////////////////////////////////////////////

  topo_RCP getCellTopology(const int & dimension, const string & shape);

  //////////////////////////////////////////////////////////////////////////////////////
  // Get a cell side topology
  //////////////////////////////////////////////////////////////////////////////////////
  
  topo_RCP getCellSideTopology(const int & dimension, const string & shape);
  
};
#endif

