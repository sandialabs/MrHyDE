/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/


//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::setReferenceData(Teuchos::RCP<GroupMetaData> & groupData) {
  
  this->setReferenceIntegrationData(groupData);
  this->setReferenceBasisData(groupData);
  
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::getFrobenius(Teuchos::RCP<GroupMetaData> & groupData,
                                           DRV jacobian, DRV fro) {
  int numip = groupData->ref_ip.extent(0);
  int numElem = fro.extent(0);
  
  DRV jacobianDet("determinant of jacobian", numElem, numip);
  CellTools::setJacobianDet(jacobianDet, jacobian);
  DRV wts("jacobian", numElem, numip);
  FuncTools::computeCellMeasure(wts, jacobianDet, groupData->ref_wts);

  parallel_for("compute measure",
               RangePolicy<AssemblyExec>(0,numElem),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type d1=0; d1<jacobian.extent(2); ++d1) {
      for (size_type d2=0; d2<jacobian.extent(3); ++d2) {
      
        for (size_type pt=0; pt<wts.extent(1); ++pt) {
          fro(elem) += jacobian(elem,pt,d1,d2)*jacobian(elem,pt,d1,d2)*wts(elem,pt);
        }
      }
    }
  });
        
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::getMyNodes(const size_t & block, Kokkos::View<LO*,AssemblyDevice> elemIDs) {
 
  Teuchos::TimeMonitor constructor_timer(*get_nodes_timer);
  vector<size_t> localIds(elemIDs.extent(0));
  auto elemIDs_host = create_mirror_view(elemIDs);
  deep_copy(elemIDs_host, elemIDs);
  
  for (size_type e=0; e<elemIDs_host.extent(0); ++e) {
    localIds[e] = my_elements[block](elemIDs_host(e));;//elemIDs_host(e);//my_elements[block](elemIDs_host(e));
  }
  DRV nodes = mesh->getMyNodes(block, localIds);
  
  return nodes;
}



/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::mapPointsToReference(DRV phys_pts, Kokkos::View<LO*,AssemblyDevice> elemIDs, 
                                                  const size_t & block, topo_RCP & cellTopo) {
  DRV nodes = this->getMyNodes(block, elemIDs);
  DRV ref_pts = this->mapPointsToReference(phys_pts, nodes, cellTopo);
  return ref_pts;

}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::mapPointsToReference(DRV phys_pts, DRV nodes,
                                                  topo_RCP & cellTopo) {
  DRV ref_pts("reference cell points",phys_pts.extent(0), phys_pts.extent(1), phys_pts.extent(2));
  CellTools::mapToReferenceFrame(ref_pts, phys_pts, nodes, *cellTopo);
  return ref_pts;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::getReferenceNodes(topo_RCP & cellTopo) {
  int dimension = cellTopo->getDimension();
  int numnodes = cellTopo->getNodeCount();
  DRV refnodes("nodes on reference element",numnodes,dimension);
  CellTools::getReferenceSubcellVertices(refnodes, dimension, 0, *cellTopo);
  return refnodes;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::mapPointsToPhysical(DRV ref_pts, Kokkos::View<LO*,AssemblyDevice> elemIDs, 
                                                 const size_t & block, topo_RCP & cellTopo) {
  DRV nodes = this->getMyNodes(block, elemIDs);
  DRV phys_pts = this->mapPointsToPhysical(ref_pts, nodes, cellTopo);
  return phys_pts;
}

// ========================================================================================
// ========================================================================================

DRV DiscretizationInterface::mapPointsToPhysical(DRV ref_pts, DRV nodes, topo_RCP & cellTopo) {
  DRV phys_pts("reference cell points",nodes.extent(0), ref_pts.extent(0), ref_pts.extent(1));
  CellTools::mapToPhysicalFrame(phys_pts, ref_pts, nodes, *cellTopo);
  return phys_pts;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::DynRankView<int,PHX::Device> DiscretizationInterface::checkInclusionPhysicalData(DRV phys_pts, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                                                         topo_RCP & cellTopo, const size_t & block,
                                                                                         const ScalarT & tol) {
  DRV nodes = this->getMyNodes(block, elemIDs);
  Kokkos::DynRankView<int,PHX::Device> check = this->checkInclusionPhysicalData(phys_pts, nodes, cellTopo, tol);
  return check;
}

// ========================================================================================
// ========================================================================================

Kokkos::DynRankView<int,PHX::Device> DiscretizationInterface::checkInclusionPhysicalData(DRV phys_pts, DRV nodes,
                                                                                         topo_RCP & cellTopo, 
                                                                                         const ScalarT & tol) {
  DRV ref_pts = this->mapPointsToReference(phys_pts, nodes, cellTopo);
  //DRV phys_pts2 = this->mapPointsToPhysical(ref_pts,nodes,cellTopo);
  DRV phys_pts2("physical cell point remapped",phys_pts.extent(0), phys_pts.extent(1), phys_pts.extent(2));
  CellTools::mapToPhysicalFrame(phys_pts2, ref_pts, nodes, *cellTopo);
  
  ScalarT reldiff = this->computeRelativeDifference(phys_pts, phys_pts2);
  //cout << "reldiff = " << reldiff << endl;

  if (reldiff > 1.0e-12) {
   // cout << "Processor " << comm->getRank() << " has a degenerate mapping" << endl;
  //  KokkosTools::print(phys_pts);
  //  KokkosTools::print(ref_pts);
  //  KokkosTools::print(phys_pts2);
  }

  Kokkos::DynRankView<int,PHX::Device> inRefCell("inRefCell", 1, phys_pts.extent(1));
  
  CellTools::checkPointwiseInclusion(inRefCell, ref_pts, *cellTopo, tol);
  
  if (!inRefCell(0,0)) {
    //KokkosTools::print(ref_pts);
  }
  return inRefCell;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

ScalarT DiscretizationInterface::computeRelativeDifference(DRV data1, DRV data2) {

  auto data1_host = create_mirror_view(data1);
  deep_copy(data1_host,data1);

  auto data2_host = create_mirror_view(data2);
  deep_copy(data2_host,data2);

  ScalarT diff = 0.0;
  ScalarT base = 0.0;
  // Assumes data1 and data2 are rank-3 ... not necessary, but this is the only use case right now
  for (size_type i=0; i<data1_host.extent(0); ++i) {
    for (size_type j=0; j<data1_host.extent(1); ++j) {
      for (size_type k=0; k<data1_host.extent(2); ++k) {
        diff += std::abs(data1_host(i,j,k) - data2_host(i,j,k));
        base += std::abs(data1_host(i,j,k));
      }
    }
  }
  return diff/base;
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the lin. alg. setup, we can get rid of the dof_lids
/////////////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::purgeLIDs() {
  dof_lids.clear();
}

// ========================================================================================
// After the setup phase, we can get rid of a few things
// ========================================================================================

void DiscretizationInterface::purgeMemory() {
  
  dof_owned.clear();
  dof_owned_and_shared.clear();
  side_info.clear();
  
}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::purgeOrientations() {
  
  panzer_orientations = Kokkos::View<Intrepid2::Orientation*,HostDevice>("panzer orients",1);
  my_elements.clear();

}
