/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::setReferenceIntegrationData(Teuchos::RCP<GroupMetaData> & groupData) {
  
  // ------------------------------------
  // Reference ip/wts/normals/tangents
  // ------------------------------------
  
  size_t dimension = groupData->dimension;
  size_t block = groupData->my_block;
  
  groupData->num_ip = ref_ip[block].extent(0);
  groupData->num_side_ip = ref_side_ip[block].extent(0);
  groupData->ref_ip = ref_ip[block];
  groupData->ref_wts = ref_wts[block];
  
  auto cellTopo = groupData->cell_topo;
  
  if (dimension == 1) {
    DRV leftpt("refSidePoints",1, dimension);
    Kokkos::deep_copy(leftpt,-1.0);
    DRV rightpt("refSidePoints",1, dimension);
    Kokkos::deep_copy(rightpt,1.0);
    groupData->ref_side_ip.push_back(leftpt);
    groupData->ref_side_ip.push_back(rightpt);
    
    DRV leftwt("refSideWts",1, dimension);
    Kokkos::deep_copy(leftwt,1.0);
    DRV rightwt("refSideWts",1, dimension);
    Kokkos::deep_copy(rightwt,1.0);
    groupData->ref_side_wts.push_back(leftwt);
    groupData->ref_side_wts.push_back(rightwt);
    
    DRV leftn("refSideNormals",1, dimension);
    Kokkos::deep_copy(leftn,-1.0);
    DRV rightn("refSideNormals",1, dimension);
    Kokkos::deep_copy(rightn,1.0);
    groupData->ref_side_normals.push_back(leftn);
    groupData->ref_side_normals.push_back(rightn);
  }
  else {
    for (size_t s=0; s<groupData->num_sides; s++) {
      DRV refSidePoints("refSidePoints",groupData->num_side_ip, dimension);
      CellTools::mapToReferenceSubcell(refSidePoints, ref_side_ip[block],
                                       dimension-1, s, *cellTopo);
      groupData->ref_side_ip.push_back(refSidePoints);
      groupData->ref_side_wts.push_back(ref_side_wts[block]);
      
      DRV refSideNormals("refSideNormals", dimension);
      DRV refSideTangents("refSideTangents", dimension);
      DRV refSideTangentsU("refSideTangents U", dimension);
      DRV refSideTangentsV("refSideTangents V", dimension);
      
      if (dimension == 2) {
        CellTools::getReferenceSideNormal(refSideNormals,s,*cellTopo);
        CellTools::getReferenceEdgeTangent(refSideTangents,s,*cellTopo);
      }
      else if (dimension == 3) {
        CellTools::getReferenceFaceTangents(refSideTangentsU, refSideTangentsV, s, *cellTopo);
      }
      
      groupData->ref_side_normals.push_back(refSideNormals);
      groupData->ref_side_tangents.push_back(refSideTangents);
      groupData->ref_side_tangentsU.push_back(refSideTangentsU);
      groupData->ref_side_tangentsV.push_back(refSideTangentsV);
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::getQuadrature(const topo_RCP & cellTopo, const int & order,
                                            DRV & ip, DRV & wts) {
  
  Intrepid2::DefaultCubatureFactory cubFactory;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double> > basisCub  = cubFactory.create<PHX::Device::execution_space, double, double>(*cellTopo, order);
  int cubDim  = basisCub->getDimension();
  int numCubPoints = basisCub->getNumPoints();
  ip = DRV("ip", numCubPoints, cubDim);
  wts = DRV("wts", numCubPoints);
  basisCub->getCubature(ip, wts);
  
}

// -------------------------------------------------
// Compute the volumetric integration information
// -------------------------------------------------

void DiscretizationInterface::getPhysicalIntegrationPts(Teuchos::RCP<GroupMetaData> & groupData,
                                                         Kokkos::View<LO*,AssemblyDevice> elemIDs, vector<View_Sc2> & ip) {
  DRV nodes = this->getMyNodes(groupData->my_block, elemIDs);
  this->getPhysicalIntegrationPts(groupData, nodes, ip);
}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getPhysicalIntegrationPts(Teuchos::RCP<GroupMetaData> & groupData,
                                                         DRV nodes, vector<View_Sc2> & ip) {
  
  Teuchos::TimeMonitor constructor_timer(*phys_vol_IP_timer);

  int dimension = groupData->dimension;
  int numip = groupData->ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  DRV tmpip("tmp ip", numElem, numip, dimension);
  
  {
    CellTools::mapToPhysicalFrame(tmpip, groupData->ref_ip, nodes, *(groupData->cell_topo));
    View_Sc2 x("x",tmpip.extent(0), tmpip.extent(1));
    auto tmpip_x = subview(tmpip, ALL(), ALL(),0);
    deep_copy(x,tmpip_x);
    ip.push_back(x);
    if (dimension > 1) {
      View_Sc2 y("y",tmpip.extent(0), tmpip.extent(1));
      auto tmpip_y = subview(tmpip, ALL(), ALL(),1);
      deep_copy(y,tmpip_y);
      ip.push_back(y);
    }
    if (dimension > 2) {
      View_Sc2 z("z",tmpip.extent(0), tmpip.extent(1));
      auto tmpip_z = subview(tmpip, ALL(), ALL(),2);
      deep_copy(z,tmpip_z);
      ip.push_back(z);
    }
    
  }
}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getPhysicalIntegrationData(Teuchos::RCP<GroupMetaData> & groupData,
                                                         Kokkos::View<LO*,AssemblyDevice> elemIDs, vector<View_Sc2> & ip, View_Sc2 wts) {
  DRV nodes = this->getMyNodes(groupData->my_block, elemIDs);
  this->getPhysicalIntegrationData(groupData, nodes, ip, wts);
}

/// @brief ////////////////////////////////////////////////////
/// @param groupData
/// @param nodes
/// @param ip
/// @param wts

void DiscretizationInterface::getPhysicalIntegrationData(Teuchos::RCP<GroupMetaData> & groupData,
                                                         DRV nodes, vector<View_Sc2> & ip, View_Sc2 wts) {
  
  Teuchos::TimeMonitor constructor_timer(*phys_vol_IP_timer);

  int dimension = groupData->dimension;
  int numip = groupData->ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  DRV jacobian ("jacobian", numElem, numip, dimension, dimension);
  DRV jacobianDet("determinant of jacobian", numElem, numip);
  DRV tmpip("tmp ip", numElem, numip, dimension);
  DRV tmpwts("tmp ip wts", numElem, numip);
  
  {
    CellTools::mapToPhysicalFrame(tmpip, groupData->ref_ip, nodes, *(groupData->cell_topo));
    View_Sc2 x("x",tmpip.extent(0), tmpip.extent(1));
    auto tmpip_x = subview(tmpip, ALL(), ALL(),0);
    deep_copy(x,tmpip_x);
    ip.push_back(x);
    if (dimension > 1) {
      View_Sc2 y("y",tmpip.extent(0), tmpip.extent(1));
      auto tmpip_y = subview(tmpip, ALL(), ALL(),1);
      deep_copy(y,tmpip_y);
      ip.push_back(y);
    }
    if (dimension > 2) {
      View_Sc2 z("z",tmpip.extent(0), tmpip.extent(1));
      auto tmpip_z = subview(tmpip, ALL(), ALL(),2);
      deep_copy(z,tmpip_z);
      ip.push_back(z);
    }
    
  }
  
  CellTools::setJacobian(jacobian, groupData->ref_ip, nodes, *(groupData->cell_topo));
  CellTools::setJacobianDet(jacobianDet, jacobian);
  FuncTools::computeCellMeasure(tmpwts, jacobianDet, groupData->ref_wts);
  Kokkos::deep_copy(wts,tmpwts);
  
}
                    
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::getJacobian(Teuchos::RCP<GroupMetaData> & groupData,
                                          Kokkos::View<LO*,AssemblyDevice> elemIDs, DRV jacobian) {
  DRV nodes = this->getMyNodes(groupData->my_block, elemIDs);
  this->getJacobian(groupData, nodes, jacobian);
}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getJacobian(Teuchos::RCP<GroupMetaData> & groupData,
                                          DRV nodes, DRV jacobian) {
  CellTools::setJacobian(jacobian, groupData->ref_ip, nodes, *(groupData->cell_topo));
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::getPhysicalWts(Teuchos::RCP<GroupMetaData> & groupData,
                                             Kokkos::View<LO*,AssemblyDevice> elemIDs, DRV jacobian, DRV wts) {

  int numip = groupData->ref_ip.extent(0);
  int numElem = jacobian.extent(0);
  
  DRV jacobianDet("determinant of jacobian", numElem, numip);
  CellTools::setJacobianDet(jacobianDet, jacobian);
  FuncTools::computeCellMeasure(wts, jacobianDet, groupData->ref_wts);
            
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::getMeasure(Teuchos::RCP<GroupMetaData> & groupData,
                                         DRV jacobian, DRV measure) {
  int numip = groupData->ref_ip.extent(0);
  int numElem = measure.extent(0);
  
  DRV jacobianDet("determinant of jacobian", numElem, numip);
  CellTools::setJacobianDet(jacobianDet, jacobian);
  DRV wts("jacobian", numElem, numip);
  FuncTools::computeCellMeasure(wts, jacobianDet, groupData->ref_wts);

  parallel_for("compute measure",
               RangePolicy<AssemblyExec>(0,numElem),
               KOKKOS_CLASS_LAMBDA (const size_type elem ) {
    for (size_type pt=0; pt<wts.extent(1); ++pt) {
      measure(elem) += wts(elem,pt);
    }
  });
        
}

// -------------------------------------------------
// Compute the basis functions at the face ip
// -------------------------------------------------

void DiscretizationInterface::getPhysicalFaceIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, const int & side,
                                                             Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                             vector<View_Sc2> & face_ip, View_Sc2 face_wts,
                                                             vector<View_Sc2> & face_normals) {

  DRV nodes = this->getMyNodes(groupData->my_block, elemIDs);
  this->getPhysicalFaceIntegrationData(groupData, side, nodes, face_ip, face_wts, face_normals);
}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getPhysicalFaceIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, const int & side,
                                                             DRV nodes,
                                                             vector<View_Sc2> & face_ip, View_Sc2 face_wts,
                                                             vector<View_Sc2> & face_normals) {
  
  Teuchos::TimeMonitor localtimer(*phys_face_data_total_timer);
  
  auto ref_ip = groupData->ref_side_ip[side];
  auto ref_wts = groupData->ref_side_wts[side];
  
  int dimension = groupData->dimension;
  int numip = ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  // Step 1: fill in ip_side, wts_side and normals
  DRV sip("side ip", numElem, numip, dimension);
  DRV jacobian ("side jac", numElem, numip, dimension, dimension);
  DRV swts("wts_side", numElem, numip);
  DRV snormals("normals", numElem, numip, dimension);
  DRV tangents("tangents", numElem, numip, dimension);
  
  {
    Teuchos::TimeMonitor localtimer(*phys_face_data_IP_timer);
    CellTools::mapToPhysicalFrame(sip, ref_ip, nodes, *(groupData->cell_topo));
    
    View_Sc2 x("cell face x",sip.extent(0), sip.extent(1));
    auto sip_x = subview(sip, ALL(), ALL(),0);
    deep_copy(x,sip_x);
    face_ip.push_back(x);
    
    if (dimension > 1) {
      View_Sc2 y("cell face y",sip.extent(0), sip.extent(1));
      auto sip_y = subview(sip, ALL(), ALL(),1);
      deep_copy(y,sip_y);
      face_ip.push_back(y);
    }
    if (dimension > 2) {
      View_Sc2 z("cell face z",sip.extent(0), sip.extent(1));
      auto sip_z = subview(sip, ALL(), ALL(),2);
      deep_copy(z,sip_z);
      face_ip.push_back(z);
    }
    
  }
  
  {
    Teuchos::TimeMonitor localtimer(*phys_face_data_set_jac_timer);
    CellTools::setJacobian(jacobian, ref_ip, nodes, *(groupData->cell_topo));
  }
  
  {
    Teuchos::TimeMonitor localtimer(*phys_face_data_wts_timer);
    
    if (dimension == 2) {
      auto ref_tangents = groupData->ref_side_tangents[side];
      RealTools::matvec(tangents, jacobian, ref_tangents);
      
      DRV rotation("rotation matrix",dimension,dimension);
      rotation(0,0) = 0;  rotation(0,1) = 1;
      rotation(1,0) = -1; rotation(1,1) = 0;
      RealTools::matvec(snormals, rotation, tangents);
      
      RealTools::vectorNorm(swts, tangents, Intrepid2::NORM_TWO);
      ArrayTools::scalarMultiplyDataData(swts, swts, ref_wts);
      
    }
    else if (dimension == 3) {
      
      auto ref_tangentsU = groupData->ref_side_tangentsU[side];
      auto ref_tangentsV = groupData->ref_side_tangentsV[side];
      
      DRV faceTanU("face tangent U", numElem, numip, dimension);
      DRV faceTanV("face tangent V", numElem, numip, dimension);
      
      RealTools::matvec(faceTanU, jacobian, ref_tangentsU);
      RealTools::matvec(faceTanV, jacobian, ref_tangentsV);
      
      RealTools::vecprod(snormals, faceTanU, faceTanV);
      
      RealTools::vectorNorm(swts, snormals, Intrepid2::NORM_TWO);
      ArrayTools::scalarMultiplyDataData(swts, swts, ref_wts);
      
    }
    
    // scale the normal vector (we need unit normal...)
    
    parallel_for("wkset transient sol seedwhat 1",
                 TeamPolicy<AssemblyExec>(snormals.extent(0), Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type pt=team.team_rank(); pt<snormals.extent(1); pt+=team.team_size() ) {
        ScalarT normalLength = 0.0;
        for (size_type sd=0; sd<snormals.extent(2); sd++) {
          normalLength += snormals(elem,pt,sd)*snormals(elem,pt,sd);
        }
        normalLength = sqrt(normalLength);
        for (size_type sd=0; sd<snormals.extent(2); sd++) {
          snormals(elem,pt,sd) = snormals(elem,pt,sd) / normalLength;
        }
      }
    });
        
    View_Sc2 nx("cell face nx",snormals.extent(0), snormals.extent(1));
    auto s_nx = subview(snormals, ALL(), ALL(),0);
    deep_copy(nx,s_nx);
    face_normals.push_back(nx);
    
    if (dimension > 1) {
      View_Sc2 ny("cell face ny", snormals.extent(0), snormals.extent(1));
      auto s_ny = subview(snormals, ALL(), ALL(),1);
      deep_copy(ny,s_ny);
      face_normals.push_back(ny);
    }
    if (dimension > 2) {
      View_Sc2 nz("cell face nz",snormals.extent(0), snormals.extent(1));
      auto s_nz = subview(snormals, ALL(), ALL(), 2);
      deep_copy(nz,s_nz);
      face_normals.push_back(nz);
    }
    
    
    Kokkos::deep_copy(face_wts,swts);
  }
}

//======================================================================
//
//======================================================================

void DiscretizationInterface::getPhysicalBoundaryIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                                 LO & localSideID,
                                                                 vector<View_Sc2> & ip, View_Sc2 wts,
                                                                 vector<View_Sc2> & normals, vector<View_Sc2> & tangents) {
  DRV nodes = this->getMyNodes(groupData->my_block, elemIDs);
  this->getPhysicalBoundaryIntegrationData(groupData, nodes, localSideID, ip, wts, normals, tangents);

}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getPhysicalBoundaryIntegrationData(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                                                 LO & localSideID,
                                                                 vector<View_Sc2> & ip, View_Sc2 wts,
                                                                 vector<View_Sc2> & normals, vector<View_Sc2> & tangents) {
  
  Teuchos::TimeMonitor localtimer(*phys_bndry_data_total_timer);
  
  int dimension = groupData->dimension;
  
  DRV ref_ip = groupData->ref_side_ip[localSideID];
  DRV ref_wts = groupData->ref_side_wts[localSideID];
  
  int numip = ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  DRV tmpip("side ip", numElem, numip, dimension);
  DRV jacobian("bijac", numElem, numip, dimension, dimension);
  //DRV jacobianDet("bijacDet", numElem, numip);
  //DRV jacobianInv("bijacInv", numElem, numip, dimension, dimension);
  DRV tmpwts("wts_side", numElem, numip);
  DRV tmpnormals("normals", numElem, numip, dimension);
  DRV tmptangents("tangents", numElem, numip, dimension);
  
  {
    Teuchos::TimeMonitor localtimer(*phys_bndry_data_IP_timer);
    CellTools::mapToPhysicalFrame(tmpip, ref_ip, nodes, *(groupData->cell_topo));
    View_Sc2 x("cell face x",tmpip.extent(0), tmpip.extent(1));
    auto tip_x = subview(tmpip, ALL(), ALL(),0);
    deep_copy(x,tip_x);
    ip.push_back(x);
    
    if (dimension > 1) {
      View_Sc2 y("cell face y",tmpip.extent(0), tmpip.extent(1));
      auto tip_y = subview(tmpip, ALL(), ALL(),1);
      deep_copy(y,tip_y);
      ip.push_back(y);
    }
    if (dimension > 2) {
      View_Sc2 z("cell face z",tmpip.extent(0), tmpip.extent(1));
      auto tip_z = subview(tmpip, ALL(), ALL(),2);
      deep_copy(z,tip_z);
      ip.push_back(z);
    }
  }
  
  {
    Teuchos::TimeMonitor localtimer(*phys_bndry_data_set_jac_timer);
    CellTools::setJacobian(jacobian, ref_ip, nodes, *(groupData->cell_topo));
  }
  
  //{
  //  Teuchos::TimeMonitor localtimer(*physBndryDataOtherJacTimer);
  //  CellTools::setJacobianInv(jacobianInv, jacobian);
  //  CellTools::setJacobianDet(jacobianDet, jacobian);
  //}
  
  {
    Teuchos::TimeMonitor localtimer(*phys_bndry_data_wts_timer);
    if (dimension == 1) {
      Kokkos::deep_copy(tmpwts,1.0);
      auto ref_normals = groupData->ref_side_normals[localSideID];
      parallel_for("bcell 1D normal copy",
                   RangePolicy<AssemblyExec>(0,tmpnormals.extent(0)),
                   KOKKOS_CLASS_LAMBDA (const int elem ) {
        tmpnormals(elem,0,0) = ref_normals(0,0);
      });
      
    }
    else if (dimension == 2) {
      DRV ref_tangents = groupData->ref_side_tangents[localSideID];
      RealTools::matvec(tmptangents, jacobian, ref_tangents);
      
      DRV rotation("rotation matrix",dimension,dimension);
      auto rotation_host = Kokkos::create_mirror_view(rotation);
      rotation_host(0,0) = 0;  rotation_host(0,1) = 1;
      rotation_host(1,0) = -1; rotation_host(1,1) = 0;
      Kokkos::deep_copy(rotation, rotation_host);
      RealTools::matvec(tmpnormals, rotation, tmptangents);
      
      RealTools::vectorNorm(tmpwts, tmptangents, Intrepid2::NORM_TWO);
      ArrayTools::scalarMultiplyDataData(tmpwts, tmpwts, ref_wts);
      
    }
    else if (dimension == 3) {
      
      DRV ref_tangentsU = groupData->ref_side_tangentsU[localSideID];
      DRV ref_tangentsV = groupData->ref_side_tangentsV[localSideID];
      
      DRV faceTanU("face tangent U", numElem, numip, dimension);
      DRV faceTanV("face tangent V", numElem, numip, dimension);
      
      RealTools::matvec(faceTanU, jacobian, ref_tangentsU);
      RealTools::matvec(faceTanV, jacobian, ref_tangentsV);
      
      RealTools::vecprod(tmpnormals, faceTanU, faceTanV);
      
      RealTools::vectorNorm(tmpwts, tmpnormals, Intrepid2::NORM_TWO);
      ArrayTools::scalarMultiplyDataData(tmpwts, tmpwts, ref_wts);
      
    }
    Kokkos::deep_copy(wts,tmpwts);
    
    View_Sc2 nx("cell face nx",tmpnormals.extent(0), tmpnormals.extent(1));
    auto t_nx = subview(tmpnormals, ALL(), ALL(),0);
    deep_copy(nx,t_nx);
    normals.push_back(nx);
    
    if (dimension > 1) {
      View_Sc2 ny("cell face ny",tmpnormals.extent(0), tmpnormals.extent(1));
      auto t_ny = subview(tmpnormals, ALL(), ALL(),1);
      deep_copy(ny,t_ny);
      normals.push_back(ny);
    }
    if (dimension > 2) {
      View_Sc2 nz("cell face z",tmpnormals.extent(0), tmpnormals.extent(1));
      auto t_nz = subview(tmpnormals, ALL(), ALL(),2);
      deep_copy(nz,t_nz);
      normals.push_back(nz);
    }
    
    View_Sc2 tx("cell face tx",tmptangents.extent(0), tmptangents.extent(1));
    auto t_tx = subview(tmptangents, ALL(), ALL(),0);
    deep_copy(tx,t_tx);
    tangents.push_back(tx);
    
    if (dimension > 1) {
      View_Sc2 ty("cell face ty",tmptangents.extent(0), tmptangents.extent(1));
      auto t_ty = subview(tmptangents, ALL(), ALL(),1);
      deep_copy(ty,t_ty);
      tangents.push_back(ty);
    }
    if (dimension > 2) {
      View_Sc2 tz("cell face tz",tmptangents.extent(0), tmptangents.extent(1));
      auto t_tz = subview(tmptangents, ALL(), ALL(),2);
      deep_copy(tz,t_tz);
      tangents.push_back(tz);
    }
  }
  
  // -------------------------------------------------
  // Rescale the normals
  // -------------------------------------------------
  
  {
    View_Sc2 nx,ny,nz;
    nx = normals[0];
    if (dimension>1) {
      ny = normals[1];
    }
    if (dimension>2) {
      nz = normals[2];
    }
    
    parallel_for("bcell normal rescale",
                 TeamPolicy<AssemblyExec>(nx.extent(0), Kokkos::AUTO),
                 KOKKOS_CLASS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      int dim = tmpip.extent(2);
      for (size_type pt=team.team_rank(); pt<nx.extent(1); pt+=team.team_size() ) {
        ScalarT normalLength = nx(elem,pt)*nx(elem,pt);
        if (dim>1) {
          normalLength += ny(elem,pt)*ny(elem,pt);
        }
        if (dim>2) {
          normalLength += nz(elem,pt)*nz(elem,pt);
        }
        normalLength = sqrt(normalLength);
        nx(elem,pt) *= 1.0/normalLength;
        if (dim>1) {
          ny(elem,pt) *= 1.0/normalLength;
        }
        if (dim>2) {
          nz(elem,pt) *= 1.0/normalLength;
        }
      }
    });
  }
}

