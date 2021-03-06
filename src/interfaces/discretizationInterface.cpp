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

#include "discretizationInterface.hpp"

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
#include "Intrepid2_HFACE_TET_In_FEM.hpp"

#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Utils.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"


#include "Panzer_STKConnManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SetupUtilities.hpp"

typedef Intrepid2::CellTools<PHX::Device::execution_space> CellTools;
typedef Intrepid2::FunctionSpaceTools<PHX::Device::execution_space> FuncTools;
typedef Intrepid2::OrientationTools<PHX::Device::execution_space> OrientTools;
typedef Intrepid2::RealSpaceTools<PHX::Device::execution_space> RealTools;
typedef Intrepid2::ArrayTools<PHX::Device::execution_space> ArrayTools;

using namespace MrHyDE;

discretization::discretization(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                               Teuchos::RCP<MpiComm> & Comm_,
                               Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                               Teuchos::RCP<physics> & phys_) :
settings(settings_), Commptr(Comm_), mesh(mesh_), phys(phys_) {
  
  debug_level = settings->get<int>("debug level",0);
  verbosity = settings->get<int>("verbosity",0);
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting discretization constructor..." << endl;
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Collect some information
  ////////////////////////////////////////////////////////////////////////////////
  
  spaceDim = mesh->getDimension();
  mesh->getElementBlockNames(blocknames);
  
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    vector<stk::mesh::Entity> stkElems;
    mesh->getMyElements(blocknames[block], stkElems);
    block_stkElems.push_back(stkElems);
  }
  mesh->getMyElements(all_stkElems);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Assemble the information we always store
  ////////////////////////////////////////////////////////////////////////////////
  
  for (size_t b=0; b<blocknames.size(); b++) {
    
    string blockID = blocknames[b];
    Teuchos::ParameterList db_settings;
    if (settings->sublist("Discretization").isSublist(blocknames[b])) {
      db_settings = settings->sublist("Discretization").sublist(blocknames[b]);
    }
    else {
      db_settings = settings->sublist("Discretization");
    }
    
    vector<stk::mesh::Entity> stk_meshElems;
    mesh->getMyElements(blockID, stk_meshElems);
    
    // list of all elements on this processor
    vector<size_t> blockmyElements = vector<size_t>(stk_meshElems.size());
    for( size_t e=0; e<stk_meshElems.size(); e++ ) {
      blockmyElements[e] = mesh->elementLocalId(stk_meshElems[e]);
    }
    myElements.push_back(blockmyElements);
    
    ///////////////////////////////////////////////////////////////////////////
    // Get the cardinality of the basis functions  on this block
    ///////////////////////////////////////////////////////////////////////////
    
    topo_RCP cellTopo = mesh->getCellTopology(blockID);
    string shape = cellTopo->getName();
    
    vector<int> blockcards;
    vector<basis_RCP> blockbasis;//(blockmaxorder);
    
    vector<vector<int> > orders = phys->unique_orders;
    vector<vector<string> > types = phys->unique_types;
    vector<int> doneorders;
    vector<string> donetypes;
    for (size_t n=0; n<orders[b].size(); n++) {
      bool go = true;
      for (size_t i=0; i<doneorders.size(); i++){
        if (doneorders[i] == orders[b][n] && donetypes[i] == types[b][n]) {
          go = false;
        }
      }
      if (go) {
        basis_RCP basis = this->getBasis(spaceDim, cellTopo, types[b][n], orders[b][n]);
        int bsize = basis->getCardinality();
        blockcards.push_back(bsize); // cardinality of the basis
        blockbasis.push_back(basis);
        doneorders.push_back(orders[b][n]);
        donetypes.push_back(types[b][n]);
      }
    }
    basis_types.push_back(donetypes);
    cards.push_back(blockcards);
    
    ///////////////////////////////////////////////////////////////////////////
    // Quadrature
    ///////////////////////////////////////////////////////////////////////////
    
    int mxorder = 0;
    for (size_t i=0; i<orders[b].size(); i++) {
      if (orders[b][i]>mxorder) {
        mxorder = orders[b][i];
      }
    }
    
    DRV qpts, qwts;
    int quadorder = db_settings.get<int>("quadrature",2*mxorder);
    this->getQuadrature(cellTopo, quadorder, qpts, qwts);
    
    ///////////////////////////////////////////////////////////////////////////
    // Side Quadrature
    ///////////////////////////////////////////////////////////////////////////
    
    topo_RCP sideTopo;
    
    if (spaceDim == 1) {
      sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Node >() ));
    }
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
      if (shape == "Triangle_3") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ));
      }
      if (shape == "Tetrahedron_4") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() ));
      }
    }
    
    DRV side_qpts, side_qwts;
    if (spaceDim == 1) {
      side_qpts = DRV("side qpts",1,1);
      Kokkos::deep_copy(side_qpts,-1.0);
      side_qwts = DRV("side wts",1,1);
      Kokkos::deep_copy(side_qwts,1.0);
    }
    else {
      int side_quadorder = db_settings.sublist(blockID).get<int>("side quadrature",2*mxorder);
      this->getQuadrature(sideTopo, side_quadorder, side_qpts, side_qwts);
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // Store locally
    ///////////////////////////////////////////////////////////////////////////
    
    basis_pointers.push_back(blockbasis);
    ref_ip.push_back(qpts);
    ref_wts.push_back(qwts);
    ref_side_ip.push_back(side_qpts);
    ref_side_wts.push_back(side_qwts);
    
    numip.push_back(qpts.extent(0));
    numip_side.push_back(side_qpts.extent(0));
    
  } // block loop
  
  this->buildDOFManagers();
  
  this->setBCData(false);
  
  this->setDirichletData(false);
  
  if (phys->have_aux) {
    this->setBCData(true);
    this->setDirichletData(true);
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished discretization constructor" << endl;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Create a pointer to an Intrepid or Panzer basis
//////////////////////////////////////////////////////////////////////////////////////

basis_RCP discretization::getBasis(const int & spaceDim, const topo_RCP & cellTopo,
                                   const string & type, const int & degree) {
  using namespace Intrepid2;
  
  basis_RCP basis;
  
  string shape = cellTopo->getName();
  
  if (type == "HGRAD") {
    if (spaceDim == 1) {
      basis = Teuchos::rcp(new Basis_HGRAD_LINE_C1_FEM<PHX::Device::execution_space>() );
    }
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        if (degree == 1) {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C1_FEM<PHX::Device::execution_space>() );
        }
        else if (degree == 2) {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C2_FEM<PHX::Device::execution_space>() );
        }
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_Cn_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      if (shape == "Triangle_3") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C1_FEM<PHX::Device::execution_space>() );
        else if (degree == 2)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C2_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_Cn_FEM<PHX::Device::execution_space>(degree,POINTTYPE_WARPBLEND) );
        }
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        if (degree  == 1)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C1_FEM<PHX::Device::execution_space>() );
        else if (degree  == 2)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C2_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_Cn_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      if (shape == "Tetrahedron_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HGRAD_TET_C1_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_TET_Cn_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
  }
  else if (type == "HVOL") {
    basis = Teuchos::rcp(new Basis_HVOL_C0_FEM<PHX::Device::execution_space>(*cellTopo));
  }
  else if (type == "HDIV") {
    if (spaceDim == 1) {
      basis = Teuchos::rcp(new Basis_HGRAD_LINE_C1_FEM<PHX::Device::execution_space>() );
    }
    else if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HDIV_QUAD_I1_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_QUAD_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Triangle_3") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HDIV_TRI_I1_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_TRI_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    else if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        if (degree  == 1)
          basis = Teuchos::rcp(new Basis_HDIV_HEX_I1_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_HEX_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Tetrahedron_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HDIV_TET_I1_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_TET_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
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
          basis = Teuchos::rcp(new Basis_HCURL_QUAD_I1_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_QUAD_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Triangle_3") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HCURL_TRI_I1_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_TRI_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    else if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        if (degree  == 1)
          basis = Teuchos::rcp(new Basis_HCURL_HEX_I1_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_HEX_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Tetrahedron_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HCURL_TET_I1_FEM<PHX::Device::execution_space>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_TET_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    
  }
  else if (type == "HFACE") {
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        basis = Teuchos::rcp(new Basis_HFACE_QUAD_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
      }
      else if (shape == "Triangle_3") {
        basis = Teuchos::rcp(new Basis_HFACE_TRI_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        basis = Teuchos::rcp(new Basis_HFACE_HEX_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
      }
      if (shape == "Tetrahedron_4") {
        basis = Teuchos::rcp(new Basis_HFACE_TET_In_FEM<PHX::Device::execution_space>(degree,POINTTYPE_EQUISPACED) );
      }
    }
  }
  
  
  return basis;
  
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void discretization::getQuadrature(const topo_RCP & cellTopo, const int & order,
                                   DRV & ip, DRV & wts) {
  
  Intrepid2::DefaultCubatureFactory cubFactory;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space> > basisCub  = cubFactory.create<PHX::Device::execution_space, ScalarT, ScalarT>(*cellTopo, order); // TMW: the mesh sublist is not the correct place
  int cubDim  = basisCub->getDimension();
  int numCubPoints = basisCub->getNumPoints();
  ip = DRV("ip", numCubPoints, cubDim);
  wts = DRV("wts", numCubPoints);
  basisCub->getCubature(ip, wts);
  
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void discretization::setReferenceData(Teuchos::RCP<CellMetaData> & cellData) {
  
  // ------------------------------------
  // Reference ip/wts/normals/tangents
  // ------------------------------------
  
  size_t dimension = cellData->dimension;
  size_t block = cellData->myBlock;
  
  cellData->numip = ref_ip[block].extent(0);
  cellData->numsideip = ref_side_ip[block].extent(0);
  cellData->ref_ip = ref_ip[block];
  cellData->ref_wts = ref_wts[block];
  
  auto cellTopo = cellData->cellTopo;
  
  if (dimension == 1) {
    DRV leftpt("refSidePoints",1, dimension);
    Kokkos::deep_copy(leftpt,-1.0);
    DRV rightpt("refSidePoints",1, dimension);
    Kokkos::deep_copy(rightpt,1.0);
    cellData->ref_side_ip.push_back(leftpt);
    cellData->ref_side_ip.push_back(rightpt);
    
    DRV leftwt("refSideWts",1, dimension);
    Kokkos::deep_copy(leftwt,1.0);
    DRV rightwt("refSideWts",1, dimension);
    Kokkos::deep_copy(rightwt,1.0);
    cellData->ref_side_wts.push_back(leftwt);
    cellData->ref_side_wts.push_back(rightwt);
    
    DRV leftn("refSideNormals",1, dimension);
    Kokkos::deep_copy(leftn,-1.0);
    DRV rightn("refSideNormals",1, dimension);
    Kokkos::deep_copy(rightn,1.0);
    cellData->ref_side_normals.push_back(leftn);
    cellData->ref_side_normals.push_back(rightn);
  }
  else {
    for (size_t s=0; s<cellData->numSides; s++) {
      DRV refSidePoints("refSidePoints",cellData->numsideip, dimension);
      CellTools::mapToReferenceSubcell(refSidePoints, ref_side_ip[block],
                                       dimension-1, s, *cellTopo);
      cellData->ref_side_ip.push_back(refSidePoints);
      cellData->ref_side_wts.push_back(ref_side_wts[block]);
      
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
      
      cellData->ref_side_normals.push_back(refSideNormals);
      cellData->ref_side_tangents.push_back(refSideTangents);
      cellData->ref_side_tangentsU.push_back(refSideTangentsU);
      cellData->ref_side_tangentsV.push_back(refSideTangentsV);
    }
  }
  
  // ------------------------------------
  // Get refnodes
  // ------------------------------------
  
  DRV refnodes("nodes on reference element",cellTopo->getNodeCount(),dimension);
  CellTools::getReferenceSubcellVertices(refnodes, dimension, 0, *cellTopo);
  cellData->refnodes = refnodes;
  
  // ------------------------------------
  // Get ref basis
  // ------------------------------------
  
  cellData->basis_pointers = basis_pointers[block];
  cellData->basis_types = basis_types[block];
  
  for (size_t i=0; i<basis_pointers[block].size(); i++) {
    
    int numb = basis_pointers[block][i]->getCardinality();
    
    DRV basisvals, basisgrad, basisdiv, basiscurl;
    DRV basisnodes;
    
    if (basis_types[block][i] == "HGRAD" || basis_types[block][i] == "HVOL") {
      
      basisvals = DRV("basisvals",numb, cellData->numip);
      basis_pointers[block][i]->getValues(basisvals, cellData->ref_ip, Intrepid2::OPERATOR_VALUE);
    
      basisnodes = DRV("basisvals",numb, refnodes.extent(0));
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basisgrad = DRV("basisgrad",numb, cellData->numip, dimension);
      basis_pointers[block][i]->getValues(basisgrad, cellData->ref_ip, Intrepid2::OPERATOR_GRAD);
      
    }
    else if (basis_types[block][i] == "HDIV"){
      
      basisvals = DRV("basisvals",numb, cellData->numip, dimension);
      basis_pointers[block][i]->getValues(basisvals, cellData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0), dimension);
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basisdiv = DRV("basisdiv",numb, cellData->numip);
      basis_pointers[block][i]->getValues(basisdiv, cellData->ref_ip, Intrepid2::OPERATOR_DIV);
      
    }
    else if (basis_types[block][i] == "HCURL"){
      
      basisvals = DRV("basisvals",numb, cellData->numip, dimension);
      basis_pointers[block][i]->getValues(basisvals, cellData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0), dimension);
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basiscurl = DRV("basiscurl",numb, cellData->numip, dimension);
      basis_pointers[block][i]->getValues(basiscurl, cellData->ref_ip, Intrepid2::OPERATOR_CURL);
      
    }
    else if (basis_types[block][i] == "HFACE"){
      
    }
    cellData->ref_basis.push_back(basisvals);
    cellData->ref_basis_curl.push_back(basiscurl);
    cellData->ref_basis_grad.push_back(basisgrad);
    cellData->ref_basis_div.push_back(basisdiv);
    cellData->ref_basis_nodes.push_back(basisnodes);
  }
  
  // Compute the basis value and basis grad values on reference element
  // at side ip
  for (size_t s=0; s<cellData->numSides; s++) {
    vector<DRV> sbasis, sbasisgrad, sbasisdiv, sbasiscurl;
    for (size_t i=0; i<basis_pointers[block].size(); i++) {
      int numb = basis_pointers[block][i]->getCardinality();
      DRV basisvals, basisgrad, basisdiv, basiscurl;
      if (basis_types[block][i] == "HGRAD" || basis_types[block][i] == "HVOL" || basis_types[block][i] == "HFACE"){
        basisvals = DRV("basisvals",numb, cellData->numsideip);
        basis_pointers[block][i]->getValues(basisvals, cellData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
        basisgrad = DRV("basisgrad",numb, cellData->numsideip, dimension);
        basis_pointers[block][i]->getValues(basisgrad, cellData->ref_side_ip[s], Intrepid2::OPERATOR_GRAD);
      }
      else if (basis_types[block][i] == "HDIV"){
        basisvals = DRV("basisvals",numb, cellData->numsideip, dimension);
        basis_pointers[block][i]->getValues(basisvals, cellData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
      }
      else if (basis_types[block][i] == "HCURL"){
        basisvals = DRV("basisvals",numb, cellData->numsideip, dimension);
        basis_pointers[block][i]->getValues(basisvals, cellData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
      }
      sbasis.push_back(basisvals);
      sbasisgrad.push_back(basisgrad);
      sbasisdiv.push_back(basisdiv);
      sbasiscurl.push_back(basiscurl);
    }
    cellData->ref_side_basis.push_back(sbasis);
    cellData->ref_side_basis_grad.push_back(sbasisgrad);
    cellData->ref_side_basis_div.push_back(sbasisdiv);
    cellData->ref_side_basis_curl.push_back(sbasiscurl);
  }
}

// -------------------------------------------------
// Compute the volumetric integration information
// -------------------------------------------------

void discretization::getPhysicalVolumetricData(Teuchos::RCP<CellMetaData> & cellData,
                                               DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                               View_Sc3 ip, View_Sc2 wts, View_Sc1 hsize,
                                               Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                               vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                               vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                               vector<View_Sc4> & basis_nodes,
                                               const bool & recompute_jac,
                                               const bool & recompute_orient) {

  Teuchos::TimeMonitor localtimer(*physVolDataTotalTimer);
  
  int dimension = cellData->dimension;
  int numip = cellData->ref_ip.extent(0);
  size_t numElem = nodes.extent(0);
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataIPTimer);
    DRV tmpip("tmp ip", numElem, numip, dimension);
    CellTools::mapToPhysicalFrame(tmpip, cellData->ref_ip, nodes, *(cellData->cellTopo));
    Kokkos::deep_copy(ip,tmpip);
  }
  
  DRV jacobian("jacobian", numElem, numip, dimension, dimension);
  {
    Teuchos::TimeMonitor localtimer(*physVolDataSetJacTimer);
    CellTools::setJacobian(jacobian, cellData->ref_ip, nodes, *(cellData->cellTopo));
  }
  
  DRV jacobianDet("determinant of jacobian", numElem, numip);
  DRV jacobianInv("inverse of jacobian", numElem, numip, dimension, dimension);
  {
    Teuchos::TimeMonitor localtimer(*physVolDataOtherJacTimer);
    CellTools::setJacobianDet(jacobianDet, jacobian);
    CellTools::setJacobianInv(jacobianInv, jacobian);
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataWtsTimer);
    DRV tmpwts("tmp ip wts", numElem, numip);
    FuncTools::computeCellMeasure(tmpwts, jacobianDet, cellData->ref_wts);
    Kokkos::deep_copy(wts,tmpwts);
  }
  
  // -------------------------------------------------
  // Compute the element sizes (h = vol^(1/dimension))
  // -------------------------------------------------
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataHsizeTimer);
    parallel_for("cell hsize",
                 RangePolicy<AssemblyExec>(0,wts.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      ScalarT vol = 0.0;
      for (size_type i=0; i<wts.extent(1); i++) {
        vol += wts(elem,i);
      }
      ScalarT dimscl = 1.0/(ScalarT)ip.extent(2);
      hsize(elem) = pow(vol,dimscl);
    });
  }
  
  // -------------------------------------------------
  // Compute the element orientations (probably never happens)
  // -------------------------------------------------
  
  if (recompute_orient) {
    this->getPhysicalOrientations(cellData, eIndex, orientation, true);
  }
  
  // -------------------------------------------------
  // Compute the basis functions at the volumetric ip
  // -------------------------------------------------
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataBasisTimer);
    for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
      
      int numb = cellData->basis_pointers[i]->getCardinality();
      
      View_Sc4 basis_vals, basis_grad_vals, basis_curl_vals, basis_node_vals;
      View_Sc3 basis_div_vals;
      
      if (cellData->basis_types[i] == "HGRAD"){
        DRV bvals("basis",numElem,numb,numip);
        DRV tmp_bvals("basis tmp",numElem,numb,numip);
        FuncTools::HGRADtransformVALUE(tmp_bvals, cellData->ref_basis[i]);
        OrientTools::modifyBasisByOrientation(bvals, tmp_bvals, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals);
        
        DRV bgrad_tmp("basis grad tmp",numElem,numb,numip,dimension);
        DRV bgrad_vals("basis grad",numElem,numb,numip,dimension);
        FuncTools::HGRADtransformGRAD(bgrad_tmp, jacobianInv, cellData->ref_basis_grad[i]);
        OrientTools::modifyBasisByOrientation(bgrad_vals, bgrad_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_grad_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_grad_vals,bgrad_vals);
        
      }
      else if (cellData->basis_types[i] == "HVOL"){
        
        DRV bvals("basis",numElem,numb,numip);
        FuncTools::HGRADtransformVALUE(bvals, cellData->ref_basis[i]);
        
        basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals);
      }
      else if (cellData->basis_types[i] == "HDIV"){
        
        DRV bvals("basis",numElem,numb,numip,dimension);
        DRV bvals_tmp("basis tmp",numElem,numb,numip,dimension);
        FuncTools::HDIVtransformVALUE(bvals_tmp, jacobian, jacobianDet, cellData->ref_basis[i]);
        OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
        Kokkos::deep_copy(basis_vals,bvals);
        
        if (cellData->requireBasisAtNodes) {
          DRV bnode_vals("basis",numElem,numb,nodes.extent(1),dimension);
          DRV bvals_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
          FuncTools::HDIVtransformVALUE(bvals_tmp, jacobian, jacobianDet, cellData->ref_basis_nodes[i]);
          OrientTools::modifyBasisByOrientation(bnode_vals, bvals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          basis_node_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_node_vals,bnode_vals);
        }
        
        DRV bdiv_vals("basis div",numElem,numb,numip);
        DRV bdiv_vals_tmp("basis div tmp",numElem,numb,numip);
        FuncTools::HDIVtransformDIV(bdiv_vals_tmp, jacobianDet, cellData->ref_basis_div[i]);
        OrientTools::modifyBasisByOrientation(bdiv_vals, bdiv_vals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_div_vals = View_Sc3("basis div values", numElem, numb, numip); // needs to be rank-3
        Kokkos::deep_copy(basis_div_vals,bdiv_vals);
      }
      else if (cellData->basis_types[i] == "HCURL"){
        
        DRV bvals("basis",numElem,numb,numip,dimension);
        DRV bvals_tmp("basis tmp",numElem,numb,numip,dimension);
        FuncTools::HCURLtransformVALUE(bvals_tmp, jacobianInv, cellData->ref_basis[i]);
        OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
        Kokkos::deep_copy(basis_vals,bvals);
        
        if (cellData->requireBasisAtNodes) {
          DRV bnode_vals("basis",numElem,numb,nodes.extent(1),dimension);
          DRV bvals_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
          FuncTools::HCURLtransformVALUE(bvals_tmp, jacobianInv, cellData->ref_basis_nodes[i]);
          OrientTools::modifyBasisByOrientation(bnode_vals, bvals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          basis_node_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_node_vals,bnode_vals);
          
        }
        
        DRV bcurl_vals("basis curl",numElem,numb,numip,dimension);
        DRV bcurl_vals_tmp("basis curl tmp",numElem,numb,numip,dimension);
        FuncTools::HCURLtransformCURL(bcurl_vals_tmp, jacobian, jacobianDet, cellData->ref_basis_curl[i]);
        OrientTools::modifyBasisByOrientation(bcurl_vals, bcurl_vals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_curl_vals = View_Sc4("basis curl values", numElem, numb, numip, dimension);
        Kokkos::deep_copy(basis_curl_vals, bcurl_vals);
        
      }
      basis.push_back(basis_vals);
      basis_grad.push_back(basis_grad_vals);
      basis_div.push_back(basis_div_vals);
      basis_curl.push_back(basis_curl_vals);
      basis_nodes.push_back(basis_node_vals);
    }
  }
}

void discretization::getPhysicalOrientations(Teuchos::RCP<CellMetaData> & cellData,
                                             Kokkos::View<LO*,AssemblyDevice> eIndex,
                                             Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                             const bool & use_block) {
  
  Teuchos::TimeMonitor localtimer(*physOrientTimer);
  
  int block = cellData->myBlock;
  int numNodesPerElem = cellData->cellTopo->getNodeCount();
  size_type numElem = eIndex.extent(0);
  
  Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices", numElem, numNodesPerElem);
  auto host_currind = Kokkos::create_mirror_view(currind);
  auto host_eIndex = Kokkos::create_mirror_view(eIndex);
  Kokkos::deep_copy(host_eIndex,eIndex);
  for (size_t i=0; i<numElem; i++) {
    vector<stk::mesh::EntityId> stk_nodeids;
    LO elemID = host_eIndex(i);
    if (use_block) {
      mesh->getNodeIdsForElement(block_stkElems[block][elemID], stk_nodeids);
    }
    else {
      mesh->getNodeIdsForElement(all_stkElems[elemID], stk_nodeids);
    }
    for (int n=0; n<numNodesPerElem; n++) {
      host_currind(i,n) = stk_nodeids[n];
    }
  }
  Kokkos::deep_copy(currind, host_currind);
  OrientTools::getOrientation(orientation, currind, *(cellData->cellTopo));
}

// -------------------------------------------------
// Compute the basis functions at the face ip
// -------------------------------------------------

void discretization::getPhysicalFaceData(Teuchos::RCP<CellMetaData> & cellData, const int & side,
                                         DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                         Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                         View_Sc3 face_ip, View_Sc2 face_wts, View_Sc3 face_normals, View_Sc1 face_hsize,
                                         vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                         const bool & recompute_jac,
                                         const bool & recompute_orient) {

  Teuchos::TimeMonitor localtimer(*physFaceDataTotalTimer);
  
  auto ref_ip = cellData->ref_side_ip[side];
  auto ref_wts = cellData->ref_side_wts[side];
  
  int dimension = cellData->dimension;
  int numip = ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  // Step 1: fill in ip_side, wts_side and normals
  DRV sip("side ip", numElem, numip, dimension);
  DRV jac("bijac", numElem, numip, dimension, dimension);
  DRV jacDet("bijacDet", numElem, numip);
  DRV jacInv("bijacInv", numElem, numip, dimension, dimension);
  DRV swts("wts_side", numElem, numip);
  DRV snormals("normals", numElem, numip, dimension);
  DRV tangents("tangents", numElem, numip, dimension);
  
  {
    Teuchos::TimeMonitor localtimer(*physFaceDataIPTimer);
    CellTools::mapToPhysicalFrame(sip, ref_ip, nodes, *(cellData->cellTopo));
    Kokkos::deep_copy(face_ip,sip);
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physFaceDataSetJacTimer);
    CellTools::setJacobian(jac, ref_ip, nodes, *(cellData->cellTopo));
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physFaceDataOtherJacTimer);
    CellTools::setJacobianInv(jacInv, jac);
    CellTools::setJacobianDet(jacDet, jac);
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physFaceDataWtsTimer);
    
    if (dimension == 2) {
      auto ref_tangents = cellData->ref_side_tangents[side];
      RealTools::matvec(tangents, jac, ref_tangents);
      
      DRV rotation("rotation matrix",dimension,dimension);
      rotation(0,0) = 0;  rotation(0,1) = 1;
      rotation(1,0) = -1; rotation(1,1) = 0;
      RealTools::matvec(snormals, rotation, tangents);
      
      RealTools::vectorNorm(swts, tangents, Intrepid2::NORM_TWO);
      ArrayTools::scalarMultiplyDataData(swts, swts, ref_wts);
      
    }
    else if (dimension == 3) {
      
      auto ref_tangentsU = cellData->ref_side_tangentsU[side];
      auto ref_tangentsV = cellData->ref_side_tangentsV[side];
      
      DRV faceTanU("face tangent U", numElem, numip, dimension);
      DRV faceTanV("face tangent V", numElem, numip, dimension);
      
      RealTools::matvec(faceTanU, jac, ref_tangentsU);
      RealTools::matvec(faceTanV, jac, ref_tangentsV);
      
      RealTools::vecprod(snormals, faceTanU, faceTanV);
      
      RealTools::vectorNorm(swts, snormals, Intrepid2::NORM_TWO);
      ArrayTools::scalarMultiplyDataData(swts, swts, ref_wts);
      
    }
    
    // scale the normal vector (we need unit normal...)
    
    parallel_for("wkset transient sol seedwhat 1",
                 TeamPolicy<AssemblyExec>(snormals.extent(0), Kokkos::AUTO, 32),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
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
    
    Kokkos::deep_copy(face_normals,snormals);
    Kokkos::deep_copy(face_wts,swts);
  }
  
  // -------------------------------------------------
  // Compute the element sizes (h = face_vol^(1/dimension-1))
  // -------------------------------------------------
  
  {
    Teuchos::TimeMonitor localtimer(*physFaceDataHsizeTimer);
    
    {
      using std::pow;
      using std::sqrt;
      parallel_for("bcell hsize",
                   RangePolicy<AssemblyExec>(0,face_wts.extent(0)),
                   KOKKOS_LAMBDA (const int e ) {
        ScalarT vol = 0.0;
        for (size_type i=0; i<face_wts.extent(1); i++) {
          vol += face_wts(e,i);
        }
        ScalarT dimscl = 1.0/((ScalarT)face_ip.extent(2)-1.0);
        face_hsize(e) = pow(vol,dimscl);
      });
    }
  }
  
  // -------------------------------------------------
  // Compute the element orientations
  // -------------------------------------------------
  
  if (recompute_orient) {
    this->getPhysicalOrientations(cellData, eIndex, orientation, true);
  }
  
  // Step 2: define basis functions at these integration points
  
  {
    Teuchos::TimeMonitor localtimer(*physFaceDataBasisTimer);
    
    for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
      int numb = cellData->basis_pointers[i]->getCardinality();
      View_Sc4 basis_vals, basis_grad_vals;//, basis_div_vals, basis_curl_vals;
      
      auto ref_basis_vals = cellData->ref_side_basis[side][i];
      
      if (cellData->basis_types[i] == "HGRAD"){
        
        DRV bvals_tmp("tmp basis_vals",numElem, numb, numip);
        DRV bvals("basis_vals",numElem, numb, numip);
        FuncTools::HGRADtransformVALUE(bvals_tmp, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals);
        
        auto ref_basis_grad_vals = cellData->ref_side_basis_grad[side][i];
        DRV bgrad_vals_tmp("tmp basis_grad_vals",numElem, numb, numip, dimension);
        DRV bgrad_vals("basis_grad_vals",numElem, numb, numip, dimension);
        FuncTools::HGRADtransformGRAD(bgrad_vals_tmp, jacInv, ref_basis_grad_vals);
        OrientTools::modifyBasisByOrientation(bgrad_vals, bgrad_vals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_grad_vals = View_Sc4("face basis grad vals",numElem,numb,numip,dimension); // Needs to be rank-4
        Kokkos::deep_copy(basis_grad_vals,bgrad_vals);
        
      }
      else if (cellData->basis_types[i] == "HVOL"){
        
        DRV bvals("basis_vals",numElem, numb, numip);
        FuncTools::HGRADtransformVALUE(bvals, ref_basis_vals);
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals);
      }
      else if (cellData->basis_types[i] == "HDIV"){
        
        DRV bvals_tmp("tmp basis_vals",numElem, numb, numip, dimension);
        DRV bvals("basis_vals",numElem, numb, numip, dimension);
        
        FuncTools::HDIVtransformVALUE(bvals_tmp, jac, jacDet, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_vals,bvals);
        
      }
      else if (cellData->basis_types[i] == "HCURL"){
        //FunctionSpaceTools<AssemblyDevice>::multiplyMeasure(basis_side[i], wts_side, ref_basis_side[s][i]);
      }
      else if (cellData->basis_types[i] == "HFACE"){
        
        DRV bvals("basis_vals",numElem, numb, numip);
        DRV bvals_tmp("basisvals_Transformed",numElem, numb, numip);
        FuncTools::HGRADtransformVALUE(bvals_tmp, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals);
        
      }
      
      basis.push_back(basis_vals);
      basis_grad.push_back(basis_grad_vals);
    }
  }
  
}


void discretization::getPhysicalBoundaryData(Teuchos::RCP<CellMetaData> & cellData,
                                             DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                             Kokkos::View<LO*,AssemblyDevice> localSideID,
                                             Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                             View_Sc3 ip, View_Sc2 wts, View_Sc3 normals, View_Sc3 tangents, View_Sc1 hsize,
                                             vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                             vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                             const bool & recompute_jac,
                                             const bool & recompute_orient) {
  
  Teuchos::TimeMonitor localtimer(*physBndryDataTotalTimer);
  
  int dimension = cellData->dimension;
  
  auto localSideID_host = Kokkos::create_mirror_view(localSideID);
  Kokkos::deep_copy(localSideID_host,localSideID);
  DRV ref_ip = cellData->ref_side_ip[localSideID_host(0)];
  DRV ref_wts = cellData->ref_side_wts[localSideID_host(0)];
  
  int numip = ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  DRV tmpip("tmp boundary ip", numElem, numip, dimension);
  DRV ijac("bijac", numElem, numip, dimension, dimension);
  DRV ijacDet("bijacDet", numElem, numip);
  DRV ijacInv("bijacInv", numElem, numip, dimension, dimension);
  DRV tmpwts("tmp boundary wts", numElem, numip);
  DRV tmpnormals("tmp boundary normals", numElem, numip, dimension);
  DRV tmptangents("tmp boundary tangents", numElem, numip, dimension);
  
  {
    Teuchos::TimeMonitor localtimer(*physBndryDataIPTimer);
    CellTools::mapToPhysicalFrame(tmpip, ref_ip, nodes, *(cellData->cellTopo));
    Kokkos::deep_copy(ip,tmpip);
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physBndryDataSetJacTimer);
    CellTools::setJacobian(ijac, ref_ip, nodes, *(cellData->cellTopo));
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physBndryDataOtherJacTimer);
    CellTools::setJacobianInv(ijacInv, ijac);
    CellTools::setJacobianDet(ijacDet, ijac);
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physBndryDataWtsTimer);
    if (dimension == 1) {
      Kokkos::deep_copy(tmpwts,1.0);
      auto ref_normals = cellData->ref_side_normals[localSideID_host(0)];
      parallel_for("bcell 1D normal copy",
                   RangePolicy<AssemblyExec>(0,tmpnormals.extent(0)),
                   KOKKOS_LAMBDA (const int elem ) {
        tmpnormals(elem,0,0) = ref_normals(0,0);
      });
      
    }
    else if (dimension == 2) {
      DRV ref_tangents = cellData->ref_side_tangents[localSideID_host(0)];
      RealTools::matvec(tmptangents, ijac, ref_tangents);
      
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
      
      DRV ref_tangentsU = cellData->ref_side_tangentsU[localSideID_host(0)];
      DRV ref_tangentsV = cellData->ref_side_tangentsV[localSideID_host(0)];
      
      DRV faceTanU("face tangent U", numElem, numip, dimension);
      DRV faceTanV("face tangent V", numElem, numip, dimension);
      
      RealTools::matvec(faceTanU, ijac, ref_tangentsU);
      RealTools::matvec(faceTanV, ijac, ref_tangentsV);
      
      RealTools::vecprod(tmpnormals, faceTanU, faceTanV);
      
      RealTools::vectorNorm(tmpwts, tmpnormals, Intrepid2::NORM_TWO);
      ArrayTools::scalarMultiplyDataData(tmpwts, tmpwts, ref_wts);
      
    }
    Kokkos::deep_copy(wts,tmpwts);
    Kokkos::deep_copy(normals,tmpnormals);
    Kokkos::deep_copy(tangents,tmptangents);
  }
  
  // -------------------------------------------------
  // Compute the element sizes (h = bndry_vol^(1/dimension-1))
  // -------------------------------------------------
  
  {
    Teuchos::TimeMonitor localtimer(*physBndryDataHsizeTimer);
    using std::pow;
    using std::sqrt;
    parallel_for("bcell hsize",
                 RangePolicy<AssemblyExec>(0,wts.extent(0)),
                 KOKKOS_LAMBDA (const int e ) {
      ScalarT vol = 0.0;
      for (size_type i=0; i<wts.extent(1); i++) {
        vol += wts(e,i);
      }
      ScalarT dimscl = 1.0/((ScalarT)ip.extent(2)-1.0);
      hsize(e) = pow(vol,dimscl);
    });
  }
  
  // -------------------------------------------------
  // Rescale the normals
  // -------------------------------------------------
  
  {
    parallel_for("bcell normal rescale",
                 TeamPolicy<AssemblyExec>(normals.extent(0), Kokkos::AUTO),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type pt=team.team_rank(); pt<normals.extent(1); pt+=team.team_size() ) {
        ScalarT normalLength = 0.0;
        for (size_type sd=0; sd<normals.extent(2); sd++) {
          normalLength += normals(elem,pt,sd)*normals(elem,pt,sd);
        }
        normalLength = sqrt(normalLength);
        for (size_type sd=0; sd<normals.extent(2); sd++) {
          normals(elem,pt,sd) = normals(elem,pt,sd) / normalLength;
        }
      }
    });
  }
  
  // -------------------------------------------------
  // Compute the element orientations
  // -------------------------------------------------
  
  // -------------------------------------------------
  // Compute the element orientations
  // -------------------------------------------------
  
  if (recompute_orient) {
    this->getPhysicalOrientations(cellData, eIndex, orientation, false);
  }
  
  /*
  {
    Teuchos::TimeMonitor localtimer(*physBndryDataOrientTimer);
    Kokkos::DynRankView<stk::mesh::EntityId,AssemblyDevice> currind("current node indices", numElem, numNodesPerElem);
    auto host_currind = Kokkos::create_mirror_view(currind);
    auto host_eIndex = Kokkos::create_mirror_view(eIndex);
    Kokkos::deep_copy(host_eIndex,eIndex);
    for (int i=0; i<numElem; i++) {
      vector<stk::mesh::EntityId> stk_nodeids;
      LO elemID = host_eIndex(i); //prog+i;//host_eIndex(i);
      mesh->getNodeIdsForElement(all_stkElems[elemID], stk_nodeids); // TMW: why is this different???
      for (int n=0; n<numNodesPerElem; n++) {
        host_currind(i,n) = stk_nodeids[n];
      }
    }
    Kokkos::deep_copy(currind, host_currind);
    
    OrientTools::getOrientation(orientation, currind, *(cellData->cellTopo));
  }*/
  
  {
    Teuchos::TimeMonitor localtimer(*physBndryDataBasisTimer);
    
    for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
      
      int numb = cellData->basis_pointers[i]->getCardinality();
      View_Sc4 basis_vals, basis_grad_vals, basis_curl_vals;
      View_Sc3 basis_div_vals;
      
      DRV ref_basis_vals = cellData->ref_side_basis[localSideID_host(0)][i];
      
      if (cellData->basis_types[i] == "HGRAD"){
        
        DRV bvals_tmp("tmp basis_vals",numElem, numb, numip);
        FuncTools::HGRADtransformVALUE(bvals_tmp, ref_basis_vals);
        DRV bvals("basis_vals",numElem, numb, numip);
        OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,1);
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals);
        
        DRV ref_bgrad_vals = cellData->ref_side_basis_grad[localSideID_host(0)][i];
        DRV bgrad_vals_tmp("basis_grad_side tmp",numElem,numb,numip,dimension);
        FuncTools::HGRADtransformGRAD(bgrad_vals_tmp, ijacInv, ref_bgrad_vals);
        
        DRV bgrad_vals("basis_grad_vals",numElem,numb,numip,dimension);
        OrientTools::modifyBasisByOrientation(bgrad_vals, bgrad_vals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_grad_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_grad_vals,bgrad_vals);
        
      }
      else if (cellData->basis_types[i] == "HVOL"){ // does not require orientations
        
        DRV bvals("basis_vals",numElem, numb, numip);
        FuncTools::HGRADtransformVALUE(bvals, ref_basis_vals);
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,1);
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals);
      }
      else if (cellData->basis_types[i] == "HFACE"){
        
        DRV bvals_tmp("tmp basis_vals",numElem, numb, numip);
        FuncTools::HGRADtransformVALUE(bvals_tmp, ref_basis_vals);
        DRV bvals("basis_vals",numElem, numb, numip);
        OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,1);
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals);
      }
      else if (cellData->basis_types[i] == "HDIV"){
        
        DRV bvals_tmp("tmp basis_vals",numElem, numb, numip, dimension);
        
        FuncTools::HDIVtransformVALUE(bvals_tmp, ijac, ijacDet, ref_basis_vals);
        DRV bvals("basis_vals",numElem, numb, numip, dimension);
        OrientTools::modifyBasisByOrientation(bvals, bvals_tmp, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_vals,bvals);
      }
      else if (cellData->basis_types[i] == "HCURL"){
        
      }
      basis.push_back(basis_vals);
      basis_grad.push_back(basis_grad_vals);
      basis_div.push_back(basis_div_vals);
      basis_curl.push_back(basis_curl_vals);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate basis at reference element integration points (should be deprecated)
//////////////////////////////////////////////////////////////////////////////////////

DRV discretization::evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts) {
  int numCells = 1;//evalpts.extent(0);
  int numpts = evalpts.extent(0);
  int numBasis = basis_pointer->getCardinality();
  DRV basisvals("basisvals", numBasis, numpts);
  basis_pointer->getValues(basisvals, evalpts, Intrepid2::OPERATOR_VALUE);
  
  DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
  FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
  
  return basisvals_Transformed;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

DRV discretization::evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts,
                                  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation) {
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

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

DRV discretization::evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
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

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

DRV discretization::evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
                                       const DRV & evalpts, const topo_RCP & cellTopo,
                                       Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation) {
  
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

/////////////////////////////////////////////////////////////////////////////////////////////
// After the mesh and the discretizations have been defined, we can create and add the physics
// to the DOF manager
/////////////////////////////////////////////////////////////////////////////////////////////

void discretization::buildDOFManagers() {
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics::buildDOF ..." << endl;
    }
  }
  
  Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));

  // DOF manager for the primary variables
  {
    DOF = Teuchos::rcp(new panzer::DOFManager());
    DOF->setConnManager(conn,*(Commptr->getRawMpiComm()));
    DOF->setOrientationsRequired(true);
    
    for (size_t b=0; b<blocknames.size(); b++) {
      for (size_t j=0; j<phys->varlist[b].size(); j++) {
        topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
        basis_RCP basis_pointer = this->getBasis(spaceDim, cellTopo, phys->types[b][j], phys->orders[b][j]);
        
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_pointer));
        
        if (phys->useDG[b][j]) {
          DOF->addField(blocknames[b], phys->varlist[b][j], Pattern, panzer::FieldType::DG);
        }
        else {
          DOF->addField(blocknames[b], phys->varlist[b][j], Pattern, panzer::FieldType::CG);
        }
      }
    }
    if (settings->sublist("Physics").isParameter("field order")) {
      //DOF->setFieldOrder(settings->sublist("Physics").get<string>("field order",""));
    }
    
    DOF->buildGlobalUnknowns();
    
    for (size_t b=0; b<blocknames.size(); b++) {
      int numGIDs = DOF->getElementBlockGIDCount(blocknames[b]);
      TEUCHOS_TEST_FOR_EXCEPTION(numGIDs > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of degrees of freedom per element on block: " + blocknames[b]);
    }
    if (verbosity>1) {
      if (Commptr->getRank() == 0) {
        DOF->printFieldInformation(std::cout);
      }
    }
  }
  
  // DOF manager for the aux variables
  if (phys->have_aux) {
    auxDOF = Teuchos::rcp(new panzer::DOFManager());
    auxDOF->setConnManager(conn,*(Commptr->getRawMpiComm()));
    auxDOF->setOrientationsRequired(true);
    
    for (size_t b=0; b<blocknames.size(); b++) {
      for (size_t j=0; j<phys->aux_varlist[b].size(); j++) {
        topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
        basis_RCP basis_pointer = this->getBasis(spaceDim, cellTopo, phys->aux_types[b][j], phys->aux_orders[b][j]);
        
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_pointer));
        
        if (phys->aux_useDG[b][j]) {
          auxDOF->addField(blocknames[b], phys->aux_varlist[b][j], Pattern, panzer::FieldType::DG);
        }
        else {
          auxDOF->addField(blocknames[b], phys->aux_varlist[b][j], Pattern, panzer::FieldType::CG);
        }
      }
    }
    
    auxDOF->buildGlobalUnknowns();
    
    for (size_t b=0; b<blocknames.size(); b++) {
      int numGIDs = auxDOF->getElementBlockGIDCount(blocknames[b]);
      TEUCHOS_TEST_FOR_EXCEPTION(numGIDs > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of aux degrees of freedom per element on block: " + blocknames[b]);
    }
    if (verbosity>1) {
      if (Commptr->getRank() == 0) {
        auxDOF->printFieldInformation(std::cout);
      }
    }
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics::buildDOF" << endl;
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void discretization::setBCData(const bool & isaux) {
  
  Teuchos::TimeMonitor localtimer(*setbctimer);
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting discretization::setBCData ..." << endl;
    }
  }
  
  vector<string> sideSets, nodeSets;
  mesh->getSidesetNames(sideSets);
  mesh->getNodesetNames(nodeSets);
  
  vector<vector<string> > varlist;
  vector<int> numVars;
  Teuchos::RCP<panzer::DOFManager> currDOF;
  if (isaux) {
    varlist = phys->aux_varlist;
    numVars = phys->aux_numVars;
    currDOF = auxDOF;
  }
  else {
    varlist = phys->varlist;
    numVars = phys->numVars;
    currDOF = DOF;
  }
  
  int maxvars = 0;
  for (size_t b=0; b<blocknames.size(); b++) {
    for (size_t j=0; j<varlist[b].size(); j++) {
      string var = varlist[b][j];
      int num = currDOF->getFieldNum(var);
      maxvars = std::max(num,maxvars);
    }
  }
    
  vector<size_t> numElem;
  for (size_t b=0; b<blocknames.size(); b++) {
    
    Kokkos::View<string**,HostDevice> currbcs("boundary conditions",
                                              varlist[b].size(),sideSets.size());
    topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
    int numSidesPerElem = 2; // default to 1D for some reason
    if (spaceDim == 2) {
      numSidesPerElem = cellTopo->getEdgeCount();
    }
    else if (spaceDim == 3) {
      numSidesPerElem = cellTopo->getFaceCount();
    }
    
    std::string blockID = blocknames[b];
    vector<stk::mesh::Entity> stk_meshElems;
    mesh->getMyElements(blockID, stk_meshElems);
    size_t maxElemLID = 0;
    for (size_t i=0; i<stk_meshElems.size(); i++) {
      size_t lid = mesh->elementLocalId(stk_meshElems[i]);
      maxElemLID = std::max(lid,maxElemLID);
    }
    std::vector<size_t> localelemmap(maxElemLID+1);
    for (size_t i=0; i<stk_meshElems.size(); i++) {
      size_t lid = mesh->elementLocalId(stk_meshElems[i]);
      localelemmap[lid] = i;
    }
    
    // TMW: is this needed?
    if (!isaux) {
      numElem.push_back(stk_meshElems.size());
    }
    
    Teuchos::ParameterList blocksettings;
    if (isaux) {
      if (settings->sublist("Aux Physics").isSublist(blockID)) {
        blocksettings = settings->sublist("Aux Physics").sublist(blockID);
      }
      else {
        blocksettings = settings->sublist("Aux Physics");
      }
    }
    else {
      if (settings->sublist("Physics").isSublist(blockID)) {
        blocksettings = settings->sublist("Physics").sublist(blockID);
      }
      else {
        blocksettings = settings->sublist("Physics");
      }
    }
    
    Teuchos::ParameterList dbc_settings = blocksettings.sublist("Dirichlet conditions");
    Teuchos::ParameterList nbc_settings = blocksettings.sublist("Neumann conditions");
    bool use_weak_dbcs = dbc_settings.get<bool>("use weak Dirichlet",false);
    
    vector<vector<int> > celloffsets;
    Kokkos::View<int****,HostDevice> currside_info("side info",stk_meshElems.size(),numVars[b],numSidesPerElem,2);
    
    std::vector<int> block_dbc_dofs;
    
    std::string perBCs = settings->sublist("Mesh").get<string>("Periodic Boundaries","");
    
    for (size_t j=0; j<phys->varlist[b].size(); j++) {
      string var = phys->varlist[b][j];
      int num = currDOF->getFieldNum(var);
      vector<int> var_offsets = currDOF->getGIDFieldOffsets(blockID,num);
      
      celloffsets.push_back(var_offsets);
      
      for( size_t side=0; side<sideSets.size(); side++ ) {
        string sideName = sideSets[side];
        
        vector<stk::mesh::Entity> sideEntities;
        mesh->getMySides(sideName, blockID, sideEntities);
        
        bool isDiri = false;
        bool isNeum = false;
        if (dbc_settings.sublist(var).isParameter("all boundaries") || dbc_settings.sublist(var).isParameter(sideName)) {
          isDiri = true;
          if (use_weak_dbcs) {
            currbcs(j,side) = "weak Dirichlet";
          }
          else {
            currbcs(j,side) = "Dirichlet";
          }
        }
        if (nbc_settings.sublist(var).isParameter("all boundaries") || nbc_settings.sublist(var).isParameter(sideName)) {
          isNeum = true;
          currbcs(j,side) = "Neumann";
        }
        
        vector<size_t>             local_side_Ids;
        vector<stk::mesh::Entity> side_output;
        vector<size_t>             local_elem_Ids;
        panzer_stk::workset_utils::getSideElements(*mesh, blockID, sideEntities, local_side_Ids, side_output);
        
        for( size_t i=0; i<side_output.size(); i++ ) {
          local_elem_Ids.push_back(mesh->elementLocalId(side_output[i]));
          size_t localid = localelemmap[local_elem_Ids[i]];
          if( isDiri ) {
            if (use_weak_dbcs) {
              currside_info(localid, j, local_side_Ids[i], 0) = 4;
            }
            else {
              currside_info(localid, j, local_side_Ids[i], 0) = 1;
            }
            currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
          }
          else if (isNeum) { // Neumann or Robin
            currside_info(localid, j, local_side_Ids[i], 0) = 2;
            currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
          }
        }
      }
      
      // nodeset loop
      string point_DBCs = blocksettings.get<std::string>(var+"_point_DBCs","");
      
      vector<int> dbc_nodes;
      for( size_t node=0; node<nodeSets.size(); node++ ) {
        string nodeName = nodeSets[node];
        std::size_t found = point_DBCs.find(nodeName);
        bool isDiri = false;
        if (found!=std::string::npos) {
          isDiri = true;
        }
        
        if (isDiri) {
          vector<stk::mesh::Entity> nodeEntities;
          mesh->getMyNodes(nodeName, blockID, nodeEntities);
          vector<GO> elemGIDs;
          
          vector<size_t> local_elem_Ids;
          vector<size_t> local_node_Ids;
          vector<stk::mesh::Entity> side_output;
          panzer_stk::workset_utils::getNodeElements(*mesh,blockID,nodeEntities,local_node_Ids,side_output);
          
          for( size_t i=0; i<side_output.size(); i++ ) {
            local_elem_Ids.push_back(mesh->elementLocalId(side_output[i]));
            size_t localid = localelemmap[local_elem_Ids[i]];
            currDOF->getElementGIDs(localid,elemGIDs,blockID);
            block_dbc_dofs.push_back(elemGIDs[var_offsets[local_node_Ids[i]]]);
          }
        }
        
      }
    }
    
    if (isaux) {
      aux_offsets.push_back(celloffsets);
      aux_var_bcs.push_back(currbcs);
    }
    else {
      offsets.push_back(celloffsets);
      var_bcs.push_back(currbcs);
      side_info.push_back(currside_info);
    }
    
    std::sort(block_dbc_dofs.begin(), block_dbc_dofs.end());
    block_dbc_dofs.erase(std::unique(block_dbc_dofs.begin(),
                                     block_dbc_dofs.end()), block_dbc_dofs.end());
    
    int localsize = (int)block_dbc_dofs.size();
    int globalsize = 0;
    
    Teuchos::reduceAll<int,int>(*Commptr,Teuchos::REDUCE_SUM,1,&localsize,&globalsize);
    int gathersize = Commptr->getSize()*globalsize;
    int *block_dbc_dofs_local = new int [globalsize];
    int *block_dbc_dofs_global = new int [gathersize];
    
    int mxdof = (int) block_dbc_dofs.size();
    for (int i = 0; i < globalsize; i++) {
      if ( i < mxdof) {
        block_dbc_dofs_local[i] = (int) block_dbc_dofs[i];
      }
      else {
        block_dbc_dofs_local[i] = -1;
      }
    }
    
    Teuchos::gatherAll(*Commptr, globalsize, &block_dbc_dofs_local[0], gathersize, &block_dbc_dofs_global[0]);
    vector<GO> all_dbcs;
    
    for (int i = 0; i < gathersize; i++) {
      all_dbcs.push_back(block_dbc_dofs_global[i]);
    }
    delete [] block_dbc_dofs_local;
    delete [] block_dbc_dofs_global;
    
    vector<GO> dbc_final;
    vector<GO> ownedAndShared;
    currDOF->getOwnedAndGhostedIndices(ownedAndShared);
    
    sort(all_dbcs.begin(),all_dbcs.end());
    sort(ownedAndShared.begin(),ownedAndShared.end());
    set_intersection(all_dbcs.begin(),all_dbcs.end(),
                     ownedAndShared.begin(),ownedAndShared.end(),
                     back_inserter(dbc_final));
    
    if (isaux) {
      aux_point_dofs.push_back(dbc_final);
    }
    else {
      point_dofs.push_back(dbc_final);
    }
    
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished discretization::setBCData" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void discretization::setDirichletData(const bool & isaux) {
  
  Teuchos::TimeMonitor localtimer(*setdbctimer);
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting discretization::setDirichletData ..." << endl;
    }
  }
  
  vector<string> sideNames;
  mesh->getSidesetNames(sideNames);
  
  vector<vector<string> > varlist;
  Teuchos::RCP<panzer::DOFManager> currDOF;
  if (isaux) {
    varlist = phys->aux_varlist;
    currDOF = auxDOF;
  }
  else {
    varlist = phys->varlist;
    currDOF = DOF;
  }
  
  for (size_t b=0; b<blocknames.size(); b++) {
    
    std::string blockID = blocknames[b];
    
    Teuchos::ParameterList dbc_settings;
    if (isaux) {
      if (settings->sublist("Aux Physics").isSublist(blockID)) {
        dbc_settings = settings->sublist("Aux Physics").sublist(blockID).sublist("Dirichlet conditions");
      }
      else {
        dbc_settings = settings->sublist("Aux Physics").sublist("Dirichlet conditions");
      }
    }
    else {
      if (settings->sublist("Physics").isSublist(blockID)) {
        dbc_settings = settings->sublist("Physics").sublist(blockID).sublist("Dirichlet conditions");
      }
      else {
        dbc_settings = settings->sublist("Physics").sublist("Dirichlet conditions");
      }
    }
    std::vector<std::vector<LO> > block_dbc_dofs;
    
    for (size_t j=0; j<varlist[b].size(); j++) {
      std::string var = varlist[b][j];
      int fieldnum = currDOF->getFieldNum(var);
      std::vector<LO> var_dofs;
      for (size_t side=0; side<sideNames.size(); side++ ) {
        std::string sideName = sideNames[side];
        vector<stk::mesh::Entity> sideEntities;
        mesh->getMySides(sideName, blockID, sideEntities);
        
        bool isDiri = false;
        if (dbc_settings.sublist(var).isParameter("all boundaries") || dbc_settings.sublist(var).isParameter(sideName)) {
          isDiri = true;
          if (isaux) {
            haveAuxDirichlet = true;
          }
          else {
            haveDirichlet = true;
          }
        }
        
        if (isDiri) {
          
          vector<size_t>             local_side_Ids;
          vector<stk::mesh::Entity>  side_output;
          vector<size_t>             local_elem_Ids;
          panzer_stk::workset_utils::getSideElements(*mesh, blockID, sideEntities,
                                                     local_side_Ids, side_output);
        
          for( size_t i=0; i<side_output.size(); i++ ) {
            LO local_EID = mesh->elementLocalId(side_output[i]);
            auto elemLIDs = currDOF->getElementLIDs(local_EID);
            const std::pair<vector<int>,vector<int> > SideIndex = currDOF->getGIDFieldOffsets_closure(blockID, fieldnum,
                                                                                                      spaceDim-1,
                                                                                                      local_side_Ids[i]);
            const vector<int> sideOffset = SideIndex.first;
            
            for( size_t i=0; i<sideOffset.size(); i++ ) { // for each node
              var_dofs.push_back(elemLIDs(sideOffset[i]));
            }
          }
        }
      }
      std::sort(var_dofs.begin(), var_dofs.end());
      var_dofs.erase(std::unique(var_dofs.begin(), var_dofs.end()), var_dofs.end());
      
      block_dbc_dofs.push_back(var_dofs);
    }
    if (isaux) {
      aux_dbc_dofs.push_back(block_dbc_dofs);
    }
    else {
      dbc_dofs.push_back(block_dbc_dofs);
    }
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished discretization::setDirichletData" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<int****,HostDevice> discretization::getSideInfo(const size_t & block,
                                                             Kokkos::View<int*,HostDevice> elem) {
  
  size_type nelem = elem.extent(0);
  size_type nvars = side_info[block].extent(1);
  size_type nelemsides = side_info[block].extent(2);
  //size_type nglobalsides = side_info[block].extent(3);
  Kokkos::View<int****,HostDevice> currsi("side info for cell",nelem,nvars,nelemsides, 2);
  for (size_type e=0; e<nelem; e++) {
    for (size_type j=0; j<nelemsides; j++) {
      for (size_type i=0; i<nvars; i++) {
        int sidetype = side_info[block](elem(e),i,j,0);
        if (sidetype > 0) { // TMW: why is this here?
          currsi(e,i,j,0) = sidetype;
          currsi(e,i,j,1) = side_info[block](elem(e),i,j,1);
        }
        else {
          currsi(e,i,j,0) = sidetype;
          currsi(e,i,j,1) = 0;
        }
      }
    }
  }
  return currsi;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<vector<int> > discretization::getOffsets(const int & block) {
  return offsets[block];
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV discretization::mapPointsToReference(DRV phys_pts, DRV nodes, topo_RCP & cellTopo) {
  DRV ref_pts("reference cell points",phys_pts.extent(0), phys_pts.extent(1), phys_pts.extent(2));
  CellTools::mapToReferenceFrame(ref_pts, phys_pts, nodes, *cellTopo);
  return ref_pts;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV discretization::getReferenceNodes(topo_RCP & cellTopo) {
  int dimension = cellTopo->getDimension();
  int numnodes = cellTopo->getNodeCount();
  DRV refnodes("nodes on reference element",numnodes,dimension);
  CellTools::getReferenceSubcellVertices(refnodes, dimension, 0, *cellTopo);
  return refnodes;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV discretization::mapPointsToPhysical(DRV ref_pts, DRV nodes, topo_RCP & cellTopo) {
  DRV phys_pts("reference cell points",nodes.extent(0), ref_pts.extent(0), ref_pts.extent(1));
  CellTools::mapToPhysicalFrame(phys_pts, ref_pts, nodes, *cellTopo);
  return phys_pts;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::DynRankView<int,PHX::Device> discretization::checkInclusionPhysicalData(DRV phys_pts, DRV nodes,
                                                                                topo_RCP & cellTopo,
                                                                                const ScalarT & tol) {
  DRV ref_pts = this->mapPointsToReference(phys_pts,nodes,cellTopo);

  Kokkos::DynRankView<int,PHX::Device> inRefCell("inRefCell", 1, phys_pts.extent(1));

  CellTools::checkPointwiseInclusion(inRefCell, ref_pts, *cellTopo, tol);
  
  return inRefCell;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV discretization::applyOrientation(DRV basis, Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                     basis_RCP & basis_pointer) {

  DRV new_basis("basis values", basis.extent(0), basis.extent(1), basis.extent(2));
  OrientTools::modifyBasisByOrientation(new_basis, basis, orientation, basis_pointer.get());
  return new_basis;
}


/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we can get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

void discretization::purgeMemory() {
  
  DOF.reset();// = Teuchos::rcp(Teuchos::NULL);
  auxDOF.reset();// = Teuchos::rcp(Teuchos::NULL);
  
  bool storeAll = settings->sublist("Solver").get<bool>("store all cell data",true);
  if (storeAll) {
    all_stkElems.clear();
    block_stkElems.clear();
  }
  
  bool write_solution = settings->sublist("Postprocess").get("write solution",false);
  bool write_aux_solution = settings->sublist("Postprocess").get("write aux solution",false);
  bool create_optim_movie = settings->sublist("Postprocess").get("create optimization movie",false);
  if (!write_solution && !write_aux_solution && !create_optim_movie) {
    mesh.reset();
  }
}
