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
#include "Panzer_NodalFieldPattern.hpp"
#include "Panzer_OrientationsInterface.hpp"

// HGRAD basis functions
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

// HVOL basis functions
#include "Intrepid2_HVOL_C0_FEM.hpp"

// HDIV basis functions
#include "Intrepid2_HDIV_QUAD_I1_FEM.hpp"
#include "Intrepid2_HDIV_QUAD_In_FEM.hpp"
#include "Intrepid2_HDIV_HEX_I1_FEM.hpp"
#include "Intrepid2_HDIV_HEX_In_FEM.hpp"
#include "Intrepid2_HDIV_TRI_I1_FEM.hpp"
#include "Intrepid2_HDIV_TRI_In_FEM.hpp"
#include "Intrepid2_HDIV_TET_I1_FEM.hpp"
#include "Intrepid2_HDIV_TET_In_FEM.hpp"

// HDIV Arbogast-Correa basis functions
#include "Intrepid2_HDIV_AC_QUAD_I1_FEM.hpp"

// HCURL basis functions
#include "Intrepid2_HCURL_QUAD_I1_FEM.hpp"
#include "Intrepid2_HCURL_QUAD_In_FEM.hpp"
#include "Intrepid2_HCURL_HEX_I1_FEM.hpp"
#include "Intrepid2_HCURL_HEX_In_FEM.hpp"
#include "Intrepid2_HCURL_TRI_I1_FEM.hpp"
#include "Intrepid2_HCURL_TRI_In_FEM.hpp"
#include "Intrepid2_HCURL_TET_I1_FEM.hpp"
#include "Intrepid2_HCURL_TET_In_FEM.hpp"

// HFACE (experimental) basis functions
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
#include "Panzer_STK_SetupUtilities.hpp"

typedef Intrepid2::CellTools<PHX::Device::execution_space> CellTools;
typedef Intrepid2::FunctionSpaceTools<PHX::Device::execution_space> FuncTools;
typedef Intrepid2::OrientationTools<PHX::Device::execution_space> OrientTools;
typedef Intrepid2::RealSpaceTools<PHX::Device::execution_space> RealTools;
typedef Intrepid2::ArrayTools<PHX::Device::execution_space> ArrayTools;

using namespace MrHyDE;

DiscretizationInterface::DiscretizationInterface(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                                                 Teuchos::RCP<MpiComm> & Comm_,
                                                 Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                                                 Teuchos::RCP<PhysicsInterface> & phys_) :
settings(settings_), Commptr(Comm_), mesh(mesh_), phys(phys_) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  debug_level = settings->get<int>("debug level",0);
  verbosity = settings->get<int>("verbosity",0);
  minimize_memory = settings->sublist("Solver").get<bool>("minimize memory",false);
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting DiscretizationInterface constructor..." << endl;
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Collect some information
  ////////////////////////////////////////////////////////////////////////////////
  
  spaceDim = mesh->getDimension();
  mesh->getElementBlockNames(blocknames);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Assemble the information we always store
  ////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<int> > orders = phys->unique_orders;
  vector<vector<string> > types = phys->unique_types;
  
  for (size_t block=0; block<blocknames.size(); ++block) {
    
    string blockID = blocknames[block];
    topo_RCP cellTopo = mesh->getCellTopology(blockID);
    string shape = cellTopo->getName();
    
    vector<stk::mesh::Entity> stk_meshElems;
    mesh->getMyElements(blockID, stk_meshElems);
    
    // list of all elements on this processor
    vector<size_t> blockmyElements = vector<size_t>(stk_meshElems.size());
    for( size_t e=0; e<stk_meshElems.size(); e++ ) {
      blockmyElements[e] = mesh->elementLocalId(stk_meshElems[e]);
    }
    myElements.push_back(blockmyElements);
    
    vector<int> blockcards;
    vector<basis_RCP> blockbasis;
    
    vector<int> doneorders;
    vector<string> donetypes;
    
    for (size_t set=0; set<phys->setnames.size(); ++set) {
      Teuchos::ParameterList db_settings = phys->setDiscSettings[set][block];
      
      ///////////////////////////////////////////////////////////////////////////
      // Get the cardinality of the basis functions  on this block
      ///////////////////////////////////////////////////////////////////////////
      
      for (size_t n=0; n<orders[block].size(); n++) {
        bool go = true;
        for (size_t i=0; i<doneorders.size(); i++){
          if (doneorders[i] == orders[block][n] && donetypes[i] == types[block][n]) {
            go = false;
          }
        }
        if (go) {
          basis_RCP basis = this->getBasis(spaceDim, cellTopo, types[block][n], orders[block][n]);
          int bsize = basis->getCardinality();
          blockcards.push_back(bsize); // cardinality of the basis
          blockbasis.push_back(basis);
          doneorders.push_back(orders[block][n]);
          donetypes.push_back(types[block][n]);
        }
      }
    }
    basis_types.push_back(donetypes);
    cards.push_back(blockcards);
    
    ///////////////////////////////////////////////////////////////////////////
    // Quadrature
    ///////////////////////////////////////////////////////////////////////////
    
    int mxorder = 0;
    for (size_t i=0; i<orders[block].size(); i++) {
      if (orders[block][i]>mxorder) {
        mxorder = orders[block][i];
      }
    }
    
    DRV qpts, qwts;
    int quadorder = phys->setDiscSettings[0][block].get<int>("quadrature",2*mxorder); // hard coded
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
      int side_quadorder = phys->setDiscSettings[0][block].get<int>("side quadrature",2*mxorder); // hard coded
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
  
  this->setBCData();
  
  this->setDirichletData();
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished DiscretizationInterface constructor" << endl;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Create a pointer to an Intrepid or Panzer basis
// Note that these always use double rather than ScalarT
//////////////////////////////////////////////////////////////////////////////////////

basis_RCP DiscretizationInterface::getBasis(const int & spaceDim, const topo_RCP & cellTopo,
                                            const string & type, const int & degree) {
  using namespace Intrepid2;
  
  Teuchos::RCP<Intrepid2::Basis<PHX::Device::execution_space, double, double > > basis;
  
  string shape = cellTopo->getName();
  
  if (type == "HGRAD") {
    if (spaceDim == 1) {
      basis = Teuchos::rcp(new Basis_HGRAD_LINE_C1_FEM<PHX::Device::execution_space,double,double>() );
    }
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        if (degree == 1) {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C1_FEM<PHX::Device::execution_space,double,double>() );
        }
        else if (degree == 2) {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C2_FEM<PHX::Device::execution_space,double,double>() );
        }
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      if (shape == "Triangle_3") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C1_FEM<PHX::Device::execution_space,double,double>() );
        else if (degree == 2)
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_C2_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_TRI_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_WARPBLEND) );
        }
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        if (degree  == 1)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C1_FEM<PHX::Device::execution_space,double,double>() );
        else if (degree  == 2)
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C2_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      if (shape == "Tetrahedron_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HGRAD_TET_C1_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_TET_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
  }
  else if (type == "HVOL") {
    basis = Teuchos::rcp(new Basis_HVOL_C0_FEM<PHX::Device::execution_space,double,double>(*cellTopo));
  }
  else if (type == "HDIV") {
    if (spaceDim == 1) {
      basis = Teuchos::rcp(new Basis_HGRAD_LINE_C1_FEM<PHX::Device::execution_space,double,double>() );
    }
    else if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HDIV_QUAD_I1_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_QUAD_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Triangle_3") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HDIV_TRI_I1_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_TRI_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    else if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        if (degree  == 1)
          basis = Teuchos::rcp(new Basis_HDIV_HEX_I1_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_HEX_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Tetrahedron_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HDIV_TET_I1_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HDIV_TET_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    
  }
  else if (type == "HDIV_AC") {
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        if (degree == 1) {
          basis = Teuchos::rcp(new Basis_HDIV_AC_QUAD_I1_FEM<PHX::Device::execution_space,double,double>() );
        }
        else {
          TEUCHOS_ASSERT(false); // there is no HDIV_AC higher order implemented yet
        }
      }
      else {
        TEUCHOS_ASSERT(false); // HDIV_AC is only defined on quadrilaterals
      }
    }
    else {
      TEUCHOS_ASSERT(false); // HDIV_AC is only defined in 2D
    }
  }
  else if (type == "HCURL") {
    if (spaceDim == 1) {
      // need to throw an error
    }
    else if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HCURL_QUAD_I1_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_QUAD_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Triangle_3") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HCURL_TRI_I1_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_TRI_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    else if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        if (degree  == 1)
          basis = Teuchos::rcp(new Basis_HCURL_HEX_I1_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_HEX_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      else if (shape == "Tetrahedron_4") {
        if (degree == 1)
          basis = Teuchos::rcp(new Basis_HCURL_TET_I1_FEM<PHX::Device::execution_space,double,double>() );
        else {
          basis = Teuchos::rcp(new Basis_HCURL_TET_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
    }
    
  }
  else if (type == "HFACE") {
    if (spaceDim == 2) {
      if (shape == "Quadrilateral_4") {
        basis = Teuchos::rcp(new Basis_HFACE_QUAD_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
      else if (shape == "Triangle_3") {
        basis = Teuchos::rcp(new Basis_HFACE_TRI_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
    }
    if (spaceDim == 3) {
      if (shape == "Hexahedron_8") {
        basis = Teuchos::rcp(new Basis_HFACE_HEX_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
      if (shape == "Tetrahedron_4") {
        basis = Teuchos::rcp(new Basis_HFACE_TET_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
    }
  }
  
  
  return basis;
  
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

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::setReferenceData(Teuchos::RCP<GroupMetaData> & cellData) {
  
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
        
    if (basis_types[block][i].substr(0,5) == "HGRAD") {
      
      basisvals = DRV("basisvals",numb, cellData->numip);
      basis_pointers[block][i]->getValues(basisvals, cellData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0));
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basisgrad = DRV("basisgrad",numb, cellData->numip, dimension);
      basis_pointers[block][i]->getValues(basisgrad, cellData->ref_ip, Intrepid2::OPERATOR_GRAD);
      
    }
    else if (basis_types[block][i].substr(0,4) == "HVOL") {
      
      basisvals = DRV("basisvals",numb, cellData->numip);
      basis_pointers[block][i]->getValues(basisvals, cellData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0));
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
    }
    else if (basis_types[block][i].substr(0,4) == "HDIV") {
      
      basisvals = DRV("basisvals",numb, cellData->numip, dimension);
      basis_pointers[block][i]->getValues(basisvals, cellData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0), dimension);
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basisdiv = DRV("basisdiv",numb, cellData->numip);
      basis_pointers[block][i]->getValues(basisdiv, cellData->ref_ip, Intrepid2::OPERATOR_DIV);
      
    }
    else if (basis_types[block][i].substr(0,5) == "HCURL"){
      
      basisvals = DRV("basisvals",numb, cellData->numip, dimension);
      basis_pointers[block][i]->getValues(basisvals, cellData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0), dimension);
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      if (dimension == 2) {
        basiscurl = DRV("basiscurl",numb, cellData->numip);
      }
      else if (dimension == 3) {
        basiscurl = DRV("basiscurl",numb, cellData->numip, dimension);
      }
      basis_pointers[block][i]->getValues(basiscurl, cellData->ref_ip, Intrepid2::OPERATOR_CURL);
      
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
      if (basis_types[block][i].substr(0,5) == "HGRAD") {
        basisvals = DRV("basisvals",numb, cellData->numsideip);
        basis_pointers[block][i]->getValues(basisvals, cellData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
        basisgrad = DRV("basisgrad",numb, cellData->numsideip, dimension);
        basis_pointers[block][i]->getValues(basisgrad, cellData->ref_side_ip[s], Intrepid2::OPERATOR_GRAD);
      }
      else if (basis_types[block][i].substr(0,4) == "HVOL" || basis_types[block][i].substr(0,5) == "HFACE") {
        basisvals = DRV("basisvals",numb, cellData->numsideip);
        basis_pointers[block][i]->getValues(basisvals, cellData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
      }
      else if (basis_types[block][i].substr(0,4) == "HDIV") {
        basisvals = DRV("basisvals",numb, cellData->numsideip, dimension);
        basis_pointers[block][i]->getValues(basisvals, cellData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
      }
      else if (basis_types[block][i].substr(0,5) == "HCURL"){
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

void DiscretizationInterface::getPhysicalVolumetricData(Teuchos::RCP<GroupMetaData> & cellData,
                                                        DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                                        vector<View_Sc2> & ip, View_Sc2 wts, View_Sc1 hsize,
                                                        Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                        vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                                        vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                                        vector<View_Sc4> & basis_nodes,
                                                        const bool & recompute_jac,
                                                        const bool & recompute_orient) {
  
  Teuchos::TimeMonitor localtimer(*physVolDataTotalTimer);
  
  int dimension = cellData->dimension;
  int numip = cellData->ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  DRV jacobian, jacobianDet, jacobianInv, tmpip, tmpwts;
  jacobian = DRV("jacobian", numElem, numip, dimension, dimension);
  jacobianDet = DRV("determinant of jacobian", numElem, numip);
  jacobianInv = DRV("inverse of jacobian", numElem, numip, dimension, dimension);
  tmpip = DRV("tmp ip", numElem, numip, dimension);
  tmpwts = DRV("tmp ip wts", numElem, numip);
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataIPTimer);
    CellTools::mapToPhysicalFrame(tmpip, cellData->ref_ip, nodes, *(cellData->cellTopo));
    View_Sc2 x("cell x",tmpip.extent(0), tmpip.extent(1));
    auto tmpip_x = subview(tmpip, ALL(), ALL(),0);
    deep_copy(x,tmpip_x);
    ip.push_back(x);
    if (dimension > 1) {
      View_Sc2 y("cell y",tmpip.extent(0), tmpip.extent(1));
      auto tmpip_y = subview(tmpip, ALL(), ALL(),1);
      deep_copy(y,tmpip_y);
      ip.push_back(y);
    }
    if (dimension > 2) {
      View_Sc2 z("cell z",tmpip.extent(0), tmpip.extent(1));
      auto tmpip_z = subview(tmpip, ALL(), ALL(),2);
      deep_copy(z,tmpip_z);
      ip.push_back(z);
    }
    
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataSetJacTimer);
    CellTools::setJacobian(jacobian, cellData->ref_ip, nodes, *(cellData->cellTopo));
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataOtherJacTimer);
    CellTools::setJacobianDet(jacobianDet, jacobian);
    CellTools::setJacobianInv(jacobianInv, jacobian);
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataWtsTimer);    
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
      ScalarT dimscl = 1.0/(ScalarT)dimension;
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
      
      // These will be redefined below for the appropriate basis types
      View_Sc4 basis_vals("tmp basis",1,1,1,1);
      View_Sc4 basis_grad_vals("tmp grad vals",1,1,1,1);
      View_Sc4 basis_curl_vals("tmp curl vals",1,1,1,1);
      View_Sc4 basis_node_vals("tmp node vals",1,1,1,1);
      View_Sc3 basis_div_vals("tmp div vals",1,1,1);

      if (cellData->basis_types[i].substr(0,5) == "HGRAD"){
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip);
        bvals2 = DRV("basis tmp",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, cellData->ref_basis[i]);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
        
        DRV bgrad1, bgrad2;
        bgrad1 = DRV("basis grad tmp",numElem,numb,numip,dimension);
        bgrad2 = DRV("basis grad",numElem,numb,numip,dimension);
        
        FuncTools::HGRADtransformGRAD(bgrad1, jacobianInv, cellData->ref_basis_grad[i]);
        OrientTools::modifyBasisByOrientation(bgrad2, bgrad1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_grad_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_grad_vals,bgrad2);
        
      }
      else if (cellData->basis_types[i].substr(0,4) == "HVOL"){
        
        DRV bvals1;
        bvals1 = DRV("basis",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, cellData->ref_basis[i]);
        
        basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals1);
      }
      else if (cellData->basis_types[i].substr(0,4) == "HDIV" ) {
        
        {
          Teuchos::TimeMonitor localtimer(*physVolDataBasisDivValTimer);
          DRV bvals1, bvals2;
          bvals1 = DRV("basis",numElem,numb,numip,dimension);
          bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
          
          FuncTools::HDIVtransformVALUE(bvals1, jacobian, jacobianDet, cellData->ref_basis[i]);
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                cellData->basis_pointers[i].get());
          basis_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_vals,bvals2);
        }
        
        if (cellData->requireBasisAtNodes) {
          DRV bnode_vals("basis",numElem,numb,nodes.extent(1),dimension);
          DRV bvals_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
          FuncTools::HDIVtransformVALUE(bvals_tmp, jacobian, jacobianDet, cellData->ref_basis_nodes[i]);
          OrientTools::modifyBasisByOrientation(bnode_vals, bvals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          basis_node_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_node_vals,bnode_vals);
        }
        
        {
          Teuchos::TimeMonitor localtimer(*physVolDataBasisDivDivTimer);
          
          DRV bdiv1, bdiv2;
          bdiv1 = DRV("basis",numElem,numb,numip);
          bdiv2 = DRV("basis tmp",numElem,numb,numip);
          
          FuncTools::HDIVtransformDIV(bdiv1, jacobianDet, cellData->ref_basis_div[i]);
          OrientTools::modifyBasisByOrientation(bdiv2, bdiv1, orientation,
                                                cellData->basis_pointers[i].get());
          basis_div_vals = View_Sc3("basis div values", numElem, numb, numip); // needs to be rank-3
          Kokkos::deep_copy(basis_div_vals,bdiv2);
        }
      }
      else if (cellData->basis_types[i].substr(0,5) == "HCURL"){
        
        {
          Teuchos::TimeMonitor localtimer(*physVolDataBasisCurlValTimer);
          
          DRV bvals1, bvals2;
          bvals1 = DRV("basis",numElem,numb,numip,dimension);
          bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
          
          FuncTools::HCURLtransformVALUE(bvals1, jacobianInv, cellData->ref_basis[i]);
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                cellData->basis_pointers[i].get());
          basis_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_vals,bvals2);
        }
        
        if (cellData->requireBasisAtNodes) {
          DRV bnode_vals("basis",numElem,numb,nodes.extent(1),dimension);
          DRV bvals_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
          FuncTools::HCURLtransformVALUE(bvals_tmp, jacobianInv, cellData->ref_basis_nodes[i]);
          OrientTools::modifyBasisByOrientation(bnode_vals, bvals_tmp, orientation,
                                                cellData->basis_pointers[i].get());
          basis_node_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_node_vals,bnode_vals);
          
        }
        
        {
          Teuchos::TimeMonitor localtimer(*physVolDataBasisCurlCurlTimer);
        
          DRV bcurl1, bcurl2;
          bcurl1 = DRV("basis",numElem,numb,numip,dimension);
          bcurl2 = DRV("basis tmp",numElem,numb,numip,dimension);
          
          FuncTools::HCURLtransformCURL(bcurl1, jacobian, jacobianDet, cellData->ref_basis_curl[i]);
          OrientTools::modifyBasisByOrientation(bcurl2, bcurl1, orientation,
                                                cellData->basis_pointers[i].get());
          basis_curl_vals = View_Sc4("basis curl values", numElem, numb, numip, dimension);
          if (spaceDim == 2) {
            auto sub_bcv = subview(basis_curl_vals,ALL(),ALL(),ALL(),0);
            deep_copy(sub_bcv,bcurl2);
          }
          else {
            Kokkos::deep_copy(basis_curl_vals, bcurl2);
          }
        }
      }
      basis.push_back(basis_vals);
      basis_grad.push_back(basis_grad_vals);
      basis_div.push_back(basis_div_vals);
      basis_curl.push_back(basis_curl_vals);
      basis_nodes.push_back(basis_node_vals);
    }
  }
}

// -------------------------------------------------
// Specialized routine to compute just the basis (not GRAD, CURL or DIV) and the wts
// -------------------------------------------------

void DiscretizationInterface::getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & cellData,
                                                         DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                                         View_Sc2 wts,
                                                         Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                         vector<View_Sc4> & basis,
                                                         const bool & recompute_jac,
                                                         const bool & recompute_orient) {
  
  Teuchos::TimeMonitor localtimer(*physVolDataTotalTimer);
  
  int dimension = cellData->dimension;
  int numip = cellData->ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  DRV jacobian, jacobianDet, jacobianInv, tmpip, tmpwts;
  jacobian = DRV("jacobian", numElem, numip, dimension, dimension);
  jacobianDet = DRV("determinant of jacobian", numElem, numip);
  jacobianInv = DRV("inverse of jacobian", numElem, numip, dimension, dimension);
  tmpip = DRV("tmp ip", numElem, numip, dimension);
  tmpwts = DRV("tmp ip wts", numElem, numip);
    
  {
    Teuchos::TimeMonitor localtimer(*physVolDataSetJacTimer);
    CellTools::setJacobian(jacobian, cellData->ref_ip, nodes, *(cellData->cellTopo));
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataOtherJacTimer);
    CellTools::setJacobianDet(jacobianDet, jacobian);
    CellTools::setJacobianInv(jacobianInv, jacobian);
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physVolDataWtsTimer);
    FuncTools::computeCellMeasure(tmpwts, jacobianDet, cellData->ref_wts);
    Kokkos::deep_copy(wts,tmpwts);
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
      
      // These will be redefined below for the appropriate basis types
      View_Sc4 basis_vals("tmp basis",1,1,1,1);
      
      if (cellData->basis_types[i].substr(0,5) == "HGRAD"){
        DRV bvals1("basis",numElem,numb,numip);
        DRV bvals2("basis tmp",numElem,numb,numip);
        FuncTools::HGRADtransformVALUE(bvals1, cellData->ref_basis[i]);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
        
        
      }
      else if (cellData->basis_types[i].substr(0,4) == "HVOL"){
        
        DRV bvals1("basis",numElem,numb,numip);
        FuncTools::HGRADtransformVALUE(bvals1, cellData->ref_basis[i]);
        
        basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals1);
      }
      else if (cellData->basis_types[i].substr(0,4) == "HDIV" ) {
        
        {
          Teuchos::TimeMonitor localtimer(*physVolDataBasisDivValTimer);
          DRV bvals1("basis",numElem,numb,numip,dimension);
          DRV bvals2("basis tmp",numElem,numb,numip,dimension);
          FuncTools::HDIVtransformVALUE(bvals1, jacobian, jacobianDet, cellData->ref_basis[i]);
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                cellData->basis_pointers[i].get());
          basis_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_vals,bvals2);
        }
        
      }
      else if (cellData->basis_types[i].substr(0,5) == "HCURL"){
        
        {
          Teuchos::TimeMonitor localtimer(*physVolDataBasisCurlValTimer);
          
          DRV bvals1("basis",numElem,numb,numip,dimension);
          DRV bvals2("basis tmp",numElem,numb,numip,dimension);
          FuncTools::HCURLtransformVALUE(bvals1, jacobianInv, cellData->ref_basis[i]);
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                cellData->basis_pointers[i].get());
          basis_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_vals,bvals2);
        }
        
      }
      basis.push_back(basis_vals);
    }
  }
}

// -------------------------------------------------
// Get the element orientations
// -------------------------------------------------

void DiscretizationInterface::getPhysicalOrientations(Teuchos::RCP<GroupMetaData> & cellData,
                                                      Kokkos::View<LO*,AssemblyDevice> eIndex,
                                                      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                      const bool & use_block) {
  
  Teuchos::TimeMonitor localtimer(*physOrientTimer);
  
  auto orientation_host = create_mirror_view(orientation);
  auto host_eIndex = Kokkos::create_mirror_view(eIndex);
  deep_copy(host_eIndex,eIndex);
  for (size_type i=0; i<host_eIndex.extent(0); i++) {
    LO elemID = host_eIndex(i);
    if (use_block) {
      elemID = myElements[cellData->myBlock][host_eIndex(i)];
    }
    orientation_host(i) = panzer_orientations[elemID];
  }
  deep_copy(orientation,orientation_host);
}

// -------------------------------------------------
// Compute the basis functions at the face ip
// -------------------------------------------------

void DiscretizationInterface::getPhysicalFaceData(Teuchos::RCP<GroupMetaData> & cellData, const int & side,
                                                  DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                                  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                  vector<View_Sc2> & face_ip, View_Sc2 face_wts,
                                                  vector<View_Sc2> & face_normals, View_Sc1 face_hsize,
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
  DRV sip, jac, jacDet, jacInv, swts, snormals, tangents;
  sip = DRV("side ip", numElem, numip, dimension);
  jac = DRV("bijac", numElem, numip, dimension, dimension);
  jacDet = DRV("bijacDet", numElem, numip);
  jacInv = DRV("bijacInv", numElem, numip, dimension, dimension);
  swts = DRV("wts_side", numElem, numip);
  snormals = DRV("normals", numElem, numip, dimension);
  tangents = DRV("tangents", numElem, numip, dimension);
  
  {
    Teuchos::TimeMonitor localtimer(*physFaceDataIPTimer);
    CellTools::mapToPhysicalFrame(sip, ref_ip, nodes, *(cellData->cellTopo));
    
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
                 TeamPolicy<AssemblyExec>(snormals.extent(0), Kokkos::AUTO, VectorSize),
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
        ScalarT dimscl = 1.0/((ScalarT)dimension-1.0);
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
      
      // These will be defined below for the appropriate basis types
      View_Sc4 basis_vals("tmp basis vals",1,1,1,1);
      View_Sc4 basis_grad_vals("tmp grad vals",1,1,1,1);
      
      // div and curl values are not currently used on boundaries
      
      auto ref_basis_vals = cellData->ref_side_basis[side][i];
      
      if (cellData->basis_types[i].substr(0,5) == "HGRAD"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip);
        bvals2 = DRV("basis tmp",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
        
        auto ref_basis_grad_vals = cellData->ref_side_basis_grad[side][i];
        DRV bgrad1, bgrad2;
        bgrad1 = DRV("basis",numElem,numb,numip,dimension);
        bgrad2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HGRADtransformGRAD(bgrad1, jacInv, ref_basis_grad_vals);
        OrientTools::modifyBasisByOrientation(bgrad2, bgrad1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_grad_vals = View_Sc4("face basis grad vals",numElem,numb,numip,dimension); // Needs to be rank-4
        Kokkos::deep_copy(basis_grad_vals,bgrad2);
        
      }
      else if (cellData->basis_types[i].substr(0,4) == "HVOL"){
        
        DRV bvals1;
        bvals1 = DRV("basis",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals1);
      }
      else if (cellData->basis_types[i].substr(0,4) == "HDIV" ) {
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip,dimension);
        bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HDIVtransformVALUE(bvals1, jac, jacDet, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_vals,bvals2);
        
      }
      else if (cellData->basis_types[i].substr(0,5) == "HCURL"){
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip,dimension);
        bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HCURLtransformVALUE(bvals1, jacInv, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_vals,bvals2);
        
      }
      else if (cellData->basis_types[i].substr(0,5) == "HFACE"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip);
        bvals2 = DRV("basis tmp",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
        
      }
      
      basis.push_back(basis_vals);
      basis_grad.push_back(basis_grad_vals);
    }
  }
  
}

//======================================================================
//
//======================================================================

void DiscretizationInterface::getPhysicalBoundaryData(Teuchos::RCP<GroupMetaData> & cellData,
                                                      DRV nodes, Kokkos::View<LO*,AssemblyDevice> eIndex,
                                                      Kokkos::View<LO*,AssemblyDevice> localSideID,
                                                      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                      vector<View_Sc2> & ip, View_Sc2 wts,
                                                      vector<View_Sc2> & normals, vector<View_Sc2> & tangents, View_Sc1 hsize,
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
  
  DRV tmpip, ijac, ijacDet, ijacInv, tmpwts, tmpnormals, tmptangents;
  tmpip = DRV("side ip", numElem, numip, dimension);
  ijac = DRV("bijac", numElem, numip, dimension, dimension);
  ijacDet = DRV("bijacDet", numElem, numip);
  ijacInv = DRV("bijacInv", numElem, numip, dimension, dimension);
  tmpwts = DRV("wts_side", numElem, numip);
  tmpnormals = DRV("normals", numElem, numip, dimension);
  tmptangents = DRV("tangents", numElem, numip, dimension);
  
  {
    Teuchos::TimeMonitor localtimer(*physBndryDataIPTimer);
    CellTools::mapToPhysicalFrame(tmpip, ref_ip, nodes, *(cellData->cellTopo));
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
      ScalarT dimscl = 1.0/((ScalarT)dimension-1.0);
      hsize(e) = pow(vol,dimscl);
    });
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
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
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
  
  // -------------------------------------------------
  // Compute the element orientations
  // -------------------------------------------------
  
  if (recompute_orient) {
    this->getPhysicalOrientations(cellData, eIndex, orientation, false);
  }
  
  {
    Teuchos::TimeMonitor localtimer(*physBndryDataBasisTimer);
    
    for (size_t i=0; i<cellData->basis_pointers.size(); i++) {
      
      int numb = cellData->basis_pointers[i]->getCardinality();
      
      // These will be redefined below for the appropriate basis type
      View_Sc4 basis_vals("tmp basis vals",1,1,1,1);
      View_Sc4 basis_grad_vals("tmp grad vals",1,1,1,1);
      View_Sc4 basis_curl_vals("tmp curl vals",1,1,1,1);
      View_Sc3 basis_div_vals("tmp div vals",1,1,1);
      
      DRV ref_basis_vals = cellData->ref_side_basis[localSideID_host(0)][i];
      
      if (cellData->basis_types[i].substr(0,5) == "HGRAD"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip);
        bvals2 = DRV("basis tmp",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,1);
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
        
        DRV bgrad1, bgrad2;
        bgrad1 = DRV("basis",numElem,numb,numip,dimension);
        bgrad2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        DRV ref_bgrad_vals = cellData->ref_side_basis_grad[localSideID_host(0)][i];
        FuncTools::HGRADtransformGRAD(bgrad1, ijacInv, ref_bgrad_vals);
        
        OrientTools::modifyBasisByOrientation(bgrad2, bgrad1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_grad_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_grad_vals,bgrad2);
        
      }
      else if (cellData->basis_types[i].substr(0,4) == "HVOL"){ // does not require orientations
        
        DRV bvals1;
        bvals1 = DRV("basis",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,1);
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals1);
      }
      else if (cellData->basis_types[i].substr(0,5) == "HFACE"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip);
        bvals2 = DRV("basis tmp",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,1);
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
      }
      else if (cellData->basis_types[i].substr(0,4) == "HDIV"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip,dimension);
        bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HDIVtransformVALUE(bvals1, ijac, ijacDet, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_vals,bvals2);
      }
      else if (cellData->basis_types[i].substr(0,5) == "HCURL"){
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip,dimension);
        bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HCURLtransformVALUE(bvals1, ijacInv, ref_basis_vals);
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              cellData->basis_pointers[i].get());
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_vals,bvals2);
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

DRV DiscretizationInterface::evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts) {
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

DRV DiscretizationInterface::evaluateBasis(const basis_RCP & basis_pointer, const DRV & evalpts,
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

DRV DiscretizationInterface::evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
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

DRV DiscretizationInterface::evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
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

void DiscretizationInterface::buildDOFManagers() {
  
  Teuchos::TimeMonitor localtimer(*dofmgrtimer);
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics::buildDOF ..." << endl;
    }
  }
  
  Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
  
  // DOF manager for the primary variables
  for (size_t set=0; set<phys->setnames.size(); ++set) {
    Teuchos::RCP<panzer::DOFManager> setDOF = Teuchos::rcp(new panzer::DOFManager());
    setDOF->setConnManager(conn,*(Commptr->getRawMpiComm()));
    setDOF->setOrientationsRequired(true);
    
    for (size_t block=0; block<blocknames.size(); ++block) {
      for (size_t j=0; j<phys->varlist[set][block].size(); j++) {
        topo_RCP cellTopo = mesh->getCellTopology(blocknames[block]);
        basis_RCP basis_pointer = this->getBasis(spaceDim, cellTopo,
                                                 phys->types[set][block][j],
                                                 phys->orders[set][block][j]);
        
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_pointer));
        
        if (phys->useDG[set][block][j]) {
          setDOF->addField(blocknames[block], phys->varlist[set][block][j], Pattern, panzer::FieldType::DG);
        }
        else {
          setDOF->addField(blocknames[block], phys->varlist[set][block][j], Pattern, panzer::FieldType::CG);
        }
        
      }
    }
    
    setDOF->buildGlobalUnknowns();
#ifndef MrHyDE_NO_AD
    for (size_t block=0; block<blocknames.size(); ++block) {
      int numGIDs = setDOF->getElementBlockGIDCount(blocknames[block]);
      TEUCHOS_TEST_FOR_EXCEPTION(numGIDs > maxDerivs,std::runtime_error,"Error: maxDerivs is not large enough to support the number of degrees of freedom per element on block: " + blocknames[block]);
    }
#endif
    if (verbosity>1) {
      if (Commptr->getRank() == 0) {
        setDOF->printFieldInformation(std::cout);
      }
    }
    DOF.push_back(setDOF);
  }

  // Create the vector of panzer orientations
  // Using the panzer orientation interface works, except when also
  // using an MPI subcommunicator, e.g., in the subgrid models
  // Leaving here for testing purposes

  //auto pOInt = panzer::OrientationsInterface(DOF[0]);
  //auto pO_orients = pOInt.getOrientations();
  //panzer_orientations = *pO_orients;
  
  {
    auto oconn = conn->noConnectivityClone();
    
    shards::CellTopology topology;
    std::vector<shards::CellTopology> elementBlockTopologies;
    oconn->getElementBlockTopologies(elementBlockTopologies);

    topology = elementBlockTopologies.at(0);
  
    const int num_nodes_per_cell = topology.getVertexCount();

    size_t totalElem = 0;
    for (size_t block=0; block<blocknames.size(); ++block) {
      totalElem += myElements[block].size();
    }

    // Make sure the conn is setup for a nodal connectivity
    panzer::NodalFieldPattern pattern(topology);
    oconn->buildConnectivity(pattern);

    // Initialize the orientations vector
    panzer_orientations.clear();
    panzer_orientations.resize(totalElem);
  
    using NodeView = Kokkos::View<GO*, Kokkos::DefaultHostExecutionSpace>;
    
    // Add owned orientations
    {
      for (size_t block=0; block<blocknames.size(); ++block) {
        for (size_t c=0; c<myElements[block].size(); ++c) {
          size_t elemID = myElements[block][c];
          const GO * nodes = oconn->getConnectivity(elemID);
          NodeView node_view("nodes",num_nodes_per_cell);
          for (int node=0; node<num_nodes_per_cell; ++node) {
            node_view(node) = nodes[node];
          }
          panzer_orientations[elemID] = Intrepid2::Orientation::getOrientation(topology, node_view);
          
        }
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

void DiscretizationInterface::setBCData() {
  
  Teuchos::TimeMonitor localtimer(*setbctimer);
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting DiscretizationInterface::setBCData ..." << endl;
    }
  }
  
  bool requires_sideinfo = false;
  if (settings->isSublist("Subgrid")) {
    requires_sideinfo = true;
  }
  
  vector<string> sideSets, nodeSets;
  mesh->getSidesetNames(sideSets);
  mesh->getNodesetNames(nodeSets);
  
  for (size_t set=0; set<phys->setnames.size(); ++set) {
    vector<vector<string> > varlist = phys->varlist[set];
    auto currDOF = DOF[set];
    
    int maxvars = 0;
    for (size_t block=0; block<blocknames.size(); ++block) {
      for (size_t j=0; j<varlist[block].size(); j++) {
        string var = varlist[block][j];
        int num = currDOF->getFieldNum(var);
        maxvars = std::max(num,maxvars);
      }
    }
  
    vector<Kokkos::View<int****,HostDevice> > set_side_info;
    vector<vector<vector<string> > > set_var_bcs; // [block][var][boundary]
    vector<vector<vector<int> > > set_offsets; // [block][var][dof]
    
    vector<vector<GO> > set_point_dofs;
    vector<vector<vector<LO> > > set_dbc_dofs;
    
    for (size_t block=0; block<blocknames.size(); ++block) {
      
      vector<vector<string> > block_var_bcs; // [var][boundary]
      
      topo_RCP cellTopo = mesh->getCellTopology(blocknames[block]);
      int numSidesPerElem = 2; // default to 1D for some reason
      if (spaceDim == 2) {
        numSidesPerElem = cellTopo->getEdgeCount();
      }
      else if (spaceDim == 3) {
        numSidesPerElem = cellTopo->getFaceCount();
      }
      
      std::string blockID = blocknames[block];
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

      Teuchos::ParameterList blocksettings = phys->setPhysSettings[set][block];
    
      Teuchos::ParameterList dbc_settings = blocksettings.sublist("Dirichlet conditions");
      Teuchos::ParameterList nbc_settings = blocksettings.sublist("Neumann conditions");
      Teuchos::ParameterList fbc_settings = blocksettings.sublist("Far-field conditions");
      Teuchos::ParameterList sbc_settings = blocksettings.sublist("Slip conditions");
      Teuchos::ParameterList flux_settings = blocksettings.sublist("Flux conditions");
      bool use_weak_dbcs = dbc_settings.get<bool>("use weak Dirichlet",false);
      
      vector<vector<int> > celloffsets;
      Kokkos::View<int****,HostDevice> currside_info;
      if (requires_sideinfo) {
        currside_info = Kokkos::View<int****,HostDevice>("side info",stk_meshElems.size(),
                                                         varlist[block].size(),numSidesPerElem,2);
      }
      else {
        currside_info = Kokkos::View<int****,HostDevice>("side info",1,1,1,2);
      }

      std::vector<int> block_dbc_dofs;
      
      std::string perBCs = settings->sublist("Mesh").get<string>("Periodic Boundaries","");

      for (size_t j=0; j<varlist[block].size(); j++) {
        vector<string> current_var_bcs(sideSets.size(),"none"); // [boundary]
        string var = varlist[block][j];
        int num = currDOF->getFieldNum(var);
        vector<int> var_offsets = currDOF->getGIDFieldOffsets(blockID,num);

        celloffsets.push_back(var_offsets);
      
        for (size_t side=0; side<sideSets.size(); side++ ) {
          string sideName = sideSets[side];
          
          vector<stk::mesh::Entity> sideEntities;
          mesh->getMySides(sideName, blockID, sideEntities);
          
          bool isDiri = false;
          bool isNeum = false;
          bool isFar  = false;
          bool isSlip = false;
          bool isFlux = false;

          if (dbc_settings.sublist(var).isParameter("all boundaries") || dbc_settings.sublist(var).isParameter(sideName)) {
            isDiri = true;
            if (use_weak_dbcs) {
              current_var_bcs[side] = "weak Dirichlet";
            }
            else {
              current_var_bcs[side] = "Dirichlet";
            }
          }
          if (nbc_settings.sublist(var).isParameter("all boundaries") || nbc_settings.sublist(var).isParameter(sideName)) {
            isNeum = true;
            current_var_bcs[side] = "Neumann";
          }
          if (fbc_settings.sublist(var).isParameter("all boundaries") || fbc_settings.sublist(var).isParameter(sideName)) {
            isFar = true;
            current_var_bcs[side] = "Far-field";
          }
          if (sbc_settings.sublist(var).isParameter("all boundaries") || sbc_settings.sublist(var).isParameter(sideName)) {
            isSlip = true;
            current_var_bcs[side] = "Slip";
          }
          if (flux_settings.sublist(var).isParameter("all boundaries") || flux_settings.sublist(var).isParameter(sideName)) {
            isFlux = true;
            current_var_bcs[side] = "Flux";
          }

          if (requires_sideinfo) {
            vector<size_t>             local_side_Ids;
            vector<stk::mesh::Entity> side_output;
            vector<size_t>             local_elem_Ids;
            panzer_stk::workset_utils::getSideElements(*mesh, blockID, sideEntities, local_side_Ids, side_output);
            
            for (size_t i=0; i<side_output.size(); i++ ) {
              local_elem_Ids.push_back(mesh->elementLocalId(side_output[i]));
              size_t localid = localelemmap[local_elem_Ids[i]];
              if (isDiri) {
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
              else if (isFar) { // Far-field
                currside_info(localid, j, local_side_Ids[i], 0) = 6;
                currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
              }
              else if (isSlip) { // Slip
                currside_info(localid, j, local_side_Ids[i], 0) = 7;
                currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
              }
              else if (isFlux) { // Flux
                currside_info(localid, j, local_side_Ids[i], 0) = 8;
                currside_info(localid, j, local_side_Ids[i], 1) = (int)side;
              }
            }
          }
        }
      
        block_var_bcs.push_back(current_var_bcs);
        
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
    
      set_offsets.push_back(celloffsets);
      set_var_bcs.push_back(block_var_bcs);
      set_side_info.push_back(currside_info);
      
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
      
      set_point_dofs.push_back(dbc_final);
      
    } // blocks
      
    
    offsets.push_back(set_offsets);
    var_bcs.push_back(set_var_bcs);
    side_info.push_back(set_side_info);
    point_dofs.push_back(set_point_dofs);
    
  } // sets
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished DiscretizationInterface::setBCData" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::setDirichletData() {
  
  Teuchos::TimeMonitor localtimer(*setdbctimer);
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting DiscretizationInterface::setDirichletData ..." << endl;
    }
  }
  
  vector<string> sideNames;
  mesh->getSidesetNames(sideNames);
  
  for (size_t set=0; set<phys->setnames.size(); ++set) {
    
    vector<vector<string> > varlist = phys->varlist[set];
    auto currDOF = DOF[set];
    
    std::vector<std::vector<std::vector<LO> > > set_dbc_dofs;
    
    for (size_t block=0; block<blocknames.size(); ++block) {
      
      std::string blockID = blocknames[block];
      
      Teuchos::ParameterList dbc_settings = phys->setPhysSettings[set][block].sublist("Dirichlet conditions");
      
      std::vector<std::vector<LO> > block_dbc_dofs;
      
      for (size_t j=0; j<varlist[block].size(); j++) {
        std::string var = varlist[block][j];
        int fieldnum = currDOF->getFieldNum(var);
        std::vector<LO> var_dofs;
        for (size_t side=0; side<sideNames.size(); side++ ) {
          std::string sideName = sideNames[side];
          vector<stk::mesh::Entity> sideEntities;
          mesh->getMySides(sideName, blockID, sideEntities);
          
          bool isDiri = false;
          if (dbc_settings.sublist(var).isParameter("all boundaries") || dbc_settings.sublist(var).isParameter(sideName)) {
            isDiri = true;
            haveDirichlet = true;
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
      set_dbc_dofs.push_back(block_dbc_dofs);
      
    }
    
    dbc_dofs.push_back(set_dbc_dofs);
    
  }
  
  if (debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished DiscretizationInterface::setDirichletData" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<int****,HostDevice> DiscretizationInterface::getSideInfo(const size_t & set, const size_t & block,
                                                                      Kokkos::View<int*,HostDevice> elem) {
  
  Kokkos::View<int****,HostDevice> currsi;
  
  int maxe = 0;
  for (size_type e=0; e<elem.extent(0); ++e) {
    maxe = std::max(elem(e),maxe);
  }
  if (maxe < (int)side_info[set][block].extent(0)) {
    size_type nelem = elem.extent(0);
    size_type nvars = side_info[set][block].extent(1);
    size_type nelemsides = side_info[set][block].extent(2);
    currsi = Kokkos::View<int****,HostDevice>("side info for cell",nelem,nvars,nelemsides, 2);
    for (size_type e=0; e<nelem; e++) {
      for (size_type j=0; j<nelemsides; j++) {
        for (size_type i=0; i<nvars; i++) {
          int sidetype = side_info[set][block](elem(e),i,j,0);
          if (sidetype > 0) {
            currsi(e,i,j,0) = sidetype;
            currsi(e,i,j,1) = side_info[set][block](elem(e),i,j,1);
          }
          else {
            currsi(e,i,j,0) = sidetype;
            currsi(e,i,j,1) = 0;
          }
        }
      }
    }
  }
  return currsi;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

vector<vector<int> > DiscretizationInterface::getOffsets(const int & set, const int & block) {
  return offsets[set][block];
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::mapPointsToReference(DRV phys_pts, DRV nodes, topo_RCP & cellTopo) {
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

DRV DiscretizationInterface::mapPointsToPhysical(DRV ref_pts, DRV nodes, topo_RCP & cellTopo) {
  DRV phys_pts("reference cell points",nodes.extent(0), ref_pts.extent(0), ref_pts.extent(1));
  CellTools::mapToPhysicalFrame(phys_pts, ref_pts, nodes, *cellTopo);
  return phys_pts;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::DynRankView<int,PHX::Device> DiscretizationInterface::checkInclusionPhysicalData(DRV phys_pts, DRV nodes,
                                                                                         topo_RCP & cellTopo,
                                                                                         const ScalarT & tol) {
  DRV ref_pts = this->mapPointsToReference(phys_pts,nodes,cellTopo);
  
  Kokkos::DynRankView<int,PHX::Device> inRefCell("inRefCell", 1, phys_pts.extent(1));
  
  CellTools::checkPointwiseInclusion(inRefCell, ref_pts, *cellTopo, tol);
  
  return inRefCell;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::applyOrientation(DRV basis, Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                              basis_RCP & basis_pointer) {
  
  DRV new_basis("basis values", basis.extent(0), basis.extent(1), basis.extent(2));
  OrientTools::modifyBasisByOrientation(new_basis, basis, orientation, basis_pointer.get());
  return new_basis;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<string**,HostDevice> DiscretizationInterface::getVarBCs(const size_t & set, const size_t & block) {
  
  vector<string> sideSets;
  mesh->getSidesetNames(sideSets);
  size_t numvars = var_bcs[set][block].size();
  Kokkos::View<string**,HostDevice> bcs("BCs for each variable",numvars, sideSets.size());
  for (size_t var=0; var<numvars; ++var) {
    for (size_t side=0; side<sideSets.size(); ++side) {
      bcs(var,side) = var_bcs[set][block][var][side];
    }
  }
  return bcs;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we can get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::purgeMemory() {
  
  DOF.clear();
  side_info.clear();
  panzer_orientations.clear();

}
