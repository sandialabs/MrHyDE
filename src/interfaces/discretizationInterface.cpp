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
//#include "Intrepid2_FunctionSpaceTools.hpp"
//#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Utils.hpp"

#include "Panzer_STKConnManager.hpp"

using namespace MrHyDE;

discretization::discretization(Teuchos::RCP<Teuchos::ParameterList> & settings,
                               Teuchos::RCP<MpiComm> & Comm_,
                               Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                               vector<vector<int> > & orders, vector<vector<string> > & types) :
Commptr(Comm_), mesh(mesh_) {
  
  milo_debug_level = settings->get<int>("debug level",0);
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting discretization constructor..." << endl;
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Collect some information
  ////////////////////////////////////////////////////////////////////////////////
  
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  std::vector<string> blocknames;
  mesh->getElementBlockNames(blocknames);
  
  string shape = settings->sublist("Mesh").get<string>("shape","quad");
  
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
    vector<int> blockcards;
    vector<basis_RCP> blockbasis;//(blockmaxorder);
    
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
    for (int i=0; i<orders[b].size(); i++) {
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
      //sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Node<> >() ));
    }
    if (spaceDim == 2) {
      if (shape == "quad") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
      if (shape == "tri") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
    }
    if (spaceDim == 3) {
      if (shape == "hex") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ));
      }
      if (shape == "tet") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() ));
      }
    }
    
    DRV side_qpts, side_qwts;
    int side_quadorder = db_settings.sublist(blockID).get<int>("side quadrature",2*mxorder);
    this->getQuadrature(sideTopo, side_quadorder, side_qpts, side_qwts);
    
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
  
  if (milo_debug_level > 0) {
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
        basis = Teuchos::rcp(new Basis_HFACE_TET_In_FEM<AssemblyExec>(degree,POINTTYPE_EQUISPACED) );
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
  Teuchos::RCP<Intrepid2::Cubature<AssemblyExec> > basisCub  = cubFactory.create<AssemblyExec, ScalarT, ScalarT>(*cellTopo, order); // TMW: the mesh sublist is not the correct place
  int cubDim  = basisCub->getDimension();
  int numCubPoints = basisCub->getNumPoints();
  ip = DRV("ip", numCubPoints, cubDim);
  wts = DRV("wts", numCubPoints);
  basisCub->getCubature(ip, wts);
  
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

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

DRV discretization::evaluateBasisGrads(const basis_RCP & basis_pointer, const DRV & nodes,
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

/////////////////////////////////////////////////////////////////////////////////////////////
// After the mesh and the discretizations have been defined, we can create and add the physics
// to the DOF manager
/////////////////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<panzer::DOFManager> discretization::buildDOF(Teuchos::RCP<panzer_stk::STK_Interface> & mesh,
                                                          vector<vector<string> > & varlist,
                                                          vector<vector<string> > & types,
                                                          vector<vector<int> > & orders,
                                                          vector<vector<bool> > & useDG) {
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics::buildDOF ..." << endl;
    }
  }
  
  Teuchos::RCP<panzer::DOFManager> DOF = Teuchos::rcp(new panzer::DOFManager());
  Teuchos::RCP<panzer::ConnManager> conn = Teuchos::rcp(new panzer_stk::STKConnManager(mesh));
  DOF->setConnManager(conn,*(Commptr->getRawMpiComm()));
  DOF->setOrientationsRequired(true);
  
  std::vector<string> blocknames;
  mesh->getElementBlockNames(blocknames);
  
  for (size_t b=0; b<blocknames.size(); b++) {
    for (size_t j=0; j<varlist[b].size(); j++) {
      topo_RCP cellTopo = mesh->getCellTopology(blocknames[b]);
      basis_RCP basis_pointer = this->getBasis(spaceDim, cellTopo, types[b][j], orders[b][j]);
      
      Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_pointer));
      
      if (useDG[b][j]) {
        DOF->addField(blocknames[b], varlist[b][j], Pattern, panzer::FieldType::DG);
      }
      else {
        DOF->addField(blocknames[b], varlist[b][j], Pattern, panzer::FieldType::CG);
      }
    }
  }
  
  DOF->buildGlobalUnknowns();
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics::buildDOF" << endl;
    }
  }
  
  return DOF;
  
}
