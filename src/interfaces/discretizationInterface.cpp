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
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SetupUtilities.hpp"

using namespace MrHyDE;

discretization::discretization(Teuchos::RCP<Teuchos::ParameterList> & settings_,
                               Teuchos::RCP<MpiComm> & Comm_,
                               Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                               Teuchos::RCP<physics> & phys_) :
settings(settings_), Commptr(Comm_), mesh(mesh_), phys(phys_) {
  
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
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting physics::buildDOF ..." << endl;
    }
  }
  
  std::vector<string> blocknames;
  mesh->getElementBlockNames(blocknames);
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
  }
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished physics::buildDOF" << endl;
    }
  }
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void discretization::setBCData(const bool & isaux) {
  
  Teuchos::TimeMonitor localtimer(*setbctimer);
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting discretization::setBCData ..." << endl;
    }
  }
  
  vector<string> blocknames, sideSets, nodeSets;
  mesh->getElementBlockNames(blocknames);
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
    
    Kokkos::View<int**,HostDevice> currbcs("boundary conditions",
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
            currbcs(j,side) = 4;
          }
          else {
            currbcs(j,side) = 1;
          }
        }
        if (nbc_settings.sublist(var).isParameter("all boundaries") || nbc_settings.sublist(var).isParameter(sideName)) {
          isNeum = true;
          currbcs(j,side) = 2;
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
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished discretization::setBCData" << endl;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void discretization::setDirichletData(const bool & isaux) {
  
  Teuchos::TimeMonitor localtimer(*setdbctimer);
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting discretization::setDirichletData ..." << endl;
    }
  }
  
  vector<string> blocknames, sideNames;
  mesh->getElementBlockNames(blocknames);
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
  
  if (milo_debug_level > 0) {
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
