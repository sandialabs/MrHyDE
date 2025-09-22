/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "discretizationInterface.hpp"
#include "Panzer_NodalFieldPattern.hpp"
#include "Panzer_OrientationsInterface.hpp"

// HGRAD basis functions
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"

// HVOL basis functions
#include "Intrepid2_HVOL_C0_FEM.hpp"

// HDIV basis functions
#include "Intrepid2_HDIV_QUAD_In_FEM.hpp"
#include "Intrepid2_HDIV_HEX_In_FEM.hpp"
#include "Intrepid2_HDIV_TRI_In_FEM.hpp"
#include "Intrepid2_HDIV_TET_In_FEM.hpp"

// HDIV Arbogast-Correa basis functions
#include "Intrepid2_HDIV_AC_QUAD_I1_FEM.hpp"

// HCURL basis functions
#include "Intrepid2_HCURL_QUAD_In_FEM.hpp"
#include "Intrepid2_HCURL_HEX_In_FEM.hpp"
#include "Intrepid2_HCURL_TRI_In_FEM.hpp"
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
                                                 Teuchos::RCP<MeshInterface> & mesh_,
                                                 Teuchos::RCP<PhysicsInterface> & physics_) :
settings(settings_), comm(Comm_), mesh(mesh_), physics(physics_) {
  
  RCP<Teuchos::Time> constructor_time = Teuchos::TimeMonitor::getNewCounter("MrHyDE::DiscretizationInterface - constructor");
  Teuchos::TimeMonitor constructor_timer(*constructor_time);
    
  debugger = Teuchos::rcp(new MrHyDE_Debugger(settings->get<int>("debug level",0), comm));
  
  verbosity = settings->get<int>("verbosity",0);
  minimize_memory = settings->sublist("Solver").get<bool>("minimize memory",false);
  
  debugger->print("**** Starting DiscretizationInterface constructor...");
  
  ////////////////////////////////////////////////////////////////////////////////
  // Collect some information
  ////////////////////////////////////////////////////////////////////////////////
  
  dimension = mesh->getDimension();
  block_names = mesh->getBlockNames();
  side_names = mesh->getSideNames();

  ////////////////////////////////////////////////////////////////////////////////
  // Assemble the information we always store
  ////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<int> > orders = physics->unique_orders;
  vector<vector<string> > types = physics->unique_types;
  
  for (size_t block=0; block<block_names.size(); ++block) {
    
    string blockID = block_names[block];
    topo_RCP cellTopo = mesh->getCellTopology(blockID);
    string shape = cellTopo->getName();
    
    if (mesh->use_stk_mesh) {
      vector<stk::mesh::Entity> stk_meshElems = mesh->getMySTKElements(blockID);
      
      // list of all elements on this processor
      Kokkos::View<LO*,HostDevice> blockmy_elements("list of elements",stk_meshElems.size());
      for( size_t e=0; e<stk_meshElems.size(); e++ ) {
        blockmy_elements(e) = mesh->getSTKElementLocalId(stk_meshElems[e]);
      }
      my_elements.push_back(blockmy_elements);
    } else {
      Kokkos::View<LO*,HostDevice> blockmy_elements("list of elements",mesh->simple_mesh->getNumCells());
      for(unsigned int i=0; i<blockmy_elements.size(); ++i)
        blockmy_elements(i) = i;
      my_elements.push_back(blockmy_elements);
      //cout << blockmy_elements.size() << endl;
      
    }
    
    vector<int> blockcards;
    vector<basis_RCP> blockbasis;
    
    vector<int> doneorders;
    vector<string> donetypes;
    
    for (size_t set=0; set<physics->set_names.size(); ++set) {
      Teuchos::ParameterList db_settings = physics->disc_settings[set][block];
      
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
          basis_RCP basis = this->getBasis(dimension, cellTopo, types[block][n], orders[block][n]);
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
    quadorder = physics->disc_settings[0][block].get<int>("quadrature",2*mxorder); // hard coded
    this->getQuadrature(cellTopo, quadorder, qpts, qwts);
    
    ///////////////////////////////////////////////////////////////////////////
    // Side Quadrature
    ///////////////////////////////////////////////////////////////////////////
    
    topo_RCP sideTopo;
    
    if (dimension == 1) {
      sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Node >() ));
    }
    if (dimension == 2) {
      if (shape == "Quadrilateral_4") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
      if (shape == "Triangle_3") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
    }
    if (dimension == 3) {
      if (shape == "Hexahedron_8") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ));
      }
      if (shape == "Tetrahedron_4") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() ));
      }
    }
    
    DRV side_qpts, side_qwts;
    if (dimension == 1) {
      side_qpts = DRV("side qpts",1,1);
      Kokkos::deep_copy(side_qpts,-1.0);
      side_qwts = DRV("side wts",1,1);
      Kokkos::deep_copy(side_qwts,1.0);
    }
    else {
      int side_quadorder = physics->disc_settings[0][block].get<int>("side quadrature",2*mxorder); // hard coded
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
  
  // We do not actually store the DOF or Connectivity managers
  // Probably require:
  // std::vector<Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device>> dof_lids; [set](elem, dof)
  // std::vector<std::vector<GO> > dof_owned, dof_owned_and_shared; // list of degrees of freedom on processor
  // std::vector<std::vector<std::vector<GO>>> dof_gids; // [set][elem][dof] 
  // vector<vector<vector<vector<int> > > > offsets; // [set][block][var][dof]

  // May also need to fill:
  // std::vector<Intrepid2::Orientation> panzer_orientations; [elem]
  // vector<int> num_derivs_required; [block] (takes max over sets)
     
  if (mesh->use_stk_mesh) {
    this->buildDOFManagers();
  }
  else {
    // GHDR: need to fill in the objects listed above (try it without the orientations and num_derivs_required)

    // GH: this simply pushes back DOFs 0,1,...,N-1 where N is the number of nodes for owned and ownedAndShared
    //vector<GO> owned;
    //for(unsigned int i=0; i < (unsigned int) mesh->simple_mesh->getNumNodes(); ++i)
    //  owned.push_back(((GO) i));
    size_t num_owned = 0;
    for (unsigned int i=0; i < (unsigned int) mesh->simple_mesh->getNumNodes(); ++i) {
      bool isshared = mesh->simple_mesh->isShared(i);
      if (!isshared) {
        num_owned++;
      }
    }
    
    Kokkos::View<GO*,HostDevice> owned("owned dofs",num_owned);
    size_t prog = 0;
    for (unsigned int i=0; i < (unsigned int) mesh->simple_mesh->getNumNodes(); ++i) {
      bool isshared = mesh->simple_mesh->isShared(i);
      if (!isshared) {
        owned(prog) = mesh->simple_mesh->localToGlobal(i);
        ++prog;
      }
    }
    dof_owned.push_back(owned);
    
    //for (size_type i=0; i<owned.extent(0); ++i) {
    //  cout << comm->getRank() << "  " << owned(i) << endl;
    //}
    
    Kokkos::View<GO*,HostDevice> owned_shared("owned and shared dofs",mesh->simple_mesh->getNumNodes());
    for (unsigned int i=0; i < (unsigned int) mesh->simple_mesh->getNumNodes(); ++i) {
      owned_shared(i) = mesh->simple_mesh->localToGlobal(i);
    }
    
    //for (size_type i=0; i<owned_shared.extent(0); ++i) {
    //  cout << comm->getRank() << "  " << owned_shared(i) << endl;
    //}
    
    dof_owned_and_shared.push_back(owned_shared);

    dof_lids.push_back(mesh->simple_mesh->getCellToNodeMap()); // [set](elem, dof)
    
    /*
    //std::vector<std::vector<std::vector<GO>>> dof_gids; // [set][elem][dof] 
    Kokkos::View<GO**,HostDevice> elemids("dof gids", dof_lids[dof_lids.size()-1].extent(0), dof_lids[dof_lids.size()-1].extent(1));
    for(unsigned int e=0; e<dof_lids[0].extent(0); ++e) {
      std::vector<GO> localelemids;
      for(unsigned int i=0; i<dof_lids[0].extent(1); ++i) {
        //localelemids.push_back(dof_lids[0](e,i));
        elemids(e,i) = dof_lids[0](e,i);
      }
      //elemids.push_back(localelemids);
    }
    dof_gids.push_back(elemids);
    */

    // vector<vector<vector<vector<int> > > > offsets; // [set][block][var][dof]
    for (size_t set=0; set<physics->set_names.size(); ++set) {
      vector<vector<string> > varlist = physics->var_list[set];
      vector<vector<vector<int> > > set_offsets; // [block][var][dof]
      for (size_t block=0; block<block_names.size(); ++block) {
        vector<vector<int> > celloffsets;
        for (size_t j=0; j<varlist[block].size(); j++) {
          string var = varlist[block][j];
          //int num = setDOF->getFieldNum(var);
          vector<int> var_offsets = {0, 1, 3, 2}; // GH: super hacky???

          celloffsets.push_back(var_offsets);
        }
        set_offsets.push_back(celloffsets);
      }
      offsets.push_back(set_offsets);

      // more hacky stuff; can't set dbcs without dof manager, but we don't have a dof manager
      std::vector<std::vector<std::vector<LO> > > set_dbc_dofs;
      std::vector<std::vector<LO> > block_dbc_dofs;
      std::vector<LO> var_dofs;
      //var_dofs.push_back(0);
      block_dbc_dofs.push_back(var_dofs);
      set_dbc_dofs.push_back(block_dbc_dofs);
      dbc_dofs.push_back(set_dbc_dofs);

      // parameter manager wants num_derivs_required
      num_derivs_required = std::vector<int>(1);

    }
    
    //panzer_orientations = Kokkos::View<Intrepid2::Orientation*,HostDevice>("panzer orient",mesh->simple_mesh->getNumCells());
    panzer_orientations = Kokkos::View<Intrepid2::Orientation*,HostDevice>("panzer orient",1);

  }
  
  //for (size_type i=0; i<dof_lids[0].extent(0); ++i) {
  //  cout << i << "  ";
  //  for (size_type j=0; j<dof_lids[0].extent(1); ++j) {
  //    cout << dof_lids[0](i,j) << " ";
  //  }
  //  cout << endl;
  //}
  
  debugger->print("**** Finished DiscretizationInterface constructor");
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Create a pointer to an Intrepid or Panzer basis
// Note that these always use double rather than ScalarT
//////////////////////////////////////////////////////////////////////////////////////

basis_RCP DiscretizationInterface::getBasis(const int & dimension, const topo_RCP & cellTopo,
                                            const string & type, const int & degree) {
  using namespace Intrepid2;
  
  Teuchos::RCP<Intrepid2::Basis<PHX::Device::execution_space, double, double > > basis;
  
  string shape = cellTopo->getName();
  
  if (type == "HGRAD") {
    if (dimension == 1) {
      basis = Teuchos::rcp(new Basis_HGRAD_LINE_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
    }
    if (dimension == 2) {
      if (shape == "Quadrilateral_4") {
        if (degree == 1) {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_C1_FEM<PHX::Device::execution_space,double,double>());
        }
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_QUAD_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
      }
      if (shape == "Triangle_3") {
        basis = Teuchos::rcp(new Basis_HGRAD_TRI_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_WARPBLEND) );
      }
    }
    if (dimension == 3) {
      if (shape == "Hexahedron_8") {
        if (degree == 1) {
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_C1_FEM<PHX::Device::execution_space,double,double>() );
        }
        else {
          basis = Teuchos::rcp(new Basis_HGRAD_HEX_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
        }
        
      }
      if (shape == "Tetrahedron_4") {
        basis = Teuchos::rcp(new Basis_HGRAD_TET_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
    }
  }
  else if (type == "HVOL") {
    basis = Teuchos::rcp(new Basis_HVOL_C0_FEM<PHX::Device::execution_space,double,double>(*cellTopo));
  }
  else if (type == "HDIV") {
    if (dimension == 1) {
      basis = Teuchos::rcp(new Basis_HGRAD_LINE_Cn_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
    }
    else if (dimension == 2) {
      if (shape == "Quadrilateral_4") {
        basis = Teuchos::rcp(new Basis_HDIV_QUAD_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
      else if (shape == "Triangle_3") {
        basis = Teuchos::rcp(new Basis_HDIV_TRI_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
    }
    else if (dimension == 3) {
      if (shape == "Hexahedron_8") {
        basis = Teuchos::rcp(new Basis_HDIV_HEX_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
      else if (shape == "Tetrahedron_4") {
        basis = Teuchos::rcp(new Basis_HDIV_TET_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
    }
    
  }
  else if (type == "HDIV_AC") {
    if (dimension == 2) {
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
    if (dimension == 1) {
      // need to throw an error
    }
    else if (dimension == 2) {
      if (shape == "Quadrilateral_4") {
        basis = Teuchos::rcp(new Basis_HCURL_QUAD_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
      else if (shape == "Triangle_3") {
        basis = Teuchos::rcp(new Basis_HCURL_TRI_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
    }
    else if (dimension == 3) {
      if (shape == "Hexahedron_8") {
        basis = Teuchos::rcp(new Basis_HCURL_HEX_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
      else if (shape == "Tetrahedron_4") {
        basis = Teuchos::rcp(new Basis_HCURL_TET_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
    }
    
  }
  else if (type == "HFACE") {
    if (dimension == 2) {
      if (shape == "Quadrilateral_4") {
        basis = Teuchos::rcp(new Basis_HFACE_QUAD_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
      else if (shape == "Triangle_3") {
        basis = Teuchos::rcp(new Basis_HFACE_TRI_In_FEM<PHX::Device::execution_space,double,double>(degree,POINTTYPE_EQUISPACED) );
      }
    }
    if (dimension == 3) {
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

void DiscretizationInterface::setReferenceData(Teuchos::RCP<GroupMetaData> & groupData) {
  
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
  
  // ------------------------------------
  // Get refnodes
  // ------------------------------------
  
  DRV refnodes("nodes on reference element",cellTopo->getNodeCount(),dimension);
  CellTools::getReferenceSubcellVertices(refnodes, dimension, 0, *cellTopo);
  groupData->ref_nodes = refnodes;
  
  // ------------------------------------
  // Get ref basis
  // ------------------------------------
  
  groupData->basis_pointers = basis_pointers[block];
  groupData->basis_types = basis_types[block];
  
  for (size_t i=0; i<basis_pointers[block].size(); i++) {
    
    int numb = basis_pointers[block][i]->getCardinality();
    
    DRV basisvals, basisgrad, basisdiv, basiscurl;
    DRV basisnodes;
        
    if (basis_types[block][i].substr(0,5) == "HGRAD") {
      
      basisvals = DRV("basisvals",numb, groupData->num_ip);
      basis_pointers[block][i]->getValues(basisvals, groupData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0));
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basisgrad = DRV("basisgrad",numb, groupData->num_ip, dimension);
      basis_pointers[block][i]->getValues(basisgrad, groupData->ref_ip, Intrepid2::OPERATOR_GRAD);
      
    }
    else if (basis_types[block][i].substr(0,4) == "HVOL") {
      
      basisvals = DRV("basisvals",numb, groupData->num_ip);
      basis_pointers[block][i]->getValues(basisvals, groupData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0));
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
    }
    else if (basis_types[block][i].substr(0,4) == "HDIV") {
      
      basisvals = DRV("basisvals",numb, groupData->num_ip, dimension);
      basis_pointers[block][i]->getValues(basisvals, groupData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0), dimension);
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basisdiv = DRV("basisdiv",numb, groupData->num_ip);
      basis_pointers[block][i]->getValues(basisdiv, groupData->ref_ip, Intrepid2::OPERATOR_DIV);
      
    }
    else if (basis_types[block][i].substr(0,5) == "HCURL"){
      
      basisvals = DRV("basisvals",numb, groupData->num_ip, dimension);
      basis_pointers[block][i]->getValues(basisvals, groupData->ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0), dimension);
      basis_pointers[block][i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      if (dimension == 2) {
        basiscurl = DRV("basiscurl",numb, groupData->num_ip);
      }
      else if (dimension == 3) {
        basiscurl = DRV("basiscurl",numb, groupData->num_ip, dimension);
      }
      basis_pointers[block][i]->getValues(basiscurl, groupData->ref_ip, Intrepid2::OPERATOR_CURL);
      
    }
    
    groupData->ref_basis.push_back(basisvals);
    groupData->ref_basis_curl.push_back(basiscurl);
    groupData->ref_basis_grad.push_back(basisgrad);
    groupData->ref_basis_div.push_back(basisdiv);
    groupData->ref_basis_nodes.push_back(basisnodes);
  }
  
  // Compute the basis value and basis grad values on reference element
  // at side ip
  for (size_t s=0; s<groupData->num_sides; s++) {
    vector<DRV> sbasis, sbasisgrad, sbasisdiv, sbasiscurl;
    for (size_t i=0; i<basis_pointers[block].size(); i++) {
      int numb = basis_pointers[block][i]->getCardinality();
      DRV basisvals, basisgrad, basisdiv, basiscurl;
      if (basis_types[block][i].substr(0,5) == "HGRAD") {
        basisvals = DRV("basisvals",numb, groupData->num_side_ip);
        basis_pointers[block][i]->getValues(basisvals, groupData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
        basisgrad = DRV("basisgrad",numb, groupData->num_side_ip, dimension);
        basis_pointers[block][i]->getValues(basisgrad, groupData->ref_side_ip[s], Intrepid2::OPERATOR_GRAD);
      }
      else if (basis_types[block][i].substr(0,4) == "HVOL" || basis_types[block][i].substr(0,5) == "HFACE") {
        basisvals = DRV("basisvals",numb, groupData->num_side_ip);
        basis_pointers[block][i]->getValues(basisvals, groupData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
      }
      else if (basis_types[block][i].substr(0,4) == "HDIV") {
        basisvals = DRV("basisvals",numb, groupData->num_side_ip, dimension);
        basis_pointers[block][i]->getValues(basisvals, groupData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
      }
      else if (basis_types[block][i].substr(0,5) == "HCURL"){
        basisvals = DRV("basisvals",numb, groupData->num_side_ip, dimension);
        basis_pointers[block][i]->getValues(basisvals, groupData->ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
      }
      sbasis.push_back(basisvals);
      sbasisgrad.push_back(basisgrad);
      sbasisdiv.push_back(basisdiv);
      sbasiscurl.push_back(basiscurl);
    }
    groupData->ref_side_basis.push_back(sbasis);
    groupData->ref_side_basis_grad.push_back(sbasisgrad);
    groupData->ref_side_basis_div.push_back(sbasisdiv);
    groupData->ref_side_basis_curl.push_back(sbasiscurl);
  }
  
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
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type pt=0; pt<wts.extent(1); ++pt) {
      measure(elem) += wts(elem,pt);
    }
  });
        
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
    localIds[e] = elemIDs_host(e);//my_elements[block](elemIDs_host(e));
  }
  DRV nodes = mesh->getMyNodes(block, localIds);
  
  return nodes;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData,
                                                         Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                         vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                                         vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                                         vector<View_Sc4> & basis_nodes,
                                                         const bool & apply_orientations) {
  DRV nodes = this->getMyNodes(groupData->my_block, elemIDs);
  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation("kv to orients",elemIDs.extent(0));
  this->getPhysicalOrientations(groupData, elemIDs, orientation, true);
  this->getPhysicalVolumetricBasis(groupData, nodes, orientation, basis, basis_grad,
                                   basis_curl, basis_div, basis_nodes, apply_orientations);
}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData,
                                                         DRV nodes,
                                                         Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                         vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                                         vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div,
                                                         vector<View_Sc4> & basis_nodes,
                                                         const bool & apply_orientations) {
  
  Teuchos::TimeMonitor localtimer(*phys_vol_data_total_timer);
  
  int dimension = groupData->dimension;
  int numip = groupData->ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  DRV jacobian, jacobianDet, jacobianInv, tmpip, tmpwts;
  jacobian = DRV("jacobian", numElem, numip, dimension, dimension);
  jacobianDet = DRV("determinant of jacobian", numElem, numip);
  jacobianInv = DRV("inverse of jacobian", numElem, numip, dimension, dimension);
  
  {
    Teuchos::TimeMonitor localtimer(*phys_vol_data_set_jac_timer);
    CellTools::setJacobian(jacobian, groupData->ref_ip, nodes, *(groupData->cell_topo));
  }
  
  {
    Teuchos::TimeMonitor localtimer(*phys_vol_data_other_jac_timer);
    CellTools::setJacobianDet(jacobianDet, jacobian);
    CellTools::setJacobianInv(jacobianInv, jacobian);
  }
  
  // -------------------------------------------------
  // Compute the basis functions at the volumetric ip
  // -------------------------------------------------
  
  {
    Teuchos::TimeMonitor localtimer(*phys_vol_data_basis_timer);
    for (size_t i=0; i<groupData->basis_pointers.size(); i++) {
      
      int numb = groupData->basis_pointers[i]->getCardinality();
      
      // These will be redefined below for the appropriate basis types
      View_Sc4 basis_vals("tmp basis",1,1,1,1);
      View_Sc4 basis_grad_vals("tmp grad vals",1,1,1,1);
      View_Sc4 basis_curl_vals("tmp curl vals",1,1,1,1);
      View_Sc4 basis_node_vals("tmp node vals",1,1,1,1);
      View_Sc3 basis_div_vals("tmp div vals",1,1,1);

      if (groupData->basis_types[i].substr(0,5) == "HGRAD"){
        {
          DRV bvals1, bvals2;
          bvals1 = DRV("basis",numElem,numb,numip);
          bvals2 = DRV("basis tmp",numElem,numb,numip);
          
          FuncTools::HGRADtransformVALUE(bvals1, groupData->ref_basis[i]);
          if (apply_orientations && groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bvals2 = bvals1;
          }
          basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
          auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
          Kokkos::deep_copy(basis_vals_slice,bvals2);
          
          DRV bgrad1, bgrad2;
          bgrad1 = DRV("basis grad tmp",numElem,numb,numip,dimension);
          bgrad2 = DRV("basis grad",numElem,numb,numip,dimension);
          
          FuncTools::HGRADtransformGRAD(bgrad1, jacobianInv, groupData->ref_basis_grad[i]);
          if (apply_orientations && groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bgrad2, bgrad1, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bgrad2 = bgrad1;
          }
          basis_grad_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
          Kokkos::deep_copy(basis_grad_vals,bgrad2);
        }

        if (groupData->require_basis_at_nodes) {
          DRV bnode_vals("basis",numElem,numb,nodes.extent(1));
          DRV bvals_tmp("basis tmp",numElem,numb,nodes.extent(1));
          FuncTools::HGRADtransformVALUE(bvals_tmp, groupData->ref_basis_nodes[i]);
          if (apply_orientations && groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bnode_vals, bvals_tmp, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bnode_vals = bvals_tmp;
          }
          basis_node_vals = View_Sc4("basis values", numElem, numb, nodes.extent(1), dimension);
          auto basis_node_vals_sv = subview(basis_node_vals, ALL(), ALL(), ALL(), 0);
          Kokkos::deep_copy(basis_node_vals_sv,bnode_vals);
        }
        
      }
      else if (groupData->basis_types[i].substr(0,4) == "HVOL"){
        
        DRV bvals1;
        bvals1 = DRV("basis",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, groupData->ref_basis[i]);
        
        basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals1);
      }
      else if (groupData->basis_types[i].substr(0,4) == "HDIV" ) {
        
        {
          Teuchos::TimeMonitor localtimer(*phys_vol_data_basis_div_val_timer);
          DRV bvals1, bvals2;
          bvals1 = DRV("basis",numElem,numb,numip,dimension);
          bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
          
          FuncTools::HDIVtransformVALUE(bvals1, jacobian, jacobianDet, groupData->ref_basis[i]);
          if (apply_orientations && groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bvals2 = bvals1;
          }
          basis_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_vals,bvals2);
        }
        
        if (groupData->require_basis_at_nodes) {
          DRV bnode_vals("basis",numElem,numb,nodes.extent(1),dimension);
          DRV bvals_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
          FuncTools::HDIVtransformVALUE(bvals_tmp, jacobian, jacobianDet, groupData->ref_basis_nodes[i]);
          if (apply_orientations && groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bnode_vals, bvals_tmp, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bnode_vals = bvals_tmp;
          }
          basis_node_vals = View_Sc4("basis values", numElem, numb, nodes.extent(1), dimension);
          Kokkos::deep_copy(basis_node_vals,bnode_vals);
        }
        
        {
          Teuchos::TimeMonitor localtimer(*phys_vol_data_basis_div_div_timer);
          
          DRV bdiv1, bdiv2;
          bdiv1 = DRV("basis",numElem,numb,numip);
          bdiv2 = DRV("basis tmp",numElem,numb,numip);
          
          FuncTools::HDIVtransformDIV(bdiv1, jacobianDet, groupData->ref_basis_div[i]);
          if (apply_orientations && groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bdiv2, bdiv1, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bdiv2 = bdiv1;
          }
          basis_div_vals = View_Sc3("basis div values", numElem, numb, numip); // needs to be rank-3
          Kokkos::deep_copy(basis_div_vals,bdiv2);
        }
      }
      else if (groupData->basis_types[i].substr(0,5) == "HCURL"){
        
        {
          Teuchos::TimeMonitor localtimer(*phys_vol_data_basis_curl_val_timer);
          DRV bvals1, bvals2;
          bvals1 = DRV("basis",numElem,numb,numip,dimension);
          bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
          
          FuncTools::HCURLtransformVALUE(bvals1, jacobianInv, groupData->ref_basis[i]);
          if (apply_orientations && groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bvals2 = bvals1;
          }
          basis_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_vals,bvals2);
        }
        
        if (groupData->require_basis_at_nodes) {
          DRV bnode_vals("basis",numElem,numb,nodes.extent(1),dimension);
          DRV bvals_tmp("basis tmp",numElem,numb,nodes.extent(1),dimension);
          FuncTools::HCURLtransformVALUE(bvals_tmp, jacobianInv, groupData->ref_basis_nodes[i]);
          if (apply_orientations && groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bnode_vals, bvals_tmp, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bnode_vals = bvals_tmp;
          }
          basis_node_vals = View_Sc4("basis values", numElem, numb, nodes.extent(1), dimension);
          Kokkos::deep_copy(basis_node_vals,bnode_vals);
          
        }
        
        {
          Teuchos::TimeMonitor localtimer(*phys_vol_data_basis_curl_curl_timer);
        
          DRV bcurl1, bcurl2;
          bcurl1 = DRV("basis",numElem,numb,numip,dimension);
          bcurl2 = DRV("basis tmp",numElem,numb,numip,dimension);
          
          FuncTools::HCURLtransformCURL(bcurl1, jacobian, jacobianDet, groupData->ref_basis_curl[i]);
          if (apply_orientations && groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bcurl2, bcurl1, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bcurl2 = bcurl1;
          }
          basis_curl_vals = View_Sc4("basis curl values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_curl_vals, bcurl2);
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

void DiscretizationInterface::getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData, 
                                                         Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                         vector<View_Sc4> & basis) {
  DRV nodes = this->getMyNodes(groupData->my_block, elemIDs);
  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation("kv to orients",elemIDs.extent(0));
  this->getPhysicalOrientations(groupData, elemIDs, orientation, true);
  this->getPhysicalVolumetricBasis(groupData, nodes, orientation, basis);                                       
}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getPhysicalVolumetricBasis(Teuchos::RCP<GroupMetaData> & groupData,
                                                         DRV nodes,
                                                         Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                         vector<View_Sc4> & basis) {
  
  Teuchos::TimeMonitor localtimer(*phys_vol_data_total_timer);
  
  int dimension = groupData->dimension;
  int numip = groupData->ref_ip.extent(0);
  int numElem = orientation.extent(0);
  
  
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  DRV jacobian, jacobianDet, jacobianInv;
  jacobian = DRV("jacobian", numElem, numip, dimension, dimension);
  jacobianDet = DRV("determinant of jacobian", numElem, numip);
  jacobianInv = DRV("inverse of jacobian", numElem, numip, dimension, dimension);
  {
    Teuchos::TimeMonitor localtimer(*phys_vol_data_set_jac_timer);
    CellTools::setJacobian(jacobian, groupData->ref_ip, nodes, *(groupData->cell_topo));
  }
  {
    Teuchos::TimeMonitor localtimer(*phys_vol_data_other_jac_timer);
    CellTools::setJacobianDet(jacobianDet, jacobian);
    CellTools::setJacobianInv(jacobianInv, jacobian);
  }
    
  // -------------------------------------------------
  // Compute the basis functions at the volumetric ip
  // -------------------------------------------------
  
  {
    Teuchos::TimeMonitor localtimer(*phys_vol_data_basis_timer);
    for (size_t i=0; i<groupData->basis_pointers.size(); i++) {
      
      int numb = groupData->basis_pointers[i]->getCardinality();
      
      // These will be redefined below for the appropriate basis types
      View_Sc4 basis_vals("tmp basis",1,1,1,1);
      
      if (groupData->basis_types[i].substr(0,5) == "HGRAD"){
        DRV bvals1("basis",numElem,numb,numip);
        DRV bvals2("basis tmp",numElem,numb,numip);
        FuncTools::HGRADtransformVALUE(bvals1, groupData->ref_basis[i]);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
        basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
        
        
      }
      else if (groupData->basis_types[i].substr(0,4) == "HVOL"){
        DRV bvals1("basis",numElem,numb,numip);
        FuncTools::HGRADtransformVALUE(bvals1, groupData->ref_basis[i]);
        
        basis_vals = View_Sc4("basis values", numElem, numb, numip, 1); // needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals1);
      }
      else if (groupData->basis_types[i].substr(0,4) == "HDIV" ) {
        
        {
          Teuchos::TimeMonitor localtimer(*phys_vol_data_basis_div_val_timer);
          DRV bvals1("basis",numElem,numb,numip,dimension);
          DRV bvals2("basis tmp",numElem,numb,numip,dimension);
          FuncTools::HDIVtransformVALUE(bvals1, jacobian, jacobianDet, groupData->ref_basis[i]);
          if (groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bvals2 = bvals1;
          }
          basis_vals = View_Sc4("basis values", numElem, numb, numip, dimension);
          Kokkos::deep_copy(basis_vals,bvals2);
        }
        
      }
      else if (groupData->basis_types[i].substr(0,5) == "HCURL"){
        {
          Teuchos::TimeMonitor localtimer(*phys_vol_data_basis_curl_val_timer);
          
          DRV bvals1("basis",numElem,numb,numip,dimension);
          DRV bvals2("basis tmp",numElem,numb,numip,dimension);
          FuncTools::HCURLtransformVALUE(bvals1, jacobianInv, groupData->ref_basis[i]);
          if (groupData->basis_pointers[i]->requireOrientation()) {
            OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                  groupData->basis_pointers[i].get());
          }
          else {
            bvals2 = bvals1;
          }
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

void DiscretizationInterface::getPhysicalOrientations(Teuchos::RCP<GroupMetaData> & groupData,
                                                      Kokkos::View<LO*,AssemblyDevice> eIndex,
                                                      Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                      const bool & use_block) {
  
  Teuchos::TimeMonitor localtimer(*phys_orient_timer);
  
  auto orientation_host = create_mirror_view(orientation);
  auto host_eIndex = Kokkos::create_mirror_view(eIndex);
  deep_copy(host_eIndex,eIndex);
  for (size_type i=0; i<host_eIndex.extent(0); i++) {
    LO elemID = host_eIndex(i);
    if (use_block) {
      elemID = my_elements[groupData->my_block](host_eIndex(i));
    }
    if ((int)panzer_orientations.extent(0) > elemID) {
      orientation_host(i) = panzer_orientations(elemID);
    }
    else { // account for simple mesh, which only needs 1 orientation
      orientation_host(i) = panzer_orientations(0);
    }
  }
  deep_copy(orientation,orientation_host);
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
}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getPhysicalFaceBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & side, 
                                                   Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                   vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad) {

  DRV nodes = this->getMyNodes(groupData->my_block, elemIDs);
  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation("kv to orients",elemIDs.extent(0));
  this->getPhysicalOrientations(groupData, elemIDs, orientation, true);
  this->getPhysicalFaceBasis(groupData, side, nodes, orientation, basis, basis_grad);

}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getPhysicalFaceBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & side,
                                                   DRV nodes,
                                                   Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                   vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad) {
    
  Teuchos::TimeMonitor localtimer(*phys_face_data_total_timer);
  
  auto ref_ip = groupData->ref_side_ip[side];
  auto ref_wts = groupData->ref_side_wts[side];
  
  int dimension = groupData->dimension;
  int numip = ref_ip.extent(0);
  int numElem = nodes.extent(0);

  
  // Step 1: fill in ip_side, wts_side and normals
  DRV jacobian("face jac", numElem, numip, dimension, dimension);
  DRV jacobianDet("face jacDet", numElem, numip);
  DRV jacobianInv("face jacInv", numElem, numip, dimension, dimension);
  CellTools::setJacobian(jacobian, ref_ip, nodes, *(groupData->cell_topo));
  CellTools::setJacobianInv(jacobianInv, jacobian);
  CellTools::setJacobianDet(jacobianDet, jacobian);
  
  // Step 2: define basis functions at these integration points
  
  {
    Teuchos::TimeMonitor localtimer(*phys_face_data_basis_timer);
    
    for (size_t i=0; i<groupData->basis_pointers.size(); i++) {
      int numb = groupData->basis_pointers[i]->getCardinality();
      
      // These will be defined below for the appropriate basis types
      View_Sc4 basis_vals("tmp basis vals",1,1,1,1);
      View_Sc4 basis_grad_vals("tmp grad vals",1,1,1,1);
      
      // div and curl values are not currently used on boundaries
      
      auto ref_basis_vals = groupData->ref_side_basis[side][i];
      
      if (groupData->basis_types[i].substr(0,5) == "HGRAD"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip);
        bvals2 = DRV("basis tmp",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
        
        auto ref_basis_grad_vals = groupData->ref_side_basis_grad[side][i];
        DRV bgrad1, bgrad2;
        bgrad1 = DRV("basis",numElem,numb,numip,dimension);
        bgrad2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HGRADtransformGRAD(bgrad1, jacobianInv, ref_basis_grad_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bgrad2, bgrad1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
        basis_grad_vals = View_Sc4("face basis grad vals",numElem,numb,numip,dimension); // Needs to be rank-4
        Kokkos::deep_copy(basis_grad_vals,bgrad2);
        
      }
      else if (groupData->basis_types[i].substr(0,4) == "HVOL"){
        
        DRV bvals1;
        bvals1 = DRV("basis",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,1); // Needs to be rank-4
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals1);
      }
      else if (groupData->basis_types[i].substr(0,4) == "HDIV" ) {
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip,dimension);
        bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HDIVtransformVALUE(bvals1, jacobian, jacobianDet, ref_basis_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_vals,bvals2);
        
      }
      else if (groupData->basis_types[i].substr(0,5) == "HCURL"){
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip,dimension);
        bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HCURLtransformVALUE(bvals1, jacobianInv, ref_basis_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
        basis_vals = View_Sc4("face basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_vals,bvals2);
        
      }
      else if (groupData->basis_types[i].substr(0,5) == "HFACE"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip);
        bvals2 = DRV("basis tmp",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
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
                   KOKKOS_LAMBDA (const int elem ) {
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
}

//======================================================================
//
//======================================================================

void DiscretizationInterface::getPhysicalBoundaryBasis(Teuchos::RCP<GroupMetaData> & groupData, Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                       LO & localSideID,
                                                       vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                                       vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div) {

  DRV nodes = this->getMyNodes(groupData->my_block, elemIDs);
  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation("kv to orients",elemIDs.extent(0));
  this->getPhysicalOrientations(groupData, elemIDs, orientation, false);
  this->getPhysicalBoundaryBasis(groupData, nodes, localSideID, orientation, basis, basis_grad, basis_curl, basis_div);

}

// ========================================================================================
// ========================================================================================

void DiscretizationInterface::getPhysicalBoundaryBasis(Teuchos::RCP<GroupMetaData> & groupData, DRV nodes,
                                                       LO & localSideID,
                                                       Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                       vector<View_Sc4> & basis, vector<View_Sc4> & basis_grad,
                                                       vector<View_Sc4> & basis_curl, vector<View_Sc3> & basis_div) {
                                                      
  Teuchos::TimeMonitor localtimer(*phys_bndry_data_total_timer);
  
  int dimension = groupData->dimension;
  
  DRV ref_ip = groupData->ref_side_ip[localSideID];
  DRV ref_wts = groupData->ref_side_wts[localSideID];
  
  int numip = ref_ip.extent(0);
  int numElem = nodes.extent(0);
  
  
  // -------------------------------------------------
  // Compute the integration information
  // -------------------------------------------------
  
  DRV jacobian = DRV("bijac", numElem, numip, dimension, dimension);
  DRV jacobianDet = DRV("bijacDet", numElem, numip);
  DRV jacobianInv = DRV("bijacInv", numElem, numip, dimension, dimension);
  CellTools::setJacobian(jacobian, ref_ip, nodes, *(groupData->cell_topo));
  CellTools::setJacobianInv(jacobianInv, jacobian);
  CellTools::setJacobianDet(jacobianDet, jacobian);
  
  {
    Teuchos::TimeMonitor localtimer(*phys_bndry_data_basis_timer);
    
    for (size_t i=0; i<groupData->basis_pointers.size(); i++) {
      
      int numb = groupData->basis_pointers[i]->getCardinality();
      
      // These will be redefined below for the appropriate basis type
      View_Sc4 basis_vals("tmp basis vals",1,1,1,1);
      View_Sc4 basis_grad_vals("tmp grad vals",1,1,1,1);
      View_Sc4 basis_curl_vals("tmp curl vals",1,1,1,1);
      View_Sc3 basis_div_vals("tmp div vals",1,1,1);
      
      DRV ref_basis_vals = groupData->ref_side_basis[localSideID][i];
      
      if (groupData->basis_types[i].substr(0,5) == "HGRAD"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip);
        bvals2 = DRV("basis tmp",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,1);
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
        
        DRV bgrad1, bgrad2;
        bgrad1 = DRV("basis",numElem,numb,numip,dimension);
        bgrad2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        DRV ref_bgrad_vals = groupData->ref_side_basis_grad[localSideID][i];
        FuncTools::HGRADtransformGRAD(bgrad1, jacobianInv, ref_bgrad_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bgrad2, bgrad1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bgrad2 = bgrad1;
        }
        basis_grad_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_grad_vals,bgrad2);
        
      }
      else if (groupData->basis_types[i].substr(0,4) == "HVOL"){ // does not require orientations
        
        DRV bvals1;
        bvals1 = DRV("basis",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,1);
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals1);
      }
      else if (groupData->basis_types[i].substr(0,5) == "HFACE"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip);
        bvals2 = DRV("basis tmp",numElem,numb,numip);
        
        FuncTools::HGRADtransformVALUE(bvals1, ref_basis_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,1);
        auto basis_vals_slice = Kokkos::subview(basis_vals,Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), 0);
        Kokkos::deep_copy(basis_vals_slice,bvals2);
      }
      else if (groupData->basis_types[i].substr(0,4) == "HDIV"){
        
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip,dimension);
        bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HDIVtransformVALUE(bvals1, jacobian, jacobianDet, ref_basis_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
        OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                              groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
        basis_vals = View_Sc4("basis vals",numElem,numb,numip,dimension);
        Kokkos::deep_copy(basis_vals,bvals2);
      }
      else if (groupData->basis_types[i].substr(0,5) == "HCURL"){
        DRV bvals1, bvals2;
        bvals1 = DRV("basis",numElem,numb,numip,dimension);
        bvals2 = DRV("basis tmp",numElem,numb,numip,dimension);
        
        FuncTools::HCURLtransformVALUE(bvals1, jacobianInv, ref_basis_vals);
        if (groupData->basis_pointers[i]->requireOrientation()) {
          OrientTools::modifyBasisByOrientation(bvals2, bvals1, orientation,
                                                groupData->basis_pointers[i].get());
        }
        else {
          bvals2 = bvals1;
        }
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

DRV DiscretizationInterface::evaluateBasis(Teuchos::RCP<GroupMetaData> & groupData, const int & block, const int & basisID, const Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                           const DRV & evalpts, topo_RCP & cellTopo) {

  DRV nodes = this->getMyNodes(block, elemIDs);
  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation("kv to orients",elemIDs.extent(0));
  this->getPhysicalOrientations(groupData, elemIDs, orientation, true);
  DRV basis = this->evaluateBasis(block, basisID, nodes, evalpts, cellTopo, orientation);
  return basis;

}

// ========================================================================================
// ========================================================================================

DRV DiscretizationInterface::evaluateBasis(const int & block, const int & basisID, DRV nodes,
                                           const DRV & evalpts, topo_RCP & cellTopo,
                                           Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int numBasis = basis_pointers[block][basisID]->getCardinality();
  
  
  DRV finalbasis;
  
  if (basis_types[block][basisID] == "HGRAD" || basis_types[block][basisID] == "HVOL") {
    DRV basisvals("basisvals", numBasis, numpts);
    basis_pointers[block][basisID]->getValues(basisvals, evalpts, Intrepid2::OPERATOR_VALUE);
  
    DRV basisvals_Transformed("basisvals_Transformed", numCells, numBasis, numpts);
    FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
    finalbasis = DRV("basisvals_Transformed", numCells, numBasis, numpts);
    if (basis_pointers[block][basisID]->requireOrientation()) {
      OrientTools::modifyBasisByOrientation(finalbasis, basisvals_Transformed,
                                            orientation, basis_pointers[block][basisID].get());
    }
    else {
      finalbasis = basisvals_Transformed;
    }
  
  }
  else if (basis_types[block][basisID] == "HDIV") {
    DRV basisvals("basisvals", numBasis, numpts, dimension);
    basis_pointers[block][basisID]->getValues(basisvals, evalpts, Intrepid2::OPERATOR_VALUE);
  
    DRV jacobian, jacobianDet;
    jacobian = DRV("jacobian", numCells, numpts, dimension, dimension);
    jacobianDet = DRV("determinant of jacobian", numCells, numpts);
    
    CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
    CellTools::setJacobianDet(jacobianDet, jacobian);
  
    DRV bvals1("basis", numCells, numBasis, numpts, dimension);
    finalbasis = DRV("basis tmp", numCells, numBasis, numpts, dimension);
    FuncTools::HDIVtransformVALUE(bvals1, jacobian, jacobianDet, basisvals);
    if (basis_pointers[block][basisID]->requireOrientation()) {
      OrientTools::modifyBasisByOrientation(finalbasis, bvals1, orientation,
                                            basis_pointers[block][basisID].get());
    }
    else {
      finalbasis = bvals1;
    }
    
  }
  else if (basis_types[block][basisID] == "HCURL") {

    DRV basisvals("basisvals", numBasis, numpts, dimension);
    basis_pointers[block][basisID]->getValues(basisvals, evalpts, Intrepid2::OPERATOR_VALUE);
  
    DRV jacobian, jacobianInv;
    jacobian = DRV("jacobian", numCells, numpts, dimension, dimension);
    jacobianInv = DRV("inverse of jacobian", numCells, numpts, dimension, dimension);
    
    CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
    CellTools::setJacobianInv(jacobianInv, jacobian);
  
    DRV bvals1("basis",numCells, numBasis, numpts, dimension);
    finalbasis = DRV("basis tmp", numCells, numBasis, numpts, dimension);

    FuncTools::HCURLtransformVALUE(bvals1, jacobianInv, basisvals);
    if (basis_pointers[block][basisID]->requireOrientation()) {
      OrientTools::modifyBasisByOrientation(finalbasis, bvals1, orientation,
                                        basis_pointers[block][basisID].get());
    }
    else {
      finalbasis = bvals1;
    }
  }

  return finalbasis;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::evaluateBasisNewQuadrature(Teuchos::RCP<GroupMetaData> & groupData, const int & block, 
                                                        const int & basisID, vector<string> & quad_rules,
                                                        Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                        DRV & wts) {
  DRV nodes = this->getMyNodes(block, elemIDs);
  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation("kv to orients",elemIDs.extent(0));
  this->getPhysicalOrientations(groupData, elemIDs, orientation, true);
  DRV basis = this->evaluateBasisNewQuadrature(block, basisID, quad_rules, nodes, orientation, wts);
  return basis;
}

// ========================================================================================
// ========================================================================================

DRV DiscretizationInterface::evaluateBasisNewQuadrature(const int & block, const int & basisID, vector<string> & quad_rules,
                                                        DRV nodes,
                                                        Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                                        DRV & wts) {
  

  Teuchos::TimeMonitor localtimer(*phys_basis_new_quad_timer);

  debugger->print("**** Starting DiscretizationInterface::evaluateBasisNewQuadrature() ...");

  DRV finalbasis;
  
  const Intrepid2::ordinal_type num_basis = basis_pointers[block][basisID]->getCardinality();  
  size_type numElem = nodes.extent(0);

  auto cellTopo = basis_pointers[block][basisID]->getBaseCellTopology();
  // Use the strings to define a tensor product quadrature rule

  // Add check that the number of quadrature rules matches the spatial dimension

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device::execution_space, double, double>> basis_cubature;
  if (dimension == 1) {

  }
  else if (dimension == 2) {

  }
  else {
    Intrepid2::EPolyType qtype_x, qtype_y, qtype_z;

    if (quad_rules[0] == "GAUSS-LOBATTO") { 
      qtype_x = Intrepid2::POLYTYPE_GAUSS_LOBATTO;
    }
    else {
      qtype_x = Intrepid2::POLYTYPE_GAUSS;
    }
    
    if (quad_rules[1] == "GAUSS-LOBATTO") { 
      qtype_y = Intrepid2::POLYTYPE_GAUSS_LOBATTO;
    }
    else {
      qtype_y = Intrepid2::POLYTYPE_GAUSS;
    }
    
    if (quad_rules[2] == "GAUSS-LOBATTO") { 
      qtype_z = Intrepid2::POLYTYPE_GAUSS_LOBATTO;
    }
    else {
      qtype_z = Intrepid2::POLYTYPE_GAUSS;
    }

    const auto line_cubature_x = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quadorder-1, qtype_x);//Intrepid2::POLYTYPE_GAUSS_LOBATTO);
    const auto line_cubature_y = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quadorder-1, qtype_y);//Intrepid2::POLYTYPE_GAUSS_LOBATTO);
    const auto line_cubature_z = Intrepid2::CubaturePolylib<PHX::Device::execution_space, double, double>(quadorder-1, qtype_z);//Intrepid2::POLYTYPE_GAUSS_LOBATTO);
    basis_cubature = Teuchos::rcp(new Intrepid2::CubatureTensor<PHX::Device::execution_space, double, double>(line_cubature_x, line_cubature_y, line_cubature_z));
  }
  const int num_pts = basis_cubature->getNumPoints();

  DRV ref_ip("reference integration points", num_pts, dimension);
  DRV ref_wts("reference weights", num_pts);
  basis_cubature->getCubature(ref_ip, ref_wts);

  DRV jacobian("jacobian", numElem, num_pts, dimension, dimension);
  DRV jacobianDet("determinant of jacobian", numElem, num_pts);
    
  CellTools::setJacobian(jacobian, ref_ip, nodes, cellTopo);
  CellTools::setJacobianDet(jacobianDet, jacobian);
  wts = DRV("physical wts", numElem, num_pts);
  FuncTools::computeCellMeasure(wts, jacobianDet, ref_wts);

  // Evaluate the basis, map to physical and apply orientations
  
  if (basis_types[block][basisID] == "HGRAD" || basis_types[block][basisID] == "HVOL") {
    DRV basisvals("reference basis values", num_basis, num_pts, dimension);
    basis_pointers[block][basisID]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);

    DRV basisvals_Transformed("basisvals_Transformed", numElem, num_basis, num_pts);
    FuncTools::HGRADtransformVALUE(basisvals_Transformed, basisvals);
    finalbasis = DRV("basisvals_Transformed", numElem, num_basis, num_pts);
    if (basis_pointers[block][basisID]->requireOrientation()) {
      OrientTools::modifyBasisByOrientation(finalbasis, basisvals_Transformed,
                                            orientation, basis_pointers[block][basisID].get());
    }
    else {
      finalbasis = basisvals_Transformed;
    }
  
  }
  else if (basis_types[block][basisID] == "HDIV") {
    DRV basisvals("basisvals", num_basis, num_pts, dimension);
    basis_pointers[block][basisID]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
  
    DRV bvals1("basis", numElem, num_basis, num_pts, dimension);
    finalbasis = DRV("basis tmp", numElem, num_basis, num_pts, dimension);
    FuncTools::HDIVtransformVALUE(bvals1, jacobian, jacobianDet, basisvals);
    if (basis_pointers[block][basisID]->requireOrientation()) {
      OrientTools::modifyBasisByOrientation(finalbasis, bvals1, orientation,
                                            basis_pointers[block][basisID].get());
    }
    else {
      finalbasis = bvals1;
    }
    
  }
  else if (basis_types[block][basisID] == "HCURL") {

    DRV basisvals("basisvals", num_basis, num_pts, dimension);
    basis_pointers[block][basisID]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
  
    DRV jacobianInv("inverse of jacobian", numElem, num_pts, dimension, dimension);
    CellTools::setJacobianInv(jacobianInv, jacobian);
  
    DRV bvals1("basis", numElem, num_basis, num_pts, dimension);
    finalbasis = DRV("basis tmp", numElem, num_basis, num_pts, dimension);

    FuncTools::HCURLtransformVALUE(bvals1, jacobianInv, basisvals);
    if (basis_pointers[block][basisID]->requireOrientation()) {
      OrientTools::modifyBasisByOrientation(finalbasis, bvals1, orientation,
                                        basis_pointers[block][basisID].get());
    }
    else {
      finalbasis = bvals1;
    }
  }

  debugger->print("**** Finished DiscretizationInterface::evaluateBasisNewQuadrature()");

  return finalbasis;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::evaluateBasisGrads(const size_t & block, const basis_RCP & basis_pointer, const Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                const DRV & evalpts, const topo_RCP & cellTopo) {
  DRV nodes = this->getMyNodes(block, elemIDs);
  DRV basisgrads = this->evaluateBasisGrads(basis_pointer, nodes, evalpts, cellTopo);
  return basisgrads;
}

// ========================================================================================
// ========================================================================================

DRV DiscretizationInterface::evaluateBasisGrads(const basis_RCP & basis_pointer, DRV nodes,
                                                const DRV & evalpts, const topo_RCP & cellTopo) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int dimension = evalpts.extent(1);
  int numBasis = basis_pointer->getCardinality();
  
  DRV basisgrads("basisgrads", numBasis, numpts, dimension);
  DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, dimension);
  basis_pointer->getValues(basisgrads, evalpts, Intrepid2::OPERATOR_GRAD);
  
  DRV jacobian("jacobian", numCells, numpts, dimension, dimension);
  DRV jacobInv("jacobInv", numCells, numpts, dimension, dimension);
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianInv(jacobInv, jacobian);
  FuncTools::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
  
  return basisgrads_Transformed;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::evaluateBasisGrads2(Teuchos::RCP<GroupMetaData> & groupData, 
                                                const size_t & block, const basis_RCP & basis_pointer, const Kokkos::View<LO*,AssemblyDevice> elemIDs,
                                                const DRV & evalpts, const topo_RCP & cellTopo) {

  DRV nodes = this->getMyNodes(block, elemIDs);
  Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation("kv to orients",elemIDs.extent(0));
  this->getPhysicalOrientations(groupData, elemIDs, orientation, true);
  DRV basisgrads = this->evaluateBasisGrads2(basis_pointer, nodes, evalpts, cellTopo, orientation);
  return basisgrads;

}

// ========================================================================================
// ========================================================================================

DRV DiscretizationInterface::evaluateBasisGrads2(const basis_RCP & basis_pointer, DRV nodes,
                                                const DRV & evalpts, const topo_RCP & cellTopo,
                                                Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & orientation) {
  
  int numCells = 1;
  int numpts = evalpts.extent(0);
  int dimension = evalpts.extent(1);
  int numBasis = basis_pointer->getCardinality();

  
  DRV basisgrads("basisgrads", numBasis, numpts, dimension);
  DRV basisgrads_Transformed("basisgrads_Transformed", numCells, numBasis, numpts, dimension);
  basis_pointer->getValues(basisgrads, evalpts, Intrepid2::OPERATOR_GRAD);
  
  DRV jacobian("jacobian", numCells, numpts, dimension, dimension);
  DRV jacobInv("jacobInv", numCells, numpts, dimension, dimension);
  CellTools::setJacobian(jacobian, evalpts, nodes, *cellTopo);
  CellTools::setJacobianInv(jacobInv, jacobian);
  FuncTools::HGRADtransformGRAD(basisgrads_Transformed, jacobInv, basisgrads);
  DRV basisgrads_to("basisgrads_Transformed", numCells, numBasis, numpts, dimension);
  if (basis_pointer->requireOrientation()) {
    OrientTools::modifyBasisByOrientation(basisgrads_to, basisgrads_Transformed,
                                      orientation, basis_pointer.get());
  }
  else {
    basisgrads_to = basisgrads_Transformed;
  }
  
  return basisgrads_to;
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the mesh and the discretizations have been defined, we can create and add the physics
// to the DOF manager
/////////////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::buildDOFManagers() {
  
  Teuchos::TimeMonitor localtimer(*dofmgr_timer);
  
  debugger->print("**** Starting discretization::buildDOF ...");
  
  Teuchos::RCP<panzer::ConnManager> conn = mesh->getSTKConnManager();
  
  num_derivs_required = vector<int>(block_names.size(),0);
  
  // DOF manager for the primary variables
  for (size_t set=0; set<physics->set_names.size(); ++set) {
    Teuchos::RCP<panzer::DOFManager> setDOF = Teuchos::rcp(new panzer::DOFManager());
    setDOF->setConnManager(conn,*(comm->getRawMpiComm()));
    setDOF->setOrientationsRequired(true);
    
    for (size_t block=0; block<block_names.size(); ++block) {
      for (size_t j=0; j<physics->var_list[set][block].size(); j++) {
        topo_RCP cellTopo = mesh->getCellTopology(block_names[block]);
        basis_RCP basis_pointer = this->getBasis(dimension, cellTopo,
                                                 physics->types[set][block][j],
                                                 physics->orders[set][block][j]);
        
        Teuchos::RCP<const panzer::Intrepid2FieldPattern> Pattern = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis_pointer));
        
        if (physics->use_DG[set][block][j]) {
          setDOF->addField(block_names[block], physics->var_list[set][block][j], Pattern, panzer::FieldType::DG);
        }
        else {
          setDOF->addField(block_names[block], physics->var_list[set][block][j], Pattern, panzer::FieldType::CG);
        }
        
      }
    }
    
    setDOF->buildGlobalUnknowns();
#ifndef MrHyDE_NO_AD
    for (size_t block=0; block<block_names.size(); ++block) {
      int numGIDs = setDOF->getElementBlockGIDCount(block_names[block]);
      if (numGIDs > num_derivs_required[block]) {
        num_derivs_required[block] = numGIDs;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(numGIDs > MAXDERIVS,std::runtime_error,"Error: MAXDERIVS is not large enough to support the number of degrees of freedom per element on block: " + block_names[block]);
    }
#endif
    if (verbosity>1) {
      if (comm->getRank() == 0) {
        setDOF->printFieldInformation(std::cout);
      }
    }

    // Instead of storing the DOF manager, which holds onto the mesh, we extract what we need
    //DOF.push_back(setDOF);
    Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device> setLIDs = setDOF->getLIDs();
    
    dof_lids.push_back(setLIDs);
    
    {
      vector<GO> owned;
      setDOF->getOwnedIndices(owned);
      Kokkos::View<GO*,HostDevice> owned_kv("owned dofs",owned.size());
      for (size_type i=0; i<owned_kv.extent(0); ++i) {
        owned_kv(i) = owned[i];
      }
      dof_owned.push_back(owned_kv);
    }
    {
      vector<GO> ownedAndShared;
      setDOF->getOwnedAndGhostedIndices(ownedAndShared);
      Kokkos::View<GO*,HostDevice> ownedas_kv("owned dofs",ownedAndShared.size());
      for (size_type i=0; i<ownedas_kv.extent(0); ++i) {
        ownedas_kv(i) = ownedAndShared[i];
      }
      dof_owned_and_shared.push_back(ownedas_kv);
    }

    vector<vector<string> > varlist = physics->var_list[set];
    vector<vector<vector<int> > > set_offsets; // [block][var][dof]
    for (size_t block=0; block<block_names.size(); ++block) {
      vector<vector<int> > celloffsets;
      for (size_t j=0; j<varlist[block].size(); j++) {
        string var = varlist[block][j];
        int num = setDOF->getFieldNum(var);
        vector<int> var_offsets = setDOF->getGIDFieldOffsets(block_names[block],num);

        celloffsets.push_back(var_offsets);
      }
      set_offsets.push_back(celloffsets);
    }
    offsets.push_back(set_offsets);
    

    this->setBCData(set,setDOF);

    this->setDirichletData(set,setDOF);

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
    for (size_t block=0; block<block_names.size(); ++block) {
      totalElem += my_elements[block].extent(0);
    }

    // Make sure the conn is setup for a nodal connectivity
    panzer::NodalFieldPattern pattern(topology);
    oconn->buildConnectivity(pattern);

    // Initialize the orientations vector
    //panzer_orientations.clear();
    panzer_orientations = Kokkos::View<Intrepid2::Orientation*,HostDevice>("panzer orients",totalElem);
  
    using NodeView = Kokkos::View<GO*, Kokkos::DefaultHostExecutionSpace>;
    
    // Add owned orientations
    {
      for (size_t block=0; block<block_names.size(); ++block) {
        for (size_t c=0; c<my_elements[block].extent(0); ++c) {
          size_t elemID = my_elements[block](c);
          const GO * nodes = oconn->getConnectivity(elemID);
          NodeView node_view("nodes",num_nodes_per_cell);
          for (int node=0; node<num_nodes_per_cell; ++node) {
            node_view(node) = nodes[node];
          }
          panzer_orientations(elemID) = Intrepid2::Orientation::getOrientation(topology, node_view);
          
        }
      }
    }
  }
  
  debugger->print("**** Finished discretization::buildDOF");
  
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::setBCData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF) {
  
  Teuchos::TimeMonitor localtimer(*set_bc_timer);
  
  debugger->print("**** Starting DiscretizationInterface::setBCData ...");
  
  bool requires_sideinfo = false;
  if (settings->isSublist("Subgrid")) {
    requires_sideinfo = true;
  }
  
  vector<string> sideSets, nodeSets;
  sideSets = mesh->getSideNames();
  nodeSets = mesh->getNodeNames();
  
  //for (size_t set=0; set<physics->setnames.size(); ++set) {
    vector<vector<string> > varlist = physics->var_list[set];
    //auto currDOF = DOF[set];
    
    vector<Kokkos::View<int****,HostDevice> > set_side_info;
    vector<vector<vector<string> > > set_var_bcs; // [block][var][boundary]
    
    vector<vector<GO> > set_point_dofs;
    vector<vector<vector<LO> > > set_dbc_dofs;
    
    for (size_t block=0; block<block_names.size(); ++block) {
      
      vector<vector<string> > block_var_bcs; // [var][boundary]
      
      topo_RCP cellTopo = mesh->getCellTopology(block_names[block]);
      int numSidesPerElem = 2; // default to 1D for some reason
      if (dimension == 2) {
        numSidesPerElem = cellTopo->getEdgeCount();
      }
      else if (dimension == 3) {
        numSidesPerElem = cellTopo->getFaceCount();
      }
      
      std::string blockID = block_names[block];
      vector<stk::mesh::Entity> stk_meshElems = mesh->getMySTKElements(blockID);
      size_t maxElemLID = 0;
      for (size_t i=0; i<stk_meshElems.size(); i++) {
        size_t lid = mesh->getSTKElementLocalId(stk_meshElems[i]);
        maxElemLID = std::max(lid,maxElemLID);
      }
      std::vector<size_t> localelemmap(maxElemLID+1);
      for (size_t i=0; i<stk_meshElems.size(); i++) {
        size_t lid = mesh->getSTKElementLocalId(stk_meshElems[i]);
        localelemmap[lid] = i;
      }

      Teuchos::ParameterList blocksettings = physics->physics_settings[set][block];
    
      Teuchos::ParameterList dbc_settings = blocksettings.sublist("Dirichlet conditions");
      Teuchos::ParameterList nbc_settings = blocksettings.sublist("Neumann conditions");
      Teuchos::ParameterList fbc_settings = blocksettings.sublist("Far-field conditions");
      Teuchos::ParameterList sbc_settings = blocksettings.sublist("Slip conditions");
      Teuchos::ParameterList flux_settings = blocksettings.sublist("Flux conditions");
      bool use_weak_dbcs = dbc_settings.get<bool>("use weak Dirichlet",false);
      
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
        string var = varlist[block][j];
        vector<string> current_var_bcs(sideSets.size(),"none"); // [boundary]
        
        for (size_t side=0; side<sideSets.size(); side++ ) {
          string sideName = sideSets[side];
          
          vector<stk::mesh::Entity> sideEntities = mesh->getMySTKSides(sideName, blockID);
          
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
            mesh->getSTKSideElements(blockID, sideEntities, local_side_Ids, side_output);
            //panzer_stk::workset_utils::getSideElements(*mesh, blockID, sideEntities, local_side_Ids, side_output);
            
            for (size_t i=0; i<side_output.size(); i++ ) {
              local_elem_Ids.push_back(mesh->getSTKElementLocalId(side_output[i]));
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
          
          if (isDiri && !use_weak_dbcs) {
            vector<stk::mesh::Entity> nodeEntities = mesh->getMySTKNodes(nodeName, blockID);
            vector<GO> elemGIDs;
            
            vector<size_t> local_elem_Ids;
            vector<size_t> local_node_Ids;
            vector<stk::mesh::Entity> side_output;
            mesh->getSTKNodeElements(blockID, nodeEntities, local_node_Ids, side_output);

            for( size_t i=0; i<side_output.size(); i++ ) {
              local_elem_Ids.push_back(mesh->getSTKElementLocalId(side_output[i]));
              size_t localid = localelemmap[local_elem_Ids[i]];
              for (size_t k=0; k<dof_lids[set].extent(1); ++k) {
                GO gid = dof_owned_and_shared[set](dof_lids[set](localid,k));
                //GO gid = dof_owned_and_shared[set][dof_lids[set](localid,k)];
                elemGIDs.push_back(gid);
                //elemGIDs.push_back(dof_gids[set](localid,k));
              }
              //elemGIDs = dof_gids[set][localid];
              //currDOF->getElementGIDs(localid,elemGIDs,blockID);
              block_dbc_dofs.push_back(elemGIDs[offsets[set][block][j][local_node_Ids[i]]]);
            }
          }
          
        }
      }
    
      
      set_var_bcs.push_back(block_var_bcs);
      set_side_info.push_back(currside_info);
      
      std::sort(block_dbc_dofs.begin(), block_dbc_dofs.end());
      block_dbc_dofs.erase(std::unique(block_dbc_dofs.begin(),
                                       block_dbc_dofs.end()), block_dbc_dofs.end());
      
      int localsize = (int)block_dbc_dofs.size();
      int globalsize = 0;
      
      Teuchos::reduceAll<int,int>(*comm,Teuchos::REDUCE_SUM,1,&localsize,&globalsize);
      int gathersize = comm->getSize()*globalsize;
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
      
      Teuchos::gatherAll(*comm, globalsize, &block_dbc_dofs_local[0], gathersize, &block_dbc_dofs_global[0]);
      vector<GO> all_dbcs;
      
      for (int i = 0; i < gathersize; i++) {
        all_dbcs.push_back(block_dbc_dofs_global[i]);
      }
      delete [] block_dbc_dofs_local;
      delete [] block_dbc_dofs_global;
      
      vector<GO> dbc_final;
      {
        vector<GO> ownedAndShared(dof_owned_and_shared[set].extent(0));
        for (size_t i=0; i<ownedAndShared.size(); ++i) {
          ownedAndShared[i] = dof_owned_and_shared[set](i);
        }
      
        sort(all_dbcs.begin(),all_dbcs.end());
        sort(ownedAndShared.begin(),ownedAndShared.end());
        set_intersection(all_dbcs.begin(),all_dbcs.end(),
                         ownedAndShared.begin(),ownedAndShared.end(),
                         back_inserter(dbc_final));
        
        //sort(dof_owned_and_shared[set].begin(),dof_owned_and_shared[set].end());
        //set_intersection(all_dbcs.begin(),all_dbcs.end(),
        //                 dof_owned_and_shared[set].begin(),dof_owned_and_shared[set].end(),
        //                 back_inserter(dbc_final));
      
        set_point_dofs.push_back(dbc_final);
      }
    } // blocks
    
    var_bcs.push_back(set_var_bcs);
    side_info.push_back(set_side_info);
    point_dofs.push_back(set_point_dofs);
    
  //} // sets
  
  debugger->print("**** Finished DiscretizationInterface::setBCData");
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::setDirichletData(const size_t & set, Teuchos::RCP<panzer::DOFManager> & DOF) {
  
  Teuchos::TimeMonitor localtimer(*set_dbc_timer);
  
  debugger->print("**** Starting DiscretizationInterface::setDirichletData ...");
  
  //vector<string> side_names;
  //mesh->getSidesetNames(side_names);
  
  //for (size_t set=0; set<physics->setnames.size(); ++set) {
    
    vector<vector<string> > varlist = physics->var_list[set];
    //auto currDOF = DOF[set];
    
    std::vector<std::vector<std::vector<LO> > > set_dbc_dofs;
    
    for (size_t block=0; block<block_names.size(); ++block) {
      
      std::string blockID = block_names[block];
      
      Teuchos::ParameterList dbc_settings = physics->physics_settings[set][block].sublist("Dirichlet conditions");
      bool use_weak_dbcs = dbc_settings.get<bool>("use weak Dirichlet",false);

      std::vector<std::vector<LO> > block_dbc_dofs;
      
      for (size_t j=0; j<varlist[block].size(); j++) {
        std::string var = varlist[block][j];
        
        int fieldnum = DOF->getFieldNum(var);

        std::vector<LO> var_dofs;
        for (size_t side=0; side<side_names.size(); side++ ) {
          std::string sideName = side_names[side];
          vector<stk::mesh::Entity> sideEntities = mesh->getMySTKSides(sideName, blockID);
          
          bool isDiri = false;
          if (dbc_settings.sublist(var).isParameter("all boundaries") || dbc_settings.sublist(var).isParameter(sideName)) {
            isDiri = true;
            have_dirichlet = true;
          }
          
          if (isDiri  && !use_weak_dbcs) {
            
            vector<size_t>             local_side_Ids;
            vector<stk::mesh::Entity>  side_output;
            vector<size_t>             local_elem_Ids;
            mesh->getSTKSideElements(blockID, sideEntities, local_side_Ids, side_output);
            //panzer_stk::workset_utils::getSideElements(*mesh, blockID, sideEntities,
            //                                           local_side_Ids, side_output);
            
            for( size_t i=0; i<side_output.size(); i++ ) {
              LO local_EID = mesh->getSTKElementLocalId(side_output[i]);
              auto elemLIDs = DOF->getElementLIDs(local_EID);
              const std::pair<vector<int>,vector<int> > SideIndex = DOF->getGIDFieldOffsets_closure(blockID, fieldnum,
                                                                                                        dimension-1,
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
    
  //}
  
  debugger->print("**** Finished DiscretizationInterface::setDirichletData");
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

vector<GO> DiscretizationInterface::getGIDs(const size_t & set, const size_t & block, const size_t & elem) {
  vector<GO> gids;
  for (size_t k=0; k<dof_lids[set].extent(1); ++k) {
    GO gid = dof_owned_and_shared[set](dof_lids[set](elem,k));
    //GO gid = dof_owned_and_shared[set][dof_lids[set](elem,k)];
    gids.push_back(gid);
    //gids.push_back(dof_gids[set](elem,k));
  }
  return gids;//dof_gids[set][elem];
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
/////////////////////////////////////////////////////////////////////////////////////////////

DRV DiscretizationInterface::applyOrientation(DRV basis, Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation,
                                              basis_RCP & basis_pointer) {
  
  DRV new_basis;
  if (basis.rank() == 3) {
    new_basis = DRV("basis values", basis.extent(0), basis.extent(1), basis.extent(2));
  }
  else {
    new_basis = DRV("basis values", basis.extent(0), basis.extent(1), basis.extent(2), basis.extent(3));
  }
  if (basis_pointer->requireOrientation()) {    
    OrientTools::modifyBasisByOrientation(new_basis, basis, orientation, basis_pointer.get());
  }
  else {
    new_basis = basis;
  }
  return new_basis;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

Kokkos::View<string**,HostDevice> DiscretizationInterface::getVarBCs(const size_t & set, const size_t & block) {
  
  
  size_t numvars = var_bcs[set][block].size();
  Kokkos::View<string**,HostDevice> bcs("BCs for each variable",numvars, side_names.size());
  for (size_t var=0; var<numvars; ++var) {
    for (size_t side=0; side<side_names.size(); ++side) {
      bcs(var,side) = var_bcs[set][block][var][side];
    }
  }
  return bcs;
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
