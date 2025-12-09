/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

void DiscretizationInterface::setReferenceBasisData(Teuchos::RCP<GroupMetaData> & groupData) {
  
  size_t dimension = groupData->dimension;
  size_t block = groupData->my_block;
  
  auto cellTopo = groupData->cell_topo;
  
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
          basis_node_vals = View_Sc4("basis values", numElem, numb, nodes.extent(1), 1);
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
