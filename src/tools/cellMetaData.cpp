/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "cellMetaData.hpp"

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

CellMetaData::CellMetaData(const Teuchos::RCP<Teuchos::ParameterList> & settings,
                           const topo_RCP & cellTopo_,
                           const Teuchos::RCP<physics> & physics_RCP_, const size_t & myBlock_,
                           const size_t & myLevel_, const bool & build_face_terms_,
                           const bool & assemble_face_terms_,
                           const vector<string> & sidenames_, DRV ref_ip_, DRV ref_wts_,
                           DRV ref_side_ip_, DRV ref_side_wts_, vector<string> & basis_types_,
                           vector<basis_RCP> & basis_pointers_,
                           const size_t & num_params,
                           DRV refnodes_) :
cellTopo(cellTopo_), physics_RCP(physics_RCP_), myBlock(myBlock_),
myLevel(myLevel_), build_face_terms(build_face_terms_), assemble_face_terms(assemble_face_terms_),
sidenames(sidenames_), ref_ip(ref_ip_), ref_wts(ref_wts_),
basis_types(basis_types_), basis_pointers(basis_pointers_), numDiscParams(num_params), refnodes(refnodes_) {
  
  Teuchos::TimeMonitor localtimer(*celltimer);
  
  compute_diff = settings->sublist("Postprocess").get<bool>("Compute Difference in Objective", true);
  useFineScale = settings->sublist("Postprocess").get<bool>("Use fine scale sensors",true);
  loadSensorFiles = settings->sublist("Analysis").get<bool>("Load Sensor Files",false);
  writeSensorFiles = settings->sublist("Analysis").get<bool>("Write Sensor Files",false);
  mortar_objective = settings->sublist("Solver").get<bool>("Use Mortar Objective",false);
  
  //if (settings->sublist("Postprocess").get<bool>("write solution", false)) {
    compute_sol_avg = true;
  //}
  
  multiscale = false;
  numnodes = cellTopo->getNodeCount();
  dimension = cellTopo->getDimension();
  
  if (dimension == 2) {
    numSides = cellTopo->getSideCount();
  }
  else if (dimension == 3) {
    numSides = cellTopo->getFaceCount();
  }
  //response_type = "global";
  response_type = settings->sublist("Postprocess").get("response type", "pointwise");
  have_cell_phi = false;
  have_cell_rotation = false;
  have_extra_data = false;
  
  numsideip = ref_side_ip_.extent(0);
  for (size_t s=0; s<numSides; s++) {
    DRV refSidePoints("refSidePoints",numsideip, dimension);
    CellTools::mapToReferenceSubcell(refSidePoints, ref_side_ip_, dimension-1, s, *cellTopo);
    ref_side_ip.push_back(refSidePoints);
    ref_side_wts.push_back(ref_side_wts_);
    
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
    
    ref_side_normals.push_back(refSideNormals);
    ref_side_tangents.push_back(refSideTangents);
    ref_side_tangentsU.push_back(refSideTangentsU);
    ref_side_tangentsV.push_back(refSideTangentsV);
  }
  
  this->setupReferenceBasis();
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

void CellMetaData::setupReferenceBasis() {
  
  for (size_t i=0; i<basis_pointers.size(); i++) {
    
    int numb = basis_pointers[i]->getCardinality();
    int numip = ref_ip.extent(0);
    
    DRV basisvals, basisgrad, basisdiv, basiscurl;
    DRV basisnodes;
    
    if (basis_types[i] == "HGRAD" || basis_types[i] == "HVOL") {
      
      basisvals = DRV("basisvals",numb, numip);
      basis_pointers[i]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
    
      basisnodes = DRV("basisvals",numb, refnodes.extent(0));
      basis_pointers[i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basisgrad = DRV("basisgrad",numb, numip, dimension);
      basis_pointers[i]->getValues(basisgrad, ref_ip, Intrepid2::OPERATOR_GRAD);
      
    }
    else if (basis_types[i] == "HDIV"){
      
      basisvals = DRV("basisvals",numb, numip, dimension);
      basis_pointers[i]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0), dimension);
      basis_pointers[i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basisdiv = DRV("basisdiv",numb, numip);
      basis_pointers[i]->getValues(basisdiv, ref_ip, Intrepid2::OPERATOR_DIV);
      
    }
    else if (basis_types[i] == "HCURL"){
      
      basisvals = DRV("basisvals",numb, numip, dimension);
      basis_pointers[i]->getValues(basisvals, ref_ip, Intrepid2::OPERATOR_VALUE);
      
      basisnodes = DRV("basisvals",numb, refnodes.extent(0), dimension);
      basis_pointers[i]->getValues(basisnodes, refnodes, Intrepid2::OPERATOR_VALUE);
      
      basiscurl = DRV("basiscurl",numb, numip, dimension);
      basis_pointers[i]->getValues(basiscurl, ref_ip, Intrepid2::OPERATOR_CURL);
      
    }
    else if (basis_types[i] == "HFACE"){
      
    }
    ref_basis.push_back(basisvals);
    ref_basis_curl.push_back(basiscurl);
    ref_basis_grad.push_back(basisgrad);
    ref_basis_div.push_back(basisdiv);
    ref_basis_nodes.push_back(basisnodes);
  }
  
  // Compute the basis value and basis grad values on reference element
  // at side ip
  for (size_t s=0; s<numSides; s++) {
    vector<DRV> sbasis, sbasisgrad, sbasisdiv, sbasiscurl;
    for (size_t i=0; i<basis_pointers.size(); i++) {
      int numb = basis_pointers[i]->getCardinality();
      DRV basisvals, basisgrad, basisdiv, basiscurl;
      if (basis_types[i] == "HGRAD" || basis_types[i] == "HVOL" || basis_types[i] == "HFACE"){
        basisvals = DRV("basisvals",numb, numsideip);
        basis_pointers[i]->getValues(basisvals, ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
        
        basisgrad = DRV("basisgrad",numb, numsideip, dimension);
        basis_pointers[i]->getValues(basisgrad, ref_side_ip[s], Intrepid2::OPERATOR_GRAD);
      }
      else if (basis_types[i] == "HDIV"){
        basisvals = DRV("basisvals",numb, numsideip, dimension);
        basis_pointers[i]->getValues(basisvals, ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
      }
      else if (basis_types[i] == "HCURL"){
        basisvals = DRV("basisvals",numb, numsideip, dimension);
        basis_pointers[i]->getValues(basisvals, ref_side_ip[s], Intrepid2::OPERATOR_VALUE);
      }
      sbasis.push_back(basisvals);
      sbasisgrad.push_back(basisgrad);
      sbasisdiv.push_back(basisdiv);
      sbasiscurl.push_back(basiscurl);
    }
    ref_side_basis.push_back(sbasis);
    ref_side_basis_grad.push_back(sbasisgrad);
    ref_side_basis_div.push_back(sbasisdiv);
    ref_side_basis_curl.push_back(sbasiscurl);
  }
  
}
