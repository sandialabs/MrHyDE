/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "maxwell_hybridized.hpp"

maxwell_HYBRID::maxwell_HYBRID(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
                               const size_t & numip_side_, const int & numElem_,
                               Teuchos::RCP<FunctionManager> & functionManager_,
                               const size_t & blocknum_) :
numip(numip_), numip_side(numip_side_), numElem(numElem_), blocknum(blocknum_) {
  
  label = "maxwell_hybrid";
  functionManager = functionManager_;
  spaceDim = settings->sublist("Mesh").get<int>("dim",3);
  
  // GH Note: it's likely none of this will make sense in the 2D case... should it require 3D?
  myvars.push_back("Ex");
  mybasistypes.push_back("HGRAD-DG");
  if (spaceDim > 1) {
    myvars.push_back("Ey");
    mybasistypes.push_back("HGRAD-DG");
  }
  if (spaceDim > 2) {
    myvars.push_back("Ez");
    mybasistypes.push_back("HGRAD-DG");
  }

  myvars.push_back("Bx");
  mybasistypes.push_back("HGRAD-DG");
  if (spaceDim > 1) {
    myvars.push_back("By");
    mybasistypes.push_back("HGRAD-DG");
  }
  if (spaceDim > 2) {
    myvars.push_back("Bz");
    mybasistypes.push_back("HGRAD-DG");
  }

  myvars.push_back("lambdax");
  mybasistypes.push_back("HFACE")
  if (spaceDim > 1) {
    myvars.push_back("lambday");
    mybasistypes.push_back("HFACE");
  }
  if (spaceDim > 2) {
    myvars.push_back("lambdaz");
    mybasistypes.push_back("HFACE");
  }
  
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("current x",fs.get<string>("current x","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("current y",fs.get<string>("current y","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("current z",fs.get<string>("current z","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("mu",fs.get<string>("mu","1.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("epsilon",fs.get<string>("epsilon","1.0"),numElem,numip,"ip",blocknum);
  
}

// ========================================================================================
// ========================================================================================

void maxwell_HYBRID::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int resindex;
  int Ex_basis = wkset->usebasis[Ex_num];
  int Bx_basis = wkset->usebasis[Bx_num];
  int Ey_basis = wkset->usebasis[Ey_num];
  int By_basis = wkset->usebasis[By_num];
  int Ez_basis = wkset->usebasis[Ez_num];
  int Bz_basis = wkset->usebasis[Bz_num];
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    current_x = functionManager->evaluate("current x","ip",blocknum);
    current_y = functionManager->evaluate("current y","ip",blocknum);
    current_z = functionManager->evaluate("current z","ip",blocknum);
    mu = functionManager->evaluate("mu","ip",blocknum);
    epsilon = functionManager->evaluate("epsilon","ip",blocknum);
  }
  
  //KokkosTools::print(epsilon);
  //KokkosTools::print(mu);
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  basis = wkset->basis[Bx_basis];
  //basis_curl = wkset->basis_curl[B_basis];
  
  // (\varepsilon \partial_t E_h, v)_{T_h}
  // - (H_h, curl(v))_{T_h}
  // (\mu \partial_t H_h, v)_{T_h}
  // - (E_h, curl(v))_{T_h}
}


// ========================================================================================
// ========================================================================================

void maxwell_HYBRID::boundaryResidual() {
  
  // - (\lambda_h, \eta)_{\Gamma_a}
  // - (g^{inc}, \eta)_{\Gamma_a}

}

// ========================================================================================
// The edge (2D) and face (3D) contributions to the residual
// ========================================================================================

void maxwell_HYBRID::faceResidual() {

  // (lambda_h, n x v)_{\partial T_h}
  // - (hat(E_h), n x v)_{\partial T_h}
  
}
// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void maxwell_HYBRID::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void maxwell_HYBRID::setVars(std::vector<string> & varlist_) {
  varlist = varlist_;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "Ex")
      Ex_num = i;
    if (varlist[i] == "Ey")
      Ey_num = i;
    if (varlist[i] == "Ez")
      Ez_num = i;
    if (varlist[i] == "Bx")
      Bx_num = i;
    if (varlist[i] == "By")
      Bx_num = i;
    if (varlist[i] == "Bz")
      Bx_num = i;
    if (varlist[i] == "lambdax")
      lambdax_num = i;
    if (varlist[i] == "lambday")
      lambday_num = i;
    if (varlist[i] == "lambdaz")
      lambdaz_num = i;
  }
}
