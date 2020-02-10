/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "shallowwater.hpp"

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

shallowwater::shallowwater(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
                           const size_t & numip_side_, const int & numElem_,
                           Teuchos::RCP<FunctionManager> & functionManager_,
                           const size_t & blocknum_) :
numip(numip_), numip_side(numip_side_), numElem(numElem_), blocknum(blocknum_) {
  
  label = "shallowwater";
  functionManager = functionManager_;
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  numElem = settings->sublist("Solver").get<int>("Workset size",1);
  
  myvars.push_back("H");
  myvars.push_back("Hu");
  myvars.push_back("Hv");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  
  if (settings->sublist("Physics").get<int>("solver",0) == 1)
    isTD = true;
  else
    isTD = false;
  
  multiscale = settings->isSublist("Subgrid");
  analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  
  numResponses = settings->sublist("Physics").get<int>("numResp_thermal",1);
  useScalarRespFx = settings->sublist("Physics").get<bool>("use scalar response function (thermal)",false);
  
  gravity = settings->sublist("Physics").get<ScalarT>("gravity",9.8);
  
  formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
  
  
  Teuchos::ParameterList fs = settings->sublist("Functions");
  functionManager->addFunction("bathymetry",fs.get<string>("bathymetry","1.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("bathymetry_x",fs.get<string>("bathymetry_x","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("bathymetry_y",fs.get<string>("bathymetry_y","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("bottom friction",fs.get<string>("bottom friction","1.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("Coriolis",fs.get<string>("Coriolis","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("source Hu",fs.get<string>("source Hu","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("source Hv",fs.get<string>("source Hv","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("flux left",fs.get<string>("flux left","0.0"),numElem,numip_side,"side ip",blocknum);
  functionManager->addFunction("flux right",fs.get<string>("flux right","0.0"),numElem,numip_side,"side ip",blocknum);
  functionManager->addFunction("flux top",fs.get<string>("flux top","0.0"),numElem,numip_side,"side ip",blocknum);
  functionManager->addFunction("flux bottom",fs.get<string>("flux bottom","0.0"),numElem,numip_side,"side ip",blocknum);
  functionManager->addFunction("Neumann source Hu",fs.get<string>("Neumann source Hu","0.0"),numElem,numip_side,"side ip",blocknum);
  functionManager->addFunction("Neumann source Hv",fs.get<string>("Neumann source Hv","0.0"),numElem,numip_side,"side ip",blocknum);
  functionManager->addFunction("bathymetry side",fs.get<string>("bathymetry","1.0"),numElem,numip_side,"side_ip",blocknum);
  
}

// ========================================================================================
// ========================================================================================

void shallowwater::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    bath = functionManager->evaluate("bathymetry","ip",blocknum);
    bath_x = functionManager->evaluate("bathymetry_x","ip",blocknum);
    bath_y = functionManager->evaluate("bathymetry_y","ip",blocknum);
    visc = functionManager->evaluate("viscosity","ip",blocknum);
    cor = functionManager->evaluate("Coriolis","ip",blocknum);
    source_Hu = functionManager->evaluate("source Hu","ip",blocknum);
    source_Hv = functionManager->evaluate("source Hv","ip",blocknum);
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  int H_basis_num = wkset->usebasis[H_num];
  Hbasis = wkset->basis[H_basis_num];
  Hbasis_grad = wkset->basis_grad[H_basis_num];
  
  int Hu_basis_num = wkset->usebasis[Hu_num];
  Hubasis = wkset->basis[Hu_basis_num];
  Hubasis_grad = wkset->basis_grad[Hu_basis_num];
  
  int Hv_basis_num = wkset->usebasis[Hv_num];
  Hvbasis = wkset->basis[Hv_basis_num];
  Hvbasis_grad = wkset->basis_grad[Hv_basis_num];
  
  //KokkosTools::print(bath);
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    for (int k=0; k<sol.dimension(2); k++ ) {
      
      AD xi = sol(e,H_num,k,0);
      AD xi_dot = sol_dot(e,H_num,k,0);
      AD H = xi + bath(e,k);
      AD Hu = sol(e,Hu_num,k,0);
      AD Hu_dot = sol_dot(e,Hu_num,k,0);
      
      AD Hv = sol(e,Hv_num,k,0);
      AD Hv_dot = sol_dot(e,Hv_num,k,0);
      
      
      for (int i=0; i<Hbasis.dimension(1); i++ ) {
        
        int resindex = offsets(H_num,i);
        v = Hbasis(e,i,k);
        
        dvdx = Hbasis_grad(e,i,k,0);
        dvdy = Hbasis_grad(e,i,k,1);
        
        res(e,resindex) += xi_dot*v - Hu*dvdx - Hv*dvdy;
        
      }
      
      for (int i=0; i<Hubasis.dimension(1); i++ ) {
        
        int resindex = offsets(Hu_num,i);
        v = Hubasis(e,i,k);
        
        dvdx = Hubasis_grad(e,i,k,0);
        dvdy = Hubasis_grad(e,i,k,1);
        
        res(e,resindex) += Hu_dot*v - (Hu*Hu/H + 0.5*gravity*(H*H-bath(e,k)*bath(e,k)))*dvdx - Hv*Hu/H*dvdy + gravity*xi*bath_x(e,k)*v;
        
      }
      
      for (int i=0; i<Hvbasis.dimension(1); i++ ) {
        
        int resindex = offsets(Hv_num,i);
        v = Hvbasis(e,i,k);
        
        dvdx = Hvbasis_grad(e,i,k,0);
        dvdy = Hvbasis_grad(e,i,k,1);
        
        res(e,resindex) += Hv_dot*v - (Hu*Hu/H)*dvdx - (Hv*Hu/H + 0.5*gravity*(H*H - bath(e,k)*bath(e,k)))*dvdy + gravity*xi*bath_y(e,k)*v;
        
      }
      
    }
    
  });
  
}


// ========================================================================================
// ========================================================================================

void shallowwater::boundaryResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  string sidename = wkset->sidename;
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    if (sidename == "left") {
      nsource = functionManager->evaluate("flux left","side ip",blocknum);
    }
    else if (sidename == "right") {
      nsource = functionManager->evaluate("flux right","side ip",blocknum);
    }
    else if (sidename == "top") {
      nsource = functionManager->evaluate("flux top","side ip",blocknum);
    }
    else if (sidename == "bottom") {
      nsource = functionManager->evaluate("flux bottom","side ip",blocknum);
    }
    //nsource_H = functionManager->evaluate("Neumann source H","side ip",blocknum);
    //nsource_Hu = functionManager->evaluate("Neumann source Hu","side ip",blocknum);
    //nsource_Hv = functionManager->evaluate("Neumann source Hv","side ip",blocknum);
    //bath_side = functionManager->evaluate("bathymetry side","side ip",blocknum);
  }
  
  sideinfo = wkset->sideinfo;
  
  int H_basis_num = wkset->usebasis[H_num];
  Hbasis = wkset->basis_side[H_basis_num];
  Hbasis_grad = wkset->basis_grad_side[H_basis_num];
  
  int Hu_basis_num = wkset->usebasis[Hu_num];
  Hubasis = wkset->basis_side[Hu_basis_num];
  Hubasis_grad = wkset->basis_grad_side[Hu_basis_num];
  
  int Hv_basis_num = wkset->usebasis[Hv_num];
  Hvbasis = wkset->basis_side[Hv_basis_num];
  Hvbasis_grad = wkset->basis_grad_side[Hv_basis_num];
  
  //KokkosTools::print(nsource);
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  ScalarT bb = 1.0;
  int cside = wkset->currentside;
  for (int e=0; e<sideinfo.dimension(0); e++) {
    if (sideinfo(e,H_num,cside,0) == 2) { // Element e is on the side
      for (int k=0; k<Hbasis.dimension(2); k++ ) {
        AD xi = sol_side(e,H_num,k,0);
        AD H = xi + bb;//bath_side(e,k);
        AD Hu = sol_side(e,Hu_num,k,0);
        AD Hv = sol_side(e,Hv_num,k,0);
        
        for (int i=0; i<Hbasis.dimension(1); i++ ) {
          int resindex = offsets(H_num,i);
          ScalarT v = Hbasis(e,i,k);
          //res(e,resindex) += (Hu*normals(e,k,0)+Hv*normals(e,k,1))*v;
          res(e,resindex) += nsource(e,k)*H*v;
        }
      }
    }
    if (sideinfo(e,Hu_num,cside,0) == 2) { // Element e is on the side
      for (int k=0; k<Hubasis.dimension(2); k++ ) {
        AD xi = sol_side(e,H_num,k,0);
        AD H = xi + bb;//bath_side(e,k);
        AD Hu = sol_side(e,Hu_num,k,0);
        AD Hv = sol_side(e,Hv_num,k,0);
        
        for (int i=0; i<Hubasis.dimension(1); i++ ) {
          int resindex = offsets(Hu_num,i);
          ScalarT v = Hubasis(e,i,k);
          //res(e,resindex) += (((Hu*Hu/H + 0.5*gravity*(H*H-bb*bb)))*normals(e,k,0) + Hv*Hu/H*normals(e,k,1))*v;
          res(e,resindex) += (nsource(e,k)*Hu + 0.0*gravity*(H*H-bb*bb)*normals(e,k,0))*v;
        }
      }
    }
    if (sideinfo(e,Hv_num,cside,0) == 2) { // Element e is on the side
      for (int k=0; k<Hvbasis.dimension(2); k++ ) {
        AD xi = sol_side(e,H_num,k,0);
        AD H = xi + bb;//bath_side(e,k);
        AD Hu = sol_side(e,Hu_num,k,0);
        AD Hv = sol_side(e,Hv_num,k,0);
        
        for (int i=0; i<Hvbasis.dimension(1); i++ ) {
          int resindex = offsets(Hv_num,i);
          ScalarT v = Hvbasis(e,i,k);
          //res(e,resindex) += (((Hu*Hu/H))*normals(e,k,0) + (Hv*Hu/H + 0.5*gravity*(H*H - bb*bb))*normals(e,k,1))*v;
          res(e,resindex) += (nsource(e,k)*Hv + 0.0*gravity*(H*H - bb*bb)*normals(e,k,1))*v;
        }
      }
    }
  }
  
}


// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void shallowwater::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void shallowwater::setVars(std::vector<string> & varlist_) {
  varlist = varlist_;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "H")
      H_num = i;
    if (varlist[i] == "Hu")
      Hu_num = i;
    if (varlist[i] == "Hv")
      Hv_num = i;
  }
}
