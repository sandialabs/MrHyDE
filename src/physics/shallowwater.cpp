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

#include "shallowwater.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

shallowwater::shallowwater(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_) {
  
  label = "shallowwater";
  spaceDim = 2; // Just 2D
  
  myvars.push_back("H");
  myvars.push_back("Hu");
  myvars.push_back("Hv");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  
  
  //gravity = settings->sublist("Physics").get<ScalarT>("gravity",9.8);
  
  formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
  
}

// ========================================================================================
// ========================================================================================

void shallowwater::defineFunctions(Teuchos::ParameterList & fs,
                                   Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  functionManager->addFunction("bathymetry",fs.get<string>("bathymetry","1.0"),"ip");
  functionManager->addFunction("bathymetry_x",fs.get<string>("bathymetry_x","0.0"),"ip");
  functionManager->addFunction("bathymetry_y",fs.get<string>("bathymetry_y","0.0"),"ip");
  functionManager->addFunction("bottom friction",fs.get<string>("bottom friction","1.0"),"ip");
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","0.0"),"ip");
  functionManager->addFunction("Coriolis",fs.get<string>("Coriolis","0.0"),"ip");
  functionManager->addFunction("source Hu",fs.get<string>("source Hu","0.0"),"ip");
  functionManager->addFunction("source Hv",fs.get<string>("source Hv","0.0"),"ip");
  functionManager->addFunction("flux left",fs.get<string>("flux left","0.0"),"side ip");
  functionManager->addFunction("flux right",fs.get<string>("flux right","0.0"),"side ip");
  functionManager->addFunction("flux top",fs.get<string>("flux top","0.0"),"side ip");
  functionManager->addFunction("flux bottom",fs.get<string>("flux bottom","0.0"),"side ip");
  functionManager->addFunction("Neumann source Hu",fs.get<string>("Neumann source Hu","0.0"),"side ip");
  functionManager->addFunction("Neumann source Hv",fs.get<string>("Neumann source Hv","0.0"),"side ip");
  functionManager->addFunction("bathymetry side",fs.get<string>("bathymetry","1.0"),"side_ip");
  
}

// ========================================================================================
// ========================================================================================

void shallowwater::volumeResidual() {
  
  View_AD2 bath, bath_x, bath_y, visc, cor, bfric, source_Hu, source_Hv;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    bath = functionManager->evaluate("bathymetry","ip");
    bath_x = functionManager->evaluate("bathymetry_x","ip");
    bath_y = functionManager->evaluate("bathymetry_y","ip");
    visc = functionManager->evaluate("viscosity","ip");
    cor = functionManager->evaluate("Coriolis","ip");
    source_Hu = functionManager->evaluate("source Hu","ip");
    source_Hv = functionManager->evaluate("source Hv","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  int H_basis_num = wkset->usebasis[H_num];
  auto Hbasis = wkset->basis[H_basis_num];
  auto Hbasis_grad = wkset->basis_grad[H_basis_num];
  
  int Hu_basis_num = wkset->usebasis[Hu_num];
  auto Hubasis = wkset->basis[Hu_basis_num];
  auto Hubasis_grad = wkset->basis_grad[Hu_basis_num];
  
  int Hv_basis_num = wkset->usebasis[Hv_num];
  auto Hvbasis = wkset->basis[Hv_basis_num];
  auto Hvbasis_grad = wkset->basis_grad[Hv_basis_num];
  
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  //KokkosTools::print(bath);
  
  auto xi = wkset->getData("H");
  auto xi_dot = wkset->getData("H_t");
  
  auto Hu = wkset->getData("Hu");
  auto Hu_dot = wkset->getData("Hu_t");
  
  auto Hv = wkset->getData("Hv");
  auto Hv_dot = wkset->getData("Hv_t");
  
  auto Hoff  = subview(wkset->offsets, H_num,  ALL());
  auto Huoff = subview(wkset->offsets, Hu_num, ALL());
  auto Hvoff = subview(wkset->offsets, Hv_num, ALL());
  
  parallel_for("SW volume resid",RangePolicy<AssemblyExec>(0,Hbasis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    ScalarT gravity = 9.8;
    for (size_type pt=0; pt<Hbasis.extent(2); pt++ ) {
      
      AD f = xi_dot(elem,pt)*wts(elem,pt);
      AD Fx = -Hu(elem,pt)*wts(elem,pt);
      AD Fy = -Hv(elem,pt)*wts(elem,pt);
      for (size_type dof=0; dof<Hbasis.extent(1); dof++ ) {
        res(elem,Hoff(dof)) += f*Hbasis(elem,dof,pt,0) + Fx*Hbasis_grad(elem,dof,pt,0) + Fy*Hbasis_grad(elem,dof,pt,1);
      }
      
      AD H = xi(elem,pt) + bath(elem,pt);
      AD uHu = Hu(elem,pt)*Hu(elem,pt)/H;
      AD uHv = Hu(elem,pt)*Hv(elem,pt)/H;
      AD vHv = Hv(elem,pt)*Hv(elem,pt)/H;
      
      f = (Hu_dot(elem,pt) - gravity*xi(elem,pt)*bath_x(elem,pt))*wts(elem,pt);
      Fx = -(uHu + 0.5*gravity*(H*H-bath(elem,pt)*bath(elem,pt)))*wts(elem,pt);
      Fy = -uHv*wts(elem,pt);
      for (size_type dof=0; dof<Hubasis.extent(1); dof++ ) {
        res(elem,Huoff(dof)) += f*Hubasis(elem,dof,pt,0) + Fx*Hubasis_grad(elem,dof,pt,0) + Fy*Hubasis_grad(elem,dof,pt,1);
      }
      
      f = (Hv_dot(elem,pt) - gravity*xi(elem,pt)*bath_y(elem,pt))*wts(elem,pt);
      Fx = -uHv*wts(elem,pt);
      Fy = -(vHv + 0.5*gravity*(H*H-bath(elem,pt)*bath(elem,pt)))*wts(elem,pt);
      
      for (size_type dof=0; dof<Hvbasis.extent(1); dof++ ) {
        res(elem,Hvoff(dof)) += f*Hubasis(elem,dof,pt,0) + Fx*Hubasis_grad(elem,dof,pt,0) + Fy*Hubasis_grad(elem,dof,pt,1);
      }
      
    }
    
  });
  
}


// ========================================================================================
// ========================================================================================

void shallowwater::boundaryResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  /*
  string sidename = wkset->sidename;
  
  View_AD2 nsource, nsource_Hu, nsource_Hv, bath_side;
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    if (sidename == "left") {
      nsource = functionManager->evaluate("flux left","side ip");
    }
    else if (sidename == "right") {
      nsource = functionManager->evaluate("flux right","side ip");
    }
    else if (sidename == "top") {
      nsource = functionManager->evaluate("flux top","side ip");
    }
    else if (sidename == "bottom") {
      nsource = functionManager->evaluate("flux bottom","side ip");
    }
    
  }
  
  auto sideinfo = wkset->sideinfo;
  
  int H_basis_num = wkset->usebasis[H_num];
  auto Hbasis = wkset->basis_side[H_basis_num];
  auto Hbasis_grad = wkset->basis_grad_side[H_basis_num];
  
  int Hu_basis_num = wkset->usebasis[Hu_num];
  auto Hubasis = wkset->basis_side[Hu_basis_num];
  auto Hubasis_grad = wkset->basis_grad_side[Hu_basis_num];
  
  int Hv_basis_num = wkset->usebasis[Hv_num];
  auto Hvbasis = wkset->basis_side[Hv_basis_num];
  auto Hvbasis_grad = wkset->basis_grad_side[Hv_basis_num];
  
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  auto normals = wkset->normals;
  
  //KokkosTools::print(nsource);
  
  //TMW: this needs to be rewritten for device and without sideinfo
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  ScalarT bb = 1.0;
  ScalarT gravity = 9.8;
  int cside = wkset->currentside;
  for (size_type e=0; e<sideinfo.extent(0); e++) {
    if (sideinfo(e,H_num,cside,0) == 2) { // Element e is on the side
      for (size_type k=0; k<Hbasis.extent(2); k++ ) {
        AD xi = sol_side(e,H_num,k,0);
        AD H = xi + bb;//bath_side(e,k);
        AD Hu = sol_side(e,Hu_num,k,0);
        AD Hv = sol_side(e,Hv_num,k,0);
        
        for (size_type i=0; i<Hbasis.extent(1); i++ ) {
          int resindex = offsets(H_num,i);
          ScalarT v = Hbasis(e,i,k,0);
          //res(e,resindex) += (Hu*normals(e,k,0)+Hv*normals(e,k,1))*v;
          res(e,resindex) += (nsource(e,k)*H*v)*wts(e,k);
        }
      }
    }
    if (sideinfo(e,Hu_num,cside,0) == 2) { // Element e is on the side
      for (size_type k=0; k<Hubasis.extent(2); k++ ) {
        AD xi = sol_side(e,H_num,k,0);
        AD H = xi + bb;//bath_side(e,k);
        AD Hu = sol_side(e,Hu_num,k,0);
        AD Hv = sol_side(e,Hv_num,k,0);
        
        for (size_type i=0; i<Hubasis.extent(1); i++ ) {
          int resindex = offsets(Hu_num,i);
          ScalarT v = Hubasis(e,i,k,0);
          //res(e,resindex) += (((Hu*Hu/H + 0.5*gravity*(H*H-bb*bb)))*normals(e,k,0) + Hv*Hu/H*normals(e,k,1))*v;
          res(e,resindex) += ((nsource(e,k)*Hu + 0.0*gravity*(H*H-bb*bb)*normals(e,k,0))*v)*wts(e,k);
        }
      }
    }
    if (sideinfo(e,Hv_num,cside,0) == 2) { // Element e is on the side
      for (size_type k=0; k<Hvbasis.extent(2); k++ ) {
        AD xi = sol_side(e,H_num,k,0);
        AD H = xi + bb;//bath_side(e,k);
        AD Hu = sol_side(e,Hu_num,k,0);
        AD Hv = sol_side(e,Hv_num,k,0);
        
        for (size_type i=0; i<Hvbasis.extent(1); i++ ) {
          int resindex = offsets(Hv_num,i);
          ScalarT v = Hvbasis(e,i,k,0);
          //res(e,resindex) += (((Hu*Hu/H))*normals(e,k,0) + (Hv*Hu/H + 0.5*gravity*(H*H - bb*bb))*normals(e,k,1))*v;
          res(e,resindex) += ((nsource(e,k)*Hv + 0.0*gravity*(H*H - bb*bb)*normals(e,k,1))*v)*wts(e,k);
        }
      }
    }
  }
  */
}


// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void shallowwater::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void shallowwater::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;

  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "H")
      H_num = i;
    if (varlist[i] == "Hu")
      Hu_num = i;
    if (varlist[i] == "Hv")
      Hv_num = i;
  }
}
