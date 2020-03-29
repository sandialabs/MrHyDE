/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "stokes.hpp"

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

stokes::stokes(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
               const size_t & numip_side_, const int & numElem_,
               Teuchos::RCP<FunctionManager> & functionManager_,
               const size_t & blocknum_) :
numip(numip_), numip_side(numip_side_), numElem(numElem_), blocknum(blocknum_) {
  
  label = "stokes";
  functionManager = functionManager_;
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  
  verbosity = settings->sublist("Physics").get<int>("Verbosity",0);
  numElem = settings->sublist("Solver").get<int>("Workset size",1);
  
  myvars.push_back("ux");
  myvars.push_back("pr");
  if (spaceDim > 1) {
    myvars.push_back("uy");
  }
  if (spaceDim > 2) {
    myvars.push_back("uz");
  }
  
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  if (spaceDim > 1) {
    mybasistypes.push_back("HGRAD");
  }
  if (spaceDim > 2) {
    mybasistypes.push_back("HGRAD");
  }
  
  
  if (settings->sublist("Solver").get<string>("solver","steady-state") == "transient")
    isTD = true;
  else
    isTD = false;
  
  //useSUPG = settings->sublist("Physics").get<bool>("useSUPG",false);
  //usePSPG = settings->sublist("Physics").get<bool>("usePSPG",false);
  T_ambient = settings->sublist("Physics").get<ScalarT>("T_ambient",0.0);
  beta = settings->sublist("Physics").get<ScalarT>("beta",1.0);
  
  //have_energy = false;
  
  numResponses = settings->sublist("Physics").get<int>("numResp_stokes",spaceDim+1);
  useScalarRespFx = settings->sublist("Physics").get<bool>("use scalar response function (stokes)",false);
  
  test = settings->sublist("Physics").get<int>("test",0);
  //test 1: lid-driven cavity
  //test 3: lid-driven cavity, two 'lids'
  //test 30: lid-driven cavity, two 'lids'; provides velocity for msconvdiff
  
  analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
  
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
  functionManager->addFunction("source ux",fs.get<string>("source ux","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("source pr",fs.get<string>("source pr","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("source uy",fs.get<string>("source uy","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("source uz",fs.get<string>("source uz","0.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","1.0"),numElem,numip,"ip",blocknum);
  
  
}

// ========================================================================================
// ========================================================================================

void stokes::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int numip = wkset->ip.extent(1);
  int numBasis;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source_ux = functionManager->evaluate("source ux","ip",blocknum);
    source_pr = functionManager->evaluate("source pr","ip",blocknum);
    if (spaceDim > 1) {
      source_uy = functionManager->evaluate("source uy","ip",blocknum);
    }
    if (spaceDim > 2) {
      source_uz = functionManager->evaluate("source uz","ip",blocknum);
    }
    visc = functionManager->evaluate("viscosity","ip",blocknum);
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  /////////////////////////////
  // ux equation
  /////////////////////////////
  
  int ux_basis = wkset->usebasis[ux_num];
  basis = wkset->basis[ux_basis];
  basis_grad = wkset->basis_grad[ux_basis];
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    ScalarT dvdz = 0.0;
    
    for (int k=0; k<sol.extent(2); k++ ) {
      
      AD ux = sol(e,ux_num,k,0);
      //AD ux_dot = sol_dot(e,ux_num,k,0);
      AD duxdx = sol_grad(e,ux_num,k,0);
      
      AD pr = sol(e,pr_num,k,0);
      AD dprdx = sol_grad(e,pr_num,k,0);
      
      AD uy, duxdy, uz, duxdz, eval;
      
      if (spaceDim > 1) {
        uy = sol(e,uy_num,k,0);
        duxdy = sol_grad(e,ux_num,k,1);
      }
      
      if (spaceDim > 2) {
        uz = sol(e,uz_num,k,0);
        duxdz = sol_grad(e,ux_num,k,2);
      }
      
      //        if (have_energy) {
      //          eval = sol(e,e_num,k,0);
      //        }
      
      
      for( int i=0; i<basis.extent(1); i++ ) {
        int resindex = offsets(ux_num,i);
        v = basis(e,i,k);
        dvdx = basis_grad(e,i,k,0);
        if (spaceDim > 1) {
          dvdy = basis_grad(e,i,k,1);
        }
        if (spaceDim > 2) {
          dvdz = basis_grad(e,i,k,2);
        }
        
        res(e,resindex) += visc(e,k)*(duxdx*dvdx + duxdy*dvdy + duxdz*dvdz) - pr*dvdx - source_ux(e,k)*v;
        
        // what is have_energy? (deleted other instances of it for now)
        //          if (have_energy) {
        //            res(e,resindex) += dens(e,k)*beta*(eval-T_ambient)*source_ux(e,k)*v;
        //          }
        // deleted SUPG stabilization since no need for it for linear equations
      }
    }
  });
  
  /////////////////////////////
  // pressure equation
  /////////////////////////////
  
  int pr_basis = wkset->usebasis[pr_num];
  basis = wkset->basis[pr_basis];
  basis_grad = wkset->basis_grad[pr_basis];
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    ScalarT dvdz = 0.0;
    
    for( int k=0; k<sol.extent(2); k++ ) {
      AD ux = sol(e,ux_num,k,0);
      //AD ux_dot = sol_dot(e,ux_num,k,0);
      AD duxdx = sol_grad(e,ux_num,k,0);
      AD pr = sol(e,pr_num,k,0);
      AD dprdx = sol_grad(e,pr_num,k,0);
      
      AD uy, duxdy, duydy, uz, duxdz, duzdz, eval;
      
      if (spaceDim > 1) {
        uy = sol(e,uy_num,k,0);
        duxdy = sol_grad(e,ux_num,k,1);
        duydy = sol_grad(e,uy_num,k,1);
      }
      
      if (spaceDim > 2) {
        uz = sol(e,uz_num,k,0);
        duxdz = sol_grad(e,ux_num,k,2);
        duzdz = sol_grad(e,uz_num,k,2);
      }
      
      //        if (have_energy) {
      //          eval = sol(e,e_num,k,0);
      //        }
      
      for( int i=0; i<basis.extent(1); i++ ) {
        
        int resindex = offsets(pr_num,i);
        v = basis(e,i,k);
        
        res(e,resindex) += (duxdx + duydy + duzdz)*v;
        
      }
    }
  });
  
  /////////////////////////////
  // uy equation
  /////////////////////////////
  
  if (spaceDim > 1) {
    
    int uy_basis = wkset->usebasis[uy_num];
    basis = wkset->basis[uy_basis];
    basis_grad = wkset->basis_grad[uy_basis];
    
    parallel_for(RangePolicy<AssemblyDevice>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      
      ScalarT v = 0.0;
      ScalarT dvdx = 0.0;
      ScalarT dvdy = 0.0;
      ScalarT dvdz = 0.0;
      
      for( int k=0; k<sol.extent(2); k++ ) {
        
        AD ux = sol(e,ux_num,k,0);
        //AD uy_dot = sol_dot(e,uy_num,k,0);
        AD duydx = sol_grad(e,uy_num,k,0);
        
        AD pr = sol(e,pr_num,k,0);
        AD dprdy = sol_grad(e,pr_num,k,1);
        
        AD uy = sol(e,uy_num,k,0);
        AD duydy = sol_grad(e,uy_num,k,1);
        
        AD uz, duydz, eval;
        if (spaceDim > 2) {
          uz = sol(e,uz_num,k,0);
          duydz = sol_grad(e,uy_num,k,2);
        }
        
        //          if (have_energy) {
        //            eval = sol(e,e_num,k,0);
        //          }
        
        for( int i=0; i<basis.extent(1); i++ ) {
          int resindex = offsets(uy_num,i);
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          if (spaceDim > 1) {
            dvdy = basis_grad(e,i,k,1);
          }
          if (spaceDim > 2) {
            dvdz = basis_grad(e,i,k,2);
          }
          
          res(e,resindex) += visc(e,k)*(duydx*dvdx + duydy*dvdy + duydz*dvdz) - pr*dvdy - source_uy(e,k)*v;
        }
      }
    });
  }
  
  /////////////////////////////
  // uz equation
  /////////////////////////////
  
  if (spaceDim > 2) {
    int uz_basis = wkset->usebasis[uz_num];
    basis = wkset->basis[uz_basis];
    basis_grad = wkset->basis_grad[uz_basis];
    
    parallel_for(RangePolicy<AssemblyDevice>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
      
      ScalarT v = 0.0;
      ScalarT dvdx = 0.0;
      ScalarT dvdy = 0.0;
      ScalarT dvdz = 0.0;
      
      for( int k=0; k<sol.extent(2); k++ ) {
        
        AD ux = sol(e,ux_num,k,0);
        //AD uz_dot = sol_dot(e,uz_num,k,0);
        AD duzdx = sol_grad(e,uz_num,k,0);
        
        AD pr = sol(e,pr_num,k,0);
        AD dprdz = sol_grad(e,pr_num,k,2);
        AD uy = sol(e,uy_num,k,0);
        AD duzdy = sol_grad(e,uz_num,k,1);
        AD uz = sol(e,uz_num,k,0);
        AD duzdz = sol_grad(e,uz_num,k,2);
        
        AD eval;
        //          if (have_energy) {
        //            eval = sol(e,e_num,k,0);
        //          }
        
        for( int i=0; i<basis.extent(1); i++ ) {
          
          int resindex = offsets(uz_num,i);
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          dvdy = basis_grad(e,i,k,1);
          dvdz = basis_grad(e,i,k,2);
          
          res(e,resindex) += visc(e,k)*(duzdx*dvdx + duzdy*dvdy + duzdz*dvdz) - pr*dvdz - source_uz(e,k)*v;
        }
      }
    });
  }
  
}

// ========================================================================================
// ========================================================================================

void stokes::boundaryResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void stokes::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void stokes::setVars(std::vector<string> & varlist_) {
  varlist = varlist_;
  //    e_num = -1;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "ux")
      ux_num = i;
    if (varlist[i] == "pr")
      pr_num = i;
    if (varlist[i] == "uy")
      uy_num = i;
    if (varlist[i] == "uz")
      uz_num = i;
    //      if (varlist[i] == "e")
    //        e_num = i;
  }
  //    if (e_num >= 0)
  //      have_energy = true;
}
