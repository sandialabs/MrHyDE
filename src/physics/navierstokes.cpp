/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "navierstokes.hpp"

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

navierstokes::navierstokes(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
                           const size_t & numip_side_, const int & numElem_,
                           Teuchos::RCP<FunctionManager> & functionManager_,
                           const size_t & blocknum_) :
numip(numip_), numip_side(numip_side_), numElem(numElem_), blocknum(blocknum_) {
  
  label = "navierstokes";
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
  
  useSUPG = settings->sublist("Physics").get<bool>("useSUPG",false);
  usePSPG = settings->sublist("Physics").get<bool>("usePSPG",false);
  T_ambient = settings->sublist("Physics").get<ScalarT>("T_ambient",0.0);
  beta = settings->sublist("Physics").get<ScalarT>("beta",1.0);
  
  have_energy = false;
  
  numResponses = settings->sublist("Physics").get<int>("numResp_navierstokes",spaceDim+1);
  useScalarRespFx = settings->sublist("Physics").get<bool>("use scalar response function (navierstokes)",false);
  
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
  functionManager->addFunction("density",fs.get<string>("density","1.0"),numElem,numip,"ip",blocknum);
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","1.0"),numElem,numip,"ip",blocknum);
  
  
}

// ========================================================================================
// ========================================================================================

void navierstokes::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int numip = wkset->ip.dimension(1);
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
    dens = functionManager->evaluate("density","ip",blocknum);
    visc = functionManager->evaluate("viscosity","ip",blocknum);
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  /////////////////////////////
  // ux equation
  /////////////////////////////
  
  int ux_basis = wkset->usebasis[ux_num];
  basis = wkset->basis[ux_basis];
  basis_grad = wkset->basis_grad[ux_basis];
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    ScalarT dvdz = 0.0;
    
    for (int k=0; k<sol.dimension(2); k++ ) {
      
      AD ux = sol(e,ux_num,k,0);
      AD ux_dot = sol_dot(e,ux_num,k,0);
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
      
      if (have_energy) {
        eval = sol(e,e_num,k,0);
      }
      
      
      for( int i=0; i<basis.dimension(1); i++ ) {
        int resindex = offsets(ux_num,i);
        v = basis(e,i,k);
        dvdx = basis_grad(e,i,k,0);
        if (spaceDim > 1) {
          dvdy = basis_grad(e,i,k,1);
        }
        if (spaceDim > 2) {
          dvdz = basis_grad(e,i,k,2);
        }
        
        res(e,resindex) += dens(e,k)*ux_dot*v + visc(e,k)*(duxdx*dvdx + duxdy*dvdy + duxdz*dvdz) + dens(e,k)*(ux*duxdx + uy*duxdy + uz*duxdz)*v - pr*dvdx - dens(e,k)*source_ux(e,k)*v;
        
        if (have_energy) {
          res(e,resindex) += dens(e,k)*beta*(eval-T_ambient)*source_ux(e,k)*v;
        }
        
        if(useSUPG) {
          AD tau = this->computeTau(visc(e,k), ux, uy, uz, wkset->h(e));
          
          AD stabres = dens(e,k)*ux_dot + dens(e,k)*(ux*duxdx + uy*duxdy + uz*duxdz) + dprdx - dens(e,k)*source_ux(e,k);
          
          if (have_energy) {
            stabres += dens(e,k)*beta*(e-T_ambient)*source_ux(e,k);
          }
          res(e,resindex) += tau*(stabres)*(ux*dvdx + uy*dvdy + uz*dvdz);
          
        }
      }
    }
  });
  
  /////////////////////////////
  // pressure equation
  /////////////////////////////
  
  int pr_basis = wkset->usebasis[pr_num];
  basis = wkset->basis[pr_basis];
  basis_grad = wkset->basis_grad[pr_basis];
  
  parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
    
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    ScalarT dvdz = 0.0;
    
    for( int k=0; k<sol.dimension(2); k++ ) {
      AD ux = sol(e,ux_num,k,0);
      AD ux_dot = sol_dot(e,ux_num,k,0);
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
      
      if (have_energy) {
        eval = sol(e,e_num,k,0);
      }
      
      for( int i=0; i<basis.dimension(1); i++ ) {
        
        int resindex = offsets(pr_num,i);
        v = basis(e,i,k);
        
        res(e,resindex) += (duxdx + duydy + duzdz)*v;
        
        if(usePSPG) {
          dvdx = basis_grad(e,i,k,0);
          
          AD tau = this->computeTau(visc(e,k), ux, uy, uz, wkset->h(e));
          
          AD stabres = dens(e,k)*ux_dot + dens(e,k)*(ux*duxdx + uy*duxdy + uz*duxdz) + dprdx - dens(e,k)*source_ux(e,k);
          
          if (have_energy) {
            stabres += dens(e,k)*(eval-T_ambient)*source_ux(e,k);
          }
          
          res(e,resindex) += tau*(stabres)*dvdx;
          
          if (spaceDim > 1) {
            dvdy = basis_grad(e,i,k,1);
            AD dprdy = sol_grad(e,pr_num,k,1);
            AD uy_dot = sol_dot(e,uy_num,k,0);
            AD duydx = sol_grad(e,uy_num,k,0);
            AD duydy = sol_grad(e,uy_num,k,1);
            AD duydz = sol_grad(e,uy_num,k,2);
            stabres = dens(e,k)*uy_dot + dens(e,k)*(ux*duydx + uy*duydy + uz*duydz) + dprdy - dens(e,k)*source_uy(e,k);
            if (have_energy) {
              stabres += dens(e,k)*(eval-T_ambient)*source_uy(e,k);
            }
            res(e,resindex) += tau*(stabres)*dvdy;
          }
          
          if (spaceDim > 2) {
            dvdz = basis_grad(e,i,k,2);
            AD dprdz = sol_grad(e,pr_num,k,2);
            AD uz_dot = sol_dot(e,uz_num,k,0);
            AD duzdx = sol_grad(e,uz_num,k,0);
            AD duzdy = sol_grad(e,uz_num,k,1);
            AD duzdz = sol_grad(e,uz_num,k,2);
            stabres = dens(e,k)*uz_dot + dens(e,k)*(ux*duzdx + uy*duzdy + uz*duzdz) + dprdz - dens(e,k)*source_uz(e,k);
            if (have_energy) {
              stabres += dens(e,k)*(eval-T_ambient)*source_uz(e,k);
            }
            res(e,resindex) += tau*(stabres)*dvdz;
            
          }
        }
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
    
    parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      
      ScalarT v = 0.0;
      ScalarT dvdx = 0.0;
      ScalarT dvdy = 0.0;
      ScalarT dvdz = 0.0;
      
      for( int k=0; k<sol.dimension(2); k++ ) {
        
        AD ux = sol(e,ux_num,k,0);
        AD uy_dot = sol_dot(e,uy_num,k,0);
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
        
        if (have_energy) {
          eval = sol(e,e_num,k,0);
        }
        
        for( int i=0; i<basis.dimension(1); i++ ) {
          int resindex = offsets(uy_num,i);
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          if (spaceDim > 1) {
            dvdy = basis_grad(e,i,k,1);
          }
          if (spaceDim > 2) {
            dvdz = basis_grad(e,i,k,2);
          }
          
          res(e,resindex) += dens(e,k)*uy_dot*v + visc(e,k)*(duydx*dvdx + duydy*dvdy + duydz*dvdz) + dens(e,k)*(ux*duydx + uy*duydy + uz*duydz)*v - pr*dvdy - dens(e,k)*source_uy(e,k)*v;
          
          if (have_energy) {
            res(e,resindex) += dens(e,k)*beta*(eval-T_ambient)*source_uy(e,k)*v;
          }
          
          if(useSUPG) {
            AD tau = this->computeTau(visc(e,k), ux, uy, uz, wkset->h(e));
            
            AD stabres = dens(e,k)*uy_dot + dens(e,k)*(ux*duydx + uy*duydy + uz*duydz) + dprdy - dens(e,k)*source_uy(e,k);
            
            if (have_energy) {
              stabres += dens(e,k)*beta*(eval-T_ambient)*source_uy(e,k);
            }
            
            res(e,resindex) += tau*(stabres)*(ux*dvdx + uy*dvdy + uz*dvdz);
            
          }
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
    
    parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      
      ScalarT v = 0.0;
      ScalarT dvdx = 0.0;
      ScalarT dvdy = 0.0;
      ScalarT dvdz = 0.0;
      
      for( int k=0; k<sol.dimension(2); k++ ) {
        
        AD ux = sol(e,ux_num,k,0);
        AD uz_dot = sol_dot(e,uz_num,k,0);
        AD duzdx = sol_grad(e,uz_num,k,0);
        
        AD pr = sol(e,pr_num,k,0);
        AD dprdz = sol_grad(e,pr_num,k,2);
        AD uy = sol(e,uy_num,k,0);
        AD duzdy = sol_grad(e,uz_num,k,1);
        AD uz = sol(e,uz_num,k,0);
        AD duzdz = sol_grad(e,uz_num,k,2);
        
        AD eval;
        if (have_energy) {
          eval = sol(e,e_num,k,0);
        }
        
        for( int i=0; i<basis.dimension(1); i++ ) {
          
          int resindex = offsets(uz_num,i);
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          dvdy = basis_grad(e,i,k,1);
          dvdz = basis_grad(e,i,k,2);
          
          res(e,resindex) += dens(e,k)*uz_dot*v + visc(e,k)*(duzdx*dvdx + duzdy*dvdy + duzdz*dvdz) + dens(e,k)*(ux*duzdx + uy*duzdy + uz*duzdz)*v - pr*dvdz - dens(e,k)*source_uz(e,k)*v;
          
          if (have_energy) {
            res(e,resindex) += dens(e,k)*(eval-T_ambient)*source_uz(e,k)*v;
          }
          
          if(useSUPG) {
            AD tau = this->computeTau(visc(e,k), ux, uy, uz, wkset->h(e));
            
            AD stabres = dens(e,k)*uz_dot + dens(e,k)*(ux*duzdx + uy*duzdy + uz*duzdz) + dprdz - dens(e,k)*source_uz(e,k);
            
            if (have_energy) {
              stabres += dens(e,k)*(e-T_ambient)*source_uz(e,k);
            }
            
            res(e,resindex) += tau*(stabres)*(ux*dvdx + uy*dvdy + uz*dvdz);
            
          }
        }
      }
    });
  }
  
}

// ========================================================================================
// ========================================================================================

void navierstokes::boundaryResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void navierstokes::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

void navierstokes::setVars(std::vector<string> & varlist_) {
  varlist = varlist_;
  e_num = -1;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "ux")
      ux_num = i;
    if (varlist[i] == "pr")
      pr_num = i;
    if (varlist[i] == "uy")
      uy_num = i;
    if (varlist[i] == "uz")
      uz_num = i;
    if (varlist[i] == "e")
      e_num = i;
  }
  if (e_num >= 0)
    have_energy = true;
}


// ========================================================================================
// return the value of the stabilization parameter
// ========================================================================================

AD navierstokes::computeTau(const AD & localdiff, const AD & xvl, const AD & yvl, const AD & zvl, const ScalarT & h) const {
  
  ScalarT C1 = 4.0;
  ScalarT C2 = 2.0;
  
  AD nvel;
  if (spaceDim == 1)
    nvel = xvl*xvl;
  else if (spaceDim == 2)
    nvel = xvl*xvl + yvl*yvl;
  else if (spaceDim == 3)
    nvel = xvl*xvl + yvl*yvl + zvl*zvl;
  
  if (nvel > 1E-12)
    nvel = sqrt(nvel);
  
  AD tau;
  tau = 1/(C1*localdiff/h/h + C2*(nvel)/h);
  return tau;
}

// ========================================================================================
// ========================================================================================

vector<string> navierstokes::extraCellFieldNames() const {
  vector<string> ef;// = udfunc->extraCellFieldNames(label);
  //ef.push_back("vorticity");
  return ef;
}

// ========================================================================================
// ========================================================================================

vector<Kokkos::View<ScalarT***,AssemblyDevice>> navierstokes::extraCellFields() {
  vector<Kokkos::View<ScalarT***,AssemblyDevice>> ef;// = udfunc->extraCellFields(label,wkset);
  /*
   DRV wts = wkset->wts;
   
   int numvort = 3;
   if (spaceDim == 2) {
   numvort = 1;
   }
   Kokkos::View<ScalarT***,AssemblyDevice> vort("vorticity",numElem,numvort,1);
   
   if (spaceDim == 2) {
   for (int e=0; e<numElem; e++) {
   ScalarT vol = 0.0;
   for (size_t k=0; k<numip; k++) {
   vol += wts(e,k);
   }
   for (int j=0; j<numip; j++) {
   ScalarT cwt = wts(e,j)/vol;
   ScalarT duxdy = wkset->local_soln_grad(e,ux_num,j,1).val();
   ScalarT duydx = wkset->local_soln_grad(e,uy_num,j,0).val();
   
   vort(e,0,0) += (duydx-duxdy)*(duydx-duxdy)*cwt;
   }
   }
   }
   else if (spaceDim == 3) {
   
   }
   
   ef.push_back(vort);
   */
  return ef;
  
}
