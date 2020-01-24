/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef POROUSHDIV_H
#define POROUSHDIV_H

#include "physics_base.hpp"

static void porousHDIVHelp() {
  cout << "********** Help and Documentation for the Porous (HDIV) Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}


class porousHDIV : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  porousHDIV() {} ;
  
  ~porousHDIV() {};
  
  porousHDIV(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
             const size_t & numip_side_, const int & numElem_,
             Teuchos::RCP<FunctionInterface> & functionManager_,
             const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_), functionManager(functionManager_),
  blocknum(blocknum_) {
    
    label = "porousHDIV";
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    myvars.push_back("p");
    myvars.push_back("u");
    mybasistypes.push_back("HVOL");
    mybasistypes.push_back("HDIV");
    
    dxnum = 0;
    dynum = 0;
    dznum = 0;
    
    // Functions
    Teuchos::ParameterList fs = settings->sublist("Functions");
    
    functionManager->addFunction("source",fs.get<string>("source","0.0"),numElem,numip,"ip",blocknum);
    
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int resindex;
    int p_basis = wkset->usebasis[pnum];
    int u_basis = wkset->usebasis[unum];
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      source = functionManager->evaluate("source","ip",blocknum);
    }
    
    basis = wkset->basis[u_basis];
    basis_div = wkset->basis_div[u_basis];
    
    // (K^-1 u,v) - (p,div v) - src*v (src not added yet)
    parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      
      ScalarT vx = 0.0;
      ScalarT vy = 0.0;
      ScalarT vz = 0.0;
      ScalarT divv = 0.0;
      AD uy = 0.0, uz = 0.0;
      
      for (int k=0; k<sol.dimension(2); k++ ) {
        for (int i=0; i<basis.dimension(1); i++ ) {
          AD p = sol(e,pnum,k,0);
          AD ux = sol(e,unum,k,0);
          
          if (spaceDim > 1) {
            uy = sol(e,unum,k,1);
          }
          if (spaceDim > 2) {
            uz = sol(e,unum,k,2);
          }
          
          vx = basis(e,i,k,0);
          
          if (spaceDim > 1) {
            vy = basis(e,i,k,1);
          }
          if (spaceDim > 2) {
            vz = basis(e,i,k,2);
          }
          divv = basis_div(e,i,k);
          int resindex = offsets(unum,i);
          res(e,resindex) += 1.0*(ux*vx+uy*vy+uz*vz) - p*divv;
          
        }
      }
      
    });
    
    basis = wkset->basis[p_basis];
    
    // -(div u,q) + src*q (src not added yet)
    parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      
      for (int k=0; k<sol.dimension(2); k++ ) {
        for (int i=0; i<basis.dimension(1); i++ ) {
          ScalarT v = basis(e,i,k,0);
          AD divu = sol_div(e,unum,k);
          int resindex = offsets(pnum,i);
          res(e,resindex) += -divu*v + source(e,k)*v;
        }
      }
    });
    
  }
  
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    
    sideinfo = wkset->sideinfo;
    Kokkos::View<int**,AssemblyDevice> bcs = wkset->var_bcs;
    
    int cside = wkset->currentside;
    int sidetype;
    sidetype = bcs(pnum,cside);
    
    basis = wkset->basis_side[unum];
    
    {
      Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
      
      if (sidetype == 1 ) {
        bsource = functionManager->evaluate("Dirichlet p " + wkset->sidename,"side ip",blocknum);
      }
      
    }
    
    // Since normals get recomputed often, this needs to be reset
    normals = wkset->normals;
    
    Teuchos::TimeMonitor localtime(*boundaryResidualFill);
    
    ScalarT vx = 0.0, vy = 0.0, vz = 0.0;
    ScalarT nx = 0.0, ny = 0.0, nz = 0.0;
    for (int e=0; e<basis.dimension(0); e++) {
      if (bcs(pnum,cside) == 1) {
        for (int k=0; k<basis.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            vx = basis(e,i,k,0);
            nx = normals(e,k,0);
            if (spaceDim>1) {
              vy = basis(e,i,k,1);
              ny = normals(e,k,1);
            }
            if (spaceDim>2) {
              vz = basis(e,i,k,2);
              nz = normals(e,k,2);
            }
            int resindex = offsets(unum,i);
            res(e,resindex) += bsource(e,k)*(vx*nx+vy*ny+vz*nz);
          }
        }
      }
    }
  }
  
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux() {
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "p")
        pnum = i;
      if (varlist[i] == "u")
        unum = i;
      if (varlist[i] == "dx")
        dxnum = i;
      if (varlist[i] == "dy")
        dynum = i;
      if (varlist[i] == "dz")
        dznum = i;
    }
  }
  
  
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;
  
  int spaceDim, numElem, numParams, numResponses, numSteps;
  size_t numip, numip_side, blocknum;
  FDATA source, bsource;
  
  int pnum, unum;
  int dxnum,dynum,dznum;
  bool isTD, addBiot;
  ScalarT biot_alpha;
  
  vector<string> varlist;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV::boundaryResidual() - evaluation of residual");
  
};

#endif
