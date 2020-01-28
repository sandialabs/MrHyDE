/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef POROUSHDIVHYBRID_H
#define POROUSHDIVHYBRID_H

#include "physics_base.hpp"

static void porousHDIVHYBRIDHelp() {
  cout << "********** Help and Documentation for the Porous (HDIV) Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}


class porousHDIV_HYBRID : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  porousHDIV_HYBRID() {} ;
  
  ~porousHDIV_HYBRID() {};
  
  porousHDIV_HYBRID(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
             const size_t & numip_side_, const int & numElem_,
             Teuchos::RCP<FunctionInterface> & functionManager_,
             const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_), functionManager(functionManager_),
  blocknum(blocknum_) {
    
    label = "porousHDIV-Hybrid";
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    include_edgeface = true;
    
    if (settings->sublist("Physics").isSublist("Active variables")) {
      if (settings->sublist("Physics").sublist("Active variables").isParameter("p")) {
        myvars.push_back("p");
        mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("p","HVOL"));
      }
      if (settings->sublist("Physics").sublist("Active variables").isParameter("u")) {
        myvars.push_back("u");
        mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("u","HDIV-DG"));
      }
      if (settings->sublist("Physics").sublist("Active variables").isParameter("lambda")) {
        myvars.push_back("lambda");
        mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("lambda","HGRAD"));
      }
    }
    else {
      myvars.push_back("p");
      myvars.push_back("u");
      myvars.push_back("lambda");
      mybasistypes.push_back("HVOL");
      mybasistypes.push_back("HDIV-DG");
      mybasistypes.push_back("HGRAD");
    }
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
          ScalarT q = basis(e,i,k,0);
          AD divu = sol_div(e,unum,k);
          int resindex = offsets(pnum,i);
          res(e,resindex) += divu*q - source(e,k)*q;
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
    
    int u_basis = wkset->usebasis[unum];
    
    basis = wkset->basis_side[u_basis];
    
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
            res(e,resindex) -= bsource(e,k)*(vx*nx+vy*ny+vz*nz);
          }
        }
      }
      if (bcs(pnum,cside) == 5) {
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
            AD bval = aux_side(e,auxlambdanum,k);
            res(e,resindex) -= bval*(vx*nx+vy*ny+vz*nz);
          }
        }
      }
    }
  }
  
  // ========================================================================================
  // The edge (2D) and face (3D) contributions to the residual
  // ========================================================================================
  
  void edgeFaceResidual() {
    
    int lambda_basis = wkset->usebasis[lambdanum];
    int u_basis = wkset->usebasis[unum];
    
    // Since normals get recomputed often, this needs to be reset
    normals = wkset->normals;
    
    Teuchos::TimeMonitor localtime(*boundaryResidualFill);
    
    ScalarT vx = 0.0, vy = 0.0, vz = 0.0;
    ScalarT nx = 0.0, ny = 0.0, nz = 0.0;
    
    // include <lambda, v \cdot n> in velocity equation
    basis = wkset->basis_side[u_basis];
    
    for (int e=0; e<basis.dimension(0); e++) {
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
          AD lambda = sol_side(e,lambdanum,k,0);
          int resindex = offsets(unum,i);
          res(e,resindex) += lambda*(vx*nx+vy*ny+vz*nz);
        }
      }
    }
    
    // include -<u \cdot n, mu> in interface equation
    AD ux = 0.0, uy = 0.0, uz = 0.0;
    basis = wkset->basis_side[lambda_basis];
    
    for (int e=0; e<basis.dimension(0); e++) {
      for (int k=0; k<basis.dimension(2); k++ ) {
        for (int i=0; i<basis.dimension(1); i++ ) {
          ux = sol_side(e,unum,k,0);
          nx = normals(e,k,0);
          if (spaceDim>1) {
            uy = sol_side(e,unum,k,1);
            ny = normals(e,k,1);
          }
          if (spaceDim>2) {
            uz = sol_side(e,unum,k,2);
            nz = normals(e,k,2);
          }
          ScalarT mu = basis(e,i,k);
          int resindex = offsets(lambdanum,i);
          res(e,resindex) += (ux*nx+uy*ny+uz*nz)*mu;
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
      if (varlist[i] == "lambda")
        lambdanum = i;
      if (varlist[i] == "dx")
        dxnum = i;
      if (varlist[i] == "dy")
        dynum = i;
      if (varlist[i] == "dz")
        dznum = i;
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setAuxVars(std::vector<string> & auxvarlist) {
    
    for (size_t i=0; i<auxvarlist.size(); i++) {
      if (auxvarlist[i] == "p")
        auxpnum = i;
      if (auxvarlist[i] == "u")
        auxunum = i;
      if (auxvarlist[i] == "lambda")
        auxlambdanum = i;
    }
  }
  
  
  
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;
  
  int spaceDim, numElem, numParams, numResponses, numSteps;
  size_t numip, numip_side, blocknum;
  FDATA source, bsource;
  
  int pnum, unum, lambdanum;
  int auxpnum=-1, auxunum=-1, auxlambdanum=-1;
  int dxnum=-1, dynum=-1, dznum=-1;
  bool isTD, addBiot;
  ScalarT biot_alpha;
  
  vector<string> varlist;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_HYBRID::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_HYBRID::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_HYBRID::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_HYBRID::boundaryResidual() - evaluation of residual");
  
};

#endif
