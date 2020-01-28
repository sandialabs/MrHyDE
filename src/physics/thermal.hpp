/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef THERMAL_H
#define THERMAL_H

#include "physics_base.hpp"

static void thermalHelp() {
  cout << "********** Help and Documentation for the Thermal Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class thermal : public physicsbase {
public:
  
  thermal() {} ;
  
  ~thermal() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  thermal(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
          const size_t & numip_side_, const int & numElem_,
          Teuchos::RCP<FunctionInterface> & functionManager_,
          const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_), functionManager(functionManager_),
  blocknum(blocknum_) {
    
    // Standard data
    label = "thermal";
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    myvars.push_back("e");
    mybasistypes.push_back("HGRAD");
    
    // Extra data
    formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
    have_nsvel = false;
    
    // Functions
    Teuchos::ParameterList fs = settings->sublist("Functions");
    
    functionManager->addFunction("thermal source",fs.get<string>("thermal source","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("density",fs.get<string>("density","1.0"),numElem,numip,"ip",blocknum);
    //functionManager->addFunction("thermal Neumann source",fs.get<string>("thermal Neumann source","0.0"),numElem,numip_side,"side ip",blocknum);
    functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),numElem,numip_side,"side ip",blocknum);
    functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),numElem,numip_side,"side ip",blocknum);
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int e_basis_num = wkset->usebasis[e_num];
    basis = wkset->basis[e_basis_num];
    basis_grad = wkset->basis_grad[e_basis_num];
    //offsets = wkset->offsets;
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      source = functionManager->evaluate("thermal source","ip",blocknum);
      diff = functionManager->evaluate("thermal diffusion","ip",blocknum);
      cp = functionManager->evaluate("specific heat","ip",blocknum);
      rho = functionManager->evaluate("density","ip",blocknum);
    }
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    if (spaceDim ==1) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(e_num,i); // TMW: e_num is not on the assembly device
            res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,e_num,k,0)*basis(e,i,k) +
                               diff(e,k)*(sol_grad(e,e_num,k,0)*basis_grad(e,i,k,0)) -
                               source(e,k)*basis(e,i,k);
            if (have_nsvel) { // TMW: have_nsvel is not on the assembly device
              res(e,resindex) += (sol(e,ux_num,k,0)*sol_grad(e,e_num,k,0)*basis(e,i,k));
            }
          }
          
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(e_num,i);
            res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,e_num,k,0)*basis(e,i,k) +
                               diff(e,k)*(sol_grad(e,e_num,k,0)*basis_grad(e,i,k,0) +
                                          sol_grad(e,e_num,k,1)*basis_grad(e,i,k,1)) -
                               source(e,k)*basis(e,i,k);
            if (have_nsvel) {
              res(e,resindex) += (sol(e,ux_num,k,0)*sol_grad(e,e_num,k,0)*basis(e,i,k) + sol(e,uy_num,k,0)*sol_grad(e,e_num,k,1)*basis(e,i,k));
            }
          }
          
        }
      });
    }
    else {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(e_num,i);
            res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,e_num,k,0)*basis(e,i,k) +
                               diff(e,k)*(sol_grad(e,e_num,k,0)*basis_grad(e,i,k,0) +
                                          sol_grad(e,e_num,k,1)*basis_grad(e,i,k,1) +
                                          sol_grad(e,e_num,k,2)*basis_grad(e,i,k,2)) -
                               source(e,k)*basis(e,i,k);
            if (have_nsvel) {
              res(e,resindex) += (sol_grad(e,ux_num,k,0)*basis_grad(e,i,k,0) + sol_grad(e,uy_num,k,1)*basis_grad(e,i,k,1)
                                  + sol_grad(e,uz_num,k,0)*basis_grad(e,i,k,2));
            }
          }
        }
      });
    }
    
  }
  
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    
    sideinfo = wkset->sideinfo;
    Kokkos::View<int**,AssemblyDevice> bcs = wkset->var_bcs;
    
    int cside = wkset->currentside;
    int sidetype;
    sidetype = bcs(e_num,cside);
    
    int e_basis_num = wkset->usebasis[e_num];
    numBasis = wkset->basis_side[e_basis_num].dimension(1);
    basis = wkset->basis_side[e_basis_num];
    basis_grad = wkset->basis_grad_side[e_basis_num];
    
    {
      Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
      
      if (sidetype == 4 ) {
        nsource = functionManager->evaluate("Dirichlet e " + wkset->sidename,"side ip",blocknum);
      }
      else if (sidetype == 2) {
        nsource = functionManager->evaluate("Neumann e " + wkset->sidename,"side ip",blocknum);
      }
      diff_side = functionManager->evaluate("thermal diffusion","side ip",blocknum);
      robin_alpha = functionManager->evaluate("robin alpha","side ip",blocknum);
      
    }
    
    ScalarT sf = formparam;
    if (wkset->isAdjoint) {
      sf = 1.0;
      adjrhs = wkset->adjrhs;
    }
    
    // Since normals get recomputed often, this needs to be reset
    normals = wkset->normals;
    
    Teuchos::TimeMonitor localtime(*boundaryResidualFill);
    
    for (int e=0; e<basis.dimension(0); e++) {
      if (bcs(e_num,cside) == 2) {
        for (int k=0; k<basis.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(e_num,i);
            res(e,resindex) += -nsource(e,k)*basis(e,i,k);
          }
        }
      }
      
      if (bcs(e_num,cside) == 4 || bcs(e_num,cside) == 5) {
        
        for (int k=0; k<basis.dimension(2); k++ ) {
          
          AD eval = sol_side(e,e_num,k,0);
          AD dedx = sol_grad_side(e,e_num,k,0);
          AD dedy, dedz;
          if (spaceDim > 1) {
            dedy = sol_grad_side(e,e_num,k,1);
          }
          if (spaceDim > 2) {
            dedz = sol_grad_side(e,e_num,k,2);
          }
          
          AD lambda;
          
          if (bcs(e_num,cside) == 5) {
            lambda = aux_side(e,auxe_num,k);
          }
          else {
            lambda = nsource(e,k);
          }
          
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(e_num,i);
            v = basis(e,i,k);
            dvdx = basis_grad(e,i,k,0);
            if (spaceDim > 1)
              dvdy = basis_grad(e,i,k,1);
            if (spaceDim > 2)
              dvdz = basis_grad(e,i,k,2);
          
            weakDiriScale = 10.0*diff_side(e,k)/wkset->h(e);
            
            res(e,resindex) += -diff_side(e,k)*dedx*normals(e,k,0)*v - sf*diff_side(e,k)*dvdx*normals(e,k,0)*(eval-lambda) + weakDiriScale*(eval-lambda)*v;
            if (spaceDim > 1) {
              res(e,resindex) += -diff_side(e,k)*dedy*normals(e,k,1)*v - sf*diff_side(e,k)*dvdy*normals(e,k,1)*(eval-lambda);
            }
            if (spaceDim > 2) {
              res(e,resindex) += -diff_side(e,k)*dedz*normals(e,k,2)*v - sf*diff_side(e,k)*dvdz*normals(e,k,2)*(eval-lambda);
            }
            if (wkset->isAdjoint) {
              adjrhs(e,resindex) += sf*diff_side(e,k)*dvdx*normals(e,k,0)*lambda - weakDiriScale*lambda*v;
              if (spaceDim > 1)
                adjrhs(e,resindex) += sf*diff_side(e,k)*dvdy*normals(e,k,1)*lambda;
              if (spaceDim > 2)
                adjrhs(e,resindex) += sf*diff_side(e,k)*dvdz*normals(e,k,2)*lambda;
            }
          }
          
        }
      }
    }
  }
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux() {
    
    ScalarT sf = 1.0;
    if (wkset->isAdjoint) {
      sf = formparam;
    }

    {
      Teuchos::TimeMonitor localtime(*fluxFunc);
      diff_side = functionManager->evaluate("thermal diffusion","side ip",blocknum);
    }
    
    // Since normals get recomputed often, this needs to be reset
    normals = wkset->normals;
    
    {
      Teuchos::TimeMonitor localtime(*fluxFill);
      
      for (int n=0; n<numElem; n++) {
        
        for (size_t i=0; i<wkset->ip_side.dimension(1); i++) {
          penalty = 10.0*diff_side(n,i)/wkset->h(n);
          flux(n,e_num,i) += sf*diff_side(n,i)*sol_grad_side(n,e_num,i,0)*normals(n,i,0) +
                             penalty*(aux_side(n,auxe_num,i)-sol_side(n,e_num,i,0));
          if (spaceDim > 1) {
            flux(n,e_num,i) += sf*diff_side(n,i)*sol_grad_side(n,e_num,i,1)*normals(n,i,1);
          }
          if (spaceDim > 2) {
            flux(n,e_num,i) += sf*diff_side(n,i)*sol_grad_side(n,e_num,i,2)*normals(n,i,2);
          }
        }
      }
    }
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    ux_num = -1;
    uy_num = -1;
    uz_num = -1;
    
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "e")
        e_num = i;
      if (varlist[i] == "ux")
        ux_num = i;
      if (varlist[i] == "uy")
        uy_num = i;
      if (varlist[i] == "uz")
        uz_num = i;
    }
    if (ux_num >=0)
      have_nsvel = true;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setAuxVars(std::vector<string> & auxvarlist) {
    
    for (size_t i=0; i<auxvarlist.size(); i++) {
      if (auxvarlist[i] == "e")
        auxe_num = i;
    }
    
  }
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;

  data grains;
 
  size_t numip, numip_side, blocknum;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int e_num, e_basis, numBasis, ux_num, uy_num, uz_num;
  int auxe_num = -1;
  ScalarT alpha;
  bool isTD;
  //int test, simNum;
  //string simName;
  
  ScalarT v, dvdx, dvdy, dvdz, x, y, z;
  AD e, e_dot, dedx, dedy, dedz, reax, weakDiriScale, lambda, penalty;
  AD ux, uy, uz;
  
  int resindex;
  
  FDATA diff, rho, cp, source, nsource, diff_side, robin_alpha;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  string analysis_type; //to know when parameter is a sample that needs to be transformed
  
  bool useScalarRespFx;
  bool multiscale, have_nsvel;
  ScalarT formparam;
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal::computeFlux() - evaluation of flux");
  
};

#endif
