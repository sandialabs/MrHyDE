/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef POROUS_H
#define POROUS_H

#include "physics_base.hpp"

class porous : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  porous() {} ;
  
  ~porous() {};
  
  porous(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
             const size_t & numip_side_, const int & numElem_,
             Teuchos::RCP<FunctionInterface> & functionManager_,
             const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_), functionManager(functionManager_),
  blocknum(blocknum_) {
    
    // Standard data
    label = "porous";
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    myvars.push_back("p");
    mybasistypes.push_back("HGRAD");
    
    // Functions
    Teuchos::ParameterList fs = settings->sublist("Functions");
    
    functionManager->addFunction("source",fs.get<string>("porous source","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("permeability",fs.get<string>("permeability","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("porosity",fs.get<string>("porosity","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference density",fs.get<string>("reference density","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference pressure",fs.get<string>("reference pressure","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("compressibility",fs.get<string>("compressibility","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("gravity",fs.get<string>("gravity","1.0"),numElem,numip,"ip",blocknum);
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int p_basis_num = wkset->usebasis[pnum];
    basis = wkset->basis[p_basis_num];
    basis_grad = wkset->basis_grad[p_basis_num];
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      source = functionManager->evaluate("source","ip",blocknum);
      perm = functionManager->evaluate("permeability","ip",blocknum);
      porosity = functionManager->evaluate("porosity","ip",blocknum);
      viscosity = functionManager->evaluate("viscosity","ip",blocknum);
      densref = functionManager->evaluate("reference density","ip",blocknum);
      pref = functionManager->evaluate("reference pressure","ip",blocknum);
      comp = functionManager->evaluate("compressibility","ip",blocknum);
      gravity = functionManager->evaluate("gravity","ip",blocknum);
    }
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    if (spaceDim == 1) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(pnum,i); // TMW: e_num is not on the assembly device
            AD dens = densref(e,k)*comp(e,k)*(sol(e,pnum,k,0) - pref(e,k));
            
            res(e,resindex) += porosity(e,k)*densref(e,k)*comp(e,k)*sol_dot(e,pnum,k,0)*basis(e,i,k) + // transient term
            perm(e,k)/viscosity(e,k)*dens*(sol_grad(e,pnum,k,0)*basis_grad(e,i,k,0)); // diffusion terms
            
          }
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(pnum,i); // TMW: e_num is not on the assembly device
            AD dens = densref(e,k)*comp(e,k)*(sol(e,pnum,k,0) - pref(e,k));
            
            res(e,resindex) += porosity(e,k)*densref(e,k)*comp(e,k)*sol_dot(e,pnum,k,0)*basis(e,i,k) + // transient term
            perm(e,k)/viscosity(e,k)*dens*(sol_grad(e,pnum,k,0)*basis_grad(e,i,k,0) + sol_grad(e,pnum,k,1)*basis_grad(e,i,k,1)); // diffusion terms
            
          }
        }
      });
    }
    else if (spaceDim == 3) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(pnum,i); // TMW: e_num is not on the assembly device
            
            AD dens = densref(e,k)*comp(e,k)*(sol(e,pnum,k,0) - pref(e,k));
            
            res(e,resindex) += porosity(e,k)*densref(e,k)*comp(e,k)*sol_dot(e,pnum,k,0)*basis(e,i,k) + // transient term
            perm(e,k)/viscosity(e,k)*dens*(sol_grad(e,pnum,k,0)*basis_grad(e,i,k,0) + sol_grad(e,pnum,k,1)*basis_grad(e,i,k,1) +
                                           (sol_grad(e,pnum,k,2) - gravity(e,k)*dens*1.0)*basis_grad(e,i,k,2)); // diffusion terms
            
          }
        }
      });
    }
  }
  
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    
    // Nothing implemented yet
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void edgeResidual() {
    
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
      if (varlist[i] == "p") {
        pnum = i;
      }
    }
  }
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;
  
  int spaceDim, numElem, blocknum;
  size_t numip, numip_side;
  
  int pnum, resindex;
  bool isTD, addBiot;
  ScalarT biot_alpha;
  
  vector<string> varlist;
  
  FDATA perm, porosity, viscosity, densref, pref, comp, gravity, source;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous::computeFlux() - evaluation of flux");
  
};

#endif
