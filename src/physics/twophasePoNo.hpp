/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef TWOPHASEPONO_H
#define TWOPHASEPONO_H

#include "physics_base.hpp"

class twophasePoNo : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  twophasePoNo() {} ;
  
  ~twophasePoNo() {};
  
  twophasePoNo(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
               const size_t & numip_side_, const int & numElem_,
               Teuchos::RCP<FunctionInterface> & functionManager_,
               const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_), functionManager(functionManager_),
  blocknum(blocknum_) {
    
    // Standard data
    label = "twophase";
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    myvars.push_back("Po");
    myvars.push_back("No");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    
    // Functions
    Teuchos::ParameterList fs = settings->sublist("Functions");
    
    functionManager->addFunction("permeability",fs.get<string>("permeability","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("porosity",fs.get<string>("porosity","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("gravity",fs.get<string>("gravity","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("cap press",fs.get<string>("capillary pressure","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("dcap press",fs.get<string>("derivative capillary pressure","0.0"),numElem,numip,"ip",blocknum);
    
    functionManager->addFunction("source oil",fs.get<string>("source oil","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("viscosity oil",fs.get<string>("viscosity oil","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("relative permeability oil",fs.get<string>("relative permeability oil","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference density oil",fs.get<string>("reference density oil","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference pressure oil",fs.get<string>("reference pressure oil","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("compressibility oil",fs.get<string>("compressibility oil","0.0"),numElem,numip,"ip",blocknum);
    
    functionManager->addFunction("source water",fs.get<string>("source water","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("viscosity water",fs.get<string>("viscosity water","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("relative permeability water",fs.get<string>("relative permeability water","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference density water",fs.get<string>("reference density water","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference pressure water",fs.get<string>("reference pressure water","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("compressibility water",fs.get<string>("compressibility water","0.0"),numElem,numip,"ip",blocknum);
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // sol, sol_grad, etc. are set by the physics_base class
    
    // This formulation solves for N_o and P_o
    
    // This does assume that both pw and po use the same basis ... easy to generalize
    
    
    int p_basis_num = wkset->usebasis[Ponum];
    basis = wkset->basis[p_basis_num];
    basis_grad = wkset->basis_grad[p_basis_num];
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      porosity = functionManager->evaluate("porosity","ip",blocknum);
      perm = functionManager->evaluate("permeability","ip",blocknum);
      
      relperm_o = functionManager->evaluate("relative permeability oil","ip",blocknum);
      source_o = functionManager->evaluate("source oil","ip",blocknum);
      viscosity_o = functionManager->evaluate("viscosity oil","ip",blocknum);
      densref_o = functionManager->evaluate("reference density oil","ip",blocknum);
      pref_o = functionManager->evaluate("reference pressure oil","ip",blocknum);
      comp_o = functionManager->evaluate("compressibility oil","ip",blocknum);
      
      relperm_w = functionManager->evaluate("relative permeability water","ip",blocknum);
      source_w = functionManager->evaluate("source water","ip",blocknum);
      viscosity_w = functionManager->evaluate("viscosity water","ip",blocknum);
      densref_w = functionManager->evaluate("reference density water","ip",blocknum);
      pref_w = functionManager->evaluate("reference pressure water","ip",blocknum);
      comp_w = functionManager->evaluate("compressibility water","ip",blocknum);
      
      gravity = functionManager->evaluate("gravity","ip",blocknum);
      cp = functionManager->evaluate("cap press","ip",blocknum);
      dcp = functionManager->evaluate("dcap press","ip",blocknum);
      
    }
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    //AD rho_o, rho_w, S_o, S_w, P_w, dP_o_dt, dS_w_dt, dP_w_dt, drho_w_dt, dN_w_dt;
    //AD dS_w_dx, dP_w_dx, dS_w_dy, dP_w_dy, dS_w_dz, dP_w_dz;
    
    if (spaceDim == 1) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) { // loop over integration points
          AD rho_o = densref_o(e,k)*(1.0+comp_o(e,k)*(sol(e,Ponum,k,0) - pref_o(e,k)));
          AD S_o = sol(e,Ponum,k,0) / rho_o;
          AD S_w = 1.0 - S_o;
          AD dP_o_dt = densref_o(e,k)*(1.0+comp_o(e,k)*sol_dot(e,Ponum,k,0));
          AD dS_w_dt = -1.0/(rho_o*rho_o)*(sol_dot(e,Nonum,k,0)*rho_o - sol(e,Nonum,k,0)*dP_o_dt);
          AD P_w = sol(e,Ponum,k,0) - cp(e,k);
          AD rho_w = densref_w(e,k)*(1.0+comp_w(e,k)*(P_w - pref_w(e,k)));
          AD dP_w_dt = sol_dot(e,Ponum,k,0) - dcp(e,k)*dS_w_dt;
          AD drho_w_dt = densref_w(e,k)*(1.0+comp_w(e,k)*dP_w_dt);
          AD dN_w_dt = S_w*drho_w_dt + dS_w_dt*rho_w;
          AD dS_w_dx = -1.0/(rho_o*rho_o)*(rho_o*sol_grad(e,Nonum,k,0) - sol(e,Nonum,k,0)*densref_o(e,k)*comp_o(e,k)*sol_grad(e,Ponum,k,0));
          AD dP_w_dx = sol_grad(e,Ponum,k,0) - dcp(e,k)*dS_w_dx;
          
          for (int i=0; i<basis.dimension(1); i++ ) { // loop over basis functions
            
            // No equation
            resindex = offsets(Nonum,i);
            
            res(e,resindex) += porosity(e,k)*sol_dot(e,Nonum,k,0)*basis(e,i,k) + // transient term
            perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rho_o*(sol_grad(e,Ponum,k,0)*basis_grad(e,i,k,0)) // diffusion terms
            -source_o(e,k)*basis(e,i,k); // source/well model
            
            // Po equation
            resindex = offsets(Ponum,i);
            
            res(e,resindex) += porosity(e,k)*dN_w_dt*basis(e,i,k) + // transient term
            perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rho_w*(dP_w_dx*basis_grad(e,i,k,0)) // diffusion terms
            -source_w(e,k)*basis(e,i,k); // source/well model
            
          }
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          AD rho_o = densref_o(e,k)*(1.0+comp_o(e,k)*(sol(e,Ponum,k,0) - pref_o(e,k)));
          AD S_o = sol(e,Ponum,k,0) / rho_o;
          AD S_w = 1.0 - S_o;
          AD dP_o_dt = densref_o(e,k)*(1.0+comp_o(e,k)*sol_dot(e,Ponum,k,0));
          AD dS_w_dt = -1.0/(rho_o*rho_o)*(sol_dot(e,Nonum,k,0)*rho_o - sol(e,Nonum,k,0)*dP_o_dt);
          AD P_w = sol(e,Ponum,k,0) - cp(e,k);
          AD rho_w = densref_w(e,k)*(1.0+comp_w(e,k)*(P_w - pref_w(e,k)));
          AD dP_w_dt = sol_dot(e,Ponum,k,0) - dcp(e,k)*dS_w_dt;
          AD drho_w_dt = densref_w(e,k)*(1.0+comp_w(e,k)*dP_w_dt);
          AD dN_w_dt = S_w*drho_w_dt + dS_w_dt*rho_w;
          
          AD dS_w_dx = -1.0/(rho_o*rho_o)*(rho_o*sol_grad(e,Nonum,k,0) - sol(e,Nonum,k,0)*densref_o(e,k)*comp_o(e,k)*sol_grad(e,Ponum,k,0));
          AD dP_w_dx = sol_grad(e,Ponum,k,0) - dcp(e,k)*dS_w_dx;
          AD dS_w_dy = -1.0/(rho_o*rho_o)*(rho_o*sol_grad(e,Nonum,k,1) - sol(e,Nonum,k,0)*densref_o(e,k)*comp_o(e,k)*sol_grad(e,Ponum,k,1));
          AD dP_w_dy = sol_grad(e,Ponum,k,1) - dcp(e,k)*dS_w_dy;
          
          for (int i=0; i<basis.dimension(1); i++ ) { // loop over basis functions
            
            // No equation
            resindex = offsets(Nonum,i);
            
            res(e,resindex) += porosity(e,k)*sol_dot(e,Nonum,k,0)*basis(e,i,k) + // transient term
            perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rho_o*(sol_grad(e,Ponum,k,0)*basis_grad(e,i,k,0) +
                                                             sol_grad(e,Ponum,k,1)*basis_grad(e,i,k,1)) // diffusion terms
            -source_o(e,k)*basis(e,i,k); // source/well model
            
            // Po equation
            resindex = offsets(Ponum,i);
            
            res(e,resindex) += porosity(e,k)*dN_w_dt*basis(e,i,k) + // transient term
            perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rho_w*(dP_w_dx*basis_grad(e,i,k,0) +
                                                             dP_w_dy*basis_grad(e,i,k,1)) // diffusion terms
            -source_w(e,k)*basis(e,i,k); // source/well model
            
          }
        }
      });
    }
    else if (spaceDim == 3) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          AD rho_o = densref_o(e,k)*(1.0+comp_o(e,k)*(sol(e,Ponum,k,0) - pref_o(e,k)));
          AD S_o = sol(e,Ponum,k,0) / rho_o;
          AD S_w = 1.0 - S_o;
          AD dP_o_dt = densref_o(e,k)*(1.0+comp_o(e,k)*sol_dot(e,Ponum,k,0));
          AD dS_w_dt = -1.0/(rho_o*rho_o)*(sol_dot(e,Nonum,k,0)*rho_o - sol(e,Nonum,k,0)*dP_o_dt);
          AD P_w = sol(e,Ponum,k,0) - cp(e,k);
          AD rho_w = densref_w(e,k)*(1.0+comp_w(e,k)*(P_w - pref_w(e,k)));
          AD dP_w_dt = sol_dot(e,Ponum,k,0) - dcp(e,k)*dS_w_dt;
          AD drho_w_dt = densref_w(e,k)*(1.0+comp_w(e,k)*dP_w_dt);
          AD dN_w_dt = S_w*drho_w_dt + dS_w_dt*rho_w;
          
          AD dS_w_dx = -1.0/(rho_o*rho_o)*(rho_o*sol_grad(e,Nonum,k,0) - sol(e,Nonum,k,0)*densref_o(e,k)*comp_o(e,k)*sol_grad(e,Ponum,k,0));
          AD dP_w_dx = sol_grad(e,Ponum,k,0) - dcp(e,k)*dS_w_dx;
          AD dS_w_dy = -1.0/(rho_o*rho_o)*(rho_o*sol_grad(e,Nonum,k,1) - sol(e,Nonum,k,0)*densref_o(e,k)*comp_o(e,k)*sol_grad(e,Ponum,k,1));
          AD dP_w_dy = sol_grad(e,Ponum,k,0) - dcp(e,k)*dS_w_dx;
          AD dS_w_dz = -1.0/(rho_o*rho_o)*(rho_o*sol_grad(e,Nonum,k,2) - sol(e,Nonum,k,0)*densref_o(e,k)*comp_o(e,k)*sol_grad(e,Ponum,k,2));
          AD dP_w_dz = sol_grad(e,Ponum,k,0) - dcp(e,k)*dS_w_dx;
          
          for (int i=0; i<basis.dimension(1); i++ ) { // loop over basis functions
            
            // No equation
            resindex = offsets(Nonum,i);
            
            res(e,resindex) += porosity(e,k)*sol_dot(e,Nonum,k,0)*basis(e,i,k) + // transient term
            perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rho_o*(sol_grad(e,Ponum,k,0)*basis_grad(e,i,k,0) +
                                                             sol_grad(e,Ponum,k,1)*basis_grad(e,i,k,1) +
                                                             sol_grad(e,Ponum,k,2)*basis_grad(e,i,k,2) -
                                                             rho_o*gravity(e,k)*1.0) // diffusion terms
            -source_o(e,k)*basis(e,i,k); // source/well model
            
            // Po equation
            resindex = offsets(Ponum,i);
            
            res(e,resindex) += porosity(e,k)*dN_w_dt*basis(e,i,k) + // transient term
            perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rho_w*(dP_w_dx*basis_grad(e,i,k,0) +
                                                             dP_w_dy*basis_grad(e,i,k,1) +
                                                             dP_w_dz*basis_grad(e,i,k,2) -
                                                             rho_w*gravity(e,k)*1.0) // diffusion terms
            -source_w(e,k)*basis(e,i,k); // source/well model
            
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
      if (varlist[i] == "Po") {
        Ponum = i;
      }
      if (varlist[i] == "No") {
        Nonum = i;
      }
    }
  }
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;
  
  int spaceDim, numElem, blocknum;
  size_t numip, numip_side;
  
  int Ponum, Nonum, resindex;
  
  vector<string> varlist;
  
  FDATA perm, porosity, gravity, cp, dcp;
  FDATA relperm_o, source_o, viscosity_o, densref_o, pref_o, comp_o;
  FDATA relperm_w, source_w, viscosity_w, densref_w, pref_w, comp_w;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::computeFlux() - evaluation of flux");
  
};

#endif
