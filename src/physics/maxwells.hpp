/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MAXWELLS_H
#define MAXWELLS_H

#include "physics_base.hpp"

class maxwells : public physicsbase {
public:
  
  maxwells() {} ;
  
  ~maxwells() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  maxwells(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_, const size_t & numip_side_) :
  numip(numip_), numip_side(numip_side_) {
    
    label = "maxwells";
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    verbosity = settings->sublist("Physics").get<int>("Verbosity",0);
    
    if (settings->sublist("Solver").get<int>("solver",0) == 1)
      isTD = true;
    else
      isTD = false;
    
    
    numResponses = settings->sublist("Physics").get<int>("numResp_maxwells",spaceDim+1); 
    useScalarRespFx = settings->sublist("Physics").get<bool>("use scalar response function (maxwells)",false);
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int numip = wkset->ip.dimension(1);
    int numBasis;
    
    ScalarT x = 0.0;
    ScalarT y = 0.0;
    ScalarT z = 0.0;
    
    ScalarT v, dvdx, dvdy, dvdz;
    AD Hx, dHxdx, dHxdy, dHxdz, Hx_dot;
    AD Hy, dHydx, dHydy, dHydz, Hy_dot;
    AD Hz, dHzdx, dHzdy, dHzdz, Hz_dot;
    
    AD Ex, dExdx, dExdy, dExdz, Ex_dot;
    AD Ey, dEydx, dEydy, dEydz, Ey_dot;
    AD Ez, dEzdx, dEzdy, dEzdz, Ez_dot;
    
    FCAD dens = udfunc->coefficient("density",wkset,false);
    FCAD visc = udfunc->coefficient("viscosity",wkset,false);
    
    // We make the stabilization params and strong residuals AD objects since
    // they are similar to the local residual
    AD tau, res;
    int resindex;
    
    // Order in physicsInterface.hpp: ux, pr, uy, uz
    
    if (spaceDim == 3) { // only implemented in 3D
      
      int Hx_basis = wkset->usebasis[Hx_num];
      int Hy_basis = wkset->usebasis[Hy_num];
      int Hz_basis = wkset->usebasis[Hz_num];
      int Ex_basis = wkset->usebasis[Ex_num];
      int Ey_basis = wkset->usebasis[Ey_num];
      int Ez_basis = wkset->usebasis[Ez_num];
      
      int numHxBasis = wkset->basis[Hx_basis].dimension(1);
      int numHyBasis = wkset->basis[Hy_basis].dimension(1);
      int numHzBasis = wkset->basis[Hz_basis].dimension(1);
      int numExBasis = wkset->basis[Ex_basis].dimension(1);
      int numEyBasis = wkset->basis[Ey_basis].dimension(1);
      int numEzBasis = wkset->basis[Ez_basis].dimension(1);
      
      FCAD source_Hx = udfunc->volumetricSource(label,"Hx",wkset);
      FCAD source_Hy = udfunc->volumetricSource(label,"Hy",wkset);
      FCAD source_Hz = udfunc->volumetricSource(label,"Hz",wkset);
      FCAD source_Ex = udfunc->volumetricSource(label,"Ex",wkset);
      FCAD source_Ey = udfunc->volumetricSource(label,"Ey",wkset);
      FCAD source_Ez = udfunc->volumetricSource(label,"Ez",wkset);
      
      for( int k=0; k<numip; k++ ) {

        // gather up all the information at the integration point

        x = wkset->ip(0,k,0);
        y = wkset->ip(0,k,1);
        z = wkset->ip(0,k,2);
        
        Hx = wkset->local_soln(Hx_num,k,0);
        Hx_dot = wkset->local_soln_dot(Hx_num,k,0);
        dHxdx = wkset->local_soln_grad(Hx_num,k,0);
        dHxdy = wkset->local_soln_grad(Hx_num,k,1);
        dHxdz = wkset->local_soln_grad(Hx_num,k,2);
        
        Hy = wkset->local_soln(Hy_num,k,0);
        Hy_dot = wkset->local_soln_dot(Hy_num,k,0);
        dHydx = wkset->local_soln_grad(Hy_num,k,0);
        dHydy = wkset->local_soln_grad(Hy_num,k,1);
        dHydz = wkset->local_soln_grad(Hy_num,k,2);
        
        Hz = wkset->local_soln(Hz_num,k,0);
        Hz_dot = wkset->local_soln_dot(Hz_num,k,0);
        dHzdx = wkset->local_soln_grad(Hz_num,k,0);
        dHzdy = wkset->local_soln_grad(Hz_num,k,1);
        dHzdz = wkset->local_soln_grad(Hz_num,k,2);
        
        Ex = wkset->local_soln(Ex_num,k,0);
        Ex_dot = wkset->local_soln_dot(Ex_num,k,0);
        dExdx = wkset->local_soln_grad(Ex_num,k,0);
        dExdy = wkset->local_soln_grad(Ex_num,k,1);
        dExdz = wkset->local_soln_grad(Ex_num,k,2);
        
        Ey = wkset->local_soln(Ey_num,k,0);
        Ey_dot = wkset->local_soln_dot(Ey_num,k,0);
        dEydx = wkset->local_soln_grad(Ey_num,k,0);
        dEydy = wkset->local_soln_grad(Ey_num,k,1);
        dEydz = wkset->local_soln_grad(Ey_num,k,2);
        
        Ez = wkset->local_soln(Ez_num,k,0);
        Ez_dot = wkset->local_soln_dot(Ez_num,k,0);
        dEzdx = wkset->local_soln_grad(Ez_num,k,0);
        dEzdy = wkset->local_soln_grad(Ez_num,k,1);
        dEzdz = wkset->local_soln_grad(Ez_num,k,2);
        
        // multiply and add into local residual

        for( int i=0; i<numHxBasis; i++ ) {
          
          // Hx equation
          resindex = wkset->offsets[Hx_num][i];
          v = wkset->basis[Hx_basis](0,i,k);
          dvdx = wkset->basis_grad[Hx_basis](0,i,k,0);
          dvdy = wkset->basis_grad[Hx_basis](0,i,k,1);
          dvdz = wkset->basis_grad[Hx_basis](0,i,k,2);
          
          //wkset->res(resindex) += visc(k)*(duxdx*dvdx + duxdy*dvdy + duxdz*dvdz) +
          //dens(k)*(ux*duxdx + uy*duxdy + uz*duxdz)*v - pr*dvdx - source_ux(k)*v;
          //if (isTD)
          //  wkset->res(resindex) += dens(k)*ux_dot*v;
        }
        
        for( int i=0; i<numHyBasis; i++ ) {
          
          // Hy equation
          resindex = wkset->offsets[Hy_num][i];
          v = wkset->basis[Hy_basis](0,i,k);
          dvdx = wkset->basis_grad[Hy_basis](0,i,k,0);
          dvdy = wkset->basis_grad[Hy_basis](0,i,k,1);
          dvdz = wkset->basis_grad[Hy_basis](0,i,k,2);
          
          //wkset->res(resindex) += visc(k)*(duydx*dvdx + duydy*dvdy + duydz*dvdz) +
          //dens(k)*(ux*duydx + uy*duydy + uz*duydz)*v - pr*dvdy - source_uy(k)*v;
          //if (isTD)
          //  wkset->res(resindex) += dens(k)*uy_dot*v;
          
        }
        
        for( int i=0; i<numHzBasis; i++ ) {
          
          // Hz equation
          resindex = wkset->offsets[Hz_num][i];
          v = wkset->basis[Hz_basis](0,i,k);
          dvdx = wkset->basis_grad[Hz_basis](0,i,k,0);
          dvdy = wkset->basis_grad[Hz_basis](0,i,k,1);
          dvdz = wkset->basis_grad[Hz_basis](0,i,k,2);
          
          //wkset->res(resindex) += visc(k)*(duzdx*dvdx + duzdy*dvdy + duzdz*dvdz) +
          //dens(k)*(ux*duzdx + uy*duzdy + uz*duzdz)*v - pr*dvdz - source_uz(k)*v;
          
          //if (isTD)
          //  wkset->res(resindex) += dens(k)*uz_dot*v;
          
        }
        
        for( int i=0; i<numExBasis; i++ ) {
          
          // Ex equation
          resindex = wkset->offsets[Ex_num][i];
          v = wkset->basis[Ex_basis](0,i,k);
          dvdx = wkset->basis_grad[Ex_basis](0,i,k,0);
          dvdy = wkset->basis_grad[Ex_basis](0,i,k,1);
          dvdz = wkset->basis_grad[Ex_basis](0,i,k,2);
          
          //wkset->res(resindex) += visc(k)*(duxdx*dvdx + duxdy*dvdy + duxdz*dvdz) +
          //dens(k)*(ux*duxdx + uy*duxdy + uz*duxdz)*v - pr*dvdx - source_ux(k)*v;
          //if (isTD)
          //  wkset->res(resindex) += dens(k)*ux_dot*v;
        }
        
        for( int i=0; i<numEyBasis; i++ ) {
          
          // Ey equation
          resindex = wkset->offsets[Ey_num][i];
          v = wkset->basis[Ey_basis](0,i,k);
          dvdx = wkset->basis_grad[Ey_basis](0,i,k,0);
          dvdy = wkset->basis_grad[Ey_basis](0,i,k,1);
          dvdz = wkset->basis_grad[Ey_basis](0,i,k,2);
          
          //wkset->res(resindex) += visc(k)*(duydx*dvdx + duydy*dvdy + duydz*dvdz) +
          //dens(k)*(ux*duydx + uy*duydy + uz*duydz)*v - pr*dvdy - source_uy(k)*v;
          //if (isTD)
          //  wkset->res(resindex) += dens(k)*uy_dot*v;
          
        }
        
        for( int i=0; i<numEzBasis; i++ ) {
          
          // Ez equation
          resindex = wkset->offsets[Ez_num][i];
          v = wkset->basis[Ez_basis](0,i,k);
          dvdx = wkset->basis_grad[Ez_basis](0,i,k,0);
          dvdy = wkset->basis_grad[Ez_basis](0,i,k,1);
          dvdz = wkset->basis_grad[Ez_basis](0,i,k,2);
          
          //wkset->res(resindex) += visc(k)*(duzdx*dvdx + duzdy*dvdy + duzdz*dvdz) +
          //dens(k)*(ux*duzdx + uy*duzdy + uz*duzdz)*v - pr*dvdz - source_uz(k)*v;
          
          //if (isTD)
          //  wkset->res(resindex) += dens(k)*uz_dot*v;
          
        }

      }
    }
  }  
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    
    
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
      if (varlist[i] == "Ex")
        Ex_num = i;
      if (varlist[i] == "Ey")
        Ey_num = i;
      if (varlist[i] == "Ez")
        Ez_num = i;
      if (varlist[i] == "Hx")
      Hx_num = i;
      if (varlist[i] == "Hy")
      Hy_num = i;
      if (varlist[i] == "Hz")
      Hz_num = i;
    }
  }
  
  
  // ========================================================================================
  // return the value of the stabilization parameter 
  // ========================================================================================
  
  AD computeTau(const AD & localdiff, const AD & xvl, const AD & yvl, const AD & zvl, const ScalarT & h) const {
    
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
  
  int getNumResponses() {
    return numResponses;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<string> ResponseFieldNames() const {
    std::vector<string> rf;
    return rf;
  }

  // ========================================================================================
  // ========================================================================================
  
  void setExtraFields(const size_t & numElem_) {
    numElem = numElem_;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<FC> extraFields() const {
    vector<FC> ef;
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD scalarRespFunc(const FCAD & integralResponses, const bool & justDeriv) const {return integralResponses;}
  
  bool useScalarRespFunc() const {return useScalarRespFx;}
  
private:
  
  size_t numip, numip_side;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int Ex_num, Ey_num, Ez_num;
  int Hx_num, Hy_num, Hz_num;
  
  bool isTD, useSUPG, usePSPG;
  int verbosity;
  bool useScalarRespFx;
};

#endif
