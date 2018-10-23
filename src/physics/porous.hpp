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
  
  porous(Teuchos::RCP<Teuchos::ParameterList> & settings, const size_t & numip_, const size_t & numip_side_) :
  numip(numip_), numip_side(numip_side_) {
    
    label = "porous";
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    if (settings->sublist("Solver").get<int>("solver",0) == 1)
      isTD = true;
    else
      isTD = false;
    
    if (isTD)
      numSteps = 1; // hard-coded for steady state ... need to read in from input file
    else
      numSteps = 1;
    
    addBiot = settings->sublist("Physics").get<bool>("Biot",false);
    biot_alpha = settings->sublist("Physics").get<double>("Biot alpha",0.0);
    numResponses = 1;
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int numip = wkset->ip.dimension(1);
    
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    AD p, dpdx, dpdy, dpdz, p_dot;
    AD ddx_dx_dot, ddy_dy_dot, ddz_dz_dot;
    
    double v = 0.0;
    double dvdx = 0.0;
    double dvdy = 0.0;
    double dvdz = 0.0;
    
    int resindex;
    int p_basis = wkset->usebasis[pnum];
    
    FCAD source = udfunc->volumetricSource(label,"p",wkset);
    FCAD perm = udfunc->coefficient("permeability",wkset,false);
    FCAD poro = udfunc->coefficient("porosity",wkset,false);
    
    for( int k=0; k<numip; k++ ) {
      x = wkset->ip(0,k,0);
      p = wkset->local_soln(pnum,k,0);
      p_dot = wkset->local_soln_dot(pnum,k,0);
      dpdx = wkset->local_soln_grad(pnum,k,0);
      if (spaceDim > 1) {
        y = wkset->ip(0,k,1);
        dpdy = wkset->local_soln_grad(pnum,k,1);
      }
      if (spaceDim > 2) {
        z = wkset->ip(0,k,2);
        dpdz = wkset->local_soln_grad(pnum,k,2);
      }
      
      for( int i=0; i<wkset->basis[p_basis].dimension(1); i++ ) {
        v = wkset->basis[p_basis](0,i,k);
        dvdx = wkset->basis_grad[p_basis](0,i,k,0);
        if (spaceDim > 1) {
          dvdy = wkset->basis_grad[p_basis](0,i,k,1);
        }
        if (spaceDim > 2) {
          dvdz = wkset->basis_grad[p_basis](0,i,k,2);
        }
        
        resindex = wkset->offsets[pnum][i];
        
        wkset->res(resindex) += perm(k)*(dpdx*dvdx + dpdy*dvdy + dpdz*dvdz) - source(k)*v;
        if (isTD) {
          wkset->res(resindex) += poro(k)*p_dot*v;
          if (addBiot) {
            ddx_dx_dot = wkset->local_soln_dot_grad(dxnum,k,0);
            
            wkset->res(resindex) += biot_alpha*ddx_dx_dot*v;
            if (spaceDim > 1) {
              ddy_dy_dot = wkset->local_soln_dot_grad(dynum,k,1);
              wkset->res(resindex) += biot_alpha*ddy_dy_dot*v;
            }
            if (spaceDim > 2) {
              ddz_dz_dot = wkset->local_soln_dot_grad(dznum,k,2);
              wkset->res(resindex) += biot_alpha*ddz_dz_dot*v;
            }
          }
        }
      }
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
      if (varlist[i] == "p")
        pnum = i;
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
  
  std::vector<string> extraFieldNames() const {
    std::vector<string> ef;
    ef.push_back("perm");
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<string> extraCellFieldNames() const {
    vector<string> ef;
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<FC> extraFields() const {
    vector<FC> ef;
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<FC> extraFields(const FC & ip, const double & time) {
    vector<FC> ef;
    /*
    FCAD targ_AD = this->target(ip, time);
    FC targ(targ_AD.dimension(0), targ_AD.dimension(1));
    for (size_t i=0; i<targ_AD.dimension(0); i++) {
      for (size_t j=0; j<targ_AD.dimension(1); j++) {
        targ(i,j) = targ_AD(i,j).val();
      }
    }
    ef.push_back(targ);*/
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<FC> extraCellFields(const FC & ip, const double & time) const {
    vector<FC> ef;
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setExtraFields(const size_t & numElem_) {
    numElem = numElem_;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD scalarRespFunc(const FCAD & integralResponses,
                                    const bool & justDeriv) const {return integralResponses;}
  bool useScalarRespFunc() const {return false;}
  
private:
  
  Teuchos::RCP<UserDefined> udfunc;
  
  int spaceDim, numElem, numParams, numResponses, numSteps;
  size_t numip, numip_side;
  
  int pnum;
  int dxnum,dynum,dznum;
  bool isTD, addBiot;
  double biot_alpha;
  
  vector<string> varlist;
  
};

#endif
