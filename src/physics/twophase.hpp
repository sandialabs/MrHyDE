/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef TWOPHASE_H
#define TWOPHASE_H

#include "physics_base.hpp"

class twophase : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  twophase() {} ;
  
  ~twophase() {};
  
  twophase(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
           const size_t & numip_side_) :
  numip(numip_), numip_side(numip_side_) {
    label = "porous2p";
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    numResponses = 1;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int numip = wkset->ip.dimension(1);
    
    ScalarT x = 0.0;
    ScalarT y = 0.0;
    ScalarT z = 0.0;
    
    AD p, dpdx, dpdy, dpdz;
    AD s, dsdx, dsdy, dsdz;
    
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    ScalarT dvdz = 0.0; 
    
    int resindex;
    int p_basis = wkset->usebasis[pw_num];
    int s_basis = wkset->usebasis[sw_num];
    
    FCAD source = udfunc->volumetricSource(label,"p",wkset);
    FCAD perm = udfunc->coefficient("permeability",wkset,false);
    FCAD poro = udfunc->coefficient("porosity",wkset,false);
    
    
    for( int k=0; k<numip; k++ ) {
      x = wkset->ip(0,k,0);
      p = wkset->local_soln(pw_num,k,0);
      s = wkset->local_soln(sw_num,k,0);
      dpdx = wkset->local_soln_grad(pw_num,k,0);
      dsdx = wkset->local_soln_grad(sw_num,k,0);
      if (spaceDim > 1) { 
        y = wkset->ip(0,k,1);
        dpdy = wkset->local_soln_grad(pw_num,k,1);
        dsdy = wkset->local_soln_grad(sw_num,k,1);
      }
      if (spaceDim > 2) { 
        z = wkset->ip(0,k,2);
        dpdz = wkset->local_soln_grad(pw_num,k,2);
        dsdz = wkset->local_soln_grad(sw_num,k,2);
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
        resindex = wkset->offsets[pw_num][i];
        
        wkset->res(resindex) += perm(k)*(dpdx*dvdx + dpdy*dvdy + dpdz*dvdz) - source(k)*v;
        
      }
      
      for( int i=0; i<wkset->basis[s_basis].dimension(1); i++ ) {
        v = wkset->basis[s_basis](0,i,k);
        dvdx = wkset->basis_grad[s_basis](0,i,k,0);
        if (spaceDim > 1) {
          dvdy = wkset->basis_grad[s_basis](0,i,k,1);
        }
        if (spaceDim > 2) { 
          dvdz = wkset->basis_grad[s_basis](0,i,k,2);
        }
        resindex = wkset->offsets[sw_num][i];
        
        wkset->res(resindex) += 0.0;
        
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
      if (varlist[i] == "pw")
        pw_num = i;
      if (varlist[i] == "sw")
        sw_num = i;
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
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  std::vector<FC > extraFields() const {
    std::vector<FC > ef;
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
  
  int spaceDim, numElem, numParams, numResponses;
  
  int pw_num, sw_num;
  size_t numip, numip_side;
  
  
  vector<string> varlist;
  
};

#endif
