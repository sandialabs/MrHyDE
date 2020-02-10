/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef PERI_H
#define PERI_H

#include "physics_base.hpp"

class peridynamics : public physicsbase {
public:
  
  peridynamics() {} ;
  
  ~peridynamics() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  peridynamics(Teuchos::RCP<Teuchos::ParameterList> & settings) {
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    PI = 3.141592653589793238463;
    
    // Parameters
    
    numResponses = 1;
  }
  
  // ========================================================================================
  // The volumetric contributions to the residual
  // ========================================================================================
  
  void volumeResidual(const ScalarT & h, const ScalarT & current_time, const FCAD & local_soln,
                      const FCAD & local_solngrad, const FCAD & local_soln_dot, 
                      const FCAD & local_param, const FCAD & local_param_grad, 
                      const FCAD & local_aux, const FCAD & local_aux_grad, 
                      const FC & ip, const vector<FC> & basis, const vector<FC> & basis_grad, 
                      const vector<int> & usebasis, const vector<vector<int> > & offsets,
                      const bool & onlyTransient, FCAD & local_resid) {
    
    // nothing implemented yet
    
  }  
  
  // ========================================================================================
  // ========================================================================================
  // The boundary contributions to the residual
  // ========================================================================================
  
  void boundaryResidual(const ScalarT & h, const ScalarT & current_time, const FCAD & local_soln,
                        const FCAD & local_solngrad, const FCAD & local_soln_dot, 
                        const FCAD & local_param, const FCAD & local_param_grad, 
                        const FCAD & local_aux, const FCAD & local_aux_grad, 
                        const FC & ip, const FC & normals, const vector<FC> & basis, 
                        const vector<FC> & basis_grad, 
                        const vector<int> & usebasis, const vector<vector<int> > & offsets,
                        const int & sidetype, const string & side_name, FCAD & local_resid) { 
    
    // nothing implemented yet
    
  }  
  
  // ========================================================================================
  // ========================================================================================
  
  void edgeResidual(const ScalarT & h, const ScalarT & current_time, const FCAD & local_soln,
                    const FCAD & local_solngrad, const FCAD & local_soln_dot, 
                    const FCAD & local_param, const FCAD & local_param_grad, 
                    const FCAD & local_aux, const FCAD & local_aux_grad, 
                    const FC & ip, const FC & normals, const vector<FC> & basis, 
                    const vector<FC> & basis_grad, 
                    const vector<int> & usebasis, const vector<vector<int> > & offsets,
                    const string & side_name, FCAD & local_resid) { 
    
  }
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux(const FC & ip, const ScalarT & h, const ScalarT & current_time,
                   const FCAD & local_soln, const FCAD & local_solngrad, 
                   const FCAD & local_soln_dot, const FCAD local_param, const FCAD local_aux, 
                   const FC & normals, FCAD & flux) {

  }

  // ========================================================================================
  // Get the value for the Dirichlet boundary data on a given side
  // ========================================================================================
  
  AD getDirichletValue(const string & var, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                       const ScalarT & t, const string & gside,
                       const std::vector<std::vector<AD > > currparams,  
                       const bool & useadjoint) const {
    AD val = 0.0; 
    return val;
  }
  
  // ========================================================================================
  // Get the initial value
  // ========================================================================================
  
  ScalarT getInitialValue(const string & var, const ScalarT & x, const ScalarT & y,
                         const ScalarT & z, const bool & useadjoint) const {
    ScalarT val = 0.0;
    return val;
  }
  
  // ========================================================================================
  // error calculation
  // ========================================================================================
  
  ScalarT trueSolution(const string & var, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                      const ScalarT & time) const {
    ScalarT val = 0.0;
    if (var == "pdx")
      val = 1.0;
    if (var == "pdy")
      val = 2.0;
    return val;
  }
  
  // ========================================================================================
  // response calculation
  // ========================================================================================
  
  FCAD response(const FCAD & local_soln, 
                               const FCAD & local_soln_grad,
                               const DRV & ip, const ScalarT & time,
                               const std::vector<std::vector<AD > > paramvals) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    FCAD resp(numCC,spaceDim,numip);
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        resp(i,0,j) = local_soln(i,pdx_num,j);
        if (spaceDim > 1)
          resp(i,1,j) = local_soln(i,pdy_num,j);
        if (spaceDim > 2)
          resp(i,2,j) = local_soln(i,pdz_num,j);
      }
    }
    
    return resp;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD target(const FC & ip, const ScalarT & time,
                             const std::vector<std::vector<AD > > paramvals) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    FCAD targ(numCC,spaceDim,numip);
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        targ(i,0,j) = 1.0;
        if (spaceDim > 1)
          targ(i,1,j) = 1.0;
        if (spaceDim > 2)
          targ(i,2,j) = 1.0;
      }
    }
    
    return targ;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    // Get the variable numbers for all of the physics that this modules needs to be aware of 
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "pdx")
        pdx_num = i;
      if (varlist[i] == "pdy")
        pdy_num = i;
      if (varlist[i] == "pdz")
        pdz_num = i;
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setUserDefined(Teuchos::RCP<userDefined> & udfunc_) {
    udfunc = udfunc_;
  }
  
  
  // ========================================================================================
  // return the source term (to be multiplied by test_function) 
  // ========================================================================================
  
  ScalarT SourceTerm(const ScalarT & x, const ScalarT & y, const ScalarT & z) const {
    ScalarT val = 0.0;
    return val;
  }
  
  // ========================================================================================
  // return the boundary source term (to be multiplied by test_function) 
  // ========================================================================================
  
  ScalarT boundarySource(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                        const string & side) const {
    ScalarT val = 0.0;
    return val;
  }
  
  // ========================================================================================
  // return the robin coefficient 
  // ========================================================================================
  
  ScalarT robinAlpha(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                    const string & side) const {
    return 0.0;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  int getNumResponses() {
    return numResponses;
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
  
  Teuchos::RCP<userDefined> udfunc;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  
  // Variable numbers for all of the variables this module requires
  int pdx_num, pdy_num, pdz_num;
  //ScalarT PI;
  
};

#endif
