/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef PHASEFIELD_H
#define PHASEFIELD_H

#include "physics_base.hpp"

class phasefield : public physicsbase {
  public:
  
  phasefield() {} ;
  
  ~phasefield() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  phasefield(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
             const size_t & numip_side_) :
  numip(numip_), numip_side(numip_side_) {
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    PI = 3.141592653589793238463;
    
    if (settings->sublist("Physics").get<int>("solver",0) == 1)
    isTD = true;
    else
    isTD = false;
    
    multiscale = settings->isSublist("Subgrid");
    analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
    
    
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int resindex;
    int numCubPoints = wkset->ip.dimension(1);
    int e_basis = wkset->usebasis[e_num];
    int numBasis = wkset->basis[e_basis].dimension(1);
    
    //    int numBasis = basis.dimension(2);
    // int numCubPoints = ip.dimension(1);
    
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    double v, dvdx, dvdy, dvdz;
    AD  phi, dphidx, dphidy, dphidz, phi_dot;
    double current_time = wkset->time;
    
    
    for( int nPt=0; nPt<numCubPoints; nPt++ ) {
      
      x = wkset->ip(0,nPt,0);
      phi = wkset->local_soln(e_num,nPt,0);
      phi_dot = wkset->local_soln_dot(e_num,nPt,0);
      dphidx = wkset->local_soln_grad(e_num,nPt,0);
      
      if (spaceDim > 1) {
        y = wkset->ip(0,nPt,1);
        dphidy = wkset->local_soln_grad(e_num,nPt,1);
      }
      
      if (spaceDim > 2) {
        z = wkset->ip(0,nPt,2);
        dphidz = wkset->local_soln_grad(e_num,nPt,2);
      }
      
      for( int i=0; i<numBasis; i++ ) {
        resindex = wkset->offsets[e_num][i];
        v = wkset->basis[e_basis](0,i,nPt);
        dvdx = wkset->basis_grad[e_basis](0,i,nPt,0);
        
        if (spaceDim > 1) {
          dvdy = wkset->basis_grad[e_basis](0,i,nPt,1);
        }
        if (spaceDim > 2) {
          dvdz = wkset->basis_grad[e_basis](0,i,nPt,2);
        }
        
        wkset->res(resindex) += L[0]*(4.0*A[0]*(phi*phi*phi - phi)*v +
                                     diff_FAD[0]*diff_FAD[0]*(dphidx*dvdx + dphidy*dvdy + dphidz*dvdz));
        
        wkset->res(resindex) += phi_dot*v;
      }
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    //    int numCubPoints = ip.dimension(1);
    //  int e_basis = usebasis[e_num];
    // int numBasis = basis[e_basis].dimension(1);
    
    //    int numBasis = basis.dimension(2);
    // int numSideCubPoints = ip.dimension(1);
    
    //    FCAD local_resid(1, numBasis);
    
    // Set the parameters
    // double x = 0.0;
    // double y = 0.0;
    // double z = 0.0;
    
    // double v, source, robin_alpha;
    // AD phi, dphidx, dphidy, dphidz;
    // std::string pname = myparams[0];
    // std::vector<AD > diff_FAD = getParameters(params, pname);
    
    // double dvdx;
    
    // for (size_t ee=0; ee<numCC; ee++) {
    //    for( int i=0; i<numBasis; i++ ) {
    //       for( int nPt=0; nPt<numSideCubPoints; nPt++ ) {
    //          v = basis(ee,i,nPt);
    //          x = ip(ee,nPt,0);
    //          phi = local_soln(ee,e_num,nPt);
    //          dphidx = local_solngrad(ee,e_num,nPt,0);
    //          dvdx = basis_grad(ee,i,nPt,0);
    
    //          if (spaceDim > 1) {
    //             y = ip(ee,nPt,1);
    //             dphidy = local_solngrad(ee,e_num,nPt,1);
    //          }
    //          if (spaceDim > 2) {
    //             z = ip(ee,nPt,2);
    //             dphidz = local_solngrad(ee,e_num,nPt,2);
    //          }
    //          source = this->boundarySource(x, y, z, current_time, side_name);
    //          robin_alpha = this->robinAlpha(x, y, z, current_time, side_name);
    
    //          local_resid(ee,i) += -diff_FAD[0]*dphidx*normals(ee,nPt,0)*v;
    //          if (spaceDim > 1) {
    //             local_resid(ee,i) += -diff_FAD[0]*dphidy*normals(ee,nPt,1)*v;
    //          }
    //          if (spaceDim > 2) {
    //             local_resid(ee,i) += -diff_FAD[0]*dphidz*normals(ee,nPt,2)*v;
    //          }
    
    //       }
    //    }
    // }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void edgeResidual() {
    
  }
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux() {
    
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    for (size_t i=0; i<wkset->ip_side.dimension(1); i++) {
      x = wkset->ip_side(0,i,0);
      if (spaceDim > 1)
      y = wkset->ip_side(0,i,1);
      if (spaceDim > 2)
      z = wkset->ip_side(0,i,2);
      
      
      AD diff = DiffusionCoeff(x,y,z);
      AD penalty = 10.0*diff/wkset->h;
      wkset->flux(e_num,i) += diff*wkset->local_soln_grad_side(e_num,i,0)*wkset->normals(0,i,0) + penalty*(wkset->local_aux_side(e_num,i)-wkset->local_soln_side(e_num,i,0));
      if (spaceDim > 1)
      wkset->flux(e_num,i) += diff*wkset->local_soln_grad_side(e_num,i,1)*wkset->normals(0,i,1);
      if (spaceDim > 2)
      wkset->flux(e_num,i) += diff*wkset->local_soln_grad_side(e_num,i,2)*wkset->normals(0,i,2);
    }
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  AD getDirichletValue(const string & var, const double & x, const double & y, const double & z,
                       const double & t, const string & gside, const bool & useadjoint) const {
    AD val = 0.0;
    return val;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  double getInitialValue(const string & var, const double & x, const double & y, const double & z,
                         const bool & useadjoint) const {
    double val = 0.0;
    
    double r0 = 0.0;
    double r1 = 30.0;
    
    r0 = pow((x-50.0)*(x-50.0) + (y-50.0)*(y-50.0),0.5);
    if(r0>r1) {
      val = -1.0;
    } else {
      val =1.0;
    }
    
    return val;
  }
  
  // ========================================================================================
  // Get the initial value
  // ========================================================================================
  
  FC getInitial(const DRV & ip, const string & var, const double & time, const bool & isAdjoint) const {
    int numip = ip.dimension(1);
    FC initial(1,numip);
    
    string dummy ("dummy");
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    for (int i=0; i<numip; i++) {
      x = ip(0,i,0);
      if (spaceDim > 1)
      y = ip(0,i,1);
      if (spaceDim > 2)
      z = ip(0,i,2);
      
      initial(0,i) = getInitialValue(dummy, x, y, z, isAdjoint);
    }
    return initial;
  }
  
  // ========================================================================================
  // error calculation
  // ========================================================================================
  
  double trueSolution(const string & var, const double & x, const double & y, const double & z,
                      const double & time) const {
    double e = sin(2*PI*x);
    if (spaceDim > 1)
    e *= sin(2*PI*y);
    if (spaceDim > 2)
    e *= sin(2*PI*z);
    
    return e;
  }
  
  // ========================================================================================
  // response calculation
  // ========================================================================================
  
  FCAD response(const FCAD & local_soln,
                const FCAD & local_soln_grad,
                const FCAD & local_psoln,
                const FCAD & local_psoln_grad,
                const DRV & ip, const double & time) const {
    int numip = ip.dimension(1);
    FCAD resp(1,numip);
    for (int j=0; j<numip; j++) {
      resp(0,j) = local_soln(e_num,j,0);
    }
    
    return resp;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  //  FCAD target(const FC & ip, const double & time) const {
  FCAD target(const FC & ip, const double & time) {
    int numip = ip.dimension(1);
    FCAD targ(1,numip);
    for (int j=0; j<numip; j++) {
      targ(0,j) = 1.0;
    }
    
    return targ;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "phi")
      e_num = i;
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setUserDefined(Teuchos::RCP<UserDefined> & udfunc_) {
    udfunc = udfunc_;
  }
  
  
  // ========================================================================================
  /* return the source term (to be multiplied by test_function) */
  // ========================================================================================
  
  AD SourceTerm(const double & x, const double & y, const double & z,
                const std::vector<AD > & tsource) const {
    if(spaceDim == 1) {
      return tsource[0]*12*PI*PI*sin(2*PI*x);
    }else if (spaceDim == 2) {
      //return (tsource[0] + 2.0*tsource[1] + 4.0*tsource[2])*8*PI*PI*sin(2*PI*x)*sin(2*PI*y);
      return 8*PI*PI*sin(2*PI*x)*sin(2*PI*y);
    } else {
      return tsource[0]*12*PI*PI*sin(2*PI*x)*sin(2*PI*y)*sin(2*PI*z);
    }
  }
  // ========================================================================================
  /* return the source term (to be multiplied by test_function) */
  // ========================================================================================
  
  double boundarySource(const double & x, const double & y, const double & z, const double & t,
                        const string & side) const {
    
    double val = 0.0;
    
    if (side == "top")
    val = 2*PI*sin(2*PI*x)*cos(2*PI*y);
    
    if (side == "bottom")
    val = -2*PI*sin(2*PI*x)*cos(2*PI*y);
    
    return val;
  }
  
  // ========================================================================================
  /* return the diffusivity coefficient */
  // ========================================================================================
  
  AD DiffusionCoeff(const double & x, const double & y, const double & z) const {
    AD diff = 0.0;
    diff = diff_FAD[0];
    return diff;
  }
  
  // ========================================================================================
  /* return the source term (to be multiplied by test_function) */
  // ========================================================================================
  
  double robinAlpha(const double & x, const double & y, const double & z, const double & t,
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
    //  ef.push_back("target");
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<string> ResponseFieldNames() const {
    std::vector<string> rf;
    return rf;
  }
  
  // ========================================================================================
  // ========================================================================================
  vector<string> extraCellFieldNames() const {
    std::vector<string> ef;
    //ef.push_back("grain");
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  std::vector<FC > extraFields() const {
    //    std::vector<FC > ef;
    vector<FC> ef;
    return ef;
  }
  
  vector<FC> extraFields(const FC & ip, const double & time) {
    std::vector<FC > ef;
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<FC> extraCellFields(const FC & ip, const double & time) const {
    vector<FC> ef;
    AD dval;
    double x,y,z;
    int numip;
    if (ip.rank() == 2)
    numip = ip.dimension(0);
    else
    numip = ip.dimension(1);
    
    FC dvals(numip);
    //FC gvals(numip);
    for (size_t i=0; i<numip; i++) {
      x = ip(i,0);
      if (ip.dimension(1) > 1)
      y = ip(i,1);
      if (ip.dimension(1) > 2)
      z = ip(i,2);
      
      dval = DiffusionCoeff(x,y,z);
      dvals(i) = dval.val();
      
      //gvals(i) = grains.getvalue(x, y, z, time, "grain");
    }
    ef.push_back(dvals);
    //ef.push_back(gvals);
    return ef;
  }
  
  
  // ========================================================================================
  // ========================================================================================
  
  void setExtraFields(const size_t & numElem_) {
    numElem = numElem_;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                        const vector<string> & paramnames) {
    
    for (size_t p=0; p<paramnames.size(); p++) {
      if (paramnames[p] == "thermal_diff")
      diff_FAD = *(params[p]);
      else if (paramnames[p] == "L")
      L = *(params[p]);
      else if (paramnames[p] == "A")
      A = *(params[p]);
      else
      cout << "Parameter not used: " << paramnames[p] << endl;
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD scalarRespFunc(const FCAD & integralResponses,
                      const bool & justDeriv) const {return integralResponses;}
  bool useScalarRespFunc() const {return false;}
  
  private:
  
  Teuchos::RCP<UserDefined> udfunc;
  size_t numip, numip_side;
    
  std::vector<AD> diff_FAD, L, A;
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int e_num;
  double PI;
  double diff, alpha;
  bool isTD;
  string analysis_type; //to know when parameter is a sample that needs to be transformed
  bool multiscale;
};

#endif
