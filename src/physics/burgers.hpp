/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef BURGERS_H
#define BURGERS_H

#include "physics_base.hpp"

class burgers : public physicsbase {
public:
  
  burgers() {} ;
  
  ~burgers() {};
  
  // ========================================================================================
  // ========================================================================================
  
  burgers(Teuchos::RCP<Teuchos::ParameterList> & settings) {
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    // Parameters
    
    useSUPG = settings->sublist("Physics").get<bool>("useSUPG",false);
    
    dt = 0.0;
    if (settings->sublist("Solver").get<int>("solver",0) == 1) 
      isTD = true;
    else
      isTD = false;
    
    if (isTD) {
      double finalT = settings->sublist("Solver").get<double>("finaltime",0.0);
      numSteps = settings->sublist("Solver").get<int>("numSteps",1);
      dt = finalT / numSteps;
    }
    
    
    test = settings->sublist("Physics").get<int>("test",0);
    
    numResponses = 1;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual(const double & h, const double & current_time, const FCAD & local_soln, 
                      const FCAD & local_solngrad, const FCAD & local_soln_dot, 
                      const FCAD & local_param, const FCAD & local_param_grad, 
                      const FCAD & local_aux, const FCAD & local_aux_grad, 
                      const FC & ip, const vector<FC> & basis, const vector<FC> & basis_grad, 
                      const vector<int> & usebasis, const vector<vector<int> > & offsets,
                      const bool & onlyTransient, FCAD & local_resid) {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int numCubPoints = ip.dimension(1);
    
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    AD T, dTdx, dTdy, dTdz, T_dot;
    AD tau, source, sres;
    AD evisc;
    double v, dvdx, dvdy, dvdz;
    
    int T_basis = usebasis[T_num];
    int resindex;
    
    if (spaceDim == 1) {
      for (int nPt=0; nPt<numCubPoints; nPt++ ) {
        x = ip(0,nPt,0);
        T = local_soln(T_num,nPt,0);
        T_dot = local_soln_dot(T_num,nPt,0);
        dTdx = local_solngrad(T_num,nPt,0);
        for (int i=0; i<basis[T_basis].dimension(1); i++ ) {
          v = basis[T_basis](0,i,nPt);
          dvdx = basis_grad[T_basis](0,i,nPt,0);
          evisc = entropyViscosity(T, dTdx, dTdy, dTdz, T_dot, h);
          resindex = offsets[T_num][i];
          
          local_resid(resindex) += T_dot*v + (viscosity[0]+evisc)*dTdx*dvdx - xconv[0]*0.5*T*T*dvdx - source*v;
          
          if(useSUPG) {
            tau = this->computeTau(xconv[0], yconv[0], zconv[0], h);
            sres = tau*(T_dot + xconv[0]*T*dTdx - source);
            local_resid(resindex) += sres*T*xconv[0]*dvdx;
          }
        }
      }
    }
    else if (spaceDim == 2) {
      for (int nPt=0; nPt<numCubPoints; nPt++ ) {
        x = ip(0,nPt,0);
        T = local_soln(T_num,nPt,0);
        T_dot = local_soln_dot(T_num,nPt,0);
        dTdx = local_solngrad(T_num,nPt,0);
        y = ip(0,nPt,1);
        dTdy = local_solngrad(T_num,nPt,1);
        for (int i=0; i<basis[T_basis].dimension(1); i++ ) {
          v = basis[T_basis](0,i,nPt);
          dvdx = basis_grad[T_basis](0,i,nPt,0);
          dvdy = basis_grad[T_basis](0,i,nPt,1);
          evisc = entropyViscosity(T, dTdx, dTdy, dTdz, T_dot, h);
          resindex = offsets[T_num][i];
          
          local_resid(resindex) += T_dot*v + (viscosity[0]+evisc)*dTdx*dvdx - xconv[0]*0.5*T*T*dvdx - source*v
                                  + (viscosity[0]+evisc)*dTdy*dvdy - yconv[0]*0.5*T*T*dvdy;
          
          if(useSUPG) {
            tau = this->computeTau(xconv[0], yconv[0], zconv[0], h);
            sres = tau*(T_dot + xconv[0]*T*dTdx + yconv[0]*T*dTdy + zconv[0]*T*dTdz - source);
            local_resid(resindex) += sres*T*xconv[0]*dvdx;
            local_resid(resindex) += sres*T*yconv[0]*dvdy;
          }
        }
      }
      
    }
    else if (spaceDim == 3) {
      for (int nPt=0; nPt<numCubPoints; nPt++ ) {
        x = ip(0,nPt,0);
        T = local_soln(T_num,nPt,0);
        T_dot = local_soln_dot(T_num,nPt,0);
        dTdx = local_solngrad(T_num,nPt,0);
        y = ip(0,nPt,1);
        dTdy = local_solngrad(T_num,nPt,1);
        z = ip(0,nPt,2);
        dTdz = local_solngrad(T_num,nPt,2);
        
        for (int i=0; i<basis[T_basis].dimension(1); i++ ) {
          // collect the information at the integration point
          v = basis[T_basis](0,i,nPt);
          dvdx = basis_grad[T_basis](0,i,nPt,0);
          
          dvdy = basis_grad[T_basis](0,i,nPt,1);
          dvdz = basis_grad[T_basis](0,i,nPt,2);
          
          evisc = entropyViscosity(T, dTdx, dTdy, dTdz, T_dot, h);
          resindex = offsets[T_num][i];
          
          // sum into the local residual
          local_resid(resindex) += T_dot*v + (viscosity[0]+evisc)*dTdx*dvdx - xconv[0]*0.5*T*T*dvdx - source*v;
          local_resid(resindex) += (viscosity[0]+evisc)*dTdy*dvdy - yconv[0]*0.5*T*T*dvdy;
          local_resid(resindex) += (viscosity[0]+evisc)*dTdz*dvdz - zconv[0]*0.5*T*T*dvdz;
          
          if(useSUPG) {
            tau = this->computeTau(xconv[0], yconv[0], zconv[0], h);
            sres = tau*(T_dot + xconv[0]*T*dTdx + yconv[0]*T*dTdy + zconv[0]*T*dTdz - source);
            local_resid(resindex) += sres*T*xconv[0]*dvdx;
            local_resid(resindex) += sres*T*yconv[0]*dvdy;
            local_resid(resindex) += sres*T*zconv[0]*dvdz;
          }
        }
      }
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual(const double & h, const double & current_time, const FCAD & local_soln, 
                        const FCAD & local_solngrad, const FCAD & local_soln_dot, 
                        const FCAD & local_param, const FCAD & local_param_grad, 
                        const FCAD & local_aux_grad, const FCAD & local_aux_grad, 
                        const FC & ip, const FC & normals, const vector<FC> & basis, 
                        const vector<FC> & basis_grad, 
                        const vector<int> & usebasis, const vector<vector<int> > & offsets,
                        const int & sidetype, const string & side_name, FCAD & local_resid) { 
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int T_basis = usebasis[T_num];
    int numBasis = basis[T_basis].dimension(1);
    
    int numSideCubPoints = ip.dimension(1);
    
    // Set the parameters
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    double v, source, robin_alpha;
    AD T, dTdx, dTdy, dTdz, tau, T_dot;
    double dvdx, dvdy, dvdz;
    
    int resindex;
    
    for( int nPt=0; nPt<numSideCubPoints; nPt++ ) {
      x = ip(0,nPt,0);
      T = local_soln(T_num,nPt,0);
      T_dot = local_soln_dot(T_num,nPt,0);
      dTdx = local_solngrad(T_num,nPt,0);
      if (spaceDim > 1) {
        y = ip(0,nPt,1);
        dTdy = local_solngrad(T_num,nPt,1);
      }
      if (spaceDim > 2) {
        z = ip(0,nPt,2);
        dTdz = local_solngrad(T_num,nPt,2);
      }
      
      for( int i=0; i<numBasis; i++ ) {
        v = basis[T_basis](0,i,nPt);
        dvdx = basis_grad[T_basis](0,i,nPt,0);
        
        if (spaceDim > 1) {
          dvdy = basis_grad[T_basis](0,i,nPt,1);
        }
        if (spaceDim > 2) {
          dvdz = basis_grad[T_basis](0,i,nPt,2);
        }
        source = 0.0;
        robin_alpha = 0.0;
        resindex = offsets[T_num][i];
        
        local_resid(resindex) += -source*v;
        local_resid(resindex) += 0.5*T*T*xconv[0]*normals(0,nPt,0)*v;
        if (spaceDim>1)
          local_resid(resindex) += 0.5*T*T*yconv[0]*normals(0,nPt,1)*v;
        if (spaceDim>2)
          local_resid(resindex) += 0.5*T*T*zconv[0]*normals(0,nPt,2)*v;
        
        
        if(useSUPG) {
          tau = this->computeTau(xconv[0], yconv[0], zconv[0], h);
          local_resid(resindex) += -tau*(T_dot + xconv[0]*T*dTdx + yconv[0]*T*dTdy + 
                                      zconv[0]*T*dTdz - source)*xconv[0]*T*v*normals(0,nPt,0);
          if (spaceDim > 1) {
            local_resid(resindex) += -tau*(T_dot + xconv[0]*T*dTdx + yconv[0]*T*dTdy + 
                                        zconv[0]*T*dTdz - source)*yconv[0]*T*v*normals(0,nPt,1);
          }
          if (spaceDim > 2) {
            local_resid(resindex) += -tau*(T_dot + xconv[0]*T*dTdx + yconv[0]*T*dTdy + 
                                        zconv[0]*T*dTdz - source)*zconv[0]*T*v*normals(0,nPt,2);
          }
        }
      }
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void edgeResidual(const double & h, const double & current_time, const FCAD & local_soln, 
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
  
  void computeFlux(const FC & ip, const double & h, const double & current_time, 
                   const FCAD & local_soln, const FCAD & local_solngrad, 
                   const FCAD & local_soln_dot, const FCAD local_param, const FCAD local_aux, 
                   const FC & normals, FCAD & flux) {

  }

  // ========================================================================================
  // ========================================================================================
  
  AD getDirichletValue(const string & var, const double & x, const double & y, const double & z, 
                       const double & t, const string & gside, const bool & useadjoint) const {
    
    AD val = 0.0;
    
    if (!useadjoint)
      val = trueSolution(var,x,y,z,t);
    
    
    //if (gside =="left")
    //  val = 1.0;
    //else
    //  val = 0.0;
    
    //if (useadjoint)
    val = 0.0;
    
    return val;
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  double getInitialValue(const string & var, const double & x, const double & y, const double & z, 
                         const bool & useadjoint) const {
    double val = 0.0;
    if (x<0.5 && y>=0.5)
      val = -0.2;
    else if (x>=0.5 && y>=0.5)
      val = -1;
    else if (x<0.5 && y<0.5)
      val = 0.5;
    else if (x>=0.5 && y<0.5)
      val = 0.8;
    
    //val = 8.0*x*(1.0-x)*y*(1.0-y);
    
    if (x<0.5 && y<0.5 && x>0.25 && y>0.25)
      val = 1.0;
    else 
      val = 0.0;
    
    //if (x<0.5)
    //  val = 1.0;
    //else 
    //  val = 0.0;
    //val = exp(-100.0*(x-.25)*(x-.25) - 100*(y-.25)*(y-0.25));
    
    if (useadjoint)
      val = 0.0;
    
    return val;
  }
  
  // ========================================================================================
  // error calculation
  // ========================================================================================
  
  double trueSolution(const string & var, const double & x, const double & y, const double & z, 
                      const double & time) const {
    double T = 0.0;
    if (x<(1.0/2.0-3.0*time/5.0)) {
      if (y>=(1.0/2.0+3.0*time/20.0)) 
        T = -0.2;
      else
        T = 0.5;
    }
    else if (x>=(1.0/2.0-3.0*time/5.0) && x<(1.0/2.0-time/4.0)) {
      if (y>=(-8.0*x/7.0+15.0/14.0-15.0*time/28.0))
        T = -1.0;
      else
        T = 0.5;
    }
    else if (x>=(1.0/2.0-time/4.0) && x<(1.0/2.0+time/2.0)) {
      if (y>=(x/6.0+5.0/12.0-5.0*time/24.0))
        T = -1.0;
      else
        T = 0.5;
    }
    else if (x>=(1.0/2.0+time/2.0) && x<(1.0/2.0+4.0*time/5.0)) {
      if (y>=(x - 5.0/(18.0*time)*(x+time-1.0/2.0)*(x+time-1.0/2.0)))
        T = -1.0;
      else
        T = (2.0*x-1.0)/(2.0*time);
    }
    else if (x>=(1.0/2.0+4.0*time/5.0) ) {
      if (y>=(1.0/2.0-time/10.0))
        T = -1.0;
      else
        T = 0.8;
    }
    
    return T;
  }
  
  // ========================================================================================
  // response calculation
  // ========================================================================================
  
  FCAD response(const FCAD & local_soln, const FCAD & local_soln_grad,
                              const DRV & ip, const double & time) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    FCAD resp(numCC,1,numip);
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        double x = ip(i,j,0);
        double y = ip(i,j,1);
        //if (abs(time - 0.5) < 1.0e-10) {
        //  if (x>=0.25 && x<=0.5 && y>=0.25 && y<=0.5)
        resp(i,0,j) = local_soln(i,T_num,j);
        //}
      }
    }
    
    return resp;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD target(const FC & ip, const double & time) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    FCAD targ(numCC,1,numip);
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        targ(i,0,j) = 0.0;
      }
    }
    
    return targ;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD auxiliaryVars(const FCAD soln) const {
    int numCC = soln.dimension(0);
    int numbasis = soln.dimension(2);
    FCAD auxvars(numCC,1,numbasis);
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numbasis; j++) {
        auxvars(i,0,j) = 1.0/3.0*soln(i,T_num,j)*soln(i,T_num,j)*soln(i,T_num,j);
      }
    }
    
    return auxvars;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "T")
        T_num = i;
    }
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setUserDefined(Teuchos::RCP<userDefined> & udfunc_) {
    udfunc = udfunc_;
  }
  
  // ========================================================================================
  // return the value of the stabilization parameter 
  // ========================================================================================
  
  template<class T>  
  T computeTau(const T & xvl, const T & yvl, const T & zvl, const double & h) const {
    
    //double C = 2.0;
    //double C1 = 1.0;
    //double C2 = 2.0;
    
    T nvel, tau;
    if (spaceDim == 1)
      nvel = xvl*xvl;
    else if (spaceDim == 2)
      nvel = xvl*xvl + yvl*yvl;
    else if (spaceDim == 3)
      nvel = xvl*xvl + yvl*yvl + zvl*zvl;
    
    if (nvel > 1E-12)
      nvel = sqrt(nvel);
    
    if (dt>1.0E-12)   
      tau = tau_C[0]/(tau_C1[0]/(dt) + tau_C2[0]*(nvel)/h);
    else
      tau = tau_C[0]/(tau_C2[0]*(nvel)/h);
    
    return tau;
    
  }
  
  // ========================================================================================
  // return the value of the stabilization parameter 
  // ========================================================================================
  
  template<class T>  
  T entropyViscosity(const T & sol, const T & solx, const T & soly, const T & solz, const T & solt, 
                     const double & h) const {
    
    double maxevisc = 0.1;
    
    T entres;
    if (spaceDim == 1)
      entres = sol*(solt+xconv[0]*sol*solx);
    else if (spaceDim == 2)
      entres = sol*(solt + sol*solx + sol*soly);
    else if (spaceDim == 3)
      entres = sol*(solt + xconv[0]*sol*solx + yconv[0]*sol*soly + zconv[0]*sol*solz);
    
    T evisc = C1[0]*h*h*abs(1.0e-12 + entres)/C2[0];
    
    if (evisc.val() > maxevisc) {
      //cout << "evisc exceeds max value: " << evisc.val() << endl;
      //evisc = maxevisc;
    }
    
    return evisc;
    
  }
  
  // ========================================================================================
  // return the source term  
  // ========================================================================================
  
  template<class T>
  T sourceTerm(const double & x, const double & y, const double & z, const double & t, 
               const std::vector<T> & smag, const std::vector<T> & xloc, const std::vector<T> & yloc, 
               const std::vector<T> & zloc) const {
    T val = 0.0;
    return val;
  }
  
  // ========================================================================================
  // return the diffusion at a point 
  // ========================================================================================
  
  double getDiff(const double & x, const double & y, const double & z) const {
    return diff;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  double constant_diff(const double & x, const double & y, const double & z) {
    return 1.0;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  double analytical_diff(const double & x, const double & y, const double & z) {
    return 1.0;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  double constant_concentration_source(const double & x, const double & y, const double & z) {
    return 1.0;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  double analytical_concentration_source(const double & x, const double & y, const double & z) {
    return 1.0;
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
    ef.push_back("entropy_viscosity");
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  std::vector<FC > extraFields() const {
    std::vector<FC > ef;
    ef.push_back(entropy_viscosity);
    return ef;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setExtraFields(const size_t & numElem_) {
    numElem = numElem_;
    entropy_viscosity = FC(numElem, numSteps+1);
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void updateParameters(const std::vector<std::vector<AD> > & params, 
                        const std::vector<string> & paramnames) {
    
    for (size_t p=0; p<paramnames.size(); p++) {
      if (paramnames[p] == "xconv") 
        xconv = params[p];
      else if (paramnames[p] == "yconv") 
        yconv = params[p];
      else if (paramnames[p] == "zconv") 
        zconv = params[p];
      else if (paramnames[p] == "viscosity") 
        viscosity = params[p];
      else if (paramnames[p] == "C1") 
        C1 = params[p];
      else if (paramnames[p] == "C2") 
        C2 = params[p];
      else if (paramnames[p] == "tau_C") 
        tau_C = params[p];
      else if (paramnames[p] == "tau_C1") 
        tau_C1 = params[p];
      else if (paramnames[p] == "tau_C2") 
        tau_C2 = params[p];
    }
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD scalarRespFunc(const FCAD & integralResponses, 
                                    const bool & justDeriv) const {return integralResponses;}
  bool useScalarRespFunc() const {return false;}
  
private:
  
  Teuchos::RCP<userDefined> udfunc;
  
  int spaceDim, numElem, numParams, numResponses, numSteps;
  
  bool isTD, useSUPG, T_vals_initialized;
  
  string convsource;
  
  int perm_type;
  data permdata;
  
  vector<string> varlist;
  int T_num;
  
  int test;
  double diff, smag, dt;
  
  vector<string> sideSets;
  FC entropy_viscosity;
  std::vector<AD> xconv, yconv, zconv, viscosity, C1, C2, tau_C, tau_C1, tau_C2;
  
};

#endif
