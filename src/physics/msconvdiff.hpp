/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MSCONVDIFF_H
#define MSCONVDIFF_H

#include "physics_base.hpp"

class msconvdiff : public physicsbase {
public:
  
  msconvdiff() {} ;
  
  ~msconvdiff() {};
  
  // ========================================================================================
  // ========================================================================================
  
  msconvdiff(Teuchos::RCP<Teuchos::ParameterList> & settings) {
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    numCells = 1;
    numSpecies = 2; //currently this deals with exactly two species; future plans to make this arbitrary number
    
    verbosity = settings->sublist("Physics").get<int>("Verbosity",0);
    
    useSUPG = settings->sublist("Physics").get<bool>("useSUPG",false);
    
    if (settings->sublist("Solver").get<int>("solver",0) == 1) 
      isTD = true;
    else
      isTD = false;
    
    PI = 3.141592653589793238463;
    
    velFromNS = settings->sublist("Physics").get<bool>("Get velocity from navierstokes",false); 
    
    //whether non-Dirichlet boundary defaults to homogeneous Neumann or no-flux boundary condition
    noFlux = settings->sublist("Physics").get<bool>("msconvdiff no-flux default",false); 
    
    analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
    numResponses = settings->sublist("Physics").get<int>("numResp_msconvdiff",numSpecies); 
    useScalarRespFx = settings->sublist("Physics").get<bool>("use scalar response function (msconvdiff)",false);
    
    test = settings->sublist("Physics").get<int>("test",0);
    //test == 30: global control (threshold) of c2 and c1, velocities from navier-stokes double-lid
    
    //specific to particular test case(s)
    regParam = settings->sublist("Analysis").sublist("ROL").get<double>("regularization parameter",1.e-6);
    finTime = settings->sublist("Solver").get<double>("finaltime",1.0);
    
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
    
    AD c1, dc1dx, dc1dy, dc1dz, c1_dot;
    AD c2, dc2dx, dc2dy, dc2dz, c2_dot;
    AD tau, c1_source, c2_source, r1, r2;
    AD xconv, yconv, zconv, diff;
    
    double v1, dv1dx, dv1dy, dv1dz, v2, dv2dx, dv2dy, dv2dz, cprev;
    
    int resindex;
    int c1_basis = usebasis[c1num];
    int c2_basis = usebasis[c2num];
    
    for( int nPt=0; nPt<numCubPoints; nPt++ ) {
      x = ip(0,nPt,0);
      c1 = local_soln(c1num,nPt);
      c1_dot = local_soln_dot(c1num,nPt);
      dc1dx = local_solngrad(c1num,nPt,0);
      c2 = local_soln(c2num,nPt);
      c2_dot = local_soln_dot(c2num,nPt);
      dc2dx = local_solngrad(c2num,nPt,0);
      if (spaceDim > 1) {
        y = ip(0,nPt,1);
        dc1dy = local_solngrad(c1num,nPt,1);
        dc2dy = local_solngrad(c2num,nPt,1);
      }
      if (spaceDim > 2) {
        z = ip(0,nPt,2);
        dc1dz = local_solngrad(c1num,nPt,2);
        dc2dz = local_solngrad(c2num,nPt,2);
      }
      
      diff = getDiff(x,y,z);
      
      if(velFromNS){
        xconv = local_soln(ux_num,nPt);
        if (spaceDim > 1)
          yconv = local_soln(uy_num,nPt);
        if (spaceDim > 2)
          zconv = local_soln(uz_num,nPt);
      }else{
        xconv = this->convectionTerm(x, y, z, current_time, 1);
        yconv = this->convectionTerm(x, y, z, current_time, 2);
        zconv = this->convectionTerm(x, y, z, current_time, 3);
      }
      
      for( int i=0; i<basis[c1_basis].dimension(1); i++ ) {
        // collect the information at the integration point
        v1 = basis[c1_basis](0,i,nPt);
        dv1dx = basis_grad[c1_basis](0,i,nPt,0);
        if (spaceDim > 1) {
          dv1dy = basis_grad[c1_basis](0,i,nPt,1);
        }
        if (spaceDim > 2) {
          dv1dz = basis_grad[c1_basis](0,i,nPt,2);
        }
        
        v2 = basis(0,1,i,nPt);
        dv2dx = basis_grad(0,1,i,nPt,0);
        
        if (spaceDim > 1) {
          y = ip(0,nPt,1);
          dc1dy = local_solngrad(c1num,nPt,1);
          dc2dy = local_solngrad(c2num,nPt,1);
          dv1dy = basis_grad(0,0,i,nPt,1);
          dv2dy = basis_grad(0,1,i,nPt,1);
        }
        if (spaceDim > 2) {
          z = ip(0,nPt,2);
          dc1dz = local_solngrad(c1num,nPt,2);
          dc2dz = local_solngrad(c2num,nPt,2);
          dv1dz = basis_grad(0,0,i,nPt,2);
          dv2dz = basis_grad(0,1,i,nPt,2);
        }
        
        
        
        if(onlyTransient){
          local_resid(0,i) += c1_dot*v1;
          local_resid(1,i) += c2_dot*v2;
          if(useSUPG) {
            tau = this->computeTau(diff, xconv, yconv, zconv, h);
            local_resid(0,i) += tau*c1_dot*(xconv*dv1dx);
            local_resid(1,i) += tau*c2_dot*(xconv*dv2dx);
            if(spaceDim > 1){
              local_resid(0,i) += tau*c1_dot*(yconv*dv1dy);
              local_resid(1,i) += tau*c2_dot*(yconv*dv2dy);
            }
            if(spaceDim > 2){
              local_resid(0,i) += tau*c1_dot*(zconv*dv1dz);
              local_resid(1,i) += tau*c2_dot*(zconv*dv2dz);
            }
          }
        }
        else{
          // sum into the local residual
          
          // c1 equation 
          int term = 1;
          
          c1_source = this->sourceTerm(x, y, z, current_time, term); 
          r1 = this->reactionTerm(x, y, z, current_time, term, c1, c2);
          
          local_resid(0,i) += diff*(dc1dx*dv1dx + dc1dy*dv1dy + dc1dz*dv1dz) + (xconv*dc1dx + yconv*dc1dy + zconv*dc1dz)*v1 - c1_source*v1 + r1*v1;
          
          if (isTD)
            local_resid(0,i) += c1_dot*v1;
          
          if(useSUPG) {
            tau = this->computeTau(diff, xconv, yconv, zconv, h);
            local_resid(0,i) += tau*(xconv*dc1dx + yconv*dc1dy + zconv*dc1dz - c1_source + r1)*xconv*dv1dx;
            if(isTD)
              local_resid(0,i) += tau*(c1_dot)*xconv*dv1dx;
            if (spaceDim > 1){
              local_resid(0,i) += tau*(xconv*dc1dx + yconv*dc1dy + zconv*dc1dz - c1_source + r1)*yconv*dv1dy;
              if(isTD)
                local_resid(0,i) += tau*(c1_dot)*yconv*dv1dy;
            }
            if (spaceDim > 2){
              local_resid(0,i) += tau*(xconv*dc1dx + yconv*dc1dy + zconv*dc1dz - c1_source + r1)*zconv*dv1dz;
              if(isTD)
                local_resid(0,i) += tau*(c1_dot)*zconv*dv1dz;
            }
          }
          
          // c2 equation
          term = 2;
          
          c2_source = this->sourceTerm(x, y, z, current_time, term); 
          r2 = this->reactionTerm(x, y, z, current_time, term, c1, c2);
          
          local_resid(1,i) += diff*(dc2dx*dv2dx + dc2dy*dv2dy + dc2dz*dv2dz) + (xconv*dc2dx + yconv*dc2dy + zconv*dc2dz)*v2 - c2_source*v2 + r2*v2;
          
          if (isTD)
            local_resid(1,i) += c2_dot*v2;
          
          if(useSUPG) {
            tau = this->computeTau(diff, xconv, yconv, zconv, h);
            local_resid(1,i) += tau*(xconv*dc2dx + yconv*dc2dy + zconv*dc2dz - c2_source + r2)*xconv*dv2dx;
            if(isTD)
              local_resid(1,i) += tau*(c2_dot)*xconv*dv2dx;
            if (spaceDim > 1){
              local_resid(1,i) += tau*(xconv*dc2dx + yconv*dc2dy + zconv*dc2dz - c2_source + r2)*yconv*dv2dy;
              if(isTD)
                local_resid(1,i) += tau*(c2_dot)*yconv*dv2dy;
            }
            if (spaceDim > 2){
              local_resid(1,i) += tau*(xconv*dc2dx + yconv*dc2dy + zconv*dc2dz - c2_source + r2)*zconv*dv2dz;
              if(isTD)
                local_resid(1,i) += tau*(c2_dot)*zconv*dv2dz;
            }
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
                        const FCAD & local_aux, const FCAD & local_aux_grad, 
                        const FC & ip, const FC & normals, const vector<FC> & basis, 
                        const vector<FC> & basis_grad, 
                        const vector<int> & usebasis, const vector<vector<int> > & offsets,
                        const int & sidetype, const string & side_name, FCAD & local_resid) { 
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int numBasis = basis.dimension(2);
    int numSideCubPoints = ip.dimension(1);
    
    FCAD local_resid(numSpecies, numBasis);
    
    // Set the parameters
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    double v1, v2;
    AD source1, source2, robin_alpha1, robin_alpha2;
    AD c1, dc1dx, dc1dy, dc1dz;
    AD c2, dc2dx, dc2dy, dc2dz;
    AD dc1dn, dc2dn; //normal derivatives
    AD diff;
    AD xconv, yconv, zconv; //for no-flux boundary conditions
    
    for( int i=0; i<numBasis; i++ ) {
      for( int nPt=0; nPt<numSideCubPoints; nPt++ ) {
        v1 = basis(0,0,i,nPt);
        v2 = basis(0,1,i,nPt);
        x = ip(0,nPt,0);
        c1 = local_soln(c1num,nPt);
        c2 = local_soln(c2num,nPt);
        dc1dx = local_solngrad(c1num,nPt,0);
        dc2dx = local_solngrad(c2num,nPt,0);                
        dc1dn = dc1dx*normals(0,nPt,0);
        dc2dn = dc2dx*normals(0,nPt,0);
        
        if (spaceDim > 1) {
          y = ip(0,nPt,1);
          dc1dy = local_solngrad(c1num,nPt,1);
          dc2dy = local_solngrad(c2num,nPt,1);
          dc1dn += dc1dy*normals(0,nPt,1);
          dc2dn += dc2dy*normals(0,nPt,1);
        }
        if (spaceDim > 2) {
          z = ip(0,nPt,2);
          dc1dz = local_solngrad(c1num,nPt,2);
          dc2dz = local_solngrad(c2num,nPt,2);
          dc1dn += dc1dz*normals(0,nPt,2);
          dc2dn += dc2dz*normals(0,nPt,2);
        }
        
        //Robin boundary condition of form alpha*u + dudn - source = 0, where u is the state and dudn is its normal derivative
        
        source1 = this->boundarySource(x, y, z, current_time, side_name, 1);
        source2 = this->boundarySource(x, y, z, current_time, side_name, 2);
        
        diff = getDiff(x,y,z);
        
        if(noFlux){
          if(velFromNS){
            xconv = local_soln(ux_num,nPt);
            if (spaceDim > 1)
              yconv = local_soln(uy_num,nPt);
            if (spaceDim > 2)
              zconv = local_soln(uz_num,nPt);
          }else{
            xconv = this->convectionTerm(x, y, z, current_time, 1);
            yconv = this->convectionTerm(x, y, z, current_time, 2);
            zconv = this->convectionTerm(x, y, z, current_time, 3);
          }
        }
        
        robin_alpha1 = this->robinAlpha(x, y, z, current_time, side_name, 1, xconv, yconv, zconv, diff);
        robin_alpha2 = this->robinAlpha(x, y, z, current_time, side_name, 2, xconv, yconv, zconv, diff);
        
        local_resid(0,i) +=  (robin_alpha1*(c1*v1) + dc1dn*v1 - source1*v1) - diff*dc1dn*v1;
        local_resid(1,i) +=  (robin_alpha2*(c2*v2) + dc2dn*v2 - source2*v2) - diff*dc2dn*v2;
        
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
  
  AD getDirichletValue(const string & var, const double & x, const double & y, const double & z, const double & t, 
                       const string & gside, const bool & useadjoint) const {
    
    AD val = 0.0;
    //"default" behavior
    if(!useadjoint){
      if(bc_params.size() > 0){
        if(var == "c1")
          val = bc_params[0];
        else if(var == "c2")
          val = bc_params[1];
      }else
        val = 0.0;
    }else{
      val = 0.0;
    }
    
    return val;
    
  }
  
  // ========================================================================================
  /* return alpha for Robin boundary condition */
  // ========================================================================================
  
  AD robinAlpha(const double & x, const double & y, const double & z, const double & t, const string & side, const int & var,
                const AD & xconv, const AD & yconv, const AD & zconv, const AD & diff) const {
    
    AD val = 0.0;
    
    if(noFlux){
      if (side == "left")
        val = xconv/diff;
      else if (side == "right")
        val = -xconv/diff;
      else if (side == "top")
        val = -yconv/diff;
      else if (side == "bottom")
        val = yconv/diff;
      else if (side == "front")
        val = -zconv/diff;
      else if (side == "back")
        val = zconv/diff;
    }else{
      val = 0.0;
    }
    return val;
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  double getInitialValue(const string & var, const double & x, const double & y, const double & z, const bool & useadjoint) const {
    double val = 0.0;
    
    if(init_params.size() > 0 && !useadjoint){
      if(var == "c1")
        val = init_params[0].val();
      if(var == "c2")
        val = init_params[1].val();
    }else
      val = 0.0;
    
    return val;
  }
  
  // ========================================================================================
  // error calculation
  // ========================================================================================
  
  double trueSolution(const string & var, const double & x, const double & y, const double & z, const double & time) const {
    double c = 0.0;
    
    if(verbosity > 0)
      cout << "No known true solution...'truth' defaults to zero..." << endl;
    
    return c;
  }
  
  // ========================================================================================
  // response calculation
  // ========================================================================================
  
  FCAD response(const FCAD & local_soln, 
                               const FCAD & local_soln_grad,
                               const DRV & ip, const double & time) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    
    FCAD resp(numCC,numResponses,numip);
    
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        if (test == 30){
          double x = ip(i,j,0) - 0.5;
          double y = ip(i,j,1) - 0.5;
          AD thresh1_loc = c1_threshold_params[0]*(1.0+sqrt(x*x+y*y)); 
          AD thresh2_loc = c2_threshold_params[0]*(1.0+sqrt(x*x+y*y));
          resp(i,0,j) = 0.5*max(local_soln(i,c1num,j)-thresh1_loc, 0.0)*max(local_soln(i,c1num,j)-thresh1_loc, 0.0);
          resp(i,1,j) = 0.5*max(local_soln(i,c2num,j)-thresh2_loc, 0.0)*max(local_soln(i,c2num,j)-thresh2_loc, 0.0);
          if(abs(time-finTime)<1.e-8){
            for (int k=0; k<c1_source_mag_params.size(); k++)
              resp(i,0,j) += regParam*c1_source_decay_params[k]*c1_source_decay_params[k];
            for (int k=0; k<c1_source_mag_extra.size(); k++)
              resp(i,0,j) += regParam*c1_source_mag_extra[k]*c1_source_mag_extra[k];
            for (int k=0; k<c2_source_mag_params.size(); k++)
              resp(i,1,j) += regParam*c2_source_decay_params[k]*c2_source_decay_params[k];
            for (int k=0; k<c2_source_mag_extra.size(); k++)
              resp(i,1,j) += regParam*c2_source_mag_extra[k]*c2_source_mag_extra[k];
          }
        }else{ //"default"
          resp(i,0,j) = local_soln(i,c1num,j);
          resp(i,1,j) = local_soln(i,c2num,j);
        }
      }
    }
    
    return resp;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD scalarRespFunc(const FCAD & integralResponses, const bool & justDeriv) const {
    return integralResponses;
  }
  
  bool useScalarRespFunc() const {
    return useScalarRespFx;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD target(const FC & ip, const double & time) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    FCAD targ(numCC,numResponses,numip);
    
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        for(int ii=0; ii<numResponses; ii++)
          targ(i,ii,j) = 1.0;
      }
    }
    
    return targ;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "c1")
        c1num = i;
      if (varlist[i] == "c2")
        c2num = i;
      if (varlist[i] == "ux")
        ux_num = i;
      if (varlist[i] == "uy")
        uy_num = i;
      if (varlist[i] == "uz")
        uz_num = i;
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
  T computeTau(const T & localdiff, const T & xvl, const T & yvl, const T & zvl, const double & h) const {
    
    double C1 = 4.0;
    double C2 = 2.0;
    
    T nvel;
    if (spaceDim == 1)
      nvel = xvl*xvl;
    else if (spaceDim == 2)
      nvel = xvl*xvl + yvl*yvl;
    else if (spaceDim == 3)
      nvel = xvl*xvl + yvl*yvl + zvl*zvl;
    
    if (nvel > 1E-12)
      nvel = sqrt(nvel);
    
    return 1.0/(C1*localdiff/h/h + C2*(nvel)/h); //convdiff has a 4.0 instead of a 1.0 in the numerator
    
  }
  
  // ========================================================================================
  // return the interior source term  
  // ========================================================================================
  
  AD sourceTerm(const double & x, const double & y, const double & z, const double & t, const int & comp) const{
    
    AD val = 0.0;
    
    if (test == 30){
      if(comp == 1){
        int nparams = c1_source_mag_params.size();
        for (int i=0; i<nparams; i++){
          val += c1_source_mag_params[i]*
          exp(-100.0*(x-c1_source_xloc_params[i])*(x-c1_source_xloc_params[i]) - 100.0*(y-c1_source_yloc_params[i])*(y-c1_source_yloc_params[i]))
          *exp(-c1_source_decay_params[i]*t);
        }
        for (int i=0; i<c1_source_mag_extra.size(); i++){ //man-made sinks
          val += c1_source_mag_extra[i]*
          (exp((-1.0*c1_source_xx_cov_xtra[i])*(x-c1_source_xloc_xtra[i])*(x-c1_source_xloc_xtra[i])
               + (-1.0*c1_source_yy_cov_xtra[i])*(y-c1_source_yloc_xtra[i])*(y-c1_source_yloc_xtra[i])
               + (-2.0*c1_source_xy_cov_xtra[i])*(x-c1_source_xloc_xtra[i])*(y-c1_source_yloc_xtra[i]) ));
        }
      }else if(comp == 2){
        int nparams = c2_source_mag_params.size();
        for (int i=0; i<nparams; i++){
          val += c2_source_mag_params[i]*
          exp(-100.0*(x-c2_source_xloc_params[i])*(x-c2_source_xloc_params[i]) - 100.0*(y-c2_source_yloc_params[i])*(y-c2_source_yloc_params[i]))
          *exp(-c2_source_decay_params[i]*t);
        }
        for (int i=0; i<c2_source_mag_extra.size(); i++){ //man-made sinks
          val += c2_source_mag_extra[i]*
          (exp((-1.0*c2_source_xx_cov_xtra[i])*(x-c2_source_xloc_xtra[i])*(x-c2_source_xloc_xtra[i])
               + (-1.0*c2_source_yy_cov_xtra[i])*(y-c2_source_yloc_xtra[i])*(y-c2_source_yloc_xtra[i])
               + (-2.0*c2_source_xy_cov_xtra[i])*(x-c2_source_xloc_xtra[i])*(y-c2_source_yloc_xtra[i]) ));
        }
      }
      //"natural sinks"
      val += -0.05*exp(-10.0*(x-0.5)*(x-0.5) - 10.0*(y-0.7)*(y-0.7)); //"forest"
      if (x < 0.3 && y < 0.3) //"ocean"
        val += -0.05;
    }else{ //"default"
      if(comp == 1){
        for(int i=0; i<c1_source_mag_params.size(); i++){
          val += c1_source_mag_params[i]*
          (exp((-1.0*c1_source_xx_cov_params[i])*(x-c1_source_xloc_params[i])*(x-c1_source_xloc_params[i])
               + (-1.0*c1_source_yy_cov_params[i])*(y-c1_source_yloc_params[i])*(y-c1_source_yloc_params[i])
               + (-2.0*c1_source_xy_cov_params[i])*(x-c1_source_xloc_params[i])*(y-c1_source_yloc_params[i]) ));
        }
        for (int i=0; i<c1_source_mag_extra.size(); i++){
          val += c1_source_mag_extra[i]*
          (exp((-1.0*c1_source_xx_cov_xtra[i])*(x-c1_source_xloc_xtra[i])*(x-c1_source_xloc_xtra[i])
               + (-1.0*c1_source_yy_cov_xtra[i])*(y-c1_source_yloc_xtra[i])*(y-c1_source_yloc_xtra[i])
               + (-2.0*c1_source_xy_cov_xtra[i])*(x-c1_source_xloc_xtra[i])*(y-c1_source_yloc_xtra[i]) ));
        }
        for (int i=0; i<c1_source_mag_xtra_stoch.size(); i++){
          val += c1_source_mag_xtra_stoch[i]*
          (exp((-1.0*c1_source_xxcov_xtra_stoch[i])*(x-c1_xlocxtra_stoch[i])*(x-c1_xlocxtra_stoch[i])
               + (-1.0*c1_source_yycov_xtra_stoch[i])*(y-c1_ylocxtra_stoch[i])*(y-c1_ylocxtra_stoch[i])
               + (-2.0*c1_source_xycov_xtra_stoch[i])*(x-c1_xlocxtra_stoch[i])*(y-c1_ylocxtra_stoch[i]) ));
        }
        if(source_const.size() > 0)
          val += source_const[0];
      }else if(comp == 2){
        for(int i=0; i<c2_source_mag_params.size(); i++){
          val += c2_source_mag_params[i]*
          (exp((-1.0*c2_source_xx_cov_params[i])*(x-c2_source_xloc_params[i])*(x-c2_source_xloc_params[i])
               + (-1.0*c2_source_yy_cov_params[i])*(y-c2_source_yloc_params[i])*(y-c2_source_yloc_params[i])
               + (-2.0*c2_source_xy_cov_params[i])*(x-c2_source_xloc_params[i])*(y-c2_source_yloc_params[i]) ));
        }
        for (int i=0; i<c2_source_mag_extra.size(); i++){
          val += c2_source_mag_extra[i]*
          (exp((-1.0*c2_source_xx_cov_xtra[i])*(x-c2_source_xloc_xtra[i])*(x-c2_source_xloc_xtra[i])
               + (-1.0*c2_source_yy_cov_xtra[i])*(y-c2_source_yloc_xtra[i])*(y-c2_source_yloc_xtra[i])
               + (-2.0*c2_source_xy_cov_xtra[i])*(x-c2_source_xloc_xtra[i])*(y-c2_source_yloc_xtra[i]) ));
        }
        for (int i=0; i<c2_source_mag_xtra_stoch.size(); i++){
          val += c2_source_mag_xtra_stoch[i]*
          (exp((-1.0*c2_source_xxcov_xtra_stoch[i])*(x-c2_xlocxtra_stoch[i])*(x-c2_xlocxtra_stoch[i])
               + (-1.0*c2_source_yycov_xtra_stoch[i])*(y-c2_ylocxtra_stoch[i])*(y-c2_ylocxtra_stoch[i])
               + (-2.0*c2_source_xycov_xtra_stoch[i])*(x-c2_xlocxtra_stoch[i])*(y-c2_ylocxtra_stoch[i]) ));
        }
        if(source_const.size() > 0)
          val += source_const[1];
      }
    }
    
    return val;
  }
  // ========================================================================================
  // return the boundary source term  
  // ========================================================================================
  
  AD boundarySource(const double & x, const double & y, const double & z, const double & t, const string & side, const int & var) const {
    
    AD val = 0.0;
    
    return val;
  }
  
  // ========================================================================================
  // return the convection term (well, just the velocity...)
  // ========================================================================================
  
  AD convectionTerm(const double & x, const double & y, const double & z, const double & t, const int & dir) const{
    AD vel = 0.0;
    
    //"default"
    if(dir == 1)
      vel = convection_params[0];
    else if(dir == 2)
      vel = convection_params[1];
    else if(dir == 3)
      vel = convection_params[2]; 
    
    return vel;
  }
  
  // ========================================================================================
  // return the reaction term  
  // ========================================================================================
  
  AD reactionTerm(const double & x, const double & y, const double & z, const double & t, const int & comp, const AD & c1, const AD & c2) const {
    AD r = 0.0;
    if(test == 30){
      if (comp == 1)
        r = 0.5*reaction_params[0]*c1*c2*c2; //species lost in reaction
      else if (comp == 2)
        r = 1.0*reaction_params[0]*c1*c2*c2; //species lost in reaction
    }else{ //"default"
      if(comp == 1)
        r = reaction_params[0]*c1;
      else if(comp == 2)
        r = reaction_params[1]*c2;
    }
    return r;
  }
  
  // ========================================================================================
  // return the diffusion at a point 
  // ========================================================================================
  
  AD getDiff(const double & x, const double & y, const double & z) const { 
    return diffusion_params[0];
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
  
  void updateParameters(const std::vector<std::vector<AD> > & params, const std::vector<string> & paramnames) {
    for (size_t p=0; p<paramnames.size(); p++) {
      if (paramnames[p] == "diffusion") 
        diffusion_params = params[p];
      else if (paramnames[p] == "convection")
        convection_params = params[p];
      else if (paramnames[p] == "reaction")
        reaction_params = params[p];
      else if (paramnames[p] == "c1_source_mag")
        c1_source_mag_params = params[p];
      else if (paramnames[p] == "c1_source_xlocation")
        c1_source_xloc_params = params[p];
      else if (paramnames[p] == "c1_source_ylocation")
        c1_source_yloc_params = params[p];
      else if (paramnames[p] == "c1_source_zlocation")
        c1_source_zlocation_params = params[p];
      else if (paramnames[p] == "c2_source_mag")
        c2_source_mag_params = params[p];
      else if (paramnames[p] == "c2_source_xlocation")
        c2_source_xloc_params = params[p];
      else if (paramnames[p] == "c2_source_ylocation")
        c2_source_yloc_params = params[p];
      else if (paramnames[p] == "c2_source_zlocation")
        c2_source_zlocation_params = params[p];
      else if (paramnames[p] == "c1_threshold")
        c1_threshold_params = params[p];
      else if (paramnames[p] == "c2_threshold")
        c2_threshold_params = params[p];
      else if (paramnames[p] == "c1_source_decay")
        c1_source_decay_params = params[p];
      else if (paramnames[p] == "c2_source_decay")
        c2_source_decay_params = params[p];
      else if (paramnames[p] == "c1_source_xxcov")
        c1_source_xx_cov_params = params[p];
      else if (paramnames[p] == "c1_source_xycov")
        c1_source_xy_cov_params = params[p];
      else if (paramnames[p] == "c1_source_yycov")
        c1_source_yy_cov_params = params[p];
      else if (paramnames[p] == "c2_source_xxcov")
        c2_source_xx_cov_params = params[p];
      else if (paramnames[p] == "c2_source_xycov")
        c2_source_xy_cov_params = params[p];
      else if (paramnames[p] == "c2_source_yycov")
        c2_source_yy_cov_params = params[p];
      else if (paramnames[p] == "c1_source_mag_xtra")
        c1_source_mag_extra = params[p];
      else if (paramnames[p] == "c1_source_xloc_xtra")
        c1_source_xloc_xtra = params[p];
      else if (paramnames[p] == "c1_source_yloc_xtra")
        c1_source_yloc_xtra = params[p];
      else if (paramnames[p] == "c1_source_xxcov_xtra")
        c1_source_xx_cov_xtra = params[p];
      else if (paramnames[p] == "c1_source_xycov_xtra")
        c1_source_xy_cov_xtra = params[p];
      else if (paramnames[p] == "c1_source_yycov_xtra")
        c1_source_yy_cov_xtra = params[p];
      else if (paramnames[p] == "c2_source_mag_xtra")
        c2_source_mag_extra = params[p];
      else if (paramnames[p] == "c2_source_xloc_xtra")
        c2_source_xloc_xtra = params[p];
      else if (paramnames[p] == "c2_source_yloc_xtra")
        c2_source_yloc_xtra = params[p];
      else if (paramnames[p] == "c2_source_xxcov_xtra")
        c2_source_xx_cov_xtra = params[p];
      else if (paramnames[p] == "c2_source_xycov_xtra")
        c2_source_xy_cov_xtra = params[p];
      else if (paramnames[p] == "c2_source_yycov_xtra")
        c2_source_yy_cov_xtra = params[p];
      else if (paramnames[p] == "c1_source_mag_xtra_stoch")
        c1_source_mag_xtra_stoch = params[p];
      else if (paramnames[p] == "c1_source_xloc_xtra_stoch")
        c1_xlocxtra_stoch = params[p];
      else if (paramnames[p] == "c1_source_yloc_xtra_stoch")
        c1_ylocxtra_stoch = params[p];
      else if (paramnames[p] == "c1_source_xxcov_xtra_stoch")
        c1_source_xxcov_xtra_stoch = params[p];
      else if (paramnames[p] == "c1_source_xycov_xtra_stoch")
        c1_source_xycov_xtra_stoch = params[p];
      else if (paramnames[p] == "c1_source_yycov_xtra_stoch")
        c1_source_yycov_xtra_stoch = params[p];
      else if (paramnames[p] == "c2_source_mag_xtra_stoch")
        c2_source_mag_xtra_stoch = params[p];
      else if (paramnames[p] == "c2_source_xloc_xtra_stoch")
        c2_xlocxtra_stoch = params[p];
      else if (paramnames[p] == "c2_source_yloc_xtra_stoch")
        c2_ylocxtra_stoch = params[p];
      else if (paramnames[p] == "c2_source_xxcov_xtra_stoch")
        c2_source_xxcov_xtra_stoch = params[p];
      else if (paramnames[p] == "c2_source_xycov_xtra_stoch")
        c2_source_xycov_xtra_stoch = params[p];
      else if (paramnames[p] == "c2_source_yycov_xtra_stoch")
        c2_source_yycov_xtra_stoch = params[p];
      else if (paramnames[p] == "msconvdiff_boundary")
        bc_params = params[p];
      else if (paramnames[p] == "msconvdiff_init")
        init_params = params[p];
      else if (paramnames[p] == "msconvdiff_source_const")
        source_const = params[p];
      else if(verbosity > 0) //false alarms if multiple physics modules used...
        cout << "Parameter not used in msconvdiff: " << paramnames[p] << endl;  
    }
  }
  
  
private:
  
  Teuchos::RCP<userDefined> udfunc;
  
  int spaceDim, numElem, numParams, numResponses, numSpecies;
  
  std::vector<AD> diffusion_params, convection_params, reaction_params,
  source_const,
  c1_threshold_params, c2_threshold_params, 
  c1_source_decay_params, c2_source_decay_params, 
  c1_source_mag_params, c2_source_mag_params,
  c1_source_xloc_params, c1_source_yloc_params, c1_source_zlocation_params,
  c1_source_xx_cov_params, c1_source_xy_cov_params, c1_source_yy_cov_params,
  c2_source_xloc_params, c2_source_yloc_params, c2_source_zlocation_params,
  c2_source_xx_cov_params, c2_source_xy_cov_params, c2_source_yy_cov_params,
  c1_source_mag_extra, c1_source_xloc_xtra, c1_source_yloc_xtra, c1_source_xx_cov_xtra, c1_source_xy_cov_xtra, c1_source_yy_cov_xtra, 
  c2_source_mag_extra, c2_source_xloc_xtra, c2_source_yloc_xtra, c2_source_xx_cov_xtra, c2_source_xy_cov_xtra, c2_source_yy_cov_xtra, 
  c1_source_mag_xtra_stoch, c2_source_mag_xtra_stoch,
  c1_source_xxcov_xtra_stoch, c1_source_xycov_xtra_stoch, c1_source_yycov_xtra_stoch, c1_xlocxtra_stoch, c1_ylocxtra_stoch,
  c2_source_xxcov_xtra_stoch, c2_source_xycov_xtra_stoch, c2_source_yycov_xtra_stoch, c2_xlocxtra_stoch, c2_ylocxtra_stoch,
  bc_params, init_params;
  
  bool isTD, useSUPG;
  
  vector<string> varlist;
  int c1num, c2num;
  
  int test;
  
  std::string analysis_type; //to know when parameter is a sample that needs to be transformed
  
  bool velFromNS;
  int ux_num, uy_num, uz_num; //if getting velocity from ns
  
  double PI;
  bool noFlux;
  int verbosity;
  bool useScalarRespFx;
  
  double finTime; //for response function
  double regParam; //regularization parameter
};

#endif
