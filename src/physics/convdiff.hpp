/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef CONVDIFF_H
#define CONVDIFF_H

#include "physics_base.hpp"

class convdiff : public physicsbase {
public:
  
  convdiff() {} ;
  
  ~convdiff() {};
  
  // ========================================================================================
  // ========================================================================================
  
  convdiff(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_, const size_t & numip_side_) :
  numip(numip_), numip_side(numip_side_) {
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    numCells = 1;
    
    PI = 3.141592653589793238463;
    
    verbosity = settings->sublist("Physics").get<int>("Verbosity",0);
    
    useSUPG = settings->sublist("Physics").get<bool>("useSUPG",false);
    
    if (settings->sublist("Solver").get<int>("solver",0) == 1) 
      isTD = true;
    else
      isTD = false;
    
    analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
    
    numResponses = settings->sublist("Physics").get<int>("numResp_convdiff",1);
    useScalarRespFx = settings->sublist("Physics").get<bool>("use scalar response function (convdiff)",false);
    noFlux = settings->sublist("Physics").get<bool>("no-flux boundaries (convdiff)",false);
    velFromNS = settings->sublist("Physics").get<bool>("Get velocity from navierstokes",false);
    burgersflux = settings->sublist("Physics").get<bool>("Add Burgers",false);
    
    test = settings->sublist("Physics").get<int>("test",0);
    //test == 100: linear-Gaussian inverse + linear control under uncertainty, forward and inverse part
    //test == 102: linear-Gaussian inverse + linear control under uncertainty, control part 
    //              (control starts after inverse timeframe (assuming test 100 ends at finaltime = 2.0))
    
    //specific to particular test cases
    
    //to transform samples from standard normal into samples from normal with mean at initial value and covariance LL'
    if(test == 102 && analysis_type == "SOL"){
      string cholfile = settings->sublist("Physics").get<string>("Cholesky-factored Covariance","postCovChol.dat");
      std::ifstream fin(cholfile.c_str());
      
      for (std::string line; std::getline(fin, line); ){
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream in(line);
        postCovChol.push_back(std::vector<ScalarT>(std::istream_iterator<ScalarT>(in),std::istream_iterator<ScalarT>()));
      }
      
      std::string meanfile = 
      settings->sublist("Parameters").sublist("source_mag_xtra_stoch").get<string>("source","c1_source_mag_xtra_stoch.dat");
      std::ifstream fin2(meanfile.c_str());
      std::istream_iterator<ScalarT> start(fin2), end;
      vector<ScalarT> meep(start, end);
      for(int i=0; i<meep.size(); i++){
        postMeanSource.push_back(meep[i]);
      }
    }
    
    regParam = settings->sublist("Analysis").sublist("ROL").get<ScalarT>("regularization parameter",1.e-6);
    moveVort = settings->sublist("Physics").get<bool>("moving vortices",true);
    finTime = settings->sublist("Solver").get<ScalarT>("finaltime",1.0);
    data_noise_std = settings->sublist("Analysis").get("Additive Normal Noise Standard Dev",0.0);
  }
  
  // ========================================================================================
  // ========================================================================================
 
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int c_basis = wkset->usebasis[cnum];
    int numBasis = wkset->basis[c_basis].dimension(1);
    int numCubPoints = wkset->ip.dimension(1);
    
    ScalarT x = 0.0;
    ScalarT y = 0.0;
    ScalarT z = 0.0;
    ScalarT current_time = wkset->time;
    AD c, dcdx, dcdy, dcdz, c_dot;
    AD tau, source, diff, xconv, yconv, zconv, reaction;
    ScalarT v, dvdx, dvdy, dvdz;
    int resindex;
    AD wx,wy,wz;
    
    for (int nPt=0; nPt<numCubPoints; nPt++ ) {
      x = wkset->ip(0,nPt,0);
      c = wkset->local_soln(cnum,nPt,0);
      c_dot = wkset->local_soln_dot(cnum,nPt,0);
      dcdx = wkset->local_soln_grad(cnum,nPt,0);
      if (spaceDim > 1) {
        y = wkset->ip(0,nPt,1);
        dcdy = wkset->local_soln_grad(cnum,nPt,1);
      }
      if (spaceDim > 2) {
        z = wkset->ip(0,nPt,2);
        dcdz = wkset->local_soln_grad(cnum,nPt,2);
      }
      
      diff = getDiff(x,y,z);
      
      if(velFromNS){
        xconv = wkset->local_soln(ux_num,nPt,0);
        if (spaceDim > 1)
          yconv = wkset->local_soln(uy_num,nPt);
        if (spaceDim > 2)
          zconv = wkset->local_soln(uz_num,nPt);
      }
      else {
        xconv = this->convectionTerm(x, y, z, current_time, 1);
        yconv = this->convectionTerm(x, y, z, current_time, 2);
        zconv = this->convectionTerm(x, y, z, current_time, 3);
      }
      if (burgersflux) {
        wx = 1.0;
        wy = 0.0;
        wz = 0.0;
      }
      
      source = this->sourceTerm(x, y, z, current_time);
      reaction = this->reactionTerm(x, y, z, current_time, c);
      tau = this->computeTau(diff, xconv, yconv, zconv, wkset->h);

      for (int i=0; i<numBasis; i++ ) {
        v = wkset->basis[c_basis](0,i,nPt);
        dvdx = wkset->basis_grad[c_basis](0,i,nPt,0);
        if (spaceDim > 1) {
          dvdy = wkset->basis_grad[c_basis](0,i,nPt,1);
        }
        if (spaceDim > 2) {
          dvdz = wkset->basis_grad[c_basis](0,i,nPt,2);
        }
        
        resindex = wkset->offsets[cnum][i];
          
        
        /*if (wkset->onlyTransient) {
          wkset->res(resindex) += c_dot*v;
          if(useSUPG) {
            int comp = 1;
            wkset->res(resindex) += tau*c_dot*(xconv*dvdx);
            if(spaceDim > 1)
              wkset->res(resindex) += tau*c_dot*(yconv*dvdy);
            if(spaceDim > 2)
              wkset->res(resindex) += tau*c_dot*(zconv*dvdz);
          }
        }
        else {
          */
          // sum into the local residual
        
          wkset->res(resindex) += diff*dcdx*dvdx - xconv*c*dvdx - source*v + reaction*v;
          if (spaceDim > 1)
            wkset->res(resindex) += diff*dcdy*dvdy - yconv*c*dvdy;
          if (spaceDim > 2)
            wkset->res(resindex) += diff*dcdz*dvdz - zconv*c*dvdz;
          
          //if (wkset->isTransient)
            wkset->res(resindex) += c_dot*v;
          
          if (burgersflux) {
            wkset->res(resindex) += -0.5*wx*c*c*dvdx;
            if (spaceDim > 1)
              wkset->res(resindex) += -0.5*wy*c*c*dvdy;
            if (spaceDim > 2)
              wkset->res(resindex) += -0.5*wz*c*c*dvdz;
          }
          
          if(useSUPG) {
            wkset->res(resindex) += tau*(xconv*dcdx + yconv*dcdy + zconv*dcdz - source + reaction)*xconv*dvdx;
            if(isTD)
              wkset->res(resindex) += tau*(c_dot)*xconv*dvdx;
            if (spaceDim > 1) {
              wkset->res(resindex) += tau*(xconv*dcdx + yconv*dcdy + zconv*dcdz - source + reaction)*yconv*dvdy;
              if(isTD)
                wkset->res(resindex) += tau*(c_dot)*yconv*dvdy;
            }
            if (spaceDim > 2) {
              wkset->res(resindex) += tau*(xconv*dcdx + yconv*dcdy + zconv*dcdz - source + reaction)*zconv*dvdz;
              if(isTD)
                wkset->res(resindex) += tau*(c_dot)*zconv*dvdz;
            }
          }
        //}
      }
    }
    
  }
  
  
  // ========================================================================================
  // ========================================================================================
 
  void boundaryResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int c_basis = wkset->usebasis[cnum];
    int numBasis = wkset->basis_side[c_basis].dimension(1);
    int numSideCubPoints = wkset->ip_side.dimension(1);
    
    // Set the parameters
    ScalarT x = 0.0;
    ScalarT y = 0.0;
    ScalarT z = 0.0;
    ScalarT current_time = wkset->time;
    
    ScalarT v, dvdx, dvdy, dvdz;
    AD source, robin_alpha;
    AD c, dcdx, dcdy, dcdz;
    AD diff;
    AD xconv, yconv, zconv; //for no-flux boundary conditions
    AD lambda;
 
    AD wx,wy,wz;
    
    int resindex;
    ScalarT nx,ny,nz;
    
    if (wkset->sidetype > 1) {
    
      for (int nPt=0; nPt<numSideCubPoints; nPt++ ) {
        x = wkset->ip_side(0,nPt,0);
        c = wkset->local_soln_side(cnum,nPt,0);
        dcdx = wkset->local_soln_grad_side(cnum,nPt,0);
        nx = wkset->normals(0,nPt,0);
        if (spaceDim > 1) {
          y = wkset->ip_side(0,nPt,1);
          dcdy = wkset->local_soln_grad_side(cnum,nPt,1);
          ny = wkset->normals(0,nPt,1);
        }
        if (spaceDim > 2) {
          z = wkset->ip_side(0,nPt,2);
          dcdz = wkset->local_soln_grad_side(cnum,nPt,2);
          nz = wkset->normals(0,nPt,2);
        }
          
        if(velFromNS){
          xconv = wkset->local_soln_side(ux_num,nPt,0);
          if (spaceDim > 1)
            yconv = wkset->local_soln_side(uy_num,nPt,0);
          if (spaceDim > 2)
            zconv = wkset->local_soln_side(uz_num,nPt,0);
        }
        else { 
          xconv = this->convectionTerm(x, y, z, current_time, 1);
          yconv = this->convectionTerm(x, y, z, current_time, 2);
          zconv = this->convectionTerm(x, y, z, current_time, 3);
        }
      
        for (int i=0; i<numBasis; i++ ) {
          v = wkset->basis_side[c_basis](0,i,nPt);
        
          source = this->boundarySource(x, y, z, current_time, wkset->sidename);
          source += -(xconv*nx+yconv*ny+zconv*nz)*c;
          resindex = wkset->offsets[cnum][i];
        
          wkset->res(resindex) += -source*v;
        }
      }
    }
    else {
      for (int nPt=0; nPt<numSideCubPoints; nPt++ ) {
        x = wkset->ip_side(0,nPt,0);
        c = wkset->local_soln_side(cnum,nPt,0);
        dcdx = wkset->local_soln_grad_side(cnum,nPt,0);
        nx = wkset->normals(0,nPt,0);
        if (spaceDim > 1) {
          y = wkset->ip_side(0,nPt,1);
          dcdy = wkset->local_soln_grad_side(cnum,nPt,1);
          ny = wkset->normals(0,nPt,1);
        }
        if (spaceDim > 2) {
          z = wkset->ip_side(0,nPt,2);
          dcdz = wkset->local_soln_grad_side(cnum,nPt,2);
          nz = wkset->normals(0,nPt,2);
        }
          
        if(velFromNS){
          xconv = wkset->local_soln_side(ux_num,nPt,0);
          if (spaceDim > 1)
            yconv = wkset->local_soln_side(uy_num,nPt,0);
          if (spaceDim > 2)
            zconv = wkset->local_soln_side(uz_num,nPt,0);
        }
        else { 
          xconv = this->convectionTerm(x, y, z, current_time, 1);
          yconv = this->convectionTerm(x, y, z, current_time, 2);
          zconv = this->convectionTerm(x, y, z, current_time, 3);
        }
        diff = getDiff(x, y, z);

        if (wkset->sidetype == -1)
          lambda = wkset->local_aux_side(cnum,nPt);
        else
          lambda = this->boundarySource(x, y, z, current_time, wkset->sidename);
          
        ScalarT sf = 1.0;
        if (wkset->isAdjoint) {
          sf = 1.0;
        }
        
        AD convscale = 0.0;
        if (spaceDim == 1) {
          convscale = abs(xconv*nx + 1.0e-14);
        }
        else if (spaceDim == 2) {
          convscale = abs(xconv*nx + 1.0e-14) + abs(yconv*ny + 1.0e-14);
        }
        else if (spaceDim == 3) {
          convscale = abs(xconv*nx + 1.0e-14) + abs(yconv*ny + 1.0e-14) + abs(zconv*ny + 1.0e-14);
        }
        AD weakDiriScale = diff*10.0/(wkset->h) + convscale;

        AD weakburgscale = 0.0;
        if (burgersflux) {
          wx = 1.0;
          wy = 0.0;
          wz = 0.0;
          if (spaceDim == 1) {
            weakburgscale = abs(c*wx*nx);
          }
          else if (spaceDim == 2) {
            weakburgscale = abs(c*wx*nx) + abs(c*wy*ny);
          }
          else if (spaceDim == 3) {
            weakburgscale = abs(c*wx*nx) + abs(c*wy*ny) + abs(c*wz*nz);
          }
        }
          
        for (int i=0; i<numBasis; i++ ) {
          resindex = wkset->offsets[cnum][i];
          v = wkset->basis_side[c_basis](0,i,nPt);
          dvdx = wkset->basis_grad_side[c_basis](0,i,nPt,0);
          if (spaceDim > 1)
            dvdy = wkset->basis_grad_side[c_basis](0,i,nPt,1);
          if (spaceDim > 2)
            dvdz = wkset->basis_grad_side[c_basis](0,i,nPt,2);

          wkset->res(resindex) += (-diff*dcdx + xconv*c)*nx*v - sf*(diff*dvdx)*nx*(c-lambda) + weakDiriScale*(c-lambda)*v;//(dedx*normals(0,nPt,0) - source + robin_alpha*e)*v;
          if (spaceDim > 1) {
            wkset->res(resindex) += (-diff*dcdy+yconv*c)*ny*v - sf*(diff*dvdy)*ny*(c-lambda);// + dedy*normals(0,nPt,1)*v;
          }
          if (spaceDim > 2)
            wkset->res(resindex) += (-diff*dcdz+zconv*c)*nz*v - sf*(diff*dvdz)*nz*(c-lambda);
          
          if (burgersflux) {
            wkset->res(resindex) += 0.5*c*c*(wx*nx+wy*ny+wz*nz)*v+10.0*weakburgscale*(c-lambda)*v;
            //wkset->res(resindex) += 1.0*(c-lambda)*(c-lambda)*(c-lambda)*v;
          }
          
        }
      }
    }
    
  }
  
  // ========================================================================================
  // ========================================================================================
 
  void edgeResidual() {}
 
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================

  void computeFlux() {

    ScalarT x = 0.0;
    ScalarT y = 0.0;
    ScalarT z = 0.0;
 
    ScalarT current_time = wkset->time;
    ScalarT sf = 1.0;
    if (wkset->isAdjoint) {
      sf = 1.0;
    }
    
    AD xconv, yconv, zconv;
    AD c, dcdx, dcdy, dcdz, lambda;
    for (size_t i=0; i<wkset->ip_side.dimension(1); i++) {
      x = wkset->ip_side(0,i,0);
      c = wkset->local_soln_side(cnum,i,0);
      dcdx = wkset->local_soln_grad_side(cnum,i,0);
      lambda = wkset->local_aux_side(cnum,i);
      if (spaceDim > 1) {
        y = wkset->ip_side(0,i,1);
        dcdy = wkset->local_soln_grad_side(cnum,i,1);
      }
      if (spaceDim > 2) {
        z = wkset->ip_side(0,i,2);
        dcdz = wkset->local_soln_grad_side(cnum,i,2);
      }
      
      if(velFromNS){
        xconv = wkset->local_soln_side(ux_num,i,0);
        if (spaceDim > 1)
          yconv = wkset->local_soln_side(uy_num,i,0);
        if (spaceDim > 2)
          zconv = wkset->local_soln_side(uz_num,i,0);
      }
      else { 
        xconv = this->convectionTerm(x, y, z, current_time, 1);
        yconv = this->convectionTerm(x, y, z, current_time, 2);
        zconv = this->convectionTerm(x, y, z, current_time, 3);
      }

      AD diff = getDiff(x,y,z);
      ScalarT convscale = 0.0;
      if (spaceDim == 1) {
        convscale = abs(xconv.val()*wkset->normals(0,i,0));
      }
      else if (spaceDim == 2) {
        convscale = abs(xconv.val()*wkset->normals(0,i,0)) + abs(yconv.val()*wkset->normals(0,i,1));
      }
      else if (spaceDim == 3) {
        convscale = abs(xconv.val()*wkset->normals(0,i,0)) + abs(yconv.val()*wkset->normals(0,i,1)) + abs(zconv.val()*wkset->normals(0,i,2));
      }
      AD burgscale = 0.0;
      if (burgersflux) {
        if (spaceDim == 2) {
          burgscale = abs(c*1.0*wkset->normals(0,i,0));
        }
          
      }
      AD penalty = 10.0*(diff/wkset->h + convscale + burgscale);
      
      wkset->flux(cnum,i) += (sf*diff*dcdx-xconv*c)*wkset->normals(0,i,0) + penalty*(lambda-c);
      if (spaceDim > 1)
        wkset->flux(cnum,i) += (sf*diff*dcdy-yconv*c)*wkset->normals(0,i,1);
      if (spaceDim > 2)
        wkset->flux(cnum,i) += (sf*diff*dcdz-zconv*c)*wkset->normals(0,i,2);
      if (burgersflux) {
        wkset->flux(cnum,i) += -0.5*c*c*wkset->normals(0,i,0);
      }
    }
  
  }

  // ========================================================================================
  // ========================================================================================
  
  AD getDirichletValue(const string & var, const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t, const string & gside,
                       const bool & useadjoint) const {
    
    AD val = 0.0;
    
    if(boundary_params.size() > 0 && !useadjoint){
      val = boundary_params[0];
    }
    
    return val;
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  ScalarT getInitialValue(const string & var, const ScalarT & x, const ScalarT & y, const ScalarT & z, const bool & useadjoint) const {
    ScalarT val = 0.0;
    
    if(init_params.size() > 0 && !useadjoint){
      val = init_params[0].val();
    }   
    
    if (test == 2) {
      if (x>=-1.0) {
        if (x<=0.0)
          val = x+1.0;
        else if (x<=1.0)
          val = 1.0-x;
      }
    }
    return val;
  }
  
  // ========================================================================================
  // Get the initial value
  // ========================================================================================
  
  FC getInitial(const DRV & ip, const string & var, const ScalarT & time, const bool & isAdjoint) const {
    int numip = ip.dimension(1);
    FC initial(1,numip);
    
    if (test == 2) {
      for (int i=0; i<numip; i++) {
        ScalarT x = ip(0,i,0);
        ScalarT val = 0.0;
        if (x>=-1.0) {
          if (x<=0.0)
            val = x+1.0;
          else if (x<=1.0)
            val = 1.0-x;
        }
        initial(0,i) = val;
      }
    }
    return initial;
  }
  
  // ========================================================================================
  /* return alpha for Robin boundary condition */
  // ========================================================================================
  
  AD robinAlpha(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t, const string & side,
                const AD & xconv, const AD & yconv, const AD & zconv, const AD & diff) const {
    
    AD val = 0.0;
    if(noFlux){ //no-flux boundary conditions for rectangle/cube
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
  // return the boundary source term  
  // ========================================================================================
  
  AD boundarySource(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t, const string & side) const {
    ScalarT val = 0.0;
    
    return val;
  }
  
  // ========================================================================================
  // error calculation
  // ========================================================================================
  
  ScalarT trueSolution(const string & var, const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const {
    
    ScalarT c = 0.0;
    
    if(verbosity > 0)
      cout << "No known true solution...'truth' defaults to zero..." << endl;
    
    return c;
  }
  
  // ========================================================================================
  // response calculation
  // ========================================================================================

  FCAD response(const FCAD & local_soln, const FCAD & local_soln_grad, const FCAD & local_psoln,
                const FCAD & local_psoln_grad, const DRV & ip, const ScalarT & time) {
    int numip = ip.dimension(1);
    
    FCAD resp(numResponses,numip);
    
    ScalarT x = 0.0;
    ScalarT y = 0.0;
    
    AD regPenalty = 0.0;
    if (test == 102 && abs(time-finTime)<1.e-8){
      for (int k=0; k<source_decay_params.size(); k++)
        regPenalty += regParam*source_decay_params[k]*source_decay_params[k];
    }
    
      for (int j=0; j<numip; j++) {
        if(test == 102){
          resp(0,j) = local_soln(cnum,j,0);
        }
        else if(test == 104){
          x = ip(0,j,0);
          y = ip(0,j,1);
          if((abs(time-finTime))<1.e-4){
            if(sqrt(0.25*(x-0.45)*(x-0.45)+(y-0.7)*(y-0.7)) < 0.1){ //north america
              resp(0,j) = 0.5*pow((local_soln(cnum,j,0)-0.42),2.0);
            }
            else if(sqrt((x-0.45)*(x-0.45)+(y-0.55)*(y-0.55)) < 0.05){ //central america
              resp(0,j) = 0.5*pow((local_soln(cnum,j,0)-0.25),2.0);
            }
            else if(sqrt((x-0.7)*(x-0.7)+(y-0.35)*(y-0.35)) < 0.15){ //south america
              resp(0,j) = 0.5*pow((local_soln(cnum,j,0)-0.25),2.0);
            }
            else if(sqrt((x-1.05)*(x-1.05)+(y-0.75)*(y-0.75)) < 0.1){ //europe
              resp(0,j) = 0.5*pow((local_soln(cnum,j,0)-0.32),2.0);
            }
            else if(sqrt((x-1.1)*(x-1.1)+(y-0.4)*(y-0.4)) < 0.15){ //africa
              resp(0,j) = 0.5*pow((local_soln(cnum,j,0)-0.19),2.0);
            }
            else if(sqrt((x-1.6)*(x-1.6)+(y-0.6)*(y-0.6)) < 0.2){ //asia
              resp(0,j) = 0.5*pow((local_soln(cnum,j,0)-0.28),2.0);
            }
            else if(sqrt((x-1.75)*(x-1.75)+(y-0.3)*(y-0.3)) < 0.1){ //australia
              resp(0,j) = 0.5*pow((local_soln(cnum,j,0)-0.19),2.0);
            }
          }
          if(abs(time-finTime)<1.e-8)
            resp(0,j) += regPenalty;
        }
        else //"default"
          resp(0,j) = local_soln(cnum,j,0);
      }
    
    
    return resp;
  }
  
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD scalarRespFunc(const FCAD & integralResponses, const bool & justDeriv) const {
    FCAD resp(integralResponses.dimension(0));
    if(test == 100){
      resp(0) = (1./(data_noise_std*data_noise_std))*integralResponses(0);
      
      if(!justDeriv){
        //prior on source mags
        ScalarT priorvar = 1.0; //prior covariance = I
        for (int k=0; k<smag_params.size(); k++)
          resp(0) += 0.5*(1./priorvar)*smag_params[k]*smag_params[k]; 
      }
    }else{ 
      for(int i=0; i<integralResponses.dimension(0); i++){
        resp(i) = integralResponses(i);
      }
    }
    
    return resp;
  }
  
  bool useScalarRespFunc() const{
    return useScalarRespFx;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD target(const FC & ip, const ScalarT & time) {
    int numip = ip.dimension(1);
    FCAD targ(numResponses,numip);
    for (int j=0; j<numip; j++) {
      for(int ii=0; ii<numResponses; ii++)
        targ(ii,j) = 0.0;
    }
    
    return targ;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(vector<string> & varlist_) {
    varlist = varlist_;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "c")
        cnum = i;
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
  
  void setUserDefined(Teuchos::RCP<UserDefined> & udfunc_) {
    udfunc = udfunc_;
  }
  
  // ========================================================================================
  // return the value of the stabilization parameter 
  // ========================================================================================
  
  template<class T>  
  T computeTau(const T & localdiff, const T & xvl, const T & yvl, const T & zvl, const ScalarT & h) const {
    
    ScalarT C1 = 4.0;
    ScalarT C2 = 2.0;
    
    T nvel;
    if (spaceDim == 1)
      nvel = xvl*xvl;
    else if (spaceDim == 2)
      nvel = xvl*xvl + yvl*yvl;
    else if (spaceDim == 3)
      nvel = xvl*xvl + yvl*yvl + zvl*zvl;
    
    if (nvel > 1E-12)
      nvel = sqrt(nvel);
    
    return 4.0/(C1*localdiff/h/h + C2*(nvel)/h); //msconvdiff has a 1.0 instead of a 4.0 in the numerator
    
  }
  
  // ========================================================================================
  // return the source term  
  // ========================================================================================
  
  AD sourceTerm(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const {
    
    AD val = 0.0;
    
    if(test == 102){
      ScalarT ctrlStartTime = 2.0;
      int numTimeBlocks = 3;
      int num_source = smag_params.size();
      int timeBlock;
      if(time > ctrlStartTime)
        timeBlock = floor(std::min(std::max(((time-ctrlStartTime)-1.e-8),0.0),(finTime-ctrlStartTime))/((finTime-ctrlStartTime)/numTimeBlocks));
      else
        timeBlock = -1;
      
      if(timeBlock == numTimeBlocks)
        timeBlock = numTimeBlocks - 1; //numerical noise   
      
      for (int i=0; i<num_source; i++){ //controlled anthro sources
        if(time > ctrlStartTime){
          val += (smag_params[i].val()-source_decay_params[i+timeBlock*num_source])*
          (exp((-1.0*source_xx_cov_params[i].val())*(x-sxloc_params[i].val())*(x-sxloc_params[i].val())
               + (-1.0*source_yy_cov_params[i].val())*(y-syloc_params[i].val())*(y-syloc_params[i].val())
               + (-2.0*source_xy_cov_params[i].val())*(x-sxloc_params[i].val())*(y-syloc_params[i].val()) ));}
        else{
          val += smag_params[i].val()*
          (exp((-1.0*source_xx_cov_params[i].val())*(x-sxloc_params[i].val())*(x-sxloc_params[i].val())
               + (-1.0*source_yy_cov_params[i].val())*(y-syloc_params[i].val())*(y-syloc_params[i].val())
               + (-2.0*source_xy_cov_params[i].val())*(x-sxloc_params[i].val())*(y-syloc_params[i].val()) ));}
      }
      for (int i=0; i<source_mag_extra.size(); i++){ //not-controlled anthro sources and ocean sinks
        val += source_mag_extra[i].val()*
        (exp((-1.0*source_xx_cov_xtra[i].val())*(x-source_xloc_xtra[i].val())*(x-source_xloc_xtra[i].val())
             + (-1.0*source_yy_cov_xtra[i].val())*(y-source_yloc_xtra[i].val())*(y-source_yloc_xtra[i].val())
             + (-2.0*source_xy_cov_xtra[i].val())*(x-source_xloc_xtra[i].val())*(y-source_yloc_xtra[i].val()) ));
      }
      for (int i=0; i<smagxtra_stoch.size(); i++){ //vegetation sources/sinks
        ScalarT xform_mag = 0.0;
        if(analysis_type == "SOL"){
          for (int j=0; j<postCovChol[i].size(); j++)
            xform_mag += smagxtra_stoch[j].val()*postCovChol[i][j];
          xform_mag += postMeanSource[i];
        }
        val += xform_mag*
        (exp((-1.0*source_xxcov_xtra_stoch[i].val())*(x-xlocxtra_stoch[i].val())*(x-xlocxtra_stoch[i].val())
             + (-1.0*source_yycov_xtra_stoch[i].val())*(y-ylocxtra_stoch[i].val())*(y-ylocxtra_stoch[i].val())
             + (-2.0*source_xycov_xtra_stoch[i].val())*(x-xlocxtra_stoch[i].val())*(y-ylocxtra_stoch[i].val()) ));
      }
    }else if (test>100) {
      for (int i=0; i<smag_params.size(); i++){ 
        val += smag_params[i]*
        (exp((-1.0*source_xx_cov_params[i].val())*(x-sxloc_params[i].val())*(x-sxloc_params[i].val())
             + (-1.0*source_yy_cov_params[i].val())*(y-syloc_params[i].val())*(y-syloc_params[i].val())
             + (-2.0*source_xy_cov_params[i].val())*(x-sxloc_params[i].val())*(y-syloc_params[i].val()) ));
      }
      for (int i=0; i<source_mag_extra.size(); i++){ 
        val += source_mag_extra[i].val()*
        (exp((-1.0*source_xx_cov_xtra[i].val())*(x-source_xloc_xtra[i].val())*(x-source_xloc_xtra[i].val())
             + (-1.0*source_yy_cov_xtra[i].val())*(y-source_yloc_xtra[i].val())*(y-source_yloc_xtra[i].val())
             + (-2.0*source_xy_cov_xtra[i].val())*(x-source_xloc_xtra[i].val())*(y-source_yloc_xtra[i].val()) ));
      }
      for (int i=0; i<smagxtra_stoch.size(); i++){
        val += smagxtra_stoch[i]*
        (exp((-1.0*source_xxcov_xtra_stoch[i].val())*(x-xlocxtra_stoch[i].val())*(x-xlocxtra_stoch[i].val())
             + (-1.0*source_yycov_xtra_stoch[i].val())*(y-ylocxtra_stoch[i].val())*(y-ylocxtra_stoch[i].val())
             + (-2.0*source_xycov_xtra_stoch[i].val())*(x-xlocxtra_stoch[i].val())*(y-ylocxtra_stoch[i].val()) ));
      }
      if(source_const.size() > 0)
        val += source_const[0];
    }
    else if (test == 2) {
      val = 0.0;
    }
    else if (test == 20) {
      ScalarT xlocs[3] = { 52.275, 104.7, 157.125 };
      ScalarT ylocs[3] = { 52.275, 104.7, 157.125 };
      ScalarT zlocs[1] = { 100.5 };
      ScalarT x,y,z;
      //ScalarT width_factor = 2*pow(0.01,2);
      ScalarT width_factor = 10000.0;
      int numip = wkset->ip.dimension(1);
      for (int k=0; k<numip; k++) {
	x = wkset->ip(0,k,0);
	y = wkset->ip(0,k,1);
	z = wkset->ip(0,k,2);
	for (int i = 0;i < 3;i++) {
	  for (int j = 0;j < 3; j++) {
	    
	    val += exp(-((x - xlocs[i])*(x - xlocs[i]) +
			(y - ylocs[j])*(y - ylocs[j]) +
			(z - zlocs[0])*(z - zlocs[0])
			 ) / width_factor);
	    //   cout <<  (x - xlocs[i])*(x - xlocs[i]) << " "
	    //		 <<  (y - ylocs[j])*(y - ylocs[j]) << " "
	    //	 <<  (z - zlocs[0])*(z - zlocs[0]) << " "
            //     <<  val << endl;
	    // if (val > 1.0) {
	    //  cout << x << " " << y << " " << z << " " << val << endl;
	  }
	}
      }
    } else {
      val = 1.0;
    }
    
    return val;
    
  }
  
  // ========================================================================================
  // return the convection term  
  // ========================================================================================
  
  AD convectionTerm(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t, const int & dir) const {
    AD vel = 0.0;
    
    if((test == 100 || test == 102) && dir < 3){
      ScalarT vort_NAx, vort_NAy, vort_SAx, vort_SAy, vort_NPlx, vort_NPly, vort_NPrx, vort_NPry, vort_SPx, vort_SPy, vort_Ix, vort_Iy;
      if(moveVort){
        vort_NAx = 0.8 + 0.05*sin(3.0*t*PI); vort_NAy = 0.65 + 0.05*sin(3.0*t*PI); //north atlantic
        vort_SAx = 0.95 + 0.05*sin(2.0*t*PI); vort_SAy = 0.35 + 0.05*sin(2.0*t*PI); //south atlantic
        vort_NPlx = 0.0 + 0.05*sin(2.0*t*PI); vort_NPly = 0.7; //north pacific, left
        vort_NPrx = 2.0 + 0.05*sin(2.0*t*PI); vort_NPry = 0.7; //north pacific, right
        vort_SPx = 0.3 + 0.05*sin(3.0*t*PI); vort_SPy = 0.35 + 0.05*sin(3.0*t*PI); //south pacific
        vort_Ix = 1.45 + 0.05*cos(2.0*t*PI); vort_Iy = 0.4 + 0.05*sin(2.0*t*PI); //indian
      }else{
        vort_NAx = 0.8; vort_NAy = 0.65; //north atlantic
        vort_SAx = 0.95; vort_SAy = 0.35; //south atlantic
        vort_NPlx = 0.0; vort_NPly = 0.7; //north pacific, left
        vort_NPrx = 2.0; vort_NPry = 0.7; //north pacific, right
        vort_SPx = 0.3; vort_SPy = 0.35; //south pacific
        vort_Ix = 1.45; vort_Iy = 0.4; //indian
      }
      
      ScalarT r0 = 0.1; //cutoff radius, below which velocity no longer obeys vortex (too fast near center)
      
      ScalarT xNA = x - vort_NAx; ScalarT yNA = y - vort_NAy; ScalarT rNA = sqrt(xNA*xNA+yNA*yNA);
      xNA = std::max(r0/rNA,1.0)*xNA; yNA = std::max(r0/rNA,1.0)*yNA;
      ScalarT xSA = x - vort_SAx; ScalarT ySA = y - vort_SAy; ScalarT rSA = sqrt(xSA*xSA+ySA*ySA);
      xSA = std::max(r0/rSA,1.0)*xSA; ySA = std::max(r0/rSA,1.0)*ySA;
      ScalarT xNPl = x - vort_NPlx; ScalarT yNPl = y - vort_NPly; ScalarT rNPl = sqrt(xNPl*xNPl+yNPl*yNPl);
      xNPl = std::max(r0/rNPl,1.0)*xNPl; yNPl = std::max(r0/rNPl,1.0)*yNPl;
      ScalarT xNPr = x - vort_NPrx; ScalarT yNPr = y - vort_NPry; ScalarT rNPr = sqrt(xNPr*xNPr+yNPr*yNPr);
      xNPr = std::max(r0/rNPr,1.0)*xNPr; yNPr = std::max(r0/rNPr,1.0)*yNPr;
      ScalarT xSP = x - vort_SPx; ScalarT ySP = y - vort_SPy; ScalarT rSP = sqrt(xSP*xSP+ySP*ySP);
      xSP = std::max(r0/rSP,1.0)*xSP; ySP = std::max(r0/rSP,1.0)*ySP;
      ScalarT xI = x - vort_Ix; ScalarT yI = y - vort_Iy; ScalarT rI = sqrt(xI*xI+yI*yI);
      xI = std::max(r0/rI,1.0)*xI; yI = std::max(r0/rI,1.0)*yI;
      
      if (dir == 1){
        vel = (convection_params[0])*(yNA/(xNA*xNA+yNA*yNA))
        + (convection_params[1])*(ySA/(xSA*xSA+ySA*ySA))
        + (convection_params[0])*(yNPl/(xNPl*xNPl+yNPl*yNPl))
        + (convection_params[0])*(yNPr/(xNPr*xNPr+yNPr*yNPr))
        + (convection_params[1])*(ySP/(xSP*xSP+ySP*ySP))
        + (convection_params[1])*(yI/(xI*xI+yI*yI));
        
        //westerlies, easterlies, trade winds
        if (y < 1.0/3.0 || y > 2.0/3.0)        
          vel += -(convection_params[2])*(sin(y*6.0*PI)); 
        else
          vel += (convection_params[2])*(sin(y*3.0*PI)); 
        
      }
      else if (dir == 2){
        vel = -(convection_params[0])*(xNA/(xNA*xNA+yNA*yNA))
        -(convection_params[1])*(xSA/(xSA*xSA+ySA*ySA))
        -(convection_params[0])*(xNPl/(xNPl*xNPl+yNPl*yNPl))
        -(convection_params[0])*(xNPr/(xNPr*xNPr+yNPr*yNPr))
        -(convection_params[1])*(xSP/(xSP*xSP+ySP*ySP))
        -(convection_params[1])*(xI/(xI*xI+yI*yI));
        
        vel += (convection_params[2])*(sin(y*6.0*PI)); //westerlies, easterlies, trade winds
      }
    }else{ //"default"
      if(dir == 1)
        vel = convection_params[0];
      else if(dir == 2)
        vel = convection_params[1];
      else if(dir == 3)
        vel = convection_params[2]; 
    }
    
    return vel;
    
  }
  
  // ========================================================================================
  // return the reaction term  
  // ========================================================================================
  
  
  AD reactionTerm(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t, const AD & c) const {
    AD r = reaction_params[0]*c*c;
    return r;
  }
  
  // ========================================================================================
  // return the diffusion at a point 
  // ========================================================================================
  
  AD getDiff(const ScalarT & x, const ScalarT & y, const ScalarT & z) const {
    AD diff = diff_params[0];
    return diff;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  int getNumResponses() {
    return numResponses;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  vector<string> extraFieldNames() const {
    vector<string> ef;
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
  
  vector<FC> extraFields(const FC & ip, const ScalarT & time) {
    vector<FC> ef;
    return ef;
  }

  // ========================================================================================
  // ========================================================================================

  vector<FC> extraCellFields(const FC & ip, const ScalarT & time) const {
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
  
  void updateParameters(const std::vector<std::vector<AD> > & params, const std::vector<string> & paramnames) {
    for (size_t p=0; p<paramnames.size(); p++) {
      if (paramnames[p] == "cd_diff") 
        diff_params = params[p];
      else if (paramnames[p] == "Source Mag")
        smag_params = params[p];
      else if (paramnames[p] == "Source xloc")
        sxloc_params = params[p];
      else if (paramnames[p] == "Source yloc")
        syloc_params = params[p];
      else if (paramnames[p] == "Source zloc")
        szloc_params = params[p];
      else if (paramnames[p] == "reaction")
        reaction_params = params[p];
      else if (paramnames[p] == "cd_convection")
        convection_params = params[p];
      else if (paramnames[p] == "source_decay")
        source_decay_params = params[p];
      else if (paramnames[p] == "source_xxcov")
        source_xx_cov_params = params[p];
      else if (paramnames[p] == "source_yycov")
        source_yy_cov_params = params[p];
      else if (paramnames[p] == "source_xycov")
        source_xy_cov_params = params[p];
      else if (paramnames[p] == "source_mag_xtra")
        source_mag_extra = params[p];
      else if (paramnames[p] == "source_xloc_xtra")
        source_xloc_xtra = params[p];
      else if (paramnames[p] == "source_yloc_xtra")
        source_yloc_xtra = params[p];
      else if (paramnames[p] == "source_xxcov_xtra")
        source_xx_cov_xtra = params[p];
      else if (paramnames[p] == "source_xycov_xtra")
        source_xy_cov_xtra = params[p];
      else if (paramnames[p] == "source_yycov_xtra")
        source_yy_cov_xtra = params[p];
      else if (paramnames[p] == "source_mag_xtra_stoch")
        smagxtra_stoch = params[p];
      else if (paramnames[p] == "source_xloc_xtra_stoch")
        xlocxtra_stoch = params[p];
      else if (paramnames[p] == "source_yloc_xtra_stoch")
        ylocxtra_stoch = params[p];
      else if (paramnames[p] == "source_xxcov_xtra_stoch")
        source_xxcov_xtra_stoch = params[p];
      else if (paramnames[p] == "source_xycov_xtra_stoch")
        source_xycov_xtra_stoch = params[p];
      else if (paramnames[p] == "source_yycov_xtra_stoch")
        source_yycov_xtra_stoch = params[p];
      else if (paramnames[p] == "convdiff_boundary")
	      boundary_params = params[p];
      else if (paramnames[p] == "convdiff_init")
	      init_params = params[p];
      else if (paramnames[p] == "convdiff_source_const")
        source_const = params[p];
      else if(verbosity > 0) //false alarms if multiple physics modules used...
        cout << "Parameter not used in convdiff: " << paramnames[p] << endl;  
    }
  }
  
private:
  
  Teuchos::RCP<UserDefined> udfunc;
  
  int spaceDim, numElem, numParams, numResponses;
  size_t numip, numip_side;
  
  
  std::vector<AD> diff_params, convection_params, reaction_params,
  source_const,
  smag_params, sxloc_params, syloc_params, szloc_params, source_decay_params,
  source_xx_cov_params, source_xy_cov_params, source_yy_cov_params, 
  source_mag_extra, source_xloc_xtra, source_yloc_xtra,
  source_xx_cov_xtra, source_xy_cov_xtra, source_yy_cov_xtra, 
  smagxtra_stoch, xlocxtra_stoch, ylocxtra_stoch,
  source_xxcov_xtra_stoch, source_xycov_xtra_stoch, source_yycov_xtra_stoch,
  boundary_params, init_params;
  
  
  bool isTD, useSUPG, burgersflux;
  
  vector<string> varlist;
  int cnum;
  
  int test;
  
  vector<string> sideSets;
  
  std::string analysis_type; //to know when parameter is a sample that needs to be transformed
  
  ScalarT finTime; //for response function
  ScalarT regParam; //regularization parameter
  bool moveVort; //to test effect of moving vortices
  
  bool noFlux; //whether non-Dirichlet boundary defaults to homo Neumann or no-flux boundary condition 
  int verbosity;
  
  bool velFromNS; //whether to get velocity from navierstokes
  ScalarT ux_num, uy_num, uz_num; //in case getting velocity from navierstokes
  
  ScalarT data_noise_std; //standard devation of additive 0-centered Gaussian noise
  vector<vector<ScalarT> > postCovChol; //to transform standard normal samples to samples from posterior of (approximation of) linear-Gauss inference
  vector<ScalarT> postMeanSource; //to transform standard normal samples to samples from posterior of (approximation of) linear-Gauss inference
  
  bool useScalarRespFx;
  
  ScalarT PI;
};

#endif
