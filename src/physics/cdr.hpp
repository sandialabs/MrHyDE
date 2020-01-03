/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef CDR_H
#define CDR_H

#include "physics_base.hpp"

class cdr : public physicsbase {
public:
  
  cdr() {} ;
  
  ~cdr() {};
  
  // ========================================================================================
  // ========================================================================================
  
  cdr(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
      const size_t & numip_side_, const int & numElem_,
      Teuchos::RCP<FunctionInterface> & functionManager_,
      const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_), functionManager(functionManager_),
  blocknum(blocknum_) {
  
    label = "cdr";
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    myvars.push_back("c");
    mybasistypes.push_back("HGRAD");
    
    //velFromNS = settings->sublist("Physics").get<bool>("Get velocity from navierstokes",false);
    //burgersflux = settings->sublist("Physics").get<bool>("Add Burgers",false);
    
    // Functions
    Teuchos::ParameterList fs = settings->sublist("Functions");
    
    functionManager->addFunction("source",fs.get<string>("source","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("diffusion",fs.get<string>("diffusion","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("density",fs.get<string>("density","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reaction",fs.get<string>("reaction","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("xvel",fs.get<string>("xvel","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("yvel",fs.get<string>("yvel","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("zvel",fs.get<string>("zvel","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("SUPG tau",fs.get<string>("SUPG tau","0.0"),numElem,numip,"ip",blocknum);
    
    //functionManager->addFunction("thermal Neumann source",fs.get<string>("thermal Neumann source","0.0"),numElem,numip_side,"side ip",blocknum);
    functionManager->addFunction("diffusion",fs.get<string>("diffusion","1.0"),numElem,numip_side,"side ip",blocknum);
    functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),numElem,numip_side,"side ip",blocknum);
    
    //regParam = settings->sublist("Analysis").sublist("ROL").get<ScalarT>("regularization parameter",1.e-6);
    //moveVort = settings->sublist("Physics").get<bool>("moving vortices",true);
    //finTime = settings->sublist("Solver").get<ScalarT>("finaltime",1.0);
    //data_noise_std = settings->sublist("Analysis").get("Additive Normal Noise Standard Dev",0.0);
  }
  
  // ========================================================================================
  // ========================================================================================
 
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int c_basis_num = wkset->usebasis[cnum];
    basis = wkset->basis[c_basis_num];
    basis_grad = wkset->basis_grad[c_basis_num];
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      source = functionManager->evaluate("source","ip",blocknum);
      diff = functionManager->evaluate("diffusion","ip",blocknum);
      cp = functionManager->evaluate("specific heat","ip",blocknum);
      rho = functionManager->evaluate("density","ip",blocknum);
      reax = functionManager->evaluate("reaction","ip",blocknum);
      xvel = functionManager->evaluate("xvel","ip",blocknum);
      yvel = functionManager->evaluate("yvel","ip",blocknum);
      zvel = functionManager->evaluate("zvel","ip",blocknum);
      tau = functionManager->evaluate("SUPG tau","ip",blocknum);
    }
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    if (spaceDim == 1) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(cnum,i); // TMW: e_num is not on the assembly device
            res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,cnum,k,0)*basis(e,i,k) + // transient term
            diff(e,k)*(sol_grad(e,cnum,k,0)*basis_grad(e,i,k,0)) + // diffusion terms
            (xvel(e,k)*sol_grad(e,cnum,k,0))*basis(e,i,k) + // convection terms
            reax(e,k)*basis(e,i,k) - source(e,k)*basis(e,i,k); // reaction and source terms
            
            res(e,resindex) += tau(e,k)*(rho(e,k)*cp(e,k)*sol_dot(e,cnum,k,0) + xvel(e,k)*sol_grad(e,cnum,k,0) + reax(e,k) - source(e,k))*(xvel(e,k)*basis_grad(e,i,k,0));
            
          }
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(cnum,i); // TMW: e_num is not on the assembly device
            res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,cnum,k,0)*basis(e,i,k) + // transient term
            diff(e,k)*(sol_grad(e,cnum,k,0)*basis_grad(e,i,k,0) + sol_grad(e,cnum,k,1)*basis_grad(e,i,k,1)) + // diffusion terms
            (xvel(e,k)*sol_grad(e,cnum,k,0) + yvel(e,k)*sol_grad(e,cnum,k,1))*basis(e,i,k) + // convection terms
            reax(e,k)*basis(e,i,k) - source(e,k)*basis(e,i,k); // reaction and source terms
            
            //res(e,resindex) += tau(e,k)*(rho(e,k)*cp(e,k)*sol_dot(e,cnum,k,0) + xvel(e,k)*sol_grad(e,cnum,k,0) + yvel(e,k)*sol_grad(e,cnum,k,1) + reax(e,k) - source(e,k))*(xvel(e,k)*basis_grad(e,i,k,0) + yvel(e,k)*basis_grad(e,i,k,1));
            
          }
        }
      });
    }
    else if (spaceDim == 3) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(cnum,i); // TMW: e_num is not on the assembly device
            res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,cnum,k,0)*basis(e,i,k) + // transient term
            diff(e,k)*(sol_grad(e,cnum,k,0)*basis_grad(e,i,k,0) + sol_grad(e,cnum,k,1)*basis_grad(e,i,k,1) + sol_grad(e,cnum,k,2)*basis_grad(e,i,k,2)) + // diffusion terms
            (xvel(e,k)*sol_grad(e,cnum,k,0) + yvel(e,k)*sol_grad(e,cnum,k,1) + zvel(e,k)*sol_grad(e,cnum,k,2))*basis(e,i,k) + // convection terms
            reax(e,k)*basis(e,i,k) - source(e,k)*basis(e,i,k); // reaction and source terms
            
            res(e,resindex) += tau(e,k)*(rho(e,k)*cp(e,k)*sol_dot(e,cnum,k,0) + xvel(e,k)*sol_grad(e,cnum,k,0) + yvel(e,k)*sol_grad(e,cnum,k,1) + zvel(e,k)*sol_grad(e,cnum,k,2) +reax(e,k) - source(e,k))*(xvel(e,k)*basis_grad(e,i,k,0) + yvel(e,k)*basis_grad(e,i,k,1) + zvel(e,k)*basis_grad(e,i,k,2));
            
          }
        }
      });
    }
  }
    /*
                   
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
        
        wkset->res(resindex) += diff*dcdx*dvdx - xconv*c*dvdx - source*v + reaction*v;
        if (spaceDim > 1)
          wkset->res(resindex) += diff*dcdy*dvdy - yconv*c*dvdy;
        if (spaceDim > 2)
          wkset->res(resindex) += diff*dcdz*dvdz - zconv*c*dvdz;
        
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
      }
    }
    
  }*/
  
  // ========================================================================================
  // ========================================================================================
 
  void boundaryResidual() {
    
    /*
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
    */
  }
  
  // ========================================================================================
  // ========================================================================================
 
  void edgeResidual() {}
 
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================

  void computeFlux() {

    /*
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
  */
  }

  // ========================================================================================
  // ========================================================================================
  
  void setVars(vector<string> & varlist_) {
    varlist = varlist_;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "c") {
        cnum = i;
      }
    }
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
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;
  
  FDATA diff, rho, cp, xvel, yvel, zvel, reax, tau, source, nsource, diff_side, robin_alpha;
  
  int spaceDim, numElem;
  size_t numip, numip_side, blocknum;
  vector<string> varlist;
  int cnum, resindex;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::cdr::computeFlux() - evaluation of flux");
  
  
};

#endif
