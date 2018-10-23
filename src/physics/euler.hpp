/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef EULER_H
#define EULER_H

#include "physics_base.hpp"

class euler : public physicsbase {
public:
  
  euler() {} ;
  
  ~euler() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  euler(Teuchos::RCP<Teuchos::ParameterList> & settings) {
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    // Parameters
    
    if (settings->sublist("Solver").get<int>("solver",0) == 1)
      isTD = true;
    else
      isTD = false;
    
    if (isTD) {
      double finalT = settings->sublist("Solver").get<double>("finaltime",0.0);
      numSteps = settings->sublist("Solver").get<int>("numSteps",1);
      dt = finalT / numSteps;
    }
    numResponses = 1;
    
    test = settings->sublist("Physics").get<string>("test","Sod");
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
    
    // assume all variables use the same basis (easy to generalize)
    int curr_basis = usebasis[rho_num];
    int numBasis = basis[curr_basis].dimension(1);
    int numCubPoints = ip.dimension(1);
    
    // Set the parameters
    
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    double v, dvdx, dvdy, dvdz; 
    double source_rho, source_rhou, source_rhov, source_rhow, source_rhoe;
    //double ux_prev, uy_prev, uz_prev;
    AD p, rho, drhodx, drhody, drhodz, rho_dot;
    AD rhou, drhoudx, drhoudy, drhoudz, rhou_dot;
    AD rhov, drhovdx, drhovdy, drhovdz, rhov_dot;
    AD rhow, drhowdx, drhowdy, drhowdz, rhow_dot;
    AD rhoe, drhoedx, drhoedy, drhoedz, rhoe_dot;
    
    AD evisc_mu, evisc_kappa, evisc_nu;
    
    // We make the stabilization params and strong residuals AD objects since
    // they are similar to the local residual
    AD tau, res, res1, res2, res3;
    
    double gamma = 1.4;
    double Pe = 0.5, Prho = 0.5;
    
    for( int i=0; i<numBasis; i++ ) {
      for( int nPt=0; nPt<numCubPoints; nPt++ ) {
        
        // gather up all the information at the integration point
        
        v = basis[curr_basis](0,i,nPt);
        dvdx = basis_grad[curr_basis](0,i,nPt,0);
        x = ip(0,nPt,0);
        
        rho = local_soln(rho_num,nPt);
        rho_dot = local_soln_dot(rho_num,nPt);
        drhodx = local_solngrad(rho_num,nPt,0);
        
        if (rho.val() < 0.0)
          cout << "Warnng: rho = " << rho << endl;
        
        rhou = local_soln(rhou_num,nPt);
        rhou_dot = local_soln_dot(rhou_num,nPt);
        drhoudx = local_solngrad(rhou_num,nPt,0);
        
        rhoe = local_soln(rhoe_num,nPt);
        rhoe_dot = local_soln_dot(rhoe_num,nPt);
        drhoedx = local_solngrad(rhoe_num,nPt,0);
        
        
        if (spaceDim > 1) {
          rhov = local_soln(rhov_num,nPt);
          rhov_dot = local_soln_dot(rhov_num,nPt);
          drhovdx = local_solngrad(rhov_num,nPt,0);
          y = ip(0,nPt,1);
          drhody = local_solngrad(rho_num,nPt,1);
          drhoedy = local_solngrad(rhoe_num,nPt,1);
          drhovdy = local_solngrad(rhov_num,nPt,1);
          dvdy = basis_grad[curr_basis](0,i,nPt,1);
        }
        
        if (spaceDim > 2) {
          rhow = local_soln(rhow_num,nPt);
          rhow_dot = local_soln_dot(rhow_num,nPt);
          drhowdx = local_solngrad(rhow_num,nPt,0);
          drhowdy = local_solngrad(rhow_num,nPt,1);
          drhowdz = local_solngrad(rhow_num,nPt,2);
          z = ip(0,nPt,2);
          drhodz = local_solngrad(rho_num,nPt,2);
          drhoudz = local_solngrad(rhou_num,nPt,2);
          drhovdz = local_solngrad(rhov_num,nPt,2);
          drhoedz = local_solngrad(rhoe_num,nPt,2);
          dvdz = basis_grad[curr_basis](0,i,nPt,2);
        }
        
        evisc_mu = entropyViscosity(rho, drhodx, drhody, drhodz, rho_dot,
                                    rhou, drhoudx, drhoudy, drhoudz, rhou_dot,
                                    rhov, drhovdx, drhovdy, drhovdz, rhov_dot,
                                    rhow, drhowdx, drhowdy, drhowdz, rhow_dot,
                                    rhoe, drhoedx, drhoedy, drhoedz, rhoe_dot,
                                    h, params);
        evisc_kappa = Pe/(gamma-1)*evisc_mu;
        evisc_nu = Prho/rho*evisc_mu;
        // multiply and add into local residual
        
        p = (gamma-1.0)*(rhoe - 0.5/rho*(rhou*rhou + rhov*rhov + rhow*rhow));
        /*
         if (var == "rho") {
         local_resid(e,i) += rho_dot*v - rhou*dvdx + evisc_nu*(drhodx*dvdx);
         if (spaceDim>1)
         local_resid(e,i) += -rhov*dvdy + evisc_nu*drhody*dvdy;
         if (spaceDim>2)
         local_resid(e,i) += -rhow*dvdz + evisc_nu*drhodz*dvdz;
         
         if(useSUPG) {
         //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
         }
         }
         if (var == "rhou") {
         local_resid(e,i) += rhou_dot*v - (rhou*rhou/rho + p)*dvdx + evisc_mu*(drhoudx)*dvdx;
         if (spaceDim>1)
         local_resid(e,i) += -(rhou*rhov/rho)*dvdy + evisc_mu*0.5*(drhoudy+drhoudy)*dvdy;
         //local_resid(e,i) += -(rhou*rhov/rho)*dvdy + evisc_mu*0.5*(drhoudy+drhovdx)*dvdy;
         if (spaceDim>2)
         local_resid(e,i) += -(rhou*rhow/rho)*dvdz + evisc_mu*0.5*(drhoudz+drhoudz)*dvdz;
         //local_resid(e,i) += -(rhou*rhow/rho)*dvdz + evisc_mu*0.5*(drhoudz+drhowdx)*dvdz;
         
         if(useSUPG) {
         //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
         }
         }
         if (var == "rhov") {
         //local_resid(e,i) += -(rhov*rhou/rho)*dvdx + evisc_mu*0.5*(drhoudy+drhovdx)*dvdx;
         local_resid(e,i) += rhov_dot*v - (rhov*rhou/rho)*dvdx + evisc_mu*0.5*(drhovdx+drhovdx)*dvdx;
         if (spaceDim>1)
         local_resid(e,i) += -(rhov*rhou/rho + p)*dvdy + evisc_mu*(drhovdy)*dvdy;
         if (spaceDim>2)
         local_resid(e,i) += -(rhov*rhow/rho)*dvdz + evisc_mu*0.5*(drhovdz+drhovdz)*dvdz;
         //local_resid(e,i) += -(rhov*rhow/rho)*dvdz + evisc_mu*0.5*(drhovdz+drhowdy)*dvdz;
         
         if(useSUPG) {
         //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
         }
         }
         if (var == "rhow") {
         local_resid(e,i) += rhow_dot*v - (rhow*rhou/rho)*dvdx + evisc_mu*0.5*(drhoudz+drhowdx)*dvdx;
         if (spaceDim>1)
         local_resid(e,i) += -(rhow*rhov/rho)*dvdy + evisc_mu*0.5*(drhovdz+drhowdy)*dvdy;
         if (spaceDim>2)
         local_resid(e,i) += -(rhow*rhow/rho + p)*dvdz + evisc_mu*(drhowdz);
         
         if(useSUPG) {
         //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
         }
         }
         if (var == "rhoe") {
         //local_resid(e,i) += -(rhou*rhoe/rho + rhou/rho*p)*dvdx + 0.0*evisc_mu*(drhoudx*rhou + 0.5*(drhoudy+drhovdx)*rhov + 0.5*(drhoudz+drhowdx)*rhow)*dvdx + evisc_kappa*drhoedx*dvdx;
         local_resid(e,i) += rhoe_dot*v - (rhou*rhoe/rho + rhou/rho*p)*dvdx + evisc_mu*(drhoudx*rhou + 0.5*(drhoudy+drhovdx)*rhov + 0.5*(drhoudz+drhowdx)*rhow)*dvdx + evisc_kappa*drhoedx*dvdx;
         if (spaceDim>1)
         local_resid(e,i) += -(rhov*rhoe/rho + rhov/rho*p)*dvdy + evisc_mu*(0.5*(drhoudy+drhovdx)*rhou + (drhovdy)*rhov + 0.5*(drhovdz+drhowdy)*rhow)*dvdy + evisc_kappa*drhoedy*dvdy;
         if (spaceDim>2)
         local_resid(e,i) += -(rhow*rhoe/rho + rhow/rho*p)*dvdz + evisc_mu*(0.5*(drhoudz+drhowdx)*rhou + 0.5*(drhovdz+drhowdy)*rhov + (drhowdz)*rhow)*dvdz + evisc_kappa*drhoedz*dvdz;
         
         if(useSUPG) {
         //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
         }
         }*/
        
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
    
    // assume all variables use the same basis
    int curr_basis = usebasis[curr_num];
    int numBasis = basis[curr_basis].dimension(1);
    int numSideCubPoints = ip.dimension(1);
    
    // Set the parameters
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    
    double v, dvdx, dvdy, dvdz;
    AD p, rho, drhodx, drhody, drhodz, rho_dot;
    AD rhou, drhoudx, drhoudy, drhoudz, rhou_dot;
    AD rhov, drhovdx, drhovdy, drhovdz, rhov_dot;
    AD rhow, drhowdx, drhowdy, drhowdz, rhow_dot;
    AD rhoe, drhoedx, drhoedy, drhoedz, rhoe_dot;
    
    double gamma = 1.4;
    
    for( int i=0; i<numBasis; i++ ) {
      for( int nPt=0; nPt<numSideCubPoints; nPt++ ) {
        v = basis[curr_basis](0,i,nPt);
        dvdy = basis_grad[curr_basis](0,i,nPt,0);
        
        x = ip(0,nPt,0);
        rho = local_soln(rho_num,nPt);
        rho_dot = local_soln_dot(rho_num,nPt);
        drhodx = local_solngrad(rho_num,nPt,0);
        
        rhou = local_soln(rhou_num,nPt);
        rhou_dot = local_soln_dot(rhou_num,nPt);
        drhoudx = local_solngrad(rhou_num,nPt,0);
        
        rhoe = local_soln(rhoe_num,nPt);
        rhoe_dot = local_soln_dot(rhoe_num,nPt);
        drhoedx = local_solngrad(rhoe_num,nPt,0);
        
        if (spaceDim > 1) {
          rhov = local_soln(rhov_num,nPt);
          rhov_dot = local_soln_dot(rhov_num,nPt);
          drhovdx = local_solngrad(rhov_num,nPt,0);
          y = ip(0,nPt,1);
          drhody = local_solngrad(rho_num,nPt,1);
          drhoedy = local_solngrad(rhoe_num,nPt,1);
          drhovdy = local_solngrad(rhov_num,nPt,1);
          dvdy = basis_grad[curr_basis](0,i,nPt,1);
        }
        
        if (spaceDim > 2) {
          rhow = local_soln(rhow_num,nPt);
          rhow_dot = local_soln_dot(rhow_num,nPt);
          drhowdx = local_solngrad(rhow_num,nPt,0);
          drhowdy = local_solngrad(rhow_num,nPt,1);
          drhowdz = local_solngrad(rhow_num,nPt,2);
          z = ip(0,nPt,2);
          drhodz = local_solngrad(rho_num,nPt,2);
          drhoudz = local_solngrad(rhou_num,nPt,2);
          drhovdz = local_solngrad(rhov_num,nPt,2);
          drhoedz = local_solngrad(rhoe_num,nPt,2);
          dvdz = basis_grad[curr_basis](0,i,nPt,2);
        }
        
        // multiply and add into local residual
        /*
        p = (gamma-1.0)*(rhoe - 0.5/rho*(rhou*rhou + rhov*rhov + rhow*rhow));
        if (var == "rho") {
          local_resid(e,i) += rhou*normals(e,nPt,0);
          if (spaceDim>1)
            local_resid(e,i) += rhov*normals(e,nPt,1);
          if (spaceDim>2)
            local_resid(e,i) += rhow*normals(e,nPt,2);
          
          if(useSUPG) {
            //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
          }
        }
        if (var == "rhou") {
          local_resid(e,i) += (rhou*rhou/rho + p)*normals(e,nPt,0);
          if (spaceDim>1)
            local_resid(e,i) += (rhou*rhov/rho)*normals(e,nPt,1);
          if (spaceDim>2)
            local_resid(e,i) += (rhou*rhow/rho)*normals(e,nPt,2);
          
          if(useSUPG) {
            //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
          }
        }
        if (var == "rhov") {
          local_resid(e,i) += (rhov*rhou/rho)*normals(e,nPt,0);
          if (spaceDim>1)
            local_resid(e,i) += (rhov*rhou/rho + p)*normals(e,nPt,1);
          if (spaceDim>2)
            local_resid(e,i) += (rhov*rhow/rho)*normals(e,nPt,2);
          
          if(useSUPG) {
            //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
          }
        }
        if (var == "rhow") {
          local_resid(e,i) += (rhow*rhou/rho)*normals(e,nPt,0);
          if (spaceDim>1)
            local_resid(e,i) += (rhow*rhov/rho)*normals(e,nPt,1);
          if (spaceDim>2)
            local_resid(e,i) += (rhow*rhow/rho + p)*normals(e,nPt,2);
          
          if(useSUPG) {
            //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
          }
        }
        if (var == "rhoe") {
          local_resid(e,i) += (rhou*rhoe/rho + rhou/rho*p)*normals(e,nPt,0);
          if (spaceDim>1)
            local_resid(e,i) += (rhov*rhoe/rho + rhov/rho*p)*normals(e,nPt,1);
          if (spaceDim>2)
            local_resid(e,i) += (rhow*rhoe/rho + rhow/rho*p)*normals(e,nPt,2);
          
          if(useSUPG) {
            //tau = this->computeTau(visc_FAD, ux, uy, uz, h[e]);
          }
        }
        */
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
                   const FCAD & local_soln_dot, const FCAD local_param, , const FCAD local_aux, 
                   const FC & normals, FCAD & flux) {

  }

  // ========================================================================================
  // ========================================================================================
  
  AD getDirichletValue(const string & var, const double & x, const double & y, const double & z, 
                       const double & t, const string & gside, const std::vector<std::vector<AD > > currparams,  
                       const bool & useadjoint) const {
    
    AD val = 0.0;
    if (test == "Sod") {
      if (var == "rho") {
        if (x<=0.5)
          val = 1.0;
        else
          val = 0.125;
      }
      if (var == "rhou")
        val = 0.0;
      if (var == "rhov")
        val = 0.0;
      if (var == "rhow")
        val = 0.0;
      if (var == "rhoe")
        if (x<0.5)
          val = 2.5;
        else
          val = 0.25;
    }
    
    if (test == "Noh") {
      if (var == "rho") {
        val = 1.0;
      }
      if (var == "rhou")
        val = -x/sqrt(1.0e-10+x*x+y*y);
      if (var == "rhov")
        val = -y/sqrt(1.0e-10+x*x+y*y);
      if (var == "rhow")
        val = 0.0;
      if (var == "rhoe")
        val = 1.0;
    }
    
    if (useadjoint) 
      val = 0.0;
    
    return val;
    
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  double getInitialValue(const string & var, const double & x, const double & y, const double & z, 
                         const bool & useadjoint) const {
    double val = 0.0;
    if (test == "Sod") {
      if (var == "rho") {
        if (x<=0.5)
          val = 1.0;
        else
          val = 0.125;
      }
      if (var == "rhou")
        val = 0.0;
      if (var == "rhov")
        val = 0.0;
      if (var == "rhow")
        val = 0.0;
      if (var == "rhoe")
        if (x<0.5)
          val = 2.5;
        else
          val = 0.25;
    }
    if (test == "Noh") {
      if (var == "rho") {
        val = 1.0;
      }
      if (var == "rhou")
        val = -x/sqrt(1.0e-10+x*x+y*y);
      if (var == "rhov")
        val = -y/sqrt(1.0e-10+x*x+y*y);
      if (var == "rhow")
        val = 0.0;
      if (var == "rhoe")
        val = 1.0;
    }
    
    if (useadjoint) 
      val = 0.0;
    
    return val;
  }
  
  // ========================================================================================
  // error calculation
  // ========================================================================================
  
  double trueSolution(const string & var, const double & x, const double & y, const double & z, 
                      const double & time) const {
    double c = 0.0;
    return c;
  }
  
  // ========================================================================================
  // response calculation
  // ========================================================================================
  
  FCAD response(const FCAD & local_soln, 
                               const FCAD & local_soln_grad,
                               const DRV & ip, const double & time, 
                               const std::vector<std::vector<AD > > paramvals) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    FCAD resp(numCC,spaceDim+2,numip);
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        resp(i,0,j) = local_soln(i,rho_num,j);
        resp(i,0,j) = local_soln(i,rhou_num,j);
        if (spaceDim > 1)
          resp(i,1,j) = local_soln(i,rhov_num,j);
        if (spaceDim > 2)
          resp(i,2,j) = local_soln(i,rhow_num,j);
        
        resp(i,spaceDim+1,j) = local_soln(i,rhoe_num,j);
      }
    }
    
    return resp;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD target(const FC & ip, const double & time, 
                             const std::vector<std::vector<AD > > paramvals) const {
    int numCC = ip.dimension(0);
    int numip = ip.dimension(1);
    FCAD targ(numCC,spaceDim+2,numip);
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numip; j++) {
        targ(i,0,j) = 0.0;
        targ(i,1,j) = 0.0;
        if (spaceDim > 1)
          targ(i,2,j) = 0.0;
        if (spaceDim > 2)
          targ(i,3,j) = 0.0;
        
        targ(i,spaceDim+1,j) = 0.0;
      }
    }
    
    return targ;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD auxiliaryVars(const FCAD soln) const {
    int numCC = soln.dimension(0);
    int numbasis = soln.dimension(2);
    FCAD auxvars(numCC,3,numbasis);
    
    double gamma = 1.4;
    AD S, ux, uy, uz, rho, p;
    for (int i=0; i<numCC; i++) {
      for (int j=0; j<numbasis; j++) {
        rho = soln(i,rho_num,j);
        ux = soln(i,rhou_num,j) / rho;
        if (spaceDim > 1)
          uy = soln(i,rhov_num,j) / rho;
        if (spaceDim > 2)
          uz = soln(i,rhow_num,j) / rho;
        p = (gamma-1.0)*(soln(i,rhoe_num,j) - 1.0/2.0*rho*(ux*ux+uy*uy+uz*uz));
        S = rho / (gamma-1.0)*log(p/pow(rho,gamma));
        auxvars(i,0,j) = ux*S;
        if (spaceDim > 1)
          auxvars(i,1,j) = uy*S;
        if (spaceDim > 2)
          auxvars(i,2,j) = uz*S;
        
      }
    }
    
    return auxvars;
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "rho")
        rho_num = i;
      if (varlist[i] == "rhou")
        rhou_num = i;
      if (varlist[i] == "rhov")
        rhov_num = i;
      if (varlist[i] == "rhow")
        rhow_num = i;
      if (varlist[i] == "rhoe")
        rhoe_num = i;
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
  
  template<class T1, class T2>
  AD computeTau(const T2 & localdiff, const T1 & xvl, const T1 & yvl, const T1 & zvl, 
                const double & h) const {
    
    double C1 = 4.0;
    double C2 = 2.0;
    
    T1 nvel;
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
  // return the value of the entropy viscosity 
  // ========================================================================================
  
  template<class T>  
  T entropyViscosity(const T & rho, const T & rhox, const T & rhoy, const T & rhoz, const T & rhot,
                     const T & rhou, const T & rhoux, const T & rhouy, const T & rhouz, const T & rhout,
                     const T & rhov, const T & rhovx, const T & rhovy, const T & rhovz, const T & rhovt,
                     const T & rhow, const T & rhowx, const T & rhowy, const T & rhowz, const T & rhowt,
                     const T & rhoe, const T & rhoex, const T & rhoey, const T & rhoez, const T & rhoet,
                     const double & h, std::vector<std::vector<AD > > params) const {
    
    double C1 = 1.0;
    double C2 = 0.25;
    double maxevisc = 0.4;
    double gamma = 1.4;
    
    T S, p, entres;
    T dSdrho, dSdp, dSdt, dSdx, dSdy, dSdz;
    T dpdrho, dpdrhou, dpdrhov, dpdrhow, dpdrhoe, dpdt, dpdx, dpdy, dpdz;
    T u, v, w, dudx, dvdy, dwdz;
    
    p = (gamma-1.0)*(rhoe - 0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/rho);
    
    dpdrho = (gamma-1.0)*(0.5*(rhou*rhou + rhov*rhov + rhow*rhow)/(rho*rho));
    dpdrhou = (gamma-1.0)*(-rhou/rho); 
    dpdrhov = (gamma-1.0)*(-rhov/rho); 
    dpdrhow = (gamma-1.0)*(-rhow/rho); 
    dpdrhoe = (gamma-1.0); 
    
    S = rho/(gamma - 1.0)*(log(p)-gamma*log(rho));
    dSdrho = rho/(gamma-1.0)*(1.0/rho);
    dSdp = 1.0/(gamma-1.0)*(log(p)-gamma*log(rho)) - gamma/(gamma-1.0);
    
    dpdt = dpdrho*rhot + dpdrhou*rhout + dpdrhov*rhovt + dpdrhow*rhowt + dpdrhoe*rhoet;
    dpdx = dpdrho*rhox + dpdrhou*rhoux + dpdrhov*rhovx + dpdrhow*rhowx + dpdrhoe*rhoex;
    dpdy = dpdrho*rhoy + dpdrhou*rhouy + dpdrhov*rhovy + dpdrhow*rhowy + dpdrhoe*rhoey;
    dpdz = dpdrho*rhoz + dpdrhou*rhouz + dpdrhov*rhovz + dpdrhow*rhowz + dpdrhoe*rhoez;
    
    dSdt = dSdrho*rhot + dSdp*dpdt;
    dSdx = dSdrho*rhox + dSdp*dpdx;
    dSdy = dSdrho*rhoy + dSdp*dpdy;
    dSdz = dSdrho*rhoz + dSdp*dpdz;
    
    
    u = rhou/rho;
    v = rhov/rho;
    w = rhow/rho;
    
    dudx = (rhoux*rho - rhox*rhou)/(rho*rho);
    dvdy = (rhovy*rho - rhoy*rhoy)/(rho*rho);
    dwdz = (rhowz*rho - rhoz*rhow)/(rho*rho);
    
    entres = dSdt + u*dSdx + v*dSdy + w*dSdz + S*(dudx+dvdy+dwdz);
    
    T evisc = C1*h*h*sqrt(1.0e-12 + entres*entres)/C2;
    
    T c = sqrt(gamma-1.0)*((rhoe+p)/rho - 0.5*(u*u+v*v+w*w));
    T ubevisc = 0.5*(sqrt(1.0e-12+u*u+v*v+w*w) + c)*h;
    
    if (evisc.val() > ubevisc.val()) {
      cout << "evisc exceeds max value: " << evisc.val() << endl;
      evisc = ubevisc;
    }
    
    return evisc;
    
  }
  
  // ========================================================================================
  /* return the source term (to be multiplied by test_function) */
  // ========================================================================================
  
  double SourceTerm(const string & var, const double & x, const double & y, const double & z) const {
    double val = 0.0;
    
    return val;
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
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  FCAD scalarRespFunc(const FCAD & integralResponses, 
                                    const bool & justDeriv) const {return integralResponses;}
  bool useScalarRespFunc() const {return false;}
  
private:
  
  Teuchos::RCP<userDefined> udfunc;
  
  int spaceDim, numElem, numParams, numResponses, numSteps;
  vector<string> varlist;
  int rho_num, rhou_num, rhov_num, rhow_num, rhoe_num;
  //double density, viscosity;
  bool isTD, useSUPG, usePSPG;
  double dt;
  
  string test;
  
  std::string analysis_type; //to know when parameter is a sample that needs to be transformed
  FC entropy_viscosity;
};

#endif
