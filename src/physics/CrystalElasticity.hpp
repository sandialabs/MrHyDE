/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef CrystalElasticity_H
#define CrystalElasticity_H

#include "preferences.hpp"

namespace MrHyDE {
  
  class CrystalElastic {
  public:
    
    CrystalElastic() {} ;
    ~CrystalElastic() {};
    
    CrystalElastic( Teuchos::RCP<Teuchos::ParameterList> & settings , const int & numElem_)
    {
      
      dimension = settings->sublist("Mesh").get<int>("dim",3);
      numElem = numElem_;//settings->sublist("Solver").get<int>("Workset size",1);
      
      Teuchos::ParameterList esettings = settings->sublist("Physics").sublist("Elastic Coefficients");
      e_ref = settings->sublist("Physics").get<ScalarT>("T_ambient",0.0);
      alpha_T = settings->sublist("Physics").get<ScalarT>("alpha_T",1.0e-6);
      
      ScalarT E = esettings.get<ScalarT>("E",1.0);
      ScalarT nu = esettings.get<ScalarT>("nu",0.4);
      
      lambda = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
      mu = E/(2.0*(1.0+nu));
      
      // Gas constant: TMW: Need to make this a parameter
      // ScalarT R_ = esettings.get<ScalarT>("R",0.0);
      
      // Elastic tensor in lattice frame
      C = Kokkos::View<ScalarT*****,ContLayout,AssemblyDevice>("CE-C",numElem,3,3,3,3);
      
      // Elastic tensor in rotated frame
      Cr = Kokkos::View<ScalarT*****,ContLayout,AssemblyDevice>("CE-Cr",numElem,3,3,3,3);
      
      // default to cubic symmetry
      c11_ = esettings.get<ScalarT>("C11",2.0*mu+lambda);
      c22_ = esettings.get<ScalarT>("C22",c11_);
      c33_ = esettings.get<ScalarT>("C33",c11_);
      c44_ = esettings.get<ScalarT>("C44",2.0*mu);
      c55_ = esettings.get<ScalarT>("C55",c44_);
      c66_ = esettings.get<ScalarT>("C66",c44_);
      c12_ = esettings.get<ScalarT>("C12",lambda);
      c13_ = esettings.get<ScalarT>("C13",c12_);
      c23_ = esettings.get<ScalarT>("C23",c12_);
      c15_ = esettings.get<ScalarT>("C15",0.0);
      c25_ = esettings.get<ScalarT>("C25",0.0);
      c35_ = esettings.get<ScalarT>("C35",0.0);
      c46_ = esettings.get<ScalarT>("C46",0.0);
      
      auto C_host = Kokkos::create_mirror_view(C);
      // fill tensor
      for (int e=0; e<numElem; e++) {
        C_host(e,0,0,0,0) = c11_;
        C_host(e,1,1,1,1) = c22_;
        C_host(e,2,2,2,2) = c33_;
        C_host(e,0,0,1,1) = c12_;
        C_host(e,1,1,0,0) = c12_;
        C_host(e,0,0,2,2) = c13_;
        C_host(e,2,2,0,0) = c13_;
        C_host(e,1,1,2,2) = c23_;
        C_host(e,2,2,1,1) = c23_;
        C_host(e,0,1,0,1) = c66_;
        C_host(e,1,0,1,0) = c66_;
        C_host(e,0,1,1,0) = c66_;
        C_host(e,1,0,0,1) = c66_;
        C_host(e,2,0,2,0) = c55_;
        C_host(e,0,2,0,2) = c55_;
        C_host(e,2,0,0,2) = c55_;
        C_host(e,0,2,0,0) = c55_;
        C_host(e,2,1,2,1) = c44_;
        C_host(e,1,2,1,2) = c44_;
        C_host(e,1,2,2,1) = c44_;
        C_host(e,2,1,1,2) = c44_;
        C_host(e,0,0,0,2) = c15_;
        C_host(e,0,0,2,0) = c15_;
        C_host(e,0,2,0,0) = c15_;
        C_host(e,2,0,0,0) = c15_;
        C_host(e,1,1,0,2) = c25_;
        C_host(e,1,1,2,0) = c25_;
        C_host(e,0,2,1,1) = c25_;
        C_host(e,2,0,1,1) = c25_;
        C_host(e,2,2,0,2) = c35_;
        C_host(e,2,2,2,0) = c35_;
        C_host(e,0,2,2,2) = c35_;
        C_host(e,2,0,2,2) = c35_;
        C_host(e,1,2,0,1) = c46_;
        C_host(e,1,2,1,0) = c46_;
        C_host(e,2,1,0,1) = c46_;
        C_host(e,2,1,1,0) = c46_;
        C_host(e,0,1,1,2) = c46_;
        C_host(e,1,0,1,2) = c46_;
        C_host(e,0,1,2,1) = c46_;
        C_host(e,1,0,2,1) = c46_;
      }
      Kokkos::deep_copy(C,C_host);
    }
    
    //----------------------------------------------------------------------------
    
    void updateParams(Teuchos::RCP<workset> & wkset) {
      
      bool foundlam = false;
      vector<AD> lvals = wkset->getParam("lambda", foundlam);
      if (foundlam) {
        lambda = lvals[0].val();
      }
      
      bool foundmu = false;
      vector<AD> muvals = wkset->getParam("mu", foundmu);
      if (foundmu) {
        mu = muvals[0].val();
      }
      
      if (!foundlam || !foundmu) {
        ScalarT E = 0.0;
        bool foundym = false;
        vector<AD> ymvals = wkset->getParam("youngs_mod", foundym);
        if (foundym) {
          E = ymvals[0].val();
        }
        
        ScalarT nu = 0.0;
        bool foundpr = false;
        vector<AD> prvals = wkset->getParam("poisson_ratio", foundpr);
        if (foundpr) {
          nu = prvals[0].val();
        }
        
        if (foundym && foundpr) {
          lambda = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
          mu = E/(2.0*(1.0+nu));
        }
      }
      
      ScalarT c11 = 2.0*mu+lambda;
      ScalarT c12 = lambda;
      ScalarT c44 = 2.0*mu;
      
      
      /*
       ScalarT c11 = 0.0;
       bool foundc11 = false;
       vector<AD> c11vals = wkset->getParam("C11", foundc11);
       if (foundc11) {
       c11 = c11vals[0].val();
       }
       
       ScalarT c12 = 0.0;
       bool foundc12 = false;
       vector<AD> c12vals = wkset->getParam("C12", foundc12);
       if (foundc12) {
       c12 = c12vals[0].val();
       }
       
       ScalarT c44 = 0.0;
       bool foundc44 = false;
       vector<AD> c44vals = wkset->getParam("C44", foundc44);
       if (foundc44) {
       c44 = c44vals[0].val();
       }
       */
      
      // default to cubic symmetry
      c11_ = c11;
      c22_ = c11_;
      c33_ = c11_;
      c44_ = c44;
      c55_ = c44_;
      c66_ = c44_;
      c12_ = c12;
      c13_ = c12_;
      c23_ = c12_;
      c15_ = 0.0;
      c25_ = 0.0;
      c35_ = 0.0;
      c46_ = 0.0;
      
      auto C_host = Kokkos::create_mirror_view(C);
      // fill tensor
      for (int e=0; e<numElem; e++) {
        C_host(e,0,0,0,0) = c11_;
        C_host(e,1,1,1,1) = c22_;
        C_host(e,2,2,2,2) = c33_;
        C_host(e,0,0,1,1) = c12_;
        C_host(e,1,1,0,0) = c12_;
        C_host(e,0,0,2,2) = c13_;
        C_host(e,2,2,0,0) = c13_;
        C_host(e,1,1,2,2) = c23_;
        C_host(e,2,2,1,1) = c23_;
        C_host(e,0,1,0,1) = c66_;
        C_host(e,1,0,1,0) = c66_;
        C_host(e,0,1,1,0) = c66_;
        C_host(e,1,0,0,1) = c66_;
        C_host(e,2,0,2,0) = c55_;
        C_host(e,0,2,0,2) = c55_;
        C_host(e,2,0,0,2) = c55_;
        C_host(e,0,2,0,0) = c55_;
        C_host(e,2,1,2,1) = c44_;
        C_host(e,1,2,1,2) = c44_;
        C_host(e,1,2,2,1) = c44_;
        C_host(e,2,1,1,2) = c44_;
        C_host(e,0,0,0,2) = c15_;
        C_host(e,0,0,2,0) = c15_;
        C_host(e,0,2,0,0) = c15_;
        C_host(e,2,0,0,0) = c15_;
        C_host(e,1,1,0,2) = c25_;
        C_host(e,1,1,2,0) = c25_;
        C_host(e,0,2,1,1) = c25_;
        C_host(e,2,0,1,1) = c25_;
        C_host(e,2,2,0,2) = c35_;
        C_host(e,2,2,2,0) = c35_;
        C_host(e,0,2,2,2) = c35_;
        C_host(e,2,0,2,2) = c35_;
        C_host(e,1,2,0,1) = c46_;
        C_host(e,1,2,1,0) = c46_;
        C_host(e,2,1,0,1) = c46_;
        C_host(e,2,1,1,0) = c46_;
        C_host(e,0,1,1,2) = c46_;
        C_host(e,1,0,1,2) = c46_;
        C_host(e,0,1,2,1) = c46_;
        C_host(e,1,0,2,1) = c46_;
      }
      Kokkos::deep_copy(C,C_host);
    }
    
    //----------------------------------------------------------------------------
    
    View_AD4 computeStress(Teuchos::RCP<workset> & wkset, vector<int> & indices, const bool & onside)
    {
      
      int dxnum = indices[0];
      int dynum = 0;
      if (dimension > 1) {
        dynum = indices[1];
      }
      int dznum = 0;
      if (dimension > 2) {
        dznum = indices[2];
      }
      
      int e_num = indices[3];
      bool have_energy = false;
      if (e_num >= 0)
        have_energy = true;
      
      int numip = wkset->numip;
      if (onside) {
        numip = wkset->numsideip;
      }
      
      this->computeRotation(wkset);
      
      View_AD4 F("CE-F",numElem,numip,dimension,dimension);
      View_AD4 sol_grad = wkset->local_soln_grad;
      View_AD4 sol = wkset->local_soln;
      if (onside) {
        sol_grad = wkset->local_soln_grad_side;
        sol = wkset->local_soln_side;
      }
      if (dimension == 1) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            F(e,k,0,0) = sol_grad(e,dxnum,k,0);
          }
        }
      }
      else if (dimension == 2) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            F(e,k,0,0) = sol_grad(e,dxnum,k,0);
            F(e,k,0,1) = sol_grad(e,dxnum,k,1);
            F(e,k,1,0) = sol_grad(e,dynum,k,0);
            F(e,k,1,1) = sol_grad(e,dynum,k,1);
          }
        }
      }
      else if (dimension == 3) {
        for (int e=0; e<numElem; e++) {
          for (int k=0; k<numip; k++) {
            F(e,k,0,0) = sol_grad(e,dxnum,k,0);
            F(e,k,0,1) = sol_grad(e,dxnum,k,1);
            F(e,k,0,2) = sol_grad(e,dxnum,k,2);
            F(e,k,1,0) = sol_grad(e,dynum,k,0);
            F(e,k,1,1) = sol_grad(e,dynum,k,1);
            F(e,k,1,2) = sol_grad(e,dynum,k,2);
            F(e,k,2,0) = sol_grad(e,dznum,k,0);
            F(e,k,2,1) = sol_grad(e,dznum,k,1);
            F(e,k,2,2) = sol_grad(e,dznum,k,2);
          }
        }
      }
      
      View_AD4 E("CE-E",numElem,numip,dimension,dimension);
      
      for (int e=0; e<numElem; e++) {
        for (int k=0; k<numip; k++) {
          for (int i=0; i<dimension; i++) {
            for (int j=0; j<dimension; j++) {
              E(e,k,i,j) = 0.5*F(e,k,i,j) + 0.5*F(e,k,j,i);
            }
          }
        }
      }
      
      View_AD4 stress("CE-stress",numElem,numip,dimension,dimension);
      // compute S = Cr*E
      
      for (int e=0; e<numElem; e++) {
        for (int q=0; q<numip; q++) {
          for ( int i = 0; i < dimension; ++i ) {
            for ( int j = 0; j < dimension; ++j ) {
              stress(e,q,i,j) = 0.0;
              for ( int k = 0; k < dimension; ++k ) {
                for ( int l = 0; l < dimension; ++l ) {
                  stress(e,q,i,j) = stress(e,q,i,j) + Cr(e,i,j,k,l)*E(e,q,k,l);
                } // end l
              } // end k
            } // end j
            
          } // end i
        }
        if (have_energy) {
          for (int q=0; q<numip; q++) {
            for ( int i = 0; i < dimension; ++i ) {
              stress(e,q,i,i) += -alpha_T*(3*lambda+2*mu)*(sol(e,e_num,q,0)-e_ref);
            }
          }
        }
      }
      
      return stress;
    }
    //----------------------------------------------------------------------------
    
    void computeRotation(Teuchos::RCP<workset> & wkset) {
      Kokkos::View<ScalarT**,AssemblyDevice> rl("CE-rl",3,3);
      
      for (int e=0; e<numElem; e++) {
        
        if (wkset->have_rotation_phi) {
          // Read Bunge Angle in degrees
          ScalarT phi1d = wkset->rotation_phi(e,0);
          ScalarT Phid = wkset->rotation_phi(e,1);
          ScalarT phi2d = wkset->rotation_phi(e,2);
          
          // From degree to rad
          ScalarT degtorad = atan(1.0)/45.0;
          ScalarT phi1_ = phi1d*degtorad; // [-180,180]
          ScalarT Phi_ = Phid*degtorad; // [0,180]
          ScalarT phi2_ = phi2d*degtorad; // [-180,180]
          
          
          // Initialize rotation matrix
          
          // Compute rotation matrix
          rl(0,0) = cos(phi1_)*cos(phi2_) - sin(phi1_)*cos(Phi_)*sin(phi2_);
          rl(1,0) = sin(phi1_)*cos(phi2_) + cos(phi1_)*cos(Phi_)*sin(phi2_);
          rl(2,0) = sin(Phi_)*sin(phi2_);
          rl(0,1) = -cos(phi1_)*sin(phi2_) - sin(phi1_)*cos(Phi_)*cos(phi2_);
          rl(1,1) = -sin(phi1_)*sin(phi2_) + cos(phi1_)*cos(Phi_)*cos(phi2_);
          rl(2,1) = sin(Phi_)*cos(phi2_);
          rl(0,2) = sin(phi1_)*sin(Phi_);
          rl(1,2) = -cos(phi1_)*sin(Phi_);
          rl(2,2) = cos(Phi_);
          
        }
        else if (wkset->have_rotation) {
          for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
              rl(i,j) = wkset->rotation(e,i,j);
            }
          }
        }
        else {
          
          rl(0,0) = 1.0;
          rl(1,1) = 1.0;
          rl(2,2) = 1.0;
        }
        
        
        // Form rotate elasticity tensor
        for ( int i = 0; i < dimension; i++ ){
          for ( int j = 0; j < dimension; j++ ){
            for ( int k = 0; k < dimension; k++ ){
              for ( int l = 0; l < dimension; l++ ){
                Cr(e,i,j,k,l) = 0.0;
                for ( int i1 = 0; i1 < dimension; i1++ ){
                  for ( int j1 = 0; j1 < dimension; j1++ ){
                    for ( int k1 = 0; k1 < dimension; k1++ ){
                      for ( int l1 = 0; l1 < dimension; l1++ ){
                        Cr(e,i,j,k,l) += rl(i,i1)*rl(j,j1)*rl(k,k1)*rl(l,l1)*C(e,i1,j1,k1,l1);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      
    }
    
    // Public Data
    int dimension, numElem;
    ScalarT c11_,c22_,c33_,c44_,c55_,c66_,c12_,c13_,c23_,c15_,c25_,c35_,c46_;
    Kokkos::View<ScalarT*****,ContLayout,AssemblyDevice> C; // unrotated stiffness tensor
    Kokkos::View<ScalarT*****,ContLayout,AssemblyDevice> Cr; // rotated stiffness tensor
    ScalarT lambda, mu, e_ref, alpha_T;
  };
  
}

#endif 
