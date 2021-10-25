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

#include "CrystalElasticity.hpp"

using namespace MrHyDE;

CrystalElastic::CrystalElastic(Teuchos::ParameterList & settings,
                               const int & dimension_)
{
  
  dimension = dimension_;
   
  Teuchos::ParameterList cesettings = settings.sublist("Crystal elastic parameters");
  e_ref = cesettings.get<ScalarT>("T_ambient",0.0);
  alpha_T = cesettings.get<ScalarT>("alpha_T",1.0e-6);
  
  allow_rotations = cesettings.get<bool>("allow rotations",true);
  
  ScalarT E = cesettings.get<ScalarT>("E",1.0);
  ScalarT nu = cesettings.get<ScalarT>("nu",0.4);
  
  lambda = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
  mu = E/(2.0*(1.0+nu));
  
  // Gas constant: TMW: Need to make this a parameter
  // ScalarT R_ = esettings.get<ScalarT>("R",0.0);
  
  // Elastic tensor in lattice frame
  C = View_Sc4("CE-C",3,3,3,3);
  
  // default to cubic symmetry
  c11_ = cesettings.get<ScalarT>("C11",2.0*mu+lambda);
  c22_ = cesettings.get<ScalarT>("C22",c11_);
  c33_ = cesettings.get<ScalarT>("C33",c11_);
  c44_ = cesettings.get<ScalarT>("C44",2.0*mu);
  c55_ = cesettings.get<ScalarT>("C55",c44_);
  c66_ = cesettings.get<ScalarT>("C66",c44_);
  c12_ = cesettings.get<ScalarT>("C12",lambda);
  c13_ = cesettings.get<ScalarT>("C13",c12_);
  c23_ = cesettings.get<ScalarT>("C23",c12_);
  c15_ = cesettings.get<ScalarT>("C15",0.0);
  c25_ = cesettings.get<ScalarT>("C25",0.0);
  c35_ = cesettings.get<ScalarT>("C35",0.0);
  c46_ = cesettings.get<ScalarT>("C46",0.0);
  
  // update, just in case they changed
  lambda = c12_;
  mu = c44_/2.0;
  
  this->computeLatticeTensor();
  
}

//=====================================================

void CrystalElastic::computeLatticeTensor() {
  
  auto C_host = create_mirror_view(C);
  
  // fill tensor
  C_host(0,0,0,0) = c11_;
  C_host(1,1,1,1) = c22_;
  C_host(2,2,2,2) = c33_;
  C_host(0,0,1,1) = c12_;
  C_host(1,1,0,0) = c12_;
  C_host(0,0,2,2) = c13_;
  C_host(2,2,0,0) = c13_;
  C_host(1,1,2,2) = c23_;
  C_host(2,2,1,1) = c23_;
  C_host(0,1,0,1) = c66_;
  C_host(1,0,1,0) = c66_;
  C_host(0,1,1,0) = c66_;
  C_host(1,0,0,1) = c66_;
  C_host(2,0,2,0) = c55_;
  C_host(0,2,0,2) = c55_;
  C_host(2,0,0,2) = c55_;
  C_host(0,2,0,0) = c55_;
  C_host(2,1,2,1) = c44_;
  C_host(1,2,1,2) = c44_;
  C_host(1,2,2,1) = c44_;
  C_host(2,1,1,2) = c44_;
  C_host(0,0,0,2) = c15_;
  C_host(0,0,2,0) = c15_;
  C_host(0,2,0,0) = c15_;
  C_host(2,0,0,0) = c15_;
  C_host(1,1,0,2) = c25_;
  C_host(1,1,2,0) = c25_;
  C_host(0,2,1,1) = c25_;
  C_host(2,0,1,1) = c25_;
  C_host(2,2,0,2) = c35_;
  C_host(2,2,2,0) = c35_;
  C_host(0,2,2,2) = c35_;
  C_host(2,0,2,2) = c35_;
  C_host(1,2,0,1) = c46_;
  C_host(1,2,1,0) = c46_;
  C_host(2,1,0,1) = c46_;
  C_host(2,1,1,0) = c46_;
  C_host(0,1,1,2) = c46_;
  C_host(1,0,1,2) = c46_;
  C_host(0,1,2,1) = c46_;
  C_host(1,0,2,1) = c46_;
  
  deep_copy(C,C_host);
  
}

//----------------------------------------------------------------------------

void CrystalElastic::updateParams(Teuchos::RCP<workset> & wkset) {
  
  ScalarT c11 = c11_;
  ScalarT c12 = c12_;
  ScalarT c44 = c44_;
  
  bool foundlam = false;
  vector<AD> lvals = wkset->getParam("lambda", foundlam);
  if (foundlam) {
#ifndef MrHyDE_NO_AD
    lambda = lvals[0].val();
#else
    lambda = lvals[0];
#endif
  }
  
  bool foundmu = false;
  vector<AD> muvals = wkset->getParam("mu", foundmu);
  if (foundmu) {
#ifndef MrHyDE_NO_AD
    mu = muvals[0].val();
#else
    mu = muvals[0];
#endif
  }
  
  if (!foundlam || !foundmu) {
    ScalarT E = 0.0;
    bool foundym = false;
    vector<AD> ymvals = wkset->getParam("youngs_mod", foundym);
    if (foundym) {
#ifndef MrHyDE_NO_AD
      E = ymvals[0].val();
#else
      E = ymvals[0];
#endif
    }
    
    ScalarT nu = 0.0;
    bool foundpr = false;
    vector<AD> prvals = wkset->getParam("poisson_ratio", foundpr);
    if (foundpr) {
#ifndef MrHyDE_NO_AD
      nu = prvals[0].val();
#else
      nu = prvals[0];
#endif
    }
    
    if (foundym && foundpr) {
      lambda = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
      mu = E/(2.0*(1.0+nu));
      c11 = 2.0*mu+lambda;
      c12 = lambda;
      c44 = 2.0*mu;
    }
  }
  else {
    c11 = 2.0*mu+lambda;
    c12 = lambda;
    c44 = 2.0*mu;
  }
  
  
  
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
  
  this->computeLatticeTensor();
  
}

//----------------------------------------------------------------------------

void CrystalElastic::computeStress(Teuchos::RCP<workset> & wkset, vector<int> & indices,
                                   const bool & onside, View_AD4 stress)
{
  
  Teuchos::TimeMonitor stimer(*computeStressTimer);
  
  //Kokkos::Timer timer;

  //timer.reset();
  int e_num = indices[3];
  bool have_energy = false;
  if (e_num >= 0)
    have_energy = true;
  
  int numip = wkset->numip;
  if (onside) {
    numip = wkset->numsideip;
  }
  int dimension_ = dimension;
  
  this->computeRotatedTensor(wkset);
  
  //double time1 = timer.seconds();
  //printf("time 1:   %e \n", time1);
  //timer.reset();
  
  View_AD4 E("CE-E",wkset->numElem,numip,dimension,dimension);
  
  //double time2 = timer.seconds();
  //printf("time 2:   %e \n", time2);
  //timer.reset();
  
  string postfix = "";
  if (onside) {
    postfix = " side";
  }
  if (dimension == 1) {
    auto dx_x = wkset->getData("grad(dx)[x]"+postfix);
    parallel_for("CE stress 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const size_type e ) {
      for (size_type k=0; k<dx_x.extent(1); k++) {
        E(e,k,0,0) = dx_x(e,k);
      }
    });
  }
  else if (dimension == 2) {
    auto dx_x = wkset->getData("grad(dx)[x]"+postfix);
    auto dx_y = wkset->getData("grad(dx)[y]"+postfix);
    auto dy_x = wkset->getData("grad(dy)[x]"+postfix);
    auto dy_y = wkset->getData("grad(dy)[y]"+postfix);
    parallel_for("CE stress 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const size_type e ) {
      for (size_type k=0; k<dx_x.extent(1); k++) {
        E(e,k,0,0) = dx_x(e,k);
        E(e,k,0,1) = 0.5*dx_y(e,k) + 0.5*dy_x(e,k);
        E(e,k,1,0) = 0.5*dy_x(e,k) + 0.5*dx_y(e,k);
        E(e,k,1,1) = dy_y(e,k);
      }
    });
  }
  else if (dimension == 3) {
    auto dx_x = wkset->getData("grad(dx)[x]"+postfix);
    auto dx_y = wkset->getData("grad(dx)[y]"+postfix);
    auto dx_z = wkset->getData("grad(dx)[z]"+postfix);
    auto dy_x = wkset->getData("grad(dy)[x]"+postfix);
    auto dy_y = wkset->getData("grad(dy)[y]"+postfix);
    auto dy_z = wkset->getData("grad(dy)[z]"+postfix);
    auto dz_x = wkset->getData("grad(dz)[x]"+postfix);
    auto dz_y = wkset->getData("grad(dz)[y]"+postfix);
    auto dz_z = wkset->getData("grad(dz)[z]"+postfix);
    parallel_for("CE stress 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const size_type e ) {
      for (size_type k=0; k<dx_x.extent(1); k++) {
        E(e,k,0,0) = dx_x(e,k);
        E(e,k,0,1) = 0.5*dx_y(e,k) + 0.5*dy_x(e,k);
        E(e,k,0,2) = 0.5*dx_z(e,k) + 0.5*dz_x(e,k);
        E(e,k,1,0) = 0.5*dy_x(e,k) + 0.5*dx_y(e,k);
        E(e,k,1,1) = dy_y(e,k);
        E(e,k,1,2) = 0.5*dy_z(e,k) + 0.5*dz_y(e,k);
        E(e,k,2,0) = 0.5*dz_x(e,k) + 0.5*dx_z(e,k);
        E(e,k,2,1) = 0.5*dz_y(e,k) + 0.5*dy_z(e,k);
        E(e,k,2,2) = dz_z(e,k);
      }
    });
  }
  
  //double time3 = timer.seconds();
  //printf("time 3:   %e \n", time3);
  //timer.reset();
  
  // compute S = Cr*E
  
  parallel_for("CE stress 3D",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const size_type e ) {
    for (int q=0; q<numip; q++) {
      for ( int i = 0; i < dimension_; ++i ) {
        for ( int j = 0; j < dimension_; ++j ) {
          stress(e,q,i,j) = 0.0;
          for ( int k = 0; k < dimension_; ++k ) {
            for ( int l = 0; l < dimension_; ++l ) {
              stress(e,q,i,j) += Cr(e,i,j,k,l)*E(e,q,k,l);
            } // end l
          } // end k
        } // end j
      } // end i
    }
  });
  if (have_energy) {
    auto T = wkset->getData("e");
    parallel_for("CE stress 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const size_type e ) {
      for (int q=0; q<numip; q++) {
        for ( int i = 0; i < dimension_; ++i ) {
          stress(e,q,i,i) += -alpha_T*(3*lambda+2*mu)*(T(e,q)-e_ref);
        }
      }
    });
  }
  //double time4 = timer.seconds();
  //printf("time 4:   %e \n\n\n", time4);
  //timer.reset();
  
}
//----------------------------------------------------------------------------

void CrystalElastic::computeRotatedTensor(Teuchos::RCP<workset> & wkset) {
  
  Teuchos::TimeMonitor rtimer(*computeRotatedTensorTimer);
  
  Kokkos::View<ScalarT**,AssemblyDevice> rl("CE-rl",3,3);
  
  bool allow_rotations_ = allow_rotations;
  bool use_phi = wkset->have_rotation_phi;
  bool use_rotation = wkset->have_rotation;
  
  auto phi = wkset->rotation_phi;
  auto rotation = wkset->rotation;
  
  int dimension_ = dimension;
  
  // Elastic tensor in rotated frame
  if (wkset->numElem > (int)Cr.extent(0)) {
    Cr = View_Sc5("CE-Cr",wkset->numElem,3,3,3,3);
  }
  
  auto C_ = C;
  
  parallel_for("CE stress 3D",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const size_type e ) {
      
    if (allow_rotations_) {
      if (use_phi) {
        // Read Bunge Angle in degrees
        ScalarT phi1d = phi(e,0);
        ScalarT Phid = phi(e,1);
        ScalarT phi2d = phi(e,2);
        
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
      else if (use_rotation) {
        for(int i=0; i<3; i++) {
          for(int j=0; j<3; j++) {
            rl(i,j) = rotation(e,i,j);
          }
        }
      }
      else {
        rl(0,0) = 1.0;
        rl(1,1) = 1.0;
        rl(2,2) = 1.0;
      }
    }
    else {
      rl(0,0) = 1.0;
      rl(1,1) = 1.0;
      rl(2,2) = 1.0;
    }
    
    // Form rotate elasticity tensor
    for ( int i = 0; i < dimension_; ++i ){
      for ( int j = 0; j < dimension_; ++j ){
        for ( int k = 0; k < dimension_; ++k ){
          for ( int l = 0; l < dimension_; ++l ){
            Cr(e,i,j,k,l) = 0.0;
            for ( int i1 = 0; i1 < dimension_; ++i1 ){
              for ( int j1 = 0; j1 < dimension_; ++j1 ){
                for ( int k1 = 0; k1 < dimension_; ++k1 ){
                  for ( int l1 = 0; l1 < dimension_; ++l1 ){
                    Cr(e,i,j,k,l) += rl(i,i1)*rl(j,j1)*rl(k,k1)*rl(l,l1)*C_(i1,j1,k1,l1);
                  }
                }
              }
            }
          }
        }
      }
    }
  });
  
}

