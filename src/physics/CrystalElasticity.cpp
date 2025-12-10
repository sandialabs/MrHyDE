/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "CrystalElasticity.hpp"

using namespace MrHyDE;

template<class EvalT>
CrystalElastic<EvalT>::CrystalElastic(Teuchos::ParameterList & settings,
                                      const int & dimension_)
{
  
  dimension = dimension_;
   
  Teuchos::ParameterList cesettings = settings.sublist("Crystal elastic parameters");

  ScalarT te_ref, talpha_T, tlambda, tmu;
  ScalarT tc11_, tc22_, tc33_, tc44_, tc55_, tc66_, tc12_, tc13_, tc23_, tc15_, tc25_, tc35_, tc46_;
  te_ref = cesettings.get<ScalarT>("T_ambient",0.0);
  talpha_T = cesettings.get<ScalarT>("alpha_T",1.0e-6);
  
  allow_rotations = cesettings.get<bool>("allow rotations",true);
  
  ScalarT E = cesettings.get<ScalarT>("E",1.0);
  ScalarT nu = cesettings.get<ScalarT>("nu",0.4);
  
  tlambda = (E*nu)/((1.0+nu)*(1.0-2.0*nu));
  tmu = E/(2.0*(1.0+nu));
  
  // Gas constant: TMW: Need to make this a parameter
  // ScalarT R_ = esettings.get<ScalarT>("R",0.0);
  
  // Elastic tensor in lattice frame
  C = View_EvalT4("CE-C",3,3,3,3);
  
  // default to cubic symmetry
  tc11_ = cesettings.get<ScalarT>("C11",2.0*tmu+tlambda);
  tc22_ = cesettings.get<ScalarT>("C22",tc11_);
  tc33_ = cesettings.get<ScalarT>("C33",tc11_);
  tc44_ = cesettings.get<ScalarT>("C44",2.0*tmu);
  tc55_ = cesettings.get<ScalarT>("C55",tc44_);
  tc66_ = cesettings.get<ScalarT>("C66",tc44_);
  tc12_ = cesettings.get<ScalarT>("C12",tlambda);
  tc13_ = cesettings.get<ScalarT>("C13",tc12_);
  tc23_ = cesettings.get<ScalarT>("C23",tc12_);
  tc15_ = cesettings.get<ScalarT>("C15",0.0);
  tc25_ = cesettings.get<ScalarT>("C25",0.0);
  tc35_ = cesettings.get<ScalarT>("C35",0.0);
  tc46_ = cesettings.get<ScalarT>("C46",0.0);
  
  // update, just in case they changed
  lambda = tc12_;
  mu = tc44_/2.0;
  e_ref = te_ref;
  alpha_T = talpha_T;

  c11_ = tc11_;
  c22_ = tc22_;
  c33_ = tc33_;
  c44_ = tc44_;
  c55_ = tc55_;
  c66_ = tc66_;
  c12_ = tc12_;
  c13_ = tc13_;
  c23_ = tc23_;
  c15_ = tc15_;
  c25_ = tc25_;
  c35_ = tc35_;
  c46_ = tc46_;

  this->computeLatticeTensor();
  
}

//=====================================================

template<class EvalT>
void CrystalElastic<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
  wkset = wkset_;
}

//----------------------------------------------------------------------------

template<class EvalT>
void CrystalElastic<EvalT>::computeLatticeTensor() {
  
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
  
  //deep_copy(C,C_host);
  
}

//----------------------------------------------------------------------------

template<class EvalT>
void CrystalElastic<EvalT>::updateParams() {
  
  EvalT c11 = c11_;
  EvalT c12 = c12_;
  EvalT c44 = c44_;
  
  bool foundlam = false;
  auto lamvals = wkset->getParameter("lambda", foundlam);
  if (foundlam) {
    lambda = lamvals(0);
  }
  
  bool foundmu = false;
  auto muvals = wkset->getParameter("mu", foundlam);
  if (foundmu) {
    mu = muvals(0);
  }
  
  if (!foundlam || !foundmu) {
    EvalT E = 0.0;
    bool foundym = false;
    auto ymvals = wkset->getParameter("youngs_mod", foundym);
    if (foundym) {
      E = ymvals(0);
    }
    
    EvalT nu = 0.0;
    bool foundpr = false;
    auto prvals = wkset->getParameter("poisson_ratio", foundpr);
    if (foundpr) {
      nu = prvals(0);
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

template<class EvalT>
void CrystalElastic<EvalT>::computeStress(Teuchos::RCP<Workset<EvalT> > & wkset, vector<int> & indices,
                                   const bool & onside, View_EvalT4 stress)
{
  
  Teuchos::TimeMonitor stimer(*computeStressTimer);
  
  //Kokkos::Timer timer;
  this->updateParams();
  
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
  
  View_EvalT4 E("CE-E",wkset->numElem,numip,dimension,dimension);
  
  //double time2 = timer.seconds();
  //printf("time 2:   %e \n", time2);
  //timer.reset();
  
  if (dimension == 1) {
    auto dx_x = wkset->getSolutionField("grad(dx)[x]");
    parallel_for("CE stress 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_CLASS_LAMBDA (const size_type e ) {
      for (size_type k=0; k<dx_x.extent(1); k++) {
        E(e,k,0,0) = dx_x(e,k);
      }
    });
  }
  else if (dimension == 2) {
    auto dx_x = wkset->getSolutionField("grad(dx)[x]");
    auto dx_y = wkset->getSolutionField("grad(dx)[y]");
    auto dy_x = wkset->getSolutionField("grad(dy)[x]");
    auto dy_y = wkset->getSolutionField("grad(dy)[y]");
    parallel_for("CE stress 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_CLASS_LAMBDA (const size_type e ) {
      for (size_type k=0; k<dx_x.extent(1); k++) {
        E(e,k,0,0) = dx_x(e,k);
        E(e,k,0,1) = 0.5*dx_y(e,k) + 0.5*dy_x(e,k);
        E(e,k,1,0) = 0.5*dy_x(e,k) + 0.5*dx_y(e,k);
        E(e,k,1,1) = dy_y(e,k);
      }
    });
  }
  else if (dimension == 3) {
    auto dx_x = wkset->getSolutionField("grad(dx)[x]");
    auto dx_y = wkset->getSolutionField("grad(dx)[y]");
    auto dx_z = wkset->getSolutionField("grad(dx)[z]");
    auto dy_x = wkset->getSolutionField("grad(dy)[x]");
    auto dy_y = wkset->getSolutionField("grad(dy)[y]");
    auto dy_z = wkset->getSolutionField("grad(dy)[z]");
    auto dz_x = wkset->getSolutionField("grad(dz)[x]");
    auto dz_y = wkset->getSolutionField("grad(dz)[y]");
    auto dz_z = wkset->getSolutionField("grad(dz)[z]");
    parallel_for("CE stress 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_CLASS_LAMBDA (const size_type e ) {
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
               KOKKOS_CLASS_LAMBDA (const size_type e ) {
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
    auto T = wkset->getSolutionField("e");
    parallel_for("CE stress 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_CLASS_LAMBDA (const size_type e ) {
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

template<class EvalT>
void CrystalElastic<EvalT>::computeRotatedTensor(Teuchos::RCP<Workset<EvalT> > & wkset) {
  
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
    Cr = View_EvalT5("CE-Cr",wkset->numElem,3,3,3,3);
  }
  
  auto C_ = C;
  
  parallel_for("CE stress 3D",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_CLASS_LAMBDA (const size_type e ) {
      
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


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::CrystalElastic<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::CrystalElastic<AD>;

// Standard built-in types
template class MrHyDE::CrystalElastic<AD2>;
template class MrHyDE::CrystalElastic<AD4>;
template class MrHyDE::CrystalElastic<AD8>;
template class MrHyDE::CrystalElastic<AD16>;
template class MrHyDE::CrystalElastic<AD18>;
template class MrHyDE::CrystalElastic<AD24>;
template class MrHyDE::CrystalElastic<AD32>;
#endif
