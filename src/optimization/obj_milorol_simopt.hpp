/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef ROL_MILO_SIMOPT_HPP
#define ROL_MILO_SIMOPT_HPP

#include "ROL_StdVector.hpp"
#include "ROL_Objective.hpp"
#include "ROL_StdVector.hpp"
#include "ROL_Vector_SimOpt.hpp"
#include "ROL_Constraint_SimOpt.hpp"
#include "ROL_Objective_SimOpt.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"

#include "ROL_BoundConstraint.hpp"
#include "Teuchos_ParameterList.hpp"

#include "solverInterface.hpp"
#include "postprocessManager.hpp"
#include "parameterManager.hpp"

//#include "ROL_CVaRVector.hpp"

#include <iostream>
#include <fstream>
#include <string>

//#include <random> //for normal noise...not sure if this is necessary...

//namespace ROL {

using namespace MrHyDE;

template<class Real>
class Objective_MILO_SimOpt : public ROL::Objective_SimOpt<Real> {
  
  private:
  
  Real noise_;                                            //standard deviation of normal additive noise to add to data (0 for now)
  Teuchos::RCP<solver> solver_MILO;                                     // Solver object for MILO (solves FWD, ADJ, computes gradient, etc.)
  Teuchos::RCP<PostprocessManager> postproc_MILO;                              // Postprocessing object for MILO (write solution, computes response, etc.)
  Teuchos::RCP<ParameterManager> params;
  
  public:
  
  /*!
   \brief A constructor generating data
   */
  Objective_MILO_SimOpt(Teuchos::RCP<solver> solver_MILO_,
                        Teuchos::RCP<PostprocessManager> postproc_MILO_,
                        Teuchos::RCP<ParameterManager> & params_) :
  solver_MILO(solver_MILO_), postproc_MILO(postproc_MILO_), params(params_) {
    
  } //end constructor
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  Real value(const ROL::Vector<Real> &u, const ROL::Vector<Real> &Params, Real &tol){
    
    // Teuchos::RCP<const std::vector<Real> > Paramsp =
    // (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(Params))).getVector();
    
    // params->updateParams(*Paramsp, 1);
    // AD val = 0.0;
    // vector_RCP F_soln = solver_MILO->forwardModel(val);
    
    // //AD val = postproc_MILO->computeObjective(F_soln);
    
    // params->stashParams(); //dumping to file, for long runs...
    
    // return val.val();
    
    Teuchos::RCP<const std::vector<Real> > Paramsp =
    (Teuchos::dyn_cast<ROL::StdVector<Real> >(const_cast<ROL::Vector<Real> &>(Params))).getVector();
    params->updateParams(*Paramsp, 1);
    params->updateParams(*Paramsp, 4);
    DFAD val= 0.0;
    solver_MILO->forwardModel(val);
    params->stashParams(); //dumping to file, for long runs...
    return val.val();
  }
  
  
  /** \brief Compute gradient with respect to first component.
   */
  void gradient_1( ROL::Vector<Real> &g, const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    Real ftol  = std::sqrt(ROL::ROL_EPSILON<Real>());
    Real h     = 0.0;
    this->update(u,z);
    Real v     = this->value(u,z,ftol);
    Real deriv = 0.0;
    Teuchos::RCP<ROL::Vector<Real> > unew = u.clone();
    g.zero();
    for (int i = 0; i < g.dimension(); i++) {
      h = u.dot(*u.basis(i))*tol;
      unew->set(u);
      unew->axpy(h,*(u.basis(i)));
      this->update(*unew,z);
      deriv = (this->value(*unew,z,ftol) - v)/h;
      g.axpy(deriv,*(g.basis(i)));
    }
    this->update(u,z);
  }
  /** \brief Compute gradient with respect to second component.
   */
  void gradient_2( ROL::Vector<Real> &g, const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    Real ftol  = std::sqrt(ROL::ROL_EPSILON<Real>());
    Real h     = 0.0;
    this->update(u,z);
    Real v     = this->value(u,z,ftol);
    Real deriv = 0.0;
    Teuchos::RCP<ROL::Vector<Real> > znew = z.clone();
    g.zero();
    for (int i = 0; i < g.dimension(); i++) {
      h = z.dot(*z.basis(i))*tol;
      znew->set(z);
      znew->axpy(h,*(z.basis(i)));
      this->update(u,*znew);
      deriv = (this->value(u,*znew,ftol) - v)/h;
      g.axpy(deriv,*(g.basis(i)));
    }
    this->update(u,z);
  }
  
  void gradient( ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real &tol ) {
    ROL::Vector_SimOpt<Real> &gs = Teuchos::dyn_cast<ROL::Vector_SimOpt<Real> >(
                                                                                Teuchos::dyn_cast<ROL::Vector<Real> >(g));
    const ROL::Vector_SimOpt<Real> &xs = Teuchos::dyn_cast<const ROL::Vector_SimOpt<Real> >(
                                                                                            Teuchos::dyn_cast<const ROL::Vector<Real> >(x));
    Teuchos::RCP<ROL::Vector<Real> > g1 = gs.get_1()->clone();
    Teuchos::RCP<ROL::Vector<Real> > g2 = gs.get_2()->clone();
    this->gradient_1(*g1,*(xs.get_1()),*(xs.get_2()),tol);
    this->gradient_2(*g2,*(xs.get_1()),*(xs.get_2()),tol);
    gs.set_1(*g1);
    gs.set_2(*g2);
  }
  
  
  /** \brief Apply Hessian approximation to vector.
   */
  void hessVec_11( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v,
                  const ROL::Vector<Real> &u,  const ROL::Vector<Real> &z, Real &tol ) {
    Real gtol = std::sqrt(ROL::ROL_EPSILON<Real>());
    // Compute step length
    Real h = tol;
    if (v.norm() > std::sqrt(ROL::ROL_EPSILON<Real>())) {
      h = std::max(1.0,u.norm()/v.norm())*tol;
    }
    // Evaluate gradient of first component at (u+hv,z)
    Teuchos::RCP<ROL::Vector<Real> > unew = u.clone();
    unew->set(u);
    unew->axpy(h,v);
    this->update(*unew,z);
    hv.zero();
    this->gradient_1(hv,*unew,z,gtol);
    // Evaluate gradient of first component at (u,z)
    Teuchos::RCP<ROL::Vector<Real> > g = hv.clone();
    this->update(u,z);
    this->gradient_1(*g,u,z,gtol);
    // Compute Newton quotient
    hv.axpy(-1.0,*g);
    hv.scale(1.0/h);
  }
  
  void hessVec_12( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v,
                  const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    Real gtol = std::sqrt(ROL::ROL_EPSILON<Real>());
    // Compute step length
    Real h = tol;
    if (v.norm() > std::sqrt(ROL::ROL_EPSILON<Real>())) {
      h = std::max(1.0,u.norm()/v.norm())*tol;
    }
    // Evaluate gradient of first component at (u,z+hv)
    Teuchos::RCP<ROL::Vector<Real> > znew = z.clone();
    znew->set(z);
    znew->axpy(h,v);
    this->update(u,*znew);
    hv.zero();
    this->gradient_1(hv,u,*znew,gtol);
    // Evaluate gradient of first component at (u,z)
    Teuchos::RCP<ROL::Vector<Real> > g = hv.clone();
    this->update(u,z);
    this->gradient_1(*g,u,z,gtol);
    // Compute Newton quotient
    hv.axpy(-1.0,*g);
    hv.scale(1.0/h);
  }
  
  void hessVec_21( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v,
                  const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol ) {
    Real gtol = std::sqrt(ROL::ROL_EPSILON<Real>());
    // Compute step length
    Real h = tol;
    if (v.norm() > std::sqrt(ROL::ROL_EPSILON<Real>())) {
      h = std::max(1.0,u.norm()/v.norm())*tol;
    }
    // Evaluate gradient of first component at (u+hv,z)
    Teuchos::RCP<ROL::Vector<Real> > unew = u.clone();
    unew->set(u);
    unew->axpy(h,v);
    this->update(*unew,z);
    hv.zero();
    this->gradient_2(hv,*unew,z,gtol);
    // Evaluate gradient of first component at (u,z)
    Teuchos::RCP<ROL::Vector<Real> > g = hv.clone();
    this->update(u,z);
    this->gradient_2(*g,u,z,gtol);
    // Compute Newton quotient
    hv.axpy(-1.0,*g);
    hv.scale(1.0/h);
  }
  
  void hessVec_22( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v,
                  const ROL::Vector<Real> &u,  const ROL::Vector<Real> &z, Real &tol ) {
    Real gtol = std::sqrt(ROL::ROL_EPSILON<Real>());
    // Compute step length
    Real h = tol;
    if (v.norm() > std::sqrt(ROL::ROL_EPSILON<Real>())) {
      h = std::max(1.0,u.norm()/v.norm())*tol;
    }
    // Evaluate gradient of first component at (u,z+hv)
    Teuchos::RCP<ROL::Vector<Real> > znew = z.clone();
    znew->set(z);
    znew->axpy(h,v);
    this->update(u,*znew);
    hv.zero();
    this->gradient_2(hv,u,*znew,gtol);
    // Evaluate gradient of first component at (u,z)
    Teuchos::RCP<ROL::Vector<Real> > g = hv.clone();
    this->update(u,z);
    this->gradient_2(*g,u,z,gtol);
    // Compute Newton quotient
    hv.axpy(-1.0,*g);
    hv.scale(1.0/h);
  }
  
  void hessVec( ROL::Vector<Real> &hv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real &tol ) {
    ROL::Vector_SimOpt<Real> &hvs = Teuchos::dyn_cast<ROL::Vector_SimOpt<Real> >(
                                                                                 Teuchos::dyn_cast<ROL::Vector<Real> >(hv));
    const ROL::Vector_SimOpt<Real> &vs = Teuchos::dyn_cast<const ROL::Vector_SimOpt<Real> >(
                                                                                            Teuchos::dyn_cast<const ROL::Vector<Real> >(v));
    const ROL::Vector_SimOpt<Real> &xs = Teuchos::dyn_cast<const ROL::Vector_SimOpt<Real> >(
                                                                                            Teuchos::dyn_cast<const ROL::Vector<Real> >(x));
    Teuchos::RCP<ROL::Vector<Real> > h11 = (hvs.get_1())->clone();
    this->hessVec_11(*h11,*(vs.get_1()),*(xs.get_1()),*(xs.get_2()),tol);
    Teuchos::RCP<ROL::Vector<Real> > h12 = (hvs.get_1())->clone();
    this->hessVec_12(*h12,*(vs.get_2()),*(xs.get_1()),*(xs.get_2()),tol);
    Teuchos::RCP<ROL::Vector<Real> > h21 = (hvs.get_2())->clone();
    this->hessVec_21(*h21,*(vs.get_1()),*(xs.get_1()),*(xs.get_2()),tol);
    Teuchos::RCP<ROL::Vector<Real> > h22 = (hvs.get_2())->clone();
    this->hessVec_22(*h22,*(vs.get_2()),*(xs.get_1()),*(xs.get_2()),tol);
    h11->plus(*h12);
    hvs.set_1(*h11);
    h22->plus(*h21);
    hvs.set_2(*h22);
  }
  
  //bvbw old code below
  
  /*
   //! Compute gradient of objective function with respect to parameters
   void gradient(Vector<Real> &g, const Vector<Real> &Params, Real &tol){
   
   Teuchos::RCP<std::vector<Real> > gp =
   Teuchos::rcp_const_cast<std::vector<Real> >((Teuchos::dyn_cast<StdVector<Real> >(g)).getVector());
   Teuchos::RCP<const std::vector<Real> > Paramsp =
   (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(Params))).getVector();
   
   solver_MILO->updateParams(*Paramsp, 1);
   
   Epetra_MultiVector F_soln = solver_MILO->forwardModel();
   Epetra_MultiVector A_soln = solver_MILO->adjointModel(F_soln);
   
   std::vector<ScalarT> sens = postproc_MILO->computeSensitivities(F_soln, A_soln);
   
   for (size_t i=0; i<sens.size(); i++)
   (*gp)[i] = sens[i];
   
   }
   
   
   //! Compute the Hessian-vector product of the objective function
   void hessVec(Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &Params, Real &tol ){
   this->ROL::Objective<Real>::hessVec(hv,v,Params,tol);
   }
   
   */
  
  //print out Hessian (estimated via component-wise FD; to get inverse covariance in linear-Gaussian Bayesian inverse problem)
  void printHess(const std::string & filename, const ROL::Vector<Real> & xin, const int & commrank){
    ROL::StdVector<Real> x = Teuchos::dyn_cast<ROL::StdVector<Real> >(const_cast<ROL::Vector<Real> &>(xin));
    int paramDim = x.getVector()->size();
    
    Teuchos::RCP<ROL::StdVector<Real> > g = Teuchos::rcp(new ROL::StdVector<Real>(Teuchos::rcp(new vector<Real>(paramDim,0.0))));
    ScalarT gtol = sqrt(ROL::ROL_EPSILON<Real>());
    this->gradient(*g,x,gtol);
    Teuchos::RCP<ROL::StdVector<Real> > gnew = Teuchos::rcp(new ROL::StdVector<Real>(Teuchos::rcp(new vector<Real>(paramDim,0.0))));
    
    vector<vector<ScalarT> > hessStash(paramDim);
    
    //Real h = 1.e-3*x.norm(); //step length
    Real h = std::max(1.0,x.norm())*sqrt(ROL::ROL_EPSILON<Real>()); ///step length...more like what ROL has...
    
    //perturb each component
    for(int i=0; i<x.dimension(); i++){
      //compute new step
      Teuchos::RCP<ROL::StdVector<Real> > xnew = Teuchos::rcp(new ROL::StdVector<Real>(Teuchos::rcp(new vector<Real>(paramDim,0.0))));
      xnew->set(x);
      xnew->axpy(h,*x.basis(i));
      
      //gradient at new step
      gnew->zero();
      this->gradient(*gnew,*xnew,gtol);
      
      //i-th column (or row...) of Hessian
      gnew->axpy(-1.0,*g);
      gnew->scale(1.0/h);
      Teuchos::RCP<vector<ScalarT> > gnewv = gnew->getVector();
      
      vector<ScalarT> row(paramDim);
      for(int j=0; j<paramDim; j++)
      row[j] = (*gnewv)[j];
      hessStash[i] = row;
    }
    
    if(commrank == 0){
      std::ofstream respOUT(filename);
      respOUT.precision(16);
      for(int i=0; i<paramDim; i++){
        for(int j=0; j<paramDim; j++)
        respOUT << hessStash[i][j] << " ";
        respOUT << endl;
      }
      respOUT.close();
    }
  }
  
  /*!
   \brief Generate data to plot objective function
   */
  void generate_plot(Real difflo, Real diffup, Real diffstep){
    
    Teuchos::RCP<std::vector<ScalarT> > Params_rcp = Teuchos::rcp(new std::vector<ScalarT>(1,0.0) );
    ROL::StdVector<ScalarT> Params(Params_rcp);
    std::ofstream output ("Objective.dat");
    
    Real diff = 0.0;
    Real val = 0.0;
    Real tol = 1.e-16;
    int n = (diffup-difflo)/diffstep + 1;
    for(int i=0;i<n;i++){
      diff = difflo + i*diffstep;
      (*Params_rcp)[0] = diff;
      val = value(Params,tol); // GH: removed "this->". need to revisit if this should call base or derived method.
      if(output.is_open()){
        output << std::scientific << diff << " " << val << "\n";
      }
    }
    output.close();
  }
  
  /*!
   \brief Generate data to plot objective function
   */
  void generate_plot(Real alo, Real aup, Real astep, Real blo, Real bup, Real bstep){
    
    Teuchos::RCP<std::vector<ScalarT> > Params_rcp = Teuchos::rcp(new std::vector<ScalarT>(2,0.0) );
    ROL::StdVector<ScalarT> Params(Params_rcp);
    std::ofstream output ("Objective.dat");
    
    Real a = 0.0;
    Real b = 0.0;
    Real val = 0.0;
    Real tol = 1.e-16;
    int n = (aup-alo)/astep + 1;
    int m = (bup-blo)/bstep + 1;
    for(int i=0;i<n;i++){
      a = alo + i*astep;
      for(int j=0;j<m;j++){
        b = blo + j*bstep;
        (*Params_rcp)[0] = a;
        (*Params_rcp)[1] = b;
        val = value(Params,tol); // GH: removed "this->". need to revisit if this should call base or derived method.
        if(output.is_open()){
          output << std::scientific << a << " " << b << " " << val << "\n";
        }
      }
    }
    output.close();
  }
  
}; //end description of Objective_MILO_simopt class

/*!
 \brief Inequality constraints on optimization parameters
 
 
 ---
 */
template<class Real>
class BoundConstraint_MILO_SimOpt : public ROL::BoundConstraint<Real> {
  private:
  /// Vector of lower bounds
  std::vector<Real> x_lo_;
  /// Vector of upper bounds
  std::vector<Real> x_up_;
  /// Half of the minimum distance between upper and lower bounds
  Real min_diff_;
  /// Scaling for the epsilon margin
  Real scale_;
  public:
  /*BoundConstraint_MILO( Real scale, Real lo_diff, Real up_diff){
   x_lo_.push_back(lo_diff);
   x_up_.push_back(up_diff);
   
   scale_ = scale;
   min_diff_ = 0.5*(x_up_[0]-x_lo_[0]);
   }
   
   BoundConstraint_MILO( Real scale, Real lo_a, Real up_a, Real lo_b, Real up_b ){
   x_lo_.push_back(lo_a);
   x_lo_.push_back(lo_b);
   
   x_up_.push_back(up_a);
   x_up_.push_back(up_b);
   
   scale_ = scale;
   min_diff_ = 0.5*std::min(x_up_[0]-x_lo_[0],x_up_[1]-x_lo_[1]);
   }*/
  
  BoundConstraint_MILO_SimOpt( Real scale, std::vector<Real> lovec, std::vector<Real> hivec){ //keeping most general version...
    min_diff_ = 0.0;
    for(int i=0; i<lovec.size(); i++){
      x_lo_.push_back(lovec[i]);
      x_up_.push_back(hivec[i]);
      min_diff_ = std::min(min_diff_,x_up_[i]-x_lo_[i]);
    }
    scale_ = scale;
    min_diff_ = 0.5*min_diff_;
  }
  
  void project( ROL::Vector<Real> &x ) {
    //Teuchos::RCP<std::vector<Real> > ex =
    //  Teuchos::rcp_const_cast<std::vector<Real> >((Teuchos::dyn_cast<StdVector<Real> >(x)).getVector());
    Teuchos::RCP<std::vector<Real> > ex;
    /* try{ //dubious programming practice, but CVaR doesn't play nice with original dynamic casting...
     ex = Teuchos::rcp_const_cast<std::vector<Real> >((Teuchos::dyn_cast<StdVector<Real> >(x)).getVector());
     }
     catch(const std::bad_cast &e){
     Teuchos::RCP<Vector<Real> > x0 = Teuchos::rcp_const_cast<Vector<Real> >(Teuchos::dyn_cast<const CVaRVector<Real> >(
     Teuchos::dyn_cast<const Vector<Real> >(x)).getVector());
     ex = (Teuchos::dyn_cast<ROL::StdVector<Real> >(const_cast<ROL::Vector<Real> &>(*x0))).getVector();
     } */
    for(int i=0; i<x_lo_.size(); i++){
      (*ex)[i] = std::max(x_lo_[i],std::min(x_up_[i],(*ex)[i]));
    }
  }
  
  bool isFeasible( const ROL::Vector<Real> &x ) {
    //Teuchos::RCP<const std::vector<Real> > ex =
    //  (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(x))).getVector();
    Teuchos::RCP<const std::vector<Real> > ex;
    /* try{ //dubious programming practice, but CVaR doesn't play nice with original dynamic casting...
     ex = (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(x))).getVector();
     }
     catch(const std::bad_cast &e){
     Teuchos::RCP<const Vector<Real> > x0 = Teuchos::rcp_const_cast<Vector<Real> >(Teuchos::dyn_cast<const CVaRVector<Real> >(
     Teuchos::dyn_cast<const Vector<Real> >(x)).getVector());
     ex = (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(*x0))).getVector();
     } */
    bool returnme = true;
    for(int i=0; i<x_lo_.size(); i++){
      returnme = (returnme && ((*ex)[i] >= this->x_lo_[i] && (*ex)[i] <= this->x_up_[i]));
    }
    
    return returnme;
  }
  
  void pruneActive(ROL::Vector<Real> &v, const ROL::Vector<Real> &x, Real eps) {
    //Teuchos::RCP<const std::vector<Real> > ex =
    //  (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(x))).getVector();
    //Teuchos::RCP<std::vector<Real> > ev =
    //  Teuchos::rcp_const_cast<std::vector<Real> >((Teuchos::dyn_cast<StdVector<Real> >(v)).getVector());
    
    Teuchos::RCP<std::vector<Real> > ev;
    /* try{ //dubious programming practice, but CVaR doesn't play nice with original dynamic casting...
     ev = Teuchos::rcp_const_cast<std::vector<Real> >((Teuchos::dyn_cast<StdVector<Real> >(v)).getVector());
     }
     catch(const std::bad_cast &e){
     Teuchos::RCP<Vector<Real> > x0 = Teuchos::rcp_const_cast<Vector<Real> >(Teuchos::dyn_cast<const CVaRVector<Real> >(
     Teuchos::dyn_cast<const Vector<Real> >(v)).getVector());
     ev = (Teuchos::dyn_cast<ROL::StdVector<Real> >(const_cast<ROL::Vector<Real> &>(*x0))).getVector();
     } */
    Teuchos::RCP<const std::vector<Real> > ex;
    /* try{ //dubious programming practice, but CVaR doesn't play nice with original dynamic casting...
     ex = (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(x))).getVector();
     }
     catch(const std::bad_cast &e){
     Teuchos::RCP<const Vector<Real> > x0 = Teuchos::rcp_const_cast<Vector<Real> >(Teuchos::dyn_cast<const CVaRVector<Real> >(
     Teuchos::dyn_cast<const Vector<Real> >(x)).getVector());
     ex = (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(*x0))).getVector();
     } */
    
    Real epsn = std::min(this->scale_*eps,this->min_diff_);
    //epsn *= this->scale_;
    for ( int i=0; i<x_lo_.size(); i++ ) {
      if ( ((*ex)[i] <= this->x_lo_[i]+epsn) ||
          ((*ex)[i] >= this->x_up_[i]-epsn) ) {
        (*ev)[i] = 0.0;
      }
    }
  }
  
  void pruneActive(ROL::Vector<Real> &v, const ROL::Vector<Real> &g, const ROL::Vector<Real> &x, Real eps) {
    //Teuchos::RCP<const std::vector<Real> > ex =
    //  (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(x))).getVector();
    //Teuchos::RCP<const std::vector<Real> > eg =
    //  (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(g))).getVector();
    //Teuchos::RCP<std::vector<Real> > ev =
    //  Teuchos::rcp_const_cast<std::vector<Real> >((Teuchos::dyn_cast<StdVector<Real> >(v)).getVector());
    
    Teuchos::RCP<const std::vector<Real> > ex;
    /* try{ //dubious programming practice, but CVaR doesn't play nice with original dynamic casting...
     ex = (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(x))).getVector();
     }
     catch(const std::bad_cast &e){
     Teuchos::RCP<const Vector<Real> > x0 = Teuchos::rcp_const_cast<Vector<Real> >(Teuchos::dyn_cast<const CVaRVector<Real> >(
     Teuchos::dyn_cast<const Vector<Real> >(x)).getVector());
     ex = (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(*x0))).getVector();
     } */
    Teuchos::RCP<const std::vector<Real> > eg;
    /* try{ //dubious programming practice, but CVaR doesn't play nice with original dynamic casting...
     eg = (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(g))).getVector();
     }
     catch(const std::bad_cast &e){
     Teuchos::RCP<const Vector<Real> > x0 = Teuchos::rcp_const_cast<Vector<Real> >(Teuchos::dyn_cast<const CVaRVector<Real> >(
     Teuchos::dyn_cast<const Vector<Real> >(g)).getVector());
     eg = (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(*x0))).getVector();
     } */
    Teuchos::RCP<std::vector<Real> > ev;
    /* try{ //dubious programming practice, but CVaR doesn't play nice with original dynamic casting...
     ev = Teuchos::rcp_const_cast<std::vector<Real> >((Teuchos::dyn_cast<StdVector<Real> >(v)).getVector());
     }
     catch(const std::bad_cast &e){
     Teuchos::RCP<Vector<Real> > x0 = Teuchos::rcp_const_cast<Vector<Real> >(Teuchos::dyn_cast<const CVaRVector<Real> >(
     Teuchos::dyn_cast<const Vector<Real> >(v)).getVector());
     ev = (Teuchos::dyn_cast<ROL::StdVector<Real> >(const_cast<ROL::Vector<Real> &>(*x0))).getVector();
     } */
    
    Real epsn = std::min(this->scale_*eps,this->min_diff_);
    //epsn *= this->scale_;
    for ( int i=0; i<x_lo_.size(); i++ ) {
      if ( ((*ex)[i] <= this->x_lo_[i]+epsn && (*eg)[i] > 0.0) ||
          ((*ex)[i] >= this->x_up_[i]-epsn && (*eg)[i] < 0.0) ) {
        (*ev)[i] = 0.0;
      }
    }
  }
}; //end description of BoundConstraint_MILO class


template<class Real>
class Constraint_MILO_SimOpt : public ROL::Constraint_SimOpt<Real> {
  private:
  
  // bvbw need the solver object
  //  Teuchos::RCP<PoissonData<Real> > data_;
  Teuchos::RCP<solver> solver_MILO;               // Solver object for MILO (solves FWD, ADJ, computes gradient, etc.)
  public:
  
  Constraint_MILO_SimOpt(Teuchos::RCP<solver> &solver_MILO_,
                         Teuchos::RCP<Teuchos::ParameterList> &parlist) {
    solver_MILO = solver_MILO_;
  }
  
  using ROL::Constraint_SimOpt<Real>::value;
  //  using ROL::Constraint_MILO_SimOpt<Real>::value;
  void value(ROL::Vector<Real> &c, const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) {
  }
  
  
  void applyJacobian_1(ROL::Vector<Real> &jv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                       const ROL::Vector<Real> &z, Real &tol) {
    
  }
  
  
  void applyJacobian_2(ROL::Vector<Real> &jv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                       const ROL::Vector<Real> &z, Real &tol) {
    
  }
  
  
  void applyAdjointJacobian_1(ROL::Vector<Real> &ajv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                              const ROL::Vector<Real> &z, Real &tol) {
    
  }
  
  
  void applyAdjointJacobian_2(ROL::Vector<Real> &ajv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                              const ROL::Vector<Real> &z, Real &tol) {
    
  }
  
  
  void applyAdjointHessian_11(ROL::Vector<Real> &ahwv, const ROL::Vector<Real> &w, const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) {
    ahwv.zero();
  }
  
  
  void applyAdjointHessian_12(ROL::Vector<Real> &ahwv, const ROL::Vector<Real> &w, const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) {
    ahwv.zero();
  }
  
  
  void applyAdjointHessian_21(ROL::Vector<Real> &ahwv, const ROL::Vector<Real> &w, const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) {
    ahwv.zero();
  }
  
  
  void applyAdjointHessian_22(ROL::Vector<Real> &ahwv, const ROL::Vector<Real> &w, const ROL::Vector<Real> &v,
                              const ROL::Vector<Real> &u, const ROL::Vector<Real> &z, Real &tol) {
    ahwv.zero();
  }
  
  
  void applyInverseJacobian_1(ROL::Vector<Real> &ijv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                              const ROL::Vector<Real> &z, Real &tol) {
    
    
  }
  
  
  void applyInverseAdjointJacobian_1(ROL::Vector<Real> &iajv, const ROL::Vector<Real> &v, const ROL::Vector<Real> &u,
                                     const ROL::Vector<Real> &z, Real &tol) {
  }
};  // end of Constraint class


//}//end namespace ROL

#endif

