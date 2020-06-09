/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef ROL_MILO_HPP
#define ROL_MILO_HPP

#include "ROL_StdVector.hpp"
#include "ROL_RiskVector.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BoundConstraint.hpp"
#include "Teuchos_ParameterList.hpp"

#include "solverInterface.hpp"
#include "postprocessManager.hpp"
#include "parameterManager.hpp"

//#include "ROL_RiskVector.hpp"

#include <iostream>
#include <fstream>
#include <string>

//#include <random> //for normal noise...not sure if this is necessary...

namespace ROL {
  
  template<class Real>
  class Objective_MILO : public Objective<Real> {
    
  private:
    
    Real noise_;                                            //standard deviation of normal additive noise to add to data (0 for now)
    Teuchos::RCP<solver> solver_MILO;                                     // Solver object for MILO (solves FWD, ADJ, computes gradient, etc.)
    Teuchos::RCP<PostprocessManager> postproc_MILO;                              // Postprocessing object for MILO (write solution, computes response, etc.)
    Teuchos::RCP<ParameterManager> params;
  public:
    
    /*!
     \brief A constructor generating data
     */
    Objective_MILO(Teuchos::RCP<solver> solver_MILO_, Teuchos::RCP<PostprocessManager> postproc_MILO_,
                   Teuchos::RCP<ParameterManager> & params_) :
    solver_MILO(solver_MILO_), postproc_MILO(postproc_MILO_), params(params_) {
      
    } //end constructor
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    Real value(const Vector<Real> &Params, Real &tol){
      
      //Real val = 0;
      //Teuchos::RCP<Teuchos::Time> valtimer = Teuchos::rcp(new Teuchos::Time("val",false)); //DEBUG
      //valtimer->start(); //DEBUG
      Teuchos::RCP<const std::vector<Real> > Paramsp =
      (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(Params))).getVector();
      //Teuchos::RCP<Teuchos::Time> paramtimer = Teuchos::rcp(new Teuchos::Time("param",false)); //DEBUG
      //paramtimer->start(); //DEBUG
      params->updateParams(*Paramsp, 1);
      params->updateParams(*Paramsp, 4);
      //paramtimer->stop(); //DEBUG
      //Teuchos::RCP<Teuchos::Time> fwdtimer = Teuchos::rcp(new Teuchos::Time("fwd",false)); //DEBUG
      //fwdtimer->start(); //DEBUG
      DFAD val= 0.0;
      
      solver_MILO->forwardModel(val);
      // do we want to write the solution each time
      // if not, then when?
      //fwdtimer->stop(); //DEBUG
      //Teuchos::RCP<Teuchos::Time> posttimer = Teuchos::rcp(new Teuchos::Time("post",false)); //DEBUG
      //posttimer->start(); //DEBUG
      //AD val = postproc_MILO->computeObjective(F_soln);
      //posttimer->stop(); //DEBUG
      //Teuchos::RCP<Teuchos::Time> stashtimer = Teuchos::rcp(new Teuchos::Time("stash",false)); //DEBUG
      //stashtimer->start(); //DEBUG
      params->stashParams(); //dumping to file, for long runs...
      //stashtimer->stop(); //DEBUG
      //valtimer->stop(); //DEBUG
      //cout << "obj fx - set: " << paramtimer->totalElapsedTime()
      //    << " fwd: " << fwdtimer->totalElapsedTime()
      //    << " post: " << posttimer->totalElapsedTime()
      //    << " stash: " << stashtimer->totalElapsedTime()
      //    << " total: " << valtimer->totalElapsedTime() << " seconds" << endl; //DEBUG
      //postproc_MILO->writeSolution(F_soln,"output_fwd_val.exo"); //DEBUG
      
      return val.val();
    }
    
    //! Compute gradient of objective function with respect to parameters
    void gradient(Vector<Real> &g, const Vector<Real> &Params, Real &tol){
      //Teuchos::RCP<Teuchos::Time> gtimer = Teuchos::rcp(new Teuchos::Time("grad",false)); //DEBUG
      //gtimer->start(); //DEBUG
      
      Teuchos::RCP<std::vector<Real> > gp =
      Teuchos::rcp_const_cast<std::vector<Real> >((Teuchos::dyn_cast<StdVector<Real> >(g)).getVector());
      Teuchos::RCP<const std::vector<Real> > Paramsp =
      (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(Params))).getVector();
      //Teuchos::RCP<Teuchos::Time> paramtimer = Teuchos::rcp(new Teuchos::Time("param",false)); //DEBUG
      //paramtimer->start(); //DEBUG
      params->updateParams(*Paramsp, 1);
      params->updateParams(*Paramsp, 4);
      //paramtimer->stop(); //DEBUG
      //Teuchos::RCP<Teuchos::Time> fwdtimer = Teuchos::rcp(new Teuchos::Time("fwd",false)); //DEBUG
      //fwdtimer->start(); //DEBUG
      
      //DFAD obj = 0.0;
      //solver_MILO->forwardModel(obj);
      
      //fwdtimer->stop(); //DEBUG
      //Teuchos::RCP<Teuchos::Time> adjtimer = Teuchos::rcp(new Teuchos::Time("adj",false)); //DEBUG
      //adjtimer->start(); //DEBUG
      std::vector<ScalarT> sens;
      
      solver_MILO->adjointModel(sens);
      //adjtimer->stop(); //DEBUG
      //Teuchos::RCP<Teuchos::Time> senstimer = Teuchos::rcp(new Teuchos::Time("sens",false)); //DEBUG
      //senstimer->start(); //DEBUG
      //std::vector<ScalarT> sens2 = postproc_MILO->computeSensitivities(F_soln, A_soln);
      //for (size_t i=0; i<sens.size(); i++)
      //  cout << i << "  " << abs(sens[i]-sens2[i]) << endl;
      
      //senstimer->stop(); //DEBUG
      for (size_t i=0; i<sens.size(); i++)
        (*gp)[i] = sens[i];
      //gtimer->stop(); //DEBUG
      //cout << "grad fx - set: " << paramtimer->totalElapsedTime()
      //    << " fwd: " << fwdtimer->totalElapsedTime()
      //    << " adjoint: " << adjtimer->totalElapsedTime()
      //    << " sens: " << senstimer->totalElapsedTime()
      //    << " total: " << gtimer->totalElapsedTime() << " seconds" << endl; //DEBUG
      
    }
    
    //! Compute the Hessian-vector product of the objective function
    void hessVec(Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &Params, Real &tol ){
      this->ROL::Objective<Real>::hessVec(hv,v,Params,tol);
    }
    
    //print out Hessian (estimated via component-wise FD; to get inverse covariance in linear-Gaussian Bayesian inverse problem)
    void printHess(const string & filename, const Vector<Real> & xin, const int & commrank){
      StdVector<Real> x = Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(xin));
      int paramDim = x.getVector()->size();
      
      Teuchos::RCP<StdVector<Real> > g = Teuchos::rcp(new StdVector<Real>(Teuchos::rcp(new vector<Real>(paramDim,0.0))));
      ScalarT gtol = sqrt(ROL_EPSILON<Real>());
      this->gradient(*g,x,gtol);
      Teuchos::RCP<StdVector<Real> > gnew = Teuchos::rcp(new StdVector<Real>(Teuchos::rcp(new vector<Real>(paramDim,0.0))));
      
      vector<vector<ScalarT> > hessStash(paramDim);
      
      //Real h = 1.e-3*x.norm(); //step length
      Real h = std::max(1.0,x.norm())*sqrt(ROL_EPSILON<Real>()); ///step length...more like what ROL has...
      
      //perturb each component
      for(int i=0; i<x.dimension(); i++){
        //compute new step
        Teuchos::RCP<StdVector<Real> > xnew = Teuchos::rcp(new StdVector<Real>(Teuchos::rcp(new vector<Real>(paramDim,0.0))));
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
        ofstream respOUT(filename);
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
        val = this->value(Params,tol);
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
          val = this->value(Params,tol);
          if(output.is_open()){
            output << std::scientific << a << " " << b << " " << val << "\n";
          }
        }
      }
      output.close();
    }
    
  }; //end description of Objective_CDR2D class
  
  /*!
   \brief Inequality constraints on optimization parameters
   
   
   ---
   */
  
}//end namespace ROL

#endif

