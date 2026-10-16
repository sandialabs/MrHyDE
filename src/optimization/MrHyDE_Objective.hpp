/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef ROL_MILO_HPP
#define ROL_MILO_HPP

#include "ROL_StdVector.hpp"
#include "ROL_RiskVector.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BoundConstraint.hpp"
#include "Teuchos_ParameterList.hpp"

#include "solverManager.hpp"
#include "postprocessManager.hpp"
#include "parameterManager.hpp"

//#include "ROL_RiskVector.hpp"

#include <iostream>
#include <fstream>
#include <string>

//#include <random> //for normal noise...not sure if this is necessary...

namespace ROL {

  using namespace MrHyDE;
  
  template<class Real>
  class Objective_MILO : public Objective<Real> {
    
  private:
    
    Real noise_;                                            //standard deviation of normal additive noise to add to data (0 for now)
    Teuchos::RCP<SolverManager<SolverNode> > solver;                                     // Solver object for MILO (solves FWD, ADJ, computes gradient, etc.)
    Teuchos::RCP<PostprocessManager<SolverNode> > postproc;                              // Postprocessing object for MILO (write solution, computes response, etc.)
    Teuchos::RCP<ParameterManager<SolverNode> > params;
    Teuchos::RCP<Teuchos::Time> valuetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Objective::value()");
    Teuchos::RCP<Teuchos::Time> gradienttimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Objective::gradient()");
    Teuchos::RCP<Teuchos::Time> hessvectimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Objective::hessVec()");
    
  public:
    
    /*!
     \brief A constructor generating data
     */
    Objective_MILO(Teuchos::RCP<SolverManager<SolverNode> > solver_,
                   Teuchos::RCP<PostprocessManager<SolverNode> > postproc_,
                   Teuchos::RCP<ParameterManager<SolverNode> > & params_) :
    solver(solver_), postproc(postproc_), params(params_) {
      
    } //end constructor
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    Real value(const Vector<Real> & Params, Real & tol){
      
      Teuchos::TimeMonitor localtimer(*valuetimer);
      
      MrHyDE_OptVector Paramsp =
      Teuchos::dyn_cast<MrHyDE_OptVector >(const_cast<Vector<Real> &>(Params));
      
      params->updateParams(Paramsp);
      
      ScalarT val = 0.0;
      solver->forwardModel(val);
      
      //params->stashParams(); //dumping to file, for long runs...
      return val;
    }
    
    //! Compute gradient of objective function with respect to parameters
    void gradient(Vector<Real> &g, const Vector<Real> &Params, Real &tol){

      Teuchos::TimeMonitor localtimer(*gradienttimer);
      bool newparams = this->checkNewParams(Params);

      if (newparams) {
        MrHyDE_OptVector Paramsp =
        Teuchos::dyn_cast<MrHyDE_OptVector >(const_cast<Vector<Real> &>(Params));
      
        params->updateParams(Paramsp);
        ScalarT val = 0.0;
        solver->forwardModel(val);

      }
      MrHyDE_OptVector sens = 
      Teuchos::dyn_cast<MrHyDE_OptVector >(const_cast<Vector<Real> &>(g));
      sens.zero();
      solver->adjointModel(sens);
      
    }
    
    bool checkNewParams(const Vector<Real> &Params) {
      MrHyDE_OptVector curr_params = params->getCurrentVector();
      auto diff = curr_params.clone();
      diff->zero();
      diff->set(curr_params);
      diff->axpy(-1.0,Params);
      ScalarT dnorm = diff->norm();
      ScalarT refnorm = curr_params.norm();
      dnorm = dnorm/refnorm;
      ScalarT reltol = 1.0e-12;
      bool newparams = false;
      if (dnorm > reltol) {
        newparams = true;
      }
      return newparams;
    }

    // Exact hessVec when src_gate is present; else ROL FD-of-gradients.
    void hessVec(Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &Params, Real &tol) override {

      if (!params->isParameter("src_gate")) {
        this->ROL::Objective<Real>::hessVec(hv, v, Params, tol);
        return;
      }

      Teuchos::TimeMonitor localtimer(*hessvectimer);

      if (this->checkNewParams(Params)) {
        MrHyDE_OptVector u =
          Teuchos::dyn_cast<MrHyDE_OptVector>(const_cast<Vector<Real>&>(Params));
        params->updateParams(u);
        ScalarT val = 0.0;
        solver->forwardModel(val);
      }

      MrHyDE_OptVector & v_dir =
        Teuchos::dyn_cast<MrHyDE_OptVector>(const_cast<Vector<Real>&>(v));
      MrHyDE_OptVector & Hv = Teuchos::dyn_cast<MrHyDE_OptVector>(hv);
      Hv.zero();

      solver->incrementalForwardModel(v_dir);
      solver->incrementalAdjointModel(Hv);

      // Tangent borrowed params for v; restore u so the next value()/gradient()
      // does not think the control changed.
      MrHyDE_OptVector u =
        Teuchos::dyn_cast<MrHyDE_OptVector>(const_cast<Vector<Real>&>(Params));
      params->updateParams(u);
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
    
  }; //end description of Objective class
  
  /*!
   \brief Inequality constraints on optimization parameters
   
   
   ---
   */
  
}//end namespace ROL

#endif

