/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef ROL_MILO_STOCH_HPP
#define ROL_MILO_STOCH_HPP

#include "ROL_StdVector.hpp"
#include "ROL_RiskVector.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BoundConstraint.hpp"
#include "ROL_SampleGenerator.hpp"
#include "Teuchos_ParameterList.hpp"

#include "solverManager.hpp"
#include "postprocessManager.hpp"
#include "parameterManager.hpp"

#include <iostream>
#include <fstream>
#include <string>

namespace ROL {

  using namespace MrHyDE;
  
  template<class Real>
  class Stochastic_Objective_MILO : public Objective<Real> {
    
  private:
    
    Teuchos::RCP<SolverManager<SolverNode> > solver;                                     // Solver object for MILO (solves FWD, ADJ, computes gradient, etc.)
    Teuchos::RCP<PostprocessManager<SolverNode> > postproc;                              // Postprocessing object for MILO (write solution, computes response, etc.)
    Teuchos::RCP<ParameterManager<SolverNode> > params;
    Teuchos::RCP<ROL::SampleGenerator<Real>> sampler;
    Teuchos::RCP<Teuchos::Time> valuetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Objective::value()");
    Teuchos::RCP<Teuchos::Time> gradienttimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Objective::gradient()");
    bool params_updated;
    
  public:
    
    /*!
     \brief A constructor generating data
     */
    Stochastic_Objective_MILO(Teuchos::RCP<SolverManager<SolverNode> > solver_,
                              Teuchos::RCP<PostprocessManager<SolverNode> > postproc_,
                              Teuchos::RCP<ParameterManager<SolverNode> > & params_,
                              Teuchos::RCP<ROL::SampleGenerator<Real>> & sampler_) :
    solver(solver_), postproc(postproc_), params(params_), sampler(sampler_) {
      params_updated = true;
    } //end constructor
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    Real value(const Vector<Real> & x, Real & tol) override {
      
      Teuchos::TimeMonitor localtimer(*valuetimer);
      
      MrHyDE_OptVector xp =
      Teuchos::dyn_cast<MrHyDE_OptVector >(const_cast<Vector<Real> &>(x));
      
      params->updateParams(xp);
      
      ScalarT val = 0.0;
      solver->forwardModel(val);
      params_updated = false;
      
      int id = IdentifySample();
      std::cout << "id = " << id << std::endl;
      // create object to store state solution after forward solve

      return val;
    }
    
    //! Compute gradient of objective function with respect to parameters
    void gradient(Vector<Real> &g, const Vector<Real> &x, Real &tol) override {

      Teuchos::TimeMonitor localtimer(*gradienttimer);
      bool new_x = this->checkNewx(x);

      int id = IdentifySample();
      // create object to access state solution after forward solve

      if (new_x || params_updated) {
        MrHyDE_OptVector xp =
        Teuchos::dyn_cast<MrHyDE_OptVector >(const_cast<Vector<Real> &>(x));
      
        params->updateParams(xp);
        ScalarT val = 0.0;
        solver->forwardModel(val);
        params_updated = false;
      }
      else
      {
        std::cout << "I am computing the gradient without repeating the state solve" << std::endl;
        // The code can be improved by caching the state solution for each inactive parameter sample
        // so that we avoid repeating the state solve when gradient is called
      }
      MrHyDE_OptVector sens = 
      Teuchos::dyn_cast<MrHyDE_OptVector >(const_cast<Vector<Real> &>(g));
      sens.zero();
      solver->adjointModel(sens);
      
    }
    
    //! Compute the Hessian-vector product of the objective function
    void hessVec(Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &x, Real &tol ) override {
      this->ROL::Objective<Real>::hessVec(hv,v,x,tol);
    }

    int IdentifySample(void)
    {
      int id = -1;
      std::vector<Real> param = params->getParams("inactive");
      for(int i = 0; i < sampler->numMySamples(); i++)
      {
        std::vector<Real> pt_i = sampler->getMyPoint(i);
        Real val = 0.0;
        for(int k = 0; k < pt_i.size(); k++)
        {
          val += std::pow(pt_i[k] - param[k],2.0);
        }
        if(val < 1.e-14)
        {
          id = i;
          break;
        }
      }
      if(id == -1)
      {
        std::cout << "Error in Stochastic_Objective_MILO: Did not identify the sample" << std::endl;
      }
      return id;
    }

    bool checkNewx(const Vector<Real> &x) {
      MrHyDE_OptVector curr_x = params->getCurrentVector();
      auto diff = curr_x.clone();
      diff->zero();
      diff->set(curr_x);
      diff->axpy(-1.0,x);
      ScalarT dnorm = diff->norm();
      ScalarT refnorm = curr_x.norm();
      dnorm = dnorm/refnorm;
      ScalarT reltol = 1.0e-12;
      bool new_x = false;
      if (dnorm > reltol) {
        new_x = true;
      }
      return new_x;
    }
    
    void setParameter(const std::vector<Real> &param) override
    {
      // The code is designed to consider only inactive scalar parameters. 
      // param.size() should be equal to the number of inactive scalar parameters
      // Generalizing to include vector parameters should be easy, but we have not done it yet
      params->updateParams(param, "inactive");
      params_updated = true;
    }

  }; //end description of Objective class
  
}//end namespace ROL

#endif

