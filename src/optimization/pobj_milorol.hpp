/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef ROL_PMILO_HPP
#define ROL_PMILO_HPP

#include "ROL_StdVector.hpp"
#include "ROL_Objective.hpp"
//#include "ROL_BoundConstraint.hpp"
#include "ROL_StdBoundConstraint.hpp"
#include "ROL_ParametrizedObjective.hpp"
#include "ROL_SampleGenerator.hpp"
#include "ROL_SparseGridGenerator.hpp"
#include "ROL_BatchManager.hpp"
#include "ROL_MonteCarloGenerator.hpp"
#include "ROL_RiskNeutralObjective.hpp"
#include "Teuchos_ParameterList.hpp"

//#include "CDBatchManager.hpp"
#include "solverInterface.hpp"
#include "postprocessInterface.hpp"
#include "parameterManager.hpp"

#include <iostream>
#include <fstream>
#include <string>

//#include <random> //for normal noise...not sure if this is necessary...

namespace ROL {
  
  template<class Real>
  class PObjective_MILO : public ParametrizedObjective<Real> {
    
  private:
    
    Real noise_;                                 //standard deviation of normal additive noise to add to data (0 for now)
    Teuchos::RCP<solver> solver_MILO;            // Solver object for MILO (solves FWD, ADJ, computes gradient, etc.)
    Teuchos::RCP<postprocess> postproc_MILO;     // Solver object for MILO (solves FWD, ADJ, computes gradient, etc.)
    Teuchos::RCP<ParameterManager> params;
    
  public:
    
    /*!
     \brief A constructor generating data
     */
    PObjective_MILO(Teuchos::RCP<solver> solver_MILO_,
                    Teuchos::RCP<postprocess> postproc_MILO_,
                    Teuchos::RCP<ParameterManager> & params_) :
    solver_MILO(solver_MILO_), postproc_MILO(postproc_MILO_), params(params_) {
      int dim = solver_MILO->getNumParams(2);
      std::vector<Real> param(dim,0.0); 
      this->setParameter(param);
    } //end constructor
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    Real value(const Vector<Real> &Params, Real &tol){ 
      Teuchos::RCP<const std::vector<Real> > Paramsp =
      (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(Params))).getVector();

      params->updateParams(*Paramsp, 1);
      params->updateParams(this->getParameter(), 2);
      params->updateParams(*Paramsp, 4);

      AD val = 0.0;
      vector_RCP F_soln = solver_MILO->forwardModel(val);
      // do we want to write the solution each time
      // if not, then when?

      //AD val = postproc_MILO->computeObjective(F_soln);

      params->stashParams();
      
      return val.val();
    }
    
    //! Compute gradient of objective function with respect to parameters
    void gradient(Vector<Real> &g, const Vector<Real> &Params, Real &tol){ 

      Teuchos::RCP<std::vector<Real> > gp =
      Teuchos::rcp_const_cast<std::vector<Real> >((Teuchos::dyn_cast<StdVector<Real> >(g)).getVector());
      Teuchos::RCP<const std::vector<Real> > Paramsp =
      (Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(Params))).getVector();

      params->updateParams(*Paramsp, 1);
	    params->updateParams(this->getParameter(), 2);
      params->updateParams(*Paramsp, 4);

      AD val = 0.0;
      std::vector<ScalarT> sens;
      vector_RCP F_soln = solver_MILO->forwardModel(val);
      vector_RCP A_soln = solver_MILO->adjointModel(F_soln, sens);

      //std::vector<ScalarT> sens = postproc_MILO->computeSensitivities(F_soln, A_soln);

      for (size_t i=0; i<sens.size(); i++)
        (*gp)[i] = sens[i];
    }
    
    //! Compute the Hessian-vector product of the objective function
    void hessVec(Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &Params, Real &tol ){ 
      this->ROL::Objective<Real>::hessVec(hv,v,Params,tol); 
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
}//end namespace ROL

#endif

