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

#ifndef KLEXP_H
#define KLEXP_H

//using namespace std;

namespace MrHyDE {
  
  class klexpansion {
  public:
    
    /////////////////////////////////////////////////////////////////////////////
    //  Various constructors depending on the characteristics of the data (spatial,
    //  transient, stochastic, etc.)
    /////////////////////////////////////////////////////////////////////////////
    
    klexpansion(const size_t & N_, const ScalarT & L_, const ScalarT & sigma_, const ScalarT & eta_) :
    N(N_), L(L_), sigma(sigma_), eta(eta_) {
      
      this->computeRoots();
      
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void computeRoots() {
      ScalarT ig = 1.0;
      ScalarT step = 1.0;
      ScalarT ctol = 1.0e-6;
      ScalarT nltol = 1.0e-10;
      int maxiter = 1000;
      int iter = 0;
      ScalarT om = ig;
      ScalarT f, df;
      while (omega.size() < N && iter < maxiter) {
        iter++;
        ig += step;
        om = ig;
        f = this->chareqn(om);
        while (std::abs(f) > nltol) {
          df = this->dchareqn(om);
          om += -f/df;
          f = this->chareqn(om);
          std::cout << "omega = " << om << "  f = " << f << std::endl;
        }
        if (omega.size() > 0) {
          if (std::abs(om-omega[omega.size()-1]) > ctol) {
            omega.push_back(om);
          }
        }
        else {
          omega.push_back(om);
        }
      }
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    ScalarT chareqn(const ScalarT & om) {
      ScalarT f = (eta*eta*om*om - 1.0)*std::sin(om*L) - 2.0*eta*om*std::cos(om*L);
      return f;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    ScalarT dchareqn(const ScalarT & om) {
      ScalarT df = 2.0*om*eta*eta*std::sin(om*L)+(eta*eta*om*om - 1.0)*L*std::cos(om*L) - 2.0*eta*std::cos(om*L) + 2.0*eta*om*L*std::sin(om*L);
      return df;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    ScalarT getEval(const int & i) const {
      ScalarT lam = (2.0*eta*sigma*sigma) / (eta*eta*omega[i]*omega[i]+1.0);
      return lam;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    ScalarT getEvec(const int & i, const ScalarT & x) const {
      ScalarT f = 1.0/(std::sqrt((eta*eta*omega[i]*omega[i]+1.0)*L/2.0 + eta))*(eta*omega[i]*std::cos(omega[i]*x) + std::sin(omega[i]*x));
      return f;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
  protected:
    
    size_t N;
    ScalarT L, sigma, eta;
    
    std::vector<ScalarT> omega;
    
  };
  
}

#endif
