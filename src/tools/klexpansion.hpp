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

#ifndef MRHYDE_KLEXP_H
#define MRHYDE_KLEXP_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  class klexpansion {
  public:
    
    /////////////////////////////////////////////////////////////////////////////
    //  Various constructors depending on the characteristics of the data (spatial,
    //  transient, stochastic, etc.)
    /////////////////////////////////////////////////////////////////////////////
    klexpansion() {};
    
    klexpansion(const size_t & N, const ScalarT & L,
                const ScalarT & sigma, const ScalarT & eta) :
    N_(N), L_(L), sigma_(sigma), eta_(eta) {
      
      omega_ = View_Sc1("storage of KL omega",N_);
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
      ScalarT f, df, fprev;
      fprev = this->chareqn(om);
      std::vector<ScalarT> tmp_omega;
      while (tmp_omega.size() < N_ && iter < maxiter) {
        iter++;
        ig += step;
        om = ig;
        f = this->chareqn(om);
        if (f*fprev < 0) {
          fprev = f;
          int nliter = 0;
          int maxnliter = 10;
          while (std::abs(f) > nltol && nliter < maxnliter) {
            nliter++;
            df = this->dchareqn(om);
            om += -f/df;
            f = this->chareqn(om);
            //std::cout << "omega = " << om << "  f = " << f << std::endl;
          }
          if (tmp_omega.size() > 0) {
            bool prefnd = false;
            for (size_t j=0; j<tmp_omega.size(); ++j) {
              if (std::abs(om-tmp_omega[j]) < ctol) {
                prefnd = true;
              }
            }
            if (!prefnd) {
              tmp_omega.push_back(om);
            }
          }
          else {
            tmp_omega.push_back(om);
          }
        }
      }
      auto host_omega = create_mirror_view(omega_);
      for (size_t k=0; k<tmp_omega.size(); ++k) {
        host_omega(k) = tmp_omega[k];
      }
      deep_copy(omega_,host_omega);
      
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    KOKKOS_INLINE_FUNCTION
    ScalarT chareqn(const ScalarT & om) {
      using namespace std;
      ScalarT f = (eta_*eta_*om*om - 1.0)*sin(om*L_) - 2.0*eta_*om*cos(om*L_);
      return f;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    KOKKOS_INLINE_FUNCTION
    ScalarT dchareqn(const ScalarT & om) {
      using namespace std;
      ScalarT df = 2.0*om*eta_*eta_*sin(om*L_)+(eta_*eta_*om*om - 1.0)*L_*cos(om*L_) - 2.0*eta_*cos(om*L_) + 2.0*eta_*om*L_*sin(om*L_);
      return df;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    KOKKOS_INLINE_FUNCTION
    ScalarT getEval(const int & i) const {
      using namespace std;
      ScalarT lam = (2.0*eta_*sigma_*sigma_) / (eta_*eta_*omega_(i)*omega_(i)+1.0);
      return lam;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    KOKKOS_INLINE_FUNCTION
    ScalarT getEvec(const int & i, const ScalarT & x) const {
      using namespace std;
      ScalarT f = 1.0/(sqrt((eta_*eta_*omega_(i)*omega_(i)+1.0)*L_/2.0 + eta_))*(eta_*omega_(i)*cos(omega_(i)*x) + sin(omega_(i)*x));
      return f;
    }
    
    size_t getNumTerms() {
      return N_;
    }
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
  private:

    size_t N_;
    ScalarT L_, sigma_, eta_;
    
    View_Sc1 omega_;
  };
  
}

#endif
