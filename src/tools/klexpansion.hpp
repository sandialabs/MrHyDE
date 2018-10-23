/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef KLEXP_H
#define KLEXP_H

using namespace std;

class klexpansion {
public:
  
  klexpansion() {} ;
  
  /////////////////////////////////////////////////////////////////////////////
  //  Various constructors depending on the characteristics of the data (spatial, 
  //  transient, stochastic, etc.)
  /////////////////////////////////////////////////////////////////////////////
  
  klexpansion(const int & N_, const double & L_, const double & sigma_, const double & eta_) :
    N(N_), L(L_), sigma(sigma_), eta(eta_) {
    
    this->computeRoots();

  }
  
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  void computeRoots() {
    double ig = 1.0;
    double step = 1.0;
    double ctol = 1.0e-6;
    double nltol = 1.0e-10;
    int maxiter = 1000;
    int iter = 0;
    double om = ig;
    double f, df;
    while (omega.size() < N && iter < maxiter) {
      iter++;
      ig += step;
      om = ig;
      f = chareqn(om);
      while (abs(f) > nltol) {
        df = dchareqn(om);
        om += -f/df;
        f = chareqn(om);
     cout << "omega = " << om << "  f = " << f << endl;
      }
      if (omega.size() > 0) {
        if (abs(om-omega[omega.size()-1]) > ctol) {
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

  double chareqn(const double & om) {
    double f = (eta*eta*om*om - 1.0)*sin(om*L) - 2.0*eta*om*cos(om*L);
    return f;
  }

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////

  double dchareqn(const double & om) {
    double df = 2.0*om*eta*eta*sin(om*L)+(eta*eta*om*om - 1.0)*L*cos(om*L) - 2.0*eta*cos(om*L) + 2.0*eta*om*L*sin(om*L);
    return df;
  }

  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  double getEval(const int & i) const {
    double lam = (2.0*eta*sigma*sigma) / (eta*eta*omega[i]*omega[i]+1.0);
    return lam; 
  }
  
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
  double getEvec(const int & i, const double & x) const {
    double f = 1.0/(sqrt((eta*eta*omega[i]*omega[i]+1.0)*L/2.0 + eta))*(eta*omega[i]*cos(omega[i]*x) + sin(omega[i]*x));
    return f; 
  }
  
  /////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////
  
protected:
  
  double eta, L, sigma;
  int N;

  vector<double> omega;
 
};

#endif
