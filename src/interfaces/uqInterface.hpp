/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef UQ_H
#define UQ_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include <random>
#include <time.h>

void static uqHelp(const string & details) {
  cout << "********** Help and Documentation for the UQ Interface **********" << endl;
}

class uqmanager {
  public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  uqmanager(const Epetra_MpiComm & Comm_, const Teuchos::ParameterList & uqsettings_,
            const std::vector<string> & param_types_,
            const std::vector<double> & param_means_, const std::vector<double> & param_variances_,
            const std::vector<double> & param_mins_, const std::vector<double> & param_maxs_);
  
  // ========================================================================================
  // ========================================================================================
  
  std::vector<std::vector<double> > getNewPoints();
  
  // ========================================================================================
  // ========================================================================================
  
  std::vector<std::vector<double> > getAllPoints();
  
  // ========================================================================================
  // ========================================================================================
  
  std::vector<double> evaluateSurrogate(Kokkos::View<double**,HostDevice> samplepts);
  
  // ========================================================================================
  // ========================================================================================
  
  Kokkos::View<double**,HostDevice> generateSamples(const int & numsamples, int & seed);
  
  // ========================================================================================
  // ========================================================================================
  
  Kokkos::View<int*,HostDevice> generateIntegerSamples(const int & numsamples, int & seed);

  // ========================================================================================
  // ========================================================================================
  
  void generateSamples(const int & numsamples, int & seed,
                       Kokkos::View<double**,HostDevice> samplepts,
                       Kokkos::View<double*,HostDevice> samplewts);
  
  // ========================================================================================
  // ========================================================================================
  
  void computeStatistics(const std::vector<double> & values);
  
  // ========================================================================================
  // ========================================================================================
  
  void computeStatistics(const vector<Kokkos::View<double**,HostDevice> > & values);
  
  // ========================================================================================
  // ========================================================================================
  
  protected:
  
  Epetra_MpiComm Comm;
  std::string surrogate;
  std::vector<std::vector<double> > points;
  int evalprog, numstochparams;
  Teuchos::ParameterList uqsettings;
  std::vector<string> param_types;
  std::vector<double> param_means, param_variances, param_mins, param_maxs;
};

#endif
