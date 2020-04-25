/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "uqInterface.hpp"

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

uqmanager::uqmanager(const MpiComm & Comm_, const Teuchos::ParameterList & uqsettings_,
                     const std::vector<string> & param_types_,
                     const std::vector<ScalarT> & param_means_, const std::vector<ScalarT> & param_variances_,
                     const std::vector<ScalarT> & param_mins_, const std::vector<ScalarT> & param_maxs_) :
Comm(Comm_), uqsettings(uqsettings_), param_types(param_types_), param_means(param_means_),
param_variances(param_variances_), param_mins(param_mins_), param_maxs(param_maxs_) {
  
  numstochparams = param_types.size();
  surrogate = uqsettings.get<std::string>("Surrogate model","regression");
  evalprog = 0;
  if (surrogate == "regression") {
    
  }
  else if (surrogate == "sparse grid") {
  }
  else if (surrogate == "voronoi") {
  }
  else {
    // complain
  }
  
}

// ========================================================================================
// ========================================================================================

std::vector<std::vector<ScalarT> > uqmanager::getNewPoints() {};

// ========================================================================================
// ========================================================================================

std::vector<std::vector<ScalarT> > uqmanager::getAllPoints() {};

// ========================================================================================
// ========================================================================================

std::vector<ScalarT> uqmanager::evaluateSurrogate(Kokkos::View<ScalarT**,HostDevice> samplepts) {};

// ========================================================================================
// ========================================================================================

Kokkos::View<ScalarT**,HostDevice> uqmanager::generateSamples(const int & numsamples, int & seed) {
  if (seed == -1) {
    //srand(time(NULL));
    seed = rand();
  }
  
  Kokkos::View<ScalarT**,HostDevice> samples("samples",numsamples, numstochparams);
  std::default_random_engine generator(seed);
  for (int j=0; j<numstochparams; j++) {
    if (param_types[j] == "uniform") {
      std::uniform_real_distribution<ScalarT> distribution(param_mins[j],param_maxs[j]);
      for (int k=0; k<numsamples; k++) {
        ScalarT number = distribution(generator);
        samples(k,j) = number;
      }
    }
    else if (param_types[j] == "Gaussian") {
      std::normal_distribution<ScalarT> distribution(param_means[j],param_variances[j]);
      for (int k=0; k<numsamples; k++) {
        ScalarT number = distribution(generator);
        samples(k,j) = number;
      }
    }
  }
  return samples;
}

// ========================================================================================
// ========================================================================================

Kokkos::View<int*,HostDevice> uqmanager::generateIntegerSamples(const int & numsamples, int & seed) {
  if (seed == -1) {
    //srand(time(NULL));
    seed = rand();
  }
  
  Kokkos::View<int*,HostDevice> samples("samples",numsamples);
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(1,1000000);
  for (int k=0; k<numsamples; k++) {
    ScalarT number = distribution(generator);
    samples(k) = number;
  }
  return samples;
}

// ========================================================================================
// ========================================================================================

void uqmanager::generateSamples(const int & numsamples, int & seed,
                                Kokkos::View<ScalarT**,HostDevice> samplepts,
                                Kokkos::View<ScalarT*,HostDevice> samplewts) {
  Kokkos::resize(samplepts,numsamples, numstochparams);
  Kokkos::resize(samplewts, numsamples);
  
  if (seed == -1) {
    //srand(time(NULL));
    seed = rand();
  }
  
  std::default_random_engine generator(seed);
  for (int j=0; j<numstochparams; j++) {
    if (param_types[j] == "uniform") {
      std::uniform_real_distribution<ScalarT> distribution(param_mins[j],param_maxs[j]);
      for (int k=0; k<numsamples; k++) {
        ScalarT number = distribution(generator);
        samplepts(k,j) = number;
        samplewts(k) = 1.0/(ScalarT)numsamples;
      }
    }
    else if (param_types[j] == "Gaussian") {
      std::normal_distribution<ScalarT> distribution(param_means[j],param_variances[j]);
      for (int k=0; k<numsamples; k++) {
        ScalarT number = distribution(generator);
        samplepts(k,j) = number;
        samplewts(k) = 1.0/(ScalarT)numsamples;
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

void uqmanager::computeStatistics(const std::vector<ScalarT> & values) {
  int numvals = values.size();
  if (uqsettings.get<bool>("Compute mean",true)) {
    ScalarT meanval = 0.0;
    for (int j=0; j<numvals; j++) {
      meanval += values[j];
    }
    meanval = meanval / numvals;
    if (Comm.getRank() == 0 )
    cout << "Mean value of the response: " << meanval << endl;
  }
  if (uqsettings.get<bool>("Compute variance",true)) {
    ScalarT meanval = 0.0;
    for (int j=0; j<numvals; j++) {
      meanval += values[j];
    }
    meanval = meanval / numvals;
    ScalarT variance = 0.0;
    for (int j=0; j<numvals; j++) {
      variance += (values[j]-meanval)*(values[j]-meanval);
    }
    variance = variance / numvals;
    if (Comm.getRank() == 0 )
    cout << "Variance of the response: " << variance << endl;
  }
  if (uqsettings.isSublist("Probability levels")) {
    Teuchos::ParameterList plevels = uqsettings.sublist("Probability levels");
    Teuchos::ParameterList::ConstIterator pl_itr = plevels.begin();
    while (pl_itr != plevels.end()) {
      ScalarT currplevel = plevels.get<ScalarT>(pl_itr->first);
      int count = 0;
      for (int j=0; j<numvals; j++) {
        if (values[j] <= currplevel)
        count += 1;
      }
      ScalarT currprob = (ScalarT)count / (ScalarT)numvals;
      if (Comm.getRank() == 0 )
      cout << "Probability the response is less than " << currplevel << " = " << currprob << endl;
      pl_itr++;
    }
  }
  
  
}

// ========================================================================================
// ========================================================================================

void uqmanager::computeStatistics(const vector<Kokkos::View<ScalarT***,HostDevice> > & values) {
  int numvals = values.size();
  // assumes that values[i] is a rank-3 FC
  int dim0 = values[0].extent(0);
  int dim1 = values[0].extent(1);
  int dim2 = values[0].extent(2);
  
  if (uqsettings.get<bool>("Compute mean",true)) {
    
    Kokkos::View<ScalarT***,HostDevice> meanval("mean values",dim0,dim1,dim2);
    for (int j=0; j<numvals; j++) {
      for (int d0=0; d0<dim0; d0++) {
        for (int d1=0; d1<dim1; d1++) {
          for (int d2=0; d2<dim2; d2++) {
            meanval(d0,d1,d2) += values[j](d0,d1,d2)/numvals;
          }
        }
      }
    }
    if (Comm.getRank() == 0 )
    cout << "Mean value of the response: " << endl;
    // KokkosTools::print(meanval); // GH: commenting this out for now; it can't tell DRV from DRVint
  }
  /*if (uqsettings.get<bool>("Compute variance",true)) {
   ScalarT meanval = 0.0;
   for (int j=0; j<numvals; j++) {
   meanval += values[j];
   }
   meanval = meanval / numvals;
   ScalarT variance = 0.0;
   for (int j=0; j<numvals; j++) {
   variance += (values[j]-meanval)*(values[j]-meanval);
   }
   variance = variance / numvals;
   if (Comm.getRank() == 0 )
   cout << "Variance of the response: " << variance << endl;
   }
   if (uqsettings.isSublist("Probability levels")) {
   Teuchos::ParameterList plevels = uqsettings.sublist("Probability levels");
   Teuchos::ParameterList::ConstIterator pl_itr = plevels.begin();
   while (pl_itr != plevels.end()) {
   ScalarT currplevel = plevels.get<ScalarT>(pl_itr->first);
   int count = 0;
   for (int j=0; j<numvals; j++) {
   if (values[j] <= currplevel)
   count += 1;
   }
   ScalarT currprob = (ScalarT)count / (ScalarT)numvals;
   if (Comm.getRank() == 0 )
   cout << "Probability the response is less than " << currplevel << " = " << currprob << endl;
   pl_itr++;
   }
   }*/
  
  
}
