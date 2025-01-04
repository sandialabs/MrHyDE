/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "uqManager.hpp"
#include "data.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

UQManager::UQManager(const Teuchos::RCP<MpiComm> comm, const Teuchos::ParameterList & uqsettings,
                     const std::vector<string> & param_types,
                     const std::vector<ScalarT> & param_means, const std::vector<ScalarT> & param_variances,
                     const std::vector<ScalarT> & param_mins, const std::vector<ScalarT> & param_maxs) :
comm_(comm), uqsettings_(uqsettings), param_types_(param_types), param_means_(param_means),
param_variances_(param_variances), param_mins_(param_mins), param_maxs_(param_maxs) {
  
  RCP<Teuchos::Time> constructortime = Teuchos::TimeMonitor::getNewCounter("MrHyDE::UQManager - constructor");
  Teuchos::TimeMonitor constructortimer(*constructortime);
  
  numstochparams_ = param_types_.size();
  surrogate_ = uqsettings_.get<std::string>("surrogate_ model","regression");
  
  use_user_defined_ = uqsettings_.get<bool>("use user defined",false);
  if (use_user_defined_) {
    Data sdata("Sample Points", numstochparams_, uqsettings_.get("source","samples.dat"));
    samples_ = sdata.getPoints();
  }
  
  if (surrogate_ == "regression") {
    
  }
  else if (surrogate_ == "sparse grid") {
  }
  else if (surrogate_ == "voronoi") {
  }
  else {
    // complain
  }
  
}

// ========================================================================================
// ========================================================================================

Kokkos::View<ScalarT**,HostDevice> UQManager::generateSamples(const int & numsamples, int & seed) {

  if (!use_user_defined_) {
    if (seed == -1) {
      srand(time(NULL));
      seed = rand();
    }

    samples_ = Kokkos::View<ScalarT**,HostDevice>("samples",numsamples, numstochparams_);
    std::default_random_engine generator(seed);
    //std::mt19937 generator(seed);
    
    for (int j=0; j<numstochparams_; j++) {
      if (param_types_[j] == "uniform") {
        std::uniform_real_distribution<ScalarT> distribution(param_mins_[j],param_maxs_[j]);
        for (int k=0; k<numsamples; k++) {
          ScalarT number = distribution(generator);
          samples_(k,j) = number;
        }
      }
      else if (param_types_[j] == "Gaussian") {
        std::normal_distribution<ScalarT> distribution(param_means_[j],param_variances_[j]);
        for (int k=0; k<numsamples; k++) {
          ScalarT number = distribution(generator);
          samples_(k,j) = number;
        }
      }
    }
  }
  
  return samples_;
}

// ========================================================================================
// ========================================================================================

Kokkos::View<int*,HostDevice> UQManager::generateIntegerSamples(const int & numsamples, int & seed) {
  if (seed == -1) {
    srand(time(NULL));
    seed = rand();
  }
  
  Kokkos::View<int*,HostDevice> isamples("samples",numsamples);
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(1,1000000);
  for (int k=0; k<numsamples; k++) {
    ScalarT number = distribution(generator);
    isamples(k) = number;
  }
  return isamples;
}

// ========================================================================================
// ========================================================================================

void UQManager::generateSamples(const int & numsamples, int & seed,
                                Kokkos::View<ScalarT**,HostDevice> samplepts,
                                Kokkos::View<ScalarT*,HostDevice> samplewts) {
  Kokkos::resize(samplepts,numsamples, numstochparams_);
  Kokkos::resize(samplewts, numsamples);

  if (seed == -1) {
    srand(time(NULL));
    seed = rand();
  }
  
  std::default_random_engine generator(seed);
  for (int j=0; j<numstochparams_; j++) {
    if (param_types_[j] == "uniform") {
      std::uniform_real_distribution<ScalarT> distribution(param_mins_[j],param_maxs_[j]);
      for (int k=0; k<numsamples; k++) {
        ScalarT number = distribution(generator);
        samplepts(k,j) = number;
        samplewts(k) = 1.0/(ScalarT)numsamples;
      }
    }
    else if (param_types_[j] == "Gaussian") {
      std::normal_distribution<ScalarT> distribution(param_means_[j],param_variances_[j]);
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

void UQManager::computeStatistics(const std::vector<ScalarT> & values) {
  int numvals = values.size();
  if (uqsettings_.get<bool>("compute mean",true)) {
    ScalarT meanval = 0.0;
    for (int j=0; j<numvals; j++) {
      meanval += values[j];
    }
    meanval = meanval / numvals;
    if (comm_->getRank() == 0 )
    cout << "Mean value of the response: " << meanval << endl;
  }
  if (uqsettings_.get<bool>("compute variance",true)) {
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
    if (comm_->getRank() == 0 )
    cout << "Variance of the response: " << variance << endl;
  }
  if (uqsettings_.isSublist("Probability levels")) {
    Teuchos::ParameterList plevels = uqsettings_.sublist("Probability levels");
    Teuchos::ParameterList::ConstIterator pl_itr = plevels.begin();
    while (pl_itr != plevels.end()) {
      ScalarT currplevel = plevels.get<ScalarT>(pl_itr->first);
      int count = 0;
      for (int j=0; j<numvals; j++) {
        if (values[j] <= currplevel)
        count += 1;
      }
      ScalarT currprob = (ScalarT)count / (ScalarT)numvals;
      if (comm_->getRank() == 0 )
      cout << "Probability the response is less than " << currplevel << " = " << currprob << endl;
      pl_itr++;
    }
  }
  
  
}

// ========================================================================================
// ========================================================================================

void UQManager::computeStatistics(const vector<Kokkos::View<ScalarT***,HostDevice> > & values) {
  int numvals = values.size();
  // assumes that values[i] is a rank-3 FC
  int dim0 = values[0].extent(0);
  int dim1 = values[0].extent(1);
  int dim2 = values[0].extent(2);
  
  if (uqsettings_.get<bool>("compute mean",true)) {
    
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
    if (comm_->getRank() == 0 )
    cout << "Mean value of the response: " << endl;
    // KokkosTools::print(meanval); // GH: comm_enting this out for now; it can't tell DRV from DRVint
  }
  /*if (uqsettings_.get<bool>("Compute variance",true)) {
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
   if (comm_.getRank() == 0 )
   cout << "Variance of the response: " << variance << endl;
   }
   if (uqsettings_.isSublist("Probability levels")) {
   Teuchos::ParameterList plevels = uqsettings_.sublist("Probability levels");
   Teuchos::ParameterList::ConstIterator pl_itr = plevels.begin();
   while (pl_itr != plevels.end()) {
   ScalarT currplevel = plevels.get<ScalarT>(pl_itr->first);
   int count = 0;
   for (int j=0; j<numvals; j++) {
   if (values[j] <= currplevel)
   count += 1;
   }
   ScalarT currprob = (ScalarT)count / (ScalarT)numvals;
   if (comm_.getRank() == 0 )
   cout << "Probability the response is less than " << currplevel << " = " << currprob << endl;
   pl_itr++;
   }
   }*/
  
  
}

// ========================================================================================
// ========================================================================================

View_Sc1 UQManager::KDE(View_Sc2 seedpts, View_Sc2 evalpts) {

  // TMW: note that this won't really work on a GPU - easy to fix though

  // This is mostly translated from a matlab implementation
  size_type Ne = evalpts.extent(0);	// number of evaluation points (not seed points)
  size_type Ns = seedpts.extent(0);	// number of seed points
  size_type dim = evalpts.extent(1); // dimension of the space

  View_Sc1 density("KDE",Ne);

  View_Sc1 variance = this->computeVariance(seedpts);

  ScalarT Nsc = static_cast<ScalarT>(Ns);
  ScalarT dimsc = static_cast<ScalarT>(dim);

  vector<ScalarT> sigma(dim,0.0), coeff(dim,0.0);
  ScalarT scale = std::pow(Nsc,-1.0/(4.0+dimsc));
  for (size_type d=0; d<dim; ++d) {
    sigma[d] = 1.06*std::sqrt(variance(d))*scale;
    coeff[d] = 1.0/(std::sqrt(2.0*PI*sigma[d]*sigma[d]));
  }

  // kernel density estimation
  for (size_type i=0; i<Ne; ++i) {
    ScalarT val = 0.0;
    for (size_type k=0; k<Ns; ++k) {
      ScalarT cval = 1.0;
      for (size_type d=0; d<dim; ++d) {
        ScalarT diff = evalpts(i,d) - seedpts(k,d);
        cval *= coeff[d]*std::exp(-1.0*diff*diff/(2.0*sigma[d]*sigma[d]));
      }
      val += cval;
    }
    density(i) = val/Nsc;
  }

  return density;
    
}

// ========================================================================================
// ========================================================================================

View_Sc1 UQManager::computeVariance(View_Sc2 pts) {
  size_type N = pts.extent(0);
  size_type dim = pts.extent(1);
  View_Sc1 vars("variance",dim);

  ScalarT Nsc = static_cast<ScalarT>(N);
  auto pts_host = create_mirror_view(pts);
  deep_copy(pts_host,pts);

  for (size_type i=0; i<dim; ++i) {
    ScalarT mean = 0.0;
    for (size_type k=0; k<N; ++k) {
      mean += pts_host(k,i)/Nsc;
    }
    ScalarT var = 0.0;
    for (size_type k=0; k<N; ++k) {
      var += (pts_host(k,i)-mean)*(pts_host(k,i)-mean)/(Nsc-1.0);
    }
    vars(i) = var;
  }
  return vars;

}

// ========================================================================================
// ========================================================================================

Kokkos::View<bool*,HostDevice> UQManager::rejectionSampling(View_Sc1 ratios, const int & seed) {

  ScalarT C = 0.0;
  for (size_type k=0; k<ratios.extent(0); ++k) {
    if (std::abs(ratios(k)) > C) {
      C = std::abs(ratios(k));
    }
  }
  Kokkos::View<bool*,HostDevice> accept("acceptance",ratios.extent(0));
  deep_copy(accept,false);

  int cseed = seed;
  if (cseed == -1) {
    srand(time(NULL));
    cseed = rand();
  }
  
  std::default_random_engine generator(cseed);
  std::uniform_real_distribution<ScalarT> distribution(0.0,1.0);
        
  for (size_type k=0; k<ratios.extent(0); ++k) {
    ScalarT check = distribution(generator);
    if (std::abs(ratios(k))/C >= check) {
      accept(k) = true;
    }
  }
  return accept;
}
