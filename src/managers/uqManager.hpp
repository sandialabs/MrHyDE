/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_UQ_MANAGER_H
#define MRHYDE_UQ_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include <random>
#include <time.h>

namespace MrHyDE {
  
  class UQManager {
    
  public:
    
    UQManager() {};
    
    ~UQManager() {};
  
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    /**
     * @brief Constructor that actually sets everything up.
     *
     * @param[in]  comm        Teuchos MPI Communicator
     * @param[in]  uqsettings  Teuchos ParameterList containing a few settings specific to performing UQ
     * @param[in]  param*      Vectors of data from parameter manager
     */

    UQManager(const Teuchos::RCP<MpiComm> comm, const Teuchos::ParameterList & uqsettings,
              const std::vector<string> & param_types,
              const std::vector<ScalarT> & param_means, const std::vector<ScalarT> & param_variances,
              const std::vector<ScalarT> & param_mins, const std::vector<ScalarT> & param_maxs);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief Generates samples from the user-specified distributions.
     *
     * @param[in]  numsamples   Number of samples to generate
     * @param[in]  seed         Sets the seed for the random number generator
     * @param[out] output       2-dimensional Kokkos View of ScalarT.  Dimensions are numsamples x parameter dimension
     */

    Kokkos::View<ScalarT**,HostDevice> generateSamples(const int & numsamples, int & seed);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief Generates integer samples from the user-specified distributions.
     *
     * @param[in]  numsamples   Number of samples to generate
     * @param[in]  seed         Sets the seed for the random number generator
     * @param[out] output       1-dimensional Kokkos View of int.  Dimension is numsamples.  May generalize in the future to multi-dimensional.
     */

    Kokkos::View<int*,HostDevice> generateIntegerSamples(const int & numsamples, int & seed);
    
    // ========================================================================================
    // ========================================================================================
    
    void generateSamples(const int & numsamples, int & seed,
                         Kokkos::View<ScalarT**,HostDevice> samplepts,
                         Kokkos::View<ScalarT*,HostDevice> samplewts);
    
    // ========================================================================================
    // ========================================================================================
    
    void computeStatistics(const std::vector<ScalarT> & values);
    
    // ========================================================================================
    // ========================================================================================
    
    void computeStatistics(const vector<Kokkos::View<ScalarT***,HostDevice> > & values);
    
    // ========================================================================================
    // ========================================================================================
    
    View_Sc1 KDE(View_Sc2 seedpts, View_Sc2 evalpts);

    // ========================================================================================
    // ========================================================================================

    View_Sc1 computeVariance(View_Sc2 pts);

    // ========================================================================================
    // ========================================================================================

    Kokkos::View<bool*,HostDevice> rejectionSampling(View_Sc1 ratios, const int & seed = -1);

    // ========================================================================================
    // ========================================================================================
    
  private:

    Teuchos::RCP<MpiComm> comm_;
    std::string surrogate_;
    //std::vector<std::vector<ScalarT> > points_;
    int numstochparams_;
    bool use_user_defined_;
    Teuchos::ParameterList uqsettings_;
    std::vector<string> param_types_;
    std::vector<ScalarT> param_means_, param_variances_, param_mins_, param_maxs_;
    
    Kokkos::View<ScalarT**,HostDevice> samples_;
  };
}
#endif
