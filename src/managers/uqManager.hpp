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
    
    // ========================================================================================
    // ========================================================================================
    
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
    
    /**
     * @brief Generates ScalarT (double) samples from the user-specified distributions.
     *
     * @param[in]  numsamples   Number of samples to generate
     * @param[in]  seed         Sets the seed for the random number generator
     * @param[out] samplepts      The actual samples.
     * @param[out] samplewts      The weights for each sample, typically 1/N
     */
    
    void generateSamples(const int & numsamples, int & seed,
                         Kokkos::View<ScalarT**,HostDevice> samplepts,
                         Kokkos::View<ScalarT*,HostDevice> samplewts);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief Compute mean, variance, etc.
     *
     * @param[in]  values    Data to compute stats from.  Stats are output to screen.
     */
    
    void computeStatistics(const std::vector<ScalarT> & values);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief Compute mean, variance, etc. for a collection of QofI..
     *
     * @param[in]  values    Data to compute stats from.  Stats are output to screen.
     */
    
    void computeStatistics(const vector<Kokkos::View<ScalarT***,HostDevice> > & values);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * @brief Standard Gaussian kernel density estimator.
     *
     * @param[in]  seedpts    Points to serve as centers for the kernels.
     * @param[in]  evalpts    Points to evaluate KDE at.  May be the same as seedpts.
     * @param[out]  output    Values of the density estimate at each eval pt.
     */
    
    View_Sc1 KDE(View_Sc2 seedpts, View_Sc2 evalpts);

    // ========================================================================================
    // ========================================================================================

    /**
     * @brief Just compute the variance for a collection of QofI.
     *
     * @param[in]  pts    2-dimensional Kokkos::View with all of the data.
     * @param[out]  output    1-dimensional Kokkos::View with the variances for each QofI.
     */
    
    View_Sc1 computeVariance(View_Sc2 pts);

    // ========================================================================================
    // ========================================================================================

    /**
     * @brief Basis rejection sampling routine.
     *
     * @param[in]  ratios    1-dimensional Kokkos::View with the ratios of two densities (target/generated)
     * @param[in]  seed    Optional bool to seed to random uniform numbers in [0,1] used in rejection sampling algorithm.
     * @param[out]  output    1-dimensional Kokkos::View with the logicals on whether to keep each data point
     */
    
    Kokkos::View<bool*,HostDevice> rejectionSampling(View_Sc1 ratios, const int & seed = -1);

    // ========================================================================================
    // ========================================================================================
    
  private:

    Teuchos::RCP<MpiComm> comm_;
    std::string surrogate_;
    int numstochparams_;
    bool use_user_defined_;
    Teuchos::ParameterList uqsettings_;
    std::vector<string> param_types_;
    std::vector<ScalarT> param_means_, param_variances_, param_mins_, param_maxs_;
    
    Kokkos::View<ScalarT**,HostDevice> samples_;
  };
}
#endif
