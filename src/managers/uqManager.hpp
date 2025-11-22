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
  
  /** \class UQManager
   *  \brief Manages uncertainty quantification tasks such as sampling, statistics,
   *         kernel density estimation, and rejection sampling.
   */
  class UQManager {
    
  public:
    
    UQManager() {}; ///< Default constructor (does nothing)
    
    ~UQManager() {}; ///< Default destructor
  
    /**
     * @brief Constructor that initializes the UQ manager and stores all parameter info.
     *
     * @param[in]  comm           Teuchos MPI Communicator
     * @param[in]  uqsettings     ParameterList with UQ configuration settings
     * @param[in]  param_types    Types of distributions for each parameter
     * @param[in]  param_means    Means of each parameter
     * @param[in]  param_variances Variances of each parameter
     * @param[in]  param_mins     Minimum allowed values for each parameter
     * @param[in]  param_maxs     Maximum allowed values for each parameter
     */
    UQManager(const Teuchos::RCP<MpiComm> comm, const Teuchos::ParameterList & uqsettings,
              const std::vector<string> & param_types,
              const std::vector<ScalarT> & param_means, const std::vector<ScalarT> & param_variances,
              const std::vector<ScalarT> & param_mins, const std::vector<ScalarT> & param_maxs);
    
    /**
     * @brief Generates samples from the user-specified probability distributions.
     *
     * @param[in]  numsamples   Number of samples to generate
     * @param[in]  seed         Seed for random number generator
     * @return 2D Kokkos::View with shape (numsamples x num_parameters)
     */
    Kokkos::View<ScalarT**,HostDevice> generateSamples(const int & numsamples, int & seed);
    
    /**
     * @brief Generates integer samples from discrete user-specified distributions.
     *
     * @param[in]  numsamples   Number of samples to generate
     * @param[in]  seed         Seed for random number generator
     * @return 1D Kokkos::View<int> of length numsamples
     */
    Kokkos::View<int*,HostDevice> generateIntegerSamples(const int & numsamples, int & seed);
    
    /**
     * @brief Generates samples and fills both sample values and their weights.
     *
     * @param[in]  numsamples   Number of samples
     * @param[in]  seed         Random number generator seed
     * @param[out] samplepts    Generated sample points
     * @param[out] samplewts    Corresponding weights (typically uniform 1/N)
     */
    void generateSamples(const int & numsamples, int & seed,
                         Kokkos::View<ScalarT**,HostDevice> samplepts,
                         Kokkos::View<ScalarT*,HostDevice> samplewts);
    
    /**
     * @brief Computes statistics such as mean and variance from a vector of values.
     *
     * @param[in] values  Data vector
     */
    void computeStatistics(const std::vector<ScalarT> & values);
    
    /**
     * @brief Computes statistics for a collection of QOIs.
     *
     * @param[in]  values  Vector of 3D Kokkos Views containing QOI data
     */
    void computeStatistics(const vector<Kokkos::View<ScalarT***,HostDevice> > & values);
    
    /**
     * @brief Standard Gaussian kernel density estimator.
     *
     * @param[in] seedpts  Kernel centers
     * @param[in] evalpts  Evaluation points
     * @return Density estimate at each eval point
     */
    View_Sc1 KDE(View_Sc2 seedpts, View_Sc2 evalpts);

    /**
     * @brief Computes the variance of each QOI in a data table.
     *
     * @param[in] pts   2D Kokkos::View with all data
     * @return 1D View containing variances
     */
    View_Sc1 computeVariance(View_Sc2 pts);

    /**
     * @brief Rejection sampling based on ratios of densities.
     *
     * @param[in] ratios  Target/generated density ratios
     * @param[in] seed    Optional seed for RNG (default: -1 = do not reseed)
     * @return Boolean View indicating which samples are accepted
     */
    Kokkos::View<bool*,HostDevice> rejectionSampling(View_Sc1 ratios, const int & seed = -1);

  private:

    Teuchos::RCP<MpiComm> comm_; ///< MPI communicator
    std::string surrogate_;      ///< Surrogate type or model name
    int numstochparams_;         ///< Number of stochastic parameters handled
    bool use_user_defined_;      ///< Flag for user-defined distribution settings
    Teuchos::ParameterList uqsettings_; ///< Parameter list storing UQ configuration

    std::vector<string> param_types_;      ///< Distribution type for each parameter
    std::vector<ScalarT> param_means_;     ///< Mean values of parameters
    std::vector<ScalarT> param_variances_; ///< Variances of parameters
    std::vector<ScalarT> param_mins_;      ///< Lower bounds of parameters
    std::vector<ScalarT> param_maxs_;      ///< Upper bounds of parameters
    
    Kokkos::View<ScalarT**,HostDevice> samples_; ///< Cached/generated sample storage
  };
}
#endif
