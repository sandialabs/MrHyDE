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

/** @file euler.hpp
 *
 * @brief Euler physics module
 *
 * Solves the Euler equations for conservation of mass, momentum, and energy.
 * Transport and thermodynamic properties are assumed to be functions
 * of temperature.
 * We employ an ideal gas law.
 */

#ifndef MRHYDE_EULER_H
#define MRHYDE_EULER_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /** Euler physics module 
   *
   * Solves the Euler equations for conservation of mass, momentum, and energy.
   * Transport and thermodynamic properties are assumed to be functions
   * of temperature.
   * We employ an ideal gas law.
   */

  class euler : public physicsbase {
  public:

    euler() {} ;
    
    ~euler() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    euler(Teuchos::ParameterList & settings, const int & dimension);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
    
    // ========================================================================================
    // ========================================================================================
    
    void boundaryResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<workset> & wkset_);

    /* @brief Update the inviscid fluxes for the residual calculation.
     *
     * @param[in] on_side  Bool indicating if we are on an element side or not
     *
     * @details When we are at an interface, the flux is evaluated using the trace variables.
     * This should be called after updating the thermodynamic properties.
     */

    void computeInviscidFluxes(const bool & on_side);

    /* @brief Update the thermodynamic properties for the residual calculation.
     *
     * @param[in] on_side  Bool indicating if we are on an element side or not
     *
     * @details When we are at an interface, the properties are evaluated using the trace variables.
     * This should be called before computing the inviscid fluxes.
     */

    void computeThermoProps(const bool & on_side);

    /* @brief Update the stabilization term for numerical flux at interfaces.
     *
     * @details When we are at an interface, the flux 
     * \f$\hat{F}(\mathbf{S}) \cdot \mathbf{n} = 
     * F(\hat{\mathbf{S}}) \cdot \mathbf{n} + Stab(\hat{\mathbf{S}},\mathbf{S}) 
     * \times (\hat{\mathbf{S}} - \mathbf{S})\f$ where \f$Stab(\hat{\mathbf{S}},\mathbf{S})\f$
     * is the local stabilization matrix. We store the vector result of applying the matrix to
     * the trace variable discrepancy \f$(\hat{\mathbf{S}} - \mathbf{S})\f$.
     */

    void computeStabilizationTerm();

    /* @brief Update the boundary flux at the domain boundary.
     *
     * @details When we are at a domain boundary, the flux \f$\hat{B}(\mathbf{S}) = B(\hat{\mathbf{S}})
     * is used to weakly enforce the boundary condition in the computeFlux() routine.
     */

    void computeBoundaryTerm();

    /* @brief Computes the local eigenvalue decomposition for the stabilization and boundary terms.
     * This is the 1-D version.
     *
     * @param[inout] leftEV  Storage for the left eigenvectors
     * @param[inout] Lambda  Storage for the eigenvalues. A vector, not a matrix since it is diagonal.
     * @param[inout] rightEV  Storage for the right eigenvectors
     * @param[in] rhoux  x-component of the momentum
     * @param[in] rho  Density
     * @param[in] a_sound  Speed of sound
     * @param[in] gamma  Ratio of specific heats
     *
     * @details Should be called using the trace variables \f$\hat{S}\f$.
     */

    KOKKOS_FUNCTION void eigendecompFluxJacobian(View_AD2 & leftEV, View_AD1 & Lambda, View_AD2 & rightEV, 
        const AD & rhoux, const AD & rho, const AD & a_sound, const ScalarT & gamma);

    /* @brief Computes the local eigenvalue decomposition for the stabilization and boundary terms.
     * This is the 2-D version.
     *
     * @param[inout] leftEV  Storage for the left eigenvectors
     * @param[inout] Lambda  Storage for the eigenvalues. A vector, not a matrix since it is diagonal.
     * @param[inout] rightEV  Storage for the right eigenvectors
     * @param[in] rhoux  x-component of the momentum
     * @param[in] rhouy  y-component of the momentum
     * @param[in] rho  Density
     * @param[in] nx  x-component of the normal vector
     * @param[in] ny  y-component of the normal vector
     * @param[in] a_sound  Speed of sound
     * @param[in] gamma  Ratio of specific heats
     *
     * @details Should be called using the trace variables \f$\hat{S}\f$.
     */

    KOKKOS_FUNCTION void eigendecompFluxJacobian(View_AD2 & leftEV, View_AD1 & Lambda, View_AD2 & rightEV, 
        const AD & rhoux, const AD & rhouy, const AD & rho, const ScalarT & nx, const ScalarT & ny,
        const AD & a_sound, const ScalarT & gamma);

    /* @brief Computes the local eigenvalue decomposition for the stabilization and boundary terms.
     * This is the 3-D version.
     *
     * @param[inout] leftEV  Storage for the left eigenvectors
     * @param[inout] Lambda  Storage for the absolute value of the eigenvalues. A vector, not a matrix since it is diagonal.
     * @param[inout] rightEV  Storage for the right eigenvectors
     * @param[in] rhoux  x-component of the momentum
     * @param[in] rhouy  y-component of the momentum
     * @param[in] rhouz  y-component of the momentum
     * @param[in] rho  Density
     * @param[in] nx  x-component of the normal vector
     * @param[in] ny  y-component of the normal vector
     * @param[in] nz  z-component of the normal vector
     * @param[in] a_sound  Speed of sound
     * @param[in] gamma  Ratio of specific heats
     *
     * @details Should be called using the trace variables \f$\hat{S}\f$.
     */

    KOKKOS_FUNCTION void eigendecompFluxJacobian(View_AD2 & leftEV, View_AD1 & Lambda, View_AD2 & rightEV, 
        const AD & rhoux, const AD & rhouy, const AD & rhouz, const AD & rho, 
        const ScalarT & nx, const ScalarT & ny, const ScalarT & nz,
        const AD & a_sound, const ScalarT & gamma);

    /* @brief Computes the local normal flux Jacobian for the boundary term.
     * This is the 1-D version.
     *
     * @param[inout] dFdn  Storage for the normal flux Jacobian
     * @param[in] rhoux  x-component of the momentum
     * @param[in] rho  Density
     * @param[in] a_sound  Speed of sound
     * @param[in] gamma  Ratio of specific heats
     *
     * @details Should be called using the trace variables \f$\hat{S}\f$.
     */

    KOKKOS_FUNCTION void updateNormalFluxJacobian(View_AD2 & dFdn, const AD & rhoux,
        const AD & rho, const AD & a_sound, const ScalarT & gamma);

    /* @brief Computes the local normal flux Jacobian for the boundary term.
     * This is the 2-D version.
     *
     * @param[inout] dFdn  Storage for the normal flux Jacobian
     * @param[in] rhoux  x-component of the momentum
     * @param[in] rhouy  y-component of the momentum
     * @param[in] rho  Density
     * @param[in] nx  x-component of the normal vector
     * @param[in] ny  y-component of the normal vector
     * @param[in] a_sound  Speed of sound
     * @param[in] gamma  Ratio of specific heats
     *
     * @details Should be called using the trace variables \f$\hat{S}\f$.
     */

    KOKKOS_FUNCTION void updateNormalFluxJacobian(View_AD2 & dFdn, const AD & rhoux,
        const AD & rhouy, const AD & rho, const AD & nx, const AD & ny, 
        const AD & a_sound, const ScalarT & gamma);

    /* @brief Computes the local normal flux Jacobian for the boundary term.
     * This is the 3-D version.
     *
     * @param[inout] dFdn  Storage for the normal flux Jacobian
     * @param[in] rhoux  x-component of the momentum
     * @param[in] rhouy  y-component of the momentum
     * @param[in] rhouz  z-component of the momentum
     * @param[in] rho  Density
     * @param[in] nx  x-component of the normal vector
     * @param[in] ny  y-component of the normal vector
     * @param[in] nz  z-component of the normal vector
     * @param[in] a_sound  Speed of sound
     * @param[in] gamma  Ratio of specific heats
     *
     * @details Should be called using the trace variables \f$\hat{S}\f$.
     */

    KOKKOS_FUNCTION void updateNormalFluxJacobian(View_AD2 & dFdn, const AD & rhoux,
        const AD & rhouy, const AD & rhouz, const AD & rho, 
        const AD & nx, const AD & ny, const AD & nz,
        const AD & a_sound, const ScalarT & gamma);

    /* @brief Computes y = Ax
     *
     * @param[in] A  Matrix
     * @param[in] x  Vector
     * @param[out] y  Result
     *
     */

    KOKKOS_FUNCTION void matVec(const View_AD2 & A, const View_AD1 & x, View_AD1 & y);

// TODO making this public for now (testing brought this issue up...)
    View_AD4 fluxes_vol, fluxes_side; // Storage for the inviscid fluxes
    View_AD3 stab_bound_side; // Storage for the stabilization term/boundary term
    View_AD3 props_vol, props_side; // Storage for the thermodynamic properties

  private:

    int spaceDim;
    
    int rho_num, rhoux_num, rhouy_num, rhouz_num, rhoE_num;
    int auxrho_num, auxrhoux_num, auxrhouy_num, auxrhouz_num, auxrhoE_num;

    // indices to access the model params
    int cp_mp_num = 0;
    int gamma_mp_num = 1;
    int RGas_mp_num = 2;
    int URef_mp_num = 3;
    int LRef_mp_num = 4;
    int rhoRef_mp_num = 5;
    int TRef_mp_num = 6;
    int MRef_mp_num = 7;

    // indices to access the thermodynamic properties
    int p0_num = 0;
    int T_num = 1;
    int a_num = 2;

    bool maxEVstab,roestab; // Options for stabilization

    //View_AD4 fluxes_vol, fluxes_side; // Storage for the inviscid fluxes
    //View_AD3 stab_bound_side; // Storage for the stabilization term/boundary term
    //View_AD3 props_vol, props_side; // Storage for the thermodynamic properties

    Kokkos::View<ScalarT*,AssemblyDevice> modelparams;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::computeFlux() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> invFluxesFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::computeInviscidFluxes() - function evaluation");
    Teuchos::RCP<Teuchos::Time> invFluxesFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::computeInviscidFluxes() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> stabCompFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::computeStabilizationTerm() - function evaluation");
    Teuchos::RCP<Teuchos::Time> stabCompFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::computeStabilizationTerm() - evaluation of product");
    Teuchos::RCP<Teuchos::Time> boundCompFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::computeBoundaryTerm() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundCompFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::computeBoundaryTerm() - evaluation of product");
    Teuchos::RCP<Teuchos::Time> thermoPropFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::euler::computeThermoProps() - evaluation of thermodynamic properties");

  };
  
}

#endif
