/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_MHD_H
#define MRHYDE_MHD_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief navierstokes physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "source ux" is the source ux.
   *   - "density" is the density.
   *   - "viscosity" is the viscosity.
   *   - "source uz" is the source uz.
   *   - "source pr" is the source pr.
   *   - "source uy" is the source uy.
   */

  template<class EvalT>
  class MHD : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    MHD() {} ;
    
    ~MHD() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    MHD(Teuchos::ParameterList & settings, const int & dimension_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT> > & functionManager_);
    
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
    
    //void setVars(std::vector<string> & varlist_);
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);
    
    // ========================================================================================
    // return the value of the stabilization parameter 
    // ========================================================================================
    
    /* @brief Returns the value of the stabilization parameter (SUPG/PSPG)
     *
     * @param[in] localdiff  Kinematic viscosity
     * @param[in] xvl  x-component of the velocity
     * @param[in] yvl  y-component of the velocity
     * @param[in] zvl  z-component of the velocity
     * @param[in] h  Element diameter
     * @param[in] spaceDim  Number of spatial dimensions
     * @param[in] dt  Timestep
     * @param[in] isTransient  Bool indicating if the simulation is transient

     * @return SUPG/PSPG stabilization parameter (type AD)
     *
     */

    KOKKOS_FUNCTION EvalT computeTauU(const EvalT & xvl, const EvalT & yvl, const EvalT & zvl,
                                      const EvalT & Bx, const EvalT & By, const EvalT & Bz, const EvalT & mu, const EvalT & ndens,
                                      const ScalarT & h, const ScalarT & dt) const;
    
    KOKKOS_FUNCTION EvalT computeTauT(const EvalT & xvl, const EvalT & yvl, const EvalT & zvl,
                                      const EvalT & Bx, const EvalT & By, const EvalT & Bz, const EvalT & rho, const EvalT & ndens,
                                      const EvalT & kappa, const ScalarT & h, const ScalarT & dt) const;
    
    KOKKOS_FUNCTION EvalT computeTauB(const EvalT & xvl, const EvalT & yvl, const EvalT & zvl,
                                      const EvalT & Bx, const EvalT & By, const EvalT & Bz, const EvalT & mu0, const EvalT & ndens,
                                      const ScalarT & h, const ScalarT & dt) const;
    
    KOKKOS_FUNCTION EvalT computeTauP(const EvalT & tauu, const ScalarT & h) const;
    
    KOKKOS_FUNCTION EvalT computeTauPsi(const EvalT & tauB, const ScalarT & h) const;
    
    KOKKOS_FUNCTION EvalT computeStrongResidualRho(const EvalT & drho_dt, const EvalT & drhoux_dx, const EvalT & drhouy_dy,
                                                   const EvalT & drhouz_dz) const;
    
    KOKKOS_FUNCTION EvalT computeStrongResidualRhoux(const EvalT & drhoux_dt, const EvalT & rhoux, const EvalT & drhoux_dx,
                                                     const EvalT & drhoux_dy, const EvalT & drhoux_dz,
                                                     const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                     const EvalT & dux_dx, const EvalT & duy_dy, const EvalT & duz_dz,
                                                     const EvalT & ndens, const EvalT & dT_dx);
    
    KOKKOS_FUNCTION EvalT computeStrongResidualRhouy(const EvalT & drhouy_dt, const EvalT & rhouy, const EvalT & drhouy_dx,
                                                     const EvalT & drhouy_dy, const EvalT & drhouy_dz,
                                                     const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                     const EvalT & dux_dx, const EvalT & duy_dy, const EvalT & duz_dz,
                                                     const EvalT & ndens, const EvalT & dT_dy);
    
    KOKKOS_FUNCTION EvalT computeStrongResidualRhouz(const EvalT & drhouz_dt, const EvalT & rhouz, const EvalT & drhouz_dx,
                                                     const EvalT & drhouz_dy, const EvalT & drhouz_dz,
                                                     const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                     const EvalT & dux_dx, const EvalT & duy_dy, const EvalT & duz_dz,
                                                     const EvalT & ndens, const EvalT & dT_dz);
    
    KOKKOS_FUNCTION EvalT computeStrongResidualT(const EvalT & dT_dt, const EvalT & T, const EvalT & dT_dx,
                                                 const EvalT & dT_dy, const EvalT & dT_dz,
                                                 const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                 const EvalT & dux_dx, const EvalT & dux_dy, const EvalT & dux_dz,
                                                 const EvalT & duy_dx, const EvalT & duy_dy, const EvalT & duy_dz,
                                                 const EvalT & duz_dx, const EvalT & duz_dy, const EvalT & duz_dz,
                                                 const EvalT & qx, const EvalT & qy, const EvalT & qz,
                                                 const EvalT & jx, const EvalT & jy, const EvalT & jz,
                                                 const EvalT & pi_xx, const EvalT & pi_xy, const EvalT & pi_xz,
                                                 const EvalT & pi_yx, const EvalT & pi_yy, const EvalT & pi_yz,
                                                 const EvalT & pi_zx, const EvalT & pi_zy, const EvalT & pi_zz,
                                                 const EvalT & ndens, const EvalT & gamma_bar, const EvalT & Re,
                                                 const EvalT & S);
    
    KOKKOS_FUNCTION EvalT computeStrongResidualBx(const EvalT & dBx_dt, const EvalT & Bx, const EvalT & dBx_dx,
                                                  const EvalT & dBx_dy, const EvalT & dBx_dz,
                                                  const EvalT & By, const EvalT & Bz,
                                                  const EvalT & dBy_dy, const EvalT & dBz_dz,
                                                  const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                  const EvalT & dux_dx, const EvalT & dux_dy, const EvalT & dux_dz,
                                                  const EvalT & duy_dy, const EvalT & duz_dz,
                                                  const EvalT & S, const EvalT & dpsi_dx);
    
    KOKKOS_FUNCTION EvalT computeStrongResidualBy(const EvalT & dBy_dt, const EvalT & By, const EvalT & dBy_dx,
                                                  const EvalT & dBy_dy, const EvalT & dBy_dz,
                                                  const EvalT & Bx, const EvalT & Bz,
                                                  const EvalT & dBx_dx, const EvalT & dBz_dz,
                                                  const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                  const EvalT & dux_dx,
                                                  const EvalT & duy_dx, const EvalT & duy_dy, const EvalT & duy_dz,
                                                  const EvalT & duz_dz,
                                                  const EvalT & S, const EvalT & dpsi_dy);
    
    KOKKOS_FUNCTION EvalT computeStrongResidualBz(const EvalT & dBz_dt, const EvalT & Bz, const EvalT & dBz_dx,
                                                  const EvalT & dBz_dy, const EvalT & dBz_dz,
                                                  const EvalT & Bx, const EvalT & By,
                                                  const EvalT & dBx_dx, const EvalT & dBy_dy,
                                                  const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                  const EvalT & dux_dx, const EvalT & duy_dy,
                                                  const EvalT & duz_dx, const EvalT & duz_dy, const EvalT & duz_dz,
                                                  const EvalT & S, const EvalT & dpsi_dz);
    
    KOKKOS_FUNCTION EvalT computeStrongResidualPsi(const EvalT & dBx_dx, const EvalT & dBy_dy, const EvalT & dBz_dz);
    
  private:
    
    int rhoux_num, rhouy_num, rhouz_num, rho_num, T_num, Bx_num, By_num, Bz_num, psi_num;
    
    bool use_stabilization;

    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
