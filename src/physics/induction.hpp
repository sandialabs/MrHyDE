/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_INDUCTION_H
#define MRHYDE_INDUCTION_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  template<class EvalT>
  class induction : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    induction() {} ;
    
    ~induction() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    induction(Teuchos::ParameterList & settings, const int & dimension_);
    
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
    
    
    KOKKOS_FUNCTION EvalT computeTauB(const EvalT & xvl, const EvalT & yvl, const EvalT & zvl,
                                      const EvalT & Bx, const EvalT & By, const EvalT & Bz, const EvalT & mu0, const EvalT & ndens,
                                      const ScalarT & h, const ScalarT & dt) const;
    
    KOKKOS_FUNCTION EvalT computeTauPsi(const EvalT & tauB, const ScalarT & h) const;
    
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
    
    int Bx_num, By_num, Bz_num, psi_num;
    
    bool use_stabilization, include_resistive, include_Hall;

    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::MHD::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
