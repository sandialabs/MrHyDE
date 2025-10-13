/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/** @file vlasov_fokker_planck.hpp
 *
 * @brief VFP 0d2v physics module
 *
 */

#ifndef MRHYDE_VFP0d2v_H
#define MRHYDE_VFP0d2v_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /** 
   * \brief Vlasov-Fokker-Planck physics module
   *
   */
  
  template<class EvalT>
  class VFP0d2v : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT*,ContLayout,AssemblyDevice> View_EvalT1;
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    typedef Kokkos::View<EvalT***,ContLayout,AssemblyDevice> View_EvalT3;
    typedef Kokkos::View<EvalT****,ContLayout,AssemblyDevice> View_EvalT4;
    

    VFP0d2v() {} ;
    
    ~VFP0d2v() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    VFP0d2v(Teuchos::ParameterList & settings, const int & dimension);
    
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
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);

    /* @brief Update the inviscid fluxes for the residual calculation.
     *
     * @param[in] on_side  Bool indicating if we are on an element side or not
     *
     * @details When we are at an interface, the flux is evaluated using the trace variables.
     * This should be called after updating the thermodynamic properties.
     */

    std::vector< std::vector<string> > setupIntegratedQuantities(const int & spaceDim);
    
  private:

    int spaceDim, velDim, H_num, C_num, G_num, E_num;
    ScalarT m_C, m_H, m_G, m_E, Z_C, Z_H, Z_G, Z_E, gamma_h;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VFP0d2v::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::VFP0d2v::volumeResidual() - evaluation of residual");
    
  };
  
}

#endif
