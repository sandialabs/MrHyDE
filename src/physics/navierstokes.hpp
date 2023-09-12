/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_NAVIERSTOKES_H
#define MRHYDE_NAVIERSTOKES_H

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
  class navierstokes : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    navierstokes() {} ;
    
    ~navierstokes() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    navierstokes(Teuchos::ParameterList & settings, const int & dimension_);
    
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

    KOKKOS_FUNCTION EvalT computeTau(const EvalT & localdiff, const EvalT & xvl, const EvalT & yvl, const EvalT & zvl, const ScalarT & h, const int & spaceDim, const ScalarT & dt, const bool & isTransient) const;
    
  private:
    
    int ux_num, uy_num, uz_num, pr_num, e_num;
    
    bool useSUPG, usePSPG;

    vector<ScalarT> pik;
    bool have_energy;
    ScalarT T_ambient, beta;
    Kokkos::View<ScalarT*,AssemblyDevice> model_params;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::navierstokes::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
