/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/** @file incompressibleSaturation.hpp
 *
 * @brief Shallow water physics module, hybridized version
 *
 * Solves the shallow water equations with a hybridized formulation.
 * See Samii (J. Sci. Comp. 2019). 
 */

#ifndef MRHYDE_INCOMPRESSIBLESATURATION_H
#define MRHYDE_INCOMPRESSIBLESATURATION_H

#include "physicsBase.hpp"
#include "wells.hpp"

namespace MrHyDE {
  
  /** 
   * \brief Two-phase, incompressible saturation equation module. 
   *
   * Solves the two-phase, incompressible saturation equation 
   * for the water phase.
   * 
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "uy" is the uy.
   *   - "f_w" is the f_w.
   *   - "uz" is the uz.
   *   - "ux" is the ux.
   *   - "source_S" is the source_S.
   */

  template<class EvalT>
  class incompressibleSaturation : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    typedef Kokkos::View<EvalT****,ContLayout,AssemblyDevice> View_EvalT4;
    
    incompressibleSaturation() {};
    
    ~incompressibleSaturation() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    incompressibleSaturation(Teuchos::ParameterList & settings, const int & dimension);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT>> & functionManager_);
    
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

    /* @brief Update the fluxes for the residual calculation.
     *
     */

    void computeFluxVector();

  private:

    int spaceDim;
    
    int S_num;

    View_EvalT4 fluxes_vol; // Storage for the fluxes

    wells<EvalT> myWells;
    bool useWells;

    ScalarT phi; // porosity

    Kokkos::View<ScalarT*,AssemblyDevice> modelparams;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::computeFlux() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> fluxVectorFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::computeFluxVector() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxVectorFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::incompressibleSaturation::computeFluxVector() - evaluation of flux");

  };
  
}

#endif
