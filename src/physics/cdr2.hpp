/***********************************************************************
MrHyDE - a framework for solving Multi-resolution Hybridized
Differential Equations and enabling beyond forward simulation for 
large-scale multiphysics and multiscale systems.

Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_CDR2_H
#define MRHYDE_CDR2_H

#include "physicsBase.hpp"

namespace MrHyDE {

/**
* \brief cdr2 physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "diffusion" is the diffusion.
   *   - "SUPG tau" is the SUPG tau.
   *   - "source" is the source.
   *   - "density" is the density.
   *   - "zvel" is the zvel.
   *   - "robin alpha" is the robin alpha.
   *   - "xvel" is the xvel.
   *   - "yvel" is the yvel.
   *   - "reaction" is the reaction.
   *   - "specific heat" is the specific heat.
   */

  template<class EvalT>
  class cdr2 : public PhysicsBase<EvalT> {
  public:
     
    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
      
    cdr2() {} ;
    
    ~cdr2() {};
    
    // ========================================================================================
    // ========================================================================================
    
    cdr2(Teuchos::ParameterList & settings, const int & dimension_);
    
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
    // ========================================================================================
    
    void edgeResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);
    //void setVars(vector<string> & varlist_);
    
    // ========================================================================================
    // return the value of the stabilization parameter 
    // ========================================================================================
    
    template<class T>  
    T computeTau(const T & localdiff, const T & xvl, const T & yvl, const T & zvl, const ScalarT & h) const;
    
  private:
    
    int cnum;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cdr2::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cdr2::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cdr2::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cdr2::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cdr2::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::cdr2::computeFlux() - evaluation of flux");
    
    
  };
  
}

#endif
