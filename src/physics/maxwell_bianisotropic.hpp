/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_maxwell_bianisotropic_H
#define MRHYDE_maxwell_bianisotropic_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief maxwell_bianisotropic physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "current z" is the current z.
   *   - "current y" is the current y.
   *   - "permeability" is the permeability.
   *   - "refractive index" is the refractive index.
   *   - "current x" is the current x.
   *   - "conductivity" is the conductivity.
   *   - "permittivity" is the permittivity.
   */

  template<class EvalT>
  class maxwell_bianisotropic : public PhysicsBase<EvalT> {
  public:
    
    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    maxwell_bianisotropic() {} ;
    
    ~maxwell_bianisotropic() {};
    
    maxwell_bianisotropic(Teuchos::ParameterList & settings, const int & dimension_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT> > & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
        
    void boundaryResidual();
    
    // ========================================================================================
    // ========================================================================================
    
    //void setVars(std::vector<string> & varlist_);
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);

  private:
    
    int Enum, Bnum, spaceDim;
    bool useLeapFrog, include_Beqn, include_Eeqn;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_bianisotropic::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_bianisotropic::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_bianisotropic::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_bianisotropic::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_bianisotropic::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_bianisotropic::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
