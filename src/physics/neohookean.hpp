/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_NEOHOOKEAN_H
#define MRHYDE_NEOHOOKEAN_H

#include "physicsBase.hpp"
#include <string>

namespace MrHyDE {
  
  /**
   * \brief neohookean physics class.
   *
   * This class computes volumetric residuals for compressible Neo-Hookean hyperelasticity.
   * The stored energy density is:
   *   W = (mu/2)(I1 - 3) + (lambda/2)(ln J)^2 - mu*ln(J)
   * where I1 = tr(F^T F), J = det(F), F = I + grad(u).
   *
   * The first Piola-Kirchhoff stress is:
   *   P = mu*F + (lambda*ln(J) - mu)*F^{-T}
   *
   * The weak form is:
   *   integral P : grad(v) dOmega_0 = integral f . v dOmega_0 + integral t . v dGamma_0
   *
   * The following functions may be specified in the input.yaml file:
   *   - "source dz" is the body force in z.
   *   - "source dy" is the body force in y.
   *   - "source dx" is the body force in x.
   *   - "mu" is the shear modulus (second Lame parameter).
   *   - "lambda" is the first Lame parameter.
   */

  template<class EvalT>
  class neohookean : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    typedef Kokkos::View<EvalT****,ContLayout,AssemblyDevice> View_EvalT4;
    
    neohookean() {} ;
    
    ~neohookean() {} ;
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    neohookean(Teuchos::ParameterList & settings, const int & dimension);
    
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
    
    // ========================================================================================
    // Compute the first Piola-Kirchhoff stress P = mu*F + (lambda*ln(J) - mu)*F^{-T}
    // ========================================================================================
    
    void computeStress(Vista<EvalT> lambda, Vista<EvalT> mu, const bool & onside);
    
    // ========================================================================================
    // ========================================================================================
    
    void updateParameters(const vector<Teuchos::RCP<vector<EvalT> > > & params,
                          const vector<string> & paramnames);
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<string> getDerivedNames();
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<View_EvalT2> getDerivedValues();
    
  private:
    
    int spaceDim, dx_num, dy_num, dz_num;
    
    View_EvalT4 stress_vol, stress_side;
    
    Kokkos::View<ScalarT*,AssemblyDevice> modelparams;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::neohookean::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::neohookean::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::neohookean::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::neohookean::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::neohookean::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::neohookean::computeFlux() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> fillStress = Teuchos::TimeMonitor::getNewCounter("MrHyDE::neohookean::computeStress()");
    
  };
  
}

#endif
