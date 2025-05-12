/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_ELLIPTICPRIOR_H
#define MRHYDE_ELLIPTICPRIOR_H

#include "physicsBase.hpp"
#include "vista.hpp"

namespace MrHyDE {
  
  /**
   * \brief ellipticPrior physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "ellipticPrior source" is the ellipticPrior source.
   *   - "advection z" is the advection z.
   *   - "advection y" is the advection y.
   *   - "density" is the density.
   *   - "advection x" is the advection x.
   *   - "robin alpha" is the robin alpha.
   *   - "ellipticPrior diffusion" is the ellipticPrior diffusion.
   *   - "specific heat" is the specific heat.
   */

  template<class EvalT>
  class ellipticPrior : public PhysicsBase<EvalT> {
  public:
    
    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    using PhysicsBase<EvalT>::adjrhs;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    ellipticPrior() {} ;
    
    ~ellipticPrior() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    ellipticPrior(Teuchos::ParameterList & settings, const int & dimension_) ;
    
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
    
    /**
     * @brief Returns the integrands and their types (boundary/volume) for integrated quantities required
     * by the ellipticPrior module. Currently, this is only used for testing purposes. 
     *
     * @return integrandsNamesAndTypes  Integrands, names, and type (boundary/volume) (matrix of strings).
     */
    
    std::vector< std::vector<string> > setupIntegratedQuantities(const int & spaceDim);


  private:
    
    int T_num = -1, ux_num = -1, uy_num = -1, uz_num = -1;
    int T_basis_num = -1;
    int auxT_num = -1;
    int IQ_start;
    
    //View_AD2 e_vol, dedt_vol, dedx_vol, dedy_vol, dedz_vol;
    //View_AD2 e_side, dedx_side, dedy_side, dedz_side;
    //View_AD2 e_face, dedx_face, dedy_face, dedz_face;
    //View_AD2 ux_vol, uy_vol, uz_vol;
    
    bool have_nsvel, test_IQs, have_advection;
    ScalarT formparam;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ellipticPrior::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ellipticPrior::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ellipticPrior::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ellipticPrior::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ellipticPrior::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ellipticPrior::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
