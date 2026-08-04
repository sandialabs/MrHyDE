/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_ELASTODYN_H
#define MRHYDE_ELASTODYN_H

#include "physicsBase.hpp"
#include "CrystalElasticity.hpp"

namespace MrHyDE {
  
  /**
   * \brief linearelasticity physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "source dz" is the source dz.
   *   - "source dy" is the source dy.
   *   - "source dx" is the source dx.
   *   - "mu" is the mu.
   *   - "lambda" is the lambda.
   */

  template<class EvalT>
  class elastodynamics : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    typedef Kokkos::View<EvalT****,ContLayout,AssemblyDevice> View_EvalT4;
    
    elastodynamics() {} ;
    
    ~elastodynamics() {} ;
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    elastodynamics(Teuchos::ParameterList & settings, const int & dimension);
    
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
    
    //void setLocalSoln(const size_t & e, const size_t & ipindex, const bool & onside);
    
    // ========================================================================================
    // ========================================================================================
    
    //void setVars(std::vector<string> & varlist_);
    
    // ========================================================================================
    // ========================================================================================
    
    //void setAuxVars(std::vector<string> & auxvarlist);
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);
    
    // ========================================================================================
    // return the stress
    // ========================================================================================
    
    void computeStress(Vista<EvalT> lambda, Vista<EvalT> mu, const bool & onside);
        
    // ========================================================================================
    // TMW: needs to be deprecated
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
    
    int spaceDim, dx_num, dy_num, dz_num, e_num, p_num, vx_num, vy_num, vz_num;
    int auxdx_num = -1, auxdy_num = -1, auxdz_num = -1, auxe_num = -1, auxp_num = -1;
    
    View_EvalT4 stress_vol, stress_side;
    
    bool useLame, addBiot, useCE, incplanestress;
    //ScalarT formparam, biot_alpha, e_ref, alpha_T, epen;
    Kokkos::View<ScalarT*,AssemblyDevice> modelparams;
    
    Teuchos::RCP<CrystalElastic<EvalT> > crystalelast;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elastodynamics::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elastodynamics::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elastodynamics::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elastodynamics::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elastodynamics::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elastodynamics::computeFlux() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> setLocalSol = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elastodynamics::setLocalSoln()");
    Teuchos::RCP<Teuchos::Time> fillStress = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elastodynamics::computeStress()");
    Teuchos::RCP<Teuchos::Time> computeBasis = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elastodynamics::computeBasisVec()");
    
  };
  
}

#endif
