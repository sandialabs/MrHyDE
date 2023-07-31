/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_LINEARELAST_H
#define MRHYDE_LINEARELAST_H

#include "physicsBase.hpp"
#include "CrystalElasticity.hpp"
#include <string>

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
  class linearelasticity : public physicsbase {
  public:
    
    linearelasticity() {} ;
    
    ~linearelasticity() {} ;
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    linearelasticity(Teuchos::ParameterList & settings, const int & dimension);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_);
    
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
    
    void setWorkset(Teuchos::RCP<Workset<AD> > & wkset_);
    
    // ========================================================================================
    // return the stress
    // ========================================================================================
    
    void computeStress(Vista lambda, Vista mu, const bool & onside);
        
    // ========================================================================================
    // TMW: needs to be deprecated
    // ========================================================================================
    
    void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                          const vector<string> & paramnames);
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<string> getDerivedNames();
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<View_AD2> getDerivedValues();
    
  private:
    
    int spaceDim, dx_num, dy_num, dz_num, e_num, p_num;
    int auxdx_num = -1, auxdy_num = -1, auxdz_num = -1, auxe_num = -1, auxp_num = -1;
    
    View_AD4 stress_vol, stress_side;
    
    bool useLame, addBiot, useCE, incplanestress;
    //ScalarT formparam, biot_alpha, e_ref, alpha_T, epen;
    Kokkos::View<ScalarT*,AssemblyDevice> modelparams;
    
    Teuchos::RCP<CrystalElastic> crystalelast;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elasticity::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elasticity::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elasticity::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elasticity::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elasticity::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elasticity::computeFlux() - evaluation of flux");
    Teuchos::RCP<Teuchos::Time> setLocalSol = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elasticity::setLocalSoln()");
    Teuchos::RCP<Teuchos::Time> fillStress = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elasticity::computeStress()");
    Teuchos::RCP<Teuchos::Time> computeBasis = Teuchos::TimeMonitor::getNewCounter("MrHyDE::elasticity::computeBasisVec()");
    
  };
  
}

#endif
