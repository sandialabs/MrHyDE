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

#ifndef MRHYDE_HELMHOLTZ_H
#define MRHYDE_HELMHOLTZ_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief helmholtz physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "c2i_x" is the c2i_x.
   *   - "c2r_x" is the c2r_x.
   *   - "c2r_z" is the c2r_z.
   *   - "omegai" is the omegai.
   *   - "alphaHr" is the alphaHr.
   *   - "c2i_z" is the c2i_z.
   *   - "source_i" is the source_i.
   *   - "alphaTr" is the alphaTr.
   *   - "omega2r" is the omega2r.
   *   - "c2i_y" is the c2i_y.
   *   - "robin_alpha_i" is the robin_alpha_i.
   *   - "alphaTi" is the alphaTi.
   *   - "source_r" is the source_r.
   *   - "omegar" is the omegar.
   *   - "source_i_side" is the source_i_side.
   *   - "c2r_y" is the c2r_y.
   *   - "robin_alpha_r" is the robin_alpha_r.
   *   - "source_r_side" is the source_r_side.
   *   - "alphaHi" is the alphaHi.
   *   - "freqExp" is the freqExp.
   *   - "omega2i" is the omega2i.
   */

  template<class EvalT>
  class helmholtz : public PhysicsBase<EvalT> {
  public:
    
    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    helmholtz() {} ;
    
    ~helmholtz() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    helmholtz(Teuchos::ParameterList & settings, const int & dimension_);
    
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
    
    
    void edgeResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);
    
  private:
    
    int ur_num, ui_num;
    
    //AD ur, durdx, durdy, durdz, durdn, c2durdn;
    //AD ui, duidx, duidy, duidz, duidn, c2duidn;
    //ScalarT vr, dvrdx, dvrdy, dvrdz;
    //ScalarT vi, dvidx, dvidy, dvidz;
    
    bool fractional;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::helmholtz::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::helmholtz::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::helmholtz::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::helmholtz::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::helmholtz::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::helmholtz::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
