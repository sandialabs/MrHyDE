/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_MSPHASEFIELD_H
#define MRHYDE_MSPHASEFIELD_H

#include "physicsBase.hpp"
#include <random>
#include <math.h>
#include <time.h>

namespace MrHyDE {
  
  /**
   * \brief msphasefield physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   */

  template<class EvalT>
  class msphasefield : public PhysicsBase<EvalT> {
  public:
    
    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    msphasefield() {} ;
    
    ~msphasefield() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    msphasefield(Teuchos::ParameterList & settings, const int & dimension_,
                 const Teuchos::RCP<MpiComm> & Comm_);
    
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

    // ========================================================================================
    /* return the source term (to be multiplied by test_function) */
    // ========================================================================================
    
    EvalT SourceTerm(const ScalarT & x, const ScalarT & y, const ScalarT & z,
                  const std::vector<EvalT> & tsource) const;
    
    // ========================================================================================
    /* return the source term (to be multiplied by test_function) */
    // ========================================================================================
    
    ScalarT boundarySource(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                           const string & side) const;
    
    // ========================================================================================
    /* return the diffusivity coefficient */
    // ========================================================================================
    
    EvalT DiffusionCoeff(const ScalarT & x, const ScalarT & y, const ScalarT & z) const;
    
    // ========================================================================================
    /* return the source term (to be multiplied by test_function) */
    // ========================================================================================
    
    ScalarT robinAlpha(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                       const string & side) const;
    
    // ========================================================================================
    // TMW: this is deprecated
    // ========================================================================================
    
    void updateParameters(const vector<Teuchos::RCP<vector<EvalT> > > & params,
                          const vector<string> & paramnames);
    
    // ========================================================================================
    // ========================================================================================
    
  private:
    
    Teuchos::RCP<MpiComm> Comm;      
    std::vector<EvalT> diff_FAD, L, A;   
    int spaceDim, numphases, numdisks;
    vector<string> varlist;
    std::vector<int> phi_num;
    ScalarT disksize;
    ScalarT xmax, xmin, ymax, ymin;
    bool uniform, variableMobility;
    std::vector<ScalarT> disk;
    std::string initialType;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::msphasefield::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::msphasefield::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::msphasefield::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::msphasefield::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::msphasefield::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::msphasefield::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
