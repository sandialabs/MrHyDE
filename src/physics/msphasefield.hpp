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
  class msphasefield : public physicsbase {
  public:
    
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
                         Teuchos::RCP<FunctionManager> & functionManager_);
    
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
    
    void setWorkset(Teuchos::RCP<Workset> & wkset_);

    // ========================================================================================
    /* return the source term (to be multiplied by test_function) */
    // ========================================================================================
    
    AD SourceTerm(const ScalarT & x, const ScalarT & y, const ScalarT & z,
                  const std::vector<AD > & tsource) const;
    
    // ========================================================================================
    /* return the source term (to be multiplied by test_function) */
    // ========================================================================================
    
    ScalarT boundarySource(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                           const string & side) const;
    
    // ========================================================================================
    /* return the diffusivity coefficient */
    // ========================================================================================
    
    AD DiffusionCoeff(const ScalarT & x, const ScalarT & y, const ScalarT & z) const;
    
    // ========================================================================================
    /* return the source term (to be multiplied by test_function) */
    // ========================================================================================
    
    ScalarT robinAlpha(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & t,
                       const string & side) const;
    
    // ========================================================================================
    // TMW: this is deprecated
    // ========================================================================================
    
    void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                          const vector<string> & paramnames);
    
    // ========================================================================================
    // ========================================================================================
    
  private:
    
    Teuchos::RCP<MpiComm> Comm;      
    std::vector<AD> diff_FAD, L, A;   
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
