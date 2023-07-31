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

#ifndef MRHYDE_PHYSICSTEST_H
#define MRHYDE_PHYSICSTEST_H

#include "physicsBase.hpp"


namespace MrHyDE {
  
  /**
   * \brief physicsTest class. (for testing only)
   * 
   * This class procedurally constructs an L^2 projection or Laplacian problem
   * based on inputs in the parameter list, and then dumps basis values and matrix entries
   * to standard output. 
   * 
   * This class currently supports the following configurations:
   *   - HGRAD discretization with projection, Laplace operators
   *   - HDIV discretization with projection operator
   *   - HDIV_AC discretization with projection operator
   *   - HCURL discretization with projection operator
   * 
   * This class is meant to be used only as a unit tester for discretizations, 
   * quadratures, and operators on a single core (outputs currently have a race condition),
   * and should not be run on GPU configurations.
   */
  class physicsTest : public physicsbase {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    physicsTest() {} ;
    
    ~physicsTest() {};
    
    /**
     * Constructor. Sets the discetization and operator according to the options provided in \p settings
     */
    physicsTest(Teuchos::ParameterList & settings, const int & dimension_);
    
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
    
    void setWorkset(Teuchos::RCP<Workset<AD> > & wkset_);
    
    // ========================================================================================
    // ========================================================================================
    
    void updatePerm(View_AD2 perm);
    
    
  private:
    
    int pnum;
    vector<string> myoperators;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::physicsTest::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::physicsTest::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::physicsTest::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::physicsTest::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::physicsTest::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::physicsTest::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
