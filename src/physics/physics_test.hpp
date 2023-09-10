/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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

  template<class EvalT>
  class physicsTest : public PhysicsBase<EvalT> {
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
    
    physicsTest() {} ;
    
    ~physicsTest() {};
    
    /**
     * Constructor. Sets the discetization and operator according to the options provided in \p settings
     */
    physicsTest(Teuchos::ParameterList & settings, const int & dimension_);
    
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
    // ========================================================================================
    
    void updatePerm(View_EvalT2 perm);
    
    
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
