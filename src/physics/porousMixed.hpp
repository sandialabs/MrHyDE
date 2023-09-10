/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_POROUSMIXED_H
#define MRHYDE_POROUSMIXED_H

#include "physicsBase.hpp"
#include "klexpansion.hpp"
#include "wells.hpp"

namespace MrHyDE {
  
  /**
   * \brief porousMixed physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "Kinv_yy" is the Kinv_yy.
   *   - "source" is the source.
   *   - "Kinv_xx" is the Kinv_xx.
   *   - "Kinv_zz" is the Kinv_zz.
   *   - "total_mobility" is the total_mobility.
   */

  template<class EvalT>
  class porousMixed : public PhysicsBase<EvalT> {
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
    
    porousMixed() {} ;
    
    ~porousMixed() {};
    
    porousMixed(Teuchos::ParameterList & settings, const int & dimension_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT>> & functionManager_);
    
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

    //void setVars(std::vector<string> & varlist_);
    
    // ========================================================================================
    // ========================================================================================
    
    //void setAuxVars(std::vector<string> & auxvarlist);
    
    // ========================================================================================
    // ========================================================================================
    
    void updatePerm(View_EvalT2 Kinv_xx, View_EvalT2 Kinv_yy, View_EvalT2 Kinv_zz);
    
    void updateKLPerm(View_EvalT2 Kinv_xx, View_EvalT2 Kinv_yy, View_EvalT2 Kinv_zz);
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<string> getDerivedNames();
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<View_EvalT2> getDerivedValues();
    
  private:
    
    int pnum=-1, unum=-1, auxpnum=-1, auxunum=-1;
    int dxnum,dynum,dznum;
    bool usePermData, useWells, useKL;
    string auxvar;
    klexpansion permKLx, permKLy, permKLz;
    Kokkos::View<size_t**,AssemblyDevice> KLindices;
    wells<EvalT> myWells;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousMixed::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousMixed::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousMixed::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousMixed::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousMixed::computeFlux() - evaluation of interface flux");
    
  };
  
}

#endif
