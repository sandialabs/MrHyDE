/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_KURAMOTO_SIVASHINSKY_H
#define MRHYDE_KURAMOTO_SIVASHINSKY_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief kuramotoSivashinsky physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   */
  // This class solves the Kuramoto-Sivashinsky equation in multiple dimensions:
  //   u_t + \Delta u + \Delta^{2} u + {\frac {1}{2}}|\nabla u|^{2} = 0.
  // It reformulates the problem as:
  //   u_t + w + \Delta w + {\frac {1}{2}|\nabla u|^{2}} = 0,
  //   \Delta u - w = 0.
  // Then solves the mixed system assuming periodic boundary conditions on u and 
  // no boundary conditions on w.

  template<class EvalT>
  class KuramotoSivashinsky : public PhysicsBase<EvalT> {
  public:
    
    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    KuramotoSivashinsky() {} ;
    
    ~KuramotoSivashinsky() {};
    
    // ========================================================================================
    // ========================================================================================
    
    KuramotoSivashinsky(Teuchos::ParameterList & settings, const int & dimension_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT> > & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);

  private:

    int u_num, w_num;
    
  };
  
}

#endif
