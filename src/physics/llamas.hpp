/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

// NOTE: This module is just for demonstration purposes.  No llamas were harmed in the development.

#ifndef MRHYDE_LLAMAS_H
#define MRHYDE_LLAMAS_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  template<class EvalT>
  class llamas : public PhysicsBase<EvalT> {
  public:

    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    
    llamas() {} ;
    
    ~llamas() {};
    
    // ========================================================================================
    // ========================================================================================
    
    llamas(Teuchos::ParameterList & settings, const int & dimension_) {
      
      label = "llamas";
      
      myvars.push_back("llama");
      mybasistypes.push_back("HGRAD");
      
    }

    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
      
      functionManager = functionManager_;
      
      // Functions
      functionManager->addFunction("sourceterm",fs.get<string>("whatever","0.0"),"ip");
      functionManager->addFunction("cterm",fs.get<string>("c","0.0"),"ip");
      
    }
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual() {
      
      // Evaluate any functions you might need
      auto source = functionManager->evaluate("sourceterm","ip");
      auto c = functionManager->evaluate("cterm","ip");
      
      // Grad some arrays from the workset
      // Rememeber that these are Views and act like pointers (no deep copies of data)
      
      auto llama = wkset->getSolutionField("llama");
      auto dllama_dx = wkset->getSolutionField("grad(llama)[x]");
      auto dllama_dy = wkset->getSolutionField("grad(llama)[y]");
      
      // These basis array are 4-dimensional Views with dimensions numElem x numDOF x numip x dimension
      // In this examples, they will be 100x4x4x2
      
      auto basis = wkset->getBasis("llama");
      auto gradbasis = wkset->getBasisGrad("llama");
      
      auto res = wkset->getResidual();
      auto offsets = wkset->getOffsets("llama");
      auto wts = wkset->getWeights();
      
      // Contributes (grad(llama),grad v) + (cllama, v) - (source,v)
      
      for (int elem=0; elem<wkset->numElem; ++elem) {
        for (size_type dof=0; dof<basis.extent(1); ++dof ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            // Multiply the basis and the integration weights
            ScalarT v = basis(elem,dof,pt,0)*wts(elem,pt);
            ScalarT dv_dx = gradbasis(elem,dof,pt,0)*wts(elem,pt);
            ScalarT dv_dy = gradbasis(elem,dof,pt,1)*wts(elem,pt);
            res(elem,offsets(dof)) += dllama_dx(elem,pt)*dv_dx + dllama_dy(elem,pt)*dv_dy + c(elem,pt)*llama(elem,pt)*v - source(elem,pt)*v;
          }
        }
      }
      
    }
    
  };
  
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::llamas<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::llamas<AD>;

// Standard built-in types
template class MrHyDE::llamas<AD2>;
template class MrHyDE::llamas<AD4>;
template class MrHyDE::llamas<AD8>;
template class MrHyDE::llamas<AD16>;
template class MrHyDE::llamas<AD18>; // AquiEEP_merge
template class MrHyDE::llamas<AD24>;
template class MrHyDE::llamas<AD32>;
#endif

#endif
