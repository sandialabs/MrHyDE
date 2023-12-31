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

#ifndef LLAMAS_H
#define LLAMAS_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  class llamas : public physicsbase {
  public:
    
    llamas() {} ;
    
    ~llamas() {};
    
    // ========================================================================================
    // ========================================================================================
    
    llamas(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_) {
      
      label = "llamas";
      
      isaux = isaux_;
      if (isaux) {
        prefix = "aux ";
      }
      
      myvars.push_back("llama");
      mybasistypes.push_back("HGRAD");
      
    }

    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_) {
      
      functionManager = functionManager_;
      
      // Functions
      functionManager->addFunction("sourceterm",fs.get<string>("source","0.0"),"ip");
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
      
      auto llama = wkset->getData("llama");
      auto dllama_dx = wkset->getData("grad(llama)[x]");
      auto dllama_dy = wkset->getData("grad(llama)[y]");
      
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

#endif
