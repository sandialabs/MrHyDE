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

#ifndef KURAMOTO_SIVASHINSKY_H
#define KURAMOTO_SIVASHINSKY_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  class KuramotoSivashinsky : public physicsbase {
  public:
    
    KuramotoSivashinsky() {} ;
    
    ~KuramotoSivashinsky() {};
    
    // ========================================================================================
    // ========================================================================================
    
    KuramotoSivashinsky(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
    
  };
  
}

#endif
