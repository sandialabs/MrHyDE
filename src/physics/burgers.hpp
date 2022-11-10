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

#ifndef MRHDYE_BURGERS_H
#define MRHDYE_BURGERS_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief burgers physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   *   - "diffusion" is the diffusion.
   *   - "zvel" is the zvel.
   *   - "Burgers source" is the Burgers source.
   *   - "yvel" is the yvel.
   *   - "xvel" is the xvel.
   */
  class Burgers : public physicsbase {
  public:
    
    Burgers() {} ;
    
    ~Burgers() {};
    
    // ========================================================================================
    // ========================================================================================
    
    Burgers(Teuchos::ParameterList & settings, const int & dimension_);
    
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
