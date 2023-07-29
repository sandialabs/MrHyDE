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
   * \brief Burgers' physics class.
   *
   * This class computes volumetric residuals for the physics described by the following strong form:
   * \f{eqnarray*}
   *   \frac{du}{dt}
   *   +
   *   \nabla \cdot \left(\frac{1}{2}\vec{\nu} u^2 - \epsilon(u) \nabla u \right)
   *   =
   *   f
   * \f}
   * Where the unknown \f$u\f$ is the quantity being solved for, \f$\nu\f$ is the advection term,
   * and \f$\epsilon\f$ is an entropy viscosity term.
   * This is also known as a entropy viscosity formulation with SUPG stabilization.
   * The following functions may be specified in the input.yaml file:
   *   - "diffusion" is the diffusion coefficient \f$\epsilon\f$.
   *   - "Burgers source" is the Burgers source term \f$f\f$.
   *   - "xvel" is the x-component of \f$\nu\f$.
   *   - "yvel" is the y-component of \f$\nu\f$.
   *   - "zvel" is the z-component of \f$\nu\f$.
   *   - "C1" is the entropy viscosity numerator.
   *   - "C2" is the entropy viscosity denominator.
   *   - "supg C" is a supg coefficient.
   *   - "supg C1" is a supg coefficient.
   *   - "supg C2" is a supg coefficient.
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
    
    void boundaryResidual();
  
  private:
  
    bool use_evisc, use_SUPG;

  };
  
}

#endif
