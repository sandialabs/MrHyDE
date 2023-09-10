/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_PHYSICS_IMPORTER_H
#define MRHYDE_PHYSICS_IMPORTER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief physicsImporter physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   */

  template<class EvalT>
  class PhysicsImporter {
    
  public:
    
    PhysicsImporter() {} ;
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    ~PhysicsImporter() {};
    
    // ========================================================================================
    // Define the functions for this module (not necessary, but probably need to be defined in all modules)
    // ========================================================================================
    
    vector<Teuchos::RCP<PhysicsBase<EvalT> > > import(vector<string> & module_list,
                                              Teuchos::ParameterList & settings,
                                              const int & dimension,
                                              Teuchos::RCP<MpiComm> & Commptr);
  };
  
}

#endif
