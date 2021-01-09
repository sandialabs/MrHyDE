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

#ifndef PHYSICS_IMP_H
#define PHYSICS_IMP_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "physics_base.hpp"

namespace MrHyDE {
  
  class physicsImporter {
    
  public:
    
    physicsImporter() {} ;
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    ~physicsImporter() {};
    
    // ========================================================================================
    // Define the functions for this module (not necessary, but probably need to be defined in all modules)
    // ========================================================================================
    
    vector<Teuchos::RCP<physicsbase> > import(vector<string> & module_list,
                                              Teuchos::RCP<Teuchos::ParameterList> & settings,
                                              const bool & isaux,
                                              Teuchos::RCP<MpiComm> & Commptr);
  };
  
}

#endif
