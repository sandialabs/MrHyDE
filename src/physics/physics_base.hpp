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

#ifndef PHYSBASE_H
#define PHYSBASE_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "klexpansion.hpp"
#include "workset.hpp"
#include "functionManager.hpp"

namespace MrHyDE {
  
  class physicsbase {
    
  public:
    
    physicsbase() {} ;
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    virtual ~physicsbase() {};
    
    // ========================================================================================
    // Define the functions for this module (not necessary, but probably need to be defined in all modules)
    // ========================================================================================
    
    virtual
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_) {} ;
    
    // ========================================================================================
    // The volumetric contributions to the residual
    // ========================================================================================
    
    virtual
    void volumeResidual() {};
    
    // ========================================================================================
    // The boundary contributions to the residual
    // ========================================================================================
    
    virtual
    void boundaryResidual() {};
    
    // ========================================================================================
    // The edge (2D) and face (3D) contributions to the residual
    // ========================================================================================
    
    virtual
    void faceResidual() {};
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    virtual
    void computeFlux() {};
    
    // ========================================================================================
    // ========================================================================================
    
    virtual void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                                  const std::vector<string> & paramnames) {} ;
    
    // ========================================================================================
    // ========================================================================================
    
    virtual void setWorkset(Teuchos::RCP<workset> & wkset_) {
      wkset = wkset_;
    };
    
    // ========================================================================================
    // ========================================================================================
    
    string label;
    
    Teuchos::RCP<workset> wkset;
    Teuchos::RCP<FunctionManager> functionManager;
    int spaceDim;
    vector<string> myvars, mybasistypes;
    bool include_face = false, isaux = false;
    string prefix = "";
    
    // Probably not used much
    View_AD2 adjrhs;
    
    // On host, so ok
    Kokkos::View<int**,HostDevice> bcs;
    
    
  };
  
}

#endif
