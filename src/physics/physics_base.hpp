/***********************************************************************
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
#include "data.hpp"
#include "klexpansion.hpp"
#include "workset.hpp"
#include "boundaryWorkset.hpp"
#include "functionInterface.hpp"

class physicsbase {
  
public:
  
  physicsbase() {} ;
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  ~physicsbase() {};
  
  // ========================================================================================
  // The volumetric contributions to the residual
  // ========================================================================================
  
  virtual
  void volumeResidual() = 0;

  // ========================================================================================
  // The boundary contributions to the residual
  // ========================================================================================
  
  virtual
  void boundaryResidual() = 0;
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  virtual
  void computeFlux() = 0;

  // ========================================================================================
  // Set the global index for each variable
  // ========================================================================================
  
  virtual void setVars(vector<string> & varlist_) = 0;
  
  // ========================================================================================
  // ========================================================================================
  
  virtual void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params, const std::vector<string> & paramnames) {} ;
  
    // ========================================================================================
    // ========================================================================================
    
  string label;
  
  Teuchos::RCP<workset> wkset;
  vector<Teuchos::RCP<BoundaryWorkset> > boundaryWkset;
  Teuchos::RCP<FunctionInterface> functionManager;
  int spaceDim;
  vector<string> myvars, mybasistypes;
  
};

#endif
