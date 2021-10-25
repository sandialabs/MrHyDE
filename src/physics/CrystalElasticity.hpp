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

#ifndef CrystalElasticity_H
#define CrystalElasticity_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"

namespace MrHyDE {
  
  class CrystalElastic {
  public:
    
    CrystalElastic() {} ;
    ~CrystalElastic() {};
    
    CrystalElastic(Teuchos::ParameterList & settings,
                   const int & dimension_);
    
    //----------------------------------------------------------------------------

    void computeLatticeTensor();

    //----------------------------------------------------------------------------

    void updateParams(Teuchos::RCP<workset> & wkset);
    
    //----------------------------------------------------------------------------
    
    void computeStress(Teuchos::RCP<workset> & wkset, vector<int> & indices,
                       const bool & onside, View_AD4 stress);
    
    //----------------------------------------------------------------------------
    
    void computeRotatedTensor(Teuchos::RCP<workset> & wkset);
    
    // Public Data
    int dimension;
    bool allow_rotations;
    ScalarT c11_,c22_,c33_,c44_,c55_,c66_,c12_,c13_,c23_,c15_,c25_,c35_,c46_;
    View_Sc4 C; // lattice stiffness tensor (does not depend on elements)
    View_Sc5 Cr; // rotated stiffness tensor
    ScalarT lambda, mu, e_ref, alpha_T;
    
    Teuchos::RCP<Teuchos::Time> computeRotatedTensorTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::CrystalElasticity::computeRotatedTensor");
    Teuchos::RCP<Teuchos::Time> computeStressTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::CrystalElasticity::computeStress");
    
  };
  
}

#endif 
