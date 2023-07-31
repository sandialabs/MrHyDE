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

#ifndef MRHYDE_CrystalElasticity_H
#define MRHYDE_CrystalElasticity_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"

namespace MrHyDE {
  
  /**
   * \brief CrystalElasticity physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   */
  class CrystalElastic {

    #ifndef MrHyDE_NO_AD
      typedef Kokkos::View<AD*,ContLayout,AssemblyDevice> View_AD1;
      typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2;
      typedef Kokkos::View<AD***,ContLayout,AssemblyDevice> View_AD3;
      typedef Kokkos::View<AD****,ContLayout,AssemblyDevice> View_AD4;
    #else
      typedef View_Sc1 View_AD1;
      typedef View_Sc2 View_AD2;
      typedef View_Sc3 View_AD3;
      typedef View_Sc4 View_AD4;
    #endif

  public:
    
    CrystalElastic() {} ;
    ~CrystalElastic() {};
    
    CrystalElastic(Teuchos::ParameterList & settings,
                   const int & dimension_);
    
    //----------------------------------------------------------------------------

    void computeLatticeTensor();

    //----------------------------------------------------------------------------

    void updateParams(Teuchos::RCP<Workset<AD> > & wkset);
    
    //----------------------------------------------------------------------------
    
    void computeStress(Teuchos::RCP<Workset<AD> > & wkset, vector<int> & indices,
                       const bool & onside, View_AD4 stress);
    
    //----------------------------------------------------------------------------
    
    void computeRotatedTensor(Teuchos::RCP<Workset<AD> > & wkset);
    
  private:
  
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
