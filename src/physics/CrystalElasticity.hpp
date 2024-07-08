/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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

  template<class EvalT>
  class CrystalElastic {

    typedef Kokkos::View<EvalT*,ContLayout,AssemblyDevice> View_EvalT1;
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    typedef Kokkos::View<EvalT***,ContLayout,AssemblyDevice> View_EvalT3;
    typedef Kokkos::View<EvalT****,ContLayout,AssemblyDevice> View_EvalT4;
    typedef Kokkos::View<EvalT*****,ContLayout,AssemblyDevice> View_EvalT5;
    
  public:
    
    CrystalElastic() {} ;
    ~CrystalElastic() {};
    
    CrystalElastic(Teuchos::ParameterList & settings,
                   const int & dimension_);
    
    //----------------------------------------------------------------------------

    void computeLatticeTensor();

    //----------------------------------------------------------------------------

    void updateParams();
    
    //----------------------------------------------------------------------------
    
    void computeStress(Teuchos::RCP<Workset<EvalT> > & wkset, vector<int> & indices,
                       const bool & onside, View_EvalT4 stress);
    
    //----------------------------------------------------------------------------
    
    void computeRotatedTensor(Teuchos::RCP<Workset<EvalT> > & wkset);
    
    //----------------------------------------------------------------------------
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);
    
  private:
  
    int dimension;
    bool allow_rotations;
    EvalT c11_,c22_,c33_,c44_,c55_,c66_,c12_,c13_,c23_,c15_,c25_,c35_,c46_;
    View_EvalT4 C; // lattice stiffness tensor (does not depend on elements)
    View_EvalT5 Cr; // rotated stiffness tensor
    EvalT lambda, mu, e_ref, alpha_T;
    Teuchos::RCP<Workset<EvalT> > wkset;
    
    Teuchos::RCP<Teuchos::Time> computeRotatedTensorTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::CrystalElasticity::computeRotatedTensor");
    Teuchos::RCP<Teuchos::Time> computeStressTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::CrystalElasticity::computeStress");
    
  };
  
}

#endif 
