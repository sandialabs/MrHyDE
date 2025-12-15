/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_LEVELSET_H
#define MRHYDE_LEVELSET_H

#include "physicsBase.hpp"
#include <optional>

#include "rothermal.hpp"

namespace MrHyDE {
  

  template<class EvalT>
  class levelSet : public PhysicsBase<EvalT> {
  public:

  // ========================================================================================
  // Kokkos view types
  // ========================================================================================
    typedef Kokkos::View<int*, AssemblyDevice> View_Int1;
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;


  // ========================================================================================
  // PhysicsBase types
  // ========================================================================================
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;


  // ========================================================================================
  // constructor and destructor
  // ========================================================================================
    levelSet() {} ;
    ~levelSet() {};

    // parameter list
    levelSet(Teuchos::ParameterList & settings, const int & dimension_);

    // ========================================================================================
    // define functions
    // ========================================================================================
    void defineFunctions(
      Teuchos::ParameterList & fs,
      Teuchos::RCP<FunctionManager<EvalT> > & functionManager_
    );
  
    // ========================================================================================
    // volume residual
    // ========================================================================================
    void volumeResidual();
    
    // ========================================================================================
    // set workset
    // ========================================================================================
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);
    
    // ========================================================================================
    // compute tau: SUPG stabilization parameter
    // ========================================================================================
    KOKKOS_FUNCTION EvalT computeTau(
      const EvalT & xvl,
      const EvalT & yvl,
      const ScalarT & h
    ) const;

    // ========================================================================================
    // used to save computed fields
    // ========================================================================================
    std::vector<string> getDerivedNames();
    std::vector<View_EvalT2> getDerivedValues();
    
  private:
    
    int phinum;

    bool useRothermal;

    Teuchos::ParameterList  settings_;

    // rothermal ROS object
    Teuchos::RCP<rothermal<EvalT> > rothermal_;

    // zero tolerance: used to avoid division by zero
    ScalarT zero_tol  = 1e-9;

    // function data: used to prepare functions for use in volume residual
    template<typename T>
    struct FuncData {
        Vista<T> beta;
        Vista<T> xvel;
        Vista<T> yvel;
    };

    // field data: used to prepare fields for use in volume residual
    template<typename T>
    struct FieldData {
        using phi_type    = decltype(std::declval<Workset<T>>().getSolutionField("phi"));
        using phi_t_type  = decltype(std::declval<Workset<T>>().getSolutionField("phi_t"));
        using grad_type   = View_EvalT2;
        using offset_type = decltype(Kokkos::subview(std::declval<Workset<T>>().offsets, 0, Kokkos::ALL()));

        phi_type phi;
        phi_t_type phi_t;
        grad_type dphi_dx;
        grad_type dphi_dy;
        offset_type off;
    };

    // prepare functions for use in volume residual
    FuncData<EvalT> prepareFunctions();

    // prepare fields for use in volume residual
    FieldData<EvalT> prepareFields();    

    // time counters
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::levelSet::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::levelSet::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::levelSet::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::levelSet::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::levelSet::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::levelSet::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
