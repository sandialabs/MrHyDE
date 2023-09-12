/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_MAXWELLS_FP_H
#define MRHYDE_MAXWELLS_FP_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  /**
   * \brief maxwells_fp physics class.
   *
   * This class computes volumetric residuals for the physics described by the following weak form:
   * \f{eqnarray*}
   *   \dots
   * \f}
   * Where the unknown ___ is the ___.
   * The following functions may be specified in the input.yaml file:
   */

  template<class EvalT>
  class maxwells_fp : public PhysicsBase<EvalT> {
  public:
    
    // These are necessary due to the combination of templating and inheritance
    using PhysicsBase<EvalT>::functionManager;
    using PhysicsBase<EvalT>::wkset;
    using PhysicsBase<EvalT>::label;
    using PhysicsBase<EvalT>::myvars;
    using PhysicsBase<EvalT>::mybasistypes;
    
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
    maxwells_fp() {};
    ~maxwells_fp() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    maxwells_fp(Teuchos::ParameterList & settings, const int & dimension_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT>> & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
    
    // ========================================================================================
    // ========================================================================================
    
    void boundaryResidual();
    
    // ========================================================================================
    // true solution for error calculation
    // ========================================================================================
    
    void edgeResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // =======================================================================================
    // return frequency
    // ======================================================================================
    
    EvalT getFreq(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
    
    // ========================================================================================
    // return magnetic permeability
    // ========================================================================================
    
    vector<EvalT> getPermeability(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
    
    // ========================================================================================
    // return inverse of magnetic permeability
    // ========================================================================================
    
    vector<EvalT> getInvPermeability(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
    
    // ========================================================================================
    // return electric permittivity
    // ========================================================================================
    
    vector<EvalT> getPermittivity(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
    
    // ========================================================================================
    // return current density in interior of domain
    // ========================================================================================
    
    vector<vector<EvalT> > getInteriorCurrent(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
    
    // ========================================================================================
    // return charge density in interior of domain
    // ========================================================================================
    
    vector<EvalT> getInteriorCharge(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
    
    // =======================================================================================
    // return electric current on boundary of domain
    // =======================================================================================
    
    vector<vector<EvalT> > getBoundaryCurrent(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time,
                                           const string & side_name, const int & boundary_type) const;
    
    // ========================================================================================
    // return charge density on boundary of domain (should be surface divergence of boundary current divided by i*omega
    // ========================================================================================
    
    vector<EvalT> getBoundaryCharge(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);
    
    // ========================================================================================
    // TMW: this needs to be deprecated
    // ========================================================================================
    
    void updateParameters(const vector<Teuchos::RCP<vector<EvalT> > > & params, const std::vector<string> & paramnames);
    
    // ========================================================================================
    // ========================================================================================
    
    
  private:
    
    vector<EvalT> mu_params; //permeability
    vector<EvalT> eps_params; //permittivity
    vector<EvalT> freq_params; //frequency
    vector<EvalT> source_params, boundary_params;
    
    int Axr_num, phir_num, Ayr_num, Azr_num, Axi_num, phii_num, Ayi_num, Azi_num;
    
    int test;
    
    Kokkos::View<ScalarT***,AssemblyDevice> Erx, Ery, Erz, Eix, Eiy, Eiz; //corresponding electric field
    bool calcE; //whether to calculate E field here (does not give smooth result like Paraview does; cause unknown)
    
    ScalarT essScale;
    
    //Kokkos::View<int****,AssemblyDevice> sideinfo;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwells_fp::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwells_fp::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwells_fp::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwells_fp::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwells_fp::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwells_fp::computeFlux() - evaluation of flux");
    
  }; //end class
  
}

#endif
