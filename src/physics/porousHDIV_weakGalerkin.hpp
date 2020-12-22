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

#ifndef POROUSHDIVWG_H
#define POROUSHDIVWG_H

#include "physics_base.hpp"

namespace MrHyDE {
  /*
  static void porousHDIVWGHelp() {
    cout << "********** Help and Documentation for the Porous (HDIV) Physics Module **********" << endl << endl;
    cout << "Model:" << endl << endl;
    cout << "User defined functions: " << endl << endl;
  }
  */
  
  /*
   * This class computes the solution to the physics formed by applying the weak Galerkin finite
   * element method to the PDE for Darcy flow. It yields the following finite element system:
   * @f{eqnarray*}
   *   (\mathbf{u},\mathbf{v})_T + (p_0, \nabla\cdot\mathbf{v})_T
   *       - \langle p_\partial, \mathbf{v} \cdot \mathbf{n} \rangle_{\mathcal{E}_T} &=& 0, \\
   *   (\mathbf{K}\mathbf{u}, \mathbf{s})_T + (\mathbf{t},\mathbf{s})_T &=& 0, \\
   *   (\nabla\cdot\mathbf{t}, q_0)_T &=& (f, q_0)_T, \\
   *   -\sum\limits{T\in\mathcal{T}_h}
   *       \langle \mathbf{t}\cdot\mathbf{n}, q_\partial\rangle_{\mathcal{E}_T} &=& 0.
   * @f}
   * Where the unknowns $p_0$, $p_\partial$, $\mathbf{u}$, and $\mathbf{t}$ are the following:
   *   - $p_0$ is the interior pressure
   *   - $p_\partial$ is the boundary pressure
   *   - $\mathbf{u}$ is the weak Gradient
   *   - $\mathbf{t}$ is the Darcy velocity
   * The following functions may be specified in the input.yaml file
   *   - "source" is the source term, $f$
   *   - "kxx" is the xx entry of the permeability tensor, $\mathbf{K}$
   *   - "kxy", "kyx", "kyy" are defined similarly, and similar terms involving z may used in 3d
   */
  class porousHDIV_WG : public physicsbase {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    porousHDIV_WG() {} ;
    
    ~porousHDIV_WG() {};
    
    porousHDIV_WG(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
    
    // ========================================================================================
    // ========================================================================================
    
    void boundaryResidual();
    
    // ========================================================================================
    // The edge (2D) and face (3D) contributions to the residual
    // ========================================================================================
    
    void faceResidual();
    
    // ========================================================================================
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    void setVars(std::vector<string> & varlist_);
    
    // ========================================================================================
    // ========================================================================================
    
    void setAuxVars(std::vector<string> & auxvarlist);
    
    // ========================================================================================
    // ========================================================================================
    
    void updatePerm(View_AD2_sv perm);
    
  private:
    
    int spaceDim;
    
    int pintnum=-1, pbndrynum=-1, unum=-1, tnum=-1;
    int auxpintnum=-1, auxpbndrynum=-1, auxunum=-1, auxtnum=-1;
    int dxnum=-1,dynum=-1,dznum=-1;
    
    
    bool usePermData;
    
    vector<string> varlist;
    Kokkos::View<int****,AssemblyDevice> sideinfo;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_WG::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_WG::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_WG::computeFlux() - evaluation of interface flux");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_WG::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porousHDIV_WG::boundaryResidual() - evaluation of residual");
    
  };
  
}

#endif
