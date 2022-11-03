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

#ifndef MRHYDE_POROUSMIXEDHYBRID_H
#define MRHYDE_POROUSMIXEDHYBRID_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  class porousMixedHybrid : public physicsbase {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    porousMixedHybrid() {} ;
    
    ~porousMixedHybrid() {};
    
    porousMixedHybrid(Teuchos::ParameterList & settings, const int & dimension_);
    
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
    
    void setWorkset(Teuchos::RCP<workset> & wkset_);

    //void setVars(std::vector<string> & varlist_);
    
    // ========================================================================================
    // ========================================================================================
    
    //void setAuxVars(std::vector<string> & auxvarlist);
    
    // ========================================================================================
    // ========================================================================================
    
    void updatePerm(View_AD2 Kinv_xx, View_AD2 Kinv_yy, View_AD2 Kinv_zz);
    
  private:
    
    int pnum=-1, unum=-1, lambdanum=-1;
    int auxpnum=-1, auxunum=-1, auxlambdanum=-1;
    int dxnum=-1, dynum=-1, dznum=-1;
    bool usePermData;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousHDIV_HYBRID::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousHDIV_HYBRID::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousHDIV_HYBRID::computeFlux() - evaluation of interface flux");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousHDIV_HYBRID::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::porousHDIV_HYBRID::boundaryResidual() - evaluation of residual");
    
  };
  
}

#endif
