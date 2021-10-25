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

#ifndef THERMAL_H
#define THERMAL_H

#include "physicsBase.hpp"
#include "vista.hpp"

namespace MrHyDE {
  
  /*
  static void thermalHelp() {
    cout << "********** Help and Documentation for the Thermal Physics Module **********" << endl << endl;
    cout << "Model:" << endl << endl;
    cout << "User defined functions: " << endl << endl;
  }
  */
  
  class thermal : public physicsbase {
  public:
    
    thermal() {} ;
    
    ~thermal() {};
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    thermal(Teuchos::ParameterList & settings, const int & dimension_) ;
    
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
    // The boundary/edge flux
    // ========================================================================================
    
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<workset> & wkset_);
    
    /**
     * @brief Returns the integrands and their types (boundary/volume) for integrated quantities required
     * by the thermal module. Currently, this is only used for testing purposes. 
     *
     * @return integrandsNamesAndTypes  Integrands, names, and type (boundary/volume) (matrix of strings).
     */
    
    std::vector< std::vector<string> > setupIntegratedQuantities(const int & spaceDim);


  private:
    
    int e_num = -1, ux_num = -1, uy_num = -1, uz_num = -1;
    int e_basis_num = -1;
    int auxe_num = -1;
    int IQ_start;
    
    //View_AD2 e_vol, dedt_vol, dedx_vol, dedy_vol, dedz_vol;
    //View_AD2 e_side, dedx_side, dedy_side, dedz_side;
    //View_AD2 e_face, dedx_face, dedy_face, dedz_face;
    //View_AD2 ux_vol, uy_vol, uz_vol;
    
    bool have_nsvel,test_IQs;
    ScalarT formparam;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::thermal::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::thermal::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::thermal::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::thermal::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::thermal::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::thermal::computeFlux() - evaluation of flux");
    
  };
  
}

#endif
