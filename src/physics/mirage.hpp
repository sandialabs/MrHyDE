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

#ifndef MIRAGE_H
#define MIRAGE_H

#include "physicsBase.hpp"

namespace MrHyDE {
  
  class mirage : public physicsbase {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    mirage() {} ;
    
    ~mirage() {};
    
    mirage(Teuchos::ParameterList & settings, const int & dimension_);
    
    // ========================================================================================
    // ========================================================================================
    
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_);
    
    // ========================================================================================
    // ========================================================================================
    
    void volumeResidual();
        
    void boundaryResidual();
    
    // ========================================================================================
    // ========================================================================================
    
    void setWorkset(Teuchos::RCP<workset> & wkset_);

    // ========================================================================================
    // ========================================================================================
    
    void planewaveSource();
    
    void isotropicPML();
    
    void anisotropicPML();
    
  private:
    
    int Enum, Bnum, spaceDim;
    bool use_explicit, use_leap_frog;
    
    // Planewave current source parameters
    bool use_planewave_source, current_cont_wave_, current_xcomponent_, current_ycomponent_, current_zcomponent_;
    double current_fr_center_, current_fr_band_, current_offset_, current_amplitude_, current_sigma_;
    double current_xmin_, current_xmax_, current_ymin_, current_ymax_, current_zmin_, current_zmax_;
    View_Sc2 current_x, current_y, current_z;
    
    // Isotropic PML parameters
    bool use_iPML;
    double iPML_sigma;
    double PML_B_factor;
    string iPML_type;
    bool iPML_have_xmin, iPML_have_xmax, iPML_have_ymin, iPML_have_ymax, iPML_have_zmin, iPML_have_zmax;
    double iPML_tol_xmin, iPML_tol_xmax, iPML_tol_ymin, iPML_tol_ymax, iPML_tol_zmin, iPML_tol_zmax;
    double iPML_pow_xmin, iPML_pow_xmax, iPML_pow_ymin, iPML_pow_ymax, iPML_pow_zmin, iPML_pow_zmax;
    double iPML_sigma_xmin, iPML_sigma_xmax, iPML_sigma_ymin, iPML_sigma_ymax, iPML_sigma_zmin, iPML_sigma_zmax;
    double iPML_xmin_start, iPML_xmax_start, iPML_ymin_start, iPML_ymax_start, iPML_zmin_start, iPML_zmax_start;
    double iPML_xmin_end, iPML_xmax_end, iPML_ymin_end, iPML_ymax_end, iPML_zmin_end, iPML_zmax_end;
    double iPML_zmin_x1, iPML_zmin_x2, iPML_zmin_y1, iPML_zmin_y2;
    double iPML_zmax_x1, iPML_zmax_x2, iPML_zmax_y1, iPML_zmax_y2;
    bool iPML_zmin_exclude, iPML_zmax_exclude;
    View_Sc2 iPML;
    
    // Anisotropic PML parameters
    bool use_aPML;
    double aPML_sigma;
    string aPML_type;
    bool aPML_have_xmin, aPML_have_xmax, aPML_have_ymin, aPML_have_ymax, aPML_have_zmin, aPML_have_zmax;
    double aPML_tol_xmin, aPML_tol_xmax, aPML_tol_ymin, aPML_tol_ymax, aPML_tol_zmin, aPML_tol_zmax;
    double aPML_pow_xmin, aPML_pow_xmax, aPML_pow_ymin, aPML_pow_ymax, aPML_pow_zmin, aPML_pow_zmax;
    double aPML_sigma_xmin, aPML_sigma_xmax, aPML_sigma_ymin, aPML_sigma_ymax, aPML_sigma_zmin, aPML_sigma_zmax;
    double aPML_xmin_start, aPML_xmax_start, aPML_ymin_start, aPML_ymax_start, aPML_zmin_start, aPML_zmax_start;
    double aPML_xmin_end, aPML_xmax_end, aPML_ymin_end, aPML_ymax_end, aPML_zmin_end, aPML_zmax_end;
    double aPML_zmin_x1, aPML_zmin_x2, aPML_zmin_y1, aPML_zmin_y2;
    double aPML_zmax_x1, aPML_zmax_x2, aPML_zmax_y1, aPML_zmax_y2;
    bool aPML_zmin_exclude, aPML_zmax_exclude;
    
    View_Sc2 aPML_xx, aPML_yy, aPML_zz;
    
    // Various other mirage closure model parameters
    double mirage_epsilon, mirage_mu, mirage_ri, mirage_sigma;
    
    Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mirage::volumeResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mirage::volumeResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mirage::boundaryResidual() - function evaluation");
    Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mirage::boundaryResidual() - evaluation of residual");
    Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mirage::computeFlux() - function evaluation");
    Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mirage::computeFlux() - evaluation of flux");
    
    Teuchos::RCP<Teuchos::Time> iPMLTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mirage::isotropicPML()");
    Teuchos::RCP<Teuchos::Time> aPMLTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mirage::anisotropicPML()");
    Teuchos::RCP<Teuchos::Time> planewaveTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::mirage::planewaveSource()");
    
  };
  
}

#endif
