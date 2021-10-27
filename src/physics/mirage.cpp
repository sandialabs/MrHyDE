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

#include "mirage.hpp"
using namespace MrHyDE;

mirage::mirage(Teuchos::ParameterList & settings, const int & dimension_)
: physicsbase(settings, dimension_)
{
  
  label = "mirage";
  
  spaceDim = dimension_;
  
  myvars.push_back("E");
  myvars.push_back("B");
  
  mybasistypes.push_back("HCURL");
  if (spaceDim == 2) {
    mybasistypes.push_back("HVOL");
  }
  else if (spaceDim == 3) {
    mybasistypes.push_back("HDIV");
  }
  
  useLeapFrog = settings.get<bool>("use leap frog",false);
  
  if (settings.isSublist("Mirage settings")) {
    
    // Extract the Mirage settings sublist
    Teuchos::ParameterList msettings = settings.sublist("Mirage settings");
    
    // Get permittivity (epsilon) from mirage
    mirage_epsilon = 1.0e-11;
    if (msettings.isSublist("PERMITTIVITY")) {
      if (msettings.sublist("PERMITTIVITY").isParameter("Value")) {
        mirage_epsilon = msettings.sublist("PERMITTIVITY").get<double>("Value");
      }
      else if (msettings.sublist("PERMITTIVITY").isParameter("epsilon")) {
        mirage_epsilon = msettings.sublist("PERMITTIVITY").get<double>("epsilon");
      }
    }
    
    // Get permeability (mu) from mirage
    mirage_mu = 1.25e-6;
    if (msettings.isSublist("INVERSE_PERMEABILITY")) {
      if (msettings.sublist("INVERSE_PERMEABILITY").isParameter("Value")) {
        mirage_mu = 1.0/msettings.sublist("INVERSE_PERMEABILITY").get<double>("Value");
      }
    }
    
    // Get refractive index from mirage
    mirage_ri = 1.0;
    if (msettings.isSublist("REFRACTIVE_INDEX")) {
      if (msettings.sublist("REFRACTIVE_INDEX").isParameter("Value")) {
        mirage_ri = msettings.sublist("REFRACTIVE_INDEX").get<double>("Value");
      }
    }
    
    // Get physical conductivity from mirage
    mirage_sigma = 0.0;
    if (msettings.isSublist("CONDUCTIVITY")) {
      if (msettings.sublist("CONDUCTIVITY").isParameter("Value")) {
        mirage_sigma = msettings.sublist("CONDUCTIVITY").get<double>("Value");
      }
    }
    
    // ---------------------------------------------------
    // Planewave current pulse settings from mirage
    
    use_planewave_source = msettings.isSublist("CURRENT");
    if (use_planewave_source) {
      // Extract the CURRENT sublist
      Teuchos::ParameterList csettings = msettings.sublist("CURRENT");
      
      double epsilon_     = csettings.get("PERMITTIVITY", double(8.854187817e-12));
      double mu_          = csettings.get("PERMEABILITY", double(1.2566370614e-6));
      bool use_wl_        = csettings.get("USE_WAVELENGTH_INSTEAD_OF_FREQUENCY", false);
      current_cont_wave_  = csettings.get("USE_CONTINUOUS_WAVE_GENERATOR", false);
      double wl_center_   = csettings.get("WAVELENGTH_CENTER", double(4e-6));
      double wl_band_     = csettings.get("WAVELENGTH_BAND", double(2e-6));
      current_fr_center_  = csettings.get("FREQUENCY_CENTER", double(75e12));
      current_fr_band_    = csettings.get("FREQUENCY_BAND", double(40e12));
      string bw_sense_    = csettings.get("BANDWIDTH_SENSE", "FWHM");
      current_offset_     = csettings.get("PULSE_TIME_OFFSET", double(150e-15));
      current_amplitude_  = csettings.get("PULSE_AMPLITUDE", double(1.0));
      current_xmin_       = csettings.get("X_MIN", double(-1e20));
      current_xmax_       = csettings.get("X_MAX", double(1e20));
      current_ymin_       = csettings.get("Y_MIN", double(-1e20));
      current_ymax_       = csettings.get("Y_MAX", double(1e20));
      current_zmin_       = csettings.get("Z_MIN", double(0e-6));
      current_zmax_       = csettings.get("Z_MAX", double(5e-6));
      current_xcomponent_ = csettings.get("ACTIVATE_CURRENT_X_COMPONENT", true);
      current_ycomponent_ = csettings.get("ACTIVATE_CURRENT_Y_COMPONENT", false);
      current_zcomponent_ = csettings.get("ACTIVATE_CURRENT_Z_COMPONENT", false);
      
      double c_         = std::sqrt(1.0/epsilon_/mu_);                                         // speed of light
      if (use_wl_) {
        current_fr_center_ = c_ / wl_center_;                                                   // center frequency
        current_fr_band_   = c_ / (std::pow(wl_center_,2) - std::pow(wl_band_,2)/4) * wl_band_; // bandwidth in terms of frequency, i.e.,
        // f_max - f_min = c/wl_min - c/wl_max
        // with wl_min = wl_center_ - wl_band_/2 and
        //      wl_max = wl_center_ + wl_band_/2
      }
      
      
      if (std::strcmp(bw_sense_.c_str(), "2SIGMA")==0) { // full bandwidth at 2-sigma
        current_sigma_ = current_fr_band_/2.0;
      }
      else if (std::strcmp(bw_sense_.c_str(), "FWHM")==0) { // Full (band)Width at Half Max
        current_sigma_ = current_fr_band_/(2.0*std::sqrt(2.0*std::log(2.0)));
      }
      else if (std::strcmp(bw_sense_.c_str(), "4SIGMA")==0) { // full bandwidth at 4-sigma
        current_sigma_ = current_fr_band_/4.0;
      }
      else if (std::strcmp(bw_sense_.c_str(), "FWTM")==0) { // Full (band)Width at one-Tenth Max
        current_sigma_ = current_fr_band_/(2.0*std::sqrt(2.0*std::log(10.0)));
      }
      else if (std::strcmp(bw_sense_.c_str(), "6SIGMA")==0) { // full bandwidth at 6-sigma
        current_sigma_ = current_fr_band_/6.0;
      }
    }
    // ---------------------------------------------------
    
    // PMLs from mirage
    // Note that both of these can be active at the same time
    // It is up to the user to avoid duplicating the PMLs on a particular boundary
    PML_B_factor = 1.13e11;
    
    use_iPML = msettings.isSublist("PML ABSORBER");
    
    if (use_iPML) {
      
      Teuchos::ParameterList pmlsettings = msettings.sublist("PML ABSORBER");
      
      iPML_sigma = pmlsettings.get("sigma",0.0);
      iPML_type = pmlsettings.get("PML type","exponential"); // or polynomial
      if (pmlsettings.isSublist("PML xmax")) {
        iPML_have_xmax  = true;
        iPML_tol_xmax   = pmlsettings.sublist("PML xmax").get("exp tolerance",double(1e-3));
        iPML_pow_xmax   = pmlsettings.sublist("PML xmax").get("poly power",double(3.0));
        iPML_sigma_xmax = pmlsettings.sublist("PML xmax").get("max sigma",double(0.0));
        iPML_xmax_start = pmlsettings.sublist("PML xmax").get("start location",double(0.0));
        iPML_xmax_end   = pmlsettings.sublist("PML xmax").get("end location",double(1.0));
      }
      else {
        iPML_have_xmax  = false;
      }
      if (pmlsettings.isSublist("PML xmin")) {
        iPML_have_xmin  = true;
        iPML_tol_xmin   = pmlsettings.sublist("PML xmin").get("exp tolerance",double(1e-3));
        iPML_pow_xmin   = pmlsettings.sublist("PML xmin").get("poly power",double(3.0));
        iPML_sigma_xmin = pmlsettings.sublist("PML xmin").get("max sigma",double(0.0));
        iPML_xmin_start = pmlsettings.sublist("PML xmin").get("start location",double(0.0));
        iPML_xmin_end   = pmlsettings.sublist("PML xmin").get("end location",double(-1.0));
      }
      else {
        iPML_have_xmin  = false;
      }
      if (pmlsettings.isSublist("PML ymax")) {
        iPML_have_ymax  = true;
        iPML_tol_ymax   = pmlsettings.sublist("PML ymax").get("exp tolerance",double(1e-3));
        iPML_pow_ymax   = pmlsettings.sublist("PML ymax").get("poly power",double(3.0));
        iPML_sigma_ymax = pmlsettings.sublist("PML ymax").get("max sigma",double(0.0));
        iPML_ymax_start = pmlsettings.sublist("PML ymax").get("start location",double(0.0));
        iPML_ymax_end   = pmlsettings.sublist("PML ymax").get("end location",double(1.0));
      }
      else {
        iPML_have_ymax  = false;
      }
      if (pmlsettings.isSublist("PML ymin")) {
        iPML_have_ymin  = true;
        iPML_tol_ymin   = pmlsettings.sublist("PML ymin").get("exp tolerance",double(1e-3));
        iPML_pow_ymin   = pmlsettings.sublist("PML ymin").get("poly power",double(3.0));
        iPML_sigma_ymin = pmlsettings.sublist("PML ymin").get("max sigma",double(0.0));
        iPML_ymin_start = pmlsettings.sublist("PML ymin").get("start location",double(0.0));
        iPML_ymin_end   = pmlsettings.sublist("PML ymin").get("end location",double(-1.0));
      }
      else {
        iPML_have_ymin  = false;
      }
      if (pmlsettings.isSublist("PML zmax")) {
        iPML_have_zmax  = true;
        iPML_tol_zmax   = pmlsettings.sublist("PML zmax").get("exp tolerance",double(1e-3));
        iPML_pow_zmax   = pmlsettings.sublist("PML zmax").get("poly power",double(3.0));
        iPML_sigma_zmax = pmlsettings.sublist("PML zmax").get("max sigma",double(0.0));
        iPML_zmax_start = pmlsettings.sublist("PML zmax").get("start location",double(0.0));
        iPML_zmax_end   = pmlsettings.sublist("PML zmax").get("end location",double(1.0));
        iPML_zmax_x1    = pmlsettings.sublist("PML zmax").get("x1",double(-1e100));
        iPML_zmax_x2    = pmlsettings.sublist("PML zmax").get("x2",double(1e100));
        iPML_zmax_y1    = pmlsettings.sublist("PML zmax").get("y1",double(-1e100));
        iPML_zmax_y2    = pmlsettings.sublist("PML zmax").get("y2",double(1e100));
        iPML_zmax_exclude = pmlsettings.sublist("PML zmax").get("exclude",false);
      }
      else {
        iPML_have_zmax  = false;
      }
      if (pmlsettings.isSublist("PML zmin")) {
        iPML_have_zmin    = true;
        iPML_tol_zmin     = pmlsettings.sublist("PML zmin").get("exp tolerance",double(1e-3));
        iPML_pow_zmin     = pmlsettings.sublist("PML zmin").get("poly power",double(3.0));
        iPML_sigma_zmin   = pmlsettings.sublist("PML zmin").get("max sigma",double(0.0));
        iPML_zmin_start   = pmlsettings.sublist("PML zmin").get("start location",double(0.0));
        iPML_zmin_end     = pmlsettings.sublist("PML zmin").get("end location",double(-1.0));
        iPML_zmin_x1      = pmlsettings.sublist("PML zmin").get("x1",double(-1e100));
        iPML_zmin_x2      = pmlsettings.sublist("PML zmin").get("x2",double(1e100));
        iPML_zmin_y1      = pmlsettings.sublist("PML zmin").get("y1",double(-1e100));
        iPML_zmin_y2      = pmlsettings.sublist("PML zmin").get("y2",double(1e100));
        iPML_zmin_exclude = pmlsettings.sublist("PML zmin").get("exclude",false);
      }
      else {
        iPML_have_zmin  = false;
      }
      
    }
    
    use_aPML = msettings.isSublist("ANISOTROPIC PML ABSORBER");
    
    if (use_aPML) {
      
      Teuchos::ParameterList pmlsettings = msettings.sublist("ANISOTROPIC PML ABSORBER");
      
      aPML_sigma = pmlsettings.get("sigma",0.0);
      aPML_type = pmlsettings.get("PML type","exponential"); // or polynomial
      if (pmlsettings.isSublist("PML xmax")) {
        aPML_have_xmax  = true;
        aPML_tol_xmax   = pmlsettings.sublist("PML xmax").get("exp tolerance",double(1e-3));
        aPML_pow_xmax   = pmlsettings.sublist("PML xmax").get("poly power",double(3.0));
        aPML_sigma_xmax = pmlsettings.sublist("PML xmax").get("max sigma",double(0.0));
        aPML_xmax_start = pmlsettings.sublist("PML xmax").get("start location",double(0.0));
        aPML_xmax_end   = pmlsettings.sublist("PML xmax").get("end location",double(1.0));
      }
      else {
        aPML_have_xmax  = false;
      }
      if (pmlsettings.isSublist("PML xmin")) {
        aPML_have_xmin  = true;
        aPML_tol_xmin   = pmlsettings.sublist("PML xmin").get("exp tolerance",double(1e-3));
        aPML_pow_xmin   = pmlsettings.sublist("PML xmin").get("poly power",double(3.0));
        aPML_sigma_xmin = pmlsettings.sublist("PML xmin").get("max sigma",double(0.0));
        aPML_xmin_start = pmlsettings.sublist("PML xmin").get("start location",double(0.0));
        aPML_xmin_end   = pmlsettings.sublist("PML xmin").get("end location",double(-1.0));
      }
      else {
        aPML_have_xmin  = false;
      }
      if (pmlsettings.isSublist("PML ymax")) {
        aPML_have_ymax  = true;
        aPML_tol_ymax   = pmlsettings.sublist("PML ymax").get("exp tolerance",double(1e-3));
        aPML_pow_ymax   = pmlsettings.sublist("PML ymax").get("poly power",double(3.0));
        aPML_sigma_ymax = pmlsettings.sublist("PML ymax").get("max sigma",double(0.0));
        aPML_ymax_start = pmlsettings.sublist("PML ymax").get("start location",double(0.0));
        aPML_ymax_end   = pmlsettings.sublist("PML ymax").get("end location",double(1.0));
      }
      else {
        aPML_have_ymax  = false;
      }
      if (pmlsettings.isSublist("PML ymin")) {
        aPML_have_ymin  = true;
        aPML_tol_ymin   = pmlsettings.sublist("PML ymin").get("exp tolerance",double(1e-3));
        aPML_pow_ymin   = pmlsettings.sublist("PML ymin").get("poly power",double(3.0));
        aPML_sigma_ymin = pmlsettings.sublist("PML ymin").get("max sigma",double(0.0));
        aPML_ymin_start = pmlsettings.sublist("PML ymin").get("start location",double(0.0));
        aPML_ymin_end   = pmlsettings.sublist("PML ymin").get("end location",double(-1.0));
      }
      else {
        aPML_have_ymin  = false;
      }
      if (pmlsettings.isSublist("PML zmax")) {
        aPML_have_zmax  = true;
        aPML_tol_zmax   = pmlsettings.sublist("PML zmax").get("exp tolerance",double(1e-3));
        aPML_pow_zmax   = pmlsettings.sublist("PML zmax").get("poly power",double(3.0));
        aPML_sigma_zmax = pmlsettings.sublist("PML zmax").get("max sigma",double(0.0));
        aPML_zmax_start = pmlsettings.sublist("PML zmax").get("start location",double(0.0));
        aPML_zmax_end   = pmlsettings.sublist("PML zmax").get("end location",double(1.0));
        aPML_zmax_x1    = pmlsettings.sublist("PML zmax").get("x1",double(-1e100));
        aPML_zmax_x2    = pmlsettings.sublist("PML zmax").get("x2",double(1e100));
        aPML_zmax_y1    = pmlsettings.sublist("PML zmax").get("y1",double(-1e100));
        aPML_zmax_y2    = pmlsettings.sublist("PML zmax").get("y2",double(1e100));
        aPML_zmax_exclude = pmlsettings.sublist("PML zmax").get("exclude",false);
      }
      else {
        aPML_have_zmax  = false;
      }
      if (pmlsettings.isSublist("PML zmin")) {
        aPML_have_zmin    = true;
        aPML_tol_zmin     = pmlsettings.sublist("PML zmin").get("exp tolerance",double(1e-3));
        aPML_pow_zmin     = pmlsettings.sublist("PML zmin").get("poly power",double(3.0));
        aPML_sigma_zmin   = pmlsettings.sublist("PML zmin").get("max sigma",double(0.0));
        aPML_zmin_start   = pmlsettings.sublist("PML zmin").get("start location",double(0.0));
        aPML_zmin_end     = pmlsettings.sublist("PML zmin").get("end location",double(-1.0));
        aPML_zmin_x1      = pmlsettings.sublist("PML zmin").get("x1",double(-1e100));
        aPML_zmin_x2      = pmlsettings.sublist("PML zmin").get("x2",double(1e100));
        aPML_zmin_y1      = pmlsettings.sublist("PML zmin").get("y1",double(-1e100));
        aPML_zmin_y2      = pmlsettings.sublist("PML zmin").get("y2",double(1e100));
        aPML_zmin_exclude = pmlsettings.sublist("PML zmin").get("exclude",false);
      }
      else {
        aPML_have_zmin  = false;
      }
      
    }
    
  }
}

// ========================================================================================
// ========================================================================================

void mirage::defineFunctions(Teuchos::ParameterList & fs,
                              Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  // The function manager knows the appropriate dimensions
  current_x = View_Sc2("mirage current x",functionManager->numElem, functionManager->numip);
  current_y = View_Sc2("mirage current y",functionManager->numElem, functionManager->numip);
  current_z = View_Sc2("mirage current z",functionManager->numElem, functionManager->numip);
  
  // Add PML views/functions
  if (use_iPML) {
    iPML = View_Sc2("mirage iPML",functionManager->numElem, functionManager->numip);
  }
  
  if (use_aPML) {
    aPML_xx = View_Sc2("mirage aPML_xx",functionManager->numElem, functionManager->numip);
    aPML_yy = View_Sc2("mirage aPML_xx",functionManager->numElem, functionManager->numip);
    aPML_zz = View_Sc2("mirage aPML_xx",functionManager->numElem, functionManager->numip);
  }
  
  functionManager->addFunction("mu",mirage_mu,"ip");
  functionManager->addFunction("refractive index",mirage_ri,"ip");
  functionManager->addFunction("epsilon",mirage_epsilon,"ip");
  functionManager->addFunction("sigma",mirage_sigma,"ip");
  
}

// ========================================================================================
// ========================================================================================

void mirage::volumeResidual() {
  
  int E_basis = wkset->usebasis[Enum];
  int B_basis = wkset->usebasis[Bnum];
  
  Vista mu, epsilon, sigma, rindex;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    if (use_planewave_source) {
      this->planewaveSource();
    }
    mu = functionManager->evaluate("mu","ip");
    epsilon = functionManager->evaluate("epsilon","ip");
    rindex = functionManager->evaluate("refractive index","ip");
    sigma = functionManager->evaluate("sigma","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  int stage = wkset->current_stage;
  
  {
    if (spaceDim == 2) {
      // (dB/dt + curl E,V) = 0
      
      auto basis = wkset->basis[B_basis];
      auto dB_dt = wkset->getData("B_t");
      
      auto off = subview(wkset->offsets, Bnum, ALL());
      auto wts = wkset->wts;
      auto res = wkset->res;
      
      if (useLeapFrog) {
        if (stage == 0) {
          auto curlE = wkset->getData("curl(E)[x]");
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD f0 = (dB_dt(elem,pt) + curlE(elem,pt))*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
              }
            }
          });
        }
        else {
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD f0 = dB_dt(elem,pt)*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
              }
            }
          });
        }
      }
      else {
        auto curlE = wkset->getData("curl(E)[x]");
        parallel_for("Maxwells B volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD f0 = (dB_dt(elem,pt) + curlE(elem,pt))*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
            }
          }
        });
      }
    }
    else if (spaceDim == 3) {
      
      // (dB/dt + curl E,V) = 0
      
      auto off = subview(wkset->offsets, Bnum, ALL());
      auto wts = wkset->wts;
      auto res = wkset->res;
      auto basis = wkset->basis[B_basis];
      auto dBx_dt = wkset->getData("B_t[x]");
      auto dBy_dt = wkset->getData("B_t[y]");
      auto dBz_dt = wkset->getData("B_t[z]");
      
      if (useLeapFrog) {
        if (stage == 0) {
          auto curlE_x = wkset->getData("curl(E)[x]");
          auto curlE_y = wkset->getData("curl(E)[y]");
          auto curlE_z = wkset->getData("curl(E)[z]");
          
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD f0 = (dBx_dt(elem,pt) + curlE_x(elem,pt))*wts(elem,pt);
              AD f1 = (dBy_dt(elem,pt) + curlE_y(elem,pt))*wts(elem,pt);
              AD f2 = (dBz_dt(elem,pt) + curlE_z(elem,pt))*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
              }
            }
          });
          
          if (use_iPML) {
            this->isotropicPML();
            auto Bx = wkset->getData("B[x]");
            auto By = wkset->getData("B[y]");
            auto Bz = wkset->getData("B[z]");
            parallel_for("Maxwells B volume resid",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int elem ) {
              for (size_type pt=0; pt<basis.extent(2); pt++ ) {
                AD f0 = PML_B_factor*iPML(elem,pt)*Bx(elem,pt)*wts(elem,pt);
                AD f1 = PML_B_factor*iPML(elem,pt)*By(elem,pt)*wts(elem,pt);
                AD f2 = PML_B_factor*iPML(elem,pt)*Bz(elem,pt)*wts(elem,pt);
                for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                  res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                  res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                  res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
                }
              }
            });
          }
          if (use_aPML) {
            this->anisotropicPML();
            auto Bx = wkset->getData("B[x]");
            auto By = wkset->getData("B[y]");
            auto Bz = wkset->getData("B[z]");
            parallel_for("Maxwells B volume resid",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int elem ) {
              for (size_type pt=0; pt<basis.extent(2); pt++ ) {
                AD f0 = PML_B_factor*aPML_xx(elem,pt)*Bx(elem,pt)*wts(elem,pt);
                AD f1 = PML_B_factor*aPML_yy(elem,pt)*By(elem,pt)*wts(elem,pt);
                AD f2 = PML_B_factor*aPML_zz(elem,pt)*Bz(elem,pt)*wts(elem,pt);
                for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                  res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                  res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                  res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
                }
              }
            });
          }
        }
        else {
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD f0 = dBx_dt(elem,pt)*wts(elem,pt);
              AD f1 = dBy_dt(elem,pt)*wts(elem,pt);
              AD f2 = dBz_dt(elem,pt)*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
              }
            }
          });
          
        }
      }
      else {
        auto curlE_x = wkset->getData("curl(E)[x]");
        auto curlE_y = wkset->getData("curl(E)[y]");
        auto curlE_z = wkset->getData("curl(E)[z]");
        
        parallel_for("Maxwells B volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD f0 = (dBx_dt(elem,pt) + curlE_x(elem,pt))*wts(elem,pt);
            AD f1 = (dBy_dt(elem,pt) + curlE_y(elem,pt))*wts(elem,pt);
            AD f2 = (dBz_dt(elem,pt) + curlE_z(elem,pt))*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
              res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
              res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
            }
          }
        });
        if (use_iPML) {
          this->isotropicPML();
          auto Bx = wkset->getData("B[x]");
          auto By = wkset->getData("B[y]");
          auto Bz = wkset->getData("B[z]");
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD f0 = PML_B_factor*iPML(elem,pt)*Bx(elem,pt)*wts(elem,pt);
              AD f1 = PML_B_factor*iPML(elem,pt)*By(elem,pt)*wts(elem,pt);
              AD f2 = PML_B_factor*iPML(elem,pt)*Bz(elem,pt)*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
              }
            }
          });
        }
        if (use_aPML) {
          this->anisotropicPML();
          auto Bx = wkset->getData("B[x]");
          auto By = wkset->getData("B[y]");
          auto Bz = wkset->getData("B[z]");
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD f0 = PML_B_factor*aPML_xx(elem,pt)*Bx(elem,pt)*wts(elem,pt);
              AD f1 = PML_B_factor*aPML_yy(elem,pt)*By(elem,pt)*wts(elem,pt);
              AD f2 = PML_B_factor*aPML_zz(elem,pt)*Bz(elem,pt)*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
              }
            }
          });
        }
      }
    }
  }
  
  {
    // (eps*dE/dt,V) - (1/mu B, curl V) + (sigma E,V) = -(current,V)
    // Rewritten as: (eps*dEdt + sigma E + current, V) - (1/mu B, curl V) = 0
    
    if (spaceDim == 2) {
      if (!useLeapFrog || stage == 1) {
        auto basis = wkset->basis[E_basis];
        auto basis_curl = wkset->basis_curl[E_basis];
        
        auto dEx_dt = wkset->getData("E_t[x]");
        auto dEy_dt = wkset->getData("E_t[y]");
        auto B = wkset->getData("B");
        auto Ex = wkset->getData("E[x]");
        auto Ey = wkset->getData("E[y]");
        auto off = subview(wkset->offsets, Enum, ALL());
        auto wts = wkset->wts;
        auto res = wkset->res;
        
        parallel_for("Maxwells E volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD f0 = (epsilon(elem,pt)*rindex(elem,pt)*rindex(elem,pt)*dEx_dt(elem,pt) + (sigma(elem,pt)*Ex(elem,pt) + current_x(elem,pt)))*wts(elem,pt);
            AD f1 = (epsilon(elem,pt)*rindex(elem,pt)*rindex(elem,pt)*dEy_dt(elem,pt) + (sigma(elem,pt)*Ey(elem,pt) + current_y(elem,pt)))*wts(elem,pt);
            AD c0 = -1.0/mu(elem,pt)*B(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0) + c0*basis_curl(elem,dof,pt,0) + f1*basis(elem,dof,pt,1);
            }
          }
        });
      }
    }
    else if (spaceDim == 3) {
      
      if (!useLeapFrog || stage == 1) {
        auto basis = wkset->basis[E_basis];
        auto basis_curl = wkset->basis_curl[E_basis];
        auto dEx_dt = wkset->getData("E_t[x]");
        auto dEy_dt = wkset->getData("E_t[y]");
        auto dEz_dt = wkset->getData("E_t[z]");
        auto Bx = wkset->getData("B[x]");
        auto By = wkset->getData("B[y]");
        auto Bz = wkset->getData("B[z]");
        auto Ex = wkset->getData("E[x]");
        auto Ey = wkset->getData("E[y]");
        auto Ez = wkset->getData("E[z]");
        auto off = subview(wkset->offsets, Enum, ALL());
        auto wts = wkset->wts;
        auto res = wkset->res;
      
        parallel_for("Maxwells E volume resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            AD f0 = (epsilon(elem,pt)*rindex(elem,pt)*rindex(elem,pt)*dEx_dt(elem,pt) + (sigma(elem,pt)*Ex(elem,pt) + current_x(elem,pt)))*wts(elem,pt);
            AD f1 = (epsilon(elem,pt)*rindex(elem,pt)*rindex(elem,pt)*dEy_dt(elem,pt) + (sigma(elem,pt)*Ey(elem,pt) + current_y(elem,pt)))*wts(elem,pt);
            AD f2 = (epsilon(elem,pt)*rindex(elem,pt)*rindex(elem,pt)*dEz_dt(elem,pt) + (sigma(elem,pt)*Ez(elem,pt) + current_z(elem,pt)))*wts(elem,pt);
            
            AD c0 = -1.0/mu(elem,pt)*Bx(elem,pt)*wts(elem,pt);
            AD c1 = -1.0/mu(elem,pt)*By(elem,pt)*wts(elem,pt);
            AD c2 = -1.0/mu(elem,pt)*Bz(elem,pt)*wts(elem,pt);
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += f0*basis(elem,dof,pt,0) + c0*basis_curl(elem,dof,pt,0);
              res(elem,off(dof)) += f1*basis(elem,dof,pt,1) + c1*basis_curl(elem,dof,pt,1);
              res(elem,off(dof)) += f2*basis(elem,dof,pt,2) + c2*basis_curl(elem,dof,pt,2);
            }
          }
        });
        if (use_iPML) {
          this->isotropicPML();
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD f0 = iPML(elem,pt)*Ex(elem,pt)*wts(elem,pt);
              AD f1 = iPML(elem,pt)*Ey(elem,pt)*wts(elem,pt);
              AD f2 = iPML(elem,pt)*Ez(elem,pt)*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
              }
            }
          });
        }
        if (use_aPML) {
          this->anisotropicPML();
          parallel_for("Maxwells B volume resid",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int elem ) {
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              AD f0 = aPML_xx(elem,pt)*Ex(elem,pt)*wts(elem,pt);
              AD f1 = aPML_yy(elem,pt)*Ey(elem,pt)*wts(elem,pt);
              AD f2 = aPML_zz(elem,pt)*Ez(elem,pt)*wts(elem,pt);
              for (size_type dof=0; dof<basis.extent(1); dof++ ) {
                res(elem,off(dof)) += f0*basis(elem,dof,pt,0);
                res(elem,off(dof)) += f1*basis(elem,dof,pt,1);
                res(elem,off(dof)) += f2*basis(elem,dof,pt,2);
              }
            }
          });
        }
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

void mirage::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  
  
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  
  if (spaceDim == 2) {
    View_Sc2 nx, ny;
    nx = wkset->getDataSc("nx side");
    ny = wkset->getDataSc("ny side");
    
    //double gamma = 0.0;
    if (bcs(Bnum,cside) == "Neumann") { // Really ABC
      // Computes -nxnxE in B equation
      
      parallel_for("maxwell bndry resid ABC",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
    
      });
    }
    
  }
  else if (spaceDim == 3) {
    View_Sc2 nx, ny, nz;
    nx = wkset->getDataSc("nx side");
    ny = wkset->getDataSc("ny side");
    nz = wkset->getDataSc("nz side");
    auto Ex = wkset->getData("E[x] side");
    auto Ey = wkset->getData("E[y] side");
    auto Ez = wkset->getData("E[z] side");
    auto off = subview(wkset->offsets, Enum, ALL());
    auto basis = wkset->basis_side[wkset->usebasis[Enum]];
    
    double gamma = -0.9944;
    if (bcs(Bnum,cside) == "Neumann") { // Really ABC
      // Contributes -<nxnxE,V> along boundary in B equation
      
      parallel_for("maxwell bndry resid ABC",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
    
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD nce_x = ny(elem,pt)*Ez(elem,pt) - nz(elem,pt)*Ey(elem,pt);
          AD nce_y = nz(elem,pt)*Ex(elem,pt) - nx(elem,pt)*Ez(elem,pt);
          AD nce_z = nx(elem,pt)*Ey(elem,pt) - ny(elem,pt)*Ex(elem,pt);
          AD c0 = -(1.0+gamma)*(ny(elem,pt)*nce_z - nz(elem,pt)*nce_y)*wts(elem,pt);
          AD c1 = -(1.0+gamma)*(nz(elem,pt)*nce_x - nx(elem,pt)*nce_z)*wts(elem,pt);
          AD c2 = -(1.0+gamma)*(nx(elem,pt)*nce_y - ny(elem,pt)*nce_x)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += c0*basis(elem,dof,pt,0) + c1*basis(elem,dof,pt,1) + c2*basis(elem,dof,pt,2);
          }
        }
      });
      
      
      /*
      auto Bx = wkset->getData("B[x] side");
      auto By = wkset->getData("B[y] side");
      auto Bz = wkset->getData("B[z] side");
      parallel_for("maxwell bndry resid ABC",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
    
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          
          AD nce_x = ny(elem,pt)*Bz(elem,pt) - nz(elem,pt)*By(elem,pt);
          AD nce_y = nz(elem,pt)*Bx(elem,pt) - nx(elem,pt)*Bz(elem,pt);
          AD nce_z = nx(elem,pt)*By(elem,pt) - ny(elem,pt)*Bx(elem,pt);
          AD c0 = nce_x*wts(elem,pt);
          AD c1 = nce_y*wts(elem,pt);
          AD c2 = nce_z*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += c0*basis(elem,dof,pt,0) + c1*basis(elem,dof,pt,1) + c2*basis(elem,dof,pt,2);
          }
        }
      });
       */
       
    }
    
  }
  
}

// ========================================================================================
// ========================================================================================

void mirage::setWorkset(Teuchos::RCP<workset> & wkset_) {
  
  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "E")
      Enum = i;
    if (varlist[i] == "B")
      Bnum = i;
    //if (varlist[i] == "E2")
    //  E2num = i;
    //if (varlist[i] == "B2")
    //  B2num = i;
  }
}

// ========================================================================================
// ========================================================================================

void mirage::planewaveSource() {
  
  Teuchos::TimeMonitor resideval(*planewaveTimer);
  
  double time = wkset->time;
  double signal = 0;
  
  if (current_cont_wave_) {
    // Planewave waveform with Gaussian ramp-up (continuous cosine wave).
    double atmax = (time >= current_offset_) ? 1.0 : 0.0;
    signal = current_amplitude_ * std::cos(  2.0*PI*current_fr_center_*(time-current_offset_) )
                        * (std::exp( -2.0*std::pow(PI*current_sigma_*(time-current_offset_), 2) ) * (1-atmax) + atmax);
  }
  else {
    // Planewave waveform based on Gaussian pulse (windowed cosine wave).
    signal = current_amplitude_ * std::cos(  2.0*PI*current_fr_center_*(time-current_offset_) )
                        * std::exp( -2.0*std::pow(PI*current_sigma_*(time-current_offset_), 2) );
  }

  double xmin_ = current_xmin_, xmax_ = current_xmax_;
  double ymin_ = current_ymin_, ymax_ = current_ymax_;
  double zmin_ = current_zmin_, zmax_ = current_zmax_;
  
  if (wkset->dimension == 3) {
    auto ip_x = wkset->getDataSc("x");
    auto ip_y = wkset->getDataSc("y");
    auto ip_z = wkset->getDataSc("z");
    
    if (current_xcomponent_) {
      parallel_for("mirage current x",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type point = 0; point < current_x.extent(1); ++point) {
          const double x = ip_x(elem,point);
          const double y = ip_y(elem,point);
          const double z = ip_z(elem,point);
          current_x(elem,point) = 0.0;
          if ((x>xmin_) && (x<xmax_) && (y>ymin_) && (y<ymax_) && (z>zmin_) && (z<zmax_)) {
            current_x(elem,point) = signal;
          }
          current_y(elem,point) = 0.0;
          current_z(elem,point) = 0.0;
        }
      });
    }
    if (current_ycomponent_) {
      parallel_for("mirage current x",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type point = 0; point < current_x.extent(1); ++point) {
          const double x = ip_x(elem,point);
          const double y = ip_y(elem,point);
          const double z = ip_z(elem,point);
          current_y(elem,point) = 0.0;
          if ((x>xmin_) && (x<xmax_) && (y>ymin_) && (y<ymax_) && (z>zmin_) && (z<zmax_)) {
            current_y(elem,point) = signal;
          }
          current_x(elem,point) = 0.0;
          current_z(elem,point) = 0.0;
        }
      });
    }
    if (current_zcomponent_) {
      parallel_for("mirage current x",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type point = 0; point < current_x.extent(1); ++point) {
          const double x = ip_x(elem,point);
          const double y = ip_y(elem,point);
          const double z = ip_z(elem,point);
          current_z(elem,point) = 0.0;
          if ((x>xmin_) && (x<xmax_) && (y>ymin_) && (y<ymax_) && (z>zmin_) && (z<zmax_)) {
            current_z(elem,point) = signal;
          }
          current_x(elem,point) = 0.0;
          current_y(elem,point) = 0.0;
        }
      });
    }
  }
  else {
    auto ip_x = wkset->getDataSc("x");
    auto ip_y = wkset->getDataSc("y");
    if (current_xcomponent_) {
      parallel_for("mirage current x",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type point = 0; point < current_x.extent(1); ++point) {
          const double x = ip_x(elem,point);
          const double y = ip_y(elem,point);
          current_x(elem,point) = 0.0;
          if ((x>xmin_) && (x<xmax_) && (y>ymin_) && (y<ymax_) ) {
            current_x(elem,point) = signal;
          }
          current_y(elem,point) = 0.0;
        }
      });
    }
    if (current_ycomponent_) {
      parallel_for("mirage current x",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type point = 0; point < current_x.extent(1); ++point) {
          const double x = ip_x(elem,point);
          const double y = ip_y(elem,point);
          current_y(elem,point) = 0.0;
          if ((x>xmin_) && (x<xmax_) && (y>ymin_) && (y<ymax_) ) {
            current_y(elem,point) = signal;
          }
          current_x(elem,point) = 0.0;
        }
      });
    }
  }
}

// ========================================================================================
// ========================================================================================

void mirage::isotropicPML() {
  
  Teuchos::TimeMonitor resideval(*iPMLTimer);
  
  int dimension = wkset->dimension;
  
  auto ip_x = wkset->getDataSc("x");
  auto ip_y = wkset->getDataSc("y");
  View_Sc2 ip_z;
  if (dimension>2) {
    ip_z = wkset->getDataSc("z");
  }
  
  using namespace std;
  
  if (iPML_type == "exponential") {
  
    parallel_for("mirage current x",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type point = 0; point < iPML.extent(1); ++point) {
        double alpha(0.0), s(0.0);
        
        const double x = ip_x(elem,point);
        const double y = ip_y(elem,point);
        
        double sigmaplus = 0.0;
        if (iPML_have_xmax) {
          if (x > iPML_xmax_start) {
            alpha = (log(iPML_sigma_xmax)-log(iPML_tol_xmax))/(iPML_xmax_end-iPML_xmax_start);
            s = iPML_xmax_start - log(iPML_tol_xmax)/alpha;
            sigmaplus += exp(alpha*(x-s));
          }
        }
        if (iPML_have_xmin) {
          if (x < iPML_xmin_start) {
            alpha = (log(iPML_sigma_xmin)-log(iPML_tol_xmin))/(iPML_xmin_end-iPML_xmin_start);
            s = iPML_xmin_start - log(iPML_tol_xmin)/alpha;
            sigmaplus += exp(alpha*(x-s));
          }
        }
        if (iPML_have_ymax) {
          if (y > iPML_ymax_start) {
            alpha = (log(iPML_sigma_ymax)-log(iPML_tol_ymax))/(iPML_ymax_end-iPML_ymax_start);
            s = iPML_ymax_start - log(iPML_tol_ymax)/alpha;
            sigmaplus += exp(alpha*(y-s));
          }
        }
        if (iPML_have_ymin) {
          if (y < iPML_ymin_start) {
            alpha = (log(iPML_sigma_ymin)-log(iPML_tol_ymin))/(iPML_ymin_end-iPML_ymin_start);
            s = iPML_ymin_start - log(iPML_tol_ymin)/alpha;
            sigmaplus += exp(alpha*(y-s));
          }
        }
        if (iPML_have_zmax) {
          const double z = ip_z(elem,point);
          if (z > iPML_zmax_start) {
            alpha = (log(iPML_sigma_zmax)-log(iPML_tol_zmax))/(iPML_zmax_end-iPML_zmax_start);
            s = iPML_zmax_start - log(iPML_tol_zmax)/alpha;
            sigmaplus += exp(alpha*(z-s));
          }
        }
        if (iPML_have_zmin) {
          const double z = ip_z(elem,point);
          if (z < iPML_zmin_start) {
            alpha = (log(iPML_sigma_zmin)-log(iPML_tol_zmin))/(iPML_zmin_end-iPML_zmin_start);
            s = iPML_zmin_start - log(iPML_tol_zmin)/alpha;
            sigmaplus += exp(alpha*(z-s));
          }
        }
        iPML(elem,point) = iPML_sigma + sigmaplus;
      }
    });
  }
  else if (iPML_type == "polynomial") {
    parallel_for("mirage current x",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type point = 0; point < iPML.extent(1); ++point) {
        double alpha(0.0);
        
        const double x = ip_x(elem,point);
        const double y = ip_y(elem,point);
        
        double sigmaplus = 0.0;
    
        if (iPML_have_xmax) {
          if (x > iPML_xmax_start) {
            alpha = iPML_sigma_xmax/pow(abs(iPML_xmax_end-iPML_xmax_start), iPML_pow_xmax);
            sigmaplus += alpha*pow(abs(x-iPML_xmax_start), iPML_pow_xmax);
          }
        }
        if (iPML_have_xmin) {
          if (x < iPML_xmin_start) {
            alpha = iPML_sigma_xmin/pow(abs(iPML_xmin_end-iPML_xmin_start), iPML_pow_xmin);
            sigmaplus += alpha*pow(abs(x-iPML_xmin_start), iPML_pow_xmin);
          }
        }
        if (iPML_have_ymax) {
          if (y > iPML_xmax_start) {
            alpha = iPML_sigma_ymax/pow(abs(iPML_ymax_end-iPML_ymax_start), iPML_pow_ymax);
            sigmaplus += alpha*pow(abs(y-iPML_ymax_start), iPML_pow_ymax);
          }
        }
        if (iPML_have_ymin) {
          if (y < iPML_xmin_start) {
            alpha = iPML_sigma_ymin/pow(abs(iPML_ymin_end-iPML_ymin_start), iPML_pow_ymin);
            sigmaplus += alpha*pow(abs(y-iPML_ymin_start), iPML_pow_ymin);
          }
        }
        if (iPML_have_zmax) {
          const double z = ip_z(elem,point);
          if (z > iPML_zmax_start) {
            alpha = iPML_sigma_zmax/pow(abs(iPML_zmax_end-iPML_zmax_start), iPML_pow_zmax);
            if (iPML_zmax_exclude) {
              if (!( (x>iPML_zmax_x1) && (x<iPML_zmax_x2) && (y>iPML_zmax_y1) && (y<iPML_zmax_y2) )) {
                sigmaplus += alpha*pow(abs(z-iPML_zmax_start), iPML_pow_zmax);
              }
            } else {
              if ( (x>iPML_zmax_x1) && (x<iPML_zmax_x2) && (y>iPML_zmax_y1) && (y<iPML_zmax_y2) ) {
                sigmaplus += alpha*pow(abs(z-iPML_zmax_start), iPML_pow_zmax);
              }
            }
          }
        }
        if (iPML_have_zmin) {
          const double z = ip_z(elem,point);
          if (z < iPML_zmin_start) {
            alpha = iPML_sigma_zmin/pow(abs(iPML_zmin_end-iPML_zmin_start), iPML_pow_zmin);
            if (iPML_zmin_exclude) {
              if (!( (x>iPML_zmin_x1) && (x<iPML_zmin_x2) && (y>iPML_zmin_y1) && (y<iPML_zmin_y2) )) {
                sigmaplus += alpha*pow(abs(z-iPML_zmin_start), iPML_pow_zmin);
              }
            } else {
              if ( (x>iPML_zmin_x1) && (x<iPML_zmin_x2) && (y>iPML_zmin_y1) && (y<iPML_zmin_y2) ) {
                sigmaplus += alpha*pow(abs(z-iPML_zmin_start), iPML_pow_zmin);
              }
            }
          }
        }
        iPML(elem,point) = iPML_sigma + sigmaplus;
        
      }
      
    });
  }
}

// ========================================================================================
// ========================================================================================

void mirage::anisotropicPML() {
  
  Teuchos::TimeMonitor resideval(*aPMLTimer);
  
  int dimension = wkset->dimension;
  
  auto ip_x = wkset->getDataSc("x");
  auto ip_y = wkset->getDataSc("y");
  View_Sc2 ip_z;
  if (dimension>2) {
    ip_z = wkset->getDataSc("z");
  }
  
  using namespace std;
  
  parallel_for("mirage current x",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type point = 0; point < iPML.extent(1); ++point) {
      double alpha(0.0), s(0.0);
      
      const double x = ip_x(elem,point);
      const double y = ip_y(elem,point);
      
      double sigmaplusx = 0.0;
      double sigmaplusy = 0.0;
      double sigmaplusz = 0.0;
      
      if (aPML_type == "exponential") {
        if (aPML_have_xmax) {
          if (x > aPML_xmax_start) {
            alpha = (log(aPML_sigma_xmax)-log(aPML_tol_xmax))/(aPML_xmax_end-aPML_xmax_start);
            s = aPML_xmax_start - log(aPML_tol_xmax)/alpha;
            sigmaplusx += exp(alpha*(x-s));
          }
        }
        if (aPML_have_xmin) {
          if (x < aPML_xmin_start) {
            alpha = (log(aPML_sigma_xmin)-log(aPML_tol_xmin))/(aPML_xmin_end-aPML_xmin_start);
            s = aPML_xmin_start - log(aPML_tol_xmin)/alpha;
            sigmaplusx += exp(alpha*(x-s));
          }
        }
        if (aPML_have_ymax) {
          if (y > aPML_ymax_start) {
            alpha = (log(aPML_sigma_ymax)-log(aPML_tol_ymax))/(aPML_ymax_end-aPML_ymax_start);
            s = aPML_ymax_start - log(aPML_tol_ymax)/alpha;
            sigmaplusy += exp(alpha*(y-s));
          }
        }
        if (aPML_have_ymin) {
          if (y < aPML_ymin_start) {
            alpha = (log(aPML_sigma_ymin)-log(aPML_tol_ymin))/(aPML_ymin_end-aPML_ymin_start);
            s = aPML_ymin_start - log(aPML_tol_ymin)/alpha;
            sigmaplusy += exp(alpha*(y-s));
          }
        }
        if (aPML_have_zmax) {
          const double z = ip_z(elem,point);
          if (z > aPML_zmax_start) {
            alpha = (log(aPML_sigma_zmax)-log(aPML_tol_zmax))/(aPML_zmax_end-aPML_zmax_start);
            s = aPML_zmax_start - log(aPML_tol_zmax)/alpha;
            sigmaplusz += exp(alpha*(z-s));
          }
        }
        if (aPML_have_zmin) {
          const double z = ip_z(elem,point);
          if (z < aPML_zmin_start) {
            alpha = (log(aPML_sigma_zmin)-log(aPML_tol_zmin))/(aPML_zmin_end-aPML_zmin_start);
            s = aPML_zmin_start - log(aPML_tol_zmin)/alpha;
            sigmaplusz += exp(alpha*(z-s));
          }
        }
      }
      else if (aPML_type == "polynomial") {
        if (aPML_have_xmax) {
          if (x > aPML_xmax_start) {
            alpha = aPML_sigma_xmax/pow(abs(aPML_xmax_end-aPML_xmax_start), aPML_pow_xmax);
            sigmaplusx += alpha*pow(abs(x-aPML_xmax_start), aPML_pow_xmax);
          }
        }
        if (aPML_have_xmin) {
          if (x < aPML_xmin_start) {
            alpha = aPML_sigma_xmin/pow(abs(aPML_xmin_end-aPML_xmin_start), aPML_pow_xmin);
            sigmaplusx += alpha*pow(abs(x-aPML_xmin_start), aPML_pow_xmin);
          }
        }
        if (aPML_have_ymax) {
          if (y > aPML_ymax_start) {
            alpha = aPML_sigma_ymax/pow(abs(aPML_ymax_end-aPML_ymax_start), aPML_pow_ymax);
            sigmaplusy += alpha*pow(abs(y-aPML_ymax_start), aPML_pow_ymax);
          }
        }
        if (aPML_have_ymin) {
          if (y < aPML_ymin_start) {
            alpha = aPML_sigma_ymin/pow(abs(aPML_ymin_end-aPML_ymin_start), aPML_pow_ymin);
            sigmaplusy += alpha*pow(abs(y-aPML_ymin_start), aPML_pow_ymin);
          }
        }
        if (aPML_have_zmax) {
          const double z = ip_z(elem,point);
          if (z > aPML_zmax_start) {
            alpha = aPML_sigma_zmax/pow(abs(aPML_zmax_end-aPML_zmax_start), aPML_pow_zmax);
            if (aPML_zmax_exclude) {
              if (!( (x>aPML_zmax_x1) && (x<aPML_zmax_x2) && (y>aPML_zmax_y1) && (y<aPML_zmax_y2) )) {
                sigmaplusz += alpha*pow(abs(z-aPML_zmax_start), aPML_pow_zmax);
              }
            } else {
              if ( (x>aPML_zmax_x1) && (x<aPML_zmax_x2) && (y>aPML_zmax_y1) && (y<aPML_zmax_y2) ) {
                sigmaplusz += alpha*pow(abs(z-aPML_zmax_start), aPML_pow_zmax);
              }
            }
          }
        }
        if (aPML_have_zmin) {
          const double z = ip_z(elem,point);
          if (z < aPML_zmin_start) {
            alpha = aPML_sigma_zmin/pow(abs(aPML_zmin_end-aPML_zmin_start), aPML_pow_zmin);
            if (aPML_zmin_exclude) {
              if (!( (x>aPML_zmin_x1) && (x<aPML_zmin_x2) && (y>aPML_zmin_y1) && (y<aPML_zmin_y2) )) {
                sigmaplusz += alpha*pow(abs(z-aPML_zmin_start), aPML_pow_zmin);
              }
            } else {
              if ( (x>aPML_zmin_x1) && (x<aPML_zmin_x2) && (y>aPML_zmin_y1) && (y<aPML_zmin_y2) ) {
                sigmaplusz += alpha*pow(abs(z-aPML_zmin_start), aPML_pow_zmin);
              }
            }
          }
        }
      }
      
      if (dimension == 2) {
        aPML_xx(elem,point) = aPML_sigma + sigmaplusy;
        aPML_yy(elem,point) = aPML_sigma + sigmaplusx + sigmaplusy;
      }
      else if (dimension == 3) {
        aPML_xx(elem,point) = aPML_sigma + sigmaplusz;
        aPML_yy(elem,point) = aPML_sigma + sigmaplusx + sigmaplusy + sigmaplusz;
        aPML_zz(elem,point) = aPML_sigma + sigmaplusx + sigmaplusy + sigmaplusz;
      }
    }
  });
}

// ========================================================================================
// ========================================================================================
