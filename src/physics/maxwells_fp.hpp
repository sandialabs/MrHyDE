/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MAXWELLS_FP_H
#define MAXWELLS_FP_H

#include "physics_base.hpp"

static void maxwells_fpHelp() {
  cout << "********** Help and Documentation for the Maxwells Vector Potential Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class maxwells_fp : public physicsbase{
public:
  
  maxwells_fp() {};
  ~maxwells_fp() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  maxwells_fp(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
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
  
  AD getFreq(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
  
  // ========================================================================================
  // return magnetic permeability
  // ========================================================================================
  
  vector<AD> getPermeability(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
  
  // ========================================================================================
  // return inverse of magnetic permeability
  // ========================================================================================
  
  vector<AD> getInvPermeability(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
  
  // ========================================================================================
  // return electric permittivity
  // ========================================================================================
  
  vector<AD> getPermittivity(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
  
  // ========================================================================================
  // return current density in interior of domain
  // ========================================================================================
  
  vector<vector<AD> > getInteriorCurrent(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
  
  // ========================================================================================
  // return charge density in interior of domain
  // ========================================================================================
  
  vector<AD> getInteriorCharge(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
  
  // =======================================================================================
  // return electric current on boundary of domain
  // =======================================================================================
  
  vector<vector<AD> > getBoundaryCurrent(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time,
                                         const string & side_name, const int & boundary_type) const;
  
  // ========================================================================================
  // return charge density on boundary of domain (should be surface divergence of boundary current divided by i*omega
  // ========================================================================================
  
  vector<AD> getBoundaryCharge(const ScalarT & x, const ScalarT & y, const ScalarT & z, const ScalarT & time) const;
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_);
  
  // ========================================================================================
  // TMW: this needs to be deprecated
  // ========================================================================================
  
  void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params, const std::vector<string> & paramnames);
  
  // ========================================================================================
  // ========================================================================================
  
  
private:
  
  size_t numip, numip_side, numElem;
  
  vector<AD> mu_params; //permeability
  vector<AD> eps_params; //permittivity
  vector<AD> freq_params; //frequency
  vector<AD> source_params, boundary_params;
  
  int spaceDim;
  int Axr_num, phir_num, Ayr_num, Azr_num,
  Axi_num, phii_num, Ayi_num, Azi_num;
  
  int verbosity, test;
  
  Kokkos::View<ScalarT***,AssemblyDevice> Erx, Ery, Erz, Eix, Eiy, Eiz; //corresponding electric field
  bool calcE; //whether to calculate E field here (does not give smooth result like Paraview does; cause unknown)
  
  ScalarT essScale;
  
  //Kokkos::View<int****,AssemblyDevice> sideinfo;
  DRV phir_basis, phir_basis_grad;
  DRV phii_basis, phii_basis_grad;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::maxwells_fp::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::maxwells_fp::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::maxwells_fp::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::maxwells_fp::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::maxwells_fp::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::maxwells_fp::computeFlux() - evaluation of flux");
  
}; //end class

#endif
