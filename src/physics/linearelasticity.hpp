/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef LINEARELAST_H
#define LINEARELAST_H

#include "physics_base.hpp"
#include "CrystalElasticity.hpp"
#include <string>

static void linearelasticityHelp() {
  cout << "********** Help and Documentation for the Linear Elasticity Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class linearelasticity : public physicsbase {
public:
  
  linearelasticity() {} ;
  
  ~linearelasticity() {} ;
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  linearelasticity(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  // ========================================================================================
  // ========================================================================================
  
  void defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
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
  
  //void setLocalSoln(const size_t & e, const size_t & ipindex, const bool & onside);
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_);
  
  // ========================================================================================
  // ========================================================================================
  
  void setAuxVars(std::vector<string> & auxvarlist);
  
  // ========================================================================================
  // return the stress
  // ========================================================================================
  
  void computeStress(FDATA lambda, FDATA mu, const bool & onside);
  
  // ========================================================================================
  /* return the SIPG / IIPG term for a given node and component at an integration point */
  // ========================================================================================
  
  //AD computeBasisVec(const AD dx, const AD dy, const AD dz, const AD mu_val, const AD lambda_val,
  //                   const DRV normals, DRV basis_grad, const int num_basis,
  //                  const int & elem, const int inode, const int k, const int component);
  
  // ========================================================================================
  // TMW: needs to be deprecated
  // ========================================================================================
  
  void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                        const vector<string> & paramnames);
  
  
private:
  
  int spaceDim;
  int dx_num, dy_num, dz_num, e_num, p_num;
  int auxdx_num = -1, auxdy_num = -1, auxdz_num = -1, auxe_num = -1, auxp_num = -1;
  
  Kokkos::View<AD****,AssemblyDevice> stress_vol, stress_side;
  
  bool useLame, addBiot, useCE, incplanestress, disp_response_type;
  //ScalarT formparam, biot_alpha, e_ref, alpha_T, epen;
  Kokkos::View<ScalarT*,AssemblyDevice> modelparams;
  
  Teuchos::RCP<CrystalElastic> crystalelast;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::computeFlux() - evaluation of flux");
  Teuchos::RCP<Teuchos::Time> setLocalSol = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::setLocalSoln()");
  Teuchos::RCP<Teuchos::Time> fillStress = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::computeStress()");
  Teuchos::RCP<Teuchos::Time> computeBasis = Teuchos::TimeMonitor::getNewCounter("MILO::elasticity::computeBasisVec()");
  
};

#endif
