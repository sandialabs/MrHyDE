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
  
  linearelasticity(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
                   const size_t & numip_side_, const int & numElem_,
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
  
  void setLocalSoln(const size_t & e, const size_t & ipindex, const bool & onside);
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_);
  
  // ========================================================================================
  // ========================================================================================
  
  void setAuxVars(std::vector<string> & auxvarlist);
  
  // ========================================================================================
  // return the stress
  // ========================================================================================
  
  void computeStress(const bool & onside);
  
  // ========================================================================================
  /* return the SIPG / IIPG term for a given node and component at an integration point */
  // ========================================================================================
  
  AD computeBasisVec(const AD dx, const AD dy, const AD dz, const AD mu_val, const AD lambda_val,
                     const DRV normals, DRV basis_grad, const int num_basis,
                     const int & elem, const int inode, const int k, const int component);
  
  // ========================================================================================
  // TMW: needs to be deprecated
  // ========================================================================================
  
  void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                        const vector<string> & paramnames);
  
  
private:
  
  size_t numip, numip_side;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int dx_num, dy_num, dz_num, e_num, p_num;
  int auxdx_num = -1, auxdy_num = -1, auxdz_num = -1, auxe_num = -1, auxp_num = -1;
  int test, simNum, cell_num;
  string response_type;
  // Parameters
  //    ScalarT lambda, mu;
  
  ScalarT v, dvdx, dvdy, dvdz;
  int resindex, dx_basis, dy_basis, dz_basis;
  ScalarT time;
  ScalarT x,y,z;
  
  // The notation here is a little unfortunate for the derivatives
  // Attempting to make it clearer by using dvar_dx where var = {dx,dy,dz}
  AD dx, ddx_dx, ddx_dy, ddx_dz;
  AD dy, ddy_dx, ddy_dy, ddy_dz;
  AD dz, ddz_dx, ddz_dy, ddz_dz;
  
  AD dpdx, dpdy, dpdz, eval, delta_e, pval;
  AD plambdax, plambday, plambdaz;
  
  //Kokkos::View<AD**,AssemblyDevice> lambda, mu, source_dx, source_dy, source_dz;
  //Kokkos::View<AD**,AssemblyDevice> lambda_side, mu_side, sourceN_dx, sourceN_dy, sourceN_dz;
  
  FDATA lambda, mu, source_dx, source_dy, source_dz;
  FDATA lambda_side, mu_side, sourceN_dx, sourceN_dy, sourceN_dz;
  
  Kokkos::View<AD****,AssemblyDevice> stress;
  
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  bool multiscale, useLame, addBiot, useCE;
  bool incplanestress;
  bool disp_response_type;
  ScalarT formparam, biot_alpha, e_ref, alpha_T, epen;
  
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
