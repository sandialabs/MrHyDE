/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SUBGRIDMODEL_H
#define SUBGRIDMODEL_H

#include "trilinos.hpp"
#include "preferences.hpp"

class SubGridModel {
public:
  
  SubGridModel() {} ;
  
  ~SubGridModel() {};
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  virtual int addMacro(const DRV macronodes_, Kokkos::View<int****,HostDevice> macrosideinfo_,
                       vector<string> & macrosidenames,
                       vector<int> & macroGIDs,
                       vector<vector<int> > & macroindex) = 0;

  virtual void subgridSolver(Kokkos::View<ScalarT***,AssemblyDevice> gl_u,
                             Kokkos::View<ScalarT***,AssemblyDevice> gl_phi,
                             const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                             const bool & compute_jacobian, const bool & compute_sens,
                             const int & num_active_params,
                             const bool & compute_disc_sens, const bool & compute_aux_sens,
                             workset & macrowkset, const int & macroelemindex,
                             const int & usernum,
                             Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) = 0;
  
  virtual Kokkos::View<ScalarT**,AssemblyDevice> computeError(const ScalarT & time,
                                                             const int & usernum) = 0;
  
  virtual Kokkos::View<AD*,AssemblyDevice> computeObjective(const string & response_type,
                                                            const int & seedwhat,
                                                            const ScalarT & time,
                                                            const int & usernum) = 0;
  
  virtual void writeSolution(const string & filename, const int & usernum) = 0;

  virtual void writeSolution(const string & filename) = 0;

  virtual void writeSolution(const string & filename, const int & usernum, const int & timeindex) = 0;

  virtual void addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points,
                          const ScalarT & sensor_loc_tol,
                          const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data,
                          const bool & have_sensor_data,
                          const vector<basis_RCP> & basisTypes, const int & usernum) = 0;
  
  virtual Teuchos::RCP<Epetra_CrsMatrix> getProjectionMatrix() = 0;
  
  virtual DRV getIP() = 0;
  
  virtual DRV getIPWts() = 0;
  
  virtual pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis(const DRV & ip) = 0;

  virtual pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis2(const DRV & ip) = 0;
  
  virtual Teuchos::RCP<Epetra_CrsMatrix> getEvaluationMatrix(const DRV & newip, Teuchos::RCP<Epetra_Map> & ip_map) = 0;
  
  virtual vector<vector<int> > getCellGIDs(const int & cellnum) = 0;
  
  virtual void solutionStorage(Teuchos::RCP<Epetra_MultiVector> & newvec,
                               const ScalarT & time, const bool & isAdjoint,
                               const int & usernum)= 0;
  
  virtual void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) = 0;
  
  virtual Kokkos::View<ScalarT**,AssemblyDevice> getCellFields(const int & usernum, const ScalarT & time) = 0;
  
  virtual void addMeshData() = 0;

  virtual void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) = 0;

  //virtual Epetra_MultiVector getVector() = 0;
  
  Teuchos::RCP<LA_MpiComm> LocalComm;
  vector<Teuchos::RCP<workset> > wkset;
  vector<basis_RCP> macro_basis_pointers;
  vector<string> macro_basis_types;
  vector<string> macro_varlist;
  vector<int> macro_usebasis;
  vector<vector<int> > macro_offsets;
  vector<string> macro_paramnames, macro_disc_paramnames;
  int macro_block;
  ScalarT cost_estimate;
  
  Teuchos::RCP<Epetra_Map> owned_map;
  Teuchos::RCP<Epetra_Map> overlapped_map;
  Teuchos::RCP<Epetra_Export> exporter;
  Teuchos::RCP<Epetra_Import> importer;
  
  vector<vector<pair<ScalarT, Teuchos::RCP<Epetra_MultiVector> > > > soln;
  vector<vector<pair<ScalarT, Teuchos::RCP<Epetra_MultiVector> > > > solndot;
  vector<vector<pair<ScalarT, Teuchos::RCP<Epetra_MultiVector> > > > adjsoln;
  
  vector<Teuchos::RCP<vector<AD> > > paramvals_AD;

  string usage;
  Kokkos::View<AD**,AssemblyDevice> paramvals_KVAD;
  
  
};
#endif
  
