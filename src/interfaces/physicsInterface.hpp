/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef PHYSICS_H
#define PHYSICS_H

#include "trilinos.hpp"
#include "preferences.hpp"

#include "physics_base.hpp"
#include "workset.hpp"

#include "Panzer_STK_Interface.hpp"
#include "Panzer_DOFManager.hpp"

static void physicsHelp(const string & details) {
  
  if (details == "none") {
    cout << "********** Help and Documentation for the Physics Interface **********" << endl;
  }
  /*
  else if (details == "porousHDIV") {
    porousHDIVHelp();
  }
  else if (details == "thermal") {
    thermalHelp();
  }
  else if (details == "thermal_enthalpy") {
    thermal_enthalpyHelp();
  }
  else if (details == "msphasefield") {
    msphasefieldHelp();
  }
  else if (details == "stokes") {
    stokesHelp();
  }
  else if (details == "navierstokes") {
    navierstokesHelp();
  }
  else if (details == "linearelasticity") {
    linearelasticityHelp();
  }
  else if (details == "helmholtz") {
    helmholtzHelp();
  }
  else if (details == "maxwells_fp") {
    maxwells_fpHelp();
  }
  else if (details == "shallowwater") {
    shallowwaterHelp();
  }
  else {
    cout << "Physics module help: unrecognized details: " << details << endl;
  }
   */
}

class physics {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  physics() {} ;
  
  physics(Teuchos::RCP<Teuchos::ParameterList> & settings, Teuchos::RCP<MpiComm> & Comm_,
          vector<topo_RCP> & cellTopo, vector<topo_RCP> & sideTopo,
          Teuchos::RCP<panzer_stk::STK_Interface> & mesh);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  // Add the requested physics modules, variables, discretization types 
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void importPhysics();
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  // Add the functions to the function managers
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void defineFunctions(vector<Teuchos::RCP<FunctionManager> > & functionManagers_);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  int getvarOwner(const int & block, const string & var);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  AD getDirichletValue(const int & block, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                       const ScalarT & t, const string & var, const string & gside,
                       const bool & useadjoint, Teuchos::RCP<workset> & wkset);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  ScalarT getInitialValue(const int & block, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                         const string & var, const bool & useadjoint);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  int getNumResponses(const int & block, const string & var);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  int getNumResponses(const int & block);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD***,AssemblyDevice> getPointResponse(const int & block,
                                                      Kokkos::View<AD****,AssemblyDevice> u_ip,
                                                      Kokkos::View<AD****,AssemblyDevice> ugrad_ip,
                                                      Kokkos::View<AD****,AssemblyDevice> p_ip,
                                                      Kokkos::View<AD****,AssemblyDevice> pgrad_ip,
                                                      const DRV ip, const ScalarT & time,
                                                      Teuchos::RCP<workset> & wkset);
  
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD***,AssemblyDevice> getResponse(const int & block,
                                                 Kokkos::View<AD****,AssemblyDevice> u_ip,
                                                 Kokkos::View<AD****,AssemblyDevice> ugrad_ip,
                                                 Kokkos::View<AD****,AssemblyDevice> p_ip,
                                                 Kokkos::View<AD****,AssemblyDevice> pgrad_ip,
                                                 const DRV ip, const ScalarT & time,
                                                 Teuchos::RCP<workset> & wkset);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  AD computeTopoResp(const size_t & block);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD***,AssemblyDevice> target(const int & block, const DRV & ip,
                                            const ScalarT & current_time,
                                            Teuchos::RCP<workset> & wkset);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<AD***,AssemblyDevice> weight(const int & block, const DRV & ip,
                                            const ScalarT & current_time,
                                            Teuchos::RCP<workset> & wkset);

  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////

  Kokkos::View<ScalarT***,AssemblyDevice> getInitial(const DRV & ip,
                                                     const int & block,
                                                     const bool & project,
                                                     Teuchos::RCP<workset> & wkset);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT**,AssemblyDevice> getDirichlet(const DRV & ip, const int & var,
                                                       const int & block,
                                                       const std::string & sidename,
                                                       Teuchos::RCP<workset> & wkset);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void setVars(size_t & block, vector<string> & vars);
  
  void setAuxVars(size_t & block, vector<string> & vars);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<string> getResponseFieldNames(const int & block);

  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<string> getExtraFieldNames(const int & block);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  vector<string> getExtraCellFieldNames(const int & block);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  vector<Kokkos::View<ScalarT***,AssemblyDevice> > getExtraFields(const int & block);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT**,AssemblyDevice> getExtraFields(const int & block,
                                                         const int & fnum,
                                                         const DRV & ip,
                                                         const ScalarT & time,
                                                         Teuchos::RCP<workset> & wkset);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<ScalarT*,AssemblyDevice> getExtraCellFields(const int & block,
                                                           const int & fnum,
                                                           DRV wts);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  int getUniqueIndex(const int & block, const std::string & var);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void setBCData(Teuchos::RCP<Teuchos::ParameterList> & settings,
                 Teuchos::RCP<panzer_stk::STK_Interface> & mesh,
                 Teuchos::RCP<panzer::DOFManager> & DOF,
                 std::vector<std::vector<int> > cards);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void setDirichletData(Teuchos::RCP<panzer_stk::STK_Interface> & mesh,
                        Teuchos::RCP<panzer::DOFManager> & DOF);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<int****,HostDevice> getSideInfo(const size_t & block, Kokkos::View<int*,HostDevice> elem);

  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  vector<vector<int> > getOffsets(const int & block, Teuchos::RCP<panzer::DOFManager> & DOF);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Kokkos::View<int**,HostDevice> getSideInfo(const int & block, int & num, size_t & e);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  //void setPeriBCs(Teuchos::RCP<Teuchos::ParameterList> & settings, Teuchos::RCP<panzer_stk::STK_Interface> & mesh);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void volumeResidual(const size_t block);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void boundaryResidual(const size_t block);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void computeFlux(const size_t block);
   
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void setWorkset(vector<Teuchos::RCP<workset> > & wkset);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////

  bool checkFace(const size_t & block);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void faceResidual(const size_t block);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  /////////////////////////////////////////////////////////////////////////////////////////////

  vector<vector<Teuchos::RCP<physicsbase> > > modules;
  vector<Teuchos::RCP<FunctionManager> > functionManagers;
  Teuchos::RCP<MpiComm> Commptr;
  vector<Teuchos::ParameterList> blockPhysSettings, blockDiscSettings;
  vector<string> blocknames, sideNames;
  int spaceDim, milo_debug_level;
  size_t numBlocks;
  Teuchos::RCP<Teuchos::ParameterList> settings;
  
  
  vector<int> numVars;
  vector<vector<bool> > useSubgrid;
  vector<vector<bool> > useDG;
  bool haveDirichlet;
  
  vector<vector<string> > varlist;
  vector<vector<int> > varowned;
  vector<vector<int> > orders;
  vector<vector<string> > types;
  vector<vector<int> > unique_orders;
  vector<vector<string> > unique_types;
  vector<vector<int> > unique_index;
  
  vector<Kokkos::View<int****,HostDevice> > side_info;
  //vector<vector<vector<size_t> > > localDirichletSideIDs, globalDirichletSideIDs;
  //vector<vector<vector<size_t> > > boundDirichletElemIDs;
  vector<vector<GO> > point_dofs;
  vector<vector<vector<LO> > > dbc_dofs;
  
  vector<Kokkos::View<int**,HostDevice> > var_bcs;
  
  //vector<FCint> offsets;
  vector<vector<vector<int> > > offsets;
  vector<string> sideSets;
  vector<string> nodeSets;
  int numSidesPerElem, numNodesPerElem;
  vector<size_t> numElem;
  string initial_type, cellfield_reduction;
  
  vector<vector<string> > extrafields_list, extracellfields_list, response_list, target_list, weight_list;
  
  Teuchos::RCP<Teuchos::Time> bctimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics::setBCData()");
  Teuchos::RCP<Teuchos::Time> dbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics::setDirichletData()");
  Teuchos::RCP<Teuchos::Time> sideinfotimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics::getSideInfo()");
  Teuchos::RCP<Teuchos::Time> responsetimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics:computeResponse()");
  Teuchos::RCP<Teuchos::Time> pointreponsetimer = Teuchos::TimeMonitor::getNewCounter("MILO::physics::computePointResponse()");
  
};


#endif
