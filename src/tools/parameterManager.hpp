/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef PARAMETER_H
#define PARAMETER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "cell.hpp"
#include "physicsInterface.hpp"


void static parameterHelp(const string & details) {
  cout << "********** Help and Documentation for the Parameter Manager **********" << endl;
}

class ParameterManager {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  ParameterManager(const Teuchos::RCP<LA_MpiComm> & Comm_,
                   Teuchos::RCP<Teuchos::ParameterList> & settings,
                   Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                   Teuchos::RCP<physics> & phys_,
                   vector<vector<Teuchos::RCP<cell> > > & cells,
                   vector<vector<Teuchos::RCP<BoundaryCell> > > & boundaryCells);
  
  // ========================================================================================
  // Set up the parameters (inactive, active, stochastic, discrete)
  // Communicate these parameters back to the physics interface and the enabled modules
  // ========================================================================================
  
  void setupParameters(Teuchos::RCP<Teuchos::ParameterList> & settings);
  
  void setupDiscretizedParameters(vector<vector<Teuchos::RCP<cell> > > & cells,
                                  vector<vector<Teuchos::RCP<BoundaryCell> > > & boundaryCells);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  int getNumParams(const int & type);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  int getNumParams(const std::string & type);
  
  // ========================================================================================
  // return the discretized parameters as vector for use with ROL
  // ========================================================================================
  
  vector<ScalarT> getDiscretizedParamsVector();
  
  // ========================================================================================
  // ========================================================================================
  
  void sacadoizeParams(const bool & seed_active);
  
  // ========================================================================================
  // ========================================================================================
  
  void updateParams(const vector<ScalarT> & newparams, const int & type);
  
  // ========================================================================================
  // ========================================================================================
  
  void updateParams(const vector<ScalarT> & newparams, const std::string & stype);
  
  // ========================================================================================
  // ========================================================================================
  
  vector<ScalarT> getParams(const int & type);
  
  // ========================================================================================
  // ========================================================================================
  
  vector<string> getParamsNames(const int & type);
  
  // ========================================================================================
  // ========================================================================================
  
  vector<size_t> getParamsLengths(const int & type);
  
  // ========================================================================================
  // ========================================================================================
  
  vector<ScalarT> getParams(const std::string & stype);
  
  // ========================================================================================
  // ========================================================================================
  
  vector<vector<ScalarT> > getParamBounds(const std::string & stype);
  
  // ========================================================================================
  // ========================================================================================
  
  void stashParams();
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP setInitialParams();
  
  // ========================================================================================
  // ========================================================================================
  
  vector<ScalarT> getStochasticParams(const std::string & whichparam);

  // ========================================================================================
  // ========================================================================================
  
  vector<ScalarT> getFractionalParams(const std::string & whichparam);
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  vector<string> blocknames;
  int spaceDim;
  
  Teuchos::RCP<const LA_Map> param_owned_map;
  Teuchos::RCP<const LA_Map> param_overlapped_map;
  
  Teuchos::RCP<LA_Export> param_exporter;
  Teuchos::RCP<LA_Import> param_importer;
  
  vector<string> paramnames;
  vector<vector<ScalarT> > paramvals;
  vector<Teuchos::RCP<vector<AD> > > paramvals_AD;
  Kokkos::View<AD**,AssemblyDevice> paramvals_KVAD;
  
  vector<vector_RCP> Psol;
  vector<vector_RCP> auxsol;
  vector<vector_RCP> dRdP;
  bool have_dRdP;
  Teuchos::RCP<const panzer::DOFManager> discparamDOF;
  vector<vector<ScalarT> > paramLowerBounds;
  vector<vector<ScalarT> > paramUpperBounds;
  vector<string> discretized_param_basis_types;
  vector<int> discretized_param_basis_orders, discretized_param_usebasis;
  vector<string> discretized_param_names;
  vector<basis_RCP> discretized_param_basis;
  Teuchos::RCP<panzer::DOFManager> paramDOF;
  vector<vector<int> > paramoffsets;
  vector<int> paramNumBasis;
  int numParamUnknowns;     					 // total number of unknowns
  int numParamUnknownsOS;     					 // total number of unknowns
  int globalParamUnknowns; // total number of unknowns across all processors
  vector<GO> paramOwned;					 // GIDs that live on the local processor.
  vector<GO> paramOwnedAndShared;				 // GIDs that live or are shared on the local processor.
  
  vector<int> paramtypes;
  vector<vector<int>> paramNodes;  // for distinguishing between parameter fields when setting initial
  vector<vector<int>> paramNodesOS;// values and bounds
  int num_inactive_params, num_active_params, num_stochastic_params, num_discrete_params, num_discretized_params;
  vector<ScalarT> initialParamValues, lowerParamBounds, upperParamBounds, discparamVariance;
  vector<ScalarT> domainRegConstants, boundaryRegConstants;
  vector<string> boundaryRegSides;
  vector<int> domainRegTypes, domainRegIndices, boundaryRegTypes, boundaryRegIndices;
  int verbosity;
  string response_type, multigrid_type, smoother_type;
  bool discretized_stochastic, use_custom_initial_param_guess;
  
  vector<string> stochastic_distribution, discparam_distribution;
  vector<ScalarT> stochastic_mean, stochastic_variance, stochastic_min, stochastic_max;
 
  vector<Teuchos::RCP<workset> > wkset;
  
  int batchID;
  
  //fractional parameters
  vector<ScalarT> s_exp;
  vector<ScalarT> h_mesh;
  
private:
  
  Teuchos::RCP<LA_MpiComm> Comm;
  Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
  Teuchos::RCP<discretization> disc;
  Teuchos::RCP<physics> phys;
  
  /* // Timers
  Teuchos::RCP<Teuchos::Time> assemblytimer = Teuchos::TimeMonitor::getNewCounter("MILO::ParameterManager::timer _1 - description");
  */
};

#endif
