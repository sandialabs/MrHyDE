/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef DISCINTERFACE_H
#define DISCINTERFACE_H

#include "trilinos.hpp"
#include "Panzer_DOFManager.hpp"

#include "preferences.hpp"
#include "cell.hpp"
#include "boundaryCell.hpp"

void static discretizationHelp(const string & details) {
  cout << "********** Help and Documentation for the Discretization Interface **********" << endl;
}

class discretization {
public:
  
  discretization() {} ;
  
  discretization(Teuchos::RCP<Teuchos::ParameterList> & settings,
                 Teuchos::RCP<LA_MpiComm> & Comm_,
                 Teuchos::RCP<panzer_stk::STK_Interface> & mesh,
                 vector<vector<int> > & orders, vector<vector<string> > & types,
                 vector<vector<Teuchos::RCP<cell> > > & cells);
  
  void setIntegrationInfo(vector<vector<Teuchos::RCP<cell> > > & cells,
                          vector<vector<Teuchos::RCP<BoundaryCell> > > & boundaryCells,
                          Teuchos::RCP<panzer::DOFManager> & DOF,
                          Teuchos::RCP<physics> & phys);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Public data
  ////////////////////////////////////////////////////////////////////////////////
  
  int milo_debug_level;
  Teuchos::RCP<LA_MpiComm> Commptr;
  
  vector<vector<basis_RCP> > basis_pointers;
  vector<vector<string> > basis_types;
  
  vector<DRV> ref_ip, ref_wts, ref_side_ip, ref_side_wts;
  vector<size_t> numip, numip_side;
  
  vector<vector<int> > cards;
  vector<vector<size_t> > myElements;
  
  
};

#endif
