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

/** \file   physicsInterface.hpp
 \brief  Contains the interface to the MrHyDE-specific physics modules.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_PHYSICSINTERFACE_H
#define MRHYDE_PHYSICSINTERFACE_H

#include "trilinos.hpp"
#include "preferences.hpp"

#include "physicsBase.hpp"
#include "workset.hpp"

#include "Panzer_STK_Interface.hpp"
#include "Panzer_DOFManager.hpp"

namespace MrHyDE {
  
  /** \class  MrHyDE::PhysicsInterface
   \brief  Interface to the MrHyDE-specific physics modules.  This is the only class that direcly interacts with the physics modules.
   */
  
  class PhysicsInterface {
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    PhysicsInterface() {} ;
    
    ~PhysicsInterface() {} ;
    
    PhysicsInterface(Teuchos::RCP<Teuchos::ParameterList> & settings, Teuchos::RCP<MpiComm> & Comm_,
                     Teuchos::RCP<panzer_stk::STK_Interface> & mesh);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Add the requested physics modules, variables, discretization types 
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void importPhysics();
    
    vector<string> breakupList(const string & list, const string & delimiter);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Add the functions to the function managers
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void defineFunctions(vector<Teuchos::RCP<FunctionManager> > & functionManagers_);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getvarOwner(const int & set, const int & block, const string & var);
    
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
    
    View_Sc3 getInitial(vector<View_Sc2> & pts, const int & set, const int & block,
                        const bool & project, Teuchos::RCP<workset> & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    /* @brief Evaluate the initial condition along the face integration point for L2 projection
     *
     * @param[in] pts  Face integration points
     * @param[in] block  Cell block
     * @param[in] project  Flag for L2 projection
     * @param[in] wkset  Workset
     * 
     * @returns View_Sc3 of the initial condition
     *
     * @warning BWR -- under development. Not sure what the nonprojection option is about.
     */
    
    View_Sc3 getInitialFace(vector<View_Sc2> & pts, const int & set, const int & block,
                            const bool & project, Teuchos::RCP<workset> & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    View_Sc2 getDirichlet(const int & var, const int & set,
                          const int & block, const std::string & sidename);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void setVars();
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getUniqueIndex(const int & set, const int & block, const std::string & var);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void volumeResidual(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void boundaryResidual(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void computeFlux(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void setWorkset(vector<Teuchos::RCP<workset> > & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    bool checkFace(const size_t & set, const size_t & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void faceResidual(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void fluxConditions(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void updateFlags(vector<bool> & newflags);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void purgeMemory();
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<Teuchos::ParameterList> settings;
    vector<Teuchos::RCP<FunctionManager> > functionManagers;
    Teuchos::RCP<MpiComm> Commptr;
    
    int spaceDim, debug_level;
    vector<string> setnames, blocknames, sidenames;
    
    vector<vector<size_t> > numVars; // [set][block]
    
    //-----------------------------------------------------
    // Data the depends on physics sets
    vector<vector<vector<Teuchos::RCP<physicsbase> > > > modules;
    
    vector<vector<Teuchos::ParameterList>> setPhysSettings, setDiscSettings, setSolverSettings; // [set][block]
    vector<vector<vector<bool> > > useSubgrid;
    vector<vector<vector<bool> > > useDG;
    vector<vector<vector<ScalarT> > > masswts, normwts;
    
    vector<vector<vector<string> > > varlist; // [set][block][var]
    vector<vector<vector<int> > > varowned; // [set][block][var]
    vector<vector<vector<int> > > orders; // [set][block][var]
    vector<vector<vector<string> > > types; // [set][block][var]
    //-----------------------------------------------------
    
    vector<vector<int> > unique_orders; // [block][basis]
    vector<vector<string> > unique_types; // [block][basis]
    vector<vector<int> > unique_index; // [block][basis]
    
    string initial_type;
    
    vector<vector<string> > extrafields_list, extracellfields_list, response_list, target_list, weight_list;
    
    Teuchos::RCP<Teuchos::Time> bctimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface::setBCData()");
    Teuchos::RCP<Teuchos::Time> dbctimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface::setDirichletData()");
    Teuchos::RCP<Teuchos::Time> sideinfotimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface::getSideInfo()");
    Teuchos::RCP<Teuchos::Time> responsetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface:computeResponse()");
    Teuchos::RCP<Teuchos::Time> pointreponsetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface::computePointResponse()");
    
  };
  
}

#endif
