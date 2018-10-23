/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MILOHELP_H
#define MILOHELP_H

#include "trilinos.hpp"
#include "preferences.hpp"

#include "userInterface.hpp"
#include "cell.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "solverInterface.hpp"
#include "postprocessInterface.hpp"
#include "analysisInterface.hpp"
#include "multiscaleInterface.hpp"
#include "uqInterface.hpp"

class MILOHelp {
public:
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  static void printHelp(const string & helpwhat, const string & details) {

    if (helpwhat == "help") {
      cout << endl;
      cout << "********** Help and Documentation for --help **********" << endl << endl;
      cout << "Purpose: To provide the user with information about some of the core functions" << endl;
      cout << "         within MILO without requiring the user to dig into the code." << endl << endl;
      cout << "Usage: mpiexec -n 1 milo --help helpwhat details" << endl;
      cout << "Options:" << endl;
      cout << "helpwhat: get information about a particular interface/tool.  Examples include user, cell, mesh, discretization, physics, solver, postprocess, analysis, multiscale, UQ, help (default) " << endl;
      cout << "details: get specific details about a topic within an interface/tool." << endl << endl;
      cout << "Contact: tmwilde@sandia.gov" << endl << endl;
    }
    else if (helpwhat == "user") {
      userHelp(details);
    }
    else if (helpwhat == "cell") {
      cellHelp(details);
    }
    else if (helpwhat == "mesh") {
      meshHelp(details);
    }
    else if (helpwhat == "discretization") {
      discretizationHelp(details);
    }
    else if (helpwhat == "physics") {
      physicsHelp(details);
    }
    else if (helpwhat == "solver") {
      solverHelp(details);
    }
    else if (helpwhat == "postprocess") {
      postprocessHelp(details);
    }
    else if (helpwhat == "analysis") {
      analysisHelp(details);
    }
    else if (helpwhat == "UQ") {
      uqHelp(details);
    }
    else if (helpwhat == "multiscale") {
      multiscaleHelp(details);
    }
    
    
  }
  
  };
#endif
