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

#ifndef MRHYDE_INTERPRETER
#define MRHYDE_INTERPRETER

#include "trilinos.hpp"
#include "preferences.hpp"
#include "term.hpp"

#include <stdio.h>
#include <ctype.h>

namespace MrHyDE {
  
  class Interpreter {
  public:
    
    Interpreter() {};
    
    ~Interpreter() {};
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////
    
    vector<string> getVars(const string & s, const vector<string> & knownops);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    int validateTerms(const vector<string> & terms,
                      const vector<string> & known_vars,
                      const vector<string> & variables,
                      const vector<string> & parameters,
                      const vector<string> & disc_parameters,
                      const vector<string> & functions);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    bool isScalar(const string & s);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    void split(vector<term> & terms, const size_t & index);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    bool isOperator(vector<term> & terms, size_t & index, vector<string> & ops);
    
  };
  
}
#endif

