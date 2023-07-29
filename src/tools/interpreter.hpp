/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
 ************************************************************************/

#ifndef MRHYDE_INTERPRETER
#define MRHYDE_INTERPRETER

#include "trilinos.hpp"
#include "preferences.hpp"
#include "dag.hpp"

#include <stdio.h>
#include <ctype.h>

namespace MrHyDE {
  
  class Interpreter {
  public:
    
    Interpreter() {};
    
    ~Interpreter() {};
        
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    bool isScalar(const string & s);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    void split(vector<Branch> & branches, const size_t & index);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    bool isOperator(vector<Branch> & branches, size_t & index, vector<string> & ops);
    
  };
  
}
#endif

