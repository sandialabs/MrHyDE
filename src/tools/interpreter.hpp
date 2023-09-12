/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
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
  
  template<class EvalT>
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
    
    void split(vector<Branch<EvalT> > & branches, const size_t & index);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    bool isOperator(vector<Branch<EvalT> > & branches, size_t & index, vector<string> & ops);
    
  };
  
}
#endif

