/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_DEBUGGER_H
#define MRHYDE_DEBUGGER_H

#include "trilinos.hpp"

namespace MrHyDE {
  
  class MrHyDE_Debugger {
  public:
    
    MrHyDE_Debugger() {} ;
    
    ~MrHyDE_Debugger() {} ;
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    MrHyDE_Debugger(const int & debug_level, const Teuchos::RCP<MpiComm> & comm);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void print(const string & message);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    void print(const int & threshhold, const string & message);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    template<class T>
    void print(T view, const string & message);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    
    template<class T>
    void print(const int & threshhold, T data, const string & message);
    
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    ///
  private:
    
    int debug_level_;
    Teuchos::RCP<MpiComm> comm_;
    
  };
}

#endif
