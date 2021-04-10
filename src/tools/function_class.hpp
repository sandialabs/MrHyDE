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

#ifndef MRHYDE_FUNCTION_H
#define MRHYDE_FUNCTION_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "term.hpp"

namespace MrHyDE {
  
  class function_class {
  public:
    
    function_class() {};
    
    function_class(const string & name, const string & expression_,
                   const size_t & dim0_, const size_t & dim1_,
                   const string & location_) :
    dim0(dim0_), dim1(dim1_), function_name(name), expression(expression_), location(location_) {
      
      term newt = term(expression);
      terms.push_back(newt);
      //terms = Kokkos::View<term*,AssemblyDevice>(expression, 1);
      
    } ;
    
    ~function_class() {};
    
    //////////////////////////////////////////////////////////////////////
    // Public data members
    //////////////////////////////////////////////////////////////////////
    
    size_t dim0, dim1;
    vector<term> terms;
    //Kokkos::View<term*,AssemblyDevice> terms;
    string function_name, expression, location;
    bool isScalar, isStatic, onSide;
    
  };
  
}

#endif

