/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MILO_FUNCTION_H
#define MILO_FUNCTION_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"
#include "term.hpp"

class function_class {
public:
  
  function_class() {};
  
  function_class(const string & name, const string & expression_,
                 const size_t & dim0_, const size_t & dim1_,
                 const string & location_) :
  function_name(name), expression(expression_), dim0(dim0_), dim1(dim1_), location(location_) {
    
    term newt = term(expression);
    terms.push_back(newt);
  } ;
  
  ~function_class() {};
  
  //////////////////////////////////////////////////////////////////////
  // Public data members
  //////////////////////////////////////////////////////////////////////
  
  size_t dim0, dim1;
  vector<term> terms;
  string function_name, expression, location;
  bool isScalar, isStatic, onSide;
  
};
#endif

