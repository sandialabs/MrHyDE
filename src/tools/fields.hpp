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

#ifndef MRHYDE_FIELDS_H
#define MRHYDE_FIELDS_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // =================================================================
  
  class SolutionField {
  public:
    
    SolutionField() {};
    
    ~SolutionField() {};
    
    SolutionField(const string & expression_,
                  const size_t & set_index_,
                  const string & vartype_,
                  const int & varindex_,
                  const string & basistype_,
                  const int & basis_index_,
                  const string & derivtype_,
                  const int & component_,
                  const int & dim0_,
                  const int & dim1_,
                  const bool & onSide_,
                  const bool & isPoint_) {
      
      expression = expression_;
      variable_type = vartype_; // solution, aux, param
      set_index = set_index_;
      variable_index = varindex_;
      basis_type = basistype_; // HGRAD, HVOL, HDIV, HCURL, HFACE
      basis_index = basis_index_;
      derivative_type = derivtype_; // grad, curl, div, time
      component = component_; // x, y, z
      isOnSide = onSide_;
      isPoint = isPoint_;
      isUpdated = false;
      isInitialized = false;
      dim1 = dim1_;
      
    }
    
    void initialize(const int & dim0) {
      data = View_AD2("solution field for " + expression, dim0, dim1);
      isInitialized = true;
    }
    
    string expression, variable_type, basis_type, derivative_type;
    size_t set_index;
    int variable_index, basis_index, component, dim1;
    bool isUpdated, isOnSide, isPoint, isInitialized;
    View_AD2 data;
    
  };
  
  class ScalarField {
  public:
    
    ScalarField() {};
    
    ~ScalarField() {};
    
    ScalarField(const string & expression_,
                const int & dim0_,
                const int & dim1_,
                const bool & onSide_,
                const bool & isPoint_) {
      
      expression = expression_;
      isOnSide = onSide_;
      isPoint = isPoint_;
      isUpdated = false;
      isInitialized = false;
      dim1 = dim1_;
      
    }
    
    void initialize(const int & dim0) {
      data = View_Sc2("scalar field for " + expression, dim0, dim1);
      isInitialized = true;
    }
    
    string expression;
    int dim1;
    bool isUpdated, isOnSide, isPoint, isInitialized;
    View_Sc2 data;
    
  };
  
}

#endif
