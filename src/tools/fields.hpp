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
    
    // =================================================================
    // =================================================================
    
    SolutionField(const string & expression_,
                  const size_t & set_index_,
                  const string & vartype_,
                  const size_t & varindex_) {
                  
      expression = expression_;
      variable_type = vartype_; // solution, aux, param
      set_index = set_index_;
      variable_index = varindex_;
      
      // defaults
      derivative_type = ""; // grad, curl, div, time
      component = 0; //component_; // x, y, z
      isOnSide = false;
      isPoint = false;
      isUpdated = false;
      isInitialized = false;
      
      // Check if the field is on a side
      {
        size_t found = expression.find("side");
        if (found!=std::string::npos) {
          isOnSide = true;
        }
      }
      
      // Check if the field is point-wise
      {
        size_t found = expression.find("point");
        if (found!=std::string::npos) {
          isPoint = true;
        }
      }
      
      // Check if the field is a component of a vector
      {
        size_t xfound = expression.find("[x]");
        if (xfound!=std::string::npos) {
          component = 0;
        }
        
        size_t yfound = expression.find("[y]");
        if (yfound!=std::string::npos) {
          component = 1;
        }
        
        size_t zfound = expression.find("[z]");
        if (zfound!=std::string::npos) {
          component = 2;
        }
      }
    
      // Check if the field is a derivative
      {
        size_t gfound = expression.find("grad");
        if (gfound!=std::string::npos) {
          derivative_type = "grad";
        }
        
        size_t dfound = expression.find("div");
        if (dfound!=std::string::npos) {
          derivative_type = "div";
        }
        
        size_t cfound = expression.find("curl");
        if (cfound!=std::string::npos) {
          derivative_type = "curl";
        }
        
        size_t tfound = expression.find("_t");
        if (tfound!=std::string::npos) {
          derivative_type = "time";
        }
      }
      
    }
  
    // =================================================================
    // =================================================================
    
    void initialize(const int & dim0, const int & dim1) {
#ifndef MrHyDE_NO_AD
      data = View_AD2("solution field for " + expression, dim0, dim1, maxDerivs);
#else
      data = View_AD2("solution field for " + expression, dim0, dim1);
#endif
      isInitialized = true;
    }
    
    // =================================================================
    // =================================================================
    
    string expression, variable_type, basis_type, derivative_type;
    size_t set_index, variable_index, component;
    bool isUpdated, isOnSide, isPoint, isInitialized;
    View_AD2 data;
    
  };
  
  // =================================================================
  // =================================================================
  
  class ScalarField {
  public:
    
    ScalarField() {};
    
    ~ScalarField() {};
  
    // =================================================================
    // =================================================================
    
    ScalarField(const string & expression_) {
                
      expression = expression_;
      isOnSide = false;
      isPoint = false;
      isUpdated = false;
      isInitialized = false;
      
      // Check if the field is on a side
      {
        size_t found = expression.find("side");
        if (found!=std::string::npos) {
          isOnSide = true;
        }
      }
      
      // Check if the field is point-wise
      {
        size_t found = expression.find("point");
        if (found!=std::string::npos) {
          isPoint = true;
        }
      }
      
    }
    
    // =================================================================
    // =================================================================
    
    void initialize(const int & dim0, const int & dim1) {
      data = View_Sc2("scalar field for " + expression, dim0, dim1);
      isInitialized = true;
    }
    
    // =================================================================
    // =================================================================
    
    string expression;
    bool isUpdated, isOnSide, isPoint, isInitialized;
    View_Sc2 data;
    
  };
  
}

#endif
