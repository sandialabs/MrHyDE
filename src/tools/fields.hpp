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
  
  template<class EvalT>
  class SolutionField {
    
    //friend class Workset<EvalT>;
    //friend class FunctionManager;

    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_AD2;
  
  public:
    
    SolutionField() {};
    
    ~SolutionField() {};
    
    // =================================================================
    // =================================================================
    
    SolutionField(const string & expression,
                  const size_t & set_index,
                  const string & vartype,
                  const size_t & varindex) {
                  
      expression_ = expression;
      variable_type_ = vartype; // solution, aux, param
      set_index_ = set_index;
      variable_index_ = varindex;
      
      // defaults
      derivative_type_ = ""; // grad, curl, div, time
      component_ = 0; //component_; // x, y, z
      is_updated_ = false;
      is_initialized_ = false;
      
      // Check if the field is a component of a vector
      {
        size_t xfound = expression_.find("[x]");
        if (xfound!=std::string::npos) {
          component_ = 0;
        }
        
        size_t yfound = expression_.find("[y]");
        if (yfound!=std::string::npos) {
          component_ = 1;
        }
        
        size_t zfound = expression_.find("[z]");
        if (zfound!=std::string::npos) {
          component_ = 2;
        }
      }
    
      // Check if the field is a derivative
      {
        size_t gfound = expression_.find("grad");
        if (gfound!=std::string::npos) {
          derivative_type_ = "grad";
        }
        
        size_t dfound = expression_.find("div");
        if (dfound!=std::string::npos) {
          derivative_type_ = "div";
        }
        
        size_t cfound = expression_.find("curl");
        if (cfound!=std::string::npos) {
          derivative_type_ = "curl";
        }
        
        size_t tfound = expression_.find("_t");
        if (tfound!=std::string::npos) {
          derivative_type_ = "time";
        }
      }
      
    }
  
    // =================================================================
    // =================================================================
    
    void initialize(const int & dim0, const int & dim1) {
#ifndef MrHyDE_NO_AD
      data_ = View_AD2("solution field for " + expression_, dim0, dim1, maxDerivs);
#else
      data_ = View_AD2("solution field for " + expression_, dim0, dim1);
#endif
      is_initialized_ = true;
    }
    
    // =================================================================
    // =================================================================
    
  //private:
    string expression_, variable_type_, basis_type_, derivative_type_;
    size_t set_index_, variable_index_, component_;
    bool is_updated_, is_initialized_;
    View_AD2 data_;
    
  };
  
  // =================================================================
  // =================================================================
  
  class ScalarField {
    //friend class Workset;
    //friend class FunctionManager;
  public:
    
    ScalarField() {};
    
    ~ScalarField() {};
  
    // =================================================================
    // =================================================================
    
    ScalarField(const string & expression) {
                
      expression_ = expression;
      is_updated_ = false;
      is_initialized_ = false;
      
    }
    

    // =================================================================
    // =================================================================
    
    void initialize(const int & dim0, const int & dim1) {
      data_ = View_Sc2("scalar field for " + expression_, dim0, dim1);
      is_initialized_ = true;
    }
    
    // =================================================================
    // =================================================================
    
    string expression_;
    bool is_updated_, is_initialized_;
    View_Sc2 data_;
    
  };
  
}

#endif
