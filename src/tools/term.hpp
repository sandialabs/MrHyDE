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

#ifndef FUNCTION_TERM_H
#define FUNCTION_TERM_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  class term {
  public:
    
    term() {};
    
    term(const string & expr) {
      
      // default settings
      isRoot = false;
      beenDecomposed = false;
      isFunc = false;
      isScalar = false;
      //isAD = true;
      isConstant = false;
      scalarIndex = 0;
      
      expression = expr;
      
    } ;
    
    ~term() {};
    
    
    void print() {
      cout << endl;
      cout << "--------------------------------------------------" << endl;
      cout << "expression: " << expression << endl;
      cout << "isAD: " << isAD << endl;
      cout << "isScalar: " << isScalar << endl;
      cout << "scalarIndex: " << scalarIndex << endl;
      cout << "isConstant: " << isConstant << endl;
      cout << "isRoot: " << isRoot << endl;
      cout << "isFunc: " << isFunc << endl;
      cout << "beenDecomposed: " << beenDecomposed << endl;
      cout << "dep_list length: " << dep_list.size() << endl;
      for (size_t k=0; k<dep_list.size(); k++) {
        cout << "dep_list[" << k << "]: " << dep_list[k] << endl;
      }
      cout << "dep_ops length: " << dep_ops.size() << endl;
      for (size_t k=0; k<dep_ops.size(); k++) {
        cout << "dep_ops[" << k << "]: " << dep_ops[k] << endl;
      }
      cout << "data dims: " << data.extent(0) << "  " << data.extent(1) << endl;
      cout << "ddata dims: " << ddata.extent(0) << "  " << ddata.extent(1) << endl;
      cout << "--------------------------------------------------" << endl;
      
      cout << endl;
    }
    
    //////////////////////////////////////////////////////////////////////
    // Public data members
    //////////////////////////////////////////////////////////////////////
    
    string expression;
    bool isRoot, beenDecomposed, isFunc, isScalar, isAD, isConstant;
    int funcIndex, scalarIndex;
    
    View_AD2 data;
    View_Sc2 ddata;
    Kokkos::View<double*,Kokkos::LayoutStride,AssemblyDevice> scalar_ddata;
    Kokkos::View<AD*,Kokkos::LayoutStride,AssemblyDevice> scalar_data;
    
    vector<int> dep_list;
    vector<string> dep_ops;
    
    
  };
  
  // =================================================================
  // New data structures that have a branch \in tree \in forest hierarchy
  // =================================================================
  
  class Branch { // replaces term
  public:
    
    Branch() {};
    
    ~Branch() {};
    
    Branch(const string & expression_) {
      expression = expression_;
      // default settings
      isLeaf = false;
      isDecomposed = false;
      isFunc = false;
      isScalar = false;
      //isAD = true;
      isConstant = false;
      scalarIndex = 0;
    }
    
    string expression;
    bool isLeaf, isDecomposed, isFunc, isScalar, isAD, isConstant;
    int funcIndex, scalarIndex;
    
    View_AD2 data;
    View_Sc2 ddata;
    Kokkos::View<double*,Kokkos::LayoutStride,AssemblyDevice> scalar_ddata;
    Kokkos::View<AD*,Kokkos::LayoutStride,AssemblyDevice> scalar_data;
    
    vector<int> dep_list;
    vector<string> dep_ops;
    
  };
  
  class Tree { // replaces function_class
  public:
    
    Tree() {};
    
    ~Tree() {};
    
    Tree(const string & name_, const string & expression_) {
      name = name_;
      expression = expression_;
      branches.push_back(Branch(expression));
    }
    
    std::vector<Branch> branches;
    string name, expression;
    
  };
  
  class Forest {
  public:
    
    Forest() {};
    
    ~Forest() {};
    
    Forest(const std::string & location_, const int & dim0_, const int & dim1_){
      location = location_;
      dim0 = dim0_;
      dim1 = dim1_;
    }
    
    void addTree(const string & name, const string & expression) {
      trees.push_back(Tree(name,expression));
    }
    
    std::string location;
    int dim0, dim1;
    std::vector<Tree> trees;
    
  };
  
}

#endif

