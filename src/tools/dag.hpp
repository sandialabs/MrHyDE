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

#ifndef MRHYDE_DAG_H
#define MRHYDE_DAG_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // New data structures that have a branch \in tree \in forest hierarchy
  // =================================================================
  
  class Branch { // replaces term
  public:
    
    Branch() {};
    
    ~Branch() {};
    
    Branch(const string & expression_) {
      expression = expression_;
      // default settings are all false
      isLeaf = false;
      isDecomposed = false;
      isFunc = false;
      isView = false;
      isAD = false;
      isWorksetData = false;
      isConstant = false;
      isParameter = false;
      isTime = false;
      
      funcIndex = 0;
      paramIndex = 0;
      
      workset_data_index = 0;
    }
    
    void print() {
    
      std::cout << "-- Printing metadata for branch: " << expression << std::endl;
      std::cout << "------ isLeaf: "        << isLeaf << std::endl;
      std::cout << "------ isDecomposed: "  << isDecomposed << std::endl;
      std::cout << "------ isFunc: "        << isFunc << std::endl;
      std::cout << "------ isView: "        << isView << std::endl;
      std::cout << "------ isAD: "          << isAD << std::endl;
      std::cout << "------ isWorksetData: " << isWorksetData << std::endl;
      std::cout << "------ isConstant: "    << isConstant << std::endl;
      std::cout << "------ isParameter: "   << isParameter << std::endl;
      std::cout << "------ isTime: "        << isTime << std::endl;
    }
    
    
    string expression;
    bool isLeaf, isDecomposed, isFunc, isView, isAD, isConstant, isWorksetData, isParameter, isTime;
    int funcIndex, paramIndex, workset_data_index;
    
    // Various data storage types
    // Only one of these will get used
    View_AD2 viewdata;
    View_Sc2 viewdata_Sc;
    ScalarT data_Sc;
    AD data;
    Kokkos::View<AD*,Kokkos::LayoutStride,AssemblyDevice> param_data;
        
    vector<int> dep_list, dep_ops_int;
    vector<string> dep_ops;
    
  };
  
  // =================================================================
  // Trees have branches
  // =================================================================
  
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
  
  // =================================================================
  // Forests have trees and are associated with a location (enables multiple forests)
  // =================================================================
  
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

