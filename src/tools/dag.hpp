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
#include "vista.hpp"

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
    
    Branch(const ScalarT value_) {
      std::stringstream stream;
      stream << std::fixed << std::setprecision(16) << value_;
      expression = stream.str();
      
      // default settings are all false
      isLeaf = true;
      isDecomposed = true;
      isFunc = false;
      isView = false;
      isAD = false;
      isWorksetData = false;
      isConstant = true;
      isParameter = false;
      isTime = false;
      
      funcIndex = 0;
      paramIndex = 0;
      
      workset_data_index = 0;
      
      data_Sc = value_;
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
    
    Tree(const string & name_, ScalarT & value_) {
      name = name_;
      std::stringstream stream;
      stream << std::fixed << std::setprecision(16) << value_;
      expression = stream.str();
      branches.push_back(Branch(value_));
    }
    
    void setupVista() {
      if (branches.size() > 0) {
        if (branches[0].isView) {
          if (branches[0].isAD) {
            vista = Vista(branches[0].viewdata);
          }
          else {
            vista = Vista(branches[0].viewdata_Sc);
          }
        }
        else {
          if (branches[0].isAD) {
            vista = Vista(branches[0].data);
          }
          else {
            vista = Vista(branches[0].data_Sc);
          }
        }
      }
    }
    
    void updateVista() {
      if (branches[0].isAD) {
        if (branches[0].isView) {
          if (branches[0].isParameter) {
            int pind = branches[0].paramIndex;
            auto pdata = branches[0].param_data;
            AD pval = pdata(pind); // Yes, I know this won't work on a GPU
            vista.update(pval);
          }
          else {
            vista.update(branches[0].viewdata);
          }
        }
        else {
          vista.update(branches[0].data);
        }
      }
      else {
        if (branches[0].isView) {
          vista.update(branches[0].viewdata_Sc);
        }
        else {
          vista.update(branches[0].data_Sc);
        }
      }
    }
    
    std::vector<Branch> branches;
    string name, expression;
    Vista vista;
    
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
    
    void addTree(const string & name, ScalarT & value) {
      trees.push_back(Tree(name,value));
    }
    
    std::string location;
    int dim0, dim1;
    std::vector<Tree> trees;
    
  };
  
}

#endif

