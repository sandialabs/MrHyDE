/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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
  
  template<class EvalT>
  class Branch {

    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    
  public:
    
    Branch() {};
    
    ~Branch() {};
    
    Branch(const string & expression) {
      expression_ = expression;
      // default settings are all false
      is_leaf_ = false;
      is_decomposed_ = false;
      is_func_ = false;
      is_view_ = false;
      is_AD_ = false;
      is_workset_data_ = false;
      is_constant_ = false;
      is_parameter_ = false;
      is_time_ = false;
      
      func_index_ = 0;
      param_index_ = 0;
      
      workset_data_index_ = 0;
    }
  
    // ========================================================================================
    // ========================================================================================
  
    Branch(const ScalarT value) {
      std::stringstream stream;
      stream << std::fixed << std::setprecision(16) << value;
      expression_ = stream.str();
      
      // default settings are all false
      is_leaf_ = true;
      is_decomposed_ = true;
      is_func_ = false;
      is_view_ = false;
      is_AD_ = false;
      is_workset_data_ = false;
      is_constant_ = true;
      is_parameter_ = false;
      is_time_ = false;
      
      func_index_ = 0;
      param_index_ = 0;
      
      workset_data_index_ = 0;
      
      data_Sc_ = value;
    }
    
    // ========================================================================================
    // ========================================================================================
  
    void print() {
    
      std::cout << "-- Printing metadata for branch: " << expression_ << std::endl;
      std::cout << "------ is_leaf_: "        << is_leaf_ << std::endl;
      std::cout << "------ is_decomposed_: "  << is_decomposed_ << std::endl;
      std::cout << "------ is_func_: "        << is_func_ << std::endl;
      std::cout << "------ is_view_: "        << is_view_ << std::endl;
      std::cout << "------ is_AD_: "          << is_AD_ << std::endl;
      std::cout << "------ is_workset_data_: " << is_workset_data_ << std::endl;
      std::cout << "------ is_constant_: "    << is_constant_ << std::endl;
      std::cout << "------ is_parameter_: "   << is_parameter_ << std::endl;
      std::cout << "------ is_time_: "        << is_time_ << std::endl;
    }
  
    // ========================================================================================
    // ========================================================================================
  
    string expression_;
    bool is_leaf_, is_decomposed_, is_func_, is_view_, is_AD_, is_constant_, is_workset_data_, is_parameter_, is_time_;
    int func_index_, param_index_, workset_data_index_;
    
    // Various data storage types
    // Only one of these will get used
    View_EvalT2 viewdata_;
    View_Sc2 viewdata_Sc_;
    ScalarT data_Sc_;
    EvalT data_;
    Kokkos::View<EvalT*,Kokkos::LayoutStride,AssemblyDevice> param_data_;
    //Kokkos::View<EvalT*,AssemblyDevice> param_data_;
        
    vector<int> dep_list_, dep_ops_int_;
    vector<string> dep_ops_;
    
  };
  
  // =================================================================
  // Trees have branches
  // =================================================================
  
  template<class EvalT>
  class Tree {
    
  public:
    
    // ========================================================================================
    // ========================================================================================
  
    Tree() {};
  
    // ========================================================================================
    // ========================================================================================
  
    ~Tree() {};
  
    // ========================================================================================
    // ========================================================================================
  
    Tree(const string & name, const string & expression) {
      name_ = name;
      expression_ = expression;
      branches_.push_back(Branch<EvalT>(expression));
    }
    
    // ========================================================================================
    // ========================================================================================
  
    Tree(const string & name, ScalarT & value) {
      name_ = name;
      std::stringstream stream;
      stream << std::fixed << std::setprecision(16) << value;
      expression_ = stream.str();
      branches_.push_back(Branch<EvalT>(value));
    }
  
    // ========================================================================================
    // ========================================================================================
  
    void setupVista() {
      if (branches_.size() > 0) {
        if (branches_[0].is_view_) {
          if (branches_[0].is_AD_) {
            vista_ = Vista<EvalT>(branches_[0].viewdata_);
          }
          else {
            vista_ = Vista<EvalT>(branches_[0].viewdata_Sc_);
          }
        }
        else {
          if (branches_[0].is_AD_) {
            //vista_ = Vista<EvalT>(branches_[0].data_);
          }
          else {
            vista_ = Vista<EvalT>(branches_[0].data_Sc_);
          }
        }
      }
    }
    
    // ========================================================================================
    // ========================================================================================
  
    void updateVista() {
      if (branches_[0].is_AD_) {
        if (branches_[0].is_view_) {
          if (branches_[0].is_parameter_) {
            int pind = branches_[0].param_index_;
            auto pdata = branches_[0].param_data_;
            EvalT pval = pdata(pind); // Yes, I know this won't work on a GPU
            #ifndef MrHyDE_NO_AD
              vista_.update(pval);
            #else
              vista_.update(pval);
            #endif
          }
          else {
            vista_.update(branches_[0].viewdata_);
          }
        }
        else {
          #ifndef MrHyDE_NO_AD
            vista_.update(branches_[0].data_);
          #else
            vista_.updateSc(branches_[0].data_);
          #endif
        }
      }
      else {
        if (branches_[0].is_view_) {
          vista_.update(branches_[0].viewdata_Sc_);
        }
        else {
          vista_.updateSc(branches_[0].data_Sc_);
        }
      }
    }
    
    // ========================================================================================
    // ========================================================================================
  
    std::vector<Branch<EvalT> > branches_;
    string name_, expression_;
    Vista<EvalT> vista_;
    
  };
  
  // =================================================================
  // Forests have trees and are associated with a location (enables multiple forests)
  // =================================================================
  
  template<class EvalT>
  class Forest {
    
  public:
    
    // ========================================================================================
    // ========================================================================================
  
    Forest() {};
  
    // ========================================================================================
    // ========================================================================================
  
    ~Forest() {};
  
    // ========================================================================================
    // ========================================================================================
  
    Forest(const std::string & location, const int & dim0, const int & dim1){
      location_ = location;
      dim0_ = dim0;
      dim1_ = dim1;
    }
    
    // ========================================================================================
    // ========================================================================================
  
    void addTree(const string & name, const string & expression) {
      trees_.push_back(Tree<EvalT>(name,expression));
    }
  
    // ========================================================================================
    // ========================================================================================
  
    void addTree(const string & name, ScalarT & value) {
      trees_.push_back(Tree<EvalT>(name,value));
    }
    
    // ========================================================================================
    // ========================================================================================
  
    std::string location_;
    int dim0_, dim1_;
    std::vector<Tree<EvalT> > trees_;
    
  };
  
}

#endif

