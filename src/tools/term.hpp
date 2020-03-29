/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MILO_TERM_H
#define MILO_TERM_H

#include "trilinos.hpp"
#include "preferences.hpp"

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
  
  FDATA data;
  FDATAd ddata;
  Kokkos::View<double*,AssemblyDevice> scalar_ddata;
  Kokkos::View<AD*,AssemblyDevice> scalar_data;
  
  vector<int> dep_list;
  vector<string> dep_ops;
  
  
};
#endif

