/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef USERDEFBASE_H
#define USERDEFBASE_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"

class UserDefinedBase {
public:
  
  UserDefinedBase() {} ;
  
  ~UserDefinedBase() {};
  
  UserDefinedBase(Teuchos::RCP<Teuchos::ParameterList> & settings) {} ;
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  virtual
  Kokkos::View<AD*,AssemblyDevice> boundaryNeumannSource(const string & physics, const string & var, const Teuchos::RCP<workset> & wkset) {
    
    int numip = wkset->ip_side.dimension(1);
    Kokkos::View<AD*,AssemblyDevice> vals("neumann values",numip); //defaults to zeros
    
    // Specialize for physics, var and test
    
    return vals;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  virtual
  Kokkos::View<AD*,AssemblyDevice> boundaryRobinSource(const string & physics, const string & var, const Teuchos::RCP<workset> & wkset) {
    
    int numip = wkset->ip_side.dimension(1);
    Kokkos::View<AD*,AssemblyDevice> vals("robin values",numip); //defaults to zeros
    
    // Specialize for physics, var and test
    
    return vals;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  virtual
  Kokkos::View<AD*,AssemblyDevice> boundaryDirichletSource(const string & physics, const string & var, const Teuchos::RCP<workset> & wkset) {
    
    int numip = wkset->ip_side.dimension(1);
    Kokkos::View<AD*,AssemblyDevice> vals("dirichlet values",numip); //defaults to zeros
    
    // Specialize for physics, var and test
    
    return vals;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  virtual
  Kokkos::View<AD*,AssemblyDevice> volumetricSource(const string & physics, const string & var,
                                                    const Teuchos::RCP<workset> & wkset) {
    
    int numip = wkset->ip.dimension(1);
    Kokkos::View<AD*,AssemblyDevice> vals("source",numip); //defaults to zeros
    
    // Specialize for physics, var and test
    
    return vals;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  virtual
  Kokkos::View<AD*,AssemblyDevice> coefficient(const string & name,
                                               const Teuchos::RCP<workset> & wkset) {
    
    int numip = wkset->ip.dimension(1);
    Kokkos::View<AD*,AssemblyDevice> vals("coefficient values",numip); //defaults to zeros
    
    // Specialize for physics, var and test
    
    return vals;
    
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  virtual
  void updateParameters(const vector<vector<AD> > & params,
                        const vector<string> & paramnames) {
    
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Update the values of the parameters
  //////////////////////////////////////////////////////////////////////////////////////
  
  virtual
  vector<vector<double> > setInitialParams(const DRV & nodes,
                                            const vector<vector<int> > & indices) {
    vector<vector<double> > param_initial_vals;
    for (int n = 0; n < indices.size(); n++) {
      param_initial_vals.push_back(vector<double>(indices[n].size()));
    }
    return param_initial_vals;
  }
  
  
protected:
  
  ////////////////////////////
  // vectors of parameters (vector<AD>)
  ////////////////////////////
  
  
};
#endif
  
