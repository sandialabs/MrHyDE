/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef TIMEINTEGRATOR_H
#define TIMEINTEGRATOR_H

#include "trilinos.hpp"
#include "preferences.hpp"

// Base class for time integration methods in MILO

class TimeIntegrator {
public:
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Generic constructor/destructor
  ///////////////////////////////////////////////////////////////////////////////////////
  
  TimeIntegrator() {} ;
  
  ~TimeIntegrator() {};
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Combine the stage solution to compute the end-node solution
  ///////////////////////////////////////////////////////////////////////////////////////
  
  virtual void computeSolution(vector_RCP & stage_sol, vector_RCP & sol);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the stage time
  ///////////////////////////////////////////////////////////////////////////////////////
  
  virtual double computeTime(const double & prevtime, const size_t snum, const double & deltat);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Public data
  ///////////////////////////////////////////////////////////////////////////////////////
  
  // Butcher tableau information (btab_bs for embedded methods)
  Kokkos::View<double*,HostDevice> btab_b, btab_bs, btab_c;
  Kokkos::View<double**,HostDevice> btab_a;
  
  size_t num_stages;
  bool sol_staggered = true; // determines how solution is layed out for LA objects (staggered or blocked)
  
};
#endif
  
