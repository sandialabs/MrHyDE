/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef RK_TIMEINTEGRATOR_H
#define RK_TIMEINTEGRATOR_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "timeIntegrator.hpp"

// Base class for time integration methods in MILO

class RungeKutta : public TimeIntegrator {
public:
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Generic constructor/destructor
  ///////////////////////////////////////////////////////////////////////////////////////
  
  RungeKutta() {} ;
  
  ~RungeKutta() {};
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Constructor
  ///////////////////////////////////////////////////////////////////////////////////////
  
  RungeKutta(const string method_, const size_t & order_, const bool & sol_staggered_) :
  method(method_), order(order_), sol_staggered(sol_staggered_) {
    // Define the Butcher tableau and the number os stages based on the method and order
    if (method == "Explicit") {
      if (order == 1) { // Forward Euler
        num_stages = 1;
        btab_a = Kokkos::View<ScalarT**,HostDevice>("butcher tableau a",num_stages,num_stages);
        btab_b = Kokkos::View<ScalarT*,HostDevice>("butcher tableau b",num_stages);
        btab_c = Kokkos::View<ScalarT*,HostDevice>("butcher tableau c",num_stages);
        btab_a(0,0) = 0.0;
        btab_b(0) = 1.0;
        btab_c(0) = 0.0;
      }
      else if (order == 4) { // Classical RK4
        num_stages = 4;
        btab_a = Kokkos::View<ScalarT**,HostDevice>("butcher tableau a",num_stages,num_stages);
        btab_b = Kokkos::View<ScalarT*,HostDevice>("butcher tableau b",num_stages);
        btab_c = Kokkos::View<ScalarT*,HostDevice>("butcher tableau c",num_stages);
        btab_a(0,0) = 0.0; btab_a(0,1) = 0.0; btab_a(0,2) = 0.0; btab_a(0,3) = 0.0;
        btab_a(1,0) = 0.5; btab_a(1,1) = 0.0; btab_a(1,2) = 0.0; btab_a(1,3) = 0.0;
        btab_a(2,0) = 0.0; btab_a(2,1) = 0.5; btab_a(2,2) = 0.0; btab_a(2,3) = 0.0;
        btab_a(3,0) = 0.0; btab_a(3,1) = 0.0; btab_a(3,2) = 1.0; btab_a(3,3) = 0.0;
        btab_b(0) = 1.0/6.0; btab_b(1) = 1.0/3.0; btab_b(2) = 1.0/3.0; btab_b(3) = 1.0/6.0;
        btab_c(0) = 0.0; btab_c(1) = 0.5; btab_c(2) = 0.5; btab_c(3) = 1.0;
      }
    }
    else if (method == "Embedded") {
      if (order == 5) { // RK45
        num_stages = 6;
        btab_a = Kokkos::View<ScalarT**,HostDevice>("butcher tableau a",num_stages,num_stages);
        btab_b = Kokkos::View<ScalarT*,HostDevice>("butcher tableau b",num_stages);
        btab_c = Kokkos::View<ScalarT*,HostDevice>("butcher tableau c",num_stages);
        btab_a(0,0) = 0.0;           btab_a(0,1) = 0.0;            btab_a(0,2) = 0.0;            btab_a(0,3) = 0.0;           btab_a(0,4) = 0.0;        btab_a(0,5) = 0.0;
        btab_a(1,0) = 0.25;          btab_a(1,1) = 0.0;            btab_a(1,2) = 0.0;            btab_a(1,3) = 0.0;           btab_a(1,4) = 0.0;        btab_a(1,5) = 0.0;
        btab_a(2,0) = 3.0/32.0;      btab_a(2,1) = 9.0/32.0;       btab_a(2,2) = 0.0;            btab_a(2,3) = 0.0;           btab_a(2,4) = 0.0;        btab_a(2,5) = 0.0;
        btab_a(3,0) = 1932.0/2197.0; btab_a(3,1) = -7200.0/2197.0; btab_a(3,2) = 7296.0/2197.0;  btab_a(3,3) = 0.0;           btab_a(3,4) = 0.0;        btab_a(3,5) = 0.0;
        btab_a(4,0) = 439.0/216.0;   btab_a(4,1) = -8.0;           btab_a(4,2) = 3680.0/513.0;   btab_a(4,3) = -845.0/4104.0; btab_a(4,4) = 0.0;        btab_a(4,5) = 0.0;
        btab_a(5,0) = -8.0/27.0;     btab_a(5,1) = 2.0;            btab_a(5,2) = -3544.0/2565.0; btab_a(5,3) = 1859.0/4104.0; btab_a(5,4) = -11.0/40.0; btab_a(5,5) = 0.0;
        
        
        btab_b(0) = 16.0/35.0;   btab_b(1) = 0.0;  btab_b(2) = 6656.0/12825.0; btab_b(3) = 28561.0/56430.0; btab_b(4) = -9.0/50.0; btab_b(5) = 2.0/55.0;
        btab_bs(0) = 25.0/216.0; btab_bs(1) = 0.0; btab_bs(2) = 1408.0/2565.0; btab_bs(3) = 2197.0/4104.0;  btab_bs(4) = -1.0/5.0; btab_bs(5) = 0.0;
        
        btab_c(0) = 0.0; btab_c(1) = 0.25; btab_c(2) = 3.0/8.0; btab_c(3) = 12.0/13.0; btab_c(4) = 1.0; btab_c(5) = 0.5;
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Runge-Kutta method.");
      }
    }
    else if (method == "DIRK") { // Diagonally Implicit RK
      if (order == 1) { // Backward Euler
        num_stages = 1;
        btab_a = Kokkos::View<ScalarT**,HostDevice>("butcher tableau a",num_stages,num_stages);
        btab_b = Kokkos::View<ScalarT*,HostDevice>("butcher tableau b",num_stages);
        btab_c = Kokkos::View<ScalarT*,HostDevice>("butcher tableau c",num_stages);
        btab_a(0,0) = 1.0;
        btab_b(0) = 1.0;
        btab_c(0) = 1.0;
      }
      else if (order == 2) {
        num_stages = 2;
        btab_a = Kokkos::View<ScalarT**,HostDevice>("butcher tableau a",num_stages,num_stages);
        btab_b = Kokkos::View<ScalarT*,HostDevice>("butcher tableau b",num_stages);
        btab_c = Kokkos::View<ScalarT*,HostDevice>("butcher tableau c",num_stages);
        ScalarT gamma = (3.0+sqrt(3))/6.0;
        btab_a(0,0) = gamma;     btab_a(0,1) = 0.0;
        btab_a(1,0) = 1.0-gamma; btab_a(1,1) = gamma;
        btab_b(0) = 1.0-gamma; btab_b(1) = gamma;
        btab_c(0) = gamma; btab_c(1) = 1.0;
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Runge-Kutta method.");
      }
    }
    else if (method == "Implicit") { // General, dense btab_a (experimental)
      if (order == 1) { // Backward Euler
        num_stages = 1;
        btab_a = Kokkos::View<ScalarT**,HostDevice>("butcher tableau a",num_stages,num_stages);
        btab_b = Kokkos::View<ScalarT*,HostDevice>("butcher tableau b",num_stages);
        btab_c = Kokkos::View<ScalarT*,HostDevice>("butcher tableau c",num_stages);
        btab_a(0,0) = 1.0;
        btab_b(0) = 1.0;
        btab_c(0) = 1.0;
      }
      if (order == 2) { // Midpoint rule
        num_stages = 1;
        btab_a = Kokkos::View<ScalarT**,HostDevice>("butcher tableau a",num_stages,num_stages);
        btab_b = Kokkos::View<ScalarT*,HostDevice>("butcher tableau b",num_stages);
        btab_c = Kokkos::View<ScalarT*,HostDevice>("butcher tableau c",num_stages);
        btab_a(0,0) = 0.5;
        btab_b(0) = 0.5;
        btab_c(0) = 1.0;
      }
      else if (order == 4) {
        num_stages = 2;
        btab_a = Kokkos::View<ScalarT**,HostDevice>("butcher tableau a",num_stages,num_stages);
        btab_b = Kokkos::View<ScalarT*,HostDevice>("butcher tableau b",num_stages);
        btab_c = Kokkos::View<ScalarT*,HostDevice>("butcher tableau c",num_stages);
        btab_a(0,0) = 0.25;               btab_a(0,1) = 0.25-sqrt(3.0)/6.0;
        btab_a(1,0) = 0.25+sqrt(3.0)/6.0; btab_a(1,1) = 0.25;
        btab_b(0) = 0.5; btab_b(1) = 0.5;
        btab_c(0) = 0.5-sqrt(3)/6.0; btab_c(1) = 0.5+sqrt(3.0)/6.0;
      }
      else if (order == 6) {
        num_stages = 3;
        btab_a = Kokkos::View<ScalarT**,HostDevice>("butcher tableau a",num_stages,num_stages);
        btab_b = Kokkos::View<ScalarT*,HostDevice>("butcher tableau b",num_stages);
        btab_c = Kokkos::View<ScalarT*,HostDevice>("butcher tableau c",num_stages);
        btab_a(0,0) = 5.0/36.0;                 btab_a(0,1) = 2.0/9.0-sqrt(15.0)/15.0; btab_a(0,2) = 5.0/36.0-sqrt(15.0)/30.0;
        btab_a(1,0) = 5.0/36.0+sqrt(15.0)/24.0; btab_a(1,1) = 2.0/9.0;                 btab_a(1,2) = 5.0/36.0-sqrt(15.0)/24.0;
        btab_a(2,0) = 5.0/36.0+sqrt(15.0)/30.0; btab_a(2,1) = 2.0/9.0+sqrt(15.0)/15.0; btab_a(2,2) = 5.0/36.0;
        btab_b(0) = 5.0/18.0; btab_b(1) = 4.0/9.0; btab_b(2) = 5.0/18.0;
        btab_c(0) = 0.5-sqrt(15.0)/10.0; btab_c(1) = 0.5; btab_c(2) = 0.5+sqrt(15.0)/10.0;
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Runge-Kutta method.");
      }
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized Runge-Kutta method.");
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Combine the stage solution to compute the end-node solution
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeSolution(vector_RCP & stage_sol, vector_RCP & sol) {
    // sol is a vector of (local) length N
    // stage_sol is a vector of (local) length N*s
    // How the data gets collected from stage_sol into sol depends on whether the stages were
    // staggered (default) or blocked
    if (sol_staggered) {
      
    }
    else {
      
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the stage time
  ///////////////////////////////////////////////////////////////////////////////////////
  
  ScalarT computeTime(const ScalarT & prevtime, const size_t snum, const ScalarT & deltat) {
    return prevtime + btab_c(snum)*deltat;
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the stage solutions
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeStageSolutions(Kokkos::View<AD***,AssemblyDevice> & stage_sol, Kokkos::View<AD***,AssemblyDevice> & sol) {
    if (sol_staggered) {
      
    }
    else {
      
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Compute the stage solutions
  ///////////////////////////////////////////////////////////////////////////////////////
  
  void computeStageTimeDerivative(Kokkos::View<AD***,AssemblyDevice> & stage_soldot, Kokkos::View<AD***,AssemblyDevice> & soldot) {
    if (sol_staggered) {
      
    }
    else {
      
    }
  }
  
  ///////////////////////////////////////////////////////////////////////////////////////
  // Public data
  ///////////////////////////////////////////////////////////////////////////////////////
  
  string method;
  size_t order;
  
};
#endif
  
