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

/** \file   functionManager.hpp
 \brief  Contains the function manager which handles all of the user-defined or physics-defined functions.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_FUNCTION_MANAGER_H
#define MRHYDE_FUNCTION_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "dag.hpp"
#include "interpreter.hpp"
#include "workset.hpp"
#include "vista.hpp"

namespace MrHyDE {
  
  /** \class  MrHyDE::FunctionManager
   \brief  Provides the functionality that allows the user (or physics modules) to define arbitrarily
   complex functions without modifying the code.
   */
  
  class FunctionManager {
  public:
    
    FunctionManager();
    
    ~FunctionManager() {};
    
    FunctionManager(const std::string & blockname, const int & num_elem,
                    const int & num_ip, const int & num_ip_side);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Add a user defined function
    //////////////////////////////////////////////////////////////////////////////////////
    
    int addFunction(const std::string & fname, std::string & expression, const std::string & location);
    
    int addFunction(const string & fname, ScalarT & value, const string & location);

    //////////////////////////////////////////////////////////////////////////////////////
    // Set the lists of variables, parameters and discretized parameters
    //////////////////////////////////////////////////////////////////////////////////////
    
    void setupLists(const std::vector<std::string> & parameters);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Decompose the functions into terms and set the evaluation tree
    // Also sets up the Kokkos::Views (subviews) to the data for all of the terms
    //////////////////////////////////////////////////////////////////////////////////////
    
    void decomposeFunctions();
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Determine if a term is a ScalarT or needs to be an AD type
    //////////////////////////////////////////////////////////////////////////////////////
    
    bool isScalarTerm(const int & findex, const int & tindex, const int & bindex);
    
    void checkDepDataType(const int & findex, const int & tindex, const int & bindex,
                          bool & isCont, bool & isVector, bool & isAD);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Evaluate a function (probably will be deprecated)
    //////////////////////////////////////////////////////////////////////////////////////
    
    //View_AD2 evaluate(const std::string & fname, const std::string & location);
    
    Vista evaluate(const std::string & fname, const std::string & location);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Evaluate a function
    //////////////////////////////////////////////////////////////////////////////////////
    
    void evaluate(const size_t & findex, const size_t & tindex, const size_t & bindex);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Evaluate an operator
    //////////////////////////////////////////////////////////////////////////////////////
    
    template<class T1, class T2>
    void evaluateOpVToV(T1 data, T2 tdata, const std::string & op);
    
    template<class T1, class T2>
    void evaluateOpParamToV(T1 data, T2 tdata, const int & pIndex_, const std::string & op);
    
    template<class T1, class T2>
    void evaluateOpSToV(T1 data, T2 & tdata, const std::string & op);
    
    template<class T1, class T2>
    void evaluateOpSToS(T1 & data, T2 & tdata, const std::string & op);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Print out the function information (mostly for debugging)
    //////////////////////////////////////////////////////////////////////////////////////
    
    void printFunctions();
    
    //void setWorkset(Teuchos::RCP<workset> & wkset);

    //////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    //////////////////////////////////////////////////////////////////////////////////////
    
    int num_elem_, num_ip_, num_ip_side_;
    Teuchos::RCP<Workset> wkset;
    
  private:

    std::string blockname_;
    std::vector<Forest> forests_;
    std::vector<std::string> parameters_, disc_parameters_;
    std::vector<std::string> known_vars_, known_ops_;
    Teuchos::RCP<Interpreter> interpreter_;
  };
  
}
#endif


