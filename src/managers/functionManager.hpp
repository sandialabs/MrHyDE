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

#ifndef FUNCTION_MANAGER_H
#define FUNCTION_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "function_class.hpp"
#include "interpreter.hpp"

namespace MrHyDE {
  
  class FunctionManager {
  public:
    
    FunctionManager();
    
    FunctionManager(const std::string & blockname, const int & numElem_,
                    const int & numip_, const int & numip_side_);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Add a user defined function
    //////////////////////////////////////////////////////////////////////////////////////
    
    //int addFunction(const std::string & fname, const std::string & expression,
    //                const size_t & dim0, const size_t & dim1, const bool & onSide);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Add a user defined function
    //////////////////////////////////////////////////////////////////////////////////////
    
    int addFunction(const std::string & fname, std::string & expression, const std::string & location);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Set the lists of variables, parameters and discretized parameters
    //////////////////////////////////////////////////////////////////////////////////////
    
    void setupLists(const std::vector<std::string> & variables_,
                    const std::vector<std::string> & aux_variables_,
                    const std::vector<std::string> & parameters_,
                    const std::vector<std::string> & disc_parameters_);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Validate all of the functions
    //////////////////////////////////////////////////////////////////////////////////////
    
    void validateFunctions();
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Decompose the functions into terms and set the evaluation tree
    // Also sets up the Kokkos::Views (subviews) to the data for all of the terms
    //////////////////////////////////////////////////////////////////////////////////////
    
    void decomposeFunctions();
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Determine if a term is a ScalarT or needs to be an AD type
    //////////////////////////////////////////////////////////////////////////////////////
    
    bool isScalarTerm(const int & findex, const int & tindex);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Evaluate a function (probably will be deprecated)
    //////////////////////////////////////////////////////////////////////////////////////
    
    View_AD2 evaluate(const std::string & fname, const std::string & location);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Evaluate a function
    //////////////////////////////////////////////////////////////////////////////////////
    
    void evaluate(const size_t & findex, const size_t & tindex);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Evaluate an operator
    //////////////////////////////////////////////////////////////////////////////////////
    
    template<class T1, class T2>
    void evaluateOp(T1 data, T2 tdata, const std::string & op);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Print out the function information (mostly for debugging)
    //////////////////////////////////////////////////////////////////////////////////////
    
    void printFunctions();
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    //////////////////////////////////////////////////////////////////////////////////////
    
    std::string blockname;
    int numElem, numip, numip_side;
    const int vectorSize = 32, teamSize = 1;
    
    std::vector<function_class> functions;
    
    std::vector<std::string> variables, aux_variables, parameters, disc_parameters;
    std::vector<std::string> known_vars, known_ops;
    Teuchos::RCP<workset> wkset;
    Teuchos::RCP<Interpreter> interpreter;
    Teuchos::RCP<Teuchos::Time> decomposeTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::function::decompose");
    Teuchos::RCP<Teuchos::Time> evaluateTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::function::evaluate");
    
  };
  
}
#endif


