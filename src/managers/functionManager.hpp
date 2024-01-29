/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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
  
  template<class EvalT>
  class FunctionManager {

    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT;
    
  public:
    
    // ========================================================================================
    // ========================================================================================
    
    FunctionManager();
    
    // ========================================================================================
    // ========================================================================================
    
    ~FunctionManager() {};
    
    /**
     * @brief Standard contructor.
     *
     * @param[in]  blockname    String associated with the block.
     * @param[in]  num_elem      Number of elements per group.
     * @param[in]  num_ip      Number of volumetric integration points
     * @param[in]  num_ip_side      Number of side integration points.
     */
    
    FunctionManager(const std::string & blockname, const int & num_elem,
                    const int & num_ip, const int & num_ip_side);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Add a function (tree) to a specific forest given by "location", e.g. "ip" "side ip" or "point"
     *
     * @param[in]  fname    String giving the name of the function.
     * @param[in]  expression    String giving the mathematical expression of the function, e.g., "sin(pi*x)"
     * @param[in]  location    Forest to put this in.
     */
    
    int addFunction(const std::string & fname, const std::string & expression, const std::string & location);
    
    /**
     * @brief Add a function (tree) to a specific forest given by "location", e.g. "ip" "side ip" or "point"
     *
     * @param[in]  fname    String giving the name of the function.
     * @param[in]  value    Scalar value for the function.  This is handles differently from expressions for memory and cost reasons.
     * @param[in]  location    Forest to put this in.
     */
    
    int addFunction(const string & fname, ScalarT & value, const string & location);

    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief A mostly deprecated function that now just sets the list of parameters for reference.  Other information is taken from the workset.
     *
     * @param[in]  parameters    Vector of strings giving the name of the parameters.
     */
    
    void setupLists(const std::vector<std::string> & parameters);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief This is one of the key routines in the function manager.  
     *   It takes a string defining a mathematical expression and decomposes it into a 
     *   tree where the leaves of the tree are known quantities.
     *   Decompose the functions into terms and set the evaluation tree.
     *   Also sets up the Kokkos::Views (subviews) to the data for all of the terms
     */
    
    void decomposeFunctions();
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Routine to determine if a term is a ScalarT or needs to be an AD type
     *
     * @param[in]  findex    Index of the function in the branch.
     * @param[in]  tindex    Index of the tree in the forest.
     * @param[in]  bindex    Index of the branch in the tree.
     */
    
    bool isScalarTerm(const int & findex, const int & tindex, const int & bindex);
    
    /**
     * @brief Routine to determine if the dependent fields for a quantity are constants, Kokkos::Views and/or AD types.
     *
     * @param[in]  findex    Index of the function in the branch.
     * @param[in]  tindex    Index of the tree in the forest.
     * @param[in]  bindex    Index of the branch in the tree.
     * @param[out]  isConst   Are the dependencies all constant?
     * @param[out]  isView   Are any of the dependencies views?
     * @param[out]  isAD   Are any of the dependencies AD?
     */
    
    void checkDepDataType(const int & findex, const int & tindex, const int & bindex,
                          bool & isConst, bool & isView, bool & isAD);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Routine to evaluate a funciton
     *
     * @param[in]  fname   String giving name of the function
     * @param[in]  location    String giving the location (forest) of the function.  Some functions may be defined in multiple forests.
     */
    
    Vista<EvalT> evaluate(const std::string & fname, const std::string & location);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Routine to determine if a term is a ScalarT or needs to be an AD type
     *
     * @param[in]  findex    Index of the function in the branch.
     * @param[in]  tindex    Index of the tree in the forest.
     * @param[in]  bindex    Index of the branch in the tree.
     */
    
    void evaluate(const size_t & findex, const size_t & tindex, const size_t & bindex);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Evaluate an operator
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Evaluate an operator.  Case when the source and target  are both Views
     *
     * @param[out]  data    Storage for the target data, e.g., sin(x)
     * @param[in]   tdata    Storage for the source data, e.g., x
     * @param[in]   op      String indicating the operator, e.g., sin()
     */
    
    template<class T1, class T2>
    void evaluateOpVToV(T1 data, T2 tdata, const std::string & op);
    
    /**
     * @brief Evaluate an operator.  Case when the source is a parameter and the target is a View.
     *
     * @param[out]  data    Storage for the evaluated data, e.g., sin(x)
     * @param[in]   tdata    Storage for the source data, e.g., x
     * @param[in]   op      String indicating the operator, e.g., sin()
     */
    
    template<class T1, class T2>
    void evaluateOpParamToV(T1 data, T2 tdata, const int & pIndex_, const std::string & op);
    
    /**
     * @brief Evaluate an operator.  Case when the source is a ScalarT and the target is a View.
     *
     * @param[out]  data    Storage for the evaluated data, e.g., sin(x)
     * @param[in]   tdata    Storage for the source data, e.g., x
     * @param[in]   op      String indicating the operator, e.g., sin()
     */
    
    template<class T1, class T2>
    void evaluateOpSToV(T1 data, T2 & tdata, const std::string & op);
    
    /**
     * @brief Evaluate an operator.  Case when the source and target are both scalars.
     *
     * @param[out]  data    Storage for the evaluated data, e.g., sin(x)
     * @param[in]   tdata    Storage for the source data, e.g., x
     * @param[in]   op      String indicating the operator, e.g., sin()
     */
    
    template<class T1, class T2>
    void evaluateOpSToS(T1 & data, T2 & tdata, const std::string & op);
    
    //////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
    
    /**
     * @brief Print all of the functions and their dependencies.  Mostly for debugging.
     */
    
    void printFunctions();
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    //////////////////////////////////////////////////////////////////////////////////////
    
    int num_elem_, num_ip_, num_ip_side_;
    Teuchos::RCP<Workset<EvalT> > wkset;
    
  private:

    std::string blockname_;
    std::vector<Forest<EvalT> > forests_;
    std::vector<std::string> parameters_, disc_parameters_;
    std::vector<std::string> known_vars_, known_ops_;
    Teuchos::RCP<Interpreter<EvalT> > interpreter_;
  };
  
}
#endif


