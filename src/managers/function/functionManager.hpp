/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_FUNCTION_MANAGER_H
#define MRHYDE_FUNCTION_MANAGER_H

/** \file functionManager.hpp
 *  \brief Contains the function manager which handles all user-defined or physics-defined functions.
 *  \author Created by T. Wildey
 */

#include "trilinos.hpp"
#include "preferences.hpp"
#include "dag.hpp"
#include "interpreter.hpp"
#include "workset.hpp"
#include "vista.hpp"

namespace MrHyDE {

/** \class FunctionManager
 *  \brief Provides functionality that allows users or physics modules to define arbitrary
 *         functions without modifying the code.
 */

template<class EvalT>
class FunctionManager {

  typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT; ///< Kokkos view type for evaluation storage.

public:

  // ========================================================================================
  // Constructors / Destructor
  // ========================================================================================

  /** \brief Default constructor. */
  FunctionManager();

  /** \brief Destructor. */
  ~FunctionManager() {};

  /** \brief Standard constructor.
   *  \param blockname  Name of the block.
   *  \param num_elem   Number of elements per group.
   *  \param num_ip     Number of volumetric integration points.
   *  \param num_ip_side Number of side integration points.
   */
  FunctionManager(const std::string & blockname, const int & num_elem,
                  const int & num_ip, const int & num_ip_side);

  // ========================================================================================
  // Function addition
  // ========================================================================================

  /** \brief Add a function defined by a string expression.
   *  \param fname       Name of the function.
   *  \param expression  Mathematical expression string, e.g., "sin(pi*x)".
   *  \param location    Forest location ("ip", "side ip", "point").
   */
  int addFunction(const std::string & fname, const std::string & expression, const std::string & location);

  /** \brief Add a function defined by a constant scalar value.
   *  \param fname     Name of the function.
   *  \param value     Constant scalar value.
   *  \param location  Forest location.
   */
  int addFunction(const string & fname, ScalarT & value, const string & location);

  // ========================================================================================
  // Setup
  // ========================================================================================

  /** \brief Deprecated: sets parameter list for reference. Other information taken from workset.
   *  \param parameters Vector of parameter names.
   */
  void setupLists(const std::vector<std::string> & parameters);

  /** \brief Decomposes mathematical expression trees into evaluable terms and sets up Kokkos views. */
  void decomposeFunctions();

  // ========================================================================================
  // Type and dependency checking
  // ========================================================================================

  /** \brief Determine if a term is ScalarT or AD.
   *  \param findex Function index.
   *  \param tindex Tree index.
   *  \param bindex Branch index.
   */
  bool isScalarTerm(const int & findex, const int & tindex, const int & bindex);

  /** \brief Determine dependencies: constants, views, and/or AD types.
   *  \param findex Function index.
   *  \param tindex Tree index.
   *  \param bindex Branch index.
   *  \param isConst Output flag indicating constant dependencies.
   *  \param isView  Output flag indicating view-based dependencies.
   *  \param isAD    Output flag indicating AD dependencies.
   */
  void checkDepDataType(const int & findex, const int & tindex, const int & bindex,
                        bool & isConst, bool & isView, bool & isAD);

  // ========================================================================================
  // Evaluation routines
  // ========================================================================================

  /** \brief Evaluate a named function at a location.
   *  \param fname    Function name.
   *  \param location Forest in which the function resides.
   */
  Vista<EvalT> evaluate(const std::string & fname, const std::string & location);

  /** \brief Evaluate a function given indices.
   *  \param findex Function index.
   *  \param tindex Tree index.
   *  \param bindex Branch index.
   */
  void evaluate(const size_t & findex, const size_t & tindex, const size_t & bindex);

  /** \brief Evaluate operator where source and target are both views.
   *  \param data  Output data view.
   *  \param tdata Input data view.
   *  \param op    Operator name (e.g., "sin").
   */
  template<class T1, class T2>
  void evaluateOpVToV(T1 data, T2 tdata, const std::string & op);

  /** \brief Evaluate operator where source is parameter and target is a view.
   *  \param data     Output data.
   *  \param tdata    Parameter source data.
   *  \param pIndex_  Parameter index.
   *  \param op       Operator.
   */
  template<class T1, class T2>
  void evaluateOpParamToV(T1 data, T2 tdata, const int & pIndex_, const std::string & op);

  /** \brief Evaluate operator where source is ScalarT and target is view.
   *  \param data  Output data.
   *  \param tdata ScalarT source.
   *  \param op    Operator.
   */
  template<class T1, class T2>
  void evaluateOpSToV(T1 data, T2 & tdata, const std::string & op);

  /** \brief Evaluate operator where source and target are scalars.
   *  \param data  Output scalar.
   *  \param tdata Input scalar.
   *  \param op    Operator.
   */
  template<class T1, class T2>
  void evaluateOpSToS(T1 & data, T2 & tdata, const std::string & op);

  /** \brief Print all functions and dependencies (debugging). */
  void printFunctions();

  /** \brief Check if a named function exists at a location.
   *  \param fname    Function name.
   *  \param location Forest location ("ip", "side ip", "point").
   *  \return True if the function exists, false otherwise.
   */
  bool hasFunction(const std::string & fname, const std::string & location = "side ip");

  // ========================================================================================
  // Public data members
  // ========================================================================================

  int num_elem_;        ///< Number of elements.
  int num_ip_;          ///< Number of volume integration points.
  int num_ip_side_;     ///< Number of side integration points.

  Teuchos::RCP<Workset<EvalT> > wkset; ///< Workset containing simulation state.

private:

  std::string blockname_;                 ///< Name of the block associated with this function manager.
  std::vector<Forest<EvalT> > forests_;   ///< Forests of expression trees for different locations.
  std::vector<std::string> parameters_;   ///< List of parameters used in expressions.
  std::vector<std::string> disc_parameters_; ///< List of discretized parameters.
  std::vector<std::string> known_vars_;   ///< Variables known to the interpreter.
  std::vector<std::string> known_ops_;    ///< Operators recognized by the interpreter.

  Teuchos::RCP<Interpreter<EvalT> > interpreter_; ///< Interpreter for expression parsing and evaluation.
};

} // namespace MrHyDE

#endif


