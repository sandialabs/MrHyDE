/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_LINEAR_ALGEBRA_OPTS_H
#define MRHYDE_LINEAR_ALGEBRA_OPTS_H

#include "trilinos.hpp"
#include "preferences.hpp"

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

// Amesos includes
#include "Amesos2.hpp"

namespace MrHyDE {
/** \class  LinearSolverOptions
 *  \brief  Stores the specifications for a given linear solver.
 *
 *  This class holds configuration options for Amesos2, Belos, and MueLu
 *  solvers and preconditioners. It also stores reusable solver components
 *  such as Jacobians, symbolic factorizations, and preconditioners.
 *
 *  \tparam Node  Tpetra execution node type.
 */
template<class Node>
class LinearSolverOptions {
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
  
public:
  /** \brief Default constructor. */
  LinearSolverOptions() {};
  
  /** \brief Destructor. */
  ~LinearSolverOptions() {};
  
  /** \brief Construct options from a parameter list.
   *  \param settings  Parameter list containing all solver settings.
   */
  LinearSolverOptions(Teuchos::ParameterList & settings) {
    amesos_type = settings.get<string>("Amesos solver","KLU2");
    belos_type = settings.get<string>("Belos solver","Block GMRES");
    belos_sublist = settings.get<string>("Belos settings","Belos Settings");
    prec_sublist = settings.get<string>("Preconditioner settings","Preconditioner Settings");
    
    use_direct = settings.get<bool>("use direct solver",false);
    prec_type = settings.get<string>("preconditioner type","AMG");
    use_preconditioner = settings.get<bool>("use preconditioner",true);
    reuse_preconditioner = settings.get<bool>("reuse preconditioner",true);
    right_preconditioner = settings.get<bool>("right preconditioner",false);
    reuse_jacobian = settings.get<bool>("reuse Jacobian",false);
    
    have_preconditioner = false;
    have_symb_factor = false;
    have_jacobian = false;
    have_param_jacobian = false;
  }
  
  // Public data members
  string amesos_type;   /**< Amesos2 solver type (e.g., KLU2). */
  string belos_type;    /**< Belos solver type (e.g., GMRES). */
  string prec_type;     /**< Preconditioner type (e.g., AMG). */
  string belos_sublist; /**< Sublist name for Belos settings. */
  string prec_sublist;  /**< Sublist name for preconditioner settings. */
  bool use_direct;            /**< Use direct Amesos2 solver. */
  bool use_preconditioner;      /**< Whether to apply a preconditioner. */
  bool right_preconditioner;    /**< Whether to apply right preconditioning. */
  bool reuse_preconditioner;    /**< Whether to reuse an existing preconditioner. */
  bool reuse_jacobian;          /**< Whether to reuse an existing Jacobian. */
  bool have_jacobian;           /**< Indicates whether a Jacobian has been constructed. */
  bool have_param_jacobian;     /**< Indicates whether a parameter Jacobian has been constructed. */
  bool have_preconditioner;     /**< Indicates whether a preconditioner exists. */
  bool have_symb_factor;        /**< Indicates whether symbolic factorization exists. */
  bool have_previous_jacobian;  /**< Indicates whether previous Jacobians exist for reuse. */
  
  Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > amesos_solver; /**< Reusable Amesos2 direct solver. */
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > prec; /**< MueLu AMG preconditioner operator. */
  Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > prec_dd; /**< Ifpack2 domain decomposition preconditioner. */
  
  matrix_RCP jac, param_jac; /**< Current Jacobian matrix. */
  vector<matrix_RCP> jac_prev; /**< Previously stored Jacobians for reuse. */
};

} // MrHyDE

#endif
