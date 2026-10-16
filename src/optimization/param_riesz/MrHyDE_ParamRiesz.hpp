/***********************************************************************
 MrHyDE - The parameter Riesz operator P: applies P (primal to dual) and
 P^-1 (dual back to primal), by sparse factorization or RefMaxwell-CG.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_PARAM_RIESZ_HPP
#define MRHYDE_PARAM_RIESZ_HPP

#include "trilinos.hpp"
#include "preferences.hpp"
#include "Amesos2.hpp"
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <iostream>

// P maps primal to dual and P^-1 back. ROL's default precond is v.dual(),
// so P^-1 also preconditions the trust-region CG.
class MrHyDE_ParamRiesz {

  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SolverNode>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> LA_MultiVector;
  typedef Tpetra::Operator<ScalarT,LO,GO,SolverNode>    LA_Operator;
  typedef Teuchos::RCP<LA_MultiVector>                  vector_RCP;
  typedef Teuchos::RCP<LA_CrsMatrix>                    matrix_RCP;

public:

  // P_factor is P with a sparser stored pattern, or null to factor P itself.
  MrHyDE_ParamRiesz(const matrix_RCP & P_, const std::string & solver_name_ = "KLU2",
                    const matrix_RCP & P_factor_ = Teuchos::null)
  : P(P_), P_factor(P_factor_.is_null() ? P_ : P_factor_), solver_name(solver_name_) {}

  // Krylov on the control rows P_live; P is diagonal on the other rows, given by diag.
  MrHyDE_ParamRiesz(const matrix_RCP & P_, const matrix_RCP & P_live_, const vector_RCP & diag,
                    const Teuchos::RCP<LA_Operator> & prec_, const std::string & solver_name_,
                    const Teuchos::RCP<Teuchos::ParameterList> & belos_params_,
                    const int verbosity_ = 0, const bool strict_ = false)
  : P(P_), P_factor(P_), solver_name(solver_name_), P_live(P_live_), prec(prec_),
    belos_params(belos_params_), verbosity(verbosity_), strict(strict_) {
    inv_diag = Teuchos::rcp(new LA_MultiVector(diag->getMap(), 1));
    inv_diag->reciprocal(*diag);
    to_live = Teuchos::rcp(new Tpetra::Import<LO,GO,SolverNode>(P->getRowMap(), P_live->getRowMap()));
    rank = P_live->getRowMap()->getComm()->getRank();
  }

  // y <- P x
  void apply(const vector_RCP & x, const vector_RCP & y) const {
    P->apply(*x, *y);
  }

  // y <- P^-1 x. One factorization serves every slab and every CG iteration.
  void applyInverse(const vector_RCP & x, const vector_RCP & y) const {
    Teuchos::TimeMonitor localtimer(*solvetimer);
    if (!P_live.is_null()) {
      this->solveControlRows(x, y);
      return;
    }
    if (solver.is_null()) {
      Teuchos::TimeMonitor localfactortimer(*factortimer);
      solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>(solver_name, P_factor, y, x);
      solver->symbolicFactorization();
      solver->numericFactorization();
    }
    solver->setX(y);
    solver->setB(x);
    solver->solve();
  }

private:

  void solveControlRows(const vector_RCP & x, const vector_RCP & y) const {
    const size_t nv = x->getNumVectors();
    y->elementWiseMultiply(1.0, *inv_diag->getVector(0), *x, 0.0);
    if (b_work.is_null() || b_work->getNumVectors() != nv) {
      b_work = Teuchos::rcp(new LA_MultiVector(P_live->getRowMap(), nv));
      z_work = Teuchos::rcp(new LA_MultiVector(P_live->getRowMap(), nv));
    }
    const vector_RCP b = b_work, z = z_work;
    z->putScalar(0.0);   // cold start, so the apply does not depend on the previous solve
    b->doImport(*x, *to_live, Tpetra::INSERT);
    if (krylov.is_null()) {
      problem = Teuchos::rcp(new Belos::LinearProblem<ScalarT,LA_MultiVector,LA_Operator>(P_live, z, b));
      if (!prec.is_null()) { problem->setLeftPrec(prec); }
      problem->setProblem();
      Belos::SolverFactory<ScalarT,LA_MultiVector,LA_Operator> factory;
      krylov = factory.create(solver_name, belos_params);
      krylov->setProblem(problem);
    }
    else {
      problem->setProblem(z, b);
      krylov->reset(Belos::Problem);
    }
    const Belos::ReturnType ret = krylov->solve();
    if (ret != Belos::Converged) {
      TEUCHOS_TEST_FOR_EXCEPTION(strict, std::runtime_error,
        "Riesz map: iterative solve stopped at " << krylov->achievedTol() << " after "
        << krylov->getNumIters() << " iterations (strict is set).");
      if (!warned && verbosity >= 1 && rank == 0) {
        std::cout << "WARNING: Riesz map: iterative solve stopped at " << krylov->achievedTol()
                  << " after " << krylov->getNumIters() << " iterations; continuing. "
                  << "Set Iterative Settings -> strict: true to make this an error." << std::endl;
      }
      warned = true;
    }
    y->doExport(*z, *to_live, Tpetra::INSERT);
  }

  matrix_RCP P, P_factor;
  std::string solver_name;
  matrix_RCP P_live;
  Teuchos::RCP<LA_Operator> prec;
  Teuchos::RCP<Teuchos::ParameterList> belos_params;
  int verbosity = 0;
  bool strict = false;
  int rank = 0;
  mutable bool warned = false;
  mutable vector_RCP b_work, z_work;   // control-row RHS and solution, reused across applies
  vector_RCP inv_diag;
  Teuchos::RCP<Tpetra::Import<LO,GO,SolverNode> > to_live;
  mutable Teuchos::RCP<Belos::LinearProblem<ScalarT,LA_MultiVector,LA_Operator> > problem;
  mutable Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > krylov;
  mutable Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > solver;
  Teuchos::RCP<Teuchos::Time> factortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParamRiesz::factor()");
  Teuchos::RCP<Teuchos::Time> solvetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParamRiesz::applyInverse()");

};

#endif
