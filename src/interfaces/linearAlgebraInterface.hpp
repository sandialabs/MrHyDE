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

#ifndef MRHYDE_LINEAR_ALGEBRA_H
#define MRHYDE_LINEAR_ALGEBRA_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "discretizationInterface.hpp"
#include "parameterManager.hpp"

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosBlockGmresSolMgr.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

// Amesos includes
#include "Amesos2.hpp"


namespace MrHyDE {
  /*
  void static solverHelp(const string & details) {
    cout << "********** Help and Documentation for the Solver Interface **********" << endl;
  }
  */
  template<class Node>
  class LinearAlgebraInterface {

    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
    typedef Tpetra::CrsGraph<LO,GO,Node>            LA_CrsGraph;
    typedef Tpetra::Export<LO,GO,Node>              LA_Export;
    typedef Tpetra::Import<LO,GO,Node>              LA_Import;
    typedef Tpetra::Map<LO,GO,Node>                 LA_Map;
    typedef Tpetra::Operator<ScalarT,LO,GO,Node>    LA_Operator;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;
    typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
    typedef typename Node::device_type              LA_device;
    typedef Belos::LinearProblem<ScalarT,LA_MultiVector,LA_Operator> LA_LinearProblem;
    
  public:
    
    // ========================================================================================
    // Constructor
    // ========================================================================================
    
    LinearAlgebraInterface(const Teuchos::RCP<MpiComm> & Comm_,
                  Teuchos::RCP<Teuchos::ParameterList> & settings_,
                  Teuchos::RCP<DiscretizationInterface> & disc_,
                  Teuchos::RCP<ParameterManager<Node> > & params_);
    
    void setupLinearAlgebra();
    
    // ========================================================================================
    // Get physics state linear algebra objects
    // ========================================================================================
    
    vector_RCP getNewVector(const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(owned_map,numvecs));
      return newvec;
    }
    
    vector_RCP getNewOverlappedVector(const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(overlapped_map,numvecs));
      return newvec;
    }
    
    matrix_RCP getNewMatrix() {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map, maxEntries));
      return newmat;
    }
    
    matrix_RCP getNewOverlappedMatrix() {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(overlapped_graph));
      return newmat;
    }
    
    // ========================================================================================
    // Get discretized parameter linear algebra objects
    // ========================================================================================
    
    vector_RCP getNewParamVector(const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(param_owned_map,numvecs));
      return newvec;
    }
    
    vector_RCP getNewParamOverlappedVector(const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,numvecs));
      return newvec;
    }
    
    matrix_RCP getNewParamMatrix() {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_owned_map, maxEntries));
      return newmat;
    }
    
    matrix_RCP getNewParamOverlappedMatrix() {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_overlapped_graph));
      return newmat;
    }
    
    // ========================================================================================
    // Get aux variable linear algebra objects
    // ========================================================================================
    
    vector_RCP getNewAuxVector(const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(aux_owned_map,numvecs));
      return newvec;
    }
    
    vector_RCP getNewAuxOverlappedVector(const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(aux_overlapped_map,numvecs));
      return newvec;
    }
    
    matrix_RCP getNewAuxMatrix() {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(aux_owned_map, maxEntries));
      return newmat;
    }
    
    matrix_RCP getNewAuxOverlappedMatrix() {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(aux_overlapped_graph));
      return newmat;
    }
    
    // ========================================================================================
    // Exporters from overlapped to owned
    // ========================================================================================
    
    void exportVectorFromOverlapped(vector_RCP & vec, vector_RCP & vec_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      vec->putScalar(0.0);
      vec->doExport(*vec_over, *exporter, Tpetra::ADD);
    }
  
    void exportParamVectorFromOverlapped(vector_RCP & vec, vector_RCP & vec_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      vec->putScalar(0.0);
      vec->doExport(*vec_over, *param_exporter, Tpetra::ADD);
    }
  
    void exportAuxVectorFromOverlapped(vector_RCP & vec, vector_RCP & vec_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      vec->putScalar(0.0);
      vec->doExport(*vec_over, *aux_exporter, Tpetra::ADD);
    }
  
    void exportMatrixFromOverlapped(matrix_RCP & mat, matrix_RCP & mat_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      mat->setAllToScalar(0.0);
      mat->doExport(*mat_over, *exporter, Tpetra::ADD);
    }
  
    void exportParamMatrixFromOverlapped(matrix_RCP & mat, matrix_RCP & mat_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      mat->setAllToScalar(0.0);
      mat->doExport(*mat_over, *param_exporter, Tpetra::ADD);
    }
  
    void exportAuxMatrixFromOverlapped(matrix_RCP & mat, matrix_RCP & mat_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      mat->setAllToScalar(0.0);
      mat->doExport(*mat_over, *aux_exporter, Tpetra::ADD);
    }
  
    // ========================================================================================
    // Importers from owned to overlapped
    // ========================================================================================
    
    void importVectorToOverlapped(vector_RCP & vec_over, vector_RCP & vec) {
      Teuchos::TimeMonitor mattimer(*importtimer);
      vec_over->putScalar(0.0);
      vec_over->doImport(*vec, *importer, Tpetra::ADD);
    }
  
    // ========================================================================================
    // Fill complete calls
    // ========================================================================================
    
    void fillCompleteParam(matrix_RCP & mat) {
      Teuchos::TimeMonitor mattimer(*fillcompletetimer);
      mat->fillComplete(owned_map, param_owned_map);
    }
  
    void fillComplete(matrix_RCP & mat) {
      Teuchos::TimeMonitor mattimer(*fillcompletetimer);
      mat->fillComplete();
    }
  
    // ========================================================================================
    // There are 3 types of matrices used in MrHyDE: Jacobian, L2 projection and boundary L2
    // There are 3 types of variables: standard, discretized parameters, aux
    // This gives 9 possible options for linear systems, but Jacobians of params are not used.
    // These can be very different and require different solver strategies.
    // Not all 9 are implemented
    // ========================================================================================
    
    Teuchos::RCP<Teuchos::ParameterList> getBelosParameterList();
    
    // ========================================================================================
    // Linear solver on Tpetra stack for Jacobians
    // ========================================================================================
    
    void linearSolver(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for Jacobians of aux vars
    // ========================================================================================
    
    void linearSolverAux(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for boundary L2 projections (Dirichlet BCs)
    // ========================================================================================
    
    void linearSolverBoundaryL2(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for boundary L2 projections (Dirichlet BCs)
    // ========================================================================================
    
    void linearSolverBoundaryL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for boundary L2 projections (Dirichlet BCs)
    // ========================================================================================
    
    void linearSolverBoundaryL2Aux(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for L2 projections (Initial conditions)
    // ========================================================================================
    
    void linearSolverL2(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for L2 projections of parameters
    // ========================================================================================
    
    void linearSolverL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for L2 projections of aux vars
    // ========================================================================================
    
    void linearSolverL2Aux(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Preconditioner for Tpetra stack
    // ========================================================================================
    
    Teuchos::RCP<MueLu::TpetraOperator<ScalarT,LO,GO,Node> > buildPreconditioner(const matrix_RCP & J,
                                                                                 const string & precSublist);
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<DiscretizationInterface> disc;
    Teuchos::RCP<ParameterManager<Node> > params;
    
    int verbosity, debug_level;
    
    // Maps, graphs, importers and exporters
    int maxEntries;
    Teuchos::RCP<const LA_Map> owned_map, overlapped_map, param_owned_map, param_overlapped_map, aux_owned_map, aux_overlapped_map;
    Teuchos::RCP<LA_CrsGraph> overlapped_graph, param_overlapped_graph, aux_overlapped_graph; // owned graphs are never used
    Teuchos::RCP<LA_Export> exporter, param_exporter, aux_exporter;
    Teuchos::RCP<LA_Import> importer, param_importer, aux_importer;
    
    // Linear solvers and preconditioners
    int maxLinearIters, maxKrylovVectors;
    bool have_preconditioner=false, have_aux_preconditioner=false, reuse_preconditioner, reuse_aux_preconditioner, have_symbolic_factor=false, have_aux_symbolic_factor=false;
    bool useDirect, useDirectL2, useDirectBL2, useDirectAux, useDirectL2Aux, useDirectBL2Aux, useDirectL2Param, useDirectBL2Param;
    bool useDomDecomp, useDomDecompL2, useDomDecompBL2, useDomDecompAux, useDomDecompL2Aux, useDomDecompBL2Aux, useDomDecompL2Param, useDomDecompBL2Param;
    bool usePrec, usePrecL2, usePrecBL2, usePrecAux, usePrecL2Aux, usePrecBL2Aux, usePrecL2Param, usePrecBL2Param;
    string belos_residual_scaling;
    ScalarT linearTOL;
    Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > Am2Solver, Am2Solver_aux;
    Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > M, M_aux; // AMG preconditioner for Jacobians
    Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > M_dd, M_dd_aux; // domain decomposition preconditioner for Jacobians
    
    Teuchos::RCP<Teuchos::Time> setupLAtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::setup");
    Teuchos::RCP<Teuchos::Time> newvectortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNew*Vector()");
    Teuchos::RCP<Teuchos::Time> newmatrixtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNew*Matrix()");
    Teuchos::RCP<Teuchos::Time> linearsolvertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::linearSolver*()");
    Teuchos::RCP<Teuchos::Time> fillcompletetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::fillComplete*()");
    Teuchos::RCP<Teuchos::Time> exporttimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::export*()");
    Teuchos::RCP<Teuchos::Time> importtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::import*()");
    
    
  };
  
}

#endif
