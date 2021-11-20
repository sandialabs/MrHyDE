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
  
  // ========================================================================================
  // Constructor for linear solver options class
  // Also may store preconditioners or direct solvers for reuse
  // ========================================================================================
  
  template<class Node>
  class SolverOptions {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
    typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
    
  public:
    
    SolverOptions(Teuchos::ParameterList & settings) {
      amesosType = settings.get<string>("Amesos solver","KLU2");
      belosType = settings.get<string>("Belos solver","Block GMRES");
      belosSublist = settings.get<string>("Belos settings","Belos Settings");
      precSublist = settings.get<string>("Preconditioner settings","Preconditioner Settings");
      
      useDirect = settings.get<bool>("use direct solver",false);
      precType = settings.get<string>("preconditioner type","AMG");
      usePreconditioner = settings.get<bool>("use preconditioner",true);
      reusePreconditioner = settings.get<bool>("reuse preconditioner",true);
      rightPreconditioner = settings.get<bool>("right preconditioner",false);
      reuseJacobian = settings.get<bool>("reuse Jacobian",false);
      
      havePreconditioner = false;
      haveSymbFactor = false;
      haveJacobian = false;
    }
    
    string amesosType, belosType, precType;
    string belosSublist, precSublist;
    bool useDirect, usePreconditioner, rightPreconditioner, reusePreconditioner, reuseJacobian;
    bool haveJacobian, havePreconditioner, haveSymbFactor;
    Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > AmesosSolver;
    Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > M; // AMG preconditioner for Jacobians
    Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > M_dd; // domain decomposition preconditioner for Jacobians
    matrix_RCP J; // Jacobian
    
  };
  
  // ========================================================================================
  // Main Interface
  // ========================================================================================
  
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
    
    vector_RCP getNewVector(const size_t & set, const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(owned_map[set],numvecs));
      return newvec;
    }
    
    vector_RCP getNewOverlappedVector(const size_t & set, const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(overlapped_map[set],numvecs));
      return newvec;
    }
    
    matrix_RCP getNewMatrix(const size_t & set) {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat;
      if (options[set]->reuseJacobian) {
        if (options[set]->haveJacobian) {
          newmat = options[set]->J;
        }
        else {
          newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], maxEntries));
          options[set]->J = newmat;
          options[set]->haveJacobian = true;
        }
      }
      else {
        newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], maxEntries));
      }
      
      return newmat;
    }
    
    bool getJacobianReuse(const size_t & set) {
      bool reuse = false;
      if (options[set]->reuseJacobian && options[set]->haveJacobian) {
        reuse = true;
      }
      return reuse;
    }
    
    void resetJacobian(const size_t & set) {
      options[set]->haveJacobian = false;
    }
    
    matrix_RCP getNewOverlappedMatrix(const size_t & set) {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(overlapped_graph[set]));
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
      //return matrix;
    }
    
    matrix_RCP getNewParamOverlappedMatrix() {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_overlapped_graph));
      return newmat;
      //return overlapped_matrix;
    }
    
    // ========================================================================================
    // Exporters from overlapped to owned
    // ========================================================================================
    
    void exportVectorFromOverlapped(const size_t & set, vector_RCP & vec, vector_RCP & vec_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      if (Comm->getSize() > 1) {
        vec->putScalar(0.0);
        vec->doExport(*vec_over, *(exporter[set]), Tpetra::ADD);
      }
      else {
        vec->assign(*vec_over);
      }
    }
  
    void exportParamVectorFromOverlapped(vector_RCP & vec, vector_RCP & vec_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      vec->putScalar(0.0);
      vec->doExport(*vec_over, *param_exporter, Tpetra::ADD);
    }
  
    void exportMatrixFromOverlapped(const size_t & set, matrix_RCP & mat, matrix_RCP & mat_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      mat->setAllToScalar(0.0);
      mat->doExport(*mat_over, *(exporter[set]), Tpetra::ADD);
    }
  
    void exportParamMatrixFromOverlapped(matrix_RCP & mat, matrix_RCP & mat_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      mat->setAllToScalar(0.0);
      mat->doExport(*mat_over, *param_exporter, Tpetra::ADD);
    }
  
    // ========================================================================================
    // Importers from owned to overlapped
    // ========================================================================================
    
    void importVectorToOverlapped(const size_t & set, vector_RCP & vec_over, vector_RCP & vec) {
      Teuchos::TimeMonitor mattimer(*importtimer);
      vec_over->putScalar(0.0);
      vec_over->doImport(*vec, *(importer[set]), Tpetra::ADD);
    }
  
    // ========================================================================================
    // Fill complete calls
    // ========================================================================================
    
    void fillCompleteParam(const size_t & set, matrix_RCP & mat) {
      Teuchos::TimeMonitor mattimer(*fillcompletetimer);
      mat->fillComplete(owned_map[set], param_owned_map);
    }
  
    void fillComplete(matrix_RCP & mat) {
      Teuchos::TimeMonitor mattimer(*fillcompletetimer);
      mat->fillComplete();
    }
  
    // ========================================================================================
    // There are 3 types of matrices used in MrHyDE: Jacobian, L2 projection and boundary L2
    // There are 3 types of variables: standard, discretized parameters, aux
    // This gives 9 possible options for linear systems.
    // These can be very different and require different solver strategies.
    // ========================================================================================
    
    Teuchos::RCP<Teuchos::ParameterList> getBelosParameterList(const string & belosSublist);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for Jacobians of states
    // ========================================================================================
    
    void linearSolver(Teuchos::RCP<SolverOptions<Node> > & opt,
                      matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    
    void linearSolver(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    void PCG(const size_t & set, matrix_RCP & J, vector_RCP & b, vector_RCP & x, vector_RCP & Minv,
             const ScalarT & tol, const int & maxiter);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for Jacobians of discretized parameters
    // ========================================================================================
    
    void linearSolverParam(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for boundary L2 projections (Dirichlet BCs)
    // ========================================================================================
    
    void linearSolverBoundaryL2(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for boundary L2 projections (Dirichlet BCs)
    // ========================================================================================
    
    void linearSolverBoundaryL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for L2 projections (Initial conditions)
    // ========================================================================================
    
    void linearSolverL2(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    // ========================================================================================
    // Linear solver on Tpetra stack for L2 projections of parameters
    // ========================================================================================
    
    void linearSolverL2Param(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
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
    vector<string> setnames;
    
    // Maps, graphs, importers and exporters
    size_t maxEntries;
    vector<Teuchos::RCP<const LA_Map> > owned_map, overlapped_map;
    vector<Teuchos::RCP<LA_CrsGraph> > overlapped_graph; // owned graphs are never used
    vector<Teuchos::RCP<LA_Export> > exporter;
    vector<Teuchos::RCP<LA_Import> > importer;
    
    Teuchos::RCP<const LA_Map> param_owned_map, param_overlapped_map;
    Teuchos::RCP<LA_CrsGraph> param_overlapped_graph;
    Teuchos::RCP<LA_Export> param_exporter;
    Teuchos::RCP<LA_Import> param_importer;
    
    vector<matrix_RCP> matrix, overlapped_matrix;
    vector<vector_RCP> q_pcg, z_pcg, p_pcg, r_pcg;
    
    // Linear solvers and preconditioner settings
    int maxLinearIters, maxKrylovVectors;
    string belos_residual_scaling;
    ScalarT linearTOL;
    
    vector<Teuchos::RCP<SolverOptions<Node> > > options, options_L2, options_BndryL2;
    Teuchos::RCP<SolverOptions<Node> > options_param, options_param_L2, options_param_BndryL2;
    
    Teuchos::RCP<Teuchos::Time> setupLAtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::setup");
    Teuchos::RCP<Teuchos::Time> newvectortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNew*Vector()");
    Teuchos::RCP<Teuchos::Time> newmatrixtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNew*Matrix()");
    Teuchos::RCP<Teuchos::Time> linearsolvertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::linearSolver*()");
    Teuchos::RCP<Teuchos::Time> fillcompletetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::fillComplete*()");
    Teuchos::RCP<Teuchos::Time> exporttimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::export*()");
    Teuchos::RCP<Teuchos::Time> importtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::import*()");
    Teuchos::RCP<Teuchos::Time> PCGtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::PCG - total");
    Teuchos::RCP<Teuchos::Time> PCGApplyOptimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::PCG - apply Op");
    Teuchos::RCP<Teuchos::Time> PCGApplyPrectimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::PCG - apply prec");
    
  };
  
}

#endif
