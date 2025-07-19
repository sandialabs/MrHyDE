/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/** \file   linearAlgebraInterface.hpp
 \brief  Contains the interface to the linear algebra tools from Trilinos.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_LINEAR_ALGEBRA_H
#define MRHYDE_LINEAR_ALGEBRA_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "discretizationInterface.hpp"
#include "parameterManager.hpp"
#include "MrHyDE_Debugger.hpp"

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
  
  /** \class  MrHyDE::LinearSolverOptions
   \brief  Stores the specifications for a given linear solver.
   */
  
  // ========================================================================================
  // Constructor for linear solver options class
  // Also may store preconditioners or direct solvers for reuse
  // ========================================================================================
  
  template<class Node>
  class LinearSolverOptions {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
    typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
    
  public:
    
    LinearSolverOptions() {};
    
    // ========================================================================================
    // ========================================================================================
    
    ~LinearSolverOptions() {};
    
    // ========================================================================================
    // ========================================================================================
    
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
    }
    
    // ========================================================================================
    // ========================================================================================
    
    // This is basically just a struct storing data, so all data members are public
    
    string amesos_type, belos_type, prec_type;
    string belos_sublist, prec_sublist;
    bool use_direct, use_preconditioner, right_preconditioner, reuse_preconditioner, reuse_jacobian;
    bool have_jacobian, have_preconditioner, have_symb_factor;
    Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > amesos_solver;
    Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, Node> > prec; // AMG preconditioner for Jacobians
    Teuchos::RCP<Ifpack2::Preconditioner<ScalarT, LO, GO, Node> > prec_dd; // domain decomposition preconditioner for Jacobians
    matrix_RCP jac; // Jacobian
    
  };
  
  // ========================================================================================
  // Main Interface
  // ========================================================================================
  
  /** \class  MrHyDE::LinearAlgebraInterface
   \brief  Interface to Tpetra, Belos, MueLu and Amesos2 for various linear algebra routines.
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
    
    LinearAlgebraInterface() {};
    
    // ========================================================================================
    // ========================================================================================
    
    ~LinearAlgebraInterface() {};
    
    // ========================================================================================
    // ========================================================================================
    
    LinearAlgebraInterface(const Teuchos::RCP<MpiComm> & Comm_,
                           Teuchos::RCP<Teuchos::ParameterList> & settings_,
                           Teuchos::RCP<DiscretizationInterface> & disc_,
                           Teuchos::RCP<ParameterManager<Node> > & params_);
    
    // ========================================================================================
    // ========================================================================================
    
    void setupLinearAlgebra();
    
    // ========================================================================================
    // Get physics state linear algebra objects
    // ========================================================================================
    
    vector_RCP getNewVector(const size_t & set, const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(owned_map[set],numvecs));
      return newvec;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    vector_RCP getNewOverlappedVector(const size_t & set, const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newovervectortimer);
      vector_RCP newvec;
      if (have_overlapped) {
        newvec = Teuchos::rcp(new LA_MultiVector(overlapped_map[set],numvecs));
      }
      else {
        newvec = Teuchos::rcp(new LA_MultiVector(owned_map[set],numvecs));
      }
      return newvec;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    matrix_RCP getNewMatrix(const size_t & set) {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat;
      if (options[set]->reuse_jacobian) {
        if (options[set]->have_jacobian) {
          newmat = options[set]->jac;
        }
        else {
          newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
          options[set]->jac = newmat;
          options[set]->have_jacobian = true;
        }
      }
      else {
        newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], max_entries));
      }
      
      return newmat;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    matrix_RCP getNewParamMatrix() {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_owned_map, max_entries));
      return newmat;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    matrix_RCP getNewParamStateMatrix(const size_t & set) {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_owned_map, max_entries));
      return newmat;
    }

    // ========================================================================================
    // ========================================================================================
    
    matrix_RCP getNewMatrix(const size_t & set, vector<size_t> & maxent) {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], maxent));
      return newmat;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    bool getJacobianReuse(const size_t & set) {
      bool reuse = false;
      if (options[set]->reuse_jacobian && options[set]->have_jacobian) {
        reuse = true;
      }
      return reuse;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    void resetJacobian(const size_t & set) {
      options[set]->have_jacobian = false;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    matrix_RCP getNewOverlappedMatrix(const size_t & set) {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(overlapped_graph[set]));
      return newmat;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    matrix_RCP getNewOverlappedRectangularMatrix(Teuchos::RCP<const LA_Map> & colmap, const size_t & set) {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(overlapped_map[set], colmap, max_entries));
      return newmat;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    matrix_RCP getNewRectangularMatrix(Teuchos::RCP<const LA_Map> & colmap, const size_t & set) {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(owned_map[set], colmap,  max_entries));
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
    
    // ========================================================================================
    // ========================================================================================
    
    vector_RCP getNewOverlappedParamVector(const int & numvecs = 1) {
      Teuchos::TimeMonitor vectimer(*newvectortimer);
      vector_RCP newvec = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,numvecs));
      return newvec;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    matrix_RCP getNewOverlappedParamMatrix() {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(param_overlapped_graph));
      return newmat;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    matrix_RCP getNewOverlappedParamStateMatrix(const size_t & set) {
      Teuchos::TimeMonitor mattimer(*newmatrixtimer);
      matrix_RCP newmat = Teuchos::rcp(new LA_CrsMatrix(paramstate_overlapped_graph[set]));
      return newmat;
    }
    
    // ========================================================================================
    // Exporters from overlapped to owned
    // ========================================================================================
    
    void exportVectorFromOverlapped(const size_t & set, vector_RCP & vec, vector_RCP & vec_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      if (comm->getSize() > 1) {
        vec->putScalar(0.0);
        vec->doExport(*vec_over, *(exporter[set]), Tpetra::ADD);
      }
      else {
        vec->assign(*vec_over);
      }
    }
  
    // ========================================================================================
    // ========================================================================================
    
    void exportVectorFromOverlappedReplace(const size_t & set, vector_RCP & vec, vector_RCP & vec_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      if (comm->getSize() > 1) {
        vec->putScalar(0.0);
        vec->doExport(*vec_over, *(exporter[set]), Tpetra::REPLACE);
      }
      else {
        vec->assign(*vec_over);
      }
    }
  
    // ========================================================================================
    // ========================================================================================
    
    void exportParamVectorFromOverlapped(vector_RCP & vec, vector_RCP & vec_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      vec->putScalar(0.0);
      vec->doExport(*vec_over, *param_exporter, Tpetra::ADD);
    }
  
    // ========================================================================================
    // ========================================================================================
    
    void exportMatrixFromOverlapped(const size_t & set, matrix_RCP & mat, matrix_RCP & mat_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      mat->setAllToScalar(0.0);
      mat->doExport(*mat_over, *(exporter[set]), Tpetra::ADD);
    }
  
    // ========================================================================================
    // ========================================================================================
    
    void exportParamStateMatrixFromOverlapped(const size_t & set, matrix_RCP & mat, matrix_RCP & mat_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      mat->setAllToScalar(0.0);
      mat->doExport(*mat_over, *(param_exporter), Tpetra::ADD);
    }
    
    // ========================================================================================
    // ========================================================================================
    
    void exportParamMatrixFromOverlapped(matrix_RCP & mat, matrix_RCP & mat_over) {
      Teuchos::TimeMonitor mattimer(*exporttimer);
      mat->setAllToScalar(0.0);
      mat->doExport(*mat_over, *param_exporter, Tpetra::ADD);
    }
  
    // ========================================================================================
    // Importers from owned to overlapped
    // ========================================================================================
    
    void importVectorToOverlapped(const size_t & set, vector_RCP & vec_over, const vector_RCP & vec) {
      Teuchos::TimeMonitor mattimer(*importtimer);
      vec_over->putScalar(0.0);
      vec_over->doImport(*vec, *(importer[set]), Tpetra::ADD);
    }
  
    // ========================================================================================
    // Fill complete calls
    // ========================================================================================
    
    void fillCompleteParamState(const size_t & set, matrix_RCP & mat) {
      Teuchos::TimeMonitor mattimer(*fillcompletetimer);
      mat->fillComplete(owned_map[set], param_owned_map);
    }
  
    // ========================================================================================
    // ========================================================================================
    
    void fillComplete(matrix_RCP & mat) {
      Teuchos::TimeMonitor mattimer(*fillcompletetimer);
      mat->fillComplete();
    }
    
    // ========================================================================================
    // ========================================================================================
 
    size_t getLocalNumElements(const size_t & set) {
      size_t numElem = 0;
      if (have_overlapped) {
        numElem = overlapped_map[set]->getLocalNumElements();
      }
      else {
        numElem = owned_map[set]->getLocalNumElements();
      }
      return numElem;
    }

    // ========================================================================================
    // ========================================================================================
 
    size_t getLocalNumParamElements() {
      size_t numElem = 0;
      if (have_overlapped) {
        numElem = param_overlapped_map->getLocalNumElements();
      }
      else {
        numElem = param_owned_map->getLocalNumElements();
      }
      return numElem;
    }

    // ========================================================================================
    // ========================================================================================
    
    Teuchos::RCP<LA_CrsGraph> getNewOverlappedGraph(const size_t & set, vector<size_t> & maxEntriesPerRow) {
      Teuchos::RCP<LA_CrsGraph> newgraph;
      if (have_overlapped) {
        newgraph = Teuchos::rcp(new LA_CrsGraph(overlapped_map[set], maxEntriesPerRow));
      }
      else {
        newgraph = Teuchos::rcp(new LA_CrsGraph(owned_map[set], maxEntriesPerRow));
      }
      return newgraph;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    Teuchos::RCP<LA_CrsGraph> getNewParamOverlappedGraph(vector<size_t> & maxEntriesPerRow) {
      Teuchos::RCP<LA_CrsGraph> newgraph;
      if (have_overlapped) {
        newgraph = Teuchos::rcp(new LA_CrsGraph(param_overlapped_map, maxEntriesPerRow));
      }
      else {
        newgraph = Teuchos::rcp(new LA_CrsGraph(param_owned_map, maxEntriesPerRow));
      }
      return newgraph;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    GO getGlobalElement(const size_t & set, const LO & lid) {
      GO gid = 0;
      if (have_overlapped) {
        gid = overlapped_map[set]->getGlobalElement(lid);
      }
      else {
        gid = owned_map[set]->getGlobalElement(lid);
      }
      return gid;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    GO getGlobalParamElement(const LO & lid) {
      GO gid = 0;
      if (have_overlapped) {
        gid = param_overlapped_map->getGlobalElement(lid);
      }
      else {
        gid = param_owned_map->getGlobalElement(lid);
      }
      return gid;
    }

    // ========================================================================================
    // ========================================================================================
    
    bool getHaveOverlapped() {
      return have_overlapped;
    }

    // ========================================================================================
    // ========================================================================================
    
    LO getOverlappedLID(const size_t & set, const GO & gid) {
      LO lid = 0;
      if (have_overlapped) {
        lid = overlapped_map[set]->getLocalElement(gid);
      }
      else {
        lid = owned_map[set]->getLocalElement(gid);
      }
      return lid;
    }

    // ========================================================================================
    // ========================================================================================
    
    LO getOwnedLID(const size_t & set, const GO & gid) {
      return owned_map[set]->getLocalElement(gid);
    }
    
      
    // ========================================================================================
    // Write the Jacobian and/or residual to a matrix-market text file
    // ========================================================================================

    void writeToFile(matrix_RCP &J, vector_RCP &r, vector_RCP &soln, 
                     const std::string &jac_filename="jacobian.mm",
                     const std::string &res_filename="residual.mm",
                     const std::string &sol_filename="solution.mm") {

      Teuchos::TimeMonitor localtimer(*writefiletimer);

      // Tpetra gathers the entire matrix or the entire residual on proc 0 
      // when Tpetra::MatrixMarket::Writer is called. Very large matrices
      // will cause this to run out of memory!

      if(do_dump_jacobian)
        Tpetra::MatrixMarket::Writer<LA_CrsMatrix>::writeSparseFile(jac_filename,*J);
      if(do_dump_residual)
        Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(res_filename,*r);
      if(do_dump_solution)
        Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(sol_filename,*soln);
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
    
    void linearSolver(Teuchos::RCP<LinearSolverOptions<Node> > & opt,
                      matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
    
    // ========================================================================================
    // ========================================================================================
    
    void linearSolver(const size_t & set, matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
    
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
    
    vector<Teuchos::RCP<const LA_Map> > owned_map, overlapped_map;
    vector<Teuchos::RCP<LA_CrsGraph> > overlapped_graph; // owned graphs are never used
    vector<Teuchos::RCP<LA_Export> > exporter;
    vector<Teuchos::RCP<LA_Import> > importer;
    
    vector<Teuchos::RCP<LinearSolverOptions<Node> > > options, options_L2, options_BndryL2;
    Teuchos::RCP<LinearSolverOptions<Node> > options_param, options_param_L2, options_param_BndryL2;
    
  private:

    Teuchos::RCP<MpiComm> comm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<DiscretizationInterface> disc;
    Teuchos::RCP<ParameterManager<Node> > params;
    Teuchos::RCP<MrHyDE_Debugger> debugger;
    
    int verbosity;
    vector<string> setnames;
    bool do_dump_jacobian, do_dump_residual, do_dump_solution;
    bool have_overlapped;

    // Maps, graphs, importers and exporters
    size_t max_entries;
    
    Teuchos::RCP<const LA_Map> param_owned_map, param_overlapped_map;
    Teuchos::RCP<LA_CrsGraph> param_overlapped_graph;
    Teuchos::RCP<LA_Export> param_exporter;
    Teuchos::RCP<LA_Import> param_importer;
    
    vector<Teuchos::RCP<const LA_Map> > paramstate_owned_map, paramstate_overlapped_map;
    vector<Teuchos::RCP<LA_CrsGraph> > paramstate_overlapped_graph;
    //vector<Teuchos::RCP<LA_Export> > paramstate_exporter;
    //vector<Teuchos::RCP<LA_Import> > paramstate_importer;
    
    vector<matrix_RCP> matrix, overlapped_matrix;
    
    // Linear solvers and preconditioner settings
    int maxLinearIters, maxKrylovVectors;
    string belos_residual_scaling;
    ScalarT linearTOL;
    bool doCondEst;
    
    Teuchos::RCP<Teuchos::Time> setupLAtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::setup");
    Teuchos::RCP<Teuchos::Time> newvectortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNewVector()");
    Teuchos::RCP<Teuchos::Time> newovervectortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNewOverlappedVector()");
    Teuchos::RCP<Teuchos::Time> newmatrixtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::getNew*Matrix()");
    Teuchos::RCP<Teuchos::Time> writefiletimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::write()");
    Teuchos::RCP<Teuchos::Time> linearsolvertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::linearSolver*()");
    Teuchos::RCP<Teuchos::Time> fillcompletetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::fillComplete*()");
    Teuchos::RCP<Teuchos::Time> exporttimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::export*()");
    Teuchos::RCP<Teuchos::Time> importtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::import*()");
    Teuchos::RCP<Teuchos::Time> prectimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::LinearAlgebraInterface::buildPreconditioner()");
    
  };
  
}

#endif
