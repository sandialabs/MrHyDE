/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "linearAlgebraInterface.hpp"
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBiCGStabSolMgr.hpp>
#include <BelosGCRODRSolMgr.hpp>
#include <BelosPCPGSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosPseudoBlockStochasticCGSolMgr.hpp>
#include <BelosPseudoBlockTFQMRSolMgr.hpp>
#include <BelosRCGSolMgr.hpp>
#include <BelosTFQMRSolMgr.hpp>

using namespace MrHyDE;

// ========================================================================================

template<class Node>
bool LinearAlgebraInterface<Node>::getJacobianReuse(const size_t & set) {
  bool reuse = false;
  if (context[set]->reuse_matrix && context[set]->have_matrix) {
    reuse = true;
  }
  return reuse;
}

// ========================================================================================

template<class Node>
bool LinearAlgebraInterface<Node>::getParamJacobianReuse() {
  bool reuse = false;
  if (context_param->reuse_matrix && context_param->have_matrix) {
    reuse = true;
  }
  return reuse;
}

// ========================================================================================

template<class Node>
bool LinearAlgebraInterface<Node>::getParamStateJacobianReuse(const size_t & set) {
  bool reuse = false;
  if (context_param_state[set]->reuse_matrix && context_param_state[set]->have_matrix) {
    reuse = true;
  }
  return reuse;
}

// ========================================================================================
// All iterative solvers use the same Belos list.  This would be easy to specialize.
// ========================================================================================

template<class Node>
Teuchos::RCP<Teuchos::ParameterList> LinearAlgebraInterface<Node>::getBelosParameterList(Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {
  Teuchos::RCP<Teuchos::ParameterList> belosList = Teuchos::rcp(new Teuchos::ParameterList());
  belosList->set("Maximum Iterations",    maxLinearIters); // Maximum number of iterations allowed
  belosList->set("Num Blocks", maxLinearIters);
  belosList->set("Convergence Tolerance", linearTOL);    // Relative convergence tolerance requested
  belosList->set("Estimate Condition Number", doCondEst); // Only implemented in Belos for Pseudo Block CG, based on AztecOO
  if (verbosity > 9) {
    belosList->set("Verbosity", Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
  }
  else {
    belosList->set("Verbosity", Belos::Errors);
  }
  if (verbosity > 8) {
    belosList->set("Output Frequency",10);
  }
  else {
    belosList->set("Output Frequency",0);
  }
  int numEqns = 1;
  if (disc->block_names.size() == 1) {
    numEqns = disc->physics->num_vars[0][0];
  }
  belosList->set("number of equations", numEqns);
  
  belosList->set("Output Style", Belos::Brief);
  belosList->set("Implicit Residual Scaling", belos_residual_scaling);
  
  if (cntxt->belos_sublist.name() != "empty") {
    //Teuchos::ParameterList inputParams = settings->sublist("Solver").sublist(belosSublist);
    belosList->setParameters(cntxt->belos_sublist);
  }
  
  return belosList;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetAllJacobian() {
  this->resetJacobian();
  this->resetL2Jacobian();
  this->resetBndryL2Jacobian();
  this->resetParamJacobian();
  this->resetPrevJacobian();
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetJacobian() {
  for (size_t set=0; set<context.size(); ++set) {
    context[set]->reset();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetL2Jacobian() {
  for (size_t set=0; set<context_L2.size(); ++set) {
    context_L2[set]->reset();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetBndryL2Jacobian() {
  for (size_t set=0; set<context_BndryL2.size(); ++set) {
    context_BndryL2[set]->reset();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetParamJacobian() {
  context_param->reset();
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetParamStateJacobian() {
  for (size_t set=0; set<context_param_state.size(); ++set) {
    context_param_state[set]->reset();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetPrevJacobian() {
  for (size_t step=0; step<context_prev.size(); ++step) {
    for (size_t set=0; set<context_prev[step].size(); ++set) {
      context_prev[step][set]->reset();
    }
  }
}


// ========================================================================================
// ========================================================================================

template<class Node>
size_t LinearAlgebraInterface<Node>::getLocalNumElements(const size_t & set) {
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

template<class Node>
size_t LinearAlgebraInterface<Node>::getLocalNumParamElements() {
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

template<class Node>
Teuchos::RCP<Tpetra::CrsGraph<LO,GO,Node> > LinearAlgebraInterface<Node>::getNewOverlappedGraph(const size_t & set, vector<size_t> & maxEntriesPerRow) {
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

template<class Node>
Teuchos::RCP<Tpetra::CrsGraph<LO,GO,Node> > LinearAlgebraInterface<Node>::getNewParamOverlappedGraph(vector<size_t> & maxEntriesPerRow) {
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

template<class Node>
GO LinearAlgebraInterface<Node>::getGlobalElement(const size_t & set, const LO & lid) {
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

template<class Node>
GO LinearAlgebraInterface<Node>::getGlobalParamElement(const LO & lid) {
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

template<class Node>
bool LinearAlgebraInterface<Node>::getHaveOverlapped() {
  return have_overlapped;
}

// ========================================================================================
// ========================================================================================

template<class Node>
LO LinearAlgebraInterface<Node>::getOverlappedLID(const size_t & set, const GO & gid) {
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

template<class Node>
LO LinearAlgebraInterface<Node>::getOwnedLID(const size_t & set, const GO & gid) {
  return owned_map[set]->getLocalElement(gid);
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::writeToFile(matrix_RCP &J, vector_RCP &r, vector_RCP &soln,
                                               const std::string &jac_filename,
                                               const std::string &res_filename,
                                               const std::string &sol_filename) {
  Teuchos::TimeMonitor localtimer(*writefiletimer);
  
  if (do_dump_jacobian) {
    Tpetra::MatrixMarket::Writer<LA_CrsMatrix>::writeSparseFile(jac_filename,*J);
  }
  if (do_dump_residual) {
    Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(res_filename,*r);
  }
  if (do_dump_solution) {
    Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(sol_filename,*soln);
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::writeStateToFile(vector<vector_RCP> & soln,
                                                    const std::string & filebase, const int & stepnum) {
  Teuchos::TimeMonitor localtimer(*writefiletimer);
  
  for (size_t set=0; set<soln.size(); ++set) {
    std::stringstream ss;
    ss << filebase << "." << set << "." << stepnum << ".mm";
    
    Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(ss.str(),*(soln[set]));
  }
}
// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::writeVectorToFile(ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > & vec, string & filename) {
  Teuchos::TimeMonitor localtimer(*writefiletimer);
  
  Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(filename,vec->getVector());
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::readParameterVectorFromFile(const std::string & filename) {
  Teuchos::TimeMonitor localtimer(*readfiletimer);
  
  vector_RCP vec = Tpetra::MatrixMarket::Reader<LA_MultiVector>::readDenseFile(filename, comm, param_owned_map);
  return vec;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::readStateVectorFromFile(const std::string & filename, const size_t & set) {
  Teuchos::TimeMonitor localtimer(*readfiletimer);
  
  vector_RCP vec = Tpetra::MatrixMarket::Reader<LA_MultiVector>::readDenseFile(filename, comm, owned_map[set]);
  return vec;
}
