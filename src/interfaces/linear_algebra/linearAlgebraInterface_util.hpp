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
  if (options[set]->reuse_jacobian && options[set]->have_jacobian) {
    reuse = true;
  }
  return reuse;
}

// ========================================================================================
// All iterative solvers use the same Belos list.  This would be easy to specialize.
// ========================================================================================

template<class Node>
Teuchos::RCP<Teuchos::ParameterList> LinearAlgebraInterface<Node>::getBelosParameterList(const string & belosSublist) {
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
  
  if (settings->sublist("Solver").isSublist(belosSublist)) {
    Teuchos::ParameterList inputParams = settings->sublist("Solver").sublist(belosSublist);
    belosList->setParameters(inputParams);
  }
  
  return belosList;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetJacobian() {
  for (size_t set=0; set<options.size(); ++set) {
    this->resetJacobian(set);
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetJacobian(const size_t & set) {
  options[set]->have_jacobian = false;
  options[set]->have_previous_jacobian = false;
  options[set]->have_preconditioner = false;
  options[set]->jac = Teuchos::null;
  options[set]->prec = Teuchos::null;
  options[set]->prec_dd = Teuchos::null;
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
  
  if(do_dump_jacobian)
    Tpetra::MatrixMarket::Writer<LA_CrsMatrix>::writeSparseFile(jac_filename,*J);
  if(do_dump_residual)
    Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(res_filename,*r);
  if(do_dump_solution)
    Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(sol_filename,*soln);
}
