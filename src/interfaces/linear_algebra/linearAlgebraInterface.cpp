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

#include "linearAlgebraInterface_construct.hpp"
#include "linearAlgebraInterface_matrix.hpp"
#include "linearAlgebraInterface_solvers.hpp"
#include "linearAlgebraInterface_util.hpp"
#include "linearAlgebraInterface_vector.hpp"

// ========================================================================================
// ========================================================================================

// Explicit template instantiations
template class MrHyDE::LinearAlgebraInterface<SolverNode>;
template class MrHyDE::LinearSolverContext<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
template class MrHyDE::LinearAlgebraInterface<SubgridSolverNode>;
template class MrHyDE::LinearSolverContext<SubgridSolverNode>; 
#endif
