/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "solverManager.hpp"

using namespace MrHyDE;

#include "solverManager_construct.hpp"
#include "solverManager_models.hpp"
#include "solverManager_setup.hpp"
#include "solverManager_solvers.hpp"
#include "solverManager_util.hpp"

// Explicit template instantiations
template class MrHyDE::SolverManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
template class MrHyDE::SolverManager<SubgridSolverNode>;
#endif
