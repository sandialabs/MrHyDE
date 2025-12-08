/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "workset.hpp"
#include "parameterManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_STKConnManager.hpp"

using namespace MrHyDE;

#include "parameterManager_access.hpp"
#include "parameterManager_construct.hpp"
#include "parameterManager_sacadoize.hpp"
#include "parameterManager_setup.hpp"
#include "parameterManager_update.hpp"
#include "parameterManager_util.hpp"

// Explicit template instantiations
template class MrHyDE::ParameterManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
  template class MrHyDE::ParameterManager<SubgridSolverNode>;
#endif
