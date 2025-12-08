/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "postprocessManager.hpp"

using namespace MrHyDE;

#include "postprocessManager_construct.hpp"
#include "postprocessManager_derived_quantities.hpp"
#include "postprocessManager_error_estimation.hpp"
#include "postprocessManager_exodus.hpp"
#include "postprocessManager_flux_response.hpp"
#include "postprocessManager_integrated_quantities.hpp"
#include "postprocessManager_objective_gradient.hpp"
#include "postprocessManager_objectives.hpp"
#include "postprocessManager_sensors.hpp"
#include "postprocessManager_util.hpp"


// Explicit template instantiations
template class MrHyDE::PostprocessManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
template class MrHyDE::PostprocessManager<SubgridSolverNode>;
#endif

