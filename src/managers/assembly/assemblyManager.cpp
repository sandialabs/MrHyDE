/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

/** \file assemblyManager.hpp
 *  \brief Contains all of the assembly routines in MrHyDE and creates element groups and worksets.
 *  \author Created by T. Wildey
 */

#include "assemblyManager.hpp"

using namespace MrHyDE;

#include "assemblyManager_construct.hpp"
#include "assemblyManager_constraints.hpp"
#include "assemblyManager_database.hpp"
#include "assemblyManager_functions.hpp"
#include "assemblyManager_gather.hpp"
#include "assemblyManager_groups.hpp"
#include "assemblyManager_initial.hpp"
#include "assemblyManager_jacres.hpp"
#include "assemblyManager_mass.hpp"
#include "assemblyManager_scatter.hpp"
#include "assemblyManager_util.hpp"
#include "assemblyManager_workset.hpp"

template class MrHyDE::AssemblyManager<SolverNode>;
#if MrHyDE_REQ_SUBGRID_ETI
template class MrHyDE::AssemblyManager<SubgridSolverNode>;
#endif

template void AssemblyManager<SolverNode>::performGather(const size_t & set, const size_t & block, const size_t & grp,
                                                         Kokkos::View<ScalarT*,AssemblyDevice> vec_dev,
                                                         const int & type, const size_t & local_entry);

template void AssemblyManager<SolverNode>::performBoundaryGather(const size_t & set, const size_t & block, const size_t & grp,
                                                                 Kokkos::View<ScalarT*,AssemblyDevice> vec_dev,
                                                                 const int & type, const size_t & local_entry);
