/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "physicsInterface.hpp"
#include "physicsImporter.hpp"

using namespace MrHyDE;

#include "physicsInterface_boundary_conditions.hpp"
#include "physicsInterface_construct.hpp"
#include "physicsInterface_functions.hpp"
#include "physicsInterface_initial.hpp"
#include "physicsInterface_residual.hpp"
#include "physicsInterface_util.hpp"
#include "physicsInterface_workset.hpp"
