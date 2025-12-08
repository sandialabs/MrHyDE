/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "functionManager.hpp"

using namespace MrHyDE;

#include "functionManager_construct.hpp"
#include "functionManager_create.hpp"
#include "functionManager_evaluate.hpp"
#include "functionManager_util.hpp"

//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::FunctionManager<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::FunctionManager<AD>;

// Standard built-in types
template class MrHyDE::FunctionManager<AD2>;
template class MrHyDE::FunctionManager<AD4>;
template class MrHyDE::FunctionManager<AD8>;
template class MrHyDE::FunctionManager<AD16>;
template class MrHyDE::FunctionManager<AD18>;
template class MrHyDE::FunctionManager<AD24>;
template class MrHyDE::FunctionManager<AD32>;
#endif
