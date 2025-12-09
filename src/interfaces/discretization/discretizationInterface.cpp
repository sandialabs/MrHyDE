/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "discretizationInterface.hpp"
#include "Panzer_NodalFieldPattern.hpp"
#include "Panzer_OrientationsInterface.hpp"

// HGRAD basis functions
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"

// HVOL basis functions
#include "Intrepid2_HVOL_C0_FEM.hpp"

// HDIV basis functions
#include "Intrepid2_HDIV_QUAD_In_FEM.hpp"
#include "Intrepid2_HDIV_HEX_In_FEM.hpp"
#include "Intrepid2_HDIV_TRI_In_FEM.hpp"
#include "Intrepid2_HDIV_TET_In_FEM.hpp"

// HDIV Arbogast-Correa basis functions
#include "Intrepid2_HDIV_AC_QUAD_I1_FEM.hpp"

// HCURL basis functions
#include "Intrepid2_HCURL_QUAD_In_FEM.hpp"
#include "Intrepid2_HCURL_HEX_In_FEM.hpp"
#include "Intrepid2_HCURL_TRI_In_FEM.hpp"
#include "Intrepid2_HCURL_TET_In_FEM.hpp"

// HFACE (experimental) basis functions
#include "Intrepid2_HFACE_QUAD_In_FEM.hpp"
#include "Intrepid2_HFACE_TRI_In_FEM.hpp"
#include "Intrepid2_HFACE_HEX_In_FEM.hpp"
#include "Intrepid2_HFACE_TET_In_FEM.hpp"

#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Utils.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"

#include "Panzer_STKConnManager.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_STK_SetupUtilities.hpp"

typedef Intrepid2::CellTools<PHX::Device::execution_space> CellTools;
typedef Intrepid2::FunctionSpaceTools<PHX::Device::execution_space> FuncTools;
typedef Intrepid2::OrientationTools<PHX::Device::execution_space> OrientTools;
typedef Intrepid2::RealSpaceTools<PHX::Device::execution_space> RealTools;
typedef Intrepid2::ArrayTools<PHX::Device::execution_space> ArrayTools;

using namespace MrHyDE;

#include "discretizationInterface_basis.hpp"
#include "discretizationInterface_construct.hpp"
#include "discretizationInterface_dof.hpp"
#include "discretizationInterface_integration.hpp"
#include "discretizationInterface_util.hpp"
