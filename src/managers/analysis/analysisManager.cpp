/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "analysisManager.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "uqManager.hpp"
#include "data.hpp"
#include "MrHyDE_Objective.hpp"
#include "MrHyDE_Stochastic_Objective.hpp"
#include "MrHyDE_OptVector.hpp"
#include "MrHyDE_Sample_Set_Reader.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_TrustRegionStep.hpp"
#include "ROL_Solver.hpp"
#include "ROL_StochasticProblem.hpp"

#include "MrHyDE_TeuchosBatchManager.hpp"
#include "ROL_MonteCarloGenerator.hpp"
#include "ROL_DistributionFactory.hpp"

#if defined(MrHyDE_ENABLE_HDSA)
#include "HDSA_Ptr.hpp"
#include "HDSA_Comm.hpp"
#include "HDSA_Random_Number_Generator.hpp"
#include "HDSA_MD_Data_Interface_MrHyDE.hpp"
#include "HDSA_MD_Opt_Prob_Interface_MrHyDE.hpp"
#include "HDSA_Output_Writer_MrHyDE.hpp"
#include "HDSA_Sparse_Matrix.hpp"
#include "HDSA_MD_Multi_State_u_Hyperparameter_Interface.hpp"
#include "HDSA_MD_u_Hyperparameter_Interface_MrHyDE.hpp"
#include "HDSA_MD_z_Hyperparameter_Interface_MrHyDE.hpp"
#include "HDSA_Prior_Operators_Interface_MrHyDE.hpp"
#include "HDSA_MD_u_Prior_Interface.hpp"
#include "HDSA_MD_Numeric_Laplacian_u_Prior_Interface.hpp"
#include "HDSA_MD_Lumped_Mass_u_Prior_Interface.hpp"
#include "HDSA_MD_Multi_State_u_Prior_Interface.hpp"
#include "HDSA_MD_z_Prior_Interface.hpp"
#include "HDSA_MD_Numeric_Laplacian_z_Prior_Interface.hpp"
#include "HDSA_MD_Vector_z_Prior_Interface.hpp"
#include "HDSA_MD_Prior_Sampling.hpp"
#include "HDSA_MD_Posterior_Sampling.hpp"
#include "HDSA_MD_Hessian_Analysis.hpp"
#include "HDSA_MD_Update.hpp"
#include "HDSA_MD_OUU_Data_Interface_MrHyDE.hpp"
#include "HDSA_MD_OUU_Opt_Prob_Interface_MrHyDE.hpp"
#include "HDSA_MD_OUU_Ensemble_Weighting_Matrix.hpp"
#include "HDSA_MD_OUU_Hyperparameter_Data_Interface.hpp"
#include "HDSA_MD_OUU_u_Prior_Interface.hpp"

#include "ROL_PrimalDualRisk.hpp"
#endif

using namespace MrHyDE;

#include "analysisManager_construct.hpp"
#include "analysisManager_solve.hpp"
#include "analysisManager_util.hpp"

