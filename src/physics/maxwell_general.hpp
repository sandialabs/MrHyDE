/***********************************************************************
MrHyDE - a framework for solving Multi-resolution Hybridized
Differential Equations and enabling beyond forward simulation for
large-scale multiphysics and multiscale systems.

Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

#ifndef MRHYDE_MAXWELL_GENERAL_H
#define MRHYDE_MAXWELL_GENERAL_H

#include "physicsBase.hpp"

namespace MrHyDE {

	/**
	* \brief Fully anisotropic and bianisotropic first-order Maxwell physics.
	*
	* This class solves a three-dimensional mixed first-order Maxwell system
	* using the normalized magnetic field H = eta0 * H_physical. The unknowns
	* are E in H(curl) and H in H(div).
	*
	* The volume equations are
	*
	*   mu_r/c0 * dH/dt + zeta_r/c0 * dE/dt + rho/eta0 * H
	*     + zeta_rho * E + curl(E) = -source_M
	*
	*   eps_r/c0 * dE/dt + xi_r/c0 * dH/dt + eta0 * sigma * E
	*     + xi_sigma * H - curl(H) = -eta0 * source_J.
	*
	* All material entries are scalar input functions. The compact function
	* names epsr_*, mur_*, xir_*, zetar_*, xisigma_*, and zetarho_* are used
	* internally. The corresponding Kairos-style aliases eps_r_*, mu_r_*,
	* xi_r_*, zeta_r_*, xi_sigma_*, and zeta_rho_* are accepted in the input.
	*/

	template<class EvalT>
	class maxwell_general : public PhysicsBase<EvalT> {
	public:

		using PhysicsBase<EvalT>::functionManager;
		using PhysicsBase<EvalT>::wkset;
		using PhysicsBase<EvalT>::label;
		using PhysicsBase<EvalT>::myvars;
		using PhysicsBase<EvalT>::mybasistypes;

		typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;

		maxwell_general() {};

		~maxwell_general() {};

		maxwell_general(Teuchos::ParameterList & settings, const int & dimension_);

		void defineFunctions(Teuchos::ParameterList & fs,
		Teuchos::RCP<FunctionManager<EvalT> > & functionManager_);

		void volumeResidual();

		void boundaryResidual();

		void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);

		private:

		int Enum, Hnum, spaceDim;
		bool include_Eeqn, include_Heqn;

		Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_general::volumeResidual() - function evaluation");
		Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_general::volumeResidual() - evaluation of residual");
		Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_general::boundaryResidual() - function evaluation");
		Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MrHyDE::maxwell_general::boundaryResidual() - evaluation of residual");

	};

}

#endif
