/***********************************************************************
MrHyDE - a framework for solving Multi-resolution Hybridized
Differential Equations and enabling beyond forward simulation for
large-scale multiphysics and multiscale systems.

Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

#include "maxwell_general.hpp"
using namespace MrHyDE;

template<class EvalT>
maxwell_general<EvalT>::maxwell_general(Teuchos::ParameterList & settings,
const int & dimension_)
: PhysicsBase<EvalT>(settings, dimension_)
{
    label = "maxwell_general";

    spaceDim = dimension_;
    Enum = -1;
    Hnum = -1;

    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 3, std::runtime_error, "Error: maxwell_general only runs in 3D");
	TEUCHOS_TEST_FOR_EXCEPTION(settings.get<bool>("use leap frog", false),
	std::runtime_error,
	"Error: maxwell_general does not support leap-frog staging");

	include_Eeqn = false;
	include_Heqn = false;

	if (settings.isParameter("active variables")) {
		string active = settings.get<string>("active variables");
		include_Eeqn = active.find("E") != std::string::npos;
		include_Heqn = active.find("H") != std::string::npos;
	}
	else {
		include_Eeqn = true;
		include_Heqn = true;
	}

	TEUCHOS_TEST_FOR_EXCEPTION(include_Eeqn != include_Heqn, std::runtime_error,
	"Error: maxwell_general requires both E and H active variables");

	if (include_Eeqn) {
		myvars.push_back("E");
		mybasistypes.push_back("HCURL");
	}
	if (include_Heqn) {
		myvars.push_back("H");
		mybasistypes.push_back("HDIV");
	}
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell_general<EvalT>::defineFunctions(
Teuchos::ParameterList & fs,
Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {

	functionManager = functionManager_;

	auto getInput = [&fs](const string & compactName,
	const string & aliasName,
	const string & defaultValue) {
	return fs.get<string>(compactName,
	fs.get<string>(aliasName, defaultValue));
	};

	auto addTensorEntry = [&](const string & compactName,
	const string & aliasName,
	const string & defaultValue) {
	functionManager->addFunction(compactName,
	getInput(compactName, aliasName, defaultValue),
	"ip");
	};

	functionManager->addFunction("c0", fs.get<string>("c0", "1.0"), "ip");
	functionManager->addFunction("eta0", fs.get<string>("eta0", "1.0"), "ip");

	functionManager->addFunction("current x",
	getInput("current x", "source J x", "0.0"), "ip");
	functionManager->addFunction("current y",
	getInput("current y", "source J y", "0.0"), "ip");
	functionManager->addFunction("current z",
	getInput("current z", "source J z", "0.0"), "ip");

	functionManager->addFunction("magnetic current x",
	getInput("magnetic current x", "source M x", "0.0"), "ip");
	functionManager->addFunction("magnetic current y",
	getInput("magnetic current y", "source M y", "0.0"), "ip");
	functionManager->addFunction("magnetic current z",
	getInput("magnetic current z", "source M z", "0.0"), "ip");

	addTensorEntry("epsr_xx", "eps_r_xx", "1.0");
	addTensorEntry("epsr_xy", "eps_r_xy", "0.0");
	addTensorEntry("epsr_xz", "eps_r_xz", "0.0");
	addTensorEntry("epsr_yx", "eps_r_yx", "0.0");
	addTensorEntry("epsr_yy", "eps_r_yy", "1.0");
	addTensorEntry("epsr_yz", "eps_r_yz", "0.0");
	addTensorEntry("epsr_zx", "eps_r_zx", "0.0");
	addTensorEntry("epsr_zy", "eps_r_zy", "0.0");
	addTensorEntry("epsr_zz", "eps_r_zz", "1.0");

	addTensorEntry("mur_xx", "mu_r_xx", "1.0");
	addTensorEntry("mur_xy", "mu_r_xy", "0.0");
	addTensorEntry("mur_xz", "mu_r_xz", "0.0");
	addTensorEntry("mur_yx", "mu_r_yx", "0.0");
	addTensorEntry("mur_yy", "mu_r_yy", "1.0");
	addTensorEntry("mur_yz", "mu_r_yz", "0.0");
	addTensorEntry("mur_zx", "mu_r_zx", "0.0");
	addTensorEntry("mur_zy", "mu_r_zy", "0.0");
	addTensorEntry("mur_zz", "mu_r_zz", "1.0");

	addTensorEntry("xir_xx", "xi_r_xx", "0.0");
	addTensorEntry("xir_xy", "xi_r_xy", "0.0");
	addTensorEntry("xir_xz", "xi_r_xz", "0.0");
	addTensorEntry("xir_yx", "xi_r_yx", "0.0");
	addTensorEntry("xir_yy", "xi_r_yy", "0.0");
	addTensorEntry("xir_yz", "xi_r_yz", "0.0");
	addTensorEntry("xir_zx", "xi_r_zx", "0.0");
	addTensorEntry("xir_zy", "xi_r_zy", "0.0");
	addTensorEntry("xir_zz", "xi_r_zz", "0.0");

	addTensorEntry("zetar_xx", "zeta_r_xx", "0.0");
	addTensorEntry("zetar_xy", "zeta_r_xy", "0.0");
	addTensorEntry("zetar_xz", "zeta_r_xz", "0.0");
	addTensorEntry("zetar_yx", "zeta_r_yx", "0.0");
	addTensorEntry("zetar_yy", "zeta_r_yy", "0.0");
	addTensorEntry("zetar_yz", "zeta_r_yz", "0.0");
	addTensorEntry("zetar_zx", "zeta_r_zx", "0.0");
	addTensorEntry("zetar_zy", "zeta_r_zy", "0.0");
	addTensorEntry("zetar_zz", "zeta_r_zz", "0.0");

	addTensorEntry("sigma_xx", "sigma_xx", "0.0");
	addTensorEntry("sigma_xy", "sigma_xy", "0.0");
	addTensorEntry("sigma_xz", "sigma_xz", "0.0");
	addTensorEntry("sigma_yx", "sigma_yx", "0.0");
	addTensorEntry("sigma_yy", "sigma_yy", "0.0");
	addTensorEntry("sigma_yz", "sigma_yz", "0.0");
	addTensorEntry("sigma_zx", "sigma_zx", "0.0");
	addTensorEntry("sigma_zy", "sigma_zy", "0.0");
	addTensorEntry("sigma_zz", "sigma_zz", "0.0");

	addTensorEntry("rho_xx", "rho_xx", "0.0");
	addTensorEntry("rho_xy", "rho_xy", "0.0");
	addTensorEntry("rho_xz", "rho_xz", "0.0");
	addTensorEntry("rho_yx", "rho_yx", "0.0");
	addTensorEntry("rho_yy", "rho_yy", "0.0");
	addTensorEntry("rho_yz", "rho_yz", "0.0");
	addTensorEntry("rho_zx", "rho_zx", "0.0");
	addTensorEntry("rho_zy", "rho_zy", "0.0");
	addTensorEntry("rho_zz", "rho_zz", "0.0");

	addTensorEntry("xisigma_xx", "xi_sigma_xx", "0.0");
	addTensorEntry("xisigma_xy", "xi_sigma_xy", "0.0");
	addTensorEntry("xisigma_xz", "xi_sigma_xz", "0.0");
	addTensorEntry("xisigma_yx", "xi_sigma_yx", "0.0");
	addTensorEntry("xisigma_yy", "xi_sigma_yy", "0.0");
	addTensorEntry("xisigma_yz", "xi_sigma_yz", "0.0");
	addTensorEntry("xisigma_zx", "xi_sigma_zx", "0.0");
	addTensorEntry("xisigma_zy", "xi_sigma_zy", "0.0");
	addTensorEntry("xisigma_zz", "xi_sigma_zz", "0.0");

	addTensorEntry("zetarho_xx", "zeta_rho_xx", "0.0");
	addTensorEntry("zetarho_xy", "zeta_rho_xy", "0.0");
	addTensorEntry("zetarho_xz", "zeta_rho_xz", "0.0");
	addTensorEntry("zetarho_yx", "zeta_rho_yx", "0.0");
	addTensorEntry("zetarho_yy", "zeta_rho_yy", "0.0");
	addTensorEntry("zetarho_yz", "zeta_rho_yz", "0.0");
	addTensorEntry("zetarho_zx", "zeta_rho_zx", "0.0");
	addTensorEntry("zetarho_zy", "zeta_rho_zy", "0.0");
	addTensorEntry("zetarho_zz", "zeta_rho_zz", "0.0");
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell_general<EvalT>::volumeResidual() {

	Teuchos::TimeMonitor resideval(*volumeResidualFill);

	Vista<EvalT> c0, eta0;
	Vista<EvalT> current_x, current_y, current_z;
	Vista<EvalT> magnetic_current_x, magnetic_current_y, magnetic_current_z;
	Vista<EvalT> epsr_xx, epsr_xy, epsr_xz, epsr_yx, epsr_yy, epsr_yz, epsr_zx, epsr_zy, epsr_zz;
	Vista<EvalT> mur_xx, mur_xy, mur_xz, mur_yx, mur_yy, mur_yz, mur_zx, mur_zy, mur_zz;
	Vista<EvalT> xir_xx, xir_xy, xir_xz, xir_yx, xir_yy, xir_yz, xir_zx, xir_zy, xir_zz;
	Vista<EvalT> zetar_xx, zetar_xy, zetar_xz, zetar_yx, zetar_yy, zetar_yz, zetar_zx, zetar_zy, zetar_zz;
	Vista<EvalT> sigma_xx, sigma_xy, sigma_xz, sigma_yx, sigma_yy, sigma_yz, sigma_zx, sigma_zy, sigma_zz;
	Vista<EvalT> rho_xx, rho_xy, rho_xz, rho_yx, rho_yy, rho_yz, rho_zx, rho_zy, rho_zz;
	Vista<EvalT> xisigma_xx, xisigma_xy, xisigma_xz, xisigma_yx, xisigma_yy, xisigma_yz, xisigma_zx, xisigma_zy, xisigma_zz;
	Vista<EvalT> zetarho_xx, zetarho_xy, zetarho_xz, zetarho_yx, zetarho_yy, zetarho_yz, zetarho_zx, zetarho_zy, zetarho_zz;

	{
		Teuchos::TimeMonitor funceval(*volumeResidualFunc);

		c0 = functionManager->evaluate("c0", "ip");
		eta0 = functionManager->evaluate("eta0", "ip");

		current_x = functionManager->evaluate("current x", "ip");
		current_y = functionManager->evaluate("current y", "ip");
		current_z = functionManager->evaluate("current z", "ip");

		magnetic_current_x = functionManager->evaluate("magnetic current x", "ip");
		magnetic_current_y = functionManager->evaluate("magnetic current y", "ip");
		magnetic_current_z = functionManager->evaluate("magnetic current z", "ip");

		epsr_xx = functionManager->evaluate("epsr_xx", "ip");
		epsr_xy = functionManager->evaluate("epsr_xy", "ip");
		epsr_xz = functionManager->evaluate("epsr_xz", "ip");
		epsr_yx = functionManager->evaluate("epsr_yx", "ip");
		epsr_yy = functionManager->evaluate("epsr_yy", "ip");
		epsr_yz = functionManager->evaluate("epsr_yz", "ip");
		epsr_zx = functionManager->evaluate("epsr_zx", "ip");
		epsr_zy = functionManager->evaluate("epsr_zy", "ip");
		epsr_zz = functionManager->evaluate("epsr_zz", "ip");

		mur_xx = functionManager->evaluate("mur_xx", "ip");
		mur_xy = functionManager->evaluate("mur_xy", "ip");
		mur_xz = functionManager->evaluate("mur_xz", "ip");
		mur_yx = functionManager->evaluate("mur_yx", "ip");
		mur_yy = functionManager->evaluate("mur_yy", "ip");
		mur_yz = functionManager->evaluate("mur_yz", "ip");
		mur_zx = functionManager->evaluate("mur_zx", "ip");
		mur_zy = functionManager->evaluate("mur_zy", "ip");
		mur_zz = functionManager->evaluate("mur_zz", "ip");

		xir_xx = functionManager->evaluate("xir_xx", "ip");
		xir_xy = functionManager->evaluate("xir_xy", "ip");
		xir_xz = functionManager->evaluate("xir_xz", "ip");
		xir_yx = functionManager->evaluate("xir_yx", "ip");
		xir_yy = functionManager->evaluate("xir_yy", "ip");
		xir_yz = functionManager->evaluate("xir_yz", "ip");
		xir_zx = functionManager->evaluate("xir_zx", "ip");
		xir_zy = functionManager->evaluate("xir_zy", "ip");
		xir_zz = functionManager->evaluate("xir_zz", "ip");

		zetar_xx = functionManager->evaluate("zetar_xx", "ip");
		zetar_xy = functionManager->evaluate("zetar_xy", "ip");
		zetar_xz = functionManager->evaluate("zetar_xz", "ip");
		zetar_yx = functionManager->evaluate("zetar_yx", "ip");
		zetar_yy = functionManager->evaluate("zetar_yy", "ip");
		zetar_yz = functionManager->evaluate("zetar_yz", "ip");
		zetar_zx = functionManager->evaluate("zetar_zx", "ip");
		zetar_zy = functionManager->evaluate("zetar_zy", "ip");
		zetar_zz = functionManager->evaluate("zetar_zz", "ip");

		sigma_xx = functionManager->evaluate("sigma_xx", "ip");
		sigma_xy = functionManager->evaluate("sigma_xy", "ip");
		sigma_xz = functionManager->evaluate("sigma_xz", "ip");
		sigma_yx = functionManager->evaluate("sigma_yx", "ip");
		sigma_yy = functionManager->evaluate("sigma_yy", "ip");
		sigma_yz = functionManager->evaluate("sigma_yz", "ip");
		sigma_zx = functionManager->evaluate("sigma_zx", "ip");
		sigma_zy = functionManager->evaluate("sigma_zy", "ip");
		sigma_zz = functionManager->evaluate("sigma_zz", "ip");

		rho_xx = functionManager->evaluate("rho_xx", "ip");
		rho_xy = functionManager->evaluate("rho_xy", "ip");
		rho_xz = functionManager->evaluate("rho_xz", "ip");
		rho_yx = functionManager->evaluate("rho_yx", "ip");
		rho_yy = functionManager->evaluate("rho_yy", "ip");
		rho_yz = functionManager->evaluate("rho_yz", "ip");
		rho_zx = functionManager->evaluate("rho_zx", "ip");
		rho_zy = functionManager->evaluate("rho_zy", "ip");
		rho_zz = functionManager->evaluate("rho_zz", "ip");

		xisigma_xx = functionManager->evaluate("xisigma_xx", "ip");
		xisigma_xy = functionManager->evaluate("xisigma_xy", "ip");
		xisigma_xz = functionManager->evaluate("xisigma_xz", "ip");
		xisigma_yx = functionManager->evaluate("xisigma_yx", "ip");
		xisigma_yy = functionManager->evaluate("xisigma_yy", "ip");
		xisigma_yz = functionManager->evaluate("xisigma_yz", "ip");
		xisigma_zx = functionManager->evaluate("xisigma_zx", "ip");
		xisigma_zy = functionManager->evaluate("xisigma_zy", "ip");
		xisigma_zz = functionManager->evaluate("xisigma_zz", "ip");

		zetarho_xx = functionManager->evaluate("zetarho_xx", "ip");
		zetarho_xy = functionManager->evaluate("zetarho_xy", "ip");
		zetarho_xz = functionManager->evaluate("zetarho_xz", "ip");
		zetarho_yx = functionManager->evaluate("zetarho_yx", "ip");
		zetarho_yy = functionManager->evaluate("zetarho_yy", "ip");
		zetarho_yz = functionManager->evaluate("zetarho_yz", "ip");
		zetarho_zx = functionManager->evaluate("zetarho_zx", "ip");
		zetarho_zy = functionManager->evaluate("zetarho_zy", "ip");
		zetarho_zz = functionManager->evaluate("zetarho_zz", "ip");
	}

	auto dHx_dt = wkset->getSolutionField("H_t[x]");
	auto dHy_dt = wkset->getSolutionField("H_t[y]");
	auto dHz_dt = wkset->getSolutionField("H_t[z]");
	auto Hx = wkset->getSolutionField("H[x]");
	auto Hy = wkset->getSolutionField("H[y]");
	auto Hz = wkset->getSolutionField("H[z]");

	auto dEx_dt = wkset->getSolutionField("E_t[x]");
	auto dEy_dt = wkset->getSolutionField("E_t[y]");
	auto dEz_dt = wkset->getSolutionField("E_t[z]");
	auto Ex = wkset->getSolutionField("E[x]");
	auto Ey = wkset->getSolutionField("E[y]");
	auto Ez = wkset->getSolutionField("E[z]");
	auto curlE_x = wkset->getSolutionField("curl(E)[x]");
	auto curlE_y = wkset->getSolutionField("curl(E)[y]");
	auto curlE_z = wkset->getSolutionField("curl(E)[z]");

	auto wts = wkset->wts;
	auto res = wkset->res;

	if (include_Heqn) {
		int H_basis = wkset->usebasis[Hnum];
		auto basis = wkset->basis[H_basis];
		auto off = subview(wkset->offsets, Hnum, ALL());

		parallel_for("maxwell_general H volume resid",
		RangePolicy<AssemblyExec>(0, wkset->numElem),
		MRHYDE_LAMBDA(const int elem) {
		for (size_type pt = 0; pt < basis.extent(2); ++pt) {
			EvalT mur_dHdt_x = mur_xx(elem,pt)*dHx_dt(elem,pt) + mur_xy(elem,pt)*dHy_dt(elem,pt) + mur_xz(elem,pt)*dHz_dt(elem,pt);
			EvalT mur_dHdt_y = mur_yx(elem,pt)*dHx_dt(elem,pt) + mur_yy(elem,pt)*dHy_dt(elem,pt) + mur_yz(elem,pt)*dHz_dt(elem,pt);
			EvalT mur_dHdt_z = mur_zx(elem,pt)*dHx_dt(elem,pt) + mur_zy(elem,pt)*dHy_dt(elem,pt) + mur_zz(elem,pt)*dHz_dt(elem,pt);

			EvalT zetar_dEdt_x = zetar_xx(elem,pt)*dEx_dt(elem,pt) + zetar_xy(elem,pt)*dEy_dt(elem,pt) + zetar_xz(elem,pt)*dEz_dt(elem,pt);
			EvalT zetar_dEdt_y = zetar_yx(elem,pt)*dEx_dt(elem,pt) + zetar_yy(elem,pt)*dEy_dt(elem,pt) + zetar_yz(elem,pt)*dEz_dt(elem,pt);
			EvalT zetar_dEdt_z = zetar_zx(elem,pt)*dEx_dt(elem,pt) + zetar_zy(elem,pt)*dEy_dt(elem,pt) + zetar_zz(elem,pt)*dEz_dt(elem,pt);

			EvalT rho_H_x = rho_xx(elem,pt)*Hx(elem,pt) + rho_xy(elem,pt)*Hy(elem,pt) + rho_xz(elem,pt)*Hz(elem,pt);
			EvalT rho_H_y = rho_yx(elem,pt)*Hx(elem,pt) + rho_yy(elem,pt)*Hy(elem,pt) + rho_yz(elem,pt)*Hz(elem,pt);
			EvalT rho_H_z = rho_zx(elem,pt)*Hx(elem,pt) + rho_zy(elem,pt)*Hy(elem,pt) + rho_zz(elem,pt)*Hz(elem,pt);

			EvalT zetarho_E_x = zetarho_xx(elem,pt)*Ex(elem,pt) + zetarho_xy(elem,pt)*Ey(elem,pt) + zetarho_xz(elem,pt)*Ez(elem,pt);
			EvalT zetarho_E_y = zetarho_yx(elem,pt)*Ex(elem,pt) + zetarho_yy(elem,pt)*Ey(elem,pt) + zetarho_yz(elem,pt)*Ez(elem,pt);
			EvalT zetarho_E_z = zetarho_zx(elem,pt)*Ex(elem,pt) + zetarho_zy(elem,pt)*Ey(elem,pt) + zetarho_zz(elem,pt)*Ez(elem,pt);

			EvalT fx = (1.0/c0(elem,pt)*(mur_dHdt_x + zetar_dEdt_x)
			+ 1.0/eta0(elem,pt)*rho_H_x
			+ zetarho_E_x + curlE_x(elem,pt)
			+ magnetic_current_x(elem,pt))*wts(elem,pt);
			EvalT fy = (1.0/c0(elem,pt)*(mur_dHdt_y + zetar_dEdt_y)
			+ 1.0/eta0(elem,pt)*rho_H_y
			+ zetarho_E_y + curlE_y(elem,pt)
			+ magnetic_current_y(elem,pt))*wts(elem,pt);
			EvalT fz = (1.0/c0(elem,pt)*(mur_dHdt_z + zetar_dEdt_z)
			+ 1.0/eta0(elem,pt)*rho_H_z
			+ zetarho_E_z + curlE_z(elem,pt)
			+ magnetic_current_z(elem,pt))*wts(elem,pt);

			for (size_type dof = 0; dof < basis.extent(1); ++dof) {
				res(elem,off(dof)) += fx*basis(elem,dof,pt,0);
				res(elem,off(dof)) += fy*basis(elem,dof,pt,1);
				res(elem,off(dof)) += fz*basis(elem,dof,pt,2);
			}
		}
		});
	}

	if (include_Eeqn) {
		int E_basis = wkset->usebasis[Enum];
		auto basis = wkset->basis[E_basis];
		auto basis_curl = wkset->basis_curl[E_basis];
		auto off = subview(wkset->offsets, Enum, ALL());

		parallel_for("maxwell_general E volume resid",
		RangePolicy<AssemblyExec>(0, wkset->numElem),
		MRHYDE_LAMBDA(const int elem) {
		for (size_type pt = 0; pt < basis.extent(2); ++pt) {
			EvalT epsr_dEdt_x = epsr_xx(elem,pt)*dEx_dt(elem,pt) + epsr_xy(elem,pt)*dEy_dt(elem,pt) + epsr_xz(elem,pt)*dEz_dt(elem,pt);
			EvalT epsr_dEdt_y = epsr_yx(elem,pt)*dEx_dt(elem,pt) + epsr_yy(elem,pt)*dEy_dt(elem,pt) + epsr_yz(elem,pt)*dEz_dt(elem,pt);
			EvalT epsr_dEdt_z = epsr_zx(elem,pt)*dEx_dt(elem,pt) + epsr_zy(elem,pt)*dEy_dt(elem,pt) + epsr_zz(elem,pt)*dEz_dt(elem,pt);

			EvalT xir_dHdt_x = xir_xx(elem,pt)*dHx_dt(elem,pt) + xir_xy(elem,pt)*dHy_dt(elem,pt) + xir_xz(elem,pt)*dHz_dt(elem,pt);
			EvalT xir_dHdt_y = xir_yx(elem,pt)*dHx_dt(elem,pt) + xir_yy(elem,pt)*dHy_dt(elem,pt) + xir_yz(elem,pt)*dHz_dt(elem,pt);
			EvalT xir_dHdt_z = xir_zx(elem,pt)*dHx_dt(elem,pt) + xir_zy(elem,pt)*dHy_dt(elem,pt) + xir_zz(elem,pt)*dHz_dt(elem,pt);

			EvalT sigma_E_x = sigma_xx(elem,pt)*Ex(elem,pt) + sigma_xy(elem,pt)*Ey(elem,pt) + sigma_xz(elem,pt)*Ez(elem,pt);
			EvalT sigma_E_y = sigma_yx(elem,pt)*Ex(elem,pt) + sigma_yy(elem,pt)*Ey(elem,pt) + sigma_yz(elem,pt)*Ez(elem,pt);
			EvalT sigma_E_z = sigma_zx(elem,pt)*Ex(elem,pt) + sigma_zy(elem,pt)*Ey(elem,pt) + sigma_zz(elem,pt)*Ez(elem,pt);

			EvalT xisigma_H_x = xisigma_xx(elem,pt)*Hx(elem,pt) + xisigma_xy(elem,pt)*Hy(elem,pt) + xisigma_xz(elem,pt)*Hz(elem,pt);
			EvalT xisigma_H_y = xisigma_yx(elem,pt)*Hx(elem,pt) + xisigma_yy(elem,pt)*Hy(elem,pt) + xisigma_yz(elem,pt)*Hz(elem,pt);
			EvalT xisigma_H_z = xisigma_zx(elem,pt)*Hx(elem,pt) + xisigma_zy(elem,pt)*Hy(elem,pt) + xisigma_zz(elem,pt)*Hz(elem,pt);

			EvalT fx = (1.0/c0(elem,pt)*(epsr_dEdt_x + xir_dHdt_x)
			+ eta0(elem,pt)*sigma_E_x + xisigma_H_x
			+ eta0(elem,pt)*current_x(elem,pt))*wts(elem,pt);
			EvalT fy = (1.0/c0(elem,pt)*(epsr_dEdt_y + xir_dHdt_y)
			+ eta0(elem,pt)*sigma_E_y + xisigma_H_y
			+ eta0(elem,pt)*current_y(elem,pt))*wts(elem,pt);
			EvalT fz = (1.0/c0(elem,pt)*(epsr_dEdt_z + xir_dHdt_z)
			+ eta0(elem,pt)*sigma_E_z + xisigma_H_z
			+ eta0(elem,pt)*current_z(elem,pt))*wts(elem,pt);

			EvalT gx = -Hx(elem,pt)*wts(elem,pt);
			EvalT gy = -Hy(elem,pt)*wts(elem,pt);
			EvalT gz = -Hz(elem,pt)*wts(elem,pt);

			for (size_type dof = 0; dof < basis.extent(1); ++dof) {
				res(elem,off(dof)) += fx*basis(elem,dof,pt,0) + gx*basis_curl(elem,dof,pt,0);
				res(elem,off(dof)) += fy*basis(elem,dof,pt,1) + gy*basis_curl(elem,dof,pt,1);
				res(elem,off(dof)) += fz*basis(elem,dof,pt,2) + gz*basis_curl(elem,dof,pt,2);
			}
		}
		});
	}
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell_general<EvalT>::boundaryResidual() {
	/*
	Important ABC limitation:
	The unit-coefficient Silver–Müller term is appropriate when the exterior boundary
	represents the normalized free-space radiation condition used by Kairos. It is not
	a generally exact absorbing condition for an anisotropic, bianisotropic, lossy, or
	impedance-mismatched exterior medium. For those cases, the boundary operator would
	need the exterior tangential admittance operator instead of simply E_t.
	*/

	Teuchos::TimeMonitor resideval(*boundaryResidualFill);

	if (!include_Eeqn) {
		return;
	}

	auto bcs = wkset->var_bcs;
	int cside = wkset->currentside;

	if (bcs(Enum,cside) != "Neumann") {
		return;
	}

	View_Sc2 nx = wkset->getScalarField("n[x]");
	View_Sc2 ny = wkset->getScalarField("n[y]");
	View_Sc2 nz = wkset->getScalarField("n[z]");

	auto Ex = wkset->getSolutionField("E[x]");
	auto Ey = wkset->getSolutionField("E[y]");
	auto Ez = wkset->getSolutionField("E[z]");

	auto basis = wkset->basis_side[wkset->usebasis[Enum]];
	auto off = subview(wkset->offsets, Enum, ALL());
	auto wts = wkset->wts_side;
	auto res = wkset->res;

	parallel_for("maxwell_general boundary ABC",
	RangePolicy<AssemblyExec>(0, wkset->numElem),
	MRHYDE_LAMBDA(const int elem) {
	for (size_type pt = 0; pt < basis.extent(2); ++pt) {
		EvalT nce_x = ny(elem,pt)*Ez(elem,pt) - nz(elem,pt)*Ey(elem,pt);
		EvalT nce_y = nz(elem,pt)*Ex(elem,pt) - nx(elem,pt)*Ez(elem,pt);
		EvalT nce_z = nx(elem,pt)*Ey(elem,pt) - ny(elem,pt)*Ex(elem,pt);

		EvalT fx = -(ny(elem,pt)*nce_z - nz(elem,pt)*nce_y)*wts(elem,pt);
		EvalT fy = -(nz(elem,pt)*nce_x - nx(elem,pt)*nce_z)*wts(elem,pt);
		EvalT fz = -(nx(elem,pt)*nce_y - ny(elem,pt)*nce_x)*wts(elem,pt);

		for (size_type dof = 0; dof < basis.extent(1); ++dof) {
			res(elem,off(dof)) += fx*basis(elem,dof,pt,0);
			res(elem,off(dof)) += fy*basis(elem,dof,pt,1);
			res(elem,off(dof)) += fz*basis(elem,dof,pt,2);
		}
	}
	});
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell_general<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

	wkset = wkset_;

	vector<string> varlist = wkset->varlist;

	for (size_t i = 0; i < varlist.size(); ++i) {
		if (varlist[i] == "E") {
			Enum = i;
		}
		if (varlist[i] == "H") {
			Hnum = i;
		}
	}
}

//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::maxwell_general<ScalarT>;

#ifndef MrHyDE_NO_AD
template class MrHyDE::maxwell_general<AD>;
template class MrHyDE::maxwell_general<AD2>;
template class MrHyDE::maxwell_general<AD4>;
template class MrHyDE::maxwell_general<AD8>;
template class MrHyDE::maxwell_general<AD16>;
template class MrHyDE::maxwell_general<AD18>;
template class MrHyDE::maxwell_general<AD24>;
template class MrHyDE::maxwell_general<AD32>;
#endif
