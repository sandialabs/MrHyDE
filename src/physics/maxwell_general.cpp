/***********************************************************************
MrHyDE - a framework for solving Multi-resolution Hybridized
Differential Equations and enabling beyond forward simulation for
large-scale multiphysics and multiscale systems.

Questions? Contact Tim Wildey (tmwilde@sandia.gov)
************************************************************************/

#include "maxwell_general.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

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
	incident_source_sideset = settings.get<string>("incident source sideset", "");

	if (settings.isSublist("Planewaves")) {
		Teuchos::ParameterList & planewaves = settings.sublist("Planewaves");
		Teuchos::ParameterList::ConstIterator source_itr = planewaves.begin();
		while (source_itr != planewaves.end()) {
			const string source_name = source_itr->first;
			TEUCHOS_TEST_FOR_EXCEPTION(!planewaves.isSublist(source_name),
			std::runtime_error,
			"Each Physics: Planewaves entry must be a sublist.");

			const string sideset =
			planewaves.sublist(source_name).get<string>("sideset", "");
			TEUCHOS_TEST_FOR_EXCEPTION(sideset.empty(), std::runtime_error,
			"Planewave '" << source_name << "' requires a sideset name.");

			if (std::find(automatic_planewave_sidesets.begin(),
			              automatic_planewave_sidesets.end(),
			              sideset) == automatic_planewave_sidesets.end()) {
				automatic_planewave_sidesets.push_back(sideset);
			}
			++source_itr;
		}
	}

    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 3, std::runtime_error, "Error: maxwell_general only runs in 3D");
	TEUCHOS_TEST_FOR_EXCEPTION(settings.get<bool>("use leap frog", false),
	std::runtime_error,
	"Error: maxwell_general does not support leap-frog staging");

	include_Eeqn = false;
	include_Heqn = false;
	has_constitutive_magnetoelectric_coupling = false;
	has_dissipative_magnetoelectric_coupling = false;

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
		const string value = getInput(compactName, aliasName, defaultValue);
		functionManager->addFunction(compactName, value, "ip");
		functionManager->addFunction(compactName, value, "side ip");
	};

	auto isConstantZero = [](const string & value) {
		const char * begin = value.c_str();
		while (*begin != '\0' &&
		       std::isspace(static_cast<unsigned char>(*begin))) {
			++begin;
		}

		char * end = nullptr;
		const double numeric_value = std::strtod(begin, &end);
		if (end == begin) {
			return false;
		}
		while (*end != '\0' &&
		       std::isspace(static_cast<unsigned char>(*end))) {
			++end;
		}
		return *end == '\0' && numeric_value == 0.0;
	};

	auto tensorIsActive = [&](const string & compactName,
	                          const string & aliasName) {
		if (!fs.isParameter(compactName) && !fs.isParameter(aliasName)) {
			return false;
		}
		return !isConstantZero(getInput(compactName, aliasName, "0.0"));
	};

	auto tensorHasActiveEntry = [&](const string & compactPrefix,
	                                const string & aliasPrefix) {
		const string entries[9] = {
			"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"};
		for (int entry = 0; entry < 9; ++entry) {
			if (tensorIsActive(compactPrefix + "_" + entries[entry],
			                   aliasPrefix + "_" + entries[entry])) {
				return true;
			}
		}
		return false;
	};

	functionManager->addFunction("c0", fs.get<string>("c0", "1.0"), "ip");
	functionManager->addFunction("eta0", fs.get<string>("eta0", "1.0"), "ip");
	functionManager->addFunction("c0", fs.get<string>("c0", "1.0"), "side ip");
	functionManager->addFunction("eta0", fs.get<string>("eta0", "1.0"), "side ip");

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

	addTensorEntry("xi_sigma_xx", "xisigma_xx", "0.0");
	addTensorEntry("xi_sigma_xy", "xisigma_xy", "0.0");
	addTensorEntry("xi_sigma_xz", "xisigma_xz", "0.0");
	addTensorEntry("xi_sigma_yx", "xisigma_yx", "0.0");
	addTensorEntry("xi_sigma_yy", "xisigma_yy", "0.0");
	addTensorEntry("xi_sigma_yz", "xisigma_yz", "0.0");
	addTensorEntry("xi_sigma_zx", "xisigma_zx", "0.0");
	addTensorEntry("xi_sigma_zy", "xisigma_zy", "0.0");
	addTensorEntry("xi_sigma_zz", "xisigma_zz", "0.0");

	addTensorEntry("zeta_rho_xx", "zetarho_xx", "0.0");
	addTensorEntry("zeta_rho_xy", "zetarho_xy", "0.0");
	addTensorEntry("zeta_rho_xz", "zetarho_xz", "0.0");
	addTensorEntry("zeta_rho_yx", "zetarho_yx", "0.0");
	addTensorEntry("zeta_rho_yy", "zetarho_yy", "0.0");
	addTensorEntry("zeta_rho_yz", "zetarho_yz", "0.0");
	addTensorEntry("zeta_rho_zx", "zetarho_zx", "0.0");
	addTensorEntry("zeta_rho_zy", "zetarho_zy", "0.0");
	addTensorEntry("zeta_rho_zz", "zetarho_zz", "0.0");

	has_constitutive_magnetoelectric_coupling =
		tensorHasActiveEntry("xir", "xi_r") ||
		tensorHasActiveEntry("zetar", "zeta_r");
	has_dissipative_magnetoelectric_coupling =
		tensorHasActiveEntry("xi_sigma", "xisigma") ||
		tensorHasActiveEntry("zeta_rho", "zetarho");

	auto addSideFunction = [&](const string & name, const string & defaultValue) {
		functionManager->addFunction(name, fs.get<string>(name, defaultValue), "side ip");
	};

	auto addSideFunctionAlias = [&](const string & name,
		const string & aliasName,
		const string & defaultValue) {
		functionManager->addFunction(name,
		fs.get<string>(name, fs.get<string>(aliasName, defaultValue)),
		"side ip");
	};

	addSideFunctionAlias("source_theta", "theta", "0.0");
	addSideFunctionAlias("source_phi", "phi", "0.0");
	addSideFunctionAlias("source_te", "te", "0.0");
	addSideFunctionAlias("source_tm", "tm", "1.0");
	addSideFunctionAlias("source_amplitude", "amplitude", "1.0");
	addSideFunctionAlias("source_c0", "c0", "1.0");
	addSideFunctionAlias("source_frequency", "frequency", "1.0");
	addSideFunctionAlias("source_min_frequency", "min_frequency", "0.0");
	addSideFunctionAlias("source_max_frequency", "max_frequency", "1.0");
	addSideFunction("source_tau", "1.0");
	addSideFunctionAlias("source_offset", "offset", "0.0");
	addSideFunctionAlias("source_tm_phase", "tm_phase", "0.0");
	addSideFunction("source_kx", "0.0");
	addSideFunction("source_ky", "0.0");
	addSideFunction("source_kz", "1.0");
	addSideFunction("source_pte_x", "0.0");
	addSideFunction("source_pte_y", "1.0");
	addSideFunction("source_pte_z", "0.0");
	addSideFunction("source_ptm_x", "1.0");
	addSideFunction("source_ptm_y", "0.0");
	addSideFunction("source_ptm_z", "0.0");
	addSideFunction("source_polarization_norm", "1.0");
	addSideFunction("source_te_weight", "0.0");
	addSideFunction("source_tm_weight", "1.0");
	addSideFunction("source_u", "0.0");
	addSideFunction("source_a", "0.0");
	addSideFunction("source_envelope", "0.0");
	addSideFunction("source_sine_ramp", "0.0");
	addSideFunction("source_waveform_te", "0.0");
	addSideFunction("source_waveform_tm", "0.0");
	addSideFunction("source_Ex", "0.0");
	addSideFunction("source_Ey", "0.0");
	addSideFunction("source_Ez", "0.0");

	addSideFunction("incident Ex", "0.0");
	addSideFunction("incident Ey", "0.0");
	addSideFunction("incident Ez", "0.0");
	addSideFunction("incident Hx", "0.0");
	addSideFunction("incident Hy", "0.0");
	addSideFunction("incident Hz", "0.0");

	const size_t automatic_source_count =
		std::max(static_cast<size_t>(1), automatic_planewave_sidesets.size());
	for (size_t source = 0; source < automatic_source_count; ++source) {
		const string prefix = "automatic_planewave_" +
			std::to_string(source) + "_";
		addSideFunction(prefix + "Ex", "0.0");
		addSideFunction(prefix + "Ey", "0.0");
		addSideFunction(prefix + "Ez", "0.0");
		addSideFunction(prefix + "Hx", "0.0");
		addSideFunction(prefix + "Hy", "0.0");
		addSideFunction(prefix + "Hz", "0.0");
		addSideFunction(prefix + "source_waveform_te", "0.0");
		addSideFunction(prefix + "source_waveform_tm", "0.0");
		addSideFunction(prefix + "source_amplitude", "0.0");
		addSideFunction(prefix + "source_te", "0.0");
		addSideFunction(prefix + "source_tm", "0.0");
	}
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
	Vista<EvalT> xi_sigma_xx, xi_sigma_xy, xi_sigma_xz, xi_sigma_yx, xi_sigma_yy, xi_sigma_yz, xi_sigma_zx, xi_sigma_zy, xi_sigma_zz;
	Vista<EvalT> zeta_rho_xx, zeta_rho_xy, zeta_rho_xz, zeta_rho_yx, zeta_rho_yy, zeta_rho_yz, zeta_rho_zx, zeta_rho_zy, zeta_rho_zz;

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

		if (has_constitutive_magnetoelectric_coupling) {
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
		}

		if (has_dissipative_magnetoelectric_coupling) {
			xi_sigma_xx = functionManager->evaluate("xi_sigma_xx", "ip");
			xi_sigma_xy = functionManager->evaluate("xi_sigma_xy", "ip");
			xi_sigma_xz = functionManager->evaluate("xi_sigma_xz", "ip");
			xi_sigma_yx = functionManager->evaluate("xi_sigma_yx", "ip");
			xi_sigma_yy = functionManager->evaluate("xi_sigma_yy", "ip");
			xi_sigma_yz = functionManager->evaluate("xi_sigma_yz", "ip");
			xi_sigma_zx = functionManager->evaluate("xi_sigma_zx", "ip");
			xi_sigma_zy = functionManager->evaluate("xi_sigma_zy", "ip");
			xi_sigma_zz = functionManager->evaluate("xi_sigma_zz", "ip");

			zeta_rho_xx = functionManager->evaluate("zeta_rho_xx", "ip");
			zeta_rho_xy = functionManager->evaluate("zeta_rho_xy", "ip");
			zeta_rho_xz = functionManager->evaluate("zeta_rho_xz", "ip");
			zeta_rho_yx = functionManager->evaluate("zeta_rho_yx", "ip");
			zeta_rho_yy = functionManager->evaluate("zeta_rho_yy", "ip");
			zeta_rho_yz = functionManager->evaluate("zeta_rho_yz", "ip");
			zeta_rho_zx = functionManager->evaluate("zeta_rho_zx", "ip");
			zeta_rho_zy = functionManager->evaluate("zeta_rho_zy", "ip");
			zeta_rho_zz = functionManager->evaluate("zeta_rho_zz", "ip");
		}
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
	auto curlEx = wkset->getSolutionField("curl(E)[x]");
	auto curlEy = wkset->getSolutionField("curl(E)[y]");
	auto curlEz = wkset->getSolutionField("curl(E)[z]");

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

			EvalT rho_x = rho_xx(elem,pt)*Hx(elem,pt) + rho_xy(elem,pt)*Hy(elem,pt) + rho_xz(elem,pt)*Hz(elem,pt);
			EvalT rho_y = rho_yx(elem,pt)*Hx(elem,pt) + rho_yy(elem,pt)*Hy(elem,pt) + rho_yz(elem,pt)*Hz(elem,pt);
			EvalT rho_z = rho_zx(elem,pt)*Hx(elem,pt) + rho_zy(elem,pt)*Hy(elem,pt) + rho_zz(elem,pt)*Hz(elem,pt);

			EvalT fx = (1.0/c0(elem,pt)*mur_dHdt_x
			+ 1.0/eta0(elem,pt)*rho_x + curlEx(elem,pt)
			+ magnetic_current_x(elem,pt))*wts(elem,pt);
			EvalT fy = (1.0/c0(elem,pt)*mur_dHdt_y
			+ 1.0/eta0(elem,pt)*rho_y + curlEy(elem,pt)
			+ magnetic_current_y(elem,pt))*wts(elem,pt);
			EvalT fz = (1.0/c0(elem,pt)*mur_dHdt_z
			+ 1.0/eta0(elem,pt)*rho_z + curlEz(elem,pt)
			+ magnetic_current_z(elem,pt))*wts(elem,pt);

			for (size_type dof = 0; dof < basis.extent(1); ++dof) {
				res(elem,off(dof)) += fx*basis(elem,dof,pt,0);
				res(elem,off(dof)) += fy*basis(elem,dof,pt,1);
				res(elem,off(dof)) += fz*basis(elem,dof,pt,2);
			}
		}
		});

		if (has_constitutive_magnetoelectric_coupling) {
			parallel_for("maxwell_general H constitutive coupling",
			RangePolicy<AssemblyExec>(0, wkset->numElem),
			MRHYDE_LAMBDA(const int elem) {
			for (size_type pt = 0; pt < basis.extent(2); ++pt) {
				EvalT zetar_dEdt_x = zetar_xx(elem,pt)*dEx_dt(elem,pt) + zetar_xy(elem,pt)*dEy_dt(elem,pt) + zetar_xz(elem,pt)*dEz_dt(elem,pt);
				EvalT zetar_dEdt_y = zetar_yx(elem,pt)*dEx_dt(elem,pt) + zetar_yy(elem,pt)*dEy_dt(elem,pt) + zetar_yz(elem,pt)*dEz_dt(elem,pt);
				EvalT zetar_dEdt_z = zetar_zx(elem,pt)*dEx_dt(elem,pt) + zetar_zy(elem,pt)*dEy_dt(elem,pt) + zetar_zz(elem,pt)*dEz_dt(elem,pt);

				const EvalT fx = wts(elem,pt)*zetar_dEdt_x/c0(elem,pt);
				const EvalT fy = wts(elem,pt)*zetar_dEdt_y/c0(elem,pt);
				const EvalT fz = wts(elem,pt)*zetar_dEdt_z/c0(elem,pt);

				for (size_type dof = 0; dof < basis.extent(1); ++dof) {
					res(elem,off(dof)) += fx*basis(elem,dof,pt,0);
					res(elem,off(dof)) += fy*basis(elem,dof,pt,1);
					res(elem,off(dof)) += fz*basis(elem,dof,pt,2);
				}
			}
			});
		}

		if (has_dissipative_magnetoelectric_coupling) {
			parallel_for("maxwell_general H dissipative coupling",
			RangePolicy<AssemblyExec>(0, wkset->numElem),
			MRHYDE_LAMBDA(const int elem) {
			for (size_type pt = 0; pt < basis.extent(2); ++pt) {
				EvalT zeta_rho_Ex = zeta_rho_xx(elem,pt)*Ex(elem,pt) + zeta_rho_xy(elem,pt)*Ey(elem,pt) + zeta_rho_xz(elem,pt)*Ez(elem,pt);
				EvalT zeta_rho_Ey = zeta_rho_yx(elem,pt)*Ex(elem,pt) + zeta_rho_yy(elem,pt)*Ey(elem,pt) + zeta_rho_yz(elem,pt)*Ez(elem,pt);
				EvalT zeta_rho_Ez = zeta_rho_zx(elem,pt)*Ex(elem,pt) + zeta_rho_zy(elem,pt)*Ey(elem,pt) + zeta_rho_zz(elem,pt)*Ez(elem,pt);

				const EvalT fx = wts(elem,pt)*zeta_rho_Ex;
				const EvalT fy = wts(elem,pt)*zeta_rho_Ey;
				const EvalT fz = wts(elem,pt)*zeta_rho_Ez;

				for (size_type dof = 0; dof < basis.extent(1); ++dof) {
					res(elem,off(dof)) += fx*basis(elem,dof,pt,0);
					res(elem,off(dof)) += fy*basis(elem,dof,pt,1);
					res(elem,off(dof)) += fz*basis(elem,dof,pt,2);
				}
			}
			});
		}
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

			EvalT sigma_Ex = sigma_xx(elem,pt)*Ex(elem,pt) + sigma_xy(elem,pt)*Ey(elem,pt) + sigma_xz(elem,pt)*Ez(elem,pt);
			EvalT sigma_Ey = sigma_yx(elem,pt)*Ex(elem,pt) + sigma_yy(elem,pt)*Ey(elem,pt) + sigma_yz(elem,pt)*Ez(elem,pt);
			EvalT sigma_Ez = sigma_zx(elem,pt)*Ex(elem,pt) + sigma_zy(elem,pt)*Ey(elem,pt) + sigma_zz(elem,pt)*Ez(elem,pt);

			EvalT fx = (1.0/c0(elem,pt)*epsr_dEdt_x
			+ eta0(elem,pt)*sigma_Ex
			+ eta0(elem,pt)*current_x(elem,pt))*wts(elem,pt);
			EvalT fy = (1.0/c0(elem,pt)*epsr_dEdt_y
			+ eta0(elem,pt)*sigma_Ey
			+ eta0(elem,pt)*current_y(elem,pt))*wts(elem,pt);
			EvalT fz = (1.0/c0(elem,pt)*epsr_dEdt_z
			+ eta0(elem,pt)*sigma_Ez
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

		if (has_constitutive_magnetoelectric_coupling) {
			parallel_for("maxwell_general E constitutive coupling",
			RangePolicy<AssemblyExec>(0, wkset->numElem),
			MRHYDE_LAMBDA(const int elem) {
			for (size_type pt = 0; pt < basis.extent(2); ++pt) {
				EvalT xir_dHdt_x = xir_xx(elem,pt)*dHx_dt(elem,pt) + xir_xy(elem,pt)*dHy_dt(elem,pt) + xir_xz(elem,pt)*dHz_dt(elem,pt);
				EvalT xir_dHdt_y = xir_yx(elem,pt)*dHx_dt(elem,pt) + xir_yy(elem,pt)*dHy_dt(elem,pt) + xir_yz(elem,pt)*dHz_dt(elem,pt);
				EvalT xir_dHdt_z = xir_zx(elem,pt)*dHx_dt(elem,pt) + xir_zy(elem,pt)*dHy_dt(elem,pt) + xir_zz(elem,pt)*dHz_dt(elem,pt);

				const EvalT fx = wts(elem,pt)*xir_dHdt_x/c0(elem,pt);
				const EvalT fy = wts(elem,pt)*xir_dHdt_y/c0(elem,pt);
				const EvalT fz = wts(elem,pt)*xir_dHdt_z/c0(elem,pt);

				for (size_type dof = 0; dof < basis.extent(1); ++dof) {
					res(elem,off(dof)) += fx*basis(elem,dof,pt,0);
					res(elem,off(dof)) += fy*basis(elem,dof,pt,1);
					res(elem,off(dof)) += fz*basis(elem,dof,pt,2);
				}
			}
			});
		}

		if (has_dissipative_magnetoelectric_coupling) {
			parallel_for("maxwell_general E dissipative coupling",
			RangePolicy<AssemblyExec>(0, wkset->numElem),
			MRHYDE_LAMBDA(const int elem) {
			for (size_type pt = 0; pt < basis.extent(2); ++pt) {
				EvalT xi_sigma_Hx = xi_sigma_xx(elem,pt)*Hx(elem,pt) + xi_sigma_xy(elem,pt)*Hy(elem,pt) + xi_sigma_xz(elem,pt)*Hz(elem,pt);
				EvalT xi_sigma_Hy = xi_sigma_yx(elem,pt)*Hx(elem,pt) + xi_sigma_yy(elem,pt)*Hy(elem,pt) + xi_sigma_yz(elem,pt)*Hz(elem,pt);
				EvalT xi_sigma_Hz = xi_sigma_zx(elem,pt)*Hx(elem,pt) + xi_sigma_zy(elem,pt)*Hy(elem,pt) + xi_sigma_zz(elem,pt)*Hz(elem,pt);

				const EvalT fx = wts(elem,pt)*xi_sigma_Hx;
				const EvalT fy = wts(elem,pt)*xi_sigma_Hy;
				const EvalT fz = wts(elem,pt)*xi_sigma_Hz;

				for (size_type dof = 0; dof < basis.extent(1); ++dof) {
					res(elem,off(dof)) += fx*basis(elem,dof,pt,0);
					res(elem,off(dof)) += fy*basis(elem,dof,pt,1);
					res(elem,off(dof)) += fz*basis(elem,dof,pt,2);
				}
			}
			});
		}
	}
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void maxwell_general<EvalT>::boundaryResidual() {
	/*
	Important ABC limitation:
	The unit-coefficient Silver–Müller term is appropriate when the exterior boundary
	represents the normalized free-space radiation condition assumed by this formulation.
	It is not a generally exact absorbing condition for an anisotropic, bianisotropic,
	lossy, or impedance-mismatched exterior medium. For those cases, the boundary
	operator would need the exterior tangential admittance operator instead of simply E_t.
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

	const bool apply_manual_incident_source =
		!incident_source_sideset.empty() &&
		wkset->sidename == incident_source_sideset;

	int automatic_source_index = -1;
	for (size_t source = 0; source < automatic_planewave_sidesets.size();
	     ++source) {
		if (wkset->sidename == automatic_planewave_sidesets[source]) {
			automatic_source_index = static_cast<int>(source);
			break;
		}
	}
	const bool apply_automatic_incident_source =
		automatic_source_index >= 0;
	const bool apply_incident_source =
		apply_manual_incident_source || apply_automatic_incident_source;

	View_Sc2 nx = wkset->getScalarField("n[x]");
	View_Sc2 ny = wkset->getScalarField("n[y]");
	View_Sc2 nz = wkset->getScalarField("n[z]");

	auto Ex = wkset->getSolutionField("E[x]");
	auto Ey = wkset->getSolutionField("E[y]");
	auto Ez = wkset->getSolutionField("E[z]");

	Vista<EvalT> incidentEx;
	Vista<EvalT> incidentEy;
	Vista<EvalT> incidentEz;
	Vista<EvalT> incidentHx;
	Vista<EvalT> incidentHy;
	Vista<EvalT> incidentHz;
	Vista<EvalT> automaticIncidentEx;
	Vista<EvalT> automaticIncidentEy;
	Vista<EvalT> automaticIncidentEz;
	Vista<EvalT> automaticIncidentHx;
	Vista<EvalT> automaticIncidentHy;
	Vista<EvalT> automaticIncidentHz;

	if (apply_manual_incident_source) {
		Teuchos::TimeMonitor funceval(*boundaryResidualFunc);

		incidentEx = functionManager->evaluate("incident Ex", "side ip");
		incidentEy = functionManager->evaluate("incident Ey", "side ip");
		incidentEz = functionManager->evaluate("incident Ez", "side ip");
		incidentHx = functionManager->evaluate("incident Hx", "side ip");
		incidentHy = functionManager->evaluate("incident Hy", "side ip");
		incidentHz = functionManager->evaluate("incident Hz", "side ip");
	}

	if (apply_automatic_incident_source) {
		Teuchos::TimeMonitor funceval(*boundaryResidualFunc);

		const string prefix = "automatic_planewave_" +
			std::to_string(automatic_source_index) + "_";
		automaticIncidentEx =
			functionManager->evaluate(prefix + "Ex", "side ip");
		automaticIncidentEy =
			functionManager->evaluate(prefix + "Ey", "side ip");
		automaticIncidentEz =
			functionManager->evaluate(prefix + "Ez", "side ip");
		automaticIncidentHx =
			functionManager->evaluate(prefix + "Hx", "side ip");
		automaticIncidentHy =
			functionManager->evaluate(prefix + "Hy", "side ip");
		automaticIncidentHz =
			functionManager->evaluate(prefix + "Hz", "side ip");
	}

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

		EvalT fx = -(ny(elem,pt)*nce_z - nz(elem,pt)*nce_y);
		EvalT fy = -(nz(elem,pt)*nce_x - nx(elem,pt)*nce_z);
		EvalT fz = -(nx(elem,pt)*nce_y - ny(elem,pt)*nce_x);

		if (apply_incident_source) {
			EvalT totalIncidentEx = 0.0;
			EvalT totalIncidentEy = 0.0;
			EvalT totalIncidentEz = 0.0;
			EvalT totalIncidentHx = 0.0;
			EvalT totalIncidentHy = 0.0;
			EvalT totalIncidentHz = 0.0;

			if (apply_manual_incident_source) {
				totalIncidentEx += incidentEx(elem,pt);
				totalIncidentEy += incidentEy(elem,pt);
				totalIncidentEz += incidentEz(elem,pt);
				totalIncidentHx += incidentHx(elem,pt);
				totalIncidentHy += incidentHy(elem,pt);
				totalIncidentHz += incidentHz(elem,pt);
			}
			if (apply_automatic_incident_source) {
				totalIncidentEx += automaticIncidentEx(elem,pt);
				totalIncidentEy += automaticIncidentEy(elem,pt);
				totalIncidentEz += automaticIncidentEz(elem,pt);
				totalIncidentHx += automaticIncidentHx(elem,pt);
				totalIncidentHy += automaticIncidentHy(elem,pt);
				totalIncidentHz += automaticIncidentHz(elem,pt);
			}

			EvalT nceinc_x = ny(elem,pt)*totalIncidentEz - nz(elem,pt)*totalIncidentEy;
			EvalT nceinc_y = nz(elem,pt)*totalIncidentEx - nx(elem,pt)*totalIncidentEz;
			EvalT nceinc_z = nx(elem,pt)*totalIncidentEy - ny(elem,pt)*totalIncidentEx;

			EvalT nxnxEinc_x = ny(elem,pt)*nceinc_z - nz(elem,pt)*nceinc_y;
			EvalT nxnxEinc_y = nz(elem,pt)*nceinc_x - nx(elem,pt)*nceinc_z;
			EvalT nxnxEinc_z = nx(elem,pt)*nceinc_y - ny(elem,pt)*nceinc_x;

			EvalT nxHinc_x = ny(elem,pt)*totalIncidentHz - nz(elem,pt)*totalIncidentHy;
			EvalT nxHinc_y = nz(elem,pt)*totalIncidentHx - nx(elem,pt)*totalIncidentHz;
			EvalT nxHinc_z = nx(elem,pt)*totalIncidentHy - ny(elem,pt)*totalIncidentHx;

			fx += nxnxEinc_x - nxHinc_x;
			fy += nxnxEinc_y - nxHinc_y;
			fz += nxnxEinc_z - nxHinc_z;
		}

		fx *= wts(elem,pt);
		fy *= wts(elem,pt);
		fz *= wts(elem,pt);

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
