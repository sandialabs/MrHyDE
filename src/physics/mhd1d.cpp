// **********************************************************************
//  This is a framework for solving Multi-resolution Hybridized
//  Differential Equations (MrHyDE), an optimized version of
//  Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)

//  Copyright 2018 National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
//  U.S. Government retains certain rights in this software.”

//  Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
//  Bart van Bloemen Waanders (bartv@sandia.gov)
//  ************************************************************************/

// MHD1D Physics module. Not working yet.
#define MrHyDE_ENABLE_MHD1D
#if defined(MrHyDE_ENABLE_MHD1D)
#include "mhd1d.hpp"
using namespace MrHyDE;

// ========================================================================================
// Constructor to set up the problem
// ========================================================================================

mhd1d::mhd1d(Teuchos::ParameterList &settings, const int &dimension_)
    : physicsbase(settings, dimension_)
{

    label = "mhd1d";

    myvars.push_back("By");
    myvars.push_back("Bz");
    myvars.push_back("rho");
    myvars.push_back("ux");
    myvars.push_back("uy");
    myvars.push_back("uz");
    myvars.push_back("E");

    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");

    /* useSUPG = settings.get<bool>("useSUPG", false);
    usePSPG = settings.get<bool>("usePSPG", false);
    T_ambient = settings.get<ScalarT>("T_ambient", 0.0);
    beta = settings.get<ScalarT>("beta", 1.0);
    model_params = Kokkos::View<ScalarT *, AssemblyDevice>("NS params on device", 2);
    auto host_params = create_mirror_view(model_params);
    host_params(0) = T_ambient;
    host_params(1) = beta;
    deep_copy(model_params, host_params); */
}

// ========================================================================================
// ========================================================================================

void mhd1d::defineFunctions(Teuchos::ParameterList &fs,
                            Teuchos::RCP<FunctionManager> &functionManager_)
{

    functionManager = functionManager_;

    functionManager->addFunction("Bx", fs.get<string>("Bx", "1.0"), "ip");
    // Viscosity of air at 273 kelvin
    functionManager->addFunction("viscosity", fs.get<string>("viscosity", "0.0000146"), "ip");
    functionManager->addFunction("permeability", fs.get<string>("permeability", "1.2e-6"), "ip");
    functionManager->addFunction("resistivity", fs.get<string>("resistivity", "0.0"), "ip");
    functionManager->addFunction("adiabatic index", fs.get<string>("adiabatic index", "1.667"), "ip");
    // Specific heat at constant pressure for one mole of gas with adiabatic index 5/3 at 273 kelvin
    functionManager->addFunction("specific heat", fs.get<string>("specific heat", "20.785"), "ip");
    // Heat diffusivity of air at 273 kelvin
    functionManager->addFunction("diffusivity", fs.get<string>("diffusivity", "0.000019"), "ip");
}

// ========================================================================================
// ========================================================================================

void mhd1d::volumeResidual()
{

    int spaceDim = wkset->dimension;
    ScalarT dt = wkset->deltat;
    bool isTransient = wkset->isTransient;
    Vista Bx, visc, mu, eta, gamma, Cp, k;
    {
        Teuchos::TimeMonitor funceval(*volumeResidualFunc);
        Bx = functionManager->evaluate("Bx", "ip");
        visc = functionManager->evaluate("viscosity", "ip");
        mu = functionManager->evaluate("permeability", "ip");
        eta = functionManager->evaluate("resistivity", "ip");
        gamma = functionManager->evaluate("adiabatic index", "ip");
        Cp = functionManager->evaluate("specific heat", "ip");
        k = functionManager->evaluate("diffusivity", "ip");
    }

    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    auto wts = wkset->wts;
    auto res = wkset->res;

    { // density
        int rho_basis = wkset->usebasis[rho_num];
        auto basis = wkset->basis[rho_basis];
        auto basis_grad = wkset->basis_grad[rho_basis];
        auto rho = wkset->getSolutionField("rho");
        auto drho_dt = wkset->getSolutionField("rho_t");
        auto ux = wkset->getSolutionField("ux");

        auto off = subview(wkset->offsets, rho_num, ALL());

        parallel_for(
            "MHD1D density volume residual",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = rho(elem,pt)*ux(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = drho_dt(elem, pt); // time derivative of density
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += -Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // momentum eqn x
        int ux_basis = wkset->usebasis[ux_num];
        auto basis = wkset->basis[ux_basis];
        auto basis_grad = wkset->basis_grad[ux_basis];
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto uz = wkset->getSolutionField("uz");

        auto rho = wkset->getSolutionField("rho");
        auto E = wkset->getSolutionField("E");

        auto By = wkset->getSolutionField("By");
        auto Bz = wkset->getSolutionField("Bz");

        auto dux_dt = wkset->getSolutionField("ux_t");
        auto dux_dx = wkset->getSolutionField("grad(ux)[x]");

        auto off = subview(wkset->offsets, ux_num, ALL());

        parallel_for(
            "MHD1D ux volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD norm_B = Bx(elem, pt)*Bx(elem, pt) + By(elem, pt)*By(elem, pt) + Bz(elem, pt)*Bz(elem, pt);
                    AD norm_u = ux(elem, pt)*ux(elem, pt) + uy(elem, pt)*uy(elem, pt) + uz(elem, pt)*uz(elem, pt);
                    AD pr = E(elem, pt) - rho(elem, pt)*norm_u/2 - norm_B/(2*mu(elem, pt));
                    pr *= gamma(elem, pt) - 1;
                    AD Fx = -4*visc(elem,pt)*dux_dx(elem,pt)/3 +
                            pr + (norm_B/2 - Bx(elem, pt)*Bx(elem, pt))/mu(elem,pt);
                    Fx *= wts(elem, pt);
                    AD F = rho(elem, pt)*(dux_dt(elem, pt) + ux(elem, pt)*dux_dx(elem, pt));
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += -Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // momentum eqn y
        int uy_basis = wkset->usebasis[uy_num];
        auto basis = wkset->basis[uy_basis];
        auto basis_grad = wkset->basis_grad[uy_basis];
        auto rho = wkset->getSolutionField("rho");
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");

        auto By = wkset->getSolutionField("By");
        auto Bz = wkset->getSolutionField("Bz");

        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        auto duy_dt = wkset->getSolutionField("uy_t");

        auto off = subview(wkset->offsets, uy_num, ALL());

        parallel_for(
            "MHD1D uy volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = -visc(elem, pt)*duy_dx(elem, pt) - Bx(elem, pt)*By(elem, pt)/mu(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = rho(elem, pt)*(duy_dt(elem, pt) + ux(elem, pt)*duy_dx(elem, pt));
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += -Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // momentum eqn z
        int uz_basis = wkset->usebasis[uz_num];
        auto basis = wkset->basis[uz_basis];
        auto basis_grad = wkset->basis_grad[uz_basis];
        auto rho = wkset->getSolutionField("rho");
        auto ux = wkset->getSolutionField("ux");
        auto uz = wkset->getSolutionField("uz");

        auto By = wkset->getSolutionField("By");
        auto Bz = wkset->getSolutionField("Bz");

        auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
        auto duz_dt = wkset->getSolutionField("uz_t");

        auto off = subview(wkset->offsets, uz_num, ALL());

        parallel_for(
            "MHD1D uz volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = -visc(elem, pt)*duz_dx(elem, pt) - Bx(elem, pt)*Bz(elem, pt)/mu(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = rho(elem, pt)*(duz_dt(elem, pt) + ux(elem, pt)*duz_dx(elem, pt));
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += -Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // magnet field y
        int By_basis = wkset->usebasis[By_num];
        auto basis = wkset->basis[By_basis];
        auto basis_grad = wkset->basis_grad[By_basis];
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");

        auto By = wkset->getSolutionField("By");

        auto dBy_dx = wkset->getSolutionField("grad(By)[x]");
        auto dBy_dt = wkset->getSolutionField("By_t");

        auto off = subview(wkset->offsets, By_num, ALL());

        parallel_for(
            "MHD1D By volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = (By(elem, pt) * ux(elem, pt) - Bx(elem, pt)*uy(elem, pt)) - dBy_dx(elem, pt)*eta(elem, pt)/mu(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = dBy_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += -Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // magnet field z
        int Bz_basis = wkset->usebasis[Bz_num];
        auto basis = wkset->basis[Bz_basis];
        auto basis_grad = wkset->basis_grad[Bz_basis];
        auto ux = wkset->getSolutionField("ux");
        auto uz = wkset->getSolutionField("uz");

        auto Bz = wkset->getSolutionField("Bz");

        auto dBz_dx = wkset->getSolutionField("grad(Bz)[x]");
        auto dBz_dt = wkset->getSolutionField("Bz_t");

        auto off = subview(wkset->offsets, Bz_num, ALL());

        parallel_for(
            "MHD1D Bz volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = Bz(elem, pt) * ux(elem, pt) - Bx(elem, pt) * uz(elem, pt) - dBz_dx(elem, pt)*eta(elem, pt)/mu(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = dBz_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += -Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    {// Energy
        int E_basis = wkset->usebasis[E_num];
        auto basis = wkset->basis[E_basis];
        auto basis_grad = wkset->basis_grad[E_basis];
        auto rho = wkset->getSolutionField("rho");

        
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto uz = wkset->getSolutionField("uz");

        auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        auto duz_dx = wkset->getSolutionField("grad(uz)[x]");

        auto By = wkset->getSolutionField("By");
        auto Bz = wkset->getSolutionField("Bz");
        auto dBy_dx = wkset->getSolutionField("grad(By)[x]");
        auto dBz_dx = wkset->getSolutionField("grad(Bz)[x]");

        auto E = wkset->getSolutionField("E");
        auto dE_dt = wkset->getSolutionField("E_t");

        auto off = subview(wkset->offsets, E_num, ALL());

        parallel_for(
            "MHD1D Energy volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD norm_u  = ux(elem, pt)*ux(elem, pt) + uy(elem, pt)*uy(elem, pt) + uz(elem, pt)*uz(elem, pt);
                    AD norm_du = dux_dx(elem, pt)*dux_dx(elem, pt) + duy_dx(elem, pt)*duy_dx(elem, pt) + duz_dx(elem, pt)*duz_dx(elem, pt);
                    AD norm_B  = Bx(elem, pt)*Bx(elem, pt) + By(elem, pt)*By(elem, pt) + Bz(elem, pt)*Bz(elem, pt);
                    AD B_dot_u = ux(elem, pt)*Bx(elem, pt) + uy(elem, pt)*By(elem, pt) + uz(elem, pt)*Bz(elem, pt);
                    AD pr = E(elem, pt) - rho(elem, pt)*norm_u/2 - norm_B/(2*mu(elem, pt));
                    pr *= gamma(elem, pt) - 1;
                    AD pr_star = pr + norm_B/(2*mu(elem, pt));
                    AD Fx = (E(elem, pt) + pr_star)*ux(elem, pt) - Bx(elem, pt)*B_dot_u +
                            4*visc(elem, pt)*rho(elem, pt)*ux(elem, pt)*dux_dx(elem, pt)/3 -
                            eta(elem, pt)*(Bz(elem, pt)*dBz_dx(elem, pt) + By(elem, pt)*dBy_dx(elem, pt))/mu(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = dE_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += -Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
}

// ========================================================================================
// ========================================================================================

void mhd1d::boundaryResidual()
{

    int spaceDim = wkset->dimension;
    auto bcs = wkset->var_bcs;

    int cside = wkset->currentside;

    string ux_sidetype = bcs(ux_num, cside);
    string uy_sidetype = "Dirichlet";
    string uz_sidetype = "Dirichlet";
    uy_sidetype = bcs(uy_num, cside);
    uz_sidetype = bcs(uz_num, cside);
    // TODO: Enforce that sidetypes are periodic
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

// TODO: Not sure why, but this causes error
// void mhd1d::computeFlux()
// {
// }

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

void mhd1d::setWorkset(Teuchos::RCP<workset> &wkset_)
{

    wkset = wkset_;

    vector<string> varlist = wkset->varlist;
    for (size_t i = 0; i < varlist.size(); i++)
    {
        if (varlist[i] == "rho")
            rho_num = i;
        if (varlist[i] == "ux")
            ux_num = i;
        if (varlist[i] == "uy")
            uy_num = i;
        if (varlist[i] == "uz")
            uz_num = i;
        if (varlist[i] == "By")
            By_num = i;
        if (varlist[i] == "Bz")
            Bz_num = i;
        if (varlist[i] == "E")
            E_num = i;
    }
}

// ========================================================================================
// return the value of the stabilization parameter
// ========================================================================================

/* KOKKOS_FUNCTION AD mhd1d::computeTau(const AD &localdiff, const AD &xvl, const AD &yvl, const AD &zvl, const ScalarT &h, const int &spaceDim, const ScalarT &dt, const bool &isTransient) const
{

  ScalarT C1 = 4.0;
  ScalarT C2 = 2.0;
  ScalarT C3 = isTransient ? 2.0 : 0.0; // only if transient -- TODO not sure BWR

  AD nvel = 0.0;
  if (spaceDim == 1)
    nvel = xvl * xvl;
  else if (spaceDim == 2)
    nvel = xvl * xvl + yvl * yvl;
  else if (spaceDim == 3)
    nvel = xvl * xvl + yvl * yvl + zvl * zvl;

  if (nvel > 1E-12)
    nvel = sqrt(nvel);

  AD tau;
  // see, e.g. wikipedia article on SUPG/PSPG
  // coefficients can be changed/tuned for different scenarios (including order of time scheme)
  // https://arxiv.org/pdf/1710.08898.pdf had a good, clear writeup of the final eqns
  tau = (C1 * localdiff / h / h) * (C1 * localdiff / h / h) + (C2 * nvel / h) * (C2 * nvel / h) + (C3 / dt) * (C3 / dt);
  tau = 1. / sqrt(tau);

  return tau;
} */
#endif