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

    myvars.push_back("rho");
    myvars.push_back("rho_ux");
    myvars.push_back("rho_uy");
    myvars.push_back("rho_uz");
    myvars.push_back("By");
    myvars.push_back("Bz");
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

    functionManager->addFunction("Bx", fs.get<string>("Bx", "0.0"), "ip");
    functionManager->addFunction("gamma", fs.get<string>("permeability", "2.0"), "ip");
}

// ========================================================================================
// ========================================================================================

void mhd1d::volumeResidual()
{

    int spaceDim = wkset->dimension;
    ScalarT dt = wkset->deltat;
    bool isTransient = wkset->isTransient;
    Vista Bx, gamma;

    {
        Teuchos::TimeMonitor funceval(*volumeResidualFunc);
        Bx = functionManager->evaluate("Bx", "ip");
        gamma = functionManager->evaluate("gamma", "ip");
    }

    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    auto wts = wkset->wts;
    auto res = wkset->res;

    { // density
        int rho_basis = wkset->usebasis[rho_num];
        auto basis = wkset->basis[rho_basis];
        auto basis_grad = wkset->basis_grad[rho_basis];
        auto rho_ux = wkset->getSolutionField("rho_ux");
        auto drho_dt = wkset->getSolutionField("rho_t");

        auto off = subview(wkset->offsets, rho_num, ALL());

        // Ux equation
        parallel_for(
            "MHD1D density volume residual",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = rho_ux(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = drho_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // momentum eqn x
        int rho_ux_basis = wkset->usebasis[rho_ux_num];
        auto basis = wkset->basis[rho_ux_basis];
        auto basis_grad = wkset->basis_grad[rho_ux_basis];
        auto rho = wkset->getSolutionField("rho");
        auto rho_ux = wkset->getSolutionField("rho_ux");
        auto rho_uy = wkset->getSolutionField("rho_uy");
        auto rho_uz = wkset->getSolutionField("rho_uz");

        auto By = wkset->getSolutionField("By");
        auto Bz = wkset->getSolutionField("Bz");
        auto E = wkset->getSolutionField("E");

        auto drho_ux_dt = wkset->getSolutionField("rho_ux_t");

        auto off = subview(wkset->offsets, rho_ux_num, ALL());

        parallel_for(
            "MHD1D rho_ux volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD norm_B = Bx(elem, pt) * Bx(elem, pt) + By(elem, pt) * By(elem, pt) + Bz(elem, pt) * Bz(elem, pt);
                    AD norm_rhou = rho_ux(elem, pt) * rho_ux(elem, pt) + rho_uy(elem, pt) * rho_uy(elem, pt) + rho_uz(elem, pt) * rho_uz(elem, pt);
                    AD pressure = (gamma(elem, pt) - 1) * (E(elem, pt) - norm_rhou / (rho(elem, pt) * 2) - norm_B / 2);
                    AD p_star = pressure + norm_B / 2;
                    AD Fx = rho_ux(elem, pt) * rho_ux(elem, pt) / rho(elem, pt) + p_star - Bx(elem, pt) * Bx(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = drho_ux_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // momentum eqn y
        int rho_uy_basis = wkset->usebasis[rho_uy_num];
        auto basis = wkset->basis[rho_uy_basis];
        auto basis_grad = wkset->basis_grad[rho_uy_basis];
        auto rho = wkset->getSolutionField("rho");
        auto rho_ux = wkset->getSolutionField("rho_ux");
        auto rho_uy = wkset->getSolutionField("rho_uy");
        auto rho_uz = wkset->getSolutionField("rho_uz");

        auto By = wkset->getSolutionField("By");
        auto Bz = wkset->getSolutionField("Bz");
        auto E = wkset->getSolutionField("E");

        auto drho_uy_dt = wkset->getSolutionField("rho_uy_t");

        auto off = subview(wkset->offsets, rho_uy_num, ALL());

        parallel_for(
            "MHD1D rho_uy volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = rho_ux(elem, pt) * rho_uy(elem, pt) / rho(elem, pt) - Bx(elem, pt) * By(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = drho_uy_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // momentum eqn z
        int rho_uz_basis = wkset->usebasis[rho_uz_num];
        auto basis = wkset->basis[rho_uz_basis];
        auto basis_grad = wkset->basis_grad[rho_uz_basis];
        auto rho = wkset->getSolutionField("rho");
        auto rho_ux = wkset->getSolutionField("rho_ux");
        auto rho_uy = wkset->getSolutionField("rho_uy");
        auto rho_uz = wkset->getSolutionField("rho_uz");

        auto By = wkset->getSolutionField("By");
        auto Bz = wkset->getSolutionField("Bz");
        auto E = wkset->getSolutionField("E");

        auto drho_uz_dt = wkset->getSolutionField("rho_uz_t");

        auto off = subview(wkset->offsets, rho_uz_num, ALL());

        parallel_for(
            "MHD1D rho_uz volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = rho_ux(elem, pt) * rho_uz(elem, pt) / rho(elem, pt) - Bx(elem, pt) * Bz(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = drho_uz_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // magnet field y
        int By_basis = wkset->usebasis[By_num];
        auto basis = wkset->basis[By_basis];
        auto basis_grad = wkset->basis_grad[By_basis];
        auto rho = wkset->getSolutionField("rho");
        auto rho_ux = wkset->getSolutionField("rho_ux");
        auto rho_uy = wkset->getSolutionField("rho_uy");

        auto By = wkset->getSolutionField("By");

        auto dBy_dt = wkset->getSolutionField("By_t");

        auto off = subview(wkset->offsets, By_num, ALL());

        parallel_for(
            "MHD1D By volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = (By(elem, pt) * rho_ux(elem, pt) - Bx(elem, pt) * rho_uy(elem, pt)) / rho(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = dBy_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // magnet field z
        int Bz_basis = wkset->usebasis[Bz_num];
        auto basis = wkset->basis[Bz_basis];
        auto basis_grad = wkset->basis_grad[Bz_basis];
        auto rho = wkset->getSolutionField("rho");
        auto rho_ux = wkset->getSolutionField("rho_ux");
        auto rho_uz = wkset->getSolutionField("rho_uz");

        auto Bz = wkset->getSolutionField("Bz");

        auto dBz_dt = wkset->getSolutionField("Bz_t");

        auto off = subview(wkset->offsets, Bz_num, ALL());

        parallel_for(
            "MHD1D By volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = (Bz(elem, pt) * rho_ux(elem, pt) - Bx(elem, pt) * rho_uz(elem, pt)) / rho(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = dBz_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    {
        /////////////////////////////
        // energy equation
        /////////////////////////////

        int E_basis = wkset->usebasis[E_num];
        auto basis = wkset->basis[E_basis];
        auto basis_grad = wkset->basis_grad[E_basis];
        auto rho = wkset->getSolutionField("rho");
        auto rho_ux = wkset->getSolutionField("rho_ux");
        auto rho_uy = wkset->getSolutionField("rho_uy");
        auto rho_uz = wkset->getSolutionField("rho_uz");

        auto By = wkset->getSolutionField("By");
        auto Bz = wkset->getSolutionField("Bz");
        auto E = wkset->getSolutionField("E");
        auto dE_dt = wkset->getSolutionField("E_t");
        auto off = subview(wkset->offsets, E_num, ALL());

        parallel_for(
            "MHD1D E volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD norm_B = Bx(elem, pt) * Bx(elem, pt) + By(elem, pt) * By(elem, pt) + Bz(elem, pt) * Bz(elem, pt);
                    AD norm_rhou = rho_ux(elem, pt) * rho_ux(elem, pt) + rho_uy(elem, pt) * rho_uy(elem, pt) + rho_uz(elem, pt) * rho_uz(elem, pt);
                    AD pressure = (gamma(elem, pt) - 1) * (E(elem, pt) - norm_rhou / (rho(elem, pt) * 2) - norm_B / 2);
                    AD p_star = pressure + norm_B / 2;
                    AD B_dot_rhou = Bx(elem, pt) * rho_ux(elem, pt) + By(elem, pt) * rho_uy(elem, pt) + Bz(elem, pt) * rho_uz(elem, pt);
                    AD Fx = ((E(elem, pt) + p_star) * rho_ux(elem, pt) - Bx(elem, pt) * B_dot_rhou) / rho(elem, pt);
                    Fx *= wts(elem, pt);
                    AD F = dE_dt(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
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

    string rho_ux_sidetype = bcs(rho_ux_num, cside);
    string rho_uy_sidetype = "Dirichlet";
    string rho_uz_sidetype = "Dirichlet";
    rho_uy_sidetype = bcs(rho_uy_num, cside);
    rho_uz_sidetype = bcs(rho_uz_num, cside);
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
        if (varlist[i] == "rho_ux")
            rho_ux_num = i;
        if (varlist[i] == "rho_uy")
            rho_uy_num = i;
        if (varlist[i] == "rho_uz")
            rho_uz_num = i;
        if (varlist[i] == "rho")
            rho_num = i;
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