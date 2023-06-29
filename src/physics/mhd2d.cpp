/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)

 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”

 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "mhd2d.hpp"
using namespace MrHyDE;

// TODO BWR -- rho is both on the convective part but we have nu and rho*source showing up too
// this is inconsistent and needs fixing!

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

mhd2d::mhd2d(Teuchos::ParameterList &settings, const int &dimension_)
    : physicsbase(settings, dimension_)
{

    label = "mhd2d";
    int spaceDim = dimension_;

    myvars.push_back("pr");
    myvars.push_back("ux");
    myvars.push_back("uy");
    myvars.push_back("uz");
    myvars.push_back("T");
    myvars.push_back("Az");
    myvars.push_back("Bx");
    myvars.push_back("By");

    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
}

// ========================================================================================
// ========================================================================================

void mhd2d::defineFunctions(Teuchos::ParameterList &fs,
                            Teuchos::RCP<FunctionManager> &functionManager_)
{

    functionManager = functionManager_;

    functionManager->addFunction("source ux", fs.get<string>("source ux", "0.0"), "ip");
    functionManager->addFunction("source uy", fs.get<string>("source uy", "0.0"), "ip");
    functionManager->addFunction("source uz", fs.get<string>("source uz", "0.0"), "ip");
    functionManager->addFunction("density", fs.get<string>("density", "1.0"), "ip");
    functionManager->addFunction("viscosity", fs.get<string>("viscosity", "1.0"), "ip");
    functionManager->addFunction("specific heat", fs.get<string>("specific heat", "1.0"), "ip");
    functionManager->addFunction("thermal conductivity", fs.get<string>("thermal conductivity", "1.0"), "ip");
    functionManager->addFunction("resistivity", fs.get<string>("resistivity", "0.0"), "ip");
    functionManager->addFunction("permeability", fs.get<string>("permeability", "1.0"), "ip");
}

// ========================================================================================
// ========================================================================================

void mhd2d::volumeResidual()
{

    int spaceDim = wkset->dimension;
    ScalarT dt = wkset->deltat;
    bool isTransient = wkset->isTransient;
    Vista dens, visc, source_ux, source_uy, source_uz, Cp, cond, eta, mu;

    {
        Teuchos::TimeMonitor funceval(*volumeResidualFunc);
        source_ux = functionManager->evaluate("source ux", "ip");
        source_uy = functionManager->evaluate("source uy", "ip");
        source_uz = functionManager->evaluate("source uz", "ip");

        dens = functionManager->evaluate("density", "ip");
        visc = functionManager->evaluate("viscosity", "ip");
        Cp = functionManager->evaluate("Specific heat", "ip");
        cond = functionManager->evaluate("thermal conductivity", "ip");
        eta = functionManager->evaluate("resistivity", "ip");
        mu = functionManager->evaluate("permeability", "ip");
    }

    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    auto wts = wkset->wts;
    auto res = wkset->res;
    {// Ux equation
        int ux_basis = wkset->usebasis[ux_num];
        auto basis = wkset->basis[ux_basis];
        auto basis_grad = wkset->basis_grad[ux_basis];
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto uz = wkset->getSolutionField("uz");
        auto dux_dt = wkset->getSolutionField("ux_t");
        auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
        auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
        auto pr = wkset->getSolutionField("pr");
        auto off = subview(wkset->offsets, ux_num, ALL());

        parallel_for(
            "NS ux volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD div_u = dux_dx(elem, pt) + duy_dy(elem, pt);
                    AD norm_B = Bx(elem, pt)*Bx(elem, pt) + By(elem, pt)*By(elem, pt) + Bz(elem, pt)*Bz(elem, pt);
                    AD diag_contr = pr(elem, pt)-2*visc(elem, pt)*div_u/3 + norm_B/(2*mu(elem, pt));
                    AD Fx = visc(elem, pt) * dux_dx(elem, pt) - Bx(elem, pt)*Bx(elem, pt)/mu(elem, pt) + diag_contr;
                    AD Fy = visc(elem, pt) * dux_dy(elem, pt) - Bx(elem, pt)*By(elem, pt)/mu(elem, pt);
                    AD Fz = visc(elem, pt) * dux_dz(elem, pt) - Bx(elem, pt)*Bz(elem, pt)/mu(elem, pt);
                    Fx *= wts(elem, pt);
                    Fy *= wts(elem, pt);
                    Fz *= wts(elem, pt);
                    AD F = dux_dt(elem, pt) - source_ux(elem, pt) +
                           ux(elem, pt) * dux_dx(elem, pt) +
                           uy(elem, pt) * dux_dy(elem, pt) +
                           uz(elem, pt) * dux_dz(elem, pt) ;
                    F *= dens(elem, pt) * wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + Fy * basis_grad(elem, dof, pt, 1) + Fz * basis_grad(elem, dof, pt, 2) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
        // SUPG contribution
        if (useSUPG) // TODO
        {
            auto h = wkset->h;
            auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
            parallel_for(
                "NS ux volume resid",
                RangePolicy<AssemblyExec>(0, wkset->numElem),
                KOKKOS_LAMBDA(const int elem) {
                    for (size_type pt = 0; pt < basis.extent(2); pt++)
                    {
                        AD tau = this->computeTau(visc(elem, pt), ux(elem, pt), uy(elem, pt), uz(elem, pt), h(elem), spaceDim, dt, isTransient);
                        AD stabres = dens(elem, pt) * dux_dt(elem, pt) + dens(elem, pt) * (ux(elem, pt) * dux_dx(elem, pt) + uy(elem, pt) * dux_dy(elem, pt) + uz(elem, pt) * dux_dz(elem, pt)) + dpr_dx(elem, pt) - dens(elem, pt) * source_ux(elem, pt);
                        AD Sx = tau * stabres * ux(elem, pt) * wts(elem, pt);
                        AD Sy = tau * stabres * uy(elem, pt) * wts(elem, pt);
                        AD Sz = tau * stabres * uz(elem, pt) * wts(elem, pt);
                        for (size_type dof = 0; dof < basis.extent(1); dof++)
                        {
                            res(elem, off(dof)) += Sx * basis_grad(elem, dof, pt, 0) + Sy * basis_grad(elem, dof, pt, 1) + Sz * basis_grad(elem, dof, pt, 2);
                        }
                    }
                });

            if (have_energy)
            {
                auto params = model_params;
                auto E = wkset->getSolutionField("e");
                parallel_for(
                    "NS ux volume resid",
                    RangePolicy<AssemblyExec>(0, wkset->numElem),
                    KOKKOS_LAMBDA(const int elem) {
                        for (size_type pt = 0; pt < basis.extent(2); pt++)
                        {
                            AD tau = this->computeTau(visc(elem, pt), ux(elem, pt), uy(elem, pt), uz(elem, pt), h(elem), spaceDim, dt, isTransient);
                            AD stabres = dens(elem, pt) * params(1) * (E(elem, pt) - params(0)) * source_ux(elem, pt);
                            AD Sx = tau * stabres * ux(elem, pt) * wts(elem, pt);
                            AD Sy = tau * stabres * uy(elem, pt) * wts(elem, pt);
                            AD Sz = tau * stabres * uz(elem, pt) * wts(elem, pt);
                            for (size_type dof = 0; dof < basis.extent(1); dof++)
                            {
                                res(elem, off(dof)) += Sx * basis_grad(elem, dof, pt, 0) + Sy * basis_grad(elem, dof, pt, 1) + Sz * basis_grad(elem, dof, pt, 2);
                            }
                        }
                    });
            }
        }
    }

    {// Uy equation
        int uy_basis = wkset->usebasis[uy_num];
        auto basis = wkset->basis[uy_basis];
        auto basis_grad = wkset->basis_grad[uy_basis];
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto uz = wkset->getSolutionField("uz");
        auto duy_dt = wkset->getSolutionField("uy_t");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
        auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
        auto pr = wkset->getSolutionField("pr");
        auto off = subview(wkset->offsets, uy_num, ALL());

        parallel_for(
            "NS uy volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD div_u = dux_dx(elem, pt) + duy_dy(elem, pt);
                    AD norm_B = Bx(elem, pt)*Bx(elem, pt) + By(elem, pt)*By(elem, pt) + Bz(elem, pt)*Bz(elem, pt);
                    AD diag_contr = pr(elem, pt)-2*visc(elem, pt)*div_u/3 + norm_B/(2*mu(elem, pt));
                    AD Fx = visc(elem, pt) * duy_dx(elem, pt) - By(elem, pt)*Bx(elem, pt)/mu(elem, pt);
                    Fx *= wts(elem, pt);
                    AD Fy = visc(elem, pt) * duy_dy(elem, pt) - By(elem, pt)*By(elem, pt)/mu(elem, pt) + diag_contr;
                    Fy *= wts(elem, pt);
                    AD Fz = visc(elem, pt) * duy_dz(elem, pt) - By(elem, pt)*Bz(elem, pt)/mu(elem, pt);
                    Fz *= wts(elem, pt);
                    AD F = duy_dt(elem, pt) + ux(elem, pt) * duy_dx(elem, pt) + uy(elem, pt) * duy_dy(elem, pt) + uz(elem, pt) * duy_dz(elem, pt) - source_uy(elem, pt);
                    F *= dens(elem, pt) * wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + Fy * basis_grad(elem, dof, pt, 1) + Fz * basis_grad(elem, dof, pt, 2) + F * basis(elem, dof, pt, 0);
                    }
                }
            });

        // SUPG contribution

        if (useSUPG) // TODO
        {
            auto h = wkset->h;
            auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
            parallel_for(
                "NS uy volume resid",
                RangePolicy<AssemblyExec>(0, wkset->numElem),
                KOKKOS_LAMBDA(const int elem) {
                    for (size_type pt = 0; pt < basis.extent(2); pt++)
                    {
                        AD tau = this->computeTau(visc(elem, pt), ux(elem, pt), uy(elem, pt), uz(elem, pt), h(elem), spaceDim, dt, isTransient);
                        AD stabres = dens(elem, pt) * duy_dt(elem, pt) + dens(elem, pt) * (ux(elem, pt) * duy_dx(elem, pt) + uy(elem, pt) * duy_dy(elem, pt) + uz(elem, pt) * duy_dz(elem, pt)) + dpr_dy(elem, pt) - dens(elem, pt) * source_uy(elem, pt);
                        AD Sx = tau * stabres * ux(elem, pt) * wts(elem, pt);
                        AD Sy = tau * stabres * uy(elem, pt) * wts(elem, pt);
                        AD Sz = tau * stabres * uz(elem, pt) * wts(elem, pt);
                        for (size_type dof = 0; dof < basis.extent(1); dof++)
                        {
                            res(elem, off(dof)) += Sx * basis_grad(elem, dof, pt, 0) + Sy * basis_grad(elem, dof, pt, 1) + Sz * basis_grad(elem, dof, pt, 2);
                        }
                    }
                });

            if (have_energy)
            {
                auto params = model_params;
                auto E = wkset->getSolutionField("e");
                parallel_for(
                    "NS uy volume resid",
                    RangePolicy<AssemblyExec>(0, wkset->numElem),
                    KOKKOS_LAMBDA(const int elem) {
                        for (size_type pt = 0; pt < basis.extent(2); pt++)
                        {
                            AD tau = this->computeTau(visc(elem, pt), ux(elem, pt), uy(elem, pt), uz(elem, pt), h(elem), spaceDim, dt, isTransient);
                            AD stabres = dens(elem, pt) * params(1) * (E(elem, pt) - params(0)) * source_uy(elem, pt);
                            AD Sx = tau * stabres * ux(elem, pt) * wts(elem, pt);
                            AD Sy = tau * stabres * uy(elem, pt) * wts(elem, pt);
                            AD Sz = tau * stabres * uz(elem, pt) * wts(elem, pt);
                            for (size_type dof = 0; dof < basis.extent(1); dof++)
                            {
                                res(elem, off(dof)) += Sx * basis_grad(elem, dof, pt, 0) + Sy * basis_grad(elem, dof, pt, 1) + Sz * basis_grad(elem, dof, pt, 2);
                            }
                        }
                    });
            }
        }
    }

    {
        // Uz equation
        int uz_basis = wkset->usebasis[uz_num];
        auto basis = wkset->basis[uz_basis];
        auto basis_grad = wkset->basis_grad[uz_basis];
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto uz = wkset->getSolutionField("uz");
        auto duz_dt = wkset->getSolutionField("uz_t");
        auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
        auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
        auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
        auto pr = wkset->getSolutionField("pr");
        auto off = subview(wkset->offsets, uy_num, ALL());

        parallel_for(
            "NS uz volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD div_u = dux_dx(elem, pt) + duy_dy(elem, pt);
                    AD norm_B = Bx(elem, pt)*Bx(elem, pt) + By(elem, pt)*By(elem, pt) + Bz(elem, pt)*Bz(elem, pt);
                    AD diag_contr = pr(elem, pt)-2*visc(elem, pt)*div_u/3 + norm_B/(2*mu(elem, pt));
                    AD Fx = visc(elem, pt) * duz_dx(elem, pt) - Bz(elem, pt)*Bx(elem, pt)/mu(elem, pt);
                    Fx *= wts(elem, pt);
                    AD Fy = visc(elem, pt) * duz_dy(elem, pt) - Bz(elem, pt)*By(elem, pt)/mu(elem, pt);
                    Fy *= wts(elem, pt);
                    AD Fz = visc(elem, pt) * duz_dz(elem, pt) - Bz(elem, pt)*Bz(elem, pt)/mu(elem, pt) + diag_contr;
                    Fz *= wts(elem, pt);
                    AD F = duz_dt(elem, pt) + ux(elem, pt) * duz_dx(elem, pt) + uy(elem, pt) * duz_dy(elem, pt) + uz(elem, pt) * duz_dz(elem, pt) - source_uz(elem, pt);
                    F *= dens(elem, pt) * wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + Fy * basis_grad(elem, dof, pt, 1) + Fz * basis_grad(elem, dof, pt, 2) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }

{
    /////////////////////////////
    // pressure equation
    /////////////////////////////

    int pr_basis = wkset->usebasis[pr_num];
    auto basis = wkset->basis[pr_basis];
    auto basis_grad = wkset->basis_grad[pr_basis];
    auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
    auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
    auto off = subview(wkset->offsets, pr_num, ALL());

    parallel_for(
        "NS pr volume resid",
        RangePolicy<AssemblyExec>(0, wkset->numElem),
        KOKKOS_LAMBDA(const int elem) {
            for (size_type pt = 0; pt < basis.extent(2); pt++)
            {
                AD divu = (dux_dx(elem, pt) + duy_dy(elem, pt)) * wts(elem, pt);
                for (size_type dof = 0; dof < basis.extent(1); dof++)
                {
                    res(elem, off(dof)) += divu * basis(elem, dof, pt, 0);
                }
            }
        });

    if (usePSPG) // TODO
    {

        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        auto dpr_dz = wkset->getSolutionField("grad(pr)[z]");
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto uz = wkset->getSolutionField("uz");
        auto dux_dt = wkset->getSolutionField("ux_t");
        auto duy_dt = wkset->getSolutionField("uy_t");
        auto duz_dt = wkset->getSolutionField("uz_t");
        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
        auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
        auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
        auto duz_dy = wkset->getSolutionField("grad(uz)[y]");

        parallel_for(
            "NS pr volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD tau = this->computeTau(visc(elem, pt), ux(elem, pt), uy(elem, pt), uz(elem, pt), h(elem), spaceDim, dt, isTransient);
                    AD Sx = dens(elem, pt) * dux_dt(elem, pt) + dens(elem, pt) * (ux(elem, pt) * dux_dx(elem, pt) + uy(elem, pt) * dux_dy(elem, pt) + uz(elem, pt) * dux_dz(elem, pt)) + dpr_dx(elem, pt) - dens(elem, pt) * source_ux(elem, pt);
                    Sx *= tau * wts(elem, pt) / dens(elem, pt);
                    AD Sy = dens(elem, pt) * duy_dt(elem, pt) + dens(elem, pt) * (ux(elem, pt) * duy_dx(elem, pt) + uy(elem, pt) * duy_dy(elem, pt) + uz(elem, pt) * duy_dz(elem, pt)) + dpr_dy(elem, pt) - dens(elem, pt) * source_uy(elem, pt);
                    Sy *= tau * wts(elem, pt) / dens(elem, pt);
                    AD Sz = dens(elem, pt) * duz_dt(elem, pt) + dens(elem, pt) * (ux(elem, pt) * duz_dx(elem, pt) + uy(elem, pt) * duz_dy(elem, pt) + uz(elem, pt) * duz_dz(elem, pt)) + dpr_dz(elem, pt) - dens(elem, pt) * source_uz(elem, pt);
                    Sz *= tau * wts(elem, pt) / dens(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Sx * basis_grad(elem, dof, pt, 0) + Sy * basis_grad(elem, dof, pt, 1) + Sz * basis_grad(elem, dof, pt, 2);
                    }
                }
            });
        if (have_energy)
        {
            // BWR TODO check and change, see above
            auto params = model_params;
            auto E = wkset->getSolutionField("e");
            parallel_for(
                "NS pr volume resid",
                RangePolicy<AssemblyExec>(0, wkset->numElem),
                KOKKOS_LAMBDA(const int elem) {
                    for (size_type pt = 0; pt < basis.extent(2); pt++)
                    {
                        AD tau = this->computeTau(visc(elem, pt), ux(elem, pt), uy(elem, pt), uz(elem, pt), h(elem), spaceDim, dt, isTransient);
                        AD Sx = dens(elem, pt) * params(1) * (E(elem, pt) - params(0)) * source_ux(elem, pt);
                        Sx *= tau * wts(elem, pt);
                        AD Sy = dens(elem, pt) * params(1) * (E(elem, pt) - params(0)) * source_uy(elem, pt);
                        Sy *= tau * wts(elem, pt);
                        AD Sz = dens(elem, pt) * params(1) * (E(elem, pt) - params(0)) * source_uz(elem, pt);
                        Sz *= tau * wts(elem, pt);
                        for (size_type dof = 0; dof < basis.extent(1); dof++)
                        {
                            res(elem, off(dof)) += Sx * basis_grad(elem, dof, pt, 0) + Sy * basis_grad(elem, dof, pt, 1) + Sz * basis_grad(elem, dof, pt, 2);
                        }
                    }
                });
            // stabres += dens(e,k)*(eval-T_ambient)*source_ux(e,k);
        }
    }
}


}

// ========================================================================================
// ========================================================================================

void mhd2d::boundaryResidual()
{

    int spaceDim = wkset->dimension;
    auto bcs = wkset->var_bcs;

    int cside = wkset->currentside;

    string ux_sidetype = bcs(ux_num, cside);
    string uy_sidetype = "Dirichlet";
    string uz_sidetype = "Dirichlet";
    if (spaceDim > 1)
    {
        uy_sidetype = bcs(uy_num, cside);
    }
    if (spaceDim > 2)
    {
        uz_sidetype = bcs(uz_num, cside);
    }

    Vista source_ux, source_uy, source_uz;

    if (ux_sidetype != "Dirichlet" || uy_sidetype != "Dirichlet" || uz_sidetype != "Dirichlet")
    {

        {
            // Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
            if (ux_sidetype == "Neumann")
            {
                source_ux = functionManager->evaluate("Neumann ux " + wkset->sidename, "side ip");
            }
            if (uy_sidetype == "Neumann")
            {
                source_uy = functionManager->evaluate("Neumann uy " + wkset->sidename, "side ip");
            }
            if (uz_sidetype == "Neumann")
            {
                source_uz = functionManager->evaluate("Neumann uz " + wkset->sidename, "side ip");
            }
        }

        // Since normals get recomputed often, this needs to be reset
        auto wts = wkset->wts_side;
        auto h = wkset->h;
        auto res = wkset->res;

        // Teuchos::TimeMonitor localtime(*boundaryResidualFill);

        if (spaceDim == 1)
        {
            int ux_basis = wkset->usebasis[ux_num];
            auto basis = wkset->basis_side[ux_basis];
            auto off = Kokkos::subview(wkset->offsets, ux_num, Kokkos::ALL());
            if (ux_sidetype == "Neumann")
            { // Neumann
                parallel_for(
                    "NS ux bndry resid 1D N",
                    RangePolicy<AssemblyExec>(0, wkset->numElem),
                    KOKKOS_LAMBDA(const int e) {
                        for (size_type k = 0; k < basis.extent(2); k++)
                        {
                            for (size_type i = 0; i < basis.extent(1); i++)
                            {
                                res(e, off(i)) += (-source_ux(e, k) * basis(e, i, k, 0)) * wts(e, k);
                            }
                        }
                    });
            }
        }
        else if (spaceDim == 2)
        {

            // ux equation boundary residual
            {
                int ux_basis = wkset->usebasis[ux_num];
                auto basis = wkset->basis_side[ux_basis];
                auto off = Kokkos::subview(wkset->offsets, ux_num, Kokkos::ALL());

                if (ux_sidetype == "Neumann")
                { // traction (Neumann)
                    parallel_for(
                        "NS ux bndry resid 2D N",
                        RangePolicy<AssemblyExec>(0, wkset->numElem),
                        KOKKOS_LAMBDA(const int e) {
                            for (size_type k = 0; k < basis.extent(2); k++)
                            {
                                for (size_type i = 0; i < basis.extent(1); i++)
                                {
                                    res(e, off(i)) += (-source_ux(e, k) * basis(e, i, k, 0)) * wts(e, k);
                                }
                            }
                        });
                }
            }

            // uy equation boundary residual
            {
                int uy_basis = wkset->usebasis[uy_num];
                auto basis = wkset->basis_side[uy_basis];
                auto off = Kokkos::subview(wkset->offsets, uy_num, Kokkos::ALL());
                if (uy_sidetype == "Neumann")
                { // traction (Neumann)
                    parallel_for(
                        "NS uy bndry resid 2D N",
                        RangePolicy<AssemblyExec>(0, wkset->numElem),
                        KOKKOS_LAMBDA(const int e) {
                            for (size_type k = 0; k < basis.extent(2); k++)
                            {
                                for (size_type i = 0; i < basis.extent(1); i++)
                                {
                                    res(e, off(i)) += (-source_uy(e, k) * basis(e, i, k, 0)) * wts(e, k);
                                }
                            }
                        });
                }
            }
        }

        else if (spaceDim == 3)
        {

            // ux equation boundary residual
            {
                int ux_basis = wkset->usebasis[ux_num];
                auto basis = wkset->basis_side[ux_basis];
                auto off = Kokkos::subview(wkset->offsets, ux_num, Kokkos::ALL());
                if (ux_sidetype == "Neumann")
                { // traction (Neumann)
                    parallel_for(
                        "NS ux bndry resid 3D N",
                        RangePolicy<AssemblyExec>(0, wkset->numElem),
                        KOKKOS_LAMBDA(const int e) {
                            for (size_type k = 0; k < basis.extent(2); k++)
                            {
                                for (size_type i = 0; i < basis.extent(1); i++)
                                {
                                    res(e, off(i)) += (-source_ux(e, k) * basis(e, i, k, 0)) * wts(e, k);
                                }
                            }
                        });
                }
            }

            // uy equation boundary residual
            {
                int uy_basis = wkset->usebasis[uy_num];
                auto basis = wkset->basis_side[uy_basis];
                auto off = Kokkos::subview(wkset->offsets, uy_num, Kokkos::ALL());
                if (uy_sidetype == "Neumann")
                { // traction (Neumann)
                    parallel_for(
                        "NS uy bndry resid 3D N",
                        RangePolicy<AssemblyExec>(0, wkset->numElem),
                        KOKKOS_LAMBDA(const int e) {
                            for (size_type k = 0; k < basis.extent(2); k++)
                            {
                                for (size_type i = 0; i < basis.extent(1); i++)
                                {
                                    res(e, off(i)) += (-source_uy(e, k) * basis(e, i, k, 0)) * wts(e, k);
                                }
                            }
                        });
                }
            }

            // uz equation boundary residual
            {
                int uz_basis = wkset->usebasis[uz_num];
                auto basis = wkset->basis_side[uz_basis];
                auto off = Kokkos::subview(wkset->offsets, uz_num, Kokkos::ALL());
                if (uz_sidetype == "Neumann")
                { // traction (Neumann)
                    parallel_for(
                        "NS uz bndry resid 3D N",
                        RangePolicy<AssemblyExec>(0, wkset->numElem),
                        KOKKOS_LAMBDA(const int e) {
                            for (size_type k = 0; k < basis.extent(2); k++)
                            {
                                for (size_type i = 0; i < basis.extent(1); i++)
                                {
                                    res(e, off(i)) += (-source_uz(e, k) * basis(e, i, k, 0)) * wts(e, k);
                                }
                            }
                        });
                }
            }
        }
    }
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void mhd2d::computeFlux()
{
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

void mhd2d::setWorkset(Teuchos::RCP<workset> &wkset_)
{

    wkset = wkset_;

    vector<string> varlist = wkset->varlist;
    e_num = -1;
    for (size_t i = 0; i < varlist.size(); i++)
    {
        if (varlist[i] == "ux")
            ux_num = i;
        if (varlist[i] == "pr")
            pr_num = i;
        if (varlist[i] == "uy")
            uy_num = i;
        if (varlist[i] == "uz")
            uz_num = i;
        if (varlist[i] == "e")
            e_num = i;
    }
    if (e_num >= 0)
        have_energy = true;
}

// ========================================================================================
// return the value of the stabilization parameter
// ========================================================================================

KOKKOS_FUNCTION AD mhd2d::computeTau(const AD &localdiff, const AD &xvl, const AD &yvl, const AD &zvl, const ScalarT &h, const int &spaceDim, const ScalarT &dt, const bool &isTransient) const
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
}
