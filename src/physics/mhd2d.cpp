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
    myvars.push_back("pr");
    myvars.push_back("ux");
    myvars.push_back("uy");
    myvars.push_back("T");
    myvars.push_back("Bx");
    myvars.push_back("By");
    myvars.push_back("Az");

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
    functionManager->addFunction("source Az", fs.get<string>("source Az", "0.0"), "ip");
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
    Vista dens, visc, source_ux, source_uy, source_E, Cp, chi, eta, mu;

    {
        Teuchos::TimeMonitor funceval(*volumeResidualFunc);
        source_ux = functionManager->evaluate("source ux", "ip");
        source_uy = functionManager->evaluate("source uy", "ip");
        source_E = functionManager->evaluate("source Az", "ip");

        dens = functionManager->evaluate("density", "ip");
        visc = functionManager->evaluate("viscosity", "ip");
        Cp = functionManager->evaluate("specific heat", "ip");
        chi = functionManager->evaluate("thermal conductivity", "ip");
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
        auto dux_dt = wkset->getSolutionField("ux_t");
        auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
        auto Bx = wkset->getSolutionField("Bx");
        auto By = wkset->getSolutionField("By");
        auto pr = wkset->getSolutionField("pr");
        auto off = subview(wkset->offsets, ux_num, ALL());

        parallel_for(
            "MHD ux volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD div_u = dux_dx(elem, pt) + duy_dy(elem, pt);
                    AD norm_B = Bx(elem, pt)*Bx(elem, pt) + By(elem, pt)*By(elem, pt);
                    AD diag_ctrb = pr(elem, pt)+2*visc(elem, pt)*div_u/3 + norm_B/(2*mu(elem, pt));
                    AD Fx = -Bx(elem, pt)*Bx(elem, pt)/mu(elem, pt) - visc(elem, pt)*2*dux_dx(elem, pt) + diag_ctrb;
                    AD Fy = -Bx(elem, pt)*By(elem, pt)/mu(elem, pt) - visc(elem, pt)*(dux_dy(elem, pt) + duy_dx(elem, pt));
                    AD F =-source_ux(elem, pt) + dens(elem, pt)*(
                           dux_dt(elem, pt) +
                           ux(elem, pt) * dux_dx(elem, pt) +
                           uy(elem, pt) * dux_dy(elem, pt) );
                    Fx *= -wts(elem, pt);
                    Fy *= -wts(elem, pt);
                    F *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + Fy * basis_grad(elem, dof, pt, 1) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
        // SUPG contribution
        if (useSUPG)
        { /* TODO */ }
    }
    {// Uy equation
        int uy_basis = wkset->usebasis[uy_num];
        auto basis = wkset->basis[uy_basis];
        auto basis_grad = wkset->basis_grad[uy_basis];
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto duy_dt = wkset->getSolutionField("uy_t");
        auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
        auto Bx = wkset->getSolutionField("Bx");
        auto By = wkset->getSolutionField("By");
        auto pr = wkset->getSolutionField("pr");
        auto off = subview(wkset->offsets, uy_num, ALL());

        parallel_for(
            "MHD uy volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD div_u = dux_dx(elem, pt) + duy_dy(elem, pt);
                    AD norm_B = Bx(elem, pt)*Bx(elem, pt) + By(elem, pt)*By(elem, pt);
                    AD diag_contr = pr(elem, pt)-2*visc(elem, pt)*div_u/3 + norm_B/(2*mu(elem, pt));
                    AD Fx = -By(elem, pt)*Bx(elem, pt)/mu(elem, pt) - visc(elem, pt)*(duy_dx(elem, pt) + dux_dy(elem, pt)) ;
                    AD Fy = -By(elem, pt)*By(elem, pt)/mu(elem, pt) - visc(elem, pt)*2*duy_dy(elem, pt) + diag_contr;
                    AD F = -source_uy(elem, pt) + dens(elem, pt)*(duy_dt(elem, pt) +
                        ux(elem, pt) * duy_dx(elem, pt) +
                        uy(elem, pt) * duy_dy(elem, pt) );
                    Fx *= -wts(elem, pt);
                    Fy *= -wts(elem, pt);
                    F *=   wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + Fy * basis_grad(elem, dof, pt, 1) + F * basis(elem, dof, pt, 0);
                    }
                }
            });

        // SUPG contribution

        if (useSUPG)
        {
            // TODO
        }
    }
    {// pr equation
        int pr_basis = wkset->usebasis[pr_num];
        auto basis = wkset->basis[pr_basis];
        auto basis_grad = wkset->basis_grad[pr_basis];
        auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
        auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
        auto off = subview(wkset->offsets, pr_num, ALL());

        parallel_for(
            "MHD pr volume resid",
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

        if (usePSPG)
        {
            // TODO
        }
    }
    { // T equation
        int T_basis = wkset->usebasis[T_num];
        auto basis = wkset->basis[T_basis];
        auto basis_grad = wkset->basis_grad[T_basis];
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto duy_dt = wkset->getSolutionField("uy_t");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
        auto dBy_dx = wkset->getSolutionField("grad(By)[x]");
        auto dBx_dy = wkset->getSolutionField("grad(Bx)[y]");
        auto pr = wkset->getSolutionField("pr");
        auto dT_dt = wkset->getSolutionField("T_t");
        auto dT_dx = wkset->getSolutionField("grad(T)[x]");
        auto dT_dy = wkset->getSolutionField("grad(T)[y]");
        auto off = subview(wkset->offsets, T_num, ALL());

        parallel_for(
            "MHD T volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Jz = (dBy_dx(elem, pt) - dBx_dy(elem, pt))/mu(elem, pt);
                    AD heat_in = dT_dt(elem, pt) + ux(elem, pt)*dT_dx(elem, pt) + uy(elem, pt)*dT_dy(elem, pt);
                    heat_in *= dens(elem, pt)*Cp(elem, pt);
                    AD qx = -chi(elem, pt)*dT_dx(elem, pt);
                    AD qy = -chi(elem, pt)*dT_dy(elem, pt);
                    AD F = heat_in - eta(elem, pt)*Jz*Jz;
                    qx *= -wts(elem, pt);
                    qy *= -wts(elem, pt);
                    F  *=  wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += qx * basis_grad(elem, dof, pt, 0) + qy * basis_grad(elem, dof, pt, 1) + F * basis(elem, dof, pt, 0);
                    }
                }
            });

    }
    { // Bx Eqn
        int Bx_basis = wkset->usebasis[Bx_num];
        auto basis = wkset->basis[Bx_basis];
        auto basis_grad = wkset->basis_grad[Bx_basis];
        auto dAz_dy = wkset->getSolutionField("grad(Az)[y]");
        auto Bx = wkset->getSolutionField("Bx");
        auto off = subview(wkset->offsets, Bx_num, ALL());

        parallel_for(
            "MHD Bx volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD curlAz_x = (Bx(elem, pt) - dAz_dy(elem, pt))*wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += curlAz_x * basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // By Eqn
        int By_basis = wkset->usebasis[By_num];
        auto basis = wkset->basis[By_basis];
        auto basis_grad = wkset->basis_grad[By_basis];
        auto dAz_dx = wkset->getSolutionField("grad(Az)[x]");
        auto By = wkset->getSolutionField("By");
        auto off = subview(wkset->offsets, By_num, ALL());

        parallel_for(
            "MHD By volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD curlAz_y = (By(elem, pt) + dAz_dx(elem, pt))*wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += curlAz_y*basis(elem, dof, pt, 0);
                    }
                }
            });
    }
    { // Az Eqn
        int Az_basis = wkset->usebasis[Az_num];
        auto basis = wkset->basis[Az_basis];
        auto basis_grad = wkset->basis_grad[Az_basis];
        auto ux = wkset->getSolutionField("ux");
        auto uy = wkset->getSolutionField("uy");
        auto dAz_dt = wkset->getSolutionField("Az_t");
        auto dAz_dx = wkset->getSolutionField("grad(Az)[x]");
        auto dAz_dy = wkset->getSolutionField("grad(Az)[y]");
        auto off = subview(wkset->offsets, Az_num, ALL());

        parallel_for(
            "MHD Az volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    AD Fx = -eta(elem, pt)*dAz_dx(elem, pt)/mu(elem, pt);
                    AD Fy = -eta(elem, pt)*dAz_dy(elem, pt)/mu(elem, pt);
                    AD F  = dAz_dt(elem, pt) + ux(elem, pt)*dAz_dx(elem, pt) + uy(elem, pt)*dAz_dy(elem, pt) + source_E(elem, pt);
                    Fx *= -wts(elem, pt);
                    Fy *= -wts(elem, pt);
                    F  *=  wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += F*basis(elem, dof, pt, 0) + Fx*basis_grad(elem, dof, pt, 0) + Fy*basis_grad(elem, dof, pt, 1);
                    }
                }
            });
    }

}

// ========================================================================================
// ========================================================================================

void mhd2d::boundaryResidual()
{
    auto bcs = wkset->var_bcs;

    int cside = wkset->currentside;

    string ux_sidetype = bcs(ux_num, cside);
    string uy_sidetype = "Dirichlet";
    uy_sidetype = bcs(uy_num, cside);
    string T_sidetype = bcs(T_num, cside);
    Vista source_ux, source_uy, chi;

    auto nx = wkset->getScalarField("n[x]");
    auto ny = wkset->getScalarField("n[y]");

    if (ux_sidetype != "Dirichlet" || uy_sidetype != "Dirichlet")
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
        }

        // Since normals get recomputed often, this needs to be reset
        auto wts = wkset->wts_side;
        auto h = wkset->h;
        auto res = wkset->res;

        // ux equation boundary residual
        { // TODO
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
    if (T_sidetype != "Dirichlet")
    {

        // Since normals get recomputed often, this needs to be reset
        auto wts = wkset->wts_side;
        auto h = wkset->h;
        auto res = wkset->res;
        chi = functionManager->evaluate("thermal conductivity", "ip");

        // T equation boundary residual
        { // TODO
            int T_basis = wkset->usebasis[T_num];
            auto basis = wkset->basis_side[T_basis];
            auto dT_dx = wkset->getSolutionField("grad(T)[x]");
            auto dT_dy = wkset->getSolutionField("grad(T)[y]");
            auto off = Kokkos::subview(wkset->offsets, T_num, Kokkos::ALL());

            if (T_sidetype == "Neumann")
            { // traction (Neumann)
                parallel_for(
                    "MHD2D T bndry resid N",
                    RangePolicy<AssemblyExec>(0, wkset->numElem),
                    KOKKOS_LAMBDA(const int e) {
                        for (size_type k = 0; k < basis.extent(2); k++)
                        {
                            for (size_type i = 0; i < basis.extent(1); i++)
                            {
                                res(e, off(i)) += -chi(e, k)*(dT_dx(e, k)*nx(e, k) + dT_dy(e, k)*ny(e, k)) * basis(e, i, k, 0) * wts(e, k);
                            }
                        }
                    });
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
    for (size_t i = 0; i < varlist.size(); i++)
    {
        if (varlist[i] == "pr")
            pr_num = i;
        if (varlist[i] == "ux")
            ux_num = i;
        if (varlist[i] == "uy")
            uy_num = i;
        if (varlist[i] == "T")
            T_num = i;
        if (varlist[i] == "Bx")
            Bx_num = i;
        if (varlist[i] == "By")
            By_num = i;
        if (varlist[i] == "Az")
            Az_num = i;
    }
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
