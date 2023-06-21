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

// MHD Physics module. Not working yet.
#define MrHyDE_ENABLE_MHD
#if defined(MrHyDE_ENABLE_MHD)
#include "mhd.hpp"
using namespace MrHyDE;

// ========================================================================================
// Constructor to set up the problem 
// ========================================================================================

mhd::mhd(Teuchos::ParameterList &settings, const int &dimension_)
    : physicsbase(settings, dimension_)
{

  label = "mhd";
  int spaceDim = dimension_;

  myvars.push_back("pr");
  myvars.push_back("ux");
  myvars.push_back("uy");
  myvars.push_back("uz");
  myvars.push_back("Bx");
  myvars.push_back("By");
  myvars.push_back("Bz");

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

void mhd::defineFunctions(Teuchos::ParameterList &fs,
                          Teuchos::RCP<FunctionManager> &functionManager_)
{

  functionManager = functionManager_;

  functionManager->addFunction("source pr", fs.get<string>("source pr", "0.0"), "ip");
  functionManager->addFunction("source ux", fs.get<string>("source ux", "0.0"), "ip");
  functionManager->addFunction("source uy", fs.get<string>("source uy", "0.0"), "ip");
  functionManager->addFunction("source uz", fs.get<string>("source uz", "0.0"), "ip");
  functionManager->addFunction("source Bx", fs.get<string>("source Bx", "0.0"), "ip");
  functionManager->addFunction("source By", fs.get<string>("source By", "0.0"), "ip");
  functionManager->addFunction("source Bz", fs.get<string>("source Bz", "0.0"), "ip");
  functionManager->addFunction("density", fs.get<string>("density", "1.0"), "ip");
  functionManager->addFunction("viscosity", fs.get<string>("viscosity", "1.0"), "ip");
  functionManager->addFunction("mu", fs.get<string>("permeability", "1.2e-7"), "ip");
  functionManager->addFunction("eta", fs.get<string>("resistivity", "0.0"), "ip");
}

// ========================================================================================
// ========================================================================================

void mhd::volumeResidual()
{

  int spaceDim = wkset->dimension;
  ScalarT dt = wkset->deltat;
  bool isTransient = wkset->isTransient;
  Vista dens, visc, mu, eta, source_ux, source_pr, source_uy, source_uz;

  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source_ux = functionManager->evaluate("source ux", "ip");
    source_pr = functionManager->evaluate("source pr", "ip");
    source_uy = functionManager->evaluate("source uy", "ip");
    source_uz = functionManager->evaluate("source uz", "ip");
    dens = functionManager->evaluate("density", "ip");
    visc = functionManager->evaluate("viscosity", "ip");
    mu  = functionManager->evaluate("mu", "ip");
    eta = functionManager->evaluate("eta", "ip");
  }

  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;

  {
    int ux_basis = wkset->usebasis[ux_num];
    auto basis = wkset->basis[ux_basis];
    auto basis_grad = wkset->basis_grad[ux_basis];
    auto ux = wkset->getSolutionField("ux");
    auto uy = wkset->getSolutionField("uy");
    auto uz = wkset->getSolutionField("uz");
    auto Bx = wkset->getSolutionField("Bx");
    auto By = wkset->getSolutionField("By");
    auto Bz = wkset->getSolutionField("Bz");

    auto dux_dt = wkset->getSolutionField("ux_t");

    auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
    auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
    auto dux_dz = wkset->getSolutionField("grad(ux)[z]");

    auto pr = wkset->getSolutionField("pr");
    auto off = subview(wkset->offsets, ux_num, ALL());

    // Ux equation
    parallel_for(
        "NS ux volume resid",
        RangePolicy<AssemblyExec>(0, wkset->numElem),
        KOKKOS_LAMBDA(const int elem) {
          for (size_type pt = 0; pt < basis.extent(2); pt++)
          {
            AD norm_B = (Bx(elem, pt)*Bx(elem, pt) + By(elem, pt)*By(elem, pt) + Bz(elem, pt)*Bz(elem, pt)) / (2*mu(elem, pt));
            AD Fx = visc(elem, pt) * dux_dx(elem, pt) +
                    Bx(elem, pt)*Bx(elem, pt)/mu(elem, pt) - (norm_B + pr(elem, pt));
            Fx *= wts(elem, pt);
            AD Fy = visc(elem, pt)*dux_dy(elem, pt) + Bx(elem, pt)*By(elem, pt)/mu(elem, pt);
            Fy *= wts(elem, pt);
            AD Fz = visc(elem, pt)*dux_dz(elem, pt) + Bx(elem, pt)*Bz(elem, pt)/mu(elem, pt);
            Fz *= wts(elem, pt);

            AD F = dux_dt(elem, pt) +
                   ux(elem, pt) * dux_dx(elem, pt) +
                   uy(elem, pt) * dux_dy(elem, pt) +
                   uz(elem, pt) * dux_dz(elem, pt) - source_ux(elem, pt);
            F *= dens(elem, pt) * wts(elem, pt);
            for (size_type dof = 0; dof < basis.extent(1); dof++)
            {
              res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) +
                                     Fy * basis_grad(elem, dof, pt, 1) +
                                     Fz * basis_grad(elem, dof, pt, 2) +
                                     F * basis(elem, dof, pt, 0);
            }
          }
        });

    // SUPG contribution

  /*   if (useSUPG)
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
    }
   */
  }

  {
    // Uy equation
    int uy_basis = wkset->usebasis[uy_num];
    auto basis = wkset->basis[uy_basis];
    auto basis_grad = wkset->basis_grad[uy_basis];
    auto ux = wkset->getSolutionField("ux");
    auto uy = wkset->getSolutionField("uy");
    auto uz = wkset->getSolutionField("uz");
    auto Bx = wkset->getSolutionField("Bx");
    auto By = wkset->getSolutionField("By");
    auto Bz = wkset->getSolutionField("Bz");

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
            AD norm_B = (Bx(elem, pt)*Bx(elem, pt) + By(elem, pt)*By(elem, pt) + Bz(elem, pt)*Bz(elem, pt)) / (2*mu(elem,pt));
            AD Fx = visc(elem, pt)*duy_dy(elem, pt) + By(elem, pt)*Bx(elem, pt)/mu(elem, pt);
            Fx *= wts(elem, pt);
            AD Fy = visc(elem, pt)*duy_dy(elem, pt) + By(elem, pt)*By(elem, pt)/mu(elem, pt) - (norm_B + pr(elem, pt));
            Fy *= wts(elem, pt);
            AD Fz = visc(elem, pt)*duy_dz(elem, pt) + By(elem, pt)*Bz(elem, pt)/mu(elem, pt);
            Fz *= wts(elem, pt);
            AD F = duy_dt(elem, pt) +
                   ux(elem, pt) * duy_dx(elem, pt) +
                   uy(elem, pt) * duy_dy(elem, pt) +
                   uz(elem, pt) * duy_dz(elem, pt) - source_uy(elem, pt);
            F *= dens(elem, pt) * wts(elem, pt);
            for (size_type dof = 0; dof < basis.extent(1); dof++)
            {
              res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + Fy * basis_grad(elem, dof, pt, 1) + Fz * basis_grad(elem, dof, pt, 2) + F * basis(elem, dof, pt, 0);
            }
          }
        });

    // SUPG contribution
  /*   if (useSUPG)
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
    }
   */
  }

  {
    // Uz equation
    int uz_basis = wkset->usebasis[uz_num];
    auto basis = wkset->basis[uz_basis];
    auto basis_grad = wkset->basis_grad[uz_basis];
    auto ux = wkset->getSolutionField("ux");
    auto uy = wkset->getSolutionField("uy");
    auto uz = wkset->getSolutionField("uz");

    auto Bx = wkset->getSolutionField("Bx");
    auto By = wkset->getSolutionField("By");
    auto Bz = wkset->getSolutionField("Bz");

    auto duz_dt = wkset->getSolutionField("uz_t");
    auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
    auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
    auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
    auto pr = wkset->getSolutionField("pr");
    auto off = subview(wkset->offsets, uy_num, ALL());

    parallel_for(
        "MHD uz volume resid",
        RangePolicy<AssemblyExec>(0, wkset->numElem),
        KOKKOS_LAMBDA(const int elem) {
          for (size_type pt = 0; pt < basis.extent(2); pt++)
          {
            AD norm_B = (Bx(elem, pt) * Bx(elem, pt) + By(elem, pt) * By(elem, pt) + Bz(elem, pt) * Bz(elem, pt)) / (2*mu(elem,pt));
            AD Fx = visc(elem, pt) * duz_dx(elem, pt) + Bz(elem, pt) * Bx(elem, pt) / mu(elem, pt);
            Fx *= wts(elem, pt);
            AD Fy = visc(elem, pt) * duz_dy(elem, pt) + Bz(elem, pt) * By(elem, pt) / mu(elem, pt);
            Fy *= wts(elem, pt);
            AD Fz = visc(elem, pt) * duz_dz(elem, pt) + Bz(elem, pt) * Bz(elem, pt) / mu(elem, pt) - (norm_B + pr(elem, pt));
            Fz *= wts(elem, pt);
            AD F = duz_dt(elem, pt) +
                   ux(elem, pt) * duz_dx(elem, pt) +
                   uy(elem, pt) * duz_dy(elem, pt) +
                   uz(elem, pt) * duz_dz(elem, pt) - source_uz(elem, pt);
            F *= dens(elem, pt) * wts(elem, pt);
            for (size_type dof = 0; dof < basis.extent(1); dof++)
            {
              res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) +
                                     Fy * basis_grad(elem, dof, pt, 1) +
                                     Fz * basis_grad(elem, dof, pt, 2) + F * basis(elem, dof, pt, 0);
            }
          }
        });

    // SUPG contribution
    /* if (useSUPG)
    {
      auto h = wkset->h;
      auto dpr_dz = wkset->getSolutionField("grad(pr)[z]");
      parallel_for(
          "NS uz volume resid",
          RangePolicy<AssemblyExec>(0, wkset->numElem),
          KOKKOS_LAMBDA(const int elem) {
            for (size_type pt = 0; pt < basis.extent(2); pt++)
            {
              AD tau = this->computeTau(visc(elem, pt), ux(elem, pt), uy(elem, pt), uz(elem, pt), h(elem), spaceDim, dt, isTransient);
              AD stabres = dens(elem, pt) * duz_dt(elem, pt) + dens(elem, pt) * (ux(elem, pt) * duz_dx(elem, pt) + uy(elem, pt) * duz_dy(elem, pt) + uz(elem, pt) * duz_dz(elem, pt)) + dpr_dz(elem, pt) - dens(elem, pt) * source_uz(elem, pt);
              AD Sx = tau * stabres * ux(elem, pt) * wts(elem, pt);
              AD Sy = tau * stabres * uy(elem, pt) * wts(elem, pt);
              AD Sz = tau * stabres * uz(elem, pt) * wts(elem, pt);
              for (size_type dof = 0; dof < basis.extent(1); dof++)
              {
                res(elem, off(dof)) += Sx * basis_grad(elem, dof, pt, 0) + Sy * basis_grad(elem, dof, pt, 1) + Sz * basis_grad(elem, dof, pt, 2);
              }
            }
          });
    } */
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
    auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
    auto off = subview(wkset->offsets, pr_num, ALL());
    bool nonlinear_solenoidal = true;
    if(nonlinear_solenoidal) {
      auto dBx_dx = wkset->getSolutionField("grad(Bx)[x]");
      auto dBy_dy = wkset->getSolutionField("grad(By)[y]");
      auto dBz_dz = wkset->getSolutionField("grad(Bz)[z]");
      parallel_for(
        "MHD pr volume resid",
        RangePolicy<AssemblyExec>(0, wkset->numElem),
        KOKKOS_LAMBDA(const int elem) {
          for (size_type pt = 0; pt < basis.extent(2); pt++)
          {
            AD divu = (dux_dx(elem, pt) + duy_dy(elem, pt) + duz_dz(elem, pt)) * wts(elem, pt);
            AD divB = (dBx_dx(elem, pt) + dBy_dy(elem, pt) + dBz_dz(elem, pt)) * wts(elem, pt);
            for (size_type dof = 0; dof < basis.extent(1); dof++)
            {
              res(elem, off(dof)) += (divu*divu + divB*divB)*basis(elem, dof, pt, 0);
            }
          }
        });
    } else {
      parallel_for(
        "MHD pr volume resid",
        RangePolicy<AssemblyExec>(0, wkset->numElem),
        KOKKOS_LAMBDA(const int elem) {
          for (size_type pt = 0; pt < basis.extent(2); pt++)
          {
            AD divu = (dux_dx(elem, pt) + duy_dy(elem, pt) + duz_dz(elem, pt)) * wts(elem, pt);
            for (size_type dof = 0; dof < basis.extent(1); dof++)
            {
              res(elem, off(dof)) += divu * basis(elem, dof, pt, 0);
            }
          }
        });
    }
    /* if (usePSPG)
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
    } */
  }
  {
    ///////////////////////
    // Induction Eqn (x) //
    ///////////////////////
    int Bx_basis = wkset->usebasis[Bx_num];
    auto basis = wkset->basis[Bx_basis];
    auto basis_grad = wkset->basis_grad[Bx_basis];
    auto Bx = wkset->getSolutionField("Bx");
    auto By = wkset->getSolutionField("By");
    auto Bz = wkset->getSolutionField("Bz");
    auto dBx_dt = wkset->getSolutionField("Bx_t");
    auto dBx_dy = wkset->getSolutionField("grad(Bx)[y]");
    auto dBx_dz = wkset->getSolutionField("grad(Bx)[z]");
    auto ux = wkset->getSolutionField("ux");
    auto uy = wkset->getSolutionField("uy");
    auto uz = wkset->getSolutionField("uz");
    auto off = subview(wkset->offsets, pr_num, ALL());

    parallel_for(
        "MHD Bx volume resid",
        RangePolicy<AssemblyExec>(0, wkset->numElem),
        KOKKOS_LAMBDA(const int elem) {
          for (size_type pt = 0; pt < basis.extent(2); pt++)
          {
            AD Fx = 0.;
            Fx *= wts(elem, pt);
            
            AD Fy = By(elem,pt)*ux(elem,pt) - uy(elem,pt)*Bx(elem,pt) -
              eta(elem,pt)*dBx_dy(elem,pt)/mu(elem,pt);
            Fy *= wts(elem, pt);

            AD Fz = Bz(elem,pt)*ux(elem,pt) - uz(elem,pt)*Bx(elem,pt) -
              eta(elem,pt)*dBx_dz(elem,pt)/mu(elem,pt);
            Fz *= wts(elem, pt);

            AD F = dBx_dt(elem, pt);
            F *= wts(elem, pt);
            for (size_type dof = 0; dof < basis.extent(1); dof++)
            {
              res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) +
                                     Fy * basis_grad(elem, dof, pt, 1) +
                                     Fz * basis_grad(elem, dof, pt, 2) + F * basis(elem, dof, pt, 0);
            }
          }
        });
  }
  {
    ///////////////////////
    // Induction Eqn (y) //
    ///////////////////////
    int By_basis = wkset->usebasis[By_num];
    auto basis = wkset->basis[By_basis];
    auto basis_grad = wkset->basis_grad[By_basis];
    auto Bx = wkset->getSolutionField("Bx");
    auto By = wkset->getSolutionField("By");
    auto Bz = wkset->getSolutionField("Bz");
    auto dBy_dt = wkset->getSolutionField("By_t");
    auto dBy_dx = wkset->getSolutionField("grad(By)[x]");
    auto dBy_dz = wkset->getSolutionField("grad(By)[z]");
    auto ux = wkset->getSolutionField("ux");
    auto uy = wkset->getSolutionField("uy");
    auto uz = wkset->getSolutionField("uz");
    auto off = subview(wkset->offsets, pr_num, ALL());

    parallel_for(
        "MHD By volume resid",
        RangePolicy<AssemblyExec>(0, wkset->numElem),
        KOKKOS_LAMBDA(const int elem) {
          for (size_type pt = 0; pt < basis.extent(2); pt++)
          {
            AD Fx = Bx(elem,pt)*uy(elem,pt) - ux(elem,pt)*By(elem,pt) - eta(elem,pt)*dBy_dx(elem,pt)/mu(elem,pt);
            Fx *= wts(elem, pt);

            AD Fy = 0.;
            Fy *= wts(elem, pt);

            AD Fz = Bz(elem,pt)*uy(elem,pt) - uz(elem,pt)*By(elem,pt) - eta(elem,pt)*dBy_dz(elem,pt)/mu(elem,pt);
            Fz *= wts(elem, pt);

            AD F = dBy_dt(elem, pt);
            F *= wts(elem, pt);
            for (size_type dof = 0; dof < basis.extent(1); dof++)
            {
              res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) +
                                     Fy * basis_grad(elem, dof, pt, 1) +
                                     Fz * basis_grad(elem, dof, pt, 2) + F * basis(elem, dof, pt, 0);
            }
          }
        });
  }
  {
    ///////////////////////
    // Induction Eqn (z) //
    ///////////////////////
    int Bz_basis = wkset->usebasis[Bz_num];
    auto basis = wkset->basis[Bz_basis];
    auto basis_grad = wkset->basis_grad[Bz_basis];
    auto Bx = wkset->getSolutionField("Bx");
    auto By = wkset->getSolutionField("By");
    auto Bz = wkset->getSolutionField("Bz");
    auto dBz_dt = wkset->getSolutionField("Bz_t");
    auto dBz_dx = wkset->getSolutionField("grad(Bz)[x]");
    auto dBz_dy = wkset->getSolutionField("grad(Bz)[y]");
    auto ux = wkset->getSolutionField("ux");
    auto uy = wkset->getSolutionField("uy");
    auto uz = wkset->getSolutionField("uz");
    auto off = subview(wkset->offsets, pr_num, ALL());

    parallel_for(
        "MHD Bz volume resid",
        RangePolicy<AssemblyExec>(0, wkset->numElem),
        KOKKOS_LAMBDA(const int elem) {
          for (size_type pt = 0; pt < basis.extent(2); pt++)
          {
            AD Fx = Bx(elem,pt)*uz(elem,pt) - ux(elem,pt)*Bz(elem,pt) - eta(elem,pt)*dBz_dx(elem,pt)/mu(elem,pt);
            Fx *= wts(elem, pt);

            AD Fy = By(elem,pt)*uz(elem,pt) - uy(elem,pt)*Bz(elem,pt) - eta(elem,pt)*dBz_dy(elem,pt)/mu(elem,pt);
            Fy *= wts(elem, pt);

            AD Fz = 0.;
            Fz *= wts(elem, pt);
            
            AD F = dBz_dt(elem, pt);
            F *= wts(elem, pt);
            for (size_type dof = 0; dof < basis.extent(1); dof++)
            {
              res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) +
                                     Fy * basis_grad(elem, dof, pt, 1) +
                                     Fz * basis_grad(elem, dof, pt, 2) + F * basis(elem, dof, pt, 0);
            }
          }
        });
  }
}

// ========================================================================================
// ========================================================================================

void mhd::boundaryResidual()
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
// void mhd::computeFlux()
// {
// }

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

void mhd::setWorkset(Teuchos::RCP<workset> &wkset_)
{

  wkset = wkset_;

  vector<string> varlist = wkset->varlist;
  for (size_t i = 0; i < varlist.size(); i++)
  {
    if (varlist[i] == "ux")
      ux_num = i;
    if (varlist[i] == "uy")
      uy_num = i;
    if (varlist[i] == "uz")
      uz_num = i;
    if (varlist[i] == "pr")
      pr_num = i;
    if (varlist[i] == "Bx")
      Bx_num = i;
    if (varlist[i] == "By")
      By_num = i;
    if (varlist[i] == "Bz")
      Bz_num = i;
  }
}


// ========================================================================================
// return the value of the stabilization parameter
// ========================================================================================

/* KOKKOS_FUNCTION AD mhd::computeTau(const AD &localdiff, const AD &xvl, const AD &yvl, const AD &zvl, const ScalarT &h, const int &spaceDim, const ScalarT &dt, const bool &isTransient) const
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