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

#include "hartmann.hpp"
using namespace MrHyDE; 

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
hartmann<EvalT>::hartmann(Teuchos::ParameterList &settings, const int &dimension_)
    : PhysicsBase<EvalT>(settings, dimension_)
{

    label = "hartmann";
    myvars.push_back("u");
    myvars.push_back("b");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void hartmann<EvalT>::defineFunctions(Teuchos::ParameterList &fs,
                            Teuchos::RCP<FunctionManager<EvalT> > &functionManager_)
{

    functionManager = functionManager_;

    functionManager->addFunction("source u", fs.get<string>("source u", "-1.0"), "ip");
    functionManager->addFunction("hartmannNum", fs.get<string>("hartmannNum", "1.0"), "ip");
    functionManager->addFunction("resistivity", fs.get<string>("resistivity", "1.0"), "ip");
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void hartmann<EvalT>::volumeResidual()
{

    Vista<EvalT> hartmannNum, source_u;
    {
        Teuchos::TimeMonitor funceval(*volumeResidualFunc);
        source_u = functionManager->evaluate("source u", "ip");
        hartmannNum = functionManager->evaluate("hartmannNum", "ip");
    }

    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    auto wts = wkset->wts;
    auto res = wkset->res;
    { // U equation
        int u_basis = wkset->usebasis[u_num];
        auto basis = wkset->basis[u_basis];
        auto basis_grad = wkset->basis_grad[u_basis];
        auto du_dx = wkset->getSolutionField("grad(u)[x]");
        auto db_dx = wkset->getSolutionField("grad(b)[x]");
        auto off = subview(wkset->offsets, u_num, ALL());

        parallel_for(
            "Hartmann u volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    EvalT Fx = -du_dx(elem, pt);

                    EvalT F = hartmannNum(elem, pt)*db_dx(elem, pt) - source_u(elem, pt);
                    Fx *= wts(elem, pt);
                    F  *= wts(elem, pt);
                    for (size_type dof = 0; dof < basis.extent(1); dof++)
                    {
                        res(elem, off(dof)) += Fx * basis_grad(elem, dof, pt, 0) + F * basis(elem, dof, pt, 0);
                    }
                }
            });
    }{ // B equation
        int b_basis = wkset->usebasis[b_num];
        auto basis = wkset->basis[b_basis];
        auto basis_grad = wkset->basis_grad[b_basis];
        auto du_dx = wkset->getSolutionField("grad(u)[x]");
        auto db_dx = wkset->getSolutionField("grad(b)[x]");
        auto off = subview(wkset->offsets, b_num, ALL());

        parallel_for(
            "Hartmann b volume resid",
            RangePolicy<AssemblyExec>(0, wkset->numElem),
            KOKKOS_LAMBDA(const int elem) {
                for (size_type pt = 0; pt < basis.extent(2); pt++)
                {
                    EvalT Fx = -db_dx(elem, pt);
                    EvalT F = hartmannNum(elem, pt)*du_dx(elem, pt);
                    Fx *= wts(elem, pt);
                    F  *= wts(elem, pt);
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

template<class EvalT>
void hartmann<EvalT>::boundaryResidual()
{
    auto bcs = wkset->var_bcs;
  //int cside = wkset->currentside;

    auto nx = wkset->getScalarField("n[x]");
    // Assume u is dirichlet
    // Assume b is Robin
    {
        Vista<EvalT> nsource, resistivity;
        resistivity = functionManager->evaluate("resistivity", "ip");
        nsource = functionManager->evaluate("Neumann b " + wkset->sidename, "side ip");

        // Since normals get recomputed often, this needs to be reset
        auto wts = wkset->wts_side;
        auto h = wkset->h;
        auto res = wkset->res;

        {
            int b_basis = wkset->usebasis[b_num];
            auto basis = wkset->basis_side[b_basis];
            auto b = wkset->getSolutionField("b");
            auto db_dx = wkset->getSolutionField("grad(b)[x]");
            auto off = Kokkos::subview(wkset->offsets, b_num, Kokkos::ALL());

            // if (bcs(b_num, cside) == "Neumann") 
            {
                parallel_for(
                    "Hartmann b bndry resid Neumann",
                    RangePolicy<AssemblyExec>(0, wkset->numElem),
                    KOKKOS_LAMBDA(const int e) {
                        for (size_type k = 0; k < basis.extent(2); k++)
                        {
                            for (size_type i = 0; i < basis.extent(1); i++)
                            {
                                res(e, off(i)) += nsource(e,k)*wts(e, k)*basis(e,i,k,0);
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

template<class EvalT>
void hartmann<EvalT>::computeFlux()
{
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void hartmann<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > &wkset_)
{

    wkset = wkset_;

    vector<string> varlist = wkset->varlist;
    for (size_t i = 0; i < varlist.size(); i++)
    {
        if (varlist[i] == "u")
            u_num = i;
        if (varlist[i] == "b")
            b_num = i;
    }
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::hartmann<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::hartmann<AD>;

// Standard built-in types
template class MrHyDE::hartmann<AD2>;
template class MrHyDE::hartmann<AD4>;
template class MrHyDE::hartmann<AD8>;
template class MrHyDE::hartmann<AD16>;
//template class MrHyDE::hartmann<AD18>; // AquiEEP_merge
template class MrHyDE::hartmann<AD24>;
template class MrHyDE::hartmann<AD32>;
#endif
