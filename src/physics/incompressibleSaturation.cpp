/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "incompressibleSaturation.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
incompressibleSaturation<EvalT>::incompressibleSaturation(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "incompressibleSaturation";

  // save spaceDim here because it is needed (potentially) before workset is finalized

  spaceDim = dimension_;

  // The state is defined by (S,u) where S is the water saturation and u is the velocity
  // The velocity is obtained with the Poisson solve, so the only variable
  // solved for here is S
  // phi is the constant porosity

  myvars.push_back("S");

  mybasistypes.push_back("HGRAD");
  
  // Params from input file

  phi = settings.get<ScalarT>("porosity",.5);

  useWells = settings.get<bool>("use well source",false);
  if (useWells) myWells = wells<EvalT>(settings);

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void incompressibleSaturation<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                            Teuchos::RCP<FunctionManager<EvalT>> & functionManager_) {
  
  functionManager = functionManager_;
  
  // TODO not supported right now?
  // Note if you are using wells, do not name any of them source_S
  functionManager->addFunction("source_S",fs.get<string>("source_S","0.0"),"ip");

  // water fractional flow
  functionManager->addFunction("f_w",fs.get<string>("f_w","1.0"),"ip");

  // Need to talk to the velocity
  functionManager->addFunction("ux",fs.get<string>("ux","0.0"),"ip");
  if (spaceDim>1) functionManager->addFunction("uy",fs.get<string>("uy","0.0"),"ip");
  if (spaceDim>2) functionManager->addFunction("uz",fs.get<string>("uz","0.0"),"ip");

  // Storage for the flux vectors

  fluxes_vol  = View_EvalT4("flux", functionManager->num_elem_,
                         functionManager->num_ip_, 1, spaceDim); // neqn = 1

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void incompressibleSaturation<EvalT>::volumeResidual() {
  
  vector<Vista<EvalT> > sourceterms;

  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    sourceterms.push_back(functionManager->evaluate("source_S","ip"));

    // Update fluxes 
    this->computeFluxVector();

    if (useWells) {
      auto h = wkset->getElementSize();
      sourceterms[0] = myWells.addWellSources(sourceterms[0],h,functionManager, 
                                              wkset->numElem, wkset->numip); 
    }
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;
  auto varlist = wkset->varlist;

  // The flux storage is (numElem,numip,eqn,dimension)
  
  // outer loop over equations
  for (size_t iEqn=0; iEqn<varlist.size(); ++iEqn) {
    int basis_num = wkset->usebasis[iEqn];
    auto basis = wkset->basis[basis_num];
    auto basis_grad = wkset->basis_grad[basis_num];
    auto dSi_dt = wkset->getSolutionField(varlist[iEqn]+"_t");
    auto source_i = sourceterms[iEqn];
    auto off = subview(wkset->offsets,iEqn,ALL());

    if (spaceDim == 1) {

      // All equations are of the form
      // (v_i,phi d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (v_i,source)

      parallel_for("saturation volume resid " + varlist[iEqn] + " 1D",
              RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += phi*dSi_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,iEqn,0)*basis_grad(elem,dof,pt,0) 
                + source_i(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }
    else if (spaceDim == 2) {

      // All equations are of the form
      // (v_i,phi d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (dv_i/dx_2,F_{y,i}) - (v_i,source)

      parallel_for("saturation volume resid " + varlist[iEqn] + " 2D",
              RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += phi*dSi_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,iEqn,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,iEqn,1)*basis_grad(elem,dof,pt,1)
                + source_i(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }
    else if (spaceDim == 3) {

      // All equations are of the form
      // (v_i,phi d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (dv_i/dx_2,F_{y,i}) - (dv_i/dx_3,F_{z,i}) -(v_i,source)

      parallel_for("saturation volume resid " + varlist[iEqn] + " 3D",
              RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += phi*dSi_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,iEqn,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,iEqn,1)*basis_grad(elem,dof,pt,1)
                + fluxes_vol(elem,pt,iEqn,2)*basis_grad(elem,dof,pt,2)
                + source_i(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void incompressibleSaturation<EvalT>::boundaryResidual() {
  
  // Nothing for now...

}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void incompressibleSaturation<EvalT>::computeFlux() {

  // Nothing for now...

}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void incompressibleSaturation<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;

  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); ++i) {
    if (varlist[i] == "S")
      S_num = i;
  }

}

// ========================================================================================
// compute the fluxes
// ========================================================================================

template<class EvalT>
void incompressibleSaturation<EvalT>::computeFluxVector() {

  Teuchos::TimeMonitor localtime(*fluxVectorFill);

  // The flux storage is (numElem,numip,eqn,dimension)
  // The volume fluxes are defined in terms of the state

  auto fluxes = fluxes_vol;
  // these are always needed
  auto ux = functionManager->evaluate("ux","ip");
  auto f_w = functionManager->evaluate("f_w","ip");
  
  if (spaceDim == 1) {

    parallel_for("saturation fluxes 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // S equation -- F_x = f_w ux

        fluxes(elem,pt,S_num,0) = f_w(elem,pt)*ux(elem,pt);

      }
    });
  } 
  else if (spaceDim == 2) {
    // get the second velocity component
    auto uy = functionManager->evaluate("uy","ip");;

    parallel_for("saturation fluxes 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // S equation -- F_x = f_w ux F_y = f_w uy

        fluxes(elem,pt,S_num,0) = f_w(elem,pt)*ux(elem,pt);
        fluxes(elem,pt,S_num,1) = f_w(elem,pt)*uy(elem,pt);

      }
    });
  } 
  else if (spaceDim == 3) {
    // get the second and third velocity component
    auto uy = functionManager->evaluate("uy","ip");;
    auto uz = functionManager->evaluate("uz","ip");;

    parallel_for("saturation fluxes 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // S equation -- F_x = f_w ux F_y = f_w uy f_w F_z = uz

        fluxes(elem,pt,S_num,0) = f_w(elem,pt)*ux(elem,pt);
        fluxes(elem,pt,S_num,1) = f_w(elem,pt)*uy(elem,pt);
        fluxes(elem,pt,S_num,2) = f_w(elem,pt)*uz(elem,pt);

      }
    });
  }
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::incompressibleSaturation<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::incompressibleSaturation<AD>;

// Standard built-in types
template class MrHyDE::incompressibleSaturation<AD2>;
template class MrHyDE::incompressibleSaturation<AD4>;
template class MrHyDE::incompressibleSaturation<AD8>;
template class MrHyDE::incompressibleSaturation<AD16>;
template class MrHyDE::incompressibleSaturation<AD18>;
template class MrHyDE::incompressibleSaturation<AD24>;
template class MrHyDE::incompressibleSaturation<AD32>;
#endif
