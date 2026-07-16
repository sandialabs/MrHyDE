/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "phase_elliptic.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
PhaseElliptic<EvalT>::PhaseElliptic(Teuchos::ParameterList & settings, const int & dimension_)
: PhysicsBase<EvalT>(settings, dimension_) {
  
  label = "phase elliptic";
  
  // save spaceDim here because it is needed (potentially) before workset is finalized
  // TODO is this needed?
  
  spaceDim = dimension_;
  
  // MrHyDE should provide a 2D mesh corresponding to the 2 velocity dimensions
  
  phaseDim = 2; // hard coded for now
  // Species: Helium, Carbon, Gold, electrons
  
  myvars.push_back("T");
  
  mybasistypes.push_back("HGRAD");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void PhaseElliptic<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                           Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"tensor ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void PhaseElliptic<EvalT>::volumeResidual() {
  
  auto wts = wkset->wts;
  auto res = wkset->res;
  auto phase_wts = wkset->phase_wts;
  
  //v_1 = functionManager->evaluate("v_1","ip");
  //v_2 = functionManager->evaluate("v_2","ip");
  //Egrad1 = functionManager->evaluate("Egrad_1","ip");
  //Egrad2 = functionManager->evaluate("Egrad_2","ip");
  
  /*
   auto IQs = wkset->getIntegratedQuantities();
   ScalarT n_H, n_C, n_G, n_E;
   n_H = IQs(0);
   n_C = IQs(1);
   n_G = IQs(2);
   n_E = Z_H*n_H + Z_C*n_C + Z_G*n_G;
   
   ScalarT ux_H, ux_C, ux_G, ux_E;
   ux_H = 1.0/n_H*IQs(3);
   ux_C = 1.0/n_C*IQs(4);
   ux_G = 1.0/n_G*IQs(5);
   ux_E = 1.0/n_E*(Z_H*n_H*ux_H + Z_C*n_C*ux_C + Z_G*n_G*ux_G);
   
   ScalarT uy_H, uy_C, uy_G, uy_E;
   uy_H = 1.0/n_H*IQs(6);
   uy_C = 1.0/n_C*IQs(7);
   uy_G = 1.0/n_G*IQs(8);
   uy_E = 1.0/n_E*(Z_H*n_H*uy_H + Z_C*n_C*uy_C + Z_G*n_G*uy_G);
   
   ScalarT Z_eff, gamma_0, beta_0, alpha_0;
   ScalarT e = 1.602e-19;
   Z_eff = (Z_H*Z_H*n_H + Z_C*Z_C*n_C + Z_G*Z_G*n_G)/(e*n_E);
   
   gamma_0 = (25.0*Z_eff*(433.0*Z_eff + 180.0*std::sqrt(2)))/(4.0*(217.0*Z_eff*Z_eff + 604*std::sqrt(2)*Z_eff+288));
   beta_0 = (30.0*Z_eff*(11.0*Z_eff+15*std::sqrt(2)))/(217.0*Z_eff*Z_eff+604*std::sqrt(2)*Z_eff+288.0);
   alpha_0 = (4.0*(16.0*Z_eff*Z_eff+61.0*std::sqrt(2)*Z_eff+72.0))/(217*Z_eff*Z_eff+604*std::sqrt(2)*Z_eff+288);
   */
  // VFP for Helium
  {
    int T_basis_num = wkset->usebasis[T_num];
    auto basis = wkset->basis[T_basis_num];
    auto basis_grad = wkset->basis_grad[T_basis_num]; // velocity grad only
    
    auto phase_basis = wkset->phase_basis[T_basis_num];
    auto phase_basis_grad = wkset->phase_basis_grad[T_basis_num]; // velocity grad only
    
    Vista<EvalT> source;
    
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    //q_H = functionManager->evaluate("q_H","ip");
    
    source = functionManager->evaluate("source","ip");
    
    // Contributes:
    // (f(T),q) + (\nabla_x T, \nabla_x q) + (\nabla_v T, \nabla_v q)
    // f(T) = dT/dt - source
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    // All of these will produce quantities at the tensor product of spatial and phase integration points
    auto T = wkset->getSolutionField("T"); //
    auto dTdt = wkset->getSolutionField("T_t"); //
    
    auto off = subview( wkset->offsets, T_num, ALL());
    auto poff = subview( wkset->phase_offsets, T_num, ALL());
    
    auto dTdx = wkset->getSolutionField("grad(T)[x]");
    auto dTdy = wkset->getSolutionField("grad(T)[y]");
    
    auto dTdu = wkset->getSolutionField("phasegrad(T)[u]");
    auto dTdv = wkset->getSolutionField("phasegrad(T)[v]");
    
    size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
    
    size_t numPhaseElem = wkset->numPhaseElem;
    size_type numPhaseIP = phase_wts.extent(0)*phase_wts.extent(1);
    size_type numPhaseOff = poff.extent(0);
    parallel_for("T residual",
                 TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
                 MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
        for (size_type pt=0; pt<basis.extent(2); ++pt ) {
          for (size_type pelem=0; pelem<numPhaseElem; ++pelem ) {
            for (size_type pdof=0; pdof<phase_basis.extent(1); ++pdof ) {
              for (size_type ppt=0; ppt<phase_basis.extent(2); ++ppt ) {
                auto offind = off(dof)*(numPhaseOff) + poff(pdof);
                auto ptind = pt*numPhaseIP + ppt;
                res(elem,offind) += (dTdt(elem,ptind) - source(elem,ptind))*wts(elem,pt)*phase_wts(pelem,ppt)*basis(elem,dof,pt,0)*phase_basis(pelem,pdof,ppt,0);
              }
            }
          }
        }
      }
    });
  }
  
}


// ========================================================================================
// ========================================================================================

template<class EvalT>
void PhaseElliptic<EvalT>::boundaryResidual() {
  
  auto bcs = wkset->var_bcs;
  
  //int cside = wkset->currentside;
  
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void PhaseElliptic<EvalT>::computeFlux() {
  
  
}

// ========================================================================================
// return the integrands for the integrated quantities
// ========================================================================================

template<class EvalT>
std::vector< std::vector<string> > PhaseElliptic<EvalT>::setupIntegratedQuantities(const int & spaceDim) {
  
  std::vector< std::vector<string> > integrandsNamesAndTypes;
  
  return integrandsNamesAndTypes;
  
}
// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void PhaseElliptic<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
  
  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "T")
      T_num = i;
    
  }
  
}



//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::PhaseElliptic<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::PhaseElliptic<AD>;

// Standard built-in types
template class MrHyDE::PhaseElliptic<AD2>;
template class MrHyDE::PhaseElliptic<AD4>;
template class MrHyDE::PhaseElliptic<AD8>;
template class MrHyDE::PhaseElliptic<AD16>;
template class MrHyDE::PhaseElliptic<AD18>;
template class MrHyDE::PhaseElliptic<AD24>;
template class MrHyDE::PhaseElliptic<AD32>;
#endif
