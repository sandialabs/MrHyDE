/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "vlasov_fokker_planck_0d2v.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
VFP0d2v<EvalT>::VFP0d2v(Teuchos::ParameterList & settings, const int & dimension_)
: PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "VFP0d2v";
  
  // MrHyDE should provide a 2D mesh corresponding to the 2 velocity dimensions
  
  if (dimension_ != 2) {
    
  }
  
  spaceDim = 0;
  velDim = 2;
  
  // Species: Helium, Carbon, Gold, electrons
  
  myvars.push_back("H");
  myvars.push_back("C");
  myvars.push_back("G");
  myvars.push_back("E"); // solve for electron temperature
  
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  
  m_H = 4.0;
  m_C = 12.0;
  m_G = 197.0;
  m_E = 1.0/1837.0;
  
  Z_H = 2.0;
  Z_C = 6.0;
  Z_G = 30.0;
  Z_E = 1.0;
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                     Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  // TODO not supported right now?
  functionManager->addFunction("gamma_h",fs.get<string>("gamma_h","1.666666666667"),"ip"); // 5/3
  functionManager->addFunction("v_1",fs.get<string>("v_1","x"),"ip");
  functionManager->addFunction("v_2", fs.get<string>("v_2", "y"),"ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::volumeResidual() {
  
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  Vista<EvalT> source, v_1, v_2, Egrad1, Egrad2;
  
  v_1 = functionManager->evaluate("v_1","ip");
  v_2 = functionManager->evaluate("v_2","ip");
  Egrad1 = functionManager->evaluate("Egrad1","ip");
  Egrad2 = functionManager->evaluate("Egrad2","ip");
  
  // VFP for Helium
  {
    int H_basis_num = wkset->usebasis[H_num];
    auto basis = wkset->basis[H_basis_num];
    auto basis_grad = wkset->basis_grad[H_basis_num]; // velocity grad only
    
    Vista<EvalT> source, v_1, v_2, q_H, Egrad1, Egrad2;
    
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    q_H = functionManager->evaluate("q_H","ip");
    
    source = functionManager->evaluate("source_H","ip");
    
    // Contributes:
    // (f(u),v) + (DF(u),nabla v)
    // f(u) = dH/dt - source
    // DF(u) = q_H/m_H*Egrad*grad_v(H)
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    auto H = wkset->getSolutionField("H"); //
    auto dHdt = wkset->getSolutionField("H_t"); //
    
    auto off = subview( wkset->offsets, H_num, ALL());
    
    auto dHdv1 = wkset->getSolutionField("grad(H)[x]");
    auto dHdv2 = wkset->getSolutionField("grad(H)[y]");
    
    size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
    
    
    parallel_for("VFP 0d2v H residual",
                 TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
        for (size_type pt=0; pt<basis.extent(2); ++pt ) {
          res(elem,off(dof)) += (dHdt(elem,pt) - source(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
          res(elem,off(dof)) += q_H(elem,pt)/m_H*Egrad1(elem,pt)*dHdv1(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,0);
          res(elem,off(dof)) += q_H(elem,pt)/m_H*Egrad1(elem,pt)*dHdv2(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,1);
          // Collisonal terms
          //AD col_HC = 0.0;
          
          // C_{H,C}
          //AD gamma_HC = 2*PI*Z_H^2*X_C^2*lambda_HC / m_H^2;
          // C_{H,C}
          
          // C_{H,C}
        }
      }
    });
  }
  
  // VFP for Carbon
  {
    int C_basis_num = wkset->usebasis[C_num];
    auto basis = wkset->basis[C_basis_num];
    auto basis_grad = wkset->basis_grad[C_basis_num]; // velocity grad only
    
    Vista<EvalT> source, v_1, v_2, q_C, Egrad1, Egrad2;
    
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    q_C = functionManager->evaluate("q_C","ip");
    
    source = functionManager->evaluate("source_C","ip");
    
    // Contributes:
    // (f(u),v) + (DF(u),nabla v)
    // f(u) = dH/dt - source
    // DF(u) = q_H/m_H*Egrad*grad_v(H)
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    auto C = wkset->getSolutionField("C"); //
    auto dCdt = wkset->getSolutionField("C_t"); //
    
    auto off = subview( wkset->offsets, C_num, ALL());
    
    auto dCdv1 = wkset->getSolutionField("grad(C)[x]");
    auto dCdv2 = wkset->getSolutionField("grad(C)[y]");
    
    size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
    
    
    parallel_for("Thermal volume resid 3D part 1",
                 TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
        for (size_type pt=0; pt<basis.extent(2); ++pt ) {
          res(elem,off(dof)) += (dCdt(elem,pt) - source(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
          res(elem,off(dof)) += q_C(elem,pt)/m_C*Egrad1(elem,pt)*dCdv1(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,0);
          res(elem,off(dof)) += q_C(elem,pt)/m_C*Egrad1(elem,pt)*dCdv2(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,1);
        }
      }
    });
  }
  
  
  // VFP for Gold
  {
    int G_basis_num = wkset->usebasis[G_num];
    auto basis = wkset->basis[G_basis_num];
    auto basis_grad = wkset->basis_grad[G_basis_num]; // velocity grad only
    
    Vista<EvalT> source, q_G;
    
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    q_G = functionManager->evaluate("q_G","ip");
    
    source = functionManager->evaluate("source_G","ip");
    
    // Contributes:
    // (f(u),v) + (DF(u),nabla v)
    // f(u) = dH/dt - source
    // DF(u) = q_H/m_H*Egrad*grad_v(H)
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    auto G = wkset->getSolutionField("G"); //
    auto dGdt = wkset->getSolutionField("G_t"); //
    
    auto off = subview( wkset->offsets, G_num, ALL());
    
    auto dGdv1 = wkset->getSolutionField("grad(G)[x]");
    auto dGdv2 = wkset->getSolutionField("grad(G)[y]");
    
    size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
    
    
    parallel_for("Thermal volume resid 3D part 1",
                 TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
        for (size_type pt=0; pt<basis.extent(2); ++pt ) {
          res(elem,off(dof)) += (dGdt(elem,pt) - source(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
          res(elem,off(dof)) += q_G(elem,pt)/m_G*Egrad1(elem,pt)*dGdv1(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,0);
          res(elem,off(dof)) += q_G(elem,pt)/m_G*Egrad1(elem,pt)*dGdv2(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,1);
        }
      }
    });
  }
  
  
  // Electron evolution
  {
    int E_basis_num = wkset->usebasis[E_num];
    auto basis = wkset->basis[E_basis_num];
    auto basis_grad = wkset->basis_grad[E_basis_num]; // velocity grad only
    
    Vista<EvalT> source, gamma_h;
    
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    
    source = functionManager->evaluate("source_E","ip");
    
    // Contributes:
    // (f(u),v) + (DF(u),nabla v)
    // f(u) = dH/dt - source
    // DF(u) = q_H/m_H*Egrad*grad_v(H)
    /*
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    auto H = wkset->getSolutionField("H"); //
    auto dHdt = wkset->getSolutionField("H_t"); //
    
    auto off = subview( wkset->offsets, H_num, ALL());
    
    auto dHdv1 = wkset->getSolutionField("grad(H)[x]");
    auto dHdv2 = wkset->getSolutionField("grad(H)[y]");
    
    size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
    
    
    parallel_for("Thermal volume resid 3D part 1",
                 TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
        for (size_type pt=0; pt<basis.extent(2); ++pt ) {
          res(elem,off(dof)) += (dHdt(elem,pt) - source(elem,pt))*wts(elem,pt)*basis(elem,dof,pt,0);
          res(elem,off(dof)) += q_H/m_H*Egrad1(elem,pt)*dHdv1(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,0);
          res(elem,off(dof)) += q_H/m_H*Egrad1(elem,pt)*dHdv2(elem,pt)*wts(elem,pt)*basis_grad(elem,dof,pt,1);
        }
      }
    });
     */
  }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::boundaryResidual() {
  
  auto bcs = wkset->var_bcs;
  
  //int cside = wkset->currentside;
  
  
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void VFP0d2v<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
  
  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "H")
      H_num = i;
    if (varlist[i] == "C")
      C_num = i;
    if (varlist[i] == "G")
      G_num = i;
    if (varlist[i] == "E")
      E_num = i;
  }

}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::VFP0d2v<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::VFP0d2v<AD>;

// Standard built-in types
template class MrHyDE::VFP0d2v<AD2>;
template class MrHyDE::VFP0d2v<AD4>;
template class MrHyDE::VFP0d2v<AD8>;
template class MrHyDE::VFP0d2v<AD16>;
template class MrHyDE::VFP0d2v<AD18>;
template class MrHyDE::VFP0d2v<AD24>;
template class MrHyDE::VFP0d2v<AD32>;
#endif
