/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "shallowwater.hpp"

using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
shallowwater<EvalT>::shallowwater(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "shallowwater";
  
  myvars.push_back("H");
  myvars.push_back("Hu");
  myvars.push_back("Hv");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  
  
  gravity = settings.get<ScalarT>("gravity",9.8);
  
  formparam = settings.get<ScalarT>("form_param",1.0);
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowwater<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                   Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;

  functionManager->addFunction("bathymetry",fs.get<string>("bathymetry","1.0"),"ip");
  functionManager->addFunction("bathymetry_x",fs.get<string>("bathymetry_x","0.0"),"ip");
  functionManager->addFunction("bathymetry_y",fs.get<string>("bathymetry_y","0.0"),"ip");
  functionManager->addFunction("bottom friction",fs.get<string>("bottom friction","1.0"),"ip");
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","0.0"),"ip");
  functionManager->addFunction("Coriolis",fs.get<string>("Coriolis","0.0"),"ip");
  functionManager->addFunction("source H",fs.get<string>("source H","0.0"),"ip");
  functionManager->addFunction("source Hu",fs.get<string>("source Hu","0.0"),"ip");
  functionManager->addFunction("source Hv",fs.get<string>("source Hv","0.0"),"ip");
  functionManager->addFunction("flux left",fs.get<string>("flux left","0.0"),"side ip");
  functionManager->addFunction("flux right",fs.get<string>("flux right","0.0"),"side ip");
  functionManager->addFunction("flux top",fs.get<string>("flux top","0.0"),"side ip");
  functionManager->addFunction("flux bottom",fs.get<string>("flux bottom","0.0"),"side ip");
  functionManager->addFunction("Neumann source Hu",fs.get<string>("Neumann source Hu","0.0"),"side ip");
  functionManager->addFunction("Neumann source Hv",fs.get<string>("Neumann source Hv","0.0"),"side ip");
  functionManager->addFunction("bathymetry side",fs.get<string>("bathymetry","1.0"),"side_ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowwater<EvalT>::volumeResidual() {
  
  Vista<EvalT> bath, bath_x, bath_y, visc, cor, bfric, source_H, source_Hu, source_Hv;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    bath = functionManager->evaluate("bathymetry","ip");
    bath_x = functionManager->evaluate("bathymetry_x","ip");
    bath_y = functionManager->evaluate("bathymetry_y","ip");
    visc = functionManager->evaluate("viscosity","ip");
    cor = functionManager->evaluate("Coriolis","ip");
    source_H = functionManager->evaluate("source H","ip");
    source_Hu = functionManager->evaluate("source Hu","ip");
    source_Hv = functionManager->evaluate("source Hv","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  
  int H_basis_num = wkset->usebasis[H_num];
  auto Hbasis = wkset->basis[H_basis_num];
  auto Hbasis_grad = wkset->basis_grad[H_basis_num];
  
  int Hu_basis_num = wkset->usebasis[Hu_num];
  auto Hubasis = wkset->basis[Hu_basis_num];
  auto Hubasis_grad = wkset->basis_grad[Hu_basis_num];
  
  int Hv_basis_num = wkset->usebasis[Hv_num];
  auto Hvbasis = wkset->basis[Hv_basis_num];
  auto Hvbasis_grad = wkset->basis_grad[Hv_basis_num];
  
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  //KokkosTools::print(bath);
  
  auto xi = wkset->getSolutionField("H");
  auto xi_dot = wkset->getSolutionField("H_t");
  
  auto Hu = wkset->getSolutionField("Hu");
  auto Hu_dot = wkset->getSolutionField("Hu_t");
  
  auto Hv = wkset->getSolutionField("Hv");
  auto Hv_dot = wkset->getSolutionField("Hv_t");
  
  auto Hoff  = subview(wkset->offsets, H_num,  ALL());
  auto Huoff = subview(wkset->offsets, Hu_num, ALL());
  auto Hvoff = subview(wkset->offsets, Hv_num, ALL());
  
  parallel_for("SW volume resid",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               MRHYDE_LAMBDA (const int elem ) {
    //ScalarT gravity = 1.0;//9.8;
    for (size_type pt=0; pt<Hbasis.extent(2); pt++ ) {
      
      EvalT f = (xi_dot(elem,pt) - source_H(elem,pt))*wts(elem,pt);
      EvalT Fx = -Hu(elem,pt)*wts(elem,pt);
      EvalT Fy = -Hv(elem,pt)*wts(elem,pt);
      for (size_type dof=0; dof<Hbasis.extent(1); dof++ ) {
        res(elem,Hoff(dof)) += f*Hbasis(elem,dof,pt,0) + Fx*Hbasis_grad(elem,dof,pt,0) + Fy*Hbasis_grad(elem,dof,pt,1);
      }
      
      EvalT H = xi(elem,pt) + bath(elem,pt);
      EvalT uHu = Hu(elem,pt)*Hu(elem,pt)/H;
      EvalT uHv = Hu(elem,pt)*Hv(elem,pt)/H;
      EvalT vHv = Hv(elem,pt)*Hv(elem,pt)/H;
      
      f = (Hu_dot(elem,pt) - gravity*xi(elem,pt)*bath_x(elem,pt) - source_Hu(elem,pt))*wts(elem,pt);
      Fx = -(uHu + 0.5*gravity*(H*H-bath(elem,pt)*bath(elem,pt)))*wts(elem,pt);
      Fy = -uHv*wts(elem,pt);
      for (size_type dof=0; dof<Hubasis.extent(1); dof++ ) {
        res(elem,Huoff(dof)) += f*Hubasis(elem,dof,pt,0) + Fx*Hubasis_grad(elem,dof,pt,0) + Fy*Hubasis_grad(elem,dof,pt,1);
      }
      
      f = (Hv_dot(elem,pt) - gravity*xi(elem,pt)*bath_y(elem,pt) - source_Hv(elem,pt))*wts(elem,pt);
      Fx = -uHv*wts(elem,pt);
      Fy = -(vHv + 0.5*gravity*(H*H-bath(elem,pt)*bath(elem,pt)))*wts(elem,pt);
      
      for (size_type dof=0; dof<Hvbasis.extent(1); dof++ ) {
        res(elem,Hvoff(dof)) += f*Hvbasis(elem,dof,pt,0) + Fx*Hvbasis_grad(elem,dof,pt,0) + Fy*Hvbasis_grad(elem,dof,pt,1);
      }
      
    }
    
  });
  
}


// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowwater<EvalT>::boundaryResidual() {
  
}


// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void shallowwater<EvalT>::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowwater<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;

  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "H")
      H_num = i;
    if (varlist[i] == "Hu")
      Hu_num = i;
    if (varlist[i] == "Hv")
      Hv_num = i;
  }
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::shallowwater<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::shallowwater<AD>;

// Standard built-in types
template class MrHyDE::shallowwater<AD2>;
template class MrHyDE::shallowwater<AD4>;
template class MrHyDE::shallowwater<AD8>;
template class MrHyDE::shallowwater<AD16>;
template class MrHyDE::shallowwater<AD18>;
template class MrHyDE::shallowwater<AD24>;
template class MrHyDE::shallowwater<AD32>;
#endif
