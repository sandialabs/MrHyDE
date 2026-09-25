/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "mhd.hpp"
using namespace MrHyDE;

// TODO BWR -- rho is both on the convective part but we have nu and rho*source showing up too
// this is inconsistent and needs fixing!

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
MHD<EvalT>::MHD(Teuchos::ParameterList & settings, const int & dimension_)
: PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "MHD";
  int spaceDim = dimension_;
  
  if (spaceDim < 3) {
    // throw an error -- just 3D for now
    // 2D will be faked using periodic BCs and one element in z as in Shadid paper
  }
  
  // TMW: note that this is 9 variables, so for the lowest order basis functions,
  //      we will need 72 derivatives on hex elements and 36 on tets
  
  myvars.push_back("rhoux");
  myvars.push_back("rhouy");
  myvars.push_back("rhouz");
  myvars.push_back("rho");
  myvars.push_back("T");
  myvars.push_back("Bx");
  myvars.push_back("By");
  myvars.push_back("Bz");
  myvars.push_back("psi"); // Lagrange multiplier for the involution contraint
  
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  
  use_stabilization = settings.get<bool>("use stabilization",true);
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void MHD<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                 Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("Re",fs.get<string>("Reynolds number","1.0"),"ip");
  functionManager->addFunction("S",fs.get<string>("Lundquist number","1.0"),"ip");
  functionManager->addFunction("gamma",fs.get<string>("heat capacity","1.0"),"ip");
  functionManager->addFunction("ndens",fs.get<string>("number density","1.0"),"ip");
  functionManager->addFunction("viscosity",fs.get<string>("viscosity","1.0"),"ip");
  functionManager->addFunction("eta",fs.get<string>("resistivity","1.0"),"ip");
  functionManager->addFunction("kappa",fs.get<string>("heat conductivity","1.0"),"ip");
  functionManager->addFunction("mu0",fs.get<string>("magnetic permeability","1.0"),"ip");
  functionManager->addFunction("mu",fs.get<string>("dynamic viscosity","1.0"),"ip");
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void MHD<EvalT>::volumeResidual() {
  
  
  
  //ScalarT dt = wkset->deltat;
  //bool isTransient = wkset->isTransient;
  Vista<EvalT> viscosity, Re, S, gamma, ndens, eta, kappa, mu0, mu;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    viscosity = functionManager->evaluate("viscosity","ip");
    Re = functionManager->evaluate("Re","ip");
    S = functionManager->evaluate("S","ip");
    ndens = functionManager->evaluate("ndens","ip");
    gamma = functionManager->evaluate("gamma","ip");
    kappa = functionManager->evaluate("kappa","ip");
    eta = functionManager->evaluate("eta","ip");
    mu0 = functionManager->evaluate("mu0","ip");
    mu = functionManager->evaluate("mu","ip");
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  // Various solutiom fields needed for the residuals
  auto rho = wkset->getSolutionField("rho");
  auto drho_dx = wkset->getSolutionField("grad(rho)[x]");
  auto drho_dy = wkset->getSolutionField("grad(rho)[y]");
  auto drho_dz = wkset->getSolutionField("grad(rho)[z]");
  auto drho_dt = wkset->getSolutionField("rho_t");
  
  auto T = wkset->getSolutionField("T");
  auto dT_dx = wkset->getSolutionField("grad(T)[x]");
  auto dT_dy = wkset->getSolutionField("grad(T)[y]");
  auto dT_dz = wkset->getSolutionField("grad(T)[z]");
  auto dT_dt = wkset->getSolutionField("T_t");
  
  auto rhoux = wkset->getSolutionField("rhoux");
  auto rhouy = wkset->getSolutionField("rhouy");
  auto rhouz = wkset->getSolutionField("rhouz");
  auto drhoux_dx = wkset->getSolutionField("grad(rhoux)[x]");
  auto drhoux_dy = wkset->getSolutionField("grad(rhoux)[y]");
  auto drhoux_dz = wkset->getSolutionField("grad(rhoux)[z]");
  auto drhouy_dx = wkset->getSolutionField("grad(rhouy)[x]");
  auto drhouy_dy = wkset->getSolutionField("grad(rhouy)[y]");
  auto drhouy_dz = wkset->getSolutionField("grad(rhouy)[z]");
  auto drhouz_dx = wkset->getSolutionField("grad(rhouz)[x]");
  auto drhouz_dy = wkset->getSolutionField("grad(rhouz)[y]");
  auto drhouz_dz = wkset->getSolutionField("grad(rhouz)[z]");
  auto drhoux_dt = wkset->getSolutionField("rhoux_t");
  auto drhouy_dt = wkset->getSolutionField("rhouy_t");
  auto drhouz_dt = wkset->getSolutionField("rhouz_t");
  
  auto Bx = wkset->getSolutionField("Bx");
  auto By = wkset->getSolutionField("By");
  auto Bz = wkset->getSolutionField("Bz");
  auto dBx_dx = wkset->getSolutionField("grad(Bx)[x]");
  auto dBx_dy = wkset->getSolutionField("grad(Bx)[y]");
  auto dBx_dz = wkset->getSolutionField("grad(Bx)[z]");
  auto dBy_dx = wkset->getSolutionField("grad(By)[x]");
  auto dBy_dy = wkset->getSolutionField("grad(By)[y]");
  auto dBy_dz = wkset->getSolutionField("grad(By)[z]");
  auto dBz_dx = wkset->getSolutionField("grad(Bz)[x]");
  auto dBz_dy = wkset->getSolutionField("grad(Bz)[y]");
  auto dBz_dz = wkset->getSolutionField("grad(Bz)[z]");
  auto dBx_dt = wkset->getSolutionField("Bx_t");
  auto dBy_dt = wkset->getSolutionField("By_t");
  auto dBz_dt = wkset->getSolutionField("Bz_t");
  
  auto psi = wkset->getSolutionField("psi");
  auto dpsi_dx = wkset->getSolutionField("grad(psi)[x]");
  auto dpsi_dy = wkset->getSolutionField("grad(psi)[y]");
  auto dpsi_dz = wkset->getSolutionField("grad(psi)[z]");
  
  bool use_stab = use_stabilization; // seems redundant but necessary to capture properly in lambdas
  auto hsize = wkset->getElementSize();
  ScalarT dt = wkset->deltat;
  
  // TMW: we will frequently need spatial derivatis of u, e.g., dux_dx, which we don't have
  // AD cannot help us here because we need spatial derivatives
  // So we compute:
  //    dux_dx = d/dx(rhoux/rho) = (drhoux_dx*rho - drho_dx*rhoux)/rho^2
  
  // Set of 5 equations (really 9 since two are vectors)
  // drho/dt + \div \cdot (\rhou) = 0
  {
    int rho_basis = wkset->usebasis[rho_num];
    auto basis = wkset->basis[rho_basis];
    auto basis_grad = wkset->basis_grad[rho_basis];
    auto off = subview(wkset->offsets,rho_num,ALL());
    
    parallel_for("MHD rho volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT Fx = rhoux(elem,pt)*wts(elem,pt);
        EvalT Fy = rhouy(elem,pt)*wts(elem,pt);
        EvalT Fz = rhouz(elem,pt)*wts(elem,pt);
        EvalT F = drho_dt(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          //res(elem,off(dof)) += F*basis(elem,dof,pt,0);
          res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
        }
        bool localflag = true;
        if (use_stab && localflag) {
          EvalT ux = rhoux(elem,pt)/rho(elem,pt);
          EvalT uy = rhouy(elem,pt)/rho(elem,pt);
          EvalT uz = rhouz(elem,pt)/rho(elem,pt);
          EvalT dux_dx = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dy = (drhouy_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dz = (drhouz_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          
          EvalT tauu = this->computeTauU(ux, uy, uz, Bx(elem,pt), By(elem,pt), Bz(elem,pt),
                                         mu(elem,pt), ndens(elem,pt), hsize(elem), dt);
          EvalT taup = this->computeTauP(tauu, hsize(elem));
          
          EvalT srhores = this->computeStrongResidualRho(drho_dt(elem,pt), drhoux_dx(elem,pt),
                                                         drhouy_dy(elem,pt), drhouz_dz(elem,pt));
          EvalT rho_prime = 0.0;//taup*srhores*wts(elem,pt);
          
          EvalT srhouxres = this->computeStrongResidualRhoux(drhoux_dt(elem,pt), rhoux(elem,pt), drhoux_dx(elem,pt),
                                                             drhoux_dy(elem,pt), drhoux_dz(elem,pt), ux, uy, uz,
                                                             dux_dx, duy_dy, duz_dz, ndens(elem,pt), dT_dx(elem,pt));
          EvalT ux_prime = tauu*srhouxres*wts(elem,pt);
          
          EvalT srhouyres = this->computeStrongResidualRhouy(drhouy_dt(elem,pt), rhouy(elem,pt), drhouy_dx(elem,pt),
                                                             drhouy_dy(elem,pt), drhouy_dz(elem,pt), ux, uy, uz,
                                                             dux_dx, duy_dy, duz_dz, ndens(elem,pt), dT_dy(elem,pt));
          EvalT uy_prime = tauu*srhouyres*wts(elem,pt);
          
          EvalT srhouzres = this->computeStrongResidualRhouz(drhouz_dt(elem,pt), rhouz(elem,pt), drhouz_dx(elem,pt),
                                                             drhouz_dy(elem,pt), drhouz_dz(elem,pt), ux, uy, uz,
                                                             dux_dx, duy_dy, duz_dz, ndens(elem,pt), dT_dz(elem,pt));
          EvalT uz_prime = tauu*srhouzres*wts(elem,pt);
          
          EvalT s_x = -rho(elem,pt)*ux_prime - ux*rho_prime;
          EvalT s_y = -rho(elem,pt)*uy_prime - uy*rho_prime;
          EvalT s_z = -rho(elem,pt)*uz_prime - uz*rho_prime;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
          }
        }
      }
    });
  }
  
  // drhou/dt + \div \cdot((\rhou x u + pI + 2/3*(1/Re)*\div(u)I - 1/Re*(grad(u) + grad(u)^T) - j x B = 0
  // Using p = nT, but assuming n is constant in stabilization term
  {
    int rhoux_basis = wkset->usebasis[rhoux_num];
    auto basis = wkset->basis[rhoux_basis];
    auto basis_grad = wkset->basis_grad[rhoux_basis];
    auto off = subview(wkset->offsets,rhoux_num,ALL());
    
    parallel_for("MHD rhoux volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT jy = 1.0/mu0(elem,pt)*(dBx_dz(elem,pt) - dBz_dx(elem,pt));
        EvalT jz = 1.0/mu0(elem,pt)*(dBy_dx(elem,pt) - dBx_dy(elem,pt));
        EvalT ux = rhoux(elem,pt)/rho(elem,pt);
        EvalT uy = rhouy(elem,pt)/rho(elem,pt);
        EvalT uz = rhouz(elem,pt)/rho(elem,pt);
        EvalT p = ndens(elem,pt)*T(elem,pt);
        EvalT dux_dx = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT dux_dy = (drhoux_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT dux_dz = (drhoux_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duy_dx = (drhouy_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duy_dy = (drhouy_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duz_dx = (drhouz_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duz_dz = (drhouz_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT Fx = (-1.0*rhoux(elem,pt)*ux - p - 2.0/3.0*1.0/Re(elem,pt)*(dux_dx+duy_dy+duz_dz) + 1.0/Re(elem,pt)*2.0*dux_dx)*wts(elem,pt);
        EvalT Fy = (-1.0*rhoux(elem,pt)*uy + 1.0/Re(elem,pt)*(dux_dy+duy_dx))*wts(elem,pt);
        EvalT Fz = (-1.0*rhoux(elem,pt)*uz + 1.0/Re(elem,pt)*(dux_dz+duz_dx))*wts(elem,pt);
        EvalT F = (drhoux_dt(elem,pt) - (jy*Bz(elem,pt) - jz*By(elem,pt)))*wts(elem,pt);
        
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
        }
        
        bool localflag = true;
        if (use_stab && localflag) {
          
          EvalT tauu = this->computeTauU(ux, uy, uz, Bx(elem,pt), By(elem,pt), Bz(elem,pt),
                                         mu(elem,pt), ndens(elem,pt), hsize(elem), dt);
          EvalT taup = this->computeTauP(tauu, hsize(elem));
          
          EvalT srhores = this->computeStrongResidualRho(drho_dt(elem,pt), drhoux_dx(elem,pt),
                                                         drhouy_dy(elem,pt), drhouz_dz(elem,pt));
          EvalT rho_prime = 0.0;  //taup*srhores*wts(elem,pt);
          
          EvalT srhouxres = this->computeStrongResidualRhoux(drhoux_dt(elem,pt), rhoux(elem,pt), drhoux_dx(elem,pt),
                                                             drhoux_dy(elem,pt), drhoux_dz(elem,pt), ux, uy, uz,
                                                             dux_dx, duy_dy, duz_dz, ndens(elem,pt), dT_dx(elem,pt));
          EvalT ux_prime = tauu*srhouxres*wts(elem,pt);
          
          EvalT s_x = rho(elem,pt)*ux_prime*ux + rho_prime*ux*ux + rho_prime;
          EvalT s_y = rho(elem,pt)*ux_prime*uy + rho_prime*ux*uy;
          EvalT s_z = rho(elem,pt)*ux_prime*uz + rho_prime*ux*uz;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
          }
        }
      }
    });
  }
  {
    int rhouy_basis = wkset->usebasis[rhouy_num];
    auto basis = wkset->basis[rhouy_basis];
    auto basis_grad = wkset->basis_grad[rhouy_basis];
    auto off = subview(wkset->offsets,rhouy_num,ALL());
    
    parallel_for("MHD rhouy volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT jx = 1.0/mu0(elem,pt)*(dBz_dy(elem,pt) - dBy_dz(elem,pt));
        EvalT jz = 1.0/mu0(elem,pt)*(dBy_dx(elem,pt) - dBx_dy(elem,pt));
        EvalT ux = rhoux(elem,pt)/rho(elem,pt);
        EvalT uy = rhouy(elem,pt)/rho(elem,pt);
        EvalT uz = rhouz(elem,pt)/rho(elem,pt);
        EvalT p = ndens(elem,pt)*T(elem,pt);
        EvalT duy_dx = (drhouy_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duy_dy = (drhouy_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duy_dz = (drhouy_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT dux_dx = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT dux_dy = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duz_dy = (drhouz_dx(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duz_dz = (drhouz_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT Fx = (-1.0*rhouy(elem,pt)*ux + 1.0/Re(elem,pt)*(dux_dy+duy_dx))*wts(elem,pt);
        EvalT Fy = (-1.0*rhouy(elem,pt)*uy - p - 2.0/3.0*1.0/Re(elem,pt)*(dux_dx + duy_dy + duz_dz) + 1.0/Re(elem,pt)*2.0*duy_dy)*wts(elem,pt);
        EvalT Fz = (-1.0*rhouy(elem,pt)*uz + 1.0/Re(elem,pt)*(duy_dz+duz_dy))*wts(elem,pt);
        EvalT F = (drhouy_dt(elem,pt) - (jz*Bx(elem,pt) - jx*Bz(elem,pt)))*wts(elem,pt);
        
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
        }
        
        bool localflag = true;
        if (use_stab && localflag) {
          
          EvalT tauu = this->computeTauU(ux, uy, uz, Bx(elem,pt), By(elem,pt), Bz(elem,pt),
                                         mu(elem,pt), ndens(elem,pt), hsize(elem), dt);
          EvalT taup = this->computeTauP(tauu, hsize(elem));
          
          EvalT srhores = this->computeStrongResidualRho(drho_dt(elem,pt), drhoux_dx(elem,pt),
                                                         drhouy_dy(elem,pt), drhouz_dz(elem,pt));
          EvalT rho_prime = 0.0;//taup*srhores*wts(elem,pt);
          
          EvalT srhouyres = this->computeStrongResidualRhouy(drhouy_dt(elem,pt), rhouy(elem,pt), drhouy_dx(elem,pt),
                                                             drhouy_dy(elem,pt), drhouy_dz(elem,pt), ux, uy, uz,
                                                             dux_dx, duy_dy, duz_dz, ndens(elem,pt), dT_dy(elem,pt));
          EvalT uy_prime = tauu*srhouyres*wts(elem,pt);
          
          EvalT s_x = rho(elem,pt)*uy_prime*ux + rho_prime*uy*ux;
          EvalT s_y = rho(elem,pt)*uy_prime*uy + rho_prime*uy*uy + rho_prime;
          EvalT s_z = rho(elem,pt)*uy_prime*uz + rho_prime*uy*uz;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
          }
        }
      }
    });
  }
  {
    int rhouz_basis = wkset->usebasis[rhouz_num];
    auto basis = wkset->basis[rhouz_basis];
    auto basis_grad = wkset->basis_grad[rhouz_basis];
    auto off = subview(wkset->offsets,rhouz_num,ALL());
    
    parallel_for("MHD rhouz volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT jx = 1.0/mu0(elem,pt)*(dBz_dy(elem,pt) - dBy_dz(elem,pt));
        EvalT jy = 1.0/mu0(elem,pt)*(dBx_dz(elem,pt) - dBz_dx(elem,pt));
        EvalT ux = rhoux(elem,pt)/rho(elem,pt);
        EvalT uy = rhouy(elem,pt)/rho(elem,pt);
        EvalT uz = rhouz(elem,pt)/rho(elem,pt);
        EvalT p = ndens(elem,pt)*T(elem,pt);
        EvalT duz_dx = (drhouz_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duz_dy = (drhouz_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duz_dz = (drhouz_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT dux_dx = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT dux_dz = (drhoux_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duy_dy = (drhouy_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duy_dz = (drhouy_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT Fx = (-1.0*rhouz(elem,pt)*ux + 1.0/Re(elem,pt)*(dux_dz+duz_dx))*wts(elem,pt);
        EvalT Fy = (-1.0*rhouz(elem,pt)*uy + 1.0/Re(elem,pt)*(duz_dy+duy_dz))*wts(elem,pt);
        EvalT Fz = (-1.0*rhouz(elem,pt)*uz - p - 2.0/3.0*1.0/Re(elem,pt)*(dux_dx + duy_dy + duz_dz) + 1.0/Re(elem,pt)*2.0*duz_dz)*wts(elem,pt);
        EvalT F = (drhouz_dt(elem,pt) - (jx*By(elem,pt) - jy*Bx(elem,pt)))*wts(elem,pt);
        
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
        }
        
        bool localflag = true;
        if (use_stab && localflag) {
          
          EvalT tauu = this->computeTauU(ux, uy, uz, Bx(elem,pt), By(elem,pt), Bz(elem,pt),
                                         mu(elem,pt), ndens(elem,pt), hsize(elem), dt);
          EvalT taup = this->computeTauP(tauu, hsize(elem));
          
          EvalT srhores = this->computeStrongResidualRho(drho_dt(elem,pt), drhoux_dx(elem,pt),
                                                         drhouy_dy(elem,pt), drhouz_dz(elem,pt));
          EvalT rho_prime = 0.0; //taup*srhores*wts(elem,pt);
          
          EvalT srhouzres = this->computeStrongResidualRhouz(drhouz_dt(elem,pt), rhouz(elem,pt), drhouz_dx(elem,pt),
                                                             drhouz_dy(elem,pt), drhouz_dz(elem,pt), ux, uy, uz,
                                                             dux_dx, duy_dy, duz_dz, ndens(elem,pt), dT_dz(elem,pt));
          EvalT uz_prime = tauu*srhouzres*wts(elem,pt);
          
          EvalT s_x = rho(elem,pt)*uz_prime*ux + rho_prime*uz*ux;
          EvalT s_y = rho(elem,pt)*uz_prime*uy + rho_prime*uz*uy;
          EvalT s_z = rho(elem,pt)*uz_prime*uz + rho_prime*uz*uz + rho_prime;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
          }
        }
      }
    });
  }

  // Energy equation
  // (n/gamma_bar)*(DT/dt + u \cdot \nabla T) + nT(\nabla \div u) + \nabla \cdot q - 1/S \|j\|^2 - 1/Re \pi : \nabla u
  {
    int T_basis = wkset->usebasis[T_num];
    auto basis = wkset->basis[T_basis];
    auto basis_grad = wkset->basis_grad[T_basis];
    auto off = subview(wkset->offsets,T_num,ALL());
    
    parallel_for("MHD T volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT gamma_bar = gamma(elem,pt) - 1.0;
        
        EvalT jx = 1.0/mu0(elem,pt)*(dBz_dy(elem,pt) - dBy_dz(elem,pt));
        EvalT jy = 1.0/mu0(elem,pt)*(dBx_dz(elem,pt) - dBz_dx(elem,pt));
        EvalT jz = 1.0/mu0(elem,pt)*(dBy_dx(elem,pt) - dBx_dy(elem,pt));
        
        EvalT qx = -1.0*kappa(elem,pt)*dT_dx(elem,pt);
        EvalT qy = -1.0*kappa(elem,pt)*dT_dy(elem,pt);
        EvalT qz = -1.0*kappa(elem,pt)*dT_dz(elem,pt);
        
        EvalT ux = rhoux(elem,pt)/rho(elem,pt);
        EvalT uy = rhouy(elem,pt)/rho(elem,pt);
        EvalT uz = rhouz(elem,pt)/rho(elem,pt);
        
        EvalT dux_dx = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT dux_dy = (drhoux_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT dux_dz = (drhoux_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duy_dx = (drhouy_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duy_dy = (drhouy_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duy_dz = (drhouy_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duz_dx = (drhouz_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duz_dy = (drhouz_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        EvalT duz_dz = (drhouz_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
        
        EvalT pi_xx = 2.0/3.0*mu(elem,pt)*(dux_dx+duy_dy+duz_dz) - mu(elem,pt)*(2.0*dux_dx);
        EvalT pi_xy = -1.0*mu(elem,pt)*(dux_dy+duy_dx);
        EvalT pi_xz = -1.0*mu(elem,pt)*(dux_dz+duz_dx);
                
        EvalT pi_yx = -1.0*mu(elem,pt)*(dux_dy+duy_dx);
        EvalT pi_yy = 2.0/3.0*mu(elem,pt)*(dux_dx+duy_dy+duz_dz) - mu(elem,pt)*(2.0*duy_dy);
        EvalT pi_yz = -1.0*mu(elem,pt)*(duy_dz+duz_dy);
                
        EvalT pi_zx = -1.0*mu(elem,pt)*(dux_dz+duz_dx);
        EvalT pi_zy = -1.0*mu(elem,pt)*(duz_dy+duy_dz);
        EvalT pi_zz = 2.0/3.0*mu(elem,pt)*(dux_dx+duy_dy+duz_dz) - mu(elem,pt)*(2.0*duz_dz);
        
        EvalT viscous_stress = -1.0/Re(elem,pt)*(pi_xx*dux_dx + pi_xy*dux_dy + pi_xz*dux_dz + pi_yx*duy_dx + pi_yy*duy_dy + pi_yz*duy_dz + pi_zx*duz_dx + pi_zy*duz_dy + pi_zz*duz_dz);
        
        EvalT Fx = (ndens(elem,pt)/gamma_bar*ux + qx)*wts(elem,pt);
        EvalT Fy = (ndens(elem,pt)/gamma_bar*uy + qy)*wts(elem,pt);
        EvalT Fz = (ndens(elem,pt)/gamma_bar*uz + qz)*wts(elem,pt);
        EvalT F = (ndens(elem,pt)/gamma_bar*dT_dt(elem,pt))*wts(elem,pt);
        //EvalT F = (ndens(elem,pt)/gamma_bar*dT_dt(elem,pt) + ndens(elem,pt)*T(elem,pt)*(dux_dx+duy_dy+duz_dz) - 1.0/S(elem,pt)*(jx*jx+jy*jy+jz*jz) - 1.0/Re(elem,pt)*viscous_stress)*wts(elem,pt);
        
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          res(elem,off(dof)) += F*basis(elem,dof,pt,0);
        }
        
        bool localflag = true;
        if (use_stab && localflag) {
          
          EvalT tauT = this->computeTauT(ux, uy, uz, Bx(elem,pt), By(elem,pt), Bz(elem,pt),
                                         rho(elem,pt), ndens(elem,pt), kappa(elem,pt), hsize(elem), dt);
          
          EvalT sTres = this->computeStrongResidualT(dT_dt(elem,pt), T(elem,pt), dT_dx(elem,pt),
                                                     dT_dy(elem,pt), dT_dz(elem,pt),
                                                     ux, uy, uz,
                                                     dux_dx, dux_dy, dux_dz,
                                                     duy_dx, duy_dy, duy_dz,
                                                     duz_dx, duz_dy, duz_dz,
                                                     qx, qy, qz,
                                                     jx, jy, jz,
                                                     pi_xx, pi_xy, pi_xz,
                                                     pi_yx, pi_yy, pi_yz,
                                                     pi_zx, pi_zy, pi_zz,
                                                     ndens(elem,pt), gamma_bar, Re(elem,pt),
                                                     S(elem,pt));
          EvalT T_prime = tauT*sTres*wts(elem,pt);
          
          EvalT s_x = 1.0/gamma_bar*ux*T_prime;
          EvalT s_y = 1.0/gamma_bar*uy*T_prime;
          EvalT s_z = 1.0/gamma_bar*uz*T_prime;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
          }
        }
      }
    });
  }

  // Magnetic field equation
  // dB/dt + \nabla \cdot (u ox B - B ox u - 1/S(\nabla B - \nabla B^T) + \psi I) = 0
  {
    int Bx_basis = wkset->usebasis[Bx_num];
    auto basis = wkset->basis[Bx_basis];
    auto basis_grad = wkset->basis_grad[Bx_basis];
    auto off = subview(wkset->offsets,Bx_num,ALL());
    
    parallel_for("MHD Bx volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        
        EvalT ux = rhoux(elem,pt)/rho(elem,pt);
        EvalT uy = rhouy(elem,pt)/rho(elem,pt);
        EvalT uz = rhouz(elem,pt)/rho(elem,pt);
        
        // D = u ox B - B ox u
        EvalT D_x = ux*Bx(elem,pt) - Bx(elem,pt)*ux;
        EvalT D_y = ux*By(elem,pt) - Bx(elem,pt)*uy;
        EvalT D_z = ux*Bz(elem,pt) - Bx(elem,pt)*uz;
        
        // H = 1/S*(nabla B - nabla B^T)
        EvalT H_x = 1.0/S(elem,pt)*(dBx_dx(elem,pt) - dBx_dx(elem,pt));
        EvalT H_y = 1.0/S(elem,pt)*(dBx_dy(elem,pt) - dBy_dx(elem,pt));
        EvalT H_z = 1.0/S(elem,pt)*(dBx_dz(elem,pt) - dBz_dx(elem,pt));
        
        EvalT Fx = (D_x - H_x + psi(elem,pt))*wts(elem,pt);
        EvalT Fy = (D_y - H_y)*wts(elem,pt);
        EvalT Fz = (D_z - H_z)*wts(elem,pt);
        EvalT F = (dBx_dt(elem,pt))*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
        }
        
        bool localflag = true;
        if (use_stab && localflag) {
          
          EvalT dux_dx = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT dux_dy = (drhoux_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT dux_dz = (drhoux_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dx = (drhouy_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dy = (drhouy_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dz = (drhouy_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dx = (drhouz_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dy = (drhouz_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dz = (drhouz_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          
          EvalT tauB = this->computeTauB(ux, uy, uz, Bx(elem,pt), By(elem,pt), Bz(elem,pt), mu0(elem,pt),
                                         ndens(elem,pt), hsize(elem), dt);
          EvalT taupsi = this->computeTauPsi(tauB, hsize(elem));
          
          EvalT sBxres = this->computeStrongResidualBx(dBx_dt(elem,pt), Bx(elem,pt), dBx_dx(elem,pt),
                                                       dBx_dy(elem,pt), dBx_dz(elem,pt), By(elem,pt), Bz(elem,pt),
                                                       dBy_dy(elem,pt), dBz_dz(elem,pt), ux, uy, uz,
                                                       dux_dx, dux_dy, dux_dz, duy_dy, duz_dz,
                                                       S(elem,pt), dpsi_dx(elem,pt));
          EvalT sByres = this->computeStrongResidualBy(dBy_dt(elem,pt), By(elem,pt), dBy_dx(elem,pt),
                                                       dBy_dy(elem,pt), dBy_dz(elem,pt),
                                                       Bx(elem,pt), Bz(elem,pt),
                                                       dBx_dx(elem,pt), dBz_dz(elem,pt), ux, uy, uz,
                                                       dux_dx, duy_dx, duy_dy, duy_dz, duz_dz,
                                                       S(elem,pt), dpsi_dy(elem,pt));
          EvalT sBzres = this->computeStrongResidualBz(dBz_dt(elem,pt), Bz(elem,pt), dBz_dx(elem,pt),
                                                       dBz_dy(elem,pt), dBz_dz(elem,pt),
                                                       Bx(elem,pt), By(elem,pt),
                                                       dBx_dx(elem,pt), dBy_dy(elem,pt), ux, uy, uz,
                                                       dux_dx, duy_dy, duz_dx, duz_dy, duz_dz,
                                                       S(elem,pt), dpsi_dz(elem,pt));
          EvalT spsires = this->computeStrongResidualPsi(dBx_dx(elem,pt), dBy_dy(elem,pt), dBz_dz(elem,pt));
          
          EvalT Bx_prime = tauB*sBxres*wts(elem,pt);
          EvalT By_prime = tauB*sByres*wts(elem,pt);
          EvalT Bz_prime = tauB*sBzres*wts(elem,pt);
          EvalT psi_prime = taupsi*spsires*wts(elem,pt);
          
          EvalT s_x = ux*Bx_prime - Bx_prime*ux + psi_prime;
          EvalT s_y = ux*By_prime - Bx_prime*uy;
          EvalT s_z = ux*Bz_prime - Bx_prime*uz;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
          }
        }
      }
    });
  }
  {
    int By_basis = wkset->usebasis[By_num];
    auto basis = wkset->basis[By_basis];
    auto basis_grad = wkset->basis_grad[By_basis];
    auto off = subview(wkset->offsets,By_num,ALL());
    
    parallel_for("MHD By volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        
        EvalT ux = rhoux(elem,pt)/rho(elem,pt);
        EvalT uy = rhouy(elem,pt)/rho(elem,pt);
        EvalT uz = rhouz(elem,pt)/rho(elem,pt);
        
        // D = u ox B - B ox u
        EvalT D_x = uy*Bx(elem,pt) - By(elem,pt)*ux;
        EvalT D_y = uy*By(elem,pt) - By(elem,pt)*uy;
        EvalT D_z = uy*Bz(elem,pt) - By(elem,pt)*uz;
        
        // H = 1/S*(nabla B - nabla B^T)
        EvalT H_x = 1.0/S(elem,pt)*(dBy_dx(elem,pt) - dBx_dy(elem,pt));
        EvalT H_y = 1.0/S(elem,pt)*(dBy_dy(elem,pt) - dBy_dy(elem,pt));
        EvalT H_z = 1.0/S(elem,pt)*(dBy_dz(elem,pt) - dBz_dy(elem,pt));
        
        EvalT Fx = (D_x - H_x)*wts(elem,pt);
        EvalT Fy = (D_y - H_y + psi(elem,pt))*wts(elem,pt);
        EvalT Fz = (D_z - H_z)*wts(elem,pt);
        EvalT F = (dBy_dt(elem,pt))*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
        }
        
        bool localflag = true;
        if (use_stab && localflag) {
          
          EvalT dux_dx = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT dux_dy = (drhoux_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT dux_dz = (drhoux_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dx = (drhouy_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dy = (drhouy_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dz = (drhouy_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dx = (drhouz_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dy = (drhouz_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dz = (drhouz_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          
          EvalT tauB = this->computeTauB(ux, uy, uz, Bx(elem,pt), By(elem,pt), Bz(elem,pt), mu0(elem,pt),
                                         ndens(elem,pt), hsize(elem), dt);
          EvalT taupsi = this->computeTauPsi(tauB, hsize(elem));
          
          EvalT sBxres = this->computeStrongResidualBx(dBx_dt(elem,pt), Bx(elem,pt), dBx_dx(elem,pt),
                                                       dBx_dy(elem,pt), dBx_dz(elem,pt), By(elem,pt), Bz(elem,pt),
                                                       dBy_dy(elem,pt), dBz_dz(elem,pt), ux, uy, uz,
                                                       dux_dx, dux_dy, dux_dz, duy_dy, duz_dz,
                                                       S(elem,pt), dpsi_dx(elem,pt));
          EvalT sByres = this->computeStrongResidualBy(dBy_dt(elem,pt), By(elem,pt), dBy_dx(elem,pt),
                                                       dBy_dy(elem,pt), dBy_dz(elem,pt),
                                                       Bx(elem,pt), Bz(elem,pt),
                                                       dBx_dx(elem,pt), dBz_dz(elem,pt), ux, uy, uz,
                                                       dux_dx, duy_dx, duy_dy, duy_dz, duz_dz,
                                                       S(elem,pt), dpsi_dy(elem,pt));
          EvalT sBzres = this->computeStrongResidualBz(dBz_dt(elem,pt), Bz(elem,pt), dBz_dx(elem,pt),
                                                       dBz_dy(elem,pt), dBz_dz(elem,pt),
                                                       Bx(elem,pt), By(elem,pt),
                                                       dBx_dx(elem,pt), dBy_dy(elem,pt), ux, uy, uz,
                                                       dux_dx, duy_dy, duz_dx, duz_dy, duz_dz,
                                                       S(elem,pt), dpsi_dz(elem,pt));
          EvalT spsires = this->computeStrongResidualPsi(dBx_dx(elem,pt), dBy_dy(elem,pt), dBz_dz(elem,pt));
          
          EvalT Bx_prime = tauB*sBxres*wts(elem,pt);
          EvalT By_prime = tauB*sByres*wts(elem,pt);
          EvalT Bz_prime = tauB*sBzres*wts(elem,pt);
          EvalT psi_prime = taupsi*spsires*wts(elem,pt);
          
          EvalT s_x = uy*Bx_prime - By_prime*ux;
          EvalT s_y = uy*By_prime - By_prime*uy + psi_prime;
          EvalT s_z = uy*Bz_prime - By_prime*uz;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
          }
        }
      }
    });
  }
  {
    int Bz_basis = wkset->usebasis[Bz_num];
    auto basis = wkset->basis[Bz_basis];
    auto basis_grad = wkset->basis_grad[Bz_basis];
    auto off = subview(wkset->offsets,Bz_num,ALL());
    
    parallel_for("MHD Bz volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        
        EvalT ux = rhoux(elem,pt)/rho(elem,pt);
        EvalT uy = rhouy(elem,pt)/rho(elem,pt);
        EvalT uz = rhouz(elem,pt)/rho(elem,pt);
        
        // D = u ox B - B ox u
        EvalT D_x = uz*Bx(elem,pt) - Bz(elem,pt)*ux;
        EvalT D_y = uz*By(elem,pt) - Bz(elem,pt)*uy;
        EvalT D_z = uz*Bz(elem,pt) - Bz(elem,pt)*uz;
        
        // H = 1/S*(nabla B - nabla B^T)
        EvalT H_x = 1.0/S(elem,pt)*(dBz_dx(elem,pt) - dBx_dz(elem,pt));
        EvalT H_y = 1.0/S(elem,pt)*(dBz_dy(elem,pt) - dBy_dz(elem,pt));
        EvalT H_z = 1.0/S(elem,pt)*(dBz_dz(elem,pt) - dBz_dz(elem,pt));
        
        EvalT Fx = (D_x - H_x)*wts(elem,pt);
        EvalT Fy = (D_y - H_y)*wts(elem,pt);
        EvalT Fz = (D_z - H_z + psi(elem,pt))*wts(elem,pt);
        EvalT F = (dBz_dt(elem,pt))*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
        }
        
        bool localflag = true;
        if (use_stab && localflag) {
          
          EvalT dux_dx = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT dux_dy = (drhoux_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT dux_dz = (drhoux_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dx = (drhouy_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dy = (drhouy_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dz = (drhouy_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dx = (drhouz_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dy = (drhouz_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dz = (drhouz_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          
          EvalT tauB = this->computeTauB(ux, uy, uz, Bx(elem,pt), By(elem,pt), Bz(elem,pt), mu0(elem,pt),
                                         ndens(elem,pt), hsize(elem), dt);
          EvalT taupsi = this->computeTauPsi(tauB, hsize(elem));
          
          EvalT sBxres = this->computeStrongResidualBx(dBx_dt(elem,pt), Bx(elem,pt), dBx_dx(elem,pt),
                                                       dBx_dy(elem,pt), dBx_dz(elem,pt), By(elem,pt), Bz(elem,pt),
                                                       dBy_dy(elem,pt), dBz_dz(elem,pt), ux, uy, uz,
                                                       dux_dx, dux_dy, dux_dz, duy_dy, duz_dz,
                                                       S(elem,pt), dpsi_dx(elem,pt));
          EvalT sByres = this->computeStrongResidualBy(dBy_dt(elem,pt), By(elem,pt), dBy_dx(elem,pt),
                                                       dBy_dy(elem,pt), dBy_dz(elem,pt),
                                                       Bx(elem,pt), Bz(elem,pt),
                                                       dBx_dx(elem,pt), dBz_dz(elem,pt), ux, uy, uz,
                                                       dux_dx, duy_dx, duy_dy, duy_dz, duz_dz,
                                                       S(elem,pt), dpsi_dy(elem,pt));
          EvalT sBzres = this->computeStrongResidualBz(dBz_dt(elem,pt), Bz(elem,pt), dBz_dx(elem,pt),
                                                       dBz_dy(elem,pt), dBz_dz(elem,pt),
                                                       Bx(elem,pt), By(elem,pt),
                                                       dBx_dx(elem,pt), dBy_dy(elem,pt), ux, uy, uz,
                                                       dux_dx, duy_dy, duz_dx, duz_dy, duz_dz,
                                                       S(elem,pt), dpsi_dz(elem,pt));
          EvalT spsires = this->computeStrongResidualPsi(dBx_dx(elem,pt), dBy_dy(elem,pt), dBz_dz(elem,pt));
          
          EvalT Bx_prime = tauB*sBxres*wts(elem,pt);
          EvalT By_prime = tauB*sByres*wts(elem,pt);
          EvalT Bz_prime = tauB*sBzres*wts(elem,pt);
          EvalT psi_prime = taupsi*spsires*wts(elem,pt);
          
          EvalT s_x = uz*Bx_prime - Bz_prime*ux;
          EvalT s_y = uz*By_prime - Bz_prime*uy;
          EvalT s_z = uz*Bz_prime - Bz_prime*uz + psi_prime;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
          }
        }
      }
    });
  }
  
  {
    int psi_basis = wkset->usebasis[psi_num];
    auto basis = wkset->basis[psi_basis];
    auto basis_grad = wkset->basis_grad[psi_basis];
    auto off = subview(wkset->offsets,psi_num,ALL());
    
    parallel_for("MHD psi volume resid",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        
        EvalT F = (dBx_dx(elem,pt) + dBy_dy(elem,pt) + dBz_dz(elem,pt))*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++) {
          res(elem,off(dof)) += F*basis(elem,dof,pt,0);
        }
        
        bool localflag = false;
        if (use_stab && localflag) {
          
          EvalT ux = rhoux(elem,pt)/rho(elem,pt);
          EvalT uy = rhouy(elem,pt)/rho(elem,pt);
          EvalT uz = rhouz(elem,pt)/rho(elem,pt);
          
          EvalT dux_dx = (drhoux_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT dux_dy = (drhoux_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT dux_dz = (drhoux_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhoux(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dx = (drhouy_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dy = (drhouy_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duy_dz = (drhouy_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouy(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dx = (drhouz_dx(elem,pt)*rho(elem,pt) - drho_dx(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dy = (drhouz_dy(elem,pt)*rho(elem,pt) - drho_dy(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          EvalT duz_dz = (drhouz_dz(elem,pt)*rho(elem,pt) - drho_dz(elem,pt)*rhouz(elem,pt))/(rho(elem,pt)*rho(elem,pt));
          
          EvalT tauB = this->computeTauB(ux, uy, uz, Bx(elem,pt), By(elem,pt), Bz(elem,pt), mu0(elem,pt),
                                         ndens(elem,pt), hsize(elem), dt);
          
          EvalT sBxres = this->computeStrongResidualBx(dBx_dt(elem,pt), Bx(elem,pt), dBx_dx(elem,pt),
                                                       dBx_dy(elem,pt), dBx_dz(elem,pt), By(elem,pt), Bz(elem,pt),
                                                       dBy_dy(elem,pt), dBz_dz(elem,pt), ux, uy, uz,
                                                       dux_dx, dux_dy, dux_dz, duy_dy, duz_dz,
                                                       S(elem,pt), dpsi_dx(elem,pt));
          EvalT sByres = this->computeStrongResidualBy(dBy_dt(elem,pt), By(elem,pt), dBy_dx(elem,pt),
                                                       dBy_dy(elem,pt), dBy_dz(elem,pt),
                                                       Bx(elem,pt), Bz(elem,pt),
                                                       dBx_dx(elem,pt), dBz_dz(elem,pt), ux, uy, uz,
                                                       dux_dx, duy_dx, duy_dy, duy_dz, duz_dz,
                                                       S(elem,pt), dpsi_dy(elem,pt));
          EvalT sBzres = this->computeStrongResidualBz(dBz_dt(elem,pt), Bz(elem,pt), dBz_dx(elem,pt),
                                                       dBz_dy(elem,pt), dBz_dz(elem,pt),
                                                       Bx(elem,pt), By(elem,pt),
                                                       dBx_dx(elem,pt), dBy_dy(elem,pt), ux, uy, uz,
                                                       dux_dx, duy_dy, duz_dx, duz_dy, duz_dz,
                                                       S(elem,pt), dpsi_dz(elem,pt));
          
          EvalT Bx_prime = tauB*sBxres*wts(elem,pt);
          EvalT By_prime = tauB*sByres*wts(elem,pt);
          EvalT Bz_prime = tauB*sBzres*wts(elem,pt);
          
          EvalT s_x = Bx_prime;
          EvalT s_y = By_prime;
          EvalT s_z = Bz_prime;
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            res(elem,off(dof)) += s_x*basis_grad(elem,dof,pt,0) + s_y*basis_grad(elem,dof,pt,1) + s_z*basis_grad(elem,dof,pt,2);
          }
        }
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void MHD<EvalT>::boundaryResidual() {
  
  /*
   int spaceDim = wkset->dimension;
   auto bcs = wkset->var_bcs;
   
   int cside = wkset->currentside;
   
   string ux_sidetype = bcs(ux_num,cside);
   string uy_sidetype = "Dirichlet";
   string uz_sidetype = "Dirichlet";
   if (spaceDim > 1) {
   uy_sidetype = bcs(uy_num,cside);
   }
   if (spaceDim > 2) {
   uz_sidetype = bcs(uz_num,cside);
   }
   
   Vista<EvalT> source_ux, source_uy, source_uz;
   
   if (ux_sidetype != "Dirichlet" || uy_sidetype != "Dirichlet" || uz_sidetype != "Dirichlet") {
   
   {
   //Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
   if (ux_sidetype == "Neumann") {
   source_ux = functionManager->evaluate("Neumann ux " + wkset->sidename,"side ip");
   }
   if (uy_sidetype == "Neumann") {
   source_uy = functionManager->evaluate("Neumann uy " + wkset->sidename,"side ip");
   }
   if (uz_sidetype == "Neumann") {
   source_uz = functionManager->evaluate("Neumann uz " + wkset->sidename,"side ip");
   }
   }
   
   // Since normals get recomputed often, this needs to be reset
   auto wts = wkset->wts_side;
   auto h = wkset->getSideElementSize();
   auto res = wkset->res;
   
   //Teuchos::TimeMonitor localtime(*boundaryResidualFill);
   
   if (spaceDim == 1) {
   int ux_basis = wkset->usebasis[ux_num];
   auto basis = wkset->basis_side[ux_basis];
   auto off = Kokkos::subview( wkset->offsets, ux_num, Kokkos::ALL());
   if (ux_sidetype == "Neumann") { // Neumann
   parallel_for("NS ux bndry resid 1D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_ux(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   else if (spaceDim == 2) {
   
   // ux equation boundary residual
   {
   int ux_basis = wkset->usebasis[ux_num];
   auto basis = wkset->basis_side[ux_basis];
   auto off = Kokkos::subview( wkset->offsets, ux_num, Kokkos::ALL());
   
   if (ux_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS ux bndry resid 2D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_ux(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   
   // uy equation boundary residual
   {
   int uy_basis = wkset->usebasis[uy_num];
   auto basis = wkset->basis_side[uy_basis];
   auto off = Kokkos::subview( wkset->offsets, uy_num, Kokkos::ALL());
   if (uy_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS uy bndry resid 2D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_uy(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   }
   
   else if (spaceDim == 3) {
   
   // ux equation boundary residual
   {
   int ux_basis = wkset->usebasis[ux_num];
   auto basis = wkset->basis_side[ux_basis];
   auto off = Kokkos::subview( wkset->offsets, ux_num, Kokkos::ALL());
   if (ux_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS ux bndry resid 3D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_ux(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   
   // uy equation boundary residual
   {
   int uy_basis = wkset->usebasis[uy_num];
   auto basis = wkset->basis_side[uy_basis];
   auto off = Kokkos::subview( wkset->offsets, uy_num, Kokkos::ALL());
   if (uy_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS uy bndry resid 3D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_uy(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   
   // uz equation boundary residual
   {
   int uz_basis = wkset->usebasis[uz_num];
   auto basis = wkset->basis_side[uz_basis];
   auto off = Kokkos::subview( wkset->offsets, uz_num, Kokkos::ALL());
   if (uz_sidetype == "Neumann") { // traction (Neumann)
   parallel_for("NS uz bndry resid 3D N",
   RangePolicy<AssemblyExec>(0,wkset->numElem),
   MRHYDE_LAMBDA (const int e ) {
   for (size_type k=0; k<basis.extent(2); k++ ) {
   for (size_type i=0; i<basis.extent(1); i++ ) {
   res(e,off(i)) += (-source_uz(e,k)*basis(e,i,k,0))*wts(e,k);
   }
   }
   });
   }
   }
   }
   }
   */
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void MHD<EvalT>::computeFlux() {
  
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void MHD<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
  
  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  rhoux_num = -1;
  rho_num = -1;
  T_num = -1;
  psi_num = -1;
  rhouy_num = -1;
  rhouz_num = -1;
  Bx_num = -1;
  By_num = -1;
  Bz_num = -1;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "rhoux")
      rhoux_num = i;
    if (varlist[i] == "rho")
      rho_num = i;
    if (varlist[i] == "T")
      T_num = i;
    if (varlist[i] == "psi")
      psi_num = i;
    if (varlist[i] == "rhouy")
      rhouy_num = i;
    if (varlist[i] == "rhouz")
      rhouz_num = i;
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

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeTauU(const EvalT & xvl, const EvalT & yvl, const EvalT & zvl,
                                              const EvalT & Bx, const EvalT & By, const EvalT & Bz,
                                              const EvalT & mu, const EvalT & ndens,
                                              const ScalarT & h, const ScalarT & dt) const {
  ScalarT C1 = 4.0;
  ScalarT C2 = 4.0;
  ScalarT C3 = 4.0;
  ScalarT C4 = 4.0;
  
  EvalT nvel = xvl*xvl + yvl*yvl + zvl*zvl;
  EvalT nB = Bx*Bx + By*By + Bz*Bz;
  if (nvel > 1.0E-12) {
    nvel = sqrt(nvel);
  }
  if (nB > 1.0E-12) {
    nB = sqrt(nB);
  }
  
  EvalT tau = C1*(2.0/dt)*(2.0/dt) + C2*(nvel/h)*(nvel/h) + C3*(nB/h)*(nB/h) + C4*mu*mu/ndens/ndens*h*h;
  tau = 1.0/sqrt(tau);
  
  return tau;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeTauT(const EvalT & xvl, const EvalT & yvl, const EvalT & zvl,
                                              const EvalT & Bx, const EvalT & By, const EvalT & Bz,
                                              const EvalT & rho, const EvalT & ndens,
                                              const EvalT & kappa, const ScalarT & h, const ScalarT & dt) const {
  ScalarT C1 = 4.0;
  ScalarT C2 = 4.0;
  ScalarT C3 = 4.0;
  ScalarT C4 = 4.0;
  
  EvalT nvel = xvl*xvl + yvl*yvl + zvl*zvl;
  EvalT nB = Bx*Bx + By*By + Bz*Bz;
  if (nvel > 1.0E-12) {
    nvel = sqrt(nvel);
  }
  if (nB > 1.0E-12) {
    nB = sqrt(nB);
  }
  
  EvalT tau = C1*(rho/dt)*(rho/dt) + C2*ndens*ndens*(nvel/h)*(nvel/h) + C3*(nB/h)*(nB/h) + C4*kappa*kappa*h*h;
  tau = 1.0/sqrt(tau);
  
  return tau;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeTauB(const EvalT & xvl, const EvalT & yvl, const EvalT & zvl,
                                              const EvalT & Bx, const EvalT & By, const EvalT & Bz,
                                              const EvalT & mu0, const EvalT & ndens,
                                              const ScalarT & h, const ScalarT & dt) const {
  ScalarT C1 = 4.0;
  ScalarT C2 = 4.0;
  ScalarT C3 = 4.0;
  ScalarT C4 = 4.0;
  
  EvalT nvel = xvl*xvl + yvl*yvl + zvl*zvl;
  EvalT nB = Bx*Bx + By*By + Bz*Bz;
  if (nvel > 1E-12) {
    nvel = sqrt(nvel);
  }
  if (nB > 1E-12) {
    nB = sqrt(nB);
  }
  
  EvalT tau = C1*(2.0/dt)*(2.0/dt) + C2*(nvel/h)*(nvel/h) + C3*(nB/h)*(nB/h) + C4*ndens*ndens/mu0/mu0*h*h;
  tau = 1./sqrt(tau);
  
  return tau;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeTauP(const EvalT & tauu, const ScalarT & h) const {
  return (h*h*tauu);
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeTauPsi(const EvalT & tauB, const ScalarT & h) const {
  return (h*h*tauB);
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeStrongResidualRho(const EvalT & drho_dt, const EvalT & drhoux_dx, const EvalT & drhouy_dy,
                                                           const EvalT & drhouz_dz) const {
  EvalT sres = drho_dt + drhoux_dx + drhouy_dy + drhouz_dz;
  // No higher-order terms
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeStrongResidualRhoux(const EvalT & drhoux_dt, const EvalT & rhoux, const EvalT & drhoux_dx,
                                                             const EvalT & drhoux_dy, const EvalT & drhoux_dz,
                                                             const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                             const EvalT & dux_dx, const EvalT & duy_dy, const EvalT & duz_dz,
                                                             const EvalT & ndens, const EvalT & dT_dx) {
  EvalT sres = drhoux_dt + (drhoux_dx*ux + rhoux*dux_dx + drhoux_dy*uy + rhoux*duy_dy + drhoux_dz*uz + rhoux*duz_dz) + ndens*dT_dx;
  // Missing approximations of the high-order terms
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeStrongResidualRhouy(const EvalT & drhouy_dt, const EvalT & rhouy, const EvalT & drhouy_dx,
                                                             const EvalT & drhouy_dy, const EvalT & drhouy_dz,
                                                             const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                             const EvalT & dux_dx, const EvalT & duy_dy, const EvalT & duz_dz,
                                                             const EvalT & ndens, const EvalT & dT_dy) {
  EvalT sres = drhouy_dt + (drhouy_dx*ux + rhouy*dux_dx + drhouy_dy*uy + rhouy*duy_dy + drhouy_dz*uz + rhouy*duz_dz) + ndens*dT_dy;
  // Missing approximations of the high-order terms
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeStrongResidualRhouz(const EvalT & drhouz_dt, const EvalT & rhouz, const EvalT & drhouz_dx,
                                                             const EvalT & drhouz_dy, const EvalT & drhouz_dz,
                                                             const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                             const EvalT & dux_dx, const EvalT & duy_dy, const EvalT & duz_dz,
                                                             const EvalT & ndens, const EvalT & dT_dz) {
  EvalT sres = drhouz_dt + (drhouz_dx*ux + rhouz*dux_dx + drhouz_dy*uy + rhouz*duy_dy + drhouz_dz*uz + rhouz*duz_dz) + ndens*dT_dz;
  // Missing approximations of the high-order terms
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeStrongResidualT(const EvalT & dT_dt, const EvalT & T, const EvalT & dT_dx,
                                                         const EvalT & dT_dy, const EvalT & dT_dz,
                                                         const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                         const EvalT & dux_dx, const EvalT & dux_dy, const EvalT & dux_dz,
                                                         const EvalT & duy_dx, const EvalT & duy_dy, const EvalT & duy_dz,
                                                         const EvalT & duz_dx, const EvalT & duz_dy, const EvalT & duz_dz,
                                                         const EvalT & qx, const EvalT & qy, const EvalT & qz,
                                                         const EvalT & jx, const EvalT & jy, const EvalT & jz,
                                                         const EvalT & pi_xx, const EvalT & pi_xy, const EvalT & pi_xz,
                                                         const EvalT & pi_yx, const EvalT & pi_yy, const EvalT & pi_yz,
                                                         const EvalT & pi_zx, const EvalT & pi_zy, const EvalT & pi_zz,
                                                         const EvalT & ndens, const EvalT & gamma_bar, const EvalT & Re,
                                                         const EvalT & S) {
  EvalT sres = ndens/gamma_bar*dT_dt + ndens/gamma_bar*(ux*dT_dx + uy*dT_dy + uz*dT_dz) + ndens*T*(dux_dx+duy_dy+duz_dz);
  // Not using q
  EvalT joule = 1.0/S*(jx*jx+jy*jy+jz*jz);
  EvalT vstress = 1.0/Re*(pi_xx*dux_dx + pi_xy*dux_dy + pi_xz*dux_dz + pi_yx*duy_dx + pi_yy*duy_dy + pi_yz*duy_dz + pi_zx*duz_dx + pi_zy*duz_dy + pi_zz*duz_dz);
  return sres-joule-vstress;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeStrongResidualBx(const EvalT & dBx_dt, const EvalT & Bx, const EvalT & dBx_dx,
                                                          const EvalT & dBx_dy, const EvalT & dBx_dz,
                                                          const EvalT & By, const EvalT & Bz,
                                                          const EvalT & dBy_dy, const EvalT & dBz_dz,
                                                          const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                          const EvalT & dux_dx, const EvalT & dux_dy, const EvalT & dux_dz,
                                                          const EvalT & duy_dy, const EvalT & duz_dz,
                                                          const EvalT & S, const EvalT & dpsi_dx) {
  EvalT sres = dBx_dt + dux_dx*Bx + ux*dBx_dx + dux_dy*By + ux*dBy_dy + dux_dz*Bz + ux*dBz_dz;
  sres += -1.0*(dBx_dx*ux + Bx*dux_dx + dBx_dy*uy + Bx*duy_dy + dBx_dz*uz + Bx*duz_dz) + dpsi_dx;
  // not using 1/S(nabla B - nabla B^T)
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeStrongResidualBy(const EvalT & dBy_dt, const EvalT & By, const EvalT & dBy_dx,
                                                          const EvalT & dBy_dy, const EvalT & dBy_dz,
                                                          const EvalT & Bx, const EvalT & Bz,
                                                          const EvalT & dBx_dx, const EvalT & dBz_dz,
                                                          const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                          const EvalT & dux_dx,
                                                          const EvalT & duy_dx, const EvalT & duy_dy, const EvalT & duy_dz,
                                                          const EvalT & duz_dz,
                                                          const EvalT & S, const EvalT & dpsi_dy) {
  EvalT sres = dBy_dt + duy_dx*Bx + uy*dBx_dx + duy_dy*By + uy*dBy_dy + duy_dz*Bz + uy*dBz_dz;
  sres += -1.0*(dBy_dx*ux + By*dux_dx + dBy_dy*uy + By*duy_dy + dBy_dz*uz + By*duz_dz) + dpsi_dy;
  // not using 1/S(nabla B - nabla B^T)
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeStrongResidualBz(const EvalT & dBz_dt, const EvalT & Bz, const EvalT & dBz_dx,
                                                          const EvalT & dBz_dy, const EvalT & dBz_dz,
                                                          const EvalT & Bx, const EvalT & By,
                                                          const EvalT & dBx_dx, const EvalT & dBy_dy,
                                                          const EvalT & ux, const EvalT & uy, const EvalT & uz,
                                                          const EvalT & dux_dx, const EvalT & duy_dy,
                                                          const EvalT & duz_dx, const EvalT & duz_dy, const EvalT & duz_dz,
                                                          const EvalT & S, const EvalT & dpsi_dz) {
  EvalT sres = dBz_dt + duz_dx*Bx + uz*dBx_dx + duz_dy*By + uz*dBy_dy + duz_dz*Bz + uz*dBz_dz;
  sres += -1.0*(dBz_dx*ux + Bz*dux_dx + dBz_dy*uy + Bz*duy_dy + dBz_dz*uz + Bz*duz_dz) + dpsi_dz;
  // not using 1/S(nabla B - nabla B^T)
  return sres;
}

template<class EvalT>
KOKKOS_FUNCTION EvalT MHD<EvalT>::computeStrongResidualPsi(const EvalT & dBx_dx, const EvalT & dBy_dy, const EvalT & dBz_dz) {
  EvalT sres = dBx_dx + dBy_dy + dBz_dz;
  return sres;
}
//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::MHD<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::MHD<AD>;

// Standard built-in types
template class MrHyDE::MHD<AD2>;
template class MrHyDE::MHD<AD4>;
template class MrHyDE::MHD<AD8>;
template class MrHyDE::MHD<AD16>;
template class MrHyDE::MHD<AD18>;
template class MrHyDE::MHD<AD24>;
template class MrHyDE::MHD<AD32>;
#endif
