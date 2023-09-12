/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "cns.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

//cns::cns(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_)
//  : physicsbase(settings, isaux_)
//{
//  
//  label = "cns";
//  int spaceDim = settings->sublist("Mesh").get<int>("dimension",2);
//
//  KEdef = "rhoux*rhoux/(rho*rho)";
//  
//  myvars.push_back("rho");
//  myvars.push_back("rhoux");
//  myvars.push_back("rhoE");
//  myvars.push_back("lambda"); 
//  if (spaceDim > 1) {
//    myvars.push_back("rhouy");
//    KEdef = KEdef + " + rhouy*rhouy/(rho*rho)";
//  }
//  if (spaceDim > 2) {
//    myvars.push_back("rhouz");
//    KEdef = KEdef + " + rhouz*rhouz/(rho*rho)";
//  }
//
//  // we take the state to be the vector of conserved quantities (\rho,\rho u_i, \rho E)
//  // where E is the total energy density per unit mass
//  
//  // TODO appropriate types?
// 
//  mybasistypes.push_back("HGRAD");
//  mybasistypes.push_back("HGRAD");
//  mybasistypes.push_back("HGRAD");
//  if (spaceDim > 1) {
//    mybasistypes.push_back("HGRAD");
//  }
//  if (spaceDim > 2) {
//    mybasistypes.push_back("HGRAD");
//  }
//  
//  // Params from input file
//  // TODO CHANGE??
//  useSUPG = settings.get<bool>("useSUPG",false);
//  usePSPG = settings.get<bool>("usePSPG",false);
//  useGRADDIV = settings.get<bool>("useGRADDIV",false);
//
//  // TODO :: Dimensionless params?
//  
//}
//
//// ========================================================================================
//// ========================================================================================
//
//void cns::defineFunctions(Teuchos::ParameterList & fs,
//                                   Teuchos::RCP<FunctionManager> & functionManager_) {
//  
//  functionManager = functionManager_;
//  
//  functionManager->addFunction("source ux",fs.get<string>("source ux","0.0"),"ip");
//  functionManager->addFunction("source uy",fs.get<string>("source uy","0.0"),"ip");
//  functionManager->addFunction("source uz",fs.get<string>("source uz","0.0"),"ip");
//  functionManager->addFunction("source E", fs.get<string>("source E", "0.0"),"ip");
//  // We default to properties of air at 293 K
//  // Dynamic viscosity  units are M/L-T
//  functionManager->addFunction("mu",fs.get<string>("mu","0.01178"),"ip");
//  // Thermal conductivity  units are M-L/T^3-K (K must be Kelvin)  //TODO CHECK
//  functionManager->addFunction("kappa",fs.get<string>("kappa","cp*mu/PrNum"),"ip"); 
//  // Thermodynamic pressure  units are M/L-T^2  Ideal gas law (non-dim)  // TODO CHECK
//  functionManager->addFunction("p0",fs.get<string>("p0","(gamma-1.)*(rhoE-.5*rho*KE)"),"ip");
//  // Temperature  units are K  (K must be Kelvin)
//  functionManager->addFunction("T",fs.get<string>("T","p0/(rho*RGas)"),"ip");
//  // Kinetic energy per unit mass  units are L^2/T^2
//  functionManager->addFunction("KE",KEdef,"ip"); // shouldn't ever be defined in input file
//  // Specific heat at constant pressure  units are L^2/T^2-K (K must be Kelvin) // TODO CHECK
//  functionManager->addFunction("cp",fs.get<string>("cp","1004.5"),"ip");
//  // Specific gas constant  units are J/kg-K
//  functionManager->addFunction("RGas",fs.get<string>("RGas","287.0"),"ip");
//  // Prandtl number
//  functionManager->addFunction("PrNum",fs.get<string>("PrNum","1.0"),"ip");
//
//}
//
//// ========================================================================================
//// ========================================================================================
//
//void cns::volumeResidual() {
//  
//  int spaceDim = wkset->dimension;
//  ScalarT dt = wkset->deltat;
//  bool isTransient = wkset->isTransient;
//  Vista source_ux, source_uy, source_uz, source_T;
//  Vista rho, mu, p0, lambda, cp;
//  
//  {
//    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
//    source_ux = functionManager->evaluate("source ux","ip");
//    source_T  = functionManager->evaluate("source T","ip");
//    if (spaceDim > 1) {
//      source_uy = functionManager->evaluate("source uy","ip");
//    }
//    if (spaceDim > 2) {
//      source_uz = functionManager->evaluate("source uz","ip");
//    }
//
//    // Update thermodynamic and transport properties
//    rho = functionManager->evaluate("rho","ip");
//    mu = functionManager->evaluate("mu","ip");
//    p0 = functionManager->evaluate("p0","ip");
//    lambda = functionManager->evaluate("lambda","ip");
//    cp = functionManager->evaluate("cp","ip"); 
//  }
//  
//  Teuchos::TimeMonitor resideval(*volumeResidualFill);
//  auto wts = wkset->wts;
//  auto res = wkset->res;
//
//  if (spaceDim == 1) {
//    {
//      int rhoux_basis = wkset->usebasis[rhoux_num];
//      auto basis = wkset->basis[rhoux_basis];
//      auto basis_grad = wkset->basis_grad[rhoux_basis];
//      auto rhoux = wkset->getSolutionField("rhoux");
//      auto drhoux_dt = wkset->getSolutionField("rhoux_t");
//      auto drhoux_dx = wkset->getSolutionField("grad(rhoux)[x]");
//      auto off = subview(wkset->offsets,ux_num,ALL());
//      
//      // Ux equation
//      // (v_1,rho du_1/dt) + (v_1,rho u_1 du_1/dx_1) - (dv_1/dx_1,p)
//      // + (dv_1/dx_1, 4/3 mu du_1/dx_1) - (v_1,source)
//      parallel_for("cns ux volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//        // TODO here and everywhere, WE NEED TO swing rho over to other side and use MU
//        // ALSO --> should thermal divergence be abstracted? i.e. it will depend on EOS
//          AD Fx = 4./3.*mu(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
//          Fx *= wts(elem,pt);
//          // TODO changing how source shows up, different from NS module
//          AD F = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt)) - source_ux(elem,pt);
//          F *= wts(elem,pt);
//          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + F*basis(elem,dof,pt,0);
//          }
//        }
//      });
//      
//      // SUPG contribution
//      // TODO viscous contribution for higher order elements?
//      // (rho u_1 dv_1/dx_1, \tau_mom R_mom,1)
//      // 1/\tau_mom^2 = (c1 \mu/h)^2 + (c2 |\rho u|/h)^2 + (c3 \rho/dt)^2
//      if (useSUPG) {
//        auto h = wkset->h;
//        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
//        parallel_for("cns ux volume resid SUPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),0.0,0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // TODO NO VISCOUS TERM
//            // TODO CHECK THIS units, etc.
//            AD strongres = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
//            AD Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0); 
//            }
//          }
//        });
//      }
//
//      // GRADDIV contribution
//      // TODO p0 contribution
//      // (dv_1/dx_1, \tau_mass R_mass) 
//      // \tau_mass is like h^2/\tau_mom
//      if (useGRADDIV) {
//        auto h = wkset->h;
//        auto T = wkset->getSolutionField("T");
//        auto dT_dt = wkset->getSolutionField("T_t");
//        auto dT_dx = wkset->getSolutionField("grad(T)[x]");
//        parallel_for("cns ux volume resid GRADDIV",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),0.0,0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // TODO FIX TAU??? the constant at least is wrong
//            tau = h(elem)*h(elem)/tau;
//            AD thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt))*wts(elem,pt);
//            AD strongres = dux_dx(elem,pt) - thermDiv;
//            AD S = tau*strongres*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += S*basis_grad(elem,dof,pt,0); 
//            }
//          }
//        });
//      }
//    }
//
//    {
//      // Energy equation // TODO this is different need offset, etc.
//      // (w,rho dT/dt) + (w,rho u_1 dT/dx_1) + (dw/dx_1,lambda/cp dT/dx_1) - (w, 1/cp[dp0/dt + Q])
//      int T_basis = wkset->usebasis[T_num];
//      auto basis = wkset->basis[T_basis];
//      auto basis_grad = wkset->basis_grad[T_basis];
//      auto T = wkset->getSolutionField("T");
//      auto dT_dt = wkset->getSolutionField("T_t");
//      auto dT_dx = wkset->getSolutionField("grad(T)[x]"); 
//      auto ux = wkset->getSolutionField("ux");
//      auto off = subview(wkset->offsets,T_num,ALL());
//
//      parallel_for("cns T volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD F = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt))*wts(elem,pt);
//          // TODO SOURCE AND DPDT TERM
//          F -= source_T(elem,pt)/cp(elem,pt)*wts(elem,pt);
//          AD Fx = lambda(elem,pt)/cp(elem,pt)*dT_dx(elem,pt)*wts(elem,pt);
//          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += F*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
//          }
//        }
//      });
//
//      // SUPG contribution
//      // TODO viscous contribution for higher order elements?
//      // TODO SOURCE AND DPDT TERM
//      // (rho u_1 dw/dx_1, \tau_T R_T)
//      // 1/\tau_T^2 = (c1 cp/lambda*h)^2 + (c2 |\rho u|/h)^2 + (c3 \rho/dt)^2
//      if (useSUPG) {
//        auto h = wkset->h;
//
//        parallel_for("cns T volume resid SUPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(lambda(elem,pt)/cp(elem,pt),ux(elem,pt),0.0,0.0,
//                rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // TODO CHECK THIS, UNITS ETC.
//            AD strongres = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt)) - source_T(elem,pt)/cp(elem,pt);
//            AD Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
//            }
//          }
//        });
//      }
//    }
//    
//    {
//      /////////////////////////////
//      // pressure equation
//      /////////////////////////////
//      // (q,du_1/dx_1) - (q,1/T(dT/dt + u_1 dT/dx_1) - 1/p0 dp0/dt)
//      
//      int pr_basis = wkset->usebasis[pr_num];
//      auto basis = wkset->basis[pr_basis];
//      auto basis_grad = wkset->basis_grad[pr_basis];
//      auto ux = wkset->getSolutionField("ux");
//      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
//      auto T = wkset->getSolutionField("T");
//      auto dT_dt = wkset->getSolutionField("T_t");
//      auto dT_dx = wkset->getSolutionField("grad(T)[x]");
//      auto off = subview(wkset->offsets,pr_num,ALL());
//      
//      parallel_for("cns pr volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD divu = dux_dx(elem,pt)*wts(elem,pt);
//          // TODO :: p0 part DONT SCREW UP WTS
//          AD thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt))*wts(elem,pt);
//          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += (divu-thermDiv)*basis(elem,dof,pt,0);
//          }
//        }
//      });
//
//      // TODO BWR -- viscous contribution 
//      // PSPG contribution
//      // (dq/dx_1, \tau_mom R_mom,1)
//      if (usePSPG) {
//        
//        auto h = wkset->h;
//        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
//        auto ux = wkset->getSolutionField("ux");
//        auto dux_dt = wkset->getSolutionField("ux_t");
//        
//        parallel_for("cns pr volume resid PSPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),0.0,0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // TODO NO VISCOUS TERM
//            // TODO CHECK THIS units, etc.
//            AD strongres = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
//            AD Sx = tau*strongres*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0); 
//            }
//          }
//        });
//      }
//    }
//  }
//  else if (spaceDim == 2) {
//    {
//      int ux_basis = wkset->usebasis[ux_num];
//      auto basis = wkset->basis[ux_basis];
//      auto basis_grad = wkset->basis_grad[ux_basis];
//      auto ux = wkset->getSolutionField("ux");
//      auto uy = wkset->getSolutionField("uy");
//      auto dux_dt = wkset->getSolutionField("ux_t");
//      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
//      auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
//      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
//      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
//      auto pr = wkset->getSolutionField("pr");
//      auto off = subview(wkset->offsets,ux_num,ALL());
//      
//      // Ux equation
//      // (v_1,rho du_1/dt) + (v_1,rho [u_1 du_1/dx_1 + u_2 du_1/dx_2]) - (dv_1/dx_1,p)  
//      // + (dv_1/dx_1, \mu [2 * du_1/dx_1 - 2/3 (du_1/dx_1 + du_2/dx_2)]) 
//      // + (dv_1/dx_2, \mu [du_1/dx_2 + du_2/dx_1]) - (v_1,source) 
//      parallel_for("cns ux volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD Fx = mu(elem,pt)*(2.*dux_dx(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt))) - pr(elem,pt);
//          //AD Fx = mu(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
//          Fx *= wts(elem,pt);
//          AD Fy = mu(elem,pt)*(dux_dy(elem,pt) + duy_dx(elem,pt));
//          //AD Fy = mu(elem,pt)*dux_dy(elem,pt);
//          Fy *= wts(elem,pt);
//          AD F = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) - source_ux(elem,pt);
//          F *= wts(elem,pt);
//          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt,0);
//          }
//        }
//      });
//      
//      // SUPG contribution
//      // TODO viscous contribution for higher order elements?
//      // (rho [u_1 dv_1/dx_1 + u_2 dv_1/dx_2], \tau_mom R_mom,1)
//      
//      if (useSUPG) {
//        auto h = wkset->h;
//        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
//        parallel_for("cns ux volume resid SUPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            AD strongres = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
//            AD Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
//            AD Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
//            }
//          }
//
//        });
//      }
//
//      // GRADDIV contribution
//      // TODO p0 contribution
//      // (dv_1/dx_1, \tau_mass R_mass) 
//      // \tau_mass is like h^2/\tau_mom
//      if (useGRADDIV) {
//        auto h = wkset->h;
//        auto T = wkset->getSolutionField("T");
//        auto dT_dt = wkset->getSolutionField("T_t");
//        auto dT_dx = wkset->getSolutionField("grad(T)[x]");
//        auto dT_dy = wkset->getSolutionField("grad(T)[y]");
//        parallel_for("cns ux volume resid GRADDIV",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // TODO FIX TAU??? the constant at least is wrong
//            tau = h(elem)*h(elem)/tau;
//            AD thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt))*wts(elem,pt);
//            AD strongres = (dux_dx(elem,pt) + duy_dx(elem,pt)) - thermDiv;
//            AD S = tau*strongres*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += S*basis_grad(elem,dof,pt,0); 
//            }
//          }
//        });
//      }
//    }
//    
//    {
//      // Uy equation
//      // (v_2, rho du_2/dt) + (v_2, rho [u_1 du_2/dx_1 + u_2 du_2/dx_2]) - (dv_2/dx_2,p)
//      // + (dv_2/dx_1, \mu [du_1/dx_2 + du_2/dx_1]) 
//      // + (dv_2/dx_2, \mu [2 * du_2/dx_2 - 2/3 (du_1/dx_1 + du_2/dx_2)]) - (v_2,source)
//      int uy_basis = wkset->usebasis[uy_num];
//      auto basis = wkset->basis[uy_basis];
//      auto basis_grad = wkset->basis_grad[uy_basis];
//      auto ux = wkset->getSolutionField("ux");
//      auto uy = wkset->getSolutionField("uy");
//      auto duy_dt = wkset->getSolutionField("uy_t");
//      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
//      auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
//      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
//      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
//      auto pr = wkset->getSolutionField("pr");
//      auto off = subview(wkset->offsets,uy_num,ALL());
//      
//      parallel_for("cns uy volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD Fx = mu(elem,pt)*(dux_dy(elem,pt) + duy_dx(elem,pt));
//          //AD Fx = mu(elem,pt)*duy_dx(elem,pt);
//          Fx *= wts(elem,pt);
//          AD Fy = mu(elem,pt)*(2.*duy_dy(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt))) - pr(elem,pt);
//          //AD Fy = mu(elem,pt)*duy_dy(elem,pt) - pr(elem,pt);
//          Fy *= wts(elem,pt);
//          AD F = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) - source_uy(elem,pt);
//          F *= wts(elem,pt);
//          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt,0);
//          }
//        }
//      });
//      
//      // SUPG contribution
//      // TODO viscous contribution for higher order elements?
//      // (rho [u_1 dv_2/dx_1 + u_2 dv_2/dx_2], \tau_mom R_mom,2)
//      // TODO CHECK UNITS HERE
//      
//      if (useSUPG) {
//        auto h = wkset->h;
//        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
//        parallel_for("cns uy volume resid SUPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            AD strongres = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) + dpr_dy(elem,pt) - source_uy(elem,pt);
//            AD Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
//            AD Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
//            }
//          }
//        });
//      }
//
//      // GRADDIV contribution
//      // TODO p0 contribution
//      // (dv_2/dx_2, \tau_mass R_mass) 
//      // \tau_mass is like h^2/\tau_mom
//      if (useGRADDIV) {
//        auto h = wkset->h;
//        auto T = wkset->getSolutionField("T");
//        auto dT_dt = wkset->getSolutionField("T_t");
//        auto dT_dx = wkset->getSolutionField("grad(T)[x]");
//        auto dT_dy = wkset->getSolutionField("grad(T)[y]");
//        parallel_for("cns ux volume resid GRADDIV",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // TODO FIX TAU??? the constant at least is wrong
//            tau = h(elem)*h(elem)/tau;
//            AD thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt))*wts(elem,pt);
//            AD strongres = (dux_dx(elem,pt) + duy_dx(elem,pt)) - thermDiv;
//            AD S = tau*strongres*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += S*basis_grad(elem,dof,pt,1); 
//            }
//          }
//        });
//      }
//    }
//
//    {
//      /////////////////////////////
//      // energy equation
//      /////////////////////////////
//      // TODO dp0 part, etc
//      // (w,rho dT/dt) + (w,rho [u_1 dT/dx_1 + u_2 dT/dx_2]) + (dw/dx_1,lambda/cp dT/dx_1)
//      // + (dw/dx_2,lambda/cp dT/dx_2) - (w,1/cp[dp0/dt + Q])
//      int T_basis = wkset->usebasis[T_num];
//      auto basis = wkset->basis[T_basis];
//      auto basis_grad = wkset->basis_grad[T_basis];
//      auto T = wkset->getSolutionField("T");
//      auto dT_dt = wkset->getSolutionField("T_t");
//      auto dT_dx = wkset->getSolutionField("grad(T)[x]"); 
//      auto dT_dy = wkset->getSolutionField("grad(T)[y]"); 
//      auto ux = wkset->getSolutionField("ux");
//      auto uy = wkset->getSolutionField("uy");
//      auto off = subview(wkset->offsets,T_num,ALL());
// 
//      parallel_for("cns T volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD F = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt))*wts(elem,pt);
//          F -= source_T(elem,pt)/cp(elem,pt)*wts(elem,pt);
//          // TODO SOURCE AND DPDT TERM
//          AD Fx = lambda(elem,pt)/cp(elem,pt)*dT_dx(elem,pt)*wts(elem,pt);
//          AD Fy = lambda(elem,pt)/cp(elem,pt)*dT_dy(elem,pt)*wts(elem,pt);
//          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += F*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
//          }
//        }
//      });
//
//      // SUPG contribution
//      // TODO viscous contribution for higher order elements?
//      // TODO SOURCE AND DPDT TERM
//      // (rho [u_1 dw/dx_1 + u_2 dw/dx_2], \tau_T R_T)
//      // 1/\tau_T^2 = (c1 cp/lambda*h)^2 + (c2 |\rho u|/h)^2 + (c3 \rho/dt)^2
//      if (useSUPG) {
//        auto h = wkset->h;
//
//        parallel_for("cns T volume resid SUPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(lambda(elem,pt)/cp(elem,pt),ux(elem,pt),uy(elem,pt),0.0,
//                rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // TODO CHECK THIS, UNITS ETC.
//            AD strongres = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) 
//                + uy(elem,pt)*dT_dy(elem,pt)) - source_T(elem,pt)/cp(elem,pt);
//            AD Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
//            AD Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
//            }
//          }
//        });
//      }
//
//    }
//    
//    {
//      /////////////////////////////
//      // pressure equation
//      /////////////////////////////
//      // (q,du_1/dx_1 + du_2/dx_2) - (q,1/T(dT/dt + u_1 dT/dx_1 + u_2 dT/dx_2) - 1/p0 dp0/dt)
//      int pr_basis = wkset->usebasis[pr_num];
//      auto basis = wkset->basis[pr_basis];
//      auto basis_grad = wkset->basis_grad[pr_basis];
//      auto ux = wkset->getSolutionField("ux");
//      auto uy = wkset->getSolutionField("uy");
//      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
//      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
//      auto T = wkset->getSolutionField("T");
//      auto dT_dt = wkset->getSolutionField("T_t");
//      auto dT_dx = wkset->getSolutionField("grad(T)[x]");
//      auto dT_dy = wkset->getSolutionField("grad(T)[y]");
//      auto off = subview(wkset->offsets,pr_num,ALL());
//      
//      parallel_for("cns pr volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD divu = (dux_dx(elem,pt) + duy_dy(elem,pt))*wts(elem,pt);
//          // TODO :: p0 part DONT SCREW UP WTS
//          // TODO forcing this to zero for now...
//          AD ovT = 1./T(elem,pt);
//          //if (T(elem,pt) <= 1e-12) std::cout << "OH NO" << std::endl; //ovT = 1e-12;
//          AD thermDiv = ovT*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt));
//          thermDiv *= wts(elem,pt);
//          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += (divu-thermDiv)*basis(elem,dof,pt,0);
//          }
//        }
//      });
//      
//      // TODO BWR -- viscous contribution 
//      // PSPG contribution
//      // (dq/dx_1, \tau_mom R_mom,1) + (dq/dx_2, \tau_mom R_mom,2)
//      if (usePSPG) {
//        
//        auto h = wkset->h;
//        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
//        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
//        auto dux_dt = wkset->getSolutionField("ux_t");
//        auto duy_dt = wkset->getSolutionField("uy_t");
//        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
//        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
//
//        parallel_for("cns pr volume resid PSPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // Strong residual x momentum
//            AD Sx = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
//            Sx *= tau*wts(elem,pt);
//            // Strong residual y momentum
//            AD Sy = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) + dpr_dy(elem,pt) - source_uy(elem,pt);
//            Sy *= tau*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
//            }
//          }
//        });
//      }
//    }
//  }
//  else if (spaceDim == 3) {
//    {
//      // Ux equation
//      // (v_1,rho du_1/dt) + (v_1, rho [u_1 du_1/dx_1 + u_2 du_1/dx_2 + u_3 du_1/dx_3]) - (dv_1/dx_1,p)
//      // + (dv_1/dx_1, \mu [2 * du_1/dx_1 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3)])
//      // + (dv_1/dx_2, \mu [du_1/dx_2 + du_2/dx_1]) + (dv_1/dx_3, \mu [du_1/dx_3 + du_3/dx_1])
//      // - (v_1,source)
//      int ux_basis = wkset->usebasis[ux_num];
//      auto basis = wkset->basis[ux_basis];
//      auto basis_grad = wkset->basis_grad[ux_basis];
//      auto ux = wkset->getSolutionField("ux");
//      auto uy = wkset->getSolutionField("uy");
//      auto uz = wkset->getSolutionField("uz");
//      auto dux_dt = wkset->getSolutionField("ux_t");
//      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
//      auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
//      auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
//      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
//      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
//      auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
//      auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
//      auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
//      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
//      auto pr = wkset->getSolutionField("pr");
//      auto off = subview(wkset->offsets,ux_num,ALL());
//      
//      parallel_for("cns ux volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD Fx = mu(elem,pt)*(2.*dux_dx(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt))) - pr(elem,pt);
//          Fx *= wts(elem,pt);
//          AD Fy = mu(elem,pt)*(dux_dy(elem,pt) + duy_dx(elem,pt));
//          Fy *= wts(elem,pt);
//          AD Fz = mu(elem,pt)*(dux_dz(elem,pt) + duz_dx(elem,pt));
//          Fz *= wts(elem,pt);
//          AD F = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) - source_ux(elem,pt);
//          F *= wts(elem,pt);
//          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
//          }
//        }
//      });
//
//      // SUPG contribution
//      // TODO viscous contribution for higher order elements?
//      // (rho [u_1 dv_1/dx_1 + u_2 dv_1/dx_2 + u_3 dv_1/dx_3], \tau_mom R_mom,1)
//      
//      if (useSUPG) {
//        auto h = wkset->h;
//        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
//        parallel_for("cns ux volume resid SUPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            AD strongres = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
//            AD Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
//            AD Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
//            AD Sz = tau*strongres*rho(elem,pt)*uz(elem,pt)*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
//            }
//          }
//        });
//      }
//    }
//
//    {
//      // Uy equation
//      // (v_2, rho du_2/dt) + (v_2, rho [u_1 du_2/dx_1 + u_2 du_2/dx_2 + u_3 du_2/dx_3]) - (dv_2/dx_2,p)
//      // + (dv_2/dx_1, \mu [du_1/dx_2 + du_2/dx_1]) 
//      // + (dv_2/dx_2, \mu [2 * du_2/dx_2 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3)]) 
//      // + (dv_2/dx_3, \mu [du_2/dx_3 + du_3/dx_2]) - (v_2,source)
//      int uy_basis = wkset->usebasis[uy_num];
//      auto basis = wkset->basis[uy_basis];
//      auto basis_grad = wkset->basis_grad[uy_basis];
//      auto ux = wkset->getSolutionField("ux");
//      auto uy = wkset->getSolutionField("uy");
//      auto uz = wkset->getSolutionField("uz");
//      auto duy_dt = wkset->getSolutionField("uy_t");
//      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
//      auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
//      auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
//      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
//      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
//      auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
//      auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
//      auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
//      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
//      auto pr = wkset->getSolutionField("pr");
//      auto off = subview(wkset->offsets,uy_num,ALL());
//      
//      parallel_for("cns uy volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD Fx = mu(elem,pt)*(dux_dy(elem,pt) + duy_dx(elem,pt));
//          Fx *= wts(elem,pt);
//          AD Fy = mu(elem,pt)*(2.*duy_dy(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt))) - pr(elem,pt);
//          Fy *= wts(elem,pt);
//          AD Fz = mu(elem,pt)*(duy_dz(elem,pt) + duz_dy(elem,pt));
//          Fz *= wts(elem,pt);
//          AD F = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) - source_uy(elem,pt);
//          F *= wts(elem,pt);
//          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
//          }
//        }
//      });
//      
//      // SUPG contribution
//      // TODO viscous contribution for higher order elements?
//      // (rho [u_1 dv_2/dx_1 + u_2 dv_2/dx_2 + u_3 dv_2/dx_3], \tau_mom R_mom,2)
//      // TODO CHECK UNITS HERE
//      
//      if (useSUPG) {
//        auto h = wkset->h;
//        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
//        parallel_for("cns uy volume resid SUPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            AD strongres = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) + dpr_dy(elem,pt) - source_uy(elem,pt);
//            AD Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
//            AD Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
//            AD Sz = tau*strongres*rho(elem,pt)*uz(elem,pt)*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
//            }
//          }
//        });
//      }
//    }
//
//    {
//      // Uz equation
//      // (v_3,rho du_3/dt) + (v_3, rho [u_1 du_3/dx_1 + u_2 du_3/dx_2 + u_3 du_3/dx_3]) - (dv_3/dx_3,p)
//      // + (dv_3/dx_1, \mu [du_3/dx_1 + du_1/dx_3]) + (dv_3/dx_2, \mu [du_3/dx_2 + du_2/dx_3])
//      // + (dv_3/dx_3, \mu [2 * du_3/dx_3 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3])) 
//      // - (v_3,source)
//      int uz_basis = wkset->usebasis[uz_num];
//      auto basis = wkset->basis[uz_basis];
//      auto basis_grad = wkset->basis_grad[uz_basis];
//      auto ux = wkset->getSolutionField("ux");
//      auto uy = wkset->getSolutionField("uy");
//      auto uz = wkset->getSolutionField("uz");
//      auto duz_dt = wkset->getSolutionField("uz_t");
//      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
//      auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
//      auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
//      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
//      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
//      auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
//      auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
//      auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
//      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
//      auto pr = wkset->getSolutionField("pr");
//      auto off = subview(wkset->offsets,uz_num,ALL());
//      
//      parallel_for("cns uz volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD Fx = mu(elem,pt)*(duz_dx(elem,pt) + dux_dz(elem,pt));
//          Fx *= wts(elem,pt);
//          AD Fy = mu(elem,pt)*(duz_dy(elem,pt) + duy_dz(elem,pt));
//          Fy *= wts(elem,pt);
//          AD Fz = mu(elem,pt)*(2.*duz_dz(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt))) - pr(elem,pt);
//          Fz *= wts(elem,pt);
//          AD F = rho(elem,pt)*(duz_dt(elem,pt) + ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) - source_uz(elem,pt);
//          F *= wts(elem,pt);
//          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
//          }
//        }
//      });
//
//      // SUPG contribution
//      // TODO viscous contribution for higher order elements?
//      // (rho [u_1 dv_3/dx_1 + u_2 dv_3/dx_2 + u_3 dv_3/dx_3], \tau_mom R_mom,3)
//      // TODO CHECK UNITS HERE
//      
//      if (useSUPG) {
//        auto h = wkset->h;
//        auto dpr_dz = wkset->getSolutionField("grad(pr)[z]");
//        parallel_for("cns uz volume resid SUPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            AD strongres = rho(elem,pt)*(duz_dt(elem,pt) + ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) + dpr_dz(elem,pt) - source_uz(elem,pt);
//            AD Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
//            AD Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
//            AD Sz = tau*strongres*rho(elem,pt)*uz(elem,pt)*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
//            }
//          }
//        });
//      }
//    }
//
//    {
//      /////////////////////////////
//      // energy equation
//      /////////////////////////////
//      // TODO dp0 part, etc
//      // (w,rho dT/dt) + (w,rho [u_1 dT/dx_1 + u_2 dT/dx_2 + u_3 dT/dx_3]) + (dw/dx_1,lambda/cp dT/dx_1)
//      // + (dw/dx_2,lambda/cp dT/dx_2) + (dw/dx_3,lambda/cp dT/dx_3) - (w,1/cp[dp0/dt + Q])
//      int T_basis = wkset->usebasis[T_num];
//      auto basis = wkset->basis[T_basis];
//      auto basis_grad = wkset->basis_grad[T_basis];
//      auto T = wkset->getSolutionField("T");
//      auto dT_dt = wkset->getSolutionField("T_t");
//      auto dT_dx = wkset->getSolutionField("grad(T)[x]"); 
//      auto dT_dy = wkset->getSolutionField("grad(T)[y]"); 
//      auto dT_dz = wkset->getSolutionField("grad(T)[z]"); 
//      auto ux = wkset->getSolutionField("ux");
//      auto uy = wkset->getSolutionField("uy");
//      auto uz = wkset->getSolutionField("uz");
//      auto off = subview(wkset->offsets,T_num,ALL());
// 
//      parallel_for("cns T volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD F = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt) + uz(elem,pt)*dT_dz(elem,pt))*wts(elem,pt);
//          F -= source_T(elem,pt)/cp(elem,pt)*wts(elem,pt);
//          // TODO SOURCE AND DPDT TERM
//          AD Fx = lambda(elem,pt)/cp(elem,pt)*dT_dx(elem,pt)*wts(elem,pt);
//          AD Fy = lambda(elem,pt)/cp(elem,pt)*dT_dy(elem,pt)*wts(elem,pt);
//          AD Fz = lambda(elem,pt)/cp(elem,pt)*dT_dz(elem,pt)*wts(elem,pt);
//          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += F*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
//          }
//        }
//      });
//
//      // SUPG contribution
//      // TODO viscous contribution for higher order elements?
//      // TODO SOURCE AND DPDT TERM
//      // (rho [u_1 dw/dx_1 + u_2 dw/dx_2 + u_3 dw/dx_3], \tau_T R_T)
//      // 1/\tau_T^2 = (c1 cp/lambda*h)^2 + (c2 |\rho u|/h)^2 + (c3 \rho/dt)^2
//      if (useSUPG) {
//        auto h = wkset->h;
//
//        parallel_for("cns T volume resid SUPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(lambda(elem,pt)/cp(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),
//                rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // TODO CHECK THIS, UNITS ETC.
//            AD strongres = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) 
//                + uy(elem,pt)*dT_dy(elem,pt) + uz(elem,pt)*dT_dz(elem,pt)) - source_T(elem,pt)/cp(elem,pt);
//            AD Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
//            AD Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
//            AD Sz = tau*strongres*rho(elem,pt)*uz(elem,pt)*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
//            }
//          }
//        });
//      }
//    }
//
//    {
//      /////////////////////////////
//      // pressure equation
//      /////////////////////////////
//      // (q,du_1/dx_1 + du_2/dx_2 + du_3/dx_3) - (q,1/T(dT/dt + u_1 dT/dx_1 + u_2 dT/dx_2 + u_3 dT/dx_3) - 1/p0 dp0/dt)
//      int pr_basis = wkset->usebasis[pr_num];
//      auto basis = wkset->basis[pr_basis];
//      auto basis_grad = wkset->basis_grad[pr_basis];
//      auto ux = wkset->getSolutionField("ux");
//      auto uy = wkset->getSolutionField("uy");
//      auto uz = wkset->getSolutionField("uz");
//      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
//      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
//      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
//      auto T = wkset->getSolutionField("T");
//      auto dT_dt = wkset->getSolutionField("T_t");
//      auto dT_dx = wkset->getSolutionField("grad(T)[x]");
//      auto dT_dy = wkset->getSolutionField("grad(T)[y]");
//      auto dT_dz = wkset->getSolutionField("grad(T)[z]");
//      auto off = subview(wkset->offsets,pr_num,ALL());
//      
//      parallel_for("cns pr volume resid",
//                   RangePolicy<AssemblyExec>(0,wkset->numElem),
//                   KOKKOS_LAMBDA (const int elem ) {
//        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//          AD divu = (dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt))*wts(elem,pt);
//          // TODO :: p0 part DONT SCREW UP WTS
//          AD thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt) + uz(elem,pt)*dT_dz(elem,pt));
//          thermDiv *= wts(elem,pt);
//          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//            res(elem,off(dof)) += (divu-thermDiv)*basis(elem,dof,pt,0);
//          }
//        }
//      });
//      
//      // TODO BWR -- viscous contribution 
//      // PSPG contribution
//      // (dq/dx_1, \tau_mom R_mom,1) + (dq/dx_2, \tau_mom R_mom,2) + (dq/dx_3, \tau_mom R_mom,3)
//      if (usePSPG) {
//        
//        auto h = wkset->h;
//        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
//        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
//        auto dpr_dz = wkset->getSolutionField("grad(pr)[z]");
//        auto dux_dt = wkset->getSolutionField("ux_t");
//        auto duy_dt = wkset->getSolutionField("uy_t");
//        auto duz_dt = wkset->getSolutionField("uz_t");
//        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
//        auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
//        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
//        auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
//        auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
//        auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
//        
//        parallel_for("NS pr volume resid PSPG",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            AD tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),rho(elem,pt),h(elem),spaceDim,dt,isTransient);
//            // Strong residual x momentum
//            AD Sx = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
//            Sx *= tau*wts(elem,pt);
//            // Strong residual y momentum
//            AD Sy = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) + dpr_dy(elem,pt) - source_uy(elem,pt);
//            Sy *= tau*wts(elem,pt);
//            // Strong residual z momentum
//            AD Sz = rho(elem,pt)*(duz_dt(elem,pt) + ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) + dpr_dz(elem,pt) - source_uz(elem,pt);
//            Sz *= tau*wts(elem,pt);
//            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
//            }
//          }
//        });
//      }
//    }
//  }
//}
//
//// ========================================================================================
//// ========================================================================================
//
//void cns::boundaryResidual() {
//  
//  int spaceDim = wkset->dimension;
//  auto bcs = wkset->var_bcs;
//
//  int cside = wkset->currentside;
//
//  Vista neusource_ux, neusource_T, neusource_uy, neusource_uz;
//
//  string ux_sidetype = bcs(ux_num,cside);
//  string T_sidetype = bcs(T_num,cside);
//  string uy_sidetype = ""; string uz_sidetype = "";
//  if (spaceDim > 1) {
//    uy_sidetype = bcs(uy_num,cside);
//  }
//  if (spaceDim > 2) {
//    uz_sidetype = bcs(uz_num,cside);
//  }
//
//  {
//    Teuchos::TimeMonitor funceval(*boundaryResidualFunc);
//
//    // evaluate Neumann or traction sources if necessary
//    // For momentum equations, the source should be 
//    // -p n_\alpha + [du_\alpha/dx_j n_j + du_j/dx_\alpha n_j] -- a traction BC
//    //
//    // For energy equation, the source should be 
//    // \lambda/cp dT/dx_j n_j 
//
//    if (ux_sidetype == "Neumann") {
//      neusource_ux = functionManager->evaluate("Neumann ux " + wkset->sidename,"side ip");
//    }
//    if (T_sidetype == "Neumann") {
//      neusource_T = functionManager->evaluate( "Neumann T "  + wkset->sidename,"side ip");
//    }
//    if (uy_sidetype == "Neumann") {
//      neusource_uy = functionManager->evaluate("Neumann uy " + wkset->sidename,"side ip");
//    }
//    if (uz_sidetype == "Neumann") {
//      neusource_uz = functionManager->evaluate("Neumann uz " + wkset->sidename,"side ip");
//    }
//  }
//
//  auto wts = wkset->wts_side;
//  auto h = wkset->h;
//  auto res = wkset->res;
//
//  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
//
//  // TODO Right now, if we are just reading sources in, things can be lumped together
//  // but to allow for flexibility later on, I'll just go with splitting things up
//
//  if (spaceDim == 1) {
//    {
//      // Ux equation
//      // -(v_1,-p n_1 + \mu [4/3*du_1/dx_1 n_1])
//      int ux_basis = wkset->usebasis[ux_num];
//      auto basis = wkset->basis_side[ux_basis];
//      auto off = subview(wkset->offsets,ux_num,ALL());
//      
//      if (ux_sidetype == "Neumann") {
//        parallel_for("cns ux boundary resid",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) -= neusource_ux(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
//            }
//          }
//        });
//      } 
//    }
//
//    {
//      // Energy equation
//      // -(w,lambda/cp dT/dx_1 n_1)
//      int T_basis = wkset->usebasis[T_num];
//      auto basis = wkset->basis_side[T_basis];
//      auto off = subview(wkset->offsets,T_num,ALL());
//      
//      if (T_sidetype == "Neumann") {
//        parallel_for("cns T boundary resid",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) -= neusource_T(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
//            }
//          }
//        });
//      } 
//    }
//  }
//  else if (spaceDim == 2) {
//    {
//      // Ux equation
//      // -(v_1,-p n_1 + \mu [2 * du_1/dx_1 n_1 + du_1/dx_2 n_2 + du_2/dx_1 n_2 
//      //                                     - 2/3 (du_1/dx_1 + du_2/dx_2) n_1])
//      int ux_basis = wkset->usebasis[ux_num];
//      auto basis = wkset->basis_side[ux_basis];
//      auto off = subview(wkset->offsets,ux_num,ALL());
//      
//      if (ux_sidetype == "Neumann") {
//        parallel_for("cns ux boundary resid",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) -= neusource_ux(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
//            }
//          }
//        });
//      } 
//    }
//
//    {
//      // Uy equation
//      // -(v_2,-p n_2 + \mu [2 * du_2/dx_2 n_2 + du_1/dx_2 n_1 + du_2/dx_1 n_1 
//      //                                     - 2/3 (du_1/dx_1 + du_2/dx_2) n_2])
//      int uy_basis = wkset->usebasis[uy_num];
//      auto basis = wkset->basis_side[uy_basis];
//      auto off = subview(wkset->offsets,uy_num,ALL());
//      
//      if (uy_sidetype == "Neumann") {
//        parallel_for("cns uy boundary resid",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) -= neusource_uy(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
//            }
//          }
//        });
//      } 
//    }
//
//    {
//      // Energy equation
//      // -(w,lambda/cp [dT/dx_1 n_1 + dT/dx_2 n_2])
//      int T_basis = wkset->usebasis[T_num];
//      auto basis = wkset->basis_side[T_basis];
//      auto off = subview(wkset->offsets,T_num,ALL());
//      
//      if (T_sidetype == "Neumann") {
//        parallel_for("cns T boundary resid",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) -= neusource_T(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
//            }
//          }
//        });
//      } 
//    }
//  }
//  else if (spaceDim == 3) {
//    {
//      // Ux equation
//      // -(v_1,-p n_1 + \mu [2 * du_1/dx_1 n_1 + du_1/dx_2 n_2 + du_2/dx_1 n_2 
//      //    + du_1/dx_3 n_3 + du_3/dx_1 n_3 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3) n_1])
//      int ux_basis = wkset->usebasis[ux_num];
//      auto basis = wkset->basis_side[ux_basis];
//      auto off = subview(wkset->offsets,ux_num,ALL());
//      
//      if (ux_sidetype == "Neumann") {
//        parallel_for("cns ux boundary resid",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) -= neusource_ux(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
//            }
//          }
//        });
//      } 
//    }
//
//    {
//      // Uy equation
//      // -(v_2,-p n_2 + \mu [2 * du_2/dx_2 n_2 + du_1/dx_2 n_1 + du_2/dx_1 n_1 
//      //    + du_2/dx_3 n_3 + du_3/dx_2 n_3 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3) n_2])
//
//      int uy_basis = wkset->usebasis[uy_num];
//      auto basis = wkset->basis_side[uy_basis];
//      auto off = subview(wkset->offsets,uy_num,ALL());
//      
//      if (uy_sidetype == "Neumann") {
//        parallel_for("cns uy boundary resid",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) -= neusource_uy(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
//            }
//          }
//        });
//      } 
//    }
//
//    {
//      // Uz equation
//      // -(v_3,-p n_3 + \mu [2 * du_3/dx_3 n_3 + du_1/dx_3 n_1 + du_3/dx_1 n_1 
//      //    + du_2/dx_3 n_2 + du_3/dx_2 n_2 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3) n_3])
//
//      int uz_basis = wkset->usebasis[uz_num];
//      auto basis = wkset->basis_side[uz_basis];
//      auto off = subview(wkset->offsets,uz_num,ALL());
//      
//      if (uz_sidetype == "Neumann") {
//        parallel_for("cns uz boundary resid",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) -= neusource_uz(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
//            }
//          }
//        });
//      } 
//    }
//
//    {
//      // Energy equation
//      // -(w,lambda/cp [dT/dx_1 n_1 + dT/dx_2 n_2 + dT/dx_3 n_3])
//      int T_basis = wkset->usebasis[T_num];
//      auto basis = wkset->basis_side[T_basis];
//      auto off = subview(wkset->offsets,T_num,ALL());
//      
//      if (T_sidetype == "Neumann") {
//        parallel_for("cns T boundary resid",
//                     RangePolicy<AssemblyExec>(0,wkset->numElem),
//                     KOKKOS_LAMBDA (const int elem ) {
//
//          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
//            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
//              res(elem,off(dof)) -= neusource_T(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
//            }
//          }
//        });
//      } 
//    }
//  }
//}
//
//// ========================================================================================
//// The boundary/edge flux
//// ========================================================================================
//
//void cns::computeFlux() {
//  
//}
//
//// ========================================================================================
//// ========================================================================================
//// ========================================================================================
//// ========================================================================================
//
//void cns::setWorkset(Teuchos::RCP<workset> & wkset_) {
//
//  wkset = wkset_;
//
//  vector<string> varlist = wkset->varlist;
//  for (size_t i=0; i<varlist.size(); i++) {
//    if (varlist[i] == "ux")
//      ux_num = i;
//    if (varlist[i] == "pr")
//      pr_num = i;
//    if (varlist[i] == "T")
//      T_num = i;
//    if (varlist[i] == "uy")
//      uy_num = i;
//    if (varlist[i] == "uz")
//      uz_num = i;
//  }
//
//}
//
//// ========================================================================================
//// return the value of the stabilization parameter
//// ========================================================================================
//
//KOKKOS_FUNCTION AD cns::computeTau(const AD & rhoDiffl, const AD & xvl, const AD & yvl, const AD & zvl, const AD & rho, const ScalarT & h, const int & spaceDim, const ScalarT & dt, const bool & isTransient) const {
//  
//  // TODO BWR if this is generalizable, maybe I should have a function for both NS classes
//  // certainly if it's identical
//  // CAN BE but only if the equations collapse
//  //
//  // TODO also -- this does not take into account the Jacobian of the mapping 
//  // to the reference element
//  
//  ScalarT C1 = 4.0;
//  ScalarT C2 = 2.0;
//  ScalarT C3 = isTransient ? 2.0 : 0.0; // only if transient -- TODO not sure BWR
//  
//  AD nvel = 0.0;
//  if (spaceDim == 1)
//    nvel = xvl*xvl;
//  else if (spaceDim == 2)
//    nvel = xvl*xvl + yvl*yvl;
//  else if (spaceDim == 3)
//    nvel = xvl*xvl + yvl*yvl + zvl*zvl;
//  
//  if (nvel > 1E-12)
//    nvel = sqrt(nvel);
//  
//  AD tau;
//  // see, e.g. wikipedia article on SUPG/PSPG 
//  // coefficients can be changed/tuned for different scenarios (including order of time scheme)
//  // https://arxiv.org/pdf/1710.08898.pdf had a good, clear writeup of the final eqns
//  // For the variable-density case, this is based on Gravemeier 2011, Int. J. Numer. Meth. Fluids
//  tau = (C1*rhoDiffl/h/h)*(C1*rhoDiffl/h/h) + (C2*rho*nvel/h)*(C2*rho*nvel/h) + (C3*rho/dt)*(C3*rho/dt);
//  tau = 1./sqrt(tau);
//
//  return tau;
//}
