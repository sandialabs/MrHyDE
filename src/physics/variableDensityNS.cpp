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

#include "variableDensityNS.hpp"
using namespace MrHyDE;

// TODO :: grad-div stab in progress

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
VDNS<EvalT>::VDNS(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "VDNS";
  int spaceDim = dimension_;
  
  myvars.push_back("ux");
  myvars.push_back("pr");
  myvars.push_back("T");
  if (spaceDim > 1) {
    myvars.push_back("uy");
  }
  if (spaceDim > 2) {
    myvars.push_back("uz");
  }
  
  // TODO appropriate types?
 
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  mybasistypes.push_back("HGRAD");
  if (spaceDim > 1) {
    mybasistypes.push_back("HGRAD");
  }
  if (spaceDim > 2) {
    mybasistypes.push_back("HGRAD");
  }
  
  // Params from input file
  useSUPG = settings.get<bool>("useSUPG",false);
  usePSPG = settings.get<bool>("usePSPG",false);
  useGRADDIV = settings.get<bool>("useGRADDIV",false);
  // If false, the background thermodynamic pressure changes over time
  openSystem = settings.get<bool>("open system",true);
  // If true, the constraint on the background pressure is different (for a closed domain)
  // see below
  inoutflow = settings.get<bool>("in/outflow",false);
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VDNS<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                   Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("source ux",fs.get<string>("source ux","0.0"),"ip");
  functionManager->addFunction("source pr",fs.get<string>("source pr","0.0"),"ip");
  functionManager->addFunction("source uy",fs.get<string>("source uy","0.0"),"ip");
  functionManager->addFunction("source uz",fs.get<string>("source uz","0.0"),"ip");
  functionManager->addFunction("source T", fs.get<string>("source T", "0.0"),"ip");
  // Default is the ideal gas law as thermal divergence expression is based on this
  functionManager->addFunction("rho",fs.get<string>("rho","p0/(RGas*T)"),"ip");
  // We default to properties of air at 293 K
  // Dynamic viscosity  units are M/L-T
  functionManager->addFunction("mu",fs.get<string>("mu","0.01178"),"ip");
  // Thermal conductivity  units are M-L/T^3-K (K must be Kelvin)
  functionManager->addFunction("lambda",fs.get<string>("lambda","cp*mu/PrNum"),"ip"); 
  // Thermodynamic pressure  units are M/L-T^2
  //functionManager->addFunction("p0",fs.get<string>("p0","100000.0"),"ip");
  // TODO currently this and dp0dt MUST be specified as inactive parameters in the input file
  // BWR -- I'm not sure at this point it's worth going through the trouble to make that automatic
  // Specific heat at constant pressure  units are L^2/T^2-K (K must be Kelvin)
  functionManager->addFunction("cp",fs.get<string>("cp","1004.5"),"ip");
  // Ratio of specific heats
  functionManager->addFunction("gamma",fs.get<string>("gamma","1.4"),"ip");
  // Specific gas constant  units are J/kg-K
  functionManager->addFunction("RGas",fs.get<string>("RGas","287.0"),"ip");
  // Prandtl number
  functionManager->addFunction("PrNum",fs.get<string>("PrNum","1.0"),"ip");

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VDNS<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  ScalarT dt = wkset->deltat;
  bool isTransient = wkset->isTransient;
  Vista<EvalT> source_ux, source_pr, source_uy, source_uz, source_T;
  Vista<EvalT> rho, mu, lambda, cp;
  bool found_p0 = false;
  bool found_dp0dt = false;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source_ux = functionManager->evaluate("source ux","ip");
    source_pr = functionManager->evaluate("source pr","ip");
    source_T  = functionManager->evaluate("source T","ip");
    if (spaceDim > 1) {
      source_uy = functionManager->evaluate("source uy","ip");
    }
    if (spaceDim > 2) {
      source_uz = functionManager->evaluate("source uz","ip");
    }

    // Update thermodynamic and transport properties
    rho = functionManager->evaluate("rho","ip");
    mu = functionManager->evaluate("mu","ip");
    lambda = functionManager->evaluate("lambda","ip");
    cp = functionManager->evaluate("cp","ip"); 

  }

  auto p0 = wkset->getParameter("p0",found_p0);
  auto dp0dt = wkset->getParameter("dp0dt",found_dp0dt);

  if ( ! ( found_p0 && found_dp0dt ) ) {

    cout << "!!!!!!!!!! WARNING !!!!!!!!!!" << endl;
    cout << "User must list p0 and dp0dt as inactive parameters " 
         << "in input file for VDNS!" << endl;
    cout << "Your job will most likely crash soon..." << endl;

  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;

  if (spaceDim == 1) {
    {
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis[ux_basis];
      auto basis_grad = wkset->basis_grad[ux_basis];
      auto ux = wkset->getSolutionField("ux");
      auto dux_dt = wkset->getSolutionField("ux_t");
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,ux_num,ALL());
      
      // Ux equation
      // (v_1,rho du_1/dt) + (v_1,rho u_1 du_1/dx_1) - (dv_1/dx_1,p)
      // + (dv_1/dx_1, 4/3 mu du_1/dx_1) - (v_1,source)
      parallel_for("VDNS ux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = 4./3.*mu(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
          Fx *= wts(elem,pt);
          EvalT F = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt)) - source_ux(elem,pt);
          F *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      // (rho u_1 dv_1/dx_1, \tau_mom R_mom,1)
      // 1/\tau_mom^2 = (c1 \mu/h)^2 + (c2 |\rho u|/h)^2 + (c3 \rho/dt)^2
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        parallel_for("VDNS ux volume resid SUPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),0.0,0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // TODO NO VISCOUS TERM
            // TODO CHECK THIS units, etc.
            EvalT strongres = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
            EvalT Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0); 
            }
          }
        });
      }

      // GRADDIV contribution
      // (dv_1/dx_1, \tau_mass R_mass) 
      // \tau_mass is like h^2/\tau_mom
      if (useGRADDIV) {
        auto h = wkset->h;
        auto T = wkset->getSolutionField("T");
        auto dT_dt = wkset->getSolutionField("T_t");
        auto dT_dx = wkset->getSolutionField("grad(T)[x]");
        parallel_for("VDNS ux volume resid GRADDIV",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),0.0,0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // TODO FIX TAU??? the constant at least is wrong
            tau = h(elem)*h(elem)/tau;
            EvalT thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt));
            thermDiv -= 1./p0(0)*dp0dt(0);
            EvalT strongres = dux_dx(elem,pt) - thermDiv;
            EvalT S = tau*strongres*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += S*basis_grad(elem,dof,pt,0); 
            }
          }
        });
      }
    }

    {
      // Energy equation
      // (w,rho dT/dt) + (w,rho u_1 dT/dx_1) + (dw/dx_1,lambda/cp dT/dx_1) - (w, 1/cp[dp0/dt + Q])
      int T_basis = wkset->usebasis[T_num];
      auto basis = wkset->basis[T_basis];
      auto basis_grad = wkset->basis_grad[T_basis];
      auto T = wkset->getSolutionField("T");
      auto dT_dt = wkset->getSolutionField("T_t");
      auto dT_dx = wkset->getSolutionField("grad(T)[x]"); 
      auto ux = wkset->getSolutionField("ux");
      auto off = subview(wkset->offsets,T_num,ALL());

      parallel_for("VDNS T volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT F = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt))*wts(elem,pt);
          F -= (dp0dt(0) + source_T(elem,pt))/cp(elem,pt)*wts(elem,pt);
          EvalT Fx = lambda(elem,pt)/cp(elem,pt)*dT_dx(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += F*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0);
          }
        }
      });

      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      // (rho u_1 dw/dx_1, \tau_T R_T)
      // 1/\tau_T^2 = (c1 cp/lambda*h)^2 + (c2 |\rho u|/h)^2 + (c3 \rho/dt)^2
      if (useSUPG) {
        auto h = wkset->h;

        parallel_for("VDNS T volume resid SUPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(lambda(elem,pt)/cp(elem,pt),ux(elem,pt),0.0,0.0,
                rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // TODO CHECK THIS, UNITS ETC.
            EvalT strongres = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt)) 
              - (dp0dt(0) + source_T(elem,pt))/cp(elem,pt);
            EvalT Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0);
            }
          }
        });
      }
    }
    
    {
      /////////////////////////////
      // pressure equation
      /////////////////////////////
      // (q,du_1/dx_1) - (q,1/T(dT/dt + u_1 dT/dx_1) - 1/p0 dp0/dt)
      
      int pr_basis = wkset->usebasis[pr_num];
      auto basis = wkset->basis[pr_basis];
      auto basis_grad = wkset->basis_grad[pr_basis];
      auto ux = wkset->getSolutionField("ux");
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto T = wkset->getSolutionField("T");
      auto dT_dt = wkset->getSolutionField("T_t");
      auto dT_dx = wkset->getSolutionField("grad(T)[x]");
      auto off = subview(wkset->offsets,pr_num,ALL());
      
      parallel_for("VDNS pr volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT divu = dux_dx(elem,pt);
          EvalT thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt));
          thermDiv -= 1./p0(0)*dp0dt(0);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (divu-thermDiv)*basis(elem,dof,pt,0)*wts(elem,pt);
          }
        }
      });

      // TODO BWR -- viscous contribution 
      // PSPG contribution
      // (dq/dx_1, \tau_mom R_mom,1)
      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto ux = wkset->getSolutionField("ux");
        auto dux_dt = wkset->getSolutionField("ux_t");
        
        parallel_for("VDNS pr volume resid PSPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),0.0,0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // TODO NO VISCOUS TERM
            // TODO CHECK THIS units, etc.
            EvalT strongres = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
            EvalT Sx = tau*strongres*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0); 
            }
          }
        });
      }
    }
  }
  else if (spaceDim == 2) {
    {
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
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,ux_num,ALL());
      
      // Ux equation
      // (v_1,rho du_1/dt) + (v_1,rho [u_1 du_1/dx_1 + u_2 du_1/dx_2]) - (dv_1/dx_1,p)  
      // + (dv_1/dx_1, \mu [2 * du_1/dx_1 - 2/3 (du_1/dx_1 + du_2/dx_2)]) 
      // + (dv_1/dx_2, \mu [du_1/dx_2 + du_2/dx_1]) - (v_1,source) 
      parallel_for("VDNS ux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = mu(elem,pt)*(2.*dux_dx(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt))) - pr(elem,pt);
          //EvalT Fx = mu(elem,pt)*dux_dx(elem,pt) - pr(elem,pt);
          Fx *= wts(elem,pt);
          EvalT Fy = mu(elem,pt)*(dux_dy(elem,pt) + duy_dx(elem,pt));
          //EvalT Fy = mu(elem,pt)*dux_dy(elem,pt);
          Fy *= wts(elem,pt);
          EvalT F = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) - source_ux(elem,pt);
          F *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      // (rho [u_1 dv_1/dx_1 + u_2 dv_1/dx_2], \tau_mom R_mom,1)
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        parallel_for("VDNS ux volume resid SUPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            EvalT strongres = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
            EvalT Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }

        });
      }

      // GRADDIV contribution
      // (dv_1/dx_1, \tau_mass R_mass) 
      // \tau_mass is like h^2/\tau_mom
      if (useGRADDIV) {
        auto h = wkset->h;
        auto T = wkset->getSolutionField("T");
        auto dT_dt = wkset->getSolutionField("T_t");
        auto dT_dx = wkset->getSolutionField("grad(T)[x]");
        auto dT_dy = wkset->getSolutionField("grad(T)[y]");
        parallel_for("VDNS ux volume resid GRADDIV",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // TODO FIX TAU??? the constant at least is wrong
            tau = h(elem)*h(elem)/tau;
            EvalT thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt));
            thermDiv -= 1./p0(0)*dp0dt(0);
            EvalT strongres = (dux_dx(elem,pt) + duy_dx(elem,pt)) - thermDiv;
            EvalT S = tau*strongres*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += S*basis_grad(elem,dof,pt,0); 
            }
          }
        });
      }
    }
    
    {
      // Uy equation
      // (v_2, rho du_2/dt) + (v_2, rho [u_1 du_2/dx_1 + u_2 du_2/dx_2]) - (dv_2/dx_2,p)
      // + (dv_2/dx_1, \mu [du_1/dx_2 + du_2/dx_1]) 
      // + (dv_2/dx_2, \mu [2 * du_2/dx_2 - 2/3 (du_1/dx_1 + du_2/dx_2)]) - (v_2,source)
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
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,uy_num,ALL());
      
      parallel_for("VDNS uy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = mu(elem,pt)*(dux_dy(elem,pt) + duy_dx(elem,pt));
          //EvalT Fx = mu(elem,pt)*duy_dx(elem,pt);
          Fx *= wts(elem,pt);
          EvalT Fy = mu(elem,pt)*(2.*duy_dy(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt))) - pr(elem,pt);
          //EvalT Fy = mu(elem,pt)*duy_dy(elem,pt) - pr(elem,pt);
          Fy *= wts(elem,pt);
          EvalT F = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) - source_uy(elem,pt);
          F *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      // (rho [u_1 dv_2/dx_1 + u_2 dv_2/dx_2], \tau_mom R_mom,2)
      // TODO CHECK UNITS HERE
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        parallel_for("VDNS uy volume resid SUPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            EvalT strongres = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) + dpr_dy(elem,pt) - source_uy(elem,pt);
            EvalT Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
      }

      // GRADDIV contribution
      // (dv_2/dx_2, \tau_mass R_mass) 
      // \tau_mass is like h^2/\tau_mom
      if (useGRADDIV) {
        auto h = wkset->h;
        auto T = wkset->getSolutionField("T");
        auto dT_dt = wkset->getSolutionField("T_t");
        auto dT_dx = wkset->getSolutionField("grad(T)[x]");
        auto dT_dy = wkset->getSolutionField("grad(T)[y]");
        parallel_for("VDNS uy volume resid GRADDIV",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // TODO FIX TAU??? the constant at least is wrong
            tau = h(elem)*h(elem)/tau;
            EvalT thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt));
            thermDiv -= 1./p0(0)*dp0dt(0);
            EvalT strongres = (dux_dx(elem,pt) + duy_dx(elem,pt)) - thermDiv;
            EvalT S = tau*strongres*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += S*basis_grad(elem,dof,pt,1); 
            }
          }
        });
      }
    }

    {
      /////////////////////////////
      // energy equation
      /////////////////////////////
      // (w,rho dT/dt) + (w,rho [u_1 dT/dx_1 + u_2 dT/dx_2]) + (dw/dx_1,lambda/cp dT/dx_1)
      // + (dw/dx_2,lambda/cp dT/dx_2) - (w,1/cp[dp0/dt + Q])
      int T_basis = wkset->usebasis[T_num];
      auto basis = wkset->basis[T_basis];
      auto basis_grad = wkset->basis_grad[T_basis];
      auto T = wkset->getSolutionField("T");
      auto dT_dt = wkset->getSolutionField("T_t");
      auto dT_dx = wkset->getSolutionField("grad(T)[x]"); 
      auto dT_dy = wkset->getSolutionField("grad(T)[y]"); 
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto off = subview(wkset->offsets,T_num,ALL());
 
      parallel_for("VDNS T volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT F = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt))*wts(elem,pt);
          F -= (dp0dt(0) + source_T(elem,pt))/cp(elem,pt)*wts(elem,pt);
          EvalT Fx = lambda(elem,pt)/cp(elem,pt)*dT_dx(elem,pt)*wts(elem,pt);
          EvalT Fy = lambda(elem,pt)/cp(elem,pt)*dT_dy(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += F*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1);
          }
        }
      });

      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      // (rho [u_1 dw/dx_1 + u_2 dw/dx_2], \tau_T R_T)
      // 1/\tau_T^2 = (c1 cp/lambda*h)^2 + (c2 |\rho u|/h)^2 + (c3 \rho/dt)^2
      if (useSUPG) {
        auto h = wkset->h;

        parallel_for("VDNS T volume resid SUPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(lambda(elem,pt)/cp(elem,pt),ux(elem,pt),uy(elem,pt),0.0,
                rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // TODO CHECK THIS, UNITS ETC.
            EvalT strongres = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) 
                + uy(elem,pt)*dT_dy(elem,pt)) - (dp0dt(0) + source_T(elem,pt))/cp(elem,pt);
            EvalT Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
      }

    }
    
    {
      /////////////////////////////
      // pressure equation
      /////////////////////////////
      // (q,du_1/dx_1 + du_2/dx_2) - (q,1/T(dT/dt + u_1 dT/dx_1 + u_2 dT/dx_2) - 1/p0 dp0/dt)
      int pr_basis = wkset->usebasis[pr_num];
      auto basis = wkset->basis[pr_basis];
      auto basis_grad = wkset->basis_grad[pr_basis];
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
      auto T = wkset->getSolutionField("T");
      auto dT_dt = wkset->getSolutionField("T_t");
      auto dT_dx = wkset->getSolutionField("grad(T)[x]");
      auto dT_dy = wkset->getSolutionField("grad(T)[y]");
      auto off = subview(wkset->offsets,pr_num,ALL());
      
      parallel_for("VDNS pr volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT divu = (dux_dx(elem,pt) + duy_dy(elem,pt));
          EvalT thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt));
          thermDiv -= 1./p0(0)*dp0dt(0);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (divu-thermDiv)*basis(elem,dof,pt,0)*wts(elem,pt);
          }
        }
      });
      
      // TODO BWR -- viscous contribution 
      // PSPG contribution
      // (dq/dx_1, \tau_mom R_mom,1) + (dq/dx_2, \tau_mom R_mom,2)
      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        auto dux_dt = wkset->getSolutionField("ux_t");
        auto duy_dt = wkset->getSolutionField("uy_t");
        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");

        parallel_for("VDNS pr volume resid PSPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),0.0,rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // Strong residual x momentum
            EvalT Sx = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
            Sx *= tau*wts(elem,pt);
            // Strong residual y momentum
            EvalT Sy = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt)) + dpr_dy(elem,pt) - source_uy(elem,pt);
            Sy *= tau*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1);
            }
          }
        });
      }
    }
  }
  else if (spaceDim == 3) {
    {
      // Ux equation
      // (v_1,rho du_1/dt) + (v_1, rho [u_1 du_1/dx_1 + u_2 du_1/dx_2 + u_3 du_1/dx_3]) - (dv_1/dx_1,p)
      // + (dv_1/dx_1, \mu [2 * du_1/dx_1 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3)])
      // + (dv_1/dx_2, \mu [du_1/dx_2 + du_2/dx_1]) + (dv_1/dx_3, \mu [du_1/dx_3 + du_3/dx_1])
      // - (v_1,source)
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
      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
      auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
      auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
      auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,ux_num,ALL());
      
      parallel_for("VDNS ux volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = mu(elem,pt)*(2.*dux_dx(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt))) - pr(elem,pt);
          Fx *= wts(elem,pt);
          EvalT Fy = mu(elem,pt)*(dux_dy(elem,pt) + duy_dx(elem,pt));
          Fy *= wts(elem,pt);
          EvalT Fz = mu(elem,pt)*(dux_dz(elem,pt) + duz_dx(elem,pt));
          Fz *= wts(elem,pt);
          EvalT F = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) - source_ux(elem,pt);
          F *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
      });

      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      // (rho [u_1 dv_1/dx_1 + u_2 dv_1/dx_2 + u_3 dv_1/dx_3], \tau_mom R_mom,1)
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        parallel_for("VDNS ux volume resid SUPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            EvalT strongres = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
            EvalT Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
            EvalT Sz = tau*strongres*rho(elem,pt)*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
      }
    }

    {
      // Uy equation
      // (v_2, rho du_2/dt) + (v_2, rho [u_1 du_2/dx_1 + u_2 du_2/dx_2 + u_3 du_2/dx_3]) - (dv_2/dx_2,p)
      // + (dv_2/dx_1, \mu [du_1/dx_2 + du_2/dx_1]) 
      // + (dv_2/dx_2, \mu [2 * du_2/dx_2 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3)]) 
      // + (dv_2/dx_3, \mu [du_2/dx_3 + du_3/dx_2]) - (v_2,source)
      int uy_basis = wkset->usebasis[uy_num];
      auto basis = wkset->basis[uy_basis];
      auto basis_grad = wkset->basis_grad[uy_basis];
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto uz = wkset->getSolutionField("uz");
      auto duy_dt = wkset->getSolutionField("uy_t");
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
      auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
      auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
      auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
      auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,uy_num,ALL());
      
      parallel_for("VDNS uy volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = mu(elem,pt)*(dux_dy(elem,pt) + duy_dx(elem,pt));
          Fx *= wts(elem,pt);
          EvalT Fy = mu(elem,pt)*(2.*duy_dy(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt))) - pr(elem,pt);
          Fy *= wts(elem,pt);
          EvalT Fz = mu(elem,pt)*(duy_dz(elem,pt) + duz_dy(elem,pt));
          Fz *= wts(elem,pt);
          EvalT F = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) - source_uy(elem,pt);
          F *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
      });
      
      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      // (rho [u_1 dv_2/dx_1 + u_2 dv_2/dx_2 + u_3 dv_2/dx_3], \tau_mom R_mom,2)
      // TODO CHECK UNITS HERE
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        parallel_for("VDNS uy volume resid SUPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            EvalT strongres = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) + dpr_dy(elem,pt) - source_uy(elem,pt);
            EvalT Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
            EvalT Sz = tau*strongres*rho(elem,pt)*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
      }
    }

    {
      // Uz equation
      // (v_3,rho du_3/dt) + (v_3, rho [u_1 du_3/dx_1 + u_2 du_3/dx_2 + u_3 du_3/dx_3]) - (dv_3/dx_3,p)
      // + (dv_3/dx_1, \mu [du_3/dx_1 + du_1/dx_3]) + (dv_3/dx_2, \mu [du_3/dx_2 + du_2/dx_3])
      // + (dv_3/dx_3, \mu [2 * du_3/dx_3 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3])) 
      // - (v_3,source)
      int uz_basis = wkset->usebasis[uz_num];
      auto basis = wkset->basis[uz_basis];
      auto basis_grad = wkset->basis_grad[uz_basis];
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto uz = wkset->getSolutionField("uz");
      auto duz_dt = wkset->getSolutionField("uz_t");
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
      auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
      auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
      auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
      auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
      auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
      auto pr = wkset->getSolutionField("pr");
      auto off = subview(wkset->offsets,uz_num,ALL());
      
      parallel_for("VDNS uz volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT Fx = mu(elem,pt)*(duz_dx(elem,pt) + dux_dz(elem,pt));
          Fx *= wts(elem,pt);
          EvalT Fy = mu(elem,pt)*(duz_dy(elem,pt) + duy_dz(elem,pt));
          Fy *= wts(elem,pt);
          EvalT Fz = mu(elem,pt)*(2.*duz_dz(elem,pt) - 2./3.*(dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt))) - pr(elem,pt);
          Fz *= wts(elem,pt);
          EvalT F = rho(elem,pt)*(duz_dt(elem,pt) + ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) - source_uz(elem,pt);
          F *= wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2) + F*basis(elem,dof,pt,0);
          }
        }
      });

      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      // (rho [u_1 dv_3/dx_1 + u_2 dv_3/dx_2 + u_3 dv_3/dx_3], \tau_mom R_mom,3)
      // TODO CHECK UNITS HERE
      
      if (useSUPG) {
        auto h = wkset->h;
        auto dpr_dz = wkset->getSolutionField("grad(pr)[z]");
        parallel_for("VDNS uz volume resid SUPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            EvalT strongres = rho(elem,pt)*(duz_dt(elem,pt) + ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) + dpr_dz(elem,pt) - source_uz(elem,pt);
            EvalT Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
            EvalT Sz = tau*strongres*rho(elem,pt)*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
      }
    }

    {
      /////////////////////////////
      // energy equation
      /////////////////////////////
      // (w,rho dT/dt) + (w,rho [u_1 dT/dx_1 + u_2 dT/dx_2 + u_3 dT/dx_3]) + (dw/dx_1,lambda/cp dT/dx_1)
      // + (dw/dx_2,lambda/cp dT/dx_2) + (dw/dx_3,lambda/cp dT/dx_3) - (w,1/cp[dp0/dt + Q])
      int T_basis = wkset->usebasis[T_num];
      auto basis = wkset->basis[T_basis];
      auto basis_grad = wkset->basis_grad[T_basis];
      auto T = wkset->getSolutionField("T");
      auto dT_dt = wkset->getSolutionField("T_t");
      auto dT_dx = wkset->getSolutionField("grad(T)[x]"); 
      auto dT_dy = wkset->getSolutionField("grad(T)[y]"); 
      auto dT_dz = wkset->getSolutionField("grad(T)[z]"); 
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto uz = wkset->getSolutionField("uz");
      auto off = subview(wkset->offsets,T_num,ALL());
 
      parallel_for("VDNS T volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT F = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt) + uz(elem,pt)*dT_dz(elem,pt))*wts(elem,pt);
          F -= (dp0dt(0) + source_T(elem,pt))/cp(elem,pt)*wts(elem,pt);
          EvalT Fx = lambda(elem,pt)/cp(elem,pt)*dT_dx(elem,pt)*wts(elem,pt);
          EvalT Fy = lambda(elem,pt)/cp(elem,pt)*dT_dy(elem,pt)*wts(elem,pt);
          EvalT Fz = lambda(elem,pt)/cp(elem,pt)*dT_dz(elem,pt)*wts(elem,pt);
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += F*basis(elem,dof,pt,0) + Fx*basis_grad(elem,dof,pt,0) + Fy*basis_grad(elem,dof,pt,1) + Fz*basis_grad(elem,dof,pt,2);
          }
        }
      });

      // SUPG contribution
      // TODO viscous contribution for higher order elements?
      // (rho [u_1 dw/dx_1 + u_2 dw/dx_2 + u_3 dw/dx_3], \tau_T R_T)
      // 1/\tau_T^2 = (c1 cp/lambda*h)^2 + (c2 |\rho u|/h)^2 + (c3 \rho/dt)^2
      if (useSUPG) {
        auto h = wkset->h;

        parallel_for("VDNS T volume resid SUPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(lambda(elem,pt)/cp(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),
                rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // TODO CHECK THIS, UNITS ETC.
            EvalT strongres = rho(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) 
                + uy(elem,pt)*dT_dy(elem,pt) + uz(elem,pt)*dT_dz(elem,pt)) - (dp0dt(0) + source_T(elem,pt))/cp(elem,pt);
            EvalT Sx = tau*strongres*rho(elem,pt)*ux(elem,pt)*wts(elem,pt);
            EvalT Sy = tau*strongres*rho(elem,pt)*uy(elem,pt)*wts(elem,pt);
            EvalT Sz = tau*strongres*rho(elem,pt)*uz(elem,pt)*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
      }
    }

    {
      /////////////////////////////
      // pressure equation
      /////////////////////////////
      // (q,du_1/dx_1 + du_2/dx_2 + du_3/dx_3) - (q,1/T(dT/dt + u_1 dT/dx_1 + u_2 dT/dx_2 + u_3 dT/dx_3) - 1/p0 dp0/dt)
      int pr_basis = wkset->usebasis[pr_num];
      auto basis = wkset->basis[pr_basis];
      auto basis_grad = wkset->basis_grad[pr_basis];
      auto ux = wkset->getSolutionField("ux");
      auto uy = wkset->getSolutionField("uy");
      auto uz = wkset->getSolutionField("uz");
      auto dux_dx = wkset->getSolutionField("grad(ux)[x]");
      auto duy_dy = wkset->getSolutionField("grad(uy)[y]");
      auto duz_dz = wkset->getSolutionField("grad(uz)[z]");
      auto T = wkset->getSolutionField("T");
      auto dT_dt = wkset->getSolutionField("T_t");
      auto dT_dx = wkset->getSolutionField("grad(T)[x]");
      auto dT_dy = wkset->getSolutionField("grad(T)[y]");
      auto dT_dz = wkset->getSolutionField("grad(T)[z]");
      auto off = subview(wkset->offsets,pr_num,ALL());
      
      parallel_for("VDNS pr volume resid",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT divu = (dux_dx(elem,pt) + duy_dy(elem,pt) + duz_dz(elem,pt));
          EvalT thermDiv = 1./T(elem,pt)*(dT_dt(elem,pt) + ux(elem,pt)*dT_dx(elem,pt) + uy(elem,pt)*dT_dy(elem,pt) + uz(elem,pt)*dT_dz(elem,pt));
          thermDiv -= 1./p0(0)*dp0dt(0);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (divu-thermDiv)*basis(elem,dof,pt,0)*wts(elem,pt);
          }
        }
      });
      
      // TODO BWR -- viscous contribution 
      // PSPG contribution
      // (dq/dx_1, \tau_mom R_mom,1) + (dq/dx_2, \tau_mom R_mom,2) + (dq/dx_3, \tau_mom R_mom,3)
      if (usePSPG) {
        
        auto h = wkset->h;
        auto dpr_dx = wkset->getSolutionField("grad(pr)[x]");
        auto dpr_dy = wkset->getSolutionField("grad(pr)[y]");
        auto dpr_dz = wkset->getSolutionField("grad(pr)[z]");
        auto dux_dt = wkset->getSolutionField("ux_t");
        auto duy_dt = wkset->getSolutionField("uy_t");
        auto duz_dt = wkset->getSolutionField("uz_t");
        auto dux_dy = wkset->getSolutionField("grad(ux)[y]");
        auto dux_dz = wkset->getSolutionField("grad(ux)[z]");
        auto duy_dx = wkset->getSolutionField("grad(uy)[x]");
        auto duy_dz = wkset->getSolutionField("grad(uy)[z]");
        auto duz_dx = wkset->getSolutionField("grad(uz)[x]");
        auto duz_dy = wkset->getSolutionField("grad(uz)[y]");
        
        parallel_for("NS pr volume resid PSPG",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            EvalT tau = this->computeTau(mu(elem,pt),ux(elem,pt),uy(elem,pt),uz(elem,pt),rho(elem,pt),h(elem),spaceDim,dt,isTransient);
            // Strong residual x momentum
            EvalT Sx = rho(elem,pt)*(dux_dt(elem,pt) + ux(elem,pt)*dux_dx(elem,pt) + uy(elem,pt)*dux_dy(elem,pt) + uz(elem,pt)*dux_dz(elem,pt)) + dpr_dx(elem,pt) - source_ux(elem,pt);
            Sx *= tau*wts(elem,pt);
            // Strong residual y momentum
            EvalT Sy = rho(elem,pt)*(duy_dt(elem,pt) + ux(elem,pt)*duy_dx(elem,pt) + uy(elem,pt)*duy_dy(elem,pt) + uz(elem,pt)*duy_dz(elem,pt)) + dpr_dy(elem,pt) - source_uy(elem,pt);
            Sy *= tau*wts(elem,pt);
            // Strong residual z momentum
            EvalT Sz = rho(elem,pt)*(duz_dt(elem,pt) + ux(elem,pt)*duz_dx(elem,pt) + uy(elem,pt)*duz_dy(elem,pt) + uz(elem,pt)*duz_dz(elem,pt)) + dpr_dz(elem,pt) - source_uz(elem,pt);
            Sz *= tau*wts(elem,pt);
            for( size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) += Sx*basis_grad(elem,dof,pt,0) + Sy*basis_grad(elem,dof,pt,1) + Sz*basis_grad(elem,dof,pt,2);
            }
          }
        });
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void VDNS<EvalT>::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;

  int cside = wkset->currentside;

  Vista<EvalT> neusource_ux, neusource_T, neusource_uy, neusource_uz;

  string ux_sidetype = bcs(ux_num,cside);
  string T_sidetype = bcs(T_num,cside);
  string uy_sidetype = ""; string uz_sidetype = "";
  if (spaceDim > 1) {
    uy_sidetype = bcs(uy_num,cside);
  }
  if (spaceDim > 2) {
    uz_sidetype = bcs(uz_num,cside);
  }

  {
    Teuchos::TimeMonitor funceval(*boundaryResidualFunc);

    // evaluate Neumann or traction sources if necessary
    // For momentum equations, the source should be 
    // -p n_\alpha + [du_\alpha/dx_j n_j + du_j/dx_\alpha n_j] -- a traction BC
    //
    // For energy equation, the source should be 
    // \lambda/cp dT/dx_j n_j 

    if (ux_sidetype == "Neumann") {
      neusource_ux = functionManager->evaluate("Neumann ux " + wkset->sidename,"side ip");
    }
    if (T_sidetype == "Neumann") {
      neusource_T = functionManager->evaluate( "Neumann T "  + wkset->sidename,"side ip");
    }
    if (uy_sidetype == "Neumann") {
      neusource_uy = functionManager->evaluate("Neumann uy " + wkset->sidename,"side ip");
    }
    if (uz_sidetype == "Neumann") {
      neusource_uz = functionManager->evaluate("Neumann uz " + wkset->sidename,"side ip");
    }
  }

  auto wts = wkset->wts_side;
  auto h = wkset->h;
  auto res = wkset->res;

  Teuchos::TimeMonitor localtime(*boundaryResidualFill);

  // TODO Right now, if we are just reading sources in, things can be lumped together
  // but to allow for flexibility later on, I'll just go with splitting things up

  if (spaceDim == 1) {
    {
      // Ux equation
      // -(v_1,-p n_1 + \mu [4/3*du_1/dx_1 n_1])
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis_side[ux_basis];
      auto off = subview(wkset->offsets,ux_num,ALL());
      
      if (ux_sidetype == "Neumann") {
        parallel_for("VDNS ux boundary resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {

          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) -= neusource_ux(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      } 
    }

    {
      // Energy equation
      // -(w,lambda/cp dT/dx_1 n_1)
      int T_basis = wkset->usebasis[T_num];
      auto basis = wkset->basis_side[T_basis];
      auto off = subview(wkset->offsets,T_num,ALL());
      
      if (T_sidetype == "Neumann") {
        parallel_for("VDNS T boundary resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {

          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) -= neusource_T(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      } 
    }
  }
  else if (spaceDim == 2) {
    {
      // Ux equation
      // -(v_1,-p n_1 + \mu [2 * du_1/dx_1 n_1 + du_1/dx_2 n_2 + du_2/dx_1 n_2 
      //                                     - 2/3 (du_1/dx_1 + du_2/dx_2) n_1])
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis_side[ux_basis];
      auto off = subview(wkset->offsets,ux_num,ALL());
      
      if (ux_sidetype == "Neumann") {
        parallel_for("VDNS ux boundary resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {

          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) -= neusource_ux(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      } 
    }

    {
      // Uy equation
      // -(v_2,-p n_2 + \mu [2 * du_2/dx_2 n_2 + du_1/dx_2 n_1 + du_2/dx_1 n_1 
      //                                     - 2/3 (du_1/dx_1 + du_2/dx_2) n_2])
      int uy_basis = wkset->usebasis[uy_num];
      auto basis = wkset->basis_side[uy_basis];
      auto off = subview(wkset->offsets,uy_num,ALL());
      
      if (uy_sidetype == "Neumann") {
        parallel_for("VDNS uy boundary resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {

          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) -= neusource_uy(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      } 
    }

    {
      // Energy equation
      // -(w,lambda/cp [dT/dx_1 n_1 + dT/dx_2 n_2])
      int T_basis = wkset->usebasis[T_num];
      auto basis = wkset->basis_side[T_basis];
      auto off = subview(wkset->offsets,T_num,ALL());
      
      if (T_sidetype == "Neumann") {
        parallel_for("VDNS T boundary resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {

          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) -= neusource_T(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      } 
    }
  }
  else if (spaceDim == 3) {
    {
      // Ux equation
      // -(v_1,-p n_1 + \mu [2 * du_1/dx_1 n_1 + du_1/dx_2 n_2 + du_2/dx_1 n_2 
      //    + du_1/dx_3 n_3 + du_3/dx_1 n_3 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3) n_1])
      int ux_basis = wkset->usebasis[ux_num];
      auto basis = wkset->basis_side[ux_basis];
      auto off = subview(wkset->offsets,ux_num,ALL());
      
      if (ux_sidetype == "Neumann") {
        parallel_for("VDNS ux boundary resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {

          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) -= neusource_ux(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      } 
    }

    {
      // Uy equation
      // -(v_2,-p n_2 + \mu [2 * du_2/dx_2 n_2 + du_1/dx_2 n_1 + du_2/dx_1 n_1 
      //    + du_2/dx_3 n_3 + du_3/dx_2 n_3 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3) n_2])

      int uy_basis = wkset->usebasis[uy_num];
      auto basis = wkset->basis_side[uy_basis];
      auto off = subview(wkset->offsets,uy_num,ALL());
      
      if (uy_sidetype == "Neumann") {
        parallel_for("VDNS uy boundary resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {

          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) -= neusource_uy(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      } 
    }

    {
      // Uz equation
      // -(v_3,-p n_3 + \mu [2 * du_3/dx_3 n_3 + du_1/dx_3 n_1 + du_3/dx_1 n_1 
      //    + du_2/dx_3 n_2 + du_3/dx_2 n_2 - 2/3 (du_1/dx_1 + du_2/dx_2 + du_3/dx_3) n_3])

      int uz_basis = wkset->usebasis[uz_num];
      auto basis = wkset->basis_side[uz_basis];
      auto off = subview(wkset->offsets,uz_num,ALL());
      
      if (uz_sidetype == "Neumann") {
        parallel_for("VDNS uz boundary resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {

          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) -= neusource_uz(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
            }
          }
        });
      } 
    }

    {
      // Energy equation
      // -(w,lambda/cp [dT/dx_1 n_1 + dT/dx_2 n_2 + dT/dx_3 n_3])
      int T_basis = wkset->usebasis[T_num];
      auto basis = wkset->basis_side[T_basis];
      auto off = subview(wkset->offsets,T_num,ALL());
      
      if (T_sidetype == "Neumann") {
        parallel_for("VDNS T boundary resid",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {

          for (size_type pt=0; pt<basis.extent(2); pt++ ) {
            for (size_type dof=0; dof<basis.extent(1); dof++ ) {
              res(elem,off(dof)) -= neusource_T(elem,pt)*wts(elem,pt)*basis(elem,dof,pt,0);
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
void VDNS<EvalT>::computeFlux() {
  
}

// ========================================================================================
// return the integrands for the integrated quantities 
// ========================================================================================

template<class EvalT>
std::vector< std::vector<string> > VDNS<EvalT>::setupIntegratedQuantities(const int & spaceDim) {

  std::vector< std::vector<string> > integrandsNamesAndTypes;

  if ( !(openSystem) ) {

    if ( inoutflow ) {
      // for in/outflow we need
      // 1 -- volume of the domain
      std::vector<string> IQdata = {"1.","VDNS vol total volume","volume"};
      integrandsNamesAndTypes.push_back(IQdata);
      // 2 -- total heating (scaled by gamma - 1)
      IQdata = {"source T*(gamma-1.)","VDNS vol Q total","volume"};
      integrandsNamesAndTypes.push_back(IQdata);
      // 3 -- heat flux on the boundary (scaled by gamma - 1)
      // 4 -- "velocity" flux on the boundary (scaled by gamma)
      string hf,vf;
      if (spaceDim == 1) {
        hf = "(gamma-1.)*lambda*(n[x]*grad(T)[x])";
        vf = "gamma*n[x]*ux";
      } else if (spaceDim == 2) {
        hf = "(gamma-1.)*lambda*(n[x]*grad(T)[x] + n[y]*grad(T)[y])";
        vf = "gamma*(n[x]*ux + n[y]*uy)";
      } else if (spaceDim == 3) {
        hf = "(gamma-1.)*lambda*(n[x]*grad(T)[x] + n[y]*grad(T)[y] + n[z]*grad(T)[z])";
        vf = "gamma*(n[x]*ux + n[y]*uy + n[z]*uz)";
      }
      IQdata = {hf,"VDNS bnd heat flux (scaled by gamma - 1)","boundary"};
      integrandsNamesAndTypes.push_back(IQdata);
      IQdata = {vf,"VDNS bnd vel flux (scaled by gamma)","boundary"};
      integrandsNamesAndTypes.push_back(IQdata);

    } else {
      // if no inflow or outflow
      // 1 -- total mass times RGas
      std::vector<string> IQdata = {"rho*RGas","VDNS vol total mass times R","volume"};
      integrandsNamesAndTypes.push_back(IQdata);
      // 2 -- 1/T
      IQdata = {"1./T","VDNS vol inverse temp","volume"};
      integrandsNamesAndTypes.push_back(IQdata);
    }

  } // end if closed system

  return integrandsNamesAndTypes;

}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void VDNS<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;

  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "ux")
      ux_num = i;
    if (varlist[i] == "pr")
      pr_num = i;
    if (varlist[i] == "T")
      T_num = i;
    if (varlist[i] == "uy")
      uy_num = i;
    if (varlist[i] == "uz")
      uz_num = i;
  }

  // we need to keep track of volume/boundary integrals in the closed domain case
  if ( !(openSystem) ) {
    int nIQs = 2;
    if ( inoutflow ) {
      nIQs = 4;
    }
    // add storage for them in the workset
    IQ_start = wkset->addIntegratedQuantities(nIQs);
  }

}

// ========================================================================================
// Update p_0 and dp_0/dt after the IQs are calculated 
// ========================================================================================

template<class EvalT>
void VDNS<EvalT>::updateIntegratedQuantitiesDependents() {

  if ( openSystem ) return; // shouldn't happen... but JIC no op.

  // since param and IQ data is stored on the assembly device 
  // we trick the compiler by doing a parallel for... I don't like this too much

  auto deltat = wkset->deltat; // TODO CHECK ME

  bool found = false; // TODO should have already caught errors so skipping for now
  auto p0 = wkset->getParameter("p0",found);
  auto dp0dt = wkset->getParameter("dp0dt",found);

  auto IQs = wkset->integrated_quantities;

  if ( inoutflow ) {
    // The IQs are volume, total heating*(gamma - 1), (gamma - 1)*heat flux, gamma*velocity flux 
    parallel_for("VDNS update p0 dp0dt",
                RangePolicy<AssemblyExec>(0,1),
                KOKKOS_LAMBDA (const int s) {

      // dp0/dt + 1/vol*p0*vf = 1/vol*hf + 1/vol*heat
      // see equation 10 in gravemeier

      dp0dt(0) = 1./IQs(IQ_start)*(IQs(IQ_start+3) + IQs(IQ_start+1) - p0(0)*IQs(IQ_start+2));
      p0(0) = p0(0) + deltat*dp0dt(0);

    });
  } else {
    // The two IQs are RGas * m_total and \int 1/T 
    parallel_for("VDNS update p0 dp0dt",
                RangePolicy<AssemblyExec>(0,1),
                KOKKOS_LAMBDA (const int s) {
    
      // p0 = R \int \rho dvol / \int 1/T dvol

      ScalarT pnew = IQs(IQ_start)/IQs(IQ_start+1);  // eqn 8 in Gravemeier

      dp0dt(0) = (pnew - p0(0))/deltat;
      p0(0) = pnew;

    });
  }
}

// ========================================================================================
// return the value of the stabilization parameter
// ========================================================================================

template<class EvalT>
KOKKOS_FUNCTION EvalT VDNS<EvalT>::computeTau(const EvalT & rhoDiffl, const EvalT & xvl, const EvalT & yvl, const EvalT & zvl, const EvalT & rho, const ScalarT & h, const int & spaceDim, const ScalarT & dt, const bool & isTransient) const {
  
  // TODO BWR if this is generalizable, maybe I should have a function for both NS classes
  // certainly if it's identical
  // CAN BE but only if the equations collapse
  //
  // TODO also -- this does not take into account the Jacobian of the mapping 
  // to the reference element
  
  ScalarT C1 = 4.0;
  ScalarT C2 = 2.0;
  ScalarT C3 = isTransient ? 2.0 : 0.0; // only if transient -- TODO not sure BWR
  
  EvalT nvel = 0.0;
  if (spaceDim == 1)
    nvel = xvl*xvl;
  else if (spaceDim == 2)
    nvel = xvl*xvl + yvl*yvl;
  else if (spaceDim == 3)
    nvel = xvl*xvl + yvl*yvl + zvl*zvl;
  
  if (nvel > 1E-12)
    nvel = sqrt(nvel);
  
  EvalT tau;
  // see, e.g. wikipedia article on SUPG/PSPG 
  // coefficients can be changed/tuned for different scenarios (including order of time scheme)
  // https://arxiv.org/pdf/1710.08898.pdf had a good, clear writeup of the final eqns
  // For the variable-density case, this is based on Gravemeier 2011, Int. J. Numer. Meth. Fluids
  tau = (C1*rhoDiffl/h/h)*(C1*rhoDiffl/h/h) + (C2*rho*nvel/h)*(C2*rho*nvel/h) + (C3*rho/dt)*(C3*rho/dt);
  tau = 1./sqrt(tau);

  return tau;
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::VDNS<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::VDNS<AD>;

// Standard built-in types
template class MrHyDE::VDNS<AD2>;
template class MrHyDE::VDNS<AD4>;
template class MrHyDE::VDNS<AD8>;
template class MrHyDE::VDNS<AD16>;
template class MrHyDE::VDNS<AD18>;
template class MrHyDE::VDNS<AD24>;
template class MrHyDE::VDNS<AD32>;
#endif
