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

#include "porousHDIV_hybridized.hpp"
using namespace MrHyDE;

porousHDIV_HYBRID::porousHDIV_HYBRID(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_) {
  
  label = "porousHDIV-Hybrid";
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  include_face = true;
  
  if (settings->sublist("Physics").isSublist("Active variables")) {
    if (settings->sublist("Physics").sublist("Active variables").isParameter("p")) {
      myvars.push_back("p");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("p","HVOL"));
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("u")) {
      myvars.push_back("u");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("u","HDIV-DG"));
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("lambda")) {
      myvars.push_back("lambda");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("lambda","HFACE"));
    }
  }
  else {
    myvars.push_back("p");
    myvars.push_back("u");
    myvars.push_back("lambda");
    mybasistypes.push_back("HVOL");
    mybasistypes.push_back("HDIV-DG");
    mybasistypes.push_back("HFACE");
  }
  usePermData = settings->sublist("Physics").get<bool>("use permeability data",false);
  
  dxnum = 0;
  dynum = 0;
  dznum = 0;
  
}

// ========================================================================================
// ========================================================================================

void porousHDIV_HYBRID::defineFunctions(Teuchos::ParameterList & fs,
                                        Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"ip");
  functionManager->addFunction("Kinv_xx",fs.get<string>("Kinv_xx","1.0"),"ip");
  functionManager->addFunction("Kinv_yy",fs.get<string>("Kinv_yy","1.0"),"ip");
  functionManager->addFunction("Kinv_zz",fs.get<string>("Kinv_zz","1.0"),"ip");
    
}

// ========================================================================================
// ========================================================================================

void porousHDIV_HYBRID::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int p_basis = wkset->usebasis[pnum];
  int u_basis = wkset->usebasis[unum];
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  View_AD2_sv source, bsource, Kinv_xx, Kinv_yy, Kinv_zz;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    if (usePermData) {
      auto wts = wkset->wts;
      Kinv_xx = View_AD2("K inverse xx",wts.extent(0),wts.extent(1));
      Kinv_yy = View_AD2("K inverse yy",wts.extent(0),wts.extent(1));
      Kinv_zz = View_AD2("K inverse zz",wts.extent(0),wts.extent(1));
      this->updatePerm(Kinv_xx, Kinv_yy, Kinv_zz);
    }
    else {
      Kinv_xx = functionManager->evaluate("Kinv_xx","ip");
      Kinv_yy = functionManager->evaluate("Kinv_yy","ip");
      Kinv_zz = functionManager->evaluate("Kinv_zz","ip");
    }
  }
  
  {
    // (K^-1 u,v) - (p,div v) - src*v (src not added yet)
    
    auto basis = wkset->basis[u_basis];
    auto basis_div = wkset->basis_div[u_basis];
    auto psol = Kokkos::subview(sol,Kokkos::ALL(), pnum, Kokkos::ALL(), 0);
    auto usol = Kokkos::subview(sol,Kokkos::ALL(), unum, Kokkos::ALL(), Kokkos::ALL());
    auto off = Kokkos::subview(offsets, unum, Kokkos::ALL());
    
    if (spaceDim == 1) { // easier to place conditional here than on device
      parallel_for("porous HDIV-HY volume resid u 1D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD p = psol(elem,pt)*wts(elem,pt);
          AD Kiux = Kinv_xx(elem,pt)*usol(elem,pt,0)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += Kiux*vx - p*divv;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for("porous HDIV-HY volume resid u 2D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD p = psol(elem,pt)*wts(elem,pt);
          AD Kiux = Kinv_xx(elem,pt)*usol(elem,pt,0)*wts(elem,pt);
          AD Kiuy = Kinv_yy(elem,pt)*usol(elem,pt,1)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT vy = basis(elem,dof,pt,1);
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += Kiux*vx + Kiuy*vy - p*divv;
          }
        }
      });
    }
    else {
      parallel_for("porous HDIV-HY volume resid u 3D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD p = psol(elem,pt)*wts(elem,pt);
          AD Kiux = Kinv_xx(elem,pt)*usol(elem,pt,0)*wts(elem,pt);
          AD Kiuy = Kinv_yy(elem,pt)*usol(elem,pt,1)*wts(elem,pt);
          AD Kiuz = Kinv_zz(elem,pt)*usol(elem,pt,2)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT vy = basis(elem,dof,pt,1);
            ScalarT vz = basis(elem,dof,pt,2);
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += Kiux*vx + Kiuy*vy + Kiuz*vz - p*divv;
          }
        }
      });
    }
  }
  
  {
    // -(div u,q) + (src,q) (src not added yet)
    
    auto basis = wkset->basis[p_basis];
    
    auto udiv = Kokkos::subview(sol_div,Kokkos::ALL(), unum, Kokkos::ALL());
    auto off = Kokkos::subview(offsets,pnum, Kokkos::ALL());
    
    parallel_for("porous HDIV-HY volume resid div(u)",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD divu = udiv(elem,pt)*wts(elem,pt);
        AD src = source(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT v = basis(elem,dof,pt,0);
          res(elem,off(dof)) += -divu*v + src*v;
        }
      }
    });
  }
}


// ========================================================================================
// ========================================================================================

void porousHDIV_HYBRID::boundaryResidual() {
  
  bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  int sidetype = bcs(pnum,cside);
  
  int u_basis = wkset->usebasis[unum];
  
  auto basis = wkset->basis_side[u_basis];
  
  View_AD2_sv bsource;
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (sidetype == 1 ) {
      bsource = functionManager->evaluate("Dirichlet p " + wkset->sidename,"side ip");
    }
    
  }
  
  // Since normals get recomputed often, this needs to be reset
  auto normals = wkset->normals;
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  auto off = Kokkos::subview(offsets, unum, Kokkos::ALL());
  auto lambda = Kokkos::subview(aux_side, Kokkos::ALL(), auxlambdanum, Kokkos::ALL(),0);
  
  if (bcs(pnum,cside) == 1) {
    parallel_for("porous HDIV-HY bndry resid Dirichlet",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD src = bsource(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = 0.0;
          for (size_type dim=0; dim<normals.extent(2); dim++) {
            vdotn += basis(elem,dof,pt,dim)*normals(elem,pt,dim);
          }
          res(elem,off(dof)) += src*vdotn;
        }
      }
    });
  }
  else if (bcs(pnum,cside) == 5) {
    parallel_for("porous HDIV-HY bndry resid MS Dirichlet",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD lam = lambda(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = 0.0;
          for (size_type dim=0; dim<normals.extent(2); dim++) {
            vdotn += basis(elem,dof,pt,dim)*normals(elem,pt,dim);
          }
          res(elem,off(dof)) += lam*vdotn;
        }
      }
    });
  }
}

// ========================================================================================
// The edge (2D) and face (3D) contributions to the residual
// ========================================================================================

void porousHDIV_HYBRID::faceResidual() {
  
  int lambda_basis = wkset->usebasis[lambdanum];
  int u_basis = wkset->usebasis[unum];
  
  // Since normals get recomputed often, this needs to be reset
  auto normals = wkset->normals;
  auto res = wkset->res;
  auto wts = wkset->wts_side;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  {
    // include <lambda, v \cdot n> in velocity equation
    auto basis = wkset->basis_face[u_basis];
    auto lambda = Kokkos::subview(sol_face,Kokkos::ALL(), lambdanum, Kokkos::ALL(), 0);
    auto off = Kokkos::subview(offsets, unum, Kokkos::ALL());
    
    parallel_for("porous HDIV-HY face resid lambda",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD lam = lambda(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = 0.0;
          for (size_type dim=0; dim<normals.extent(2); dim++) {
            vdotn += basis(elem,dof,pt,dim)*normals(elem,pt,dim);
          }
          res(elem,off(dof)) += lam*vdotn;
        }
      }
    });
  }
  
  {
    // include -<u \cdot n, mu> in interface equation
    auto basis = wkset->basis_face[lambda_basis];
    auto usol = Kokkos::subview(sol_face, Kokkos::ALL(), unum, Kokkos::ALL(), Kokkos::ALL());
    auto off = Kokkos::subview(offsets, lambdanum, Kokkos::ALL());
    
    parallel_for("porous HDIV-HY face resid u dot n",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD udotn = 0.0;
        for (size_type dim=0; dim<normals.extent(2); dim++) {
          udotn += usol(elem,pt,dim)*normals(elem,pt,dim);
        }
        udotn *= wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) -= udotn*basis(elem,dof,pt,0);
        }
      }
    });
  }
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void porousHDIV_HYBRID::computeFlux() {
  
  auto normals = wkset->normals;
  
  // Just need the basis for the number of active elements (any side basis will do)
  auto basis = wkset->basis_side[wkset->usebasis[unum]];
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto usol = Kokkos::subview(sol_side, Kokkos::ALL(), unum, Kokkos::ALL(), Kokkos::ALL());
    auto uflux = Kokkos::subview(flux, Kokkos::ALL(), auxlambdanum, Kokkos::ALL());
    
    parallel_for("porous HDIV-HY flux",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<uflux.extent(1); pt++) {
        AD udotn = 0.0;
        for (size_type dim=0; dim<normals.extent(2); dim++) {
          udotn += usol(elem,pt,dim)*normals(elem,pt,dim);
        }
        uflux(elem,pt) = -udotn;
      }
    });
  }
  
}

// ========================================================================================
// ========================================================================================

void porousHDIV_HYBRID::setVars(std::vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "p")
      pnum = i;
    if (varlist[i] == "u")
      unum = i;
    if (varlist[i] == "lambda")
      lambdanum = i;
    if (varlist[i] == "dx")
      dxnum = i;
    if (varlist[i] == "dy")
      dynum = i;
    if (varlist[i] == "dz")
      dznum = i;
  }
}

// ========================================================================================
// ========================================================================================

void porousHDIV_HYBRID::setAuxVars(std::vector<string> & auxvarlist) {
  
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "p")
      auxlambdanum = i; // hard-coded for now
    if (auxvarlist[i] == "u")
      auxunum = i;
    if (auxvarlist[i] == "lambda")
      auxlambdanum = i;
  }
}

// ========================================================================================
// ========================================================================================

void porousHDIV_HYBRID::updatePerm(View_AD2_sv Kinv_xx, View_AD2_sv Kinv_yy, View_AD2_sv Kinv_zz) {
  
  View_Sc2 data = wkset->extra_data;
  
  parallel_for(RangePolicy<AssemblyExec>(0,data.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<Kinv_xx.extent(1); pt++) {
      Kinv_xx(elem,pt) = data(elem,0);
      Kinv_yy(elem,pt) = data(elem,0);
      Kinv_zz(elem,pt) = data(elem,0);
    }
  });
}
