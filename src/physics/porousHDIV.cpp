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

#include "porousHDIV.hpp"
using namespace MrHyDE;

porousHDIV::porousHDIV(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_) {
  
  label = "porousHDIV";
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  
  if (settings->sublist("Physics").isSublist("Active variables")) {
    if (settings->sublist("Physics").sublist("Active variables").isParameter("p")) {
      myvars.push_back("p");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("p","HVOL"));
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("u")) {
      myvars.push_back("u");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("u","HDIV"));
    }
  }
  else {
    myvars.push_back("p");
    myvars.push_back("u");
    mybasistypes.push_back("HVOL");
    mybasistypes.push_back("HDIV");
  }
  
  usePermData = settings->sublist("Physics").get<bool>("use permeability data",false);
  useWells = settings->sublist("Physics").get<bool>("use well source",false);
  dxnum = 0;
  dynum = 0;
  dznum = 0;
  
}

// ========================================================================================
// ========================================================================================

void porousHDIV::defineFunctions(Teuchos::ParameterList & fs,
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

void porousHDIV::volumeResidual() {
  
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
    
    if (useWells) {
      auto h = wkset->h;
      parallel_for("porous HDIV update well source",RangePolicy<AssemblyExec>(0,Kinv_xx.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        ScalarT C = std::log(0.25*std::exp(-0.5772)*h(elem)/2.0);
        for (size_type pt=0; pt<source.extent(1); pt++) {
          ScalarT Kval = 1.0/Kinv_xx(elem,pt).val();
          source(elem,pt) *= 2.0*PI/C*Kval;
        }
      });
    }
  }
  
  Teuchos::TimeMonitor funceval(*volumeResidualFill);
  
  {
    // (K^-1 u,v) - (p,div v) - src*v (src not added yet)
    
    auto basis = wkset->basis[u_basis];
    auto basis_div = wkset->basis_div[u_basis];
    auto psol = Kokkos::subview(sol,Kokkos::ALL(), pnum, Kokkos::ALL(), 0);
    auto usol = Kokkos::subview(sol,Kokkos::ALL(), unum, Kokkos::ALL(), Kokkos::ALL());
    auto off = Kokkos::subview(offsets, unum, Kokkos::ALL());
    
    if (spaceDim == 1) { // easier to place conditional here than on device
      parallel_for("porous HDIV volume resid u 1D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
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
      parallel_for("porous HDIV volume resid u 2D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
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
      parallel_for("porous HDIV volume resid u 3D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
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
    
    parallel_for("porous HDIV volume resid div(u)",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD F = source(elem,pt) - udiv(elem,pt);
        F *= wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT v = basis(elem,dof,pt,0);
          res(elem,off(dof)) += F*v;
        }
      }
    });
  }
}


// ========================================================================================
// ========================================================================================

void porousHDIV::boundaryResidual() {
  
  bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  int sidetype;
  sidetype = bcs(pnum,cside);
  
  auto basis = wkset->basis_side[unum];
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  auto normals = wkset->normals;
  
  View_AD2_sv bsource;
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (sidetype == 1 ) {
      bsource = functionManager->evaluate("Dirichlet p " + wkset->sidename,"side ip");
    }
    
  }
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  auto off = Kokkos::subview(offsets, unum, Kokkos::ALL());
  auto lambda = Kokkos::subview(aux_side, Kokkos::ALL(), auxpnum, Kokkos::ALL(),0);
  
  if (bcs(pnum,cside) == 1) {
    parallel_for("porous HDIV bndry resid Dirichlet",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
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
    parallel_for("porous HDIV boundary resid MS Dirichlet",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
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
// The boundary/edge flux
// ========================================================================================

void porousHDIV::computeFlux() {
  
  // Since normals get recomputed often, this needs to be reset
  auto normals = wkset->normals;
  
  // Just need the basis for the number of active elements (any side basis will do)
  auto basis = wkset->basis_side[wkset->usebasis[unum]];
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto uflux = Kokkos::subview(flux, Kokkos::ALL(), auxpnum, Kokkos::ALL());
    auto usol = Kokkos::subview(sol_side,Kokkos::ALL(), unum, Kokkos::ALL(), Kokkos::ALL());
    
    parallel_for("porous HDIV flux ",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<normals.extent(1); pt++) {
        AD udotn = 0.0;
        for (size_type dim=0; dim<normals.extent(2); dim++) {
          udotn += usol(elem,pt,dim)*normals(elem,pt,dim);
        }
        uflux(elem,pt) = udotn;
      }
    });
  }
  
}

// ========================================================================================
// ========================================================================================

void porousHDIV::setVars(std::vector<string> & varlist) {
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "p")
      pnum = i;
    if (varlist[i] == "u")
      unum = i;
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

void porousHDIV::setAuxVars(std::vector<string> & auxvarlist) {
  
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "p")
      auxpnum = i;
    if (auxvarlist[i] == "lambda")
      auxpnum = i;
    if (auxvarlist[i] == "pbndry")
      auxpnum = i;
    if (auxvarlist[i] == "u")
      auxunum = i;
  }
}

// ========================================================================================
// ========================================================================================

void porousHDIV::updatePerm(View_AD2_sv Kinv_xx, View_AD2_sv Kinv_yy, View_AD2_sv Kinv_zz) {
  
  View_Sc2 data = wkset->extra_data;
  
  parallel_for("porous HDIV update perm",RangePolicy<AssemblyExec>(0,data.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<Kinv_xx.extent(1); pt++) {
      Kinv_xx(elem,pt) = 1.0/data(elem,0);
      Kinv_yy(elem,pt) = 1.0/data(elem,0);
      Kinv_zz(elem,pt) = 1.0/data(elem,0);
    }
  });
}
