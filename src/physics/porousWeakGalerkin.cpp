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

#include "porousWeakGalerkin.hpp"

using namespace MrHyDE;

porousWeakGalerkin::porousWeakGalerkin(Teuchos::ParameterList & settings, const int & dimension_)
: physicsbase(settings, dimension_)
{
  
  label = "porousWeakGalerkin";
  include_face = settings.get<bool>("Include face terms","true");
  useAC = settings.get<bool>("useAC",false);
  
  if (settings.isSublist("Active variables")) {
    if (settings.sublist("Active variables").isParameter("pint")) {
      myvars.push_back("pint");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("pint","HVOL"));
    }
    if (settings.sublist("Active variables").isParameter("pbndry")) {
      myvars.push_back("pbndry");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("pbndry","HFACE")); // TODO: turn into HFACE-DG
    }
    if (settings.sublist("Active variables").isParameter("u")) {
      myvars.push_back("u");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("u","HDIV-DG"));
    }
    if (settings.sublist("Active variables").isParameter("t")) {
      myvars.push_back("t");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("t","HDIV-DG"));
    }
  }
  else {
    myvars.push_back("pint");
    myvars.push_back("pbndry");
    myvars.push_back("u");
    myvars.push_back("t");
    mybasistypes.push_back("HVOL");
    mybasistypes.push_back("HFACE"); // TODO: turn into HFACE-DG
    
    if(useAC) {
      mybasistypes.push_back("HDIV_AC-DG");
      mybasistypes.push_back("HDIV_AC-DG");
    }
    else {
      mybasistypes.push_back("HDIV-DG");
      mybasistypes.push_back("HDIV-DG");
    }
  }
  
  usePermData = settings.get<bool>("use permeability data",false);
  
  dxnum = 0;
  dynum = 0;
  dznum = 0;
  
}

// ========================================================================================
// ========================================================================================

void porousWeakGalerkin::defineFunctions(Teuchos::ParameterList & fs,
                                         Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  // Functions
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"ip");
  functionManager->addFunction("perm",fs.get<string>("perm","1.0"),"ip");
  //functionManager->addFunction("kxx",fs.get<string>("kxx","1.0"),"ip");
  //functionManager->addFunction("kxy",fs.get<string>("kxy","0.0"),"ip");
  //functionManager->addFunction("kyx",fs.get<string>("kyx","0.0"),"ip");
  //functionManager->addFunction("kyy",fs.get<string>("kyy","1.0"),"ip");
  //functionManager->addFunction("kxz",fs.get<string>("kxz","0.0"),"ip");
  //functionManager->addFunction("kzx",fs.get<string>("kzx","0.0"),"ip");
  //functionManager->addFunction("kyz",fs.get<string>("kyz","0.0"),"ip");
  //functionManager->addFunction("kzy",fs.get<string>("kzy","0.0"),"ip");
  //functionManager->addFunction("kzz",fs.get<string>("kzz","1.0"),"ip");
}

// ========================================================================================
// ========================================================================================

void porousWeakGalerkin::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int pint_basis = wkset->usebasis[pintnum];
  int u_basis = wkset->usebasis[unum];
  int t_basis = wkset->usebasis[tnum];
  
  Vista source, perm;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    //perm = functionManager->evaluate("perm","ip");
    if (usePermData) {
      auto wts = wkset->wts;
      View_AD2 viewperm("permeability",wts.extent(0),wts.extent(1));
      this->updatePerm(viewperm);
      perm = Vista(viewperm);
    }
    else {
      perm = functionManager->evaluate("perm","ip");
    }
    //kxx = functionManager->evaluate("kxx","ip");
    //kxy = functionManager->evaluate("kxy","ip");
    //kyx = functionManager->evaluate("kyx","ip");
    //kyy = functionManager->evaluate("kyy","ip");
    //kxy = functionManager->evaluate("kxz","ip");
    //kyz = functionManager->evaluate("kyz","ip");
    //kzx = functionManager->evaluate("kzx","ip");
    //kzy = functionManager->evaluate("kzy","ip");
    //kzz = functionManager->evaluate("kzz","ip");
  }
  
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  {
    auto basis = wkset->basis[u_basis];
    auto basis_div = wkset->basis_div[u_basis];
    
    auto pintsol = wkset->getSolutionField("pint");
    auto off = subview(wkset->offsets, unum, ALL());
    
    // (u,v) + (p_0,div(v))
    if (spaceDim == 1) {
      auto ux = wkset->getSolutionField("u[x]");
      parallel_for("porous WG volume resid: u 1D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD pint = pintsol(elem,pt)*wts(elem,pt);
          AD uxw = ux(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += (uxw*vx) + pint*divv;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      auto ux = wkset->getSolutionField("u[x]");
      auto uy = wkset->getSolutionField("u[y]");
      parallel_for("porous WG volume resid: u 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD pint = pintsol(elem,pt)*wts(elem,pt);
          AD uxw = ux(elem,pt)*wts(elem,pt);
          AD uyw = uy(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT vy = basis(elem,dof,pt,1);
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += (uxw*vx+uyw*vy) + pint*divv;
          }
        }
      });
    }
    else {
      auto ux = wkset->getSolutionField("u[x]");
      auto uy = wkset->getSolutionField("u[y]");
      auto uz = wkset->getSolutionField("u[z]");
      parallel_for("porous WG volume resid: u 3D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD pint = pintsol(elem,pt)*wts(elem,pt);
          AD uxw = ux(elem,pt)*wts(elem,pt);
          AD uyw = uy(elem,pt)*wts(elem,pt);
          AD uzw = uz(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT vy = basis(elem,dof,pt,1);
            ScalarT vz = basis(elem,dof,pt,2);
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += (uxw*vx+uyw*vy+uzw*vz) + pint*divv;
          }
        }
      });
    }
    
  }
  
  {
    //  (Ku,s) + (t,s)
    
    auto basis = wkset->basis[t_basis];
    auto basis_div = wkset->basis_div[t_basis];
    
    auto off = subview(wkset->offsets, tnum, ALL());
    
    if (spaceDim == 1) {
      auto ux = wkset->getSolutionField("u[x]");
      auto tx = wkset->getSolutionField("t[x]");
      parallel_for("porous WG volume resid t 1D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Kux = perm(elem,pt)*ux(elem,pt)*wts(elem,pt);
          AD txw = tx(elem,pt)*wts(elem,pt);
          
          AD dx = Kux + txw;
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT sx = basis(elem,dof,pt,0);
            res(elem,off(dof)) += dx*sx;
          }
        }
        
      });
    }
    else if (spaceDim == 2) {
      auto ux = wkset->getSolutionField("u[x]");
      auto uy = wkset->getSolutionField("u[y]");
      auto tx = wkset->getSolutionField("t[x]");
      auto ty = wkset->getSolutionField("t[y]");
      parallel_for("porous WG volume resid t 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Kux = perm(elem,pt)*ux(elem,pt)*wts(elem,pt);
          AD txw = tx(elem,pt)*wts(elem,pt);
          AD Kuy = perm(elem,pt)*uy(elem,pt)*wts(elem,pt);
          AD tyw = ty(elem,pt)*wts(elem,pt);
          
          AD dx = Kux + txw;
          AD dy = Kuy + tyw;
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT sx = basis(elem,dof,pt,0);
            ScalarT sy = basis(elem,dof,pt,1);
            res(elem,off(dof)) += dx*sx + dy*sy;
          }
        }
        
      });
    }
    else {
      auto ux = wkset->getSolutionField("u[x]");
      auto uy = wkset->getSolutionField("u[y]");
      auto uz = wkset->getSolutionField("u[z]");
      auto tx = wkset->getSolutionField("t[x]");
      auto ty = wkset->getSolutionField("t[y]");
      auto tz = wkset->getSolutionField("t[z]");
      parallel_for("porous WG volume resid t 3D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD Kux = perm(elem,pt)*ux(elem,pt)*wts(elem,pt);
          AD txw = tx(elem,pt)*wts(elem,pt);
          AD Kuy = perm(elem,pt)*uy(elem,pt)*wts(elem,pt);
          AD tyw = ty(elem,pt)*wts(elem,pt);
          AD Kuz = perm(elem,pt)*uz(elem,pt)*wts(elem,pt);
          AD tzw = tz(elem,pt)*wts(elem,pt);
          
          AD dx = Kux + txw;
          AD dy = Kuy + tyw;
          AD dz = Kuz + tzw;
          
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            
            ScalarT sx = basis(elem,dof,pt,0);
            ScalarT sy = basis(elem,dof,pt,1);
            ScalarT sz = basis(elem,dof,pt,2);
            // should be k_ij u_i s_j, but currently we assume K=1
            res(elem,off(dof)) += dx*sx + dy*sy + dz*sz;
            //res(e,resindex) += kxx(e,k)*ux*sx
            //                 + kxy(e,k)*ux*sy
            //                 + kyx(e,k)*uy*sx
            //                 + kyy(e,k)*uy*sy
            //                 + kxz(e,k)*ux*sz
            //                 + kyz(e,k)*uy*sz
            //                 + kzx(e,k)*uz*sx
            //                 + kzy(e,k)*uz*sy
            //                 + kzz(e,k)*uz*sz;
            
            //res(e,resindex) += (tx*sx + ty*sy + tz*sz)*wts(e,k);
            
          }
        }
        
      });
    }
    
  }
  
  {
    //  (div(t),q_0) - (f,q_0)
    auto basis = wkset->basis[pint_basis];
    auto tdiv = wkset->getSolutionField("div(t)");
    auto off = subview(wkset->offsets, pintnum, ALL());
    
    parallel_for("porous WG volume resid div(t)",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD divt = tdiv(elem,pt)*wts(elem,pt);
        AD S = source(elem,pt)*wts(elem,pt);
        AD tdiff = divt-S;
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT qint = basis(elem,dof,pt,0);
          res(elem,off(dof)) += tdiff*qint;
        }
      }
      
    });
  }
}


// ========================================================================================
// ========================================================================================

void porousWeakGalerkin::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  string bctype = bcs(pintnum,cside);
  
  // TMW: changed this from t_basis to u_basis (check on this)
  int u_basis = wkset->usebasis[unum];
  
  auto basis = wkset->basis_side[u_basis];
  
  Vista bsource;
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (bctype == "Dirichlet" ) {
      bsource = functionManager->evaluate("Dirichlet pbndry " + wkset->sidename,"side ip");
    }
    
  }
  
  // Since normals get recomputed often, this needs to be reset
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  View_Sc2 nx, ny, nz;
  View_AD2 ux, uy, uz;
  nx = wkset->getScalarField("n[x]");
  ux = wkset->getSolutionField("u[x]");
  if (spaceDim > 1) {
    ny = wkset->getScalarField("n[y]");
    uy = wkset->getSolutionField("u[y]");
  }
  if (spaceDim > 2) {
    nz = wkset->getScalarField("n[z]");
    uz = wkset->getSolutionField("u[z]");
  }
  
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  if (bcs(pintnum,cside) == "Dirichlet") {
    auto off = subview(wkset->offsets, unum, ALL());
    parallel_for("porous WG bndry resid Dirichlet",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD S = bsource(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = basis(elem,dof,pt,0)*nx(elem,pt);
          if (dim > 1) {
            vdotn += basis(elem,dof,pt,1)*ny(elem,pt);
          }
          if (dim > 2) {
            vdotn += basis(elem,dof,pt,2)*nz(elem,pt);
          }
          res(elem,off(dof)) -= S*vdotn;
        }
      }
    });
  }
  else if (bcs(pintnum,cside) == "interface") {
    auto off = subview(wkset->offsets, unum, ALL());
    auto lambda = wkset->getSolutionField("aux pbndry");
    parallel_for("porous WG bndry resid MS Dirichlet",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD lam = lambda(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = basis(elem,dof,pt,0)*nx(elem,pt);
          if (dim > 1) {
            vdotn += basis(elem,dof,pt,1)*ny(elem,pt);
          }
          if (dim > 2) {
            vdotn += basis(elem,dof,pt,2)*nz(elem,pt);
          }
          res(elem,off(dof)) -= lam*vdotn;
        }
      }
    });
  }
  
}


// ========================================================================================
// The edge (2D) and face (3D) contributions to the residual
// ========================================================================================

void porousWeakGalerkin::faceResidual() {
  
  int spaceDim = wkset->dimension;
  int pbndry_basis = wkset->usebasis[pbndrynum];
  int u_basis = wkset->usebasis[unum];
  
  // Since normals get recomputed often, this needs to be reset
  View_Sc2 nx, ny, nz;
  View_AD2 tx, ty, tz;
  nx = wkset->getScalarField("n[x]");
  tx = wkset->getSolutionField("t[x]");
  if (spaceDim > 1) {
    ny = wkset->getScalarField("n[y]");
    ty = wkset->getSolutionField("t[y]");
  }
  if (spaceDim > 2) {
    nz = wkset->getScalarField("n[z]");
    tz = wkset->getSolutionField("t[z]");
  }
  
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  {
    // include <pbndry, v \cdot n> in velocity equation
    auto basis = wkset->basis_side[u_basis];
    auto off = subview(wkset->offsets, unum, ALL());
    auto pbndry = wkset->getSolutionField("pbndry");
    
    parallel_for("porous WG face resid pbndry",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD p = pbndry(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = basis(elem,dof,pt,0)*nx(elem,pt);
          if (dim > 1) {
            vdotn += basis(elem,dof,pt,1)*ny(elem,pt);
          }
          if (dim > 2) {
            vdotn += basis(elem,dof,pt,2)*nz(elem,pt);
          }
          
          //int resindex = offsets(unum,i);
          res(elem,off(dof)) -= p*vdotn;
        }
      }
    });
  }
  
  {
    // include -<t \cdot n, qbndry> in interface equation
    auto ubasis = wkset->basis_side[u_basis];
    auto basis = wkset->basis_side[pbndry_basis];
    auto off = subview(wkset->offsets, pbndrynum, ALL());
    // TMW: previous code used u instead of t
    
    parallel_for("porous WG face resid t dot n",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = ubasis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD tdotn = tx(elem,pt)*nx(elem,pt);
        if (dim > 1) {
          tdotn += ty(elem,pt)*ny(elem,pt);
        }
        if (dim > 2) {
          tdotn += tz(elem,pt)*nz(elem,pt);
        }
        
        
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT qbndry = basis(elem,dof,pt,0)*wts(elem,pt);
          res(elem,off(dof)) -= tdotn*qbndry;
        }
      }
    });
  }
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void porousWeakGalerkin::computeFlux() {
  
  int spaceDim = wkset->dimension;
  
  View_Sc2 nx, ny, nz;
  View_AD2 tx, ty, tz;
  nx = wkset->getScalarField("n[x]");
  tx = wkset->getSolutionField("t[x]");
  if (spaceDim > 1) {
    ny = wkset->getScalarField("n[y]");
    ty = wkset->getSolutionField("t[y]");
  }
  if (spaceDim > 2) {
    nz = wkset->getScalarField("n[z]");
    tz = wkset->getSolutionField("t[z]");
  }
  
  // Just need the basis for the number of active elements (any side basis will do)
  auto basis = wkset->basis_side[wkset->usebasis[unum]];
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto mflux = subview(wkset->flux, ALL(), auxpbndrynum, ALL());
    
    parallel_for("porous WG flux",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<mflux.extent(1); pt++) {
        AD tdotn = tx(elem,pt)*nx(elem,pt);
        if (dim > 1) {
          tdotn += ty(elem,pt)*ny(elem,pt);
        }
        if (dim > 2) {
          tdotn += tz(elem,pt)*nz(elem,pt);
        }
        
        mflux(elem,pt) = tdotn;
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

void porousWeakGalerkin::setWorkset(Teuchos::RCP<Workset> & wkset_) {
  
  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "pint")
      pintnum = i;
    if (varlist[i] == "pbndry")
      pbndrynum = i;
    if (varlist[i] == "u")
      unum = i;
    if (varlist[i] == "t")
      tnum = i;
    if (varlist[i] == "dx")
      dxnum = i;
    if (varlist[i] == "dy")
      dynum = i;
    if (varlist[i] == "dz")
      dznum = i;
  }
  
  vector<string> auxvarlist = wkset->aux_varlist;
  
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "pbndry" || auxvarlist[i] == "lambda")
      auxpbndrynum = i;
    if (auxvarlist[i] == "pint")
      auxpbndrynum = i; // hard-coded for now
    if (auxvarlist[i] == "u")
      auxunum = i;
    if (auxvarlist[i] == "t")
      auxtnum = i;
  }
}

// ========================================================================================
// ========================================================================================

void porousWeakGalerkin::updatePerm(View_AD2 perm) {
  
  View_Sc2 data = wkset->extra_data;
  parallel_for("porous WG update perm",RangePolicy<AssemblyExec>(0,perm.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<perm.extent(1); pt++) {
      perm(elem,pt) = data(elem,0);
    }
  });
}

