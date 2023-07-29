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

#include "porousMixed.hpp"
using namespace MrHyDE;

porousMixed::porousMixed(Teuchos::ParameterList & settings, const int & dimension_)
  : physicsbase(settings, dimension_)
{
  
  label = "porousMixed";
  int spaceDim = dimension_;
  
  if (settings.isSublist("Active variables")) {
    if (settings.sublist("Active variables").isParameter("p")) {
      myvars.push_back("p");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("p","HVOL"));
    }
    if (settings.sublist("Active variables").isParameter("u")) {
      myvars.push_back("u");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("u","HDIV"));
    }
  }
  else {
    myvars.push_back("p");
    myvars.push_back("u");
    
    if (dimension_ == 1) { // to avoid the error in 1D HDIV
      mybasistypes.push_back("HVOL");
      mybasistypes.push_back("HGRAD");
    }
    else {
      mybasistypes.push_back("HVOL");
      mybasistypes.push_back("HDIV");
    }
  }
  
  usePermData = settings.get<bool>("use permeability data",false);
  useWells = settings.get<bool>("use well source",false);
  if (useWells) myWells = wells(settings);
  dxnum = 0;
  dynum = 0;
  dznum = 0;
  
  useKL = settings.get<bool>("use KL expansion",false);
  if (useKL) {
    if (settings.isSublist("KL parameters")) {
      auto KLlist = settings.sublist("KL parameters");
      permKLx = klexpansion(KLlist.sublist("x-direction").get<int>("N"),
                            KLlist.sublist("x-direction").get<double>("L"),
                            KLlist.sublist("x-direction").get<double>("sigma"),
                            KLlist.sublist("x-direction").get<double>("eta"));
      int numindices = permKLx.getNumTerms();
      if (spaceDim > 1) {
        permKLy = klexpansion(KLlist.sublist("y-direction").get<int>("N"),
                              KLlist.sublist("y-direction").get<double>("L"),
                              KLlist.sublist("y-direction").get<double>("sigma"),
                              KLlist.sublist("y-direction").get<double>("eta"));
        numindices *= permKLy.getNumTerms();
      }
      if (spaceDim > 2) {
        permKLz = klexpansion(KLlist.sublist("z-direction").get<int>("N"),
                              KLlist.sublist("z-direction").get<double>("L"),
                              KLlist.sublist("z-direction").get<double>("sigma"),
                              KLlist.sublist("z-direction").get<double>("eta"));
        numindices *= permKLz.getNumTerms();
      }

      // Need to define these indices so the coeffs are ordered properly
      // BWR -- attempting a simple total order sorting
      // it is not particularly efficient but should work CHECK ME
      KLindices = Kokkos::View<size_t**,AssemblyDevice>("KL indices",numindices,spaceDim);
      int prog = 0;
      if (spaceDim == 1) {
        for (size_t i=0; i<permKLx.getNumTerms(); ++i) {
          KLindices(prog,0) = i;
          prog++;
        }
      }
      else if (spaceDim == 2) {
        size_t alpha_max = permKLx.getNumTerms() + permKLy.getNumTerms() - 1; // max total order
        // loop over alpha (total basis order)
        for (size_t alpha=0; alpha<alpha_max; ++alpha) {
          // loop over y-direction basis
          for (size_t j=0; j<permKLy.getNumTerms(); ++j) {
            // loop over x-direction basis
            for (size_t i=0; i<permKLx.getNumTerms(); ++i) {
              if ( i + j == alpha ) {
                KLindices(prog,0) = i;
                KLindices(prog,1) = j;
                prog++;
              }
            }
          }
        }
      }
      else if (spaceDim == 3) {
        size_t alpha_max = permKLx.getNumTerms() + permKLy.getNumTerms() + permKLz.getNumTerms() - 2; // max total order
        // loop over alpha (total basis order)
        for (size_t alpha=0; alpha<alpha_max; ++alpha) {
          // loop over z-direction basis
          for (size_t k=0; k<permKLz.getNumTerms(); ++k) {
            // loop over y-direction basis
            for (size_t j=0; j<permKLy.getNumTerms(); ++j) {
              // loop over x-direction basis
              for (size_t i=0; i<permKLx.getNumTerms(); ++i) {
                if ( i + j + k == alpha ) {
                  KLindices(prog,0) = i;
                  KLindices(prog,1) = j;
                  KLindices(prog,2) = k;
                  prog++;
                }
              }
            }
          }
        }
      }
    }
    else {
      // throw an error
    }
  }
  
}

// ========================================================================================
// ========================================================================================

void porousMixed::defineFunctions(Teuchos::ParameterList & fs,
                                 Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  // Functions
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"ip");
  functionManager->addFunction("Kinv_xx",fs.get<string>("Kinv_xx","1.0"),"ip");
  functionManager->addFunction("Kinv_yy",fs.get<string>("Kinv_yy","1.0"),"ip");
  functionManager->addFunction("Kinv_zz",fs.get<string>("Kinv_zz","1.0"),"ip");

  functionManager->addFunction("total_mobility",fs.get<string>("total_mobility","1.0"),"ip");

}

// ========================================================================================
// ========================================================================================

void porousMixed::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int p_basis = wkset->usebasis[pnum];
  int u_basis = wkset->usebasis[unum];
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  Vista source, bsource, Kinv_xx, Kinv_yy, Kinv_zz, mobility;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    mobility = functionManager->evaluate("total_mobility","ip");
    
    if (usePermData) {
      View_AD2 view_Kinv_xx("K inverse xx",wts.extent(0),wts.extent(1));
      View_AD2 view_Kinv_yy("K inverse yy",wts.extent(0),wts.extent(1));
      View_AD2 view_Kinv_zz("K inverse zz",wts.extent(0),wts.extent(1));
      this->updatePerm(view_Kinv_xx, view_Kinv_yy, view_Kinv_zz);
      Kinv_xx = Vista(view_Kinv_xx);
      Kinv_yy = Vista(view_Kinv_yy);
      Kinv_zz = Vista(view_Kinv_zz);
    }
    else {
      Kinv_xx = functionManager->evaluate("Kinv_xx","ip");
      Kinv_yy = functionManager->evaluate("Kinv_yy","ip");
      Kinv_zz = functionManager->evaluate("Kinv_zz","ip");
    }
    
    if (useKL) {
      View_AD2 KL_Kxx("KL K xx",wts.extent(0),wts.extent(1));
      View_AD2 KL_Kyy("KL K yy",wts.extent(0),wts.extent(1));
      View_AD2 KL_Kzz("KL K zz",wts.extent(0),wts.extent(1));
      this->updateKLPerm(KL_Kxx, KL_Kyy, KL_Kzz);
      
      View_AD2 new_Kxx("new K xx",wts.extent(0),wts.extent(1));
      View_AD2 new_Kyy("new K yy",wts.extent(0),wts.extent(1));
      View_AD2 new_Kzz("new K zz",wts.extent(0),wts.extent(1));
      
      parallel_for("porous mixed update KL",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<KL_Kxx.extent(1); ++pt) {
          new_Kxx(elem,pt) = Kinv_xx(elem,pt)/exp(KL_Kxx(elem,pt));
        }
        
        if (spaceDim > 1) {
          for (size_type pt=0; pt<KL_Kyy.extent(1); ++pt) {
            new_Kyy(elem,pt) = Kinv_yy(elem,pt)/exp(KL_Kyy(elem,pt));
          }
        }
        if (spaceDim > 2) {
          for (size_type pt=0; pt<KL_Kzz.extent(1); ++pt) {
            new_Kzz(elem,pt) = Kinv_zz(elem,pt)/exp(KL_Kzz(elem,pt));
          }
        }
        
      });
      
      Kinv_xx = Vista(new_Kxx);
      Kinv_yy = Vista(new_Kyy);
      Kinv_zz = Vista(new_Kzz);
      
    }
    
    if (useWells) {
      auto h = wkset->h;
      source = myWells.addWellSources(source,h,functionManager,
      wts.extent(0) /* numElem */, wts.extent(1) /* numIp */ ); 
    }
  }
  
  Teuchos::TimeMonitor funceval(*volumeResidualFill);
  
  {
    // ((mobility \times K)^-1 u,v) - (p,div v) - src*v (src not added yet)
    
    auto basis = wkset->basis[u_basis];
    auto psol = wkset->getSolutionField("p");
    auto off = subview(wkset->offsets, unum, ALL());
    
    if (spaceDim == 1) { // easier to place conditional here than on device
      auto ux = wkset->getSolutionField("u");
      auto basis_div = wkset->basis_grad[u_basis];
        
      parallel_for("porous HDIV volume resid u 1D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD p = psol(elem,pt)*wts(elem,pt);
          AD Kiux = Kinv_xx(elem,pt)*ux(elem,pt)*wts(elem,pt);
          Kiux /= mobility(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT divv = basis_div(elem,dof,pt,0);
            res(elem,off(dof)) += Kiux*vx - p*divv;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      auto ux = wkset->getSolutionField("u[x]");
      auto uy = wkset->getSolutionField("u[y]");
      auto basis_div = wkset->basis_div[u_basis];
      
      parallel_for("porous HDIV volume resid u 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD p = psol(elem,pt)*wts(elem,pt);
          AD Kiux = Kinv_xx(elem,pt)*ux(elem,pt)*wts(elem,pt);
          AD Kiuy = Kinv_yy(elem,pt)*uy(elem,pt)*wts(elem,pt);
          Kiux /= mobility(elem,pt);
          Kiuy /= mobility(elem,pt);
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
      auto ux = wkset->getSolutionField("u[x]");
      auto uy = wkset->getSolutionField("u[y]");
      auto uz = wkset->getSolutionField("u[z]");
      auto basis_div = wkset->basis_div[u_basis];
      
      parallel_for("porous HDIV volume resid u 3D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          AD p = psol(elem,pt)*wts(elem,pt);
          AD Kiux = Kinv_xx(elem,pt)*ux(elem,pt)*wts(elem,pt);
          AD Kiuy = Kinv_yy(elem,pt)*uy(elem,pt)*wts(elem,pt);
          AD Kiuz = Kinv_zz(elem,pt)*uz(elem,pt)*wts(elem,pt);
          Kiux /= mobility(elem,pt);
          Kiuy /= mobility(elem,pt);
          Kiuz /= mobility(elem,pt);
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
    auto off = subview(wkset->offsets,pnum, ALL());
    View_AD2 udiv;
    if (spaceDim == 1) {
      udiv = wkset->getSolutionField("grad(u)[x]");
    }
    else {
      udiv = wkset->getSolutionField("div(u)");
    }
    
    parallel_for("porous HDIV volume resid div(u)",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
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

void porousMixed::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  string bctype = bcs(pnum,cside);
  
  auto basis = wkset->basis_side[unum];
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  
  View_Sc2 nx, ny, nz;
  View_AD2 ux, uy, uz;
  nx = wkset->getScalarField("n[x]");
  
  if (spaceDim == 1) {
    ux = wkset->getSolutionField("u");
  }
  else {
    ux = wkset->getSolutionField("u[x]");
  }
  if (spaceDim > 1) {
    ny = wkset->getScalarField("n[y]");
    uy = wkset->getSolutionField("u[y]");
  }
  if (spaceDim > 2) {
    nz = wkset->getScalarField("n[z]");
    uz = wkset->getSolutionField("u[z]");
  }
  
  Vista bsource;
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (bctype == "Dirichlet" ) {
      bsource = functionManager->evaluate("Dirichlet p " + wkset->sidename,"side ip");
    }
    
  }
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  auto off = subview(wkset->offsets, unum, ALL());
  
  if (bcs(pnum,cside) == "Dirichlet") {
    parallel_for("porous HDIV bndry resid Dirichlet",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        AD src = bsource(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = basis(elem,dof,pt,0)*nx(elem,pt);
          if (dim > 1) {
            vdotn += basis(elem,dof,pt,1)*ny(elem,pt);
          }
          if (dim > 2) {
            vdotn += basis(elem,dof,pt,2)*nz(elem,pt);
          }
          res(elem,off(dof)) += src*vdotn;
        }
      }
    });
  }
  else if (bcs(pnum,cside) == "interface") {
    auto lambda = wkset->getSolutionField("aux "+auxvar);
    parallel_for("porous HDIV boundary resid MS Dirichlet",
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
          res(elem,off(dof)) += lam*vdotn;
        }
      }
    });
  }
   
}


// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void porousMixed::computeFlux() {
  
  int spaceDim = wkset->dimension;
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto uflux = subview(wkset->flux, ALL(), auxpnum, ALL());
    View_Sc2 nx, ny, nz;
    View_AD2 ux, uy, uz;
    if (spaceDim == 1) {
      nx = wkset->getScalarField("n[x]");
      ux = wkset->getSolutionField("u");
      parallel_for("porous HDIV flux ",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<nx.extent(1); pt++) {
          AD udotn = ux(elem,pt)*nx(elem,pt);
          uflux(elem,pt) = udotn;
        }
      });
    }
    else if (spaceDim == 2) {
      nx = wkset->getScalarField("n[x]");
      ux = wkset->getSolutionField("u[x]");
      ny = wkset->getScalarField("n[y]");
      uy = wkset->getSolutionField("u[y]");
      parallel_for("porous HDIV flux ",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<nx.extent(1); pt++) {
          AD udotn = ux(elem,pt)*nx(elem,pt);
          udotn += uy(elem,pt)*ny(elem,pt);
          uflux(elem,pt) = udotn;
        }
      });
    }
    else if (spaceDim == 3) {
      nx = wkset->getScalarField("n[x]");
      ux = wkset->getSolutionField("u[x]");
      ny = wkset->getScalarField("n[y]");
      uy = wkset->getSolutionField("u[y]");
      nz = wkset->getScalarField("n[z]");
      uz = wkset->getSolutionField("u[z]");
      
      parallel_for("porous HDIV flux ",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<nx.extent(1); pt++) {
          AD udotn = ux(elem,pt)*nx(elem,pt);
          udotn += uy(elem,pt)*ny(elem,pt);
          udotn += uz(elem,pt)*nz(elem,pt);
          uflux(elem,pt) = udotn;
        }
      });
    }
    
    
  }
  
}

// ========================================================================================
// ========================================================================================

void porousMixed::setWorkset(Teuchos::RCP<Workset> & wkset_) {

  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  
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

  vector<string> auxvarlist = wkset->aux_varlist;
  
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "p") {
      auxpnum = i;
      auxvar = "p";
    }
    if (auxvarlist[i] == "lambda") {
      auxpnum = i;
      auxvar = "lambda";
    }
    if (auxvarlist[i] == "pbndry") {
      auxpnum = i;
      auxvar = "pbndry";
    }
      
    if (auxvarlist[i] == "u")
      auxunum = i;
  }
}

// ========================================================================================
// ========================================================================================

void porousMixed::updatePerm(View_AD2 Kinv_xx, View_AD2 Kinv_yy, View_AD2 Kinv_zz) {
  
  View_Sc2 data = wkset->extra_data;
  
  parallel_for("porous HDIV update perm",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<Kinv_xx.extent(1); pt++) {
      Kinv_xx(elem,pt) = 1.0/data(elem,0);
      Kinv_yy(elem,pt) = 1.0/data(elem,0);
      Kinv_zz(elem,pt) = 1.0/data(elem,0);
    }
  });
}


void porousMixed::updateKLPerm(View_AD2 KL_Kxx,
                               View_AD2 KL_Kyy, View_AD2 KL_Kzz) {
  
  int spaceDim = wkset->dimension;
  
  bool foundUQ = false;
  auto KLUQcoeffs = wkset->getParameter("KLUQcoeffs",foundUQ);
  size_type prog = 0;
  auto xpts = wkset->getScalarField("x");
  auto indices = KLindices;
  
  if (foundUQ) {
    size_type maxind = std::min(KLUQcoeffs.extent(0),indices.extent(0));
    if (spaceDim == 1) {
      parallel_for("porous KL update perm",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        
        for (size_type k=0; k<maxind; ++k) {
          ScalarT eval = permKLx.getEval(k);
          for (size_type pt=0; pt<xpts.extent(1); ++pt) {
            ScalarT evec = permKLx.getEvec(k,xpts(elem,pt));
            KL_Kxx(elem,pt) += KLUQcoeffs(k)*sqrt(eval)*evec;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      auto ypts = wkset->getScalarField("y");
      parallel_for("porous KL update perm",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        
        for (size_type k=0; k<maxind; ++k) {
          int xind = indices(k,0);
          int yind = indices(k,1);
          ScalarT evalx = permKLx.getEval(xind);
          ScalarT evaly = permKLy.getEval(yind);
          for (size_type pt=0; pt<xpts.extent(1); ++pt) {
            ScalarT evecx = permKLx.getEvec(xind,xpts(elem,pt));
            ScalarT evecy = permKLy.getEvec(yind,ypts(elem,pt));
            KL_Kxx(elem,pt) += KLUQcoeffs(k)*sqrt(evalx*evaly)*evecx*evecy;
            KL_Kyy(elem,pt) += KLUQcoeffs(k)*sqrt(evalx*evaly)*evecx*evecy;
          }
        }
      });
    }
    else if (spaceDim == 3) {
      auto ypts = wkset->getScalarField("y");
      auto zpts = wkset->getScalarField("z");
      parallel_for("porous KL update perm",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        
        for (size_type k=0; k<maxind; ++k) {
          int xind = indices(k,0);
          int yind = indices(k,1);
          int zind = indices(k,2);
          ScalarT evalx = permKLx.getEval(xind);
          ScalarT evaly = permKLy.getEval(yind);
          ScalarT evalz = permKLy.getEval(zind);
          for (size_type pt=0; pt<xpts.extent(1); ++pt) {
            ScalarT evecx = permKLx.getEvec(xind,xpts(elem,pt));
            ScalarT evecy = permKLy.getEvec(yind,ypts(elem,pt));
            ScalarT evecz = permKLz.getEvec(zind,zpts(elem,pt));
            KL_Kxx(elem,pt) += KLUQcoeffs(k)*sqrt(evalx*evaly*evalz)*evecx*evecy*evecz;
            KL_Kyy(elem,pt) += KLUQcoeffs(k)*sqrt(evalx*evaly*evalz)*evecx*evecy*evecz;
            KL_Kxx(elem,pt) += KLUQcoeffs(k)*sqrt(evalx*evaly*evalz)*evecx*evecy*evecz;
          }
        }
      });
    }
    
    prog += KLUQcoeffs.extent(0);
  }
  
  bool foundStoch = false;
  auto KLStochcoeffs = wkset->getParameter("KLStochcoeffs",foundStoch);
  
  if (foundStoch) {
    
    size_type maxind = std::min(indices.extent(0), prog+KLStochcoeffs.extent(0));
    if (spaceDim == 1) {
      parallel_for("porous KL update perm",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        
        for (size_type k=prog; k<maxind; ++k) {
          ScalarT eval = permKLx.getEval(k);
          for (size_type pt=0; pt<xpts.extent(1); ++pt) {
            ScalarT evec = permKLx.getEvec(k,xpts(elem,pt));
            KL_Kxx(elem,pt) += KLStochcoeffs(k-prog)*sqrt(eval)*evec;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      auto ypts = wkset->getScalarField("y");
      
      parallel_for("porous KL update perm",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        
        for (size_type k=prog; k<maxind; ++k) {
          int xind = indices(k,0);
          int yind = indices(k,1);
          ScalarT evalx = permKLx.getEval(xind);
          ScalarT evaly = permKLy.getEval(yind);
          for (size_type pt=0; pt<xpts.extent(1); ++pt) {
            ScalarT evecx = permKLx.getEvec(xind,xpts(elem,pt));
            ScalarT evecy = permKLy.getEvec(yind,ypts(elem,pt));
            KL_Kxx(elem,pt) += KLStochcoeffs(k-prog)*sqrt(evalx*evaly)*evecx*evecy;
            KL_Kyy(elem,pt) += KLStochcoeffs(k-prog)*sqrt(evalx*evaly)*evecx*evecy;
          }
        }
      });
    }
    else if (spaceDim == 3) {
      auto ypts = wkset->getScalarField("y");
      auto zpts = wkset->getScalarField("z");
      parallel_for("porous KL update perm",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        
        for (size_type k=prog; k<maxind; ++k) {
          int xind = indices(k,0);
          int yind = indices(k,1);
          int zind = indices(k,2);
          ScalarT evalx = permKLx.getEval(xind);
          ScalarT evaly = permKLy.getEval(yind);
          ScalarT evalz = permKLz.getEval(zind);
          for (size_type pt=0; pt<xpts.extent(1); ++pt) {
            ScalarT evecx = permKLx.getEvec(xind,xpts(elem,pt));
            ScalarT evecy = permKLy.getEvec(yind,ypts(elem,pt));
            ScalarT evecz = permKLz.getEvec(zind,zpts(elem,pt));
            KL_Kxx(elem,pt) += KLStochcoeffs(k-prog)*sqrt(evalx*evaly*evalz)*evecx*evecy*evecz;
            KL_Kyy(elem,pt) += KLStochcoeffs(k-prog)*sqrt(evalx*evaly*evalz)*evecx*evecy*evecz;
            KL_Kzz(elem,pt) += KLStochcoeffs(k-prog)*sqrt(evalx*evaly*evalz)*evecx*evecy*evecz;
          }
        }
      });
    }
  }
}


// ========================================================================================
// ========================================================================================

std::vector<string> porousMixed::getDerivedNames() {
  std::vector<string> derived;
  derived.push_back("permeability_x");
  derived.push_back("permeability_y");
  derived.push_back("permeability_z");
  return derived;
}

// ========================================================================================
// ========================================================================================

std::vector<View_AD2> porousMixed::getDerivedValues() {
  std::vector<View_AD2> derived;
  
  View_AD2 K_xx, K_yy, K_zz;
  
  Vista Kinv_xx, Kinv_yy, Kinv_zz;
  int spaceDim = wkset->dimension;
  
  // First compute Kinvxx, Kinvyy, Kinvzz
  
  auto wts = wkset->wts;
  {
    if (usePermData) {
      View_AD2 view_Kinv_xx("K inverse xx",wts.extent(0),wts.extent(1));
      View_AD2 view_Kinv_yy("K inverse yy",wts.extent(0),wts.extent(1));
      View_AD2 view_Kinv_zz("K inverse zz",wts.extent(0),wts.extent(1));
      this->updatePerm(view_Kinv_xx, view_Kinv_yy, view_Kinv_zz);
      Kinv_xx = Vista(view_Kinv_xx);
      Kinv_yy = Vista(view_Kinv_yy);
      Kinv_zz = Vista(view_Kinv_zz);
    }
    else {
      Kinv_xx = functionManager->evaluate("Kinv_xx","ip");
      Kinv_yy = functionManager->evaluate("Kinv_yy","ip");
      Kinv_zz = functionManager->evaluate("Kinv_zz","ip");
    }
    
    if (useKL) {
      View_AD2 KL_Kxx("KL K xx",wts.extent(0),wts.extent(1));
      View_AD2 KL_Kyy("KL K yy",wts.extent(0),wts.extent(1));
      View_AD2 KL_Kzz("KL K zz",wts.extent(0),wts.extent(1));
      this->updateKLPerm(KL_Kxx, KL_Kyy, KL_Kzz);
      
      View_AD2 new_Kxx("new K xx",wts.extent(0),wts.extent(1));
      View_AD2 new_Kyy("new K yy",wts.extent(0),wts.extent(1));
      View_AD2 new_Kzz("new K zz",wts.extent(0),wts.extent(1));
      
      parallel_for("porous gdv perm",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<KL_Kxx.extent(1); ++pt) {
          new_Kxx(elem,pt) = Kinv_xx(elem,pt)/exp(KL_Kxx(elem,pt));
        }
        if (spaceDim > 1) {
          for (size_type pt=0; pt<KL_Kyy.extent(1); ++pt) {
            new_Kyy(elem,pt) = Kinv_yy(elem,pt)/exp(KL_Kyy(elem,pt));
          }
        }
        if (spaceDim > 2) {
          for (size_type pt=0; pt<KL_Kzz.extent(1); ++pt) {
            new_Kzz(elem,pt) = Kinv_zz(elem,pt)/exp(KL_Kzz(elem,pt));
          }
        }
        
      });
      
      Kinv_xx = Vista(new_Kxx);
      Kinv_yy = Vista(new_Kyy);
      Kinv_zz = Vista(new_Kzz);
    }
    
  }
  
  K_xx = View_AD2("K xx",wts.extent(0),wts.extent(1));
  K_yy = View_AD2("K yy",wts.extent(0),wts.extent(1));
  K_zz = View_AD2("K zz",wts.extent(0),wts.extent(1));
    
  parallel_for("porous gdv perm 2",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<K_xx.extent(1); ++pt) {
      K_xx(elem,pt) = 1.0/Kinv_xx(elem,pt);
    }
    if (spaceDim > 1) {
      for (size_type pt=0; pt<K_yy.extent(1); ++pt) {
        K_yy(elem,pt) = 1.0/Kinv_yy(elem,pt);
      }
    }
    if (spaceDim > 2) {
      for (size_type pt=0; pt<K_zz.extent(1); ++pt) {
        K_zz(elem,pt) = 1.0/Kinv_zz(elem,pt);
      }
    }
  });
  
  derived.push_back(K_xx);
  derived.push_back(K_yy);
  derived.push_back(K_zz);
  
  return derived;
}
