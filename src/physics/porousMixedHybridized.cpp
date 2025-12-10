/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "porousMixedHybridized.hpp"
using namespace MrHyDE;

template<class EvalT>
porousMixedHybrid<EvalT>::porousMixedHybrid(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "porousMixedHybrid";
  include_face = true;
  
  if (settings.isSublist("Active variables")) {
    if (settings.sublist("Active variables").isParameter("p")) {
      myvars.push_back("p");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("p","HVOL"));
    }
    if (settings.sublist("Active variables").isParameter("u")) {
      myvars.push_back("u");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("u","HDIV-DG"));
    }
    if (settings.sublist("Active variables").isParameter("lambda")) {
      myvars.push_back("lambda");
      mybasistypes.push_back(settings.sublist("Active variables").get<string>("lambda","HFACE"));
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
  usePermData = settings.get<bool>("use permeability data",false);
  
  dxnum = 0;
  dynum = 0;
  dznum = 0;
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void porousMixedHybrid<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                        Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  
  functionManager->addFunction("source",fs.get<string>("source","0.0"),"ip");
  functionManager->addFunction("Kinv_xx",fs.get<string>("Kinv_xx","1.0"),"ip");
  functionManager->addFunction("Kinv_yy",fs.get<string>("Kinv_yy","1.0"),"ip");
  functionManager->addFunction("Kinv_zz",fs.get<string>("Kinv_zz","1.0"),"ip");
    
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void porousMixedHybrid<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  int p_basis = wkset->usebasis[pnum];
  int u_basis = wkset->usebasis[unum];
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  Vista<EvalT> source, bsource, Kinv_xx, Kinv_yy, Kinv_zz;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    if (usePermData) {
      auto wts = wkset->wts;
      View_EvalT2 view_Kinv_xx("K inverse xx",wts.extent(0),wts.extent(1));
      View_EvalT2 view_Kinv_yy("K inverse yy",wts.extent(0),wts.extent(1));
      View_EvalT2 view_Kinv_zz("K inverse zz",wts.extent(0),wts.extent(1));
      this->updatePerm(view_Kinv_xx, view_Kinv_yy, view_Kinv_zz);
      Kinv_xx = Vista<EvalT>(view_Kinv_xx);
      Kinv_yy = Vista<EvalT>(view_Kinv_yy);
      Kinv_zz = Vista<EvalT>(view_Kinv_zz);
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
    auto psol = wkset->getSolutionField("p");
    auto off = subview(wkset->offsets, unum, ALL());
    
    if (spaceDim == 1) { // easier to place conditional here than on device
      auto ux = wkset->getSolutionField("u[x]");
      parallel_for("porous HDIV-HY volume resid u 1D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   MRHYDE_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT p = psol(elem,pt)*wts(elem,pt);
          EvalT Kiux = Kinv_xx(elem,pt)*ux(elem,pt)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += Kiux*vx - p*divv;
          }
        }
      });
    }
    else if (spaceDim == 2) {
      auto ux = wkset->getSolutionField("u[x]");
      auto uy = wkset->getSolutionField("u[y]");
      parallel_for("porous HDIV-HY volume resid u 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   MRHYDE_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT p = psol(elem,pt)*wts(elem,pt);
          EvalT Kiux = Kinv_xx(elem,pt)*ux(elem,pt)*wts(elem,pt);
          EvalT Kiuy = Kinv_yy(elem,pt)*uy(elem,pt)*wts(elem,pt);
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
      parallel_for("porous HDIV-HY volume resid u 3D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   MRHYDE_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT p = psol(elem,pt)*wts(elem,pt);
          EvalT Kiux = Kinv_xx(elem,pt)*ux(elem,pt)*wts(elem,pt);
          EvalT Kiuy = Kinv_yy(elem,pt)*uy(elem,pt)*wts(elem,pt);
          EvalT Kiuz = Kinv_zz(elem,pt)*uz(elem,pt)*wts(elem,pt);
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
    auto udiv = wkset->getSolutionField("div(u)");
    auto off = subview(wkset->offsets, pnum, ALL());
    
    parallel_for("porous HDIV-HY volume resid div(u)",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT divu = udiv(elem,pt)*wts(elem,pt);
        EvalT src = source(elem,pt)*wts(elem,pt);
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

template<class EvalT>
void porousMixedHybrid<EvalT>::boundaryResidual() {
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  string bctype = bcs(pnum,cside);
  
  int u_basis = wkset->usebasis[unum];
  
  auto basis = wkset->basis_side[u_basis];
  View_Sc2 nx, ny, nz;
  View_EvalT2 ux, uy, uz;
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
  
  Vista<EvalT> bsource;
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (bctype == "Dirichlet" ) {
      bsource = functionManager->evaluate("Dirichlet p " + wkset->sidename,"side ip");
    }
    
  }
  
  // Since normals get recomputed often, this needs to be reset
  
  auto wts = wkset->wts_side;
  auto res = wkset->res;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  auto off = subview(wkset->offsets, unum, ALL());
  
  if (bcs(pnum,cside) == "Dirichlet") {
    parallel_for("porous HDIV-HY bndry resid Dirichlet",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT src = bsource(elem,pt)*wts(elem,pt);
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
    auto lambda = wkset->getSolutionField("aux p");
    parallel_for("porous HDIV-HY bndry resid MS Dirichlet",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT lam = lambda(elem,pt)*wts(elem,pt);
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
// The edge (2D) and face (3D) contributions to the residual
// ========================================================================================

template<class EvalT>
void porousMixedHybrid<EvalT>::faceResidual() {
  
  int spaceDim = wkset->dimension;
  int lambda_basis = wkset->usebasis[lambdanum];
  int u_basis = wkset->usebasis[unum];
  
  // Since normals get recomputed often, this needs to be reset
  auto res = wkset->res;
  auto wts = wkset->wts_side;
  View_Sc2 nx, ny, nz;
  View_EvalT2 ux, uy, uz;
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
  
  {
    // include <lambda, v \cdot n> in velocity equation
    auto basis = wkset->basis_side[u_basis];
    auto off = subview(wkset->offsets, unum, ALL());
    auto lambda = wkset->getSolutionField("lambda");
    
    parallel_for("porous HDIV-HY face resid lambda",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT lam = lambda(elem,pt)*wts(elem,pt);
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
  
  {
    // include -<u \cdot n, mu> in interface equation
    auto basis = wkset->basis_side[lambda_basis];
    auto ubasis = wkset->basis_side[u_basis];
    auto off = subview(wkset->offsets, lambdanum, ALL());
    
    parallel_for("porous HDIV-HY face resid u dot n",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      size_type dim = ubasis.extent(3);
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT udotn = ux(elem,pt)*nx(elem,pt);
        if (dim > 1) {
          udotn += uy(elem,pt)*ny(elem,pt);
        }
        if (dim> 2) {
          udotn += uz(elem,pt)*nz(elem,pt);
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

template<class EvalT>
void porousMixedHybrid<EvalT>::computeFlux() {
  
  int spaceDim = wkset->dimension;
  // Just need the basis for the number of active elements (any side basis will do)
  auto basis = wkset->basis_side[wkset->usebasis[unum]];
 
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    auto uflux = subview(wkset->flux, ALL(), auxpnum, ALL());
    View_Sc2 nx, ny, nz;
    View_EvalT2 ux, uy, uz;
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
    
    parallel_for("porous HDIV flux ",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      size_type dim = basis.extent(3);
      for (size_type pt=0; pt<nx.extent(1); pt++) {
        EvalT udotn = ux(elem,pt)*nx(elem,pt);
        if (dim> 1) {
          udotn += uy(elem,pt)*ny(elem,pt);
        }
        if (dim > 2) {
          udotn += uz(elem,pt)*nz(elem,pt);
        }
        uflux(elem,pt) = udotn;
      }
    });
  }
  
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void porousMixedHybrid<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  
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

  vector<string> auxvarlist = wkset->aux_varlist;
  
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "p")
      auxpnum = i; // hard-coded for now
    if (auxvarlist[i] == "u")
      auxunum = i;
    if (auxvarlist[i] == "lambda")
      auxlambdanum = i;
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void porousMixedHybrid<EvalT>::updatePerm(View_EvalT2 Kinv_xx, View_EvalT2 Kinv_yy, View_EvalT2 Kinv_zz) {
  
  View_Sc2 data = wkset->extra_data;
  
  parallel_for("porous HDIV update perm",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               MRHYDE_LAMBDA (const int elem ) {
    for (size_type pt=0; pt<Kinv_xx.extent(1); pt++) {
      Kinv_xx(elem,pt) = data(elem,0);
      Kinv_yy(elem,pt) = data(elem,0);
      Kinv_zz(elem,pt) = data(elem,0);
    }
  });
}

//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::porousMixedHybrid<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::porousMixedHybrid<AD>;

// Standard built-in types
template class MrHyDE::porousMixedHybrid<AD2>;
template class MrHyDE::porousMixedHybrid<AD4>;
template class MrHyDE::porousMixedHybrid<AD8>;
template class MrHyDE::porousMixedHybrid<AD16>;
template class MrHyDE::porousMixedHybrid<AD18>;
template class MrHyDE::porousMixedHybrid<AD24>;
template class MrHyDE::porousMixedHybrid<AD32>;
#endif
