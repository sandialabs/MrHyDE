/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "porousHDIV_weakGalerkin.hpp"

porousHDIV_WG::porousHDIV_WG(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  label = "porousHDIV-WeakGalerkin";
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  include_face = settings->sublist("Physics").get<bool>("Include face terms","true");
  
  if (settings->sublist("Physics").isSublist("Active variables")) {
    if (settings->sublist("Physics").sublist("Active variables").isParameter("pint")) {
      myvars.push_back("pint");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("p","HVOL"));
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("pbndry")) {
      myvars.push_back("pbndry");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("pbndry","HFACE")); // TODO: turn into HFACE-DG
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("u")) {
      myvars.push_back("u");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("u","HDIV-DG"));
    }
    if (settings->sublist("Physics").sublist("Active variables").isParameter("t")) {
      myvars.push_back("t");
      mybasistypes.push_back(settings->sublist("Physics").sublist("Active variables").get<string>("t","HDIV-DG"));
    }
  }
  else {
    myvars.push_back("pint");
    myvars.push_back("pbndry");
    myvars.push_back("u");
    myvars.push_back("t");
    mybasistypes.push_back("HVOL");
    mybasistypes.push_back("HFACE"); // TODO: turn into HFACE-DG
    mybasistypes.push_back("HDIV-DG");
    mybasistypes.push_back("HDIV-DG");
  }
  
  usePermData = settings->sublist("Physics").get<bool>("use permeability data",false);

  dxnum = 0;
  dynum = 0;
  dznum = 0;
  
}

// ========================================================================================
// ========================================================================================

void porousHDIV_WG::defineFunctions(Teuchos::RCP<Teuchos::ParameterList> & settings,
                                    Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;

  // Functions
  Teuchos::ParameterList fs = settings->sublist("Functions");
  
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

void porousHDIV_WG::volumeResidual() {
  
  // NOTES:
  // 1. basis and basis_grad already include the integration weights
  
  int resindex;
  int pint_basis = wkset->usebasis[pintnum];
  int u_basis = wkset->usebasis[unum];
  int t_basis = wkset->usebasis[tnum];
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source = functionManager->evaluate("source","ip");
    //perm = functionManager->evaluate("perm","ip");
    if (usePermData) {
      this->updatePerm();
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
  
  wts = wkset->wts;
  
  {
    basis = wkset->basis[u_basis];
    basis_div = wkset->basis_div[u_basis];
    
    auto usol = Kokkos::subview(sol,Kokkos::ALL(), unum, Kokkos::ALL(), Kokkos::ALL());
    auto pintsol = Kokkos::subview(sol,Kokkos::ALL(), pintnum, Kokkos::ALL(), 0);
    auto off = Kokkos::subview(offsets,unum, Kokkos::ALL());
    
    // (u,v) + (p_0,div(v))
    if (spaceDim == 1) {
      parallel_for("porous WG volume resid: u 1D",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_t pt=0; pt<sol.extent(2); pt++ ) {
          
          AD pint = pintsol(elem,pt)*wts(elem,pt);
          AD ux = usol(elem,pt,0)*wts(elem,pt);
          
          for (size_t dof=0; dof<basis.extent(1); dof++ ) {
            
            ScalarT vx = basis(elem,dof,pt,0);
            
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += (ux*vx) + pint*divv;
            
          }
        }
        
      });
    }
    else if (spaceDim == 2) {
      parallel_for("porous WG volume resid: u 2D",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_t pt=0; pt<sol.extent(2); pt++ ) {
          
          AD pint = pintsol(elem,pt)*wts(elem,pt);
          AD ux = usol(elem,pt,0)*wts(elem,pt);
          AD uy = usol(elem,pt,1)*wts(elem,pt);
          
          for (size_t dof=0; dof<basis.extent(1); dof++ ) {
            
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT vy = basis(elem,dof,pt,1);
            
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += (ux*vx+uy*vy) + pint*divv;
            
          }
        }
        
      });
    }
    else {
      parallel_for("porous WG volume resid: u 3D",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_t pt=0; pt<sol.extent(2); pt++ ) {
          
          AD pint = pintsol(elem,pt)*wts(elem,pt);
          AD ux = usol(elem,pt,0)*wts(elem,pt);
          AD uy = usol(elem,pt,1)*wts(elem,pt);
          AD uz = usol(elem,pt,2)*wts(elem,pt);
          
          for (size_t dof=0; dof<basis.extent(1); dof++ ) {
            
            ScalarT vx = basis(elem,dof,pt,0);
            ScalarT vy = basis(elem,dof,pt,1);
            ScalarT vz = basis(elem,dof,pt,2);
            
            ScalarT divv = basis_div(elem,dof,pt);
            res(elem,off(dof)) += (ux*vx+uy*vy+uz*vz) + pint*divv;
            
          }
        }
        
      });
    }
    
  }
  
  {
    //  (Ku,s) + (t,s)
    
    basis = wkset->basis[t_basis];
    basis_div = wkset->basis_div[t_basis];
    
    auto usol = Kokkos::subview(sol, Kokkos::ALL(), unum, Kokkos::ALL(), Kokkos::ALL());
    auto tsol = Kokkos::subview(sol, Kokkos::ALL(), tnum, Kokkos::ALL(), Kokkos::ALL());
    auto off = Kokkos::subview(offsets, tnum, Kokkos::ALL());
    
    if (spaceDim == 1) {
      parallel_for("porous WG volume resid t 1D",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_t pt=0; pt<basis.extent(2); pt++ ) {
          AD Kux = perm(elem,pt)*usol(elem,pt,0)*wts(elem,pt);
          AD tx = tsol(elem,pt,0)*wts(elem,pt);
          
          AD dx = Kux + tx;
          
          for (size_t dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT sx = basis(elem,dof,pt,0);
            res(elem,off(dof)) += dx*sx;
          }
        }
        
      });
    }
    else if (spaceDim == 2) {
      parallel_for("porous WG volume resid t 2D",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_t pt=0; pt<basis.extent(2); pt++ ) {
          AD Kux = perm(elem,pt)*usol(elem,pt,0)*wts(elem,pt);
          AD tx = tsol(elem,pt,0)*wts(elem,pt);
          AD Kuy = perm(elem,pt)*usol(elem,pt,1)*wts(elem,pt);
          AD ty = tsol(elem,pt,1)*wts(elem,pt);
          
          AD dx = Kux + tx;
          AD dy = Kuy + ty;
          
          for (size_t dof=0; dof<basis.extent(1); dof++ ) {
            ScalarT sx = basis(elem,dof,pt,0);
            ScalarT sy = basis(elem,dof,pt,1);
            res(elem,off(dof)) += dx*sx + dy*sy;
          }
        }
        
      });
    }
    else {
      parallel_for("porous WG volume resid t 3D",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_t pt=0; pt<basis.extent(2); pt++ ) {
          AD Kux = perm(elem,pt)*usol(elem,pt,0)*wts(elem,pt);
          AD tx = tsol(elem,pt,0)*wts(elem,pt);
          AD Kuy = perm(elem,pt)*usol(elem,pt,1)*wts(elem,pt);
          AD ty = tsol(elem,pt,1)*wts(elem,pt);
          AD Kuz = perm(elem,pt)*usol(elem,pt,2)*wts(elem,pt);
          AD tz = tsol(elem,pt,2)*wts(elem,pt);
          
          AD dx = Kux + tx;
          AD dy = Kuy + ty;
          AD dz = Kuz + tz;
          
          for (size_t dof=0; dof<basis.extent(1); dof++ ) {
            
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
    basis = wkset->basis[pint_basis];
    auto tdiv = Kokkos::subview(sol_div, Kokkos::ALL(), tnum, Kokkos::ALL());
    auto off = Kokkos::subview(offsets, pintnum, Kokkos::ALL());
    
    parallel_for("porous WG volume resid div(t)",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      
      for (size_t pt=0; pt<basis.extent(2); pt++ ) {
        AD divt = tdiv(elem,pt)*wts(elem,pt);
        AD S = source(elem,pt)*wts(elem,pt);
        AD tdiff = divt-S;
        for (size_t dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT qint = basis(elem,dof,pt);
          res(elem,off(dof)) += tdiff*qint;
        }
      }
      
    });
  }
}


// ========================================================================================
// ========================================================================================

void porousHDIV_WG::boundaryResidual() {
  
  bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  int sidetype;
  sidetype = bcs(pintnum,cside);
  
  // TMW: changed this from t_basis to u_basis (check on this)
  int u_basis = wkset->usebasis[unum];
  
  basis = wkset->basis_side[u_basis];
  
  {
    Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
    
    if (sidetype == 1 ) {
      bsource = functionManager->evaluate("Dirichlet pbndry " + wkset->sidename,"side ip");
    }
    
  }
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;
  wts = wkset->wts_side;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  if (bcs(pintnum,cside) == 1) {
    auto off = Kokkos::subview(offsets, unum, Kokkos::ALL());
    parallel_for("porous WG bndry resid Dirichlet",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<basis.extent(2); pt++ ) {
        AD S = bsource(elem,pt)*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = 0.0;
          for (int dim=0; dim<normals.extent(2); dim++) {
            vdotn += basis(elem,dof,pt,dim)*normals(elem,pt,dim);
          }
          res(elem,off(dof)) -= S*vdotn;
        }
      }
    });
  }
  else if (bcs(pintnum,cside) == 5) {
    auto off = Kokkos::subview(offsets, unum, Kokkos::ALL());
    auto lambda = Kokkos::subview(aux_side, Kokkos::ALL(), auxpbndrynum, Kokkos::ALL());
    parallel_for("porous WG bndry resid MS Dirichlet",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (int pt=0; pt<basis.extent(2); pt++ ) {
        AD lam = lambda(elem,pt)*wts(elem,pt);
        for (int dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = 0.0;
          for (int dim=0; dim<normals.extent(2); dim++) {
            vdotn += basis(elem,dof,pt,dim)*normals(elem,pt,dim);
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

void porousHDIV_WG::faceResidual() {
  
  int pbndry_basis = wkset->usebasis[pbndrynum];
  int u_basis = wkset->usebasis[unum];
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;
  wts = wkset->wts_side;
  
  Teuchos::TimeMonitor localtime(*boundaryResidualFill);
  
  {
    // include <pbndry, v \cdot n> in velocity equation
    basis = wkset->basis_face[u_basis];
    auto off = Kokkos::subview(offsets, unum, Kokkos::ALL());
    auto pbndry = Kokkos::subview(sol_face, Kokkos::ALL(), pbndrynum, Kokkos::ALL(), 0);
    
    parallel_for("porous WG face resid pbndry",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_t pt=0; pt<basis.extent(2); pt++ ) {
        AD p = pbndry(elem,pt)*wts(elem,pt);
        for (size_t dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT vdotn = 0.0;
          for (int dim=0; dim<normals.extent(2); dim++) {
            vdotn += basis(elem,dof,pt,dim)*normals(elem,pt,dim);
          }
          
          //int resindex = offsets(unum,i);
          res(elem,off(dof)) -= p*vdotn;
        }
      }
    });
  }
  
  {
    // include -<t \cdot n, qbndry> in interface equation
    AD tx = 0.0, ty = 0.0, tz = 0.0;
    basis = wkset->basis_face[pbndry_basis];
    auto off = Kokkos::subview(offsets, pbndrynum, Kokkos::ALL());
    auto tsol = Kokkos::subview(sol_face, Kokkos::ALL(), tnum, Kokkos::ALL(), Kokkos::ALL());
    // TMW: previous code used u instead of t
    
    parallel_for("porous WG face resid t dot n",RangePolicy<AssemblyExec>(0,res.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_t pt=0; pt<basis.extent(2); pt++ ) {
        AD tdotn = 0.0;
        for (int dim=0; dim<normals.extent(2); dim++) {
          tdotn += tsol(elem,pt,dim)*normals(elem,pt,dim);
        }
        tdotn *= wts(elem,pt);
        
        for (size_t dof=0; dof<basis.extent(1); dof++ ) {
          ScalarT qbndry = basis(elem,dof,pt);
          res(elem,off(dof)) -= tdotn*qbndry;
        }
      }
    });
  }
}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void porousHDIV_WG::computeFlux() {
  
  // Since normals get recomputed often, this needs to be reset
  normals = wkset->normals;

  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    auto tsol = Kokkos::subview(sol_side, Kokkos::ALL(), tnum, Kokkos::ALL(), Kokkos::ALL());
    auto mflux = Kokkos::subview(flux, Kokkos::ALL(), auxpbndrynum, Kokkos::ALL());
    // TMW: previous code used u instead of t
    
    parallel_for("porous WG flux",RangePolicy<AssemblyExec>(0,mflux.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      
      for (size_t pt=0; pt<mflux.extent(1); pt++) {
        
        AD tdotn = 0.0;
        for (int dim=0; dim<normals.extent(2); dim++) {
          tdotn += tsol(elem,pt,dim)*normals(elem,pt,dim);
        }
        
        mflux(elem,pt) = -tdotn;
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

void porousHDIV_WG::setVars(std::vector<string> & varlist) {
  
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
}

// ========================================================================================
// ========================================================================================

void porousHDIV_WG::setAuxVars(std::vector<string> & auxvarlist) {

  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "pbndry")
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

void porousHDIV_WG::updatePerm() {
  
  wts = wkset->wts;
  perm = Kokkos::View<AD**,AssemblyDevice>("permeability",wts.extent(0),wts.extent(1));
  Kokkos::View<ScalarT**,AssemblyDevice> data = wkset->extra_data;
  
  parallel_for("porous WG update perm",RangePolicy<AssemblyExec>(0,perm.extent(0)), KOKKOS_LAMBDA (const int elem ) {
    for (int pt=0; pt<perm.extent(1); pt++) {
      perm(elem,pt) = data(elem,0);
    }
  });
}

