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

#include "linearelasticity.hpp"
#include "CrystalElasticity.hpp"
#include <string>
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

linearelasticity::linearelasticity(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  
  label = "linearelasticity";
  
  spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  
  if (spaceDim == 1) {
    myvars = {"dx"};
    mybasistypes = {"HGRAD"};
  }
  else if (spaceDim == 2) {
    myvars = {"dx","dy"};
    mybasistypes = {"HGRAD","HGRAD"};
  }
  else if (spaceDim == 3) {
    myvars = {"dx","dy","dz"};
    mybasistypes = {"HGRAD","HGRAD","HGRAD"};
  }
  
  useCE = settings->sublist("Physics").get<bool>("use crystal elasticity",false);
  int numElem = settings->sublist("Solver").get<int>("workset size",1);
  if (useCE) {
    crystalelast = Teuchos::rcp(new CrystalElastic(settings, numElem));
  }
  
  
  incplanestress = settings->sublist("Physics").get<bool>("incplanestress",false);
  useLame = settings->sublist("Physics").get<bool>("use Lame parameters",true);
  addBiot = settings->sublist("Physics").get<bool>("Biot",false);
  
  modelparams = Kokkos::View<ScalarT*,AssemblyDevice>("parameters for LE",5); 
  auto modelparams_host = Kokkos::create_mirror_view(modelparams); 
 
  modelparams_host(0) = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
  modelparams_host(1) = settings->sublist("Physics").get<ScalarT>("penalty",10.0);
  modelparams_host(2) = settings->sublist("Physics").get<ScalarT>("Biot alpha",0.0);
  modelparams_host(3) = settings->sublist("Physics").get<ScalarT>("T_ambient",0.0);
  modelparams_host(4) = settings->sublist("Physics").get<ScalarT>("alpha_T",1.0e-6);
  
  Kokkos::deep_copy(modelparams, modelparams_host);
}

// ========================================================================================
// ========================================================================================

void linearelasticity::defineFunctions(Teuchos::ParameterList & fs,
                                       Teuchos::RCP<FunctionManager> & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("lambda",fs.get<string>("lambda","1.0"),"ip");
  functionManager->addFunction("mu",fs.get<string>("mu","0.5"),"ip");
  functionManager->addFunction("source dx",fs.get<string>("source dx","0.0"),"ip");
  functionManager->addFunction("source dy",fs.get<string>("source dy","0.0"),"ip");
  functionManager->addFunction("source dz",fs.get<string>("source dz","0.0"),"ip");
  functionManager->addFunction("lambda",fs.get<string>("lambda","1.0"),"side ip");
  functionManager->addFunction("mu",fs.get<string>("mu","0.5"),"side ip");
  
  stress_vol = Kokkos::View<AD****,AssemblyDevice>("stress tensor",
                                                   functionManager->numElem,
                                                   functionManager->numip,
                                                   spaceDim, spaceDim);
  
  stress_side = Kokkos::View<AD****,AssemblyDevice>("stress tensor",
                                                    functionManager->numElem,
                                                    functionManager->numip_side,
                                                    spaceDim, spaceDim);
  
}
// ========================================================================================
// ========================================================================================

void linearelasticity::volumeResidual() {
  
  FDATA lambda, mu, source_dx, source_dy, source_dz;
  
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    source_dx = functionManager->evaluate("source dx","ip");
    if (spaceDim > 1) {
      source_dy = functionManager->evaluate("source dy","ip");
    }
    if (spaceDim > 2) {
      source_dz = functionManager->evaluate("source dz","ip");
    }
    lambda = functionManager->evaluate("lambda","ip");
    mu = functionManager->evaluate("mu","ip");
  }
  
  // fills in stress tensor
  this->computeStress(lambda, mu, false);
  auto stress = stress_vol;
  
  Teuchos::TimeMonitor localtime(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;
  
  if (spaceDim == 1) {
    int dx_basis = wkset->usebasis[dx_num];
    auto basis = wkset->basis[dx_basis];
    auto basis_grad = wkset->basis_grad[dx_basis];
    auto off = Kokkos::subview( offsets, dx_num, Kokkos::ALL());
    
    parallel_for("LE volume resid 1D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += (stress(elem,pt,0,0)*basis_grad(elem,dof,pt,0) - source_dx(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
        }
      }
    });
  }
  else if (spaceDim == 2) {
    
    {
      // first equation
      int dx_basis = wkset->usebasis[dx_num];
      auto basis = wkset->basis[dx_basis];
      auto basis_grad = wkset->basis_grad[dx_basis];
      auto off = Kokkos::subview( offsets, dx_num, Kokkos::ALL());
      
      parallel_for("LE ux volume resid 2D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (stress(elem,pt,0,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,0,1)*basis_grad(elem,dof,pt,1) - source_dx(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
    
    {
      // second equation
      int dy_basis = wkset->usebasis[dy_num];
      auto basis = wkset->basis[dy_basis];
      auto basis_grad = wkset->basis_grad[dy_basis];
      auto off = Kokkos::subview( offsets, dy_num, Kokkos::ALL());
      
      parallel_for("LE uy volume resid 2D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (stress(elem,pt,1,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,1,1)*basis_grad(elem,dof,pt,1) - source_dy(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
  }
  else if (spaceDim == 3) {
    
    // first equation
    {
      int dx_basis = wkset->usebasis[dx_num];
      auto basis = wkset->basis[dx_basis];
      auto basis_grad = wkset->basis_grad[dx_basis];
      auto off = Kokkos::subview( offsets, dx_num, Kokkos::ALL());
      
      parallel_for("LE ux volume resid 3D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (stress(elem,pt,0,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,0,1)*basis_grad(elem,dof,pt,1) + stress(elem,pt,0,2)*basis_grad(elem,dof,pt,2) - source_dx(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
    
    // second equation
    {
      int dy_basis = wkset->usebasis[dy_num];
      auto basis = wkset->basis[dy_basis];
      auto basis_grad = wkset->basis_grad[dy_basis];
      auto off = Kokkos::subview( offsets, dy_num, Kokkos::ALL());
      
      parallel_for("LE uy volume resid 3D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (stress(elem,pt,1,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,1,1)*basis_grad(elem,dof,pt,1) + stress(elem,pt,1,2)*basis_grad(elem,dof,pt,2) - source_dy(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
    
    // third equation
    {
      int dz_basis = wkset->usebasis[dz_num];
      auto basis = wkset->basis[dz_basis];
      auto basis_grad = wkset->basis_grad[dz_basis];
      auto off = Kokkos::subview( offsets, dz_num, Kokkos::ALL());
      
      parallel_for("LE uz volume resid 3D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (stress(elem,pt,2,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,2,1)*basis_grad(elem,dof,pt,1) + stress(elem,pt,2,2)*basis_grad(elem,dof,pt,2) - source_dz(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
  }
  //KokkosTools::print(wkset->res);
}

// ========================================================================================
// ========================================================================================

void linearelasticity::boundaryResidual() {
  
  bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  
  //TMW : will be an error using formparam \neq 1.0 with adjoints
  //ScalarT sf = formparam;
  //if (wkset->isAdjoint) {
  //  sf = 1.0;
  //}
  
  int dy_sidetype = 0;
  int dz_sidetype = 0;
  int dx_sidetype = bcs(dx_num,cside);
  if (spaceDim > 1) {
    dy_sidetype = bcs(dy_num,cside);
  }
  if (spaceDim > 2) {
    dz_sidetype = bcs(dz_num,cside);
  }
  
  FDATA lambda_side, mu_side, sourceN_dx, sourceN_dy, sourceN_dz;
  
  if (dx_sidetype > 1 || dy_sidetype > 1 || dz_sidetype > 1) {
    
    {
      Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
      if (dx_sidetype == 2) {
        sourceN_dx = functionManager->evaluate("Neumann dx " + wkset->sidename,"side ip");
      }
      if (dy_sidetype == 2) {
        sourceN_dy = functionManager->evaluate("Neumann dy " + wkset->sidename,"side ip");
      }
      if (dz_sidetype == 2) {
        sourceN_dz = functionManager->evaluate("Neumann dz " + wkset->sidename,"side ip");
      }
      
      lambda_side = functionManager->evaluate("lambda","side ip");
      mu_side = functionManager->evaluate("mu","side ip");
      
    }
    
    // Since normals get recomputed often, this needs to be reset
    auto normals = wkset->normals;
    auto wts = wkset->wts_side;
    auto h = wkset->h;
    auto res = wkset->res;
    
    Teuchos::TimeMonitor localtime(*boundaryResidualFill);
    
    this->computeStress(lambda_side, mu_side, true);
    auto stress = stress_side;
    
    if (spaceDim == 1) {
      int dx_basis = wkset->usebasis[dx_num];
      auto basis = wkset->basis_side[dx_basis];
      auto basis_grad = wkset->basis_grad_side[dx_basis];
      auto off = Kokkos::subview( offsets, dx_num, Kokkos::ALL());
      
      if (dx_sidetype == 2) { // Neumann
        parallel_for("LE ux bndry resid 1D N",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += (-sourceN_dx(e,k)*basis(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
      else if (dx_sidetype == 4) { // weak Dirichlet
        auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
        parallel_for("LE ux bndry resid 1D wD",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
            AD deltadx = dx(e,k); // should be - dval(e,k), but this is set to 0.0
            AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*normals(e,k,0);
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += ((-stress(e,k,0,0)*normals(e,k,0))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*bx*basis_grad(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
      else if (dx_sidetype == 5) { // weak Dirichlet for multiscale
        auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
        auto lambdax = Kokkos::subview( aux_side, Kokkos::ALL(), auxdx_num, Kokkos::ALL());
        parallel_for("LE ux bndry resid 1D wD-ms",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
            AD deltadx = dx(e,k) - lambdax(e,k);
            AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*normals(e,k,0);
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += ((-stress(e,k,0,0)*normals(e,k,0))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*bx*basis_grad(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
    }
    else if (spaceDim == 2) {
      
      // dx equation boundary residual
      {
        int dx_basis = wkset->usebasis[dx_num];
        auto basis = wkset->basis_side[dx_basis];
        auto basis_grad = wkset->basis_grad_side[dx_basis];
        auto off = Kokkos::subview( offsets, dx_num, Kokkos::ALL());
        if (dx_sidetype == 2) { // traction (Neumann)
          parallel_for("LE ux bndry resid 2D N",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-sourceN_dx(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == 4) { // weak Dirichlet (set to 0.0)
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          parallel_for("LE ux bndry resid 2D wD",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k); // ditto
              AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*normals(e,k,0) + lambda_side(e,k)*deltady*normals(e,k,1);
              AD by = mu_side(e,k)*deltady*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,1);
              
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*normals(e,k,0) - stress(e,k,0,1)*normals(e,k,1))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1)))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == 5) { // weak Dirichlet for multiscale
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          auto lambdax = Kokkos::subview( aux_side, Kokkos::ALL(), auxdx_num, Kokkos::ALL());
          auto lambday = Kokkos::subview( aux_side, Kokkos::ALL(), auxdy_num, Kokkos::ALL());
          parallel_for("LE ux bndry resid 1D wD-ms",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k);
              AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*normals(e,k,0) + lambda_side(e,k)*deltady*normals(e,k,1);
              AD by = mu_side(e,k)*deltady*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,1);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*normals(e,k,0) - stress(e,k,0,1)*normals(e,k,1))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0) + by*basis_grad(e,i,k,1)))*wts(e,k);
              }
            }
          });
        }
      }
      
      // dy equation boundary residual
      {
        int dy_basis = wkset->usebasis[dy_num];
        auto basis = wkset->basis_side[dy_basis];
        auto basis_grad = wkset->basis_grad_side[dy_basis];
        auto off = Kokkos::subview( offsets, dy_num, Kokkos::ALL());
        if (dy_sidetype == 2) { // traction (Neumann)
          parallel_for("LE uy bndry resid 2D N",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-sourceN_dy(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == 4) { // weak Dirichlet (set to 0.0)
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          parallel_for("LE uy bndry resid 2D wD",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k); // ditto
              AD bx = mu_side(e,k)*deltady*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,1);
              AD by = lambda_side(e,k)*deltadx*normals(e,k,0) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltady*normals(e,k,1);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*normals(e,k,0) - stress(e,k,1,1)*normals(e,k,1))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1)))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == 5) { // weak Dirichlet for multiscale
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          auto lambdax = Kokkos::subview( aux_side, Kokkos::ALL(), auxdx_num, Kokkos::ALL());
          auto lambday = Kokkos::subview( aux_side, Kokkos::ALL(), auxdy_num, Kokkos::ALL());
          parallel_for("LE uy bndry resid 2D wD-ms",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k);
              AD bx = mu_side(e,k)*deltady*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,1);
              AD by = lambda_side(e,k)*deltadx*normals(e,k,0) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltady*normals(e,k,1);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*normals(e,k,0) - stress(e,k,1,1)*normals(e,k,1))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0) + by*basis_grad(e,i,k,1)))*wts(e,k);
              }
            }
          });
        }
      }
    }
    
    else if (spaceDim == 3) {
      
      // dx equation boundary residual
      {
        int dx_basis = wkset->usebasis[dx_num];
        auto basis = wkset->basis_side[dx_basis];
        auto basis_grad = wkset->basis_grad_side[dx_basis];
        auto off = Kokkos::subview( offsets, dx_num, Kokkos::ALL());
        if (dx_sidetype == 2) { // traction (Neumann)
          parallel_for("LE ux bndry resid 3D N",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-sourceN_dx(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == 4) { // weak Dirichlet (set to 0.0)
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          auto dz = Kokkos::subview( sol_side, Kokkos::ALL(), dz_num, Kokkos::ALL(), 0);
          parallel_for("LE ux bndry resid 3D wD",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k); // ditto
              AD deltadz = dz(e,k); // ditto
              AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*normals(e,k,0) + lambda_side(e,k)*deltady*normals(e,k,1) + lambda_side(e,k)*deltadz*normals(e,k,2);
              AD by = mu_side(e,k)*deltady*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,1);
              AD bz = mu_side(e,k)*deltadz*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,2);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*normals(e,k,0) - stress(e,k,0,1)*normals(e,k,1) - stress(e,k,0,2)*normals(e,k,2))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == 5) { // weak Dirichlet for multiscale
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          auto dz = Kokkos::subview( sol_side, Kokkos::ALL(), dz_num, Kokkos::ALL(), 0);
          auto lambdax = Kokkos::subview( aux_side, Kokkos::ALL(), auxdx_num, Kokkos::ALL());
          auto lambday = Kokkos::subview( aux_side, Kokkos::ALL(), auxdy_num, Kokkos::ALL());
          auto lambdaz = Kokkos::subview( aux_side, Kokkos::ALL(), auxdz_num, Kokkos::ALL());
          parallel_for("LE ux bndry resid 3D wD-ms",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k); // ditto
              AD deltadz = dz(e,k) - lambdaz(e,k); // ditto
              AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*normals(e,k,0) + lambda_side(e,k)*deltady*normals(e,k,1) + lambda_side(e,k)*deltadz*normals(e,k,2);
              AD by = mu_side(e,k)*deltady*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,1);
              AD bz = mu_side(e,k)*deltadz*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,2);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*normals(e,k,0) - stress(e,k,0,1)*normals(e,k,1) - stress(e,k,0,2)*normals(e,k,2))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
              }
            }
          });
        }
      }
      
      // dy equation boundary residual
      {
        int dy_basis = wkset->usebasis[dy_num];
        auto basis = wkset->basis_side[dy_basis];
        auto basis_grad = wkset->basis_grad_side[dy_basis];
        auto off = Kokkos::subview( offsets, dy_num, Kokkos::ALL());
        if (dy_sidetype == 2) { // traction (Neumann)
          parallel_for("LE uy bndry resid 3D N",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-sourceN_dy(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == 4) { // weak Dirichlet (set to 0.0)
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          auto dz = Kokkos::subview( sol_side, Kokkos::ALL(), dz_num, Kokkos::ALL(), 0);
          parallel_for("LE uy bndry resid 3D wD",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k); // ditto
              AD deltadz = dz(e,k); // ditto
              AD bx = mu_side(e,k)*deltady*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,1);
              AD by = lambda_side(e,k)*deltadx*normals(e,k,0) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltady*normals(e,k,1) + lambda_side(e,k)*deltadz*normals(e,k,2);
              AD bz = mu_side(e,k)*deltadz*normals(e,k,1) + mu_side(e,k)*deltady*normals(e,k,2);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*normals(e,k,0) - stress(e,k,1,1)*normals(e,k,1) - stress(e,k,1,2)*normals(e,k,2))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == 5) { // weak Dirichlet for multiscale
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          auto dz = Kokkos::subview( sol_side, Kokkos::ALL(), dz_num, Kokkos::ALL(), 0);
          auto lambdax = Kokkos::subview( aux_side, Kokkos::ALL(), auxdx_num, Kokkos::ALL());
          auto lambday = Kokkos::subview( aux_side, Kokkos::ALL(), auxdy_num, Kokkos::ALL());
          auto lambdaz = Kokkos::subview( aux_side, Kokkos::ALL(), auxdz_num, Kokkos::ALL());
          parallel_for("LE uy bndry resid 3D wD-ms",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k); // ditto
              AD deltadz = dz(e,k) - lambdaz(e,k); // ditto
              AD bx = mu_side(e,k)*deltady*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,1);
              AD by = lambda_side(e,k)*deltadx*normals(e,k,0) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltady*normals(e,k,1) + lambda_side(e,k)*deltadz*normals(e,k,2);
              AD bz = mu_side(e,k)*deltadz*normals(e,k,1) + mu_side(e,k)*deltady*normals(e,k,2);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*normals(e,k,0) - stress(e,k,1,1)*normals(e,k,1) - stress(e,k,1,2)*normals(e,k,2))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
              }
            }
          });
        }
      }
      // dz equation boundary residual
      {
        int dz_basis = wkset->usebasis[dz_num];
        auto basis = wkset->basis_side[dz_basis];
        auto basis_grad = wkset->basis_grad_side[dz_basis];
        auto off = Kokkos::subview( offsets, dz_num, Kokkos::ALL());
        if (dz_sidetype == 2) { // traction (Neumann)
          parallel_for("LE uz bndry resid 3D N",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-sourceN_dz(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dz_sidetype == 4) { // weak Dirichlet (set to 0.0)
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          auto dz = Kokkos::subview( sol_side, Kokkos::ALL(), dz_num, Kokkos::ALL(), 0);
          parallel_for("LE uz bndry resid 3D wD",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k); // ditto
              AD deltadz = dz(e,k); // ditto
              AD bx = mu_side(e,k)*deltadz*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,2);
              AD by = mu_side(e,k)*deltadz*normals(e,k,1) + mu_side(e,k)*deltady*normals(e,k,2);
              AD bz = lambda_side(e,k)*deltadx*normals(e,k,0) + lambda_side(e,k)*deltady*normals(e,k,1) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltadz*normals(e,k,2);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,2,0)*normals(e,k,0) - stress(e,k,2,1)*normals(e,k,1) - stress(e,k,2,2)*normals(e,k,2))*basis(e,i,k,0) + penalty*deltadz*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
              }
            }
          });
        }
        else if (dz_sidetype == 5) { // weak Dirichlet for multiscale
          auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
          auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
          auto dz = Kokkos::subview( sol_side, Kokkos::ALL(), dz_num, Kokkos::ALL(), 0);
          auto lambdax = Kokkos::subview( aux_side, Kokkos::ALL(), auxdx_num, Kokkos::ALL());
          auto lambday = Kokkos::subview( aux_side, Kokkos::ALL(), auxdy_num, Kokkos::ALL());
          auto lambdaz = Kokkos::subview( aux_side, Kokkos::ALL(), auxdz_num, Kokkos::ALL());
          parallel_for("LE uz bndry resid 3D wD-ms",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k); // ditto
              AD deltadz = dz(e,k) - lambdaz(e,k); // ditto
              AD bx = mu_side(e,k)*deltadz*normals(e,k,0) + mu_side(e,k)*deltadx*normals(e,k,2);
              AD by = mu_side(e,k)*deltadz*normals(e,k,1) + mu_side(e,k)*deltady*normals(e,k,2);
              AD bz = lambda_side(e,k)*deltadx*normals(e,k,0) + lambda_side(e,k)*deltady*normals(e,k,1) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltadz*normals(e,k,2);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,2,0)*normals(e,k,0) - stress(e,k,2,1)*normals(e,k,1) - stress(e,k,2,2)*normals(e,k,2))*basis(e,i,k,0) + penalty*deltadz*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
              }
            }
          });
        }
      }
    }
  }
}


// ========================================================================================
// The boundary/edge flux
// ========================================================================================

void linearelasticity::computeFlux() {
  
  // TMW this will break adjoint computations if sf \neq 1.0
  //ScalarT sf = 1.0;
  //if (wkset->isAdjoint) {
  //  sf = modelparams(0);
  //}
  FDATA lambda_side, mu_side;
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    
    lambda_side = functionManager->evaluate("lambda","side ip");
    mu_side = functionManager->evaluate("mu","side ip");
  }
  
  auto normals = wkset->normals;
  auto h = wkset->h;
  
  // Just need the basis for the number of active elements (any side basis will do)
  auto basis = wkset->basis_side[wkset->usebasis[dx_num]];
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    this->computeStress(lambda_side, mu_side, true);
    
    if (spaceDim == 1) {
      auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
      auto lambdax = Kokkos::subview( aux_side, Kokkos::ALL(), auxdx_num, Kokkos::ALL());
      auto flux_x = Kokkos::subview( flux, Kokkos::ALL(), dx_num, Kokkos::ALL());
      parallel_for("LE flux 1D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_type k=0; k<flux_x.extent(1); k++) {
          AD penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
          flux_x(e,k) = 1.0*stress_side(e,k,0,0)*normals(e,k,0) + penalty*(lambdax(e,k)-dx(e,k));
        }
      });
    }
    else if (spaceDim == 2) {
      auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
      auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
      auto lambdax = Kokkos::subview( aux_side, Kokkos::ALL(), auxdx_num, Kokkos::ALL());
      auto lambday = Kokkos::subview( aux_side, Kokkos::ALL(), auxdy_num, Kokkos::ALL());
      auto flux_x = Kokkos::subview( flux, Kokkos::ALL(), dx_num, Kokkos::ALL());
      auto flux_y = Kokkos::subview( flux, Kokkos::ALL(), dy_num, Kokkos::ALL());
      parallel_for("LE flux 2D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_type k=0; k<flux_x.extent(1); k++) {
          AD penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
          flux_x(e,k) = 1.0*(stress_side(e,k,0,0)*normals(e,k,0) + stress_side(e,k,0,1)*normals(e,k,1)) + penalty*(lambdax(e,k)-dx(e,k));
          flux_y(e,k) = 1.0*(stress_side(e,k,1,0)*normals(e,k,0) + stress_side(e,k,1,1)*normals(e,k,1)) + penalty*(lambday(e,k)-dy(e,k));
        }
      });
    }
    else if (spaceDim == 3) {
      auto dx = Kokkos::subview( sol_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), 0);
      auto dy = Kokkos::subview( sol_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), 0);
      auto dz = Kokkos::subview( sol_side, Kokkos::ALL(), dz_num, Kokkos::ALL(), 0);
      auto lambdax = Kokkos::subview( aux_side, Kokkos::ALL(), auxdx_num, Kokkos::ALL());
      auto lambday = Kokkos::subview( aux_side, Kokkos::ALL(), auxdy_num, Kokkos::ALL());
      auto lambdaz = Kokkos::subview( aux_side, Kokkos::ALL(), auxdz_num, Kokkos::ALL());
      auto flux_x = Kokkos::subview( flux, Kokkos::ALL(), dx_num, Kokkos::ALL());
      auto flux_y = Kokkos::subview( flux, Kokkos::ALL(), dy_num, Kokkos::ALL());
      auto flux_z = Kokkos::subview( flux, Kokkos::ALL(), dz_num, Kokkos::ALL());
      parallel_for("LE flux 3D",RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (size_type k=0; k<flux_x.extent(1); k++) {
          AD penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
          flux_x(e,k) = 1.0*(stress_side(e,k,0,0)*normals(e,k,0) + stress_side(e,k,0,1)*normals(e,k,1) + stress_side(e,k,0,2)*normals(e,k,2)) + penalty*(lambdax(e,k)-dx(e,k));
          flux_y(e,k) = 1.0*(stress_side(e,k,1,0)*normals(e,k,0) + stress_side(e,k,1,1)*normals(e,k,1) + stress_side(e,k,1,2)*normals(e,k,2)) + penalty*(lambday(e,k)-dy(e,k));
          flux_z(e,k) = 1.0*(stress_side(e,k,2,0)*normals(e,k,0) + stress_side(e,k,2,1)*normals(e,k,1) + stress_side(e,k,2,2)*normals(e,k,2)) + penalty*(lambdaz(e,k)-dz(e,k));
        }
      });
    }
  }
  //KokkosTools::print(stress);
  //KokkosTools::print(flux);
  
}

// ========================================================================================
// ========================================================================================

void linearelasticity::setVars(std::vector<string> & varlist) {
  //varlist = varlist_;
  dx_num = -1;
  dy_num = -1;
  dz_num = -1;
  e_num = -1;
  p_num = -1;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "dx")
      dx_num = i;
    else if (varlist[i] == "dy")
      dy_num = i;
    else if (varlist[i] == "dz")
      dz_num = i;
    else if (varlist[i] == "e")
      e_num = i;
    else if (varlist[i] == "p")
      p_num = i;
    else if (varlist[i] == "Po")
      p_num = i;
    else if (varlist[i] == "Pw")
      p_num = i;
    
  }
}

// ========================================================================================
// ========================================================================================

void linearelasticity::setAuxVars(std::vector<string> & auxvarlist) {
  
  for (size_t i=0; i<auxvarlist.size(); i++) {
    if (auxvarlist[i] == "dx")
      auxdx_num = i;
    else if (auxvarlist[i] == "dy")
      auxdy_num = i;
    else if (auxvarlist[i] == "dz")
      auxdz_num = i;
    else if (auxvarlist[i] == "e")
      auxe_num = i;
    else if (auxvarlist[i] == "p")
      auxp_num = i;
    else if (auxvarlist[i] == "Po")
      auxp_num = i;
    else if (auxvarlist[i] == "Pw")
      auxp_num = i;
    
  }
}

// ========================================================================================
// return the stress
// ========================================================================================

void linearelasticity::computeStress(FDATA lambda, FDATA mu, const bool & onside) {
  
  Teuchos::TimeMonitor localtime(*fillStress);
           
  auto mp = modelparams;
  
  if (useCE) {
    vector<int> indices = {dx_num, dy_num, dz_num, e_num};
    if (onside) {
      stress_side = crystalelast->computeStress(wkset, indices, onside);
    }
    else {
      stress_vol = crystalelast->computeStress(wkset, indices, onside);
    }
  }
  else {
    
    if (onside){
      auto stress = stress_side;
      if (spaceDim == 1) {
        auto grad_dx = Kokkos::subview( sol_grad_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), Kokkos::ALL());
        if (incplanestress) { // lambda = 2*mu
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = 4.0*mu(e,k)*grad_dx(e,k,0);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = Kokkos::subview( sol_side, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
            parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
              }
            });
          }
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*grad_dx(e,k,0);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = Kokkos::subview( sol_side, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
            parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              }
            });
          }
        }
        if (addBiot) {
          auto pres = Kokkos::subview( sol_side, Kokkos::ALL(), p_num, Kokkos::ALL(), 0);
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += mp(2)*pres(e,k);
            }
          });
        }
        
      }
      else if (spaceDim == 2) {
        auto grad_dx = Kokkos::subview( sol_grad_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), Kokkos::ALL());
        auto grad_dy = Kokkos::subview( sol_grad_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), Kokkos::ALL());
        if (incplanestress) { // lambda = 2*mu
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = 4.0*mu(e,k)*grad_dx(e,k,0) + 2.0*mu(e,k)*grad_dy(e,k,1);
              stress(e,k,0,1) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
              stress(e,k,1,0) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
              stress(e,k,1,1) = 4.0*mu(e,k)*grad_dy(e,k,1) + 2.0*mu(e,k)*grad_dx(e,k,0);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = Kokkos::subview( sol_side, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
            parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
                stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
              }
            });
          }
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*grad_dx(e,k,0) + lambda(e,k)*grad_dy(e,k,1);
              stress(e,k,0,1) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
              stress(e,k,1,0) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
              stress(e,k,1,1) = (2.0*mu(e,k)+lambda(e,k))*grad_dy(e,k,1) + lambda(e,k)*grad_dx(e,k,0);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = Kokkos::subview( sol_side, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
            parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
                stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              }
            });
          }
        }
        if (addBiot) {
          auto pres = Kokkos::subview( sol_side, Kokkos::ALL(), p_num, Kokkos::ALL(), 0);
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += -mp(2)*pres(e,k);
              stress(e,k,1,1) += -mp(2)*pres(e,k);
            }
          });
        }
      }
      else if (spaceDim == 3) {
        auto grad_dx = Kokkos::subview( sol_grad_side, Kokkos::ALL(), dx_num, Kokkos::ALL(), Kokkos::ALL());
        auto grad_dy = Kokkos::subview( sol_grad_side, Kokkos::ALL(), dy_num, Kokkos::ALL(), Kokkos::ALL());
        auto grad_dz = Kokkos::subview( sol_grad_side, Kokkos::ALL(), dz_num, Kokkos::ALL(), Kokkos::ALL());
        
        parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<stress.extent(1); k++) {
            stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*grad_dx(e,k,0) + lambda(e,k)*(grad_dy(e,k,1) + grad_dz(e,k,2));
            stress(e,k,0,1) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
            stress(e,k,0,2) = mu(e,k)*(grad_dx(e,k,2) + grad_dz(e,k,0));
            stress(e,k,1,0) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
            stress(e,k,1,1) = (2.0*mu(e,k)+lambda(e,k))*grad_dy(e,k,1) + lambda(e,k)*(grad_dx(e,k,0) + grad_dz(e,k,2));
            stress(e,k,1,2) = mu(e,k)*(grad_dy(e,k,2) + grad_dz(e,k,1));
            stress(e,k,2,0) = mu(e,k)*(grad_dx(e,k,2) + grad_dz(e,k,0));
            stress(e,k,2,1) = mu(e,k)*(grad_dy(e,k,2) + grad_dz(e,k,1));
            stress(e,k,2,2) = (2.0*mu(e,k)+lambda(e,k))*grad_dz(e,k,2) + lambda(e,k)*(grad_dx(e,k,0) + grad_dy(e,k,1));
          }
        });
        if (e_num>=0) { // include thermoelastic
          auto T = Kokkos::subview( sol_side, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              stress(e,k,2,2) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
            }
          });
        }
      }
      if (addBiot) {
        auto pres = Kokkos::subview( sol_side, Kokkos::ALL(), p_num, Kokkos::ALL(), 0);
        parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<stress.extent(1); k++) {
            stress(e,k,0,0) += -mp(2)*pres(e,k);
            stress(e,k,1,1) += -mp(2)*pres(e,k);
            stress(e,k,2,2) += -mp(2)*pres(e,k);
          }
        });
      }
    }
    else {
      auto stress = stress_vol;
      if (spaceDim == 1) {
        auto grad_dx = Kokkos::subview( sol_grad, Kokkos::ALL(), dx_num, Kokkos::ALL(), Kokkos::ALL());
        if (incplanestress) { // lambda = 2*mu
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = 4.0*mu(e,k)*grad_dx(e,k,0);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = Kokkos::subview( sol, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
            parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
              }
            });
          }
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*grad_dx(e,k,0);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = Kokkos::subview( sol, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
            parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              }
            });
          }
        }
        if (addBiot) {
          auto pres = Kokkos::subview( sol, Kokkos::ALL(), p_num, Kokkos::ALL(), 0);
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += mp(2)*pres(e,k);
            }
          });
        }
        
      }
      else if (spaceDim == 2) {
        auto grad_dx = Kokkos::subview( sol_grad, Kokkos::ALL(), dx_num, Kokkos::ALL(), Kokkos::ALL());
        auto grad_dy = Kokkos::subview( sol_grad, Kokkos::ALL(), dy_num, Kokkos::ALL(), Kokkos::ALL());
        if (incplanestress) { // lambda = 2*mu
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = 4.0*mu(e,k)*grad_dx(e,k,0) + 2.0*mu(e,k)*grad_dy(e,k,1);
              stress(e,k,0,1) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
              stress(e,k,1,0) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
              stress(e,k,1,1) = 4.0*mu(e,k)*grad_dy(e,k,1) + 2.0*mu(e,k)*grad_dx(e,k,0);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = Kokkos::subview( sol, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
            parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
                stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
              }
            });
          }
        }
        else {
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*grad_dx(e,k,0) + lambda(e,k)*grad_dy(e,k,1);
              stress(e,k,0,1) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
              stress(e,k,1,0) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
              stress(e,k,1,1) = (2.0*mu(e,k)+lambda(e,k))*grad_dy(e,k,1) + lambda(e,k)*grad_dx(e,k,0);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = Kokkos::subview( sol, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
            parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
                stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              }
            });
          }
        }
        if (addBiot) {
          auto pres = Kokkos::subview( sol, Kokkos::ALL(), p_num, Kokkos::ALL(), 0);
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += -mp(2)*pres(e,k);
              stress(e,k,1,1) += -mp(2)*pres(e,k);
            }
          });
        }
      }
      else if (spaceDim == 3) {
        auto grad_dx = Kokkos::subview( sol_grad, Kokkos::ALL(), dx_num, Kokkos::ALL(), Kokkos::ALL());
        auto grad_dy = Kokkos::subview( sol_grad, Kokkos::ALL(), dy_num, Kokkos::ALL(), Kokkos::ALL());
        auto grad_dz = Kokkos::subview( sol_grad, Kokkos::ALL(), dz_num, Kokkos::ALL(), Kokkos::ALL());
        
        parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<stress.extent(1); k++) {
            stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*grad_dx(e,k,0) + lambda(e,k)*(grad_dy(e,k,1) + grad_dz(e,k,2));
            stress(e,k,0,1) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
            stress(e,k,0,2) = mu(e,k)*(grad_dx(e,k,2) + grad_dz(e,k,0));
            stress(e,k,1,0) = mu(e,k)*(grad_dx(e,k,1) + grad_dy(e,k,0));
            stress(e,k,1,1) = (2.0*mu(e,k)+lambda(e,k))*grad_dy(e,k,1) + lambda(e,k)*(grad_dx(e,k,0) + grad_dz(e,k,2));
            stress(e,k,1,2) = mu(e,k)*(grad_dy(e,k,2) + grad_dz(e,k,1));
            stress(e,k,2,0) = mu(e,k)*(grad_dx(e,k,2) + grad_dz(e,k,0));
            stress(e,k,2,1) = mu(e,k)*(grad_dy(e,k,2) + grad_dz(e,k,1));
            stress(e,k,2,2) = (2.0*mu(e,k)+lambda(e,k))*grad_dz(e,k,2) + lambda(e,k)*(grad_dx(e,k,0) + grad_dy(e,k,1));
          }
        });
        if (e_num>=0) { // include thermoelastic
          auto T = Kokkos::subview( sol, Kokkos::ALL(), e_num, Kokkos::ALL(), 0);
          parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              stress(e,k,2,2) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
            }
          });
        }
      }
      if (addBiot) {
        auto pres = Kokkos::subview( sol, Kokkos::ALL(), p_num, Kokkos::ALL(), 0);
        parallel_for(RangePolicy<AssemblyExec>(0,stress.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<stress.extent(1); k++) {
            stress(e,k,0,0) += -mp(2)*pres(e,k);
            stress(e,k,1,1) += -mp(2)*pres(e,k);
            stress(e,k,2,2) += -mp(2)*pres(e,k);
          }
        });
      }
    }
  }
}

// ========================================================================================
// TMW: needs to be deprecated
//      Need to update crystal elasticity to use function manager or wkset
// ========================================================================================

void linearelasticity::updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                                        const vector<string> & paramnames) {
  if (useCE) {
    crystalelast->updateParams(wkset);
  }
}

