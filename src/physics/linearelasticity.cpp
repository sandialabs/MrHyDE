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

linearelasticity::linearelasticity(Teuchos::RCP<Teuchos::ParameterList> & settings, const bool & isaux_)
  : physicsbase(settings, isaux_)
{
  
  label = "linearelasticity";
  
  spaceDim = 0;
  spaceDim = settings->sublist("Mesh").get<int>("dim",0);
  if (spaceDim == 0) {
    spaceDim = settings->sublist("Mesh").get<int>("dimension",0);
  }
  if (spaceDim == 0) {
    //
  }
  
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
  int numElem = settings->sublist("Solver").get<int>("workset size",100);
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
  
  stress_vol = View_AD4("stress tensor", functionManager->numElem,
                        functionManager->numip, spaceDim, spaceDim);
  
  stress_side = View_AD4("stress tensor", functionManager->numElem,
                         functionManager->numip_side, spaceDim, spaceDim);
  
}
// ========================================================================================
// ========================================================================================

void linearelasticity::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  View_AD2 lambda, mu, source_dx, source_dy, source_dz;
  
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
    auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
    
    parallel_for("LE volume resid 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
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
      auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
      
      parallel_for("LE ux volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
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
      auto off = Kokkos::subview( wkset->offsets, dy_num, Kokkos::ALL());
      
      parallel_for("LE uy volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
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
      auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
      
      parallel_for("LE ux volume resid 3D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
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
      auto off = Kokkos::subview( wkset->offsets, dy_num, Kokkos::ALL());
      
      parallel_for("LE uy volume resid 3D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
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
      auto off = Kokkos::subview( wkset->offsets, dz_num, Kokkos::ALL());
      
      parallel_for("LE uz volume resid 3D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
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
  
  int spaceDim = wkset->dimension;
  auto bcs = wkset->var_bcs;
  
  int cside = wkset->currentside;
  
  string dx_sidetype = bcs(dx_num,cside);
  string dy_sidetype = "Dirichlet";
  string dz_sidetype = "Dirichlet";
  if (spaceDim > 1) {
    dy_sidetype = bcs(dy_num,cside);
  }
  if (spaceDim > 2) {
    dz_sidetype = bcs(dz_num,cside);
  }
  
  View_AD2 lambda_side, mu_side, source_dx, source_dy, source_dz;
  
  if (dx_sidetype != "Dirichlet" || dy_sidetype != "Dirichlet" || dz_sidetype != "Dirichlet") {
    
    {
      Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
      if (dx_sidetype == "Neumann") {
        source_dx = functionManager->evaluate("Neumann dx " + wkset->sidename,"side ip");
      }
      else if (dx_sidetype == "weak Dirichlet") {
        source_dx = functionManager->evaluate("Dirichlet dx " + wkset->sidename,"side ip");
      }
      if (dy_sidetype == "Neumann") {
        source_dy = functionManager->evaluate("Neumann dy " + wkset->sidename,"side ip");
      }
      else if (dy_sidetype == "weak Dirichlet") {
        source_dy = functionManager->evaluate("Dirichlet dy " + wkset->sidename,"side ip");
      }
      if (dz_sidetype == "Neumann") {
        source_dz = functionManager->evaluate("Neumann dz " + wkset->sidename,"side ip");
      }
      else if (dz_sidetype == "weak Dirichlet") {
        source_dz = functionManager->evaluate("Dirichlet dz " + wkset->sidename,"side ip");
      }
      
      lambda_side = functionManager->evaluate("lambda","side ip");
      mu_side = functionManager->evaluate("mu","side ip");
      
    }
    
    // Since normals get recomputed often, this needs to be reset
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
      auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
      auto nx = wkset->getDataSc("nx side");
      if (dx_sidetype == "Neumann") { // Neumann
        parallel_for("LE ux bndry resid 1D N",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += (-source_dx(e,k)*basis(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
      else if (dx_sidetype == "weak Dirichlet") { // weak Dirichlet
        auto dx = wkset->getData("dx side");
        parallel_for("LE ux bndry resid 1D wD",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
            AD deltadx = dx(e,k) - source_dx(e,k);
            AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*nx(e,k);
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*bx*basis_grad(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
      else if (dx_sidetype == "interface") { // weak Dirichlet for multiscale
        auto dx = wkset->getData("dx side");
        auto lambdax = wkset->getData("aux dx side");
        parallel_for("LE ux bndry resid 1D wD-ms",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
            AD deltadx = dx(e,k) - lambdax(e,k);
            AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*nx(e,k);
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*bx*basis_grad(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
    }
    else if (spaceDim == 2) {
      auto nx = wkset->getDataSc("nx side");
      auto ny = wkset->getDataSc("ny side");
      
      // dx equation boundary residual
      {
        int dx_basis = wkset->usebasis[dx_num];
        auto basis = wkset->basis_side[dx_basis];
        auto basis_grad = wkset->basis_grad_side[dx_basis];
        auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
        
        if (dx_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("LE ux bndry resid 2D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dx(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == "weak Dirichlet") { // weak Dirichlet (set to 0.0)
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          parallel_for("LE ux bndry resid 2D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - source_dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k) - source_dy(e,k); // ditto
              AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*nx(e,k) + lambda_side(e,k)*deltady*ny(e,k);
              AD by = mu_side(e,k)*deltady*nx(e,k) + mu_side(e,k)*deltadx*ny(e,k);
              
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k) - stress(e,k,0,1)*ny(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1)))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == "interface") { // weak Dirichlet for multiscale
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          auto lambdax = wkset->getData("aux dx side");
          auto lambday = wkset->getData("aux dy side");
          parallel_for("LE ux bndry resid 1D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k);
              AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*nx(e,k) + lambda_side(e,k)*deltady*ny(e,k);
              AD by = mu_side(e,k)*deltady*nx(e,k) + mu_side(e,k)*deltadx*ny(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k) - stress(e,k,0,1)*ny(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0) + by*basis_grad(e,i,k,1)))*wts(e,k);
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
        auto off = Kokkos::subview( wkset->offsets, dy_num, Kokkos::ALL());
        if (dy_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("LE uy bndry resid 2D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dy(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == "weak Dirichlet") { // weak Dirichlet (set to 0.0)
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          parallel_for("LE uy bndry resid 2D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - source_dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k) - source_dy(e,k); // ditto
              AD bx = mu_side(e,k)*deltady*nx(e,k) + mu_side(e,k)*deltadx*ny(e,k);
              AD by = lambda_side(e,k)*deltadx*nx(e,k) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltady*ny(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*nx(e,k) - stress(e,k,1,1)*ny(e,k))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1)))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == "interface") { // weak Dirichlet for multiscale
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          auto lambdax = wkset->getData("aux dx side");
          auto lambday = wkset->getData("aux dy side");
          parallel_for("LE uy bndry resid 2D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k);
              AD bx = mu_side(e,k)*deltady*nx(e,k) + mu_side(e,k)*deltadx*ny(e,k);
              AD by = lambda_side(e,k)*deltadx*nx(e,k) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltady*ny(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*nx(e,k) - stress(e,k,1,1)*ny(e,k))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0) + by*basis_grad(e,i,k,1)))*wts(e,k);
              }
            }
          });
        }
      }
    }
    
    else if (spaceDim == 3) {
      auto nx = wkset->getDataSc("nx side");
      auto ny = wkset->getDataSc("ny side");
      auto nz = wkset->getDataSc("nz side");
      
      // dx equation boundary residual
      {
        int dx_basis = wkset->usebasis[dx_num];
        auto basis = wkset->basis_side[dx_basis];
        auto basis_grad = wkset->basis_grad_side[dx_basis];
        auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
        if (dx_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("LE ux bndry resid 3D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dx(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == "weak Dirichlet") { // weak Dirichlet (set to 0.0)
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          auto dz = wkset->getData("dz side");
          parallel_for("LE ux bndry resid 3D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - source_dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k) - source_dy(e,k); // ditto
              AD deltadz = dz(e,k) - source_dz(e,k); // ditto
              AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*nx(e,k) + lambda_side(e,k)*deltady*ny(e,k) + lambda_side(e,k)*deltadz*nz(e,k);
              AD by = mu_side(e,k)*deltady*nx(e,k) + mu_side(e,k)*deltadx*ny(e,k);
              AD bz = mu_side(e,k)*deltadz*nx(e,k) + mu_side(e,k)*deltadx*nz(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k) - stress(e,k,0,1)*ny(e,k) - stress(e,k,0,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == "interface") { // weak Dirichlet for multiscale
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          auto dz = wkset->getData("dz side");
          auto lambdax = wkset->getData("aux dx side");
          auto lambday = wkset->getData("aux dy side");
          auto lambdaz = wkset->getData("aux dz side");
          parallel_for("LE ux bndry resid 3D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k); // ditto
              AD deltadz = dz(e,k) - lambdaz(e,k); // ditto
              AD bx = (lambda_side(e,k) + 2.0*mu_side(e,k))*deltadx*nx(e,k) + lambda_side(e,k)*deltady*ny(e,k) + lambda_side(e,k)*deltadz*nz(e,k);
              AD by = mu_side(e,k)*deltady*nx(e,k) + mu_side(e,k)*deltadx*ny(e,k);
              AD bz = mu_side(e,k)*deltadz*nx(e,k) + mu_side(e,k)*deltadx*nz(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k) - stress(e,k,0,1)*ny(e,k) - stress(e,k,0,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
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
        auto off = Kokkos::subview( wkset->offsets, dy_num, Kokkos::ALL());
        if (dy_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("LE uy bndry resid 3D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dy(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == "weak Dirichlet") { // weak Dirichlet (set to 0.0)
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          auto dz = wkset->getData("dz side");
          parallel_for("LE uy bndry resid 3D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - source_dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k) - source_dy(e,k); // ditto
              AD deltadz = dz(e,k) - source_dz(e,k); // ditto
              AD bx = mu_side(e,k)*deltady*nx(e,k) + mu_side(e,k)*deltadx*ny(e,k);
              AD by = lambda_side(e,k)*deltadx*nx(e,k) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltady*ny(e,k) + lambda_side(e,k)*deltadz*nz(e,k);
              AD bz = mu_side(e,k)*deltadz*ny(e,k) + mu_side(e,k)*deltady*nz(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*nx(e,k) - stress(e,k,1,1)*ny(e,k) - stress(e,k,1,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == "interface") { // weak Dirichlet for multiscale
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          auto dz = wkset->getData("dz side");
          auto lambdax = wkset->getData("aux dx side");
          auto lambday = wkset->getData("aux dy side");
          auto lambdaz = wkset->getData("aux dz side");
          parallel_for("LE uy bndry resid 3D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k); // ditto
              AD deltadz = dz(e,k) - lambdaz(e,k); // ditto
              AD bx = mu_side(e,k)*deltady*nx(e,k) + mu_side(e,k)*deltadx*ny(e,k);
              AD by = lambda_side(e,k)*deltadx*nx(e,k) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltady*ny(e,k) + lambda_side(e,k)*deltadz*nz(e,k);
              AD bz = mu_side(e,k)*deltadz*ny(e,k) + mu_side(e,k)*deltady*nz(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*nx(e,k) - stress(e,k,1,1)*ny(e,k) - stress(e,k,1,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
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
        auto off = Kokkos::subview( wkset->offsets, dz_num, Kokkos::ALL());
        if (dz_sidetype == "Neumann") { // traction (Neumann)
          parallel_for("LE uz bndry resid 3D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dz(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dz_sidetype == "weak Dirichlet") { // weak Dirichlet (set to 0.0)
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          auto dz = wkset->getData("dz side");
          parallel_for("LE uz bndry resid 3D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - source_dx(e,k); // should be - dval(e,k), but this is set to 0.0
              AD deltady = dy(e,k) - source_dy(e,k); // ditto
              AD deltadz = dz(e,k) - source_dz(e,k); // ditto
              AD bx = mu_side(e,k)*deltadz*nx(e,k) + mu_side(e,k)*deltadx*nz(e,k);
              AD by = mu_side(e,k)*deltadz*ny(e,k) + mu_side(e,k)*deltady*nz(e,k);
              AD bz = lambda_side(e,k)*deltadx*nx(e,k) + lambda_side(e,k)*deltady*ny(e,k) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltadz*nz(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,2,0)*nx(e,k) - stress(e,k,2,1)*ny(e,k) - stress(e,k,2,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltadz*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
              }
            }
          });
        }
        else if (dz_sidetype == "interface") { // weak Dirichlet for multiscale
          auto dx = wkset->getData("dx side");
          auto dy = wkset->getData("dy side");
          auto dz = wkset->getData("dz side");
          auto lambdax = wkset->getData("aux dx side");
          auto lambday = wkset->getData("aux dy side");
          auto lambdaz = wkset->getData("aux dz side");
          
          parallel_for("LE uz bndry resid 3D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              AD penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              AD deltadx = dx(e,k) - lambdax(e,k);
              AD deltady = dy(e,k) - lambday(e,k); // ditto
              AD deltadz = dz(e,k) - lambdaz(e,k); // ditto
              AD bx = mu_side(e,k)*deltadz*nx(e,k) + mu_side(e,k)*deltadx*nz(e,k);
              AD by = mu_side(e,k)*deltadz*ny(e,k) + mu_side(e,k)*deltady*nz(e,k);
              AD bz = lambda_side(e,k)*deltadx*nx(e,k) + lambda_side(e,k)*deltady*ny(e,k) + (lambda_side(e,k)+2.0*mu_side(e,k))*deltadz*nz(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,2,0)*nx(e,k) - stress(e,k,2,1)*ny(e,k) - stress(e,k,2,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltadz*basis(e,i,k,0) - modelparams(0)*(bx*basis_grad(e,i,k,0)+by*basis_grad(e,i,k,1) + bz*basis_grad(e,i,k,2)))*wts(e,k);
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
  
  int cside = wkset->currentside;
  string dx_sidetype = wkset->var_bcs(dx_num,cside);
  string dy_sidetype = "Dirichlet";
  string dz_sidetype = "Dirichlet";
  if (spaceDim > 1) {
    dy_sidetype = wkset->var_bcs(dy_num,cside);
  }
  if (spaceDim > 2) {
    dz_sidetype = wkset->var_bcs(dz_num,cside);
  }
  
  int spaceDim = wkset->dimension;
  View_AD2 lambda_side, mu_side;
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    
    lambda_side = functionManager->evaluate("lambda","side ip");
    mu_side = functionManager->evaluate("mu","side ip");
  }
  
  auto h = wkset->h;
  
  // Just need the basis for the number of active elements (any side basis will do)
  auto basis = wkset->basis_side[wkset->usebasis[dx_num]];
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    this->computeStress(lambda_side, mu_side, true);
    
    if (spaceDim == 1) {
      auto nx = wkset->getDataSc("nx side");
      auto dx = wkset->getData("dx side");
      View_AD2 source_dx;
      if (dx_sidetype == "Neumann") {
        source_dx = functionManager->evaluate("Neumann dx " + wkset->sidename,"side ip");
      }
      else if (dx_sidetype == "interface") {
        source_dx = wkset->getData("aux dx side");
      }
      else if (dx_sidetype == "weak Dirichlet" || dx_sidetype == "Dirichlet") {
        source_dx = functionManager->evaluate("Dirichlet dx " + wkset->sidename,"side ip");
      }
      auto flux_x = subview(wkset->flux, ALL(), dx_num, ALL());
      parallel_for("LE flux 1D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_type k=0; k<flux_x.extent(1); k++) {
          AD penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
          flux_x(e,k) = 1.0*stress_side(e,k,0,0)*nx(e,k) + penalty*(source_dx(e,k)-dx(e,k));
        }
      });
    }
    else if (spaceDim == 2) {
      auto nx = wkset->getDataSc("nx side");
      auto ny = wkset->getDataSc("ny side");
      auto dx = wkset->getData("dx side");
      auto dy = wkset->getData("dy side");
      View_AD2 source_dx, source_dy;
      if (dx_sidetype == "Neumann") {
        source_dx = functionManager->evaluate("Neumann dx " + wkset->sidename,"side ip");
      }
      else if (dx_sidetype == "interface") {
        source_dx = wkset->getData("aux dx side");
      }
      else if (dx_sidetype == "weak Dirichlet" || dx_sidetype == "Dirichlet") {
        source_dx = functionManager->evaluate("Dirichlet dx " + wkset->sidename,"side ip");
      }
      
      if (dy_sidetype == "Neumann") {
        source_dy = functionManager->evaluate("Neumann dy " + wkset->sidename,"side ip");
      }
      else if (dy_sidetype == "interface") {
        source_dy = wkset->getData("aux dy side");
      }
      else if (dy_sidetype == "weak Dirichlet" || dy_sidetype == "Dirichlet") {
        source_dy = functionManager->evaluate("Dirichlet dy " + wkset->sidename,"side ip");
      }
      
      auto flux_x = subview( wkset->flux, ALL(), dx_num, ALL());
      auto flux_y = subview( wkset->flux, ALL(), dy_num, ALL());
      parallel_for("LE flux 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_type k=0; k<flux_x.extent(1); k++) {
          AD penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
          flux_x(e,k) = 1.0*(stress_side(e,k,0,0)*nx(e,k) + stress_side(e,k,0,1)*ny(e,k)) + penalty*(source_dx(e,k)-dx(e,k));
          flux_y(e,k) = 1.0*(stress_side(e,k,1,0)*nx(e,k) + stress_side(e,k,1,1)*ny(e,k)) + penalty*(source_dy(e,k)-dy(e,k));
        }
      });
    }
    else if (spaceDim == 3) {
      auto nx = wkset->getDataSc("nx side");
      auto ny = wkset->getDataSc("ny side");
      auto nz = wkset->getDataSc("nz side");
      auto dx = wkset->getData("dx side");
      auto dy = wkset->getData("dy side");
      auto dz = wkset->getData("dz side");
      
      View_AD2 source_dx, source_dy, source_dz;
      if (dx_sidetype == "interface") {
        source_dx = wkset->getData("aux dx side");
      }
      else if (dx_sidetype == "weak Dirichlet" || dx_sidetype == "Dirichlet") {
        source_dx = functionManager->evaluate("Dirichlet dx " + wkset->sidename,"side ip");
      }
      
      if (dy_sidetype == "interface") {
        source_dy = wkset->getData("aux dy side");
      }
      else if (dy_sidetype == "weak Dirichlet" || dy_sidetype == "Dirichlet") {
        source_dy = functionManager->evaluate("Dirichlet dy " + wkset->sidename,"side ip");
      }
      
      if (dz_sidetype == "interface") {
        source_dz = wkset->getData("aux dz side");
      }
      else if (dz_sidetype == "weak Dirichlet" || dz_sidetype == "Dirichlet") {
        source_dz = functionManager->evaluate("Dirichlet dz " + wkset->sidename,"side ip");
      }
      
      auto flux_x = subview( wkset->flux, ALL(), dx_num, ALL());
      auto flux_y = subview( wkset->flux, ALL(), dy_num, ALL());
      auto flux_z = subview( wkset->flux, ALL(), dz_num, ALL());
      parallel_for("LE flux 3D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int e ) {
        for (size_type k=0; k<flux_x.extent(1); k++) {
          AD penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
          flux_x(e,k) = 1.0*(stress_side(e,k,0,0)*nx(e,k) + stress_side(e,k,0,1)*ny(e,k) + stress_side(e,k,0,2)*nz(e,k)) + penalty*(source_dx(e,k)-dx(e,k));
          flux_y(e,k) = 1.0*(stress_side(e,k,1,0)*nx(e,k) + stress_side(e,k,1,1)*ny(e,k) + stress_side(e,k,1,2)*nz(e,k)) + penalty*(source_dy(e,k)-dy(e,k));
          flux_z(e,k) = 1.0*(stress_side(e,k,2,0)*nx(e,k) + stress_side(e,k,2,1)*ny(e,k) + stress_side(e,k,2,2)*nz(e,k)) + penalty*(source_dz(e,k)-dz(e,k));
        }
      });
    }
  }
}

// ========================================================================================
// ========================================================================================

void linearelasticity::setWorkset(Teuchos::RCP<workset> & wkset_) {

  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
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
  
  vector<string> auxvarlist = wkset->aux_varlist;
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

void linearelasticity::computeStress(View_AD2 lambda, View_AD2 mu, const bool & onside) {
  
  Teuchos::TimeMonitor localtime(*fillStress);
           
  auto mp = modelparams;
  int spaceDim = wkset->dimension;
  
  if (useCE) {
    vector<int> indices = {dx_num, dy_num, dz_num, e_num};
    if (onside) {
      crystalelast->computeStress(wkset, indices, onside, stress_side);
    }
    else {
      crystalelast->computeStress(wkset, indices, onside, stress_vol);
    }
  }
  else {
    
    if (onside){
      auto stress = stress_side;
      if (spaceDim == 1) {
        auto ddx_dx = wkset->getData("grad(dx)[x] side");
        if (incplanestress) { // lambda = 2*mu
          parallel_for("LE stress 1D",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = 4.0*mu(e,k)*ddx_dx(e,k);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = wkset->getData("e side");
            parallel_for("LE stress 1D TE",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
              }
            });
          }
        }
        else {
          parallel_for("LE stress 1D",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*ddx_dx(e,k);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = wkset->getData("e side");
            parallel_for("LE stress 1D TE",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              }
            });
          }
        }
        if (addBiot) {
          auto pres = wkset->getData("p side");
          parallel_for("LE stress 1D PE",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += mp(2)*pres(e,k);
            }
          });
        }
        
      }
      else if (spaceDim == 2) {
        auto ddx_dx = wkset->getData("grad(dx)[x] side");
        auto ddx_dy = wkset->getData("grad(dx)[y] side");
        auto ddy_dx = wkset->getData("grad(dy)[x] side");
        auto ddy_dy = wkset->getData("grad(dy)[y] side");
        if (incplanestress) { // lambda = 2*mu
          parallel_for("LE stress 2D",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = 4.0*mu(e,k)*ddx_dx(e,k) + 2.0*mu(e,k)*ddy_dy(e,k);
              stress(e,k,0,1) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
              stress(e,k,1,0) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
              stress(e,k,1,1) = 4.0*mu(e,k)*ddy_dy(e,k) + 2.0*mu(e,k)*ddx_dx(e,k);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = wkset->getData("e side");
            parallel_for("LE stress 2D TE",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
                stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
              }
            });
          }
        }
        else {
          parallel_for("LE stress 2D",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*ddx_dx(e,k) + lambda(e,k)*ddy_dy(e,k);
              stress(e,k,0,1) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
              stress(e,k,1,0) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
              stress(e,k,1,1) = (2.0*mu(e,k)+lambda(e,k))*ddy_dy(e,k) + lambda(e,k)*ddx_dx(e,k);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = wkset->getData("e side");
            parallel_for("LE stress 2D TE",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
                stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              }
            });
          }
        }
        if (addBiot) {
          auto pres = wkset->getData("p side");
          parallel_for("LE stress 2D PE",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += -mp(2)*pres(e,k);
              stress(e,k,1,1) += -mp(2)*pres(e,k);
            }
          });
        }
      }
      else if (spaceDim == 3) {
        auto ddx_dx = wkset->getData("grad(dx)[x] side");
        auto ddx_dy = wkset->getData("grad(dx)[y] side");
        auto ddx_dz = wkset->getData("grad(dx)[z] side");
        auto ddy_dx = wkset->getData("grad(dy)[x] side");
        auto ddy_dy = wkset->getData("grad(dy)[y] side");
        auto ddy_dz = wkset->getData("grad(dy)[z] side");
        auto ddz_dx = wkset->getData("grad(dz)[x] side");
        auto ddz_dy = wkset->getData("grad(dz)[y] side");
        auto ddz_dz = wkset->getData("grad(dz)[z] side");
                
        parallel_for("LE stress 3D",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<stress.extent(1); k++) {
            stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*ddx_dx(e,k) + lambda(e,k)*(ddy_dy(e,k) + ddz_dz(e,k));
            stress(e,k,0,1) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
            stress(e,k,0,2) = mu(e,k)*(ddx_dz(e,k) + ddz_dx(e,k));
            stress(e,k,1,0) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
            stress(e,k,1,1) = (2.0*mu(e,k)+lambda(e,k))*ddy_dy(e,k) + lambda(e,k)*(ddx_dx(e,k) + ddz_dz(e,k));
            stress(e,k,1,2) = mu(e,k)*(ddy_dz(e,k) + ddz_dy(e,k));
            stress(e,k,2,0) = mu(e,k)*(ddx_dz(e,k) + ddz_dx(e,k));
            stress(e,k,2,1) = mu(e,k)*(ddy_dz(e,k) + ddz_dy(e,k));
            stress(e,k,2,2) = (2.0*mu(e,k)+lambda(e,k))*ddz_dz(e,k) + lambda(e,k)*(ddx_dx(e,k) + ddy_dy(e,k));
          }
        });
        if (e_num>=0) { // include thermoelastic
          auto T = wkset->getData("e side");
          parallel_for("LE stress 3D TE",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              stress(e,k,2,2) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
            }
          });
        }
      }
      if (addBiot) {
        auto pres = wkset->getData("p side");
        parallel_for("LE stress 3D PE",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int e ) {
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
        auto ddx_dx = wkset->getData("grad(dx)[x]");
        if (incplanestress) { // lambda = 2*mu
          parallel_for("LE stress 1D",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = 4.0*mu(e,k)*ddx_dx(e,k);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = wkset->getData("e");
            parallel_for("LE stress 1D TE",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
              }
            });
          }
        }
        else {
          parallel_for("LE stress 1D",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*ddx_dx(e,k);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = wkset->getData("e");
            parallel_for("LE stress 1D TE",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              }
            });
          }
        }
        if (addBiot) {
          auto pres = wkset->getData("p");
          parallel_for("LE stress 1D PE",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += mp(2)*pres(e,k);
            }
          });
        }
        
      }
      else if (spaceDim == 2) {
        auto ddx_dx = wkset->getData("grad(dx)[x]");
        auto ddx_dy = wkset->getData("grad(dx)[y]");
        auto ddy_dx = wkset->getData("grad(dy)[x]");
        auto ddy_dy = wkset->getData("grad(dy)[y]");
        if (incplanestress) { // lambda = 2*mu
          parallel_for("LE stress 2D",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = 4.0*mu(e,k)*ddx_dx(e,k) + 2.0*mu(e,k)*ddy_dy(e,k);
              stress(e,k,0,1) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
              stress(e,k,1,0) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
              stress(e,k,1,1) = 4.0*mu(e,k)*ddy_dy(e,k) + 2.0*mu(e,k)*ddx_dx(e,k);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = wkset->getData("e");
            parallel_for("LE stress 3D TE",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
                stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(5.0*mu(e,k));
              }
            });
          }
        }
        else {
          parallel_for("LE stress 2D",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*ddx_dx(e,k) + lambda(e,k)*ddy_dy(e,k);
              stress(e,k,0,1) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
              stress(e,k,1,0) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
              stress(e,k,1,1) = (2.0*mu(e,k)+lambda(e,k))*ddy_dy(e,k) + lambda(e,k)*ddx_dx(e,k);
            }
          });
          if (e_num>=0) { // include thermoelastic
            auto T = wkset->getData("e");
            parallel_for("LE stress 2D TE",
                         RangePolicy<AssemblyExec>(0,wkset->numElem),
                         KOKKOS_LAMBDA (const int e ) {
              for (size_type k=0; k<stress.extent(1); k++) {
                stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
                stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              }
            });
          }
        }
        if (addBiot) {
          auto pres = wkset->getData("p");
          parallel_for("LE stress 2D PE",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += -mp(2)*pres(e,k);
              stress(e,k,1,1) += -mp(2)*pres(e,k);
            }
          });
        }
      }
      else if (spaceDim == 3) {
        auto ddx_dx = wkset->getData("grad(dx)[x]");
        auto ddx_dy = wkset->getData("grad(dx)[y]");
        auto ddx_dz = wkset->getData("grad(dx)[z]");
        auto ddy_dx = wkset->getData("grad(dy)[x]");
        auto ddy_dy = wkset->getData("grad(dy)[y]");
        auto ddy_dz = wkset->getData("grad(dy)[z]");
        auto ddz_dx = wkset->getData("grad(dz)[x]");
        auto ddz_dy = wkset->getData("grad(dz)[y]");
        auto ddz_dz = wkset->getData("grad(dz)[z]");
        
        parallel_for("LE stress 3D",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int e ) {
          for (size_type k=0; k<stress.extent(1); k++) {
            stress(e,k,0,0) = (2.0*mu(e,k)+lambda(e,k))*ddx_dx(e,k) + lambda(e,k)*(ddy_dy(e,k) + ddz_dz(e,k));
            stress(e,k,0,1) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
            stress(e,k,0,2) = mu(e,k)*(ddx_dz(e,k) + ddz_dx(e,k));
            stress(e,k,1,0) = mu(e,k)*(ddx_dy(e,k) + ddy_dx(e,k));
            stress(e,k,1,1) = (2.0*mu(e,k)+lambda(e,k))*ddy_dy(e,k) + lambda(e,k)*(ddx_dx(e,k) + ddz_dz(e,k));
            stress(e,k,1,2) = mu(e,k)*(ddy_dz(e,k) + ddz_dy(e,k));
            stress(e,k,2,0) = mu(e,k)*(ddx_dz(e,k) + ddz_dx(e,k));
            stress(e,k,2,1) = mu(e,k)*(ddy_dz(e,k) + ddz_dy(e,k));
            stress(e,k,2,2) = (2.0*mu(e,k)+lambda(e,k))*ddz_dz(e,k) + lambda(e,k)*(ddx_dx(e,k) + ddy_dy(e,k));
          }
        });
        if (e_num>=0) { // include thermoelastic
          auto T = wkset->getData("e");
          parallel_for("LE stress 3D TE",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       KOKKOS_LAMBDA (const int e ) {
            for (size_type k=0; k<stress.extent(1); k++) {
              stress(e,k,0,0) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              stress(e,k,1,1) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
              stress(e,k,2,2) += -mp(4)*(T(e,k) - mp(3))*(3.0*lambda(e,k) + 2.0*mu(e,k));
            }
          });
        }
      }
      if (addBiot) {
        auto pres = wkset->getData("p");
        parallel_for("LE stress 3D PE",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int e ) {
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


// ========================================================================================
// ========================================================================================

std::vector<string> linearelasticity::getDerivedNames() {
  std::vector<string> derived;
  derived.push_back("VM stress");
  derived.push_back("MAG stress");
  return derived;
}

// ========================================================================================
// ========================================================================================

std::vector<View_AD2> linearelasticity::getDerivedValues() {
  std::vector<View_AD2> derived;
  
  auto lambda = functionManager->evaluate("lambda","ip");
  auto mu = functionManager->evaluate("mu","ip");
  this->computeStress(lambda, mu, false);
  auto stress = stress_vol;

  View_AD2 vmstress("von mises stress",stress.extent(0),stress.extent(1)); // numElem x numip
  View_AD2 magstress("magnitude of stress",stress.extent(0),stress.extent(1)); // numElem x numip
  
  int dimension = wkset->dimension;
  using namespace std;
  
  if (dimension == 1) {
    parallel_for("LE derived fill",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<vmstress.extent(1); ++pt) {
        AD sxx = stress(elem,pt,0,0);
        vmstress(elem,pt) = sqrt(sxx*sxx);
        magstress(elem,pt) = sqrt(sxx*sxx);
      }
    });
  }
  else if (dimension == 2) {
    parallel_for("LE derived fill",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<vmstress.extent(1); ++pt) {
        AD sxx = stress(elem,pt,0,0);
        AD syy = stress(elem,pt,1,1);
        AD sxy = stress(elem,pt,0,1);
        vmstress(elem,pt) = sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy);
        magstress(elem,pt) = sqrt(sxx*sxx + syy*syy);
      }
    });
  }
  else if (dimension == 3) {
    parallel_for("LE derived fill",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<vmstress.extent(1); ++pt) {
        AD sxx = stress(elem,pt,0,0);
        AD syy = stress(elem,pt,1,1);
        AD szz = stress(elem,pt,2,2);
        AD sxy = stress(elem,pt,0,1);
        AD syz = stress(elem,pt,1,2);
        AD szx = stress(elem,pt,2,0);
        vmstress(elem,pt) = sqrt(0.5*((sxx-syy)*(sxx-syy) + (syy-szz)*(syy-szz) + (szz-sxx)*(szz-sxx)) + 3.0*(sxy*sxy+syz*syz+szx*szx) );
        magstress(elem,pt) = sqrt(sxx*sxx + syy*syy + szz*szz);
      }
    });
  }
  
  derived.push_back(vmstress);
  derived.push_back(magstress);
  
  return derived;
}
