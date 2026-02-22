/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/**
 * Compressible Neo-Hookean hyperelasticity module.
 *
 * Stored energy:
 *   W = (mu/2)(I1 - 3) + (lambda/2)(ln J)^2 - mu*ln(J)
 *
 * First Piola-Kirchhoff stress:
 *   P = mu*F + (lambda*ln(J) - mu)*F^{-T}
 *
 * Weak form (reference configuration):
 *   integral P : grad(v) dOmega_0 = integral f . v dOmega_0 + integral t . v dGamma_0
 *
 * Reduces to linear elasticity for small strains:
 *   sigma ~ lambda*(tr eps)*I + 2*mu*eps
 */

#include "neohookean.hpp"
#include <string>
using namespace MrHyDE;

typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_EvalT2;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
neohookean<EvalT>::neohookean(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "neohookean";
  
  spaceDim = dimension_;
  
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
  
  modelparams = Kokkos::View<ScalarT*,AssemblyDevice>("parameters for NH",2); 
  auto modelparams_host = Kokkos::create_mirror_view(modelparams); 
 
  modelparams_host(0) = settings.get<ScalarT>("form_param",1.0);
  modelparams_host(1) = settings.get<ScalarT>("penalty",10.0);
  
  Kokkos::deep_copy(modelparams, modelparams_host);
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void neohookean<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                                     Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  functionManager->addFunction("lambda",fs.get<string>("lambda","1.0"),"ip");
  functionManager->addFunction("mu",fs.get<string>("mu","0.5"),"ip");
  functionManager->addFunction("source dx",fs.get<string>("source dx","0.0"),"ip");
  functionManager->addFunction("source dy",fs.get<string>("source dy","0.0"),"ip");
  functionManager->addFunction("source dz",fs.get<string>("source dz","0.0"),"ip");
  functionManager->addFunction("lambda",fs.get<string>("lambda","1.0"),"side ip");
  functionManager->addFunction("mu",fs.get<string>("mu","0.5"),"side ip");
  
  stress_vol = View_EvalT4("PK1 stress tensor", functionManager->num_elem_,
                        functionManager->num_ip_, spaceDim, spaceDim);
  
  stress_side = View_EvalT4("PK1 stress tensor", functionManager->num_elem_,
                         functionManager->num_ip_side_, spaceDim, spaceDim);
  
}

// ========================================================================================
// Volume residual: integral P : grad(v) dOmega_0 - integral f . v dOmega_0
// The assembly structure is identical to linear elasticity since
// P_ij * dv_i/dX_j has the same index structure as sigma_ij * dv_i/dx_j
// ========================================================================================

template<class EvalT>
void neohookean<EvalT>::volumeResidual() {
  
  int spaceDim = wkset->dimension;
  Vista<EvalT> lambda, mu, source_dx, source_dy, source_dz;
  
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
  
  // fills in PK1 stress tensor
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
    
    parallel_for("NH volume resid 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,off(dof)) += (stress(elem,pt,0,0)*basis_grad(elem,dof,pt,0) - source_dx(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
        }
      }
    });
  }
  else if (spaceDim == 2) {
    
    {
      // first equation: integral P_0j * dv/dX_j
      int dx_basis = wkset->usebasis[dx_num];
      auto basis = wkset->basis[dx_basis];
      auto basis_grad = wkset->basis_grad[dx_basis];
      auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
      
      parallel_for("NH ux volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   MRHYDE_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (stress(elem,pt,0,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,0,1)*basis_grad(elem,dof,pt,1) - source_dx(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
    
    {
      // second equation: integral P_1j * dv/dX_j
      int dy_basis = wkset->usebasis[dy_num];
      auto basis = wkset->basis[dy_basis];
      auto basis_grad = wkset->basis_grad[dy_basis];
      auto off = Kokkos::subview( wkset->offsets, dy_num, Kokkos::ALL());
      
      parallel_for("NH uy volume resid 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   MRHYDE_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (stress(elem,pt,1,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,1,1)*basis_grad(elem,dof,pt,1) - source_dy(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
  }
  else if (spaceDim == 3) {
    
    // first equation: integral P_0j * dv/dX_j
    {
      int dx_basis = wkset->usebasis[dx_num];
      auto basis = wkset->basis[dx_basis];
      auto basis_grad = wkset->basis_grad[dx_basis];
      auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
      
      size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
      
      parallel_for("NH ux volume resid 3D",
                   TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
                   MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += (stress(elem,pt,0,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,0,1)*basis_grad(elem,dof,pt,1) + stress(elem,pt,0,2)*basis_grad(elem,dof,pt,2) - source_dx(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
    
    // second equation: integral P_1j * dv/dX_j
    {
      int dy_basis = wkset->usebasis[dy_num];
      auto basis = wkset->basis[dy_basis];
      auto basis_grad = wkset->basis_grad[dy_basis];
      auto off = Kokkos::subview( wkset->offsets, dy_num, Kokkos::ALL());
      
      size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
      
      parallel_for("NH uy volume resid 3D",
                   TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
                   MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += (stress(elem,pt,1,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,1,1)*basis_grad(elem,dof,pt,1) + stress(elem,pt,1,2)*basis_grad(elem,dof,pt,2) - source_dy(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
    
    // third equation: integral P_2j * dv/dX_j
    {
      int dz_basis = wkset->usebasis[dz_num];
      auto basis = wkset->basis[dz_basis];
      auto basis_grad = wkset->basis_grad[dz_basis];
      auto off = Kokkos::subview( wkset->offsets, dz_num, Kokkos::ALL());
      
      size_t teamSize = std::min(wkset->maxTeamSize,basis.extent(1));
      
      parallel_for("NH uz volume resid 3D",
                   TeamPolicy<AssemblyExec>(wkset->numElem, teamSize, VECTORSIZE),
                   MRHYDE_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
          for (size_type pt=0; pt<basis.extent(2); ++pt ) {
            res(elem,off(dof)) += (stress(elem,pt,2,0)*basis_grad(elem,dof,pt,0) + stress(elem,pt,2,1)*basis_grad(elem,dof,pt,1) + stress(elem,pt,2,2)*basis_grad(elem,dof,pt,2) - source_dz(elem,pt)*basis(elem,dof,pt,0))*wts(elem,pt);
          }
        }
      });
    }
  }
}

// ========================================================================================
// Boundary residual: Neumann and weak Dirichlet conditions
// For weak Dirichlet, we use the same Nitsche-type approach as linear elasticity
// with a penalty based on (lambda + 2*mu) as an approximation of the tangent modulus.
// ========================================================================================

template<class EvalT>
void neohookean<EvalT>::boundaryResidual() {
  
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
  
  Vista<EvalT> lambda_side, mu_side, source_dx, source_dy, source_dz;
  
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
    
    auto wts = wkset->wts_side;
    auto h = wkset->getSideElementSize();
    auto res = wkset->res;
    
    Teuchos::TimeMonitor localtime(*boundaryResidualFill);
    
    this->computeStress(lambda_side, mu_side, true);
    auto stress = stress_side;
    
    if (spaceDim == 1) {
      int dx_basis = wkset->usebasis[dx_num];
      auto basis = wkset->basis_side[dx_basis];
      auto basis_grad = wkset->basis_grad_side[dx_basis];
      auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
      auto nx = wkset->getScalarField("n[x]");
      if (dx_sidetype == "Neumann") {
        parallel_for("NH ux bndry resid 1D N",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     MRHYDE_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += (-source_dx(e,k)*basis(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
      else if (dx_sidetype == "weak Dirichlet") {
        auto dx = wkset->getSolutionField("dx");
        parallel_for("NH ux bndry resid 1D wD",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     MRHYDE_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
            EvalT deltadx = dx(e,k) - source_dx(e,k);
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
      else if (dx_sidetype == "interface") {
        auto dx = wkset->getSolutionField("dx");
        auto lambdax = wkset->getSolutionField("aux dx");
        parallel_for("NH ux bndry resid 1D wD-ms",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     MRHYDE_LAMBDA (const int e ) {
          for (size_type k=0; k<basis.extent(2); k++ ) {
            EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
            EvalT deltadx = dx(e,k) - lambdax(e,k);
            for (size_type i=0; i<basis.extent(1); i++ ) {
              res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0))*wts(e,k);
            }
          }
        });
      }
    }
    else if (spaceDim == 2) {
      auto nx = wkset->getScalarField("n[x]");
      auto ny = wkset->getScalarField("n[y]");
      
      // dx equation boundary residual
      {
        int dx_basis = wkset->usebasis[dx_num];
        auto basis = wkset->basis_side[dx_basis];
        auto basis_grad = wkset->basis_grad_side[dx_basis];
        auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
        
        if (dx_sidetype == "Neumann") {
          parallel_for("NH ux bndry resid 2D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dx(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == "weak Dirichlet") {
          auto dx = wkset->getSolutionField("dx");
          parallel_for("NH ux bndry resid 2D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltadx = dx(e,k) - source_dx(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k) - stress(e,k,0,1)*ny(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == "interface") {
          auto dx = wkset->getSolutionField("dx");
          auto lambdax = wkset->getSolutionField("aux dx");
          parallel_for("NH ux bndry resid 2D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltadx = dx(e,k) - lambdax(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k) - stress(e,k,0,1)*ny(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0))*wts(e,k);
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
        if (dy_sidetype == "Neumann") {
          parallel_for("NH uy bndry resid 2D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dy(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == "weak Dirichlet") {
          auto dy = wkset->getSolutionField("dy");
          parallel_for("NH uy bndry resid 2D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltady = dy(e,k) - source_dy(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*nx(e,k) - stress(e,k,1,1)*ny(e,k))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == "interface") {
          auto dy = wkset->getSolutionField("dy");
          auto lambday = wkset->getSolutionField("aux dy");
          parallel_for("NH uy bndry resid 2D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltady = dy(e,k) - lambday(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*nx(e,k) - stress(e,k,1,1)*ny(e,k))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
      }
    }
    
    else if (spaceDim == 3) {
      auto nx = wkset->getScalarField("n[x]");
      auto ny = wkset->getScalarField("n[y]");
      auto nz = wkset->getScalarField("n[z]");
      
      // dx equation boundary residual
      {
        int dx_basis = wkset->usebasis[dx_num];
        auto basis = wkset->basis_side[dx_basis];
        auto basis_grad = wkset->basis_grad_side[dx_basis];
        auto off = Kokkos::subview( wkset->offsets, dx_num, Kokkos::ALL());
        if (dx_sidetype == "Neumann") {
          parallel_for("NH ux bndry resid 3D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dx(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == "weak Dirichlet") {
          auto dx = wkset->getSolutionField("dx");
          parallel_for("NH ux bndry resid 3D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltadx = dx(e,k) - source_dx(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k) - stress(e,k,0,1)*ny(e,k) - stress(e,k,0,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dx_sidetype == "interface") {
          auto dx = wkset->getSolutionField("dx");
          auto lambdax = wkset->getSolutionField("aux dx");
          parallel_for("NH ux bndry resid 3D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltadx = dx(e,k) - lambdax(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,0,0)*nx(e,k) - stress(e,k,0,1)*ny(e,k) - stress(e,k,0,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltadx*basis(e,i,k,0))*wts(e,k);
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
        if (dy_sidetype == "Neumann") {
          parallel_for("NH uy bndry resid 3D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dy(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == "weak Dirichlet") {
          auto dy = wkset->getSolutionField("dy");
          parallel_for("NH uy bndry resid 3D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltady = dy(e,k) - source_dy(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*nx(e,k) - stress(e,k,1,1)*ny(e,k) - stress(e,k,1,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dy_sidetype == "interface") {
          auto dy = wkset->getSolutionField("dy");
          auto lambday = wkset->getSolutionField("aux dy");
          parallel_for("NH uy bndry resid 3D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltady = dy(e,k) - lambday(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,1,0)*nx(e,k) - stress(e,k,1,1)*ny(e,k) - stress(e,k,1,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltady*basis(e,i,k,0))*wts(e,k);
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
        if (dz_sidetype == "Neumann") {
          parallel_for("NH uz bndry resid 3D N",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += (-source_dz(e,k)*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dz_sidetype == "weak Dirichlet") {
          auto dz = wkset->getSolutionField("dz");
          parallel_for("NH uz bndry resid 3D wD",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltadz = dz(e,k) - source_dz(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,2,0)*nx(e,k) - stress(e,k,2,1)*ny(e,k) - stress(e,k,2,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltadz*basis(e,i,k,0))*wts(e,k);
              }
            }
          });
        }
        else if (dz_sidetype == "interface") {
          auto dz = wkset->getSolutionField("dz");
          auto lambdaz = wkset->getSolutionField("aux dz");
          parallel_for("NH uz bndry resid 3D wD-ms",
                       RangePolicy<AssemblyExec>(0,wkset->numElem),
                       MRHYDE_LAMBDA (const int e ) {
            for (size_type k=0; k<basis.extent(2); k++ ) {
              EvalT penalty = modelparams(1)*(lambda_side(e,k) + 2.0*mu_side(e,k))/h(e);
              EvalT deltadz = dz(e,k) - lambdaz(e,k);
              for (size_type i=0; i<basis.extent(1); i++ ) {
                res(e,off(i)) += ((-stress(e,k,2,0)*nx(e,k) - stress(e,k,2,1)*ny(e,k) - stress(e,k,2,2)*nz(e,k))*basis(e,i,k,0) + penalty*deltadz*basis(e,i,k,0))*wts(e,k);
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

template<class EvalT>
void neohookean<EvalT>::computeFlux() {
  
  int cside = wkset->currentside;
  string dx_sidetype = wkset->var_bcs(dx_num,cside);
  string dy_sidetype = "Dirichlet";
  string dz_sidetype = "Dirichlet";
  int spaceDim = wkset->dimension;
  if (spaceDim > 1) {
    dy_sidetype = wkset->var_bcs(dy_num,cside);
  }
  if (spaceDim > 2) {
    dz_sidetype = wkset->var_bcs(dz_num,cside);
  }
  
  Vista<EvalT> lambda_side, mu_side;
  {
    Teuchos::TimeMonitor localtime(*fluxFunc);
    lambda_side = functionManager->evaluate("lambda","side ip");
    mu_side = functionManager->evaluate("mu","side ip");
  }
  
  auto h = wkset->getSideElementSize();
  auto basis = wkset->basis_side[wkset->usebasis[dx_num]];
  
  {
    Teuchos::TimeMonitor localtime(*fluxFill);
    
    this->computeStress(lambda_side, mu_side, true);
    
    if (spaceDim == 1) {
      auto nx = wkset->getScalarField("n[x]");
      auto dx = wkset->getSolutionField("dx");
      Vista<EvalT> source_dx;
      if (dx_sidetype == "interface") {
        auto vsource_dx = wkset->getSolutionField("aux dx");
        source_dx = Vista<EvalT>(vsource_dx);
      }
      else if (dx_sidetype == "weak Dirichlet" || dx_sidetype == "Dirichlet") {
        source_dx = functionManager->evaluate("Dirichlet dx " + wkset->sidename,"side ip");
      }
      auto flux_x = subview(wkset->flux, ALL(), dx_num, ALL());
      parallel_for("NH flux 1D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   MRHYDE_LAMBDA (const int e ) {
        for (size_type k=0; k<flux_x.extent(1); k++) {
          EvalT penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
          flux_x(e,k) = 1.0*stress_side(e,k,0,0)*nx(e,k) + penalty*(source_dx(e,k)-dx(e,k));
        }
      });
    }
    else if (spaceDim == 2) {
      auto nx = wkset->getScalarField("n[x]");
      auto ny = wkset->getScalarField("n[y]");
      auto dx = wkset->getSolutionField("dx");
      auto dy = wkset->getSolutionField("dy");
      Vista<EvalT> source_dx, source_dy;
      if (dx_sidetype == "interface") {
        auto vsource_dx = wkset->getSolutionField("aux dx");
        source_dx = Vista<EvalT>(vsource_dx);
      }
      else if (dx_sidetype == "weak Dirichlet" || dx_sidetype == "Dirichlet") {
        source_dx = functionManager->evaluate("Dirichlet dx " + wkset->sidename,"side ip");
      }
      if (dy_sidetype == "interface") {
        auto vsource_dy = wkset->getSolutionField("aux dy");
        source_dy = Vista<EvalT>(vsource_dy);
      }
      else if (dy_sidetype == "weak Dirichlet" || dy_sidetype == "Dirichlet") {
        source_dy = functionManager->evaluate("Dirichlet dy " + wkset->sidename,"side ip");
      }
      auto flux_x = subview( wkset->flux, ALL(), dx_num, ALL());
      auto flux_y = subview( wkset->flux, ALL(), dy_num, ALL());
      parallel_for("NH flux 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   MRHYDE_LAMBDA (const int e ) {
        for (size_type k=0; k<flux_x.extent(1); k++) {
          EvalT penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
          flux_x(e,k) = 1.0*(stress_side(e,k,0,0)*nx(e,k) + stress_side(e,k,0,1)*ny(e,k)) + penalty*(source_dx(e,k)-dx(e,k));
          flux_y(e,k) = 1.0*(stress_side(e,k,1,0)*nx(e,k) + stress_side(e,k,1,1)*ny(e,k)) + penalty*(source_dy(e,k)-dy(e,k));
        }
      });
    }
    else if (spaceDim == 3) {
      auto nx = wkset->getScalarField("n[x]");
      auto ny = wkset->getScalarField("n[y]");
      auto nz = wkset->getScalarField("n[z]");
      auto dx = wkset->getSolutionField("dx");
      auto dy = wkset->getSolutionField("dy");
      auto dz = wkset->getSolutionField("dz");
      
      Vista<EvalT> source_dx, source_dy, source_dz;
      bool compute_dx = true, compute_dy = true, compute_dz = true;
      
      if (dx_sidetype == "interface") {
        auto vsource_dx = wkset->getSolutionField("aux dx");
        source_dx = Vista<EvalT>(vsource_dx);
      }
      else if (dx_sidetype == "weak Dirichlet" || dx_sidetype == "Dirichlet") {
        source_dx = functionManager->evaluate("Dirichlet dx " + wkset->sidename,"side ip");
      }
      else { compute_dx = false; }
      
      if (dy_sidetype == "interface") {
        auto vsource_dy = wkset->getSolutionField("aux dy");
        source_dy = Vista<EvalT>(vsource_dy);
      }
      else if (dy_sidetype == "weak Dirichlet" || dy_sidetype == "Dirichlet") {
        source_dy = functionManager->evaluate("Dirichlet dy " + wkset->sidename,"side ip");
      }
      else { compute_dy = false; }
      
      if (dz_sidetype == "interface") {
        auto vsource_dz = wkset->getSolutionField("aux dz");
        source_dz = Vista<EvalT>(vsource_dz);
      }
      else if (dz_sidetype == "weak Dirichlet" || dz_sidetype == "Dirichlet") {
        source_dz = functionManager->evaluate("Dirichlet dz " + wkset->sidename,"side ip");
      }
      else { compute_dz = false; }
      
      if (compute_dx) {
        auto flux_x = subview( wkset->flux, ALL(), dx_num, ALL());
        parallel_for("NH flux 3D dx",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     MRHYDE_LAMBDA (const int e ) {
          for (size_type k=0; k<flux_x.extent(1); k++) {
            EvalT penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
            flux_x(e,k) = 1.0*(stress_side(e,k,0,0)*nx(e,k) + stress_side(e,k,0,1)*ny(e,k) + stress_side(e,k,0,2)*nz(e,k)) + penalty*(source_dx(e,k)-dx(e,k));
          }
        });
      }
      if (compute_dy) {
        auto flux_y = subview( wkset->flux, ALL(), dy_num, ALL());
        parallel_for("NH flux 3D dy",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     MRHYDE_LAMBDA (const int e ) {
          for (size_type k=0; k<flux_y.extent(1); k++) {
            EvalT penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
            flux_y(e,k) = 1.0*(stress_side(e,k,1,0)*nx(e,k) + stress_side(e,k,1,1)*ny(e,k) + stress_side(e,k,1,2)*nz(e,k)) + penalty*(source_dy(e,k)-dy(e,k));
          }
        });
      }
      if (compute_dz) {
        auto flux_z = subview( wkset->flux, ALL(), dz_num, ALL());
        parallel_for("NH flux 3D dz",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     MRHYDE_LAMBDA (const int e ) {
          for (size_type k=0; k<flux_z.extent(1); k++) {
            EvalT penalty = modelparams(1)/h(e)*(lambda_side(e,k) + 2.0*mu_side(e,k));
            flux_z(e,k) = 1.0*(stress_side(e,k,2,0)*nx(e,k) + stress_side(e,k,2,1)*ny(e,k) + stress_side(e,k,2,2)*nz(e,k)) + penalty*(source_dz(e,k)-dz(e,k));
          }
        });
      }
    }
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void neohookean<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;
  
  vector<string> varlist = wkset->varlist;
  dx_num = -1;
  dy_num = -1;
  dz_num = -1;
  for (size_t i=0; i<varlist.size(); i++) {
    if (varlist[i] == "dx")
      dx_num = i;
    else if (varlist[i] == "dy")
      dy_num = i;
    else if (varlist[i] == "dz")
      dz_num = i;
  }
}

// ========================================================================================
// Compute the first Piola-Kirchhoff stress:
//   P = mu*F + (lambda*ln(J) - mu)*F^{-T}
//
// where:
//   F = I + grad(u)            deformation gradient
//   J = det(F)                 volume ratio
//   F^{-T} = (1/J)*cof(F)     inverse transpose via cofactor matrix
// ========================================================================================

template<class EvalT>
void neohookean<EvalT>::computeStress(Vista<EvalT> lambda, Vista<EvalT> mu, const bool & onside) {
  
  Teuchos::TimeMonitor localtime(*fillStress);
  
  int spaceDim = wkset->dimension;
  
  // Select volume or side stress tensor
  auto stress = onside ? stress_side : stress_vol;
  
  if (spaceDim == 1) {
    // F = 1 + du/dx (scalar)
    // J = F
    // F^{-T} = 1/F
    // P = mu*F + (lambda*ln(F) - mu)/F
    auto ddx_dx = wkset->getSolutionField("grad(dx)[x]");
    
    parallel_for("NH stress 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int e ) {
      for (size_type k=0; k<stress.extent(1); k++) {
        EvalT F11 = 1.0 + ddx_dx(e,k);
        EvalT J = F11;
        EvalT lnJ = log(J);
        EvalT coeff = lambda(e,k)*lnJ - mu(e,k);
        stress(e,k,0,0) = mu(e,k)*F11 + coeff/F11;
      }
    });
  }
  else if (spaceDim == 2) {
    // F = [[1+du/dx, du/dy],
    //      [dv/dx, 1+dv/dy]]
    // J = F00*F11 - F01*F10
    // cof(F) = [[F11, -F10], [-F01, F00]]
    // F^{-T} = cof(F)/J
    auto ddx_dx = wkset->getSolutionField("grad(dx)[x]");
    auto ddx_dy = wkset->getSolutionField("grad(dx)[y]");
    auto ddy_dx = wkset->getSolutionField("grad(dy)[x]");
    auto ddy_dy = wkset->getSolutionField("grad(dy)[y]");
    
    parallel_for("NH stress 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int e ) {
      for (size_type k=0; k<stress.extent(1); k++) {
        // Deformation gradient
        EvalT F00 = 1.0 + ddx_dx(e,k);
        EvalT F01 = ddx_dy(e,k);
        EvalT F10 = ddy_dx(e,k);
        EvalT F11 = 1.0 + ddy_dy(e,k);
        
        // Determinant
        EvalT J = F00*F11 - F01*F10;
        EvalT lnJ = log(J);
        
        // Coefficient for the F^{-T} term
        EvalT coeff = (lambda(e,k)*lnJ - mu(e,k)) / J;
        
        // Cofactor matrix (= J * F^{-T})
        // cof00 =  F11, cof01 = -F10
        // cof10 = -F01, cof11 =  F00
        
        // P = mu*F + coeff*cof(F)
        stress(e,k,0,0) = mu(e,k)*F00 + coeff*F11;
        stress(e,k,0,1) = mu(e,k)*F01 + coeff*(-F01); //coeff*(-F10);
        stress(e,k,1,0) = mu(e,k)*F10 + coeff*(-F10); //coeff*(-F01);
        stress(e,k,1,1) = mu(e,k)*F11 + coeff*F00;
      }
    });
  }
  else if (spaceDim == 3) {
    // F = [[1+du/dx, du/dy, du/dz],
    //      [dv/dx, 1+dv/dy, dv/dz],
    //      [dw/dx, dw/dy, 1+dw/dz]]
    // cof(F)_ij = cofactor of F_ji (i.e., cofactor matrix, not transposed)
    // F^{-T} = cof(F)/J
    auto ddx_dx = wkset->getSolutionField("grad(dx)[x]");
    auto ddx_dy = wkset->getSolutionField("grad(dx)[y]");
    auto ddx_dz = wkset->getSolutionField("grad(dx)[z]");
    auto ddy_dx = wkset->getSolutionField("grad(dy)[x]");
    auto ddy_dy = wkset->getSolutionField("grad(dy)[y]");
    auto ddy_dz = wkset->getSolutionField("grad(dy)[z]");
    auto ddz_dx = wkset->getSolutionField("grad(dz)[x]");
    auto ddz_dy = wkset->getSolutionField("grad(dz)[y]");
    auto ddz_dz = wkset->getSolutionField("grad(dz)[z]");
    
    parallel_for("NH stress 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int e ) {
      for (size_type k=0; k<stress.extent(1); k++) {
        // Deformation gradient F = I + grad(u)
        EvalT F00 = 1.0 + ddx_dx(e,k);
        EvalT F01 = ddx_dy(e,k);
        EvalT F02 = ddx_dz(e,k);
        EvalT F10 = ddy_dx(e,k);
        EvalT F11 = 1.0 + ddy_dy(e,k);
        EvalT F12 = ddy_dz(e,k);
        EvalT F20 = ddz_dx(e,k);
        EvalT F21 = ddz_dy(e,k);
        EvalT F22 = 1.0 + ddz_dz(e,k);
        
        // Cofactor matrix of F
        // cof(F)_ij = (-1)^{i+j} * M_ji where M_ji is the minor of F at (j,i)
        // This equals J * F^{-T}
        EvalT cof00 = F11*F22 - F12*F21;
        EvalT cof01 = F12*F20 - F10*F22;
        EvalT cof02 = F10*F21 - F11*F20;
        EvalT cof10 = F02*F21 - F01*F22;
        EvalT cof11 = F00*F22 - F02*F20;
        EvalT cof12 = F01*F20 - F00*F21;
        EvalT cof20 = F01*F12 - F02*F11;
        EvalT cof21 = F02*F10 - F00*F12;
        EvalT cof22 = F00*F11 - F01*F10;
        
        // Determinant via first row expansion
        EvalT J = F00*cof00 + F01*cof01 + F02*cof02;
        EvalT lnJ = log(J);
        
        // Coefficient: (lambda*ln(J) - mu) / J
        // since F^{-T} = cof(F)/J, we have
        // P = mu*F + (lambda*lnJ - mu)/J * cof(F)
        EvalT coeff = (lambda(e,k)*lnJ - mu(e,k)) / J;
        
        // First Piola-Kirchhoff stress: P = mu*F + coeff*cof(F)
        stress(e,k,0,0) = mu(e,k)*F00 + coeff*cof00;
        stress(e,k,0,1) = mu(e,k)*F01 + coeff*cof01;
        stress(e,k,0,2) = mu(e,k)*F02 + coeff*cof02;
        stress(e,k,1,0) = mu(e,k)*F10 + coeff*cof10;
        stress(e,k,1,1) = mu(e,k)*F11 + coeff*cof11;
        stress(e,k,1,2) = mu(e,k)*F12 + coeff*cof12;
        stress(e,k,2,0) = mu(e,k)*F20 + coeff*cof20;
        stress(e,k,2,1) = mu(e,k)*F21 + coeff*cof21;
        stress(e,k,2,2) = mu(e,k)*F22 + coeff*cof22;
      }
    });
  }
}

// ========================================================================================
// ========================================================================================

template<class EvalT>
std::vector<string> neohookean<EvalT>::getDerivedNames() {
  std::vector<string> derived;
  derived.push_back("VM stress");
  derived.push_back("MAG stress");
  return derived;
}

// ========================================================================================
// Derived values: convert P (first Piola-Kirchhoff) to Cauchy stress for visualization
//   sigma = (1/J) * P * F^T
// Then compute von Mises and magnitude from Cauchy stress.
// ========================================================================================

template<class EvalT>
std::vector<Kokkos::View<EvalT**,ContLayout,AssemblyDevice> > neohookean<EvalT>::getDerivedValues() {
  std::vector<View_EvalT2> derived;
  
  auto lambda = functionManager->evaluate("lambda","ip");
  auto mu = functionManager->evaluate("mu","ip");
  this->computeStress(lambda, mu, false);
  auto P = stress_vol;  // This is the PK1 stress

  View_EvalT2 vmstress("von mises stress",P.extent(0),P.extent(1));
  View_EvalT2 magstress("magnitude of stress",P.extent(0),P.extent(1));
  
  int dimension = wkset->dimension;
  
  if (dimension == 1) {
    auto ddx_dx = wkset->getSolutionField("grad(dx)[x]");
    parallel_for("NH derived fill 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<vmstress.extent(1); ++pt) {
        EvalT F11 = 1.0 + ddx_dx(elem,pt);
        EvalT J = F11;
        // sigma = (1/J)*P*F^T = (1/J)*P11*F11
        EvalT sxx = P(elem,pt,0,0)*F11 / J;
        vmstress(elem,pt) = sqrt(sxx*sxx);
        magstress(elem,pt) = sqrt(sxx*sxx);
      }
    });
  }
  else if (dimension == 2) {
    auto ddx_dx = wkset->getSolutionField("grad(dx)[x]");
    auto ddx_dy = wkset->getSolutionField("grad(dx)[y]");
    auto ddy_dx = wkset->getSolutionField("grad(dy)[x]");
    auto ddy_dy = wkset->getSolutionField("grad(dy)[y]");
    parallel_for("NH derived fill 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<vmstress.extent(1); ++pt) {
        EvalT F00 = 1.0 + ddx_dx(elem,pt);
        EvalT F01 = ddx_dy(elem,pt);
        EvalT F10 = ddy_dx(elem,pt);
        EvalT F11 = 1.0 + ddy_dy(elem,pt);
        EvalT J = F00*F11 - F01*F10;
        EvalT Jinv = 1.0/J;
        
        // Cauchy stress: sigma = (1/J)*P*F^T
        // sigma_ij = (1/J) * sum_k P_ik * F_jk
        EvalT sxx = Jinv*(P(elem,pt,0,0)*F00 + P(elem,pt,0,1)*F01);
        EvalT sxy = Jinv*(P(elem,pt,0,0)*F10 + P(elem,pt,0,1)*F11);
        EvalT syx = Jinv*(P(elem,pt,1,0)*F00 + P(elem,pt,1,1)*F01);
        EvalT syy = Jinv*(P(elem,pt,1,0)*F10 + P(elem,pt,1,1)*F11);
        
        // Symmetrize (should be symmetric by construction, but ensure numerically)
        EvalT sxy_avg = 0.5*(sxy + syx);
        
        vmstress(elem,pt) = sqrt(sxx*sxx - sxx*syy + syy*syy + 3.0*sxy_avg*sxy_avg);
        magstress(elem,pt) = sqrt(sxx*sxx + syy*syy);
      }
    });
  }
  else if (dimension == 3) {
    auto ddx_dx = wkset->getSolutionField("grad(dx)[x]");
    auto ddx_dy = wkset->getSolutionField("grad(dx)[y]");
    auto ddx_dz = wkset->getSolutionField("grad(dx)[z]");
    auto ddy_dx = wkset->getSolutionField("grad(dy)[x]");
    auto ddy_dy = wkset->getSolutionField("grad(dy)[y]");
    auto ddy_dz = wkset->getSolutionField("grad(dy)[z]");
    auto ddz_dx = wkset->getSolutionField("grad(dz)[x]");
    auto ddz_dy = wkset->getSolutionField("grad(dz)[y]");
    auto ddz_dz = wkset->getSolutionField("grad(dz)[z]");
    parallel_for("NH derived fill 3D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 MRHYDE_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<vmstress.extent(1); ++pt) {
        EvalT F00 = 1.0 + ddx_dx(elem,pt);
        EvalT F01 = ddx_dy(elem,pt);
        EvalT F02 = ddx_dz(elem,pt);
        EvalT F10 = ddy_dx(elem,pt);
        EvalT F11 = 1.0 + ddy_dy(elem,pt);
        EvalT F12 = ddy_dz(elem,pt);
        EvalT F20 = ddz_dx(elem,pt);
        EvalT F21 = ddz_dy(elem,pt);
        EvalT F22 = 1.0 + ddz_dz(elem,pt);
        
        EvalT J = F00*(F11*F22 - F12*F21)
                - F01*(F10*F22 - F12*F20)
                + F02*(F10*F21 - F11*F20);
        EvalT Jinv = 1.0/J;
        
        // Cauchy stress: sigma_ij = (1/J) * sum_k P_ik * F_jk
        EvalT sxx = Jinv*(P(elem,pt,0,0)*F00 + P(elem,pt,0,1)*F01 + P(elem,pt,0,2)*F02);
        EvalT sxy = Jinv*(P(elem,pt,0,0)*F10 + P(elem,pt,0,1)*F11 + P(elem,pt,0,2)*F12);
        EvalT sxz = Jinv*(P(elem,pt,0,0)*F20 + P(elem,pt,0,1)*F21 + P(elem,pt,0,2)*F22);
        EvalT syy = Jinv*(P(elem,pt,1,0)*F10 + P(elem,pt,1,1)*F11 + P(elem,pt,1,2)*F12);
        EvalT syz = Jinv*(P(elem,pt,1,0)*F20 + P(elem,pt,1,1)*F21 + P(elem,pt,1,2)*F22);
        EvalT szz = Jinv*(P(elem,pt,2,0)*F20 + P(elem,pt,2,1)*F21 + P(elem,pt,2,2)*F22);
        
        vmstress(elem,pt) = sqrt(0.5*((sxx-syy)*(sxx-syy) + (syy-szz)*(syy-szz) + (szz-sxx)*(szz-sxx)) + 3.0*(sxy*sxy+syz*syz+sxz*sxz) );
        magstress(elem,pt) = sqrt(sxx*sxx + syy*syy + szz*szz);
      }
    });
  }
  
  derived.push_back(vmstress);
  derived.push_back(magstress);
  
  return derived;
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::neohookean<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::neohookean<AD>;

// Standard built-in types
template class MrHyDE::neohookean<AD2>;
template class MrHyDE::neohookean<AD4>;
template class MrHyDE::neohookean<AD8>;
template class MrHyDE::neohookean<AD16>;
template class MrHyDE::neohookean<AD18>;
template class MrHyDE::neohookean<AD24>;
template class MrHyDE::neohookean<AD32>;
#endif
