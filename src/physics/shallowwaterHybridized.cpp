/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "shallowwaterHybridized.hpp"
using namespace MrHyDE;

// ========================================================================================
/* Constructor to set up the problem */
// ========================================================================================

template<class EvalT>
shallowwaterHybridized<EvalT>::shallowwaterHybridized(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "shallowwaterHybridized";

  // save spaceDim here because it is needed (potentially) before workset is finalized

  spaceDim = dimension_;

  // The state is defined by (H,Hu) where H is the depth and u is the velocity

  myvars.push_back("H");
  myvars.push_back("Hux");
  if (spaceDim > 1) {
    myvars.push_back("Huy");
  }

  // TODO appropriate types?
  // I think this is setup to do the fully hybridized
 
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

  // Stabilization options, need to choose one
  maxEVstab = settings.get<bool>("max EV stabilization",false);
  roestab = settings.get<bool>("Roe-like stabilization",false);

  if ( ! ( maxEVstab || roestab ) ) {
    std::cout << "Error: No stabilization method chosen! Specify in input file!" << std::endl;
  }

  // TODO Will need more so fix here
  // TODO apparently this functionality is not needed UPDATE
  modelparams = Kokkos::View<ScalarT*,AssemblyDevice>("parameters for shallow water",1);
  auto modelparams_host = Kokkos::create_mirror_view(modelparams);

  // Acceleration due to gravity  units are L/T^2 
  modelparams_host(gravity_mp_num) = settings.get<ScalarT>("g",9.81);

  Kokkos::deep_copy(modelparams, modelparams_host);

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowwaterHybridized<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                            Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
  
  functionManager = functionManager_;
  
  // If the bottom surface varies, user can add -g*h*dbdx to source Hux
  // and -g*h*dbdy to source Huy
  // See Fig 1 in Samii 2019
  functionManager->addFunction("source H",fs.get<string>("source H","0.0"),"ip");
  functionManager->addFunction("source Hux",fs.get<string>("source Hux","0.0"),"ip");
  if (spaceDim > 1) {
    functionManager->addFunction("source Huy",fs.get<string>("source Huy","0.0"),"ip");
  }

  // Storage for the flux vectors

  fluxes_vol  = View_EvalT4("flux", functionManager->num_elem_,
                         functionManager->num_ip_, spaceDim + 1, spaceDim); // neqn = spaceDim + 1
  fluxes_side = View_EvalT4("flux", functionManager->num_elem_,
                         functionManager->num_ip_side_, spaceDim + 1, spaceDim); // see above 

  // Storage for stabilization term/boundary flux

  // The stabilization term which completes the numerical flux along interfaces
  // \hat{F} \cdot n = F(\hat{S}) \cdot n + Stab(S,\hat{S}) \times (S - \hat{S}) 
  // We store all of Stab(S,\hat{S}) \times (S - \hat{S}) 
  
  // Additionally, this storage is used for the boundary flux B(\hat{S}).
  // This is needed by the computeFlux routine (see for more details).

  stab_bound_side = View_EvalT3("stab/boundary term", functionManager->num_elem_,
                             functionManager->num_ip_side_, spaceDim + 1); // see above 

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowwaterHybridized<EvalT>::volumeResidual() {
  
  vector<Vista<EvalT> > sourceterms;

  // TODO is this a good way to do this??
  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    sourceterms.push_back(functionManager->evaluate("source H","ip"));
    sourceterms.push_back(functionManager->evaluate("source Hux","ip"));
    if (spaceDim > 1) {
      sourceterms.push_back(functionManager->evaluate("source Huy","ip"));
    }

    // Update fluxes 
    this->computeFluxVector(false); // not on_side 
  }
  
  Teuchos::TimeMonitor resideval(*volumeResidualFill);
  auto wts = wkset->wts;
  auto res = wkset->res;
  auto varlist = wkset->varlist;

  // The flux storage is (numElem,numip,eqn,dimension)
  
  // outer loop over equations
  for (size_t iEqn=0; iEqn<varlist.size(); ++iEqn) {
    int basis_num = wkset->usebasis[iEqn];
    auto basis = wkset->basis[basis_num];
    auto basis_grad = wkset->basis_grad[basis_num];
    auto dSi_dt = wkset->getSolutionField(varlist[iEqn]+"_t");
    auto source_i = sourceterms[iEqn];
    auto off = subview(wkset->offsets,iEqn,ALL());

    if (spaceDim == 1) {

      // All equations are of the form
      // (v_i,d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (v_i,source)

      parallel_for("shallow water volume resid " + varlist[iEqn] + " 1D",
              RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += dSi_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,iEqn,0)*basis_grad(elem,dof,pt,0) 
                + source_i(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });

    }
    else if (spaceDim == 2) {

      // All equations are of the form
      // (v_i,d S_i/dt) - (dv_i/dx_1,F_{x,i}) - (dv_i/dx_2,F_{y,i}) - (v_i,source)

      parallel_for("shallow water volume resid " + varlist[iEqn] + " 2D",
              RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for( size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += dSi_dt(elem,pt)*basis(elem,dof,pt,0)*wts(elem,pt);
            res(elem,off(dof)) += -( fluxes_vol(elem,pt,iEqn,0)*basis_grad(elem,dof,pt,0) 
                + fluxes_vol(elem,pt,iEqn,1)*basis_grad(elem,dof,pt,1)
                + source_i(elem,pt)*basis(elem,dof,pt,0) )*wts(elem,pt);
          }
        }
      });
    }
  }

}

// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowwaterHybridized<EvalT>::boundaryResidual() {
  
  auto bcs = wkset->var_bcs;

  //int cside = wkset->currentside;

  // TODO do we need sources or sidetypes?

  {
    Teuchos::TimeMonitor funceval(*boundaryResidualFunc);

    // Update fluxes and stabilization term

    this->computeFluxVector(true); // on_side 
    this->computeStabilizationTerm();

  }

  auto wts = wkset->wts_side;
  auto h = wkset->h;
  auto res = wkset->res;
  auto varlist = wkset->varlist;

  Teuchos::TimeMonitor localtime(*boundaryResidualFill);

  // These are always needed
  auto nx = wkset->getScalarField("n[x]");
  auto fluxes = fluxes_side;
  auto stab = stab_bound_side;

  // all boundary contributions are of the form ( F(\hat{S}_i) \cdot n, v_i )
  // and ( StabTerm, v_i ).
  // The boundary conditions are enforced weakly with the trace variables
  // so we ALWAYS compute the aforementioned inner product here

  // outer loop over equations
  for (size_t iEqn=0; iEqn<varlist.size(); ++iEqn) {
  
    int basis_num = wkset->usebasis[iEqn];
    auto basis = wkset->basis_side[basis_num];
    auto off = subview(wkset->offsets,iEqn,ALL());
    
    if (spaceDim == 1) {
      parallel_for("shallow water boundary resid " + varlist[iEqn] + " 1D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += (fluxes(elem,pt,iEqn,0)*nx(elem,pt) + 
                stab(elem,pt,iEqn))*wts(elem,pt)*basis(elem,dof,pt,0);
          }
        }
      });
    }
    else if (spaceDim == 2) {

      // need ny
      auto ny = wkset->getScalarField("n[y]");

      parallel_for("shallow water boundary resid " + varlist[iEqn] + " 2D",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,off(dof)) += 
              (fluxes(elem,pt,iEqn,0)*nx(elem,pt) + fluxes(elem,pt,iEqn,1)*ny(elem,pt) +
                stab(elem,pt,iEqn))*wts(elem,pt)*basis(elem,dof,pt,0);
          }
        }
      });
    }
  }

}

// ========================================================================================
// The boundary/edge flux
// ========================================================================================

template<class EvalT>
void shallowwaterHybridized<EvalT>::computeFlux() {

  // see Samii 2019 for the details of the implementation
  // they take fluxes on the interfaces to be F(\hat{S}) \cdot n + Stab(S,\hat{S}) ( S - \hat{S} )
  // where Stab = L \| \Lambda \| R or \lambda_{max} I
  //
  // L \Lambda R is the eigendecomposition of J = dF(\hat{S})/d\hat{S} \cdot n
  // the flux Jacobian evaluated using the trace variables.
  //
  // This routine contributes F \cdot n on the interior skeleton and 
  // B(\hat{S}) (the boundary flux) on the boundary of the domain
  //
  // TODO Currently, the BCs are more or less coupled so if one side type 
  // indicates we are at a boundary, then ALL others follow.
  
  auto bcs = wkset->var_bcs;

  int cside = wkset->currentside;
  string sidetype = bcs(H_num,cside);

  size_t nVar = wkset->varlist.size();

  {
    Teuchos::TimeMonitor localtime(*fluxFunc);

    this->computeFluxVector(true); // on_side 
    // see note above
    if (sidetype == "Far-field" || sidetype == "Slip") {
      this->computeBoundaryTerm();
    } 
    else {
      this->computeStabilizationTerm();
    }
  }

  {
    Teuchos::TimeMonitor localtime(*fluxFill);

    if (sidetype == "Far-field" || sidetype == "Slip") {
      // just need to copy the boundary term

      auto interfaceFlux = wkset->flux;
      auto bound = stab_bound_side;

      parallel_for("Shallow water boundary flux copy",
                   RangePolicy<AssemblyExec>(0,wkset->numElem),
                   KOKKOS_LAMBDA (const int elem ) {
        for (int ieqn=0; ieqn<nVar; ++ieqn) {
          for (size_type pt=0; pt<bound.extent(1); ++pt) {
            interfaceFlux(elem,ieqn,pt) = bound(elem,pt,ieqn);
          }
        }
      });
    }
    else if (sidetype == "interface") {
      
      // These are always needed
      auto nx = wkset->getScalarField("n[x]");

      auto fluxes = fluxes_side;
      auto stab = stab_bound_side;

      auto interfaceFlux = wkset->flux;

      if (spaceDim == 1) {

        parallel_for("Shallow water flux 1D",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_t iEqn=0; iEqn<nVar; ++iEqn) {
            for (size_type pt=0; pt<nx.extent(1); ++pt) {
              interfaceFlux(elem,iEqn,pt) = fluxes(elem,pt,iEqn,0)*nx(elem,pt)
                + stab(elem,pt,iEqn);
            }
          }
        });
      } 
      else if (spaceDim == 2) {
        // second normal needed
        auto ny = wkset->getScalarField("n[y]");

        parallel_for("Shallow water flux 2D",
                     RangePolicy<AssemblyExec>(0,wkset->numElem),
                     KOKKOS_LAMBDA (const int elem ) {
          for (size_t iEqn=0; iEqn<nVar; ++iEqn) {
            for (size_type pt=0; pt<nx.extent(1); ++pt) {
              interfaceFlux(elem,iEqn,pt) = fluxes(elem,pt,iEqn,0)*nx(elem,pt)
                + fluxes(elem,pt,iEqn,1)*ny(elem,pt) + stab(elem,pt,iEqn);
            }
          }
        });
      } 
      
    } 
    else {
      cout << "Something's gone wrong... shallowwaterHybridized computeFlux()" << endl;
    }
  }
}

// ========================================================================================
// ========================================================================================
// ========================================================================================
// ========================================================================================

template<class EvalT>
void shallowwaterHybridized<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {

  wkset = wkset_;

  // TODO make this less hard codey?
  vector<string> varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); ++i) {
    if (varlist[i] == "H")
      H_num = i;
    else if (varlist[i] == "Hux")
      Hux_num = i;
    else if (varlist[i] == "Huy")
      Huy_num = i;
  }

  // TODO make this less hard codey?
  vector<string> auxvarlist = wkset->aux_varlist;
  for (size_t i=0; i<auxvarlist.size(); ++i) {
    if (auxvarlist[i] == "H")
      auxH_num = i;
    else if (auxvarlist[i] == "Hux")
      auxHux_num = i;
    else if (auxvarlist[i] == "Huy")
      auxHuy_num = i;
  }

}

// ========================================================================================
// compute the fluxes
// ========================================================================================

template<class EvalT>
void shallowwaterHybridized<EvalT>::computeFluxVector(const bool & on_side) {

  Teuchos::TimeMonitor localtime(*fluxVectorFill);

  // The flux storage is (numElem,numip,eqn,dimension)
  // The face fluxes are defined in terms of the trace variables
  // The volume fluxes are defined in terms of the state
  // TODO does being on a side == interface for these schemes?
  // will we require data in any other context....

  auto fluxes = on_side ? fluxes_side : fluxes_vol;
  // these are always needed
  auto H = on_side ? wkset->getSolutionField("aux H") : wkset->getSolutionField("H");
  auto Hux = on_side ? wkset->getSolutionField("aux Hux") : wkset->getSolutionField("Hux");
  
  // TODO this is the same for face or side, can I collapse?

  if (spaceDim == 1) {

    parallel_for("shallow water fluxes 1D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // H equation -- F_x = Hux

        fluxes(elem,pt,H_num,0) = Hux(elem,pt);

        // Hux equation -- F_x = Hux*Hux/H + 1/2 g H*H

        fluxes(elem,pt,Hux_num,0) = 
          Hux(elem,pt)*Hux(elem,pt)/H(elem,pt) + 
            .5*H(elem,pt)*H(elem,pt)*modelparams(gravity_mp_num);

      }
    });
  } 
  else if (spaceDim == 2) {
    // get the second scaled velocity component
    auto Huy = on_side ? wkset->getSolutionField("aux Huy") : wkset->getSolutionField("Huy");

    parallel_for("shallow water fluxes 2D",
                 RangePolicy<AssemblyExec>(0,wkset->numElem),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<fluxes.extent(1); ++pt) {

        // H equation -- F_x = Hux F_y = Huy

        fluxes(elem,pt,H_num,0) = Hux(elem,pt);
        fluxes(elem,pt,H_num,1) = Huy(elem,pt);

        // Hux equation -- F_x = Hux**2/H + 1/2 g H*H F_y = Hux*Huy/H

        fluxes(elem,pt,Hux_num,0) = 
          Hux(elem,pt)*Hux(elem,pt)/H(elem,pt) + 
            .5*H(elem,pt)*H(elem,pt)*modelparams(gravity_mp_num);
        fluxes(elem,pt,Hux_num,1) = Hux(elem,pt)*Huy(elem,pt)/H(elem,pt);

        // Huy equation -- F_x = Hux*Huy/H F_y = Huy**2/H + 1/2 g H*H

        fluxes(elem,pt,Huy_num,0) = 
          Hux(elem,pt)*Huy(elem,pt)/H(elem,pt); 
        fluxes(elem,pt,Huy_num,1) = 
          Huy(elem,pt)*Huy(elem,pt)/H(elem,pt) + 
            .5*H(elem,pt)*H(elem,pt)*modelparams(gravity_mp_num);

      }
    });

  } 

}

// ========================================================================================
// compute the stabilization term
// ========================================================================================

template<class EvalT>
void shallowwaterHybridized<EvalT>::computeStabilizationTerm() {

  // The two proposed stabilization matrices in Samii et al. are based off of eigendecompositions
  // of the flux Jacobians
  // Let A = dF_x(\hat{S})/d\hat{S} n_x + dF_y(\hat{S})/d\hat{S} n_y
  // Then A = R Lambda L.
  //
  // The two options are Stab = R \| Lambda \| L or Stab = \lambda_max I
  // where \lambda_max is the maximum of \| Lambda \| (the absolute values of Lambda).
  //
  // Here we are interested in computing the result Stab \times (S - \hat{S})
  
  Teuchos::TimeMonitor localtime(*stabCompFill);

  using namespace std;
  
  // these are always needed
  auto H = wkset->getSolutionField("H");
  auto H_hat = wkset->getSolutionField("aux H");
  auto Hux = wkset->getSolutionField("Hux");
  auto Hux_hat = wkset->getSolutionField("aux Hux");

  auto stabterm = stab_bound_side;
  auto nx = wkset->getScalarField("n[x]");

  View_EvalT2 Huy, Huy_hat; // only assign if necessary
  View_Sc2 ny;

  size_t nVar = wkset->varlist.size();

  if (spaceDim > 1) {
    Huy = wkset->getSolutionField("Huy");
    Huy_hat = wkset->getSolutionField("aux Huy");
    ny = wkset->getScalarField("n[y]");
  }

  parallel_for("euler stabilization",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {

    View_EvalT2 leftEV,rightEV; // Local eigendecomposition
    View_EvalT1 Lambda; // diagonal matrix
    View_EvalT1 deltaS; // S - \hat{S} vector
    View_EvalT1 tmp; // temporary vector

    deltaS = View_EvalT1("delta S", nVar);

    if (roestab) {
      Lambda = View_EvalT1("Lambda", nVar); 
      leftEV = View_EvalT2("left EV", nVar, nVar); 
      rightEV = View_EvalT2("right EV", nVar, nVar); 
      tmp = View_EvalT1("tmp", nVar);
    }
    
    for (size_type pt=0; pt<stabterm.extent(1); ++pt) {

      // get the appropriate portion of the stability term
      auto stab_sub = Kokkos::subview( stabterm, elem, pt, Kokkos::ALL());

      // form (S - \hat{S})
      deltaS(H_num) = H(elem,pt) - H_hat(elem,pt);
      deltaS(Hux_num) = Hux(elem,pt) - Hux_hat(elem,pt);

      if (spaceDim > 1) {
        deltaS(Huy_num) = Huy(elem,pt) - Huy_hat(elem,pt);
      }

      if (roestab) {
        // fill the stabilization matrices
        if (spaceDim == 1) {
          this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              Hux_hat(elem,pt),H_hat(elem,pt));
        } else if (spaceDim == 2) {
          this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              Hux_hat(elem,pt),Huy_hat(elem,pt),H_hat(elem,pt),
              nx(elem,pt),ny(elem,pt));
        } 

        this->matVec(leftEV,deltaS,tmp); // L deltaS --> tmp
        // hit with the absolute value of the diagonal matrix
        for (int i=0; i<nVar; ++i) {
          tmp(i) *= abs( Lambda(i) );
        }
        // R tmp = R AbsLambda L deltaS --> stab_sub 
        this->matVec(rightEV,tmp,stab_sub);

      } else {
        // the stabilization is just the max abs EV times delta S
        EvalT vn = nx(elem,pt)*Hux_hat(elem,pt)/H_hat(elem,pt);
        EvalT a = sqrt(H_hat(elem,pt) * modelparams(gravity_mp_num));
        if (spaceDim > 1) vn += ny(elem,pt)*Huy_hat(elem,pt)/H_hat(elem,pt);
        // max of | vn + a |, | vn - a |
        // a = sqrt{gh}
        EvalT lambdaMax = max(abs(vn + a),abs(vn - a));

        for (int i=0; i<nVar; ++i) {
          stab_sub(i) = deltaS(i) * lambdaMax;
        }
      }
    }
  });
}

// ========================================================================================
// compute the boundary flux
// ========================================================================================

template<class EvalT>
void shallowwaterHybridized<EvalT>::computeBoundaryTerm() {

  // The two BC types in Peraire 2011/Samii 2019 are inflow/outflow and slip.
  //
  // The inflow/outflow BC is based off of eigendecompositions
  // of the flux Jacobians
  // Let A = dF_x(\hat{S})/d\hat{S} n_x + dF_y(\hat{S})/d\hat{S} n_y
  // Then A = R Lambda L.
  //
  // B = A^+ ( S - \hat{S} ) - A^- ( S_\infty - \hat{S} )
  // A^{\pm} = ( A \pm | A | ) / 2
  // S_\infty is the freestream condition
  
  using namespace std;
  
  Teuchos::TimeMonitor localtime(*boundCompFill);

  auto bcs = wkset->var_bcs;

  int cside = wkset->currentside;
  // TODO since our BCs come in for the whole state, just need one... TODO CHECK CHECK GENERALIZE??
  string sidetype = bcs(H_num,cside);

  // TODO Periodic conditions?
  if ( (sidetype != "Far-field") && (sidetype != "Slip") 
        && (sidetype != "interface") ) {
    cout << "Error :: Shallow water module does not support your chosen boundary condition!" << endl;
  }

  // these are always needed
  auto H = wkset->getSolutionField("H");
  auto H_hat = wkset->getSolutionField("aux H");
  auto Hux = wkset->getSolutionField("Hux");
  auto Hux_hat = wkset->getSolutionField("aux Hux");

  auto boundterm = stab_bound_side;
  auto nx = wkset->getScalarField("n[x]");

  size_t nVar = wkset->varlist.size();

  View_EvalT2 Huy, Huy_hat; // and only assign if necessary?
  View_Sc2 ny;

  if (spaceDim > 1) {
    Huy = wkset->getSolutionField("Huy");
    Huy_hat = wkset->getSolutionField("aux Huy");
    ny = wkset->getScalarField("n[y]");
  }

  // Get the freestream info if needed
  Vista<EvalT> source_H, source_Hux, source_Huy;

  if (sidetype == "Far-field") { 
    source_H = functionManager->evaluate("Far-field H " + wkset->sidename,"side ip");
    source_Hux = functionManager->evaluate("Far-field Hux " + wkset->sidename,"side ip");
    if (spaceDim > 1) { 
      source_Huy = functionManager->evaluate("Far-field Huy " + wkset->sidename,"side ip");
    }
   
  }

  parallel_for("Shallow water boundary term",
               RangePolicy<AssemblyExec>(0,wkset->numElem),
               KOKKOS_LAMBDA (const int elem ) {

    View_EvalT2 leftEV,rightEV; // Local eigendecomposition
    View_EvalT1 Lambda; // diagonal matrix
    View_EvalT1 deltaS; // S - \hat{S} vector
    View_EvalT1 tmp; // temporary vector

    if (sidetype == "Far-field") {

      // allocate storage
      deltaS = View_EvalT1("delta S", nVar);
      Lambda = View_EvalT1("Lambda", nVar); 
      leftEV = View_EvalT2("left EV", nVar, nVar); 
      rightEV = View_EvalT2("right EV", nVar, nVar); 
      tmp = View_EvalT1("tmp", nVar);
    }
    
    for (size_type pt=0; pt<boundterm.extent(1); ++pt) {

      // get the appropriate portion of the boundary term
      // TODO Not sure about FORCING this to be View_EvalT1 here (and above)
      auto bound_sub = Kokkos::subview( boundterm, elem, pt, Kokkos::ALL());
      
      if (sidetype == "Far-field") {
        // get the local eigendecomposition
        if (spaceDim == 1) {
          this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              Hux_hat(elem,pt),H_hat(elem,pt));
        } else if (spaceDim == 2) {
          this->eigendecompFluxJacobian(leftEV,Lambda,rightEV,
              Hux_hat(elem,pt),Huy_hat(elem,pt),H_hat(elem,pt),
              nx(elem,pt),ny(elem,pt));
        }

        // form (S - \hat{S})
        deltaS(H_num) = H(elem,pt) - H_hat(elem,pt);
        deltaS(Hux_num) = Hux(elem,pt) - Hux_hat(elem,pt);

        if (spaceDim > 1) {
          deltaS(Huy_num) = Huy(elem,pt) - Huy_hat(elem,pt);
        }

        // A^+ = R ( Lambda + | Lambda | ) L / 2
        // First, get A^+ \times delta S

        this->matVec(leftEV,deltaS,tmp); // L deltaS --> tmp
        // hit with the diagonal matrix
        for (int i=0; i<nVar; ++i) {
          tmp(i) *= ( Lambda(i) + abs( Lambda(i) ) ) / 2.;
        }
        // R tmp = A^+ deltaS --> bound_sub 
        this->matVec(rightEV,tmp,bound_sub);

        // now form ( S_\infty - \hat{S} )

        deltaS(H_num) = source_H(elem,pt) - H_hat(elem,pt);
        deltaS(Hux_num) = source_Hux(elem,pt) - Hux_hat(elem,pt);

        if (spaceDim > 1) {
          deltaS(Huy_num) = source_Huy(elem,pt) - Huy_hat(elem,pt);
        }

        // A^- = R ( Lambda - | Lambda | ) L / 2
        // First, get A^- \times delta S

        this->matVec(leftEV,deltaS,tmp); // L deltaS --> tmp
        // hit with the diagonal matrix
        for (int i=0; i<nVar; ++i) {
          tmp(i) *= ( Lambda(i) - abs( Lambda(i) ) ) / 2.;
        }
        // R tmp = A^- deltaS --> deltaS
        this->matVec(rightEV,tmp,deltaS);

        // finalize
        
        for (int i=0; i<nVar; ++i) {
          bound_sub(i) -= deltaS(i);
        }

      } else {
        // Apply the slip condition
        EvalT vn = nx(elem,pt)*Hux(elem,pt)/H(elem,pt);
        if (spaceDim > 1) vn += ny(elem,pt)*Huy(elem,pt)/H(elem,pt);

        // Depth
        
        bound_sub(H_num) = H(elem,pt) - H_hat(elem,pt);

        // force normal velocity to be zero

        bound_sub(Hux_num) = 
          ( Hux(elem,pt)/H(elem,pt) - vn*nx(elem,pt) ) - Hux_hat(elem,pt)/H_hat(elem,pt);

        if (spaceDim > 1) {
          bound_sub(Huy_num) = 
            ( Huy(elem,pt)/H(elem,pt) - vn*ny(elem,pt) ) - Huy_hat(elem,pt)/H_hat(elem,pt);
        }
      }
    }
  });
}

// ========================================================================================
// Fill in the local eigendecomposition matrices
// ========================================================================================

template<class EvalT>
KOKKOS_FUNCTION void shallowwaterHybridized<EvalT>::eigendecompFluxJacobian(View_EvalT2 leftEV, View_EvalT1 Lambda, View_EvalT2 rightEV, 
        const EvalT & Hux, const EvalT & H) {

  // In 1D, the eigenvalues are ux - a and ux + a
  // The right eigenvectors are 
  // [1, ux - a]^T
  // [1, ux + a]^T
  //
  // The corresponding left eigenvectors are TODO CHECK
  // 1/2a[ux + a, -1]
  // 1/2a[-ux + a, 1]

  // TODO CHECK BELOW 

  EvalT a = sqrt(H*modelparams(gravity_mp_num));
  
  rightEV(0,0) = 1.; rightEV(1,0) = Hux/H - a; 
  rightEV(0,1) = 1.; rightEV(1,1) = Hux/H + a;

  leftEV(0,0) = (Hux/H + a)/(2.*a); leftEV(0,1) = -1./(2.*a);
  leftEV(1,0) = (a - Hux/H)/(2.*a); leftEV(1,1) =  1./(2.*a);

  Lambda(0) = Hux/H - a; Lambda(1) = Hux/H + a; 
  
}

template<class EvalT>
KOKKOS_FUNCTION void shallowwaterHybridized<EvalT>::eigendecompFluxJacobian(View_EvalT2 leftEV, View_EvalT1 Lambda, View_EvalT2 rightEV, 
    const EvalT & Hux, const EvalT & Huy, const EvalT & H, const ScalarT & nx, const ScalarT & ny) {

  EvalT vn = Hux/H*nx + Huy/H*ny;

  EvalT a = sqrt(H*modelparams(gravity_mp_num));

  // With eigenvalues vn + a, vn, vn - a (flipped order from above)
  // See e.g. Li, Liu (2001) -- note there is a typo in their expression
  // for (A,B) \cdot n (entry 2,3 should be u n_y)

  // The right EVs are
  // [1, ux + a*nx, uy + a*ny]^T
  // [0, -a*ny, a*nx]^T
  // [1, ux - a*nx, uv - a*ny]^T

  // The left EVs are 
  // 1/(2*a)*[a - vn, nx, ny]
  // 1/(2*a)*[2*(ux*ny - uy*nx), -2*ny, 2*nx]
  // 1/(2*a)*[a + vn, -nx, -ny] 

  rightEV(0,0) = 1.; rightEV(1,0) = Hux/H + a*nx; rightEV(2,0) = Huy/H + a*ny;
  rightEV(0,1) = 0.; rightEV(1,1) = -a*ny; rightEV(2,1) = a*nx;
  rightEV(0,2) = 1.; rightEV(1,2) = Hux/H - a*nx; rightEV(2,2) = Huy/H - a*ny;

  leftEV(0,0) = .5 - vn/(2.*a); leftEV(0,1) =  nx/(2.*a); leftEV(0,2) =  ny/(2.*a);
  leftEV(1,0) = (ny*Hux/H - nx*Huy/H)/a; leftEV(1,1) = -ny/a; leftEV(1,2) = nx/a; 
  leftEV(2,0) = .5 + vn/(2.*a); leftEV(2,1) = -nx/(2.*a); leftEV(2,2) = -ny/(2.*a);

  Lambda(0) = vn + a; Lambda(1) = vn; Lambda(2) = vn - a;

}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::shallowwaterHybridized<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::shallowwaterHybridized<AD>;

// Standard built-in types
template class MrHyDE::shallowwaterHybridized<AD2>;
template class MrHyDE::shallowwaterHybridized<AD4>;
template class MrHyDE::shallowwaterHybridized<AD8>;
template class MrHyDE::shallowwaterHybridized<AD16>;
template class MrHyDE::shallowwaterHybridized<AD18>;
template class MrHyDE::shallowwaterHybridized<AD24>;
template class MrHyDE::shallowwaterHybridized<AD32>;
#endif
