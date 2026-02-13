/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include <exodusII.h>
#include <Teuchos_ParameterList.hpp>

#include "levelSet.hpp"
#include "preferences.hpp"
#include "rothermal.hpp"

using namespace MrHyDE;


// ========================================================================================
// constructor
// ========================================================================================
template<class EvalT>
levelSet<EvalT>::levelSet(Teuchos::ParameterList & settings, const int & dimension_)
  : PhysicsBase<EvalT>(settings, dimension_)
{
  
  label = "levelSet";

  settings_ = settings;

  myvars.push_back("phi");
  mybasistypes.push_back("HGRAD");

  // in the case rothermal is true, instiatinite rothermal object and pass settings
  useRothermal = settings.get<bool>("use rothermal",false);
  if (useRothermal) {
    auto & rothermalSettings = settings.sublist("rothermal");
    rothermal_       = Teuchos::rcp(new rothermal<EvalT>(rothermalSettings));
    this->haveNodalR = rothermalSettings.get<bool>("use_Rnodal", false);
  }

}


template<class EvalT>
void levelSet<EvalT>::defineFunctions(
  Teuchos::ParameterList & fs,
  Teuchos::RCP<FunctionManager<EvalT> > & functionManager_
)
{
  functionManager = functionManager_;

  functionManager->addFunction("beta", fs.get<string>("beta","0.1"), "ip");
  functionManager->addFunction("xvel", fs.get<string>("xvel","1.0"), "ip");
  functionManager->addFunction("yvel", fs.get<string>("yvel","1.0"), "ip");

  if (useRothermal) {

    Teuchos::ParameterList & rlist = fs.sublist("rothermal");

    haveAnalyticR = fs.isParameter("R");
    useExternalR  = haveAnalyticR || haveNodalR;



    if (haveNodalR) {
      rothermal_->readNodalR();
    }


    // keep original binding; if you later override in prepareFunctions(), this is ignored
    functionManager->addFunction("R", fs.get<string>("R", "1.0"), "ip");

    // scaling (unchanged)
    const double diag       = rothermal_->meshData.Diag;
    const double hs         = rlist.get<double>("heavisideScale", 10.0);
    const double Hphi_scale = 2.0 * hs / diag;
    const string s          = std::to_string(Hphi_scale);
    functionManager->addFunction("Hphi_scale", s, "ip");
    functionManager->addFunction("Hphi_scale", s, "point");
  }
}



template<class EvalT>
typename levelSet<EvalT>::template FuncData<EvalT>
levelSet<EvalT>::prepareFunctions()
{
  FuncData<EvalT> funcData;

  {
    Teuchos::TimeMonitor funceval(*volumeResidualFunc);
    funcData.beta = functionManager->evaluate("beta","ip");
    funcData.xvel = functionManager->evaluate("xvel","ip");
    funcData.yvel = functionManager->evaluate("yvel","ip");
  }

  if (useRothermal) {
    funcData.R = functionManager->evaluate("R","ip");
    if (haveNodalR && !rothermal_->Rnodal_host_.empty()) {
      funcData.R = rothermal_->evalNodalR(wkset);
    }
  }
  return funcData;
}


// ========================================================================================
// prepare fields for use in volume residual
// ========================================================================================
template<class EvalT>
typename levelSet<EvalT>::template FieldData<EvalT> levelSet<EvalT>::prepareFields()
{

    FieldData<EvalT> fieldData;

    fieldData.phi     = wkset->getSolutionField("phi");
    fieldData.phi_t   = wkset->getSolutionField("phi_t");
    fieldData.dphi_dx = wkset->getSolutionField("grad(phi)[x]");
    fieldData.dphi_dy = wkset->getSolutionField("grad(phi)[y]");
    fieldData.off     = Kokkos::subview(wkset->offsets, phinum, Kokkos::ALL());

    return fieldData;
}


// ========================================================================================
// volume residual
// ========================================================================================
template<class EvalT>
void levelSet<EvalT>::volumeResidual()
{

  // prepare functions and fields
  auto funcs = prepareFunctions();
  auto field = prepareFields();

  // retrieve workset data
  int phi_basis_num = wkset->usebasis[phinum];
  auto basis        = wkset->basis[phi_basis_num];   
  auto basis_grad   = wkset->basis_grad[phi_basis_num];
  auto h            = wkset->getElementSize();
  auto wts          = wkset->wts;
  auto res          = wkset->res;

  // retrieve functions
  auto beta        = funcs.beta; 
  auto xvel        = funcs.xvel;
  auto yvel        = funcs.yvel;
  
  // retrieve fields
  auto phi         = field.phi;
  auto dPhi_dt     = field.phi_t;
  auto dPhi_dx     = field.dphi_dx;
  auto dPhi_dy     = field.dphi_dy;
  auto off         = field.off;

  typename rothermal<EvalT>::ResponseFields fields;

  // if the flag is true, get the rothermal fields to compute velocity
  if (useRothermal) {

    fields = rothermal_->computeFields(field.phi,
                                       field.dphi_dx,
                                       field.dphi_dy,
                                       funcs.xvel,
                                       funcs.yvel,
                                       wkset,
                                       useExternalR,
                                       funcs.R);
  }  

  // loop over elements
  Kokkos::parallel_for("levelSet volume resid 2D",
                       RangePolicy<AssemblyExec>(0, wkset->numElem),
                       MRHYDE_LAMBDA(const int elem)
  {

    // element size
    auto he = h(elem);

    // initializing acumulating variables
    EvalT sum_grad = 0.0;
    EvalT sum_umag = 0.0;
    EvalT sum_beta = 0.0;
    EvalT sum_wts  = 0.0;

    // (pass 1) loop over quadrature points
    for (size_type pt = 0; pt < basis.extent(2); ++pt) {

      EvalT velMag;

      EvalT gradNorm = sqrt(dPhi_dx(elem, pt) * dPhi_dx(elem, pt) +
                            dPhi_dy(elem, pt) * dPhi_dy(elem, pt) +
                            zero_tol * zero_tol);


      // the velocity terms depends on if using rothermal
      if (useRothermal) {

        velMag = fields.ROS(elem, pt);

      } else {

        velMag = sqrt(xvel(elem, pt) * xvel(elem, pt) +
                      yvel(elem, pt) * yvel(elem, pt) +
                      zero_tol * zero_tol);

      }

      // acumulating values over element
      sum_grad     += gradNorm * wts(elem, pt);
      sum_umag     += velMag * wts(elem, pt);
      sum_beta     += beta(elem, pt) * wts(elem, pt);
      sum_wts      += wts(elem, pt);

    }

    // averages over element for P0 projection
    EvalT gradAvg  = sum_grad / sum_wts;
    EvalT velAvg   = sum_umag / sum_wts;
    EvalT beta_e   = sum_beta  / sum_wts;
    EvalT lambda2  = 0.5 * beta_e * he * he * velAvg;
    EvalT pe_const = lambda2 * (gradAvg - 1.0);


    // (pass 2) assemble volume residual
    for (size_type pt = 0; pt < basis.extent(2); ++pt) {


      // initialize velocity variables
      EvalT ux;
      EvalT uy;

      // gradient norm required for penalty term
      EvalT gradNorm = sqrt(dPhi_dx(elem, pt) * dPhi_dx(elem, pt) +
                            dPhi_dy(elem, pt) * dPhi_dy(elem, pt) +
                            zero_tol * zero_tol);
      EvalT invGrad = 1.0 / gradNorm;


      // the velocity terms depends on if using rothermal
      if (useRothermal) {
        ux = fields.ROS(elem, pt) * dPhi_dx(elem, pt) / gradNorm;
        uy = fields.ROS(elem, pt) * dPhi_dy(elem, pt) / gradNorm;
      } else {
        ux = xvel(elem, pt);
        uy = yvel(elem, pt);
      }

      // advection terms
      EvalT time_term   = dPhi_dt(elem, pt) * wts(elem, pt);
      EvalT adve_term_x = ux * dPhi_dx(elem, pt) * wts(elem, pt);
      EvalT adve_term_y = uy * dPhi_dy(elem, pt) * wts(elem, pt);

      // penalty term
      EvalT regx  = pe_const * (dPhi_dx(elem,pt) * invGrad) * wts(elem,pt);
      EvalT regy  = pe_const * (dPhi_dy(elem,pt) * invGrad) * wts(elem,pt);

      // stabilization terms
      EvalT stabres = dPhi_dt(elem, pt) + ux * dPhi_dx(elem, pt) + uy * dPhi_dy(elem, pt);
      EvalT tau     = computeTau(ux, uy, he);
      EvalT stabx   = tau * stabres * ux * wts(elem, pt);
      EvalT staby   = tau * stabres * uy * wts(elem, pt);

      // loop over basis functions
      for (size_type dof = 0; dof < basis.extent(1); ++dof) {
        
        res(elem, off(dof)) += time_term   * basis(elem, dof, pt, 0);
        res(elem, off(dof)) += adve_term_x * basis(elem, dof, pt, 0);
        res(elem, off(dof)) += adve_term_y * basis(elem, dof, pt, 0);

        res(elem, off(dof)) += regx * basis_grad(elem, dof, pt, 0);
        res(elem, off(dof)) += regy * basis_grad(elem, dof, pt, 1);

        res(elem, off(dof)) += stabx * basis_grad(elem, dof, pt, 0);
        res(elem, off(dof)) += staby * basis_grad(elem, dof, pt, 1);

      }
    }
  });
}

// ========================================================================================
// set workset
// ========================================================================================

template<class EvalT>
void levelSet<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
  wkset = wkset_;
  const auto & varlist = wkset->varlist;
  for (size_t i=0; i<varlist.size(); ++i) if (varlist[i] == "phi") phinum = i;
}


// ========================================================================================
// SUPG stabilization parameter
// ========================================================================================
template<class EvalT>
KOKKOS_FUNCTION EvalT levelSet<EvalT>::computeTau(
  const EvalT & xvl,
  const EvalT & yvl,
  const ScalarT & h
) const
{
  
  constexpr ScalarT C = 0.5;
  EvalT nvel          = 0.0;  
  nvel                = xvl*xvl + yvl*yvl;
  
  if (nvel > 1E-12)
    nvel = sqrt(nvel);
  
  return C * h / (2 * nvel);
}

// ========================================================================================
// get derived names
// ========================================================================================
template<class EvalT>
vector<string> levelSet<EvalT>::getDerivedNames()
{
  vector<string> names;
  names.push_back("gradNorm");
  if (useRothermal) {
    // names.push_back("fuelFraction");
    names.push_back("fuelCorrection");
    names.push_back("ROS");
    names.push_back("Rothermal");
    names.push_back("windComponent");
    names.push_back("windCorrection");
    names.push_back("slopeComponent");
    names.push_back("slopeCorrection");
    names.push_back("Hphi");
  }
  return names;
}

// ========================================================================================
// get derived values
// ========================================================================================
template<class EvalT>
vector<typename levelSet<EvalT>::View_EvalT2>
levelSet<EvalT>::getDerivedValues()
{

  // initialize vector of derived values
  vector<View_EvalT2> vals;

  // retrieve fields
  auto field = prepareFields();

  // retrieve basis data
  const int phi_basis_num = wkset->usebasis[phinum];
  auto basis              = wkset->basis[phi_basis_num];
  const int numElem       = wkset->numElem;
  const int numQuad       = static_cast<int>(basis.extent(2));

  // derivatives of phi
  auto dPhi_dx = field.dphi_dx;
  auto dPhi_dy = field.dphi_dy;

  // allocate memory for gradNorm
  View_EvalT2 gradNorm("gradNorm", numElem, numQuad);
  Kokkos::parallel_for("levelSet::computeGradNorm",
                       RangePolicy<AssemblyExec>(0, numElem),
                       MRHYDE_LAMBDA(const int e) {
    for (int q = 0; q < numQuad; ++q) {
      gradNorm(e,q) = sqrt(dPhi_dx(e,q) * dPhi_dx(e,q) + dPhi_dy(e,q) * dPhi_dy(e,q) + zero_tol);
    }
  });

  // add gradNorm to vector of derived values
  vals.push_back(gradNorm);

  // if using rothermal, compute derived values
  if (useRothermal && rothermal_) {

    // prepare functions
    auto funcs = prepareFunctions();

    // compute the rothermal fields
    auto F = rothermal_->computeFields(field.phi,
                              field.dphi_dx,
                              field.dphi_dy,
                              funcs.xvel,
                              funcs.yvel,
                              wkset,
                              useExternalR,
                              funcs.R);

    // add derived values to vector of derived values
    // vals.push_back(F.fuelFraction);
    vals.push_back(F.fuelCorrection);
    vals.push_back(F.ROS);
    vals.push_back(F.Rothermal);
    vals.push_back(F.windComponent);
    vals.push_back(F.windCorrection);
    vals.push_back(F.slopeComponent);
    vals.push_back(F.slopeCorrection);
    vals.push_back(F.Hphi);
  }

  return vals;
}


template class MrHyDE::levelSet<ScalarT>;

#ifndef MrHyDE_NO_AD

template class MrHyDE::levelSet<AD>;

template class MrHyDE::levelSet<AD2>;
template class MrHyDE::levelSet<AD4>;
template class MrHyDE::levelSet<AD8>;
template class MrHyDE::levelSet<AD16>;
template class MrHyDE::levelSet<AD18>;
template class MrHyDE::levelSet<AD24>;
template class MrHyDE::levelSet<AD32>;
#endif
