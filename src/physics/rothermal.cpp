/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "rothermal.hpp"
#include "preferences.hpp"
#include <Kokkos_Complex.hpp>

using namespace MrHyDE;

// ===============================
// constructor
// ===============================
template<class EvalT>
rothermal<EvalT>::rothermal(Teuchos::ParameterList& p)
  : rothermalSettings_(p)
{
  std::cout << "Entering rothermalt<>::constructor()" << std::endl;  

  initScalars();         // constants and hyperparameters
  initParameterFields(); // fuel and slope parameter fields
  getMeshData();         // mesh data: used to normalize the problem

//   haveNodalR = rothermalSettings_.get("use_Rnodal", false);
//   cout << "haveNodalR: " << haveNodalR << endl;

  std::cout << "Leaving rothermalt<>::constructor()" << std::endl;  
}


// ===============================
// initialize scalars
// ===============================
template<class EvalT>
void rothermal<EvalT>::initScalars()
{
  // fire dies when fuel fraction is below this threshold
  Fthresh = rothermalSettings_.get("Fthresh", Fthresh);

  // smoothing parameters
  absoluteValueScale = rothermalSettings_.get("absoluteValueScale", absoluteValueScale);
  heavisideScale     = rothermalSettings_.get("heavisideScale", heavisideScale);

  // constants for fuel, wind, and slope magnification/damping
  burnConstant       = rothermalSettings_.get("burnConstant", burnConstant);
  windConstant       = rothermalSettings_.get("windConstant", windConstant);
  slopeConstant      = rothermalSettings_.get("slopeConstant", slopeConstant);
}


// ===============================
// initialize parameter fields
// ===============================
template<class EvalT>
void rothermal<EvalT>::initParameterFields()
{

    // read in parameter fields from exodus file
    exoReader_ = std::make_unique<ExoReader>("mesh.exo");
    exoReader_->open();
    exoReader_->computeGridAndDomain();

    // save parameter fields as views
    distFields.slope  = exoReader_->readCellToViewT<View_Scalar1>("slope");
    distFields.xSlope = exoReader_->readCellToViewT<View_Scalar1>("xSlope");
    distFields.ySlope = exoReader_->readCellToViewT<View_Scalar1>("ySlope");
    distFields.isFuel = exoReader_->readCellToViewT<View_Scalar1>("isFuel");
    distFields.w0     = exoReader_->readCellToViewT<View_Scalar1>("w0");
    distFields.sigma  = exoReader_->readCellToViewT<View_Scalar1>("sigma");
    distFields.delta  = exoReader_->readCellToViewT<View_Scalar1>("delta");
    distFields.Mx     = exoReader_->readCellToViewT<View_Scalar1>("Mx");
  
    exoReader_->close();

    // constant fuel-parameter values
    constFields.h    = 8000.0;
    constFields.rhoP = 32.0;
    constFields.Mf   = 0.01;
    constFields.St   = 0.0555;
    constFields.Se   = 0.010;

}

// ===============================
// absolute value
// ===============================
template<class EvalT>
KOKKOS_FUNCTION
EvalT
rothermal<EvalT>::absoluteValue(const EvalT & x)
{
    return sqrt(x * x + absoluteValueScale);
}

// ===============================
// heaviside
// ===============================
template<class EvalT>
KOKKOS_FUNCTION
EvalT
rothermal<EvalT>::heaviside(const EvalT & x)
{
    return 0.5 * (1.0 + tanh(x));
}

// ===============================
// get mesh data
// ===============================
template<class EvalT>
void rothermal<EvalT>::getMeshData()
{
    meshData.Lx   = exoReader_->domainX;    // domain width in x-direction
    meshData.Ly   = exoReader_->domainY;    // domain width in y-direction
    meshData.Diag = exoReader_->domainDiag; // domain diagonal
    meshData.x0   = exoReader_->x0;         // domain origin in x-direction
    meshData.y0   = exoReader_->y0;         // domain origin in y-direction
    meshData.dx   = exoReader_->dx;         // element width in x-direction
    meshData.dy   = exoReader_->dy;         // element width in y-direction
    meshData.nx   = exoReader_->nx;         // number of elements in x-direction
    meshData.ny   = exoReader_->ny;         // number of elements in y-direction
}

// ===============================
// get element ids: this is used to map local element ids to global element ids
// ===============================
template<class EvalT>
typename rothermal<EvalT>::View_Int1
rothermal<EvalT>::getElementIds(Teuchos::RCP<Workset<EvalT> > & wkset)
{

  const int numElem = wkset->numElem;
  View_Int1 wksetIds("ws_eid", numElem);

  auto x = wkset->getScalarField("x");
  auto y = wkset->getScalarField("y");

  Kokkos::parallel_for("map_ws_to_exo", RangePolicy<AssemblyExec>(0, numElem),
    MRHYDE_LAMBDA(int e){
      const ScalarT X = x(e,0);
      const ScalarT Y = y(e,0);
      int i = (int)floor((X - meshData.x0)/meshData.dx);  if (i<0) i=0; if (i>=meshData.nx) i=meshData.nx-1;
      int j = (int)floor((Y - meshData.y0)/meshData.dy);  if (j<0) j=0; if (j>=meshData.ny) j=meshData.ny-1;
      wksetIds(e) = j*meshData.nx + i;
    });

  return wksetIds;

}

// ===============================
// evalNodalR: this is used to interpolate R from nodal data
// ===============================
template<class EvalT>
void rothermal<EvalT>::readNodalR()
{
    // resolve mesh path
    string meshPath = "mesh.exo";
    ExoReader rdr(meshPath);
    rdr.open();
    rdr.readNodalField("Rnodal", Rnodal_host_, 1);
    rdr.close();
}


template<class EvalT>
typename rothermal<EvalT>::View_EvalT2
rothermal<EvalT>::evalNodalR(Teuchos::RCP<Workset<EvalT> > & wkset)
{

    // IP coordinates
    const int numElem = wkset->numElem;
    auto xip          = wkset->getScalarField("x");
    auto yip          = wkset->getScalarField("y");
    const int numQuad = static_cast<int>(xip.extent(1));

    // mesh geometry from rothermal_ (uniform rect grid)
    const double x0 = meshData.x0;
    const double y0 = meshData.y0;
    const double dx = meshData.dx;
    const double dy = meshData.dy;
    const int    nx = meshData.nx;
    const int    ny = meshData.ny;


    // copy nodal array to device for use in kernel
    const size_t nNodes = Rnodal_host_.size();
    Kokkos::View<double*, AssemblyDevice> Rnodal_d("Rnodal_d", nNodes);
    {
      auto Rnodal_h = Kokkos::create_mirror_view(Rnodal_d);
      for (size_t i = 0; i < nNodes; ++i) Rnodal_h(i) = Rnodal_host_[i];
      Kokkos::deep_copy(Rnodal_d, Rnodal_h);
    }

    View_EvalT2 R_ip("R_ip", numElem, numQuad);
    
    Kokkos::parallel_for("R_from_Rnodal_same_mesh",
        RangePolicy<AssemblyExec>(0, numElem),
        MRHYDE_LAMBDA(const int e) {
          for (int q = 0; q < numQuad; ++q) {


            const double x = xip(e,q);
            const double y = yip(e,q);

            // cell indices and local coords
            const double sx = (x - x0) / dx;
            const double sy = (y - y0) / dy;

            int i0 = static_cast<int>(floor(sx));
            int j0 = static_cast<int>(floor(sy));

            // clamp so (i0+1)<=nx and (j0+1)<=ny
            if (i0 < 0) i0 = 0; if (i0 > nx-1) i0 = nx-1;
            if (j0 < 0) j0 = 0; if (j0 > ny-1) j0 = ny-1;

            const double tx = sx - i0;
            const double ty = sy - j0;

            const int nnx = nx + 1;
            const size_t n00 = static_cast<size_t>(j0    ) * nnx + static_cast<size_t>(i0    );
            const size_t n10 = static_cast<size_t>(j0    ) * nnx + static_cast<size_t>(i0 + 1);
            const size_t n01 = static_cast<size_t>(j0 + 1) * nnx + static_cast<size_t>(i0    );
            const size_t n11 = static_cast<size_t>(j0 + 1) * nnx + static_cast<size_t>(i0 + 1);

            const double r00 = Rnodal_d(n00);
            const double r10 = Rnodal_d(n10);
            const double r01 = Rnodal_d(n01);
            const double r11 = Rnodal_d(n11);

            const double w00 = (1.0 - tx) * (1.0 - ty);
            const double w10 =        tx  * (1.0 - ty);
            const double w01 = (1.0 - tx) *        ty ;
            const double w11 =        tx  *        ty ;

            R_ip(e,q) = w00*r00 + w10*r10 + w01*r01 + w11*r11;
          }
        });

    return R_ip;
}



// ===============================
// compute base equations: these are the orginal base equations from Rothermal (1972)
// ===============================
template<class EvalT>
KOKKOS_FUNCTION
typename rothermal<EvalT>::baseEquationsResult
rothermal<EvalT>::computeBaseEquations(
    const int elem,
    const View_Int1 & wksetIds
)
{
    // output struct
    baseEquationsResult out;
    
    // get parameter fields
    auto w0    = distFields.w0(wksetIds(elem));
    auto delta = distFields.delta(wksetIds(elem));
    auto sigma = distFields.sigma(wksetIds(elem));
    auto Mx    = distFields.Mx(wksetIds(elem));
    auto Mf    = constFields.Mf;
    auto St    = constFields.St;
    auto Se    = constFields.Se;
    auto MfMx  = Mf / Mx;
    
    // store equation results to struct
    out.gammaMax = pow(sigma, 1.5) / ( 495 + 0.0594 * pow(sigma, 1.5) );
    out.BetaOp   = 3.348 *  pow(sigma, -0.8189);
    out.A        = 1 / ( 4.774 * pow(sigma , 0.1) - 7.27 );
    out.etaM     = 1 - 2.59 * MfMx + 5.11 * pow(MfMx, 2) - 3.52 * pow(MfMx, 3);
    out.etaS     = 0.174 * pow(Se, -0.19);
    // out.C        = 7.47 * exp( -0.133 * pow(sigma, 0.55) );
    // out.B        = 0.02526 * pow(sigma, 0.54);
    // out.E        = 0.715 * exp( -0.000359 * sigma);
    out.wn       = w0 / (1 + St);
    out.rhoB     = w0 / delta;
    out.eps      = exp(-138 / sigma);
    out.Qig      = 250 + 1116 * Mf;

    return out;
}


// ===============================
// compute Rothermals: this returns ROS without wind/slope
// ===============================
template<class EvalT>
KOKKOS_FUNCTION
EvalT
rothermal<EvalT>::computeRothermals(
    const int elem,
    const View_Int1 & wksetIds
)
{

    // auto slope = distFields.slope(wksetIds(elem));

    // get parameter fields
    auto sigma = distFields.sigma(wksetIds(elem));
    auto h     = constFields.h;
    auto rhoP  = constFields.rhoP;

    // compute base equations
    baseEquationsResult baseResult = computeBaseEquations(elem, wksetIds);

    // retrieve base equation results
    EvalT rhoB     = baseResult.rhoB;
    // EvalT C        = baseResult.C;
    // EvalT B        = baseResult.B;
    // EvalT E        = baseResult.E;
    EvalT betaOp   = baseResult.BetaOp;
    EvalT A        = baseResult.A;
    EvalT gammaMax = baseResult.gammaMax;
    EvalT wn       = baseResult.wn;
    EvalT etaM     = baseResult.etaM;
    EvalT etaS     = baseResult.etaS;
    EvalT eps      = baseResult.eps;
    EvalT Qig      = baseResult.Qig;
    
    // compute some ROS equations
    EvalT beta      = rhoB / rhoP;
    EvalT xi        = exp( (0.792 + 0.681 * pow(sigma, 0.5)) * (beta + 0.1) ) / (192 + 0.2595 * sigma);
    // EvalT phiS      = 5.275 * pow(beta, -0.3) * pow(tan(slope * Kokkos::numbers::pi / 180.0), 2);
    EvalT betaRatio = beta / betaOp;
    // EvalT phiW      = C * pow(windComponentAbs, B) * pow(betaRatio, -E);
    EvalT gamma     = gammaMax * pow(betaRatio, A) * exp( A * (1 - betaRatio) );
    EvalT IR        = gamma * wn * h * etaM * etaS;
    EvalT R0        = (IR * xi) / (rhoB * eps * Qig);

    // return R0 * (1 + phiW + phiS);
    return R0;
}


// ===============================
// compute fields: final ROS as well as auxiliary fields
// ===============================
template<class EvalT>
typename rothermal<EvalT>::ResponseFields
rothermal<EvalT>::computeFields(
    const View_EvalT2& phi,
    const View_EvalT2& dphi_dx,
    const View_EvalT2& dphi_dy,
    const Vista<EvalT>& xvel,
    const Vista<EvalT>& yvel,
    Teuchos::RCP<Workset<EvalT> > & wkset,
    const bool & hasExternalR,
    Vista<EvalT>& R
)
{

    // get element ids
    View_Int1 wksetIds = getElementIds(wkset);

    // get workset size
    const int numElem = wkset->numElem;
    const int numQuad = phi.extent(1);

    // get mesh data and parameter fields
    auto diag   = meshData.Diag;
    auto slope  = distFields.slope;
    auto xSlope = distFields.xSlope;
    auto ySlope = distFields.ySlope;
    auto isFuel = distFields.isFuel;

    // initialize views
    // View_EvalT1 Rothermal_       ("Rothermal",       numElem);

    // initialize views based on workset size
    // View_EvalT2 fuelFraction_    ("fuelFraction",    numElem, numQuad);
    View_EvalT2 fuelCorrection_  ("fuelCorrection",  numElem, numQuad);
    View_EvalT2 ROS_             ("ROS",             numElem, numQuad);
    View_EvalT2 Rothermal_       ("Rothermal",       numElem, numQuad);
    View_EvalT2 windComponent_   ("windComponent",   numElem, numQuad);
    View_EvalT2 windCorrection_  ("windCorrection",  numElem, numQuad);
    View_EvalT2 slopeComponent_ ("slopeComponent",   numElem, numQuad);
    View_EvalT2 slopeCorrection_ ("slopeCorrection", numElem, numQuad);
    View_EvalT2 Hphi_            ("Hphi",            numElem, numQuad);


    // output struct
    ResponseFields fields;

    // loop over elements
    Kokkos::parallel_for("rothermal computeFields",
        Kokkos::RangePolicy<AssemblyExec>(0, numElem),
        MRHYDE_LAMBDA(const int elem)
    {
        
        // get element id
        int id = wksetIds(elem);

        // compute base rothermals without wind/slope
        // Rothermal_(elem) = computeRothermals(elem, wksetIds); // * old

        // * either use supplied or computed rothermals base ROS
        const EvalT R0_elem = hasExternalR ? R(elem, 0) : computeRothermals(elem, wksetIds);

        // loop over quadrature points
        for (size_type pt = 0; pt < (size_type)numQuad; ++pt)
        {

            Rothermal_(elem, pt) = R0_elem;

            // scaled SDF, indicator, and interface gate
            EvalT phiScaled = heavisideScale * 2 * phi(elem, pt) / diag;
            EvalT Hphi      = heaviside(phiScaled); 
            Hphi_(elem, pt) = Hphi;
            EvalT phiGate   = exp(-(phiScaled*phiScaled));

            // gradient-based normal
            EvalT gradNorm2   = dphi_dx(elem,pt)*dphi_dx(elem,pt)
                              + dphi_dy(elem,pt)*dphi_dy(elem,pt)
                              + zero_tol * zero_tol;
            EvalT invGradNorm = 1.0 / sqrt(gradNorm2);
            EvalT normalX     = dphi_dx(elem,pt) * invGradNorm;
            EvalT normalY     = dphi_dy(elem,pt) * invGradNorm;

            // fuel fraction
            EvalT fuelArg             = burnConstant * phiScaled;
            fuelCorrection_(elem, pt) = heaviside(fuelArg - Fthresh);

            // wind
            EvalT windComp            = xvel(elem,pt) * normalX + yvel(elem,pt) * normalY;
            windComponent_(elem, pt)  = windComp;
            EvalT windNorm            = sqrt(xvel(elem,pt) * xvel(elem,pt) + yvel(elem,pt) * yvel(elem,pt) + zero_tol * zero_tol);
            EvalT windArg             = phiGate * windConstant * (windComp / (1.0 + windNorm));
            windCorrection_(elem, pt) = exp(windArg);

            // slope
            EvalT sx                   = xSlope(id);
            EvalT sy                   = ySlope(id);
            EvalT slopeComp            = sx * normalX + sy * normalY;
            slopeComponent_(elem, pt)  = slopeComp;
            EvalT slopeNorm            = sqrt(sx * sx + sy * sy + zero_tol * zero_tol);
            EvalT slopeArg             = phiGate * slopeConstant * slopeComp / (1.0 + slopeNorm);
            slopeCorrection_(elem, pt) = exp(slopeArg);

            // * either use suppplied or computed rothermals base ROS
            const EvalT R_here = hasExternalR ? R(elem,pt) : R0_elem;

            // final ROS
            ROS_(elem, pt) = R_here
                           * fuelCorrection_(elem, pt)
                           * windCorrection_(elem, pt)
                           * slopeCorrection_(elem, pt)
                           * isFuel(id);
        }
    });

    // return fields
    // fields.fuelFraction    = fuelFraction_;
    fields.fuelCorrection  = fuelCorrection_;
    fields.ROS             = ROS_;
    fields.Rothermal       = Rothermal_;
    fields.windComponent   = windComponent_;
    fields.windCorrection  = windCorrection_;
    fields.slopeComponent  = slopeComponent_;
    fields.slopeCorrection = slopeCorrection_;
    fields.Hphi            = Hphi_;

    return fields;
}



template class MrHyDE::rothermal<ScalarT>;

#ifndef MrHyDE_NO_AD

template class MrHyDE::rothermal<AD>;

template class MrHyDE::rothermal<AD2>;
template class MrHyDE::rothermal<AD4>;
template class MrHyDE::rothermal<AD8>;
template class MrHyDE::rothermal<AD16>;
template class MrHyDE::rothermal<AD18>;
template class MrHyDE::rothermal<AD24>;
template class MrHyDE::rothermal<AD32>;
#endif
