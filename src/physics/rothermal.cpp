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

  std::cout << "Leaving rothermalt<>::constructor()" << std::endl;  
}


// ===============================
// initialize scalars
// ===============================
template<class EvalT>
void rothermal<EvalT>::initScalars()
{
  // fire dies when fuel fraction is below this threshold
  Fthresh = rothermalSettings_.get("Fthresh",            Fthresh);

  // smoothing parameters
  absoluteValueScale = rothermalSettings_.get("absoluteValueScale", absoluteValueScale);
  heavisideScale     = rothermalSettings_.get("heavisideScale",     heavisideScale);

  // constants for fuel, wind, and slope magnification/damping
  burnConstant       = rothermalSettings_.get("burnConstant",       burnConstant);
  windConstant       = rothermalSettings_.get("windConstant",       windConstant);
  slopeConstant      = rothermalSettings_.get("slopeConstant",      slopeConstant);
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
    Teuchos::RCP<Workset<EvalT> > & wkset
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
    View_EvalT1 Rothermal_       ("Rothermal",       numElem);

    // initialize views based on workset size
    View_EvalT2 fuelFraction_    ("fuelFraction",    numElem, numQuad);
    View_EvalT2 ROS_             ("ROS",             numElem, numQuad);
    // View_EvalT2 Rothermal_       ("Rothermal",       numElem, numQuad);
    View_EvalT2 windComponent_   ("windComponent",   numElem, numQuad);
    View_EvalT2 windCorrection_  ("windCorrection",  numElem, numQuad);
    View_EvalT2 slopeComponent_ ("slopeComponent", numElem, numQuad);
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
        Rothermal_(elem) = computeRothermals(elem, wksetIds);

        // loop over quadrature points
        for (size_type pt = 0; pt < (size_type)numQuad; ++pt)
        {

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
            EvalT fuelArg           = burnConstant * phiScaled;
            EvalT ExpFuelArg        = exp(fuelArg);
            fuelFraction_(elem, pt) = Hphi + (1 - Hphi) * ExpFuelArg / (1 + ExpFuelArg);
            EvalT fuelCorrection    = heaviside(heavisideScale * (fuelFraction_(elem, pt) - Fthresh));

            // wind
            EvalT windComp            = xvel(elem,pt) * normalX + yvel(elem,pt) * normalY;
            windComponent_(elem, pt)  = windComp;
            EvalT windNorm            = sqrt(xvel(elem,pt) * xvel(elem,pt) + yvel(elem,pt) * yvel(elem,pt) + zero_tol*zero_tol);
            EvalT windArg             = phiGate * windConstant * (windComp / (1.0 + windNorm));
            windCorrection_(elem, pt) = exp(windArg);

            // slope
            EvalT sx                   = xSlope(id);
            EvalT sy                   = ySlope(id);
            EvalT slopeComp            = sx * normalX + sy * normalY;
            slopeComponent_(elem, pt)  = slopeComp;
            EvalT slopeNorm            = sqrt(sx * sx + sy * sy + zero_tol*zero_tol);
            EvalT slopeArg             = phiGate * slopeConstant * ((slope(id) / (slope(id) + 1.0)) * (slopeComp / slopeNorm));
            slopeCorrection_(elem, pt) = exp(slopeArg);

            // base rothermals from |u dot n|
            // EvalT windAbs        = sqrt(windComp*windComp + zero_tol * zero_tol);

            // final ROS
            ROS_(elem, pt) = Rothermal_(elem)
                           * fuelCorrection
                           * windCorrection_(elem, pt)
                           * slopeCorrection_(elem, pt)
                           * isFuel(id);
        }
    });

    // return fields
    fields.fuelFraction    = fuelFraction_;
    fields.ROS             = ROS_;
    // fields.Rothermal       = Rothermal_;
    fields.windComponent   = windComponent_;
    fields.windCorrection  = windCorrection_;
    fields.slopeComponent  = slopeComponent_;
    fields.slopeCorrection = slopeCorrection_;
    fields.Hphi            = Hphi_;

    return fields;
}

// ExoReader

// ========================================================================================
// constructor: initialize with file path
// ========================================================================================
ExoReader::ExoReader(const std::string& path) : file_(path) {}

// ========================================================================================
// destructor: automatically close file if open
// ========================================================================================
ExoReader::~ExoReader() {
  try { close(); } catch (...) {}
}

// ========================================================================================
// ensure file is open: throw exception if not
// ========================================================================================
void ExoReader::ensureOpen_() const {
  if (exoid_ < 0) throw std::runtime_error("ExoReader: file is not open");
}

// ========================================================================================
// open exodus file for reading
// ========================================================================================
void ExoReader::open() {
  if (isOpen()) return;
  int cpu_ws    = 8;   // use doubles
  int io_ws     = 0;   // read as stored
  float version = 0.0f;
  exoid_        = ex_open(file_.c_str(), EX_READ, &cpu_ws, &io_ws, &version);


  if (exoid_ < 0) throw std::runtime_error("ExoReader: ex_open failed for '" + file_ + "'");

  // reset caches
  grid_ok_       = false;
  var_map_ok_    = false;
  block_id_      = 0;
  num_elems_     = -1;
  num_elem_vars_ = 0;

  name_to_varidx_.clear();
}

// ========================================================================================
// close exodus file
// ========================================================================================
void ExoReader::close() {
  if (!isOpen()) return;
  (void)ex_close(exoid_); // keep state consistent even if close errors
  exoid_ = -1;
}

// ========================================================================================
// query exodus initialization data: get basic file dimensions
// ========================================================================================
void ExoReader::queryInit_(int& nd, int& nn, int& ne, int& nb, int& nns, int& nss) {
  ensureOpen_();
  char title[MAX_LINE_LENGTH+1];
  std::memset(title, 0, sizeof(title));
  nd=nn=ne=nb=nns=nss=0;
  if (ex_get_init(exoid_, title, &nd, &nn, &ne, &nb, &nns, &nss) != 0) {
    throw std::runtime_error("ExoReader: ex_get_init failed");
  }
}

// ========================================================================================
// require single element block: ensure exactly one block exists and cache its info
// ========================================================================================
void ExoReader::requireSingleBlock_() {
  ensureOpen_();
  if (block_id_ != 0 && num_elems_ >= 0) return; // already cached

  int nd=0, nn=0, ne=0, nb=0, nns=0, nss=0;
  queryInit_(nd, nn, ne, nb, nns, nss);
  if (nb != 1) {
    throw std::runtime_error("ExoReader: expected exactly 1 element block, found " + std::to_string(nb));
  }

  std::vector<ex_entity_id> ids(nb, 0);
  if (ex_get_ids(exoid_, EX_ELEM_BLOCK, ids.data()) != 0) {
    throw std::runtime_error("ExoReader: ex_get_ids(EX_ELEM_BLOCK) failed");
  }
  block_id_ = ids[0];

  // query element count of this block
  char elem_type[MAX_STR_LENGTH+1] = {0};
  int64_t nele=0, nnpe=0, nedge=0, nface=0, nattr=0;
  if (ex_get_block(exoid_, EX_ELEM_BLOCK, block_id_, elem_type, &nele, &nnpe, &nedge, &nface, &nattr) != 0) {
    throw std::runtime_error("ExoReader: ex_get_block failed");
  }
  num_elems_ = nele;
}

// ========================================================================================
// get element block ID
// ========================================================================================
ex_entity_id ExoReader::blockId() {
  requireSingleBlock_();
  return block_id_;
}

// ========================================================================================
// get number of elements in the block
// ========================================================================================
int64_t ExoReader::numElems() {
  requireSingleBlock_();
  return num_elems_;
}

// ========================================================================================
// compute grid and domain geometry from mesh coordinates
// ========================================================================================
void ExoReader::computeGridAndDomain() {
  ensureOpen_();

  int nd=0, nn=0, ne=0, nb=0, nns=0, nss=0;
  queryInit_(nd, nn, ne, nb, nns, nss);

  // coordinates
  std::vector<double> X(nn), Y(nn);
  if (ex_get_coord(exoid_, X.data(), Y.data(), nullptr) != 0) {
    throw std::runtime_error("ExoReader: ex_get_coord failed");
  }
  if (nn <= 0) throw std::runtime_error("ExoReader: no nodes in file");

  // unique sorted X/Y with tolerance
  const double eps = 1e-10;
  auto Xu = unique_sorted_(X, eps);
  auto Yu = unique_sorted_(Y, eps);
  if (Xu.size() < 2 || Yu.size() < 2) {
    throw std::runtime_error("ExoReader: degenerate grid (need >= 2 unique coords per axis)");
  }

  // compute spacings and sanity-check uniform rectilinear grid
  std::vector<double> dX, dY;
  for (size_t i = 1; i < Xu.size(); ++i) dX.push_back(Xu[i] - Xu[i-1]);
  for (size_t j = 1; j < Yu.size(); ++j) dY.push_back(Yu[j] - Yu[j-1]);

  double dx_mean = (dX.empty()? 1.0 : (Xu.back() - Xu.front()) / (double)(Xu.size()-1));
  double dy_mean = (dY.empty()? 1.0 : (Yu.back() - Yu.front()) / (double)(Yu.size()-1));

  // fill public geometry
  x0         = Xu.front();
  y0         = Yu.front();
  dx         = dx_mean;
  dy         = dy_mean;
  nx         = static_cast<int>(Xu.size()) - 1;
  ny         = static_cast<int>(Yu.size()) - 1;
  domainX    = Xu.back() - Xu.front();
  domainY    = Yu.back() - Yu.front();
  domainDiag = std::sqrt(domainX*domainX + domainY*domainY);
  inv_dx_    = 1.0 / dx;
  inv_dy_    = 1.0 / dy;

  // consistency with element count
  requireSingleBlock_();
  if (static_cast<int64_t>(nx) * static_cast<int64_t>(ny) != num_elems_) {
    throw std::runtime_error("ExoReader: nx*ny != number of elements in block");
  }

  grid_ok_ = true;
}

// ========================================================================================
// get element variable name to index mapping
// ========================================================================================
const std::unordered_map<std::string,int>& ExoReader::elemVarIndex() {
  if (!var_map_ok_) buildElemVarNameIndex_();
  return name_to_varidx_;
}

// ========================================================================================
// build element variable name to index mapping
// ========================================================================================
void ExoReader::buildElemVarNameIndex_() {
  ensureOpen_();
  // number of element variables
  int nvar = 0;
  if (ex_get_variable_param(exoid_, EX_ELEM_BLOCK, &nvar) != 0) {
    throw std::runtime_error("ExoReader: ex_get_variable_param(EX_ELEM_BLOCK) failed");
  }
  num_elem_vars_ = nvar;

  name_to_varidx_.clear();
  if (nvar <= 0) { var_map_ok_ = true; return; }

  // exodus names are fixed-length char arrays
  std::vector<char*> names(nvar, nullptr);
  std::vector<std::vector<char>> storage(nvar, std::vector<char>(MAX_STR_LENGTH+1, 0));
  for (int i = 0; i < nvar; ++i) names[i] = storage[i].data();

  if (ex_get_variable_names(exoid_, EX_ELEM_BLOCK, nvar, names.data()) != 0) {
    throw std::runtime_error("ExoReader: ex_get_variable_names(EX_ELEM_BLOCK) failed");
  }

  for (int i = 0; i < nvar; ++i) {
    std::string nm = trimPad_(names[i]);
    if (!nm.empty()) name_to_varidx_[nm] = i+1; // 1-based index
  }
  var_map_ok_ = true;
}

// ========================================================================================
// read element variable by index
// ========================================================================================
std::vector<double> ExoReader::readElemVarByIndex(int var_idx, int time_step) {
  ensureOpen_();
  requireSingleBlock_();

  if (num_elems_ <= 0) return {};
  if (num_elem_vars_ == 0) buildElemVarNameIndex_(); // ensure we know count

  // accept 0- or 1-based input for robustness; do not document change
  int ex_var_idx = (var_idx >= 1) ? var_idx : (var_idx + 1);
  if (ex_var_idx <= 0 || ex_var_idx > num_elem_vars_) {
    throw std::out_of_range("ExoReader: element variable index out of range");
  }

  std::vector<double> vals(static_cast<size_t>(num_elems_), 0.0);
  if (ex_get_var(exoid_, time_step, EX_ELEM_BLOCK, ex_var_idx, block_id_, num_elems_, vals.data()) != 0) {
    throw std::runtime_error("ExoReader: ex_get_var failed for element variable index");
  }
  return vals;
}

// ========================================================================================
// read element variable by name
// ========================================================================================
std::vector<double> ExoReader::readElemVarByName(const std::string& name, int time_step) {
  if (!var_map_ok_) buildElemVarNameIndex_();
  auto it = name_to_varidx_.find(name);
  if (it == name_to_varidx_.end())
    throw std::runtime_error("ExoReader: element variable '" + name + "' not found");
  return readElemVarByIndex(it->second, time_step);
}


// ========================================================================================
// trim padding from exodus string: remove trailing spaces and nulls
// ========================================================================================
std::string ExoReader::trimPad_(const char* s) {
  if (!s) return std::string();
  std::string t(s);
  while (!t.empty() && (t.back() == ' ' || t.back() == '\0'))
    t.pop_back();
  return t;
}

// ========================================================================================
// get unique sorted values with tolerance: remove duplicates within epsilon
// ========================================================================================
std::vector<double> ExoReader::unique_sorted_(const std::vector<double>& a, double eps) {
  std::vector<double> b = a;
  std::sort(b.begin(), b.end());
  auto eq = [eps](double A, double B){ return std::abs(A-B) < eps; };
  b.erase(std::unique(b.begin(), b.end(), eq), b.end());
  return b;
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
