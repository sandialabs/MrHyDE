/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_ROTHERMAL_H
#define MRHYDE_ROTHERMAL_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"

//ExoReader

#include <exodusII.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>

namespace MrHyDE {

  class ExoReader {

  typedef Kokkos::View<ScalarT*, AssemblyDevice> View_Scalar1;

  public:

    // ========================================================================================
    // public domain geometry data
    // ========================================================================================
    
    // dimensions of the domain
    double domainX    = 0.0;  // domain width in x-direction
    double domainY    = 0.0;  // domain width in y-direction
    double domainDiag = 0.0;  // domain diagonal length

    // origin, spacing, and number of cells in the domain
    double x0 = 0.0;  // domain origin in x-direction
    double y0 = 0.0;  // domain origin in y-direction
    double dx = 1.0;  // element width in x-direction
    double dy = 1.0;  // element width in y-direction
    int    nx = 1;    // number of elements in x-direction
    int    ny = 1;    // number of elements in y-direction

    // ========================================================================================
    // constructor and destructor
    // ========================================================================================
    explicit ExoReader(const std::string& path);  // constructor: takes path to exodus file
    ~ExoReader();  // destructor: automatically closes file if open

    // ========================================================================================
    // file operations
    // ========================================================================================
    void open();  // open the exodus file for reading
    void close();  // close the exodus file
    bool isOpen() const { return exoid_ >= 0; }  // check if file is currently open

    // ========================================================================================
    // metadata access
    // ========================================================================================
    ex_entity_id blockId();  // get the element block ID
    int64_t      numElems();  // get the number of elements in the block

    // ========================================================================================
    // geometry computation
    // ========================================================================================
    void computeGridAndDomain();  // compute domain geometry from mesh coordinates

    // ========================================================================================
    // variable access
    // ========================================================================================
    const std::unordered_map<std::string,int>& elemVarIndex();  // get variable name to index mapping

    // read a variable by index
    std::vector<double> readElemVarByIndex(int var_idx, int time_step = 1);

    // read a variable by name
    std::vector<double> readElemVarByName (const std::string& name, int time_step = 1);

    // templated API: caller controls the memory space by ViewType
    template <class ViewType>
    ViewType readCellToViewT(const std::string& name, int time_step = 1);

  private:
    // ========================================================================================
    // private member variables
    // ========================================================================================
    
    // path and exodus handle
    std::string file_;  // path to the exodus file
    int  exoid_ = -1;   // exodus file handle (negative means closed)

    // cached metadata
    bool grid_ok_          = false;  // flag indicating if grid geometry has been computed
    bool var_map_ok_       = false;  // flag indicating if variable name map has been built
    ex_entity_id block_id_ = 0;      // cached element block ID
    int64_t num_elems_     = -1;     // cached number of elements
    int num_elem_vars_     = 0;      // cached number of element variables
    std::unordered_map<std::string,int> name_to_varidx_;  // variable name to index mapping

    // derived helpers
    double inv_dx_ = 1.0;  // inverse of x-direction spacing (1/dx)
    double inv_dy_ = 1.0;  // inverse of y-direction spacing (1/dy)

    // ========================================================================================
    // internal helper methods
    // ========================================================================================
    void ensureOpen_() const;  // ensure file is open, throw if not
    void queryInit_(int& nd, int& nn, int& ne, int& nb, int& nns, int& nss);  // query exodus initialization data
    void requireSingleBlock_();  // ensure exactly one element block exists
    void buildElemVarNameIndex_();  // build variable name to index mapping
    static std::string trimPad_(const char* s);  // trim padding from exodus string
    static std::vector<double> unique_sorted_(const std::vector<double>& a, double eps);  // get unique sorted values with tolerance
};


// ========================================================================================
// templated method implementation: read element variable to Kokkos view
// ========================================================================================
template <class ViewType>
ViewType ExoReader::readCellToViewT(const std::string& name, int time_step) {
  ensureOpen_();
  requireSingleBlock_();

  if (!var_map_ok_) buildElemVarNameIndex_();
  auto it = name_to_varidx_.find(name);
  if (it == name_to_varidx_.end())
    throw std::runtime_error("ExoReader::readCellToView: element variable '" + name + "' not found");

  const int ex_var_idx = it->second;           // Exodus is 1-based
  const int64_t ne = num_elems_;

  static_assert(ViewType::rank == 1, "readCellToViewT expects a 1D View");
  using value_type = typename ViewType::non_const_value_type;

  // allocate return view in the caller-selected memory space
  ViewType v(Kokkos::view_alloc(Kokkos::WithoutInitializing, "exo_" + name), ne);

  // read Exodus into a host buffer (double), then copy/convert to v
  std::vector<double> buf(ne);
  if (ex_get_var(exoid_, time_step, EX_ELEM_BLOCK, ex_var_idx, block_id_, ne, buf.data()) != 0)
    throw std::runtime_error("ExoReader::readCellToView: ex_get_var failed for '" + name + "'");

  auto h = Kokkos::create_mirror_view(v);
  for (int64_t i = 0; i < ne; ++i) h(i) = static_cast<value_type>(buf[i]);
  Kokkos::deep_copy(v, h);

  return v;
}
  
  template<class EvalT>
  class rothermal {

    // ===============================
    // Kokkos view types
    // ===============================
    typedef Kokkos::View<int*, AssemblyDevice> View_Int1;
    typedef Kokkos::View<ScalarT*, AssemblyDevice> View_Scalar1;
    typedef Kokkos::View<EvalT*,ContLayout,AssemblyDevice> View_EvalT1; 
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;    
    
  public:
    
    // ===============================
    // constructor and destructor
    // ===============================
    rothermal() {};
    ~rothermal() {};
    
    // parameter list
    rothermal(Teuchos::ParameterList & rothermalSettings);

    // ===============================
    // structs
    // ===============================

    // mesh data
    struct MeshData {
      double Lx;   // domain width in x-direction
      double Ly;   // domain width in y-direction
      double Diag; // domain diagonal
      double x0;   // domain origin in x-direction
      double y0;   // domain origin in y-direction
      double dx;   // element width in x-direction
      double dy;   // element width in y-direction
      int nx;      // number of elements in x-direction
      int ny;      // number of elements in y-direction
    };

    // distributed parameter fields
    struct DistributedFields {
      View_Scalar1 slope;  // slope of the terrain
      View_Scalar1 xSlope; // slope of the terrain in x-direction
      View_Scalar1 ySlope; // slope of the terrain in y-direction
      View_Scalar1 isFuel; // indicator function for fuel
      View_Scalar1 w0; 
      View_Scalar1 sigma; 
      View_Scalar1 delta;
      View_Scalar1 Mx;

    };

    // constant parameter fields
    struct ConstantFields {
      EvalT h;
      EvalT rhoP;
      EvalT Mf;
      EvalT St;
      EvalT Se;
    };

    // fields related to ROS
    struct ResponseFields {
      View_EvalT2 ROS;             // rate of spread
      View_EvalT2 fuelFraction;    // fraction of fuel not burnt
      // View_EvalT2 Rothermal;
      View_EvalT2 windComponent;   // Intermediate/diganostic field
      View_EvalT2 windCorrection;  // used to account for wind
      View_EvalT2 slopeComponent;  // Intermediate/diganostic field
      View_EvalT2 slopeCorrection; // used to account for slope
      View_EvalT2 Hphi;            // indicator function for SDF phi
    };

    // ===============================
    // struct instances
    // ===============================
    DistributedFields distFields;
    ConstantFields constFields;
    MeshData meshData;

    // ===============================
    // get element ids: this is used to map local element ids to global element ids
    // ===============================
    View_Int1 getElementIds(Teuchos::RCP<Workset<EvalT> > & wkset);

    // ===============================
    // compute fields: final ROS as well as auxiliary fields
    // ===============================
    ResponseFields computeFields(
      const View_EvalT2 & phi,
      const View_EvalT2 & dphi_dx,
      const View_EvalT2 & dphi_dy,
      const Vista<EvalT> & xvel,
      const Vista<EvalT> & yvel,
      Teuchos::RCP<Workset<EvalT> > & wkset
    );


  private:

    // parameter list
    Teuchos::ParameterList rothermalSettings_;

    // exodus reader
    std::unique_ptr<ExoReader> exoReader_;

    // ScalarT burnRate           = 1.0;
    // ScalarT burnCons           = 0.8514;

    // constants for fuel, wind, and slope magnification/damping
    ScalarT burnConstant  = 1.0;
    ScalarT windConstant  = 1.0;
    ScalarT slopeConstant = 1.0;


    // threshold for fuel fraction
    ScalarT Fthresh            = 5e-2;

    // scaling parameters
    // ScalarT slopeFactor        = 1.0;
    ScalarT absoluteValueScale = 1.0;
    ScalarT heavisideScale     = 1.0;

    // zero tolerance
    ScalarT zero_tol           = 1e-8;

    // struct for base equations results
    struct baseEquationsResult {
      EvalT gammaMax;
      EvalT BetaOp;
      EvalT A;
      EvalT etaM;
      EvalT etaS;
      // EvalT C;
      // EvalT B;
      // EvalT E;
      EvalT wn;
      EvalT rhoB;
      EvalT eps;
      EvalT Qig;

    };


    // ===============================
    // get mesh data: this is used to get the domain width, height, and origin
    // ===============================
    void getMeshData();

    // ===============================
    // initialize scalars: these are constants and hyperparameters
    // ===============================
    void initScalars();

    // ===============================
    // initialize parameter fields: these are the parameter fields that are distributed over the mesh
    // ===============================
    void initParameterFields();

    // ===============================
    // absolute value: this is used to compute the absolute value of a scalar
    // ===============================
    KOKKOS_FUNCTION EvalT absoluteValue(const EvalT & x);

    // ===============================
    // heaviside: this is used to compute the heaviside function of a scalar
    // ===============================
    KOKKOS_FUNCTION EvalT heaviside(const EvalT & x);

    // ===============================
    // compute base equations: these are the orginal base equations from Rothermal (1972)
    // ===============================
    KOKKOS_FUNCTION baseEquationsResult computeBaseEquations(
      const int elem,
      /* const size_type pt, */
      const View_Int1 & wksetIds
    );

    // ===============================
    // compute Rothermals: this returns ROS without wind/slope
    // ===============================
    KOKKOS_FUNCTION EvalT computeRothermals(
      const int elem,
      const View_Int1 & wksetIds
    );

  };

  
    
}

#endif 
