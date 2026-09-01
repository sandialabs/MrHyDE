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
#include "ExoReader.hpp"

namespace MrHyDE {

  
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

    // check if nodal R is supplied
    // bool haveNodalR = false;

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
      // View_EvalT2 fuelFraction;    // fraction of fuel not burnt
      View_EvalT2 fuelCorrection;
      View_EvalT2 Rothermal;       // rate of spread with wind and slope
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
    // used for reading nodal R
    // ===============================
    vector<double> Rnodal_host_;                   // filled by ExoReader
    Kokkos::View<double*, AssemblyDevice> Rnodal_dev_;  // device copy of nodal R

    // ===============================
    // read nodal R: this is used to read the nodal R data from the exodus file
    // ===============================
    void readNodalR();

    // ===============================
    // get element ids: this is used to map local element ids to global element ids
    // ===============================
    View_Int1 getElementIds(Teuchos::RCP<Workset<EvalT> > & wkset);

    // ===============================
    // interploates R from nodal data
    // ===============================
    View_EvalT2 evalNodalR(Teuchos::RCP<Workset<EvalT> > & wkset);

    // ===============================
    // compute fields: final ROS as well as auxiliary fields
    // ===============================
    ResponseFields computeFields(
      const View_EvalT2 & phi,
      const View_EvalT2 & dphi_dx,
      const View_EvalT2 & dphi_dy,
      const Vista<EvalT> & xvel,
      const Vista<EvalT> & yvel,
      Teuchos::RCP<Workset<EvalT> > & wkset,
      const bool & hasExternalR,
      Vista<EvalT>& R
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
