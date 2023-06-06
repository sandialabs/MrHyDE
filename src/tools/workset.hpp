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

#ifndef MRHYDE_WKSET_H
#define MRHYDE_WKSET_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "fields.hpp"
#include "compressedView.hpp"

namespace MrHyDE {
  
  // =================================================================
  // =================================================================
  
  class workset {
  public:
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    ////////////////////////////////////////////////////////////////////////////////////
    
    workset() {};
    
    ~workset() {};
    
    workset(const vector<int> & cellinfo,
            const vector<size_t> & numVars_, 
            const bool & isTransient_,
            const vector<string> & basis_types_,
            const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
            const topo_RCP & topo);
            
    ////////////////////////////////////////////////////////////////////////////////////
    // Public functions
    ////////////////////////////////////////////////////////////////////////////////////
    
    void createSolutionFields();
    
    void addSolutionFields(vector<string> & names, vector<string> & types, vector<int> & basis_indices);
    
    void addSolutionField(string & var, size_t & set_index,
                          size_t & var_index, string & basistype, string & soltype);

    void addScalarFields(vector<string> & fields);
      
    ////////////////////////////////////////////////////////////////////////////////////
    // Reset solution to zero
    ////////////////////////////////////////////////////////////////////////////////////
    
    void reset();
    
    void resetResidual();
    
    void resetSolutionFields();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the seeded solutions for general transient problems
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnTransientSeeded(const size_t & set,
                                    View_Sc3 u,
                                    View_Sc4 u_prev,
                                    View_Sc4 u_stage,
                                    const int & seedwhat,
                                    const int & index=0);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the seeded solutions for steady-state problems
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnSteadySeeded(const size_t & set, View_Sc3 u, const int & seedwhat);
    
    void computeParamSteadySeeded(View_Sc3 params, const int & seedwhat);
    
    void computeAuxSolnSteadySeeded(View_Sc3 aux, const int & seedwhat);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at general set of points
    ////////////////////////////////////////////////////////////////////////////////////
    
    void evaluateSolutionField(const int & fieldnum);

    void evaluateSideSolutionField(const int & fieldnum);
        
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at the side ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnSideIP(const int & side);
    
    //////////////////////////////////////////////////////////////
    // Add Aux
    //////////////////////////////////////////////////////////////
    
    void addAux(const vector<string> & auxlist, Kokkos::View<int**,AssemblyDevice> aoffs);
    
    //////////////////////////////////////////////////////////////
    // Get a pointer to vector of parameters
    //////////////////////////////////////////////////////////////
    
    vector<AD> getParam(const string & name, bool & found);
    
    //////////////////////////////////////////////////////////////
    // Get a subview associated with a vector of parameters
    //////////////////////////////////////////////////////////////

    Kokkos::View<AD*,Kokkos::LayoutStride,AssemblyDevice> getParameter(const string & name, bool & found);
      
    //////////////////////////////////////////////////////////////
    // Set the time
    //////////////////////////////////////////////////////////////
    
    void setTime(const ScalarT & newtime);
    
    //////////////////////////////////////////////////////////////
    // Set deltat
    //////////////////////////////////////////////////////////////
    
    void setDeltat(const ScalarT & newdt);
    
    //////////////////////////////////////////////////////////////
    // Set the stage index
    //////////////////////////////////////////////////////////////
    
    void setStage(const int & newstage);
    
    //////////////////////////////////////////////////////////////
    // Data extraction methods
    //////////////////////////////////////////////////////////////
    
    View_AD2 getResidual();
    
    View_Sc2 getWeights();
    
    View_Sc2 getSideWeights();
    
    //////////////////////////////////////////////////////////////
    // Interact with the scalar/solution fields
    //////////////////////////////////////////////////////////////
    
    void checkSolutionFieldAllocation(const size_t & ind);
    
    void checkScalarFieldAllocation(const size_t & ind);
    
    void printSolutionFields();
    
    void printScalarFields();
    
    View_AD2 getSolutionField(const string & label, const bool & evaluate = SOL_FIELD_EVAL,
                              const bool & markUpdated = false);
    
    View_Sc2 getScalarField(const string & label);
        
    //////////////////////////////////////////////////////////////
    // Interact with the basis functions
    //////////////////////////////////////////////////////////////
    
    CompressedViewSc4 getBasis(const string & var);
    
    CompressedViewSc4 getBasis(const int & varindex);
    
    CompressedViewSc4 getBasisGrad(const string & var);
    
    CompressedViewSc4 getBasisGrad(const int & varindex);
    
    CompressedViewSc4 getBasisCurl(const string & var);
    
    CompressedViewSc4 getBasisCurl(const int & varindex);
    
    CompressedViewSc3 getBasisDiv(const string & var);
    
    CompressedViewSc3 getBasisDiv(const int & varindex);
    
    CompressedViewSc4 getBasisSide(const string & var);
    
    CompressedViewSc4 getBasisSide(const int & varindex);
    
    CompressedViewSc4 getBasisGradSide(const string & var);
    
    CompressedViewSc4 getBasisGradSide(const int & varindex);
    
    CompressedViewSc4 getBasisCurlSide(const string & var);
    
    CompressedViewSc4 getBasisCurlSide(const int & varindex);
        
    //////////////////////////////////////////////////////////////
    // Get decompressed bases
    //////////////////////////////////////////////////////////////
    
    View_Sc4 getDecompressedBasis(const int & varindex);
    
    View_Sc4 getDecompressedBasisGrad(const int & varindex);
    
    View_Sc4 getDecompressedBasisCurl(const int & varindex);
    
    View_Sc3 getDecompressedBasisDiv(const int & varindex);
    
    View_Sc4 getDecompressedBasisSide(const int & varindex);
    
    View_Sc4 getDecompressedBasisGradSide(const int & varindex);
    
    View_Sc4 getDecompressedBasisCurlSide(const int & varindex);
    
    //////////////////////////////////////////////////////////////
    // Get the offsets or a subview of the offsets
    //////////////////////////////////////////////////////////////
    
    Kokkos::View<int**,AssemblyDevice> getOffsets();
    
    Kokkos::View<int*,Kokkos::LayoutStride,AssemblyDevice> getOffsets(const string & var);
    
    //////////////////////////////////////////////////////////////
    // Checks to determine if a string is a known variable
    //////////////////////////////////////////////////////////////
    
    bool findBasisIndex(const string & var, int & basisindex);

    bool isVar(const string & var, int & index);
    
    bool isParameter(const string & var, int & index);
    
    bool isAux(const string & var, int & index);
    
    //////////////////////////////////////////////////////////////
    // Functions to add data to storage
    //////////////////////////////////////////////////////////////
    
    /**
     * @brief Add storage for integrated quantities to the workset.
     * Should only be called during physics initialization.
     *
     * In the case of multiple physics defined on a block, the index returned may be greater than
     * zero so the individual modules will know which portion of the storage they own.
     *
     * @param[in] nRequest  The number of IQs required by the physics module (residual calculation).
     * @return The index where the first requested IQ will be placed (integer type).
     */

    int addIntegratedQuantities(const int & nRequested);
    
    //////////////////////////////////////////////////////////////
    // Functions to set scalar data
    //////////////////////////////////////////////////////////////
    
    void setScalarField(View_Sc2 newdata, const string & expression);
    
    //////////////////////////////////////////////////////////////
    // Function to carefully copy data
    //////////////////////////////////////////////////////////////
    
    template<class V1, class V2>
    void copyData(V1 view1, V2 view2);
    
    //////////////////////////////////////////////////////////////
    // Functions to set solution data (these are not all implemented and will be deprecated eventually)
    //////////////////////////////////////////////////////////////
    
    void setSolution(View_AD4 newsol, const string & pfix = "");
    
    void setSolutionGrad(View_AD4 newsolgrad, const string & pfix = "");
    
    void setSolutionDiv(View_AD3 newsoldiv, const string & pfix = "");
    
    void setSolutionCurl(View_AD4 newsolcurl, const string & pfix = "");
    
    void setSolutionPoint(View_AD2 newsol);
    
    void setSolutionGradPoint(View_AD2 newsol);
      
    void setParam(View_AD4 newsol, const string & pfix = "");
    
    void setParamGrad(View_AD4 newsolgrad, const string & pfix = "");
    
    void setParamDiv(View_AD3 newsoldiv, const string & pfix = "");
    
    void setParamCurl(View_AD4 newsolcurl, const string & pfix = "");
    
    void setParamPoint(View_AD2 newsol);
    
    void setParamGradPoint(View_AD2 newsol);

    string getParamBasisType(string & name);

    /**
     * @brief Set the value of the aux variables at the integration points
     * 
     * The new solution values must be ordered as the variables are in aux_varlist.
     * 
     * @param[in] newsol  The new solution values
     * @param[in] pfix  Optional suffix to the variable string
     */ 
    
    void setAux(View_AD4 newsol, const string & pfix = "");
    
    void setAuxGrad(View_AD4 newsolgrad, const string & pfix = "");
    
    void setAuxDiv(View_AD3 newsoldiv, const string & pfix = "");
    
    void setAuxCurl(View_AD4 newsolcurl, const string & pfix = "");
    
    void setAuxPoint(View_AD2 newsol);
    
    //////////////////////////////////////////////////////////////
    // Function to change the current physics set
    //////////////////////////////////////////////////////////////
    
    void updatePhysicsSet(const size_t & current_set_);
    
    void allocateRotations();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Public data
    ////////////////////////////////////////////////////////////////////////////////////
    
    // Should be the only view stored on Host
    // Used by physics modules to determine the proper contribution to the boundary residual
    
    bool isAdjoint, onlyTransient, isTransient;
    bool isInitialized, usebcs, isOnSide, isOnPoint;
    topo_RCP celltopo;
    size_t numsides, numip, numsideip, numParams, maxRes, maxTeamSize, current_set, numSets;
    int dimension, numElem, current_stage;
    size_type maxElem;
    
    vector<string> basis_types;
    vector<int> numbasis;
    vector<basis_RCP> basis_pointers;
    
    vector<Teuchos::RCP<vector<AD> > > params;
    Kokkos::View<AD**,AssemblyDevice> params_AD;
    vector<string> paramnames;
    
    ScalarT time, alpha, deltat;
    
    size_t block, localEID, globalEID;
    
    vector<SolutionField> soln_fields, side_soln_fields, point_soln_fields;
    vector<ScalarField> scalar_fields, side_scalar_fields, point_scalar_fields;
    
    View_Sc1 h;
    View_Sc2 wts, wts_side;
    vector<CompressedViewSc4> basis, basis_grad, basis_curl, basis_side, basis_grad_side, basis_curl_side;
    vector<CompressedViewSc3> basis_div;
    
    View_AD2 res, adjrhs;
    View_AD3 flux;
    Kokkos::View<int**,AssemblyDevice> offsets, paramoffsets, aux_offsets;
    vector<View_AD2> pvals;
    vector<string> param_varlist;
    vector<int> paramusebasis;
    vector<int> paramvars_HGRAD, paramvars_HVOL, paramvars_HDIV, paramvars_HCURL, paramvars_HFACE;
    vector<string> paramvarlist_HGRAD, paramvarlist_HVOL, paramvarlist_HDIV, paramvarlist_HCURL, paramvarlist_HFACE;
    
    size_t numAux;
    
    // Editing for multi-set
    vector<size_t> numVars;
    //vector<vector<View_AD2> > uvals, u_dotvals;
    vector<View_AD2> uvals, u_dotvals;
    vector<vector<size_t>> uvals_index; // [set][var]

    Kokkos::View<string**,HostDevice> var_bcs;
    vector<Kokkos::View<string**,HostDevice> > set_var_bcs;
    
    vector<Kokkos::View<int**,AssemblyDevice> > set_offsets;
    vector<vector<string> > set_varlist;
    vector<string> varlist, aux_varlist;
    
    Kokkos::View<ScalarT**,AssemblyDevice> butcher_A;
    Kokkos::View<ScalarT*,AssemblyDevice> butcher_b, butcher_c, BDF_wts;

    vector<Kokkos::View<ScalarT**,AssemblyDevice> > set_butcher_A; // [set]
    vector<Kokkos::View<ScalarT*,AssemblyDevice> > set_butcher_b, set_butcher_c, set_BDF_wts; // [set]
    
    vector<vector<int> > set_usebasis;
    vector<int> usebasis;
    vector<vector<int> > vars_HGRAD, vars_HVOL, vars_HDIV, vars_HCURL, vars_HFACE;
    vector<vector<string> > varlist_HGRAD, varlist_HVOL, varlist_HDIV, varlist_HCURL, varlist_HFACE;
    
    // Storage for integrated quantities
    
    View_Sc1 integrated_quantities;
        
    int sidetype;
    //Kokkos::View<int****,AssemblyDevice> sideinfo;
    string sidename, blockname;//, var;
    int currentside, time_step;
    
    bool have_rotation, have_rotation_phi;
    View_Sc3 rotation;
    View_Sc2 rotation_phi, extra_data;
    View_Sc3 multidata;

    //Kokkos::View<LO*,AssemblyDevice> basis_index;
    
    // Profile timers
    Teuchos::RCP<Teuchos::Time> worksetUpdateIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::update - integration data");
    Teuchos::RCP<Teuchos::Time> worksetUpdateBasisMMTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::update::multiplyMeasure");
    Teuchos::RCP<Teuchos::Time> worksetUpdateBasisHGTGTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::update::HGRADTransformGrad");
    Teuchos::RCP<Teuchos::Time> worksetAddSideTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::addSide");
    Teuchos::RCP<Teuchos::Time> worksetSideUpdateIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::updateSide - integration data");
    Teuchos::RCP<Teuchos::Time> worksetSideUpdateBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::updateSide - basis data");
    Teuchos::RCP<Teuchos::Time> worksetFaceUpdateIPTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::updateFace - integration data");
    Teuchos::RCP<Teuchos::Time> worksetFaceUpdateBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::updateFace - basis data");
    Teuchos::RCP<Teuchos::Time> worksetResetTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::reset*");
    Teuchos::RCP<Teuchos::Time> worksetComputeSolnVolTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::computeSolnVolIP - compute seeded sol at ip");
    Teuchos::RCP<Teuchos::Time> worksetComputeSolnSeededTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::computeSolnVolIP - allocate/compute seeded");
    Teuchos::RCP<Teuchos::Time> worksetComputeSolnSideTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::computeSolnSideIP");
    Teuchos::RCP<Teuchos::Time> worksetComputeParamVolTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::computeParamVolIP");
    Teuchos::RCP<Teuchos::Time> worksetComputeParamSideTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::computeParamSideIP");
    Teuchos::RCP<Teuchos::Time> worksetgetTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::get()");
    Teuchos::RCP<Teuchos::Time> worksetgetDataTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::getData");
    Teuchos::RCP<Teuchos::Time> worksetgetDataScTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::getDataSc");
    Teuchos::RCP<Teuchos::Time> worksetgetDataScIndexTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::getDataScIndex");
    Teuchos::RCP<Teuchos::Time> worksetgetBasisTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::getBasis*");
    Teuchos::RCP<Teuchos::Time> worksetcopyDataTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::copyData");
    
    Teuchos::RCP<Teuchos::Time> worksetDebugTimer0 = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::debug0");
    Teuchos::RCP<Teuchos::Time> worksetDebugTimer1 = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::debug1");
    Teuchos::RCP<Teuchos::Time> worksetDebugTimer2 = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::debug2");
    
  };
  
}

#endif
