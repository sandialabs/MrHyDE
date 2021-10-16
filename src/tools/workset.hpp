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

#ifndef WKSET_H
#define WKSET_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  // =================================================================
  // =================================================================
  
  class SolutionField {
  public:
    
    SolutionField() {};
    
    ~SolutionField() {};
    
    SolutionField(const string & expression_,
                  const string & vartype_,
                  const int & varindex_,
                  const string & basistype_,
                  const int & basis_index_,
                  const string & derivtype_,
                  const int & component_,
                  const int & dim0, // typically zero for first touch allocations
                  const int & dim1,
                  const bool & onSide_,
                  const bool & isPoint_) {
      
      expression = expression_;
      variable_type = vartype_; // solution, aux, param
      variable_index = varindex_;
      basis_type = basistype_; // HGRAD, HVOL, HDIV, HCURL, HFACE
      basis_index = basis_index_;
      derivative_type = derivtype_; // grad, curl, div, time
      component = component_; // x, y, z
      data = View_AD2("solution field for " + expression, dim0, dim1);
      isOnSide = onSide_;
      isPoint = isPoint_;
      isUpdated = false;
      
    }
    
    string expression, variable_type, basis_type, derivative_type;
    int variable_index, basis_index, component;
    bool isUpdated, isOnSide, isPoint;
    View_AD2 data;
    
  };
  
  class workset {
  public:
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    ////////////////////////////////////////////////////////////////////////////////////
    
    workset() {};
    
    workset(const vector<int> & cellinfo, const bool & isTransient_,
            const vector<string> & basis_types_,
            const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
            const topo_RCP & topo, Kokkos::View<string**,HostDevice> & var_bcs_);
            
    ////////////////////////////////////////////////////////////////////////////////////
    // Public functions
    ////////////////////////////////////////////////////////////////////////////////////
    
    void createSolns();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Reset solution to zero
    ////////////////////////////////////////////////////////////////////////////////////
    
    void reset();
    
    void resetResidual();
    
    void resetSolutionFields();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the seeded solutions for general transient problems
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnTransientSeeded(View_Sc3 u,
                                    View_Sc4 u_prev,
                                    View_Sc4 u_stage,
                                    const int & seedwhat,
                                    const int & index=0);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the seeded solutions for steady-state problems
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnSteadySeeded(View_Sc3 u, const int & seedwhat);
    
    void computeParamSteadySeeded(View_Sc3 params, const int & seedwhat);
    
    void computeAuxSolnSteadySeeded(View_Sc3 aux, const int & seedwhat);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at general set of points
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSoln(const int & type, const bool & onside);
    
    void evaluateSolutionField(const int & fieldnum);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at the volumetric ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnVolIP();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the discretized parameters at the volumetric ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeParamVolIP();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at the side ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeSolnSideIP();
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the discretized parameters at the side ip
    ////////////////////////////////////////////////////////////////////////////////////
    
    void computeParamSideIP();
    
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
    
    void checkDataAllocation(const size_t & ind);
    
    void checkDataScAllocation(const size_t & ind);
    
    View_AD2 findData(const string & label);
    
    View_AD2 getData(const string & label);
    
    View_Sc2 getDataSc(const string & label);
    
    size_t getDataScIndex(const string & label);
    
#ifndef MrHyDE_NO_AD
    void get(const string & label, View_AD2 & dataout);
#endif
    
    void get(const string & label, View_Sc2 & dataout);
    
    View_Sc4 getBasis(const string & var);
    
    View_Sc4 getBasis(const int & varindex);
    
    View_Sc4 getBasisGrad(const string & var);
    
    View_Sc4 getBasisGrad(const int & varindex);
    
    View_Sc4 getBasisCurl(const string & var);
    
    View_Sc4 getBasisCurl(const int & varindex);
    
    View_Sc3 getBasisDiv(const string & var);
    
    View_Sc3 getBasisDiv(const int & varindex);
    
    View_Sc4 getBasisSide(const string & var);
    
    View_Sc4 getBasisSide(const int & varindex);
    
    View_Sc4 getBasisGradSide(const string & var);
    
    View_Sc4 getBasisGradSide(const int & varindex);
    
    View_Sc4 getBasisCurlSide(const string & var);
    
    View_Sc4 getBasisCurlSide(const int & varindex);
    
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
    
    void addData(const string & label, const int & dim0, const int & dim1);
    
    void addDataSc(const string & label, const int & dim0, const int & dim1);

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
    // Functions to set the data
    //////////////////////////////////////////////////////////////
    
#ifndef MrHyDE_NO_AD
    void setData(const string & label, View_AD2 newdata);
#endif
    
    void setDataSc(const string & label, View_Sc2 newdata);
    
    void reorderData();
    
    void printMetaData();
    
    void setIP(vector<View_Sc2> & newip, const string & pfix = "");
    
    void setNormals(vector<View_Sc2> & newnormals);
    
    void setTangents(vector<View_Sc2> & newtangents);
    
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
    
    void setAux(View_AD4 newsol, const string & pfix = "");
    
    void setAuxGrad(View_AD4 newsolgrad, const string & pfix = "");
    
    void setAuxDiv(View_AD3 newsoldiv, const string & pfix = "");
    
    void setAuxCurl(View_AD4 newsolcurl, const string & pfix = "");
    
    void setAuxPoint(View_AD2 newsol);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Public data
    ////////////////////////////////////////////////////////////////////////////////////
    
    // Should be the only view stored on Host
    // Used by physics modules to determine the proper contribution to the boundary residual
    Kokkos::View<string**,HostDevice> var_bcs, aux_var_bcs;
    
    Kokkos::View<int**,AssemblyDevice> offsets, paramoffsets, aux_offsets;
    vector<string> varlist, aux_varlist, param_varlist;
    Kokkos::View<ScalarT**,AssemblyDevice> butcher_A, aux_butcher_A;
    Kokkos::View<ScalarT*,AssemblyDevice> butcher_b, butcher_c, BDF_wts, aux_butcher_b, aux_butcher_c, aux_BDF_wts;
    
    vector<int> usebasis, paramusebasis, auxusebasis;
    vector<int> vars_HGRAD, vars_HVOL, vars_HDIV, vars_HCURL, vars_HFACE;
    vector<int> paramvars_HGRAD, paramvars_HVOL, paramvars_HDIV, paramvars_HCURL, paramvars_HFACE;
    vector<int> auxvars_HGRAD, auxvars_HVOL, auxvars_HDIV, auxvars_HCURL, auxvars_HFACE;
    
    vector<string> varlist_HGRAD, varlist_HVOL, varlist_HDIV, varlist_HCURL, varlist_HFACE;
    vector<string> paramvarlist_HGRAD, paramvarlist_HVOL, paramvarlist_HDIV, paramvarlist_HCURL, paramvarlist_HFACE;
    vector<string> auxvarlist_HGRAD, auxvarlist_HVOL, auxvarlist_HDIV, auxvarlist_HCURL, auxvarlist_HFACE;
    
    bool isAdjoint, onlyTransient, isTransient;
    bool isInitialized, usebcs, onDemand;
    topo_RCP celltopo;
    size_t numsides, numip, numsideip, numVars, numParams, numAux, maxRes, maxTeamSize;
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
    
    // Views that use ContLayout for hierarchical parallelism
    vector<SolutionField> fields;
    
    vector<View_AD2> data;
    vector<string> data_labels;
    vector<int> data_usage;
    
    vector<View_Sc2> data_Sc;
    vector<string> data_Sc_labels;
    vector<int> data_Sc_usage;
    
    View_Sc1 h;
    View_Sc2 wts, wts_side;
    
    vector<View_AD2> uvals, u_dotvals, pvals, auxvals, aux_dotvals;
    
    vector<View_Sc4> basis, basis_grad, basis_curl, basis_side, basis_grad_side, basis_curl_side;
    vector<View_Sc3> basis_div;
        
    View_AD3 flux;
    View_AD2 res, adjrhs;

    // Storage for integrated quantities TODO where to initialize? OK to do in the postprocess setup?
    
    View_Sc1 integrated_quantities;
        
    int sidetype;
    Kokkos::View<int****,AssemblyDevice> sideinfo;
    string sidename, var;
    int currentside;
    
    bool have_rotation, have_rotation_phi;
    View_Sc3 rotation;
    View_Sc2 rotation_phi, extra_data;
    
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
