/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
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
  template<class EvalT>
  class Workset {

  public:

    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    typedef Kokkos::View<EvalT***,ContLayout,AssemblyDevice> View_EvalT3;
    typedef Kokkos::View<EvalT****,ContLayout,AssemblyDevice> View_EvalT4;
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Constructors
    ////////////////////////////////////////////////////////////////////////////////////
    
    Workset() {};
    
    // ========================================================================================
    // ========================================================================================
  
    ~Workset() {};
  
    // ========================================================================================
    // ========================================================================================
  
    Workset(const vector<int> & cellinfo,
            const vector<size_t> & numVars_, 
            const bool & isTransient_,
            const vector<string> & basis_types_,
            const vector<basis_RCP> & basis_pointers_, const vector<basis_RCP> & param_basis_,
            const topo_RCP & topo);

    // ========================================================================================
    // ========================================================================================
  
    Workset(const size_t & block_, const size_t & num_sets ) {
      block = block_;
      numSets = num_sets;
      isInitialized = false;

      set_BDF_wts = vector<Kokkos::View<ScalarT*,AssemblyDevice> >(numSets);
      set_butcher_A = vector<Kokkos::View<ScalarT**,AssemblyDevice> >(numSets);
      set_butcher_b = vector<Kokkos::View<ScalarT*,AssemblyDevice> >(numSets);
      set_butcher_c = vector<Kokkos::View<ScalarT*,AssemblyDevice> >(numSets);
  
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // Public functions
    ////////////////////////////////////////////////////////////////////////////////////
    
    void createSolutionFields();
    
    // ========================================================================================
    // ========================================================================================
  
    void addSolutionFields(vector<string> & names, vector<string> & types, vector<int> & basis_indices);
  
    // ========================================================================================
    // ========================================================================================
  
    void addSolutionField(string & var, size_t & set_index,
                          size_t & var_index, string & basistype, string & soltype);

    // ========================================================================================
    // ========================================================================================
  
    void addScalarFields(vector<string> & fields);
      
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setSolutionFields(vector<int> & maxnumsteps, vector<int> & maxnumstages);

    ////////////////////////////////////////////////////////////////////////////////////
    // Reset solution to zero
    ////////////////////////////////////////////////////////////////////////////////////
    
    void reset();
    
    // ========================================================================================
    // ========================================================================================
  
    void resetResidual();
  
    // ========================================================================================
    // ========================================================================================
  
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
    
    // ========================================================================================
    // ========================================================================================
  
    void computeParamSteadySeeded(View_Sc3 params, const int & seedwhat);
  
    // ========================================================================================
    // ========================================================================================
  
    void computeAuxSolnSteadySeeded(View_Sc3 aux, const int & seedwhat);
    
    ////////////////////////////////////////////////////////////////////////////////////
    // Compute the solutions at general set of points
    ////////////////////////////////////////////////////////////////////////////////////
    
    void evaluateSolutionField(const int & fieldnum);

    // ========================================================================================
    // ========================================================================================
  
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
    // Get a subview associated with a vector of parameters
    //////////////////////////////////////////////////////////////

    Kokkos::View<EvalT*,Kokkos::LayoutStride,AssemblyDevice> getParameter(const string & name, bool & found);
      
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
    
    View_EvalT2 getResidual();
    
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc2> getWeights();
  
    // ========================================================================================
    // ========================================================================================
  
    View_Sc2 getSideWeights();
    
    //////////////////////////////////////////////////////////////
    // Interact with the scalar/solution fields
    //////////////////////////////////////////////////////////////
    
    void checkSolutionFieldAllocation(const size_t & ind);
    
    // ========================================================================================
    // ========================================================================================
  
    void checkScalarFieldAllocation(const size_t & ind);
  
    // ========================================================================================
    // ========================================================================================
  
    void printSolutionFields();
  
    // ========================================================================================
    // ========================================================================================
  
    void printScalarFields();
  
    // ========================================================================================
    // ========================================================================================
  
    View_EvalT2 getSolutionField(const string & label, const bool & evaluate = SOL_FIELD_EVAL,
                              const bool & markUpdated = false);
    // ========================================================================================
    // ========================================================================================
  
    View_Sc2 getScalarField(const string & label);
        
    //////////////////////////////////////////////////////////////
    // Interact with the basis functions
    //////////////////////////////////////////////////////////////
    
    CompressedView<View_Sc4> getBasis(const string & var);
    
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasis(const int & varindex);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisGrad(const string & var);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisGrad(const int & varindex);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisCurl(const string & var);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisCurl(const int & varindex);
    
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc3> getBasisDiv(const string & var);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc3> getBasisDiv(const int & varindex);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisSide(const string & var);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisSide(const int & varindex);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisGradSide(const string & var);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisGradSide(const int & varindex);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisCurlSide(const string & var);
  
    // ========================================================================================
    // ========================================================================================
  
    CompressedView<View_Sc4> getBasisCurlSide(const int & varindex);
    
    //////////////////////////////////////////////////////////////
    // Get the offsets or a subview of the offsets
    //////////////////////////////////////////////////////////////
    
    Kokkos::View<int**,AssemblyDevice> getOffsets();
    
    // ========================================================================================
    // ========================================================================================
  
    Kokkos::View<int*,Kokkos::LayoutStride,AssemblyDevice> getOffsets(const string & var);
    
    //////////////////////////////////////////////////////////////
    // Checks to determine if a string is a known variable
    //////////////////////////////////////////////////////////////
    
    bool findBasisIndex(const string & var, int & basisindex);

    // ========================================================================================
    // ========================================================================================
  
    bool isVar(const string & var, int & index);
  
    // ========================================================================================
    // ========================================================================================
  
    bool isParameter(const string & var, int & index);
    
    // ========================================================================================
    // ========================================================================================
  
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
    
    void setSolution(View_EvalT4 newsol, const string & pfix = "");
    
    // ========================================================================================
    // ========================================================================================
  
    void setSolutionGrad(View_EvalT4 newsolgrad, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setSolutionDiv(View_EvalT3 newsoldiv, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setSolutionCurl(View_EvalT4 newsolcurl, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setSolutionPoint(View_EvalT2 newsol);
    
    // ========================================================================================
    // ========================================================================================
  
    void setSolutionGradPoint(View_EvalT2 newsol);
  
    // ========================================================================================
    // ========================================================================================
  
    void setParam(View_EvalT4 newsol, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setParamGrad(View_EvalT4 newsolgrad, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setParamDiv(View_EvalT3 newsoldiv, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setParamCurl(View_EvalT4 newsolcurl, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setParamPoint(View_EvalT2 newsol);
  
    // ========================================================================================
    // ========================================================================================
  
    void setParamGradPoint(View_EvalT2 newsol);

    // ========================================================================================
    // ========================================================================================
  
    string getParamBasisType(string & name);

    /**
     * @brief Set the value of the aux variables at the integration points
     * 
     * The new solution values must be ordered as the variables are in aux_varlist.
     * 
     * @param[in] newsol  The new solution values
     * @param[in] pfix  Optional suffix to the variable string
     */ 
    
    void setAux(View_EvalT4 newsol, const string & pfix = "");
    
    // ========================================================================================
    // ========================================================================================
  
    void setAuxGrad(View_EvalT4 newsolgrad, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setAuxDiv(View_EvalT3 newsoldiv, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setAuxCurl(View_EvalT4 newsolcurl, const string & pfix = "");
  
    // ========================================================================================
    // ========================================================================================
  
    void setAuxPoint(View_EvalT2 newsol);
    
    //////////////////////////////////////////////////////////////
    // Function to change the current physics set
    //////////////////////////////////////////////////////////////
    
    void updatePhysicsSet(const size_t & current_set_);
  
    // ========================================================================================
    // ========================================================================================
  
    void allocateRotations();
  
    // ========================================================================================
    // ========================================================================================
  
    View_Sc1 getElementSize();
  
    // ========================================================================================
    // ========================================================================================
  
    View_Sc1 getSideElementSize();

    ////////////////////////////////////////////////////////////////////////////////////
    // Public data
    ////////////////////////////////////////////////////////////////////////////////////
    
    // Should be the only view stored on Host
    // Used by physics modules to determine the proper contribution to the boundary residual
    
    bool isAdjoint, onlyTransient, isTransient, only_scalar=false;
    bool isInitialized, usebcs, isOnSide, isOnPoint;
    topo_RCP celltopo;
    size_t numsides, numip, numsideip, numScalarParams, numDiscParams, maxRes, maxTeamSize, current_set, numSets;
    int dimension, numElem, current_stage;
    size_type maxElem;
    
    vector<string> basis_types;
    vector<int> numbasis;
    vector<basis_RCP> basis_pointers;
    
    Kokkos::View<EvalT**,AssemblyDevice> params_AD, params_dot_AD;
    vector<string> paramnames;
    
    ScalarT time, alpha, deltat;
    
    size_t block, localEID, globalEID;
    
    vector<SolutionField<EvalT> > soln_fields, side_soln_fields, point_soln_fields;
    vector<ScalarField> scalar_fields, side_scalar_fields, point_scalar_fields;
    
    // Actual DOFs for current group or boundary
    vector<View_Sc3> sol, phi;
    View_Sc3 param, param_dot, aux; // (elem,var,numdof)
    vector<View_Sc3> sol_avg, sol_alt;
    View_Sc3 param_avg, aux_avg; // (elem,var,dim)
    vector<View_Sc4> sol_prev, phi_prev, aux_prev, sol_stage, phi_stage, aux_stage; // (elem,var,numdof,step or stage)
    
    //View_Sc1 h;
    View_Sc2 wts_side;
    CompressedView<View_Sc2> wts;
    vector<CompressedView<View_Sc4>> basis, basis_grad, basis_curl, basis_side, basis_grad_side, basis_curl_side;
    vector<CompressedView<View_Sc3>> basis_div;
    
    View_EvalT2 res, adjrhs;
    View_EvalT3 flux;
    Kokkos::View<int**,AssemblyDevice> offsets, paramoffsets, aux_offsets;
    vector<View_EvalT2> pvals;
    vector<string> param_varlist;
    vector<int> paramusebasis;
    vector<int> paramvars_HGRAD, paramvars_HVOL, paramvars_HDIV, paramvars_HCURL, paramvars_HFACE;
    vector<string> paramvarlist_HGRAD, paramvarlist_HVOL, paramvarlist_HDIV, paramvarlist_HCURL, paramvarlist_HFACE;
    
    size_t numAux;
    
    // Editing for multi-set
    vector<size_t> numVars;
    //vector<vector<View_EvalT2> > uvals, u_dotvals;
    vector<View_EvalT2> sol_vals, sol_dot_vals;
    vector<vector<size_t>> sol_vals_index; // [set][var]

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
    string sidename, blockname;//, var;
    int currentside, time_step;
    
    bool have_rotation, have_rotation_phi;
    View_Sc3 rotation;
    View_Sc2 rotation_phi, extra_data;
    View_Sc3 multidata;
    
    // Profile timers
    Teuchos::RCP<Teuchos::Time> worksetResetTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::reset*");
    Teuchos::RCP<Teuchos::Time> worksetComputeSolnSeededTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::computeSolnVolIP - allocate/compute seeded");
    Teuchos::RCP<Teuchos::Time> worksetComputeSolnSideTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::computeSolnSideIP");
    Teuchos::RCP<Teuchos::Time> worksetgetDataTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::getData");
    Teuchos::RCP<Teuchos::Time> worksetgetDataScTimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::workset::getDataSc");
    
  };
  
}

#endif
