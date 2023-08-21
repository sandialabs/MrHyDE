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

#ifndef MRHYDE_PARAMETER_MANAGER_H
#define MRHYDE_PARAMETER_MANAGER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "group.hpp"
#include "boundaryGroup.hpp"
#include "Panzer_STK_Interface.hpp"
#include "discretizationInterface.hpp"
#include "MrHyDE_OptVector.hpp"

namespace MrHyDE {
  
  template<class Node>
  class ParameterManager {
    
    typedef Tpetra::Export<LO, GO, Node>            LA_Export;
    typedef Tpetra::Import<LO, GO, Node>            LA_Import;
    typedef Tpetra::Map<LO, GO, Node>               LA_Map;
    typedef Tpetra::CrsGraph<LO,GO,Node>            LA_CrsGraph;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;
    typedef typename Node::device_type              LA_device;
    
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    ParameterManager() {};
    
    ~ParameterManager() {};
    
    ParameterManager(const Teuchos::RCP<MpiComm> & Comm_,
                     Teuchos::RCP<Teuchos::ParameterList> & settings,
                     Teuchos::RCP<panzer_stk::STK_Interface> & mesh_,
                     Teuchos::RCP<PhysicsInterface> & phys_,
                     Teuchos::RCP<DiscretizationInterface> & disc_);
    
    // ========================================================================================
    // Set up the parameters (inactive, active, stochastic, discrete)
    // Communicate these parameters back to the physics interface and the enabled modules
    // ========================================================================================
    
    void setupParameters();
    
    void setupDiscretizedParameters(std::vector<std::vector<Teuchos::RCP<Group> > > & groups,
                                    std::vector<std::vector<Teuchos::RCP<BoundaryGroup> > > & boundary_groups);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getNumParams(const int & type);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getNumParams(const std::string & type);
    
    // ========================================================================================
    // return the discretized parameters as std::vector for use with ROL
    // ========================================================================================
    
    std::vector<ScalarT> getDiscretizedParamsVector();
    
    vector_RCP getDiscretizedParams();
    
    std::vector<vector_RCP> getDynamicDiscretizedParams();
    
    // ========================================================================================
    // ========================================================================================
    
    void sacadoizeParams(const bool & seed_active);
    
    void sacadoizeParamsSc(const bool & seed_active,
                         Kokkos::View<int*,AssemblyDevice> ptypes,
                         Kokkos::View<size_t*,AssemblyDevice> plengths,
                         Kokkos::View<size_t**,AssemblyDevice> pseed,
                         Kokkos::View<ScalarT**,AssemblyDevice> pvals,
                         vector<Teuchos::RCP<vector<ScalarT> > > & v_pvals,
                         Kokkos::View<ScalarT**,AssemblyDevice> kv_pvals);

    template<class EvalT>
    void sacadoizeParams(const bool & seed_active,
                         Kokkos::View<int*,AssemblyDevice> ptypes,
                         Kokkos::View<size_t*,AssemblyDevice> plengths,
                         Kokkos::View<size_t**,AssemblyDevice> pseed,
                         Kokkos::View<ScalarT**,AssemblyDevice> pvals,
                         vector<Teuchos::RCP<vector<EvalT> > > & v_pvals,
                         Kokkos::View<EvalT**,AssemblyDevice> kv_pvals);

    // ========================================================================================
    // ========================================================================================
    
    void updateParams(MrHyDE_OptVector & newparams);

    void updateParams(const std::vector<ScalarT> & newparams, const int & type);
    
    void updateDynamicParams(const int & timestep);
    
    // ========================================================================================
    // ========================================================================================
    
    void updateParams(const std::vector<ScalarT> & newparams, const std::string & stype);
    
    // ========================================================================================
    // ========================================================================================
    
    void setParam(const std::vector<ScalarT> & newparams, const std::string & name);
    
    // ========================================================================================
    // ========================================================================================
    
    Teuchos::RCP<std::vector<ScalarT> > getParams(const int & type);
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<std::string> getParamsNames(const int & type);
    
    // ========================================================================================
    // ========================================================================================
    
    bool isParameter(const std::string & name);
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<size_t> getParamsLengths(const int & type);
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<ScalarT> getParams(const std::string & stype);
    
    MrHyDE_OptVector getCurrentVector();
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<Teuchos::RCP<std::vector<ScalarT> > > getActiveParamBounds();

    std::vector<vector_RCP> getDiscretizedParamBounds();
    
    // ========================================================================================
    // ========================================================================================
    
    void stashParams();
    
    // ========================================================================================
    // ========================================================================================
    
    void setInitialParams();
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<ScalarT> getStochasticParams(const std::string & whichparam);
    
    // ========================================================================================
    // ========================================================================================
    
    std::vector<ScalarT> getFractionalParams(const std::string & whichparam);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void purgeMemory();
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    std::vector<std::string> blocknames;
    int spaceDim, debug_level, numTimeSteps;
    
    Teuchos::RCP<const LA_Map> param_owned_map;
    Teuchos::RCP<const LA_Map> param_overlapped_map;
    Teuchos::RCP<LA_CrsGraph> param_overlapped_graph;
    
    Teuchos::RCP<LA_Export> param_exporter;
    Teuchos::RCP<LA_Import> param_importer;
    
    std::vector<std::string> paramnames;
    std::vector<std::vector<ScalarT> > paramvals;

    std::vector<Teuchos::RCP<std::vector<ScalarT> > > paramvals_Sc;
#ifndef MrHyDE_NO_AD
    std::vector<Teuchos::RCP<std::vector<AD> > > paramvals_AD;
    std::vector<Teuchos::RCP<std::vector<AD2> > > paramvals_AD2;
    std::vector<Teuchos::RCP<std::vector<AD4> > > paramvals_AD4;
    std::vector<Teuchos::RCP<std::vector<AD8> > > paramvals_AD8;
    std::vector<Teuchos::RCP<std::vector<AD16> > > paramvals_AD16;
    std::vector<Teuchos::RCP<std::vector<AD18> > > paramvals_AD18;
    std::vector<Teuchos::RCP<std::vector<AD24> > > paramvals_AD24;
    std::vector<Teuchos::RCP<std::vector<AD32> > > paramvals_AD32;
#endif
    
    Kokkos::View<ScalarT**,AssemblyDevice> paramvals_KV;
#ifndef MrHyDE_NO_AD
    Kokkos::View<AD**,AssemblyDevice> paramvals_KVAD;
    Kokkos::View<AD2**,AssemblyDevice> paramvals_KVAD2;
    Kokkos::View<AD4**,AssemblyDevice> paramvals_KVAD4;
    Kokkos::View<AD8**,AssemblyDevice> paramvals_KVAD8;
    Kokkos::View<AD16**,AssemblyDevice> paramvals_KVAD16;
    Kokkos::View<AD18**,AssemblyDevice> paramvals_KVAD18;
    Kokkos::View<AD24**,AssemblyDevice> paramvals_KVAD24;
    Kokkos::View<AD32**,AssemblyDevice> paramvals_KVAD32;
#endif
    
    vector_RCP Psol, Psol_over;
    std::vector<vector_RCP> dynamic_Psol, dynamic_Psol_over;
    //std::vector<vector_RCP> auxsol;
    bool have_dynamic;
    
    Teuchos::RCP<const panzer::DOFManager> discparamDOF;
    std::vector<Kokkos::View<const LO**, Kokkos::LayoutRight, PHX::Device>> DOF_LIDs;
    std::vector<std::vector<GO> > DOF_owned, DOF_ownedAndShared;
    std::vector<std::vector<std::vector<std::vector<GO>>>> DOF_GIDs; // [set][block][elem][gid] may consider a different storage strategy
    

    std::vector<std::vector<ScalarT> > paramLowerBounds;
    std::vector<std::vector<ScalarT> > paramUpperBounds;
    std::vector<std::string> discretized_param_basis_types;
    std::vector<int> discretized_param_basis_orders, discretized_param_usebasis;
    std::vector<std::string> discretized_param_names;
    std::vector<basis_RCP> discretized_param_basis;
    std::vector<bool> discretized_param_dynamic;
    Teuchos::RCP<panzer::DOFManager> paramDOF;
    std::vector<std::vector<int> > paramoffsets;
    std::vector<int> paramNumBasis;
    int numParamUnknowns;     					 // total number of unknowns
    int numParamUnknownsOS;     					 // total number of unknowns
    int globalParamUnknowns; // total number of unknowns across all processors
    std::vector<GO> paramOwned;					 // GIDs that live on the local processor.
    std::vector<GO> paramOwnedAndShared;				 // GIDs that live or are shared on the local processor.
    
    std::vector<int> paramtypes;
    std::vector<std::vector<GO>> paramNodes;  // for distinguishing between parameter fields when setting initial
    std::vector<std::vector<GO>> paramNodesOS;// values and bounds
    size_t num_inactive_params, num_active_params, num_stochastic_params, num_discrete_params, num_discretized_params;
    std::vector<ScalarT> initialParamValues, lowerParamBounds, upperParamBounds, discparamVariance;
    
    //std::vector<ScalarT> domainRegConstants, boundaryRegConstants;
    //std::vector<std::string> boundaryRegSides;
    //std::vector<int> domainRegTypes, domainRegIndices, boundaryRegTypes, boundaryRegIndices;
    
    int verbosity;
    string response_type, multigrid_type, smoother_type;
    bool discretized_stochastic, use_custom_initial_param_guess;
    
    std::vector<std::string> stochastic_distribution, discparam_distribution;
    std::vector<ScalarT> stochastic_mean, stochastic_variance, stochastic_min, stochastic_max;
    
    int batchID;
    
    //fractional parameters
    std::vector<ScalarT> s_exp;
    std::vector<ScalarT> h_mesh;
    
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
    Teuchos::RCP<DiscretizationInterface> disc;
    Teuchos::RCP<PhysicsInterface> phys;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    
  };
  
}

#endif
