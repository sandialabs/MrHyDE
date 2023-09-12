/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_SUBGRIDMODEL_H
#define MRHYDE_SUBGRIDMODEL_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "solutionStorage.hpp"

namespace MrHyDE {
  
  class SubGridModel {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SubgridSolverNode>   SG_CrsMatrix;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,SubgridSolverNode> SG_MultiVector;
    typedef Tpetra::Map<LO, GO, SubgridSolverNode>               SG_Map;
    typedef Teuchos::RCP<SG_MultiVector>                         vector_RCP;
    typedef Teuchos::RCP<SG_CrsMatrix>                           matrix_RCP;
    
  public:
    
    SubGridModel() {} ;
    
    virtual ~SubGridModel() {};
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    virtual int addMacro(DRV & macronodes_,
                         Kokkos::View<int****,HostDevice> & macrosideinfo_,
                         LIDView macroLIDs,
                         Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> & macroorientation) = 0;
    
    
    virtual void finalize(const int & globalSize, const int & globalPID,
                          const bool & write_subgrid_soln,
                          vector<string> & appends) = 0;
    
    virtual void subgridSolver(View_Sc3 coarse_fwdsoln,
                               View_Sc4 coarse_prevsoln,
                               View_Sc3 coarse_adjsoln,
                               const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                               const bool & compute_jacobian, const bool & compute_sens,
                               const int & num_active_params,
                               const bool & compute_disc_sens, const bool & compute_aux_sens,
                               Workset<AD> & macrowkset, const int & macroelemindex,
                               const int & macrogrp,
                               Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) = 0;
    
    //virtual Kokkos::View<ScalarT**,AssemblyDevice> computeError(const ScalarT & time,
    //                                                           const int & macrogrp) = 0;
    
    virtual vector<std::pair<string, string> > getErrorList() = 0;
    
    virtual Kokkos::View<ScalarT**,HostDevice> computeError(vector<std::pair<string, string> > & sub_error_list,
                                                            const vector<ScalarT> & times) = 0;
    
    virtual Kokkos::View<ScalarT*,HostDevice> computeError(const ScalarT & times) = 0;
    
    virtual Kokkos::View<AD*,AssemblyDevice> computeObjective(const string & response_type,
                                                              const int & seedwhat,
                                                              const ScalarT & time,
                                                              const int & macrogrp) = 0;
    
    //virtual void writeSolution(const string & filename, const int & macrogrp) = 0;
    
    virtual void writeSolution(const ScalarT & time, const string & append="") = 0;
    
    virtual void addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points,
                            const ScalarT & sensor_loc_tol,
                            const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data,
                            const bool & have_sensor_data,
                            const vector<basis_RCP> & basisTypes, const int & macrogrp) = 0;
    
    virtual matrix_RCP getProjectionMatrix() = 0;
    
    virtual matrix_RCP getProjectionMatrix(DRV & ip, DRV & wts, Teuchos::RCP<const SG_Map> & other_owned_map,
                                           Teuchos::RCP<const SG_Map> & other_overlapped_map,
                                           std::pair<Kokkos::View<int**,AssemblyDevice> , vector<DRV> > & other_basisinfo) = 0;
    
    virtual vector_RCP getVector() = 0;

    virtual void advance() = 0;
    
    virtual void advanceStage() = 0;
    
    virtual DRV getIP() = 0;
    
    virtual DRV getIPWts() = 0;
    
    virtual std::pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis2(const DRV & ip) = 0;
    
    virtual matrix_RCP getEvaluationMatrix(const DRV & newip, Teuchos::RCP<SG_Map> & ip_map) = 0;
    
    virtual LIDView getCellLIDs(const int & cellnum) = 0;
    
    virtual void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) = 0;
    
    virtual void addMeshData() = 0;
    
    virtual void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) = 0;
    
    virtual ScalarT getPreviousTime() = 0;

    virtual void setPreviousTime(ScalarT & time) = 0;

    virtual void updateActive(vector<bool> & new_active) = 0;

    //virtual SG_Map getLinearAlgebraMap() = 0;
    
    Teuchos::RCP<MpiComm> LocalComm;
    Teuchos::RCP<SolutionStorage<SubgridSolverNode> > soln, solndot, adjsoln;
    vector<Teuchos::RCP<SG_MultiVector> > prev_soln, curr_soln, stage_soln;
    vector<Teuchos::RCP<SG_MultiVector> > prev_adjsoln, curr_adjsoln;
    
    Teuchos::RCP<const SG_Map> owned_map, overlapped_map;

    bool useMachineLearning = false;
    
    vector<Teuchos::RCP<Workset<AD> > > wkset;
    vector<basis_RCP> macro_basis_pointers;
    vector<string> macro_basis_types;
    vector<string> macro_varlist;
    vector<int> macro_usebasis;
    //vector<vector<int> > macro_offsets;
    Kokkos::View<LO**,AssemblyDevice> macro_offsets;
    Kokkos::View<int*,AssemblyDevice> macro_numDOF;
    
    vector<string> macro_paramnames, macro_disc_paramnames, macrosidenames;
    size_t macro_block;
    ScalarT cost_estimate;
    bool subgrid_static;
    
    vector<Teuchos::RCP<vector<AD> > > paramvals_AD;
    
    string usage, name;
    Kokkos::View<AD**,AssemblyDevice> paramvals_KVAD;
    
    vector<string> varlist;
    vector<bool> active;
    
  };
  
}

#endif

