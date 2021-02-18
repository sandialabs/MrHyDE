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

#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include "trilinos.hpp"
#include "Panzer_DOFManager.hpp"

#include "preferences.hpp"
#include "cell.hpp"
#include "boundaryCell.hpp"
#include "workset.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "parameterManager.hpp"

namespace MrHyDE {
  /*
  void static assemblyHelp(const std::string & details) {
    cout << "********** Help and Documentation for the Assembly Manager **********" << endl;
  }
  */
  
  template< class Node>
  class AssemblyManager {
    
    typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;
    typedef Tpetra::CrsGraph<LO,GO,Node>            LA_CrsGraph;
    typedef Tpetra::Export<LO, GO, Node>            LA_Export;
    typedef Tpetra::Import<LO, GO, Node>            LA_Import;
    typedef Tpetra::Map<LO, GO, Node>               LA_Map;
    typedef Tpetra::Operator<ScalarT,LO,GO,Node>    LA_Operator;
    typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;
    typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;
    typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
    typedef typename Node::device_type              LA_device;
    typedef typename Node::memory_space             LA_mem;
    
  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
                    Teuchos::RCP<panzer_stk::STK_Interface> & mesh_, Teuchos::RCP<discretization> & disc_,
                    Teuchos::RCP<physics> & phys_, Teuchos::RCP<ParameterManager<Node> > & params_,
                    const int & numElemPerCell_);
    
    
    // ========================================================================================
    // ========================================================================================
    
    void createFixedDOFs();

    // ========================================================================================
    // ========================================================================================
    
    void createCells();
    
    // ========================================================================================
    // ========================================================================================
    
    void createWorkset();
    
    // ========================================================================================
    // ========================================================================================
    
    void updateJacDBC(matrix_RCP & J, const std::vector<std::vector<GO> > & dofs,
                      const size_t & block, const bool & compute_disc_sens);
    
    void updateJacDBC(matrix_RCP & J, const std::vector<LO> & dofs, const bool & compute_disc_sens);
    
    // ========================================================================================
    // ========================================================================================
    
    void setInitial(vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                    const bool & lumpmass=false, const ScalarT & scale = 1.0);
    
    // ========================================================================================
    // ========================================================================================
    
    void setDirichlet(vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                      const ScalarT & time, const bool & lumpmass=false);
    
    void setInitial(vector_RCP & initial, const bool & useadjoint);
    
    // ========================================================================================
    // ========================================================================================
    
    void assembleJacRes(vector_RCP & u, vector_RCP & phi,
                        const bool & compute_jacobian, const bool & compute_sens,
                        const bool & compute_disc_sens,
                        vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                        const ScalarT & current_time, const bool & useadjoint,
                        const bool & store_adjPrev,
                        const int & num_active_params, vector_RCP & Psol,
                        const bool & is_final_time,
                        const ScalarT & deltat);
    
    
    void assembleJacRes(const bool & compute_jacobian, const bool & compute_sens,
                        const bool & compute_disc_sens,
                        vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                        const ScalarT & current_time, const bool & useadjoint,
                        const bool & store_adjPrev,
                        const int & num_active_params,
                        const bool & is_final_time, const int & block,
                        const ScalarT & deltat);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    void dofConstraints(matrix_RCP & J, vector_RCP & res, const ScalarT & current_time,
                        const bool & compute_jacobian, const bool & compute_disc_sens);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    void resetPrevSoln();
    
    void resetStageSoln();
    
    void updateStageNumber(const int & stage);
    
    void updateStageSoln();
    
    // ========================================================================================
    //
    // ========================================================================================
    
    void performGather(const vector_RCP & vec, const int & type, const size_t & index);
    
    template<class ViewType>
    void performGather(ViewType vec_dev, const int & type);
    //void performGather(Kokkos::View<ScalarT*,AssemblyDevice> vec_dev, const int & type);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    template<class ViewType>
    void performBoundaryGather(ViewType vec_dev, const int & type);
    //void performBoundaryGather(Kokkos::View<ScalarT*,AssemblyDevice> vec_dev, const int & type);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    template<class MatType, class VecViewType, class LocalViewType, class LIDViewType>
    void scatter(MatType J_kcrs, VecViewType res_view,
                 LocalViewType local_res, LocalViewType local_J,
                 LIDViewType LIDs, LIDViewType paramLIDs,
                 const bool & compute_jacobian, const bool & compute_disc_sens);

    template<class MatType, class VecViewType, class LIDViewType>
    void scatter(MatType J_kcrs, VecViewType res_view,
                 LIDViewType LIDs, LIDViewType paramLIDs,
                 const int & block,
                 const bool & compute_jacobian,
                 const bool & compute_sens,
                 const bool & compute_disc_sens,
                 const bool & isAdjoint);
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    
    // Need
    std::vector<std::string> blocknames;
    std::vector<std::vector<std::string> > varlist;
    std::vector<LO> numVars;
    
    Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
    Teuchos::RCP<discretization> disc;
    Teuchos::RCP<physics> phys;
    
    size_t globalParamUnknowns;
    int verbosity, debug_level;
    
    std::vector<Teuchos::RCP<CellMetaData> > cellData;
    std::vector<std::vector<Teuchos::RCP<cell> > > cells;
    std::vector<std::vector<Teuchos::RCP<BoundaryCell> > > boundaryCells;
    std::vector<Teuchos::RCP<workset> > wkset;
    
    bool usestrongDBCs, use_meas_as_dbcs, multiscale, isTransient, fix_zero_rows;
    std::string assembly_partitioning;
    std::vector<bool> assemble_volume_terms, assemble_boundary_terms, assemble_face_terms; // use basis functions in assembly
    std::vector<bool> build_volume_terms, build_boundary_terms, build_face_terms; // set up basis function
    Kokkos::View<bool*,LA_device> isFixedDOF;
    vector<vector<Kokkos::View<LO*,LA_device> > > fixedDOF;
    Teuchos::RCP<ParameterManager<Node> > params;
    int numElemPerCell;
      
    Teuchos::RCP<Teuchos::Time> assemblytimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - total assembly");
    Teuchos::RCP<Teuchos::Time> gathertimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::gather()");
    Teuchos::RCP<Teuchos::Time> phystimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - physics evaluation");
    Teuchos::RCP<Teuchos::Time> boundarytimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - boundary evaluation");
    Teuchos::RCP<Teuchos::Time> scattertimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::scatter()");
    Teuchos::RCP<Teuchos::Time> dbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::dofConstraints()");
    Teuchos::RCP<Teuchos::Time> completetimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - fill complete");
    Teuchos::RCP<Teuchos::Time> msprojtimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::computeJacRes() - multiscale projection");
    Teuchos::RCP<Teuchos::Time> setinittimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::setInitial()");
    Teuchos::RCP<Teuchos::Time> setdbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::setDirichlet()");
    Teuchos::RCP<Teuchos::Time> celltimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::createCells()");
    Teuchos::RCP<Teuchos::Time> wksettimer = Teuchos::TimeMonitor::getNewCounter("MILO::assembly::createWorkset()");
    
  };
  
}

#endif
