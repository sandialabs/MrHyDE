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

/** \file   analysisManager.hpp
 \brief  Contains all of the assembly routines in MrHyDE.  Also creates the elements groups and the worksets.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_ASSEMBLY_MANAGER_H
#define MRHYDE_ASSEMBLY_MANAGER_H

#include "trilinos.hpp"
#include "Panzer_DOFManager.hpp"

#include "preferences.hpp"
#include "groupMetaData.hpp"
#include "group.hpp"
#include "boundaryGroup.hpp"
#include "workset.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "parameterManager.hpp"

namespace MrHyDE {
  
  /** \class  MrHyDE::AssemblyManager
   \brief  Provides the functionality for the MrHyDE-specific assembly routines for both implicit and explicit methods.
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
    
    AssemblyManager() {};
    
    ~AssemblyManager() {};
    
    AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_,
                    Teuchos::RCP<Teuchos::ParameterList> & settings,
                    Teuchos::RCP<MeshInterface> & mesh_,
                    Teuchos::RCP<DiscretizationInterface> & disc_,
                    Teuchos::RCP<PhysicsInterface> & phys_,
                    Teuchos::RCP<ParameterManager<Node> > & params_);
    
    
    // ========================================================================================
    // ========================================================================================
    
    void createFixedDOFs();

    // ========================================================================================
    // ========================================================================================
    
    void createGroups();
    
    void allocateGroupStorage();
      
    // ========================================================================================
    // ========================================================================================
    
    void createWorkset();
    
    // ========================================================================================
    // ========================================================================================
    
    void updateJacDBC(matrix_RCP & J, const std::vector<std::vector<GO> > & dofs,
                      const size_t & block, const bool & compute_disc_sens);
    
    void updateJacDBC(matrix_RCP & J, const std::vector<LO> & dofs, const bool & compute_disc_sens);
    
    
    void setDirichlet(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                      const ScalarT & time, const bool & lumpmass=false);
    
    // ========================================================================================
    // ========================================================================================
    
    void setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                    const bool & lumpmass=false, const ScalarT & scale = 1.0);
    
    void setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                    const bool & lumpmass, const ScalarT & scale,
                    const size_t & block, const size_t & groupblock);
    
    void setInitial(const size_t & set, vector_RCP & initial, const bool & useadjoint);

    // TODO BWR -- finish when appropriate
    /* @brief Create the mass matrix and RHS for an L2 projection of the initial
     * condition over the faces.
     *
     * @param[inout] rhs  RHS vector
     * @param[inout] mass Mass matrix
     * @param[in] lumpmass Bool indicating if a lumped mass matrix approximation is requested
     *
     * @details The current use case is for projection the coarse-scale initial condition on the 
     * mesh skeleton (HFACE).
     *
     * @warning BWR -- Under development, I am trying to take things from setDirichlet and setInitial.
     * I think some combination of the two should work, but need to better understand.
     */
    
    void setInitialFace(const size_t & set, vector_RCP & rhs, matrix_RCP & mass,const bool & lumpmass=false);

    void getWeightedMass(const size_t & set, matrix_RCP & mass, vector_RCP & massdiag);
    
    void getWeightVector(const size_t & set, vector_RCP & wts);
    
    // ========================================================================================
    // ========================================================================================
    
    void assembleJacRes(const size_t & set, vector_RCP & u, vector_RCP & phi,
                        const bool & compute_jacobian, const bool & compute_sens,
                        const bool & compute_disc_sens,
                        vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                        const ScalarT & current_time, const bool & useadjoint,
                        const bool & store_adjPrev,
                        const int & num_active_params, vector_RCP & Psol,
                        const bool & is_final_time,
                        const ScalarT & deltat);
    
    
    void assembleJacRes(const size_t & set, const bool & compute_jacobian, const bool & compute_sens,
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
    
    void dofConstraints(const size_t & set, matrix_RCP & J, vector_RCP & res, const ScalarT & current_time,
                        const bool & compute_jacobian, const bool & compute_disc_sens);
    
    // ========================================================================================
    //
    // ========================================================================================
    
    void resetPrevSoln(const size_t & set);
    
    void revertSoln(const size_t & set);
    
    void resetStageSoln(const size_t & set);
    
    void updateStage(const int & stage, const ScalarT & current_time, const ScalarT & deltat);
    
    void updateStageSoln(const size_t & set);
    
    void updatePhysicsSet(const size_t & set);
    
    // ========================================================================================
    // Gather 
    // ========================================================================================
    
    void performGather(const size_t & set, const vector_RCP & vec, const int & type, const size_t & index);
    
    template<class ViewType>
    void performGather(const size_t & set, ViewType vec_dev, const int & type);
        
    template<class ViewType>
    void performBoundaryGather(const size_t & set, ViewType vec_dev, const int & type);
    
    // ========================================================================================
    // Scatter 
    // ========================================================================================
    
    template<class MatType, class LocalViewType, class LIDViewType>
    void scatterJac(const size_t & set, MatType J_kcrs, LocalViewType local_J,
                    LIDViewType LIDs, LIDViewType paramLIDs,
                    const bool & compute_disc_sens);

    template<class VecViewType, class LocalViewType, class LIDViewType>
    void scatterRes(VecViewType res_view,
                    LocalViewType local_res, LIDViewType LIDs);

    template<class MatType, class VecViewType, class LIDViewType>
    void scatter(const size_t & set,MatType J_kcrs, VecViewType res_view,
                 LIDViewType LIDs, LIDViewType paramLIDs,
                 const int & block,
                 const bool & compute_jacobian,
                 const bool & compute_sens,
                 const bool & compute_disc_sens,
                 const bool & isAdjoint);
    
    // Computes y = M*x
    void applyMassMatrixFree(const size_t & set, vector_RCP & x, vector_RCP & y);
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    void purgeMemory();
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<MpiComm> Comm;
    Teuchos::RCP<Teuchos::ParameterList> settings;
    
    // Need
    std::vector<std::string> blocknames;
    std::vector<std::vector<std::vector<std::string> > > varlist; // [set][block][var]
    
    //Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
    Teuchos::RCP<MeshInterface>  mesh;
    Teuchos::RCP<DiscretizationInterface> disc;
    Teuchos::RCP<PhysicsInterface> phys;
    
    size_t globalParamUnknowns;
    int verbosity, debug_level;
    
    // Groupss and worksets are unique to each block, but span the physics sets
    std::vector<Teuchos::RCP<GroupMetaData> > groupData;
    std::vector<std::vector<Teuchos::RCP<Group> > > groups;
    std::vector<std::vector<Teuchos::RCP<BoundaryGroup> > > boundary_groups;
    std::vector<Teuchos::RCP<workset> > wkset;
    
    bool usestrongDBCs, use_meas_as_dbcs, multiscale, isTransient, fix_zero_rows, lump_mass, matrix_free;
    
    std::string assembly_partitioning;
    std::vector<std::vector<bool> > assemble_volume_terms, assemble_boundary_terms, assemble_face_terms; // use basis functions in assembly [block][set]
    std::vector<bool> build_volume_terms, build_boundary_terms, build_face_terms; // set up basis function [block]
    std::vector<Kokkos::View<bool*,LA_device> > isFixedDOF; // [set]
    std::vector<vector<vector<Kokkos::View<LO*,LA_device> > > > fixedDOF; // [set][block][var]
    Teuchos::RCP<ParameterManager<Node> > params;
      
    Teuchos::RCP<Teuchos::Time> assemblytimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - total assembly");
    Teuchos::RCP<Teuchos::Time> gathertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::gather()");
    Teuchos::RCP<Teuchos::Time> phystimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - physics evaluation");
    Teuchos::RCP<Teuchos::Time> boundarytimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - boundary evaluation");
    Teuchos::RCP<Teuchos::Time> scattertimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::scatter()");
    Teuchos::RCP<Teuchos::Time> dbctimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::dofConstraints()");
    Teuchos::RCP<Teuchos::Time> completetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - fill complete");
    Teuchos::RCP<Teuchos::Time> msprojtimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - multiscale projection");
    Teuchos::RCP<Teuchos::Time> setinittimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::setInitial()");
    Teuchos::RCP<Teuchos::Time> setdbctimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::setDirichlet()");
    Teuchos::RCP<Teuchos::Time> grouptimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::createGroups()");
    Teuchos::RCP<Teuchos::Time> wksettimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::createWorkset()");
    Teuchos::RCP<Teuchos::Time> groupdatabaseCreatetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::dataBase - assignment");
    Teuchos::RCP<Teuchos::Time> groupdatabaseBasistimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::dataBase - basis");
  };
  
}

#endif
