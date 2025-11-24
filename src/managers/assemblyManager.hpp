/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

/** \file assemblyManager.hpp
 *  \brief Contains all of the assembly routines in MrHyDE and creates element groups and worksets.
 *  \author Created by T. Wildey
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
#include "multiscaleManager.hpp"
#include "functionManager.hpp"
#include "data.hpp"
#include "MrHyDE_Debugger.hpp"

namespace MrHyDE {

/** \class AssemblyManager
 *  \brief Provides the functionality for MrHyDE's assembly routines for both implicit and explicit formulations.
 *
 *  This class manages DOFs, worksets, boundary conditions, group creation,
 *  function evaluation, Jacobian updates, and initial condition projections.
 */
template<class Node>
class AssemblyManager {
  
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,Node>   LA_CrsMatrix;  ///< Local alias for Tpetra CRS matrix
  typedef Tpetra::CrsGraph<LO,GO,Node>            LA_CrsGraph;   ///< Local alias for Tpetra CRS graph
  typedef Tpetra::Export<LO, GO, Node>            LA_Export;     ///< Local alias for Tpetra Export
  typedef Tpetra::Import<LO, GO, Node>            LA_Import;     ///< Local alias for Tpetra Import
  typedef Tpetra::Map<LO, GO, Node>               LA_Map;        ///< Local alias for Tpetra Map
  typedef Tpetra::Operator<ScalarT,LO,GO,Node>    LA_Operator;   ///< Local alias for Tpetra Operator
  typedef Tpetra::MultiVector<ScalarT,LO,GO,Node> LA_MultiVector;///< Local alias for Tpetra MultiVector
  typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;    ///< RCP to MultiVector
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;    ///< RCP to CRS matrix
  typedef typename Node::device_type              LA_device;     ///< Device type
  typedef typename Node::memory_space             LA_mem;        ///< Memory space
  
public:
  
  /** \brief Default constructor. */
  AssemblyManager() {};
  
  /** \brief Default destructor. */
  ~AssemblyManager() {};
  
  /** \brief Construct the AssemblyManager and initialize major system interfaces.
   *
   *  \param[in] Comm_  MPI communicator
   *  \param[in] settings Parameter list for configuration
   *  \param[in] mesh_  Mesh interface
   *  \param[in] disc_  Discretization interface
   *  \param[in] phys_  Physics interface
   *  \param[in] params_ Parameter manager
   */
  AssemblyManager(const Teuchos::RCP<MpiComm> & Comm_,
                  Teuchos::RCP<Teuchos::ParameterList> & settings,
                  Teuchos::RCP<MeshInterface> & mesh_,
                  Teuchos::RCP<DiscretizationInterface> & disc_,
                  Teuchos::RCP<PhysicsInterface> & phys_,
                  Teuchos::RCP<ParameterManager<Node> > & params_);
  
  /** \brief Identify and create the fixed DOFs for the problem. */
  void createFixedDOFs();
  
  /** \brief Create element groups required for assembly. */
  void createGroups();
  
  /** \brief Allocate storage for all computed group quantities. */
  void allocateGroupStorage();
  
  /** \brief Build the workset that drives integration and assembly. */
  void createWorkset();
  
  /** \brief Register a user-specified function for evaluation.
   *  \param[in] block Block index
   *  \param[in] name Function name
   *  \param[in] expression Mathematical expression
   *  \param[in] location Evaluation location (e.g., cell, side)
   */
  void addFunction(const int & block, const string & name, const string & expression, const string & location);
  
  /** \brief Evaluate a registered function.
   *  \param[in] block Block index
   *  \param[in] name Function name
   *  \param[in] location Evaluation location
   *  \return 2D view of scalar function values
   */
  View_Sc2 evaluateFunction(const int & block, const string & name, const string & location);
  
  /** \brief Evaluate a registered function and its sensitivities.
   *  \param[in] block Block index
   *  \param[in] name Function name
   *  \param[in] location Evaluation location
   *  \return 3D view of function values and sensitivities
   */
  View_Sc3 evaluateFunctionWithSens(const int & block, const string & name, const string & location);
  
  /** \brief Apply Dirichlet BC modifications to the Jacobian.
   *  \param[inout] J Jacobian matrix
   *  \param[in] dofs DOF list per variable/block
   *  \param[in] block Block index
   *  \param[in] compute_disc_sens Whether to compute sensitivities
   */
  void updateJacDBC(matrix_RCP & J, const std::vector<std::vector<GO> > & dofs,
                    const size_t & block, const bool & compute_disc_sens);
  
  /** \brief Apply Dirichlet BC modifications for a single DOF list.
   *  \param[inout] J Jacobian matrix
   *  \param[in] dofs DOF list
   *  \param[in] compute_disc_sens Whether to compute sensitivities
   */
  void updateJacDBC(matrix_RCP & J, const std::vector<LO> & dofs, const bool & compute_disc_sens);
  
  /** \brief Apply Dirichlet boundary conditions to RHS and optionally mass matrix.
   *  \param[in] set Boundary condition set index
   *  \param[inout] rhs Right-hand-side vector
   *  \param[inout] mass Mass matrix
   *  \param[in] useadjoint Whether adjoint form is used
   *  \param[in] time Current simulation time
   *  \param[in] lumpmass Whether mass matrix should be lumped
   */
  void setDirichlet(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                    const ScalarT & time, const bool & lumpmass=false);
  
  /** \brief Apply initial conditions to RHS and mass matrix.
   *  \param[in] set Initial condition set index
   *  \param[inout] rhs Right-hand-side vector
   *  \param[inout] mass Mass matrix
   *  \param[in] useadjoint Whether adjoint form is used
   *  \param[in] lumpmass Whether mass is lumped
   *  \param[in] scale Scaling factor for initial conditions
   */
  void setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                  const bool & lumpmass=false, const ScalarT & scale = 1.0);
  
  /** \brief Apply initial conditions for a specific block and group.
   *  \param[in] set Initial condition set index
   *  \param[inout] rhs RHS vector
   *  \param[inout] mass Mass matrix
   *  \param[in] useadjoint Whether adjoint is used
   *  \param[in] lumpmass Lumped mass matrix flag
   *  \param[in] scale Scaling factor
   *  \param[in] block Block index
   *  \param[in] groupblock Group block index
   */
  void setInitial(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & useadjoint,
                  const bool & lumpmass, const ScalarT & scale,
                  const size_t & block, const size_t & groupblock);
  
  /** \brief Apply initial conditions directly to a vector.
   *  \param[in] set Initial condition set index
   *  \param[inout] initial Vector to populate
   *  \param[in] useadjoint Whether adjoint form is used
   */
  void setInitial(const size_t & set, vector_RCP & initial, const bool & useadjoint);
  
  /** \brief Create mass matrix and RHS for L2 projection of initial conditions on faces.
   *  \param[inout] rhs RHS vector
   *  \param[inout] mass Mass matrix
   *  \param[in] lumpmass Whether mass matrix should be lumped
   *  \warning Under development
   */
  void setInitialFace(const size_t & set, vector_RCP & rhs, matrix_RCP & mass, const bool & lumpmass=false);
  
  /**
   * @brief Compute a weighted mass matrix and its diagonal.
   *
   * @param set Index of the set for which to compute the weighted mass.
   * @param mass Reference-counted pointer to the mass matrix to be filled.
   * @param massdiag Reference-counted pointer to the vector storing the diagonal of the mass matrix.
   */
  void getWeightedMass(const size_t & set, matrix_RCP & mass, vector_RCP & massdiag);
  
  /**
   * @brief Compute the parameter mass matrix and its diagonal.
   *
   * @param mass Reference-counted pointer to the parameter mass matrix.
   * @param massdiag Reference-counted pointer to the vector storing the diagonal of the mass matrix.
   */
  void getParamMass(matrix_RCP & mass, vector_RCP & massdiag);
  
  /**
   * @brief Compute the weight vector for a given set.
   *
   * @param set Index of the set.
   * @param wts Reference-counted pointer to the weight vector to be filled.
   */
  void getWeightVector(const size_t & set, vector_RCP & wts);
  
  /**
   * @brief Assemble the Jacobian and residual for a given set.
   *
   * @param set Set index.
   * @param stage Stage index.
   * @param sol Solution vectors.
   * @param sol_stage Stage solution vectors.
   * @param sol_prev Previous solution vectors.
   * @param phi Basis/test function vectors.
   * @param phi_stage Stage basis/test function vectors.
   * @param phi_step Step basis/test function vectors.
   * @param compute_jacobian Flag to compute Jacobian.
   * @param compute_sens Flag to compute sensitivities.
   * @param compute_disc_sens Flag to compute discretization sensitivities.
   * @param compute_previous_jac Flag to compute previous Jacobian.
   * @param stepindex Step index.
   * @param res Residual vector.
   * @param J Jacobian matrix.
   * @param isTransient Whether the problem is transient.
   * @param current_time Current time.
   * @param useadjoint Whether adjoint mode is used.
   * @param store_adjPrev Whether to store previous adjoint.
   * @param num_active_params Number of active parameters.
   * @param Psol Parameter solution vector.
   * @param Pdot Parameter time-derivative vector.
   * @param is_final_time Whether this is the final time step.
   * @param deltat Time step size.
   */
  void assembleJacRes(const size_t & set, const size_t & stage,
                      vector<vector_RCP> & sol,
                      vector<vector_RCP> & sol_stage,
                      vector<vector_RCP> & sol_prev,
                      vector<vector_RCP> & phi,
                      vector<vector_RCP> & phi_stage,
                      vector<vector_RCP> & phi_step,
                      const bool & compute_jacobian, const bool & compute_sens,
                      const bool & compute_disc_sens,
                      const bool & compute_previous_jac, const size_t & stepindex,
                      vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                      const ScalarT & current_time, const bool & useadjoint,
                      const bool & store_adjPrev,
                      const int & num_active_params,
                      vector_RCP & Psol,
                      vector_RCP & Pdot,
                      const bool & is_final_time,
                      const ScalarT & deltat);
  
  /**
   * @brief Template version of assembleJacRes for specialized evaluation types.
   *
   * @tparam EvalT Evaluation type.
   * @param set Set index.
   * @param stage Stage index.
   * @param sol Solution vectors.
   * @param sol_stage Stage solution vectors.
   * @param sol_prev Previous solution vectors.
   * @param phi Basis/test function vectors.
   * @param phi_stage Stage basis/test function vectors.
   * @param phi_step Step basis/test function vectors.
   * @param param_sol Parameter solution vector.
   * @param param_dot Parameter derivatives.
   * @param compute_jacobian Whether to compute Jacobian.
   * @param compute_sens Whether to compute sensitivities.
   * @param compute_disc_sens Whether to compute discretization sensitivities.
   * @param compute_previous_jac Whether to compute previous Jacobian.
   * @param stepindex Step index.
   * @param res Residual vector.
   * @param J Jacobian matrix.
   * @param isTransient Whether the problem is transient.
   * @param current_time The current simulation time.
   * @param useadjoint Whether to compute in adjoint mode.
   * @param store_adjPrev Whether to store previous adjoint.
   * @param num_active_params Number of active parameters.
   * @param is_final_time Whether this is final time step.
   * @param block Block index.
   * @param deltat Time step size.
   */
  template<class EvalT>
  void assembleJacRes(const size_t & set, const size_t & stage,
                      vector<vector_RCP> & sol,
                      vector<vector_RCP> & sol_stage,
                      vector<vector_RCP> & sol_prev,
                      vector<vector_RCP> & phi,
                      vector<vector_RCP> & phi_stage,
                      vector<vector_RCP> & phi_step,
                      vector_RCP & param_sol,
                      vector_RCP & param_dot,
                      const bool & compute_jacobian, const bool & compute_sens,
                      const bool & compute_disc_sens,
                      const bool & compute_previous_jac, const size_t & stepindex,
                      vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                      const ScalarT & current_time, const bool & useadjoint,
                      const bool & store_adjPrev,
                      const int & num_active_params,
                      const bool & is_final_time, const int & block,
                      const ScalarT & deltat);
  
  /**
   * @brief Assemble the residual for the given physics set.
   *
   * This routine computes the residual vector for a specified set and stage of the
   * simulation. It uses the solution fields, basis evaluations, and parameter
   * information to form the residual contributions. The Jacobian matrix is
   * provided in the argument list but is not assembled in this version.
   *
   * @param set Index of the physics set being assembled.
   * @param stage Stage index for multi-stage time integration.
   * @param sol Solution fields for all sets at the current time/state.
   * @param sol_stage Stage-level solution fields.
   * @param sol_prev Solution fields from the previous time step.
   * @param phi Basis/test function evaluations.
   * @param phi_stage Stage-level basis/test function evaluations.
   * @param phi_step Step-level basis/test function evaluations.
   * @param param_sol Parameter solution vector.
   * @param param_dot Time derivative of the parameter solution vector.
   * @param res Residual vector to be filled.
   * @param J Jacobian matrix (unused in this version, but provided for consistency).
   * @param isTransient Flag indicating whether the simulation is transient.
   * @param current_time Current simulation time.
   * @param deltat Time-step size.
   */
  void assembleRes(const size_t & set, const size_t & stage,
                   vector<vector_RCP> & sol,
                   vector<vector_RCP> & sol_stage,
                   vector<vector_RCP> & sol_prev,
                   vector<vector_RCP> & phi,
                   vector<vector_RCP> & phi_stage,
                   vector<vector_RCP> & phi_step,
                   vector_RCP & param_sol,
                   vector_RCP & param_dot,
                   vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                   const ScalarT & current_time,
                   const ScalarT & deltat);
  
  /**
   * @brief Assemble the residual for a specific block of a physics set.
   *
   * This overload provides block-level residual assembly, allowing partial sets or
   * decomposed physics components to be evaluated independently.
   *
   * @param set Index of the physics set.
   * @param stage Time-integration stage index.
   * @param sol Solution fields for all sets.
   * @param sol_stage Stage-level solution values.
   * @param sol_prev Solution values from the previous time step.
   * @param phi Basis/test function evaluations.
   * @param phi_stage Stage-level basis/test function data.
   * @param phi_step Step-level basis/test function data.
   * @param param_sol Parameter solution vector.
   * @param param_dot Time derivative of parameter vector.
   * @param res Residual vector to be assembled.
   * @param J Jacobian matrix (not used here).
   * @param isTransient Whether the simulation includes time-dependence.
   * @param current_time Current simulation time.
   * @param block Block index within the set.
   * @param deltat Time-step size.
   */
  void assembleRes(const size_t & set, const size_t & stage,
                   vector<vector_RCP> & sol,
                   vector<vector_RCP> & sol_stage,
                   vector<vector_RCP> & sol_prev,
                   vector<vector_RCP> & phi,
                   vector<vector_RCP> & phi_stage,
                   vector<vector_RCP> & phi_step,
                   vector_RCP & param_sol,
                   vector_RCP & param_dot,
                   vector_RCP & res, matrix_RCP & J, const bool & isTransient,
                   const ScalarT & current_time,
                   const int & block,
                   const ScalarT & deltat);
  
  /**
   * @brief Update time information in the workset for a given block.
   *
   * This sets the current time, time-step size, and transient flag in the workset
   * associated with the specified block.
   *
   * @param block Block index whose workset should be updated.
   * @param isTransient Whether the simulation is time-dependent.
   * @param current_time Current time of the simulation.
   * @param deltat Time-step size.
   */
  void updateWorksetTime(const size_t & block, const bool & isTransient,
                         const ScalarT & current_time, const ScalarT & deltat);
  
  /**
   * @brief Template version of workset time update.
   *
   * Updates the time-related metadata inside a provided workset object.
   *
   * @tparam EvalT Evaluation type.
   * @param wset Workset to update.
   * @param isTransient Flag indicating transient behavior.
   * @param current_time Current simulation time.
   * @param deltat Time-step size.
   */
  template<class EvalT>
  void updateWorksetTime(Teuchos::RCP<Workset<EvalT> > & wset, const bool & isTransient,
                         const ScalarT & current_time, const ScalarT & deltat);
  
  /**
   * @brief Update the adjoint mode flag for a given block.
   *
   * @param block Block index.
   * @param isAdjoint If true, the workset will be marked as operating in adjoint mode.
   */
  void updateWorksetAdjoint(const size_t & block, const bool & isAdjoint);
  
  /**
   * @brief Template version of adjoint-flag update.
   *
   * @tparam EvalT Evaluation type.
   * @param wset Workset whose adjoint flag is being updated.
   * @param isAdjoint New adjoint-mode value.
   */
  template<class EvalT>
  void updateWorksetAdjoint(Teuchos::RCP<Workset<EvalT> > & wset, const bool & isAdjoint);
  
  /**
   * @brief Update the element ID for the specified block.
   *
   * @param block Block index.
   * @param eid Element ID to set in the workset.
   */
  void updateWorksetEID(const size_t & block, const size_t & eid);
  
  /**
   * @brief Template version of element ID update.
   *
   * @tparam EvalT Evaluation type.
   * @param wset Workset whose element ID should be updated.
   * @param eid Element ID to assign.
   */
  template<class EvalT>
  void updateWorksetEID(Teuchos::RCP<Workset<EvalT> > & wset, const size_t & eid);
  
  /**
   * @brief Update whether the workset is operating on a boundary side.
   *
   * @param block Block index for which to update the flag.
   * @param on_side True if the workset corresponds to a side evaluation.
   */
  void updateWorksetOnSide(const size_t & block, const bool & on_side);
  
  /**
   * @brief Template version of side-flag update.
   *
   * @tparam EvalT Evaluation type.
   * @param wset Workset to update.
   * @param on_side Side-evaluation flag.
   */
  template<class EvalT>
  void updateWorksetOnSide(Teuchos::RCP<Workset<EvalT> > & wset, const bool & on_side);
  
  /**
   * @brief Update the residual-related metadata in the workset.
   *
   * @param block Block index.
   */
  void updateWorksetResidual(const size_t & block);
  
  /**
   * @brief Template version of residual metadata update.
   *
   * @tparam EvalT Evaluation type.
   * @param wset Workset whose residual fields should be updated.
   */
  template<class EvalT>
  void updateWorksetResidual(Teuchos::RCP<Workset<EvalT> > & wset);
  
  /**
   * @brief Apply degrees-of-freedom constraints to the residual and Jacobian.
   *
   * This may enforce boundary conditions, Dirichlet conditions, or other constraint
   * structures depending on the physics set.
   *
   * @param set Physics set index.
   * @param J Jacobian matrix to modify if constraints affect derivatives.
   * @param res Residual vector to enforce constraints on.
   * @param current_time Current time.
   * @param compute_jacobian Whether the Jacobian should be modified.
   * @param compute_disc_sens Whether discretization sensitivity information is required.
   */
  void dofConstraints(const size_t & set, matrix_RCP & J, vector_RCP & res,
                      const ScalarT & current_time,
                      const bool & compute_jacobian,
                      const bool & compute_disc_sens);
  
  /**
   * @brief Reset the previous solution vector for the specified set.
   *
   * @param set Physics set index.
   */
  void resetPrevSoln(const size_t & set);
  
  /**
   * @brief Revert the current solution for the given set to a previous stored state.
   *
   * @param set Physics set index.
   */
  void revertSoln(const size_t & set);
  
  /**
   * @brief Reset stage-level solution fields for multi-stage integrators.
   *
   * @param set Physics set index.
   */
  void resetStageSoln(const size_t & set);
  
  /**
   * @brief Update stage-related time information for multi-stage methods.
   *
   * @param stage Stage index.
   * @param current_time Current simulation time.
   * @param deltat Time-step size.
   */
  void updateStage(const int & stage, const ScalarT & current_time,
                   const ScalarT & deltat);
  
  /**
   * @brief Template version of stage update routine.
   *
   * @tparam EvalT Evaluation type.
   * @param wset Workset to update.
   * @param stage Stage index.
   * @param current_time Current simulation time.
   * @param deltat Time-step size.
   */
  template<class EvalT>
  void updateStage(Teuchos::RCP<Workset<EvalT> > & wset, const int & stage,
                   const ScalarT & current_time, const ScalarT & deltat);
  
  /**
   * @brief Update stage solution fields for the given set.
   *
   * @param set Physics set index.
   */
  void updateStageSoln(const size_t & set);
  
  /**
   * @brief Update the physics set being operated on by the assembly routines.
   *
   * @param set Physics set index.
   */
  void updatePhysicsSet(const size_t & set);
  
  /**
   * @brief Update the internal time-step counter.
   *
   * @param timestep New time-step index.
   */
  void updateTimeStep(const int & timestep);
  
  /**
   * @brief Set Butcher tableau data in the workset for a given set and block.
   *
   * This routine stores the Runge–Kutta Butcher coefficients (A, b, c) into the
   * workset so that time-integration evaluations can access stage-coupling
   * information. This version targets a specific block belonging to the physics set.
   *
   * @param set Physics set index.
   * @param block Block index associated with the workset.
   * @param butcher_A Two-dimensional Kokkos view of the Butcher A matrix.
   * @param butcher_b One-dimensional Kokkos view of the Butcher b coefficients.
   * @param butcher_c One-dimensional Kokkos view of the Butcher c coefficients.
   */
  void setWorksetButcher(const size_t & set, const size_t & block,
                         Kokkos::View<ScalarT**,AssemblyDevice> butcher_A,
                         Kokkos::View<ScalarT*,AssemblyDevice> butcher_b,
                         Kokkos::View<ScalarT*,AssemblyDevice> butcher_c);
  
  /**
   * @brief Template version of setWorksetButcher, operating directly on a workset object.
   *
   * @tparam EvalT Evaluation type.
   * @param set Physics set index.
   * @param wset Workset object to update.
   * @param butcher_A Butcher A matrix (RK stage coupling).
   * @param butcher_b Butcher b vector (RK combination weights).
   * @param butcher_c Butcher c vector (RK time offsets).
   */
  template<class EvalT>
  void setWorksetButcher(const size_t & set, Teuchos::RCP<Workset<EvalT> > & wset,
                         Kokkos::View<ScalarT**,AssemblyDevice> butcher_A,
                         Kokkos::View<ScalarT*,AssemblyDevice> butcher_b,
                         Kokkos::View<ScalarT*,AssemblyDevice> butcher_c);
  
  /**
   * @brief Set BDF (Backward Differentiation Formula) weights in the workset.
   *
   * @param set Physics set index.
   * @param block Block index within the set.
   * @param BDF_wts Vector of BDF weights (order determined externally).
   */
  void setWorksetBDF(const size_t & set, const size_t & block,
                     Kokkos::View<ScalarT*,AssemblyDevice> BDF_wts);
  
  /**
   * @brief Template version of setWorksetBDF, writing weights directly to a workset.
   *
   * @tparam EvalT Evaluation type.
   * @param set Physics set index.
   * @param wset Workset to update.
   * @param BDF_wts BDF weight coefficients.
   */
  template<class EvalT>
  void setWorksetBDF(const size_t & set, Teuchos::RCP<Workset<EvalT> > & wset,
                     Kokkos::View<ScalarT*,AssemblyDevice> BDF_wts);
  
  /**
   * @brief Perform gather operation for a given block/group and a vector of device views.
   *
   * @tparam ViewType Kokkos view type.
   * @param block Block index.
   * @param grp Group index within the block.
   * @param vec Collection of views to gather from.
   * @param type Field type indicator.
   */
  template<class ViewType>
  void performGather(const size_t & block, const size_t & grp,
                     const vector<ViewType> & vec, const int & type);
  
  /**
   * @brief Perform gather for a single device view with a specified local entry.
   *
   * @tparam ViewType Kokkos view type.
   * @param set Physics set index.
   * @param block Block index.
   * @param grp Group index.
   * @param vec_dev Device view containing the data to gather.
   * @param type Field type indicator.
   * @param local_entry Local index inside the group.
   */
  template<class ViewType>
  void performGather(const size_t & set, const size_t & block, const size_t & grp,
                     ViewType vec_dev, const int & type, const size_t & local_entry);
  
  /**
   * @brief Perform 4-D gather for a multi-dimensional field.
   *
   * @tparam ViewType Kokkos view type, rank-4.
   * @param set Physics set index.
   * @param block Block index.
   * @param grp Group index.
   * @param vec_dev Device view for multi-dimensional gather.
   * @param type Field type indicator.
   * @param local_entry Local entry index.
   */
  template<class ViewType>
  void performGather4D(const size_t & set, const size_t & block, const size_t & grp,
                       ViewType vec_dev, const int & type, const size_t & local_entry);
  
  /**
   * @brief Specialization of performGather for standard vector_RCP.
   *
   * @param set Physics set index.
   * @param block Block index.
   * @param grp Group index.
   * @param vec Vector holding host-side or device-side data.
   * @param type Field type indicator.
   * @param local_entry Local element index.
   */
  void performGather(const size_t & set, const size_t & block, const size_t & grp,
                     vector_RCP & vec, const int & type, const size_t & local_entry);
  
  /**
   * @brief Full gather routine for solution, basis, and parameter fields.
   *
   * @tparam ViewType Kokkos view type used to store gathered data.
   * @param current_set Physics set index.
   * @param block Block index.
   * @param grp Group index.
   * @param include_adjoint Whether adjoint variables should be gathered.
   * @param stage Stage index (for RK or multi-stage schemes).
   * @param use_only_sol Whether phi/basis values should be skipped.
   * @param sol Solution fields.
   * @param sol_stage Stage-level solutions.
   * @param sol_prev Previous time-step solutions.
   * @param phi Basis evaluations.
   * @param phi_stage Stage-level basis.
   * @param phi_prev Previous-step basis.
   * @param params Parameter field views.
   * @param param_dot Parameter time derivatives.
   */
  template<class ViewType>
  void performGather(const size_t & current_set, const size_t & block, const size_t & grp,
                     const bool & include_adjoint, const size_t & stage, const bool & use_only_sol,
                     vector<ViewType> & sol, vector<ViewType> & sol_stage, vector<ViewType> & sol_prev,
                     vector<ViewType> & phi, vector<ViewType> & phi_stage, vector<ViewType> & phi_prev,
                     vector<ViewType> & params, vector<ViewType> & param_dot);
  
  /**
   * @brief Perform gather for boundary evaluations on a single device vector.
   *
   * @tparam ViewType Kokkos view type.
   * @param set Physics set index.
   * @param vec_dev Device view containing boundary data.
   * @param type Field type indicator.
   */
  template<class ViewType>
  void performBoundaryGather(const size_t & set, ViewType vec_dev, const int & type);
  
  /**
   * @brief Full-field boundary gather, including solution, basis, and parameter views.
   *
   * @tparam ViewType Kokkos view type.
   * @param current_set Physics set index.
   * @param block Block index.
   * @param grp Group index.
   * @param include_adjoint Whether adjoint values are included.
   * @param stage Stage index.
   * @param use_only_sol Whether basis fields should be ignored.
   * @param sol Solution fields.
   * @param sol_stage Stage solution.
   * @param sol_prev Previous solution.
   * @param phi Basis functions.
   * @param phi_stage Stage basis.
   * @param phi_prev Previous basis.
   * @param params Parameter fields.
   * @param param_dot Parameter derivatives.
   */
  template<class ViewType>
  void performBoundaryGather(const size_t & current_set, const size_t & block, const size_t & grp,
                             const bool & include_adjoint, const size_t & stage, const bool & use_only_sol,
                             vector<ViewType> & sol, vector<ViewType> & sol_stage, vector<ViewType> & sol_prev,
                             vector<ViewType> & phi, vector<ViewType> & phi_stage, vector<ViewType> & phi_prev,
                             vector<ViewType> & params, vector<ViewType> & param_dot);
  
  /**
   * @brief Boundary gather for a single multi-dimensional field entry.
   *
   * @tparam ViewType Kokkos view type.
   * @param set Physics set index.
   * @param block Block index.
   * @param grp Group index.
   * @param vec_dev Device data to gather.
   * @param type Field type indicator.
   * @param local_entry Local index.
   */
  template<class ViewType>
  void performBoundaryGather(const size_t & set, const size_t & block, const size_t & grp,
                             ViewType vec_dev, const int & type, const size_t & local_entry);
  
  /**
   * @brief Perform a 4-D boundary gather operation.
   *
   * @tparam ViewType Kokkos view type, rank-4.
   * @param set Physics set index.
   * @param block Block index.
   * @param grp Group index.
   * @param vec_dev Multi-dimensional data.
   * @param type Field type identifier.
   * @param local_entry Local index within 4-D tensor.
   */
  template<class ViewType>
  void performBoundaryGather4D(const size_t & set, const size_t & block, const size_t & grp,
                               ViewType vec_dev, const int & type, const size_t & local_entry);
  
  /**
   * @brief Scatter the local Jacobian into the global CRS matrix.
   *
   * @tparam MatType CRS matrix type on device.
   * @tparam LocalViewType Local element-level matrix view.
   * @tparam LIDViewType View storing element DOF local IDs.
   * @param set Physics set index.
   * @param J_kcrs Global CRS Jacobian.
   * @param local_J Local element Jacobian.
   * @param LIDs Local DOF indices.
   * @param paramLIDs Local parameter DOF indices.
   * @param compute_disc_sens Whether discretization sensitivity contributions should be inserted.
   */
  template<class MatType, class LocalViewType, class LIDViewType>
  void scatterJac(const size_t & set, MatType J_kcrs, LocalViewType local_J,
                  LIDViewType LIDs, LIDViewType paramLIDs,
                  const bool & compute_disc_sens);
  
  /**
   * @brief Scatter residual contributions from a local workset to the global vector.
   *
   * @tparam VecViewType Vector view type.
   * @tparam LocalViewType Local residual view.
   * @tparam LIDViewType Local DOF index view.
   * @param res_view Global residual view.
   * @param local_res Local residual contributions.
   * @param LIDs Local DOF IDs.
   */
  template<class VecViewType, class LocalViewType, class LIDViewType>
  void scatterRes(VecViewType res_view, LocalViewType local_res, LIDViewType LIDs);
  
  /**
   * @brief Scatter Jacobian and residual contributions for a workset.
   *
   * This version does not take an explicit workset pointer and updates based on
   * the given set and block.
   *
   * @tparam MatType CRS matrix type.
   * @tparam VecViewType Vector view type.
   * @tparam LIDViewType Local ID type.
   * @tparam EvalT Evaluation type.
   *
   * @param set Physics set index.
   * @param J_kcrs Global Jacobian.
   * @param res_view Global residual.
   * @param LIDs Local DOF IDs.
   * @param paramLIDs Parameter DOF IDs.
   * @param block Block index.
   * @param compute_jacobian Whether to scatter Jacobian data.
   * @param compute_sens Whether to include sensitivity contributions.
   * @param compute_disc_sens Whether to insert discretization sensitivities.
   * @param isAdjoint Whether adjoint mode is active.
   * @param dummyval Dummy evaluation object for template resolution.
   */
  template<class MatType, class VecViewType, class LIDViewType, class EvalT>
  void scatter(const size_t & set, MatType J_kcrs, VecViewType res_view,
               LIDViewType LIDs, LIDViewType paramLIDs,
               const int & block,
               const bool & compute_jacobian,
               const bool & compute_sens,
               const bool & compute_disc_sens,
               const bool & isAdjoint,
               EvalT & dummyval);
  
  /**
   * @brief Scatter Jacobian and residual contributions using a provided workset.
   *
   * @tparam MatType CRS matrix type.
   * @tparam VecViewType Vector view type.
   * @tparam LIDViewType DOF index type.
   * @tparam EvalT Evaluation type.
   *
   * @param wset Workset providing local data.
   * @param set Physics set index.
   * @param J_kcrs Global Jacobian matrix.
   * @param res_view Global residual vector.
   * @param LIDs Local DOF indices.
   * @param paramLIDs Parameter DOF indices.
   * @param block Block index.
   * @param compute_jacobian Whether Jacobian entries are scattered.
   * @param compute_sens Whether sensitivity terms are included.
   * @param compute_disc_sens Whether discretization sensitivity is included.
   * @param isAdjoint Whether adjoint assembly mode is used.
   */
  template<class MatType, class VecViewType, class LIDViewType, class EvalT>
  void scatter(Teuchos::RCP<Workset<EvalT> > & wset, const size_t & set,
               MatType J_kcrs, VecViewType res_view,
               LIDViewType LIDs, LIDViewType paramLIDs,
               const int & block,
               const bool & compute_jacobian,
               const bool & compute_sens,
               const bool & compute_disc_sens,
               const bool & isAdjoint);
  
  // ========================================================================================
  // ========================================================================================
  /**
   * @brief Scatter the residual vector values into the global residual structure.
   *
   * This routine inserts or accumulates local residual contributions into the
   * global residual vector for the specified set and block. It maps local degrees
   * of freedom (DOFs), given in @p LIDs, to entries in the global residual
   * vector @p res_view. This version includes block information and is intended
   * for use in multiphysics or multi-block assembly contexts.
   *
   * @tparam VecViewType  Type of the Kokkos or device view containing the global residual.
   * @tparam LIDViewType  Type of the view containing local IDs (LIDs) mapping local entries to global DOFs.
   *
   * @param set           Physics set index being assembled.
   * @param res_view      View of the global residual vector into which values will be scattered.
   * @param LIDs          Local ID list mapping each entry of the local residual vector to a global DOF.
   * @param block         Block index corresponding to the physics block or element block involved.
   *
   * @return void
   */
  template<class VecViewType, class LIDViewType>
  void scatterRes(const size_t & set, VecViewType res_view,
                  LIDViewType LIDs, const int & block);
  
  /**
   * @brief Apply the mass matrix action in a matrix-free manner.
   *
   * This function computes y = M * x, where M is the mass matrix. Instead of
   * forming the mass matrix explicitly, it invokes the underlying mass operator
   * within the element computations. This approach reduces memory consumption and
   * is often more efficient for large-scale, high-order finite element problems.
   *
   * @param set   The physics set for which the mass matrix-free action is computed.
   * @param x     Input vector to be multiplied by the mass operator.
   * @param y     Output vector storing the result M * x.
   *
   * @return void
   */
  void applyMassMatrixFree(const size_t & set, const vector_RCP & x, vector_RCP & y);
  
  /**
   * @brief Build element-level or block-level volumetric data structures.
   *
   * Constructs data structures required for volumetric integration on a specific
   * block. This may include connectivity tables, Jacobian data, or any
   * precomputed geometric/physical information required by the assembly engine.
   *
   * @param block    The block index for which to build the volumetric database.
   *
   * @return void
   */
  void buildDatabase(const size_t & block);
  
  /**
   * @brief Write volumetric data for postprocessing or debugging.
   *
   * This routine outputs volumetric information for the specified block, making
   * use of provided orientation data. It may write to files or internal buffers
   * depending on implementation. Useful for visualization, diagnostics, or
   * verification.
   *
   * @param block         Block index for which volumetric data is written.
   * @param all_orients   Array containing orientation information for each volume element.
   *
   * @return void
   */
  void writeVolumetricData(const size_t & block, vector<vector<size_t>> & all_orients);
  
  /**
   * @brief Identify the first user of each volumetric data entry.
   *
   * Analyzes volumetric database usage to determine which element or component
   * first references a specific data record. This is useful for minimizing memory
   * duplication by sharing entries between users.
   *
   * @param block        Block index being analyzed.
   * @param first_users  Vector of (element, index) pairs indicating the first user of each record.
   *
   * @return void
   */
  void identifyVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users);
  
  /**
   * @brief Identify the first user of each boundary-related data record.
   *
   * Similar to identifyVolumetricDatabase(), but specific to boundary/interface
   * data structures required for boundary integrals.
   *
   * @param block                   Block index being analyzed.
   * @param first_boundary_users    Output list of first-user pairs for each boundary record.
   *
   * @return void
   */
  void identifyBoundaryDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_boundary_users);
  
  /**
   * @brief Identify first users of volumetric integration-point (IP) data.
   *
   * Performs user-identification separately for x-, y-, and z-components of
   * integration-point level data, enabling reuse and memory optimization.
   *
   * @param block         Block index.
   * @param first_users_x Output list of first users for x-IP data.
   * @param first_users_y Output list of first users for y-IP data.
   * @param first_users_z Output list of first users for z-IP data.
   *
   * @return void
   */
  void identifyVolumetricIPDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users_x,
                                    vector<std::pair<size_t,size_t> > & first_users_y,
                                    vector<std::pair<size_t,size_t> > & first_users_z);
  
  /**
   * @brief Build volumetric database structures according to previously identified first users.
   *
   * Allocates and constructs volumetric data entries, assigning ownership to the
   * first users identified by identifyVolumetricDatabase().
   *
   * @param block        Block index.
   * @param first_users  List of first users for each volumetric record.
   *
   * @return void
   */
  void buildVolumetricDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users);
  
  /**
   * @brief Build the boundary database using previously identified first boundary users.
   *
   * Constructs all boundary-related precomputed structures required for surface
   * integrals (e.g., normals, tangent vectors, boundary Jacobians).
   *
   * @param block                   Block index.
   * @param first_boundary_users    List of first users for boundary records.
   *
   * @return void
   */
  void buildBoundaryDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_boundary_users);
  
  /**
   * @brief Build the volumetric integration-point (IP) database.
   *
   * Using the previously identified first users for x-, y-, and z-data, this
   * routine builds all integration point–level volumetric structures required by
   * physics kernels and assembly procedures.
   *
   * @param block         Block index.
   * @param first_users_x First users for x-IP data.
   * @param first_users_y First users for y-IP data.
   * @param first_users_z First users for z-IP data.
   *
   * @return void
   */
  void buildVolumetricIPDatabase(const size_t & block, vector<std::pair<size_t,size_t> > & first_users_x,
                                 vector<std::pair<size_t,size_t> > & first_users_y,
                                 vector<std::pair<size_t,size_t> > & first_users_z);
  
  /**
   * @brief Finalize all function-related data structures.
   *
   * Called after construction of the necessary databases but before assembly
   * begins. Typically used to finalize expression parsing, precompute constants,
   * or validate function dependencies.
   *
   * @return void
   */
  void finalizeFunctions();
  
  /**
   * @brief Template version of finalizeFunctions().
   *
   * Finalizes user-defined functions stored in @p fman and binds them to the
   * appropriate workset data structures. Ensures all function evaluation
   * dependencies are satisfied before assembly.
   *
   * @tparam EvalT  Evaluation type (Residual, Jacobian, Tangent, etc.).
   *
   * @param fman   Function manager responsible for parsing and evaluating user-defined functions.
   * @param wset   Workset associated with a physics block.
   *
   * @return void
   */
  template<class EvalT>
  void finalizeFunctions(Teuchos::RCP<FunctionManager<EvalT> > & fman, Teuchos::RCP<Workset<EvalT> > & wset);
  
  /**
   * @brief Compute flux and optional sensitivity with respect to parameters.
   *
   * Wrapper for computeFluxEvalT(), automatically providing a dummy AD value for
   * type resolution. Computes the flux across a control surface or geometric
   * face, using state, gradient, and parameter values at integration points.
   *
   * @tparam ViewType  Type of the Kokkos view for u, du, and dp.
   *
   * @param block            Block index.
   * @param grp              Group (element group) index.
   * @param u_kv             State value view at quadrature points.
   * @param du_kv            Gradient of state variable.
   * @param dp_kv            Parameter derivative or sensitivity vector.
   * @param lambda           Weighting vector or adjoint multiplier.
   * @param time             Current simulation time.
   * @param side             Index of the element side being evaluated.
   * @param coarse_h         Characteristic length scale for stabilization or metric computation.
   * @param compute_sens     Whether sensitivity with respect to parameters should be computed.
   * @param fluxwt           Flux scaling factor or weighting.
   * @param useTransientSol  Whether transient (time-dependent) solution values should be used.
   *
   * @return void
   */
  template<class ViewType>
  void computeFlux(const int & block, const int & grp, ViewType u_kv,
                   ViewType du_kv, ViewType dp_kv, View_Sc3 lambda,
                   const ScalarT & time, const int & side, const ScalarT & coarse_h,
                   const bool & compute_sens, const ScalarT & fluxwt,
                   bool & useTransientSol) {
    
    AD dummyval = 0.0;
    this->computeFluxEvalT(dummyval, block, grp, u_kv, du_kv, dp_kv, lambda, time, side, coarse_h, compute_sens, fluxwt, useTransientSol);
    
  }
  
  /**
   * @brief Evaluation-type–templated flux computation routine.
   *
   * This is the underlying implementation of computeFlux(), templated on EvalT
   * so that automatic differentiation, sensitivities, and other evaluation modes
   * can be employed consistently. Computes flux values at quadrature points and,
   * if requested, their sensitivities with respect to parameters.
   *
   * @tparam ViewType  Kokkos or device view type used for field data.
   * @tparam EvalT     Evaluation type (Residual, Jacobian, etc.).
   *
   * @param dummyval       Dummy variable of type EvalT used to differentiate overloads.
   * @param block          Block index.
   * @param grp            Group (element group) index.
   * @param u_kv           State value view.
   * @param du_kv          State gradient view.
   * @param dp_kv          Parameter derivative view.
   * @param lambda         Weighting vector.
   * @param time           Current simulation time.
   * @param side           Side index of the element.
   * @param coarse_h       Characteristic length scale.
   * @param compute_sens   Whether to compute parameter sensitivity.
   * @param fluxwt         Weighting applied to flux.
   * @param useTransientSol Whether transient solution values should be employed.
   *
   * @return void
   */
  template<class ViewType, class EvalT>
  void computeFluxEvalT(EvalT & dummyval, const int & block, const int & grp,
                        ViewType u_kv, ViewType du_kv, ViewType dp_kv, View_Sc3 lambda,
                        const ScalarT & time, const int & side, const ScalarT & coarse_h,
                        const bool & compute_sens, const ScalarT & fluxwt,
                        bool & useTransientSol) {
  
#ifndef MrHyDE_NO_AD
  typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_AD2;
  int wkblock = 0;
  
  wkset_AD[wkblock]->setTime(time);
  wkset_AD[wkblock]->sidename = boundary_groups[block][grp]->sidename;
  wkset_AD[wkblock]->currentside = boundary_groups[block][grp]->sidenum;
  wkset_AD[wkblock]->numElem = boundary_groups[block][grp]->numElem;
  
  // Currently hard coded to one physics sets
  int set = 0;
  
  vector<View_AD2> sol_vals = wkset_AD[wkblock]->sol_vals;
  //auto param_AD = wkset_AD->pvals;
  //auto ulocal = groupData[block]->sol[set];
  auto ulocal = boundary_groups[block][grp]->sol[set];
  auto currLIDs = boundary_groups[block][grp]->LIDs[set];
  
  if (useTransientSol) {
    int stage = wkset_AD[wkblock]->current_stage;
    auto b_A = wkset_AD[wkblock]->butcher_A;
    auto b_b = wkset_AD[wkblock]->butcher_b;
    auto BDF = wkset_AD[wkblock]->BDF_wts;
    
    ScalarT one = 1.0;
    
    for (size_type var=0; var<ulocal.extent(1); var++ ) {
      size_t uindex = wkset_AD[wkblock]->sol_vals_index[set][var];
      auto u_AD = sol_vals[uindex];
      auto off = subview(wkset_AD[wkblock]->set_offsets[set],var,ALL());
      auto cu = subview(ulocal,ALL(),var,ALL());
      //auto cu_prev = subview(groupData[block]->sol_prev[set],ALL(),var,ALL(),ALL());
      //auto cu_stage = subview(groupData[block]->sol_stage[set],ALL(),var,ALL(),ALL());
      
      auto cu_prev = subview(boundary_groups[block][grp]->sol_prev[set],ALL(),var,ALL(),ALL());
      auto cu_stage = subview(boundary_groups[block][grp]->sol_stage[set],ALL(),var,ALL(),ALL());
      
      parallel_for("wkset transient sol seedwhat 1",
                   TeamPolicy<AssemblyExec>(currLIDs.extent(0), Kokkos::AUTO, VECTORSIZE),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        ScalarT beta_u;//, beta_t;
        ScalarT alpha_u = b_A(stage,stage)/b_b(stage);
        //ScalarT timewt = one/dt/b_b(stage);
        //ScalarT alpha_t = BDF(0)*timewt;
        
        for (size_type dof=team.team_rank(); dof<u_AD.extent(1); dof+=team.team_size() ) {
          
          // Seed the stage solution
          AD stageval = AD(MAXDERIVS,0,cu(elem,dof));
          for( size_t p=0; p<du_kv.extent(1); p++ ) {
            stageval.fastAccessDx(p) = fluxwt*du_kv(currLIDs(elem,off(dof)),p);
          }
          // Compute the evaluating solution
          beta_u = (one-alpha_u)*cu_prev(elem,dof,0);
          for (int s=0; s<stage; s++) {
            beta_u += b_A(stage,s)/b_b(s) * (cu_stage(elem,dof,s) - cu_prev(elem,dof,0));
          }
          u_AD(elem,dof) = alpha_u*stageval+beta_u;
          
          // Compute the time derivative
          //beta_t = zero;
          //for (size_type s=1; s<BDF.extent(0); s++) {
          //  beta_t += BDF(s)*cu_prev(elem,dof,s-1);
          //}
          //beta_t *= timewt;
          //u_dot_AD(elem,dof) = alpha_t*stageval + beta_t;
        }
        
      });
      
    }
  }
  else {
    //Teuchos::TimeMonitor localtimer(*fluxGatherTimer);
    
    if (compute_sens) {
      for (size_t var=0; var<ulocal.extent(1); var++) {
        auto u_AD = sol_vals[var];
        auto offsets = subview(wkset_AD[wkblock]->offsets,var,ALL());
        parallel_for("flux gather",
                     RangePolicy<AssemblyExec>(0,ulocal.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          for( size_t dof=0; dof<u_AD.extent(1); dof++ ) {
            u_AD(elem,dof) = AD(u_kv(currLIDs(elem,offsets(dof)),0));
          }
        });
      }
    }
    else {
      for (size_t var=0; var<ulocal.extent(1); var++) {
        auto u_AD = sol_vals[var];
        auto offsets = subview(wkset_AD[wkblock]->offsets,var,ALL());
        parallel_for("flux gather",
                     RangePolicy<AssemblyExec>(0,ulocal.extent(0)),
                     KOKKOS_LAMBDA (const int elem ) {
          for( size_t dof=0; dof<u_AD.extent(1); dof++ ) {
            u_AD(elem,dof) = AD(MAXDERIVS, 0, u_kv(currLIDs(elem,offsets(dof)),0));
            for( size_t p=0; p<du_kv.extent(1); p++ ) {
              u_AD(elem,dof).fastAccessDx(p) = du_kv(currLIDs(elem,offsets(dof)),p);
            }
          }
        });
      }
    }
  }
  
  {
    //Teuchos::TimeMonitor localtimer(*fluxWksetTimer);
    wkset_AD[wkblock]->computeSolnSideIP(boundary_groups[block][grp]->sidenum);//, u_AD, param_AD);
  }
  
  if (wkset_AD[wkblock]->numAux > 0) {
    
    // Teuchos::TimeMonitor localtimer(*fluxAuxTimer);
    
    auto numAuxDOF = groupData[wkblock]->num_aux_dof;
    
    for (size_type var=0; var<numAuxDOF.extent(0); var++) {
      auto abasis = boundary_groups[block][grp]->auxside_basis[boundary_groups[block][grp]->auxusebasis[var]];
      auto off = subview(boundary_groups[block][grp]->auxoffsets,var,ALL());
      string varname = wkset_AD[wkblock]->aux_varlist[var];
      auto local_aux = wkset_AD[wkblock]->getSolutionField("aux "+varname,false);
      Kokkos::deep_copy(local_aux,0.0);
      //auto local_aux = Kokkos::subview(wkset_AD->local_aux_side,Kokkos::ALL(),var,Kokkos::ALL(),0);
      auto localID = boundary_groups[block][grp]->localElemID;
      auto varaux = subview(lambda,ALL(),var,ALL());
      parallel_for("flux aux",
                   RangePolicy<AssemblyExec>(0,localID.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type dof=0; dof<abasis.extent(1); ++dof) {
          AD auxval = AD(MAXDERIVS,off(dof), varaux(localID(elem),dof));
          auxval.fastAccessDx(off(dof)) *= fluxwt;
          for (size_type pt=0; pt<abasis.extent(2); ++pt) {
            local_aux(elem,pt) += auxval*abasis(elem,dof,pt);
          }
        }
      });
    }
    
  }
  
  {
    //Teuchos::TimeMonitor localtimer(*fluxEvalTimer);
    physics->computeFlux<AD>(0,groupData[block]->my_block);
  }
#endif
  //wkset_AD->isOnSide = false;
}

////////////////////////////////////////////////////////////////////////////////
// Functionality moved from boundary groups into here
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Compute the boundary contributions to the Jacobian and residual.
 *
 * This routine evaluates both Jacobian and residual terms associated with
 * boundary integrals for a given block and group. Depending on flags, it may
 * compute Jacobian contributions, sensitivities with respect to parameters,
 * adjoint terms, or discontinuous sensitivities. The routine fills the
 * element-local residual and Jacobian arrays @p local_res and @p local_J.
 *
 * @param block              Block index for the boundary being evaluated.
 * @param grp                Group or element-group index within the block.
 * @param time               Current simulation time.
 * @param isTransient        Whether time-dependent effects should be included.
 * @param isAdjoint          Whether adjoint (reverse-mode) terms should be used.
 * @param compute_jacobian   Flag indicating whether Jacobian entries should be computed.
 * @param compute_sens       Whether sensitivities with respect to model parameters are computed.
 * @param num_active_params  Number of active parameters involved in sensitivity computation.
 * @param compute_disc_sens  Whether discontinuous sensitivities are computed.
 * @param compute_aux_sens   Whether auxiliary sensitivities (e.g., flux-related) are computed.
 * @param store_adjPrev      Whether the previous-step adjoint solution should be stored.
 * @param local_res          Element-local residual array to be filled.
 * @param local_J            Element-local Jacobian array to be filled.
 *
 * @return void
 */
void computeJacResBoundary(const int & block, const size_t & grp,
                           const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                           const bool & compute_jacobian, const bool & compute_sens,
                           const int & num_active_params, const bool & compute_disc_sens,
                           const bool & compute_aux_sens, const bool & store_adjPrev,
                           View_Sc3 local_res, View_Sc3 local_J);

// ========================================================================================

/**
 * @brief Update all boundary-related workset fields (ScalarT version).
 *
 * Selects the appropriate workset for the given block and group, and updates
 * its boundary-related data (states, gradients, parameters, basis data, etc.).
 * Uses ScalarT as the evaluation type, meaning no AD capabilities are used.
 *
 * @param block              Block index.
 * @param grp                Group or element-group index.
 * @param seedwhat           Identifier for what quantity is seeded (state, parameter, etc.).
 * @param seedindex          Optional index of the item being seeded (default = 0).
 * @param override_transient If true, overrides transient settings in the workset.
 *
 * @return void
 */
void updateWorksetBoundary(const int & block, const size_t & grp, const int & seedwhat,
                           const int & seedindex=0, const bool & override_transient=false);

// ========================================================================================

/**
 * @brief Update boundary-related workset fields using AD (automatic differentiation).
 *
 * Same as updateWorksetBoundary(), but uses AD evaluation types, enabling
 * Jacobian and sensitivity computation.
 *
 * @param block              Block index.
 * @param grp                Group index.
 * @param seedwhat           Quantity to seed in the workset.
 * @param seedindex          Optional seed index.
 * @param override_transient Whether to override transient information.
 *
 * @return void
 */
void updateWorksetBoundaryAD(const int & block, const size_t & grp, const int & seedwhat,
                             const int & seedindex=0, const bool & override_transient=false);

// ========================================================================================

/**
 * @brief Partially templated workset boundary update.
 *
 * Selects the correct workset corresponding to @p EvalT and invokes the fully
 * templated implementation of updateWorksetBoundary(). Allows dispatch to
 * Residual, Jacobian, Tangent, or other evaluation types.
 *
 * @tparam EvalT    Evaluation type.
 *
 * @param block              Block index.
 * @param grp                Group index.
 * @param seedwhat           Quantity to seed.
 * @param seedindex          Optional seed index.
 * @param override_transient Whether transient data is overridden.
 *
 * @return void
 */
template<class EvalT>
void updateWorksetBoundary(const int & block, const size_t & grp, const int & seedwhat,
                           const int & seedindex=0, const bool & override_transient=false);

// ========================================================================================

/**
 * @brief Fully templated boundary workset updater.
 *
 * Performs the full update of boundary workset fields for the specified
 * evaluation type @p EvalT. This includes updating dependent fields such as
 * solution values, basis data, gradients, parameters, and time information.
 *
 * @tparam EvalT    Evaluation type (Residual, Jacobian, etc.).
 *
 * @param wset               Workset instance to update.
 * @param block              Block index.
 * @param grp                Group index.
 * @param seedwhat           Identifier of quantity to seed.
 * @param seedindex          Index of the seeded value.
 * @param override_transient Whether transient settings should be forced.
 *
 * @return void
 */
template<class EvalT>
void updateWorksetBoundary(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp,
                           const int & seedwhat, const int & seedindex,
                           const bool & override_transient);

// ========================================================================================

/**
 * @brief Compute auxiliary boundary quantities (ScalarT version).
 *
 * Evaluates additional auxiliary fields tied to boundary integrals, such as
 * stabilization terms, surrogate variables, or flux-like quantities.
 *
 * @param block      Block index.
 * @param grp        Group index.
 * @param seedwhat   Type of auxiliary quantity to compute.
 *
 * @return void
 */
void computeBoundaryAux(const int & block, const size_t & grp, const int & seedwhat);

// ========================================================================================

/**
 * @brief Templated version of auxiliary boundary computation.
 *
 * Invokes the full auxiliary computation using the evaluation type @p EvalT.
 *
 * @tparam EvalT   Evaluation type.
 *
 * @param block      Block index.
 * @param grp        Group index.
 * @param seedwhat   Auxiliary quantity identifier.
 * @param wset       Workset for the evaluation.
 *
 * @return void
 */
template<class EvalT>
void computeBoundaryAux(const int & block, const size_t & grp, const int & seedwhat,
                        Teuchos::RCP<Workset<EvalT> > & wset);

// ========================================================================================

/**
 * @brief Backward-compatible boundary data update.
 *
 * Calls the fully templated version using AD evaluation types.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return void
 */
void updateDataBoundary(const int & block, const size_t & grp);

// ========================================================================================

/**
 * @brief Explicit AD version of updateDataBoundary().
 *
 * Updates all boundary-dependent data fields using AD types.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return void
 */
void updateDataBoundaryAD(const int & block, const size_t & grp);

// ========================================================================================

/**
 * @brief Templated boundary data update wrapper.
 *
 * Selects the correct evaluation type and calls the fully templated implementation.
 *
 * @tparam EvalT  Evaluation type.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return void
 */
template<class EvalT>
void updateDataBoundary(const int & block, const size_t & grp);

// ========================================================================================

/**
 * @brief Fully templated boundary data update.
 *
 * Performs all boundary-related updates needed by kernels, including geometry,
 * basis, parameters, and time data.
 *
 * @tparam EvalT   Evaluation type.
 *
 * @param wset     Workset to update.
 * @param block    Block index.
 * @param grp      Group index.
 *
 * @return void
 */
template<class EvalT>
void updateDataBoundary(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp);

// ========================================================================================

/**
 * @brief Update boundary basis data (ScalarT version).
 *
 * Updates all basis functions, gradients, and Jacobian determinants required
 * for boundary integrals. Uses ScalarT evaluation type.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return void
 */
void updateWorksetBasisBoundary(const int & block, const size_t & grp);

// ========================================================================================

/**
 * @brief AD version of boundary basis update.
 *
 * Same as updateWorksetBasisBoundary(), but uses AD-enabled evaluation types.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return void
 */
void updateWorksetBasisBoundaryAD(const int & block, const size_t & grp);

// ========================================================================================

/**
 * @brief Templated boundary basis update wrapper.
 *
 * Selects correct evaluation type and calls full implementation.
 *
 * @tparam EvalT  Evaluation type.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return void
 */
template<class EvalT>
void updateWorksetBasisBoundary(const int & block, const size_t & grp);

// ========================================================================================

/**
 * @brief Fully templated boundary basis update.
 *
 * Computes basis functions, gradients, and surface Jacobians for boundary
 * integrals for the specified workset.
 *
 * @tparam EvalT   Evaluation type.
 *
 * @param wset     Workset to update.
 * @param block    Block index.
 * @param grp      Group index.
 *
 * @return void
 */
template<class EvalT>
void updateWorksetBasisBoundary(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp);

// ========================================================================================

/**
 * @brief Update the boundary residual contributions.
 *
 * Computes and fills the local residual for boundary terms for the given block
 * and group. Sensitivity computation may be enabled.
 *
 * @param block          Block index.
 * @param grp            Group index.
 * @param compute_sens   Whether parameter sensitivities should be computed.
 * @param local_res      Local boundary residual storage.
 *
 * @return void
 */
void updateResBoundary(const int & block, const size_t & grp,
                       const bool & compute_sens, View_Sc3 local_res);

// ========================================================================================

/**
 * @brief Templated version of updateResBoundary().
 *
 * Performs evaluation using type @p EvalT.
 *
 * @tparam EvalT     Evaluation type.
 *
 * @param block          Block index.
 * @param grp            Group index.
 * @param compute_sens   Whether sensitivities are computed.
 * @param local_res      Local residual array.
 * @param wset           Workset for evaluation.
 *
 * @return void
 */
template<class EvalT>
void updateResBoundary(const int & block, const size_t & grp,
                       const bool & compute_sens, View_Sc3 local_res,
                       Teuchos::RCP<Workset<EvalT> > & wset);

// ========================================================================================

/**
 * @brief Update the boundary Jacobian contributions.
 *
 * Fills local Jacobian entries for boundary contributions. May include adjoint
 * terms depending on @p useadjoint.
 *
 * @param block        Block index.
 * @param grp          Group index.
 * @param useadjoint   Whether adjoint formulation is used.
 * @param local_J      Local Jacobian array to be filled.
 *
 * @return void
 */
void updateJacBoundary(const int & block, const size_t & grp,
                       const bool & useadjoint, View_Sc3 local_J);

// ========================================================================================

/**
 * @brief Templated version of updateJacBoundary().
 *
 * Uses evaluation type @p EvalT to compute Jacobian entries for boundary
 * integrals.
 *
 * @tparam EvalT   Evaluation type.
 *
 * @param block        Block index.
 * @param grp          Group index.
 * @param useadjoint   Whether an adjoint Jacobian is used.
 * @param local_J      Local Jacobian array.
 * @param wset         Workset used for evaluation.
 *
 * @return void
 */
template<class EvalT>
void updateJacBoundary(const int & block, const size_t & grp,
                       const bool & useadjoint, View_Sc3 local_J,
                       Teuchos::RCP<Workset<EvalT> > & wset);

// ========================================================================================
// ========================================================================================

/**
 * @brief Compute parameter-related contributions to the boundary Jacobian.
 *
 * This routine evaluates Jacobian terms that arise from parameters appearing
 * in boundary conditions or boundary integrals. These contributions may stem
 * from material parameters, model coefficients, or user-defined boundary
 * functions. Results are written directly into @p local_J.
 *
 * @param block     Block index associated with the boundary region.
 * @param grp       Group (element group) index inside the block.
 * @param local_J   Local Jacobian storage to be filled with parameter
 *                  derivative contributions.
 *
 * @return void
 */
void updateParamJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J);

// ========================================================================================

/**
 * @brief Templated version of updateParamJacBoundary().
 *
 * Performs the boundary parameter Jacobian computation using the evaluation
 * type @p EvalT (Residual, Jacobian, Tangent, AD types, etc.). The provided
 * workset contains all parameter and field information necessary for Jacobian
 * evaluation.
 *
 * @tparam EvalT    Evaluation type used for automatic differentiation.
 *
 * @param block     Block index of the boundary region.
 * @param grp       Group index inside the block.
 * @param local_J   Local Jacobian storage for parameter sensitivities.
 * @param wset      Workset supplying fields, parameters, and basis data.
 *
 * @return void
 */
template<class EvalT>
void updateParamJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J,
                            Teuchos::RCP<Workset<EvalT> > & wset);

// ========================================================================================

/**
 * @brief Compute auxiliary contributions to the boundary Jacobian.
 *
 * Auxiliary terms arise from additional physics models or stabilization
 * components that contribute to the Jacobian through boundary integrals.
 * This routine computes those contributions and stores them in @p local_J.
 *
 * @param block     Block index of the boundary region.
 * @param grp       Group index inside the block.
 * @param local_J   Local Jacobian array to be modified.
 *
 * @return void
 */
void updateAuxJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J);

// ========================================================================================

/**
 * @brief Templated version of updateAuxJacBoundary().
 *
 * Invokes auxiliary boundary Jacobian computation using evaluation type
 * @p EvalT. The full workset supplies geometry, basis data, field values,
 * parameters, and time information.
 *
 * @tparam EvalT    Evaluation type used for AD-enabled evaluation.
 *
 * @param block     Block index corresponding to the boundary.
 * @param grp       Group index inside the block.
 * @param local_J   Local Jacobian contributions array.
 * @param wset      Workset containing all needed evaluation data.
 *
 * @return void
 */
template<class EvalT>
void updateAuxJacBoundary(const int & block, const size_t & grp, View_Sc3 local_J,
                          Teuchos::RCP<Workset<EvalT> > & wset);

// ========================================================================================

/**
 * @brief Get a view containing Dirichlet boundary values.
 *
 * Returns a rank-2 view containing prescribed Dirichlet values associated
 * with the given boundary set. Values may depend on block, group, and set
 * identifiers and can be time-dependent or parameter-dependent.
 *
 * @param block   Block index of the boundary.
 * @param grp     Group index within the block.
 * @param set     Boundary condition set identifier.
 *
 * @return View_Sc2  A Kokkos view containing the Dirichlet boundary values.
 */
View_Sc2 getDirichletBoundary(const int & block, const size_t & grp, const size_t & set);

// ========================================================================================

/**
 * @brief Get the mass matrix contributions associated with a boundary set.
 *
 * Returns a rank-3 view containing boundary mass operator entries for the
 * specified block, group, and set. Values may be used for transient mass
 * assembly, stabilization, or other boundary mass effects.
 *
 * @param block   Block index associated with the boundary.
 * @param grp     Group index inside the block.
 * @param set     Boundary mass set identifier.
 *
 * @return View_Sc3  View storing boundary mass contributions.
 */
View_Sc3 getMassBoundary(const int & block, const size_t & grp, const size_t & set);

// ========================================================================================

/**
 * @brief Update a workset using ScalarT (no AD).
 *
 * Selects and updates the appropriate workset for the given block and group,
 * filling dependent fields such as solution, gradients, basis data,
 * parameters, and time information. No AD derivatives are stored.
 *
 * @param block              Block index.
 * @param grp                Group index.
 * @param seedwhat           Identifier for the quantity to seed.
 * @param seedindex          Index of the seeded quantity.
 * @param override_transient Whether to override transient data in the workset.
 *
 * @return void
 */
void updateWorkset(const int & block, const size_t & grp, const int & seedwhat,
                   const int & seedindex, const bool & override_transient=false);

// ========================================================================================

/**
 * @brief Update a workset using AD-enabled evaluation types.
 *
 * Same as updateWorkset(), but uses automatic differentiation to enable full
 * Jacobian, tangent, or adjoint computations.
 *
 * @param block              Block index.
 * @param grp                Group index.
 * @param seedwhat           Quantity to seed.
 * @param seedindex          Index of seed.
 * @param override_transient Whether to override transient state.
 *
 * @return void
 */
void updateWorksetAD(const int & block, const size_t & grp, const int & seedwhat,
                     const int & seedindex, const bool & override_transient=false);

// ========================================================================================

/**
 * @brief Templated wrapper for workset updating.
 *
 * Identifies and selects the correct workset instance for evaluation type
 * @p EvalT, then invokes the fully templated implementation.
 *
 * @tparam EvalT    Evaluation type.
 *
 * @param block              Block index.
 * @param grp                Group index.
 * @param seedwhat           Quantity to seed.
 * @param seedindex          Seed index.
 * @param override_transient Whether to override transient parameters.
 *
 * @return void
 */
template<class EvalT>
void updateWorkset(const int & block, const size_t & grp, const int & seedwhat,
                   const int & seedindex, const bool & override_transient=false);

// ========================================================================================

/**
 * @brief Fully templated workset update.
 *
 * Performs the full update of solution fields, gradients, parameters, time
 * data, and basis information needed for kernels associated with the given
 * block and group.
 *
 * @tparam EvalT    Evaluation type.
 *
 * @param wset               Workset instance to populate.
 * @param block              Block index.
 * @param grp                Group index.
 * @param seedwhat           Seed identifier.
 * @param seedindex          Seed index.
 * @param override_transient Whether transient behavior is overridden.
 *
 * @return void
 */
template<class EvalT>
void updateWorkset(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp,
                   const int & seedwhat, const int & seedindex,
                   const bool & override_transient);

// ========================================================================================

/**
 * @brief Compute averaged solution quantities over the given block and group.
 *
 * Computes averages of selected solution variables, typically for diagnostic
 * output, stabilization, or auxiliary model evaluations.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return void
 */
void computeSolAvg(const int & block, const size_t & grp);

// ========================================================================================

/**
 * @brief Compute the average of a named solution variable.
 *
 * Extracts and averages a solution field identified by @p var over the
 * corresponding block and group. Output is stored in @p csol.
 *
 * @param block   Block index.
 * @param grp     Group index.
 * @param var     Name of the variable to average.
 * @param csol    Output average values (rank-2 view).
 *
 * @return void
 */
void computeSolutionAverage(const int & block, const size_t & grp,
                            const string & var, View_Sc2 csol);

// ========================================================================================

/**
 * @brief Compute the average value of a parameter over the block/group region.
 *
 * Retrieves and averages a named parameter field. Result is written to @p sol.
 *
 * @param block   Block index.
 * @param grp     Group index.
 * @param var     Parameter name.
 * @param sol     Output storage for averaged parameter values.
 *
 * @return void
 */
void computeParameterAverage(const int & block, const size_t & grp,
                             const string & var, View_Sc2 sol);

// ========================================================================================

/**
 * @brief Update face-specific workset fields (ScalarT version).
 *
 * Updates basis, geometry, and solution fields for a specific face number
 * in preparation for flux, mortar, or DG-type face integrals.
 *
 * @param block     Block index.
 * @param grp       Group index.
 * @param facenum   Face index (local to the element group).
 *
 * @return void
 */
void updateWorksetFace(const int & block, const size_t & grp, const size_t & facenum);

// ========================================================================================

/**
 * @brief AD-enabled version of updateWorksetFace().
 *
 * Same as updateWorksetFace(), but creates fields with derivative information.
 *
 * @param block     Block index.
 * @param grp       Group index.
 * @param facenum   Face index.
 *
 * @return void
 */
void updateWorksetFaceAD(const int & block, const size_t & grp, const size_t & facenum);

// ========================================================================================

/**
 * @brief Partially templated face workset update.
 *
 * Selects the appropriate workset for evaluation type @p EvalT and forwards
 * to the fully templated implementation.
 *
 * @tparam EvalT    Evaluation type.
 *
 * @param block     Block index.
 * @param grp       Group index.
 * @param facenum   Face index.
 *
 * @return void
 */
template<class EvalT>
void updateWorksetFace(const int & block, const size_t & grp, const size_t & facenum);

// ========================================================================================

/**
 * @brief Fully templated face workset update.
 *
 * Performs all needed updates for face integrals: geometry, normals, basis
 * values, field extraction, and time/parameter configuration.
 *
 * @tparam EvalT    Evaluation type.
 *
 * @param wset      Face workset handle.
 * @param block     Block index.
 * @param grp       Group index.
 * @param facenum   Face number.
 *
 * @return void
 */
template<class EvalT>
void updateWorksetFace(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp,
                       const size_t & facenum);

// ========================================================================================

/**
 * @brief Compute both Jacobian and residual terms for volume and/or face integrals.
 *
 * This is the main routine assembling contributions to element-local residual
 * and Jacobian matrices. Depending on the flags supplied, the routine may
 * assemble volume terms, face terms, time-dependent contributions, adjoint
 * terms, parameter sensitivities, discontinuous sensitivities, or auxiliary
 * contributions.
 *
 * @param block                Block index.
 * @param grp                  Group index.
 * @param time                 Current simulation time.
 * @param isTransient          Whether transient effects are included.
 * @param isAdjoint            Whether adjoint formulation is used.
 * @param compute_jacobian     Whether Jacobian contributions should be assembled.
 * @param compute_sens         Whether parameter sensitivities are computed.
 * @param num_active_params    Number of active parameters used in sensitivity evaluation.
 * @param compute_disc_sens    Whether discontinuous sensitivities are evaluated.
 * @param compute_aux_sens     Whether auxiliary sensitivities are computed.
 * @param store_adjPrev        Whether the previous-step adjoint solution is stored.
 * @param local_res            Local residual array to fill.
 * @param local_J              Local Jacobian array to fill.
 * @param assemble_volume_terms  Whether volume integral terms are assembled.
 * @param assemble_face_terms    Whether face integral terms are assembled.
 *
 * @return void
 */
void computeJacRes(const int & block, const size_t & grp,
                   const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                   const bool & compute_jacobian, const bool & compute_sens,
                   const int & num_active_params, const bool & compute_disc_sens,
                   const bool & compute_aux_sens, const bool & store_adjPrev,
                   Kokkos::View<ScalarT***,AssemblyDevice> local_res,
                   Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                   const bool & assemble_volume_terms,
                   const bool & assemble_face_terms);

// ========================================================================================
// ========================================================================================

/**
 * @brief Update the residual vector for the specified block and group.
 *
 * This routine performs residual assembly for the given element block and
 * group. It computes contributions from basis functions, flux terms, source
 * terms, and other physics-specific quantities, and inserts them into the
 * provided residual array.
 *
 * @param block         Index of the block within the global problem partition.
 * @param grp           Index of the target element group.
 * @param compute_sens  If true, sensitivities and derivative information are
 *                      also computed during residual assembly.
 * @param local_res     3-D view storing residual contributions for each element,
 *                      basis function, and equation.
 */
void updateRes(const int & block, const size_t & grp,
               const bool & compute_sens, View_Sc3 local_res);

/**
 * @brief Update the residual using evaluation-type dependent data.
 *
 * This templated overload performs residual assembly while incorporating
 * evaluation-type–dependent logic (e.g., AD types), enabling automatic
 * differentiation or tangent/adjoint computations through the provided workset.
 *
 * @tparam EvalT        Evaluation type (Residual, Jacobian, Tangent, etc.).
 * @param block         Block index.
 * @param grp           Group index.
 * @param compute_sens  Whether derivative/sensitivity information should be computed.
 * @param local_res     Local residual storage array.
 * @param wset          Workset containing element-level data, basis functions,
 *                      integration rules, and evaluation-type objects.
 */
template<class EvalT>
void updateRes(const int & block, const size_t & grp,
               const bool & compute_sens, View_Sc3 local_res,
               Teuchos::RCP<Workset<EvalT> > & wset);

/**
 * @brief Assemble the Jacobian matrix for the specified block and group.
 *
 * Computes the linearization of the governing equations and stores contributions
 * into the local Jacobian array. If @p useadjoint is true, the adjoint-form
 * Jacobian is assembled instead of the forward Jacobian.
 *
 * @param block       Block index.
 * @param grp         Group index.
 * @param useadjoint  Whether to assemble the adjoint Jacobian.
 * @param local_J     3-D view containing the local Jacobian contributions.
 */
void updateJac(const int & block, const size_t & grp,
               const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J);

/**
 * @brief Assemble the Jacobian using evaluation-type–specific structures.
 *
 * Templated version that allows AD and other evaluation-type logic to be used
 * directly during Jacobian assembly.
 *
 * @tparam EvalT     Evaluation type.
 * @param block      Block index.
 * @param grp        Group index.
 * @param useadjoint Whether to assemble the adjoint Jacobian.
 * @param local_J    Local Jacobian storage array.
 * @param wset       Workset containing element data and evaluation-type fields.
 */
template<class EvalT>
void updateJac(const int & block, const size_t & grp,
               const bool & useadjoint, Kokkos::View<ScalarT***,AssemblyDevice> local_J,
               Teuchos::RCP<Workset<EvalT> > & wset);

/**
 * @brief Apply diagonal corrections to the Jacobian matrix.
 *
 * Modifies diagonal entries to improve stability, apply regularization, or
 * impose certain constraints. May use residual information to determine
 * scaling or correction magnitudes.
 *
 * @param block       Block index.
 * @param grp         Group index.
 * @param local_J     Local Jacobian to modify.
 * @param local_res   Local residual vector used for computing corrections.
 */
void fixDiagJac(const int & block, const size_t & grp,
                Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                Kokkos::View<ScalarT***,AssemblyDevice> local_res);

/**
 * @brief Assemble Jacobian contributions with respect to model parameters.
 *
 * Computes ∂R/∂p contributions for all active parameters in the specified block
 * and group and stores them in @p local_J.
 *
 * @param block     Block index.
 * @param grp       Group index.
 * @param local_J   Local parameter-Jacobian storage.
 */
void updateParamJac(const int & block, const size_t & grp,
                    Kokkos::View<ScalarT***,AssemblyDevice> local_J);

/**
 * @brief Assemble parameter-related Jacobian contributions using evaluation-type logic.
 *
 * @tparam EvalT   Evaluation type.
 * @param block    Block index.
 * @param grp      Group index.
 * @param local_J  Parameter Jacobian storage.
 * @param wset     Workset containing AD or other evaluation-type objects.
 */
template<class EvalT>
void updateParamJac(const int & block, const size_t & grp,
                    Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                    Teuchos::RCP<Workset<EvalT> > & wset);

/**
 * @brief Assemble Jacobian contributions from auxiliary variables.
 *
 * Auxiliary variables may include internal variables or additional DOFs needed
 * by models (plasticity, damage, phase-field, etc.). This routine constructs
 * their Jacobian contributions.
 *
 * @param block     Block index.
 * @param grp       Group index.
 * @param local_J   Auxiliary Jacobian storage.
 */
void updateAuxJac(const int & block, const size_t & grp,
                  Kokkos::View<ScalarT***,AssemblyDevice> local_J);

/**
 * @brief Assemble auxiliary-variable Jacobian contributions using evaluation types.
 *
 * @tparam EvalT   Evaluation type.
 * @param block    Block index.
 * @param grp      Group index.
 * @param local_J  Auxiliary Jacobian storage.
 * @param wset     Workset with AD/evaluation-type fields.
 */
template<class EvalT>
void updateAuxJac(const int & block, const size_t & grp,
                  Kokkos::View<ScalarT***,AssemblyDevice> local_J,
                  Teuchos::RCP<Workset<EvalT> > & wset);

/**
 * @brief Retrieve initial-condition data for a block and group.
 *
 * Optionally projects initial data onto basis functions, and may return
 * adjoint-compatible initial conditions depending on @p isAdjoint.
 *
 * @param block     Block index.
 * @param grp       Group index.
 * @param project   Whether to project the initial field to the basis.
 * @param isAdjoint Whether to return adjoint initial conditions.
 *
 * @return A 2-D view containing initial values per node and equation.
 */
View_Sc2 getInitial(const int & block, const size_t & grp,
                    const bool & project, const bool & isAdjoint);

/**
 * @brief Retrieve initial-condition data located on faces.
 *
 * @param block     Block index.
 * @param grp       Group index.
 * @param project   Whether to project face initial data to the basis.
 *
 * @return 2-D view of initial face data.
 */
View_Sc2 getInitialFace(const int & block, const size_t & grp, const bool & project);

/**
 * @brief Retrieve mass matrix entries for the specified block and group.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return Compressed mass matrix storage.
 */
CompressedView<View_Sc3> getMass(const int & block, const size_t & grp);

/**
 * @brief Retrieve a weighted mass matrix for the given block and group.
 *
 * @param block     Block index.
 * @param grp       Group index.
 * @param masswts   Vector of scalar weights applied to each mass entry.
 *
 * @return Compressed mass matrix with weights applied.
 */
CompressedView<View_Sc3> getWeightedMass(const int & block, const size_t & grp,
                                         vector<ScalarT> & masswts);

/**
 * @brief Retrieve mass matrix associated with parameters.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return Compressed parameter-mass matrix view.
 */
CompressedView<View_Sc3> getParamMass(const int & block, const size_t & grp);

/**
 * @brief Retrieve face-based mass matrix entries.
 *
 * @param block   Block index.
 * @param grp     Group index.
 *
 * @return Compressed face mass matrix.
 */
CompressedView<View_Sc3> getMassFace(const int & block, const size_t & grp);

/**
 * @brief Extract solution values at mesh nodes for a given variable.
 *
 * @param block   Block index.
 * @param grp     Group index.
 * @param var     Variable index.
 *
 * @return 3-D view of nodal solution values.
 */
Kokkos::View<ScalarT***,AssemblyDevice>
getSolutionAtNodes(const int & block, const size_t & grp, const int & var);

/**
 * @brief Update group-specific data using evaluation-type dependent workset.
 *
 * @tparam EvalT   Evaluation type.
 * @param wset     Workset to populate.
 * @param block    Block index.
 * @param grp      Group index.
 */
template<class EvalT>
void updateGroupData(Teuchos::RCP<Workset<EvalT> > & wset, const int & block, const size_t & grp);

/**
 * @brief Set microstructure-related data in cells and boundary cells.
 *
 * Calls microstructure generation and initialization routines.
 */
void setMeshData();

/**
 * @brief Import microstructure data from external files.
 *
 * Reads microstructure descriptions and sets data in cells and boundary cells.
 */
void importMeshData();

/**
 * @brief Import quadrature data for all elements and boundaries.
 */
void importQuadratureData();

/**
 * @brief Assign each cell and boundary cell to a microstructure grain.
 *
 * @param randSeed   Random seed used for stochastic microstructure generation.
 * @param seeds      2-D view of grain seed locations.
 */
void importNewMicrostructure(int & randSeed, View_Sc2 seeds);

/**
 * @brief Identify subgrid models required for each block/group pair.
 *
 * @return Vector of subgrid model identifiers.
 */
vector<vector<int> > identifySubgridModels();

/**
 * @brief Create user-defined and physics-specific functions for the problem.
 */
void createFunctions();

/**
 * @brief Retrieve quadrature data for the specified block.
 *
 * @param block   Block name string.
 * @return 2-D quadrature data array.
 */
View_Sc2 getQuadratureData(string & block);

/**
 * @brief Retrieve boundary quadrature for a block and side name.
 *
 * @param block     Block name.
 * @param sidename  Side name identifier.
 *
 * @return 2-D view of boundary quadrature points/weights.
 */
View_Sc2 getboundaryQuadratureData(string & block, string & sidename);

/**
 * @brief Free allocated memory and clear cached internal data structures.
 */
void purgeMemory();

///////////////////////////////////////////////////////////////////////////////////////////
// Public data members
///////////////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<MpiComm> comm;                                 // MPI communicator used for parallel execution
Teuchos::RCP<Teuchos::ParameterList> settings;              // Global parameter list containing runtime configuration

// Need
std::vector<std::string> blocknames;                        // Names of blocks in the mesh
std::vector<std::vector<std::vector<std::string> > > varlist; // Variables per set/block/var index [set][block][var]

//Teuchos::RCP<panzer_stk::STK_Interface>  mesh;
Teuchos::RCP<MeshInterface> mesh;                           // Mesh interface abstraction
Teuchos::RCP<DiscretizationInterface> disc;                 // Discretization handler (basis, quadrature, dofs)
Teuchos::RCP<PhysicsInterface> physics;                     // Primary physics model interface
Teuchos::RCP<MultiscaleManager> multiscale_manager;         // Handles multiscale coupling and projections
Teuchos::RCP<MrHyDE_Debugger> debugger;                     // Optional debugging and output utilities

std::vector<Teuchos::RCP<FunctionManager<ScalarT> > > function_managers; // Scalar evaluators per block
#ifndef MrHyDE_NO_AD
std::vector<Teuchos::RCP<FunctionManager<AD> > > function_managers_AD;       // AD(1) function evaluators
std::vector<Teuchos::RCP<FunctionManager<AD2> > > function_managers_AD2;     // AD(2) evaluators
std::vector<Teuchos::RCP<FunctionManager<AD4> > > function_managers_AD4;     // AD(4) evaluators
std::vector<Teuchos::RCP<FunctionManager<AD8> > > function_managers_AD8;     // AD(8) evaluators
std::vector<Teuchos::RCP<FunctionManager<AD16> > > function_managers_AD16;   // AD(16) evaluators
std::vector<Teuchos::RCP<FunctionManager<AD18> > > function_managers_AD18;   // AD(18) evaluators
std::vector<Teuchos::RCP<FunctionManager<AD24> > > function_managers_AD24;   // AD(24) evaluators
std::vector<Teuchos::RCP<FunctionManager<AD32> > > function_managers_AD32;   // AD(32) evaluators
#endif

size_t globalParamUnknowns;                                 // Total number of global parameter unknowns
int verbosity;                                               // Level of diagnostic output

// Groups and worksets are unique to each block, but span the physics sets
std::vector<Teuchos::RCP<GroupMetaData> > groupData;         // Metadata per block [block]
std::vector<std::vector<Teuchos::RCP<Group> > > groups;       // Element groups [block][grp]
std::vector<std::vector<Teuchos::RCP<BoundaryGroup> > > boundary_groups; // Boundary groups [block][bgrp]

std::vector<Teuchos::RCP<Workset<ScalarT> > > wkset;         // Worksets for ScalarT evaluation
#ifndef MrHyDE_NO_AD
std::vector<Teuchos::RCP<Workset<AD> > > wkset_AD;           // AD(1) worksets
std::vector<Teuchos::RCP<Workset<AD2> > > wkset_AD2;         // AD(2) worksets
std::vector<Teuchos::RCP<Workset<AD4> > > wkset_AD4;         // AD(4) worksets
std::vector<Teuchos::RCP<Workset<AD8> > > wkset_AD8;         // AD(8) worksets
std::vector<Teuchos::RCP<Workset<AD16> > > wkset_AD16;       // AD(16) worksets
std::vector<Teuchos::RCP<Workset<AD18> > > wkset_AD18;       // AD(18) worksets
std::vector<Teuchos::RCP<Workset<AD24> > > wkset_AD24;       // AD(24) worksets
std::vector<Teuchos::RCP<Workset<AD32> > > wkset_AD32;       // AD(32) worksets
#endif

bool usestrongDBCs,                                          // Whether strong (Dirichlet) constraints are applied
use_meas_as_dbcs,                                       // Use measurements as boundary conditions
multiscale,                                             // Whether multiscale coupling is active
isTransient,                                            // Whether the problem is transient
fix_zero_rows,                                          // Whether to enforce row-fixing for singular matrices
lump_mass,                                              // Use mass-lumped matrices where applicable
matrix_free,                                            // Flag for matrix-free assembly path
allow_autotune,                                         // Enable autotuning options
store_nodes;                                            // Whether to store node coordinates

std::string assembly_partitioning;                            // Strategy used for parallel assembly partitioning
std::vector<std::vector<bool> > assemble_volume_terms,        // Assembly flags for volume terms [block][set]
assemble_boundary_terms,      // Assembly flags for boundary terms [block][set]
assemble_face_terms;          // Assembly flags for face terms [block][set]

std::vector<bool> build_volume_terms,                         // Whether to build volume-basis structures [block]
build_boundary_terms,                     // Whether to build boundary-basis structures [block]
build_face_terms;                         // Whether to build face-basis structures [block]

std::vector<Kokkos::View<bool*,LA_device> > isFixedDOF;       // DOF flags for fixed constraints [set]
std::vector<vector<vector<Kokkos::View<LO*,LA_device> > > > fixedDOF; // Fixed DOF lists [set][block][var]

Teuchos::RCP<ParameterManager<Node> > params;                 // Manager for nodal parameter values

vector<int> num_derivs_required;                              // Required AD derivative count per variable set
int type_AD;                                                  // Selected AD type identifier

///////////////////////////////////////////////////////////////////////////////////////////
// Private data members
///////////////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Teuchos::Time> assembly_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - total assembly"); // Total assembly timer
Teuchos::RCP<Teuchos::Time> assembly_res_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeRes() - residual assembly"); // Residual assembly timer
Teuchos::RCP<Teuchos::Time> assembly_jac_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJac() - Jacobian assembly"); // Jacobian assembly timer
Teuchos::RCP<Teuchos::Time> gather_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::gather()"); // DOF gather timer
Teuchos::RCP<Teuchos::Time> physics_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - physics evaluation"); // Physics evaluation timer
Teuchos::RCP<Teuchos::Time> boundary_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - boundary evaluation"); // Boundary evaluation timer
Teuchos::RCP<Teuchos::Time> scatter_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::scatter()"); // DOF scatter timer
Teuchos::RCP<Teuchos::Time> dbc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::dofConstraints()"); // Dirichlet constraint timer
Teuchos::RCP<Teuchos::Time> complete_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - fill complete"); // Final fill-complete timer
Teuchos::RCP<Teuchos::Time> ms_proj_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::computeJacRes() - multiscale projection"); // Multiscale projection timer
Teuchos::RCP<Teuchos::Time> set_init_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::setInitial()"); // Initial condition setup timer
Teuchos::RCP<Teuchos::Time> set_dbc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::setDirichlet()"); // Dirichlet BC setup timer
Teuchos::RCP<Teuchos::Time> group_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::createGroups()"); // Group creation timer
Teuchos::RCP<Teuchos::Time> wkset_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::createWorkset()"); // Workset creation timer
Teuchos::RCP<Teuchos::Time> group_database_create_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::dataBase - assignment"); // Group database assignment timer
Teuchos::RCP<Teuchos::Time> group_database_basis_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::AssemblyManager::dataBase - basis"); // Group basis-building timer

}; // Class: AssemblyManager

} // MrHyDE namespace

#endif
