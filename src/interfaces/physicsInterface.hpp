/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

/**
 * @file physicsInterface.hpp
 * @brief Contains the interface to the MrHyDE-specific physics modules.
 * @details
 *   This file defines the primary interface class used to load, initialize,
 *   and communicate with physics modules in the MrHyDE framework. It handles
 *   variable registration, discretization setup, automatic differentiation
 *   type management, and boundary/initial condition evaluation.
 */

#ifndef MRHYDE_PHYSICSINTERFACE_H
#define MRHYDE_PHYSICSINTERFACE_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "MrHyDE_Debugger.hpp"

#include "physicsBase.hpp"
#include "workset.hpp"
#include "Panzer_DOFManager.hpp"

namespace MrHyDE {

/**
 * @class PhysicsInterface
 * @brief Interface between MrHyDE and all physics modules.
 * @details
 *   This class is the sole central mechanism through which the framework
 *   loads physics modules, registers variables, distributes discretization
 *   information, and evaluates analytic/functional data (e.g. boundary and
 *   initial conditions). It also manages the use of automatic differentiation
 *   types and builds function managers for physics modules.
 */
class PhysicsInterface {
  
#ifndef MrHyDE_NO_AD
  typedef Kokkos::View<AD*,   ContLayout, AssemblyDevice> View_AD1; ///< 1D AD view
  typedef Kokkos::View<AD**,  ContLayout, AssemblyDevice> View_AD2; ///< 2D AD view
  typedef Kokkos::View<AD***, ContLayout, AssemblyDevice> View_AD3; ///< 3D AD view
  typedef Kokkos::View<AD****,ContLayout, AssemblyDevice> View_AD4; ///< 4D AD view
#else
  typedef View_Sc1 View_AD1; ///< 1D scalar view (no AD)
  typedef View_Sc2 View_AD2; ///< 2D scalar view (no AD)
  typedef View_Sc3 View_AD3; ///< 3D scalar view (no AD)
  typedef View_Sc4 View_AD4; ///< 4D scalar view (no AD)
#endif
  
public:
  
  /**
   * @brief Default constructor.
   * @details Creates an empty physics interface with no modules loaded.
   */
  PhysicsInterface() {}
  
  /**
   * @brief Destructor (trivial).
   */
  ~PhysicsInterface() {}
  
  /**
   * @brief Main constructor providing all required problem metadata.
   *
   * @param settings        Parameter list describing physics modules, variables,
   *                        discretization settings, and options.
   * @param Comm_           MPI communicator wrapper.
   * @param block_names_    Names of mesh blocks (volumetric regions).
   * @param side_names_     Names of sidesets (boundary regions).
   * @param dimension_      Spatial dimension of the problem.
   */
  PhysicsInterface(Teuchos::RCP<Teuchos::ParameterList> & settings,
                   Teuchos::RCP<MpiComm> & Comm_,
                   std::vector<string> block_names_,
                   std::vector<string> side_names_,
                   int dimension_);
  
  /**
   * @brief Load and initialize physics modules and variables.
   * @details
   *   Reads module specifications from the parameter list and constructs
   *   the appropriate physics module objects, variables, and discretizations.
   */
  void importPhysics();
  
  /**
   * @brief Import physics using an explicit automatic-differentiation type.
   *
   * @param type_AD_  Integer flag identifying the AD mode (e.g., none, forward,
   *                  reverse, hybrid).
   */
  void importPhysicsAD(int & type_AD_);
  
  /**
   * @brief Split a delimited string into a vector of tokens.
   *
   * @param list       String containing delimited values.
   * @param delimiter  The substring used as a separator.
   * @return Vector of extracted string tokens.
   */
  vector<string> breakupList(const string & list, const string & delimiter);
  
  /**
   * @brief Register user-defined functions into function managers.
   *
   * @tparam EvalT              Evaluation type (Residual, Jacobian, etc.)
   * @param functionManagers_   Vector of function manager objects to populate.
   */
  template<class EvalT>
  void defineFunctions(vector<Teuchos::RCP<FunctionManager<EvalT>>> & functionManagers_);
  
  /**
   * @brief Define functions for each physics module grouped by block/set.
   *
   * @tparam EvalT   Evaluation type for automatic differentiation.
   * @param func_managers  Set of function managers to populate.
   * @param mods           Nested collection of physics modules providing
   *                       function definitions.
   */
  template<class EvalT>
  void defineFunctions(vector<Teuchos::RCP<FunctionManager<EvalT>>> & func_managers,
                       vector<vector<vector<Teuchos::RCP<PhysicsBase<EvalT>>>>> & mods);
  
  /**
   * @brief Determine the owning physics module for a variable.
   *
   * @param set    Set index within the module structure.
   * @param block  Block index.
   * @param var    Variable name.
   * @return Index of the physics module that owns the variable.
   */
  int getvarOwner(const int & set, const int & block, const string & var);
  
  /**
   * @brief Evaluate a Dirichlet boundary condition value.
   *
   * @param block       Block index.
   * @param x,y,z       Spatial coordinates where the BC is evaluated.
   * @param t           Time value.
   * @param var         Name of the variable.
   * @param gside       Boundary sideset name.
   * @param useadjoint  Whether to use the adjoint version of the BC.
   * @param wkset       Workset supplying basis, geometry, and integration data.
   * @return AD         The evaluated boundary condition value.
   */
  AD getDirichletValue(const int & block,
                       const ScalarT & x, const ScalarT & y, const ScalarT & z,
                       const ScalarT & t,
                       const string & var,
                       const string & gside,
                       const bool & useadjoint,
                       Teuchos::RCP<Workset<AD>> & wkset);
    
  /////////////////////////////////////////////////////////////////////////////////////////////
  // INITIAL CONDITION UTILITIES
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Evaluate the initial condition for a variable at a spatial point.
   *
   * @param block        Block index.
   * @param x,y,z        Physical coordinates.
   * @param var          Name of the variable whose initial condition is requested.
   * @param useadjoint   Whether to evaluate the adjoint initial condition.
   *
   * @return ScalarT     The evaluated initial condition value.
   *
   * @details
   *   This function queries the physics module owning the variable and returns
   *   the appropriate analytic or functional initial condition.
   */
  ScalarT getInitialValue(const int & block,
                          const ScalarT & x, const ScalarT & y, const ScalarT & z,
                          const string & var, const bool & useadjoint);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // BULK INITIALIZATION ON VOLUME POINTS
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Compute initial condition values at integration points in a volume.
   *
   * @param pts       Vector of integration point coordinate arrays.
   * @param set       Set index.
   * @param block     Block index.
   * @param project   Whether to project the initial data into the FE basis.
   * @param wkset     Workset providing basis and geometry information.
   *
   * @return View_Sc4 4-dimensional array containing initial condition values.
   *
   * @details
   *   This routine evaluates initial conditions for all variables in a block,
   *   possibly involving L2 projection depending on @p project.
   */
  View_Sc4 getInitial(vector<View_Sc2> & pts, const int & set, const int & block,
                      const bool & project, Teuchos::RCP<Workset<ScalarT>> & wkset);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // FACE INITIAL CONDITIONS
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Evaluate initial conditions on face integration points.
   *
   * @param pts       2D coordinate arrays for face integration points.
   * @param set       Set index.
   * @param block     Block index.
   * @param project   Whether to apply L2 projection.
   * @param wkset     Workset providing geometry and basis data.
   *
   * @return View_Sc3 3-dimensional array of initial condition values.
   *
   * @warning Under development — behavior of non-projection branch is not finalized.
   *
   * @details
   *   Used when initializing face-based variables or DG traces. Supports projection
   *   when required by discretization type or user settings.
   */
  View_Sc3 getInitialFace(vector<View_Sc2> & pts, const int & set, const int & block,
                          const bool & project, Teuchos::RCP<Workset<ScalarT>> & wkset);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // DIRICHLET CONDITION EXTRACTION
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Retrieve Dirichlet boundary condition values for a variable.
   *
   * @param var         Variable index.
   * @param set         Set index.
   * @param block       Block index.
   * @param sidename    Name of the boundary sideset.
   *
   * @return View_Sc2   2D array containing Dirichlet values for each IP.
   */
  View_Sc2 getDirichlet(const int & var, const int & set,
                        const int & block, const std::string & sidename);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // VARIABLE REGISTRATION
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Populate internal variable tables based on physics metadata.
   *
   * @details
   *   This function initializes:
   *    - variable lists,
   *    - variable ownership,
   *    - discretization orders,
   *    - type categories,
   *    - unique basis information.
   */
  void setVars();


  /////////////////////////////////////////////////////////////////////////////////////////////
  // VARIABLE OWNERSHIP QUERIES
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Return an index identifying the unique (set,block,var) combination.
   *
   * @param set     Set index.
   * @param block   Block index.
   * @param var     Variable name.
   *
   * @return int    Unique integer index for the variable.
   */
  int getUniqueIndex(const int & set, const int & block, const std::string & var);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // RESIDUAL ASSEMBLY ROUTINES
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Compute the volume (cell-interior) residual contributions.
   *
   * @tparam EvalT  Evaluation type (Residual, Jacobian, Tangent, etc.)
   *
   * @param set   Set index.
   * @param block Block index.
   *
   * @details
   *   Loops over cells and physics modules to assemble strong-form volume terms.
   */
  template<class EvalT>
  void volumeResidual(const size_t & set, const size_t block);


  /**
   * @brief Compute residual contributions on boundary sidesets.
   *
   * @tparam EvalT  Evaluation type.
   *
   * @param set   Set index.
   * @param block Block index.
   */
  template<class EvalT>
  void boundaryResidual(const size_t & set, const size_t block);


  /**
   * @brief Compute fluxes for a given block.
   *
   * @tparam EvalT  Evaluation type.
   *
   * @param set   Set index.
   * @param block Block index.
   */
  template<class EvalT>
  void computeFlux(const size_t & set, const size_t block);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // WORKSET INITIALIZATION
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Assign worksets for ScalarT evaluation type.
   *
   * @param wkset   Vector of workset objects (cell, side, boundary, etc.)
   */
  void setWorkset(vector<Teuchos::RCP<Workset<ScalarT>>> & wkset);


  #ifndef MrHyDE_NO_AD
  /// @name AD Workset Initializers
  ///@{

  /** @brief Set worksets for AD evaluation. */
  void setWorkset(vector<Teuchos::RCP<Workset<AD>>> & wkset);

  /** @brief Set worksets for AD2 evaluation. */
  void setWorkset(vector<Teuchos::RCP<Workset<AD2>>> & wkset);

  /** @brief Set worksets for AD4 evaluation. */
  void setWorkset(vector<Teuchos::RCP<Workset<AD4>>> & wkset);

  /** @brief Set worksets for AD8 evaluation. */
  void setWorkset(vector<Teuchos::RCP<Workset<AD8>>> & wkset);

  /** @brief Set worksets for AD16 evaluation. */
  void setWorkset(vector<Teuchos::RCP<Workset<AD16>>> & wkset);

  /** @brief Set worksets for AD18 evaluation. */
  void setWorkset(vector<Teuchos::RCP<Workset<AD18>>> & wkset);

  /** @brief Set worksets for AD24 evaluation. */
  void setWorkset(vector<Teuchos::RCP<Workset<AD24>>> & wkset);

  /** @brief Set worksets for AD32 evaluation. */
  void setWorkset(vector<Teuchos::RCP<Workset<AD32>>> & wkset);

  ///@}
  #endif


  /////////////////////////////////////////////////////////////////////////////////////////////
  // FACE RESIDUALS AND FLUX CONDITIONS
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Check if a block has any face-based physics contributions.
   *
   * @param set    Set index.
   * @param block  Block index.
   *
   * @return bool  True if face terms exist.
   */
  bool checkFace(const size_t & set, const size_t & block);


  /**
   * @brief Compute DG or interface-type residual contributions on faces.
   *
   * @tparam EvalT Evaluation type.
   *
   * @param set    Set index.
   * @param block  Block index.
   */
  template<class EvalT>
  void faceResidual(const size_t & set, const size_t block);


  /**
   * @brief Apply flux conditions along faces or boundaries.
   *
   * @tparam EvalT Evaluation type.
   *
   * @param set    Set index.
   * @param block  Block index.
   */
  template<class EvalT>
  void fluxConditions(const size_t & set, const size_t block);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // FLAG UPDATES
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Update physics-module flags based on changes detected during assembly.
   *
   * @param newflags  Boolean list indicating updated activation states.
   */
  void updateFlags(vector<bool> & newflags);


  /////////////////////////////////////////////////////////////////////////////////////////////
  // VARIABLE LIST ACCESSORS
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Get the variable list for all (set,block,var) entries.
   *
   * @return Nested vector of variable names.
   */
  vector<vector<vector<string>>> getVarList() { return var_list; }


  /**
   * @brief Get variable type information (e.g., scalar, vector, tensor).
   *
   * @return Nested vector of variable types.
   */
  vector<vector<vector<string>>> getVarTypes() { return types; }


  /**
   * @brief Collect lists of derived quantities from all physics modules.
   *
   * @return Nested vector of derived field names.
   */
  vector<vector<vector<vector<string>>>> getDerivedList() {
      vector<vector<vector<vector<string>>>> dlist;
      for (size_t set=0; set<modules.size(); ++set) {
          vector<vector<vector<string>>> setlist;
          for (size_t blk=0; blk<modules[set].size(); ++blk) {
              vector<vector<string>> blklist;
              for (size_t mod=0; mod<modules[set][blk].size(); ++mod) {
                  blklist.push_back(modules[set][blk][mod]->getDerivedNames());
              }
              setlist.push_back(blklist);
          }
          dlist.push_back(setlist);
      }
      return dlist;
  }


  /////////////////////////////////////////////////////////////////////////////////////////////
  // MEMORY CLEANUP
  /////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Release memory associated with physics modules and internal data structures.
   *
   * @details
   *   Intended for use after a solve, when modules need to be cleared or reset.
   */
  void purgeMemory();

  /////////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  /** @name Settings & Communication
   *  @{ */
  Teuchos::RCP<Teuchos::ParameterList> settings; ///< Parameter list containing physics and solver settings.
  Teuchos::RCP<MpiComm> comm; ///< Parallel communicator used throughout the interface.
  int dimension, type_AD; ///< Spatial dimension and automatic differentiation type.
  vector<string> set_names, block_names, side_names; ///< Names of sets, blocks, and sides in the mesh.
  /** @} */

  /** @name Variable & Discretization Information
   *  @{ */
  vector<vector<size_t> > num_vars; ///< Number of variables per set and block.
  vector<int> num_derivs_required; ///< Number of derivatives required by the AD system.
  vector<vector<vector<string> > > var_list; ///< List of variables.
  vector<vector<vector<int> > > var_owned; ///< Variable ownership map.
  vector<vector<vector<int> > > orders; ///< Polynomial orders of variables.
  vector<vector<vector<string> > > types; ///< Variable types.
  vector<vector<int> > unique_orders; ///< Unique polynomial orders.
  vector<vector<string> > unique_types; ///< Unique variable types.
  vector<vector<int> > unique_index; ///< Unique indices.
  string initial_type; ///< Type of initial condition.
  /** @} */

  /** @name Function Managers
   *  @{ */
  vector<Teuchos::RCP<FunctionManager<ScalarT> > > function_managers; ///< Scalar function managers.
  #ifndef MrHyDE_NO_AD
  vector<Teuchos::RCP<FunctionManager<AD> > > function_managers_AD; ///< AD function managers.
  vector<Teuchos::RCP<FunctionManager<AD2> > > function_managers_AD2; ///< AD2 function managers.
  vector<Teuchos::RCP<FunctionManager<AD4> > > function_managers_AD4; ///< AD4 function managers.
  vector<Teuchos::RCP<FunctionManager<AD8> > > function_managers_AD8; ///< AD8 function managers.
  vector<Teuchos::RCP<FunctionManager<AD16> > > function_managers_AD16; ///< AD16 function managers.
  vector<Teuchos::RCP<FunctionManager<AD18> > > function_managers_AD18; ///< AD18 function managers.
  vector<Teuchos::RCP<FunctionManager<AD24> > > function_managers_AD24; ///< AD24 function managers.
  vector<Teuchos::RCP<FunctionManager<AD32> > > function_managers_AD32; ///< AD32 function managers.
  #endif
  /** @} */

  /** @name Physics & Discretization Settings
   *  @{ */
  vector<vector<Teuchos::ParameterList>> physics_settings, disc_settings, solver_settings; ///< Physics, discretization, and solver settings per set and block.
  vector<vector<vector<bool> > > use_subgrid; ///< Flags for subgrid modeling.
  vector<vector<vector<bool> > > use_DG; ///< Flags for DG discretization.
  vector<vector<vector<ScalarT> > > mass_wts, norm_wts; ///< Mass and norm weights.
  /** @} */

  /** @name Extra Fields & Responses
   *  @{ */
  vector<vector<string> > extra_fields_list, extra_cell_fields_list, response_list, target_list, weight_list; ///< Extra field and response lists.
  /** @} */

  /** @name Physics Modules
   *  @{ */
  vector<vector<vector<Teuchos::RCP<PhysicsBase<ScalarT> > > > > modules; ///< Scalar physics modules.
  #ifndef MrHyDE_NO_AD
  vector<vector<vector<Teuchos::RCP<PhysicsBase<AD> > > > > modules_AD; ///< AD physics modules.
  vector<vector<vector<Teuchos::RCP<PhysicsBase<AD2> > > > > modules_AD2; ///< AD2 physics modules.
  vector<vector<vector<Teuchos::RCP<PhysicsBase<AD4> > > > > modules_AD4; ///< AD4 physics modules.
  vector<vector<vector<Teuchos::RCP<PhysicsBase<AD8> > > > > modules_AD8; ///< AD8 physics modules.
  vector<vector<vector<Teuchos::RCP<PhysicsBase<AD16> > > > > modules_AD16; ///< AD16 physics modules.
  vector<vector<vector<Teuchos::RCP<PhysicsBase<AD18> > > > > modules_AD18; ///< AD18 physics modules.
  vector<vector<vector<Teuchos::RCP<PhysicsBase<AD24> > > > > modules_AD24; ///< AD24 physics modules.
  vector<vector<vector<Teuchos::RCP<PhysicsBase<AD32> > > > > modules_AD32; ///< AD32 physics modules.
  #endif
  /** @} */
  
private:
  /** @name Internal Debugger & Performance Timers
   *  @{ */
  Teuchos::RCP<MrHyDE_Debugger> debugger; ///< Internal debugging utility for MrHyDE.
  Teuchos::RCP<Teuchos::Time> bc_timer; ///< Timer for boundary-condition setup operations.
  Teuchos::RCP<Teuchos::Time> dbc_timer; ///< Timer for Dirichlet boundary data setup.
  Teuchos::RCP<Teuchos::Time> side_info_timer; ///< Timer for side-information extraction.
  Teuchos::RCP<Teuchos::Time> response_timer; ///< Timer for global response computations.
  Teuchos::RCP<Teuchos::Time> point_reponse_timer; ///< Timer for point-wise response evaluations.
  /** @} */

  
};

}

#endif
