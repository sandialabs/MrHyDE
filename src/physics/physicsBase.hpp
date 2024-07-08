/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_PHYSICS_BASE_H
#define MRHYDE_PHYSICS_BASE_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"
#include "functionManager.hpp"
#include "compressedView.hpp"

namespace MrHyDE {
  
  /**
   * \brief The base physics class that all physics modules must derive from.
   * 
   * This class contains virtual interfaces for computing residuals.
   * When computing residuals for physics, it is expected that the appropriate 
   * virtual methods are overridden. If the appropriate virtual method is not
   * overridden, a message will be printed out.
   */

  template<class EvalT>
  class PhysicsBase {
    
  public:

    typedef Kokkos::View<EvalT*,ContLayout,AssemblyDevice> View_EvalT1;
    typedef Kokkos::View<EvalT**,ContLayout,AssemblyDevice> View_EvalT2;
    typedef Kokkos::View<EvalT***,ContLayout,AssemblyDevice> View_EvalT3;
    typedef Kokkos::View<EvalT****,ContLayout,AssemblyDevice> View_EvalT4;
    
    PhysicsBase() {};
    
    virtual 
    ~PhysicsBase() {};
    
    /**
     * \brief Constructor for physics base
     * 
     * \param[in] settings  The parameter list of settings
     * \param[in] dimension_  Spatial dimensionality
     */
    PhysicsBase(Teuchos::ParameterList & settings, const int & dimension_);

    // (not necessary, but probably need to be defined in all modules)
    /**
     * \brief Define the functions for the module based on the input parameterList.
     * 
     * \param[in] fs The parameter list of options (usually obtained from the input deck)
     * \param[in] functionManager_ The function manager used to create \ref functionManager.
     */
    virtual
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT> > & functionManager_);
    
    /**
     * \brief Compute the volumetric contributions to the residual.
     */
    virtual
    void volumeResidual();
    
    /**
     * \brief Compute the boundary contributions to the residual.
     */
    virtual
    void boundaryResidual();
    
    /**
     * \brief Compute the edge (2D) and face (3D) contributions to the residual
     */
    virtual
    void faceResidual();

    /**
     * \brief Compute the boundary/edge flux
     */
    virtual
    void computeFlux();
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Update physics parameters using 
     * 
     * \param[in] params The input parameters
     * \param[in] paramnames The names of the input parameters
     * \note This will likely be deprecated, as this is used in few cases.
     */
    //virtual
    //void updateParameters(const vector<Teuchos::RCP<vector<EvalT> > > & params,
    //                              const std::vector<string> & paramnames);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Setter for \ref wkset member variable
     * 
     * \param[in] wkset_ An RCP for the workset to assign
     */
    virtual 
    void setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_);
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Get the name of derived quantities for the class.
     */
    virtual 
    std::vector<string> getDerivedNames();
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Get the values of derived quantities for the class.
     */
    virtual 
    std::vector<View_EvalT2> getDerivedValues() {
      std::vector<View_EvalT2> derived;
      return derived;
    }
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Update flags for the class.
     * 
     * \param[in] newflags The flags to use for the update.
     * \note This currently does nothing.
     */
    virtual 
    void updateFlags(std::vector<bool> & newflags);
    
    // ========================================================================================
    // ========================================================================================

    /**
     * \brief Returns the integrand and its type (boundary/volume) for integrated quantities required
     * by the physics module. 
     *
     * In general, the user may also request integrated quantities in the input
     * file. The number of spatial dimensions is required explicitly here because the workset is 
     * not finalized before the postprocessing manager is set up.
     *
     * \param[in] spaceDim  The number of spatial dimensions.
     * \return integrandsNamesAndTypes  Integrands, names, and type (boundary/volume) (matrix of strings).
     */
    virtual 
    std::vector< std::vector<string> > setupIntegratedQuantities(const int & spaceDim);

    /**
     * \brief Updates any values needed by the residual which depend on integrated quantities
     * required by the physics module.
     *
     * This must be called after the postprocessing routine.
     */
    virtual 
    void updateIntegratedQuantitiesDependents();
    
    // ========================================================================================
    // ========================================================================================

    /**
     * The name of the physics module.
     */
    string label;
    
    /** 
     * The \ref workset for the class. This contains a variety of metadata 
     * and numerical data necessary for computing residuals.
     */ 
    Teuchos::RCP<Workset<EvalT> > wkset;

    /**
     * The FunctionManager for the class. Depending on the physics module,
     * this contains a wide variety of functions to evaluate at integration points. 
     */
    Teuchos::RCP<FunctionManager<EvalT> > functionManager;

    vector<string> myvars, mybasistypes;
    bool include_face = false, isaux = false;
    string prefix = "";

    /**
     * The verbosity for the class. A verbosity strictly greater
     * than 10 will cause warnings to print to standard out when 
     * this class' default implementations are called.
     */
    int verbosity;
    
    // Probably not used much
    View_EvalT2 adjrhs;
    
  };
  
}

//template class MrHyDE::PhysicsBase<AD>;

#endif
