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
  class physicsbase {
    
  public:
    
    physicsbase() {};
    
    virtual ~physicsbase() {};
    
    /**
     * \brief Constructor for physics base
     * 
     * \param[in] settings  The parameter list of settings
     * \param[in] dimension_  Spatial dimensionality
     */
    physicsbase(Teuchos::ParameterList & settings, const int & dimension_) {
      verbosity = settings.get<int>("verbosity",0);
    };
    
    // (not necessary, but probably need to be defined in all modules)
    /**
     * \brief Define the functions for the module based on the input parameterList.
     * 
     * \param[in] fs The parameter list of options (usually obtained from the input deck)
     * \param[in] functionManager_ The function manager used to create \ref functionManager.
     */
    virtual
    void defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager> & functionManager_) {
      functionManager = functionManager_;
      // GH: these print statements may be annoying when running on multiple MPI ranks
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::defineFunctions called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    /**
     * \brief Compute the volumetric contributions to the residual.
     */
    virtual
    void volumeResidual() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::volumeResidual called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    /**
     * \brief Compute the boundary contributions to the residual.
     */
    virtual
    void boundaryResidual() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::boundaryResidual called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    /**
     * \brief Compute the edge (2D) and face (3D) contributions to the residual
     */
    virtual
    void faceResidual() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::faceResidual called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    /**
     * \brief Compute the boundary/edge flux
     */
    virtual
    void computeFlux() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::computeFlux called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Update physics parameters using 
     * 
     * \param[in] params The input parameters
     * \param[in] paramnames The names of the input parameters
     * \note This will likely be deprecated, as this is used in few cases.
     */
    virtual void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params,
                                  const std::vector<string> & paramnames) {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::updateParameters called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Setter for \ref wkset member variable
     * 
     * \param[in] wkset_ An RCP for the workset to assign
     */
    virtual void setWorkset(Teuchos::RCP<Workset> & wkset_) {
      wkset = wkset_;
    };
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Get the name of derived quantities for the class.
     */
    virtual std::vector<string> getDerivedNames() {
      std::vector<string> derived;
      return derived;
    };
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Get the values of derived quantities for the class.
     */
    virtual std::vector<View_AD2> getDerivedValues() {
      std::vector<View_AD2> derived;
      return derived;
    };
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Update flags for the class.
     * 
     * \param[in] newflags The flags to use for the update.
     * \note This currently does nothing.
     */
    virtual void updateFlags(std::vector<bool> & newflags) {
      // default is to do nothing
    };
    
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
    virtual std::vector< std::vector<string> > setupIntegratedQuantities(const int & spaceDim) {
      std::vector< std::vector<string> > integrandsNamesAndTypes;
      return integrandsNamesAndTypes;
    };

    /**
     * \brief Updates any values needed by the residual which depend on integrated quantities
     * required by the physics module.
     *
     * This must be called after the postprocessing routine.
     */
    virtual void updateIntegratedQuantitiesDependents() {
      if (verbosity > 10) {
        std::cout << "*** Warning: physicsBase::updateIntegratedQuantitiesDependents() called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
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
    Teuchos::RCP<Workset> wkset;

    /**
     * The FunctionManager for the class. Depending on the physics module,
     * this contains a wide variety of functions to evaluate at integration points. 
     */
    Teuchos::RCP<FunctionManager> functionManager;

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
    View_AD2 adjrhs;
    
    // On host, so ok
    // Kokkos::View<int**,HostDevice> bcs;
    
  };
  
}

#endif
