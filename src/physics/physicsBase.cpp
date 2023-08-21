/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
 ************************************************************************/

#include "physicsBase.hpp"

using namespace MrHyDE;

    /**
     * \brief Constructor for physics base
     * 
     * \param[in] settings  The parameter list of settings
     * \param[in] dimension_  Spatial dimensionality
     */
    template<class EvalT>
    PhysicsBase<EvalT>::PhysicsBase(Teuchos::ParameterList & settings, const int & dimension_) {
      verbosity = settings.get<int>("verbosity",0);
    };
    
    // (not necessary, but probably need to be defined in all modules)
    /**
     * \brief Define the functions for the module based on the input parameterList.
     * 
     * \param[in] fs The parameter list of options (usually obtained from the input deck)
     * \param[in] functionManager_ The function manager used to create \ref functionManager.
     */
    //virtual
    template<class EvalT>
    void PhysicsBase<EvalT>::defineFunctions(Teuchos::ParameterList & fs,
                         Teuchos::RCP<FunctionManager<EvalT> > & functionManager_) {
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
    //virtual
    template<class EvalT>
    void PhysicsBase<EvalT>::volumeResidual() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::volumeResidual called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    /**
     * \brief Compute the boundary contributions to the residual.
     */
    //virtual
    template<class EvalT>
    void PhysicsBase<EvalT>::boundaryResidual() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::boundaryResidual called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    /**
     * \brief Compute the edge (2D) and face (3D) contributions to the residual
     */
    //virtual
    template<class EvalT>
    void PhysicsBase<EvalT>::faceResidual() {
      if (verbosity > 10) {
        std::cout << "Warning: physicsBase::faceResidual called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    /**
     * \brief Compute the boundary/edge flux
     */
    //virtual
    template<class EvalT>
    void PhysicsBase<EvalT>::computeFlux() {
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
    //virtual 
    template<class EvalT>
    void PhysicsBase<EvalT>::updateParameters(const vector<Teuchos::RCP<vector<EvalT> > > & params,
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
    //virtual 
    template<class EvalT>
    void PhysicsBase<EvalT>::setWorkset(Teuchos::RCP<Workset<EvalT> > & wkset_) {
      wkset = wkset_;
    };
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Get the name of derived quantities for the class.
     */
    //virtual 
    template<class EvalT>
    std::vector<string> PhysicsBase<EvalT>::getDerivedNames() {
      std::vector<string> derived;
      return derived;
    };
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Get the values of derived quantities for the class.
     */
    //virtual 
    //template<class EvalT>
    //std::vector<View_EvalT2> PhysicsBase<EvalT>::getDerivedValues() {
    //  std::vector<View_EvalT2> derived;
    //  return derived;
    //};
    
    // ========================================================================================
    // ========================================================================================
    
    /**
     * \brief Update flags for the class.
     * 
     * \param[in] newflags The flags to use for the update.
     * \note This currently does nothing.
     */
    //virtual 
    template<class EvalT>
    void PhysicsBase<EvalT>::updateFlags(std::vector<bool> & newflags) {
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
    //virtual 
    template<class EvalT>
    std::vector< std::vector<string> > PhysicsBase<EvalT>::setupIntegratedQuantities(const int & spaceDim) {
      std::vector< std::vector<string> > integrandsNamesAndTypes;
      return integrandsNamesAndTypes;
    };

    /**
     * \brief Updates any values needed by the residual which depend on integrated quantities
     * required by the physics module.
     *
     * This must be called after the postprocessing routine.
     */
    //virtual 
    template<class EvalT>
    void PhysicsBase<EvalT>::updateIntegratedQuantitiesDependents() {
      if (verbosity > 10) {
        std::cout << "*** Warning: physicsBase::updateIntegratedQuantitiesDependents() called!" << std::endl;
        std::cout << "*** This probably means the functionality requested is not implemented in the physics module." << std::endl;
      }
    };
    
    // ========================================================================================
    // ========================================================================================


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::PhysicsBase<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::PhysicsBase<AD>;

// Standard built-in types
template class MrHyDE::PhysicsBase<AD2>;
template class MrHyDE::PhysicsBase<AD4>;
template class MrHyDE::PhysicsBase<AD8>;
template class MrHyDE::PhysicsBase<AD16>;
template class MrHyDE::PhysicsBase<AD18>;
template class MrHyDE::PhysicsBase<AD24>;
template class MrHyDE::PhysicsBase<AD32>;
#endif
