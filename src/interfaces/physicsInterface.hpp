/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

/** \file   physicsInterface.hpp
 \brief  Contains the interface to the MrHyDE-specific physics modules.
 \author Created by T. Wildey
 */

#ifndef MRHYDE_PHYSICSINTERFACE_H
#define MRHYDE_PHYSICSINTERFACE_H

#include "trilinos.hpp"
#include "preferences.hpp"

#include "physicsBase.hpp"
#include "workset.hpp"

//#include "Panzer_STK_Interface.hpp"
#include "Panzer_DOFManager.hpp"

namespace MrHyDE {
  
  /** \class  MrHyDE::PhysicsInterface
   \brief  Interface to the MrHyDE-specific physics modules.  This is the only class that direcly interacts with the physics modules.
   */
  
  class PhysicsInterface {

    #ifndef MrHyDE_NO_AD
      typedef Kokkos::View<AD*,ContLayout,AssemblyDevice> View_AD1;
      typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2;
      typedef Kokkos::View<AD***,ContLayout,AssemblyDevice> View_AD3;
      typedef Kokkos::View<AD****,ContLayout,AssemblyDevice> View_AD4;
    #else
      typedef View_Sc1 View_AD1;
      typedef View_Sc2 View_AD2;
      typedef View_Sc3 View_AD3;
      typedef View_Sc4 View_AD4;
    #endif

  public:
    
    // ========================================================================================
    /* Constructor to set up the problem */
    // ========================================================================================
    
    PhysicsInterface() {} ;
    
    ~PhysicsInterface() {} ;
    
    PhysicsInterface(Teuchos::RCP<Teuchos::ParameterList> & settings, Teuchos::RCP<MpiComm> & Comm_,
                     std::vector<string> block_names_, std::vector<string> side_names_,
                     int dimension_);
                     //Teuchos::RCP<panzer_stk::STK_Interface> & mesh);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Add the requested physics modules, variables, discretization types 
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void importPhysics();
    
    vector<string> breakupList(const string & list, const string & delimiter);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Add the functions to the function managers
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    template<class EvalT>
    void defineFunctions(vector<Teuchos::RCP<FunctionManager<EvalT> > > & functionManagers_);
    
    template<class EvalT>
    void defineFunctions(vector<Teuchos::RCP<FunctionManager<EvalT> > > & func_managers,
                         vector<vector<vector<Teuchos::RCP<PhysicsBase<EvalT> > > > > & mods);

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getvarOwner(const int & set, const int & block, const string & var);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    AD getDirichletValue(const int & block, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                         const ScalarT & t, const string & var, const string & gside,
                         const bool & useadjoint, Teuchos::RCP<Workset<AD> > & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    ScalarT getInitialValue(const int & block, const ScalarT & x, const ScalarT & y, const ScalarT & z,
                            const string & var, const bool & useadjoint);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    View_Sc4 getInitial(vector<View_Sc2> & pts, const int & set, const int & block,
                        const bool & project, Teuchos::RCP<Workset<ScalarT> > & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    /* @brief Evaluate the initial condition along the face integration point for L2 projection
     *
     * @param[in] pts  Face integration points
     * @param[in] block  Cell block
     * @param[in] project  Flag for L2 projection
     * @param[in] wkset  Workset
     * 
     * @returns View_Sc3 of the initial condition
     *
     * @warning BWR -- under development. Not sure what the nonprojection option is about.
     */
    
    View_Sc3 getInitialFace(vector<View_Sc2> & pts, const int & set, const int & block,
                            const bool & project, Teuchos::RCP<Workset<ScalarT> > & wkset);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////

    View_Sc2 getDirichlet(const int & var, const int & set,
                          const int & block, const std::string & sidename);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void setVars();
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void updateParameters(vector<Teuchos::RCP<vector<ScalarT> > > & params, const vector<string> & paramnames);
    
#ifndef MrHyDE_NO_AD
    void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames);
    void updateParameters(vector<Teuchos::RCP<vector<AD2> > > & params, const vector<string> & paramnames);
    void updateParameters(vector<Teuchos::RCP<vector<AD4> > > & params, const vector<string> & paramnames);
    void updateParameters(vector<Teuchos::RCP<vector<AD8> > > & params, const vector<string> & paramnames);
    void updateParameters(vector<Teuchos::RCP<vector<AD16> > > & params, const vector<string> & paramnames);
    void updateParameters(vector<Teuchos::RCP<vector<AD18> > > & params, const vector<string> & paramnames);
    void updateParameters(vector<Teuchos::RCP<vector<AD24> > > & params, const vector<string> & paramnames);
    void updateParameters(vector<Teuchos::RCP<vector<AD32> > > & params, const vector<string> & paramnames);
#endif

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    int getUniqueIndex(const int & set, const int & block, const std::string & var);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    template<class EvalT>
    void volumeResidual(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    template<class EvalT>
    void boundaryResidual(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    template<class EvalT>
    void computeFlux(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void setWorkset(vector<Teuchos::RCP<Workset<ScalarT> > > & wkset);
    
#ifndef MrHyDE_NO_AD
    void setWorkset(vector<Teuchos::RCP<Workset<AD> > > & wkset);
    void setWorkset(vector<Teuchos::RCP<Workset<AD2> > > & wkset);
    void setWorkset(vector<Teuchos::RCP<Workset<AD4> > > & wkset);
    void setWorkset(vector<Teuchos::RCP<Workset<AD8> > > & wkset);
    void setWorkset(vector<Teuchos::RCP<Workset<AD16> > > & wkset);
    void setWorkset(vector<Teuchos::RCP<Workset<AD18> > > & wkset);
    void setWorkset(vector<Teuchos::RCP<Workset<AD24> > > & wkset);
    void setWorkset(vector<Teuchos::RCP<Workset<AD32> > > & wkset);
#endif

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    bool checkFace(const size_t & set, const size_t & block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    template<class EvalT>
    void faceResidual(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    template<class EvalT>
    void fluxConditions(const size_t & set, const size_t block);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void updateFlags(vector<bool> & newflags);
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    vector<vector<vector<string> > > getVarList() {
      return var_list;
    }

    vector<vector<vector<string> > > getVarTypes() {
      return types;
    }

    vector<vector<vector<vector<string> > > > getDerivedList() {
      vector<vector<vector<vector<string> > > > dlist;
      for (size_t set=0; set<modules.size(); ++set) {
        vector<vector<vector<string> > > setlist;
        for (size_t blk=0; blk<modules[set].size(); ++blk) {
          vector<vector<string> > blklist;
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
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    void purgeMemory();
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Public data members
    /////////////////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<Teuchos::ParameterList> settings;
    Teuchos::RCP<MpiComm> comm;    
    int dimension, debug_level;
    vector<string> set_names, block_names, side_names;
    
    vector<vector<size_t> > num_vars; // [set][block]
    vector<int> num_derivs_required;
    
    vector<Teuchos::RCP<FunctionManager<ScalarT> > > function_managers; // always defined
#ifndef MrHyDE_NO_AD
    vector<Teuchos::RCP<FunctionManager<AD> > > function_managers_AD; // always defined for now for BW-compat
    vector<Teuchos::RCP<FunctionManager<AD2> > > function_managers_AD2;
    vector<Teuchos::RCP<FunctionManager<AD4> > > function_managers_AD4;
    vector<Teuchos::RCP<FunctionManager<AD8> > > function_managers_AD8;
    vector<Teuchos::RCP<FunctionManager<AD16> > > function_managers_AD16;
    vector<Teuchos::RCP<FunctionManager<AD18> > > function_managers_AD18;
    vector<Teuchos::RCP<FunctionManager<AD24> > > function_managers_AD24;
    vector<Teuchos::RCP<FunctionManager<AD32> > > function_managers_AD32;
#endif

    vector<vector<Teuchos::ParameterList>> physics_settings, disc_settings, solver_settings; // [set][block]
    vector<vector<vector<bool> > > use_subgrid;
    vector<vector<vector<bool> > > use_DG;
    vector<vector<vector<ScalarT> > > mass_wts, norm_wts;
    
    vector<vector<vector<string> > > var_list; // [set][block][var]
    vector<vector<vector<int> > > var_owned; // [set][block][var]
    vector<vector<vector<int> > > orders; // [set][block][var]
    vector<vector<vector<string> > > types; // [set][block][var]
    //-----------------------------------------------------
    
    vector<vector<int> > unique_orders; // [block][basis]
    vector<vector<string> > unique_types; // [block][basis]
    vector<vector<int> > unique_index; // [block][basis]
    
    string initial_type;
    
    vector<vector<string> > extra_fields_list, extra_cell_fields_list, response_list, target_list, weight_list;
    
    vector<vector<vector<Teuchos::RCP<PhysicsBase<ScalarT> > > > > modules; // always defined
#ifndef MrHyDE_NO_AD
    vector<vector<vector<Teuchos::RCP<PhysicsBase<AD> > > > > modules_AD; // always defined for now for BW-compat
    vector<vector<vector<Teuchos::RCP<PhysicsBase<AD2> > > > > modules_AD2;
    vector<vector<vector<Teuchos::RCP<PhysicsBase<AD4> > > > > modules_AD4;
    vector<vector<vector<Teuchos::RCP<PhysicsBase<AD8> > > > > modules_AD8;
    vector<vector<vector<Teuchos::RCP<PhysicsBase<AD16> > > > > modules_AD16;
    vector<vector<vector<Teuchos::RCP<PhysicsBase<AD18> > > > > modules_AD18;
    vector<vector<vector<Teuchos::RCP<PhysicsBase<AD24> > > > > modules_AD24;
    vector<vector<vector<Teuchos::RCP<PhysicsBase<AD32> > > > > modules_AD32;
#endif

  private:

    Teuchos::RCP<Teuchos::Time> bc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface::setBCData()");
    Teuchos::RCP<Teuchos::Time> dbc_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface::setDirichletData()");
    Teuchos::RCP<Teuchos::Time> side_info_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface::getSideInfo()");
    Teuchos::RCP<Teuchos::Time> response_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface:computeResponse()");
    Teuchos::RCP<Teuchos::Time> point_reponse_timer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::PhysicsInterface::computePointResponse()");
    
  };
  
}

#endif
