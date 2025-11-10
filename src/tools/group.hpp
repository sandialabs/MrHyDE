/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_GROUP_H
#define MRHYDE_GROUP_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"
#include "subgridModel.hpp"
#include "groupMetaData.hpp"
#include "discretizationInterface.hpp"
#include "compressedView.hpp"

#include <iostream>     
#include <iterator>     

namespace MrHyDE {
  
  class Group {

  public:
    
    // ========================================================================================
    // ========================================================================================
  
    Group() {} ;
  
    // ========================================================================================
    // ========================================================================================
  
    ~Group() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    Group(const Teuchos::RCP<GroupMetaData> & group_data_,
          const Kokkos::View<LO*,AssemblyDevice> localID_,
          Teuchos::RCP<DiscretizationInterface> & disc_,
          const bool & storeAll_);
    
    // ========================================================================================
    // ========================================================================================
  
    Group(const Teuchos::RCP<GroupMetaData> & group_data_,
          const Kokkos::View<LO*,AssemblyDevice> localID_,
          DRV nodes,
          Teuchos::RCP<DiscretizationInterface> & disc_,
          const bool & storeAll_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeFaceSize();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void initializeBasisIndex();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void computeBasis(const bool & keepnodes);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void createHostLIDs();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setIP();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setParams(LIDView paramLIDS_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Add the aux basis functions at the integration points.
    // This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void addAuxDiscretization(const vector<basis_RCP> & abasis_pointers, const vector<DRV> & abasis,
                              const vector<DRV> & abasisGrad, const vector<vector<DRV> > & asideBasis,
                              const vector<vector<DRV> > & asideBasisGrad);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Add the aux variables
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void addAuxVars(const vector<string> & auxlist_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Define which basis each variable will use
    ///////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Define which basis each variable will use.
     * 
     * @todo Is that really true? Seems like this allocates scalar storage for the solution
     * and required solution history.
     * 
     * @param[in] usebasis_ Which basis should each variable use for each physics set
     * @param[in] maxnumsteps  Maximum number of BDF steps for each physics set
     * @param[in] maxnumstages Maximum number of RK stages for each physics set
     * 
     */ 
    
    void setUseBasis(vector<vector<int> > & usebasis_, const vector<int> & maxnumsteps, 
                     const vector<int> & maxnumstages, const bool & allocate_storage = false);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Define which basis each discretized parameter will use
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setParamUseBasis(vector<int> & pusebasis_, vector<int> & paramnumbasis_,
                          const bool & allocate_storage = false);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Define which basis each aux variable will use
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setAuxUseBasis(vector<int> & ausebasis_, const bool & allocate_storage = false);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Reset the data stored in the previous step/stage solutions
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void resetPrevSoln(const size_t & set);
    
    // ========================================================================================
    // ========================================================================================
  
    void revertSoln(const size_t & set);
  
    // ========================================================================================
    // ========================================================================================
  
    void resetStageSoln(const size_t & set);
  
    // ========================================================================================
    // ========================================================================================
  
    void updateStageSoln(const size_t & set);
        
    ///////////////////////////////////////////////////////////////////////////////////////
    // Subgrid Plotting
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void writeSubgridSolution(const std::string & filename);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Subgrid Plotting
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void writeSubgridSolution(Teuchos::RCP<panzer_stk::STK_Interface> & globalmesh,
                              string & subblockname, bool & isTD, int & offset);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setUpSubGradient(const int & numParams) {
      if (group_data->requires_adjoint) {
        subgradient = View_Sc2("subgrid gradient",numElem,numParams);
      }
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Update the subgrid model
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void updateSubgridModel(vector<Teuchos::RCP<SubGridModel> > & models);
        
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    size_t getVolumetricStorage();
    
    // ========================================================================================
    // ========================================================================================
  
    size_t getFaceStorage();
  
    // ========================================================================================
    // ========================================================================================
  
    View_Sc2 getWts();

    // ========================================================================================
    // ========================================================================================
  
    vector<View_Sc2> getIntegrationPts();

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    // Public data
    
    // Data created elsewhere
    vector<LIDView> LIDs;
    LIDView paramLIDs, auxLIDs;
    
    // Creating LIDs on host device for host assembly
    vector<LIDView_host> LIDs_host;
    LIDView_host paramLIDs_host;
    
    Teuchos::RCP<GroupMetaData> group_data;
    
    vector<Teuchos::RCP<SubGridModel> > subgridModels;
    Kokkos::View<LO*,AssemblyDevice> localElemID;
    vector<Kokkos::View<int****,HostDevice> > sideinfo; // may need to move this to Assembly
    DRV nodes;
    vector<size_t> data_seed, data_seedindex;
    size_t subgrid_model_index; // which subgrid model is used for each time step
    size_t subgrid_usernum; // what is the index for this group in the subgrid model (should be deprecated)
    
    Teuchos::RCP<DiscretizationInterface> disc;
    
    // Data created here (Views should all be AssemblyDevice)
    size_t numElem;
    vector<CompressedView<View_Sc2> > ip;
    //View_Sc2 wts; 
    vector<vector<View_Sc2>> ip_face, normals_face;
    vector<View_Sc2> wts_face;
    vector<View_Sc1> hsize_face;
    Kokkos::View<LO*,AssemblyDevice> basis_index, ip_x_index, ip_y_index, ip_z_index;
    
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation;
    vector<View_Sc3> sol, phi;
    View_Sc3 param, aux; // (elem,var,numdof)
    vector<View_Sc3> sol_avg, sol_alt;
    View_Sc3 param_avg, aux_avg; // (elem,var,dim)
    vector<View_Sc4> sol_prev, phi_prev, aux_prev, sol_stage, phi_stage, aux_stage; // (elem,var,numdof,step or stage)
    
    // basis information
    vector<CompressedView<View_Sc4>> basis, basis_grad, basis_curl, basis_nodes;
    vector<CompressedView<View_Sc3>> basis_div, local_mass, local_jacobian;
    CompressedView<View_Sc3> local_param_mass;
    CompressedView<View_Sc2> wts;

    vector<vector<CompressedView<View_Sc4>>> basis_face, basis_grad_face;
    View_Sc1 hsize;
    
    // Aux variable Information
    vector<string> auxlist;
    Kokkos::View<LO**,AssemblyDevice> auxoffsets;
    vector<int> auxusebasis;
    vector<basis_RCP> auxbasisPointers;
    vector<DRV> auxbasis, auxbasisGrad; // this does cause a problem
    vector<vector<DRV> > auxside_basis, auxside_basisGrad;
    
    // Storage information
    bool active, storeAll, storeMass, usealtsol = false, haveBasis, have_ip = false, have_sols = false, have_nodes;

    Kokkos::View<ScalarT**,AssemblyDevice> subgradient, data;
    vector<ScalarT> data_distance;
    
  };
  
}

#endif
