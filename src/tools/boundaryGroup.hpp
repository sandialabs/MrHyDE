/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_BOUNDARYGROUP_H
#define MRHYDE_BOUNDARYGROUP_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "workset.hpp"
#include "groupMetaData.hpp"
#include "discretizationInterface.hpp"
#include "compressedView.hpp"

#include <iostream>     
#include <iterator>     

namespace MrHyDE {
  
  class BoundaryGroup {
    
    typedef Tpetra::MultiVector<ScalarT,LO,GO,AssemblyNode> SG_MultiVector;
    typedef Teuchos::RCP<SG_MultiVector> SG_vector_RCP;
    
  public:
    
    // ========================================================================================
    // ========================================================================================
    
    BoundaryGroup() {} ;
    
    // ========================================================================================
    // ========================================================================================
    
    ~BoundaryGroup() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    BoundaryGroup(const Teuchos::RCP<GroupMetaData> & group_data_,
                  const Kokkos::View<LO*,AssemblyDevice> localID_,
                  LO & sideID_,
                  const int & sidenum_, const string & sidename_,
                  const int & groupID_,
                  Teuchos::RCP<DiscretizationInterface> & disc_,
                  const bool & storeAll_);

    // ========================================================================================
    // ========================================================================================
    
    BoundaryGroup(const Teuchos::RCP<GroupMetaData> & group_data_,
                  const Kokkos::View<LO*,AssemblyDevice> localID_,
                  DRV nodes_, LO & sideID_,
                  const int & sidenum_, const string & sidename_,
                  const int & groupID_,
                  Teuchos::RCP<DiscretizationInterface> & disc_,
                  const bool & storeAll_);

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////

    void computeSize();

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
    
    void computeSizeNormals();
    
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void setParams(LIDView paramLIDs_);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Add the aux basis functions at the integration points.
    // This version assumes the basis functions have been evaluated elsewhere (as in multiscale)
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void addAuxDiscretization(const vector<basis_RCP> & abasis_pointers,
                              const vector<DRV> & asideBasis,
                              const vector<DRV> & asideBasisGrad);
    
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
    // Map the coarse grid solution to the fine grid integration points
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
    // Compute boundary contribution to the regularization and nodes located on the boundary
    ///////////////////////////////////////////////////////////////////////////////////////
    
    AD computeBoundaryRegularization(const vector<ScalarT> reg_constants, const vector<int> reg_types,
                                     const vector<int> reg_indices, const vector<string> reg_sides);
        
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    size_t getStorage();
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
      
    // Public data 
    Teuchos::RCP<GroupMetaData> group_data;
    
    Kokkos::View<LO*,AssemblyDevice> localElemID;
    LO localSideID;
    Kokkos::DynRankView<Intrepid2::Orientation,PHX::Device> orientation;
    
    // Geometry Information
    size_t numElem = 0; // default value ... used to check if proc. has elements on boundary
    int sidenum, groupID, wksetBID;
    DRV nodes;
    vector<View_Sc2> ip, normals, tangents;
    View_Sc2 wts;
    View_Sc1 hsize;
    bool storeAll, haveBasis, have_sols = false, have_nodes;
    Kokkos::View<LO*,AssemblyDevice> basis_index;
    
    vector<Kokkos::View<int****,HostDevice> > sideinfo; // may need to move this to Assembly
    string sidename;
        
    // DOF information
    LIDView paramLIDs, auxLIDs;
    vector<LIDView> LIDs;
    
    Teuchos::RCP<DiscretizationInterface> disc;
    
    // Creating LIDs on host device for host assembly
    LIDView_host paramLIDs_host, auxLIDs_host;
    vector<LIDView_host> LIDs_host;
    
    vector<View_Sc3> sol, phi;
    View_Sc3 param, aux;
    
    vector<View_Sc4> sol_prev, phi_prev, sol_stage, phi_stage; // (elem,var,numdof,step or stage)
    
    // basis information
    vector<CompressedView<View_Sc4>> basis, basis_grad, basis_curl;
    vector<CompressedView<View_Sc3>> basis_div;
    
    // Aux variable Information
    vector<string> auxlist;
    Kokkos::View<LO**,AssemblyDevice> auxoffsets;
    vector<int> auxusebasis;
    vector<basis_RCP> auxbasisPointers;
    vector<DRV> auxbasis, auxbasisGrad;
    vector<DRV> auxside_basis, auxside_basisGrad;
    vector<size_t> auxMIDs;
    Kokkos::View<size_t*,AssemblyDevice> auxMIDs_dev;
    
    vector<size_t> data_seed, data_seedindex;
    View_Sc2 data;
    vector<ScalarT> data_distance;
    View_Sc3 multidata;
    
  };
  
}

#endif
