/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_SUBGRIDTOOLS2_H
#define MRHYDE_SUBGRIDTOOLS2_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "discretizationInterface.hpp"

namespace MrHyDE {
  
  class SubGridTools2 {
  public:
    
    SubGridTools2() {} ;
    
    ~SubGridTools2() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    SubGridTools2(const Teuchos::RCP<MpiComm> & local_comm, const string & shape,
                 const string & subshape, const DRV nodes,
                 std::string & mesh_type, std::string & mesh_file);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Given the coarse grid nodes and shape, define the subgrid nodes, connectivity, and sideinfo
    //////////////////////////////////////////////////////////////////////////////////////
    
    void createSubMesh(const int & numrefine);
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Uniformly refine an element
    //////////////////////////////////////////////////////////////////////////////////////
    
    void refineSubCell(const int & e);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Check if a sub-grid nodes has already been added to the list
    ///////////////////////////////////////////////////////////////////////////////////////
    
    bool checkExistingSubNodes(const vector<ScalarT> & newpt,
                               const ScalarT & tol, int & index);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the sub-grid nodes as a list: output is (Nnodes x dimension)
    ///////////////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<ScalarT**,HostDevice> getListOfPhysicalNodes(DRV newmacronodes, topo_RCP & macro_topo,
                                                              Teuchos::RCP<DiscretizationInterface> & disc);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the sub-grid nodes on each element: output is (Nelem x Nnperelem x dimension)
    ///////////////////////////////////////////////////////////////////////////////////////
    
    DRV getPhysicalNodes(DRV newmacronodes, topo_RCP & macro_topo,
                         Teuchos::RCP<DiscretizationInterface> & disc);
        
    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the sub-grid connectivity
    ///////////////////////////////////////////////////////////////////////////////////////
    
    vector<vector<GO> > getPhysicalConnectivity(int & reps);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the boundary groups
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void getBoundaryGroups(size_t & numMacro, vector<vector<size_t> > & boundary_groups);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    size_t getNumRefNodes();

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
  private:

    int dimension_, num_macro_sides_;
    Teuchos::RCP<MpiComm> local_comm_;
    string shape_, subshape_, mesh_type_, mesh_file_;
    Kokkos::View<ScalarT**,HostDevice> nodes_;
    vector<vector<ScalarT> > subnodes_;
    DRV subnodes_list_;
    vector<Kokkos::View<bool*,HostDevice> > subsidemap_;
    vector<vector<GO> > subconnectivity_;
    
    Teuchos::RCP<panzer_stk::STK_Interface> ref_mesh_; // used for Exodus and panzer meshes
    
  };
}

#endif

