/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
 ************************************************************************/

#ifndef MRHYDE_SUBGRIDTOOLS_H
#define MRHYDE_SUBGRIDTOOLS_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "discretizationInterface.hpp"

namespace MrHyDE {
  
  class SubGridTools {
  public:
    
    SubGridTools() {} ;
    
    ~SubGridTools() {} ;
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    SubGridTools(const Teuchos::RCP<MpiComm> & local_comm, const string & shape,
                 const string & subshape, const DRV nodes,
                 Kokkos::View<int****,HostDevice> sideinfo,
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
    // Get the sub-grid side info
    ///////////////////////////////////////////////////////////////////////////////////////
    
    Kokkos::View<int****,HostDevice> getPhysicalSideinfo(Kokkos::View<int****,HostDevice> macrosideinfo);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the sub-grid connectivity
    ///////////////////////////////////////////////////////////////////////////////////////
    
    vector<vector<GO> > getPhysicalConnectivity(int & reps);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // Get the unique subgrid side names and indices
    ///////////////////////////////////////////////////////////////////////////////////////
    
    void getUniqueSides(Kokkos::View<int****,HostDevice> & newsi, vector<int> & unique_sides,
                        vector<int> & unique_local_sides, vector<string> & unique_names,
                        vector<string> & macrosidenames,
                        vector<vector<size_t> > & boundary_groups);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    size_t getNumRefNodes();

    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
  private:

    int dimension_;
    Teuchos::RCP<MpiComm> local_comm_;
    string shape_, subshape_, mesh_type_, mesh_file_;
    Kokkos::View<ScalarT**,HostDevice> nodes_;
    Kokkos::View<int****,HostDevice> sideinfo_;
    vector<vector<ScalarT> > subnodes_;
    DRV subnodes_list_;
    vector<Kokkos::View<int**,HostDevice> > subsidemap_;
    vector<vector<GO> > subconnectivity_;
    
    Teuchos::RCP<panzer_stk::STK_Interface> ref_mesh_; // used for Exodus and panzer meshes
    
  };
}

#endif

