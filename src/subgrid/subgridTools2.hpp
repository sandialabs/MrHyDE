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

#ifndef SUBGRIDTOOLS2_H
#define SUBGRIDTOOLS2_H

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
    
    SubGridTools2(const Teuchos::RCP<MpiComm> & LocalComm_, const string & shape_,
                 const string & subshape_, const DRV nodes_,
                 std::string & mesh_type_, std::string & mesh_file_);
    
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
    // Get the unique subgrid side names and indices
    ///////////////////////////////////////////////////////////////////////////////////////
    /*
    void getUniqueSides(vector<int> & unique_sides,
                        vector<int> & unique_local_sides, vector<string> & unique_names,
                        vector<string> & macrosidenames,
                        vector<vector<size_t> > & boundary_groups);
    */
    
    void getBoundaryGroups(size_t & numMacro, vector<vector<size_t> > & boundary_groups);
    
    ///////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    int dimension, numMacroSides;
    Teuchos::RCP<MpiComm> LocalComm;
    string shape, subshape, mesh_type, mesh_file;
    Kokkos::View<ScalarT**,HostDevice> nodes;
    vector<vector<ScalarT> > subnodes;
    DRV subnodes_list;
    vector<Kokkos::View<bool*,HostDevice> > subsidemap;
    vector<vector<GO> > subconnectivity;
    
    Teuchos::RCP<panzer_stk::STK_Interface> ref_mesh; // used for Exodus and panzer meshes
    
  };
}

#endif

