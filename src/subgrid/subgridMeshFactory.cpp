/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include <subgridMeshFactory.hpp>

namespace panzer_stk{
  
  ///////////////////////////////////////////////////////////////
  //! Destructor
  ///////////////////////////////////////////////////////////////
  
  SubGridMeshFactory::~SubGridMeshFactory() {}
  
  ///////////////////////////////////////////////////////////////
  // Add block
  ///////////////////////////////////////////////////////////////
  
  void SubGridMeshFactory::addElems(DRV newnodes,
                                    std::vector<std::vector<GO> > & newconn) {
    //nodes.push_back(newnodes);
    conn_.push_back(newconn);
    dimension_ = newnodes.extent(1);
    
    Kokkos::View<ScalarT**,HostDevice> newnodes_host("new nodes",
                                                     newnodes.extent(0), newnodes.extent(1));
    auto nodes_host = Kokkos::create_mirror_view(newnodes);
    Kokkos::deep_copy(nodes_host,newnodes);
    Kokkos::deep_copy(newnodes_host,nodes_host);
    nodes_.push_back(newnodes_host);
  }
  
  ///////////////////////////////////////////////////////////////
  //! Build the mesh object
  ///////////////////////////////////////////////////////////////
  
  Teuchos::RCP<STK_Interface> SubGridMeshFactory::buildMesh(stk::ParallelMachine parallelMach) const
  {
    
    Teuchos::RCP<STK_Interface> mesh = Teuchos::rcp(new STK_Interface(dimension_));
    
    shards::CellTopology cellTopo;
    if (dimension_ == 1) {
      cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );// lin. cell topology on the interior
    }
    if (dimension_ == 2) {
      if (shape_ == "quad") {
        cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() );// lin. cell topology on the interior
      }
      if (shape_ == "tri") {
        cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() );// lin. cell topology on the interior
      }
    }
    if (dimension_ == 3) {
      if (shape_ == "hex") {
        cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<> >() );// lin. cell topology on the interior
      }
      if (shape_ == "tet") {
        cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<> >() );// lin. cell topology on the interior
      }
      
    }
    
    // build meta information: blocks and side set setups
    const CellTopologyData * ctd = cellTopo.getCellTopologyData();
    for (size_t b=0; b<blockname_.size(); b++) {
      mesh->addElementBlock(blockname_,ctd);
    }
    // add sidesets (not really needed)
    
    // commit meta data
    //mesh->initialize(parallelMach);
    
    return mesh;
  }
  
  ///////////////////////////////////////////////////////////////
  // Finalize
  ///////////////////////////////////////////////////////////////
  
  void SubGridMeshFactory::completeMeshConstruction(STK_Interface & mesh,
                                                    stk::ParallelMachine parallelMach) const {
    
    mesh.initialize(parallelMach);
    
    this->modifyMesh(mesh);
    
    // build bulk data
    mesh.buildSubcells();
    
    mesh.buildLocalElementIDs();
    
    // calls Stk_MeshFactory::rebalance
    this->rebalance(mesh);
  }
  
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
  
  void SubGridMeshFactory::modifyMesh(STK_Interface & mesh) const {
    
    mesh.beginModification();
    
    int nprog = 0;
    
    // build the nodes
    for (size_t b=0; b<nodes_.size(); b++) {
      for (size_type e=0; e<nodes_[b].extent(0); e++) {
        vector<double> newnode;
        for (size_type n=0; n<nodes_[b].extent(1); n++) {
          newnode.push_back(static_cast<double>(nodes_[b](e,n)));
        }
        mesh.addNode(nprog+1,newnode);
        nprog++;
      }
    }
    
    // build the elements
    stk::mesh::Part * block = mesh.getElementBlockPart(blockname_);
    
    int eprog = 0;
    int cprog = 1;
    for (size_t b=0; b<conn_.size(); b++) {
      int ccprog = 0;
      for (size_t e=0; e<conn_[b].size(); e++) {
        stk::mesh::EntityId gid = eprog+1;
        eprog++;
        std::vector<stk::mesh::EntityId> elem_(conn_[b][e].size());
        
        for (size_t j=0; j<conn_[b][e].size(); j++) {
          elem_[j] = conn_[b][e][j]+cprog;
        }
        ccprog++;
        Teuchos::RCP<ElementDescriptor> ed = Teuchos::rcp(new ElementDescriptor(gid,elem_));
        mesh.addElement(ed,block);
        
      }
      cprog += ccprog;
    }
    
    mesh.endModification();
    
  }
  
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
  
  Teuchos::RCP<STK_Interface> SubGridMeshFactory::buildUncommitedMesh(stk::ParallelMachine parallelMach) const {
    Teuchos::RCP<STK_Interface> mesh = Teuchos::rcp(new STK_Interface(dimension_));
    return mesh;
  }
  
  ///////////////////////////////////////////////////////////////
  //! From ParameterListAcceptor
  ///////////////////////////////////////////////////////////////
  
  void SubGridMeshFactory::setParameterList(const Teuchos::RCP<Teuchos::ParameterList> & paramList) {}
  
  ///////////////////////////////////////////////////////////////
  //! From ParameterListAcceptor
  ///////////////////////////////////////////////////////////////
  
  Teuchos::RCP<const Teuchos::ParameterList> SubGridMeshFactory::getValidParameters() const {
    Teuchos::RCP<Teuchos::ParameterList> validparams;
    return validparams;
  }
  
}

