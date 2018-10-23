/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include <subgridMeshFactory.hpp>

namespace panzer_stk{
  
  //! Destructor
  SubGridMeshFactory::~SubGridMeshFactory() {};
  
  // Add block
  void SubGridMeshFactory::addElems(std::vector<std::vector<double> > & newnodes,
                std::vector<std::vector<int> > & newconn) {
    nodes.push_back(newnodes);
    conn.push_back(newconn);
    dimension = nodes[0][0].size();
  }
  
  
  //! Build the mesh object
  Teuchos::RCP<STK_Interface> SubGridMeshFactory::buildMesh(stk::ParallelMachine parallelMach) const
  {
    
    Teuchos::RCP<STK_Interface> mesh = Teuchos::rcp(new STK_Interface(dimension));
    
    shards::CellTopology cellTopo;
    if (dimension == 1) {
      cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() );// lin. cell topology on the interior
    }
    if (dimension == 2) {
      if (shape == "quad") {
        cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() );// lin. cell topology on the interior
      }
      if (shape == "tri") {
        cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() );// lin. cell topology on the interior
      }
    }
    if (dimension == 3) {
      if (shape == "hex") {
        cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<> >() );// lin. cell topology on the interior
      }
      if (shape == "tet") {
        cellTopo = shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<> >() );// lin. cell topology on the interior
      }
      
    }
    
    // build meta information: blocks and side set setups
    const CellTopologyData * ctd = cellTopo.getCellTopologyData();
    for (size_t b=0; b<blockname.size(); b++) {
      mesh->addElementBlock(blockname,ctd);
    }
    // add sidesets (not really needed)
    
    // commit meta data
    //mesh->initialize(parallelMach);
    
    return mesh;
  }
  
  void SubGridMeshFactory::completeMeshConstruction(STK_Interface & mesh,stk::ParallelMachine parallelMach) const {
    
    mesh.initialize(parallelMach);
    mesh.beginModification();
    
    int nprog = 0;
    
    // build the nodes
    for (int b=0; b<nodes.size(); b++) {
      for(int p=0; p<nodes[b].size(); p++) {
        mesh.addNode(nprog+1,nodes[b][p]);
        nprog++;
      }
    }
    
    //std::cout << conn.size() << "  " << conn[0].size() << " " << conn[0][0].size() << std::endl;
    //std::cout << "cmc 0 " << nprog << std::endl;
    
    // build the elements
    stk::mesh::Part * block = mesh.getElementBlockPart(blockname);
    
    int eprog = 0;
    int cprog = 1;
    for (int b=0; b<conn.size(); b++) {
      int ccprog = 0;
      for(int e=0; e<conn[b].size(); e++) {
        stk::mesh::EntityId gid = eprog+1;
        eprog++;
        std::vector<stk::mesh::EntityId> elem(conn[b][e].size());
        //std::cout << "elem size = " << elem.size() << std::endl;
        
        for (int j=0; j<conn[b][e].size(); j++) {
          elem[j] = conn[b][e][j]+cprog;
          //std::cout << b << " " << e << " " << elem[j]<< std::endl;
        }
        //std::cout << "herehere 0" << std::endl;
        ccprog++;
        //cprog += conn[b][e].size()-1;
        Teuchos::RCP<ElementDescriptor> ed = Teuchos::rcp(new ElementDescriptor(gid,elem));
        //std::cout << "herehere 1" << std::endl;
        mesh.addElement(ed,block);
        //std::cout << "herehere 2" << std::endl;
        
      }
      cprog += ccprog;
      //std::cout << cprog << std::endl;
    }
    
    //std::cout << "cmc 1" << std::endl;
    
    mesh.endModification();
    
    // build bulk data
    mesh.buildLocalElementIDs();
    
    // calls Stk_MeshFactory::rebalance
    this->rebalance(mesh);
    
  }
  
  Teuchos::RCP<STK_Interface> SubGridMeshFactory::buildUncommitedMesh(stk::ParallelMachine parallelMach) const {};
  
  //! From ParameterListAcceptor
  void SubGridMeshFactory::setParameterList(const Teuchos::RCP<Teuchos::ParameterList> & paramList) {};
  
  //! From ParameterListAcceptor
  Teuchos::RCP<const Teuchos::ParameterList> SubGridMeshFactory::getValidParameters() const {};
  
  //! what is the 2D tuple describe this processor distribution
  Teuchos::Tuple<std::size_t,2> SubGridMeshFactory::procRankToProcTuple(std::size_t procRank) const {};
}

