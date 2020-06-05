/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef subgridMesh 
#define subgridMesh 

#include <Panzer_STK_MeshFactory.hpp>
#include <Panzer_STK_Interface.hpp>
#include "trilinos.hpp"
#include "preferences.hpp"

typedef double ScalarT;

namespace panzer_stk {
  
  class STK_Interface;
  
  class SubGridMeshFactory : public STK_MeshFactory {
  public:
    //! Constructor
    
    SubGridMeshFactory(const std::string & shape_) {
      shape = shape_;
    }
    
    SubGridMeshFactory(const std::string & shape_, //const shards::CellTopology & cellTopo_,
                       std::vector<std::vector<ScalarT> > & nodes_,
                       std::vector<std::vector<GO> > & conn_, std::string & blockname_)
    {
      shape = shape_;
      blockname = blockname_;
      nodes.push_back(nodes_);
      conn.push_back(conn_);
      dimension = nodes[0][0].size();
    }
    
    
    //! Destructor
    virtual ~SubGridMeshFactory();
    
    // Add block
    void addElems(std::vector<std::vector<ScalarT> > & newnodes, std::vector<std::vector<GO> > & newconn);
    
    //! Build the mesh object
    Teuchos::RCP<STK_Interface> buildMesh(stk::ParallelMachine parallelMach) const;
    virtual void completeMeshConstruction(STK_Interface & mesh,stk::ParallelMachine parallelMach) const;
    
    Teuchos::RCP<STK_Interface> buildUncommitedMesh(stk::ParallelMachine parallelMach) const;
    
    //! From ParameterListAcceptor
    void setParameterList(const Teuchos::RCP<Teuchos::ParameterList> & paramList);
    
    //! From ParameterListAcceptor
    Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;
    
    //! what is the 2D tuple describe this processor distribution
    Teuchos::Tuple<std::size_t,2> procRankToProcTuple(std::size_t procRank) const;
   
    
  protected:
    
    std::string shape;
    std::string blockname;
    std::vector<std::vector<std::vector<ScalarT> > > nodes;
    std::vector<std::vector<std::vector<GO> > > conn;
    int dimension;
    
  };
  
}

#endif
