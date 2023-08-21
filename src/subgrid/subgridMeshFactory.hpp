/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
 ************************************************************************/

#ifndef MRHYDE_subgridMesh 
#define MRHYDE_subgridMesh 

#include <Panzer_STK_MeshFactory.hpp>
#include <Panzer_STK_Interface.hpp>
#include "trilinos.hpp"
#include "preferences.hpp"

namespace panzer_stk {
  
  class STK_Interface;
  
  class SubGridMeshFactory : public STK_MeshFactory {
  public:
    //! Constructor
    
    SubGridMeshFactory(const std::string & shape) {
      shape_ = shape;
    }
    
    SubGridMeshFactory(const std::string & shape,
                       Kokkos::View<ScalarT**,HostDevice> nodes,
                       std::vector<std::vector<GO> > & conn,
                       std::string & blockname)
    {
      shape_ = shape;
      blockname_ = blockname;
      nodes_.push_back(nodes);
      conn_.push_back(conn);
      dimension_ = nodes.extent(1);
    }
    
    
    //! Destructor
    virtual ~SubGridMeshFactory();
    
    // Add block
    void addElems(DRV newnodes,
                  std::vector<std::vector<GO> > & newconn);
    
    //! Build the mesh object
    Teuchos::RCP<STK_Interface> buildMesh(stk::ParallelMachine parallelMach) const;
    virtual void completeMeshConstruction(STK_Interface & mesh, stk::ParallelMachine parallelMach) const;
    
    void modifyMesh(STK_Interface & mesh) const;
    
    Teuchos::RCP<STK_Interface> buildUncommitedMesh(stk::ParallelMachine parallelMach) const;
    
    //! From ParameterListAcceptor
    void setParameterList(const Teuchos::RCP<Teuchos::ParameterList> & paramList);
    
    //! From ParameterListAcceptor
    Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;
        
  private:
    
    std::string shape_;
    std::string blockname_;
    std::vector<Kokkos::View<ScalarT**,HostDevice> > nodes_;
    std::vector<std::vector<std::vector<GO> > > conn_;
    int dimension_;
    
  };
  
}

#endif
