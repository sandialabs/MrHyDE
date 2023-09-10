/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_SPLITCOMM_H
#define MRHYDE_SPLITCOMM_H

#include "trilinos.hpp"
#include "preferences.hpp"

namespace MrHyDE {
  
  class SplitComm {
  public:
    
    SplitComm() {};
    
    ~SplitComm() {};
    
    SplitComm(Teuchos::RCP<Teuchos::ParameterList> & settings,
              MpiComm & Comm,
              Teuchos::RCP<MpiComm> & tcomm_LA,
              Teuchos::RCP<MpiComm> & tcomm_S);
    
    void split_mpi_communicators( Teuchos::RCP<MpiComm> & Comm_linalg,
                                 Teuchos::RCP<MpiComm> & Comm_collocation,
                                 int rank, int M, int Ngroups );
    
    void split_mpi_communicators(MpiComm & Comm_orig,
                                 Teuchos::RCP<MpiComm> & Comm_main,
                                 Teuchos::RCP<MpiComm> & Comm_ms,
                                 int myrank, int Nmain , bool & include_main);
    
  };
  
}

#endif
