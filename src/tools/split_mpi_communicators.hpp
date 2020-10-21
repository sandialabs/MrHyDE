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

#ifndef SPLITCOMM_H
#define SPLITCOMM_H

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
