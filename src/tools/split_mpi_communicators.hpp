/***********************************************************************
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

class SplitComm {
  public:
  
  SplitComm() {};
  
  ~SplitComm() {};
  
  SplitComm(Teuchos::RCP<Teuchos::ParameterList> & settings,
            LA_MpiComm & Comm,
            Teuchos::RCP<LA_MpiComm> & tcomm_LA,
            Teuchos::RCP<LA_MpiComm> & tcomm_S);
  
  void split_mpi_communicators( Teuchos::RCP<LA_MpiComm> & Comm_linalg,
                               Teuchos::RCP<LA_MpiComm> & Comm_collocation,
                               int rank, int M, int Ngroups );
  
  void split_mpi_communicators(LA_MpiComm & Comm_orig,
                               Teuchos::RCP<LA_MpiComm> & Comm_main,
                               Teuchos::RCP<LA_MpiComm> & Comm_ms,
                               int myrank, int Nmain , bool & include_main);

};

#endif
