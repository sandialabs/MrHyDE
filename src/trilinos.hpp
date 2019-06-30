/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef TRILINOS_H
#define TRILINOS_H

// This file only contains the core Trilinos headers
// Individual files/interfaces may use other Trilinos tools

// STL includes
#include <iostream>
#include <vector>
#include <set>
#include <stdio.h>
#include <random>

// Teuchos includes
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Time.hpp"
#include "Teuchos_TimeMonitor.hpp"

// Kokkos include
#include "Kokkos_Core.hpp"
#include "kokkosTools.hpp"

// Sacado
#include "Sacado.hpp"

//Tpetra includes
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Export.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

#ifdef HAVE_MPI
//#include "Epetra_MpiComm.h"
#include "mpi.h"
#endif


#endif
