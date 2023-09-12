/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_TRILINOS_H
#define MRHYDE_TRILINOS_H

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

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"

#ifdef HAVE_MPI
//#include "Epetra_MpiComm.h"
#include "mpi.h"
#endif


#endif
