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
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"
#include "Teuchos_Time.hpp"
#include "Teuchos_TimeMonitor.hpp"

// Kokkos include
#include "Kokkos_Core.hpp"

//Epetra includes
#include "Epetra_Map.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Import.h"
#include "Epetra_Export.h"

//Tpetra includes
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Export.hpp"

// Sacado
#include "Sacado.hpp"

// Intrepid includes
// HGRAD functionality
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid2_HGRAD_QUAD_C1_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_C2_FEM.hpp"
#include "Intrepid2_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_C2_FEM.hpp"
#include "Intrepid2_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C1_FEM.hpp"
#include "Intrepid2_HGRAD_TET_C2_FEM.hpp"
#include "Intrepid2_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_C1_FEM.hpp"
#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid2_HVOL_C0_FEM.hpp"

// HDIV functionality
#include "Intrepid2_HDIV_QUAD_I1_FEM.hpp"
#include "Intrepid2_HDIV_QUAD_In_FEM.hpp"
#include "Intrepid2_HDIV_HEX_I1_FEM.hpp"
#include "Intrepid2_HDIV_HEX_In_FEM.hpp"
#include "Intrepid2_HDIV_TRI_I1_FEM.hpp"
#include "Intrepid2_HDIV_TRI_In_FEM.hpp"
#include "Intrepid2_HDIV_TET_I1_FEM.hpp"
#include "Intrepid2_HDIV_TET_In_FEM.hpp"

// HCURL functionality
#include "Intrepid2_HCURL_QUAD_I1_FEM.hpp"
#include "Intrepid2_HCURL_QUAD_In_FEM.hpp"
#include "Intrepid2_HCURL_HEX_I1_FEM.hpp"
#include "Intrepid2_HCURL_HEX_In_FEM.hpp"
#include "Intrepid2_HCURL_TRI_I1_FEM.hpp"
#include "Intrepid2_HCURL_TRI_In_FEM.hpp"
#include "Intrepid2_HCURL_TET_I1_FEM.hpp"
#include "Intrepid2_HCURL_TET_In_FEM.hpp"

// Intrepid2 extras
#include "Intrepid2_PointTools.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_ArrayTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Utils.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Panzer includes
#include "Panzer_ConnManager.hpp"
#include "Panzer_FieldPattern.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_DOFManager.hpp"

// Panzer_STK includes
#include "Panzer_STK_MeshFactory.hpp"
#include "Panzer_STK_LineMeshFactory.hpp"
#include "Panzer_STK_SquareQuadMeshFactory.hpp"
#include "Panzer_STK_SquareTriMeshFactory.hpp"
#include "Panzer_STK_CubeHexMeshFactory.hpp"
#include "Panzer_STK_CubeTetMeshFactory.hpp"
#include "Panzer_STKConnManager.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"
#include "Panzer_STK_PamgenReaderFactory.hpp"

// AztecOO
#include "AztecOO.h"

// ML includes
#include "ml_MultiLevelPreconditioner.h"

// Amesos includes
#include "Amesos.h"
#include "Amesos_BaseSolver.h"
#include "Amesos2.hpp"

// ROL includes
#include "ROL_LineSearchStep.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Bounds.hpp"
#include "ROL_TrustRegionStep.hpp"
#include "ROL_StdVector.hpp"

// Exodus includes
#include "exodusII.h"

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosTpetraAdapter.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_Utilities.hpp>

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#include "mpi.h"
#endif


#endif
