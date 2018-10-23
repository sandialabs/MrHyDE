/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
************************************************************************/

#ifndef PREFERENCES_H
#define PREFERENCES_H

using namespace std;
using namespace Intrepid2;
using Kokkos::parallel_for;
using Kokkos::RangePolicy;

#define MILO_VERSION "1.0"

typedef double ScalarT;
typedef double RealType;

#define maxDerivs 64 // adjust this to improve performance
#define PI 3.141592653589793238463
#define MILO_DEBUG false

// AD typedefs
typedef Sacado::Fad::DFad<double> DFAD; // used only when absolutely necessary
typedef Sacado::Fad::SFad<double,maxDerivs> AD;

// Kokkos Device typedefs
typedef Kokkos::Serial AssemblyDevice;
typedef Kokkos::Serial HostDevice;
typedef Kokkos::Serial SubgridDevice;

// Kokkos object typedefs (preferable to use Kokkos::View<*,Device>)
typedef Kokkos::DynRankView<double,AssemblyDevice> DRV;
typedef Kokkos::DynRankView<int,AssemblyDevice> DRVint;
typedef Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> FDATA;
typedef Kokkos::View<double**,Kokkos::LayoutStride,AssemblyDevice> FDATAd;

// Intrepid and shards typedefs
typedef Teuchos::RCP<Intrepid2::Basis<AssemblyDevice, double, double > > basis_RCP;
typedef Teuchos::RCP<const shards::CellTopology> topo_RCP;

// Linear algebra typedefs
typedef Epetra_MultiVector   LA_MultiVector;
typedef Epetra_CrsMatrix     LA_CrsMatrix;
typedef Epetra_Map           LA_Map;
typedef Epetra_CrsGraph      LA_CrsGraph;
typedef Epetra_Export        LA_Export;
typedef Epetra_Import        LA_Import;
typedef Epetra_MpiComm       LA_MpiComm;
typedef Epetra_LinearProblem LA_LinearProblem;

/* // Tpetra typedefs (not used yet)
typedef Tpetra_MultiVector LA_vector;
typedef Tpetra_CrsMatrix   LA_matrix;
typedef Tpetra_MultiVector LA_vector;
typedef Tpetra_Export      LA_export;
typedef Tpetra_Import      LA_import;
*/

// RCP to LA objects (may be removed in later version)
typedef Teuchos::RCP<LA_MultiVector> vector_RCP;
typedef Teuchos::RCP<LA_CrsMatrix>   matrix_RCP;

// Class for printing and working with Kokkos::View and DRV
#include "kokkosTools.hpp"

#endif
