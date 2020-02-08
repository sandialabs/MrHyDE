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

#include "PanzerCore_config.hpp"
#include "Intrepid2_Basis.hpp"

using namespace std;
using Kokkos::parallel_for;
using Kokkos::RangePolicy;

#define MILO_VERSION "1.0"

typedef double ScalarT;
typedef int LO;
typedef panzer::GlobalOrdinal GO; // this should really be panzer::GlobalOrdinal

#define maxDerivs 64 // adjust this to improve performance
#define PI 3.141592653589793238463
#define MILO_DEBUG false
typedef Teuchos::MpiComm<int> MpiComm;

// AD typedefs
// For implicit time integration
typedef Sacado::Fad::DFad<ScalarT> DFAD; // used only when absolutely necessary
typedef Sacado::Fad::SFad<ScalarT,maxDerivs> AD;

// For explicit time integration
//typedef ScalarT DFAD; // used only when absolutely necessary
//typedef ScalarT AD;

// Kokkos Device typedefs
typedef Kokkos::Serial AssemblyDevice;
typedef Kokkos::Serial HostDevice;
typedef Kokkos::Serial SubgridDevice;
typedef Kokkos::Compat::KokkosSerialWrapperNode HostNode;
typedef Kokkos::Compat::KokkosSerialWrapperNode SubgridNode;
//typedef Kokkos::Compat::KokkosOpenMPWrapperNode HostNode;
//typedef Kokkos::Compat::KokkosThreadsWrapperNode HostNode;
//typedef Kokkos::Compat::KokkosCudaWrapperNode HostNode;

// Kokkos object typedefs (preferable to use Kokkos::View<*,Device>)
typedef Kokkos::DynRankView<ScalarT,AssemblyDevice> DRV;
typedef Kokkos::DynRankView<int,AssemblyDevice> DRVint;
typedef Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> FDATA;
typedef Kokkos::View<ScalarT**,Kokkos::LayoutStride,AssemblyDevice> FDATAd;

// Intrepid and shards typedefs
//typedef Teuchos::RCP<Intrepid2::Basis<AssemblyDevice, ScalarT, ScalarT > > basis_RCP;
typedef Teuchos::RCP<const shards::CellTopology> topo_RCP;
typedef Teuchos::RCP<Intrepid2::Basis<AssemblyDevice, ScalarT, ScalarT > > basis_RCP;

// Tpetra linear algebra typedefs (Epetra is non longer supported)
typedef Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>   LA_CrsMatrix;
typedef Tpetra::CrsGraph<LO,GO,HostNode>            LA_CrsGraph;
typedef Tpetra::Export<LO, GO, HostNode>            LA_Export;
typedef Tpetra::Import<LO, GO, HostNode>            LA_Import;
typedef Tpetra::Map<LO, GO, HostNode>               LA_Map;
typedef Tpetra::Operator<ScalarT,LO,GO,HostNode>    LA_Operator;
typedef Tpetra::MultiVector<ScalarT,LO,GO,HostNode> LA_MultiVector;
//typedef Belos::LinearProblem<ScalarT, LA_MultiVector, LA_Operator> LA_LinearProblem;


// RCP to LA objects (may be removed in later version)
typedef Teuchos::RCP<LA_MultiVector> vector_RCP;
typedef Teuchos::RCP<LA_CrsMatrix>   matrix_RCP;

#endif
