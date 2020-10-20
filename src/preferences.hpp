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
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"
#include "Phalanx_DataLayout.hpp"

using namespace std;
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::RangePolicy;

#define MILO_VERSION "1.0"

typedef double ScalarT;
typedef int LO;
typedef panzer::GlobalOrdinal GO; // this should really be panzer::GlobalOrdinal

#ifdef MrHyDE_SET_MAX_DERIVS
#define maxDerivs MrHyDE_SET_MAX_DERIVS // allow us to set this at configure-time with the MrHyDE_MAX_DERIVS flag
#else
  #define maxDerivs 24 // adjust this to improve performance
#endif


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

// Kokkos Execution Space typedefs
// Format: Kokkos::*
// Options: Serial, OpenMP, Threads, Cuda
typedef Kokkos::Serial HostExec; // cannot be Cuda right now
//#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
//  typedef Kokkos::Cuda AssemblyExec;
//#else
  typedef Kokkos::Serial AssemblyExec;
//#endif
#if defined(MrHyDE_SUBGRIDSPACE_CUDA)
  typedef Kokkos::Cuda SubgridExec;
#else
  typedef Kokkos::Serial SubgridExec;
#endif

// Kokkos Memory Space typedefs
// Format: Kokkos::*
// Options: HostSpace, CudaSpace, CudaUVMSpace
typedef Kokkos::HostSpace HostMem; // cannot be CudaSpace right now
//#if defined(MrHyDE_ASSEMBLYMEM_CUDAUVM)
//  typedef Kokkos::CudaUVMSpace AssemblyMem;
//#else
  typedef Kokkos::HostSpace AssemblyMem;
//#endif
#if defined(MrHyDE_SUBGRIDMEM_CUDAUVM)
  typedef Kokkos::CudaUVMSpace SubgridMem;
#else
  typedef Kokkos::HostSpace SubgridMem;
#endif

// Define a unified memory space for data required on Host and Device
// If HostMem == AssemblyMem == HostSpace, then UnifiedMem = HostSpace
// If HostMem == HostSpace and AssemblyMem == CudaSpace, then UnifiedMem = CudaUVMSpace
//typedef Kokkos::HostSpace UnifiedMem;
typedef Kokkos::HostSpace UnifiedMem;

// Kokkos Node typedefs
// Format: Kokkos::Compat::Kokkos*WrapperNode
// Options: Serial, OpenMP, Threads, Cuda
typedef Kokkos::Compat::KokkosSerialWrapperNode HostNode;
typedef Kokkos::Compat::KokkosSerialWrapperNode AssemblyNode;
typedef Kokkos::Compat::KokkosSerialWrapperNode SubgridNode;

// Typedef Kokkos devices based on Exec, Mem
typedef Kokkos::Device<HostExec,HostMem> HostDevice;
typedef Kokkos::Device<AssemblyExec,AssemblyMem> AssemblyDevice;
typedef Kokkos::Device<SubgridExec,SubgridMem> SubgridDevice;
typedef Kokkos::Device<AssemblyExec,UnifiedMem> UnifiedDevice;


// Kokkos object typedefs (preferable to use Kokkos::View<*,Device>)
typedef Kokkos::DynRankView<ScalarT,AssemblyDevice> DRV;
typedef Kokkos::DynRankView<int,AssemblyDevice> DRVint;
typedef Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> FDATA;
typedef Kokkos::View<ScalarT**,Kokkos::LayoutStride,AssemblyDevice> FDATAd;
typedef Kokkos::View<LO**,AssemblyDevice> LIDView;
typedef Kokkos::View<LO**,HostDevice> LIDView_host;
typedef Kokkos::View<AD*>::size_type size_type;

// Intrepid and shards typedefs
//typedef Teuchos::RCP<Intrepid2::Basis<AssemblyDevice, ScalarT, ScalarT > > basis_RCP;
typedef Teuchos::RCP<const shards::CellTopology> topo_RCP;
typedef Teuchos::RCP<Intrepid2::Basis<PHX::Device::execution_space, ScalarT, ScalarT > > basis_RCP;
typedef Intrepid2::CellTools<AssemblyExec> CellTools;
typedef Intrepid2::FunctionSpaceTools<AssemblyExec> FuncTools;
typedef Intrepid2::OrientationTools<AssemblyExec> OrientTools;

// Tpetra linear algebra typedefs (Epetra is no longer supported)
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
