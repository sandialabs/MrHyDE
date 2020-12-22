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

#ifndef PREFERENCES_H
#define PREFERENCES_H

#include "PanzerCore_config.hpp"
#include "Intrepid2_Basis.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"
#include "Phalanx_DataLayout.hpp"

using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::RangePolicy;
using Kokkos::MDRangePolicy;
using Kokkos::TeamPolicy;
using Kokkos::Rank;
using std::string;
using std::vector;
using std::cout;
using std::endl;

#define MRHYDE_VERSION "1.0"

typedef double ScalarT;
typedef int LO; // same as panzer::LocalOrdinal
typedef panzer::GlobalOrdinal GO;

// Number of derivatives in SFAD objects
#ifdef MrHyDE_SET_MAX_DERIVS
  #define maxDerivs MrHyDE_SET_MAX_DERIVS // allow us to set this at configure-time with the MrHyDE_MAX_DERIVS flag
#else
  #define maxDerivs 64 // adjust this to improve performance
#endif

// Size of vectors for hierarchical parallel policies
//#ifdef MrHyDE_SET_VECTOR_SIZE
//  #define VectorSize MrHyDE_SET_VECTOR_SIZE // allow us to set this at configure-time with the MrHyDE_VECTOR_SIZE flag
//#else
//  #define VectorSize 32 // probably fine for most architectures
//#endif


#define PI 3.141592653589793238463
typedef Teuchos::MpiComm<int> MpiComm;

// AD typedefs
typedef Sacado::Fad::DFad<ScalarT> DFAD; // used only when absolutely necessary
typedef Sacado::Fad::SFad<ScalarT,maxDerivs> AD;

// Kokkos Execution Space typedefs
// Format: Kokkos::*
// Options: Serial, OpenMP, Threads, Cuda
typedef Kokkos::Serial HostExec; // cannot be Cuda right now
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  typedef Kokkos::Cuda AssemblyExec;
#else
  typedef Kokkos::Serial AssemblyExec;
#endif

// Kokkos contiguous layout for optimal use of hierarchical parallelism
typedef Kokkos::LayoutContiguous<AssemblyExec::array_layout,32> ContLayout;
//typedef Kokkos::LayoutContiguous<Kokkos::LayoutStride,VectorSize> ContLayout;

// Kokkos Memory Space typedefs
// Format: Kokkos::*
// Options: HostSpace, CudaSpace, CudaUVMSpace
typedef Kokkos::HostSpace HostMem; // cannot be CudaSpace right now
#if defined(MrHyDE_ASSEMBLYMEM_CUDA)
  typedef Kokkos::CudaSpace AssemblyMem;
#elif defined(MrHyDE_ASSEMBLYMEM_CUDAUVM) // to be deprecated
  typedef Kokkos::CudaUVMSpace AssemblyMem;
#else
  typedef Kokkos::HostSpace AssemblyMem;
#endif

// Kokkos Node typedefs
// Format: Kokkos::Compat::Kokkos*WrapperNode
// Options: Serial, OpenMP, Threads, Cuda
typedef Kokkos::Compat::KokkosSerialWrapperNode HostNode;
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  typedef Kokkos::Compat::KokkosCudaWrapperNode AssemblyNode;
#else
  typedef Kokkos::Compat::KokkosSerialWrapperNode AssemblyNode;
#endif
#if defined(MrHyDE_SOLVERSPACE_CUDA)
  typedef Kokkos::Compat::KokkosCudaWrapperNode SolverNode;
#else
  typedef Kokkos::Compat::KokkosSerialWrapperNode SolverNode;
#endif
typedef AssemblyNode SubgridSolverNode;
//typedef typename SolverNode::device_type SolverDevice;

// Typedef Kokkos devices based on Exec, Mem
typedef Kokkos::Device<HostExec,HostMem> HostDevice;
typedef Kokkos::Device<AssemblyExec,AssemblyMem> AssemblyDevice;

// Kokkos object typedefs (preferable to use Kokkos::View<*,Device>)
typedef Kokkos::DynRankView<ScalarT,PHX::Device> DRV; // for interacting with Intrepid2/Panzer
typedef Kokkos::View<LO**,AssemblyDevice> LIDView;
typedef Kokkos::View<LO**,HostDevice> LIDView_host;
typedef Kokkos::View<ScalarT*>::size_type size_type;

// Use ContLayout for faster hierarchical parallelism
typedef Kokkos::View<AD*,ContLayout,AssemblyDevice> View_AD1;
typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2; // replaces FDATA
typedef Kokkos::View<AD***,ContLayout,AssemblyDevice> View_AD3;
typedef Kokkos::View<AD****,ContLayout,AssemblyDevice> View_AD4;
typedef Kokkos::View<ScalarT*,AssemblyDevice> View_Sc1;
typedef Kokkos::View<ScalarT**,AssemblyDevice> View_Sc2; // replaces FDATAd
typedef Kokkos::View<ScalarT***,AssemblyDevice> View_Sc3;
typedef Kokkos::View<ScalarT****,AssemblyDevice> View_Sc4;

// Special Views for function manager
// These must be created as subviews and cannot be constructed directly without a given stride
typedef Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> View_AD2_sv; // replaces FDATA
typedef Kokkos::View<ScalarT**,Kokkos::LayoutStride,AssemblyDevice> View_Sc2_sv; // replaces FDATAd

// Intrepid and shards typedefs
typedef Teuchos::RCP<const shards::CellTopology> topo_RCP;
typedef Teuchos::RCP<Intrepid2::Basis<PHX::Device::execution_space, ScalarT, ScalarT > > basis_RCP;
typedef Intrepid2::CellTools<PHX::Device::execution_space> CellTools;
typedef Intrepid2::FunctionSpaceTools<PHX::Device::execution_space> FuncTools;
typedef Intrepid2::OrientationTools<PHX::Device::execution_space> OrientTools;
typedef Intrepid2::RealSpaceTools<PHX::Device::execution_space> RealTools;
typedef Intrepid2::ArrayTools<PHX::Device::execution_space> ArrayTools;

#endif
