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
using Kokkos::Rank;
using std::string;
using std::vector;
using std::cout;
using std::endl;

#define MRHYDE_VERSION "1.0"

typedef double ScalarT;
typedef int LO; // same as panzer::LocalOrdinal
typedef panzer::GlobalOrdinal GO;

#ifdef MrHyDE_SET_MAX_DERIVS
  #define maxDerivs MrHyDE_SET_MAX_DERIVS // allow us to set this at configure-time with the MrHyDE_MAX_DERIVS flag
#else
  #define maxDerivs 64 // adjust this to improve performance
#endif


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
typedef Kokkos::View<AD**,Kokkos::LayoutStride,AssemblyDevice> FDATA;
typedef Kokkos::View<ScalarT**,Kokkos::LayoutStride,AssemblyDevice> FDATAd;
typedef Kokkos::View<LO**,AssemblyDevice> LIDView;
typedef Kokkos::View<LO**,HostDevice> LIDView_host;
typedef Kokkos::View<ScalarT*>::size_type size_type;

// Intrepid and shards typedefs
typedef Teuchos::RCP<const shards::CellTopology> topo_RCP;
typedef Teuchos::RCP<Intrepid2::Basis<PHX::Device::execution_space, ScalarT, ScalarT > > basis_RCP;
typedef Intrepid2::CellTools<PHX::Device::execution_space> CellTools;
typedef Intrepid2::FunctionSpaceTools<PHX::Device::execution_space> FuncTools;
typedef Intrepid2::OrientationTools<PHX::Device::execution_space> OrientTools;
typedef Intrepid2::RealSpaceTools<PHX::Device::execution_space> RealTools;
typedef Intrepid2::ArrayTools<PHX::Device::execution_space> ArrayTools;

#endif
