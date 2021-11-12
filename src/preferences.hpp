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
#include "Phalanx_DataLayout.hpp"

using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::RangePolicy;
using Kokkos::MDRangePolicy;
using Kokkos::TeamPolicy;
using Kokkos::Rank;
using Kokkos::subview;
using Kokkos::ALL;
using Kokkos::create_mirror_view;
using Kokkos::deep_copy;
using Kokkos::fence;

using Teuchos::RCP;
using Teuchos::rcp;

using std::string;
using std::vector;
using std::cout;
using std::endl;

#define MRHYDE_VERSION "1.0"


#if defined(MrHyDE_SINGLE_PRECISION)
typedef float ScalarT;
#else
typedef double ScalarT;
#endif

typedef int LO; // same as panzer::LocalOrdinal
typedef panzer::GlobalOrdinal GO;

// Number of derivatives in SFAD objects
#ifdef MrHyDE_SET_MAX_DERIVS
  #define maxDerivs MrHyDE_SET_MAX_DERIVS // allow us to set this at configure-time with the MrHyDE_MAX_DERIVS flag
#else
  #define maxDerivs 64 // adjust this to improve performance
#endif

// Size of vectors for hierarchical parallel policies
#ifdef MrHyDE_SET_VECTOR_SIZE
  #define VectorSize MrHyDE_SET_VECTOR_SIZE // allow us to set this at configure-time with the MrHyDE_VECTOR_SIZE flag
#else
  #define VectorSize maxDerivs
#endif


#define PI 3.141592653589793238463
typedef Teuchos::MpiComm<int> MpiComm;

// AD typedefs
typedef Sacado::Fad::DFad<ScalarT> DFAD; // used only when absolutely necessary

#ifdef MrHyDE_NO_AD
typedef ScalarT AD;
#else
typedef Sacado::Fad::SFad<ScalarT,maxDerivs> AD;
#endif

// Host Execution Space
#if defined(MrHyDE_HOSTEXEC_OPENMP)
  typedef Kokkos::OpenMP HostExec;
#else
  typedef Kokkos::Serial HostExec;
#endif

// Host Memory Space
typedef Kokkos::HostSpace HostMem;

// Assembly Execution Space
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  typedef Kokkos::Cuda AssemblyExec;
#elif defined(MrHyDE_ASSEMBLYSPACE_OPENMP)
  typedef Kokkos::OpenMP AssemblyExec;
#else
  typedef Kokkos::Serial AssemblyExec;
#endif

// Assembly Memory Space (No UVM option)
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  typedef Kokkos::CudaSpace AssemblyMem;
#else
  typedef Kokkos::HostSpace AssemblyMem;
#endif

// Host Node
#if defined(MrHyDE_SOLVERSPACE_OPENMP)
  typedef Kokkos::Compat::KokkosOpenMPWrapperNode HostNode;
#else
  typedef Kokkos::Compat::KokkosSerialWrapperNode HostNode;
#endif

// Assembly Node
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  typedef Kokkos::Compat::KokkosCudaWrapperNode AssemblyNode;
#elif defined(MrHyDE_ASSEMBLYSPACE_OPENMP)
  typedef Kokkos::Compat::KokkosOpenMPWrapperNode AssemblyNode;
#else
  typedef Kokkos::Compat::KokkosSerialWrapperNode AssemblyNode;
#endif

// Solver Node
#if defined(MrHyDE_SOLVERSPACE_CUDA)
  typedef Kokkos::Compat::KokkosCudaWrapperNode SolverNode;
#elif defined(MrHyDE_SOLVERSPACE_OPENMP)
  typedef Kokkos::Compat::KokkosOpenMPWrapperNode SolverNode;
#else
  typedef Kokkos::Compat::KokkosSerialWrapperNode SolverNode;
#endif

// Subgrid Solver Node (defaults to assembly node)
typedef AssemblyNode SubgridSolverNode;

// Need to determine if the SolverNode == SubgridSolverNode for explicit template instantiation
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA) && !defined(MrHyDE_SOLVERSPACE_CUDA)
  #define MrHyDE_REQ_SUBGRID_ETI true
#elif defined(MrHyDE_ASSEMBLYSPACE_OPENMP) && !defined(MrHyDE_SOLVERSPACE_OPENMP)
  #define MrHyDE_REQ_SUBGRID_ETI true
#else
  #define MrHyDE_REQ_SUBGRID_ETI false
#endif

// Typedef Kokkos devices based on Exec, Mem
typedef Kokkos::Device<HostExec,HostMem> HostDevice;
typedef Kokkos::Device<AssemblyExec,AssemblyMem> AssemblyDevice;

// Kokkos object typedefs (preferable to use Kokkos::View<*,Device>)
typedef Kokkos::DynRankView<double,PHX::Device> DRV; // for interacting with Intrepid2/Panzer
typedef Kokkos::View<LO**,AssemblyDevice> LIDView;
typedef Kokkos::View<LO**,HostDevice> LIDView_host;
typedef Kokkos::View<ScalarT*>::size_type size_type;

// Use ContLayout for faster hierarchical parallelism
typedef Kokkos::LayoutContiguous<AssemblyExec::array_layout,VectorSize> ContLayout;

typedef Kokkos::View<ScalarT*,AssemblyDevice> View_Sc1;
typedef Kokkos::View<ScalarT**,AssemblyDevice> View_Sc2;
typedef Kokkos::View<ScalarT***,AssemblyDevice> View_Sc3;
typedef Kokkos::View<ScalarT****,AssemblyDevice> View_Sc4;
typedef Kokkos::View<ScalarT*****,AssemblyDevice> View_Sc5;
#ifndef MrHyDE_NO_AD
typedef Kokkos::View<AD*,ContLayout,AssemblyDevice> View_AD1;
typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2;
typedef Kokkos::View<AD***,ContLayout,AssemblyDevice> View_AD3;
typedef Kokkos::View<AD****,ContLayout,AssemblyDevice> View_AD4;
#else
typedef View_Sc1 View_AD1;
typedef View_Sc2 View_AD2;
typedef View_Sc3 View_AD3;
typedef View_Sc4 View_AD4;
#endif

// Intrepid and shards typedefs
typedef Teuchos::RCP<const shards::CellTopology> topo_RCP;
typedef Teuchos::RCP<Intrepid2::Basis<PHX::Device::execution_space, double, double > > basis_RCP;

#endif
