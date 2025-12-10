/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef MRHYDE_PREFERENCES_H
#define MRHYDE_PREFERENCES_H

#include "PanzerCore_config.hpp"
#include "Intrepid2_Basis.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado.hpp"

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
  #define MAXDERIVS MrHyDE_SET_MAX_DERIVS // allow us to set this at configure-time with the MrHyDE_MAX_DERIVS flag
#else
  #define MAXDERIVS 64 // adjust this to improve performance
#endif

// Size of vectors for hierarchical parallel policies
#ifdef MrHyDE_SET_VECTOR_SIZE
  #define VECTORSIZE MrHyDE_SET_VECTOR_SIZE // allow us to set this at configure-time with the MrHyDE_VECTOR_SIZE flag
#else
  #define VECTORSIZE MAXDERIVS
#endif

// Sets default behavior for evaluating solution fields using basis functions
// Can be explicitly overruled regardless of choice
#ifdef MrHyDE_NO_SOL_FIELD_EVAL
  #define SOL_FIELD_EVAL false // Useful for certain unit tests
#else
  #define SOL_FIELD_EVAL true // otherwise this should be chosen
#endif

#define PI 3.141592653589793238463
typedef Teuchos::MpiComm<int> MpiComm;

// AD typedefs
typedef Sacado::Fad::DFad<ScalarT> DFAD; // used only when absolutely necessary

#ifdef MrHyDE_NO_AD
typedef ScalarT AD;
#else
typedef Sacado::Fad::SFad<ScalarT,MAXDERIVS> AD;
#endif

#ifndef MrHyDE_NO_AD
// Commonly used AD types
typedef Sacado::Fad::SFad<ScalarT,2> AD2;
typedef Sacado::Fad::SFad<ScalarT,4> AD4;
typedef Sacado::Fad::SFad<ScalarT,8> AD8;
typedef Sacado::Fad::SFad<ScalarT,16> AD16;
typedef Sacado::Fad::SFad<ScalarT,18> AD18;
typedef Sacado::Fad::SFad<ScalarT,24> AD24;
typedef Sacado::Fad::SFad<ScalarT,32> AD32;

// Rarely used
typedef Sacado::Fad::SFad<ScalarT,128> AD128;
typedef Sacado::Fad::SFad<ScalarT,256> AD256;
typedef Sacado::Fad::SFad<ScalarT,512> AD512;
typedef Sacado::Fad::SFad<ScalarT,1024> AD1024;
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


// Trilinos 14 and later redefines the Kokkos::Compat wrappers in Tpetra to Tpetra::KokkosCompat
// Define intermediate namespaces here to clean up the logic for the HostNode,AssemblyNode,SolverNode
#ifdef MrHyDE_HAVE_TRILINOS14
  namespace MrHyDECompat = Tpetra::KokkosCompat;
#else
  namespace MrHyDECompat = Kokkos::Compat;
#endif

// Host Node
#if defined(MrHyDE_SOLVERSPACE_OPENMP)
  typedef MrHyDECompat::KokkosOpenMPWrapperNode HostNode;
#else
  typedef MrHyDECompat::KokkosSerialWrapperNode HostNode;
#endif

// Assembly Node
#if defined(MrHyDE_ASSEMBLYSPACE_CUDA)
  typedef MrHyDECompat::KokkosCudaWrapperNode AssemblyNode;
#elif defined(MrHyDE_ASSEMBLYSPACE_OPENMP)
  typedef MrHyDECompat::KokkosOpenMPWrapperNode AssemblyNode;
#else
  typedef MrHyDECompat::KokkosSerialWrapperNode AssemblyNode;
#endif

// Solver Node
#if defined(MrHyDE_SOLVERSPACE_CUDA)
  typedef MrHyDECompat::KokkosCudaWrapperNode SolverNode;
#elif defined(MrHyDE_SOLVERSPACE_OPENMP)
  typedef MrHyDECompat::KokkosOpenMPWrapperNode SolverNode;
#else
  typedef MrHyDECompat::KokkosSerialWrapperNode SolverNode;
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
typedef Kokkos::View<ScalarT**,HostDevice> NodeView_host;
typedef Kokkos::View<ScalarT*>::size_type size_type;

// Use ContLayout for faster hierarchical parallelism
//typedef Kokkos::LayoutContiguous<AssemblyExec::array_layout,VectorSize> ContLayout;
typedef Kokkos::LayoutContiguous<AssemblyExec::array_layout> ContLayout;

typedef Kokkos::View<ScalarT*,AssemblyDevice> View_Sc1;
typedef Kokkos::View<ScalarT**,AssemblyDevice> View_Sc2;
typedef Kokkos::View<ScalarT***,AssemblyDevice> View_Sc3;
typedef Kokkos::View<ScalarT****,AssemblyDevice> View_Sc4;
typedef Kokkos::View<ScalarT*****,AssemblyDevice> View_Sc5;

// Intrepid and shards typedefs
typedef Teuchos::RCP<const shards::CellTopology> topo_RCP;
typedef Teuchos::RCP<Intrepid2::Basis<PHX::Device::execution_space, double, double > > basis_RCP;

#include <exception>

#define PRINT_VAR(var) \
std::cout << #var << " = " << var << std::endl;

#define PRINT_VECTOR(var) \
std::cout << #var << "(" << var.size() << ") = [" << var[0]; \
for(unsigned int i=1; i<var.size(); ++i) \
  std::cout << " " << var[i]; \
std::cout << "]" << std::endl;

#define PRINT_VIEW1D(var) \
std::cout << #var << "(" << var.extent(0) << ") = [" << var(0); \
for(unsigned int i=1; i<var.extent(0); ++i) \
  std::cout << " " << var(i); \
std::cout << "]" << std::endl;

#define PRINT_VIEW2D(var) \
std::cout << #var << "(" << var.extent(0) << "," << var.extent(1) << ") = ["; \
for(unsigned int i=0; i<var.extent(0); ++i) \
  for(unsigned int j=0; j<var.extent(1); ++j) \
    std::cout << " " << var(i,j); \
std::cout << "]" << std::endl;

#define PRINT_VIEW3D(var) \
std::cout << #var << "(" << var.extent(0) << "," << var.extent(1) << "," << var.extent(2) << ") = ["; \
for(unsigned int i=0; i<var.extent(0); ++i) \
  for(unsigned int j=0; j<var.extent(1); ++j) \
    for(unsigned int k=0; k<var.extent(2); ++k) \
      std::cout << " " << var(i,j,k); \
std::cout << "]" << std::endl;

#endif

#if !defined(MRHYDE_LAMBDA)
#define MRHYDE_LAMBDA [&]
#endif
