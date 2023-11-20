/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "trilinos.hpp"
#include "preferences.hpp"

using namespace std;

#include <stdio.h>
#include <stdint.h>
#include <math.h>

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);

  Kokkos::initialize();

  {
    const uint64_t umin=1;
    const uint64_t umax=10000000000LL;
    double sum = 0.0;
    
    //#pragma omp parallel for reduction(+:sum)
    //for(uint64_t u=umin; u<umax; u++)
    //  sum+=1./u/u/log(u+1);
    
    parallel_reduce(RangePolicy<AssemblyExec>(umin, umax),
                    KOKKOS_LAMBDA (const uint64_t u, ScalarT& update) {
      update += 1./u/u/log(u+1);
    }, sum);
    
    printf("%e\n", sum);
  }
  Kokkos::finalize();
  int val = 0;
  return val;
}
