/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "trilinos.hpp"
#include "preferences.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  typedef Kokkos::TeamPolicy<AssemblyExec> TeamPolicy;
  typedef TeamPolicy::member_type member_type;
  
  {
    int numElem = 10000;
    if (argc>1) {
      numElem = atof(argv[1]);
    }
    std::cout << "Number of elements: " << numElem << std::endl;
 
#define TeamSize 1
    std::cout << "Team size: " << TeamSize << std::endl;
  
    typedef ScalarT EvalT;
    
#define VSize 1
    std::cout << "Vector size: " << VSize << std::endl;
    typedef Kokkos::LayoutContiguous<AssemblyExec::array_layout,VSize> CL;

    int numip = 8;
    int dimension = 3;
    int numdof = 12;
    
    ////////////////////////////////////////////////
    // Set up timer and views
    ////////////////////////////////////////////////
    
    Kokkos::Timer timer;
 
    Kokkos::View<EvalT****,AssemblyDevice> basis("basis",numElem,numdof,numip,dimension);
    Kokkos::View<EvalT****,AssemblyDevice> basis_grad("basis",numElem,numdof,numip,dimension);
    
    Kokkos::View<double****,AssemblyDevice> dbasis("basis",numElem,numdof,numip,dimension);
    Kokkos::View<double****,AssemblyDevice> dbasis_grad("basis",numElem,numdof,numip,dimension);
                                                      
    Kokkos::deep_copy(dbasis,1.0);
    Kokkos::deep_copy(dbasis_grad,2.0);
    
    Kokkos::deep_copy(basis,dbasis);
    Kokkos::deep_copy(basis_grad,dbasis_grad);
    
    Kokkos::View<EvalT**,CL,AssemblyDevice> dT_dx("dTdx",numElem,numip);
    Kokkos::View<EvalT**,CL,AssemblyDevice> dT_dy("dTdy",numElem,numip);
    Kokkos::View<EvalT**,CL,AssemblyDevice> diff("diff",numElem,numip);
    Kokkos::View<EvalT**,CL,AssemblyDevice> source("src",numElem,numip);
    Kokkos::View<ScalarT**,AssemblyDevice> wts("wts",numElem,numip);
    
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,basis.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<dT_dx.extent(1); ++pt) {
        dT_dx(elem,pt) = 100.0;
        dT_dy(elem,pt) = -100.0;
      }
    });
    Kokkos::deep_copy(diff,1.0);
    Kokkos::deep_copy(source,1.0);
    Kokkos::deep_copy(wts,1.0);
    
    Kokkos::View<EvalT**,CL,AssemblyDevice> res("res",numElem,numdof);
         
    ////////////////////////////////////////////////
    // Time the residual computation
    ////////////////////////////////////////////////

    timer.reset();
    parallel_for("copy of thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,basis.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT f = -1.0*source(elem,pt)*wts(elem,pt);
        EvalT DFx = diff(elem,pt)*dT_dx(elem,pt)*wts(elem,pt);
        EvalT DFy = diff(elem,pt)*dT_dy(elem,pt)*wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,dof) += f*basis(elem,dof,pt,0) + DFx*basis_grad(elem,dof,pt,0) + DFy*basis_grad(elem,dof,pt,1);
        }
      }
    });
    Kokkos::fence();
    double sol_time1 = timer.seconds();
    printf("Residual time using flat parallelism:   %e \n", sol_time1);
    
    Kokkos::deep_copy(res,0.0); 

    ////////////////////////////////////////////////
    // Hierarchical version 1: team over (elem,dof)
    ////////////////////////////////////////////////
      
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 TeamPolicy(basis.extent(0), Kokkos::AUTO, VSize),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      for (size_type dof=team.team_rank(); dof<basis.extent(1); dof+=team.team_size() ) {
        EvalT f=0.0, DFx = 0.0, DFy = 0.0;
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          f = -1.0*source(elem,pt)*wts(elem,pt);
          DFx = diff(elem,pt)*dT_dx(elem,pt)*wts(elem,pt);
          DFy = diff(elem,pt)*dT_dy(elem,pt)*wts(elem,pt);
          res(elem,dof) += f*basis(elem,dof,pt,0) + DFx*basis_grad(elem,dof,pt,0) + DFy*basis_grad(elem,dof,pt,1);
        }
      }
    });
    
    Kokkos::fence();
    double sol_times = timer.seconds();
    printf("Residual time using hierarchical parallelism:   %e \n", sol_times);
    
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


