
#include "trilinos.hpp"
#include "preferences.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  
  
  {
    int numElem = 1000;
    if (argc>1) {
      numElem = atof(argv[1]);
    }
    std::cout << "Number of elements: " << numElem << std::endl;
 
    const int TeamSize = 1; 
    std::cout << "Team size: " << TeamSize << std::endl;
  
    const int VectorSize = 32;
    std::cout << "Vector size: " << VectorSize << std::endl;
    

    const int numDerivs = 64;
    typedef Sacado::Fad::SFad<ScalarT,numDerivs> EvalT;
    typedef Kokkos::LayoutContiguous<AssemblyExec::array_layout,VectorSize> CL;

    int numip = 8;
    int dimension = 3;
    int numdof = 8;
    
    ////////////////////////////////////////////////
    // Set up timer and views
    ////////////////////////////////////////////////
    
    Kokkos::Timer timer;
 
    Kokkos::View<ScalarT****,CL,AssemblyDevice> basis("basis",numElem,numdof,numip,dimension);
    Kokkos::View<ScalarT****,CL,AssemblyDevice> basis_grad("basis",numElem,numdof,numip,dimension);
    
    Kokkos::deep_copy(basis,1.0);
    Kokkos::deep_copy(basis_grad,2.0);
    
    Kokkos::View<EvalT***,CL,AssemblyDevice> gradT("sol grad",numElem,numip,dimension,numDerivs);
    Kokkos::View<EvalT**,CL,AssemblyDevice> diff("diff",numElem,numip,numDerivs);
    Kokkos::View<EvalT**,CL,AssemblyDevice> source("src",numElem,numip,numDerivs);
    Kokkos::View<ScalarT**,CL,AssemblyDevice> wts("wts",numElem,numip);
    
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,basis.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<gradT.extent(1); ++pt) {
        for (size_type dim=0; dim<gradT.extent(2); ++dim) {
          gradT(elem,pt,dim) = EvalT(32,pt,100.0);
        }
      }
    });
    Kokkos::deep_copy(diff,1.0);
    Kokkos::deep_copy(source,1.0);
    Kokkos::deep_copy(wts,1.0);
    
    Kokkos::View<EvalT**,CL,AssemblyDevice> res("res",numElem,numdof,numDerivs);
    Kokkos::View<EvalT**,CL,AssemblyDevice> res2("res2",numElem,numdof,numDerivs);
    
    Kokkos::View<ScalarT***,CL,AssemblyDevice> rJdiff("error",numElem,numdof,numDerivs+1);
   
    Kokkos::View<EvalT***,CL,AssemblyDevice> scratch("scratch vals",numElem,numip,3,numDerivs);

     
    ////////////////////////////////////////////////
    // Baseline (current implementation in MrHyDE)
    ////////////////////////////////////////////////

    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,basis.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<basis.extent(2); pt++ ) {
        EvalT f = -1.0*source(elem,pt);
        EvalT DFx = diff(elem,pt)*gradT(elem,pt,0);
        EvalT DFy = diff(elem,pt)*gradT(elem,pt,1);
        f *= wts(elem,pt);
        DFx *= wts(elem,pt);
        DFy *= wts(elem,pt);
        for (size_type dof=0; dof<basis.extent(1); dof++ ) {
          res(elem,dof) += f*basis(elem,dof,pt,0) + DFx*basis_grad(elem,dof,pt,0) + DFy*basis_grad(elem,dof,pt,1);
        }
      }
    });
    Kokkos::fence();
    double sol_time1 = timer.seconds();
    printf("Baseline time:   %e \n", sol_time1);
    

    ////////////////////////////////////////////////
    // Hierarchical version 1: team over (elem,dof)
    ////////////////////////////////////////////////
      
    typedef Kokkos::TeamPolicy<AssemblyExec> Policy;
    typedef Policy::member_type member_type;
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 Policy(basis.extent(0), TeamSize, VectorSize),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      int ti = team.team_rank();
      int ts = team.team_size();
      for (size_type dof=ti; dof<basis.extent(1); dof+=ts ) {
        EvalT f=0.0, DFx = 0.0, DFy = 0.0;
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          f = -1.0*source(elem,pt)*wts(elem,pt);
          DFx = diff(elem,pt)*gradT(elem,pt,0)*wts(elem,pt);
          DFy = diff(elem,pt)*gradT(elem,pt,1)*wts(elem,pt);
          res2(elem,dof) += f*basis(elem,dof,pt,0) + DFx*basis_grad(elem,dof,pt,0) + DFy*basis_grad(elem,dof,pt,1);
        }
      }
    });
    
    Kokkos::fence();
    double sol_time3 = timer.seconds();
    printf("MD2 ratio:   %e \n", sol_time3/sol_time1);
    
    Kokkos::deep_copy(res2,0.0); 

    ////////////////////////////////////////////////
    // Hierarchical version 2: team over (elem,pt) to fill scratch, then team over (elem,dof)
    ////////////////////////////////////////////////
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 Policy(basis.extent(0),TeamSize, VectorSize),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      int ti = team.team_rank();
      int ts = team.team_size();
      for (size_type pt=ti; pt<scratch.extent(1); pt+=ts ) {
        scratch(elem,pt,0) = -1.0*source(elem,pt)*wts(elem,pt);
        scratch(elem,pt,1) = diff(elem,pt)*gradT(elem,pt,0)*wts(elem,pt);
        scratch(elem,pt,2) = diff(elem,pt)*gradT(elem,pt,1)*wts(elem,pt);
      }
    });

    parallel_for("Thermal volume resid 2D",
                 Policy(basis.extent(0), TeamSize, VectorSize),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      int ti = team.team_rank();
      int ts = team.team_size();
      for (size_type dof=ti; dof<basis.extent(1); dof+=ts ) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          res2(elem,dof) += scratch(elem,pt,0)*basis(elem,dof,pt,0) + scratch(elem,pt,1)*basis_grad(elem,dof,pt,0) + scratch(elem,pt,2)*basis_grad(elem,dof,pt,1);
        }
      }
    });
    
    Kokkos::fence();
    double sol_time4 = timer.seconds();
    printf("MD3 ratio:   %e \n", sol_time4/sol_time1);

    ////////////////////////////////////////////////
    // Verify that the baseline and latest hierarchical got the same answer 
    ////////////////////////////////////////////////
 
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,rJdiff.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type dof=0; dof<rJdiff.extent(1); dof++ ) {
        rJdiff(elem,dof,0) = res(elem,dof).val() - res2(elem,dof).val();
        for (size_type d=1; d<rJdiff.extent(2); d++) {
          rJdiff(elem,dof,d) = res(elem,dof).fastAccessDx(d-1) - res2(elem,dof).fastAccessDx(d-1);
        }
      }
    });
   
    std::cout << "computed diffs on device" << std::endl;
 
    auto rJdiff_host = Kokkos::create_mirror_view(rJdiff);
    Kokkos::deep_copy(rJdiff_host,rJdiff);
    
    ScalarT valerror = 0.0, deriverror = 0.0;
    for (size_type elem=0; elem<rJdiff_host.extent(0); elem++ ) {
      for (size_type dof=0; dof<rJdiff_host.extent(1); dof++ ) {
        valerror += abs(rJdiff_host(elem,dof,0));
        for (size_type d=1; d<rJdiff_host.extent(2); d++) {
          deriverror += abs(rJdiff_host(elem,dof,d));
        }
      }
    }
    std::cout << "value error: " << valerror << std::endl;
    std::cout << "deriv error: " << deriverror << std::endl;
    
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


