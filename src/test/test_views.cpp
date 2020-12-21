
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
  
    const int numDerivs = 32;
    typedef Sacado::Fad::SFad<ScalarT,numDerivs> EvalT;
    
    const int VectorSize = 32;
    std::cout << "Vector size: " << VectorSize << std::endl;
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
                 Policy(basis.extent(0), Kokkos::AUTO, VectorSize),
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
                 Policy(basis.extent(0), Kokkos::AUTO, VectorSize),
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
                 Policy(basis.extent(0), Kokkos::AUTO, VectorSize),
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
    

    ////////////////////////////////////////////////
    // Another key kernel
    ////////////////////////////////////////////////

    Kokkos::View<ScalarT****,CL,AssemblyDevice> cbasis("basis",numElem,numdof,numip,dimension);
    Kokkos::View<ScalarT****,CL,AssemblyDevice> cbasis_grad("basis",numElem,numdof,numip,dimension);
    
    Kokkos::deep_copy(cbasis,1.0);
    Kokkos::deep_copy(cbasis_grad,2.0);
    
    Kokkos::View<EvalT**,CL,AssemblyDevice> cuvals("sol",numElem,numdof,numDerivs);
    
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,cuvals.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type dof=0; dof<cuvals.extent(1); ++dof) {
        cuvals(elem,dof) = EvalT(32,dof,100.0);
      }
    });
    Kokkos::View<EvalT**,CL,AssemblyDevice> csol("diff",numElem,numip,numDerivs);
    Kokkos::View<EvalT***,CL,AssemblyDevice> csol_grad("src",numElem,numip,dimension,numDerivs);
    Kokkos::View<EvalT**,CL,AssemblyDevice> csol2("diff",numElem,numip,numDerivs);
    Kokkos::View<EvalT***,CL,AssemblyDevice> csol2_grad("src",numElem,numip,dimension,numDerivs);
    Kokkos::View<EvalT**,CL,AssemblyDevice> csol3("diff",numElem,numip,numDerivs);
    Kokkos::View<EvalT***,CL,AssemblyDevice> csol3_grad("src",numElem,numip,dimension,numDerivs);
    
    Kokkos::View<ScalarT**,CL,AssemblyDevice> sol2diff("error",numElem,numDerivs+1);
    Kokkos::View<ScalarT**,CL,AssemblyDevice> sol3diff("error",numElem,numDerivs+1);
    
    ////////////////////////////////////////////////
    // Baseline (current implementation in MrHyDE)
    ////////////////////////////////////////////////

    timer.reset();
    parallel_for("wkset soln ip HGRAD",
                 RangePolicy<AssemblyExec>(0,cbasis.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
        EvalT uval = cuvals(elem,dof);
        if ( dof == 0) {
          for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
            csol(elem,pt) = uval*cbasis(elem,dof,pt,0);
            for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
              csol_grad(elem,pt,s) = uval*cbasis_grad(elem,dof,pt,s);
            }
          }
        }
        else {
          for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
            csol(elem,pt) += uval*cbasis(elem,dof,pt,0);
            for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
              csol_grad(elem,pt,s) += uval*cbasis_grad(elem,dof,pt,s);
            }
          }
        }
      }
    });
    Kokkos::fence();
    double ker2_time = timer.seconds();
    printf("Baseline 2:   %e \n", ker2_time);

    ////////////////////////////////////////////////
    // Simple modification
    ////////////////////////////////////////////////

    timer.reset();
    parallel_for("wkset soln ip HGRAD",
                 RangePolicy<AssemblyExec>(0,cbasis.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<cbasis.extent(2); pt++ ) {
        csol2(elem,pt) = 0.0;
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          csol2(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
        }
        for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
          csol2_grad(elem,pt,s) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csol2_grad(elem,pt,s) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,s);
          }
        }
      }
    });
    Kokkos::fence();
    double ker2_time2 = timer.seconds();
    printf("New baseline ratio:   %e \n", ker2_time2/ker2_time);

    
    
    ////////////////////////////////////////////////
    // Hierarchical modified
    ////////////////////////////////////////////////
    
    timer.reset();
    parallel_for("wkset soln ip HGRAD",
                 Policy(basis.extent(0), Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      int ti = team.team_rank();
      int ts = team.team_size();
      for (size_type pt=ti; pt<cbasis.extent(2); pt+=ts ) {
        csol3(elem,pt) = 0.0;
        for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
          csol3(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
        }
        for (size_type s=0; s<cbasis_grad.extent(3); s++ ) {
          csol3_grad(elem,pt,s) = 0.0;
          for (size_type dof=0; dof<cbasis.extent(1); dof++ ) {
            csol3_grad(elem,pt,s) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,s);
          }
        }
      }
    });
    Kokkos::fence();
    double ker2_time3 = timer.seconds();
    printf("Hierarchical ratio:   %e \n", ker2_time3/ker2_time);

    ////////////////////////////////////////////////
    // Verify the results
    ////////////////////////////////////////////////
 
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,sol2diff.extent(0)),
                 KOKKOS_LAMBDA (const int elem ) {
      for (size_type pt=0; pt<csol2.extent(1); pt++ ) {
        sol2diff(elem,0) += csol(elem,pt).val() - csol2(elem,pt).val();
        for (size_type d=0; d<csol2(elem,pt).size(); d++) {
          sol2diff(elem,d+1) = csol(elem,pt).fastAccessDx(d) - csol2(elem,pt).fastAccessDx(d);
        }
        sol3diff(elem,0) += csol(elem,pt).val() - csol3(elem,pt).val();
        for (size_type d=0; d<csol3(elem,pt).size(); d++) {
          sol3diff(elem,d+1) = csol(elem,pt).fastAccessDx(d) - csol3(elem,pt).fastAccessDx(d);
        }
      }
    });
   
    //std::cout << "computed diffs on device" << std::endl;
 
    auto sol2diff_host = Kokkos::create_mirror_view(sol2diff);
    Kokkos::deep_copy(sol2diff_host,sol2diff);
    
    auto sol3diff_host = Kokkos::create_mirror_view(sol3diff);
    Kokkos::deep_copy(sol3diff_host,sol3diff);
    
    valerror = 0.0, deriverror = 0.0;
    for (size_type elem=0; elem<sol2diff_host.extent(0); elem++ ) {
      valerror += abs(sol2diff_host(elem,0));
      for (size_type d=1; d<sol2diff_host.extent(2); d++) {
        deriverror += abs(sol2diff_host(elem,d));
      }
    }
    std::cout << "value error: " << valerror << std::endl;
    std::cout << "deriv error: " << deriverror << std::endl;
    
    valerror = 0.0, deriverror = 0.0;
    for (size_type elem=0; elem<sol3diff_host.extent(0); elem++ ) {
      valerror += abs(sol3diff_host(elem,0));
      for (size_type d=1; d<sol3diff_host.extent(2); d++) {
        deriverror += abs(sol3diff_host(elem,d));
      }
    }
    std::cout << "hier. value error: " << valerror << std::endl;
    std::cout << "hier. deriv error: " << deriverror << std::endl;
    
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


