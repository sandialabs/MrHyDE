
#include "trilinos.hpp"
#include "preferences.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  //typedef Kokkos::LayoutContiguous<Kokkos::LayoutLeft> Layout;
  
  const int numDerivs = 24;
  typedef Sacado::Fad::SFad<ScalarT,numDerivs> EvalT;
  //typedef double EvalT;
  //typedef Kokkos::View<EvalT*,AssemblyDevice> View1;
  //typedef Kokkos::View<EvalT**,AssemblyDevice> View2;
  //typedef Kokkos::View<EvalT***,AssemblyDevice> View3;
  typedef Kokkos::View<EvalT****,AssemblyDevice> View4;
  //typedef Kokkos::View<EvalT**,HostDevice> View2_host;
  //typedef Kokkos::View<EvalT****,HostDevice> View4_host;
  
  
  
  {
    int numElem = 1000;
    if (argc>1) {
      numElem = atof(argv[1]);
    }
    std::cout << "Number of elements: " << numElem << std::endl;

    int numip = 8;
    int numvars = 3;
    int dimension = 3;
    //int numrepeats = 1;
    int numdof = 8;
    
    ////////////////////////////////////////////////
    // Baseline (current implementation in MrHyDE)
    ////////////////////////////////////////////////
    Kokkos::Timer timer;
 
    Kokkos::View<ScalarT****,AssemblyDevice> basis("basis",numElem,numdof,numip,dimension);
    Kokkos::View<ScalarT****,AssemblyDevice> basis_grad("basis",numElem,numdof,numip,dimension);
    
    Kokkos::deep_copy(basis,1.0);
    Kokkos::deep_copy(basis_grad,2.0);
    
    Kokkos::View<EvalT***,AssemblyDevice> gradT("sol grad",numElem,numip,dimension,numDerivs);
    Kokkos::View<EvalT**,AssemblyDevice> diff("diff",numElem,numip,numDerivs);
    Kokkos::View<EvalT**,AssemblyDevice> source("src",numElem,numip,numDerivs);
    Kokkos::View<ScalarT**,AssemblyDevice> wts("wts",numElem,numip);
    
    Kokkos::deep_copy(gradT,100.0);
    Kokkos::deep_copy(diff,1.0);
    Kokkos::deep_copy(source,1.0);
    Kokkos::deep_copy(wts,1.0);
    
    Kokkos::View<EvalT**,AssemblyDevice> res("res",numElem,numdof,numDerivs);
    Kokkos::View<EvalT**,AssemblyDevice> res2("res2",numElem,numdof,numDerivs);
    
    Kokkos::View<ScalarT***,AssemblyDevice> rJdiff("res",numElem,numdof,numDerivs+1);
    
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
    

    /*
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 MDRangePolicy<AssemblyExec,Kokkos::Rank<2>>({0,0},{basis.extent(0),basis.extent(1)}),
                 KOKKOS_LAMBDA (const int elem , const int dof) {
      EvalT val = 0.0;
      for (size_type pt=0; pt<wts.extent(1); pt++ ) {
        EvalT f = -1.0*source(elem,pt);
        EvalT DFx = diff(elem,pt)*gradT(elem,pt,0);
        EvalT DFy = diff(elem,pt)*gradT(elem,pt,1);
        f *= wts(elem,pt);
        DFx *= wts(elem,pt);
        DFy *= wts(elem,pt);
        val += f*basis(elem,dof,pt,0) + DFx*basis_grad(elem,dof,pt,0) + DFy*basis_grad(elem,dof,pt,1);
      }
      res2(elem,dof) = val;
    });
    Kokkos::fence();
    double sol_time2 = timer.seconds();
    //printf("MD1 time:   %e \n", sol_time2);
    printf("MD1 ratio:   %e \n", sol_time2/sol_time1);
    */

    Kokkos::View<EvalT**,AssemblyDevice> f("f vals",numElem,numip);
    Kokkos::View<EvalT***,AssemblyDevice> DF("f vals",numElem,numip,2);
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 MDRangePolicy<AssemblyExec,Kokkos::Rank<2>>({0,0},{basis.extent(0),basis.extent(2)}),
                 KOKKOS_LAMBDA (const int elem , const int pt) {
      f(elem,pt) = -1.0*source(elem,pt)*wts(elem,pt);
      DF(elem,pt,0) = diff(elem,pt)*gradT(elem,pt,0)*wts(elem,pt);
      DF(elem,pt,1) = diff(elem,pt)*gradT(elem,pt,1)*wts(elem,pt);
    });

    parallel_for("Thermal volume resid 2D",
                 MDRangePolicy<AssemblyExec,Kokkos::Rank<2>>({0,0},{basis.extent(0),basis.extent(1)}),
                 KOKKOS_LAMBDA (const int elem , const int dof) {
      EvalT val = 0.0;
      for (size_type pt=0; pt<basis.extent(2); ++pt) {
        val += f(elem,pt)*basis(elem,dof,pt,0) + DF(elem,pt,0)*basis_grad(elem,dof,pt,0) + DF(elem,pt,1)*basis_grad(elem,dof,pt,1);
      }
      res2(elem,dof) = val;
    });

    Kokkos::fence();
    double sol_time3 = timer.seconds();
    //printf("MD2 time:   %e \n", sol_time3);
    printf("MD2 ratio:   %e \n", sol_time3/sol_time1);
   
    /*
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 MDRangePolicy<AssemblyExec,Kokkos::Rank<2>>({0,0},{basis.extent(0),basis.extent(2)}),
                 KOKKOS_LAMBDA (const int elem , const int pt) {
      EvalT val = 0.0;
      EvalT f = -1.0*source(elem,pt);
      EvalT DFx = diff(elem,pt)*gradT(elem,pt,0);
      EvalT DFy = diff(elem,pt)*gradT(elem,pt,1);
      f *= wts(elem,pt);
      DFx *= wts(elem,pt);
      DFy *= wts(elem,pt);
      for (size_type dof=0; dof<res2.extent(1); ++dof ) {
        val = f*basis(elem,dof,pt,0) + DFx*basis_grad(elem,dof,pt,0) + DFy*basis_grad(elem,dof,pt,1);
        Kokkos::atomic_add(&(res2(elem,dof)),val);// += f*basis(elem,dof,pt,0) + DFx*basis_grad(elem,dof,pt,0) + DFy*basis_grad(elem,dof,pt,1);
      }
    });
    Kokkos::fence();
    double sol_time4 = timer.seconds();
    //printf("MD3 time:   %e \n", sol_time4);
    printf("MD3 ratio:   %e \n", sol_time4/sol_time1);
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 MDRangePolicy<AssemblyExec,Kokkos::Rank<3>>({0,0,0},{basis.extent(0),basis.extent(1),basis.extent(2)}),
                 KOKKOS_LAMBDA (const int elem , const int dof , const int pt) {
      EvalT val = 0.0;
      EvalT f = -1.0*source(elem,pt);
      EvalT DFx = diff(elem,pt)*gradT(elem,pt,0);
      EvalT DFy = diff(elem,pt)*gradT(elem,pt,1);
      f *= wts(elem,pt);
      DFx *= wts(elem,pt);
      DFy *= wts(elem,pt);
      //for (size_type dof=0; dof<res2.extent(1); ++dof ) {
        val = f*basis(elem,dof,pt,0) + DFx*basis_grad(elem,dof,pt,0) + DFy*basis_grad(elem,dof,pt,1);
        Kokkos::atomic_add(&(res2(elem,dof)),val);// += f*basis(elem,dof,pt,0) + DFx*basis_grad(elem,dof,pt,0) + DFy*basis_grad(elem,dof,pt,1);
      //}
    });
    Kokkos::fence();
    double sol_time5 = timer.seconds();
    //printf("MD3 time:   %e \n", sol_time4);
    printf("MD4 ratio:   %e \n", sol_time5/sol_time1);
    */
     
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
    
    /*
    {
      timer.reset();
      
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (size_type var=0; var<sol_dof.extent(1); var++) {
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            EvalT uval = sol_dof(elem,var,dof);
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              for (size_type s=0; s<basis.extent(3); s++ ) {
                sol_ip(elem,var,pt,s) += uval*basis(elem,dof,pt,s);
              }
            }
          }
        }
      });
      
      Kokkos::fence();
      double sol_time1 = timer.seconds();
      printf("Baseline time:   %e \n", sol_time1);
      
    }
    
    {
      timer.reset();
      
      parallel_for(Kokkos::MDRangePolicy<AssemblyExec, Kokkos::Rank<3>>({0,0,0},{basis.extent(0),sol_dof.extent(1),basis.extent(1)}), KOKKOS_LAMBDA (const int elem , const int var, const int dof) {
        //for (size_type var=0; var<sol_dof.extent(1); var++) {
        //  for (size_type dof=0; dof<basis.extent(1); dof++) {
            EvalT uval = sol_dof(elem,var,dof);
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              for (size_type s=0; s<basis.extent(3); s++ ) {
                sol_ip(elem,var,pt,s) += uval*basis(elem,dof,pt,s);
              }
            }
        //  }
        //}
      });
      
      Kokkos::fence();
      double sol_time1 = timer.seconds();
      printf("MD range time:   %e \n", sol_time1);
      
    }
    */
    /*
    {
      typedef Kokkos::TeamPolicy<AssemblyExec> TeamPolicy;
      const int vector_size = 1;
      const int team_size = 1;//256;
      
      timer.reset();
      
      //parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      parallel_for(TeamPolicy(basis.extent(0),team_size,vector_size), KOKKOS_LAMBDA (const typename TeamPolicy::member_type& team) {
        const size_t elem = team.league_rank();
        for (size_type var=0; var<sol_dof.extent(1); var++) {
          
          //for (int dof=team_index; dof<basis.extent(1); dof+=team_size) {
          for (size_type dof=0; dof<basis.extent(1); dof++) {
            EvalT uval = sol_dof(elem,var,dof);
            for (size_type pt=0; pt<basis.extent(2); pt++ ) {
              for (size_type s=0; s<basis.extent(3); s++ ) {
                sol_ip(elem,var,pt,s) += uval*basis(elem,dof,pt,s);
              }
            }

          }
        }
      });
      
      Kokkos::fence();
      double sol_time1 = timer.seconds();
      printf("GPU time (basic team):   %e \n", sol_time1);
      
    }
    */
    /*
    {
      typedef Kokkos::TeamPolicy<AssemblyExec> TeamPolicy;
      const int vector_size = 1;
      const int team_size = 1;//256;
      using Kokkos::TeamThreadRange;
      using Kokkos::parallel_for;
      using Kokkos::parallel_reduce;
      
      timer.reset();
      
      //parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      parallel_for(TeamPolicy(basis.extent(0),team_size,vector_size), KOKKOS_LAMBDA (const typename TeamPolicy::member_type& team) {
        const size_t elem = team.league_rank();
        const int team_index = team.team_rank();
        
        for (int var=0; var<sol_dof.extent(1); var++) {
          auto csol = Kokkos::subview(sol_dof,elem,var,Kokkos::ALL());
          auto cbasis = Kokkos::subview(basis,elem,Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
          auto csolip = Kokkos::subview(sol_ip,elem,var,Kokkos::ALL(),Kokkos::ALL());
          parallel_for(TeamThreadRange(team, basis.extent(1)), KOKKOS_LAMBDA (const int dof) {
            for (size_t pt=0; pt<basis.extent(2); pt++ ) {
              for (int s=0; s<basis.extent(3); s++ ) {
                csolip(pt,s) += csol(dof)*cbasis(dof,pt,s);
              }
            }
          });
        }
      });
      
      Kokkos::fence();
      double sol_time1 = timer.seconds();
      printf("GPU time (nested):   %e \n", sol_time1);
      
    }
     */
    
    /*
    {
      View4_host sol_ip("solution at ip",numElem,numvars,numip,dimension);
      Kokkos::View<ScalarT****,HostDevice> basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<EvalT***,HostDevice> sol_dof("sol at dof",numElem,numvars,numdof);
      
      parallel_for(RangePolicy<HostExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int dof=0; dof<basis.extent(1); dof++) {
          for (int pt=0; pt<basis.extent(2); pt++) {
            for (int dim=0; dim<basis.extent(3); dim++) {
              basis(elem,dof,pt,dim) = 1.0;
            }
          }
        }
      });
      parallel_for(RangePolicy<HostExec>(0,sol_dof.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int var=0; var<sol_dof.extent(1); var++) {
          for (int dof=0; dof<sol_dof.extent(2); dof++) {
            sol_dof(elem,var,dof) = 1.0;
          }
        }
      });
      
      timer.reset();
      for (int r=0; r<numrepeats; r++) {
        
        parallel_for(RangePolicy<HostExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          for (int var=0; var<sol_dof.extent(1); var++) {
            for (int dof=0; dof<sol_dof.extent(2); dof++) {
              EvalT uval = sol_dof(elem,var,dof);
              
              if (dof == 0) {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,var,pt,s) = uval*basis(elem,dof,pt,s);
                  }
                }
              }
              else {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,var,pt,s) += uval*basis(elem,dof,pt,s);
                  }
                }
              }
            }
          }
        });
        
      }
      Kokkos::fence();
      double sol_time = timer.seconds();
      printf("Host time 1:   %e \n", sol_time);
    }
    */
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


