
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
  typedef Kokkos::View<EvalT*,AssemblyDevice> View1;
  typedef Kokkos::View<EvalT**,AssemblyDevice> View2;
  typedef Kokkos::View<EvalT***,AssemblyDevice> View3;
  typedef Kokkos::View<EvalT****,AssemblyDevice> View4;
  typedef Kokkos::View<EvalT**,HostDevice> View2_host;
  typedef Kokkos::View<EvalT****,HostDevice> View4_host;
  
  
  
  {
    int numElem = 20000;
    int numip = 8;
    int numvars = 3;
    int dimension = 3;
    int numrepeats = 1;
    int numdof = 8;
    
    ////////////////////////////////////////////////
    // Baseline (current implementation in MrHyDE)
    ////////////////////////////////////////////////
    Kokkos::Timer timer;
 
    {
      //typedef Kokkos::TeamPolicy<AssemblyExec> TeamPolicy;
      //const int vector_size = 32;
      //const int team_size = 8;
 
      View4 sol_ip("solution at ip",numElem,numvars,numip,dimension);
      Kokkos::View<ScalarT****,AssemblyDevice> basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<EvalT***,AssemblyDevice> sol_dof("sol at dof",numElem,numvars,numdof);
      
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {

        for (int dof=0; dof<basis.extent(1); dof++) {
          for (int pt=0; pt<basis.extent(2); pt++) {
            for (int dim=0; dim<basis.extent(3); dim++) {
              basis(elem,dof,pt,dim) = 1.0;
            }
          }
        }
      });
      parallel_for(RangePolicy<AssemblyExec>(0,sol_dof.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int var=0; var<numvars; var++) {
          for (int dof=0; dof<sol_dof.extent(1); dof++) {
            sol_dof(elem,var,dof) = 1.0;
          }
        }
      });
      
      timer.reset();
      for (int r=0; r<1; r++) {
        
        parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        //parallel_for(TeamPolicy(basis.extent(0),team_size,vector_size), KOKKOS_LAMBDA (const typename TeamPolicy::member_type& team) {
          //const size_t elem = team.league_rank();
          //const int team_index = team.team_rank();
          for (int var=0; var<sol_dof.extent(1); var++) {

            //for (int dof=team_index; dof<basis.extent(1); dof+=team_size) {
            for (int dof=0; dof<basis.extent(1); dof++) {
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
      double sol_time1 = timer.seconds();
      printf("GPU time (multiple calls):   %e \n", sol_time1);
      
    }
    
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

    {
      //typedef Kokkos::TeamPolicy<AssemblyExec> TeamPolicy;
      //const int vector_size = 32;
      //const int team_size = 8;
 
      View2 sol_ip("solution at ip",numElem,numip);
      Kokkos::View<ScalarT***,AssemblyDevice> basis("basis",numElem,numdof,numip);
      Kokkos::View<EvalT**,AssemblyDevice> sol_dof("sol at dof",numElem,numdof);
      
      parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {

        for (int dof=0; dof<basis.extent(1); dof++) {
          for (int pt=0; pt<basis.extent(2); pt++) {
            //for (int dim=0; dim<basis.extent(3); dim++) {
              basis(elem,dof,pt) = 1.0;
            //}
          }
        }
      });
      parallel_for(RangePolicy<AssemblyExec>(0,sol_dof.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        //for (int var=0; var<numvars; var++) {
          for (int dof=0; dof<sol_dof.extent(1); dof++) {
            sol_dof(elem,dof) = 1.0;
          }
        //}
      });
      
      timer.reset();
      for (int r=0; r<1; r++) {
        
        parallel_for(RangePolicy<AssemblyExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        //parallel_for(TeamPolicy(basis.extent(0),team_size,vector_size), KOKKOS_LAMBDA (const typename TeamPolicy::member_type& team) {
          //const size_t elem = team.league_rank();
          //const int team_index = team.team_rank();
          //for (int var=0; var<sol_dof.extent(1); var++) {

            //for (int dof=team_index; dof<basis.extent(1); dof+=team_size) {
            for (int dof=0; dof<basis.extent(1); dof++) {
              EvalT uval = sol_dof(elem,dof);
              
              if (dof == 0) {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  //for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,pt) = uval*basis(elem,dof,pt);
                  //}
                }
              }
              else {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  //for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,pt) += uval*basis(elem,dof,pt);
                  //}
                }
              }
            }
          //}
        });
        
      }
      Kokkos::fence();
      double sol_time1 = timer.seconds();
      printf("GPU time (multiple calls):   %e \n", sol_time1);
      
    }
    
    {
      View2_host sol_ip("solution at ip",numElem,numip);
      Kokkos::View<ScalarT***,HostDevice> basis("basis",numElem,numdof,numip);
      Kokkos::View<EvalT**,HostDevice> sol_dof("sol at dof",numElem,numdof);
      
      parallel_for(RangePolicy<HostExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int dof=0; dof<basis.extent(1); dof++) {
          for (int pt=0; pt<basis.extent(2); pt++) {
            //for (int dim=0; dim<basis.extent(3); dim++) {
              basis(elem,dof,pt) = 1.0;
            //}
          }
        }
      });
      parallel_for(RangePolicy<HostExec>(0,sol_dof.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        //for (int var=0; var<sol_dof.extent(1); var++) {
          for (int dof=0; dof<sol_dof.extent(2); dof++) {
            sol_dof(elem,dof) = 1.0;
          }
        //}
      });
      
      timer.reset();
      for (int r=0; r<numrepeats; r++) {
        
        parallel_for(RangePolicy<HostExec>(0,basis.extent(0)), KOKKOS_LAMBDA (const int elem ) {
          //for (int var=0; var<sol_dof.extent(1); var++) {
            for (int dof=0; dof<sol_dof.extent(2); dof++) {
              EvalT uval = sol_dof(elem,dof);
              
              if (dof == 0) {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  //for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,pt) = uval*basis(elem,dof,pt);
                  //}
                }
              }
              else {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  //for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,pt) += uval*basis(elem,dof,pt);
                  //}
                }
              }
            }
          //}
        });
        
      }
      Kokkos::fence();
      double sol_time = timer.seconds();
      printf("Host time 1:   %e \n", sol_time);
    }

    //////////////////////////////////////
    // Try a DRV
    ///////////////////////////////////////
    /* 
    {
      View4 sol_ip("solution at ip",numElem,numvars,numip,dimension);
      DRV basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<ScalarT***,AssemblyExec> sol_dof("sol at dof",numElem,numvars,numdof);
      
      for (int elem=0; elem<numElem; elem++) {
        for (int pt=0; pt<numip; pt++) {
          for (int dof=0; dof<numdof; dof++) {
            for (int dim=0; dim<dimension; dim++) {
              basis(elem,dof,pt,dim) = 1.0;
            }
          }
        }
      }
      for (int elem=0; elem<numElem; elem++) {
        for (int var=0; var<numvars; var++) {
          for (int dof=0; dof<numdof; dof++) {
            sol_dof(elem,var,dof) = 1.0;
          }
        }
      }
      
      timer.reset();
      for (int r=0; r<numrepeats; r++) {
        
        for (int var=0; var<numvars; var++) {
          for (int elem=0; elem<numElem; elem++) {
            for (int dof=0; dof<numdof; dof++) {
              AD uval = AD(18,dof,sol_dof(elem,var,dof));
              
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
        }
        
      }
      Kokkos::fence();
      double sol_time2 = timer.seconds();
      
      //printf("Deep copy time:   %e \n", copy_time);
      printf("sol time 2:   %e \n", sol_time2);
    }
    */
    ////////////////////////////////////////////////
    // Try reorganizing the data
    ////////////////////////////////////////////////
    /*
    {
      Kokkos::View<ScalarT****,AssemblyExec> sol_ip("solution at ip",numElem,numvars,numip,dimension);
      Kokkos::View<ScalarT****,AssemblyExec> basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<ScalarT***,AssemblyExec> sol_dof("sol at dof",numElem,numvars,numdof);
      
      for (int elem=0; elem<numElem; elem++) {
        for (int pt=0; pt<numip; pt++) {
          for (int dof=0; dof<numdof; dof++) {
            for (int dim=0; dim<dimension; dim++) {
              basis(elem,dof,pt,dim) = 1.0;
            }
          }
        }
      }
      for (int elem=0; elem<numElem; elem++) {
        for (int var=0; var<numvars; var++) {
          for (int dof=0; dof<numdof; dof++) {
            sol_dof(elem,var,dof) = 1.0;
          }
        }
      }
      
      timer.reset();
      for (int r=0; r<numrepeats; r++) {
        for (int elem=0; elem<numElem; elem++) {
          for (int var=0; var<numvars; var++) {
            
            for (int dof=0; dof<numdof; dof++) {
              ScalarT uval = sol_dof(elem,var,dof);//AD(18,dof,sol_dof(elem,dof));
              if (dof == 0) {
                for (size_t pt=0; pt<numip; pt++ ) {
                  for (int dim=0; dim<dimension; dim++) {
                    sol_ip(elem,var,pt,dim) = uval*basis(elem,dof,pt,dim);
                  }
                }
              }
              else {
                for (size_t pt=0; pt<numip; pt++ ) {
                  for (int dim=0; dim<dimension; dim++) {
                    sol_ip(elem,var,pt,dim) += uval*basis(elem,dof,pt,dim);
                  }
                }
              }
            }
          }
        }
      }
      Kokkos::fence();
      double sol_time3 = timer.seconds();
      
      //printf("Deep copy time:   %e \n", copy_time);
      printf("sol time 3:   %e \n", sol_time3);
    }
    */
    ////////////////////////////////////////////////
    // Baseline (current implementation in MrHyDE)
    ////////////////////////////////////////////////
    /*
    {
      View4 sol_ip("solution at ip",numElem,numvars,numip,dimension);
      Kokkos::View<ScalarT****,AssemblyExec> basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<ScalarT***,AssemblyExec> sol_dof("sol at dof",numElem,numvars,numdof);
      View3 sol_dof_AD("seeded solution at dof",numElem,numvars,numdof);
      for (int elem=0; elem<numElem; elem++) {
        for (int pt=0; pt<numip; pt++) {
          for (int dof=0; dof<numdof; dof++) {
            for (int dim=0; dim<dimension; dim++) {
              basis(elem,dof,pt,dim) = 1.0;
            }
          }
        }
      }
      for (int elem=0; elem<numElem; elem++) {
        for (int var=0; var<numvars; var++) {
          for (int dof=0; dof<numdof; dof++) {
            sol_dof(elem,var,dof) = 1.0;
          }
        }
      }
      
      for (int var=0; var<numvars; var++) {
        for (int elem=0; elem<numElem; elem++) {
          for (int dof=0; dof<numdof; dof++) {
            sol_dof_AD(elem,var,dof) = AD(18,dof,sol_dof(elem,var,dof));
          }
        }
      }
      
      timer.reset();
      for (int r=0; r<numrepeats; r++) {
        
        for (int var=0; var<numvars; var++) {
          for (int elem=0; elem<numElem; elem++) {
            for (int dof=0; dof<numdof; dof++) {
              
              if (dof == 0) {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,var,pt,s) = sol_dof_AD(elem,var,dof)*basis(elem,dof,pt,s);
                  }
                }
              }
              else {
                for (size_t pt=0; pt<basis.extent(2); pt++ ) {
                  for (int s=0; s<basis.extent(3); s++ ) {
                    sol_ip(elem,var,pt,s) += sol_dof_AD(elem,var,dof)*basis(elem,dof,pt,s);
                  }
                }
              }
            }
          }
        }
        
      }
      Kokkos::fence();
      double sol_time4 = timer.seconds();
      printf("sol time 4:   %e \n", sol_time4);
    }
    */
    /*
    {
      Kokkos::View<ScalarT***,AssemblyExec> sol_ip("solution at ip",numElem,numip,dimension);
      Kokkos::View<ScalarT****,AssemblyExec> basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<ScalarT**,AssemblyExec> sol_dof("sol at dof",numElem,numdof);
      
      for (int elem=0; elem<numElem; elem++) {
        for (int pt=0; pt<numip; pt++) {
          for (int dof=0; dof<numdof; dof++) {
            for (int dim=0; dim<dimension; dim++) {
              basis(elem,dof,pt,dim) = 1.0;
            }
          }
        }
      }
      for (int elem=0; elem<numElem; elem++) {
        for (int dof=0; dof<numdof; dof++) {
          sol_dof(elem,dof) = 1.0;
        }
      }
      
      timer.reset();
      for (int r=0; r<numrepeats; r++) {
        for (int elem=0; elem<numElem; elem++) {
          for (int dof=0; dof<numdof; dof++) {
            ScalarT uval = sol_dof(elem,dof);//AD(18,dof,sol_dof(elem,dof));
            if (dof == 0) {
              for (size_t pt=0; pt<numip; pt++ ) {
                for (int dim=0; dim<dimension; dim++) {
                  sol_ip(elem,pt,dim) = uval*basis(elem,dof,pt,dim);
                }
              }
            }
            else {
              for (size_t pt=0; pt<numip; pt++ ) {
                for (int dim=0; dim<dimension; dim++) {
                  sol_ip(elem,pt,dim) += uval*basis(elem,dof,pt,dim);
                }
              }
            }
          }
        }
        
      }
      Kokkos::fence();
      double sol_time5 = timer.seconds();
      
      //printf("Deep copy time:   %e \n", copy_time);
      printf("sol time 5:   %e \n", sol_time5);
    }
    */
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


