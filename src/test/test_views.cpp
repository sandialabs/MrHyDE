
#include "trilinos.hpp"
#include "preferences.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  typedef Sacado::Fad::SFad<ScalarT,64> EvalT;
  typedef Kokkos::View<EvalT*,AssemblyDevice> View1;
  typedef Kokkos::View<EvalT**,AssemblyDevice> View2;
  typedef Kokkos::View<EvalT***,AssemblyDevice> View3;
  typedef Kokkos::View<EvalT****,AssemblyDevice> View4;
  typedef Kokkos::View<EvalT****,HostDevice> View4_host;
  
  
  
  {
    int numElem = 10000;
    int numip = 8;
    int numvars = 2;
    int dimension = 3;
    int numrepeats = 20;
    int numdof = 9;
    
    //View2 A("test",numElem,numvars);
    //View3 A("test",numElem,numvars,numdof);
    View4 A("test",numElem,numvars,numip,dimension,64);
    View4_host host_A("test",numElem,numvars,numip,dimension,64);
    View4_host host_A2("test",numElem,numvars,numip,dimension);

    
    auto host_A_sv = Kokkos::subview(host_A,10,0,Kokkos::ALL(), Kokkos::ALL());
    
    // Run some performance comparisons
    Kokkos::Timer timer;
    
    
    timer.reset();
    for (int r=0; r<numrepeats; r++) {
      //A = View2("test",numElem,numvars);
      //A = View3("test",numElem,numvars,numdof);
      A = View4("test",numElem,numvars,numip,dimension);
    }
    Kokkos::fence();
    double allocate_time = timer.seconds();
    printf("Allocate time:   %e \n", allocate_time);
    
    ScalarT val = 0.0;
    
    timer.reset();
    for (int r=0; r<numrepeats; r++) {
      parallel_for(RangePolicy<AssemblyExec>(0,A.extent(0)), KOKKOS_LAMBDA (const int e ) {
        //EvalT val = 0.0;
        for (int n=0; n<A.extent(1); n++) {
          //A(e,n) = val;
          for (int k=0; k<A.extent(2); k++) {
          //  A(e,n,k) = val;
            for (int s=0; s<A.extent(3); s++) {
              //A(e,n,k,s) = val;
              AD newval = AD(64,30,0.0);
              A(e,n,k,s) = 1.0+newval;
              //A(e,n,k,s).zero();
            }
          }
        }
      });
    }
    Kokkos::fence();
    double assign_time = timer.seconds();
    printf("Assign time:   %e \n", assign_time);
    
    //View2 A_ref("ref vals",numElem,numvars);
    //View3 A_ref("ref vals",numElem,numvars,numip);
    //View4 A_ref("ref vals",numElem,numvars,numip,dimension);
    timer.reset();
    for (int r=0; r<numrepeats; r++) {
      Kokkos::deep_copy(A,0.0);
    }
    Kokkos::fence();
    double copy_time = timer.seconds();
    printf("Copy time:   %e \n", copy_time);
    
    ////////////////////////////////////////////////
    // How long does it take for one add/mult
    ////////////////////////////////////////////////
    
    EvalT a=1.0,b=1.0;
    ScalarT c = 1.0;
    
    timer.reset();
    for (int i=0; i<1000000; i++) {
      ScalarT c = 1.0;
      a += b*c;
    }
    double addmulttime = timer.seconds()/1000000;
    printf("time for one add mult:   %.16e \n", addmulttime);
    printf("optimal time:   %e \n", addmulttime*numElem*numvars*numdof*numip*dimension*numrepeats);
    cout << "Number of add/mults: " << numElem*numvars*numdof*numip*dimension*numrepeats << endl;
    ////////////////////////////////////////////////
    // Baseline (current implementation in MrHyDE)
    ////////////////////////////////////////////////
    
    {
      View4 sol_ip("solution at ip",numElem,numvars,numip,dimension,64);
      Kokkos::View<ScalarT****,AssemblyDevice> basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<ScalarT***,AssemblyDevice> sol_dof("sol at dof",numElem,numvars,numdof);
      
      //for (int elem=0; elem<numElem; elem++) {
      parallel_for(RangePolicy<AssemblyExec>(0,A.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int pt=0; pt<numip; pt++) {
          for (int dof=0; dof<numdof; dof++) {
            for (int dim=0; dim<dimension; dim++) {
              basis(elem,dof,pt,dim) = 1.0;
            }
          }
        }
      });
      parallel_for(RangePolicy<AssemblyExec>(0,A.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      //for (int elem=0; elem<numElem; elem++) {
        for (int var=0; var<numvars; var++) {
          for (int dof=0; dof<numdof; dof++) {
            sol_dof(elem,var,dof) = 1.0;
          }
        }
      });
      
      timer.reset();
      for (int r=0; r<100; r++) {
        
        parallel_for(RangePolicy<AssemblyExec>(0,A.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        //for (int elem=0; elem<numElem; elem++) {
          for (int var=0; var<sol_dof.extent(1); var++) {
            for (int dof=0; dof<sol_dof.extent(2); dof++) {
              AD uval = AD(64,dof,sol_dof(elem,var,dof));
              
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
      
      timer.reset();
      
      parallel_for(RangePolicy<AssemblyExec>(0,A.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int r=0; r<100; r++) {
          //for (int elem=0; elem<numElem; elem++) {
          for (int var=0; var<sol_dof.extent(1); var++) {
            for (int dof=0; dof<sol_dof.extent(2); dof++) {
              AD uval = AD(64,dof,sol_dof(elem,var,dof));
              
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
      });
      
      Kokkos::fence();
      sol_time1 = timer.seconds();
      printf("GPU time (inner repeats):   %e \n", sol_time1);
    }
    
    {
      View4_host sol_ip("solution at ip",numElem,numvars,numip,dimension,64);
      Kokkos::View<ScalarT****,HostDevice> basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<ScalarT***,HostDevice> sol_dof("sol at dof",numElem,numvars,numdof);
      
      //for (int elem=0; elem<numElem; elem++) {
      parallel_for(RangePolicy<HostExec>(0,A.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        for (int pt=0; pt<numip; pt++) {
          for (int dof=0; dof<numdof; dof++) {
            for (int dim=0; dim<dimension; dim++) {
              basis(elem,dof,pt,dim) = 1.0;
            }
          }
        }
      });
      parallel_for(RangePolicy<HostExec>(0,A.extent(0)), KOKKOS_LAMBDA (const int elem ) {
      //for (int elem=0; elem<numElem; elem++) {
        for (int var=0; var<numvars; var++) {
          for (int dof=0; dof<numdof; dof++) {
            sol_dof(elem,var,dof) = 1.0;
          }
        }
      });
      
      timer.reset();
      for (int r=0; r<numrepeats; r++) {
        
        parallel_for(RangePolicy<HostExec>(0,A.extent(0)), KOKKOS_LAMBDA (const int elem ) {
        //for (int elem=0; elem<numElem; elem++) {
          for (int var=0; var<sol_dof.extent(1); var++) {
            for (int dof=0; dof<sol_dof.extent(2); dof++) {
              AD uval = AD(64,dof,sol_dof(elem,var,dof));
              
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


