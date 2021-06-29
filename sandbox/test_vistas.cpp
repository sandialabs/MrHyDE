
#include "trilinos.hpp"
#include "preferences.hpp"
#include "vista.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  using MrHyDE::Vista;
  
  {
    int numElem = 100000;
    if (argc>1) {
      numElem = atof(argv[1]);
    }
    std::cout << "Number of elements: " << numElem << std::endl;
 
    int numip = 8;
    
    ////////////////////////////////////////////////
    // Set up timer and views
    ////////////////////////////////////////////////
    
    Kokkos::Timer timer;
 
    ScalarT val = 1.2;
    AD vad = 5.6;
    View_Sc2 valview("view of vals",numElem, numip);
    deep_copy(valview,val);
    
    View_AD2 vadview("view of vads",numElem, numip);
    deep_copy(vadview,vad);
    
    Vista v1 = Vista(val);
    Vista v2 = Vista(vad);
    Vista v3 = Vista(valview);
    Vista v4 = Vista(vadview);
    
    size_type i0 = 1;
    size_type i1 = 4;
    cout << v1(i0,i1) << endl;
    cout << v2(i0,i1) << endl;
    cout << v3(i0,i1) << endl;
    cout << v4(i0,i1) << endl;
    
    
    View_AD2 tstview("view of vads",numElem, numip);
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*v1(elem,pt);
      }
    });
    
    cout << timer.seconds() << endl;
    
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*v2(elem,pt);
      }
    });
    
    cout << timer.seconds() << endl;
    
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*v3(elem,pt);
      }
    });
    
    cout << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*valview(elem,pt);
      }
    });
    
    cout << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*v4(elem,pt);
      }
    });
    
    cout << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*vadview(elem,pt);
      }
    });
    
    cout << timer.seconds() << endl;
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


