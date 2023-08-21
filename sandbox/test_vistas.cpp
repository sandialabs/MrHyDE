
#include "trilinos.hpp"
#include "preferences.hpp"
#include "vista.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  #ifndef MrHyDE_NO_AD
    typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2;
  #else
    typedef View_Sc2 View_AD2;
  #endif
    
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  using MrHyDE::Vista;
  
  typedef Kokkos::TeamPolicy<AssemblyExec> TeamPolicy;
  typedef TeamPolicy::member_type member_type;

  {
    int numElem = 1000;
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
    ScalarT vad = 5.6;
    View_Sc2 valview("view of vals",numElem, numip);
    deep_copy(valview,val);
    
    View_AD2 vadview("view of vads",numElem, numip);
    deep_copy(vadview,vad);
    
    auto v1 = Vista<AD>(val);
    Vista v2 = Vista<AD>(vad);
    Vista v3 = Vista<AD>(valview);
    Vista v4 = Vista<AD>(vadview);
    
    View_AD2 tstview("view of vads",numElem, numip);
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*v1(elem,pt);
      }
    });
    
    cout << "Vista Scalar Range time: " << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 TeamPolicy(tstview.extent(0), 1, 32),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      for (size_type pt=team.team_rank(); pt<tstview.extent(1); pt+=team.team_size() ) {
        tstview(elem,pt) = 2.0*v1(elem,pt);
      }
    });
    
    cout << "Vista Scalar Team time: " << timer.seconds() << endl;
    
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*v2(elem,pt);
      }
    });
    
    cout << "Vista AD Range time: " << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 TeamPolicy(tstview.extent(0), 1, 32),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      for (size_type pt=team.team_rank(); pt<tstview.extent(1); pt+=team.team_size() ) {
        tstview(elem,pt) = 2.0*v2(elem,pt);
      }
    });
    
    cout << "Vista AD Team time: " << timer.seconds() << endl;
    
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*v3(elem,pt);
      }
    });
    
    cout << "Vista View Scalar Range time: " << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 TeamPolicy(tstview.extent(0), 1, 32),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      for (size_type pt=team.team_rank(); pt<tstview.extent(1); pt+=team.team_size() ) {
        tstview(elem,pt) = 2.0*v3(elem,pt);
      }
    });
    
    cout << "Vista View Scalar Team time: " << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*v4(elem,pt);
      }
    });
    
    cout << "Vista View AD Range time: " << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 TeamPolicy(tstview.extent(0), 1, 32),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      for (size_type pt=team.team_rank(); pt<tstview.extent(1); pt+=team.team_size() ) {
        tstview(elem,pt) = 2.0*v4(elem,pt);
      }
    });
    
    cout << "Vista View AD Team time: " << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*valview(elem,pt);
      }
    });
    
    cout << "Reference View Scalar Range time: " << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 TeamPolicy(tstview.extent(0), 1, 32),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      for (size_type pt=team.team_rank(); pt<tstview.extent(1); pt+=team.team_size() ) {
        tstview(elem,pt) = 2.0*valview(elem,pt);
      }
    });
    
    cout << "Reference View Scalar Team time: " << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 RangePolicy<AssemblyExec>(0,tstview.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type pt=0; pt<tstview.extent(1); ++pt) {
        tstview(elem,pt) = 2.0*vadview(elem,pt);
      }
    });
    
    cout << "Reference View AD Range time: " << timer.seconds() << endl;
    
    timer.reset();
    parallel_for("Thermal volume resid 2D",
                 TeamPolicy(tstview.extent(0), 1, 32),
                 KOKKOS_LAMBDA (member_type team ) {
      int elem = team.league_rank();
      for (size_type pt=team.team_rank(); pt<tstview.extent(1); pt+=team.team_size() ) {
        tstview(elem,pt) = 2.0*vadview(elem,pt);
      }
    });
    cout << "Reference View AD Team time: " << timer.seconds() << endl;
    
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


