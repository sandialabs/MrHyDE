
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
    int numElem = 100;
    if (argc>1) {
      numElem = atof(argv[1]);
    }
    std::cout << "Number of elements: " << numElem << std::endl;
 
#define TeamSize 1
    std::cout << "Team size: " << TeamSize << std::endl;
  
    typedef double EvalT;
    //typedef Sacado::Fad::SFad<ScalarT,32> EvalT;
#define VSize 1
    std::cout << "Vector size: " << VSize << std::endl;
    
    int numip = 8;
    int dimension = 3;
    int numdof = 12;
    
    typedef Kokkos::LayoutContiguous<AssemblyExec::array_layout,VSize> CL;

    ////////////////////////////////////////////////
    // Set up timer and views
    ////////////////////////////////////////////////
    
    Kokkos::Timer timer;
 
    ////////////////////////////////////////////////
    // Another key kernel
    ////////////////////////////////////////////////

    {
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis_grad("basis",numElem,numdof,numip,dimension);
      
      Kokkos::deep_copy(cbasis,1.0);
      Kokkos::deep_copy(cbasis_grad,2.0);
      
      Kokkos::View<EvalT**,CL,AssemblyDevice> cuvals("sol",numElem,numdof);
      Kokkos::deep_copy(cuvals,100.0);
      
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol("diff",numElem,numip);
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol_x("src",numElem,numip);
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol_y("src",numElem,numip);
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol_z("src",numElem,numip);
      
      ////////////////////////////////////////////////
      // Hierarchical modified
      ////////////////////////////////////////////////
      
      timer.reset();
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy(cbasis.extent(0), Kokkos::AUTO, VSize),
                   KOKKOS_LAMBDA (member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csol(elem,pt) = cuvals(elem,0)*cbasis(elem,0,pt,0);
          for (size_type dof=1; dof<cbasis.extent(1); dof++ ) {
            csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
          }
          csol_x(elem,pt) = cuvals(elem,0)*cbasis_grad(elem,0,pt,0);
          csol_y(elem,pt) = cuvals(elem,0)*cbasis_grad(elem,0,pt,1);
          csol_z(elem,pt) = cuvals(elem,0)*cbasis_grad(elem,0,pt,2);
          for (size_type dof=1; dof<cbasis.extent(1); dof++ ) {
            csol_x(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,0);
            csol_y(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,1);
            csol_z(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,2);
          }
          
        }
      });
    }
    
    Kokkos::fence();
    double ker2_time3 = timer.seconds();
    printf("Btime 1:   %e \n", ker2_time3);
    
    {
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis("basis",numElem,numip,numdof,dimension);
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis_grad("basis",numElem,numip,numdof,dimension);
      
      Kokkos::deep_copy(cbasis,1.0);
      Kokkos::deep_copy(cbasis_grad,2.0);
      
      Kokkos::View<EvalT**,CL,AssemblyDevice> cuvals("sol",numElem,numdof);
      Kokkos::deep_copy(cuvals,100.0);
      
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol("diff",numElem,numip);
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol_x("src",numElem,numip);
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol_y("src",numElem,numip);
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol_z("src",numElem,numip);
      
      ////////////////////////////////////////////////
      // Hierarchical modified
      ////////////////////////////////////////////////
      
      timer.reset();
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy(cbasis.extent(0), Kokkos::AUTO, VSize),
                   KOKKOS_LAMBDA (member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<cbasis.extent(1); pt+=team.team_size() ) {
          csol(elem,pt) = cuvals(elem,0)*cbasis(elem,pt,0,0);
          for (size_type dof=1; dof<cbasis.extent(2); dof++ ) {
            csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,pt,dof,0);
          }
          csol_x(elem,pt) = cuvals(elem,0)*cbasis_grad(elem,pt,0,0);
          csol_y(elem,pt) = cuvals(elem,0)*cbasis_grad(elem,pt,0,1);
          csol_z(elem,pt) = cuvals(elem,0)*cbasis_grad(elem,pt,0,2);
          for (size_type dof=1; dof<cbasis.extent(2); dof++ ) {
            csol_x(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,pt,dof,0);
            csol_y(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,pt,dof,1);
            csol_z(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,pt,dof,2);
          }
          
        }
      });
    }
    
    Kokkos::fence();
    printf("Btime 2:   %e \n", timer.seconds());


    {
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<ScalarT****,AssemblyDevice> cbasis_grad("basis",numElem,numdof,numip,dimension);
      
      Kokkos::deep_copy(cbasis,1.0);
      Kokkos::deep_copy(cbasis_grad,2.0);
      
      Kokkos::View<EvalT**,CL,AssemblyDevice> cuvals("sol",numElem,numdof);
      Kokkos::deep_copy(cuvals,100.0);
      
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol("diff",numElem,numip);
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol_x("src",numElem,numip);
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol_y("src",numElem,numip);
      Kokkos::View<EvalT**,CL,AssemblyDevice> csol_z("src",numElem,numip);
      
      ////////////////////////////////////////////////
      // Hierarchical modified
      ////////////////////////////////////////////////
      
      timer.reset();
      parallel_for("wkset soln ip HGRAD",
                   TeamPolicy(cbasis.extent(0), Kokkos::AUTO, VSize),
                   KOKKOS_LAMBDA (member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
          csol(elem,pt) = cuvals(elem,0)*cbasis(elem,0,pt,0);
          csol_x(elem,pt) = cuvals(elem,0)*cbasis_grad(elem,0,pt,0);
          csol_y(elem,pt) = cuvals(elem,0)*cbasis_grad(elem,0,pt,1);
          csol_z(elem,pt) = cuvals(elem,0)*cbasis_grad(elem,0,pt,2);
        }
        
        for (size_type dof=1; dof<cbasis.extent(1); dof++ ) {
          for (size_type pt=team.team_rank(); pt<cbasis.extent(2); pt+=team.team_size() ) {
            csol(elem,pt) += cuvals(elem,dof)*cbasis(elem,dof,pt,0);
            csol_x(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,0);
            csol_y(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,1);
            csol_z(elem,pt) += cuvals(elem,dof)*cbasis_grad(elem,dof,pt,2);
          }
          
        }
      });
    }
    
    Kokkos::fence();
    double ker3_time3 = timer.seconds();
    printf("Btime 3:   %e \n", ker3_time3);
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


