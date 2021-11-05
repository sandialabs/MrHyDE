
#include "trilinos.hpp"
#include "preferences.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  {
    int numElem = 10000;
    int numip = 4;
    int dimension = 3;
    int numdof = 12;
    int numTerms = 10000;
    typedef double EvalT;
    
    ////////////////////////////////////////////////
    // Set up timer and views
    ////////////////////////////////////////////////
    
    Kokkos::Timer timer;
 
    ////////////////////////////////////////////////
    // Another key kernel
    ////////////////////////////////////////////////

    {
      Kokkos::View<ScalarT****,AssemblyDevice> basis("basis",numElem,numdof,numip,dimension);
      Kokkos::View<EvalT**,AssemblyDevice> res("res",numElem,numdof);
      
      Kokkos::View<ScalarT**,AssemblyDevice> nx("nx",numElem,numip);
      Kokkos::View<ScalarT**,AssemblyDevice> ny("ny",numElem,numip);
      Kokkos::View<ScalarT**,AssemblyDevice> nz("nz",numElem,numip);
      
      Kokkos::View<ScalarT**,AssemblyDevice> x("x",numElem,numip);
      Kokkos::View<ScalarT**,AssemblyDevice> y("y",numElem,numip);
      Kokkos::View<ScalarT**,AssemblyDevice> z("z",numElem,numip);
      
      Kokkos::View<ScalarT**,AssemblyDevice> wts("wts",numElem,numip);
      
      Kokkos::View<ScalarT**,AssemblyDevice> A("A",numTerms,dimension);
      Kokkos::View<ScalarT*,AssemblyDevice> k("k",numTerms);
      
      // Just fill with something
      Kokkos::deep_copy(basis,1.0);
      Kokkos::deep_copy(nx,1.0);
      Kokkos::deep_copy(ny,0.0);
      Kokkos::deep_copy(nz,0.0);
      
      Kokkos::deep_copy(x,1.0);
      Kokkos::deep_copy(y,1.0);
      Kokkos::deep_copy(z,1.0);
      
      Kokkos::deep_copy(wts,1.0);
      Kokkos::deep_copy(A,1.0);
      
      double gamma = 0.0;
      double time = 1.0e-15;
      Kokkos::View<EvalT**,AssemblyDevice> Ex("Ex",numElem,numip);
      Kokkos::View<EvalT**,AssemblyDevice> Ey("Ey",numElem,numip);
      Kokkos::View<EvalT**,AssemblyDevice> Ez("Ez",numElem,numip);
      
      
      timer.reset();
      for (size_type elem=0; elem<numElem; ++elem) {
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (int term=0; term<numTerms; ++term) {
            Ex(elem,pt) += A(term,0)*(std::sin(k(term)*x(elem,pt)*time) + std::cos(k(term)*x(elem,pt)*time));
            Ey(elem,pt) += A(term,1)*(std::sin(k(term)*y(elem,pt)*time) + std::cos(k(term)*y(elem,pt)*time));
            Ez(elem,pt) += A(term,2)*(std::sin(k(term)*z(elem,pt)*time) + std::cos(k(term)*z(elem,pt)*time));
          }
        }
      }
      
      printf("Time to construct series for E (std for):   %e \n", timer.seconds());
      
      timer.reset();
      parallel_for("maxwell bndry resid ABC",
                   RangePolicy<AssemblyExec>(0,numElem),
                   KOKKOS_LAMBDA (const int elem ) {
    
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          for (int term=0; term<numTerms; ++term) {
            Ex(elem,pt) += A(term,0)*(std::sin(k(term)*x(elem,pt)*time) + std::cos(k(term)*x(elem,pt)*time));
            Ey(elem,pt) += A(term,1)*(std::sin(k(term)*y(elem,pt)*time) + std::cos(k(term)*y(elem,pt)*time));
            Ez(elem,pt) += A(term,2)*(std::sin(k(term)*z(elem,pt)*time) + std::cos(k(term)*z(elem,pt)*time));
          }
        }
      });
      
      printf("Time to construct series for E (kokkos for):   %e \n", timer.seconds());
      
                   
      timer.reset();
      
      parallel_for("maxwell bndry resid ABC",
                   RangePolicy<AssemblyExec>(0,numElem),
                   KOKKOS_LAMBDA (const int elem ) {
    
        for (size_type pt=0; pt<basis.extent(2); pt++ ) {
          EvalT nce_x = ny(elem,pt)*Ez(elem,pt) - nz(elem,pt)*Ey(elem,pt);
          EvalT nce_y = nz(elem,pt)*Ex(elem,pt) - nx(elem,pt)*Ez(elem,pt);
          EvalT nce_z = nx(elem,pt)*Ey(elem,pt) - ny(elem,pt)*Ex(elem,pt);
          EvalT c0 = -(1.0+gamma)*(ny(elem,pt)*nce_z - nz(elem,pt)*nce_y)*wts(elem,pt);
          EvalT c1 = -(1.0+gamma)*(nz(elem,pt)*nce_x - nx(elem,pt)*nce_z)*wts(elem,pt);
          EvalT c2 = -(1.0+gamma)*(nx(elem,pt)*nce_y - ny(elem,pt)*nce_x)*wts(elem,pt);
          for (size_type dof=0; dof<basis.extent(1); dof++ ) {
            res(elem,dof) += c0*basis(elem,dof,pt,0) + c1*basis(elem,dof,pt,1) + c2*basis(elem,dof,pt,2);
          }
        }
      });
      
      printf("Time to compute residual:   %e \n", timer.seconds());
      
    }
    
    
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


