
#include "Kokkos_Core.hpp"

// Kokkos doesn't have print options for Views
// There are very good reasons for this
// But we want to print things here
void print(Kokkos::View<double*> V, const std::string & message="") {
  
  std::cout << std::endl;
  std::cout << message << std::endl;
  std::cout << "Printing data for View: " << V.label() << std::endl;
  
  std::cout << "  i  " << "  value  " << std::endl;
  std::cout << "--------------------" << std::endl;
  
  for (long unsigned int i=0; i<V.extent(0); i++) {
    std::cout << "  " << i << "  " << "  " << "  " << V(i) << "  " << std::endl;
  }
  std::cout << "--------------------" << std::endl;
  
}

// This is a very basic introduction to using Kokkos::Views

int main(int argc, char * argv[]) {

  Kokkos::initialize();

  {
    // Let's create 1D array and play with it
    int N = 5;
    Kokkos::View<double*> a("a",N);
    
    print(a,"Initial values");
    
    a(2) = 5.46;
    print(a,"After modifying an entry");
    
    Kokkos::deep_copy(a,1.0);
    print(a,"After performing deep copy");
    
    for (long unsigned int k=0; k<a.extent(0); ++k) {
      a(k) = 2.0*k;
    }
    print(a,"After modifying in a loop");
    
    // Let's create another View
    Kokkos::View<double*> b("b",N);
    print(b,"Initial values of b");
    
    // Setting b equal to a just changes the pointer to the data
    b = a;
    print(b,"New values"); // It's even called "a" now
    
    a(0) = -1.0;
    print(b,"New values after changing a");
    
  }
  
  Kokkos::finalize();
  
  return 0;
}


