
#include "Sacado.hpp"

using namespace std;

int main(int argc, char * argv[]) {

  // We are going to use specific type of AD with a fixed number of derivatives
  // We only need one derivative for this exercise
  typedef Sacado::Fad::SFad<double,1> AD;

  // First, let's create an AD variable and see what it is
  {
    AD x = 4.2;
    std::cout << "x = " << x << std::endl;
    
    // We can also access the value and derivatives
    std::cout << "x.val() = " << x.val() << std::endl;
    std::cout << "x.fastAccessDx(0) = " << x.fastAccessDx(0) << std::endl;
  }
  
  // Note that the derivative is set to 0
  // Let's create an AD variable that is "seeded"
  // We will use a different constructor: AD(number of derivs, deriv to seed, value)
  {
    AD x(1,0,4.2);
    std::cout << "x = " << x << std::endl;
  }
  
  // Let's create some more AD variables that depend on x and check the derivatives
  {
    AD x(1,0,4.2);
    AD f = cos(x);
    std::cout << "f = " << f << std::endl;
    
    double true_dfdx = -sin(x.val());
    std::cout << "Analytical value of df/dx = " << true_dfdx << std::endl;
    std::cout << "Error in derivative = " << abs(f.fastAccessDx(0) - true_dfdx) << std::endl;
    
    // We can also propagate derivatives through functions of f
    AD g = exp(f);
    std::cout << "g = " << g << std::endl;
    
    double true_dgdx = -sin(x.val())*exp(f.val());
    std::cout << "Analytical value of dg/dx = " << true_dgdx << std::endl;
    std::cout << "Error in derivative = " << abs(g.fastAccessDx(0) - true_dgdx) << std::endl;
    
  }
  
  return 0;
}


