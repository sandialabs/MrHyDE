/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "test_template.hpp"
//template class TT<int>;

template<class M>
TT<M>::TT() {
  a = 0.0;
  b = 0.0;
}
template<class M>
void TT<M>::add(int & x, int & y) {
  a += x;
  b += y;
}


int main(int argc, char * argv[]) {
 
  
  {
    
    TT<int> P = TT<int>();
    int x = 1.0, y = 2.0;
    P.add(x,y);
    
  }
  
  int val = 0;
  return val;
}


