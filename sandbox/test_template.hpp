/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

template<class M>
class TT {
public:
  TT();
  //{
  //  a = 0.0;
  //  b = 0.0;
  //}
  void add(int & x, int & y);
  //{
  //  a += x;
  //  b+= y;
  //}
  
  M a,b;
};
