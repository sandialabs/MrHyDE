/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

template<class EvalT>
FunctionManager<EvalT>::FunctionManager() {
  // This really should NOT be constructed
  
  num_elem_ = 1;
  num_ip_ = 1;
  num_ip_side_ = 1;
  
  known_vars_ = {"x","y","z","t","pi","h"};
  known_ops_ = {"sin","cos","exp","log","tan","abs","max","min","mean","emax","emin","emean","sqrt", "sinh", "cosh"};
  
  interpreter_ = Teuchos::rcp( new Interpreter<EvalT>());
  
}

template<class EvalT>
FunctionManager<EvalT>::FunctionManager(const string & blockname, const int & num_elem,
                                        const int & num_ip, const int & num_ip_side) :
num_elem_(num_elem), num_ip_(num_ip), num_ip_side_(num_ip_side), blockname_(blockname) {
  
  interpreter_ = Teuchos::rcp( new Interpreter<EvalT>());

  known_vars_ = {"x","y","z","t","pi","h"};
  known_ops_ = {"sin","cos","exp","log","tan","abs","max","min","mean","emax","emin","emean","sqrt","sinh","cosh"};
  
  forests_.push_back(Forest<EvalT>("ip",num_elem_,num_ip_));
  forests_.push_back(Forest<EvalT>("side ip",num_elem_,num_ip_side_));
  forests_.push_back(Forest<EvalT>("point",1,1));
}
