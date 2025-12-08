/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////
// Print out the function information (mostly for debugging)
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void FunctionManager<EvalT>::printFunctions() {
  
  cout << endl;
  cout << "===========================================================" << endl;
  cout << "Printing functions on block: " << blockname_ << endl;
  cout << "-----------------------------------------------------------" << endl;
  
  for (size_t k=0; k<forests_.size(); k++) {
    
    cout << "Forest Name:" << forests_[k].location_ << endl;
    cout << "Number of Trees: " << forests_[k].trees_.size() << endl;
    for (size_t t=0; t<forests_[k].trees_.size(); t++) {
      cout << "    Tree: " << forests_[k].trees_[t].name_ << endl;
      cout << "    Number of branches: " << forests_[k].trees_[t].branches_.size() << endl;
      for (size_t b=0; b<forests_[k].trees_[t].branches_.size(); b++) {
        cout << "        " << forests_[k].trees_[t].branches_[b].expression_ << endl;
      }
    }
    
    cout << "-----------------------------------------------------------" << endl;
    
  }
  
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
bool FunctionManager<EvalT>::hasFunction(const std::string & fname, const std::string & location) {
  
  // Find the forest corresponding to the location
  for (size_t k=0; k<forests_.size(); k++) {
    if (forests_[k].location_ == location) {
      // Search for the function name in this forest's trees
      for (size_t t=0; t<forests_[k].trees_.size(); t++) {
        if (forests_[k].trees_[t].name_ == fname) {
          return true;
        }
      }
    }
  }
  return false;
}
