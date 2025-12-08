/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////
// Add a user defined function
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
int FunctionManager<EvalT>::addFunction(const string & fname, const string & expression, const string & location) {
  bool found = false;
  int findex = 0;
  
  for (size_t k=0; k<forests_.size(); k++) {
    if (forests_[k].location_ == location) {
      for (size_t j=0; j<forests_[k].trees_.size(); ++j) {
        if (forests_[k].trees_[j].name_ == fname) {
          found = true;
          findex = j;
        }
      }
      if (!found) {
        forests_[k].addTree(fname, expression);
        findex = forests_[k].trees_.size()-1;
      }
    }
  }
  return findex;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Add a user defined function
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
int FunctionManager<EvalT>::addFunction(const string & fname, ScalarT & value, const string & location) {
  bool found = false;
  int findex = 0;
  
  for (size_t k=0; k<forests_.size(); k++) {
    if (forests_[k].location_ == location) {
      for (size_t j=0; j<forests_[k].trees_.size(); ++j) {
        if (forests_[k].trees_[j].name_ == fname) {
          found = true;
          findex = j;
        }
      }
      if (!found) {
        forests_[k].addTree(fname, value);
        findex = forests_[k].trees_.size()-1;
      }
    }
  }
  return findex;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Set the list of parameters
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void FunctionManager<EvalT>::setupLists(const vector<string> & parameters) {
  parameters_ = parameters;
}

//////////////////////////////////////////////////////////////////////////////////////
// Decompose the functions into terms and set the evaluation tree
// Also sets up the Kokkos::Views (subviews) to the data for all of the terms
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void FunctionManager<EvalT>::decomposeFunctions() {
  
  if (wkset->isInitialized) {
    
    for (size_t fiter=0; fiter<forests_.size(); fiter++) {
      
      int maxiter = 50; // maximum number of recursions
      
      for (size_t titer=0; titer<forests_[fiter].trees_.size(); titer++) {
        
        bool done = false; // will turn to "true" when the tree is fully decomposed
        int iter = 0;
        
        while (!done && iter < maxiter) {

          iter++;
          size_t Nbranches = forests_[fiter].trees_[titer].branches_.size();
          
          for (size_t k=0; k<Nbranches; k++) {
            
            // HAVE WE ALREADY LOOKED AT THIS TERM?
            bool decompose = true;
            if (forests_[fiter].trees_[titer].branches_[k].is_leaf_ || forests_[fiter].trees_[titer].branches_[k].is_decomposed_) {
              decompose = false;
            }
            
            string expr = forests_[fiter].trees_[titer].branches_[k].expression_;
            
            // Is it an AD data stored in the workset?
            if (decompose) {
              
              if (forests_[fiter].location_ == "side ip") {
                bool found = 0;
                size_t j=0;
                while (!found && j<wkset->side_soln_fields.size()) {
                  if (expr == wkset->side_soln_fields[j].expression_) {
                    decompose = false;
                    forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_AD_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_workset_data_ = true;
                  
                    forests_[fiter].trees_[titer].branches_[k].workset_data_index_ = j;
                    wkset->isOnSide = true;
                    wkset->checkSolutionFieldAllocation(j);
                    wkset->isOnSide = false;
                    found = true;
                  }
                  j++;
                }
              }
              else if (forests_[fiter].location_ == "point") {
                bool found = 0;
                size_t j=0;
                while (!found && j<wkset->point_soln_fields.size()) {
                  if (expr == wkset->point_soln_fields[j].expression_) {
                    decompose = false;
                    forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_AD_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_workset_data_ = true;
                  
                    forests_[fiter].trees_[titer].branches_[k].workset_data_index_ = j;
                    
                    wkset->isOnPoint = true;
                    wkset->checkSolutionFieldAllocation(j);
                    wkset->isOnPoint = false;
                    found = true;
                  }
                  j++;
                }
              }
              else {
                bool found = 0;
                size_t j=0;
                while (!found && j<wkset->soln_fields.size()) {
                  if (expr == wkset->soln_fields[j].expression_) {
                    decompose = false;
                    forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_AD_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_workset_data_ = true;
                  
                    forests_[fiter].trees_[titer].branches_[k].workset_data_index_ = j;
                    wkset->checkSolutionFieldAllocation(j);
                    found = true;
                  }
                  j++;
                }
              }
            }
            
            // Is it a Scalar data stored in the workset?
            if (decompose) {
              
              if (forests_[fiter].location_ == "side ip") {
                bool found = 0;
                size_t j=0;
                while (!found && j<wkset->side_scalar_fields.size()) {
                  if (expr == wkset->side_scalar_fields[j].expression_) {
                    decompose = false;
                    forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_AD_ = false;
                    forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_workset_data_ = true;
                    forests_[fiter].trees_[titer].branches_[k].workset_data_index_ = j;
                    wkset->isOnSide = true;
                    wkset->checkScalarFieldAllocation(j);
                    wkset->isOnSide = false;
                    found = true;
                  }
                  j++;
                }
              }
              else if (forests_[fiter].location_ == "point") {
                bool found = 0;
                size_t j=0;
                while (!found && j<wkset->point_scalar_fields.size()) {
                  if (expr == wkset->point_scalar_fields[j].expression_) {
                    decompose = false;
                    forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_AD_ = false;
                    forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_workset_data_ = true;
                    forests_[fiter].trees_[titer].branches_[k].workset_data_index_ = j;
                    wkset->isOnPoint = true;
                    wkset->checkScalarFieldAllocation(j);
                    wkset->isOnPoint = false;
                    found = true;
                  }
                  j++;
                }
              }
              else {
                bool found = 0;
                size_t j=0;
                while (!found && j<wkset->scalar_fields.size()) {
                  if (expr == wkset->scalar_fields[j].expression_) {
                    decompose = false;
                    forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_AD_ = false;
                    forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_workset_data_ = true;
                    forests_[fiter].trees_[titer].branches_[k].workset_data_index_ = j;
                    wkset->checkScalarFieldAllocation(j);
                    found = true;
                  }
                  j++;
                }
              }
            }
            
            // check if it is a parameter
            if (decompose) {
              
              for (unsigned int j=0; j<parameters_.size(); j++) {
                
                if (expr == parameters_[j]) {
                  forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_AD_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_parameter_ = true;
                  forests_[fiter].trees_[titer].branches_[k].param_index_ = 0;
                  
                  decompose = false;
                  
                  forests_[fiter].trees_[titer].branches_[k].param_data_ = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
                  
                }
                else { // look for param(*) or param(**) this does mean that over 100 scalar parameters is not supported
                  bool found = true;
                  int sindex = 0;
                  size_t nexp = expr.length();
                  if (nexp == parameters_[j].length()+3) {
                    for (size_t n=0; n<parameters_[j].length(); n++) {
                      if (expr[n] != parameters_[j][n]) {
                        found = false;
                      }
                    }
                    if (found) {
                      if (expr[nexp-3] == '(' && expr[nexp-1] == ')') {
                        string check = "";
                        check += expr[nexp-2];
                        if (isdigit(check[0])) {
                          sindex = std::stoi(check);
                        }
                        else {
                          found = false;
                        }
                      }
                      else {
                        found = false;
                      }
                    }
                  }
                  else if (nexp == parameters_[j].length()+4) {
                    for (size_t n=0; n<parameters_[j].length(); n++) {
                      if (expr[n] != parameters_[j][n]) {
                        found = false;
                      }
                    }
                    if (found) {
                      if (expr[nexp-4] == '(' && expr[nexp-1] == ')') {
                        string check = "";
                        check += expr[nexp-3];
                        check += expr[nexp-2];
                        if (isdigit(check[0]) && isdigit(check[1])) {
                          sindex = std::stoi(check);
                        }
                        else {
                          found = false;
                        }
                      }
                      else {
                        found = false;
                      }
                    }
                  }
                  else {
                    found = false;
                  }
                  
                  if (found) {
                    forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_AD_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_parameter_ = true;
                    
                    forests_[fiter].trees_[titer].branches_[k].param_index_ = sindex;
                    
                    decompose = false;
                    
                    forests_[fiter].trees_[titer].branches_[k].param_data_ = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
                  }
                }
              }
            }
            
            // check if it is the time-derivative of a parameter
            if (decompose) {
              
              for (unsigned int j=0; j<parameters_.size(); j++) {
                
                if (expr == parameters_[j]+"_t") {
                  forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_AD_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_parameter_ = true;
                  forests_[fiter].trees_[titer].branches_[k].param_index_ = 0;
                  
                  decompose = false;
                  
                  forests_[fiter].trees_[titer].branches_[k].param_data_ = Kokkos::subview(wkset->params_dot_AD, j, Kokkos::ALL());
                  
                }
                else { // look for param_t(*) or param_t(**) this does mean that over 100 scalar parameters is not supported
                  bool found = true;
                  int sindex = 0;
                  size_t nexp = expr.length();
                  if (nexp == parameters_[j].length()+5) {
                    for (size_t n=0; n<parameters_[j].length(); n++) {
                      if (expr[n] != parameters_[j][n]) {
                        found = false;
                      }
                    }
                    if (found) {
                      if (expr[nexp-5] == '_' && expr[nexp-4] == 't' && expr[nexp-3] == '(' && expr[nexp-1] == ')') {
                        string check = "";
                        check += expr[nexp-2];
                        if (isdigit(check[0])) {
                          sindex = std::stoi(check);
                        }
                        else {
                          found = false;
                        }
                      }
                      else {
                        found = false;
                      }
                    }
                  }
                  else if (nexp == parameters_[j].length()+6) {
                    for (size_t n=0; n<parameters_[j].length(); n++) {
                      if (expr[n] != parameters_[j][n]) {
                        found = false;
                      }
                    }
                    if (found) {
                      if (expr[nexp-6] == '_' && expr[nexp-5] == 't' && expr[nexp-4] == '(' && expr[nexp-1] == ')') {
                        string check = "";
                        check += expr[nexp-3];
                        check += expr[nexp-2];
                        if (isdigit(check[0]) && isdigit(check[1])) {
                          sindex = std::stoi(check);
                        }
                        else {
                          found = false;
                        }
                      }
                      else {
                        found = false;
                      }
                    }
                  }
                  else {
                    found = false;
                  }
                  
                  if (found) {
                    forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_view_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_AD_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                    forests_[fiter].trees_[titer].branches_[k].is_parameter_ = true;
                    
                    forests_[fiter].trees_[titer].branches_[k].param_index_ = sindex;
                    
                    decompose = false;
                    
                    forests_[fiter].trees_[titer].branches_[k].param_data_ = Kokkos::subview(wkset->params_dot_AD, j, Kokkos::ALL());
                  }
                }
              }
            }
            
            // check if it is a function
            if (decompose) {
              for (unsigned int j=0; j<forests_[fiter].trees_.size(); j++) {
                if (expr == forests_[fiter].trees_[j].name_) {
                  forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_func_ = true;
                  
                  forests_[fiter].trees_[titer].branches_[k].func_index_ = j;
                  decompose = false;
                }
              }
            }
            
            // IS THE TERM A SIMPLE SCALAR: 2.03, 1.0E2, etc.
            if (decompose) {
              bool isnum = interpreter_->isScalar(expr);
              if (isnum) {
                forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                forests_[fiter].trees_[titer].branches_[k].is_constant_ = true;
                
                ScalarT val = std::stod(expr);
                forests_[fiter].trees_[titer].branches_[k].data_Sc_ = val;
                
                decompose = false;
              }
            }
            
            // IS THE TERM ONE OF THE KNOWN VARIABLES: t or pi
            if (decompose) {
              for (size_t j=0; j<known_vars_.size(); j++) {
                if (expr == known_vars_[j]) {
                  decompose = false;
                  forests_[fiter].trees_[titer].branches_[k].is_leaf_ = true;
                  forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
                  
                  if (known_vars_[j] == "t") {
                    forests_[fiter].trees_[titer].branches_[k].is_time_ = true;
                    forests_[fiter].trees_[titer].branches_[k].data_Sc_ = wkset->time;
                  }
                  else if (known_vars_[j] == "pi") {
                    forests_[fiter].trees_[titer].branches_[k].is_constant_ = true; // means in does not need to be copied every time
                    forests_[fiter].trees_[titer].branches_[k].data_Sc_ = PI;
                  }
                }
              }
            } // end known_vars_
            
            // IS THIS TERM ONE OF THE KNOWN OPERATORS: sin(...), exp(...), etc.
            if (decompose) {
              bool isop = interpreter_->isOperator(forests_[fiter].trees_[titer].branches_, k, known_ops_);
              if (isop) {
                decompose = false;
              }
            }
            
            if (decompose) {
              size_t cnumb = forests_[fiter].trees_[titer].branches_.size();
              interpreter_->split(forests_[fiter].trees_[titer].branches_,k);
              forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
              if (cnumb == forests_[fiter].trees_[titer].branches_.size()) {
                // This means that it didn't actually add anything
                // Most likely due to leaf being undefined
                // There are case where this is ok:
                string expr = forests_[fiter].trees_[titer].branches_[k].expression_;
                if (expr == "n[x]" || expr == "n[y]" || expr == "n[z]" || expr == "t[x]" || expr == "t[y]" || expr == "t[z]") {
                  // this is fine
                }
                else {
                  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE was not able to decompose or find: " + forests_[fiter].trees_[titer].branches_[k].expression_);
                }
              }
            }
          }
          
          bool isdone = true;
          for (size_t k=0; k<forests_[fiter].trees_[titer].branches_.size(); k++) {
            if (!forests_[fiter].trees_[titer].branches_[k].is_leaf_ && !forests_[fiter].trees_[titer].branches_[k].is_decomposed_) {
              isdone = false;
            }
          }
          done = isdone;
          
        }
        
        if (!done && iter >= maxiter) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE was not able to decompose " + forests_[fiter].trees_[titer].name_);
        }
      } // trees_
    } // forests_
    
    // After all of the forests_/trees_ have been decomposed, we can determine if we need to use arrays of ScalarT or AD
    // Only the leafs should be designated as ScalarT or AD at this point
    
    for (size_t f=0; f<forests_.size(); ++f) {
      for (size_t k=0; k<forests_[f].trees_.size(); k++) {
        for (size_t j=0; j<forests_[f].trees_[k].branches_.size(); j++) {
          
          // Rewrite this section
          
          bool isConst = true, isView = false, isAD = false;
          
          this->checkDepDataType(f,k,j, isConst, isView, isAD); // is this term a ScalarT
          
          forests_[f].trees_[k].branches_[j].is_constant_ = isConst;
          forests_[f].trees_[k].branches_[j].is_view_ = isView;
          forests_[f].trees_[k].branches_[j].is_AD_ = isAD;
          
          if (isView) {
            string expr = forests_[f].trees_[k].branches_[j].expression_;
            if (isAD) {
              forests_[f].trees_[k].branches_[j].viewdata_ = View_EvalT("data for " + expr,
                                                                      forests_[f].dim0_, forests_[f].dim1_);
            }
            else {
              forests_[f].trees_[k].branches_[j].viewdata_Sc_ = View_Sc2("data for " + expr,
                                                                         forests_[f].dim0_, forests_[f].dim1_);
            }
          }
        }
      }
    }
    
    // Now evaluate all of the constant branches (meaning all deps are const, !vector, !AD)
    for (size_t f=0; f<forests_.size(); ++f) {
      for (size_t k=0; k<forests_[f].trees_.size(); k++) {
        for (size_t j=0; j<forests_[f].trees_[k].branches_.size(); j++) {
          if (forests_[f].trees_[k].branches_[j].is_constant_) {
            if (!forests_[f].trees_[k].branches_[j].is_leaf_) { // leafs are already filled
              this->evaluate(f,k,j);
            }
          }
        }
        forests_[f].trees_[k].setupVista();
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Determine if a branch is a ScalarT or needs to be an AD type
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
bool FunctionManager<EvalT>::isScalarTerm(const int & findex, const int & tindex, const int & bindex) {
  if (forests_[findex].trees_[tindex].branches_[bindex].currently_checking_) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE detected a cyclic graph in: " + forests_[findex].trees_[tindex].branches_[bindex].expression_);
  }
  else {
    forests_[findex].trees_[tindex].branches_[bindex].currently_checking_ = true;
  }
  
  bool is_scalar = true;
  if (forests_[findex].trees_[tindex].branches_[bindex].is_leaf_) {
    if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
      is_scalar = false;
    }
  }
  else if (forests_[findex].trees_[tindex].branches_[bindex].is_func_) {
    is_scalar = false;
  }
  else {
    for (size_t k=0; k<forests_[findex].trees_[tindex].branches_[bindex].dep_list_.size(); k++){
      bool depcheck = isScalarTerm(findex, tindex, forests_[findex].trees_[tindex].branches_[bindex].dep_list_[k]);
      if (!depcheck) {
        is_scalar = false;
      }
    }
  }
  forests_[findex].trees_[tindex].branches_[bindex].currently_checking_ = false;
  return is_scalar;
}


template<class EvalT>
void FunctionManager<EvalT>::checkDepDataType(const int & findex, const int & tindex, const int & bindex,
                                              bool & isConst, bool & isView, bool & isAD) {
  
  if (forests_[findex].trees_[tindex].branches_[bindex].currently_checking_) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE detected a cyclic graph in: " + forests_[findex].trees_[tindex].branches_[bindex].expression_);
  }
  else {
    forests_[findex].trees_[tindex].branches_[bindex].currently_checking_ = true;
  }
  if (forests_[findex].trees_[tindex].branches_[bindex].is_leaf_) {
    if (!forests_[findex].trees_[tindex].branches_[bindex].is_constant_) {
      isConst = false;
    }
    if (forests_[findex].trees_[tindex].branches_[bindex].is_view_) {
      isView = true;
    }
    if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
      isAD = true;
    }
  }
  else if (forests_[findex].trees_[tindex].branches_[bindex].is_func_) {
    this->checkDepDataType(findex, forests_[findex].trees_[tindex].branches_[bindex].func_index_, 0,
                           isConst, isView, isAD);
  }
  else {
    for (size_t k=0; k<forests_[findex].trees_[tindex].branches_[bindex].dep_list_.size(); k++){
      this->checkDepDataType(findex, tindex, forests_[findex].trees_[tindex].branches_[bindex].dep_list_[k],
                             isConst, isView, isAD);
    }
  }
  forests_[findex].trees_[tindex].branches_[bindex].currently_checking_ = false;
}
