/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "functionManager.hpp"

using namespace MrHyDE;

template<class EvalT>
FunctionManager<EvalT>::FunctionManager() {
  // This really should NOT be constructed
  
  num_elem_ = 1;
  num_ip_ = 1;
  num_ip_side_ = 1;
  
  known_vars_ = {"x","y","z","t","nx","ny","nz","pi","h"};
  known_ops_ = {"sin","cos","exp","log","tan","abs","max","min","mean","emax","emin","emean","sqrt", "sinh", "cosh"};
  
  interpreter_ = Teuchos::rcp( new Interpreter<EvalT>());
  
}

template<class EvalT>
FunctionManager<EvalT>::FunctionManager(const string & blockname, const int & num_elem,
                                 const int & num_ip, const int & num_ip_side) :
num_elem_(num_elem), num_ip_(num_ip), num_ip_side_(num_ip_side), blockname_(blockname) {
  
  interpreter_ = Teuchos::rcp( new Interpreter<EvalT>());

  known_vars_ = {"x","y","z","t","nx","ny","nz","pi","h"};
  known_ops_ = {"sin","cos","exp","log","tan","abs","max","min","mean","emax","emin","emean","sqrt","sinh","cosh"};
  
  forests_.push_back(Forest<EvalT>("ip",num_elem_,num_ip_));
  forests_.push_back(Forest<EvalT>("side ip",num_elem_,num_ip_side_));
  forests_.push_back(Forest<EvalT>("point",1,1));
}

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
      
      int maxiter = 20; // maximum number of recursions
      
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
                else { // look for param(*) or param(**)
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
              interpreter_->split(forests_[fiter].trees_[titer].branches_,k);
              forests_[fiter].trees_[titer].branches_[k].is_decomposed_ = true;
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
  return is_scalar;
}


template<class EvalT>
void FunctionManager<EvalT>::checkDepDataType(const int & findex, const int & tindex, const int & bindex,
                                       bool & isConst, bool & isView, bool & isAD) {
  
  
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
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
Vista<EvalT> FunctionManager<EvalT>::evaluate(const string & fname, const string & location) {
  
  bool ffound = false, tfound = false;
  size_t fiter=0, titer=0;
  while(!ffound && fiter<forests_.size()) {
    if (forests_[fiter].location_ == location) {
      ffound = true;
      tfound = false;
      while (!tfound && titer<forests_[fiter].trees_.size()) {
        if (fname == forests_[fiter].trees_[titer].name_) {
          tfound = true;
          if (!forests_[fiter].trees_[titer].branches_[0].is_decomposed_) {
            this->decomposeFunctions();
          }
          if (!forests_[fiter].trees_[titer].branches_[0].is_constant_) {
            this->evaluate(fiter,titer,0);
          }
        }
        else {
          titer++;
        }
      }
    }
    else {
      fiter++;
    }
  }
  
  if (!ffound || !tfound) { // meaning that the requested function was not registered at this location
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: function manager could not evaluate: " + fname + " at " + location);
  }
  
  if (!forests_[fiter].trees_[titer].branches_[0].is_constant_) {
    forests_[fiter].trees_[titer].updateVista();
  }
  
  return forests_[fiter].trees_[titer].vista_;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void FunctionManager<EvalT>::evaluate( const size_t & findex, const size_t & tindex, const size_t & bindex) {
  
  //if (!forests_[findex].trees_[tindex].branches_[bindex].isConstant) {
    if (forests_[findex].trees_[tindex].branches_[bindex].is_leaf_) {
      if (forests_[findex].trees_[tindex].branches_[bindex].is_workset_data_) {
        int wdindex = forests_[findex].trees_[tindex].branches_[bindex].workset_data_index_;
        if (wkset->isOnSide) {
          if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
            if (!wkset->side_soln_fields[wdindex].is_updated_) {
              wkset->evaluateSideSolutionField(wdindex);
            }
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_ = wkset->side_soln_fields[wdindex].data_;
          }
          else {
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_ = wkset->side_scalar_fields[wdindex].data_;
          } 
        }
        else if (wkset->isOnPoint) {
          if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
            if (!wkset->point_soln_fields[wdindex].is_updated_) {
              wkset->evaluateSolutionField(wdindex);
            }
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_ = wkset->point_soln_fields[wdindex].data_;
          }
          else {
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_ = wkset->point_scalar_fields[wdindex].data_;
          }
        }
        else {
          if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
            if (!wkset->soln_fields[wdindex].is_updated_) {
              wkset->evaluateSolutionField(wdindex);
            }
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_ = wkset->soln_fields[wdindex].data_;
          }
          else {
            forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_ = wkset->scalar_fields[wdindex].data_;
          }
        }
      }
      else if (forests_[findex].trees_[tindex].branches_[bindex].is_parameter_) {
        // Should be set correctly already
      }
      else if (forests_[findex].trees_[tindex].branches_[bindex].is_time_) {
        forests_[findex].trees_[tindex].branches_[bindex].data_Sc_ = wkset->time;
      }
    }
    else if (forests_[findex].trees_[tindex].branches_[bindex].is_func_) {
      int funcIndex = forests_[findex].trees_[tindex].branches_[bindex].func_index_;
      this->evaluate(findex,funcIndex, 0);
      
      if (forests_[findex].trees_[tindex].branches_[bindex].is_AD_) {
        if (forests_[findex].trees_[tindex].branches_[bindex].is_view_) { // use viewdata
          forests_[findex].trees_[tindex].branches_[bindex].viewdata_ = forests_[findex].trees_[funcIndex].branches_[0].viewdata_;
        }
        else { // use data
          forests_[findex].trees_[tindex].branches_[bindex].data_ = forests_[findex].trees_[funcIndex].branches_[0].data_;
        }
      }
      else {
        if (forests_[findex].trees_[tindex].branches_[bindex].is_view_) { // use viewdata_Sc
          forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_ = forests_[findex].trees_[funcIndex].branches_[0].viewdata_Sc_;
        }
        else { // use data_Sc
          forests_[findex].trees_[tindex].branches_[bindex].data_Sc_ = forests_[findex].trees_[funcIndex].branches_[0].data_Sc_;
        }
      }
    }
    else {
      bool isAD = forests_[findex].trees_[tindex].branches_[bindex].is_AD_;
      bool isView = forests_[findex].trees_[tindex].branches_[bindex].is_view_;
      for (size_t k=0; k<forests_[findex].trees_[tindex].branches_[bindex].dep_list_.size(); k++) {
        
        int dep = forests_[findex].trees_[tindex].branches_[bindex].dep_list_[k];
        this->evaluate(findex, tindex, dep);
        
        bool termisAD = forests_[findex].trees_[tindex].branches_[dep].is_AD_;
        bool termisView = forests_[findex].trees_[tindex].branches_[dep].is_view_;
        bool termisParameter = forests_[findex].trees_[tindex].branches_[dep].is_parameter_;
        if (isView) {
          if (termisView) {
            if (isAD) {
              if (termisAD) {
                if (termisParameter) {
                  this->evaluateOpParamToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                           forests_[findex].trees_[tindex].branches_[dep].param_data_,
                                           forests_[findex].trees_[tindex].branches_[dep].param_index_,
                                           forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
                }
                else {
                  this->evaluateOpVToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                       forests_[findex].trees_[tindex].branches_[dep].viewdata_,
                                       forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
                }
                
              }
              else {
                this->evaluateOpVToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                     forests_[findex].trees_[tindex].branches_[dep].viewdata_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
            else {
              if (termisAD) {
                // output error
              }
              else {
                this->evaluateOpVToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_,
                                     forests_[findex].trees_[tindex].branches_[dep].viewdata_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
          }
          else { // Scalar data
            if (isAD) {
              if (termisAD) {
                this->evaluateOpSToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
              else {
                this->evaluateOpSToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
            else {
              if (termisAD) {
                //error
              }
              else {
                this->evaluateOpSToV(forests_[findex].trees_[tindex].branches_[bindex].viewdata_Sc_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
            
          }
        }
        else {
          if (termisView) {
            //error
          }
          else {
            if (isAD) {
              if (termisAD) {
                this->evaluateOpSToS(forests_[findex].trees_[tindex].branches_[bindex].data_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
              else {
                this->evaluateOpSToS(forests_[findex].trees_[tindex].branches_[bindex].data_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
            else {
              if (termisAD) {
                //error
              }
              else {
                this->evaluateOpSToS(forests_[findex].trees_[tindex].branches_[bindex].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[dep].data_Sc_,
                                     forests_[findex].trees_[tindex].branches_[bindex].dep_ops_[k]);
              }
            }
          }
        }
      }
    }
 // }
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT> 
template<class T1, class T2>
void FunctionManager<EvalT>::evaluateOpVToV(T1 data, T2 tdata, const string & op) {
  
  size_t dim0 = std::min(data.extent(0),tdata.extent(0));
  using namespace std;
  
  if (op == "") {
    parallel_for("funcman evaluate equals",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tdata(elem,pt);
      }
    });
  }
  else if (op == "plus") {
    parallel_for("funcman evaluate plus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += tdata(elem,pt);
      }
    });
  }
  else if (op == "minus") {
    parallel_for("funcman evaluate minus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += -tdata(elem,pt);
      }
    });
  }
  else if (op == "times") {
    parallel_for("funcman evaluate times",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) *= tdata(elem,pt);
      }
    });
  }
  else if (op == "divide") {
    parallel_for("funcman evaluate divide",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) /= tdata(elem,pt);
      }
    });
  }
  else if (op == "power") {
    parallel_for("funcman evaluate power",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = pow(data(elem,pt),tdata(elem,pt));
      }
    });
  }
  else if (op == "sin") {
    parallel_for("funcman evaluate sin",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sin(tdata(elem,pt));
      }
    });
  }
  else if (op == "cos") {
    parallel_for("funcman evaluate cos",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cos(tdata(elem,pt));
      }
    });
  }
  else if (op == "tan") {
    parallel_for("funcman evaluate tan",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tan(tdata(elem,pt));
      }
    });
  }
  else if (op == "exp") {
    parallel_for("funcman evaluate exp",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = exp(tdata(elem,pt));
      }
    });
  }
  else if (op == "log") {
    parallel_for("funcman evaluate log",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = log(tdata(elem,pt));
      }
    });
  }
  else if (op == "abs") {
    parallel_for("funcman evaluate abs",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (tdata(elem,pt) < 0.0) {
          data(elem,pt) = -tdata(elem,pt);
        }
        else {
          data(elem,pt) = tdata(elem,pt);
        }
      }
    });
  }
  else if (op == "max") {
    parallel_for("funcman evaluate max",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(e,n) > data(e,n)) {
          data(e,n) = tdata(e,n);
        }
      }
    });
  }
  else if (op == "min") {
    parallel_for("funcman evaluate min",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(e,n) < data(e,n)) {
          data(e,n) = tdata(e,n);
        }
      }
    });
  }
  else if (op == "mean") {
    parallel_for("funcman evaluate mean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) = 0.5*data(e,n) + 0.5*tdata(e,n);
      }
    });
  }
  else if (op == "emax") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for("funcman evaluate emax",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      data(e,0) = tdata(e,0);
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(e,n) > tdata(e,0)) {
          data(e,0) = tdata(e,n);
        }
      }
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emin") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for("funcman evaluate emin",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      data(e,0) = tdata(e,0);
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(e,n) < tdata(e,0)) {
          data(e,0) = tdata(e,n);
        }
      }
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emean") { // mean over rows ... usually corr. to mean over element/face
    parallel_for("funcman evaluate emean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      ScalarT scale = (ScalarT)dim1;
      data(e,0) = tdata(e,0)/scale;
      for (unsigned int n=0; n<dim1; n++) {
        data(e,0) += tdata(e,n)/scale;
      }
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "lt") {
    parallel_for("funcman evaluate lt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) < tdata(elem,pt)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "lte") {
    parallel_for("funcman evaluate lte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) <= tdata(elem,pt)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gt") {
    parallel_for("funcman evaluate gt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) > tdata(elem,pt)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gte") {
    parallel_for("funcman evaluate gte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) >= tdata(elem,pt)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "sqrt") {
    parallel_for("funcman evaluate sqrt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (tdata(elem,pt) <= 0.0) {
          data(elem,pt) = 0.0;
        }
        else {
          data(elem,pt) = sqrt(tdata(elem,pt));
        }
      }
    });
  }
  else if (op == "sinh") {
    parallel_for("funcman evaluate sinh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sinh(tdata(elem,pt));
      }
    });
  }
  else if (op == "cosh") {
    parallel_for("funcman evaluate cosh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cosh(tdata(elem,pt));
      }
    });
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
template<class T1, class T2>
void FunctionManager<EvalT>::evaluateOpParamToV(T1 data, T2 tdata, const int & pIndex_, const string & op) {
  
  size_t dim0 = data.extent(0);
  using namespace std;
  
  int pIndex = pIndex_;
  
  if (op == "") {
    parallel_for("funcman evaluate equals",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tdata(pIndex);
      }
    });
  }
  else if (op == "plus") {
    parallel_for("funcman evaluate plus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += tdata(pIndex);
      }
    });
  }
  else if (op == "minus") {
    parallel_for("funcman evaluate minus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += -tdata(pIndex);
      }
    });
  }
  else if (op == "times") {
    parallel_for("funcman evaluate times",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) *= tdata(pIndex);
      }
    });
  }
  else if (op == "divide") {
    parallel_for("funcman evaluate divide",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) /= tdata(pIndex);
      }
    });
  }
  else if (op == "power") {
    parallel_for("funcman evaluate power",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = pow(data(elem,pt),tdata(pIndex));
      }
    });
  }
  else if (op == "sin") {
    parallel_for("funcman evaluate sin",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sin(tdata(pIndex));
      }
    });
  }
  else if (op == "cos") {
    parallel_for("funcman evaluate cos",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cos(tdata(pIndex));
      }
    });
  }
  else if (op == "sinh") {
    parallel_for("funcman evaluate sinh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sinh(tdata(pIndex));
      }
    });
  }
  else if (op == "cosh") {
    parallel_for("funcman evaluate cosh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cosh(tdata(pIndex));
      }
    });
  }
  else if (op == "tan") {
    parallel_for("funcman evaluate tan",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tan(tdata(pIndex));
      }
    });
  }
  else if (op == "exp") {
    parallel_for("funcman evaluate exp",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = exp(tdata(pIndex));
      }
    });
  }
  else if (op == "log") {
    parallel_for("funcman evaluate log",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = log(tdata(pIndex));
      }
    });
  }
  else if (op == "sqrt") {
    parallel_for("funcman evaluate sqrt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if(tdata(pIndex) <= 0.) {
          data(elem,pt) = 0.;
        } else {
          data(elem,pt) = sqrt(tdata(pIndex));
        }
      }
    });
  }
  else if (op == "abs") {
    parallel_for("funcman evaluate abs",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (tdata(pIndex) < 0.0) {
          data(elem,pt) = -tdata(pIndex);
        }
        else {
          data(elem,pt) = tdata(pIndex);
        }
      }
    });
  }
  else if (op == "max") {
    parallel_for("funcman evaluate max",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(pIndex) > data(e,n)) {
          data(e,n) = tdata(pIndex);
        }
      }
    });
  }
  else if (op == "min") {
    parallel_for("funcman evaluate min",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        if (tdata(pIndex) < data(e,n)) {
          data(e,n) = tdata(pIndex);
        }
      }
    });
  }
  else if (op == "mean") {
    parallel_for("funcman evaluate mean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = 0.5*data(e,n) + 0.5*tdata(pIndex);
      }
    });
  }
  else if (op == "emax") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for("funcman evaluate emax",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata(pIndex);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emin") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for("funcman evaluate emin",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata(pIndex);
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emean") { // mean over rows ... usually corr. to mean over element/face
    parallel_for("funcman evaluate emean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata(pIndex);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "lt") {
    parallel_for("funcman evaluate lt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) < tdata(pIndex)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "lte") {
    parallel_for("funcman evaluate lte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) <= tdata(pIndex)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gt") {
    parallel_for("funcman evaluate gt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) > tdata(pIndex)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gte") {
    parallel_for("funcman evaluate gte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) >= tdata(pIndex)) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
template<class T1, class T2>
void FunctionManager<EvalT>::evaluateOpSToV(T1 data, T2 & tdata_, const string & op) {
  
  T2 tdata = tdata_; // Probably don't need to do this if pass by value
  size_t dim0 = data.extent(0);
  using namespace std;
  
  if (op == "") {
    parallel_for("funcman evaluate equals",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tdata;
      }
    });
  }
  else if (op == "plus") {
    parallel_for("funcman evaluate plus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += tdata;
      }
    });
  }
  else if (op == "minus") {
    parallel_for("funcman evaluate minus",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) += -tdata;
      }
    });
  }
  else if (op == "times") {
    parallel_for("funcman evaluate times",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) *= tdata;
      }
    });
  }
  else if (op == "divide") {
    parallel_for("funcman evaluate divide",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) /= tdata;
      }
    });
  }
  else if (op == "power") {
    parallel_for("funcman evaluate power",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = pow(data(elem,pt),tdata);
      }
    });
  }
  else if (op == "sin") {
    parallel_for("funcman evaluate sin",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sin(tdata);
      }
    });
  }
  else if (op == "cos") {
    parallel_for("funcman evaluate cos",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cos(tdata);
      }
    });
  }
  else if (op == "tan") {
    parallel_for("funcman evaluate tan",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tan(tdata);
      }
    });
  }
  else if (op == "sinh") {
    parallel_for("funcman evaluate sinh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = sinh(tdata);
      }
    });
  }
  else if (op == "cosh") {
    parallel_for("funcman evaluate cosh",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cosh(tdata);
      }
    });
  }
  else if (op == "exp") {
    parallel_for("funcman evaluate exp",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = exp(tdata);
      }
    });
  }
  else if (op == "log") {
    parallel_for("funcman evaluate log",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = log(tdata);
      }
    });
  }
  else if (op == "sqrt") {
    parallel_for("funcman evaluate sqrt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if(tdata < 0) {
          data(elem,pt) = 0.0;
        } else {
          data(elem,pt) = sqrt(tdata);
        }
      }
    });
  }
  else if (op == "abs") {
    parallel_for("funcman evaluate abs",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (tdata < 0.0) {
          data(elem,pt) = -tdata;
        }
        else {
          data(elem,pt) = tdata;
        }
      }
    });
  }
  else if (op == "max") {
    parallel_for("funcman evaluate max",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata > data(e,n)) {
          data(e,n) = tdata;
        }
      }
    });
  }
  else if (op == "min") {
    parallel_for("funcman evaluate min",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        if (tdata < data(e,n)) {
          data(e,n) = tdata;
        }}
    });
  }
  else if (op == "mean") {
    parallel_for("funcman evaluate mean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = 0.5*data(e,n) + 0.5*tdata;
      }
    });
  }
  else if (op == "emax") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for("funcman evaluate emax",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata;
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emin") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for("funcman evaluate emin",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata;
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "emean") { // mean over rows ... usually corr. to mean over element/face
    parallel_for("funcman evaluate emean",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata;
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "lt") {
    parallel_for("funcman evaluate lt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) < tdata) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "lte") {
    parallel_for("funcman evaluate lte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) <= tdata) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gt") {
    parallel_for("funcman evaluate gt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) > tdata) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  else if (op == "gte") {
    parallel_for("funcman evaluate gte",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VECTORSIZE),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        if (data(elem,pt) >= tdata) {
          data(elem,pt) = 1.0;
        }
        else {
          data(elem,pt) = 0.0;
        }
      }
    });
  }
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
template<class T1, class T2>
void FunctionManager<EvalT>::evaluateOpSToS(T1 & data, T2 & tdata, const string & op) {
  
  using namespace std;
  
  if (op == "") {
    data = tdata;
  }
  else if (op == "plus") {
    data += tdata;
  }
  else if (op == "minus") {
    data += -tdata;
  }
  else if (op == "times") {
    data *= tdata;
  }
  else if (op == "divide") {
    data /= tdata;
  }
  else if (op == "power") {
    data = pow(data,tdata);
  }
  else if (op == "sin") {
    data = sin(tdata);
  }
  else if (op == "cos") {
    data = cos(tdata);
  }
  else if (op == "tan") {
    data = tan(tdata);
  }
  else if (op == "sinh") {
    data = sinh(tdata);
  }
  else if (op == "cosh") {
    data = cosh(tdata);
  }
  else if (op == "exp") {
    data = exp(tdata);
  }
  else if (op == "log") {
    data = log(tdata);
  }
  else if (op == "sqrt") {
    if (tdata <= 0.0) {
      data = 0.0;
    }
    else {
      data = sqrt(tdata);
    }
  }
  else if (op == "abs") {
    if (tdata < 0.0) {
      data = -tdata;
    }
    else {
      data = tdata;
    }
  }
  else if (op == "max") {
    data = max(data,tdata);
  }
  else if (op == "min") {
    data = min(data,tdata);
  }
  else if (op == "mean") {
    data = 0.5*data+0.5*tdata;
  }
  else if (op == "emax") { // maximum over rows ... usually corr. to max over element/face at ip
    data = tdata;
  }
  else if (op == "emin") { // minimum over rows ... usually corr. to min over element/face at ip
    data = tdata;
  }
  else if (op == "emean") { // mean over rows ... usually corr. to mean over element/face
    data = tdata;
  }
  else if (op == "lt") {
    if (data < tdata) {
      data = 1.0;
    }
    else {
      data = 0.0;
    }
  }
  else if (op == "lte") {
    if (data <= tdata) {
      data = 1.0;
    }
    else {
      data = 0.0;
    }
  }
  else if (op == "gt") {
    if (data > tdata) {
      data = 1.0;
    }
    else {
      data = 0.0;
    }
  }
  else if (op == "gte") { 
    if (data >= tdata) {
      data = 1.0;
    }
    else {
      data = 0.0;
    }
  }
  
}


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


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::FunctionManager<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::FunctionManager<AD>;

// Standard built-in types
template class MrHyDE::FunctionManager<AD2>;
template class MrHyDE::FunctionManager<AD4>;
template class MrHyDE::FunctionManager<AD8>;
template class MrHyDE::FunctionManager<AD16>;
template class MrHyDE::FunctionManager<AD18>;
template class MrHyDE::FunctionManager<AD24>;
template class MrHyDE::FunctionManager<AD32>;
#endif