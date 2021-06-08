/***********************************************************************
 This is a framework for solving Multi-resolution Hybridized
 Differential Equations (MrHyDE), an optimized version of
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "functionManager.hpp"

using namespace MrHyDE;

FunctionManager::FunctionManager() {
  // This really should NOT be constructed
  
  numElem = 1;
  numip = 1;
  numip_side = 1;
  
  known_vars = {"x","y","z","t","nx","ny","nz","pi","h"};
  known_ops = {"sin","cos","exp","log","tan","abs","max","min","mean","sqrt"};
  
  interpreter = Teuchos::rcp( new Interpreter());
  
}


FunctionManager::FunctionManager(const string & blockname_, const int & numElem_,
                                 const int & numip_, const int & numip_side_) :
blockname(blockname_), numElem(numElem_), numip(numip_), numip_side(numip_side_) {
  
  known_vars = {"x","y","z","t","nx","ny","nz","pi","h"};
  known_ops = {"sin","cos","exp","log","tan","abs","max","min","mean","sqrt"};
  
  interpreter = Teuchos::rcp( new Interpreter());
  
  forests.push_back(Forest("ip",numElem,numip));
  forests.push_back(Forest("side ip",numElem,numip_side));
  forests.push_back(Forest("point",1,1));
}

//////////////////////////////////////////////////////////////////////////////////////
// Add a user defined function
//////////////////////////////////////////////////////////////////////////////////////

int FunctionManager::addFunction(const string & fname, string & expression, const string & location) {
  bool found = false;
  int findex = 0;
  
  for (size_t k=0; k<forests.size(); k++) {
    if (forests[k].location == location) {
      for (size_t j=0; j<forests[k].trees.size(); ++j) {
        if (forests[k].trees[j].name == fname) {
          found = true;
          findex = j;
        }
      }
      if (!found) {
        forests[k].addTree(fname, expression);
        findex = forests[k].trees.size()-1;
      }
    }
  }
  return findex;
  
  /*
   for (size_t k=0; k<functions.size(); k++) {
   if (functions[k].function_name == fname && functions[k].location == location) {
   found = true;
   findex = k;
   }
   }
   if (!found) {
   int dim1 = 0;
   if (location == "ip") {
   dim1 = numip;
   }
   else if (location == "side ip") {
   dim1 = numip_side;
   }
   else if (location == "point") {
   dim1 = 1;
   }
   functions.push_back(function_class(fname, expression, numElem, dim1, location));
   findex = functions.size()-1;
   }
   return findex;
   */
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Set the lists of variables, parameters and discretized parameters
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::setupLists(const vector<string> & variables_,
                                 const vector<string> & aux_variables_,
                                 const vector<string> & parameters_,
                                 const vector<string> & disc_parameters_) {
  variables = variables_;
  aux_variables = aux_variables_;
  parameters = parameters_;
  disc_parameters = disc_parameters_;
}

//////////////////////////////////////////////////////////////////////////////////////
// Decompose the functions into terms and set the evaluation tree
// Also sets up the Kokkos::Views (subviews) to the data for all of the terms
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::decomposeFunctions() {
  
  Teuchos::TimeMonitor ttimer(*decomposeTimer);
  
  if (wkset->isInitialized) {
    
    for (size_t fiter=0; fiter<forests.size(); fiter++) {
      
      int maxiter = 20; // maximum number of recursions
      
      for (size_t titer=0; titer<forests[fiter].trees.size(); titer++) {
        
        bool done = false; // will turn to "true" when the tree is fully decomposed
        int iter = 0;
        
        while (!done && iter < maxiter) {
          
          iter++;
          size_t Nbranches = forests[fiter].trees[titer].branches.size();
          
          for (size_t k=0; k<Nbranches; k++) {
            
            // HAVE WE ALREADY LOOKED AT THIS TERM?
            bool decompose = true;
            if (forests[fiter].trees[titer].branches[k].isLeaf || forests[fiter].trees[titer].branches[k].isDecomposed) {
              decompose = false;
            }
            
            string expr = forests[fiter].trees[titer].branches[k].expression;
            
            // Is it an AD data stored in the workset?
            if (decompose) {
              vector<string> data_labels = wkset->data_labels;
              
              string mod_expr = expr;
              if (forests[fiter].location == "side ip") {
                mod_expr += " side";
              }
              else if (forests[fiter].location == "point") {
                mod_expr += " point";
              }
              bool found = 0;
              size_t j=0;
              while (!found && j<data_labels.size()) {
                if (mod_expr == data_labels[j]) {
                  decompose = false;
                  forests[fiter].trees[titer].branches[k].isLeaf = true;
                  forests[fiter].trees[titer].branches[k].isDecomposed = true;
                  forests[fiter].trees[titer].branches[k].isView = true;
                  forests[fiter].trees[titer].branches[k].isAD = true;
                  forests[fiter].trees[titer].branches[k].isWorksetData = true;
                  
                  forests[fiter].trees[titer].branches[k].workset_data_index = j;
                  wkset->checkDataAllocation(j);
                  found = true;
                }
                j++;
              }
            }
            
            // Is it a Scalar data stored in the workset?
            if (decompose) {
              vector<string> data_Sc_labels = wkset->data_Sc_labels;
              string mod_expr = expr;
              if (forests[fiter].location == "side ip") {
                mod_expr += " side";
              }
              else if (forests[fiter].location == "point") {
                mod_expr += " point";
              }
              bool found = 0;
              size_t j=0;
              while (!found && j<data_Sc_labels.size()) {
                if (mod_expr == data_Sc_labels[j]) {
                  decompose = false;
                  forests[fiter].trees[titer].branches[k].isLeaf = true;
                  forests[fiter].trees[titer].branches[k].isDecomposed = true;
                  forests[fiter].trees[titer].branches[k].isView = true;
                  forests[fiter].trees[titer].branches[k].isWorksetData = true;
                  
                  forests[fiter].trees[titer].branches[k].workset_data_index = j;
                  wkset->checkDataScAllocation(j);
                  found = true;
                }
                j++;
              }
            }
            
            // check if it is a parameter
            if (decompose) {
              
              for (unsigned int j=0; j<parameters.size(); j++) {
                
                if (expr == parameters[j]) {
                  forests[fiter].trees[titer].branches[k].isLeaf = true;
                  forests[fiter].trees[titer].branches[k].isView = true;
                  forests[fiter].trees[titer].branches[k].isAD = true;
                  forests[fiter].trees[titer].branches[k].isDecomposed = true;
                  forests[fiter].trees[titer].branches[k].isParameter = true;
                  forests[fiter].trees[titer].branches[k].paramIndex = 0;
                  
                  decompose = false;
                  
                  forests[fiter].trees[titer].branches[k].param_data = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
                  
                }
                else { // look for param(*) or param(**)
                  bool found = true;
                  int sindex = 0;
                  size_t nexp = expr.length();
                  if (nexp == parameters[j].length()+3) {
                    for (size_t n=0; n<parameters[j].length(); n++) {
                      if (expr[n] != parameters[j][n]) {
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
                  else if (nexp == parameters[j].length()+4) {
                    for (size_t n=0; n<parameters[j].length(); n++) {
                      if (expr[n] != parameters[j][n]) {
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
                    forests[fiter].trees[titer].branches[k].isLeaf = true;
                    forests[fiter].trees[titer].branches[k].isView = true;
                    forests[fiter].trees[titer].branches[k].isAD = true;
                    forests[fiter].trees[titer].branches[k].isDecomposed = true;
                    forests[fiter].trees[titer].branches[k].isParameter = true;
                    
                    forests[fiter].trees[titer].branches[k].paramIndex = sindex;
                    
                    decompose = false;
                    
                    forests[fiter].trees[titer].branches[k].param_data = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
                  }
                }
              }
            }
            
            // check if it is a function
            if (decompose) {
              for (unsigned int j=0; j<forests[fiter].trees.size(); j++) {
                if (expr == forests[fiter].trees[j].name) {
                  forests[fiter].trees[titer].branches[k].isDecomposed = true;
                  forests[fiter].trees[titer].branches[k].isFunc = true;
                  
                  forests[fiter].trees[titer].branches[k].funcIndex = j;
                  decompose = false;
                }
              }
            }
            
            // IS THE TERM A SIMPLE SCALAR: 2.03, 1.0E2, etc.
            if (decompose) {
              bool isnum = interpreter->isScalar(expr);
              if (isnum) {
                forests[fiter].trees[titer].branches[k].isLeaf = true;
                forests[fiter].trees[titer].branches[k].isDecomposed = true;
                forests[fiter].trees[titer].branches[k].isConstant = true;
                
                ScalarT val = std::stod(expr);
                forests[fiter].trees[titer].branches[k].data_Sc = val;
                
                decompose = false;
              }
            }
            
            // IS THE TERM ONE OF THE KNOWN VARIABLES: t or pi
            if (decompose) {
              for (size_t j=0; j<known_vars.size(); j++) {
                if (expr == known_vars[j]) {
                  decompose = false;
                  forests[fiter].trees[titer].branches[k].isLeaf = true;
                  forests[fiter].trees[titer].branches[k].isDecomposed = true;
                  
                  if (known_vars[j] == "t") {
                    forests[fiter].trees[titer].branches[k].isTime = true;
                    forests[fiter].trees[titer].branches[k].data_Sc = wkset->time;
                  }
                  else if (known_vars[j] == "pi") {
                    forests[fiter].trees[titer].branches[k].isConstant = true; // means in does not need to be copied every time
                    forests[fiter].trees[titer].branches[k].data_Sc = PI;
                  }
                }
              }
            } // end known_vars
            
            // IS THIS TERM ONE OF THE KNOWN OPERATORS: sin(...), exp(...), etc.
            if (decompose) {
              bool isop = interpreter->isOperator(forests[fiter].trees[titer].branches, k, known_ops);
              if (isop) {
                decompose = false;
              }
            }
            
            if (decompose) {
              interpreter->split(forests[fiter].trees[titer].branches,k);
              forests[fiter].trees[titer].branches[k].isDecomposed = true;
            }
          }
          
          bool isdone = true;
          for (size_t k=0; k<forests[fiter].trees[titer].branches.size(); k++) {
            if (!forests[fiter].trees[titer].branches[k].isLeaf && !forests[fiter].trees[titer].branches[k].isDecomposed) {
              isdone = false;
            }
          }
          done = isdone;
          
        }
        
        if (!done && iter >= maxiter) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE was not able to decompose " + forests[fiter].trees[titer].name);
        }
      } // trees
    } // forests
    
    // After all of the forests/trees have been decomposed, we can determine if we need to use arrays of ScalarT or AD
    // Only the leafs should be designated as ScalarT or AD at this point
    
    for (size_t f=0; f<forests.size(); ++f) {
      for (size_t k=0; k<forests[f].trees.size(); k++) {
        for (size_t j=0; j<forests[f].trees[k].branches.size(); j++) {
          
          // Rewrite this section
          
          bool isConst = true, isView = false, isAD = false;
          
          this->checkDepDataType(f,k,j, isConst, isView, isAD); // is this term a ScalarT
          
          forests[f].trees[k].branches[j].isConstant = isConst;
          forests[f].trees[k].branches[j].isView = isView;
          forests[f].trees[k].branches[j].isAD = isAD;
          
          if (isView) {
            if (isAD) {
              forests[f].trees[k].branches[j].viewdata = View_AD2("data", forests[f].dim0, forests[f].dim1);
            }
            else {
              forests[f].trees[k].branches[j].viewdata_Sc = View_Sc2("data", forests[f].dim0, forests[f].dim1);
            }
          }
        }
      }
    }
    
    // Now evaluate all of the constant branches (meaning all deps are const, !vector, !AD)
    for (size_t f=0; f<forests.size(); ++f) {
      for (size_t k=0; k<forests[f].trees.size(); k++) {
        for (size_t j=0; j<forests[f].trees[k].branches.size(); j++) {
          if (forests[f].trees[k].branches[j].isConstant) {
            if (!forests[f].trees[k].branches[j].isLeaf) { // leafs are already filled
              this->evaluate(f,k,j);
            }
          }
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Determine if a branch is a ScalarT or needs to be an AD type
//////////////////////////////////////////////////////////////////////////////////////

bool FunctionManager::isScalarTerm(const int & findex, const int & tindex, const int & bindex) {
  bool is_scalar = true;
  if (forests[findex].trees[tindex].branches[bindex].isLeaf) {
    if (forests[findex].trees[tindex].branches[bindex].isAD) {
      is_scalar = false;
    }
  }
  else if (forests[findex].trees[tindex].branches[bindex].isFunc) {
    is_scalar = false;
  }
  else {
    for (size_t k=0; k<forests[findex].trees[tindex].branches[bindex].dep_list.size(); k++){
      bool depcheck = isScalarTerm(findex, tindex, forests[findex].trees[tindex].branches[bindex].dep_list[k]);
      if (!depcheck) {
        is_scalar = false;
      }
    }
  }
  return is_scalar;
}


void FunctionManager::checkDepDataType(const int & findex, const int & tindex, const int & bindex,
                                       bool & isConst, bool & isView, bool & isAD) {
  
  
  if (forests[findex].trees[tindex].branches[bindex].isLeaf) {
    if (!forests[findex].trees[tindex].branches[bindex].isConstant) {
      isConst = false;
    }
    if (forests[findex].trees[tindex].branches[bindex].isView) {
      isView = true;
    }
    if (forests[findex].trees[tindex].branches[bindex].isAD) {
      isAD = true;
    }
  }
  else if (forests[findex].trees[tindex].branches[bindex].isFunc) {
    this->checkDepDataType(findex, forests[findex].trees[tindex].branches[bindex].funcIndex, 0,
                           isConst, isView, isAD);
  }
  else {
    for (size_t k=0; k<forests[findex].trees[tindex].branches[bindex].dep_list.size(); k++){
      this->checkDepDataType(findex, tindex, forests[findex].trees[tindex].branches[bindex].dep_list[k],
                             isConst, isView, isAD);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function (probably will be deprecated)
//////////////////////////////////////////////////////////////////////////////////////

View_AD2 FunctionManager::evaluate(const string & fname, const string & location) {
  //Teuchos::TimeMonitor ttimer(*evaluateExtTimer);
  
  bool ffound = false, tfound = false;
  size_t fiter=0, titer=0;
  while(!ffound && fiter<forests.size()) {
    if (forests[fiter].location == location) {
      ffound = true;
      tfound = false;
      while (!tfound && titer<forests[fiter].trees.size()) {
        if (fname == forests[fiter].trees[titer].name) {
          tfound = true;
          if (!forests[fiter].trees[titer].branches[0].isDecomposed) {
            this->decomposeFunctions();
          }
          if (!forests[fiter].trees[titer].branches[0].isConstant) {
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
  
  View_AD2 output = forests[fiter].trees[titer].branches[0].viewdata;
  if (output.extent(0) == 0) {
    output = View_AD2("data",forests[fiter].dim0,forests[fiter].dim1);
    forests[fiter].trees[titer].branches[0].viewdata = output;
    if (forests[fiter].trees[titer].branches[0].isConstant) {
      deep_copy(output,forests[fiter].trees[titer].branches[0].data_Sc);
    }
  }
  
  if (forests[fiter].trees[titer].branches[0].isAD) {
    if (forests[fiter].trees[titer].branches[0].isView) {
      if (forests[fiter].trees[titer].branches[0].isParameter) {
        int pind = forests[fiter].trees[titer].branches[0].paramIndex;
        auto pdata = forests[fiter].trees[titer].branches[0].param_data;
        //output = View_AD2("data",forests[fiter].dim0,forests[fiter].dim1);
        parallel_for("funcman copy double to AD",
                     TeamPolicy<AssemblyExec>(output.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<output.extent(1); pt+=team.team_size() ) {
            output(elem,pt) = pdata(pind);
          }
        });
      }
      //else {
      //  output = forests[fiter].trees[titer].branches[0].viewdata;
      //}
    }
    else {
      AD val = forests[fiter].trees[titer].branches[0].data;
      // Can this be a deep_copy?
      //output = View_AD2("data",forests[fiter].dim0,forests[fiter].dim1);
      parallel_for("funcman copy double to AD",
                   TeamPolicy<AssemblyExec>(output.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<output.extent(1); pt+=team.team_size() ) {
          output(elem,pt) = val;
        }
      });
    }
  }
  else {
    //output = View_AD2("output data",forests[fiter].dim0, forests[fiter].dim1);
    if (forests[fiter].trees[titer].branches[0].isView) {
      auto doutput = forests[fiter].trees[titer].branches[0].viewdata_Sc;
      parallel_for("funcman copy double to AD",
                   TeamPolicy<AssemblyExec>(output.extent(0), Kokkos::AUTO, VectorSize),
                   KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
        int elem = team.league_rank();
        for (size_type pt=team.team_rank(); pt<output.extent(1); pt+=team.team_size() ) {
          output(elem,pt) = doutput(elem,pt);
        }
      });
    }
    else {
      if (!forests[fiter].trees[titer].branches[0].isConstant) {
        ScalarT data = forests[fiter].trees[titer].branches[0].data_Sc;
        deep_copy(output,data);
      }
    }
  }
  return output;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::evaluate( const size_t & findex, const size_t & tindex, const size_t & bindex) {
  
  //Teuchos::TimeMonitor ttimer(*evaluateIntTimer);
  
  //if (!forests[findex].trees[tindex].branches[bindex].isConstant) {
    if (forests[findex].trees[tindex].branches[bindex].isLeaf) {
      if (forests[findex].trees[tindex].branches[bindex].isWorksetData) {
        int wdindex = forests[findex].trees[tindex].branches[bindex].workset_data_index;
        if (forests[findex].trees[tindex].branches[bindex].isAD) {
          forests[findex].trees[tindex].branches[bindex].viewdata = wkset->data[wdindex];
        }
        else {
          forests[findex].trees[tindex].branches[bindex].viewdata_Sc = wkset->data_Sc[wdindex];
        }
      }
      else if (forests[findex].trees[tindex].branches[bindex].isParameter) {
        // Should be set correctly already
      }
      else if (forests[findex].trees[tindex].branches[bindex].isTime) {
        forests[findex].trees[tindex].branches[bindex].data_Sc = wkset->time_KV(0);
      }
    }
    else if (forests[findex].trees[tindex].branches[bindex].isFunc) {
      int funcIndex = forests[findex].trees[tindex].branches[bindex].funcIndex;
      this->evaluate(findex,funcIndex, 0);
      
      if (forests[findex].trees[tindex].branches[bindex].isAD) {
        if (forests[findex].trees[tindex].branches[bindex].isView) { // use viewdata
          forests[findex].trees[tindex].branches[bindex].viewdata = forests[findex].trees[funcIndex].branches[0].viewdata;
        }
        else { // use data
          forests[findex].trees[tindex].branches[bindex].data = forests[findex].trees[funcIndex].branches[0].data;
        }
      }
      else {
        if (forests[findex].trees[tindex].branches[bindex].isView) { // use viewdata_Sc
          forests[findex].trees[tindex].branches[bindex].viewdata_Sc = forests[findex].trees[funcIndex].branches[0].viewdata_Sc;
        }
        else { // use data_Sc
          forests[findex].trees[tindex].branches[bindex].data_Sc = forests[findex].trees[funcIndex].branches[0].data_Sc;
        }
      }
    }
    else {
      bool isAD = forests[findex].trees[tindex].branches[bindex].isAD;
      bool isView = forests[findex].trees[tindex].branches[bindex].isView;
      for (size_t k=0; k<forests[findex].trees[tindex].branches[bindex].dep_list.size(); k++) {
        
        int dep = forests[findex].trees[tindex].branches[bindex].dep_list[k];
        this->evaluate(findex, tindex, dep);
        
        bool termisAD = forests[findex].trees[tindex].branches[dep].isAD;
        bool termisView = forests[findex].trees[tindex].branches[dep].isView;
        bool termisParameter = forests[findex].trees[tindex].branches[dep].isParameter;
        if (isView) {
          if (termisView) {
            if (isAD) {
              if (termisAD) {
                if (termisParameter) {
                  this->evaluateOpParamToV(forests[findex].trees[tindex].branches[bindex].viewdata,
                                           forests[findex].trees[tindex].branches[dep].param_data,
                                           forests[findex].trees[tindex].branches[dep].paramIndex,
                                           forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
                }
                else {
                  this->evaluateOpVToV(forests[findex].trees[tindex].branches[bindex].viewdata,
                                       forests[findex].trees[tindex].branches[dep].viewdata,
                                       forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
                }
                
              }
              else {
                this->evaluateOpVToV(forests[findex].trees[tindex].branches[bindex].viewdata,
                                     forests[findex].trees[tindex].branches[dep].viewdata_Sc,
                                     forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
              }
            }
            else {
              if (termisAD) {
                // output error
              }
              else {
                this->evaluateOpVToV(forests[findex].trees[tindex].branches[bindex].viewdata_Sc,
                                     forests[findex].trees[tindex].branches[dep].viewdata_Sc,
                                     forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
              }
            }
          }
          else { // Scalar data
            if (isAD) {
              if (termisAD) {
                this->evaluateOpSToV(forests[findex].trees[tindex].branches[bindex].viewdata,
                                     forests[findex].trees[tindex].branches[dep].data,
                                     forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
              }
              else {
                this->evaluateOpSToV(forests[findex].trees[tindex].branches[bindex].viewdata,
                                     forests[findex].trees[tindex].branches[dep].data_Sc,
                                     forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
              }
            }
            else {
              if (termisAD) {
                //error
              }
              else {
                this->evaluateOpSToV(forests[findex].trees[tindex].branches[bindex].viewdata_Sc,
                                     forests[findex].trees[tindex].branches[dep].data_Sc,
                                     forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
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
                this->evaluateOpSToS(forests[findex].trees[tindex].branches[bindex].data,
                                     forests[findex].trees[tindex].branches[dep].data,
                                     forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
              }
              else {
                this->evaluateOpSToS(forests[findex].trees[tindex].branches[bindex].data,
                                     forests[findex].trees[tindex].branches[dep].data_Sc,
                                     forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
              }
            }
            else {
              if (termisAD) {
                //error
              }
              else {
                this->evaluateOpSToS(forests[findex].trees[tindex].branches[bindex].data_Sc,
                                     forests[findex].trees[tindex].branches[dep].data_Sc,
                                     forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
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

template<class T1, class T2>
void FunctionManager::evaluateOpVToV(T1 data, T2 tdata, const string & op) {
  
  //Teuchos::TimeMonitor ttimer(*evaluateOpTimer);
  
  size_t dim0 = std::min(data.extent(0),tdata.extent(0));
  using namespace std;
  
  if (op == "") {
    parallel_for("funcman evaluate equals",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
  else if (op == "max") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for("funcman evaluate max",RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
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
  else if (op == "min") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for("funcman evaluate min",RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
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
  else if (op == "mean") { // mean over rows ... usually corr. to mean over element/face
    parallel_for("funcman evaluate mean",RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      double scale = (double)dim1;
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
  /*else if (op == "lte") { // TMW: commenting this for now
   parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
   size_t dim1 = min(data.extent(1),tdata.extent(1));
   for (unsigned int n=0; n<dim1; n++) {
   if (data(e,n) <= tdata(e,n)) {
   data(e,n) = 1.0;
   }
   else {
   data(e,n) = 0.0;
   }
   }
   });
   }*/
  else if (op == "gt") {
    parallel_for("funcman evaluate gt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
  /*else if (op == "gte") { // TMW: commenting this for now
   parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
   size_t dim1 = min(data.extent(1),tdata.extent(1));
   for (unsigned int n=0; n<dim1; n++) {
   if (data(e,n) >= tdata(e,n)) {
   data(e,n) = 1.0;
   }
   else {
   data(e,n) = 0.0;
   }
   }
   });
   }*/
  else if (op == "sqrt") {
    parallel_for("funcman evaluate sqrt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class T1, class T2>
void FunctionManager::evaluateOpParamToV(T1 data, T2 tdata, const int & pIndex_, const string & op) {
  
  //Teuchos::TimeMonitor ttimer(*evaluateOpTimer);
  
  size_t dim0 = data.extent(0);
  using namespace std;
  
  int pIndex = pIndex_;
  
  if (op == "") {
    parallel_for("funcman evaluate equals",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = cos(tdata(pIndex));
      }
    });
  }
  else if (op == "tan") {
    parallel_for("funcman evaluate tan",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = log(tdata(pIndex));
      }
    });
  }
  else if (op == "abs") {
    parallel_for("funcman evaluate abs",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
  else if (op == "max") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for("funcman evaluate max",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata(pIndex);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "min") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for("funcman evaluate min",RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata(pIndex);
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "mean") { // mean over rows ... usually corr. to mean over element/face
    parallel_for("funcman evaluate mean",RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata(pIndex);
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "lt") {
    parallel_for("funcman evaluate lt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
  /*else if (op == "lte") { // TMW: commenting this for now
   parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
   size_t dim1 = min(data.extent(1),tdata.extent(1));
   for (unsigned int n=0; n<dim1; n++) {
   if (data(e,n) <= tdata(e,n)) {
   data(e,n) = 1.0;
   }
   else {
   data(e,n) = 0.0;
   }
   }
   });
   }*/
  else if (op == "gt") {
    parallel_for("funcman evaluate gt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
  /*else if (op == "gte") { // TMW: commenting this for now
   parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
   size_t dim1 = min(data.extent(1),tdata.extent(1));
   for (unsigned int n=0; n<dim1; n++) {
   if (data(e,n) >= tdata(e,n)) {
   data(e,n) = 1.0;
   }
   else {
   data(e,n) = 0.0;
   }
   }
   });
   }*/
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class T1, class T2>
void FunctionManager::evaluateOpSToV(T1 data, T2 & tdata_, const string & op) {
  
  //Teuchos::TimeMonitor ttimer(*evaluateOpTimer);
  
  T2 tdata = tdata_; // Probably don't need to do this if pass by value
  size_t dim0 = data.extent(0);
  using namespace std;
  
  if (op == "") {
    parallel_for("funcman evaluate equals",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = tan(tdata);
      }
    });
  }
  else if (op == "exp") {
    parallel_for("funcman evaluate exp",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      size_t dim1 = data.extent(1);
      for (size_type pt=team.team_rank(); pt<dim1; pt+=team.team_size() ) {
        data(elem,pt) = log(tdata);
      }
    });
  }
  else if (op == "abs") {
    parallel_for("funcman evaluate abs",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
  else if (op == "max") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for("funcman evaluate max",
                 RangePolicy<AssemblyExec>(0,dim0),
                 KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata;
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "min") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for("funcman evaluate min",RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata;
      for (unsigned int n=0; n<dim1; n++) { // copy min value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "mean") { // mean over rows ... usually corr. to mean over element/face
    parallel_for("funcman evaluate mean",RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = data.extent(1);
      data(e,0) = tdata;
      for (unsigned int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "lt") {
    parallel_for("funcman evaluate lt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
  /*else if (op == "lte") { // TMW: commenting this for now
   parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
   size_t dim1 = min(data.extent(1),tdata.extent(1));
   for (unsigned int n=0; n<dim1; n++) {
   if (data(e,n) <= tdata(e,n)) {
   data(e,n) = 1.0;
   }
   else {
   data(e,n) = 0.0;
   }
   }
   });
   }*/
  else if (op == "gt") {
    parallel_for("funcman evaluate gt",
                 TeamPolicy<AssemblyExec>(dim0, Kokkos::AUTO, VectorSize),
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
  /*else if (op == "gte") { // TMW: commenting this for now
   parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
   size_t dim1 = min(data.extent(1),tdata.extent(1));
   for (unsigned int n=0; n<dim1; n++) {
   if (data(e,n) >= tdata(e,n)) {
   data(e,n) = 1.0;
   }
   else {
   data(e,n) = 0.0;
   }
   }
   });
   }*/
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class T1, class T2>
void FunctionManager::evaluateOpSToS(T1 & data, T2 & tdata, const string & op) {
  
  //Teuchos::TimeMonitor ttimer(*evaluateOpTimer);
  
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
  else if (op == "exp") {
    data = exp(tdata);
  }
  else if (op == "log") {
    data = log(tdata);
  }
  else if (op == "abs") {
    if (tdata < 0.0) {
      data = -tdata;
    }
    else {
      data = tdata;
    }
  }
  else if (op == "max") { // maximum over rows ... usually corr. to max over element/face at ip
    data = tdata;
  }
  else if (op == "min") { // minimum over rows ... usually corr. to min over element/face at ip
    data = tdata;
  }
  else if (op == "mean") { // mean over rows ... usually corr. to mean over element/face
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
  /*else if (op == "lte") { // TMW: commenting this for now
   parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
   size_t dim1 = min(data.extent(1),tdata.extent(1));
   for (unsigned int n=0; n<dim1; n++) {
   if (data(e,n) <= tdata(e,n)) {
   data(e,n) = 1.0;
   }
   else {
   data(e,n) = 0.0;
   }
   }
   });
   }*/
  else if (op == "gt") {
    if (data > tdata) {
      data = 1.0;
    }
    else {
      data = 0.0;
    }
  }
  /*else if (op == "gte") { // TMW: commenting this for now
   parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
   size_t dim1 = min(data.extent(1),tdata.extent(1));
   for (unsigned int n=0; n<dim1; n++) {
   if (data(e,n) >= tdata(e,n)) {
   data(e,n) = 1.0;
   }
   else {
   data(e,n) = 0.0;
   }
   }
   });
   }*/
  
}


//////////////////////////////////////////////////////////////////////////////////////
// Print out the function information (mostly for debugging)
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::printFunctions() {
  
  cout << endl;
  cout << "===========================================================" << endl;
  cout << "Printing functions on block: " << blockname << endl;
  cout << "-----------------------------------------------------------" << endl;
  
  for (size_t k=0; k<forests.size(); k++) {
    
    cout << "Forest Name:" << forests[k].location << endl;
    cout << "Number of Trees: " << forests[k].trees.size() << endl;
    for (size_t t=0; t<forests[k].trees.size(); t++) {
      cout << "    Tree: " << forests[k].trees[t].name << endl;
      cout << "    Number of branches: " << forests[k].trees[t].branches.size() << endl;
      for (size_t b=0; b<forests[k].trees[t].branches.size(); b++) {
        cout << "        " << forests[k].trees[t].branches[b].expression << endl;
      }
    }
    
    cout << "-----------------------------------------------------------" << endl;
    
  }
  
}


