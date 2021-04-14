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
  known_ops = {"sin","cos","exp","log","tan","abs","max","min","mean"};
  
  interpreter = Teuchos::rcp( new Interpreter());
  
}


FunctionManager::FunctionManager(const string & blockname_, const int & numElem_,
                                 const int & numip_, const int & numip_side_) :
blockname(blockname_), numElem(numElem_), numip(numip_), numip_side(numip_side_) {
  
  known_vars = {"x","y","z","t","nx","ny","nz","pi","h"};
  known_ops = {"sin","cos","exp","log","tan","abs","max","min","mean"};
  
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
// Validate all of the functions
//////////////////////////////////////////////////////////////////////////////////////

/*
// TMW: THIS HAS BEEN DEPRECATED
void FunctionManager::validateFunctions(){
  vector<string> function_names;
  for (size_t k=0; k<functions.size(); k++) {
    function_names.push_back(functions[k].function_name);
  }
  for (size_t k=0; k<functions.size(); k++) {
    vector<string> vars = interpreter->getVars(functions[k].expression, known_ops);
    int numfails = interpreter->validateTerms(vars,known_vars,variables,parameters,disc_parameters,function_names);
    if (numfails > 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"Error: MrHyDE could not identify one or more terms in: " + functions[k].function_name);
    }
  }
}
*/

//////////////////////////////////////////////////////////////////////////////////////
// Decompose the functions into terms and set the evaluation tree
// Also sets up the Kokkos::Views (subviews) to the data for all of the terms
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::decomposeFunctions() {
  
  Teuchos::TimeMonitor ttimer(*decomposeTimer);
  
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
        
          // Is it an AD data stored in the workset?
          if (decompose) {
            vector<string> data_labels = wkset->data_labels;
            string label = forests[fiter].trees[titer].branches[k].expression;
            if (forests[fiter].location == "side ip") {
              label += " side";
            }
            else if (forests[fiter].location == "point") {
              label += " point";
            }
            bool found = 0;
            size_t j=0;
            while (!found && j<data_labels.size()) {
              if (label == data_labels[j]) {
                decompose = false;
                forests[fiter].trees[titer].branches[k].isLeaf = true;
                forests[fiter].trees[titer].branches[k].isDecomposed = true;
                forests[fiter].trees[titer].branches[k].isAD = true;
                forests[fiter].trees[titer].branches[k].isWorksetData = true;
                forests[fiter].trees[titer].branches[k].workset_data_index = j;
                //forests[fiter].trees[titer].branches[k].data = wkset->data[j];
                
                found = true;
              }
              j++;
            }
          }
        
          // Is it a Scalar data stored in the workset?
          if (decompose) {
            vector<string> data_Sc_labels = wkset->data_Sc_labels;
            string label = forests[fiter].trees[titer].branches[k].expression;
            if (forests[fiter].location == "side ip") {
              label += " side";
            }
            else if (forests[fiter].location == "point") {
              label += " point";
            }
            bool found = 0;
            size_t j=0;
            while (!found && j<data_Sc_labels.size()) {
              if (label == data_Sc_labels[j]) {
                decompose = false;
                forests[fiter].trees[titer].branches[k].isLeaf = true;
                forests[fiter].trees[titer].branches[k].isDecomposed = true;
                forests[fiter].trees[titer].branches[k].isAD = false;
                forests[fiter].trees[titer].branches[k].isWorksetData = true;
                forests[fiter].trees[titer].branches[k].workset_data_index = j;
                
                //forests[fiter].trees[titer].branches[k].ddata = wkset->data_Sc[j];
                found = true;
              }
              j++;
            }
          }
        
          // check if it is a parameter
          if (decompose) {
            
            for (unsigned int j=0; j<parameters.size(); j++) {
              
              if (forests[fiter].trees[titer].branches[k].expression == parameters[j]) {
                forests[fiter].trees[titer].branches[k].isLeaf = true;
                forests[fiter].trees[titer].branches[k].isAD = true;
                forests[fiter].trees[titer].branches[k].isDecomposed = true;
                forests[fiter].trees[titer].branches[k].isScalar = true;
                forests[fiter].trees[titer].branches[k].isConstant = false; // needs to be copied
                forests[fiter].trees[titer].branches[k].scalarIndex = 0;
                
                decompose = false;
                
                forests[fiter].trees[titer].branches[k].scalar_data = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
                
                View_AD2 tdata("scalar data",forests[fiter].dim0,forests[fiter].dim1);
                forests[fiter].trees[titer].branches[k].data = tdata;
                
              }
              else { // look for param(*) or param(**)
                bool found = true;
                int sindex = 0;
                size_t nexp = forests[fiter].trees[titer].branches[k].expression.length();
                if (nexp == parameters[j].length()+3) {
                  for (size_t n=0; n<parameters[j].length(); n++) {
                    if (forests[fiter].trees[titer].branches[k].expression[n] != parameters[j][n]) {
                      found = false;
                    }
                  }
                  if (found) {
                    if (forests[fiter].trees[titer].branches[k].expression[nexp-3] == '(' && forests[fiter].trees[titer].branches[k].expression[nexp-1] == ')') {
                      string check = "";
                      check += forests[fiter].trees[titer].branches[k].expression[nexp-2];
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
                    if (forests[fiter].trees[titer].branches[k].expression[n] != parameters[j][n]) {
                      found = false;
                    }
                  }
                  if (found) {
                    if (forests[fiter].trees[titer].branches[k].expression[nexp-4] == '(' && forests[fiter].trees[titer].branches[k].expression[nexp-1] == ')') {
                      string check = "";
                      check += forests[fiter].trees[titer].branches[k].expression[nexp-3];
                      check += forests[fiter].trees[titer].branches[k].expression[nexp-2];
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
                  forests[fiter].trees[titer].branches[k].isAD = true;
                  forests[fiter].trees[titer].branches[k].isDecomposed = true;
                  forests[fiter].trees[titer].branches[k].isScalar = true;
                  forests[fiter].trees[titer].branches[k].isConstant = false; // needs to be copied
                  forests[fiter].trees[titer].branches[k].scalarIndex = sindex;
                  
                  decompose = false;
                  
                  forests[fiter].trees[titer].branches[k].scalar_data = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
                  
                  View_AD2 tdata("scalar data",forests[fiter].dim0,forests[fiter].dim1);
                  forests[fiter].trees[titer].branches[k].data = tdata;
                }
              }
            }
          }
          
          // check if it is a function
          if (decompose) {
            for (unsigned int j=0; j<forests[fiter].trees.size(); j++) {
              if (forests[fiter].trees[titer].branches[k].expression == forests[fiter].trees[j].name) {
                forests[fiter].trees[titer].branches[k].isFunc = true;
                forests[fiter].trees[titer].branches[k].isAD = true;//functions[j].terms[0].isAD;
                forests[fiter].trees[titer].branches[k].funcIndex = j;
                forests[fiter].trees[titer].branches[k].isDecomposed = true;
                forests[fiter].trees[titer].branches[k].data = forests[fiter].trees[j].branches[0].data;
                forests[fiter].trees[titer].branches[k].ddata = forests[fiter].trees[j].branches[0].ddata;
                decompose = false;
              }
            }
          }
        
          // IS THE TERM A SIMPLE SCALAR: 2.03, 1.0E2, etc.
          if (decompose) {
            bool isnum = interpreter->isScalar(forests[fiter].trees[titer].branches[k].expression);
            if (isnum) {
              if (k==0) {
                forests[fiter].trees[titer].branches[k].isLeaf = true;
                forests[fiter].trees[titer].branches[k].isAD = true;
                forests[fiter].trees[titer].branches[k].isDecomposed = true;
                forests[fiter].trees[titer].branches[k].isScalar = true;
                forests[fiter].trees[titer].branches[k].isConstant = true; // means in does not need to be copied every time
                forests[fiter].trees[titer].branches[k].scalar_ddata = Kokkos::View<double*,AssemblyDevice>("scalar double data",1);
                ScalarT val = std::stod(forests[fiter].trees[titer].branches[k].expression);
                Kokkos::deep_copy(forests[fiter].trees[titer].branches[k].scalar_ddata, val);
                
                // Copy the data just once
                View_AD2 tdata("scalar data",forests[fiter].dim0,forests[fiter].dim1);
                forests[fiter].trees[titer].branches[k].data = tdata;
                Kokkos::deep_copy(forests[fiter].trees[titer].branches[k].data, val);
                decompose = false;
              }
              else {
                forests[fiter].trees[titer].branches[k].isLeaf = true;
                forests[fiter].trees[titer].branches[k].isAD = false;
                forests[fiter].trees[titer].branches[k].isDecomposed = true;
                forests[fiter].trees[titer].branches[k].isScalar = true;
                forests[fiter].trees[titer].branches[k].isConstant = true; // means in does not need to be copied every time
                forests[fiter].trees[titer].branches[k].scalar_ddata = Kokkos::View<double*,AssemblyDevice>("scalar double data",1);
                ScalarT val = std::stod(forests[fiter].trees[titer].branches[k].expression);
                Kokkos::deep_copy(forests[fiter].trees[titer].branches[k].scalar_ddata, val);
                
                // Copy the data just once
                View_Sc2 tdata("scalar data",forests[fiter].dim0,forests[fiter].dim1);
                forests[fiter].trees[titer].branches[k].ddata = tdata;
                Kokkos::deep_copy(forests[fiter].trees[titer].branches[k].ddata, val);
                decompose = false;
              }
            }
          }
        
          // IS THE TERM ONE OF THE KNOWN VARIABLES: t or pi
          if (decompose) {
            for (size_t j=0; j<known_vars.size(); j++) {
              if (forests[fiter].trees[titer].branches[k].expression == known_vars[j]) {
                decompose = false;
                //bool have_data = false;
                forests[fiter].trees[titer].branches[k].isLeaf = true;
                forests[fiter].trees[titer].branches[k].isDecomposed = true;
                forests[fiter].trees[titer].branches[k].isAD = false;
                forests[fiter].trees[titer].branches[k].isConstant = false;
                if (known_vars[j] == "t") {
                  forests[fiter].trees[titer].branches[k].scalar_ddata = wkset->time_KV;
                  forests[fiter].trees[titer].branches[k].isScalar = true;
                  forests[fiter].trees[titer].branches[k].ddata = View_Sc2("data",forests[fiter].dim0,forests[fiter].dim1);
                }
                else if (known_vars[j] == "pi") {
                  forests[fiter].trees[titer].branches[k].isLeaf = true;
                  forests[fiter].trees[titer].branches[k].isAD = false;
                  forests[fiter].trees[titer].branches[k].isDecomposed = true;
                  forests[fiter].trees[titer].branches[k].isScalar = true;
                  forests[fiter].trees[titer].branches[k].isConstant = true; // means in does not need to be copied every time
                  View_Sc2 tdata("scalar data", forests[fiter].dim0, forests[fiter].dim1);
                  forests[fiter].trees[titer].branches[k].ddata = tdata;
                  
                  Kokkos::deep_copy(forests[fiter].trees[titer].branches[k].ddata, PI);
                  decompose = false;
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
        bool termcheck = this->isScalarTerm(f,k,j); // is this term a ScalarT
        if (termcheck) {
          forests[f].trees[k].branches[j].isAD = false;
          if (!forests[f].trees[k].branches[j].isLeaf) {
            View_Sc2 tdata("data", forests[f].dim0, forests[f].dim1);
            forests[f].trees[k].branches[j].ddata = tdata;
          }
          if (j==0) { // always need this allocated
            View_AD2 tdata("data", forests[f].dim0, forests[f].dim1);
            forests[f].trees[k].branches[j].data = tdata;
          }
        }
        else if (!forests[f].trees[k].branches[j].isLeaf) {
          forests[f].trees[k].branches[j].isAD = true;
          View_AD2 tdata("data",forests[f].dim0,forests[f].dim1);
          forests[f].trees[k].branches[j].data = tdata;
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

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function (probably will be deprecated)
//////////////////////////////////////////////////////////////////////////////////////

View_AD2 FunctionManager::evaluate(const string & fname, const string & location) {
  Teuchos::TimeMonitor ttimer(*evaluateTimer);
  
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
          this->evaluate(fiter,titer,0);
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
  
  View_AD2 output = forests[fiter].trees[titer].branches[0].data;
  if (!forests[fiter].trees[titer].branches[0].isAD) {
    auto doutput = forests[fiter].trees[titer].branches[0].ddata;
    parallel_for("funcman copy double to AD",
                 TeamPolicy<AssemblyExec>(output.extent(0), Kokkos::AUTO, VectorSize),
                 KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
      int elem = team.league_rank();
      for (size_type pt=team.team_rank(); pt<output.extent(1); pt+=team.team_size() ) {
        output(elem,pt) = doutput(elem,pt);
      }
    });
  }
  return output;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::evaluate( const size_t & findex, const size_t & tindex, const size_t & bindex) {
  
  //if (verbosity > 10) {
  //  cout << "------- Evaluating: " << forests[findex].trees[tindex].branches[bindex].expression << endl;
  //}
  
  //forests[findex].trees[tindex].branches[bindex].print();
  
  if (forests[findex].trees[tindex].branches[bindex].isLeaf) {
    if (forests[findex].trees[tindex].branches[bindex].isScalar && !forests[findex].trees[tindex].branches[bindex].isConstant) {
      if (forests[findex].trees[tindex].branches[bindex].isAD) { // TMW change to deep_copy
        auto data0 = forests[findex].trees[tindex].branches[bindex].data;
        auto data1 = forests[findex].trees[tindex].branches[bindex].scalar_data;
        parallel_for("funcman copy scalar to View_AD2",
                     TeamPolicy<AssemblyExec>(data0.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<data0.extent(1); pt+=team.team_size() ) {
            data0(elem,pt) = data1(0);
          }
        });
      }
      else { // TMW change to deep_copy
        auto data0 = forests[findex].trees[tindex].branches[bindex].ddata;
        auto data1 = forests[findex].trees[tindex].branches[bindex].scalar_ddata;
        parallel_for("funcman copy scalar to View_Sc2",
                     TeamPolicy<AssemblyExec>(data0.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<data0.extent(1); pt+=team.team_size() ) {
            data0(elem,pt) = data1(0);
          }
        });
      }
    }
    else if (forests[findex].trees[tindex].branches[bindex].isWorksetData) {
      int wdindex = forests[findex].trees[tindex].branches[bindex].workset_data_index;
      if (forests[findex].trees[tindex].branches[bindex].isAD) {
        forests[findex].trees[tindex].branches[bindex].data = wkset->data[wdindex];
      }
      else {
        forests[findex].trees[tindex].branches[bindex].ddata = wkset->data_Sc[wdindex];
      }
    }
  }
  else if (forests[findex].trees[tindex].branches[bindex].isFunc) {
    int funcIndex = forests[findex].trees[tindex].branches[bindex].funcIndex;
    this->evaluate(findex,funcIndex, 0);
    if (forests[findex].trees[tindex].branches[bindex].isAD) {
      if (forests[findex].trees[funcIndex].branches[0].isAD) {
        forests[findex].trees[tindex].branches[bindex].data = forests[findex].trees[funcIndex].branches[0].data;
      }
      else { // TMW try to change to deep copy
        auto data0 = forests[findex].trees[tindex].branches[bindex].data;
        auto data1 = forests[findex].trees[funcIndex].branches[0].ddata;
        parallel_for("funcman copy View_Sc2 to View_AD2",
                     TeamPolicy<AssemblyExec>(data0.extent(0), Kokkos::AUTO, VectorSize),
                     KOKKOS_LAMBDA (TeamPolicy<AssemblyExec>::member_type team ) {
          int elem = team.league_rank();
          for (size_type pt=team.team_rank(); pt<data0.extent(1); pt+=team.team_size() ) {
            data0(elem,pt) = data1(elem,pt);
          }
        });
      }
    }
    else {
      forests[findex].trees[tindex].branches[bindex].ddata = forests[findex].trees[funcIndex].branches[0].ddata;
    }
  }
  else {
    bool isAD = forests[findex].trees[tindex].branches[bindex].isAD;
    for (size_t k=0; k<forests[findex].trees[tindex].branches[bindex].dep_list.size(); k++) {
      
      int dep = forests[findex].trees[tindex].branches[bindex].dep_list[k];
      this->evaluate(findex, tindex, dep);
      
      bool termisAD = forests[findex].trees[tindex].branches[dep].isAD;
      if (isAD) {
        if (termisAD) {
          this->evaluateOp(forests[findex].trees[tindex].branches[bindex].data,
                           forests[findex].trees[tindex].branches[dep].data,
                           forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
          
        }
        else {
          this->evaluateOp(forests[findex].trees[tindex].branches[bindex].data,
                           forests[findex].trees[tindex].branches[dep].ddata,
                           forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
        }
      }
      else { // termisAD must also be false
        this->evaluateOp(forests[findex].trees[tindex].branches[bindex].ddata,
                         forests[findex].trees[tindex].branches[dep].ddata,
                         forests[findex].trees[tindex].branches[bindex].dep_ops[k]);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class T1, class T2>
void FunctionManager::evaluateOp(T1 data, T2 tdata, const string & op) {
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


