/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "functionManager.hpp"
#include "interpreter.hpp"

FunctionManager::FunctionManager() {
  // This really should NOT be constructed
  
  numElem = 1;
  numip = 1;
  numip_side = 1;
  known_vars = {"x","y","z","t","nx","ny","nz","pi","h"};
  known_ops = {"sin","cos","exp","log","tan","abs","max","min","mean"};
  
  /*
  vector<string> known_vars_str = {"x","y","z","t","nx","ny","nz","pi","h"};
  vector<string> known_ops_str = {"sin","cos","exp","log","tan","abs","max","min","mean"};
  
  known_vars = Kokkos::View<string*,UnifiedDevice>("known variables",known_vars_str.size());
  known_ops = Kokkos::View<string*,UnifiedDevice>("known operators",known_ops_str.size());
   */
}


FunctionManager::FunctionManager(const string & blockname_, const int & numElem_,
                                 const int & numip_, const int & numip_side_) :
blockname(blockname_), numElem(numElem_), numip(numip_), numip_side(numip_side_) {
  known_vars = {"x","y","z","t","nx","ny","nz","pi","h"};
  known_ops = {"sin","cos","exp","log","tan","abs","max","min","mean"};
}

//////////////////////////////////////////////////////////////////////////////////////
// Add a user defined function
//////////////////////////////////////////////////////////////////////////////////////

int FunctionManager::addFunction(const string & fname, const string & expression, const string & location) {
  bool found = false;
  int findex = 0;
  
  for (size_t k=0; k<functions.size(); k++) {
    if (functions[k].function_name == fname && functions[k].location == location) {
      found = true;
      findex = k;
    }
  }
  if (!found) {
    int dim1;
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
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Set the lists of variables, parameters and discretized parameters
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::setupLists(const vector<string> & variables_,
                                 const vector<string> & parameters_,
                                 const vector<string> & disc_parameters_) {
  variables = variables_;
  parameters = parameters_;
  disc_parameters = disc_parameters_;
}

//////////////////////////////////////////////////////////////////////////////////////
// Validate all of the functions
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::validateFunctions(){
  vector<string> function_names;
  for (size_t k=0; k<functions.size(); k++) {
    function_names.push_back(functions[k].function_name);
  }
  for (size_t k=0; k<functions.size(); k++) {
    vector<string> vars = getVars(functions[k].expression, known_ops);
    
    int numfails = validateTerms(vars,known_vars,variables,parameters,disc_parameters,function_names);
    if (numfails > 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"Error: MILO could not identify one or more terms in: " + functions[k].function_name);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Decompose the functions into terms and set the evaluation tree
// Also sets up the Kokkos::Views (subviews) to the data for all of the terms
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::decomposeFunctions() {
  
  Teuchos::TimeMonitor ttimer(*decomposeTimer);
  
  for (size_t fiter=0; fiter<functions.size(); fiter++) {
    
    bool done = false; // will turn to "true" when the function is fully decomposed
    int maxiter = 20; // maximum number of recursions
    int iter = 0;
    
    while (!done && iter < maxiter) {
      
      iter++;
      size_t Nterms = functions[fiter].terms.size();
      
      for (size_t k=0; k<Nterms; k++) {
        
        // HAVE WE ALREADY LOOKED AT THIS TERM?
        bool decompose = true;
        if (functions[fiter].terms[k].isRoot || functions[fiter].terms[k].beenDecomposed) {
          decompose = false;
        }
        
        // IS THE TERM ONE OF THE KNOWN VARIABLES: x,y,z,t
        if (decompose) {
          for (size_t j=0; j<known_vars.size(); j++) {
            if (functions[fiter].terms[k].expression == known_vars[j]) {
              decompose = false;
              bool have_data = false;
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].beenDecomposed = true;
              functions[fiter].terms[k].isAD = false;
              if (known_vars[j] == "x") {
                if (functions[fiter].location == "side ip") {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->ip_side_KV, Kokkos::ALL(), Kokkos::ALL(), 0);
                }
                else if (functions[fiter].location == "point") {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->point_KV, Kokkos::ALL(), Kokkos::ALL(), 0);
                }
                else {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->ip_KV, Kokkos::ALL(), Kokkos::ALL(), 0);
                }
              }
              else if (known_vars[j] == "y") {
                if (functions[fiter].location == "side ip") {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->ip_side_KV, Kokkos::ALL(), Kokkos::ALL(), 1);
                }
                else if (functions[fiter].location == "point") {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->point_KV, Kokkos::ALL(), Kokkos::ALL(), 1);
                }
                else {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->ip_KV, Kokkos::ALL(), Kokkos::ALL(), 1);
                }
              }
              else if (known_vars[j] == "z") {
                if (functions[fiter].location == "side ip") {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->ip_side_KV, Kokkos::ALL(), Kokkos::ALL(), 2);
                }
                else if (functions[fiter].location == "point") {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->point_KV, Kokkos::ALL(), Kokkos::ALL(), 2);
                }
                else {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->ip_KV, Kokkos::ALL(), Kokkos::ALL(), 2);
                }
              }
              else if (known_vars[j] == "t") {
                //functions[b][fiter].terms[k].scalar_ddata = Kokkos::subview(wkset->time_KV, Kokkos::ALL(), 0);
                functions[fiter].terms[k].scalar_ddata = wkset->time_KV;
                functions[fiter].terms[k].isScalar = true;
                functions[fiter].terms[k].isConstant = false;
                Kokkos::View<double***,AssemblyDevice> tdata("data",functions[fiter].dim0,functions[fiter].dim1,1);
                functions[fiter].terms[k].ddata = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
                
              }
              else if (known_vars[j] == "nx") {
                if (functions[fiter].location == "side ip") {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->normals_KV, Kokkos::ALL(), Kokkos::ALL(), 0);
                }
                else {
                  //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: normals can only be used in functions defined on boundaries or faces");
                }
              }
              else if (known_vars[j] == "ny") {
                if (functions[fiter].location == "side ip") {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->normals_KV, Kokkos::ALL(), Kokkos::ALL(), 1);
                }
                else {
                  //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: normals can only be used in functions defined on boundaries or faces");
                }
              }
              else if (known_vars[j] == "nz") {
                if (functions[fiter].location == "side ip") {
                  functions[fiter].terms[k].ddata = Kokkos::subview(wkset->normals_KV, Kokkos::ALL(), Kokkos::ALL(), 2);
                }
                else {
                  //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: normals can only be used in functions defined on boundaries or faces");
                }
              }
              else if (known_vars[j] == "pi") {
                functions[fiter].terms[k].isRoot = true;
                functions[fiter].terms[k].isAD = false;
                functions[fiter].terms[k].beenDecomposed = true;
                functions[fiter].terms[k].isScalar = true;
                functions[fiter].terms[k].isConstant = true; // means in does not need to be copied every time
                have_data = true;
                // Copy the data just once
                Kokkos::View<double***,AssemblyDevice> tdata("scalar data",
                                                             functions[fiter].dim0,
                                                             functions[fiter].dim1,1);
                functions[fiter].terms[k].ddata = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
                for (size_t k2=0; k2<functions[fiter].dim0; k2++) {
                  for (size_t j2=0; j2<functions[fiter].dim1; j2++) {
                    functions[fiter].terms[k].ddata(k2,j2) = PI;
                  }
                }
                decompose = false;
              }
            }
          }
        } // end known_vars
        
        // IS THIS TERM ONE OF THE KNOWN OPERATORS: sin(...), exp(...), etc.
        if (decompose) {
          bool isop = isOperator(functions[fiter].terms, k, known_ops);
          // isOperator takes care of the decomposition if it is of this form
          if (isop) {
            decompose = false;
          }
        }
        
        // IS IT ONE OF THE VARIABLES (
        if (decompose) {
          for (unsigned int j=0; j<variables.size(); j++) {
            if (functions[fiter].terms[k].expression == variables[j]) { // just scalar variables
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              decompose = false;
              if (functions[fiter].location == "side ip") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_side, Kokkos::ALL(), j, Kokkos::ALL(), 0);
              }
              else if (functions[fiter].location == "point") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_point, Kokkos::ALL(), j, Kokkos::ALL(), 0);
              }
              else {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln, Kokkos::ALL(), j, Kokkos::ALL(), 0);
              }
            }
            else if (functions[fiter].terms[k].expression == (variables[j]+"_x")) { // deriv. of scalar var. w.r.t x
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              decompose = false;
              if (functions[fiter].location == "side ip") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 0);
              }
              else if (functions[fiter].location == "point") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_point, Kokkos::ALL(), j, Kokkos::ALL(), 0);
              }
              else {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 0);
              }
            }
            else if (functions[fiter].terms[k].expression == (variables[j]+"_y")) { // deriv. of scalar var. w.r.t y
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              decompose = false;
              if (functions[fiter].location == "side ip") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 1);
              }
              else if (functions[fiter].location == "point") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_point, Kokkos::ALL(), j, Kokkos::ALL(), 1);
              }
              else {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 1);
              }
            }
            else if (functions[fiter].terms[k].expression == (variables[j]+"_z")) { // deriv. of scalar var. w.r.t z
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              decompose = false;
              if (functions[fiter].location == "side ip") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 2);
              }
              else if (functions[fiter].location == "point") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_point, Kokkos::ALL(), j, Kokkos::ALL(), 2);
              }
              else {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 2);
              }
            }
            else if (functions[fiter].terms[k].expression == (variables[j]+"_t")) { // deriv. of scalar var. w.r.t x
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              decompose = false;
              if (functions[fiter].location == "side ip" || functions[fiter].location == "point") {
                TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MILO currently does not support the time derivative of a variable on boundaries or point evaluation points.");
              }
              else {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_dot, Kokkos::ALL(), j, Kokkos::ALL(), 0);
              }
            }
            else if (functions[fiter].terms[k].expression == (variables[j]+"[x]")) { // x-component of vector scalar var.
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              decompose = false;
              if (functions[fiter].location == "side ip") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 0);
              }
              else { // TMW: NOT UPDATED FOR point
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 0);
              }
            }
            else if (functions[fiter].terms[k].expression == (variables[j]+"[y]")) { // y-component of vector scalar var.
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              decompose = false;
              if (functions[fiter].location == "side ip") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 1);
              }
              else { // TMW: NOT UPDATED FOR point
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 1);
              }
            }
            else if (functions[fiter].terms[k].expression == (variables[j]+"[z]")) { // z-component of vector scalar var.
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              decompose = false;
              if (functions[fiter].location == "side ip") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 2);
              }
              else { // TMW: NOT UPDATED FOR point
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 2);
              }
            }
            
          }
        }
        
        // IS THE TERM A SIMPLE SCALAR: 2.03, 1.0E2, etc.
        if (decompose) {
          bool isnum = isScalar(functions[fiter].terms[k].expression);
          if (isnum) {
            functions[fiter].terms[k].isRoot = true;
            functions[fiter].terms[k].isAD = false;
            functions[fiter].terms[k].beenDecomposed = true;
            functions[fiter].terms[k].isScalar = true;
            functions[fiter].terms[k].isConstant = true; // means in does not need to be copied every time
            functions[fiter].terms[k].scalar_ddata = Kokkos::View<double*,AssemblyDevice>("scalar double data",1);
            functions[fiter].terms[k].scalar_ddata(0) = std::stod(functions[fiter].terms[k].expression);
            
            // Copy the data just once
            Kokkos::View<double***,AssemblyDevice> tdata("scalar data",functions[fiter].dim0,functions[fiter].dim1,1);
            functions[fiter].terms[k].ddata = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
            for (size_t k2=0; k2<functions[fiter].dim0; k2++) {
              for (size_t j2=0; j2<functions[fiter].dim1; j2++) {
                functions[fiter].terms[k].ddata(k2,j2) = functions[fiter].terms[k].scalar_ddata(0);
              }
            }
            decompose = false;
          }
        }
        
        // check if it is a discretized parameter
        if (decompose) { // TMW: NOT UPDATED FOR PARAM GRAD
          
          for (unsigned int j=0; j<disc_parameters.size(); j++) {
            if (functions[fiter].terms[k].expression == disc_parameters[j]) {
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              decompose = false;
              
              if (functions[fiter].location == "side ip") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_param_side, Kokkos::ALL(), j, Kokkos::ALL());
              }
              else if (functions[fiter].location == "point") {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_param_point, Kokkos::ALL(), j, Kokkos::ALL());
              }
              else {
                functions[fiter].terms[k].data = Kokkos::subview(wkset->local_param, Kokkos::ALL(), j, Kokkos::ALL());
              }
            }
          }
        }
        
        
        // check if it is a parameter
        if (decompose) {
          
          for (unsigned int j=0; j<parameters.size(); j++) {
            
            if (functions[fiter].terms[k].expression == parameters[j]) {
              functions[fiter].terms[k].isRoot = true;
              functions[fiter].terms[k].isAD = true;
              functions[fiter].terms[k].beenDecomposed = true;
              functions[fiter].terms[k].isScalar = true;
              functions[fiter].terms[k].isConstant = false; // needs to be copied
              functions[fiter].terms[k].scalarIndex = 0;
              
              decompose = false;
              
              functions[fiter].terms[k].scalar_data = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
              
              Kokkos::View<AD***,AssemblyDevice> tdata("scalar data",functions[fiter].dim0,functions[fiter].dim1,1);
              functions[fiter].terms[k].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
              
            }
            else { // look for param(*) or param(**)
              bool found = true;
              int sindex = 0;
              size_t nexp = functions[fiter].terms[k].expression.length();
              if (nexp == parameters[j].length()+3) {
                for (size_t n=0; n<parameters[j].length(); n++) {
                  if (functions[fiter].terms[k].expression[n] != parameters[j][n]) {
                    found = false;
                  }
                }
                if (found) {
                  if (functions[fiter].terms[k].expression[nexp-3] == '(' && functions[fiter].terms[k].expression[nexp-1] == ')') {
                    string check = "";
                    check += functions[fiter].terms[k].expression[nexp-2];
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
                  if (functions[fiter].terms[k].expression[n] != parameters[j][n]) {
                    found = false;
                  }
                }
                if (found) {
                  if (functions[fiter].terms[k].expression[nexp-4] == '(' && functions[fiter].terms[k].expression[nexp-1] == ')') {
                    string check = "";
                    check += functions[fiter].terms[k].expression[nexp-3];
                    check += functions[fiter].terms[k].expression[nexp-2];
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
                functions[fiter].terms[k].isRoot = true;
                functions[fiter].terms[k].isAD = true;
                functions[fiter].terms[k].beenDecomposed = true;
                functions[fiter].terms[k].isScalar = true;
                functions[fiter].terms[k].isConstant = false; // needs to be copied
                functions[fiter].terms[k].scalarIndex = sindex;
                
                decompose = false;
                
                functions[fiter].terms[k].scalar_data = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
                
                Kokkos::View<AD***,AssemblyDevice> tdata("scalar data",functions[fiter].dim0,functions[fiter].dim1,1);
                functions[fiter].terms[k].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
              }
            }
          }
        }
        
        // check if it is a function
        if (decompose) {
          for (unsigned int j=0; j<functions.size(); j++) {
            if (functions[fiter].terms[k].expression == functions[j].function_name &&
                functions[fiter].location == functions[j].location) {
              functions[fiter].terms[k].isFunc = true;
              functions[fiter].terms[k].isAD = functions[j].terms[0].isAD;
              functions[fiter].terms[k].funcIndex = j;
              functions[fiter].terms[k].beenDecomposed = true;
              functions[fiter].terms[k].data = functions[j].terms[0].data;
              functions[fiter].terms[k].ddata = functions[j].terms[0].ddata;
              decompose = false;
            }
          }
        }
        
        if (decompose) {
          int numterms = 0;
          numterms = split(functions[fiter].terms,k);
          functions[fiter].terms[k].beenDecomposed = true;
        }
      }
      
      bool isdone = true;
      for (size_t k=0; k<functions[fiter].terms.size(); k++) {
        if (!functions[fiter].terms[k].isRoot && !functions[fiter].terms[k].beenDecomposed) {
          isdone = false;
        }
      }
      done = isdone;
      
    }
    
    if (!done && iter >= maxiter) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MILO reached the maximum number of recursive function calls for " + functions[fiter].function_name + ".  See functionInterface.hpp to increase this");
    }
  }
  
  // After all of the functions have been decomposed, we can determine if we need to use arrays of ScalarT or AD
  // Only the roots should be designated as ScalarT or AD at this point
  
  for (size_t k=0; k<functions.size(); k++) {
    for (size_t j=0; j<functions[k].terms.size(); j++) {
      bool termcheck = this->isScalarTerm(k,j); // is this term a ScalarT
      if (termcheck) {
        functions[k].terms[j].isAD = false;
        if (!functions[k].terms[j].isRoot) {
          Kokkos::View<double***,AssemblyDevice> tdata("data",
                                                       functions[k].dim0,
                                                       functions[k].dim1,1);
          functions[k].terms[j].ddata = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
        }
        if (j==0) { // always need this allocated
          Kokkos::View<AD***,AssemblyDevice> tdata("data",
                                                   functions[k].dim0,
                                                   functions[k].dim1,1);
          functions[k].terms[j].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
        }
      }
      else if (!functions[k].terms[j].isRoot) {
        functions[k].terms[j].isAD = true;
        Kokkos::View<AD***,AssemblyDevice> tdata("data",functions[k].dim0,functions[k].dim1,1);
        functions[k].terms[j].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
      }
      //functions[k].terms[j].print();
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Determine if a term is a ScalarT or needs to be an AD type
//////////////////////////////////////////////////////////////////////////////////////

bool FunctionManager::isScalarTerm(const int & findex, const int & tindex) {
  bool is_scalar = true;
  if (functions[findex].terms[tindex].isRoot) {
    if (functions[findex].terms[tindex].isAD) {
      is_scalar = false;
    }
  }
  //else if (functions[block][findex].terms[tindex].isFunc) {
    //is_scalar = false;
  //}
  else {
    for (size_t k=0; k<functions[findex].terms[tindex].dep_list.size(); k++){
      bool depcheck = isScalarTerm(findex, functions[findex].terms[tindex].dep_list[k]);
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

FDATA FunctionManager::evaluate(const string & fname, const string & location) {
  Teuchos::TimeMonitor ttimer(*evaluateTimer);
  
  
  int findex = -1;
  for (size_t i=0; i<functions.size(); i++) {
    if (fname == functions[i].function_name && functions[i].location == location) {
      evaluate(i,0);
      findex = i;
    }
  }
  
  if (findex == -1) { // meaning that the requested function was not registered at this location
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: function manager could not evaluate: " + fname + " at " + location);
  }
  
  FDATA output = functions[findex].terms[0].data;
  if (!functions[findex].terms[0].isAD) {
    FDATAd doutput = functions[findex].terms[0].ddata;
    parallel_for(RangePolicy<AssemblyExec>(0,output.extent(0)), KOKKOS_LAMBDA (const int e ) {
      for (unsigned int n=0; n<output.extent(1); n++) {
        output(e,n) = doutput(e,n);
      }
    });
  }
  return output;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function
//////////////////////////////////////////////////////////////////////////////////////

void FunctionManager::evaluate( const size_t & findex, const size_t & tindex) {
  
  //if (verbosity > 10) {
  //  cout << "------- Evaluating: " << functions[findex].terms[tindex].expression << endl;
  //}
  
  //functions[block][findex].terms[tindex].print();
  
  if (functions[findex].terms[tindex].isRoot) {
    if (functions[findex].terms[tindex].isScalar && !functions[findex].terms[tindex].isConstant) {
      if (functions[findex].terms[tindex].isAD) {
        FDATA data0 = functions[findex].terms[tindex].data;
        Kokkos::View<AD*,Kokkos::LayoutStride,AssemblyDevice> data1 = functions[findex].terms[tindex].scalar_data;
        parallel_for(RangePolicy<AssemblyExec>(0,data0.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (unsigned int n=0; n<data0.extent(1); n++) {
            data0(e,n) = data1(0);
          }
        });
      }
      else {
        FDATAd data0 = functions[findex].terms[tindex].ddata;
        Kokkos::View<double*,Kokkos::LayoutStride,AssemblyDevice> data1 = functions[findex].terms[tindex].scalar_ddata;
        parallel_for(RangePolicy<AssemblyExec>(0,data0.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (unsigned int n=0; n<data0.extent(1); n++) {
            data0(e,n) = data1(0);
          }
        });
      }
    }
  }
  else if (functions[findex].terms[tindex].isFunc) {
    int funcIndex = functions[findex].terms[tindex].funcIndex;
    this->evaluate(funcIndex, 0);
    if (functions[findex].terms[tindex].isAD) {
      if (functions[funcIndex].terms[0].isAD) {
        functions[findex].terms[tindex].data = functions[funcIndex].terms[0].data;
      }
      else {
        FDATA data0 = functions[findex].terms[tindex].data;
        FDATAd data1 = functions[funcIndex].terms[0].ddata;
        parallel_for(RangePolicy<AssemblyExec>(0,data0.extent(0)), KOKKOS_LAMBDA (const int e ) {
          for (unsigned int n=0; n<data0.extent(1); n++) {
            data0(e,n) = data1(e,n);
          }
        });
      }
    }
    else {
      functions[findex].terms[tindex].ddata = functions[funcIndex].terms[0].ddata;
    }
  }
  else {
    bool isAD = functions[findex].terms[tindex].isAD;
    for (size_t k=0; k<functions[findex].terms[tindex].dep_list.size(); k++) {
      
      int dep = functions[findex].terms[tindex].dep_list[k];
      this->evaluate(findex, dep);
      
      bool termisAD = functions[findex].terms[dep].isAD;
      if (isAD) {
        if (termisAD) {
          this->evaluateOp(functions[findex].terms[tindex].data,
                           functions[findex].terms[dep].data,
                           functions[findex].terms[tindex].dep_ops[k]);
          
        }
        else {
          this->evaluateOp(functions[findex].terms[tindex].data,
                           functions[findex].terms[dep].ddata,
                           functions[findex].terms[tindex].dep_ops[k]);
        }
      }
      else { // termisAD must also be false
        this->evaluateOp(functions[findex].terms[tindex].ddata,
                         functions[findex].terms[dep].ddata,
                         functions[findex].terms[tindex].dep_ops[k]);
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
  //size_t dim1 = std::min(data.extent(1),tdata.extent(1));
  
  if (op == "") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) = tdata(e,n);
      }
    });
  }
  else if (op == "plus") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) += tdata(e,n);
      }
    });
  }
  else if (op == "minus") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) += -tdata(e,n);
      }
    });
  }
  else if (op == "times") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) *= tdata(e,n);
      }
    });
  }
  else if (op == "divide") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) /= tdata(e,n);
      }
    });
  }
  else if (op == "power") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) = pow(data(e,n),tdata(e,n));
      }
    });
  }
  else if (op == "sin") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) = sin(tdata(e,n));
      }
    });
  }
  else if (op == "cos") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) = cos(tdata(e,n));
      }
    });
  }
  else if (op == "tan") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) = tan(tdata(e,n));
      }
    });
  }
  else if (op == "exp") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) = exp(tdata(e,n));
      }
    });
  }
  else if (op == "log") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        data(e,n) = log(tdata(e,n));
      }
    });
  }
  else if (op == "abs") {
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        if (tdata(e,n) < 0.0) {
          data(e,n) = -tdata(e,n);
        }
        else {
          data(e,n) = tdata(e,n);
        }
      }
    });
  }
  else if (op == "max") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
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
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
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
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
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
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        if (data(e,n) < tdata(e,n)) {
          data(e,n) = 1.0;
        }
        else {
          data(e,n) = 0.0;
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
    parallel_for(RangePolicy<AssemblyExec>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      size_t dim1 = min(data.extent(1),tdata.extent(1));
      for (unsigned int n=0; n<dim1; n++) {
        if (data(e,n) > tdata(e,n)) {
          data(e,n) = 1.0;
        }
        else {
          data(e,n) = 0.0;
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
  /*
  for (size_t b=0; b<functions.size(); b++) {
    cout << "Block Number: " << b << endl;
    for (size_t n=0; n<functions[b].size(); n++) {
      cout << "Function Name:" << functions[b][n].function_name << endl;
      cout << "Location: " << functions[b][n].location << endl << endl;
      cout << "Terms: " << endl;
      for (size_t t=0; t<functions[b][n].terms.size(); t++) {
        cout << "    " << functions[b][n].terms[t].expression << endl;
      }
      cout << endl;
      cout << "First term information:" << endl;
      functions[b][n].terms[0].print();
      cout << endl << endl;
    }
  }
  */
}


