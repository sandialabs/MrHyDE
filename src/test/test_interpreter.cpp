#include "interpreter.hpp"
#include "term.hpp"
#include "kokkosTools.hpp"

using namespace std;

int main(int argc, char * argv[]) {

  /*
  string s = argv[1];
  
  // Test if the integrater extracts the variables correctly
  vector<string> vars = getVars(s);
  
  for (size_t i=0; i<vars.size(); i++) {
    cout << " " << vars[i];
  }
  cout << endl;
  
  // Test if the interpreter validates the variables correctly
  vector<string> variables = {"a","b","c","d"};
  vector<string> parameters = {"mu"};
  vector<string> disc_parameters = {"ff"};
  vector<string> functions = {"g"};
  vector<string> opers = {"sin", "cos","exp","log","tan"};
  
  int numfails = validateTerms(vars,variables,parameters,disc_parameters,functions);
  
  // Test if we can decompose the string correctly
  vector<term> terms;
  term newt = term(s);
  terms.push_back(newt);
  
  bool done = false;
  int iter = 0;
  while (!done && iter < 7) {
    
    iter++;
    
    size_t Nterms = terms.size();
    
    for (size_t k=0; k<Nterms; k++) {
      
      bool decompose = true;
      if (terms[k].isRoot || terms[k].beenDecomposed) {
        decompose = false;
      }
      
      if (decompose) {
        bool isop = isOperator(terms, k, opers);
        if (isop) {
          decompose = false;
        }
      }
      
      if (decompose) {
        
        bool isnum = isScalar(terms[k].expression);
        cout << terms[k].expression << " " << isnum << endl;
        
        if (isnum) {
          terms[k].isRoot = false;
          terms[k].beenDecomposed = true;
          terms[k].isScalar = true;
          terms[k].scalar_ddata = std::stod(terms[k].expression);
          cout << stod(terms[k].expression) << endl;
          
        }
        else {
          // check if it is a variable
          bool isvar = false;
          for (int j=0; j<variables.size(); j++) {
            if (terms[k].expression == variables[j]) {
              isvar = true;
              terms[k].isRoot = true;
              terms[k].beenDecomposed = true;
            }
          }
          // check if it is a parameter
          bool isparam = false;
          for (int j=0; j<parameters.size(); j++) {
            if (terms[k].expression == parameters[j]) {
              isparam = true;
              terms[k].isRoot = true;
              terms[k].beenDecomposed = true;
            }
          }
          
          // check if it is a discretized parameter
          bool isdparam = false;
          
          // check if it is a function
          bool isfunc = false;
          for (int j=0; j<functions.size(); j++) {
            if (terms[k].expression == functions[j]) {
              isfunc = true;
              terms[k].isFunc = true;
              terms[k].beenDecomposed = true;
            }
          }
          
          int numterms = 0;
          if (!isvar && !isparam && !isdparam && !isfunc) {
            numterms = split(terms,k);
            terms[k].beenDecomposed = true;
            if (numterms == 1) { // e.g. s = sin(a+b)
              // may need to modify the logic here to avoid redundant calc.
            }
          }
          
          
        }
      }
    }
    
    bool isdone = true;
    for (size_t k=0; k<terms.size(); k++) {
      if (!terms[k].isRoot && !terms[k].beenDecomposed) {
        isdone = false;
      }
    }
    done = isdone;
  }
  
  cout << "numterms = " << terms.size() << endl;
  */
  // Test if we can evaluate the expression
  /*
  int dim0 = 6;
  int dim1 = 2;
  
  size_t numvars = variables.size();
  Kokkos::View<double***,AssemblyDevice> vardata("variable data for testing",dim0,dim1,numvars);
  for (size_t k=0; k<dim0; k++) {
    for (size_t j=0; j<dim1; j++) {
      for (size_t n=0; n<numvars; n++) {
        vardata(k,j,n) = n+2.0;
      }
    }
  }
  
  for (size_t k=0; k<terms.size(); k++) {
    
    if (terms[k].isRoot) {
      for (size_t j=0; j<numvars; j++) {
        if (terms[k].expression == variables[j]) {
          terms[k].data = Kokkos::subview(vardata, Kokkos::ALL(), Kokkos::ALL(), j);
        }
      }
    }
    else if (terms[k].isScalar) {
      Kokkos::View<double***,AssemblyDevice> tdata("scalar data for testing",dim0,dim1,1);
      terms[k].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
      for (size_t k2=0; k2<dim0; k2++) {
        for (size_t j2=0; j2<dim1; j2++) {
          terms[k].data(k2,j2) = terms[k].scalar_value;
        }
      }
      
    }
    else {
      Kokkos::View<double***,AssemblyDevice> tdata("data for testing",dim0,dim1,1);
      terms[k].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
    }
  }
  
  int index = 0;
  Kokkos::View<double**,Kokkos::LayoutStride,AssemblyDevice> result = evaluate(terms, index);
  
  cout << sin(exp(2+3)) << endl;
  
  for (size_t k=0; k<dim0; k++) {
    for (size_t j=0; j<dim1; j++) {
      cout << k << " " << j << " " << terms[0].data(k,j) << endl;
    }
  }
   */
  /*
  bool isnum = isScalar(s);
  cout << " numterms = " << numterms << endl;
  
  for (int j=0; j<numterms; j++) {
    // check if a term is one of the variables
    bool isvar = false;
    for (int k=0; k<variables.size(); k++) {
      if (terms[j] == variables[k]) {
        isvar = true;
      }
    }
    // could do the same for parameters and functions
    
    // check if term is a standard operator: sin(), cos(), etc.
    string oper, argu;
    bool isop = isOperator(terms[j], opers, oper, argu);
    cout << terms[j] << "  " << isop << "  " << oper << "  " << argu << endl;
    
    if (!isvar && isop) {
      terms.push_back(argu);
      ops.push_back(oper);
      int numtt = split(argu, terms, ops);
    }
    else if (!isvar) {
      int nts = split(terms[j],terms,ops);
    }
    //cout << terms[j] << "  " << terms.size() << endl;
    
  }
  
  for (int j=0; j<terms.size(); j++) {
    cout << j << "  " << terms[j] << endl;
  }
  
  for (int j=0; j<ops.size(); j++) {
    cout << j << "  " << ops[j] << endl;
  }
  */
  return 0;
}


