/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "interpreter.hpp"

using namespace MrHyDE;

//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
bool Interpreter<EvalT>::isScalar(const string & s) {
  
  bool isnum = true;
  int numdots = 0;
  int nume = 0;
  for (size_t k=0; k<s.length(); k++) {
    if (!isdigit(s[k])) { // might still be a number
      if (s[k] == '.') { // one of these is ok
        if (numdots == 0) {
          numdots += 1;
        }
        else { // 2 or more is not
          isnum = false;
        }
      }
      else if (s[k] == 'e' || s[k] == 'E') {
        if (nume == 0) {
          nume += 1;
        }
        else { // 2 or more is not
          isnum = false;
        }
      }
      else if (s[k] == '+' || s[k] == '-') { // these are ok if first or after 'e' or 'E'
        if (k>0) {
          if (s[k-1] == 'e' || s[k-1] == 'E') {
            // this is ok
          }
          else {
            isnum = false;
          }
        }
      }
      else {
        isnum = false;
      }
    }
    
  }
  return isnum;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
void Interpreter<EvalT>::split(vector<Branch<EvalT> > & branches, const size_t & index) {
  
  string s = branches[index].expression_;
  
  // If the first character is a minus sign, add 0.0 to the string to get a proper tree
  if (s[0] == '-') {
    s = "0.0" + s;
  }
  
  if (s.length() == 0) {
    // do nothing - why are we even in here?
  }
  else if (s.length() == 1) { // TMW: why is this case needed?
    string currbranch = "";
    string currop = "";
    currbranch += s[0];
    bool found = false;
    if (!found) {
      auto nbranch = Branch<EvalT>(currbranch);
      branches.push_back(nbranch);
      branches[index].dep_list_.push_back(branches.size()-1);
      branches[index].dep_ops_.push_back(currop);
    }
  }
  else {
    size_t num_pm = 0; // +,-
    size_t num_mdp = 0; // *,/,<,>,<=,>=
    size_t num_pow = 0; // ^
    
    string currbranch = "";
    string currop = "";
    int paren = 0;
    
    for (size_t i=0; i<s.length(); i++) {
      if (s[i] == '('){
        paren += 1;
      }
      else if (s[i] == ')') {
        paren += -1;
      }
      else if (paren == 0) {//} && i>0) {
        if (s[i] == '+' || s[i] == '-') {
          num_pm += 1;
        }
        if (s[i] == '*' || s[i] == '/'
            || s[i] == '<' || s[i] == '>') {
          // || s[i] == '<=' // TMW: this might fail - don't use <= or >=
          // || s[i] == '>=') {
          num_mdp += 1;
        }
        if (s[i] == '^') {
          num_pow += 1;
        }
      }
    }
    paren = 0;
    
    if (num_pm > 0) {
      
      for (size_t i=0; i<s.length(); i++) {
        
        if (s[i] == ' ') {
          // do nothing ... skip spaces
        }
        else if (s[i] == '=') {
          // do nothing ... skip equals (from lte, gte)
        }
        else if (s[i] == '('){
          paren += 1;
          currbranch += s[i];
        }
        else if (s[i] == ')'){
          paren += -1;
          currbranch += s[i];
        }
        else if (paren == 0 && s[i] == '+' && currbranch.length() > 0){
          bool found = false;
          if (!found) {
            auto nbranch = Branch<EvalT>(currbranch);
            branches.push_back(nbranch);
            branches[index].dep_list_.push_back(branches.size()-1);
            branches[index].dep_ops_.push_back(currop);
          }
          currbranch = "";
          currop = "plus";
        }
        else if (paren == 0 && s[i] == '-' && currbranch.length() > 0) {
          bool found = false;
          if (!found) {
            auto nbranch = Branch<EvalT>(currbranch);
            branches.push_back(nbranch);
            branches[index].dep_list_.push_back(branches.size()-1);
            branches[index].dep_ops_.push_back(currop);
          }
          currbranch = "";
          currop = "minus";
        }
        else {
          currbranch += s[i];
        }
        
        if (i == s.length()-1 && currbranch.length()>0) {
          if (num_pm>0) {
            bool found = false;
            if (!found) {
              auto nbranch = Branch<EvalT>(currbranch);
              branches.push_back(nbranch);
              branches[index].dep_list_.push_back(branches.size()-1);
              branches[index].dep_ops_.push_back(currop);
            }
          }
        }
        
      }
    }
    else if (num_mdp > 0) {
      string currbranch = "";
      string currop = "";
      for (size_t i=0; i<s.length(); i++) {
        if (s[i] == ' ') {
          // do nothing
        }
        else if (s[i] == '('){
          paren += 1;
          currbranch += s[i];
        }
        else if (s[i] == ')'){
          paren += -1;
          currbranch += s[i];
        }
        else if (paren == 0 && s[i] == '*'){
          bool found = false;
          if (!found) {
            auto nbranch = Branch<EvalT>(currbranch);
            branches.push_back(nbranch);
            branches[index].dep_list_.push_back(branches.size()-1);
            branches[index].dep_ops_.push_back(currop);
          }
          currbranch = "";
          currop = "times";
        }
        else if (paren == 0 && s[i] == '/') {
          bool found = false;
          if (!found) {
            auto nbranch = Branch<EvalT>(currbranch);
            branches.push_back(nbranch);
            branches[index].dep_list_.push_back(branches.size()-1);
            branches[index].dep_ops_.push_back(currop);
          }
          currbranch = "";
          currop = "divide";
        }
        else if ( paren == 0 && s[i] == '<') {
          bool found = false;
          if (!found) {
            auto nbranch = Branch<EvalT>(currbranch);
            branches.push_back(nbranch);
            branches[index].dep_list_.push_back(branches.size()-1);
            branches[index].dep_ops_.push_back(currop);
          }
          currbranch = "";
          if (s[i+1] == '=') {
            currop = "lte";
            ++i;
          }
          else {
            currop = "lt";
          }
        }
        else if ( paren == 0 && s[i] == '>') {
          bool found = false;
          if (!found) {
            auto nbranch = Branch<EvalT>(currbranch);
            branches.push_back(nbranch);
            branches[index].dep_list_.push_back(branches.size()-1);
            branches[index].dep_ops_.push_back(currop);
          }
          currbranch = "";
          
          if (s[i+1] == '=') {
            currop = "gte";
            ++i;
          }
          else {
            currop = "gt";
          }
        }
        else {
          currbranch += s[i];
        }
        
        if (i == s.length()-1 && num_mdp > 0) {
          bool found = false;
          if (!found) {
            auto nbranch = Branch<EvalT>(currbranch);
            branches.push_back(nbranch);
            branches[index].dep_list_.push_back(branches.size()-1);
            branches[index].dep_ops_.push_back(currop);
          }
        } 
      }
      
    }
    else if (num_pow > 0) {
      string currbranch = "";
      string currop = "";
      for (size_t i=0; i<s.length(); i++) {
        if (s[i] == '('){
          paren += 1;
          currbranch += s[i];
        }
        else if (s[i] == ')'){
          paren += -1;
          currbranch += s[i];
        }
        else if ( paren == 0 && s[i] == '^') {
          bool found = false;
          if (!found) {
            auto nbranch = Branch<EvalT>(currbranch);
            branches.push_back(nbranch);
            branches[index].dep_list_.push_back(branches.size()-1);
            branches[index].dep_ops_.push_back(currop);
          }
          currbranch = "";
          currop = "power";
        }
        else {
          currbranch += s[i];
        }
        
        if (i == s.length()-1 && num_pow > 0) {
          bool found = false;
          if (!found) {
            auto nbranch = Branch<EvalT>(currbranch);
            branches.push_back(nbranch);
            branches[index].dep_list_.push_back(branches.size()-1);
            branches[index].dep_ops_.push_back(currop);
          }
        }
      }
    }
    else {
      if (s[0] == '(' && s[s.length()-1] == ')') {
        string currbranch;
        for (size_t k=1; k<s.length()-1; k++) {
          currbranch += s[k];
        }
        auto nbranch = Branch<EvalT>(currbranch);
        branches.push_back(nbranch);
        branches[index].dep_list_.push_back(branches.size()-1);
        branches[index].dep_ops_.push_back(currop);
      }
      else {
        bool foundparen = false;
        size_t pindex = 0;
        for (size_t k=1; k<s.length()-1; k++) {
          if (s[k] == '(' && !foundparen) {
            foundparen = true;
            pindex = k;
          }
        }
        if (foundparen && s[s.length()-1] == ')') {
          string currbranch;
          for (size_t k=pindex+1; k<s.length()-1; k++) {
            currbranch += s[k];
          }
          auto nbranch = Branch<EvalT>(currbranch);
          branches.push_back(nbranch);
          branches[index].dep_list_.push_back(branches.size()-1);
          currop = "";
          for (size_t k=0; k<pindex; ++k) {
            currop += s[k];
          }
          branches[index].dep_ops_.push_back(currop);
          currop = "";
        }
      }
    }
    if (paren > 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE found an unclosed parenthesis in: " + s);
    }
    else if (paren < 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MrHyDE found an extra parenthesis in: " + s);
    }
    
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////////////////////////////////

template<class EvalT>
bool Interpreter<EvalT>::isOperator(vector<Branch<EvalT> > & branches, size_t & index, vector<string> & ops) {
  // checks if the string in s can be written as a standard operator on a branch
  // example: sin(whatever)
  
  string s = branches[index].expression_;
  string oper, argument;
  bool found = false;
  size_t k=0;
  while (!found && k<ops.size()) {
    size_t L = ops[k].length();
    if (s.length() >= L) {
      bool iseq = true;
      for (size_t j=0; j<L; j++) {
        if (s[j] != ops[k][j]) {
          iseq = false;
        }
      }
      if (iseq) {
        if (s[L] != '(' || s[s.length()-1] != ')') {
          iseq = false;
        }
        if (s[L] == '(') {
          for (size_t j=L+1; j<s.length()-1; j++) {
            if (s[j] == ')') {
              if (j<s.length()-1) {
                iseq = false;
              }
            }
          }
        }
        
      }
      if (iseq) {
        found = true;
        oper = ops[k];
        argument = "";
        for (size_t i=L+1; i<s.length()-1; i++) {
          argument += s[i];
        }
      }
    }
    k += 1;
  }
  
  if (found) {
    bool has_comma = false;
    size_t comma_ind = 0;
    for (size_t i=0; i<argument.length()-1; ++i) {
      if (argument[i] == ',') {
        has_comma = true;
        comma_ind = i;
      }
    }
    if (has_comma) {
      string argument1 = "";
      for (size_t i=0; i<comma_ind; i++) {
        argument1 += argument[i];
      }
      string argument2 = "";
      for (size_t i=comma_ind+1; i<argument.length(); i++) {
        argument2 += argument[i];
      }
      
      auto nbranch1 = Branch<EvalT>(argument1);
      branches.push_back(nbranch1);
      branches[index].dep_list_.push_back(branches.size()-1);
      branches[index].dep_ops_.push_back("");
      
      auto nbranch2 = Branch<EvalT>(argument2);
      branches.push_back(nbranch2);
      branches[index].dep_list_.push_back(branches.size()-1);
      branches[index].dep_ops_.push_back(oper);
      branches[index].is_decomposed_ = true;
    }
    else {
      auto nbranch = Branch<EvalT>(argument);
      branches.push_back(nbranch);
      branches[index].dep_list_.push_back(branches.size()-1);
      branches[index].dep_ops_.push_back(oper);
      branches[index].is_decomposed_ = true;
    }
  }
  
  
  return found;
}


//////////////////////////////////////////////////////////////
// Explicit template instantiations
//////////////////////////////////////////////////////////////

template class MrHyDE::Interpreter<ScalarT>;

#ifndef MrHyDE_NO_AD
// Custom AD type
template class MrHyDE::Interpreter<AD>;

// Standard built-in types
template class MrHyDE::Interpreter<AD2>;
template class MrHyDE::Interpreter<AD4>;
template class MrHyDE::Interpreter<AD8>;
template class MrHyDE::Interpreter<AD16>;
template class MrHyDE::Interpreter<AD18>;
template class MrHyDE::Interpreter<AD24>;
template class MrHyDE::Interpreter<AD32>;
#endif
