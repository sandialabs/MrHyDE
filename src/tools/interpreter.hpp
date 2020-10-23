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

#include "trilinos.hpp"
#include "preferences.hpp"
#include "term.hpp"

#include <stdio.h>
#include <ctype.h>

namespace MrHyDE {
  
  class Interpreter {
  public:
    
    Interpreter() {};
    
    ~Interpreter() {};
    
    //////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////
    
    vector<string> getVars(const string & s, const vector<string> & knownops) {
      vector<string> vars;
      bool interm = false;
      string var;
      for (size_t i=0; i<s.length(); i++) {
        if (isalpha(s[i]) || s[i] == '_' || s[i] == '[' || s[i] == ']') {
          if (!interm) {
            if (s[i] == 'e' || s[i] == 'E') {
              if (i>0 && i<s.length()-1) {
                if (isdigit(s[i-1])) {
                  // ex.: 1.0e2 (not a variable)
                }
                else {
                  interm = true;
                  var = s[i];
                }
              }
              else {
                interm = true;
                var = s[i];
              }
            }
            else {
              interm = true;
              var = s[i];
            }
          }
          else {
            var += s[i];
          }
          if (i == (s.length()-1)) {
            bool isknown = false;
            for (size_t j=0; j<knownops.size(); j++) {
              if (var == knownops[j]) {
                isknown = true;
              }
            }
            if (!isknown) {
              vars.push_back(var);
            }
          }
        }
        else {
          if (interm) {
            bool isknown = false;
            for (size_t j=0; j<knownops.size(); j++) {
              if (var == knownops[j]) {
                isknown = true;
              }
            }
            if (!isknown) {
              vars.push_back(var);
            }
            interm = false;
          }
        }
      }
      
      return vars;
      
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    int validateTerms(const vector<string> & terms,
                             const vector<string> & known_vars,
                             const vector<string> & variables,
                             const vector<string> & parameters,
                             const vector<string> & disc_parameters,
                             const vector<string> & functions) {
      
      int failures = 0;
      
      for (size_t k=0; k<terms.size(); k++) {
        bool found = false;
        for (size_t j=0; j<known_vars.size(); j++) {
          if (terms[k] == known_vars[j]) {
            found = true;
          }
        }
        if (!found) {
          for (size_t j=0; j<variables.size(); j++) {
            if (terms[k] == variables[j]) {
              found = true;
            }
            else if (terms[k] == (variables[j]+"_x")) {
              found = true;
            }
            else if (terms[k] == (variables[j]+"_y")) {
              found = true;
            }
            else if (terms[k] == (variables[j]+"_z")) {
              found = true;
            }
            else if (terms[k] == (variables[j]+"_t")) {
              found = true;
            }
            else if (terms[k] == (variables[j]+"[x]")) {
              found = true;
            }
            else if (terms[k] == (variables[j]+"[y]")) {
              found = true;
            }
            else if (terms[k] == (variables[j]+"[z]")) {
              found = true;
            }
          }
        }
        if (!found) {
          for (size_t j=0; j<parameters.size(); j++) {
            if (terms[k] == parameters[j]) {
              found = true;
            }
          }
        }
        if (!found) {
          for (size_t j=0; j<disc_parameters.size(); j++) {
            if (terms[k] == disc_parameters[j]) {
              found = true;
            }
          }
        }
        if (!found) {
          for (size_t j=0; j<functions.size(); j++) {
            if (terms[k] == functions[j]) {
              found = true;
            }
          }
        }
        
        if (!found) {
          cout << "Error: MILO could not identify the following term: " << terms[k] << endl;
          cout << "The options are: " << "t, " << "x, " << "y, " << "z" << endl;
          for (size_t j=0; j<variables.size(); j++) {
            cout << "                 " << variables[j] << ", " << variables[j]+"_x, " << variables[j]+"_y, " << variables[j]+"_z, " << variables[j]+"_t" << endl;
          }
          for (size_t j=0; j<parameters.size(); j++) {
            cout << "                 " << parameters[j] << ", ";
          }
          cout << endl;
          for (size_t j=0; j<functions.size(); j++) {
            cout << "                 " << functions[j] << ", ";
          }
          cout << endl;
          
          failures += 1;
        }
      }
      
      return failures;
      
    }
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    bool isScalar(const string & s) {
      
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
    
    void split(vector<term> & terms, const size_t & index) {
      
      string s = terms[index].expression;
      
      if (s.length() == 0) {
        //return 0;
      }
      else if (s.length() == 1) { // TMW: why is this case needed?
        string currterm = "";
        string currop = "";
        currterm += s[0];
        bool found = false;
        for (size_t k=0; k<terms.size(); k++) {
          if (terms[k].expression == currterm) {
            found = true;
            terms[index].dep_list.push_back(k);
            terms[index].dep_ops.push_back(currop);
          }
        }
        if (!found) {
          term nterm = term(currterm);
          terms.push_back(nterm);
          terms[index].dep_list.push_back(terms.size()-1);
          terms[index].dep_ops.push_back(currop);
        }
        //return 1;
      }
      else {
        size_t num_pm = 0; // +,-
        size_t num_mdp = 0; // *,/,^,<,>,<=,>=
        
        string currterm = "";
        string currop = "";
        int paren = 0;
        
        for (size_t i=0; i<s.length(); i++) {
          if (s[i] == '('){
            paren += 1;
          }
          else if (s[i] == ')') {
            paren += -1;
          }
          else if (paren == 0 && i>0) {
            if (s[i] == '+' || s[i] == '-') {
              num_pm += 1;
            }
            if (s[i] == '*' || s[i] == '/' || s[i] == '^'
                || s[i] == '<' || s[i] == '>') {
              // || s[i] == '<=' // TMW: this might fail - don't use <= or >=
              // || s[i] == '>=') {
              num_mdp += 1;
            }
          }
        }
        paren = 0;
        
        if (num_pm > 0) {
          
          for (size_t i=0; i<s.length(); i++) {
            if (s[i] == ' ') {
              // do nothing ... skip spaces
            }
            else if (s[i] == '('){
              paren += 1;
              currterm += s[i];
            }
            else if (s[i] == ')'){
              paren += -1;
              currterm += s[i];
            }
            else if (paren == 0 && s[i] == '+' && currterm.length() > 0){
              bool found = false;
              for (size_t k=0; k<terms.size(); k++) {
                if (terms[k].expression == currterm) {
                  found = true;
                  terms[index].dep_list.push_back(k);
                  terms[index].dep_ops.push_back(currop);
                }
              }
              if (!found) {
                term nterm = term(currterm);
                terms.push_back(nterm);
                terms[index].dep_list.push_back(terms.size()-1);
                terms[index].dep_ops.push_back(currop);
              }
              currterm = "";
              currop = "plus";
            }
            else if (paren == 0 && s[i] == '-' && currterm.length() > 0) {
              bool found = false;
              for (size_t k=0; k<terms.size(); k++) {
                if (terms[k].expression == currterm) {
                  found = true;
                  terms[index].dep_list.push_back(k);
                  terms[index].dep_ops.push_back(currop);
                }
              }
              if (!found) {
                term nterm = term(currterm);
                terms.push_back(nterm);
                terms[index].dep_list.push_back(terms.size()-1);
                terms[index].dep_ops.push_back(currop);
              }
              currterm = "";
              currop = "minus";
            }
            else {
              currterm += s[i];
            }
            
            if (i == s.length()-1 && currterm.length()>0) {
              if (paren>0) {
                // add error
              }
              else if (paren < 0) {
                // add error
              }
              else if (num_pm>0) {
                bool found = false;
                for (size_t k=0; k<terms.size(); k++) {
                  if (terms[k].expression == currterm) {
                    found = true;
                    terms[index].dep_list.push_back(k);
                    terms[index].dep_ops.push_back(currop);
                  }
                }
                if (!found) {
                  term nterm = term(currterm);
                  terms.push_back(nterm);
                  terms[index].dep_list.push_back(terms.size()-1);
                  terms[index].dep_ops.push_back(currop);
                }
              }
            }
            
          }
          //return num_pm+1;
        }
        else if (num_mdp > 0) {
          string currterm = "";
          string currop = "";
          for (size_t i=0; i<s.length(); i++) {
            if (s[i] == '('){
              paren += 1;
              currterm += s[i];
            }
            else if (s[i] == ')'){
              paren += -1;
              currterm += s[i];
            }
            else if (paren == 0 && s[i] == '*'){
              bool found = false;
              for (size_t k=0; k<terms.size(); k++) {
                if (terms[k].expression == currterm) {
                  found = true;
                  terms[index].dep_list.push_back(k);
                  terms[index].dep_ops.push_back(currop);
                }
              }
              if (!found) {
                term nterm = term(currterm);
                terms.push_back(nterm);
                terms[index].dep_list.push_back(terms.size()-1);
                terms[index].dep_ops.push_back(currop);
              }
              currterm = "";
              currop = "times";
            }
            else if (paren == 0 && s[i] == '/') {
              bool found = false;
              for (size_t k=0; k<terms.size(); k++) {
                if (terms[k].expression == currterm) {
                  found = true;
                  terms[index].dep_list.push_back(k);
                  terms[index].dep_ops.push_back(currop);
                }
              }
              if (!found) {
                term nterm = term(currterm);
                terms.push_back(nterm);
                terms[index].dep_list.push_back(terms.size()-1);
                terms[index].dep_ops.push_back(currop);
              }
              currterm = "";
              currop = "divide";
            }
            else if ( paren == 0 && s[i] == '^') {
              bool found = false;
              for (size_t k=0; k<terms.size(); k++) {
                if (terms[k].expression == currterm) {
                  found = true;
                  terms[index].dep_list.push_back(k);
                  terms[index].dep_ops.push_back(currop);
                }
              }
              if (!found) {
                term nterm = term(currterm);
                terms.push_back(nterm);
                terms[index].dep_list.push_back(terms.size()-1);
                terms[index].dep_ops.push_back(currop);
              }
              currterm = "";
              currop = "power";
            }
            else if ( paren == 0 && s[i] == '<') {
              bool found = false;
              for (size_t k=0; k<terms.size(); k++) {
                if (terms[k].expression == currterm) {
                  found = true;
                  terms[index].dep_list.push_back(k);
                  terms[index].dep_ops.push_back(currop);
                }
              }
              if (!found) {
                term nterm = term(currterm);
                terms.push_back(nterm);
                terms[index].dep_list.push_back(terms.size()-1);
                terms[index].dep_ops.push_back(currop);
              }
              currterm = "";
              currop = "lt";
            }
            else if ( paren == 0 && s[i] == '>') {
              bool found = false;
              for (size_t k=0; k<terms.size(); k++) {
                if (terms[k].expression == currterm) {
                  found = true;
                  terms[index].dep_list.push_back(k);
                  terms[index].dep_ops.push_back(currop);
                }
              }
              if (!found) {
                term nterm = term(currterm);
                terms.push_back(nterm);
                terms[index].dep_list.push_back(terms.size()-1);
                terms[index].dep_ops.push_back(currop);
              }
              currterm = "";
              currop = "gt";
            }
            else {
              currterm += s[i];
            }
            
            if (i == s.length()-1 && num_mdp > 0) {
              bool found = false;
              for (size_t k=0; k<terms.size(); k++) {
                if (terms[k].expression == currterm) {
                  found = true;
                  terms[index].dep_list.push_back(k);
                  terms[index].dep_ops.push_back(currop);
                }
              }
              if (!found) {
                term nterm = term(currterm);
                terms.push_back(nterm);
                terms[index].dep_list.push_back(terms.size()-1);
                terms[index].dep_ops.push_back(currop);
              }
            }
            
          }
          //return num_mdp+1;
        }
        else {
          if (s[0] == '(' && s[s.length()-1] == ')') {
            string currterm;
            for (size_t k=1; k<s.length()-1; k++) {
              currterm += s[k];
            }
            term nterm = term(currterm);
            terms.push_back(nterm);
            terms[index].dep_list.push_back(terms.size()-1);
            terms[index].dep_ops.push_back(currop);
          }
          //return 1;
        }
        
      }
    }
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    
    bool isOperator(vector<term> & terms, size_t & index, vector<string> & ops) {
      // checks if the string in s can be written as a standard operator on a term
      // example: sin(whatever)
      
      string s = terms[index].expression;
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
              int paren = 1;
              for (size_t j=L+1; j<s.length()-1; j++) {
                if (s[j] == '(') {
                  paren += 1;
                }
                if (s[j] == ')') {
                  paren += -1;
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
        term nterm = term(argument);
        terms.push_back(nterm);
        terms[index].dep_list.push_back(terms.size()-1);
        terms[index].dep_ops.push_back(oper);
        terms[index].beenDecomposed = true;
      }
      
      
      return found;
    }
    
  };
  
}
