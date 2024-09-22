/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include "trilinos.hpp"
#include "preferences.hpp"
#include "functionManager.hpp"
#include "workset.hpp"

#include <Kokkos_Random.hpp>

using namespace std;
using namespace MrHyDE;

int main(int argc, char * argv[]) {
  
  #ifndef MrHyDE_NO_AD
    typedef Kokkos::View<AD**,ContLayout,AssemblyDevice> View_AD2;
    typedef Kokkos::View<AD****,ContLayout,AssemblyDevice> View_AD4;
  #else
    typedef View_Sc2 View_AD2;
    typedef View_Sc4 View_AD4;
  #endif
    
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  {
    //----------------------------------------------------------------------
    // Setup
    //----------------------------------------------------------------------
    
    // Set up a dummy workset just to test interplay between function manager and workset
    Teuchos::RCP<Workset<AD> > wkset = Teuchos::rcp( new Workset<AD>() );
    
    // Define some parameters
    int numElem = 10;
    vector<string> variables = {"a","b","c","d","p","Ha"};
    vector<string> scalars = {"x","y","z"};
    int numip = 4;
    vector<string> btypes = {"HGRAD"};
    vector<int> usebasis = {0,0,0,0,0,0};

    // Set necessary parameters in workset
    wkset->usebasis = usebasis;
    wkset->set_usebasis.push_back(usebasis);
    wkset->set_varlist.push_back(variables);
    wkset->maxElem = numElem;
    wkset->numSets = 1;
    wkset->numip = numip;
    wkset->isInitialized = true;
    wkset->addSolutionFields(variables, btypes, usebasis);
    wkset->addScalarFields(scalars);
    
    wkset->createSolutionFields();
    
    // Create a function manager
    Teuchos::RCP<FunctionManager<AD> > functionManager = Teuchos::rcp(new FunctionManager<AD>("eblock",numElem,numip,numip));
    functionManager->wkset = wkset;
    
    //----------------------------------------------------------------------
    // Fill in some data
    //----------------------------------------------------------------------
    
    // Set the solution fields in the workset
    View_AD4 sol("sol", numElem, variables.size(), numip, 1);
    vector<AD> solvals = {1.0, 2.5, 3.3, -1.2, 13.2, 1.0};
    
    // This won't actually work on a GPU
    parallel_for("sol vals",
                 RangePolicy<AssemblyExec>(0,sol.extent(0)),
                 KOKKOS_LAMBDA (const size_type elem ) {
      for (size_type var=0; var<sol.extent(1); ++var) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          sol(elem,var,pt,0) = solvals[var];
        }
      }
    });
    
    wkset->setSolution(sol);
    
    // Set the scalar fields in the workset
    View_Sc2 xip("int pts",numElem,numip);
    View_Sc2 yip("int pts",numElem,numip);
    Kokkos::Random_XorShift64_Pool<> rand_pool(1979);
    Kokkos::fill_random(xip,rand_pool,0.0,1.0);
    Kokkos::fill_random(yip,rand_pool,0.0,1.0);
  
    wkset->setScalarField(xip,"x");
    wkset->setScalarField(yip,"y");
    
    //----------------------------------------------------------------------
    // Add some functions to test
    //----------------------------------------------------------------------
    
    vector<string> ref_names, ref_funcs;
    vector<View_AD2> ref_vals;
    
    auto a = wkset->getSolutionField("a",false);
    auto b = wkset->getSolutionField("b",false);
    auto c = wkset->getSolutionField("c",false);
    auto x = wkset->getScalarField("x");
    auto y = wkset->getScalarField("y");
    auto gradax = wkset->getSolutionField("grad(a)[x]",false);
    KokkosTools::print(a);
    
    {
      string name = "test1";
      string test = "sin(a+b+c)";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = sin(a(elem,pt)+b(elem,pt) + c(elem,pt));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "test2";
      string test = "a+exp(b)";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = a(elem,pt) + exp(b(elem,pt));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "test3";
      string test = "8*(pi^2)*sin(2*pi*x+1)*sin(2*pi*y+1)";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = 8*PI*PI*sin(2.0*PI*x(elem,pt)+1.0)*sin(2.0*PI*y(elem,pt)+1.0);
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "test4";
      string test = "-exp(x)";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = -exp(x(elem,pt));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    
    {
      string name = "test5";
      string test = "(a-sin(x))^(2+b)";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = std::pow(a(elem,pt)-sin(x(elem,pt)),2.0+b(elem,pt));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "test6";
      string test = "(a+2.0)*(b-pi)";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = (a(elem,pt)+2.0)*(b(elem,pt)-PI);
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "test7";
      string test = "-a";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = -a(elem,pt);
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "test8";
      string test = "(a+b) + ((x+y)*a - 2.0)";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = (a(elem,pt)+b(elem,pt)) + ((x(elem,pt)+y(elem,pt))*a(elem,pt) - 2.0);
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "test9";
      string test = "exp(-(a+b)^2)";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = exp(-pow(a(elem,pt)+b(elem,pt),2.0));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "testten";
      string test = "sin(gtst)";
      string g = "a+b";
      functionManager->addFunction("gtst",g,"ip");
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = sin(a(elem,pt)+b(elem,pt));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "pisq";
      string test = "8*pi^2";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = 8.0*PI*PI;
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "min";
      string test = "min(a,b)";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = min(a(elem,pt),b(elem,pt));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }

    {
      string name = "leq";
      string test = "a <= b";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          if (a(elem,pt) <= b(elem,pt)) {
            ref(elem,pt) = 1.0;
          }
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    
    {
      string name = "testgradsq";
      string test = "grad(a)[x]";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = 1.0+gradax(elem,pt);//*gradax(elem,pt);
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }
    

    {
      string name = "SW MS 1";
      string test = "-3.0*exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c))*cos(3.0*c) - sin(a - 4.0*c) + cos(b + 4.0*c)" ;
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = -3.0*exp(sin(3.0*a(elem,pt))*sin(3.0*b(elem,pt)) - sin(3.0*c(elem,pt)))*cos(3.0*c(elem,pt)) 
                          - sin(a(elem,pt) - 4.0*c(elem,pt)) + cos(b(elem,pt) + 4.0*c(elem,pt));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);

    }

    {
      // FIRST LINE OK
      // SECOND LINE OK
      // For some reason, the 2sin(3C) in the 6exp term seems to be the one that is breaking this...
      // What is going on?
      string name = "SW MS 2";
      string test = "(-0.75*(sin(3.0*a)*sin(b + 4.0*c)*cos(3.0*b) + sin(3.0*b)*cos(3.0*a)*cos(a - 4.0*c))*(exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c)) + 2.0)"
      "*exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c))*cos(a - 4.0*c) + (0.5*exp(sin(3.0*a)*sin(3.0*b)) + 1.0*exp(sin(3.0*c)))^3"
      "*(12.0*exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c))*sin(3.0*b)*cos(3.0*a) + 6.0*exp(2*sin(3.0*a)*sin(3.0*b) - 2.*sin(3.0*c))*sin(3.0*b)*cos(3.0*a) + 8.0*sin(a - 4.0*c))*exp(-3*sin(3.0*c))"
      " + (0.5*exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c)) + 1)^2*(-2*sin(a - 4.0*c) + cos(b + 4.0*c))*cos(a - 4.0*c))"
      "/((0.5*exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c)) + 1)^2*(exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c)) + 2.0))";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        AD A,B,C;
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          A = a(elem,pt); B = b(elem,pt); C = c(elem,pt);
          ref(elem,pt) = (-0.75*(sin(3.0*A)*sin(B + 4.0*C)*cos(3.0*B) + sin(3.0*B)*cos(3.0*A)*cos(A - 4.0*C))*(exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C)) + 2.0)
            *exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C))*cos(A - 4.0*C) + pow(0.5*exp(sin(3.0*A)*sin(3.0*B)) + 1.0*exp(sin(3.0*C)),3)
            *(12.0*exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C))*sin(3.0*B)*cos(3.0*A) + 6.0*exp(2*sin(3.0*A)*sin(3.0*B) - 2.*sin(3.0*C))*sin(3.0*B)*cos(3.0*A) + 8.0*sin(A - 4.0*C))*exp(-3*sin(3.0*C))
            + pow(0.5*exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C)) + 1,2)*(-2*sin(A - 4.0*C) + cos(B + 4.0*C))*cos(A - 4.0*C))
            /(pow(0.5*exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C)) + 1,2)*(exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C)) + 2.0));

        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);

    }

    {
      string name = "SW MS 3";
      string test = "(-0.75*(sin(3.0*a)*sin(b + 4.0*c)*cos(3.0*b) + sin(3.0*b)*cos(3.0*a)*cos(a - 4.0*c))*(exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c)) + 2.0)"
      "*exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c))*sin(b + 4.0*c) + (0.5*exp(sin(3.0*a)*sin(3.0*b)) + 1.0*exp(sin(3.0*c)))^3"
      "*(12.0*exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c))*sin(3.0*a)*cos(3.0*b) + 6.0*exp(2*sin(3.0*a)*sin(3.0*b) - 2*sin(3.0*c))*sin(3.0*a)*cos(3.0*b) + 8.0*cos(b + 4.0*c))*exp(-3*sin(3.0*c))"
      " + (0.5*exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c)) + 1)^2*(-sin(a - 4.0*c) + 2*cos(b + 4.0*c))*sin(b + 4.0*c))"
      "/((0.5*exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c)) + 1)^2*(exp(sin(3.0*a)*sin(3.0*b) - sin(3.0*c)) + 2.0))";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        AD A,B,C;
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          A = a(elem,pt); B = b(elem,pt); C = c(elem,pt);
          ref(elem,pt) = (-0.75*(sin(3.0*A)*sin(B + 4.0*C)*cos(3.0*B) + sin(3.0*B)*cos(3.0*A)*cos(A - 4.0*C))*(exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C)) + 2.0)
            *exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C))*sin(B + 4.0*C) + pow(0.5*exp(sin(3.0*A)*sin(3.0*B)) + 1.0*exp(sin(3.0*C)),3)
            *(12.0*exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C))*sin(3.0*A)*cos(3.0*B) + 6.0*exp(2*sin(3.0*A)*sin(3.0*B) - 2*sin(3.0*C))*sin(3.0*A)*cos(3.0*B) + 8.0*cos(B + 4.0*C))*exp(-3*sin(3.0*C)) 
            + pow(0.5*exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C)) + 1,2)*(-sin(A - 4.0*C) + 2*cos(B + 4.0*C))*sin(B + 4.0*C))
            /(pow(0.5*exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C)) + 1,2)*(exp(sin(3.0*A)*sin(3.0*B) - sin(3.0*C)) + 2.0));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);

    }

    {
      string name = "test sinh";
      string test = "(1+exp(-2.0*Ha))/(2.0*exp(-1.0*Ha))";
      functionManager->addFunction(name,test,"ip");
      
      View_AD2 ref("ref soln",numElem,numip);
      parallel_for("sol vals",
                   RangePolicy<AssemblyExec>(0,sol.extent(0)),
                   KOKKOS_LAMBDA (const size_type elem ) {
        for (size_type pt=0; pt<sol.extent(2); ++pt) {
          ref(elem,pt) = (1.0+exp(-2))/(2*exp(-1));
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }

    //----------------------------------------------------------------------
    // Make sure everything is defined properly and setup decompositions
    //----------------------------------------------------------------------
    
    functionManager->decomposeFunctions();
    functionManager->printFunctions();
    
    //----------------------------------------------------------------------
    // Evaluate the functions and check against ref solutions
    //----------------------------------------------------------------------
    
    for (size_t tst=0; tst<ref_names.size(); ++tst) {
      auto vals = functionManager->evaluate(ref_names[tst],"ip");
      double err = 0.0, normref = 0.0, normtst = 0.0;
      auto refsol = ref_vals[tst];
      for (size_type elem=0; elem<refsol.extent(0); ++elem) {
        for (size_type pt=0; pt<refsol.extent(1); ++pt) {
#ifndef MrHyDE_NO_AD
          normref += abs(refsol(elem,pt).val());
          normtst += abs(vals(elem,pt).val());
          err += abs(refsol(elem,pt).val() - vals(elem,pt).val());
#else
          normref += abs(refsol(elem,pt));
          normtst += abs(vals(elem,pt));
          err += abs(refsol(elem,pt) - vals(elem,pt));
#endif
        }
      }
      cout << endl;
      cout << "---------------------------------------------------" << endl;
      cout << ref_names[tst] << ": " << ref_funcs[tst] << endl;
      cout << "Norm of ref: " << normref <<endl;
      cout << "Norm of test: " << normtst <<endl;
      cout << "Error: " << err <<endl;
      cout << "---------------------------------------------------" << endl;
    }
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


