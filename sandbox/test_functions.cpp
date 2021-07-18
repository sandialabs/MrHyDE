
#include "trilinos.hpp"
#include "preferences.hpp"
#include "functionManager.hpp"
#include "workset.hpp"
#include "discretizationInterface.hpp"

#include <Kokkos_Random.hpp>

using namespace std;
using namespace MrHyDE;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  {
    //----------------------------------------------------------------------
    // Set up a dummy workset just to test interplay between function manager and workset
    //----------------------------------------------------------------------
    
    Teuchos::RCP<DiscretizationInterface> disc = Teuchos::rcp( new DiscretizationInterface() );
    
    topo_RCP cellTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ) );
    topo_RCP sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
    
    vector<basis_RCP> basis = {Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::Device::execution_space>() )};
    int quadorder = 2;
    int numElem = 10;
    int numvars = 5;
    vector<string> variables = {"a","b","c","d","p"};
    vector<string> aux_variables;
    vector<string> param_vars;
    
    vector<int> cellinfo = {2,numvars,1,0,numElem};
    DRV ip, wts, sip, swts;
    
    disc->getQuadrature(cellTopo, quadorder, ip, wts);
    disc->getQuadrature(sideTopo, quadorder, sip, swts);
    int numip = ip.extent(0);
    cellinfo.push_back(ip.extent(0));
    cellinfo.push_back(sip.extent(0));
    vector<string> btypes = {"HGRAD"};
    Kokkos::View<string**,HostDevice> bcs("bcs",1,1);
    Teuchos::RCP<workset> wkset = Teuchos::rcp( new workset(cellinfo, false,
                                                            btypes, basis, basis, cellTopo,bcs) );
    
    vector<int> usebasis = {0,0,0,0,0};
    wkset->usebasis = usebasis;
    wkset->varlist = variables;
    
    wkset->createSolns();
    
    Teuchos::RCP<FunctionManager> functionManager = Teuchos::rcp(new FunctionManager("eblock",numElem,numip,numip));
    vector<string> parameters;
    vector<string> disc_parameters;
    
    functionManager->setupLists(variables, aux_variables, parameters, disc_parameters);
    
    functionManager->wkset = wkset;
    
    View_AD4 sol("sol",numElem,numvars,numip,1);
    vector<AD> solvals = {1.0, 2.5, 3.3, -1.2, 13.2};
    
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
    
    View_Sc2 xip("int pts",numElem,numip);
    View_Sc2 yip("int pts",numElem,numip);
    Kokkos::Random_XorShift64_Pool<> rand_pool(1979);
    Kokkos::fill_random(xip,rand_pool,0.0,1.0);
    Kokkos::fill_random(yip,rand_pool,0.0,1.0);
    std::vector<View_Sc2> pip = {xip,yip};
    wkset->setIP(pip);
    
    //----------------------------------------------------------------------
    // Add some functions to test
    //----------------------------------------------------------------------
    
    vector<string> ref_names, ref_funcs;
    vector<View_AD2> ref_vals;
    
    auto a = wkset->getData("a");
    auto b = wkset->getData("b");
    auto c = wkset->getData("c");
    auto x = wkset->getDataSc("x");
    auto y = wkset->getDataSc("y");
    
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
          ref(elem,pt) = a(elem,pt) <= b(elem,pt);
        }
      });
      ref_names.push_back(name);
      ref_vals.push_back(ref);
      ref_funcs.push_back(test);
    }

    //----------------------------------------------------------------------
    // Make sure everything is defined properly and setup decompositions
    //----------------------------------------------------------------------
    
    //functionManager->validateFunctions();
    functionManager->decomposeFunctions();
    functionManager->printFunctions();
    
    for (size_t f=0; f<functionManager->forests[0].trees.size(); ++f) {
      functionManager->forests[0].trees[f].branches[0].print();
      if (functionManager->forests[0].trees[f].branches[0].isConstant) {
        cout << functionManager->forests[0].trees[f].branches[0].data_Sc << endl;
      }
    }
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


