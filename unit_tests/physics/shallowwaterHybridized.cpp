
#include "trilinos.hpp"
#include "preferences.hpp"
#include "shallowwaterHybridized.hpp"
#include "workset.hpp"
#include "discretizationInterface.hpp"

using namespace std;
using namespace MrHyDE;

Teuchos::RCP<FunctionManager> setupDummyFunctionManager(int & dimension);
bool testEVDecomp1D(Teuchos::RCP<shallowwaterHybridized> module);
bool testEVDecomp2D(Teuchos::RCP<shallowwaterHybridized> module);
bool testMatVec(Teuchos::RCP<shallowwaterHybridized> module);
bool testStabTerm(Teuchos::RCP<shallowwaterHybridized> module);
bool checkClose(View_AD3 & truth, View_AD3 & test);
bool checkClose(View_AD2 & truth, View_AD2 & test);
bool checkClose(View_AD1 & truth, View_AD1 & test);

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  // The rest of this code should be scope-guarded for Kokkos to work properly
  {
    bool success;

    int dimension = 2; // used partially as a dummy, but does need to be set
    // correctly for residual/stabilization calculations
    // Properties and reference params which would normally be read in
    ScalarT g = 9.81;

    Teuchos::ParameterList settings;

    settings.set<bool>("Roe-like stabilization",true); // We must pick a stabilization method
    settings.set<ScalarT>("g",9.81);
    
    Teuchos::RCP<shallowwaterHybridized> module = Teuchos::rcp(new shallowwaterHybridized(settings, dimension));

    // create a dummy function manager with workset and assign to the module
    auto fm = setupDummyFunctionManager(dimension);
    module->setWorkset(fm->wkset);
    module->defineFunctions(settings,fm);

    // Test mat vec product
    success = testMatVec(module);

    TEUCHOS_TEST_FOR_EXCEPTION(!success,std::runtime_error,"shallowwaterHybridized::matVec() unit test fail!");

    // Test eigendecomposition routines
    // TODO :: these tests only cover the expressions are correct, not the eigendecomp directly
    // To some extent this is checked in the Jupyter notebook
    success = testEVDecomp1D(module);

    TEUCHOS_TEST_FOR_EXCEPTION(!success,std::runtime_error,"shallowwaterHybridized::eigendecompFluxJacobian() unit test fail (1D)!");

    success = testEVDecomp2D(module);

    TEUCHOS_TEST_FOR_EXCEPTION(!success,std::runtime_error,"shallowwaterHybridized::eigendecompFluxJacobian() unit test fail (2D)!");

    success = testStabTerm(module);
    
    TEUCHOS_TEST_FOR_EXCEPTION(!success,std::runtime_error,"shallowwaterHybridized::computeStabilizationTerm() unit test fail (2D)!");

  }
  
  Kokkos::finalize();

  int val = 0;
  return val;
}

Teuchos::RCP<FunctionManager> setupDummyFunctionManager(int & dimension) {

  // Modified from test_functions for now...
  
  Teuchos::RCP<workset> wkset = Teuchos::rcp( new workset() );
  
  // Define some parameters
  int numElem = 10;
  // TODO can I do add aux?
  // Trying to test some of the residual calcs...
  // Still think that I'll need offsets to be set up, so may be moot.
  vector<string> variables = {"H","Hux","Huy"};
  vector<string> scalars = {"x","y","z","n[x]","n[y]"};
  int numip = 4;
  vector<string> btypes = {"HGRAD"};
  vector<int> usebasis = {0,0,0};
  vector<vector<size_t>> uvals_index;
  vector<size_t> u_ind = {0,1,2};
  uvals_index.push_back(u_ind);
  Kokkos::View<int**,AssemblyDevice> aoffs("offsets",3,4); // vars x size

  // Set necessary parameters in workset
  wkset->usebasis = usebasis;
  wkset->maxElem = numElem;
  wkset->numElem = numElem;
  wkset->numip = numip;
  wkset->uvals_index = uvals_index; 

  wkset->isInitialized = true;
  wkset->addSolutionFields(variables, btypes, usebasis);
  wkset->varlist = variables;
  wkset->aux_varlist = variables; // trace unknowns are the state
  wkset->addScalarFields(scalars);
  wkset->addAux(variables,aoffs);

  // Create a function manager
  Teuchos::RCP<FunctionManager> functionManager = Teuchos::rcp(new FunctionManager("eblock",numElem,numip,numip));
  functionManager->wkset = wkset;

  //----------------------------------------------------------------------
  // Fill in some data
  //----------------------------------------------------------------------
  
  // Set the solution fields in the workset
  View_AD4 sol("sol", numElem, variables.size(), numip, 1);
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
  
  // now the aux vars
  vector<AD> auxvals = {4.0, -1.2, 8.1};
  
  // This won't actually work on a GPU
  parallel_for("aux vals",
               RangePolicy<AssemblyExec>(0,sol.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type var=0; var<sol.extent(1); ++var) {
      for (size_type pt=0; pt<sol.extent(2); ++pt) {
        sol(elem,var,pt,0) = auxvals[var];
      }
    }
  });

  wkset->setAux(sol);

  // Set the scalar fields in the workset
  View_Sc2 xip("int pts",numElem,numip);
  View_Sc2 yip("int pts",numElem,numip);
  View_Sc2 zip("int pts",numElem,numip);
  Kokkos::Random_XorShift64_Pool<> rand_pool(1979);
  Kokkos::fill_random(xip,rand_pool,0.0,1.0);
  Kokkos::fill_random(yip,rand_pool,0.0,1.0);
  Kokkos::fill_random(zip,rand_pool,0.0,1.0);
  
  wkset->setScalarField(xip,"x");
  wkset->setScalarField(yip,"y");
  wkset->setScalarField(zip,"z");

  return functionManager;
}

bool checkClose(View_AD3 & truth, View_AD3 & test) {

  ScalarT norm = 0.;
  for (size_t i=0; i<truth.extent(0); i++) {
    for (size_t j=0; j<truth.extent(1); j++) {
      for (size_t k=0; k<truth.extent(2); k++) {
        norm += (truth(i,j,k).val() - test(i,j,k).val())*
          (truth(i,j,k).val() - test(i,j,k).val());
      }
    }
  }

  if ( std::sqrt(norm > 1e-14) ) {
    return false;
  }
  return true;
}

bool checkClose(View_AD2 & truth, View_AD2 & test) {

  ScalarT norm = 0.;
  for (size_t i=0; i<truth.extent(0); i++) {
    for (size_t j=0; j<truth.extent(1); j++) {
      norm += (truth(i,j).val() - test(i,j).val())*(truth(i,j).val() - test(i,j).val());
    }
  }

  if ( std::sqrt(norm > 1e-14) ) {
    return false;
  }
  return true;
}

bool checkClose(View_AD1 & truth, View_AD1 & test) {

  ScalarT norm = 0.;
  for (size_t i=0; i<truth.extent(0); i++) {
    norm += (truth(i).val() - test(i).val())*(truth(i).val() - test(i).val());
  }

  if ( std::sqrt(norm > 1e-14) ) {
    return false;
  }
  return true;
}

bool testEVDecomp1D(Teuchos::RCP<shallowwaterHybridized> module){

  View_AD1 lam = View_AD1("eigenvalues",2);
  View_AD2 LEv = View_AD2("left eigenvectors", 2,2);
  View_AD2 REv = View_AD2("right eigenvectors",2,2);

  View_AD1 lamExact = View_AD1("eigenvalues exact",2);
  View_AD2 LEvExact = View_AD2("left eigenvectors exact", 2,2);
  View_AD2 REvExact = View_AD2("right eigenvectors exact",2,2);

  // the following inputs/output were generated with invFluxesSWE.ipynb

  AD Hux = -6.599787072453922; AD H = 4.240456895918983; 
  
  lamExact(0) = -8.00610538139873; lamExact(1) = 4.8933336992691165; 
  
  LEvExact(0,0) = 0.3793446884525907; LEvExact(1,0) = 0.6206553115474094; 
  LEvExact(0,1) = -0.07752275069841462; LEvExact(1,1) = 0.07752275069841462; 
  
  REvExact(0,0) = 1.0; REvExact(1,0) = -8.00610538139873; 
  REvExact(0,1) = 1.0; REvExact(1,1) = 4.8933336992691165; 

  module->eigendecompFluxJacobian(LEv,lam,REv,Hux,H);

  cout << "Checking eigenvalue calculation..." << endl;
  if (!checkClose(lamExact,lam)) return false;

  cout << "Checking right eigenvector calculation..." << endl;
  if (!checkClose(REvExact,REv)) return false;

  cout << "Checking left eigenvector calculation..." << endl;
  if (!checkClose(LEvExact,LEv)) return false;

  cout << "PASS :: testEVDecomp1D" << endl;
  return true;

}
  
bool testEVDecomp2D(Teuchos::RCP<shallowwaterHybridized> module){

  View_AD1 lam = View_AD1("eigenvalues",3);
  View_AD2 LEv = View_AD2("left eigenvectors", 3,3);
  View_AD2 REv = View_AD2("right eigenvectors",3,3);

  View_AD1 lamExact = View_AD1("eigenvalues exact",3);
  View_AD2 LEvExact = View_AD2("left eigenvectors exact", 3,3);
  View_AD2 REvExact = View_AD2("right eigenvectors exact",3,3);

  // the following inputs/output were generated with invFluxesSWE.ipynb
  //
  AD Hux = -39.22846164587605; AD H = 4.039388075475017; 
  AD Huy = 34.522032232062756; ScalarT nx = -0.3714635145497569; ScalarT ny = -0.9284475522927198;
  
  lamExact(0) = 1.9675733580306787; lamExact(1) = -4.327376762535005; lamExact(2) = -10.622326883100689; 
  
  LEvExact(0,0) = 0.8437181136985826; LEvExact(1,0) = 1.936673574640384; LEvExact(2,0) = 0.1562818863014173; 
  LEvExact(0,1) = -0.029504881487160697; LEvExact(1,1) = 0.14749085131897544; LEvExact(2,1) = 0.029504881487160697; 
  LEvExact(0,2) = -0.07374542565948772; LEvExact(1,2) = -0.059009762974321395; LEvExact(2,2) = 0.07374542565948772; 
  
  REvExact(0,0) = 1.0; REvExact(1,0) = -12.049830519084452; REvExact(2,0) = 2.7018209376747606; 
  REvExact(0,1) = 0.0; REvExact(1,1) = 5.844531031243971; REvExact(2,1) = -2.338344295700745; 
  REvExact(0,2) = 1.0; REvExact(1,2) = -7.373141927682962; REvExact(2,2) = 14.390883000162702;

  module->eigendecompFluxJacobian(LEv,lam,REv,Hux,Huy,H,nx,ny);

  cout << "Checking eigenvalue calculation..." << endl;
  if (!checkClose(lamExact,lam)) return false;

  cout << "Checking right eigenvector calculation..." << endl;
  if (!checkClose(REvExact,REv)) return false;

  cout << "Checking left eigenvector calculation..." << endl;
  if (!checkClose(LEvExact,LEv)) return false;

  cout << "PASS :: testEVDecomp2D" << endl;
  return true;

}

bool testStabTerm(Teuchos::RCP<shallowwaterHybridized> module) {

  auto wkset = module->wkset;

  // Set the solution fields in the workset
  View_AD4 sol("sol", wkset->numElem, wkset->varlist.size(), wkset->numip, 1);
  vector<AD> solvals = {14.05058372085475,-47.548104009458434,-3.665477736805561};

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
  
  // now the aux vars
  vector<AD> auxvals = {4.260916195707914,-2.938711226880664,12.295799778378594};
 
  // This won't actually work on a GPU
  parallel_for("aux vals",
               RangePolicy<AssemblyExec>(0,sol.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type var=0; var<sol.extent(1); ++var) {
      for (size_type pt=0; pt<sol.extent(2); ++pt) {
        sol(elem,var,pt,0) = auxvals[var];
      }
    }
  });

  wkset->setAux(sol);

  // Set the scalar fields in the workset
  View_Sc2 nx("int pts",wkset->numElem,wkset->numip);
  View_Sc2 ny("int pts",wkset->numElem,wkset->numip);
  ScalarT nxVal = 0.5586803483930811; ScalarT nyVal = -0.8293830648858135;

  parallel_for("normals",
               RangePolicy<AssemblyExec>(0,nx.extent(0)),
               KOKKOS_LAMBDA (const size_type elem) {
    for (size_type pt=0; pt<nx.extent(1); ++pt) {
      nx(elem,pt) = nxVal;
      ny(elem,pt) = nyVal;
    }
  });
  
  wkset->setScalarField(nx,"n[x]");
  wkset->setScalarField(ny,"n[y]");

  View_AD3 stabExact = View_AD3("stabExact",wkset->numElem,wkset->numip,wkset->varlist.size()); 
  vector<AD> stabExactVals = {56.623325001331374,-210.5408037820637,138.96559181837443};

  parallel_for("stabexact",
               RangePolicy<AssemblyExec>(0,stabExact.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type pt=0; pt<stabExact.extent(1); ++pt) {
      for (size_type var=0; var<stabExact.extent(2); ++var) {
        stabExact(elem,pt,var) = stabExactVals[var];
      }
    }
  });

  module->computeStabilizationTerm();

  cout << "Checking stabilization... (Roe-like)" << endl;
  if (!checkClose(stabExact,module->stab_bound_side)) return false;

  // now check max-EV

  module->roestab = false;
  module->maxEVstab = true;

  stabExactVals = {90.49511309124402,-412.36661351587117,-147.54511428575358};
  parallel_for("stabexact",
               RangePolicy<AssemblyExec>(0,stabExact.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type pt=0; pt<stabExact.extent(1); ++pt) {
      for (size_type var=0; var<stabExact.extent(2); ++var) {
        stabExact(elem,pt,var) = stabExactVals[var];
      }
    }
  });

  module->computeStabilizationTerm();

  cout << "Checking stabilization... (max EV)" << endl;
  if (!checkClose(stabExact,module->stab_bound_side)) return false;

  // change back JIC

  module->roestab = true;
  module->maxEVstab = false;

  cout << "PASS :: testStabTerm" << endl;
  return true;
}
  
bool testMatVec(Teuchos::RCP<shallowwaterHybridized> module) {

  View_AD2 A = View_AD2("A_mat",6,6);
  View_AD1 x = View_AD1("x",6);
  View_AD1 y = View_AD1("y",6);
  View_AD1 yExact = View_AD1("yExact",6);

  // Create a matrix about the size we'll be dealing with
  A(0,0) =   3.; A(0,1) =  -1.; A(0,2) = -10.; A(0,3) = -4.; A(0,4) =  5.;  A(0,5) =   5.;
  A(1,0) = -10.; A(1,1) =  -9.; A(1,2) =   7.; A(1,3) = -6.; A(1,4) = -10.; A(1,5) =   3.;
  A(2,0) =  -1.; A(2,1) = -10.; A(2,2) =   7.; A(2,3) =  2.; A(2,4) = -3.;  A(2,5) =   0.;
  A(3,0) =  -4.; A(3,1) =   2.; A(3,2) =   8.; A(3,3) =  1.; A(3,4) = -6.;  A(3,5) =  -9.;
  A(4,0) =  -8.; A(4,1) =  -6.; A(4,2) =   4.; A(4,3) = -9.; A(4,4) =  9.;  A(4,5) = -10.;
  A(5,0) =   5.; A(5,1) =  -8.; A(5,2) =  -5.; A(5,3) =  8.; A(5,4) =  5.;  A(5,5) =  -5.;

  x(0) = -5.; x(1) = -9.; x(2) = -6.; x(3) = -6.; x(4) = -1.; x(5) = 2.;

  yExact(0) = 83.; yExact(1) = 141.; yExact(2) = 44.; yExact(3) = -64.; yExact(4) = 95.; yExact(5) = 14.;

  module->matVec(A,x,y);

  cout << "Checking matrix vector product..." << endl;
  if (!checkClose(yExact,y)) return false;

  cout << "PASS :: testMatVec" << endl;
  return true;

}
