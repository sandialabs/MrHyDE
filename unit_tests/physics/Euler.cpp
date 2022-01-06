
#include "trilinos.hpp"
#include "preferences.hpp"
#include "euler.hpp"
#include "workset.hpp"
#include "discretizationInterface.hpp"

using namespace std;
using namespace MrHyDE;

Teuchos::RCP<FunctionManager> setupDummyFunctionManager(int & dimension);
bool testEVDecomp1D(Teuchos::RCP<euler> module);
bool testEVDecomp2D(Teuchos::RCP<euler> module);
bool testEVDecomp3D(Teuchos::RCP<euler> module);
bool testMatVec(Teuchos::RCP<euler> module);
bool testThermoProps(Teuchos::RCP<euler> module, vector<ScalarT> & props);
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

    int dimension = 3; // used partially as a dummy
    // Properties and reference params which would normally be read in
    ScalarT cp = 1004.5;
    ScalarT gamma = 1.4;
    ScalarT RGas = 287.0;
    ScalarT URef = 5.0;
    ScalarT TRef = 293.0;
    ScalarT rhoRef = 1.1;

    vector<ScalarT> props = {cp,gamma,RGas,URef,TRef,rhoRef};

    Teuchos::ParameterList settings;

    settings.set<bool>("Roe-like stabilization",true); // We must pick a stabilization method
    settings.set<ScalarT>("cp",cp);
    settings.set<ScalarT>("gamma",gamma);
    settings.set<ScalarT>("RGas",RGas);
    settings.set<ScalarT>("URef",URef);
    settings.set<ScalarT>("TRef",TRef);
    settings.set<ScalarT>("rhoRef",rhoRef);
    
    Teuchos::RCP<euler> module = Teuchos::rcp(new euler(settings, dimension));

    // create a dummy function manager with workset and assign to the module
    auto fm = setupDummyFunctionManager(dimension);
    module->setWorkset(fm->wkset);
    module->defineFunctions(settings,fm);

    // Test mat vec product
    success = testMatVec(module);

    TEUCHOS_TEST_FOR_EXCEPTION(!success,std::runtime_error,"Euler::matVec() unit test fail!");

    // Test eigendecomposition routines
    // TODO :: these tests only cover the expressions from Rohde are correct, not the eigendecomp directly
    success = testEVDecomp1D(module);

    TEUCHOS_TEST_FOR_EXCEPTION(!success,std::runtime_error,"Euler::eigendecompFluxJacobian() unit test fail (1D)!");

    success = testEVDecomp2D(module);

    TEUCHOS_TEST_FOR_EXCEPTION(!success,std::runtime_error,"Euler::eigendecompFluxJacobian() unit test fail (2D)!");

    success = testEVDecomp3D(module);

    TEUCHOS_TEST_FOR_EXCEPTION(!success,std::runtime_error,"Euler::eigendecompFluxJacobian() unit test fail (3D)!");

    success = testThermoProps(module,props);

    TEUCHOS_TEST_FOR_EXCEPTION(!success,std::runtime_error,"Euler::computeThermoProps() unit test fail!")

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
  vector<string> variables = {"rho","rhoux","rhouy","rhouz","rhoE"};
  vector<string> scalars = {"x","y","z"};
  int numip = 4;
  vector<string> btypes = {"HGRAD"};
  vector<int> usebasis = {0,0,0,0,0};
  vector<vector<size_t>> uvals_index;
  vector<size_t> u_ind = {0,1,2,3,4};
  uvals_index.push_back(u_ind);

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

bool testEVDecomp1D(Teuchos::RCP<euler> module){

  View_AD1 lam = View_AD1("eigenvalues",3);
  View_AD2 LEv = View_AD2("left eigenvectors", 3,3);
  View_AD2 REv = View_AD2("right eigenvectors",3,3);

  View_AD1 lamExact = View_AD1("eigenvalues exact",3);
  View_AD2 LEvExact = View_AD2("left eigenvectors exact", 3,3);
  View_AD2 REvExact = View_AD2("right eigenvectors exact",3,3);

  // the following inputs/output were generated with invFluxesEuler.ipynb

  AD rhoux = -66.17085268420277; AD rho = 8.93795147280951; AD a = 5.0; ScalarT gam = 1.4;
  
  lamExact(0) = -12.40335779238718; lamExact(1) = -7.4033577923871805; lamExact(2) = -2.4033577923871805; 
  
  LEvExact(0,0) = -0.5210969528303182; LEvExact(1,0) = 0.5615223471832003; LEvExact(2,0) = 0.9595746056471179; 
  LEvExact(0,1) = -0.07630925506436104; LEvExact(1,1) = -0.11845372467819486; LEvExact(2,1) = 0.12369074493563897; 
  LEvExact(0,2) = 0.007999999999999998; LEvExact(1,2) = -0.015999999999999997; LEvExact(2,2) = 0.007999999999999998; 
  
  REvExact(0,0) = 1.0; REvExact(1,0) = -12.40335779238718; REvExact(2,0) = 126.92164226298591; 
  REvExact(0,1) = 1.0; REvExact(1,1) = -7.4033577923871805; REvExact(2,1) = 27.404853301049993; 
  REvExact(0,2) = 1.0; REvExact(1,2) = -2.4033577923871805; REvExact(2,2) = 52.88806433911411; 

  module->eigendecompFluxJacobian(LEv,lam,REv,rhoux,rho,a,gam);

  cout << "Checking eigenvalue calculation..." << endl;
  if (!checkClose(lamExact,lam)) return false;

  cout << "Checking right eigenvector calculation..." << endl;
  if (!checkClose(REvExact,REv)) return false;

  cout << "Checking left eigenvector calculation..." << endl;
  if (!checkClose(LEvExact,LEv)) return false;

  cout << "PASS :: testEVDecomp1D" << endl;
  return true;

}
  
bool testEVDecomp2D(Teuchos::RCP<euler> module){

  View_AD1 lam = View_AD1("eigenvalues",4);
  View_AD2 LEv = View_AD2("left eigenvectors", 4,4);
  View_AD2 REv = View_AD2("right eigenvectors",4,4);

  View_AD1 lamExact = View_AD1("eigenvalues exact",4);
  View_AD2 LEvExact = View_AD2("left eigenvectors exact", 4,4);
  View_AD2 REvExact = View_AD2("right eigenvectors exact",4,4);

  // the following inputs/output were generated with invFluxesEuler.ipynb

  AD rhoux = -41.995507195881174; AD rho = 5.529853089643556; AD a = 5.0; ScalarT gam = 1.4;

  AD rhouy = -48.70870124510341; ScalarT nx = 0.47511518633154637; ScalarT ny = 0.8799236101600751;

  lamExact(0) = -16.358826076210935; lamExact(1) = -11.358826076210935; lamExact(2) = -6.358826076210935; lamExact(3) = -11.358826076210935; 

  LEvExact(0,0) = -0.5948416345419925; LEvExact(1,0) = -0.08208194615820205; LEvExact(2,0) = 1.6769235807001945; LEvExact(3,0) = 2.497461399134285; 
  LEvExact(0,1) = 0.01324308951299142; LEvExact(1,1) = -0.12150921629229211; LEvExact(2,1) = 0.10826612677930068; LEvExact(3,1) = 0.8799236101600751; 
  LEvExact(0,2) = -0.017525821734772355; LEvExact(1,2) = -0.1409330785624703; LEvExact(2,2) = 0.15845890029724266; LEvExact(3,2) = -0.47511518633154637; 
  LEvExact(0,3) = 0.007999999999999998; LEvExact(1,3) = -0.015999999999999997; LEvExact(2,3) = 0.007999999999999998; LEvExact(3,3) = 0.0; 

  REvExact(0,0) = 1.0; REvExact(1,0) = -9.96990194992599; REvExact(2,0) = -13.207935460954772; REvExact(3,0) = 186.92425201594233; 
  REvExact(0,1) = 1.0; REvExact(1,1) = -7.594326018268259; REvExact(2,1) = -8.808317410154396; REvExact(3,1) = 67.63012163488764; 
  REvExact(0,2) = 1.0; REvExact(1,2) = -5.218750086610527; REvExact(2,2) = -4.408699359354021; REvExact(3,2) = 73.33599125383297; 
  REvExact(0,3) = 0.0; REvExact(1,3) = 0.8799236101600751; REvExact(2,3) = -0.47511518633154637; REvExact(3,3) = -2.497461399134285; 

  module->eigendecompFluxJacobian(LEv,lam,REv,rhoux,rhouy,rho,nx,ny,a,gam);

  cout << "Checking eigenvalue calculation..." << endl;
  if (!checkClose(lamExact,lam)) return false;

  cout << "Checking right eigenvector calculation..." << endl;
  if (!checkClose(REvExact,REv)) return false;

  cout << "Checking left eigenvector calculation..." << endl;
  if (!checkClose(LEvExact,LEv)) return false;

  cout << "PASS :: testEVDecomp2D" << endl;
  return true;

}
  
bool testEVDecomp3D(Teuchos::RCP<euler> module){

  View_AD1 lam = View_AD1("eigenvalues",5);
  View_AD2 LEv = View_AD2("left eigenvectors", 5,5);
  View_AD2 REv = View_AD2("right eigenvectors",5,5);

  View_AD1 lamExact = View_AD1("eigenvalues exact",5);
  View_AD2 LEvExact = View_AD2("left eigenvectors exact", 5,5);
  View_AD2 REvExact = View_AD2("right eigenvectors exact",5,5);

  // Here we should at least check three cases (n_x biggest, n_y biggest, n_z biggest)
  // the following inputs/output were generated with invFluxesEuler.ipynb

  AD rhoux = -35.318949861307395; AD rho = 9.104665144973422; AD a = 5.0; ScalarT gam = 1.4;

  AD rhouy = -82.31728602153784; AD rhouz = -48.66546405161359; ScalarT nx = 0.761738089013404; ScalarT ny = 0.3806659965435591; ScalarT nz = -0.5242599382193022;

  lamExact(0) = -8.594402143434934; lamExact(1) = -3.594402143434934; lamExact(2) = 1.4055978565650662; lamExact(3) = -3.594402143434934; lamExact(4) = -3.594402143434934; 

  LEvExact(0,0) = 0.14200862043061663; LEvExact(1,0) = -0.0028976695482199944; LEvExact(2,0) = 0.8608890491176034; LEvExact(3,0) = -10.072955563077338; LEvExact(4,0) = 9.49081338142702; 
  LEvExact(0,1) = -0.04514009218673366; LEvExact(1,1) = -0.062067433429213476; LEvExact(2,1) = 0.10720752561594715; LEvExact(3,1) = 0.3806659965435591; LEvExact(4,1) = 0.5242599382193022; 
  LEvExact(0,2) = 0.03426316510759953; LEvExact(1,2) = -0.1446595295239109; LEvExact(2,2) = 0.11039636441631136; LEvExact(3,2) = -1.1225556545071915; LEvExact(4,2) = 0.2619902230287498; 
  LEvExact(0,3) = 0.0951868978435185; LEvExact(1,3) = -0.08552180804317656; LEvExact(2,3) = -0.009665089800341935; LEvExact(3,3) = -0.2619902230287498; LEvExact(4,3) = 0.9519696174278259; 
  LEvExact(0,4) = 0.007999999999999998; LEvExact(1,4) = -0.015999999999999997; LEvExact(2,4) = 0.007999999999999998; LEvExact(3,4) = 0.0; LEvExact(4,4) = 0.0; 

  REvExact(0,0) = 1.0; REvExact(1,0) = -7.687905034392863; REvExact(2,0) = -10.944550577962229; REvExact(3,0) = -2.7238133116020253; REvExact(4,0) = 143.15311506393846; 
  REvExact(0,1) = 1.0; REvExact(1,1) = -3.8792145893258434; REvExact(2,1) = -9.041220595244432; REvExact(3,1) = -5.345113002698536; REvExact(4,1) = 62.68110434676376; 
  REvExact(0,2) = 1.0; REvExact(1,2) = -0.07052414425882336; REvExact(2,2) = -7.137890612526637; REvExact(3,2) = -7.966412693795046; REvExact(4,2) = 107.2090936295891; 
  REvExact(0,3) = 0.0; REvExact(1,3) = 0.3806659965435591; REvExact(2,3) = -0.761738089013404; REvExact(3,3) = 0.0; REvExact(4,3) = 5.410357011118089; 
  REvExact(0,4) = 0.0; REvExact(1,4) = 0.5242599382193022; REvExact(2,4) = 0.0; REvExact(3,4) = 0.761738089013404; REvExact(4,4) = -6.105292965175663; 

  module->eigendecompFluxJacobian(LEv,lam,REv,rhoux,rhouy,rhouz,rho,nx,ny,nz,a,gam);

  cout << "|n_x| largest..." << endl;
  cout << "Checking eigenvalue calculation..." << endl;
  if (!checkClose(lamExact,lam)) return false;

  cout << "Checking right eigenvector calculation..." << endl;
  if (!checkClose(REvExact,REv)) return false;

  cout << "Checking left eigenvector calculation..." << endl;
  if (!checkClose(LEvExact,LEv)) return false;

  rhoux = 36.565224685177505; rho = 7.96463650944264; a = 5.0; gam = 1.4;
  
  rhouy = -52.093177117880686; rhouz = 64.00857644932476; nx = -0.4254264379064594; ny = 0.7440579055743148; nz = 0.5151603430802738;

  lamExact(0) = -7.679528854969166; lamExact(1) = -2.679528854969166; lamExact(2) = 2.320471145030834; lamExact(3) = -2.679528854969166; lamExact(4) = -2.679528854969166; 

  LEvExact(0,0) = 0.2458175428181773; LEvExact(1,0) = -0.02754085663018785; LEvExact(2,0) = 0.7817233138120105; LEvExact(3,0) = -4.638086139462035; LEvExact(4,0) = 12.656251961745985; 
  LEvExact(0,1) = 0.005815067191938051; LEvExact(1,1) = 0.07345515319741579; LEvExact(2,1) = -0.07927022038935383; LEvExact(3,1) = 1.1007373751348715; LEvExact(4,1) = -0.29455077093515397; 
  LEvExact(0,2) = -0.02208131643372675; LEvExact(1,2) = -0.10464894824740947; LEvExact(2,2) = 0.12673026468113624; LEvExact(3,2) = 0.4254264379064594; LEvExact(4,2) = 0.5151603430802738; 
  LEvExact(0,3) = -0.11580881289089234; LEvExact(1,3) = 0.1285855571657299; LEvExact(2,3) = -0.01277674427483757; LEvExact(3,3) = 0.29455077093515397; LEvExact(4,3) = -0.9873019497728371; 
  LEvExact(0,4) = 0.007999999999999998; LEvExact(1,4) = -0.015999999999999997; LEvExact(2,4) = 0.007999999999999998; LEvExact(3,4) = 0.0; LEvExact(4,4) = 0.0; 

  REvExact(0,0) = 1.0; REvExact(1,0) = 6.718079264370784; REvExact(2,0) = -10.260848793334667; REvExact(3,0) = 5.4607956074567525; REvExact(4,0) = 140.1189478142326; 
  REvExact(0,1) = 1.0; REvExact(1,1) = 4.590947074838487; REvExact(2,1) = -6.540559265463093; REvExact(3,1) = 8.036597322858121; REvExact(4,1) = 64.22130353938675; 
  REvExact(0,2) = 1.0; REvExact(1,2) = 2.4638148853061903; REvExact(2,2) = -2.820269737591519; REvExact(3,2) = 10.61239903825949; REvExact(4,2) = 113.32365926454094; 
  REvExact(0,3) = 0.0; REvExact(1,3) = 0.7440579055743148; REvExact(2,3) = 0.4254264379064594; REvExact(3,3) = 0.0; REvExact(4,3) = 0.6334036348847998; 
  REvExact(0,4) = 0.0; REvExact(1,4) = 0.0; REvExact(2,4) = 0.5151603430802738; REvExact(3,4) = -0.7440579055743148; REvExact(4,4) = -9.34913052712279; 

  module->eigendecompFluxJacobian(LEv,lam,REv,rhoux,rhouy,rhouz,rho,nx,ny,nz,a,gam);

  cout << "|n_y| largest..." << endl;
  cout << "Checking eigenvalue calculation..." << endl;
  if (!checkClose(lamExact,lam)) return false;

  cout << "Checking right eigenvector calculation..." << endl;
  if (!checkClose(REvExact,REv)) return false;

  cout << "Checking left eigenvector calculation..." << endl;
  if (!checkClose(LEvExact,LEv)) return false;

  rhoux = -52.48911557656745; rho = 8.973275634066631; a = 5.0; gam = 1.4;
  
  rhouy = -49.07363972214395; rhouz = -64.02380041979175; nx = -0.6435391026342729; ny = -0.05138386262247015; nz = -0.7636865338885254;

  lamExact(0) = 4.4942481654751045; lamExact(1) = 9.494248165475105; lamExact(2) = 14.494248165475105; lamExact(3) = 9.494248165475105; lamExact(4) = 9.494248165475105; 

  LEvExact(0,0) = 1.409554657866186; LEvExact(1,0) = 0.07974031736264905; LEvExact(2,0) = -0.489294975228835; LEvExact(3,0) = -0.34101250559584656; LEvExact(4,0) = -6.522328702983507; 
  LEvExact(0,1) = 0.11114985655291375; LEvExact(1,1) = -0.09359189257897291; LEvExact(2,1) = -0.01755796397394085; LEvExact(3,1) = 0.7671438442126463; LEvExact(4,1) = 0.043299866338580795; 
  LEvExact(0,2) = 0.048889312210259224; LEvExact(1,2) = -0.08750185189602443; LEvExact(2,2) = 0.0386125396857652; LEvExact(3,2) = -0.043299866338580795; LEvExact(4,2) = -1.3059804702639668; 
  LEvExact(0,3) = 0.13344818869409006; LEvExact(1,3) = -0.11415907061047503; LEvExact(2,3) = -0.019289118083615015; LEvExact(3,3) = -0.6435391026342729; LEvExact(4,3) = 0.05138386262247015; 
  LEvExact(0,4) = 0.007999999999999998; LEvExact(1,4) = -0.015999999999999997; LEvExact(2,4) = 0.007999999999999998; LEvExact(3,4) = 0.0; LEvExact(4,4) = 0.0; 

  REvExact(0,0) = 1.0; REvExact(1,0) = -2.6317977730144424; REvExact(2,0) = -5.211946430389177; REvExact(3,0) = -3.316509243712065; REvExact(4,0) = 72.54498933745894; 
  REvExact(0,1) = 1.0; REvExact(1,1) = -5.849493286185807; REvExact(2,1) = -5.468865743501528; REvExact(3,1) = -7.134941913154692; REvExact(4,1) = 57.51623016483445; 
  REvExact(0,2) = 1.0; REvExact(1,2) = -9.067188799357172; REvExact(2,2) = -5.725785056613879; REvExact(3,2) = -10.953374582597318; REvExact(4,2) = 167.48747099220998; 
  REvExact(0,3) = 0.0; REvExact(1,3) = 0.7636865338885254; REvExact(2,3) = 0.0; REvExact(3,3) = -0.6435391026342729; REvExact(4,3) = 0.12443486340779408; 
  REvExact(0,4) = 0.0; REvExact(1,4) = 0.0; REvExact(2,4) = -0.7636865338885254; REvExact(3,4) = 0.05138386262247015; REvExact(4,4) = 3.8098782488715295; 

  module->eigendecompFluxJacobian(LEv,lam,REv,rhoux,rhouy,rhouz,rho,nx,ny,nz,a,gam);

  cout << "|n_z| largest..." << endl;
  cout << "Checking eigenvalue calculation..." << endl;
  if (!checkClose(lamExact,lam)) return false;

  cout << "Checking right eigenvector calculation..." << endl;
  if (!checkClose(REvExact,REv)) return false;

  cout << "Checking left eigenvector calculation..." << endl;
  if (!checkClose(LEvExact,LEv)) return false;

  cout << "PASS :: testEVDecomp3D" << endl;
  return true;

}

bool testMatVec(Teuchos::RCP<euler> module) {

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

bool testThermoProps(Teuchos::RCP<euler> module, vector<ScalarT> & props) {

  auto wkset = module->wkset;
  auto cp = props[0];
  auto gamma = props[1];
  auto RGas = props[2];
  auto URef = props[3];
  auto TRef = props[4];
  auto rhoRef = props[5];

  // create dummy solutions
  // TODO hardcoded number of variables
  View_AD4 sol("sol",wkset->numElem,5,wkset->numip,1);

  // rhoE_ref = rhoRef*URef**2
  ScalarT rho = 2.;
  ScalarT rhoux = 1.5;
  ScalarT rhouy = -.5;
  ScalarT rhouz = 5.;
  ScalarT rhoE = 10.;
  ScalarT rhoD = rho*rhoRef;
  ScalarT rhouxD = rhoux*rhoRef*URef;
  ScalarT rhouyD = rhouy*rhoRef*URef;
  ScalarT rhouzD = rhouz*rhoRef*URef;
  ScalarT rhoED = rhoE*rhoRef*URef*URef;

  // code is non-dimensionalized
  vector<AD> solvals = {rho,rhoux,rhouy,rhouz,rhoE};

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

  // get exact thermoprops (will non-dim below)
  ScalarT KE = .5*(rhouxD*rhouxD + rhouyD*rhouyD + rhouzD*rhouzD)/rhoD;
  ScalarT p0 = (gamma-1.)*(rhoED - KE);
  ScalarT T = p0/(RGas*rhoD);
  ScalarT a = sqrt(gamma*RGas*T);

  p0 = p0/(rhoRef*URef*URef);
  T = T/TRef;
  a = a/URef;

  vector<AD> thermvals = {p0,T,a};

  module->computeThermoProps(false); // not on_side

  View_AD3 thermExact("thermExact",wkset->numElem,wkset->numip,3);
  // Storage here is slightly different from above
  parallel_for("sol vals",
               RangePolicy<AssemblyExec>(0,thermExact.extent(0)),
               KOKKOS_LAMBDA (const size_type elem ) {
    for (size_type pt=0; pt<thermExact.extent(1); ++pt) {
      for (size_type var=0; var<thermExact.extent(2); ++var) {
        thermExact(elem,pt,var) = thermvals[var];
      }
    }
  });

  cout << "Checking ThermoProps calculation..." << endl;
  if (!checkClose(thermExact,module->props_vol)) return false;
  
  cout << "PASS :: testThermoProps" << endl;
  return true;

}