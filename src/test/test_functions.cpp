
#include "trilinos.hpp"
#include "preferences.hpp"
#include "functionManager.hpp"
#include "workset.hpp"
#include "discretizationTools.hpp"

using namespace std;

int main(int argc, char * argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();
  
  Teuchos::RCP<FunctionManager> functionManager = Teuchos::rcp(new FunctionManager());
  Teuchos::RCP<DiscTools> discTools = Teuchos::rcp( new DiscTools() );
  vector<string> variables = {"a","b","c","d","p"};
  vector<string> parameters = {"mu"};
  vector<string> disc_parameters = {"ff"};
  
  functionManager->setupLists(variables, parameters, disc_parameters);
  
  topo_RCP cellTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ) );
  topo_RCP sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
  
  vector<basis_RCP> basis = {Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<AssemblyDevice>() )};
  int quadorder = 2;
  int numElem = 10;
  int numip = 4;
  int numvars = 5;
  
  vector<int> cellinfo = {2,numvars,1,0,16,numElem};
  DRV ip, wts, sip, swts;
  
  discTools->getQuadrature(cellTopo, quadorder, ip, wts);
  discTools->getQuadrature(sideTopo, quadorder, sip, swts);
  
  vector<string> btypes = {"HGRAD"};
  Kokkos::View<int**,AssemblyDevice> bcs("bcs",1,1);
  Teuchos::RCP<workset> wkset = Teuchos::rcp( new workset(cellinfo, ip, wts,
                                            sip, swts, btypes, basis, basis, cellTopo,bcs) );
  
  for (size_t i=0; i<numElem; i++) {
    for (size_t j=0; j<numip; j++) {
      for (size_t k=0; k<numvars; k++) {
        wkset->local_soln(i,j,k,0) = k+2;
      }
      for (size_t k=0; k<2; k++) {
        wkset->ip_KV(i,j,k) = k+2;
      }
    }
  }
  KokkosTools::print(wkset->ip);
  
  functionManager->wkset = wkset;
  
  string test1 = "sin(a+b+c)";
  functionManager->addFunction("test1",test1,numElem,numip,"ip",0);
  
  string test2 = "a+exp(b)";
  functionManager->addFunction("test2",test2,numElem,numip,"ip",0);
  
  string test3 = "8*(pi^2)*sin(2*pi*x+1)*sin(2*pi*y+1)";
  functionManager->addFunction("g",test3,numElem,numip,"ip",0);
  
  string bubble = "-0.001*(x*x+y*y)";
  string well = "100.0*exp(bubble)";
  string welll = "p*well";
  string wellr = "2.0*well";
  
  string source = "wellr - welll";
//  source = "0.5*p*p+2";
  functionManager->addFunction("pres","p",numElem,numip,"ip",0);
  
  functionManager->addFunction("bubble",bubble,numElem,numip,"ip",0);
  functionManager->addFunction("well",well,numElem,numip,"ip",0);
  functionManager->addFunction("welll",welll,numElem,numip,"ip",0);
  functionManager->addFunction("wellr",wellr,numElem,numip,"ip",0);
  functionManager->addFunction("source",source,numElem,numip,"ip",0);
  
  functionManager->validateFunctions();
  functionManager->decomposeFunctions();
  
  functionManager->printFunctions();
  
  FDATA datax = functionManager->evaluate("pres","ip",0);
  
  FDATA data1 = functionManager->evaluate("bubble","ip",0);
  FDATA data2 = functionManager->evaluate("well","ip",0);
  FDATA data3 = functionManager->evaluate("source","ip",0);
  
  for (size_t i=0; i<numElem; i++) {
    for (size_t j=0; j<numip; j++) {
      cout << "data1(i,j) = " << data1(i,j) << endl;
      cout << "data2(i,j) = " << data2(i,j) << endl;
      cout << "data3(i,j) = " << data3(i,j) << endl;
    }
  }
  
  for (int m=0; m<10; m++) {
    for (size_t i=0; i<numElem; i++) {
      for (size_t j=0; j<numip; j++) {
        for (size_t k=0; k<numvars; k++) {
          wkset->local_soln(i,k,j,0) = k+3+m;
        }
      }
    }
    FDATA datap = functionManager->evaluate("pres","ip",0);
    
    FDATA data1 = functionManager->evaluate("wellr","ip",0);
    FDATA data2 = functionManager->evaluate("welll","ip",0);
    FDATA data3 = functionManager->evaluate("source","ip",0);
    
    for (size_t i=0; i<numElem; i++) {
      for (size_t j=0; j<numip; j++) {
        cout << "datap(i,j) = " << datap(i,j) << endl;
        cout << "data1(i,j) = " << data1(i,j) << endl;
        cout << "data2(i,j) = " << data2(i,j) << endl;
        cout << "data3(i,j) = " << data3(i,j) << endl;
      }
    }
    
  }
  
  
  Teuchos::TimeMonitor::summarize();
  
  Kokkos::finalize();
  
  
  return 0;
}


