
#include "trilinos.hpp"
#include "preferences.hpp"
#include "functionInterface.hpp"
#include "workset.hpp"
#include "discretizationTools.hpp"

using namespace std;

int main(int argc, char * argv[]) {

  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  LA_MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();
  
  Teuchos::RCP<FunctionInterface> functionManager = Teuchos::rcp(new FunctionInterface());
  
  vector<string> variables = {"a","b","c","d"};
  vector<string> parameters = {"mu"};
  vector<string> disc_parameters = {"ff"};
  
  functionManager->setupLists(variables, parameters, disc_parameters);
  
  topo_RCP cellTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ) );
  topo_RCP sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
  
  vector<basis_RCP> basis = {Teuchos::rcp(new Basis_HGRAD_QUAD_C1_FEM<AssemblyDevice>() )};
  int quadorder = 2;
  int numElem = 100;
  int numip = 4;
  int numvars = 4;
  
  vector<int> cellinfo = {2,numvars,1,0,16,numElem};
  DRV ip, wts, sip, swts;
  
  DiscTools::getQuadrature(cellTopo, quadorder, ip, wts);
  DiscTools::getQuadrature(sideTopo, quadorder, sip, swts);
  
  vector<string> btypes = {"HGRAD"};
  Kokkos::View<int**,AssemblyDevice> bcs("bcs",1,1);
  Teuchos::RCP<workset> wkset = Teuchos::rcp( new workset(cellinfo, ip, wts,
                                            sip, swts, btypes, basis, basis, cellTopo,bcs) );
  
  for (size_t i=0; i<numElem; i++) {
    for (size_t j=0; j<numip; j++) {
      for (size_t k=0; k<numvars; k++) {
        wkset->local_soln(i,j,k,0) = k+2;
      }
    }
  }
  
  functionManager->wkset = wkset;
  
  string test1 = "sin(a+b+c)";
  functionManager->addFunction("test1",test1,numElem,numip,"ip",0);
  
  string test2 = "a+exp(b)";
  functionManager->addFunction("test2",test2,numElem,numip,"ip",0);
  
  string test3 = "8*(pi^2)*sin(2*pi*x+1)*sin(2*pi*y+1)";
  functionManager->addFunction("g",test3,numElem,numip,"ip",0);
  
  functionManager->validateFunctions();
  functionManager->decomposeFunctions();
  
  FDATA data1 = functionManager->evaluate("test1","ip",0);
  FDATA data2 = functionManager->evaluate("test2","ip",0);
  FDATA data3 = functionManager->evaluate("g","ip",0);
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
          wkset->local_soln(i,j,k,0) = k+3+m;
        }
      }
    }
    
    data1 = functionManager->evaluate("test1","ip",0);
    data2 = functionManager->evaluate("test2","ip",0);
    data3 = functionManager->evaluate("g","ip",0);
  }
  Teuchos::TimeMonitor::summarize();
  
  Kokkos::finalize();
  
  
  return 0;
}


