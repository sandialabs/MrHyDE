
#include "trilinos.hpp"
#include "preferences.hpp"
#include "functionManager.hpp"
#include "workset.hpp"
#include "discretizationInterface.hpp"

using namespace std;
using namespace MrHyDE;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  typedef Kokkos::DynRankView<ScalarT,Kokkos::LayoutStride,AssemblyExec> DRVtst;
  
  {
    Teuchos::RCP<discretization> disc = Teuchos::rcp( new discretization() );
    
    topo_RCP cellTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ) );
    topo_RCP sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
    
    vector<basis_RCP> basis = {Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<AssemblyExec>() )};
    int quadorder = 2;
    int numElem = 10;
    int numip = 4;
    int numvars = 5;
    
    vector<int> cellinfo = {2,numvars,1,0,16,numElem};
    DRV ip, wts, sip, swts;
    //DRVtst tst("testing",numip,2,2,2);
 
    disc->getQuadrature(cellTopo, quadorder, ip, wts);
    disc->getQuadrature(sideTopo, quadorder, sip, swts);
    
    vector<string> btypes = {"HGRAD"};
    Kokkos::View<int**,HostDevice> bcs("bcs",1,1);
    Teuchos::RCP<workset> wkset = Teuchos::rcp( new workset(cellinfo, false, ip, wts,
                                                            sip, swts, btypes, basis, basis, cellTopo,bcs) );
    
    Teuchos::RCP<FunctionManager> functionManager = Teuchos::rcp(new FunctionManager("eblock",numElem,numip,numip));
    vector<string> variables = {"a","b","c","d","p"};
    vector<string> parameters = {"mu"};
    vector<string> disc_parameters = {"ff"};
    
    functionManager->setupLists(variables, parameters, disc_parameters);
    
    
    /*
     parallel_for(RangePolicy<AssemblyExec>(0,numElem), KOKKOS_LAMBDA (const int i ) {
     for (size_t j=0; j<numip; j++) {
     for (size_t k=0; k<numvars; k++) {
     wkset->local_soln(i,j,k,0) = k+2;
     }
     for (size_t k=0; k<2; k++) {
     wkset->ip_KV(i,j,k) = k+2;
     }
     }
     });
     */
    
    //KokkosTools::print(wkset->ip);
    
    
    functionManager->wkset = wkset;
    
    string test1 = "sin(a+b+c)";
    functionManager->addFunction("test1",test1,"ip");
    
    string test2 = "a+exp(b)";
    functionManager->addFunction("test2",test2,"ip");
    
    string test3 = "8*(pi^2)*sin(2*pi*x+1)*sin(2*pi*y+1)";
    functionManager->addFunction("g",test3,"ip");
    
    string bubble = "-0.001*(x*x+y*y)";
    string well = "100.0*exp(bubble)";
    string welll = "p*well";
    string wellr = "2.0*well";
    
    string source = "wellr - welll";
    //  source = "0.5*p*p+2";
    functionManager->addFunction("pres","p","ip");
    
    functionManager->addFunction("bubble",bubble,"ip");
    functionManager->addFunction("well",well,"ip");
    functionManager->addFunction("welll",welll,"ip");
    functionManager->addFunction("wellr",wellr,"ip");
    functionManager->addFunction("source",source,"ip");
    
    functionManager->validateFunctions();
    functionManager->decomposeFunctions();
    
    //functionManager->printFunctions();
    
    FDATA datax = functionManager->evaluate("pres","ip");
    
    FDATA data1 = functionManager->evaluate("bubble","ip");
    FDATA data2 = functionManager->evaluate("well","ip");
    FDATA data3 = functionManager->evaluate("source","ip");
    
    /*
     parallel_for(RangePolicy<AssemblyExec>(0,numElem), KOKKOS_LAMBDA (const int i ) {
     for (size_t j=0; j<numip; j++) {
     printf("data1(i,j) : %f\n", data1(i,j).val());
     printf("data2(i,j) : %f\n", data2(i,j).val());
     printf("data3(i,j) : %f\n", data3(i,j).val());
     //cout << "data1(i,j) = " << data1(i,j) << endl;
     //cout << "data2(i,j) = " << data2(i,j) << endl;
     //cout << "data3(i,j) = " << data3(i,j) << endl;
     }
     });
     */
    
    for (int m=0; m<10; m++) {
      /*
       parallel_for(RangePolicy<AssemblyExec>(0,numElem), KOKKOS_LAMBDA (const int i ) {
       for (size_t j=0; j<numip; j++) {
       for (size_t k=0; k<numvars; k++) {
       wkset->local_soln(i,k,j,0) = k+3; // +m (but m is not on device)
       }
       }
       });
       */
      FDATA datap = functionManager->evaluate("pres","ip");
      
      FDATA data1 = functionManager->evaluate("wellr","ip");
      FDATA data2 = functionManager->evaluate("welll","ip");
      FDATA data3 = functionManager->evaluate("source","ip");
      /*
       parallel_for(RangePolicy<AssemblyExec>(0,numElem), KOKKOS_LAMBDA (const int i ) {
       for (size_t j=0; j<numip; j++) {
       printf("datap(i,j) : %f\n", datap(i,j).val());
       printf("data1(i,j) : %f\n", data1(i,j).val());
       printf("data2(i,j) : %f\n", data2(i,j).val());
       printf("data3(i,j) : %f\n", data3(i,j).val());
       //cout << "datap(i,j) = " << datap(i,j) << endl;
       //cout << "data1(i,j) = " << data1(i,j) << endl;
       //cout << "data2(i,j) = " << data2(i,j) << endl;
       //cout << "data3(i,j) = " << data3(i,j) << endl;
       }
       });
       */
    }
    
    
    //Teuchos::TimeMonitor::summarize();
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


