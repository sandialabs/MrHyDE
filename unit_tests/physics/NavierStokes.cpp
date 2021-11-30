
#include "trilinos.hpp"
#include "preferences.hpp"
#include "navierstokes.hpp"

using namespace std;
using namespace MrHyDE;

int main(int argc, char * argv[]) {
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  MpiComm Comm(MPI_COMM_WORLD);
  
  Kokkos::initialize();

  // The rest of this code should be scope-guarded for Kokkos to work properly
  {
    int dimension = 3;
    Teuchos::ParameterList settings;
    settings.set<bool>("useSUPG",true);
    settings.set<bool>("usePSPG",true);
    
    Teuchos::RCP<navierstokes> module = Teuchos::rcp(new navierstokes(settings, dimension));
  }
  
  Kokkos::finalize();
  
  
  int val = 0;
  return val;
}


